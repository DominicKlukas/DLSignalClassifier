import numpy as np
import scipy.signal as sc
import h5py
from .modulation import MODS
from dataclasses import asdict, dataclass

@dataclass
class SignalConfig:
    modulation: str
    symbol_rate: np.float32
    signal_length: int
    phase_shift: np.float32
    frequency_offset: np.float32
    sps_int: int                    # Resolution of internal representation of signal
    SNR: np.float32
    sr_out: np.float32
    pulse_shaping_filter_num_taps: int

class SignalGenerator:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def gen_rrc(self, sps, num_taps):
        beta = 0.35
        Ts = float(sps)
        t = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) // 2
        x = t / Ts
        den = 1 - (2 * beta * x) ** 2
        den[np.isclose(den, 0.0)] = np.finfo(np.float64).eps
        h = np.sinc(x) * np.cos(np.pi * beta * x) / den
        return h

    def generate_batch(self, signal_length: int, num_signals: int, modulation_types: list,
                        symbol_rate_bnds: list, phase_shift_bnds: list, frequency_offset_bnds: list,
                        SNR_bnds: list, sr_out: int):
        # signal_length: final length of signal
        # num_singals: number of signals to add to the batch
        # modulation_types: list of strings, each is a modulation type in the MODS dict defined in Modulation.py
        if signal_length <= 0:
            raise ValueError("signal_length must be positive")
        if num_signals <= 0:
            raise ValueError("num_signals must be positive")
        if sr_out <= 0:
            raise ValueError("sr_out must be positive")
        if not modulation_types:
            raise ValueError("modulation_types must not be empty")
        if symbol_rate_bnds[0] <= 0 or symbol_rate_bnds[0] > symbol_rate_bnds[1]:
            raise ValueError("symbol_rate_bnds must be positive and ordered [min, max]")
        if phase_shift_bnds[0] > phase_shift_bnds[1]:
            raise ValueError("phase_shift_bnds must be ordered [min, max]")
        if frequency_offset_bnds[0] > frequency_offset_bnds[1]:
            raise ValueError("frequency_offset_bnds must be ordered [min, max]")
        if SNR_bnds[0] > SNR_bnds[1]:
            raise ValueError("SNR_bnds must be ordered [min, max]")

        signal_cnfgs = []
        for _ in range(num_signals):
            modulation = self.rng.choice(modulation_types)
            symbol_rate = self.rng.uniform(symbol_rate_bnds[0], symbol_rate_bnds[1])
            phase_shift = self.rng.uniform(phase_shift_bnds[0], phase_shift_bnds[1])
            frequency_offset = self.rng.uniform(frequency_offset_bnds[0], frequency_offset_bnds[1])
            SNR = self.rng.uniform(SNR_bnds[0], SNR_bnds[1])

            # Choose a practical oversampling factor from the output-rate/symbol-rate ratio.
            # Keep it bounded so generation cost stays predictable across the batch.
            sps_from_ratio = int(np.ceil(sr_out / symbol_rate))
            sps_int = int(np.clip(sps_from_ratio, 4, 8))

            # Use an RRC span of 10 symbols, which is a common practical default.
            # Taps scale with oversampling, not final signal length.
            pulse_shaping_filter_num_taps = 10 * sps_int + 1

            cnfg = SignalConfig(modulation, symbol_rate, signal_length, phase_shift, frequency_offset,
                                sps_int, SNR, sr_out, pulse_shaping_filter_num_taps)

            signal_cnfgs.append(cnfg)
        signals = []
        for cnfg in signal_cnfgs:
            signals.append(self.generate_signal(cnfg))
        metadata = [asdict(cnfg) for cnfg in signal_cnfgs]
        return signals, metadata

    def save_batch_to_h5(self, file_path: str, signal_length: int, num_signals: int,
                         modulation_types: list, symbol_rate_bnds: list, phase_shift_bnds: list,
                         frequency_offset_bnds: list, SNR_bnds: list, sr_out: int,
                         dataset_name: str = "signals", compression: str = "gzip"):
        signals, metadata = self.generate_batch(
            signal_length=signal_length,
            num_signals=num_signals,
            modulation_types=modulation_types,
            symbol_rate_bnds=symbol_rate_bnds,
            phase_shift_bnds=phase_shift_bnds,
            frequency_offset_bnds=frequency_offset_bnds,
            SNR_bnds=SNR_bnds,
            sr_out=sr_out,
        )

        if not metadata:
            raise ValueError("num_signals must be at least 1")

        signal_array = np.stack(
            [np.stack((np.real(signal), np.imag(signal)), axis=0) for signal in signals],
            axis=0,
        ).astype(np.float32)

        modulation_to_index = {
            modulation: idx for idx, modulation in enumerate(sorted(set(modulation_types)))
        }
        label_array = np.asarray(
            [modulation_to_index[item["modulation"]] for item in metadata],
            dtype=np.int64,
        )

        with h5py.File(file_path, "w") as h5_file:
            h5_file.create_dataset(
                dataset_name,
                data=signal_array,
                compression=compression,
                chunks=True,
            )
            h5_file.create_dataset(
                "labels",
                data=label_array,
                compression=compression,
                chunks=True,
            )

            metadata_group = h5_file.create_group("metadata")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            for key in metadata[0]:
                values = [item[key] for item in metadata]
                if isinstance(values[0], str):
                    data = np.asarray(values, dtype=string_dtype)
                else:
                    data = np.asarray(values)
                metadata_group.create_dataset(
                    key,
                    data=data,
                    compression=compression,
                    chunks=True,
                )

            h5_file.attrs["signal_layout"] = "NCH"
            h5_file.attrs["signal_channels"] = np.asarray(["I", "Q"], dtype=string_dtype)
            h5_file.attrs["class_names"] = np.asarray(
                [name for name, _ in sorted(modulation_to_index.items(), key=lambda item: item[1])],
                dtype=string_dtype,
            )

        return file_path

    def generate_signal(self, cnfg : SignalConfig):
        if cnfg.modulation not in MODS:
            raise ValueError(f"Unsupported modulation '{cnfg.modulation}'")
        if cnfg.signal_length <= 0:
            raise ValueError("signal_length must be positive")
        if cnfg.symbol_rate <= 0:
            raise ValueError("symbol_rate must be positive")
        if cnfg.sr_out <= 0:
            raise ValueError("sr_out must be positive")
        if cnfg.sps_int <= 0:
            raise ValueError("sps_int must be positive")
        if cnfg.pulse_shaping_filter_num_taps <= 0:
            raise ValueError("pulse_shaping_filter_num_taps must be positive")

        # Get the lookup table and bit rate for the signal
        mod_dict = MODS[cnfg.modulation]
        bps = mod_dict["bps"] # bits per symbol
        LUT = mod_dict["lut"] # lookup table for this modulation type

        internal_sample_rate = float(cnfg.sps_int) * float(cnfg.symbol_rate)
        target_internal_length = int(
            np.ceil(cnfg.signal_length * internal_sample_rate / float(cnfg.sr_out))
        )
        symbols_needed = max(
            1,
            int(
                np.ceil(
                    (target_internal_length + cnfg.pulse_shaping_filter_num_taps - 1)
                    / float(cnfg.sps_int)
                )
            ),
        )

        bit_str_length = symbols_needed * bps
        bit_str = self.rng.integers(0, 2, size=bit_str_length, dtype=np.uint8)
        bit_groups = bit_str.reshape(symbols_needed, bps)
        bit_weights = (1 << np.arange(bps, dtype=np.uint32)).astype(np.uint32)
        symbol_indices = bit_groups.astype(np.uint32) @ bit_weights

        # Insert each symbol on the symbol clock and leave zeros between impulses.
        symbols = np.zeros(symbols_needed * cnfg.sps_int, dtype=np.complex64)
        symbols[::cnfg.sps_int] = LUT[symbol_indices]

        # Upsampling using RRC fitler
        rrc = self.gen_rrc(cnfg.sps_int, cnfg.pulse_shaping_filter_num_taps)
        signal = sc.convolve(symbols, rrc, mode='valid')

        # Adding white noise

        #Compute the standard deviation for the requested SNR
        signal_power = np.mean(np.abs(signal)**2) # Here, signal is complex, so to compute power we take the absolute value
        snr_linear = 10**(cnfg.SNR / 10)
        noise_power = signal_power / snr_linear
        sigma = np.sqrt(noise_power)
        #Generate the white noise
        noise = (
            self.rng.normal(0, sigma / np.sqrt(2), signal.shape)
            + 1j * self.rng.normal(0, sigma / np.sqrt(2), signal.shape)
        )
        signal = signal + noise


        # Apply transformations
        # Phase Transormation
        rotation = np.exp(1j * cnfg.phase_shift)
        signal = rotation * signal

        # Frequency Offset
        f_s = internal_sample_rate  # frequency of the "continuous" signal
        n = np.arange(len(signal))
        freq_offset = np.exp(1j * (cnfg.frequency_offset / f_s) * 2 * np.pi * n)
        
        # Offset the signal
        signal = signal * freq_offset
        signal = sc.resample(signal, cnfg.signal_length)
        return signal.astype(np.complex64)
