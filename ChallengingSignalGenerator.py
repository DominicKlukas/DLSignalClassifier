import math
from dataclasses import asdict, dataclass

import h5py
import numpy as np
import scipy.signal as sc

from Modulation import MODS
from SignalGenerator import SignalGenerator


@dataclass
class ChallengingSignalConfig:
    modulation: str
    signal_length: int
    symbol_rate: np.float32
    sr_out: np.float32
    sps_int: int
    pulse_shaping_filter_num_taps: int
    SNR: np.float32
    phase_shift: np.float32
    frequency_offset: np.float32
    timing_offset: np.float32
    sample_rate_ppm: np.float32
    iq_gain_imbalance_db: np.float32
    iq_phase_error_deg: np.float32
    dc_i: np.float32
    dc_q: np.float32
    multipath_profile: str
    multipath_strength: np.float32
    colored_noise_strength: np.float32
    nonlinearity_strength: np.float32
    burst_fraction: np.float32
    burst_start: np.float32
    interferer_modulation: str
    interferer_snr_db: np.float32
    interferer_frequency_offset: np.float32
    interferer_phase_shift: np.float32


class ChallengingSignalGenerator(SignalGenerator):
    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)

    def _choose_config(
        self,
        signal_length: int,
        modulation_types: list[str],
        symbol_rate_bnds: list[float],
        phase_shift_bnds: list[float],
        frequency_offset_bnds: list[float],
        snr_bnds: list[float],
        sr_out: float,
    ) -> ChallengingSignalConfig:
        modulation = str(self.rng.choice(modulation_types))
        symbol_rate = np.float32(self.rng.uniform(*symbol_rate_bnds))
        phase_shift = np.float32(self.rng.uniform(*phase_shift_bnds))
        frequency_offset = np.float32(self.rng.uniform(*frequency_offset_bnds))
        snr = np.float32(self.rng.uniform(*snr_bnds))

        sps_from_ratio = int(np.ceil(sr_out / float(symbol_rate)))
        sps_int = int(np.clip(sps_from_ratio, 6, 12))
        pulse_taps = 12 * sps_int + 1

        timing_offset = np.float32(self.rng.uniform(-0.45, 0.45))
        sample_rate_ppm = np.float32(self.rng.uniform(-60.0, 60.0))
        iq_gain_imbalance_db = np.float32(self.rng.uniform(-2.5, 2.5))
        iq_phase_error_deg = np.float32(self.rng.uniform(-8.0, 8.0))
        dc_i = np.float32(self.rng.uniform(-0.08, 0.08))
        dc_q = np.float32(self.rng.uniform(-0.08, 0.08))
        multipath_profile = str(self.rng.choice(["none", "mild", "moderate"]))
        multipath_strength = np.float32(self.rng.uniform(0.05, 0.55))
        colored_noise_strength = np.float32(self.rng.uniform(0.0, 0.45))
        nonlinearity_strength = np.float32(self.rng.uniform(0.0, 0.5))
        burst_fraction = np.float32(self.rng.uniform(0.55, 1.0))
        burst_start = np.float32(self.rng.uniform(0.0, 1.0 - burst_fraction))

        interferer_on = bool(self.rng.random() < 0.45)
        interferer_modulation = str(self.rng.choice(modulation_types)) if interferer_on else "none"
        interferer_snr_db = np.float32(self.rng.uniform(-6.0, 14.0) if interferer_on else -120.0)
        interferer_frequency_offset = np.float32(
            self.rng.uniform(-1.5 * sr_out / 4.0, 1.5 * sr_out / 4.0) if interferer_on else 0.0
        )
        interferer_phase_shift = np.float32(
            self.rng.uniform(-math.pi, math.pi) if interferer_on else 0.0
        )

        return ChallengingSignalConfig(
            modulation=modulation,
            signal_length=signal_length,
            symbol_rate=symbol_rate,
            sr_out=np.float32(sr_out),
            sps_int=sps_int,
            pulse_shaping_filter_num_taps=pulse_taps,
            SNR=snr,
            phase_shift=phase_shift,
            frequency_offset=frequency_offset,
            timing_offset=timing_offset,
            sample_rate_ppm=sample_rate_ppm,
            iq_gain_imbalance_db=iq_gain_imbalance_db,
            iq_phase_error_deg=iq_phase_error_deg,
            dc_i=dc_i,
            dc_q=dc_q,
            multipath_profile=multipath_profile,
            multipath_strength=multipath_strength,
            colored_noise_strength=colored_noise_strength,
            nonlinearity_strength=nonlinearity_strength,
            burst_fraction=burst_fraction,
            burst_start=burst_start,
            interferer_modulation=interferer_modulation,
            interferer_snr_db=interferer_snr_db,
            interferer_frequency_offset=interferer_frequency_offset,
            interferer_phase_shift=interferer_phase_shift,
        )

    def _baseband_from_config(
        self,
        modulation: str,
        symbol_rate: float,
        signal_length: int,
        sr_out: float,
        sps_int: int,
        pulse_taps: int,
        pad_symbols: int,
    ) -> np.ndarray:
        mod_dict = MODS[modulation]
        bps = mod_dict["bps"]
        lut = mod_dict["lut"]
        internal_sample_rate = float(sps_int) * float(symbol_rate)

        target_internal_length = int(
            np.ceil((signal_length + 2 * pad_symbols * sps_int) * internal_sample_rate / float(sr_out))
        )
        symbols_needed = max(
            8,
            int(np.ceil((target_internal_length + pulse_taps - 1) / float(sps_int))) + 2 * pad_symbols,
        )

        bit_str = self.rng.integers(0, 2, size=symbols_needed * bps, dtype=np.uint8)
        bit_groups = bit_str.reshape(symbols_needed, bps)
        bit_weights = (1 << np.arange(bps, dtype=np.uint32)).astype(np.uint32)
        symbol_indices = bit_groups.astype(np.uint32) @ bit_weights

        symbols = np.zeros(symbols_needed * sps_int, dtype=np.complex64)
        symbols[::sps_int] = lut[symbol_indices]
        rrc = self.gen_rrc(sps_int, pulse_taps)
        return sc.convolve(symbols, rrc, mode="valid").astype(np.complex64)

    def _apply_timing_and_rate_offset(
        self,
        signal: np.ndarray,
        cnfg: ChallengingSignalConfig,
        output_length: int,
    ) -> np.ndarray:
        scale = 1.0 + float(cnfg.sample_rate_ppm) * 1e-6
        positions = float(cnfg.timing_offset) + scale * np.arange(output_length, dtype=np.float64)
        positions = np.clip(positions, 0.0, len(signal) - 1.001)
        base = np.arange(len(signal), dtype=np.float64)
        real = np.interp(positions, base, signal.real)
        imag = np.interp(positions, base, signal.imag)
        return (real + 1j * imag).astype(np.complex64)

    def _apply_burst_window(self, signal: np.ndarray, cnfg: ChallengingSignalConfig) -> np.ndarray:
        length = len(signal)
        start = int(round(float(cnfg.burst_start) * length))
        burst_len = max(8, int(round(float(cnfg.burst_fraction) * length)))
        end = min(length, start + burst_len)
        window = np.zeros(length, dtype=np.float32)
        if end > start:
            ramp = min(max(4, length // 32), max(1, (end - start) // 4))
            active = np.ones(end - start, dtype=np.float32)
            if 2 * ramp < active.size:
                active[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False)
                active[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=False)
            window[start:end] = active
        return (signal * window).astype(np.complex64)

    def _apply_multipath(self, signal: np.ndarray, cnfg: ChallengingSignalConfig) -> np.ndarray:
        if cnfg.multipath_profile == "none":
            return signal

        if cnfg.multipath_profile == "mild":
            delays = [0, 1, 3]
        else:
            delays = [0, 1, 2, 5, 8]

        taps = np.zeros(delays[-1] + 1, dtype=np.complex64)
        taps[0] = 1.0 + 0.0j
        for delay in delays[1:]:
            amp = float(cnfg.multipath_strength) * float(self.rng.uniform(0.15, 1.0))
            phase = float(self.rng.uniform(-math.pi, math.pi))
            taps[delay] = amp * np.exp(1j * phase)
        power = np.sqrt(np.sum(np.abs(taps) ** 2))
        taps /= max(power, 1e-6)
        return sc.lfilter(taps, [1.0], signal).astype(np.complex64)

    def _apply_frequency_and_phase(self, signal: np.ndarray, phase_shift: float, frequency_offset: float, sample_rate: float) -> np.ndarray:
        n = np.arange(len(signal), dtype=np.float64)
        phase = float(phase_shift) + 2.0 * math.pi * float(frequency_offset) * n / float(sample_rate)
        return (signal * np.exp(1j * phase)).astype(np.complex64)

    def _apply_iq_imbalance(self, signal: np.ndarray, cnfg: ChallengingSignalConfig) -> np.ndarray:
        gain = 10.0 ** (float(cnfg.iq_gain_imbalance_db) / 20.0)
        phase_err = math.radians(float(cnfg.iq_phase_error_deg))
        i = signal.real * gain
        q = signal.imag / gain
        q_rot = q * math.cos(phase_err) + i * math.sin(phase_err)
        return (i + 1j * q_rot).astype(np.complex64)

    def _apply_nonlinearity(self, signal: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 1e-6:
            return signal
        amplitude = np.abs(signal)
        phase = np.angle(signal)
        compressed = np.tanh((1.0 + 2.5 * strength) * amplitude) / max(1.0 + strength, 1e-6)
        ampm = 0.15 * strength * amplitude**2
        return (compressed * np.exp(1j * (phase + ampm))).astype(np.complex64)

    def _add_noise(self, signal: np.ndarray, snr_db: float, colored_strength: float) -> np.ndarray:
        signal_power = float(np.mean(np.abs(signal) ** 2))
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / max(snr_linear, 1e-8)
        sigma = math.sqrt(noise_power)

        white = (
            self.rng.normal(0.0, sigma / math.sqrt(2.0), signal.shape)
            + 1j * self.rng.normal(0.0, sigma / math.sqrt(2.0), signal.shape)
        )
        if colored_strength <= 1e-6:
            return (signal + white).astype(np.complex64)

        colored = sc.lfilter([1.0], [1.0, -0.88], white).astype(np.complex64)
        noise = (1.0 - colored_strength) * white + colored_strength * colored
        return (signal + noise).astype(np.complex64)

    def _add_interferer(self, signal: np.ndarray, cnfg: ChallengingSignalConfig) -> np.ndarray:
        if cnfg.interferer_modulation == "none":
            return signal

        interferer = self._baseband_from_config(
            modulation=cnfg.interferer_modulation,
            symbol_rate=float(cnfg.symbol_rate) * float(self.rng.uniform(0.7, 1.3)),
            signal_length=cnfg.signal_length + 128,
            sr_out=float(cnfg.sr_out),
            sps_int=max(4, cnfg.sps_int - 1),
            pulse_taps=max(7, cnfg.pulse_shaping_filter_num_taps - 4),
            pad_symbols=8,
        )
        interferer = self._apply_timing_and_rate_offset(interferer, cnfg, len(signal))
        interferer = self._apply_frequency_and_phase(
            interferer,
            phase_shift=float(cnfg.interferer_phase_shift),
            frequency_offset=float(cnfg.interferer_frequency_offset),
            sample_rate=float(cnfg.sr_out),
        )

        sig_power = float(np.mean(np.abs(signal) ** 2))
        target_ratio = 10.0 ** (float(cnfg.interferer_snr_db) / 10.0)
        int_power = float(np.mean(np.abs(interferer) ** 2))
        scale = math.sqrt(sig_power / max(int_power * target_ratio, 1e-8))
        return (signal + scale * interferer).astype(np.complex64)

    def generate_challenging_signal(self, cnfg: ChallengingSignalConfig) -> np.ndarray:
        pad_symbols = 24
        baseband = self._baseband_from_config(
            modulation=cnfg.modulation,
            symbol_rate=float(cnfg.symbol_rate),
            signal_length=cnfg.signal_length,
            sr_out=float(cnfg.sr_out),
            sps_int=cnfg.sps_int,
            pulse_taps=cnfg.pulse_shaping_filter_num_taps,
            pad_symbols=pad_symbols,
        )

        oversampled_length = cnfg.signal_length + 2 * pad_symbols * cnfg.sps_int
        signal = self._apply_timing_and_rate_offset(baseband, cnfg, oversampled_length)
        signal = self._apply_burst_window(signal, cnfg)
        signal = self._apply_multipath(signal, cnfg)
        signal = self._apply_frequency_and_phase(
            signal,
            phase_shift=float(cnfg.phase_shift),
            frequency_offset=float(cnfg.frequency_offset),
            sample_rate=float(cnfg.sr_out),
        )
        signal = self._add_interferer(signal, cnfg)
        signal = self._apply_iq_imbalance(signal, cnfg)
        signal = signal + np.complex64(cnfg.dc_i + 1j * cnfg.dc_q)
        signal = self._apply_nonlinearity(signal, float(cnfg.nonlinearity_strength))
        signal = self._add_noise(signal, float(cnfg.SNR), float(cnfg.colored_noise_strength))

        crop_start = max(0, (len(signal) - cnfg.signal_length) // 2)
        cropped = signal[crop_start : crop_start + cnfg.signal_length]
        if len(cropped) != cnfg.signal_length:
            cropped = sc.resample(signal, cnfg.signal_length)

        norm = np.sqrt(np.mean(np.abs(cropped) ** 2) + 1e-8)
        return (cropped / norm).astype(np.complex64)

    def generate_batch(
        self,
        signal_length: int,
        num_signals: int,
        modulation_types: list[str],
        symbol_rate_bnds: list[float],
        phase_shift_bnds: list[float],
        frequency_offset_bnds: list[float],
        SNR_bnds: list[float],
        sr_out: int,
    ):
        if signal_length <= 0:
            raise ValueError("signal_length must be positive")
        if num_signals <= 0:
            raise ValueError("num_signals must be positive")
        if sr_out <= 0:
            raise ValueError("sr_out must be positive")
        if not modulation_types:
            raise ValueError("modulation_types must not be empty")

        signal_cnfgs = [
            self._choose_config(
                signal_length=signal_length,
                modulation_types=modulation_types,
                symbol_rate_bnds=symbol_rate_bnds,
                phase_shift_bnds=phase_shift_bnds,
                frequency_offset_bnds=frequency_offset_bnds,
                snr_bnds=SNR_bnds,
                sr_out=sr_out,
            )
            for _ in range(num_signals)
        ]
        signals = [self.generate_challenging_signal(cnfg) for cnfg in signal_cnfgs]
        metadata = [asdict(cnfg) for cnfg in signal_cnfgs]
        return signals, metadata

    def save_batch_to_h5(
        self,
        file_path: str,
        signal_length: int,
        num_signals: int,
        modulation_types: list[str],
        symbol_rate_bnds: list[float],
        phase_shift_bnds: list[float],
        frequency_offset_bnds: list[float],
        SNR_bnds: list[float],
        sr_out: int,
        dataset_name: str = "signals",
        compression: str = "gzip",
    ) -> str:
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
            h5_file.create_dataset("labels", data=label_array, compression=compression, chunks=True)

            metadata_group = h5_file.create_group("metadata")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            for key in metadata[0]:
                values = [item[key] for item in metadata]
                if isinstance(values[0], str):
                    data = np.asarray(values, dtype=string_dtype)
                else:
                    data = np.asarray(values)
                metadata_group.create_dataset(key, data=data, compression=compression, chunks=True)

            h5_file.attrs["signal_layout"] = "NCH"
            h5_file.attrs["signal_channels"] = np.asarray(["I", "Q"], dtype=string_dtype)
            h5_file.attrs["class_names"] = np.asarray(
                [name for name, _ in sorted(modulation_to_index.items(), key=lambda item: item[1])],
                dtype=string_dtype,
            )
            h5_file.attrs["dataset_profile"] = "challenging_sdr_synthetic_v1"

        return file_path
