import math
from dataclasses import asdict, dataclass

import h5py
import numpy as np
import scipy.signal as sc


@dataclass
class WaveformConfig:
    waveform: str
    signal_length: int
    sr_out: np.float32
    snr_db: np.float32
    phase_shift: np.float32
    center_frequency: np.float32
    sample_rate_scale: np.float32
    amplitude_scale: np.float32
    burst_fraction: np.float32
    burst_start: np.float32
    chirp_rate: np.float32
    fm_deviation: np.float32
    am_depth: np.float32
    symbol_rate: np.float32
    hop_rate: np.float32
    spread_factor: int
    occupied_bandwidth: np.float32


class WaveformFamilyGenerator:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _random_config(
        self,
        signal_length: int,
        sr_out: float,
        waveform_types: list[str],
        snr_db_bnds: list[float],
        center_frequency_bnds: list[float],
        sample_rate_scale_choices: list[float],
        occupied_bandwidth_bnds: list[float],
    ) -> WaveformConfig:
        waveform = str(self.rng.choice(waveform_types))
        occupied_bandwidth = np.float32(self.rng.uniform(*occupied_bandwidth_bnds))
        symbol_rate = np.float32(self.rng.uniform(0.2 * occupied_bandwidth, 0.7 * occupied_bandwidth))
        hop_rate = np.float32(self.rng.uniform(8.0, 80.0))
        spread_factor = int(self.rng.integers(4, 16))
        duration = signal_length / max(sr_out, 1.0)
        chirp_span = self.rng.uniform(0.35, 0.95) * float(occupied_bandwidth)
        chirp_rate = np.float32(self.rng.choice([-1.0, 1.0]) * chirp_span / max(duration, 1e-6))
        fm_deviation = np.float32(self.rng.uniform(0.1, 0.45) * occupied_bandwidth)
        am_depth = np.float32(self.rng.uniform(0.2, 0.95))
        burst_fraction = np.float32(self.rng.uniform(0.45, 1.0))
        burst_start = np.float32(self.rng.uniform(0.0, 1.0 - burst_fraction))
        return WaveformConfig(
            waveform=waveform,
            signal_length=signal_length,
            sr_out=np.float32(sr_out),
            snr_db=np.float32(self.rng.uniform(*snr_db_bnds)),
            phase_shift=np.float32(self.rng.uniform(-math.pi, math.pi)),
            center_frequency=np.float32(self.rng.uniform(*center_frequency_bnds)),
            sample_rate_scale=np.float32(self.rng.choice(sample_rate_scale_choices)),
            amplitude_scale=np.float32(self.rng.uniform(0.7, 1.3)),
            burst_fraction=burst_fraction,
            burst_start=burst_start,
            chirp_rate=chirp_rate,
            fm_deviation=fm_deviation,
            am_depth=am_depth,
            symbol_rate=symbol_rate,
            hop_rate=hop_rate,
            spread_factor=spread_factor,
            occupied_bandwidth=occupied_bandwidth,
        )

    def _time_axis(self, cnfg: WaveformConfig) -> np.ndarray:
        return np.arange(cnfg.signal_length, dtype=np.float64) / float(cnfg.sr_out)

    def _apply_sample_rate_scale(self, signal: np.ndarray, cnfg: WaveformConfig) -> np.ndarray:
        scale = max(float(cnfg.sample_rate_scale), 1e-3)
        if np.isclose(scale, 1.0):
            return signal.astype(np.complex64)

        scaled_length = max(8, int(round(len(signal) * scale)))
        scaled = sc.resample(signal, scaled_length).astype(np.complex64)

        if scaled_length >= len(signal):
            start = (scaled_length - len(signal)) // 2
            return scaled[start : start + len(signal)].astype(np.complex64)

        pad_left = (len(signal) - scaled_length) // 2
        pad_right = len(signal) - scaled_length - pad_left
        return np.pad(scaled, (pad_left, pad_right), mode="constant").astype(np.complex64)

    def _apply_burst_window(self, signal: np.ndarray, cnfg: WaveformConfig) -> np.ndarray:
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

    def _add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        power = float(np.mean(np.abs(signal) ** 2))
        noise_power = power / max(10.0 ** (snr_db / 10.0), 1e-8)
        sigma = math.sqrt(noise_power)
        noise = (
            self.rng.normal(0.0, sigma / math.sqrt(2.0), size=signal.shape)
            + 1j * self.rng.normal(0.0, sigma / math.sqrt(2.0), size=signal.shape)
        )
        return (signal + noise).astype(np.complex64)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(np.abs(signal) ** 2) + 1e-8)
        return (signal / rms).astype(np.complex64)

    def _rrc_like_pulse(self, sps: int, span_symbols: int = 8) -> np.ndarray:
        taps = span_symbols * sps + 1
        t = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
        x = t / max(float(sps), 1.0)
        h = np.sinc(x) * np.hamming(taps)
        h /= np.sqrt(np.sum(h**2) + 1e-12)
        return h

    def _gen_cw(self, cnfg: WaveformConfig) -> np.ndarray:
        t = self._time_axis(cnfg)
        return np.exp(1j * (2 * math.pi * float(cnfg.center_frequency) * t + float(cnfg.phase_shift)))

    def _gen_am(self, cnfg: WaveformConfig) -> np.ndarray:
        t = self._time_axis(cnfg)
        mod_freq = float(self.rng.uniform(5.0, 80.0))
        envelope = 1.0 + float(cnfg.am_depth) * np.sin(2 * math.pi * mod_freq * t)
        carrier = np.exp(1j * (2 * math.pi * float(cnfg.center_frequency) * t + float(cnfg.phase_shift)))
        return envelope * carrier

    def _gen_fm(self, cnfg: WaveformConfig) -> np.ndarray:
        t = self._time_axis(cnfg)
        mod_freq = float(self.rng.uniform(5.0, max(15.0, 0.15 * float(cnfg.occupied_bandwidth))))
        phase = (
            2 * math.pi * float(cnfg.center_frequency) * t
            + float(cnfg.phase_shift)
            + (float(cnfg.fm_deviation) / max(mod_freq, 1e-3)) * np.sin(2 * math.pi * mod_freq * t)
        )
        return np.exp(1j * phase)

    def _gen_lfm_chirp(self, cnfg: WaveformConfig) -> np.ndarray:
        t = self._time_axis(cnfg)
        phase = (
            float(cnfg.phase_shift)
            + 2 * math.pi * float(cnfg.center_frequency) * t
            + math.pi * float(cnfg.chirp_rate) * t**2
        )
        return np.exp(1j * phase)

    def _gen_sc_burst(self, cnfg: WaveformConfig) -> np.ndarray:
        sps = max(4, int(round(float(cnfg.sr_out) / max(float(cnfg.symbol_rate), 1.0))))
        num_symbols = max(16, cnfg.signal_length // sps + 8)
        constellation = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / math.sqrt(2)
        symbols = self.rng.choice(constellation, size=num_symbols)
        up = np.zeros(num_symbols * sps, dtype=np.complex64)
        up[::sps] = symbols
        pulse = self._rrc_like_pulse(sps)
        shaped = sc.lfilter(pulse, [1.0], up)
        shaped = shaped[: cnfg.signal_length]
        t = self._time_axis(cnfg)
        return shaped * np.exp(1j * (2 * math.pi * float(cnfg.center_frequency) * t + float(cnfg.phase_shift)))

    def _gen_ofdm(self, cnfg: WaveformConfig) -> np.ndarray:
        nfft = int(self.rng.choice([32, 64, 128]))
        cp = nfft // 8
        constellation = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / math.sqrt(2)
        frames = []
        while sum(len(frame) for frame in frames) < cnfg.signal_length + nfft:
            bins = np.zeros(nfft, dtype=np.complex64)
            bin_bw = float(cnfg.sr_out) / nfft
            active_bins = max(4, int(round(float(cnfg.occupied_bandwidth) / max(bin_bw, 1e-6))))
            active_bins = min(active_bins, nfft // 2)
            start = nfft // 2 - active_bins // 2
            stop = start + active_bins
            active = slice(start, stop)
            bins[active] = self.rng.choice(constellation, size=(active.stop - active.start))
            time_symbol = np.fft.ifft(np.fft.ifftshift(bins))
            with_cp = np.concatenate([time_symbol[-cp:], time_symbol])
            frames.append(with_cp.astype(np.complex64))
        signal = np.concatenate(frames)[: cnfg.signal_length]
        t = self._time_axis(cnfg)
        return signal * np.exp(1j * (2 * math.pi * float(cnfg.center_frequency) * t + float(cnfg.phase_shift)))

    def _gen_dsss(self, cnfg: WaveformConfig) -> np.ndarray:
        chips_per_symbol = cnfg.spread_factor
        num_symbols = max(8, cnfg.signal_length // chips_per_symbol + 4)
        symbols = self.rng.choice(np.asarray([1.0, -1.0], dtype=np.float32), size=num_symbols)
        pn = self.rng.choice(np.asarray([1.0, -1.0], dtype=np.float32), size=num_symbols * chips_per_symbol)
        spread = np.repeat(symbols, chips_per_symbol) * pn[: num_symbols * chips_per_symbol]
        signal = spread.astype(np.complex64)
        signal = sc.resample(signal, cnfg.signal_length)
        norm_bw = min(0.49, float(cnfg.occupied_bandwidth) / max(float(cnfg.sr_out) / 2.0, 1e-6))
        taps = sc.firwin(63, norm_bw)
        signal = sc.lfilter(taps, [1.0], signal)
        t = self._time_axis(cnfg)
        return signal * np.exp(1j * (2 * math.pi * float(cnfg.center_frequency) * t + float(cnfg.phase_shift)))

    def _gen_fhss(self, cnfg: WaveformConfig) -> np.ndarray:
        t = self._time_axis(cnfg)
        hop_period = max(4, int(round(float(cnfg.sr_out) / max(float(cnfg.hop_rate), 1.0))))
        num_hops = int(np.ceil(cnfg.signal_length / hop_period))
        offsets = self.rng.uniform(
            -0.45 * float(cnfg.occupied_bandwidth),
            0.45 * float(cnfg.occupied_bandwidth),
            size=num_hops,
        )
        freq_track = np.repeat(offsets, hop_period)[: cnfg.signal_length]
        phase = float(cnfg.phase_shift) + 2 * math.pi * np.cumsum(float(cnfg.center_frequency) + freq_track) / float(cnfg.sr_out)
        return np.exp(1j * phase)

    def generate_signal(self, cnfg: WaveformConfig) -> np.ndarray:
        generators = {
            "CW": self._gen_cw,
            "AM": self._gen_am,
            "FM": self._gen_fm,
            "OFDM": self._gen_ofdm,
            "LFM_CHIRP": self._gen_lfm_chirp,
            "DSSS": self._gen_dsss,
            "FHSS": self._gen_fhss,
            "SC_BURST": self._gen_sc_burst,
        }
        if cnfg.waveform not in generators:
            raise ValueError(f"Unsupported waveform '{cnfg.waveform}'")

        signal = generators[cnfg.waveform](cnfg).astype(np.complex64)
        signal *= float(cnfg.amplitude_scale)
        signal = self._apply_burst_window(signal, cnfg)
        signal = self._apply_sample_rate_scale(signal, cnfg)
        signal = self._add_awgn(signal, float(cnfg.snr_db))
        return self._normalize(signal)

    def generate_batch(
        self,
        signal_length: int,
        num_signals: int,
        waveform_types: list[str],
        snr_db_bnds: list[float],
        center_frequency_bnds: list[float],
        sample_rate_scale_choices: list[float],
        sr_out: int,
        occupied_bandwidth_bnds: list[float] | None = None,
    ):
        if signal_length <= 0 or num_signals <= 0 or sr_out <= 0:
            raise ValueError("signal_length, num_signals, and sr_out must be positive")
        if not waveform_types:
            raise ValueError("waveform_types must not be empty")
        if not sample_rate_scale_choices:
            raise ValueError("sample_rate_scale_choices must not be empty")
        if occupied_bandwidth_bnds is None:
            occupied_bandwidth_bnds = [0.03 * sr_out, 0.12 * sr_out]
        if occupied_bandwidth_bnds[0] <= 0 or occupied_bandwidth_bnds[0] > occupied_bandwidth_bnds[1]:
            raise ValueError("occupied_bandwidth_bnds must be positive and ordered [min, max]")

        configs = [
            self._random_config(
                signal_length=signal_length,
                sr_out=sr_out,
                waveform_types=waveform_types,
                snr_db_bnds=snr_db_bnds,
                center_frequency_bnds=center_frequency_bnds,
                sample_rate_scale_choices=sample_rate_scale_choices,
                occupied_bandwidth_bnds=occupied_bandwidth_bnds,
            )
            for _ in range(num_signals)
        ]
        signals = [self.generate_signal(cnfg) for cnfg in configs]
        metadata = [asdict(cnfg) for cnfg in configs]
        return signals, metadata

    def save_batch_to_h5(
        self,
        file_path: str,
        signal_length: int,
        num_signals: int,
        waveform_types: list[str],
        snr_db_bnds: list[float],
        center_frequency_bnds: list[float],
        sample_rate_scale_choices: list[float],
        sr_out: int,
        occupied_bandwidth_bnds: list[float] | None = None,
        dataset_name: str = "signals",
        compression: str = "gzip",
    ) -> str:
        signals, metadata = self.generate_batch(
            signal_length=signal_length,
            num_signals=num_signals,
            waveform_types=waveform_types,
            snr_db_bnds=snr_db_bnds,
            center_frequency_bnds=center_frequency_bnds,
            sample_rate_scale_choices=sample_rate_scale_choices,
            occupied_bandwidth_bnds=occupied_bandwidth_bnds,
            sr_out=sr_out,
        )

        signal_array = np.stack(
            [np.stack((np.real(signal), np.imag(signal)), axis=0) for signal in signals],
            axis=0,
        ).astype(np.float32)

        class_to_index = {name: idx for idx, name in enumerate(sorted(set(waveform_types)))}
        label_array = np.asarray([class_to_index[item["waveform"]] for item in metadata], dtype=np.int64)

        with h5py.File(file_path, "w") as h5_file:
            h5_file.create_dataset(dataset_name, data=signal_array, compression=compression, chunks=True)
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
                [name for name, _ in sorted(class_to_index.items(), key=lambda item: item[1])],
                dtype=string_dtype,
            )
            h5_file.attrs["dataset_profile"] = "waveform_family_frequency_domain_v1"

        return file_path
