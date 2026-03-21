from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signal_generation.signal_generator import SignalGenerator
from signal_generation.waveform_family_generator import WaveformFamilyGenerator
ASSETS_DIR = ROOT / "docs" / "assets"

SIGNAL_LENGTH = 1024
SR_OUT = 8000


def make_spectrogram(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, times, spec = signal.spectrogram(
        iq,
        fs=SR_OUT,
        nperseg=128,
        noverlap=96,
        nfft=256,
        return_onesided=False,
        mode="magnitude",
    )
    freqs = np.fft.fftshift(np.fft.fftfreq(256, d=1.0 / SR_OUT))
    spec = np.fft.fftshift(spec, axes=0)
    spec_db = 20 * np.log10(spec + 1e-6)
    return freqs, times, spec_db


def save_modulation_vs_waveform_panel() -> Path:
    mod_gen = SignalGenerator(seed=0)
    wave_gen = WaveformFamilyGenerator(seed=0)

    modulation_classes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4"]
    waveform_classes = ["CW", "AM", "FM", "OFDM", "LFM_CHIRP", "DSSS", "FHSS", "SC_BURST"]

    mod_signals, _ = mod_gen.generate_batch(
        signal_length=SIGNAL_LENGTH,
        num_signals=len(modulation_classes),
        modulation_types=modulation_classes,
        symbol_rate_bnds=[600.0, 600.0],
        phase_shift_bnds=[0.0, 0.0],
        frequency_offset_bnds=[0.0, 0.0],
        SNR_bnds=[20.0, 20.0],
        sr_out=SR_OUT,
    )
    wave_signals, _ = wave_gen.generate_batch(
        signal_length=SIGNAL_LENGTH,
        num_signals=len(waveform_classes),
        waveform_types=waveform_classes,
        snr_db_bnds=[20.0, 20.0],
        center_frequency_bnds=[0.0, 0.0],
        sample_rate_scale_choices=[1.0],
        sr_out=SR_OUT,
        occupied_bandwidth_bnds=[500.0, 500.0],
    )

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Waveform-Family Spectrograms Show Larger Visible Differences Than Modulation Spectrograms", fontsize=14)

    mod_examples = [("BPSK", mod_signals[0]), ("QPSK", mod_signals[1]), ("16QAM", mod_signals[3]), ("64QAM", mod_signals[4])]
    wave_examples = [("CW", wave_signals[0]), ("AM", wave_signals[1]), ("FM", wave_signals[2]), ("OFDM", wave_signals[3])]

    for ax, (name, iq) in zip(axes[0], mod_examples):
        freqs, times, spec_db = make_spectrogram(iq)
        ax.imshow(spec_db, aspect="auto", origin="lower", extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap="magma")
        ax.set_title(name)
        ax.set_ylabel("Freq (Hz)")
        ax.set_xlabel("Time (s)")

    for ax, (name, iq) in zip(axes[1], wave_examples):
        freqs, times, spec_db = make_spectrogram(iq)
        ax.imshow(spec_db, aspect="auto", origin="lower", extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap="magma")
        ax.set_title(name)
        ax.set_ylabel("Freq (Hz)")
        ax.set_xlabel("Time (s)")

    out = ASSETS_DIR / "modulation_vs_waveform_spectrograms.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    path = save_modulation_vs_waveform_panel()
    print(path)


if __name__ == "__main__":
    main()
