import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import numpy as np

from SignalGenerator import SignalGenerator, SignalConfig
from Modulation import MODS


def build_config(values: dict) -> SignalConfig:
    return SignalConfig(
        modulation=values["modulation"],
        symbol_rate=np.float32(values["symbol_rate"]),
        signal_length=int(values["signal_length"]),
        phase_shift=np.float32(values["phase_shift"]),
        frequency_offset=np.float32(values["frequency_offset"]),
        sps_int=int(values["sps_int"]),
        SNR=np.float32(values["SNR"]),
        sr_out=np.float32(values["sr_out"]),
        pulse_shaping_filter_num_taps=int(values["pulse_shaping_filter_num_taps"]),
    )


def main() -> None:
    fixed_seed = 0
    state = {
        "modulation": "QPSK",
        "symbol_rate": 2000.0,
        "signal_length": 512,
        "phase_shift": 0.0,
        "frequency_offset": 0.0,
        "sps_int": 8,
        "SNR": 20.0,
        "sr_out": 8000.0,
        "pulse_shaping_filter_num_taps": 101,
    }

    fig, (ax_iq, ax_time, ax_fft) = plt.subplots(3, 1, figsize=(12, 10))
    fig.subplots_adjust(left=0.36, bottom=0.05, right=0.98, top=0.95, hspace=0.45)

    signal = SignalGenerator(seed=fixed_seed).generate_signal(build_config(state))

    line_i, = ax_time.plot(np.real(signal), lw=1.0, label="I")
    line_q, = ax_time.plot(np.imag(signal), lw=1.0, label="Q")
    ax_time.set_title("Time Domain")
    ax_time.set_xlabel("Sample")
    ax_time.set_ylabel("Amplitude")
    ax_time.legend(loc="upper right")
    ax_time.grid(True, alpha=0.25)

    scat = ax_iq.scatter(np.real(signal), np.imag(signal), s=8, alpha=0.6)
    ax_iq.set_title("IQ Scatter")
    ax_iq.set_xlabel("I")
    ax_iq.set_ylabel("Q")
    ax_iq.grid(True, alpha=0.25)
    ax_iq.axis("equal")

    fft_vals = np.fft.fftshift(np.fft.fft(signal))
    fft_mag_db = 20.0 * np.log10(np.abs(fft_vals) + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1.0 / state["sr_out"]))
    line_fft, = ax_fft.plot(freqs, fft_mag_db, lw=1.0)
    ax_fft.set_title("Frequency Domain (FFT Magnitude)")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude (dB)")
    ax_fft.grid(True, alpha=0.25)

    status = fig.text(0.36, 0.97, "", fontsize=9)

    slider_axes = {
        "symbol_rate": plt.axes([0.06, 0.86, 0.26, 0.03]),
        "signal_length": plt.axes([0.06, 0.81, 0.26, 0.03]),
        "phase_shift": plt.axes([0.06, 0.76, 0.26, 0.03]),
        "frequency_offset": plt.axes([0.06, 0.71, 0.26, 0.03]),
        "sps_int": plt.axes([0.06, 0.66, 0.26, 0.03]),
        "SNR": plt.axes([0.06, 0.61, 0.26, 0.03]),
        "sr_out": plt.axes([0.06, 0.56, 0.26, 0.03]),
        "pulse_shaping_filter_num_taps": plt.axes([0.06, 0.51, 0.26, 0.03]),
    }

    sliders = {
        "symbol_rate": Slider(slider_axes["symbol_rate"], "sym_rate", 100.0, 20000.0, valinit=state["symbol_rate"]),
        "signal_length": Slider(slider_axes["signal_length"], "sig_len", 64, 4096, valinit=state["signal_length"], valstep=1),
        "phase_shift": Slider(slider_axes["phase_shift"], "phase", -np.pi, np.pi, valinit=state["phase_shift"]),
        "frequency_offset": Slider(slider_axes["frequency_offset"], "f_offset", -5000.0, 5000.0, valinit=state["frequency_offset"]),
        "sps_int": Slider(slider_axes["sps_int"], "sps_int", 1, 64, valinit=state["sps_int"], valstep=1),
        "SNR": Slider(slider_axes["SNR"], "SNR(dB)", -10.0, 60.0, valinit=state["SNR"]),
        "sr_out": Slider(slider_axes["sr_out"], "sr_out", 100.0, 50000.0, valinit=state["sr_out"]),
        "pulse_shaping_filter_num_taps": Slider(
            slider_axes["pulse_shaping_filter_num_taps"],
            "taps",
            1,
            4097,
            valinit=state["pulse_shaping_filter_num_taps"],
            valstep=2,
        ),
    }

    radio_ax = plt.axes([0.06, 0.08, 0.26, 0.38])
    modulation_names = sorted(MODS.keys())
    active_mod = modulation_names.index(state["modulation"]) if state["modulation"] in modulation_names else 0
    radio = RadioButtons(radio_ax, modulation_names, active=active_mod)
    radio_ax.set_title("modulation")

    def update_plot() -> None:
        try:
            cfg = build_config(state)
            sig = SignalGenerator(seed=fixed_seed).generate_signal(cfg)
            x = np.arange(len(sig))
            line_i.set_data(x, np.real(sig))
            line_q.set_data(x, np.imag(sig))
            ax_time.relim()
            ax_time.autoscale_view()
            scat.set_offsets(np.column_stack((np.real(sig), np.imag(sig))))
            ax_iq.relim()
            ax_iq.autoscale_view()
            ax_iq.axis("equal")
            fft_vals = np.fft.fftshift(np.fft.fft(sig))
            fft_mag_db = 20.0 * np.log10(np.abs(fft_vals) + 1e-12)
            freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), d=1.0 / state["sr_out"]))
            line_fft.set_data(freqs, fft_mag_db)
            ax_fft.relim()
            ax_fft.autoscale_view()
            status.set_text("")
        except Exception as exc:
            status.set_text(f"Error: {exc}")
        fig.canvas.draw_idle()

    def on_slider_change(_val) -> None:
        state["symbol_rate"] = float(sliders["symbol_rate"].val)
        state["signal_length"] = int(sliders["signal_length"].val)
        state["phase_shift"] = float(sliders["phase_shift"].val)
        state["frequency_offset"] = float(sliders["frequency_offset"].val)
        state["sps_int"] = int(sliders["sps_int"].val)
        state["SNR"] = float(sliders["SNR"].val)
        state["sr_out"] = float(sliders["sr_out"].val)
        state["pulse_shaping_filter_num_taps"] = int(sliders["pulse_shaping_filter_num_taps"].val)
        update_plot()

    def on_modulation_change(label: str) -> None:
        state["modulation"] = label
        update_plot()

    for s in sliders.values():
        s.on_changed(on_slider_change)
    radio.on_clicked(on_modulation_change)

    plt.show()


if __name__ == "__main__":
    main()
