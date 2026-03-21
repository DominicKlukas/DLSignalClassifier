import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np


def _decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


class DatasetInspector:
    def __init__(self, file_path: Path, dataset_name: str = "signals") -> None:
        self.file_path = file_path
        self.dataset_name = dataset_name

        with h5py.File(file_path, "r") as h5_file:
            self.signals = h5_file[dataset_name][:]
            self.labels = h5_file["labels"][:] if "labels" in h5_file else None
            self.attrs = {
                key: [_decode_value(v) for v in h5_file.attrs[key]]
                if isinstance(h5_file.attrs[key], np.ndarray)
                else _decode_value(h5_file.attrs[key])
                for key in h5_file.attrs.keys()
            }
            self.metadata = {}
            if "metadata" in h5_file:
                metadata_group = h5_file["metadata"]
                for key in metadata_group.keys():
                    self.metadata[key] = [_decode_value(value) for value in metadata_group[key][:]]

        if self.signals.ndim != 3:
            raise ValueError(
                f"Expected `{dataset_name}` to have shape (N, C, L), got {self.signals.shape}"
            )
        if self.signals.shape[1] < 2:
            raise ValueError(
                f"Expected at least two channels (I/Q), got shape {self.signals.shape}"
            )

        self.num_signals = int(self.signals.shape[0])
        self.signal_length = int(self.signals.shape[2])
        self.class_names = list(self.attrs.get("class_names", []))
        self.modulations = list(self.metadata.get("modulation", []))
        self.indices_by_filter = self._build_filter_index()
        self.current_filter = "All"
        self.filtered_indices = self.indices_by_filter[self.current_filter]
        self.current_position = 0

        self.fig = None
        self.ax_iq = None
        self.ax_time = None
        self.ax_fft = None
        self.ax_meta = None
        self.sample_slider = None
        self.filter_radio = None
        self.prev_button = None
        self.next_button = None
        self.line_i = None
        self.line_q = None
        self.line_fft = None
        self.scat = None
        self.meta_text = None
        self.status_text = None

    def _build_filter_index(self) -> dict[str, list[int]]:
        indices = {"All": list(range(self.num_signals))}
        for idx, modulation in enumerate(self.modulations):
            indices.setdefault(str(modulation), []).append(idx)
        return indices

    def _get_signal(self, sample_idx: int) -> np.ndarray:
        return self.signals[sample_idx, 0].astype(np.float64) + 1j * self.signals[sample_idx, 1].astype(np.float64)

    def _get_sample_rate(self, sample_idx: int) -> float:
        if "sr_out" in self.metadata:
            return float(self.metadata["sr_out"][sample_idx])
        return float(self.signal_length)

    def _get_label_name(self, sample_idx: int) -> str:
        if self.modulations:
            return str(self.modulations[sample_idx])
        if self.labels is not None and self.class_names:
            label_idx = int(self.labels[sample_idx])
            if 0 <= label_idx < len(self.class_names):
                return str(self.class_names[label_idx])
        if self.labels is not None:
            return str(int(self.labels[sample_idx]))
        return "unknown"

    def _format_dataset_summary(self) -> str:
        lines = [
            f"File: {self.file_path.name}",
            f"Dataset: {self.dataset_name}",
            f"Signals: {self.num_signals}",
            f"Shape: {tuple(self.signals.shape)}",
            f"Layout: {self.attrs.get('signal_layout', 'unknown')}",
            f"Channels: {', '.join(self.attrs.get('signal_channels', ['I', 'Q']))}",
        ]
        if self.class_names:
            lines.append(f"Classes: {', '.join(map(str, self.class_names))}")
        return "\n".join(lines)

    def _format_sample_metadata(self, sample_idx: int) -> str:
        lines = [
            self._format_dataset_summary(),
            "",
            f"Visible sample: {self.current_position + 1}/{len(self.filtered_indices)}",
            f"Global index: {sample_idx}",
            f"Label: {self._get_label_name(sample_idx)}",
        ]
        if self.labels is not None:
            lines.append(f"Label index: {int(self.labels[sample_idx])}")
        for key in sorted(self.metadata.keys()):
            lines.append(f"{key}: {self.metadata[key][sample_idx]}")
        return "\n".join(lines)

    def _update_slider_bounds(self) -> None:
        max_position = max(len(self.filtered_indices) - 1, 0)
        self.sample_slider.valmin = 0
        self.sample_slider.valmax = max_position
        self.sample_slider.ax.set_xlim(self.sample_slider.valmin, self.sample_slider.valmax or 1)
        self.sample_slider.valstep = 1
        self.sample_slider.label.set_text(f"sample ({self.current_filter})")
        if self.current_position > max_position:
            self.current_position = max_position

    def _set_current_position(self, position: int) -> None:
        if not self.filtered_indices:
            self.status_text.set_text(f"No samples found for filter '{self.current_filter}'")
            self.fig.canvas.draw_idle()
            return
        self.current_position = int(np.clip(position, 0, len(self.filtered_indices) - 1))
        self.sample_slider.set_val(self.current_position)

    def _refresh_plot(self) -> None:
        if not self.filtered_indices:
            return

        sample_idx = self.filtered_indices[self.current_position]
        signal = self._get_signal(sample_idx)
        x = np.arange(len(signal))

        self.line_i.set_data(x, np.real(signal))
        self.line_q.set_data(x, np.imag(signal))
        self.ax_time.relim()
        self.ax_time.autoscale_view()

        self.scat.set_offsets(np.column_stack((np.real(signal), np.imag(signal))))
        self.ax_iq.relim()
        self.ax_iq.autoscale_view()
        self.ax_iq.axis("equal")

        sample_rate = self._get_sample_rate(sample_idx)
        fft_vals = np.fft.fftshift(np.fft.fft(signal))
        fft_mag_db = 20.0 * np.log10(np.abs(fft_vals) + 1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1.0 / sample_rate))
        self.line_fft.set_data(freqs, fft_mag_db)
        self.ax_fft.relim()
        self.ax_fft.autoscale_view()

        self.meta_text.set_text(self._format_sample_metadata(sample_idx))
        self.status_text.set_text("")
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, value: float) -> None:
        self.current_position = int(value)
        self._refresh_plot()

    def _on_filter_change(self, label: str) -> None:
        self.current_filter = label
        self.filtered_indices = self.indices_by_filter[label]
        self.current_position = 0
        self._update_slider_bounds()
        self.sample_slider.set_val(self.current_position)
        self._refresh_plot()

    def _step(self, delta: int) -> None:
        if not self.filtered_indices:
            return
        self._set_current_position(self.current_position + delta)

    def _on_key(self, event) -> None:
        if event.key in {"right", "down"}:
            self._step(1)
        elif event.key in {"left", "up"}:
            self._step(-1)

    def show(self) -> None:
        fig = plt.figure(figsize=(15, 9))
        fig.subplots_adjust(left=0.28, right=0.98, bottom=0.14, top=0.95, wspace=0.3, hspace=0.35)
        grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])

        self.fig = fig
        self.ax_iq = fig.add_subplot(grid[0, 0])
        self.ax_meta = fig.add_subplot(grid[0, 1])
        self.ax_time = fig.add_subplot(grid[1, 0])
        self.ax_fft = fig.add_subplot(grid[1, 1])

        initial_signal = self._get_signal(0)
        x = np.arange(len(initial_signal))
        sample_rate = self._get_sample_rate(0)

        self.scat = self.ax_iq.scatter(np.real(initial_signal), np.imag(initial_signal), s=8, alpha=0.6)
        self.ax_iq.set_title("IQ Scatter")
        self.ax_iq.set_xlabel("I")
        self.ax_iq.set_ylabel("Q")
        self.ax_iq.grid(True, alpha=0.25)
        self.ax_iq.axis("equal")

        self.line_i, = self.ax_time.plot(x, np.real(initial_signal), lw=1.0, label="I")
        self.line_q, = self.ax_time.plot(x, np.imag(initial_signal), lw=1.0, label="Q")
        self.ax_time.set_title("Time Domain")
        self.ax_time.set_xlabel("Sample")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.legend(loc="upper right")
        self.ax_time.grid(True, alpha=0.25)

        fft_vals = np.fft.fftshift(np.fft.fft(initial_signal))
        fft_mag_db = 20.0 * np.log10(np.abs(fft_vals) + 1e-12)
        freqs = np.fft.fftshift(np.fft.fftfreq(len(initial_signal), d=1.0 / sample_rate))
        self.line_fft, = self.ax_fft.plot(freqs, fft_mag_db, lw=1.0)
        self.ax_fft.set_title("Frequency Domain (FFT Magnitude)")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Magnitude (dB)")
        self.ax_fft.grid(True, alpha=0.25)

        self.ax_meta.axis("off")
        self.meta_text = self.ax_meta.text(
            0.0,
            1.0,
            self._format_sample_metadata(0),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            transform=self.ax_meta.transAxes,
        )

        slider_ax = plt.axes([0.28, 0.06, 0.52, 0.03])
        self.sample_slider = Slider(
            slider_ax,
            "sample",
            0,
            max(self.num_signals - 1, 1),
            valinit=0,
            valstep=1,
        )
        self.sample_slider.on_changed(self._on_slider_change)

        radio_ax = plt.axes([0.05, 0.36, 0.18, 0.48])
        filter_names = ["All"] + [name for name in sorted(self.indices_by_filter.keys()) if name != "All"]
        self.filter_radio = RadioButtons(radio_ax, filter_names, active=0)
        radio_ax.set_title("modulation filter")
        self.filter_radio.on_clicked(self._on_filter_change)

        prev_ax = plt.axes([0.82, 0.055, 0.07, 0.04])
        next_ax = plt.axes([0.90, 0.055, 0.07, 0.04])
        self.prev_button = Button(prev_ax, "Prev")
        self.next_button = Button(next_ax, "Next")
        self.prev_button.on_clicked(lambda _event: self._step(-1))
        self.next_button.on_clicked(lambda _event: self._step(1))

        self.status_text = fig.text(0.28, 0.975, "", fontsize=9, color="tab:red")
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_slider_bounds()
        self._refresh_plot()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an HDF5 signal dataset with IQ/time/FFT plots.")
    parser.add_argument(
        "file",
        nargs="?",
        default="challenging_signals_dataset.h5",
        help="Path to the HDF5 dataset file (default: signals_dataset.h5).",
    )
    parser.add_argument(
        "--dataset-name",
        default="signals",
        help="Dataset name that stores the signal tensor (default: signals).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspector = DatasetInspector(Path(args.file), dataset_name=args.dataset_name)
    inspector.show()


if __name__ == "__main__":
    main()
