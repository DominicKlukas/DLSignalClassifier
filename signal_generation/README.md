# Signal Generation

This package contains the reusable signal-generation code used by the synthetic experiments.

Modules:

- `modulation.py`: constellation and modulation lookup tables
- `signal_generator.py`: synthetic modulation-family dataset generation
- `waveform_family_generator.py`: synthetic waveform-family dataset generation

These modules are used by:

- `experiments/exp01_iq_vs_fft`
- `experiments/tools/generate_dataset_figures.py`
