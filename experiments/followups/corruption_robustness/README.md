# Corruption Robustness Follow-Up

This follow-up tests a clean-train / corrupted-test setting for the repo's IQ, FFT, and fusion models.

Current scope:

- default dataset: synthetic `waveform_family`
- training regimes:
  - `clean`: clean train/validation splits only
  - `augmented`: corruption-augmented train/validation splits
- evaluation: corrupted test split only
- models:
  - IQ expert
  - FFT expert
  - gated IQ+FFT fusion
  - frozen-expert residual fusion

Corruption families included:

- `awgn`
- `impulse_noise`
- `narrowband_interferer`
- `sample_rate_distortion`
- `clipping`

Artifacts:

- `results_clean.json`: clean-train / corrupted-test metrics and summaries
- `results_augmented.json`: augmented-train / corrupted-test metrics and summaries
- `Results_Summary.md`: cross-dataset interpretation of the augmented-training follow-up
- `results_by_dataset/`: per-dataset augmented-training result bundles and plots
- `plots_clean/`: robustness plots for the clean-train run
- `plots_augmented/`: robustness plots for the augmented-train run

Run:

```bash
./.venv/bin/python experiments/followups/corruption_robustness/run.py
```

Useful options:

```bash
./.venv/bin/python experiments/followups/corruption_robustness/run.py --dataset waveform_family --seed 0 --results-path experiments/followups/corruption_robustness/results_clean.json
./.venv/bin/python experiments/followups/corruption_robustness/run.py --dataset modulation_family
./.venv/bin/python experiments/followups/corruption_robustness/run.py --dataset waveform_family --train-regime augmented --results-path experiments/followups/corruption_robustness/results_augmented.json
./.venv/bin/python experiments/followups/corruption_robustness/run.py --train-regime augmented --datasets modulation_family subghz_real_512 captured_npy_real_128 orbit_rf --results-dir experiments/followups/corruption_robustness/results_by_dataset
```
