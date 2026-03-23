# Corruption Robustness Follow-Up

This follow-up tests a clean-train / corrupted-test setting for the repo's IQ, FFT, and fusion models.

Current scope:

- default dataset: synthetic `waveform_family`
- training: clean train/validation splits only
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

- `results.json`: machine-readable metrics and summaries
- `plots/`: robustness plots by corruption family

Run:

```bash
./.venv/bin/python experiments/followups/corruption_robustness/run.py
```

Useful options:

```bash
./.venv/bin/python experiments/followups/corruption_robustness/run.py --dataset waveform_family --seed 0
./.venv/bin/python experiments/followups/corruption_robustness/run.py --dataset modulation_family
```
