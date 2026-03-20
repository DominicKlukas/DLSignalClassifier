# Experiments

## Overview

This repository evolved from clean synthetic modulation classification into a set of robustness experiments motivated by out-of-distribution SDR generalization. The main themes were:

- robustness to global phase shifts,
- robustness to frequency offsets,
- robustness to sample-rate scaling,
- whether architectural bias helps beyond augmentation.

The sections below summarize the motivation, hypothesis, concrete result, and conclusion for each experiment family that has been discussed and/or run so far.

## 1. Clean Synthetic Modulation Classification

### Motivation

The original task was IQ modulation classification on synthetic signals generated from clean constellations, RRC pulse shaping, AWGN, random global phase shifts, and frequency offsets.

### Hypothesis

A vanilla CNN should perform strongly on this synthetic problem, but the benchmark may be too easy to say much about real-world robustness.

### Implementations

- `train_cnn.ipynb`
- `train_fourier_cnn.ipynb`
- `train_spectrograph_cnn.ipynb`
- `train_phase_equivariant_cnn.ipynb`

### Result

Saved checkpoint evaluation on the held-out clean synthetic test split gave:

- Vanilla time-domain CNN
  - validation accuracy: `0.823`
  - test accuracy: `0.796`
  - macro F1: `0.797`
- Fourier 1D CNN
  - validation accuracy: `0.179`
  - test accuracy: `0.170`
  - macro F1: `0.082`
- Spectrogram CNN
  - validation accuracy: `0.425`
  - test accuracy: `0.419`
  - macro F1: `0.413`
- Phase-equivariant CNN
  - validation accuracy: `0.792`
  - test accuracy: `0.794`
  - macro F1: `0.797`

The clean dataset therefore strongly favored the raw-IQ and phase-equivariant models, while the simple Fourier and spectrogram baselines underperformed badly on this task.

### Conclusion

The frequency domain is not helpful if the frequency content of the different classes of signals is all the same. On top of that, the CNN works very well as if the data is already heavily augmented. What's the problem that we are trying to solve in the first place?

## 2. Harder Synthetic Modulation Robustness Benchmark

### Motivation

To make the modulation problem more realistic, a harder generator was built with:

- timing offset,
- sample-rate mismatch,
- burst truncation,
- multipath,
- IQ imbalance,
- DC offset,
- colored noise,
- nonlinearity,
- optional interferers.


### Hypothesis

A harder synthetic benchmark should expose failure modes that the clean modulation dataset hides, and allow comparison between:

- vanilla CNN,
- vanilla CNN + augmentation,
- phase-equivariant front end.

### Implementations

- `ChallengingSignalGenerator.py`
- `generate_challenging_dataset.py`
- `train_challenging_comparison.ipynb`
- `ablate_signal_degradations.ipynb`

### Result

Saved checkpoint evaluation on the harder modulation benchmark gave:

- Vanilla CNN without augmentation
  - validation accuracy: `0.461`
  - test accuracy: `0.423`
  - macro F1: `0.412`
- Vanilla CNN with augmentation
  - validation accuracy: `0.459`
  - test accuracy: `0.431`
  - macro F1: `0.417`
- Phase-equivariant augmented CNN
  - validation accuracy: `0.465`
  - test accuracy: `0.446`
  - macro F1: `0.421`

The improvement from architectural bias and augmentation was real but modest on this harder benchmark.

The ablation checkpoint from the clean-trained vanilla model on the challenging dataset identified the three most damaging degradation-focused subsets as:

- `low_snr`
  - subset accuracy: `0.169`
  - drop vs clean-ish subset: `0.045`
  - samples: `2250`
- `high_freq_offset`
  - subset accuracy: `0.174`
  - drop vs clean-ish subset: `0.040`
  - samples: `2250`
- `interferer`
  - subset accuracy: `0.174`
  - drop vs clean-ish subset: `0.040`
  - samples: `2341`

### Conclusion

This branch is ready for future runs, but it is not the source of the quantitative conclusions below.

## 3. Waveform-Family Dataset

### Motivation

Scale and frequency invariance are a poor fit for modulation classification because the task depends heavily on time-domain symbol structure. A waveform-family task is a better match for frequency-domain inductive biases.

### Classes

- `CW`
- `AM`
- `FM`
- `OFDM`
- `LFM_CHIRP`
- `DSSS`
- `FHSS`
- `SC_BURST`

### Hypothesis

Frequency-domain models should transfer better across frequency shifts and possibly sample-rate distortions than time-domain models.

### Implementations

- `WaveformFamilyGenerator.py`
- `generate_waveform_family_dataset.py`
- `train_waveform_family_cnn.py`

### Conclusion

This dataset became the main platform for the OOD experiments below.

## 4. Frequency-Offset OOD With FFT CNN

### Motivation

Train on waveform-family signals without center-frequency offsets and test on signals with frequency offsets.

### Hypothesis

An FFT-based CNN should be relatively robust to frequency offsets because the class signal is largely spectral.

### Protocol

- Train dataset: `waveform_family_no_offset_dataset.h5`
- Test dataset: `waveform_family_dataset.h5`
- Model: `train_waveform_family_ood_frequency_fft.py`

### Result

Best validation accuracy: `0.941`

OOD test accuracy: `0.749`

Macro F1: `0.70`

Weighted F1: `0.71`

Per-class behavior:

- Very strong: `DSSS 1.00`, `OFDM 0.99`, `FM 0.91`, `SC_BURST 0.91`
- Weak: `AM 0.28`, `CW 0.21`
- Moderate: `FHSS 0.69`, `LFM_CHIRP 0.64`

### Conclusion

FFT features help substantially for frequency-offset transfer, but the model still confuses some narrowband spectral classes:

- `AM` often collapses into `FHSS`, `FM`, or `LFM_CHIRP`
- `CW` often collapses into `LFM_CHIRP` and `FHSS`

The result is useful but not invariant in a strong sense.

## 5. Sample-Rate OOD: Early Exploratory Run

### Motivation

Before the exact-scale datasets were introduced, an FFT CNN was trained for the sample-rate OOD problem.

### Hypothesis

FFT-domain processing might already be robust to moderate sampling distortions.

### Protocol

- Train dataset: `waveform_family_no_rate_offset_dataset.h5`
- Test dataset: earlier `waveform_family_dataset.h5`
- Model: `train_waveform_family_ood_sample_rate_fft.py`

### Result

Best validation accuracy: `0.919`

OOD test accuracy: `0.916`

Macro F1: `0.92`

### Conclusion

This run became obsolete after the generator was refactored so that `waveform_family_dataset.h5` no longer carried explicit sample-rate scaling. The result is therefore not the corrected sample-rate-scale benchmark and should not be used as the main conclusion.

## 6. Corrected Wideband Sample-Rate-Scale OOD

### Motivation

Create an explicit sample-rate-scale benchmark with exact scales and compare:

- plain FFT CNN,
- scale-group FFT CNN.

### Hypothesis

A shared-weight scale-group model over FFT spectra should outperform a plain FFT CNN when tested on exact scale factors.

### Datasets

- Train: `waveform_family_no_rate_offset_dataset.h5` with scale `1.0`
- Test: `waveform_family_rate_scaled_dataset.h5` or `waveform_family_exact_scale_dataset.h5`
- Exact test scales: `1.0`, `1.33`, `2.0`, `4.0`

### Models

- `train_waveform_family_ood_sample_rate_fft.py`
- `train_waveform_family_ood_sample_rate_scale_group_fft.py`
- `test_waveform_family_exact_scale_group_fft.py`

### Plain FFT CNN Result

Best validation accuracy: `0.925`

OOD test accuracy on rate-scaled dataset: `0.558`

Macro F1: `0.53`

Main confusions:

- `DSSS` almost entirely collapses into `OFDM` and `SC_BURST`
- `OFDM` often collapses into `SC_BURST`
- `CW` remains easy, with recall `0.94`

### Scale-Group FFT CNN Result

Best validation accuracy: `0.955`

OOD test accuracy on rate-scaled dataset: `0.570`

Macro F1: `0.54`

This is only a marginal improvement over the plain FFT CNN.

### Exact-Scale Evaluation Of The Scale-Group FFT CNN

Overall exact-scale accuracy: `0.655`

Per-scale accuracies:

- `1.00x`: `0.938`
- `1.33x`: `0.776`
- `2.00x`: `0.570`
- `4.00x`: `0.336`

### Conclusion

The wideband exact-scale experiment showed:

- strong performance at `1.0x`,
- moderate degradation at `1.33x`,
- large degradation at `2.0x`,
- catastrophic degradation at `4.0x`.

The scale-group architecture did not provide anything close to true scale invariance. The best interpretation is:

- it is an approximate scale-channel model,
- not an exact scale-equivariant model,
- and the applied data transformation is more destructive than a clean theoretical group action on ideal spectra.

## 7. Narrowband Exact-Scale Rerun

### Motivation

The previous wideband sample-rate experiment likely violated the intended invariance assumptions because the `4x` transform pushed too much energy and structure into a distorted regime. To reduce this mismatch, the experiment was recreated so that even at the largest scale the occupied bandwidth stayed below one quarter of the full sampled bandwidth.

### Hypothesis

Restricting all waveform families to narrower bandwidth should make sample-rate scaling more nearly label-preserving and improve the scale-group model’s transfer.

### Datasets

- Train: `waveform_family_narrowband_no_rate_offset_dataset.h5`
- Test: `waveform_family_narrowband_exact_scale_dataset.h5`
- Occupied bandwidth bounds: `80 Hz` to `400 Hz`
- Exact scales: `1.0`, `1.33`, `2.0`, `4.0`

### Model

- `train_waveform_family_narrowband_ood_sample_rate_scale_group_fft.py`
- `test_waveform_family_narrowband_exact_scale_group_fft.py`

### Result

Best validation accuracy: `0.966`

Overall OOD test accuracy: `0.730`

Macro F1: `0.74`

Per-scale accuracies:

- `1.00x`: `0.963`
- `1.33x`: `0.888`
- `2.00x`: `0.689`
- `4.00x`: `0.379`

### Comparison Against The Wideband Scale-Group Run

Wideband exact-scale overall: `0.655`

Narrowband exact-scale overall: `0.730`

Wideband per-scale:

- `1.00x`: `0.938`
- `1.33x`: `0.776`
- `2.00x`: `0.570`
- `4.00x`: `0.336`

Narrowband per-scale:

- `1.00x`: `0.963`
- `1.33x`: `0.888`
- `2.00x`: `0.689`
- `4.00x`: `0.379`

### Conclusion

This was the clearest positive result in the scale experiments:

- narrowing the waveform bandwidth materially improved transfer at every scale,
- especially at `1.33x` and `2.0x`,
- but `4.0x` remained hard.

This strongly suggests the earlier failures were not only architectural. A large part of the rolloff came from the transformation itself becoming too destructive for the chosen data distribution.

## 8. Narrowband Scale-Channel Pilot

### Motivation

Older scale-equivariant models often generalize poorly outside the sampled scale range. Scale-channel networks are designed to generalize better to previously unseen scales by building a multi-scale representation and pooling over scale channels, rather than learning convolutions over the scale axis itself.

### Hypothesis

On the narrowband exact-scale benchmark, a scale-channel FFT model should extrapolate better than the earlier scale-group FFT model, especially when trained only on `1.0x` signals and tested zero-shot on `1.33x`, `2.0x`, and `4.0x`.

### Protocol

- Train: `waveform_family_narrowband_no_rate_offset_dataset.h5`
- Test: `waveform_family_narrowband_exact_scale_dataset.h5`
- Model: `train_waveform_family_narrowband_ood_sample_rate_scale_channel_fft.py`
- Evaluation: `test_waveform_family_narrowband_exact_scale_channel_fft.py`
- Internal scale-channel grid: `0.75`, `1.0`, `1.33`, `2.0`, `4.0`

To keep the experiment runnable in this environment, the FFT input was reduced from `1024` bins to `256` bins before the scale-channel stack, and the pilot was trained for `20` epochs.

### Result

Best validation accuracy: `0.590`

Overall exact-scale OOD accuracy: `0.386`

Macro F1: `0.40`

Per-scale accuracies:

- `1.00x`: `0.602`
- `1.33x`: `0.465`
- `2.00x`: `0.317`
- `4.00x`: `0.160`

### Comparison Against Narrowband Scale-Group FFT

Narrowband scale-group FFT:

- overall: `0.730`
- `1.00x`: `0.963`
- `1.33x`: `0.888`
- `2.00x`: `0.689`
- `4.00x`: `0.379`

Narrowband scale-channel pilot:

- overall: `0.386`
- `1.00x`: `0.602`
- `1.33x`: `0.465`
- `2.00x`: `0.317`
- `4.00x`: `0.160`

### Conclusion

This pilot did not support the hypothesis. In this implementation, the scale-channel model performed substantially worse than the earlier narrowband scale-group model at every scale.

The most likely reasons are:

- the quick scale-channel implementation was much shallower and more aggressively pooled than the scale-group model,
- the pilot used a compressed `256`-bin FFT representation to keep runtime manageable on CPU,
- simple average/max pooling over scale channels may have thrown away too much structure too early.

So the conclusion is not that scale-channel networks are ineffective in principle. The conclusion is narrower:

- this first scale-channel implementation did not beat the existing narrowband scale-group baseline,
- and a more faithful SCN-style implementation would need a stronger per-scale backbone and less destructive pooling.

## 9. Cross-Experiment Conclusions

### Main Lessons

1. Clean synthetic modulation classification is too easy to support strong claims about robustness.
2. Frequency-offset robustness is much easier to obtain in the waveform-family FFT setting than sample-rate-scale robustness.
3. A plain FFT CNN is already strong for frequency-offset OOD.
4. The initial scale-group FFT implementation did not achieve true scale invariance.
5. Narrowing the signal bandwidth improves approximate scale robustness substantially.
6. The first narrowband scale-channel pilot underperformed the scale-group baseline, so the architectural idea is still unproven in this codebase.
7. The remaining drop at `4x` indicates that:
   - the transformation is still not perfectly label-preserving, and/or
   - the implemented scale-group action is still only an approximation of the real data transformation.

### Recommended Next Steps

1. Rebuild the scale-channel model with a stronger per-scale backbone and delayed scale pooling, instead of the lightweight pilot used here.
2. Compare plain FFT CNN vs scale-group FFT CNN vs scale-channel FFT on the narrowband exact-scale benchmark with per-scale confusion matrices side by side.
3. Train on a subset of scales such as `1.0`, `1.33`, `2.0` and test on `4.0` only, to separate interpolation from extrapolation.
4. Move from linear-frequency FFT inputs to log-frequency or wavelet-like representations.
5. Revisit the scale action so the model-side transform matches the data-generation transform more faithfully.
