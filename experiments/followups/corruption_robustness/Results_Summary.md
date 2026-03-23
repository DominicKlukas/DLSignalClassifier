# Corruption Robustness Results Summary

This note summarizes the corruption-augmented training results from the corruption robustness follow-up. The main question was whether multiview IQ+FFT models become more useful once they are trained on a mixture of clean and corrupted examples rather than being asked to generalize to corruption entirely out of distribution.

## Scope

- Training regime: `augmented`
- Corruption families:
  - `awgn`
  - `impulse_noise`
  - `narrowband_interferer`
  - `sample_rate_distortion`
  - `clipping`
- Models:
  - `iq_cnn`
  - `fft_cnn`
  - `gated_multimodal_cnn`
  - `frozen_expert_residual_fusion`

Datasets covered here:

- `waveform_family`
- `modulation_family`
- `subghz_real_512`
- `captured_npy_real_128`
- `orbit_rf`

Primary artifacts:

- `results_augmented.json`
- `results_by_dataset/batch_summary_augmented.json`
- `results_by_dataset/<dataset>/results_augmented.json`

## Main Takeaways

1. Corruption augmentation changed the story substantially relative to clean-train evaluation.
2. Multiview fusion is not uniformly beneficial; its value depends strongly on the dataset.
3. `gated_multimodal_cnn` was the strongest model on `waveform_family`, `subghz_real_512`, and `captured_npy_real_128`.
4. `frozen_expert_residual_fusion` was strongest on `orbit_rf` and improved over the best single expert on every tested corruption there.
5. `modulation_family` remained IQ-dominant even with corruption augmentation; both fusion variants stayed below the IQ expert.

## Dataset-Level Summary

### `waveform_family`

Artifact:

- `results_augmented.json`

Summary:

- Frozen residual beat the best single expert on all five corruption families.
- Average frozen-residual delta vs best single expert: `+0.0415`
- Gated fusion was stronger than frozen residual on every corruption family.

Representative numbers:

- `awgn`: FFT `0.568`, gated `0.685`, frozen `0.623`, frozen delta `+0.055`
- `impulse_noise`: FFT `0.614`, gated `0.734`, frozen `0.661`, frozen delta `+0.047`
- `sample_rate_distortion`: FFT `0.666`, gated `0.786`, frozen `0.710`, frozen delta `+0.045`

Interpretation:

- This is the clearest case that corruption-aware multiview training can exploit view complementarity.

### `modulation_family`

Artifact:

- `results_by_dataset/modulation_family/results_augmented.json`

Summary:

- Frozen residual lost to the best single expert on all five corruption families.
- Average frozen-residual delta vs best single expert: `-0.0828`
- IQ remained the dominant view throughout.

Representative numbers:

- `awgn`: IQ `0.532`, gated `0.485`, frozen `0.457`, frozen delta `-0.075`
- `impulse_noise`: IQ `0.583`, gated `0.527`, frozen `0.473`, frozen delta `-0.109`
- `sample_rate_distortion`: IQ `0.621`, gated `0.589`, frozen `0.493`, frozen delta `-0.129`

Interpretation:

- This dataset still behaves like an IQ-first task. Corruption augmentation was not enough to create a meaningful advantage for IQ+FFT fusion.

### `subghz_real_512`

Artifact:

- `results_by_dataset/subghz_real_512/results_augmented.json`

Summary:

- Frozen residual was nearly neutral overall.
- Average frozen-residual delta vs best single expert: `+0.0011`
- Gated fusion was best or tied-best on every corruption family.

Representative numbers:

- `awgn`: IQ `0.796`, gated `0.800`, frozen `0.790`, frozen delta `-0.006`
- `sample_rate_distortion`: IQ `0.799`, gated `0.812`, frozen `0.808`, frozen delta `+0.010`
- `clipping`: IQ `0.816`, gated `0.820`, frozen `0.819`, frozen delta `+0.003`

Interpretation:

- The corruption-aware multiview story is plausible here, but the effect is modest. This looks more like a “small adaptive benefit” dataset than a dramatic complementarity dataset.

### `captured_npy_real_128`

Artifact:

- `results_by_dataset/captured_npy_real_128/results_augmented.json`

Summary:

- Frozen residual was approximately neutral overall.
- Average frozen-residual delta vs best single expert: `-0.0016`
- Gated fusion was clearly strongest across all corruption families.

Representative numbers:

- `awgn`: IQ `0.772`, gated `0.823`, frozen `0.775`, frozen delta `+0.004`
- `impulse_noise`: IQ `0.914`, gated `0.934`, frozen `0.911`, frozen delta `-0.002`
- `sample_rate_distortion`: IQ `0.895`, gated `0.932`, frozen `0.899`, frozen delta `+0.004`

Interpretation:

- On this captured benchmark, the flexible gated fusion model appears much better suited than the conservative residual-anchor model.

### `orbit_rf`

Artifact:

- `results_by_dataset/orbit_rf/results_augmented.json`

Summary:

- Frozen residual beat the best single expert on all five corruption families.
- Average frozen-residual delta vs best single expert: `+0.0203`
- Frozen residual was also stronger than gated fusion on every tested corruption family.

Representative numbers:

- `awgn`: gated `0.534`, frozen `0.538`, frozen delta `+0.015`
- `impulse_noise`: IQ `0.665`, gated `0.678`, frozen `0.697`, frozen delta `+0.019`
- `clipping`: FFT `0.687`, gated `0.689`, frozen `0.713`, frozen delta `+0.026`

Interpretation:

- This is the strongest evidence in favor of the frozen residual architecture under corruption-aware training.

## Cross-Dataset Interpretation

The corruption-augmented results support a more precise claim than "fusion always helps."

- Fusion helps when the task contains exploitable view complementarity under corruption.
- The best fusion architecture depends on the task.
- Gated fusion seems better when the problem rewards aggressive, input-dependent adaptation.
- Frozen residual seems better when a stable expert-preserving correction is enough, as in this first `orbit_rf` run.
- Some tasks remain essentially single-view even after augmentation, with `modulation_family` as the clearest example.

## Suggested Paper Framing

- Use `waveform_family` as the cleanest synthetic demonstration that corruption-aware multiview learning can pay off.
- Use `orbit_rf` as the strongest frozen-residual success case.
- Use `subghz_real_512` and `captured_npy_real_128` as evidence that the effect can be modest or architecture-dependent on real data.
- Use `modulation_family` as an explicit negative/control case showing that fusion is not universally beneficial.

## Runtime Note

The four-dataset augmented batch (`modulation_family`, `subghz_real_512`, `captured_npy_real_128`, `orbit_rf`) completed in about `840.9` seconds, or roughly `14.0` minutes, as recorded in `results_by_dataset/batch_summary_augmented.json`.
