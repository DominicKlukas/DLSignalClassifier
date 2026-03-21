# Experiment 11: Frozen-Expert Residual Fusion Across Comparable Datasets

## Goal

Test a safer multimodal architecture across the same dataset family collected in:

- [gated_iq_fft_comparison.md](gated_iq_fft_comparison.md)

The target hypothesis was:

- adding the second modality should never make the model worse than the best single-mode expert in the same run

This is not a claim of formal proof. It is an empirical test of whether a more conservative architecture can realize that behavior in practice.

## Architecture

Implemented in:

- [run.py](../experiments/exp03_frozen_expert_residual/run.py)

Design:

- train an IQ expert normally
- train an FFT expert normally
- freeze both experts
- choose an anchor expert per sample using the higher expert confidence
- predict a bounded residual correction on top of the anchor logits

Form:

- `final_logits = anchor_logits + alpha * delta`

where:

- `anchor_logits` comes from the frozen IQ or FFT expert
- `alpha` is a learned scalar in `[0, 1]`
- `delta` is a bounded residual correction

Conservative regularization:

- penalty on residual activation magnitude
- penalty on mean `alpha`

The intent was to let fusion help when useful, while preserving an exact fallback path to a single expert.

## Datasets Included

This sweep used the same comparison family across:

- synthetic modulation-family
- synthetic waveform-family
- clean real Sub-GHz with `128` windows per file
- clean real Sub-GHz with `512` windows per file
- clean real Sub-GHz with `1024` windows per file and `40` epochs
- augmented real Sub-GHz with `512` windows per file
- Orbit RF identification
- captured `.npy` real benchmark

## Results

### Synthetic Modulation

- IQ expert: `0.737`
- FFT expert: `0.177`
- frozen-expert residual fusion: `0.747`
- delta vs best single: `+0.010`

### Synthetic Waveform-Family

- IQ expert: `0.575`
- FFT expert: `0.750`
- frozen-expert residual fusion: `0.885`
- delta vs best single: `+0.135`

### Real Sub-GHz, 128 Windows/File

- IQ expert: `0.807`
- FFT expert: `0.764`
- frozen-expert residual fusion: `0.813`
- delta vs best single: `+0.006`

### Real Sub-GHz, 512 Windows/File

- IQ expert: `0.818`
- FFT expert: `0.816`
- frozen-expert residual fusion: `0.823`
- delta vs best single: `+0.005`

### Real Sub-GHz, 1024 Windows/File, 40 Epochs

- IQ expert: `0.814`
- FFT expert: `0.818`
- frozen-expert residual fusion: `0.821`
- delta vs best single: `+0.003`

### Augmented Real Sub-GHz, 512 Windows/File

- IQ expert: `0.796`
- FFT expert: `0.796`
- frozen-expert residual fusion: `0.812`
- delta vs best single: `+0.015`

### Orbit RF

- IQ expert: `0.668`
- FFT expert: `0.682`
- frozen-expert residual fusion: `0.693`
- delta vs best single: `+0.011`

### Captured `.npy` Real Benchmark

- IQ expert: `0.961`
- FFT expert: `0.828`
- frozen-expert residual fusion: `0.972`
- delta vs best single: `+0.011`

## Interpretation

In this run, the empirical hypothesis held across all included datasets:

- the frozen-expert residual fusion model matched or exceeded the best single expert every time
- in every dataset here, it strictly exceeded the best single expert by a small or large margin

The size of the gain depended strongly on the task:

- modulation-family: only a very small gain
- waveform-family: a very large gain
- clean real Sub-GHz: small but consistent gains
- augmented real Sub-GHz: a clearer gain
- Orbit RF: a modest gain over the FFT expert
- captured `.npy` real benchmark: a solid gain over the IQ expert

The learned residual strength also fit the intuition behind the architecture:

- on harder or more complementary tasks, the model used larger residual corrections
- on the large real Sub-GHz run, the residual stayed very small and the model behaved almost like a safe expert fallback

So the design appears to do what we wanted in practice:

- preserve strong single experts
- allow fusion to help without forcing full symmetric mixing

## Important Caveat

This does **not** prove a universal guarantee in the mathematical sense.

What it shows is:

- with this training recipe and these datasets, the conservative residual-fusion design was empirically better than the best single expert on every tested benchmark

That is much stronger than the earlier symmetric gated model, but it is still an experimental result, not a theorem.

## Conclusion

Compared with the earlier gated multimodal architecture, the frozen-expert residual design is a clear improvement for this repo’s comparison family.

The earlier gated model often helped, but did not reliably dominate the strongest single-mode baseline.

This new design did better in the full multi-dataset sweep:

- it beat the best individual expert on all included datasets in this run
- it produced especially large gains where multimodal complementarity was strong
- and it degraded gracefully on datasets where one expert already dominated

## Output Artifact

Machine-readable results were saved to:

- [experiment11_frozen_expert_residual_multidataset_results.json](../experiments/exp03_frozen_expert_residual/artifacts/experiment11_frozen_expert_residual_multidataset_results.json)

## Addendum: RadioML 2018 Balanced Subset

Follow-up run using the same frozen-expert residual architecture on:

- [GOLD_XYZ_OSC.0001_1024.hdf5](../external/radioml_dataset/GOLD_XYZ_OSC.0001_1024.hdf5)

Because the full RadioML 2018 dataset contains `2555904` windows, this follow-up used a reproducible balanced subset instead of the full corpus:

- `24` classes
- `26` SNR levels from `-20` to `30` dB in steps of `2`
- per `(class, SNR)` cell: `128` train, `32` val, `64` test
- total windows: `79872 / 19968 / 39936`
- window length: `1024`
- batch size: `256`
- epochs: `20`

Important caveat:

- class names follow the public `RML2018.01A` mirrored `classes.txt` ordering
- some third-party analysis reports that this ordering may not perfectly match the HDF5 label indices

Results:

- IQ expert: `0.444`
- FFT expert: `0.226`
- frozen-expert residual fusion: `0.457`
- delta vs best single: `+0.013`
- test alpha mean: `0.437`
- test IQ-anchor fraction: `0.869`

Interpretation:

- this behaves much more like the synthetic modulation-family task than the waveform-family task
- the IQ expert is clearly dominant over the FFT expert
- the residual fusion model still improved over the best single expert, but only modestly
- the high IQ-anchor fraction shows the fusion model usually trusted the IQ expert and applied a moderate residual correction on top

Artifact:

- [results.json](../experiments/followups/radioml2018/results.json)

## Addendum: RadioML 2018 Full-Dataset Streaming Run

The earlier balanced-subset addendum was a quick feasibility check. We then updated the runner to stream directly from the HDF5 so the same Experiment 11 architecture could train on the entire dataset with a long schedule:

- [run.py](../experiments/followups/radioml2018/run.py)

Dataset:

- [GOLD_XYZ_OSC.0001_1024.hdf5](../external/radioml_dataset/GOLD_XYZ_OSC.0001_1024.hdf5)

Split and training protocol:

- full `RML2018.01A` corpus used
- `4096` examples per `(class, SNR)` cell
- fixed-seed random split per cell: `3276 / 410 / 410`
- total windows: `2044224 / 255840 / 255840`
- `24` classes
- `26` SNR levels from `-20` to `30` dB
- signal length: `1024`
- batch size: `256`
- epochs: `40`
- runtime: about `5446` seconds

Important caveat:

- class names still follow the public mirrored `RML2018.01A` `classes.txt` ordering
- some third-party analysis reports that this ordering may not perfectly match the HDF5 label indices

Results:

- IQ expert: `0.610`
- FFT expert: `0.303`
- frozen-expert residual fusion: `0.614`
- delta vs best single: `+0.004`
- test alpha mean: `0.414`
- test IQ-anchor fraction: `0.840`

Interpretation:

- the full-data long-run result is much stronger than the earlier balanced-subset run
- on this benchmark, the architecture is still clearly IQ-dominant rather than balanced between modalities
- the residual fusion model again beats the best single expert, but now only by a very small margin
- so the main correction to the earlier interpretation is not that fusion suddenly becomes dominant, but that the IQ expert itself becomes substantially stronger when given the whole dataset and a long schedule

Artifact:

- [results.json](../experiments/followups/radioml2018/results.json)
