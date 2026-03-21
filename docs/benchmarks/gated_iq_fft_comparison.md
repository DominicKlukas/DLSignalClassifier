# IQ vs FFT vs Gated IQ+FFT Comparisons Across Datasets

This file collects the experiments that use the same core three-way comparison:

- IQ time-domain CNN
- FFT CNN
- gated IQ+FFT multimodal CNN

The entries below span any dataset where that same comparison is available, even when the result came from combining the IQ/FFT baseline run with the later gated-multimodal run on the same split.

It excludes experiments that changed the model family beyond that comparison, such as:

- plain concatenation-only multimodal as the main target
- wavelet-augmented models
- cross-attention models
- AWGN curve sweeps rather than dataset-level benchmark runs

## Synthetic Modulation Dataset

Source experiments:

- IQ and FFT baselines from [experiments.md](../experiments.md)
- gated multimodal result from [experiments.md](../experiments.md)

Dataset characteristics:

- time-feature-rich modulation classification
- classes: `BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `PAM4`
- same split used across the baseline and gated runs

Results:

- Time CNN: test `0.730`, macro F1 `0.727`
- FFT CNN: test `0.193`, macro F1 `0.149`
- Gated multimodal CNN: test `0.680`, macro F1 `0.676`
- Gate weights: best-val `[0.998, 0.001, 0.001]`, test `[0.998, 0.001, 0.001]`

Ranking:

- `Time CNN > Gated multimodal >> FFT CNN`

Interpretation:

- raw IQ is the correct representation for this task
- FFT is mostly unhelpful on its own
- the gated model learned to rely almost entirely on IQ, but still did not recover the pure time-CNN baseline

Artifacts:

- [experiment1_results.json](../experiments/exp01_iq_vs_fft/artifacts/experiment1_results.json)
- [experiment3_results.json](../experiments/exp02_gated_multimodal/artifacts/experiment3_results.json)

## Synthetic Waveform-Family Dataset

Source experiments:

- IQ and FFT baselines from [experiments.md](../experiments.md)
- gated multimodal result from [experiments.md](../experiments.md)

Dataset characteristics:

- frequency-feature-rich waveform-family classification
- classes: `CW`, `AM`, `FM`, `OFDM`, `LFM_CHIRP`, `DSSS`, `FHSS`, `SC_BURST`
- same split used across the baseline and gated runs

Results:

- Time CNN: test `0.598`, macro F1 `0.584`
- FFT CNN: test `0.728`, macro F1 `0.728`
- Gated multimodal CNN: test `0.908`, macro F1 `0.908`
- Gate weights: best-val `[0.117, 0.000, 0.882]`, test `[0.117, 0.000, 0.883]`

Ranking:

- `Gated multimodal >> FFT CNN > Time CNN`

Interpretation:

- the waveform-family task clearly benefits from multimodal fusion
- FFT is the stronger single representation
- the gated model leaned primarily on the fusion path, not the direct FFT-only head

Artifacts:

- [experiment1_results.json](../experiments/exp01_iq_vs_fft/artifacts/experiment1_results.json)
- [experiment3_results.json](../experiments/exp02_gated_multimodal/artifacts/experiment3_results.json)

## Real Sub-GHz Dataset Benchmark

Dataset:

- [train.h5](../data/real_mat_dataset/train.h5)
- [val.h5](../data/real_mat_dataset/val.h5)
- [test.h5](../data/real_mat_dataset/test.h5)

Setup:

- window length: `1024`
- split policy: file-level `60/20/20`
- normalization: per-window complex RMS
- at most `128` evenly spaced windows per source file
- source files: `75 / 25 / 25`
- windows: `9600 / 3200 / 3200`

Class set:

- `80211ah`
- `802154g`
- `Lora`
- `Noise`
- `Sigfox`
- `sunOFDM`

Implementation:

- [run.py](../experiments/legacy/exp06_real_h5/run.py)

Results:

- Time CNN: val `0.796`, test `0.814`, macro F1 `0.615`
- FFT CNN: val `0.756`, test `0.774`, macro F1 `0.409`
- Gated multimodal CNN: val `0.793`, test `0.816`, macro F1 `0.619`
- Gate weights: best-val `[0.353, 0.545, 0.101]`, test `[0.341, 0.535, 0.124]`

Ranking:

- `Gated multimodal ~= Time CNN > FFT CNN`

Artifact:

- [results_default.json](../experiments/legacy/exp06_real_h5/results_default.json)

## Real Sub-GHz Follow-Up: Larger Real-Data Subset

Same dataset, same model family, same split, with more windows per source file.

Setup:

- `512` evenly spaced windows per source file
- windows: `38400 / 12800 / 12800`

Results:

- Time CNN: val `0.801`, test `0.815`, macro F1 `0.621`
- FFT CNN: val `0.784`, test `0.819`, macro F1 `0.637`
- Gated multimodal CNN: val `0.804`, test `0.823`, macro F1 `0.645`
- Gate weights: best-val `[0.583, 0.257, 0.160]`, test `[0.569, 0.259, 0.172]`

Ranking:

- `Gated multimodal > FFT CNN ~= Time CNN`

Artifact:

- [results_medium.json](../experiments/legacy/exp06_real_h5/results_medium.json)

## Real Sub-GHz Follow-Up: Even Larger Subset, 40 Epochs

Same dataset and same three-model comparison again, with broader coverage and longer training.

Setup:

- `1024` evenly spaced windows per source file
- `40` training epochs
- windows: `76800 / 25600 / 25600`

Results:

- Time CNN: val `0.791`, test `0.812`, macro F1 `0.613`
- FFT CNN: val `0.776`, test `0.816`, macro F1 `0.638`
- Gated multimodal CNN: val `0.798`, test `0.821`, macro F1 `0.648`
- Gate weights: best-val `[0.507, 0.120, 0.373]`, test `[0.504, 0.119, 0.377]`

Ranking:

- `Gated multimodal > FFT CNN > Time CNN`

Artifact:

- [results_large.json](../experiments/legacy/exp06_real_h5/results_large.json)

## Augmented Real Sub-GHz Dataset Benchmark

Dataset:

- [train.h5](../data/real_mat_dataset_augmented/train.h5)
- [val.h5](../data/real_mat_dataset_augmented/val.h5)
- [test.h5](../data/real_mat_dataset_augmented/test.h5)

Setup:

- derived from the real Sub-GHz benchmark
- file-level split preserved exactly
- `512` evenly spaced windows per source file
- window length: `1024`
- normalization: post-augmentation complex RMS
- added frequency shift, sample-rate scaling, and AWGN
- windows: `38400 / 12800 / 12800`

Implementation:

- [run.py](../experiments/legacy/exp06_real_h5/run.py)

Results:

- Time CNN: val `0.777`, test `0.797`, macro F1 `0.520`
- FFT CNN: val `0.771`, test `0.775`, macro F1 `0.524`
- Gated multimodal CNN: val `0.800`, test `0.826`, macro F1 `0.660`
- Gate weights: best-val `[0.278, 0.374, 0.348]`, test `[0.249, 0.371, 0.380]`

Ranking:

- `Gated multimodal > Time CNN > FFT CNN`

Interpretation:

- corruption increased the value of fusion
- the gated model opened a clearer gap over both single-mode baselines than on the clean benchmark

Artifact:

- [results_augmented.json](../experiments/legacy/exp06_real_h5/results_augmented.json)

## Orbit RF Identification Dataset

This is a different real RF task, but it uses the same three-way comparison.

Dataset:

- [orbit_rf_identification_dataset_updated](../orbit_rf_identification_dataset_updated)

Setup:

- real WiFi preamble captures
- packet length: `256` IQ samples
- task: classify transmitter identity
- train: first `3` days
- val: fourth day
- test: fifth day
- packet cap: at most `256` packets per transmitter per day
- sizes: `43742 / 14372 / 14643`

Implementation:

- [run.py](../experiments/legacy/exp05_orbit_rf/run.py)

Results:

- Time CNN: val `0.767`, test `0.634`, macro F1 `0.558`
- FFT CNN: val `0.794`, test `0.719`, macro F1 `0.665`
- Gated multimodal CNN: val `0.776`, test `0.707`, macro F1 `0.644`
- Gate weights: best-val `[0.322, 0.000, 0.678]`, test `[0.328, 0.000, 0.671]`

Ranking:

- `FFT CNN > Gated multimodal > Time CNN`

Artifact:

- [results.json](../experiments/legacy/exp05_orbit_rf/results.json)

## Captured `.npy` Dataset Re-Run

Dataset:

- [CapturedData/dataset](../CapturedData/dataset)
- [metadata.csv](../CapturedData/dataset/metadata.csv)

The captured recordings were converted into Experiment 5-style HDF5 splits using:

- [convert_to_h5.py](../dataset_prep/owned/captured/convert_to_h5.py)

Generated dataset artifacts:

- [train.h5](../data/captured_npy_dataset_experiment5/train.h5)
- [val.h5](../data/captured_npy_dataset_experiment5/val.h5)
- [test.h5](../data/captured_npy_dataset_experiment5/test.h5)

Setup:

- source recordings: `1095`
- class count: `8`
- classes: `ais`, `am`, `aprs`, `atc_voice`, `fm`, `noaa`, `noise`, `vhf`
- window length: `1024`
- split policy: file-level `60/20/20`
- normalization: per-window complex RMS
- at most `128` evenly spaced non-overlapping windows per source file
- source files: `657 / 219 / 219`
- windows: `84096 / 28032 / 28032`

Implementation:

- [run.py](../experiments/legacy/exp06_real_h5/run.py)

Results:

- Time CNN: val `0.949`, test `0.957`, macro F1 `0.968`
- FFT CNN: val `0.824`, test `0.826`, macro F1 `0.832`
- Gated multimodal CNN: val `0.959`, test `0.973`, macro F1 `0.980`
- Gate weights: best-val `[0.331, 0.096, 0.573]`, test `[0.328, 0.104, 0.568]`

Ranking:

- `Gated multimodal > Time CNN >> FFT CNN`

Artifact:

- [results_captured_npy.json](../experiments/legacy/exp06_real_h5/results_captured_npy.json)

## High-Level Pattern

Across the datasets using this same comparison, the ordering is task-dependent:

- modulation-family: IQ is best and fusion does not recover the IQ baseline
- waveform-family: gated multimodal is dominant
- clean real Sub-GHz protocol classification: gated multimodal is usually best, but often only slightly
- augmented real Sub-GHz: gated multimodal pulls clearly ahead
- Orbit RF transmitter identification: FFT is best
- captured `.npy` benchmark: gated multimodal shows the strongest win of the group

So the shared comparison does not produce one universal ranking across every dataset. The narrower conclusion is:

- gated IQ+FFT fusion is often the best overall model
- IQ is strongest when time-domain structure dominates
- FFT is strongest when spectral identity dominates
- the value of fusion grows when the task genuinely contains complementary time and frequency cues
