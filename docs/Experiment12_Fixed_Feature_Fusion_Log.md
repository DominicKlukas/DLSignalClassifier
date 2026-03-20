# Experiment 12: Fixed Hidden-Feature Fusion Depth Study

## Goal

Test which hidden feature depth from frozen IQ and FFT experts is most useful as the input to a new fixed fusion head.

The experiment design was:

- train an IQ expert to completion
- train an FFT expert to completion
- freeze both experts
- tap hidden features at several depths
- train the same fusion head on top of those frozen tapped features
- compare which tap level works best

Datasets:

- waveform-family
- clean real Sub-GHz benchmark with `128` windows per file

We also compare the best fixed-feature fusion result against:

- the best single expert in the same run
- the frozen-expert residual mixture model from Experiment 11

## Implementation

Implemented in:

- [run_experiment12_fixed_feature_fusion.py](/home/klukasdh/Projects/DLSignalClassifier/experiments/run_experiment12_fixed_feature_fusion.py)

Results artifact:

- [experiment12_fixed_feature_fusion_results.json](/home/klukasdh/Projects/DLSignalClassifier/experiments/experiment12_fixed_feature_fusion_results.json)

Reference mixture-of-experts result:

- [experiment11_frozen_expert_residual_multidataset_results.json](/home/klukasdh/Projects/DLSignalClassifier/experiments/experiment11_frozen_expert_residual_multidataset_results.json)

## Architecture

Expert backbone taps:

- `block1`: `32` channels
- `block2`: `64` channels
- `block3`: `128` channels
- `final`: `256` channels

For each tap:

- global-average-pool the tapped feature map
- project IQ features to a fixed `128`-dim vector
- project FFT features to a fixed `128`-dim vector
- concatenate the two projected vectors
- apply the same MLP head for classification

Important fairness constraint:

- the fusion head architecture after projection stayed the same across all tap levels
- only the frozen feature depth changed

## Waveform-Family Results

Single experts:

- IQ expert: `0.598`
- FFT expert: `0.698`
- best single expert: `0.698`

Fixed-feature fusion by tap:

- `block1`: `0.480`
- `block2`: `0.753`
- `block3`: `0.873`
- `final`: `0.878`

Best tap:

- `final`
- test acc: `0.878`
- delta vs best single expert: `+0.180`

Comparison to the Experiment 11 residual mixture model:

- residual mixture: `0.885`
- best fixed-feature tap: `0.878`
- gap: `-0.008`

Interpretation:

- very early features were not good fusion inputs here
- middle-to-late features were dramatically better
- the latest pooled representation was best
- but the residual mixture model still held a small edge over pure fixed-feature fusion

## Sub-GHz Real Results

Single experts:

- IQ expert: `0.808`
- FFT expert: `0.775`
- best single expert: `0.808`

Fixed-feature fusion by tap:

- `block1`: `0.803`
- `block2`: `0.806`
- `block3`: `0.813`
- `final`: `0.814`

Best tap:

- `final`
- test acc: `0.814`
- delta vs best single expert: `+0.006`

Comparison to the Experiment 11 residual mixture model:

- residual mixture: `0.8125`
- best fixed-feature tap: `0.81375`
- gap: `+0.001`

Interpretation:

- the same ordering appeared again: later features were better
- early features were not strong enough to beat the best expert
- both `block3` and `final` slightly improved over the best single expert
- on this dataset, the best fixed-feature fusion slightly beat the residual mixture model

## Cross-Dataset Pattern

Across both datasets, the ranking by feature depth was consistent:

- `final` was best
- `block3` was second-best and often close
- `block2` was much weaker
- `block1` was clearly worst

So for this class of models, the most useful frozen encoder inputs for fusion are the late semantic features, not the early local features.

That makes sense:

- early layers mostly carry low-level modality-specific structure
- later layers carry more task-aligned representations
- fusion seems to work best once each expert has already formed a strong internal abstraction

## Conclusion

The main conclusion is:

- the optimal hidden feature depth for fixed-feature fusion was the latest available feature level in both tested datasets

Compared to the best single expert:

- waveform-family: fixed-feature fusion gave a large gain
- Sub-GHz real: fixed-feature fusion gave a small but real gain

Compared to the residual mixture model:

- waveform-family: residual mixture remained slightly better
- Sub-GHz real: best fixed-feature fusion was slightly better

So the fixed-feature fusion family is viable, and the best version is clearly the late-feature version.

The practical takeaway for future work is:

- if we fuse frozen expert features in this repo, we should fuse late hidden states by default
- and if maximum performance matters, we should continue comparing against the residual mixture model, because it remains a very strong baseline
