# Experiment 11 Correct-Set Overlap Analysis

## Goal

For the frozen-expert residual fusion experiment, analyze the **sets of test samples** that each model gets correct:

- IQ expert
- FFT expert
- frozen-expert residual fusion model

We want to see:

- how much the IQ-correct and FFT-correct sets overlap
- whether the fusion model covers the union of the expert-correct sets
- how many additional test samples the fusion model gets correct that neither expert got right

## Outputs

Machine-readable results:

- [results.json](../experiments/analysis/correct_set_overlap/results.json)

Per-dataset overlap plots:

- [modulation_family_overlap.png](../experiments/analysis/correct_set_overlap/plots/modulation_family_overlap.png)
- [waveform_family_overlap.png](../experiments/analysis/correct_set_overlap/plots/waveform_family_overlap.png)
- [subghz_real_128_overlap.png](../experiments/analysis/correct_set_overlap/plots/subghz_real_128_overlap.png)
- [subghz_real_512_overlap.png](../experiments/analysis/correct_set_overlap/plots/subghz_real_512_overlap.png)
- [subghz_real_1024_40ep_overlap.png](../experiments/analysis/correct_set_overlap/plots/subghz_real_1024_40ep_overlap.png)
- [subghz_real_augmented_512_overlap.png](../experiments/analysis/correct_set_overlap/plots/subghz_real_augmented_512_overlap.png)
- [orbit_rf_overlap.png](../experiments/analysis/correct_set_overlap/plots/orbit_rf_overlap.png)
- [captured_npy_real_128_overlap.png](../experiments/analysis/correct_set_overlap/plots/captured_npy_real_128_overlap.png)

## How to Read the Plots

Each plot has two panels.

Left panel:

- the `8` possible correctness combinations across `IQ / FFT / Fusion`
- examples:
  - `IQ only`
  - `FFT only`
  - `Fusion only`
  - `IQ+Fusion`
  - `FFT+Fusion`
  - `All three`
  - `None`

Right panel:

- total samples correct by IQ
- total samples correct by FFT
- total samples correct by Fusion
- total samples correct by the **union** of IQ and FFT
- how many of those union samples Fusion also got right
- how many extra samples Fusion got right outside that union
- how many samples from the expert union Fusion missed

This is the most useful view here because it shows both:

- overlap between the expert-correct sets
- where the fusion model adds value or loses coverage

## Main Pattern

Across every dataset:

- the IQ and FFT correct sets overlap a lot
- but they are not identical
- the fusion model usually covers most of the expert union
- and in several datasets it also gets extra samples right that neither expert got right

The most important numbers are:

- `expert_union_correct`
- `fusion_covers_expert_union`
- `fusion_extra_beyond_expert_union`
- `fusion_misses_from_expert_union`

## Dataset-by-Dataset Summary

### Modulation Family

- IQ correct: `221`
- FFT correct: `53`
- fusion correct: `224`
- expert union: `237`
- fusion covers union: `218`
- fusion misses from union: `19`
- fusion extra beyond union: `6`

Interpretation:

- IQ dominates this task
- FFT adds only a small amount of complementary coverage
- fusion does not preserve the full expert union here
- but it still gains a few new correct samples outside the expert union

### Waveform Family

- IQ correct: `230`
- FFT correct: `300`
- fusion correct: `354`
- expert union: `355`
- fusion covers union: `341`
- fusion misses from union: `14`
- fusion extra beyond union: `13`

Interpretation:

- the expert union is already strong
- fusion recovers almost all of it
- and adds a meaningful number of new correct samples
- this is a strong complementarity case

### Real Sub-GHz, 128 Windows/File

- IQ correct: `2581`
- FFT correct: `2444`
- fusion correct: `2600`
- expert union: `2609`
- fusion covers union: `2562`
- fusion misses from union: `47`
- fusion extra beyond union: `38`

Interpretation:

- the experts overlap heavily
- fusion still adds useful new coverage
- but it does not exactly preserve the full expert union

### Real Sub-GHz, 512 Windows/File

- IQ correct: `10476`
- FFT correct: `10444`
- fusion correct: `10540`
- expert union: `10585`
- fusion covers union: `10540`
- fusion misses from union: `45`
- fusion extra beyond union: `0`

Interpretation:

- here the gain comes almost entirely from better agreement over the shared expert-correct region
- fusion does not add extra samples beyond the expert union
- but it still ends up close to that union and beats either expert alone

### Real Sub-GHz, 1024 Windows/File, 40 Epochs

- IQ correct: `20847`
- FFT correct: `20930`
- fusion correct: `21010`
- expert union: `21073`
- fusion covers union: `21007`
- fusion misses from union: `66`
- fusion extra beyond union: `3`

Interpretation:

- again, overlap between experts is extremely high
- fusion is close to the union, but does not perfectly cover it
- extra beyond the expert union is very small in this setting

### Augmented Real Sub-GHz, 512 Windows/File

- IQ correct: `10192`
- FFT correct: `10184`
- fusion correct: `10390`
- expert union: `10579`
- fusion covers union: `10390`
- fusion misses from union: `189`
- fusion extra beyond union: `0`

Interpretation:

- corruption increases disagreement between the experts
- fusion still improves over each expert individually
- but it does so without exceeding the expert union
- this looks more like “better global balancing” than “new discovery beyond both experts”

### Orbit RF

- IQ correct: `9781`
- FFT correct: `9990`
- fusion correct: `10146`
- expert union: `10920`
- fusion covers union: `10054`
- fusion misses from union: `866`
- fusion extra beyond union: `92`

Interpretation:

- Orbit is the clearest case where the experts disagree meaningfully
- fusion adds many new correct samples
- but it also fails to preserve a large chunk of the expert union
- so its improvement over the best expert is a trade: gain some, lose some, win overall

### Captured `.npy` Real Benchmark

- IQ correct: `26930`
- FFT correct: `23198`
- fusion correct: `27246`
- expert union: `27290`
- fusion covers union: `27019`
- fusion misses from union: `271`
- fusion extra beyond union: `227`

Interpretation:

- this is one of the strongest practical fusion cases
- the expert union is already very large
- fusion misses some union samples
- but it also gains a large number of entirely new correct samples
- net effect: strong improvement over either expert alone

## Overall Interpretation

The key message is:

- the frozen-expert fusion model does **not** literally classify every sample that either expert classified correctly

So the model is not acting like a perfect set-theoretic superset of the individual experts.

What it is doing instead is:

- preserving **most** of the expert union
- sometimes losing a subset of expert-correct samples
- often gaining a new set of correct samples that neither expert had

That means the performance gain comes from a trade:

- lose some expert-union samples
- gain some entirely new ones
- end up with a higher total accuracy than either expert alone

This is especially visible in:

- waveform-family
- Orbit RF
- captured `.npy` real

while the large clean real Sub-GHz runs look more conservative and union-like.

## Conclusion

The overlap analysis refines the earlier conclusion:

- the frozen-expert residual fusion model outperformed the individual experts in total accuracy
- but it did **not** strictly contain the experts’ correct-sample sets

So the right interpretation is:

- it is better as a classifier
- it is not simply “expert union plus extras”
- its advantage comes from reshaping the decision boundary, not just inheriting every correct decision from both experts
