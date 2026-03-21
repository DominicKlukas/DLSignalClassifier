# Experiments

This file is the canonical high-level experiment summary for the repo.

The main result is a three-step progression:

1. IQ and FFT each win on different synthetic tasks.
2. A learned gated multimodal model helps, but does not reliably dominate the best single expert.
3. A frozen-expert residual fusion model does achieve that stronger behavior across the comparable benchmark family.

## Experiment 1: IQ vs FFT Representation Baselines

Question:

- does raw IQ win when the task is mostly time-structure driven?
- does FFT win when the task is mostly frequency-structure driven?

Implementation:

- [run.py](../experiments/exp01_iq_vs_fft/run.py)

Artifacts:

- [modulation_time_features.h5](../experiments/exp01_iq_vs_fft/artifacts/modulation_time_features.h5)
- [waveform_frequency_features.h5](../experiments/exp01_iq_vs_fft/artifacts/waveform_frequency_features.h5)
- [experiment1_results.json](../experiments/exp01_iq_vs_fft/artifacts/experiment1_results.json)

Result:

- modulation-family: IQ `0.730`, FFT `0.193`
- waveform-family: IQ `0.598`, FFT `0.728`

Takeaway:

- the two representations are not redundant
- IQ is the right inductive bias for modulation-family
- FFT is the stronger representation for waveform-family

This is the motivation for fusion.

## Experiment 2: Gated IQ+FFT Fusion

Question:

- if IQ and FFT each win on different tasks, can a gated multimodal model become strictly better than the best single expert?

Implementation:

- [run.py](../experiments/exp02_gated_multimodal/run.py)

Artifact:

- [experiment3_results.json](../experiments/exp02_gated_multimodal/artifacts/experiment3_results.json)

Reference comparison set:

- [Experiment5_Comparable_Experiments.md](Experiment5_Comparable_Experiments.md)

Results across all datasets in this comparison family:

- modulation-family: IQ `0.730`, FFT `0.193`, gated `0.680`
- waveform-family: IQ `0.598`, FFT `0.728`, gated `0.908`
- Sub-GHz real `128`: IQ `0.814`, FFT `0.774`, gated `0.816`
- Sub-GHz real `512`: IQ `0.815`, FFT `0.819`, gated `0.823`
- Sub-GHz real `1024`, `40` epochs: IQ `0.812`, FFT `0.816`, gated `0.821`
- augmented Sub-GHz real `512`: IQ `0.797`, FFT `0.775`, gated `0.826`
- Orbit RF: IQ `0.634`, FFT `0.719`, gated `0.707`
- captured `.npy`: IQ `0.957`, FFT `0.826`, gated `0.973`

Takeaway:

- the gated model can be much better when the modalities are complementary
- but it does **not** provide the stronger property we wanted
- on modulation-family, it still falls below the best individual expert
- on Orbit RF, FFT remains the strongest model
- on several real datasets, gated fusion is best but often only by a small margin

So the experimental progression does not end at multimodal fusion alone.

## Experiment 3: Frozen-Expert Residual Fusion

Question:

- can we build a multimodal model that keeps a true single-expert fallback path and only learns a bounded correction on top?

Implementation:

- [run.py](../experiments/exp03_frozen_expert_residual/run.py)

Main artifact:

- [experiment11_frozen_expert_residual_multidataset_results.json](../experiments/exp03_frozen_expert_residual/artifacts/experiment11_frozen_expert_residual_multidataset_results.json)

Detailed write-up:

- [Experiment11_Frozen_Expert_Residual_Multidataset_Log.md](Experiment11_Frozen_Expert_Residual_Multidataset_Log.md)

Results across the full comparable benchmark family:

- modulation-family: IQ `0.737`, FFT `0.177`, frozen residual `0.747`
- waveform-family: IQ `0.575`, FFT `0.750`, frozen residual `0.885`
- Sub-GHz real `128`: IQ `0.807`, FFT `0.764`, frozen residual `0.813`
- Sub-GHz real `512`: IQ `0.818`, FFT `0.816`, frozen residual `0.823`
- Sub-GHz real `1024`, `40` epochs: IQ `0.814`, FFT `0.818`, frozen residual `0.821`
- augmented Sub-GHz real `512`: IQ `0.796`, FFT `0.796`, frozen residual `0.812`
- Orbit RF: IQ `0.668`, FFT `0.682`, frozen residual `0.693`
- captured `.npy`: IQ `0.961`, FFT `0.828`, frozen residual `0.972`

Core result:

- on every dataset in that benchmark family, the frozen-expert residual model matched or exceeded the best single expert in the same run

Takeaway:

- this is the architecture that resolves the main open question in the repo
- it keeps the expert structure explicit
- and it empirically achieves the stronger behavior that the gated model did not

### RadioML 2018 Addendum

The same frozen-expert residual design was also evaluated on RadioML 2018 as a follow-up benchmark.

Full-data streaming result:

- IQ `0.610`
- FFT `0.303`
- frozen residual `0.614`

Interpretation:

- this benchmark is clearly IQ-dominant in our runs
- the frozen residual model only improves slightly over the IQ expert
- but that full-data result is still in a competitive range relative to published RML2018.01A average-accuracy comparisons, which are often reported around the low-to-mid `60%` range for strong models

References:

- [Experiment11_Frozen_Expert_Residual_Multidataset_Log.md](Experiment11_Frozen_Expert_Residual_Multidataset_Log.md)
- [Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges](https://ore.exeter.ac.uk/repository/bitstream/10871/131623/1/Deep%20Learning%20Based%20Automatic%20Modulation%20Recognition.pdf)
- [A Novel Approach for Robust Automatic Modulation Recognition Based on Reversible Column Networks](https://www.mdpi.com/2079-9292/14/3/618)

## Recommended Reading Order

- [Experiments.md](Experiments.md)
- [Experiment5_Comparable_Experiments.md](Experiment5_Comparable_Experiments.md)
- [Experiment11_Frozen_Expert_Residual_Multidataset_Log.md](Experiment11_Frozen_Expert_Residual_Multidataset_Log.md)

## Interesting Appendix

This is not part of the core experiment sequence, but it sharpens the interpretation of Experiment 3:

- [Experiment11_Correct_Set_Overlap_Log.md](Experiment11_Correct_Set_Overlap_Log.md)

That analysis shows the frozen model wins in total accuracy, but it does not literally preserve the full union of IQ-correct and FFT-correct samples on every dataset.
