# Recent Waveform Follow-Ups

This note summarizes the recent follow-up experiments we ran around the waveform-family dataset and the question of whether a sufficiently strong time-domain model can recover frequency-domain usefulness.

## 1. Waveform-Family Capacity Scaling

We started from the hypothesis that a time-domain model might match the FFT-domain model on the waveform-family dataset if it were given enough capacity to learn frequency-like structure internally.

The first attempt was a simple wider/deeper time CNN. That helped a little, but not enough:

- the larger time CNN improved over the original time baseline
- but it still remained clearly behind the FFT baseline

This suggested that generic scaling was not sufficient.

## 2. Three Time-Domain Scaling Families

We then expanded the experiment into three more targeted time-domain architecture families:

- multi-scale temporal CNNs
- large-kernel temporal CNNs
- learnable-filterbank-front-end CNNs

For each family, we trained small, medium, and large variants on the waveform-family dataset and recorded:

- test accuracy
- trainable parameter count
- model family and scale

### Main Outcome

The important result was that the *type* of capacity mattered much more than raw parameter count alone.

- multi-scale models were the strongest time-domain family
- large-kernel models were also strong
- filterbank models helped, but were less stable across scales

In the main sweep:

- `multiscale_l` reached about `0.890` test accuracy
- `largekernel_l` reached about `0.863`
- the original FFT baseline in that sweep reached about `0.708`
- the original time baseline reached about `0.610`

So, on this run, the best time-domain model not only closed the gap to FFT, but exceeded the FFT baseline.

### Interpretation

This supports a more specific claim than "more layers help":

- time-domain CNNs can become highly competitive on FFT-favored tasks
- but they need the *right inductive bias*
- especially multi-scale receptive fields and large early kernels

In other words, the relevant question is not just "does more capacity help?" but "what kind of capacity helps?"

## 3. Capacity-Matched Gated Fusion Models

After the stronger time-domain models appeared, we added larger gated fusion models so the fusion baseline could be compared at similar parameter budgets.

We trained:

- the original `gated` model
- `gated_medium`
- `gated_large`

The new fusion models were sized to land near the stronger multi-scale time models:

- `gated_medium`: about `7.53M` parameters
- `multiscale_m`: about `6.48M`
- `gated_large`: about `14.42M`
- `multiscale_l`: about `16.11M`

### Main Outcome

The capacity-matched fusion models remained very strong:

- `gated_medium` beat `multiscale_m`
- `gated_large` slightly beat `multiscale_l`

In the comparison run:

- `gated`: about `0.895` test accuracy
- `gated_medium`: about `0.888`
- `gated_large`: about `0.900`
- `multiscale_l`: about `0.890`

So fusion still performed best overall, but the gap to the strongest time-only model became small.

## 4. Parameter Count vs Accuracy Plot

We also generated a scaling plot showing parameter count versus test accuracy across the sweep, including the later capacity-matched gated points.

Artifact:

- [param_count_vs_test_accuracy.png](/home/klukasdh/Projects/DLSignalClassifier/experiments/followups/waveform_time_capacity/artifacts/param_count_vs_test_accuracy.png)

This plot makes the scaling story easier to see visually:

- some extra parameters help a lot
- but architecture family matters more than parameter count alone

## 5. FFT Approximation Side Experiment

We then explored a different question:

- can a `multiscale_l`-style convolutional model explicitly learn to approximate the Fourier transform from raw IQ input?

We first discussed complex FFT prediction, but then narrowed the task to:

- predicting **FFT magnitude** only

### Dataset

To avoid training on a narrow special case, we built a synthetic IQ-to-FFT-magnitude dataset that mixes several signal types:

- complex Gaussian signals
- multi-tone signals
- chirps
- AM/FM-like signals
- bursts
- filtered noise
- piecewise-frequency signals

The goal was to teach the model a broad approximation rather than a narrow dataset-specific mapping.

### Model

We implemented a sequence-to-sequence convolutional model using the same multi-scale residual design ideas as the strong `multiscale_l` classifier family.

The model takes raw IQ as input and predicts a length-`1024` FFT-magnitude sequence.

### Training Issue and Fix

The first heavy run appeared stalled, but the real issue was the data loader:

- it reopened the HDF5 file for every sample

This made training extremely slow and made the run look frozen.

We fixed that by:

- keeping a persistent HDF5 handle open per dataset instance
- adding batch-level progress logging so long runs show visible movement

## 6. Lightweight FFT-Magnitude Run

Because the full run was too heavy for quick iteration, we switched to a lighter setting:

- train: `2048`
- val: `256`
- test: `256`
- epochs: `10`

### Result

The model did learn a nontrivial approximation to FFT magnitude, but the result was still rough:

- best epoch: `6`
- test MSE: about `0.670`
- test relative RMSE: about `0.818`
- test magnitude cosine similarity: about `0.549`

### Interpretation

This shows:

- the multiscale convolutional model is learning some genuine spectral structure
- but it is still far from being a strong numerical FFT surrogate

So the evidence supports:

- "the model can learn FFT-like features"

more strongly than:

- "the model can accurately compute the FFT"

## 7. Main Conceptual Takeaway

The recent experiments strengthen the following interpretation:

- IQ and FFT are information-equivalent in principle
- but they are not equally easy for practical CNNs to exploit

What now seems plausible is:

- a time-domain CNN can learn useful FFT-like features relatively naturally
- but recovering time-domain usefulness from FFT-domain inputs may be harder for CNNs, especially when temporal structure depends on local alignment, phase, bursts, and transitions

So although the two views contain the same underlying information, there is an important *inductive-bias asymmetry* in how easily CNNs can use them.

## Artifacts

Waveform scaling artifacts:

- [results_full_sweep_before_gated_capacity_match.json](/home/klukasdh/Projects/DLSignalClassifier/experiments/followups/waveform_time_capacity/artifacts/results_full_sweep_before_gated_capacity_match.json)
- [results.json](/home/klukasdh/Projects/DLSignalClassifier/experiments/followups/waveform_time_capacity/artifacts/results.json)
- [param_count_vs_test_accuracy.png](/home/klukasdh/Projects/DLSignalClassifier/experiments/followups/waveform_time_capacity/artifacts/param_count_vs_test_accuracy.png)

FFT magnitude approximation artifacts:

- [results.json](/home/klukasdh/Projects/DLSignalClassifier/experiments/followups/fft_seq2seq_multiscale/artifacts/results.json)
- [fft_magnitude_approximation_dataset.h5](/home/klukasdh/Projects/DLSignalClassifier/experiments/followups/fft_seq2seq_multiscale/artifacts/fft_magnitude_approximation_dataset.h5)
- [best_fft_approximator.pt](/home/klukasdh/Projects/DLSignalClassifier/experiments/followups/fft_seq2seq_multiscale/artifacts/checkpoints/best_fft_approximator.pt)
