# General Signal Classifier

## Conclusion on experiments up to this point

- The "scale invariant" to the solutions to this OOD problem don't seem particularly elegant. At the end of the day, you are just doing architectural augmentation in order to accomodate multiple different scales. Since CCNs with data augmentation are already so incredibly effective, there seems to be little point in pursuing this direction.

## Experiment 1: Comparing TSCNN and FFTCNN performance on TS feature vs FFT feature signals
- CNNs on the time domain are effective at classifying specific features in data, and CNNs on the frequency domain are effective at classifying other features in data.
- Our first task, is to show that the CNN in the time domain performs poorly at classifying signals that have rich diverse features in the frequency domain.
- In light of this, for a first experiment, cleanly seperate two datasets:one that has rich frequency features, and another which has rich time featrues (QPSK, etc). (You arleady made these datasets... just use the ones you made already). Then, compare the FFT CNN and the time series CNN on both of these.

### Result


## Experiment 2: Creating a multi-modal CNN
### Motivation
Since some signals have more features in the time domain and some in the frequency domain, we want a multi modal CNN that takes both IQ and FFT input simultaniously.

### Design
We have two CNNs, one for FFT data and one for TS data. The flattened outputs are then concatenated as an input the the MLP.

### Hypothesis
The multimodal (IQ and FFT) CNN should outperform both single-mode CNNs on both datasets.

### Experiment
Compare the performance on across both datasets between the IQ CNN, FFT CNN and IQ-FFT CNN. Use the SignalGenerator.py and the WaveformFamilyGenerator.py to create the datasets.