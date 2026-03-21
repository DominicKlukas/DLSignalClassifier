# DLSignalClassifier

This repo studies when raw IQ, FFT features, and fusion architectures are best for RF signal classification.

The main experimental sequence is:

1. IQ beats FFT on the synthetic modulation-family dataset.
2. FFT beats IQ on the synthetic waveform-family dataset.
3. Gated multimodal fusion helps, but does not reliably dominate the best expert.
4. Frozen-expert residual fusion is the strongest main model across the comparable benchmark family.

## Start Here

- Read [docs/Experiments.md](docs/Experiments.md) for the main experimental summary.
- Read [docs/Datasets.md](docs/Datasets.md) for dataset provenance, classes, and placement details.
- Read [experiments/README.md](experiments/README.md) for the experiment layout.

## Fresh Clone Workflow

If you cloned the repo and do not already have the local datasets:

1. Run the synthetic experiments first:

```bash
./.venv/bin/python experiments/recreate_main_story.py --mode synthetic-only
```

This reproduces Experiments 1 and 2 from a clean clone because those datasets are generated inside the repo.

2. Audit dataset availability:

```bash
./.venv/bin/python experiments/check_data.py
```

3. Retrieve and place the raw datasets using the instructions in [dataset_prep/README.md](dataset_prep/README.md) and [docs/Datasets.md](docs/Datasets.md).

4. Re-run the full experiment sequence once those datasets are present:

```bash
./.venv/bin/python experiments/recreate_main_story.py --mode full
```

## What Runs Without Extra Data

Runs from a clean clone:

- `experiments/exp01_iq_vs_fft`
- `experiments/exp02_gated_multimodal` after Experiment 1 generates its artifacts
- `experiments/recreate_main_story.py --mode synthetic-only`

Requires external or local datasets:

- `experiments/exp03_frozen_expert_residual`
- `experiments/followups/radioml2018`
- `experiments/followups/radarcnn`
- `experiments/analysis/correct_set_overlap`
- most of `experiments/legacy`

## Repo Layout

- `docs/`: experiment and dataset documentation
- `dataset_prep/`: raw-data retrieval and conversion instructions
- `experiments/`: runnable experiment code and results
- `garbage_bin/`: preserved historical material intentionally removed from the main surface

## Notes

- This repo assumes a local virtualenv at `.venv/` in the examples above because that is how the project has been run here.
- The main reproduction entrypoint now distinguishes between a clean-clone synthetic run and a full-data run.
