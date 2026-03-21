# Experiment Reproduction

This directory is organized around four layers:

1. `exp01_iq_vs_fft`, `exp02_gated_multimodal`, `exp03_frozen_expert_residual`
2. `followups/`
3. `legacy/`
4. `analysis/` and `tools/`

## Canonical Experiment Sequence

The main experiment sequence lives in:

- `experiments/exp01_iq_vs_fft`
- `experiments/exp02_gated_multimodal`
- `experiments/exp03_frozen_expert_residual`

These are the directories to use if someone wants to recreate the results discussed in `docs/experiments.md`.

### Fastest Way To Recreate The Main Experiments

```bash
./.venv/bin/python experiments/recreate_main_story.py --mode auto
```

`auto` runs Experiments 1 and 2 unconditionally, and then runs Experiment 3 only if its required local datasets are present.

For a clean clone with no local datasets, use:

```bash
./.venv/bin/python experiments/recreate_main_story.py --mode synthetic-only
```

For a strict full reproduction once all local data is in place, use:

```bash
./.venv/bin/python experiments/recreate_main_story.py --mode full
```

Results are written to:

- `experiments/main_story_results.json`

Detailed artifacts remain in each experiment's `artifacts/` directory.

### Check What Data Is Missing

```bash
./.venv/bin/python experiments/check_data.py
```

This prints which local datasets are present, which are missing, and which experiment families depend on them.

## Follow-Ups

Follow-up work that extends the main fusion idea but is not part of the core experiment sequence now lives in:

- `experiments/followups/radioml2018`
- `experiments/followups/occluded_objects`
- `experiments/followups/cross_attention_multidataset`

## Legacy

Earlier branches and intermediate baselines that are useful for reference, but not part of the cleaned main experiment sequence, now live in:

- `experiments/legacy/exp02_plain_multimodal`
- `experiments/legacy/exp04_spectrogram`
- `experiments/legacy/exp05_orbit_rf`
- `experiments/legacy/exp05_subghz_real`
- `experiments/legacy/exp06_real_h5`
- `experiments/legacy/exp07_awgn_ablation_real`
- `experiments/legacy/exp08_awgn_ablation_waveform`
- `experiments/legacy/exp09_wavelet_multidataset`

## Analysis And Utilities

- `experiments/analysis/correct_set_overlap` contains the overlap analysis used by the Experiment 11 follow-up note.
- `experiments/check_data.py` audits required local dataset paths for the main experiment sequence and follow-up benchmarks.
- `experiments/tools/generate_dataset_figures.py` generates documentation figures.
- `dataset_prep/` contains dataset retrieval and conversion scripts.
