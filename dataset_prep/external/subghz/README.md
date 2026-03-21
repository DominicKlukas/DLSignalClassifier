# Sub-GHz Dataset Prep

Raw source:

- Official IDLab/UGent page: https://idlab.ugent.be/resources/iq-samples-of-subghz-technologies
- Public GitHub mirror: https://github.com/JaronFontaine/Sub-GHz-IQ-signals-dataset

What to download:

- The raw `.mat` dataset archive or extracted directory tree corresponding to the IDLab/UGent Sub-GHz IQ Signals dataset.

Where to place it:

- Put the extracted dataset under `external/subghz_raw/Dataset/`

Expected raw layout:

- `external/subghz_raw/Dataset/<band>/<protocol>/*.mat`

Prep commands:

```bash
./.venv/bin/python dataset_prep/external/subghz/convert_to_h5.py
./.venv/bin/python dataset_prep/external/subghz/build_augmented_h5.py
```

Outputs:

- `data/real_mat_dataset/train.h5`
- `data/real_mat_dataset/val.h5`
- `data/real_mat_dataset/test.h5`
- `data/real_mat_dataset_augmented/train.h5`
- `data/real_mat_dataset_augmented/val.h5`
- `data/real_mat_dataset_augmented/test.h5`

Used by:

- `experiments/exp03_frozen_expert_residual`
- `experiments/legacy/exp05_subghz_real`
- `experiments/legacy/exp06_real_h5`
- `experiments/legacy/exp07_awgn_ablation_real`
