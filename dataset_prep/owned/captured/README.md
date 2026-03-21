# Captured Dataset Prep

Raw source:

- Project-owned RTL-SDR recordings stored under `CapturedData/dataset/`

What must be present:

- `CapturedData/dataset/metadata.csv`
- the `.npy` recordings referenced by that metadata file

Prep command:

```bash
./.venv/bin/python dataset_prep/owned/captured/convert_to_h5.py
```

Outputs:

- `data/captured_npy_dataset_experiment5/train.h5`
- `data/captured_npy_dataset_experiment5/val.h5`
- `data/captured_npy_dataset_experiment5/test.h5`

Used by:

- `experiments/exp03_frozen_expert_residual`
- `experiments/legacy/exp06_real_h5`
