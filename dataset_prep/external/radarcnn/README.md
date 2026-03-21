# RadarCNN Dataset Prep

Raw source:

- Bundled external repo note: `external/radar_iq_datasets/README.md`
- Google Drive file listed there: https://drive.google.com/file/d/14u6mn3t8BanRQPwuWk2M0pSJxX-iRae5/view?usp=sharing

What to download:

- `radarcnn_dataset.zip`

Where to place it:

- The automated script writes it to `external/radar_iq_datasets/data/radarcnn_dataset.zip`

Prep command:

```bash
./.venv/bin/python dataset_prep/external/radarcnn/fetch_and_extract.py
```

Outputs:

- `external/radar_iq_datasets/data/radarcnn_dataset.zip`
- `external/radar_iq_datasets/data/radarcnn_unpacked/data/`

Used by:

- `experiments/followups/radarcnn`
