# ORBIT RF Dataset Prep

Raw source:

- Local dataset note: `orbit_rf_identification_dataset_updated/Readme.txt`
- Associated paper: "Open Set Wireless Transmitter Authorization: Deep Learning Approaches and Dataset Considerations"

What to retrieve:

- The updated ORBIT RF identification dataset day files matching the filenames used in this repo:
  - `grid_2019_12_25.pkl`
  - `grid_2020_02_03.pkl`
  - `grid_2020_02_04.pkl`
  - `grid_2020_02_05.pkl`
  - `grid_2020_02_06.pkl`

Where to place them:

- `orbit_rf_identification_dataset_updated/`

Notes:

- This repo does not currently provide an automated fetch script for ORBIT because the dataset is handled here as a pre-prepared set of day-level pickle files.
- The experiment code expects exactly the filenames above.

Used by:

- `experiments/exp03_frozen_expert_residual`
- `experiments/legacy/exp05_orbit_rf`
