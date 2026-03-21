# Dataset Preparation

This directory contains retrieval and preparation code for datasets that do not come from a clean clone of the repo.

Structure:

- `external/`: datasets obtained from the internet or third-party research releases
- `owned/`: datasets created in this project and then converted into experiment-ready formats

Each dataset directory contains a short README describing:

- where the raw data comes from
- where this repo expects it to live
- which prep script to run
- which experiment families depend on it
