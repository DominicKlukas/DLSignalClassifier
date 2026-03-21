# RadioML Dataset Prep

Raw source:

- Official DeepSig datasets page: https://www.deepsig.ai/datasets/

What to download:

- The historical `RADIOML 2018.01A` dataset HDF5 file used by this repo.

Where to place it:

- `external/radioml_dataset/GOLD_XYZ_OSC.0001_1024.hdf5`

Notes:

- This repo currently does not include a dedicated downloader for RadioML 2018.01A.
- Once the HDF5 file is placed at the path above, the follow-up benchmark can be run directly.

Used by:

- `experiments/followups/radioml2018`
