# MS-BART single-spectrum pilot

This folder holds a small inference experiment with the authors' released MassSpecGym checkpoint. It is separate from the five OpenAI prompt experiments in the milestone notebook.

## Local files

Place the downloaded processed test file at:

`data/MassSpecGym-MIST_fps_selfies_threshold_0.11.tsv`

Place the final MassSpecGym model files together under `model-weights/`. Download the files from the [authors' Figshare release](https://figshare.com/articles/dataset/MS-BART-Model-Weights-Data/30393544), using the `MassSpecGym/model-wegihts` group in the release metadata. The source folder name contains a typo; use `model-weights` locally.

The `data/`, `model-weights/`, and `results/` folders are ignored by Git. Keep the original filenames and verify the file IDs and checksums against the release metadata before inference.

## Method note

The `fps` column contains MIST-predicted fingerprints derived from spectra. MS-BART generates SELFIES from this representation. It does not receive the peak-text prompt used in the OpenAI experiment. A later comparison should use the same held-out MassSpecGym identifiers and the same structure-scoring method, and should avoid using the reference formula to rank predictions unless that extra information is also accounted for.
