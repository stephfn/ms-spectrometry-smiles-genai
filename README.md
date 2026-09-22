# MS/MS-to-SMILES Molecular Structure Generation

An in-progress DSC670 capstone project exploring generative AI for predicting molecular structures, represented as SMILES strings, from tandem mass spectrometry (MS/MS) data.

This project connects chemistry, scientific data preparation, and machine learning. Its goal is to compare a base generative model with a supervised fine-tuned model on a manageable subset of MassSpecGym.

## Current Status

**Data exploration and validation are underway. Model training and evaluation have not yet been performed.**

The exploratory notebook currently includes:

* Inspection of the dataset and its provided training, validation, and test folds.
* Assessment of missing collision-energy and instrument metadata.
* Checks for blank spectrum and SMILES fields.
* Verification that mass-to-charge and intensity arrays have matching lengths.
* Checks that spectral values are numeric and finite, mass-to-charge values are positive, and intensities are nonnegative.
* Verification that each spectrum contains at least one positive intensity.
* SMILES parsing and nonempty-molecule validation using RDKit.
* Checks for molecular identifier overlap across folds.

All 231,104 spectrum records passed the implemented spectrum and SMILES checks. No overlap was found across folds using the dataset’s provided molecular identifiers. These checks establish basic data consistency; they do not establish that structures can be uniquely identified from their spectra.

Missing collision-energy and instrument metadata remain documented for later preprocessing decisions.

## Dataset

The project uses [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym), loaded through [Hugging Face Datasets](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym).

The provided training, validation, and test assignments will be preserved.

## Repository Contents

* [`notebooks/01_massspecgym_feasibility.ipynb`](notebooks/01_massspecgym_feasibility.ipynb): exploratory analysis, data-quality checks, and preparation decisions.

The notebook is an evolving research and learning record.

## Planned Work

1. Select a manageable subset while accounting for repeated spectra from the same molecule.
2. Develop a consistent representation of spectral inputs and SMILES targets.
3. Compare base-model and supervised fine-tuned-model predictions.
4. Evaluate SMILES validity, standardized molecular exact match, and fingerprint similarity.
5. Document results, limitations, and possible improvements.

## Tools

Python, pandas, Hugging Face Datasets, RDKit, and Jupyter Notebook.

## Scope

This is an academic research project. It explores molecular structure prediction from MS/MS data; it does not currently perform pharmacokinetic modeling or provide a validated analytical identification workflow.
