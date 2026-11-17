# MS/MS-to-SMILES Molecular Structure Generation

An ongoing research project exploring generative machine learning approaches to molecular structure prediction from tandem mass spectrometry (MS/MS) data.

This project combines computational chemistry, mass spectrometry, and deep learning to investigate whether generative models can reconstruct molecular structures from spectral information. The research focuses on molecular representation, structure validation, candidate generation, and the potential improvements achievable through supervised fine-tuning.

## Project Overview

Identifying molecular structures from MS/MS spectra is a challenging computational problem. Fragmentation patterns provide information about molecular composition and structure, but different molecules can produce similar spectra. Structural isomers, stereochemistry, and variations in experimental conditions further complicate identification.

Traditional spectral library searches rank existing molecular candidates against observed spectra. Generative approaches offer the possibility of predicting structures beyond those contained in a reference library, although producing chemically valid structures does not guarantee correct molecular identification.

This project investigates these challenges using MassSpecGym, RDKit, and MS-BART, a specialized sequence-generation model.

### Current Project Status

Data exploration, molecular validation, pretrained MS-BART inference, and a small-scale full-parameter fine-tuning experiment have been completed.

The fine-tuning pilot used 128 training examples and 32 held-out validation examples. All 104,105,472 model parameters were trainable, and the model was updated over one training epoch.

Although fine-tuning changed the generated predictions, the experiment did not demonstrate reliable improvement in molecular identification accuracy. Both the pretrained and fine-tuned models generated chemically valid structures for all 32 validation examples, but neither produced an exact reference match.

The current development phase focuses on integrating the saved fine-tuned model into a Streamlit application for interactive molecular candidate generation and visualization.

### Data Exploration and Validation

Initial exploratory work includes:

- Examination of dataset structure and predefined training, validation, and test folds.
- Assessment of missing collision-energy and instrument metadata.
- Validation of mass-to-charge and intensity arrays.
- Verification of numeric, finite, and chemically appropriate spectral values.
- Molecular structure parsing and validation using RDKit.
- Examination of molecular identifier overlap across dataset partitions.

All 231,104 spectrum records passed the implemented spectrum and SMILES consistency checks. These results establish basic data integrity but do not demonstrate that molecular structures can be uniquely identified from the available spectra.

### MS-BART Baseline Evaluation

A preliminary inference experiment was conducted using a pretrained MS-BART model and a small validation subset derived from MIST-associated molecular representations.

The current pilot uses molecular fingerprint inputs and generates structures encoded as SELFIES, which are subsequently converted into SMILES for chemical validation.

Initial results from 32 validation examples:

| Evaluation Metric | Result |
|---|---|
| Valid molecular structures | 32/32 (100%) |
| Canonical SMILES exact matches | 0/32 (0%) |
| Tautomer-standardized connectivity matches | 0/32 (0%) |
| Completed generations | 32/32 (100%) |

These results demonstrate successful generation of chemically valid structures within the pilot sample, but no correct identifications under the implemented comparison methods.

The small sample size limits generalization. Additional evaluation is needed to determine whether alternative decoding strategies, molecular standardization, and fine-tuning improve structural accuracy.

### MS-BART Fine-Tuning and Evaluation

A full-parameter fine-tuning experiment was performed using 128 MIST-derived fingerprint-to-SELFIES training examples. The pretrained and fine-tuned models were evaluated on the same 32 held-out validation examples using consistent decoding and molecular comparison procedures.

| Evaluation Metric | Pretrained MS-BART | Fine-Tuned MS-BART |
|---|---:|---:|
| Valid generated structures | 32/32 | 32/32 |
| Canonical SMILES exact matches | 0/32 | 0/32 |
| Standardized connectivity matches | 0/32 | 0/32 |
| Mean Morgan fingerprint Tanimoto similarity | 0.2282 | 0.2341 |

Fine-tuning changed 24 of the 32 generated SELFIES sequences. However, the structural similarity results were mixed: 12 examples improved, 12 worsened, and eight remained unchanged.

The mean Tanimoto similarity increased by approximately 0.00594, but the 95% bootstrap confidence interval included zero (−0.00945 to 0.02416). Consequently, the results do not establish a reliable improvement in molecular similarity or identification accuracy.

These findings reinforce the distinction between successful model optimization, chemically valid structure generation, and correct molecular identification.

The model currently generates molecular structures from MIST-derived fingerprint representations rather than directly processing raw tandem mass spectra. The resulting predictions should be treated as exploratory molecular candidates requiring independent chemical validation.

## Datasets and Molecular Representations

The project draws on [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym), a benchmark for machine learning applications involving tandem mass spectrometry, available through [Hugging Face Datasets](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym).

Additional experiments explore MIST-associated molecular fingerprint and SELFIES representations for compatibility with MS-BART.

The project examines several molecular representations:

- **MS/MS spectra:** Mass-to-charge ratios and corresponding fragment intensities.
- **SMILES:** Text-based representations of molecular structures.
- **SELFIES:** Molecular string representations designed to support robust generative modeling.
- **InChIKeys:** Standardized identifiers used for molecular structure comparisons.
- **Molecular fingerprints:** Encoded structural information used in the specialized generative modeling workflow.

These representations serve different purposes and are not interchangeable. An important part of the research is evaluating how representation choices affect the interpretation of model predictions.

## Research Methods

The workflow includes dataset validation, molecular representation processing, model inference, and chemical structure evaluation.

RDKit is used to parse generated molecular structures, calculate canonical representations, and investigate structural equivalence.

Evaluation currently distinguishes between canonical SMILES exact matching and tautomer-standardized molecular connectivity matching. Planned experiments will additionally examine ranked molecular candidates through beam search and top-k evaluation metrics.

Supervised fine-tuning has been evaluated against the pretrained model using consistent validation procedures.

## Repository Contents

- [`notebooks/01_massspecgym_feasibility.ipynb`](notebooks/01_massspecgym_feasibility.ipynb): Dataset exploration, data-quality assessment, and initial feasibility analysis

- [https://github.com/stephfn/ms-spectrometry-smiles-genai/blob/main/ms_bart_milestone3_upload/ms_bart_milestone3_final_review.ipynb?utm_source=chatgpt.com]

Additional experimental notebooks document generative model testing and MS-BART pilot development.

The notebooks serve as evolving research records, including experimental decisions, intermediate results, and observed limitations.

## Next Steps

1. Integrate the saved fine-tuned MS-BART checkpoint into an interactive Streamlit application.
2. Implement fingerprint input and candidate molecular structure generation.
3. Add SELFIES decoding, canonical SMILES validation, and RDKit molecular visualization.
4. Provide optional reference comparisons using molecular fingerprint similarity.
5. Test the application and document its environment, dependencies, and limitations.
6. Expand evaluation using additional held-out molecular examples and ranked predictions.
7. Investigate whether incorporating precursor mass, fragment-ion information, and neutral-loss relationships can improve molecular structure identification.

## Technologies

**Languages and libraries:** Python, pandas, PyTorch, Hugging Face Transformers, Hugging Face Datasets, RDKit, and SELFIES.

**Development tools:** JupyterLab, Conda, Git, and GitHub.

**Modeling:** MS-BART, sequence generation, beam search, and supervised fine-tuning.

## Research Scope and Limitations

This project is an exploratory investigation of molecular structure generation and evaluation. Current experiments use a limited validation sample and should not be interpreted as evidence of reliable molecular identification.

Generating a chemically valid molecule does not establish that the structure corresponds to the observed spectrum. Additional considerations include molecular standardization, stereochemical ambiguity, dataset coverage, and the relationship between model inputs and experimental spectra.

The project is intended to investigate computational approaches and their limitations, not to provide a validated analytical identification system.
