# MS/MS-to-SMILES Molecular Structure Generation

An exploratory research project investigating generative machine learning approaches to molecular structure prediction from tandem mass spectrometry (MS/MS) data.

This project combines computational chemistry, mass spectrometry, and deep learning to examine molecular representations, candidate generation, chemical validation, and the effects of supervised fine-tuning.

The completed pilot includes a fine-tuned MS-BART model and an interactive Streamlit application for generating and comparing molecular structure candidates.

## Project Overview

Identifying molecular structures from tandem mass spectra is a challenging computational problem. Fragmentation patterns provide information about molecular composition and structure, but structural isomers, stereochemistry, and differences in experimental conditions can complicate molecular identification.

Traditional spectral library methods compare experimental spectra against known molecular candidates. Generative approaches offer the possibility of producing molecular structures beyond existing reference libraries. However, generating a chemically valid structure does not establish that it corresponds to the molecule responsible for an observed spectrum.

This project investigates these challenges through MassSpecGym data exploration, MS-BART inference, supervised fine-tuning, and molecular structure evaluation.

**Important distinction:** The current MS-BART pilot generates molecular structures from MIST-derived fingerprint token representations rather than directly accepting raw MS/MS spectra. The application demonstrates one component of a broader molecular structure elucidation workflow.

## Current Project Status

The following components have been completed:

- MassSpecGym data exploration and validation.
- Preliminary molecular structure generation using pretrained MS-BART.
- Full-parameter fine-tuning of MS-BART on 128 training examples.
- Evaluation using 32 held-out validation examples.
- Comparison of pretrained and fine-tuned predictions.
- Canonical SMILES and standardized connectivity evaluation using RDKit.
- Molecular fingerprint similarity analysis.
- Interactive Streamlit application using the saved fine-tuned checkpoint.
- SELFIES decoding and side-by-side molecular visualization.
- Distribution of the fine-tuned checkpoint through a GitHub Release.

The fine-tuning experiment updated all 104,105,472 model parameters during one training epoch.

Fine-tuning changed many generated predictions but did not establish a reliable improvement in molecular identification. Both the pretrained and fine-tuned models produced chemically valid structures for all 32 validation examples, but neither identified an exact reference match.

The current application provides a working demonstration of model inference, molecular representation conversion, and visual comparison.

## Streamlit Molecular Structure Workbench

The interactive application is available in [`streamlit_app/`](streamlit_app/).

It loads the saved fine-tuned MS-BART checkpoint and generates candidate molecular structures from MIST-derived fingerprint inputs.

### Application Features

- Select from 32 held-out validation examples.
- Inspect or edit fingerprint token inputs.
- Generate candidate SELFIES using the fine-tuned MS-BART checkpoint.
- Decode generated SELFIES into SMILES.
- Display predicted and reference molecular structures side by side.
- Examine the relationship between generated candidates and their reference molecules.
- Disable reference comparison when a fingerprint has been modified.
- Review model limitations and interpretation notes.

The application uses **SmilesDrawer** to render two-dimensional molecular diagrams in the browser.

RDKit was used for the earlier molecular evaluation but is not required for running the Streamlit interface.

Successful SELFIES decoding or visual rendering does not independently demonstrate chemical validity or correct molecular identification.

### Application Workflow

```text
MIST-derived fingerprint tokens
             |
             v
     Fine-tuned MS-BART
             |
             v
      Generated SELFIES
             |
             v
     SELFIES-to-SMILES
             |
             v
      2D Visualization
             |
             v
 Predicted / Reference Comparison
```

Reference structures are displayed for comparison after prediction. They are not provided to the model during generation.

## Installation and Reproduction

### 1. Clone the Repository

```bash
git clone https://github.com/stephfn/ms-spectrometry-smiles-genai.git
cd ms-spectrometry-smiles-genai
```

### 2. Install Application Dependencies

The application dependencies are specified in [`streamlit_app/requirements.txt`](streamlit_app/requirements.txt).

```bash
python -m pip install -r streamlit_app/requirements.txt
```

The application uses PyTorch, Hugging Face Transformers, pandas, SELFIES, safetensors, and Streamlit.

A compatible Python environment is required. GPU acceleration is optional; the application uses CUDA when available and falls back to CPU otherwise.

CUDA-enabled PyTorch installations may require additional platform-specific setup.

### 3. Download the Fine-Tuned Checkpoint

The saved model is distributed separately from the repository source code.

**[Download the MS-BART Fine-Tuned Pilot Checkpoint](https://github.com/stephfn/ms-spectrometry-smiles-genai/releases/tag/v0.1.0-ms-bart-pilot)**

Download the release asset:

`ms_bart_finetuned_release_clean.zip`

Extract the archive into `pilot_outputs/` at the repository root.

After extraction, the directory structure should be:

```text
pilot_outputs/
├── val_32.tsv
└── ms_bart_finetuned_128_continued/
    ├── added_tokens.json
    ├── config.json
    ├── generation_config.json
    ├── merges.txt
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.json
```

The checkpoint directory must retain the name `ms_bart_finetuned_128_continued`, because the application's model-loading utility references that relative path.

### 4. Launch the Streamlit Application

From the repository root, run:

```bash
python -m streamlit run streamlit_app/app.py
```

Streamlit normally opens the application at:

http://localhost:8501

Select a validation example and click **Generate Prediction**.

The application loads the fine-tuned checkpoint, generates a SELFIES candidate, decodes it into SMILES, and displays its two-dimensional molecular representation.

When the input matches the selected validation example, the corresponding reference structure is also displayed.

**Note:** The browser-based molecular renderer loads SmilesDrawer from an external CDN. Internet access is required for that component in the current implementation.

## Data Exploration and Validation

The initial exploration examined dataset integrity, spectral representations, and molecular structure consistency.

The work included:

- Examination of training, validation, and test partitions.
- Assessment of missing collision-energy and instrument metadata.
- Validation of mass-to-charge and intensity arrays.
- Verification of numeric and finite spectral values.
- Parsing and validation of molecular structures using RDKit.
- Investigation of molecular identifier overlap across dataset partitions.

All 231,104 spectrum records passed the implemented spectrum and SMILES consistency checks.

These results support the integrity of the inspected data under the implemented validation procedures. They do not establish that the molecular structures can be uniquely reconstructed from the available spectral information.

## MS-BART Model Development

### Pretrained Baseline

The initial MS-BART pilot used a pretrained encoder-decoder model to generate SELFIES molecular representations from MIST-derived fingerprint tokens.

A 32-example validation subset was used to evaluate whether the model could generate chemically valid structures and recover the corresponding reference molecules.

| Evaluation Metric | Pretrained MS-BART |
|---|---:|
| Completed generations | 32/32 |
| Valid molecular structures | 32/32 |
| Canonical SMILES exact matches | 0/32 |
| Tautomer-standardized connectivity matches | 0/32 |

The model generated chemically valid structures for all validation examples, but none matched the reference molecules under the implemented comparison methods.

This distinction motivated the subsequent fine-tuning experiment.

### Full-Parameter Fine-Tuning

The pretrained MS-BART model was fine-tuned using 128 MIST-derived fingerprint-to-SELFIES training examples.

Training configuration:

| Parameter | Configuration |
|---|---|
| Model architecture | MS-BART encoder-decoder |
| Training examples | 128 |
| Held-out validation examples | 32 |
| Fine-tuning approach | Full-parameter |
| Trainable parameters | 104,105,472 |
| Epochs | 1 |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁵ |
| Batch size | 1 |
| Gradient clipping | Maximum norm 1.0 |
| Inference decoding | Greedy, `num_beams=1` |
| Maximum generated tokens | 128 |

The checkpoint was saved and successfully reloaded for independent inference.

The training experiment was performed using PyTorch and an NVIDIA GPU.

### Pretrained vs. Fine-Tuned Evaluation

Both models were evaluated using the same 32 held-out validation examples and consistent generation and molecular comparison procedures.

| Evaluation Metric | Pretrained | Fine-Tuned |
|---|---:|---:|
| Valid generated structures | 32/32 | 32/32 |
| Canonical SMILES exact matches | 0/32 | 0/32 |
| Standardized connectivity matches | 0/32 | 0/32 |
| Mean Morgan fingerprint Tanimoto similarity | 0.2282 | 0.2341 |

Fine-tuning changed 24 of the 32 generated SELFIES sequences.

The paired structural similarity comparisons showed:

- 12 examples improved.
- 12 examples worsened.
- Eight examples remained unchanged.

Mean Tanimoto similarity increased by approximately 0.00594. However, the 95% bootstrap confidence interval ranged from −0.00945 to 0.02416 and included zero.

Therefore, the experiment did not establish a reliable improvement in molecular similarity or identification accuracy.

Although the fine-tuned model produced different predictions, these changes cannot be interpreted as evidence of better molecular identification.

### Interpretation

The results highlight three distinct evaluation objectives:

1. **Successful generation:** Whether the model can produce an output sequence.
2. **Chemical validity:** Whether a generated molecular representation can be parsed as a chemically valid structure.
3. **Molecular identification accuracy:** Whether the generated structure corresponds to the expected reference molecule.

These outcomes are related but not interchangeable.

The pilot demonstrates that fine-tuning can alter model output without necessarily improving molecular identification.

Because the experiment used a small training and validation sample, its findings should not be generalized to broader molecular datasets without additional evaluation.

## Datasets and Molecular Representations

The project draws on [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym), a benchmark for machine learning applications involving tandem mass spectrometry.

MassSpecGym is also available through [Hugging Face Datasets](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym).

Additional experiments use MIST-associated fingerprint-to-SELFIES representations obtained from the original MS-BART research data.

The project examines several molecular data representations:

| Representation | Purpose |
|---|---|
| MS/MS spectra | Experimental fragment mass-to-charge ratios and intensities |
| Molecular fingerprints | Encoded structural features used as model inputs in the current pilot |
| SELFIES | Molecular sequence representation generated by MS-BART |
| SMILES | Molecular structure representation used for decoding and comparison |
| InChIKeys | Standardized molecular identifiers for structure comparison |

These representations serve different purposes and should not be treated as interchangeable sources of molecular information.

In particular, the current fingerprint-to-SELFIES model is not equivalent to a complete raw-spectrum-to-structure prediction pipeline.

## Molecular Structure Evaluation

Molecular evaluation was performed using RDKit in the research notebook workflow.

The implemented evaluation includes:

- SELFIES-to-SMILES conversion.
- Molecular parsing and chemical validity checks.
- Canonical SMILES comparison.
- Tautomer-standardized connectivity comparison.
- Morgan molecular fingerprint generation.
- Tanimoto similarity calculation.
- Paired pretrained and fine-tuned model comparison.
- Bootstrap uncertainty estimation.

These procedures help distinguish molecular validity from structural similarity and exact identification.

The Streamlit application currently focuses on inference and visualization. It does not reproduce the full RDKit evaluation pipeline at runtime.

## Repository Contents

| Location | Description |
|---|---|
| [`streamlit_app/`](streamlit_app/) | Streamlit interface, model utilities, and application dependencies |
| [`pilot_outputs/val_32.tsv`](pilot_outputs/val_32.tsv) | Held-out validation records used by the application |
| [`ms_bart_milestone3_upload/`](ms_bart_milestone3_upload/) | MS-BART fine-tuning and evaluation artifacts |
| [`notebooks/`](notebooks/) | Research notebooks and experimental development |
| [`data/`](data/) | Data-related resources |
| [`docs/`](docs/) | Supporting documentation |
| [`references/`](references/) | Research references and supporting materials |
| [`reports/`](reports/) | Analysis and reporting artifacts |

The primary fine-tuning research notebook is available here:

[MS-BART Milestone 3: Fine-Tuning and Evaluation](ms_bart_milestone3_upload/ms_bart_milestone3_final_review.ipynb)

The trained model checkpoint is distributed through the GitHub Release rather than stored directly in Git history.

## Technologies

**Languages and libraries:** Python, pandas, PyTorch, Hugging Face Transformers, SELFIES, RDKit, Streamlit, and SmilesDrawer.

**Development tools:** JupyterLab, Conda, Git, GitHub, and Windows Terminal.

**Modeling methods:** Encoder-decoder sequence generation, full-parameter supervised fine-tuning, greedy decoding, and molecular representation learning.

**Evaluation methods:** Canonical structure comparison, tautomer standardization, molecular fingerprints, Tanimoto similarity, and bootstrap confidence intervals.

RDKit is used in the notebook evaluation workflow; SmilesDrawer is used for browser-based structure rendering in the Streamlit application.

## Limitations and Future Work

The current project is an exploratory research pilot rather than a validated molecular identification system.

Its primary limitations include:

- Small training and validation samples.
- No exact reference identifications in the 32-example evaluation.
- Limited evidence of improvement following fine-tuning.
- Fingerprint-based inputs rather than direct processing of experimental MS/MS spectra.
- Lack of integrated chemical validity and molecular similarity scoring in the Streamlit application.
- Potential ambiguities involving stereochemistry, tautomerism, and molecular representation.
- Browser-rendered molecular diagrams that do not independently verify chemical correctness.

Future work may include:

1. Expand training and evaluation to larger, independently held-out molecular datasets.
2. Evaluate ranked structure candidates using beam search and top-k metrics.
3. Integrate chemical validation and molecular similarity scoring into the application.
4. Investigate precursor mass, fragment-ion information, and neutral-loss relationships as model inputs.
5. Evaluate alternative molecular representations and generation strategies.
6. Study whether additional fine-tuning or improved input representations produce meaningful gains in molecular identification.

The overall objective remains to understand when generative molecular modeling produces chemically meaningful results and when apparently successful generation fails to recover the correct structure.

## Original Model Attribution and Licensing

This project builds upon **MS-BART**, developed by Han and colleagues.

Original research:

*MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation* (NeurIPS 2025).

- [Original MS-BART GitHub repository](https://github.com/OpenDFM/MS-BART)
- [Original model weights and research data](https://doi.org/10.6084/m9.figshare.30393544)
- Original source code license: MIT
- Original model weights and data license: Creative Commons Attribution 4.0 International (CC BY 4.0)

The fine-tuned checkpoint distributed with this project is an independently modified derivative of the original pretrained MS-BART model.

The original MS-BART authors are not responsible for this project's fine-tuning procedure, evaluation results, or Streamlit implementation.

The original attribution and applicable licensing terms should be retained when redistributing the derivative checkpoint.

## Research Scope

This repository documents an ongoing investigation into generative molecular structure prediction, including experimental methodology, model limitations, and reproducibility.

The project is intended for research, education, and portfolio demonstration.

**Generated molecular structures should not be treated as verified chemical identifications without additional independent evaluation.**
