
"""
MS-BART Model Utilities

Loads the fine-tuned MS-BART checkpoint, generates molecular
structure candidates from MIST-derived fingerprint tokens,
and decodes generated SELFIES into SMILES.

The model does not directly process raw MS/MS spectra.
Generated structures are exploratory molecular candidates,
not confirmed molecular identifications.
"""

from pathlib import Path

import selfies
import torch
from transformers import BartForConditionalGeneration, BartTokenizer


# --------------------------------------------------
# Project paths and model configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = (
    PROJECT_DIR
    / "pilot_outputs"
    / "ms_bart_finetuned_128_continued"
)

MAX_NEW_TOKENS = 128
NUM_BEAMS = 1


# --------------------------------------------------
# Load fine-tuned MS-BART model
# --------------------------------------------------

def load_model():
    """
    Load the saved fine-tuned MS-BART model and tokenizer.

    Uses CUDA when available, otherwise falls back to CPU.
    Returns the model, tokenizer, and selected device.
    """

    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint directory not found: {CHECKPOINT_DIR}"
        )

    weights_available = (
        (CHECKPOINT_DIR / "model.safetensors").is_file()
        or (CHECKPOINT_DIR / "pytorch_model.bin").is_file()
        or (CHECKPOINT_DIR / "model.safetensors.index.json").is_file()
        or (CHECKPOINT_DIR / "pytorch_model.bin.index.json").is_file()
    )

    if not weights_available:
        raise FileNotFoundError(
            f"No saved model weights found in: {CHECKPOINT_DIR}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BartForConditionalGeneration.from_pretrained(
        CHECKPOINT_DIR
    ).to(device)

    tokenizer = BartTokenizer.from_pretrained(
        CHECKPOINT_DIR
    )

    model.eval()

    return model, tokenizer, device


# --------------------------------------------------
# Generate molecular structure candidate
# --------------------------------------------------

def generate_prediction(fingerprint, model, tokenizer, device):
    """
    Generate a SELFIES sequence from fingerprint tokens.

    Preserves the inference configuration used in Milestone 3:
    - add_special_tokens=False
    - max_new_tokens=128
    - num_beams=1

    Reference molecular structures are not used in generation.
    """

    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError(
            "Fingerprint input must be a non-empty string."
        )

    inputs = tokenizer(
        fingerprint.strip(),
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    model.eval()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS
        )

    prediction = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    ).replace(" ", "")

    if not prediction:
        raise ValueError(
            "The model generated an empty SELFIES sequence."
        )

    return prediction


# --------------------------------------------------
# Decode SELFIES into SMILES
# --------------------------------------------------

def decode_structure(selfies_sequence):
    """
    Decode a SELFIES sequence into a SMILES string.

    Returns None if decoding fails.

    Successful decoding does not independently establish
    chemical validity or correct molecular identification.
    """

    if not isinstance(selfies_sequence, str):
        return None

    if not selfies_sequence.strip():
        return None

    try:
        smiles = selfies.decoder(selfies_sequence)

        return smiles if smiles else None

    except selfies.DecoderError:
        return None
