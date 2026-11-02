"""Evaluate one MS-BART prediction without using the reference to generate it."""
import csv
from pathlib import Path

import selfies as sf
import torch
from rdkit import Chem
from transformers import BartForConditionalGeneration, BartTokenizer

root = Path(__file__).resolve().parent
model_dir = root / "model-weights"
test_file = root / "data" / "MassSpecGym-MIST_fps_selfies_threshold_0.11.tsv"
with test_file.open(newline="", encoding="utf-8") as handle:
    row = next(csv.DictReader(handle, delimiter="\t"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained(model_dir, local_files_only=True)
model = BartForConditionalGeneration.from_pretrained(model_dir, local_files_only=True).to(device)
model.eval()
inputs = tokenizer(row["fps"], return_tensors="pt", add_special_tokens=False)
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
with torch.inference_mode():
    ids = model.generate(**inputs, max_new_tokens=256, num_beams=1)
prediction = tokenizer.decode(ids[0], skip_special_tokens=True).replace(" ", "")

def canonical_smiles(selfies_text):
    try:
        molecule = Chem.MolFromSmiles(sf.decoder(selfies_text))
        return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule else None
    except (ValueError, sf.DecoderError):
        return None

predicted_smiles = canonical_smiles(prediction)
reference_smiles = canonical_smiles(row["selfies"])
print("Identifier:", row["identifier"])
print("Predicted SELFIES:", prediction)
print("Predicted valid molecule:", predicted_smiles is not None)
print("Reference valid molecule:", reference_smiles is not None)
print("Exact structure match:", predicted_smiles == reference_smiles if predicted_smiles and reference_smiles else False)
