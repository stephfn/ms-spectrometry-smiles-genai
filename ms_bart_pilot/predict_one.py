import csv
from pathlib import Path

import torch
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
    output = model.generate(**inputs, max_new_tokens=256, num_beams=1)

print("Predicted SELFIES:", tokenizer.decode(output[0], skip_special_tokens=True).replace(" ", ""))
