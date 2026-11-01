
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

pilot_dir = Path(__file__).resolve().parent / "pilot_outputs"

data = pd.read_csv(
    pilot_dir / "decoded_evaluation_32.tsv",
    sep="\t"
)

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

def similarity(smiles_a, smiles_b):
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if mol_a is None or mol_b is None:
        return None

    fp_a = generator.GetFingerprint(mol_a)
    fp_b = generator.GetFingerprint(mol_b)

    return DataStructs.TanimotoSimilarity(fp_a, fp_b)

data["tanimoto_baseline"] = data.apply(
    lambda row: similarity(
        row["prediction_baseline_smiles"],
        row["reference_smiles"]
    ),
    axis=1
)

data["tanimoto_finetuned"] = data.apply(
    lambda row: similarity(
        row["prediction_finetuned_smiles"],
        row["reference_smiles"]
    ),
    axis=1
)

data["difference"] = (
    data["tanimoto_finetuned"]
    - data["tanimoto_baseline"]
)

print("Pretrained mean:", data["tanimoto_baseline"].mean())
print("Fine-tuned mean:", data["tanimoto_finetuned"].mean())
print("Mean difference:", data["difference"].mean())

print("Improved:", (data["difference"] > 0).sum())
print("Worsened:", (data["difference"] < 0).sum())
print("Unchanged:", (data["difference"] == 0).sum())

output_path = pilot_dir / "similarity_comparison_32.tsv"

if output_path.exists():
    raise FileExistsError(f"Output already exists: {output_path}")

data.to_csv(output_path, sep="\t", index=False)

print("Saved:", output_path)
