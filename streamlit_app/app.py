from pathlib import Path
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from model_utils import load_model, generate_prediction, decode_structure

# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
VALIDATION_FILE = PROJECT_DIR / "pilot_outputs" / "val_32.tsv"

st.set_page_config(
    page_title="MS-BART Molecular Structure Workbench",
    page_icon="🧬",
    layout="wide",
)

# --------------------------------------------------
# Cached resources
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_validation_data():
    return pd.read_csv(VALIDATION_FILE, sep="\t")

# --------------------------------------------------
# Molecule rendering helper
# --------------------------------------------------

def render_molecule(smiles: str):
    """Render a 2D molecular structure using SmilesDrawer."""

    if not smiles:
        st.warning("No SMILES available for visualization.")
        return

    smiles_json = (
        json.dumps(smiles)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )

    html_code = """
    <div style="
        background: white;
        padding: 12px;
        border: 1px solid #dfe3e8;
        border-radius: 8px;
        text-align: center;
        min-height: 390px;
    ">
        <svg
            id="molecule"
            width="520"
            height="340"
            xmlns="http://www.w3.org/2000/svg">
        </svg>
        <p id="status"
           style="color:#b91c1c; font-family:Arial; font-size:14px;">
        </p>
    </div>

    <script src="https://unpkg.com/smiles-drawer@2.4.1/dist/smiles-drawer.min.js"></script>

    <script>
        const smiles = SMILES_PLACEHOLDER;
        const status = document.getElementById("status");

        try {
            if (!window.SmilesDrawer) {
                throw new Error("SmilesDrawer library could not be loaded.");
            }

            const drawer = new SmilesDrawer.SmiDrawer({
                width: 520,
                height: 340,
                bondLength: 19,
                bondThickness: 1.1,
                padding: 18
            });

            drawer.draw(
                smiles,
                "#molecule",
                "light",
                false,
                function(error) {
                    status.textContent = "Unable to render molecule: " + error;
                }
            );

        } catch (error) {
            status.textContent = "Rendering error: " + error.message;
        }
    </script>
    """

    html_code = html_code.replace("SMILES_PLACEHOLDER", smiles_json)

    components.html(html_code, height=390, scrolling=False)

# --------------------------------------------------
# Initialize model and data
# --------------------------------------------------

try:
    model, tokenizer, device = get_model()
    validation_data = get_validation_data()
except Exception as exc:
    st.error(f"Application initialization failed: {exc}")
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("MS-BART Molecular Structure Workbench")
st.write(
    "Generate candidate molecular structures from MIST-derived "
    "fingerprint representations using a fine-tuned MS-BART model."
)

st.info(
    "Research demonstration: the model generates candidate structures "
    "from molecular fingerprint tokens, not directly from raw MS/MS spectra. "
    "Predictions are exploratory outputs and are not confirmed molecular identifications."
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Model Information")
st.sidebar.write("**Model:** Fine-tuned MS-BART")
st.sidebar.write(f"**Device:** {device}")
st.sidebar.write("**Training examples:** 128")
st.sidebar.write("**Validation examples:** 32")
st.sidebar.write("**Checkpoint:** ms_bart_finetuned_128_continued")

st.sidebar.divider()

st.sidebar.header("Evaluation Snapshot")
st.sidebar.write("**Pilot result:** 0 canonical exact matches out of 32 generated structures")
st.sidebar.caption(
    "This reflects earlier pilot evaluation work and is shown here for project context. "
    "The current app demonstrates inference and comparison, not full benchmark scoring."
)

st.sidebar.divider()

st.sidebar.caption(
    "This application is intended as a research and portfolio demonstration. "
    "Generated structures should be interpreted cautiously."
)

# --------------------------------------------------
# Input section
# --------------------------------------------------

st.subheader("1. Select a Validation Example")

example_index = st.selectbox(
    "Validation record",
    options=range(len(validation_data)),
    format_func=lambda i: str(validation_data.iloc[i]["identifier"]),
)

example = validation_data.iloc[example_index]

fingerprint = st.text_area(
    "Molecular fingerprint tokens",
    value=str(example["fps"]),
    height=140,
    key=f"fingerprint_{example_index}",
)

# --------------------------------------------------
# Prediction section
# --------------------------------------------------

st.subheader("2. Generate Candidate Structure")

if st.button("Generate Prediction", type="primary"):
    try:
        with st.spinner("Generating candidate structure..."):
            prediction = generate_prediction(
                fingerprint,
                model,
                tokenizer,
                device
            )
            predicted_smiles = decode_structure(prediction)

        matches_validation = fingerprint.strip() == str(example["fps"]).strip()

        reference_selfies = str(example["selfies"]) if matches_validation else None
        reference_smiles = (
            decode_structure(reference_selfies)
            if reference_selfies is not None
            else None
        )

        st.success("Prediction generated successfully.")

        # ------------------------------------------
        # Side-by-side structure comparison
        # ------------------------------------------

        st.subheader("3. Molecular Structure Comparison")

        col1, col2 = st.columns(2)

        # ------------------------------------------
        # Predicted structure column
        # ------------------------------------------

        with col1:
            st.markdown("### Predicted Structure")

            st.write("**Generated SELFIES**")
            st.code(prediction, language=None)

            if predicted_smiles is not None:
                st.write("**Decoded SMILES**")
                st.code(predicted_smiles, language=None)

                st.write("**2D Predicted Structure**")
                render_molecule(predicted_smiles)

                st.info(
                    "SELFIES decoding succeeded. Independent chemical validation "
                    "has not been performed."
                )
            else:
                st.warning("The generated SELFIES could not be decoded.")

        # ------------------------------------------
        # Reference structure column
        # ------------------------------------------

        with col2:
            st.markdown("### Reference Structure")

            if matches_validation:
                st.write("**Reference SELFIES**")
                st.code(reference_selfies, language=None)

                if reference_smiles is not None:
                    st.write("**Reference SMILES**")
                    st.code(reference_smiles, language=None)

                    st.write("**2D Reference Structure**")
                    render_molecule(reference_smiles)
                else:
                    st.warning("The reference SELFIES could not be decoded.")

                st.caption(
                    "The reference structure is displayed for comparison only. "
                    "It was not provided to the model during inference."
                )
            else:
                st.info(
                    "The fingerprint was modified. Reference comparison is disabled "
                    "because the current input no longer matches the selected validation example."
                )

        # ------------------------------------------
        # Optional interpretation section
        # ------------------------------------------

        st.subheader("4. Interpretation Notes")

        st.write(
            "This interface allows visual comparison between the predicted molecular structure "
            "and the held-out reference structure associated with the selected validation example."
        )

        st.write(
            "A decoded and rendered structure does not establish that the prediction is correct. "
            "It only shows that the generated SELFIES could be translated into a SMILES string "
            "and displayed as a 2D diagram."
        )

        if matches_validation and predicted_smiles is not None and reference_smiles is not None:
            st.write(
                "In this example, the predicted and reference structures can be compared visually. "
                "Differences between them help illustrate that chemically interpretable output does "
                "not necessarily correspond to the expected target molecule."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# --------------------------------------------------
# Project notes / limitations
# --------------------------------------------------

st.divider()

with st.expander("5. Project Notes and Limitations"):
    st.write(
        "This application demonstrates exploratory molecular candidate generation using a "
        "fine-tuned MS-BART model trained on MIST-derived fingerprint token representations."
    )

    st.write(
        "The app is designed to show model inference, SELFIES decoding, and side-by-side "
        "comparison with a held-out reference structure. It is not a production chemical "
        "identification system."
    )

    st.write(
        "Important limitations include the small pilot scale, the absence of integrated chemical "
        "validity checks, and the fact that displayed 2D structures do not independently confirm "
        "molecular correctness."
    )

    st.write(
        "Future improvements could include larger-scale evaluation, similarity scoring, "
        "chemical validation, and structure comparison metrics beyond exact-match inspection."
    )

st.caption(
    "This application demonstrates exploratory molecular candidate generation using a fine-tuned "
    "MS-BART model. SELFIES decoding and 2D visualization do not establish chemical validity or "
    "correct molecular identification."
)
