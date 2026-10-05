"""Streamlit web app: upload a dog photo and get the predicted breed.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud (see README.md)
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_utils import load_class_names, load_model, predict

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
WEIGHTS_PATH = MODEL_DIR / "dog_breed_efficientnet_b0.pth"
CLASSES_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
BREED_INFO_PATH = ROOT / "breed_info.json"
EXAMPLES_DIR = ROOT / "examples"

CONFIDENCE_THRESHOLD = 0.50  # below this, warn that the photo may not be one of the 10 breeds
GITHUB_URL = "https://github.com/BabajideAlao-knn/dog-breed-classifier"
DATASET_URL = "https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset"

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")


# ------------------------------------------------------------------ cached loads
@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    if not WEIGHTS_PATH.exists():
        return None, load_class_names(CLASSES_PATH)
    return load_model(WEIGHTS_PATH, CLASSES_PATH)


@st.cache_data
def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


model, class_names = get_model()
breed_info = read_json(BREED_INFO_PATH)
metrics = read_json(METRICS_PATH)
examples = sorted(
    p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
) if EXAMPLES_DIR.exists() else []


# ----------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the model")
    st.markdown(
        "**EfficientNet-B0** pretrained on ImageNet and fine-tuned with PyTorch on the "
        f"[Kaggle Dog Breed Image Dataset]({DATASET_URL})."
    )
    if metrics.get("test_accuracy") is not None:
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        c2.metric("Test images", metrics.get("test_images", "?"))
    st.markdown("**Breeds it knows**")
    st.markdown("\n".join(f"- {b}" for b in class_names))
    st.markdown("---")
    st.markdown(f"Built by **Babajide Alao** · [Source code on GitHub]({GITHUB_URL})")


# -------------------------------------------------------------------------- main
st.title("🐶 Dog Breed Classifier")
st.write(
    f"Upload a photo of a dog and the model will predict its breed. "
    f"It recognises **{len(class_names)} breeds** (see the sidebar)."
)

if model is None:
    st.error("Model weights not found in `model/`. Train the model with the notebook first.")
    st.stop()

tab_upload, tab_camera = st.tabs(["📁 Upload a photo", "📷 Take a photo"])
with tab_upload:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
with tab_camera:
    snapshot = st.camera_input("Take a photo", label_visibility="collapsed")

source = uploaded or snapshot
if source is None and examples:
    choice = st.selectbox("…or try an example", ["—"] + [p.name for p in examples])
    if choice != "—":
        source = EXAMPLES_DIR / choice

if source is None:
    st.info("Upload a dog photo to get a prediction.")
    st.stop()

try:
    image = Image.open(source).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("That file couldn't be read as an image. Please try a JPG or PNG.")
    st.stop()

with st.spinner("Predicting ..."):
    probs = predict(model, image, class_names, top_k=5)
top_breed, top_prob = next(iter(probs.items()))

col_img, col_pred = st.columns([1, 1], gap="large")
with col_img:
    st.image(image, width="stretch")

with col_pred:
    st.subheader(top_breed)
    st.metric("Confidence", f"{top_prob:.1%}")
    if top_prob < CONFIDENCE_THRESHOLD:
        st.warning(
            "The model isn't very sure. Mixed breeds, other breeds or non-dog photos "
            "can give unreliable results."
        )
    st.markdown("**Top predictions**")
    for breed, p in probs.items():
        st.progress(p, text=f"{breed} — {p:.1%}")

info = breed_info.get(top_breed.lower())
if info:
    st.markdown("---")
    st.markdown(f"#### About the {top_breed}")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Origin**  \n{info['origin']}")
    c2.markdown(f"**Size**  \n{info['size']}")
    c3.markdown(f"**Temperament**  \n{info['temperament']}")
    st.info(f"💡 {info['fact']}")
