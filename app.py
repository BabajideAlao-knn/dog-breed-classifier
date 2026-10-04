"""Gradio web app: upload a dog photo and get the predicted breed.

Run locally:      python app.py
Deploy:           Hugging Face Spaces (Gradio SDK) — see README.md
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from model_utils import load_class_names, load_model, predict

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
WEIGHTS_PATH = MODEL_DIR / "dog_breed_efficientnet_b0.pth"
CLASSES_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
BREED_INFO_PATH = ROOT / "breed_info.json"
EXAMPLES_DIR = ROOT / "examples"

CONFIDENCE_THRESHOLD = 0.50  # below this, warn that the photo may not be one of the 10 breeds

# --------------------------------------------------------------------------- load
if WEIGHTS_PATH.exists():
    MODEL, CLASS_NAMES = load_model(WEIGHTS_PATH, CLASSES_PATH)
else:
    MODEL, CLASS_NAMES = None, load_class_names(CLASSES_PATH)
    print(f"WARNING: {WEIGHTS_PATH} not found. Train the model with the notebook first.")

BREED_INFO = json.loads(BREED_INFO_PATH.read_text()) if BREED_INFO_PATH.exists() else {}
METRICS = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

EXAMPLES = sorted(
    str(p) for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
) if EXAMPLES_DIR.exists() else []


# ------------------------------------------------------------------------ predict
def describe(breed: str, confidence: float) -> str:
    info = BREED_INFO.get(breed.lower())
    lines = [f"## {breed}", f"**Confidence:** {confidence:.1%}"]
    if confidence < CONFIDENCE_THRESHOLD:
        lines.append(
            "\n⚠️ *The model isn't very sure. It only knows these breeds: "
            + ", ".join(CLASS_NAMES)
            + ". Mixed breeds, other breeds or non-dog photos can give unreliable results.*"
        )
    if info:
        lines += [
            "",
            f"| Origin | Size | Temperament |\n|---|---|---|\n"
            f"| {info['origin']} | {info['size']} | {info['temperament']} |",
            "",
            f"💡 {info['fact']}",
        ]
    return "\n".join(lines)


def classify(image):
    if image is None:
        return None, "Upload a photo of a dog to get a prediction."
    if MODEL is None:
        raise gr.Error("Model weights not found. Train the model with the notebook and put the "
                       "weights in the model/ folder.")
    probs = predict(MODEL, image, CLASS_NAMES, top_k=5)
    top_breed, top_prob = next(iter(probs.items()))
    return probs, describe(top_breed, top_prob)


# ----------------------------------------------------------------------------- UI
metrics_line = ""
if METRICS.get("test_accuracy") is not None:
    metrics_line = (
        f" Test accuracy: **{METRICS['test_accuracy']:.1%}** "
        f"(top-3: {METRICS.get('test_top3_accuracy', 0):.1%}) on {METRICS.get('test_images', '?')} held-out images."
    )

with gr.Blocks(title="Dog Breed Classifier") as demo:
    gr.Markdown(
        "# 🐶 Dog Breed Classifier\n"
        "Upload a photo of a dog and the model will predict its breed. "
        f"It recognises **{len(CLASS_NAMES)} breeds**: {', '.join(CLASS_NAMES)}."
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Dog photo", height=380,
                                sources=["upload", "webcam", "clipboard"])
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                submit_btn = gr.Button("Predict breed", variant="primary")
        with gr.Column(scale=1):
            label_out = gr.Label(num_top_classes=5, label="Top predictions")
            details_out = gr.Markdown("Upload a photo of a dog to get a prediction.")

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=image_in, outputs=[label_out, details_out],
                    fn=classify, cache_examples=False, label="Try an example")

    gr.Markdown(
        "---\n"
        "**Model:** EfficientNet-B0 pretrained on ImageNet, fine-tuned with PyTorch on the "
        "[Kaggle Dog Breed Image Dataset](https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset)."
        + metrics_line
    )

    submit_btn.click(classify, inputs=image_in, outputs=[label_out, details_out])
    image_in.upload(classify, inputs=image_in, outputs=[label_out, details_out])
    clear_btn.add([image_in, label_out, details_out])


if __name__ == "__main__":
    demo.launch()
