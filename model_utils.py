"""Shared model + preprocessing utilities.

Used by the Streamlit app (streamlit_app.py). The training notebook defines the exact same
architecture and transforms, so weights saved there load here unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B0 with a new classification head (must match the notebook)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_class_names(path: Union[str, Path]) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["classes"] if isinstance(data, dict) else list(data)


def load_model(
    weights_path: Union[str, Path],
    class_names_path: Union[str, Path],
    device: str = "cpu",
) -> Tuple[nn.Module, List[str]]:
    class_names = load_class_names(class_names_path)
    model = build_model(len(class_names))
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, class_names


_EVAL_TF = get_eval_transform()


@torch.inference_mode()
def predict(
    model: nn.Module,
    image: Union[Image.Image, np.ndarray],
    class_names: List[str],
    top_k: int = 5,
    device: str = "cpu",
) -> Dict[str, float]:
    """Return {breed: probability} for the top_k classes, highest first."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    x = _EVAL_TF(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu()
    k = min(top_k, len(class_names))
    values, indices = probs.topk(k)
    return {class_names[i]: float(v) for v, i in zip(values.tolist(), indices.tolist())}
