"""Model definitions, checkpoint loading, and inference pipeline.

Supports torchvision ResNet-18 and Hugging Face ViT-B/16 models
fine-tuned for binary e-vaporiser classification.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

try:
    from transformers import ViTForImageClassification
except ImportError:
    ViTForImageClassification = None


class ViTWrapper(nn.Module):
    """Wraps Hugging Face ViTForImageClassification to return plain logits."""

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=x).logits


def build_model(backbone: str, num_classes: int, vit_checkpoint: str = "google/vit-base-patch16-224") -> nn.Module:
    """Recreate the network architecture so weights can be loaded cleanly."""
    backbone_lower = backbone.lower()
    if backbone_lower == "resnet":
        net = resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net

    if backbone_lower == "vit":
        if ViTForImageClassification is None:
            raise RuntimeError("Hugging Face transformers is required to instantiate ViT.")
        vit = ViTForImageClassification.from_pretrained(
            vit_checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        return ViTWrapper(vit)

    raise ValueError(f"Unknown backbone: {backbone}")


class LoadedModel:
    """Holds a trained model alongside its preprocessing transform and metadata."""

    def __init__(self, name: str, ckpt_path: str, device: torch.device):
        self.name = name
        self.device = device

        ckpt = torch.load(ckpt_path, map_location=device)
        self.class_names = ckpt.get("class_names", ["not_vape", "vape"])
        self.backbone = ckpt.get("backbone", "resnet")
        self.strategy = ckpt.get("strategy", "full")

        net = build_model(
            self.backbone,
            len(self.class_names),
            ckpt.get("vit_checkpoint", "google/vit-base-patch16-224"),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval().to(device)
        self.net = net

        norm = ckpt.get("norm", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
        img_size = ckpt.get("img_size", 224)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"]),
        ])

        self.vape_idx = (
            self.class_names.index("vape")
            if "vape" in self.class_names
            else len(self.class_names) - 1
        )

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Tuple[Dict[str, float], float]:
        """Predict class distribution and vape probability for a single PIL image."""
        x = self.transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.net(x)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
        scores = {self.class_names[i]: float(probs[i]) for i in range(len(probs))}
        return scores, float(probs[self.vape_idx])

    @property
    def pretty_name(self) -> str:
        return f"{self.backbone.upper()} · {self.strategy.replace('_', ' ')}"


class DemoFallbackModel:
    """Heuristic fallback used only if weights are not downloaded locally yet."""

    def __init__(self, name: str, backbone: str, strategy: str):
        self.name = name
        self.backbone = backbone
        self.strategy = strategy
        self.class_names = ["not_vape", "vape"]

    def predict(self, pil_img: Image.Image) -> Tuple[Dict[str, float], float]:
        # Simple heuristic for preview mode based on image aspect and color tones
        # Encourages users to run the notebook to train real weights
        w, h = pil_img.size
        ratio = max(w, h) / (min(w, h) + 1e-5)
        # Pen/cylindrical shapes typically have higher aspect ratio
        prob = min(0.92, max(0.08, (ratio - 1.0) * 0.35 + 0.15))
        return {"not_vape": 1.0 - prob, "vape": prob}, prob

    @property
    def pretty_name(self) -> str:
        return f"{self.backbone.upper()} · {self.strategy.replace('_', ' ')} (Demo)"


def load_models(models_dir: str, device: torch.device) -> List[LoadedModel]:
    """Load the top-2 model checkpoints from models_dir or manifest.json."""
    manifest_path = os.path.join(models_dir, "manifest.json")
    entries = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    elif os.path.isdir(models_dir):
        entries = [
            {"name": os.path.splitext(f)[0], "path": os.path.join(models_dir, f)}
            for f in sorted(os.listdir(models_dir))
            if f.endswith(".pt") or f.endswith(".pth")
        ]

    models = []
    for e in entries[:2]:
        local = os.path.join(models_dir, os.path.basename(e["path"]))
        path = local if os.path.exists(local) else e["path"]
        if os.path.exists(path):
            models.append(LoadedModel(e["name"], path, device))

    return models
