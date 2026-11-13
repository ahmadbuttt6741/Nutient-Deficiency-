"""
Model Architecture Definition for Plant Nutrient Deficiency Classification.
Supports Transfer Learning with MobileNetV2, EfficientNet-B0, and ResNet18/50.
"""

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

def build_model(
    architecture: str = "mobilenet_v2",
    num_classes: int = 9,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Constructs a transfer learning model with a custom classification head.
    """
    arch = architecture.lower()
    
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model

def save_checkpoint(
    model: nn.Module,
    filepath: Path,
    class_names: list,
    architecture: str = "mobilenet_v2",
    val_accuracy: float = 0.0
):
    """Saves model weights along with metadata (classes, architecture, accuracy)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "architecture": architecture,
        "class_names": class_names,
        "state_dict": model.state_dict(),
        "val_accuracy": val_accuracy
    }
    torch.save(state, str(filepath))
    print(f"Model checkpoint saved successfully to {filepath}")

def load_checkpoint(
    filepath: Path,
    device: Optional[torch.device] = None
) -> tuple[nn.Module, list, str, float]:
    """Loads model from checkpoint and sets to eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(str(filepath), map_location=device)
    architecture = checkpoint.get("architecture", "mobilenet_v2")
    class_names = checkpoint.get("class_names", [])
    val_acc = checkpoint.get("val_accuracy", 0.0)

    model = build_model(
        architecture=architecture,
        num_classes=len(class_names),
        pretrained=False
    )
    state_dict = checkpoint["state_dict"]

    # Adapt classifier head topology to match checkpoint exactly
    if architecture == "mobilenet_v2" and "classifier.1.weight" in state_dict:
        hidden_dim = state_dict["classifier.1.weight"].shape[0]
        in_feat = state_dict["classifier.1.weight"].shape[1]
        has_bn = any("classifier.2.running_mean" in k for k in state_dict)
        if has_bn:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.35),
                nn.Linear(in_feat, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(hidden_dim, len(class_names))
            )
        elif hidden_dim != 256:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_feat, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.15),
                nn.Linear(hidden_dim, len(class_names))
            )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names, architecture, val_acc
