"""
Comprehensive Model Evaluation Script.
Computes Test Accuracy, Precision, Recall, F1-Score, and generates Confusion Matrix.
Fulfills Section 9 of the Academic Project Proposal.
"""

import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import (
    SAVED_MODEL_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH
)
from src.dataset import get_dataloaders
from src.model import load_checkpoint

def run_evaluation():
    print("=" * 60)
    print("Starting Comprehensive Model Evaluation on Test Split")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not SAVED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {SAVED_MODEL_PATH}. Train model first.")

    model, class_names, architecture, _ = load_checkpoint(SAVED_MODEL_PATH, device=device)
    _, _, test_loader, _ = get_dataloaders()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Metrics Calculation
    test_acc = accuracy_score(all_targets, all_preds)
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report_text)
    print(f"Overall Test Accuracy: {test_acc:.2%}")

    # 2. Confusion Matrix Plotting
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix ({architecture.upper()}) - Accuracy: {test_acc:.2%}", fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Ground Truth Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(CONFUSION_MATRIX_PATH), dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {CONFUSION_MATRIX_PATH}")

    # 3. Save Metrics JSON
    metrics_data = {
        "architecture": architecture,
        "test_accuracy": round(float(test_acc), 4),
        "macro_avg": report_dict.get("macro avg", {}),
        "weighted_avg": report_dict.get("weighted avg", {}),
        "per_class": {
            cls: report_dict.get(cls, {})
            for cls in class_names
        }
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Metrics saved to: {METRICS_PATH}")
    print("=" * 60)
    return metrics_data

if __name__ == "__main__":
    run_evaluation()
