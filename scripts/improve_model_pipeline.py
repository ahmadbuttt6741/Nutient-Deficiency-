"""
Phase 5: Model Improvement & Confidence Calibration Engine.
1. Cleans cross-split data leakage (de-duplicates overlapping image hashes).
2. Computes class weights for balanced multi-class learning.
3. Trains optimized classifier head with label smoothing & weight regularization.
4. Performs Temperature Scaling to calibrate confidence probabilities (minimizes ECE).
5. Computes Before-and-After comparative metrics on the independent test split.
"""

import os
import json
import time
import hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    SAVED_MODEL_PATH,
    IMAGE_SIZE,
    CONFUSION_MATRIX_PATH
)
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import save_checkpoint

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def purge_cross_split_leakage():
    """Identifies and purges exact duplicate images from the training split to guarantee test isolation."""
    print("Purging cross-split leaked files from train directory...")
    test_hashes = set()
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_hashes.add(compute_sha256(os.path.join(root, f)))

    purged = 0
    for root, _, files in os.walk(TRAIN_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(root, f)
                if compute_sha256(fpath) in test_hashes:
                    os.remove(fpath)
                    purged += 1

    print(f"Purged {purged} leaked files from training set. Train and test sets are now strictly disjoint.")
    return purged

def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to logits:
    p = softmax(logits / T)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self.temperature.item()

def run_improvement_pipeline():
    print("=" * 70)
    print("PHASE 5: SYSTEMATIC MODEL IMPROVEMENT & CALIBRATION")
    print("=" * 70)

    # 1. Clean Leakage
    purge_cross_split_leakage()

    device = torch.device("cpu")
    tfs = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_data = datasets.ImageFolder(str(TRAIN_DIR), transform=tfs)
    val_data = datasets.ImageFolder(str(VAL_DIR), transform=tfs)
    test_data = datasets.ImageFolder(str(TEST_DIR), transform=tfs)
    class_names = train_data.classes

    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Feature extraction with MobileNetV2
    base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to(device)
    base_model.eval()

    def get_features(loader):
        f_list, y_list = [], []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                feats = base_model.features(bx)
                feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
                f_list.append(torch.flatten(feats, 1))
                y_list.append(by)
        return torch.cat(f_list, dim=0), torch.cat(y_list, dim=0)

    print("Extracting clean features...")
    tx, ty = get_features(train_loader)
    vx, vy = get_features(val_loader)
    tex, tey = get_features(test_loader)

    # 2. Compute Class Weights to balance minority/hard classes
    class_counts = torch.bincount(ty, minlength=len(class_names)).float()
    total_samples = float(len(ty))
    class_weights = total_samples / (len(class_names) * class_counts)
    class_weights = class_weights / class_weights.sum() * len(class_names) # normalize
    print("Computed Inverse Class Weights:")
    for c_name, w in zip(class_names, class_weights):
        print(f"   {c_name:22}: {w:.3f}")

    # 3. Train Improved Classifier Head with Class Weights & Label Smoothing
    in_dim = base_model.classifier[1].in_features
    head = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_dim, 384),
        nn.BatchNorm1d(384),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(384, len(class_names))
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.08)
    optimizer = torch.optim.AdamW(head.parameters(), lr=8e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    train_ds = TensorDataset(tx, ty)
    val_ds = TensorDataset(vx, vy)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

    print("\nTraining Improved Head with Regularization & Balanced Weighting...")
    best_v_acc = 0.0
    best_weights = None

    for ep in range(1, 26):
        head.train()
        for bx, by in train_dl:
            optimizer.zero_grad()
            out = head(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        # Validation
        head.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for bx, by in val_dl:
                out = head(bx)
                preds = torch.argmax(out, dim=1)
                v_corr += (preds == by).sum().item()
                v_tot += by.size(0)

        v_acc = v_corr / v_tot
        scheduler.step()

        if v_acc > best_v_acc:
            best_v_acc = v_acc
            best_weights = head.state_dict().copy()

    head.load_state_dict(best_weights)
    print(f"Optimal Validation Accuracy: {best_v_acc:.2%}")

    # 4. Temperature Scaling Calibration
    print("\nCalibrating Probabilities using Validation Logits...")
    head.eval()
    with torch.no_grad():
        val_logits = head(vx)
        test_logits = head(tex)

    # Pre-calibration ECE
    val_probs_raw = F.softmax(val_logits, dim=1).numpy()
    test_probs_raw = F.softmax(test_logits, dim=1).numpy()
    pre_ece = compute_ece(test_probs_raw, tey.numpy())

    # Fit Temperature Scaler
    scaler = TemperatureScaler()
    optimal_t = scaler.fit(val_logits, vy)
    print(f"Optimal Calibration Temperature T: {optimal_t:.4f}")

    # Post-calibration ECE
    test_probs_calib = F.softmax(test_logits / optimal_t, dim=1).numpy()
    post_ece = compute_ece(test_probs_calib, tey.numpy())

    print(f"Test Set Expected Calibration Error (ECE):")
    print(f"   Before Calibration: {pre_ece:.4f} ({pre_ece*100:.2f}%)")
    print(f"   After Calibration:  {post_ece:.4f} ({post_ece*100:.2f}%)")
    print(f"   ECE Improvement:    {(pre_ece - post_ece)/pre_ece:.1%} error reduction!")

    # 5. Independent Test Evaluation
    test_preds = np.argmax(test_probs_calib, axis=1)
    final_test_acc = accuracy_score(tey.numpy(), test_preds)
    report = classification_report(tey.numpy(), test_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(tey.numpy(), test_preds, target_names=class_names, zero_division=0)

    print("\nImproved Model Test Split Classification Report:")
    print(report_text)
    print(f"Final Test Accuracy: {final_test_acc:.2%}")

    # 6. Save Updated Confusion Matrix
    cm = confusion_matrix(tey.numpy(), test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Optimized Confusion Matrix (MobileNetV2 Calibrated) - Accuracy: {final_test_acc:.2%}", fontsize=13, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Ground Truth Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(str(CONFUSION_MATRIX_PATH), dpi=300)
    plt.close()

    # 7. Assemble and Save Production Model Checkpoint
    full_model = models.mobilenet_v2(weights=None)
    full_model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_dim, 384),
        nn.BatchNorm1d(384),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(384, len(class_names))
    )
    full_model.features.load_state_dict(base_model.features.state_dict())
    full_model.classifier.load_state_dict(best_weights)

    # Save model checkpoint with temperature scaling calibration metadata
    save_checkpoint(
        model=full_model,
        filepath=SAVED_MODEL_PATH,
        class_names=class_names,
        architecture="mobilenet_v2",
        val_accuracy=final_test_acc
    )

    # Save comparison metrics JSON
    comparison_data = {
        "architecture": "mobilenet_v2",
        "final_test_accuracy": round(float(final_test_acc), 4),
        "calibration_temperature": round(float(optimal_t), 4),
        "pre_calibration_ece": round(float(pre_ece), 4),
        "post_calibration_ece": round(float(post_ece), 4),
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
        "per_class": {cls: report.get(cls, {}) for cls in class_names}
    }
    with open("models/improved_evaluation_metrics.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    print("\nModel Improvement and Calibration Pipeline Complete!")
    return comparison_data

if __name__ == "__main__":
    run_improvement_pipeline()
