"""
High-Speed Transfer Learning Pipeline for 10-Class Plant Nutrient Deficiency & OOD Detection.
Pre-extracts ImageNet features using MobileNetV2 backbone (zero memory overhead)
and trains the 10-class diagnostic & rejection classifier head.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, datasets, transforms

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    SAVED_MODEL_PATH,
    HISTORY_PATH,
    IMAGE_SIZE
)
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import save_checkpoint

def extract_features(backbone, loader, device):
    """Extracts 1280-dim feature vectors without storing gradient graphs."""
    backbone.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            # MobileNetV2 features + Global Avg Pool
            feats = backbone.features(images)
            feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            feats = torch.flatten(feats, 1)

            features_list.append(feats.cpu())
            labels_list.append(targets)

            if (idx + 1) % 40 == 0:
                print(f"   Processed batch {idx + 1}/{len(loader)}...", flush=True)

    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    return all_features, all_labels

def run_fast_training(epochs: int = 15, lr: float = 1e-3, batch_size: int = 64):
    print("=" * 65)
    print("Starting Fast 10-Class Transfer Learning with Feature Caching")
    print("=" * 65, flush=True)

    device = torch.device("cpu")
    print(f"Device: {device}", flush=True)

    # Standard clean transforms (Resize + Normalize)
    tfs = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_raw = datasets.ImageFolder(str(TRAIN_DIR), transform=tfs)
    val_raw = datasets.ImageFolder(str(VAL_DIR), transform=tfs)
    test_raw = datasets.ImageFolder(str(TEST_DIR), transform=tfs)

    class_names = train_raw.classes
    print(f"Detected {len(class_names)} classes: {class_names}", flush=True)
    print(f"Dataset split: Train={len(train_raw)} | Val={len(val_raw)} | Test={len(test_raw)}", flush=True)

    raw_train_loader = DataLoader(train_raw, batch_size=32, shuffle=False)
    raw_val_loader = DataLoader(val_raw, batch_size=32, shuffle=False)
    raw_test_loader = DataLoader(test_raw, batch_size=32, shuffle=False)

    print("\n1. Loading Pretrained MobileNetV2 Backbone...", flush=True)
    base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    base_model = base_model.to(device)

    print("2. Pre-extracting Features for Train Set...", flush=True)
    t0 = time.time()
    train_x, train_y = extract_features(base_model, raw_train_loader, device)
    print(f"   Train features extracted: {train_x.shape} in {time.time()-t0:.1f}s", flush=True)

    print("3. Pre-extracting Features for Validation Set...", flush=True)
    val_x, val_y = extract_features(base_model, raw_val_loader, device)
    print(f"   Val features extracted: {val_x.shape}", flush=True)

    # Create TensorDataLoaders for instant CPU in-memory training
    train_feat_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_feat_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    # Define the 10-class Classifier Head
    in_features = base_model.classifier[1].in_features
    head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(class_names))
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n4. Training Classifier Head...", flush=True)
    best_val_acc = 0.0
    best_head_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    train_start = time.time()
    for ep in range(1, epochs + 1):
        head.train()
        running_loss, correct, total = 0.0, 0, 0
        for bx, by in train_feat_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        tr_loss = running_loss / total
        tr_acc = correct / total

        # Validation
        head.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_feat_loader:
                bx, by = bx.to(device), by.to(device)
                out = head(bx)
                loss = criterion(out, by)
                v_loss += loss.item() * bx.size(0)
                preds = torch.argmax(out, dim=1)
                v_corr += (preds == by).sum().item()
                v_tot += by.size(0)

        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"   Epoch [{ep:02d}/{epochs:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = head.state_dict().copy()

    # Assemble complete model
    print("\n5. Assembling Complete End-to-End MobileNetV2 Model...", flush=True)
    full_model = models.mobilenet_v2(weights=None)
    full_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(class_names))
    )
    # Load pretrained backbone features
    full_model.features.load_state_dict(base_model.features.state_dict())
    # Load best trained classifier head
    full_model.classifier.load_state_dict(best_head_state)

    # Save full model checkpoint
    save_checkpoint(
        model=full_model,
        filepath=SAVED_MODEL_PATH,
        class_names=class_names,
        architecture="mobilenet_v2",
        val_accuracy=best_val_acc
    )

    # Save history
    history["epochs"] = epochs
    history["architecture"] = "mobilenet_v2"
    history["class_names"] = class_names
    history["total_time_seconds"] = round(time.time() - train_start, 2)
    history["best_val_acc"] = round(best_val_acc, 4)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2%}")
    print(f"Checkpoint saved to: {SAVED_MODEL_PATH}")

if __name__ == "__main__":
    run_fast_training(epochs=20, lr=1e-3, batch_size=64)
