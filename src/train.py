"""
Training Pipeline for Plant Nutrient Deficiency Detection Model.
Uses Transfer Learning (MobileNetV2 / EfficientNet / ResNet), AdamW,
and Cosine Annealing learning rate schedule.
"""

import sys
import json
import time
import gc
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import (
    SAVED_MODEL_PATH,
    HISTORY_PATH,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY
)
from src.dataset import get_dataloaders
from src.model import build_model, save_checkpoint

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_training(
    architecture: str = "mobilenet_v2",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE
):
    print("=" * 60)
    print(f"Starting Training: {architecture.upper()}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=batch_size)
    print(f"Detected {len(class_names)} classes: {class_names}")
    print(f"Dataset sizes - Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = build_model(
        architecture=architecture,
        num_classes=len(class_names),
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": epochs,
        "architecture": architecture,
        "class_names": class_names
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        duration = time.time() - t0

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                filepath=SAVED_MODEL_PATH,
                class_names=class_names,
                architecture=architecture,
                val_accuracy=val_acc
            )
            print(f"  --> Saved new best checkpoint with Val Acc: {val_acc:.2%}", flush=True)

        gc.collect()

    total_time = time.time() - start_time
    history["total_time_seconds"] = round(total_time, 2)
    history["best_val_acc"] = round(best_val_acc, 4)

    # Save training history JSON
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training Complete in {total_time/60:.2f} mins. Best Validation Accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {SAVED_MODEL_PATH}")
    print(f"History saved to: {HISTORY_PATH}")
    print("=" * 60)

    return history

if __name__ == "__main__":
    run_training()
