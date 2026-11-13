"""
PyTorch Dataset and DataLoaders for Plant Nutrient Deficiency Classification.
Includes standard data augmentations and ImageNet normalization.
"""

from pathlib import Path
from typing import Tuple, Dict, List
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS
)

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms() -> Dict[str, transforms.Compose]:
    """Returns data transform pipelines for train, val, and test splits."""
    return {
        "train": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]),
        "test": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]),
        "inference": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    }

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Constructs and returns train, val, and test DataLoaders along with detected class names.
    """
    tfs = get_transforms()

    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        raise FileNotFoundError(
            f"Train directory '{TRAIN_DIR}' does not exist or is empty. "
            "Please run 'python scripts/prepare_dataset.py' first."
        )

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=tfs["train"])
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=tfs["val"])
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=tfs["test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names
