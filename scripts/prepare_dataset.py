"""
Dataset Preparation and Standardization Script.
Ingests images from downloaded Kaggle datasets (Banana and Rice nutrient deficiency folders),
normalizes class labels, balances image counts per category, and splits into train/val/test sets.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

# Import configuration
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BANANA_RAW_DIR,
    BANANA_AUG_DIR,
    RICE_RAW_DIR,
    TARGET_CLASSES
)

def verify_and_clean_image(filepath: Path) -> bool:
    """Verifies that an image file is valid, readable, and not corrupted."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Verify it can be converted to RGB without error
        with Image.open(filepath) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False

def collect_images_by_class(max_per_class: int = 500):
    """
    Collects raw and augmented image paths for each target class.
    Balancing to prevent class imbalance issues.
    """
    class_images = {cls_name: [] for cls_name in TARGET_CLASSES}

    # 1. Collect from Rice dataset (Nitrogen, Phosphorus, Potassium)
    rice_map = {
        "Nitrogen(N)": "Nitrogen (N)",
        "Phosphorus(P)": "Phosphorus (P)",
        "Potassium(K)": "Potassium (K)"
    }
    if RICE_RAW_DIR.exists():
        print(f"Reading Rice dataset from: {RICE_RAW_DIR}")
        for folder_name, target_cls in rice_map.items():
            folder_path = RICE_RAW_DIR / folder_name
            if folder_path.exists():
                files = [p for p in folder_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                class_images[target_cls].extend(files)
                print(f"  Found {len(files)} files for {target_cls} in Rice dataset.")

    # 2. Collect from Banana dataset (Calcium, Healthy, Iron, Magnesium, Potassium, Sulphur, Zinc)
    banana_map = {
        "calcium": "Calcium (Ca)",
        "healthy": "Healthy",
        "iron": "Iron (Fe)",
        "magnesium": "Magnesium (Mg)",
        "potassium": "Potassium (K)",
        "sulphur": "Sulphur (S)",
        "zinc": "Zinc (Zn)"
    }

    # Check RAW banana images first
    if BANANA_RAW_DIR.exists():
        print(f"Reading Banana RAW dataset from: {BANANA_RAW_DIR}")
        for folder_name, target_cls in banana_map.items():
            folder_path = BANANA_RAW_DIR / folder_name
            if folder_path.exists():
                files = [p for p in folder_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                class_images[target_cls].extend(files)
                print(f"  Found {len(files)} files for {target_cls} in Banana RAW.")

    # For classes with fewer images (e.g. Iron, Magnesium), augment from BANANA_AUG_DIR if needed
    if BANANA_AUG_DIR.exists():
        print(f"Supplementing with Banana Augmented dataset if needed from: {BANANA_AUG_DIR}")
        for folder_name, target_cls in banana_map.items():
            current_count = len(class_images[target_cls])
            if current_count < 350:
                folder_path = BANANA_AUG_DIR / folder_name
                if folder_path.exists():
                    aug_files = [p for p in folder_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                    needed = 450 - current_count
                    sampled = random.sample(aug_files, min(len(aug_files), needed))
                    class_images[target_cls].extend(sampled)
                    print(f"  Added {len(sampled)} augmented images for {target_cls} (total: {len(class_images[target_cls])})")

    # Limit and shuffle per class to ensure clean, balanced dataset
    final_dataset = {}
    for cls_name, paths in class_images.items():
        # Remove duplicates
        unique_paths = list(set(paths))
        random.seed(42)
        random.shuffle(unique_paths)
        if max_per_class:
            unique_paths = unique_paths[:max_per_class]
        final_dataset[cls_name] = unique_paths
        print(f"Class '{cls_name}': {len(unique_paths)} valid candidate images.")

    return final_dataset

def build_split_dataset(train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Copies images into train/val/test split directories.
    """
    print("\n--- Preparing Standardized Dataset ---")
    dataset_by_class = collect_images_by_class(max_per_class=400)

    # Clean existing processed data directory if present
    if PROCESSED_DATA_DIR.exists():
        print(f"Cleaning existing processed directory: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)

    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)

    summary = {"train": 0, "val": 0, "test": 0, "per_class": {}}

    for cls_name, filepaths in dataset_by_class.items():
        if not filepaths:
            print(f"WARNING: No images found for {cls_name}")
            continue

        # Sanitize folder name
        safe_cls_dir = cls_name.replace(" ", "_").replace("(", "").replace(")", "")
        (TRAIN_DIR / safe_cls_dir).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / safe_cls_dir).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / safe_cls_dir).mkdir(parents=True, exist_ok=True)

        n = len(filepaths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_files = filepaths[:n_train]
        val_files = filepaths[n_train:n_train + n_val]
        test_files = filepaths[n_train + n_val:]

        summary["per_class"][cls_name] = {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": n
        }

        # Helper to copy and verify
        def copy_files(files, dest_dir):
            copied = 0
            for idx, fpath in enumerate(files):
                ext = fpath.suffix.lower()
                dest_file = dest_dir / f"{safe_cls_dir}_{idx:04d}{ext}"
                try:
                    if verify_and_clean_image(fpath):
                        shutil.copy2(fpath, dest_file)
                        copied += 1
                except Exception as e:
                    pass
            return copied

        c_train = copy_files(train_files, TRAIN_DIR / safe_cls_dir)
        c_val = copy_files(val_files, VAL_DIR / safe_cls_dir)
        c_test = copy_files(test_files, TEST_DIR / safe_cls_dir)

        summary["train"] += c_train
        summary["val"] += c_val
        summary["test"] += c_test

        print(f"Processed '{cls_name}': {c_train} train | {c_val} val | {c_test} test")

    print("\n=== Dataset Curation Complete ===")
    print(f"Total Train Images: {summary['train']}")
    print(f"Total Val Images  : {summary['val']}")
    print(f"Total Test Images : {summary['test']}")
    print(f"Grand Total       : {summary['train'] + summary['val'] + summary['test']}")
    return summary

if __name__ == "__main__":
    build_split_dataset()
