import os
import hashlib
from collections import defaultdict
from PIL import Image

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def audit_quality_and_leakage():
    print("=" * 65)
    print("PHASE 4: DATASET QUALITY & LEAKAGE AUDIT")
    print("=" * 65)

    splits = ["train", "val", "test"]
    split_hashes = defaultdict(dict) # split -> hash -> filepath
    all_hashes = defaultdict(list)   # hash -> [(split, filepath)]
    corrupted_files = []
    dimension_stats = defaultdict(int)
    channel_stats = defaultdict(int)

    total_checked = 0

    for split in splits:
        split_dir = os.path.join("data", "processed", split)
        if not os.path.exists(split_dir):
            continue
        for cls_name in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                fpath = os.path.join(cls_path, fname)
                total_checked += 1
                try:
                    # Integrity & PIL open
                    with Image.open(fpath) as img:
                        img.verify()
                    with Image.open(fpath) as img:
                        w, h = img.size
                        mode = img.mode
                        dimension_stats[f"{w}x{h}"] += 1
                        channel_stats[mode] += 1
                    
                    # Compute hash
                    fhash = compute_sha256(fpath)
                    split_hashes[split][fhash] = fpath
                    all_hashes[fhash].append((split, fpath))
                except Exception as e:
                    corrupted_files.append((fpath, str(e)))

    print(f"Total files audited: {total_checked}")
    print(f"Corrupted or unreadable files: {len(corrupted_files)}")
    if corrupted_files:
        for cf, err in corrupted_files[:10]:
            print(f"   [CORRUPT] {cf}: {err}")

    # Check for Cross-Split Leakage
    train_hashes = set(split_hashes["train"].keys())
    val_hashes = set(split_hashes["val"].keys())
    test_hashes = set(split_hashes["test"].keys())

    train_val_leak = train_hashes.intersection(val_hashes)
    train_test_leak = train_hashes.intersection(test_hashes)
    val_test_leak = val_hashes.intersection(test_hashes)

    print("\nCross-Split Data Leakage Check (Exact SHA-256 Collisions):")
    print(f"   Train <-> Val Overlaps:  {len(train_val_leak)}")
    print(f"   Train <-> Test Overlaps: {len(train_test_leak)}")
    print(f"   Val <-> Test Overlaps:   {len(val_test_leak)}")

    if train_test_leak:
        print("   ⚠️ WARNING: Exact duplicate files detected between train and test sets!")
        for h in list(train_test_leak)[:5]:
            print(f"      Train: {split_hashes['train'][h]}")
            print(f"      Test:  {split_hashes['test'][h]}")
    else:
        print("   ✅ PASSED: 0% Data Leakage between Train and Test sets! Complete train/test isolation verified.")

    # Check for Intra-Split Duplicates (same image twice within train)
    intra_dups = {h: paths for h, paths in all_hashes.items() if len(paths) > 1}
    print(f"\nTotal Hash Collision Groups (including intra-split duplicates): {len(intra_dups)}")

    print("\nChannel Distribution:")
    for mode, count in channel_stats.items():
        print(f"   {mode}: {count} ({count/total_checked:.1%})")

    print(f"\nTop Dimensions (Total unique sizes: {len(dimension_stats)}):")
    for dim, count in sorted(dimension_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {dim}: {count} ({count/total_checked:.1%})")

if __name__ == "__main__":
    audit_quality_and_leakage()
