"""
Extract ~2000 images (by complete IDs) from train_sam3_Lizard
and their corresponding pattern images.

Usage:
    python reid_scripts/extract_subset.py [--target 2000] [--seed 42] [--dry-run]
"""

import os
import shutil
import random
import argparse
from collections import defaultdict

# ====== CONFIG ======
SRC_TRAIN = "/data/yil708/Code-Skink/sam3/dataset/train_sam3_Lizard"
SRC_PATTERN = "/data/yil708/Code-Skink/sam3/dataset/train_sam3_Lizard_pattern"
DST_TRAIN = "/data/yil708/Code-Skink/sam3/dataset/reid_skink_data/try/train"
DST_PATTERN = "/data/yil708/Code-Skink/sam3/dataset/reid_skink_data/try/pattern"
# ====================


def parse_id(filename):
    """Extract ID from filename format: {ID}_{-1}_{index}_{originalname}.jpg"""
    return filename.split("_")[0]


def main():
    parser = argparse.ArgumentParser(description="Extract ~2000 images by complete IDs")
    parser.add_argument("--target", type=int, default=2000, help="Target number of images (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying files")
    args = parser.parse_args()

    random.seed(args.seed)

    # Step 1: Group train images by ID
    id_to_files = defaultdict(list)
    for f in os.listdir(SRC_TRAIN):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if os.path.isdir(os.path.join(SRC_TRAIN, f)):
            continue
        img_id = parse_id(f)
        id_to_files[img_id].append(f)

    all_ids = sorted(id_to_files.keys(), key=lambda x: int(x))
    total_images = sum(len(v) for v in id_to_files.values())
    print(f"Total images: {total_images}")
    print(f"Total unique IDs: {len(all_ids)}")
    print(f"Avg images per ID: {total_images / len(all_ids):.1f}")

    # Step 2: Randomly shuffle IDs and pick until reaching target
    shuffled_ids = list(all_ids)
    random.shuffle(shuffled_ids)

    selected_ids = []
    selected_count = 0
    for img_id in shuffled_ids:
        n = len(id_to_files[img_id])
        if selected_count + n > args.target + 50:
            # Allow a small overshoot (+50) but not too much
            continue
        selected_ids.append(img_id)
        selected_count += n
        if selected_count >= args.target:
            break

    # If we haven't reached the target, add remaining IDs
    if selected_count < args.target:
        for img_id in shuffled_ids:
            if img_id in selected_ids:
                continue
            selected_ids.append(img_id)
            selected_count += len(id_to_files[img_id])
            if selected_count >= args.target:
                break

    selected_ids.sort(key=lambda x: int(x))
    print(f"\nSelected {len(selected_ids)} IDs with {selected_count} images (target: {args.target})")

    if args.dry_run:
        print("\n[DRY RUN] No files copied.")
        # Show some sample IDs
        for img_id in selected_ids[:10]:
            print(f"  ID {img_id}: {len(id_to_files[img_id])} images")
        if len(selected_ids) > 10:
            print(f"  ... and {len(selected_ids) - 10} more IDs")
        return

    # Step 3: Copy train images
    os.makedirs(DST_TRAIN, exist_ok=True)
    os.makedirs(DST_PATTERN, exist_ok=True)

    train_copied = 0
    pattern_copied = 0
    pattern_missing = 0

    for img_id in selected_ids:
        for f in id_to_files[img_id]:
            # Copy train image
            src = os.path.join(SRC_TRAIN, f)
            dst = os.path.join(DST_TRAIN, f)
            shutil.copy2(src, dst)
            train_copied += 1

            # Copy corresponding pattern images (pattern_{1..N})
            fn, ext = os.path.splitext(f)
            for i in range(1, 20):  # Check up to 20 patterns per image
                pattern_name = f"{fn}_pattern_{i}{ext}"
                pattern_src = os.path.join(SRC_PATTERN, pattern_name)
                if os.path.exists(pattern_src):
                    pattern_dst = os.path.join(DST_PATTERN, pattern_name)
                    shutil.copy2(pattern_src, pattern_dst)
                    pattern_copied += 1
                else:
                    if i == 1:
                        pattern_missing += 1
                    break  # No more patterns for this image

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Train images copied:   {train_copied}")
    print(f"Pattern images copied: {pattern_copied}")
    print(f"Images with no pattern: {pattern_missing}")
    print(f"Selected IDs: {len(selected_ids)}")
    print(f"{'='*60}")
    print(f"\nTrain output:   {DST_TRAIN}")
    print(f"Pattern output: {DST_PATTERN}")


if __name__ == "__main__":
    main()
