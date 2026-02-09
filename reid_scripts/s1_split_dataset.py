#!/usr/bin/env python3
"""
Split Dataset by ID for Re-identification Task
Splits images into train, query, and gallery with proper ID separation.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse


def parse_image_id(filename):
    """
    Extract ID from filename.
    Format: {ID}_{-1}_{index}_{originalname}.jpg
    Example: 1_-1_0_Airport-gs02-02-04-02R.jpg -> ID=1
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 1:
        try:
            return int(parts[0])
        except ValueError:
            print(f"Warning: Could not parse ID from {filename}")
            return None
    return None


def group_images_by_id(dataset_dir):
    """
    Group all images by their ID.
    Returns: dict {ID: [list of image filenames]}
    """
    images_by_id = defaultdict(list)
    
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id = parse_image_id(filename)
            if image_id is not None:
                images_by_id[image_id].append(filename)
    
    return images_by_id


def split_dataset(dataset_dir, output_base_dir, train_ratio=0.6, query_ratio=0.15, 
                  gallery_ratio=0.25, seed=42, dry_run=False):
    """
    Split dataset into train, query, and gallery sets.
    
    Rules:
    1. Train IDs are completely separate from query+gallery IDs
    2. Single-image IDs go to train
    3. Query and gallery can share IDs but not images
    4. Target ratio: 60% train, 15% query, 25% gallery
    """
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    
    # Group images by ID
    print("Grouping images by ID...")
    images_by_id = group_images_by_id(dataset_dir)
    
    total_images = sum(len(images) for images in images_by_id.values())
    total_ids = len(images_by_id)
    
    print(f"Total images: {total_images}")
    print(f"Total unique IDs: {total_ids}")
    
    # Separate single-image and multi-image IDs
    single_image_ids = []
    multi_image_ids = []
    
    for img_id, images in images_by_id.items():
        if len(images) == 1:
            single_image_ids.append(img_id)
        else:
            multi_image_ids.append(img_id)
    
    print(f"Single-image IDs: {len(single_image_ids)} ({sum(len(images_by_id[i]) for i in single_image_ids)} images)")
    print(f"Multi-image IDs: {len(multi_image_ids)} ({sum(len(images_by_id[i]) for i in multi_image_ids)} images)")
    
    # Calculate target counts
    target_train = int(total_images * train_ratio)
    target_query = int(total_images * query_ratio)
    target_gallery = int(total_images * gallery_ratio)
    
    print(f"\nTarget distribution:")
    print(f"  Train: {target_train} ({train_ratio*100}%)")
    print(f"  Query: {target_query} ({query_ratio*100}%)")
    print(f"  Gallery: {target_gallery} ({gallery_ratio*100}%)")
    
    # Step 1: All single-image IDs go to train
    train_ids = set(single_image_ids)
    train_image_count = sum(len(images_by_id[i]) for i in train_ids)
    
    print(f"\nStep 1: Assigned {len(single_image_ids)} single-image IDs to train ({train_image_count} images)")
    
    # Step 2: Split multi-image IDs between train and query+gallery
    random.shuffle(multi_image_ids)
    
    query_gallery_ids = set()
    
    for img_id in multi_image_ids:
        img_count = len(images_by_id[img_id])
        
        # If adding to train would get us closer to target, add to train
        # Otherwise, add to query+gallery pool
        if train_image_count + img_count <= target_train:
            train_ids.add(img_id)
            train_image_count += img_count
        else:
            query_gallery_ids.add(img_id)
    
    # If we haven't reached train target, add more multi-image IDs
    remaining_multi = [i for i in multi_image_ids if i not in train_ids and i not in query_gallery_ids]
    for img_id in remaining_multi:
        img_count = len(images_by_id[img_id])
        if train_image_count < target_train:
            train_ids.add(img_id)
            train_image_count += img_count
        else:
            query_gallery_ids.add(img_id)
    
    print(f"Step 2: Total train IDs: {len(train_ids)} ({train_image_count} images)")
    print(f"        Query+Gallery IDs: {len(query_gallery_ids)} ({sum(len(images_by_id[i]) for i in query_gallery_ids)} images)")
    
    # Step 3: Split query+gallery images between query and gallery
    # Aim for query:gallery ratio within the query+gallery pool
    query_images = []
    gallery_images = []
    
    qg_total = target_query + target_gallery
    qg_query_ratio = target_query / qg_total if qg_total > 0 else 0.375  # 15/(15+25)
    
    for img_id in query_gallery_ids:
        images = images_by_id[img_id]
        random.shuffle(images)
        
        # Split this ID's images between query and gallery
        num_for_query = max(1, int(len(images) * qg_query_ratio))
        
        query_images.extend(images[:num_for_query])
        gallery_images.extend(images[num_for_query:])
    
    # Collect train images
    train_images = []
    for img_id in train_ids:
        train_images.extend(images_by_id[img_id])
    
    print(f"\nFinal distribution:")
    print(f"  Train: {len(train_images)} images from {len(train_ids)} IDs")
    print(f"  Query: {len(query_images)} images from {len(query_gallery_ids)} IDs")
    print(f"  Gallery: {len(gallery_images)} images from {len(query_gallery_ids)} IDs")
    
    # Verify no overlap
    train_id_set = set(train_ids)
    qg_id_set = set(query_gallery_ids)
    overlap = train_id_set & qg_id_set
    
    if overlap:
        print(f"\n⚠️  WARNING: {len(overlap)} IDs overlap between train and query+gallery!")
    else:
        print(f"\n✅ Verified: No ID overlap between train and query+gallery")
    
    # Copy files
    if not dry_run:
        print(f"\nCopying files to output directories...")
        
        train_dir = os.path.join(output_base_dir, 'train')
        query_dir = os.path.join(output_base_dir, 'query')
        gallery_dir = os.path.join(output_base_dir, 'gallery')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(gallery_dir, exist_ok=True)
        
        # Copy train images
        print(f"  Copying {len(train_images)} images to train/")
        for i, filename in enumerate(train_images, 1):
            src = os.path.join(dataset_dir, filename)
            dst = os.path.join(train_dir, filename)
            shutil.copy2(src, dst)
            if i % 1000 == 0:
                print(f"    Progress: {i}/{len(train_images)}")
        
        # Copy query images
        print(f"  Copying {len(query_images)} images to query/")
        for i, filename in enumerate(query_images, 1):
            src = os.path.join(dataset_dir, filename)
            dst = os.path.join(query_dir, filename)
            shutil.copy2(src, dst)
            if i % 1000 == 0:
                print(f"    Progress: {i}/{len(query_images)}")
        
        # Copy gallery images
        print(f"  Copying {len(gallery_images)} images to gallery/")
        for i, filename in enumerate(gallery_images, 1):
            src = os.path.join(dataset_dir, filename)
            dst = os.path.join(gallery_dir, filename)
            shutil.copy2(src, dst)
            if i % 1000 == 0:
                print(f"    Progress: {i}/{len(gallery_images)}")
        
        print(f"\n✅ Dataset split complete!")
        print(f"   Train: {train_dir}")
        print(f"   Query: {query_dir}")
        print(f"   Gallery: {gallery_dir}")
    else:
        print(f"\n[DRY RUN] No files copied. Run without --dry-run to execute.")
    
    # Return statistics
    return {
        'train': {'images': len(train_images), 'ids': len(train_ids)},
        'query': {'images': len(query_images), 'ids': len(query_gallery_ids)},
        'gallery': {'images': len(gallery_images), 'ids': len(query_gallery_ids)},
        'total': {'images': total_images, 'ids': total_ids}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset by ID for re-identification task'
    )
    parser.add_argument(
        '--dataset-dir',
        default='/data/yil708/Code-Skink/sam3/dataset',
        help='Source dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        default='/data/yil708/Code-Skink/sam3',
        help='Output base directory (will create train/query/gallery subdirs)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.6,
        help='Train ratio (default: 0.6)'
    )
    parser.add_argument(
        '--query-ratio',
        type=float,
        default=0.15,
        help='Query ratio (default: 0.15)'
    )
    parser.add_argument(
        '--gallery-ratio',
        type=float,
        default=0.25,
        help='Gallery ratio (default: 0.25)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview split without copying files'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory {args.dataset_dir} does not exist")
        return
    
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Ratio - Train:{args.train_ratio}, Query:{args.query_ratio}, Gallery:{args.gallery_ratio}")
    print(f"Random seed: {args.seed}")
    print(f"Dry run: {args.dry_run}\n")
    
    stats = split_dataset(
        args.dataset_dir,
        args.output_dir,
        args.train_ratio,
        args.query_ratio,
        args.gallery_ratio,
        args.seed,
        args.dry_run
    )


if __name__ == '__main__':
    main()
