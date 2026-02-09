#!/usr/bin/env python3
"""
Compare two directories and find duplicate/unique image files.
"""

import os
from pathlib import Path


def get_image_filenames(directory):
    """Get all image filenames from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    filenames = set()
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return filenames
    
    for filename in os.listdir(directory):
        if Path(filename).suffix.lower() in image_extensions:
            filenames.add(filename)
    
    return filenames


def compare_directories(dir1, dir2, output_file):
    """Compare two directories and save results."""
    print(f"Comparing directories:")
    print(f"  Directory 1: {dir1}")
    print(f"  Directory 2: {dir2}")
    print()
    
    # Get filenames from both directories
    files1 = get_image_filenames(dir1)
    files2 = get_image_filenames(dir2)
    
    # Find common and unique files
    common_files = files1 & files2  # Intersection
    only_in_dir1 = files1 - files2  # Files only in dir1
    only_in_dir2 = files2 - files1  # Files only in dir2
    
    # Print summary
    print(f"Summary:")
    print(f"  Total in {os.path.basename(dir1)}: {len(files1)}")
    print(f"  Total in {os.path.basename(dir2)}: {len(files2)}")
    print(f"  Common (duplicates): {len(common_files)}")
    print(f"  Only in {os.path.basename(dir1)}: {len(only_in_dir1)}")
    print(f"  Only in {os.path.basename(dir2)}: {len(only_in_dir2)}")
    print()
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DIRECTORY COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Directory 1: {dir1}\n")
        f.write(f"Directory 2: {dir2}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total images in train_sam3_Animal: {len(files1)}\n")
        f.write(f"Total images in train_sam3_Lizard: {len(files2)}\n")
        f.write(f"Common images (duplicates): {len(common_files)}\n")
        f.write(f"Unique to train_sam3_Animal: {len(only_in_dir1)}\n")
        f.write(f"Unique to train_sam3_Lizard: {len(only_in_dir2)}\n\n")
        
        # Write common files
        f.write("="*80 + "\n")
        f.write(f"COMMON IMAGES ({len(common_files)} files)\n")
        f.write("="*80 + "\n")
        if common_files:
            for filename in sorted(common_files):
                f.write(f"{filename}\n")
        else:
            f.write("(No common files)\n")
        f.write("\n")
        
        # Write files only in dir1
        f.write("="*80 + "\n")
        f.write(f"ONLY IN train_sam3_Animal ({len(only_in_dir1)} files)\n")
        f.write("="*80 + "\n")
        if only_in_dir1:
            for filename in sorted(only_in_dir1):
                f.write(f"{filename}\n")
        else:
            f.write("(No unique files)\n")
        f.write("\n")
        
        # Write files only in dir2
        f.write("="*80 + "\n")
        f.write(f"ONLY IN train_sam3_Lizard ({len(only_in_dir2)} files)\n")
        f.write("="*80 + "\n")
        if only_in_dir2:
            for filename in sorted(only_in_dir2):
                f.write(f"{filename}\n")
        else:
            f.write("(No unique files)\n")
    
    print(f"Results saved to: {output_file}")
    return {
        'total_dir1': len(files1),
        'total_dir2': len(files2),
        'common': len(common_files),
        'only_dir1': len(only_in_dir1),
        'only_dir2': len(only_in_dir2)
    }


if __name__ == '__main__':
    dir1 = "/data/yil708/Code-Skink/sam3/dataset/train_sam3_Animal"
    dir2 = "/data/yil708/Code-Skink/sam3/dataset/train_sam3_Lizard"
    output_file = "/data/yil708/Code-Skink/sam3/dataset/folder_comparison_results.txt"
    
    compare_directories(dir1, dir2, output_file)
