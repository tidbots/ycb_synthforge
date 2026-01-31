#!/usr/bin/env python3
"""
Merge YOLO datasets for incremental learning.
Handles class ID remapping when combining datasets with different class definitions.

Usage:
    # Merge base YCB dataset with new object dataset
    python merge_for_incremental.py \
        --base /path/to/ycb_subset \
        --new /path/to/new_objects \
        --output /path/to/merged
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def load_dataset_config(dataset_dir: Path) -> dict:
    """Load dataset.yaml configuration."""
    yaml_path = dataset_dir / 'dataset.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found in {dataset_dir}")

    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def create_class_mapping(
    base_names: Dict[int, str],
    new_names: Dict[int, str]
) -> tuple:
    """
    Create class mapping for merging datasets.

    Returns:
        - merged_names: Combined class names
        - new_to_merged: Mapping from new dataset class IDs to merged IDs
    """
    merged_names = dict(base_names)  # Start with base classes
    new_to_merged = {}

    next_id = max(base_names.keys()) + 1 if base_names else 0

    for new_id, name in new_names.items():
        # Check if class already exists in base
        existing_id = None
        for base_id, base_name in base_names.items():
            if base_name.lower() == name.lower():
                existing_id = base_id
                break

        if existing_id is not None:
            # Map to existing class
            new_to_merged[new_id] = existing_id
            print(f"  Class '{name}' already exists (ID {new_id} -> {existing_id})")
        else:
            # Add as new class
            merged_names[next_id] = name
            new_to_merged[new_id] = next_id
            print(f"  New class '{name}' added (ID {new_id} -> {next_id})")
            next_id += 1

    return merged_names, new_to_merged


def remap_labels(
    source_label: Path,
    dest_label: Path,
    class_mapping: Dict[int, int]
):
    """Remap class IDs in a YOLO label file."""
    lines = []

    with open(source_label, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                old_class = int(parts[0])
                new_class = class_mapping.get(old_class, old_class)
                parts[0] = str(new_class)
                lines.append(' '.join(parts))

    with open(dest_label, 'w') as f:
        f.write('\n'.join(lines))
        if lines:
            f.write('\n')


def copy_dataset_split(
    source_dir: Path,
    dest_dir: Path,
    split: str,
    prefix: str,
    class_mapping: Optional[Dict[int, int]] = None
) -> int:
    """
    Copy a dataset split with optional class remapping.

    Returns:
        Number of images copied
    """
    source_images = source_dir / 'images' / split
    source_labels = source_dir / 'labels' / split

    if not source_images.exists():
        return 0

    dest_images = dest_dir / 'images' / split
    dest_labels = dest_dir / 'labels' / split

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    image_extensions = ['*.png', '*.jpg', '*.jpeg']

    for ext in image_extensions:
        for img_path in source_images.glob(ext):
            # Create unique filename with prefix
            new_name = f"{prefix}_{img_path.name}"
            dest_img = dest_images / new_name

            # Copy image
            shutil.copy2(img_path, dest_img)

            # Copy/remap label
            label_path = source_labels / f"{img_path.stem}.txt"
            dest_label = dest_labels / f"{prefix}_{img_path.stem}.txt"

            if label_path.exists():
                if class_mapping:
                    remap_labels(label_path, dest_label, class_mapping)
                else:
                    shutil.copy2(label_path, dest_label)
            else:
                # Create empty label file
                dest_label.touch()

            count += 1

    return count


def merge_datasets(
    base_dir: Path,
    new_dirs: List[Path],
    output_dir: Path,
    base_prefix: str = 'base',
    new_prefixes: Optional[List[str]] = None
):
    """
    Merge multiple YOLO datasets.

    Args:
        base_dir: Base dataset (class IDs preserved)
        new_dirs: List of new datasets to merge
        output_dir: Output directory for merged dataset
        base_prefix: Prefix for base dataset files
        new_prefixes: Prefixes for new dataset files
    """
    if new_prefixes is None:
        new_prefixes = [f'new{i}' for i in range(len(new_dirs))]

    # Load base configuration
    base_config = load_dataset_config(base_dir)
    base_names = base_config.get('names', {})
    if isinstance(base_names, list):
        base_names = {i: name for i, name in enumerate(base_names)}

    print(f"Base dataset: {len(base_names)} classes")

    # Process new datasets and build merged class names
    merged_names = dict(base_names)
    all_mappings = []

    for i, new_dir in enumerate(new_dirs):
        print(f"\nProcessing new dataset {i+1}: {new_dir}")
        new_config = load_dataset_config(new_dir)
        new_names = new_config.get('names', {})
        if isinstance(new_names, list):
            new_names = {j: name for j, name in enumerate(new_names)}

        print(f"  Original classes: {len(new_names)}")
        merged_names, mapping = create_class_mapping(merged_names, new_names)
        all_mappings.append(mapping)

    print(f"\nMerged dataset: {len(merged_names)} total classes")
    print(f"  New classes added: {len(merged_names) - len(base_names)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy base dataset (no remapping needed)
    print(f"\nCopying base dataset...")
    for split in ['train', 'val', 'test']:
        count = copy_dataset_split(base_dir, output_dir, split, base_prefix, None)
        if count > 0:
            print(f"  {split}: {count} images")

    # Copy and remap new datasets
    for i, (new_dir, mapping, prefix) in enumerate(
        zip(new_dirs, all_mappings, new_prefixes)
    ):
        print(f"\nCopying new dataset {i+1}...")
        for split in ['train', 'val', 'test']:
            count = copy_dataset_split(new_dir, output_dir, split, prefix, mapping)
            if count > 0:
                print(f"  {split}: {count} images")

    # Create merged dataset.yaml
    merged_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(merged_names),
        'names': merged_names,
    }

    if (output_dir / 'images' / 'test').exists():
        merged_config['test'] = 'images/test'

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, allow_unicode=True)

    print(f"\nCreated {yaml_path}")

    return merged_names


def main():
    parser = argparse.ArgumentParser(
        description='Merge YOLO datasets for incremental learning'
    )
    parser.add_argument(
        '--base', type=str, required=True,
        help='Base dataset directory (class IDs preserved)'
    )
    parser.add_argument(
        '--new', type=str, nargs='+', required=True,
        help='New dataset directories to merge'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory for merged dataset'
    )
    parser.add_argument(
        '--base_prefix', type=str, default='base',
        help='Prefix for base dataset files (default: base)'
    )
    parser.add_argument(
        '--new_prefixes', type=str, nargs='+',
        help='Prefixes for new dataset files'
    )

    args = parser.parse_args()

    base_dir = Path(args.base)
    new_dirs = [Path(d) for d in args.new]
    output_dir = Path(args.output)

    print("=" * 60)
    print("Merging YOLO Datasets for Incremental Learning")
    print("=" * 60)
    print(f"Base: {base_dir}")
    for i, d in enumerate(new_dirs):
        print(f"New {i+1}: {d}")
    print(f"Output: {output_dir}")

    merge_datasets(
        base_dir, new_dirs, output_dir,
        args.base_prefix, args.new_prefixes
    )

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
