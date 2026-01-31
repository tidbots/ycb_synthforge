#!/usr/bin/env python3
"""
Create a balanced subset from YOLO format dataset.
Samples images ensuring class balance for incremental learning.
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def parse_yolo_label(label_path: Path) -> list:
    """Parse YOLO format label file and return class IDs."""
    classes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.append(int(parts[0]))
    return classes


def create_balanced_subset(
    source_dir: Path,
    output_dir: Path,
    num_samples: int,
    split: str = 'train',
    seed: int = 42
):
    """
    Create a balanced subset from YOLO dataset.

    Args:
        source_dir: Source YOLO dataset directory
        output_dir: Output directory for subset
        num_samples: Target number of samples
        split: Dataset split (train/val)
        seed: Random seed
    """
    random.seed(seed)

    source_images = source_dir / 'images' / split
    source_labels = source_dir / 'labels' / split

    if not source_images.exists():
        print(f"Error: {source_images} not found")
        return

    # Build class-to-images mapping
    class_to_images = defaultdict(list)
    all_images = list(source_images.glob('*.png')) + list(source_images.glob('*.jpg'))

    print(f"Scanning {len(all_images)} images...")

    for img_path in all_images:
        label_path = source_labels / f"{img_path.stem}.txt"
        classes = parse_yolo_label(label_path)

        for cls_id in set(classes):  # Unique classes in this image
            class_to_images[cls_id].append(img_path)

    num_classes = len(class_to_images)
    print(f"Found {num_classes} classes")

    # Calculate samples per class
    samples_per_class = max(1, num_samples // num_classes)
    print(f"Target: {samples_per_class} samples per class")

    # Select images (prioritize underrepresented classes)
    selected_images = set()

    # Sort classes by frequency (least frequent first)
    sorted_classes = sorted(class_to_images.keys(),
                           key=lambda c: len(class_to_images[c]))

    for cls_id in sorted_classes:
        images = class_to_images[cls_id]
        available = [img for img in images if img not in selected_images]

        # Sample from available images
        to_select = min(samples_per_class, len(available))
        if available:
            selected = random.sample(available, to_select)
            selected_images.update(selected)

    # If we need more samples, add randomly
    if len(selected_images) < num_samples:
        remaining = [img for img in all_images if img not in selected_images]
        additional = min(num_samples - len(selected_images), len(remaining))
        selected_images.update(random.sample(remaining, additional))

    print(f"Selected {len(selected_images)} images")

    # Create output directories
    out_images = output_dir / 'images' / split
    out_labels = output_dir / 'labels' / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img_path in selected_images:
        # Copy image
        shutil.copy2(img_path, out_images / img_path.name)

        # Copy label
        label_path = source_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, out_labels / label_path.name)

    print(f"Copied to {output_dir}")

    # Print class distribution
    print("\nClass distribution in subset:")
    subset_class_counts = defaultdict(int)
    for img_path in selected_images:
        label_path = source_labels / f"{img_path.stem}.txt"
        classes = parse_yolo_label(label_path)
        for cls_id in classes:
            subset_class_counts[cls_id] += 1

    for cls_id in sorted(subset_class_counts.keys())[:10]:
        print(f"  Class {cls_id}: {subset_class_counts[cls_id]} annotations")
    if len(subset_class_counts) > 10:
        print(f"  ... and {len(subset_class_counts) - 10} more classes")


def copy_dataset_yaml(source_dir: Path, output_dir: Path):
    """Copy and update dataset.yaml."""
    source_yaml = source_dir / 'dataset.yaml'
    if source_yaml.exists():
        with open(source_yaml, 'r') as f:
            config = yaml.safe_load(f)

        # Update paths
        config['path'] = str(output_dir.absolute())

        output_yaml = output_dir / 'dataset.yaml'
        with open(output_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"Created {output_yaml}")


def main():
    parser = argparse.ArgumentParser(
        description='Create balanced subset from YOLO dataset'
    )
    parser.add_argument(
        '--source', type=str, required=True,
        help='Source YOLO dataset directory'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory for subset'
    )
    parser.add_argument(
        '--num_samples', type=int, default=5000,
        help='Number of samples to select (default: 5000)'
    )
    parser.add_argument(
        '--val_samples', type=int, default=500,
        help='Number of validation samples (default: 500)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    print("=" * 60)
    print("Creating balanced subset")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Train samples: {args.num_samples}")
    print(f"Val samples: {args.val_samples}")
    print()

    # Create train subset
    print("Processing train split...")
    create_balanced_subset(
        source_dir, output_dir,
        args.num_samples, 'train', args.seed
    )

    # Create val subset
    print("\nProcessing val split...")
    create_balanced_subset(
        source_dir, output_dir,
        args.val_samples, 'val', args.seed
    )

    # Copy dataset.yaml
    copy_dataset_yaml(source_dir, output_dir)

    print("\n" + "=" * 60)
    print("Subset creation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
