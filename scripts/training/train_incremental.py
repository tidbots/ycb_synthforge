#!/usr/bin/env python3
"""
Incremental learning training script for YOLO.
Supports backbone freezing and replay-based training.

Usage:
    # Basic incremental training with frozen backbone
    python train_incremental.py \
        --weights /path/to/ycb_trained.pt \
        --data /path/to/merged_dataset/dataset.yaml \
        --freeze 10

    # Full fine-tuning (unfrozen)
    python train_incremental.py \
        --weights /path/to/ycb_trained.pt \
        --data /path/to/merged_dataset/dataset.yaml \
        --freeze 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_incremental(
    weights: str,
    data: str,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    freeze: int = 10,
    project: str = 'outputs/incremental',
    name: str = 'run',
    device: str = '0',
    workers: int = 8,
    patience: int = 20,
    lr0: float = 0.001,  # Lower learning rate for fine-tuning
    resume: bool = False,
):
    """
    Train YOLO model incrementally with optional backbone freezing.

    Args:
        weights: Path to pre-trained weights
        data: Path to dataset.yaml
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        freeze: Number of layers to freeze (0 = no freezing)
        project: Project directory
        name: Run name
        device: CUDA device
        workers: Number of data loader workers
        patience: Early stopping patience
        lr0: Initial learning rate (lower for fine-tuning)
        resume: Resume from last checkpoint
    """
    print("=" * 60)
    print("Incremental Learning Training")
    print("=" * 60)
    print(f"Weights: {weights}")
    print(f"Dataset: {data}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Frozen layers: {freeze}")
    print(f"Learning rate: {lr0}")
    print()

    # Load model
    model = YOLO(weights)

    # Print model info
    print(f"Model classes: {len(model.names)}")
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # Train with incremental settings
    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        freeze=freeze,  # Freeze first N layers
        project=project,
        name=name,
        device=device,
        workers=workers,
        patience=patience,
        lr0=lr0,
        lrf=0.01,  # Final learning rate = lr0 * lrf
        warmup_epochs=3,
        cos_lr=True,  # Cosine learning rate scheduler
        resume=resume,
        # Augmentation settings (moderate for fine-tuning)
        mosaic=0.5,  # Reduced mosaic
        mixup=0.0,   # No mixup
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Print results summary
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Incremental learning training for YOLO'
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to pre-trained weights'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to dataset.yaml'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of epochs (default: 50)'
    )
    parser.add_argument(
        '--batch', type=int, default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help='Image size (default: 640)'
    )
    parser.add_argument(
        '--freeze', type=int, default=10,
        help='Number of layers to freeze (default: 10, 0=no freezing)'
    )
    parser.add_argument(
        '--project', type=str, default='outputs/incremental',
        help='Project directory'
    )
    parser.add_argument(
        '--name', type=str, default='run',
        help='Run name'
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help='CUDA device (default: 0)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--patience', type=int, default=20,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--lr0', type=float, default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from last checkpoint'
    )

    args = parser.parse_args()

    train_incremental(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()
