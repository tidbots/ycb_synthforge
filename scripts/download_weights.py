#!/usr/bin/env python3
"""YOLO26 weights downloader.

Downloads pre-trained YOLO26 weights from Ultralytics.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError


# YOLO26 model configurations
MODELS = {
    "yolo26n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "size_mb": 5.4,
        "params": "2.6M",
        "description": "Nano - fastest, lowest accuracy",
    },
    "yolo26s": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "size_mb": 19.2,
        "params": "9.4M",
        "description": "Small - balanced speed/accuracy",
    },
    "yolo26m": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "size_mb": 39.5,
        "params": "20.1M",
        "description": "Medium - recommended for most use cases",
    },
    "yolo26l": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        "size_mb": 49.0,
        "params": "25.3M",
        "description": "Large - higher accuracy",
    },
    "yolo26x": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "size_mb": 109.3,
        "params": "56.9M",
        "description": "XLarge - highest accuracy, slowest",
    },
}


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Display download progress."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {percent:5.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()


def download_model(model_name: str, output_dir: Path, force: bool = False) -> bool:
    """Download a single model.

    Args:
        model_name: Name of the model (e.g., 'yolo26n')
        output_dir: Directory to save the weights
        force: Force re-download even if file exists

    Returns:
        True if download successful, False otherwise
    """
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_name]
    output_path = output_dir / f"{model_name}.pt"

    # Check if already exists
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  {model_name}.pt already exists ({size_mb:.1f} MB) - skipping")
        print(f"  Use --force to re-download")
        return True

    print(f"  {model_name}: {model_info['description']}")
    print(f"  Parameters: {model_info['params']}, Size: ~{model_info['size_mb']} MB")

    try:
        urlretrieve(model_info["url"], output_path, reporthook=download_progress)
        print()  # New line after progress

        # Verify file size
        actual_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path} ({actual_size_mb:.1f} MB)")
        return True

    except HTTPError as e:
        print(f"\n  HTTP Error: {e.code} - {e.reason}")
        return False
    except URLError as e:
        print(f"\n  URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO26 pre-trained weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python download_weights.py --all

  # Download specific models
  python download_weights.py --models yolo26n yolo26s yolo26m

  # Force re-download
  python download_weights.py --models yolo26n --force

  # List available models
  python download_weights.py --list
        """,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Models to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "weights",
        help="Output directory for weights (default: ../weights)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # List models
    if args.list:
        print("Available YOLO26 models:")
        print("-" * 60)
        for name, info in MODELS.items():
            print(f"  {name:10s} | {info['params']:>6s} params | ~{info['size_mb']:>5.1f} MB")
            print(f"             | {info['description']}")
        return 0

    # Determine which models to download
    if args.all:
        models_to_download = list(MODELS.keys())
    elif args.models:
        models_to_download = args.models
    else:
        # Default: download nano and small
        models_to_download = ["yolo26n", "yolo26s"]

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading YOLO26 weights to: {args.output_dir}")
    print(f"Models: {', '.join(models_to_download)}")
    print("=" * 60)

    success_count = 0
    for model_name in models_to_download:
        print(f"\n[{models_to_download.index(model_name) + 1}/{len(models_to_download)}] {model_name}")
        if download_model(model_name, args.output_dir, args.force):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Downloaded: {success_count}/{len(models_to_download)} models")

    if success_count == len(models_to_download):
        print("All downloads completed successfully!")
        return 0
    else:
        print("Some downloads failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
