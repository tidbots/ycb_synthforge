#!/usr/bin/env python3
"""YCB Object Model downloader.

Downloads 3D models from the YCB Object and Model Set.
http://ycb-benchmarks.com/
"""

import argparse
import os
import sys
import tarfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

# YCB dataset base URL
YCB_BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/berkeley"

# All 103 YCB objects
YCB_OBJECTS = [
    # Food items (001-010)
    "001_chips_can",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    # Fruits (011-018)
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    # Kitchen items (019-033)
    "019_pitcher_base",
    "021_bleach_cleanser",
    "022_windex_bottle",
    "023_wine_glass",
    "024_bowl",
    "025_mug",
    "026_sponge",
    "027-skillet",
    "028_skillet_lid",
    "029_plate",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    # Tools (035-052)
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "038_padlock",
    "039_key",
    "040_large_marker",
    "041_small_marker",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "046_plastic_bolt",
    "047_plastic_nut",
    "048_hammer",
    "049_small_clamp",
    "050_medium_clamp",
    "051_large_clamp",
    "052_extra_large_clamp",
    # Sports (053-058)
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    # Misc (059-077)
    "059_chain",
    "061_foam_brick",
    "062_dice",
    "063-a_marbles",
    "063-b_marbles",
    "063-c_marbles",
    "063-d_marbles",
    "063-e_marbles",
    "063-f_marbles",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "065-e_cups",
    "065-f_cups",
    "065-g_cups",
    "065-h_cups",
    "065-i_cups",
    "065-j_cups",
    "070-a_colored_wood_blocks",
    "070-b_colored_wood_blocks",
    "071_nine_hole_peg_test",
    "072-a_toy_airplane",
    "072-b_toy_airplane",
    "072-c_toy_airplane",
    "072-d_toy_airplane",
    "072-e_toy_airplane",
    "072-f_toy_airplane",
    "072-g_toy_airplane",
    "072-h_toy_airplane",
    "072-i_toy_airplane",
    "072-j_toy_airplane",
    "072-k_toy_airplane",
    "073-a_lego_duplo",
    "073-b_lego_duplo",
    "073-c_lego_duplo",
    "073-d_lego_duplo",
    "073-e_lego_duplo",
    "073-f_lego_duplo",
    "073-g_lego_duplo",
    "073-h_lego_duplo",
    "073-i_lego_duplo",
    "073-j_lego_duplo",
    "073-k_lego_duplo",
    "073-l_lego_duplo",
    "073-m_lego_duplo",
    "076_timer",
    "077_rubiks_cube",
]

# Object categories for filtering
CATEGORIES = {
    "food": [
        "001_chips_can", "002_master_chef_can", "003_cracker_box",
        "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle",
        "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
        "010_potted_meat_can",
    ],
    "fruit": [
        "011_banana", "012_strawberry", "013_apple", "014_lemon",
        "015_peach", "016_pear", "017_orange", "018_plum",
    ],
    "kitchen": [
        "019_pitcher_base", "021_bleach_cleanser", "022_windex_bottle",
        "023_wine_glass", "024_bowl", "025_mug", "026_sponge",
        "027-skillet", "028_skillet_lid", "029_plate", "030_fork",
        "031_spoon", "032_knife", "033_spatula",
    ],
    "tool": [
        "035_power_drill", "036_wood_block", "037_scissors", "038_padlock",
        "039_key", "040_large_marker", "041_small_marker",
        "042_adjustable_wrench", "043_phillips_screwdriver",
        "044_flat_screwdriver", "046_plastic_bolt", "047_plastic_nut",
        "048_hammer", "049_small_clamp", "050_medium_clamp",
        "051_large_clamp", "052_extra_large_clamp",
    ],
    "sport": [
        "053_mini_soccer_ball", "054_softball", "055_baseball",
        "056_tennis_ball", "057_racquetball", "058_golf_ball",
    ],
    "toy": [
        "062_dice", "063-a_marbles", "063-b_marbles", "063-c_marbles",
        "063-d_marbles", "063-e_marbles", "063-f_marbles",
        "070-a_colored_wood_blocks", "070-b_colored_wood_blocks",
        "071_nine_hole_peg_test",
        "072-a_toy_airplane", "072-b_toy_airplane", "072-c_toy_airplane",
        "072-d_toy_airplane", "072-e_toy_airplane", "072-f_toy_airplane",
        "072-g_toy_airplane", "072-h_toy_airplane", "072-i_toy_airplane",
        "072-j_toy_airplane", "072-k_toy_airplane",
        "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo",
        "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo",
        "073-g_lego_duplo", "073-h_lego_duplo", "073-i_lego_duplo",
        "073-j_lego_duplo", "073-k_lego_duplo", "073-l_lego_duplo",
        "073-m_lego_duplo", "077_rubiks_cube",
    ],
    "misc": [
        "059_chain", "061_foam_brick",
        "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups",
        "065-e_cups", "065-f_cups", "065-g_cups", "065-h_cups",
        "065-i_cups", "065-j_cups", "076_timer",
    ],
}


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Display download progress."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    Downloading: {percent:5.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()
    else:
        downloaded = block_num * block_size
        downloaded_mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r    Downloading: {downloaded_mb:.1f} MB")
        sys.stdout.flush()


def download_object(
    obj_name: str,
    output_dir: Path,
    keep_tgz: bool = False,
    force: bool = False,
) -> bool:
    """Download and extract a single YCB object.

    Args:
        obj_name: Name of the object (e.g., '001_chips_can')
        output_dir: Directory to save the model
        keep_tgz: Keep the tgz file after extraction
        force: Force re-download even if exists

    Returns:
        True if successful, False otherwise
    """
    obj_dir = output_dir / obj_name
    tgz_file = output_dir / f"{obj_name}_berkeley_meshes.tgz"
    url = f"{YCB_BASE_URL}/{obj_name}/{obj_name}_berkeley_meshes.tgz"

    # Check if already exists
    if obj_dir.exists() and not force:
        # Verify it has content
        obj_files = list(obj_dir.rglob("*.obj"))
        if obj_files:
            print(f"    Already exists ({len(obj_files)} .obj files) - skipping")
            return True

    print(f"    URL: {url}")

    try:
        # Download
        urlretrieve(url, tgz_file, reporthook=download_progress)
        print()  # New line after progress

        # Extract
        print(f"    Extracting...")
        with tarfile.open(tgz_file, "r:gz") as tar:
            tar.extractall(path=output_dir)

        # Verify extraction
        if obj_dir.exists():
            obj_files = list(obj_dir.rglob("*.obj"))
            print(f"    Extracted: {len(obj_files)} .obj files")
        else:
            print(f"    Warning: Expected directory {obj_dir} not found after extraction")

        # Clean up tgz
        if not keep_tgz and tgz_file.exists():
            tgz_file.unlink()

        return True

    except HTTPError as e:
        print(f"\n    HTTP Error: {e.code} - {e.reason}")
        return False
    except URLError as e:
        print(f"\n    URL Error: {e.reason}")
        return False
    except tarfile.TarError as e:
        print(f"\n    Tar Error: {e}")
        return False
    except Exception as e:
        print(f"\n    Error: {e}")
        return False
    finally:
        # Clean up partial downloads on error
        if tgz_file.exists() and not keep_tgz:
            try:
                tgz_file.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Download YCB Object 3D models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all 103 models
  python download_ycb_models.py --all

  # Download specific objects
  python download_ycb_models.py --objects 001_chips_can 002_master_chef_can

  # Download by category
  python download_ycb_models.py --category food fruit

  # List available objects
  python download_ycb_models.py --list

  # List categories
  python download_ycb_models.py --list-categories

  # Force re-download
  python download_ycb_models.py --all --force

  # Keep tgz files
  python download_ycb_models.py --all --keep-tgz
        """,
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        help="Specific objects to download",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        choices=list(CATEGORIES.keys()),
        help="Download objects by category",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all 103 objects",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "ycb",
        help="Output directory (default: ../models/ycb)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists",
    )
    parser.add_argument(
        "--keep-tgz",
        action="store_true",
        help="Keep tgz files after extraction",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available objects",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List categories and their objects",
    )

    args = parser.parse_args()

    # List objects
    if args.list:
        print(f"Available YCB objects ({len(YCB_OBJECTS)} total):")
        print("-" * 50)
        for i, obj in enumerate(YCB_OBJECTS):
            print(f"  {i+1:3d}. {obj}")
        return 0

    # List categories
    if args.list_categories:
        print("YCB Object Categories:")
        print("=" * 50)
        for cat, objects in CATEGORIES.items():
            print(f"\n{cat.upper()} ({len(objects)} objects):")
            for obj in objects:
                print(f"  - {obj}")
        return 0

    # Determine which objects to download
    objects_to_download = []

    if args.all:
        objects_to_download = YCB_OBJECTS.copy()
    elif args.category:
        for cat in args.category:
            objects_to_download.extend(CATEGORIES[cat])
        # Remove duplicates while preserving order
        seen = set()
        objects_to_download = [x for x in objects_to_download if not (x in seen or seen.add(x))]
    elif args.objects:
        # Validate object names
        for obj in args.objects:
            if obj in YCB_OBJECTS:
                objects_to_download.append(obj)
            else:
                print(f"Warning: Unknown object '{obj}', skipping")
    else:
        parser.print_help()
        print("\nError: Specify --all, --category, or --objects")
        return 1

    if not objects_to_download:
        print("No objects to download")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading YCB models to: {args.output_dir}")
    print(f"Objects: {len(objects_to_download)}")
    print("=" * 60)

    success_count = 0
    failed = []

    for i, obj_name in enumerate(objects_to_download):
        print(f"\n[{i+1}/{len(objects_to_download)}] {obj_name}")
        if download_object(obj_name, args.output_dir, args.keep_tgz, args.force):
            success_count += 1
        else:
            failed.append(obj_name)

    print("\n" + "=" * 60)
    print(f"Downloaded: {success_count}/{len(objects_to_download)} objects")

    if failed:
        print(f"\nFailed objects ({len(failed)}):")
        for obj in failed:
            print(f"  - {obj}")
        return 1
    else:
        print("All downloads completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
