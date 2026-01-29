#!/usr/bin/env python3
"""CC0 Textures downloader from ambientCG.

Downloads PBR textures from ambientCG.com (formerly CC0 Textures).
All textures are CC0 licensed (public domain).
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# User-Agent header to avoid 403 errors
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# ambientCG API endpoint
API_URL = "https://ambientCG.com/api/v2/full_json"

# Texture categories for domain randomization
CATEGORIES = {
    "floor": [
        "Wood", "WoodFloor", "Planks", "Tiles", "PavingStones",
        "Concrete", "Marble", "Terrazzo", "Carpet",
    ],
    "wall": [
        "Bricks", "Concrete", "PaintedPlaster", "Plaster",
        "Wallpaper", "Facade", "Stucco",
    ],
    "table": [
        "Wood", "Metal", "Plastic", "Marble", "Granite",
    ],
    "metal": [
        "Metal", "MetalPlates", "DiamondPlate", "PaintedMetal",
        "CorrugatedSteel", "Rust", "MetalWalkway",
    ],
    "fabric": [
        "Fabric", "Leather", "Carpet", "Wicker",
    ],
    "natural": [
        "Ground", "Grass", "Rock", "Rocks", "Gravel",
        "Sand", "Snow", "Ice", "Bark", "Leaf", "LeafSet",
    ],
    "industrial": [
        "Asphalt", "Concrete", "CorrugatedSteel", "Rust",
        "DiamondPlate", "MetalPlates",
    ],
}

# Default texture list for YCB SynthForge (curated selection)
DEFAULT_TEXTURES = [
    # Wood floors and planks
    "WoodFloor001", "WoodFloor002", "WoodFloor003", "WoodFloor004", "WoodFloor005",
    "WoodFloor010", "WoodFloor012", "WoodFloor015", "WoodFloor020", "WoodFloor025",
    "Planks001", "Planks002", "Planks003", "Planks005", "Planks010",
    # Wood surfaces
    "Wood001", "Wood002", "Wood003", "Wood004", "Wood005",
    "Wood010", "Wood015", "Wood020", "Wood025", "Wood030",
    # Tiles
    "Tiles001", "Tiles002", "Tiles003", "Tiles004", "Tiles005",
    "Tiles010", "Tiles015", "Tiles020", "Tiles030", "Tiles040",
    # Concrete
    "Concrete001", "Concrete002", "Concrete003", "Concrete004", "Concrete005",
    "Concrete010", "Concrete015", "Concrete020", "Concrete025", "Concrete030",
    # Metal
    "Metal001", "Metal002", "Metal003", "Metal004", "Metal005",
    "Metal010", "Metal015", "Metal020", "Metal025", "Metal030",
    # Fabric
    "Fabric001", "Fabric002", "Fabric003", "Fabric004", "Fabric005",
    "Fabric010", "Fabric015", "Fabric020", "Fabric025", "Fabric030",
    # Marble
    "Marble001", "Marble002", "Marble003", "Marble004", "Marble005",
    "Marble006", "Marble007", "Marble008", "Marble010", "Marble012",
    # Ground
    "Ground001", "Ground002", "Ground003", "Ground004", "Ground005",
    "Ground010", "Ground015", "Ground020", "Ground025", "Ground030",
    # Bricks
    "Bricks001", "Bricks002", "Bricks003", "Bricks004", "Bricks005",
    "Bricks010", "Bricks015", "Bricks020", "Bricks025", "Bricks030",
    # Leather
    "Leather001", "Leather002", "Leather003", "Leather004", "Leather005",
    "Leather010", "Leather015", "Leather020", "Leather025", "Leather030",
]


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


def fetch_texture_list(include_filter: str = None, limit: int = None) -> list:
    """Fetch available textures from ambientCG API.

    Args:
        include_filter: Filter by asset ID prefix (e.g., "Wood", "Metal")
        limit: Maximum number of textures to return

    Returns:
        List of texture info dicts
    """
    params = {
        "type": "Material",
        "sort": "Popular",
        "limit": limit or 500,
    }
    if include_filter:
        params["id"] = include_filter

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API_URL}?{query}"

    try:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data.get("foundAssets", [])
    except Exception as e:
        print(f"Error fetching texture list: {e}")
        return []


def get_download_url(asset_id: str, resolution: str = "2K") -> str:
    """Get download URL for a texture.

    Args:
        asset_id: Texture asset ID (e.g., "Wood001")
        resolution: Resolution (1K, 2K, 4K, 8K)

    Returns:
        Download URL
    """
    return f"https://ambientCG.com/get?file={asset_id}_{resolution}-JPG.zip"


def download_texture(
    asset_id: str,
    output_dir: Path,
    resolution: str = "2K",
    force: bool = False,
) -> bool:
    """Download and extract a single texture.

    Args:
        asset_id: Texture asset ID (e.g., "Wood001")
        output_dir: Directory to save the texture
        resolution: Resolution (1K, 2K, 4K)
        force: Force re-download even if exists

    Returns:
        True if successful, False otherwise
    """
    texture_dir = output_dir / asset_id
    zip_file = output_dir / f"{asset_id}_{resolution}-JPG.zip"
    url = get_download_url(asset_id, resolution)

    # Check if already exists
    if texture_dir.exists() and not force:
        color_files = list(texture_dir.glob("*_Color.*"))
        if color_files:
            print(f"    Already exists - skipping")
            return True

    print(f"    URL: {url}")

    try:
        # Download with User-Agent header
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(zip_file, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress display
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r    Downloading: {percent:5.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                    else:
                        downloaded_mb = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r    Downloading: {downloaded_mb:.1f} MB")
                    sys.stdout.flush()

        print()  # New line after progress

        # Extract
        print(f"    Extracting...")
        texture_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(texture_dir)

        # Count extracted files
        extracted_files = list(texture_dir.glob("*"))
        print(f"    Extracted: {len(extracted_files)} files")

        # Clean up zip
        zip_file.unlink()

        return True

    except HTTPError as e:
        print(f"\n    HTTP Error: {e.code} - {e.reason}")
        if e.code == 404:
            print(f"    Texture '{asset_id}' not found at {resolution} resolution")
        return False
    except URLError as e:
        print(f"\n    URL Error: {e.reason}")
        return False
    except zipfile.BadZipFile as e:
        print(f"\n    Zip Error: {e}")
        return False
    except Exception as e:
        print(f"\n    Error: {e}")
        return False
    finally:
        # Clean up partial downloads
        if zip_file.exists():
            try:
                zip_file.unlink()
            except Exception:
                pass


def get_textures_by_category(categories: list) -> list:
    """Get texture prefixes for given categories.

    Args:
        categories: List of category names

    Returns:
        List of texture prefixes
    """
    prefixes = []
    for cat in categories:
        if cat in CATEGORIES:
            prefixes.extend(CATEGORIES[cat])
    return list(set(prefixes))


def main():
    parser = argparse.ArgumentParser(
        description="Download CC0 textures from ambientCG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default curated textures (100 textures)
  python download_cctextures.py

  # Download specific textures
  python download_cctextures.py --textures Wood001 Metal002 Tiles005

  # Download by category
  python download_cctextures.py --category floor wall table

  # Download by prefix (all Wood* textures)
  python download_cctextures.py --prefix Wood Metal --limit 20

  # Download at different resolution
  python download_cctextures.py --resolution 4K

  # List available categories
  python download_cctextures.py --list-categories

  # Search available textures (requires internet)
  python download_cctextures.py --search Wood --limit 50
        """,
    )
    parser.add_argument(
        "--textures",
        nargs="+",
        help="Specific texture IDs to download (e.g., Wood001 Metal002)",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        choices=list(CATEGORIES.keys()),
        help="Download textures by category",
    )
    parser.add_argument(
        "--prefix",
        nargs="+",
        help="Download textures by prefix (e.g., Wood, Metal)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit textures per prefix (default: 20)",
    )
    parser.add_argument(
        "--resolution",
        choices=["1K", "2K", "4K", "8K"],
        default="2K",
        help="Texture resolution (default: 2K)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "resources" / "cctextures",
        help="Output directory (default: ../resources/cctextures)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories",
    )
    parser.add_argument(
        "--search",
        help="Search for textures online (prefix filter)",
    )

    args = parser.parse_args()

    # List categories
    if args.list_categories:
        print("Texture Categories for Domain Randomization:")
        print("=" * 50)
        for cat, prefixes in CATEGORIES.items():
            print(f"\n{cat.upper()}:")
            print(f"  Prefixes: {', '.join(prefixes)}")
        return 0

    # Search online
    if args.search:
        print(f"Searching for '{args.search}' textures...")
        textures = fetch_texture_list(args.search, args.limit)
        if textures:
            print(f"\nFound {len(textures)} textures:")
            for t in textures:
                print(f"  - {t['assetId']}")
        else:
            print("No textures found")
        return 0

    # Determine which textures to download
    textures_to_download = []

    if args.textures:
        textures_to_download = args.textures
    elif args.category:
        prefixes = get_textures_by_category(args.category)
        for prefix in prefixes:
            for i in range(1, args.limit + 1):
                textures_to_download.append(f"{prefix}{i:03d}")
    elif args.prefix:
        for prefix in args.prefix:
            for i in range(1, args.limit + 1):
                textures_to_download.append(f"{prefix}{i:03d}")
    else:
        # Default: download curated textures
        textures_to_download = DEFAULT_TEXTURES.copy()

    # Remove duplicates while preserving order
    seen = set()
    textures_to_download = [x for x in textures_to_download if not (x in seen or seen.add(x))]

    if not textures_to_download:
        print("No textures to download")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CC0 textures from ambientCG.com")
    print(f"Output directory: {args.output_dir}")
    print(f"Resolution: {args.resolution}")
    print(f"Textures: {len(textures_to_download)}")
    print("=" * 60)

    success_count = 0
    failed = []

    for i, texture_id in enumerate(textures_to_download):
        print(f"\n[{i+1}/{len(textures_to_download)}] {texture_id}")
        if download_texture(texture_id, args.output_dir, args.resolution, args.force):
            success_count += 1
        else:
            failed.append(texture_id)

    print("\n" + "=" * 60)
    print(f"Downloaded: {success_count}/{len(textures_to_download)} textures")

    if failed:
        print(f"\nFailed textures ({len(failed)}):")
        for t in failed[:20]:  # Show first 20
            print(f"  - {t}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        return 1
    else:
        print("All downloads completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
