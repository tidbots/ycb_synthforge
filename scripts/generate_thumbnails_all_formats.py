#!/usr/bin/env python3
# BlenderProc must be imported first before any other imports
import blenderproc as bproc

"""
YCB Object Thumbnail Generator - All Formats Comparison
Generates thumbnail images comparing clouds/google_16k/poisson/tsdf formats.
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# All formats to compare
FORMATS = ['clouds', 'google_16k', 'poisson', 'tsdf']


def setup_simple_scene():
    """Setup a simple scene with ground plane and lighting."""
    ground = bproc.object.create_primitive("PLANE", scale=[2, 2, 1])
    ground.set_location([0, 0, 0])
    mat = bproc.material.create("ground_material")
    mat.set_principled_shader_value("Base Color", [0.8, 0.8, 0.8, 1.0])
    ground.replace_materials(mat)
    return ground


def setup_lighting():
    """Setup studio-style lighting for thumbnails."""
    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_location([1.5, -1.5, 2.0])
    key_light.set_rotation_euler([math.radians(45), 0, math.radians(45)])
    key_light.set_energy(200)

    fill_light = bproc.types.Light()
    fill_light.set_type("AREA")
    fill_light.set_location([-1.5, -1.0, 1.5])
    fill_light.set_rotation_euler([math.radians(50), 0, math.radians(-30)])
    fill_light.set_energy(100)

    rim_light = bproc.types.Light()
    rim_light.set_type("AREA")
    rim_light.set_location([0, 2.0, 1.5])
    rim_light.set_rotation_euler([math.radians(-60), 0, math.radians(180)])
    rim_light.set_energy(80)

    return [key_light, fill_light, rim_light]


def setup_camera_for_object(obj):
    """Setup camera to frame the object nicely."""
    bbox = obj.get_bound_box()
    bbox_array = np.array(bbox)
    center = bbox_array.mean(axis=0)
    size = bbox_array.max(axis=0) - bbox_array.min(axis=0)
    max_dim = max(size)
    distance = max_dim * 2.5

    cam_x = center[0] + distance * 0.7
    cam_y = center[1] - distance * 0.7
    cam_z = center[2] + distance * 0.5

    cam_pose = bproc.math.build_transformation_mat(
        [cam_x, cam_y, cam_z],
        bproc.camera.rotation_from_forward_vec(
            center - np.array([cam_x, cam_y, cam_z])
        )
    )
    bproc.camera.add_camera_pose(cam_pose)


def get_model_path(obj_dir: Path, format_name: str) -> Path:
    """Get the model file path for a given format."""
    if format_name == 'clouds':
        return obj_dir / format_name / 'merged_cloud.ply'
    else:
        return obj_dir / format_name / 'textured.obj'


def render_object_thumbnail(model_path: Path, output_path: Path, format_name: str):
    """Render a single object thumbnail."""
    try:
        bproc.clean_up(clean_up_camera=True)
        ground = setup_simple_scene()
        lights = setup_lighting()

        # Load object based on format
        if format_name == 'clouds':
            # Load PLY point cloud
            try:
                objs = bproc.loader.load_obj(str(model_path))
            except Exception as e:
                # Try loading as point cloud using Blender directly
                import bpy
                bpy.ops.wm.ply_import(filepath=str(model_path))
                # Get the imported object
                imported = [obj for obj in bpy.context.selected_objects]
                if not imported:
                    logger.warning(f"Could not load point cloud: {model_path}")
                    return False
                # Convert to BlenderProc object
                objs = [bproc.types.MeshObject(obj) for obj in imported]
        else:
            objs = bproc.loader.load_obj(str(model_path))

        if not objs:
            logger.error(f"Failed to load object from {model_path}")
            return False

        obj = objs[0]

        # Center object on ground
        bbox = obj.get_bound_box()
        bbox_array = np.array(bbox)
        min_z = bbox_array[:, 2].min()
        obj.set_location([0, 0, -min_z + 0.01])

        setup_camera_for_object(obj)

        bproc.renderer.set_max_amount_of_samples(64)
        bproc.renderer.set_output_format(file_format="PNG")

        try:
            import bpy
            bpy.context.scene.cycles.use_denoising = True
        except Exception:
            pass

        data = bproc.renderer.render()

        from PIL import Image
        colors = data["colors"][0]
        if colors.max() <= 1:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

        img = Image.fromarray(colors)
        img.save(output_path)

        return True

    except Exception as e:
        logger.error(f"Error rendering {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_all_thumbnails(ycb_dir: Path, output_dir: Path):
    """Generate thumbnails for all YCB objects in all formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for obj_dir in sorted(ycb_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        obj_name = obj_dir.name

        for format_name in FORMATS:
            model_path = get_model_path(obj_dir, format_name)

            if not model_path.exists():
                results.append({
                    'object_name': obj_name,
                    'format': format_name,
                    'success': False,
                    'reason': 'not_found'
                })
                continue

            output_filename = f"{obj_name}_{format_name}.png"
            output_path = output_dir / output_filename

            logger.info(f"Rendering: {obj_name}/{format_name}...")

            success = render_object_thumbnail(model_path, output_path, format_name)

            results.append({
                'object_name': obj_name,
                'format': format_name,
                'success': success,
                'output_path': str(output_path) if success else None
            })

            if success:
                logger.info(f"  -> Saved to {output_path}")
            else:
                logger.error(f"  -> FAILED")

    return results


def create_comparison_grid(output_dir: Path, ycb_dir: Path):
    """Create a comparison grid showing all 4 formats side by side."""
    from PIL import Image, ImageDraw, ImageFont

    # Collect all thumbnails
    thumbnails = {}
    for img_path in output_dir.glob("*.png"):
        if img_path.name == "comparison_grid_all.png":
            continue
        name = img_path.stem
        for fmt in FORMATS:
            suffix = f"_{fmt}"
            if name.endswith(suffix):
                obj_name = name[:-len(suffix)]
                if obj_name not in thumbnails:
                    thumbnails[obj_name] = {}
                thumbnails[obj_name][fmt] = img_path
                break

    if not thumbnails:
        logger.warning("No thumbnails found for comparison grid")
        return

    # Settings
    thumb_size = (200, 200)
    padding = 5
    label_height = 20
    header_height = 30
    name_width = 180

    # Calculate grid size
    num_objects = len(thumbnails)
    cols = len(FORMATS)

    grid_width = name_width + cols * (thumb_size[0] + padding) + padding
    grid_height = header_height + num_objects * (thumb_size[1] + padding + label_height) + padding

    # Create grid image
    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Header
    x = name_width + padding
    for fmt in FORMATS:
        draw.text((x + 10, 8), fmt, fill=(0, 0, 0), font=font)
        x += thumb_size[0] + padding

    y = header_height
    for obj_name in sorted(thumbnails.keys()):
        obj_thumbs = thumbnails[obj_name]

        # Object name label (truncate if too long)
        display_name = obj_name[:22] if len(obj_name) > 22 else obj_name
        draw.text((padding, y + thumb_size[1]//2 - 6), display_name, fill=(0, 0, 0), font=font_small)

        x = name_width + padding

        for fmt in FORMATS:
            if fmt in obj_thumbs:
                try:
                    img = Image.open(obj_thumbs[fmt])
                    img.thumbnail(thumb_size)
                    grid.paste(img, (x, y))
                except Exception as e:
                    draw.rectangle([x, y, x+thumb_size[0], y+thumb_size[1]], outline=(255, 0, 0))
                    draw.text((x+5, y+thumb_size[1]//2), "ERROR", fill=(255, 0, 0), font=font_small)
            else:
                draw.rectangle([x, y, x+thumb_size[0], y+thumb_size[1]], outline=(200, 200, 200), fill=(240, 240, 240))
                draw.text((x+thumb_size[0]//2-15, y+thumb_size[1]//2-6), "N/A", fill=(150, 150, 150), font=font_small)

            x += thumb_size[0] + padding

        y += thumb_size[1] + padding + label_height

    grid_path = output_dir / "comparison_grid_all.png"
    grid.save(grid_path)
    logger.info(f"Comparison grid saved to {grid_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate YCB object thumbnails for all formats')
    parser.add_argument('--ycb-dir', type=str,
                        default='/workspace/models/ycb',
                        help='Path to YCB models directory')
    parser.add_argument('--output', type=str,
                        default='/workspace/data/thumbnails_all_formats',
                        help='Output directory for thumbnails')
    parser.add_argument('--comparison-grid', action='store_true', default=True,
                        help='Generate comparison grid after thumbnails')

    args = parser.parse_args()

    ycb_dir = Path(args.ycb_dir)
    output_dir = Path(args.output)

    if not ycb_dir.exists():
        logger.error(f"YCB directory not found: {ycb_dir}")
        sys.exit(1)

    logger.info(f"Generating thumbnails for YCB objects in: {ycb_dir}")
    logger.info(f"Formats: {', '.join(FORMATS)}")
    logger.info(f"Output directory: {output_dir}")

    bproc.init()

    results = generate_all_thumbnails(ycb_dir, output_dir)

    # Summary
    for fmt in FORMATS:
        fmt_results = [r for r in results if r['format'] == fmt]
        successful = sum(1 for r in fmt_results if r['success'])
        not_found = sum(1 for r in fmt_results if r.get('reason') == 'not_found')
        failed = len(fmt_results) - successful - not_found
        logger.info(f"{fmt}: {successful} successful, {not_found} not found, {failed} failed")

    if args.comparison_grid:
        logger.info("\nGenerating comparison grid...")
        create_comparison_grid(output_dir, ycb_dir)

    logger.info(f"\nThumbnails saved to: {output_dir}")


if __name__ == '__main__':
    main()
