#!/usr/bin/env python3
# BlenderProc must be imported first before any other imports
import blenderproc as bproc

"""
YCB Object Dataset Generator using BlenderProc
Generates photorealistic synthetic data with domain randomization for YOLO training.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from camera import CameraRandomizer
from lighting import LightingRandomizer
from materials import MaterialRandomizer
from scene_setup import SceneSetup
from ycb_classes import NUM_CLASSES, YCB_CLASSES, YCB_NAME_TO_ID, get_material_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


# Objects to exclude due to mesh/texture issues
EXCLUDED_OBJECTS = {
    "019_pitcher_base",   # Mesh appears deformed/crumpled
    "022_windex_bottle",  # Mesh appears deformed/crumpled
}


def get_ycb_model_paths(ycb_dir: str) -> Dict[str, str]:
    """
    Get paths to all YCB model OBJ files.

    Args:
        ycb_dir: Path to YCB models directory

    Returns:
        Dictionary mapping object name to OBJ file path
    """
    model_paths = {}
    ycb_path = Path(ycb_dir)

    for obj_name in YCB_CLASSES.values():
        # Skip excluded objects
        if obj_name in EXCLUDED_OBJECTS:
            logger.info(f"Excluding {obj_name}: known mesh/texture issues")
            continue

        # ONLY use google_16k format (poisson format has corrupted normals)
        google_path = ycb_path / obj_name / "google_16k" / "textured.obj"
        if google_path.exists():
            model_paths[obj_name] = str(google_path)
        else:
            logger.warning(f"Skipping {obj_name}: google_16k format not available")

    logger.info(f"Found {len(model_paths)} YCB models")
    return model_paths


def load_ycb_objects(
    model_paths: Dict[str, str],
    selected_objects: List[str],
    material_randomizer: MaterialRandomizer,
    use_physics: bool = False,
) -> List[bproc.types.MeshObject]:
    """
    Load and prepare YCB objects for the scene.

    Args:
        model_paths: Dictionary of object names to file paths
        selected_objects: List of object names to load
        material_randomizer: Material randomizer instance
        use_physics: Whether to enable physics simulation

    Returns:
        List of loaded mesh objects
    """
    loaded_objects = []

    for obj_name in selected_objects:
        if obj_name not in model_paths:
            logger.warning(f"Skipping {obj_name}: model path not found")
            continue

        try:
            # Load the object
            objs = bproc.loader.load_obj(model_paths[obj_name])

            for obj in objs:
                # Set object name for identification
                obj.set_name(obj_name)

                # Set custom property for class ID
                class_id = YCB_NAME_TO_ID[obj_name]
                obj.set_cp("category_id", class_id)
                obj.set_cp("class_name", obj_name)

                # Skip material randomization to preserve original textures
                # TODO: Fix material randomization to not break textures
                # material_type = get_material_type(obj_name)
                # material_randomizer.randomize_object_material(obj, material_type)

                # Enable physics only if physics simulation will be used
                if use_physics:
                    obj.enable_rigidbody(
                        active=True,
                        collision_shape="CONVEX_HULL",
                        mass=0.1,
                    )

                loaded_objects.append(obj)

        except Exception as e:
            logger.error(f"Error loading {obj_name}: {e}")
            continue

    return loaded_objects


def generate_scene(
    config: Dict[str, Any],
    model_paths: Dict[str, str],
    scene_setup: SceneSetup,
    lighting_randomizer: LightingRandomizer,
    camera_randomizer: CameraRandomizer,
    material_randomizer: MaterialRandomizer,
    scene_idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Generate a single scene with domain randomization.

    Args:
        config: Configuration dictionary
        model_paths: Dictionary of YCB model paths
        scene_setup: Scene setup instance
        lighting_randomizer: Lighting randomizer instance
        camera_randomizer: Camera randomizer instance
        material_randomizer: Material randomizer instance
        scene_idx: Current scene index

    Returns:
        Scene metadata or None if generation failed
    """
    try:
        # Clear previous scene (keep Blender initialized)
        bproc.clean_up(clean_up_camera=True)

        # Setup background (floor, walls, table)
        room_result = scene_setup.create_room()
        surface_height = room_result.get("surface_height", 0)

        # Get placement config early to check physics setting
        placement_config = config.get("placement", {})
        use_physics = placement_config.get("use_physics", False)

        # Randomly select YCB objects for this scene
        num_objects = random.randint(
            config["scene"]["objects_per_scene"][0],
            config["scene"]["objects_per_scene"][1],
        )
        available_objects = list(model_paths.keys())
        selected_objects = random.sample(
            available_objects,
            min(num_objects, len(available_objects)),
        )

        # Load YCB objects
        ycb_objects = load_ycb_objects(
            model_paths,
            selected_objects,
            material_randomizer,
            use_physics=use_physics,
        )

        if not ycb_objects:
            logger.warning(f"Scene {scene_idx}: No objects loaded, skipping")
            return None

        # Position objects with grid-based spacing to avoid overlap
        pos_cfg = placement_config.get("position", {})
        x_range = pos_cfg.get("x_range", [-0.2, 0.2])
        y_range = pos_cfg.get("y_range", [-0.2, 0.2])

        # Calculate grid spacing based on number of objects
        num_objs = len(ycb_objects)
        grid_size = int(np.ceil(np.sqrt(num_objs)))
        x_step = (x_range[1] - x_range[0]) / max(grid_size, 1)
        y_step = (y_range[1] - y_range[0]) / max(grid_size, 1)

        # Place objects in a grid with random offset
        for i, obj in enumerate(ycb_objects):
            grid_x = i % grid_size
            grid_y = i // grid_size

            # Base position from grid
            base_x = x_range[0] + (grid_x + 0.5) * x_step
            base_y = y_range[0] + (grid_y + 0.5) * y_step

            # Add small random offset (within cell)
            offset_x = random.uniform(-x_step * 0.3, x_step * 0.3)
            offset_y = random.uniform(-y_step * 0.3, y_step * 0.3)

            x = base_x + offset_x
            y = base_y + offset_y
            z = surface_height + 0.05

            obj.set_location([x, y, z])

            # Rotation - upright with random z rotation
            rx = np.radians(random.uniform(-10, 10))
            ry = np.radians(random.uniform(-10, 10))
            rz = np.radians(random.uniform(0, 360))
            obj.set_rotation_euler([rx, ry, rz])

            # Scale up objects to be more visible
            scale_var = placement_config.get("scale_variation", [-0.05, 0.05])
            base_scale = 1.5  # Scale up by 50%
            scale = base_scale * (1.0 + random.uniform(scale_var[0], scale_var[1]))
            obj.set_scale([scale, scale, scale])

        # Run physics simulation to settle objects
        if use_physics:
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=2,
                max_simulation_time=6,
                check_object_interval=1,
                substeps_per_frame=20,
            )

        # Setup lighting
        lighting_randomizer.setup_random_lighting()

        # Setup camera
        camera_randomizer.setup_camera(ycb_objects)

        # Render settings
        render_cfg = config.get("rendering", {})
        bproc.renderer.set_max_amount_of_samples(render_cfg.get("samples", 128))

        if render_cfg.get("use_denoising", True):
            # Enable denoiser via Blender API
            try:
                import bpy
                bpy.context.scene.cycles.use_denoising = True
            except Exception as e:
                logger.debug(f"Could not enable denoiser: {e}")

        # Enable COCO annotations
        for obj in ycb_objects:
            obj.set_cp("category_id", obj.get_cp("category_id"))

        # Render the scene
        data = bproc.renderer.render()

        # Generate COCO annotations
        # Get 2D bounding boxes
        # Use default_value to handle background objects
        seg_data = bproc.renderer.render_segmap(
            map_by=["category_id", "instance", "name"],
            default_values={"category_id": -1}
        )

        # Collect scene metadata
        scene_metadata = {
            "scene_idx": scene_idx,
            "objects": [
                {
                    "name": obj.get_name(),
                    "class_id": obj.get_cp("category_id"),
                    "location": obj.get_location().tolist(),
                    "rotation": obj.get_rotation_euler().tolist(),
                }
                for obj in ycb_objects
            ],
        }

        return {
            "data": data,
            "seg_data": seg_data,
            "metadata": scene_metadata,
            "objects": ycb_objects,
        }

    except Exception as e:
        logger.error(f"Error generating scene {scene_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_coco_annotations(
    output_dir: str,
    all_annotations: List[Dict],
    all_images: List[Dict],
) -> None:
    """
    Save annotations in COCO format.

    Args:
        output_dir: Output directory path
        all_annotations: List of annotation dictionaries
        all_images: List of image metadata dictionaries
    """
    # Build category list
    categories = [
        {"id": class_id, "name": class_name, "supercategory": "ycb_object"}
        for class_id, class_name in YCB_CLASSES.items()
    ]

    coco_format = {
        "info": {
            "description": "YCB Object Synthetic Dataset",
            "version": "1.0",
            "year": 2026,
            "contributor": "BlenderProc",
        },
        "licenses": [],
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories,
    }

    output_path = Path(output_dir) / "annotations.json"
    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=2)

    logger.info(f"Saved COCO annotations to {output_path}")


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic YCB dataset with domain randomization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/scripts/blenderproc/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/data/synthetic/coco",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="Number of scenes to generate (overrides config)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for scene numbering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Setup random seeds
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    setup_random_seeds(seed)

    # Create output directories
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Get YCB model paths
    ycb_dir = config["paths"]["ycb_models"]
    model_paths = get_ycb_model_paths(ycb_dir)

    if not model_paths:
        logger.error("No YCB models found. Exiting.")
        sys.exit(1)

    # Initialize randomizers
    textures_dir = config["paths"]["textures"]

    scene_setup = SceneSetup(config, textures_dir)
    lighting_randomizer = LightingRandomizer(config)
    camera_randomizer = CameraRandomizer(config)
    material_randomizer = MaterialRandomizer(config)

    # Initialize BlenderProc once
    bproc.init()

    # Determine number of scenes
    num_scenes = args.num_scenes if args.num_scenes is not None else config["scene"]["num_images"]

    logger.info(f"Generating {num_scenes} scenes")
    logger.info(f"Output directory: {output_dir}")

    # Track all annotations and images for COCO format
    all_annotations = []
    all_images = []
    annotation_id = 1

    # Generate scenes
    for scene_idx in range(args.start_idx, args.start_idx + num_scenes):
        logger.info(f"Generating scene {scene_idx + 1}/{args.start_idx + num_scenes}")

        result = generate_scene(
            config,
            model_paths,
            scene_setup,
            lighting_randomizer,
            camera_randomizer,
            material_randomizer,
            scene_idx,
        )

        if result is None:
            continue

        # Save rendered image
        image_filename = f"scene_{scene_idx:06d}.png"
        image_path = images_dir / image_filename

        # Get the rendered image
        colors = result["data"]["colors"][0]

        # Save using BlenderProc's writer or manually
        from PIL import Image
        img = Image.fromarray((colors * 255).astype(np.uint8) if colors.max() <= 1 else colors.astype(np.uint8))
        img.save(image_path)

        # Get image dimensions
        height, width = colors.shape[:2]

        # Add image metadata
        image_info = {
            "id": scene_idx,
            "file_name": image_filename,
            "width": width,
            "height": height,
        }
        all_images.append(image_info)

        # Generate bounding box annotations from segmentation
        seg_data = result["seg_data"]

        # Get instance segmentation map
        if "instance_segmaps" not in seg_data:
            logger.warning(f"Scene {scene_idx}: No instance segmentation map")
            continue

        instance_map = seg_data["instance_segmaps"][0]

        # Get instance attribute maps for category_id lookup
        instance_attrs_list = seg_data.get("instance_attribute_maps", [])
        instance_attrs = instance_attrs_list[0] if instance_attrs_list else []

        # Build a mapping from instance idx to attributes
        idx_to_attrs = {}
        if isinstance(instance_attrs, list):
            for attr_dict in instance_attrs:
                if isinstance(attr_dict, dict) and 'idx' in attr_dict:
                    idx_to_attrs[attr_dict['idx']] = attr_dict

        # Build a mapping from object name to category_id from our loaded objects
        obj_name_to_category = {}
        for obj in result.get("objects", []):
            try:
                obj_name = obj.get_name()
                cat_id = obj.get_cp("category_id")
                if cat_id is not None and cat_id >= 0:
                    obj_name_to_category[obj_name] = cat_id
            except Exception:
                pass

        # Get unique instances
        unique_instances = np.unique(instance_map)

        for inst_id in unique_instances:
            if inst_id == 0:  # Skip background
                continue

            # Get mask for this instance
            mask = instance_map == inst_id

            # Get category ID from instance attributes
            category_id = -1

            if inst_id in idx_to_attrs:
                attrs = idx_to_attrs[inst_id]
                # Get category_id directly
                category_id = attrs.get("category_id", -1)

                # If category_id is -1, try to match by name
                if category_id < 0:
                    obj_name = attrs.get("name", "")
                    if obj_name in obj_name_to_category:
                        category_id = obj_name_to_category[obj_name]

            if category_id < 0 or category_id >= NUM_CLASSES:
                continue

            # Calculate bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not np.any(rows) or not np.any(cols):
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # COCO format: [x, y, width, height]
            bbox = [
                int(cmin),
                int(rmin),
                int(cmax - cmin + 1),
                int(rmax - rmin + 1),
            ]

            # Calculate area
            area = int(np.sum(mask))

            # Create annotation
            annotation = {
                "id": annotation_id,
                "image_id": scene_idx,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
            all_annotations.append(annotation)
            annotation_id += 1

        # Log progress
        if (scene_idx + 1) % 100 == 0:
            logger.info(f"Progress: {scene_idx + 1}/{args.start_idx + num_scenes} scenes completed")

    # Save COCO annotations
    save_coco_annotations(str(output_dir), all_annotations, all_images)

    logger.info("Dataset generation complete!")
    logger.info(f"Generated {len(all_images)} images with {len(all_annotations)} annotations")


if __name__ == "__main__":
    main()
