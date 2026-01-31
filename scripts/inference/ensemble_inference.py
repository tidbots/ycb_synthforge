#!/usr/bin/env python3
"""
Ensemble inference with multiple YOLO models.
Combines predictions from multiple models using NMS.

Usage:
    # Image inference
    python ensemble_inference.py \
        --models yolo26n.pt ycb_best.pt custom.pt \
        --source image.jpg \
        --output results/

    # Webcam inference
    python ensemble_inference.py \
        --models yolo26n.pt ycb_best.pt \
        --source 0 \
        --realtime

    # With custom class offsets
    python ensemble_inference.py \
        --models yolo26n.pt ycb_best.pt \
        --offsets 0 80 \
        --source images/
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class EnsembleDetector:
    """Ensemble detector combining multiple YOLO models."""

    def __init__(
        self,
        model_paths: List[str],
        class_offsets: Optional[List[int]] = None,
        device: str = 'cuda:0',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize ensemble detector.

        Args:
            model_paths: List of paths to YOLO model weights
            class_offsets: Class ID offsets for each model (auto-calculated if None)
            device: Device to run inference on
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load models
        print("Loading models...")
        self.models = []
        for path in model_paths:
            print(f"  Loading {path}...")
            model = YOLO(path)
            model.to(device)
            self.models.append(model)

        # Calculate class offsets if not provided
        if class_offsets is None:
            self.class_offsets = self._calculate_offsets()
        else:
            self.class_offsets = class_offsets

        # Build merged class names
        self.class_names = self._build_class_names()

        print(f"\nLoaded {len(self.models)} models")
        print(f"Total classes: {len(self.class_names)}")
        print(f"Class offsets: {self.class_offsets}")

    def _calculate_offsets(self) -> List[int]:
        """Calculate class offsets based on model class counts."""
        offsets = [0]
        for i, model in enumerate(self.models[:-1]):
            num_classes = len(model.names)
            offsets.append(offsets[-1] + num_classes)
        return offsets

    def _build_class_names(self) -> Dict[int, str]:
        """Build merged class names dictionary."""
        class_names = {}
        for model, offset in zip(self.models, self.class_offsets):
            for cls_id, name in model.names.items():
                merged_id = cls_id + offset
                # Add model suffix if class name already exists
                if merged_id in class_names:
                    class_names[merged_id] = f"{name}_m{self.models.index(model)}"
                else:
                    class_names[merged_id] = name
        return class_names

    def predict(
        self,
        image: np.ndarray,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Run ensemble prediction on an image.

        Args:
            image: Input image (BGR format)
            verbose: Print timing info

        Returns:
            List of detections with keys: xyxy, conf, cls, class_name
        """
        all_detections = []
        timings = []

        # Run each model
        for i, (model, offset) in enumerate(zip(self.models, self.class_offsets)):
            t0 = time.time()

            results = model(image, conf=self.conf_threshold, verbose=False)[0]

            t1 = time.time()
            timings.append(t1 - t0)

            # Extract detections
            boxes = results.boxes
            for j in range(len(boxes)):
                det = {
                    'xyxy': boxes.xyxy[j].cpu().numpy(),
                    'conf': float(boxes.conf[j]),
                    'cls': int(boxes.cls[j]) + offset,
                    'model_idx': i,
                }
                det['class_name'] = self.class_names.get(det['cls'], f"class_{det['cls']}")
                all_detections.append(det)

        # Apply NMS
        t0 = time.time()
        merged = self._apply_nms(all_detections)
        t_nms = time.time() - t0

        if verbose:
            for i, t in enumerate(timings):
                print(f"  Model {i}: {t*1000:.1f}ms")
            print(f"  NMS: {t_nms*1000:.1f}ms")
            print(f"  Total detections: {len(all_detections)} -> {len(merged)}")

        return merged

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to merge detections."""
        if not detections:
            return []

        # Convert to tensors
        boxes = torch.tensor([d['xyxy'] for d in detections])
        scores = torch.tensor([d['conf'] for d in detections])
        classes = torch.tensor([d['cls'] for d in detections])

        # Apply NMS per class
        keep_indices = []
        unique_classes = classes.unique()

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_indices = torch.where(cls_mask)[0]

            if len(cls_indices) == 0:
                continue

            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            # NMS
            from torchvision.ops import nms
            keep = nms(cls_boxes, cls_scores, self.iou_threshold)

            keep_indices.extend(cls_indices[keep].tolist())

        # Return kept detections
        return [detections[i] for i in keep_indices]

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_conf: bool = True,
        show_model: bool = False,
    ) -> np.ndarray:
        """Draw detections on image."""
        result = image.copy()

        # Color palette for different models
        colors = [
            (0, 255, 0),    # Green - Model 0
            (255, 0, 0),    # Blue - Model 1
            (0, 0, 255),    # Red - Model 2
            (255, 255, 0),  # Cyan - Model 3
            (255, 0, 255),  # Magenta - Model 4
        ]

        for det in detections:
            x1, y1, x2, y2 = map(int, det['xyxy'])
            cls_name = det['class_name']
            conf = det['conf']
            model_idx = det.get('model_idx', 0)

            color = colors[model_idx % len(colors)]

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Build label
            label = cls_name
            if show_conf:
                label += f" {conf:.2f}"
            if show_model:
                label += f" (M{model_idx})"

            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - h - 5), (x1 + w, y1), color, -1)

            # Draw label text
            cv2.putText(
                result, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return result


def run_image_inference(
    detector: EnsembleDetector,
    source: str,
    output_dir: str,
    show_model: bool = False,
):
    """Run inference on images."""
    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = list(source_path.glob('*.jpg')) + \
                      list(source_path.glob('*.png')) + \
                      list(source_path.glob('*.jpeg'))

    print(f"\nProcessing {len(image_files)} images...")

    all_results = []

    for img_path in image_files:
        print(f"  {img_path.name}...", end=' ')

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print("FAILED (cannot read)")
            continue

        # Predict
        t0 = time.time()
        detections = detector.predict(image)
        t1 = time.time()

        print(f"{len(detections)} detections ({(t1-t0)*1000:.0f}ms)")

        # Save result
        result_image = detector.draw_detections(image, detections, show_model=show_model)
        result_path = output_path / f"result_{img_path.name}"
        cv2.imwrite(str(result_path), result_image)

        # Store results
        all_results.append({
            'image': img_path.name,
            'detections': [
                {
                    'class': d['class_name'],
                    'class_id': d['cls'],
                    'confidence': d['conf'],
                    'bbox': d['xyxy'].tolist(),
                }
                for d in detections
            ]
        })

    # Save JSON results
    json_path = output_path / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def run_realtime_inference(
    detector: EnsembleDetector,
    camera_id: int = 0,
    show_model: bool = True,
):
    """Run realtime inference on webcam."""
    print(f"\nStarting webcam {camera_id}...")
    print("Press 'q' to quit, 's' to save screenshot")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    fps_start = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict
        detections = detector.predict(frame)

        # Draw results
        result = detector.draw_detections(frame, detections, show_model=show_model)

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Draw FPS
        cv2.putText(
            result, f"FPS: {fps:.1f} | Detections: {len(detections)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # Show
        cv2.imshow('Ensemble Detection', result)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, result)
            print(f"Saved: {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble inference with multiple YOLO models'
    )
    parser.add_argument(
        '--models', type=str, nargs='+', required=True,
        help='Paths to YOLO model weights'
    )
    parser.add_argument(
        '--source', type=str, required=True,
        help='Image/directory path or camera ID (0, 1, ...)'
    )
    parser.add_argument(
        '--output', type=str, default='outputs/ensemble_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--offsets', type=int, nargs='+',
        help='Class ID offsets for each model (auto-calculated if not specified)'
    )
    parser.add_argument(
        '--conf', type=float, default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou', type=float, default=0.5,
        help='IoU threshold for NMS (default: 0.5)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device (default: cuda:0)'
    )
    parser.add_argument(
        '--realtime', action='store_true',
        help='Run realtime webcam inference'
    )
    parser.add_argument(
        '--show-model', action='store_true',
        help='Show model index in detection labels'
    )

    args = parser.parse_args()

    # Create detector
    detector = EnsembleDetector(
        model_paths=args.models,
        class_offsets=args.offsets,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    # Run inference
    if args.realtime or args.source.isdigit():
        camera_id = int(args.source) if args.source.isdigit() else 0
        run_realtime_inference(detector, camera_id, args.show_model)
    else:
        run_image_inference(detector, args.source, args.output, args.show_model)


if __name__ == '__main__':
    main()
