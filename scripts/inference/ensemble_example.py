#!/usr/bin/env python3
"""
Simple example of ensemble inference usage.

This example shows how to combine:
- YOLO26 pre-trained model (COCO 80 classes)
- YCB trained model (85 classes)
- Custom model (N classes)
"""

from ensemble_inference import EnsembleDetector
import cv2


def example_basic():
    """Basic ensemble usage."""

    # Initialize with multiple models
    detector = EnsembleDetector(
        model_paths=[
            'weights/yolo26n.pt',           # COCO: 80 classes (0-79)
            'outputs/ycb_best.pt',          # YCB: 85 classes (80-164)
            # 'outputs/custom_best.pt',     # Custom: N classes (165+)
        ],
        # class_offsets=[0, 80, 165],  # Auto-calculated if not specified
        device='cuda:0',
        conf_threshold=0.3,
        iou_threshold=0.5,
    )

    # Run inference on image
    image = cv2.imread('test_image.jpg')
    detections = detector.predict(image, verbose=True)

    # Process results
    for det in detections:
        print(f"  {det['class_name']}: {det['conf']:.2f} at {det['xyxy']}")

    # Draw and save
    result = detector.draw_detections(image, detections, show_model=True)
    cv2.imwrite('result.jpg', result)


def example_same_classes():
    """
    Ensemble with overlapping classes.
    Useful when you have multiple models trained on similar data.
    """

    detector = EnsembleDetector(
        model_paths=[
            'model_v1.pt',  # Same classes
            'model_v2.pt',  # Same classes
        ],
        class_offsets=[0, 0],  # Same offset = merge same classes
        iou_threshold=0.5,
    )

    # This will merge predictions for same classes via NMS
    image = cv2.imread('test.jpg')
    detections = detector.predict(image)

    return detections


def example_selective_merge():
    """
    Selective ensemble - only use specific model for certain scenarios.
    """
    from ultralytics import YOLO

    # Load models separately
    coco_model = YOLO('yolo26n.pt')
    ycb_model = YOLO('outputs/ycb_best.pt')

    def smart_predict(image, scene_type='general'):
        """
        Choose model based on scene type.
        """
        if scene_type == 'warehouse':
            # Use YCB model for warehouse/industrial scenes
            results = ycb_model(image, conf=0.3)
        elif scene_type == 'general':
            # Use COCO model for general scenes
            results = coco_model(image, conf=0.3)
        else:
            # Use both (ensemble)
            detector = EnsembleDetector(
                model_paths=['yolo26n.pt', 'outputs/ycb_best.pt'],
                device='cuda:0'
            )
            return detector.predict(image)

        return results

    return smart_predict


def example_parallel_gpu():
    """
    Parallel inference on multiple GPUs.
    Each model on a different GPU for maximum throughput.
    """
    import threading
    import queue

    class ParallelEnsemble:
        def __init__(self, model_configs):
            """
            model_configs: List of (model_path, device, class_offset)
            """
            self.models = []
            self.queues = []

            for path, device, offset in model_configs:
                model = YOLO(path)
                model.to(device)
                self.models.append({
                    'model': model,
                    'device': device,
                    'offset': offset,
                    'queue': queue.Queue()
                })

        def predict(self, image):
            threads = []
            results = []

            # Start parallel inference
            for m in self.models:
                t = threading.Thread(
                    target=self._predict_single,
                    args=(m, image)
                )
                t.start()
                threads.append(t)

            # Wait for all
            for t in threads:
                t.join()

            # Collect results
            all_dets = []
            for m in self.models:
                dets = m['queue'].get()
                all_dets.extend(dets)

            # Apply NMS
            return self._nms(all_dets)

        def _predict_single(self, model_info, image):
            model = model_info['model']
            offset = model_info['offset']

            results = model(image, verbose=False)[0]

            dets = []
            for box in results.boxes:
                dets.append({
                    'xyxy': box.xyxy[0].cpu().numpy(),
                    'conf': float(box.conf),
                    'cls': int(box.cls) + offset,
                })

            model_info['queue'].put(dets)

        def _nms(self, detections):
            # Simplified NMS
            # In production, use torchvision.ops.nms
            return detections

    # Usage
    ensemble = ParallelEnsemble([
        ('yolo26n.pt', 'cuda:0', 0),
        ('ycb_best.pt', 'cuda:1', 80),
    ])

    return ensemble


if __name__ == '__main__':
    print("Ensemble Inference Examples")
    print("=" * 50)
    print("\nSee the code for usage examples:")
    print("  - example_basic(): Basic ensemble usage")
    print("  - example_same_classes(): Merge same classes")
    print("  - example_selective_merge(): Smart model selection")
    print("  - example_parallel_gpu(): Multi-GPU parallel inference")
