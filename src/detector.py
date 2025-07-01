import cv2
import time
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, Tuple


def detect_and_save_license_plates(
    input_dir: str,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.25
) -> Tuple[int, float, float]:
    """
    Detect license plates in images and save cropped, binarized plates.

    Args:
        input_dir: Directory with input images.
        output_dir: Directory to save cropped plates. Defaults to input_dir/detected_plates.
        conf_threshold: Minimum confidence for detections.

    Returns:
        Tuple of (number_of_plates_detected, total_processing_time_seconds, avg_time_per_image_seconds)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "detected_plates"
    output_path.mkdir(exist_ok=True, parents=True)

    start_time = time.time()
    model = YOLO("../license_plate_detector.pt")
    model.model.eval()
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    detected_count = 0
    batch_size = 4

    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i+batch_size]
        batch_images = []
        batch_originals = []

        for img_path in batch_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                batch_images.append(img)
                batch_originals.append((img_path, img))

        if not batch_images:
            continue

        results = model(batch_images, conf=conf_threshold)

        for idx, result in enumerate(results):
            img_path, original_img = batch_originals[idx]
            boxes = result.boxes

            if len(boxes) == 0:
                continue

            seen_boxes = []

            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    h, w = original_img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Filter overlapping boxes (IoU > 0.5)
                    box_area = (x2 - x1) * (y2 - y1)
                    skip = False
                    for sx1, sy1, sx2, sy2 in seen_boxes:
                        inter_x1 = max(x1, sx1)
                        inter_y1 = max(y1, sy1)
                        inter_x2 = min(x2, sx2)
                        inter_y2 = min(y2, sy2)
                        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            union_area = box_area + (sx2 - sx1) * (sy2 - sy1) - inter_area
                            iou = inter_area / union_area
                            if iou > 0.5:
                                skip = True
                                break
                    if skip:
                        continue

                    seen_boxes.append((x1, y1, x2, y2))

                    plate_img = original_img[y1:y2, x1:x2]
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    _, bw_plate = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    output_file_path = output_path / f"{img_path.stem}_plate{detected_count}{img_path.suffix}"
                    cv2.imwrite(str(output_file_path), bw_plate)

                    detected_count += 1

                except (IndexError, ValueError):
                    continue

    total_elapsed = time.time() - start_time
    num_images = len(image_files)
    avg_time = total_elapsed / num_images if num_images > 0 else 0

    print(f"Detection Summary: {detected_count} plates from {num_images} images")
    print(f"Detection Time: {total_elapsed:.2f}s (avg: {avg_time:.4f}s/img, est. 100 imgs: {avg_time*100:.2f}s)")

    return detected_count, total_elapsed, avg_time
