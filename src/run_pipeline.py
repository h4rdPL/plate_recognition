import os
import cv2
from pathlib import Path
from license_plate_ocr import LicensePlateOCR
from ultralytics import YOLO
import torch
import time
import xml.etree.ElementTree as ET


def load_ground_truth_from_xml(xml_path):
    """
    Parses the annotations.xml file and returns a dictionary mapping image filenames
    to a list of ground truth plate strings.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ground_truth = {
    }

    for image in root.findall("image"):
        filename = image.attrib.get("name")
        plates = []
        for plate in image.findall("plate"):
            if plate.text:
                plates.append(plate.text.strip())
        if filename:
            ground_truth[filename] = plates

    return ground_truth


def detect_license_plates(input_dir, conf_threshold=0.25):
    model = YOLO("../license_plate_detector.pt")
    model.model.eval()
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    input_path = Path(input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    detections = {}

    batch_size = 4
    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i + batch_size]
        batch_images = []
        batch_filenames = []

        for img_path in batch_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                batch_images.append(img)
                batch_filenames.append(img_path.name)

        if not batch_images:
            continue

        results = model(batch_images, conf=conf_threshold)

        for idx, result in enumerate(results):
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            bboxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bboxes.append((x1, y1, x2, y2))
            detections[batch_filenames[idx]] = bboxes

    return detections


def main():
    input_dir = "../data/raw/photos"
    output_dir = "../output/debug"
    os.makedirs(output_dir, exist_ok=True)

    annotations_path = "../data/raw/annotations.xml"
    ground_truth = load_ground_truth_from_xml(annotations_path)
    print(f"Loaded ground truth for {len(ground_truth)} images.")

    print("Starting license plate detection...")
    detections = detect_license_plates(input_dir, conf_threshold=0.25)

    ocr = LicensePlateOCR()

    total_images_processed = 0
    total_plates = 0

    start_time = time.time()

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        bboxes = detections.get(filename)
        if not bboxes:
            print(f"No plates detected for {filename}, skipping OCR")
            continue

        total_images_processed += 1

        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            cropped_img = img[y_min:y_max, x_min:x_max]

            # Save cropped plate
            cropped_path = os.path.join(output_dir, f"{filename}_plate{i}_cropped.jpg")
            cv2.imwrite(cropped_path, cropped_img)

            # Preprocess variants
            variants = ocr.preprocess_plate(cropped_img)
            for j, var_img in enumerate(variants):
                var_path = os.path.join(output_dir, f"{filename}_plate{i}_preproc_{j}.jpg")
                cv2.imwrite(var_path, var_img)

            # OCR
            plate_text, confidence = ocr.read_plate(cropped_img)
            print(f"Image {filename} Plate {i}: {plate_text} (Confidence: {confidence:.2f})")

            total_plates += 1

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\nProcessed {total_images_processed} images, {total_plates} plates detected.")
    print(f"Total Processing Time: {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()
