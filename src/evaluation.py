import numpy as np
from typing import Dict, Tuple


def evaluate_ocr_performance(
    ground_truth: Dict[str, str],
    ocr_results: Dict[str, Tuple[str, float]]
) -> Dict[str, float]:
    """
    Compare OCR results against ground truth.

    Args:
        ground_truth: Dict mapping image_name -> true plate text.
        ocr_results: Dict mapping image_name -> (recognized_text, confidence).

    Returns:
        Dict with evaluation metrics: total_images, detected_images,
        exact_match_accuracy, char_level_accuracy, avg_confidence.
    """
    total = len(ground_truth)
    correct = 0
    char_accuracies = []

    for img_name, true_text in ground_truth.items():
        if img_name not in ocr_results:
            continue

        recognized_text, confidence = ocr_results[img_name]

        # Exact match check
        if recognized_text == true_text:
            correct += 1

        # Character-level accuracy
        matched_chars = sum(t == r for t, r in zip(true_text, recognized_text))
        if len(true_text) > 0:
            char_acc = matched_chars / len(true_text)
            char_accuracies.append(char_acc)

    avg_char_accuracy = np.mean(char_accuracies) if char_accuracies else 0
    avg_confidence = np.mean([conf for _, conf in ocr_results.values()]) if ocr_results else 0

    return {
        'total_images': total,
        'detected_images': len(ocr_results),
        'exact_match_accuracy': correct / total if total > 0 else 0,
        'char_level_accuracy': avg_char_accuracy,
        'avg_confidence': avg_confidence
    }


def calculate_final_grade(
    accuracy_percent: float,
    processing_time_sec: float
) -> float:
    """
    Calculate a final grade (2.0 to 5.0) based on OCR accuracy and processing time.

    Args:
        accuracy_percent: OCR accuracy percentage (0–100).
        processing_time_sec: Time in seconds to process 100 images.

    Returns:
        Grade rounded to nearest 0.5.
    """
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0

    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50

    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score

    return round(grade * 2) / 2
