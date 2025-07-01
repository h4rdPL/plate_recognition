import cv2
import easyocr
import torch
import pytesseract
import re
import numpy as np
from collections import Counter

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class LicensePlateOCR:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.reader = easyocr.Reader(
            ['en'],
            gpu=self.gpu_available,
            model_storage_directory='model/ocr_models',
            download_enabled=True
        )

        self.plate_patterns = [
            r'^[A-Z]{2,3}\d{4,5}$',
            r'^\d{2}[A-Z]{2}\d{4}$',
            r'^[A-Z]{2}\d{5}$',
            r'^[A-Z]{3}\d{3}$',
            r'^[A-Z]{2}\d{3}[A-Z]{2}$'
        ]

        self.common_prefixes = {
            'W', 'K', 'P', 'G', 'B', 'L', 'S', 'C', 'D', 'E',
            'F', 'H', 'J', 'N', 'R', 'T', 'Z', 'WA', 'KA',
            'PO', 'GA', 'BA', 'LO', 'SC', 'SD', 'EL', 'GD'
        }

    def simple_upscale(self, img, scale_factor=2):
        height, width = img.shape[:2]
        return cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_CUBIC)

    def preprocess_plate(self, img, upscale=True):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if upscale and (gray.shape[0] < 50 or gray.shape[1] < 200):
            gray = self.simple_upscale(gray, 2)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        otsu_inv_clean = cv2.morphologyEx(otsu_inv, cv2.MORPH_CLOSE, kernel)

        return [gray, enhanced, denoised, otsu_clean, otsu_inv_clean, adaptive]

    def validate_plate(self, text):
        text = text.upper().replace(" ", "").replace("-", "")
        if not (4 <= len(text) <= 8):
            return False
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        if not (has_letter and has_number):
            return False
        for pattern in self.plate_patterns:
            if re.fullmatch(pattern, text):
                return True
        return True

    def correct_plate(self, text):
        corrected = []
        for i, c in enumerate(text):
            if i < 3 and c in {'0', '1', '2', '5', '8'}:
                replacements = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
                corrected.append(replacements.get(c, c))
            elif i >= 2 and c in {'O', 'I', 'Z', 'S', 'B'}:
                replacements = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}
                corrected.append(replacements.get(c, c))
            else:
                corrected.append(c)
        return ''.join(corrected)

    def normalize_plate(self, text):
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def ocr_pass(self, image, width_ths=0.7, height_ths=0.7):
        try:
            results = self.reader.readtext(
                image,
                width_ths=width_ths,
                height_ths=height_ths,
                text_threshold=0.7,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                detail=1
            )
            return [(res[1].upper().replace(" ", "").replace("-", ""), res[2])
                    for res in results if len(res[1].strip()) >= 4]
        except:
            return []

    def tesseract_ocr(self, img):
        try:
            config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(img, config=config)
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            return clean_text if len(clean_text) >= 4 else ""
        except:
            return ""

    def vote_best_candidate(self, candidates):
        if not candidates:
            return "", 0.0

        by_length = {}
        for text, conf in candidates:
            length = len(text)
            if length not in by_length:
                by_length[length] = []
            by_length[length].append((text, conf))

        preferred_lengths = [6, 7, 5, 8]
        selected_candidates = []

        for length in preferred_lengths:
            if length in by_length:
                selected_candidates = by_length[length]
                break

        if not selected_candidates:
            selected_candidates = candidates

        return max(selected_candidates, key=lambda x: x[1])

    def read_plate(self, img):
        variants = self.preprocess_plate(img)
        variants.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        candidates = []
        test_variants = variants[:4] + [variants[-1]]
        param_sets = [(0.7, 0.7), (0.9, 0.9)]

        for variant in test_variants:
            if np.mean(variant) > 127:
                variant = cv2.bitwise_not(variant)
            for width_ths, height_ths in param_sets:
                results = self.ocr_pass(variant, width_ths, height_ths)
                candidates.extend(results)

        for variant in variants[:2]:
            if np.mean(variant) > 127:
                variant = cv2.bitwise_not(variant)
            tesseract_result = self.tesseract_ocr(variant)
            if tesseract_result:
                candidates.append((tesseract_result, 0.6))

        if not candidates:
            return "", 0.0

        best_text, best_conf = self.vote_best_candidate(candidates)
        corrected = self.correct_plate(best_text)

        if self.validate_plate(corrected):
            return corrected, best_conf
        elif self.validate_plate(best_text):
            return best_text, best_conf
        else:
            return best_text, best_conf * 0.5
