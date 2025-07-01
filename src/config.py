
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations')

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TEXT_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'recognized_texts')
VISUALS_DIR = os.path.join(RESULTS_DIR, 'visualizations')

TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

for path in [RAW_DIR, PROCESSED_DIR, ANNOTATIONS_DIR, TEXT_OUTPUT_DIR, VISUALS_DIR]:
    os.makedirs(path, exist_ok=True)
