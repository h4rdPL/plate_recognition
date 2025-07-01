import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, List

def parse_cvat_annotations(xml_path: Path) -> Dict[str, List[Tuple[List[int], str]]]:
    """
    Parses CVAT XML annotations and extracts plate bounding boxes and plate numbers.

    Args:
        xml_path: Path to annotations.xml file

    Returns:
        Dictionary mapping image filenames to a list of tuples: (bbox, plate_number)
        bbox is a list [x_min, y_min, x_max, y_max]
    """
    ground_truth = {}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for image in root.findall('image'):
            filename = image.get('name')
            boxes = []
            for box in image.findall('box'):
                if box.get('label') == 'plate':
                    xtl = float(box.get('xtl'))
                    ytl = float(box.get('ytl'))
                    xbr = float(box.get('xbr'))
                    ybr = float(box.get('ybr'))
                    plate_number_element = box.find("attribute[@name='plate number']")
                    plate_number = plate_number_element.text if plate_number_element is not None else ""

                    bbox = [int(xtl), int(ytl), int(xbr), int(ybr)]
                    boxes.append((bbox, plate_number.upper().replace(" ", "") if plate_number else ""))

            if boxes:
                ground_truth[filename] = boxes

    except Exception as e:
        print(f"Error parsing XML: {e}")

    return ground_truth
