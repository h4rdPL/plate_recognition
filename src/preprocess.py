import cv2

def crop_plate(img, bbox):
    """
    Crop the plate area from the image using bounding box coordinates.

    bbox: [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    return img[y_min:y_max, x_min:x_max]

def preprocess_image(img, bbox):

    x, y, w, h = bbox
    cropped = img[y:y+h, x:x+w]

    return cropped