import cv2
import os
import numpy as np

preprocessed_dir = "preprocessed/"

def preprocess_receipt(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None

    # 2. Rescaling - Helpful for small font detection in EasyOCR
    # Upscaling by 2x using cubic interpolation for better edge preservation
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Handling Noise and Blur 
    # Median blur is effective against "salt and pepper" noise common in scans
    denoised = cv2.medianBlur(gray, 3)

    coords = np.column_stack(np.where(denoised > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # Adjusting the angle for OpenCV's coordinate system
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(denoised, matrix, (w, h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    
    if not os.path.exists(preprocessed_dir):
        os.mkdir(preprocessed_dir)
    filename = image_path.split("/")[-1]
    cv2.imwrite(os.path.join(preprocessed_dir, filename), rotated)

    # return rotated