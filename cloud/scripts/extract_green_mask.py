import os
import cv2
import numpy as np
from PIL import Image

def fixed_crop(img: np.ndarray, crop_box: tuple) -> np.ndarray:
    """
    Crop the image to the specified box (x_min, y_min, x_max, y_max).

    Args:
        img (np.ndarray): Input image.
        crop_box (tuple): (x_min, y_min, x_max, y_max).

    Returns:
        np.ndarray: Cropped image.
    """
    x_min, y_min, x_max, y_max = crop_box

    # Check if the crop box is valid
    h, w = img.shape
    if x_max > w or y_max > h:
        raise ValueError(f"Crop box {crop_box} exceeds image size {(w, h)}.")

    cropped = img[y_min:y_max, x_min:x_max]
    return cropped

def extract_green_mask(image_path: str, output_path: str, crop_box=(21, 144, 1045, 656)) -> None:
    """
    Extract green borders from the input image, crop using fixed coordinates, and save as a binary mask.

    Args:
        image_path (str): Path to the input image with green borders.
        output_path (str): Path to save the generated binary mask.
        crop_box (tuple): (x_min, y_min, x_max, y_max).
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image {image_path}")
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define green color range
    lower_green = np.array([35, 40, 40])   # 可以微调
    upper_green = np.array([85, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Optional: clean up small noise (morphology)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Crop using fixed box
    mask_cropped = fixed_crop(mask, crop_box)

    # Save as 0/255 binary mask
    mask_img = Image.fromarray(mask_cropped)
    mask_img.save(output_path)

    print(f"Mask saved to {output_path}, cropped with box {crop_box}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract green borders, crop using fixed box, and save as mask.")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_mask", type=str, required=True, help="Path to output mask")
    args = parser.parse_args()

    # 固定 crop 坐标
    crop_box = (21, 144, 1045, 656)

    extract_green_mask(args.input_image, args.output_mask, crop_box)