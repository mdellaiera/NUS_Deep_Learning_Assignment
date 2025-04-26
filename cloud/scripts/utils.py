import os
from typing import List, Tuple, Union
from PIL import Image
import numpy as np


def get_filenames(input_dir: str) -> List[str]:
    """
    Get the list of image filenames from the input directory.
    Args:
        input_dir (str): Directory containing the images.
    Returns:
        List[str]: List of image filenames.
    """
    filenames = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            filenames.append(os.path.join(input_dir, file))
    return sorted(filenames)


def load_image(
        path: str,
        crop: Tuple[int, int, int, int] = None,
        convert: str = "RGB") -> Union[np.ndarray, None]:
    """
    Load an image from the given path and convert it to a numpy array.
    Args:
        path (str): Path to the image.
        crop (Tuple[int, int, int, int]): Coordinates for cropping (left, upper, right, lower).
        convert (str): Color mode to convert the image to (default is "RGB").
    Returns:
        np.ndarray: Numpy array of the image.
    """
    try:
        with Image.open(path) as img:
            img = img.convert(convert)
            if crop is not None:
                img = img.crop(crop)
            return np.array(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def patchify(
        img: np.ndarray,
        patch_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """
    Patchify the image into patches of size patch_size.
    Args:
        img (np.ndarray): The image to patchify.
        patch_size (Tuple[int, int]): The size of the patches.
    Returns:
        List[np.ndarray]: A list of patches.
    """
    patches = []
    h, w, _ = img.shape
    ph, pw = patch_size
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            patch = img[i:i+ph, j:j+pw]
            if patch.shape[0] == ph and patch.shape[1] == pw:
                patches.append(patch)
    return patches


def save_patches(
        patches: List[np.ndarray], 
        output_dir: str,
        starting_index: int) -> None:
    """
    Save the patches to the output directory.
    Args:
        patches: The list of patches.
        output_dir: The output directory.
        starting_index: The starting index for naming the files.
    Returns:
        None
    """
    for idx, patch in enumerate(patches):
        filename = os.path.join(output_dir, f"{starting_index + idx:06d}.jpg")
        patch_img = Image.fromarray(patch)
        patch_img.save(filename)