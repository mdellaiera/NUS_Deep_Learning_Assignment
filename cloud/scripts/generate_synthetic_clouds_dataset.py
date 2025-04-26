import os
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from utils import load_image, patchify, save_patches, get_filenames


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process synthetic clouds dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the images to process.")
    parser.add_argument("--input_mask", type=str, required=True, help="Path to the mask image.")
    parser.add_argument("--patch_size", type=int, nargs=2, required=True, help="Size of the patches (height, width).")
    parser.add_argument("--crop", type=int, nargs=4, required=True, help="Coordinates for cropping (left, upper, right, lower).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the patches.")
    parser.add_argument("--max_files", type=int, default=-1, help="Maximum number of files to process. Default is -1 (all files).")
    return parser


def generate_perlin_noise_2d(shape: Tuple[int, int], res: Tuple[int, int]) -> np.ndarray:
    """
    Generate a 2D Perlin noise array.

    Args:
        shape: tuple (height, width), final image size
        res: tuple (number of periods of noise to generate along height, width)

    Returns:
        np.ndarray: Generated perlin noise
    """
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])

    # Random gradients
    gradients = np.random.randn(res[0]+1, res[1]+1, 2)
    gradients /= np.linalg.norm(gradients, axis=2, keepdims=True)

    # Coordinates
    x = np.linspace(0, res[1], shape[1], endpoint=False)
    y = np.linspace(0, res[0], shape[0], endpoint=False)

    x0 = x.astype(int)
    y0 = y.astype(int)

    # Grid indices
    xi, yi = np.meshgrid(x0, y0)
    xf, yf = np.meshgrid(x - x0, y - y0)

    # Gradient vectors
    g00 = gradients[yi    % res[0], xi    % res[1]]
    g10 = gradients[yi    % res[0], (xi+1)% res[1]]
    g01 = gradients[(yi+1)% res[0], xi    % res[1]]
    g11 = gradients[(yi+1)% res[0], (xi+1)% res[1]]

    # Dot products
    dot00 = (g00[:,:,0] * xf + g00[:,:,1] * yf)
    dot10 = (g10[:,:,0] * (xf-1) + g10[:,:,1] * yf)
    dot01 = (g01[:,:,0] * xf + g01[:,:,1] * (yf-1))
    dot11 = (g11[:,:,0] * (xf-1) + g11[:,:,1] * (yf-1))

    # Interpolation
    u = f(xf)
    v = f(yf)
    nx0 = dot00*(1-u) + dot10*u
    nx1 = dot01*(1-u) + dot11*u
    nxy = nx0*(1-v) + nx1*v

    return nxy


def generate_synthetic_clouds(
        shape: Tuple[int, int],
        res: Tuple[int, int],
        octaves: int) -> np.ndarray:
    """Generate a 2D Perlin noise with multiple octaves."""
    noise = np.zeros(shape, dtype=np.float32)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (res[0]*frequency, res[1]*frequency))
        frequency *= 2
        amplitude /= 2
    return noise


def apply_synthetic_clouds_to_mask(
        noise: np.ndarray,
        mask: np.ndarray) -> np.ndarray:
    """Normalize noise and apply it to mask."""
    noise_min = noise.min()
    noise_max = noise.max()
    noise = (noise - noise_min) / (noise_max - noise_min)
    noise = (noise * 255).astype(np.uint8)

    mask = mask.astype(np.uint8)
    result = np.clip(noise + mask, 0, 255).astype(np.uint8)
    return result


def process(
        N: int, 
        input_mask: str,
        patch_size: Tuple[int, int],
        crop: Tuple[int, int, int, int],
        output_dir: str) -> None:
    """
    Generate synthetic cloud images and save them.
    """
    mask_img = load_image(input_mask, crop, convert="L")
    if mask_img is None:
        raise ValueError("Failed to load the mask.")

    index = 0
    for _ in tqdm(range(N), desc="Generating synthetic images"):
        noise = generate_synthetic_clouds(mask_img.shape, res=(4, 4), octaves=6)
        synthetic = apply_synthetic_clouds_to_mask(noise, mask_img)
        patches = patchify(np.stack([synthetic]*3, axis=-1), patch_size)  # make it 3-channel
        save_patches(patches, output_dir, starting_index=index)
        index += len(patches)


def main():
    parser = build_argparser()
    args = parser.parse_args()
    filenames = get_filenames(args.input_dir)
    if args.max_files > 0:
        filenames = filenames[:args.max_files]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    process(len(filenames), args.input_mask, args.patch_size, args.crop, args.output_dir)


if __name__ == "__main__":
    main()