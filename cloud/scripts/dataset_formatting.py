from typing import List, Tuple
import os
from pathlib import Path
import argparse
import random

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process synthetic cloud dataset.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing the images to process."
    )
    parser.add_argument(
        "--patches", type=str, required=True,
        help="Directory containing the original patches to process (modality A)."
    )
    parser.add_argument(
        "--synthetic_patches", type=str, required=True,
        help="Directory containing the synthetic patches to process (modality B)."
    )
    parser.add_argument(
        "--folders", type=str, nargs=4, required=True,
        help="List of folder names to create for training and testing data."
    )
    parser.add_argument(
        "--alpha", type=float, required=True,
        help="Ratio of training data to total data."
    )
    return parser

def browse_folder(
        path: Path, 
        patches: str, 
        synthetic_patches: str) -> Tuple[List[Path], List[Path]]:
    """
    Browse a directory and return lists of all PNG files in patches and synthetic_patches subfolders.
    """
    patches_dir = path / patches
    synthetic_patches_dir = path / synthetic_patches

    patches_files = list(patches_dir.glob("*.jpg"))
    synthetic_patches_files = list(synthetic_patches_dir.glob("*.jpg"))

    print(f"Found {len(patches_files)} patches and {len(synthetic_patches_files)} synthetic patches.")
    return patches_files, synthetic_patches_files

def check_existence(
        filenames: List[Path]) -> None:
    """
    Check if the paths exist.
    """
    for f in filenames:
        if not f.exists():
            raise FileNotFoundError(f"{f} does not exist.")

def create_folders(
        input_dir: Path, 
        folder_list: List[str]) -> None:
    """
    Create folders for training and testing data.
    """
    for folder in folder_list:
        folder_path = input_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder_path}")

def split_train_test(
        filenames: List[Path], 
        alpha: float) -> Tuple[List[Path], List[Path]]:
    """
    Split the filenames into training and testing sets.
    """
    filenames = filenames.copy()
    random.shuffle(filenames)

    n_train = int(len(filenames) * alpha)
    train_files = filenames[:n_train]
    test_files = filenames[n_train:]

    print(f"Training files: {len(train_files)}, Testing files: {len(test_files)}")
    return train_files, test_files

def create_symlinks(
        filenames: List[Path],
        input_dir: Path,
        split_dir: Path) -> None:
    """
    Create symbolic links for the training and testing images.
    """
    for filepath in filenames:
        # 生成目标文件路径
        target = input_dir / filepath.relative_to(input_dir)
        link_name = split_dir / filepath.name

        if link_name.exists():
            link_name.unlink()  # Remove existing symlink or file

        os.symlink(target, link_name)

def process(input_dir: str, patches: str, synthetic_patches: str, folders: List[str], alpha: float) -> None:
    """
    Format the dataset for training and testing.
    """
    input_dir = Path(input_dir)
    create_folders(input_dir, folders)

    patches_files, synthetic_patches_files = browse_folder(input_dir, patches, synthetic_patches)

    check_existence(patches_files)
    check_existence(synthetic_patches_files)

    patches_train, patches_test = split_train_test(patches_files, alpha)
    synthetic_patches_train, synthetic_patches_test = split_train_test(synthetic_patches_files, alpha)

    create_symlinks(patches_train, input_dir / patches, input_dir / folders[0])  # trainA
    create_symlinks(synthetic_patches_train, input_dir / synthetic_patches, input_dir / folders[1])  # trainB
    create_symlinks(patches_test, input_dir / patches, input_dir / folders[2])   # testA
    create_symlinks(synthetic_patches_test, input_dir / synthetic_patches, input_dir / folders[3])   # testB

def main():
    parser = build_argparser()
    args = parser.parse_args()

    process(args.input_dir, args.patches, args.synthetic_patches, args.folders, args.alpha)

if __name__ == "__main__":
    main()