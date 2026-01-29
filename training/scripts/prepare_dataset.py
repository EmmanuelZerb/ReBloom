#!/usr/bin/env python3
"""
ReBloom - Script de préparation du dataset

Ce script prépare le dataset pour l'entraînement:
1. Vérifie l'appariement des images sharp/blur
2. Valide les dimensions et formats
3. Divise en train/validation
4. Génère des statistiques

Usage:
    python prepare_dataset.py --sharp ./processed/sharp --blur ./processed/blur --val-split 0.1
"""

import argparse
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
from tqdm import tqdm


def get_image_info(path: Path) -> Dict:
    """Récupère les informations d'une image."""
    img = cv2.imread(str(path))
    if img is None:
        return None

    h, w, c = img.shape
    size_bytes = path.stat().st_size

    return {
        'path': path,
        'width': w,
        'height': h,
        'channels': c,
        'size_bytes': size_bytes,
        'format': path.suffix.lower(),
    }


def validate_pair(sharp_info: Dict, blur_info: Dict) -> Tuple[bool, str]:
    """Valide une paire d'images."""
    if sharp_info is None:
        return False, "Sharp image could not be read"
    if blur_info is None:
        return False, "Blur image could not be read"

    # Vérifier les dimensions
    if sharp_info['width'] != blur_info['width']:
        return False, f"Width mismatch: {sharp_info['width']} vs {blur_info['width']}"
    if sharp_info['height'] != blur_info['height']:
        return False, f"Height mismatch: {sharp_info['height']} vs {blur_info['height']}"

    # Vérifier les channels
    if sharp_info['channels'] != blur_info['channels']:
        return False, f"Channel mismatch: {sharp_info['channels']} vs {blur_info['channels']}"

    # Dimensions minimales
    if sharp_info['width'] < 64 or sharp_info['height'] < 64:
        return False, f"Image too small: {sharp_info['width']}x{sharp_info['height']}"

    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--sharp', '-s', type=str, required=True,
                        help='Directory with sharp images')
    parser.add_argument('--blur', '-b', type=str, required=True,
                        help='Directory with blurred images')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: create train/val in parent)')
    parser.add_argument('--val-split', '-v', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of moving')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only validate, do not move/copy files')

    args = parser.parse_args()

    sharp_dir = Path(args.sharp)
    blur_dir = Path(args.blur)

    if not sharp_dir.exists():
        print(f"Error: Sharp directory not found: {sharp_dir}")
        return

    if not blur_dir.exists():
        print(f"Error: Blur directory not found: {blur_dir}")
        return

    # Lister les images
    sharp_files = {f.stem: f for f in sharp_dir.iterdir() if f.is_file()}
    blur_files = {f.stem: f for f in blur_dir.iterdir() if f.is_file()}

    print(f"Found {len(sharp_files)} sharp images")
    print(f"Found {len(blur_files)} blur images")

    # Trouver les paires
    common_stems = set(sharp_files.keys()) & set(blur_files.keys())
    missing_blur = set(sharp_files.keys()) - set(blur_files.keys())
    missing_sharp = set(blur_files.keys()) - set(sharp_files.keys())

    if missing_blur:
        print(f"\nWarning: {len(missing_blur)} sharp images without blur pair")
    if missing_sharp:
        print(f"Warning: {len(missing_sharp)} blur images without sharp pair")

    print(f"\nValidating {len(common_stems)} pairs...")

    # Valider les paires
    valid_pairs = []
    invalid_pairs = []
    stats = defaultdict(list)

    for stem in tqdm(common_stems, desc="Validating"):
        sharp_path = sharp_files[stem]
        blur_path = blur_files[stem]

        sharp_info = get_image_info(sharp_path)
        blur_info = get_image_info(blur_path)

        is_valid, reason = validate_pair(sharp_info, blur_info)

        if is_valid:
            valid_pairs.append((sharp_path, blur_path))
            stats['widths'].append(sharp_info['width'])
            stats['heights'].append(sharp_info['height'])
            stats['sizes'].append(sharp_info['size_bytes'])
        else:
            invalid_pairs.append((stem, reason))

    # Afficher les statistiques
    print(f"\n{'=' * 50}")
    print("DATASET STATISTICS")
    print(f"{'=' * 50}")
    print(f"Valid pairs: {len(valid_pairs)}")
    print(f"Invalid pairs: {len(invalid_pairs)}")

    if valid_pairs:
        print(f"\nImage dimensions:")
        print(f"  Width:  min={min(stats['widths'])}, max={max(stats['widths'])}, "
              f"avg={sum(stats['widths']) / len(stats['widths']):.0f}")
        print(f"  Height: min={min(stats['heights'])}, max={max(stats['heights'])}, "
              f"avg={sum(stats['heights']) / len(stats['heights']):.0f}")
        print(f"\nTotal size: {sum(stats['sizes']) / 1e9:.2f} GB")

    if invalid_pairs:
        print(f"\nInvalid pairs (first 10):")
        for stem, reason in invalid_pairs[:10]:
            print(f"  {stem}: {reason}")

    if args.dry_run:
        print("\n[DRY RUN] No files were moved/copied")
        return

    if not valid_pairs:
        print("\nNo valid pairs found. Aborting.")
        return

    # Créer les dossiers de sortie
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = sharp_dir.parent

    train_sharp = output_dir / 'train' / 'sharp'
    train_blur = output_dir / 'train' / 'blur'
    val_sharp = output_dir / 'val' / 'sharp'
    val_blur = output_dir / 'val' / 'blur'

    for d in [train_sharp, train_blur, val_sharp, val_blur]:
        d.mkdir(parents=True, exist_ok=True)

    # Diviser train/val
    random.shuffle(valid_pairs)
    val_count = int(len(valid_pairs) * args.val_split)
    val_pairs = valid_pairs[:val_count]
    train_pairs = valid_pairs[val_count:]

    print(f"\nTrain set: {len(train_pairs)} pairs")
    print(f"Val set: {len(val_pairs)} pairs")

    # Copier/déplacer les fichiers
    file_op = shutil.copy2 if args.copy else shutil.move
    op_name = "Copying" if args.copy else "Moving"

    print(f"\n{op_name} training files...")
    for sharp_path, blur_path in tqdm(train_pairs, desc="Train"):
        file_op(sharp_path, train_sharp / sharp_path.name)
        file_op(blur_path, train_blur / blur_path.name)

    print(f"{op_name} validation files...")
    for sharp_path, blur_path in tqdm(val_pairs, desc="Val"):
        file_op(sharp_path, val_sharp / sharp_path.name)
        file_op(blur_path, val_blur / blur_path.name)

    print(f"\nDone!")
    print(f"Train: {train_sharp.parent}")
    print(f"Val: {val_sharp.parent}")


if __name__ == '__main__':
    main()
