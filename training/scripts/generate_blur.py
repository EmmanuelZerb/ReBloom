#!/usr/bin/env python3
"""
ReBloom - Script de génération de flou synthétique

Ce script prend des images nettes (HD) et génère des paires flou/net
pour l'entraînement du modèle de défloutage.

Usage:
    python generate_blur.py --input ./raw --output-sharp ./processed/sharp --output-blur ./processed/blur

Types de flou supportés:
    - gaussian: Flou gaussien classique
    - motion: Flou de mouvement (bougé)
    - defocus: Flou de mise au point
    - all: Applique un type aléatoire par image
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================
# Configuration
# ============================================

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

BLUR_PARAMS = {
    'gaussian': {
        'light': {'ksize': (5, 5), 'sigma': (0.5, 1.5)},
        'medium': {'ksize': (9, 9), 'sigma': (1.5, 3.0)},
        'heavy': {'ksize': (15, 15), 'sigma': (3.0, 5.0)},
    },
    'motion': {
        'light': {'kernel_size': (5, 10), 'angle': (-45, 45)},
        'medium': {'kernel_size': (10, 20), 'angle': (-90, 90)},
        'heavy': {'kernel_size': (20, 35), 'angle': (-180, 180)},
    },
    'defocus': {
        'light': {'radius': (2, 5)},
        'medium': {'radius': (5, 10)},
        'heavy': {'radius': (10, 20)},
    },
}


# ============================================
# Blur Functions
# ============================================

def apply_gaussian_blur(image: np.ndarray, intensity: str = 'medium') -> np.ndarray:
    """Applique un flou gaussien."""
    params = BLUR_PARAMS['gaussian'][intensity]
    ksize = params['ksize']
    sigma = random.uniform(*params['sigma'])
    return cv2.GaussianBlur(image, ksize, sigma)


def apply_motion_blur(image: np.ndarray, intensity: str = 'medium') -> np.ndarray:
    """Applique un flou de mouvement."""
    params = BLUR_PARAMS['motion'][intensity]
    kernel_size = random.randint(*params['kernel_size'])
    angle = random.uniform(*params['angle'])

    # Créer le kernel de motion blur
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    # Rotation du kernel
    center = (kernel_size // 2, kernel_size // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))

    # Normaliser
    kernel = kernel / kernel.sum()

    return cv2.filter2D(image, -1, kernel)


def apply_defocus_blur(image: np.ndarray, intensity: str = 'medium') -> np.ndarray:
    """Applique un flou de mise au point (disk blur)."""
    params = BLUR_PARAMS['defocus'][intensity]
    radius = random.randint(*params['radius'])

    # Créer un kernel circulaire
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel = kernel / kernel.sum()

    return cv2.filter2D(image, -1, kernel)


def apply_random_blur(image: np.ndarray, intensity: str = 'medium') -> Tuple[np.ndarray, str]:
    """Applique un type de flou aléatoire."""
    blur_type = random.choice(['gaussian', 'motion', 'defocus'])
    blur_funcs = {
        'gaussian': apply_gaussian_blur,
        'motion': apply_motion_blur,
        'defocus': apply_defocus_blur,
    }
    return blur_funcs[blur_type](image, intensity), blur_type


# ============================================
# Image Processing
# ============================================

def add_noise(image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """Ajoute du bruit gaussien."""
    if noise_level <= 0:
        return image

    noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_jpeg_artifacts(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """Ajoute des artefacts de compression JPEG."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def resize_if_needed(image: np.ndarray, max_size: int = 2048, min_size: int = 256) -> np.ndarray:
    """Redimensionne l'image si nécessaire."""
    h, w = image.shape[:2]

    # Trop grande
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

    # Trop petite
    if min(h, w) < min_size:
        return None  # Ignorer les images trop petites

    return image


def process_single_image(
    input_path: Path,
    output_sharp: Path,
    output_blur: Path,
    blur_types: List[str],
    intensity: str,
    add_noise_flag: bool,
    add_jpeg_flag: bool,
    output_format: str,
) -> Optional[Tuple[str, str]]:
    """Traite une seule image."""
    try:
        # Charger l'image
        image = cv2.imread(str(input_path))
        if image is None:
            return None

        # Redimensionner si nécessaire
        image = resize_if_needed(image)
        if image is None:
            return None

        # Générer le nom de sortie
        stem = input_path.stem
        ext = f'.{output_format}'

        # Sauvegarder l'image nette
        sharp_path = output_sharp / f'{stem}{ext}'
        cv2.imwrite(str(sharp_path), image)

        # Appliquer le flou
        if 'all' in blur_types or 'random' in blur_types:
            blurred, blur_type = apply_random_blur(image, intensity)
        else:
            blur_type = random.choice(blur_types)
            if blur_type == 'gaussian':
                blurred = apply_gaussian_blur(image, intensity)
            elif blur_type == 'motion':
                blurred = apply_motion_blur(image, intensity)
            elif blur_type == 'defocus':
                blurred = apply_defocus_blur(image, intensity)
            else:
                blurred = apply_gaussian_blur(image, intensity)

        # Ajouter du bruit (optionnel)
        if add_noise_flag:
            noise_level = random.uniform(0.005, 0.02)
            blurred = add_noise(blurred, noise_level)

        # Ajouter des artefacts JPEG (optionnel)
        if add_jpeg_flag:
            quality = random.randint(30, 70)
            blurred = add_jpeg_artifacts(blurred, quality)

        # Sauvegarder l'image floue
        blur_path = output_blur / f'{stem}{ext}'
        cv2.imwrite(str(blur_path), blurred)

        return str(sharp_path), str(blur_path)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Generate blur/sharp image pairs for training')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory with HD images')
    parser.add_argument('--output-sharp', '-os', type=str, required=True,
                        help='Output directory for sharp images')
    parser.add_argument('--output-blur', '-ob', type=str, required=True,
                        help='Output directory for blurred images')
    parser.add_argument('--blur-types', '-b', type=str, default='all',
                        help='Blur types: gaussian,motion,defocus,all (default: all)')
    parser.add_argument('--intensity', '-n', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='Blur intensity (default: medium)')
    parser.add_argument('--add-noise', action='store_true',
                        help='Add random noise to blurred images')
    parser.add_argument('--add-jpeg', action='store_true',
                        help='Add JPEG compression artifacts')
    parser.add_argument('--format', '-f', type=str, default='png',
                        choices=['png', 'jpg', 'webp'],
                        help='Output format (default: png)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of parallel workers (default: 4)')

    args = parser.parse_args()

    # Créer les dossiers de sortie
    input_dir = Path(args.input)
    output_sharp = Path(args.output_sharp)
    output_blur = Path(args.output_blur)

    output_sharp.mkdir(parents=True, exist_ok=True)
    output_blur.mkdir(parents=True, exist_ok=True)

    # Lister les images
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Blur types: {args.blur_types}")
    print(f"Intensity: {args.intensity}")

    # Parser les types de flou
    blur_types = [b.strip() for b in args.blur_types.split(',')]

    # Traiter les images en parallèle
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_image,
                img_path,
                output_sharp,
                output_blur,
                blur_types,
                args.intensity,
                args.add_noise,
                args.add_jpeg,
                args.format,
            ): img_path
            for img_path in image_files
        }

        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)

    print(f"\nDone! Processed {successful} images, {failed} failed")
    print(f"Sharp images: {output_sharp}")
    print(f"Blurred images: {output_blur}")


if __name__ == '__main__':
    main()
