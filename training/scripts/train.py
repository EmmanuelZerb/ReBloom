#!/usr/bin/env python3
"""
ReBloom - Script d'entraînement

Script principal pour lancer le fine-tuning du modèle Real-ESRGAN.

Usage:
    python train.py                           # Entraînement standard
    python train.py --resume                  # Reprendre l'entraînement
    python train.py --config custom.yaml      # Config personnalisée

Prérequis:
    - GPU NVIDIA avec CUDA
    - Modèle pré-entraîné dans pretrained_models/
    - Dataset préparé dans datasets/
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Ajouter le chemin pour basicsr
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Vérifie que l'environnement est correctement configuré."""
    print("=" * 50)
    print("ENVIRONMENT CHECK")
    print("=" * 50)

    # Python
    print(f"Python: {sys.version}")

    # PyTorch
    print(f"PyTorch: {torch.__version__}")

    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Count: {torch.cuda.device_count()}")
    else:
        print("CUDA: NOT AVAILABLE")
        print("WARNING: Training without GPU will be extremely slow!")

    print("=" * 50)


def check_files(config_path: str):
    """Vérifie que tous les fichiers nécessaires existent."""
    base_path = Path(__file__).parent.parent

    # Config
    config_file = base_path / 'configs' / config_path
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        return False

    # Pretrained model
    pretrained_path = base_path / 'pretrained_models' / 'RealESRGAN_x4plus.pth'
    if not pretrained_path.exists():
        print(f"WARNING: Pretrained model not found: {pretrained_path}")
        print("Download it with:")
        print(f"  wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O {pretrained_path}")

    # Dataset
    train_sharp = base_path / 'datasets' / 'train' / 'sharp'
    train_blur = base_path / 'datasets' / 'train' / 'blur'

    if not train_sharp.exists() or not train_blur.exists():
        print("WARNING: Training dataset not found")
        print(f"Expected: {train_sharp}")
        print(f"Expected: {train_blur}")
        print("Run prepare_dataset.py first")

    return True


def main():
    parser = argparse.ArgumentParser(description='Train ReBloom model')
    parser.add_argument('--config', '-c', type=str, default='training_config.yaml',
                        help='Config file name (in configs/)')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode (less iterations)')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')

    args = parser.parse_args()

    # Vérifier l'environnement
    check_environment()

    # Vérifier les fichiers
    if not check_files(args.config):
        sys.exit(1)

    # Importer basicsr (après vérifications)
    try:
        from basicsr.train import train_pipeline
    except ImportError:
        print("ERROR: basicsr not installed")
        print("Run: pip install basicsr")
        sys.exit(1)

    # Construire le chemin de config
    base_path = Path(__file__).parent.parent
    config_path = str(base_path / 'configs' / args.config)

    # Options de ligne de commande pour basicsr
    opt = {
        'config': config_path,
        'launcher': 'none',
        'auto_resume': args.resume,
        'is_train': True,
        'local_rank': args.local_rank,
    }

    # Debug mode
    if args.debug:
        print("\n[DEBUG MODE] Reducing iterations for testing")
        opt['force_yml'] = {
            'train:total_iter': 1000,
            'val:val_freq': 100,
            'logger:print_freq': 10,
            'logger:save_checkpoint_freq': 500,
        }

    print(f"\nStarting training with config: {args.config}")
    print("Press Ctrl+C to stop\n")

    # Lancer l'entraînement
    try:
        train_pipeline(config_path)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("You can resume with: python train.py --resume")
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        raise


if __name__ == '__main__':
    main()
