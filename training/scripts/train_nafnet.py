#!/usr/bin/env python3
"""
ReBloom - Script d'entraînement NAFNet pour rendu naturel

Ce script fine-tune NAFNet pour le deblurring avec un rendu
naturel (sans artefacts IA).

Usage:
    python train_nafnet.py                    # Entraînement standard
    python train_nafnet.py --resume           # Reprendre
    python train_nafnet.py --test             # Tester sur quelques images
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torchvision.transforms import functional as TF

import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

# Métriques
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ============================================
# Dataset
# ============================================

class PairedImageDataset(Dataset):
    """Dataset pour paires flou/net."""

    def __init__(self, blur_dir: str, sharp_dir: str, patch_size: int = 256, augment: bool = True):
        self.blur_dir = Path(blur_dir)
        self.sharp_dir = Path(sharp_dir)
        self.patch_size = patch_size
        self.augment = augment

        # Lister les images
        self.images = sorted([
            f.stem for f in self.blur_dir.iterdir()
            if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ])

        if len(self.images) == 0:
            raise ValueError(f"Aucune image trouvée dans {blur_dir}")

        print(f"Dataset: {len(self.images)} paires d'images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        # Charger les images
        blur_path = list(self.blur_dir.glob(f"{name}.*"))[0]
        sharp_path = list(self.sharp_dir.glob(f"{name}.*"))[0]

        blur = Image.open(blur_path).convert('RGB')
        sharp = Image.open(sharp_path).convert('RGB')

        # Crop aléatoire
        i, j, h, w = T.RandomCrop.get_params(blur, (self.patch_size, self.patch_size))
        blur = TF.crop(blur, i, j, h, w)
        sharp = TF.crop(sharp, i, j, h, w)

        # Augmentation
        if self.augment:
            if torch.rand(1) > 0.5:
                blur = TF.hflip(blur)
                sharp = TF.hflip(sharp)
            if torch.rand(1) > 0.5:
                blur = TF.vflip(blur)
                sharp = TF.vflip(sharp)

        # To tensor
        blur = TF.to_tensor(blur)
        sharp = TF.to_tensor(sharp)

        return blur, sharp


# ============================================
# NAFNet Simplifié
# ============================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        return self.norm(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Bloc NAFNet simplifié."""

    def __init__(self, channels):
        super().__init__()

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels, channels * 2, 3, padding=1, groups=channels)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(channels, channels, 1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Spatial attention
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = self.conv3(y)
        x = x + y * self.beta

        # Channel attention
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        x = x + y * self.gamma

        return x


class NAFNetSimple(nn.Module):
    """NAFNet simplifié pour le deblurring."""

    def __init__(self, in_channels=3, width=32, num_blocks=16):
        super().__init__()

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)

        self.body = nn.Sequential(*[
            NAFBlock(width) for _ in range(num_blocks)
        ])

        self.outro = nn.Conv2d(width, in_channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.intro(x)
        x = self.body(x)
        x = self.outro(x)
        return x + residual  # Connexion résiduelle globale


# ============================================
# Losses pour rendu naturel
# ============================================

class CharbonnierLoss(nn.Module):
    """Charbonnier loss - plus douce que L1, meilleure pour les détails."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class EdgeLoss(nn.Module):
    """Loss sur les contours - préserve les bords naturels."""

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        # Edges de la prédiction
        pred_x = torch.nn.functional.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_y = torch.nn.functional.conv2d(pred, self.sobel_y, padding=1, groups=3)

        # Edges de la cible
        target_x = torch.nn.functional.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_y = torch.nn.functional.conv2d(target, self.sobel_y, padding=1, groups=3)

        return nn.functional.l1_loss(pred_x, target_x) + nn.functional.l1_loss(pred_y, target_y)


# ============================================
# Training Loop
# ============================================

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Modèle
        self.model = NAFNetSimple(
            width=config.get('width', 32),
            num_blocks=config.get('num_blocks', 16)
        ).to(self.device)

        # Charger poids pré-entraînés si disponibles
        pretrained_path = config.get('pretrained_path')
        if pretrained_path and Path(pretrained_path).exists():
            print(f"Chargement des poids: {pretrained_path}")
            state = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)

        # Losses pour rendu naturel
        self.l1_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss().to(self.device)

        # Poids des losses
        self.l1_weight = config.get('l1_weight', 1.0)
        self.edge_weight = config.get('edge_weight', 0.1)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.9)
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=1e-7
        )

        # Tensorboard
        self.writer = SummaryWriter(config.get('log_dir', 'runs/natural_deblur'))

        # Meilleur score
        self.best_psnr = 0

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for blur, sharp in pbar:
            blur = blur.to(self.device)
            sharp = sharp.to(self.device)

            # Forward
            pred = self.model(blur)

            # Losses
            l1 = self.l1_loss(pred, sharp)
            edge = self.edge_loss(pred, sharp)
            loss = self.l1_weight * l1 + self.edge_weight * edge

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'l1': l1.item(), 'edge': edge.item()})

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0

        for blur, sharp in dataloader:
            blur = blur.to(self.device)
            sharp = sharp.to(self.device)

            pred = self.model(blur)

            # Calculer PSNR/SSIM
            for i in range(pred.shape[0]):
                p = pred[i].cpu().numpy().transpose(1, 2, 0)
                s = sharp[i].cpu().numpy().transpose(1, 2, 0)

                p = np.clip(p, 0, 1)
                s = np.clip(s, 0, 1)

                total_psnr += psnr(s, p, data_range=1.0)
                total_ssim += ssim(s, p, data_range=1.0, channel_axis=2)

        n = len(dataloader.dataset)
        return total_psnr / n, total_ssim / n

    def save_checkpoint(self, epoch, psnr_val, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'psnr': psnr_val,
        }, path)

    def train(self, train_loader, val_loader, epochs):
        print(f"\nDémarrage de l'entraînement: {epochs} epochs")
        print(f"L1 weight: {self.l1_weight}, Edge weight: {self.edge_weight}")
        print("=" * 50)

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            psnr_val, ssim_val = self.validate(val_loader)

            # Scheduler
            self.scheduler.step()

            # Log
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Metrics/PSNR', psnr_val, epoch)
            self.writer.add_scalar('Metrics/SSIM', ssim_val, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            print(f"Epoch {epoch}: Loss={train_loss:.4f}, PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}")

            # Save best
            if psnr_val > self.best_psnr:
                self.best_psnr = psnr_val
                self.save_checkpoint(epoch, psnr_val, 'checkpoints/best_model.pth')
                print(f"  → Nouveau meilleur modèle! PSNR: {psnr_val:.2f}dB")

            # Save periodic
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, psnr_val, f'checkpoints/epoch_{epoch}.pth')

        print("\nEntraînement terminé!")
        print(f"Meilleur PSNR: {self.best_psnr:.2f}dB")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train NAFNet for natural deblurring')
    parser.add_argument('--config', type=str, default='configs/natural_deblur_config.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, help='Path to checkpoint')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    args = parser.parse_args()

    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'width': 32,
        'num_blocks': 16,
        'patch_size': 256,
        'l1_weight': 1.0,
        'edge_weight': 0.1,  # Préserve les contours naturels
        'pretrained_path': 'pretrained_models/NAFNet-GoPro-width64.pth',
        'log_dir': f'runs/natural_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    }

    # Mode test
    if args.test:
        config['epochs'] = 2
        print("Mode test: 2 epochs seulement")

    # Créer dossiers
    Path('checkpoints').mkdir(exist_ok=True)

    # Datasets
    base_path = Path(__file__).parent.parent

    train_dataset = PairedImageDataset(
        blur_dir=base_path / 'datasets/processed/blur',
        sharp_dir=base_path / 'datasets/processed/sharp',
        patch_size=config['patch_size'],
        augment=True
    )

    val_dataset = PairedImageDataset(
        blur_dir=base_path / 'datasets/validation/blur',
        sharp_dir=base_path / 'datasets/validation/sharp',
        patch_size=config['patch_size'],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Trainer
    trainer = Trainer(config)

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Reprise depuis epoch {checkpoint['epoch']}")

    # Train
    trainer.train(train_loader, val_loader, config['epochs'])


if __name__ == '__main__':
    main()
