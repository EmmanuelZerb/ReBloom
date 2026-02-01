#!/bin/bash
# ==============================================
# ReBloom - Entraînement NAFNet sur RunPod
# ==============================================
# 
# Usage: Copie ce script sur RunPod et exécute-le
# Temps estimé: ~4-6h sur RTX 3090/A100
#
# ==============================================

set -e

echo "🚀 ReBloom NAFNet Training Setup"
echo "================================="

# 1. Installation des dépendances
echo ""
echo "📦 Installation des dépendances..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow tqdm tensorboard scikit-image gdown pyyaml

# 2. Téléchargement du dataset GoPro
echo ""
echo "📥 Téléchargement du dataset GoPro (~6GB)..."
cd /workspace
mkdir -p datasets
cd datasets

# GoPro dataset (Google Drive)
gdown "1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2" -O gopro.zip

echo "📂 Extraction du dataset..."
unzip -q gopro.zip
rm gopro.zip

# Organiser les fichiers
mkdir -p train/blur train/sharp val/blur val/sharp

# GoPro structure: GOPRO_Large/train/GOPR*/blur et sharp
mv GOPRO_Large/train/*/blur/* train/blur/ 2>/dev/null || true
mv GOPRO_Large/train/*/sharp/* train/sharp/ 2>/dev/null || true
mv GOPRO_Large/test/*/blur/* val/blur/ 2>/dev/null || true
mv GOPRO_Large/test/*/sharp/* val/sharp/ 2>/dev/null || true

echo "✅ Dataset prêt: $(ls train/blur | wc -l) images train, $(ls val/blur | wc -l) images val"

# 3. Téléchargement du modèle pré-entraîné NAFNet
echo ""
echo "📥 Téléchargement NAFNet pré-entraîné..."
cd /workspace
mkdir -p pretrained_models
cd pretrained_models
wget -q https://github.com/megvii-research/NAFNet/releases/download/v0.0.1/NAFNet-GoPro-width64.pth
echo "✅ NAFNet-GoPro-width64.pth téléchargé"

# 4. Créer le script d'entraînement
echo ""
echo "📝 Création du script d'entraînement..."
cd /workspace

cat > train_nafnet.py << 'TRAINSCRIPT'
#!/usr/bin/env python3
"""
ReBloom NAFNet Training - Professional Deblurring
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
from tqdm import tqdm

# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ============================================
# Dataset
# ============================================

class GoPRODataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, patch_size=256, augment=True):
        self.blur_dir = Path(blur_dir)
        self.sharp_dir = Path(sharp_dir)
        self.patch_size = patch_size
        self.augment = augment
        
        self.images = sorted([f.name for f in self.blur_dir.iterdir() 
                             if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
        print(f"Dataset: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        name = self.images[idx]
        
        blur = Image.open(self.blur_dir / name).convert('RGB')
        sharp = Image.open(self.sharp_dir / name).convert('RGB')
        
        # Random crop
        w, h = blur.size
        if w >= self.patch_size and h >= self.patch_size:
            x = torch.randint(0, w - self.patch_size + 1, (1,)).item()
            y = torch.randint(0, h - self.patch_size + 1, (1,)).item()
            blur = blur.crop((x, y, x + self.patch_size, y + self.patch_size))
            sharp = sharp.crop((x, y, x + self.patch_size, y + self.patch_size))
        else:
            blur = blur.resize((self.patch_size, self.patch_size))
            sharp = sharp.resize((self.patch_size, self.patch_size))
        
        # Augmentation
        if self.augment:
            if torch.rand(1) > 0.5:
                blur = TF.hflip(blur)
                sharp = TF.hflip(sharp)
            if torch.rand(1) > 0.5:
                blur = TF.vflip(blur)
                sharp = TF.vflip(sharp)
            if torch.rand(1) > 0.5:
                k = torch.randint(1, 4, (1,)).item()
                blur = TF.rotate(blur, k * 90)
                sharp = TF.rotate(sharp, k * 90)
        
        return TF.to_tensor(blur), TF.to_tensor(sharp)


# ============================================
# NAFNet Architecture (Full Version)
# ============================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=12,
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ============================================
# Losses
# ============================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.mm(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)

    def conv_gauss(self, img):
        n_channels = img.shape[1]
        img = nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        kernel = self.kernel.to(img.device)
        return nn.functional.conv2d(img, kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = torch.mean(torch.abs(self.laplacian_kernel(x) - self.laplacian_kernel(y)))
        return loss


# ============================================
# Trainer
# ============================================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Model - Full NAFNet
        self.model = NAFNet(
            width=config['width'],
            middle_blk_num=config['middle_blk_num'],
            enc_blk_nums=config['enc_blk_nums'],
            dec_blk_nums=config['dec_blk_nums']
        ).to(self.device)
        
        # Count parameters
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {params/1e6:.2f}M")
        
        # Load pretrained
        if config.get('pretrained') and os.path.exists(config['pretrained']):
            print(f"Loading pretrained: {config['pretrained']}")
            state = torch.load(config['pretrained'], map_location=self.device)
            if 'params' in state:
                state = state['params']
            self.model.load_state_dict(state, strict=False)
            print("✅ Pretrained weights loaded")
        
        # Losses
        self.charbonnier = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.9)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-7
        )
        
        # Tensorboard
        self.writer = SummaryWriter(f"runs/nafnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.best_psnr = 0
        self.start_epoch = 0
    
    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for blur, sharp in pbar:
            blur = blur.to(self.device)
            sharp = sharp.to(self.device)
            
            # Forward
            pred = self.model(blur)
            
            # Loss
            loss_char = self.charbonnier(pred, sharp)
            loss_edge = self.edge_loss(pred, sharp)
            loss = loss_char + 0.05 * loss_edge
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(loader)
    
    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        count = 0
        
        for blur, sharp in tqdm(loader, desc="Validation"):
            blur = blur.to(self.device)
            sharp = sharp.to(self.device)
            
            pred = self.model(blur)
            
            for i in range(pred.shape[0]):
                p = pred[i].cpu().numpy().transpose(1, 2, 0)
                s = sharp[i].cpu().numpy().transpose(1, 2, 0)
                p = np.clip(p, 0, 1)
                s = np.clip(s, 0, 1)
                
                total_psnr += psnr(s, p, data_range=1.0)
                total_ssim += ssim(s, p, data_range=1.0, channel_axis=2)
                count += 1
        
        return total_psnr / count, total_ssim / count
    
    def save(self, path, epoch, psnr_val):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'psnr': psnr_val,
        }, path)
    
    def train(self, train_loader, val_loader):
        epochs = self.config['epochs']
        print(f"\n🚀 Starting training: {epochs} epochs")
        print("=" * 50)
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            # Train
            loss = self.train_epoch(train_loader, epoch)
            
            # Validate every 5 epochs
            if epoch % 5 == 0 or epoch == epochs:
                psnr_val, ssim_val = self.validate(val_loader)
                print(f"Epoch {epoch}: Loss={loss:.4f}, PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.4f}")
                
                # Log
                self.writer.add_scalar('Loss/train', loss, epoch)
                self.writer.add_scalar('PSNR', psnr_val, epoch)
                self.writer.add_scalar('SSIM', ssim_val, epoch)
                
                # Save best
                if psnr_val > self.best_psnr:
                    self.best_psnr = psnr_val
                    self.save('best_nafnet.pth', epoch, psnr_val)
                    print(f"  ✅ New best model! PSNR: {psnr_val:.2f}dB")
            else:
                print(f"Epoch {epoch}: Loss={loss:.4f}")
            
            # Scheduler
            self.scheduler.step()
            
            # Save checkpoint every 20 epochs
            if epoch % 20 == 0:
                self.save(f'checkpoint_epoch_{epoch}.pth', epoch, self.best_psnr)
        
        print(f"\n✅ Training complete! Best PSNR: {self.best_psnr:.2f}dB")
        return self.best_psnr


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--pretrained', type=str, default='pretrained_models/NAFNet-GoPro-width64.pth')
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'width': args.width,
        'middle_blk_num': 12,
        'enc_blk_nums': [2, 2, 4, 8],
        'dec_blk_nums': [2, 2, 2, 2],
        'pretrained': args.pretrained,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Datasets
    train_dataset = GoPRODataset(
        'datasets/train/blur',
        'datasets/train/sharp',
        patch_size=256,
        augment=True
    )
    
    val_dataset = GoPRODataset(
        'datasets/val/blur',
        'datasets/val/sharp',
        patch_size=256,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)
    
    print("\n📦 Exporting model for deployment...")
    # Export clean state dict for Replicate
    state = torch.load('best_nafnet.pth')
    torch.save(state['model_state_dict'], 'rebloom-nafnet-v2.pth')
    print("✅ Model saved as rebloom-nafnet-v2.pth")


if __name__ == '__main__':
    main()
TRAINSCRIPT

chmod +x train_nafnet.py

# 5. Lancer l'entraînement
echo ""
echo "🏋️ Lancement de l'entraînement NAFNet..."
echo "Temps estimé: ~4-6h sur RTX 3090"
echo ""

python train_nafnet.py --epochs 100 --batch-size 8 --lr 1e-4

echo ""
echo "✅ Entraînement terminé!"
echo ""
echo "📁 Fichiers générés:"
echo "   - best_nafnet.pth (checkpoint complet)"
echo "   - rebloom-nafnet-v2.pth (pour Replicate)"
echo ""
echo "Prochaine étape: Déployer sur Replicate"
