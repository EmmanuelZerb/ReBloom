# ReBloom Sharp v1 - Documentation du Modèle

## Résumé

| Propriété | Valeur |
|-----------|--------|
| **Nom** | rebloom-sharp-v1 |
| **Architecture** | SimpleUNet |
| **Tâche** | Image Deblurring (défloutage) |
| **Date d'entraînement** | 30 janvier 2026 |
| **Fichier** | `rebloom-sharp-v1.pth` |

---

## Architecture

### SimpleUNet

Un réseau encodeur-décodeur avec connexions résiduelles (skip connections) :

```
Input (H×W×3)
    │
    ▼
┌─────────────────┐
│  Encoder 1      │ Conv(3→64) + ReLU + Conv(64→64) + ReLU
│  64 channels    │────────────────────────────────────────┐
└────────┬────────┘                                        │
         │ Downsample (stride=2)                           │
         ▼                                                 │
┌─────────────────┐                                        │
│  Encoder 2      │ Conv(64→128) + ReLU + Conv(128→128)    │
│  128 channels   │─────────────────────────────────┐      │
└────────┬────────┘                                 │      │
         │ Downsample (stride=2)                    │      │
         ▼                                          │      │
┌─────────────────┐                                 │      │
│  Encoder 3      │ Conv(128→256) + ReLU            │      │
│  256 channels   │                                 │      │
└────────┬────────┘                                 │      │
         │                                          │      │
         ▼                                          │      │
┌─────────────────┐                                 │      │
│  Middle         │ Conv(256→256) + ReLU ×2         │      │
│  256 channels   │                                 │      │
└────────┬────────┘                                 │      │
         │                                          │      │
         ▼                                          │      │
┌─────────────────┐                                 │      │
│  Decoder 3      │ ConvTranspose(256→128)          │      │
│  128 channels   │◄────────────────────────────────┘      │
└────────┬────────┘ (concat with Encoder 2)                │
         │                                                 │
         ▼                                                 │
┌─────────────────┐                                        │
│  Decoder 2      │ ConvTranspose(256→64)                  │
│  64 channels    │◄───────────────────────────────────────┘
└────────┬────────┘ (concat with Encoder 1)
         │
         ▼
┌─────────────────┐
│  Decoder 1      │ Conv(128→64) + Conv(64→3)
│  3 channels     │
└────────┬────────┘
         │
         ▼
    Output + Input (résidu)
         │
         ▼
   Final Output (H×W×3)
```

### Caractéristiques

- **Skip connections** : Préservent les détails haute fréquence
- **Connexion résiduelle finale** : `output = model(x) + x`
- **Activation** : ReLU
- **Pas de normalisation** : Pas de BatchNorm (évite les artefacts)

---

## Dataset

### Source

- **DIV2K** : 800 images haute résolution (2K)
- **Téléchargement** : http://data.vision.ee.ethz.ch/cvl/DIV2K/

### Préparation

```
datasets/
├── raw/                    # 800 images originales DIV2K
├── processed/
│   ├── sharp/              # 720 images (90%) - Ground truth
│   └── blur/               # 720 images (90%) - Input flou
└── validation/
    ├── sharp/              # 80 images (10%)
    └── blur/               # 80 images (10%)
```

### Génération du flou

Trois types de flou appliqués aléatoirement :

| Type | Paramètres | Description |
|------|------------|-------------|
| **Gaussian** | kernel 5-15, sigma 1-4 | Flou général |
| **Motion** | kernel 10-30, angle ±45° | Flou de bougé |
| **Defocus** | radius 3-10 | Flou de mise au point |

```python
def apply_varied_blur(image):
    blur_type = random.choice(['gaussian', 'motion', 'defocus'])
    
    if blur_type == 'gaussian':
        ksize = random.choice([5, 7, 9, 11, 15])
        return cv2.GaussianBlur(image, (ksize, ksize), random.uniform(1, 4))
    
    elif blur_type == 'motion':
        size = random.randint(10, 30)
        kernel = np.zeros((size, size))
        kernel[size//2, :] = 1.0 / size
        angle = random.uniform(-45, 45)
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        return cv2.filter2D(image, -1, kernel)
    
    else:  # defocus
        radius = random.randint(3, 10)
        kernel = np.zeros((radius*2+1, radius*2+1), np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1, -1)
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel)
```

---

## Entraînement

### Configuration

| Paramètre | Valeur |
|-----------|--------|
| **GPU** | NVIDIA RTX 4090 (24GB VRAM) |
| **Plateforme** | RunPod |
| **Epochs** | ~100 |
| **Batch size** | 8 |
| **Patch size** | 256×256 |
| **Learning rate** | 1e-4 |
| **Optimizer** | AdamW |
| **Loss** | L1 Loss (MAE) |

### Data Augmentation

- Random crop (256×256)
- Horizontal flip (50%)

### Code d'entraînement

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

class DeblurDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, patch_size=256):
        self.blur_files = sorted(Path(blur_dir).glob('*.png'))
        self.sharp_dir = Path(sharp_dir)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        blur = Image.open(self.blur_files[idx]).convert('RGB')
        sharp = Image.open(self.sharp_dir / self.blur_files[idx].name).convert('RGB')
        
        # Random crop
        i = torch.randint(0, blur.size[1] - self.patch_size, (1,)).item()
        j = torch.randint(0, blur.size[0] - self.patch_size, (1,)).item()
        blur = TF.crop(blur, i, j, self.patch_size, self.patch_size)
        sharp = TF.crop(sharp, i, j, self.patch_size, self.patch_size)
        
        # Random flip
        if torch.rand(1) > 0.5:
            blur = TF.hflip(blur)
            sharp = TF.hflip(sharp)
        
        return TF.to_tensor(blur), TF.to_tensor(sharp)

# Training loop
device = torch.device('cuda')
model = SimpleUNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

for epoch in range(100):
    model.train()
    for blur, sharp in train_loader:
        blur, sharp = blur.to(device), sharp.to(device)
        pred = model(blur)
        loss = criterion(pred, sharp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Résultats

| Métrique | Valeur |
|----------|--------|
| **Train Loss finale** | ~0.029 |
| **Val Loss finale** | ~0.027 |
| **Meilleur checkpoint** | Epoch 35 (Val Loss = 0.0273) |

---

## Utilisation

### Chargement du modèle

```python
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Définir l'architecture
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), 
                                   nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), 
                                   nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), 
                                   nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.middle = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), 
                                     nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(), 
                                   nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 2, stride=2), nn.ReLU(), 
                                   nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), 
                                   nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.middle(e3)
        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        return d1 + x  # Résidu

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleUNet().to(device)
model.load_state_dict(torch.load('rebloom-sharp-v1.pth', map_location=device))
model.eval()
```

### Inférence

```python
def deblur_image(model, image_path, device):
    # Charger l'image
    img = Image.open(image_path).convert('RGB')
    
    # Redimensionner à une taille multiple de 4
    w, h = img.size
    new_w = (w // 4) * 4
    new_h = (h // 4) * 4
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Convertir en tensor
    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
    
    # Inférence
    with torch.no_grad():
        output = model(tensor)
    
    # Convertir en image
    output_img = TF.to_pil_image(output.squeeze(0).cpu().clamp(0, 1))
    
    return output_img

# Exemple
result = deblur_image(model, 'image_floue.jpg', device)
result.save('image_nette.png')
```

---

## Limitations

1. **Taille d'image** : Les dimensions doivent être multiples de 4
2. **Flou extrême** : Performances réduites sur flou très intense
3. **Artefacts JPEG** : Non entraîné spécifiquement pour la compression
4. **Bruit** : Denoising limité (pas l'objectif principal)

---

## Améliorations futures (v2)

- [ ] Architecture NAFNet (state-of-the-art)
- [ ] Dataset plus large (Flickr2K + DIV2K)
- [ ] Perceptual Loss pour meilleure qualité visuelle
- [ ] Support des artefacts JPEG
- [ ] Entraînement sur vraies photos floues

---

## Licence

Modèle propriétaire - Usage exclusif pour ReBloom SaaS.

---

*Créé le 30 janvier 2026*
