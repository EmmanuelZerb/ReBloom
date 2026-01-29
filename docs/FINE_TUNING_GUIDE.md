# Guide Complet de Fine-Tuning pour ReBloom

Ce guide vous accompagne étape par étape pour créer votre propre modèle de défloutage d'images fine-tuné.

## Table des Matières

1. [Création du Dataset](#1-création-du-dataset)
2. [Choix du Modèle](#2-choix-du-modèle)
3. [Process de Fine-Tuning](#3-process-de-fine-tuning)
4. [Entraînement Cloud](#4-entraînement-cloud)
5. [Déploiement](#5-déploiement)

---

## 1. Création du Dataset

### Structure des Dossiers

```
training/
├── datasets/
│   ├── raw/                    # Images sources HD originales
│   ├── processed/
│   │   ├── sharp/              # Images nettes (ground truth)
│   │   └── blur/               # Images floutées (input)
│   └── validation/             # 10-15% du dataset pour validation
```

### Convention de Nommage

```
sharp/0001.png  <-->  blur/0001.png
sharp/0002.png  <-->  blur/0002.png
...
```

Les paires doivent avoir **exactement le même nom** pour l'appariement automatique.

### Combien d'Images ?

| Objectif | Minimum | Recommandé | Optimal |
|----------|---------|------------|---------|
| Test/POC | 100 | 500 | - |
| Usage général | 1,000 | 5,000 | 10,000+ |
| Domaine spécifique | 500 | 2,000 | 5,000+ |

**Conseil** : La qualité prime sur la quantité. 1,000 images de haute qualité > 10,000 images médiocres.

### Sources de Datasets Existants

#### Datasets publics recommandés :

| Dataset | Images | Résolution | Lien |
|---------|--------|------------|------|
| **DIV2K** | 1,000 | 2K | [Site officiel](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| **Flickr2K** | 2,650 | 2K | [GitHub](https://github.com/limbee/NTIRE2017) |
| **OST** | 10,324 | Variable | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| **FFHQ** (visages) | 70,000 | 1024x1024 | [GitHub](https://github.com/NVlabs/ffhq-dataset) |

#### Téléchargement DIV2K :

```bash
# DIV2K Training (800 images)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip -d training/datasets/raw/

# DIV2K Validation (100 images)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_valid_HR.zip -d training/datasets/validation/
```

### Génération du Flou Synthétique

Le script `generate_blur.py` dans `training/scripts/` génère des images floues réalistes :

```bash
cd training
python scripts/generate_blur.py \
  --input datasets/raw \
  --output-sharp datasets/processed/sharp \
  --output-blur datasets/processed/blur \
  --blur-types gaussian,motion,defocus \
  --intensity medium
```

#### Types de flou supportés :

1. **Gaussian Blur** : Flou général, le plus commun
2. **Motion Blur** : Flou de mouvement (bougé)
3. **Defocus Blur** : Flou de mise au point
4. **Compression Artifacts** : Artefacts JPEG (optionnel)

### Data Augmentation

Augmentez votre dataset avec des transformations :

```python
# Dans le script de génération
augmentations = [
    'horizontal_flip',      # Miroir horizontal
    'vertical_flip',        # Miroir vertical
    'rotation_90',          # Rotations 90°
    'color_jitter',         # Variation couleurs légère
    'random_crop',          # Crops aléatoires
]
```

**Important** : Appliquez les MÊMES augmentations aux paires sharp/blur.

---

## 2. Choix du Modèle

### Comparatif des Architectures

| Modèle | Performance | Vitesse | VRAM | Recommandé pour |
|--------|-------------|---------|------|-----------------|
| **Real-ESRGAN** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 4-6 GB | Usage général, photos |
| **SwinIR** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 8-12 GB | Qualité maximale |
| **NAFNet** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 4-6 GB | Équilibre qualité/vitesse |
| **BSRGAN** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 4-6 GB | Dégradations réelles |

### Ma Recommandation : Real-ESRGAN

**Pourquoi ?**

1. **Écosystème mature** : Documentation abondante, communauté active
2. **Pré-entraîné robuste** : Le modèle de base gère déjà bien la plupart des cas
3. **Fine-tuning rapide** : Converge vite avec peu de données
4. **Déploiement facile** : Support natif Replicate et HuggingFace
5. **Bon équilibre** : Qualité/vitesse/ressources

### Architecture Real-ESRGAN

```
Input (LR) → RRDB Network → Pixel Shuffle → Output (HR)
              ↓
         23 RRDB blocks
         (Residual in Residual Dense Block)
```

Pour le fine-tuning, on gèle généralement les premières couches et on entraîne les dernières.

---

## 3. Process de Fine-Tuning

### Prérequis

```bash
# Python 3.8+
python --version

# CUDA (pour GPU)
nvidia-smi

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate   # Windows

# Installer dépendances
pip install -r training/requirements.txt
```

### Requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
basicsr>=1.4.2
realesrgan>=0.3.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.65.0
tensorboard>=2.14.0
wandb>=0.15.0  # Optionnel, pour tracking
```

### Configuration d'Entraînement

Le fichier `training/configs/training_config.yaml` :

```yaml
# training_config.yaml
name: rebloom_finetune_v1
model_type: RealESRGANModel

# Dataset
datasets:
  train:
    name: ReBloomTrain
    type: PairedImageDataset
    dataroot_gt: ./datasets/processed/sharp
    dataroot_lq: ./datasets/processed/blur
    filename_tmpl: '{}'
    io_backend:
      type: disk

    gt_size: 256  # Patch size pour l'entraînement
    use_hflip: true
    use_rot: true

    # Dataloader
    num_worker_per_gpu: 4
    batch_size_per_gpu: 8
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    name: ReBloomVal
    type: PairedImageDataset
    dataroot_gt: ./datasets/validation/sharp
    dataroot_lq: ./datasets/validation/blur
    io_backend:
      type: disk

# Network
network_g:
  type: RRDBNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
  scale: 4

# Pretrained (à télécharger)
path:
  pretrain_network_g: ./pretrained_models/RealESRGAN_x4plus.pth
  strict_load_g: false
  resume_state: ~

# Training
train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 1e-4  # Learning rate (réduire pour fine-tuning)
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [50000, 100000, 150000]
    gamma: 0.5

  total_iter: 200000
  warmup_iter: -1

  # Losses
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv5_4': 1
    vgg_type: vgg19
    use_input_norm: true
    perceptual_weight: 1.0
    style_weight: 0
    range_norm: false
    criterion: l1

  gan_opt:
    type: GANLoss
    gan_type: vanilla
    real_label_val: 1.0
    fake_label_val: 0.0
    loss_weight: !!float 1e-1

# Validation
val:
  val_freq: !!float 5e3
  save_img: true
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: true
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: true

# Logging
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: rebloom
    resume_id: ~

# Dist
dist_params:
  backend: nccl
  port: 29500
```

### Script d'Entraînement

```python
# training/scripts/train.py
import os
import torch
from basicsr.train import train_pipeline

def main():
    # Configuration
    config_path = 'configs/training_config.yaml'

    # Vérifier GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, training will be slow!")

    # Lancer l'entraînement
    train_pipeline(config_path)

if __name__ == '__main__':
    main()
```

### Lancer l'Entraînement

```bash
cd training

# Single GPU
python scripts/train.py

# Multi-GPU (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=4321 \
  scripts/train.py --launcher pytorch

# Reprendre un entraînement
python scripts/train.py --resume experiments/rebloom_finetune_v1/training_states/latest.state
```

### Monitoring

```bash
# TensorBoard
tensorboard --logdir experiments/rebloom_finetune_v1/tb_logger

# Ou Weights & Biases (si configuré)
wandb login
# Les métriques apparaîtront sur wandb.ai
```

### Métriques à Surveiller

| Métrique | Bon signe | Mauvais signe |
|----------|-----------|---------------|
| **L1 Loss** | Descend régulièrement | Stagne ou remonte |
| **Perceptual Loss** | Descend lentement | Oscille fortement |
| **PSNR** | Monte (>28 dB = bon) | Stagne sous 25 dB |
| **SSIM** | Monte (>0.85 = bon) | Stagne sous 0.8 |

### Checkpoints et Early Stopping

Les checkpoints sont sauvegardés automatiquement dans :
```
experiments/rebloom_finetune_v1/
├── models/
│   ├── net_g_5000.pth
│   ├── net_g_10000.pth
│   └── ...
├── training_states/
│   └── latest.state
└── tb_logger/
```

**Early stopping manuel** : Si PSNR stagne pendant 20k+ itérations, arrêtez l'entraînement.

---

## 4. Entraînement Cloud

### Comparatif des Plateformes

| Plateforme | GPU Dispo | Prix/h (A100) | Facilité | Recommandé |
|------------|-----------|---------------|----------|------------|
| **RunPod** | A100, RTX 4090 | ~$1.50 | ⭐⭐⭐⭐ | ✅ Meilleur rapport qualité/prix |
| **Vast.ai** | Variable | ~$1.00 | ⭐⭐⭐ | ✅ Le moins cher |
| **Lambda Labs** | A100, H100 | ~$2.00 | ⭐⭐⭐⭐⭐ | Pour production |
| **Google Colab Pro** | T4, A100 | $10/mois | ⭐⭐⭐⭐ | Pour tests |

### Estimation Temps/Coût

| Dataset | GPU | Itérations | Temps | Coût estimé |
|---------|-----|------------|-------|-------------|
| 1,000 images | RTX 4090 | 100k | ~8h | ~$12 |
| 5,000 images | A100 | 200k | ~12h | ~$24 |
| 10,000 images | A100 | 300k | ~20h | ~$40 |

### Setup RunPod

1. **Créer un compte** : [runpod.io](https://runpod.io)

2. **Lancer un pod** :
   - Template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0`
   - GPU: RTX 4090 ou A100
   - Volume: 50GB minimum

3. **Transférer les données** :
```bash
# Depuis votre machine locale
rsync -avz --progress training/ runpod:/workspace/training/

# Ou via S3/GCS
aws s3 sync training/ s3://your-bucket/training/
```

4. **Sur le pod** :
```bash
cd /workspace/training
pip install -r requirements.txt

# Télécharger le modèle pré-entraîné
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -O pretrained_models/RealESRGAN_x4plus.pth

# Lancer l'entraînement
python scripts/train.py
```

5. **Récupérer le modèle** :
```bash
# Depuis le pod vers votre machine
rsync -avz runpod:/workspace/training/experiments/ ./experiments/
```

### Setup Google Colab Pro

```python
# Dans un notebook Colab

# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cloner le repo
!git clone https://github.com/your-repo/rebloom.git
%cd rebloom/training

# Installer dépendances
!pip install -r requirements.txt

# Copier dataset depuis Drive
!cp -r /content/drive/MyDrive/rebloom_dataset/* datasets/

# Lancer l'entraînement
!python scripts/train.py
```

---

## 5. Déploiement

### Option A : Replicate (Recommandé)

#### 1. Installer Cog

```bash
# Mac
brew install cog

# Linux
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_linux_x86_64
chmod +x /usr/local/bin/cog
```

#### 2. Créer cog.yaml

```yaml
# training/deployment/cog.yaml
build:
  python_version: "3.10"
  python_packages:
    - torch==2.0.1
    - torchvision==0.15.2
    - basicsr==1.4.2
    - realesrgan==0.3.0
    - pillow==10.0.0
  gpu: true
  cuda: "11.8"

predict: "predict.py:Predictor"
```

#### 3. Créer predict.py

```python
# training/deployment/predict.py
import cog
from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class Predictor(BasePredictor):
    def setup(self):
        """Charger le modèle fine-tuné"""
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        )

        self.upsampler = RealESRGANer(
            scale=4,
            model_path='models/rebloom_v1.pth',  # Votre modèle
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True  # FP16 pour vitesse
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: int = Input(description="Upscale factor", default=4, choices=[2, 4]),
        face_enhance: bool = Input(description="Enhance faces", default=False)
    ) -> Path:
        """Traiter une image"""
        img = Image.open(str(image))

        output, _ = self.upsampler.enhance(
            cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR),
            outscale=scale
        )

        output_path = Path("/tmp/output.png")
        cv2.imwrite(str(output_path), output)

        return output_path
```

#### 4. Déployer sur Replicate

```bash
cd training/deployment

# Test local
cog predict -i image=@test.jpg

# Login Replicate
cog login

# Push
cog push r8.im/your-username/rebloom-custom
```

#### 5. Mettre à jour ReBloom

Dans `.env` :
```env
CUSTOM_MODEL_ENABLED=true
CUSTOM_MODEL_PATH=your-username/rebloom-custom:latest
CUSTOM_MODEL_PROVIDER=replicate
```

### Option B : Hugging Face Inference Endpoints

1. **Upload sur HF Hub** :
```bash
huggingface-cli login
huggingface-cli upload your-username/rebloom-model ./models/rebloom_v1.pth
```

2. **Créer un Inference Endpoint** sur [huggingface.co/inference-endpoints](https://huggingface.co/inference-endpoints)

3. **Configurer l'endpoint** avec un handler custom

### Option C : Self-Hosted (avec GPU)

Pour du self-hosted, utilisez Docker + NVIDIA Container Toolkit :

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

RUN pip install torch torchvision basicsr realesrgan fastapi uvicorn

COPY models/rebloom_v1.pth /app/models/
COPY server.py /app/

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Checklist Finale

- [ ] Dataset créé (1000+ images)
- [ ] Flou synthétique généré
- [ ] Modèle pré-entraîné téléchargé
- [ ] Config adaptée à votre cas
- [ ] Entraînement lancé
- [ ] Métriques surveillées
- [ ] Checkpoint final exporté
- [ ] Déployé sur Replicate/HF
- [ ] Intégré dans ReBloom

---

## Ressources

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [BasicSR Documentation](https://basicsr.readthedocs.io/)
- [Replicate Cog Guide](https://replicate.com/docs/guides/push-a-model)
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

---

*Besoin d'aide ? Ouvrez une issue sur le repo ReBloom.*
