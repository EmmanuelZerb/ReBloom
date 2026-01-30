# Guide Pas à Pas : Fine-Tuning pour Rendu Naturel

Ce guide vous accompagne pour créer un modèle de deblurring avec un **rendu naturel** (sans artefacts IA).

---

## Pourquoi les images font "IA" ?

| Symptôme | Cause | Notre solution |
|----------|-------|----------------|
| Peau plastique/lisse | GAN loss hallucine des textures | Pas de GAN loss |
| Détails inventés | Le modèle "invente" des détails | Losses conservatrices |
| Halos autour des objets | Sur-entraînement | Early stopping + monitoring |
| Couleurs saturées/HDR | Dataset de mauvaise qualité | Images naturelles HD |
| Artefacts aux bords | Mauvaise architecture | NAFNet (meilleur pour deblur) |

---

## Étape 1 : Préparer l'environnement

### 1.1 Prérequis

- Python 3.8+
- GPU NVIDIA avec 8GB+ VRAM (ou cloud GPU)
- 10GB d'espace disque

### 1.2 Installation

```bash
cd /Users/emmanuel.zerbib/ReBloom/training

# Rendre le script exécutable
chmod +x scripts/setup_nafnet.sh

# Lancer l'installation
./scripts/setup_nafnet.sh
```

Si vous êtes sur Mac sans GPU NVIDIA, passez directement à l'étape Cloud (Étape 5).

### 1.3 Vérification

```bash
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Étape 2 : Collecter les images

### 2.1 Critères pour un rendu naturel

| Critère | Bon | Mauvais |
|---------|-----|---------|
| Résolution | 1080p minimum | < 720p |
| Compression | PNG ou JPEG 90%+ | JPEG < 70% |
| Type | Photos naturelles | Images déjà retouchées/filtrées |
| Diversité | Paysages, portraits, objets | Un seul type |
| Éclairage | Varié (jour, nuit, intérieur) | Un seul type |

### 2.2 Sources recommandées (gratuites)

1. **DIV2K** (recommandé) - 900 images HD
   ```bash
   cd datasets/raw
   wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
   unzip DIV2K_train_HR.zip
   mv DIV2K_train_HR/* .
   rm -rf DIV2K_train_HR DIV2K_train_HR.zip
   ```

2. **Flickr2K** - 2650 images
   ```bash
   # Via Google Drive ou Kaggle
   ```

3. **Unsplash** - Photos pro gratuites
   - Téléchargez manuellement 500+ photos variées

### 2.3 Nombre d'images recommandé

| Objectif | Minimum | Idéal |
|----------|---------|-------|
| Test rapide | 100 | - |
| Résultat correct | 500 | 1000 |
| Résultat pro | 1000 | 3000+ |

---

## Étape 3 : Générer le dataset

### 3.1 Générer les paires flou/net

```bash
cd /Users/emmanuel.zerbib/ReBloom/training
source venv/bin/activate

python scripts/generate_blur.py \
  --input datasets/raw \
  --output-sharp datasets/processed/sharp \
  --output-blur datasets/processed/blur \
  --blur-types gaussian,motion,defocus \
  --intensity medium \
  --format png
```

### 3.2 Options de flou

| Option | Description | Recommandé pour |
|--------|-------------|-----------------|
| `gaussian` | Flou général | Usage quotidien |
| `motion` | Flou de bougé | Photos smartphone |
| `defocus` | Flou de mise au point | Photos DSLR |
| `all` | Mix aléatoire | Généralisation |

### 3.3 Intensité du flou

| Intensité | Quand l'utiliser |
|-----------|------------------|
| `light` | Si vos utilisateurs ont des images légèrement floues |
| `medium` | Usage général (recommandé) |
| `heavy` | Pour des images très dégradées |

**Conseil** : Commencez par `medium`, ajustez si nécessaire.

### 3.4 Créer le set de validation

```bash
# Prendre 10% pour la validation
mkdir -p datasets/validation/{sharp,blur}

# Déplacer 10% des images
cd datasets/processed
ls sharp | shuf | head -n $(( $(ls sharp | wc -l) / 10 )) | while read f; do
  mv sharp/$f ../validation/sharp/
  mv blur/$f ../validation/blur/
done
```

### 3.5 Vérifier le dataset

```bash
echo "Train: $(ls datasets/processed/sharp | wc -l) images"
echo "Val: $(ls datasets/validation/sharp | wc -l) images"
```

Vous devriez avoir environ 90% train / 10% validation.

---

## Étape 4 : Lancer l'entraînement (Local)

### 4.1 Entraînement complet

```bash
cd /Users/emmanuel.zerbib/ReBloom/training
source venv/bin/activate

python scripts/train_nafnet.py \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.0001
```

### 4.2 Test rapide (2 epochs)

```bash
python scripts/train_nafnet.py --test
```

### 4.3 Monitoring avec TensorBoard

```bash
# Dans un autre terminal
tensorboard --logdir runs/

# Ouvrez http://localhost:6006
```

### 4.4 Quoi surveiller ?

| Métrique | Bon signe | Mauvais signe | Action |
|----------|-----------|---------------|--------|
| **Loss** | Descend régulièrement | Stagne ou remonte | Réduire LR |
| **PSNR** | Monte (> 28dB = bien) | Stagne < 25dB | Plus de données |
| **SSIM** | Monte (> 0.9 = bien) | < 0.85 | Vérifier dataset |

### 4.5 Quand arrêter ?

- **Early stopping** : Si PSNR ne monte plus pendant 10 epochs
- **Bon score** : PSNR > 30dB, SSIM > 0.92
- **Maximum** : 100-200 epochs suffisent généralement

---

## Étape 5 : Entraînement Cloud (Recommandé)

Si vous n'avez pas de GPU, utilisez un service cloud.

### 5.1 Option A : Google Colab (Gratuit)

1. Allez sur [colab.research.google.com](https://colab.research.google.com)
2. Créez un nouveau notebook
3. Collez ce code :

```python
# [Cellule 1] Monter Drive
from google.colab import drive
drive.mount('/content/drive')

# [Cellule 2] Cloner le repo
!git clone https://github.com/VOTRE_USERNAME/ReBloom.git
%cd ReBloom/training

# [Cellule 3] Installer dépendances
!pip install torch torchvision
!pip install -r requirements.txt

# [Cellule 4] Copier dataset depuis Drive
# (uploadez votre dataset dans Drive d'abord)
!cp -r /content/drive/MyDrive/rebloom_dataset/* datasets/

# [Cellule 5] Lancer l'entraînement
!python scripts/train_nafnet.py --epochs 50 --batch-size 4

# [Cellule 6] Sauvegarder le modèle sur Drive
!cp checkpoints/best_model.pth /content/drive/MyDrive/
```

### 5.2 Option B : RunPod (~$0.50/h)

1. Créez un compte sur [runpod.io](https://runpod.io)
2. Lancez un pod avec template PyTorch + RTX 4090
3. Uploadez votre code et dataset
4. Lancez l'entraînement

```bash
# Sur le pod RunPod
cd /workspace
git clone https://github.com/VOTRE_USERNAME/ReBloom.git
cd ReBloom/training

pip install -r requirements.txt

# Télécharger dataset (si sur S3/GCS)
aws s3 sync s3://your-bucket/dataset ./datasets/

# Train
python scripts/train_nafnet.py --epochs 100 --batch-size 16
```

### 5.3 Estimation temps/coût

| Plateforme | GPU | 1000 images | Coût |
|------------|-----|-------------|------|
| Colab Free | T4 | ~4h | Gratuit |
| Colab Pro | A100 | ~1h | $10/mois |
| RunPod | RTX 4090 | ~2h | ~$1 |
| RunPod | A100 | ~1h | ~$2 |

---

## Étape 6 : Tester le modèle

### 6.1 Créer un script de test

Créez `scripts/test_model.py` :

```python
#!/usr/bin/env python3
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
import sys

# Importer le modèle
sys.path.insert(0, str(Path(__file__).parent))
from train_nafnet import NAFNetSimple

def test_image(model_path: str, image_path: str, output_path: str):
    # Charger modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NAFNetSimple(width=32, num_blocks=16).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Charger image
    img = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    # Inférence
    with torch.no_grad():
        output = model(tensor)

    # Sauvegarder
    output = output.squeeze(0).cpu()
    output = TF.to_pil_image(output.clamp(0, 1))
    output.save(output_path)
    print(f"Sauvegardé: {output_path}")

if __name__ == '__main__':
    test_image(
        model_path='checkpoints/best_model.pth',
        image_path='test_input.jpg',
        output_path='test_output.png'
    )
```

### 6.2 Lancer le test

```bash
python scripts/test_model.py
```

### 6.3 Évaluer la qualité

Vérifiez que l'image de sortie :
- [ ] N'a pas de "peau plastique"
- [ ] N'invente pas de détails (zoom sur les textures)
- [ ] Garde des couleurs naturelles
- [ ] N'a pas de halos aux contours

---

## Étape 7 : Déployer sur Replicate

### 7.1 Installer Cog

```bash
# Mac
brew install cog

# Linux
curl -o /usr/local/bin/cog -L \
  https://github.com/replicate/cog/releases/latest/download/cog_linux_x86_64
chmod +x /usr/local/bin/cog
```

### 7.2 Créer le fichier cog.yaml

Créez `training/deployment/cog.yaml` :

```yaml
build:
  python_version: "3.10"
  python_packages:
    - torch==2.0.1
    - torchvision==0.15.2
    - pillow>=10.0.0
  gpu: true
  cuda: "11.8"

predict: "predict.py:Predictor"
```

### 7.3 Créer predict.py

Créez `training/deployment/predict.py` :

```python
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from cog import BasePredictor, Input, Path

class NAFNetSimple(torch.nn.Module):
    # ... (copier la classe depuis train_nafnet.py)
    pass

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device('cuda')
        self.model = NAFNetSimple(width=32, num_blocks=16).to(self.device)

        checkpoint = torch.load('best_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(
        self,
        image: Path = Input(description="Image floue à restaurer"),
    ) -> Path:
        # Charger
        img = Image.open(str(image)).convert('RGB')
        tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)

        # Inférence
        with torch.no_grad():
            output = self.model(tensor)

        # Sauvegarder
        output = output.squeeze(0).cpu()
        output = TF.to_pil_image(output.clamp(0, 1))

        output_path = Path('/tmp/output.png')
        output.save(str(output_path))

        return output_path
```

### 7.4 Déployer

```bash
cd training/deployment

# Copier le modèle
cp ../checkpoints/best_model.pth .

# Test local
cog predict -i image=@test.jpg

# Login Replicate
cog login

# Push
cog push r8.im/VOTRE_USERNAME/rebloom-natural
```

### 7.5 Mettre à jour ReBloom

Dans votre `.env` :

```env
CUSTOM_MODEL_ENABLED=true
CUSTOM_MODEL_PATH=VOTRE_USERNAME/rebloom-natural:latest
CUSTOM_MODEL_PROVIDER=replicate
```

---

## Récapitulatif

### Checklist

- [ ] Environnement installé
- [ ] 500+ images HD collectées
- [ ] Dataset généré (sharp/blur pairs)
- [ ] Set de validation créé (10%)
- [ ] Entraînement lancé
- [ ] PSNR > 28dB atteint
- [ ] Test visuel OK (pas d'artefacts IA)
- [ ] Modèle déployé sur Replicate
- [ ] Intégré dans ReBloom

### Configuration optimale pour rendu naturel

| Paramètre | Valeur | Pourquoi |
|-----------|--------|----------|
| GAN Loss | **Aucune** | Cause #1 des artefacts |
| L1 Loss | 1.0 | Fidélité aux pixels |
| Edge Loss | 0.1 | Préserve les contours |
| Perceptual Loss | 0.1 (max) | Évite hallucinations |
| Learning Rate | 1e-4 | Conservateur |
| Epochs | 100 | Suffisant |

### Temps estimé

| Étape | Temps |
|-------|-------|
| Setup | 15 min |
| Collecter images | 1-2h |
| Générer dataset | 30 min |
| Entraînement | 2-4h (cloud) |
| Test & deploy | 30 min |
| **Total** | **4-8h** |

---

## Support

Si vous rencontrez des problèmes :

1. Vérifiez les logs TensorBoard
2. Testez avec `--test` d'abord
3. Assurez-vous que vos images sont de bonne qualité
4. Réduisez le `batch-size` si erreur mémoire

Bonne chance !
