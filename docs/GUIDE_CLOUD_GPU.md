# Guide : Louer un GPU Cloud pour Fine-Tuning

Ce guide vous explique comment louer un GPU et entraîner votre modèle dessus.

---

## Option 1 : RunPod (Recommandé pour débuter)

### Étape 1 : Créer un compte

1. Allez sur [runpod.io](https://www.runpod.io/)
2. Cliquez "Sign Up"
3. Créez un compte (email ou GitHub)

### Étape 2 : Ajouter du crédit

1. Allez dans "Billing" (icône profil en haut à droite)
2. Cliquez "Add Credits"
3. Ajoutez **$10-20** pour commencer (suffisant pour ~20h de RTX 4090)
4. Payez par carte bancaire

### Étape 3 : Lancer un GPU

1. Cliquez **"+ Deploy"** ou **"Pods"** > **"+ New Pod"**

2. Choisissez un GPU :
   | GPU | VRAM | Prix/h | Recommandé pour |
   |-----|------|--------|-----------------|
   | RTX 3090 | 24GB | ~$0.30 | Budget serré |
   | RTX 4090 | 24GB | ~$0.45 | **Recommandé** |
   | A100 40GB | 40GB | ~$1.00 | Dataset très grand |

3. Configuration :
   - **Template** : `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - **Container Disk** : 20 GB
   - **Volume Disk** : 50 GB (pour stocker le dataset)

4. Cliquez **"Deploy On-Demand"**

5. Attendez ~2 min que le pod démarre (status "Running")

### Étape 4 : Se connecter au pod

1. Cliquez sur votre pod dans la liste
2. Cliquez **"Connect"** > **"Start Web Terminal"**

   OU pour SSH (plus stable) :
   - Cliquez "Connect" > copiez la commande SSH
   - Dans votre terminal local : `ssh root@xxx.xxx.xxx.xxx -p XXXXX`

### Étape 5 : Uploader votre code et dataset

**Option A : Git (recommandé pour le code)**
```bash
# Dans le terminal du pod
cd /workspace
git clone https://github.com/VOTRE_USERNAME/ReBloom.git
cd ReBloom/training
```

**Option B : Depuis votre Mac via SCP**
```bash
# Dans VOTRE terminal local (pas le pod)
# Récupérez l'IP et port depuis RunPod > Connect > SSH

# Uploader le dossier training
scp -P XXXXX -r /Users/emmanuel.zerbib/ReBloom/training root@IP_DU_POD:/workspace/

# Uploader le dataset (peut prendre du temps)
scp -P XXXXX -r /Users/emmanuel.zerbib/ReBloom/training/datasets root@IP_DU_POD:/workspace/training/
```

**Option C : Via Google Drive / Dropbox**
```bash
# Dans le terminal du pod

# Installer rclone
curl https://rclone.org/install.sh | bash
rclone config  # Configurer Google Drive

# Télécharger depuis Drive
rclone copy gdrive:ReBloom/datasets /workspace/training/datasets
```

### Étape 6 : Installer les dépendances

```bash
# Dans le terminal du pod
cd /workspace/training

# Installer les requirements
pip install -r requirements.txt

# Vérifier que CUDA fonctionne
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Étape 7 : Lancer l'entraînement

```bash
# Test rapide (2 epochs)
python scripts/train_nafnet.py --test

# Si OK, lancer l'entraînement complet
python scripts/train_nafnet.py --epochs 100 --batch-size 16
```

**Conseil** : Utilisez `screen` ou `tmux` pour que l'entraînement continue même si vous fermez le terminal :

```bash
# Installer screen
apt install screen -y

# Créer une session
screen -S training

# Lancer l'entraînement
python scripts/train_nafnet.py --epochs 100

# Détacher la session : Ctrl+A puis D
# Revenir : screen -r training
```

### Étape 8 : Récupérer le modèle

```bash
# Dans VOTRE terminal local
scp -P XXXXX root@IP_DU_POD:/workspace/training/checkpoints/best_model.pth ~/Downloads/
```

### Étape 9 : Arrêter le pod (IMPORTANT !)

1. Retournez sur [runpod.io](https://www.runpod.io/)
2. Cliquez sur votre pod
3. Cliquez **"Stop"** (pause) ou **"Terminate"** (supprimer)

⚠️ **ATTENTION** : Le pod vous facture tant qu'il tourne ! N'oubliez pas de l'arrêter.

---

## Option 2 : Vast.ai (Moins cher)

### Différences avec RunPod

| Aspect | RunPod | Vast.ai |
|--------|--------|---------|
| Prix | ~$0.40-0.70/h | ~$0.20-0.50/h |
| Interface | Simple | Un peu plus technique |
| Fiabilité | Haute | Variable (machines perso) |
| Support | Bon | Basique |

### Étapes Vast.ai

1. Créez un compte sur [vast.ai](https://vast.ai/)

2. Ajoutez du crédit (Billing > Add Credit)

3. Allez dans **"Search"** ou **"Create"**

4. Filtrez :
   - GPU : RTX 4090 ou RTX 3090
   - CUDA : 11.8+
   - Disk : 50GB+

5. Cliquez **"Rent"** sur une offre

6. Une fois démarré, connectez-vous via SSH :
   ```bash
   ssh -p PORT root@IP -L 8080:localhost:8080
   ```

7. Suivez les mêmes étapes que RunPod (upload, install, train)

---

## Option 3 : Google Colab Pro (Le plus simple)

### Avantages
- Interface notebook (pas de terminal)
- Pas besoin de gérer les serveurs
- $10/mois illimité (avec limites d'usage)

### Inconvénients
- Sessions limitées (~12h max)
- Peut être interrompu
- Moins de contrôle

### Comment faire

1. Allez sur [colab.research.google.com](https://colab.research.google.com)

2. Souscrivez à Colab Pro ($10/mois) pour avoir accès aux GPU A100

3. Créez un nouveau notebook

4. Changez le runtime : **Runtime > Change runtime type > GPU > A100**

5. Copiez ce notebook :

```python
# ===== CELLULE 1 : Monter Google Drive =====
from google.colab import drive
drive.mount('/content/drive')

# ===== CELLULE 2 : Cloner le repo =====
!git clone https://github.com/VOTRE_USERNAME/ReBloom.git
%cd ReBloom/training

# ===== CELLULE 3 : Installer dépendances =====
!pip install torch torchvision --quiet
!pip install tqdm tensorboard scikit-image pillow opencv-python einops timm --quiet

# ===== CELLULE 4 : Vérifier GPU =====
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===== CELLULE 5 : Copier dataset depuis Drive =====
# Uploadez d'abord votre dataset dans Drive > MyDrive > rebloom_dataset/
!cp -r /content/drive/MyDrive/rebloom_dataset/* datasets/

# Vérifier
!ls datasets/processed/sharp | wc -l

# ===== CELLULE 6 : Lancer l'entraînement =====
!python scripts/train_nafnet.py --epochs 50 --batch-size 8

# ===== CELLULE 7 : Sauvegarder le modèle sur Drive =====
!cp checkpoints/best_model.pth /content/drive/MyDrive/
print("Modèle sauvegardé dans Google Drive!")
```

---

## Récapitulatif : Quelle option choisir ?

| Votre situation | Recommandation |
|-----------------|----------------|
| Première fois, veut simple | **RunPod** |
| Budget serré | **Vast.ai** |
| Veut un notebook, pas de terminal | **Google Colab Pro** |
| Gros dataset (>10GB) | **RunPod ou Vast.ai** |
| Entraînement long (>24h) | **RunPod** (plus stable) |

---

## Estimation coûts

| Dataset | GPU | Temps | Coût RunPod | Coût Vast.ai |
|---------|-----|-------|-------------|--------------|
| 500 images | RTX 4090 | ~1h | ~$0.50 | ~$0.25 |
| 1000 images | RTX 4090 | ~2h | ~$1.00 | ~$0.50 |
| 3000 images | RTX 4090 | ~4h | ~$2.00 | ~$1.00 |
| 3000 images | A100 | ~2h | ~$2.00 | ~$1.50 |

---

## Checklist avant de commencer

- [ ] Compte créé sur la plateforme
- [ ] Crédit ajouté ($10-20)
- [ ] Dataset prêt localement (paires sharp/blur)
- [ ] Code pushé sur GitHub (ou prêt à uploader)

---

## Troubleshooting

### "CUDA out of memory"
→ Réduisez `--batch-size` (8 → 4 → 2)

### "Connection closed"
→ Utilisez `screen` ou `tmux` pour garder la session active

### "Pod not starting"
→ Essayez un autre GPU ou une autre région

### "Slow upload"
→ Compressez le dataset : `tar -czf dataset.tar.gz datasets/`

---

## Commandes utiles

```bash
# Voir l'utilisation GPU en temps réel
watch -n 1 nvidia-smi

# Voir les logs d'entraînement
tail -f runs/*/events.*

# Compresser le modèle final
tar -czf model.tar.gz checkpoints/best_model.pth

# Tuer un processus Python bloqué
pkill -f train_nafnet
```
