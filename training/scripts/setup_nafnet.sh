#!/bin/bash
# ============================================
# ReBloom - Setup NAFNet pour Fine-Tuning
# ============================================
#
# Ce script installe tout le nécessaire pour
# fine-tuner NAFNet avec un rendu naturel
#
# Usage: bash setup_nafnet.sh
# ============================================

set -e  # Arrêter en cas d'erreur

echo "======================================"
echo "ReBloom NAFNet Setup"
echo "======================================"

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Répertoire de base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}[1/6]${NC} Vérification de Python..."
python3 --version || { echo -e "${RED}Python 3 non trouvé${NC}"; exit 1; }

echo -e "${GREEN}[2/6]${NC} Création de l'environnement virtuel..."
cd "$BASE_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Environnement créé: venv/"
else
    echo "Environnement existe déjà"
fi

echo -e "${GREEN}[3/6]${NC} Activation de l'environnement..."
source venv/bin/activate

echo -e "${GREEN}[4/6]${NC} Installation des dépendances..."
pip install --upgrade pip

# PyTorch avec CUDA (si disponible)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU NVIDIA détecté, installation PyTorch+CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}Pas de GPU détecté, installation PyTorch CPU...${NC}"
    pip install torch torchvision
fi

# Dépendances
pip install -r requirements.txt

# NAFNet
echo -e "${GREEN}[5/6]${NC} Clonage de NAFNet..."
if [ ! -d "NAFNet" ]; then
    git clone https://github.com/megvii-research/NAFNet.git
    cd NAFNet
    pip install -e .
    cd ..
else
    echo "NAFNet déjà cloné"
fi

echo -e "${GREEN}[6/6]${NC} Téléchargement du modèle pré-entraîné..."
mkdir -p pretrained_models

# NAFNet-GoPro (motion deblurring)
if [ ! -f "pretrained_models/NAFNet-GoPro-width64.pth" ]; then
    echo "Téléchargement NAFNet-GoPro..."
    wget -O pretrained_models/NAFNet-GoPro-width64.pth \
        "https://github.com/megvii-research/NAFNet/releases/download/v0.0.0/NAFNet-GoPro-width64.pth"
else
    echo "Modèle déjà téléchargé"
fi

echo ""
echo "======================================"
echo -e "${GREEN}Setup terminé !${NC}"
echo "======================================"
echo ""
echo "Prochaines étapes:"
echo "  1. Placez vos images HD dans: datasets/raw/"
echo "  2. Générez le dataset: python scripts/generate_blur.py ..."
echo "  3. Lancez l'entraînement: python scripts/train_nafnet.py"
echo ""
echo "Activez l'environnement avec: source venv/bin/activate"
