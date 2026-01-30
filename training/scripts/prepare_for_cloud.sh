#!/bin/bash
# ============================================
# Préparer le projet pour upload cloud
# ============================================
# Ce script crée une archive optimisée pour
# uploader sur RunPod/Vast.ai
#
# Usage: ./prepare_for_cloud.sh
# ============================================

set -e

echo "======================================"
echo "Préparation pour Cloud GPU"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$BASE_DIR/cloud_package"

# Créer le dossier de sortie
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[1/4] Copie du code..."
cp -r "$BASE_DIR/scripts" "$OUTPUT_DIR/"
cp -r "$BASE_DIR/configs" "$OUTPUT_DIR/"
cp "$BASE_DIR/requirements.txt" "$OUTPUT_DIR/"

echo "[2/4] Copie du dataset..."
if [ -d "$BASE_DIR/datasets/processed/sharp" ] && [ "$(ls -A $BASE_DIR/datasets/processed/sharp 2>/dev/null)" ]; then
    mkdir -p "$OUTPUT_DIR/datasets"
    cp -r "$BASE_DIR/datasets/processed" "$OUTPUT_DIR/datasets/"
    cp -r "$BASE_DIR/datasets/validation" "$OUTPUT_DIR/datasets/" 2>/dev/null || true

    TRAIN_COUNT=$(ls "$OUTPUT_DIR/datasets/processed/sharp" 2>/dev/null | wc -l | tr -d ' ')
    VAL_COUNT=$(ls "$OUTPUT_DIR/datasets/validation/sharp" 2>/dev/null | wc -l | tr -d ' ')
    echo "   Train: $TRAIN_COUNT images"
    echo "   Val: $VAL_COUNT images"
else
    echo "   ⚠️  Pas de dataset trouvé dans datasets/processed/"
    echo "   Générez-le d'abord avec: python scripts/generate_blur.py"
    mkdir -p "$OUTPUT_DIR/datasets/processed/sharp"
    mkdir -p "$OUTPUT_DIR/datasets/processed/blur"
    mkdir -p "$OUTPUT_DIR/datasets/validation/sharp"
    mkdir -p "$OUTPUT_DIR/datasets/validation/blur"
fi

echo "[3/4] Création du script de démarrage..."
cat > "$OUTPUT_DIR/start_training.sh" << 'EOF'
#!/bin/bash
# Script de démarrage pour cloud GPU
set -e

echo "=== Installation des dépendances ==="
pip install -r requirements.txt

echo ""
echo "=== Vérification GPU ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

echo ""
echo "=== Vérification dataset ==="
TRAIN=$(ls datasets/processed/sharp 2>/dev/null | wc -l)
VAL=$(ls datasets/validation/sharp 2>/dev/null | wc -l)
echo "Train: $TRAIN images"
echo "Val: $VAL images"

if [ "$TRAIN" -lt 10 ]; then
    echo "⚠️  Pas assez d'images! Uploadez votre dataset dans datasets/"
    exit 1
fi

echo ""
echo "=== Démarrage de l'entraînement ==="
echo "Utilisez Ctrl+C pour arrêter"
echo ""

# Créer dossier checkpoints
mkdir -p checkpoints

# Lancer l'entraînement
python scripts/train_nafnet.py --epochs 100 --batch-size 8

echo ""
echo "=== Entraînement terminé ==="
echo "Modèle sauvegardé dans: checkpoints/best_model.pth"
EOF
chmod +x "$OUTPUT_DIR/start_training.sh"

echo "[4/4] Création de l'archive..."
cd "$BASE_DIR"
tar -czf cloud_training.tar.gz -C "$OUTPUT_DIR" .

SIZE=$(du -h cloud_training.tar.gz | cut -f1)
echo ""
echo "======================================"
echo "✅ Package prêt!"
echo "======================================"
echo ""
echo "Fichier: $BASE_DIR/cloud_training.tar.gz ($SIZE)"
echo ""
echo "Pour uploader sur RunPod/Vast.ai:"
echo "  1. Connectez-vous au pod via SSH"
echo "  2. scp -P PORT cloud_training.tar.gz root@IP:/workspace/"
echo "  3. cd /workspace && tar -xzf cloud_training.tar.gz"
echo "  4. ./start_training.sh"
echo ""
