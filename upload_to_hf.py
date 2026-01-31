"""
Script pour uploader le modèle ReBloom Sharp v1 sur Hugging Face

Usage: py upload_to_hf.py TON_TOKEN_ICI
"""
import sys
from huggingface_hub import HfApi, login

if len(sys.argv) < 2:
    print("Usage: py upload_to_hf.py TON_TOKEN_HUGGINGFACE")
    print("Récupère ton token sur: https://huggingface.co/settings/tokens")
    sys.exit(1)

token = sys.argv[1]

# 1. Login
print("=== Connexion à Hugging Face ===")
login(token=token)
print("✓ Connecté avec succès!")

# 2. Upload du modèle
print("\n=== Upload du modèle ===")
api = HfApi()

repo_id = "EmmanuelZerb/rebloom-sharp-v1.pth"
file_path = "rebloom-sharp-v1.pth"

print(f"Upload de {file_path} vers {repo_id}...")

api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="rebloom-sharp-v1.pth",
    repo_id=repo_id,
)

print("✓ Upload terminé avec succès!")
print(f"Ton modèle est disponible sur: https://huggingface.co/{repo_id}")
