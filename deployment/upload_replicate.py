import replicate
import os

# Vérifier le token
token = os.environ.get("REPLICATE_API_TOKEN")
print(f"Token configuré: {token[:10]}..." if token else "❌ Token manquant!")

# Lister tes modèles
models = replicate.models.list()
for m in models:
    print(f"- {m.owner}/{m.name}")
