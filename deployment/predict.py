import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from cog import BasePredictor, Input, Path


class SimpleUNet(nn.Module):
    """Architecture UNet pour le défloutage d'images."""
    
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.middle(e3)
        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        return d1 + x  # Connexion résiduelle


class Predictor(BasePredictor):
    def setup(self):
        """Charger le modèle au démarrage."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleUNet().to(self.device)
        
        # Charger les poids
        checkpoint = torch.load('rebloom-sharp-v1.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"Modèle chargé sur {self.device}")

    def predict(
        self,
        image: Path = Input(description="Image floue à restaurer"),
    ) -> Path:
        """Déflouter une image."""
        
        # Charger l'image
        img = Image.open(str(image)).convert('RGB')
        original_size = img.size
        
        # Redimensionner à une taille multiple de 4
        w, h = img.size
        new_w = (w // 4) * 4
        new_h = (h // 4) * 4
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Convertir en tensor
        tensor = TF.to_tensor(img_resized).unsqueeze(0).to(self.device)
        
        # Inférence
        with torch.no_grad():
            output = self.model(tensor)
        
        # Convertir en image
        output_img = TF.to_pil_image(output.squeeze(0).cpu().clamp(0, 1))
        
        # Redimensionner à la taille originale si nécessaire
        if output_img.size != original_size:
            output_img = output_img.resize(original_size, Image.BILINEAR)
        
        # Sauvegarder
        output_path = Path('/tmp/output.png')
        output_img.save(str(output_path), quality=95)
        
        return output_path
