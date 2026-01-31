import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from cog import BasePredictor, Input, Path

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.middle = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 2, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.middle(e3)
        d3 = self.dec3(m)
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        return d1 + x

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleUNet().to(self.device)
        self.model.load_state_dict(torch.load('rebloom-sharp-v1.pth', map_location=self.device))
        self.model.eval()

    def predict(self, image: Path = Input(description="Image floue")) -> Path:
        img = Image.open(str(image)).convert('RGB')
        w, h = img.size
        new_w, new_h = (w // 4) * 4, (h // 4) * 4
        img = img.resize((new_w, new_h), Image.BILINEAR)
        tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        output_img = TF.to_pil_image(output.squeeze(0).cpu().clamp(0, 1))
        output_path = Path('/tmp/output.png')
        output_img.save(str(output_path))
        return output_path
