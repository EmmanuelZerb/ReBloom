"""
ReBloom Sharp v2 - NAFNet Deblurring
Professional-grade motion deblurring model
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from cog import BasePredictor, Input, Path


# ============================================
# NAFNet Architecture
# ============================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=12,
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ============================================
# Predictor for Replicate
# ============================================

class Predictor(BasePredictor):
    def setup(self):
        """Load the model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = NAFNet(
            width=32,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2]
        ).to(self.device)
        
        # Load weights
        state = torch.load('rebloom-nafnet-v2.pth', map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        self.model.load_state_dict(state)
        self.model.eval()
        
        print(f"✅ NAFNet loaded on {self.device}")

    def predict(
        self,
        image: Path = Input(description="Blurry image to deblur")
    ) -> Path:
        """Run deblurring on the input image"""
        
        # Load image
        img = Image.open(str(image)).convert('RGB')
        original_size = img.size
        
        # To tensor
        tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(tensor)
        
        # To PIL
        output = output.squeeze(0).cpu().clamp(0, 1)
        output_img = TF.to_pil_image(output)
        
        # Save
        output_path = Path('/tmp/deblurred.png')
        output_img.save(str(output_path), quality=95)
        
        return output_path
