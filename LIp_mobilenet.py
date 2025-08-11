import torch
import numpy as np
import torch.nn as nn
from newmodel import AV_MobileNetV3
from mobilenetv3_server.tiny_clip.src.config import TinyCLIPVisionConfig
from mobilenetv3_server.tiny_clip.src import models as tinyclip_models


class LipFD(nn.Module):
    def __init__(self, num_classes=1):
        super(LipFD, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=5)  # (1120, 1120) -> (224, 224)

        # === TINYCLIP ===
        self.tinyclip = tinyclip_models.TinyCLIPVisionEncoder(TinyCLIPVisionConfig())
        ckpt_path = "./mobilenetv3_server/tiny_clip/vision.ckpt"
        state_dict = torch.load(ckpt_path, map_location='cpu')  # Chuyển sang CPU để load

        if any(k.startswith("vision_encoder.") for k in state_dict.keys()):
            state_dict = {
                k.replace("vision_encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("vision_encoder.")
            }

        self.tinyclip.load_state_dict(state_dict)
        self.tinyclip.eval()  # đặt ở chế độ eval

        # === ĐÓNG BĂNG ===
        for param in self.tinyclip.parameters():
            param.requires_grad = False

        self.backbone = AV_MobileNetV3()

    def forward(self, x, feature):
        return self.backbone(x, feature)

    def get_features(self, x):
        x = self.conv1(x)
        with torch.no_grad():  # đảm bảo không tính gradient cho TinyCLIP
            features = self.tinyclip(x)
        return features


class RALoss(nn.Module):
    def __init__(self):
        super(RALoss, self).__init__()

    def forward(self, alphas_max, alphas_org):
        loss = 0.0
        batch_size = alphas_org[0].shape[0]
        for i in range(len(alphas_org)):
            loss_wt = 0.0
            for j in range(batch_size):
                loss_wt += torch.tensor([10.0], device=alphas_max[i][j].device) / torch.exp(
                    alphas_max[i][j] - alphas_org[i][j]
                )

            loss += loss_wt / batch_size
        return float(loss/len(alphas_org))
 