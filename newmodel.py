import torch
import torch.nn as nn
from torch.nn.functional import softmax
from PIL import Image
import torchvision.transforms as T

from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights

class AV_MobileNetV3(nn.Module):
    def __init__(self, pretrained_mobilenetv3=True):
        super(AV_MobileNetV3, self).__init__()
        if pretrained_mobilenetv3:
            weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
        self.shufflenet_v2 = shufflenet_v2_x0_5(weights=weights)
        # self.clip_proj = nn.Linear(512, 768)
        self.get_weight = nn.Sequential(
            nn.Linear(1512, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(1512, 1)

    def forward(self, x, feature):
        features, weights, parts, weights_org, weights_max = [[] for _ in range(5)] 
        for t in range(len(x[0])):  # Loop over time steps
            features.clear()
            weights.clear()

            for b in range(len(x)):  # Loop over batch
                f = self.shufflenet_v2(x[b][t])       # [1, 960, 1, 1]
                f = torch.flatten(f, 1)           # [1, 960]

                concat = torch.cat([f, feature], dim=1)  # [1, 1728]
                features.append(concat)
                weights.append(self.get_weight(concat))  # [1, 1]

            features_stack = torch.stack(features, dim=2)   # [1, 1728, B]
            weights_stack = torch.stack(weights, dim=2)     # [1, 1, B]
            weights_stack = softmax(weights_stack, dim=2)

            weights_max.append(weights_stack.squeeze(1).max(dim=1)[0])
            weights_org.append(weights_stack.squeeze(1)[:, 0])
            part = features_stack.mul(weights_stack).sum(2) / weights_stack.sum(2)
            parts.append(part)
            

        parts_stack = torch.stack(parts, dim=0)        # [T, 1, 1728]
        out_mlp = parts_stack.sum(0) / parts_stack.shape[0]  # [1, 1728]
#        out_mlp = self.feature_dis(out_mlp)            # [1, 960]
        pred_score = self.fc(out_mlp)                      # [1, 1]
        return pred_score, weights_max, weights_org, out_mlp
