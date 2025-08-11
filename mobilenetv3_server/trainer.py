import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from LIp_mobilenet import RALoss, AV_MobileNetV3
from datetime import datetime
import json
from tiny_clip.src.config import TinyCLIPVisionConfig
from tiny_clip.src import models as tinyclip_models

def get_loss():
    return RALoss()

class Trainer(nn.Module):
    def __init__(
        self,
        checkpoints_dir="./checkpoints",
        save_name="ckpt",
        lr=1e-4,
        beta1=0.9,
        weight_decay=1e-4,
        optim="adam",
        fix_encoder=False,
        load_checkpoint=""
    ):
        super().__init__()

        self.total_steps = 0
        self.save_dir = os.path.join(checkpoints_dir, save_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # === MODEL ===
        self.model = AV_MobileNetV3().to(self.device)

        # === LOSS ===
        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.BCEWithLogitsLoss()  # phù hợp nếu label là float

        # === OPTIMIZER ===
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if optim == "adam":
            self.optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
        elif optim == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
        else:
            raise ValueError("optim should be ['adam', 'sgd']")

        # === FREEZE ENCODER ===
        if fix_encoder:
            for name, p in self.model.named_parameters():
                if name.startswith("encoder"):
                    p.requires_grad = False
            print("[INFO] Encoder frozen.")

        # === TINYCLIP ===
        self.tinyclip = tinyclip_models.TinyCLIPVisionEncoder(TinyCLIPVisionConfig())
        ckpt_path = "./tiny_clip/vision.ckpt"
        state_dict = torch.load(ckpt_path, map_location=self.device)

        if any(k.startswith("vision_encoder.") for k in state_dict.keys()):
            state_dict = {
                k.replace("vision_encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("vision_encoder.")
            }
        self.tinyclip.load_state_dict(state_dict)
        self.tinyclip.to(self.device)
        self.tinyclip.eval()

        # === RESUME ===
        if load_checkpoint:
            print(f"[INFO] Loading checkpoint from: {load_checkpoint}")
            state = torch.load(load_checkpoint, map_location="cpu")
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.total_steps = state.get("total_steps", 0)
            match = re.search(r'(\d+)', os.path.basename(load_checkpoint))
            self.step_bias = int(match.group(1)) if match else 0
        else:
            print("[INFO] Training from scratch.")
            self.step_bias = 0

    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    def extract_tinyclip_feature(self, image):
        with torch.no_grad():
            return self.tinyclip(image)

    def set_input(self, input):
        self.input = input[0].to(self.device)  # image input for tinyclip
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]  # video frames
        self.label = input[2].to(self.device).float()

    def get_features(self):
        with torch.no_grad():
            self.features = self.extract_tinyclip_feature(self.input)
            self.features_tiny = self.features.to(self.device)

    def forward(self):
        self.get_features()

        self.output, self.weights_max, self.weights_org, self.output_mlp = self.model(
            self.crops, self.features_tiny
        )
        self.output = self.output.view(-1)

        loss_clip = self.criterion(self.weights_max, self.weights_org)
        loss_ce = self.criterion1(self.output, self.label)

        self.loss = loss_clip + loss_ce

    def get_loss(self):
        return self.loss.item()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.total_steps += 1

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }
        torch.save(state_dict, save_path)

    def log_model(self, epoch, loss, acc, loss_val, log_filename="model_mobile.json"):
        save_dir = self.save_dir  # dùng chung save_dir
        log_path = os.path.join(save_dir, log_filename)

        log_data = {
            "Model_name": f"model_epoch_{epoch}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Loss": loss,
            "Accuracy": acc,
            "Loss": loss_val
        }

        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                try:
                    existing_logs = json.load(f)
                except json.JSONDecodeError:
                    existing_logs = []
        else:
            existing_logs = []

        existing_logs.append(log_data)

        with open(log_path, "w") as f:
            json.dump(existing_logs, f, indent=4)

        print(f"[LOG] Model logged to {log_path}")
