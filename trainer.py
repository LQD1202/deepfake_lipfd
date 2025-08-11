import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from LIp_mobilenet import RALoss, LipFD
from datetime import datetime
import json
from mobilenetv3_server import newmodel

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
        # self.save_dir = os.path.join(checkpoints_dir, save_name)
        # os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # === MODEL ===
        self.model = LipFD()
        self.model_teacher = newmodel.AV_MobileNetV3().to(self.device)

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
        self.model.to(self.device)
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

    def get_features(self):
        self.features = self.model.get_features(self.input)

    def set_input(self, input):
        self.input = input[0].to(self.device)  # image input for tinyclip
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]  # video frames
        self.label = input[2].to(self.device).float()

    def distillation_loss(self, student_logits, teacher_logits, T=4.0):
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)

    def forward(self):
        self.get_features()

        self.output, self.weights_max, self.weights_org, self.output_mlp = self.model(
            self.crops, self.features
        )
        with torch.no_grad():
            _, _, _, self.output_mlp_origin = self.model_teacher(self.crops, self.features)
        
        self.output = self.output.view(-1)

        self.loss_clip = self.criterion(self.weights_max, self.weights_org)
        self.loss_ce = self.criterion1(self.output, self.label)
        self.loss_kd = self.distillation_loss(self.output_mlp, self.output_mlp_origin)

        self.loss = 0.2 * self.loss_clip + 0.2 * self.loss_ce + 0.6 * self.loss_kd
        self.loss_list = [self.loss_clip, self.loss_ce, self.loss_kd]

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

    def log_model(self, epoch, loss, acc, ap, fpr, fnr, loss_val, log_filename="model_mobile.json"):

        save_dir = "./checkpoints"
        log_path = os.path.join(save_dir, log_filename)

        # Dữ liệu cần ghi vào log (chuyển loss_clip sang float nếu là tensor)
        log_data = {
            "Model_name": f"model_lipfd_w_temporal_{epoch}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Loss": loss,
            "Accuracy": acc,
            "AP": ap,
            "FPR": fpr,
            "FNR": fnr,
            "Loss_val": loss_val,
            "Loss_detail": {
                "loss_clip": float(self.loss_clip.item()) if hasattr(self.loss_clip, "item") else float(self.loss_clip),
                "loss_ce": float(self.loss_ce.item()) if hasattr(self.loss_ce, "item") else float(self.loss_ce),
                "loss_kd": float(self.loss_kd.item()) if hasattr(self.loss_kd, "item") else float(self.loss_kd)
            }
        }

        # Kiểm tra nếu file đã tồn tại, đọc và cập nhật log
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                try:
                    existing_logs = json.load(f)
                except json.JSONDecodeError:
                    existing_logs = []
        else:
            existing_logs = []

        # Thêm log mới vào danh sách
        existing_logs.append(log_data)

        # Ghi lại vào file JSON
        with open(log_path, "w") as f:
            json.dump(existing_logs, f, indent=4)

        print(f"Model log saved to {log_path}")


