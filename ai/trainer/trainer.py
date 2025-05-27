import os
import torch
import torch.nn as nn
from ai.models import build_model, get_loss
import json
from datetime import datetime
import os

class Trainer(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.total_steps = 0
        self.save_dir = "./checkpoints/"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model chưa chuyển sang GPU
        self.model = build_model("CLIP:ViT-L/14")
        self.pretrained = pretrained

        # Tìm step
        self.step_bias = 0
        ckpt_path = "./checkpoints/ckpt_0.pth"
        if os.path.exists(ckpt_path):
            try:
                self.step_bias = int(ckpt_path.split("_")[-1].split(".")[0]) + 1
            except ValueError:
                print(f"Warning: Không thể đọc step từ {ckpt_path}")

        # Load pretrained nếu có
        if self.pretrained:
            ckpt_load_path = "./checkpoints/ckpt.pth"
            if os.path.exists(ckpt_load_path):
                state_dict = torch.load(ckpt_load_path, map_location="cpu")
                self.model.load_state_dict(state_dict["model"])
                self.total_steps = state_dict.get("total_steps", 0)
                print(f"Model loaded from {ckpt_load_path}")

        # Đóng băng encoder
        for name, p in self.model.named_parameters():
            if name.startswith("encoder"):
                p.requires_grad = False

        # Optimizer: chỉ update phần trainable
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=2e-9,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Loss functions
        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss().to(self.device)

    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    def set_input(self, input):
        """Chuyển input tensor vào device, không cần chuyển model"""
        self.input = input[0].to(self.device, non_blocking=True)
        self.crops = [[t.to(self.device, non_blocking=True) for t in sublist] for sublist in input[1]]
        self.label = input[2].to(self.device, dtype=torch.float)

    def forward(self):
        self.get_features()
        # Chuyển phần model cần sang GPU
        self.model = self.model.to(self.device)
        self.output, self.weights_max, self.weights_org = self.model.forward(self.crops, self.features)
        self.output = self.output.view(-1)
        self.loss = self.criterion(self.weights_max, self.weights_org) + self.criterion1(self.output, self.label)

    def get_loss(self):
        return self.loss.item()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_features(self):
        # Chuyển phần model cần sang GPU
        self.model = self.model.to(self.device)
        self.features = self.model.get_features(self.input)

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_networks(self, load_filename):
        load_path = os.path.join(self.save_dir, load_filename)
        
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"No saved model found at {load_path}")

        # Load checkpoint về CPU hoặc GPU tùy thiết bị
        checkpoint = torch.load(load_path, map_location=self.device)

        # Load model và optimizer state
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)  # Đẩy model lên GPU hoặc CPU

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)

        print(f"Model loaded from {load_path} to {self.device}")

    def log_model(self, epoch, loss, acc, log_filename="model_log_temporal_w.json"):
        log_path = os.path.join(self.save_dir, log_filename)

        # Dữ liệu cần ghi vào log
        log_data = {
            "Model_name": "model_lipfd_temporal_w_{}".format(epoch),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Loss": loss,
            "Accuracy": acc
        }

        # Kiểm tra nếu file đã tồn tại, đọc và cập nhật log
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                try:
                    existing_logs = json.load(f)
                except json.JSONDecodeError:  # Nếu file rỗng hoặc lỗi, khởi tạo lại
                    existing_logs = []
        else:
            existing_logs = []

        # Thêm log mới vào danh sách
        existing_logs.append(log_data)

        # Ghi lại vào file JSON
        with open(log_path, "w") as f:
            json.dump(existing_logs, f, indent=4)

        print(f"Model log saved to {log_path}")



