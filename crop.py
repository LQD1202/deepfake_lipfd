import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
from PIL import Image

def load_crop_and_save_image(img_path, output_dir):
    # Tạo thư mục lưu ảnh nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Đặt tên gốc từ ảnh
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Chuyển đổi ảnh và chuẩn hóa
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Error reading image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)  # Tensor: (C, H, W)

    # Resize toàn ảnh về (1120, 1120)
    img = F.resize(img, (1120, 1120))

    crop_idx = [(28, 196), (61, 163)]

    for i in range(5):
        crop_1x = img[:, 500:, i:i + 500]
        crop_1x_resized = F.resize(crop_1x, (224, 224))
        save_tensor_image(crop_1x_resized, os.path.join(output_dir, f"{base_name}_crop{i}_1x.jpg"))

        crop_065x = crop_1x[:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]
        crop_065x_resized = F.resize(crop_065x, (224, 224))
        save_tensor_image(crop_065x_resized, os.path.join(output_dir, f"{base_name}_crop{i}_065x.jpg"))

        crop_045x = crop_1x[:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]
        crop_045x_resized = F.resize(crop_045x, (224, 224))
        save_tensor_image(crop_045x_resized, os.path.join(output_dir, f"{base_name}_crop{i}_045x.jpg"))

def save_tensor_image(tensor_img, save_path):
    # Bỏ chuẩn hóa (để hiển thị đúng màu)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = tensor_img * std + mean
    img = torch.clamp(img, 0, 1)

    # Chuyển sang PIL và lưu
    pil_img = transforms.ToPILImage()(img)
    pil_img.save(save_path)
