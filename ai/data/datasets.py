import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os

class AVLip(Dataset):
    def __init__(self, data_dir):
        super(AVLip, self).__init__()
        self.data_list = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1120, 1120)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        #print(img_path)
        img = cv2.imread(img_path)
        #print(img)
        if img is None:
            raise RuntimeError(f"Error reading image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        # Cắt crop
        crops = [[], [], []]
        crop_idx = [(28, 196), (61, 163)]

        for i in range(5):
            crop_1x = img[:, 500:, i:i + 500]
            crops[0].append(transforms.functional.resize(crop_1x, (224, 224)))

            crop_065x = crop_1x[:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]
            crops[1].append(transforms.functional.resize(crop_065x, (224, 224)))

            crop_045x = crop_1x[:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]
            crops[2].append(transforms.functional.resize(crop_045x, (224, 224)))

        return img, crops
