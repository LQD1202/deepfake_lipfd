import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class AVLip(Dataset):
    def __init__(self, data_list):
        super(AVLip, self).__init__()
        self.data_list = data_list  # List of tuples: (full_image_path, label)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        if img_path is None:
            raise RuntimeError(f"Error reading image: {img_path}")
        img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])(img)
        crops = [[transforms.Resize((224, 224))(img[:, 500:, i:i + 500]) for i in range(5)], [], []]
        crop_idx = [(28, 196), (61, 163)]
        for i in range(len(crops[0])):
            crops[1].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            crops[2].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
        img = transforms.Resize((1120, 1120))(img)
        return img, crops, label
