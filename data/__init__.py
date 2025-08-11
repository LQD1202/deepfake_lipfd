import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from .datasets import AVLip

def get_bal_sampler(dataset):
    targets = np.concatenate([d.targets for d in dataset.datasets])
    class_counts = np.bincount(targets)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = weights[targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

def create_dataloader(dataset, batch_size=1, class_bal=False, num_threads=0, is_train=True, serial_batches=False):
    sampler = get_bal_sampler(dataset) if class_bal else None
    # shuffle = is_train and not class_bal and not serial_batches
    shuffle = True
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_threads,
        pin_memory=True if torch.cuda.is_available() else False
    )
