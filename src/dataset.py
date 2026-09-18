"""
Loads the FER2013 dataset from its standard CSV distribution
(columns: `emotion`, `pixels`, `Usage`) and exposes it as a PyTorch Dataset.

The CSV is not shipped in this repository (see data/README.md for how to
obtain it) because it is ~300MB and redistributing it is outside the scope
of this project.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import IMG_SIZE


class FER2013Dataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to fer2013.csv
    usage : str
        One of "Training", "PublicTest", "PrivateTest" - matches the
        `Usage` column FER2013 ships with, used to split train/val/test.
    augment : bool
        If True, apply light data augmentation (random crop + horizontal
        flip). Only meaningful for the training split.
    """

    def __init__(self, csv_path: str, usage: str = "Training", augment: bool = False):
        df = pd.read_csv(csv_path)
        if usage not in df["Usage"].unique():
            raise ValueError(
                f"'{usage}' not found in the Usage column. "
                f"Available values: {df['Usage'].unique().tolist()}"
            )
        df = df[df["Usage"] == usage].reset_index(drop=True)

        self.pixels = df["pixels"].values
        self.labels = df["emotion"].values.astype(np.int64)

        base = [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.RandomCrop(IMG_SIZE, padding=4),
                ]
                + base
            )
        else:
            self.transform = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        pixel_str = self.pixels[idx]
        img = np.fromstring(pixel_str, dtype=np.uint8, sep=" ").reshape(IMG_SIZE, IMG_SIZE)
        img = self.transform(img)
        label = int(self.labels[idx])
        return img, label
