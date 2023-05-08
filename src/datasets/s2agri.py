import os

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SAT6(Dataset):
    classes = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]
    all_bands = ["R", "G", "B", "N"]
    rgb_bands = ["R", "G", "B"]

    def __init__(self, root, split="train", bands=rgb_bands, transforms=None):
        assert split in ["train", "test"]
        for band in bands:
            assert band in self.all_bands

        self.band_indices = [self.all_bands.index(band) for band in bands]

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.num_classes = len(self.classes)

        self.path = os.path.join(root, self.filename)

        self.X = torch.from_numpy(X).to(torch.uint8).permute(3, 2, 0, 1)
        self.y = torch.from_numpy(y).to(torch.long).argmax(dim=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        image = self.X[index]
        image = image[self.band_indices, ...]
        label = self.y[index]
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class SAT6DataModule(LightningDataModule):
    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float() / 255.0
        return sample

    def __init__(
        self, root, bands=SAT6.rgb_bands, batch_size=32, num_workers=8, seed=0
    ):
        self.root = root
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self):
        self.train_dataset = SAT6(
            root=self.root,
            split="train",
            bands=self.bands,
            transforms=SAT6DataModule.preprocess,
        )
        self.test_dataset = SAT6(
            root=self.root,
            split="test",
            bands=self.bands,
            transforms=SAT6DataModule.preprocess,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
