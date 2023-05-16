import os

import kornia.augmentation as K
import torch
from lightning import LightningDataModule
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


class SAT6(Dataset):
    classes = ["building", "barren_land", "trees", "grassland", "road", "water"]
    filename = "sat-6-full.mat"
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
        data = loadmat(self.path)
        if split == "train":
            X, y = data["train_x"], data["train_y"]
        else:
            X, y = data["test_x"], data["test_y"]

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
    # stats computed on train set after normalizing by 255.0
    min = torch.tensor([0.0, 0.0, 0.0, 0.0])
    min = torch.tensor([1.0, 1.0, 1.0, 1.0])
    mean = torch.tensor([0.4405, 0.4497, 0.4478, 0.4201])
    std = torch.tensor([0.2147, 0.1878, 0.1456, 0.3007])

    norm_rgb = K.Normalize(mean=mean[[0, 1, 2]], std=std[[0, 1, 2]])

    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float()
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
