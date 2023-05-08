import glob
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class PadMissingBands:
    def __call__(self, sample):
        B01, B09, B10 = torch.zeros(
            (3, 1, *sample["image"].shape[1:]), dtype=torch.float
        )
        sample["image"] = torch.cat(
            [B01, sample["image"][:8], B09, B10, sample["image"][8:]], dim=0
        )
        return sample


class S2Agri(Dataset):
    classes = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]
    all_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    missing_bands = ["B1", "B9", "B10"]
    rgb_bands = ["B4", "B3", "B2"]

    def __init__(self, root, bands=rgb_bands, transforms=None):
        for band in bands:
            assert band in self.all_bands

        self.band_indices = [self.all_bands.index(band) for band in bands]

        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.num_classes = len(self.classes)

        self.image_root = os.path.join(root, "s2-2017-IGARSS-NNI-NPY", "DATA")
        self.images = glob.glob(os.path.join(self.image_root, "*.npy"))
        with open(os.path.join(root, "labels.json")) as f:
            self.labels = json.load(f)["label_19class"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        id = os.path.splitext(os.path.basename(path))[0]
        image = np.load(self.images[index])
        label = torch.tensor(int(self.labels(id)), dtype=torch.long)
        image = image[:, self.band_indices, ...]
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class So2SatDataModule(LightningDataModule):
    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float() / 10000.0
        return sample

    def __init__(
        self,
        root,
        bands=S2Agri.rgb_bands,
        batch_size=32,
        num_workers=8,
        seed=0,
        pad_missing_bands=False,
    ):
        self.root = root
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_missing_bands = pad_missing_bands
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self):
        transforms = [self.preprocess]
        if self.pad_missing_bands:
            transforms.append(PadMissingBands())

        dataset = S2Agri(
            root=self.root,
            version=self.version,
            bands=self.bands,
            transforms=T.Compose(transforms),
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
