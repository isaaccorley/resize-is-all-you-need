import json
import os

import numpy as np
import rasterio
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class TreeSatAI(Dataset):
    classes = [
        "Abies",
        "Acer",
        "Alnus",
        "Betula",
        "Cleared",
        "Fagus",
        "Fraxinus",
        "Larix",
        "Picea",
        "Pinus",
        "Populus",
        "Prunus",
        "Pseudotsuga",
        "Quercus",
        "Tilia",
    ]
    splits = {"train": "train_filenames.lst", "test": "test_filenames.lst"}
    sizes = {20: "200m", 6: "60m"}
    labels_path = os.path.join("labels", "TreeSatBA_v9_60m_multi_labels.json")
    all_bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B11",
        "B12",
        "B1",
        "B9",
    ]
    rgb_bands = ["B4", "B3", "B2"]
    correct_band_order = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ]

    def __init__(
        self,
        root,
        split="train",
        bands=rgb_bands,
        multilabel=False,
        transforms=None,
        size=20,
        pad_missing_band=True,
    ):
        assert split in self.splits
        assert size in self.sizes
        for band in bands:
            assert band in self.all_bands

        self.band_indices = [self.all_bands.index(band) for band in bands]
        self.band_indices_rasterio = [idx + 1 for idx in self.band_indices]

        self.root = root
        self.split = split
        self.size = size
        self.bands = bands
        self.multilabel = multilabel
        self.transforms = transforms
        self.pad_missing_band = pad_missing_band
        self.num_classes = len(self.classes)

        image_root = os.path.join(root, "s2", self.sizes[size])
        split_path = os.path.join(root, self.splits[split])
        with open(split_path) as f:
            images = f.read().strip().splitlines()
        self.images = [os.path.join(image_root, image) for image in images]

        if self.multilabel:
            labels_path = os.path.join(root, self.labels_path)
            with open(labels_path) as f:
                self.labels = json.load(f)
        else:
            self.labels = [
                os.path.basename(image).split("_")[0] for image in self.images
            ]

    def __len__(self):
        return len(self.images)

    def _load_image(self, path):
        with rasterio.open(path) as f:
            image = f.read(
                self.band_indices_rasterio, out_shape=(self.size, self.size)
            ).astype(np.int32)

        image = torch.from_numpy(image)

        if self.pad_missing_band:
            missing_band = torch.zeros((1, self.size, self.size), dtype=torch.int32)
            image = torch.cat([image[:10], missing_band, image[10:]], dim=0)

        image = image.to(torch.float).clip(min=0.0, max=None)
        return image

    def _load_target(self, index):
        if self.multilabel:
            path = self.images[index]
            filename = os.path.basename(path)
            classes = [label[0] for label in self.labels[filename]]
            onehot = torch.zeros((self.num_classes,), dtype=torch.long)
            for cls in classes:
                idx = self.classes.index(cls)
                onehot[idx] = 1
            return onehot
        else:
            cls = self.labels[index]
            label = self.classes.index(cls)
            label = torch.tensor(label).to(torch.long)
            return label

    def __getitem__(self, index):
        path = self.images[index]
        image = self._load_image(path)
        label = self._load_target(index)
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class TreeSatAIDataModule(LightningDataModule):
    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float() / 10000.0
        return sample

    def __init__(
        self,
        root,
        bands,
        multilabel=False,
        size=20,
        pad_missing_band=True,
        batch_size=32,
        num_workers=8,
        seed=0,
    ):
        self.root = root
        self.bands = bands
        self.multilabel = multilabel
        self.batch_size = batch_size
        self.size = size
        self.num_workers = num_workers
        self.pad_missing_band = pad_missing_band
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self):
        self.train_dataset = TreeSatAI(
            root=self.root,
            split="train",
            bands=self.bands,
            multilabel=self.multilabel,
            transforms=TreeSatAIDataModule.preprocess,
            size=self.size,
            pad_missing_band=self.pad_missing_band,
        )
        self.test_dataset = TreeSatAI(
            root=self.root,
            split="test",
            bands=self.bands,
            multilabel=self.multilabel,
            transforms=TreeSatAIDataModule.preprocess,
            size=self.size,
            pad_missing_band=self.pad_missing_band,
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