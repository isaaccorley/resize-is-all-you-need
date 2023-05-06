from lightning import LightningDataModule
from torchgeo.datasets import So2Sat
import torch
from torch.utils.data import DataLoader


class So2SatDataModule(LightningDataModule):
    def preprocess(self, sample):
        sample["image"] = sample["image"].float() / 10000.0

        if self.pad_missing_band:
            B01, B09 = torch.zeros((2, 1, *sample["image"][1:]), dtype=torch.float)
            sample["image"] = torch.cat([B01, sample["image"][:8], B09, sample["image"][8:]], dim=0)
        return sample

    def __init__(
        self, root, bands=So2Sat.rgb_bands, batch_size=32, num_workers=8, seed=0, pad_missing_bands=False
    ):
        self.root = root
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_missing_bands = pad_missing_bands
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self):
        self.train_dataset = So2Sat(
            root=self.root,
            split="train",
            version="3",
            bands=self.bands,
            transforms=So2SatDataModule.preprocess,
        )
        self.test_dataset = So2Sat(
            root=self.root,
            split="test",
            version="3",
            bands=self.bands,
            transforms=So2SatDataModule.preprocess,
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
