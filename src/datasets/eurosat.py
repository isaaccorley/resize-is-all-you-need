import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import EuroSAT


class EuroSATMinimal(LightningDataModule):
    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float() / 10000.0
        return sample

    def __init__(self, root, band_set="rgb", batch_size=32, num_workers=8, seed=0):
        """DataModule for small EuroSAT experiments.

        ***NOTE***: this uses random 90/10 splits instead of the torchgeo splits.

        Options for *band_set*: all, rgb
        """
        self.root = root
        self.band_set = band_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self):
        ds_train = EuroSAT(
            root=self.root,
            split="train",
            bands=EuroSAT.BAND_SETS[self.band_set],
            transforms=EuroSATMinimal.preprocess,
        )
        ds_val = EuroSAT(
            root=self.root,
            split="val",
            bands=EuroSAT.BAND_SETS[self.band_set],
            transforms=EuroSATMinimal.preprocess,
        )
        ds_test = EuroSAT(
            root=self.root,
            split="test",
            bands=EuroSAT.BAND_SETS[self.band_set],
            transforms=EuroSATMinimal.preprocess,
        )
        ds_all = ds_train + ds_val + ds_test
        self.train_dataset, self.test_dataset = random_split(
            ds_all, [0.8, 0.2], generator=self.generator
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
