import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import RESISC45
from torchvision.transforms import Normalize

MEAN = torch.tensor([93.89391792, 97.11226906, 87.56775284])
STD = torch.tensor([51.84919672, 47.2365918 , 47.06308786])


class RESISC45Minimal(LightningDataModule):
    def get_preprocess(self, method="divide"):
        normalize_all = Normalize(mean=MEAN, std=STD)

        def preprocess_normal(sample):
            sample["image"] = sample["image"].float() / 255.0
            return sample

        def preprocess_standardization(sample):
            sample["image"] = normalize_all(sample["image"].float())
            return sample

        if method == "divide":
            return preprocess_normal
        elif method == "standardization":
            return preprocess_standardization
        elif method == "none":
            return lambda x: x
        else:
            raise ValueError("Method not supported")

    def __init__(
        self,
        root,
        normalization_method="divide",
        batch_size=32,
        num_workers=8,
        train_pct=1.0,
        use_both_trainval=False,
    ):
        """DataModule for small RESISC45 experiments.

        ***NOTE***: this uses random 90/10 splits instead of the torchgeo splits.

        Options for *normalization_method*: divide, standardization
        """
        self.root = root
        self.normalization_method = normalization_method
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        preprocess = self.get_preprocess(
            method=self.normalization_method
        )
        ds_train = RESISC45(
            root=self.root,
            split="train",
            transforms=preprocess,
        )

        ds_val = RESISC45(
            root=self.root,
            split="val",
            transforms=preprocess,
        )
        ds_test = RESISC45(
            root=self.root,
            split="test",
            transforms=preprocess,
        )
        self.train_dataset = ds_train
        self.test_dataset = ds_test

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
