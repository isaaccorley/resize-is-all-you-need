import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import So2Sat


class PadMissingBands:
    def __call__(self, sample):
        B01, B09, B10 = torch.zeros(
            (3, 1, *sample["image"].shape[1:]), dtype=torch.float
        )
        sample["image"] = torch.cat(
            [B01, sample["image"][:8], B09, B10, sample["image"][8:]], dim=0
        )
        return sample


class So2SatDataModule(LightningDataModule):
    # stats computed on train set (data is already normalized to [0, 1])

    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float() / 1.0  # values are already in [0, 1]
        return sample

    def __init__(
        self,
        root,
        bands=So2Sat.rgb_bands,
        version="3_random",
        batch_size=32,
        num_workers=8,
        seed=0,
        pad_missing_bands=False,
    ):
        self.root = root
        self.bands = bands
        self.version = version
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_missing_bands = pad_missing_bands
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self):
        transforms = [self.preprocess]
        if self.pad_missing_bands:
            transforms.append(PadMissingBands())

        self.train_dataset = So2Sat(
            root=self.root,
            split="train",
            version=self.version,
            bands=self.bands,
            transforms=T.Compose(transforms),
        )
        self.test_dataset = So2Sat(
            root=self.root,
            split="test",
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
