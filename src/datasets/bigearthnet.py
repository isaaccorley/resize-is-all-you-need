import kornia.augmentation as K
import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import BigEarthNet


class PadMissingBands:
    def __call__(self, sample):
        B10 = torch.zeros((1, *sample["image"].shape[1:]), dtype=torch.float)
        sample["image"] = torch.cat(
            [sample["image"][:8], B10, sample["image"][8:]], dim=0
        )
        return sample


class SelectRGB:
    def __call__(self, sample):
        indices = torch.tensor([3, 2, 1])
        sample["image"] = torch.index_select(sample["image"], dim=0, index=indices)
        return sample


class BigEarthNetDataModule(LightningDataModule):
    # stats from https://github.com/ServiceNow/seasonal-contrast/blob/main/datasets/bigearthnet_dataset.py
    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    mean = (
        torch.tensor(
            [
                340.76769064,
                429.9430203,
                614.21682446,
                590.23569706,
                950.68368468,
                1792.46290469,
                2075.46795189,
                2218.94553375,
                2266.46036911,
                2246.0605464,
                0.0,  # padded band
                1594.42694882,
                1009.32729131,
            ]
        )
        / 10000.0
    )

    std = (
        torch.tensor(
            [
                554.81258967,
                572.41639287,
                582.87945694,
                675.88746967,
                729.89827633,
                1096.01480586,
                1273.45393088,
                1365.45589904,
                1356.13789355,
                1302.3292881,
                10000.0,  # padded band
                1079.19066363,
                818.86747235,
            ]
        )
        / 10000.0
    )

    norm_rgb = K.Normalize(mean=mean[[3, 2, 1]], std=std[[3, 2, 1]])
    norm_msi = K.Normalize(mean=mean, std=std)

    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float()
        return sample

    def __init__(
        self,
        root,
        bands="s2",
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
        if self.bands == "rgb":
            transforms.append(SelectRGB())

        self.train_dataset = BigEarthNet(
            root=self.root,
            split="train",
            bands="s2",
            num_classes=19,
            transforms=T.Compose(transforms),
        )
        self.test_dataset = BigEarthNet(
            root=self.root,
            split="test",
            bands="s2",
            num_classes=19,
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
