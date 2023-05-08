import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import UCMerced
from torchvision.transforms import Normalize

MEAN = torch.tensor(
    [122.60405554, 124.08536424, 114.21736564]
)

STD = torch.tensor(
    [55.80735538, 51.78990456, 50.0009605]
)


class UCMMinimal(LightningDataModule):
    def get_preprocess(self, method="divide"):
        normalize_all = Normalize(mean=MEAN, std=STD)

        def preprocess_normal(sample):
            sample["image"] = sample["image"].float() / 255.0
            return sample

        def preprocess_standardization(sample):
            sample["image"] = normalize_all(sample["image"].float())
            return sample

        if method == "standardization":
            return preprocess_standardization
        elif method == "minmax" or method == "divide":
            return preprocess_normal
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
        use_both_trainval=False,
    ):
        """DataModule for small UCM experiments.

        Options for *normalization_method*: divide, standardization, minmax
        """
        self.root = root
        self.normalization_method = normalization_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_both_trainval = use_both_trainval

    def setup(self):
        preprocess = self.get_preprocess(
            method=self.normalization_method
        )
        ds_train = UCMerced(
            root=self.root,
            split="train",
            transforms=preprocess,
        )
        ds_val = UCMerced(
            root=self.root,
            split="val",
            transforms=preprocess,
        )
        ds_test = UCMerced(
            root=self.root,
            split="test",
            transforms=preprocess,
        )
        if self.use_both_trainval:
            ds_train = ds_train + ds_val
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
