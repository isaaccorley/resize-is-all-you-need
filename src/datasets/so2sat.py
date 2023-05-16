import kornia.augmentation as K
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
    min = {
        "3_random": torch.tensor(
            [
                0.0,  # padded band
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0,  # padded band
                0.0,  # padded band
                0.0001,
                0.0001,
            ]
        ),
        "3_culture_10": torch.tensor(
            [
                0.0,  # padded band
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0,  # padded band
                0.0,  # padded band
                0.0001,
                0.0001,
            ]
        ),
    }
    max = {
        "3_random": torch.tensor(
            [
                1.0,  # padded band
                2.8,
                2.8,
                2.8,
                2.8001,
                2.8002,
                2.8003,
                2.8002,
                2.8005,
                1.0,  # padded band
                1.0,  # padded band
                2.8001,
                2.8,
            ]
        ),
        "3_culture_10": torch.tensor(
            [
                1.0,  # padded band
                2.8,
                2.8,
                2.8,
                2.8001,
                2.8002,
                2.8003,
                2.8001,
                2.8003,
                1.0,  # padded band
                1.0,  # padded band
                2.8001,
                2.8,
            ]
        ),
    }
    mean = {
        "3_random": torch.tensor(
            [
                0.0,  # padded band
                0.1242865659255651,
                0.11001677360484904,
                0.10230652363788878,
                0.11532195523298688,
                0.15989486016901647,
                0.18204406481475396,
                0.17513562590032622,
                0.19565546642694676,
                0.0,  # padded band
                0.0,  # padded band
                0.15648722648417757,
                0.11122536335888582,
            ]
        ),
        "3_culture_10": torch.tensor(
            [
                0.0,  # padded band
                0.12375696117628263,
                0.10927746363668964,
                0.10108552033058443,
                0.11423986161160338,
                0.15926566920213808,
                0.18147236008864062,
                0.17457403122800244,
                0.19501607349535974,
                0.0,  # padded band
                0.0,  # padded band
                0.15428468872571866,
                0.10905050699494197,
            ]
        ),
    }
    std = {
        "3_random": torch.tensor(
            [
                1.0,  # padded band
                0.03922694991002109,
                0.047091672083101936,
                0.06532640870812462,
                0.06240566539720062,
                0.07583674859568054,
                0.08917173211036335,
                0.09050921895641584,
                0.09968561363966205,
                1.0,  # padded band
                1.0,  # padded band
                0.09901878556417713,
                0.08733859110756595,
            ]
        ),
        "3_culture_10": torch.tensor(
            [
                1.0,  # padded band
                0.03958795985744495,
                0.0477782627510975,
                0.06636616706379514,
                0.06358874912517777,
                0.0774438714841826,
                0.0910163508568612,
                0.09218466562281988,
                0.10164581234627947,
                1.0,  # padded band
                1.0,  # padded band
                0.0999177304257295,
                0.08780632508691358,
            ]
        ),
    }

    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float()  # values are already in [0, 1]
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
        self.norm_rgb = K.Normalize(
            mean=self.mean[version][[3, 2, 1]], std=self.std[version][[3, 2, 1]]
        )
        self.norm_msi = K.Normalize(mean=self.mean[version], std=self.std[version])

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
