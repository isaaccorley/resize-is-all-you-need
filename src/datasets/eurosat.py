import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import EuroSAT
from torchvision.transforms import Normalize

MAX = torch.tensor(
    [17720.0, 28000.0, 28000.0, 28000.0, 23381.0, 27791.0, 28001.0, 28002.0, 15384.0, 183.0, 24704.0, 22210.0, 28000.0]
).unsqueeze(1).unsqueeze(1)

MIN = torch.tensor(
    [816.0, 0.0, 0.0, 0.0, 174.0, 153.0, 128.0, 0.0, 40.0, 1.0, 5.0, 1.0, 91.0]
).unsqueeze(1).unsqueeze(1)

MEAN = torch.tensor(
    [
        1354.40546513,
        1118.24399958,
        1042.92983953,
        947.62620298,
        1199.47283961,
        1999.79090914,
        2369.22292565,
        2296.82608323,
        732.08340178,
        12.11327804,
        1819.01027855,
        1118.92391149,
        2594.14080798,
    ]
)

STD = torch.tensor(
    [
        245.71762908,
        333.00778264,
        395.09249139,
        593.75055589,
        566.4170017,
        861.18399006,
        1086.63139075,
        1117.98170791,
        404.91978886,
        4.77584468,
        1002.58768311,
        761.30323499,
        1231.58581042,
    ]
)

IMAGENET_NORM = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class EuroSATMinimal(LightningDataModule):

    def get_preprocess(self, rgb=True, method="divide"):
        normalize_all = Normalize(mean=MEAN, std=STD)
        normalize_rgb = Normalize(mean=MEAN[[3,2,1]], std=STD[[3,2,1]])

        def preprocess_normal(sample):
            sample["image"] = sample["image"].float() / 10000.0
            return sample

        def preprocess_standardization(sample):
            if rgb:
                sample["image"] = normalize_rgb(sample["image"].float())
            else:
                sample["image"] = normalize_all(sample["image"].float())
            return sample

        def preprocess_min_max(sample):
            if rgb:
                sample["image"] = (sample["image"].float() - MIN[[3,2,1]]) / (MAX[[3,2,1]] - MIN[[3,2,1]])
            else:
                sample["image"] = (sample["image"].float() - MIN) / (MAX - MIN)
            return sample

        def preprocess_min_max_imagenet(sample):
            if rgb:
                sample["image"] = (sample["image"].float() - MIN[[3,2,1]]) / (MAX[[3,2,1]] - MIN[[3,2,1]])
                sample["image"] = IMAGENET_NORM(sample["image"])
            else:
                raise ValueError("Method not supported")
            return sample

        if method == "divide":
            return preprocess_normal
        elif method == "standardization":
            return preprocess_standardization
        elif method == "minmax":
            return preprocess_min_max
        elif method == "minmax_imagenet":
            return preprocess_min_max_imagenet
        else:
            raise ValueError("Method not supported")

    def __init__(self, root, band_set="rgb", normalization_method="divide", batch_size=32, num_workers=8, seed=0):
        """DataModule for small EuroSAT experiments.

        ***NOTE***: this uses random 90/10 splits instead of the torchgeo splits.

        Options for *band_set*: all, rgb
        Options for *normalization_method*: divide, standardization, minmax
        """
        self.root = root
        self.band_set = band_set
        self.normalization_method = normalization_method
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self):
        preprocess = self.get_preprocess(rgb=self.band_set == "rgb", method=self.normalization_method)
        ds_train = EuroSAT(
            root=self.root,
            split="train",
            bands=EuroSAT.BAND_SETS[self.band_set],
            transforms=preprocess,
        )
        ds_val = EuroSAT(
            root=self.root,
            split="val",
            bands=EuroSAT.BAND_SETS[self.band_set],
            transforms=preprocess,
        )
        ds_test = EuroSAT(
            root=self.root,
            split="test",
            bands=EuroSAT.BAND_SETS[self.band_set],
            transforms=preprocess,
        )
        # ds_all = ds_train + ds_val + ds_test
        # self.train_dataset, self.test_dataset = random_split(
        #     ds_all, [0.8, 0.2], generator=self.generator
        # )
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
