import argparse
import os
import glob

import rasterio
import kornia.augmentation as K
import lightning
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets.folder import is_image_file
from tqdm import tqdm

from src.models import get_model_by_name
from src.transforms import sentinel2_transforms, ssl4eo_transforms


class DatasetFolder(torch.utils.data.Dataset):
    # YOUR DATASET MEAN/STD STATS GO HERE
    rgb_mean, rgb_std = torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0])
    msi_mean, msi_std = torch.zeros(13), torch.ones(13)

    norm_rgb = K.Normalize(mean=rgb_mean, std=rgb_std)
    norm_msi = K.Normalize(mean=msi_mean, std=msi_std)

    rgb_bands = (3, 2, 1)

    def __init__(self, root, rgb=True, transforms=None):
        self.root = root
        self.rgb = rgb
        self.transforms = transforms
        self.images = sorted(glob.glob(os.path.join(root, "*")))
        self.images = [path for path in self.images if is_image_file(path)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with rasterio.open(path) as f:
            image = torch.from_numpy(f.read().astype(float)).to(torch.float)
            if self.rgb:
                image = image[self.rgb_bands, ...]

        if self.transforms is not None:
            image = self.transforms(image)

        return image


@torch.no_grad()
@torch.inference_mode()
def extract_features(model, dataloader, device, transforms):
    x = []

    for images in tqdm(dataloader, total=len(dataloader)):
        if transforms is not None:
            images = transforms(images)

            features = model(images.to(device))
            if isinstance(features, torch.Tensor):
                features = features.cpu()
            else:
                if "norm" in features:
                    features = features["norm"].cpu()
                else:
                    features = features["global_pool"].cpu()

        x.append(features)

    x = torch.cat(x, dim=0).numpy()
    return x


def main(args):
    lightning.seed_everything(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = DatasetFolder(root=args.root, rgb=args.rgb)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    if "mosaiks_zca" in args.model:
        model = get_model_by_name(
            args.model, args.rgb, device=device, dataset=dataset, seed=args.seed
        )
    else:
        model = get_model_by_name(
            args.model, args.rgb, device=device, dataset=None, seed=args.seed
        )

    if args.model == "imagestats":
        transforms = [nn.Identity()]
    elif "moco" in args.model:
        transforms = [K.Resize(args.image_size), *ssl4eo_transforms()]
    elif "imagenet" in args.model:
        if args.rgb:
            transforms = [K.Resize(args.image_size), *sentinel2_transforms(), dataset.norm_rgb]
        else:
            transforms = [K.Resize(args.image_size), *sentinel2_transforms(), dataset.norm_msi]
    else:
        transforms = [K.Resize(args.image_size), *sentinel2_transforms()]

    transforms = nn.Sequential(*transforms).to(device)

    x = extract_features(model, dataloader, device, transforms)

    filename = os.path.join(args.output_dir, f"{args.model}_features.npy")
    np.save(filename, x)


if __name__ == "__main__":
    model_names = [
        "resnet50_pretrained_moco",
        "imagestats",
        "resnet50_pretrained_imagenet",
        "resnet50_randominit",
        "mosaiks_512_3",
        "mosaiks_zca_512_3",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="resnet50_pretrained_moco", choices=model_names)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
