import os

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.datasets.eurosat import EuroSATMinimal
from src.models import get_model_by_name
from src.utils import extract_features


def main():
    os.makedirs("results/", exist_ok=True)
    device = torch.device("cuda:0")
    model_names = [
        "resnet50_pretrained_moco",
        "resnet50_pretrained_imagenet",
        "resnet50_randominit",
        "resnet18_pretrained_moco",
    ]
    for rgb in [True, False]:
        for model_name in model_names:
            model = get_model_by_name(model_name, rgb=rgb, device="cuda:0")

            results = []
            sizes = list(range(32, 256 + 1, 16))
            for size in tqdm(sizes):
                transforms = nn.Sequential(K.Resize(size)).to(device)

                dm = EuroSATMinimal(
                    root="data/eurosat/",
                    band_set="rgb" if rgb else "all",
                    batch_size=64,
                    num_workers=8,
                )
                dm.setup()

                x_train, y_train = extract_features(
                    model,
                    dm.train_dataloader(),
                    device,
                    transforms=transforms,
                    verbose=False,
                )
                x_test, y_test = extract_features(
                    model,
                    dm.test_dataloader(),
                    device,
                    transforms=transforms,
                    verbose=False,
                )

                knn_model = KNeighborsClassifier(n_neighbors=5)
                knn_model.fit(x_train, y_train)
                acc = knn_model.score(x_test, y_test)

                results.append(acc)
            results = np.array(results)

            np.save("results/eurosat_size_vs_performance_sizes.npy", sizes)
            np.save(f"results/eurosat_size_vs_performance-{model_name}-{rgb}.npy", results)


if __name__ == "__main__":
    main()
