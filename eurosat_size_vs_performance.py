import os

import pandas as pd
import kornia.augmentation as K
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.datasets.eurosat import EuroSATMinimal
from src.models import get_model_by_name
from src.utils import extract_features


def main():
    os.makedirs("results/", exist_ok=True)
    device = torch.device("cuda:0")
    model_names = [
        "resnet50_pretrained_seco",
        "resnet50_pretrained_moco",
        "resnet50_pretrained_imagenet",
        "resnet50_randominit",
    ]
    rows = []
    for preprocess_method in ["divide", "standardization", "minmax", "for_seco"]:
        for rgb in [True, False]:
            for model_name in model_names:

                if "seco" in model_name and not rgb:
                    continue

                model = get_model_by_name(model_name, rgb=rgb, device="cuda:0")
                sizes = list(range(32, 256 + 1, 16))
                for size in tqdm(sizes):
                    transforms = nn.Sequential(K.Resize(size)).to(device)

                    dm = EuroSATMinimal(
                        root="data/eurosat/",
                        band_set="rgb" if rgb else "all",
                        normalization_method=preprocess_method,
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

                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(x_train)
                    x_test = scaler.transform(x_test)

                    knn_model = KNeighborsClassifier(n_neighbors=5)
                    knn_model.fit(x_train, y_train)
                    acc_scaled = knn_model.score(x_test, y_test)

                    rows.append({
                        "model": model_name,
                        "preprocess_method": preprocess_method,
                        "rgb": rgb,
                        "size": size,
                        "acc": acc,
                        "acc_scaled": acc_scaled,
                    })

    df = pd.DataFrame(rows)
    df.to_csv("results/eurosat_size_vs_performance.csv", index=False)


if __name__ == "__main__":
    main()
