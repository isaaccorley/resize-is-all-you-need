import os
import sys

sys.path.append("..")

import json
import pickle
from itertools import product
from pprint import pprint

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

from src.datasets import BigEarthNetDataModule
from src.models import get_model_by_name
from src.transforms import (
    imagenet_transforms,
    seco_rgb_transforms,
    sentinel2_transforms,
    ssl4eo_transforms,
)
from src.utils import extract_features

if __name__ == "__main__":
    device = torch.device("cuda")
    directory = "bigearthnet"
    os.makedirs(directory, exist_ok=True)

    # Fit
    model_names = [
        "resnet50_pretrained_moco",
        "imagestats",
        "resnet50_pretrained_seco",
        "resnet50_pretrained_imagenet",
        "resnet50_randominit",
        "mosaiks_512_3",
    ]
    rgbs = [False, True]
    sizes = [120, 224]

    for model_name, rgb, size in product(model_names, rgbs, sizes):
        run = f"{model_name}{'_rgb' if rgb else ''}_{size}"
        print(f"Extracting features for {run}")

        # Skip if features were already exracted
        if os.path.exists(os.path.join(directory, f"{run}.pkl")):
            continue

        # SeCo only supports RGB
        if model_name == "resnet50_pretrained_seco" and not rgb:
            continue
        if model_name == "imagestats" and size == 224:
            continue

        # Pad missing bands
        if rgb:
            bands = "rgb"
            pad_missing_bands = False
        else:
            bands = "all"
            pad_missing_bands = True

        dm = BigEarthNetDataModule(
            root="../data/bigearthnet/",
            bands=bands,
            batch_size=32,
            num_workers=16,
            pad_missing_bands=pad_missing_bands,
            seed=0,
        )
        dm.setup()

        model = get_model_by_name(model_name, rgb, device=device)

        if model_name == "imagestats":
            transforms = [nn.Identity()]
        elif "seco" in model_name:
            transforms = [K.Resize(size), *seco_rgb_transforms()]
        elif "moco" in model_name:
            transforms = [K.Resize(size), *ssl4eo_transforms()]
        else:
            transforms = [K.Resize(size), *sentinel2_transforms()]

        transforms = nn.Sequential(*transforms).to(device)

        x_train, y_train = extract_features(
            model, dm.train_dataloader(), device, transforms=transforms
        )
        x_test, y_test = extract_features(
            model, dm.test_dataloader(), device, transforms=transforms
        )
        data = dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        with open(os.path.join(directory, f"{run}.pkl"), "wb") as f:
            pickle.dump(data, f)

    # Eval
    output = os.path.join(directory, "bigearthnet-results.json")
    if not os.path.exists(output):
        with open(output, "w") as f:
            json.dump({}, f, indent=2)

    K = 5
    for model_name, rgb, size in product(model_names, rgbs, sizes):
        with open(output) as f:
            results = json.load(f)

        # SeCo only supports RGB
        if model_name == "resnet50_pretrained_seco" and not rgb:
            continue
        if model_name == "imagestats" and size == 224:
            continue

        run = f"{model_name}{'_rgb' if rgb else ''}_{size}"
        print(f"Evaluating {run}")

        if run in results:
            continue

        filename = os.path.join(directory, f"{run}.pkl")
        if not os.path.exists(filename):
            continue

        with open(filename, "rb") as f:
            data = pickle.load(f)

        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        knn_model = KNeighborsClassifier(n_neighbors=K, n_jobs=8)
        knn_model.fit(X=x_train, y=y_train)

        y_test_onehot = label_binarize(y_test, classes=np.arange(19))
        y_pred = knn_model.predict(x_test)
        y_score = knn_model.predict_proba(x_test)

        metrics = {
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_micro": f1_score(y_test, y_pred, average="micro"),
            "precision_micro": precision_score(y_test, y_pred, average="micro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_micro": recall_score(y_test, y_pred, average="micro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_test, y_pred),
        }
        pprint(metrics)
        results[run] = metrics

        with open(output, "w") as f:
            json.dump(results, f, indent=2)
