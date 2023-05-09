import argparse
import multiprocessing as mp
import os
import sys

sys.path.append("..")

import json
import pickle
from itertools import product
from pprint import pprint

import kornia.augmentation as K
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.datasets import TreeSatAI, TreeSatAIDataModule
from src.models import get_model_by_name
from src.transforms import seco_rgb_transforms, sentinel2_transforms, ssl4eo_transforms
from src.utils import extract_features, sparse_to_dense


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.directory, exist_ok=True)

    # Fit
    model_names = [
        "resnet50_pretrained_moco",
        "imagestats",
        "resnet50_pretrained_imagenet",
        "resnet50_randominit",
        "mosaiks_512_3",
        # "resnet50_pretrained_seco",
    ]
    rgbs = [False, True]
    sizes = [34, 224]

    for model_name, rgb, size in product(model_names, rgbs, sizes):
        run = f"{model_name}{'_rgb' if rgb else ''}_{size}"
        print(f"Extracting features for {run}")

        # Skip if features were already extracted
        if os.path.exists(os.path.join(args.directory, f"{run}.pkl")):
            continue

        # SeCo only supports RGB
        if model_name == "resnet50_pretrained_seco" and not rgb:
            continue
        if model_name == "imagestats" and size == 224:
            continue

        # Pad missing bands
        if rgb:
            bands = TreeSatAI.rgb_bands
            pad_missing_bands = False
        else:
            bands = TreeSatAI.correct_band_order
            pad_missing_bands = True

        dm = TreeSatAIDataModule(
            root="../data/treesatai/",
            bands=bands,
            multilabel=True,
            size=20,
            batch_size=args.batch_size,
            num_workers=args.workers,
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
        with open(os.path.join(args.directory, f"{run}.pkl"), "wb") as f:
            pickle.dump(data, f)

    # Eval
    output = os.path.join(args.directory, "treesatai-results.json")
    if not os.path.exists(output):
        with open(output, "w") as f:
            json.dump({}, f, indent=2)

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

        filename = os.path.join(args.directory, f"{run}.pkl")
        if not os.path.exists(filename):
            continue

        with open(filename, "rb") as f:
            data = pickle.load(f)

        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        if model_name == "imagestats":
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        knn_model = KNeighborsClassifier(n_neighbors=args.k, n_jobs=args.workers)
        knn_model.fit(X=x_train, y=y_train)

        y_pred = knn_model.predict(x_test)
        y_score = knn_model.predict_proba(x_test)
        score = sparse_to_dense(y_score)

        metrics = {
            "map_weighted": average_precision_score(y_test, score, average="weighted"),
            "map_macro": average_precision_score(y_test, score, average="macro"),
            "map_micro": average_precision_score(y_test, score, average="micro"),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="treesatai")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
