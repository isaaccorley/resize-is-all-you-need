import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
from itertools import product
from pprint import pprint

import kornia.augmentation as K
import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append("..")
from src.datasets import SAT6DataModule
from src.models import get_model_by_name
from src.transforms import uint8_transforms
from src.utils import extract_features


def main(args):
    pl.seed_everything(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.directory, exist_ok=True)

    # Fit
    if args.seed == 0:
        model_names = [
            "resnet50_pretrained_moco",
            "imagestats",
            "resnet50_pretrained_imagenet",
            "resnet50_randominit",
            "mosaiks_512_3",
            "mosaiks_zca_512_3",
        ]
    else:
        model_names = ["resnet50_randominit", "mosaiks_512_3", "mosaiks_zca_512_3"]
    rgbs = [True]
    sizes = [34, 224]

    for model_name, rgb, size in product(model_names, rgbs, sizes):
        run = f"{model_name}{'_rgb' if rgb else ''}_{size}"
        print(f"Extracting features for {run}")

        # Skip if features were already extracted
        if os.path.exists(os.path.join(args.directory, f"{run}.npz")):
            continue

        if model_name == "imagestats" and size == 224:
            continue

        dm = SAT6DataModule(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.workers,
            seed=args.seed,
        )
        dm.setup()

        if "mosaiks_zca" in model_name:
            model = get_model_by_name(
                model_name, rgb, device=device, dataset=dm.train_dataset, seed=args.seed
            )
        else:
            model = get_model_by_name(
                model_name, rgb, device=device, dataset=None, seed=args.seed
            )

        if model_name == "imagestats":
            transforms = [nn.Identity()]
        elif "moco" in model_name:
            transforms = [K.Resize(size), *uint8_transforms()]
        elif "imagenet" in model_name:
            transforms = [K.Resize(size), *uint8_transforms(), dm.norm_rgb]
        else:
            transforms = [K.Resize(size), *uint8_transforms()]

        transforms = nn.Sequential(*transforms).to(device)

        x_train, y_train = extract_features(
            model, dm.train_dataloader(), device, transforms=transforms
        )
        x_test, y_test = extract_features(
            model, dm.test_dataloader(), device, transforms=transforms
        )

        filename = os.path.join(args.directory, f"{run}.npz")
        np.savez(
            filename,
            x_train=x_train,
            y_train=y_train.astype(np.int16),
            x_test=x_test,
            y_test=y_test.astype(np.int16),
        )
        del model, dm, transforms
        torch.cuda.empty_cache()
        gc.collect()

    # Eval
    output = os.path.join(args.directory, "sat6-results.json")
    if not os.path.exists(output):
        with open(output, "w") as f:
            json.dump({}, f, indent=2)

    for model_name, rgb, size in product(model_names, rgbs, sizes):
        with open(output) as f:
            results = json.load(f)

        if model_name == "imagestats" and size == 224:
            continue

        run = f"{model_name}{'_rgb' if rgb else ''}_{size}"
        print(f"Evaluating {run}")

        if run in results:
            continue

        filename = os.path.join(args.directory, f"{run}.npz")
        if not os.path.exists(filename):
            continue

        data = np.load(filename)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        if (
            model_name == "imagestats"
            or "random" in model_name
            or model_name.startswith("mosaiks")
        ):
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        if args.faiss:
            from src.knn import FaissKNNClassifier

            knn_model = FaissKNNClassifier(n_neighbors=args.k, device=args.device)
        else:
            knn_model = KNeighborsClassifier(
                n_neighbors=args.k, algorithm="brute", n_jobs=args.workers
            )

        knn_model.fit(X=x_train, y=y_train)
        y_pred = knn_model.predict(x_test)

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

        del knn_model

    # Convert to csv
    with open(output) as f:
        results = json.load(f)

    df = pd.DataFrame.from_dict(results).transpose()
    df["rgb"] = ["RGB" if "rgb" in model_name else "MSI" for model_name in df.index]
    df["size"] = [int(model_name.split("_")[-1]) for model_name in df.index]
    df["encoder"] = [
        model_name.rsplit("_", 1)[0].replace("_rgb", "") for model_name in df.index
    ]
    df = df.sort_values(["rgb", "encoder", "size"], ascending=True)
    df.to_csv(output.replace(".json", ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../data/sat6/")
    parser.add_argument("--directory", type=str, default="sat6")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faiss", action="store_true")
    args = parser.parse_args()
    args.directory = f"{args.directory}_{args.seed}"
    main(args)
