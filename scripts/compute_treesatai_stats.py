import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("..")
from src.datasets import TreeSatAI

ROOT = os.path.join("../data", "treesatai")
NUM_WORKERS = 8
BATCH_SIZE = 32

if __name__ == "__main__":
    dataset = TreeSatAI(root=ROOT, split="train", bands=TreeSatAI.all_bands, size=20)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    stats = {}
    x = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        x.append(batch["image"])

    x = torch.cat(x, dim=0)
    stats["min"] = x.amin(dim=(0, 2, 3)).numpy().tolist()
    stats["max"] = x.amax(dim=(0, 2, 3)).numpy().tolist()
    stats["mean"] = x.mean(dim=(0, 2, 3)).numpy().tolist()
    stats["std"] = x.std(dim=(0, 2, 3)).numpy().tolist()

    with open("treesatai_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
