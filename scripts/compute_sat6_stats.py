import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("..")
from src.datasets import SAT6

ROOT = os.path.join("../data", "sat6")
NUM_WORKERS = 8
BATCH_SIZE = 32

if __name__ == "__main__":
    dataset = SAT6(root=ROOT, split="train", bands=SAT6.all_bands)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    stats = {}
    x = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        x.append(batch["image"])

    x = torch.cat(x, dim=0).to(torch.float)
    stats["min"] = x.amin(dim=(0, 2, 3)).numpy().tolist()
    stats["max"] = x.amax(dim=(0, 2, 3)).numpy().tolist()
    stats["mean"] = x.mean(dim=(0, 2, 3)).numpy().tolist()
    stats["std"] = x.std(dim=(0, 2, 3)).numpy().tolist()

    with open("sat6.json", "w") as f:
        json.dump(stats, f, indent=2)
