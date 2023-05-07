import gc
import json
import os

import h5py
from tqdm import tqdm

if __name__ == "__main__":
    versions = "random", "block", "culture_10"
    root = os.path.join("data", "so2sat")

    stats = {version: {} for version in versions}
    for version in tqdm(versions):
        path = os.path.join(root, version, "training.h5")
        f = h5py.File(path)
        x = f["sen2"][:]
        stats[version]["min"] = x.min(axis=(0, 1, 2)).tolist()
        stats[version]["max"] = x.max(axis=(0, 1, 2)).tolist()
        stats[version]["mean"] = x.mean(axis=(0, 1, 2)).tolist()
        stats[version]["std"] = x.std(axis=(0, 1, 2)).tolist()
        f.close()
        del x
        gc.collect()

    with open("so2sat_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
