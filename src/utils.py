import numpy as np
import torch
from tqdm import tqdm


def extract_features(model, dataloader, device, transforms=None, verbose=True):
    x_all = []
    y_all = []

    enumerator = enumerate(dataloader)
    if verbose:
        enumerator = enumerate(tqdm(dataloader, total=len(dataloader)))

    for i, batch in enumerator:
        images = batch["image"].to(device)
        labels = batch["label"].numpy()

        if transforms is not None:
            images = transforms(images)

        with torch.no_grad():
            with torch.inference_mode():
                features = model(images)
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
                else:
                    if "norm" in features:
                        features = features["norm"].detach().cpu().numpy()
                    else:
                        features = features["global_pool"].detach().cpu().numpy()

        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def sparse_to_dense(sparse):
    num_classes = len(sparse)
    num_samples = sparse[0].shape[0]
    dense = np.zeros(shape=(num_samples, num_classes), dtype=np.float32)
    for idx, preds in enumerate(sparse):
        dense[:, idx] = preds[:, 1]
    return dense
