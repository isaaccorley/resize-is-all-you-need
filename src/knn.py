"""Modified from https://github.com/scikit-learn-contrib/DESlib/blob/master/deslib/util/faiss_knn_wrapper.py#L19-L203"""
import faiss
import numpy as np


class FaissKNNClassifier:
    def __init__(self, n_neighbors, n_classes=None, device="cpu"):
        self.n_neighbors = n_neighbors
        self.n_classes = n_classes

        if device == "cpu":
            self.cuda = False
            self.device = None
        else:
            self.cuda = True
            if ":" in device:
                self.device = int(device.split(":")[-1])
            else:
                self.device = 0

    def create_index(self, d):
        if self.cuda:
            self.res = faiss.StandardGpuResources()
            self.config = faiss.GpuIndexFlatConfig()
            self.config.device = self.device
            self.index = faiss.GpuIndexFlatL2(self.res, d, self.config)
        else:
            self.index = faiss.IndexFlatL2(d)

    def fit(self, X, y):
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
        self.create_index(X.shape[-1])
        self.index.add(X)
        self.y = y.astype(int)
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        return self

    def __del__(self):
        if hasattr(self, "index"):
            self.index.reset()
            del self.index
        if hasattr(self, "res"):
            self.res.noTempMemory()
            del self.res

    def predict(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, self.n_neighbors)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),
            axis=1,
            arr=class_idx.astype(np.int16),
        )
        preds = np.argmax(counts, axis=1)
        return preds

    def predict_proba(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, self.n_neighbors)
        class_idx = self.y[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes),
            axis=1,
            arr=class_idx.astype(np.int16),
        )

        preds_proba = counts / self.n_neighbors
        return preds_proba


class FaissKNNMultilabelClassifier(FaissKNNClassifier):
    def predict(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, self.n_neighbors)
        class_idx = self.y[idx]

        preds = []
        for i in range(class_idx.shape[-1]):
            class_idx_i = class_idx[..., i]
            counts_i = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=2),
                axis=1,
                arr=class_idx_i.astype(np.int16),
            )
            preds_i = np.argmax(counts_i, axis=1)
            preds.append(preds_i)

        preds = np.stack(preds, axis=1)
        return preds

    def predict_proba(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        _, idx = self.index.search(X, self.n_neighbors)
        class_idx = self.y[idx]

        preds_proba = []
        for i in range(class_idx.shape[-1]):
            class_idx_i = class_idx[..., i]
            counts_i = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=2),
                axis=1,
                arr=class_idx_i.astype(np.int16),
            )
            preds_proba_i = counts_i / self.n_neighbors
            preds_proba_i = preds_proba_i[:, 1]
            preds_proba.append(preds_proba_i)

        preds_proba = np.stack(preds_proba, axis=1)
        return preds_proba
