from src.models import get_model_by_name
from src.datasets.eurosat import EuroSATMinimal
from src.utils import extract_features
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import kornia.augmentation as K
import numpy as np

NUM_REPEATS = 10

def main():
    device = torch.device("cuda:0")

    model = get_model_by_name("resnet18_pretrained_moco", device="cuda:0")

    results = []
    sizes = list(range(32,256+1,32))
    for size in tqdm(sizes):
        transforms = nn.Sequential(
            K.Resize(size)
        ).to(device)

        t_results = []
        for i in range(NUM_REPEATS):
            dm = EuroSATMinimal(
                root="data/eurosat/",
                band_set="rgb",
                batch_size=64,
                num_workers=8,
                seed=i,
            )
            dm.setup()

            x_train, y_train = extract_features(model, dm.train_dataloader(), device, transforms=transforms, verbose=False)
            x_test, y_test = extract_features(model, dm.test_dataloader(), device, transforms=transforms, verbose=False)

            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(x_train, y_train)
            acc = knn_model.score(x_test, y_test)

            t_results.append(acc)
        results.append(t_results)
    results = np.array(results)
    
    np.save("eurosat_size_vs_performance_sizes.npy", sizes)
    np.save("eurosat_size_vs_performance.npy", results)



if __name__ == "__main__":
    main()