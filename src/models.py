from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from torchgeo.datasets import NonGeoDataset
from torchgeo.models import RCF, ResNet18_Weights, ResNet50_Weights
from torchgeo.models.api import get_model
from torchvision.models.feature_extraction import create_feature_extractor


class ImageStatisticsModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat(
            [
                torch.mean(x, dim=(2, 3)),
                torch.std(x, dim=(2, 3)),
                torch.amax(x, dim=(2, 3)),
                torch.amin(x, dim=(2, 3)),
            ],
            dim=1,
        )


class MOSAIKS(RCF):
    """This model extracts MOSAIKS features from its input.
    MOSAIKS features are described in Multi-task Observation using Satellite Imagery &
    Kitchen Sinks https://www.nature.com/articles/s41467-021-24638-z. Briefly, this
    model is instantiated with a dataset, samples patches from the dataset, ZCA whitens
    the patches, then uses those as convolutional filters to extract features with.
    .. note::
        This Module is *not* trainable. It is only used as a feature extractor.
    """

    def _normalize(
        self,
        patches: "np.typing.NDArray[np.float32]",
        min_divisor: float = 1e-8,
        zca_bias: float = 0.001,
    ) -> "np.typing.NDArray[np.float32]":
        """Does ZCA whitening on a set of input patches.
        Copied from https://github.com/Global-Policy-Lab/mosaiks-paper/blob/7efb09ed455505562d6bb04c2aaa242ef59f0a82/code/mosaiks/featurization.py#L120
        Args:
            patches: a numpy array of size (N, C, H, W)
            min_divisor: a small number to guard against division by zero
            zca_bias: bias term for ZCA whitening
        Returns
            a numpy array of size (N, C, H, W) containing the normalized patches
        """  # noqa: E501
        n_patches = patches.shape[0]
        orig_shape = patches.shape
        patches = patches.reshape(patches.shape[0], -1)

        # Zero mean every feature
        patches = patches - np.mean(patches, axis=1, keepdims=True)

        # Normalize
        patch_norms = np.linalg.norm(patches, axis=1)

        # Get rid of really small norms
        patch_norms[np.where(patch_norms < min_divisor)] = 1

        # Make features unit norm
        patches = patches / patch_norms[:, np.newaxis]

        patchesCovMat = 1.0 / n_patches * patches.T.dot(patches)

        (E, V) = np.linalg.eig(patchesCovMat)

        E += zca_bias
        sqrt_zca_eigs = np.sqrt(E)
        inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
        global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
        patches_normalized: "np.typing.NDArray[np.float32]" = (
            (patches).dot(global_ZCA).dot(global_ZCA.T)
        )

        return patches_normalized.reshape(orig_shape).astype("float32")

    def __init__(
        self,
        dataset: NonGeoDataset,
        in_channels: int = 4,
        features: int = 16,
        kernel_size: int = 3,
        bias: float = -1.0,
        seed: Optional[int] = None,
    ):
        """Initializes the MOSAIKS model.
        This is the main model used in Multi-task Observation using Satellite Imagery
        & Kitchen Sinks (MOSAIKS) method proposed in
        https://www.nature.com/articles/s41467-021-24638-z.
        Args:
            dataset: a torch dataset that returns dictionaries with an "image" key
            in_channels: number of input channels
            features: number of features to compute, must be divisible by 2
            kernel_size: size of the kernel used to compute the RCFs
            bias: bias of the convolutional layer
            seed: random seed used to initialize the convolutional layer
        .. versionadded:: 0.5.0
        """
        super().__init__(in_channels, features, kernel_size, bias, seed)

        # sample dataset patches
        generator = np.random.default_rng(seed=seed)
        num_patches = features // 2
        num_channels, height, width = dataset[0]["image"].shape
        assert num_channels == in_channels

        patches = np.zeros(
            (num_patches, num_channels, kernel_size, kernel_size), dtype=np.float32
        )
        for i in range(num_patches):
            idx = generator.integers(0, len(dataset))
            img = dataset[idx]["image"]
            y = generator.integers(0, height - kernel_size)
            x = generator.integers(0, width - kernel_size)
            patches[i] = img[:, y : y + kernel_size, x : x + kernel_size]

        patches = self._normalize(patches)
        self.weights = torch.tensor(patches)


def get_model_by_name(model_name, rgb=True, device="cuda", dataset=None, seed=None):
    if model_name == "resnet50_pretrained_seco":
        if not rgb:
            raise ValueError("SeCo weights only support RGB")
        model = get_model("resnet50", weights=ResNet50_Weights.SENTINEL2_RGB_SECO)
        model = create_feature_extractor(model, return_nodes=["global_pool"])
    elif model_name == "resnet18_pretrained_seco":
        if not rgb:
            raise ValueError("SeCo weights only support RGB")
        model = get_model("resnet18", weights=ResNet18_Weights.SENTINEL2_RGB_SECO)
        model = create_feature_extractor(model, return_nodes=["global_pool"])
    elif model_name == "resnet50_pretrained_moco":
        if rgb:
            model = get_model("resnet50", weights=ResNet50_Weights.SENTINEL2_RGB_MOCO)
        else:
            model = get_model("resnet50", weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)
        model = create_feature_extractor(model, return_nodes=["global_pool"])
    elif model_name == "resnet18_pretrained_moco":
        if rgb:
            model = get_model("resnet18", weights=ResNet18_Weights.SENTINEL2_RGB_MOCO)
        else:
            model = get_model("resnet18", weights=ResNet18_Weights.SENTINEL2_ALL_MOCO)
        model = create_feature_extractor(model, return_nodes=["global_pool"])
    elif model_name == "resnet50_pretrained_imagenet":
        if rgb:
            model = timm.create_model("resnet50", in_chans=3, pretrained=True)
        else:
            model = timm.create_model("resnet50", in_chans=13, pretrained=True)
        model = create_feature_extractor(model, return_nodes=["global_pool"])
    elif model_name == "resnet50_randominit":
        if rgb:
            model = timm.create_model("resnet50", in_chans=3, pretrained=False)
        else:
            model = timm.create_model("resnet50", in_chans=13, pretrained=False)
        model = create_feature_extractor(model, return_nodes=["global_pool"])
    elif model_name == "imagestats":
        model = ImageStatisticsModel()
    elif model_name == "mosaiks_512_3":
        if rgb:
            model = RCF(in_channels=3, features=512, kernel_size=3, seed=seed)
        else:
            model = RCF(in_channels=13, features=512, kernel_size=3, seed=seed)
    elif model_name == "mosaiks_zca_512_3":
        if rgb:
            model = MOSAIKS(
                dataset, in_channels=3, features=512, kernel_size=3, seed=seed
            )
        else:
            model = MOSAIKS(
                dataset, in_channels=13, features=512, kernel_size=3, seed=seed
            )
    else:
        raise ValueError(f"{model_name} is invalid")

    model = model.eval().to(device)
    return model
