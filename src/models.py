import timm
import torch
import torch.nn as nn
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


def get_model_by_name(model_name, rgb=True, device="cuda"):
    if model_name == "resnet50_pretrained_moco":
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
            model = RCF(in_channels=3, features=512, kernel_size=3, seed=0)
        else:
            model = RCF(in_channels=13, features=512, kernel_size=3, seed=0)
    else:
        raise ValueError(f"{model_name} is invalid")

    model = model.eval().to(device)
    return model
