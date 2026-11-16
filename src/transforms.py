import torch
import torch.nn.functional as F

from stable_pretraining import data as dt
from torchvision.transforms import v2 as transforms

from typing import Optional 

imagenet_stats = dt.dataset_stats.ImageNet
IMAGENET_MEAN = torch.tensor(imagenet_stats["mean"]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(imagenet_stats["std"]).view(1, 3, 1, 1)

# IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
# IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def img_transform(image_size: int):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**dt.dataset_stats.ImageNet, inplace=False),
            transforms.Resize(size=image_size),
        ]
    )
    return transform

# def preprocess_pixels(pixels: torch.Tensor, image_size: int) -> torch.Tensor:
#     """(B, T, H, W, 3) uint8 -> (B, T, 3, S, S) ImageNet-normalized float, on pixels' device."""
#     B, T = pixels.shape[:2]
#     x = pixels.flatten(0, 1).permute(0, 3, 1, 2).float().div_(255)
#     if x.shape[-1] != image_size or x.shape[-2] != image_size:
#         x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", antialias=True, align_corners=False)
#     x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
#     return x.view(B, T, *x.shape[1:])

def preprocess_pixels(pixels: torch.Tensor, image_size: int) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 -> (B, T, 3, S, S) ImageNet-normalized float, on pixels' device."""
    pixels = pixels.permute(0, 1, 4, 2, 3)
    return img_transform(image_size)(pixels)

def preprocess_actions(
        actions: torch.Tensor, 
        fillin_with: float, 
        action_mean: Optional[torch.Tensor] = None, 
        action_std: Optional[torch.Tensor] = None
):
    actions.nan_to_num(fillin_with)

    if action_mean is not None and action_std is not None:
        actions = (actions - action_mean) / action_std

    return actions