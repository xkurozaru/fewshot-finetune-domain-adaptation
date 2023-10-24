import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T


def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageTransform:
    """Image transformation module."""

    def __init__(self, input_size=512, phase="train"):
        if phase == "train":
            self.data_transform = nn.Sequential(
                T.Resize(800),
                T.RandomResizedCrop(input_size, (0.25, 1.0), (3 / 4, 4 / 3)),
                T.RandomChoice(
                    [
                        T.RandomRotation((0, 0)),
                        T.RandomRotation((90, 90)),
                        T.RandomRotation((180, 180)),
                        T.RandomRotation((270, 270)),
                    ],
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.5),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            )
        elif phase == "test":
            self.data_transform = nn.Sequential(
                T.Resize(input_size),
                T.CenterCrop(input_size),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            )

    def __call__(self, img):
        return self.data_transform(img)
