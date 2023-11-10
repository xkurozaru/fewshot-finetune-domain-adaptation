import glob
import os
import random
import shutil

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


def remove_glob(pathname: str):
    for p in glob.glob(pathname, recursive=True):
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)


class ImageTransform:
    """Image transformation module."""

    def __init__(self, input_size=384, phase="train"):
        if phase == "train":
            self.data_transform = nn.Sequential(
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
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
        elif phase == "test":
            self.data_transform = nn.Sequential(
                T.Resize(input_size),
                T.CenterCrop(input_size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )

    def __call__(self, img):
        return self.data_transform(img)
