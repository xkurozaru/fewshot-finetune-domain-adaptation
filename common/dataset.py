import os
import random

import torch
from torchvision.io import read_image


def make_dataset(root: str, class_to_idx: dict):
    items = []
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        for dir, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(dir, fname)
                item = (path, class_to_idx[target])
                items.append(item)

    return items


def make_class_to_idx(root: str):
    classes = [d.name for d in os.scandir(root) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform):
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = make_class_to_idx(root)
        self.items = make_dataset(root, self.class_to_idx)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.items[index]
        img = read_image(path) / 255.0
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.items)


class AnkerDataset(torch.utils.data.Dataset):
    def __init__(self, anker_root: str, sample_root: str, transform, sample_type="positive"):
        self.classes = [d.name for d in os.scandir(anker_root) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = make_class_to_idx(anker_root)

        self.anker_items = make_dataset(anker_root, self.class_to_idx)
        self.sample_items = {c: [] for c in self.classes}
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(sample_root, target)
            if not os.path.isdir(d):
                continue
            for dir, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(dir, fname)
                    item = (path, self.class_to_idx[target])
                    self.sample_items[target].append(item)
        self.transform = transform
        self.sample_type = sample_type

    def get_positive_image(self, target):
        path, target = random.choice(self.sample_items[target])
        img = read_image(path) / 255.0
        img = self.transform(img)
        return img, target

    def get_negative_image(self, target):
        removed = self.classes.copy()
        removed.remove(target)
        path, target = random.choice(self.sample_items[random.choice(removed)])
        img = read_image(path) / 255.0
        img = self.transform(img)
        return img, target

    def __getitem__(self, index):
        anker_path, anker_target = self.anker_items[index]
        anker_img = read_image(anker_path) / 255.0
        anker_img = self.transform(anker_img)

        if self.sample_type == "positive":
            positive_img, positive_target = self.get_positive_image(self.classes[anker_target])
            return (anker_img, anker_target), (positive_img, positive_target)

        elif self.sample_type == "negative":
            negative_img, negative_target = self.get_negative_image(self.classes[anker_target])
            return (anker_img, anker_target), (negative_img, negative_target)

        elif self.sample_type == "both":
            positive_img, positive_target = self.get_positive_image(self.classes[anker_target])
            negative_img, negative_target = self.get_negative_image(self.classes[anker_target])
            return (anker_img, anker_target), (positive_img, positive_target), (negative_img, negative_target)

        else:
            raise ValueError("sample_type must be 'positive', 'negative' or 'both'.")

    def __len__(self):
        return len(self.anker_items)


class DoubleDataset(torch.utils.data.Dataset):
    def __init__(self, src_root: str, tgt_root, transform):
        self.classes = [d.name for d in os.scandir(src_root) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = make_class_to_idx(src_root)

        self.src_items = make_dataset(src_root, self.class_to_idx)
        self.tgt_items = make_dataset(tgt_root, self.class_to_idx)
        self.transform = transform

    def __getitem__(self, index):
        src_path, src_target = self.src_items[index]
        src_img = read_image(src_path) / 255.0
        src_img = self.transform(src_img)

        tgt_path, tgt_target = random.choice(self.tgt_items)
        tgt_img = read_image(tgt_path) / 255.0
        tgt_img = self.transform(tgt_img)

        return (src_img, src_target), (tgt_img, tgt_target)

    def __len__(self):
        return len(self.src_items)
