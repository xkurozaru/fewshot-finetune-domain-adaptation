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


class DivideDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform, device: torch.device = torch.device("cuda")):
        self.classes = [d.name for d in os.scandir(root) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = make_class_to_idx(root)
        self.device = device

        self.items = {c: [] for c in self.classes}
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(root, target)
            if not os.path.isdir(d):
                continue
            for dir, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(dir, fname)
                    item = path
                    self.items[target].append(item)

        self.transform = transform

    def get_image_on_device(self, path):
        img = read_image(path).to(self.device) / 255.0
        img = self.transform(img)
        return img

    def get_all_images_by_label(self, label):
        images = []
        target = self.classes[label]
        for path in self.items[target]:
            img = self.get_image_on_device(path)
            images.append(img)

        true = []
        for _ in range(len(images)):
            true.append(label)

        return torch.stack(images), torch.tensor(true).to(self.device)

    def get_random_images_by_labels(self, labels):
        images = []
        targets = [self.classes[label] for label in labels]
        for target in targets:
            path = random.choice(self.items[target])
            img = self.get_image_on_device(path)
            images.append(img)

        true = []
        for target in targets:
            true.append(self.classes.index(target))

        return torch.stack(images), torch.tensor(true).to(self.device)

    def get_random_notequal_images_by_labels(self, labels):
        images = []
        targets = []
        for label in labels:
            removed = self.classes.copy()
            removed.remove(self.classes[label])
            target = random.choice(removed)
            targets.append(target)
        for target in targets:
            path = random.choice(self.items[target])
            img = self.get_image_on_device(path)
            images.append(img)

        true = []
        for target in targets:
            true.append(self.classes.index(target))

        return torch.stack(images), torch.tensor(true).to(self.device)


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
