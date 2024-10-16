import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torchinfo import summary
from tqdm import tqdm

from common import param
from common.utils import Dataset, ImageTransform, set_seed
from module.efficient_net import EfficientNetEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu_ids
warnings.filterwarnings("ignore")

BATCH_SIZE = 128


def main():
    set_seed(param.seed)
    device = torch.device("cuda")

    dataset = Dataset(
        root=param.src_path,
        transform=ImageTransform(input_size=480, phase="test"),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=True)

    testset = Dataset(
        root=param.test_path,
        transform=ImageTransform(input_size=480, phase="test"),
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)

    encoder = EfficientNetEncoder().to(device)
    encoder.load_state_dict(torch.load("weights/triplet_tune_encoder.pth"))
    encoder = nn.DataParallel(encoder)

    summary(model=encoder.module, input_size=(BATCH_SIZE, 3, 480, 480))

    encoder.eval()

    # numpy形式で特徴量を保存
    X = np.empty((0, 1280))
    y = []
    print("Start embedding...")
    with torch.inference_mode():
        for inputs, labels, _ in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = encoder(inputs)
            X = np.append(X, outputs.cpu().detach().numpy(), axis=0)
            y.extend(labels.tolist())
            if len(y) >= len(testset):
                break
    print("Finish embedding!")

    X_test = np.empty((0, 1280))
    y_test = []
    print("Start embedding...")
    with torch.inference_mode():
        for inputs, labels, _ in tqdm(testloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = encoder(inputs)
            X_test = np.append(X_test, outputs.cpu().detach().numpy(), axis=0)
            y_test.extend(labels.tolist())
    print("Finish embedding!")

    # yの値をクラス名に変換
    idx2class = {v: k for k, v in dataset.class_to_idx.items()}
    y = [idx2class[i] + "-S" for i in y]
    y_test = [idx2class[i] + "-T" for i in y_test]

    # t-SNEで次元削減
    print("Start t-SNE...")
    features = np.concatenate([X, X_test], axis=0)
    targets = y + y_test
    reducer = TSNE(n_components=2, random_state=42)
    embeddings = reducer.fit_transform(features)
    print("Finish t-SNE!")

    # targetsで色分けしてプロット
    print("Start plotting...")
    plt.figure(figsize=(16, 16))
    plotclasses = list(set(targets))
    plotclasses.sort()

    for i, target in enumerate(plotclasses):
        indices = [i for i, x in enumerate(targets) if x == target]
        if "-S" in target:
            maker = "o"
            coler = plt.get_cmap("gist_rainbow")(i / len(plotclasses))
            size = 20
        elif "-T" in target:
            maker = "x"
            coler = plt.get_cmap("gist_rainbow")(i / len(plotclasses))
            size = 20
        plt.scatter(
            embeddings[indices, 0],
            embeddings[indices, 1],
            s=size,
            c=coler,
            marker=maker,
            label=target,
            alpha=0.5,
        )
    plt.legend()
    plt.axis("off")
    plt.savefig("results/embeddings.png")
    print("Finish plotting!")


if __name__ == "__main__":
    main()
