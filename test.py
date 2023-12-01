import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchinfo import summary
from tqdm import tqdm

from common import Dataset, ImageTransform, param, set_seed
from module import EfficientNetClassifier, EfficientNetEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
warnings.filterwarnings("ignore")

BATCH_SIZE = 256


def main():
    set_seed(42)
    device = torch.device("cuda")
    dataset = Dataset(
        root=param.test_path,
        transform=ImageTransform(phase="test"),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=True)

    encoder = EfficientNetEncoder().to(device)
    classifier = EfficientNetClassifier(num_classes=len(dataset.classes)).to(device)

    encoder.load_state_dict(torch.load("weight/finetune_encoder.pth_epoch_100"))
    classifier.load_state_dict(torch.load("weight/finetune_classifier.pth_epoch_100"))

    encoder = nn.DataParallel(encoder)
    classifier = nn.DataParallel(classifier)

    summary(model=encoder.module, input_size=(BATCH_SIZE, 3, 480, 480))
    summary(model=classifier.module, input_size=(BATCH_SIZE, 1280))

    encoder.eval()
    classifier.eval()

    predict_labels = []
    true_labels = []

    print("Start testing...")
    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = classifier(encoder(inputs))
            _, preds = torch.max(outputs, 1)

            predict_labels.extend(preds.tolist())
            true_labels.extend(labels.tolist())
    print("Finish testing!")

    # snsのフォントサイズを設定
    sns.set(font_scale=2.5)

    # classification reportをseabornで表示して画像保存
    report = classification_report(
        y_true=true_labels,
        y_pred=predict_labels,
        target_names=dataset.classes,
        digits=3,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose() * 100.0
    report_df = report_df.drop("support", axis=1)

    plt.figure(figsize=(20, 15))
    sns.heatmap(
        report_df.iloc[:-1, :].astype(float),
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar=False,
    )
    plt.savefig("result/report.png")

    # confusion matrixをseabornで表示して画像保存
    cm = confusion_matrix(y_true=true_labels, y_pred=predict_labels, normalize="true")
    cm_df = pd.DataFrame(cm, index=dataset.classes, columns=dataset.classes) * 100.0

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_df, annot=True, fmt=".1f", cmap="BuGn", square=True, cbar=False)
    plt.savefig("result/confusion_matrix.png")


if __name__ == "__main__":
    main()
