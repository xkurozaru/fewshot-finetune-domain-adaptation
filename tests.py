import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from common import Dataset, ImageTransform, param, set_seed
from module import EfficientNetClassifier, EfficientNetEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu_ids
warnings.filterwarnings("ignore")

BATCH_SIZE = 256


def main():
    set_seed(param.seed)
    device = torch.device("cuda")
    dataset = Dataset(
        root=param.test_path,
        transform=ImageTransform(phase="test"),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=True)

    for epoch in range(100, 1001, 100):
        encoder = EfficientNetEncoder().to(device)
        classifier = EfficientNetClassifier(num_classes=len(dataset.classes)).to(device)
        encoder.load_state_dict(torch.load(f"weight/dist_tune_encoder.pth_epoch_{epoch}"))
        classifier.load_state_dict(torch.load(f"weight/dist_tune_classifier.pth_epoch_{epoch}"))
        encoder = nn.DataParallel(encoder)
        classifier = nn.DataParallel(classifier)

        encoder.eval()
        classifier.eval()
        predict_labels = []
        true_labels = []

        print(f"{epoch} Start testing...")
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
        plt.savefig(f"result/report_{epoch}.png")

        # confusion matrixをseabornで表示して画像保存
        cm = confusion_matrix(y_true=true_labels, y_pred=predict_labels, normalize="true")
        cm_df = pd.DataFrame(cm, index=dataset.classes, columns=dataset.classes) * 100.0

        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_df, annot=True, fmt=".1f", cmap="BuGn", square=True, cbar=False)
        plt.savefig(f"result/confusion_matrix_{epoch}.png")


if __name__ == "__main__":
    main()
