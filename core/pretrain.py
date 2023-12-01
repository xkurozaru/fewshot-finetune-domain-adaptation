import os

import torch
import torch.nn as nn
from tqdm import tqdm

from common import Dataset, ImageTransformV2, param, remove_glob
from module import EfficientNetClassifier, EfficientNetEncoder


def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    transform = ImageTransformV2()
    dataset = Dataset(
        root=param.src_path,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.pretrain_batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    # model
    encoder = EfficientNetEncoder().to(device)
    encoder = nn.DataParallel(encoder)
    classifier = EfficientNetClassifier(len(dataset.classes)).to(device)
    classifier = nn.DataParallel(classifier)

    # learning settings
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters()},
            {"params": classifier.parameters()},
        ],
        lr=param.lr,
    )
    scaler = torch.cuda.amp.GradScaler(2**12)

    encoder.train()
    classifier.train()
    print("Start pretraining...")
    min_loss = 100.0
    for epoch in range(param.pretrain_num_epochs):
        epoch_loss = 0.0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images = transform.augment_transform(images)

            # forward
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                features = encoder(images)
                outputs = classifier(features)
                loss = criterion(outputs, labels)

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch: {epoch+1}/{param.pretrain_num_epochs} | Loss: {epoch_loss:.4f}")

        # save model
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            remove_glob(f"{param.pretrain_encoder_weight}_best_*")
            remove_glob(f"{param.pretrain_classifier_weight}_best_*")
            torch.save(encoder.module.state_dict(), f"{param.pretrain_encoder_weight}_best_{epoch+1}")
            torch.save(classifier.module.state_dict(), f"{param.pretrain_classifier_weight}_best_{epoch+1}")

        if (epoch + 1) % (param.pretrain_num_epochs // 10) == 0:
            torch.save(encoder.module.state_dict(), f"{param.pretrain_encoder_weight}_epoch_{epoch+1}")
            torch.save(classifier.module.state_dict(), f"{param.pretrain_classifier_weight}_epoch_{epoch+1}")

    print("Finished pretraining!")
