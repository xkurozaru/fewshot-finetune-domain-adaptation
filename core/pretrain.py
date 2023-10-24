import os

import torch
import torch.nn as nn
from tqdm import tqdm

from common import Dataset, ImageTransform, param
from module import EfficientNetClassifier, EfficientNetEncoder


def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = Dataset(
        root=param.src_path,
        transform=ImageTransform(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=param.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True
    )

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
    for epoch in range(param.pretrain_num_epochs):
        epoch_loss = 0.0
        for images, labels, _ in tqdm(dataloader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

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
    print("Finished pretraining!")

    # save weights
    torch.save(encoder.module.state_dict(), param.pretrain_encoder_weight)
    torch.save(classifier.module.state_dict(), param.pretrain_classifier_weight)
