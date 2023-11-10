import os

import torch
import torch.nn as nn
from common import Dataset, ImageTransform, param, remove_glob
from module import EfficientNetClassifier, EfficientNetEncoder
from tqdm import tqdm


def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = Dataset(
        root=param.tgt_path,
        transform=ImageTransform(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=param.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True
    )

    # model
    encoder = EfficientNetEncoder().to(device)
    encoder.load_state_dict(torch.load(param.pretrain_encoder_weight))
    encoder = nn.DataParallel(encoder)
    classifier = EfficientNetClassifier(len(dataset.classes)).to(device)
    classifier.load_state_dict(torch.load(param.pretrain_classifier_weight))
    classifier = nn.DataParallel(classifier)

    # learning settings
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(
        [
            {"params": encoder.parameters()},
            {"params": classifier.parameters()},
        ],
        lr=param.lr,
    )
    scaler = torch.cuda.amp.GradScaler(2**12)

    encoder.train()
    classifier.train()
    print("Start finetuning...")
    for epoch in range(param.finetune_num_epochs):
        epoch_loss = 0.0
        min_loss = 100.0
        for images, labels in tqdm(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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
        print(f"Epoch: {epoch+1}/{param.finetune_num_epochs} | Loss: {epoch_loss:.4f}")

        # save model
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            remove_glob(f"{param.finetune_encoder_weight}_*")
            remove_glob(f"{param.finetune_classifier_weight}_*")
            torch.save(encoder.module.state_dict(), f"{param.finetune_encoder_weight}_{epoch}")
            torch.save(classifier.module.state_dict(), f"{param.finetune_classifier_weight}_{epoch}")

    print("Finished finetuning!")
