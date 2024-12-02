import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import AnkerDataset, ImageTransform, param
from module import EfficientNetClassifier, EfficientNetEncoder, L2Loss


def dist_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = AnkerDataset(
        anker_root=param.tgt_path,
        sample_root=param.src_path,
        transform=ImageTransform(),
        sample_type="positive",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=param.batch_size,
        shuffle=True,
        num_workers=param.num_workers,
        pin_memory=True,
    )

    # model
    encoder = EfficientNetEncoder().to(device)
    encoder.load_state_dict(torch.load(param.pretrain_encoder_weight))
    encoder = nn.DataParallel(encoder)
    classifier = EfficientNetClassifier(len(dataset.classes)).to(device)
    classifier.load_state_dict(torch.load(param.pretrain_classifier_weight))
    classifier = nn.DataParallel(classifier)

    # learning settings
    classify_criterion = nn.CrossEntropyLoss()
    dist_criterion = L2Loss()
    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters()},
            {"params": classifier.parameters()},
        ],
        lr=param.lr,
    )
    scaler = torch.GradScaler(device=device.type, init_scale=2**16)

    encoder.train()
    classifier.train()
    print("Start dist finetuning...")
    for epoch in range(param.finetune_num_epochs):
        epoch_classify_loss = 0.0
        epoch_dist_loss = 0.0
        for (tgt_images, tgt_labels), (src_images, src_labels) in tqdm(dataloader):
            tgt_images, tgt_labels = tgt_images.to(device, non_blocking=True), tgt_labels.to(device, non_blocking=True)
            src_images, src_labels = src_images.to(device, non_blocking=True), src_labels.to(device, non_blocking=True)

            # forward
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                src_feature = encoder(src_images)
                tgt_feature = encoder(tgt_images)

                # src_outputs = classifier(src_feature)
                tgt_outputs = classifier(tgt_feature)

                # classify_loss = classify_criterion(src_outputs, src_labels)
                classify_loss = classify_criterion(tgt_outputs, tgt_labels)
                dist_loss = dist_criterion(tgt_feature, src_feature)
                loss = classify_loss + dist_loss

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_classify_loss += classify_loss.item()
            epoch_dist_loss += dist_loss.item()

        epoch_classify_loss /= len(dataloader)
        epoch_dist_loss /= len(dataloader)
        print(f"Epoch: {epoch + 1}/{param.finetune_num_epochs} | Classify Loss: {epoch_classify_loss:.4f} | Dist Loss: {epoch_dist_loss:.4f}")

        # save model
        if (epoch + 1) in param.test_epochs:
            torch.save(encoder.module.state_dict(), f"{param.dist_tune_encoder_weight}_epoch_{epoch+1}")
            torch.save(classifier.module.state_dict(), f"{param.dist_tune_classifier_weight}_epoch_{epoch+1}")

    print("Finished dist finetuning!")
