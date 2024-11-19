import torch
import torch.nn as nn
from tqdm import tqdm

from common import Dataset, DivideDataset, ImageTransform, param
from module import EfficientNetClassifier, EfficientNetEncoder, TripletLoss


def triplet_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    tgt_dataset = Dataset(
        root=param.tgt_path,
        transform=ImageTransform(),
    )
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=param.batch_size, shuffle=True, num_workers=param.num_workers, pin_memory=True)
    src_dataset = DivideDataset(
        root=param.src_path,
        transform=ImageTransform(),
        device=device,
    )

    # model
    encoder = EfficientNetEncoder().to(device)
    encoder.load_state_dict(torch.load(param.pretrain_encoder_weight))
    encoder = nn.DataParallel(encoder)
    classifier = EfficientNetClassifier(len(tgt_dataset.classes)).to(device)
    classifier.load_state_dict(torch.load(param.pretrain_classifier_weight))
    classifier = nn.DataParallel(classifier)

    # learning settings
    classify_criterion = nn.CrossEntropyLoss()
    dist_criterion = TripletLoss()
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
    print("Start triplet finetuning...")
    for epoch in range(param.finetune_num_epochs):
        epoch_classify_loss = 0.0
        epoch_dist_loss = 0.0
        for tgt_images, tgt_labels in tqdm(tgt_dataloader):
            tgt_images, tgt_labels = tgt_images.to(device, non_blocking=True), tgt_labels.to(device, non_blocking=True)

            p_images, p_labels = src_dataset.get_random_images_by_labels(tgt_labels.tolist())
            p_images, p_labels = p_images.to(device, non_blocking=True), p_labels.to(device, non_blocking=True)

            n_images, n_labels = src_dataset.get_random_notequal_images_by_labels(tgt_labels.tolist())
            n_images, n_labels = n_images.to(device, non_blocking=True), n_labels.to(device, non_blocking=True)

            # forward
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                p_feature = encoder(p_images)
                n_feature = encoder(n_images)
                tgt_feature = encoder(tgt_images)

                # p_outputs = classifier(p_feature)
                tgt_outputs = classifier(tgt_feature)

                # classify_loss = classify_criterion(p_outputs, p_labels)
                classify_loss = classify_criterion(tgt_outputs, tgt_labels)
                dist_loss = dist_criterion(tgt_feature, p_feature, n_feature)
                loss = classify_loss + dist_loss

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_classify_loss += classify_loss.item()
            epoch_dist_loss += dist_loss.item()

        epoch_classify_loss /= len(tgt_dataloader)
        epoch_dist_loss /= len(tgt_dataloader)
        print(f"Epoch: {epoch + 1}/{param.finetune_num_epochs} | Classify Loss: {epoch_classify_loss:.4f} | Dist Loss: {epoch_dist_loss:.4f}")

        # save model
        if (epoch + 1) in param.test_epochs:
            torch.save(encoder.module.state_dict(), f"{param.triplet_tune_encoder_weight}_epoch_{epoch+1}")
            torch.save(classifier.module.state_dict(), f"{param.triplet_tune_classifier_weight}_epoch_{epoch+1}")

    print("Finished triplet finetuning!")
