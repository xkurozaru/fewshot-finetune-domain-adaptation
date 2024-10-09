import torch
import torch.nn as nn
from tqdm import tqdm

from common import Dataset, DivideDataset, ImageTransform, param
from module import DANN


def dann_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    tgt_dataset = Dataset(
        root=param.tgt_path,
        transform=ImageTransform(),
    )
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=param.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    src_dataset = DivideDataset(
        root=param.src_path,
        transform=ImageTransform(),
        device=device,
    )

    # model
    model = DANN(len(tgt_dataset.classes)).to(device)
    model.encoder.load_state_dict(torch.load(param.pretrain_encoder_weight))
    model.classifier.load_state_dict(torch.load(param.pretrain_classifier_weight))
    model = nn.DataParallel(model)

    # learning settings
    classify_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr)
    scaler = torch.cuda.amp.GradScaler(2**12)

    model.train()
    print("Start DANN finetuning...")
    for epoch in range(param.finetune_num_epochs):
        epoch_classify_loss = 0.0
        epoch_domain_loss = 0.0
        for tgt_images, tgt_labels in tqdm(tgt_dataloader):
            tgt_images, tgt_labels = tgt_images.to(device, non_blocking=True), tgt_labels.to(device, non_blocking=True)

            src_images, src_labels = src_dataset.get_random_images_by_labels(tgt_labels.tolist())
            src_images, src_labels = src_images.to(device, non_blocking=True), src_labels.to(device, non_blocking=True)

            domains = torch.cat([torch.zeros(src_images.size(0), 1), torch.ones(tgt_images.size(0), 1)], dim=0).to(device)

            # forward
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, preds, tgt_domains_preds = model(tgt_images)
                _, _, src_domains_preds = model(src_images)
                classify_loss = classify_criterion(preds, tgt_labels)
                domains_preds = torch.cat([src_domains_preds, tgt_domains_preds], dim=0)

                domain_loss = domain_criterion(domains_preds, domains)
                loss = classify_loss + domain_loss

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_classify_loss += classify_loss.item()
            epoch_domain_loss += domain_loss.item()

        epoch_classify_loss /= len(tgt_dataloader)
        epoch_domain_loss /= len(tgt_dataloader)
        print(f"Epoch: {epoch + 1}/{param.finetune_num_epochs} | Classify Loss: {epoch_classify_loss:.4f} | Domain Loss: {epoch_domain_loss:.4f}")

        # save model
        if (epoch + 1) in param.test_epochs:
            torch.save(model.module.state_dict(), f"{param.dann_tune_model_weight}_epoch_{epoch+1}")

    print("Finished DANN finetuning!")
