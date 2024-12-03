import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import ImageTransform, param
from common.dataset import DoubleDataset
from module import DANN


def dann():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = DoubleDataset(
        src_root=param.src_path,
        tgt_root=param.test_path,
        transform=ImageTransform(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=param.pretrain_batch_size,
        shuffle=True,
        num_workers=param.num_workers,
        pin_memory=True,
    )

    # model
    model = DANN(len(dataset.classes)).to(device)
    model = nn.DataParallel(model)

    # learning settings
    classify_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr)
    scaler = torch.GradScaler(device=device.type, init_scale=2**16)

    model.train()
    print("Start DANN training...")
    for epoch in range(param.pretrain_num_epochs):
        epoch_classify_loss = 0.0
        epoch_domain_loss = 0.0
        for (src_images, src_labels), (tgt_images, _) in tqdm(dataloader):
            src_images, src_labels = src_images.to(device, non_blocking=True), src_labels.to(device, non_blocking=True)
            tgt_images = tgt_images.to(device, non_blocking=True)
            domains = torch.cat([torch.zeros(src_images.size(0), 1), torch.ones(tgt_images.size(0), 1)], dim=0).to(device, non_blocking=True)

            # forward
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                preds, src_domains_preds = model(src_images)
                _, tgt_domains_preds = model(tgt_images)

                classify_loss = classify_criterion(preds, src_labels)
                domains_preds = torch.cat([src_domains_preds, tgt_domains_preds], dim=0)
                domain_loss = domain_criterion(domains_preds, domains)
                loss = classify_loss + domain_loss

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_classify_loss += classify_loss.item()
            epoch_domain_loss += domain_loss.item()

        epoch_classify_loss /= len(dataloader)
        epoch_domain_loss /= len(dataloader)
        print(f"Epoch: {epoch + 1}/{param.pretrain_num_epochs} | Classify Loss: {epoch_classify_loss:.4f} | Domain Loss: {epoch_domain_loss:.4f}")

        # save model
        if (epoch + 1) % 10 == 0:
            torch.save(model.module.state_dict(), f"{param.dann_model_weight}_epoch_{epoch+1}")

    print("Finished DANN training!")
