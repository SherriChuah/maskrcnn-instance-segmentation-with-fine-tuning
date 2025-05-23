from tqdm import tqdm


def train_one_epoch(model, optimiser, data_loader, device, epoch):
    model.train()

    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimiser.zero_grad()
        losses.backward()
        optimiser.step()
    
    print(f"Loss: {losses.item():.4f}")