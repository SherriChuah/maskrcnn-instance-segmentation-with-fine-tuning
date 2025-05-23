import torch
import json
import os

from torch.utils.data import DataLoader

from dataset.foodseg_json_dataset import FoodSegJSONDataset
from models.mask_rcnn import get_model
from utils.transforms import get_transform
from engine.train import train_one_epoch
from utils.visualise import show_image_with_masks


def collate_fn(batch):
    return tuple(zip(*batch))


def load_categories(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    return {cat['id']: cat['title'] for cat in meta['classes']}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_data_path = "data/foodseg103/train"
    test_data_path = "data/foodseg103/test"
    meta_path = "data/foodseg103/meta.json"

    categories = load_categories(meta_path)
    print(categories)

    # Dataset and Dataloader
    dataset = FoodSegJSONDataset(train_data_path, transforms=get_transform(train=True))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    test_dataset = FoodSegJSONDataset(test_data_path, transforms=get_transform(train=False))
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Model
    num_classes = max(categories.keys()) + 1 # background + category count
    print(f"num classes: {num_classes}")
    model = get_model(num_classes).to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training Loop
    for epoch in range(3):
        train_one_epoch(model, optimiser, data_loader, device, epoch)

        os.makedirs("outputs/models", exist_ok=True)
        torch.save(model.state_dict(), f"outputs/models/model_epoch_{epoch}.pth")

    
    # Visualize predictions on test set
    model.eval()
    with torch.no_grad():
        img, _ = test_dataset[0]
        pred = model([img.to(device)])[0]
    
    show_image_with_masks(img, pred, categories)


if __name__ == "__main__":
    main()
