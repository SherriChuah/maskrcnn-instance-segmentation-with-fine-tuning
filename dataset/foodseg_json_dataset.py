import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class FoodSegJSONDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "img")
        self.ann_dir = os.path.join(root_dir, "ann")
        self.transforms = transforms

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.jpg')
        ])

    def load_annotation(self, ann_path, image_size):
        with open(ann_path) as f:
            data = json.load(f)

        masks = []
        boxes = []
        labels = []

        for obj in data['objects']:
            label_id = obj['category_id']
            polygons = obj['segmentation']

            # Draw mask from polygons
            mask = Image.new("L", image_size, 0)
            for poly in polygons:
                ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)

            mask_np = np.array(mask, dtype=np.uint8)
            pos = np.where(mask_np)
            if pos[0].size == 0 or pos[1].size == 0:
                continue

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(mask_np)
            labels.append(label_id)

        if not masks:
            return None

        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "area": (masks.sum(dim=(1, 2))).float(),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.replace('.jpg', '.json'))

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        target = self.load_annotation(ann_path, (width, height))
        if target is None:
            return self.__getitem__((idx + 1) % len(self))  # skip bad mask

        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img, target = self.transforms(img, target)

        img = F.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.img_files)
