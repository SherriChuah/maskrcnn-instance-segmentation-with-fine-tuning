import matplotlib.pyplot as plt
import numpy as np
import torch
import random

def show_image_with_masks(img, pred, categories=None, score_thresh=0.5):
    img = img.permute(1,2,0).numpy()
    
    plt.figure(figsize=(10,10))
    plt.imshow(img)

    ax = plt.gca()

    masks = pred["masks"]
    boxes = pred["boxes"]
    labels = pred["labels"]
    scores = pred["scores"]

    for i in range(len(masks)):
        if scores[i] < score_thresh:
            continue

        mask = masks[i,0].mul(255).byte().cpu().numpy()
        color = np.random.rand(3,)
        
        ax.contour(mask, levels=[0.5], colors=[color])

        x1, y1, x2, y2 = boxes[i].detach().cpu().numpy()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, color=color, linewidth=2))
        label_id = labels[i].item()
        label_name = categories[label_id] if categories and label_id in categories else str(label_id)
        ax.text(x1, y1, f"{label_id}:{scores[i]:.2f}", color=color, fontsize=12,
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))
    
    plt.axis("off")
    plt.tight_layout()
    plt.show()


