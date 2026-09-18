"""
Evaluates a trained checkpoint on the held-out FER2013 test split
(PrivateTest) and reports per-class precision/recall/F1 plus a confusion
matrix image.

Example
-------
    python -m src.evaluate --data data/fer2013.csv --checkpoint models/best_model.pt
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import FER2013Dataset
from src.model import EmotionCNN
from src.utils import EMOTIONS, get_device, ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained emotion classifier")
    p.add_argument("--data", type=str, default="data/fer2013.csv")
    p.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    p.add_argument("--usage", type=str, default="PrivateTest", choices=["PublicTest", "PrivateTest"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--assets-dir", type=str, default="assets")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    ensure_dir(args.assets_dir)

    test_ds = FER2013Dataset(args.data, usage=args.usage, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = EmotionCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(reported val_acc={checkpoint.get('val_acc', float('nan')):.4f})")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(f"\nEvaluated on {args.usage} split ({len(all_labels)} samples)\n")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, EMOTIONS, f"{args.assets_dir}/confusion_matrix.png")


def plot_confusion_matrix(cm, class_names, out_path):
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
