"""
Trains EmotionCNN on FER2013.

Example
-------
    python -m src.train --data data/fer2013.csv --epochs 40 --batch-size 64

Saves the best checkpoint (by validation accuracy) to models/best_model.pt
and a loss/accuracy curve plot to assets/training_curves.png.
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")  # headless: no display server required
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import FER2013Dataset
from src.model import EmotionCNN
from src.utils import set_seed, get_device, ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train the FER2013 emotion classifier")
    p.add_argument("--data", type=str, default="data/fer2013.csv", help="Path to fer2013.csv")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7, help="Early-stopping patience (epochs)")
    p.add_argument("--output-dir", type=str, default="models")
    p.add_argument("--assets-dir", type=str, default="assets")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    torch.set_grad_enabled(True)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    ensure_dir(args.output_dir)
    ensure_dir(args.assets_dir)

    print("Loading datasets...")
    train_ds = FER2013Dataset(args.data, usage="Training", augment=True)
    val_ds = FER2013Dataset(args.data, usage="PublicTest", augment=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "val_acc": val_acc, "epoch": epoch},
                f"{args.output_dir}/best_model.pt",
            )
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    plot_curves(history, f"{args.assets_dir}/training_curves.png")
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {args.output_dir}/best_model.pt")


def plot_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


if __name__ == "__main__":
    main()
