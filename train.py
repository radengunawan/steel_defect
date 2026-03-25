"""
Training Script — Steel Surface Defect Classifier
---------------------------------------------------
Dataset : NEU Steel Surface Defect Dataset
          https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
          Expected folder layout:
            data/
              train/
                Crazing/       *.jpg
                Inclusion/     *.jpg
                Patches/       *.jpg
                Pitted_Surface/*.jpg
                Rolled-in_Scale/*.jpg
                Scratches/     *.jpg
              val/   (same structure)

Run
---
  python train.py --data_dir data --epochs 30 --batch_size 32 --output checkpoints/
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.classifier import SteelDefectClassifier, DEFECT_CLASSES


# ── Augmentation ───────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


# ── Mixup ──────────────────────────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training Loop ──────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, device, use_mixup=True, epoch=0
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

        if batch_idx % 10 == 0:
            print(
                f"  Epoch {epoch:3d} | batch {batch_idx:4d}/{len(loader)} "
                f"| loss {loss.item():.4f} | acc {correct/total:.3f}"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0] * len(DEFECT_CLASSES)
    class_total   = [0] * len(DEFECT_CLASSES)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)

        for c in range(len(DEFECT_CLASSES)):
            mask = labels == c
            class_correct[c] += preds[mask].eq(labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()

    per_class = {
        DEFECT_CLASSES[i]: round(class_correct[i] / max(class_total[i], 1), 3)
        for i in range(len(DEFECT_CLASSES))
    }
    return total_loss / total, correct / total, per_class


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=get_transforms("val"))

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"Classes: {train_ds.classes}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model     = SteelDefectClassifier(num_classes=len(DEFECT_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze full backbone after warm-up
        if epoch == args.warmup_epochs + 1:
            print(">> Unfreezing all backbone weights")
            model.unfreeze_all()
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.1

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=(epoch > 3), epoch=epoch
        )
        val_loss, val_acc, per_class = evaluate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:3d}/{args.epochs} — "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s"
        )
        print("  Per-class val acc:", per_class)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": DEFECT_CLASSES,
            }, ckpt_path)
            print(f"  >> Saved best model (val_acc={val_acc:.4f}) → {ckpt_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train steel defect classifier")
    parser.add_argument("--data_dir",      default="data",        help="Root of train/val folders")
    parser.add_argument("--output",        default="checkpoints", help="Where to save .pt files")
    parser.add_argument("--epochs",        type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--workers",       type=int, default=4)
    main(parser.parse_args())
