"""
ECE 662 Team 9: Checkpoint 3 retraining for the INT4 collapse diagnosis (Amanda).

Retrains MobileNetV2 with the team's exact fine-tuning recipe (ImageNet-pretrained, AdamW with
features lr 1e-4 / classifier lr 1e-3, weight decay 1e-4, cosine schedule, 10 epochs, batch 128,
45k/5k train/validation split with seed 42, best checkpoint chosen on validation), changing ONE thing
per variant so the W4 collapse can be traced to its cause:

  res64_seed1         Same recipe at 64x64, different training seed (is the collapse reproducible?)
  res32               32x32 input, torchvision strides (resolution change only)
  res32_cifar_strides 32x32 input, stride 1 at features.0.0 and features.2.conv.1.0 (the two layers where
                      the hub model differs), i.e. the hub model's architecture with ImageNet initialization

All variants keep ImageNet normalization, so input statistics only change through resolution.

Run from the repo folder, one variant at a time:
    python retrain_variants.py --variant res32_cifar_strides
    python retrain_variants.py --variant res32
    python retrain_variants.py --variant res64_seed1

Outputs (checkpoints/):
    <variant>_best.pth              best-validation weights (used by the notebook)
    <variant>_epoch<k>.pth          weights after each epoch (training-length sensitivity)
    <variant>_log.json              settings + per-epoch train/validation accuracy
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VARIANTS = {
    "res64_seed1": {"size": 64, "cifar_strides": False, "seed": 1},
    "res32": {"size": 32, "cifar_strides": False, "seed": 42},
    "res32_cifar_strides": {"size": 32, "cifar_strides": True, "seed": 42},
}


def build_model(cifar_strides, pretrained=True):
    """MobileNetV2 with a 10-class head; optionally the hub model's two stride-1 changes."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    if cifar_strides:
        model.features[0][0].stride = (1, 1)
        model.features[2].conv[1][0].stride = (1, 1)
    return model


def make_transforms(size):
    resize = [transforms.Resize((size, size))] if size != 32 else []
    train_transform = transforms.Compose(resize + [
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose(resize + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="debug only: limit batches per epoch")
    parser.add_argument("--no-pretrained", action="store_true", help="debug only: skip ImageNet weights")
    args = parser.parse_args()

    settings = VARIANTS[args.variant]
    torch.manual_seed(settings["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Variant {args.variant} | {settings} | device {device}", flush=True)

    train_transform, eval_transform = make_transforms(settings["size"])
    train_data = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    val_data = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=eval_transform)

    # Same 45k / 5k split as the team (seed 42), regardless of the training seed
    permutation = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices, val_indices = permutation[:45000], permutation[45000:]
    pin = device.type == "cuda"
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=128, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=256, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)

    model = build_model(settings["cifar_strides"], pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    log = {"variant": args.variant, **settings, "epochs": args.epochs, "history": []}
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        correct = total = 0
        for step, (images, labels) in enumerate(train_loader):
            if args.max_batches and step >= args.max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        train_acc = 100.0 * correct / max(total, 1)
        val_acc = evaluate(model, val_loader, device)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, os.path.join(args.out_dir, f"{args.variant}_epoch{epoch}.pth"))
        if val_acc > best_val:
            best_val = val_acc
            torch.save(state, os.path.join(args.out_dir, f"{args.variant}_best.pth"))
        log["history"].append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc,
                               "seconds": time.time() - start})
        log["best_val_acc"] = best_val
        log["best_epoch"] = max(log["history"], key=lambda h: h["val_acc"])["epoch"]
        with open(os.path.join(args.out_dir, f"{args.variant}_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"Epoch {epoch}/{args.epochs} | train {train_acc:.2f}% | val {val_acc:.2f}% | "
              f"{time.time() - start:.0f}s{' | best' if val_acc == best_val else ''}", flush=True)

    print(f"Done. Best validation accuracy {best_val:.2f}% (epoch {log['best_epoch']}).")


if __name__ == "__main__":
    main()
