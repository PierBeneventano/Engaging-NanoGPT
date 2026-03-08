#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, img_size: int, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat_size = 128 * (img_size // 4) * (img_size // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    epoch_seconds: float


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_dataset(name: str, data_dir: Path):
    if name == "mnist":
        train_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_set = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=train_t)
        test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=test_t)
        return train_set, test_set, 1, 28

    if name == "cifar10":
        train_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_t)
        test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_t)
        return train_set, test_set, 3, 32

    raise ValueError(f"Unsupported dataset: {name}")


def get_model(model_name: str, in_channels: int, img_size: int) -> nn.Module:
    if model_name == "mlp":
        return MLP(in_features=in_channels * img_size * img_size)
    if model_name == "cnn":
        return SmallCNN(in_channels=in_channels, img_size=img_size)
    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch_train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, leave=False, desc="train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def run_epoch_eval(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    return (
        running_loss / total,
        correct / total,
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )


def save_plots(df: pd.DataFrame, cm: np.ndarray, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axs[0].plot(df["epoch"], df["test_loss"], label="test_loss")
    axs[0].set_title("Loss curves")
    axs[0].set_xlabel("epoch")
    axs[0].legend()

    axs[1].plot(df["epoch"], df["train_acc"], label="train_acc")
    axs[1].plot(df["epoch"], df["test_acc"], label="test_acc")
    axs[1].set_title("Accuracy curves")
    axs[1].set_xlabel("epoch")
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2)
    ax2.set_title("Confusion matrix (final epoch)")
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("true")
    fig2.tight_layout()
    fig2.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--model", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data_dir = Path(args.data_dir)
    results_root = Path(args.results_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / f"{args.dataset}_{args.model}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set, test_set, in_channels, img_size = get_dataset(args.dataset, data_dir)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = get_model(args.model, in_channels=in_channels, img_size=img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    rows = []
    preds = None
    targets = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch_train(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc, preds, targets = run_epoch_eval(model, test_loader, criterion, device)
        dt = time.time() - t0

        row = EpochMetrics(
            epoch=epoch,
            train_loss=tr_loss,
            train_acc=tr_acc,
            test_loss=te_loss,
            test_acc=te_acc,
            epoch_seconds=dt,
        )
        rows.append(row)
        print(
            f"[{args.dataset}/{args.model}] "
            f"epoch={epoch} train_acc={tr_acc:.4f} test_acc={te_acc:.4f} "
            f"train_loss={tr_loss:.4f} test_loss={te_loss:.4f} time={dt:.1f}s"
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(out_dir / "metrics.csv", index=False)

    cm = confusion_matrix(targets, preds)
    save_plots(df, cm, out_dir)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved run outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
