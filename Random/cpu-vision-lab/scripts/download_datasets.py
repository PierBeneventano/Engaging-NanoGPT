#!/usr/bin/env python3
import argparse
from pathlib import Path

from torchvision import datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MNIST...")
    datasets.MNIST(root=str(data_dir), train=True, download=True)
    datasets.MNIST(root=str(data_dir), train=False, download=True)

    print("Downloading CIFAR10...")
    datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    datasets.CIFAR10(root=str(data_dir), train=False, download=True)

    print(f"Datasets are ready in: {data_dir.resolve()}")


if __name__ == "__main__":
    main()
