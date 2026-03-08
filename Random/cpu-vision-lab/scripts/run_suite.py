#!/usr/bin/env python3
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_cmd(cmd, cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def latest_run_dir(results_root: Path, dataset: str, model: str) -> Path:
    pattern = f"{dataset}_{model}_*"
    candidates = sorted(results_root.glob(pattern))
    if not candidates:
        raise RuntimeError(f"No run directory found for {dataset}/{model}")
    return candidates[-1]


def plot_suite(summary_df: pd.DataFrame, suite_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary_df, x="dataset", y="final_test_acc", hue="model", ax=ax1)
    ax1.set_title("Final test accuracy by dataset/model")
    ax1.set_ylim(0.0, 1.0)
    fig1.tight_layout()
    fig1.savefig(suite_dir / "suite_accuracy.png", dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary_df, x="dataset", y="avg_epoch_seconds", hue="model", ax=ax2)
    ax2.set_title("Average epoch time (seconds)")
    fig2.tight_layout()
    fig2.savefig(suite_dir / "suite_speed.png", dpi=160)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_root = (root / args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    combos = [
        ("mnist", "mlp"),
        ("mnist", "cnn"),
        ("cifar10", "mlp"),
        ("cifar10", "cnn"),
    ]

    summary_rows = []
    for dataset, model in combos:
        cmd = [
            "python",
            "scripts/train_and_plot.py",
            "--dataset", dataset,
            "--model", model,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--data-dir", args.data_dir,
            "--results-root", args.results_root,
            "--num-workers", str(args.num_workers),
        ]
        run_cmd(cmd, cwd=root)
        run_dir = latest_run_dir(results_root, dataset, model)
        metrics = pd.read_csv(run_dir / "metrics.csv")
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "final_test_acc": float(metrics["test_acc"].iloc[-1]),
                "best_test_acc": float(metrics["test_acc"].max()),
                "avg_epoch_seconds": float(metrics["epoch_seconds"].mean()),
                "run_dir": str(run_dir),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    suite_dir = results_root / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(suite_dir / "suite_summary.csv", index=False)
    plot_suite(summary_df, suite_dir)

    print(f"Suite complete. Summary: {suite_dir / 'suite_summary.csv'}")
    print(f"Plots: {suite_dir / 'suite_accuracy.png'}, {suite_dir / 'suite_speed.png'}")


if __name__ == "__main__":
    main()
