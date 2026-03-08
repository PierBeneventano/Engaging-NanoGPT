# CPU Vision Lab (MNIST + CIFAR10)

Reproducible CPU-only setup for small-network experiments on MNIST and CIFAR10.

## What this includes

- CPU-only PyTorch install
- One-shot dataset download script
- Small model training script with metrics and plots
- Suite runner for multiple experiments + summary plots

## Quick start on EC2

```bash
cd /home/ec2-user/.openclaw/workspace/cpu-vision-lab
bash setup.sh
source .venv/bin/activate
python scripts/download_datasets.py --data-dir data
python scripts/run_suite.py --epochs 3 --batch-size 128 --results-root results
```

## Output

Per run, files are written under `results/<dataset>_<model>_<timestamp>/`:

- `metrics.csv` (epoch metrics)
- `config.json` (run config)
- `curves.png` (loss + accuracy curves)
- `confusion_matrix.png` (final test confusion matrix)

Suite-level:

- `results/suite_<timestamp>/suite_summary.csv`
- `results/suite_<timestamp>/suite_accuracy.png`
- `results/suite_<timestamp>/suite_speed.png`
