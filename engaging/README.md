# MIT Engaging Quick Setup (Project-Specific)

This is a practical setup layer for this repository on MIT Engaging.

## 1) First-time account activation

1. Sign in once at [Engaging OnDemand](https://engaging-ood.mit.edu) with Kerberos + Duo.
2. After that, SSH access to Engaging is enabled.

## 2) Local machine setup (your laptop/workstation)

### 2.1 Add SSH host alias

Copy the template and set your Kerberos username:

```bash
cp engaging/templates/ssh_config.engaging ~/.ssh/config.engaging
chmod 600 ~/.ssh/config.engaging
```

Then edit `~/.ssh/config` and add:

```text
Include ~/.ssh/config.engaging
```

Test:

```bash
ssh engaging
```

### 2.2 Sync this repo to Engaging

From your local clone:

```bash
ENGAGING_USER=<your_kerberos_username> bash engaging/scripts/sync_to_engaging.sh
```

Default remote target is `/home/<username>/Engaging-NanoGPT`.

## 3) On Engaging: environment bootstrap

SSH in and run:

```bash
cd /home/$USER/Engaging-NanoGPT
bash engaging/scripts/bootstrap_on_engaging.sh
```

This wraps the repo's `setup_env.sh` and uses:

- `CONDA_SH=$HOME/miniforge3/etc/profile.d/conda.sh` (default)
- `ENV_PATH=$HOME/conda_envs/nanogpt_env` (default)

Override either variable before running if needed.

## 4) Storage guidance

Typical locations:

- `/home/$USER` for code and small personal files
- group `pool` for shared lab data (if allocated)
- `scratch` for active, temporary high-throughput job data

For large training runs, keep checkpoints and temp artifacts in scratch/pool instead of home.

## 5) Module usage

Quick checks:

```bash
module avail
module load cuda/12.4.0 cudnn/9.8.0.87-cuda12
module list
```

## 6) Interactive allocations

General GPU example:

```bash
bash engaging/scripts/start_interactive.sh
```

Poggio partition example (1x A100):

```bash
PARTITION=pi_tpoggio GPU_TYPE=a100 GPUS=1 CPUS=16 MEM=64G TIME=04:00:00 \
  bash engaging/scripts/start_interactive.sh
```

## 7) Batch jobs for this repo

### Baseline NanoGPT GPT-2

General GPU partitions:

```bash
PARTITION=mit_normal_gpu GPU_TYPE=h100 GPUS_PER_NODE=1 GRAD_ACC_STEPS=40 \
  bash engaging/scripts/submit_nanogpt_gpt2.sh
```

Poggio partition, 4x A100:

```bash
PARTITION=pi_tpoggio GPU_TYPE=a100 GPUS_PER_NODE=4 GRAD_ACC_STEPS=10 CPUS_PER_TASK=64 \
  bash engaging/scripts/submit_nanogpt_gpt2.sh
```

### Modded speedrun path

General GPU partitions:

```bash
PARTITION=mit_normal_gpu GPU_TYPE=h100 NUM_GPUS=2 CPUS_PER_TASK=32 \
  bash engaging/scripts/submit_modded_speedrun.sh
```

Poggio partition, 8x A100:

```bash
PARTITION=pi_tpoggio GPU_TYPE=a100 NUM_GPUS=8 CPUS_PER_TASK=120 TIME=06:00:00 \
  bash engaging/scripts/submit_modded_speedrun.sh
```

## 8) Monitor/cancel jobs

```bash
squeue -u "$USER"
sacct -u "$USER" --format=JobID,JobName,Partition,State,Elapsed,ExitCode
scancel <jobid>
```

## 9) Partition/GPU visibility

```bash
sinfo -o "%P %G %N %a" | rg gpu
sinfo -p pi_tpoggio -N -o "%N %P %c %m %G %T %l"
```

## 10) Maintenance and support

- Login nodes typically restart weekly on Mondays around 7:00 AM.
- Monthly maintenance typically occurs on the 3rd Tuesday.
- Support: `orcd-help-engaging@mit.edu`

## 11) Official references

- [ORCD Engaging System](https://orcd.mit.edu/engaging)
- [ORCD Getting Started](https://orcd.mit.edu/getting-started)
- [ORCD SSH Login](https://orcd.mit.edu/engaging/ssh-login)
- [ORCD Filesystems](https://orcd.mit.edu/engaging/filesystems)
- [ORCD Transferring Files](https://orcd.mit.edu/engaging/transferring-files)
- [ORCD Modules](https://orcd.mit.edu/engaging/modules)
- [ORCD Requesting Resources](https://orcd.mit.edu/engaging/requesting-resources)
- [ORCD Running Jobs Overview](https://orcd.mit.edu/engaging/running-jobs)
- [Engaging OnDemand Portal](https://engaging-ood.mit.edu)
