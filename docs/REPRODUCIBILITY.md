# Reproducibility

This document records the expected workflow for reproducing the main training and evaluation runs.

## Environment

Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or build the Docker image:

```bash
docker build -t hit-har .
```

## Preflight

Run:

```bash
bash scripts/preflight.sh
python -m pytest
```

Dataset-dependent tests are skipped if the full dataset has not been downloaded.

## Training

The recommended five-class configuration is:

```bash
python train.py \
  --config configs/beta_sweep_5class_b03.yaml \
  --processed-dir data/processed_ego4d \
  --train-labels data/processed_clean/train.csv \
  --val-labels data/processed_clean/val.csv \
  --scenario-labels data/labels/scenario_labels.csv \
  --run-name hit-har-5class \
  --no-wandb
```

To enable Weights & Biases logging, remove `--no-wandb` and set:

```bash
export WANDB_PROJECT=hit-har
export WANDB_ENTITY=<your-wandb-entity>
```

## Evaluation

```bash
python src/evaluation/evaluate.py \
  --checkpoint checkpoints/hit-har-5class-beta03-headline.pth \
  --config configs/beta_sweep_5class_b03.yaml \
  --processed-dir data/processed_ego4d \
  --test-labels data/processed_clean/test.csv \
  --train-labels data/processed_clean/train.csv \
  --scenario-labels data/labels/scenario_labels.csv
```

If the checkpoint does not contain normalization statistics, the evaluator recomputes them from the training split passed through `--train-labels`.

## Expected Data Split

Use `data/processed_clean/{train,val,test}.csv` for paper-aligned experiments. These files are produced from the validated label pipeline and filtered to reduce noisy stationary labels.
