# HiT-HAR

### Paper: arXiv link pending | CVPR Workshop 2026

Official implementation of **"Beyond Motion Primitives: Behavioral Activity Recognition from Head-Mounted IMU"**.

HiT-HAR (Hierarchical Temporal Human Activity Recognition) studies behavior-level activity recognition from head-mounted inertial measurement unit (IMU) signals. The project focuses on fine-grained daily activity labels derived from egocentric narration events and human validation, using a five-class taxonomy:

- Object Transfer
- Task Operation
- Stationary
- Locomotion
- Search

The repository contains the model code, training pipeline, evaluation utilities, analysis scripts, and experiment configurations used for the paper.

---

## Overview

HiT-HAR trains hierarchical IMU models on timestamp-aligned narration labels. Each video sequence is represented by preprocessed 6-axis IMU windows, and each narration timestamp provides an action label over the five behavioral classes above.

The codebase includes:

- hierarchical action and scenario prediction models;
- language-guided and IMU-only model variants;
- data cleaning and split-generation utilities;
- evaluation scripts for action and scenario metrics;
- analysis scripts for label quality, taxonomy ambiguity, and IMU separability.

---

## Installation

Python 3.10 is recommended.

```bash
git clone https://github.com/Harvard-AI-and-Robotics-Lab/HiT-HAR.git
cd HiT-HAR
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

A minimal Docker workflow is also provided:

```bash
docker build -t hit-har .
docker run --rm -it -v "$PWD":/workspace -w /workspace hit-har bash
```

---

## Data

Full label files and dataset release artifacts are intended to be distributed through the Harvard AI and Robotics Lab Hugging Face organization:

https://huggingface.co/harvardairobotics/datasets

After downloading the dataset package, place the files under:

```text
data/
├── processed_clean/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── labels/
│   └── scenario_labels.csv
└── processed_ego4d/
    └── <video_uid>/seq.npz
```

The `processed_ego4d/` directory should contain one `seq.npz` file per video. Each file stores preprocessed IMU windows and timestamps. Raw Ego4D assets are not included in this repository.

See [docs/DATA.md](docs/DATA.md) for the expected file layout and label schema.

---

## Model Weights

Pretrained checkpoints are intended to be released as repository release assets or through the lab Hugging Face organization. The headline five-class checkpoint is:

```text
hit-har-5class-beta03-headline.pth
```

See [MODEL_ZOO.md](MODEL_ZOO.md) for the full checkpoint manifest and evaluation commands.

---

## Quick Start

Run the test suite:

```bash
python -m pytest
```

Tests that require the full dataset are skipped automatically when the dataset has not been downloaded.

Train a five-class model:

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

Evaluate a checkpoint:

```bash
python src/evaluation/evaluate.py \
  --checkpoint checkpoints/hit-har-5class-beta03-headline.pth \
  --config configs/beta_sweep_5class_b03.yaml \
  --processed-dir data/processed_ego4d \
  --test-labels data/processed_clean/test.csv \
  --train-labels data/processed_clean/train.csv \
  --scenario-labels data/labels/scenario_labels.csv
```

---

## Repository Structure

```text
HiT-HAR/
├── analysis/              # Label, signal, and figure analysis scripts
├── baselines/             # Baseline model implementations
├── configs/               # Experiment configuration files
├── data/                  # Dataset placeholder and layout notes
├── docs/                  # Data and reproducibility documentation
├── scripts/               # Data preparation and utility scripts
├── src/
│   ├── data/              # Dataset loading and label processing
│   ├── evaluation/        # Evaluation entry points
│   ├── models/            # HiT-HAR model definitions
│   └── training/          # Losses and training loop
├── tests/                 # Unit and integration tests
├── train.py               # Main training entry point
└── requirements.txt
```

---

## Reproducibility

The paper-aligned release uses `data/processed_clean/{train,val,test}.csv` as the main split. These splits are grouped by `video_uid` to avoid cross-split video leakage.

Recommended checks before running experiments:

```bash
bash scripts/preflight.sh
python -m pytest
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for additional training and evaluation notes.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{huang2026beyondmotion,
  title  = {Beyond Motion Primitives: Behavioral Activity Recognition from Head-Mounted IMU},
  author = {Huang, Chung-Ta and Das, Léopold and Zhou, Jeffrey and Siddique, Faizaan and Baek, Julia Seungjoo and Liu, Serena Yuchen and Rusli, Andrew and Zhou, Todd Y. and Yu, Freddy and Hansen, Sinclair and Hu, Ziling and Sharma, Arnav and Wang, Mengyu},
  year   = {2026}
}
```

---

## License

The code and dataset licenses will be added after lab approval.

---

## Acknowledgements

This project builds on the Ego4D dataset and the broader egocentric perception research ecosystem. We thank the Harvard AI and Robotics Lab for supporting the release.
