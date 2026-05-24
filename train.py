"""
HiT-HAR Training Entry Point — Standalone IMU Pipeline.

Usage:
  Baseline (no distillation):
    python train.py --config configs/baseline_5class.yaml --phase 0

  Gold fine-tuning:
    python train.py --config configs/baseline_5class.yaml --phase 2 \
      --checkpoint checkpoints/baseline/best_model.pth
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.models.hit_har import build_model
from src.training.losses_hit_har import HiTHARLoss
from src.training.trainer import HiTHARTrainer
from src.data.paired_dataset import build_dataloaders


def main():
    parser = argparse.ArgumentParser(description='HiT-HAR Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--phase', type=int, default=0, choices=[0, 2],
                        help='Training phase: 0=baseline, 2=gold fine-tune')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for Phase 2 fine-tuning')
    parser.add_argument('--processed-dir', type=str, default='data/processed_ego4d',
                        help='Path to processed IMU NPZ directory')
    parser.add_argument('--train-labels', type=str, default='data/processed/train.csv',
                        help='Path to training split action labels CSV')
    parser.add_argument('--val-labels', type=str, default='data/processed/val.csv',
                        help='Path to validation split action labels CSV')
    parser.add_argument('--scenario-labels', type=str, default='data/labels/scenario_labels.csv',
                        help='Path to scenario labels CSV')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for model checkpoints')
    parser.add_argument('--run-name', type=str, default=None,
                        help='WandB run name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume checkpoint (e.g. checkpoints/baseline/resume_checkpoint.pth)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')
    parser.add_argument('--wandb-group', type=str, default=None,
                        help='WandB group name for experiment organization')
    parser.add_argument('--wandb-tags', type=str, default=None,
                        help='Comma-separated WandB tags')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load config
    config = load_config(args.config)

    # Override epochs if specified
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    # Build unique output directory: {base}/{run_name}_{timestamp}_s{seed}
    # This prevents checkpoint overwriting between different runs
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.run_name:
        run_dir = f"{args.run_name}_{timestamp}_s{args.seed}"
    else:
        config_name = Path(args.config).stem
        run_dir = f"{config_name}_{timestamp}_s{args.seed}"

    args.output_dir = str(Path(args.output_dir) / run_dir)

    print(f"=== HiT-HAR Phase {args.phase} Training ===")
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")

    # Build data
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        config,
        processed_dir=args.processed_dir,
        train_labels_path=args.train_labels,
        val_labels_path=args.val_labels,
        scenario_labels_path=args.scenario_labels,
    )

    # Build model
    model = build_model(config)
    total_params = model.count_parameters()
    print(f"Model parameters: {total_params:,}")

    # Build loss
    loss_config = {
        'num_actions': config.get('num_actions', 5),
        'num_scenarios': config['hla']['num_classes'],
        'embedding_dim': config['lle']['embedding_dim'],
        'beta_task': config['training'].get('beta_task', 0.7),
        'action_weights': config['training'].get('action_weights', [1.0, 1.7, 2.3, 4.4, 8.8]),
        'scenario_weights': config['training'].get('scenario_weights', None),
        'lambda_lang': config.get('lambda_lang', 0.0),
        'lambda_coarse': config['training'].get('lambda_coarse', 0.0),
        'label_smoothing': config['training'].get('label_smoothing', 0.0),
        'lambda_orth': config['training'].get('lambda_orth', 0.0),
        'lambda_var': config['training'].get('lambda_var', 0.0),
        'dual_space': config.get('dual_space', False),
        'lang_subspace_dim': config.get('lang_subspace_dim', None),
        # Learnable beta modes
        'learnable_beta': config['training'].get('learnable_beta', False),
        'per_class_beta': config['training'].get('per_class_beta', False),
    }
    criterion = HiTHARLoss(loss_config)

    # Build trainer
    trainer = HiTHARTrainer(model, criterion, config, args)

    # Load checkpoint for Phase 2
    if args.phase == 2 and args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    best_score, best_epoch = trainer.train(train_loader, val_loader, train_ds, val_ds)
    print(f"\nTraining complete! Best score: {best_score:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()
