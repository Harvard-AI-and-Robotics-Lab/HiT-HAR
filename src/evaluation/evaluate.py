"""
Test evaluation for HiT-HAR models.

Loads a trained model checkpoint and evaluates on test split.
Reports: per-class F1, macro F1, confusion matrices, generalization gap.
Logs all results to WandB if --wandb-run is specified.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.models.hit_har import build_model
from src.data.paired_dataset import PairedDataset, collate_fn, ACTION_MAP, SCENARIO_MAP

import pandas as pd


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', load_config(args.config))

    # Build model
    model = build_model(config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {args.checkpoint}")
    print(f"  Best score: {ckpt.get('best_score', 'N/A')}")
    print(f"  Best epoch: {ckpt.get('best_epoch', 'N/A')}")

    # Build test dataset — UIDs from test CSV (not scenario_labels.csv split column)
    test_df = pd.read_csv(args.test_labels)
    test_uids = test_df['video_uid'].unique().tolist()
    print(f"Test videos: {len(test_uids)}")

    # Use training normalization stats to prevent data leakage
    norm_stats = ckpt.get('norm_stats', None)
    if norm_stats is not None:
        print(f"  Using norm_stats from checkpoint (no leakage)")
    elif args.train_labels:
        # Fallback: compute norm_stats from train set (same as training would)
        print(f"  No norm_stats in checkpoint — computing from train set: {args.train_labels}")
        train_df = pd.read_csv(args.train_labels)
        train_uids = train_df['video_uid'].unique().tolist()
        train_ds_tmp = PairedDataset(
            train_uids, args.processed_dir, args.train_labels,
            args.scenario_labels, config, training=False,
        )
        norm_stats = {
            'mean': train_ds_tmp.global_mean,
            'std': train_ds_tmp.global_std,
        }
        del train_ds_tmp
        print(f"  Computed train norm_stats (shape: {norm_stats['mean'].shape})")
    else:
        print(f"  WARNING: No norm_stats and no --train-labels — test computes own stats (potential leakage)")

    test_ds = PairedDataset(
        test_uids, args.processed_dir, args.test_labels,
        args.scenario_labels, config, training=False,
        norm_stats=norm_stats,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    # Evaluate
    all_s_preds, all_s_labels = [], []
    all_a_preds, all_a_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(device)
            model_out = model(inputs)

            s_preds = model_out['scenario_logits'].argmax(dim=1).cpu()
            a_preds = model_out['action_logits'].argmax(dim=-1).cpu()

            all_s_preds.extend(s_preds.numpy())
            all_s_labels.extend(batch['scenario_label'].numpy())

            a_flat = batch['action_labels'].view(-1)
            a_pred_flat = a_preds.view(-1)
            valid = a_flat != -1
            all_a_preds.extend(a_pred_flat[valid].numpy())
            all_a_labels.extend(a_flat[valid].numpy())

    # Scenario metrics
    idx_to_scenario = {v: k for k, v in SCENARIO_MAP.items()}
    scenario_ids = list(range(len(SCENARIO_MAP)))
    scenario_names = [idx_to_scenario.get(i, f'cls{i}') for i in scenario_ids]
    s_f1 = f1_score(all_s_labels, all_s_preds, average='macro',
                    labels=scenario_ids, zero_division=0)

    print(f"\n{'='*60}")
    print(f"SCENARIO CLASSIFICATION (Test)")
    print(f"{'='*60}")
    print(f"Macro F1: {s_f1:.4f}")
    print(classification_report(all_s_labels, all_s_preds, labels=scenario_ids,
                                target_names=scenario_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_s_labels, all_s_preds, labels=scenario_ids))

    # Action metrics — use num_actions from config to support 4-class mode
    num_actions = config.get('num_actions', len(ACTION_MAP))
    idx_to_action = {v: k for k, v in ACTION_MAP.items()}
    action_ids = list(range(num_actions))
    action_names = [idx_to_action.get(i, f'cls{i}') for i in action_ids]
    a_f1 = f1_score(all_a_labels, all_a_preds, average='macro',
                    labels=action_ids, zero_division=0)

    print(f"\n{'='*60}")
    print(f"ACTION CLASSIFICATION (Test)")
    print(f"{'='*60}")
    print(f"Macro F1: {a_f1:.4f}")
    print(classification_report(all_a_labels, all_a_preds, labels=action_ids,
                                target_names=action_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_a_labels, all_a_preds, labels=action_ids))

    # Per-class F1 for actions
    per_class_a_f1 = f1_score(all_a_labels, all_a_preds, average=None,
                               labels=action_ids, zero_division=0)
    # Per-class F1 for scenarios
    per_class_s_f1 = f1_score(all_s_labels, all_s_preds, average=None,
                               labels=scenario_ids, zero_division=0)

    # Generalization gap — report action and scenario SEPARATELY
    val_score = ckpt.get('best_score', None)
    val_action_f1 = ckpt.get('val_action_f1', val_score)  # best_score is typically action_f1
    val_scenario_f1 = ckpt.get('val_scenario_f1', None)

    print(f"\n{'='*60}")
    print(f"GENERALIZATION GAP")
    print(f"{'='*60}")

    if val_action_f1 is not None:
        action_gap = val_action_f1 - a_f1
        action_gap_pct = action_gap / val_action_f1 * 100 if val_action_f1 > 0 else 0
        print(f"Action F1:   Val {val_action_f1:.4f} → Test {a_f1:.4f}  (gap: {action_gap:+.4f}, {action_gap_pct:+.1f}%)")

    if val_scenario_f1 is not None:
        scenario_gap = val_scenario_f1 - s_f1
        scenario_gap_pct = scenario_gap / val_scenario_f1 * 100 if val_scenario_f1 > 0 else 0
        print(f"Scenario F1: Val {val_scenario_f1:.4f} → Test {s_f1:.4f}  (gap: {scenario_gap:+.4f}, {scenario_gap_pct:+.1f}%)")
    else:
        print(f"Scenario F1: Val N/A → Test {s_f1:.4f}")

    if val_action_f1 is None:
        print(f"(No val metrics in checkpoint)")

    # ---- WandB logging ----
    if args.wandb_run:
        try:
            import wandb
            wandb.init(
                project='har-imu-training',
                name=f"test-{args.wandb_run}",
                group=args.wandb_group or 'test_eval',
                tags=['test', 'evaluation'],
                config={
                    'checkpoint': args.checkpoint,
                    'test_labels': args.test_labels,
                    'best_epoch': ckpt.get('best_epoch', None),
                    'val_best_score': val_score,
                },
            )
            # Log summary metrics — action and scenario SEPARATELY
            log_dict = {
                'test/action_f1_macro': a_f1,
                'test/scenario_f1_macro': s_f1,
            }
            if val_action_f1 is not None:
                log_dict['test/val_action_f1'] = val_action_f1
                log_dict['test/action_gap'] = val_action_f1 - a_f1
            if val_scenario_f1 is not None:
                log_dict['test/val_scenario_f1'] = val_scenario_f1
                log_dict['test/scenario_gap'] = val_scenario_f1 - s_f1
            wandb.log(log_dict)
            # Log per-class action F1
            for i, name in enumerate(action_names):
                if i < len(per_class_a_f1):
                    wandb.log({f'test/action_f1_{name.replace(" ", "_")}': per_class_a_f1[i]})
            # Log per-class scenario F1
            for i, name in enumerate(scenario_names):
                if i < len(per_class_s_f1):
                    wandb.log({f'test/scenario_f1_{name.replace(" ", "_")}': per_class_s_f1[i]})
            # Log confusion matrix as table
            cm_action = confusion_matrix(all_a_labels, all_a_preds, labels=action_ids)
            wandb.log({
                'test/action_confusion_matrix': wandb.plot.confusion_matrix(
                    y_true=all_a_labels, preds=all_a_preds,
                    class_names=action_names,
                ),
            })
            wandb.finish()
            print(f"\nWandB: test results logged to run 'test-{args.wandb_run}'")
        except ImportError:
            print("\nWandB not available — skipping logging")
        except Exception as e:
            print(f"\nWandB logging failed: {e}")

    return s_f1, a_f1


def main():
    parser = argparse.ArgumentParser(description='HiT-HAR Test Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--processed-dir', type=str, default='data/processed_ego4d')
    parser.add_argument('--test-labels', type=str, default='data/processed/test.csv',
                        help='Path to test split action labels CSV')
    parser.add_argument('--train-labels', type=str, default=None,
                        help='Path to train CSV — used to compute norm_stats if not in checkpoint')
    parser.add_argument('--scenario-labels', type=str, default='data/labels/scenario_labels.csv')
    parser.add_argument('--wandb-run', type=str, default=None,
                        help='WandB run name for logging test results (e.g., "hit-har-5class-beta03")')
    parser.add_argument('--wandb-group', type=str, default=None,
                        help='WandB group name (e.g., "v3_rerun")')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
