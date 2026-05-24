"""Mantis-8M baseline for HiT-HAR: frozen feature extraction + SVM, and fine-tuning.

Runs two evaluation modes:
  1. Frozen Mantis-8M backbone -> feature extraction -> LinearSVC
  2. Mantis-8M fine-tuned end-to-end via MantisTrainer.fit()

Reports macro F1, per-class F1 for both 5-class and 4-class settings.

Usage:
    # 5-class frozen + fine-tuned (20 epochs)
    python baselines/mantis_baseline.py \
        --processed-dir data/processed_ego4d \
        --train-labels data/processed_clean/train.csv \
        --test-labels data/processed_clean/test.csv \
        --num-classes 5 --num-epochs 20

    # 4-class (excludes Search)
    python baselines/mantis_baseline.py \
        --processed-dir data/processed_ego4d \
        --train-labels data/processed_clean/train.csv \
        --test-labels data/processed_clean/test.csv \
        --num-classes 4 --num-epochs 20

    # Fine-tune only (skip frozen+SVM)
    python baselines/mantis_baseline.py \
        --processed-dir data/processed_ego4d \
        --train-labels data/processed_clean/train.csv \
        --test-labels data/processed_clean/test.csv \
        --num-classes 5 --finetune-only --num-epochs 20

Requires: pip install mantis-tsfm
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Constants matching src/data/paired_dataset.py
# ---------------------------------------------------------------------------
ACTION_NAMES_5 = ['OT', 'TaskOp', 'Stationary', 'Locomotion', 'Search']
ACTION_NAMES_4 = ['OT', 'TaskOp', 'Stationary', 'Locomotion']

# Mantis expects input length proportional to 32; pretrained on 512
MANTIS_SEQ_LEN = 512


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_windows(csv_path: str, processed_dir: str, num_classes: int = 5):
    """Load per-window IMU data and labels from our pipeline.

    Each NPZ file lives at <processed_dir>/<video_uid>/seq.npz with keys:
        traj      : (N_windows, 50, 6)   -- 1-sec windows at 50 Hz, 6 IMU channels
        timestamp : (N_windows, 50)       -- per-sample timestamps

    We match each labelled row (video_uid, timestamp_sec) to the closest
    window whose time range covers that timestamp (with +-0.5 s padding,
    mirroring paired_dataset.py).

    Returns:
        X : np.ndarray (M, 50, 6) -- matched IMU windows
        y : np.ndarray (M,)       -- action class labels
    """
    df = pd.read_csv(csv_path)
    if num_classes == 4:
        df = df[df['action_id'] != 4].copy()  # exclude Search

    X_all, y_all = [], []
    skipped_missing = 0
    skipped_oob = 0
    pad = 0.5  # seconds, same as paired_dataset.py

    for video_uid, group in df.groupby('video_uid'):
        npz_path = Path(processed_dir) / video_uid / 'seq.npz'
        if not npz_path.exists():
            skipped_missing += 1
            continue

        data = np.load(npz_path)
        traj = data['traj']          # (N, 50, 6)
        timestamps = data['timestamp']  # (N, 50)

        if len(traj) == 0:
            continue

        # Per-window time ranges: (start, end) for each window
        win_starts = timestamps[:, 0]   # (N,)
        win_ends = timestamps[:, -1]    # (N,)

        for _, row in group.iterrows():
            ts = row['timestamp_sec']
            label = int(row['action_id'])

            # Find windows whose time range covers this timestamp (with padding)
            mask = (win_starts <= ts + pad) & (win_ends >= ts - pad)
            candidates = np.where(mask)[0]

            if len(candidates) == 0:
                skipped_oob += 1
                continue

            # Pick the window whose center is closest to the timestamp
            centers = (win_starts[candidates] + win_ends[candidates]) / 2.0
            best = candidates[np.argmin(np.abs(centers - ts))]

            X_all.append(traj[best])  # (50, 6)
            y_all.append(label)

    print(f"  Loaded {len(X_all)} windows from {csv_path}")
    print(f"  Skipped: {skipped_missing} missing videos, {skipped_oob} OOB timestamps")

    if len(X_all) == 0:
        raise ValueError(f"No data loaded from {csv_path}!")

    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int64)


# ---------------------------------------------------------------------------
# Mantis feature extraction
# ---------------------------------------------------------------------------
def prepare_for_mantis(X: np.ndarray) -> np.ndarray:
    """Reshape and interpolate IMU windows for Mantis input.

    Input:  X of shape (N, 50, 6) -- time-last from our pipeline
    Output: X of shape (N, 6, 512) -- channels-first, length=512 for Mantis
    """
    # (N, 50, 6) -> (N, 6, 50) channels-first
    X_t = np.transpose(X, (0, 2, 1)).copy()
    # Interpolate to 512 via Mantis-recommended resize
    X_tensor = torch.tensor(X_t, dtype=torch.float32)
    X_resized = F.interpolate(X_tensor, size=MANTIS_SEQ_LEN,
                              mode='linear', align_corners=False)
    return X_resized.numpy()


def extract_frozen_features_pair(X_train_mantis, X_test_mantis, device='cuda'):
    """Extract features from frozen Mantis-8M for both train and test.

    Fits the PCA adapter on train only, then transforms both.
    Loads the model once and reuses it.

    Returns:
        feat_train: (N_train, D) numpy array
        feat_test:  (N_test, D) numpy array
    """
    from mantis.adapters import MultichannelProjector
    from mantis.architecture import MantisV1
    from mantis.trainer import MantisTrainer

    print("  Loading Mantis-8M pretrained model...")
    network = MantisV1(device=device)
    network = network.from_pretrained("paris-noah/Mantis-8M")

    print("  Fitting MultichannelProjector (PCA, 6 -> 5 channels) on train...")
    adapter = MultichannelProjector(new_num_channels=5, base_projector='pca')
    adapter.fit(X_train_mantis)

    X_train_adapted = adapter.transform(X_train_mantis)
    X_test_adapted = adapter.transform(X_test_mantis)
    print(f"  Adapted shapes: train={X_train_adapted.shape}, test={X_test_adapted.shape}")

    model = MantisTrainer(device=device, network=network)

    print("  Extracting frozen features (train)...")
    feat_train = model.transform(X_train_adapted)
    print(f"  Train feature shape: {feat_train.shape}")

    print("  Extracting frozen features (test)...")
    feat_test = model.transform(X_test_adapted)
    print(f"  Test feature shape: {feat_test.shape}")

    return feat_train, feat_test


def run_frozen_svm(feat_train, y_train, feat_test, y_test, num_classes, action_names):
    """Train LinearSVC on frozen Mantis features and evaluate."""
    scaler = StandardScaler()
    feat_train_s = scaler.fit_transform(feat_train)
    feat_test_s = scaler.transform(feat_test)

    print("  Training LinearSVC (class_weight='balanced', max_iter=10000)...")
    svm = LinearSVC(max_iter=10000, class_weight='balanced', random_state=42)
    svm.fit(feat_train_s, y_train)
    y_pred = svm.predict(feat_test_s)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    labels = list(range(num_classes))
    report = classification_report(
        y_test, y_pred, labels=labels,
        target_names=action_names[:num_classes], digits=4, zero_division=0,
    )

    return macro_f1, y_pred, report


def run_fine_tuned(X_train_mantis, y_train, X_test_mantis, y_test,
                   num_classes, action_names, device='cuda', num_epochs=20):
    """Fine-tune Mantis-8M end-to-end using MantisTrainer.fit().

    Uses LinearChannelCombiner adapter for multi-channel input +
    'adapter_head' fine-tuning mode.
    """
    from mantis.adapters import LinearChannelCombiner
    from mantis.architecture import MantisV1
    from mantis.trainer import MantisTrainer

    print("  Loading Mantis-8M pretrained model for fine-tuning...")
    network = MantisV1(device=device)
    network = network.from_pretrained("paris-noah/Mantis-8M")

    model = MantisTrainer(device=device, network=network)
    adapter = LinearChannelCombiner(
        num_channels=X_train_mantis.shape[1],  # 6
        new_num_channels=5,
    )

    print(f"  Fine-tuning Mantis-8M (adapter_head, {num_epochs} epochs)...")
    model.fit(X_train_mantis, y_train,
              adapter=adapter, fine_tuning_type='adapter_head',
              num_epochs=num_epochs)

    print("  Predicting on test set...")
    y_pred = model.predict(X_test_mantis)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    labels = list(range(num_classes))
    report = classification_report(
        y_test, y_pred, labels=labels,
        target_names=action_names[:num_classes], digits=4, zero_division=0,
    )

    return macro_f1, y_pred, report


# ---------------------------------------------------------------------------
# Fallback: handcrafted features (if Mantis unavailable)
# ---------------------------------------------------------------------------
def extract_handcrafted_features(X: np.ndarray) -> np.ndarray:
    """Statistical features as a fallback when Mantis is unavailable.

    Per-channel: mean, std, max, min, median, IQR, zero-crossing rate, energy.
    Input:  X of shape (N, 50, 6)
    Output: features of shape (N, 48)
    """
    feats = []
    feats.append(X.mean(axis=1))                          # (N, 6)
    feats.append(X.std(axis=1))                            # (N, 6)
    feats.append(X.max(axis=1))                            # (N, 6)
    feats.append(X.min(axis=1))                            # (N, 6)
    feats.append(np.median(X, axis=1))                     # (N, 6)
    feats.append(np.percentile(X, 75, axis=1) -
                 np.percentile(X, 25, axis=1))             # (N, 6) IQR
    # Zero-crossing rate per channel
    signs = np.sign(X[:, 1:, :] - X[:, :-1, :])
    zcr = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1) / X.shape[1]
    feats.append(zcr)                                      # (N, 6)
    # Energy (sum of squares)
    feats.append(np.sum(X ** 2, axis=1) / X.shape[1])     # (N, 6)

    return np.concatenate(feats, axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Mantis-8M baseline for HiT-HAR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--processed-dir', required=True,
                        help='Path to data/processed_ego4d/')
    parser.add_argument('--train-labels', required=True,
                        help='Path to train.csv')
    parser.add_argument('--test-labels', required=True,
                        help='Path to test.csv')
    parser.add_argument('--val-labels', default=None,
                        help='Optional val.csv (unused currently)')
    parser.add_argument('--num-classes', type=int, default=5, choices=[4, 5],
                        help='Number of action classes (4 excludes Search)')
    parser.add_argument('--device', default='cuda',
                        help='Device for Mantis (cuda or cpu)')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of epochs for fine-tuning (default: 20)')
    parser.add_argument('--skip-finetune', action='store_true',
                        help='Skip fine-tuning (only run frozen+SVM)')
    parser.add_argument('--finetune-only', action='store_true',
                        help='Skip frozen+SVM (only run fine-tuning)')
    parser.add_argument('--output-json', default=None,
                        help='Path to save results as JSON')
    args = parser.parse_args()

    action_names = ACTION_NAMES_5 if args.num_classes == 5 else ACTION_NAMES_4

    print("=" * 60)
    print(f"Mantis-8M Baseline ({args.num_classes}-class)")
    print("=" * 60)

    # ---- Load data ----
    print(f"\n[1/4] Loading data ({args.num_classes}-class)...")
    t0 = time.time()
    X_train, y_train = load_windows(args.train_labels, args.processed_dir,
                                    args.num_classes)
    X_test, y_test = load_windows(args.test_labels, args.processed_dir,
                                  args.num_classes)
    print(f"  Train: {len(X_train)} windows, Test: {len(X_test)} windows")
    print(f"  Train class dist: {np.bincount(y_train, minlength=args.num_classes)}")
    print(f"  Test  class dist: {np.bincount(y_test, minlength=args.num_classes)}")
    print(f"  Data loading took {time.time() - t0:.1f}s")

    # ---- Prepare Mantis input ----
    print(f"\n[2/4] Preparing Mantis input (interpolate 50 -> {MANTIS_SEQ_LEN})...")
    t0 = time.time()
    X_train_mantis = prepare_for_mantis(X_train)  # (N, 6, 512)
    X_test_mantis = prepare_for_mantis(X_test)
    print(f"  Train Mantis shape: {X_train_mantis.shape}")
    print(f"  Test  Mantis shape: {X_test_mantis.shape}")
    print(f"  Preparation took {time.time() - t0:.1f}s")

    results = {}

    # ---- Mode 1: Frozen features + SVM ----
    if not args.finetune_only:
        print(f"\n[3/4] Mode 1: Frozen Mantis-8M + LinearSVC")
        print("-" * 50)
        use_mantis = True
        try:
            t0 = time.time()
            feat_train, feat_test = extract_frozen_features_pair(
                X_train_mantis, X_test_mantis, args.device)
            print(f"  Feature extraction took {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  WARNING: Mantis loading failed: {e}")
            print("  Falling back to handcrafted features...")
            use_mantis = False
            feat_train = extract_handcrafted_features(X_train)
            feat_test = extract_handcrafted_features(X_test)
            print(f"  Handcrafted feature shape: {feat_train.shape}")

        mode1_name = "Mantis-8M frozen + SVM" if use_mantis else "Handcrafted + SVM"
        macro_f1_svm, y_pred_svm, report_svm = run_frozen_svm(
            feat_train, y_train, feat_test, y_test, args.num_classes, action_names)

        print(f"\n{'=' * 50}")
        print(f"{mode1_name} ({args.num_classes}-class)")
        print(f"Test Macro F1: {macro_f1_svm:.4f}")
        print(f"{'=' * 50}")
        print(report_svm)

        results['frozen_svm'] = {
            'method': mode1_name,
            'macro_f1': float(macro_f1_svm),
            'per_class_f1': {
                name: float(f1_score(y_test == i, y_pred_svm == i, zero_division=0))
                for i, name in enumerate(action_names[:args.num_classes])
            },
            'num_classes': args.num_classes,
        }
    else:
        print("\n[3/4] Skipping frozen+SVM (--finetune-only)")

    # ---- Mode 2: Fine-tuned Mantis ----
    if not args.skip_finetune:
        print(f"\n[4/4] Mode 2: Mantis-8M fine-tuned (adapter_head, {args.num_epochs} epochs)")
        print("-" * 50)
        try:
            t0 = time.time()
            macro_f1_ft, y_pred_ft, report_ft = run_fine_tuned(
                X_train_mantis, y_train, X_test_mantis, y_test,
                args.num_classes, action_names, args.device,
                num_epochs=args.num_epochs)
            print(f"  Fine-tuning + eval took {time.time() - t0:.1f}s")

            print(f"\n{'=' * 50}")
            print(f"Mantis-8M fine-tuned ({args.num_classes}-class)")
            print(f"Test Macro F1: {macro_f1_ft:.4f}")
            print(f"{'=' * 50}")
            print(report_ft)

            results['fine_tuned'] = {
                'method': f'Mantis-8M fine-tuned (adapter_head, {args.num_epochs}ep)',
                'macro_f1': float(macro_f1_ft),
                'per_class_f1': {
                    name: float(f1_score(y_test == i, y_pred_ft == i, zero_division=0))
                    for i, name in enumerate(action_names[:args.num_classes])
                },
                'num_classes': args.num_classes,
                'num_epochs': args.num_epochs,
            }
        except Exception as e:
            print(f"  WARNING: Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            results['fine_tuned'] = {'error': str(e)}
    else:
        print("\n[4/4] Skipping fine-tuning (--skip-finetune)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mode, res in results.items():
        if 'error' in res:
            print(f"  {mode}: FAILED - {res['error']}")
        else:
            print(f"  {res['method']}: Macro F1 = {res['macro_f1']:.4f}")
            for cls_name, cls_f1 in res['per_class_f1'].items():
                print(f"    {cls_name}: {cls_f1:.4f}")

    # ---- Save results ----
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    main()
