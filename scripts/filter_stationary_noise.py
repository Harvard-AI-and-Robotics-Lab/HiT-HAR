#!/usr/bin/env python3
"""Filter noisy Stationary labels from train/val/test CSVs by checking IMU energy.

Many narrations labeled "Stationary" actually correspond to high-energy IMU windows
(e.g., the camera wearer is walking while the narration describes a static scene).
This script identifies and removes those noisy rows.

**Algorithm:**
  1. For each row where action == 'Stationary', locate the matching IMU window
     by timestamp (±0.5 s padding, mirroring paired_dataset.py:172-177).
  2. Compute three energy features for the matched raw IMU window:
       - accel_norm_mean: mean of per-sample accel L2 norms
       - accel_norm_std:  std  of per-sample accel L2 norms
       - gyro_norm_std:   std  of per-sample gyro  L2 norms
  3. If ≥ 2 of 3 thresholds are exceeded AND the row is NOT gold tier-1,
     flag the row as noisy.
  4. Write a clean CSV with noisy rows removed.

Thresholds (q92 on Stationary training data):
  accel_norm_mean > 0.9625
  accel_norm_std  > 0.5796
  gyro_norm_std   > 1.2810

Usage:
  python scripts/filter_stationary_noise.py \
      --input-dir  data/processed \
      --imu-dir    data/processed_ego4d \
      --output-dir data/processed_clean \
      --threshold  q92
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Threshold presets (estimated from Stationary training distribution)
# ---------------------------------------------------------------------------
THRESHOLD_PRESETS = {
    'q92': {
        'accel_norm_mean': 0.9625,
        'accel_norm_std': 0.5796,
        'gyro_norm_std': 1.2810,
    },
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_imu_data(imu_dir: Path, uid: str):
    """Load IMU arrays for a single video UID.

    Returns (traj, timestamps) or (None, None) if the NPZ doesn't exist.
      traj:       np.ndarray of shape (N, 50, 6)  — accel(3) + gyro(3)
      timestamps: np.ndarray of shape (N, 50)
    """
    seq_path = imu_dir / uid / 'seq.npz'
    if not seq_path.exists():
        return None, None
    data = np.load(seq_path)
    return data['traj'], data['timestamp']


def find_matching_window(timestamp_sec: float, timestamps: np.ndarray,
                         pad: float = 0.5):
    """Find the IMU window whose time range covers *timestamp_sec* (±pad).

    Mirrors the matching logic in ``paired_dataset.py`` lines 172-177::

        w_start = ts_windows[w_idx][0]
        w_end   = ts_windows[w_idx][-1]
        match   = (timestamp_sec >= w_start - pad) &
                  (timestamp_sec <= w_end   + pad)

    Parameters
    ----------
    timestamp_sec : float
        Narration timestamp (seconds).
    timestamps : np.ndarray, shape (N, 50)
        Per-window timestamp arrays from the NPZ file.
    pad : float
        Symmetric padding in seconds (default 0.5 s).

    Returns
    -------
    int or None
        Index of the best matching window, or None if no match.
    """
    w_starts = timestamps[:, 0]   # (N,)
    w_ends = timestamps[:, -1]    # (N,)

    mask = (timestamp_sec >= w_starts - pad) & (timestamp_sec <= w_ends + pad)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None

    # If multiple windows match, pick the one whose midpoint is closest.
    midpoints = (w_starts[indices] + w_ends[indices]) / 2.0
    best = indices[np.argmin(np.abs(midpoints - timestamp_sec))]
    return int(best)


def compute_window_energy(traj_window: np.ndarray):
    """Compute energy features for a single raw IMU window.

    Parameters
    ----------
    traj_window : np.ndarray, shape (50, 6)
        Raw accel (0:3) + gyro (3:6) samples.

    Returns
    -------
    dict with keys: accel_norm_mean, accel_norm_std, gyro_norm_std
    """
    accel = traj_window[:, :3]
    gyro = traj_window[:, 3:6]

    accel_norms = np.linalg.norm(accel, axis=1)  # (50,)
    gyro_norms = np.linalg.norm(gyro, axis=1)    # (50,)

    return {
        'accel_norm_mean': float(np.mean(accel_norms)),
        'accel_norm_std': float(np.std(accel_norms)),
        'gyro_norm_std': float(np.std(gyro_norms)),
    }


def flag_noisy_stationary(df: pd.DataFrame, imu_dir: Path,
                          thresholds: dict, pad: float = 0.5):
    """Flag noisy Stationary rows by checking IMU energy.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: action, video_uid, timestamp_sec, tier.
    imu_dir : Path
        Directory containing ``<uid>/seq.npz`` files.
    thresholds : dict
        Keys: accel_norm_mean, accel_norm_std, gyro_norm_std.
    pad : float
        Timestamp matching padding (seconds).

    Returns
    -------
    np.ndarray of bool
        True for rows that are flagged as noisy.
    """
    n = len(df)
    noisy = np.zeros(n, dtype=bool)

    # Only evaluate Stationary rows
    stationary_mask = df['action'] == 'Stationary'
    if stationary_mask.sum() == 0:
        return noisy

    # Gold tier-1 rows are never flagged
    has_tier = 'tier' in df.columns
    if has_tier:
        gold_tier1_mask = (df['source'] == 'gold') & (df['tier'] == 1)
    else:
        gold_tier1_mask = pd.Series(False, index=df.index)

    # Cache loaded IMU data per UID
    imu_cache = {}

    for idx in df.index[stationary_mask]:
        # Skip gold tier-1
        if gold_tier1_mask.loc[idx]:
            continue

        uid = df.loc[idx, 'video_uid']
        ts = df.loc[idx, 'timestamp_sec']

        # Load IMU (with caching)
        if uid not in imu_cache:
            traj, timestamps = load_imu_data(imu_dir, uid)
            imu_cache[uid] = (traj, timestamps)
        else:
            traj, timestamps = imu_cache[uid]

        if traj is None:
            continue

        # Find matching window by timestamp
        win_idx = find_matching_window(ts, timestamps, pad=pad)
        if win_idx is None:
            continue

        # Compute energy
        energy = compute_window_energy(traj[win_idx])

        # Count threshold violations
        violations = 0
        if energy['accel_norm_mean'] > thresholds['accel_norm_mean']:
            violations += 1
        if energy['accel_norm_std'] > thresholds['accel_norm_std']:
            violations += 1
        if energy['gyro_norm_std'] > thresholds['gyro_norm_std']:
            violations += 1

        if violations >= 2:
            noisy[df.index.get_loc(idx)] = True

    return noisy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Filter noisy Stationary labels using IMU energy checks')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with train.csv / val.csv / test.csv')
    parser.add_argument('--imu-dir', type=str, required=True,
                        help='Directory with <uid>/seq.npz IMU data')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for cleaned output CSVs')
    parser.add_argument('--threshold', type=str, default='q92',
                        choices=list(THRESHOLD_PRESETS.keys()),
                        help='Threshold preset name (default: q92)')
    parser.add_argument('--pad', type=float, default=0.5,
                        help='Timestamp matching padding in seconds (default: 0.5)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    imu_dir = Path(args.imu_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = THRESHOLD_PRESETS[args.threshold]
    print(f"Threshold preset: {args.threshold}")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")

    for split in ['train', 'val', 'test']:
        csv_path = input_dir / f'{split}.csv'
        if not csv_path.exists():
            print(f"\n[{split}] {csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        n_total = len(df)
        n_stat = (df['action'] == 'Stationary').sum()
        print(f"\n[{split}] {n_total} rows, {n_stat} Stationary")

        noisy = flag_noisy_stationary(df, imu_dir, thresholds, pad=args.pad)
        n_noisy = noisy.sum()

        clean_df = df[~noisy].copy()
        out_path = output_dir / f'{split}.csv'
        clean_df.to_csv(out_path, index=False)

        print(f"  Flagged {n_noisy} noisy Stationary rows "
              f"({n_noisy / max(n_stat, 1) * 100:.1f}% of Stationary)")
        print(f"  Clean: {len(clean_df)} rows -> {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
