#!/usr/bin/env python3
"""L2-1: IMU Feature Extraction for Gold-Labeled Windows.

Extracts per-window statistical features from raw IMU NPZ files for all
gold-labeled timestamps. Produces a feature CSV used by downstream L2 analyses.

Features per window (1s = 50 samples):
  Per axis (ax, ay, az, gx, gy, gz) — 6 axes:
    mean, std, min, max, energy (sum of squares), zero_crossing_rate
  Global:
    accel_norm_mean, accel_norm_std, gyro_norm_mean, gyro_norm_std
    dominant_freq_accel, dominant_freq_gyro (FFT peak)

Total: 6 axes x 6 stats + 6 global = 42 features per window

Output:
  - analysis/layer2_signals/outputs/gold_imu_features.csv
    Columns: video_uid, timestamp_sec, action, scenario, tier, confidence,
             ax_mean, ax_std, ..., dominant_freq_gyro

Usage:
    python analysis/layer2_signals/l2_feature_extraction.py \
        --imu-dir data/processed_ego4d \
        --output analysis/layer2_signals/outputs/gold_imu_features.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.shared.data_loader import load_gold, load_tier_assignments, ACTION_ORDER

AXIS_NAMES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SAMPLES = 50  # 1s @ 50Hz
PAD_SEC = 0.5


def extract_window_features(traj_window):
    """Extract statistical features from a (N, 6) IMU window.

    Args:
        traj_window: numpy array of shape (N, 6) — [ax, ay, az, gx, gy, gz]
    Returns:
        dict of feature_name -> value
    """
    features = {}
    N = len(traj_window)

    if N < 5:
        # Too few samples, return NaN
        for axis in AXIS_NAMES:
            for stat in ['mean', 'std', 'min', 'max', 'energy', 'zcr']:
                features[f'{axis}_{stat}'] = np.nan
        for key in ['accel_norm_mean', 'accel_norm_std', 'gyro_norm_mean', 'gyro_norm_std',
                     'dominant_freq_accel', 'dominant_freq_gyro']:
            features[key] = np.nan
        return features

    for i, axis in enumerate(AXIS_NAMES):
        signal = traj_window[:, i]
        features[f'{axis}_mean'] = np.mean(signal)
        features[f'{axis}_std'] = np.std(signal)
        features[f'{axis}_min'] = np.min(signal)
        features[f'{axis}_max'] = np.max(signal)
        features[f'{axis}_energy'] = np.sum(signal ** 2) / N
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(signal - np.mean(signal)))) > 0) / max(N - 1, 1)
        features[f'{axis}_zcr'] = zcr

    # Norms
    accel_norm = np.linalg.norm(traj_window[:, :3], axis=1)
    gyro_norm = np.linalg.norm(traj_window[:, 3:6], axis=1)
    features['accel_norm_mean'] = np.mean(accel_norm)
    features['accel_norm_std'] = np.std(accel_norm)
    features['gyro_norm_mean'] = np.mean(gyro_norm)
    features['gyro_norm_std'] = np.std(gyro_norm)

    # Dominant frequency (FFT)
    if N >= 10:
        for name, signal in [('accel', accel_norm), ('gyro', gyro_norm)]:
            fft_vals = np.abs(np.fft.rfft(signal - np.mean(signal)))
            freqs = np.fft.rfftfreq(N, d=1.0/50)  # 50Hz
            if len(fft_vals) > 1:
                # Skip DC component
                peak_idx = np.argmax(fft_vals[1:]) + 1
                features[f'dominant_freq_{name}'] = freqs[peak_idx]
            else:
                features[f'dominant_freq_{name}'] = 0
    else:
        features['dominant_freq_accel'] = 0
        features['dominant_freq_gyro'] = 0

    return features


def main():
    parser = argparse.ArgumentParser(description='Extract IMU features for gold samples')
    parser.add_argument('--imu-dir', type=str, required=True,
                        help='Path to processed_ego4d/ directory with NPZ files')
    parser.add_argument('--output', type=str,
                        default='analysis/layer2_signals/outputs/gold_imu_features.csv')
    args = parser.parse_args()

    imu_dir = Path(args.imu_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("L2-1: IMU Feature Extraction for Gold-Labeled Windows")
    print("=" * 60)

    gold = load_gold()
    tiers = load_tier_assignments()

    # Join tier info
    gold = gold.merge(
        tiers[['video_uid', 'timestamp_sec', 'tier', 'confidence']],
        on=['video_uid', 'timestamp_sec'], how='left', suffixes=('', '_tier')
    )
    gold['tier'] = gold['tier'].fillna(2).astype(int)
    gold['confidence'] = gold.get('confidence_tier', gold.get('confidence', 0.8))

    # Group by video for efficient NPZ loading
    video_groups = gold.groupby('video_uid')
    print(f"Gold samples: {len(gold)}, Videos: {len(video_groups)}")

    all_rows = []
    missing_videos = 0

    for vid, grp in tqdm(video_groups, desc='Extracting features'):
        # NPZ files are stored as {video_uid}/seq.npz
        npz_path = imu_dir / vid / 'seq.npz'
        if not npz_path.exists():
            # Fallback: try flat {video_uid}.npz
            npz_path = imu_dir / f'{vid}.npz'
        if not npz_path.exists():
            missing_videos += 1
            continue

        data = np.load(npz_path)
        traj = data['traj']        # (N, 6)
        timestamps = data['timestamp']  # (N,)

        for _, row in grp.iterrows():
            ts = row['timestamp_sec']
            # Find samples within +/-PAD_SEC of timestamp
            mask = (timestamps >= ts - PAD_SEC) & (timestamps <= ts + PAD_SEC)
            window = traj[mask]

            if len(window) < 5:
                continue

            # Truncate to WINDOW_SAMPLES
            if len(window) > WINDOW_SAMPLES:
                center = len(window) // 2
                start = max(0, center - WINDOW_SAMPLES // 2)
                window = window[start:start + WINDOW_SAMPLES]

            features = extract_window_features(window)
            features['video_uid'] = vid
            features['timestamp_sec'] = ts
            features['action'] = row['action']
            features['scenario'] = row['scenario']
            features['tier'] = row['tier']
            features['confidence'] = row.get('confidence', 0.8)
            all_rows.append(features)

    result = pd.DataFrame(all_rows)
    print(f"\nExtracted features for {len(result)} windows "
          f"(missing {missing_videos} videos)")
    print(f"Feature columns: {len([c for c in result.columns if c not in ['video_uid', 'timestamp_sec', 'action', 'scenario', 'tier', 'confidence']])}")

    result.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
