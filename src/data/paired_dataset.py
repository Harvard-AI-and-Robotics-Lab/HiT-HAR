"""
Paired Dataset for HiT-HAR training.

Extends the old HierarchicalDataset pattern with:
  - 5-class action labels
  - Confidence weights per window
  - Data augmentation (jittering, scaling)

Loads IMU data from NPZ files.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


ACTION_MAP = {
    'Object Transfer': 0,
    'Task Operation': 1,
    'Stationary': 2,
    'Locomotion': 3,
    'Search': 4,
}

SCENARIO_MAP = {
    'Carpentry': 0,
    'Cleaning': 1,
    'Cooking': 2,
    'Desk Work': 3,
    'Gardening': 4,
    'Mechanical Repair': 5,
    'Playing Instrument': 6,
    'Walking Outdoors': 7,
}


class PairedDataset(Dataset):
    """Dataset pairing IMU windows with labels."""

    def __init__(self, take_uids, processed_dir, action_labels_path,
                 scenario_labels_path, config,
                 training=True, norm_stats=None):
        """
        Args:
            take_uids: list of video UIDs to include
            processed_dir: path to processed_ego4d/ with <uid>/seq.npz
            action_labels_path: path to cleaned 5-class action CSV
            scenario_labels_path: path to scenario_labels.csv
            config: config dict
            training: enable augmentation
        """
        self.samples = []
        self.config = config
        self.training = training

        seq_len = config['hla']['seq_len']  # 30
        stride = config['data'].get('stride', 5)
        action_pad = config['data'].get('action_label_pad', 0.5)
        add_norms = config['data'].get('add_norm_features', True)
        per_video_center = config['data'].get('per_video_center', True)
        augmentation = config['data'].get('augmentation', False) and training

        self.augmentation = augmentation
        self.num_action_classes = len(ACTION_MAP)

        # Load labels
        print("Loading labels...")
        scenario_df = pd.read_csv(scenario_labels_path).set_index('video_uid')
        action_df = pd.read_csv(action_labels_path)

        # Filter to valid actions
        action_df = action_df[action_df['action'].isin(ACTION_MAP.keys())].copy()
        if 'action_id' not in action_df.columns:
            action_df['action_id'] = action_df['action'].map(ACTION_MAP)

        # Support excluding action classes (e.g., for 4-class mode without Search)
        exclude_actions = config.get('exclude_actions', [])
        if exclude_actions:
            before = len(action_df)
            action_df = action_df[~action_df['action'].isin(exclude_actions)].copy()
            print(f"Excluded actions {exclude_actions}: {before} -> {len(action_df)} rows")

        # Build scenario map
        self.scenario_map = SCENARIO_MAP
        self.num_scenarios = len(self.scenario_map)

        # ---- Global normalization stats ----
        # Use provided stats (from train set) to avoid split leakage
        if norm_stats is not None:
            self.global_mean = norm_stats['mean']
            self.global_std = norm_stats['std']
            print(f"Using provided normalization stats (shape: {self.global_mean.shape})")
            # Still need to find valid UIDs
            valid_uids = []
            for uid in tqdm(take_uids, desc='Checking data'):
                seq_path = Path(processed_dir) / uid / 'seq.npz'
                if seq_path.exists() and uid in scenario_df.index:
                    valid_uids.append(uid)
        else:
            print("Computing global normalization statistics...")
            all_data = []
            valid_uids = []

            for uid in tqdm(take_uids, desc='Collecting stats'):
                seq_path = Path(processed_dir) / uid / 'seq.npz'
                if not seq_path.exists():
                    continue
                try:
                    data = np.load(seq_path)
                    traj = data['traj']  # (N, 50, 6)
                    if add_norms:
                        traj = _add_norm_features(traj)
                    if len(traj) > 0 and uid in scenario_df.index:
                        all_data.append(traj)
                        valid_uids.append(uid)
                except Exception:
                    continue

            if len(all_data) == 0:
                raise ValueError("No valid IMU data found!")

            all_concat = np.concatenate(all_data, axis=0)
            self.global_mean = all_concat.mean(axis=(0, 1))
            self.global_std = all_concat.std(axis=(0, 1)) + 1e-6
            print(f"Normalization from {len(valid_uids)} videos, shape: {self.global_mean.shape}")

        # ---- Build samples ----
        print("Building samples...")
        n_no_scenario = 0

        for uid in tqdm(valid_uids, desc='Building samples'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            data = np.load(seq_path)
            traj = data['traj']
            timestamps = data['timestamp']  # (N, 50)

            if add_norms:
                traj = _add_norm_features(traj)

            # Global normalize + per-video center
            traj = (traj - self.global_mean) / self.global_std
            if per_video_center:
                traj = traj - traj.mean(axis=(0, 1), keepdims=True)

            # Scenario label
            if uid not in scenario_df.index:
                n_no_scenario += 1
                continue
            scenario_name = scenario_df.loc[uid, 'scenario']
            if scenario_name not in self.scenario_map:
                continue
            scenario_label = self.scenario_map[scenario_name]

            # Action labels for this video
            video_actions = action_df[action_df['video_uid'] == uid]

            # Sliding window sequences
            num_windows = len(traj)
            num_seqs = (num_windows - seq_len) // stride + 1
            if num_seqs <= 0:
                continue

            for i in range(num_seqs):
                start_idx = i * stride
                end_idx = start_idx + seq_len

                window_seq = traj[start_idx:end_idx]       # (seq_len, 50, C)
                ts_windows = timestamps[start_idx:end_idx]  # (seq_len, 50)

                # Match action labels per window
                action_labels = []
                confidences = []
                for w_idx in range(seq_len):
                    w_start = ts_windows[w_idx][0]
                    w_end = ts_windows[w_idx][-1]
                    match = video_actions[
                        (video_actions['timestamp_sec'] >= w_start - action_pad) &
                        (video_actions['timestamp_sec'] <= w_end + action_pad)
                    ]
                    if not match.empty:
                        counts = match['action'].value_counts()
                        act_name = counts.idxmax()
                        if act_name in ACTION_MAP:
                            action_labels.append(ACTION_MAP[act_name])
                            if 'confidence' in match.columns:
                                confidences.append(match['confidence'].mean())
                            else:
                                confidences.append(1.0)
                        else:
                            action_labels.append(-1)
                            confidences.append(0.0)
                    else:
                        action_labels.append(-1)
                        confidences.append(0.0)

                if len(action_labels) != seq_len:
                    continue

                sample = {
                    'video_uid': uid,
                    'inputs': torch.tensor(np.ascontiguousarray(window_seq), dtype=torch.float32),
                    'scenario_label': torch.tensor(scenario_label, dtype=torch.long),
                    'action_labels': torch.tensor(action_labels, dtype=torch.long),
                    'confidence': torch.tensor(confidences, dtype=torch.float32),
                }

                self.samples.append(sample)

        print(f"\nDataset built: {len(self.samples)} sequences")
        print(f"  Videos without scenario: {n_no_scenario}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.augmentation:
            inputs = sample['inputs'].clone()
            inputs = _apply_augmentation(inputs)
            result = {k: v for k, v in sample.items() if k != 'inputs' and k != 'video_uid'}
            result['inputs'] = inputs
            return result

        return {k: v for k, v in sample.items() if k != 'video_uid'}


def _add_norm_features(traj):
    """Add accel/gyro norm channels: (N, 50, 6) → (N, 50, 8)."""
    accel = traj[..., :3]
    gyro = traj[..., 3:6]
    accel_norm = np.linalg.norm(accel, axis=2, keepdims=True)
    gyro_norm = np.linalg.norm(gyro, axis=2, keepdims=True)
    return np.concatenate([traj, accel_norm, gyro_norm], axis=2)


def _apply_augmentation(x):
    """Apply jittering + scaling + time masking augmentation.
    x: (seq_len, 50, channels) tensor

    Uses torch RNG (not Python random) for reproducibility with
    DataLoader worker_init_fn seeding.
    """
    if torch.rand(1).item() < 0.5:
        noise = torch.randn_like(x) * 0.02
        x = x + noise
    if torch.rand(1).item() < 0.5:
        scale = 0.9 + torch.rand(1).item() * 0.2  # uniform(0.9, 1.1)
        x = x * scale
    # Time masking: zero out 5-sample spans, p=0.3
    if torch.rand(1).item() < 0.3:
        S, W, C = x.shape
        n_masks = torch.randint(1, 4, (1,)).item()
        for _ in range(n_masks):
            w_idx = torch.randint(0, S, (1,)).item()
            start = torch.randint(0, max(1, W - 4), (1,)).item()
            x[w_idx, start:start + 5, :] = 0.0
    # Rotation augmentation: random small 3D rotation on accel+gyro (p=0.5)
    # Critical for head-mounted IMU where orientation varies between wearings
    if torch.rand(1).item() < 0.5:
        # Random rotation angle (±15 degrees)
        angle = (torch.rand(3) - 0.5) * 0.52  # ~±15° in radians
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)

        # Rotation matrices for each axis
        Rx = torch.tensor([[1, 0, 0], [0, cos_a[0], -sin_a[0]], [0, sin_a[0], cos_a[0]]])
        Ry = torch.tensor([[cos_a[1], 0, sin_a[1]], [0, 1, 0], [-sin_a[1], 0, cos_a[1]]])
        Rz = torch.tensor([[cos_a[2], -sin_a[2], 0], [sin_a[2], cos_a[2], 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx  # Combined rotation

        S, W, C = x.shape
        # Rotate accel (channels 0:3) and gyro (channels 3:6)
        if C >= 6:
            accel = x[:, :, :3]  # (S, W, 3)
            gyro = x[:, :, 3:6]  # (S, W, 3)
            accel_rot = torch.einsum('ij,swj->swi', R, accel)
            gyro_rot = torch.einsum('ij,swj->swi', R, gyro)
            x = x.clone()
            x[:, :, :3] = accel_rot
            x[:, :, 3:6] = gyro_rot
            # Recompute norms if present (channels 6:8)
            if C >= 8:
                x[:, :, 6] = torch.norm(accel_rot, dim=2)
                x[:, :, 7] = torch.norm(gyro_rot, dim=2)
    return x


def collate_fn(batch):
    """Custom collate that handles optional fields safely.

    Uses intersection of keys across all samples to avoid KeyError.
    """
    # Use intersection of keys to be safe
    all_keys = [set(b.keys()) for b in batch]
    common_keys = all_keys[0]
    for ks in all_keys[1:]:
        common_keys = common_keys & ks
    common_keys.discard('video_uid')

    collated = {}
    for k in common_keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals)
        else:
            collated[k] = vals

    return collated


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker for reproducibility.

    Each worker gets a unique seed derived from the base seed + worker_id,
    ensuring different workers produce different augmentations while
    remaining reproducible across runs.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(config, processed_dir, train_labels_path, val_labels_path,
                      scenario_labels_path):
    """Build train/val dataloaders from pre-split label CSVs.

    Video UIDs are derived from the label CSVs themselves (not scenario_labels.csv split column).
    scenario_labels.csv is still needed for scenario name -> ID mapping per video.
    """
    train_df = pd.read_csv(train_labels_path)
    val_df = pd.read_csv(val_labels_path)

    train_uids = train_df['video_uid'].unique().tolist()
    val_uids = val_df['video_uid'].unique().tolist()

    # Sanity: no overlap
    overlap = set(train_uids) & set(val_uids)
    if overlap:
        raise ValueError(f"Train/val video overlap: {len(overlap)} videos! "
                         f"First 5: {list(overlap)[:5]}")

    print(f"Train videos: {len(train_uids)}, Val videos: {len(val_uids)}")

    train_ds = PairedDataset(
        train_uids, processed_dir, train_labels_path, scenario_labels_path,
        config, training=True,
        norm_stats=None,
    )
    train_norm = {'mean': train_ds.global_mean, 'std': train_ds.global_std}
    val_ds = PairedDataset(
        val_uids, processed_dir, val_labels_path, scenario_labels_path,
        config, training=False,
        norm_stats=train_norm,
    )

    bs = config['training']['batch_size']
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, val_loader, train_ds, val_ds
