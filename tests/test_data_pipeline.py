"""Tests for data pipeline integrity — catches the bugs found 2026-03-22."""

import re
import tempfile

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def require_dataset_file(path):
    if not path.exists():
        pytest.skip(f"Dataset file not present: {path}")


class TestScenarioMap:
    """P0-1: SCENARIO_MAP must match scenario_labels.csv."""

    def test_all_scenarios_in_map(self):
        from src.data.paired_dataset import SCENARIO_MAP
        path = PROJECT_ROOT / 'data' / 'labels' / 'scenario_labels.csv'
        require_dataset_file(path)
        sc = pd.read_csv(path)
        for scenario in sc['scenario'].unique():
            assert scenario in SCENARIO_MAP, \
                f'"{scenario}" in scenario_labels.csv but NOT in SCENARIO_MAP'

    def test_no_mech_repair_typo(self):
        from src.data.paired_dataset import SCENARIO_MAP
        assert 'Mech Repair' not in SCENARIO_MAP, \
            'SCENARIO_MAP has "Mech Repair" — should be "Mechanical Repair"'
        assert 'Mechanical Repair' in SCENARIO_MAP


class TestSplitIntegrity:
    """P0-2: No video overlap between splits."""

    @pytest.fixture
    def splits(self):
        train_path = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
        val_path = PROJECT_ROOT / 'data' / 'processed' / 'val.csv'
        test_path = PROJECT_ROOT / 'data' / 'processed' / 'test.csv'
        for path in [train_path, val_path, test_path]:
            require_dataset_file(path)
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        return train, val, test

    def test_no_video_overlap(self, splits):
        train, val, test = splits
        train_v = set(train['video_uid'])
        val_v = set(val['video_uid'])
        test_v = set(test['video_uid'])
        assert len(train_v & val_v) == 0, 'Train/val video overlap!'
        assert len(train_v & test_v) == 0, 'Train/test video overlap!'
        assert len(val_v & test_v) == 0, 'Val/test video overlap!'

    def test_no_llm_source(self, splits):
        """No LLM-only rows should remain after fix."""
        train, val, test = splits
        for name, df in [('train', train), ('val', val), ('test', test)]:
            assert 'llm' not in df['source'].values, \
                f'{name}.csv still contains LLM rows!'


class TestNoOtherPersonLeakage:
    """P0-3: #O narrations must not appear in processed data."""

    def test_no_O_in_processed(self):
        for split in ['train', 'val', 'test']:
            path = PROJECT_ROOT / 'data' / 'processed' / f'{split}.csv'
            require_dataset_file(path)
            df = pd.read_csv(path)
            has_O = df['narration_text'].apply(
                lambda t: bool(re.search(r'#O\b', str(t), re.IGNORECASE))
            )
            n_O = has_O.sum()
            assert n_O == 0, \
                f'{split}.csv has {n_O} #O rows (other person narrations)!'


class TestActionMap:
    """ACTION_MAP consistency between clean_labels and paired_dataset."""

    def test_action_maps_match(self):
        from src.data.clean_labels import ACTION_MAP as CL_MAP
        from src.data.paired_dataset import ACTION_MAP as PD_MAP
        assert CL_MAP == PD_MAP, f'ACTION_MAP mismatch: {CL_MAP} vs {PD_MAP}'

    def test_no_essential_operation(self):
        for split in ['train', 'val', 'test']:
            path = PROJECT_ROOT / 'data' / 'processed' / f'{split}.csv'
            require_dataset_file(path)
            df = pd.read_csv(path)
            assert 'Essential Operation' not in df['action'].values, \
                f'{split}.csv still has "Essential Operation"'


# ---------------------------------------------------------------------------
# Stationary noise filter tests (uses mock data, no real NPZ files needed)
# ---------------------------------------------------------------------------

class TestStationaryNoiseFilter:
    """Tests for scripts/filter_stationary_noise.py flag_noisy_stationary()."""

    @pytest.fixture
    def mock_imu_dir(self, tmp_path):
        """Create a temporary IMU directory with one fake NPZ file.

        The NPZ has 10 windows, each 50 samples at 1-second spacing.
        Window i covers timestamps [i*1.0 .. i*1.0 + 0.49*1.0] (50 samples).
        """
        uid = 'test-video-001'
        uid_dir = tmp_path / uid
        uid_dir.mkdir()

        n_windows = 10
        n_samples = 50

        # Timestamps: window i spans [i, i + 0.98] with 50 evenly spaced samples
        timestamps = np.zeros((n_windows, n_samples))
        for i in range(n_windows):
            timestamps[i] = np.linspace(i, i + 0.98, n_samples)

        # Default: low-energy IMU (near-zero accel and gyro) — true stationary
        traj = np.random.randn(n_windows, n_samples, 6) * 0.01

        # Window 3: inject HIGH energy — this should trigger the noise flag
        # Large accel norms and gyro variation
        traj[3, :, :3] = np.random.randn(n_samples, 3) * 5.0 + 3.0  # accel
        traj[3, :, 3:6] = np.random.randn(n_samples, 3) * 4.0        # gyro

        # Window 7: inject moderate energy — only accel mean threshold exceeded
        traj[7, :, :3] = np.array([1.2, 0.0, 0.0])
        traj[7, :, 3:6] = 0.01

        np.savez(uid_dir / 'seq.npz', traj=traj, timestamp=timestamps)
        return tmp_path, uid, timestamps

    @pytest.fixture
    def thresholds(self):
        """Use the q92 threshold preset."""
        return {
            'accel_norm_mean': 0.9625,
            'accel_norm_std': 0.5796,
            'gyro_norm_std': 1.2810,
        }

    def test_flags_high_energy_stationary(self, mock_imu_dir, thresholds):
        """A Stationary row matching a high-energy IMU window should be flagged."""
        from scripts.filter_stationary_noise import flag_noisy_stationary

        imu_dir, uid, timestamps = mock_imu_dir

        # Row at timestamp ~3.5 should match window 3 (high energy)
        # Row at timestamp ~0.5 should match window 0 (low energy, safe)
        # Row at timestamp ~7.5 should match window 7 (moderate, only 1 threshold)
        df = pd.DataFrame({
            'video_uid': [uid, uid, uid, uid],
            'timestamp_sec': [3.5, 0.5, 7.5, 5.0],
            'action': ['Stationary', 'Stationary', 'Stationary', 'Locomotion'],
            'source': ['propagated', 'propagated', 'propagated', 'gold'],
            'tier': [2, 2, 2, 1],
        })

        noisy = flag_noisy_stationary(df, Path(imu_dir), thresholds)

        # Window 3 (ts=3.5) should be flagged: high energy, >=2 thresholds
        assert noisy[0] is True or noisy[0] == True, \
            'High-energy Stationary at window 3 should be flagged as noisy'

        # Window 0 (ts=0.5) should NOT be flagged: low energy
        assert noisy[1] is False or noisy[1] == False, \
            'Low-energy Stationary at window 0 should NOT be flagged'

        # Window 7 (ts=7.5): only 1 threshold exceeded → NOT flagged
        assert noisy[2] is False or noisy[2] == False, \
            'Moderate-energy Stationary (1 threshold) should NOT be flagged'

        # Locomotion row should never be flagged (not Stationary)
        assert noisy[3] is False or noisy[3] == False, \
            'Non-Stationary rows should never be flagged'

    def test_preserves_gold_tier1(self, mock_imu_dir, thresholds):
        """Gold tier-1 Stationary rows must NEVER be flagged, even with high energy."""
        from scripts.filter_stationary_noise import flag_noisy_stationary

        imu_dir, uid, timestamps = mock_imu_dir

        # Both rows point to window 3 (high energy), but one is gold tier-1
        df = pd.DataFrame({
            'video_uid': [uid, uid],
            'timestamp_sec': [3.5, 3.5],
            'action': ['Stationary', 'Stationary'],
            'source': ['gold', 'propagated'],
            'tier': [1, 2],
        })

        noisy = flag_noisy_stationary(df, Path(imu_dir), thresholds)

        # Gold tier-1 must be preserved
        assert noisy[0] is False or noisy[0] == False, \
            'Gold tier-1 Stationary must NEVER be flagged as noisy'

        # Propagated tier-2 at same high-energy window should be flagged
        assert noisy[1] is True or noisy[1] == True, \
            'Propagated tier-2 Stationary at high-energy window should be flagged'
