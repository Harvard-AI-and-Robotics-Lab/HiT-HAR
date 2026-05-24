# tests/test_tier_system.py
"""Tests for tier assignment system."""
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestTierAssignment:
    def test_output_has_required_columns(self):
        """Tier assignments CSV must have these columns."""
        path = PROJECT_ROOT / 'data' / 'processed' / 'tier_assignments.csv'
        if not path.exists():
            pytest.skip("Run assign_tiers.py first")
        df = pd.read_csv(path)
        required = ['video_uid', 'timestamp_sec', 'tier', 'confidence',
                     'has_secondary', 'secondary_action']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_tiers_are_valid(self):
        path = PROJECT_ROOT / 'data' / 'processed' / 'tier_assignments.csv'
        if not path.exists():
            pytest.skip("Run assign_tiers.py first")
        df = pd.read_csv(path)
        assert set(df['tier'].unique()).issubset({1, 2, 3, 4})

    def test_confidence_matches_tier(self):
        path = PROJECT_ROOT / 'data' / 'processed' / 'tier_assignments.csv'
        if not path.exists():
            pytest.skip("Run assign_tiers.py first")
        df = pd.read_csv(path)
        for _, row in df.sample(min(100, len(df))).iterrows():
            if row['tier'] == 1:
                assert row['confidence'] == 1.0
            elif row['tier'] == 2:
                assert row['confidence'] == 0.8
            elif row['tier'] == 3:
                assert row['confidence'] in (0.5, 0.6, 0.7)
            elif row['tier'] == 4:
                assert row['confidence'] == 0.0

    def test_no_tier4_in_training_data(self):
        """After pipeline runs, train.csv should have no tier 4."""
        path = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
        if not path.exists():
            pytest.skip("Run clean_labels.py first")
        df = pd.read_csv(path)
        if 'tier' in df.columns:
            assert (df['tier'] != 4).all(), "Tier 4 samples should be excluded"

    def test_r001_covered(self):
        """R001 samples should be assigned tiers too."""
        path = PROJECT_ROOT / 'data' / 'processed' / 'tier_assignments.csv'
        if not path.exists():
            pytest.skip("Run assign_tiers.py first")
        df = pd.read_csv(path)
        # R001 had ~17K samples, at least some should be present
        assert len(df) > 15000, f"Only {len(df)} tier assignments, expected >15K"
