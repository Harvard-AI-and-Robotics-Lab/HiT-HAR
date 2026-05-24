"""Tests for analysis shared utilities."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.shared.plot_style import apply_style, save_figure, COLORS_5CLASS, COLORS_4CLASS
from analysis.shared.data_loader import (
    load_gold, load_llm, load_r1_raw, load_r2_raw, load_tier_assignments,
    normalize_narration, extract_verb, ACTION_ORDER, SCENARIO_ORDER,
    ACTION_ORDER_4CLASS, ACTION_ORDER_3CLASS, map_to_4class, map_to_3class,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def require_dataset_file(path):
    if not path.exists():
        pytest.skip(f"Dataset file not present: {path}")


class TestPlotStyle:
    def test_apply_style_sets_rcparams(self):
        apply_style()
        assert plt.rcParams['font.size'] >= 8
        assert plt.rcParams['figure.dpi'] >= 150

    def test_colors_5class_has_5_entries(self):
        assert len(COLORS_5CLASS) == 5

    def test_colors_4class_has_4_entries(self):
        assert len(COLORS_4CLASS) == 4

    def test_save_figure_creates_files(self, tmp_path):
        apply_style()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        save_figure(fig, tmp_path / 'test_fig')
        assert (tmp_path / 'test_fig.pdf').exists()
        assert (tmp_path / 'test_fig.png').exists()
        plt.close(fig)


class TestDataLoader:
    def test_normalize_narration_removes_hashtags(self):
        assert normalize_narration('#C C picks up the cup.') == 'c picks up the cup'

    def test_normalize_narration_strips_punctuation(self):
        assert normalize_narration('C opens the door!!!') == 'c opens the door'

    def test_map_to_4class(self):
        assert map_to_4class('Object Transfer') == 'Manipulation'
        assert map_to_4class('Task Operation') == 'Manipulation'
        assert map_to_4class('Locomotion') == 'Locomotion'
        assert map_to_4class('Stationary') == 'Stationary'
        assert map_to_4class('Search') == 'Search'

    def test_map_to_3class(self):
        assert map_to_3class('Object Transfer') == 'Manipulation'
        assert map_to_3class('Task Operation') == 'Manipulation'
        assert map_to_3class('Locomotion') == 'Locomotion'
        assert map_to_3class('Stationary') == 'Passive'
        assert map_to_3class('Search') == 'Passive'

    def test_action_orders(self):
        assert len(ACTION_ORDER) == 5
        assert len(ACTION_ORDER_4CLASS) == 4
        assert len(ACTION_ORDER_3CLASS) == 3

    def test_load_gold_returns_dataframe(self):
        require_dataset_file(PROJECT_ROOT / 'data' / 'gold' / 'har_gold_unified.csv')
        df = load_gold()
        assert isinstance(df, pd.DataFrame)
        assert 'action' in df.columns
        assert 'video_uid' in df.columns
        assert len(df) > 25000

    def test_extract_verb_filters_non_verbs(self):
        # Subject nouns / articles should not be returned as verbs
        assert extract_verb('#C C man walks') == 'walks'
        assert extract_verb('#C C the person walks') == 'walks'
        assert extract_verb('#C C a person sits') == 'sits'

    def test_extract_verb_handles_c_prefix(self):
        # After removing '#C', 'C' is the camera wearer prefix
        assert extract_verb('#C C picks up the cup') == 'picks'
        assert extract_verb('#C C adjusts the camera') == 'adjusts'

    def test_load_tier_assignments_returns_dataframe(self):
        require_dataset_file(PROJECT_ROOT / 'data' / 'processed' / 'tier_assignments.csv')
        df = load_tier_assignments()
        assert isinstance(df, pd.DataFrame)
        assert 'tier' in df.columns
        assert set(df['tier'].unique()).issubset({1, 2, 3, 4})
