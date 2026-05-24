"""Tests for Layer 1 analysis functions using synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLLMAccuracy:
    def test_compute_agreement_rate(self):
        from analysis.layer1_labels.l1_llm_accuracy import compute_agreement

        gold = pd.DataFrame({
            'narr_norm': ['picks up cup', 'opens door', 'walks forward', 'looks around'],
            'action': ['Object Transfer', 'Task Operation', 'Locomotion', 'Search'],
        })
        llm = pd.DataFrame({
            'narr_norm': ['picks up cup', 'opens door', 'walks forward', 'looks around'],
            'action': ['Object Transfer', 'Task Operation', 'Locomotion', 'Stationary'],
        })
        result = compute_agreement(gold, llm, on='narr_norm')
        assert result['overall_agreement'] == 0.75  # 3/4 match (OT, TO, Loco agree; Search disagree)
        assert result['per_class']['Object Transfer'] == 1.0  # 1/1 correct OT
        assert result['per_class']['Locomotion'] == 1.0

    def test_build_correction_matrix(self):
        from analysis.layer1_labels.l1_llm_accuracy import build_correction_matrix

        disagreements = pd.DataFrame({
            'gold_action': ['Object Transfer', 'Search', 'Search'],
            'llm_action': ['Task Operation', 'Stationary', 'Stationary'],
        })
        mat = build_correction_matrix(disagreements)
        assert mat.loc['Task Operation', 'Object Transfer'] == 1
        assert mat.loc['Stationary', 'Search'] == 2


class TestConflictAnalysis:
    def test_find_timestamp_conflicts(self):
        from analysis.layer1_labels.l1_conflict_analysis import find_timestamp_conflicts

        df = pd.DataFrame({
            'video_uid': ['v1', 'v1', 'v1', 'v2'],
            'timestamp_sec': [1.0, 1.0, 2.0, 1.0],
            'action': ['Object Transfer', 'Task Operation', 'Locomotion', 'Search'],
        })
        conflicts = find_timestamp_conflicts(df)
        assert len(conflicts) == 2  # v1@1.0 has 2 rows with different actions
        assert conflicts.iloc[0]['video_uid'] == 'v1'

    def test_classify_conflict_source(self):
        from analysis.layer1_labels.l1_conflict_analysis import classify_conflict_source

        pair = ('Object Transfer', 'Task Operation')
        source = classify_conflict_source(pair)
        assert source == 'taxonomy_ambiguity'

        pair2 = ('Locomotion', 'Search')
        source2 = classify_conflict_source(pair2)
        assert source2 == 'llm_error'


class TestAnnotationQuality:
    def test_secondary_action_stats(self):
        from analysis.layer1_labels.l1_annotation_quality import compute_secondary_stats

        df = pd.DataFrame({
            'action': ['Stationary', 'Stationary', 'Object Transfer', 'Locomotion'],
            'secondary_action': ['Task Operation', '', 'Task Operation', ''],
            'verdict': ['Gold', 'Gold', 'Gold', 'Gold'],
        })
        stats = compute_secondary_stats(df)
        assert stats['total_with_secondary'] == 2
        assert stats['secondary_rate'] == 0.5
        assert ('Stationary', 'Task Operation') in stats['top_pairs']


class TestTaxonomyBoundary:
    def test_compute_verb_class_matrix(self):
        from analysis.layer1_labels.l1_taxonomy_boundary import compute_verb_class_matrix

        df = pd.DataFrame({
            'verb': ['picks', 'picks', 'walks', 'walks', 'looks'],
            'action': ['Object Transfer', 'Task Operation', 'Locomotion', 'Locomotion', 'Search'],
        })
        mat = compute_verb_class_matrix(df)
        assert mat.loc['picks', 'Object Transfer'] == 1
        assert mat.loc['picks', 'Task Operation'] == 1
        assert mat.loc['walks', 'Locomotion'] == 2

    def test_compute_ambiguity_rate(self):
        from analysis.layer1_labels.l1_taxonomy_boundary import compute_ambiguity_rate

        df = pd.DataFrame({
            'verb': ['picks', 'picks', 'walks', 'walks', 'looks'],
            'action': ['Object Transfer', 'Task Operation', 'Locomotion', 'Locomotion', 'Search'],
        })
        # 'picks' spans 2 classes, 'walks' spans 1, 'looks' spans 1
        rate = compute_ambiguity_rate(df, min_classes=2)
        # 2 out of 5 samples have ambiguous verb ('picks')
        assert rate == pytest.approx(2/5)
