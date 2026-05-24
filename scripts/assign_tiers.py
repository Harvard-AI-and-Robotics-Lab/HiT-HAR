#!/usr/bin/env python3
"""Assign quality tiers to gold annotations using R001/R002 raw exports.

Tier system (see paper/2026-03-22-annotation-quality-analysis.md §七):
  Tier 1 (HIGH):     Gold + no secondary + non-ambiguous verb → confidence=1.0
  Tier 2 (MODERATE): Gold + (secondary_action OR ambiguous verb) → confidence=0.8
  Tier 3 (CORRECTED):Bad + corrected_action → confidence=0.5-0.7
  Tier 4 (EXCLUDE):  Skip/Delete → confidence=0.0

Output: data/processed/tier_assignments.csv
"""
import re
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Verbs that appear in 3+ action classes in R2 gold annotations
# Source: paper/2026-03-22-annotation-quality-analysis.md §五
AMBIGUOUS_VERBS = {
    'moves', 'turns', 'takes', 'looks', 'holds', 'puts', 'picks',
    'adjusts', 'opens', 'closes', 'pulls', 'pushes', 'touches',
    'places', 'lifts', 'drops', 'carries', 'grabs', 'reaches',
    'checks', 'walks', 'stands', 'sits',
}

# High-confusion correction pairs (see confusion matrix in analysis doc)
HIGH_CONFUSION_PAIRS = {
    frozenset({'Stationary', 'Task Operation'}),
    frozenset({'Object Transfer', 'Stationary'}),
    frozenset({'Object Transfer', 'Task Operation'}),
    frozenset({'Search', 'Stationary'}),
}


def extract_verb(narration_text):
    """Extract the first verb from a narration like '#C C picks up the cup'."""
    if not isinstance(narration_text, str):
        return ''
    # Remove hashtags and person identifiers
    text = re.sub(r'#\w+', '', narration_text).strip()
    # Remove "C " prefix (camera wearer)
    text = re.sub(r'^[A-Z]\s+', '', text)
    # First word is typically the verb
    words = text.strip().split()
    return words[0].lower() if words else ''


def load_r001(path):
    """Load and normalize R001 raw export."""
    df = pd.read_csv(path)
    df['round'] = 'r001'
    # R001 uses status_main for verdict
    df['verdict'] = df['status_main'].fillna('')
    df['secondary_action'] = ''  # R001 has no secondary_action
    # Normalize label rename
    df['action'] = df['action'].replace({'Essential Operation': 'Task Operation'})
    if 'corrected_action' in df.columns:
        df['corrected_action'] = df['corrected_action'].replace(
            {'Essential Operation': 'Task Operation'}).fillna('')
    else:
        df['corrected_action'] = ''
    return df[['video_uid', 'timestamp_sec', 'narration_text', 'action',
               'verdict', 'corrected_action', 'secondary_action', 'round']]


def load_r002(directory):
    """Load and concatenate all R002 batch CSVs."""
    dfs = []
    for f in sorted(Path(directory).glob('HAR_B*_R2.csv')):
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df['round'] = 'r002'
    df['secondary_action'] = df.get('secondary_action', pd.Series(dtype=str)).fillna('')
    df['corrected_action'] = df.get('corrected_action', pd.Series(dtype=str)).fillna('')
    # R002 uses 'verdict' column directly
    if 'verdict' not in df.columns:
        df['verdict'] = df.get('status_main', '').fillna('')
    df['verdict'] = df['verdict'].fillna('')
    df['action'] = df['action'].replace({'Essential Operation': 'Task Operation'})
    df['corrected_action'] = df['corrected_action'].replace(
        {'Essential Operation': 'Task Operation'})
    return df[['video_uid', 'timestamp_sec', 'narration_text', 'action',
               'verdict', 'corrected_action', 'secondary_action', 'round']]


def assign_tier(row):
    """Assign tier and confidence to a single annotation row."""
    verdict = str(row['verdict']).strip()
    has_secondary = bool(row['secondary_action'] and str(row['secondary_action']).strip())
    verb = extract_verb(row['narration_text'])
    is_ambiguous_verb = verb in AMBIGUOUS_VERBS

    # Tier 4: Skip/Delete/unreviewed
    if verdict in ('Skip', 'Delete Row', 'Delete', ''):
        return 4, 0.0

    # Tier 3: Bad (corrected)
    if verdict == 'Bad':
        corrected = str(row['corrected_action']).strip()
        original = str(row['action']).strip()
        if corrected and original:
            pair = frozenset({original, corrected})
            if pair in HIGH_CONFUSION_PAIRS:
                return 3, 0.5  # High-confusion correction
            return 3, 0.7  # Other correction
        return 3, 0.6  # Bad but no clear correction

    # Tier 1 or 2: Gold
    if verdict == 'Gold':
        if has_secondary or is_ambiguous_verb:
            return 2, 0.8
        return 1, 1.0

    # Fallback (silver, etc.)
    return 2, 0.8


def main():
    parser = argparse.ArgumentParser(description='Assign quality tiers to gold annotations')
    parser.add_argument('--r001', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'annotations' / 'r001_export.csv'))
    parser.add_argument('--r002-dir', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'annotations' / 'r002_exports'))
    parser.add_argument('--output', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'tier_assignments.csv'))
    args = parser.parse_args()

    print("Loading R001...")
    r001 = load_r001(args.r001)
    print(f"  R001: {len(r001)} rows")

    print("Loading R002...")
    r002 = load_r002(args.r002_dir)
    print(f"  R002: {len(r002)} rows")

    combined = pd.concat([r001, r002], ignore_index=True)
    print(f"Combined: {len(combined)} rows")

    # Assign tiers
    print("Assigning tiers...")
    tiers = combined.apply(assign_tier, axis=1, result_type='expand')
    combined['tier'] = tiers[0].astype(int)
    combined['confidence'] = tiers[1].astype(float)
    combined['has_secondary'] = combined['secondary_action'].apply(
        lambda x: bool(x and str(x).strip()))

    # Summary
    print("\n=== Tier Distribution ===")
    for tier in [1, 2, 3, 4]:
        subset = combined[combined['tier'] == tier]
        pct = len(subset) / len(combined) * 100
        conf = subset['confidence'].mean()
        print(f"  Tier {tier}: {len(subset):>6} ({pct:.1f}%) avg_confidence={conf:.2f}")

    # Per-round breakdown
    for rnd in ['r001', 'r002']:
        subset = combined[combined['round'] == rnd]
        print(f"\n  {rnd}: {len(subset)} rows")
        for tier in [1, 2, 3, 4]:
            t = subset[subset['tier'] == tier]
            print(f"    Tier {tier}: {len(t)}")

    # Save
    output_cols = ['video_uid', 'timestamp_sec', 'narration_text', 'action',
                   'verdict', 'corrected_action', 'secondary_action',
                   'round', 'tier', 'confidence', 'has_secondary']
    combined[output_cols].to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
