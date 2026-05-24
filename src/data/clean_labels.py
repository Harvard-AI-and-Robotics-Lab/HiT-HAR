"""
Data cleaning with Gold label propagation and train/val/test split for HiT-HAR.

Pipeline:
  1. Clean full CSV (355K): remove #O, #Summary, empty, Error/Invalid;
     rename Essential Operation -> Task Operation
  2. Clean gold CSV (27K): filter status, remove #O, resolve corrected_action,
     remove multi-label narrations
  3. Propagate gold labels to full CSV via normalized narration matching
  4. Merge gold + propagated rows ONLY (LLM rows dropped)
  5. Split by video_uid (70/15/15), stratified by scenario, seed=42

Inputs:
  - har_gold_unified.csv            (27,355 rows, Gold validated 5-class)
  - action_labels_llm_clean_refined.csv  (355,580 rows, used only for propagation targets)

Outputs:
  - data/processed/train.csv        (~107K rows, gold+propagated)
  - data/processed/val.csv          (~23K rows)
  - data/processed/test.csv         (~23K rows)
  - data/processed/class_weights.json
"""

import json
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ACTION_MAP = {
    'Object Transfer': 0,
    'Task Operation': 1,
    'Stationary': 2,
    'Locomotion': 3,
    'Search': 4,
}

LABEL_RENAME = {
    'Essential Operation': 'Task Operation',
}

SEED = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_narration(text: str) -> str:
    """Normalize narration text for consistency matching."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'#\w+', '', text)           # remove hashtags
    text = re.sub(r'[.,;:!?]+$', '', text)     # remove trailing punctuation
    return text.strip()


def has_other_person_tag(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r'#O\b', text, re.IGNORECASE))


def has_summary_tag(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r'#summary', text, re.IGNORECASE))


def has_unsure_tag(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r'#unsure', text, re.IGNORECASE))


def compute_class_weights(counts: dict, method: str = 'effective_number',
                          beta: float = 0.9999) -> dict:
    """Compute class weights (Cui et al. CVPR 2019). Normalized to sum=num_classes."""
    classes = sorted(counts.keys(), key=lambda c: ACTION_MAP.get(c, 99))
    n = np.array([counts[c] for c in classes], dtype=np.float64)

    if method == 'effective_number':
        effective_n = (1.0 - np.power(beta, n)) / (1.0 - beta)
        raw_w = 1.0 / effective_n
    elif method == 'inverse_freq':
        raw_w = 1.0 / n
    elif method == 'sqrt_inverse':
        raw_w = 1.0 / np.sqrt(n)
    else:
        raise ValueError(f"Unknown method: {method}")

    raw_w = raw_w / raw_w.sum() * len(classes)
    return {c: float(w) for c, w in zip(classes, raw_w)}


def print_split_stats(df: pd.DataFrame, name: str):
    """Print action, scenario, and source distribution for a split."""
    print(f"\n  [{name}] {len(df)} samples, {df['video_uid'].nunique()} videos")

    action_dist = df['action'].value_counts()
    for action in ACTION_MAP:
        cnt = action_dist.get(action, 0)
        pct = cnt / len(df) * 100
        print(f"    {action}: {cnt} ({pct:.1f}%)")

    if 'source' in df.columns:
        src_dist = df['source'].value_counts()
        print(f"    Sources: {', '.join(f'{s}={c}' for s, c in src_dist.items())}")

    scenario_dist = df['scenario'].value_counts()
    print(f"    Scenarios: {', '.join(f'{s}={c}' for s, c in scenario_dist.items())}")


# ---------------------------------------------------------------------------
# Step 1: Clean full CSV
# ---------------------------------------------------------------------------

def clean_full_csv(full_df: pd.DataFrame) -> pd.DataFrame:
    """Clean 355K full CSV: remove #O, #Summary, empty, Error/Invalid;
    rename Essential Operation -> Task Operation; add narr_norm."""
    df = full_df.copy()
    print(f"  Raw full: {len(df)} rows")

    # Rename legacy labels
    n_renamed = df['action'].isin(LABEL_RENAME).sum()
    if n_renamed > 0:
        df['action'] = df['action'].replace(LABEL_RENAME)
        print(f"  Renamed {n_renamed} Essential Operation -> Task Operation")

    # Normalize narrations
    df['narr_norm'] = df['narration_text'].apply(normalize_narration)

    # Remove #O
    mask_O = df['narration_text'].apply(has_other_person_tag)
    df = df[~mask_O].copy()
    print(f"  After removing #O: {len(df)}")

    # Remove #Summary, empty, invalid actions
    mask_summary = df['narration_text'].apply(has_summary_tag)
    mask_empty = df['narr_norm'].str.len() < 3
    mask_invalid = df['action'].isin(['Error', 'Invalid', 'Unknown', ''])
    df = df[~(mask_summary | mask_empty | mask_invalid)].copy()
    print(f"  After removing #Summary/empty/invalid: {len(df)}")

    # Mark #unsure
    df['is_unsure'] = df['narration_text'].apply(has_unsure_tag)

    return df


# ---------------------------------------------------------------------------
# Step 2: Clean gold CSV
# ---------------------------------------------------------------------------

def clean_gold_csv(gold_df: pd.DataFrame) -> pd.DataFrame:
    """Clean gold data: filter status, resolve corrections, remove #O, remove multi-label."""
    gold = gold_df.copy()

    # Filter to Gold status
    status_col = 'status_main' if 'status_main' in gold.columns else \
                 'status' if 'status' in gold.columns else None
    if status_col is not None:
        n_before = len(gold)
        gold = gold[gold[status_col] == 'Gold'].copy()
        print(f"  Filtered to Gold ('{status_col}'): {n_before} -> {len(gold)}")

    # Remove #O (other person's action — wrong labels for camera wearer's IMU)
    mask_O = gold['narration_text'].apply(has_other_person_tag)
    n_O = mask_O.sum()
    if n_O > 0:
        gold = gold[~mask_O].copy()
        print(f"  Removed {n_O} #O rows (other person's actions)")

    # Use corrected_action if available
    gold['final_action'] = gold['action'].copy()
    if 'corrected_action' in gold.columns:
        has_corr = gold['corrected_action'].notna() & (gold['corrected_action'] != '')
        gold.loc[has_corr, 'final_action'] = gold.loc[has_corr, 'corrected_action']

    # Filter to valid 5-class
    gold = gold[gold['final_action'].isin(ACTION_MAP)].copy()
    gold['action'] = gold['final_action']

    # Mark #unsure (downweight confidence later)
    gold['is_unsure'] = gold['narration_text'].apply(has_unsure_tag)

    # Normalize
    gold['narr_norm'] = gold['narration_text'].apply(normalize_narration)

    # Remove multi-label narrations (same narr_norm -> different actions)
    nunique = gold.groupby('narr_norm')['action'].nunique()
    multi_narrs = set(nunique[nunique > 1].index)
    if multi_narrs:
        mask = gold['narr_norm'].isin(multi_narrs)
        n_multi = mask.sum()
        gold = gold[~mask].copy()
        print(f"  Removed {n_multi} multi-label rows ({len(multi_narrs)} narrations)")

    print(f"  Clean gold: {len(gold)} rows, {gold['video_uid'].nunique()} videos")
    return gold


# ---------------------------------------------------------------------------
# Step 3: Build gold narration -> action map
# ---------------------------------------------------------------------------

def build_gold_label_map(gold: pd.DataFrame) -> dict:
    """Build narr_norm -> validated action map from gold data.

    Uses majority vote; excludes narrations with agreement < 80%.
    """
    narr_action = gold.groupby('narr_norm')['action']
    label_map = {}

    for narr, group in narr_action:
        counts = group.value_counts()
        majority_action = counts.index[0]
        agreement = counts.iloc[0] / counts.sum()
        if agreement >= 0.8:
            label_map[narr] = majority_action

    return label_map


# ---------------------------------------------------------------------------
# Step 4: Propagate gold labels to full CSV
# ---------------------------------------------------------------------------

def propagate_labels(full_clean: pd.DataFrame, gold_clean: pd.DataFrame,
                     gold_label_map: dict) -> pd.DataFrame:
    """Merge gold + propagated rows only (no LLM).

    Returns combined DataFrame with 'source' and 'confidence' columns.
    """
    # --- Identify gold rows in full CSV by (video_uid, timestamp_sec) ---
    gold_keys = set(zip(gold_clean['video_uid'], gold_clean['timestamp_sec']))
    full_is_gold = full_clean.apply(
        lambda r: (r['video_uid'], r['timestamp_sec']) in gold_keys, axis=1
    )

    # Gold rows: take directly from gold_clean (already validated)
    gold_rows = gold_clean[['video_uid', 'timestamp_sec', 'narration_text',
                            'scenario', 'action', 'narr_norm']].copy()
    gold_rows['source'] = 'gold'
    # FIX C3: Use pre-existing tier confidence (set in Step 2b), fall back to 1.0
    if 'confidence' in gold_clean.columns:
        gold_rows['confidence'] = gold_clean['confidence'].values
    else:
        gold_rows['confidence'] = np.where(gold_clean['is_unsure'].values, 0.8, 1.0)
    if 'tier' in gold_clean.columns:
        gold_rows['tier'] = gold_clean['tier'].values
    else:
        gold_rows['tier'] = 1
    print(f"  Gold rows: {len(gold_rows)}")

    # Non-gold rows from full CSV
    non_gold = full_clean[~full_is_gold].copy()
    print(f"  Non-gold full rows: {len(non_gold)}")

    # Propagated: non-gold rows whose narr_norm matches gold label map
    has_match = non_gold['narr_norm'].isin(gold_label_map)
    propagated = non_gold[has_match].copy()
    propagated['action'] = propagated['narr_norm'].map(gold_label_map)
    propagated['source'] = 'propagated'
    # Inherit tier from gold via narr_norm mapping (use worst/highest tier if multiple golds)
    if 'tier' in gold_clean.columns:
        gold_tier_map = gold_clean.groupby('narr_norm')['tier'].max().to_dict()
        propagated['tier'] = propagated['narr_norm'].map(gold_tier_map).fillna(2).astype(int)
        # Propagated labels use lower confidence than gold (text-match, not human-validated)
        # Gold confidence: {1: 1.0, 2: 0.8, 3: 0.6}  (set in Step 2b)
        # Propagated confidence: {1: 0.7, 2: 0.5, 3: 0.3}
        tier_to_conf_propagated = {1: 0.7, 2: 0.5, 3: 0.3}
        propagated['confidence'] = propagated['tier'].map(tier_to_conf_propagated).fillna(0.5)
        propagated.loc[propagated['is_unsure'], 'confidence'] = 0.3
    else:
        propagated['confidence'] = np.where(propagated['is_unsure'], 0.5, 1.0)
        propagated['tier'] = 2
    print(f"  Propagated rows (gold-matched): {len(propagated)}")

    # LLM rows: DROPPED (known noisy, 26% error on Stationary)
    n_dropped = (~has_match).sum()
    print(f"  Dropped LLM-only rows: {n_dropped}")

    # Merge gold + propagated only
    out_cols = ['video_uid', 'timestamp_sec', 'narration_text', 'scenario',
                'action', 'source', 'confidence', 'tier']
    combined = pd.concat([
        gold_rows[out_cols],
        propagated[out_cols],
    ], ignore_index=True)

    # Add action_id
    combined['action_id'] = combined['action'].map(ACTION_MAP)

    return combined


# ---------------------------------------------------------------------------
# Split by video_uid
# ---------------------------------------------------------------------------

def split_by_video(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15,
                   test_ratio=0.15, seed=SEED):
    """Split data by video_uid groups. Returns (train, val, test)."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    video_info = df.groupby('video_uid').agg(
        scenario=('scenario', lambda x: x.mode()[0]),
    ).reset_index()
    videos = video_info['video_uid'].values
    scenarios = video_info['scenario'].values

    # train vs (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio,
                             random_state=seed)
    train_idx, rest_idx = next(gss1.split(videos, scenarios, groups=videos))
    train_vids = set(videos[train_idx])
    rest_vids = videos[rest_idx]
    rest_scenarios = scenarios[rest_idx]

    # val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test,
                             random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_vids, rest_scenarios,
                                         groups=rest_vids))
    val_vids = set(rest_vids[val_idx])
    test_vids = set(rest_vids[test_idx])

    return (df[df['video_uid'].isin(train_vids)].copy(),
            df[df['video_uid'].isin(val_vids)].copy(),
            df[df['video_uid'].isin(test_vids)].copy())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    default_gold = project_root / 'data' / 'gold' / 'har_gold_unified.csv'
    default_full = project_root / 'data' / 'labels' / 'action_labels_llm_clean_refined.csv'
    default_out = project_root / 'data' / 'processed'

    parser = argparse.ArgumentParser(
        description='Clean labels with Gold propagation and split for HiT-HAR')
    parser.add_argument('--gold-labels', type=str, default=str(default_gold))
    parser.add_argument('--full-labels', type=str, default=str(default_full))
    parser.add_argument('--output-dir', type=str, default=str(default_out))
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load and clean full CSV ----
    print("=== Step 1: Clean full CSV ===")
    full_df = pd.read_csv(args.full_labels)
    full_clean = clean_full_csv(full_df)

    # ---- Step 2: Load and clean gold CSV ----
    print("\n=== Step 2: Clean gold CSV ===")
    gold_df = pd.read_csv(args.gold_labels)
    gold_clean = clean_gold_csv(gold_df)

    # ---- Step 2b: Join quality tiers ----
    print("\n=== Step 2b: Join quality tiers ===")
    tier_path = project_root / 'data' / 'processed' / 'tier_assignments.csv'
    if tier_path.exists():
        tier_df = pd.read_csv(tier_path)

        # FIX C2: Round timestamps to avoid float precision mismatch between CSVs
        PRECISION = 3  # 1ms precision
        gold_clean['ts_rounded'] = gold_clean['timestamp_sec'].round(PRECISION)
        tier_df['ts_rounded'] = tier_df['timestamp_sec'].round(PRECISION)

        # Split tier_df: Tier 1-2 (Gold) join to gold_clean; Tier 3 (Bad corrected) added separately
        tier_gold = tier_df[tier_df['tier'].isin([1, 2])].copy()
        tier_bad = tier_df[tier_df['tier'] == 3].copy()
        # Exclude Tier 4 entirely

        # Join Gold tiers by (video_uid, ts_rounded)
        gold_clean = gold_clean.merge(
            tier_gold[['video_uid', 'ts_rounded', 'tier', 'confidence']],
            on=['video_uid', 'ts_rounded'],
            how='left',
        )
        gold_clean['tier'] = gold_clean['tier'].fillna(2).astype(int)
        gold_clean['confidence'] = gold_clean['confidence'].fillna(0.8)
        gold_clean.drop(columns=['ts_rounded'], inplace=True)

        # FIX C4: Add Tier 3 (Bad corrected) rows as additional gold-quality data
        # These were excluded by clean_gold_csv()'s Gold filter, but their corrected labels are usable
        if len(tier_bad) > 0:
            tier_bad_usable = tier_bad[
                tier_bad['corrected_action'].notna() &
                (tier_bad['corrected_action'] != '') &
                tier_bad['corrected_action'].isin(ACTION_MAP)
            ].copy()
            tier_bad_usable['action'] = tier_bad_usable['corrected_action']
            tier_bad_usable['narr_norm'] = tier_bad_usable['narration_text'].apply(normalize_narration)
            # Add required columns
            if 'scenario' not in tier_bad_usable.columns:
                # Join scenario from gold data by video_uid
                scenario_map = gold_clean.groupby('video_uid')['scenario'].first()
                tier_bad_usable['scenario'] = tier_bad_usable['video_uid'].map(scenario_map)
            tier_bad_usable['is_unsure'] = tier_bad_usable['narration_text'].apply(has_unsure_tag)
            # Apply same filters as clean_gold_csv: remove #O and multi-label narrations
            mask_O = tier_bad_usable['narration_text'].apply(has_other_person_tag)
            tier_bad_usable = tier_bad_usable[~mask_O].copy()
            nunique = tier_bad_usable.groupby('narr_norm')['action'].nunique()
            multi_narrs = set(nunique[nunique > 1].index)
            if multi_narrs:
                tier_bad_usable = tier_bad_usable[~tier_bad_usable['narr_norm'].isin(multi_narrs)].copy()
            tier_bad_usable = tier_bad_usable.dropna(subset=['scenario'])
            print(f"  Added {len(tier_bad_usable)} Tier 3 (Bad corrected) rows")
            gold_clean = pd.concat([gold_clean, tier_bad_usable], ignore_index=True)

        n_t1 = (gold_clean['tier'] == 1).sum()
        n_t2 = (gold_clean['tier'] == 2).sum()
        n_t3 = (gold_clean['tier'] == 3).sum()
        print(f"  Tier 1 (high): {n_t1}, Tier 2 (moderate): {n_t2}, Tier 3 (corrected): {n_t3}")
    else:
        print("  No tier_assignments.csv found, using default confidence=1.0")
        gold_clean['tier'] = 1
        gold_clean['confidence'] = 1.0

    # ---- Step 3: Build gold label map ----
    print("\n=== Step 3: Build gold label map ===")
    gold_label_map = build_gold_label_map(gold_clean)
    print(f"  Gold label map: {len(gold_label_map)} unique narrations")

    # ---- Step 4: Propagate labels ----
    print("\n=== Step 4: Label propagation ===")
    combined = propagate_labels(full_clean, gold_clean, gold_label_map)
    print(f"\n  Combined total: {len(combined)} rows, "
          f"{combined['video_uid'].nunique()} videos")

    # Source breakdown
    src_counts = combined['source'].value_counts()
    for src, cnt in src_counts.items():
        pct = cnt / len(combined) * 100
        print(f"    {src}: {cnt} ({pct:.1f}%)")

    # Action distribution
    print("\n  Action distribution (all):")
    for action in ACTION_MAP:
        cnt = (combined['action'] == action).sum()
        pct = cnt / len(combined) * 100
        print(f"    {action}: {cnt} ({pct:.1f}%)")

    # ---- Step 5: Split by video_uid ----
    print("\n=== Step 5: Train/Val/Test Split (70/15/15 by video_uid) ===")
    train_df, val_df, test_df = split_by_video(combined, seed=args.seed)

    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print_split_stats(split_df, name)

    # Sanity: no video overlap
    train_vids = set(train_df['video_uid'])
    val_vids = set(val_df['video_uid'])
    test_vids = set(test_df['video_uid'])
    assert len(train_vids & val_vids) == 0
    assert len(train_vids & test_vids) == 0
    assert len(val_vids & test_vids) == 0
    print("\n  No video overlap between splits.")

    # ---- Save ----
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = output_dir / f'{name}.csv'
        split_df.to_csv(path, index=False)

    print(f"\nSaved: train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)})")

    # ---- Class weights (from training set) ----
    train_counts = train_df['action'].value_counts().to_dict()
    weights = {}
    for method in ['effective_number', 'inverse_freq', 'sqrt_inverse']:
        weights[method] = compute_class_weights(train_counts, method=method)

    weights_path = output_dir / 'class_weights.json'
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"\n=== Class Weights (from training set, saved to {weights_path}) ===")
    print(f"  Effective number: {weights['effective_number']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
