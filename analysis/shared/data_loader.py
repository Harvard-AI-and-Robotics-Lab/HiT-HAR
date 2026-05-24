"""Shared data loading utilities for HAR analysis scripts.

All paths are relative to PROJECT_ROOT (HiT-HAR/).
"""
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # analysis/shared/ → HiT-HAR/
LAB_ROOT = PROJECT_ROOT.parent / 'HAR_Lab_Initiative_AI'

# Label constants
ACTION_ORDER = ['Object Transfer', 'Task Operation', 'Stationary', 'Locomotion', 'Search']
ACTION_ORDER_4CLASS = ['Manipulation', 'Stationary', 'Locomotion', 'Search']
ACTION_ORDER_3CLASS = ['Manipulation', 'Passive', 'Locomotion']
SCENARIO_ORDER = [
    'Cooking', 'Cleaning', 'Mechanical Repair', 'Playing Instrument',
    'Carpentry', 'Walking Outdoors', 'Desk Work', 'Gardening',
]

MAP_5_TO_4 = {
    'Object Transfer': 'Manipulation',
    'Task Operation': 'Manipulation',
    'Stationary': 'Stationary',
    'Locomotion': 'Locomotion',
    'Search': 'Search',
}

MAP_5_TO_3 = {
    'Object Transfer': 'Manipulation',
    'Task Operation': 'Manipulation',
    'Stationary': 'Passive',
    'Locomotion': 'Locomotion',
    'Search': 'Passive',
}

AMBIGUOUS_VERBS = {
    'moves', 'turns', 'takes', 'looks', 'holds', 'puts', 'picks',
    'adjusts', 'opens', 'closes', 'pulls', 'pushes', 'touches',
    'places', 'lifts', 'drops', 'carries', 'grabs', 'reaches',
    'checks', 'walks', 'stands', 'sits',
}


def normalize_narration(text):
    """Lowercase, remove #hashtags, strip trailing punctuation."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'#\S+', '', text).strip()
    text = re.sub(r'[.,;:!?]+$', '', text).strip()
    return text


# Tokens that are never action verbs (subjects, articles, pronouns)
_NON_VERB_TOKENS = {
    # Articles and determiners
    'the', 'a', 'an', 'some', 'this', 'that', 'these', 'those',
    # Pronouns and subject references
    'c', 'he', 'she', 'it', 'they', 'his', 'her', 'their',
    # Person nouns (not actions)
    'man', 'woman', 'person', 'child', 'lady', 'guy', 'boy', 'girl',
    'someone', 'people', 'female', 'male', 'cashier', 'worker',
    'customer', 'instructor', 'player', 'chef', 'driver',
    # Prepositions that sometimes appear first
    'with', 'on', 'in', 'at', 'to', 'from', 'of', 'by', 'for',
}


def extract_verb(narration_text):
    """Extract first action verb from narration like '#C C picks up the cup'.

    Skips hashtags, camera-wearer prefix 'C', articles, pronouns,
    and common non-verb tokens to find the actual action verb.
    """
    if not isinstance(narration_text, str):
        return ''
    text = re.sub(r'#\w+', '', narration_text).strip()
    # Remove single-letter camera wearer prefix (e.g., 'C ')
    text = re.sub(r'^[A-Z]\s+', '', text)
    words = text.strip().split()
    # Find first word that is plausibly a verb
    for word in words:
        w = word.lower().rstrip('.,;:!?')
        if w and w not in _NON_VERB_TOKENS and len(w) > 1:
            return w
    return words[0].lower() if words else ''


def map_to_4class(action):
    """Map 5-class action to 4-class (OT+TaskOp → Manipulation)."""
    return MAP_5_TO_4.get(action, action)


def map_to_3class(action):
    """Map 5-class action to 3-class (OT+TaskOp→Manipulation, Stat+Search→Passive)."""
    return MAP_5_TO_3.get(action, action)


def load_gold():
    """Load har_gold_unified.csv with Essential Operation → Task Operation."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'har_gold_unified.csv'
    df = pd.read_csv(path)
    df['action'] = df['action'].replace({'Essential Operation': 'Task Operation'})
    df['narr_norm'] = df['narration_text'].apply(normalize_narration)
    df['verb'] = df['narration_text'].apply(extract_verb)
    return df


def load_llm():
    """Load LLM-generated labels (355K rows)."""
    path = LAB_ROOT / 'data' / 'labels' / 'action_labels_llm_clean_refined.csv'
    df = pd.read_csv(path)
    df['action'] = df['action'].replace({'Essential Operation': 'Task Operation'})
    df['narr_norm'] = df['narration_text'].apply(normalize_narration)
    df['verb'] = df['narration_text'].apply(extract_verb)
    return df


def load_r1_raw():
    """Load R001 raw annotation export."""
    path = LAB_ROOT / 'data' / 'annotation_rounds' / 'r001_gold_exports' / 'r001_raw' / 'export.csv'
    df = pd.read_csv(path)
    df['action'] = df['action'].replace({'Essential Operation': 'Task Operation'})
    df['round'] = 'r001'
    df['verdict'] = df['status_main'].fillna('')
    if 'corrected_action' in df.columns:
        df['corrected_action'] = df['corrected_action'].replace(
            {'Essential Operation': 'Task Operation'}).fillna('')
    else:
        df['corrected_action'] = ''
    df['secondary_action'] = ''
    return df


def load_r2_raw():
    """Load and concatenate all R002 batch CSVs."""
    raw_dir = LAB_ROOT / 'data' / 'annotation_rounds' / 'r002_gold_exports' / 'r002_raw'
    dfs = []
    for f in sorted(raw_dir.glob('HAR_B*_R2.csv')):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    df['action'] = df['action'].replace({'Essential Operation': 'Task Operation'})
    df['round'] = 'r002'
    if 'secondary_action' not in df.columns:
        df['secondary_action'] = ''
    else:
        df['secondary_action'] = df['secondary_action'].fillna('')
    if 'corrected_action' not in df.columns:
        df['corrected_action'] = ''
    else:
        df['corrected_action'] = df['corrected_action'].fillna('')
    if 'verdict' not in df.columns:
        df['verdict'] = ''
    else:
        df['verdict'] = df['verdict'].fillna('')
    return df


def load_tier_assignments():
    """Load precomputed tier assignments."""
    path = PROJECT_ROOT / 'data' / 'processed' / 'tier_assignments.csv'
    return pd.read_csv(path)


def load_train():
    """Load training split."""
    path = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
    return pd.read_csv(path)
