#!/usr/bin/env python3
"""
Gold Data Quality Analysis for HiT-HAR
======================================
Analyzes the final gold dataset (HAR_dataset.csv) and produces:
  - Markdown report:      data/analysis/gold_data_report.md
  - Figures:              data/analysis/figures/*.png
  - Multi-label issues:   data/analysis/multi_label_issues.csv

Must be run from the HiT-HAR directory:
    cd HiT-HAR && python scripts/analyze_gold_data.py
"""

import os
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # HiT-HAR/
DATA_PATH = PROJECT_DIR / "data" / "gold" / "har_gold_unified.csv"
OUT_DIR = PROJECT_DIR / "data" / "analysis"
FIG_DIR = OUT_DIR / "figures"

# ── Constants ─────────────────────────────────────────────────────────────────
ACTION_MAP = {"Essential Operation": "Task Operation"}
ACTION_ORDER = ["Object Transfer", "Task Operation", "Stationary", "Locomotion", "Search"]
SCENARIO_ORDER = [
    "Cooking", "Cleaning", "Mechanical Repair", "Playing Instrument",
    "Carpentry", "Walking Outdoors", "Desk Work", "Gardening",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_narration(text: str) -> str:
    """Lowercase, remove #hashtags, strip trailing punctuation."""
    text = text.lower().strip()
    text = re.sub(r"#\S+", "", text).strip()       # remove hashtags
    text = re.sub(r"[.,;:!?]+$", "", text).strip()  # trailing punctuation
    return text


def effective_number_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """Compute effective-number-of-samples weights (Cui et al., CVPR 2019)."""
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(weights)  # normalize so mean=1
    return weights


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # -- Setup output dirs --
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load data --
    data_path = DATA_PATH.resolve()
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    # -- Label unification --
    df["action"] = df["action"].replace(ACTION_MAP)

    # -- Normalize narrations --
    df["narration_norm"] = df["narration_text"].astype(str).apply(normalize_narration)

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ 1. Basic Statistics                                                  ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    total_rows = len(df)
    unique_videos = df["video_uid"].nunique()
    round_dist = df["round"].value_counts().sort_index()
    unique_narrations_raw = df["narration_text"].nunique()
    unique_narrations_norm = df["narration_norm"].nunique()
    unique_scenarios = df["scenario"].nunique()

    basic_stats = textwrap.dedent(f"""\
    ## 1. Basic Statistics

    | Metric | Value |
    |--------|-------|
    | Total samples | {total_rows:,} |
    | Unique videos | {unique_videos:,} |
    | Unique scenarios | {unique_scenarios} |
    | Unique narrations (raw) | {unique_narrations_raw:,} |
    | Unique narrations (normalized) | {unique_narrations_norm:,} |
    """)
    # Round distribution table
    basic_stats += "\n### Round Distribution\n\n| Round | Count | % |\n|-------|------:|---:|\n"
    for rnd, cnt in round_dist.items():
        basic_stats += f"| {rnd} | {cnt:,} | {cnt/total_rows*100:.1f}% |\n"

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ 2. Action Distribution                                              ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    action_counts = df["action"].value_counts().reindex(ACTION_ORDER).fillna(0).astype(int)
    action_pct = action_counts / total_rows * 100

    action_table = "\n## 2. Action Distribution\n\n| Action | Count | % |\n|--------|------:|---:|\n"
    for act in ACTION_ORDER:
        action_table += f"| {act} | {action_counts[act]:,} | {action_pct[act]:.1f}% |\n"

    # -- Figure: action bar chart --
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(ACTION_ORDER))
    bars = ax.bar(ACTION_ORDER, action_counts.values, color=colors, edgecolor="white")
    for bar, cnt, pct in zip(bars, action_counts.values, action_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_rows * 0.005,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Action Class Distribution (Gold Dataset)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Action")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "action_distribution.png", dpi=150)
    plt.close(fig)
    print("  -> action_distribution.png")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ 3. Scenario Distribution                                            ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    scenario_counts = df["scenario"].value_counts().reindex(SCENARIO_ORDER).fillna(0).astype(int)
    scenario_pct = scenario_counts / total_rows * 100

    scenario_table = "\n## 3. Scenario Distribution\n\n| Scenario | Count | % |\n|----------|------:|---:|\n"
    for sc in SCENARIO_ORDER:
        scenario_table += f"| {sc} | {scenario_counts[sc]:,} | {scenario_pct[sc]:.1f}% |\n"

    # -- Figure: scenario bar chart --
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_sc = sns.color_palette("Paired", len(SCENARIO_ORDER))
    bars = ax.barh(SCENARIO_ORDER[::-1], scenario_counts.values[::-1], color=colors_sc[::-1], edgecolor="white")
    for bar, cnt, pct in zip(bars, scenario_counts.values[::-1], scenario_pct.values[::-1]):
        ax.text(bar.get_width() + total_rows * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,} ({pct:.1f}%)", ha="left", va="center", fontsize=9)
    ax.set_title("Scenario Distribution (Gold Dataset)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "scenario_distribution.png", dpi=150)
    plt.close(fig)
    print("  -> scenario_distribution.png")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ 4. Action x Scenario Cross-Analysis Heatmap                         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    cross = pd.crosstab(df["scenario"], df["action"])
    cross = cross.reindex(index=SCENARIO_ORDER, columns=ACTION_ORDER).fillna(0).astype(int)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    cross_table = "\n## 4. Action x Scenario Cross-Analysis\n\n"
    cross_table += "Row-normalized percentages (% of each scenario):\n\n"
    cross_table += "| Scenario | " + " | ".join(ACTION_ORDER) + " | Total |\n"
    cross_table += "|----------|" + "|".join(["-----:" for _ in ACTION_ORDER]) + "|------:|\n"
    for sc in SCENARIO_ORDER:
        row = f"| {sc} "
        for act in ACTION_ORDER:
            row += f"| {cross_pct.loc[sc, act]:.1f}% "
        row += f"| {cross.loc[sc].sum():,} |\n"
        cross_table += row

    # -- Figure: heatmap --
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cross, annot=True, fmt=",d", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_title("Action x Scenario Heatmap (Counts)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Scenario")
    ax.set_xlabel("Action")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "action_scenario_heatmap.png", dpi=150)
    plt.close(fig)
    print("  -> action_scenario_heatmap.png")

    # -- Figure: row-normalized heatmap --
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cross_pct, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5,
                ax=ax, cbar_kws={"label": "% of Scenario"})
    ax.set_title("Action x Scenario Heatmap (Row-Normalized %)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Scenario")
    ax.set_xlabel("Action")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "action_scenario_heatmap_pct.png", dpi=150)
    plt.close(fig)
    print("  -> action_scenario_heatmap_pct.png")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ 5. R001 vs R002 Label Distribution Comparison                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    rounds = sorted(df["round"].dropna().unique())
    round_action_cross = pd.crosstab(df["round"], df["action"])
    round_action_cross = round_action_cross.reindex(columns=ACTION_ORDER).fillna(0).astype(int)
    round_action_pct = round_action_cross.div(round_action_cross.sum(axis=1), axis=0) * 100

    round_table = "\n## 5. R001 vs R002 Label Distribution\n\n"
    round_table += "### Counts\n\n"
    round_table += "| Round | " + " | ".join(ACTION_ORDER) + " | Total |\n"
    round_table += "|-------|" + "|".join(["-----:" for _ in ACTION_ORDER]) + "|------:|\n"
    for rnd in rounds:
        row = f"| {rnd} "
        for act in ACTION_ORDER:
            val = round_action_cross.loc[rnd, act] if rnd in round_action_cross.index else 0
            row += f"| {val:,} "
        row += f"| {round_action_cross.loc[rnd].sum():,} |\n"
        round_table += row

    round_table += "\n### Percentages\n\n"
    round_table += "| Round | " + " | ".join(ACTION_ORDER) + " |\n"
    round_table += "|-------|" + "|".join(["-----:" for _ in ACTION_ORDER]) + "|\n"
    for rnd in rounds:
        row = f"| {rnd} "
        for act in ACTION_ORDER:
            val = round_action_pct.loc[rnd, act] if rnd in round_action_pct.index else 0.0
            row += f"| {val:.1f}% "
        row += "|\n"
        round_table += row

    # -- Figure: grouped bar chart for round comparison --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ACTION_ORDER))
    width = 0.35
    for i, rnd in enumerate(rounds):
        vals = [round_action_cross.loc[rnd, a] if rnd in round_action_cross.index else 0 for a in ACTION_ORDER]
        axes[0].bar(x + i * width, vals, width, label=rnd)
    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(ACTION_ORDER, rotation=15, ha="right")
    axes[0].set_title("Action Counts by Round")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    for i, rnd in enumerate(rounds):
        vals = [round_action_pct.loc[rnd, a] if rnd in round_action_pct.index else 0 for a in ACTION_ORDER]
        axes[1].bar(x + i * width, vals, width, label=rnd)
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels(ACTION_ORDER, rotation=15, ha="right")
    axes[1].set_title("Action % by Round")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend()

    plt.suptitle("R001 vs R002 Label Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "round_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> round_comparison.png")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ 6. Narration Label Consistency                                      ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    narr_actions = df.groupby("narration_norm")["action"].apply(set).reset_index()
    narr_actions["n_labels"] = narr_actions["action"].apply(len)
    multi_label = narr_actions[narr_actions["n_labels"] > 1].copy()
    multi_label["action_labels"] = multi_label["action"].apply(lambda s: " | ".join(sorted(s)))

    # Expand to find all affected rows
    multi_narrations = set(multi_label["narration_norm"])
    affected_rows = df[df["narration_norm"].isin(multi_narrations)].copy()
    affected_rows = affected_rows.sort_values(["narration_norm", "action"])

    # Save multi-label issues CSV
    affected_rows.to_csv(OUT_DIR / "multi_label_issues.csv", index=False)
    print(f"  -> multi_label_issues.csv ({len(affected_rows):,} rows, {len(multi_label):,} unique narrations)")

    consistency_section = f"\n## 6. Narration Label Consistency\n\n"
    consistency_section += f"- Unique normalized narrations: **{len(narr_actions):,}**\n"
    consistency_section += f"- Narrations with exactly 1 action label: **{(narr_actions['n_labels'] == 1).sum():,}**\n"
    consistency_section += f"- Narrations with multiple action labels: **{len(multi_label):,}** ({len(multi_label)/len(narr_actions)*100:.2f}%)\n"
    consistency_section += f"- Affected rows: **{len(affected_rows):,}** ({len(affected_rows)/total_rows*100:.1f}%)\n"

    # Top multi-label narrations
    if len(multi_label) > 0:
        top_multi = affected_rows.groupby("narration_norm").agg(
            n_rows=("action", "size"),
            labels=("action", lambda x: " | ".join(sorted(x.unique())))
        ).sort_values("n_rows", ascending=False).head(20)
        consistency_section += "\n### Top 20 Multi-Label Narrations\n\n"
        consistency_section += "| Narration | # Rows | Labels |\n|-----------|-------:|--------|\n"
        for narr, row in top_multi.iterrows():
            # Truncate long narrations
            display_narr = narr if len(narr) <= 60 else narr[:57] + "..."
            consistency_section += f"| {display_narr} | {row['n_rows']:,} | {row['labels']} |\n"

    # -- Figure: label conflict distribution --
    if len(multi_label) > 0:
        label_pair_counts = multi_label["action_labels"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        label_pair_counts.plot(kind="barh", ax=ax, color=sns.color_palette("Set2", len(label_pair_counts)))
        ax.set_xlabel("Number of Narrations")
        ax.set_title("Top Label Conflict Pairs (Multi-Label Narrations)", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(FIG_DIR / "label_conflict_pairs.png", dpi=150)
        plt.close(fig)
        print("  -> label_conflict_pairs.png")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ 7. Temporal & Intra-Video Action Sequence Patterns                  ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    temporal_section = "\n## 7. Temporal & Intra-Video Action Sequence Patterns\n\n"

    # 7a. Samples per video distribution
    samples_per_video = df.groupby("video_uid").size()
    temporal_section += "### Samples per Video\n\n"
    temporal_section += f"- Mean: {samples_per_video.mean():.1f}\n"
    temporal_section += f"- Median: {samples_per_video.median():.1f}\n"
    temporal_section += f"- Min: {samples_per_video.min()}\n"
    temporal_section += f"- Max: {samples_per_video.max()}\n"
    temporal_section += f"- Std: {samples_per_video.std():.1f}\n"

    # -- Figure: samples per video histogram --
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(samples_per_video.values, bins=50, edgecolor="white", color="steelblue")
    ax.axvline(samples_per_video.mean(), color="red", linestyle="--", label=f"Mean={samples_per_video.mean():.1f}")
    ax.axvline(samples_per_video.median(), color="orange", linestyle="--", label=f"Median={samples_per_video.median():.1f}")
    ax.set_title("Distribution of Samples per Video", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Number of Videos")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "samples_per_video.png", dpi=150)
    plt.close(fig)
    print("  -> samples_per_video.png")

    # 7b. Action transition patterns (bigrams)
    temporal_section += "\n### Action Transition Bigrams (within-video)\n\n"
    df_sorted = df.sort_values(["video_uid", "timestamp_sec"])
    df_sorted["next_action"] = df_sorted.groupby("video_uid")["action"].shift(-1)
    transitions = df_sorted.dropna(subset=["next_action"])
    trans_matrix = pd.crosstab(transitions["action"], transitions["next_action"])
    trans_matrix = trans_matrix.reindex(index=ACTION_ORDER, columns=ACTION_ORDER).fillna(0).astype(int)
    trans_pct = trans_matrix.div(trans_matrix.sum(axis=1), axis=0) * 100

    temporal_section += "Row-normalized transition probabilities (%):\n\n"
    temporal_section += "| From \\ To | " + " | ".join(ACTION_ORDER) + " |\n"
    temporal_section += "|-----------|" + "|".join(["-----:" for _ in ACTION_ORDER]) + "|\n"
    for act_from in ACTION_ORDER:
        row = f"| {act_from} "
        for act_to in ACTION_ORDER:
            row += f"| {trans_pct.loc[act_from, act_to]:.1f}% "
        row += "|\n"
        temporal_section += row

    # -- Figure: transition heatmap --
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(trans_pct, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5,
                ax=ax, cbar_kws={"label": "% Transition Probability"})
    ax.set_title("Action Transition Probabilities (within-video)", fontsize=13, fontweight="bold")
    ax.set_ylabel("From Action")
    ax.set_xlabel("To Action")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "action_transitions.png", dpi=150)
    plt.close(fig)
    print("  -> action_transitions.png")

    # 7c. Action diversity per video
    actions_per_video = df.groupby("video_uid")["action"].nunique()
    temporal_section += f"\n### Action Diversity per Video\n\n"
    temporal_section += f"- Mean unique actions per video: {actions_per_video.mean():.2f}\n"
    temporal_section += f"- Videos with all 5 actions: {(actions_per_video == 5).sum()} / {unique_videos}\n"
    temporal_section += f"- Videos with only 1 action: {(actions_per_video == 1).sum()} / {unique_videos}\n"

    div_dist = actions_per_video.value_counts().sort_index()
    temporal_section += "\n| # Unique Actions | # Videos | % |\n|----------------:|---------:|---:|\n"
    for n_act, n_vid in div_dist.items():
        temporal_section += f"| {n_act} | {n_vid} | {n_vid/unique_videos*100:.1f}% |\n"

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ 8. Class Imbalance Analysis                                         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    imbalance_section = "\n## 8. Class Imbalance Analysis\n\n"

    counts_arr = action_counts.values.astype(float)
    labels_arr = action_counts.index.tolist()

    # Imbalance ratio
    max_count = counts_arr.max()
    min_count = counts_arr.min()
    imbalance_ratio = max_count / min_count
    imbalance_section += f"- Imbalance ratio (max/min): **{imbalance_ratio:.2f}**\n"
    imbalance_section += f"- Majority class: {labels_arr[np.argmax(counts_arr)]} ({int(max_count):,})\n"
    imbalance_section += f"- Minority class: {labels_arr[np.argmin(counts_arr)]} ({int(min_count):,})\n\n"

    # Inverse frequency weights
    inv_freq = total_rows / (len(ACTION_ORDER) * counts_arr)
    inv_freq_norm = inv_freq / inv_freq.sum() * len(ACTION_ORDER)

    # Effective number weights (Cui et al., CVPR 2019)
    eff_weights = effective_number_weights(counts_arr, beta=0.9999)

    # Sqrt inverse frequency (common alternative)
    sqrt_inv = np.sqrt(total_rows / counts_arr)
    sqrt_inv_norm = sqrt_inv / sqrt_inv.sum() * len(ACTION_ORDER)

    imbalance_section += "### Class Weights\n\n"
    imbalance_section += "| Action | Count | Inv-Freq (norm) | Eff-Number (norm) | Sqrt-Inv (norm) |\n"
    imbalance_section += "|--------|------:|----------------:|------------------:|----------------:|\n"
    for i, act in enumerate(labels_arr):
        imbalance_section += (
            f"| {act} | {int(counts_arr[i]):,} | "
            f"{inv_freq_norm[i]:.4f} | {eff_weights[i]:.4f} | {sqrt_inv_norm[i]:.4f} |\n"
        )

    imbalance_section += f"\n*Effective number weights computed with beta=0.9999 (Cui et al., CVPR 2019)*\n"

    # -- Figure: class weights comparison --
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels_arr))
    w = 0.25
    ax.bar(x - w, inv_freq_norm, w, label="Inverse Freq (norm)", color="steelblue")
    ax.bar(x, eff_weights, w, label="Effective Number (norm)", color="coral")
    ax.bar(x + w, sqrt_inv_norm, w, label="Sqrt Inverse (norm)", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_arr, rotation=15, ha="right")
    ax.set_ylabel("Normalized Weight")
    ax.set_title("Class Weights Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "class_weights_comparison.png", dpi=150)
    plt.close(fig)
    print("  -> class_weights_comparison.png")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ Assemble and Write Report                                           ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    report = textwrap.dedent(f"""\
    # Gold Data Quality Report

    **Dataset:** HAR_dataset.csv (final gold dataset)
    **Source script:** `scripts/analyze_gold_data.py`
    **Total rows:** {total_rows:,} | **Status:** All Gold | **Label mapping:** Essential Operation -> Task Operation

    ---

    """)
    report += basic_stats
    report += action_table
    report += f"\n![Action Distribution](figures/action_distribution.png)\n"
    report += scenario_table
    report += f"\n![Scenario Distribution](figures/scenario_distribution.png)\n"
    report += cross_table
    report += f"\n![Action x Scenario Heatmap](figures/action_scenario_heatmap.png)\n"
    report += f"\n![Action x Scenario Heatmap (Row-Normalized %)](figures/action_scenario_heatmap_pct.png)\n"
    report += round_table
    report += f"\n![Round Comparison](figures/round_comparison.png)\n"
    report += consistency_section
    if len(multi_label) > 0:
        report += f"\n![Label Conflict Pairs](figures/label_conflict_pairs.png)\n"
    report += temporal_section
    report += f"\n![Samples per Video](figures/samples_per_video.png)\n"
    report += f"\n![Action Transitions](figures/action_transitions.png)\n"
    report += imbalance_section
    report += f"\n![Class Weights Comparison](figures/class_weights_comparison.png)\n"

    report_path = OUT_DIR / "gold_data_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {report_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Report:        {report_path}")
    print(f"  Figures:       {FIG_DIR}/  ({len(list(FIG_DIR.glob('*.png')))} PNG files)")
    print(f"  Multi-label:   {OUT_DIR / 'multi_label_issues.csv'}")
    print(f"  Total rows:    {total_rows:,}")
    print(f"  Multi-label narrations: {len(multi_label):,} / {len(narr_actions):,} unique")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
