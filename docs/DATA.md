# Data

The full HiT-HAR dataset package is intended to be distributed through the Harvard AI and Robotics Lab Hugging Face organization:

https://huggingface.co/harvardairobotics/datasets

This GitHub repository does not include the full CSV splits, raw annotation exports, model checkpoints, or raw Ego4D assets.

## Expected Layout

After downloading the dataset package, arrange the files as follows:

```text
data/
├── processed_clean/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── labels/
│   └── scenario_labels.csv
└── processed_ego4d/
    └── <video_uid>/seq.npz
```

`processed_ego4d/` should contain one directory per video. Each video directory should contain `seq.npz`, with preprocessed IMU windows and timestamp arrays.

## Label Schema

The training and evaluation CSV files are expected to contain:

| Column | Description |
| --- | --- |
| `video_uid` | Ego4D video identifier. |
| `timestamp_sec` | Narration timestamp in seconds. |
| `narration_text` | Original narration text. |
| `scenario` | Scenario label. |
| `action` | One of the five HiT-HAR action classes. |
| `source` | Label source, such as human-verified or propagated. |
| `confidence` | Per-sample confidence weight. |
| `tier` | Label quality tier. |
| `action_id` | Integer action label. |

## Action Classes

| ID | Action |
| --- | --- |
| 0 | Object Transfer |
| 1 | Task Operation |
| 2 | Stationary |
| 3 | Locomotion |
| 4 | Search |

## Notes

- `processed_clean` is the recommended split for paper-aligned experiments.
- Splits are grouped by `video_uid` to avoid video leakage.
- Raw Ego4D assets are governed by Ego4D access terms and are not redistributed here.
- Raw annotation exports should be reviewed for privacy and release policy before public distribution.
