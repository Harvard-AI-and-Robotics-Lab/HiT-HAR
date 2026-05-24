# Model Zoo

Pretrained checkpoints are released separately from Git history. Download the checkpoint files from the project release assets or the Harvard AI and Robotics Lab Hugging Face organization, then place them under:

```text
checkpoints/
```

The headline paper checkpoint is `hit-har-5class-beta03-headline.pth`.

## Checkpoints

| File | Description | Size | SHA-256 |
| --- | --- | ---: | --- |
| `hit-har-5class-beta03-headline.pth` | Headline five-class HiT-HAR checkpoint, beta=0.3. | 2.7 MB | `25c8c6f42ec1b27df98b68744d4b97c3d90870d67c9c7d903c92cb1a3a278415` |
| `hit-har-5class-beta00-action-only.pth` | Five-class action-only ablation, beta=0.0. | 2.7 MB | `dd832372d450ab9f5a47a8dd1b6398664b12e2540d1eb43de24387d00d390b95` |
| `hit-har-5class-lle-baseline.pth` | Lightweight local encoder baseline. | 2.7 MB | `dccab20ff2c04c56e748a39ad776440aa23dd1fb37acf6dad67bd825bc3b481e` |
| `hit-har-5class-v3-nolang-beta03.pth` | Larger five-class no-language ablation, beta=0.3. | 4.5 MB | `b8a437bbdab59de8c80536f5e497b624968d91e9f9c026004cde1be909c2235c` |
| `hit-har-5class-v3-language-beta03.pth` | Larger five-class language-guided ablation, beta=0.3. | 424 MB | `a6e88683ff010c3e08362d12e7369e119e7afdb76dc67565cd35d227be4de23d` |
| `hit-har-4class-v3-language.pth` | Four-class language-guided ablation. | 424 MB | `4bd82116f4a88fcb386ed2d79d3e048c5e24bef20ccde1f9691c42745264b200` |

## Evaluate

```bash
python src/evaluation/evaluate.py \
  --checkpoint checkpoints/hit-har-5class-beta03-headline.pth \
  --config configs/beta_sweep_5class_b03.yaml \
  --processed-dir data/processed_ego4d \
  --test-labels data/processed_clean/test.csv \
  --train-labels data/processed_clean/train.csv \
  --scenario-labels data/labels/scenario_labels.csv
```

## Notes

- The two 424 MB checkpoints include larger language-guided variants and should be distributed through release assets or Hugging Face rather than committed to Git.
- Checkpoints include model weights, training configuration, and normalization statistics when available.
- Raw dataset files are still required for evaluation.
