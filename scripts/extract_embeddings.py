"""Extract h_cls embeddings from a trained HiT-HAR model for t-SNE visualization."""
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path so 'src' is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequence-level h_cls embeddings from a checkpoint."
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pth checkpoint')
    parser.add_argument('--config', required=True,
                        help='Path to YAML config used for training')
    parser.add_argument('--test-labels', required=True,
                        help='Path to test action labels CSV')
    parser.add_argument('--processed-dir', default='data/processed_ego4d',
                        help='Directory with <uid>/seq.npz files')
    parser.add_argument('--train-labels', default=None,
                        help='Path to train labels CSV (for computing norm stats)')
    parser.add_argument('--scenario-labels', default='data/labels/scenario_labels.csv',
                        help='Path to scenario_labels.csv')
    parser.add_argument('--output', default='embeddings_test.npz',
                        help='Output .npz file path')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for extraction (override config)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    from src.config import load_config
    config = load_config(args.config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Build model from the config stored in checkpoint, falling back to file config
    ckpt_config = ckpt.get('config', config)
    from src.models.hit_har import build_model
    model = build_model(ckpt_config)

    # Load state dict — prefer EMA weights if available
    if 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
        print("Using EMA state dict from checkpoint")
        state_dict = ckpt['ema_state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        # Assume the checkpoint is a raw state_dict
        state_dict = ckpt

    # Remove 'module.' prefix from DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"WARNING: Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model = model.to(device)
    model.eval()
    print(f"Model loaded: {model.count_parameters():,} parameters")

    # Build test dataset using the same pipeline as training
    import pandas as pd
    from src.data.paired_dataset import PairedDataset, collate_fn

    test_df = pd.read_csv(args.test_labels)
    test_uids = test_df['video_uid'].unique().tolist()
    print(f"Test set: {len(test_uids)} unique videos")

    # Get normalization stats: from checkpoint or recompute from train set
    norm_stats = ckpt.get('train_norm_stats', None)
    if norm_stats is None and args.train_labels is not None:
        print("Norm stats not in checkpoint — computing from train set...")
        train_df = pd.read_csv(args.train_labels)
        train_uids = train_df['video_uid'].unique().tolist()
        # Build a dummy train dataset just to get norm stats
        train_ds = PairedDataset(
            train_uids, args.processed_dir,
            args.train_labels, args.scenario_labels,
            config, training=False, norm_stats=None,
        )
        norm_stats = {'mean': train_ds.global_mean, 'std': train_ds.global_std}
        print(f"Computed norm stats from {len(train_uids)} train videos")

    test_ds = PairedDataset(
        test_uids, args.processed_dir,
        args.test_labels, args.scenario_labels,
        config, training=False, norm_stats=norm_stats,
    )
    print(f"Test dataset: {len(test_ds)} sequences")

    bs = args.batch_size
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=collate_fn,
    )

    # Extract embeddings
    all_h_cls = []
    all_scenario_labels = []
    all_action_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings"):
            inputs = batch['inputs'].to(device)   # (B, S, 50, C)
            out = model(inputs)
            h_cls = out['h_cls'].cpu().numpy()     # (B, 128)
            scenario = batch['scenario_label'].cpu().numpy()  # (B,)
            actions = batch['action_labels'].cpu().numpy()     # (B, S)

            all_h_cls.append(h_cls)
            all_scenario_labels.append(scenario)
            all_action_labels.append(actions)

    h_cls = np.concatenate(all_h_cls, axis=0)
    scenarios = np.concatenate(all_scenario_labels, axis=0)
    actions = np.concatenate(all_action_labels, axis=0)

    print(f"\nExtracted {h_cls.shape[0]} embeddings, dim={h_cls.shape[1]}")

    # Compute dominant action per sequence (mode of valid action labels)
    dominant_actions = []
    for i in range(actions.shape[0]):
        valid = actions[i][actions[i] >= 0]
        if len(valid) > 0:
            vals, counts = np.unique(valid, return_counts=True)
            dominant_actions.append(vals[counts.argmax()])
        else:
            dominant_actions.append(-1)
    dominant_actions = np.array(dominant_actions, dtype=np.int64)

    # Save
    np.savez(args.output,
             h_cls=h_cls,
             scenario_labels=scenarios,
             action_labels=dominant_actions,
             raw_action_labels=actions)

    print(f"Saved to {args.output}")
    print(f"Scenario distribution: {np.bincount(scenarios)}")
    valid_mask = dominant_actions >= 0
    print(f"Action distribution (valid={valid_mask.sum()}/{len(dominant_actions)}): "
          f"{np.bincount(dominant_actions[valid_mask])}")


if __name__ == '__main__':
    main()
