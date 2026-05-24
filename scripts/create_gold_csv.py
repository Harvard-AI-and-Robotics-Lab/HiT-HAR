"""Create unified gold CSV with label rename: Essential Operation -> Task Operation."""
import os
import sys

# Try to use pandas, fall back to manual CSV if unavailable
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def main():
    # Source: already copied into HiT-HAR/data/gold/
    # Original was HAR_Lab_Initiative_AI/data/annotation_rounds/final_gold_dataset/HAR_dataset.csv
    src = os.path.join(os.path.dirname(__file__), '..', 'data', 'gold', 'har_gold_unified.csv')
    src = os.path.abspath(src)
    dst_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'gold')
    dst_dir = os.path.abspath(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, 'har_gold_unified.csv')

    if HAS_PANDAS:
        df = pd.read_csv(src)
        df['action'] = df['action'].replace({'Essential Operation': 'Task Operation'})
        df.to_csv(dst, index=False)
        print(f"Saved {len(df)} rows to {dst}")
        print(f"Action distribution:\n{df['action'].value_counts()}")
    else:
        # Fallback: manual line-by-line replacement
        with open(src, 'r') as f:
            lines = f.readlines()
        with open(dst, 'w') as f:
            for line in lines:
                f.write(line.replace('Essential Operation', 'Task Operation'))
        print(f"Saved {len(lines) - 1} rows to {dst} (manual replacement)")

    # Verify no Essential Operation remains
    with open(dst, 'r') as f:
        content = f.read()
    assert 'Essential Operation' not in content, "ERROR: 'Essential Operation' still found in output!"
    print("Verification PASSED: no 'Essential Operation' in output.")

if __name__ == '__main__':
    main()
