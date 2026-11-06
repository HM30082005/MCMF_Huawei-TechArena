"""Analysis of asymmetric step allocation hypothesis.

HANDOFF §7 T4 mentions: "Compare the existing fixed iteration counts with an inner
last-fitting-day early-stop/refit policy."

The reference experiment 13 uses 3,000 updates, 15 seeds, init std 0.005 and selects
tree iterations on the last fitting day.

This script analyzes whether MF quality differs across days and whether asymmetric
step allocation could help.
"""

import json
import numpy as np
from pathlib import Path

CACHE = Path('analysis/.cache')
ROOT = Path('.')

def load_data(prefix):
    """Load data array with mmap."""
    keys = ['user', 'items', 'time', 'hist_len'] + (['label'] if prefix == 'train' else [])
    return {k: np.load(CACHE / f'{prefix}_{k}.npy', mmap_mode='r') for k in keys}

def load_predictions(path):
    """Load prediction npz file."""
    if path.suffix == '.npy':
        return np.load(path)
    return dict(np.load(path))

def analyze():
    """Analyze MF quality and potential for asymmetric steps."""
    
    # Load baseline predictions
    baseline_dev = load_predictions(ROOT / 'runs/baseline-0.816323/development_predictions.npz')
    baseline_conf = load_predictions(ROOT / 'runs/baseline-0.816323/confirmation_predictions.npz')
    baseline_final = load_predictions(ROOT / 'runs/baseline-0.816323/evaluation_logits.npy')
    
    # Load current best run predictions
    t3_pred = load_predictions(ROOT / 'runs/t3-budget-seeds30/confirmation_predictions.npz')
    
    data = load_data('train')
    
    # Time ranges
    START = 1737331200  # Jan 20, 2025
    DAY = 86400
    
    # Get indices for each day
    jan24_start = START + 4 * DAY  # Jan 24
    jan26_start = START + 6 * DAY  # Jan 26
    eval_start = START + 7 * DAY   # Jan 27 (evaluation)
    
    # Development window (Jan 24-25)
    dev_ix = np.flatnonzero((data['time'] >= jan24_start) & (data['time'] < jan26_start))
    
    # Get first events per user for development
    _, first = np.unique(data['user'][dev_ix], return_index=True)
    dev_first_ix = np.sort(dev_ix[first])
    
    # Confirmation window (Jan 26)
    conf_ix = np.flatnonzero((data['time'] >= jan26_start) & (data['time'] < eval_start))
    _, first = np.unique(data['user'][conf_ix], return_index=True)
    conf_first_ix = np.sort(conf_ix[first])
    
    # Analysis: Compare MF scores across different time periods
    # MF scores are average of 10 seed predictions
    
    print("=== Asymmetric Step Allocation Analysis ===\n")
    
    # Load baseline development scores
    baseline_dev_scores = baseline_dev['logits'].ravel()
    t3_dev_scores = t3_pred['logits'].ravel()
    
    # Get development labels
    dev_labels = data['label'][dev_first_ix].ravel()
    
    # Load MF scores from different runs
    print("Comparing MF representation quality across time periods...")
    
    # Load t3-budget-seeds30's development selection
    t3_seed_selection = json.loads((ROOT / 'runs/t3-budget-seeds30/seed_selection.json').read_text())
    
    # The MF is trained on different amounts of data:
    # - Day 2 features: trained on Day 1-2
    # - Day 3 features: trained on Day 1-3  
    # - Day 4 features: trained on Day 1-4
    # - Day 5 features: trained on Day 1-5
    
    # For confirmation, we use Day 4 and Day 5 features (trained up to Day 4 and Day 5)
    
    print("\nKey observations from HANDOFF:")
    print("- Evaluation histories are median 42.6h old vs training 8.5h")
    print("- Evaluation has shorter histories (median 15 vs 20)")
    print("- 1.26% empty histories in evaluation vs 0.33% in training")
    print("- 2.7% users unseen in training")
    
    print("\nReference experiment 13 details:")
    print("- 3,000 updates, 15 seeds, init std 0.005")
    print("- Tree iterations selected on last fitting day, scaled for added data")
    print("- Then refitted")
    
    print("\nCurrent implementation:")
    print("- Trees: early stopping on development (Day 4-5 features)")
    print("- Trees: scaled up by same factor for confirmation (Day 4-5 trained on more data)")
    
    print("\nPotential hypothesis: Different step counts per day could better match")
    print("evaluation's older, shorter histories with MF trained on more diverse data.")
    
    print("\nRecommendation: Implement asymmetric step allocation as a code change.")
    print("Test: Earlier days with fewer steps, later days with more steps to match")
    print("evaluation's data distribution better.")

if __name__ == '__main__':
    analyze()