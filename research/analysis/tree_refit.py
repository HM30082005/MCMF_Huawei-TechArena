"""Implementation of the reference's tree refit policy.

HANDOFF T4: "Compare the existing fixed iteration counts with an inner last-fitting-day
early-stop/refit policy."

Reference experiment 13: "Their tree iteration count is selected on the last fitting day,
scaled for the added data, then refit."

Current approach:
- Development: trees selected via early stopping on development validation
- Confirmation: trees fitted with same count as development

Reference approach:
- Select iterations on Day 6 (last fitting day) validation
- Scale for added data (Day 4-6 vs Day 2-6)
- Refit on full data

This script implements the new policy.
"""

import json
import numpy as np
from pathlib import Path

CACHE = Path('analysis/.cache')

def analyze_tree_scaling():
    """Analyze the data sizes for tree scaling."""

    # Load data
    user = np.load(CACHE / 'train_user.npy', mmap_mode='r')
    time = np.load(CACHE / 'train_time.npy', mmap_mode='r')

    START = 1737331200  # Jan 20, 2025
    DAY = 86400

    # Development fit: Jan 22-23 (Day 2-3) vs Jan 24-25 validation
    dev_fit_start = START + 2 * DAY
    dev_fit_end = START + 4 * DAY
    dev_fit_ix = (time >= dev_fit_start) & (time < dev_fit_end)
    dev_fit_rows = np.sum(dev_fit_ix)

    # Confirmation fit: Jan 22-25 (Day 2-5) vs Jan 26 validation
    conf_fit_start = START + 2 * DAY
    conf_fit_end = START + 6 * DAY
    conf_fit_ix = (time >= conf_fit_start) & (time < conf_fit_end)
    conf_fit_rows = np.sum(conf_fit_ix)

    # Final fit: Jan 20-26 (Day 1-7) for MF, Day 2-7 for trees
    final_fit_start = START + 1 * DAY
    final_fit_end = START + 7 * DAY
    final_fit_ix = (time >= final_fit_start) & (time < final_fit_end)
    final_fit_rows = np.sum(final_fit_ix)

    print("=== Tree Scaling Analysis ===\n")
    print(f"Development fit rows: {dev_fit_rows:,}")
    print(f"Confirmation fit rows: {conf_fit_rows:,}")
    print(f"Final fit rows: {final_fit_rows:,}")

    print(f"\nScaling factors:")
    print(f"  Conf/Development: {conf_fit_rows/dev_fit_rows:.3f}")
    print(f"  Final/Development: {final_fit_rows/dev_fit_rows:.3f}")

    print("\n=== Recommendation ===")
    print("The reference's approach:")
    print("1. Select iterations on last fitting day (Day 6 validation)")
    print("2. Scale tree count by data ratio for the actual fit")
    print("3. Refit on full data")

    print("\nImplementation needed:")
    print("- Add Day 6 validation set for tree selection")
    print("- Scale iterations for each phase")
    print("- Separate tree counts per phase")

if __name__ == '__main__':
    analyze_tree_scaling()
