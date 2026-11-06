#!/usr/bin/env python3
"""Test CatBoost + LightGBM blends using confirmation predictions.

T5 experiment: Test blend weights on confirmation data (Jan 26) to estimate blend gain.
Note: This is for analysis only - final blends should be tested on development only.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts' / 'history_upgrade'

def main():
    # Load confirmation predictions from baseline - these have labels
    lgbm_conf = np.load(OUT / 'confirmation_LightGBM.npz')
    cat_conf = np.load(OUT / 'confirmation_CatBoost.npz')

    print("LightGBM confirmation keys:", list(lgbm_conf.keys()))
    print("CatBoost confirmation keys:", list(cat_conf.keys()))

    # Load labels
    lgbm_labels = lgbm_conf['labels'].ravel()
    lgbm_logits = lgbm_conf['logits'].ravel()
    cat_logits = cat_conf['logits'].ravel()

    print(f"\nLabels shape: {lgbm_labels.shape}")
    print(f"LGBM logits shape: {lgbm_logits.shape}")
    print(f"CatBoost logits shape: {cat_logits.shape}")

    print(f"Positive rate: {lgbm_labels.mean():.4f}")
    print(f"LGBM AUC (confirmation): {roc_auc_score(lgbm_labels, lgbm_logits):.6f}")
    print(f"CatBoost AUC (confirmation): {roc_auc_score(lgbm_labels, cat_logits):.6f}")

    # Test blend weights
    print("\n=== Blending logits (confirmation data) ===")
    best_auc = 0
    best_w = 0.5

    for w in np.arange(0, 1.01, 0.05):
        blended = w * lgbm_logits + (1-w) * cat_logits
        auc = roc_auc_score(lgbm_labels, blended)
        marker = " *" if auc > best_auc else ""
        print(f"Weight LGBM={w:.2f}: AUC={auc:.6f}{marker}")
        if auc > best_auc:
            best_auc = auc
            best_w = w

    print(f"\nBest blend: weight={best_w:.2f}, AUC={best_auc:.6f}")
    print(f"Best model alone: LGBM={roc_auc_score(lgbm_labels, lgbm_logits):.6f}")
    print(f"Ensemble gain: {best_auc - roc_auc_score(lgbm_labels, lgbm_logits):.6f}")

if __name__ == '__main__':
    main()
