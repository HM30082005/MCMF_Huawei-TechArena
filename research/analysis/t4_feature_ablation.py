#!/usr/bin/env python3
"""T4 experiment: Feature ablation analysis for tree fitting.

Analyzes feature groups for potential ablation studies.
"""
import json
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "artifacts" / "history_upgrade"

# Load results from t3-budget-seeds30
results = json.loads((OUT / "results.json").read_text())
all_features = results["feature_names"]

print(f"Total features in baseline: {len(all_features)}")
print(f"\nTop 15 features by importance:")
for f in results["feature_importance"][:15]:
    print(f"  {f['feature']}: {f['importance']:.0f}")

# Define feature groups for ablation
mf_features = [f for f in all_features if f.startswith("mf_")]
spectral_features = [f for f in all_features if f.startswith("spectral_")]
cf_features = [f for f in all_features if f.startswith("cf_")]
ctr_features = [f for f in all_features if "ctr" in f.lower()]
extended_history_features = [f for f in all_features if f.startswith("long_")]
time_features = [f for f in all_features if f in ["time_bucket", "utc_hour", "weekend"]]
basic_features = [f for f in all_features if f not in mf_features and not f.startswith("spectral_")
                  and not f.startswith("cf_") and "ctr" not in f.lower()
                  and not f.startswith("long_") and f not in time_features]

print(f"\nFeature group sizes:")
print(f"  MF features (seed disagreement signals): {len(mf_features)}")
print(f"  Spectral features: {len(spectral_features)}")
print(f"  CF features (collaborative filtering): {len(cf_features)}")
print(f"  CTR features: {len(ctr_features)}")
print(f"  Extended history features: {len(extended_history_features)}")
print(f"  Time features: {len(time_features)}")
print(f"  Basic behavioral features: {len(basic_features)}")

# Test 1: Spectral features importance
spectral_importance = sum(f["importance"] for f in results["feature_importance"] if f["feature"].startswith("spectral_"))
total_importance = sum(f["importance"] for f in results["feature_importance"])
print(f"\nSpectral features contribution: {spectral_importance/total_importance*100:.1f}% of total importance")

# Test 2: Extended history features importance
extended_importance = sum(f["importance"] for f in results["feature_importance"] if f["feature"].startswith("long_"))
print(f"Extended history features contribution: {extended_importance/total_importance*100:.1f}% of total importance")

# Test 3: Compact feature sets
print(f"\n=== Suggested compact feature sets for T4 ===")
spectral_removed = [f for f in all_features if not f.startswith("spectral_")]
print(f"1. Remove spectral: {len(spectral_removed)} features ({len(spectral_features)} removed)")

cf_removed = [f for f in all_features if not f.startswith("cf_")]
print(f"2. Remove CF: {len(cf_removed)} features ({len(cf_features)} removed)")

extended_removed = [f for f in all_features if not f.startswith("long_")]
print(f"3. Remove extended history: {len(extended_removed)} features ({len(extended_history_features)} removed)")

spectral_cf_removed = [f for f in all_features if not (f.startswith("spectral_") or f.startswith("cf_"))]
print(f"4. Remove spectral+CF: {len(spectral_cf_removed)} features ({len(spectral_features) + len(cf_features)} removed)")

# Reference-like set (39 features mentioned in HANDOFF)
# The reference uses 39 features vs our 70
print(f"\nBaseline: 70 features")
print(f"Reference: 39 features (per HANDOFF section 4)")

# MF features (seed disagreement signals) importance
mf_importance = sum(f["importance"] for f in results["feature_importance"] if f["feature"].startswith("mf_"))
print(f"\nMF features contribution: {mf_importance/total_importance*100:.1f}% of total importance")
