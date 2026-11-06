"""P1 Audit: Count additional real historical clicks from supplied query histories.

This script audits the main unclosed implementation difference from the reference:
the use of pre-cutoff clicks from the supplied histories of held-out/evaluation queries.

Current implementation (A): histories of fitting events + clicked fitting candidates
Reference implementation (B): A + allowable pre-cutoff clicks from query histories

These are real historical positives (clicks) available in the query histories but
not currently used for MF fitting. This analysis counts what would be added.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import csv
from collections import Counter

import numpy as np

BASE = Path(__file__).resolve().parent
CACHE = BASE / '.cache'
ROOT = BASE.parent

# Cutoff dates to analyze
CUTOFFS = ['2025-01-24', '2025-01-25', '2025-01-26']


def load_ids():
    """Load user and item ID mappings."""
    train_items = json.loads((CACHE / 'item_ids.json').read_text())
    all_items = len(train_items)
    
    users = json.loads((CACHE / 'user_ids.json').read_text())
    all_users = len(users)
    
    return all_users, all_items


def count_query_history_clicks_for_cutoff():
    """
    Count unique clicks from supplied query histories that would be
    available before each cutoff date.
    
    These are clicks that appear in the history_item_ids of evaluation events
    but are not currently used in MF fitting.
    """
    eval_csv = ROOT / 'evaluation.csv'
    train_csv = ROOT / 'train.csv'
    
    # Build a set of all training+fitting clicks (what we currently use)
    # Using string keys for user IDs to match evaluation data
    training_clicks = set()  # (user_str, item_str, timestamp) tuples
    
    # First pass: count clicks in training data
    with train_csv.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        col_user = header.index('user_id')
        col_item = header.index('item_id')
        col_time = header.index('timestamp')
        col_hist_items = header.index('history_item_ids')
        col_hist_times = header.index('history_timestamps')
        
        for row in reader:
            user = row[col_user]
            timestamp = int(row[col_time])
            
            # Training labels - the clicked item
            clicked_item = row[col_item]
            training_clicks.add((user, clicked_item, timestamp))
            
            # History items from training events
            hist_items = json.loads(row[col_hist_items])
            hist_times = json.loads(row[col_hist_times])
            for item, t in zip(hist_items, hist_times):
                training_clicks.add((user, item, t))
    
    print(f"Training clicks (including history): {len(training_clicks):,}")
    
    # Second pass: analyze evaluation query histories
    eval_clicks_by_cutoff = {cutoff: set() for cutoff in CUTOFFS}
    new_users_by_cutoff = {cutoff: set() for cutoff in CUTOFFS}
    new_items_by_cutoff = {cutoff: set() for cutoff in CUTOFFS}
    
    with eval_csv.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        col_user = header.index('user_id')
        col_hist_items = header.index('history_item_ids')
        col_hist_times = header.index('history_timestamps')
        
        cutoff_timestamps = {
            cutoff: int(datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp())
            for cutoff in CUTOFFS
        }
        
        for row in reader:
            user = row[col_user]
            hist_items = json.loads(row[col_hist_items])
            hist_times = json.loads(row[col_hist_times])
            
            # Check each history click against each cutoff
            for item, t in zip(hist_items, hist_times):
                click_key = (user, item, t)
                
                # Is this not in training clicks?
                if click_key not in training_clicks:
                    for cutoff, cutoff_ts in cutoff_timestamps.items():
                        if t < cutoff_ts:
                            eval_clicks_by_cutoff[cutoff].add(click_key)
                            new_users_by_cutoff[cutoff].add(user)
                            new_items_by_cutoff[cutoff].add(item)
    
    # Report results
    results = {}
    for cutoff in CUTOFFS:
        new_clicks = eval_clicks_by_cutoff[cutoff]
        new_users = new_users_by_cutoff[cutoff]
        new_items = new_items_by_cutoff[cutoff]
        
        results[cutoff] = {
            'new_unique_clicks': len(new_clicks),
            'new_unique_users': len(new_users),
            'new_unique_items': len(new_items),
        }
        
        print(f"\n{cutoff}:")
        print(f"  New unique clicks: {len(new_clicks):,}")
        print(f"  New unique users: {len(new_users):,}")
        print(f"  New unique items: {len(new_items):,}")
    
    return results


def analyze_rare_item_support():
    """
    Analyze how the additional clicks would affect support for rare items.
    Items with fewer than 10 training impressions are considered 'rare'.
    """
    eval_csv = ROOT / 'evaluation.csv'
    
    # Count training impressions per item (using string IDs to match eval)
    item_impressions = Counter()
    
    with (ROOT / 'train.csv').open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        col_item = header.index('item_id')
        col_hist_items = header.index('history_item_ids')
        
        for row in reader:
            # Candidate item
            item_impressions[row[col_item]] += 1
            # History items
            hist_items = json.loads(row[col_hist_items])
            for item in hist_items:
                item_impressions[item] += 1
    
    # Identify rare items (< 10 impressions)
    rare_items = {item for item, count in item_impressions.items() if count < 10}
    total_items = len(item_impressions)
    
    print(f"\nRare item analysis:")
    print(f"  Total items: {total_items:,}")
    print(f"  Rare items (< 10 impressions): {len(rare_items):,}")
    print(f"  Rare item fraction: {len(rare_items) / total_items:.4f}")
    
    # Count how many rare items appear in evaluation histories
    rare_in_eval_hist = set()
    with eval_csv.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        col_hist_items = header.index('history_item_ids')
        
        for row in reader:
            hist_items = json.loads(row[col_hist_items])
            for item in hist_items:
                if item in rare_items:
                    rare_in_eval_hist.add(item)
    
    print(f"  Rare items in eval histories: {len(rare_in_eval_hist):,}")
    
    return {
        'total_items': total_items,
        'rare_items': len(rare_items),
        'rare_item_fraction': len(rare_items) / total_items,
        'rare_in_eval_hist': len(rare_in_eval_hist),
    }


if __name__ == '__main__':
    print("P1 Audit: Counting additional historical clicks from query histories")
    print("=" * 60)
    
    all_users, all_items = load_ids()
    print(f"Total users in evaluation: {all_users:,}")
    print(f"Total items: {all_items:,}")
    
    print("\n" + "=" * 60)
    print("Counting new unique clicks from query histories...")
    print("=" * 60)
    
    results = count_query_history_clicks_for_cutoff()
    rare_info = analyze_rare_item_support()
    
    output = {
        'evaluation_users': all_users,
        'evaluation_items': all_items,
        'clicks_by_cutoff': results,
        'rare_item_analysis': rare_info,
        'hypothesis': 'Adding pre-cutoff clicks from evaluation query histories to MF fitting improves representation for cold users, short histories, and rare items',
        'information_access_policy': 'Only using clicks that appear in supplied query histories, filtered to before each day-specific fitting cutoff',
    }
    
    # Save results
    (BASE / 'p1_audit_results.json').write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {BASE / 'p1_audit_results.json'}")