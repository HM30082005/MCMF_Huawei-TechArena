"""Full-data descriptive statistics and chronological baseline comparisons."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parent
CACHE = BASE / '.cache'


def load(prefix):
    return {key: np.load(CACHE / f'{prefix}_{key}.npy', mmap_mode='r') for key in
            ['user', 'items', 'time', 'bucket', 'weekend', 'hist', 'hist_time', 'hist_len', 'label']}


def quantiles(a):
    return dict(zip(['min', 'p10', 'p25', 'median', 'p75', 'p90', 'p99', 'max'],
                    map(float, np.quantile(a, [0, .1, .25, .5, .75, .9, .99, 1]))))


def auc(y, score):
    y = np.asarray(y).ravel()
    score = np.asarray(score).ravel()
    pos = int(y.sum())
    if not pos or pos == len(y):
        return None
    return float((rankdata(score)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def summarize(train, evaluation, sizes):
    output = {}
    for name, data in [('train', train), ('evaluation', evaluation)]:
        uid, ids, stamp, hl = data['user'], data['items'], data['time'], data['hist_len']
        counts = np.bincount(uid)
        seen = np.zeros(ids.shape, bool)
        for k in range(10):
            seen[:, k] = np.any(data['hist'] == ids[:, k, None], axis=1)
        stats = {
            'rows': int(ids.size), 'events': len(ids), 'unique_users': len(np.unique(uid)),
            'unique_candidate_items': len(np.unique(ids)),
            'unique_history_items': len(np.unique(data['hist'][data['hist'] >= 0])),
            'start_utc': datetime.fromtimestamp(int(stamp.min()), timezone.utc).isoformat(),
            'end_utc': datetime.fromtimestamp(int(stamp.max()), timezone.utc).isoformat(),
            'events_per_user': quantiles(counts[counts > 0]),
            'history_length': quantiles(hl), 'empty_history_fraction': float(np.mean(hl == 0)),
            'full_history_fraction': float(np.mean(hl == 20)),
            'candidate_in_history_fraction': float(seen.mean()),
            'last_click_age_hours': quantiles((stamp[hl > 0] - data['hist_time'][np.flatnonzero(hl > 0), hl[hl > 0] - 1]) / 3600),
            'bucket_event_counts': np.bincount(data['bucket'], minlength=4).tolist(),
        }
        if name == 'train':
            y = data['label']
            stats.update({
                'clicks': int(y.sum()), 'row_ctr': float(y.mean()),
                'no_click_event_fraction': float(np.mean(y.sum(axis=1) == 0)),
                'candidate_seen_before_ctr': float(y[seen].mean()) if seen.any() else None,
                'candidate_not_seen_before_ctr': float(y[~seen].mean()),
                'candidate_position_ctr': y.mean(axis=0).tolist(),
                'by_day': [], 'by_bucket': [], 'by_history_length': [],
            })
            for day in np.unique(stamp // 86400):
                mask = stamp // 86400 == day
                stats['by_day'].append({'date': datetime.fromtimestamp(int(day)*86400, timezone.utc).date().isoformat(),
                                        'events': int(mask.sum()), 'ctr': float(y[mask].mean())})
            for bucket in range(4):
                mask = data['bucket'] == bucket
                stats['by_bucket'].append({'bucket': bucket, 'events': int(mask.sum()), 'ctr': float(y[mask].mean())})
            for lo, hi in [(0, 0), (1, 4), (5, 9), (10, 19), (20, 20)]:
                mask = (hl >= lo) & (hl <= hi)
                stats['by_history_length'].append({'length': f'{lo}-{hi}', 'events': int(mask.sum()),
                                                    'ctr': float(y[mask].mean()) if mask.any() else None})
        output[name] = stats
    tu = np.bincount(train['user'], minlength=sizes['all_users'])
    ti = np.bincount(train['items'].ravel(), minlength=sizes['all_items'])
    eu = np.unique(evaluation['user'])
    output['coverage'] = {
        'new_evaluation_users': int(np.sum(tu[eu] == 0)),
        'new_evaluation_user_fraction': float(np.mean(tu[eu] == 0)),
        'evaluation_rows_new_candidate_item_fraction': float(np.mean(ti[evaluation['items']] == 0)),
        'train_events_per_evaluation_user': quantiles(tu[eu]),
        'training_impressions_per_item': quantiles(ti[ti > 0]),
        'all_id_counts': sizes,
    }
    return output


def benchmark(data, sizes, cutoff):
    stamp = int(datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp())
    fit = np.flatnonzero(data['time'] < stamp)
    after = np.flatnonzero(data['time'] >= stamp)
    _, first = np.unique(data['user'][after], return_index=True)
    valid = np.sort(after[first])
    uid = data['user'][fit]
    ids = data['items'][fit]
    y = data['label'][fit]
    nu, ni = sizes['all_users'], sizes['all_items']
    item_impr = np.bincount(ids.ravel(), minlength=ni)
    item_click = np.bincount(ids.ravel(), weights=y.ravel(), minlength=ni)
    user_impr = np.bincount(uid, minlength=nu) * 10
    user_click = np.bincount(uid, weights=y.sum(axis=1), minlength=nu)
    rate = float(y.mean())
    item_p = (item_click + 50 * rate) / (item_impr + 50)
    user_p = (user_click + 50 * rate) / (user_impr + 50)
    vi, vu, vy = data['items'][valid], data['user'][valid], data['label'][valid]
    base = logit(item_p[vi]) + logit(user_p[vu, None]) - logit(rate)
    metrics = {'item_ctr': auc(vy, item_p[vi]),
               'user_ctr': auc(vy, np.broadcast_to(user_p[vu, None], vy.shape)),
               'item_plus_user_ctr': auc(vy, base)}
    # Reproduce the starter's binary per-user interaction sets with sparse algebra.
    hist = data['hist'][fit]
    hu = np.broadcast_to(uid[:, None], hist.shape)[hist >= 0]
    hi = hist[hist >= 0]
    clicked_u = np.broadcast_to(uid[:, None], ids.shape)[y == 1]
    clicked_i = ids[y == 1]
    interactions = sparse.coo_matrix((np.ones(len(hi) + len(clicked_i), np.float32),
                                      (np.concatenate([hu, clicked_u]), np.concatenate([hi, clicked_i]))),
                                     shape=(nu, ni)).tocsr()
    interactions.data.fill(1)
    popularity = np.asarray(interactions.sum(axis=0)).ravel()
    co = (interactions.T @ interactions).tocsr()
    co.setdiag(0)
    scale = sparse.diags(1 / np.sqrt(np.maximum(popularity, 1)))
    co = (scale @ co @ scale).tocsr()
    # Keep the 50,000-item graph sparse and retrieve only required pairs.
    cos = co
    vh, vt = data['hist'][valid], data['hist_time'][valid]
    similarity = np.zeros(vi.shape)
    recency_similarity = np.zeros(vi.shape)
    for k in range(20):
        h = vh[:, k]
        pairs = cos[vi, np.maximum(h[:, None], 0)]
        value = (pairs.toarray() if sparse.issparse(pairs) else np.asarray(pairs)) * (h[:, None] >= 0)
        similarity += value
        age_hours = (data['time'][valid] - vt[:, k]) / 3600
        recency_similarity += value * np.exp2(-age_hours[:, None] / 24)
    metrics['starter_exact_formula'] = auc(vy, base + 25 * similarity)
    repeats = np.sum(vi[:, :, None] == vh[:, None, :], axis=2)
    metrics['item_user_plus_repeat2'] = auc(vy, base + 2 * (repeats > 0))
    metrics['starter_plus_repeat2'] = auc(vy, base + 25 * similarity + 2 * (repeats > 0))
    metrics['repeat_weight_sweep'] = {str(w): auc(vy, base + 25 * similarity + w * (repeats > 0)) for w in [.5, 1, 1.5, 2, 3]}
    metrics['cf_weight_sweep'] = {str(w): auc(vy, base + w * similarity) for w in [0, 5, 10, 25, 50, 100]}
    metrics['recency_cf_weight_sweep'] = {str(w): auc(vy, base + w * recency_similarity) for w in [5, 10, 25, 50, 100]}
    # Context-conditioned popularity: each bucket shrinks towards the item's pooled CTR.
    item_bucket = ids * 4 + data['bucket'][fit, None]
    exposure = np.bincount(item_bucket.ravel(), minlength=ni * 4).reshape(ni, 4)
    clicks = np.bincount(item_bucket.ravel(), weights=y.ravel(), minlength=ni * 4).reshape(ni, 4)
    bucket_p = (clicks + 100 * item_p[:, None]) / (exposure + 100)
    time_base = logit(bucket_p[vi, data['bucket'][valid, None]]) + logit(user_p[vu, None]) - logit(rate)
    metrics['time_context_plus_cf25'] = auc(vy, time_base + 25 * similarity)
    cold = user_impr[vu] == 0
    empty = data['hist_len'][valid] == 0
    for name, mask in [('known_users', ~cold), ('new_users', cold), ('empty_history', empty), ('has_history', ~empty)]:
        metrics[name] = {'events': int(mask.sum()), 'clicks': int(vy[mask].sum()),
                         'starter_auc': auc(vy[mask], (base + 25 * similarity)[mask]) if mask.any() else None,
                         'popularity_auc': auc(vy[mask], item_p[vi][mask]) if mask.any() else None}
    result = {'cutoff_utc': cutoff, 'fit_events': len(fit), 'valid_events': len(valid),
              'valid_clicks': int(vy.sum()), 'valid_new_user_fraction': float(cold.mean()),
              'valid_empty_history_fraction': float(empty.mean()), 'metrics': metrics}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    train, evaluation = load('train'), load('evaluation')
    sizes = json.loads((CACHE / 'id_counts.json').read_text())
    report = summarize(train, evaluation, sizes)
    print(json.dumps(report, indent=2), flush=True)
    (BASE / 'dataset_profile.json').write_text(json.dumps(report, indent=2))
    report['validation'] = [benchmark(train, sizes, cutoff) for cutoff in ['2025-01-25', '2025-01-26']]
    (BASE / 'dataset_profile.json').write_text(json.dumps(report, indent=2))
