"""An exploratory low-rank history model; all learned statistics precede the cutoff."""
import json
from datetime import datetime, timezone

import numpy as np
from scipy import sparse

from study import BASE, CACHE, auc, load, logit


def run(data, sizes, cutoff, rank=64):
    stamp = int(datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc).timestamp())
    fit = np.flatnonzero(data['time'] < stamp)
    after = np.flatnonzero(data['time'] >= stamp)
    _, first = np.unique(data['user'][after], return_index=True)
    valid = np.sort(after[first])
    nu, ni = sizes['all_users'], sizes['all_items']
    uid, ids, hist, labels = (data[k][fit] for k in ['user', 'items', 'hist', 'label'])
    hu = np.broadcast_to(uid[:, None], hist.shape)[hist >= 0]
    hi = hist[hist >= 0]
    cu = np.broadcast_to(uid[:, None], ids.shape)[labels == 1]
    ci = ids[labels == 1]
    x = sparse.coo_matrix((np.ones(len(hi) + len(ci), np.float32),
                           (np.concatenate([hu, cu]), np.concatenate([hi, ci]))), shape=(nu, ni)).tocsr()
    x.data.fill(1)
    row_pop = np.asarray(x.sum(axis=1)).ravel()
    col_pop = np.asarray(x.sum(axis=0)).ravel()
    x = (sparse.diags(1 / np.sqrt(np.maximum(row_pop, 1))) @ x @
         sparse.diags(1 / np.sqrt(np.maximum(col_pop, 1)))).tocsr()
    rng = np.random.default_rng(42)
    q, _ = np.linalg.qr(x @ rng.normal(size=(ni, rank + 16)).astype(np.float32), mode='reduced')
    for step in range(3):
        z, _ = np.linalg.qr(x.T @ q, mode='reduced')
        q, _ = np.linalg.qr(x @ z, mode='reduced')
        print(cutoff, 'spectral iteration', step + 1, flush=True)
    _, s, vt = np.linalg.svd(np.asarray(q.T @ x), full_matrices=False)
    # Discard the popularity/trivial graph component; retain learned item relationships.
    emb = vt[1:rank + 1].T * np.sqrt(s[1:rank + 1])
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    vi, vh, vy = (data[k][valid] for k in ['items', 'hist', 'label'])
    query = np.zeros((len(valid), rank), np.float32)
    for k in range(20):
        h = vh[:, k]
        query += emb[np.maximum(h, 0)] * (h[:, None] >= 0)
    query /= np.maximum(data['hist_len'][valid, None], 1)
    score = np.zeros(vi.shape)
    for k in range(10):
        score[:, k] = np.einsum('ij,ij->i', emb[vi[:, k]], query)
    rate = labels.mean()
    item_n = np.bincount(ids.ravel(), minlength=ni)
    item_y = np.bincount(ids.ravel(), weights=labels.ravel(), minlength=ni)
    user_n = np.bincount(uid, minlength=nu) * 10
    user_y = np.bincount(uid, weights=labels.sum(axis=1), minlength=nu)
    ip = (item_y + 50 * rate) / (item_n + 50)
    up = (user_y + 50 * rate) / (user_n + 50)
    base = logit(ip[vi]) + logit(up[data['user'][valid, None]]) - logit(rate)
    result = {'cutoff': cutoff, 'valid_events': len(valid), 'rank': rank,
              'history_embedding_only_auc': auc(vy, score),
              'history_embedding_plus_priors': {str(w): auc(vy, base + w * score) for w in [1, 2, 5, 10, 20]}}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    data = load('train')
    sizes = json.loads((CACHE / 'id_counts.json').read_text())
    result = [run(data, sizes, cutoff) for cutoff in ['2025-01-25', '2025-01-26']]
    (BASE / 'spectral_probe.json').write_text(json.dumps(result, indent=2))
