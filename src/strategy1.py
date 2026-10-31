"""Supervised history-aware logistic factorization for TechArena click prediction.

Run `python src/strategy1.py train` for dimension selection, a locked chronological
confirmation, full-data fitting, and a checked submission. No evaluation labels
are available or consumed. `predict` reloads the saved full-data checkpoint.
"""
import argparse
import csv
import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
from scipy import sparse
from scipy.special import expit
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'analysis' / '.cache'
OUT = ROOT / 'artifacts' / 'strategy1'
VERSION = 1


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def utc_stamp(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def roc_auc(y, scores):
    y, scores = np.asarray(y).ravel(), np.asarray(scores).ravel()
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite predictions')
    positive = int(y.sum())
    if positive == 0 or positive == len(y):
        return None
    return float((rankdata(scores)[y == 1].sum(dtype=np.float64) - positive * (positive + 1) / 2)
                 / (positive * (len(y) - positive)))


def metrics(y, logits):
    y = np.asarray(y)
    logits = np.asarray(logits, dtype=np.float64)
    return {'roc_auc': roc_auc(y, logits),
            'log_loss': float(np.mean(np.logaddexp(0, logits) - y * logits)),
            'rows': int(y.size), 'clicks': int(y.sum())}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(prefix):
    keys = ['user', 'items', 'time', 'bucket', 'weekend', 'hist', 'hist_time', 'hist_len']
    if prefix == 'train':
        keys.append('label')
    return {key: np.load(CACHE / f'{prefix}_{key}.npy', mmap_mode='r') for key in keys}


@lru_cache(maxsize=1)
def cache_fingerprint():
    """Bind cached initialization and saved checkpoints to actual encoded inputs."""
    digest = hashlib.sha256()
    files = sorted(CACHE.glob('train_*.npy')) + [CACHE / name for name in
            ['user_ids.json', 'item_ids.json', 'id_counts.json']]
    for path in files:
        digest.update(path.name.encode())
        with path.open('rb') as src:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def chronological_split(data, cutoff, end=None):
    fit = np.flatnonzero(data['time'] < utc_stamp(cutoff))
    mask = data['time'] >= utc_stamp(cutoff)
    if end:
        mask &= data['time'] < utc_stamp(end)
    after = np.flatnonzero(mask)
    _, first = np.unique(data['user'][after], return_index=True)
    valid = np.sort(after[first])
    if not len(fit) or not len(valid) or np.intersect1d(fit, valid).size:
        raise ValueError('Invalid chronological split')
    return fit, valid


def spectral_initialization(data, fit, sizes, dim, name, seed=42, output_dir=None):
    """Fitting-only binary user/item graph; repeated history records count once.

    Binarizing each user-item edge deduplicates (user,item,timestamp) records
    and additionally treats repeat clicks on the same item as one graph edge.
    Actual repeat occurrences remain available to supervised history features.
    """
    path = Path(output_dir or OUT) / f'{name}_spectral_{dim}.npz'
    fit_digest = hashlib.sha256(np.asarray(fit, dtype=np.int64).tobytes()).hexdigest()
    if path.exists():
        with np.load(path) as old:
            if (str(old['fit_digest']) == fit_digest and int(old['seed']) == seed
                    and 'cache_fingerprint' in old and str(old['cache_fingerprint']) == cache_fingerprint()):
                return old['embedding'].copy()
    uid, hist, ids, y = (data[k][fit] for k in ['user', 'hist', 'items', 'label'])
    has = hist >= 0
    h_user = np.broadcast_to(uid[:, None], hist.shape)[has]
    c_user = np.broadcast_to(uid[:, None], ids.shape)[y == 1]
    rows = np.concatenate([h_user, c_user])
    cols = np.concatenate([hist[has], ids[y == 1]])
    x = sparse.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(sizes['all_users'], sizes['all_items'])).tocsr()
    x.data.fill(1)
    row_count = np.asarray(x.sum(axis=1)).ravel()
    col_count = np.asarray(x.sum(axis=0)).ravel()
    x = (sparse.diags(1 / np.sqrt(np.maximum(row_count, 1))) @ x @
         sparse.diags(1 / np.sqrt(np.maximum(col_count, 1)))).tocsr()
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(x @ rng.normal(size=(x.shape[1], dim + 16)).astype(np.float32), mode='reduced')
    for _ in range(3):
        z, _ = np.linalg.qr(x.T @ q, mode='reduced')
        q, _ = np.linalg.qr(x @ z, mode='reduced')
    _, singular, vt = np.linalg.svd(np.asarray(q.T @ x), full_matrices=False)
    embedding = vt[1:dim + 1].T * np.sqrt(singular[1:dim + 1])
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
    embedding = embedding.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as target:
        np.savez(target, embedding=embedding, fit_digest=fit_digest, seed=seed,
                 cache_fingerprint=cache_fingerprint())
    temporary.replace(path)
    log(f'{name}: initialized {dim}-dimensional embeddings using {len(fit):,} fitting events')
    return embedding


def prepare_tensors(data, device):
    """All arrays are inference inputs; only the explicit label key is a target."""
    hist = np.array(data['hist'], dtype=np.int64) + 1  # zero is padding/unknown
    items = np.array(data['items'], dtype=np.int64) + 1
    n = len(items)
    # A few explicit, label-free features required by strategy 1.
    repeat = np.zeros((n, 10, 3), np.float32)
    for k in range(10):
        match = (hist == items[:, k, None]) & (hist > 0)
        count = match.sum(axis=1)
        last = np.max(np.where(match, data['hist_time'], 0), axis=1)
        repeat[:, k, 0] = count > 0
        repeat[:, k, 1] = np.log1p(count)
        age = np.log1p(np.maximum(data['time'] - last, 0) / 86400)
        repeat[:, k, 2] = age * (count > 0)
    hl = np.array(data['hist_len'], dtype=np.float32)
    nonempty = hl > 0
    last_time = data['hist_time'][np.arange(n), np.maximum(hl.astype(int) - 1, 0)]
    age = np.log1p(np.maximum(data['time'] - last_time, 0) / 86400) * nonempty
    event_features = np.column_stack([hl / 20, age, ~nonempty]).astype(np.float32)
    result = {
        'user': torch.tensor(np.array(data['user']) + 1, dtype=torch.long, device=device),
        'items': torch.tensor(items, dtype=torch.long, device=device),
        'hist': torch.tensor(hist, dtype=torch.long, device=device),
        'bucket': torch.tensor(np.array(data['bucket']), dtype=torch.long, device=device),
        'weekend': torch.tensor(np.array(data['weekend']), dtype=torch.long, device=device),
        'repeat': torch.tensor(repeat, device=device),
        'event_features': torch.tensor(event_features, device=device),
    }
    if 'label' in data:
        result['label'] = torch.tensor(np.array(data['label']), dtype=torch.float32, device=device)
    return result


class HistoryFactorization(nn.Module):
    """Pointwise logistic MF with history pooling, biases, and small context terms."""
    def __init__(self, n_users, n_items, dim, user_counts, item_counts, rate,
                 initial=None, user_dropout=.2, embedding_reg=1e-5, user_reg=1e-3):
        super().__init__()
        self.dim, self.user_dropout = dim, user_dropout
        self.embedding_reg, self.user_reg = embedding_reg, user_reg
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.history = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.user = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)
        self.user_bias = nn.Embedding(n_users + 1, 1, padding_idx=0)
        self.item_context = nn.Embedding(n_items + 1, 8, padding_idx=0)
        self.bucket_context = nn.Embedding(4, 8)
        self.day_context = nn.Embedding(2, 8)
        self.bucket_bias = nn.Embedding(4, 1)
        self.day_bias = nn.Embedding(2, 1)
        self.repeat_linear = nn.Linear(3, 1, bias=False)
        self.event_linear = nn.Linear(3, 1, bias=False)
        self.intercept = nn.Parameter(torch.tensor(math.log(rate / (1 - rate)), dtype=torch.float32))
        self.history_scale = nn.Parameter(torch.tensor(10., dtype=torch.float32))
        # Exposure counts contain no outcome statistics. Unknown users are hard masked.
        counts = np.asarray(user_counts, np.float32)
        self.register_buffer('user_gate', torch.tensor(np.r_[0., counts / (counts + 10)], dtype=torch.float32))
        self.register_buffer('known_item', torch.tensor(np.r_[0., np.asarray(item_counts) > 0], dtype=torch.float32))
        with torch.no_grad():
            for layer in [self.item, self.history, self.user, self.item_context,
                          self.bucket_context, self.day_context]:
                layer.weight.normal_(0, .02)
                if layer.padding_idx is not None:
                    layer.weight[0].zero_()
            for layer in [self.item_bias, self.user_bias, self.bucket_bias, self.day_bias]:
                layer.weight.zero_()
            self.repeat_linear.weight.copy_(torch.tensor([[1., .2, 0.]]))
            self.event_linear.weight.zero_()
            if initial is not None:
                self.item.weight[1:].copy_(torch.as_tensor(initial))
                self.history.weight[1:].copy_(torch.as_tensor(initial))

    def forward(self, batch, regularize=False):
        uid, ids, hist = batch['user'], batch['items'], batch['hist']
        item = self.item(ids)
        history = self.history(hist)
        mask = (hist > 0).to(item.dtype)
        pooled = (history * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        gate = self.user_gate[uid]
        if self.training and self.user_dropout:
            gate = gate * (torch.rand_like(gate) >= self.user_dropout)
        user = self.user(uid)
        preference = (item * (user * gate.unsqueeze(-1)).unsqueeze(1)).sum(-1)
        history_score = (item * pooled.unsqueeze(1)).sum(-1)
        context = self.item_context(ids)
        time_context = self.bucket_context(batch['bucket']) + self.day_context(batch['weekend'])
        logits = (self.intercept + self.item_bias(ids).squeeze(-1) * self.known_item[ids]
                  + self.user_bias(uid) * gate.unsqueeze(-1)
                  + preference + self.history_scale * history_score
                  + (context * time_context.unsqueeze(1)).sum(-1)
                  + self.bucket_bias(batch['bucket']) + self.day_bias(batch['weekend'])
                  + self.repeat_linear(batch['repeat']).squeeze(-1)
                  + self.event_linear(batch['event_features']))
        if not regularize:
            return logits
        penalty = (self.embedding_reg * (item.square().sum(-1).mean() + history.square().sum(-1).mean())
                   + self.user_reg * user.square().sum(-1).mean()
                   + 1e-4 * context.square().sum(-1).mean()
                   + 1e-3 * (self.user_bias(uid).square().mean() + self.item_bias(ids).square().mean()))
        return logits, penalty


def batch_at(tensors, indices):
    return {key: value[indices] for key, value in tensors.items()}


@torch.inference_mode()
def predict_logits(model, tensors, indices, batch_size):
    model.eval()
    device = next(model.parameters()).device
    result = []
    for start in range(0, len(indices), batch_size):
        ix = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long, device=device)
        result.append(model(batch_at(tensors, ix)).cpu().numpy())
    return np.concatenate(result)


def model_counts(data, fit, sizes):
    return (np.bincount(data['user'][fit], minlength=sizes['all_users']),
            np.bincount(data['items'][fit].ravel(), minlength=sizes['all_items']))


def fit_model(data, tensors, fit, valid, sizes, dim, name, args, epochs=None, output_dir=None):
    seed_all(args.seed)
    directory = Path(output_dir or OUT)
    directory.mkdir(parents=True, exist_ok=True)
    initial = spectral_initialization(data, fit, sizes, dim, name, args.seed, output_dir=directory)
    counts = model_counts(data, fit, sizes)
    rate = float(data['label'][fit].mean())
    model = HistoryFactorization(sizes['all_users'], sizes['all_items'], dim, *counts, rate,
                                 initial=initial, user_dropout=args.user_dropout,
                                 embedding_reg=args.embedding_reg, user_reg=args.user_reg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    max_epochs = epochs if epochs is not None else args.max_epochs
    best_auc, best_epoch, best_state, stale = -1., 0, None, 0
    history = []
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    fit_tensor = torch.as_tensor(fit.copy(), dtype=torch.long, device=device)
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        order = fit_tensor[torch.tensor(rng.permutation(len(fit)), dtype=torch.long, device=device)]
        total_loss = 0.
        for begin in range(0, len(fit), args.batch_size):
            ix = order[begin:begin + args.batch_size]
            batch = batch_at(tensors, ix)
            optimizer.zero_grad(set_to_none=True)
            logits, penalty = model(batch, regularize=True)
            loss = F.binary_cross_entropy_with_logits(logits, batch['label']) + penalty
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(ix)
        record = {'epoch': epoch, 'train_objective': total_loss / len(fit)}
        if valid is not None:
            logits = predict_logits(model, tensors, valid, args.batch_size)
            record.update(metrics(data['label'][valid], logits))
            current = record['roc_auc']
            if current > best_auc + 1e-5:
                best_auc, best_epoch, stale = current, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        record['seconds'] = round(time.time() - start, 2)
        history.append(record)
        write_json(directory / f'{name}_d{dim}_history.json', history)
        log(f'{name} d={dim}: {record}')
        if valid is not None and epochs is None and stale >= args.patience:
            break
    if valid is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = max_epochs
    checkpoint = {'version': VERSION, 'dim': dim, 'sizes': sizes,
                  'cache_fingerprint': cache_fingerprint(),
                  'user_counts': counts[0].tolist(), 'item_counts': counts[1].tolist(),
                  'rate': rate, 'best_epoch': best_epoch, 'fit_events': len(fit),
                  'max_fit_time': int(data['time'][fit].max()),
                  'model_state': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  'config': {key: value for key, value in vars(args).items()}}
    path = directory / f'{name}_d{dim}.pt'
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return model, {'dim': dim, 'best_epoch': best_epoch,
                   'best_auc': best_auc if valid is not None else None, 'history': history}


def evaluate_slices(data, indices, scores, fit_user_counts):
    y, hl = data['label'][indices], data['hist_len'][indices]
    new = fit_user_counts[data['user'][indices]] == 0
    results = {'all': metrics(y, scores)}
    for label, mask in [('known_user', ~new), ('new_user', new), ('empty_history', hl == 0),
                        ('short_history_1_9', (hl > 0) & (hl < 10)), ('history_10_plus', hl >= 10)]:
        if mask.any():
            results[label] = metrics(y[mask], scores[mask])
            results[label]['events'] = int(mask.sum())
    return results


def bootstrap_auc(y, score, repeats=100, seed=927):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        ix = rng.integers(0, len(y), size=len(y))
        values.append(roc_auc(y[ix], score[ix]))
    return {'method': 'percentile bootstrap of first-event-per-user groups',
            'resamples': repeats, 'lower_95': float(np.quantile(values, .025)),
            'upper_95': float(np.quantile(values, .975))}


def write_submission(logits, filename='submission_strategy1.csv'):
    probabilities = expit(np.asarray(logits, dtype=np.float64)).ravel()
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid probabilities')
    dest = ROOT / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    cached = load_data('evaluation')
    user_ids = json.loads((CACHE / 'user_ids.json').read_text())
    item_ids = json.loads((CACHE / 'item_ids.json').read_text())
    with (ROOT / 'evaluation.csv').open(newline='') as source, dest.open('w', newline='') as target:
        reader, writer = csv.DictReader(source), csv.writer(target)
        if 'clicked' in reader.fieldnames:
            raise ValueError('Unexpected evaluation labels; review evaluation workflow')
        writer.writerow(['rowid', 'predicted_probability'])
        count = 0
        for count, row in enumerate(reader, start=1):
            if count > len(probabilities) or row['rowid'] in seen:
                raise ValueError('Duplicate rowid or prediction count mismatch')
            event, position = divmod(count - 1, 10)
            if (user_ids[row['user_id']] != cached['user'][event]
                    or item_ids[row['item_id']] != cached['items'][event, position]
                    or int(row['timestamp']) != cached['time'][event]):
                raise ValueError('Cached inputs are out of order or stale; rebuild preprocessing')
            seen.add(row['rowid'])
            writer.writerow([row['rowid'], f'{probabilities[count - 1]:.12g}'])
    if count != len(probabilities):
        raise ValueError('Prediction count mismatch')
    # Independently re-read the written CSV and verify every key against evaluation order.
    with dest.open(newline='') as out, (ROOT / 'evaluation.csv').open(newline='') as source:
        output, original = csv.DictReader(out), csv.DictReader(source)
        if output.fieldnames != ['rowid', 'predicted_probability']:
            raise ValueError('Wrong submission header')
        for actual, expected in zip(output, original, strict=True):
            p = float(actual['predicted_probability'])
            if actual['rowid'] != expected['rowid'] or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Submission verification failed')
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {'path': str(dest), 'rows': count, 'unique_rowids': len(seen), 'sha256': digest,
            'min_probability': float(probabilities.min()), 'max_probability': float(probabilities.max()),
            'mean_probability': float(probabilities.mean()), 'evaluation_roc_auc': None,
            'evaluation_auc_reason': 'evaluation.csv has no click labels; official scoring requires competition submission'}


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('cache_fingerprint') != cache_fingerprint():
        raise ValueError('Checkpoint and current encoded data/mappings do not match')
    sizes = checkpoint['sizes']
    model = HistoryFactorization(sizes['all_users'], sizes['all_items'], checkpoint['dim'],
                                 checkpoint['user_counts'], checkpoint['item_counts'], checkpoint['rate'])
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device), checkpoint


def train(args):
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data('train')
    sizes = json.loads((CACHE / 'id_counts.json').read_text())
    tensors = prepare_tensors(data, args.device)
    dev_fit, dev_valid = chronological_split(data, args.dev_cutoff, args.test_cutoff)
    test_fit, test_valid = chronological_split(data, args.test_cutoff)
    assert not np.intersect1d(dev_valid, test_valid).size
    report = {'version': VERSION, 'torch': torch.__version__, 'numpy': np.__version__,
              'config': vars(args), 'development_fit_events': len(dev_fit),
              'development_valid_events': len(dev_valid), 'confirmation_fit_events': len(test_fit),
              'confirmation_valid_events': len(test_valid), 'development_runs': []}
    for dim in args.dims:
        _, result = fit_model(data, tensors, dev_fit, dev_valid, sizes, dim, 'development', args)
        report['development_runs'].append(result)
        write_json(OUT / 'results.json', report)
    best = max(report['development_runs'], key=lambda row: row['best_auc'])
    report['selected'] = {key: best[key] for key in ['dim', 'best_epoch', 'best_auc']}
    # Freeze dimensionality, training length, and all optimization settings before confirmation.
    write_json(OUT / 'selection_locked.json', report['selected'])
    log(f'Locked configuration before January 26 confirmation: {report["selected"]}')
    confirmation, _ = fit_model(data, tensors, test_fit, None, sizes, best['dim'],
                                 'confirmation', args, epochs=best['best_epoch'])
    scores = predict_logits(confirmation, tensors, test_valid, args.batch_size)
    np.savez(OUT / 'confirmation_predictions.npz', event_indices=test_valid,
             labels=np.array(data['label'][test_valid]), logits=scores)
    counts = model_counts(data, test_fit, sizes)[0]
    report['confirmation'] = evaluate_slices(data, test_valid, scores, counts)
    report['confirmation']['auc_interval'] = bootstrap_auc(data['label'][test_valid], scores)
    write_json(OUT / 'results.json', report)
    log(f'Confirmation AUC: {report["confirmation"]["all"]["roc_auc"]:.6f}')
    all_fit = np.arange(len(data['user']))
    final, _ = fit_model(data, tensors, all_fit, None, sizes, best['dim'],
                         'final', args, epochs=best['best_epoch'])
    del tensors
    evaluation = load_data('evaluation')
    eval_tensors = prepare_tensors(evaluation, args.device)
    logits = predict_logits(final, eval_tensors, np.arange(len(evaluation['user'])), args.batch_size)
    np.save(OUT / 'evaluation_logits.npy', logits)
    report['submission'] = write_submission(logits)
    write_json(OUT / 'results.json', report)
    # The same encoding maps used for training must accompany the final checkpoint.
    for name in ['user_ids.json', 'item_ids.json', 'id_counts.json']:
        (OUT / name).write_bytes((CACHE / name).read_bytes())
    log(json.dumps({'confirmation': report['confirmation']['all'], 'submission': report['submission']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps'])
    parser.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--max-epochs', type=int, default=18)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr', type=float, default=.004)
    parser.add_argument('--user-dropout', type=float, default=.2)
    parser.add_argument('--embedding-reg', type=float, default=1e-5)
    parser.add_argument('--user-reg', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--dev-cutoff', default='2025-01-24')
    parser.add_argument('--test-cutoff', default='2025-01-26')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    log(f'PyTorch {torch.__version__}; device={args.device}; dimensions={args.dims}')
    if not (CACHE / 'id_counts.json').exists():
        raise SystemExit('First run: python analysis/profile_data.py')
    if args.command == 'train':
        train(args)
    else:
        if not args.checkpoint:
            raise SystemExit('predict requires --checkpoint artifacts/strategy1/final_dNN.pt')
        model, _ = load_checkpoint(args.checkpoint, args.device)
        data = load_data('evaluation')
        logits = predict_logits(model, prepare_tensors(data, args.device), np.arange(len(data['user'])), args.batch_size)
        log(json.dumps(write_submission(logits)))


if __name__ == '__main__':
    main()
