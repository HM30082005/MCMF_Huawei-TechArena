"""Chronological out-of-time behavioral features + CatBoost for click prediction.

Training rows receive features computed only from previous UTC days. All ten
candidates are excluded together from their outcome-derived features. External
validation/evaluation features remain frozen at the fitting cutoff. Native
categorical CTRs are target-free counters to prevent sibling-event target leaks.
"""
import argparse
from collections import OrderedDict
import gc
import hashlib
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier, FeaturesData, Pool
import catboost
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
import torch

import strategy1 as common

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts' / 'strategy2'
FEATURES = OUT / 'features'
FEATURE_VERSION = 1
SEED = 42
DIM = 128


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + '.tmp')
    common.write_json(tmp, payload)
    tmp.replace(path)


def atomic_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as target:
        np.save(target, array)
    tmp.replace(path)


def identity():
    return {'feature_version': FEATURE_VERSION, 'data_fingerprint': common.cache_fingerprint()}


def rank_within_event(score):
    """Average rank for ties; candidate position carries no preference meaning."""
    less = np.sum(score[:, None, :] < score[:, :, None], axis=2)
    equal = np.sum(score[:, None, :] == score[:, :, None], axis=2) - 1
    return (less + .5 * equal) / (score.shape[1] - 1)


class PastStatistics:
    def __init__(self, data, fit, sizes, cutoff):
        self.cutoff = int(cutoff)
        if len(fit) and int(data['time'][fit].max()) >= cutoff:
            raise ValueError('Fitting statistics include query-period outcomes')
        self.fit_events = len(fit)
        self.max_fit_time = int(data['time'][fit].max()) if len(fit) else None
        nu, ni = sizes['all_users'], sizes['all_items']
        uid, ids, y = (data[key][fit] for key in ['user', 'items', 'label'])
        self.rate = float(y.mean()) if len(fit) else .05
        self.user_n = np.bincount(uid, minlength=nu) * 10
        self.user_y = np.bincount(uid, weights=y.sum(axis=1), minlength=nu)
        self.item_n = np.bincount(ids.ravel(), minlength=ni)
        self.item_y = np.bincount(ids.ravel(), weights=y.ravel(), minlength=ni)
        self.item_p = (self.item_y + 50 * self.rate) / (self.item_n + 50)
        self.user_p = (self.user_y + 50 * self.rate) / (self.user_n + 50)
        self.item_bucket_n = np.bincount((ids * 4 + data['bucket'][fit, None]).ravel(), minlength=ni * 4).reshape(ni, 4)
        iby = np.bincount((ids * 4 + data['bucket'][fit, None]).ravel(), weights=y.ravel(), minlength=ni * 4).reshape(ni, 4)
        self.item_bucket_p = (iby + 100 * self.item_p[:, None]) / (self.item_bucket_n + 100)
        self.user_bucket_n = np.bincount(uid * 4 + data['bucket'][fit], minlength=nu * 4).reshape(nu, 4) * 10
        uby = np.bincount(uid * 4 + data['bucket'][fit], weights=y.sum(axis=1), minlength=nu * 4).reshape(nu, 4)
        self.user_bucket_p = (uby + 100 * self.user_p[:, None]) / (self.user_bucket_n + 100)
        recent = data['time'][fit] >= cutoff - 86400
        self.recent_item_n = np.bincount(ids[recent].ravel(), minlength=ni)
        recent_iy = np.bincount(ids[recent].ravel(), weights=y[recent].ravel(), minlength=ni)
        self.recent_item_p = (recent_iy + 100 * self.item_p) / (self.recent_item_n + 100)
        self.recent_user_n = np.bincount(uid[recent], minlength=nu) * 10
        recent_uy = np.bincount(uid[recent], weights=y[recent].sum(axis=1), minlength=nu)
        self.recent_user_p = (recent_uy + 50 * self.user_p) / (self.recent_user_n + 50)

    def build_similarity(self, data, fit, sizes):
        nu, ni = sizes['all_users'], sizes['all_items']
        if not len(fit):
            self.co = sparse.csr_matrix((ni, ni), dtype=np.float32)
            self.item_users = np.zeros(ni, np.float32)
            return
        uid, ids, h, y = (data[key][fit] for key in ['user', 'items', 'hist', 'label'])
        has = h >= 0
        rows = np.concatenate([np.broadcast_to(uid[:, None], h.shape)[has],
                               np.broadcast_to(uid[:, None], ids.shape)[y == 1]])
        cols = np.concatenate([h[has], ids[y == 1]])
        interactions = sparse.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(nu, ni)).tocsr()
        interactions.data.fill(1)
        self.item_users = np.asarray(interactions.sum(axis=0)).ravel()
        co = (interactions.T @ interactions).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        norm = sparse.diags(1 / np.sqrt(np.maximum(self.item_users, 1)))
        self.co = (norm @ co @ norm).tocsr()


def fit_embedding(data, tensors, fit, sizes, cutoff, args):
    """Fixed strategy-1 recipe, fitted solely before the current feature cutoff."""
    if not len(fit):
        return None, np.zeros((sizes['all_items'], DIM), np.float32), None
    tag = 'past_' + str(int(cutoff))
    candidates = {
        common.utc_stamp('2025-01-24'): ROOT / 'artifacts/strategy1/development_d128.pt',
        common.utc_stamp('2025-01-26'): ROOT / 'artifacts/strategy1/confirmation_d128.pt',
        common.utc_stamp('2025-01-27'): ROOT / 'artifacts/strategy1/final_d128.pt',
    }
    existing = candidates.get(int(cutoff))
    path = existing if existing is not None and existing.exists() else OUT / 'embeddings' / f'{tag}_d128.pt'
    if path.exists():
        model, record = common.load_checkpoint(path, args.device)
        if (record['fit_events'] != len(fit) or record['max_fit_time'] >= cutoff
                or record['best_epoch'] != 2 or record['dim'] != DIM):
            raise ValueError('Embedding checkpoint does not match the strictly earlier fitting window')
    else:
        recipe = argparse.Namespace(seed=SEED, lr=.004, user_dropout=.2,
                    embedding_reg=1e-5, user_reg=1e-3, device=args.device,
                    batch_size=4096, max_epochs=2, patience=3)
        model, _ = common.fit_model(data, tensors, fit, None, sizes, DIM, tag,
                                     recipe, epochs=2, output_dir=OUT / 'embeddings')
        record = {'fit_events': len(fit), 'max_fit_time': int(data['time'][fit].max())}
    initial = common.spectral_initialization(data, fit, sizes, DIM, tag, SEED,
                                               output_dir=OUT / 'embeddings')
    provenance = {'checkpoint': str(path), 'fit_events': len(fit),
                  'max_fit_time': record['max_fit_time'], 'cutoff': int(cutoff), 'epochs': 2, 'dim': DIM}
    return model, initial, provenance


def engineered_block(data, indices, stats, initial, learned_tables, neural_logits):
    """Feature inputs exclude the query data's label field entirely."""
    ids, uid, h, ht, hl, ts = (np.asarray(data[k][indices]) for k in
                               ['items', 'user', 'hist', 'hist_time', 'hist_len', 'time'])
    if np.any(ts < stats.cutoff):
        raise ValueError('Query precedes the feature cutoff')
    n, count = ids.shape
    present = h >= 0
    safe_h = np.maximum(h, 0)
    denom = np.maximum(hl, 1)
    features = OrderedDict()

    def add(name, value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 1:
            value = value[:, None]
        features[name] = np.broadcast_to(value, (n, count))

    def relative(name, score):
        add(name + '_centered', score - score.mean(axis=1, keepdims=True))
        add(name + '_rank', rank_within_event(score))
        add(name + '_event_mean', score.mean(axis=1))
        add(name + '_event_max', score.max(axis=1))
        add(name + '_event_std', score.std(axis=1))

    add('has_past_fit', np.full(n, stats.fit_events > 0))
    add('global_ctr', np.full(n, stats.rate))
    add('item_ctr', stats.item_p[ids])
    add('user_ctr', stats.user_p[uid])
    add('item_impressions_log', np.log1p(stats.item_n[ids]))
    add('user_impressions_log', np.log1p(stats.user_n[uid]))
    add('item_clicks_log', np.log1p(stats.item_y[ids]))
    add('user_clicks_log', np.log1p(stats.user_y[uid]))
    add('known_user', stats.user_n[uid] > 0)
    add('known_item', stats.item_n[ids] > 0)
    bucket = np.asarray(data['bucket'][indices])
    add('item_bucket_ctr', stats.item_bucket_p[ids, bucket[:, None]])
    add('user_bucket_ctr', stats.user_bucket_p[uid, bucket])
    add('item_bucket_impressions_log', np.log1p(stats.item_bucket_n[ids, bucket[:, None]]))
    add('user_bucket_impressions_log', np.log1p(stats.user_bucket_n[uid, bucket]))
    add('recent_training_item_ctr', stats.recent_item_p[ids])
    add('recent_training_user_ctr', stats.recent_user_p[uid])
    add('recent_training_item_impressions_log', np.log1p(stats.recent_item_n[ids]))
    add('recent_training_user_impressions_log', np.log1p(stats.recent_user_n[uid]))
    add('time_bucket', bucket)
    add('weekend', np.asarray(data['weekend'][indices]))
    add('utc_hour', (ts % 86400) / 3600)
    add('history_length', hl)
    add('empty_history', hl == 0)
    sorted_h = np.sort(h, axis=1)
    unique = (sorted_h[:, 0] >= 0).astype(int) + np.sum((sorted_h[:, 1:] >= 0) & (np.diff(sorted_h, axis=1) != 0), axis=1)
    add('history_distinct_items', unique)
    add('history_repeat_fraction', (hl - unique) / denom)
    last_time = ht[np.arange(n), np.maximum(hl - 1, 0)]
    last_age = np.log1p(np.maximum(ts - last_time, 0) / 3600)
    add('last_click_age_log_hours', np.where(hl > 0, last_age, -1))
    span = np.maximum(last_time - ht[:, 0], 0)
    add('history_span_log_hours', np.where(hl > 1, np.log1p(span / 3600), 0))
    gaps = np.diff(ht, axis=1)
    gap_mask = np.arange(19)[None, :] < (hl[:, None] - 1)
    gap_mean = np.sum(np.where(gap_mask, gaps, 0), axis=1) / np.maximum(hl - 1, 1)
    add('history_mean_gap_log_hours', np.log1p(np.maximum(gap_mean, 0) / 3600))
    repeats = np.zeros((n, count), np.float32)
    repeat_age = np.full((n, count), -1., np.float32)
    for j in range(count):
        matches = (h == ids[:, j, None]) & present
        repeats[:, j] = matches.sum(axis=1)
        last = np.where(matches, ht, 0).max(axis=1)
        repeat_age[:, j] = np.where(repeats[:, j] > 0, np.log1p(np.maximum(ts - last, 0) / 3600), -1)
    add('candidate_repeat', repeats > 0)
    add('candidate_repeat_count', repeats)
    add('candidate_repeat_age_log_hours', repeat_age)
    add('event_repeat_candidates', (repeats > 0).sum(axis=1))
    add('history_item_ctr_mean', np.sum(stats.item_p[safe_h] * present, axis=1) / denom)
    add('history_item_ctr_max', np.where(present, stats.item_p[safe_h], 0).max(axis=1))
    prior = np.log(stats.item_p[ids] / (1-stats.item_p[ids])) + np.log(stats.user_p[uid, None] / (1-stats.user_p[uid, None])) - np.log(stats.rate/(1-stats.rate))
    add('prior_logit', prior)
    add('neural_logit', neural_logits)
    relative('neural', neural_logits)

    # Candidate-history similarities use fitting-only vectors and no outcome of this event.
    for prefix, item_table, hist_table in [
            ('spectral', initial, initial),
            ('learned', *learned_tables)]:
        cand = item_table[ids]
        history = hist_table[safe_h] * present[:, :, None]
        cand_norm = cand / np.maximum(np.linalg.norm(cand, axis=2, keepdims=True), 1e-8)
        hist_norm = history / np.maximum(np.linalg.norm(history, axis=2, keepdims=True), 1e-8)
        pairs = np.einsum('nkd,njd->nkj', hist_norm, cand_norm, optimize=True)
        mean = pairs.sum(axis=1) / denom[:, None]
        maximum = np.where(present[:, :, None], pairs, -1e9).max(axis=1)
        maximum[hl == 0] = 0
        last = pairs[np.arange(n), np.maximum(hl - 1, 0)]
        last[hl == 0] = 0
        pooled = history.sum(axis=1) / denom[:, None]
        raw = np.einsum('njd,nd->nj', cand, pooled)
        add(prefix + '_mean_cosine', mean)
        add(prefix + '_max_cosine', maximum)
        add(prefix + '_last_cosine', last)
        add(prefix + '_raw_pooled_dot', raw)
        if prefix == 'spectral':
            relative(prefix, mean)
        else:
            add('candidate_vector_norm', np.linalg.norm(cand, axis=2))
            add('history_vector_norm', np.linalg.norm(pooled, axis=1))
    cf_sum, cf_max = np.zeros((n, count), np.float32), np.zeros((n, count), np.float32)
    for k in range(h.shape[1]):
        values = stats.co[ids, safe_h[:, k, None]]
        values = (values.toarray() if sparse.issparse(values) else np.asarray(values)) * present[:, k, None]
        cf_sum += values
        np.maximum(cf_max, values, out=cf_max)
    add('cf_cosine_sum', cf_sum)
    add('cf_cosine_mean', cf_sum / denom[:, None])
    add('cf_cosine_max', cf_max)
    add('item_interacting_users_log', np.log1p(stats.item_users[ids]))
    add('starter_logit', prior + 25 * cf_sum)
    relative('cf', cf_sum)
    matrix = np.stack(list(features.values()), axis=2).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError('Non-finite engineered features')
    return matrix, list(features)


def feature_file(name):
    return FEATURES / (name + '.npy')


def cached_features(name, query, query_indices, cutoff, train, train_tensors, query_tensors, sizes, args):
    path = feature_file(name)
    meta_path = path.with_suffix('.json')
    expected = dict(identity(), cutoff=int(cutoff),
                    query_indices_hash=hashlib.sha256(np.asarray(query_indices, np.int64).tobytes()).hexdigest(),
                    query_events=len(query_indices))
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in expected.items()):
            common.log('Resuming cached features: ' + name)
            return path, meta
    fit = np.flatnonzero(train['time'] < cutoff)
    common.log(f'Building {name}: {len(query_indices):,} query events, {len(fit):,} earlier fitting events')
    stats = PastStatistics(train, fit, sizes, cutoff)
    stats.build_similarity(train, fit, sizes)
    model, initial, provenance = fit_embedding(train, train_tensors, fit, sizes, cutoff, args)
    if model is not None:
        logits = common.predict_logits(model, query_tensors, query_indices, 4096)
        # Transfer these tables once per cutoff, rather than once per feature block.
        learned_tables = (model.item.weight.detach().cpu().numpy()[1:],
                          model.history.weight.detach().cpu().numpy()[1:])
    else:
        logits = np.full((len(query_indices), 10), np.log(.05/.95), np.float32)
        learned_tables = (initial, initial)
    tmp_path = path.with_name(path.name + '.tmp')
    output = None
    names = None
    for start in range(0, len(query_indices), args.feature_batch):
        indices = query_indices[start:start + args.feature_batch]
        values, columns = engineered_block(query, indices, stats, initial, learned_tables,
                                            logits[start:start + len(indices)])
        if output is None:
            names = columns
            output = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=np.float32,
                    shape=(len(query_indices), 10, len(columns)))
        assert columns == names
        output[start:start + len(indices)] = values
    output.flush()
    del output
    tmp_path.replace(path)
    meta = dict(expected, feature_names=names, max_fit_time=stats.max_fit_time,
                embedding=provenance, shape=[len(query_indices), 10, len(names)])
    atomic_json(meta_path, meta)
    common.log(f'Saved {name}: {len(names)} features')
    del stats, model, initial, learned_tables
    gc.collect()
    if args.device == 'mps':
        torch.mps.empty_cache()
    return path, meta


def prepare(args):
    FEATURES.mkdir(parents=True, exist_ok=True)
    train, evaluation = common.load_data('train'), common.load_data('evaluation')
    sizes = json.loads((common.CACHE / 'id_counts.json').read_text())
    tensors = common.prepare_tensors(train, args.device)
    daily = []
    for day in np.unique(train['time'] // 86400):
        ix = np.flatnonzero(train['time'] // 86400 == day)
        path, meta = cached_features('day_' + str(int(day)), train, ix, int(day) * 86400,
                                      train, tensors, tensors, sizes, args)
        daily.append((path, meta, ix))
    names = daily[0][1]['feature_names']
    combined = FEATURES / 'train.npy'
    manifest = FEATURES / 'train.json'
    if not combined.exists() or not manifest.exists() or json.loads(manifest.read_text()).get('identity') != identity():
        tmp = combined.with_name('train.npy.tmp')
        dest = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32,
                    shape=(len(train['time']), 10, len(names)))
        for path, meta, ix in daily:
            assert meta['feature_names'] == names
            dest[ix] = np.load(path, mmap_mode='r')
        dest.flush()
        del dest
        tmp.replace(combined)
        atomic_json(manifest, {'identity': identity(), 'feature_names': names,
                              'daily_files': [str(x[0]) for x in daily]})
    _, dev_valid = common.chronological_split(train, '2025-01-24', '2025-01-26')
    _, test_valid = common.chronological_split(train, '2025-01-26')
    # Both Jan 24 and Jan 25 development events use only fitting data before Jan 24.
    cached_features('development_valid', train, dev_valid, common.utc_stamp('2025-01-24'),
                    train, tensors, tensors, sizes, args)
    cached_features('confirmation_valid', train, test_valid, common.utc_stamp('2025-01-26'),
                    train, tensors, tensors, sizes, args)
    eval_tensors = common.prepare_tensors(evaluation, args.device)
    cached_features('evaluation', evaluation, np.arange(len(evaluation['time'])), common.utc_stamp('2025-01-27'),
                    train, tensors, eval_tensors, sizes, args)
    common.log('All chronological feature caches complete')


def make_pool(features, data, indices, names, quantize=True, threads=12):
    numeric = np.asarray(features).reshape(-1, len(names))
    if len(numeric) != len(indices) * 10:
        raise ValueError('Feature rows do not match event indices')
    users = np.asarray(data['user'][indices])
    ids = np.asarray(data['items'][indices])
    user_keys = np.asarray([f'u{x}'.encode() for x in range(int(data['user'].max()) + 1)], dtype=object)
    item_keys = np.asarray([f'i{x}'.encode() for x in range(int(data['items'].max()) + 1)], dtype=object)
    cats = np.empty((len(numeric), 2), dtype=object)
    cats[:, 0] = user_keys[np.repeat(users, 10)]
    cats[:, 1] = item_keys[ids.ravel()]
    label = np.asarray(data['label'][indices]).ravel() if 'label' in data else None
    fd = FeaturesData(num_feature_data=numeric, cat_feature_data=cats,
                      num_feature_names=names, cat_feature_names=['user_id', 'item_id'])
    pool = Pool(fd, label=label, thread_count=threads)
    if quantize:
        pool.quantize(border_count=64)
    return pool


def tree_params(depth, iterations, args):
    return dict(iterations=iterations, depth=depth, learning_rate=.08,
                loss_function='Logloss', eval_metric='AUC', random_seed=SEED,
                thread_count=args.threads, task_type='CPU', has_time=True,
                boosting_type='Plain', l2_leaf_reg=20, random_strength=1.,
                bootstrap_type='Bernoulli', subsample=.8, rsm=.85,
                border_count=64, one_hot_max_size=4, max_ctr_complexity=1,
                simple_ctr=['Counter:Prior=0'], combinations_ctr=['Counter:Prior=0'],
                counter_calc_method='SkipTest', allow_writing_files=True,
                train_dir=str(OUT / 'catboost_info'), verbose=25)


def fit_tree(pool, valid_pool, depth, iterations, stage, args):
    path = OUT / f'{stage}_depth{depth}.cbm'
    meta_path = path.with_suffix('.json')
    params = tree_params(depth, iterations, args)
    key = dict(identity(), depth=depth, iterations=iterations, stage=stage,
               learning_rate=.08, target_free_categorical_counters=True)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('key') == key:
            model = CatBoostClassifier()
            model.load_model(path)
            common.log('Resuming saved tree model: ' + str(path))
            return model, meta
    model = CatBoostClassifier(**params)
    start = time.time()
    model.fit(pool, eval_set=valid_pool, use_best_model=valid_pool is not None,
              early_stopping_rounds=50 if valid_pool is not None else None,
              save_snapshot=True, snapshot_file=str(OUT / f'{stage}_depth{depth}.snapshot'),
              snapshot_interval=30)
    temporary = path.with_name(path.name + '.tmp')
    model.save_model(temporary, format='cbm')
    temporary.replace(path)
    meta = {'key': key, 'trees': model.tree_count_, 'seconds': time.time()-start,
            'params': model.get_all_params(), 'path': str(path)}
    if valid_pool is not None:
        meta['development_auc'] = float(model.get_best_score()['validation']['AUC'])
    atomic_json(meta_path, meta)
    return model, meta


def train(args):
    train_data = common.load_data('train')
    manifest = json.loads((FEATURES / 'train.json').read_text())
    if manifest['identity'] != identity():
        raise ValueError('Feature cache does not match the current encoded data; run prepare first')
    names = manifest['feature_names']
    full = np.load(FEATURES / 'train.npy', mmap_mode='r')
    if full.shape != (len(train_data['time']), 10, len(names)):
        raise ValueError('Feature cache has an unexpected shape')
    for tag in ['development_valid', 'confirmation_valid', 'evaluation']:
        meta = json.loads(feature_file(tag).with_suffix('.json').read_text())
        if any(meta.get(k) != v for k, v in identity().items()) or meta['feature_names'] != names:
            raise ValueError('Inconsistent validation/evaluation feature cache; run prepare first')
    dev_fit, dev_valid = common.chronological_split(train_data, '2025-01-24', '2025-01-26')
    test_fit, test_valid = common.chronological_split(train_data, '2025-01-26')
    np.testing.assert_array_equal(dev_fit, np.arange(len(dev_fit)))
    np.testing.assert_array_equal(test_fit, np.arange(len(test_fit)))
    reference = ROOT / 'artifacts/strategy1/confirmation_predictions.npz'
    if reference.exists():
        with np.load(reference) as ref:
            np.testing.assert_array_equal(test_valid, ref['event_indices'])
            np.testing.assert_array_equal(train_data['label'][test_valid], ref['labels'])
    report = {'identity': identity(), 'catboost': catboost.__version__, 'feature_names': names,
              'config': vars(args), 'development': [], 'development_fit_events': len(dev_fit),
              'development_valid_events': len(dev_valid), 'confirmation_fit_events': len(test_fit),
              'confirmation_valid_events': len(test_valid)}
    common.log('Quantizing chronological development fitting pool')
    fit_pool = make_pool(full[:len(dev_fit)], train_data, dev_fit, names, threads=args.threads)
    valid_pool = make_pool(np.load(feature_file('development_valid'), mmap_mode='r'),
                             train_data, dev_valid, names, quantize=False, threads=args.threads)
    for depth in args.depths:
        model, meta = fit_tree(fit_pool, valid_pool, depth, args.iterations, 'development', args)
        report['development'].append(meta)
        atomic_json(OUT / 'results.json', report)
    best = max(report['development'], key=lambda row: row['development_auc'])
    report['selected'] = {'depth': best['key']['depth'], 'trees': best['trees'],
                          'development_auc': best['development_auc']}
    atomic_json(OUT / 'selection_locked.json', report['selected'])
    common.log('Locked before January 26 confirmation: ' + json.dumps(report['selected']))
    del fit_pool, valid_pool, model
    gc.collect()
    fit_pool = make_pool(full[:len(test_fit)], train_data, test_fit, names, threads=args.threads)
    query_pool = make_pool(np.load(feature_file('confirmation_valid'), mmap_mode='r'),
                            train_data, test_valid, names, quantize=False, threads=args.threads)
    model, meta = fit_tree(fit_pool, None, report['selected']['depth'], report['selected']['trees'], 'confirmation', args)
    logits = model.predict(query_pool, prediction_type='RawFormulaVal', thread_count=args.threads).reshape(-1, 10)
    labels = np.asarray(train_data['label'][test_valid])
    assert abs(common.roc_auc(labels, logits) - roc_auc_score(labels.ravel(), logits.ravel())) < 1e-12
    np.savez(OUT / 'confirmation_predictions.npz', event_indices=test_valid, labels=labels, logits=logits)
    nu = int(train_data['user'].max()) + 1
    counts = np.bincount(train_data['user'][test_fit], minlength=nu)
    report['confirmation'] = common.evaluate_slices(train_data, test_valid, logits, counts)
    report['confirmation']['auc_interval'] = common.bootstrap_auc(labels, logits)
    loaded = CatBoostClassifier()
    loaded.load_model(meta['path'])
    np.testing.assert_allclose(logits.ravel(), loaded.predict(query_pool, prediction_type='RawFormulaVal'), rtol=0, atol=1e-12)
    report['confirmation']['checkpoint_reload_verified'] = True
    atomic_json(OUT / 'results.json', report)
    common.log('STRATEGY 2 CONFIRMATION: ' + json.dumps(report['confirmation']['all']))
    del fit_pool, query_pool, model, loaded
    gc.collect()
    all_indices = np.arange(len(train_data['time']))
    fit_pool = make_pool(full, train_data, all_indices, names, threads=args.threads)
    model, meta = fit_tree(fit_pool, None, report['selected']['depth'], report['selected']['trees'], 'final', args)
    evaluation = common.load_data('evaluation')
    query_pool = make_pool(np.load(feature_file('evaluation'), mmap_mode='r'), evaluation,
                             np.arange(len(evaluation['time'])), names, quantize=False, threads=args.threads)
    logits = model.predict(query_pool, prediction_type='RawFormulaVal', thread_count=args.threads).reshape(-1, 10)
    np.save(OUT / 'evaluation_logits.npy', logits)
    report['submission'] = common.write_submission(logits, filename='artifacts/strategy2/submission_strategy2.csv')
    importance = model.get_feature_importance()
    report['feature_importance'] = sorted([{'feature': n, 'importance': float(v)} for n, v in
                                           zip(names + ['user_id', 'item_id'], importance)],
                                          key=lambda x:x['importance'], reverse=True)
    report['final_model'] = meta['path']
    atomic_json(OUT / 'results.json', report)
    common.log('STRATEGY 2 COMPLETE: ' + json.dumps({'holdout': report['confirmation']['all'], 'submission': report['submission']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'all'], default='all', nargs='?')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--feature-batch', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--depths', type=int, nargs='+', default=[6, 8])
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    if args.command in ['prepare', 'all']:
        prepare(args)
    if args.command in ['train', 'all']:
        train(args)


if __name__ == '__main__':
    main()
