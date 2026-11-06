"""History-only shared-item MF, historical-click loss, seed averaging, and trees.

Numerical training and data preparation live here. LightGBM runs in a separate
process to avoid loading its OpenMP runtime together with PyTorch on macOS.
"""
import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F

import strategy1 as common
from strategy2 import atomic_json, atomic_npy, rank_within_event

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'artifacts' / 'history_upgrade'
OUT = DEFAULT_OUT
VERSION = 1
START = common.utc_stamp('2025-01-20')
DAY = 86400
FIRST_DAY = 2
PART_NAMES = ['mf_logit', 'mf_item_bias', 'mf_user_bias', 'mf_history_dot']
TREE_CANDIDATES = 'CatBoost:6,CatBoost:8,LightGBM:31,LightGBM:63'
# Baseline MF settings. Defaults keep recipe identities, and so saved checkpoints, unchanged.
MF_DEFAULTS = dict(dim=32, l2=.05, lr=.003, aux_weight=.1, negatives=9, history_limit=100, init_std=.05)


def setting(args, name):
    return getattr(args, name, MF_DEFAULTS[name])


def recipe(args):
    historical_inputs = 'fitting histories and labels only; own query history at inference'
    if getattr(args, 'query_clicks', False):
        historical_inputs += "; auxiliary loss also uses pre-cutoff clicks from scored queries' histories"
    result = dict(version=VERSION, dim=setting(args, 'dim'), l2=setting(args, 'l2'), lr=setting(args, 'lr'),
                  auxiliary_weight=setting(args, 'aux_weight'), negatives=setting(args, 'negatives'),
                  history_limit=setting(args, 'history_limit'), init_std=setting(args, 'init_std'),
                  seeds=args.seeds, steps=args.steps, batch_events=args.batch_events,
                  event_sampling='independent uniform with replacement per seed',
                  historical_inputs=historical_inputs)
    if getattr(args, 'steps_per_seed', None):
        result['steps_per_seed'] = list(args.steps_per_seed)
    for name in ('stale_context_hours', 'stale_history_hours'):
        if getattr(args, name, 0):
            result[name] = getattr(args, name)
    return result


def mf_identity(args):
    """Everything the MF models and their score components depend on."""
    return {'data_fingerprint': common.cache_fingerprint(), 'recipe': recipe(args)}


def run_identity(args):
    """MF identity plus settings that only change the tree features."""
    identity = mf_identity(args)
    subset = getattr(args, 'feature_subset', 'baseline')
    return dict(identity, feature_subset=subset) if subset != 'baseline' else identity


def write_recipe(args):
    """Refuse a directory holding another recipe before anything in it is modified."""
    path, identity = OUT/'recipe.json', run_identity(args)
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f'{OUT} holds a different recipe; use a new --output-dir')
    atomic_json(path, identity)


def unique_clicks(data, fit):
    """One record per (user, timestamp, item); different repeat times are kept."""
    u, h, ht = (np.asarray(data[k][fit]) for k in ['user', 'hist', 'hist_time'])
    valid = h >= 0
    rows, pos = np.nonzero(data['label'][fit])
    count = int(valid.sum()) + len(rows)
    records = np.empty(count, dtype=[('user', 'i4'), ('time', 'i8'), ('item', 'i4')])
    split = int(valid.sum())
    records['user'][:split] = np.broadcast_to(u[:, None], h.shape)[valid]
    records['time'][:split] = ht[valid]
    records['item'][:split] = h[valid] + 1
    records['user'][split:] = u[rows]
    records['time'][split:] = data['time'][fit][rows]
    records['item'][split:] = data['items'][fit][rows, pos] + 1
    return np.unique(records)


def scored_query_clicks(data, day):
    """Pre-cutoff clicks in the supplied histories of the queries this cutoff's model scores:
    the evaluation file on day 7, the development or confirmation window on days 4 and 6, and
    each user's first event of the day otherwise. Histories hold only clicks before their own
    event and are filtered to the cutoff here; no outcome of a scored row is read."""
    cutoff = START + day * DAY
    if day == 7:
        query = common.load_data('evaluation')
        ix = np.arange(len(query['time']))
    else:
        query = data
        if day == 4:
            _, ix = common.chronological_split(data, '2025-01-24', '2025-01-26')
        elif day == 6:
            _, ix = common.chronological_split(data, '2025-01-26')
        else:
            on_day = np.flatnonzero((data['time'] >= cutoff) & (data['time'] < cutoff + DAY))
            _, first = np.unique(np.asarray(data['user'][on_day]), return_index=True)
            ix = np.sort(on_day[first])
    h, ht, u = (np.asarray(query[k][ix]) for k in ['hist', 'hist_time', 'user'])
    valid = (h >= 0) & (ht < cutoff)
    records = np.empty(int(valid.sum()), dtype=[('user', 'i4'), ('time', 'i8'), ('item', 'i4')])
    records['user'] = np.broadcast_to(u[:, None], h.shape)[valid]
    records['time'] = ht[valid]
    records['item'] = h[valid] + 1
    return np.unique(records)


def long_history(data, indices, clicks, n_items, limit=100, gap=0):
    """Merge fitting clicks and the event's own history, strictly before its time minus gap
    seconds. A training gap mimics the older histories of the scored population."""
    uid, ts = np.asarray(data['user'][indices]), np.asarray(data['time'][indices])
    key = clicks['user'].astype(np.int64) * 10_000_000_000 + clicks['time']
    starts = np.searchsorted(key, uid.astype(np.int64) * 10_000_000_000, side='left')
    ends = np.searchsorted(key, uid.astype(np.int64) * 10_000_000_000 + ts - gap, side='left')
    ix = ends[:, None] - 1 - np.arange(limit)[None, :]
    valid = ix >= starts[:, None]
    if len(clicks):
        safe = np.maximum(ix, 0)
        items = np.where(valid, clicks['item'][safe], 0)
        times = np.where(valid, clicks['time'][safe], 0)
    else:
        items = np.zeros(ix.shape, np.int32)
        times = np.zeros(ix.shape, np.int64)
    h, ht = np.asarray(data['hist'][indices]), np.asarray(data['hist_time'][indices])
    if np.any((h >= 0) & (ht >= ts[:, None])):
        raise ValueError('An event history contains a current/future click')
    own = (h >= 0) & (ht < ts[:, None] - gap)
    own_items, own_times = np.where(own, h + 1, 0), np.where(own, ht, 0)
    codes = np.concatenate([times * n_items + items, own_times * n_items + own_items], axis=1)
    codes.sort(axis=1)
    duplicates = np.concatenate([np.zeros((len(indices), 1), bool), codes[:, 1:] == codes[:, :-1]], axis=1)
    codes[duplicates] = 0
    codes.sort(axis=1)
    codes = codes[:, -limit:][:, ::-1]
    return (codes % n_items).astype(np.int32), (codes > 0).sum(axis=1).astype(np.int16)


def other_click_context(clicks, limit=100, gap=0):
    """Other clicks of the same user; omit the exact target occurrence, and with a gap also
    every click within gap seconds of it, as if the user's history stopped before the target."""
    users = clicks['user']
    if not len(users):
        return np.zeros((0, limit), np.int32), np.zeros(0, np.int16), np.zeros(0, np.int32)
    boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    counts = np.diff(boundaries)
    starts = np.repeat(boundaries[:-1], counts)
    ends = np.repeat(boundaries[1:], counts)
    # Clicks are sorted by user and time, so the omitted clicks are one block [low, high).
    low, high = np.arange(len(users)), np.arange(len(users)) + 1
    if gap:
        key = users.astype(np.int64) * 10_000_000_000 + clicks['time']
        low, high = np.searchsorted(key, key - gap + 1, side='left'), np.searchsorted(key, key + gap, side='left')
    lengths = np.minimum(ends - starts - (high - low), limit)
    result = np.zeros((len(users), limit), np.int32)
    # Column blocks avoid an additional N x 100 int64 temporary allocation.
    for j in range(limit):
        idx = ends - 1 - j
        idx -= (idx < high) * (high - low)
        valid = idx >= starts
        result[valid, j] = clicks['item'][idx[valid]]
    has = lengths > 0
    return result[has], lengths[has].astype(np.int16), clicks['item'][has].copy()


class SharedHistoryEnsemble(nn.Module):
    """Independent seeds batched on the device; each has one item table and biases."""
    def __init__(self, n_users, n_items, seeds=10, dim=32, init_std=.05):
        super().__init__()
        self.n_users, self.n_items, self.seeds, self.dim = n_users, n_items, seeds, dim
        self.item = nn.Embedding(seeds * n_items, dim)
        self.item_bias = nn.Embedding(seeds * n_items, 1)
        self.user_bias = nn.Embedding(seeds * n_users, 1)
        self.intercept = nn.Parameter(torch.full((seeds,), math.log(.063 / .937)))
        self.register_buffer('seed_index', torch.arange(seeds))
        with torch.no_grad():
            self.item_bias.weight.zero_()
            self.user_bias.weight.zero_()
            for seed in range(seeds):
                torch.manual_seed(seed)
                self.item.weight[seed*n_items:(seed+1)*n_items].normal_(0, init_std)
                self.item.weight[seed*n_items].zero_()

    def seed_state(self, seed):
        n, u = self.n_items, self.n_users
        return (self.item.weight[seed*n:(seed+1)*n].detach().clone(),
                self.item_bias.weight[seed*n:(seed+1)*n].detach().clone(),
                self.user_bias.weight[seed*u:(seed+1)*u].detach().clone(),
                self.intercept[seed].detach().clone())

    @torch.no_grad()
    def load_seed_state(self, seed, state):
        n, u = self.n_items, self.n_users
        self.item.weight[seed*n:(seed+1)*n] = state[0]
        self.item_bias.weight[seed*n:(seed+1)*n] = state[1]
        self.user_bias.weight[seed*u:(seed+1)*u] = state[2]
        self.intercept[seed] = state[3]

    def lookup(self, table, indices, stride):
        offsets = self.seed_index.reshape((-1,) + (1,) * (indices.ndim-1)) * stride
        return table(indices + offsets)

    def pooled(self, hist, lengths):
        vectors = self.lookup(self.item, hist, self.n_items)
        return (vectors * (hist > 0).unsqueeze(-1)).sum(-2) / lengths.clamp(min=1).unsqueeze(-1)

    def forward(self, users, candidates, hist, lengths, regularize=False):
        pooled = self.pooled(hist, lengths)
        vectors = self.lookup(self.item, candidates, self.n_items)
        ib = self.lookup(self.item_bias, candidates, self.n_items).squeeze(-1)
        ub = self.lookup(self.user_bias, users, self.n_users).expand_as(ib)
        dot = (pooled.unsqueeze(-2) * vectors).sum(-1)
        score = self.intercept[:, None, None] + ib + ub + dot
        components = torch.stack([score, ib, ub, dot], dim=-1)
        if not regularize:
            return components
        count = candidates.shape[-1]
        reg = vectors.square().sum((1, 2, 3)) / count + pooled.square().sum((1, 2))
        return components, reg / (candidates.shape[1] * count)

    def auxiliary(self, context, lengths, target, negatives):
        candidates = torch.cat([target.unsqueeze(-1), negatives], dim=-1)
        pooled = self.pooled(context, lengths)
        vectors = self.lookup(self.item, candidates, self.n_items)
        score = self.lookup(self.item_bias, candidates, self.n_items).squeeze(-1)
        score = score + (pooled.unsqueeze(-2) * vectors).sum(-1)
        mask = torch.cat([torch.zeros_like(target.unsqueeze(-1), dtype=torch.bool),
                          negatives == target.unsqueeze(-1)], dim=-1)
        score = score.masked_fill(mask, -1e9)
        loss = -F.log_softmax(score, dim=-1)[..., 0].mean(dim=1)
        count = candidates.shape[-1]
        reg = vectors.square().sum((1, 2, 3)) / count + pooled.square().sum((1, 2))
        return loss, reg / (candidates.shape[1] * count)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def atomic_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def fit_ensemble(data, day, sizes, args):
    cutoff = START + day * DAY
    fit = np.flatnonzero(data['time'] < cutoff)
    identity = dict(mf_identity(args), cutoff=cutoff, fitting_events=len(fit))
    path = OUT / 'models' / f'day{day}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True) if path.exists() else None
    if checkpoint is not None and checkpoint['identity'] != identity:
        raise ValueError('Checkpoint recipe/data changed; use a separate output directory')
    model = SharedHistoryEnsemble(sizes['all_users'], sizes['all_items']+1, args.seeds,
                                  dim=setting(args, 'dim'), init_std=setting(args, 'init_std')).to(args.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        common.log(f'Reusing completed day-{day} ensemble')
        return model, checkpoint['provenance']
    clicks = unique_clicks(data, fit)
    if len(clicks) and clicks['time'].max() >= cutoff:
        raise ValueError('Fitting click log crosses the model cutoff')
    n_items = sizes['all_items']+1
    limit, l2, negatives = setting(args, 'history_limit'), setting(args, 'l2'), setting(args, 'negatives')
    per_seed = getattr(args, 'steps_per_seed', None)
    steps = max(per_seed) if per_seed else args.steps
    hist, lengths = long_history(data, fit, clicks, n_items, limit,
                                 gap=int(getattr(args, 'stale_history_hours', 0)*3600))
    auxiliary_clicks, added_clicks = clicks, 0
    if getattr(args, 'query_clicks', False):
        # Auxiliary only: impressions and reconstructed fitting histories stay unchanged.
        auxiliary_clicks = np.union1d(clicks, scored_query_clicks(data, day))
        added_clicks = len(auxiliary_clicks) - len(clicks)
        if auxiliary_clicks['time'].max() >= cutoff:
            raise ValueError('Query-history clicks cross the model cutoff')
        common.log(f'Day {day}: query histories add {added_clicks:,} auxiliary clicks to {len(clicks):,}')
    ctx, ctx_len, targets = other_click_context(auxiliary_clicks, limit,
                                                gap=int(getattr(args, 'stale_context_hours', 0)*3600))
    provenance = dict(identity, max_fitting_event_time=int(data['time'][fit].max()),
                      unique_clicks=len(clicks), warmup_clicks=int((clicks['time'] < START).sum()),
                      max_click_time=int(clicks['time'].max()), auxiliary_examples=len(targets),
                      mean_history_length=float(lengths.mean()), added_auxiliary_clicks=added_clicks,
                      click_log_sha256=hashlib.sha256(clicks.tobytes()).hexdigest())
    common.log(f"Day {day}: {len(fit):,} fitting events, {len(clicks):,} unique clicks, "
               f"{args.seeds} seeds, {per_seed or steps} steps; mean history {lengths.mean():.1f}")
    def tensor(value, dtype=torch.long):
        return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=args.device)
    t = {'user': tensor(data['user'][fit]), 'items': tensor(data['items'][fit]+1),
         'label': tensor(data['label'][fit], torch.float32), 'hist': tensor(hist),
         'lengths': tensor(lengths, torch.float32), 'context': tensor(ctx),
         'context_lengths': tensor(ctx_len, torch.float32), 'targets': tensor(targets)}
    counts = np.bincount((data['items'][fit]+1).ravel(), minlength=n_items).astype(np.float64)
    cdf = np.cumsum(counts / counts.sum()).astype(np.float32)
    cdf[-1] = 1
    t['cdf'] = tensor(cdf, torch.float32)
    del hist, ctx, clicks, auxiliary_clicks
    gc.collect()
    optimizer = torch.optim.Adam(model.parameters(), lr=setting(args, 'lr'))
    torch.manual_seed(42)
    resume_path = path.with_name(path.stem + '.resume.pt')
    start_step = 0
    if resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu', weights_only=True)
        if resume['identity'] != identity:
            raise ValueError('Resume checkpoint belongs to a different recipe')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        start_step = resume['step']
        torch.set_rng_state(resume['cpu_rng'])
        if args.device == 'mps':
            torch.mps.set_rng_state(resume['device_rng'])
        common.log(f'Resuming day {day} at step {start_step}')
        del resume
    # A finished seed is held at its stopping point; Adam momentum would otherwise keep moving it.
    frozen = {seed: model.seed_state(seed) for seed, last in enumerate(per_seed or []) if last <= start_step}
    begin = time.time()
    model.train()
    for step in range(start_step, steps):
        ix = torch.randint(len(fit), (args.seeds, args.batch_events), device=args.device)
        components, reg = model(t['user'][ix], t['items'][ix], t['hist'][ix], t['lengths'][ix], regularize=True)
        primary = F.binary_cross_entropy_with_logits(components[..., 0], t['label'][ix], reduction='none').mean((1, 2))
        ai = torch.randint(len(targets), (args.seeds, args.batch_events), device=args.device)
        neg = torch.searchsorted(t['cdf'], torch.rand((args.seeds, args.batch_events, negatives), device=args.device), right=True)
        neg = neg.clamp(max=n_items-1)
        auxiliary, a_reg = model.auxiliary(t['context'][ai], t['context_lengths'][ai], t['targets'][ai], neg)
        losses = primary + l2*reg + setting(args, 'aux_weight')*(auxiliary + l2*a_reg)
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
        for seed, last in enumerate(per_seed or []):
            if step+1 == last:
                frozen[seed] = model.seed_state(seed)
            elif step+1 > last:
                model.load_seed_state(seed, frozen[seed])
        if (step+1) % args.checkpoint_every == 0 or step+1 == steps:
            values = losses.detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError('Non-finite training objective')
            elapsed = time.time()-begin
            common.log(f'Day {day}, step {step+1}/{steps}: objective {values.mean():.5f}, {elapsed:.1f}s since resume')
            if step+1 < steps:
                atomic_checkpoint(resume_path, dict(identity=identity, step=step+1,
                    model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                    cpu_rng=torch.get_rng_state(),
                    device_rng=torch.mps.get_rng_state() if args.device == 'mps' else torch.get_rng_state()))
    provenance['training_seconds_this_session'] = time.time()-begin
    atomic_checkpoint(path, dict(identity=identity, provenance=provenance, model=cpu_tree(model.state_dict())))
    atomic_json(path.with_suffix('.json'), provenance)
    resume_path.unlink(missing_ok=True)
    del t, optimizer
    gc.collect()
    if args.device == 'mps':
        torch.mps.empty_cache()
    return model, provenance


@torch.inference_mode()
def predict_components(model, query, indices, clicks, args):
    model.eval()
    result = np.empty((args.seeds, len(indices), 10, 4), np.float32)
    for start in range(0, len(indices), 1024):
        ix = indices[start:start+1024]
        hist, lengths = long_history(query, ix, clicks, model.n_items, setting(args, 'history_limit'))
        def expand(value, dtype=torch.long):
            t = torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=args.device)
            return t.unsqueeze(0).expand((args.seeds,) + t.shape)
        parts = model(expand(query['user'][ix]), expand(query['items'][ix]+1),
                      expand(hist), expand(lengths, torch.float32))
        result[:, start:start+len(ix)] = parts.cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError('Invalid MF components')
    return result


def ensure_parts(day, data, sizes, args):
    """One frozen daily model also supplies development or confirmation scores."""
    directory = OUT / 'parts'
    directory.mkdir(parents=True, exist_ok=True)
    queries = []
    if day < 7:
        ix = np.flatnonzero((data['time']-START)//DAY == day)
        queries.append((f'day{day}', data, ix))
    if day == 4:
        _, ix = common.chronological_split(data, '2025-01-24', '2025-01-26')
        queries.append(('development', data, ix))
    if day == 6:
        _, ix = common.chronological_split(data, '2025-01-26')
        queries.append(('confirmation', data, ix))
    if day == 7:
        query = common.load_data('evaluation')
        queries.append(('evaluation', query, np.arange(len(query['time']))))
    pending = []
    for name, query, ix in queries:
        path = directory / (name+'.npy')
        expected = dict(mf_identity(args), day=day, query_events=len(ix),
                        query_indices_sha256=hashlib.sha256(ix.astype(np.int64).tobytes()).hexdigest())
        meta_path = path.with_suffix('.json')
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if all(meta.get(k) == v for k, v in expected.items()):
                common.log('Reusing MF component cache: '+name)
                continue
        pending.append((name, query, ix, path, expected))
    if not pending:
        return
    model, provenance = fit_ensemble(data, day, sizes, args)
    fit = np.flatnonzero(data['time'] < START+day*DAY)
    clicks = unique_clicks(data, fit)
    for name, query, ix, path, expected in pending:
        components = predict_components(model, query, ix, clicks, args)
        atomic_npy(path, components)
        atomic_json(path.with_suffix('.json'), dict(expected, provenance=provenance,
                    component_names=PART_NAMES, shape=list(components.shape)))
        common.log('Saved MF components: '+name)
    del model, clicks
    gc.collect()
    if args.device == 'mps':
        torch.mps.empty_cache()


DROPPED_BASE = {'baseline': (), 'no-extended': (), 'no-spectral': ('spectral_',),
                'reference': ('spectral_', 'cf_', 'item_interacting_users', 'starter_logit')}


def base_columns(feature_subset='baseline'):
    """Base feature columns of a subset; every subset excludes the old learned-MF features."""
    meta = json.loads((ROOT/'artifacts/strategy2/features/train.json').read_text())
    names = meta['feature_names']
    keep = [i for i,n in enumerate(names) if not n.startswith(('neural_', 'learned_'))
            and n not in ['candidate_vector_norm', 'history_vector_norm']
            and not n.startswith(DROPPED_BASE[feature_subset])]
    if meta['identity']['data_fingerprint'] != common.cache_fingerprint():
        raise ValueError('Base features and encoded data do not match')
    return keep, [names[i] for i in keep]


def upgraded_features(base, parts, hist, lengths, candidates, keep, base_names, seeds, extended=True):
    mean = parts[:seeds].mean(axis=0, dtype=np.float64).astype(np.float32)
    score = mean[..., 0]
    extra = OrderedDict((name, mean[..., i]) for i,name in enumerate(PART_NAMES))
    extra['mf_centered'] = score-score.mean(axis=1, keepdims=True)
    extra['mf_rank'] = rank_within_event(score)
    extra['mf_minus_max'] = score-score.max(axis=1, keepdims=True)
    lse = logsumexp(score, axis=1, keepdims=True)
    extra['mf_softmax_share'] = np.exp(score-lse)
    def event(value):
        return np.broadcast_to(np.asarray(value)[:, None], score.shape)
    extra['mf_event_lse'] = np.broadcast_to(lse, score.shape)
    extra['mf_event_mean'] = event(score.mean(axis=1))
    extra['mf_event_max'] = event(score.max(axis=1))
    extra['mf_event_std'] = event(score.std(axis=1))
    extra['mf_seed_std'] = parts[:seeds, ..., 0].std(axis=0)
    extra['long_history_length'] = event(lengths)
    sorted_hist = np.sort(hist, axis=1)
    unique = (sorted_hist[:,0] > 0).astype(int) + ((sorted_hist[:,1:] > 0) & (np.diff(sorted_hist,axis=1) != 0)).sum(axis=1)
    extra['long_history_distinct_items'] = event(unique)
    repeat = np.stack([(hist == candidates[:,j,None]+1).sum(axis=1) for j in range(10)],axis=1)
    extra['long_history_candidate_repeat_count'] = repeat
    if not extended:
        for name in ['long_history_length', 'long_history_distinct_items', 'long_history_candidate_repeat_count']:
            del extra[name]
    result = np.concatenate([base[...,keep], np.stack(list(extra.values()),axis=2)],axis=2).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError('Non-finite tree features')
    return result, base_names+list(extra)


def ensure_features(name, data, sizes, seeds, args):
    part_path = OUT/'parts'/(name+'.npy')
    part_meta = json.loads(part_path.with_suffix('.json').read_text())
    day = part_meta['day']
    if name.startswith('day'):
        ix = np.flatnonzero((data['time']-START)//DAY == day)
        base = np.load(ROOT/'artifacts/strategy2/features'/('day_'+str((START//DAY)+day)+'.npy'),mmap_mode='r')
    elif name == 'development':
        _,ix = common.chronological_split(data,'2025-01-24','2025-01-26')
        base = np.load(ROOT/'artifacts/strategy2/features/development_valid.npy',mmap_mode='r')
    elif name == 'confirmation':
        _,ix = common.chronological_split(data,'2025-01-26')
        base = np.load(ROOT/'artifacts/strategy2/features/confirmation_valid.npy',mmap_mode='r')
    else:
        data = common.load_data('evaluation')
        ix = np.arange(len(data['time']))
        base = np.load(ROOT/'artifacts/strategy2/features/evaluation.npy',mmap_mode='r')
    feature_subset = getattr(args, 'feature_subset', 'baseline')
    keep, base_names = base_columns(feature_subset)
    identity = dict(run_identity(args), seeds_averaged=seeds, name=name, events=len(ix))
    path = OUT/'features'/(name+'.npy')
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text())['identity'] == identity:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    train = common.load_data('train')
    clicks = unique_clicks(train,np.flatnonzero(train['time'] < START+day*DAY))
    parts = np.load(part_path,mmap_mode='r')
    temporary = path.with_name(path.name+'.tmp')
    output = None
    for start in range(0,len(ix),2048):
        selected = ix[start:start+2048]
        hist,lengths = long_history(data,selected,clicks,sizes['all_items']+1,setting(args,'history_limit'))
        values,names = upgraded_features(base[start:start+len(selected)],parts[:,start:start+len(selected)],
                             hist,lengths,data['items'][selected],keep,base_names,seeds,
                             extended=feature_subset not in ('no-extended','reference'))
        if output is None:
            output = np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(ix),10,len(names)))
        output[start:start+len(selected)] = values
    if output is not None:
        output.flush()
    del output
    temporary.replace(path)
    atomic_json(meta_path,dict(identity=identity,feature_names=names,shape=[len(ix),10,len(names)]))
    common.log(f'Saved {name}: {len(names)} upgraded tree features')


def combine_features(data, through_day, seeds, args):
    paths = [OUT/'features'/f'day{d}.npy' for d in range(FIRST_DAY,through_day+1)]
    meta = json.loads(paths[0].with_suffix('.json').read_text())
    expected = dict(run_identity(args), through_day=through_day, seeds_averaged=seeds)
    path = OUT/'features'/f'train_through{through_day}.npy'
    mp = path.with_suffix('.json')
    if path.exists() and mp.exists() and json.loads(mp.read_text())['identity'] == expected:
        return
    first = int(np.searchsorted(data['time'],START+FIRST_DAY*DAY))
    last = int(np.searchsorted(data['time'],START+(through_day+1)*DAY))
    tmp = path.with_name(path.name+'.tmp')
    out = np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(last-first,10,len(meta['feature_names'])))
    position = 0
    for p in paths:
        block = np.load(p,mmap_mode='r')
        out[position:position+len(block)] = block
        position += len(block)
    assert position == last-first
    out.flush()
    del out
    tmp.replace(path)
    atomic_json(mp,dict(identity=expected,feature_names=meta['feature_names'],first_event=first,last_event=last))


def run_trees(phase,args):
    env = dict(os.environ)
    # Use the bundled runtime without importing PyTorch in the LightGBM process.
    runtime = Path(torch.__file__).resolve().parent/'lib'
    env['DYLD_LIBRARY_PATH'] = str(runtime)+(os.pathsep+env.get('DYLD_LIBRARY_PATH') if env.get('DYLD_LIBRARY_PATH') else '')
    cmd=[sys.executable,str(Path(__file__).resolve().parent/'history_upgrade_trees.py'),phase,
         '--threads',str(args.threads),'--output-dir',str(OUT),'--candidates',args.tree_candidates]
    if args.tree_refit_policy:
        cmd.append('--tree-refit-policy')
    subprocess.run(cmd,env=env,check=True)


def run(args):
    OUT.mkdir(parents=True,exist_ok=True)
    write_recipe(args)
    data = common.load_data('train')
    sizes = json.loads((common.CACHE/'id_counts.json').read_text())
    for day in [2,3,4]:
        ensure_parts(day,data,sizes,args)
    _,valid = common.chronological_split(data,'2025-01-24','2025-01-26')
    components = np.load(OUT/'parts/development.npy',mmap_mode='r')
    trials=[]
    for seeds in sorted(set([1,min(3,args.seeds),min(5,args.seeds),args.seeds])):
        score=components[:seeds,...,0].mean(axis=0,dtype=np.float64)
        trials.append(dict(seeds=seeds,development_auc=common.roc_auc(data['label'][valid],score)))
    best=max(trials,key=lambda r:r['development_auc'])
    seeds=best['seeds']
    atomic_json(OUT/'seed_selection.json',dict(trials=trials,selected_seeds=seeds))
    common.log('Seed averaging selected on development: '+json.dumps(trials))
    for name in ['day2','day3','development']:
        ensure_features(name,data,sizes,seeds,args)
    combine_features(data,3,seeds,args)
    run_trees('development',args)
    if args.until=='development':
        common.log('Stopping after development as requested')
        return
    # Model settings, seed count, and tree learner are now locked before confirmation.
    for day in [5,6]:
        ensure_parts(day,data,sizes,args)
    for name in ['day4','day5','confirmation']:
        ensure_features(name,data,sizes,seeds,args)
    combine_features(data,5,seeds,args)
    run_trees('confirmation',args)
    if args.until=='confirmation':
        common.log('Stopping after confirmation as requested')
        return
    ensure_parts(7,data,sizes,args)
    for name in ['day6','evaluation']:
        ensure_features(name,data,sizes,seeds,args)
    combine_features(data,6,seeds,args)
    run_trees('final',args)
    # Shared writer validates every output ID against the raw evaluation dataset.
    logits=np.load(OUT/'evaluation_logits.npy')
    report=json.loads((OUT/'results.json').read_text())
    report['submission']=common.write_submission(logits,filename=str(OUT/'submission_history_upgrade.csv'))
    report['comparison_submissions']={}
    for kind in report['final_models']:
        if kind!=report['confirmation']['selected_kind']:
            scores=np.load(OUT/f'evaluation_{kind}_logits.npy')
            report['comparison_submissions'][kind]=common.write_submission(scores,
                filename=str(OUT/f'submission_{kind}.csv'))
    report['recipe']=recipe(args)
    report['seed_selection']=json.loads((OUT/'seed_selection.json').read_text())
    atomic_json(OUT/'results.json',report)
    common.log('HISTORY UPGRADE COMPLETE: '+json.dumps({'submission':report['submission']}))


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--batch-events',type=int,default=820)
    parser.add_argument('--checkpoint-every',type=int,default=500)
    parser.add_argument('--threads',type=int,default=12)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT,
                        help='Run directory; use a new one for every changed recipe or experiment')
    parser.add_argument('--until',choices=['development','confirmation','final'],default='final',
                        help='Last stage to run; development is the cheap search stage')
    parser.add_argument('--tree-candidates',default=TREE_CANDIDATES,
                        help='Comma-separated Kind:size tree learners compared on development')
    parser.add_argument('--steps-per-seed',type=lambda v:[int(x) for x in v.split(',')],default=None,
                        help='Comma-separated MF updates for each seed, instead of --steps for all')
    parser.add_argument('--query-clicks',action='store_true',
                        help="Add pre-cutoff clicks from the scored queries' histories to the auxiliary loss")
    parser.add_argument('--feature-subset',choices=['baseline', 'reference', 'no-spectral', 'no-extended'],
                        default='baseline',
                        help='Feature subset: baseline (70 features), reference (~39), no-spectral, no-extended')
    parser.add_argument('--tree-refit-policy',action='store_true',
                        help='Pick the tree count by early stopping on the last fitting day, scale it, refit on all days')
    parser.add_argument('--stale-context-hours',type=float,default=0.,
                        help="Auxiliary context omits the user's clicks within this many hours of the target")
    parser.add_argument('--stale-history-hours',type=float,default=0.,
                        help='MF training histories stop this many hours before each event; scoring is unchanged')
    for name,value in MF_DEFAULTS.items():
        parser.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    return parser


def main():
    parser=build_parser()
    args=parser.parse_args()
    if args.steps_per_seed and len(args.steps_per_seed)!=args.seeds:
        parser.error('--steps-per-seed needs one value per seed')
    global OUT
    OUT=args.output_dir.resolve()
    if args.device=='auto':
        args.device='mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    run(args)


if __name__=='__main__':
    main()
