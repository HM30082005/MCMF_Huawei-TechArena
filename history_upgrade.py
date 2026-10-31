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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts' / 'history_upgrade'
VERSION = 1
START = common.utc_stamp('2025-01-20')
DAY = 86400
FIRST_DAY = 2
PART_NAMES = ['mf_logit', 'mf_item_bias', 'mf_user_bias', 'mf_history_dot']


def recipe(args):
    return dict(version=VERSION, dim=32, l2=.05, lr=.003, auxiliary_weight=.1,
                negatives=9, history_limit=100, init_std=.05, seeds=args.seeds,
                steps=args.steps, batch_events=args.batch_events,
                event_sampling='independent uniform with replacement per seed',
                historical_inputs='fitting histories and labels only; own query history at inference')


def run_identity(args):
    return {'data_fingerprint': common.cache_fingerprint(), 'recipe': recipe(args)}


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


def long_history(data, indices, clicks, n_items, limit=100):
    """Merge fitting clicks and the event's own history, strictly before its time."""
    uid, ts = np.asarray(data['user'][indices]), np.asarray(data['time'][indices])
    key = clicks['user'].astype(np.int64) * 10_000_000_000 + clicks['time']
    starts = np.searchsorted(key, uid.astype(np.int64) * 10_000_000_000, side='left')
    ends = np.searchsorted(key, uid.astype(np.int64) * 10_000_000_000 + ts, side='left')
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
    own_items, own_times = np.maximum(h + 1, 0), np.where(h >= 0, ht, 0)
    codes = np.concatenate([times * n_items + items, own_times * n_items + own_items], axis=1)
    codes.sort(axis=1)
    duplicates = np.concatenate([np.zeros((len(indices), 1), bool), codes[:, 1:] == codes[:, :-1]], axis=1)
    codes[duplicates] = 0
    codes.sort(axis=1)
    codes = codes[:, -limit:][:, ::-1]
    return (codes % n_items).astype(np.int32), (codes > 0).sum(axis=1).astype(np.int16)


def other_click_context(clicks, limit=100):
    """Other clicks of the same user; omit the exact target occurrence."""
    users = clicks['user']
    if not len(users):
        return np.zeros((0, limit), np.int32), np.zeros(0, np.int16), np.zeros(0, np.int32)
    boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    counts = np.diff(boundaries)
    starts = np.repeat(boundaries[:-1], counts)
    ends = np.repeat(boundaries[1:], counts)
    positions = np.arange(len(users))
    lengths = np.minimum(ends - starts - 1, limit)
    result = np.zeros((len(users), limit), np.int32)
    # Column blocks avoid an additional N x 100 int64 temporary allocation.
    for j in range(limit):
        idx = ends - 1 - j
        idx -= idx <= positions
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
    identity = dict(run_identity(args), cutoff=cutoff, fitting_events=len(fit))
    path = OUT / 'models' / f'day{day}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True) if path.exists() else None
    if checkpoint is not None and checkpoint['identity'] != identity:
        raise ValueError('Checkpoint recipe/data changed; use a separate output directory')
    model = SharedHistoryEnsemble(sizes['all_users'], sizes['all_items']+1, args.seeds).to(args.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        common.log(f'Reusing completed day-{day} ensemble')
        return model, checkpoint['provenance']
    clicks = unique_clicks(data, fit)
    if len(clicks) and clicks['time'].max() >= cutoff:
        raise ValueError('Fitting click log crosses the model cutoff')
    n_items = sizes['all_items']+1
    hist, lengths = long_history(data, fit, clicks, n_items)
    ctx, ctx_len, targets = other_click_context(clicks)
    provenance = dict(identity, max_fitting_event_time=int(data['time'][fit].max()),
                      unique_clicks=len(clicks), warmup_clicks=int((clicks['time'] < START).sum()),
                      max_click_time=int(clicks['time'].max()), auxiliary_examples=len(targets),
                      mean_history_length=float(lengths.mean()),
                      click_log_sha256=hashlib.sha256(clicks.tobytes()).hexdigest())
    common.log(f'Day {day}: {len(fit):,} fitting events, {len(clicks):,} unique clicks, '
               f'{args.seeds} seeds, {args.steps} steps; mean history {lengths.mean():.1f}')
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
    del hist, ctx, clicks
    gc.collect()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
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
    begin = time.time()
    model.train()
    for step in range(start_step, args.steps):
        ix = torch.randint(len(fit), (args.seeds, args.batch_events), device=args.device)
        components, reg = model(t['user'][ix], t['items'][ix], t['hist'][ix], t['lengths'][ix], regularize=True)
        primary = F.binary_cross_entropy_with_logits(components[..., 0], t['label'][ix], reduction='none').mean((1, 2))
        ai = torch.randint(len(targets), (args.seeds, args.batch_events), device=args.device)
        neg = torch.searchsorted(t['cdf'], torch.rand((args.seeds, args.batch_events, 9), device=args.device), right=True)
        neg = neg.clamp(max=n_items-1)
        auxiliary, a_reg = model.auxiliary(t['context'][ai], t['context_lengths'][ai], t['targets'][ai], neg)
        losses = primary + .05*reg + .1*(auxiliary + .05*a_reg)
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
        if (step+1) % args.checkpoint_every == 0 or step+1 == args.steps:
            values = losses.detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError('Non-finite training objective')
            elapsed = time.time()-begin
            common.log(f'Day {day}, step {step+1}/{args.steps}: objective {values.mean():.5f}, {elapsed:.1f}s since resume')
            if step+1 < args.steps:
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
        hist, lengths = long_history(query, ix, clicks, model.n_items)
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
        expected = dict(run_identity(args), day=day, query_events=len(ix),
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


def base_columns():
    meta = json.loads((ROOT/'artifacts/strategy2/features/train.json').read_text())
    # Remove every feature derived from the previous learned preference model.
    names = meta['feature_names']
    keep = [i for i,n in enumerate(names) if not n.startswith(('neural_', 'learned_'))
            and n not in ['candidate_vector_norm', 'history_vector_norm']]
    if meta['identity']['data_fingerprint'] != common.cache_fingerprint():
        raise ValueError('Base features and encoded data do not match')
    return keep, [names[i] for i in keep]


def upgraded_features(base, parts, hist, lengths, candidates, keep, base_names, seeds):
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
    keep, base_names = base_columns()
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
        hist,lengths = long_history(data,selected,clicks,sizes['all_items']+1)
        values,names = upgraded_features(base[start:start+len(selected)],parts[:,start:start+len(selected)],
                             hist,lengths,data['items'][selected],keep,base_names,seeds)
        if output is None:
            output = np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(ix),10,len(names)))
        output[start:start+len(selected)] = values
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
    env['DYLD_LIBRARY_PATH'] = str(runtime)+(os.pathsep+env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
    subprocess.run([sys.executable,str(ROOT/'history_upgrade_trees.py'),phase,'--threads',str(args.threads)],env=env,check=True)


def run(args):
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'recipe.json',run_identity(args))
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
    # Model settings, seed count, and tree learner are now locked before confirmation.
    for day in [5,6]:
        ensure_parts(day,data,sizes,args)
    for name in ['day4','day5','confirmation']:
        ensure_features(name,data,sizes,seeds,args)
    combine_features(data,5,seeds,args)
    run_trees('confirmation',args)
    ensure_parts(7,data,sizes,args)
    for name in ['day6','evaluation']:
        ensure_features(name,data,sizes,seeds,args)
    combine_features(data,6,seeds,args)
    run_trees('final',args)
    # Shared writer validates every output ID against the raw evaluation dataset.
    logits=np.load(OUT/'evaluation_logits.npy')
    report=json.loads((OUT/'results.json').read_text())
    report['submission']=common.write_submission(logits,filename='artifacts/history_upgrade/submission_history_upgrade.csv')
    report['comparison_submissions']={}
    for kind in report['final_models']:
        if kind!=report['confirmation']['selected_kind']:
            scores=np.load(OUT/f'evaluation_{kind}_logits.npy')
            report['comparison_submissions'][kind]=common.write_submission(scores,
                filename=f'artifacts/history_upgrade/submission_{kind}.csv')
    report['recipe']=recipe(args)
    report['seed_selection']=json.loads((OUT/'seed_selection.json').read_text())
    atomic_json(OUT/'results.json',report)
    common.log('HISTORY UPGRADE COMPLETE: '+json.dumps({'confirmation':report['confirmation']['selected'],'submission':report['submission']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--batch-events',type=int,default=820)
    parser.add_argument('--checkpoint-every',type=int,default=500)
    parser.add_argument('--threads',type=int,default=12)
    args=parser.parse_args()
    if args.device=='auto':
        args.device='mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    run(args)


if __name__=='__main__':
    main()
