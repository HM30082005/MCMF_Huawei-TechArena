"""Benchmark actual downloaded FuxiCTR DIN, DeepFM, and DCNv2 classes.

Upstream model source is unchanged. This adapter provides the same chronological
splits, input information, initialization options, and scoring used by strategy1.
"""
import argparse
from collections import OrderedDict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

import strategy1 as common

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'ctr_benchmark'
VENDOR = OUT / 'vendor' / 'FuxiCTR'
PINNED_REVISION = '183da1c944ba8930fcdcd54114837c081959d748'
MODEL_PATHS = {
    'DIN': 'model_zoo/DIN/src/DIN.py',
    'DeepFM': 'model_zoo/DeepFM/DeepFM_torch/src/DeepFM.py',
    'DCNv2': 'model_zoo/DCNv2/src/DCNv2.py',
}


def upstream_classes():
    if not VENDOR.exists():
        raise RuntimeError('Download pinned FuxiCTR into artifacts/ctr_benchmark/vendor/FuxiCTR first')
    revision = subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PINNED_REVISION:
        raise RuntimeError('Unexpected upstream revision: ' + revision)
    sys.path.insert(0, str(VENDOR))
    classes = {}
    for name, relative in MODEL_PATHS.items():
        spec = importlib.util.spec_from_file_location('upstream_ctr_' + name, VENDOR / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes[name] = getattr(module, name)
    return classes


def feature_map(name, sizes, dim):
    from fuxictr.features import FeatureMap
    schema = FeatureMap('techarena', str(OUT))
    schema.default_emb_dim = dim
    schema.labels = ['clicked']
    for key, vocab in [('user_id', sizes['all_users'] + 1), ('item_id', sizes['all_items'] + 1),
                       ('time_bucket', 5), ('day_type', 3)]:
        schema.features[key] = {'type': 'categorical', 'source': '', 'vocab_size': vocab, 'padding_idx': 0}
    for key in ['history_length', 'last_click_age', 'empty_history',
                'candidate_repeat', 'candidate_repeat_count', 'candidate_repeat_age']:
        schema.features[key] = {'type': 'numeric', 'source': ''}
    history = {'type': 'sequence', 'source': '', 'vocab_size': sizes['all_items'] + 1,
               'padding_idx': 0, 'max_len': 20, 'share_embedding': 'item_id'}
    if name != 'DIN':
        history['feature_encoder'] = 'layers.MaskedAveragePooling()'
    schema.features['click_history'] = history
    schema.num_fields = schema.get_num_fields()
    schema.total_features = sum(spec.get('vocab_size', 1) for spec in schema.features.values())
    schema.set_column_index()
    return schema


class EventAdapter:
    """Expand cached events to upstream per-impression batches without copying 7GB CSVs."""
    def __init__(self, tensors, user_counts, item_counts, device):
        self.tensors, self.device = tensors, device
        self.known_users = torch.tensor(np.r_[False, np.asarray(user_counts) > 0], device=device)
        # Histories may contain fitting-period items absent as candidate impressions.
        self.known_items = torch.tensor(np.r_[False, np.asarray(item_counts) > 0], device=device)

    def __call__(self, indices, training=False, user_dropout=.2):
        t = {key: value[indices] for key, value in self.tensors.items()}
        uid = t['user'].repeat_interleave(10)
        uid = torch.where(self.known_users[uid], uid, 0)
        if training and user_dropout:
            # Match the existing model's user-dropout convention at event level.
            keep = (torch.rand(len(indices), device=self.device) >= user_dropout).repeat_interleave(10)
            uid = uid * keep
        items = t['items'].reshape(-1)
        items = torch.where(self.known_items[items], items, 0)
        hist = t['hist'].repeat_interleave(10, dim=0)
        hist = torch.where(self.known_items[hist], hist, 0)
        event = t['event_features'].repeat_interleave(10, dim=0)
        repeat = t['repeat'].reshape(-1, 3)
        result = OrderedDict([
            ('user_id', uid), ('item_id', items),
            ('time_bucket', t['bucket'].repeat_interleave(10) + 1),
            ('day_type', t['weekend'].repeat_interleave(10) + 1),
            ('history_length', event[:, 0]), ('last_click_age', event[:, 1]),
            ('empty_history', event[:, 2]), ('candidate_repeat', repeat[:, 0]),
            ('candidate_repeat_count', repeat[:, 1]), ('candidate_repeat_age', repeat[:, 2]),
            ('click_history', hist),
        ])
        if 'label' in t:
            result['clicked'] = t['label'].reshape(-1, 1)
        return result


def fit_counts(data, fit, sizes):
    users = np.bincount(data['user'][fit], minlength=sizes['all_users'])
    item = np.bincount(data['items'][fit].ravel(), minlength=sizes['all_items'])
    h = data['hist'][fit]
    item += np.bincount(h[h >= 0], minlength=sizes['all_items'])
    return users, item


def build_model(cls, name, sizes, config, device, initial=None):
    common.seed_all(config['seed'])
    schema = feature_map(name, sizes, config['dim'])
    kwargs = dict(model_id=name, gpu=-1, model_root=str(OUT / 'upstream_checkpoints'),
                  verbose=0, metrics=['AUC', 'logloss'], task='binary_classification',
                  optimizer='adam', loss='binary_crossentropy', learning_rate=config['lr'],
                  embedding_dim=config['dim'], embedding_regularizer=0, net_regularizer=0,
                  net_dropout=.1, batch_norm=False)
    if name == 'DIN':
        kwargs.update(dnn_hidden_units=[128, 64], attention_hidden_units=[64, 32],
                      din_target_field=['item_id'], din_sequence_field=['click_history'])
    elif name == 'DeepFM':
        kwargs.update(hidden_units=[128, 64])
    else:
        kwargs.update(parallel_dnn_hidden_units=[128, 64], model_structure='parallel',
                      num_cross_layers=3, use_low_rank_mixture=False)
    model = cls(schema, **kwargs)
    if initial is not None:
        # Public item and history embeddings share the same table in every model.
        table = (model.embedding_layer.embedding_layers if name == 'DIN'
                 else model.embedding_layer.embedding_layer.embedding_layers)
        with torch.no_grad():
            table['item_id'].weight[1:].copy_(torch.as_tensor(initial))
    model.device = torch.device(device)
    model.to(device)
    # Optimizer is set after moving to MPS; model.forward is upstream, unchanged.
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    return model


def as_logits(probabilities):
    return logit(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1 - 1e-15))


@torch.inference_mode()
def predict(model, adapter, indices, event_batch):
    model.eval()
    output = []
    for start in range(0, len(indices), event_batch):
        ix = torch.as_tensor(indices[start:start + event_batch], dtype=torch.long, device=model.device)
        y = model(adapter(ix))['y_pred']
        output.append(y.reshape(-1, 10).cpu().numpy())
    p = np.concatenate(output)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError('Invalid upstream model probabilities')
    return p


def run_fit(cls, name, data, tensors, fit, valid, sizes, config, stage, args, epochs=None):
    tag = f'{stage}_{name}_d{config["dim"]}_{config["initialization"]}'
    initial = None
    if config['initialization'] == 'spectral':
        initial = common.spectral_initialization(data, fit, sizes, config['dim'], stage, args.seed,
                                                  output_dir=OUT / 'initialization')
    model = build_model(cls, name, sizes, config, args.device, initial)
    user_counts, item_counts = fit_counts(data, fit, sizes)
    adapter = EventAdapter(tensors, user_counts, item_counts, args.device)
    rng = np.random.default_rng(args.seed)
    fit_device = torch.tensor(fit, dtype=torch.long, device=args.device)
    best_auc, best_epoch, stale, best_state = -1., 0, 0, None
    curve = []
    for epoch in range(1, (epochs or args.max_epochs) + 1):
        start = time.time()
        model.train()
        perm = torch.tensor(rng.permutation(len(fit)), dtype=torch.long, device=args.device)
        order = fit_device[perm]
        objective = 0.
        for begin in range(0, len(fit), args.event_batch):
            ix = order[begin:begin + args.event_batch]
            batch = adapter(ix, training=True)
            model.optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            loss = F.binary_cross_entropy(result['y_pred'].clamp(1e-7, 1 - 1e-7), batch['clicked'])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            model.optimizer.step()
            objective += float(loss.detach().cpu()) * len(ix)
        row = {'epoch': epoch, 'train_log_loss': objective / len(fit)}
        if valid is not None:
            p = predict(model, adapter, valid, args.event_batch)
            row.update(common.metrics(data['label'][valid], as_logits(p)))
            if row['roc_auc'] > best_auc + 1e-5:
                best_auc, best_epoch, stale = row['roc_auc'], epoch, 0
                best_state = {key: v.detach().cpu().clone() for key, v in model.state_dict().items()}
            else:
                stale += 1
        row['seconds'] = round(time.time() - start, 2)
        curve.append(row)
        common.write_json(OUT / f'{tag}_curve.json', curve)
        common.log(tag + ': ' + json.dumps(row))
        if valid is not None and stale >= args.patience:
            break
    if valid is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    checkpoint = {'model_name': name, 'config': config, 'sizes': sizes, 'stage': stage,
                  'best_epoch': best_epoch, 'user_counts': user_counts.tolist(),
                  'item_counts': item_counts.tolist(), 'cache_fingerprint': common.cache_fingerprint(),
                  'upstream_revision': PINNED_REVISION,
                  'state_dict': {key: v.detach().cpu() for key, v in model.state_dict().items()}}
    torch.save(checkpoint, OUT / f'{tag}.pt')
    summary = {'config': config, 'best_epoch': best_epoch, 'best_auc': best_auc if valid is not None else None,
               'parameters': sum(x.numel() for x in model.parameters()), 'curve': curve,
               'checkpoint': str(OUT / f'{tag}.pt')}
    return model, adapter, summary


def verification_smoke(classes, sizes, data, tensors, fit, args):
    uc, ic = fit_counts(data, fit, sizes)
    adapter = EventAdapter(tensors, uc, ic, args.device)
    ix = torch.tensor(fit[:16], dtype=torch.long, device=args.device)
    checks = {}
    for name, cls in classes.items():
        config = {'dim': 32, 'initialization': 'scratch', 'lr': args.lr, 'seed': args.seed}
        model = build_model(cls, name, sizes, config, args.device)
        batch = adapter(ix)
        model.eval()
        with torch.no_grad():
            p = model(batch)['y_pred']
            changed = dict(batch, clicked=1-batch['clicked'])
            assert torch.allclose(p, model(changed)['y_pred'])
            permutation = torch.arange(len(p) - 1, -1, -1, device=args.device)
            shuffled = {key: value[permutation] for key, value in batch.items()}
            assert torch.allclose(p[permutation], model(shuffled)['y_pred'], atol=1e-5, rtol=1e-4)
            empty = dict(batch, click_history=torch.zeros_like(batch['click_history']))
            assert torch.isfinite(model(empty)['y_pred']).all()
        model.train()
        loss = F.binary_cross_entropy(model(batch)['y_pred'], batch['clicked'])
        loss.backward()
        assert any(x.grad is not None and float(x.grad.abs().sum().cpu()) > 0 for x in model.parameters())
        checks[name] = {'finite_empty_histories': True, 'labels_ignored_in_forward': True,
                        'row_permutation_equivariant': True, 'training_gradients': True}
    common.write_json(OUT / 'smoke_checks.json', checks)
    common.log('Upstream model correctness smoke checks passed')


def verify_reload(cls, name, checkpoint_path, adapter, indices, expected, args):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['cache_fingerprint'] != common.cache_fingerprint():
        raise RuntimeError('Checkpoint feature mappings changed')
    model = build_model(cls, name, checkpoint['sizes'], checkpoint['config'], args.device)
    model.load_state_dict(checkpoint['state_dict'])
    actual = predict(model, adapter, indices, args.event_batch)
    diff = float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-5):
        raise RuntimeError('Checkpoint reload did not reproduce predictions')
    return {'checkpoint_reloaded': True, 'max_absolute_probability_difference': diff}


def train(args):
    OUT.mkdir(parents=True, exist_ok=True)
    classes = upstream_classes()
    data = common.load_data('train')
    sizes = json.loads((common.CACHE / 'id_counts.json').read_text())
    tensors = common.prepare_tensors(data, args.device)
    dev_fit, dev_valid = common.chronological_split(data, '2025-01-24', '2025-01-26')
    test_fit, test_valid = common.chronological_split(data, '2025-01-26')
    assert not np.intersect1d(dev_valid, test_valid).size
    with np.load(ROOT / 'artifacts/strategy1/confirmation_predictions.npz') as previous:
        np.testing.assert_array_equal(test_valid, previous['event_indices'])
        np.testing.assert_array_equal(data['label'][test_valid], previous['labels'])
    report = {'source': 'https://github.com/reczoo/FuxiCTR', 'revision': PINNED_REVISION,
              'upstream_model_files_sha256': {name: hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest()
                                               for name, rel in MODEL_PATHS.items()},
              'config': vars(args), 'cache_fingerprint': common.cache_fingerprint(),
              'development_fit_events': len(dev_fit), 'development_valid_events': len(dev_valid),
              'confirmation_fit_events': len(test_fit), 'confirmation_valid_events': len(test_valid),
              'models': {}, 'notes': ['No evaluation labels exist or are used.',
                  'Best development configuration and epoch are fixed before confirmation.',
                  'Candidate and history IDs share embeddings; DeepFM/DCNv2 use masked mean history pooling.',
                  'Public model forward implementations are unchanged; custom event adapter, AdamW loop, and MPS device placement.',
                  'Same user dropout and label-free numeric features as strategy1; public architectures retain their own parameterization.']}
    verification_smoke(classes, sizes, data, tensors, dev_fit, args)
    configs = [{'dim': dim, 'initialization': 'spectral', 'lr': args.lr, 'seed': args.seed} for dim in args.dims]
    if args.scratch_comparison:
        configs.append({'dim': 64, 'initialization': 'scratch', 'lr': args.lr, 'seed': args.seed})
    # Select every architecture on the development window before scoring any on confirmation.
    for name in args.models:
        model_report = {'development': []}
        report['models'][name] = model_report
        for config in configs:
            model, adapter, result = run_fit(classes[name], name, data, tensors, dev_fit, dev_valid,
                                             sizes, config, 'development', args)
            model_report['development'].append(result)
            common.write_json(OUT / 'results.json', report)
            del model, adapter
            gc.collect()
            if args.device == 'mps':
                torch.mps.empty_cache()
        best = max(model_report['development'], key=lambda row: row['best_auc'])
        model_report['selected'] = {'config': best['config'], 'epochs': best['best_epoch'],
                                     'development_auc': best['best_auc']}
        common.write_json(OUT / 'results.json', report)
    common.write_json(OUT / 'selection_locked.json', {name: row['selected'] for name, row in report['models'].items()})
    common.log('All configurations locked; beginning January 26 confirmation')
    for name in args.models:
        chosen = report['models'][name]['selected']
        model, adapter, result = run_fit(classes[name], name, data, tensors, test_fit, None,
                                         sizes, chosen['config'], 'confirmation', args, epochs=chosen['epochs'])
        p = predict(model, adapter, test_valid, args.event_batch)
        labels = np.array(data['label'][test_valid])
        assert abs(common.roc_auc(labels, p) - roc_auc_score(labels.ravel(), p.ravel())) < 1e-12
        np.savez(OUT / f'{name}_confirmation_predictions.npz', event_indices=test_valid, labels=labels, probabilities=p)
        result['slices'] = common.evaluate_slices(data, test_valid, as_logits(p),
                                                 fit_counts(data, test_fit, sizes)[0])
        result['verification'] = verify_reload(classes[name], name, result['checkpoint'], adapter,
                                                test_valid, p, args)
        report['models'][name]['confirmation'] = result
        common.write_json(OUT / 'results.json', report)
        common.log(name + ' CONFIRMATION: ' + json.dumps(result['slices']['all']))
        del model, adapter
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
    if args.submissions:
        eval_data = common.load_data('evaluation')
        eval_tensors = common.prepare_tensors(eval_data, args.device)
        all_fit = np.arange(len(data['user']))
        for name in args.models:
            chosen = report['models'][name]['selected']
            model, _, result = run_fit(classes[name], name, data, tensors, all_fit, None,
                                       sizes, chosen['config'], 'final', args, epochs=chosen['epochs'])
            adapter = EventAdapter(eval_tensors, *fit_counts(data, all_fit, sizes), args.device)
            p = predict(model, adapter, np.arange(len(eval_data['user'])), args.event_batch)
            np.save(OUT / f'{name}_evaluation_probabilities.npy', p)
            report['models'][name]['submission'] = common.write_submission(as_logits(p),
                         filename=f'artifacts/ctr_benchmark/submissions/{name}.csv')
            report['models'][name]['final'] = result
            common.write_json(OUT / 'results.json', report)
            common.log(name + ' verified CSV: ' + str(OUT / f'submissions/{name}.csv'))
            del model, adapter
            gc.collect()
            if args.device == 'mps':
                torch.mps.empty_cache()
    for name, relative in MODEL_PATHS.items():
        if hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest() != report['upstream_model_files_sha256'][name]:
            raise RuntimeError('Upstream model source changed during benchmarking')
    common.write_json(OUT / 'results.json', report)
    common.log('BENCHMARK COMPLETE: ' + json.dumps({name: row['confirmation']['slices']['all'] for name, row in report['models'].items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_PATHS), default=list(MODEL_PATHS))
    parser.add_argument('--dims', nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--event-batch', type=int, default=1024)
    parser.add_argument('--max-epochs', type=int, default=18)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scratch-comparison', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--submissions', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    common.log(f'Downloaded FuxiCTR benchmark on {args.device}')
    train(args)


if __name__ == '__main__':
    main()
