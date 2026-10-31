"""Parse the full CSVs once into compact, validated event-level NumPy arrays."""
import csv
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'analysis' / '.cache'
CACHE.mkdir(exist_ok=True)
users, items = {}, {}


def encode(mapping, key):
    value = mapping.get(key)
    if value is None:
        value = len(mapping)
        mapping[key] = value
    return value


def parse(name, expected):
    start = time.time()
    arrays = {
        'user': np.empty(expected, np.int32),
        'items': np.empty((expected, 10), np.int32),
        'time': np.empty(expected, np.int64),
        'bucket': np.empty(expected, np.int8),
        'weekend': np.empty(expected, np.int8),
        'hist': np.full((expected, 20), -1, np.int32),
        'hist_time': np.zeros((expected, 20), np.int64),
        'hist_len': np.empty(expected, np.int8),
        'label': np.zeros((expected, 10), np.int8),
    }
    event_ids = set()
    invalid_history_order = 0
    event_count = 0
    with (ROOT / (name + '.csv')).open(newline='') as src:
        reader = csv.reader(src)
        header = next(reader)
        print(name, 'columns:', header, flush=True)
        while batch := list(islice(reader, 10)):
            n = event_count
            assert n < expected and len(batch) == 10
            first = batch[0]
            assert first[1] not in event_ids, 'Event is not contiguous or is duplicated'
            event_ids.add(first[1])
            assert all(len(row) == len(header) and all(row) for row in batch)
            assert all(row[1:3] == first[1:3] and row[4:9] == first[4:9] for row in batch)
            assert len({row[3] for row in batch}) == 10
            hist = json.loads(first[7])
            hist_time = json.loads(first[8])
            assert len(hist) == len(hist_time) <= 20
            stamp = int(first[4])
            assert all(t < stamp for t in hist_time)
            invalid_history_order += int(hist_time != sorted(hist_time))
            arrays['user'][n] = encode(users, first[2])
            arrays['items'][n] = [encode(items, row[3]) for row in batch]
            arrays['time'][n] = stamp
            arrays['bucket'][n] = int(first[5])
            arrays['weekend'][n] = int(first[6])
            arrays['hist_len'][n] = len(hist)
            arrays['hist'][n, :len(hist)] = [encode(items, item) for item in hist]
            arrays['hist_time'][n, :len(hist)] = hist_time
            if name == 'train':
                labels = [int(row[9]) for row in batch]
                assert all(y in (0, 1) for y in labels) and sum(labels) <= 1
                arrays['label'][n] = labels
            event_count += 1
            if event_count % 100000 == 0:
                print(f'{name}: {event_count:,} events, {time.time()-start:.1f}s', flush=True)
    assert event_count == expected
    assert invalid_history_order == 0
    print(f'{name}: done; sorted={bool(np.all(np.diff(arrays["time"]) >= 0))}, '
          f'{len(users):,} user IDs / {len(items):,} item IDs including histories; '
          f'{time.time()-start:.1f}s', flush=True)
    for key, value in arrays.items():
        np.save(CACHE / f'{name}_{key}.npy', value)


if __name__ == '__main__':
    parse('train', 686824)
    train_users, train_items = len(users), len(items)
    parse('evaluation', 52939)
    (CACHE / 'id_counts.json').write_text(json.dumps({
        'train_users_in_mapping': train_users, 'train_items_in_mapping': train_items,
        'all_users': len(users), 'all_items': len(items),
    }, indent=2))
    (CACHE / 'user_ids.json').write_text(json.dumps(users))
    (CACHE / 'item_ids.json').write_text(json.dumps(items))
