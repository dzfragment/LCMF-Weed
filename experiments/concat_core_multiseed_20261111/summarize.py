"""Aggregate completed arms only; retain each seed and label incomplete groups."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
METRICS = ('point_recall', 'nPE_mean_px', 'point_ap_10px',
           'point_precision', 'nPE_median_px', 'penalized_error_10px')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def aggregate(records, arms):
    result = []
    for arm in arms:
        for split in ('val', 'test'):
            for tag in ('last', 'best'):
                selected = [r for r in records if
                            (r['arm'], r['split'], r['tag']) == (arm, split, tag)]
                seeds = sorted(r['seed'] for r in selected)
                assert len(seeds) == len(set(seeds))
                if not selected:
                    continue
                entry = dict(arm=arm, split=split, tag=tag, seeds=seeds,
                             n_seeds=len(seeds), complete=seeds == [42, 43, 44])
                entry['metrics'] = {
                    k: dict(mean=statistics.mean(r[k] for r in selected),
                            sample_std=statistics.stdev(r[k] for r in selected)
                            if len(selected) > 1 else None)
                    for k in METRICS
                }
                result.append(entry)
    return result


def main():
    plan = json.loads((HERE / 'suite_plan.json').read_text())
    roots = {42: HERE / plan['baseline_seed42'],
             **{s: HERE / f'seed{s}' for s in plan['seeds']}}
    records = []
    for seed, root in roots.items():
        for arm in plan['arms']:
            folder = root / 'results' / arm
            if not (folder / 'complete.json').exists():
                continue
            for split in ('val', 'test'):
                for tag in ('last', 'best'):
                    record = json.loads((folder / f'eval_{split}_{tag}.json').read_text())
                    ck = folder / f'pad_stage3_{tag}.pt'
                    assert record['checkpoint_sha256'] == digest(ck), ck
                    assert record['arm'] == arm
                    assert record.get('seed', seed) == seed
                    records.append(record | dict(seed=seed))
    groups = aggregate(records, plan['arms'])
    report = dict(updated=datetime.now(timezone.utc).isoformat(),
                  expected_seeds=[42, 43, 44], standard_deviation='sample, ddof=1',
                  records=records, aggregates=groups)
    write(HERE / 'summary.json', json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    lines = ['# Core ablations: seeds 42, 43, 44', '',
             'Primary comparison: test / raw last. Mean ± sample standard deviation.',
             'Rows with fewer than three seeds are incomplete; one-seed std is undefined.', '',
             '| Arm | Seeds | Recall (%) | Mean distance (px) | AP10 |',
             '|---|---|---:|---:|---:|']
    for r in groups:
        if r['split'] != 'test' or r['tag'] != 'last':
            continue
        cells = []
        for key, scale, digits in [('point_recall', 100, 3),
                                   ('nPE_mean_px', 1, 4), ('point_ap_10px', 1, 5)]:
            m = r['metrics'][key]
            std = f"{m['sample_std'] * scale:.{digits}f}" if m['sample_std'] is not None else '—'
            cells.append(f"{m['mean'] * scale:.{digits}f} ± {std}")
        lines.append(f"| {r['arm']} | {','.join(map(str, r['seeds']))} | " + ' | '.join(cells) + ' |')
    write(HERE / 'RESULTS.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
