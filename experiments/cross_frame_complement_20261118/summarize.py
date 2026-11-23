"""One-to-one point metrics and paired, group-clustered inference diagnostics."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import pickle
import re

import numpy as np

from evaluate import CLEAN, DATA_ROOT, HERE, REAL
from pose_metrics import pose_metrics


def ranked_matches(prediction, gt):
    order = np.argsort(-prediction[:, 4], kind='stable')
    scores = prediction[order, 4]
    points = prediction[order, 5:]
    matched = np.full(len(scores), -1, dtype=np.int32)
    errors = np.zeros(len(scores), dtype=np.float32)
    if len(gt) and len(scores):
        distances = np.linalg.norm(points[:, None] - gt[None, :, 4:], axis=2)
        unavailable = np.zeros(len(gt), dtype=bool)
        for idx in np.flatnonzero(np.any(distances <= 10., axis=1)):
            choices = np.where(unavailable, np.inf, distances[idx])
            target = int(choices.argmin())
            if choices[target] <= 10.:
                unavailable[target] = True
                matched[idx] = target
                errors[idx] = choices[target]
    return scores, matched, errors


def metrics(items, total_gt):
    scores = np.concatenate([score for score, _, _ in items])
    hits = np.concatenate([match >= 0 for _, match, _ in items])
    errors = np.concatenate([error for _, _, error in items])
    order = np.argsort(-scores, kind='stable')
    scores, hits, errors = scores[order], hits[order], errors[order]
    ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True]) if len(scores) else np.array([], int)
    true_positives = hits.cumsum()[ends]
    precision = true_positives / (1 + ends)
    recall = true_positives / total_gt
    ap10 = float(np.sum(np.diff(np.r_[0., recall]) * np.maximum.accumulate(precision[::-1])[::-1]))
    feasible = np.flatnonzero(1 + ends - true_positives <= len(items))
    threshold = float(scores[ends[feasible[-1]]]) if len(feasible) else None

    def operating_point(conf):
        high = scores >= conf if conf is not None else np.zeros(len(scores), bool)
        num_pred, num_hit = int(high.sum()), int((high & hits).sum())
        return dict(recall=num_hit / total_gt, precision=num_hit / max(1, num_pred),
                    fp_per_group=(num_pred - num_hit) / len(items), confidence=conf,
                    mean_error_px=float(errors[high & hits].mean()) if num_hit else None)

    return dict(ap10=ap10, at25=operating_point(.25), at_fp1=operating_point(threshold))


def flags(items, rows, conf):
    result = []
    for (scores, gt_matches, _), row in zip(items, rows, strict=True):
        mask = np.zeros(len(row['gt']), dtype=bool)
        if conf is not None:
            mask[gt_matches[(scores >= conf) & (gt_matches >= 0)]] = True
        result.append(mask)
    return result


def clusters_for(rows, case):
    if case == 'controlled':
        prov = json.loads((CLEAN / 'provenance.json').read_text())
        return [public_session(prov[row['name']]['source']) if row['kind'].startswith('public')
                else row['cluster'] for row in rows]
    owners, parent = {}, list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            index = parent[index]
        return index

    for index, row in enumerate(rows):
        for digest in row['hashes']:
            owner = owners.setdefault(digest, index)
            parent[root(owner)] = root(index)
    return [rows[root(index)]['name'] for index in range(len(rows))]


def public_session(source):
    match = re.fullmatch(r'(ave|vwg)-(\d{4})-\d{4}', source)
    return f'public-{match.group(2) if match else source}'


def paired_interval(per_sequence, sizes, clusters):
    by_cluster = defaultdict(lambda: np.zeros(2))
    for delta, size, cluster in zip(per_sequence, sizes, clusters, strict=True):
        by_cluster[cluster] += (delta, size)
    group_values = np.asarray(list(by_cluster.values()))
    rng = np.random.default_rng(20261118)
    draw = rng.integers(len(group_values), size=(3000, len(group_values)))
    totals = group_values[draw].sum(axis=1)
    return dict(delta=sum(per_sequence) / sum(sizes), n_gt=int(sum(sizes)),
                ci95=np.quantile(totals[totals[:, 1] > 0, 0] /
                                 totals[totals[:, 1] > 0, 1], [.025, .975]).tolist())


def analyze(rows, case):
    keys = sorted(rows[0]['predictions'])
    ranked = {key: [ranked_matches(row['predictions'][key], row['gt']) for row in rows]
              for key in keys}
    sizes = [len(row['gt']) for row in rows]
    clusters = clusters_for(rows, case)
    table = {key: metrics(value, sum(sizes)) | pose_metrics(rows, key) for key, value in ranked.items()}
    hit = {key: flags(value, rows, table[key]['at_fp1']['confidence'])
           for key, value in ranked.items()}
    hit25 = {key: flags(value, rows, .25) for key, value in ranked.items()}
    seeds = sorted({key.split('_')[0] for key in keys if key.startswith('mf')})
    contrasts, local, pooled = {}, {}, {}
    prefixes = seeds + ['sf_fuse']
    if case == 'controlled':
        full, same = ('AB', 'BA'), ('AA', 'BB')
        regions = [row['region'] for row in rows]
        mean_differences = defaultdict(list)
        for prefix in prefixes:
            all_names = [f'{prefix}_{mode}' for mode in (*full, *same)]
            contrasts[prefix] = dict(ap10=(sum(table[name]['ap10'] for name in all_names[:2]) -
                                           sum(table[name]['ap10'] for name in all_names[2:])) / 2)
            for label, hits in [('fp1', hit), ('at25', hit25)]:
                counts = {name: np.array([row.sum() for row in hits[name]]) for name in all_names}
                difference = (sum(counts[name] for name in all_names[:2]) -
                              sum(counts[name] for name in all_names[2:])) / 2
                contrasts[prefix][label] = paired_interval(difference, sizes, clusters)
                mean_differences[label].append(difference)
            local[prefix] = {}
            for region, side in [(0, 'A'), (1, 'B')]:
                eligible = [value == region for value in regions]
                same_mode = 'BB' if region == 0 else 'AA'
                for reference, mode in enumerate(('AB', 'BA') if region == 0 else ('BA', 'AB'), 1):
                    local[prefix][f'{side}_ref{reference}'] = {}
                    for label, hits in [('fp1', hit), ('at25', hit25)]:
                        delta = [(new[mask].astype(int) - old[mask]).sum() for new, old, mask in
                                 zip(hits[f'{prefix}_{mode}'], hits[f'{prefix}_{same_mode}'], eligible, strict=True)]
                        local[prefix][f'{side}_ref{reference}'][label] = paired_interval(
                            delta, [mask.sum() for mask in eligible], clusters)
                both_mode = 'AA' if region == 0 else 'BB'
                for reference, mode in enumerate(('AB', 'BA') if region == 0 else ('BA', 'AB'), 1):
                    local[prefix][f'{side}_remove_ref{3-reference}'] = {}
                    for label, hits in [('fp1', hit), ('at25', hit25)]:
                        delta = [(new[mask].astype(int) - old[mask]).sum() for new, old, mask in
                                 zip(hits[f'{prefix}_{mode}'], hits[f'{prefix}_{both_mode}'], eligible, strict=True)]
                        local[prefix][f'{side}_remove_ref{3-reference}'][label] = paired_interval(
                            delta, [mask.sum() for mask in eligible], clusters)
        pooled['complement_minus_redundant'] = dict(
            ap10=float(np.mean([contrasts[seed]['ap10'] for seed in seeds])),
            **{label: paired_interval(np.mean(values[:-1], axis=0), sizes, clusters)
               for label, values in mean_differences.items()})
        pooled['complement_mf_minus_sf_late'] = dict(
            ap10=float(np.mean([table[f'{seed}_{mode}']['ap10'] - table[f'sf_fuse_{mode}']['ap10']
                               for mode in full for seed in seeds])))
        for label, hits in [('fp1', hit), ('at25', hit25)]:
            differences = [[new.sum() - old.sum() for new, old in
                            zip(hits[f'{seed}_{mode}'], hits[f'sf_fuse_{mode}'], strict=True)]
                           for mode in full for seed in seeds]
            pooled['complement_mf_minus_sf_late'][label] = paired_interval(
                np.mean(differences, axis=0), sizes, clusters)
    else:
        mean_differences = defaultdict(list)
        for seed in seeds:
            full = f'{seed}_R1R2'
            controls = [f'{seed}_{mode}' for mode in ('KK', 'R1R1', 'R2R2')] + ['sf_K', 'sf_fuse_R1R2']
            for control in controls:
                contrasts[f'{full}-{control}'] = dict(
                    ap10=table[full]['ap10'] - table[control]['ap10'])
                for label, hits in [('fp1', hit), ('at25', hit25)]:
                    difference = [new.sum() - old.sum() for new, old in zip(hits[full], hits[control], strict=True)]
                    contrasts[f'{full}-{control}'][label] = paired_interval(difference, sizes, clusters)
                    mean_differences[(control.removeprefix(f'{seed}_'), label)].append(difference)
        for control in ('KK', 'R1R1', 'R2R2', 'sf_K', 'sf_fuse_R1R2'):
            after = [f'{seed}_{control}' if not control.startswith('sf') else control for seed in seeds]
            pooled[f'R1R2-{control}'] = dict(
                ap10=float(np.mean([table[f'{seed}_R1R2']['ap10'] - table[base]['ap10']
                                   for seed, base in zip(seeds, after, strict=True)])),
                **{label: paired_interval(np.mean(mean_differences[(control, label)], axis=0), sizes, clusters)
                   for label in ('fp1', 'at25')})
    return dict(n_groups=len(rows), n_clusters=len(set(clusters)), n_gt=sum(sizes),
                metrics=table, contrasts=contrasts, local=local, seed_mean_contrasts=pooled)


def controlled_audit():
    data = DATA_ROOT
    prov = json.loads((CLEAN / 'provenance.json').read_text())
    stems = {(kind, split): {path.stem for path in (data / folder / split / 'images').iterdir()}
             for kind, folder in [('private', 'weedataset'), ('public', 'public_weed_le256_split_1021')]
             for split in ('train', 'val', 'test')}
    public_sessions = {split: {public_session(source) for source in stems['public', split]}
                       for split in ('train', 'val', 'test')}
    audit, allowed = {}, set()
    for seq in sorted((CLEAN / 'test').glob('seq_*')):
        row = prov[seq.name]
        kind = 'public' if row['kind'].startswith('public') else 'private'
        source = row['source']
        splits = [split for split in ('train', 'val', 'test') if source in stems[kind, split]]
        audit[seq.name] = dict(kind=row['kind'], source=source, source_splits=splits)
        if (kind == 'public' and splits == ['test'] and
                all(public_session(source) not in public_sessions[split] for split in ('train', 'val'))):
            allowed.add(seq.name)
    return audit, allowed


def compare_fusion(first_rows, second_rows, first_summary, second_summary, case):
    for first, second in zip(first_rows, second_rows, strict=True):
        assert first['name'] == second['name'] and np.array_equal(first['gt'], second['gt'])
        sf_name = 'sf_K'
        np.testing.assert_allclose(first['predictions'][sf_name], second['predictions'][sf_name],
                                   rtol=0, atol=1e-6)
    sizes = [len(row['gt']) for row in first_rows]
    clusters = clusters_for(first_rows, case)
    modes = ('AB', 'BA') if case == 'controlled' else ('R1R2',)
    names = [f'mf{seed}_{mode}' for seed in (42, 43, 44) for mode in modes]
    output = dict(ap10=float(np.mean([first_summary['metrics'][name]['ap10'] -
                                      second_summary['metrics'][name]['ap10'] for name in names])))
    for label, key in [('fp1', 'at_fp1'), ('at25', 'at25')]:
        differences = []
        for name in names:
            first_items = [ranked_matches(row['predictions'][name], row['gt']) for row in first_rows]
            second_items = [ranked_matches(row['predictions'][name], row['gt']) for row in second_rows]
            first_hits = flags(first_items, first_rows, first_summary['metrics'][name][key]['confidence'])
            second_hits = flags(second_items, second_rows, second_summary['metrics'][name][key]['confidence'])
            differences.append([before.sum() - after.sum() for before, after in zip(first_hits, second_hits, strict=True)])
        output[label] = paired_interval(np.mean(differences, axis=0), sizes, clusters)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=HERE)
    args = parser.parse_args()
    cases = ('real_testset_blurred', 'real_testset', 'controlled')
    loaded = {case: pickle.loads((args.run_dir / f'{case}.pkl').read_bytes()) for case in cases}
    assert all(run['hashes'] == loaded[cases[0]]['hashes'] for run in loaded.values())
    blurred, original = (loaded[case]['rows'] for case in cases[:2])
    for edited, clean in zip(blurred, original, strict=True):
        assert edited['name'] == clean['name'] and np.array_equal(edited['gt'], clean['gt'])
        assert edited['hashes'] == clean['hashes']
    edited_patterns = Counter()
    for row in blurred:
        changed = ''.join(str(idx) for idx, digest in enumerate(row['hashes'])
                          if sha256((REAL / cases[0] / row['name'] / f'{idx+1:03d}.png').read_bytes())
                          .hexdigest() != digest)
        edited_patterns[changed] += 1
    audit, allowed = controlled_audit()
    controlled = loaded['controlled']['rows']
    summary = dict(checkpoints=loaded[cases[0]]['hashes'], case_audits={
        'real_blur_changed_slots': dict(edited_patterns),
        'real_distinct_source_frames': dict(Counter(len(set(row['hashes'])) for row in blurred)),
        'real_blur_registered_refs': dict(Counter(sum(t is not None for t in row['transforms']) for row in blurred)),
        'controlled_source_splits': dict(Counter(f"{row['kind']}/{','.join(row['source_splits'])}"
                                                 for row in audit.values())),
        'controlled_allowed_before_geometry_filter': len(allowed),
        'controlled_geometry_skips': loaded['controlled']['skipped']})
    for case in cases[:2]:
        rows = loaded[case]['rows']
        summary[case] = analyze(rows, case)
        summary[f'{case}_distinct'] = analyze([row for row in rows if len(set(row['hashes'])) == 3], case)
    public_rows = [row for row in controlled if row['name'] in allowed]
    summary['controlled_public_holdout'] = analyze(public_rows, 'controlled')
    avg_cases = [('real_testset_blurred', 'real_testset_blurred', blurred),
                 ('controlled_public_holdout', 'controlled', public_rows)]
    avg_paths = [(key, case, concat_rows, args.run_dir / f'avg_{case}.pkl')
                 for key, case, concat_rows in avg_cases]
    available = [item for item in avg_paths if item[3].is_file()]
    if available:
        summary['avg'] = {}
        for key, case, concat_rows, path in available:
            avg = pickle.loads(path.read_bytes())
            assert avg['case'] == case and avg['fusion'] == 'avg'
            assert avg['hashes']['sf'] == summary['checkpoints']['sf']
            avg_rows = [row for row in avg['rows'] if row['name'] in allowed or case != 'controlled']
            avg_summary = analyze(avg_rows, case)
            summary['avg'][key] = dict(checkpoints=avg['hashes'], analysis=avg_summary,
                                       concat_minus_avg=compare_fusion(concat_rows, avg_rows,
                                                                     summary[key], avg_summary, case))
    output = args.run_dir / 'summary.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(output)


if __name__ == '__main__':
    main()
