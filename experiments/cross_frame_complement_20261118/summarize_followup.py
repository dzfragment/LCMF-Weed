"""Paired metrics and target-level counterfactual hits for the follow-up experiments."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
from zlib import crc32

import numpy as np

from evaluate_followup import DIRECTIONAL, HERE, placebo_modes
from pose_metrics import pose_metrics
from summarize import flags, metrics, paired_interval, ranked_matches


def load(run_dir, case):
    filename = 'followup_placebo_screened.pkl' if case == 'placebo' else f'followup_{case}.pkl'
    with (run_dir / filename).open('rb') as handle:
        result = pickle.load(handle)
    assert result['case'] == case
    return result


def scores(rows):
    keys = sorted(rows[0]['predictions'])
    assert all(set(row['predictions']) == set(keys) for row in rows)
    total_gt = sum(len(row['gt']) for row in rows)
    ranked = {key: [ranked_matches(row['predictions'][key], row['gt']) for row in rows] for key in keys}
    summary = {key: metrics(items, total_gt) | pose_metrics(rows, key) for key, items in ranked.items()}
    hits = {operating: {key: flags(items, rows, summary[key][operating]['confidence'])
                        for key, items in ranked.items()}
            for operating in ('at25', 'at_fp1')}
    return summary, hits


def mean_intervals(values, sizes, clusters):
    delta = np.mean(values, axis=0)
    return paired_interval(delta, sizes, clusters)


def audit_directional(rows, old_run):
    old_public = {row['name']: row for row in old_run['rows'] if row['kind'].startswith('public')}
    for row in rows:
        previous = old_public[row['name']]
        assert np.array_equal(row['gt'], previous['gt'])
        assert all(np.array_equal(after, before) for before, after in
                   zip(row['transforms'], previous['transforms'], strict=True))


def controlled_result(result, old_run=None):
    rows = result['rows']
    if old_run is not None:
        assert result['hashes'] == old_run['hashes']
        audit_directional(rows, old_run)
    summary, hitsets = scores(rows)
    elig = [row['eligible'] for row in rows]
    sizes = [int(keep.sum()) for keep in elig]
    n_all = sum(sizes)
    clusters = [row['cluster'] for row in rows]
    seeds = sorted(f'mf{seed}' for seed in result['hashes'] if seed != 'sf')
    regional = result['case'] == 'regional'
    joint = {'XY': (), 'YX': ()} if regional else {'XY': ('XK', 'KY'), 'YX': ('YK', 'KX')}
    redundant = ('XX', 'YY')
    arms = ('XX', 'XY', 'YX', 'YY', 'KK') if regional else tuple(DIRECTIONAL)
    both_singles = {arm for singles in joint.values() for arm in singles}
    relevant = tuple(arm for arm in arms if arm not in joint)
    if regional:
        geometry = [row['gt'][keep, 4 + crc32(row['name'].encode()) % 2] < 512
                    for row, keep in zip(rows, elig, strict=True)]
    primary = {}
    for operating, hits in hitsets.items():
        rates, detections = {}, {}
        for model in [*seeds, 'sf']:
            states = {arm: [state[keep] for state, keep in zip(hits[f'{model}_{arm}'], elig, strict=True)]
                      for arm in arms}
            per_seq = defaultdict(list)
            for idx, _ in enumerate(rows):
                state = {arm: mask[idx] for arm, mask in states.items()}
                mix_hit = sum(state[arm].astype(float) for arm in joint) / 2
                same_hit = sum(state[arm].astype(float) for arm in redundant) / 2
                per_seq['orthogonal_minus_same'].append((mix_hit - same_hit).sum())
                for arm in arms:
                    per_seq[f'hits_{arm}'].append(state[arm].sum())
                if regional:
                    continue
                no_redundant = ~np.logical_or.reduce([state[arm] for arm in redundant])
                no_controls = ~np.logical_or.reduce([state[arm] for arm in relevant])
                for arm, singles in joint.items():
                    for single in singles:
                        per_seq[f'{arm}_minus_{single}'].append(state[arm].sum() - state[single].sum())
                    no_single = ~np.logical_or.reduce([state['KK'], *[state[single] for single in singles]])
                    per_seq[f'{arm}_corresponding_singles_miss'].append(no_single.sum())
                    per_seq[f'{arm}_only_after_both'].append((no_single & state[arm]).sum())
                    per_seq[f'{arm}_strict_joint_only'].append((no_controls & state[arm]).sum())
                    per_seq[f'{arm}_strict_joint_only_and_sf_full_misses'].append(
                        (no_controls & state[arm] & ~hits[f'sf_{arm}'][idx][elig[idx]]).sum())
                both_mix = state['XY'] & state['YX']
                per_seq['both_orders_strict_joint_only'].append((no_controls & both_mix).sum())
                per_seq['orthogonal_only'].append((no_redundant & (state['XY'] | state['YX'])).sum())
                per_seq['same_only'].append((~(state['XY'] | state['YX']) & ~no_redundant).sum())
                per_seq['pure_single_union'].append(np.logical_or.reduce([state['KK'], *[state[arm] for arm in both_singles]]).sum())
            rates[model] = {key: float(sum(counts) / n_all) for key, counts in per_seq.items()}
            rates[model]['orthogonal_minus_same'] = paired_interval(per_seq['orthogonal_minus_same'], sizes, clusters)
            detections[model] = per_seq
        primary[operating] = dict(
            rates_by_model=rates,
            orthogonal_minus_same_mean=mean_intervals([detections[seed]['orthogonal_minus_same'] for seed in seeds], sizes, clusters),
            orthogonal_minus_same_sf=rates['sf']['orthogonal_minus_same'],
            joint_target_rates_mean={key: mean_intervals([detections[seed][key] for seed in seeds], sizes, clusters)
                                     for key in detections[seeds[0]] if key != 'orthogonal_minus_same'},
        )
        if regional:
            local = {}
            for is_first, label, control in [(True, 'A', 'YY'), (False, 'B', 'XX')]:
                masks = [region == is_first for region in geometry]
                ref_modes = ('XY', 'YX') if is_first else ('YX', 'XY')
                for ref, mode in enumerate(ref_modes, 1):
                    delta = []
                    for model in seeds:
                        delta.append([after[eligible][region].sum() - before[eligible][region].sum()
                                      for after, before, eligible, region in zip(hits[f'{model}_{mode}'],
                                          hits[f'{model}_{control}'], elig, masks, strict=True)])
                    local[f'{label}_ref{ref}'] = mean_intervals(delta, [mask.sum() for mask in masks], clusters)
            primary[operating]['regional_recovery'] = local
    return dict(n_sequences=len(rows), n_gt=sum(len(row['gt']) for row in rows),
                n_eligible=n_all, n_sessions=len(set(clusters)), metrics=summary, paired=primary,
                skipped=result['skipped'], hashes=result['hashes'])


def audit_placebo(rows, old_run, geometry):
    models = sorted(f'mf{seed}' for seed in old_run['hashes'] if seed != 'sf')
    varied = {row['name']: row for row in rows}
    old_by_name = {row['name']: row for row in old_run['rows']}
    assert set(varied) == set(old_by_name)
    bad_pairs = {(pair['recipient'], pair['donor']) for pair in geometry['plausible_matches']}
    assert geometry['final_pairs_screened'] == 3 * len(rows)
    for repeat in range(3):
        assert sorted(row['donor_names'][repeat] for row in rows) == sorted(varied)
    for row in rows:
        old = old_by_name[row['name']]
        assert np.array_equal(row['gt'], old['gt']) and row['hashes'] == old['hashes']
        assert len(set(row['donor_names'])) == 3
        for donor_name in row['donor_names']:
            donor = varied[donor_name]
            assert not (set(row['hashes']) & set(donor['hashes'])) and row['pattern'] == donor['pattern']
            assert (row['name'], donor_name) not in bad_pairs
            assert row['cluster'] != donor['cluster']
        for model in models:
            np.testing.assert_allclose(row['predictions'][f'{model}_MATCH'], old['predictions'][f'{model}_R1R2'])
            np.testing.assert_allclose(row['predictions'][f'{model}_KK'], old['predictions'][f'{model}_KK'])


def placebo_result(result, old_run):
    rows = result['rows']
    assert all(old_run['hashes'][seed] == value for seed, value in result['hashes'].items())
    audit_placebo(rows, old_run, result['geometry_audit'])
    old_by_name = {row['name']: row['predictions'] for row in old_run['rows']}
    models = sorted(f'mf{seed}' for seed in result['hashes'])
    extra = [f'{model}_{arm}' for model in models for arm in ('R1R1', 'R2R2')] + ['sf_K', 'sf_fuse_R1R2']
    rows = [row | {'predictions': row['predictions'] | {key: old_by_name[row['name']][key] for key in extra}}
            for row in rows]
    summary, hitsets = scores(rows)
    sizes = [len(row['gt']) for row in rows]
    clusters = [row['cluster'] for row in rows]
    parts = {arm: [arm] for arm in ('MATCH', 'KK', 'R1R1', 'R2R2', 'sf_K', 'sf_fuse_R1R2')}
    parts.update({prefix: [arm for arm in placebo_modes() if arm.startswith(f'{prefix}_')]
                  for prefix in ('WRONG1', 'WRONG2', 'WRONGBOTH')})

    def compare(high, low):
        def keys(part, model):
            return parts[part] if part.startswith('sf_') else [f'{model}_{arm}' for arm in parts[part]]

        primary = dict(ap10=np.mean([np.mean([summary[key]['ap10'] for key in keys(high, model)]) -
                                     np.mean([summary[key]['ap10'] for key in keys(low, model)]) for model in models]))
        for operating, hits in hitsets.items():
            count = {arm: np.array([row.sum() for row in masks]) for arm, masks in hits.items()}
            primary[operating] = mean_intervals([
                np.mean([count[key] for key in keys(high, model)], axis=0) -
                np.mean([count[key] for key in keys(low, model)], axis=0) for model in models], sizes, clusters)
        return primary

    contrasts = {f'MATCH-{low}': compare('MATCH', low) for low in parts if low != 'MATCH'}
    contrasts['WRONGBOTH-KK'] = compare('WRONGBOTH', 'KK')
    return dict(n_sequences=len(rows), n_gt=sum(sizes), n_clusters=len(set(clusters)),
                hashes=result['hashes'], metrics=summary, contrasts=contrasts,
                donors={row['name']: row['donor_names'] for row in rows}, geometry_audit=result['geometry_audit'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=HERE)
    args = parser.parse_args()
    output = {}
    for case, ref, summarize in [('directional', 'controlled', controlled_result),
                                 ('interleaved', 'controlled', controlled_result),
                                 ('placebo', 'real_testset_blurred', placebo_result)]:
        with (HERE / f'{ref}.pkl').open('rb') as file:
            previous = pickle.load(file)
        output[case] = summarize(load(args.run_dir, case), previous)
    summary_path = args.run_dir / 'followup_summary.json'
    temporary = summary_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    temporary.replace(summary_path)
    print(summary_path)


if __name__ == '__main__':
    main()
