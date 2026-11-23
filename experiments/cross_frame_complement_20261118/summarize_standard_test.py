"""Official test reference-ablation metrics, paired by original recording group."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import numpy as np

from evaluate import HERE, digest, model_paths
from summarize import paired_interval
from summarize_followup import scores


def verify_original(rows, hashes):
    paths, _ = model_paths([42, 43, 44], 'concat')
    reference = json.loads((HERE.parent / 'concat_core_multiseed_20261111/pose_ap/summary.json').read_text())
    pose_by_seed = {str(row['seed']): row for row in reference['models'] if row['arm'] == '01_main'}
    comparisons = {}
    for seed, path in paths.items():
        expected = json.loads((path.parent / 'eval_test_last.json').read_text())
        assert hashes[seed] == expected['checkpoint_sha256'] == digest(path)
        with (path.parent / 'preds_test_last.pkl').open('rb') as handle:
            saved = pickle.load(handle)
        assert len(saved) == len(rows)
        max_error = 0.
        for idx, row in enumerate(rows):
            np.testing.assert_array_equal(row['gt'][:, 4:], saved[idx]['gt'][:, 4:])
            np.testing.assert_allclose(row['gt'][:, :4], saved[idx]['gt'][:, :4], atol=.002, rtol=0)
            pred = row['predictions'][f'mf{seed}_full']
            np.testing.assert_allclose(pred, saved[idx]['pred'], atol=.015, rtol=1e-5)
            if len(pred):
                max_error = max(max_error, float(np.abs(pred - saved[idx]['pred']).max()))
        comparisons[seed] = dict(expected_ap10=expected['point_ap_10px'],
                                 expected_recall25=expected['point_recall'],
                                 expected_mean_error_px=expected['nPE_mean_px'],
                                 expected_pose_ap50_95=pose_by_seed[seed]['pose_ap50_95'],
                                 max_prediction_diff=max_error)
    return comparisons


def evaluate_rows(rows, seeds):
    table, matches = scores(rows)
    groups = [row['group'] for row in rows]
    sizes = [len(row['gt']) for row in rows]
    parts = {mode: [mode] for mode in ('full', 'ref1_twice', 'ref2_twice', 'key_tripled', 'sf_key')}
    parts['wrong_both'] = [f'wrong_both_{idx}' for idx in (1, 2, 3)]

    def keys(arm, seed):
        return ['sf_key'] if arm == 'sf_key' else [f'mf{seed}_{mode}' for mode in parts[arm]]

    means, contrasts = {}, {}
    for arm in parts:
        by_seed = [[table[key] for key in keys(arm, seed)] for seed in (seeds if arm != 'sf_key' else ['sf'])]
        means[arm] = {metric: float(np.mean([item[metric] for entries in by_seed for item in entries]))
                      for metric in ('ap10', 'pose_ap50', 'pose_ap75', 'pose_ap50_95')}
        for op in matches:
            means[arm][op] = {metric: float(np.mean([item[op][metric] for entries in by_seed for item in entries]))
                              for metric in ('recall', 'precision', 'fp_per_group', 'mean_error_px')}
        if arm != 'full':
            contrasts[arm] = {metric: means['full'][metric] - means[arm][metric]
                              for metric in ('ap10', 'pose_ap50', 'pose_ap75', 'pose_ap50_95')}
            contrasts[arm]['mean_error_px_at25'] = means['full']['at25']['mean_error_px'] - means[arm]['at25']['mean_error_px']
            for op, values in matches.items():
                by_model = []
                for seed in seeds:
                    full = np.asarray([hits.sum() for hits in values[f'mf{seed}_full']])
                    control = np.mean([[hits.sum() for hits in values[key]] for key in keys(arm, seed)], axis=0)
                    by_model.append(full - control)
                contrasts[arm][op] = paired_interval(np.mean(by_model, axis=0), sizes, groups)
    return dict(n_sequences=len(rows), n_gt=sum(sizes), n_groups=len(set(groups)),
                means=means, full_minus=contrasts, models_and_repeats=table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, default=HERE / 'standard_test_predictions.pkl')
    parser.add_argument('--output', type=Path, default=HERE / 'standard_test_summary.json')
    args = parser.parse_args()
    with args.predictions.open('rb') as handle:
        run = pickle.load(handle)
    rows, hashes = run['rows'], run['model_hashes']
    assert len(rows) == 575 and sum(len(row['gt']) for row in rows) == 7633
    comparisons = verify_original(rows, hashes)
    seeds = sorted(seed for seed in hashes if seed != 'sf')
    by_name = {row['name']: row for row in rows}
    cross_kind = Counter()
    for row in rows:
        for donor in row['donors']:
            source = by_name[donor]
            assert row['group'] != source['group'] and row['blur'][1:] == source['blur'][1:]
            cross_kind['cross' if source['kind'] != row['kind'] else 'same'] += 1
    report = dict(test_root=run['test_root'], model_hashes=hashes, manifest_sha256=run['manifest_sha256'],
                  replay_coverage=Counter(f"{row['kind']}/{sum(row['replay_identical'])} identical frames"
                                          for row in rows), cross_kind=cross_kind,
                  donors={row['name']: row['donors'] for row in rows}, original_prediction_check=comparisons)
    report['all'] = evaluate_rows(rows, seeds)
    for seed, original in comparisons.items():
        full = report['all']['models_and_repeats'][f'mf{seed}_full']
        assert abs(full['ap10'] - original['expected_ap10']) < .0002
        assert abs(full['at25']['recall'] - original['expected_recall25']) < .0002
        assert abs(full['at25']['mean_error_px'] - original['expected_mean_error_px']) < .00001
        assert abs(full['pose_ap50_95'] - original['expected_pose_ap50_95']) < .00001
    report['key_blur_with_clear_ref'] = evaluate_rows([row for row in rows if row['blur'][0]
                                                       and not all(row['blur'][1:])], seeds)
    report['public_only'] = evaluate_rows([row for row in rows if row['kind'] == 'public'], seeds)
    report['verified_blur_flags_only'] = evaluate_rows([row for row in rows if all(row['replay_identical'])
                                                       and all(all(by_name[donor]['replay_identical'])
                                                               for donor in row['donors'])], seeds)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
