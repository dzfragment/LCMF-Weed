"""Audit and score the controlled degradations of the model's own official test split."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from evaluate import HERE, digest, model_paths
from summarize_followup import controlled_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=HERE)
    args = parser.parse_args()
    root = args.run_dir
    manifest = root / 'official_clean_manifest.json'
    provenance = json.loads(manifest.read_text())['sequences']
    with (HERE / 'standard_test_predictions.pkl').open('rb') as handle:
        reference = {row['name']: row['gt'] for row in pickle.load(handle)['rows']}
    result = {}
    for case, fusion in [('regional', 'concat'), ('interleaved', 'concat'),
                         ('directional', 'concat'), ('regional', 'avg')]:
        key = f'{case}_avg' if fusion == 'avg' else case
        with (root / f'official_{key}.pkl').open('rb') as handle:
            run = pickle.load(handle)
        rows = run['rows']
        assert run['case'] == case and run['fusion'] == fusion and not run['skipped']
        assert run['official_manifest_sha256'] == digest(manifest)
        paths, sf_path = model_paths([42, 43, 44], fusion)
        assert run['hashes'] == ({seed: digest(path) for seed, path in paths.items()} | {'sf': digest(sf_path)})
        assert len(rows) == len(reference) == 575 and {row['name'] for row in rows} == set(reference) == set(provenance)
        assert sum(len(row['gt']) for row in rows) == 7633
        for row in rows:
            record = provenance[row['name']]
            np.testing.assert_array_equal(row['gt'], reference[row['name']])
            origins = np.asarray(record['origins'])
            np.testing.assert_array_equal([matrix[:2, 2] for matrix in row['transforms']], origins[1:] - origins[0])
            assert row['cluster'] == record['group'] and row['kind'] == record['kind']
        result[key] = controlled_result(run)
        result[key]['n_public'] = sum(row['kind'] == 'public' for row in rows)
        result[key]['n_private'] = sum(row['kind'] == 'private' for row in rows)
        print(f'{key}: 575 original test sequences, {result[key]["n_eligible"]} eligible targets', flush=True)
    output = root / 'official_mechanisms_summary.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
