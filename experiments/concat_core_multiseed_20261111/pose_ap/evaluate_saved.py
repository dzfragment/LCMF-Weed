"""Score the original 15 test prediction sets with the pinned YOLO pose metrics."""

import csv
import hashlib
import json
import os
import pickle
import statistics
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CORE = HERE.parent
EXPERIMENTS = CORE.parent
FIRST_RUN = EXPERIMENTS / 'concat_ablation12_20261106'
os.environ['ULTRALYTICS_OFFLINE'] = 'true'
os.environ['YOLO_CONFIG_DIR'] = str(EXPERIMENTS / 'compute_profile_20261115/settings')

import ultralytics
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, kpt_iou


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score_pose(predictions, thresholds):
    matcher = SimpleNamespace(iouv=thresholds)
    scores, correct = [], []
    n_labels = 0
    for record in predictions.values():
        pred = torch.from_numpy(record['pred'])
        gt = torch.from_numpy(record['gt'])
        score = pred[:, 4]
        assert len(pred) <= 300 and (score >= .001).all()
        assert pred.isfinite().all() and gt.isfinite().all()
        gt_area = (gt[:, 2:4] - gt[:, :2]).prod(1) * .53
        assert (gt_area > 0).all()
        n_labels += len(gt)
        if not len(pred):
            continue
        gt_kpt = torch.cat((gt[:, 4:6], torch.ones(len(gt), 1)), dim=1).unsqueeze(1)
        oks = kpt_iou(gt_kpt, pred[:, 5:7].unsqueeze(1), area=gt_area, sigma=[1.0])
        matched = BaseValidator.match_predictions(matcher, torch.zeros(len(pred)), torch.zeros(len(gt)), oks)
        correct.append(matched.numpy())
        scores.append(score.numpy())

    scores = np.concatenate(scores)
    correct = np.concatenate(correct)
    ap = ap_per_class(correct, scores, np.zeros(len(scores)), np.zeros(n_labels))[5][0]
    assert np.all((ap >= 0) & (ap <= 1))
    return ap, len(scores), n_labels


def main():
    torch.set_num_threads(1)
    plan = json.loads((CORE / 'suite_plan.json').read_text())
    expected_gt = expected_sequences = None
    thresholds = torch.linspace(.5, .95, 10)
    rows = []
    for seed, arm in product((42, *plan['seeds']), plan['arms']):
        root = FIRST_RUN if seed == 42 else CORE / f'seed{seed}'
        folder = root / 'results' / arm
        meta = json.loads((folder / 'eval_test_last.json').read_text())
        assert meta['arm'] == arm and meta.get('seed', seed) == seed
        assert (meta['split'], meta['tag'], meta['weights']) == ('test', 'last', 'model_raw')
        assert sha256(folder / 'pad_stage3_last.pt') == meta['checkpoint_sha256']
        prediction_file = folder / 'preds_test_last.pkl'
        # The pickle is a trusted local artifact written by the original evaluation.
        with prediction_file.open('rb') as stream:
            predictions = pickle.load(stream)
        sequences = json.loads((folder / 'per_seq_test_last.json').read_text())
        assert len(predictions) == len(sequences) == meta['n_sequences']
        sequence_names = [r['sequence'] for r in sequences]
        if expected_sequences is None:
            expected_sequences = sequence_names
            expected_gt = [predictions[i]['gt'] for i in range(len(sequences))]
        assert sequence_names == expected_sequences
        for index, seq in enumerate(sequences):
            prediction = predictions[index]
            assert seq['sequence_index'] == index and len(prediction['gt']) == seq['n_gt']
            assert np.array_equal(prediction['gt'], expected_gt[index])
            assert np.count_nonzero(prediction['pred'][:, 4] >= .25) == seq['n_pred']
        ap, n_predictions, n_labels = score_pose(predictions, thresholds)
        assert n_labels == meta['n_gt']
        result = dict(arm=arm, model=arm, seed=seed, pose_ap50=float(ap[0]),
                      pose_ap75=float(ap[5]), pose_ap50_95=float(ap.mean()),
                      ap_by_threshold=ap.tolist(), n_predictions=n_predictions,
                      n_gt=n_labels, checkpoint_sha256=meta['checkpoint_sha256'],
                      predictions=str(prediction_file), predictions_sha256=sha256(prediction_file))
        rows.append(result)
        print(f'{arm} {seed}: {ap[0]:.6f} {ap.mean():.6f}', flush=True)

    aggregates = []
    for arm in dict.fromkeys(row['arm'] for row in rows):
        group = [row for row in rows if row['arm'] == arm]
        assert [row['seed'] for row in group] == [42, 43, 44]
        fields = ('pose_ap50', 'pose_ap75', 'pose_ap50_95')
        aggregate = dict(arm=arm, model=group[0]['model'], seeds=[42, 43, 44])
        for field in fields:
            values = [row[field] for row in group]
            aggregate[field] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
        aggregates.append(aggregate)
    protocol = dict(split='test', weights='stage3 last raw', input_size=1024,
                    n_sequences=575, n_gt=7633, score_floor=.001, max_predictions_per_image=300,
                    predicted_boxes_used=False, sigma=1., gt_area_factor=.53,
                    oks_thresholds=[round(v, 2) for v in thresholds.tolist()],
                    scorer=f'Ultralytics {ultralytics.__version__}',
                    prediction_source='original AP10 predictions, reused without model inference')
    output = dict(protocol=protocol, models=rows, aggregates=aggregates)
    (HERE / 'summary.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
