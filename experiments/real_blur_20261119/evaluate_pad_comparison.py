"""Compare the four Exp4 methods that share the trained single-frame PAD base."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch

from evaluate import (ABLATION, ARMS, DATA, HERE, SIZE, architecture_flags, args_for,
                      assert_checkpoint, digest, load_evaluators, prepare, read_sequence)
from common import SF
from evaluate_dis_fusion import (CONF, FLOW_SIZE, MAX_POINTS, POINT_NMS_PX, ROUND_TRIP_PX,
                                 fuse, summarize)


RESULTS = HERE / 'results'
EXTRA_RUNS = ('motion_travel_4_current', 'gaussian_4_motion_4_current',
              'gaussian_4_motion_7_current', 'gaussian_12_motion_30_current')
ARMS_TO_TEST = {'fft_concat': '02_no_global_no_dcn', 'main': '01_main'}


def read_json(path):
    return json.loads(path.read_text())


def load_plan(additional_runs=()):
    base = read_json(RESULTS / 'manifest.json')['sequences']
    indices = [i for i, seq in enumerate(base) if seq['cohort'] == 'current']
    manifest = [base[i] for i in indices]
    plan = [(condition, indices) for condition in read_json(RESULTS / 'summary.json')['conditions']]
    subset_indices = list(range(len(manifest)))
    folders = [RESULTS / name for name in EXTRA_RUNS]
    for folder in (*folders, *additional_runs):
        if read_json(folder / 'manifest.json')['sequences'] != manifest:
            raise ValueError(f'Sequence manifest mismatch in {folder}')
        condition, = read_json(folder / 'summary.json')['conditions']
        plan.append((condition, subset_indices))
    return manifest, plan


def load_pad_models(training, device):
    from pad_core import PADSingleFrame

    checksum = digest(SF)
    checkpoint = torch.load(SF, map_location='cpu', weights_only=False)
    single = PADSingleFrame(checkpoint['base_model'], checkpoint['head_cfg'])
    weights = checkpoint['model']
    state = weights.float().state_dict() if isinstance(weights, torch.nn.Module) else weights
    single.load_state_dict(state, strict=True)
    single = single.float().to(device).eval()
    single_meta = dict(checkpoint=str(SF), checkpoint_sha256=checksum,
                       parameters=sum(p.numel() for p in single.parameters()),
                       base_model=checkpoint['base_model'], head_cfg=checkpoint['head_cfg'],
                       weights='single-frame initialization checkpoint')
    models = {}
    config = dict(single=single_meta, dis_pad=single_meta)
    arms = {arm['name']: arm for arm in ARMS}
    for label, arm_name in ARMS_TO_TEST.items():
        arm = arms[arm_name]
        folder = ABLATION / 'results' / arm_name
        init = read_json(folder / 'init.json')
        if Path(init['sf']).resolve() != SF.resolve() or init['sf_sha256'] != checksum:
            raise ValueError(f'{arm_name} was trained with a different base')
        path = folder / 'pad_stage3_last.pt'
        metadata = assert_checkpoint(path, arm, epoch=250)
        model, _ = training.load_ckpt(str(path), args_for(architecture_flags(arm)),
                                     device, prefer_ema=False)
        models[label] = model.float().eval()
        config[label] = dict(checkpoint=str(path), checkpoint_sha256=digest(path),
                             parameters=sum(p.numel() for p in model.parameters()),
                             arm=arm, metadata=metadata, weights='stage3 last raw',
                             base_checkpoint_sha256=checksum, seed=init['config']['seed'])
    return single, models, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS / 'pad_base_comparison_current')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--condition', action='append', help='Run only a named condition; may be repeated')
    parser.add_argument('--additional-run', type=Path, action='append', default=[],
                        help='Add a saved one-condition input manifest to the evaluation plan')
    args = parser.parse_args()
    manifest, plan = load_plan(args.additional_run)
    if args.condition:
        unknown = set(args.condition) - {c['name'] for c, _ in plan}
        if unknown:
            parser.error(f'Unknown condition(s): {sorted(unknown)}')
        plan = [(c, ids) for c, ids in plan if c['name'] in args.condition]
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=False)
    training, _, pad_infer, _, metrics, score_pose, dump, atomic_bytes = load_evaluators(args.out)
    single, multiframe, weights = load_pad_models(training, 'cuda:0')
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    result = dict(data=str(DATA), n_sequences=len(manifest), n_gt=sum(s['n_gt'] for s in manifest),
                  scope='current', resolution=SIZE, batch_size=args.batch_size, conf=CONF,
                  conf_floor=.001, max_candidates=MAX_POINTS, point_match_radius_px=10,
                  weights=weights, algorithm=dict(flow='OpenCV DIS PRESET_FAST', flow_resolution=FLOW_SIZE,
                          round_trip_max_px=ROUND_TRIP_PX, point_nms_radius_px=POINT_NMS_PX,
                          source_scores='unchanged PAD confidence'),
                  conditions=[])
    dump(args.out / 'manifest.json', manifest)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        sequences = list(executor.map(read_sequence, (DATA / row['sequence'] for row in manifest)))
        if [seq.manifest for seq in sequences] != manifest:
            raise ValueError('Source sequence manifest changed')
        print(f"{result['n_sequences']} sequences, {result['n_gt']} labeled keypoints", flush=True)
        for condition, indices in plan:
            name, kind, strength = (condition[key] for key in ('name', 'kind', 'strength'))
            records = dict(single={}, dis_pad={}, **{key: {} for key in multiframe})
            provenance = []
            rows = executor.map(lambda seq: prepare(seq, kind, strength), sequences)
            for batch in batched(rows, args.batch_size):
                samples, masks, scores = zip(*batch, strict=True)
                frames, targets = training.MultiFrameDataset.collate_fn(samples)
                views = [pad_infer(single, [(frames[:, slot].contiguous(), targets)], SIZE)
                         for slot in range(frames.shape[1])]
                fused = {key: pad_infer(model, [(frames, targets)], SIZE)
                         for key, model in multiframe.items()}
                start = len(provenance)
                for j, (mask, sharp) in enumerate(zip(masks, scores, strict=True)):
                    i = start + j
                    origin = indices[i]
                    if mask != condition['changed_slots'][origin]:
                        raise ValueError(f'Blur mask mismatch for {sequences[i].name}')
                    np.testing.assert_allclose(sharp, condition['laplacian_variance'][origin], rtol=1e-6)
                    keyframe = views[0][j]
                    arrangements = [view[j]['pred'] for view in views]
                    flow_pred, slots, info = fuse(frames[j], arrangements, dis)
                    records['single'][i] = keyframe
                    records['dis_pad'][i] = dict(pred=flow_pred, gt=keyframe['gt'])
                    for key, preds in fused.items():
                        if not np.array_equal(preds[j]['gt'], keyframe['gt']):
                            raise ValueError(f'GT mismatch for {sequences[i].name} in {key}')
                        records[key][i] = preds[j]
                    provenance.append(dict(sequence=sequences[i].name, views=arrangements,
                                           fused_source_slots=slots, **info))
            if len(provenance) != len(sequences):
                raise ValueError(f'Incomplete evaluation: {name}')
            scores = {key: summarize(rows, metrics, score_pose) for key, rows in records.items()}
            result['conditions'].append(dict(name=name, kind=kind, strength=strength, metrics=scores,
                selected_conf25_by_slot=np.sum([p['selected_conf25_by_slot'] for p in provenance], axis=0).tolist()))
            atomic_bytes(args.out / f'{name}_predictions.pkl', pickle.dumps(dict(records=records, provenance=provenance)))
            dump(args.out / 'summary.json', result)
            print(name, ' '.join(f'{k}: R10={m["point_recall"]:.3f} AP10={m["point_ap_10px"]:.3f}'
                                 for k, m in scores.items()), flush=True)


if __name__ == '__main__':
    main()
