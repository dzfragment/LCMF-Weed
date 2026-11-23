"""Show the actual image triplet and the models' keyframe predictions."""

import argparse
import json
from pathlib import Path
import pickle

import cv2
from matplotlib import patheffects
from matplotlib.lines import Line2D
import numpy as np

from evaluate import DATA, HERE, prepare, read_sequence
from plot_keypoints import CANVAS, INK, MUTED, configure, match_points, plt


RESULTS = HERE / 'results'
DEFAULT_SEQUENCES = ('0032_f004645_x0905_y0141', '0023_f002667_x1264_y0197')
MODEL_NAMES = (
    ('yolo', '单帧 YOLOv8n-pose'),
    ('dis_yolo', 'YOLO + DIS 光流后融合'),
    ('main', 'Exp4 三帧主模型'),
)
PAD_MODEL_NAMES = (
    ('single', '单帧 P2L+PAD'),
    ('dis_pad', 'P2L+PAD + DIS 后融合'),
    ('fft_concat', '三帧 FFT+concat'),
    ('main', 'Exp4 三帧主模型'),
)
FRAME_NAMES = ('关键帧', '参考帧 1', '参考帧 2')


def render(index, blur_index, meta, bundle, blur, models, conf, radius, output):
    sequence = read_sequence(DATA / meta['sequence'])
    if sequence.manifest != meta:
        raise ValueError(f'Input files differ from the evaluation manifest for {sequence.name}')
    sample, mask, sharp = prepare(sequence, blur['kind'], blur['strength'])
    if mask != blur['changed_slots'][blur_index]:
        raise ValueError(f'Blurred frame assignment differs for {sequence.name}')
    np.testing.assert_allclose(sharp, blur['laplacian_variance'][blur_index], rtol=1e-6)
    frames = sample[0].permute(0, 2, 3, 1).numpy()

    fig = plt.figure(figsize=(13 if len(models) == 4 else 11, 9.08), facecolor=CANVAS)
    rows = fig.add_gridspec(2, 1, left=.072, right=.96, top=.805, bottom=.13, hspace=.285)
    inputs = rows[0].subgridspec(1, len(FRAME_NAMES), wspace=.073).subplots()
    outputs = rows[1].subgridspec(1, len(models), wspace=.073).subplots()
    is_original = blur['kind'] == 'original'
    title = '原图' if is_original else f'仅 {blur["strength"]} px 运动模糊'
    subtitle = f'{sequence.name}  ·  {meta["n_gt"]} 个标注点'
    if not is_original:
        subtitle += f'  ·  运动方向 {meta["motion_angle_deg"]}°'
    fig.text(.078, .948, f'{title}｜输入与茎点检测', fontsize=23, fontweight='bold', color=INK)
    fig.text(.078, .91, subtitle, color=MUTED, fontsize=11)
    fig.text(.078, .866, '模型输入', fontsize=15, fontweight='bold', color=INK)
    for slot, (ax, name) in enumerate(zip(inputs, FRAME_NAMES, strict=True)):
        ax.imshow(frames[slot], interpolation='hanning')
        ax.set_axis_off()
        applied = f'运动 {blur["strength"]} px' if mask[slot] else '原图'
        color = '#B85024' if mask[slot] else '#087F75'
        ax.set_title(name, loc='left', fontweight='bold', fontsize=11, pad=6)
        ax.set_title(applied, loc='right', color=color, fontsize=10, pad=6)

    fig.text(.078, .464, '关键帧检测结果', fontsize=15, fontweight='bold', color=INK)
    styles = (
        dict(marker='o', facecolors='none', edgecolors='#68F7CD', s=120, linewidths=1.6),
        dict(marker='+', color='#FFE174', s=75, linewidths=1.65),
        dict(marker='x', color='#FF8195', s=66, linewidths=1.6),
    )
    stroke = [patheffects.withStroke(linewidth=3, foreground='#17242D')]
    for ax, (key, label) in zip(outputs, models, strict=True):
        record = bundle['records'][key][index]
        gt, tp, fp = match_points(record, conf, radius)
        ax.imshow(frames[0], interpolation='hanning')
        ax.set_axis_off()
        ax.set_title(label, loc='left', fontweight='bold', fontsize=11, pad=6)
        for coords, style in zip((gt, tp, fp), styles, strict=True):
            overlay = ax.scatter(coords[:, 0], coords[:, 1], **style)
            overlay.set_path_effects(stroke)
        details = f'命中 {len(tp)}/{len(gt)}   ·   误报 {len(fp)}'
        if key in ('dis_yolo', 'dis_pad'):
            provenance = bundle['provenance'][index]['fused_source_slots']
            refs = np.count_nonzero((record['pred'][:, 4] >= conf) & (provenance > 0))
            details += f'   ·   {refs} 个预测来自参考帧'
        ax.text(0, -.031, details, transform=ax.transAxes, va='top', color=INK, fontsize=9)

    symbols = [
        Line2D([0], [0], marker='o', markersize=8, markeredgewidth=1.5,
               markerfacecolor='none', markeredgecolor='#087F75', linestyle='none'),
        Line2D([0], [0], marker='+', markersize=9, markeredgewidth=1.8,
               color='#A67311', linestyle='none'),
        Line2D([0], [0], marker='x', markersize=8, markeredgewidth=1.6,
               color='#C7415F', linestyle='none'),
    ]
    fig.legend(symbols, ['标注点', f'{radius:g} px 内匹配成功', '未匹配的预测'],
               ncol=3, loc='upper left', bbox_to_anchor=(.071, .073), frameon=False)
    fig.text(.078, .021, f'置信度 ≥ {conf:.2f}  ·  下排背景均为同一张关键帧',
             fontsize=9, color=MUTED)
    fig.savefig(output, dpi=198)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('original', 'motion_travel_4', 'motion_travel_10'),
                        default='motion_travel_4')
    parser.add_argument('--sequence', action='append', help='Choose sequence by directory name')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    configure()
    folder, models = 'visualizations', MODEL_NAMES
    if args.condition == 'original':
        base, flow = RESULTS, RESULTS / 'dis_yolo_late_fusion_current'
        folder = 'visualizations_original'
    elif args.condition == 'motion_travel_10':
        base = RESULTS / 'motion_travel_10_current'
        flow = RESULTS / 'pad_base_motion_travel_10_current'
        models = PAD_MODEL_NAMES
    else:
        base = RESULTS / 'motion_travel_4_current'
        flow = RESULTS / 'dis_yolo_motion_travel_4_current'
    summary = json.loads((flow / 'summary.json').read_text())
    manifest = json.loads((flow / 'manifest.json').read_text())
    base_manifest = json.loads((base / 'manifest.json').read_text())['sequences']
    base_indices = {row['sequence']: i for i, row in enumerate(base_manifest)}
    if any(base_manifest[base_indices[row['sequence']]] != row for row in manifest):
        raise ValueError('The two evaluations used different source frames or settings.')
    blur = next(c for c in json.loads((base / 'summary.json').read_text())['conditions']
                if c['name'] == args.condition)
    with (flow / f'{blur["name"]}_predictions.pkl').open('rb') as stream:
        bundle = pickle.load(stream)
    indices = {row['sequence']: i for i, row in enumerate(manifest)}
    output = flow / folder
    output.mkdir(exist_ok=True)
    for name in args.sequence or DEFAULT_SEQUENCES:
        if name not in indices:
            parser.error(f'Unknown sequence: {name}')
        i = indices[name]
        path = output / f'{name}.png'
        render(i, base_indices[name], manifest[i], bundle, blur, models,
               summary['conf'], summary['point_match_radius_px'], path)
        print(path)


if __name__ == '__main__':
    main()
