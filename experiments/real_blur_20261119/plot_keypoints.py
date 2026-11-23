"""Visualize 10 px recall, matched-point error and stored predictions."""

import argparse
import json
from pathlib import Path
import pickle

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib import patheffects
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

from evaluate import DATA, prepare, read_sequence


HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
INK = '#182431'
MUTED = '#456078'
CANVAS = '#F5F7FA'
GRID = '#DBE4ED'
MODELS = {
    'main': ('主模型 · 三帧', '#087F75', 'o', '-'),
    'yolo': ('YOLOv8n-pose · 单帧', '#B85024', 's', '--'),
}
EXAMPLES = {
    'original': '原图',
    'gaussian_8': '高斯 σ = 8 px',
    'motion_29': '运动核 29 px',
    'motion_43': '运动核 43 px',
}


def configure():
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        'font.family': [font_manager.FontProperties(fname=font_path).get_name(), 'DejaVu Sans'],
        'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.unicode_minus': False,
    })


def plot_curves(summary, *, scope="all"):
    baseline = summary["conditions"][0]["metrics"]["main"][scope]
    cohort = "Exp4 真实测试集" if scope == "all" else "Exp4 带帧号与坐标的目录子集"
    filename = "recall_error_curves.png" if scope == "all" else "current_recall_error_curves.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex='col', sharey='row',
                            facecolor=CANVAS)
    fig.subplots_adjust(left=.112, right=.91, top=.77, bottom=.125, hspace=.28, wspace=.20)
    fig.text(.112, .935, '模糊强度与关键点定位表现', fontsize=24, fontweight='bold')
    fig.text(.112, .893, f'{cohort}  ·  {baseline["n_sequences"]} 组  ·  '
             f'{baseline["n_gt"]} 个关键点  ·  置信度 ≥ {summary["conf"]:.2f}', color=MUTED)

    rows = (
        ('point_recall', '10 px 召回率', (0, .8 if scope == 'all' else .85), .2, PercentFormatter(1), '{:.1%}'),
        ('nPE_mean_px', '平均点误差（px）', (3.5, 6.5), .5, None, '{:.2f}'),
    )
    families = (
        ('gaussian', '高斯模糊', '高斯 σ（px）'),
        ('motion', '运动模糊', '运动核宽度（px）'),
    )
    panels = summary['conditions']
    for col, (family, name, xlabel) in enumerate(families):
        conditions = [row for row in panels if row['kind'] in ('original', family)]
        xs = [row['strength'] for row in conditions]
        axes[0, col].set_title(name, loc='left', fontsize=15, fontweight='bold', pad=10)
        axes[1, col].set_xlabel(xlabel, labelpad=6)
        for row, (metric, ylabel, ylim, tick, formatter, template) in enumerate(rows):
            ax = axes[row, col]
            ax.set_ylim(ylim)
            ax.set_xlim(-.05 * xs[-1], 1.12 * xs[-1])
            ax.set_xticks(xs, ['原图', *map(str, xs[1:])])
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=10)
            ax.yaxis.set_major_locator(MultipleLocator(tick))
            if formatter:
                ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(length=0, pad=5)
            ax.grid(axis='y', linewidth=.9, color=GRID)
            ax.set_axisbelow(True)
            for edge in ('top', 'right', 'left'):
                ax.spines[edge].set_visible(False)
            ax.spines['bottom'].set_color('#B9C8D6')
            endpoints = {key: conditions[-1]['metrics'][key][scope][metric] for key in MODELS}
            for key, (label, color, marker, style) in MODELS.items():
                ys = [c['metrics'][key][scope][metric] for c in conditions]
                ax.plot(xs, ys, color=color, marker=marker, linestyle=style, label=label,
                        linewidth=2.25, markersize=5.75)
                offset = 0 if row == 0 else (4 if ys[-1] == max(endpoints.values()) else -4)
                ax.annotate(template.format(ys[-1]), (xs[-1], ys[-1]),
                            xytext=(7, offset), textcoords='offset points',
                            va='center', color=color, fontweight='bold', fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc='upper left',
               bbox_to_anchor=(.106, .876), handlelength=2.6)
    fig.text(.112, .032, '平均点误差仅统计 10 px 内一对一匹配成功的点。', fontsize=9.5, color=MUTED)
    fig.savefig(RESULTS / filename, dpi=190)
    plt.close(fig)


def match_points(record, conf, radius):
    gt = record['gt'][:, 4:6]
    pred = record['pred'][record['pred'][:, 4] >= conf]
    claimed = np.zeros(len(gt), dtype=bool)
    matched = np.zeros(len(pred), dtype=bool)
    for i in np.argsort(-pred[:, 4]):
        if claimed.all():
            break
        dist = np.hypot(gt[:, 0] - pred[i, 5], gt[:, 1] - pred[i, 6])
        dist[claimed] = np.inf
        nearest = int(dist.argmin())
        if dist[nearest] <= radius:
            claimed[nearest] = True
            matched[i] = True
    return gt, pred[matched, 5:7], pred[~matched, 5:7]


def select_sequence(sequences, records, conditions, conf, radius):
    full_recall = np.array([[conditions[c]['metrics'][model]['all']['point_recall']
                             for model in MODELS] for c in EXAMPLES])
    candidates = []
    for i, row in enumerate(sequences):
        n = row['n_gt']
        if (not 8 <= n <= 18 or row['blurred_reference_slot'] is None
                or len(set(row['frame_sha256'])) != 3):
            continue
        recall = np.array([[len(match_points(records[c][model][i], conf, radius)[1]) / n
                            for model in MODELS] for c in EXAMPLES])
        candidates.append((np.abs(recall - full_recall).mean(), i))
    return min(candidates)[1]


def plot_examples(summary, manifest):
    conf, radius = summary['conf'], summary['point_match_radius_px']
    conditions = {row['name']: row for row in summary['conditions']}
    records = {}
    for condition in EXAMPLES:
        with (RESULTS / f'{condition}_predictions.pkl').open('rb') as stream:
            records[condition] = pickle.load(stream)

    i = select_sequence(manifest['sequences'], records, conditions, conf, radius)
    meta = manifest['sequences'][i]
    sequence = read_sequence(DATA / meta['sequence'])
    assert sequence.manifest == meta

    fig, axes = plt.subplots(2, 4, figsize=(14.1, 8.55), facecolor=CANVAS)
    fig.subplots_adjust(left=.091, right=.982, top=.785, bottom=.077, wspace=.055, hspace=.155)
    fig.text(.091, .933, '同一关键帧上的预测对比', fontsize=24, fontweight='bold')
    fig.text(.091, .893,
             f'{sequence.name}  ·  {meta["n_gt"]} 个标注点  ·  置信度 ≥ {conf:.2f}  ·  '
             f'运动模糊方向 {meta["motion_angle_deg"]}°', color=MUTED)
    gt_style = dict(marker='o', facecolors='none', edgecolors='#6DFBD1', s=105, linewidths=1.45)
    tp_style = dict(marker='+', color='#FFE274', s=70, linewidths=1.65)
    fp_style = dict(marker='x', color='#FF778D', s=58, linewidths=1.55)
    styles = (gt_style, tp_style, fp_style)
    stroke = [patheffects.withStroke(linewidth=2.9, foreground='#102028')]
    for col, (key, label) in enumerate(EXAMPLES.items()):
        condition = conditions[key]
        sample, mask, sharp = prepare(sequence, condition['kind'], condition['strength'])
        assert mask == condition['changed_slots'][i]
        np.testing.assert_allclose(sharp, condition['laplacian_variance'][i], rtol=1e-6)
        frame = sample[0][0].permute(1, 2, 0).numpy()
        axes[0, col].set_title(label, fontsize=12, fontweight='bold', loc='left', pad=7)
        for row, model in enumerate(MODELS):
            ax = axes[row, col]
            ax.imshow(frame, interpolation='hanning')
            coords = match_points(records[key][model][i], conf, radius)
            for points, style in zip(coords, styles, strict=True):
                markers = ax.scatter(points[:, 0], points[:, 1], **style)
                markers.set_path_effects(stroke)
            ax.set_axis_off()
            ax.text(.0, -.022, f'命中 {len(coords[1])}/{len(coords[0])}   ·   误报 {len(coords[2])}',
                    transform=ax.transAxes, fontsize=10, va='top', color=INK)

    for row, model in enumerate(MODELS):
        box = axes[row, 0].get_position()
        label, color, _, _ = MODELS[model]
        _, input_label = label.split(' · ')
        model_name = '主模型' if model == 'main' else 'YOLOv8n'
        fig.text(.018, (box.y0 + box.y1) / 2 + .016, model_name, fontsize=11,
                 color=color, fontweight='bold')
        fig.text(.018, (box.y0 + box.y1) / 2 - .012, input_label, fontsize=9, color=MUTED)
    markers = [
        Line2D([0], [0], marker='o', markersize=8, markeredgewidth=1.5,
               markerfacecolor='none', markeredgecolor='#087F75', linestyle='none'),
        Line2D([0], [0], marker='+', markersize=9, markeredgewidth=1.9,
               color='#A67311', linestyle='none'),
        Line2D([0], [0], marker='x', markersize=8, markeredgewidth=1.6,
               color='#C7415F', linestyle='none'),
    ]
    fig.legend(markers, ['标注点', '10 px 内匹配的预测', '未匹配的预测'],
               ncol=3, loc='upper left', bbox_to_anchor=(.082, .877), frameon=False)
    fig.text(.091, .025, f'示例的参考帧 {meta["blurred_reference_slot"]} 随关键帧加模糊，'
             '另一张参考帧保持原图。', fontsize=9.5, color=MUTED)
    fig.savefig(RESULTS / 'keypoint_examples.png', dpi=190)
    plt.close(fig)
    print(f'Example sequence: {sequence.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('all', 'current'), default='all')
    args = parser.parse_args()
    configure()
    summary = json.loads((RESULTS / 'summary.json').read_text())
    plot_curves(summary, scope=args.scope)
    if args.scope == 'all':
        manifest = json.loads((RESULTS / 'manifest.json').read_text())
        plot_examples(summary, manifest)


if __name__ == '__main__':
    main()
