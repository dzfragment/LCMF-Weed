"""Export the blur sweep as a tidy CSV and a two-metric comparison figure."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


MODELS = {
    'main': ('Exp4 01_main · 3 frames', '#087F75', 'o'),
    'yolo': ('Original YOLOv8n-pose · 1 frame', '#B85024', 's'),
}
COLS = ('condition', 'blur_family', 'strength_px', 'model', 'scope', 'n_sequences', 'n_gt',
        'point_ap_10px', 'ap10_change_pp', 'recall_at_fp1', 'fp1_conf', 'point_recall',
        'point_precision', 'nPE_mean_px', 'nPE_median_px', 'fp_per_image',
        'n_pred', 'n_tp', 'point_ap_20px', 'pose_ap50', 'pose_ap75', 'pose_ap50_95')


def export_csv(conditions, path):
    baseline = conditions[0]['metrics']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS)
        writer.writeheader()
        for condition in conditions:
            for model, scopes in condition['metrics'].items():
                for scope, metrics in scopes.items():
                    row = {field: metrics[field] for field in COLS if field in metrics}
                    fixed_fp = metrics['recall_at_fp1']
                    gain = metrics['point_ap_10px'] - baseline[model][scope]['point_ap_10px']
                    row.update(condition=condition['name'], blur_family=condition['kind'],
                               strength_px=condition['strength'], model=model, scope=scope,
                               ap10_change_pp=100 * gain, recall_at_fp1=fixed_fp['point_recall'],
                               fp1_conf=fixed_fp['conf'])
                    writer.writerow(row)


def export_figure(conditions, path, n_sequences, n_gt):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelcolor': '#252D36', 'axes.titleweight': 'bold',
                         'xtick.color': '#425161', 'ytick.color': '#425161'})
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey='row')
    figure.patch.set_facecolor('#F7F8FA')
    figure.subplots_adjust(left=.10, right=.97, top=.78, bottom=.12, hspace=.24, wspace=.16)
    figure.text(.10, .95, 'Keypoint accuracy under blur', fontsize=23, fontweight='bold', color='#17212D')
    figure.text(.10, .913, f'Exp4 real test set · {n_sequences} sequences · {n_gt} labeled keypoints',
                fontsize=11, color='#425161')

    fields = (
        ('Point AP10', .65, lambda m: m['point_ap_10px']),
        ('Recall at ≤1 FP / image', .45, lambda m: m['recall_at_fp1']['point_recall']),
    )
    families = (
        ('gaussian', 'Gaussian blur', 'Gaussian σ (px)'),
        ('motion', 'Motion blur', 'Motion kernel width (px)'),
    )
    for col, (kind, title, xlabel) in enumerate(families):
        series = [c for c in conditions if c['kind'] in ('original', kind)]
        strength = [c['strength'] for c in series]
        axes[0, col].set_title(title, loc='left', pad=9)
        for row, (ylabel, ymax, get_metric) in enumerate(fields):
            ax = axes[row, col]
            ax.set_facecolor('white')
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#DCE3E9', linewidth=.8)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color('#B2BDC8')
            ax.tick_params(length=0, pad=6)
            ax.set_xticks(strength, ['0 (original)', *map(str, strength[1:])])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_locator(MultipleLocator(.1))
            ax.set_ylim(0, ymax)
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=10)
            if row == 1:
                ax.set_xlabel(xlabel, labelpad=9)
            for model, (label, color, marker) in MODELS.items():
                ax.plot(strength, [get_metric(c['metrics'][model]['all']) for c in series],
                        label=label, color=color, marker=marker, linewidth=2.5, markersize=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper left', bbox_to_anchor=(.09, .885), ncol=2, frameon=False)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    data = json.loads((args.results / 'summary.json').read_text())
    manifest = json.loads((args.results / 'manifest.json').read_text())
    conditions = data['conditions']
    if len(conditions) != 9:
        raise ValueError('The blur sweep is incomplete')
    export_csv(conditions, args.results / 'metrics.csv')
    export_figure(conditions, args.results / 'curves.png', manifest['n_sequences'], manifest['n_gt'])


if __name__ == '__main__':
    main()
