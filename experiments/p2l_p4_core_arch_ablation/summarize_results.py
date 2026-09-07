#!/usr/bin/env python3
"""Collect the best Stage-3 validation metrics into Markdown and CSV reports."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = Path(__import__('os').environ.get('RESULTS_ROOT', ROOT / 'results'))
EXPERIMENTS = [
    ('00_full_fft_dcn_spatial', 'FFT + DCN + Spatial'),
    ('01_remove_fft', '去除 FFT'),
    ('02_remove_dcn', '去除 DCN'),
    ('03_remove_spatial', '去除 Spatial（平均融合）'),
    ('05_ablate_concat_fusion', 'Feature FFT + DCN + Concat'),
    ('06_ablate_dav2_single', 'Feature FFT + 单阶段 DAV2 + Spatial'),
]


def as_float(row, key, default=float('nan')):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def read_best(exp_id):
    path = RESULTS_ROOT / exp_id / 'stage3_log.csv'
    if not path.exists():
        return {'status': 'missing', 'log': str(path)}
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {'status': 'empty', 'log': str(path)}
    valid = [r for r in rows if r.get('PoseAP', '').strip()]
    if not valid:
        return {'status': 'no_pose_ap', 'log': str(path)}
    best = max(valid, key=lambda r: as_float(r, 'PoseAP'))
    return {
        'status': 'ok',
        'epoch': int(float(best['epoch'])),
        'PoseAP': as_float(best, 'PoseAP'),
        'BoxAP': as_float(best, 'BoxAP'),
        'val_loss': as_float(best, 'val_loss'),
        'log': str(path),
    }


def main():
    records = []
    for exp_id, label in EXPERIMENTS:
        item = {'experiment': exp_id, 'label': label, **read_best(exp_id)}
        records.append(item)

    control = next((r for r in records if r['experiment'].startswith('00_') and r['status'] == 'ok'), None)
    for r in records:
        r['delta_pose_ap'] = (
            r['PoseAP'] - control['PoseAP']
            if control and r['status'] == 'ok' else float('nan')
        )

    csv_path = ROOT / 'results_tables' / 'RESULTS_SUMMARY.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fields = ['experiment', 'label', 'status', 'epoch', 'BoxAP', 'PoseAP', 'delta_pose_ap', 'val_loss', 'log']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    report_path = ROOT / 'EXPERIMENT_REPORT.md'
    lines = [
        '# P2L-P4 核心架构消融实验报告', '',
        '## 当前结果', '',
        '| 实验 | 状态 | 最佳 epoch | BoxAP | PoseAP | ΔPoseAP（相对完整模型） | Val loss |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in records:
        if r['status'] == 'ok':
            lines.append(
                f"| {r['label']} | 完成 | {r['epoch']} | {r['BoxAP']:.4f} | "
                f"{r['PoseAP']:.4f} | {r['delta_pose_ap']:+.4f} | {r['val_loss']:.4f} |"
            )
        else:
            lines.append(f"| {r['label']} | {r['status']} | — | — | — | — | — |")
    lines += [
        '', '## 解释框架', '',
        '- `去除 FFT - 完整模型`：FFT 全局粗对齐的边际贡献。',
        '- `去除 DCN - 完整模型`：DCN 局部精对齐的边际贡献。',
        '- `去除 Spatial - 完整模型`：学习式空间选择相对无参数平均的边际贡献。',
        '- `RGB FFT - 完整模型`：RGB 层 FFT 相对 Feature FFT 的差异。',
        '- `Concat - 完整模型`：Concat 投影相对 Spatial gate 的差异。',
        '- `单阶段 DAV2 - 完整模型`：去除第二阶段精修后的差异。',
        '- 负的 ΔPoseAP 表示该替代或移除方案低于完整模型。',
        '', '## 备注', '',
        f'- 结果根目录：`{RESULTS_ROOT}`',
        '- 本页由 `summarize_results.py` 自动更新。',
    ]
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {csv_path}')
    print(f'Wrote {report_path}')


if __name__ == '__main__':
    main()
