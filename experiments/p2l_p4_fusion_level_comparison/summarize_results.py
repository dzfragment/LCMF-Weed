#!/usr/bin/env python3
"""Summarize P3/P4 fusion-level comparison from Stage-3 validation logs."""
from __future__ import annotations
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = Path(os.environ.get('RESULTS_ROOT', ROOT / 'results'))
EXPERIMENTS = [
    ('01_fuse_p4', 'P4 only'),
    ('02_fuse_p3', 'P3 only'),
    ('03_fuse_p3_p4', 'P3 + P4'),
]


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def collect(exp_id):
    path = RESULTS_ROOT / exp_id / 'stage3_log.csv'
    if not path.exists():
        return {'status': 'missing', 'log': str(path)}
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if not math.isnan(number(r, 'PoseAP'))]
    if not valid:
        return {'status': 'empty', 'log': str(path)}
    best = max(valid, key=lambda r: number(r, 'PoseAP'))
    return {
        'status': 'ok',
        'epoch': int(float(best['epoch'])),
        'BoxAP': number(best, 'BoxAP'),
        'PoseAP': number(best, 'PoseAP'),
        'val_loss': number(best, 'val_loss'),
        'log': str(path),
    }


def main():
    records = [dict(experiment=e, label=l, **collect(e)) for e, l in EXPERIMENTS]
    p4 = next((r for r in records if r['experiment'] == '01_fuse_p4' and r['status'] == 'ok'), None)
    for r in records:
        r['delta_vs_p4'] = r['PoseAP'] - p4['PoseAP'] if p4 and r['status'] == 'ok' else math.nan

    with (ROOT / 'RESULTS_SUMMARY.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['experiment', 'label', 'status', 'epoch', 'BoxAP', 'PoseAP', 'delta_vs_p4', 'val_loss', 'log']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)

    lines = [
        '# P2L-P4 不同特征层融合对比实验报告', '',
        '## 验证集结果', '',
        '| 融合层 | 状态 | 最佳 epoch | BoxAP | PoseAP | ΔPoseAP vs P4 | Val loss |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in records:
        if r['status'] == 'ok':
            lines.append(
                f"| {r['label']} | 完成 | {r['epoch']} | {r['BoxAP']:.4f} | "
                f"{r['PoseAP']:.4f} | {r['delta_vs_p4']:+.4f} | {r['val_loss']:.4f} |")
        else:
            lines.append(f"| {r['label']} | {r['status']} | — | — | — | — | — |")
    lines += [
        '', '## 解释框架', '',
        '- P3 only 优于 P4 only：高分辨率时序细节通过 FPN 向 P4 传播，并带来净收益。',
        '- P4 only 优于 P3 only：更深语义特征更稳定，且 P3 对齐噪声或成本较高。',
        '- P3+P4 优于两种单层：多尺度时序信息具有互补性。',
        '- P3+P4 未超过最佳单层：双层可能引入冗余、优化冲突或额外对齐误差。',
        '', '## 固定边界', '',
        '- P2 始终不融合。',
        '- FPN 始终为 DarkFPN_P2_lite。',
        '- 检测头始终只有 P4，stride 16。',
        '- 本实验改变的是时序融合注入层，不改变检测层。',
        '', f'- 结果目录：`{RESULTS_ROOT}`',
    ]
    (ROOT / 'EXPERIMENT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {ROOT / "RESULTS_SUMMARY.csv"}')
    print(f'Wrote {ROOT / "EXPERIMENT_REPORT.md"}')


if __name__ == '__main__':
    main()
