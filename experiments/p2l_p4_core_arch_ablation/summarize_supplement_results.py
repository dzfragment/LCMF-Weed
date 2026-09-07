#!/usr/bin/env python3
"""Summarize only new module/training-stage experiments while reading old controls as references."""
from __future__ import annotations
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get('RESULTS_ROOT', ROOT / 'results'))
NEW = [
    ('07_ablate_dav2_triple', 'DAV2 3-stage', 'dcn_depth'),
    ('08_fusion_channel', 'Channel gate', 'fusion'),
    ('09_fusion_convgru', 'ConvGRU', 'fusion'),
    ('10_fusion_tsa', 'TSA', 'fusion'),
    ('11_fusion_tsa_add', 'TSA-add', 'fusion'),
    ('12_train_stage3_only', 'Stage 3 only', 'training_stage'),
]
REFERENCES = [
    ('00_full_fft_dcn_spatial', 'DAV2 2-stage + Spatial', 'reference'),
    ('03_remove_spatial', 'Average', 'reference'),
    ('05_ablate_concat_fusion', 'Concat', 'reference'),
    ('06_ablate_dav2_single', 'DAV2 1-stage + Spatial', 'reference'),
]


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return math.nan


def best(exp_id):
    path = RESULTS / exp_id / 'stage3_log.csv'
    if not path.exists():
        return {'status': 'missing', 'log': str(path)}
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if not math.isnan(num(r, 'PoseAP'))]
    if not valid:
        return {'status': 'empty', 'log': str(path)}
    row = max(valid, key=lambda r: num(r, 'PoseAP'))
    return {
        'status': 'ok', 'epoch': int(float(row['epoch'])),
        'BoxAP': num(row, 'BoxAP'), 'PoseAP': num(row, 'PoseAP'),
        'val_loss': num(row, 'val_loss'), 'log': str(path),
    }


def main():
    records = [dict(experiment=e, label=l, group=g, is_new=True, **best(e)) for e, l, g in NEW]
    refs = [dict(experiment=e, label=l, group=g, is_new=False, **best(e)) for e, l, g in REFERENCES]
    all_rows = refs + records
    full = next((r for r in refs if r['experiment'] == '00_full_fft_dcn_spatial' and r['status'] == 'ok'), None)
    for r in all_rows:
        r['delta_vs_main'] = r['PoseAP'] - full['PoseAP'] if full and r['status'] == 'ok' else math.nan

    fields = ['experiment', 'label', 'group', 'is_new', 'status', 'epoch',
              'BoxAP', 'PoseAP', 'delta_vs_main', 'val_loss', 'log']
    with (ROOT / 'results_tables' / 'SUPPLEMENT_RESULTS_SUMMARY.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(all_rows)

    def table(rows):
        out = ['| 实验 | 状态 | 最佳 epoch | BoxAP | PoseAP | ΔPoseAP vs 主模型 |',
               '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['status'] == 'ok':
                out.append(f"| {r['label']} | 完成 | {r['epoch']} | {r['BoxAP']:.4f} | {r['PoseAP']:.4f} | {r['delta_vs_main']:+.4f} |")
            else:
                out.append(f"| {r['label']} | {r['status']} | — | — | — | — |")
        return out

    dcn = [r for r in all_rows if r['experiment'] in ('06_ablate_dav2_single', '00_full_fft_dcn_spatial', '07_ablate_dav2_triple')]
    fusion = [r for r in all_rows if r['experiment'] in ('03_remove_spatial', '05_ablate_concat_fusion', '00_full_fft_dcn_spatial', '08_fusion_channel', '09_fusion_convgru', '10_fusion_tsa', '11_fusion_tsa_add')]
    stage = [r for r in all_rows if r['experiment'] in ('00_full_fft_dcn_spatial', '12_train_stage3_only')]

    lines = ['# 补充实验报告', '',
             '本报告只汇总新增实验，并读取已完成的旧实验作为参照，不会重新运行旧实验。', '',
             '## DAV2 层数', ''] + table(dcn) + [
             '', '比较 1、2、3 个独立 offset/mask/DCN 精修阶段。', '',
             '## 融合模块', ''] + table(fusion) + [
             '', '- Average：无门控基线。',
             '- Concat：拼接后自由投影。',
             '- Channel：逐通道可信度门控。',
             '- Spatial：逐像素空间门控。',
             '- ConvGRU：递归时序状态融合。',
             '- TSA：关键帧与参考帧相似度驱动的时序注意力。',
             '- TSA-add：限制残差为非负的 TSA。', '',
             '## 训练阶段', ''] + table(stage) + [
             '', '主模型采用 Stage 2 预训练融合模块后进入 Stage 3；Stage 3 only 从同一单帧基座直接端到端训练。',
             '该比较测量 Stage 2 预热流程的贡献。默认二者 Stage 3 都训练 500 epoch，主流程额外包含 100 epoch Stage 2。', '',
             '## 固定设置', '',
             '- 数据集：`$DATASET_ROOT`（weed_sliding_merged，需自行准备）',
             '- 基座：SF-P2L-P4 seed42',
             '- 融合层：P4',
             '- 检测层：P4，stride 16',
             '- Feature FFT downsample 4',
             '- seed 42，输入 1024，Stage 3 Head 解冻，frame dropout 0.1', '',
             f'- 读取结果目录：`{RESULTS}`']
    (ROOT / 'SUPPLEMENT_EXPERIMENT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {ROOT / "results_tables" / "SUPPLEMENT_RESULTS_SUMMARY.csv"}')
    print(f'Wrote {ROOT / "SUPPLEMENT_EXPERIMENT_REPORT.md"}')


if __name__ == '__main__':
    main()
