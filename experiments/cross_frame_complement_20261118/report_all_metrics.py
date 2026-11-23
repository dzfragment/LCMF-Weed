"""Publish a single metrics appendix from all complementarity summary artifacts."""
import argparse
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'robustness/results/exp4_01main_refs_20261111'
METRIC_NAMES = ('ap10', 'recall10', 'mean_error_px', 'pose_ap50', 'pose_ap50_95')


def read(path):
    return json.loads(path.read_text())


def flat(record):
    return dict(ap10=record['ap10'], recall10=record['at25']['recall'],
                mean_error_px=record['at25']['mean_error_px'], pose_ap50=record['pose_ap50'],
                pose_ap50_95=record['pose_ap50_95'])


def average(records):
    records = [flat(row) for row in records]
    return {key: mean(valid) if (valid := [row[key] for row in records if row[key] is not None]) else None
            for key in METRIC_NAMES}


def mode(table, models, modes):
    return average([table[f'{model}_{arm}'] for model in models for arm in modes])


def sequence_arms(table, models):
    output = {arm: mode(table, models, [arm]) for arm in ('R1R2', 'R1R1', 'R2R2', 'KK')}
    output.update({key: flat(record) for key, record in table.items() if key.startswith('sf_')})
    return output


def balanced_arms(table, models, complementary, redundant, singles=()):
    output = {}
    for label, prefix in [('MF', models), ('SF', ['sf_fuse'] if 'sf_K' in table else ['sf'])]:
        output[f'{label} 互补：' + '/'.join(complementary)] = mode(table, prefix, complementary)
        output[f'{label} 重叠：' + '/'.join(redundant)] = mode(table, prefix, redundant)
        if singles:
            output[f'{label} 单参考（四种平均）'] = mode(table, prefix, singles)
        output[f'{label} 仅 K'] = flat(table['sf_K'] if label == 'SF' and 'sf_K' in table else
                                     table['sf_KK']) if label == 'SF' else mode(table, prefix, ['KK'])
    return output


def archive():
    combined = {}
    for name in ('normal_clean_offset_withap', 'norefs_clean_offset_withap', 'normal_standard_withap',
                 'norefs_standard_withap', 'exp2_normal_withap', 'dose_dense_withap'):
        for key, record in read(LEGACY / f'{name}.json').items():
            aggregate = record['agg']
            fixed, pose = aggregate['at_conf_0.25'], record['ap']
            values = dict(ap10=aggregate['point_ap_10px'], recall10=fixed['recall'],
                          mean_error_px=fixed['nPE_mean_px'], pose_ap50=pose['pose_ap50'],
                          pose_ap50_95=pose['pose_ap'])
            if key in combined:
                assert all(abs(values[field] - combined[key][field]) < 1e-5 for field in METRIC_NAMES), key
            combined[key] = values
    return combined


def append_table(lines, title, arms):
    lines.extend([f'### {title}', '',
                  '| 条件 | AP10 (%) | R@10 (%) | 平均点距 (px) | Pose AP50 (%) | Pose AP50:95 (%) |',
                  '|---|---:|---:|---:|---:|---:|'])
    for label, values in arms.items():
        columns = []
        for field in METRIC_NAMES:
            value = values[field]
            columns.append('—' if value is None else (f'{value:.3f}' if field == 'mean_error_px' else f'{100 * value:.2f}'))
        lines.append(f'| {label.replace(chr(124), " / ")} | {" | ".join(columns)} |')
    lines.append('')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=HERE)
    args = parser.parse_args()
    standard = read(args.run_dir / 'standard_test_summary.json')
    first = read(args.run_dir / 'summary.json')
    followup = read(args.run_dir / 'followup_summary.json')
    models = ['mf42', 'mf43', 'mf44']
    lines = ['# exp4｜跨帧互补实验：统一指标', '',
             'R@10 与平均点距使用 conf=.25、1024 输入坐标、10px 半径的一对一匹配；点距只平均匹配成功的 TP，'
             '不能用来代表被漏检目标。AP10 和 Pose AP 不使用 .25 截断。',
             '新实验的 Pose AP50:95 使用 exp4 核心表原口径（Ultralytics 8.4.80、单关键点 OKS σ=1、GT 框面积 ×0.53、'
             '0.50:0.05:0.95，最多 300 预测/图，最低分 .001）。Pose AP 使用全图标注；区域及同株诊断仍看筛选子集上的 R@10，'
             '不把非子集目标误计为背景。MF 按种子取均值，错配再平均三套供体，SF 为既有一个权重。', '',
             '## 正式测试集（主结果）', '']
    for key, title in [('all', '全部 575 组'), ('key_blur_with_clear_ref', '关键帧被模糊且有未额外加模糊的参考：288 组'),
                       ('public_only', '公开数据子集：492 组'), ('verified_blur_flags_only', '重放可精确验证的条件：570 组')]:
        arms = {name: flat(row) for name, row in standard[key]['means'].items()}
        append_table(lines, title, arms)
    mechanisms_path = args.run_dir / 'official_mechanisms_summary.json'
    if mechanisms_path.exists():
        mechanisms = read(mechanisms_path)
        lines.extend(['## 正式测试集的受控机制实验（575 组）', '',
                      '重放正式 test 的原始裁剪窗口，去除生成器退化后只施加指定模糊。'
                      '保留 492 组公开来源和 83 组私有来源，标签和样本与正式 test 一一对应。'
                      '全图四项指标如下；配对子集召回见 `official_mechanisms_summary.json`。', ''])
        for key, title in [('regional', '不同区域模糊：concat'), ('regional_avg', '不同区域模糊：等权平均'),
                           ('interleaved', '同株交错条带'), ('directional', '正交方向模糊')]:
            append_table(lines, title, balanced_arms(mechanisms[key]['metrics'], models, ('XY', 'YX'),
                         ('XX', 'YY'), () if key.startswith('regional') else ('XK', 'YK', 'KX', 'KY')))
    lines.extend(['## 历史机制实验：仅公开来源的替代测试资产（不作为主结果）', ''])
    for key, title in [('controlled_public_holdout', '不同区域模糊：concat'),
                       ('avg', '不同区域模糊：等权平均')]:
        block = first[key]
        if key == 'avg':
            block = block['controlled_public_holdout']['analysis']
        append_table(lines, title, balanced_arms(block['metrics'], models, ('AB', 'BA'), ('AA', 'BB')))
    for key, title in [('interleaved', '同株交错条带'), ('directional', '正交方向模糊')]:
        append_table(lines, title, balanced_arms(followup[key]['metrics'], models, ('XY', 'YX'), ('XX', 'YY'),
                                                  ('XK', 'YK', 'KX', 'KY')))
    lines.extend(['各输入槽顺序、单参考、随机种子及区域/同株子集细项见 `summary.json` 和 `followup_summary.json`。', '',
                  '## 49 组历史诊断（不作为论文主证据）', ''])
    for key, title in [('real_testset', '原图'), ('real_testset_blurred', '人工模糊：concat'),
                       ('avg', '人工模糊：等权平均')]:
        block = first[key] if key != 'avg' else first['avg']['real_testset_blurred']['analysis']
        append_table(lines, title, sequence_arms(block['metrics'], models))
    table = followup['placebo']['metrics']
    arms = {arm: mode(table, models, [arm]) for arm in ('MATCH', 'R1R1', 'R2R2', 'KK')}
    for variant in ('WRONG1', 'WRONG2', 'WRONGBOTH'):
        arms[variant] = mode(table, models, [f'{variant}_{idx}' for idx in (1, 2, 3)])
    arms.update({key: flat(table[key]) for key in ('sf_K', 'sf_fuse_R1R2')})
    append_table(lines, '配准筛查后的错配', arms)
    lines.extend(['## 早期人工遮挡 pilot（历史归档）', '',
                  '参考开关、offset 遮挡、key/refs/all 和密集 dose-response 的四项指标此前已计算，此处合并完整记录。'
                  '这里的 Pose AP 保留原 PAD/YOLO 工具计算口径，不与上面的 Ultralytics Pose AP 数值混比；'
                  '旧 clean/人工遮挡资产未经统一 holdout 重算，不作独立主证据。', ''])
    append_table(lines, '原始历史指标', archive())
    output = args.run_dir / 'ALL_METRICS.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
