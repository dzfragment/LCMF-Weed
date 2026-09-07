# P2L-P4 核心架构消融

## 实验目的

测量多帧模型三个核心组件的边际贡献：

1. **FFT**：估计帧间全局平移，并在 P2/P3/P4 特征层执行分数位移。
2. **DCN / DAV2**：在 P4 上进行可变形局部精对齐。
3. **Spatial gate**：根据空间位置选择参考帧信息。

采用单组件移除与替代机制消融设计，完整模型作为共同对照。

## 实验矩阵

| 编号 | FFT | DCN | Spatial | Spatial 的替代方式 | 研究问题 |
|---|---:|---:|---:|---|---|
| 00 完整模型 | ✓ | ✓ | ✓ | — | 完整架构性能基准 |
| 01 去 FFT | ✗ | ✓ | ✓ | — | 全局粗对齐的贡献 |
| 02 去 DCN | ✓ | ✗ | ✓ | — | 局部精对齐的贡献 |
| 03 去 Spatial | ✓ | ✓ | ✗ | 无参数平均 | 学习式空间选择的贡献 |
| 05 Concat 融合 | ✓ | ✓ | Concat | 学习式拼接投影 | Spatial gate 是否优于自由拼接学习？ |
| 06 单阶段 DAV2 | ✓ | 单阶段 | ✓ | — | 第二阶段 DCN 精修是否必要？ |

### 操作定义

> 2026-08-11 更新: 对齐模块已切换到一致性约束管线 (ConsistentAligner)。
> 原 `--fft-feature-align` 替换为 `--consistent-align`: 语义保持 feature 模式
> (RGB 只估计位移, 实际平移在 P2/P3/P4 特征层), 位移估计改为三帧 top-K +
> 矢量自洽约束 + 期望注入 (weed_sliding_merged 三集 40px 口径 100%)。
> 推理/评估脚本在加载 ckpt 后按 consistent_align 标志自动启用一致管线。

- **去 FFT**：不传对齐参数，原始 Backbone 特征直接进入 P4 DAV2。
- **去 DCN**：使用 `--no-align`，对齐后的特征跳过 DAV2，直接进入 Spatial gate。
- **去 Spatial**：使用 `--fusion-gate-type avg --fusion-mode avg`，以无参数平均替代学习式门控。
- **一致性特征层对齐 (原 Feature FFT)**：使用 `--consistent-align`，RGB 只估计位移（三帧 top-K + 矢量自洽），实际平移在 P2/P3/P4 特征层，再进入 P4 DAV2。
- **Concat 融合**：使用 `--fusion-gate-type concat`，以拼接和学习投影替代 Spatial gate。
- **单阶段 DAV2**：使用 `--dav2-single`，保留第一阶段偏移估计，去除第二阶段精修。

01～03 属于单组件移除，04～05 属于融合机制和容量替代实验。

## 固定条件

```text
基座：SF-P2L-P4 seed42
融合层：P4 only
检测层：P4 only，stride 16
数据集：weed_sliding_merged（位于仓库外，由环境变量 `DATASET_ROOT` 指定）
输入：1024×1024
Seed：42
Stage 2：100 epochs，lr=1e-3
Stage 3：500 epochs，lr=1e-4
Stage 3：frame dropout=0.1，Head 解冻
```

所有组共享相同训练设置，只有 FFT、DCN、Spatial 的存在性发生变化。

## 运行

先做 3+3 epoch 七组试跑：

```bash
cd p2l_p4_core_arch_ablation
export DATASET_ROOT=/path/to/weed_sliding_merged   # 含 train/val/test 序列目录
bash run_pilot.sh
```

正式串行运行全部四组：

```bash
bash run_all.sh
```

单独运行：

```bash
bash 00_full_fft_dcn_spatial.sh
bash 01_remove_fft.sh
bash 02_remove_dcn.sh
bash 03_remove_spatial.sh
bash 05_ablate_concat_fusion.sh
bash 06_ablate_dav2_single.sh
```

## 参数覆盖

```bash
BATCH_SIZE=2 S2_EPOCHS=20 S3_EPOCHS=100 bash run_all.sh
```

支持：

```text
DATASET_ROOT
PRETRAINED
RESULTS_ROOT
DEVICE
BATCH_SIZE
INPUT_SIZE
NUM_WORKERS
SEED
S2_EPOCHS
S3_EPOCHS
S2_LR
S3_LR
FRAME_DROPOUT
```

## 输出

```text
p2l_p4_core_arch_ablation/
├── results/
│   ├── 00_full_fft_dcn_spatial/
│   ├── 01_remove_fft/
│   ├── 02_remove_dcn/
│   ├── 03_remove_spatial/
│   ├── 05_ablate_concat_fusion/
│   └── 06_ablate_dav2_single/
├── EXPERIMENT_REPORT.md
└── results_tables/RESULTS_SUMMARY.csv
```

每个实验目录包含 checkpoint、训练日志、验证指标和 `run_config.txt`。

## 自动报告

训练完成后 `run_all.sh` 会执行：

```bash
python summarize_results.py
```

它读取七组 `stage3_log.csv` 中验证集 PoseAP 最佳的 epoch，更新：

```text
EXPERIMENT_REPORT.md
results_tables/RESULTS_SUMMARY.csv
```

也可以随时手动更新报告：

```bash
python summarize_results.py
```

## 补充实验：只运行新增缺项

原有 `00`～`06` 实验已经完成，补充队列不会重新运行它们。新增实验为：

```text
07 三层 DAV2
08 Channel gate
09 ConvGRU
10 TSA
11 TSA-add
12 跳过 Stage 2，直接 Stage 3
```

数据集固定为：

```text
$DATASET_ROOT/
```

先运行结构、梯度和 checkpoint 测试：

```bash
python test_supplement_configs.py
```

执行新增实验的短程 Pilot：

```bash
bash run_supplement_pilot.sh
```

执行新增实验正式训练：

```bash
bash run_supplement_all.sh
```

`run_supplement_all.sh` 只调用 `07`～`12`，不会调用旧的 `00`～`06`。新增结果默认写入 `results/`，Pilot 默认写入 `supplement_pilot_results/`。

补充结果汇总为：

```text
SUPPLEMENT_EXPERIMENT_REPORT.md
results_tables/SUPPLEMENT_RESULTS_SUMMARY.csv
```

07 使用真正独立的第三个 offset/mask/DCN 精修模块，包含 `s3_conv`、`s3_offset`、`s3_mask` 和 `s3_dcn`。08～11 保持 Feature FFT 和双阶段 DAV2 不变，仅替换融合模块。12 从同一 SF-P2L-P4 seed42 单帧权重直接进入 Stage 3，跳过 Stage 2；主模型和 Stage 3-only 都使用 500 个 Stage 3 epoch，主模型额外包含 100 个 Stage 2 epoch。

## 结果解释

对每个移除实验计算：

```text
ΔPoseAP = 移除组件后的 PoseAP - 完整模型 PoseAP
```

例如：

```text
去 FFT ΔPoseAP = -0.012
```

表示移除 FFT 后 PoseAP 下降 0.012，FFT 在该训练配置下具有约 0.012 的边际贡献。

严格来说，01～03 测得的是组件在当前系统中的条件贡献，三个下降量不能简单相加，因为 FFT、DCN 和 Spatial 之间可能存在交互效应。若后续要分离全部主效应与交互效应，需要补充两两移除和三者全移除的完整 2³ 因子实验。
