# fft_align — 独立 FFT 快速帧间对齐模块

从 `models/fft_align.py` 拆出的独立包，纯算法实现，不依赖任何模型/训练代码。

## 结构

```
fft_align/
├── __init__.py   # 公共接口: FFTAlign / align_frame / chain_align_frames /
│                 #          apply_feature_shift / ConsistentAligner
├── core.py       # FFTAlign 类 (核心算法)
├── utils.py      # numpy 接口 / 三帧链式对齐 / 特征空间平移
└── consistent.py # ConsistentAligner (三帧一致性约束增强管线, 推荐)
```

## 算法

1. 提取 G 通道，降采样 (4×/8×)
2. Sobel 梯度幅值 + 去均值（抑制均匀土壤噪声）
3. FFT 去均值互相关 → 峰值位移 + SNR（可选相位归一化 normalize=True）
4. MSE 自校验消歧方向
5. 整数切片平移（无插值）

速度 ~3ms/帧对 (1024², 8×, GPU)；精度 ±4px (8×) / ±2px (4×)。

## 用法

```python
from fft_align import FFTAlign, align_frame, chain_align_frames

aligner = FFTAlign(downsample=4, do_mse_check=True, normalize=True).to(device).eval()
dx, dy, snr = aligner.find_displacement(mov, ref)   # (B,3,H,W) tensor

aligned, dx, dy, snr = align_frame(ref_np, ref_np)  # numpy HWC

a1, a2, (dx1, dy1), (dx2, dy2), (snr1, snr2) = \
    chain_align_frames(key, ref1, ref2)             # 三帧链式对齐
```

位移约定：`find_displacement(mov, ref)` 返回 mov 内容相对 ref 的位移。

## 推荐管线: ConsistentAligner (三帧一致性约束)

利用滑动序列物理约束 d(f1→f0) + d(f2→f1) ≈ d(f2→f0)：
top1 自洽即采用 (99.87% 序列零额外开销)；不自洽时期望位移注入 +
枚举组合 (残差 + 质量评分)；高置信 f1f0 被改坏时回退。

```python
from fft_align import ConsistentAligner
al = ConsistentAligner(downsample=4, k=5)
dx1, dy1, dx2c, dy2c, info = al.align_chain(f0, f1, f2)  # uint8 BGR numpy

# 模型集成 (批量 tensor 接口):
a1, a2, dx1, dy1, dx2, dy2 = al.align_batch(key, ref1, ref2,
                                            return_displacement=True)
```

weed_sliding_merged 全三集 (test/val/train, 5240 序列) 40px 口径 100%
(标签真值), 修复旧管线全部 13 个失败序列, 零回归。
8× 档同样 100% (27ms/序列); 2× 档适合 ±2px 严格口径 (82.7%)。
批量 tensor 接口 ~2.6ms/序列 (batch=4), 比旧管线 (含 MSE 校验) 快。
详见 research/RESEARCH_REPORT.md。

## 模型集成 (MultiFrameYOLO)

v4 架构: RGB 只用于估计位移, 实际平移发生在 P2/P3/P4 特征层
(apply_feature_shift 分数位移)。一致性管线取代的是"位移估计"环节:

```python
# 推荐: feature 模式内置一致位移估计 (训练入口)
train.py --consistent-align --fft-downsample 4   # 取代 --fft-feature-align

# 或代码构建
MultiFrameYOLO(global_align_stage='feature', consistent_align=True, ...)

# eval/推理: 从 ckpt 读取标志自动启用
if ckpt.get('consistent_align', False) and model.global_align_stage == 'feature':
    from models.mf_yolo import enable_consistent_align
    enable_consistent_align(model)
```

`apply_consistent_image_align` (RGB 层包装) 与 `apply_fft_image_align` 均为
备选 image 模式入口, 保留兼容。

## 兼容性

`models/fft_align.py` 现为兼容 re-export 层（保留 `apply_fft_image_align`
模型集成函数），历史 `from models.fft_align import FFTAlign` 不受影响。

## 评估

`../eval_fft_align_success.py` 在 weed_sliding_merged/test 上评测对齐成功率
（真值来自标签匹配投票），输出见 `../fft_align_success_output/`。
`../research/` 含交叉验证、失败分析、一致性管线研究数据。
