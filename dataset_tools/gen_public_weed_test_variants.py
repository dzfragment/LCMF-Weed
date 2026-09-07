#!/usr/bin/env python3
"""生成 10 个不同种子的 test 集 (v2 规则) 用于观察种子敏感性。

规则与 gen_public_weed_sliding_v2.py 完全一致:
  Frame 0 随机遮挡 45% (train/val/test 统一)
  Frame 0 标注区域遮挡选中比例 30%
  仅 SEED_BASE 不同 → 窗口采样不同。

种子: SEED_BASE = 20261001 + i*1000, test 实际种子 = SEED_BASE + 448
输出: $WEED_EXP_ROOT/weed_public_sliding_v2_test_variants/seed_<test_seed>/test/seq_XXXXX/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_public_weed_sliding_v2 as g

_DATA_ROOT = os.environ.get('WEED_EXP_ROOT')
if not _DATA_ROOT:
    raise SystemExit('[dataset_tools] 请设置环境变量 WEED_EXP_ROOT：实验数据集根目录（旧 exp2：'
                     'weed1024 / public_weed_le256_* / weed_public_* / weed_sliding_merged 等所在）')

OUT_BASE = os.path.join(_DATA_ROOT, 'weed_public_sliding_v2_test_variants')
SEED_BASE0 = 20261001   # 起始 SEED_BASE
STEP = 1000             # 种子间隔
N_VARIANTS = 10


def _variant_done(out_dir):
    """变体是否已完整生成 (test 目录下存在序列且无缺文件)"""
    tdir = os.path.join(out_dir, 'test')
    if not os.path.isdir(tdir):
        return False
    seqs = [x for x in os.listdir(tdir)
            if x.startswith('seq_') and os.path.isdir(os.path.join(tdir, x))]
    if not seqs:
        return False
    for s in seqs:
        for f in ['frame_0.jpg', 'frame_1.jpg', 'frame_2.jpg',
                  'labels_frame0.txt', 'labels_frame1.txt',
                  'labels_frame2.txt']:
            if not os.path.exists(os.path.join(tdir, s, f)):
                return False
    return True


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    for i in range(N_VARIANTS):
        sb = SEED_BASE0 + i * STEP
        g.SEED_BASE = sb  # 覆盖模块级常量, main 内读取时生效
        test_seed = sb + sum(map(ord, 'test'))
        out_dir = os.path.join(OUT_BASE, f"seed_{test_seed}")
        if _variant_done(out_dir):
            print(f"[{i:02d}/{N_VARIANTS}] seed_{test_seed} 已存在且完整, 跳过")
            continue
        print(f"\n{'='*60}\n[{i:02d}/{N_VARIANTS}] SEED_BASE={sb} "
              f"→ test 实际种子 {test_seed}\n{'='*60}")
        g.main(split_only='test', out_root=out_dir)

    print(f"\n{'='*60}")
    print(f"全部完成: {OUT_BASE}")
    for d in sorted(os.listdir(OUT_BASE)):
        p = os.path.join(OUT_BASE, d, 'test')
        n = len([x for x in os.listdir(p)
                 if x.startswith('seq_') and os.path.isdir(os.path.join(p, x))]) \
            if os.path.exists(p) else 0
        print(f"  {d}: {n} 序列")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
