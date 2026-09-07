#!/usr/bin/env python3
"""
合并滑动窗口多帧数据集:
  老: $WEED_EXP_ROOT/weedataset_sliding_v6    (train 1027 / val 68 / test 83)
  新: $WEED_EXP_ROOT/weed_public_sliding      (train 3053 / val 328 / test 681, 编号已错开)
  → $WEED_EXP_ROOT/weed_sliding_merged

每个 split 内 seq 编号唯一(新数据 train 从 01028 起 / val 从 00069 起 /
test 从 00084 起),合并零冲突。
"""

import os
import shutil

_DATA_ROOT = os.environ.get('WEED_EXP_ROOT')
if not _DATA_ROOT:
    raise SystemExit('[dataset_tools] 请设置环境变量 WEED_EXP_ROOT：实验数据集根目录（旧 exp2：'
                     'weed1024 / public_weed_le256_* / weed_public_* / weed_sliding_merged 等所在）')

OLD = os.path.join(_DATA_ROOT, 'weedataset_sliding_v6')
NEW = os.path.join(_DATA_ROOT, 'weed_public_sliding')
OUT = os.path.join(_DATA_ROOT, 'weed_sliding_merged')

SPLITS = ['train', 'val', 'test']


def collect_seq_dirs(root):
    """返回 {split: {seq_name: seq_path}}"""
    result = {}
    for split in SPLITS:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            result[split] = {}
            continue
        seqs = {}
        for name in sorted(os.listdir(d)):
            if name.startswith('seq_') and os.path.isdir(os.path.join(d, name)):
                seqs[name] = os.path.join(d, name)
        result[split] = seqs
    return result


def main():
    print("=" * 60)
    print("合并滑动窗口多帧数据集")
    print(f"  老: {OLD}")
    print(f"  新: {NEW}")
    print(f"  输出: {OUT}")
    print("=" * 60)

    old_seqs = collect_seq_dirs(OLD)
    new_seqs = collect_seq_dirs(NEW)

    if os.path.exists(OUT):
        shutil.rmtree(OUT)

    stats = {}
    for split in SPLITS:
        dst = os.path.join(OUT, split)
        os.makedirs(dst, exist_ok=True)

        n_old = len(old_seqs[split])
        n_new = len(new_seqs[split])

        # 冲突检查
        dup = set(old_seqs[split]) & set(new_seqs[split])
        if dup:
            print(f"  [FATAL] {split} 序列编号冲突: {sorted(dup)[:5]}")
            return

        # 复制
        for name, src in sorted(old_seqs[split].items()):
            shutil.copytree(src, os.path.join(dst, name))
        for name, src in sorted(new_seqs[split].items()):
            shutil.copytree(src, os.path.join(dst, name))

        stats[split] = (n_old, n_new, n_old + n_new)
        print(f"  [{split}] 老 {n_old} + 新 {n_new} = {n_old + n_new} 序列")

    # 校验: 每 split 内编号唯一 & 帧文件完整
    print("\n[校验]")
    ok = True
    for split in SPLITS:
        d = os.path.join(OUT, split)
        seqs = sorted(n for n in os.listdir(d)
                      if n.startswith('seq_') and os.path.isdir(os.path.join(d, n)))
        n_frames = 0
        for s in seqs:
            for f in ['frame_0.jpg', 'frame_1.jpg', 'frame_2.jpg',
                      'labels_frame0.txt', 'labels_frame1.txt', 'labels_frame2.txt']:
                if not os.path.exists(os.path.join(d, s, f)):
                    print(f"  [WARN] {split}/{s} 缺 {f}")
                    ok = False
            n_frames += 3
        print(f"  [{split}] {len(seqs)} 序列 / {n_frames} 帧")
    print("  ✓ 校验通过" if ok else "  ✗ 存在问题")

    # YAML
    total_seq = sum(v[2] for v in stats.values())
    yaml_path = os.path.join(OUT, "weed_sliding_merged.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"""# 合并滑动窗口多帧数据集 (weedataset_sliding_v6 + weed_public_sliding)
# 老数据集: 国内田间拍摄 (video/session/single)
# 新数据集: CropAndWeed 子集 (旋转90°, 单独滑动规则, 垂直双向同向)
#            Frame0 随机遮挡概率: train/val 45%, test 30% (老数据集为 30%)
#            Frame0 标注区域遮挡覆盖率: test 60% (train/val 及老数据集为 36%)
train: ./train
val: ./val
test: ./test

nc: 1
kpt_shape: [1, 2]
flip_idx: [0]
names: ['weed']

num_frames: 3
frame_size: 1024
min_labels_keyframe: 4

# 序列构成: 老 {stats['train'][0]}+{stats['val'][0]}+{stats['test'][0]} /
#           新 {stats['train'][1]}+{stats['val'][1]}+{stats['test'][1]}
""")
    print(f"\n  YAML: {yaml_path}")
    print(f"  总序列: {total_seq} ({total_seq * 3} 帧)")
    print("完成!")


if __name__ == "__main__":
    main()
