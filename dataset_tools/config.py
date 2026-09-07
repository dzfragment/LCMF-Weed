"""
数据集处理配置文件
多帧融合数据集：大图拆分 + 多帧合成
"""

import os

# ============ 路径配置 ============
# 数据目录一律通过环境变量提供（仓库不携带任何数据）：
#   WEED_SRC_ROOT — 素材根目录，须含 weedataset/（私有源图 train/val/test 三集合），
#                    stage1/stage2 的输出（weedataset_split、weedataset_sliding_v6 等）也写在这里
def _env_root(name, hint):
    root = os.environ.get(name)
    if not root:
        raise SystemExit(f'[dataset_tools] 请设置环境变量 {name}：{hint}')
    return root

BASE_DIR = _env_root('WEED_SRC_ROOT', '含 weedataset 源图素材的根目录（stage1/stage2 的输入输出根）')
WEED_DIR = os.path.join(BASE_DIR, "weedataset")

# 输入
TRAIN_IMAGES = os.path.join(WEED_DIR, "train", "images")
TRAIN_LABELS = os.path.join(WEED_DIR, "train", "labels")
VAL_IMAGES = os.path.join(WEED_DIR, "val", "images")
VAL_LABELS = os.path.join(WEED_DIR, "val", "labels")

# === Stage 1 输出 ===
STAGE1_OUT = os.path.join(BASE_DIR, "weedataset_split")
STAGE1_YAML = "weed_split.yaml"

# === Stage 2 输出 ===
STAGE2_OUT = os.path.join(BASE_DIR, "weedataset_synthetic_v2")
STAGE2_YAML = "weed_syn.yaml"

# ============ 拆分参数 ============
RESIZE_TARGET = 2048   # 统一 resize 目标
TILE_SIZE = 1024       # 子图尺寸
GRID_COLS = 2
GRID_ROWS = 2

# 跨边界标注：面积占比阈值
MIN_AREA_RATIO = 0.3

# 可视化
VIZ_SAMPLE_COUNT = 10  # split_viz 随机抽样数

# ============ 多帧合成参数 ============
# Frame 1 偏移（相对于 Frame 0）—— 双向水平，单向垂直（模拟前进运动）
FRAME1_DX_MIN, FRAME1_DX_MAX = -40, 40         # 水平双向
FRAME1_DY_MIN, FRAME1_DY_MAX = 110, 150       # 垂直单向正位移 110-150

# 位移最小幅度（避免采样到接近 0 的值导致帧间几乎无变化）
DX_MIN_MAGNITUDE = 15                          # dx1 绝对值至少 15px
DY_MIN_MAGNITUDE = 110                         # dy1 至少 110px（与 FRAME1_DY_MIN 一致）

# Frame 2 偏移（相对于 Frame 1 的附加位移，与 Frame 1 同量级）
FRAME2_DX_MIN, FRAME2_DX_MAX = 8, 20           # 水平附加
FRAME2_DY_MIN, FRAME2_DY_MAX = 110, 150        # 垂直附加 110-150 → dy2 ∈ [220, 300]

# 镜像 padding 尺寸（需 ≥ 最大累积偏移: max(150+150=300, 40+20=60)）
MIRROR_PADDING = 512                            # 覆盖 300

# ═══ 噪声参数（各类型独立随机，不再从池子抽选）═══

# 1) 振动失焦 — 独立概率
BLUR_PROB_KEYFRAME = 0.55    # Frame 0 模糊概率 (v6: +5%)
BLUR_PROB_REFERENCE = 0.35   # Frame 1/2 模糊概率 (v6: +5%)
GAUSS_KERNEL_MIN, GAUSS_KERNEL_MAX = 7, 17     # (v6: 上调, 原 3-11)
GAUSS_SIGMA_MIN, GAUSS_SIGMA_MAX = 2.5, 5.0    # (v6: 上调, 原 1.0-3.0)
MOTION_LEN_MIN, MOTION_LEN_MAX = 12, 25         # (v6: 上调, 原 5-15)

# 2) 过曝 — 独立概率
OVEREXPOSE_PROB = 0.50
OVEREXPOSE_COUNT_MIN, OVEREXPOSE_COUNT_MAX = 1, 1
OVEREXPOSE_AREA_MIN, OVEREXPOSE_AREA_MAX = 0.05, 0.15
OVEREXPOSE_BRIGHT_MIN, OVEREXPOSE_BRIGHT_MAX = 1.25, 1.45
OVEREXPOSE_SAT_MIN, OVEREXPOSE_SAT_MAX = 0.45, 0.70

# 3) 画面随机位置杂草遮挡 — 独立概率
RANDOM_OCC_PROB_KEYFRAME = 0.30    # Frame 0
RANDOM_OCC_PROB_REFERENCE = 0.15   # Frame 1/2
OCCLUSION_COUNT_MIN, OCCLUSION_COUNT_MAX = 1, 3
OCCLUSION_SIZE_MIN, OCCLUSION_SIZE_MAX = 20, 40   # px
OCCLUSION_ALPHA_MIN, OCCLUSION_ALPHA_MAX = 1.0, 1.0

# 4) 标注区域杂草遮挡 — 始终执行，按比例选中标注框
# Frame 0 至少需要的标注数，不足则跳过该序列
MIN_LABELS_FOR_KEYFRAME = 4
# Frame 0 遮挡概率：选中多少比例的标注框（0~1）
OCCLUDE_LABEL_RATIO_KEYFRAME = 0.50
# Frame 1/2 遮挡概率
OCCLUDE_LABEL_RATIO_REFERENCE = 0.15
# 每个被选中的标注框内，遮挡块覆盖面积比例（0.36 = 边长约为 bbox 的 60%）
OCCLUDE_BBOX_AREA_RATIO = 0.36
# 遮挡块最大边长 (px)，防止超大标注框产生巨型遮挡块
OCCLUDE_SIZE_MAX = 40

# 5) 背景区域不透明遮挡（模拟标注遗漏的杂草 / false negative）
BG_OCCLUDE_COUNT_MIN, BG_OCCLUDE_COUNT_MAX = 2, 5   # 每帧背景遮挡块数量
BG_OCCLUDE_SIZE_MIN, BG_OCCLUDE_SIZE_MAX = 20, 40   # px

# 每张源图生成的序列组数（不同随机偏移&噪声）
SAMPLES_PER_IMAGE_TRAIN = 3
SAMPLES_PER_IMAGE_VAL = 1

# 可视化
SYN_VIZ_SAMPLE_COUNT = 12

# ============ 滑动窗口多帧合成 (stage2_sliding) ============
# 输入：weedataset 原始三集合（train/val/test）
SRC_TRAIN_IMAGES = TRAIN_IMAGES
SRC_TRAIN_LABELS = TRAIN_LABELS
SRC_VAL_IMAGES   = VAL_IMAGES
SRC_VAL_LABELS   = VAL_LABELS
SRC_TEST_IMAGES  = os.path.join(WEED_DIR, "test", "images")
SRC_TEST_LABELS  = os.path.join(WEED_DIR, "test", "labels")

# 输出
STAGE2_SLIDING_OUT = os.path.join(BASE_DIR, "weedataset_sliding_v6")
STAGE2_SLIDING_YAML = "weed_sliding.yaml"

# 随机种子基数（用于复现）
BASE_SEED = 18345

# 每张源图尝试多少次随机窗口位置
MAX_ATTEMPTS_PER_IMAGE = 50

# 每张源图生成多少组合格序列
SLIDING_SAMPLES_TRAIN = 8
SLIDING_SAMPLES_VAL   = 2
SLIDING_SAMPLES_TEST  = 4

# 滑动窗口的 stride（用于系统性覆盖，与随机采样同时使用）
SLIDING_STRIDE = 192  # 每192px采样一个候选位置（v6: 更密集利用扩大的有效区域）
