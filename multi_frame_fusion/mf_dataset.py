"""
多帧数据集加载器

加载 3 帧滑动序列数据（每序列 frame_0/1/2.jpg + 关键帧标注）。
关键帧 (frame_0) 的标注作为训练目标。

数据增强规则：
- 同步几何变换：resize、random_perspective、flip 三帧使用相同随机种子
- 颜色抖动：HSV 增强可以不同（每帧独立）
"""
import os
import random
import math

import cv2
import numpy as np
import torch
from torch.utils import data


def _load_sequence_images(seq_dir):
    """解码一个序列的 3 帧图片（供 preload_cache 并行调用，线程安全）。"""
    return [cv2.imread(os.path.join(seq_dir, f'frame_{i}.jpg')) for i in range(3)]


class MultiFrameDataset(data.Dataset):
    """
    多帧数据集。

    每个序列目录包含:
        frame_0.jpg, frame_1.jpg, frame_2.jpg, labels.txt

    labels.txt 格式（关键帧标注，归一化坐标）:
        cls x_center y_center width height kpt_x kpt_y [kpt_vis]

    返回:
        frames: (3, 3, H, W) tensor, 3帧 × 3通道
        targets: dict 含 cls, box, kpt, idx（仅关键帧标签）
    """
    def __init__(self, dataset_root, split='train', input_size=1024, params=None, augment=True,
                 skip_incomplete=True):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.augment = augment
        self.params = params or {}
        self.skip_incomplete = skip_incomplete
        # 解码图片缓存（preload_cache() 填充；主进程预加载后 dataloader worker
        # 经 fork COW 共享，避免每 epoch 重复 imread+JPEG 解码。仅跳过解码，
        # 增强逻辑与输出张量逐位不变）
        self._image_cache = {}

        # 收集所有序列目录
        split_dir = os.path.join(dataset_root, split)
        raw_dirs = sorted([
            os.path.join(split_dir, d) for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

        # 过滤不完整的序列（缺少任意帧或标签文件无法读取）
        skipped = 0
        self.seq_dirs = []
        for seq_dir in raw_dirs:
            if self.skip_incomplete and not self._is_complete_sequence(seq_dir):
                skipped += 1
                continue
            self.seq_dirs.append(seq_dir)
        self.n = len(self.seq_dirs)

        # 预加载所有标签
        self._load_all_labels()

        print(f'[MultiFrameDataset] {split}: {self.n} sequences' +
              (f' (skipped {skipped} incomplete)' if skipped else ''))

    def preload_cache(self, num_workers=None):
        """预解码所有序列图片到内存，供 dataloader worker 共享。

        用 ThreadPoolExecutor 并行（cv2.imread 解码释放 GIL），直接写入主进程
        堆内存，不经 pickle 传输；随后 DataLoader fork 出的 worker 经 COW 共享
        这些页，物理内存只算一份。仅跳过 imread+解码，增强输出逐位不变。
        """
        if self._image_cache:
            return
        if num_workers is None:
            num_workers = min(24, os.cpu_count() or 1)
        from concurrent.futures import ThreadPoolExecutor
        cache = {}
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for seq_dir, imgs in zip(self.seq_dirs,
                                     ex.map(_load_sequence_images, self.seq_dirs)):
                cache[seq_dir] = imgs
        self._image_cache = cache
        if cache:
            gb = sum(imgs[0].nbytes for imgs in cache.values()) * 3 / 1e9
            print(f'[MultiFrameDataset] {self.split}: preloaded {len(cache)} '
                  f'sequences -> ~{gb:.1f} GB in RAM')

    def _is_complete_sequence(self, seq_dir):
        """检查序列是否完整：3帧图片都存在且可读"""
        for i in range(3):
            img_path = os.path.join(seq_dir, f'frame_{i}.jpg')
            if not os.path.exists(img_path):
                return False
            # 快速检查文件是否可读（不加载全图，只检查文件头）
            try:
                with open(img_path, 'rb') as f:
                    header = f.read(4)
                    if len(header) < 4:
                        return False
            except (IOError, OSError):
                return False
        return True

    def _load_all_labels(self):
        """预加载所有序列的标签"""
        self.boxes = []
        self.kpts = []

        for seq_dir in self.seq_dirs:
            label_path = os.path.join(seq_dir, 'labels_frame0.txt')
            if os.path.exists(label_path):
                with open(label_path) as f:
                    lines = [x.strip() for x in f.readlines() if x.strip()]
                if lines:
                    label = np.array([list(map(float, l.split())) for l in lines], dtype=np.float32)
                    box = label[:, :5]
                    kpt = label[:, 5:]
                else:
                    box = np.zeros((0, 5), dtype=np.float32)
                    kpt = np.zeros((0, 0), dtype=np.float32)
            else:
                box = np.zeros((0, 5), dtype=np.float32)
                kpt = np.zeros((0, 0), dtype=np.float32)

            # 确保 kpt 有正确的形状 (nk, 3) — 含可见性维度
            nk = 1  # 1个关键点
            if kpt.size > 0:
                # labels.txt 只含 (x, y)，补可见性维度 vis=2 (labeled and visible)
                vis = np.full((kpt.shape[0], 1), 2.0, dtype=np.float32)
                kpt = np.concatenate([kpt, vis], axis=1)
                kpt = kpt.reshape(-1, nk, 3)
            else:
                kpt = np.zeros((0, nk, 3), dtype=np.float32)

            self.boxes.append(box)
            self.kpts.append(kpt)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        seq_dir = self.seq_dirs[index]

        # 加载3帧图像（优先内存缓存；浅拷贝防任何潜在原地修改污染缓存）
        cached = self._image_cache.get(seq_dir)
        if cached is not None:
            images = [img.copy() for img in cached]
        else:
            images = []
            for i in range(3):
                img_path = os.path.join(seq_dir, f'frame_{i}.jpg')
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f'Cannot read {img_path}')
                images.append(img)

        # 同步几何增强
        if self.augment:
            images, box, kpt = self._sync_augment(images, self.boxes[index].copy(), self.kpts[index].copy())
        else:
            # 验证模式：简单 resize（记录原始尺寸，与 _resize_img 一致）
            orig_h, orig_w = images[0].shape[:2]
            images = [self._resize_img(img) for img in images]
            box = self.boxes[index].copy()
            kpt = self.kpts[index].copy()
            if box.size:
                r = min(self.input_size / orig_h, self.input_size / orig_w)
                r = min(r, 1.0)
                pad_w = int(round(orig_w * r))
                pad_h = int(round(orig_h * r))
                dw = (self.input_size - pad_w) / 2
                dh = (self.input_size - pad_h) / 2
                box[:, 1:] = self._wh2xy(box[:, 1:], pad_w, pad_h, dw, dh)
                kpt[..., 0] = kpt[..., 0] * pad_w + dw
                kpt[..., 1] = kpt[..., 1] * pad_h + dh

        nl = len(box)
        h, w = images[0].shape[:2]

        # 归一化
        if nl:
            box[:, 1:5] = self._xy2wh(box[:, 1:5], w, h)
            kpt[..., 0] /= w
            kpt[..., 1] /= h

        # 每帧独立颜色增强
        if self.augment:
            for i in range(3):
                images[i] = self._color_augment(images[i])

        # 转换为 tensor
        frames = []
        for img in images:
            # HWC → CHW, BGR → RGB
            sample = img.transpose((2, 0, 1))[::-1]
            sample = np.ascontiguousarray(sample)
            frames.append(torch.from_numpy(sample))

        frames = torch.stack(frames, dim=0)  # (3, 3, H, W)

        # 构建 targets
        cls = torch.from_numpy(box[:, 0:1]).float() if nl else torch.zeros((0, 1))
        target_box = torch.from_numpy(box[:, 1:5]).float() if nl else torch.zeros((0, 4))
        target_kpt = torch.from_numpy(kpt).float() if nl else torch.zeros((0, 1, 3))

        return frames, cls, target_box, target_kpt, torch.zeros(nl, dtype=torch.long)

    def _sync_augment(self, images, box, kpt):
        """
        同步几何增强：3帧使用相同随机参数。

        返回: augmented_images (list), box, kpt
        """
        h, w = images[0].shape[:2]
        aug_params = {}

        # Step 1: Resize（同步）
        r = min(self.input_size / h, self.input_size / w)
        r = min(r, 1.0)
        pad_w = int(round(w * r))
        pad_h = int(round(h * r))
        dw = (self.input_size - pad_w) / 2
        dh = (self.input_size - pad_h) / 2

        aug_params['ratio'] = (r, r)
        aug_params['pad'] = (dw, dh)

        for i in range(3):
            if (images[i].shape[1], images[i].shape[0]) != (pad_w, pad_h):
                images[i] = cv2.resize(images[i], (pad_w, pad_h),
                                       interpolation=self._resample())
            images[i] = cv2.copyMakeBorder(images[i],
                                           int(round(dh - 0.1)), int(round(dh + 0.1)),
                                           int(round(dw - 0.1)), int(round(dw + 0.1)),
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 更新标注坐标到 resize+pad 后的空间
        if box.size:
            box[:, 1:] = self._wh2xy(box[:, 1:], pad_w, pad_h, dw, dh)
            kpt[..., 0] = kpt[..., 0] * pad_w + dw
            kpt[..., 1] = kpt[..., 1] * pad_h + dh

        # Step 2: RandomPerspective（同步——同一种子）
        if self.params.get('use_perspective', True):
            images, box, kpt = self._sync_perspective(images, box, kpt)

        # Step 3: Flip（同步）
        images, box, kpt = self._sync_flip(images, box, kpt)

        return images, box, kpt

    def _sync_perspective(self, images, box, kpt):
        """同步透视变换"""
        h, w = images[0].shape[:2]
        params = self.params

        # 生成一次随机参数
        center = np.eye(3)
        center[0, 2] = -w / 2
        center[1, 2] = -h / 2

        rotate = np.eye(3)
        a = random.uniform(-params.get('degrees', 0), params.get('degrees', 0))
        s = random.uniform(1 - params.get('scale', 0.5), 1 + params.get('scale', 0.5))
        rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        shear = np.eye(3)
        shear[0, 1] = math.tan(random.uniform(-params.get('shear', 0), params.get('shear', 0)) * math.pi / 180)
        shear[1, 0] = math.tan(random.uniform(-params.get('shear', 0), params.get('shear', 0)) * math.pi / 180)

        translate = np.eye(3)
        translate[0, 2] = random.uniform(0.5 - params.get('translate', 0.1),
                                         0.5 + params.get('translate', 0.1)) * w
        translate[1, 2] = random.uniform(0.5 - params.get('translate', 0.1),
                                         0.5 + params.get('translate', 0.1)) * h

        matrix = translate @ shear @ rotate @ center

        # 对3帧应用相同矩阵
        for i in range(3):
            images[i] = cv2.warpAffine(images[i], matrix[:2], dsize=(w, h),
                                       borderValue=(0, 0, 0))

        # 更新标注
        if box.size:
            n = len(box)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = box[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ matrix.T
            xy = xy[:, :2].reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new_box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            nk = kpt.shape[1]
            xy_kpt = np.ones((n * nk, 3))
            xy_kpt[:, :2] = kpt[..., :2].reshape(n * nk, 2)
            xy_kpt = xy_kpt @ matrix.T
            xy_kpt = xy_kpt[:, :2]
            kpt[..., :2] = xy_kpt.reshape(n, nk, 2)

            # Clip
            new_box[:, [0, 2]] = new_box[:, [0, 2]].clip(0, w)
            new_box[:, [1, 3]] = new_box[:, [1, 3]].clip(0, h)
            # Keep transformed coordinates: clamping a point outside the crop
            # to x=0/y=0 invents a valid target at the image boundary. The PAD
            # target builder rejects out-of-image points, including after flips.
            inside = ((kpt[..., 0] >= 0) & (kpt[..., 0] < w)
                      & (kpt[..., 1] >= 0) & (kpt[..., 1] < h))
            kpt[..., 2] = np.where(inside, kpt[..., 2], 0)

            # 过滤退化框：透视变换可能把目标移出画面，clip 后宽或高 ≤ 1px
            valid_w = (new_box[:, 2] - new_box[:, 0]) > 1
            valid_h = (new_box[:, 3] - new_box[:, 1]) > 1
            valid = valid_w & valid_h
            if not valid.all():
                new_box = new_box[valid]
                box = box[valid]
                kpt = kpt[valid]

            box[:, 1:5] = new_box

        return images, box, kpt

    def _sync_flip(self, images, box, kpt):
        """同步翻转（box 为像素坐标 [cls, x1, y1, x2, y2]）"""
        params = self.params
        w_img = images[0].shape[1]
        h_img = images[0].shape[0]

        # Flip LR
        if random.random() < params.get('flip_lr', 0.5):
            for i in range(3):
                images[i] = np.fliplr(images[i])
            if box.size:
                old_x1 = box[:, 1].copy()
                old_x2 = box[:, 3].copy()
                box[:, 1] = w_img - old_x2
                box[:, 3] = w_img - old_x1
                kpt[..., 0] = w_img - kpt[..., 0]
                # 翻转关键点索引（如果定义了）
                if 'flip_index' in params:
                    kpt = np.ascontiguousarray(kpt[:, params['flip_index'], :])

        # Flip UD
        if random.random() < params.get('flip_ud', 0.0):
            for i in range(3):
                images[i] = np.flipud(images[i])
            if box.size:
                old_y1 = box[:, 2].copy()
                old_y2 = box[:, 4].copy()
                box[:, 2] = h_img - old_y2
                box[:, 4] = h_img - old_y1
                kpt[..., 1] = h_img - kpt[..., 1]

        return images, box, kpt

    def _color_augment(self, image):
        """单帧独立颜色增强（HSV）"""
        params = self.params
        h_gain = params.get('hsv_h', 0.015)
        s_gain = params.get('hsv_s', 0.7)
        v_gain = params.get('hsv_v', 0.4)

        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        x = np.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype('uint8')
        lut_s = np.clip(x * r[1], 0, 255).astype('uint8')
        lut_v = np.clip(x * r[2], 0, 255).astype('uint8')

        hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _resize_img(self, image):
        """验证模式 resize（保持比例，不放大）"""
        h, w = image.shape[:2]
        r = min(self.input_size / h, self.input_size / w)
        r = min(r, 1.0)

        pad_w = int(round(w * r))
        pad_h = int(round(h * r))
        dw = (self.input_size - pad_w) / 2
        dh = (self.input_size - pad_h) / 2

        if (w, h) != (pad_w, pad_h):
            image = cv2.resize(image, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image

    @staticmethod
    def _wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
        """[xc, yc, w, h] normalized → [x1, y1, x2, y2] pixel"""
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
        return y

    @staticmethod
    def _xy2wh(x, w, h):
        """[x1, y1, x2, y2] pixel → [xc, yc, w, h] normalized"""
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1e-3)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1e-3)
        y = np.copy(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
        y[:, 2] = (x[:, 2] - x[:, 0]) / w
        y[:, 3] = (x[:, 3] - x[:, 1]) / h
        return y

    @staticmethod
    def _resample():
        """选择 resize 插值方法。训练时固定 INTER_LINEAR 以保证可复现性。"""
        return cv2.INTER_LINEAR

    @staticmethod
    def collate_fn(batch):
        """自定义 collate：堆叠 frames 和 targets"""
        frames, cls, box, kpt, indices = zip(*batch)

        frames = torch.stack(frames, 0)  # (B, 3, 3, H, W)
        cls = torch.cat(cls, 0)
        box = torch.cat(box, 0)
        kpt = torch.cat(kpt, 0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, 0)

        targets = {
            'cls': cls,
            'box': box,
            'kpt': kpt,
            'idx': indices,
        }
        return frames, targets


def create_dataloaders(dataset_root, input_size=1024, batch_size=4,
                       num_workers=4, params=None, no_preload=False):
    """创建训练和验证 DataLoader

    no_preload=True: 不预载解码缓存，改为惰性从磁盘读取（大幅降低常驻内存，
    适用于共享机器上频繁被 OOM killer 选中大内存进程的场景）。"""
    if params is None:
        params = {
            'kpt_shape': [1, 3],
            'degrees': 0.0,
            'scale': 0.5,
            'shear': 0.0,
            'translate': 0.1,
            'flip_lr': 0.5,
            'flip_ud': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'use_perspective': True,
        }

    # 训练增强开关：MF_TRAIN_AUGMENT=0 时训练集也走 resize-only（同 val 路径），
    # 大幅降低 dataloader CPU 开销。注意会改变训练数据分布（仅用于提速/特殊对照）。
    train_augment = os.environ.get('MF_TRAIN_AUGMENT', '1') != '0'
    train_dataset = MultiFrameDataset(
        dataset_root, split='train', input_size=input_size,
        params=params, augment=train_augment,
    )
    val_dataset = MultiFrameDataset(
        dataset_root, split='val', input_size=input_size,
        params=params, augment=False,
    )
    # 主进程预加载解码缓存（fork 前），dataloader worker 经 COW 共享，不重复占用。
    # no_preload=True 时跳过预载，__getitem__ 回退到惰性磁盘读取，常驻内存从 ~42GB 降到几 GB。
    if not no_preload:
        train_dataset.preload_cache()
        val_dataset.preload_cache()

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn,
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn,
    )

    return train_loader, val_loader
