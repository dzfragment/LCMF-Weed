"""
consistent.py — 三帧一致性约束对齐管线 (FFTAlign 的鲁棒增强, 项目默认推荐)。

背景:
  FFTAlign(normalize=True) 单帧对对齐在 weed_sliding_merged 上已达 99%+,
  但仍有少量失败, 机理包括:
    - 原点伪峰压制 (真实峰第 2 名, 如 seq_00425)
    - MSE 方向校验翻反 (如 seq_00499)
    - 噪声峰占据 top-K, 正确峰缺失 (如 val seq_00030/00049, train 9 例)

方法:
  滑动序列的三帧位移必须满足矢量自洽: d(f1->f0) + d(f2->f1) ≈ d(f2->f0)。
  管线:
    1. 每帧对取相关面 top-K 候选 (梯度幅值 + 相位归一化互相关)
    2. 若三帧 top1 已自洽 (残差 ≤ 40px) → 直接采用 (99.87% 序列)
    3. 不自洽时: 期望位移注入 (f2f0_top1 - f1f0_top1 等合成方向) +
       枚举组合, 评分 = 残差 + 50×(质量损失)
    4. 兜底: 枚举把高置信 f1f0 改坏时回退 top1

效果 (weed_sliding_merged, 40px 口径, GT=标签投票):
    test : f1f0 329/329, f2f0 213/213   (旧管线 2 例失败)
    val  : f1f0 177/177, f2f0 140/140   (旧管线 2 例失败)
    train: f1f0 1779/1779, f2f0 1381/1381 (旧管线 9 例失败)
  全部零失败, 三帧残差最大 17px。8× 档同样 100% (27ms/序列)。
  纯计算 ~15ms/序列 (4×), 比旧管线 (含 MSE 校验) 快 ~14%。
"""

import itertools
import math

import torch

from .core import FFTAlign


class ConsistentAligner:
    """top-K 候选 + 三帧矢量自洽约束 + 质量评分 + 回退。

    两种接口:
      - align_chain(f0, f1, f2): numpy uint8 BGR, 单序列 (研究/评估)
      - align_batch(key, ref1, ref2): (B,3,H,W) float [0,1] tensor, 批量 (模型集成)
    """

    CONSISTENT_EPS = 40.0   # top1 残差低于此值视为自洽
    HARD_RES = 16.0         # 枚举硬约束: 残差 ≤ 此值优先
    QUALITY_WEIGHT = 50.0   # 质量惩罚权重: 残差 1px ≈ 质量 0.02
    FALLBACK_MOVE = 40.0    # f1f0 被改动的距离阈值 (px)
    FALLBACK_QUAL = 0.7     # f1f0 候选质量低于 top1 的此比例视为"被改坏"

    def __init__(self, downsample=4, k=5, do_mse_check=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.aligner = FFTAlign(downsample=downsample, do_mse_check=do_mse_check,
                                normalize=True).to(self.device).eval()
        self.k = k
        self.ds = downsample

    # ═══════════════ 候选提取 (GPU) ═══════════════

    def _grad_corr_candidates(self, mov, ref, k):
        """单序列 top-K 候选。mov/ref: (1,3,H,W) [0,1] → [(dx, dy, corr_val)]"""
        return self._grad_corr_candidates_batch(mov, ref, k)[0]

    @torch.no_grad()
    def _grad_corr_candidates_batch(self, mov, ref, k):
        """批量 top-K 候选。mov/ref: (B,3,H,W) [0,1] → [B 个 [(dx, dy, val)]]"""
        B, _, H, W = mov.shape
        ds = self.ds
        Hs, Ws = H // ds, W // ds
        g_ref = torch.nn.functional.interpolate(
            self.aligner._extract_g(ref), size=(Hs, Ws), mode='bilinear', align_corners=False)
        g_mov = torch.nn.functional.interpolate(
            self.aligner._extract_g(mov), size=(Hs, Ws), mode='bilinear', align_corners=False)
        rg = self.aligner._gradient_magnitude(g_ref)
        mg = self.aligner._gradient_magnitude(g_mov)
        corr = self.aligner._compute_correlation(rg, mg)          # (B, Hs, Ws)
        flat = corr.reshape(B, -1)
        vals, idxs = torch.topk(flat, k, dim=1)
        out = []
        for b in range(B):
            cands = []
            for v, i in zip(vals[b].tolist(), idxs[b].tolist()):
                py, px = divmod(i, Ws)
                dy = -(py if py <= Hs // 2 else py - Hs)
                dx = -(px if px <= Ws // 2 else px - Ws)
                cands.append((round(dx * ds), round(dy * ds), v))
            out.append(cands)
        return out

    # ═══════════════ 候选求解 (纯逻辑, 无图像依赖) ═══════════════

    @staticmethod
    def _unwrap_near(wrapped, anchor, period):
        """把圆周位移 ``wrapped`` 展开到最靠近 ``anchor`` 的周期。

        FFT 相关面只能表示一个周期内的主值。相邻帧链式和 ``anchor``
        提供周期编号，使直接 f2->f0 峰在超过半幅时仍可保留原相关质量。
        """
        if period is None:
            return wrapped
        period = float(period)
        if period <= 0:
            raise ValueError(f'period must be positive, got {period}')
        # 主值区间内直接相关没有发生必然混叠，保持历史决策，避免旋转或
        # 错误链式候选仅因靠近半幅边界就把 c20 带到相邻周期。只有链式
        # 锚点明确越过半幅时才需要决定周期编号。
        if abs(anchor) <= period / 2:
            return wrapped
        return wrapped + math.floor((anchor - wrapped) / period + 0.5) * period

    @staticmethod
    def _axis_needs_unwrap(step1, step2, period):
        """判断相邻两段是否提供了可信的跨周期锚点。

        除总和越过半幅外，还要求两段同向、均有足够幅值且量级接近。
        这对应连续滑动序列，避免旋转/伪峰用一大一小的随机位移触发展开。
        """
        period = float(period)
        a, b = float(step1), float(step2)
        small, large = sorted((abs(a), abs(b)))
        return (abs(a + b) > period / 2 and a * b > 0 and
                small >= period / 8 and large <= 2.0 * small)

    @staticmethod
    def _steps_are_coherent(step1, step2):
        """两段二维运动是否足以作为周期展开锚点。"""
        x1, y1 = float(step1[0]), float(step1[1])
        x2, y2 = float(step2[0]), float(step2[1])
        n1, n2 = math.hypot(x1, y1), math.hypot(x2, y2)
        if min(n1, n2) < 1e-6 or max(n1, n2) > 2.0 * min(n1, n2):
            return False
        cosine = (x1 * x2 + y1 * y2) / (n1 * n2)
        return cosine >= 0.9

    @classmethod
    def _unwrap_c20(cls, cand20, chain_dx, chain_dy, image_hw):
        """按链式锚点分别展开 c20 的 x/y 圆周位移。"""
        dx3, dy3, q3 = cand20
        if image_hw is None:
            return dx3, dy3, q3
        image_h, image_w = image_hw
        return (cls._unwrap_near(dx3, chain_dx, image_w),
                cls._unwrap_near(dy3, chain_dy, image_h), q3)

    @classmethod
    def _solve_from_candidates(cls, c10, c21, c20, image_hw=None):
        """从三组候选解出最终位移。

        c10/c21/c20: [(dx, dy, corr_val)] 按相关值降序。
        image_hw: 原图 (H, W)。提供后，c20 圆周候选会按 c10+c21
            展开到最近周期；省略时保持历史行为，兼容研究脚本。
        返回 (dx1, dy1, dx2c, dy2c, res, mode):
          dx1/dy1: f1->f0; dx2c/dy2c: 展开后的 f2->f0;
          res: 三帧残差; mode: top1/enumerate，发生展开时追加 _unwrapped。
        """
        t1, t2, t3_wrapped = c10[0], c21[0], c20[0]
        chain_top1 = (t1[0] + t2[0], t1[1] + t2[1])
        # 序列级门控：只有 top1 链式锚点明确越过某轴半幅，才允许整次
        # 求解使用周期展开。否则所有次级候选沿用历史普通欧氏逻辑，避免
        # 旋转/伪峰候选偶然跨界后改变半幅内样本的排序。
        unwrap_hw = None
        if image_hw is not None:
            image_h, image_w = image_hw
            coherent_steps = cls._steps_are_coherent(t1[:2], t2[:2])
            unwrap_x = coherent_steps and cls._axis_needs_unwrap(
                t1[0], t2[0], image_w)
            unwrap_y = coherent_steps and cls._axis_needs_unwrap(
                t1[1], t2[1], image_h)
            if unwrap_x or unwrap_y:
                # None 表示该轴维持普通主值，不参与周期展开。
                unwrap_hw = (image_h if unwrap_y else None,
                             image_w if unwrap_x else None)
        if unwrap_hw is None:
            return cls._solve_without_unwrap(c10, c21, c20)
        t3 = cls._unwrap_c20(t3_wrapped, *chain_top1, unwrap_hw)
        res_top1 = math.hypot(chain_top1[0] - t3[0], chain_top1[1] - t3[1])
        top1_unwrapped = t3[:2] != t3_wrapped[:2]

        if res_top1 <= cls.CONSISTENT_EPS:
            mode = 'top1_unwrapped' if top1_unwrapped else 'top1'
            return t1[0], t1[1], t3[0], t3[1], res_top1, mode

        # ── 不自洽: 期望位移注入 (矢量合成推导) ──
        c10 = list(c10)
        c21 = list(c21)
        c20 = list(c20)
        # c20 是圆周候选，必须先相对当前 c10 的合理邻域展开，才能反推 c21。
        # 这里用 c21 top1 给出临时链式锚点；完整组合仍由下方枚举决定。
        for dx1_, dy1_, _ in c10[:4]:
            anchor_dx, anchor_dy = dx1_ + t2[0], dy1_ + t2[1]
            for cand3 in c20[:4]:
                dx3_, dy3_, _ = cls._unwrap_c20(
                    cand3, anchor_dx, anchor_dy, unwrap_hw)
                exp_dx, exp_dy = dx3_ - dx1_, dy3_ - dy1_
                c21.append((exp_dx, exp_dy, 0.0))
                for nx, ny in ((4, 0), (-4, 0), (0, 4), (0, -4)):
                    c21.append((exp_dx + nx, exp_dy + ny, 0.0))
        for (dx1_, dy1_, _), (dx2_, dy2_, _) in itertools.product(c10[:4], c21[:8]):
            exp_dx, exp_dy = dx1_ + dx2_, dy1_ + dy2_
            c20.append((exp_dx, exp_dy, 0.0))
        c10 = cls._dedup(c10)
        c21 = cls._dedup(c21)
        c20 = cls._dedup(c20)

        q_max = t1[2] + t2[2] + t3_wrapped[2]

        def resolve_combo(comb):
            cand1, cand2, cand3_wrapped = comb
            chain_dx = cand1[0] + cand2[0]
            chain_dy = cand1[1] + cand2[1]
            cand3 = cls._unwrap_c20(
                cand3_wrapped, chain_dx, chain_dy, unwrap_hw)
            res = math.hypot(chain_dx - cand3[0], chain_dy - cand3[1])
            quality_loss = q_max - (cand1[2] + cand2[2] + cand3[2])
            score = res + cls.QUALITY_WEIGHT * quality_loss
            unwrapped = cand3[:2] != cand3_wrapped[:2]
            return score, res, cand1, cand2, cand3, unwrapped

        resolved = [resolve_combo(c) for c in itertools.product(c10, c21, c20)]
        hard = [r for r in resolved if r[1] <= cls.HARD_RES]
        pool = hard if hard else sorted(resolved, key=lambda r: r[0])[:50]
        pick = min(pool, key=lambda r: r[0])
        _, res_pick, cand1, cand2, cand3, was_unwrapped = pick
        dx1_, dy1_, q1 = cand1
        dx2_, dy2_, _ = cand2
        dx3_, dy3_, _ = cand3

        # ── 兜底: 高置信 f1f0 被改坏时回退 ──
        f1_moved = math.hypot(dx1_ - t1[0], dy1_ - t1[1]) > cls.FALLBACK_MOVE
        if f1_moved and q1 < t1[2] * cls.FALLBACK_QUAL:
            dx1_, dy1_ = t1[0], t1[1]
            target = (dx1_ + dx2_, dy1_ + dy2_)
            resolved20 = []
            for cand_wrapped in c20:
                cand = cls._unwrap_c20(cand_wrapped, *target, unwrap_hw)
                distance = math.hypot(cand[0] - target[0], cand[1] - target[1])
                resolved20.append((distance, cand, cand[:2] != cand_wrapped[:2]))
            _, best20, fallback_unwrapped = min(resolved20, key=lambda item: item[0])
            dx3_, dy3_ = best20[0], best20[1]
            was_unwrapped = was_unwrapped or fallback_unwrapped
            res_pick = math.hypot(target[0] - dx3_, target[1] - dy3_)

        mode = 'enumerate_unwrapped' if was_unwrapped else 'enumerate'
        return dx1_, dy1_, dx3_, dy3_, res_pick, mode

    @classmethod
    def _solve_without_unwrap(cls, c10, c21, c20):
        """历史普通欧氏求解路径；半幅内逐项保持原实现。"""
        t1, t2, t3 = c10[0], c21[0], c20[0]
        res_top1 = math.hypot(
            t1[0] + t2[0] - t3[0], t1[1] + t2[1] - t3[1])
        if res_top1 <= cls.CONSISTENT_EPS:
            return t1[0], t1[1], t3[0], t3[1], res_top1, 'top1'

        c10 = list(c10)
        c21 = list(c21)
        c20 = list(c20)
        for (dx1_, dy1_, _), (dx3_, dy3_, _) in itertools.product(
                c10[:4], c20[:4]):
            exp_dx, exp_dy = dx3_ - dx1_, dy3_ - dy1_
            c21.append((exp_dx, exp_dy, 0.0))
            for nx, ny in ((4, 0), (-4, 0), (0, 4), (0, -4)):
                c21.append((exp_dx + nx, exp_dy + ny, 0.0))
        for (dx1_, dy1_, _), (dx2_, dy2_, _) in itertools.product(
                c10[:4], c21[:8]):
            c20.append((dx1_ + dx2_, dy1_ + dy2_, 0.0))
        c10 = cls._dedup(c10)
        c21 = cls._dedup(c21)
        c20 = cls._dedup(c20)
        q_max = t1[2] + t2[2] + t3[2]

        def combo_score(comb):
            (dx1_, dy1_, q1), (dx2_, dy2_, q2), (dx3_, dy3_, q3) = comb
            res = math.hypot(dx1_ + dx2_ - dx3_, dy1_ + dy2_ - dy3_)
            return res + cls.QUALITY_WEIGHT * (q_max - (q1 + q2 + q3))

        combos = list(itertools.product(c10, c21, c20))
        hard = [c for c in combos
                if math.hypot(c[0][0] + c[1][0] - c[2][0],
                              c[0][1] + c[1][1] - c[2][1]) <= cls.HARD_RES]
        pool = hard if hard else sorted(combos, key=combo_score)[:50]
        pool.sort(key=combo_score)
        pick = pool[0]
        (dx1_, dy1_, q1), (dx2_, dy2_, _), (dx3_, dy3_, _) = pick
        res_pick = math.hypot(dx1_ + dx2_ - dx3_, dy1_ + dy2_ - dy3_)

        f1_moved = math.hypot(dx1_ - t1[0], dy1_ - t1[1]) > cls.FALLBACK_MOVE
        if f1_moved and q1 < t1[2] * cls.FALLBACK_QUAL:
            dx1_, dy1_ = t1[0], t1[1]
            target = (dx1_ + dx2_, dy1_ + dy2_)
            best20 = min(c20, key=lambda c: math.hypot(
                c[0] - target[0], c[1] - target[1]))
            dx3_, dy3_ = best20[0], best20[1]
            res_pick = math.hypot(dx1_ + dx2_ - dx3_, dy1_ + dy2_ - dy3_)
        return dx1_, dy1_, dx3_, dy3_, res_pick, 'enumerate'

    @staticmethod
    def _dedup(cands):
        seen, out = set(), []
        for d in cands:
            key = (d[0], d[1])
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    # ═══════════════ 公开接口 ═══════════════

    @torch.no_grad()
    def align_chain(self, key_np, ref1_np, ref2_np):
        """三帧一致对齐 (numpy 单序列)。

        Args:
            key_np, ref1_np, ref2_np: (H, W, 3) uint8 BGR numpy
        Returns:
            (dx1, dy1, dx2c, dy2c, info)
        """
        import numpy as np

        def to_t(img):
            return torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).to(self.device)
        k_t, r1_t, r2_t = to_t(key_np), to_t(ref1_np), to_t(ref2_np)
        c10 = self._grad_corr_candidates(r1_t, k_t, self.k)
        c21 = self._grad_corr_candidates(r2_t, r1_t, self.k)
        c20 = self._grad_corr_candidates(r2_t, k_t, self.k)
        dx1, dy1, dx2, dy2, res, mode = self._solve_from_candidates(
            c10, c21, c20, image_hw=key_np.shape[:2])
        info = {'res': res, 'mode': mode,
                'unwrapped': mode.endswith('_unwrapped'),
                'c10': c10[:3], 'c21': c21[:3], 'c20': c20[:3]}
        return dx1, dy1, dx2, dy2, info

    @torch.no_grad()
    def align_batch(self, key, ref1, ref2, return_displacement=False):
        """三帧一致对齐 (tensor 批量, 模型集成接口)。

        Args:
            key, ref1, ref2: (B, 3, H, W) float32, 值域 [0, 1]
            return_displacement: 是否返回位移
        Returns:
            (a1, a2) 对齐后的 ref1/ref2 (边界填 0);
            return_displacement=True 时返回 (a1, a2, dx1, dy1, dx2, dy2)
        """
        B = key.shape[0]
        if B == 0:
            empty = torch.zeros_like(ref1)
            if return_displacement:
                z = torch.zeros(0, device=key.device)
                return empty, empty.clone(), z, z, z, z
            return empty, empty.clone()

        c10s = self._grad_corr_candidates_batch(ref1, key, self.k)
        c21s = self._grad_corr_candidates_batch(ref2, ref1, self.k)
        c20s = self._grad_corr_candidates_batch(ref2, key, self.k)

        dx1s, dy1s, dx2s, dy2s = [], [], [], []
        for b in range(B):
            dx1, dy1, dx2, dy2, res, mode = self._solve_from_candidates(
                c10s[b], c21s[b], c20s[b], image_hw=key.shape[-2:])
            dx1s.append(dx1)
            dy1s.append(dy1)
            dx2s.append(dx2)
            dy2s.append(dy2)

        dx1 = torch.tensor(dx1s, dtype=torch.float32, device=key.device)
        dy1 = torch.tensor(dy1s, dtype=torch.float32, device=key.device)
        dx2 = torch.tensor(dx2s, dtype=torch.float32, device=key.device)
        dy2 = torch.tensor(dy2s, dtype=torch.float32, device=key.device)
        a1 = FFTAlign._apply_shift(ref1, dx1, dy1)
        a2 = FFTAlign._apply_shift(ref2, dx2, dy2)

        if return_displacement:
            return a1, a2, dx1, dy1, dx2, dy2
        return a1, a2
