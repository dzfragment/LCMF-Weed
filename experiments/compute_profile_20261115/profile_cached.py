"""Warm-cache full model profile: one new backbone, two cached RAW backbones.

The actual model's FFT/G/DCN/fusion/FPN/head path is kept unchanged.
The cache adapter is for measurement/parity checks, not a video frontend.
"""
from pathlib import Path
import json
import gc
import hashlib
from datetime import datetime, timezone

import profile_models as p
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
OUT = BASE / 'cached'
OUT.mkdir(exist_ok=True)
p.HERE = OUT


class CachedBackbone(nn.Module):
    def __init__(self, original, ref1, ref2):
        super().__init__()
        self.original = original
        self.history = (tuple(v.detach() for v in ref1), tuple(v.detach() for v in ref2))
        self.calls = 0
        self.new_forwards = 0

    def reset(self):
        self.calls = 0
        self.new_forwards = 0

    def forward(self, x):
        index = self.calls
        self.calls += 1
        if index == 0:
            self.new_forwards += 1
            return self.original(x)
        if index in (1, 2):
            return self.history[index - 1]
        raise RuntimeError('Unexpected additional backbone call')


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    full = json.loads((BASE / 'summary.json').read_text())
    full_by_name = {x['name']: x for x in full['models']}
    dataset = p.t.MultiFrameDataset(str(p.common.DATA), split='test', input_size=1024,
                                    augment=False, params=None)
    frames = dataset[0][0].unsqueeze(0).to('cuda:0').float()/255
    rows = []
    backbone = None
    for arm in p.common.ARMS:
        name = arm['name']
        ck = p.common.RESULTS / name / 'pad_stage3_last.pt'
        args = p.common.args_for(p.common.architecture_flags(arm))
        model, _ = p.t.load_ckpt(str(ck), args, 'cuda:0', prefer_ema=False)
        model = model.float().eval()
        if backbone is None:
            backbone = p.measure(model.backbone, frames[:, 0], 'backbone_p2l', ck, 1)
        # Cold reference includes all three raw backbones and every original alignment operation.
        expected = tuple(v.detach().clone() for v in model(frames))
        raw_ref1 = model.backbone(frames[:, 1])
        raw_ref2 = model.backbone(frames[:, 2])
        cache = CachedBackbone(model.backbone, raw_ref1, raw_ref2)
        model.backbone = cache
        def reset_before_forward(module, args):
            module.backbone.reset()
        handle = model.register_forward_pre_hook(reset_before_forward)
        actual = model(frames)
        errors = [float((a-b).abs().max()) for a,b in zip(expected,actual)]
        assert all(torch.allclose(a,b,rtol=1e-5,atol=1e-6) for a,b in zip(expected,actual)),(name,errors)
        assert cache.calls == 3 and cache.new_forwards == 1
        row = p.measure(model, frames, name, ck, 3)
        assert cache.calls == 3 and cache.new_forwards == 1
        estimate = full_by_name[name]['gflops'] - 2 * backbone['gflops']
        assert abs(row['gflops']-estimate) < 1e-8,(name,row['gflops'],estimate)
        assert row['parameters'] == full_by_name[name]['parameters']
        row.update(uncached_gflops=full_by_name[name]['gflops'],cached_gflops=row['gflops'],
            one_backbone_gflops=backbone['gflops'],saved_gflops=2*backbone['gflops'],
            reduction_percent=100*(full_by_name[name]['gflops']-row['gflops'])/full_by_name[name]['gflops'],
            actual_new_backbone_calls=cache.new_forwards,history_backbones_reused=2,
            full_vs_cached_max_abs_error=max(errors),formula_verified=True,
            cache_boundary='raw P2/P3/P4 backbone features BEFORE FFT/global residual/DCN alignment',
            fft_global_dcn_fusion_fpn_head='original forward path kept intact')
        p.write_json(OUT / (name+'.json'),row)
        rows.append(row)
        handle.remove()
        print(f"CACHE {name}: {row['cached_gflops']:.6f} G; saved {row['saved_gflops']:.6f} G ({row['reduction_percent']:.2f}%); max output error={max(errors):.3g}",flush=True)
        del model, actual, expected, raw_ref1, raw_ref2, cache
        gc.collect();torch.cuda.empty_cache()
    protocol = dict(created=datetime.now(timezone.utc).isoformat(),input_hw=[1024,1024],batch=1,
        logical_frames=3,new_frames_per_output=1,steady_state_only=True,
        gflops_definition=full['protocol']['gflops_definition'],
        included=full['protocol']['included'],excluded=full['protocol']['excluded'],
        no_frame_averaging='Cost is per new output frame, not divided by 3',
        startup='Historical feature construction and the first two frames are outside steady-state cost',
        cached_content='two historical RGB images plus two RAW multiscale backbone features; FPN and alignment not cached',
        existing_helper_limitations='Pinned StreamingInference rejects feature-FFT models, omits learned G, and infer()+update_cache() recomputes the new backbone. Measurements use the parity-verified adapter instead.',
        formula='cached = full_3frame - 2 * backbone_single_frame',
        original_full_profile_sha256=p.sha(BASE/'summary.json'))
    p.write_json(OUT/'summary.json',dict(protocol=protocol,backbone=backbone,models=rows))
    import csv
    with (OUT/'summary.csv').open('w',newline='') as f:
        fields=['name','parameters','uncached_gflops','cached_gflops','saved_gflops','reduction_percent','actual_new_backbone_calls','full_vs_cached_max_abs_error']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# 三帧模型的稳态缓存计算量','',
        '1024×1024，batch1；每来一张新图输出一次检测。复用两帧原始骨干特征，只计算新帧骨干；FFT/G/DCN/融合/FPN/头保留原模型路径。',
        f"单次骨干为 **{backbone['gflops']:.9f} GFLOPs**；每次输出节省 **{2*backbone['gflops']:.9f} GFLOPs**。",'',
        '缓存GFLOPs = 完整三帧GFLOPs − 2×单帧骨干GFLOPs。不是将整个三帧模型除以3，也不是扣除两个完整单帧模型。','',
        '| 模型 | 原三帧GFLOPs | 缓存后GFLOPs/新输出帧 | 降幅 |',
        '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['uncached_gflops']:.6f} | {r['cached_gflops']:.6f} | {r['reduction_percent']:.2f}% |")
    lines+=['','与上一轮保持同一MAC口径：已含DCN卷积，不含FFT、DCN插值/调制、grid_sample及多数逐元素运算。FFT/采样仍在实际forward中执行，只是这份MAC表不统计其算术开销。',
        '已对12组逐一实跑缓存forward：每次只有1次实际骨干调用，缓存与原完整三帧输出allclose，计算量与减去两次骨干的公式一致，参数量不变。',
        '缓存必须保存未对齐的原始P2/P3/P4特征；当前关键帧变化后，需要重新对齐历史特征。RGB缓存用于FFT。历史特征初始化开销不计入稳态表。',
        '当前冻结源码中的StreamingInference尚不能直接支持feature FFT；而且infer与update_cache各跑一次新帧骨干。其无FFT路径还未应用learned G。这里统计用户描述的正确缓存方案，由独立计算适配器验证，未更改历史训练/推理源码或模型权重。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    p.write_json(OUT/'validation.json',dict(passed=True,models=len(rows),one_new_backbone_per_forward=True,
        all_outputs_equivalent=True,formula_matched=True,parameters_unchanged=True,
        max_abs_output_error=max(r['full_vs_cached_max_abs_error'] for r in rows)))
    print('ALL 12 CACHED MODELS VERIFIED',flush=True)


if __name__=='__main__':
    main()
