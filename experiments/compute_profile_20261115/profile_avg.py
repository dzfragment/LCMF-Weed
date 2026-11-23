"""Supplementary true-average profile using the established MAC/DCN convention."""
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent
import profile_models as p
from profile_models import common
import torch
from torch import nn

OUT = BASE / 'avg_supplement'
OUT.mkdir(exist_ok=True)
p.HERE = OUT


class CachedBackbone(nn.Module):
    def __init__(self, original, history):
        super().__init__()
        self.original = original
        self.history = history
        self.calls = 0
        self.actual_calls = 0

    def reset(self):
        self.calls = self.actual_calls = 0

    def forward(self, x):
        index = self.calls
        self.calls += 1
        if index == 0:
            self.actual_calls += 1
            return self.original(x)
        assert index in (1, 2)
        return self.history[index - 1]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    checkpoint = common.avg_run(42) / 'results/13_fusion_avg/pad_stage3_last.pt'
    arm = common.AVG_ARM
    common.assert_checkpoint(checkpoint, arm, epoch=250)
    model, _ = p.t.load_ckpt(str(checkpoint), common.args_for(common.architecture_flags(arm)),
                             'cuda:0', prefer_ema=False)
    model.float().eval()
    ds = p.t.MultiFrameDataset(str(common.DATA), split='test', input_size=1024,
                               augment=False, params=None)
    frames = ds[0][0].unsqueeze(0).to('cuda:0').float() / 255
    full = p.measure(model, frames, '13_fusion_avg_full', checkpoint, 3)
    backbone = p.measure(model.backbone, frames[:, 0], 'backbone_p2l', checkpoint, 1)
    expected = tuple(v.detach().clone() for v in model(frames))
    history = tuple(tuple(v.detach() for v in model.backbone(frames[:, i])) for i in (1, 2))
    cache = CachedBackbone(model.backbone, history)
    model.backbone = cache
    handle = model.register_forward_pre_hook(lambda module, args: cache.reset())
    actual = model(frames)
    max_error = max(float((a-b).abs().max()) for a, b in zip(expected, actual))
    assert max_error == 0
    cached = p.measure(model, frames, '13_fusion_avg_cached', checkpoint, 3)
    assert cache.calls == 3 and cache.actual_calls == 1
    assert abs(cached['gflops'] - (full['gflops'] - 2*backbone['gflops'])) < 1e-8
    handle.remove()
    main_full = json.loads((BASE / '01_main.json').read_text())
    main_cached = json.loads((BASE / 'cached/01_main.json').read_text())
    comparisons = []
    for mode, concat, avg in [('full', main_full, full), ('cached', main_cached, cached)]:
        comparisons.append(dict(mode=mode, concat_gflops=concat['gflops'], avg_gflops=avg['gflops'],
            saved_gflops=concat['gflops']-avg['gflops'],
            reduction_percent=100*(concat['gflops']-avg['gflops'])/concat['gflops'],
            concat_parameters=concat['parameters'], avg_parameters=avg['parameters']))
    p.write_json(OUT / 'summary.json', dict(comparisons=comparisons, full=full, cached=cached,
        validation=dict(passed=True, full_vs_cached_max_abs_error=max_error,
                        actual_new_backbone_calls=cache.actual_calls),
        protocol=json.loads((BASE / 'summary.json').read_text())['protocol']))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
