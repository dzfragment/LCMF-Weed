#!/usr/bin/env python3
"""PAD 全基座冒烟测试: 7 个单帧基座 forward + point_loss + decode + 参数量/MACs
+ COCO 部分加载张量数. 不写结果文件, 只打印表格."""
import os
import sys
import time

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP3_DIR = os.path.dirname(_SCRIPT_DIR)
YOLO_DIR = os.path.join(EXP3_DIR, 'YOLOv8-pose')
sys.path.insert(0, YOLO_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from pad_core import (PAD_SF_BASES, PADSingleFrame, point_loss,  # noqa: E402
                      decode_predictions)

BASES = ['v8n', 'p3', 'p34', 'p34l', 'p2', 'p2l', 'p2l_p4', 'p2l_p4_only']
COCO_CKPT = os.path.join(EXP3_DIR, 'YOLOv8-pose', 'v8_n_pose.pt')

DUMMY_TARGETS = {
    'idx': torch.zeros(3, dtype=torch.long),
    'cls': torch.zeros(3, 1),
    'box': torch.tensor([[.5, .5, .1, .1], [.3, .3, .08, .08],
                         [.7, .6, .12, .1]]),
    'kpt': torch.tensor([[[.5, .5, 1.]], [[.3, .3, 1.]], [[.7, .6, 1.]]]),
}


def coco_load_count(model):
    if not os.path.exists(COCO_CKPT):
        return -1
    ck = torch.load(COCO_CKPT, map_location='cpu', weights_only=False)
    sd = ck['model'].state_dict() if hasattr(ck['model'], 'state_dict') \
        else ck['model']
    md = model.state_dict()
    return sum(1 for k, v in sd.items()
               if k in md and v.shape == md[k].shape)


def macs_of(model):
    try:
        from thop import profile
        x = torch.zeros(1, 3, 1024, 1024)
        macs, _ = profile(model.cpu().eval(), inputs=(x,), verbose=False)
        return macs / 1e9
    except Exception as e:
        print(f'  [thop skip: {e}]')
        return float('nan')


def main():
    device = 'cuda'
    ok = True
    print(f'{"base":8s} {"params":>10s} {"MACs(G)":>9s} {"coco":>5s} '
          f'{"fwd_ms":>7s} {"loss":>7s} {"n_pred":>6s}  head_levels')
    for name in BASES:
        try:
            model = PADSingleFrame(name, {'use_size': False}).to(device)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            x = torch.zeros(2, 3, 1024, 1024, device=device)
            t0 = time.time()
            with torch.no_grad():
                out = model(x)
            fwd_ms = (time.time() - t0) * 1000 / 2
            hm, off = out[0], out[1]
            assert hm.shape == (2, 1, 128, 128), hm.shape
            assert off.shape == (2, 2, 128, 128), off.shape
            loss, _parts, _n_pos = point_loss(out, DUMMY_TARGETS, None)
            preds = decode_predictions(out, input_size=1024, conf=0.001, topk=100)
            n_pred = sum(len(p) for p in preds)
            macs = macs_of(model)
            n_coco = coco_load_count(model)
            levels = PAD_SF_BASES[name][2]
            print(f'{name:8s} {n_params:10,d} {macs:9.2f} {n_coco:5d} '
                  f'{fwd_ms:7.1f} {float(loss):7.3f} {n_pred:6d}  {levels}')
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            ok = False
            print(f'{name:8s} FAIL: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
    print('ALL OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
