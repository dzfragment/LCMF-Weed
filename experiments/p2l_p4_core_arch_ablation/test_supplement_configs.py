#!/usr/bin/env python3
"""Lightweight structure/gradient/checkpoint tests for supplement experiments; no dataset training."""
from __future__ import annotations
import os
import sys
import tempfile
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))  # 仓库根
PROJECT = os.path.join(REPO, 'multi_frame_fusion')
sys.path.insert(0, PROJECT)

from models.mf_yolo import MultiFrameYOLO
from models.modules import DeformAlignV2Triple
from train import apply_dav2_triple, load_ckpt, save_ckpt

PRETRAINED = os.path.join(
    os.path.dirname(os.path.dirname(ROOT)), 'experiments', 'single_frame_baseline', 'results',
    'yolo_v8_n_p2_lite_p4', 'seed42', 'best.pt')


def output_sum(output):
    if isinstance(output, torch.Tensor):
        return output.sum()
    if isinstance(output, (tuple, list)):
        return sum(output_sum(x) for x in output)
    raise TypeError(type(output))


def test_gate(gate_type):
    model = MultiFrameYOLO(
        fusion_levels=['p4'], pretrained_path=PRETRAINED,
        fusion_gate_type=gate_type,
        global_align_stage='feature', fft_downsample=4)
    model.train()
    output = model(torch.rand(1, 3, 3, 128, 128))
    output_sum(output).backward()
    finite = sum(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.fusion.fusion['p4'].parameters())
    assert finite > 0, f'{gate_type}: no finite fusion gradients'
    print(f'PASS gate={gate_type}: finite_grad_params={finite}')


def test_triple_roundtrip():
    model = MultiFrameYOLO(
        fusion_levels=['p4'], pretrained_path=PRETRAINED,
        fusion_gate_type='spatial', global_align_stage='feature', fft_downsample=4)
    apply_dav2_triple(model)
    assert isinstance(model.fusion.align_ref1['p4'], DeformAlignV2Triple)
    model.train()
    output_sum(model(torch.rand(1, 3, 3, 128, 128))).backward()
    s3_grads = [p.grad for n, p in model.named_parameters() if '.s3_' in n]
    assert s3_grads and all(g is not None and torch.isfinite(g).all() for g in s3_grads)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
    scaler = torch.cuda.amp.GradScaler()
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        save_ckpt(
            model, optimizer, scheduler, scaler, None, 1, 1.0, 0.1, path,
            fft_feature_align=True, fft_downsample=4,
            dav2_triple=True, fusion_gate_type='spatial')
        loaded, state = load_ckpt(
            path, device='cpu', fusion_levels=['p4'],
            fusion_gate_type='spatial', fft_feature_align=True,
            fft_downsample=4, dav2_triple=True)
        assert isinstance(loaded.fusion.align_ref1['p4'], DeformAlignV2Triple)
        assert state['epoch'] == 1
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        assert ckpt['dav2_stages'] == 3
        assert ckpt['fusion_gate_type'] == 'spatial'
    finally:
        os.remove(path)
    print(f'PASS DAV2 triple: s3_grad_tensors={len(s3_grads)}, strict roundtrip')


def main():
    assert os.path.isfile(PRETRAINED), PRETRAINED
    for gate in ('channel', 'convgru', 'tsa', 'tsa_add'):
        test_gate(gate)
    test_triple_roundtrip()
    print('PASS: all supplement configuration tests')


if __name__ == '__main__':
    main()
