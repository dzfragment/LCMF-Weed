#!/usr/bin/env python3
"""
MultiFrameYOLO 推理脚本

支持三种模式:
  --mode single    单序列推理（三帧图片 → 检测结果）
  --mode video     视频推理（逐帧检测 + 输出标注视频）
  --mode streaming 流式推理（特征缓存复用，更快）

用法:
  # 单序列
  python inference.py --mode single --input seq_0001/ --output result.jpg

  # 视频
  python inference.py --mode video --input video.mp4 --output result.mp4

  # 流式视频（更快）
  python inference.py --mode streaming --input video.mp4 --output result.mp4
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_WORKSPACE, 'YOLOv8-pose'))
sys.path.insert(0, _SCRIPT_DIR)

from models.mf_yolo import MultiFrameYOLO, StreamingInference
from utils import util

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════
# 预处理
# ═══════════════════════════════════════════════════════════

def preprocess_image(img, input_size=1024):
    """
    Resize + pad 到 input_size，返回 (1, 3, H, W) tensor [0,1]。
    同时返回 pad 信息用于后处理坐标映射。
    """
    h, w = img.shape[:2]
    r = min(input_size / h, input_size / w)
    r = min(r, 1.0)  # 不放大

    pad_w = int(round(w * r))
    pad_h = int(round(h * r))
    dw = (input_size - pad_w) / 2
    dh = (input_size - pad_h) / 2

    if (w, h) != (pad_w, pad_h):
        img = cv2.resize(img, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # HWC BGR → CHW RGB → tensor [0,1]
    tensor = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy())
    tensor = tensor.float().unsqueeze(0) / 255.0  # (1, 3, H, W)

    pad_info = {'pad_w': pad_w, 'pad_h': pad_h, 'dw': dw, 'dh': dh,
                'ratio': r, 'orig_shape': (h, w)}
    return tensor, pad_info


def preprocess_sequence(frame0_path, frame1_path, frame2_path, input_size=1024):
    """加载三帧并预处理为 (1, 3, 3, H, W) tensor"""
    frames = []
    for path in [frame0_path, frame1_path, frame2_path]:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'Cannot read {path}')
        tensor, pad_info = preprocess_image(img, input_size)
        frames.append(tensor.squeeze(0))

    batch = torch.stack(frames, dim=0).unsqueeze(0)  # (1, 3, 3, H, W)
    return batch, pad_info


# ═══════════════════════════════════════════════════════════
# 后处理
# ═══════════════════════════════════════════════════════════

def postprocess(outputs, pad_info, conf_thres=0.25, iou_thres=0.7):
    """
    NMS → 像素坐标映射。

    返回: list of {'box': [x1,y1,x2,y2], 'conf': float, 'kpt': [x,y]}
    """
    outputs = util.non_max_suppression(outputs, conf_thres, iou_thres, nc=1)

    results = []
    for output in outputs:
        if output is None or output.shape[0] == 0:
            continue
        pred = output.cpu().numpy()
        for det in pred:
            x1, y1, x2, y2, conf, _, kx, ky = det[:8]
            # 从 padded 坐标映射回原图坐标
            pad_w, pad_h = pad_info['pad_w'], pad_info['pad_h']
            dw, dh = pad_info['dw'], pad_info['dh']
            orig_h, orig_w = pad_info['orig_shape']

            # 去掉 padding，缩回原图比例
            x1 = (x1 - dw) / pad_w * orig_w
            y1 = (y1 - dh) / pad_h * orig_h
            x2 = (x2 - dw) / pad_w * orig_w
            y2 = (y2 - dh) / pad_h * orig_h
            kx = (kx - dw) / pad_w * orig_w
            ky = (ky - dh) / pad_h * orig_h

            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            results.append({
                'box': [x1, y1, x2, y2],
                'conf': float(conf),
                'kpt': [float(kx), float(ky)],
            })
    return results


def draw_results(img, results, color=(0, 255, 0), thickness=2):
    """在图像上绘制检测结果"""
    img = img.copy()
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r['box']]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f'{r["conf"]:.2f}'
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        kx, ky = int(r['kpt'][0]), int(r['kpt'][1])
        cv2.circle(img, (kx, ky), 3, (0, 0, 255), -1)
    return img


# ═══════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════

def load_model(checkpoint_path, pretrained_path=None):
    """加载 MultiFrameYOLO 模型和权重"""
    if pretrained_path is None:
        pretrained_path = os.path.join(
            _WORKSPACE,
            'experiments/single_frame_baseline/results/yolo_v8_n_p2_lite_p4/seed42/best.pt')

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if ckpt.get('base_model') != 'p2l_p4':
        raise ValueError(
            f'仅支持新版 P2L-P4 checkpoint，收到 base_model={ckpt.get("base_model")}')
    fusion_levels = ckpt.get('fusion_levels')
    if fusion_levels not in (['p3'], ['p4'], ['p3', 'p4']):
        raise ValueError(
            f'Checkpoint fusion_levels={fusion_levels} 非法，仅支持 p3、p4 或 p3,p4')

    # 从 checkpoint 读取 fusion_version / align_mode / use_sharpener
    fv = ckpt.get('fusion_version', 2)
    am = ckpt.get('align_mode', None)
    sp = ckpt.get('use_sharpener', False)
    global_align_stage = ckpt.get('global_align_stage', 'none')
    # 兼容第一版 feature checkpoint 元数据。
    if ckpt.get('fft_feature_align', False):
        global_align_stage = 'feature'
    fft_downsample = ckpt.get('fft_downsample', 4)
    fusion_gate_type = ckpt.get('fusion_gate_type', 'spatial')
    dav2_single = ckpt.get('dav2_single', False)
    dav2_triple = ckpt.get('dav2_triple', False)
    if dav2_single and dav2_triple:
        raise ValueError('Checkpoint 同时声明 dav2_single 与 dav2_triple')

    model = MultiFrameYOLO(
        num_classes=1, kpt_shape=(1, 2),
        fusion_levels=fusion_levels,
        pretrained_path=pretrained_path,
        fusion_version=fv,
        align_mode=am,
        use_sharpener=sp,
        fusion_gate_type=fusion_gate_type,
        global_align_stage='feature' if global_align_stage == 'feature' else 'none',
        fft_downsample=fft_downsample,
    )
    model = model.to(DEVICE)
    if dav2_single:
        from train import apply_dav2_single
        apply_dav2_single(model)
    elif dav2_triple:
        from train import apply_dav2_triple
        apply_dav2_triple(model)

    sd = ckpt.get('ema_model') or ckpt.get('model')
    if isinstance(sd, dict):
        model.load_state_dict(sd, strict=True)
    else:
        model = sd.float().to(DEVICE)

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════
# 推理模式
# ═══════════════════════════════════════════════════════════

def infer_single(model, input_path, output_path, input_size=1024):
    """单序列推理（目录或三帧文件）"""
    print(f'[single] input={input_path}')

    # 自动检测：目录或文件列表
    if os.path.isdir(input_path):
        frame0 = os.path.join(input_path, 'frame_0.jpg')
        frame1 = os.path.join(input_path, 'frame_1.jpg')
        frame2 = os.path.join(input_path, 'frame_2.jpg')
    else:
        # 假设是逗号分隔的三帧路径
        parts = input_path.split(',')
        if len(parts) != 3:
            raise ValueError('single mode: input 应为序列目录或 "f0,f1,f2" 三帧路径')
        frame0, frame1, frame2 = [p.strip() for p in parts]

    frames_batch, pad_info = preprocess_sequence(frame0, frame1, frame2, input_size)
    frames_batch = frames_batch.to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(frames_batch)

    results = postprocess(outputs.float(), pad_info)

    # 可视化
    img = cv2.imread(frame0)
    img = draw_results(img, results)
    cv2.imwrite(output_path, img)

    print(f'  {len(results)} detections → {output_path}')
    for r in results:
        print(f'    box={[f"{v:.0f}" for v in r["box"]]}  conf={r["conf"]:.3f}  '
              f'kpt=({r["kpt"][0]:.0f},{r["kpt"][1]:.0f})')


def infer_video(model, input_path, output_path, input_size=1024):
    """批量视频推理：每3帧一组"""
    print(f'[video] input={input_path}')

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    buffer = []  # 保留最近3帧
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)
        frame_count += 1

        if len(buffer) < 3:
            out.write(frame)
            continue

        # 取三帧推理
        f0, f1, f2 = buffer[-3], buffer[-2], buffer[-1]
        try:
            t0 = preprocess_image(f0, input_size)[0]
            t1 = preprocess_image(f1, input_size)[0]
            t2 = preprocess_image(f2, input_size)[0]
            frames_batch = torch.cat([t0, t1, t2], dim=0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(frames_batch)
            results = postprocess(outputs.float(), preprocess_image(f0, input_size)[1])
            out_frame = draw_results(f1, results)  # 标注在中间帧
        except Exception as e:
            print(f'  [WARN] frame {frame_count}: {e}')
            out_frame = f1

        out.write(out_frame)

        # 滑动窗口
        if len(buffer) > 5:
            buffer.pop(0)

    cap.release()
    out.release()
    print(f'  {frame_count} frames → {output_path}')


def infer_streaming(model, input_path, output_path, input_size=1024):
    """流式推理：特征缓存复用，每帧只过一次 backbone"""
    print(f'[streaming] input={input_path}')

    stream = StreamingInference(model, window_size=3)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    t_total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        tensor, pad_info = preprocess_image(frame, input_size)
        tensor = tensor.to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = stream.infer(tensor)
                if outputs is not None:
                    outputs = outputs.float()
                else:
                    # 缓存不足，直接写原帧
                    out.write(frame)
                    stream.update_cache(tensor)
                    continue
        t_total += time.time() - t0

        results = postprocess(outputs, pad_info)
        out_frame = draw_results(frame, results)
        out.write(out_frame)

        stream.update_cache(tensor)

    cap.release()
    out.release()

    avg_time = t_total / max(frame_count, 1) * 1000
    print(f'  {frame_count} frames → {output_path}  '
          f'avg={avg_time:.1f}ms/frame  '
          f'fps={1000/avg_time:.1f}')


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='MultiFrameYOLO 推理')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'video', 'streaming'],
                        help='推理模式')
    parser.add_argument('--input', type=str, required=True,
                        help='输入：序列目录/视频路径/三帧路径(逗号分隔)')
    parser.add_argument('--output', type=str, default='result.jpg',
                        help='输出路径')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(_SCRIPT_DIR,
                            'weights/mf_stage2_best.pt'),
                        help='模型权重路径')
    parser.add_argument('--input-size', type=int, default=1024,
                        help='输入尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print(f'Device: {DEVICE}')
    print(f'Checkpoint: {args.checkpoint}')

    model = load_model(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    if args.mode == 'single':
        infer_single(model, args.input, args.output, args.input_size)
    elif args.mode == 'video':
        infer_video(model, args.input, args.output, args.input_size)
    elif args.mode == 'streaming':
        infer_streaming(model, args.input, args.output, args.input_size)


if __name__ == '__main__':
    main()
