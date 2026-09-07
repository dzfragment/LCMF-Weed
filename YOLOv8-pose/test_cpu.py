#!/usr/bin/env python3
"""
CPU-compatible test script for YOLOv8-pose.
Runs pose estimation on a single image and saves the result.
"""

import argparse
import numpy
import torch
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import util

# COCO keypoint skeleton and colors
PALETTE = numpy.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
], dtype=numpy.uint8)

SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

KPT_COLOR = PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
LIMB_COLOR = PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


def preprocess(image_path, input_size=640, device='cpu'):
    """Preprocess image for YOLOv8-pose inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    shape = image.shape[:2]  # [height, width]

    # Letterbox resize
    r = min(input_size / shape[0], input_size / shape[1])
    pad_w = int(round(shape[1] * r))
    pad_h = int(round(shape[0] * r))
    dw = input_size - pad_w
    dh = input_size - pad_h
    dw = int(dw // 2)
    dh = int(dh // 2)

    if (pad_w, pad_h) != (shape[1], shape[0]):
        image = cv2.resize(image, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)

    image = cv2.copyMakeBorder(image, dh, input_size - pad_h - dh,
                               dw, input_size - pad_w - dw,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # HWC -> CHW, BGR -> RGB, to tensor
    image_tensor = image.transpose((2, 0, 1))[::-1]  # BGR to RGB
    image_tensor = numpy.ascontiguousarray(image_tensor)
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0)
    image_tensor = image_tensor.to(device).float() / 255.0

    return image_tensor, shape, (dw, dh, r)


def draw_results(frame, outputs, shape, pad_info, kpt_shape=(17, 3)):
    """Draw bounding boxes and keypoints on the frame."""
    dw, dh, r = pad_info

    for output in outputs:
        if len(output) == 0:
            continue

        output = output.clone()

        box_output = output[:, :6]
        if output.shape[1] > 6:
            kps_output = output[:, 6:].view(len(output), kpt_shape[0], kpt_shape[1])
        else:
            kps_output = output[:, 6:]

        # Unpad boxes
        scale_h = frame.shape[0] / (frame.shape[0] * r)
        scale_w = frame.shape[1] / (frame.shape[1] * r)
        box_output[:, [0, 2]] -= dw
        box_output[:, [1, 3]] -= dh
        box_output[:, :4] /= r
        box_output[:, 0].clamp_(0, shape[1])
        box_output[:, 1].clamp_(0, shape[0])
        box_output[:, 2].clamp_(0, shape[1])
        box_output[:, 3].clamp_(0, shape[0])

        if len(kps_output):
            kps_output[..., 0] -= dw
            kps_output[..., 1] -= dh
            kps_output[..., 0] /= r
            kps_output[..., 1] /= r
            kps_output[..., 0].clamp_(0, shape[1])
            kps_output[..., 1].clamp_(0, shape[0])

        # Draw boxes
        for box in box_output.cpu().numpy():
            x1, y1, x2, y2, score, cls_id = box
            if score < 0.25:
                continue
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw keypoints and skeleton
        for kpt in reversed(kps_output.cpu().numpy()):
            for i, k in enumerate(kpt):
                color_k = [int(x) for x in KPT_COLOR[i]]
                x_c, y_c = int(k[0]), int(k[1])
                if x_c % shape[1] != 0 and y_c % shape[0] != 0:
                    if kpt.shape[1] == 3 and k[2] < 0.5:
                        continue
                    cv2.circle(frame, (x_c, y_c), 4, color_k, -1, lineType=cv2.LINE_AA)

            for i, sk in enumerate(SKELETON):
                pos1 = (int(kpt[sk[0] - 1, 0]), int(kpt[sk[0] - 1, 1]))
                pos2 = (int(kpt[sk[1] - 1, 0]), int(kpt[sk[1] - 1, 1]))
                if kpt.shape[1] == 3:
                    if kpt[sk[0] - 1, 2] < 0.5 or kpt[sk[1] - 1, 2] < 0.5:
                        continue
                if (pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or
                    pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0):
                    continue
                cv2.line(frame, pos1, pos2, [int(x) for x in LIMB_COLOR[i]],
                         thickness=2, lineType=cv2.LINE_AA)

    return frame


def run_inference(model, image_tensor, conf_thres=0.25, iou_thres=0.7):
    """Run model inference and NMS."""
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    outputs = util.non_max_suppression(outputs, conf_thres, iou_thres, model.head.nc)
    return outputs


def main():
    parser = argparse.ArgumentParser(description='YOLOv8-pose CPU Test')
    parser.add_argument('--image', type=str, default='test_image.jpg',
                        help='Path to input image')
    parser.add_argument('--weights', type=str, default='./weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--output', type=str, default='output_pose.jpg',
                        help='Path to save output image')
    parser.add_argument('--input-size', type=int, default=640,
                        help='Model input size')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold for NMS')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Running on device: {device}")

    # Load model
    print(f"Loading model from {args.weights}...")
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model = ckpt['model'].float()
    model.to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Stride: {model.stride.cpu().numpy()}")

    # Preprocess
    print(f"Preprocessing image: {args.image}")
    image_tensor, original_shape, pad_info = preprocess(
        args.image, args.input_size, device
    )

    # Inference
    print("Running inference...")
    outputs = run_inference(model, image_tensor, args.conf, args.iou)

    # Draw results
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    num_persons = 0
    for out in outputs:
        num_persons += len(out)
    print(f"Detected {num_persons} person(s)")

    frame = draw_results(frame, outputs, original_shape, pad_info)

    cv2.imwrite(args.output, frame)
    print(f"Result saved to: {args.output}")


if __name__ == '__main__':
    main()
