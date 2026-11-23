"""Standalone YOLOv8n-pose and DIS point fusion for three related images."""

from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from dis_flow import CONF, MAX_POINTS, SIZE, fuse


CANDIDATE_CONF = .001
BOX_NMS_IOU = .7
SOURCE_NAMES = ('keyframe', 'reference_1', 'reference_2')


@dataclass(frozen=True)
class Geometry:
    """The same letterbox and coordinate mapping for all three images."""

    width: int
    height: int
    scaled_width: int
    scaled_height: int
    left: int
    top: int

    @property
    def native(self):
        return self.width == self.height == SIZE

    def valid_points(self, predictions):
        x, y = predictions[:, 5], predictions[:, 6]
        return ((x >= self.left) & (x < self.left + self.scaled_width)
                & (y >= self.top) & (y < self.top + self.scaled_height))

    def restore(self, predictions):
        if self.native:
            return predictions
        restored = predictions.copy()
        restored[:, [0, 2, 5]] = (restored[:, [0, 2, 5]] - self.left) * (self.width / self.scaled_width)
        restored[:, [1, 3, 6]] = (restored[:, [1, 3, 6]] - self.top) * (self.height / self.scaled_height)
        return restored


@dataclass(frozen=True)
class Result:
    """Predictions are x1, y1, x2, y2, confidence, keypoint_x, keypoint_y."""

    predictions: np.ndarray
    source_slots: np.ndarray
    references: list
    inference_ms: float
    flow_and_fusion_ms: float


def prepare_frames(keyframe, reference_1, reference_2):
    """Convert three same-size uint8 BGR images to RGB tensors on the CPU."""
    images = (keyframe, reference_1, reference_2)
    shape = getattr(keyframe, 'shape', ())
    if (len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 1
            or any(not isinstance(image, np.ndarray) or image.shape != shape or image.dtype != np.uint8
                   for image in images)):
        raise ValueError('Expected three same-size H×W×3 uint8 BGR images.')
    height, width = shape[:2]
    ratio = SIZE / max(height, width)
    sw, sh = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (SIZE - sw) // 2, (SIZE - sh) // 2
    geometry = Geometry(width, height, sw, sh, left, top)
    tensors = []
    for image in images:
        if not geometry.native:
            canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
            canvas[top:top + sh, left:left + sw] = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
            image = canvas
        rgb_chw = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        tensors.append(torch.from_numpy(rgb_chw))
    return torch.stack(tensors), geometry


def load_ultralytics():
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('COMET_MODE', 'DISABLED')
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression

    return YOLO, non_max_suppression


class YoloDIS:
    """Keep a single detector and a DIS instance resident between triplets."""

    def __init__(self, device='auto', cpu_threads=1, *, weights):
        if cpu_threads < 1:
            raise ValueError('cpu_threads must be positive.')
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Select auto, cpu, or a CUDA device such as cuda:0.')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this PyTorch installation.')
        if not hasattr(cv2, 'DISOpticalFlow_create'):
            raise RuntimeError('This OpenCV build does not provide DISOpticalFlow_create.')
        cv2.setNumThreads(cpu_threads)
        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        weight_path = Path(weights)
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)
        YOLO, self.nms = load_ultralytics()
        self.model = YOLO(str(weight_path.resolve())).model.to(self.device).float().eval()
        head = self.model.model[-1]
        if getattr(head, 'nc', None) != 1 or tuple(getattr(head, 'kpt_shape', ())) != (1, 2):
            raise ValueError('Expected a one-class pose checkpoint with a single 2D keypoint.')
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _detect(self, frame, geometry):
        image = frame.unsqueeze(0).to(self.device).float() / 255
        detection, = self.nms(self.model(image), conf_thres=CANDIDATE_CONF, iou_thres=BOX_NMS_IOU,
                              nc=1, max_det=MAX_POINTS, max_time_img=10.)
        if detection.shape[1] != 8:
            raise RuntimeError(f'Unexpected pose output with {detection.shape[1]} columns.')
        results = detection[:, [0, 1, 2, 3, 4, 6, 7]].cpu().numpy().astype(np.float32)
        keep = (geometry.valid_points(results) & (results[:, 2] - results[:, 0] > 1)
                & (results[:, 3] - results[:, 1] > 1))
        return results[keep]

    @torch.inference_mode()
    def infer_prepared(self, frames, geometry, *, confidence=CONF):
        """Infer from prepare_frames output. Timing includes three forwards and CPU fusion."""
        if not CANDIDATE_CONF <= confidence <= 1:
            raise ValueError(f'confidence must be in [{CANDIDATE_CONF}, 1].')
        self._sync()
        start = perf_counter()
        views = [self._detect(frame, geometry) for frame in frames]
        predictions, slots, info = fuse(frames, views, self.dis)
        keep = (predictions[:, 4] >= confidence) & geometry.valid_points(predictions)
        restored = geometry.restore(predictions[keep])
        slots = slots[keep]
        self._sync()
        return Result(restored, slots, info['refs'], (perf_counter() - start) * 1000,
                      info['flow_and_fusion_ms'])

    def predict(self, keyframe, reference_1, reference_2, *, confidence=CONF):
        """Return keyframe-coordinate predictions for three same-size BGR images."""
        frames, geometry = prepare_frames(keyframe, reference_1, reference_2)
        return self.infer_prepared(frames, geometry, confidence=confidence)
