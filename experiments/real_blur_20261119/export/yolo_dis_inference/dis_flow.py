"""Exp4 DIS optical flow, point projection, and fusion."""

from time import perf_counter

import cv2
import numpy as np

SIZE = 1024
FLOW_SIZE = SIZE // 2
FLOW_SCALE = FLOW_SIZE / SIZE
ROUND_TRIP_PX = 3.
POINT_NMS_PX = 8.
MAX_POINTS = 300
CONF = .25


def sample_displacement(flow, points):
    """Sample half-resolution flow at full-resolution points and restore its units."""
    xy = np.asarray((points + .5) * FLOW_SCALE - .5, dtype=np.float32)
    vectors = cv2.remap(flow, xy[:, :1].copy(), xy[:, 1:].copy(), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return vectors[:, 0] / FLOW_SCALE


def project(predictions, forward, backward):
    """Return consistent reference predictions in keyframe coordinates."""
    if not len(predictions):
        return predictions.copy(), 0

    points = predictions[:, 5:7]
    shift = sample_displacement(forward, points)
    mapped = points + shift
    inside = np.flatnonzero((np.isfinite(mapped) & (mapped >= 0) & (mapped < SIZE)).all(axis=1))
    if not len(inside):
        return predictions[:0].copy(), 0

    returned = mapped[inside] + sample_displacement(backward, mapped[inside])
    error = np.linalg.norm(returned - points[inside], axis=1)
    keep = inside[error <= ROUND_TRIP_PX]
    projected = predictions[keep].copy()
    projected[:, :4] += np.tile(shift[keep], 2)
    projected[:, 5:7] = mapped[keep]
    return projected, len(inside)


def point_nms(predictions):
    """Score-ordered 8 px suppression, matching the previous late-fusion radius."""
    candidates = np.concatenate(predictions)
    slots = np.repeat(np.arange(len(predictions)), [len(view) for view in predictions])
    order = np.argsort(-candidates[:, 4], kind='stable')
    keep = []
    while len(order) and len(keep) < MAX_POINTS:
        winner, order = order[0], order[1:]
        keep.append(winner)
        delta = candidates[order, 5:7] - candidates[winner, 5:7]
        order = order[np.sum(delta**2, axis=1) > POINT_NMS_PX**2]
    return candidates[keep], slots[keep]


def to_gray(frame):
    # Resize first to keep the flow preprocessing at half resolution.
    rgb = frame.permute(1, 2, 0).contiguous().numpy()
    small = cv2.resize(rgb, (FLOW_SIZE, FLOW_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def fuse(frames, views, dis):
    tick = perf_counter()
    key = to_gray(frames[0])
    aligned = [views[0]]
    ref_diagnostics = []
    for frame, pred in zip(frames[1:], views[1:], strict=True):
        inside = 0
        projected = pred
        if len(pred):
            reference = to_gray(frame)
            if not np.array_equal(reference, key):
                forward = dis.calc(reference, key, None)
                backward = dis.calc(key, reference, None)
                projected, inside = project(pred, forward, backward)
            else:
                inside = len(pred)
        aligned.append(projected)
        ref_diagnostics.append(dict(candidates=len(pred), inside_keyframe=inside,
                                    kept_after_flow=len(projected)))
    fused, source = point_nms(aligned)
    elapsed_ms = (perf_counter() - tick) * 1000
    counts = np.bincount(source[fused[:, 4] >= CONF], minlength=3)
    return fused, source, dict(refs=ref_diagnostics, selected_conf25_by_slot=counts.tolist(),
                               flow_and_fusion_ms=elapsed_ms)
