"""Count the CPU optical-flow path with the existing predictions and prepared frames."""
import argparse
import ast
import ctypes
from pathlib import Path
import pickle
from time import perf_counter

import cv2
import numpy as np


class Frame:
    """Provide the uint8 layout methods used by the CPU path without loading CUDA."""
    def __init__(self, memory):
        self.memory = memory

    def permute(self, *dims):
        return Frame(self.memory.transpose(dims))

    def contiguous(self):
        return Frame(np.ascontiguousarray(self.memory))

    def numpy(self):
        return self.memory


def load_cpu_fusion(source):
    tree = ast.parse(source.read_text())
    constants = {'FLOW_SIZE', 'FLOW_SCALE', 'ROUND_TRIP_PX', 'POINT_NMS_PX', 'MAX_POINTS', 'CONF'}
    names = {'sample_displacement', 'project', 'point_nms', 'to_gray', 'fuse'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(x, ast.Name) and x.id in constants for x in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
    scope = dict(SIZE=1024, cv2=cv2, np=np, perf_counter=perf_counter)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope['fuse']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_cpu_fusion', type=Path)
    parser.add_argument('frames', type=Path)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('roi_library', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print('DIS_CPU_FEATURES', cv2.getCPUFeaturesLine(), flush=True)
    library = ctypes.CDLL(str(args.roi_library.resolve()))
    library.exp4_flops_begin.argtypes = library.exp4_flops_end.argtypes = ()
    library.exp4_flops_begin.restype = library.exp4_flops_end.restype = None
    frames = np.load(args.frames, mmap_mode='r')
    with args.predictions.open('rb') as handle:
        data = pickle.load(handle)
    seqs = data['provenance']
    fuse = load_cpu_fusion(args.source_cpu_fusion)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    # Exclude lazy OpenCV initialization while warming the same path.
    fuse([Frame(x) for x in frames[0]], seqs[0]['views'], dis)
    for i, (rgb, seq) in enumerate(zip(frames, seqs, strict=True)):
        wrapped = [Frame(x) for x in rgb]
        library.exp4_flops_begin()
        pred, slots, _ = fuse(wrapped, seq['views'], dis)
        library.exp4_flops_end()
        np.testing.assert_array_equal(slots, seq['fused_source_slots'])
        np.testing.assert_array_equal(pred, data['records']['dis_yolo'][i]['pred'])
    print(f'DIS_CPU_REPLAY_EXACT {len(seqs)}', flush=True)


if __name__ == '__main__':
    main()
