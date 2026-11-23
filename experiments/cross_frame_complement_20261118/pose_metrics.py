"""Use the same pinned single-keypoint OKS protocol as the exp4 core table."""
from functools import lru_cache
import importlib.util

import torch

from evaluate import HERE


@lru_cache(maxsize=1)
def scorer():
    path = HERE.parent / 'concat_core_multiseed_20261111/pose_ap/evaluate_saved.py'
    spec = importlib.util.spec_from_file_location('core_exp4_pose', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    return module.score_pose


def pose_metrics(rows, key):
    records = {idx: {'pred': row['predictions'][key], 'gt': row['gt']}
               for idx, row in enumerate(rows)}
    if not any(len(record['pred']) for record in records.values()):
        return dict(pose_ap50=0., pose_ap75=0., pose_ap50_95=0.)
    ap, _, _ = scorer()(records, torch.linspace(.5, .95, 10))
    return dict(pose_ap50=float(ap[0]), pose_ap75=float(ap[5]), pose_ap50_95=float(ap.mean()))
