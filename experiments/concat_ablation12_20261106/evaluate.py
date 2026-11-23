"""Raw best/last checkpoint metrics; no test-based model selection."""
import argparse
import gc
import pickle
from pathlib import Path

from common import (ARMS, AVG_ARM, DATA, RESULTS, architecture_flags, args_for,
                    assert_checkpoint, digest, dump, penalized_point_error, trainer)


def main():
    p = argparse.ArgumentParser()
    arms = {arm['name']: arm for arm in [*ARMS, AVG_ARM]}
    p.add_argument('--arm', choices=arms, required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--tag', choices=['last', 'best'], required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--data-root', type=Path, default=DATA)
    p.add_argument('--run-dir', type=Path, default=None)
    a = p.parse_args()
    import torch
    arm = arms[a.arm]
    folder = (a.run_dir or RESULTS / a.arm).resolve()
    ck = folder / f'pad_stage3_{a.tag}.pt'
    metadata = assert_checkpoint(ck, arm)
    t = trainer()
    args = args_for(architecture_flags(arm))
    model, _ = t.load_ckpt(str(ck), args, 'cuda', prefer_ema=False)
    model.eval()
    ds = t.MultiFrameDataset(str(a.data_root), split=a.split, input_size=1024, augment=False, params=None)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=t.MultiFrameDataset.collate_fn)
    with torch.no_grad():
        met, preds = t.evaluate_points(model, loader, 1024, conf=.25, topk=300, tau_px=10.)
    record = dict(met) | dict(arm=a.arm, split=a.split, tag=a.tag, ckpt=str(ck), metadata=metadata,
                             n_sequences=len(ds), conf=.25, radius_px=10, weights='model_raw',
                             checkpoint_sha256=digest(ck), data_root=str(a.data_root.resolve()))
    record['penalized_error_10px'] = penalized_point_error(met)
    rows = [dict(sequence=Path(ds.seq_dirs[int(seq)]).name, sequence_index=int(seq),
                 **t.match_points({seq: d}, tau_px=10., conf=.25))
            for seq, d in preds.items()]
    with (folder / f'preds_{a.split}_{a.tag}.pkl').open('wb') as f:
        pickle.dump(preds, f)
    dump(folder / f'per_seq_{a.split}_{a.tag}.json', rows)
    dump(folder / f'eval_{a.split}_{a.tag}.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
