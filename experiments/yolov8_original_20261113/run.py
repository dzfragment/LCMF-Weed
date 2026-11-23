"""Official YOLOv8 model/loss/trainer; callbacks add point validation and durable copies."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import yaml

from comparison import (HERE, DATA, COCO_WEIGHTS, SF, SF_LAST, YOLO, atomic_bytes, digest, dump, final_comparison,
                        loader, metrics, should_select, yolo_infer)
import torch


def now():
    return datetime.now(timezone.utc).isoformat()


def status(**fields):
    dump(HERE / 'status.json', dict(updated=now(), pid=os.getpid(), **fields))


def preflight(model):
    for path in (model, SF, SF_LAST):
        if not path.is_file():
            raise FileNotFoundError(path)
    for split in ('train', 'val', 'test'):
        for name in ('images', 'labels'):
            folder = DATA / split / name
            if not folder.is_dir() or not any(folder.iterdir()):
                raise ValueError(f'Expected a populated data directory: {folder}')


def prepare_dataset():
    # Ultralytics writes label caches next to the split directories.
    view = HERE / 'data'
    for split in ('train', 'val', 'test'):
        (view / split).mkdir(parents=True, exist_ok=True)
        for name in ('images', 'labels'):
            link = view / split / name
            target = DATA / split / name
            if not link.is_symlink() and not link.exists():
                link.symlink_to(target, target_is_directory=True)
            elif not link.is_symlink() or link.resolve() != target.resolve():
                raise ValueError(f'Dataset view already points to another location: {link}')
    data_yaml = HERE / 'data.yaml'
    atomic_bytes(data_yaml, yaml.safe_dump(dict(path=str(view), train='train/images',
        val='val/images', test='test/images', names={0: 'weed'},
        kpt_shape=[1, 2], flip_idx=[0]), sort_keys=False).encode())
    return data_yaml


class PointSelection:
    def __init__(self, output=HERE, val_loader=None):
        self.output = Path(output)
        self.batches = val_loader

    def __call__(self, trainer):
        epoch = int(trainer.epoch) + 1
        folder = self.output / 'point_validation'
        folder.mkdir(parents=True, exist_ok=True)
        # EMA evaluation is read-only. Do not fuse/mutate the live training model.
        model = deepcopy(trainer.ema.ema).float().eval()
        if self.batches is None:
            self.batches = loader('val')
        met = metrics(yolo_infer(model, self.batches, conf=.25), ap=False)
        del model
        old_path = self.output / 'selection.json'
        old = json.loads(old_path.read_text()) if old_path.exists() else {}
        selected = old.get('selected')
        is_best = should_select(met, selected)
        if is_best:
            best_path = self.output / 'selected' / f'epoch{epoch:03d}.pt'
            atomic_bytes(best_path, Path(trainer.last).read_bytes())
            selected = met | dict(epoch=epoch, checkpoint=str(best_path), checkpoint_sha256=digest(best_path))
        # Keep a recoverable native checkpoint before Ultralytics strips optimizer at final_eval.
        atomic_bytes(self.output / 'resume.pt', Path(trainer.last).read_bytes())
        record = met | dict(epoch=epoch, selected=is_best, updated=now())
        dump(folder / f'epoch{epoch:03d}.json', record)
        dump(old_path, dict(last_epoch=epoch, selected=selected,
             rule='min mean TP distance with R10>=.85; delta=.005px, tie prefers recall',
             fallback='last if no eligible epoch', resume_sha256=digest(self.output / 'resume.pt')))
        status(state='training', epoch=epoch, epochs=trainer.epochs, validation=met,
               selected_epoch=selected.get('epoch') if selected else None)
        print(f'POINT_VAL E{epoch}: {met}; selected={is_best}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--weights', type=Path, default=COCO_WEIGHTS,
                   help='Original COCO YOLOv8n-pose checkpoint')
    args = p.parse_args()
    lock = (HERE / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight(args.weights)
    if args.preflight_only:
        print('Preflight passed: dataset and checkpoint paths are available')
        return
    if (HERE / 'complete.json').exists():
        done = json.loads((HERE / 'complete.json').read_text())
        assert digest(HERE / 'summary.json') == done['summary_sha256']
        assert digest(HERE / 'profile.json') == done['profile_sha256']
        for r in json.loads((HERE / 'summary.json').read_text()):
            assert digest(r['checkpoint']) == r['checkpoint_sha256']
        print('SKIP completed official YOLOv8 comparison', flush=True)
        return
    status(state='preflight_passed')
    config = json.loads((HERE / 'train_config.json').read_text())
    model_path = args.weights.resolve()
    plan = dict(config=config, data=str(DATA), weights=digest(model_path),
                sf_best=digest(SF), sf_last=digest(SF_LAST))
    plan_path = HERE / 'run_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Comparison inputs changed within an existing run')
    dump(plan_path, plan)
    config.update(data=str(prepare_dataset()), project=str(HERE / 'training'))
    epochs = config['epochs']
    resume = HERE / 'resume.pt'
    native_last = HERE / 'training/seed42/weights/last.pt'
    completed = 0
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        completed = int(ck['epoch']) + 1
        assert 0 < completed <= epochs
        assert ck['optimizer'] is not None
        del ck
    selector = PointSelection()
    try:
        # Reconcile a crash between native save, point selection, and our durable-copy journal.
        newest_path = None
        newest_epoch = 0
        for path in [resume, native_last]:
            if path.exists():
                ck = torch.load(path, map_location='cpu', weights_only=False)
                epoch = int(ck.get('epoch', -1)) + 1
                if epoch > newest_epoch and ck.get('optimizer') is not None:
                    newest_path, newest_epoch = path, epoch
                del ck
        selected_epoch = json.loads((HERE / 'selection.json').read_text()).get('last_epoch', 0) if (HERE / 'selection.json').exists() else 0
        if newest_epoch > selected_epoch:
            from types import SimpleNamespace
            ck = torch.load(newest_path, map_location='cpu', weights_only=False)
            em = ck['ema'].to('cuda:0').float()
            selector(SimpleNamespace(epoch=newest_epoch - 1, epochs=epochs,
                ema=SimpleNamespace(ema=em), last=newest_path))
            completed = newest_epoch
            del em, ck
        if completed < epochs:
            model = YOLO(str(resume if completed else model_path))
            model.add_callback('on_model_save', selector)
            if completed:
                model.train(resume=str(resume))
            else:
                model.train(**config)
            del model
            torch.cuda.empty_cache()
        selection = json.loads((HERE / 'selection.json').read_text())
        assert selection['last_epoch'] == epochs
        assert digest(resume) == selection['resume_sha256']
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        assert int(ck['epoch']) + 1 == epochs
        del ck
        best = Path(selection['selected']['checkpoint']) if selection['selected'] else resume
        if selection['selected']:
            assert digest(best) == selection['selected']['checkpoint_sha256']
        dump(HERE / 'training_complete.json', dict(completed=now(), epochs=epochs, seed=42,
             best=str(best), fallback_used=not bool(selection['selected']), last=str(resume)))
        status(state='evaluating', epochs=epochs, best=str(best))
        records = final_comparison(best, resume)
        dump(HERE / 'complete.json', dict(completed=now(), epochs=epochs, seed=42,
            n_evaluations=len(records), summary_sha256=digest(HERE / 'summary.json'),
            profile_sha256=digest(HERE / 'profile.json')))
        status(state='complete', epochs=epochs, n_evaluations=len(records))
        print('OFFICIAL YOLOV8 COMPARISON COMPLETE', flush=True)
    except Exception as exc:
        status(state='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
