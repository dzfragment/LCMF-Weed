#!/usr/bin/env python3
"""Durable serial queue: Exp4 ablations and required G pretraining."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from common import (ARMS, AVG_ARM, CORE_ARMS, DATA, DEFAULTS, HERE, PYTHON, SF,
                    SOURCE, TRAIN, architecture_flags, assert_checkpoint,
                    avg_run, check_frozen_head, concat_run, digest, dump,
                    init_model, save_init, train_flags)

ACTIVE = None
SCRIPT_DIR = HERE


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def stop(signum, frame):
    old = json.loads((HERE / 'status.json').read_text()) if (HERE / 'status.json').exists() else {}
    dump(HERE / 'status.json', old | dict(state='interrupted', signal=signum, updated=now(), pid=os.getpid()))
    if ACTIVE is not None and ACTIVE.poll() is None:
        try:
            os.killpg(ACTIVE.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            ACTIVE.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(ACTIVE.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    raise SystemExit(128 + signum)


def run_process(cmd, logfile, state):
    global ACTIVE
    if shutil.disk_usage(HERE).free < 3 * 1024**3:
        raise RuntimeError('Less than 3 GiB free; queue stopped before launching next process')
    dump(HERE / 'status.json', state | dict(state='running', updated=now(), pid=os.getpid(),
                                         log=str(logfile), completed_arms=completed_names()))
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MF_TRAIN_AUGMENT='1', TQDM_MININTERVAL='30')
    with Path(logfile).open('a') as f:
        ACTIVE = subprocess.Popen(cmd, cwd=SOURCE / 'multi_frame_fusion', env=env,
                                  stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        dump(Path(logfile).with_suffix('.process.json'), dict(pid=ACTIVE.pid, command=cmd, started=now()))
        rc = ACTIVE.wait()
    ACTIVE = None
    if rc:
        raise RuntimeError(f'Exit {rc}: {logfile}')


def completed_names():
    return [a['name'] for a in RUN_ARMS if (RESULTS / a['name'] / 'complete.json').exists()]


def stage(arm, config, folder, number, epochs, initial, frozen=False, prerequisite=False):
    folder.mkdir(parents=True, exist_ok=True)
    last = folder / f'pad_stage{number}_last.pt'
    marker = folder / f'stage{number}_complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['sha256'] == digest(last), f'Completed checkpoint changed: {last}'
        assert_checkpoint(last, arm, epochs)
        if frozen:
            check_frozen_head(initial, last)
        log(f'SKIP complete {folder.name} stage{number}')
        return last
    cmd = [PYTHON, str(TRAIN), *architecture_flags(arm), *train_flags(config),
           '--stage', str(number), '--epochs', str(epochs),
           '--lr', str(config['stage2_lr'] if number == 2 else config['stage3_lr']),
           '--output-dir', str(folder)]
    if frozen:
        cmd += ['--freeze-global-head']
    if number == 3:
        cmd += ['--unfreeze-head']
    if last.exists():
        metadata = assert_checkpoint(last, arm)
        assert metadata['epoch'] <= epochs
        cmd += ['--resume', str(last)]
    else:
        cmd += ['--weights', str(initial)]
    dump(folder / f'stage{number}_command.json', cmd)
    log(f'START {folder.name} stage{number}, {epochs} epochs, frozen_G={frozen}')
    run_process(cmd, folder / f'stage{number}.log',
                dict(arm=arm['name'], phase=f'stage{number}', epochs=epochs,
                     prerequisite=prerequisite, output=str(folder)))
    audit = assert_checkpoint(last, arm, epochs)
    result = json.loads((folder / f'stage{number}_results.json').read_text())
    assert result['epochs_completed'] == epochs
    delta = check_frozen_head(initial, last) if frozen else None
    dump(marker, dict(completed=now(), metadata=audit, checkpoint=str(last),
                      sha256=digest(last), frozen_G_max_change=delta))
    log(f'DONE {folder.name} stage{number}')
    return last


def head_pretrain(fft, config):
    name = 'global_fft' if fft else 'global_nofft'
    folder = PRETRAIN / name
    folder.mkdir(parents=True, exist_ok=True)
    arm = dict(name=name, fft=fft, gh=True, depth=0, fusion='concat', strategy='gh_joint')
    initial = folder / 'init.pt'
    if not initial.exists():
        save_init(arm, config, initial)
    ck2 = stage(arm, config, folder, 2, config['pretrain_stage2_epochs'], initial,
                prerequisite=True)
    ck3 = stage(arm, config, folder, 3, config['pretrain_stage3_epochs'], ck2,
                prerequisite=True)
    return ck3


def evaluate(arm, folder, config):
    for split in ('val', 'test'):
        for tag in ('last', 'best'):
            marker = folder / f'eval_{split}_{tag}.json'
            if marker.exists():
                evaluation = json.loads(marker.read_text())
                ck = folder / f'pad_stage3_{tag}.pt'
                if (evaluation.get('checkpoint_sha256') == digest(ck)
                        and evaluation.get('data_root') == str(DATA.resolve())):
                    continue
            cmd = [PYTHON, str(SCRIPT_DIR / 'evaluate.py'), '--arm', arm['name'],
                   '--split', split, '--tag', tag, '--batch-size', str(config['batch_size']),
                   '--run-dir', str(folder)]
            dump(folder / f'eval_{split}_{tag}_command.json', cmd)
            run_process(cmd, folder / f'eval_{split}_{tag}.log',
                        dict(arm=arm['name'], phase=f'eval_{split}_{tag}', prerequisite=False))
            assert marker.exists()


def summarize():
    records = []
    for arm in RUN_ARMS:
        folder = RESULTS / arm['name']
        for split in ('val', 'test'):
            for tag in ('last', 'best'):
                p = folder / f'eval_{split}_{tag}.json'
                if p.exists():
                    records.append(json.loads(p.read_text()))
    dump(HERE / 'summary.json', records)
    keys = ['arm', 'split', 'tag', 'point_recall', 'nPE_mean_px', 'nPE_median_px',
            'point_precision', 'penalized_error_10px', 'point_ap_10px', 'n_gt', 'n_tp']
    tmp = HERE / 'summary.csv.tmp'
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)
    os.replace(tmp, HERE / 'summary.csv')


def main():
    global HERE, PRETRAIN, RESULTS, RUN_ARMS
    p = argparse.ArgumentParser(description=__doc__,
        epilog='Set LCMF_DATA_ROOT to the directory containing weed_sliding_merged. '
               'Set LCMF_SF_CHECKPOINT to the clean PAD p2l best checkpoint.')
    p.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    p.add_argument('--suite', choices=('concat', 'avg'), default='concat')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--reference-run', type=Path, help='Same-seed concat run for the avg suite')
    p.add_argument('--write-plan', action='store_true')
    args = p.parse_args()
    if args.reference_run and args.suite != 'avg':
        p.error('--reference-run requires --suite avg')

    is_avg = args.suite == 'avg'
    default_run = avg_run(args.seed) if is_avg else concat_run(args.seed)
    HERE = (args.output_dir or default_run).resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    PRETRAIN, RESULTS = HERE / 'pretrain', HERE / 'results'
    RUN_ARMS = [AVG_ARM] if is_avg else ARMS if args.seed == 42 else CORE_ARMS
    lock = (HERE / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = DEFAULTS | {'seed': args.seed}

    if not SF.is_file():
        p.error(f'Supply a PAD p2l checkpoint through LCMF_SF_CHECKPOINT: {SF}')
    if any(not (DATA / split).is_dir() for split in ('train', 'val', 'test')):
        p.error(f'Expected train, val and test under {DATA}; set LCMF_DATA_ROOT')
    dependencies = {}
    if is_avg:
        reference = (args.reference_run or concat_run(args.seed)).resolve()
        reference_plan = json.loads((reference / 'plan.json').read_text())
        if reference_plan['config'] != config or reference_plan['sf_sha256'] != digest(SF):
            raise ValueError('The avg suite requires a matched concat run with the same seed and SF checkpoint')
        dependencies = {
            'head': reference / 'pretrain/global_fft/pad_stage3_last.pt',
            'init': reference / 'results/01_main/init.pt',
        }
        assert_checkpoint(dependencies['head'], ARMS[3], config['pretrain_stage3_epochs'])
        assert_checkpoint(dependencies['init'], ARMS[0], 0)

    source_files = [p for name in ('YOLOv8-pose', 'multi_frame_fusion')
                    for p in (SOURCE / name).rglob('*.py')]
    source_files.extend(SCRIPT_DIR.glob('*.py'))
    plan = dict(arms=RUN_ARMS, config=config, data=str(DATA), sf=str(SF), sf_sha256=digest(SF),
                dependencies={name: {'path': str(path), 'sha256': digest(path)}
                              for name, path in dependencies.items()},
                source_sha256={str(path.relative_to(SOURCE)): digest(path)
                               for path in sorted(source_files)})
    plan_path = HERE / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Run inputs or source changed. Choose a new --output-dir.')
    dump(HERE / 'config.json', config)
    dump(plan_path, plan)
    if args.write_plan:
        return
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        dump(HERE / 'status.json', dict(state='checking', phase='preflight', pid=os.getpid(),
                                       updated=now(), completed_arms=completed_names()))
        import torch
        assert torch.cuda.is_available(), 'CUDA is unavailable'
        log(f'QUEUE START, pid={os.getpid()}, GPU={torch.cuda.get_device_name(0)}')
        template = HERE / 'common_dcn_init.pt'
        if not is_avg and not template.exists():
            model = init_model(ARMS[0], config)
            temp = template.with_suffix('.tmp')
            torch.save({k: v for k, v in model.state_dict().items() if 'align_ref' in k}, temp)
            temp.replace(template)
            del model
        for arm in RUN_ARMS:
            folder = RESULTS / arm['name']
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'complete.json').exists():
                completed = json.loads((folder / 'complete.json').read_text())
                assert completed['arm'] == arm
                final = folder / 'pad_stage3_last.pt'
                assert completed['sha256'] == digest(final), f'Completed checkpoint changed: {final}'
                expected_epochs = config['end_to_end_epochs'] if arm['strategy'] == 'end_to_end' else config['stage3_epochs']
                assert_checkpoint(final, arm, expected_epochs)
                # Also validates checkpoint hashes on cached evaluations.
                evaluate(arm, folder, config)
                summarize()
                log(f'SKIP completed arm {arm["name"]}')
                continue
            dump(folder / 'arm.json', arm)
            frozen = arm['gh'] and arm['strategy'] == 'staged'
            head = dependencies.get('head')
            if frozen and head is None:
                head = head_pretrain(arm['fft'], config)
            initial = folder / 'init.pt'
            first_stage = 3 if arm['strategy'] == 'end_to_end' else 2
            if not initial.exists():
                save_init(arm, config, initial, head, stage=first_stage,
                          template=template if not is_avg else None,
                          reference_init=dependencies.get('init'))
            if first_stage == 2:
                ck2 = stage(arm, config, folder, 2, config['stage2_epochs'], initial, frozen)
                final = stage(arm, config, folder, 3, config['stage3_epochs'], ck2, frozen)
            else:
                final = stage(arm, config, folder, 3, config['end_to_end_epochs'], initial, frozen=False)
            evaluate(arm, folder, config)
            dump(folder / 'complete.json', dict(arm=arm, completed=now(), checkpoint=str(final),
                                               sha256=digest(final)))
            summarize()
            dump(HERE / 'status.json', dict(state='arm_complete', arm=arm['name'], updated=now(),
                                          pid=os.getpid(), completed_arms=completed_names()))
            log(f'ARM DONE {arm["name"]}; {len(completed_names())}/{len(RUN_ARMS)}')
        dump(HERE / 'status.json', dict(state='complete', updated=now(), completed_arms=completed_names()))
        log(f'ALL {len(RUN_ARMS)} ARMS COMPLETE')
    except Exception as exc:
        old = json.loads((HERE / 'status.json').read_text()) if (HERE / 'status.json').exists() else {}
        dump(HERE / 'status.json', old | dict(state='failed', error=str(exc), updated=now(), pid=os.getpid()))
        log(f'QUEUE FAILED: {exc}')
        raise


if __name__ == '__main__':
    main()
