"""Fixed protocol and checkpoint validation for the twelve serial ablations."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENTS = HERE.parent
DATA_ROOT = Path(os.environ.get('LCMF_DATA_ROOT', ROOT / 'dataset')).expanduser().resolve()
DATA = DATA_ROOT / 'weed_sliding_merged'
SF = Path(os.environ.get(
    'LCMF_SF_CHECKPOINT',
    EXPERIMENTS / 'single_frame_baseline/results/pad_sf_clean/p2l/best.pt',
)).expanduser().resolve()
SOURCE = ROOT
TRAIN = SOURCE / 'multi_frame_fusion/train_pad_mf.py'
PYTHON = sys.executable
RESULTS = HERE / 'results'


def concat_run(seed):
    if seed == 42:
        return HERE
    return EXPERIMENTS / 'concat_core_multiseed_20261111' / f'seed{seed}'


def avg_run(seed):
    return EXPERIMENTS / 'fusion_avg_multiseed_20261115' / f'seed{seed}'


# This order is also the execution order. Prerequisites run immediately before use.
ARMS = [
    dict(name='01_main', fft=True, gh=True, depth=1, fusion='concat', strategy='staged'),
    dict(name='02_no_global_no_dcn', fft=True, gh=False, depth=0, fusion='concat', strategy='staged'),
    dict(name='03_no_global', fft=True, gh=False, depth=1, fusion='concat', strategy='staged'),
    dict(name='04_no_dcn', fft=True, gh=True, depth=0, fusion='concat', strategy='staged'),
    dict(name='05_no_global_pretrain', fft=True, gh=True, depth=1, fusion='concat', strategy='gh_joint'),
    dict(name='06_no_fft', fft=False, gh=True, depth=1, fusion='concat', strategy='staged'),
    dict(name='07_end_to_end', fft=True, gh=True, depth=1, fusion='concat', strategy='end_to_end'),
    dict(name='08_dcn2', fft=True, gh=True, depth=2, fusion='concat', strategy='staged'),
    dict(name='09_dcn3', fft=True, gh=True, depth=3, fusion='concat', strategy='staged'),
    dict(name='10_fusion_tsa', fft=True, gh=True, depth=1, fusion='tsa', strategy='staged'),
    dict(name='11_fusion_convgru', fft=True, gh=True, depth=1, fusion='convgru', strategy='staged'),
    dict(name='12_fusion_channel', fft=True, gh=True, depth=1, fusion='channel', strategy='staged'),
]

CORE_ARMS = [ARMS[i] for i in (0, 1, 2, 3, 5)]
AVG_ARM = dict(name='13_fusion_avg', fft=True, gh=True, depth=1, fusion='avg', strategy='staged')

DEFAULTS = dict(seed=42, batch_size=8, input_size=1024, num_workers=4,
                stage2_epochs=100, stage3_epochs=250, end_to_end_epochs=700,
                pretrain_stage2_epochs=100, pretrain_stage3_epochs=250,
                stage2_lr=1e-3, stage3_lr=1e-4,
                early_stopping=False, primary_checkpoint='last_raw',
                stage_transition='last_raw', eval_every=1,
                head_pretrain='concat/no-DCN, fresh SF, stage2+stage3; copy G only',
                final_init='fresh common SF; same untrained shared DCN tensors; G only if pretrained',
                global_head_freeze='pretrained G frozen throughout final stages',
                test_use='no test during training; final best/last evaluation only',
                augmentation=True, image_preload=False)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def trainer():
    for p in (SOURCE / 'YOLOv8-pose', SOURCE / 'multi_frame_fusion'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import train_pad_mf
    return train_pad_mf


def architecture_flags(arm):
    flags = ['--base-model', 'p2l', '--fusion-levels', 'p2,p3,p4',
             '--fusion-gate-type', arm['fusion'], '--fusion-mode', 'gate_diff',
             '--fft-downsample', '4', '--offset-reuse-mode', 'bilinear',
             '--convgru-kernel', '3', '--convgru-layers', '1']
    if arm['fft']:
        flags += ['--consistent-align']
    if arm['gh']:
        flags += ['--global-head', '--global-head-level', 'p3', '--global-head-hidden', '32',
                  '--global-head-max-shift', '32', '--global-head-lr-mult', '5']
    if arm['depth'] == 0:
        flags += ['--no-align', '--shared-offset-level', 'none']
    else:
        flags += ['--shared-offset-level', 'p3']
        if arm['depth'] == 1:
            flags += ['--dav2-single']
        elif arm['depth'] == 3:
            flags += ['--dav2-triple']
    return flags


def train_flags(config, data=DATA):
    return ['--seed', str(config['seed']), '--batch-size', str(config['batch_size']),
            '--input-size', str(config['input_size']), '--num-workers', str(config['num_workers']),
            '--dataset-root', str(data), '--no-preload', '--frame-dropout', '0.1',
            '--select-metric', 'point_error', '--min-recall', '0.85',
            '--min-delta-err', '0.005', '--align-loss-weight', '0.5',
            '--align-cos-weight', '1', '--eval-every', str(config['eval_every']),
            '--patience', '0', '--skip-test-eval', '--weights-source', 'model']


def args_for(flags):
    t = trainer()
    saved = sys.argv
    try:
        sys.argv = [str(TRAIN), *flags]
        return t.get_args()
    finally:
        sys.argv = saved


def assert_checkpoint(path, arm, epoch=None):
    import torch
    ck = torch.load(path, map_location='meta', weights_only=False)
    expected = dict(base_model='p2l', fusion_levels=['p2', 'p3', 'p4'],
                    fusion_version=2, fusion_gate_type=arm['fusion'],
                    consistent_align=arm['fft'], fft_downsample=4, global_head=arm['gh'],
                    no_align=arm['depth'] == 0, dav2_single=arm['depth'] == 1,
                    dav2_triple=arm['depth'] == 3,
                    shared_offset_level=None if arm['depth'] == 0 else 'p3')
    for key, val in expected.items():
        assert ck[key] == val, (str(path), key, ck[key], val)
    sd = ck['model']
    assert any(k.startswith('global_head_') for k in sd) == arm['gh']
    assert any('align_ref' in k for k in sd) == (arm['depth'] > 0)
    if arm['depth']:
        for ref in ('align_ref1', 'align_ref2'):
            for depth in (1, 2, 3):
                exists = any(k.startswith(f'fusion.{ref}.p3.s{depth}_dcn.') for k in sd)
                assert exists == (depth <= arm['depth']), (path, ref, depth, exists)
    if epoch is not None:
        assert ck['epoch'] == epoch, (str(path), ck['epoch'], epoch)
    return {k: ck[k] for k in expected} | {'epoch': ck['epoch'], 'stage': ck['stage']}


class EmptyState:
    def state_dict(self):
        return None


def init_model(arm, config, head_source=None, device='cpu', *, template=None,
               reference_init=None):
    """Fresh SF initialization; never import a trained fusion/backbone/DCN."""
    import torch
    t = trainer()
    t.setup_seed(config['seed'])
    args = args_for(architecture_flags(arm))
    model = t.build_model(args, 'cpu', str(SF))
    # All compatible untrained alignment tensors come from a single seeded template.
    # This keeps changes in fusion/depth construction from changing the DCN RNG draw.
    if reference_init is not None:
        reference = torch.load(reference_init, map_location='cpu', weights_only=False)['model']
        keys = model.state_dict().keys()
        removed = {f'fusion.fusion.{level}.proj.{kind}'
                   for level in ('p2', 'p3', 'p4') for kind in ('weight', 'bias')}
        if reference.keys() - keys != removed:
            raise ValueError('Mean fusion must differ from concat only by the projection parameters')
        model.load_state_dict({name: reference[name] for name in keys}, strict=True)
    if arm['depth'] and template is not None and template.exists():
        state = torch.load(template, map_location='cpu', weights_only=False)
        sd = model.state_dict()
        copied = {k: v for k, v in state.items() if k in sd and sd[k].shape == v.shape}
        model.load_state_dict(copied, strict=False)
    if head_source is not None:
        assert arm['gh']
        ck = torch.load(head_source, map_location='cpu', weights_only=False)
        assert ck['consistent_align'] == arm['fft'], 'FFT regime mismatch in G pretraining'
        t._load_global_head_from_ckpt(model, str(head_source))
    return model.to(device)


def save_init(arm, config, path, head_source=None, stage=2, *, template=None,
              reference_init=None):
    t = trainer()
    model = init_model(arm, config, head_source, template=template,
                       reference_init=reference_init)
    empty = EmptyState()
    t.save_ckpt(model, empty, empty, empty, None, 0, -1., str(path), stage=stage,
                best_epoch=0, best_err=float('inf'), best_recall=-1., select_metric='point_error')
    audit = assert_checkpoint(path, arm, epoch=0)
    dump(Path(path).with_suffix('.json'), dict(arm=arm, config=config, metadata=audit,
         checkpoint_sha256=digest(path), sf=str(SF), sf_sha256=digest(SF),
         head_source=str(head_source) if head_source else None,
         head_sha256=digest(head_source) if head_source else None,
         reference_init=str(reference_init) if reference_init else None,
         reference_init_sha256=digest(reference_init) if reference_init else None,
         copied_trained_components=['global_head_r1', 'global_head_r2'] if head_source else []))
    del model


def check_frozen_head(initial, final):
    import torch
    a = torch.load(initial, map_location='cpu', weights_only=False)['model']
    b = torch.load(final, map_location='cpu', weights_only=False)['model']
    keys = [k for k in a if k.startswith('global_head_')]
    assert keys
    delta = max((a[k].float() - b[k].float()).abs().max().item() for k in keys)
    assert delta == 0., (str(final), 'frozen G changed', delta)
    return delta


def penalized_point_error(metrics):
    """Missed GT costs 10px; do not multiply an undefined TP mean by zero."""
    n_gt, n_tp = metrics['n_gt'], metrics['n_tp']
    if n_gt == 0:
        return None
    if n_tp == 0:
        return 10.0
    mean = metrics['nPE_mean_px']
    if not math.isfinite(mean):
        raise ValueError('Nonfinite point error despite nonzero TP count')
    return (mean * n_tp + 10.0 * (n_gt - n_tp)) / n_gt
