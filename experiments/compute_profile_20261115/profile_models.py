"""Uniform B1/1024 full-forward MAC profiling, with explicit DCN convolution hooks.

Never modifies completed experiment sources, checkpoints or training configurations.
"""
from pathlib import Path
import os
import sys
import json
import csv
import gc
import hashlib
from copy import deepcopy
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
RUN = EXPERIMENTS / 'concat_ablation12_20261106'
OFFICIAL = EXPERIMENTS / 'yolov8_original_20261113'
os.environ['YOLO_CONFIG_DIR'] = str(HERE / 'settings')
(HERE / 'settings').mkdir(exist_ok=True)
os.environ['ULTRALYTICS_OFFLINE'] = 'true'
sys.path.insert(0, str(RUN))
import common
t = common.trainer()
from pad_core import PADSingleFrame
from models.modules import DeformConvPackV2
import torch
import thop
from torch.utils.flop_counter import FlopCounterMode, conv_flop_count
from ultralytics import YOLO


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def count_dcn(module, args, output):
    # B*Hout*Wout*Cout*Cin*Kh*Kw; deformable bilinear sampling is separate.
    macs = int(output.numel()) * int(module.weight.shape[1]) * int(module.weight.shape[2]) * int(module.weight.shape[3])
    module.total_ops += macs


def dcn_formula(*args, out_shape=None, **kwargs):
    return conv_flop_count(args[0], args[1], out_shape)


@torch.inference_mode()
def measure(model, x, name, checkpoint, frames):
    model = model.float().eval().to('cuda:0')
    parameters = sum(p.numel() for p in model.parameters())
    # Register call hooks so repeated shared modules are counted on every actual invocation.
    component_params = {}
    for key in ['backbone', 'net', 'fpn', 'head', 'fusion', 'global_head_r1', 'global_head_r2']:
        module = getattr(model, key, None)
        if module is not None:
            component_params[key] = sum(p.numel() for p in module.parameters())
    dcn_calls = []
    def capture(module, args, output):
        dcn_calls.append(dict(input=list(args[0].shape), weight=list(module.weight.shape),
            output=list(output.shape), macs=int(output.numel()) * module.weight[0].numel()))
    handles = [m.register_forward_hook(capture) for m in model.modules() if isinstance(m, DeformConvPackV2)]
    macs, thop_params = thop.profile(model, inputs=(x,),
        custom_ops={DeformConvPackV2: count_dcn}, verbose=False)
    for h in handles:
        h.remove()
    dcn_macs = sum(c['macs'] for c in dcn_calls)
    # Independent dispatcher-based check includes functional Sobel convolutions,
    # but normally omits BN, pooling etc. DCN's convolution receives its own formula.
    mappings = {torch.ops.torchvision.deform_conv2d: dcn_formula}
    with FlopCounterMode(display=False, custom_mapping=mappings) as counter:
        output = model(x)
    operator_flops = counter.get_total_flops()
    ops = {str(k): int(v) for k,v in counter.flop_counts['Global'].items()}
    dcn_operator_flops = sum(v for k,v in ops.items() if 'deform_conv2d' in k)
    assert dcn_operator_flops == 2 * dcn_macs, (name,dcn_operator_flops,2*dcn_macs)
    # Hook and dispatcher methods must agree on convolutional scale.
    assert abs(operator_flops - 2*macs) / max(2*macs,1) < .03, (name,operator_flops,2*macs)
    row = dict(name=name,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        frames=frames,input=list(x.shape),parameters=int(parameters),parameters_m=parameters/1e6,
        macs=float(macs),gmacs=macs/1e9,gflops=2*macs/1e9,
        dcn_convolution_gflops=2*dcn_macs/1e9,
        thop_without_custom_dcn_gflops=2*(macs-dcn_macs)/1e9,
        dispatcher_core_gflops=operator_flops/1e9,dispatcher_operators=ops,
        dcn_calls=dcn_calls,component_parameters=component_params,
        output_shapes=[list(v.shape) for v in output if isinstance(v,torch.Tensor)] if isinstance(output,tuple) else str(type(output)))
    write_json(HERE / (name+'.json'),row)
    print(f"{name}: {row['gflops']:.6f} GFLOPs; {parameters:,} params; DCN conv {row['dcn_convolution_gflops']:.6f} G",flush=True)
    del output
    return row


def main():
    torch.set_num_threads(4)
    ds=t.MultiFrameDataset(str(common.DATA),split='test',input_size=1024,augment=False,params=None)
    sample=ds[0][0].unsqueeze(0).to('cuda:0').float()/255
    assert tuple(sample.shape)==(1,3,3,1024,1024)
    x=sample[:,0]
    rows=[]
    checkpoint=common.SF
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=PADSingleFrame(ck['base_model'],ck['head_cfg'])
    state = ck['model']
    model.load_state_dict(state.float().state_dict() if isinstance(state, torch.nn.Module) else state,
                          strict=True)
    del ck
    rows.append(measure(model,x,'sf_p2l_pad',checkpoint,1))
    del model
    for name,checkpoint in [('yolov8n_pose_weed_1kp',OFFICIAL/'training/seed42/weights/best.pt'),
                            ('yolov8n_pose_original_coco17',Path(os.environ.get(
                                'LCMF_YOLO_COCO_WEIGHTS', OFFICIAL/'initial/yolov8n-pose.pt')))]:
        model=YOLO(str(checkpoint)).model
        rows.append(measure(model,x,name,checkpoint,1))
        if name=='yolov8n_pose_original_coco17':
            small=torch.nn.functional.interpolate(x,size=(640,640),mode='bilinear',align_corners=False)
            rows.append(measure(model,small,name+'_640',checkpoint,1))
            del small
        del model
        gc.collect();torch.cuda.empty_cache()
    for arm in common.ARMS:
        checkpoint=common.RESULTS/arm['name']/'pad_stage3_last.pt'
        common.assert_checkpoint(checkpoint,arm)
        args=common.args_for(common.architecture_flags(arm))
        model,_=t.load_ckpt(str(checkpoint),args,'cuda:0',prefer_ema=False)
        rows.append(measure(model,sample,arm['name'],checkpoint,3))
        del model
        gc.collect();torch.cuda.empty_cache()
    by_name={r['name']:r for r in rows}
    for name in ['05_no_global_pretrain','07_end_to_end']:
        assert by_name[name]['gflops']==by_name['01_main']['gflops'],name
    assert by_name['02_no_global_no_dcn']['dcn_convolution_gflops']==0
    assert by_name['04_no_dcn']['dcn_convolution_gflops']==0
    assert by_name['09_dcn3']['gflops']>by_name['08_dcn2']['gflops']>by_name['01_main']['gflops']
    protocol=dict(created=datetime.now(timezone.utc).isoformat(),batch=1,main_input_hw=[1024,1024],
        precision='fp32',fused=False,inference='full forward, no streaming feature cache',
        gflops_definition='2 * THOP MACs, with explicit DeformConvPackV2 convolution hook',
        package_versions=dict(torch=torch.__version__,thop=thop.__version__),
        included='standard THOP operations (Conv/Linear/BN/pooling etc) plus all DCN weighted convolution MACs',
        excluded='FFT complex arithmetic, DCN interpolation/modulation, grid_sample/interpolate functional arithmetic, most activations/elementwise ops, data IO, NMS, training/backward',
        cross_check='PyTorch dispatcher Conv/Linear/MatMul including functional Sobel plus custom DCN formula; may omit BN/pooling',
        representative_sequence=str(ds.seq_dirs[0]),
        seed_note='Seed does not change architecture compute; repeated seeds are represented once')
    write_json(HERE/'summary.json',dict(protocol=protocol,models=rows))
    with (HERE/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['name','frames','parameters','gmacs','gflops','dcn_convolution_gflops','dispatcher_core_gflops','checkpoint'],extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    lines=['# 模型计算量：统一1024输入','',
        'Batch1；单帧1张输入，多帧3张输入；完整forward，不复用历史帧特征；未融合BN。GFLOPs=2×MACs。',
        '已补计通用THOP会遗漏的DCN卷积。下表是常规神经网络MAC口径，**不含FFT、DCN插值/调制、grid_sample及多数逐元素算子**，不能视为端到端全部算术开销。',
        'FFT与采样不是零开销；本表未将其估算成精确FLOPs。序列/种子不改变本表主要卷积结构；不同训练策略但同结构的模型计算量相同。','',
        '| 模型 | 帧数 | 输入 | 参数(M) | GMACs | GFLOPs | 其中DCN卷积GFLOPs |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['frames']} | {r['input'][-1]}×{r['input'][-1]} | {r['parameters_m']:.6f} | {r['gmacs']:.6f} | {r['gflops']:.6f} | {r['dcn_convolution_gflops']:.6f} |")
    lines+=['','原始COCO模型保留17个人体关键点；weed模型为1类、1个二维茎基点。640行仅用于与常见官方规格对照，其余统一1024。',
            '独立PyTorch算子级统计及全部DCN调用形状保存在逐模型JSON，可检查漏算或重复计算。原始模型与实验checkpoint的SHA保存在summary.json；没有修改训练权重。']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('ALL MODELS PROFILED',flush=True)


if __name__=='__main__':
    main()
