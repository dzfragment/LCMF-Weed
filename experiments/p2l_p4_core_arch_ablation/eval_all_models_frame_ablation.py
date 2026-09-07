#!/usr/bin/env python3
"""
所有 7 个消融模型 × 7 种帧模式 × <DATASET_ROOT>/test 的帧消融评估。
"""
import os, sys, csv
from collections import OrderedDict
import numpy as np
import torch
from torch.utils import data

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP1_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # 仓库根
sys.path.insert(0, os.path.join(_EXP1_DIR, 'YOLOv8-pose'))
sys.path.insert(0, os.path.join(_EXP1_DIR, 'multi_frame_fusion'))

from models.mf_yolo import MultiFrameYOLO
from train import apply_ablation, apply_dav2_single
from models.fft_align import apply_consistent_image_align
from utils import util

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED = os.path.join(_EXP1_DIR, 'experiments', 'single_frame_baseline', 'results',
                          'yolo_v8_n_p2_lite_p4', 'seed42', 'best.pt')
RESULTS_ROOT = os.environ.get('RESULTS_ROOT') or os.path.join(_SCRIPT_DIR, 'results')
DS_TEST = os.path.join(os.environ.get('DATASET_ROOT') or os.path.join(_EXP1_DIR, 'weed_sliding_merged'), 'test')
INPUT_SIZE = 1024

MODELS = OrderedDict({
    '00_full_fft_dcn_spatial':   {'label': 'FFT + DCN + Spatial',    'gs': 'feature', 'ds':4, 'gt':'spatial', 'fm':'gate_diff', 'na':False, 'ds2':False, 'rf':False},
    '01_remove_fft':             {'label': 'w/o FFT',                 'gs': 'none',    'ds':4, 'gt':'spatial', 'fm':'gate_diff', 'na':False, 'ds2':False, 'rf':False},
    '02_remove_dcn':             {'label': 'w/o DCN',                 'gs': 'feature', 'ds':4, 'gt':'spatial', 'fm':'gate_diff', 'na':True,  'ds2':False, 'rf':False},
    '03_remove_spatial':         {'label': 'w/o Spatial (avg)',       'gs': 'feature', 'ds':4, 'gt':'avg',     'fm':'avg',       'na':False, 'ds2':False, 'rf':False},
    '05_ablate_concat_fusion':   {'label': 'FFT + DCN + Concat',      'gs': 'feature', 'ds':4, 'gt':'concat',  'fm':'gate_diff', 'na':False, 'ds2':False, 'rf':False},
    '06_ablate_dav2_single':     {'label': 'FFT + Single DAV2',       'gs': 'feature', 'ds':4, 'gt':'spatial', 'fm':'gate_diff', 'na':False, 'ds2':True,  'rf':False},
})

FRAME_MODES = OrderedDict({
    '1f: key3':          lambda f: torch.stack([f[:,0], f[:,0], f[:,0]], dim=1),
    '2f: key+ref1':      lambda f: torch.stack([f[:,0], f[:,1], f[:,1]], dim=1),
    '2f: key+ref2':      lambda f: torch.stack([f[:,0], f[:,2], f[:,2]], dim=1),
    '3f: key+ref1+ref2': lambda f: f,
    '1f: ref13':         lambda f: torch.stack([f[:,1], f[:,1], f[:,1]], dim=1),
    '1f: ref23':         lambda f: torch.stack([f[:,2], f[:,2], f[:,2]], dim=1),
    '2f: ref1+ref2':     lambda f: torch.stack([f[:,1], f[:,2], f[:,2]], dim=1),
})

class EvalDataset(data.Dataset):
    def __init__(self):
        self.dirs = sorted([os.path.join(DS_TEST,d) for d in os.listdir(DS_TEST)
                           if os.path.isdir(os.path.join(DS_TEST,d))])
    def __len__(self): return len(self.dirs)
    def __getitem__(self, i):
        import cv2
        imgs=[]
        for j in range(3):
            img=cv2.imread(os.path.join(self.dirs[i],f'frame_{j}.jpg'))
            if img is None: raise FileNotFoundError(f'{self.dirs[i]}/frame_{j}.jpg')
            imgs.append(img)
        h,w=imgs[0].shape[:2]; r=min(INPUT_SIZE/h,INPUT_SIZE/w,1.0)
        pw,ph=int(round(w*r)),int(round(h*r)); dw,dh=(INPUT_SIZE-pw)/2,(INPUT_SIZE-ph)/2
        samples=[]
        for img in imgs:
            if (w,h)!=(pw,ph): img=cv2.resize(img,(pw,ph),interpolation=cv2.INTER_LINEAR)
            t=int(round(dh-0.1)); b=int(round(dh+0.1)); l=int(round(dw-0.1)); r=int(round(dw+0.1))
            img=cv2.copyMakeBorder(img,t,b,l,r,cv2.BORDER_CONSTANT,value=(0,0,0))
            samples.append(torch.from_numpy(np.ascontiguousarray(img.transpose((2,0,1))[::-1])))
        fn=torch.stack(samples,dim=0)
        lp=os.path.join(self.dirs[i],'labels_frame0.txt')
        if os.path.exists(lp):
            with open(lp) as f: lines=[x.strip() for x in f.readlines() if x.strip()]
            if lines:
                label=np.array([list(map(float,l.split())) for l in lines],dtype=np.float32)
                box,kpt=label[:,:5],label[:,5:]
            else: box=np.zeros((0,5),dtype=np.float32); kpt=np.zeros((0,0),dtype=np.float32)
        else: box=np.zeros((0,5),dtype=np.float32); kpt=np.zeros((0,0),dtype=np.float32)
        if kpt.size>0: kpt=np.concatenate([kpt,np.full((kpt.shape[0],1),2.0,dtype=np.float32)],axis=1).reshape(-1,1,3)
        else: kpt=np.zeros((0,1,3),dtype=np.float32)
        nl=len(box)
        if nl:
            box[:,1:]=_wh2xy(box[:,1:],pw,ph,dw,dh)
            kpt[...,0]=kpt[...,0]*pw+dw; kpt[...,1]=kpt[...,1]*ph+dh
            box[:,1:]=_xy2wh(box[:,1:],INPUT_SIZE,INPUT_SIZE)
            kpt[...,0]/=INPUT_SIZE; kpt[...,1]/=INPUT_SIZE
        cls=torch.from_numpy(box[:,0:1]).float() if nl else torch.zeros((0,1))
        tb=torch.from_numpy(box[:,1:5]).float() if nl else torch.zeros((0,4))
        tk=torch.from_numpy(kpt).float() if nl else torch.zeros((0,1,3))
        return fn,cls,tb,tk,torch.zeros(nl,dtype=torch.long)
    @staticmethod
    def collate_fn(batch):
        frames,cls,box,kpt,indices=zip(*batch)
        frames=torch.stack(frames,0); cls=torch.cat(cls,0); box=torch.cat(box,0); kpt=torch.cat(kpt,0)
        ni=list(indices)
        for i in range(len(ni)): ni[i]+=i
        return frames,{'cls':cls,'box':box,'kpt':kpt,'idx':torch.cat(ni,0)}

def _wh2xy(x,w,h,dw,dh):
    y=np.copy(x); y[:,0]=w*(x[:,0]-x[:,2]/2)+dw; y[:,1]=h*(x[:,1]-x[:,3]/2)+dh
    y[:,2]=w*(x[:,0]+x[:,2]/2)+dw; y[:,3]=h*(x[:,1]+x[:,3]/2)+dh; return y
def _xy2wh(x,w,h):
    x[:,[0,2]]=x[:,[0,2]].clip(0,w-1e-3); x[:,[1,3]]=x[:,[1,3]].clip(0,h-1e-3)
    y=np.copy(x); y[:,0]=((x[:,0]+x[:,2])/2)/w; y[:,1]=((x[:,1]+x[:,3])/2)/h
    y[:,2]=(x[:,2]-x[:,0])/w; y[:,3]=(x[:,3]-x[:,1])/h; return y

@torch.no_grad()
def evaluate_mode(model, dataloader, mode_fn):
    iou_v=torch.linspace(0.5,0.95,10,device=DEVICE); ni=iou_v.numel(); ks=model.head.kpt_shape
    bm,km=[],[]
    for frames_native,targets in dataloader:
        frames=mode_fn(frames_native.to(DEVICE).float()/255.0)
        with torch.amp.autocast('cuda',dtype=torch.float16): outputs=model(frames)
        if isinstance(outputs,tuple): outputs=outputs[0]
        if isinstance(outputs,list): outputs=torch.cat([o.flatten(2) for o in outputs],dim=2)
        outputs=util.non_max_suppression(outputs.float(),0.001,0.7,model.head.nc)
        _,_,_,h,w=frames.shape; scale=torch.tensor((w,h,w,h),device=DEVICE)
        for i,output in enumerate(outputs):
            idx=targets['idx']==i; cls=targets['cls'][idx].to(DEVICE)
            tb=targets['box'][idx].to(DEVICE); tk=targets['kpt'][idx].to(DEVICE)
            cb=torch.zeros(output.shape[0],ni,dtype=torch.bool,device=DEVICE)
            ck=torch.zeros(output.shape[0],ni,dtype=torch.bool,device=DEVICE)
            if output.shape[0]==0:
                if cls.shape[0]:
                    bm.append((cb,*torch.zeros((2,0),device=DEVICE),cls.squeeze(-1)))
                    km.append((ck,*torch.zeros((2,0),device=DEVICE),cls.squeeze(-1)))
                continue
            pred=output.clone(); pk=pred[:,6:].view(output.shape[0],ks[0],-1)
            if cls.shape[0]:
                target=torch.cat((cls,util.wh2xy(tb)*scale),1)
                tkp=tk.clone(); tkp[...,0]*=w; tkp[...,1]*=h
                cb=util.compute_metric(pred[:,:6],target,iou_v)
                ck=util.compute_metric(pred[:,:6],target,iou_v,pk,tkp)
            bm.append((cb,output[:,4],output[:,5],cls.squeeze(-1)))
            km.append((ck,output[:,4],output[:,5],cls.squeeze(-1)))
    def ap(ml):
        arr=[torch.cat(x,0).cpu().numpy() for x in zip(*ml)]
        if len(arr) and arr[0].any(): *_,a=util.compute_ap(*arr); return float(a)
        return 0.0
    return ap(bm),ap(km)

def build_model(cfg):
    model=MultiFrameYOLO(num_classes=1,kpt_shape=(1,2),fusion_levels=['p4'],
        pretrained_path=None,fusion_version=2,
        fusion_gate_type=cfg['gt'], global_align_stage=cfg['gs'],
        fft_downsample=cfg['ds']).to(DEVICE)
    apply_ablation(model,no_align=cfg['na'],fusion_mode=cfg['fm'])
    if cfg['ds2']: apply_dav2_single(model)
    if cfg['rf']: apply_consistent_image_align(model,downsample=cfg['ds'])
    ckpt=torch.load(os.path.join(RESULTS_ROOT,cfg['exp_id'],'mf_stage3_best.pt'),
                    map_location=DEVICE,weights_only=False)
    sd=ckpt.get('model')
    if isinstance(sd,dict): model.load_state_dict(sd,strict=False)
    if ckpt.get('consistent_align', False) and getattr(model, 'global_align_stage', '') == 'feature':
        from models.mf_yolo import enable_consistent_align
        enable_consistent_align(model)
    else: model=sd.float().to(DEVICE)
    model.eval(); return model

if __name__=='__main__':
    dataset=EvalDataset()
    loader=data.DataLoader(dataset,batch_size=4,shuffle=False,
                           num_workers=4,pin_memory=True,collate_fn=EvalDataset.collate_fn)
    print(f'Test set: {len(dataset)} sequences')
    mode_names=list(FRAME_MODES.keys())
    results=OrderedDict()

    for exp_id, cfg in MODELS.items():
        cfg['exp_id']=exp_id
        label=cfg['label']
        print(f'\n{"="*60}\n{label}\n{"="*60}')
        model=build_model(cfg)
        mr=OrderedDict()
        for mn, mf in FRAME_MODES.items():
            ba,pa=evaluate_mode(model,loader,mf)
            mr[mn]=(round(ba,4),round(pa,4))
        results[label]=mr
        p1=mr['1f: key3'][1]; p2=mr['2f: key+ref1'][1]; p3=mr['3f: key+ref1+ref2'][1]
        print(f'  1f={p1:.4f}  2f+r1={p2:.4f}(Δ{p2-p1:+.4f})  3f={p3:.4f}(Δ{p3-p1:+.4f})')
        del model; torch.cuda.empty_cache()

    # PoseAP
    print('\n' + '='*120)
    print('=== PoseAP (test set, mAP@50:95) ===')
    hdr=f"{'Model':<28s}"
    for m in mode_names: hdr+=f" {m:>14s}"
    print(hdr); print('-'*len(hdr))
    for label in results:
        row=f"{label:<28s}"
        for m in mode_names: row+=f" {results[label][m][1]:14.4f}"
        print(row)

    # Δ vs 1f
    print('\n=== PoseAP Δ vs 1f key³ ===')
    hdr=f"{'Model':<28s}"
    for m in mode_names[1:]: hdr+=f" {'Δ '+m:>14s}"
    print(hdr); print('-'*len(hdr))
    for label in results:
        base=results[label][mode_names[0]][1]
        row=f"{label:<28s}"
        for m in mode_names[1:]: row+=f" {results[label][m][1]-base:+14.4f}"
        print(row)

    # BoxAP
    print('\n=== BoxAP (test set, mAP@50:95) ===')
    hdr=f"{'Model':<28s}"
    for m in mode_names[:4]: hdr+=f" {m:>14s}"
    print(hdr); print('-'*len(hdr))
    for label in results:
        row=f"{label:<28s}"
        for m in mode_names[:4]: row+=f" {results[label][m][0]:14.4f}"
        print(row)

    # save
    csv_path=os.path.join(_SCRIPT_DIR,'results_tables','frame_ablation_all_models_test.csv')
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['model','mode','BoxAP','PoseAP'])
        for label in results:
            for m in mode_names:
                w.writerow([label,m,results[label][m][0],results[label][m][1]])
    print(f'\nSaved: {csv_path}')
