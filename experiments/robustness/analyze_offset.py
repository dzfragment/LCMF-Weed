#!/usr/bin/env python3
"""分析 offset-occlusion 实验: 遮挡块中心远离关键点."""
import csv
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TYPES = ['natural_mixed', 'natural_soil', 'natural_veg',
         'green_solid', 'gray_solid', 'noise']
rng = np.random.default_rng(20261101)


def agg(rows):
    n = sum(r['n_gt'] for r in rows)
    tp25 = sum(r.get('n_tp_conf25', 0) for r in rows)
    pred25 = sum(r.get('n_pred_conf25', 0) for r in rows)
    tp1 = sum(r.get('n_tp_fp1', 0) for r in rows)
    vis_gt = sum(r.get('n_gt_vis', 0) for r in rows)
    vis_tp = sum(r.get('n_tp_vis_conf25', 0) for r in rows)
    cov_gt = sum(r.get('n_gt_cov', 0) for r in rows)
    cov_tp = sum(r.get('n_tp_cov_conf25', 0) for r in rows)
    errs = [e for r in rows for e in r.get('errs_conf25', [])]
    return {'R25': tp25 / max(n, 1), 'Rfp1': tp1 / max(n, 1),
            'P25': tp25 / max(pred25, 1),
            'FPimg': (pred25 - tp25) / max(len(rows), 1),
            'nPE': float(np.mean(errs)) if errs else None,
            'R_vis': vis_tp / max(vis_gt, 1), 'hit_cov': cov_tp / max(cov_gt, 1),
            'n_vis': vis_gt, 'n_cov': cov_gt}


def boot(a_rows, b_rows, metric, groups):
    def counts(rows):
        d = defaultdict(lambda: [0, 0])
        for r in rows:
            g = groups.get(r['seq'], r['seq'])
            if metric == 'R25':
                a, b = r.get('n_tp_conf25', 0), r['n_gt']
            elif metric == 'hit_cov':
                a, b = r.get('n_tp_cov_conf25', 0), r.get('n_gt_cov', 0)
            else:
                raise ValueError(metric)
            d[g][0] += a; d[g][1] += b
        return d
    da, db = counts(a_rows), counts(b_rows)
    keys = sorted(set(da) | set(db))
    vals = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys), replace=True)
        ta = sum(da[keys[i]][0] for i in pick); na = sum(da[keys[i]][1] for i in pick)
        tb = sum(db[keys[i]][0] for i in pick); nb = sum(db[keys[i]][1] for i in pick)
        vals.append(ta / max(na, 1) - tb / max(nb, 1))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    base = Path(__file__).resolve().parent / 'results'
    data = Path(os.environ.get('LCMF_DATA_ROOT', Path(__file__).resolve().parents[2] / 'dataset'))
    NORMAL = json.loads((base / 'offset_normal.json').read_text())
    NOREFS = json.loads((base / 'offset_norefs.json').read_text())
    provenance = json.loads((data / 'robustness/clean/provenance.json').read_text())
    groups = {name: row.get('source') or row.get('group') or name
              for name, row in provenance.items()}
    out = []
    for t in TYPES:
        v = f'offset_occ_{t}_p050'
        rn = NORMAL[f'{v}|mf']; rs = NORMAL[f'{v}|sf']; rr = NOREFS[f'{v}|mf_norefs']
        am, asf, ar = agg(rn['per_seq']), agg(rs['per_seq']), agg(rr['per_seq'])
        d = boot(rn['per_seq'], rs['per_seq'], 'R25', groups)
        dc = boot(rn['per_seq'], rr['per_seq'], 'R25', groups)
        dh = boot(rn['per_seq'], rs['per_seq'], 'hit_cov', groups)
        out.append({
            'type': t,
            'mf_AP10': rn['agg']['point_ap_10px'], 'sf_AP10': rs['agg']['point_ap_10px'],
            'mf_norefs_AP10': rr['agg']['point_ap_10px'],
            'mf_R25': am['R25'], 'sf_R25': asf['R25'], 'mf_norefs_R25': ar['R25'],
            'mf_Rfp1': am['Rfp1'], 'sf_Rfp1': asf['Rfp1'],
            'mf_hit_cov': am['hit_cov'], 'sf_hit_cov': asf['hit_cov'],
            'mf_norefs_hit_cov': ar['hit_cov'],
            'mf_R_vis': am['R_vis'], 'sf_R_vis': asf['R_vis'],
            'mf_FPimg': am['FPimg'], 'sf_FPimg': asf['FPimg'],
            'mf_minus_sf_R25': d[0], 'mf_minus_sf_R25_lo': d[1], 'mf_minus_sf_R25_hi': d[2],
            'compl_gain_R25': dc[0], 'compl_gain_lo': dc[1], 'compl_gain_hi': dc[2],
            'mf_minus_sf_hit_cov': dh[0], 'mf_minus_sf_hit_cov_lo': dh[1],
            'mf_minus_sf_hit_cov_hi': dh[2],
        })
    with open(base / 'offset_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    lines = ['# Offset-occlusion experiment', '',
             'Occluder center is 25-40px from GT keypoint; a center-predictor cannot score.',
             '',
             '| type | MF AP10 | SF AP10 | MF no-refs AP10 | MF R25 | SF R25 | MF no-refs R25 | MF hit_cov | SF hit_cov | MF no-refs hit_cov | MF FP/img |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in out:
        lines.append(f"| {r['type']} | {r['mf_AP10']:.3f} | {r['sf_AP10']:.3f} | "
                     f"{r['mf_norefs_AP10']:.3f} | {r['mf_R25']:.3f} | {r['sf_R25']:.3f} | "
                     f"{r['mf_norefs_R25']:.3f} | {r['mf_hit_cov']:.3f} | "
                     f"{r['sf_hit_cov']:.3f} | {r['mf_norefs_hit_cov']:.3f} | {r['mf_FPimg']:.2f} |")
    lines += ['', '## MF vs SF', '',
              '| type | MF-SF R25 [95% CI] | MF-SF hit_cov [95% CI] | complementarity gain R25 (MF normal - no-refs) |',
              '|---|---:|---:|---:|']
    for r in out:
        lines.append(f"| {r['type']} | {r['mf_minus_sf_R25']*100:+.2f}pp "
                     f"[{r['mf_minus_sf_R25_lo']*100:+.2f},{r['mf_minus_sf_R25_hi']*100:+.2f}] | "
                     f"{r['mf_minus_sf_hit_cov']*100:+.2f}pp "
                     f"[{r['mf_minus_sf_hit_cov_lo']*100:+.2f},{r['mf_minus_sf_hit_cov_hi']*100:+.2f}] | "
                     f"{r['compl_gain_R25']*100:+.2f}pp "
                     f"[{r['compl_gain_lo']*100:+.2f},{r['compl_gain_hi']*100:+.2f}] |")
    (base / 'offset_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
