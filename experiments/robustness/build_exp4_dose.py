#!/usr/bin/env python3
"""Build key-only and refs-only natural-occlusion dose-response variants on exp4 clean test."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_exp4_variants as b

LEVELS = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
MODES = ['key', 'refs']

def main():
    bank = b.build_patch_bank()
    os.makedirs(b.OUT_ROOT, exist_ok=True)
    for mode in MODES:
        for p in LEVELS:
            name = f'occ_{mode}_p{int(round(p*100)):03d}'
            out = b.OUT_ROOT / name / 'test'
            if (b.OUT_ROOT / name / 'meta.json').exists() and out.exists():
                print(f'[skip] {name}')
                continue
            print(f'[build] {name}', flush=True)
            b.make_variant(name, 'occlusion', mode, p, bank, green=False)
    print('DOSE BUILD DONE')

if __name__ == '__main__':
    main()
