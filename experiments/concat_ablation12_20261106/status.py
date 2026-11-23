#!/usr/bin/env python3
"""Read queue state and latest epoch without loading any checkpoints."""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=HERE)
    args = parser.parse_args()
    p = args.run_dir / 'status.json'
    status = json.loads(p.read_text()) if p.exists() else {'state': 'not_started'}
    folder = status.get('output')
    phase = status.get('phase', '')
    if folder and phase.startswith('stage'):
        log = Path(folder) / f'{phase}_log.csv'
        if log.exists():
            with log.open() as f:
                rows = list(csv.DictReader(f))
            if rows:
                status['latest_validation'] = rows[-1]
        console = Path(folder) / f'{phase}.log'
        if console.exists():
            with console.open('rb') as f:
                f.seek(max(0, console.stat().st_size - 1600))
                status['latest_console_line'] = f.read().decode(errors='replace').replace('\r', '\n').strip().split('\n')[-1]
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
