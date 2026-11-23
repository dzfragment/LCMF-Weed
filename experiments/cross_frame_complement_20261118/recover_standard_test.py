"""Recover the official test source groups and degradations by replaying its generators."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('LCMF_DATA_ROOT', HERE.parents[1] / 'dataset')).resolve()
TEST = DATA / 'weed_sliding_merged/test'


def run(output, clean_dir=None):
    if clean_dir is not None:
        clean_dir.mkdir(parents=True, exist_ok=False)
    write_image = cv2.imwrite
    os.environ.update(WEED_SRC_ROOT=str(DATA), WEED_EXP_ROOT=str(DATA),
                      OCCLUDER_BANK=str(DATA / 'patch_bank_train.npy'))
    sys.path.insert(0, str(HERE.parents[1] / 'dataset_tools'))
    import gen_public_weed_sliding as public
    import stage2_sliding as private
    from common import get_group_key

    rows = {}
    with TemporaryDirectory(prefix='exp4-test-replay-') as temp:
        for generator, kind in [(public, 'public'), (private, 'private')]:
            state = {}
            load = generator.load_labels
            noise = generator.apply_noises
            save = generator.save_labels
            mapping = generator.map_labels_to_frame

            def map_frame(labels, frame_x, frame_y, *args, **kwargs):
                state.setdefault('origins', []).append([frame_x, frame_y])
                return mapping(labels, frame_x, frame_y, *args, **kwargs)

            def load_source(path):
                state['source'] = Path(path).stem
                return load(path)

            def blur_flags(image, index, *args, **kwargs):
                if index == 0:
                    state['flags'] = []
                    state['occ'] = generator.common.CURRENT_OCC_MODE
                    if clean_dir is not None:
                        state['clean'] = []
                if clean_dir is not None:
                    state['clean'].append(image.copy())
                state['flags'].append(bool(kwargs['blur_flag']))
                return noise(image, index, *args, **kwargs)

            def record_image(path, image, params):
                path = Path(path)
                name = path.parent.name
                status, jpeg = cv2.imencode('.jpg', image, params)
                assert status
                official = cv2.imread(str(TEST / name / path.name))
                actual = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                identical = np.array_equal(official, actual)
                low_official = cv2.resize(official, (128, 128), interpolation=cv2.INTER_AREA)
                low_actual = cv2.resize(actual, (128, 128), interpolation=cv2.INTER_AREA)
                correlation = np.corrcoef(low_official[:, :, 1].ravel(), low_actual[:, :, 1].ravel())[0, 1]
                assert correlation > .95, (path, correlation)
                if path.name == 'frame_0.jpg':
                    assert name not in rows
                    source = state['source']
                    group = f'public-{source.split("-")[1]}' if kind == 'public' else get_group_key(source)
                    rows[name] = dict(source=source, group=group, kind=kind, blur=state['flags'].copy(),
                                      occ_mode=state['occ'], hashes=[], replay_identical=[], replay_correlation=[])
                    assert len(state['origins']) == 3
                    rows[name]['origins'] = state.pop('origins')
                    if clean_dir is not None:
                        (clean_dir / name).mkdir()
                        rows[name]['clean_hashes'] = []
                if clean_dir is not None:
                    index = int(path.stem[-1])
                    target = clean_dir / name / f'frame_{index}.png'
                    assert write_image(str(target), state['clean'][index], [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    rows[name]['clean_hashes'].append(sha256(target.read_bytes()).hexdigest())
                rows[name]['hashes'].append(sha256((TEST / name / path.name).read_bytes()).hexdigest())
                rows[name]['replay_identical'].append(bool(identical))
                rows[name]['replay_correlation'].append(float(correlation))
                return True

            def check_label(path, labels):
                save(path, labels)
                path = Path(path)
                assert path.read_bytes() == (TEST / path.parent.name / path.name).read_bytes(), path

            with (patch.object(generator, 'load_labels', load_source),
                  patch.object(generator, 'map_labels_to_frame', map_frame),
                  patch.object(generator, 'save_labels', check_label),
                  patch.object(generator, 'apply_noises', blur_flags),
                  patch.object(cv2, 'imwrite', record_image)):
                if kind == 'public':
                    with patch.object(public, 'OUT_ROOT', str(Path(temp) / kind)):
                        public.main('test')
                else:
                    process = private.process_split_from_pool

                    def only_test(*args, **kwargs):
                        split = kwargs['split_name']
                        if split == 'test':
                            return process(*args, **kwargs)
                        count = {'train': 1027, 'val': 68}[split]
                        return dict(total_sequences=count, total_source=len(args[0]),
                                    total_skipped_no_label=0, total_skipped_few_labels=0,
                                    total_skipped_no_valid_pos=0), kwargs['global_seq_idx'] + count

                    Path(temp, kind).mkdir()
                    with (patch.object(private, 'STAGE2_SLIDING_OUT', str(Path(temp) / kind)),
                          patch.object(private, 'process_split_from_pool', only_test),
                          patch.object(private, 'create_syn_visualization'),
                          patch.object(private, 'create_noise_demo')):
                        private.main()
    assert set(rows) == {entry.name for entry in TEST.glob('seq_*')}
    assert all(len(row['hashes']) == len(row['blur']) == 3 for row in rows.values())
    assert len(rows) == 575
    output.write_text(json.dumps(dict(test_root=str(TEST), sequences=rows), indent=2) + '\n')
    count = sum(sum(row['replay_identical']) for row in rows.values())
    print(f'Labels match in {len(rows)} official sequences; {count} replayed images match pixel-for-pixel.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'standard_test_manifest.json')
    parser.add_argument('--clean-dir', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run(args.output, args.clean_dir)
