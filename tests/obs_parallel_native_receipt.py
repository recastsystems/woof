"""Small native concurrency evidence, explicitly outside the C3 public replay.

Run on the CPU box with the real Rust bridges. The fields are synthetic. No
forecast fraction or Iowa/CONUS speed claim can be made from this receipt.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from woof.obs.cc_qc import CcQcParams
from woof.obs.dealias import DealiasParams
from woof.obs.radar_grid import write_radar_grid
from woof.obs.superob import SuperobParams, merge_contributions, superob_volume
from woof.timeline.ledger import Ledger, span
from tools.nexrad_archive import ObservationRss, require_native_observations
from test_obs_superob_rust_parity import _grid, _volume, ACCUMULATORS, FIELDS


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require_native_observations()
    grid = _grid(nx=31, ny=31, nz=8)
    volumes = [_volume(grid, j=15, i=15, site_id=f'KP{i:02d}', seed=i) for i in range(2)]
    params = SuperobParams(dealias=DealiasParams(engine='region-global'), cc_qc=CcQcParams())
    result = {'scope': 'synthetic native concurrency evidence, not C3 or a public replay', 'arms': {}}
    for arm, workers in [('before', 1), ('after', 2)]:
        folder = out / arm
        folder.mkdir(exist_ok=True)
        ledger = Ledger(folder / 'timeline.jsonl', run_id=f'synthetic-native-{arm}',
                        rental_id='cpu-verification', product='da')
        ledger.emit('worker.claim', attrs={'basis': 'bench proxy envelope',
            'scope': 'synthetic native verification only', 'worker_count': workers})
        def site(item):
            index, volume = item
            with span('prep.obs.radar', ledger=ledger,
                      attrs={'slot': 0, 'site': volume.site.id, 'span_id': f'radar-{index}'}):
                return superob_volume(volume, grid, params=params, clear_air_from_censor=True)
        with ObservationRss() as rss:
            if workers == 1:
                contributions = [site(item) for item in enumerate(volumes)]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    contributions = list(pool.map(site, enumerate(volumes)))
            with span('prep.obs.merge', ledger=ledger, attrs={'slot': 0, 'span_id': 'merge'}):
                merged = merge_contributions(contributions, grid, params=params)
            path = folder / 'radar.nc'
            with span('prep.obs.write', ledger=ledger, attrs={'slot': 0, 'span_id': 'write'}):
                write_radar_grid(path, merged, grid, valid_time='2026-10-07T12:00:00Z', params=params)
        ledger.emit('finish', attrs={'basis': 'bench proxy envelope', 'scope': 'synthetic native verification only'})
        result['arms'][arm] = {'workers': workers, 'rss': rss.receipt(),
            'radar_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'accumulator_sha256': {f'{i}/{name}': digest(getattr(c, name))
                for i, c in enumerate(contributions) for name in ACCUMULATORS},
            'merged_field_sha256': {name: digest(getattr(merged, name)) for name in FIELDS}}
        del contributions, merged
    before, after = (result['arms'][arm] for arm in ('before', 'after'))
    for key in ('radar_file_sha256', 'accumulator_sha256', 'merged_field_sha256'):
        assert before[key] == after[key], key
    result['identity'] = 'PASS for this synthetic native fixture only'
    result['forecast_proxy_ff'] = None
    (out / 'identity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['identity'])


if __name__ == '__main__':
    main()
