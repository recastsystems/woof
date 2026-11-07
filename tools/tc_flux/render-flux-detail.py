"""Render a tighter flux view before a native collector retires its scratch.

Consumes the converter's actual frame, already carrying TC_ENTHALPY and
terrain from its pinned init. No source field is synthesized or altered.
"""
import argparse
import json
import os
from pathlib import Path
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('scratch', type=Path)
p.add_argument('--maximum', type=float, default=1000)
a = p.parse_args()
if a.maximum <= 0:
    p.error('maximum must be positive')
os.environ.update(CUDA_VISIBLE_DEVICES='', RUSTWX_RASTER_DEVICE='cpu')
from woof import rustwx
from woof.hex.render_door import resolve_renderer_exe

conversion = json.loads((a.scratch / 'convert-report.json').read_text())
source = Path(conversion['frames'][0]['output'])
out = a.scratch.parent / f'enthalpy-detail-{a.maximum:g}'
out.mkdir(exist_ok=True)
theme = a.scratch / 'flux-detail-theme.json'
theme.write_text(json.dumps(dict(colormaps=dict(levels={
    'var_wrf_tc_enthalpy': [-500, 0, *[a.maximum * n / 12 for n in range(1, 13)]]}))))
began = time.monotonic()
written, failures, skipped = rustwx.run_renderer(resolve_renderer_exe(), source,
    store_root=a.scratch / 'flux-detail-store', out_dir=out,
    products='var:wrf_tc_enthalpy', frames='all', width=1000, height=800,
    theme=str(theme), source_label='hex upward enthalpy flux, detail scale')
receipt = dict(source=str(source), conversion_receipt=str(a.scratch / 'convert-report.json'),
    maximum_w_m2=a.maximum, written=[str(x) for x in written], failures=failures,
    skipped=skipped, wall_seconds=time.monotonic()-began)
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
# Snow change is an unrelated derived-field warning emitted by the renderer.
requested_skips = [item for item in skipped if item[0] != 'snow_depth_change']
if failures or requested_skips or len(written) != 1:
    raise RuntimeError(f'incomplete flux detail: {receipt}')
