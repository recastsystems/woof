#!/usr/bin/env python3
"""Build an offline browser for existing Rust-rendered PNGs using filenames only."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
frames = {}
for path in sorted(args.root.glob('*/png/*/*/*/*.png')):
    dataset, _, domain, product, day, _ = path.relative_to(args.root).parts
    stamp = re.search(r'_valid_(\d{8})_(\d{6})z', path.name)
    if stamp:
        valid = datetime.strptime(''.join(stamp.groups()), '%Y%m%d%H%M%S')
    else:
        lead = int(re.search(r'_f(\d+)_', path.name)[1])
        hour = int(re.search(r'_(\d+)z_f', path.name)[1])
        valid = datetime.strptime(day, '%Y-%m-%d') + timedelta(hours=hour + lead)
    key = f'{dataset} / {domain} / {valid:%Y-%m-%d %H:%M} UTC'
    frames.setdefault(key, {})[product] = path.relative_to(args.root).as_posix()
if not frames:
    parser.error('no domain/product/day PNGs found')
html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Trailer B fallback evidence</title>
<style>body{font:17px system-ui;margin:24px;background:#f5f7fa;color:#182335}h1{font-size:28px}p{max-width:1050px;line-height:1.5}select{font:inherit;margin:5px;padding:8px;max-width:95vw}.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%;background:white}figure{margin:0}figcaption{padding:8px}a{color:#164ba0}@media(max-width:900px){.row{grid-template-columns:1fr}}.note{background:#fff3cf;padding:14px}section{margin:24px 0}</style>
<h1>Trailer B: source door and LES compute evidence</h1>
<p class="note">These are measured fallback runs. No mature Xenia forecast segments were available. The 111 m tree is a calm stability benchmark; a separate 3 km Texas precipitation-band run develops convection. Neither establishes a moving 1 km Xenia handoff or tornado realism. Scales change between frames; read the numbers on each color bar.</p>
<p>Each row pairs vertical velocity, Earth-relative relative vertical vorticity, and column-maximum reflectivity at the same valid time. Heights are geometric AGL; vorticity uses horizontal derivatives at constant geometric height with terrain-slope correction. Statistics exclude two boundary cells. Click any picture to view its full size.</p>
<section><h2>Handoff before / after at the first valid time</h2>
<p>Left: 60 km hex source. Right: initialized 3 km WOOF-ArWen. These use different geographic footprints and separately chosen scales. The nonzero imported vertical motion is visible; this is a field-import comparison. A same-storm, matched-footprint moving-seam comparison remains pending.</p><div id="pair" style="display:grid;grid-template-columns:1fr 1fr;gap:12px"></div></section>
<section><h2>Precipitation-band handoff</h2>
<p>Left: 60 km hex, 03:00 UTC. Right: 3 km WOOF-ArWen. The w pair uses the same 03:00 UTC; reflectivity uses 03:05 UTC after its first diagnostic update. The broad precipitation band remains visible and develops through 05:00 UTC. Geographic footprints differ. Initial WOOF-ArWen reflectivity is zero before physics diagnoses it; the five-minute map is labeled explicitly.</p><div id="bandpair" style="display:grid;grid-template-columns:1fr 1fr;gap:12px"></div></section>
<section><h2>Named-height maps and paired reflectivity</h2>
<label>Run, domain, time <select id="frame"></select></label>
<label>Height <select id="height"><option>500</option><option selected>1000</option><option>3000</option></select> m AGL</label>
<div class="row" id="maps"></div><p id="caption"></p></section>
<section><h2>Time series</h2><div id="charts"></div></section>
<p><a href="CAPTIONS.md">Inspection notes and captions</a> · <a href="../README.md">Commands, timings, memory and remaining work</a></p>
<script>const frames=__FRAMES__;const captions=__CAPTIONS__;
const f=document.querySelector('#frame'),h=document.querySelector('#height');
Object.keys(frames).sort().forEach(key=>f.add(new Option(key,key)));
f.value=Object.keys(frames).find(k=>k.startsWith('full / d03'))||f.value;
function picture(path,label){if(!path)return '';return `<figure><a href="${path}" target="_blank"><img src="${path}" alt="${label}" loading="lazy"></a><figcaption>${label}</figcaption></figure>`;}
function show(){const p=frames[f.value],height=h.value;document.querySelector('#maps').innerHTML=picture(p['wrf_storm_w_'+height+'m'],'Vertical velocity (m/s), '+height+' m AGL')+picture(p['wrf_storm_zeta_'+height+'m'],'Earth-relative vertical vorticity (s⁻¹), '+height+' m AGL')+picture(p.wrf_storm_refl_max,'Column-maximum reflectivity (dBZ)');document.querySelector('#caption').textContent=captions[f.value.split(' / ')[0]]||'';}f.onchange=show;h.onchange=show;show();
for(const [product,label] of [['wrf_storm_w_1000m','w at 1,000 m AGL'],['wrf_storm_zeta_1000m','vertical vorticity at 1,000 m AGL'],['wrf_storm_refl_max','column-maximum reflectivity']])for(const dataset of ['source','fallback']){const key=Object.keys(frames).sort().find(k=>k.startsWith(dataset+' / '));if(key)document.querySelector('#pair').innerHTML+=picture(frames[key][product],(dataset==='source'?'Before: hex source, ':'After: 3 km WOOF-ArWen, ')+label);}
for(const [product,after,label] of [['wrf_storm_w_1000m','03:00','w at 1,000 m AGL'],['wrf_storm_refl_max','03:05','column-maximum reflectivity']])for(const dataset of ['source','band']){const stamp='2026-10-16 '+(dataset==='source'?'03:00':after);const key=Object.keys(frames).find(k=>k.startsWith(dataset+' / ')&&k.includes(stamp));if(key)document.querySelector('#bandpair').innerHTML+=picture(frames[key][product],(dataset==='source'?'Before: hex, ':'After: WOOF-ArWen, ')+stamp+' UTC, '+label);}
for(const name of __CHARTS__)document.querySelector('#charts').innerHTML+=picture(name,name.replaceAll('-',' ').replace('.png',''));
</script></html>'''
captions = {
    'source': '2026-10-16 60 km source, three-hour native history cadence. Broad context footprint; weak vertical motion and clear reflectivity in the WOOF-ArWen footprint.',
    'fallback': '2026-10-16 3 km WOOF-ArWen fallback, two hours with five-minute histories. 49 levels and adaptive stepping. Field-import and runtime evidence.',
    'reduced': 'Reduced fixed 1 km / 333 m / 111 m tree, 48/72/108 cells and 96 levels, ten model minutes. Physics active; PRO 6000 run shown. Separate RTX 4090 benchmark passed.',
    'full': 'Full fixed 1 km / 333 m / 111 m tree, 120/180/270 cells and 96 levels, two model hours. Five-minute diagnostics; selected map times. Calm fallback stability benchmark.',
    'regional': 'Real regional hex history export on a fixed 201×201 latitude/longitude grid. This window samples coarse source cells; it is not the requested 1 km storm source.',
    'band': 'Separate 3 km, 49-level Texas precipitation-band fallback, 03–05 UTC, two model hours. The band develops convective updrafts; this is not the 111 m run or a tracked Xenia supercell.',
}
charts = [p.name for p in sorted(args.root.glob('*time-series*.png'))]
html = html.replace('__FRAMES__', json.dumps(frames).replace('<','\\u003c')).replace('__CAPTIONS__', json.dumps(captions)).replace('__CHARTS__', json.dumps(charts))
(args.root / 'index.html').write_text(html)
print(f'Gallery: {len(frames)} domain/time entries, {sum(map(len,frames.values()))} existing PNGs')
