#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Classify the live device-to-host copies of an hrrrv5 partial plan into checked host values.
Nothing here seals a plan; it writes proposals only.

A `checked_value` copy is one whose bytes are a pure function of plan inputs the seal can
evaluate, so the replay downloads them and refuses unless they equal `expected_hex`:

* device_pools_v842.py:143 `bool(packed.any())` (consumer: release_prepared, :205). Proven
  shape: upload U (payload P) -> cupy_cub_any_pass1(input = U's allocation at offset 0,
  element count word == U.bytes / itemsize) -> cupy_cub_any_pass2(output = copy source) ->
  1-byte copy, with no other write to U's allocation in between. Expected = any(P != 0),
  evaluated as float (so -0.0 is zero) and as integer; a payload where the two differ is
  left unresolved rather than guessed.
* deferred_checks_v842.py:149 (consumer: poll(), a nonzero word raises). Expected = all zero:
  a recording that finished means every word was zero, and a nonzero word in replay is the
  forward's own refusal.
Everything else stays in the residual with its site.
Usage: classify_d2h_checked.py PLAN SITES.json PAYLOAD_ROOT OUT.json
"""
import gzip
import json
import struct
import sys
from pathlib import Path

plan_path, sites_path, payload_root, out = sys.argv[1:5]
# optional: {site: {function, image_sha256, expected_hex}} from image_kernel_value.py runs
image_constants = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else {}
# The hexcore source the recording ran (the installed package this tool ships in, or argv[6]). The two host-read
# sites of the long-wave wrapper are found by file and source text, not by line number. THE BREAKAGE THIS PREVENTS:
# the rules named rrtmg_lw_v842.py:1190 and :1019; the file grew (the lines are 1222 and 1029 on this tree), so a
# door recording from --init at 120 km could not seal: the observer's own reads of the layer counts entered the plan
# as forward host reads ("device_to_host_consumer 2", box-d jobs d120-d and d120-f, 2026-11-11), and the histogram
# read lost its launch-shape check without a word.
hexcore_src = Path(sys.argv[6]) if len(sys.argv) > 6 else Path(__file__).resolve().parents[3]
LW_FILE = 'woof/hex/mpasphys/rrtmg_lw_v842.py'
LW_GROUPS_CALL = '_layer_groups('       # the caller line: groups=_layer_groups(counts,nm,...)
LW_HIST_READ = 'sizes = hist.get()'     # the one host read inside _layer_groups
_lw_lines = None


def lw_site(site, text, caller=False):
    """True when the innermost frame of SITE is a line of rrtmg_lw_v842.py that holds TEXT (caller: not its def)."""
    global _lw_lines
    rel, _, line = site.split(';')[0].rpartition(':')
    if rel != LW_FILE or not line.isdigit():
        return False
    if _lw_lines is None:
        try:
            _lw_lines = (hexcore_src / rel.split('/', 1)[1]).read_text().splitlines()
        except OSError as error:
            raise SystemExit(f'DEFECT: classify_d2h_checked: cannot read {rel} under {hexcore_src}: {error}')
    if not 1 <= int(line) <= len(_lw_lines):
        return False
    found = _lw_lines[int(line) - 1]
    return text in found and not (caller and found.lstrip().startswith('def '))


plan = json.load(gzip.open(plan_path, 'rt'))
sites = json.load(open(sites_path))
site_of = {}
for row in sites['sites']:
    for blk in row['blocks']:
        pass
# rebuild sequence -> site from the per-row stacks is not kept; recompute from residual list
seq_site = {}
for row in sites['sites']:
    seq_site.update({s: row['site'] for s in row.get('sequences', [])})


def touches(op, alloc, lo=0, hi=1 << 62):
    """Any write or read of [lo, hi) of allocation `alloc` (kernel pointer arguments are treated
    as touching from their offset; uploads/clears/copies by their byte range)."""
    if op['kind'] == 'kernel':
        return any(a.get('kind') == 'device' and a['target']['allocation'] == alloc
                   and lo <= a['target']['offset'] < hi for a in op['arguments'])
    d = op.get('destination') or op.get('target')
    if not d or d.get('allocation') != alloc:
        return False
    n = op.get('bytes') or (op.get('data') or {}).get('bytes') or 1
    return d.get('offset', 0) < hi and lo < d.get('offset', 0) + n


def payload(data):
    p = Path(payload_root) / Path(data['path']).name
    return gzip.open(p, 'rb').read()


def any_nonzero(raw, itemsize):
    if itemsize == 4:
        f = struct.unpack(f'<{len(raw)//4}f', raw)
        i = struct.unpack(f'<{len(raw)//4}I', raw)
    elif itemsize == 8:
        f = struct.unpack(f'<{len(raw)//8}d', raw)
        i = struct.unpack(f'<{len(raw)//8}Q', raw)
    elif itemsize == 1:
        return any(raw), any(raw)
    else:
        return None, None
    return any(x != 0.0 for x in f), any(x != 0 for x in i)


LW_BINS = 1024  # rrtmg_lw_v842._LAYER_HISTOGRAM_BINS
result = {'schema': 'ad.binder.phys-d2h-checked/v1', 'complete': False, 'checked': [], 'residual': []}
for name, ops in plan['blocks'].items():
    for k, op in enumerate(ops):
        if op['kind'] != 'device_to_host':
            continue
        seq = op['recorded_sequence']
        site = seq_site.get(seq, '')
        entry = {'block': name, 'sequence': seq, 'bytes': op['bytes'], 'site': site}
        if 'device_pools_v842.py:143' in site and op['bytes'] == 1:
            src = op['source']
            p2 = next((j for j in range(k - 1, -1, -1) if ops[j]['kind'] == 'kernel'
                       and ops[j]['function'] == 'cupy_cub_any_pass2'), None)
            ok = p2 is not None and ops[p2]['arguments'][1]['target'] == src
            p1 = next((j for j in range(p2 - 1, -1, -1) if ops[j]['kind'] == 'kernel'
                       and ops[j]['function'] == 'cupy_cub_any_pass1'), None) if ok else None
            ok = ok and p1 is not None
            if ok:
                inp = ops[p1]['arguments'][0]['target']
                count = struct.unpack('<I', bytes.fromhex(ops[p1]['arguments'][3]['bytes_hex']))[0]
                u = next((j for j in range(p1 - 1, -1, -1) if ops[j]['kind'] == 'upload'
                          and ops[j].get('destination') == inp), None)
                uop = ops[u] if u is not None else None
                ok = (uop is not None and uop.get('data') and count and uop['data']['bytes'] % count == 0)
                lo, hi = inp['offset'], inp['offset'] + (uop['data']['bytes'] if ok else 1)
                between = [ops[j] for j in range(u + 1, k) if ok and j not in (p1, p2)
                           and ops[j]['kind'] != 'synchronization' and touches(ops[j], inp['allocation'], lo, hi)]
                ok = ok and not between
            if ok:
                raw = payload(uop['data'])
                itemsize = uop['data']['bytes'] // count
                fa, ia = any_nonzero(raw, itemsize)
                if fa is not None and fa == ia:
                    entry.update(rule='prepared_any_of_immutable_upload', expected_hex='01' if fa else '00',
                                 upload_sequence=uop['recorded_sequence'], upload_sha256=uop['data']['sha256'],
                                 itemsize=itemsize, elements=count)
                    result['checked'].append(entry)
                    continue
                entry['why'] = f'float/int any() disagree or itemsize {itemsize}'
            else:
                entry['why'] = 'any() chain not proven (upload -> pass1 -> pass2 -> copy)'
        elif 'deferred_checks_v842.py:149' in site:
            entry.update(rule='raise_only_deferred_check_words', expected_hex='00' * op['bytes'])
            result['checked'].append(entry)
            continue
        elif 'cuda_transport_v841.py:1652' in site and op['bytes'] % 4 == 0:
            # species_work_plan (cuda_transport_v841.py:1655-1675) reads only z = (words == 0):
            # zero0 = first zero tracer stays active; every other zero tracer is inactive and is
            # copied from output[zero0]. So a recorded structure with copies fixes z exactly
            # (z = {zero0} + inactive); with no copies, z has at most one zero word.
            nt = op['bytes'] // 4
            passes, copies, j = [], [], k + 1
            while j < len(ops) and not (ops[j]['kind'] == 'kernel'
                                        and ops[j]['function'] == 'hex_transport_species_words'):
                o = ops[j]
                if o['kind'] == 'kernel' and o['function'] in ('transport_edge_values', 'transport_edge_values_mono_v841'):
                    cnt = struct.unpack('<i', bytes.fromhex(o['arguments'][0]['bytes_hex']))[0]
                    passes.append((o['arguments'][6]['target'], cnt))
                if (o['kind'] == 'kernel' and o['function'].startswith('cupy_copy_') and passes
                        and o['arguments'][0].get('kind') == 'patched'
                        and len(o['arguments'][0]['bytes_hex']) == 96 and len(o['arguments'][1]['bytes_hex']) == 96
                        and bytes.fromhex(o['arguments'][0]['bytes_hex'])[32:40] == bytes(8)):
                    # _copy_zero_species: output[first:last] = output[reference], a rank-2 copy
                    # whose source broadcasts one tracer (stride 0); it follows every pass.
                    copies.append(o)
                    break
                j += 1
            why = None
            if not passes:
                why = 'no transport passes after the species words'
            else:
                stride = None
                # tracer stride from the copy CArray (destination strides[0]) when there is a copy
                base = passes[0][0]
                if copies:
                    dst = copies[0]['arguments'][1]
                    raw = bytes.fromhex(dst['bytes_hex'])
                    stride = struct.unpack('<q', raw[32:40])[0]
                    ncopy = struct.unpack('<q', raw[16:24])[0]
                active_ranges = []
                for tgt, cnt in passes:
                    if tgt['allocation'] != base['allocation']:
                        why = 'passes span allocations'; break
                    if stride is None:
                        first = None
                    else:
                        d = tgt['offset'] - base['offset']
                        if d % stride:
                            why = 'pass offset not a tracer multiple'; break
                        first = d // stride
                    active_ranges.append((first, cnt))
                # P1 (ad/phys): when the passes cover every tracer there is no inactive tracer and so no
                # _copy_zero_species; a copy the scan found is another kernel (it runs to the next species
                # words or the plan end). Breakage: the 18-step hrrrv5 seal (a recording host, 2026-11-04) crashed with
                # IndexError on runs[0] at steps 9 and 18 (13 of 13 tracers active, a 2562-row cupy copy found).
                covers_all = why is None and sum(c for _, c in active_ranges) == nt
                if why is None and (not copies or covers_all):
                    if sum(c for _, c in active_ranges) != nt:
                        why = 'no copy but passes do not cover every tracer'
                    else:
                        entry.update(rule='species_zero_words', expected_zero_words=[],
                                     meaning='at most one zero word (no copy in the recorded structure)')
                        result['checked'].append(entry)
                        continue
                if why is None:
                    active = [False] * nt
                    for first, cnt in active_ranges:
                        for t in range(first, first + cnt):
                            active[t] = True
                    inactive = [t for t in range(nt) if not active[t]]
                    runs = []
                    for t in inactive:
                        if runs and runs[-1][1] == t:
                            runs[-1][1] = t + 1
                        else:
                            runs.append([t, t + 1])
                    # one copy kernel per inactive run; the first one names zero0 through its source
                    src = copies[0]['arguments'][0]
                    sraw = bytes.fromhex(src['bytes_hex'])
                    sstride0 = struct.unpack('<q', sraw[32:40])[0]
                    dptr = copies[0]['arguments'][1]['pointers'][0]['target']
                    sptr = src['pointers'][0]['target']
                    out_base = dptr['offset'] - runs[0][0] * stride
                    s, rem = divmod(sptr['offset'] - out_base, stride)
                    if (sstride0 != 0 or sptr['allocation'] != dptr['allocation'] or rem or not 0 <= s < nt
                            or not active[s] or any(not active[t] for t in range(s)) or ncopy != runs[0][1] - runs[0][0]):
                        why = 'copy source/destination do not match a species_work_plan structure'
                    else:
                        entry.update(rule='species_zero_words', expected_zero_words=sorted([s] + inactive),
                                     meaning='exactly these words are zero (zero0 + the copied tracers)',
                                     recorded_passes=[[f, f + c] for f, c in active_ranges],
                                     recorded_copy_runs=runs, zero0=s)
                        result['checked'].append(entry)
                        continue
            entry['why'] = why
        elif lw_site(site, LW_GROUPS_CALL, caller=True):
            # G2 (P1 milestone 2): the recorder observer's own reads (observe_layer_groups.py wraps _layer_groups:
            # counts.get() before the call and indices.get() of every group after it), issued from the wrapper, so the
            # innermost hexcore frame is the caller line (groups=_layer_groups(...)). The forward's only read of
            # this call is sizes = hist.get() inside _layer_groups. THE BREAKAGE THIS PREVENTS: the observer's checks entered the plan as forward host
            # reads and the seal refused them (device_to_host_consumer), 3 per LW call.
            entry.update(rule='recorder_observer_layer_groups_read', observer_read=True)
            result['checked'].append(entry)
            continue
        elif lw_site(site, LW_HIST_READ) and op['bytes'] == 4 * (LW_BINS + 1):
            # G2 (P1 milestone 2): `sizes = hist.get()` in _layer_groups (rrtmg_lw_v842.py), the one host read
            # of the LW wrapper. It fixes every later launch shape of the call, so the recorded launches imply it: each
            # wrapper batch is one mpas_lw_prepare launch with by-value (ng, nm, nl); per layer count the batches sum to
            # the bin (layer count - (nm + 1)), and every column is in exactly one batch. Expected = that implied
            # histogram (overflow bin 0); the replay downloads the live one, refuses on any difference, and rebuilds the
            # batches with the Rust port (layer_groups.rs groups + the recorded chunk) to check the plan's launches.
            # THE BREAKAGE THIS PREVENTS: with radiation live, this copy had no consumer rule and the seal refused
            # (device_to_host_consumer) at every LW call; with radiation held the LW launch shapes were never in a plan.
            hk = next((j for j in range(k - 1, -1, -1) if ops[j]['kind'] == 'kernel'
                       and ops[j]['function'] == 'mpas_layer_histogram'), None)
            i32 = lambda a: struct.unpack('<i', bytes.fromhex(a['bytes_hex'])[:4])[0]
            why = None
            if hk is None or ops[hk]['arguments'][4].get('target') != op['source']:
                why = 'no mpas_layer_histogram launch writes this copy source'
            else:
                ha = ops[hk]['arguments']
                nc, base, nbins = i32(ha[0]), i32(ha[1]), i32(ha[2])
                batches, j = [], k + 1
                while j < len(ops) and not (ops[j]['kind'] == 'kernel' and ops[j]['function'] in ('mpas_layer_counts', 'mpas_layer_histogram')):
                    o = ops[j]
                    if o['kind'] == 'kernel' and o['function'] == 'mpas_lw_prepare':
                        ng, nm, nl = (i32(o['arguments'][t]) for t in range(3))
                        batches.append([nl, ng, nm])
                    j += 1
                if nbins != LW_BINS or not batches or len({b[2] for b in batches}) != 1 or batches[0][2] + 1 != base:
                    why = f'LW batches do not follow the histogram (nbins {nbins}, {len(batches)} batches, base {base})'
                else:
                    hist = [0] * (LW_BINS + 1)
                    for nl, ng, _ in batches:
                        b = nl - base
                        if not 0 <= b < LW_BINS:
                            why = f'layer count {nl} outside the histogram (overflow path, not this rule)'; break
                        hist[b] += ng
                    order = [b[0] for b in batches]
                    if why is None and (sum(hist) != nc or order != sorted(order)):
                        why = f'batches cover {sum(hist)} of {nc} columns or are not ascending in layers'
                    if why is None:
                        entry.update(rule='lw_layer_histogram_from_launches',
                                     expected_hex=struct.pack(f'<{LW_BINS + 1}i', *hist).hex(),
                                     layer_groups={'nc': nc, 'nm': batches[0][2], 'batches': [[nl, ng] for nl, ng, _ in batches]},
                                     groups={str(base + b): n for b, n in enumerate(hist) if n})
                        result['checked'].append(entry)
                        continue
            entry['why'] = why
        elif site in image_constants and k > 0:
            # input-free kernel of a recorded image writes the copy source (image_kernel_value.py)
            want = image_constants[site]
            prev = next((ops[j] for j in range(k - 1, -1, -1) if ops[j]['kind'] == 'kernel'), None)
            if (prev and prev['function'] == want['function'] and prev['module'] == want['image_sha256']
                    and len(prev['arguments']) == 1 and prev['arguments'][0]['target'] == op['source']
                    and len(want['expected_hex']) == 2 * op['bytes']):
                entry.update(rule='input_free_image_constant', expected_hex=want['expected_hex'],
                             image_sha256=want['image_sha256'], function=want['function'])
                result['checked'].append(entry)
                continue
            entry['why'] = 'image-constant kernel does not write this copy source'
        result['residual'].append(entry)
counts = {}
for e in result['checked']:
    counts[e['rule']] = counts.get(e['rule'], 0) + 1
result['counts'] = counts
result['residual_by_site'] = {}
for e in result['residual']:
    result['residual_by_site'][e['site'] or '?'] = result['residual_by_site'].get(e['site'] or '?', 0) + 1
result['expected_true'] = sum(1 for e in result['checked'] if e.get('expected_hex') == '01')
json.dump(result, open(out, 'w'), indent=1)
print(json.dumps({k: result[k] for k in ('counts', 'residual_by_site', 'expected_true')}, indent=1))
