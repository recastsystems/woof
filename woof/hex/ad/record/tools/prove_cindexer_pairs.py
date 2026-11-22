#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find the CuPy source and image for every CIndexer-scratch word the stage-2 sealer
listed, and check the source shape `carray.rs` verify_copy_indexer_sources requires:
the kernel body is `CUPY_FOR(i, _ind.size()) {` followed by `_ind.set(i);` before any
use of `_ind.get()`. Output: one row per (function, rank, source sha256, image sha256)
with the source line, for carray::copy_indexer_proved. """
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

residual_path, plan_path, cache_dir, out_path = sys.argv[1:5]  # cache_dir: the observer directory (sources/, source-bindings)
residual = json.load(open(residual_path))['residual']['cindexer_scratch_proof']
plan = json.load(gzip.open(plan_path, 'rt'))
modules = {m['id']: m for m in plan['modules']}
by_image = {}
observer = Path(cache_dir)
with gzip.open(observer / 'source-bindings.jsonl.gz', 'rt') as f:
    for line in f:
        b = json.loads(line)
        path = observer / 'sources' / f"{b['source_sha256']}.cu.gz"
        image = observer / 'images' / f"{b['image_sha256']}.gz"
        if path.is_file() and image.is_file():
            # the driver-side image hash used by plans is SHA-256(image + NUL) (source_signatures.py:220)
            driver_sha = hashlib.sha256(gzip.decompress(image.read_bytes()) + bytes(1)).hexdigest()
            by_image.setdefault(driver_sha, set()).add(path)
rows, problems = {}, []
for block, function, arg, rank, module in residual:
    m = modules.get(module)
    if m is None:
        problems.append([function, rank, module, 'module not in plan']); continue
    found = by_image.get(m['sha256'], set())
    if len(found) != 1:
        problems.append([function, rank, module, f'{len(found)} observed sources for image']); continue
    source = next(iter(found))
    raw_source = gzip.open(source).read()
    text = raw_source.decode()
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if re.search(r'\b' + re.escape(function) + r'\s*\(', l)), None)
    if start is None:
        problems.append([function, rank, module, 'function not in source']); continue
    loop = next((i for i in range(start, len(lines)) if lines[i].strip() == 'CUPY_FOR(i, _ind.size()) {'), None)
    nxt = None if loop is None else next((i for i in range(loop + 1, len(lines)) if lines[i].strip()), None)
    ok = loop is not None and nxt is not None and lines[nxt].strip() == '_ind.set(i);'
    gets_before = loop is not None and any('_ind.get' in l for l in lines[start:loop])
    kind = 'copy'
    if not ok:
        # CuPy reduction template: `_in_ind.set(_j);` then the first use `_in_ind.get()` on the next line,
        # and no `_in_ind.get` anywhere before that set (cupy/_core/_reduction.pyx kernel template).
        rset = next((i for i in range(start, len(lines)) if lines[i].strip() == '_in_ind.set(_j);'), None)
        rnext = None if rset is None else next((i for i in range(rset + 1, len(lines)) if lines[i].strip()), None)
        if (rset is not None and rnext is not None and '_in_ind.get()' in lines[rnext]
                and not any('_in_ind.get' in l for l in lines[start:rset])):
            ok, loop, gets_before, kind = True, rset, False, 'reduction_in'
    key = (function, rank, hashlib.sha256(raw_source).hexdigest(), m['sha256'])
    row = rows.setdefault(key, {'function': function, 'rank': rank, 'source_sha256': key[2], 'image_sha256': key[3],
                                'source': str(Path(source).relative_to(observer.parent)), 'line': (loop + 1) if loop is not None else 0,
                                'set_before_get': bool(ok and not gets_before), 'kind': kind, 'words': 0})
    row['words'] += 1
result = {'schema': 'ad.binder.phys-cindexer-pairs/v1', 'pairs': list(rows.values()), 'problems': problems}
Path(out_path).write_text(json.dumps(result, indent=1))
for r in rows.values():
    print(r['function'], r['kind'], r['rank'], r['set_before_get'], r['line'], r['words'], r['source_sha256'][:12], r['image_sha256'][:12])
print('problems', problems[:10], len(problems))
