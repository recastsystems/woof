#!/usr/bin/env python3
"""Apply the fast-route launch transforms to the coalesced 8f8e8b59 column.

Input: the interleaved single-kernel ``model.cu`` that ``coalesce.py`` writes
and its two nz-affine schemas. Output: the shipped ``model.cu`` (the strict
``mynn471_step`` kernel is the same native statement sequence; the fast
route launches the phase kernels in native order) and the schemas with the
per-column arena grown by the bounded cross-launch reserve.

    speed.py IN/model.cu IN/schema49.json IN/schema50.json OUT_DIR
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phases import phases  # noqa: E402
from deadstore import deadstore  # noqa: E402
from fold_locals import fold_locals  # noqa: E402
from overlay import overlay  # noqa: E402
from recurrence import recurrence  # noqa: E402
from zerosink import zerosink  # noqa: E402
from driverzero import driverzero  # noqa: E402
from initialised import initialised  # noqa: E402


def unroll(text, factor):
    """Let ptxas overlap independent loads in innermost call-free loops.

    The lowering pins every loop to ``#pragma unroll 1`` (whole-driver NVVM
    blew up otherwise). Unrolling an innermost loop changes no arithmetic,
    order or rounding (strict builds have no contraction or reassociation);
    it only lets one column's independent arena loads be in flight together.
    """
    out = []; pos = 0; count = 0
    for m in re.finditer(r'#pragma unroll 1\nfor \(', text):
        i = text.index('{', m.end()) + 1; depth = 1
        while depth:
            depth += (text[i] == '{') - (text[i] == '}'); i += 1
        body = text[text.index('{', m.end()) + 1:i - 1]
        if 'for (' in body or re.search(r'\w+_\(m471_work', body):
            continue
        out.append(text[pos:m.start()]); out.append(f'#pragma unroll {factor}\nfor (')
        pos = m.end(); count += 1
    out.append(text[pos:])
    return ''.join(out), count


def transform(text, schema49, schema50, factor=4, stack_frames=True):
    assert 'template<int M471_PHASE>' not in text, 'input is already transformed'
    s49, s50 = schema49['scratch_bytes'], schema50['scratch_bytes']
    text, s49, s50, stack = overlay(text, s49, s50) if stack_frames else (text, s49, s50, None)
    used = f'({s50 - s49}*(MYNN471_NZ-49)+{s49})'
    base = f'(({used}+7)/8*8)'
    text, unread = deadstore(text)
    text, sunk = zerosink(text)
    cleared_by_column = initialised(text, schema49)
    text, cleared = driverzero(text)
    text = fold_locals(text)
    text = recurrence(text)
    text, report = phases(text, base)
    report['unread_private_values'] = unread
    report['plume_clears_moved'] = sunk
    report['driver_clears_left_to_dmp_mf'] = cleared
    if factor > 1:
        text, report['unrolled_loops'] = unroll(text, factor)
    reserve = report['reserve_bytes'] + 8
    out49, out50 = dict(schema49), dict(schema50)
    out49['initialised_by_column'] = out50['initialised_by_column'] = cleared_by_column
    out49['scratch_bytes'] = s49 + reserve
    out50['scratch_bytes'] = s50 + reserve
    report['arena_reserve_bytes'] = reserve
    report['stack'] = stack
    return text, out49, out50, report


if __name__ == '__main__':
    model, first, second, out = map(Path, sys.argv[1:5])
    text, o49, o50, report = transform(model.read_text(), json.loads(first.read_text()),
                                       json.loads(second.read_text()))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.cu').write_text(text, newline='\n')
    (out / 'schema49.json').write_text(json.dumps(o49, indent=2) + '\n', newline='\n')
    (out / 'schema50.json').write_text(json.dumps(o50, indent=2) + '\n', newline='\n')
    print(json.dumps(report, indent=1))
