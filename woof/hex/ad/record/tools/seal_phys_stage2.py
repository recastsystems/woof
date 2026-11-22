#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of the hrrrv5 milestone-1 plan sealer .

Applies two sealing rules (103.2 and 103.3 of the recording notes) to a partial plan
(make_partial_plan.py) of recording A, with recording B as the second independent recording:

* R3 holds. Every operation issued inside a held slot (radiation_sw, radiation_lw,
  cloudiness, o3climatology_from_MPAS, land_surface) leaves the plan, except `allocate`
  (allocation lifetimes stay structural). At the slot end the plan uploads exactly the
  words the slot changed in the ordinary forward, per step, from the tracer's
  `HeldSlotWrite` records (driver_trace_cupti_held.c). Those uploads carry origin
  `held: radiation (milestone 1)` or `held: land (milestone 1)`.
* R2 constants. A live kernel scalar word identical at every aligned launch of every
  recorded step in BOTH recordings is carried as recorded bytes with fixed_source class
  `recorded_constant_two_recordings`; the receipt carries both word tables' SHA-256.
* Varying live words get the live rule their source gives:
  - the step counter (`mynnsfc_columns` arg 7, `cupy_fill` arg 0): StepInteger(1, 0, 4);
  - TEMPO `tempo_warp_steps` arg 20 (`radii`, tempo_v303.py:5022 radii_due):
    PhysicsFrame `tempo_radii_due_i32`;
  - CuPy CIndexer scratch words (carray.cuh CIndexer::index_, written by set() before get()):
    listed for the CupyCopyIndexer proof; the bytes are kept with class
    `cupy_indexer_scratch_pending_proof` (the runner refuses that class until proved).
* Image-less library launches: native CUB/Thrust operations as in stage 1.

The output is still complete=false; the residual lists everything left.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

from classify_r2_r3 import HELD, MILESTONE, Recording, masked, jsonl, load

HOLD_LABEL = {'radiation': 'held: radiation (milestone 1)', 'land': 'held: land (milestone 1)'}
HOLD_SLOTS = ('radiation_sw', 'radiation_lw', 'cloudiness', 'o3climatology_from_MPAS', 'land_surface')
if MILESTONE == '2':
    HOLD_SLOTS = ()  # G2: radiation and land live
STEP_WORDS = {('mynnsfc_columns', 7), ('cupy_fill', 0)}
RADII_WORD = ('tempo_warp_steps', 20)


def cindexer_scratch_only(words, nbytes):
    """True when every differing byte lies in a CIndexer index_ region: the struct is
    size (8) + shape (8*ndim) + index (8*ndim), so ndim = (nbytes - 8) / 16."""
    if (nbytes - 8) % 16:
        return False, None
    ndim = (nbytes - 8) // 16
    head = 8 + 8 * ndim
    heads = {w[:2 * head] for w in words}
    return len(heads) == 1, ndim


def concat_count_word(arg, count):
    """True when arg's 8-byte words are exactly CuPy's CArray (ptr, size, shape, stride) or CIndexer
    (size, shape, index) for a rank-1 array of `count` elements (P1, cupy_concatenate_same_size)."""
    if count is None:
        return False
    raw = bytes.fromhex(arg['bytes_hex'])
    words = [int.from_bytes(raw[k:k + 8], 'little') for k in range(0, len(raw), 8)]
    if len(raw) == 32:
        return words[1] == count and words[2] == count
    if len(raw) == 24:
        return words[0] == count and words[1] == count
    return False


def resolve(catalog_allocs, base, clock):
    hits = [a for a in catalog_allocs
            if a['recorded_base'] <= base < a['recorded_base'] + a['bytes']
            and a['allocated_clock_ns'] <= clock
            and (a.get('freed_clock_ns') is None or clock < a['freed_clock_ns'])]
    if len(hits) != 1:
        return None
    return hits[0]



# ---- P1 sealer-output-steps (ad/phys, 2026-11-04): rules for the 3-hourly output steps (9 and 18 of the 18-step
# hrrrv5 plan). Breakage: the 18-step seal of the a recording host pair (r18a/r18b) refused at finalize with 425 history
# device-to-host copies, 92 unresolved kernel arguments, 11 uploads without a rule and 2 CIndexer proofs, all at
# steps 9 and 18 only; steps 1-8 and 10-17 had none.

#: The history/diagnostics stream's download (streams_output_v842.py:426 `_queue_download`, memcpyAsync of each
#: queued field to its pinned host target). The host bytes go to the netCDF writer thread only; nothing in the
#: forecast reads them back, so the replay downloads them as output (logged with their SHA-256), never as state.
HISTORY_OUTPUT_SITES = ('woof/hex/mpasphys/streams_output_v842.py:426',)

#: Constant uploads whose bytes are a literal in the pinned source line (mpasdiag/isobaric.py). Each is checked
#: byte for byte against the float32 encoding of the literal, and the line text is checked when the source is given.
SOURCE_LITERALS = {
    'mpasdiag/isobaric.py:201': ('targets = cp.broadcast_to(cp.asarray(LEVELS, dtype=cp.float32), (nc, 8)).copy()',
                                 (50, 100, 200, 250, 500, 700, 850, 925)),
    'mpasdiag/isobaric.py:207': ('else cp.broadcast_to(cp.asarray(LEVELS, dtype=cp.float32), (nv, 8)).copy())',
                                 (50, 100, 200, 250, 500, 700, 850, 925)),
    'mpasdiag/isobaric.py:231': ('d["t_iso_levels"][:5] = cp.asarray((30000, 35000, 40000, 45000, 50000), cp.float32)',
                                 (30000, 35000, 40000, 45000, 50000)),
    'mpasdiag/isobaric.py:237': ('d["z_iso_levels"][:13] = cp.asarray(tuple(range(30000, 90001, 5000)), cp.float32)',
                                 tuple(range(30000, 90001, 5000))),
}
#: Pool fields re-packed after atm_reset_diagnostics (driver_rrfs.py:974 reset_native_diagnostics: the host storage
#: is reset to the row's diagnostic_reset_values, then device_pools_v842.py:106 packs it). Accepted only when the
#: payload is all zero bytes inside that slot, and replayed as a zero clear of the same range.
#: The resident output diagnostics read a produced field's initial host-pool words once, when the diagnostic is first
#: produced (the first output step, not step 1), and upload them into the field's device buffer; every later change
#: comes from resident views or the diagnostic kernels (resident_output_diagnostics_v842.py buffer()). Such an upload
#: is an immutable input of the trajectory at its own step when its sha is on the run's retention list (equal in both
#: recordings). Matched by file and source text, not by line number. THE BREAKAGE THIS PREVENTS: a 1 h x1.2562 hrrrv5
#: pair on woof hex 0.4.15 refused with 3 'upload_needs_live_rule' words (steps 2 and 3, slot outside, 604,868 and
#: 1,208,556 bytes) because immutable inputs were admitted at step 1 only.
#: The moisture slot the isobaric diagnostics read (current_qv) starts the same way, from the initial level-1 words of
#: state scalars(index_qv), uploaded by cp.array; the observer records cp.array of a host array as it records
#: cp.asarray, so that upload has its site. THE BREAKAGE THIS PREVENTS: the 12 km release example recorded by the
#: release recorder refused at seal with one 'upload_needs_live_rule' word, step
#: 15, sequence 361733, allocation 1343, 682,276 bytes = 59 x 2,891 float32 (qv 0 to 0.0178), "payload is not a table
#: of 8-byte words": the upload had no site because only cp.asarray was observed.
RESIDENT_OUTPUT_INITIAL = ('mpasphys/resident_output_diagnostics_v842.py',
                           "self.buffers[key] = self.cp.asarray(host, order='C')",
                           "self.buffers[key] = cp.array(initial[index], dtype=initial.dtype, copy=True, order='C')")


#: The MCICA jump table of the long-wave and short-wave radiation (rrtmg_lw_v842.py
#: xorshift_jump_table(draws, gpoints): the KISS xorshift advanced g*draws steps for each g-point g, a (gpoints, 32)
#: uint32 table). It is a function of two integers only (draws = the layer count of a long-wave layer group, or twice
#: it; gpoints = 140 long-wave, 112 short-wave), built on the host and uploaded the first time a layer count is met.
#: THE BREAKAGE THIS PREVENTS: an upload is admitted as immutable input only at step 1, so a window in which a layer
#: group first appears at a later radiation step could not be sealed (x1.40962 2022-01-29_12, 4 steps: the group with
#: 62 layers first appears at step 4; "residual uploads not equal in both recordings (or without sha):
#: [('e02b8c11...', 17920)]", a recording host, 2026-11-09). The rule recomputes the
#: table for every draw count up to KISS_JUMP_MAX_DRAWS and admits the upload, at any step, only when the SHA-256 of
#: the recomputed bytes equals the recorded upload's; an upload the recorder did not retain gets the recomputed bytes
#: as its payload (the same bytes: the SHA-256 is the recorded one).
KISS_JUMP_SITE = 'mpasphys/rrtmg_lw_v842.py:1084'
KISS_JUMP_TEXT = 'table = cp.asarray(np.asarray(rows, dtype=np.uint32))'
KISS_JUMP_GPOINTS = (140, 112)


#: A rule above names its site by a line of the forward it was written on (file:line). The shipped forward is not
#: that file: lines move, and a line can be rewritten without changing what it does. So each rule site is found in the
#: shipped source by TEXT: every line of the same file whose stripped text is one of the site's texts IS that site
#: (shipped_sites). Texts: the history stream's two downloads in _queue_download (staged through a device block when
#: the card has room, else in the step's own stream; the same fields to the same pinned host targets, read by the
#: writer only), the pool re-pack after the diagnostics reset (both spellings of the one upload), the isobaric
#: literals and the MCICA jump table.
#: THE BREAKAGE THIS PREVENTS: a 0.6.0 build's recorder could not seal any window that holds an output
#: time. The 12 km example (15 steps of 60 s, history every 15 minutes) recorded from the wheel ended "pair cannot
#: seal: stage-2 residual {'upload_needs_live_rule': 9, 'device_to_host_consumer': 27, ...}", every word at step 15:
#: the 27 history downloads came from streams_output_v842.py:426 (the rule said :394), the isobaric literals from
#: isobaric.py:201 and :207 (the rule said :199 and :205) and the reset re-pack from device_pools_v842.py:106 (the
#: rule said :104). The shipped 12 km and 3 km example plans had come from another recorder that runs an older copy
#: of those files, so nothing had shown it (a 12 km recording on an RTX 4090, 2026-11-16 02:15Z).
SITE_TEXTS = {
    'mpasphys/streams_output_v842.py:426': (
        'cp.cuda.runtime.memcpyAsync(target.ctypes.data, stage.data.ptr, stage.nbytes,',
        'cp.cuda.runtime.memcpyAsync(target.ctypes.data, source.data.ptr, source.nbytes,'),
    'mpasphys/device_pools_v842.py:106': (
        'packed=self.cp.ascontiguousarray(self.cp.asarray(selected)).reshape(selected.shape)',
        "packed=self.cp.asarray(selected, order='C').reshape(selected.shape)"),
    KISS_JUMP_SITE: (KISS_JUMP_TEXT,),
    **{site: (text,) for site, (text, _values) in SOURCE_LITERALS.items()},
}


def shipped_sites(hexcore_src):
    """({'rel/path.py:line' of the shipped source: rule site}, {rule sites found in the shipped source}).
    A shipped line that carries a rule's text is that rule's site, wherever it now sits; a line whose text belongs
    to two rule sites is left out (it names neither)."""
    alias, verified, twice = {}, set(), set()
    for site, texts in SITE_TEXTS.items():
        rel = site.rsplit(':', 1)[0]
        try:
            lines = [line.strip() for line in (Path(hexcore_src) / rel).read_text().splitlines()]
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if line in texts:
                key = f'{rel}:{number}'
                if alias.get(key, site) != site:
                    twice.add(key)
                alias[key] = site
                verified.add(site)
    for key in twice:
        alias.pop(key)
    return alias, verified
KISS_JUMP_MAX_DRAWS = 1024
_KISS_JUMP_SHA = {}


def kiss_jump_table_bytes(draws, gpoints):
    """The bytes rrtmg_lw_v842.xorshift_jump_table(draws, gpoints) uploads (little-endian uint32, row-major)."""
    import struct

    def apply(columns, v):
        out, i = 0, 0
        while v:
            if v & 1:
                out ^= columns[i]
            v >>= 1
            i += 1
        return out

    def compose(outer, inner):
        return [apply(outer, column) for column in inner]
    base = []
    for i in range(32):
        v = 1 << i
        v ^= (v << 13) & 0xffffffff
        v ^= v >> 17
        v ^= (v << 5) & 0xffffffff
        base.append(v)
    identity = [1 << i for i in range(32)]
    power, n = identity, int(draws)
    while n:
        if n & 1:
            power = compose(base, power)
        base = compose(base, base)
        n >>= 1
    rows, current = [], identity
    for _ in range(int(gpoints)):
        rows.append(current)
        current = compose(power, current)
    return struct.pack(f'<{32 * int(gpoints)}I', *[w for row in rows for w in row])


def kiss_jump_table_draws(sha256, nbytes):
    """The draw count whose jump table has this SHA-256 and size, or None (not a jump table)."""
    if nbytes % 128 or nbytes // 128 not in KISS_JUMP_GPOINTS:
        return None
    gpoints = nbytes // 128
    if not any(k[0] == gpoints for k in _KISS_JUMP_SHA):
        for draws in range(1, KISS_JUMP_MAX_DRAWS + 1):
            _KISS_JUMP_SHA[(gpoints, hashlib.sha256(kiss_jump_table_bytes(draws, gpoints)).hexdigest())] = draws
    return _KISS_JUMP_SHA.get((gpoints, sha256))


def block_key(name):
    """Sort key that puts step blocks in step order. THE BREAKAGE THIS PREVENTS: the names are
    step_01 .. step_NN, and sorted as text a plan of 100 or more steps visits step_100 before step_11, so a
    first-use scan over the whole plan reads the steps out of order. Below 100 steps both orders are the same."""
    head, _, tail = name.partition('_')
    return (0, int(tail), '') if head == 'step' and tail.isdigit() else (1, 0, name)


def resident_output_initial(site, hexcore_src, rule=RESIDENT_OUTPUT_INITIAL):
    """True when SITE ('rel/path.py:line') is the upload RULE names (file, then source texts; default the resident
    output diagnostics' first-use buffer upload)."""
    if not site or not hexcore_src:
        return False
    rel, _, line = site.rpartition(':')
    if rel != rule[0] or not line.isdigit():
        return False
    try:
        text = (Path(hexcore_src) / rel).read_text().splitlines()[int(line) - 1]
    except (OSError, IndexError):
        return False
    return any(t in text for t in rule[1:])


#: A limited-area forecast's boundary refresh: when the window reaches the next boundary-file time the regional
#: forecast reads that file's values on the host and uploads them onto the boundary support
#: (cuda_regional_forecast_v841.py _derive_on_support: upload(), cp.asarray(picked)). Those bytes are the file's, not the
#: state's: an immutable input of the trajectory at the step it is read, admitted (as the resident output's first
#: words are) when its sha is on the run's retention list, i.e. equal in both recordings (gen_immutable.py names it in
#: the retention round). Matched by file and source text, not by line number. THE BREAKAGE THIS PREVENTS: a 30 km
#: 1 h window from 12Z (25 steps of 150 s) refused at seal with 17 'upload_needs_live_rule' words at step 25, all at
#: cuda_regional_forecast_v841.py:1255, the 13Z boundary file (a 30 km recording, 2026-11-18), so no
#: limited-area window could cross a boundary-file hour; the example windows (15 and 16 min) never reach one.
LBC_FILE_VALUES = ('cuda_regional_forecast_v841.py', 'return cp.ascontiguousarray(cp.asarray(picked).T)')


RESET_ZERO_SITES = {('mpasphys/device_pools_v842.py:106', 'atm_reset_diagnostics')}
#: rule: the same re-pack when the recorder did not keep its bytes. The recorder hashes every upload
#: (upload_capture_thread.h, the HostUploadBytes record: host address, size, SHA-256) but keeps the bytes only when the
#: SHA-256 is on the mesh's retention list or the step is <= 1; the first diagnostics reset of a run is at a later
#: step (9 at 480 km, 15 at 120 km, 30 at 60 km, 60 on the 3 km box), where driver_rrfs.py:1003 binds three
#: diag_physics fields no scheme has bound yet (ave_prate_1min, _5min, _10min: nCellsSolve float32 words of untouched
#: host storage each) just before the reset copy plan fills them. 480 km and 120 km seal only because their lists
#: happen to hold the SHA-256 of that many zero bytes. THE BREAKAGE THIS PREVENTS: x1.163842 (60 km), 60 steps,
#: 2023-03-31_12: "upload_needs_live_rule": 3, sequences 409360, 409381, 409402 at step 30, "not a copy plan, RUC or
#: TEMPO table" (stage2-residual.json, 2026-11-10), so no 60 km window passed 2 h 54 min and no 3 km
#: window passed 14 min 45 s. The rule admits such an upload only when the recorded SHA-256 equals the SHA-256 the
#: sealer computes for that many zero bytes, and emits the row the retained payload gives (a zero clear); a size that
#: is zero or not whole float32 words, any other SHA-256, any other site or slot stay refused as before.
RESET_ZERO_SLOTS = {slot for _, slot in RESET_ZERO_SITES}
RESET_ZERO_MAX_BYTES = 1 << 31
_ZERO_SHA = {}


def zero_sha256(nbytes):
    """SHA-256 (hex) of nbytes zero bytes."""
    if nbytes not in _ZERO_SHA:
        digest, left, chunk = hashlib.sha256(), nbytes, bytes(1 << 20)
        while left > 0:
            digest.update(chunk[:left])
            left -= min(left, len(chunk))
        _ZERO_SHA[nbytes] = digest.hexdigest()
    return _ZERO_SHA[nbytes]


def reset_zero_sha_rule(op, site, slot, sha256, nbytes):
    """A reset re-pack with no retained payload whose recorded SHA-256 is that of nbytes zero bytes becomes the zero
    clear literal_upload_rule gives for the retained payload (the same row, key for key), or None."""
    if op.get('kind') != 'upload' or site is None or (site, slot) not in RESET_ZERO_SITES:
        return None
    if type(nbytes) is not int or not 0 < nbytes <= RESET_ZERO_MAX_BYTES or nbytes % 4:
        return None
    if not isinstance(sha256, str) or sha256 != zero_sha256(nbytes):
        return None
    return {'kind': 'clear', 'stream': 0, 'destination': op['destination'], 'bytes': nbytes, 'value': 0,
            'proof': {'rule': 'reset_diagnostics_zero', 'site': site, 'slot': slot, 'upload_sha256': sha256}}


def upload_hash_record(seq, copies, records, sequences=None):
    """(sha256, bytes) the tracer hashed for the host-to-device copy at recorded sequence seq, or (None, None).
    copies: {copy sequence: (host address, byte count)} of the MemcpyHtoD records; records: the HostUploadBytes
    records sorted by sequence. The hash record is the nearest one before the copy (the tracer writes it 3 sequence
    numbers ahead; under 8 is required, as for the copy-plan and pointer tables) and must be of the same host
    address and size as the copy and of every byte (written == bytes)."""
    import bisect
    call = copies.get(seq)
    if call is None:
        return None, None
    if sequences is None:
        sequences = [r['sequence'] for r in records]
    k = bisect.bisect_left(sequences, seq) - 1
    if k < 0 or not 0 < seq - sequences[k] < 8:
        return None, None
    r = records[k]
    if (r.get('source'), r.get('bytes')) != tuple(call) or r.get('written') != r.get('bytes'):
        return None, None
    return r.get('sha256'), r.get('bytes')


def history_output_rule(op, site, alias=None):
    """A device-to-host copy of the history stream becomes an output-only download, or None. ALIAS: shipped_sites()'s
    map, so the download is known by its text in the shipped source and not by a line number."""
    first = (site or '').split(';')[0]
    if alias and first.startswith('woof/hex/'):
        first = 'woof/hex/' + alias.get(first[len('woof/hex/'):], first[len('woof/hex/'):])
    if op.get('kind') != 'device_to_host' or first not in HISTORY_OUTPUT_SITES:
        return None
    live = {k: op[k] for k in ('kind', 'stream', 'source', 'bytes', 'asynchronous', 'recorded_sequence') if k in op}
    live.update(host_consumer='observational_output', proof='history_stream_output', site=site.split(';')[0])
    return live


def literal_upload_rule(op, site, slot, payload, verified=None):
    """A retained upload whose bytes are a source literal (immutable input) or a reset zero (clear), or None.
    SITE is the rule site (shipped_sites has already read the shipped line's text); VERIFIED: the rule sites whose
    text is in the shipped source, when the source was given."""
    import struct
    if op.get('kind') != 'upload' or not op.get('data') or payload is None or site is None:
        return None
    if site in SOURCE_LITERALS:
        text, values = SOURCE_LITERALS[site]
        if payload != struct.pack(f'<{len(values)}f', *values):
            return None
        if verified is not None and site not in verified:
            return None
        return {'kind': 'upload', 'stream': 0, 'destination': op['destination'], 'data': op['data'],
                'pointers': [], 'origin': 'immutable_input',
                'proof': {'rule': 'source_literal_float32', 'site': site, 'values': list(values)}}
    if (site, slot) in RESET_ZERO_SITES and payload and payload == bytes(len(payload)):
        return {'kind': 'clear', 'stream': 0, 'destination': op['destination'], 'bytes': len(payload), 'value': 0,
                'proof': {'rule': 'reset_diagnostics_zero', 'site': site, 'slot': slot,
                          'upload_sha256': op['data'].get('sha256')}}
    return None


def elf_kernel_names(image):
    import re
    return set(n.decode() for n in re.findall(rb'\.text\.([A-Za-z_][A-Za-z_0-9]*)\x00', image))


def layout_type_arguments(op, sig, allocs, clock):
    """Type a kernel's unresolved arguments from its source signature (make_partial_plan's source rule):
    pointers must resolve to one live allocation at the launch clock (null stays bytes), scalars become bytes for
    the R2 verdict. Returns (arguments, unresolved_count), or None when the signature does not fit the launch."""
    args = op['arguments']
    if sig is None or len(sig) != len(args) or any(width != len(arg.get('bytes_hex', '')) // 2
                                                   for (_, width), arg in zip(sig, args)):
        return None
    out, left = [], 0
    for (kind, _), arg in zip(sig, args):
        if arg.get('kind') != 'unresolved_argument':
            out.append(arg); continue
        raw = bytes.fromhex(arg['bytes_hex'])
        if kind == 'scalar':
            out.append({'kind': 'bytes', 'bytes_hex': arg['bytes_hex'], 'source_image_sha256': op['module'],
                        'frame_rule': 'unproved'})
        elif kind == 'device_pointer':
            address = int.from_bytes(raw, 'little')
            if address == 0:
                out.append({'kind': 'bytes', 'bytes_hex': arg['bytes_hex'], 'source_image_sha256': op['module'],
                            'frame_rule': 'unproved'})
                continue
            root = resolve(allocs, address, clock)
            if root is None:
                out.append(arg); left += 1
            else:
                out.append({'kind': 'device', 'target': {'allocation': root['id'], 'offset': address - root['recorded_base']},
                            'source_image_sha256': op['module']})
        else:
            out.append(arg); left += 1
    return out, left


def main():
    ap = argparse.ArgumentParser()
    for tag in ('a', 'b'):
        ap.add_argument(f'--plan-{tag}', required=True)
        ap.add_argument(f'--driver-{tag}', required=True)
        ap.add_argument(f'--slots-{tag}', required=True)
    ap.add_argument('--catalog-a', required=True)
    ap.add_argument('--library-a', required=True)
    ap.add_argument('--cindexer-pairs', required=True)
    ap.add_argument('--cublas-a', required=True)
    ap.add_argument('--cublas-library', required=True)
    ap.add_argument('--cublas-library-sha256', required=True)
    ap.add_argument('--copy-plans-a', required=True)
    ap.add_argument('--copy-plans-b', required=True)
    ap.add_argument('--ruc-pointers-a', required=True)
    ap.add_argument('--tempo-pointers-a', required=True)
    ap.add_argument('--asarray-a', required=True, help='host-asarray observation (upload source sites)')
    ap.add_argument('--payloads-a', required=True, help='recorder payload directory')
    ap.add_argument('--immutable-candidates', required=True, help='the step<=1 retention list (sha256 bytes per line)')
    ap.add_argument('--d2h-checked', default=None,
                    help='classify_d2h_checked.py output: copies whose bytes the seal proved (checked_value)')
    ap.add_argument('--layout-source', action='append', default=[],
                    help='CUDA source (.cu or .cu.gz) of a module loaded from raw bytes with no observed source '
                         '(P1: hexcore mpasdiag/_artifact.cuda_source(), the isobaric diagnostics)')
    ap.add_argument('--images-a', default=None, help='recorder image directory of recording A (layout-source check)')
    ap.add_argument('--hexcore-src', default=None, help='hexcore source root: source-literal line text check')
    ap.add_argument('--out', required=True)
    ap.add_argument('--residual', required=True)
    a = ap.parse_args()
    d2h_checked = {}
    if a.d2h_checked:
        for e in json.load(open(a.d2h_checked))['checked']:
            d2h_checked[e['sequence']] = e
    ra = Recording(a.plan_a, a.driver_a, a.slots_a)
    rb = Recording(a.plan_b, a.driver_b, a.slots_b)
    residual = collections.defaultdict(list)
    receipt = collections.Counter()
    # ---- P1 sealer-output-steps: kernels of a module with no observed source (the isobaric diagnostics, loaded by
    # cp.cuda.function.Module().load(bytes) at the output steps only, so the source observer never saw them) are
    # typed from the pinned source when the image's kernel-name set equals the source's and every launch's
    # parameter count and widths equal the parsed signature. Recording B's words stay raw (the R2 verdict reads bytes).
    if a.layout_source:
        from source_signatures import image_signatures
        layout = {}
        for path in a.layout_source:
            raw = Path(path).read_bytes()
            if path.endswith('.gz'):
                raw = gzip.decompress(raw)
            sigs = image_signatures(raw)
            for name, sig in sigs.items():
                layout[name] = (sig, hashlib.sha256(raw).hexdigest(), set(sigs))
        allocs_early = load(a.catalog_a)['allocations']
        checked_images = {}
        for name, ops in ra.plan['blocks'].items():
            for op in ops:
                if op.get('kind') != 'kernel' or not any(x.get('kind') == 'unresolved_argument' for x in op['arguments']):
                    continue
                entry = layout.get(op['function'])
                if entry is None:
                    continue
                if op['module'] not in checked_images:
                    image = gzip.decompress((Path(a.images_a) / f"{op['module']}.gz").read_bytes())
                    checked_images[op['module']] = elf_kernel_names(image) == entry[2]
                if not checked_images[op['module']]:
                    residual['layout_source_names_differ'].append([name, op['function'], op['module']])
                    continue
                typed = layout_type_arguments(op, entry[0], allocs_early, ra.clock.get(op.get('recorded_sequence'), 0))
                if typed is None:
                    residual['layout_source_signature_differs'].append([name, op['function'], op['module']])
                    continue
                op['arguments'], left = typed
                op['layout_source_sha256'] = entry[1]
                receipt['layout_source_typed_launches'] += 1
                receipt['layout_source_unresolved_left'] += left
        receipt['layout_source_images_checked'] = sum(checked_images.values())
    # rule sites in the shipped source, found by text (SITE_TEXTS); without --hexcore-src sites are taken as recorded
    site_alias, site_verified = shipped_sites(a.hexcore_src) if a.hexcore_src else ({}, None)
    ba, bb = ra.annotate(), rb.annotate()
    plan = ra.plan

    # ---- held writes from the tracer
    held = collections.defaultdict(list)  # step -> [(clock, op, record)]
    hex_names = {s.encode().hex(): s for s in HOLD_SLOTS}
    catalog = load(a.catalog_a)
    allocs = catalog['allocations']
    stacks = []  # (sequence, frames) of device-to-host Python sites (BINDER_D2H_STACKS)
    hostbytes = []  # (sequence, sha256, bytes) of every host upload source, for copy-plan matching
    upload_records, upload_copies = [], {}  # HostUploadBytes records; {MemcpyHtoD sequence: (host address, bytes)}
    by_base = collections.defaultdict(list)
    for x in allocs:
        by_base[x['recorded_base']].append(x)
    held_groups = []  # one per held slot call: [step, slot, last clock, [records]]
    # G2 (P1 milestone 2): the Thrust argsort of the LW wrapper (rrtmg_lw_v842.py:1018). Its temporary storage came
    # from CuPy's pool inside a live allocation; the plan replays it there, so the original library writes the same
    # bytes at the same place as the ordinary forward. The base is the lowest device address the call's own
    # launches use outside its input and index arrays (the radix-sort temporaries), from recording A's raw launches.
    thrust_seqs = {q for prop in load(a.library_a)['proposals'] if prop['operation']['kind'] == 'native_thrust_argsort'
                   for q in prop['recorded_sequences']}
    thrust_params = {}
    open_group = None
    for r in jsonl(a.driver_a):
        t = r.get('type')
        if t == 'LaunchKernel' and r.get('sequence') in thrust_seqs:
            thrust_params[r['sequence']] = (r['clock_ns'], [int.from_bytes(bytes.fromhex(x['bytes_hex']), 'little')
                                                            for x in r.get('parameters', []) if x.get('size') == 8])
        if t == 'HostUploadBytes':
            hostbytes.append((r['sequence'], r['sha256'], r['bytes']))
            upload_records.append(r)
            continue
        if t and t.startswith('MemcpyHtoD'):
            upload_copies[r['sequence']] = (r.get('b'), r.get('count'))
        if t == 'PyStack':
            stacks.append((r['sequence'], r.get('frames', ''), r.get('gil')))
            continue
        if t == 'HeldSlotError':
            raise SystemExit(f'tracer reported a held-slot capture error: {r}')
        if t == 'HostSlot':
            open_group = None
            continue
        if t != 'HeldSlotWrite':
            continue
        slot = hex_names.get(r['operation_hex'])
        if slot is None:
            raise SystemExit(f'held write from an unlisted slot {r}')
        if open_group is None:
            open_group = [r['step'], slot, r['clock_ns'], []]
            held_groups.append(open_group)
        open_group[2] = r['clock_ns']
        open_group[3].append(r)
    # Each slot call's ranges are packed into one blob (recording/held/<sha>.gz) and one held_upload_ranges op.
    held_dir = Path(a.out).parent / (Path(a.out).name.split('.')[0] + '-held')
    held_dir.mkdir(exist_ok=True)
    # P1 (ad/phys): one held slot call per thread (payload reads, zlib and sha256 release the GIL); results are
    # taken in held_groups order, so the plan is the same as the serial loop's. Breakage: on the 18-step hrrrv5
    # pair (a recording host, 2026-11-04) this loop ran on one core for about 10 minutes (455 s of zlib.compress alone in a
    # 1,079 s profile of the 6-step seal) while the card waited.
    def pack(group):
        step, slot, clock, records = group
        blob, ranges, whole, unresolved = bytearray(), [], [], []
        for r in records:
            hits = [x for x in by_base.get(r['base'], [])
                    if x['allocated_clock_ns'] <= r['clock_ns']
                    and (x.get('freed_clock_ns') is None or r['clock_ns'] < x['freed_clock_ns'])]
            if len(hits) != 1:
                unresolved.append([step, slot, r['base'], r['offset'], r['bytes']])
                continue
            payload = gzip.decompress((Path(a.payloads_a) / f"{r['sha256']}.gz").read_bytes())
            if len(payload) != r['bytes'] or hashlib.sha256(payload).hexdigest() != r['sha256']:
                raise SystemExit(f'held payload mismatch {r}')
            ranges.append({'destination': {'allocation': hits[0]['id'], 'offset': r['offset']},
                           'source_offset': len(blob), 'bytes': r['bytes']})
            if r.get('new_allocation') and r['offset'] == 0 and r['bytes'] == hits[0]['bytes']:
                whole.append(hits[0]['id'])
            blob += payload
        if not ranges:
            return None, 0, ranges, whole, unresolved
        sha = hashlib.sha256(bytes(blob)).hexdigest()
        (held_dir / f'{sha}.gz').write_bytes(gzip.compress(bytes(blob), 1))
        return sha, len(blob), ranges, whole, unresolved
    import os
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        packed = list(pool.map(pack, held_groups))
    for (step, slot, clock, records), (sha, nbytes, ranges, whole, unresolved) in zip(held_groups, packed):
        residual['held_write_unresolved_allocation'].extend(unresolved)
        if not ranges:
            continue
        op = {'kind': 'held_upload_ranges', 'stream': 0,
              'data': {'encoding': 'gzip', 'path': f'{held_dir.name}/{sha}.gz', 'sha256': sha, 'bytes': nbytes},
              'ranges': ranges, 'origin': HOLD_LABEL[HELD[slot]]}
        held[step].append((clock, op, whole))
        receipt[f'held_slot_calls_{HELD[slot]}'] += 1
        receipt[f'held_ranges_{HELD[slot]}'] += len(ranges)
        receipt[f'held_bytes_{HELD[slot]}'] += nbytes

    stacks.sort()
    stack_seqs = [s[0] for s in stacks]

    # ---- G2: bytes each device-to-host copy returned in recordings A and B (recorder BINDER_D2H_PAYLOADS=1:
    # a DeviceToHostBytes record right after the copy's own records, payload content-addressed)
    def d2h_bytes_index(driver):
        import shutil
        import subprocess
        rows = []
        if shutil.which('pigz'):
            proc = subprocess.Popen(['pigz', '-dc', str(driver)], stdout=subprocess.PIPE, bufsize=1 << 22)
            stream = proc.stdout
        else:
            proc, stream = None, gzip.open(driver, 'rb')
        for line in stream:
            if b'"DeviceToHostBytes"' in line:
                d = json.loads(line)
                rows.append((d['sequence'], d['sha256'], d['bytes']))
        if proc is not None:
            proc.wait()
        rows.sort()
        return rows
    d2h_rows_a, d2h_rows_b = d2h_bytes_index(a.driver_a), d2h_bytes_index(a.driver_b)
    payloads_b = Path(a.driver_b).parent / 'payloads'
    d2h_index_a = {n: [op['recorded_sequence'] for _, op in ops if op['kind'] == 'device_to_host'] for n, ops in ba.items()}
    d2h_index_b = {n: [op['recorded_sequence'] for _, op in ops if op['kind'] == 'device_to_host'] for n, ops in bb.items()}

    def d2h_payload(rows, root, seq, nbytes):
        import bisect as _b
        k = _b.bisect_right(rows, (seq, '', 0))
        if k >= len(rows) or rows[k][0] - seq > 6 or rows[k][2] != nbytes:
            return None
        path = Path(root) / f'{rows[k][1]}.gz'
        if not path.is_file():
            return None
        data = gzip.decompress(path.read_bytes())
        return data if len(data) == nbytes and hashlib.sha256(data).hexdigest() == rows[k][1] else None

    def recorded_d2h(name, seq, op):
        if not d2h_rows_a or not d2h_rows_b or (op.get('bytes') or 0) > (1 << 16):
            return None
        ia, ib = d2h_index_a.get(name, []), d2h_index_b.get(name, [])
        if seq not in ia or len(ia) != len(ib):
            return None
        sb = ib[ia.index(seq)]
        da = d2h_payload(d2h_rows_a, a.payloads_a, seq, op['bytes'])
        db = d2h_payload(d2h_rows_b, payloads_b, sb, op['bytes'])
        if da is None or db is None or da != db:
            residual['device_to_host_recordings_differ' if da is not None and db is not None else
                     'device_to_host_bytes_missing'].append([name, seq, op['bytes']])
            return None
        return da.hex()

    # ---- R2 verdicts on live kernels
    def live(blocks):
        return {n: [op for slot, op in ops if op['kind'] == 'kernel' and slot not in HELD] for n, ops in blocks.items()}
    ka, kb = live(ba), live(bb)
    mismatch = {n for n in ka if [k['function'] for k in ka[n]] != [k['function'] for k in kb.get(n, [])]}
    groups = collections.defaultdict(list)
    for n in ka:
        groups[tuple(k['function'] for k in ka[n])].append(n)
    verdict, table_a, table_b = {}, [], []
    for names in groups.values():
        for i, k0 in enumerate(ka[names[0]]):
            for j, arg in enumerate(k0['arguments']):
                if arg.get('kind') not in ('bytes', 'patched'):
                    continue
                wa = {masked(ka[n][i]['arguments'][j]) for n in names}
                if any(n in mismatch for n in names):
                    v, wb = 'single_recording', set()
                else:
                    wb = {masked(kb[n][i]['arguments'][j]) for n in names}
                    v = 'varies_within' if (len(wa) > 1 or len(wb) > 1) else ('differs_between' if wa != wb else 'constant_both')
                row = len(table_a) + 1
                key = f"{','.join(names)}|{i}|{k0['function']}|{j}"
                table_a.append(f'{key}|{sorted(wa)}')
                table_b.append(f'{key}|{sorted(wb)}')
                if v == 'varies_within' and all(masked(ka[n][i]['arguments'][j]) == masked(kb[n][i]['arguments'][j]) for n in names):
                    v = 'varies_within_equal_per_step'
                for n in names:
                    verdict[(n, i, j)] = (v, row, wa | wb)
    text_a, text_b = '\n'.join(table_a) + '\n', '\n'.join(table_b) + '\n'
    sha_a = hashlib.sha256(text_a.encode()).hexdigest()
    sha_b = hashlib.sha256(text_b.encode()).hexdigest()
    table_name = Path(a.out).name.split('.')[0] + '.r2-word-table-a.txt'
    (Path(a.out).parent / table_name).write_text(text_a)
    (Path(a.out).parent / table_name.replace('-a.txt', '-b.txt')).write_text(text_b)

    # ---- proved CIndexer source/image pairs (prove_cindexer_pairs.py; carray.rs copy_indexer_proved)
    proved_pairs = {(r['function'], r['rank'], r['image_sha256']): r
                    for r in load(a.cindexer_pairs)['pairs'] if r['set_before_get']}

    # ---- library replacements (stage 1 rule)
    proposals = load(a.library_a)['proposals']
    replaced = {}
    # G2 (P1 milestone 2): memsets the original Thrust argsort issues between its own launches (radix-sort
    # histogram clears) belong to the library call the native operation replays whole. THE BREAKAGE THIS PREVENTS:
    # kept as plan clears after the native call, they zeroed the library's final temporaries (forward words 2562,
    # replay 0) and allocation 392 failed the step-12 hash of the radiation-live plan.
    thrust_spans = [(min(prop['recorded_sequences']), max(prop['recorded_sequences'])) for prop in proposals
                    if prop['operation']['kind'] == 'native_thrust_argsort']
    for prop in proposals:
        seqs = prop['recorded_sequences']
        replaced[seqs[0]] = prop
        for s in seqs[1:]:
            replaced[s] = None
    dims = []

    def dim_slice(value):
        dims.append(int(value))
        return {'allocation': 900001, 'offset': 8 * (len(dims) - 1)}

    # ---- RRFS glue copy plans (glue_plan.py:155-176; copy_plan.rs). The table bytes are rebuilt live from
    # current allocations; each row's shape, strides and fill are host words that must be identical, in
    # observation order per (step, name), in the second recording (R2 applied to these host words).
    hostbytes.sort()
    host_seqs = [h[0] for h in hostbytes]
    upload_records.sort(key=lambda r: r['sequence'])
    upload_record_seqs = [r['sequence'] for r in upload_records]

    def plan_obs(path):
        rows = collections.defaultdict(list)
        for r in jsonl(path):
            rows[(r['step'], r['name'])].append(r)
        return rows
    obs_a, obs_b = plan_obs(a.copy_plans_a), plan_obs(a.copy_plans_b)

    def stripped(r):
        return [[i['dst_stride'], i['src_stride'], i['shape'], i['fill'], i['mode']] for i in r['inputs']]
    by_sha = {}
    for key, rows in obs_a.items():
        other = obs_b.get(key, [])
        for n, r in enumerate(rows):
            same = n < len(other) and stripped(other[n]) == stripped(r)
            by_sha.setdefault((r['step'], r['sha256'], r['bytes']), []).append((r, same))
    copy_cache_ids = {}
    cache_writes = []

    def copy_plan_upload(step, seq, destination):
        import bisect
        i = bisect.bisect_left(host_seqs, seq) - 1
        if i < 0 or seq - host_seqs[i] >= 8:
            return None, 'no host bytes record'
        _, sha, nbytes = hostbytes[i]
        found = by_sha.get((step, sha, nbytes), [])
        if not found:
            return None, 'not a copy plan'
        r, same = found.pop(0)
        if not same:
            return None, 'copy-plan host words differ from the second recording'
        rows = []
        for item in r['inputs']:
            dst = resolve(allocs, item['dst'], r['clock_ns'])
            if dst is None:
                return None, 'copy-plan destination unresolved'
            row = {'dst': {'allocation': dst['id'], 'offset': item['dst'] - dst['recorded_base']},
                   'dst_stride': [dim_slice(x) for x in item['dst_stride']],
                   'shape': [dim_slice(x) for x in item['shape']]}
            if item['mode'] == 0:
                src = resolve(allocs, item['src'], r['clock_ns'])
                if src is None:
                    return None, 'copy-plan source unresolved'
                row.update({'src': {'allocation': src['id'], 'offset': item['src'] - src['recorded_base']},
                            'src_stride': [dim_slice(x) for x in item['src_stride']], 'fill': None})
            else:
                row.update({'src': None, 'src_stride': None, 'fill': dim_slice(item['fill'])})
            rows.append(row)
        key = (r['name'], len(r['inputs']))
        cache = copy_cache_ids.setdefault(key, 900100 + len(copy_cache_ids))
        # The forward's CopyPlan._grid/_tiled after this build (glue_plan.py:163-185, the source formula),
        # recorded so stage 3 can write the carried cache owner's per-step reference.
        largest, tiled = 0, False
        for item in r['inputs']:
            shp, ds, ss = item['shape'], item['dst_stride'], item['src_stride']
            order = sorted(range(4), key=lambda k: (shp[k] > 1, -abs(ds[k])))
            shp, ds, ss = [shp[k] for k in order], [ds[k] for k in order], [ss[k] for k in order]
            count = shp[0] * shp[1] * shp[2] * shp[3]
            largest = max(largest, count)
            tiled |= (item['mode'] == 0 and count >= 1048576 and shp[0] == 1 and shp[1] == 1
                      and ds[3] == 1 and ss[2] == 1 and shp[2] > 1 and shp[3] > 1)
        grid_x = max(1, min(1024, -(-largest // 256)))
        cache_writes.append({'step': step, 'cache': cache, 'grid': [grid_x, len(r['inputs'])], 'tiled': tiled,
                             'table': destination, 'rows': len(r['inputs'])})
        return {'kind': 'rrfs_copy_plan_upload', 'stream': 0, 'destination': destination, 'rows': rows,
                'cache': {'allocation': cache, 'offset': 0}}, None

    # ---- RUC boundary tables (resident_ruc_rrfs.py:127; ruc.rs pointer_table) and TEMPO pointer arrays
    # (tempo_v303.py:4548-4574; tempo.rs). Each observed call carries the uploaded bytes' SHA-256, so the
    # upload is matched exactly; the views are resolved to plan allocations at the call's clock.
    def numel(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    def observed_target(view, clock):
        if view is None:
            return None
        root = resolve(allocs, view['base'] + view['offset'], clock)
        if root is None:
            raise LookupError('view unresolved')
        return {'allocation': root['id'], 'offset': view['base'] + view['offset'] - root['recorded_base']}

    def c_contiguous(view):
        item, expect = int(view['dtype'][-1]), int(view['dtype'][-1])
        for extent, stride in reversed(list(zip(view['shape'], view['strides']))):
            if extent > 1 and stride != expect:
                return False
            expect *= extent
        return True

    pointer_obs = {}
    for r in jsonl(a.ruc_pointers_a):
        pointer_obs.setdefault((r['step'], r['sha256'], r['bytes']), []).append(('ruc', r))
    for r in jsonl(a.tempo_pointers_a):
        if r.get('success'):
            pointer_obs.setdefault((r['step'], r['sha256'], r['bytes']), []).append(('tempo', r))

    def ruc_view(view, clock, kind):
        extent = numel(view['shape']) * int(view['dtype'][-1])
        return {'target': observed_target(view, clock) if extent else None, 'bytes': extent,
                'dtype': kind, 'contiguous': c_contiguous(view),
                'last_dimension': view['shape'][-1] if view['shape'] else None}

    def pointer_upload(step, seq, destination):
        import bisect
        i = bisect.bisect_left(host_seqs, seq) - 1
        if i < 0 or seq - host_seqs[i] >= 8:
            return None, 'no host bytes record'
        _, sha, nbytes = hostbytes[i]
        found = pointer_obs.get((step, sha, nbytes), [])
        if not found:
            return None, 'not a copy plan, RUC or TEMPO table'
        kind, r = found.pop(0)
        try:
            if kind == 'ruc':
                clock = r['clock_ns']
                return {'kind': 'ruc_pointer_table_upload', 'stream': 0, 'destination': destination, 'entry': r['entry'],
                        'arguments': {f['name']: ruc_view(f['view'], clock, 'i32' if f['type'] == 'int' else 'f32')
                                      for f in r['fields']},
                        'flags': {f['name']: ruc_view(f['view'], clock, 'i32') for f in r['flags']},
                        'tile_stop': ruc_view(r['tile_stop'], clock, 'i32') if r['tile_stop'] else None}, None
            clock = r['enter_ns']
            entries = [None if v is None else {'target': observed_target(v, clock) if numel(v['shape']) else None,
                                               'dtype': v['dtype'], 'shape': v['shape'], 'strides': v['strides'],
                                               'device': r['device']} for v in r['inputs']]
            return {'kind': 'tempo_pointer_table_bind', 'stream': 0, 'binding': f'tempo_s{step}_q{seq}',
                    'destination': destination, 'entries': entries, 'expected_shape': r.get('shape'),
                    'device': r['device'], 'optional': r['optional']}, None
        except LookupError as e:
            return None, f'{kind} {e}'

    immutable_shas = {line.split()[0] for line in open(a.immutable_candidates) if line.strip() and not line.startswith('#')}

    # ---- small pointer tables retained by BINDER_RETAIN_SMALL_UPLOADS: every 8-byte word must resolve to a
    # live allocation at the upload clock, and the upload's Python site must be a known table builder.
    # multirank (L20): same-host rank tables: (row width in words, address word positions within a row).
    # partition_samehost_transport_v841.py:632 the fused halo pack/unpack descriptor (live, packet, indices, rows, count,
    # itemsize, ndim, dims, unit); :576 a field's (shape, stride) pairs; cuda_solve_region_v841.py:443 the solve-region
    # rows (pointer, count, stride, solve, flags). THE BREAKAGE THIS PREVENTS: 576 uploads of a 2-rank recording had no
    # rule (stage-2 residual upload_needs_live_rule), so no decomposed plan could seal.
    MIXED_TABLE_RULES = {'partition_samehost_transport_v841.py:632': (9, (0, 1, 2, 7)),
                         'partition_samehost_transport_v841.py:576': (2, ()),
                         'cuda_solve_region_v841.py:443': (5, (0,))}
    TABLE_RULES = {'cuda_physics_prep_v841.py:585': 'cuda_physics_prep_device_pointer_table'}
    # The RRFS stage-sources rate table (driver_rrfs.py: the pointers of the tendency arrays physics_get_tend hands the
    # dycore) is found by file and source text, not by line number. THE BREAKAGE THIS PREVENTS: the rule named line
    # 2657; the file grew by 99 lines (the upload is at 2756 on this tree), so a door recording from --init at 120 km
    # refused with "pointer table site without a rule: mpasphys/rrfs/driver_rrfs.py:2756" (upload_needs_live_rule 1,
    # box-d jobs d120-d and d120-f, 2026-11-11).
    TABLE_TEXT_RULES = {'mpasphys/rrfs/driver_rrfs.py': (
        'table = cp.asarray(np.array([array.data.ptr for array in table_arrays], dtype=np.uint64))',
        'rrfs_stage_sources_rate_table')}

    # The same-host transport's tables are found by source text too, not only by line number. THE BREAKAGE THIS
    # PREVENTS: the rules named lines 632 and 576 of partition_samehost_transport_v841.py; the file moved (the fused
    # descriptor is built at 606 and 644, the (shape, stride) pairs at 588 on this tree), so a 4-rank recording of the
    # 103,950-column zoom globe refused with 4,372 upload_needs_live_rule, 1,872 of them "pointer table site without a
    # rule: partition_samehost_transport_v841.py:644" (18 Nov 2026).
    MIXED_TABLE_TEXT_RULES = {'partition_samehost_transport_v841.py': (
        (('asarray(', 'host_descriptor'), (9, (0, 1, 2, 7))),
        (('asarray(', 'zip(array.shape, array.strides)'), (2, ())))}

    def mixed_rule(site):
        named = MIXED_TABLE_RULES.get(site)
        if named is not None or not site or not a.hexcore_src:
            return named
        rel, _, line = site.rpartition(':')
        rules = MIXED_TABLE_TEXT_RULES.get(rel)
        if not rules or not line.isdigit():
            return None
        try:
            found = (Path(a.hexcore_src) / rel).read_text().splitlines()[int(line) - 1]
        except (OSError, IndexError):
            return None
        return next((shape for words, shape in rules if all(w in found for w in words)), None)

    def table_rule(site):
        named = TABLE_RULES.get(site)
        if named is not None or not site or not a.hexcore_src:
            return named
        rel, _, line = site.rpartition(':')
        text, named = TABLE_TEXT_RULES.get(rel, (None, None))
        if text is None or not line.isdigit():
            return None
        try:
            found = (Path(a.hexcore_src) / rel).read_text().splitlines()[int(line) - 1]
        except (OSError, IndexError):
            return None
        return named if text in found else None
    asarray = sorted((r['enter_ns'], r['exit_ns'], r['stack'][0]['path'].split('/hexcore/')[-1] + ':' + str(r['stack'][0]['line']))
                     for r in jsonl(a.asarray_a) if r.get('stack'))

    def site_at(clock):
        hits = [x for x in asarray if x[0] <= clock <= x[1]]
        site = min(hits, key=lambda x: x[1] - x[0])[2] if hits else None
        return site_alias.get(site, site)

    # ---- (3 km storm box, 2026-11-07): the limited-area garbage-column scrub descriptor tables.
    # woof/hex/cuda_solve_region_v841.py:443 SolveRegionKernels.table() uploads, for every scrub_columns() launch whose
    # rows are not in its device cache, an int64 table of rows (pointer, rows, stride, solve, pool float32 bits):
    # solve_region_scrub_columns_v841 then writes pool into element r*stride + solve of every array. Word 0 is a device
    # address (resolved to a live allocation at the upload clock, rebuilt from the replay's own binding), words 1-4 are
    # literal (counts, element stride, the garbage column, the pool value) and are replayed as recorded. THE BREAKAGE
    # THIS PREVENTS: 26,968 of these uploads (about 450 a step) landed in upload_needs_live_rule ("pointer table site
    # without a rule"), and every 8-byte word had to be a live address for the plain pointer-table rule, which these
    # rows are not. Refused unless the payload is whole 40-byte rows, every pointer is live and the row's last element
    # (pointer + 4*((rows-1)*stride + solve)) stays inside that allocation.
    SCRUB_SITE = 'cuda_solve_region_v841.py:443'

    def scrub_descriptor_upload(op, clock, raw):
        if len(raw) % 40:
            return None, 'scrub descriptor table is not whole 40-byte rows'
        rows = []
        for k in range(0, len(raw), 40):
            w = [int.from_bytes(raw[k + 8 * j:k + 8 * j + 8], 'little') for j in range(5)]
            root = resolve(allocs, w[0], clock)
            if root is None:
                return None, f'scrub descriptor row {k // 40}: pointer is not a live device address'
            n, stride, solve = (x if x < 1 << 63 else x - (1 << 64) for x in w[1:4])
            if n <= 0 or stride < 0 or solve < 0:
                return None, f'scrub descriptor row {k // 40}: rows {n} stride {stride} solve {solve}'
            offset = w[0] - root['recorded_base']
            if offset + 4 * ((n - 1) * stride + solve) + 4 > root['bytes'] or offset % 4:
                return None, f'scrub descriptor row {k // 40}: last element outside allocation {root["id"]}'
            rows.append({'pointer': {'allocation': root['id'], 'offset': offset}, 'words': w[1:]})
        return {'kind': 'descriptor_table_upload', 'stream': 0, 'destination': op['destination'],
                'rule': 'solve_region_scrub_descriptors', 'rows': rows}, None

    hostbytes_seqs = [h[0] for h in hostbytes]
    upload_records.sort(key=lambda r: r['sequence'])
    upload_record_seqs = [r['sequence'] for r in upload_records]

    def kiss_jump_upload(op, clock, seq):
        """An upload at the jump-table site whose recorded SHA-256 is that of a recomputed MCICA jump
        table (see KISS_JUMP_SITE) -> immutable input at any step, or None."""
        import bisect
        if site_at(clock) != KISS_JUMP_SITE:
            return None
        if site_verified is not None and KISS_JUMP_SITE not in site_verified:
            return None
        data = op.get('data')
        if data:
            sha, nbytes = data.get('sha256'), data.get('bytes')
        else:   # not retained: the tracer's hash record of this copy's host bytes is the nearest one before the copy
            k = bisect.bisect_left(hostbytes_seqs, seq) - 1
            if k < 0 or seq - hostbytes[k][0] > 8:
                return None
            sha, nbytes = hostbytes[k][1], hostbytes[k][2]
        draws = kiss_jump_table_draws(sha, nbytes)
        if draws is None:
            return None
        path = Path(a.payloads_a) / f'{sha}.gz'
        if not path.is_file():
            raw = kiss_jump_table_bytes(draws, nbytes // 128)
            assert hashlib.sha256(raw).hexdigest() == sha
            tmp = path.with_name(f'.{sha}.sealer.gz')
            tmp.write_bytes(gzip.compress(raw, 1))
            tmp.replace(path)
        return {'kind': 'upload', 'stream': 0, 'destination': op['destination'],
                'data': {'bytes': nbytes, 'encoding': 'gzip', 'path': f'recording/payloads/{sha}.gz', 'sha256': sha},
                'pointers': [], 'origin': 'immutable_input',
                'proof': {'rule': 'kiss_xorshift_jump_table', 'site': KISS_JUMP_SITE, 'draws': draws,
                          'gpoints': nbytes // 128,
                          'payload': 'retained by the recorder' if data else 'recomputed by the sealer (same SHA-256)'}}

    def table_upload(op, clock):
        data = op.get('data') or {}
        path = Path(a.payloads_a) / Path(data.get('path', '')).name
        if not data or not path.is_file():
            return None, 'no retained payload'
        raw = gzip.decompress(path.read_bytes())
        if len(raw) % 8 or not raw:
            return None, 'payload is not a table of 8-byte words'
        site = site_at(clock)
        if site == SCRUB_SITE:
            return scrub_descriptor_upload(op, clock, raw)
        mixed = mixed_rule(site)
        if mixed is not None:
            # multirank (L20): a same-host table that mixes device addresses (at fixed word positions of fixed-width
            # rows) with integer words (counts, strides, item sizes, flags). The address words become relocation patches
            # of an ordinary upload; every other word is kept as recorded and must not be a live device address.
            width, positions = mixed
            if len(raw) % (8 * width):
                return None, f'{site}: {len(raw)} bytes is not rows of {width} words'
            patches = []
            for k in range(0, len(raw), 8):
                word = int.from_bytes(raw[k:k + 8], 'little')
                root = resolve(allocs, word, clock) if word else None
                if (k // 8) % width in positions:
                    if root is None:
                        return None, f'{site}: address word {k // 8} is not a live device address'
                    patches.append({'byte_offset': k, 'target': {'allocation': root['id'], 'offset': word - root['recorded_base']}})
                elif root is not None and word > (1 << 40):
                    return None, f'{site}: integer word {k // 8} is a live device address'
            return {'kind': 'upload', 'stream': 0, 'destination': op['destination'], 'data': op['data'],
                    'pointers': patches, 'origin': f'samehost table: {site}'}, None
        rule = table_rule(site)
        if rule is None and site is None and op.get('recorded_sequence') in concat_tables:
            rule = 'cupy_stack_same_size'
        if rule is None:
            return None, f'pointer table site without a rule: {site}'
        entries = []
        for k in range(0, len(raw), 8):
            word = int.from_bytes(raw[k:k + 8], 'little')
            root = resolve(allocs, word, clock)
            if root is None:
                return None, f'word {k // 8} is not a live device address'
            entries.append({'allocation': root['id'], 'offset': word - root['recorded_base']})
        return {'kind': 'pointer_table_upload', 'stream': 0, 'destination': op['destination'], 'rule': rule,
                'entries': entries}, None

    # ---- the cuBLAS transposes (audit_cublas_geam.py): C = A^T with alpha 1, beta 0, host scalars,
    # transa = transb = 1, lda = n, ldc = m; native_cublas.rs issues the same original cublasSgeam.
    cublas = {}
    for call in load(a.cublas_a)['calls']:
        e = call['entry']
        if not (e['transa'] == 1 and e['transb'] == 1 and e['alpha_bits'] == 1065353216 and e['beta_bits'] == 0
                and e['host_scalars'] and e['pointer_mode'] == 0 and e['lda'] == e['n'] and e['ldc'] == e['m']
                and e['a'] == e['b'] and call['exit']['status'] == 0):
            residual['cublas_call_shape_unsupported'].append(call['entry']['id'])
            continue
        seqs = [l['sequence'] for l in call['launches']]
        cublas[seqs[0]] = call
        for q in seqs[1:]:
            cublas[q] = None

    blocks = {}
    # P1 (ad/phys): CuPy concatenate/stack pointer tables. cupy.concatenate of same-size arrays uploads the
    # inputs' data pointers ([a.data.ptr for a in arrays], uint64, in order) to a small device buffer and launches
    # cupy_concatenate_same_size on it; there is no hexcore Python frame at the upload (site None). An upload is
    # this table only when the next kernel in its block, within 8 ops, is cupy_concatenate_same_size with a
    # device or CArray-pointer argument at the upload destination. Every word must still resolve to a live allocation
    # (table_upload), so the runner rebuilds the same table from current bindings (rule cupy_stack_same_size).
    concat_tables = set()
    for _name, _ops in ba.items():
        for _i, (_slot, _op) in enumerate(_ops):
            if _op['kind'] != 'upload' or not _op.get('data'):
                continue
            for _slot2, _op2 in _ops[_i + 1:_i + 9]:
                if _op2['kind'] != 'kernel':
                    continue
                if _op2.get('function') == 'cupy_concatenate_same_size' and any(
                        (_arg.get('kind') == 'device' and _arg.get('target') == _op['destination'])
                        or any(_ptr.get('target') == _op['destination'] for _ptr in _arg.get('pointers', []))
                        for _arg in _op2.get('arguments', [])):
                    concat_tables.add(_op.get('recorded_sequence'))
                break
    # ---- P1 (ad/phys): the recorder front end's own reads (record_observer.py: TEMPO cost/order/hist copied to the
    # host after each outer boundary, OBSERVER/tempo-sched/step-N.json with pointer and bytes) are not forward
    # operations: read-only device-to-host copies the recorder issued, at the start of step N+1. They are left out
    # of the plan, each matched exactly (block step N+1, recorded source address == the row's pointer, bytes ==
    # the row's bytes). Breakage: the 6-step seal of the 18-step hrrrv5 pair (a recording host, 2026-11-04) refused with 15
    # device_to_host_consumer residuals (3 per step, steps 2-6, allocations 414 and 153), all of them these reads.
    observer_reads = set()
    sched = Path(a.slots_a).parent / 'tempo-sched'
    if sched.is_dir():
        base_of = {x['id']: x['recorded_base'] for x in allocs}
        reads = collections.defaultdict(set)
        for path in sched.glob('step-*.json'):
            for row in json.load(open(path))['rows']:
                if 'hex' in row:
                    reads[int(path.stem.split('-')[1]) + 1].add((row['pointer'], row['bytes']))
        for name, ops in ba.items():
            want = reads.get(int(name.split('_')[1]), set())
            for slot, op in ops:
                src = op.get('source') if op['kind'] == 'device_to_host' else None
                if src and src.get('allocation') in base_of and                         (base_of[src['allocation']] + src.get('offset', 0), op.get('bytes')) in want:
                    observer_reads.add(op.get('recorded_sequence'))
    for name, ops in ba.items():
        step = int(name.split('_')[1])
        out, k_live = [], -1
        k_concat_count = None
        clocks = []
        for slot, op in ops:
            seq = op.get('recorded_sequence')
            clock = ra.clock.get(seq, 0)
            kind = op['kind']
            if slot in HELD and kind == 'synchronization':
                # Event and stream ordering touch no memory; keeping them keeps event lifetimes paired
                # for the live slots that share the same recorded handles.
                out.append(op); clocks.append(clock); receipt['held_synchronization_kept'] += 1
                continue
            if slot in HELD:
                if kind == 'allocate':
                    # A pool allocation made inside a held slot and still live at its end is written whole by the
                    # held upload (HeldSlotWrite new_allocation=1, offset 0, every byte), so it is overwritten
                    # before any later read. Without such a record it stays a poison-proof residual.
                    whole = any(op['id'] in w for _, _, w in held.get(step, []))
                    op = dict(op)
                    if whole:
                        op['overwritten_before_read'] = True
                        receipt['held_allocation_written_whole'] += 1
                    else:
                        residual['held_allocation_not_written_whole'].append([name, slot, op['id'], op['bytes']])
                    out.append(op); clocks.append(clock)
                else:
                    receipt[f'held_removed_{HELD[slot]}_{kind}'] += 1
                continue
            if kind == 'clear' and any(lo < (seq or 0) < hi for lo, hi in thrust_spans):
                receipt['thrust_internal_memset_removed'] += 1
                continue
            if kind == 'kernel':
                k_live += 1
                if seq in replaced:
                    prop = replaced[seq]
                    if prop is not None:
                        operation = dict(prop['operation'])
                        geom = prop['observed_geometry_for_check_only']
                        for key in ('count', 'segments', 'segment_size'):
                            if key in operation and operation[key] is None:
                                value = geom.get(key, geom.get('shape', [None])[0] if key == 'count' else None)
                                if value is None:
                                    residual['library_dimension_unknown'].append([name, seq, key])
                                else:
                                    operation[key] = dim_slice(value)
                        if operation['kind'] == 'native_thrust_argsort' and operation.get('workspace') is None:
                            n = geom.get('shape', [0])[0]
                            busy = [(operation['input'], n * (4 if operation['dtype'] == 4 else 8)), (operation['indices'], n * 8)]
                            found = []
                            for q in prop['recorded_sequences']:
                                if q not in thrust_params:
                                    continue
                                c0, words = thrust_params[q]
                                for w in words:
                                    hit = resolve(allocs, w, c0)
                                    if hit is None:
                                        continue
                                    off = w - hit['recorded_base']
                                    if any(hit['id'] == b['allocation'] and b['offset'] <= off < b['offset'] + nb for b, nb in busy):
                                        continue
                                    found.append((w, hit['id'], off))
                            if found:
                                w, ident, off = min(found)
                                operation['workspace'] = {'allocation': ident, 'offset': off}
                                receipt['thrust_workspace_from_forward_launches'] += 1
                            else:
                                residual['thrust_workspace_unknown'].append([name, seq])
                        out.append(operation); clocks.append(clock)
                        receipt['library_operations'] += 1
                    continue
                if seq in cublas:
                    call = cublas[seq]
                    if call is not None:
                        e = call['entry']
                        out.append({'kind': 'native_cublas_transpose', 'stream': 0,
                                    'library_path': a.cublas_library, 'library_sha256': a.cublas_library_sha256,
                                    'input': call['bindings']['a']['root'], 'output': call['bindings']['c']['root'],
                                    'm': dim_slice(e['m']), 'n': dim_slice(e['n'])})
                        clocks.append(clock)
                        receipt['cublas_transpose_operations'] += 1
                    continue
                args = []
                for j, arg in enumerate(op['arguments']):
                    k = arg.get('kind')
                    if k == 'device':
                        args.append({'kind': 'device', 'target': arg['target']})
                    elif k == 'module_global':
                        args.append({x: arg[x] for x in ('kind', 'module', 'symbol', 'offset', 'expected_bytes')})
                    elif k in ('bytes', 'patched'):
                        v, row, words = verdict[(name, k_live, j)]
                        fn = op['function']
                        base = {'kind': k, 'bytes_hex': arg['bytes_hex']}
                        if k == 'patched':
                            base['pointers'] = arg['pointers']
                        if v == 'constant_both':
                            base['fixed_source'] = {'class': 'recorded_constant_two_recordings',
                                                    'source': table_name, 'source_sha256': sha_a, 'line': row}
                            args.append(base); receipt['r2_constant_words'] += 1
                        elif (fn, j) in STEP_WORDS and int.from_bytes(bytes.fromhex(arg['bytes_hex']), 'little') == step:
                            args.append({'kind': 'step_integer', 'scale': 1, 'bias': 0, 'width': 4}); receipt['live_step_integer'] += 1
                        elif (fn, j) == RADII_WORD:
                            args.append({'kind': 'physics_frame', 'name': 'tempo_radii_due_i32'})
                            residual['radii_recorded_for_check_only'].append([name, arg['bytes_hex']])
                            receipt['live_tempo_radii'] += 1
                        else:
                            ok, ndim = cindexer_scratch_only({masked(arg)} | words, len(arg['bytes_hex']) // 2)
                            if (not ok and v in ('varies_within', 'varies_within_equal_per_step')
                                    and k_live < len(kb.get(name, []))):
                                # G2 (P1 milestone 2): a CIndexer whose size/shape change from step to step (LW/SW group
                                # sizes) but are the same in both recordings at this launch; only the index scratch
                                # differs. Its shape words go to the host table per launch, as for any copy indexer.
                                # THE BREAKAGE THIS PREVENTS: the whole-group head check sent 32 such words to
                                # frame_rule_needed although each launch's head was proved by both recordings.
                                ok2, nd2 = cindexer_scratch_only({masked(arg), masked(kb[name][k_live]['arguments'][j])},
                                                                 len(arg['bytes_hex']) // 2)
                                if ok2 and nd2 and nd2 >= 1:
                                    ok, ndim = True, nd2
                                    receipt['cindexer_per_step_head_two_recordings'] += 1
                            pair = proved_pairs.get((fn, ndim, op['module'])) if ok else None
                            if pair is not None:
                                raw = bytes.fromhex(arg['bytes_hex'])
                                shape = [dim_slice(int.from_bytes(raw[8 + 8 * d:16 + 8 * d], 'little', signed=True))
                                         for d in range(ndim)]
                                args.append({'kind': 'cupy_copy_indexer', 'shape': shape,
                                             'source': {'class': ('cupy_reduction_in_indexer_set_before_get'
                                                                  if pair.get('kind') == 'reduction_in'
                                                                  else 'cupy_copy_indexer_set_before_get'),
                                                        'source': f"recording/sources/{pair['source_sha256']}.cu",
                                                        'source_sha256': pair['source_sha256'], 'line': pair['line']}})
                                receipt['cindexer_copy_indexer_rule'] += 1
                            elif ok:
                                base['fixed_source'] = {'class': 'cupy_indexer_scratch_pending_proof',
                                                        'source': 'cupy/_core/include/cupy/carray.cuh', 'source_sha256': '0' * 64, 'line': 525}
                                args.append(base); receipt['cindexer_scratch_words'] += 1
                                residual['cindexer_scratch_proof'].append([name, fn, j, ndim, op['module']])
                            elif fn == 'cupy_concatenate_same_size' and concat_count_word(arg, k_concat_count):
                                # P1 (ad/phys): the count of arrays CuPy concatenates (deferred-check flags, batched by
                                # host timing, so 12 in one ordinary run and 13 in another at the same step) is written
                                # into the CArray size/shape words. It equals the entry count of the pointer table this
                                # launch reads, which is itself a live table; the runner re-checks that equality.
                                base['fixed_source'] = {'class': 'cupy_concat_count_from_table', 'source': table_name,
                                                        'source_sha256': sha_a, 'line': row}
                                args.append(base); receipt['cupy_concat_count_from_table'] += 1
                            elif v == 'varies_within_equal_per_step':
                                # G2 (P1 milestone 2): a launch word that changes from step to step (the LW/SW group
                                # sizes of the CuPy gathers/scatters/copies, the ozone and solar time words, the
                                # cloudiness step words) and is byte-identical between the two independent recordings
                                # at every step: a function of the trajectory, not of the run. Carried per step as
                                # recorded; the replay is held to it by the checked host reads that fix these shapes
                                # (LW histogram, SW sunlit count and time words) and by every boundary hash. The
                                # source-formula rule for each family (binder-phys-full item 2) is the named follow-up.
                                # THE BREAKAGE THIS PREVENTS: with radiation live, 2,112 such words had no rule and the
                                # seal refused (frame_rule_needed).
                                base['fixed_source'] = {'class': 'recorded_per_step_two_recordings', 'source': table_name,
                                                        'source_sha256': sha_a, 'line': row}
                                args.append(base); receipt['r2_per_step_words'] += 1
                                receipt['r2_per_step_words:' + fn] += 1
                            else:
                                args.append(base); residual['frame_rule_needed'].append([name, fn, j, v, sorted(words)[:4]])
                    else:
                        args.append(arg); residual['unresolved_argument'].append([name, slot, op['function'], j, k])
                out.append({'kind': 'kernel', 'stream': 0, 'module': op['module'], 'function': op['function'],
                            'grid': op['grid'], 'block': op['block'], 'shared_bytes': op['shared_bytes'], 'arguments': args})
                clocks.append(clock)
                continue
            why = None
            if kind == 'upload' and slot in RESET_ZERO_SLOTS:
                # rule: a reset re-pack with no payload in the recording (not retained, or retained and
                # then thinned away) is admitted by its recorded SHA-256 alone; see RESET_ZERO_SLOTS.
                data = op.get('data')
                if not (data and (Path(a.payloads_a) / Path(data.get('path', '')).name).is_file()):
                    sha, nbytes = ((data.get('sha256'), data.get('bytes')) if data else
                                   upload_hash_record(seq, upload_copies, upload_records, upload_record_seqs))
                    lit = reset_zero_sha_rule(op, site_at(clock), slot, sha, nbytes)
                    if lit is not None:
                        out.append(lit); clocks.append(clock)
                        receipt['output_step_reset_diagnostics_zero'] += 1
                        receipt['reset_diagnostics_zero_from_sha256'] += 1
                        continue
            if kind == 'upload' and op.get('data'):
                path = Path(a.payloads_a) / Path(op['data'].get('path', '')).name
                lit = literal_upload_rule(op, site_at(clock), slot,
                                          gzip.decompress(path.read_bytes()) if path.is_file() else None, site_verified)
                if lit is not None:
                    out.append(lit); clocks.append(clock); receipt['output_step_' + lit['proof']['rule']] += 1
                    continue
                live, why = table_upload(op, clock)
                if live is not None:
                    out.append(live); clocks.append(clock); receipt[live['kind']] += 1
                    if live.get('rule') == 'cupy_stack_same_size':
                        k_concat_count = len(live['entries'])
                    continue
            if kind == 'upload':
                if op.get('data') and step == 1 and op['data'].get('sha256') in immutable_shas:
                    out.append({'kind': 'upload', 'stream': 0, 'destination': op['destination'], 'data': op['data'],
                                'pointers': op.get('pointers', []), 'origin': 'immutable_input'})
                elif (op.get('data') and op['data'].get('sha256') in immutable_shas
                        and resident_output_initial(site_at(clock), a.hexcore_src)):
                    out.append({'kind': 'upload', 'stream': 0, 'destination': op['destination'], 'data': op['data'],
                                'pointers': op.get('pointers', []), 'origin': 'immutable_input',
                                'proof': {'rule': 'resident_output_initial_words', 'site': site_at(clock)}})
                    receipt['resident_output_initial_upload'] += 1
                elif (op.get('data') and op['data'].get('sha256') in immutable_shas and not op.get('pointers')
                        and resident_output_initial(site_at(clock), a.hexcore_src, LBC_FILE_VALUES)):
                    out.append({'kind': 'upload', 'stream': 0, 'destination': op['destination'], 'data': op['data'],
                                'pointers': [], 'origin': 'immutable_input',
                                'proof': {'rule': 'lateral_boundary_file_values', 'site': site_at(clock)}})
                    receipt['lateral_boundary_file_upload'] += 1
                elif op.get('data'):
                    # Retained only because it is small (BINDER_RETAIN_SMALL_UPLOADS) and no pointer-table site rule
                    # matched. P1 (ad/phys): a retained payload is still a copy plan, RUC or TEMPO table upload when
                    # the observers say so; the v26 recording (no retention) classified these uploads that way, and
                    # without this fall-through every one of them landed in the residual.
                    live, why2 = copy_plan_upload(step, seq, op['destination'])
                    if live is None and why2 == 'not a copy plan':
                        live, why2 = pointer_upload(step, seq, op['destination'])
                    if live is None:
                        live = kiss_jump_upload(op, clock, seq)
                    if live is not None:
                        out.append(live); receipt[live['kind'] if 'proof' not in live else 'upload_' + live['proof']['rule']] += 1
                    else:
                        residual['upload_needs_live_rule'].append([name, slot, seq, op.get('destination'), f'{why}; {why2}'])
                        out.append(op)
                else:
                    live, why = copy_plan_upload(step, seq, op['destination'])
                    if live is None and why == 'not a copy plan':
                        live, why = pointer_upload(step, seq, op['destination'])
                    if live is None:
                        live = kiss_jump_upload(op, clock, seq)
                    if live is not None:
                        out.append(live); receipt[live['kind'] if 'proof' not in live else 'upload_' + live['proof']['rule']] += 1
                    else:
                        residual['upload_needs_live_rule'].append([name, slot, seq, op.get('destination'), why])
                        out.append(op)
            elif kind == 'device_to_host' and seq in observer_reads:
                receipt['removed_recorder_observer_read_device_to_host'] += 1
                continue
            elif kind == 'device_to_host' and seq in d2h_checked:
                e = d2h_checked[seq]
                if e.get('observer_read'):  # G2: the recorder observer's own read, not a forward operation
                    receipt['removed_recorder_observer_read_device_to_host:' + e['rule']] += 1
                    continue
                assert e['bytes'] == op['bytes'], e
                live = {k: op[k] for k in ('kind', 'stream', 'source', 'bytes', 'asynchronous', 'recorded_sequence')}
                if 'expected_zero_words' in e:
                    live.update(host_consumer='checked_zero_words', expected_zero_words=e['expected_zero_words'])
                else:
                    assert len(e['expected_hex']) == 2 * op['bytes'], e
                    live.update(host_consumer='checked_value', expected_hex=e['expected_hex'])
                live.update(proof=e['rule'], site=e['site'])
                if 'layer_groups' in e:  # G2: the LW batches the replay rebuilds with layer_groups.rs
                    live['layer_groups'] = e['layer_groups']
                out.append(live); receipt['checked_device_to_host:' + e['rule']] += 1
            elif kind == 'device_to_host':
                import bisect
                k = bisect.bisect_left(stack_seqs, seq) - 1
                site = stacks[k][1] if k >= 0 and seq - stacks[k][0] < 64 else ''
                live = history_output_rule(op, site, site_alias)
                got = None if live is not None else recorded_d2h(name, seq, op)
                if live is not None:
                    out.append(live); receipt['history_output_device_to_host'] += 1
                    receipt['history_output_bytes'] += op.get('bytes') or 0
                elif got is not None:
                    # G2 (P1 milestone 2): a host read whose value picks launch shapes or scalars of a live slot (SW
                    # time words rrtmg_sw_v842.py:354, sunlit count :612, deferred guards :641, RUC land-use flags
                    # ruc_lsm_8c9f1a8.py:6104). Both recordings returned the same bytes at the aligned copy (recorder
                    # BINDER_D2H_PAYLOADS=1), so the replay downloads the live value and refuses on any difference:
                    # the recorded launches that follow are then exactly the forward's. THE BREAKAGE THIS PREVENTS: with
                    # radiation and land live these reads had no expected value and the seal refused them.
                    live = {k: op[k] for k in ('kind', 'stream', 'source', 'bytes', 'asynchronous', 'recorded_sequence')}
                    live.update(host_consumer='checked_value', expected_hex=got, proof='recorded_two_recordings_equal', site=site)
                    out.append(live); receipt['checked_device_to_host:recorded_two_recordings_equal'] += 1
                    receipt['checked_device_to_host_site:' + site] += 1
                else:
                    residual['device_to_host_consumer'].append([name, slot, seq, op.get('bytes'), site])
                    out.append(op)
            elif kind == 'allocate':
                residual['allocation_poison_proof'].append([name, slot, op['id']])
                out.append(op)
            else:
                out.append(op)
            clocks.append(clock)
        # insert held uploads after the last kept op issued before the held write
        merged = list(zip(clocks, out))
        for clock, hop, _ in sorted(held.get(step, []), key=lambda x: x[0]):
            idx = 0
            for pos, (c, _) in enumerate(merged):
                if c and c <= clock:
                    idx = pos + 1
            merged.insert(idx, (clock, hop))
        blocks[name] = [op for _, op in merged]
    plan['blocks'] = blocks

    # ---- static overwrite-before-read for dynamic allocations still uncertified: in plan order, the first
    # operation that names the allocation (any 'allocation' field: kernel targets, copy rows, RUC/TEMPO
    # views, held uploads, library buffers) is a clear of every byte, or nothing ever names it.
    def names(x, ident):
        if isinstance(x, dict):
            if x.get('allocation') == ident:
                return True
            return any(names(v, ident) for v in x.values())
        if isinstance(x, list):
            return any(names(v, ident) for v in x)
        return False
    # step order, not text order (as text step_100 sorts before step_11; the same order below 100 steps)
    order = [(n, i) for n in sorted(blocks, key=block_key) for i in range(len(blocks[n]))]
    for pos, (n, i) in enumerate(order):
        op = blocks[n][i]
        if op['kind'] != 'allocate' or op.get('overwritten_before_read'):
            continue
        first = next(((m, j) for m, j in order[pos + 1:] if names(blocks[m][j], op['id'])), None)
        if first is None:
            op['overwritten_before_read'] = True
            receipt['allocation_never_named'] += 1
            continue
        f = blocks[first[0]][first[1]]
        if (f['kind'] == 'clear' and f['destination'] == {'allocation': op['id'], 'offset': 0}
                and f['bytes'] == op['bytes']):
            op['overwritten_before_read'] = True
            receipt['allocation_cleared_whole_first'] += 1
        else:
            residual['allocation_first_use_not_whole_write'].append([n, op['id'], op['bytes'], f['kind'], f.get('function')])
    residual.pop('allocation_poison_proof', None)
    residual.pop('held_allocation_not_written_whole', None)
    plan['complete'] = False
    plan['milestone'] = ('hrrrv5 milestone 2 (G2): radiation and land live' if MILESTONE == '2' else
                         'hrrrv5 milestone 1 (COMMON-AD 56.1, 103.3): radiation and land held')
    plan.setdefault('host_allocations', []).append({
        'id': 900001, 'bytes': 8 * max(1, len(dims)), 'origin': 'initial_state',
        'initial': {'encoding': 'hex', 'bytes': b''.join(d.to_bytes(8, 'little', signed=True) for d in dims).hex() or '00' * 8,
                    'sha256': hashlib.sha256(b''.join(d.to_bytes(8, 'little', signed=True) for d in dims) or bytes(8)).hexdigest()}})
    for cache in sorted(copy_cache_ids.values()):
        # copy_plan::CACHE_BYTES = 40; the live build writes it; it is a carried host owner (main.rs ~603)
        plan['host_allocations'].append({'id': cache, 'bytes': 40, 'origin': 'initial_state',
                                         'initial': {'encoding': 'hex', 'bytes': '00' * 40,
                                                     'sha256': hashlib.sha256(bytes(40)).hexdigest()}})
        plan.setdefault('host_carried', []).append(cache)
    receipt['copy_plan_cache_owners'] = len(copy_cache_ids)
    plan['_copy_plan_cache_writes'] = cache_writes  # consumed and removed by seal_phys_stage3.py
    with gzip.open(a.out, 'wt') as f:
        json.dump(plan, f, separators=(',', ':'))
    summary = {'schema': 'ad.binder.phys-seal-stage2/v1', 'complete': False,
               'recording_a': a.plan_a, 'recording_b': a.plan_b,
               'r2_word_table_sha256': {'a': sha_a, 'b': sha_b},
               'receipt_counts': dict(receipt),
               'rule_sites_found_by_text': {shipped: site for shipped, site in sorted(site_alias.items())
                                            if shipped != site},
               'launch_dimension_words': len(dims),
               'residual_counts': {k: len(v) for k, v in residual.items()},
               'device_to_host_by_site': dict(collections.Counter(
                   (x[4] or 'no-python-site').split(';')[0] for x in residual['device_to_host_consumer'])),
               'residual': residual}
    Path(a.residual).write_text(json.dumps(summary, indent=1, default=list))
    print(json.dumps({k: summary[k] for k in ('receipt_counts', 'residual_counts', 'device_to_host_by_site', 'r2_word_table_sha256')}, indent=1))


if __name__ == '__main__':
    main()
