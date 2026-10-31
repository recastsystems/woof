"""Bitwise round-trip tests for tilestream.gather.

Run directly (``python -m tilestream.test_gather``) or under pytest.

WHAT IS BEING PROVEN
--------------------
1. DEVICE ROUND TRIP -- fill a full-domain state with unique values, gather
   every tile, scatter every interior back into a ZEROED sink, and require the
   sink to equal the source bitwise, for every field in the persisted
   inventory.  Scattering into a zeroed sink (rather than back over the
   source) is deliberate: it makes an un-copied cell show up as a mismatch
   instead of being masked by the value that was already there.  A gather that
   silently moved nothing cannot pass.
2. HOST ROUND TRIP -- the same, with a PINNED host store on the domain side,
   exercising the real H2D/D2H path rather than D2D.
3. ALL 47 NAMES -- a synthetic store carrying every name in
   ``STATE_SERIALIZED_ATTRS`` at its true stagger, including the 2-D
   ``(ny, nx)`` surface fields (``mup``, ``nwfa2d``, ``nifa2d``) and the
   ``(nz+1, ny, nx)`` vertical-face fields (``w``, ``php``).  A dry
   milestone-one ``DomainState`` only allocates 9 of the 47, so the real-state
   tests above cannot cover the rest; this one does.  The count is pinned
   exactly and moves only together with a per-field justification -- see
   ``NEW_SINCE_41`` at the test.
4. STAGGERED PLACEMENT -- a gathered ``u`` tile is compared against the
   directly-indexed slice of the full ``u`` array, so an off-by-one in the
   x-face treatment fails loudly instead of hiding inside a round trip that is
   self-consistently wrong.

Both a ``cupy.array_equal`` comparison and a SHA-256 over the raw bytes are
required to pass.  ``array_equal`` alone would call NaN != NaN a failure and
+0.0 == -0.0 a success; the byte hash has neither problem, and disagreeing
answers from the two would itself be informative.

THE PERIODIC STAGGERED-DUPLICATE INVARIANT
------------------------------------------
"Fill with unique values" cannot be taken literally for ``u`` and ``v`` on a
periodic domain.  ``u`` has ``nx+1`` faces for ``nx`` mass columns and face
``nx`` IS face ``0``; likewise ``v`` row ``ny`` is row ``0``.  woof's own
seeding enforces exactly this (harness.py:_SEED_FIELDS marks ``u`` x-duplicate
and ``v`` y-duplicate, copied from npref.py:2284).  ``_fill_unique`` below
therefore assigns a distinct value to every element and then re-imposes the
duplicate, and ``test_duplicate_invariant_is_load_bearing`` demonstrates that
without it the round trip genuinely fails -- so the invariant is stated and
checked, not assumed.

TWO INDEPENDENT SPECS
---------------------
``tilestream/spec.py`` is owned by another agent.  It was not on disk when
these tests were written, so ``_LocalTileSpec`` / ``plan_tiles`` below are a
minimal stand-in implementing the agreed contract; they live HERE and are
never imported by ``gather.py``.  ``spec.py`` has since landed, and every
round trip now runs against BOTH -- the real module and the stand-in -- via
``SPEC_PROVIDERS``.  That is deliberate: the two disagree on details
(the stand-in expresses segments as ``(src, dst)`` pairs while
``spec.Transfer`` names its sides ``full``/``tile``; they pick different tiles
to own the periodic alias slot ``u[..., nx]``), so passing both shows
``gather.py`` is driven by the geometry rather than co-tuned to one
implementation's quirks.
"""

from __future__ import annotations

import hashlib
import sys
import traceback

import numpy as np

from tilestream import gather as G
from tilestream import harness


# ==========================================================================
# local stand-in for tilestream.spec (TEST FILE ONLY)
# ==========================================================================

class _LocalTileSpec:
    """Per-kind ``(src_slices, dst_slices)`` segment lists for one tile.

    Exposes ``spec.gather[kind]`` and ``spec.scatter[kind]``, one of the
    spellings ``gather.segments_for`` accepts.
    """

    def __init__(self, i0, j0, tile_nx, tile_ny, halo, nx, ny):
        self.i0, self.j0 = i0, j0
        self.tile_nx, self.tile_ny = tile_nx, tile_ny
        self.halo = halo
        self.nx, self.ny = nx, ny
        self.gather: dict[str, list] = {}
        self.scatter: dict[str, list] = {}

    def __repr__(self):
        return f"<TileSpec i0={self.i0} j0={self.j0} halo={self.halo}>"


def _axis_runs(start, length, period):
    """``[(src_start, n, dst_offset)]`` for ``(start + k) % period``."""
    if length > period:
        raise ValueError(f"gathered extent {length} exceeds period {period}")
    runs, pos, remaining, off = [], start % period, length, 0
    while remaining > 0:
        n = min(remaining, period - pos)
        runs.append((pos, n, off))
        remaining -= n
        off += n
        pos = 0
    return runs


def plan_tiles(nx, ny, tile_nx, tile_ny, halo=harness.HALO, periodic=True):
    """Tile the ``(ny, nx)`` mass grid into interior ``tile_ny x tile_nx`` tiles.

    ``tile_nx``/``tile_ny`` are the INTERIOR extents; a gathered tile is
    ``tile_nx + 2*halo`` by ``tile_ny + 2*halo`` mass columns.  Only the
    periodic case is implemented (milestone one).

    Staggered periodicity: ``u`` wraps with period ``nx`` even though the
    array is ``nx+1`` wide, because face ``nx`` and face ``0`` are the same
    face; ``v`` likewise with period ``ny``.  On scatter the tile whose
    interior ends at ``nx`` also writes the duplicate face ``nx`` (and
    similarly at ``ny`` for ``v``), so the full arrays are covered exactly
    once.
    """
    if not periodic:
        raise NotImplementedError("milestone one is periodic-only")
    if nx % tile_nx or ny % tile_ny:
        raise ValueError("stand-in requires the tiling to divide the domain")

    specs = []
    for j0 in range(0, ny, tile_ny):
        for i0 in range(0, nx, tile_nx):
            spec = _LocalTileSpec(i0, j0, tile_nx, tile_ny, halo, nx, ny)
            gx, gy = tile_nx + 2 * halo, tile_ny + 2 * halo
            for kind in G.KINDS:
                xlen = gx + (1 if kind == "u" else 0)
                ylen = gy + (1 if kind == "v" else 0)
                xruns = _axis_runs(i0 - halo, xlen, nx)
                yruns = _axis_runs(j0 - halo, ylen, ny)
                spec.gather[kind] = [
                    ((slice(sy, sy + hy), slice(sx, sx + hx)),
                     (slice(dy, dy + hy), slice(dx, dx + hx)))
                    for sy, hy, dy in yruns
                    for sx, hx, dx in xruns
                ]
                ex = 1 if (kind == "u" and i0 + tile_nx == nx) else 0
                ey = 1 if (kind == "v" and j0 + tile_ny == ny) else 0
                spec.scatter[kind] = [
                    ((slice(halo, halo + tile_ny + ey),
                      slice(halo, halo + tile_nx + ex)),
                     (slice(j0, j0 + tile_ny + ey),
                      slice(i0, i0 + tile_nx + ex)))
                ]
            specs.append(spec)
    return specs


def _vram_used_mib():
    """Total VRAM in use per ``nvidia-smi``, in MiB, or ``None``.

    Read ONCE at import, before this process has initialised CUDA, so it is
    a clean reading of what everybody ELSE is holding.  The per-process
    ``--query-compute-apps`` table would be the better source, but it comes
    back EMPTY on this WSL2 host even with 12 GB resident, so it cannot be
    used; and ``cudaMemGetInfo`` counts our own pool.  Sampling before we
    allocate is what is left, and it is exactly what the GPU-etiquette rule
    asks for anyway.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


#: VRAM held by other workloads when this module was imported.
BASELINE_VRAM_MIB = _vram_used_mib()


try:
    from tilestream import spec as real_spec
except ImportError:                                    # pragma: no cover
    real_spec = None

#: ``{label: plan_tiles}``.  Every round trip runs against each.
SPEC_PROVIDERS: dict[str, object] = {"stand-in": plan_tiles}
if real_spec is not None and hasattr(real_spec, "plan_tiles"):
    SPEC_PROVIDERS["tilestream.spec"] = real_spec.plan_tiles


# ==========================================================================
# helpers
# ==========================================================================

def _sync():
    """Fence default-stream setup against the non-blocking worker stream.

    Setup below (``cp.zeros``, ``arr[...] = cp.asarray(...)``) is issued on
    the CURRENT (default) stream, while every gather/scatter runs on a
    ``non_blocking=True`` stream -- which by definition does NOT synchronize
    with the default stream.  Without this fence a gather can legally read a
    buffer before its fill lands.  That is not hypothetical: it is what made
    ``test_wrapped_edge_tile_matches_rolled_reference`` fail inside the full
    suite and pass in isolation.  The same ordering obligation falls on the
    real pipeline -- ``gather_tile`` issues async copies and synchronizes
    nothing, so the caller must order them against other streams with an
    event.
    """
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()


def _fill_unique(arrays, *, enforce_duplicates=True):
    """Give every element a distinct float32 value, then re-impose periodicity.

    Values are drawn from a strictly increasing integer sequence cast to
    float32 and offset per field, so any two elements anywhere in the
    inventory differ -- a misplaced copy of the right SHAPE still fails.
    """
    import cupy as cp

    base = 1.0
    for name, arr in arrays.items():
        n = int(np.prod(arr.shape))
        vals = (base + np.arange(n, dtype=np.float64)).reshape(arr.shape)
        vals = vals.astype(np.float32)
        base += n + 7.0
        if enforce_duplicates:
            if arr.ndim >= 2 and name in ("u", "nudging_tendency_u"):
                vals[..., -1] = vals[..., 0]
            if arr.ndim >= 2 and name in ("v", "nudging_tendency_v"):
                vals[..., -1, :] = vals[..., 0, :]
        if isinstance(arr, cp.ndarray):
            arr[...] = cp.asarray(vals)
        else:
            arr[...] = vals
    _sync()


def _zero(arrays):
    for arr in arrays.values():
        arr[...] = 0
    _sync()


def _bytes_of(arr):
    import cupy as cp
    host = cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else np.asarray(arr)
    return np.ascontiguousarray(host).tobytes(order="C")


def _digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = arrays[name]
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(_bytes_of(arr))
    return h.hexdigest()


def _compare(ref, got, label):
    """Both cupy.array_equal AND a raw-byte SHA-256 must agree."""
    import cupy as cp

    bad = []
    for name in ref:
        a, b = ref[name], got[name]
        eq = bool(cp.array_equal(cp.asarray(a), cp.asarray(b)))
        same_bytes = _bytes_of(a) == _bytes_of(b)
        if not (eq and same_bytes):
            diff = int(np.count_nonzero(
                np.frombuffer(_bytes_of(a), np.uint8)
                != np.frombuffer(_bytes_of(b), np.uint8)))
            bad.append(f"{name}: array_equal={eq} bytes_equal={same_bytes} "
                       f"differing_bytes={diff}/{len(_bytes_of(a))}")
    dref, dgot = _digest(ref), _digest(got)
    if bad or dref != dgot:
        raise AssertionError(
            f"{label}: MISMATCH\n  " + "\n  ".join(bad)
            + f"\n  sha256 ref={dref}\n  sha256 got={dgot}")
    return dref


def _roundtrip(store_arrays, sink_arrays, tile_arrays, specs, stream=None):
    """Gather every tile then scatter its interior into ``sink_arrays``."""
    import cupy as cp

    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
    gathered_bytes = scattered_bytes = 0
    with stream:
        for spec in specs:
            gp = G.gather_tile(store_arrays, tile_arrays, spec, stream)
            sp = G.scatter_tile(tile_arrays, sink_arrays, spec, stream)
            gathered_bytes += gp.nbytes
            scattered_bytes += sp.nbytes
            stream.synchronize()   # tile buffer is reused; serialize the test
    stream.synchronize()
    cp.cuda.runtime.deviceSynchronize()
    return gathered_bytes, scattered_bytes


# ==========================================================================
# synthetic all-41 inventory
# ==========================================================================

def _synthetic_shapes(nz, ny, nx):
    """``{name: shape}`` for every one of the serialized attributes (47).

    Staggering follows the WRF-ARW registry: ``u`` on x faces, ``v`` on y
    faces, ``w``/``php`` on w levels, ``mup``/``nwfa2d``/``nifa2d`` 2-D at the
    surface, every microphysics/aerosol/TKE species on 3-D mass points.
    """
    two_d = {"mup", "nwfa2d", "nifa2d"}
    shapes = {}
    for name in G.serialized_attrs():
        if name in ("u", "nudging_tendency_u"):
            shapes[name] = (nz, ny, nx + 1)
        elif name in ("v", "nudging_tendency_v"):
            shapes[name] = (nz, ny + 1, nx)
        elif name in ("w", "php", "nudging_tendency_geopotential"):
            shapes[name] = (nz + 1, ny, nx)
        elif name in two_d:
            shapes[name] = (ny, nx)
        else:
            shapes[name] = (nz, ny, nx)
    return shapes


def _device_dict(shapes, dtype=np.float32):
    import cupy as cp
    out = {n: cp.zeros(s, dtype=dtype) for n, s in shapes.items()}
    _sync()                       # the zero-fill is on the default stream
    return out


def _pinned_dict(shapes, dtype=np.float32):
    return {n: G.pinned_empty(s, dtype) for n, s in shapes.items()}


# ==========================================================================
# tests
# ==========================================================================

NX = NY = 96
TILE = 32
HALO = harness.HALO           # 16 for the default time_step_sound=4
NZ = 49


def test_classify_derives_stagger_from_shape():
    nz, ny, nx = 49, 96, 96
    assert G.classify((nz, ny, nx), nz, ny, nx) == "mass"
    assert G.classify((ny, nx), nz, ny, nx) == "mass"          # 2-D surface
    assert G.classify((nz, ny, nx + 1), nz, ny, nx) == "u"
    assert G.classify((nz, ny + 1, nx), nz, ny, nx) == "v"
    assert G.classify((nz + 1, ny, nx), nz, ny, nx) == "w"
    for bad in ((nz, ny, nx + 2), (nz, ny + 2, nx), (nz + 2, ny, nx),
                (nz, ny + 1, nx + 1)):
        try:
            G.classify(bad, nz, ny, nx)
        except G.SpecError:
            pass
        else:
            raise AssertionError(f"classify accepted impossible shape {bad}")
    # non-square, so an nx/ny transposition cannot pass by accident
    assert G.classify((7, 40, 81), 7, 40, 80) == "u"
    assert G.classify((7, 41, 80), 7, 40, 80) == "v"
    print("  classify: mass/u/v/w/2-D derived from shape; 4 impossible "
          "shapes rejected")


def test_extents_inferred_from_inventory():
    shapes = _synthetic_shapes(6, 40, 80)
    arrays = _device_dict(shapes)
    assert G.domain_extents(arrays) == (6, 40, 80)
    # a dry inventory (9 fields, incl. 2-D mup and nz+1 php) infers the same
    dry = {k: arrays[k] for k in
           ("u", "v", "w", "thp", "php", "mup", "p", "al", "alt")}
    assert G.domain_extents(dry) == (6, 40, 80)
    print("  domain_extents: (6, 40, 80) from both the full 41 and the dry 9")


def test_layered_fields_gather_like_mass_fields():
    """Soil/snow fields -- leading axis 3, 4 or 7 -- must move bit-exactly.

    The physics carrier set contains ``fields/smois`` and friends at
    ``(4, ny, nx)``, Noah-MP's snow layers at ``(3, ny, nx)`` and its
    snow-soil coordinate at ``(7, ny, nx)``.  Their leading axis is NOT
    vertical, and two things used to happen: ``classify`` raised, and
    ``domain_extents`` silently answered ``nz = 3`` for a 49-level domain.
    This exercises both halves of the fix against a modulo-indexed reference,
    with a negative control on the guard that is still supposed to fire.
    """
    import cupy as cp

    from tilestream import spec as tspec

    nz, ny, nx, halo, tile = 49, 40, 32, 8, 16
    # classification, both modes
    for lead in (3, 4, 7):
        try:
            G.classify((lead, ny, nx), nz, ny, nx)
        except G.SpecError:
            pass
        else:
            raise AssertionError(
                f"classify accepted leading extent {lead} without layers_ok; "
                "the guard that catches a genuine shape bug is gone")
        assert G.classify((lead, ny, nx), nz, ny, nx, layers_ok=True) == "mass"

    layered = {"smois": (4, ny, nx), "zsnsoxy": (7, ny, nx),
               "tsnoxy": (3, ny, nx), "thp": (nz, ny, nx),
               "u": (nz, ny, nx + 1), "php": (nz + 1, ny, nx)}
    dom = {n: cp.asarray(np.arange(int(np.prod(s)), dtype=np.float32)
                         .reshape(s) * (1.0 + i))
           for i, (n, s) in enumerate(layered.items())}

    # domain_extents must REFUSE to guess, and must accept an explicit nz.
    try:
        G.domain_extents(dom)
    except G.SpecError as exc:
        assert "leading extents" in str(exc)
    else:
        raise AssertionError(
            "domain_extents inferred a vertical extent from an inventory "
            "with non-vertical leading axes; it used to answer nz=3")
    assert G.domain_extents(dom, nz=nz) == (nz, ny, nx)

    specs = tspec.plan_tiles(nx, ny, tile, tile, halo, True)
    tspec.validate_plan(specs, ny, nx)
    gy, gx = tile + 2 * halo, tile + 2 * halo
    tiles = {n: cp.zeros(G.tile_shape_for(
        G.classify(s, nz, ny, nx, layers_ok=True), s, gy, gx),
        dtype=cp.float32) for n, s in layered.items()}

    moved = 0
    for spec in specs:
        G.gather_tile(dom, tiles, spec, names=list(layered), nz=nz)
        cp.cuda.runtime.deviceSynchronize()
        for name, shape in layered.items():
            kind = G.classify(shape, nz, ny, nx, layers_ok=True)
            # Staggered periodicity: face nx IS face 0, so the wrap period is
            # the MASS extent even for an (nx+1)-wide array (spec.py's
            # _wrap_segments does the same, and getting it wrong here would
            # make the reference agree with a bug).
            ys = np.arange(spec.cj0, spec.cj0 + gy
                           + (1 if kind == "v" else 0)) % ny
            xs = np.arange(spec.ci0, spec.ci0 + gx
                           + (1 if kind == "u" else 0)) % nx
            want = cp.asnumpy(dom[name])[..., ys[:, None], xs[None, :]]
            got = cp.asnumpy(tiles[name])
            if not np.array_equal(want, got):
                raise AssertionError(
                    f"layered gather is wrong for {name} {shape} on tile "
                    f"{spec.index}: {int((want != got).sum())} elements differ")
            moved += got.size
    print(f"  layered (3/4/7-deep) fields gather bit-exactly through the mass "
          f"path: {len(specs)} tiles x {len(layered)} fields, "
          f"{moved} elements, all verified against a modulo-indexed reference; "
          f"classify still rejects them without layers_ok and domain_extents "
          f"still refuses to guess nz")


def test_rank4_carrier_gathers_and_scatters_every_plane():
    """A (k, nz, ny, nx) carrier must move all k * nz planes, both ways.

    Grell 3D carries ``fields/g3_lev`` at (9, nz, ny, nx).  The copy depth
    used to be ``shape[0]``, so a gather moved its first 9 (ny, nx) planes
    and a scatter wrote back only those: the streamed store's g3_lev
    differed from the resident one in all 9 slots after one step.  Gather
    against a modulo-indexed reference, then scatter every tile's interior
    into a zeroed sink that must come back equal to the source.
    """
    import cupy as cp

    from tilestream import spec as tspec

    nz, ny, nx, halo, tile = 6, 40, 32, 8, 16
    shapes = {"g3_lev": (9, nz, ny, nx), "thp": (nz, ny, nx), "mup": (ny, nx)}
    dom = {n: cp.asarray(np.arange(int(np.prod(s)), dtype=np.float32)
                         .reshape(s) + 1.0) for n, s in shapes.items()}
    sink = {n: cp.zeros(s, dtype=cp.float32) for n, s in shapes.items()}
    specs = tspec.plan_tiles(nx, ny, tile, tile, halo, True)
    tspec.validate_plan(specs, ny, nx)
    gy, gx = tile + 2 * halo, tile + 2 * halo
    tiles = {n: cp.zeros(G.tile_shape_for(
        G.classify(s, nz, ny, nx, layers_ok=True), s, gy, gx),
        dtype=cp.float32) for n, s in shapes.items()}
    assert tiles["g3_lev"].shape == (9, nz, gy, gx)
    for spec in specs:
        G.gather_tile(dom, tiles, spec, names=list(shapes), nz=nz)
        cp.cuda.runtime.deviceSynchronize()
        ys = np.arange(spec.cj0, spec.cj0 + gy) % ny
        xs = np.arange(spec.ci0, spec.ci0 + gx) % nx
        for name in shapes:
            want = cp.asnumpy(dom[name])[..., ys[:, None], xs[None, :]]
            got = cp.asnumpy(tiles[name])
            assert np.array_equal(want, got), (
                f"gather of {name} {shapes[name]} on tile {spec.index}: "
                f"{int((want != got).sum())} elements differ")
        G.scatter_tile(tiles, sink, spec, names=list(shapes), nz=nz)
        cp.cuda.runtime.deviceSynchronize()
    for name in shapes:
        assert np.array_equal(cp.asnumpy(dom[name]), cp.asnumpy(sink[name])), (
            f"scatter of {name} {shapes[name]} left "
            f"{int((cp.asnumpy(dom[name]) != cp.asnumpy(sink[name])).sum())} "
            "elements unwritten")


def test_device_roundtrip_real_state():
    """Full-domain DomainState -> tiles -> zeroed DomainState, bitwise."""
    cfg = harness.make_config(NX, NY, NZ)
    src = harness.make_state(cfg)
    sink = harness.make_state(cfg)
    gx, gy = TILE + 2 * HALO, TILE + 2 * HALO
    tile = harness.make_state(harness.tile_config(cfg, gx, gy))

    src_arrays = G.inventory(src)
    sink_arrays = G.inventory(sink)
    tile_arrays = G.inventory(tile)
    assert set(src_arrays) == set(sink_arrays) == set(tile_arrays)
    total = sum(int(a.nbytes) for a in src_arrays.values())
    digests = {}

    for label, planner in SPEC_PROVIDERS.items():
        _fill_unique(src_arrays)
        _zero(sink_arrays)
        ref = {n: a.copy() for n, a in src_arrays.items()}
        specs = planner(NX, NY, TILE, TILE, halo=HALO, periodic=True)
        gb, sb = _roundtrip(src_arrays, sink_arrays, tile_arrays, specs)
        digests[label] = _compare(ref, sink_arrays, f"device round trip [{label}]")
        _compare(ref, src_arrays, f"source was mutated [{label}]")
        assert sb == total, (
            f"[{label}] scatter moved {sb} B but the domain is {total} B -- "
            "the interiors do not tile the domain exactly")
        print(f"  [{label}] {len(specs)} tiles, {len(src_arrays)} fields, "
              f"domain {total/1e6:.2f} MB, gathered {gb/1e6:.2f} MB, "
              f"scattered {sb/1e6:.2f} MB (== domain)")
        print(f"  [{label}] sha256 {digests[label]}")
    assert len(set(digests.values())) == 1, digests
    print(f"  {len(digests)} spec implementation(s) agree bit-for-bit")


def test_host_roundtrip_pinned():
    """Pinned host store -> device tiles -> pinned host sink, bitwise."""
    cfg = harness.make_config(NX, NY, NZ)
    state = harness.make_state(cfg)
    gx, gy = TILE + 2 * HALO, TILE + 2 * HALO
    tile = harness.make_state(harness.tile_config(cfg, gx, gy))

    store = G.make_host_store(state, copy=False)
    sink = G.make_host_store(state, copy=False)
    store_arrays = store.as_dict()
    sink_arrays = sink.as_dict()
    tile_arrays = G.inventory(tile)
    for name, arr in store_arrays.items():
        assert G.is_pinned(arr), f"{name} is not pinned"
        assert not G.is_device_array(arr), f"{name} should be host memory"
    for name, arr in tile_arrays.items():
        assert G.is_device_array(arr), f"tile {name} should be device memory"

    digests = {}
    for label, planner in SPEC_PROVIDERS.items():
        _fill_unique(store_arrays)
        _zero(sink_arrays)
        ref = {n: a.copy() for n, a in store_arrays.items()}
        specs = planner(NX, NY, TILE, TILE, halo=HALO, periodic=True)
        gb, sb = _roundtrip(store_arrays, sink_arrays, tile_arrays, specs)
        digests[label] = _compare(ref, sink_arrays, f"host round trip [{label}]")
        print(f"  [{label}] pinned store {store.nbytes()/1e6:.2f} MB, "
              f"{len(specs)} tiles, gathered {gb/1e6:.2f} MB (H2D), "
              f"scattered {sb/1e6:.2f} MB (D2H)")
        print(f"  [{label}] sha256 {digests[label]}")
    assert len(set(digests.values())) == 1, digests


#: The names that joined ``STATE_SERIALIZED_ATTRS`` after this suite was
#: written against its 41.  Each is a restart-stream transcription from WRF's
#: own registry (the contract's comments carry the chapter and verse), and
#: each is a field a big-domain streamed run would silently LOSE on resume if
#: the transport never learned it:
#:
#:   ``qir``/``qib``      mp_physics=50 (P3) rime mass and rime volume --
#:                        rho_rime = qirim/birim picks the fall-speed table,
#:                        so dropping them resumes with different ice physics
#:                        (commit 252e9c6bf).
#:   ``th_old``/``qv_old`` P3's cross-step supersaturation carriers; re-zeroed
#:                        they replay the first-step transient mid-trajectory
#:                        (same commit).
#:   ``nh``               mp_physics=9 (Milbrandt-Yau) hail number -- a zero
#:                        moment under nonzero hail mass on resume
#:                        (commit 6b0cb580f).
#:   ``nn``               mp_physics=16 (WDM6) CCN concentration, same class,
#:                        same commit.
#:
#: All six are plain 3-D mass-point fields, so their transport is the mass
#: path the round trip below drives -- but "same path" is an argument, and
#: the assertion beneath is the measurement.
#:
#:   ``rthften``          the dycore's exported advective theta forcing --
#:   ``rqvften``          and its qv twin -- joined the contract at
#:                        02ca83316 (the GF forcing-pair export made them
#:                        checkpoint state) without this gate moving, the
#:                        exact drift the 41-to-47 note above records.
#:                        Both are plain 3-D mass-point fields; the round
#:                        trip below is the measurement that the default
#:                        stagger is their stagger.
#:   ``volg``             TEMPO's (mp_physics = "tempo") graupel volume
#:                        moment, the scheme's discriminator: graupel density
#:                        is qg / volg, so a resume that dropped it would
#:                        restart graupel at a different density.  It joined
#:                        the contract with the RRFS v2 hex + TEMPO preset; a
#:                        plain 3-D mass-point field like qvolg.
#:   ``qt``, ``f_ice``,   Ferrier-Aligo (mp_physics = 5, 2.9.0): WRF's
#:   ``f_rain``,          fer_mp_hires package state (Registry.EM_COMMON
#:   ``f_rimef``          :3019, restart stream).  F_RIMEF decides the
#:                        trajectory: FER_HIRES keeps it for every ice cell
#:                        between calls, so a streamed resume that re-set it
#:                        to 1 would restore unrimed ice.  Plain 3-D
#:                        mass-point fields.
#:   ``qke_adv``          MYNN's advected TKE carrier (bl_mynn_tkeadvect,
#:                        admitted in 2.9.0), a plain 3-D mass-point field.
#: All five joined the contract in the 2.9.0 round-2 merge; the gate moved
#: with them (192 -> 197) once this file reached the GPU shard there.
#:   ``tke_edmf``         the CCPP sa-TKE-EDMF PBL's advected TKE (2.9.1,
#:                        bl_pbl_physics = 901), a plain 3-D mass-point field
#:                        carried so a streamed run does not cold-start it
#:                        (197 -> 198).
#:   ``p_perturbation``   the WRF exact chain's strict-diagnostics carrier
#:                        (2.9.1 intake): WRF's perturbation-pressure words,
#:                        which pressure force and history read across stage
#:                        boundaries; rebuilt from rounded full pressure they
#:                        lose their low bits.  Allocated only under the strict
#:                        diagnostics build (None otherwise, so ordinary
#:                        inventories do not carry it).  A plain 3-D
#:                        mass-point field, p's own stagger (198 -> 199).
#:   ``nudging_tendency_{u,v,temperature,geopotential,qv}``  analysis
#:                        nudging's held FDDA tendencies (labs/nudging,
#:                        2.9.11): they persist between nudging calculation
#:                        times, so a resume that dropped them would nudge
#:                        with zeros until the next calculation.  Attached
#:                        only when nudging holds them (none otherwise), each
#:                        the stagger of its target: u and v their faces,
#:                        temperature and qv mass points, geopotential the
#:                        w levels (199 -> 204).
NEW_SINCE_41 = ("qir", "qib", "th_old", "qv_old", "nh", "nn",
                "rthften", "rqvften", "volg",
                "qt", "f_ice", "f_rain", "f_rimef", "qke_adv", "tke_edmf",
                "p_perturbation",
                "nudging_tendency_u", "nudging_tendency_v",
                "nudging_tendency_temperature", "nudging_tendency_geopotential",
                "nudging_tendency_qv")

#: FAST_SBM (mp_physics = 30) spectral bins: ff1/ff5/ff6 at 33 bins each and
#: ff8 at 43, 142 names (``woof.core.fast_sbm_contract.BIN_SPECIES``).  They
#: joined the contract at 624bb29c4 ("fast-sbm: carry the spectral fields in
#: native state and scalar transport", merged into 2.8.9 at b2b60eacb)
#: without this gate moving, which is what turned the 2.8.9 cut gate red at
#: a1388eae7 (192 names, gate knew 50).  WRF 4.6.1 Registry/registry.sbm
#: declares all 142 with the restart flag (ff1i01 at line 3: IO "h3rusdf")
#: and its line 318 package switches them on under mp_physics == 30, so
#: they are in the restart stream: bin state is prognostic and cannot be
#: rebuilt from the bulk mixing ratios, so a streamed resume that dropped
#: them would restart the spectral scheme from empty bins.  Each is a 3-D
#: mass-point view ``fast_sbm_bins[k]`` of one (nbin, nz, ny, nx) array
#: (``DomainState`` mp30 arm), so the default stagger in _synthetic_shapes
#: is their stagger; the round trip below is the measurement.  Spelled out
#: here, not imported, so a change to the bin table moves this gate on
#: purpose rather than silently.
FSBM_BINS = tuple(f"{group}i{i:02d}"
                  for group, count in (("ff1", 33), ("ff5", 33),
                                       ("ff6", 33), ("ff8", 43))
                  for i in range(1, count + 1))


def test_all_49_names_device_and_host():
    """Every name in STATE_SERIALIZED_ATTRS, both memory paths, both specs.

    The count is asserted EXACTLY, not as a floor: this suite synthesizes a
    shape for every contract name, so a name this file's stagger table does
    not know would otherwise ride through as 3-D mass by default.  When the
    contract grows, this number moves ON PURPOSE, together with a named
    justification above -- 41 became 47 when the P3 restart fix and the
    mp9/WDM6 moment fix landed their six fields without this gate moving,
    which left the streamed-serialization round trip formally proven for a
    set the product no longer serialized.
    """
    names = G.serialized_attrs()
    assert len(names) == 204, (
        f"STATE_SERIALIZED_ATTRS has {len(names)} names, this gate knows 204 "
        "(62 bulk + 142 FAST_SBM bins). "
        "A new field joined (or left) the streamed serialization set: extend "
        "NEW_SINCE_41's justification block and this count TOGETHER, and "
        "check the new name's stagger against _synthetic_shapes")
    assert len(FSBM_BINS) == 142
    for name in NEW_SINCE_41 + FSBM_BINS:
        assert name in names, (
            f"{name} left STATE_SERIALIZED_ATTRS; its restart consequence is "
            "documented above -- removing it needs an argument, not a tidy-up")
    nz, ny, nx, tile, halo = 6, 48, 48, 16, 8
    shapes = _synthetic_shapes(nz, ny, nx)
    assert set(shapes) == set(names)
    gx, gy = tile + 2 * halo, tile + 2 * halo
    tile_shapes = _synthetic_shapes(nz, gy, gx)

    results = {}
    for spec_label, planner in SPEC_PROVIDERS.items():
        specs = planner(nx, ny, tile, tile, halo=halo, periodic=True)
        for mem_label, mk_store in (("device", _device_dict),
                                    ("pinned host", _pinned_dict)):
            store = mk_store(shapes)
            sink = mk_store(shapes)
            tile_arrays = _device_dict(tile_shapes)
            _fill_unique(store)
            _zero(sink)
            ref = {n: (a.copy() if hasattr(a, "copy") else np.array(a))
                   for n, a in store.items()}
            gb, sb = _roundtrip(store, sink, tile_arrays, specs)
            key = f"{spec_label}/{mem_label}"
            results[key] = _compare(ref, sink, f"all-47 round trip [{key}]")
            total = sum(int(a.nbytes) for a in ref.values())
            assert sb == total, (key, sb, total)
            # The six post-41 joiners moved on THIS path, not by argument:
            # each was filled, gathered, scattered into a zeroed sink and
            # byte-compared -- the compare above walks all 47, and this pins
            # that the six were among them in both directions.
            for name in NEW_SINCE_41 + FSBM_BINS:
                assert name in ref and name in sink, name
                assert _bytes_of(sink[name]) == _bytes_of(ref[name]), name
            print(f"  [{key}] {len(names)}/{len(names)} fields, "
                  f"{len(specs)} tiles, "
                  f"{total/1e6:.2f} MB domain, gathered {gb/1e6:.2f} MB, "
                  f"sha256 {results[key][:16]}...")
    assert len(set(results.values())) == 1, (
        f"paths produced different bytes: {results}")
    print(f"  all {len(results)} (spec x memory) combinations agree bitwise")
    kinds = {}
    for n, s in shapes.items():
        kinds.setdefault(G.classify(s, nz, ny, nx), []).append(n)
    print("  kinds covered: " + ", ".join(
        f"{k}={len(v)}" for k, v in sorted(kinds.items())))
    assert set(kinds) == set(G.KINDS)
    print(f"  fields by kind: u={kinds['u']} v={kinds['v']} w={kinds['w']} "
          f"2-D={[n for n, s in shapes.items() if len(s) == 2]}")


def test_staggered_u_tile_matches_direct_slice():
    """A gathered u tile == the directly-indexed slice of the full u array."""
    import cupy as cp

    cfg = harness.make_config(NX, NY, NZ)
    state = harness.make_state(cfg)
    gx, gy = TILE + 2 * HALO, TILE + 2 * HALO
    tile = harness.make_state(harness.tile_config(cfg, gx, gy))
    src_arrays = G.inventory(state)
    tile_arrays = G.inventory(tile)
    _fill_unique(src_arrays)

    for label, planner in SPEC_PROVIDERS.items():
        specs = planner(NX, NY, TILE, TILE, halo=HALO, periodic=True)
        # the tile at (i0, j0) = (32, 32) spans mass [16, 80) in both axes:
        # strictly interior, so every gathered index is a plain direct slice.
        spec = next(s for s in specs if (s.i0, s.j0) == (TILE, TILE))
        i0, j0 = spec.i0 - HALO, spec.j0 - HALO
        assert i0 > 0 and j0 > 0 and i0 + gx < NX and j0 + gy < NY

        _zero(tile_arrays)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            G.gather_tile(src_arrays, tile_arrays, spec, stream)
        stream.synchronize()

        checks = {
            "u":   (slice(None), slice(j0, j0 + gy), slice(i0, i0 + gx + 1)),
            "v":   (slice(None), slice(j0, j0 + gy + 1), slice(i0, i0 + gx)),
            "w":   (slice(None), slice(j0, j0 + gy), slice(i0, i0 + gx)),
            "php": (slice(None), slice(j0, j0 + gy), slice(i0, i0 + gx)),
            "thp": (slice(None), slice(j0, j0 + gy), slice(i0, i0 + gx)),
            "mup": (slice(j0, j0 + gy), slice(i0, i0 + gx)),
        }
        for name, sl in checks.items():
            want = src_arrays[name][sl]
            got = tile_arrays[name]
            assert got.shape == want.shape, (name, got.shape, want.shape)
            assert bool(cp.array_equal(got, want)), \
                f"[{label}] {name} tile != direct slice"
            assert _bytes_of(got) == _bytes_of(cp.ascontiguousarray(want)), name
            print(f"  [{label}] {name:4s} tile {tuple(got.shape)} == "
                  f"full{sl_repr(sl)}")

        # and the WRONG slice must not accidentally match (guards a tautology)
        assert tile_arrays["u"].shape != \
            src_arrays["u"][:, j0:j0 + gy, i0:i0 + gx].shape
        off_by_one = src_arrays["u"][:, j0:j0 + gy, i0 + 1:i0 + gx + 2]
        assert not bool(cp.array_equal(tile_arrays["u"], off_by_one)), (
            f"[{label}] u tile matches an off-by-one slice too -- the test is "
            "not discriminating")
        print(f"  [{label}] off-by-one u slice correctly does NOT match")


def test_wrapped_edge_tile_matches_rolled_reference():
    """A corner tile whose halo wraps == a np.roll-built reference.

    The (0, 0) tile's compute window starts at ``-halo``, so its gather is
    split into four rectangles.  Building the expected answer with ``np.roll``
    -- an entirely different mechanism from the segment arithmetic -- checks
    the multi-segment wrap path against something that shares no code with it.
    """
    import cupy as cp

    nz, ny, nx, tile, halo = 5, 48, 48, 16, 8
    gx = gy = tile + 2 * halo
    shapes = _synthetic_shapes(nz, ny, nx)
    store = _device_dict(shapes)
    tile_arrays = _device_dict(_synthetic_shapes(nz, gy, gx))
    _fill_unique(store)

    for label, planner in SPEC_PROVIDERS.items():
        specs = planner(nx, ny, tile, tile, halo=halo, periodic=True)
        spec = next(s for s in specs if (s.i0, s.j0) == (0, 0))
        _zero(tile_arrays)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            G.gather_tile(store, tile_arrays, spec, stream)
        stream.synchronize()

        n_seg = len(G.segments_for(spec, "gather", "mass"))
        assert n_seg >= 4, f"[{label}] corner tile produced {n_seg} rectangles"

        for name, extra_y, extra_x in (("thp", 0, 0), ("mup", 0, 0),
                                       ("w", 0, 0), ("php", 0, 0),
                                       ("u", 0, 1), ("v", 1, 0)):
            full = cp.asnumpy(store[name])
            # Face nx IS face 0 on a periodic domain, so the INDEPENDENT data
            # is the [0, ny) x [0, nx) core; the alias row/column is dropped
            # before wrapping and re-derived by the modulo below.
            core = full[..., :full.shape[-2] - extra_y,
                        :full.shape[-1] - extra_x]
            yi = ((np.arange(gy + extra_y) - halo) % ny)[:, None]
            xi = ((np.arange(gx + extra_x) - halo) % nx)[None, :]
            want = core[..., yi, xi]
            got = cp.asnumpy(tile_arrays[name])
            assert got.shape == want.shape, (label, name, got.shape, want.shape)
            assert np.array_equal(got, want), f"[{label}] wrapped {name} wrong"
        print(f"  [{label}] corner tile: {n_seg} mass rectangles; "
              f"mass/w/2-D/u/v all match the modulo-indexed reference")


def sl_repr(sl):
    def one(s):
        if isinstance(s, slice):
            if s.start is None and s.stop is None:
                return ":"
            return f"{s.start}:{s.stop}"
        return str(s)
    return "[" + ", ".join(one(s) for s in sl) + "]"


def test_duplicate_invariant_is_load_bearing():
    """Without u[..,nx]==u[..,0] the periodic round trip must FAIL.

    Demonstrates the invariant is a real property of the data rather than a
    convenience the test quietly relies on.
    """
    nz, ny, nx, tile, halo = 4, 32, 32, 16, 4
    shapes = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
              "thp": (nz, ny, nx)}
    store = _device_dict(shapes)
    sink = _device_dict(shapes)
    tile_arrays = _device_dict(
        {"u": (nz, tile + 2 * halo, tile + 2 * halo + 1),
         "v": (nz, tile + 2 * halo + 1, tile + 2 * halo),
         "thp": (nz, tile + 2 * halo, tile + 2 * halo)})
    for label, planner in SPEC_PROVIDERS.items():
        specs = planner(nx, ny, tile, tile, halo=halo, periodic=True)

        _fill_unique(store, enforce_duplicates=False)
        ref = {n: a.copy() for n, a in store.items()}
        _zero(sink)
        _roundtrip(store, sink, tile_arrays, specs)
        try:
            _compare(ref, sink, f"no-duplicate control [{label}]")
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"[{label}] round trip passed WITHOUT the periodic duplicate "
                "invariant; the u/v face treatment is not being exercised")

        _fill_unique(store, enforce_duplicates=True)
        ref = {n: a.copy() for n, a in store.items()}
        _zero(sink)
        _roundtrip(store, sink, tile_arrays, specs)
        _compare(ref, sink, f"with-duplicate [{label}]")
        print(f"  [{label}] control fails without u[..,-1]==u[..,0] / "
              "v[..,-1,:]==v[..,0,:], passes with it")


def test_pageable_host_is_rejected():
    nz, ny, nx, tile, halo = 4, 32, 32, 16, 4
    store = {"thp": np.zeros((nz, ny, nx), dtype=np.float32)}   # PAGEABLE
    tile_arrays = _device_dict({"thp": (nz, tile + 2 * halo, tile + 2 * halo)})
    spec = plan_tiles(nx, ny, tile, tile, halo=halo)[0]
    assert not G.is_pinned(store["thp"])
    try:
        G.gather_tile(store, tile_arrays, spec)
    except G.PageableHostMemoryError as exc:
        assert "pageable" in str(exc).lower()
    else:
        raise AssertionError("pageable host memory was accepted")
    pinned = {"thp": G.pinned_empty((nz, ny, nx), np.float32)}
    G.gather_tile(pinned, tile_arrays, spec)
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    print("  pageable numpy rejected with PageableHostMemoryError; "
          "pinned accepted")


def test_no_allocation_and_plan_reuse():
    """Repeated gathers allocate no device memory and reuse one plan.

    Run against both spec implementations on purpose: ``tilestream.spec``'s
    ``TileSpec`` is a FROZEN dataclass, so the plan cache reaches it through
    ``spec.__dict__[...] = ...`` rather than ``setattr``.  If that silently
    failed, every tile would rebuild its plan -- correct, but a per-tile
    Python cost that nothing else here would notice.
    """
    import cupy as cp

    nz, ny, nx, tile, halo = 8, 64, 64, 16, 8
    shapes = _synthetic_shapes(nz, ny, nx)
    store = _pinned_dict(shapes)
    tile_a = _device_dict(_synthetic_shapes(nz, tile + 2 * halo, tile + 2 * halo))
    tile_b = _device_dict(_synthetic_shapes(nz, tile + 2 * halo, tile + 2 * halo))
    stream = cp.cuda.Stream(non_blocking=True)
    pool = cp.get_default_memory_pool()

    for label, planner in SPEC_PROVIDERS.items():
        specs = planner(nx, ny, tile, tile, halo=halo, periodic=True)
        with stream:
            for spec in specs:                  # warm up plans and pool
                G.gather_tile(store, tile_a, spec, stream)
                G.gather_tile(store, tile_b, spec, stream)
        stream.synchronize()
        before = pool.used_bytes(), pool.total_bytes()
        plans = []
        with stream:
            for _ in range(5):
                for spec in specs:
                    plans.append(G.gather_tile(store, tile_a, spec, stream))
                    plans.append(G.gather_tile(store, tile_b, spec, stream))
        stream.synchronize()
        after = pool.used_bytes(), pool.total_bytes()
        assert after == before, f"[{label}] memory pool grew: {before} -> {after}"
        # one plan object per spec, shared across both tile buffers and reps
        assert len({id(p) for p in plans}) == len(specs), (
            f"[{label}] {len({id(p) for p in plans})} distinct plans for "
            f"{len(specs)} specs -- the plan cache is not sticking to the "
            "spec object")
        frozen = type(specs[0]).__dataclass_params__.frozen \
            if hasattr(type(specs[0]), "__dataclass_params__") else False
        print(f"  [{label}] {len(plans)} gathers over 2 tile buffers "
              f"(spec frozen={frozen}): pool used/total unchanged at "
              f"{before}, {len(specs)} plan objects reused")


def test_bad_spec_is_rejected():
    """Extent mismatches and missing kinds raise instead of copying garbage."""
    nz, ny, nx, tile, halo = 4, 32, 32, 16, 4
    store = _device_dict({"thp": (nz, ny, nx)})
    tile_arrays = _device_dict({"thp": (nz, tile + 2 * halo, tile + 2 * halo)})

    class _Empty:
        gather = {}
        scatter = {}
    try:
        G.gather_tile(store, tile_arrays, _Empty())
    except G.SpecError:
        pass
    else:
        raise AssertionError("a spec with no segments was accepted")

    class _Mismatched:
        gather = {"mass": [((slice(0, 8), slice(0, 8)),
                            (slice(0, 24), slice(0, 24)))]}
        scatter = {}
    try:
        G.gather_tile(store, tile_arrays, _Mismatched())
    except G.SpecError as exc:
        assert "extent" in str(exc)
    else:
        raise AssertionError("a src/dst extent mismatch was accepted")

    class _VerticalSlice:
        gather = {"mass": [((slice(0, 2), slice(0, 24), slice(0, 24)),
                            (slice(None), slice(0, 24), slice(0, 24)))]}
        scatter = {}
    try:
        G.gather_tile(store, tile_arrays, _VerticalSlice())
    except G.SpecError as exc:
        assert "vertical" in str(exc)
    else:
        raise AssertionError("a partial vertical slice was accepted")

    # A tile array one column too wide.  This is the dangerous case: every
    # rectangle still fits inside the array, so nothing is out of bounds --
    # the gather would just leave the last column holding stale values.  Only
    # the "rectangles must cover the tile exactly" check sees it.
    g = tile + 2 * halo
    bad_tile = _device_dict({"thp": (nz, g, g + 1)})
    spec = plan_tiles(nx, ny, tile, tile, halo=halo)[0]
    try:
        G.gather_tile(store, bad_tile, spec)
    except G.SpecError as exc:
        assert "cover" in str(exc) and f"of {g * (g + 1)} points" in str(exc), exc
        print(f"    oversized tile: {exc}")
    else:
        raise AssertionError("an incompletely covered tile array was accepted")
    # ...and it IS accepted with the completeness check switched off, which
    # shows the check is what rejected it rather than some other guard.
    G.gather_tile(store, bad_tile, spec, require_full_gather=False)

    # overlapping destination rectangles: a point written twice
    class _Overlapping:
        gather = {"mass": [((slice(0, g), slice(0, g)),
                            (slice(0, g), slice(0, g))),
                           ((slice(0, 4), slice(0, 4)),
                            (slice(0, 4), slice(0, 4)))]}
        scatter = {}
    good_tile = _device_dict({"thp": (nz, g, g)})
    try:
        G.gather_tile(store, good_tile, _Overlapping())
    except G.SpecError as exc:
        assert "overlap" in str(exc), exc
    else:
        raise AssertionError("overlapping destination rectangles were accepted")

    print("  rejected: empty segments, extent mismatch, partial z slice, "
          "incomplete tile coverage, overlapping destination rectangles")


def test_real_spec_is_under_test():
    """Report which spec implementations the round trips above exercised.

    Fails loudly if only the stand-in is available: passing tests that never
    touched the real ``tilestream.spec`` would be a much weaker result than
    they look.
    """
    print(f"  specs exercised: {sorted(SPEC_PROVIDERS)}")
    if "tilestream.spec" not in SPEC_PROVIDERS:
        raise AssertionError(
            "tilestream.spec is missing or has no plan_tiles; every result "
            "above came from the in-test stand-in only")
    spec = SPEC_PROVIDERS["tilestream.spec"](96, 96, 32, 32, halo=16,
                                             periodic=True)[0]
    for kind in G.KINDS:
        g = G.segments_for(spec, "gather", kind)
        s = G.segments_for(spec, "scatter", kind)
        assert g and s, (kind, len(g), len(s))
        print(f"    {kind:5s} gather {len(g)} rect, scatter {len(s)} rect")
    if real_spec is not None and hasattr(real_spec, "validate_plan"):
        real_spec.validate_plan(
            real_spec.plan_tiles(96, 96, 32, 32, halo=16, periodic=True),
            96, 96)
        print("    spec.validate_plan(): exact coverage confirmed by the "
              "spec's own checker")


def report_transport_shape():
    """How many memcpy3DAsync calls one tile costs.  Structural, not timed.

    This is the number the whole ``cudaMemcpy3DAsync`` design turns on and it
    is immune to GPU contention, so it is reported unconditionally.  The
    alternative implementation -- one ``cudaMemcpy2DAsync`` per k-plane --
    would need ``nz`` (or ``nz+1``) launches for every one of these.
    """
    cfg = harness.make_config(NX, NY, NZ)
    state = harness.make_state(cfg)
    gx, gy = TILE + 2 * HALO, TILE + 2 * HALO
    tile = harness.make_state(harness.tile_config(cfg, gx, gy))
    store = G.make_host_store(state, copy=False)
    planner = SPEC_PROVIDERS.get("tilestream.spec", plan_tiles)
    specs = planner(NX, NY, TILE, TILE, halo=HALO, periodic=True)

    for which, spec in (("corner tile (halo wraps, 4 rects/field)", specs[0]),
                        ("interior tile (1 rect/field)",
                         next(s for s in specs if (s.i0, s.j0) == (TILE, TILE)))):
        gp = G.make_plan(G.inventory(store), G.inventory(tile), spec, "gather")
        sp = G.make_plan(G.inventory(tile), G.inventory(store), spec, "scatter")
        planes = sum(c.depth for c in gp.copies)
        rows = sorted(c.w_bytes for c in gp.copies)
        print(f"  {which}")
        print(f"    gather : {len(gp.copies):3d} memcpy3DAsync calls, "
              f"{gp.nbytes/1e6:7.2f} MB  "
              f"(per-plane memcpy2D would need {planes})")
        print(f"    scatter: {len(sp.copies):3d} memcpy3DAsync calls, "
              f"{sp.nbytes/1e6:7.2f} MB")
        print(f"    gather contiguous-row widths: min {rows[0]} B, "
              f"max {rows[-1]} B")
    print("  NOTE: a wrapping tile's halo-side rectangles are only `halo`")
    print("  columns wide -- 64 B rows at halo=16, float32. Those rows are")
    print("  far below the ~4 KB where a strided PCIe copy reaches full rate,")
    print("  so edge tiles WILL move bytes more slowly than interior tiles.")
    print("  Not a correctness issue; it is a reason to prefer large tiles,")
    print("  and a thing to measure before trusting any aggregate GB/s.")


def benchmark_gather_bandwidth(*, force=False):
    """Pinned-host H2D gather bandwidth.  DEFERS under GPU contention.

    Accuracy constraints this obeys:
      * working set is ~1.8 GB, far above the 96 MB L2 and the 200 MB floor,
        so no cache can inflate it;
      * timed with CUDA events plus a full device sync;
      * paired with a correctness check on the same transfers -- a gather
        that skipped work would look like free bandwidth, so the tile is
        verified against a modulo-indexed reference and the moved byte count
        is checked against the geometry.
    """
    import cupy as cp

    if (BASELINE_VRAM_MIB is not None and BASELINE_VRAM_MIB > 8000
            and not force):
        print(f"  DEFERRED: {BASELINE_VRAM_MIB:.0f} MiB of VRAM was already "
              "held by another workload when this process started. A "
              "contended GPU produces bandwidth numbers that are not signal "
              "(the spike already saw a sweep come back physically "
              "impossible under contention). Re-run with --force-bench on an "
              "idle card.")
        return None

    nz, ny, nx, tile, halo = NZ, 1024, 1024, 256, HALO
    gx = gy = tile + 2 * halo
    names = ("u", "v", "w", "thp", "php", "mup", "p", "al", "alt")
    shapes = {n: s for n, s in _synthetic_shapes(nz, ny, nx).items()
              if n in names}
    store = _pinned_dict(shapes)
    tile_arrays = _device_dict(
        {n: s for n, s in _synthetic_shapes(nz, gy, gx).items() if n in names})
    _fill_unique(store)

    planner = SPEC_PROVIDERS.get("tilestream.spec", plan_tiles)
    specs = planner(nx, ny, tile, tile, halo=halo, periodic=True)
    stream = cp.cuda.Stream(non_blocking=True)

    with stream:                                    # warm up plans + pinning
        for spec in specs:
            plan = G.gather_tile(store, tile_arrays, spec, stream)
    stream.synchronize()

    # correctness on the LAST tile gathered, before any timing is reported
    spec = specs[-1]
    with stream:
        G.gather_tile(store, tile_arrays, spec, stream)
    stream.synchronize()
    yi = ((np.arange(gy) + spec.i0 * 0 + (spec.j0 - halo)) % ny)[:, None]
    xi = ((np.arange(gx) + (spec.i0 - halo)) % nx)[None, :]
    want = store["thp"][..., yi, xi]
    assert np.array_equal(cp.asnumpy(tile_arrays["thp"]), want), \
        "benchmark gather is WRONG; the bandwidth number below would be fake"
    expect = sum(int(np.prod(_synthetic_shapes(nz, gy, gx)[n])) * 4
                 for n in names)
    assert plan.nbytes == expect, (plan.nbytes, expect)

    reps = 3
    start, end = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.runtime.deviceSynchronize()
    start.record(stream)
    with stream:
        for _ in range(reps):
            for spec in specs:
                G.gather_tile(store, tile_arrays, spec, stream)
    end.record(stream)
    end.synchronize()
    cp.cuda.runtime.deviceSynchronize()
    ms = cp.cuda.get_elapsed_time(start, end)
    moved = plan.nbytes * len(specs) * reps
    gbs = moved / (ms * 1e-3) / 1e9
    print(f"  domain {nx}x{ny}x{nz}, {len(names)} fields, "
          f"store {sum(a.nbytes for a in store.values())/1e9:.2f} GB pinned")
    print(f"  tile {gx}x{gy} = {plan.nbytes/1e6:.1f} MB in "
          f"{len(plan.copies)} memcpy3DAsync calls")
    print(f"  {len(specs)} tiles x {reps} reps: {moved/1e9:.2f} GB in "
          f"{ms:.2f} ms = {gbs:.1f} GB/s H2D "
          f"({100*gbs/57.1:.0f}% of the 57.1 GB/s pinned peak)")
    print("  (gather verified against a modulo-indexed reference first)")
    return gbs


# ==========================================================================

TESTS = [
    test_classify_derives_stagger_from_shape,
    test_extents_inferred_from_inventory,
    test_layered_fields_gather_like_mass_fields,
    test_rank4_carrier_gathers_and_scatters_every_plane,
    test_device_roundtrip_real_state,
    test_host_roundtrip_pinned,
    test_all_49_names_device_and_host,
    test_staggered_u_tile_matches_direct_slice,
    test_wrapped_edge_tile_matches_rolled_reference,
    test_duplicate_invariant_is_load_bearing,
    test_pageable_host_is_rejected,
    test_no_allocation_and_plan_reuse,
    test_bad_spec_is_rejected,
    test_real_spec_is_under_test,
]


def main() -> int:
    import cupy as cp

    print(f"cupy {cp.__version__}  device "
          f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cudaMemGetInfo: {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB")
    print(f"domain {NX}x{NY}x{NZ}, interior tile {TILE}x{TILE}, halo {HALO}")
    print()
    failed = 0
    for fn in TESTS:
        print(f"{fn.__name__} ...")
        try:
            fn()
        except Exception:
            failed += 1
            print("  FAILED")
            traceback.print_exc()
        else:
            print("  PASS")
        print()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    print()
    print("transport shape (structural, not timed) ...")
    report_transport_shape()
    print()
    print("gather bandwidth ...")
    benchmark_gather_bandwidth(force="--force-bench" in sys.argv)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
