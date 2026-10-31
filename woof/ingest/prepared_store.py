"""The prepared cache, loaded straight into a pinned host store.

The resident road -- :func:`woof.ingest.prepared_cache.restore_prepared_cache`
followed by :func:`woof.ingest.hrrr_physics.initialize_prepared_physics` --
builds ONE full-domain ``DomainState`` on the card and then lets
:func:`woof.core.streaming.attach` copy it, carrier by carrier, into pinned
host RAM.  Every byte of that state is dead the moment the copy finishes: a
streamed forecast integrates out of the store and never reads it again.  It is
also the ceiling.  MEASURED on a 16 GB card, nz = 49: the bare state costs
11 276.5 B per column and the prepared case about 15 780, so 1024 x 1024
refuses inside ``initialize_physics`` while the forecast it would have fed
needs about 6 GiB and would have run comfortably.  The streamed engine on the
same card has stepped 2624 x 2624.

This module removes the intermediary.  It reads the cache one ROW SLAB at a
time, builds a slab-height state, attaches physics to THAT, and scatters the
result into full-domain pinned host arrays -- so the largest device
allocation in a preparation is one slab, and the domain the card can carry
stops being a property of the card at all.

:func:`tilestream.bigdomain.build_store_by_slabs` already does exactly this
for an analytic initial condition, and this is deliberately its structure:
price the manifest, refuse before the first pinned byte, then fill.  The two
differences are both about where the rows come from.

*The rows come off disk, not out of a sounding.*  The cache's integrity
guarantee is a SHA-256 over each array WHOLE. Read-only memory maps are
checked against the manifest and hashed directly by the Rust CPU bridge,
then those same maps serve the slab rows. Older bridges retain the checked
full-array read before serving mapped rows. Rows use read-only memory maps,
so what stays resident is page cache the kernel may evict rather than a
process that has swallowed its own input.  Neither pass puts anything on the
device at domain shape.  See :class:`_CachePayload` -- holding the payload
whole would have cost about 11 GiB of prognostics at 1024 x 1024 x 49 on top
of the 9 GiB store, and multiplied on every stretch rung above it.

*Physics is attached per slab.*  The cache carries
``STATE_SERIALIZED_ATTRS`` only -- the dynamics and microphysics prognostics.
Everything the store also needs (``fields/*``, ``driver/*``, ``cumulus/*``,
``scratch/*``) is made by ``initialize_prepared_physics``, which is
column-local given windowed inputs: its one apparent reduction,
``GREENFRAC.min(axis=0)``, is over the twelve MONTHS and not over space.  The
two places a slab could disagree with the domain are named and handled --
``center_lat`` is passed down explicitly rather than re-derived from the
slab's own grid (``initialize_landuse`` reads it for the LANDUSE.TBL season),
and the y-staggered inputs take one extra row so the ``V10`` face average at
a slab's last row sees the same neighbour the whole domain would.  The
``topo_wind = 1`` terrain laplacian also uses one neighbour row on each
side before its coefficients are retained in the geography store.

That the two roads agree is not argued from this docstring: a domain small
enough for both produces bit-identical frames, which is the parity gate.
"""

from __future__ import annotations

import gc
import copy
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np

__all__ = [
    "PreparedStore",
    "PreparedStoreError",
    "default_slab_rows",
    "store_from_prepared_cache",
]


class PreparedStoreError(RuntimeError):
    """A prepared cache could not be loaded into a host store."""


@dataclass(frozen=True)
class PreparedStore:
    """What a streamed run needs, with no resident domain state in it.

    ``store`` and ``geography`` are dicts of pinned full-domain host arrays,
    keyed exactly as this run's own ``inventory_fn`` -- by default
    :func:`woof.core.streaming.streamed_store_inventory`, the rule the tile
    buffers are built by -- and :func:`tilestream.driver.geography_inventory`
    key them, which is what :class:`tilestream.driver.TiledRun` requires of
    both.

    ``base`` is the WHOLE domain's base state, and the word to hold onto is
    DOMAIN-SHAPED rather than domain-invariant.  Flat terrain makes it look
    invariant -- :class:`woof.core.grid.BaseState` is then a scalar ``mub``
    and 1-D profiles that broadcast against any height -- but with terrain
    ``mub`` is ``(ny, nx)`` and ``pb``/``alb``/``thb``/``phb`` are full 3-D
    ``(nz[,+1], ny, nx)`` arrays.  Every slab is therefore built against a
    ROW-WINDOWED copy (:func:`_slab_base`), while the bundle carries the
    un-windowed original: a consumer holding this object wants the DOMAIN's
    base state, not whichever slab happened to be built last.

    ``template`` is a SLAB-HEIGHT state, not a domain-shaped one.  It exists
    because :func:`woof.core.streaming.prepared_tile_state_factory` reads
    three things off "the domain": its scheme adapters (to twin per buffer),
    its vertical coordinate, and the ``ndim < 2`` half of its setup arrays.
    All three are identical at every height -- the eta table does not vary by
    row -- so a slab carries them faithfully and costs a fraction of a
    domain.  The horizontally-varying setup arrays are NOT taken from it;
    those are gathered per tile out of ``geography``.
    """

    store: dict
    geography: dict
    scalars: dict
    template: object
    coord: object
    base: object
    boundaries: object
    missing: tuple
    receipt: object


def _row_window(arr, j0: int, rows: int, ny: int):
    """``rows`` rows of ``arr`` at ``j0``, one more if it is y-staggered.

    The horizontal axes are the last two -- the rule the whole store is
    derived from (``hoststore.manifest_from_arrays``).  A field whose
    second-to-last extent is ``ny + 1`` carries the closing face, so slab k's
    last row IS slab k+1's first; both are the same number, so the overlap is
    a duplicate rather than a conflict.  Anything else is refused loudly:
    silently windowing an array of unexpected height is how a slab ends up
    holding a shifted copy of the domain.
    """
    if arr is None:
        return None
    if isinstance(arr, (str, bytes, bool, int, float)):
        # A static bundle carries scalars and identity strings beside its
        # fields (MMINLU and friends).  Windowing is a horizontal operation
        # and these have no horizontal extent; passing them through
        # unchanged is what keeps a slab's static mapping the same KIND of
        # object the whole-domain one is, rather than a 0-d array that the
        # consumer will silently treat as a number.
        return arr
    array = np.asarray(arr)
    if array.ndim < 2:
        return array
    height = int(array.shape[-2])
    if height == int(ny):
        take = int(rows)
    elif height == int(ny) + 1:
        take = int(rows) + 1
    else:
        raise PreparedStoreError(
            f"cannot window an array of height {height} against a domain of "
            f"{ny} mass rows: it is neither mass- nor y-staggered, so which "
            "rows belong to a slab is undefined")
    return array[..., j0:j0 + take, :]


def _window_mapping(mapping, j0: int, rows: int, ny: int):
    """Every array in a static/surface/met mapping, windowed by rows."""
    if mapping is None:
        return None
    return {name: _row_window(value, j0, rows, ny)
            for name, value in mapping.items()}


def _complete_slab_topo_wind(state, cfg, static, base, j0, rows, ny):
    """Use true terrain neighbours before harvesting a slab's coefficients.

    Physics initialization sees only the slab's rows. topo_wind=1 needs a
    terrain neighbour on either side, so that initial local-edge clamp is
    replaced with the domain stencil here. Only one extra row per side is
    uploaded; no domain-shaped device array is needed. The local land mask
    and statistics remain the initialized column's own values.
    """
    option = int(getattr(cfg, "topo_wind", 0))
    if option != 1 or (j0 == 0 and rows == ny):
        return
    import cupy as cp

    from woof.core.terrain_drag import topo_wind_coefficients

    first, last = max(0, j0 - 1), min(ny, j0 + rows + 1)
    before, after = j0 - first, last - (j0 + rows)
    height = (cp.zeros((last - first, int(cfg.nx)), dtype=cp.float32)
              if base.terrain_z is None else
              cp.asarray(_row_window(base.terrain_z, first, last - first, ny),
                         dtype=cp.float32))
    # Padding supplies only discarded coefficient rows. The kernel's
    # Laplacian reads height; it never reads a neighbour's land mask.
    xland = cp.pad(state.physics.fields["xland"],
                   ((before, after), (0, 0)), mode="edge")
    ctopo, ctopo2, _ = topo_wind_coefficients(
        height, xland, topo_wind=option,
        var_sso=_row_window(static["VAR_SSO"], first, last - first, ny))
    take = slice(before, before + rows)
    state.physics.terrain_drag.ctopo[...] = ctopo[take]
    state.physics.terrain_drag.ctopo2[...] = ctopo2[take]


def _slab_base(base, j0: int, rows: int, ny: int):
    """``base`` with every ARRAY field row-windowed, its scalars untouched.

    THE BASE STATE IS NOT DOMAIN-INVARIANT, and believing that it was is the
    defect this function exists to remove.  Flat terrain makes it look
    invariant: :class:`woof.core.grid.BaseState` is then a SCALAR ``mub``
    and four 1-D columns that broadcast against ``(nz, ny, nx)`` at any
    height, so a slab handed the domain's base gets the right numbers by
    accident.  WITH TERRAIN -- which is every real case -- ``mub`` is
    ``(ny, nx)``, ``pb``/``alb``/``thb`` are ``(nz, ny, nx)``, ``phb`` is
    ``(nz + 1, ny, nx)`` and ``terrain_z`` is ``(ny, nx)``;
    :meth:`woof.core.state.DomainState.load_base` assigns each into a
    SLAB-shaped device array with ``dev[...] = ...``, which broadcasts only
    while the heights agree and refuses the moment they do not.  MEASURED on
    the parity gate at 384 x 384 with 64-row slabs, three times on real
    data: ``ValueError: operands could not be broadcast together with shapes
    (49, 384, 384) (49, 64, 384)``.  The store's own unit tests missed it
    because they use a flat base and an ``ny`` that fits in one slab, and a
    one-slab domain-height array broadcasts.

    The field list is DERIVED, from :func:`dataclasses.fields`, so a field
    added to ``BaseState`` is windowed the day it is added rather than being
    handed to every slab whole -- the same failure again, in a new array,
    with the unit tests still green.  Each field goes through
    :func:`_row_window`, so the mass/y-staggered rule and its loud refusal
    are the store's ONE rule rather than a second copy of it that can drift;
    ``p_top``, and ``mub`` when the terrain is flat, fall through it as
    themselves, which is what keeps a flat-terrain base exactly the object
    it was.
    """
    if not is_dataclass(base):
        raise PreparedStoreError(
            f"the prepared cache's base state is a {type(base).__name__}, "
            "which is not a dataclass: the per-slab window is derived from "
            "dataclasses.fields precisely so that no field can be missed, "
            "and an object it cannot enumerate is one it cannot window "
            "safely")
    return replace(base, **{
        spec.name: _row_window(getattr(base, spec.name), j0, rows, ny)
        for spec in fields(base) if spec.init})


def _as_host(arr) -> np.ndarray:
    import cupy as cp

    return cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else np.asarray(arr)


def _hold_domain_carrier(dst: np.ndarray, src, name: str, index: int) -> None:
    """Keep a domain-owned carrier whole: every slab must build the same words.

    A carrier ``tilestream.global_inventory.global_keys`` names (fixed shape,
    passed whole to every tile, e.g. the 3D PWP ocean status word
    ``fields/_pwp_status``, shape (1,)) has no rows to scatter.  Scattering
    it raised "IndexError: tuple index out of range" at the start of every
    store-direct forecast with sf_ocean_physics = 2, which 48 h runs select
    by default (full CONUS HRRR, --devices 2, 2026-10-21).  Slab 0's
    words are kept; a later slab that built different words is refused,
    because the domain would otherwise start from one slab's arbitrary view.
    """
    host = np.asarray(_as_host(src))
    if host.shape != dst.shape:
        raise PreparedStoreError(
            f"slab {index} built domain carrier {name!r} at shape "
            f"{host.shape}, not its fixed shape {dst.shape}")
    if index == 0:
        dst[...] = host
    elif not np.array_equal(dst.view(np.uint8), np.ascontiguousarray(host).view(np.uint8)):
        raise PreparedStoreError(
            f"slab {index} built domain carrier {name!r} with words that "
            "differ from slab 0's; a domain-owned carrier must be "
            "slab-invariant")


def _scatter_rows(dst: np.ndarray, src, j0: int) -> None:
    """Write ``src``'s rows into ``dst`` at ``j0``, inferring the staggering.

    :func:`tilestream.bigdomain._scatter_rows`' rule, and for the same
    reason: the destination row span is the source's own second-to-last
    extent, so a v-staggered field writes ``rows + 1`` rows.
    """
    host = _as_host(src)
    dst[..., j0:j0 + host.shape[-2], :] = host


def _seam_disagreement(dst: np.ndarray, src: np.ndarray, j0: int,
                       rows: int) -> int:
    """Differing values on a y-staggered carrier's SHARED face.

    A y-staggered field's first row at ``j0`` is the same face as the
    previous slab's last row, so it is written twice.  For anything COPIED
    out of the cache that is a duplicate and harmless: both slabs read the
    same number.  For anything a slab COMPUTES it need not be, and the
    failure is specific and nasty -- slab k writes that face as its own
    domain-TOP edge, slab k+1 overwrites it as its own domain-BOTTOM edge,
    and neither is the interior value a whole domain would have produced.
    That is exactly the signature ``tilestream.test_realdata`` records for
    ``y_halo = 0``: one field, wrong in ``nslabs - 1`` rows, by an amount
    small enough to look like weather.

    So the overlap is not assumed to be a duplicate, it is COUNTED, over
    raw bits so that a signed zero or a NaN pair is judged the way the
    parity gate will judge it.  Returns how many values on the shared face
    the second writer would change.
    """
    # ONLY a y-staggered carrier has a shared face.  A mass-height field's
    # row j0 is the first row THIS slab owns and nobody has written it yet,
    # so comparing it against whatever the buffer happens to hold reports a
    # disagreement that does not exist -- measured on a zero-filled store,
    # every mass field scored a full row of false positives.  The staggering
    # is inferred the way the rest of this module infers it: from the
    # source's own height against the rows the slab owns.
    if j0 <= 0 or int(src.shape[-2]) != int(rows) + 1:
        return 0
    existing = dst[..., j0, :]
    incoming = src[..., 0, :]
    if existing.shape != incoming.shape or existing.dtype != incoming.dtype:
        return 0
    if existing.dtype.kind not in ("f", "i", "u"):
        return 0
    width = existing.dtype.itemsize
    return int(np.count_nonzero(
        np.ascontiguousarray(existing).view(f"u{width}")
        != np.ascontiguousarray(incoming).view(f"u{width}")))


def _plan_slabs(ny: int, rows_per_slab: int) -> tuple[tuple[int, int], ...]:
    """``(j0, rows)`` pairs partitioning ``[0, ny)`` exactly.

    A partition, not an overlay: every mass row is written by exactly one
    slab.  The last slab is short rather than the loop refusing an ny that
    ``rows_per_slab`` does not divide -- ``build_store_by_slabs`` may demand
    divisibility because it also sizes an analytic noise field by it, and
    nothing here does.
    """
    ny = int(ny)
    rows_per_slab = max(1, int(rows_per_slab))
    if ny < 1:
        raise PreparedStoreError(f"a domain of {ny} rows cannot be slabbed")
    out = []
    j0 = 0
    while j0 < ny:
        out.append((j0, min(rows_per_slab, ny - j0)))
        j0 += rows_per_slab
    return tuple(out)


def _slab_state(cfg_slab, coord, base_slab, static_slab, cache_rows,
                state_names):
    """One slab-height state, carrying the cache's own rows.

    The construction is ``restore_prepared_cache``'s, at slab height:
    ``DomainState`` for the shapes, ``load_base`` for the vertical
    coordinate and the base state, and ``set_map_coriolis`` for the slab's
    own map factors.  Then the prognostics, which is the only part that
    reads the cache.

    THE TWO ARGUMENTS ARE NOT THE SAME KIND OF THING, and an earlier version
    of this docstring said they were.  ``coord`` is the DOMAIN's and is
    passed whole because the eta table has no horizontal extent -- it is
    identical at every row, at any slab height.  ``base_slab`` is the
    domain's base state ALREADY WINDOWED to this slab's rows by
    :func:`_slab_base`, because under terrain the base carries an
    ``(ny, nx)`` dry mass and full 3-D profiles.  The window is the caller's
    to make rather than this function's so that the SAME windowed object
    also reaches ``CachedInitialResult`` -- the state and the physics have
    to agree about which rows the base describes, and building it twice is
    how they would stop agreeing.
    """
    import cupy as cp

    from woof.core.state import DomainState

    state = DomainState(cfg_slab)
    state.load_base(coord, base_slab)
    state.set_map_coriolis(
        static_slab["MAPFAC_M"], static_slab["MAPFAC_U"],
        static_slab["MAPFAC_V"], static_slab["F"], static_slab["E"],
        sina=static_slab["SINALPHA"], cosa=static_slab["COSALPHA"],
        msfv_inv=static_slab.get("MF_VX_INV"))
    for name in state_names:
        target = getattr(state, name, None)
        if target is None:
            raise PreparedStoreError(
                f"the cache carries state/{name} but a slab-height state has "
                "no such array; the active config does not match the cache")
        rows = cache_rows(name)
        if tuple(rows.shape) != tuple(target.shape) \
                or rows.dtype != target.dtype:
            raise PreparedStoreError(
                f"prepared cache state/{name} windowed to {rows.shape} "
                f"{rows.dtype} against a slab expecting {target.shape} "
                f"{target.dtype}")
        # ascontiguousarray, not asarray: a row window of a memory-mapped
        # array is a strided VIEW, and a host->device copy of one is either
        # a silent extra copy or a refusal depending on the backend's mood.
        # Made explicit so the cost is visible and the behaviour is not a
        # property of which CuPy is installed.
        target[...] = cp.asarray(np.ascontiguousarray(rows))
    refuse_strict_streamed_base(state.thb)
    return state


def refuse_strict_streamed_base(thb) -> None:
    """Strict WRF mode refuses a tile-streamed prepared slab whose base theta is not 300 K.

    THE BREAKAGE THIS PREVENTS (2.9.1): every strict kernel reads ``thp`` as
    WRF's ``t = theta - 300`` (``thb == 300``).  The resident prepared door
    re-expresses its state that way (woof.core.state.adopt_wrf_theta_carrier);
    this tile-streamed road keeps the preparation's base profile in ``thb``,
    so strict mode would read theta up to 164 K low and stop non-finite in its
    first steps (measured on the resident road before its fix).  The road's
    setup fingerprint binds that base per slab, so it is refused here by name
    rather than re-expressed.  Default mode, and a base already at 300 K, pass.
    """
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        return
    host = thb.get() if hasattr(thb, "get") else np.asarray(thb)
    if bool((host == np.float32(300.0)).all()):
        return
    raise PreparedStoreError(
        "strict WRF mode (GPUWM_WRF_EXACT=1) does not run a tile-streamed prepared "
        "domain: this road holds the preparation's base theta, not WRF's "
        "T = theta - 300 that every strict kernel reads, so it would stop "
        "non-finite; run strict mode on a domain that fits on the card resident, "
        "or through the WRF-input door (woof run --wrfinput)")


class _CachePayload:
    """Row-windowed access to the same mappings whose whole digest was checked.

    Rust hashes immutable C-contiguous mappings in bounded batches. A bridge
    without that additive ABI, or a non-contiguous legacy array, retains the
    original checked full-array read. Coordinate and base arrays use this
    same verification rather than being materialized and hashed again.

    Mappings retain file-backed pages without retaining file descriptors,
    including when a caller keeps an array view after this payload closes.
    Published cache files must remain immutable while mapped. File identity,
    size and timestamps are checked around verification and before reuse,
    so missing, replaced or changed payloads are refused. Those checks cannot
    protect a mapping against concurrent in-place truncation; cache writers
    publish private completed files by rename and never modify them in place.
    """

    def __init__(self, reader, *, verify: bool = True, log=print):
        self._reader = reader
        self._specs = copy.deepcopy(reader.arrays)
        self._header = copy.deepcopy(getattr(reader, "header", None))
        self._content_sha256 = getattr(reader, "content_sha256", None)
        self._maps: dict = {}
        self._files: dict = {}
        self._verified: set = set()
        if verify:
            self.verify_all(log=log)

    def _file_identity(self, key):
        from woof.ingest.prepared_cache import (
            PreparedCacheCorruptError, _raise_if_resource_limit)
        try:
            stat = (self._reader.path / self._specs[key]["file"]).stat()
        except OSError as exc:
            _raise_if_resource_limit(
                exc, context=f"prepared cache array {key!r}")
            raise PreparedCacheCorruptError(
                f"prepared cache array {key!r} is unreadable") from exc
        return (stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)

    def _check_unchanged(self, key):
        from woof.ingest.prepared_cache import PreparedCacheCorruptError
        if self._file_identity(key) != self._files[key]:
            raise PreparedCacheCorruptError(
                f"prepared cache array {key!r} changed during mapped use")

    def _verify(self, keys):
        from woof.ingest.prepared_cache import (
            PreparedCacheCorruptError, _raise_if_resource_limit)
        from woof.ingest.prepared_writer import (
            batch_budget, hash_arrays, native_hasher)
        for key in keys:
            if key in self._verified:
                self._check_unchanged(key)
        keys = [key for key in keys if key not in self._verified]
        if not keys:
            return
        try:
            entry = native_hasher()
        except OSError as exc:
            _raise_if_resource_limit(exc, context="prepared cache native verifier")
            raise
        workers, budget = batch_budget()
        pending = []
        pending_bytes = 0

        def flush():
            if not pending:
                return
            try:
                hashes = hash_arrays(entry, [self[key] for key in pending], workers=workers)
            except OSError as exc:
                _raise_if_resource_limit(exc, context="prepared cache native verifier")
                raise
            for key, digest in zip(pending, hashes):
                self._check_unchanged(key)
                if digest != self._specs[key]["sha256"]:
                    raise PreparedCacheCorruptError(
                        f"prepared cache array {key!r} fails its manifest")
                self._verified.add(key)
            pending.clear()

        for key in keys:
            mapped = self[key]
            if entry is None or not mapped.flags.c_contiguous:
                self._reader.read_array(key)
                self._check_unchanged(key)
                self._verified.add(key)
                continue
            if pending and (len(pending) >= workers
                            or pending_bytes + mapped.nbytes > budget):
                flush()
                pending_bytes = 0
            pending.append(key)
            pending_bytes += mapped.nbytes
        flush()

    def verify_all(self, *, log=print):
        t0 = time.perf_counter()
        self._verify(sorted(self._specs))
        log(f"    cache payload verified against its manifest: "
            f"{len(self._reader.arrays)} arrays, "
            f"{self._reader.payload_bytes / hoststore_gib():.2f} GiB, "
            f"in {time.perf_counter() - t0:.1f}s")

    def verified_array(self, key):
        self._verify([key])
        return self[key]

    def __getitem__(self, key: str):
        mapped = self._maps.get(key)
        if mapped is None:
            from woof.ingest.prepared_cache import (
                PreparedCacheCorruptError, _raise_if_resource_limit)
            from woof.ingest.prepared_mmap import map_npy_readonly
            try:
                spec = self._specs[key]
            except KeyError as exc:
                raise PreparedStoreError(
                    f"prepared cache is missing array {key!r}") from exc
            before = self._file_identity(key)
            try:
                mapped = map_npy_readonly(self._reader.path / spec["file"])
            except (OSError, EOFError, ValueError) as exc:
                _raise_if_resource_limit(
                    exc, context=f"prepared cache array {key!r}")
                raise PreparedCacheCorruptError(
                    f"prepared cache array {key!r} is unreadable") from exc
            if list(mapped.shape) != list(spec["shape"]) \
                    or str(mapped.dtype) != str(spec["dtype"]) \
                    or int(mapped.nbytes) != int(spec["nbytes"]):
                raise PreparedCacheCorruptError(
                    f"prepared cache array {key!r} maps as {mapped.shape} "
                    f"{mapped.dtype} against a manifest declaring "
                    f"{spec['shape']} {spec['dtype']}")
            self._maps[key] = mapped
            self._files[key] = before
        self._check_unchanged(key)
        return mapped

    def close(self) -> None:
        self._maps.clear()
        self._files.clear()
        self._verified.clear()

    def read_only_reader(self):
        """An explicit immutable view, leaving the public reader mutable."""
        return _ReadOnlyCacheReader(self)

    def copying_reader(self):
        """Writable independent arrays from the retained verified mappings."""
        return _CopyingCacheReader(self)

    def require_binding(self, path, expected_identity, *, reader=None):
        """Validate the current sealed header before reusing verified maps.

        This never adopts a new mapping based on timestamps. Only this
        payload's already-hashed immutable maps can be reused. A fresh
        header validation and file identity checks refuse changed authority
        or replaced payload files before a slab is built.
        """
        from pathlib import Path
        from woof.ingest.prepared_cache import (
            PreparedCacheReader, PreparedCacheCorruptError)
        if (reader is not None and reader is not self._reader
                or Path(path).resolve() != self._reader.path.resolve()
                or not isinstance(self._reader, PreparedCacheReader)
                or self._content_sha256 is None):
            raise PreparedStoreError(
                "retained prepared payload is bound to a different reader or path")
        current = PreparedCacheReader(path, expected_identity=expected_identity)
        if (current.content_sha256 != self._content_sha256
                or current.header != self._header
                or self._reader.header != self._header
                or current.arrays != self._specs
                or self._reader.arrays != self._specs
                or self._reader.content_sha256 != self._content_sha256):
            raise PreparedCacheCorruptError(
                "prepared cache header changed before retained mapped use")
        self.verify_all(log=lambda _: None)
        return self._reader


class _ReadOnlyCacheReader:
    """A sealed reader's metadata and its retained SHA-verified mappings."""

    def __init__(self, payload):
        self._payload = payload

    def __getattr__(self, name):
        return getattr(self._payload._reader, name)

    def read_array(self, key):
        return self._payload.verified_array(key)

    def verify_all(self):
        from woof.ingest.prepared_cache import PREPARED_CACHE_SCHEMA
        self._payload.verify_all(log=lambda _: None)
        return {
            "schema": PREPARED_CACHE_SCHEMA, "status": "PASS",
            "path": str(self.path.resolve()),
            "content_sha256": self.content_sha256,
            "array_count": len(self.arrays), "payload_bytes": self.payload_bytes,
        }


class _CopyingCacheReader(_ReadOnlyCacheReader):
    """Preserve normal prepared-reader copy semantics without a second hash."""

    def read_array(self, key):
        mapped = self._payload.verified_array(key)
        result = np.array(mapped, copy=True, order="K")
        self._payload._check_unchanged(key)
        return result


def hoststore_gib() -> int:
    from tilestream import hoststore

    return hoststore.GIB


def _boundaries_from_cache(reader, metadata):
    """The domain's ``LateralBoundaries``, on the HOST and attached to nothing.

    ``restore_prepared_cache`` calls ``attach_lateral_boundaries(state, ...)``
    because it has a state to attach them to.  A streamed run does not: the
    tables are windowed per tile by
    :func:`woof.core.streaming.tile_boundary_tables` and each buffer holds
    only its own edge, which is the single-slot streaming bind.  So they are
    returned as they come off disk -- the whole series, host-side -- and the
    windowing decides what ever reaches the card.
    """
    from woof.ingest.prepared_cache import _reader_boundaries

    return _reader_boundaries(reader)


def default_slab_rows(nx: int, ny: int) -> int:
    """The existing column-bound fallback when no card budget is known.

    Known budgets price the established 64-row ceiling and shrink it before
    construction. An unread card retains the original column bound, so a
    wider requested grid cannot silently enlarge an unpriced allocation.
    """
    from woof.config import DEFAULT_COLUMN_CHUNK
    nx, ny = int(nx), int(ny)
    if nx < 1 or ny < 1:
        raise PreparedStoreError("prepared-store dimensions must be positive")
    return min(64, ny, max(1, DEFAULT_COLUMN_CHUNK // nx))


def maximum_slab_rows(nx: int, ny: int) -> int:
    """The established loader ceiling, before a known memory budget shrinks it."""
    if int(nx) < 1 or int(ny) < 1:
        raise PreparedStoreError("prepared-store dimensions must be positive")
    return min(64, int(ny))


def _plan_loader_slabs(ny: int, rows_per_slab: int, template_rows: int):
    """Partition every row once, retaining a fixed-height last template."""
    if not template_rows:
        return _plan_slabs(ny, rows_per_slab)
    tail = min(int(ny), max(1, int(template_rows)))
    if tail > int(rows_per_slab):
        raise PreparedStoreError(
            "the retained template exceeds the admitted slab height, "
            "which would allocate an unpriced device state")
    prefix = int(ny) - tail
    return (() if prefix == 0 else _plan_slabs(prefix, rows_per_slab)) + ((prefix, tail),)


def _store_boundaries(reader, metadata, boundary_source=None):
    """The host series, eager at a seal and lazy at a preparation head.

    Check the declared schedule without reading future intervals. A head
    contains only start-state arrays; opening its boundary arrays eagerly
    would wait for the whole preparation instead of the model's next seam.
    """
    lbc = metadata.get("lbc") or {}
    intervals = lbc.get("intervals")
    if boundary_source is None:
        if reader.header.get("status") == "HEAD" and intervals is not None:
            raise PreparedStoreError(
                "a prepared head's host store requires its streamed boundary "
                "source: later boundary arrays do not exist at the head")
        return (None if intervals is None else
                _boundaries_from_cache(reader, metadata))
    bounds = getattr(boundary_source.intervals, "bounds", ())
    if intervals is None or [list(map(float, row)) for row in bounds] != [
            [float(row["start_seconds"]), float(row["end_seconds"])]
            for row in intervals]:
        raise PreparedStoreError(
            "streamed boundary schedule differs from the prepared host "
            "store's declared intervals")
    for name in ("spec_bdy_width", "spec_zone", "relax_zone"):
        if int(getattr(boundary_source, name)) != int(lbc[name]):
            raise PreparedStoreError(
                f"streamed boundary {name} differs from the prepared host "
                "store, which would apply forcing to different edge cells")
    fields = getattr(boundary_source.intervals, "fields", None)
    if fields is not None and any(
            sorted(fields) != sorted(row["fields"]) for row in intervals):
        raise PreparedStoreError(
            "streamed boundary fields differ from the prepared host store, "
            "which would omit or replace its declared forcing carriers")
    return boundary_source


def store_from_prepared_cache(path, *, expected_identity, cfg, static,
                              landuse_attrs, grid, valid_time,
                              rows_per_slab: int | None = None,
                              budget_bytes: int | None = None,
                              verify_payload: bool = True,
                              center_lat=None, constant_glw_wm2=None,
                              inventory_fn=None, physics_initializer=None,
                              reader=None, boundary_source=None,
                              cache_payload=None,
                              retain_template_rows: int | None = None,
                              column_chunk: int | None = None,
                              log=print) -> PreparedStore:
    """Load a prepared cache into pinned host arrays, slab by slab.

    An omitted row count uses :func:`default_slab_rows`, whose bounded
    column count keeps a wide grid's initialization within the same memory
    contract its tiles are planned against. Explicit counts remain available
    to partition/parity probes and callers with their own priced policy.

    ``physics_initializer`` optionally initializes a caller's existing
    physics contract on each row window. It receives the same inputs as
    ``initialize_prepared_physics`` plus ``row_start`` and ``domain_rows``
    for windowing caller-owned setup fields. The default is unchanged.

    ``reader`` may be the checked start-state reader of a chained head.
    Its ``boundary_source`` is retained lazily on the host, so each tile
    takes a checked interval only when the model reaches that seam.

    ``cache_payload`` may retain a sealed direct preflight's already-hashed
    immutable mappings. Its reader, current header and file identities must
    still match. Other callers keep the ordinary verification path.

    No domain-shaped device array is ever allocated.  The peak device
    residency is one slab's state plus its physics; the peak HOST transient
    beyond the store itself is one full-domain array, because that is the
    unit ``PreparedCacheReader`` can verify against its digest.

    The budget guard is ``build_store_by_slabs``' and runs in the same place:
    the whole request is priced from the manifest the first slab reveals, and
    :func:`tilestream.hoststore.check_allocatable` refuses BEFORE the first
    ``alloc_pinned_array``.  A guard that fires on the way up has already
    taken most of what it is refusing, and pinned pages cannot be swapped.

    ``inventory_fn`` IS THE RULE THE TILE BUFFERS ARE BUILT BY, and it is a
    parameter rather than a literal so that the store and the buffers cannot
    be given two different ones.  It defaults to
    :func:`woof.core.streaming.streamed_store_inventory` -- the exact
    callable ``store_domain_builder`` and ``prepared_domain_builder`` hand
    :func:`woof.core.streaming.attach` -- and :class:`tilestream.driver.
    TiledRun` compares the two inventories key for key before it steps.
    Harvesting each slab with ``physics_inventory.carrier_inventory``
    instead cost the store exactly one carrier: ``scratch/refl_10cm`` is
    REBUILT scratch, excluded from the plain manifest BY CONSTRUCTION, so
    ``prime_lazy_carriers`` allocated it on every slab and nothing collected
    it.  MEASURED, before the first step: ``TiledRunError: tile state
    inventory [143] != store inventory [142] ... in TILE not in STORE:
    ['scratch/refl_10cm']``.
    """
    import cupy as cp

    from woof.core.streaming import (prime_lazy_carriers,
                                      streamed_store_inventory)
    from woof.ingest.hrrr_physics import initialize_prepared_physics
    from woof.ingest.prepared_cache import (CachedInitialResult,
                                             PreparedCacheReader)
    from tilestream import driver as tdriver
    from tilestream import hoststore
    from tilestream import physics_inventory as physinv
    from tilestream import realdata as _realdata

    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    if rows_per_slab is None:
        from woof.core.resident_admission import device_free_bytes
        rows_per_slab = (default_slab_rows(nx, ny) if device_free_bytes() is None
                         else maximum_slab_rows(nx, ny))
        if retain_template_rows is None:
            retain_template_rows = 1
    if cache_payload is not None:
        if boundary_source is not None:
            raise PreparedStoreError(
                "a streamed head cannot reuse a sealed retained payload")
        reader = cache_payload.require_binding(
            path, expected_identity, reader=reader)
    elif reader is None:
        reader = PreparedCacheReader(path, expected_identity=expected_identity)
    metadata = reader.header["metadata"]
    if boundary_source is not None or reader.header.get("status") == "HEAD":
        _store_boundaries(reader, metadata, boundary_source)

    from woof.core.grid import BaseState, VerticalCoord

    cached = (cache_payload if cache_payload is not None
              else _CachePayload(reader, verify=False, log=log))
    coord_values = dict(metadata["coord_scalars"])
    for name in metadata["coord_arrays"]:
        coord_values[name] = cached.verified_array(f"coord/{name}")
    coord = VerticalCoord(**coord_values)
    base_values = dict(metadata["base_scalars"])
    for name in metadata["base_arrays"]:
        base_values[name] = cached.verified_array(f"base/{name}")
    base = BaseState(**base_values)

    state_names = list(metadata["state_names"])
    surface_names = list(metadata.get("surface_fields", []))
    met_names = list(metadata["met_fields"])

    # Resolved once, here, so every slab is harvested by the same object the
    # buffers will be built by; see the docstring for the one carrier the
    # plain carrier inventory silently left out of the store.
    inventory_fn = (streamed_store_inventory() if inventory_fn is None
                    else inventory_fn)

    # THE SLAB'S DEVICE PEAK, PRICED BEFORE ITS CONSTRUCTOR.  The budget
    # guard below prices the pinned HOST store; nothing priced the slab
    # state, its physics and its carriers on the card, so an explicit
    # rows_per_slab=1024 at 1792x1024x55 mp=18 built a full-size state on the
    # card and stopped in a CUDA out-of-memory.  The slab is priced as the
    # tile planner prices the loader, and a slab too tall for the card's
    # free memory is loaded in the tallest slab that fits (the height
    # partitions the same cache bytes and changes no operand), or refused by
    # name when not even one row fits.
    from woof.core.resident_admission import admitted_slab_rows

    if retain_template_rows is not None:
        from woof.core.resident_admission import device_free_bytes
        if device_free_bytes() is None:
            rows_per_slab = min(rows_per_slab, default_slab_rows(nx, ny))

    rows_per_slab = admitted_slab_rows(
        cfg, min(max(1, int(rows_per_slab)), ny), p_top=float(base.p_top),
        log=log, column_chunk=column_chunk)

    slabs = _plan_loader_slabs(ny, rows_per_slab, retain_template_rows or 0)
    log(f"    {len(slabs)} row slabs of <= {rows_per_slab} rows, "
        f"{len(state_names)} prognostics + {len(surface_names)} surface "
        f"+ {len(met_names)} met from the cache")
    if verify_payload:
        cached.verify_all(log=log)

    hydrometeors = MappingProxyType(
        dict(metadata.get("hydrometeor_initialization", {})))
    cen_lat = (float(getattr(grid, "cen_lat", grid.ref_lat))
               if center_lat is None else float(center_lat))

    store: dict = {}
    geo_store: dict = {}
    scalars: dict = {}
    seen: set = set()
    seams: dict = {}
    allocated = False
    manifest = None
    template = None

    for index, (j0, rows) in enumerate(slabs):
        t0 = time.perf_counter()
        cfg_slab = replace(cfg, ny=int(rows))
        static_slab = _window_mapping(static, j0, rows, ny)
        surface_slab = SimpleNamespace(fields=MappingProxyType({
            name: _row_window(cached[f"surface/{name}"], j0, rows, ny)
            for name in surface_names}))
        met_slab = SimpleNamespace(fields=MappingProxyType({
            name: _row_window(cached[f"met/{name}"], j0, rows, ny)
            for name in met_names}))
        grid_slab = _realdata.window_grid(grid, 0, j0, nx, int(rows))

        # The base state is windowed exactly as the statics are, and for
        # exactly the same reason: with terrain it is domain-SHAPED, so a
        # slab handed the whole thing dies inside load_base's dev[...]
        # assignment.  Built ONCE per slab and used twice -- the state and
        # the physics must agree about which rows the base describes.
        base_slab = _slab_base(base, j0, rows, ny)

        state = _slab_state(
            cfg_slab, coord, base_slab, static_slab,
            lambda name: _row_window(cached[f"state/{name}"], j0, rows, ny),
            state_names)
        result = CachedInitialResult(
            state=state, coord=coord, base=base_slab,
            surface_pressure=_row_window(
                cached["result/surface_pressure"], j0, rows, ny),
            surface_qv=_row_window(
                cached["result/surface_qv"], j0, rows, ny),
            hydrometeor_initialization=hydrometeors)
        # center_lat is the DOMAIN's, passed rather than re-derived:
        # initialize_landuse reads it for the LANDUSE.TBL season, and a slab
        # that used its own centre would pick a different season for the same
        # column depending on which slab it landed in.
        physics_args = (
            result, cfg_slab, met_slab, surface_slab, static_slab,
            landuse_attrs, grid_slab, valid_time)
        physics_kwargs = dict(
            center_lat=cen_lat, constant_glw_wm2=constant_glw_wm2)
        if physics_initializer is None:
            initialize_prepared_physics(*physics_args, **physics_kwargs)
        else:
            physics_initializer(
                *physics_args, **physics_kwargs,
                row_start=j0, domain_rows=ny)
        _complete_slab_topo_wind(state, cfg, static, base, j0, rows, ny)
        # Do not retain result/state through this call tuple into the next slab.
        del physics_args
        # Exactly where the resident road primes the DOMAIN before attach
        # (prepared_domain_builder), and for the same reason: REFL_10CM and
        # the other lazily-allocated slots are absent from a state that has
        # never stepped, so a store built without them has nowhere for the
        # tiles' windows to join and the first due history frame is refused
        # an hour into an otherwise healthy forecast.  Primed per slab so
        # the full-domain store carries the slot the buffers carry.
        prime_lazy_carriers(state, cfg_slab)

        # The RUN's inventory rule, not the plain carrier one: the store and
        # the tile buffers have to be built by the same rule or TiledRun
        # refuses the pair, and REFL_10CM is the key the two disagreed on.
        # ``names=None`` because a slab reveals the manifest rather than
        # being asked for a known one -- streaming_inventory takes the
        # argument, carrier_inventory defaults it, and the call has to be
        # right for whichever rule a caller passes.
        inv = inventory_fn(state, None)
        ginv = tdriver.geography_inventory(state)
        if not allocated:
            allocated = True
            manifest = hoststore.manifest_from_arrays(
                inv, nz, int(rows), nx)
            geo_shapes = {
                name: (tuple(arr.shape[:-2])
                       + (ny + (int(arr.shape[-2]) - int(rows)),
                          int(arr.shape[-1])))
                for name, arr in ginv.items()}
            # THE WHOLE REQUEST, BEFORE THE FIRST BYTE OF IT IS TAKEN.
            planned = sum(spec.nbytes(nz, ny, nx) for spec in manifest)
            planned += sum(int(np.prod(shape))
                           * np.dtype(ginv[name].dtype).itemsize
                           for name, shape in geo_shapes.items())
            hoststore.check_allocatable(planned, budget_bytes=budget_bytes)
            log(f"    store: {planned / hoststore.GIB:.2f} GiB pinned "
                f"({len(manifest)} carriers + {len(geo_shapes)} geography), "
                "budget checked")
            for spec in manifest:
                store[spec.name] = hoststore.alloc_pinned_array(
                    spec.shape(nz, ny, nx), spec.dtype)
            domain_carriers = {spec.name for spec in manifest
                               if spec.fixed_shape is not None}
            for name, shape in geo_shapes.items():
                geo_store[name] = hoststore.alloc_pinned_array(
                    shape, np.dtype(ginv[name].dtype))
            scalars = physinv.carrier_scalars(state)

        for name, arr in inv.items():
            if name not in store:
                raise PreparedStoreError(
                    f"slab {index} produced carrier {name!r} that slab 0 did "
                    "not; the manifest is not slab-invariant")
            host = _as_host(arr)
            if name in domain_carriers:
                _hold_domain_carrier(store[name], host, name, index)
                seen.add(name)
                continue
            disagreed = _seam_disagreement(store[name], host, j0, rows)
            if disagreed:
                seams[name] = seams.get(name, 0) + disagreed
            _scatter_rows(store[name], host, j0)
            seen.add(name)
        for name, arr in ginv.items():
            host = _as_host(arr)
            disagreed = _seam_disagreement(geo_store[name], host, j0, rows)
            if disagreed:
                seams[name] = seams.get(name, 0) + disagreed
            _scatter_rows(geo_store[name], host, j0)

        if index == len(slabs) - 1:
            # Kept, not rebuilt: the last slab already carries the scheme
            # adapters, the eta table and the ndim < 2 setup arrays that
            # prepared_tile_state_factory reads off "the domain", and every
            # one of those is the same at any height.  See PreparedStore.
            template = state
        else:
            del state
        del inv, ginv, result
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        log(f"    slab {index + 1}/{len(slabs)} rows {j0}..{j0 + rows} "
            f"in {time.perf_counter() - t0:.1f}s")

    cached.close()
    if seams:
        # Not a warning.  A carrier whose shared face disagrees between the
        # two slabs that own it is wrong in nslabs - 1 rows by construction,
        # and it is wrong by an amount that looks like weather rather than
        # like a bug -- which is precisely why it must stop the load rather
        # than annotate it.  The fix is a y-halo on the offending carrier,
        # not a smaller tolerance.
        detail = ", ".join(f"{name} ({count} values)"
                           for name, count in sorted(seams.items()))
        raise PreparedStoreError(
            "these y-staggered carriers disagree with themselves on the "
            f"faces two slabs share: {detail}.  A y-staggered field's first "
            "row is the previous slab's last row, so a carrier the physics "
            "COMPUTES rather than copies is written there twice -- once as "
            "one slab's top edge and once as the next slab's bottom edge, "
            "and neither is the interior value the whole domain would have "
            "produced.  Slabbing this cache needs a y-halo for those "
            "carriers before its store can be trusted.")
    boundaries = _store_boundaries(reader, metadata, boundary_source)
    missing = tuple(sorted(set(store) - seen))
    receipt = MappingProxyType({
        "schema": "gpuwm-prepared-store-v1",
        "status": "LOADED",
        "path": str(reader.path.resolve()),
        "content_sha256": reader.content_sha256,
        "payload_bytes": reader.payload_bytes,
        "slabs": len(slabs),
        "rows_per_slab": int(rows_per_slab),
        "template_rows": int(slabs[-1][1]),
        "carriers": len(store),
        "geography": len(geo_store),
        "store_bytes": int(sum(a.nbytes for a in store.values())),
        "geography_bytes": int(sum(a.nbytes for a in geo_store.values())),
        "missing": missing,
    })
    return PreparedStore(
        store=store, geography=geo_store, scalars=scalars, template=template,
        # THE WHOLE-DOMAIN base, deliberately, and never a slab's: the
        # windowed copies exist only to build the slabs above, and a
        # consumer that reads bundle.base wants the domain's dry mass and
        # profiles -- publishing the last slab's would be a base state
        # silently one slab tall.
        coord=coord, base=base, boundaries=boundaries, missing=missing,
        receipt=receipt)
