"""Canonical, bounded device operands for streamed nest transactions.

The live store owns prognostics, and the geography inventory owns horizontal
setup. A slab template is only a source for vertical arrays and scalars.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class NestWindowSource:
    """Read exact canonical windows, preferring current owned rank slabs."""

    def __init__(self, state, *, read_only=False):
        self.state = state
        self.owner = getattr(state, "_streamed_domain", None)
        self.read_only = bool(read_only)
        run = getattr(self.owner, 'tiled_run', None)
        self._ranked = (run if self.read_only and getattr(run, 'ranked', False)
                        else None)
        reader = getattr(self.owner, "read_store", None)
        self._store = ((reader() if self.read_only and callable(reader)
                        else self.owner.store)
                       if self.owner is not None and self._ranked is None else None)
        if self.owner is None:
            from woof.core.streaming import domain_store
            self._store = domain_store(state)
        self.geography = (getattr(self.owner, "_geography", None) or {})
        self.template = (getattr(self.owner, "template_state", None) or state)
        self._geography_flags = None
        self._batch = {}
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0
        self.max_operand_bytes = 0
        self.max_host_scratch_bytes = 0

    @property
    def store(self):
        if self._store is None and self.owner is not None:
            reader = getattr(self.owner, 'read_store', None)
            self._store = (reader() if self.read_only and callable(reader)
                           else self.owner.store)
        return self._store

    def shape(self, name):
        """Query dimensions without reading a possibly stale host value."""
        key = name if '/' in name else f'state/{name}'
        if self._ranked is not None and key in self._ranked.raw_store:
            return self._ranked.raw_store[key].shape
        return self.array(name).shape

    @property
    def mass_carriers_ready(self):
        if self.owner is not None:
            return getattr(self.owner, "scalars", {}).get("wrf_mass_carriers_ready", True)
        return getattr(self.state, "_wrf_mass_carriers_ready", True)

    def array(self, name):
        if self.owner is not None or self.store is not None:
            if name in ("has_msf", "rotational"):
                if self._geography_flags is None:
                    from tilestream.driver import geography_scalars
                    self._geography_flags = geography_scalars(self.geography)
                return self._geography_flags[name]
            key = name if "/" in name else f"state/{name}"
            keys = self._ranked.raw_store if self._ranked is not None else self.store
            if key in keys:
                return self.store[key]
            key = f"setup/{name}"
            if key in self.geography:
                return self.geography[key]
            value = self._resident_array(name)
            if self.owner is not None and getattr(value, "ndim", 0) >= 2:
                raise RuntimeError(
                    f"canonical nest operand {name!r} is missing; a slab "
                    "template cannot supply horizontal domain data")
            return value
        return self._resident_array(name)

    def _resident_array(self, name):
        if name.startswith("scratch/"):
            return self.template.existing_scratch(name.split("/", 1)[1])
        if name.startswith("fields/"):
            driver = getattr(self.state, "physics", None)
            return getattr(driver, "fields", {}).get(name.split("/", 1)[1])
        return getattr(self.template, name, None)

    def device_into(self, name, window, out):
        """Pack a carrier into an already priced transaction arena view."""
        value = self.array(name)[(...,) + window]
        if value.shape != out.shape or value.dtype != out.dtype:
            raise ValueError(f"nest carrier {name} does not match its arena view")
        if isinstance(value, np.ndarray):
            out.set(np.ascontiguousarray(value))
            self.host_to_device_bytes += int(out.nbytes)
        else:
            out[...] = value
        self.max_operand_bytes = max(self.max_operand_bytes, int(out.nbytes))
        return out

    @staticmethod
    def request_key(name, window):
        return (name, None if window is None else tuple(
            (sl.start, sl.stop, sl.step) for sl in window))

    def coupled_requests(self, kind, window):
        ny, nx = self.shape("mup")
        y, x = window
        j0, j1 = max(0, y.start-1), min(ny, y.stop+1)
        i0, i1 = max(0, x.start-1), min(nx, x.stop+1)
        requests = [(name, (slice(j0, j1+int(name == "msfv")),
                            slice(i0, i1+int(name == "msfu"))))
                    for name in ("mup", "mub2d", "thb", "c1h", "c2h",
                                 "c1f", "c2f", "msft", "msfu", "msfv", "has_msf")]
        if kind != "mu":
            attr = {"t": "thp", "ph": "php"}.get(kind, kind)
            requests.append((attr, (slice(j0, j1+int(kind == "v")),
                                    slice(i0, i1+int(kind == "u")))))
        return requests

    def prepare(self, requests):
        """Stage each distinct FORCE operand once, before coupling starts."""
        requests = dict((self.request_key(name, window), (name, window))
                        for name, window in requests)
        ranked = {}
        if self._ranked is not None:
            selected = [(name if '/' in name else f'state/{name}', window)
                        for name, window in requests.values()]
            ranked = self._ranked.read_windows(selected)
        host = []
        for token, (name, window) in requests.items():
            key = name if '/' in name else f'state/{name}'
            value = ranked.get(self.request_key(key, window))
            if value is not None:
                self._batch[token] = value
                self.host_to_device_bytes += int(value.nbytes)
                self.device_to_host_bytes += int(value.nbytes)
                self.max_operand_bytes = max(self.max_operand_bytes, int(value.nbytes))
                continue
            value = self.array(name)
            if value is None or not hasattr(value, 'ndim') or np.isscalar(value):
                self._batch[token] = value
                continue
            if window is not None and value.ndim >= 2:
                value = value[(...,)+window]
            if isinstance(value, np.ndarray):
                host.append((token, value))
            else:
                self._batch[token] = self.device_array(value)
        if host:
            import cupy as cp
            from tilestream.gather import pinned_empty
            # One pinned arena and one upload for all canonical/setup values.
            offsets, size = [], 0
            for token, value in host:
                size = (size+value.dtype.itemsize-1)//value.dtype.itemsize*value.dtype.itemsize
                offsets.append(size)
                size += value.nbytes
            staging = pinned_empty((size,), np.uint8)
            for (_, value), offset in zip(host, offsets):
                np.ndarray(value.shape, value.dtype, staging, offset)[...] = value
            arena = cp.asarray(staging)
            # Keep staging alive until this upload completes, including when
            # the caller uses a non-default coordinator stream.
            cp.cuda.get_current_stream().synchronize()
            for (token, value), offset in zip(host, offsets):
                self._batch[token] = arena[offset:offset+value.nbytes].view(value.dtype).reshape(value.shape)
                self.host_to_device_bytes += int(value.nbytes)
                self.max_operand_bytes = max(self.max_operand_bytes, int(value.nbytes))

    def device(self, name, window=None):
        token = self.request_key(name, window)
        if token in self._batch:
            return self._batch[token]
        if self._ranked is not None and window is not None:
            key = name if '/' in name else f'state/{name}'
            value = self._ranked.read_window(key, window)
            if value is not None:
                self.host_to_device_bytes += int(value.nbytes)
                self.device_to_host_bytes += int(value.nbytes)
                self.max_operand_bytes = max(self.max_operand_bytes, int(value.nbytes))
                return value
        return self.device_array(self.array(name), window)

    def device_array(self, value, window=None):
        """Pack one bounded view, including a separately inventoried carrier."""
        if value is None or not hasattr(value, "ndim") or np.isscalar(value):
            return value
        import cupy as cp
        if window is not None and value.ndim >= 2:
            value = value[(...,) + window]
        if isinstance(value, np.ndarray):
            self.host_to_device_bytes += int(value.nbytes)
            result = cp.asarray(np.ascontiguousarray(value))
        else:
            result = cp.ascontiguousarray(value)
        self.max_operand_bytes = max(self.max_operand_bytes, int(result.nbytes))
        return result

    def write(self, name, window, value):
        if self.read_only:
            raise RuntimeError("a read-only nest source cannot accept feedback")
        import cupy as cp

        target = self.array(name)[(...,) + window]
        if isinstance(target, np.ndarray):
            # cp.asnumpy packs only this bounded result, never a full field.
            target[...] = cp.asnumpy(value)
            self.device_to_host_bytes += int(value.nbytes)
        else:
            target[...] = value
        if self.store is not None and self.owner is None:
            # Legacy published-store seam also keeps its resident reader live.
            resident = self._resident_array(name)
            if resident is not None and resident is not self.array(name):
                # Reuse the already transferred host window for a host mirror.
                resident[(...,) + window] = target if isinstance(resident, np.ndarray) else value

    def coupled(self, kind, window):
        """Run the unchanged coupling kernel on a one-cell mass halo.

        The extra mass row/column supplies face averages at artificial
        window edges. Only true physical edges use the kernel's clamp.
        """
        import cupy as cp
        from woof.ingest.lateral_bc import couple_nest_field

        ny, nx = self.shape("mup")
        y, x = window
        j0, j1 = max(0, y.start - 1), min(ny, y.stop + 1)
        i0, i1 = max(0, x.start - 1), min(nx, x.stop + 1)
        local = SimpleNamespace()
        for name in ("mup", "mub2d", "thb", "c1h", "c2h", "c1f", "c2f",
                     "msft", "msfu", "msfv", "has_msf"):
            extra_y, extra_x = (int(name == "msfv"), int(name == "msfu"))
            setattr(local, name, self.device(name, (
                slice(j0, j1 + extra_y), slice(i0, i1 + extra_x))))
        attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
        if kind != "mu":
            setattr(local, attr, self.device(attr, (
                slice(j0, j1 + int(kind == "v")),
                slice(i0, i1 + int(kind == "u")))))
        field = local.mup[None] if kind == "mu" else getattr(local, attr)
        out = cp.empty(field.shape, dtype=cp.float32)
        couple_nest_field(local, kind, out=out)
        from woof.wrf_exact import ENABLED as wrf_exact
        if wrf_exact and ((kind == "u" and i0 == 0) or (kind == "v" and j0 == 0)):
            from woof.core.nest_round_trip import couple_halo_faces
            halo = self.device("scratch/wrf_exact_mu_halo", (slice(j0, j1), slice(i0, i1)))
            local.existing_scratch = lambda _slot: halo
            local._wrf_mass_carriers_ready = self.mass_carriers_ready
            couple_halo_faces(local, kind, out)
        return cp.ascontiguousarray(out[
            :, y.start - j0:y.stop - j0, x.start - i0:x.stop - i0])

    def raw(self, kind, window):
        """Read WRF's uncoupled feedback spelling, including theta - 300."""
        attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
        value = self.device(attr, window)
        from woof.wrf_exact import ENABLED as wrf_exact
        if kind == "t" and not wrf_exact:
            value = value.copy()
            thb = self.device("thb", window)
            value += thb if thb.ndim == 3 else thb[:, None, None]
            value -= np.float32(300.0)
        return value

    def transitioned(self, contract, kind, window):
        """Diagnose one exact donor window with the existing edge kernel."""
        import cupy as cp
        from woof.core.microphysics_transition import (
            edge_parent_planes, launch_microphysics_edge_field,
            transition_source_field_shape, transition_source_window)

        if self._ranked is not None or self._batch:
            # The same column-local kernel and packing helper, with already
            # bounded inputs. No whole canonical field is materialized.
            ny, nx = self.shape('qv')[-2:]
            if not isinstance(window, tuple) or len(window) != 2:
                raise ValueError('transition window must be two bounded slices')
            for sl, extent in zip(window, (ny, nx)):
                if (not isinstance(sl, slice) or sl.step not in (None, 1)
                        or sl.start is None or sl.stop is None
                        or not 0 <= sl.start < sl.stop <= extent):
                    raise ValueError('transition window is outside the parent')
            source = SimpleNamespace(**{
                name: self.device(name, window) for name in
                edge_parent_planes() + ('thb', 'c1h', 'c2h')})
            local = transition_source_window(source, (
                slice(0, window[0].stop-window[0].start),
                slice(0, window[1].stop-window[1].start)))
            out = cp.empty(transition_source_field_shape(local, kind), dtype=cp.float32)
            return launch_microphysics_edge_field(
                contract, local, kind, out=out, coupled=True)
        source = SimpleNamespace(**{
            name: self.array(name) for name in
            edge_parent_planes() + ("thb", "c1h", "c2h", "use_theta_m")})
        local = transition_source_window(source, window)
        for name, value in vars(source).items():
            if isinstance(value, np.ndarray):
                selected = value[(...,) + window] if value.ndim >= 2 else value
                self.host_to_device_bytes += int(selected.nbytes)
            prepared = getattr(local, name, None)
            if hasattr(prepared, "nbytes"):
                self.max_operand_bytes = max(self.max_operand_bytes,
                                             int(prepared.nbytes))
        out = cp.empty(transition_source_field_shape(local, kind), dtype=cp.float32)
        return launch_microphysics_edge_field(
            contract, local, kind, out=out, coupled=True)


def streamed_chunk_shape(state):
    """Use the admitted tile dimensions, never an unpriced full field."""
    owner = getattr(state, "_streamed_domain", None)
    decision = getattr(owner, "decision", None)
    ny, nx = (getattr(decision, "tile_ny", None),
              getattr(decision, "tile_nx", None))
    if ny is None or nx is None or int(ny) < 1 or int(nx) < 1:
        raise RuntimeError("bounded nest coupling needs admitted tile dimensions")
    return int(ny), int(nx)


def refresh_canonical_base(source):
    """Refresh base-derived carriers through bounded CUDA operands.

    This is a once-per-position setup transaction, not a second full-domain
    device state. The next tile gather consumes this geography and rebuilds
    its own height cache. Serialized perturbations remain unchanged.
    """
    import cupy as cp
    ny, nx = source.array("mup").shape
    for window in rectangle_windows((slice(0, ny), slice(0, nx)),
                                    streamed_chunk_shape(source.state)):
        phb = source.device("phb", window)
        source.write("dphb_resid", window,
                     (cp.diff(phb.astype(cp.float64), axis=0)
                      - cp.diff(phb, axis=0).astype(cp.float64)).astype(cp.float32))
        if "state/p_perturbation" in source.store:
            pressure = cp.add(source.device("pb", window), source.device("p_perturbation", window))
            source.write("p", window, pressure)
            volume = cp.add(source.device("al", window), source.device("alb", window))
            source.write("alt", window, volume)
    source.owner._statics_setup = None


def boundary_windows(reg, width, chunk_shape):
    """Yield side, exact child rectangle and rolling-table destination."""
    sy, sx = chunk_shape
    for side in ("west", "east"):
        x = slice(0, width) if side == "west" else slice(reg.nxc-width, reg.nxc)
        for j in range(0, reg.nyc, sy):
            y = slice(j, min(j + sy, reg.nyc))
            yield side, (y, x), (slice(None), y, slice(None))
    for side in ("south", "north"):
        y = slice(0, width) if side == "south" else slice(reg.nyc-width, reg.nyc)
        for i in range(0, reg.nxc, sx):
            x = slice(i, min(i + sx, reg.nxc))
            yield side, (y, x), (slice(None), slice(None), x)


def rectangle_windows(window, chunk_shape):
    y, x = window
    sy, sx = chunk_shape
    for j in range(y.start, y.stop, sy):
        for i in range(x.start, x.stop, sx):
            yield (slice(j, min(j+sy, y.stop)), slice(i, min(i+sx, x.stop)))


def smooth_canonical_parent(source, kind, reg, *, smooth_option, chunk_shape):
    """Use the original two-stage CUDA smoother with a host J-pass owner.

    Every J output exists before any I output overwrites the source. The
    single host scratch rectangle retains the original one-cell outer ring.
    This is an explicit additional host allocation, reported on ``source``;
    device operands remain at most one tile plus that ring.
    """
    import cupy as cp
    from woof.core.nest_interp import (
        SMDSM_XNU, _kernel, _launch, smoother_parent_window)

    i0, j0, niw, njw = smoother_parent_window(reg)
    if smooth_option == 0 or niw <= 0 or njw <= 0:
        return
    if smooth_option not in (1, 2):
        raise ValueError("smooth_option must be 0, 1 or 2")
    attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(kind, kind)
    field = source.array(attr)
    nz = field.shape[0] if field.ndim == 3 else 1
    snapshot = np.empty((nz, njw+2, niw+2), dtype=np.float32)
    source.max_host_scratch_bytes = max(source.max_host_scratch_bytes, snapshot.nbytes)
    output = (slice(j0, j0+njw), slice(i0, i0+niw))
    ring = (slice(j0-1, j0+njw+1), slice(i0-1, i0+niw+1))
    mode = np.int32(smooth_option == 2)
    passes = SMDSM_XNU if smooth_option == 2 else (np.float32(0.0),)
    for xnu in passes:
        # Preserve the exact global ring. A device store is staged in the
        # same bounded chunks; a host store needs no device roundtrip.
        if isinstance(field, np.ndarray):
            snapshot[...] = field[(...,)+ring]
        else:
            for win in rectangle_windows(ring, chunk_shape):
                y, x = win
                value = source.device(attr, win)
                snapshot[:, y.start-j0+1:y.stop-j0+1,
                         x.start-i0+1:x.stop-i0+1] = cp.asnumpy(value)
                source.device_to_host_bytes += value.nbytes
        for win in rectangle_windows(output, chunk_shape):
            y, x = win
            ny, nx = y.stop-y.start, x.stop-x.start
            halo = (slice(y.start-1, y.stop+1), slice(x.start-1, x.stop+1))
            values = source.device(attr, halo)
            work = cp.empty((nz, ny+2, nx+2), dtype=cp.float32)
            dims = tuple(np.int32(v) for v in (1, 1, nx, ny, nz, ny+2, nx+2))
            _launch(_kernel("nest_smooth_j"), work.size,
                    (values, work, mode, xnu, *dims))
            block = cp.ascontiguousarray(work[:, 1:-1, 1:-1])
            snapshot[:, y.start-j0+1:y.stop-j0+1,
                     x.start-i0+1:x.stop-i0+1] = cp.asnumpy(block)
            source.device_to_host_bytes += block.nbytes
        for win in rectangle_windows(output, chunk_shape):
            y, x = win
            ny, nx = y.stop-y.start, x.stop-x.start
            values = source.device_array(snapshot[:, y.start-j0:y.stop-j0+2,
                                                   x.start-i0:x.stop-i0+2])
            work = cp.empty(values.shape, dtype=cp.float32)
            dims = tuple(np.int32(v) for v in (1, 1, nx, ny, nz, ny+2, nx+2))
            _launch(_kernel("nest_smooth_i"), nz*ny*nx,
                    (work, values, mode, xnu, *dims))
            result = work[:, 1:-1, 1:-1]
            source.write(attr, win, result[0] if field.ndim == 2 else result)



def couple_round_trip_canonical(source, run, *, chunk_shape, emit=None):
    """Apply the unchanged strict operator to disjoint canonical windows.

    One mass halo supplies u/v face neighbours; only physical domain edges
    take the original operator's one-sided formulas. Prognostics are point
    local, so earlier writes cannot change another window's operands.
    """
    from woof.core.nest_round_trip import couple_round_trip, round_trip_targets
    from woof.core.nest_fields import nest_field_kinds
    ny, nx = source.array("mup").shape
    attrs = [{"t": "thp", "ph": "php"}.get(k, k)
             for k in nest_field_kinds(run) if k != "mu"]
    for win in rectangle_windows((slice(0, ny), slice(0, nx)), chunk_shape):
        y, x = win
        j0, j1 = max(0, y.start-1), min(ny, y.stop+1)
        i0, i1 = max(0, x.start-1), min(nx, x.stop+1)
        halo = (slice(j0, j1), slice(i0, i1))
        local = SimpleNamespace(**{name: source.device(name, halo) for name in
            ("mup", "mub2d", "c1h", "c2h", "c1f", "c2f", "has_msf", "msft")})
        local.msfu = source.device("msfu", (slice(j0, j1), slice(i0, i1+1)))
        local.msfv = source.device("msfv", (slice(j0, j1+1), slice(i0, i1)))
        mass_halo = source.device("scratch/wrf_exact_mu_halo", halo)
        local.existing_scratch = lambda _slot: mass_halo
        local._wrf_mass_carriers_ready = source.mass_carriers_ready
        for name in attrs:
            value = source.array(name)
            if value is not None:
                setattr(local, name, source.device(name, (
                    slice(j0, j1+int(name == "v")),
                    slice(i0, i1+int(name == "u")))))
        # Always copy: a resident operand may already be contiguous. The
        # crop owns the only writes, including the closing physical faces.
        targets = {name: value.copy() for name, value in round_trip_targets(local, run).items()}
        couple_round_trip(local, run, targets)
        for name, value in targets.items():
            jy = y.stop + int(name == "v" and y.stop == ny)
            ix = x.stop + int(name == "u" and x.stop == nx)
            (source.write if emit is None else emit)(name, (slice(y.start, jy), slice(x.start, ix)),
                value[..., y.start-j0:jy-j0, x.start-i0:ix-i0])


def radiation_canonical_frame(source, fields, *, chunk_shape, destinations):
    """Present the strict held heating exactly as the resident writer does.

    Only bounded device windows do arithmetic. Canonical diagnostics and
    producer words remain unchanged by history publication.
    """
    if "RTHRATEN" not in fields or source.array("diag/_wrf_rthraten") is None:
        return fields
    import cupy as cp
    driver = getattr(source.template, "physics", None)
    coupled = (getattr(source.owner, "scalars", {}).get(
        "wrf_rthraten_coupled", getattr(driver, "_wrf_rthraten_coupled", True)))
    name = "RTHRATEN"
    if name not in destinations:
        destinations[name] = np.empty_like(fields[name])
    target = destinations[name]
    ny, nx = source.array("mup").shape
    c1h, c2h = source.device("c1h"), source.device("c2h")
    for win in rectangle_windows((slice(0, ny), slice(0, nx)), chunk_shape):
        heating = source.device("diag/_wrf_rthraten", win)
        if coupled:
            mut = (source.device("scratch/wrf_exact_muts", win)
                   if source.mass_carriers_ready else None)
            if mut is None:
                mut = source.device("mub2d", win) + source.device("mup", win)
            mass = c1h[:, None, None] * mut[None] + c2h[:, None, None]
            heating = heating / mass
        target[(...,) + win] = cp.asnumpy(heating)
    result = dict(fields)
    result[name] = target
    return result


def round_tripped_canonical_frame(source, run, fields, plan, *, chunk_shape,
                                  destinations):
    """Write the next FORCE's strict history words without mutating state.

    As on the resident path, T retains th_phy_m_t0. The field plan supplies
    the carrier aliases; THM is the sole derived spelling of a forced word.
    Device arithmetic stays bounded and uses the production round trip.
    """
    import cupy as cp
    from woof.core.nest_fields import nest_field_kinds
    attrs = {{"t": "thp", "ph": "php"}.get(k, k)
             for k in nest_field_kinds(run) if k != "mu"}
    names = {}
    result = dict(fields)
    for name, value in fields.items():
        key = plan.origin.get(name, "")
        attr = "thp" if name == "THM" else key.removeprefix("state/")
        if name == "T" or attr not in attrs:
            continue
        names.setdefault(attr, []).append(name)
        if name not in destinations:
            destinations[name] = np.empty_like(value)
        result[name] = destinations[name]
    def emit(attr, window, value):
        if attr in names:
            host = cp.asnumpy(value)
            for name in names[attr]:
                destinations[name][(...,) + window] = host
    couple_round_trip_canonical(source, run, chunk_shape=chunk_shape, emit=emit)
    return result


def diagnose_canonical_parent(source, window, *, hypsometric_opt, chunk_shape):
    """Recompute changed columns using the unchanged local diagnostics kernel."""
    import cupy as cp
    from woof.core.diagnostics import update_diagnostics

    names = ("thp", "php", "mup", "qv", "thb", "phb", "dphb_resid", "alb", "pb",
             "rdnw", "c1h", "c2h", "c3h", "c4h", "c3f", "c4f",
             "dc3f", "dc4f", "mub2d", "p_top", "use_theta_m")
    nz = source.array("p").shape[0]
    for win in rectangle_windows(window, chunk_shape):
        y, x = win
        local = SimpleNamespace(**{name: source.device(name, win) for name in names})
        local.use_theta_m = int(local.use_theta_m or 0)
        for name in ("p", "al", "alt"):
            setattr(local, name, cp.empty((nz, y.stop-y.start, x.stop-x.start),
                                         dtype=cp.float32))
        update_diagnostics(local, hypsometric_opt)
        for name in ("p", "al", "alt"):
            source.write(name, win, getattr(local, name))
        if getattr(local, "p_perturbation", None) is not None:
            source.write("p_perturbation", win, local.p_perturbation)
