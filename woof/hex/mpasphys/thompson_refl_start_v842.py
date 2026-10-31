"""Thompson's own 10 cm reflectivity of the start state, for the held f000 frame (MPAS-A suites).

MPAS-A forms ``refl10cm_max``, ``refl10cm_1km`` and ``refl10cm_1km_max`` only inside a microphysics
diagnostics step (mpas_atmphys_driver_microphysics.F:738-792 compute_radar_reflectivity), from the state
that step leaves.  The start frame (f000) is written before any physics has run, so it carries the init's
placeholder: zero everywhere, even when the init holds the first guess's own rain, snow and graupel
(rw_mpas_init ``--first-guess-hydrometeors``, the init door's default).

THE BREAKAGE THIS ENDS (hex-meshes-sources review, 2026-10-29): the default hex route (convection_permitting,
mp_thompson) started from HRRR 2026-10-28 12Z with HRRR's condensate in its state (qr up to 0.86 g/kg) and
its f000 composite reflectivity map still read "No values to plot"; the polygon f000 map
(the final polygon maps, 16:08Z) did the same.

This runs the same ``mpas_thompson_radar`` kernel (calc_refl10cm, the column maximum and the 1 km
interpolation) the model's diagnostics step launches, from the same module (thompson_v842._module), on the
frame's captured start state, on a side thread and card stream while step 1 runs.  Temperature is
theta * (p / p0) ** (rgas / cp), as MPAS's exner.  Only the f000 diagnostics frame changes; the model state is
untouched.  The MPAS suite route holds the frame in correct mode only (streams_output_v842
first_frame_from_first_call), so exact mode writes native MPAS-A's f000 (zero) bit for bit.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import numpy as np

RGAS, CP_DRY, P0 = 287.0, 1004.5, 1.0e5

#: The microphysics whose CUDA slot forms reflectivity with ``thompson_v842.launch_radar``
#: (thompson_aero_v842 launches the same kernel through it).
SCHEMES = ('mp_thompson', 'mp_thompson_aerosols')

#: The diagnostics stream's reflectivity fields (MPAS-A v8.4.2 Registry ``diagnostics`` stream), plus the 3-D
#: ``refl10cm`` where a stream carries it.
START_TARGETS = ('refl10cm', 'refl10cm_max', 'refl10cm_1km', 'refl10cm_1km_max')

NEED = ('qv', 'qc', 'qr', 'nr', 'qs', 'qg', 'theta')


def device_math_source() -> str:
    """The certified mp_thompson device maths text (resources_build MATHS), the text the model's slot reads."""
    from . import resources_build
    return resources_build.thompson_maths(resources_build._vendored_maths())


def radar(state: Mapping[str, Any], zgrid: Any | None, *, maths: str, cp: Any) -> dict[str, Any]:
    """``refl10cm`` ``(nz, ncells)``, ``refl10cm_max``, ``refl10cm_1km`` and ``refl10cm_1km_max`` (dBZ, device
    float32) of one state through ``thompson_v842.launch_radar``.  ``state``: ``qv qc qr nr qs qg theta p``
    ``(nz, ncells)``; ``zgrid`` ``(nz + 1, ncells)`` or None (then no 1 km fields)."""
    from . import thompson_v842 as thompson

    f = {k: cp.ascontiguousarray(cp.asarray(state[k], dtype=cp.float32)) for k in (*NEED, 'p')}
    nz, ncells = f['qv'].shape
    for name, value in f.items():
        if value.shape != (nz, ncells):
            raise ValueError(f'Thompson start reflectivity input {name} is {value.shape}, not {(nz, ncells)}')
    pii = ((f['p'].astype(cp.float64) / P0) ** (RGAS / CP_DRY)).astype(cp.float32)
    if zgrid is not None:
        z = cp.asarray(zgrid, dtype=cp.float32)
        if z.shape != (nz + 1, ncells):
            raise ValueError(f'zgrid is {z.shape}, not {(nz + 1, ncells)}')
    else:
        # Column-index heights, 500 m apart: they bracket 1 km so the kernel's interpolation stays finite; the
        # 1 km fields are not written without the frame's own zgrid.
        z = cp.broadcast_to(cp.arange(nz + 1, dtype=cp.float32)[:, None] * cp.float32(500.0), (nz + 1, ncells))
    fields = {'qv': f['qv'], 'qc': f['qc'], 'qr': f['qr'], 'nr': f['nr'], 'qs': f['qs'], 'qg': f['qg'],
              'th': f['theta'], 'pii': cp.ascontiguousarray(pii), 'p': f['p'],
              'z': cp.ascontiguousarray(z[:nz]), 'dz': cp.ascontiguousarray(z[1:nz + 1] - z[:nz])}
    out = {'refl10cm': cp.empty((nz, ncells), cp.float32), 'refl10cm_max': cp.empty((ncells,), cp.float32),
           'refl10cm_1km': cp.empty((ncells,), cp.float32),
           # The kernel keeps a running maximum: from the floor, it is this state's own 1 km value.
           'refl10cm_1km_max': cp.full((ncells,), -35.0, dtype=cp.float32)}
    thompson.launch_radar(fields, out, device_math_source=maths)
    return out


def radar_staged(state: Mapping[str, np.ndarray], zgrid: np.ndarray | None, *, maths: str, cp: Any,
                 stream: Any = None, columns: int | None = None) -> dict[str, np.ndarray]:
    """:func:`radar` of a HOST state one slab of ``columns`` columns at a time; host float32 results, the same words
    as one whole-state call (every column is formed from its own words only).

    THE BREAKAGE THIS PREVENTS (the same as tempo_refl_v303.refl10cm_staged): on a same-host multi-card run the
    f000 frame is the GLOBAL frame held on rank 0, and a whole-state copy would land on rank 0's card at its step-1
    peak; slabs bound the card's share at any mesh size."""
    from .rrfs.tempo_refl_v303 import STAGE_COLUMNS
    step = max(1, int(columns or STAGE_COLUMNS))
    nz, ncells = state['qv'].shape
    out = {'refl10cm': np.empty((nz, ncells), np.float32)}
    out.update({k: np.empty(ncells, np.float32) for k in START_TARGETS[1:]})
    for first in range(0, ncells, step):
        cols = slice(first, min(ncells, first + step))
        slab = {k: np.ascontiguousarray(v[:, cols]) for k, v in state.items()}
        part = radar(slab, None if zgrid is None else np.ascontiguousarray(zgrid[:, cols]), maths=maths, cp=cp)
        for name, value in part.items():
            out[name][..., cols] = cp.asnumpy(value, stream=stream)
        if stream is not None:
            stream.synchronize()
        del part, slab
    return out


class StartFrameReflectivity:
    """The held f000 frame's Thompson reflectivity fields, formed from its own start state beside step 1.

    ``NEED``, ``THREAD``, ``NEEDS_ZGRID`` and :meth:`_form` are what another scheme's start frame changes
    (wsm6_refl_start_v842)."""

    NEED = NEED
    THREAD = 'f000-reflectivity-thompson'
    NEEDS_ZGRID = False

    def __init__(self, snapshot: Mapping[str, Any], *, scheme: str, device: int) -> None:
        from .rrfs.tempo_refl_v303 import _logical, _placeholder
        self.targets = [n for n in START_TARGETS if n in snapshot and _placeholder(snapshot[n])]
        self.record: dict[str, Any] = dict(operator=f'{scheme} calc_refl10cm of the start state '
                                           '(thompson_v842 mpas_thompson_radar)', fields={})
        self.result: dict[str, np.ndarray] | None = None
        self.error: BaseException | None = None
        self.thread = None
        if not self.targets:
            self.record['applied'] = False
            self.record['reason'] = 'the frame holds no reflectivity placeholder (every field already varies)'
            return
        missing = [n for n in self.NEED if n not in snapshot]
        if 'pressure' not in snapshot and not all(n in snapshot for n in ('pressure_p', 'pressure_base')):
            missing.append('pressure')
        if self.NEEDS_ZGRID and 'zgrid' not in snapshot:
            missing.append('zgrid')
        if missing:
            self.targets = []
            self.record['applied'] = False
            self.record['reason'] = f'the start frame carries no {missing}'
            return
        if 'zgrid' not in snapshot:
            for name in ('refl10cm_1km', 'refl10cm_1km_max'):
                if name in self.targets:
                    self.targets.remove(name)
            self.record['refl10cm_1km'] = 'not formed: the frame carries no zgrid'
        state = {n: _logical(snapshot[n])[0] for n in self.NEED}
        if 'pressure' in snapshot:
            state['p'] = _logical(snapshot['pressure'])[0]
        else:
            state['p'] = (_logical(snapshot['pressure_p'])[0] + _logical(snapshot['pressure_base'])[0]).astype(np.float32)
        zgrid = _logical(snapshot['zgrid'])[0] if 'zgrid' in snapshot else None
        self._args = (state, zgrid, int(device))
        self.thread = threading.Thread(target=self._run, name=self.THREAD, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            import cupy as cp
            state, zgrid, device = self._args
            started = time.time()
            with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True) as stream:
                # In slabs: on a same-host multi-card run this is the global frame on rank 0's card, beside step 1.
                host = self._form(state, zgrid, cp, stream)
            self.result = host
            self.record['seconds'] = round(time.time() - started, 2)
        except BaseException as error:  # reported in the receipt; never fails the forecast
            self.error = error

    def _form(self, state, zgrid, cp, stream) -> dict[str, np.ndarray]:
        """The start state's reflectivity fields, host float32 (Thompson: mpas_thompson_radar in column slabs)."""
        from .rrfs.tempo_refl_v303 import STAGE_COLUMNS
        self.record['card_slab_columns'] = STAGE_COLUMNS
        return radar_staged(state, zgrid, maths=device_math_source(), cp=cp, stream=stream)

    def apply(self, snapshot: dict[str, Any], field_type: Any) -> dict[str, Any]:
        """Join the side thread and write the formed fields into the held frame; return the record."""
        from .rrfs.tempo_refl_v303 import _restore
        if self.thread is None:
            return self.record
        waited = time.time()
        self.thread.join()
        self.record['step1_wait_seconds'] = round(time.time() - waited, 3)
        if self.error is not None or self.result is None:
            self.record['applied'] = False
            self.record['reason'] = f'start reflectivity failed: {self.error!r}'
            return self.record
        for name in self.targets:
            item = snapshot[name]
            value = self.result[name]
            snapshot[name] = field_type(_restore(value, item), item.dimensions, item.attrs)
            self.record['fields'][name] = dict(minimum=float(value.min()), maximum=float(value.max()),
                                               cells_ge_20=int((value >= 20).sum()))
        self.record['applied'] = True
        return self.record


def start_frame_reflectivity(configs: Mapping[str, Any], snapshot: Mapping[str, Any], *,
                             device: int) -> StartFrameReflectivity | None:
    """The f000 reflectivity completion for this run's microphysics (None: not a Thompson run)."""
    scheme = str(configs.get('config_microp_scheme', '')).strip()
    if scheme not in SCHEMES:
        return None
    return StartFrameReflectivity(snapshot, scheme=scheme, device=device)
