"""HAILCAST hail size and the hail size-distribution maxima on hex, from WOOF-ArWen's GPU code.

What it writes.  Diagnostics-stream fields with HRRR's (WRF's) names and units, each the maximum since the
previous diagnostics frame at a whole hour (HRRR resets them at every hourly output; with hourly diagnostics
frames these are HRRR's hourly maxima, and a frame between whole hours carries the maximum so far that hour):

* ``HAILCAST_DIAM_MAX`` / ``HAILCAST_DIAM_MEAN`` / ``HAILCAST_DIAM_STD`` (mm): the HAILCAST stone model's largest,
  mean and spread of its five embryos' surviving stones; ``HAILCAST_DHAIL1`` .. ``HAILCAST_DHAIL5`` (mm) each
  embryo's.  Computed where the mesh is finer than ``FINE_M`` (WRF/HRRR run HAILCAST on grids of 4 km or less).
* ``HAIL_MAXK1`` / ``HAIL_MAX2D`` (m): the largest hail of the microphysics' own size distribution at the lowest
  level and in the column.  On a route whose microphysics forms them (RRFS TEMPO: ``max_hail_diameter_sfc`` and
  ``max_hail_diameter_column``, mm) they are those words, folded every step; on the MPAS-A Thompson and WSM6
  routes they come from WOOF-ArWen's ``hail_psd`` with the scheme's row (woof.core.hail.PSD_ROWS 8 and 6).

The GPU code is WOOF-ArWen's, shared, not rewritten: the stone model kernels ``hailcast_columns_wrf39`` (operational
HRRR v4.1.21, the default) and ``hailcast_columns`` (WRF 4.6.1) and ``hail_psd`` are compiled from the installed
woof's own ``woof/core/kernels/hail.cu`` and headers through woof's loader (its flags: no FMA contraction).
Those are the bytes whose columns match WRF's and HRRR's Fortran (woof fix/hailcast; the 94 real storm columns
in ``tests/data/oracles/hailcast``); ``SHARED_SHA256`` pins them.  What is hex's own is the MPAS side: the column
inputs formed from the MPAS state (temperature from theta_m and exner, total pressure, moist density
rho_zz*zz*(1+qv), mass-level heights from zgrid), the updraft-duration count over a cell and its cellsOnCell
neighbours (WRF's 3x3 neighbourhood on the hexagonal mesh), the cadence (every ``HAILDT`` s of model time, WRF's
haildt) and the window.

Words.  ``WOOF_HEX_HAILCAST``: unset or ``auto`` = on where the mesh has cells finer than 4 km (nothing is built
on a coarser mesh); ``0``/``off`` = none of these fields; ``1``/``on`` = HAILCAST in every cell whatever its size.
``WOOF_HEX_HAILCAST_VERSION``: ``wrf_39_noaa`` (default, HRRR v4's stone model) or ``wrf_461``.

Multi-rank runs (same-host thread or process ranks, ``halo`` set).  Each rank counts its owned cells; the 3x3
neighbourhood reads the previous step's mask and count of halo cells from their owners (one halo exchange of the two
words each fold, before the count), so a rank's count is the one-rank run's.  The diagnostics words are gathered to
rank 0 at every diagnostics frame (only the cells holding hail travel) and written there on the global mesh.

Seals.  The updraft mask and count, the hour's maxima, the run maximum and the last fold's time ride the seal with the
output diagnostics' running accumulators (``accumulator_state`` / ``resume_accumulators``), so a run resumed from a
seal folds on exactly as the run that never stopped.  THE BREAKAGE THIS FIXES: a resumed run started its count at zero
and made no stone for its first 15 minutes, and its first hour's maxima lost what fell before the seal.
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ENV = "WOOF_HEX_HAILCAST"
ENV_VERSION = "WOOF_HEX_HAILCAST_VERSION"
#: Cells whose mean edge spacing is at most this run HAILCAST (WRF hailcast_opt=-1: dx, dy <= 4 km).
FINE_M = 4000.0
#: WRF haildt (s): the stone model runs once per this much model time.
HAILDT = 60.0
#: WRF/HRRR: a column is integrated once an updraft of 10 m/s has persisted longer than this (s).
DURATION_S = 900.0
#: hailcast_version -> woof kernel (woof.core.hail.HAILCAST_KERNELS).
KERNELS = {"wrf_39_noaa": "hailcast_columns_wrf39", "wrf_461": "hailcast_columns"}
DEFAULT_VERSION = "wrf_39_noaa"
#: The woof files whose stone model and PSD were checked against WRF's and HRRR's Fortran (woof fix/hailcast
#: 7bb4db932d; the same bytes in gpuwm 2.9.12, 3.0.3 and 3.0.4).  THE BREAKAGE THIS PREVENTS: an installed woof
#: whose stone model differs would write hail sizes no Fortran comparison stands behind, under the same names.
SHARED_SHA256 = {
    "hail.cu": "eaa6302d020f5c8b",
    "hailcast_support.cuh": "a6b82a22fc3d4a80",
    "hailcast_wrf.cuh": "4cf483057a46b815",
    "hailcast_wrf39.cuh": "1057e2bde6bebbc2",
}
HAILCAST_FIELDS = ("HAILCAST_DIAM_MAX", "HAILCAST_DIAM_MEAN", "HAILCAST_DIAM_STD",
                   *(f"HAILCAST_DHAIL{i}" for i in range(1, 6)))
PSD_FIELDS = ("HAIL_MAXK1", "HAIL_MAX2D")
#: The prefix of HAILCAST's words among a seal's output accumulators (Hailcast.carry).
CARRY = "hailcast:"
UNITS = {**{name: "mm" for name in HAILCAST_FIELDS}, **{name: "m" for name in PSD_FIELDS}}
LONG_NAMES = {
    "HAILCAST_DIAM_MAX": "HAILCAST maximum surviving hail diameter, maximum since the last whole-hour diagnostics frame",
    "HAILCAST_DIAM_MEAN": "HAILCAST mean surviving hail diameter of the five embryos",
    "HAILCAST_DIAM_STD": "HAILCAST standard deviation of the five embryos' hail diameters",
    "HAIL_MAXK1": "maximum hail diameter of the size distribution at the lowest model level",
    "HAIL_MAX2D": "maximum hail diameter of the size distribution in the column",
}
#: MPAS microphysics -> woof.core.hail.PSD_ROWS row (graupel density kg m-3, intercept m-4, Thompson's n0).
PSD_ROWS = {
    "mp_thompson": {"density_g": 500., "thompson": 1},
    "mp_thompson_aerosols": {"density_g": 500., "thompson": 1},
    "mp_wsm6": {"density_g": 500., "intercept": 4.e6},
}

# Hex's own kernels: MPAS state -> the column words ArWen's stone model reads (WRF's hail_prepare / hail_duration
# on the MPAS layout).  Pool arrays are (level, cell) with a row stride in words; compact outputs are (level, K).
_SOURCE = r"""
extern "C" __global__ void hex_hail_mask(
    const int n, const int nf, const float* __restrict__ w, const long long sw, float* __restrict__ strong,
    float* __restrict__ wmax) {
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= n) return;
    float s = 0.f, m = -1.e30f;
    for (int k = 0; k < nf; k++) { const float v = w[(long long)k * sw + c]; m = fmaxf(m, v); if (v >= 10.f) s = 1.f; }
    strong[c] = s;
    wmax[c] = m;
}
// WRF hail_duration (gpuwm hail.cu) with the 3x3 neighbourhood as the cell and its cellsOnCell.  ``n`` cells are
// counted (a rank's owned cells); a neighbour may be any of the ``n_ext`` local cells (its halo cells, whose old mask
// and count the owners sent before this launch).
extern "C" __global__ void hex_hail_duration(
    const int n, const int n_ext, const int maxedges, const int* __restrict__ nedges, const int* __restrict__ coc,
    const float* __restrict__ mask, const float* __restrict__ oldmask, const float* __restrict__ olddur,
    float* __restrict__ dur, const float dt) {
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= n) return;
    float seen = fmaxf(mask[c], oldmask[c]), longest = olddur[c];
    const int ne = nedges[c];
    for (int e = 0; e < ne; e++) {
        const int j = coc[(long long)c * maxedges + e];
        if (j < 0 || j >= n_ext) { dur[c] = 0.f; return; } // an edge cell (WRF: no neighbour, no count)
        seen = fmaxf(seen, oldmask[j]); longest = fmaxf(longest, olddur[j]);
    }
    dur[c] = (seen == 1.f) ? longest + dt : 0.f;
}
// The columns WRF's driver hands the stone model (gpuwm hail_prepare): T, mass-level height, total pressure,
// moist density, the six water species and w on the lower faces; ht and the duration.
extern "C" __global__ void hex_hail_gather(
    const int K, const int nl, const int* __restrict__ idx, const float rvord,
    const float* __restrict__ theta_m, const long long s_th, const float* __restrict__ exner, const long long s_ex,
    const float* __restrict__ pb, const long long s_pb, const float* __restrict__ pp, const long long s_pp,
    const float* __restrict__ rho_zz, const long long s_rz, const float* __restrict__ zz, const long long s_zz,
    const float* __restrict__ zgrid, const long long s_zg, const float* __restrict__ w, const long long s_w,
    const float* const* __restrict__ q, const long long* __restrict__ s_q, const float* __restrict__ dur_full,
    float* __restrict__ out, float* __restrict__ ht, float* __restrict__ dur) {
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= K) return;
    const long long c = idx[j], plane = (long long)nl * K;
    for (int k = 0; k < nl; k++) {
        const long long o = (long long)k * K + j;
        const float qv = q[0] ? q[0][k * s_q[0] + c] : 0.f;
        out[o] = theta_m[k * s_th + c] / (1.f + rvord * qv) * exner[k * s_ex + c];
        out[plane + o] = 0.5f * (zgrid[k * s_zg + c] + zgrid[(k + 1) * s_zg + c]);
        out[2 * plane + o] = pb[k * s_pb + c] + pp[k * s_pp + c];
        out[3 * plane + o] = rho_zz[k * s_rz + c] * zz[k * s_zz + c] * (1.f + qv);
        for (int s = 0; s < 6; s++) out[(4 + s) * plane + o] = q[s] ? q[s][k * s_q[s] + c] : 0.f;
        out[10 * plane + o] = w[k * s_w + c];
    }
    ht[j] = zgrid[c];
    dur[j] = dur_full[c];
}
// TEMPO's own size-distribution words (mm) folded into the window maxima (m).
extern "C" __global__ void hex_hail_psd_fold(
    const int n, const float* __restrict__ sfc, const float* __restrict__ column,
    float* __restrict__ k1, float* __restrict__ m2) {
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= n) return;
    k1[c] = fmaxf(k1[c], 0.001f * sfc[c]);
    m2[c] = fmaxf(m2[c], 0.001f * column[c]);
}
"""


def mode() -> str:
    """``auto`` (unset), ``on`` or ``off``, from ``WOOF_HEX_HAILCAST``."""
    value = os.environ.get(ENV, "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return "off"
    if value in ("1", "on", "true", "yes"):
        return "on"
    return "auto"


def version() -> str:
    value = os.environ.get(ENV_VERSION, "").strip() or DEFAULT_VERSION
    if value not in KERNELS:
        raise ValueError(f"{ENV_VERSION}={value!r}: choose one of {', '.join(KERNELS)}")
    return value


def shared_sources() -> dict[str, str]:
    """The installed woof's hail kernel files and their sha256 prefixes (raises if woof has none)."""
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()[:16] for name in SHARED_SHA256}


def check_shared() -> None:
    """Refuse a woof whose hail kernels are not the bytes compared with WRF's and HRRR's Fortran."""
    try:
        found = shared_sources()
    except (ImportError, OSError) as error:
        raise RuntimeError(f"the installed woof has no HAILCAST kernels ({error}); gpuwm 2.9.12 or later has them")
    wrong = {name: digest for name, digest in found.items() if digest != SHARED_SHA256[name]}
    if wrong:
        raise RuntimeError("the installed woof's hail kernels are not the ones compared with WRF's Fortran: "
                           + ", ".join(f"{name} {digest} (proven {SHARED_SHA256[name]})" for name, digest in wrong.items()))


def gpuwm_kernel(name: str):
    """WOOF-ArWen's own kernel from the installed woof (its loader, its compile flags)."""
    from woof.core.kernels import get_kernel
    return get_kernel("hail", name)


def cell_spacing(dc_edge: np.ndarray, edges_on_cell: np.ndarray, n_edges_on_cell: np.ndarray, n: int) -> np.ndarray:
    """Each cell's mean edge spacing (m); ``edges_on_cell`` is (nCells, maxEdges), zero-based."""
    eoc = np.asarray(edges_on_cell, dtype=np.int64)[:n]
    count = np.asarray(n_edges_on_cell, dtype=np.int64).reshape(-1)[:n]
    dc = np.asarray(dc_edge, dtype=np.float64).reshape(-1)
    valid = (np.arange(eoc.shape[1])[None, :] < count[:, None]) & (eoc >= 0) & (eoc < dc.size)
    total = np.where(valid, dc[np.where(valid, eoc, 0)], 0.0).sum(axis=1)
    return total / np.maximum(valid.sum(axis=1), 1)


class Hailcast:
    """HAILCAST and the PSD maxima on the card for one mesh (or one rank's owned cells)."""

    def __init__(self, cp: Any, *, n_cells: int, n_levels: int, cells_on_cell: np.ndarray,
                 n_edges_on_cell: np.ndarray, fine: np.ndarray, n_out: int | None = None,
                 stone_model: str = DEFAULT_VERSION, psd: str | None = None, psd_row: Mapping | None = None,
                 n_ext: int | None = None, halo: Any = None):
        self.cp = cp
        self.n = int(n_cells)
        self.nl = int(n_levels)
        self.n_out = int(n_out if n_out is not None else n_cells)
        # A rank: n owned cells counted, n_ext local cells (owned first, then halo) whose words a neighbour reads;
        # ``halo.exchange(label, [(name, array, 'cell')])`` fills the halo words from their owners.
        self.n_ext = int(n_ext if n_ext is not None else n_cells)
        self.halo = halo
        if self.n_ext < self.n:
            raise ValueError(f"the local extent ({self.n_ext}) is smaller than the counted cells ({self.n})")
        if self.nl > 256:
            raise ValueError("HAILCAST column storage supports at most 256 mass levels (woof HailColumn<256>)")
        coc = np.asarray(cells_on_cell, dtype=np.int64)[:self.n]
        self.maxedges = int(coc.shape[1])
        # C order whatever the caller's layout: the kernel reads row c at c*maxEdges.  THE BREAKAGE THIS FIXES (the
        # 21 May 2024 Iowa case, 19 Nov 2026): the pool's (maxEdges, nCells) table arrives transposed, np.where kept its
        # Fortran order, and every cell read other cells' neighbours: the updraft count never grew under the updrafts
        # (0 s at all 149 strong cells at 19:45Z, 1,540 s in clear air), so stones landed in a few dozen cells.
        self.coc = cp.asarray(np.ascontiguousarray(np.where((coc >= 0) & (coc < self.n_ext), coc, -1), dtype=np.int32))
        self.nedges = cp.asarray(np.asarray(n_edges_on_cell, dtype=np.int32).reshape(-1)[:self.n])
        fine = np.asarray(fine, dtype=bool).reshape(-1)[:self.n]
        self.fine = cp.asarray(fine)
        self.n_fine = int(fine.sum())
        self.stone_model = stone_model
        self.psd = psd                      # 'tempo' (fold TEMPO's words), 'arwen' (woof hail_psd), or None
        self.psd_row = dict(psd_row or {})
        f32 = cp.float32
        self.mask = cp.zeros(self.n_ext, f32)
        self.dur = cp.zeros(self.n_ext, f32)
        self._new_mask = cp.zeros(self.n_ext, f32)
        self._new_dur = cp.zeros(self.n_ext, f32)
        self.wmax = cp.zeros(self.n, f32)
        self.exchanges = 0
        self.resumed = None
        # Window maxima (HAILCAST_FIELDS order: max, mean, std, embryos 1-5) and the run maximum, ArWen's slots.
        self.window = cp.zeros((len(HAILCAST_FIELDS), self.n_out), f32)
        self.runmax = cp.zeros(self.n_out, f32)
        self.psd_window = cp.zeros((2, self.n_out), f32) if psd else None
        self.module = cp.RawModule(code=_SOURCE)
        self._stone = gpuwm_kernel(KERNELS[stone_model])
        self._psd_kernel = gpuwm_kernel("hail_psd") if psd == "arwen" else None
        self.last = None                    # model time (s) of the last fold
        self.updates = self.integrations = self.columns = self.frames = 0
        self.error = None

    def fail(self, where: str, error: BaseException) -> None:
        if self.error is None:
            self.error = f"{where}: {type(error).__name__}: {error}"
            print(f"hailcast: switched off for this run ({self.error}); the forecast goes on without the hail fields",
                  file=sys.stderr, flush=True)

    def _launch(self, name: str, count: int, args: tuple, block: int = 128) -> None:
        if count:
            self.module.get_function(name)(((count + block - 1) // block,), (block,), args)

    @staticmethod
    def _rows(array: Any, rows: int, name: str, n: int) -> tuple[Any, np.int64]:
        if array.dtype != np.float32 or array.ndim != 2 or array.shape[0] < rows or array.shape[1] < n:
            raise ValueError(f"{name} must be float32 (>= {rows}, >= {n}), got {array.dtype} {array.shape}")
        if array.strides[1] != 4 or array.strides[0] % 4:
            raise ValueError(f"{name} must have unit cell stride")
        return array, np.int64(array.strides[0] // 4)

    def update(self, arrays: Mapping[str, Any], *, seconds: float, dt: float) -> None:
        """Fold one step's state (``seconds``: model time after the step, s; ``dt``: the step, s)."""
        if self.error is not None:
            return
        try:
            self._update(arrays, float(seconds), float(dt))
        except Exception as error:  # noqa: BLE001 -- an optional diagnostic never stops the forecast
            self.fail("update", error)

    def _update(self, arrays, seconds, dt):
        cp, n, nl = self.cp, self.n, self.nl
        if self.last is not None and seconds <= self.last:
            return      # this instant is folded already (an output event after its deferred step)
        w, sw = self._rows(arrays["w"], nl + 1, "w", n)
        # The count grows by the model time since the last fold (WRF adds dt because it folds every step).  On the
        # MPAS-physics resident route the fold runs every step (elapsed = dt = 20 s throughout the 21 May run), so this
        # changes nothing there; it keeps the count in seconds on a route that folds less often.
        elapsed = dt if self.last is None else seconds - self.last
        self.elapsed_max = max(getattr(self, "elapsed_max", 0.0), elapsed)
        if self.halo is not None:
            # The previous step's mask and count of the halo cells, from their owners: the 3x3 neighbourhood of a
            # cell at a partition edge reads them below.  THE BREAKAGE THIS PREVENTS: without them a rank's edge
            # cells read zeros (or stale words), so their count stopped at every partition boundary.
            self.halo.exchange("hailcast", [("hailcast/mask", self.mask, "cell"), ("hailcast/dur", self.dur, "cell")])
            self.exchanges += 1
        self._launch("hex_hail_mask", n, (np.int32(n), np.int32(nl + 1), w, sw, self._new_mask, self.wmax))
        self._launch("hex_hail_duration", n, (np.int32(n), np.int32(self.n_ext), np.int32(self.maxedges), self.nedges,
                                              self.coc, self._new_mask, self.mask, self.dur, self._new_dur,
                                              np.float32(elapsed)))
        self.mask, self._new_mask = self._new_mask, self.mask
        self.dur, self._new_dur = self._new_dur, self.dur
        start = seconds - dt
        if self.psd == "tempo":
            self._launch("hex_hail_psd_fold", n, (np.int32(n), arrays["max_hail_diameter_sfc"],
                                                  arrays["max_hail_diameter_column"], self.psd_window[0],
                                                  self.psd_window[1]))
        stones = int(seconds // HAILDT) > int(start // HAILDT)
        if stones and os.environ.get("WOOF_HEX_HAILCAST_DEBUG") and int(seconds) % 900 < HAILDT:
            mask, count = self.mask[:n], self.dur[:n]
            fine_strong = int((self.fine & (mask > 0)).sum())
            print(f"hailcast debug t={seconds:.0f} strong={int(mask.sum())} fine_strong={fine_strong} "
                  f"dur_max={float(count.max()):.0f} fine_dur_max={float(cp.where(self.fine, count, 0).max()):.0f} "
                  f"due={int((self.fine & (count > DURATION_S)).sum())} w_max={float(w[:, :n].max()):.1f} "
                  f"updates={self.updates} elapsed={elapsed:.0f} elapsed_max={self.elapsed_max:.0f}",
                  file=sys.stderr, flush=True)
        if stones or self.psd == "arwen":
            # ArWen: the size distribution every step, the stone model every haildt of model time.
            self._integrate(arrays, stones=stones)
        self.last = seconds
        self.updates += 1

    def _integrate(self, arrays, *, stones=True):
        cp, n, nl = self.cp, self.n, self.nl
        due = self.fine & (self.dur[:n] > DURATION_S)
        psd_all = self.psd == "arwen"
        if not self.n_fine:
            return
        idx_stone = cp.nonzero(due)[0].astype(cp.int32) if stones else cp.zeros(0, cp.int32)
        idx = cp.nonzero(self.fine)[0].astype(cp.int32) if psd_all else idx_stone
        K = int(idx.size)
        if not K:
            return
        cols = {}
        for name, rows in (("theta_m", nl), ("exner", nl), ("pressure_base", nl), ("pressure_p", nl),
                           ("rho_zz", nl), ("zz", nl), ("zgrid", nl + 1), ("w", nl + 1)):
            cols[name] = self._rows(arrays[name], rows, name, n)
        species = []
        strides = []
        for name in ("qv", "qi", "qc", "qr", "qs", "qg"):
            value = arrays.get(name)
            if value is None:
                species.append(0)
                strides.append(0)
            else:
                value, stride = self._rows(value, nl, name, n)
                species.append(value.data.ptr)
                strides.append(int(stride))
        q = cp.asarray(np.asarray(species, dtype=np.uint64))
        s_q = cp.asarray(np.asarray(strides, dtype=np.int64))
        out = cp.empty((11, nl, K), cp.float32)
        ht = cp.empty(K, cp.float32)
        dur = cp.empty(K, cp.float32)
        rvord = np.float32(np.float32(461.6) / np.float32(287.0))
        self._launch("hex_hail_gather", K, (np.int32(K), np.int32(nl), idx, rvord,
                     *cols["theta_m"], *cols["exner"], *cols["pressure_base"], *cols["pressure_p"],
                     *cols["rho_zz"], *cols["zz"], *cols["zgrid"], *cols["w"], q, s_q, self.dur, out, ht, dur))
        if psd_all:
            self._psd(out, idx, K)
            if not stones or not int(idx_stone.size):
                return
            # The stone columns among the fine ones: their rows of the gathered words.
            pick = cp.nonzero(due[idx])[0]
            out = cp.ascontiguousarray(out[:, :, pick])
            ht, dur, idx = ht[pick], dur[pick], idx[pick]
            K = int(idx.size)
        slots = cp.ascontiguousarray(self.window[:, idx])     # max, mean, std, e1..e5 (this window so far)
        runmax = cp.ascontiguousarray(self.runmax[idx])
        embryos = [cp.ascontiguousarray(slots[3 + i]) for i in range(5)]
        maximum, mean, sd = (cp.ascontiguousarray(slots[i]) for i in range(3))
        # woof hailcast_columns*: (t, z, p, rho, qv, qi, qc, qr, qs, qg, w, ht, dur, e1..e5, max, mean, sd, runmax,
        # nx, ny, nz) with every column array (nz, ny*nx): here ny = 1, nx = K.
        self._stone(((K + 31) // 32,), (32,), (*(out[v] for v in range(11)), ht, dur, *embryos, maximum, mean, sd,
                                                runmax, np.int32(K), np.int32(1), np.int32(nl)))
        if os.environ.get("WOOF_HEX_HAILCAST_DEBUG") and self.integrations % 15 == 0:
            j = 0
            print(f"hailcast debug K={K} cell={int(idx[j])} max_out={float(maximum.max()):.2f} landed={int((maximum > 0).sum())} "
                  f"T={float(out[0, 0, j]):.1f}/{float(out[0, :, j].min()):.1f} z0={float(out[1, 0, j]):.0f} ht={float(ht[j]):.0f} "
                  f"p={float(out[2, 0, j]):.0f}/{float(out[2, -1, j]):.0f} rho={float(out[3, 0, j]):.3f} "
                  f"qv={float(out[4, 0, j]):.4f} qc={float(out[6].max()):.4f} qg={float(out[9].max()):.4f} "
                  f"w={float(out[10, :, j].max()):.1f} dur={float(dur[j]):.0f}", file=sys.stderr, flush=True)
        for i, value in enumerate((maximum, mean, sd, *embryos)):
            self.window[i, idx] = value
        self.runmax[idx] = runmax
        self.integrations += 1
        self.columns += K

    def _psd(self, out, idx, K):
        """woof hail_psd on the fine columns (Thompson/WSM6 rows): the window's lowest-level and column maxima."""
        cp, nl, row = self.cp, self.nl, self.psd_row
        zero = cp.zeros((nl, K), cp.float32)
        low, col = cp.zeros(K, cp.float32), cp.zeros(K, cp.float32)
        run_low, run_col = cp.zeros(K, cp.float32), cp.zeros(K, cp.float32)
        rho, qg = cp.ascontiguousarray(out[3]), cp.ascontiguousarray(out[9])
        self._psd_kernel(((K + 127) // 128,), (128,), (
            rho, qg, zero, zero, zero, zero, zero, low, col, run_low, run_col,
            np.int32(K), np.int32(1), np.int32(nl), np.int32(0), np.int32(0), np.int32(0), np.int32(0),
            np.int32(row.get("thompson", 0)), np.float32(row.get("density_g", 500.)), np.float32(900.),
            np.float32(0.), np.float32(0.), np.float32(row.get("intercept", 0.)), np.float32(0.)))
        self.psd_window[0, idx] = cp.maximum(self.psd_window[0, idx], low)
        self.psd_window[1, idx] = cp.maximum(self.psd_window[1, idx], col)

    def fields(self) -> dict[str, tuple[np.ndarray, tuple[str, ...], str]]:
        """The window's words on the host in the writer's extra-field form (name -> (array, ('nCells',), units))."""
        if self.error is not None:
            return {}
        try:
            host = self.cp.asnumpy(self.window)
            out = {name: (host[i], ("nCells",), UNITS[name]) for i, name in enumerate(HAILCAST_FIELDS)}
            if os.environ.get("WOOF_HEX_HAILCAST_DEBUG"):
                # the instant's column-maximum w as this module reads it, its 10 m/s mask and the updraft count
                pad = lambda a: np.pad(self.cp.asnumpy(a), (0, self.n_out - self.n))
                out.update(HAILCAST_DEBUG_WMAX=(pad(self.wmax), ("nCells",), "m s-1"),
                           HAILCAST_DEBUG_MASK=(pad(self.mask[:self.n]), ("nCells",), "1"),
                           HAILCAST_DEBUG_WDUR=(pad(self.dur[:self.n]), ("nCells",), "s"))
            if self.psd_window is not None:
                psd = self.cp.asnumpy(self.psd_window)
                out.update({name: (psd[i], ("nCells",), UNITS[name]) for i, name in enumerate(PSD_FIELDS)})
        except Exception as error:  # noqa: BLE001
            self.fail("fields", error)
            return {}
        self.frames += 1
        return out

    def reset_window(self) -> None:
        self.window.fill(0)
        if self.psd_window is not None:
            self.psd_window.fill(0)

    def carry(self) -> dict[str, np.ndarray]:
        """The words a seal carries (owned cells, last axis = cells): mask, count, the hour's maxima, run maximum,
        and the last fold's model time (one word repeated per cell, so the rank assembly takes it like the rest)."""
        n, host = self.n, self.cp.asnumpy
        out = {CARRY + "mask": host(self.mask[:n]), CARRY + "dur": host(self.dur[:n]),
               CARRY + "window": host(self.window[:, :n]), CARRY + "runmax": host(self.runmax[:n]),
               CARRY + "last": np.full(n, np.nan if self.last is None else self.last, np.float64)}
        if self.psd_window is not None:
            out[CARRY + "psd"] = host(self.psd_window[:, :n])
        return out

    def resume(self, state: Mapping[str, Any]) -> list[str]:
        """Continue a seal's words (``carry``); each array's first ``n`` cells along its last axis are this
        session's owned cells (one card: the global cells, a native dummy cell may follow; a rank: its own)."""
        cp, n, taken = self.cp, self.n, []

        def put(name, target):
            value = state.get(CARRY + name)
            if value is None or target is None:
                return
            value = np.asarray(value)
            if value.shape[-1] < n or value.shape[:-1] != target.shape[:-1]:
                raise ValueError(f"sealed {CARRY}{name} {value.shape} does not fit {target.shape[:-1]} x {n} cells")
            target[..., :n] = cp.asarray(np.ascontiguousarray(value[..., :n], dtype=np.float32))
            taken.append(name)

        put("mask", self.mask)
        put("dur", self.dur)
        put("window", self.window)
        put("runmax", self.runmax)
        put("psd", self.psd_window)
        last = state.get(CARRY + "last")
        if last is not None and np.size(last) and np.isfinite(np.asarray(last).reshape(-1)[0]):
            self.last = float(np.asarray(last).reshape(-1)[0])
            taken.append("last")
        self.resumed = taken
        return taken

    def receipt(self) -> dict[str, Any]:
        return {"stone_model": self.stone_model, "psd": self.psd, "fine_cells": self.n_fine, "cells": self.n,
                "updates": self.updates, "integrations": self.integrations, "columns_integrated": self.columns,
                "frames": self.frames, "fields": list(HAILCAST_FIELDS) + (list(PSD_FIELDS) if self.psd else []),
                "halo_exchanges": self.exchanges, "resumed_from_seal": self.resumed, "error": self.error}


def gather_to_rank0(comm: Any, rank: int, cell_l2g: Any, n_owned: int, global_cells: int,
                    fields: Mapping[str, tuple]) -> dict[str, tuple]:
    """Every rank's owned words of the (nCells,) extra fields on rank 0's global mesh; ``{}`` elsewhere.

    Only the cells where some field is not zero travel (hail is sparse).  A rank with no words (switched off) sends
    None and rank 0 then writes none of these fields: half a map is worse than none.  Every rank calls this at
    every diagnostics frame, in the same order (it is a collective on ``comm``).
    """
    packet = None
    if fields:
        names = sorted(fields)
        stack = np.stack([np.asarray(fields[k][0], np.float32).reshape(-1)[:n_owned] for k in names])
        if stack.shape[1] != n_owned:
            raise ValueError(f"an extra field holds {stack.shape[1]} words for {n_owned} owned cells")
        keep = np.nonzero((stack != 0).any(axis=0))[0]
        ids = np.asarray(cell_l2g, dtype=np.int64)[:n_owned][keep]
        packet = (names, [fields[k][2] for k in names], ids, np.ascontiguousarray(stack[:, keep]))
    packets = comm.gather(packet, root=0)
    if rank != 0:
        return {}
    if any(p is None for p in packets):
        if any(p is not None for p in packets):
            print("hailcast: a rank holds no hail words (switched off there); this frame carries none",
                  file=sys.stderr, flush=True)
        return {}
    names, units = packets[0][0], packets[0][1]
    out = np.zeros((len(names), int(global_cells)), np.float32)
    for got_names, _, ids, values in packets:
        if list(got_names) != list(names):
            raise ValueError(f"the ranks disagree on the hail fields ({got_names} against {names})")
        out[:, ids] = values
    return {name: (out[i], ("nCells",), units[i]) for i, name in enumerate(names)}


def _whole_hour(when: Any) -> bool:
    return isinstance(when, datetime) and when.minute == 0 and when.second == 0 and when.microsecond == 0


def attach_mpas(diagnostics: Any, output: Any, *, halo: Any = None, suite_mp: str | None = None) -> dict[str, Any] | None:
    """Arm HAILCAST and the PSD maxima on the MPAS-physics routes (v8.4.2 and RRFS); returns its receipt row.

    On a multi-rank run every rank arms or none does (the halo exchange and the frame gather are collectives): the
    ranks agree first, and a rank with no fine cell is armed with nothing to count.
    """
    if mode() == "off":
        return None
    try:
        armed = _prepare_mpas(diagnostics, output, halo=halo, suite_mp=suite_mp)
    except Exception as error:  # noqa: BLE001 -- an optional diagnostic never stops the forecast
        armed = {"enabled": False, "error": f"{type(error).__name__}: {error}"}
    if halo is not None:
        comm = halo.execution.comm
        mine = (callable(armed), bool(callable(armed) and armed.fine_here), None if callable(armed) else armed)
        every = comm.allgather(mine)
        if not all(ok for ok, _, _ in every):
            reason = next(row for ok, _, row in every if not ok)
            print(f"hailcast: not armed on any rank ({reason.get('error') or reason.get('reason')}); "
                  "the forecast goes on without the hail fields", file=sys.stderr, flush=True)
            return reason
        if not any(fine for _, fine, _ in every):
            return {"enabled": False, "reason": f"no cell finer than {FINE_M:.0f} m on any rank"}
    elif not callable(armed):
        if "error" in armed:
            print(f"hailcast: not armed ({armed['error']}); the forecast goes on without the hail fields",
                  file=sys.stderr, flush=True)
        return armed
    elif not armed.fine_here:
        return {"enabled": False, "reason": armed.no_fine_reason}
    return armed()


def _prepare_mpas(diagnostics, output, *, halo=None, suite_mp=None):
    """Everything up to arming: this session's Hailcast, and a call that wires it into the writer."""
    if diagnostics is None or not hasattr(output, "streams"):
        return {"enabled": False, "reason": "this route has no resident output diagnostics"}
    check_shared()
    stone_model = version()
    cp = diagnostics.cp
    dims = diagnostics.session.layout.dimensions
    n_solve, n_cells, nl = int(dims["nCellsSolve"]), int(dims["nCells"]), int(dims["nVertLevels"])
    execution = None if halo is None else halo.execution
    n_ext = n_solve
    if execution is not None:
        n_ext = int(execution.tables[execution.rank].n_local_cells)
        if int(execution.partition.n_owned_cells) != n_solve:
            raise ValueError(f"the rank owns {execution.partition.n_owned_cells} cells and solves {n_solve}")
    mesh = diagnostics.pools["mesh"]
    host = {name: cp.asnumpy(mesh[name]) for name in ("cellsOnCell", "nEdgesOnCell", "edgesOnCell", "dcEdge")}
    # Native (maxEdges, nCells) one-based with the nCells+1 sentinel (the convective module transposes them too).
    coc = np.asarray(host["cellsOnCell"], dtype=np.int64)
    eoc = np.asarray(host["edgesOnCell"], dtype=np.int64)
    coc, eoc = coc.T - 1, eoc.T - 1          # native (maxEdges, nCells), as convective_rrfs reads them (.T)
    spacing = cell_spacing(host["dcEdge"], eoc, host["nEdgesOnCell"], n_solve)
    fine = np.ones(n_solve, bool) if mode() == "on" else spacing <= FINE_M
    indices = diagnostics.session.layout.indices
    physics = diagnostics.pools.get("diag_physics") if isinstance(diagnostics.pools, dict) else None
    psd, row = None, None
    if physics is not None and "max_hail_diameter_sfc" in physics and "max_hail_diameter_column" in physics:
        psd = "tempo"
    else:
        configs = getattr(diagnostics.session.layout, "configs", None) or {}
        scheme = str(suite_mp or configs.get("config_microp_scheme", "") or "").strip()
        if scheme in PSD_ROWS:
            psd, row = "arwen", PSD_ROWS[scheme]
    hail = Hailcast(cp, n_cells=n_solve, n_levels=nl, cells_on_cell=coc, n_edges_on_cell=host["nEdgesOnCell"],
                    fine=fine, n_out=n_cells, stone_model=stone_model, psd=psd, psd_row=row,
                    n_ext=n_ext, halo=halo)

    def arm():
        _arm_mpas(diagnostics, output, hail, execution=execution, indices=indices, psd=psd)
        if os.environ.get("WOOF_HEX_HAILCAST_DEBUG"):
            print(f"hailcast debug cellsOnCell raw shape {host['cellsOnCell'].shape} cell 0 neighbours "
                  f"{coc[0][:8].tolist()} nEdges {int(np.asarray(host['nEdgesOnCell']).reshape(-1)[0])} "
                  f"fine {int(fine.sum())}", file=sys.stderr, flush=True)
        where = "" if execution is None else (f" on rank {execution.rank} ({n_ext - n_solve} halo cells exchanged "
                                              "each fold, frames gathered to rank 0)")
        every = "" if mode() == "auto" else " (every cell: WOOF_HEX_HAILCAST=on)"
        print(f"hailcast: armed, stone model {stone_model}, {hail.n_fine} of {n_solve} cells finer than {FINE_M:.0f} m"
              f"{every}{where}, size distribution {psd or 'none on this microphysics'}", file=sys.stderr, flush=True)
        return {"enabled": True, **hail.receipt()}

    arm.fine_here = bool(fine.any())
    arm.no_fine_reason = (f"no cell finer than {FINE_M:.0f} m (finest {spacing.min():.0f} m)" if n_solve
                          else "no cells")
    return arm


def _arm_mpas(diagnostics, output, hail, *, execution, indices, psd):
    def arrays():
        a = diagnostics.authority
        out = {name: a(pool, name) for pool, name in (
            ("state", "theta_m"), ("state", "rho_zz"), ("state", "w"), ("diag", "exner"),
            ("diag", "pressure_base"), ("diag", "pressure_p"), ("mesh", "zgrid"), ("mesh", "zz"))}
        scalars = a("state", "scalars")
        for name in ("qv", "qi", "qc", "qr", "qs", "qg"):
            key = ("state", f"index_{name}")
            if key in indices:
                out[name] = scalars[int(indices[key]) - 1]
        if psd == "tempo":
            for name in ("max_hail_diameter_sfc", "max_hail_diameter_column"):
                out[name] = a("diag_physics", name)
        return out

    dt = float(diagnostics.dycore.config.config_dt)
    original = diagnostics._update_convective

    def update_convective_and_hail():
        original()
        if hail.error is None:
            when = diagnostics.when
            try:
                seconds = ((when.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()
                           if isinstance(when, datetime) else float(when))
                hail.update(arrays(), seconds=seconds, dt=dt)
            except Exception as error:  # noqa: BLE001
                hail.fail("pool views", error)

    diagnostics._update_convective = update_convective_and_hail
    diagnostics.accumulates = True
    original_after = diagnostics.after_snapshot
    state = {}

    def after_snapshot():
        try:
            return original_after()
        finally:
            if state.pop("reset", False):
                hail.reset_window()

    diagnostics.after_snapshot = after_snapshot
    previous = getattr(output, "extra_diagnostics", None)

    def extra_diagnostics():
        fields = hail.fields()
        if fields and int(diagnostics.session.layout.dimensions["nCells"]) != hail.n_out:
            hail.fail("diagnostics", ValueError("nCells moved from the cells HAILCAST holds"))
            fields = {}
        if execution is not None:
            # Rank 0 writes the global frame: every rank's owned words, gathered (a collective at every frame).
            fields = gather_to_rank0(execution.comm, execution.rank, execution.partition.cell_l2g, hail.n,
                                     int(execution.global_layout.dimensions["nCells"]), fields)
        # The window restarts after a whole-hour frame (HRRR's hourly maxima).
        state["reset"] = _whole_hour(getattr(diagnostics, "when", None))
        return {**(previous() if previous else {}), **fields}

    output.extra_diagnostics = extra_diagnostics
    # The seal carries the count and the window with the convective accumulators (checkpoint_v842.rank_continuation
    # reads accumulator_state; restore_continuation hands the carried words to resume_accumulators; same-host ranks
    # assemble them over owned cells and project them back, moving_mpi_v842).
    original_state = getattr(diagnostics, "accumulator_state", None)
    if callable(original_state):
        def accumulator_state():
            words = dict(original_state() or {})
            if hail.error is None:
                words.update(hail.carry())
            return words

        diagnostics.accumulator_state = accumulator_state
    original_resume = getattr(diagnostics, "resume_accumulators", None)
    if callable(original_resume):
        def resume_accumulators(words):
            ours = {k: v for k, v in words.items() if k.startswith(CARRY)}
            if ours:
                try:
                    taken = hail.resume(ours)
                    print(f"hailcast: resumed from the seal ({', '.join(taken)}): the updraft count and the hour's "
                          "maxima go on as in the run that never stopped", file=sys.stderr, flush=True)
                except Exception as error:  # noqa: BLE001
                    hail.fail("resume", error)
            return original_resume({k: v for k, v in words.items() if not k.startswith(CARRY)})

        diagnostics.resume_accumulators = resume_accumulators
    diagnostics.hailcast = hail
