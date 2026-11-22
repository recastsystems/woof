"""How RRTMG's McICA sub-column cloud generator is seeded (``config_radt_mcica_seed``).

``pressure`` (the default) is MPAS's own rule and changes nothing: each column's KISS stream is seeded from the
fractional digits of its four lowest layer pressures in Pa (MPAS module_ra_rrtmg_lw.F:2432-2440, LW, and
module_ra_rrtmg_sw.F:1745-1748, SW), as hex's ``mpas_lw_mcica_seeds`` and ``mpas_sw_mcica_seeds`` do word for word.
Under that rule any change of the state, down to one bit of one pressure, re-draws the column's cloud sample at the
next radiation call.

``cell`` seeds each column from what does not move with the state: the cell's global index (MPAS ``indexToCellID``),
the radiation call's model time step, the band (long-wave 0, short-wave 1) and the base seed ``BASE_SEED``. Two runs
from nearby starts then draw the same sub-columns at every radiation call, on any number of ranks (a cell's global
index and the step are the same on every rank count). It is for derivative checks and twin
experiments: there the cloud sample is a constant of the linearisation, as the linear model already treats it. It is
not MPAS's rule, so a run that names it is not bitwise to MPAS.

The ``cell`` words are written by one launch, ``mpas_mcica_seeds_column``, over the seed words the pressure rule has
just written (the pressure launch itself is unchanged), so the default path runs exactly the launches it ran before
this option existed. The launch reads integer keys only and writes uint32 words only: the derivative runner treats
it as inactive (identity both ways is its exact derivative; hexad wrules.rs, ``mpas_mcica_seeds_column``), and a
perturbed forward replaying a recorded plan writes the same words, so a plan recorded under ``cell`` replays its own
cloud sample in every perturbed forward.
"""
from __future__ import annotations

import numpy as np

#: The namelist word (&physics) and its values; the default is MPAS's rule.
OPTION = "config_radt_mcica_seed"
PRESSURE = "pressure"
CELL = "cell"
MODES = (PRESSURE, CELL)
DEFAULT = PRESSURE
#: The base seed of the ``cell`` rule: MPAS's own fixed long-wave wrapper seed (module_ra_rrtmg_lw.F:12175-12185,
#: permuteseed 150), so the rule is stated in one number a reader can find in MPAS.
BASE_SEED = 150
#: The band word of the hash: one stream for long-wave, another for short-wave.
LONGWAVE, SHORTWAVE = 0, 1
#: The kernel the derivative runner knows by name (inactive, integer words only).
KERNEL = "mpas_mcica_seeds_column"

_SOURCE = r"""
// The cell rule of config_radt_mcica_seed (hexcore/mpasphys/mcica_seed_rule.py). Each of a column's four KISS words
// is the murmur3 finaliser chained over (base, global cell id, model time step, band*4 + word). Zero is a fixed
// point of the xorshift and of both multiply-with-carry components of KISS, so a zero word is replaced by the golden
// ratio constant; m = a*65536-1, the other MWC fixed point, is replaced the same way.
__device__ __forceinline__ unsigned int hex_fmix32(unsigned int h){
    h^=h>>16;h*=0x85ebca6bu;h^=h>>13;h*=0xc2b2ae35u;h^=h>>16;return h;
}
extern "C" __global__ void mpas_mcica_seeds_column(int n,const int* ids,unsigned int step,unsigned int band,
    unsigned int base,unsigned int* seeds){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    unsigned int h0=hex_fmix32(hex_fmix32(base^0x9e3779b9u)^(unsigned int)ids[c]);
    h0=hex_fmix32(h0^step);
    for(int k=0;k<4;++k){
        unsigned int h=hex_fmix32(h0^(band*4u+(unsigned int)k+1u));
        if(h==0u||(k==2&&h==1179647999u)||(k==3&&h==2025259007u))h^=0x9e3779b9u;
        seeds[c*4+k]=h;
    }
}
"""

_MODULES: dict = {}


def mode(options) -> str:
    """The seed rule an option set selects. An unknown value refuses by name (a misspelled rule would otherwise run
    MPAS's rule under a receipt that names another)."""
    value = str((options or {}).get(OPTION, DEFAULT)).strip().strip("'\"").casefold() or DEFAULT
    if value not in MODES:
        raise ValueError(f"{OPTION} = {value!r}: want one of {', '.join(MODES)} ({PRESSURE}, the default, is MPAS's "
                         f"rule; {CELL} seeds from the cell's global index and the radiation call's time step)")
    return value


def _kernel():
    import cupy as cp
    device = int(cp.cuda.Device().id)
    if device not in _MODULES:
        from woof import nvrtc_ptx_cache
        ptx, _ = nvrtc_ptx_cache.compile_using_nvrtc(_SOURCE, ("-std=c++17",), None, "mpas_mcica_seeds_column.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
        _MODULES[device] = module
    return _MODULES[device].get_function(KERNEL)


class Held:
    """The ``cell`` rule's keys for one radiation call: the device int32 global ids of the call's columns (in the
    call's own column order) and the call's model time step."""

    __slots__ = ("ids", "step")

    def __init__(self, ids, step):
        import cupy as cp
        self.ids = cp.ascontiguousarray(cp.asarray(ids).astype(cp.int32, copy=False))
        self.step = int(step)

    def take(self, columns):
        """The keys of a batch of this call's columns (a device index array)."""
        out = Held.__new__(Held)
        import cupy as cp
        out.ids = cp.ascontiguousarray(self.ids[columns])
        out.step = self.step
        return out

    def write(self, seeds, band):
        """Overwrite a batch's (n, 4) uint32 seed words with the cell rule's words."""
        n = int(seeds.shape[0])
        if n != int(self.ids.size):
            raise ValueError(f"{OPTION}: {n} seed rows for {int(self.ids.size)} cell ids")
        if n:
            _kernel()(((n + 127) // 128,), (128,),
                      (np.int32(n), self.ids, np.uint32(self.step & 0xFFFFFFFF), np.uint32(band),
                       np.uint32(BASE_SEED), seeds))


def held(options, ids, step):
    """``Held`` keys when the options select the ``cell`` rule, else None (MPAS's rule: nothing is launched).
    ``ids`` may be a callable that returns the ids, so the pressure rule never reads the mesh."""
    if mode(options) != CELL:
        return None
    return Held(ids() if callable(ids) else ids, step)


def host_words(ids, step, band, base=BASE_SEED):
    """The cell rule's words computed on the host (numpy), for tests and receipts."""
    def fmix(h):
        h = np.uint32(h)
        h ^= h >> np.uint32(16); h = np.uint32((int(h) * 0x85EBCA6B) & 0xFFFFFFFF)
        h ^= h >> np.uint32(13); h = np.uint32((int(h) * 0xC2B2AE35) & 0xFFFFFFFF)
        h ^= h >> np.uint32(16)
        return h
    out = np.empty((len(ids), 4), dtype=np.uint32)
    for c, cell in enumerate(ids):
        h0 = fmix(np.uint32(fmix(np.uint32(base) ^ np.uint32(0x9E3779B9))) ^ np.uint32(np.int32(cell).view(np.uint32)))
        h0 = fmix(np.uint32(h0) ^ np.uint32(step & 0xFFFFFFFF))
        for k in range(4):
            h = int(fmix(np.uint32(h0) ^ np.uint32(band * 4 + k + 1)))
            if h == 0 or (k == 2 and h == 1179647999) or (k == 3 and h == 2025259007):
                h ^= 0x9E3779B9
            out[c, k] = h
    return out
