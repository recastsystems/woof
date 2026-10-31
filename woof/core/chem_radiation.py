"""Chemistry aerosol optics at the radiation clock.

The live transported rows feed both spectra, including clear-sky radiation.
Optics above the model top are zero. No aerosol is extrapolated into the
radiation buffer layers. All field transforms and mixtures run on CUDA.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class RadiationAerosols:
    ctx: object
    legacy_optics: object = None

    def selected(self, value, columns):
        import cupy as cp
        nz = value.shape[0]
        return cp.ascontiguousarray(value.reshape(nz, -1)[:, columns]).reshape(nz, 1, -1)

    def inputs(self, columns):
        from woof.core.chem_optics import rows
        rs = rows(self.ctx.table)
        metadata = [{"optics": dict(r.optics) if r.optics else None,
                     "units": r.units} for r in rs]
        fields = [self.selected(self.ctx.field(r), columns) for r in rs]
        return metadata, fields

    def legacy(self, columns, nlay, kind):
        """RRTMG tau/asy/ssa (ncol,14,nlay), or LW tau (ncol,nlay,16)."""
        import cupy as cp
        from woof.core.chem_optics import rrtmg_sw_bands
        result = self.legacy_optics
        if result is None:
            result = self.legacy_optics = _legacy_optics(self.ctx)
        sw = kind == "sw"
        nz = self.ctx.met("alt").shape[0]
        ncol = self.ctx.met("alt").size // nz
        nc = len(range(*columns.indices(ncol))) if isinstance(columns, slice) else len(columns)
        # Each column's Mie arithmetic is independent. Gather the full-domain
        # bands instead of evaluating all wavelengths again for every chunk.
        def selected(name):
            value = result[name]
            nb = value.shape[0]
            return cp.ascontiguousarray(value.reshape(nb, nz, -1)[:, :, columns]).reshape(nb, nz, 1, nc)
        if sw:
            bands = rrtmg_sw_bands(selected("tauaer"), selected("waer"), selected("gaer"))
            arrays = []
            for key, neutral in (("tau", 0.), ("asm", 0.), ("ssa", 1.)):
                out = cp.full((nc, 14, nlay), neutral, cp.float32)
                out[:, :, :nz] = bands[key][:, :, 0, :].transpose(2, 1, 0)
                arrays.append(out)
            return tuple(arrays)
        out = cp.zeros((nc, nlay, 16), cp.float32)
        out[:, :nz, :] = selected("tauaerlw")[:, :, 0, :].transpose(2, 1, 0)
        return out

    def modern(self, columns, nlay, kind):
        """Native MERRA table bands; dry column mass from dry rho * dz."""
        import cupy as cp
        from woof.core.chem_rrtmgp_aerosol import load_tables, launch
        from woof.core.constants import G
        rows, fields = self.inputs(columns)
        # ALT is inverse dry density; the total pressure difference would
        # include vapor mass. This is rho_dry * dz * g, in Pa.
        dry_dp = cp.ascontiguousarray(
            self.selected(self.ctx.met("dz8w"), columns) /
            self.selected(self.ctx.met("alt"), columns) * np.float32(G))
        rh = cp.clip(self.selected(self.ctx.met("rh"), columns), 0., 1.)
        tables = load_tables(kind)
        result = launch(rows, fields, dry_dp, rh, tables, gravity=float(G))
        nc, nz, nb = result[0].shape
        outputs = []
        for value in result[:3]:
            out = cp.zeros((nc, nlay, nb), cp.float32)
            out[:, :nz, :] = value
            outputs.append(out)
        return tuple(outputs)


def _legacy_optics(ctx):
    from woof.core.chem_optics import gocart_optics, _row_dicts, rows
    optics = gocart_optics(
        _row_dicts(ctx.table), [ctx.field(r) for r in rows(ctx.table)],
        ctx.met("alt"), ctx.met("rh"), ctx.met("dz8w"),
        outputs=("tauaer", "waer", "gaer", "tauaerlw"), moments=False,
        out={"EXTCOF55": ctx.diag["extcof55"],
             "AOD5502D": ctx.diag["aod550"]})
    ctx.state.chem.gocart_optics_tauaer = optics["tauaer"]
    return optics


def prepare(state, cfg):
    """Refresh once at each live radiation call, including time zero."""
    if not getattr(cfg, "aer_ra_feedback", 0):
        return None
    chem = getattr(state, "chem", None)
    if chem is None or not chem.table.rows_for("optics.gocart"):
        raise ValueError("aerosol feedback needs active optics.gocart mass rows")
    from woof.core.chem_driver import _context
    from woof.core.chem_optics import update_diagnostics
    now = float(state.elapsed_seconds)
    cached = getattr(chem, "radiation_aerosols", None)
    if cached is None or getattr(chem, "radiation_aerosols_time", None) != now:
        ktau = int(round(now / float(cfg.dt))) + 1
        ctx = _context(state, cfg, ktau)
        ctx.prep.refresh(state)
        from woof.physics_compat import rrtmg_variant, RRTMG_VARIANT_LEGACY
        optics = None
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            optics = _legacy_optics(ctx)
        else:
            update_diagnostics(ctx)
        cached = chem.radiation_aerosols = RadiationAerosols(ctx, optics)
        chem.radiation_aerosols_time = now
    return cached


def finalize(tables, gas, cloud, mask, aerosol, *, clear=False):
    """Mix gas/cloud/aerosol before SW delta scaling; aerosol ignores McICA."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    tau = gas.tau
    shape = tau.shape
    out = tuple(cp.empty_like(tau) for _ in range(1 if tables.kind == "lw" else 3))
    dummy = tau
    get_kernel("chem_radiation", "chem_radiation_finalize")(
        ((tau.size + 255) // 256,), (256,),
        (tau, gas.ssa if gas.ssa is not None else dummy,
         cloud.tau, cloud.ssa, cloud.g, *aerosol,
         tables.to_device().gpoint_bands, mask if mask is not None else dummy,
         out[0], out[1] if len(out)>1 else dummy, out[2] if len(out)>1 else dummy,
         np.int32(tau.size), np.int32(tables.ngpt), np.int32(tables.nband),
         np.int32(tables.kind == "sw"), np.int32(mask is not None),
         np.int32(clear)))
    return out


def solve_lw(aerosols, columns, tables, gas, cloud, mask, emiss, planck, *,
             out=None, incident_out=None):
    from woof.core.rrtmgp import _lw_rte
    aer = aerosols.modern(columns, gas.tau.shape[1], "lw")
    clear_tau, = finalize(tables, gas, cloud, mask, aer, clear=True)
    clear = _lw_rte(clear_tau, None, None, None, emiss, top_at_1=False,
                    out=None, incident_out=None, planck=planck)
    del clear_tau
    all_tau, = finalize(tables, gas, cloud, mask, aer)
    flux = _lw_rte(all_tau, None, None, None, emiss, top_at_1=False,
                   out=out, incident_out=incident_out, planck=planck)
    return flux, clear


def solve_sw(aerosols, columns, tables, gas, cloud, mask, mu, albedo, inc, *,
             out=None):
    from woof.core.rrtmgp import _sw_rte
    aer = aerosols.modern(columns, gas.tau.shape[1], "sw")
    clear_optics = finalize(tables, gas, cloud, mask, aer, clear=True)
    clear = _sw_rte(*clear_optics, mu, albedo, albedo, inc, top_at_1=False, out=None)
    del clear_optics
    all_optics = finalize(tables, gas, cloud, mask, aer)
    flux = _sw_rte(*all_optics, mu, albedo, albedo, inc, top_at_1=False, out=out)
    return flux, clear


def memory_shapes(cfg, p_top, column_chunk):
    """Extra device storage, outside the radiation and chemistry inventories.

    Mie table extents are pinned by chem/optics/MANIFEST.txt. MERRA extents
    are pinned by TABLE_PINS (7 types, 5 sizes, 36 RH points). Row-count
    terms include packing constants and pointer owners. The modern path
    keeps its finalized g-point cubes outside the shared phase workspace.
    Legacy chunk optics are priced by legacy_radiation_vram_bytes instead.
    """
    from woof.chem_table import load
    from woof.physics_compat import rrtmg_variant, RRTMG_VARIANT_LEGACY
    table = load(cfg)
    if table is None or not table.rows_for("optics.gocart"):
        return {}, {}
    nr = len(table.rows_for("optics.gocart"))
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    held = {"chem_optics/tau4": (4, nz, ny, nx),
            "chem_optics/mie_packed": (20, 11, 2450),
            "chem_optics/mie_grids": (20, 2, 7),
            "chem_optics/mie_refractive": (20, 11, 3),
            "chem_optics/mie_scalar_tables": (4*9 + 5 + 25 + 28,)}
    transient = {"chem_optics/diagnostic_status": (ny, nx),
                 "chem_optics/row_packing": (nr, 64)}
    if not getattr(cfg, "aer_ra_feedback", 0):
        return held, transient
    if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
        # tau4 is already counted above. Retain SW ssa/asymmetry and LW tau
        # across chunks; old and new generations can overlap at refresh.
        held["chem_optics/radiation_bands"] = (24, nz, ny, nx)
        transient["chem_optics/radiation_refresh"] = (28, nz, ny, nx)
        return held, transient
    from woof.core.rrtmgp import rrtmgp_above_model_layer_counts
    nc = min(nx * ny, column_chunk)
    nlay = nz + max(rrtmgp_above_model_layer_counts(p_top))
    held["chem_optics/merra_tables"] = (7*30*5*36*3 + 2*(10+36),)
    transient.update({
        "chem_optics/selected_mass_and_met": (nr+5, nz, nc),
        "chem_optics/native_band_outputs": (7, nc, nlay, 16),
        "chem_optics/finalized_gpoints": (3, nc, nlay, 256),
    })
    return held, transient


def feedback_row_count(cfg):
    """Host metadata used by both forecast admission and its allocation probe."""
    if not getattr(cfg, "aer_ra_feedback", 0):
        return 0
    from woof.chem_table import load
    return len(load(cfg).rows_for("optics.gocart"))
