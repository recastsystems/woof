"""WRF surface feedback for the admitted Noah-MP option branches.

Native sources: WRF v4.6.1 Registry/registry.noahmp and
Registry/Registry.EM_COMMON; share/interp_fcn.F:1747-1906. The native
copy_fcnm/copy_fcni assignment copies a word; it does not average it.
"""
from __future__ import annotations


def selected_noahmp(cfg):
    from woof.core.noahmp_surface_device import selected_options
    return (int(getattr(cfg, "sf_surface_physics", 0)) == 4
            and selected_options(cfg))


def feedback_fields(opt_run=3,opt_soil=1,dveg=4,soiltstep=0.0,opt_tdrn=0,noahmp_output=1,opt_irr=0,opt_crop=0):
    """The native-feedback subset of the fields the runtime allocates.

    TAUSS and RS have no native feedback flag. PGS, SMOISEQ, RECH and
    SMCWTD also have no feedback flag and must retain the parent's words.
    RUN5 only feeds back ZWT, terrain and the four geography fields; the
    other groundwater carriers have native restart/history flags only.
    """
    from woof.core.noahmp_runtime import (
        NOAHMP_DIAGNOSTICS_2D, NOAHMP_STATE_2D,
        NOAHMP_STATE_SNOW_3D, NOAHMP_STATE_SNOWSOIL_3D)
    from woof.core.noahmp_runtime import state_fields_2d,diagnostic_fields_2d
    from woof.core.noahmp_soil_options import soil_input_shapes
    from woof.core.noahmp_soil_cadence import shapes as cadence_shapes
    from woof.core.noahmp_output import shapes as output_shapes
    from woof.core.noahmp_irrigation_slab import MAPS,COUNTERS,ACCUMULATED
    from woof.core.noahmp_groundwater_slab import CARRIERS as GW_CARRIERS
    groundwater_no_feedback = set(GW_CARRIERS) - {"zwtxy","ht","fdepthxy","eqzwt","rechclim","riverbedxy"} if int(opt_run)==5 else set()
    irrigation=(*MAPS,*COUNTERS,*ACCUMULATED) if int(opt_irr)>0 else ()
    return tuple(name for name in (
        *irrigation,
        *(("grainxy","gddxy","croptype","planting","harvest","season_gdd") if opt_crop else ()),
        *output_shapes(noahmp_output,(1,1)),
        *cadence_shapes(soiltstep,(1,1)),
        *soil_input_shapes(opt_soil,(1,1)),
        *state_fields_2d(opt_run,dveg,opt_tdrn,opt_irr,opt_crop), *diagnostic_fields_2d(dveg,opt_crop),
        *(("lfmassxy","rtmassxy","stmassxy","woodxy","stblcpxy","fastcpxy") if opt_crop and dveg not in (2,5,6) else ()),
        *NOAHMP_STATE_SNOW_3D, *NOAHMP_STATE_SNOWSOIL_3D,
        "isnowxy", "tslb", "smois", "sh2o", "snow", "snowh", "canwat",
        "tsk", "snowc", "acsnow", "acsnom") if not (int(opt_soil)==4 and name.endswith("_3d")) and name not in ("taussxy", "rs", "smcwtdxy", "rechxy", "smoiseq") and name not in groundwater_no_feedback)


def feedback_selected_noahmp(node, reg, *, spec_zone, chunk_shape):
    """Commit bounded words into the canonical parent, including host stores.

    The option-off path returns before accessing a field or allocating an
    operand. One-way coupling returns in NestCoupler.feedback_commit.
    """
    import cupy as cp
    import numpy as np
    from woof.core.nest_interp import feedback_parent_bounds
    from woof.core.nest_operands import NestWindowSource

    parent = node.parent
    pcfg, ccfg = parent.cfg.run, node.cfg.run
    if not (selected_noahmp(pcfg) or selected_noahmp(ccfg)):
        return ()
    if any(int(getattr(cfg, "sf_surface_physics", 0)) != 4
           for cfg in (pcfg, ccfg)):
        raise NotImplementedError(
            "selected Noah-MP feedback requires Noah-MP on both domains")
    source, target = NestWindowSource(node.state), NestWindowSource(parent.state)
    # NestWindowSource resolves both owned and legacy published stores. A
    # resident domain has no store and reads its attached fields directly.
    snow_modes = tuple(int(getattr(cfg, "opt_alb", 2)) == 3 for cfg in (pcfg, ccfg))
    if any(snow_modes) and not all(snow_modes):
        raise ValueError("snow from papers requires opt_alb=3 on both feedback domains")
    names = feedback_fields(getattr(pcfg,"opt_run",3),getattr(pcfg,"opt_soil",1),getattr(pcfg,"dveg",4),getattr(pcfg,"soiltstep",0.0),getattr(pcfg,"opt_tdrn",0),getattr(pcfg,"noahmp_output",1),getattr(pcfg,"opt_irr",0),getattr(pcfg,"opt_crop",0))
    if all(snow_modes):
        names += ("taussxy", "snicar_statexy")  # effective radius is physical state for this option
    # Validate the complete transaction before the first surface field write.
    for name in names:
        a, b = target.array(f"fields/{name}"), source.array(f"fields/{name}")
        if a is None or b is None:
            raise RuntimeError(f"Noah-MP nest feedback carrier missing: {name}")
        dtype = np.dtype(np.int32 if name in ("isnowxy","irnumsi","irnummi","irnumfi") else np.float32)
        if a.dtype != dtype or b.dtype != dtype or a.shape[:-2] != b.shape[:-2]:
            raise ValueError(f"Noah-MP nest feedback ABI differs: {name}")
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=spec_zone)
    py, px = (max(1, int(chunk_shape[0]) // reg.nrj),
              max(1, int(chunk_shape[1]) // reg.nri))
    # These offsets are the native expressions, including ratios other than
    # three: odd offsets are ISTAG/JSTAG, not an assumed ratio//2.
    oy = (0 if reg.ystag else 1) if reg.nrj % 2 else reg.nrj // 2 - 1
    ox = (0 if reg.xstag else 1) if reg.nrj % 2 else reg.nri // 2 - 1
    for j in range(jlo, jhi + 1, py):
        for i in range(ilo, ihi + 1, px):
            je, ie = min(j + py, jhi + 1), min(i + px, ihi + 1)
            y0, x0 = ((j + 1 - reg.j_parent_start) * reg.nrj + oy,
                      (i + 1 - reg.i_parent_start) * reg.nri + ox)
            y1, x1 = y0 + (je - j - 1) * reg.nrj + 1, x0 + (ie - i - 1) * reg.nri + 1
            if min(y0, x0) < 0 or y1 > reg.nyc or x1 > reg.nxc:
                raise ValueError("Noah-MP feedback donor lies outside child")
            donor, window = (slice(y0, y1), slice(x0, x1)), (slice(j, je), slice(i, ie))
            for name in names:
                key = f"fields/{name}"
                block = source.device(key, donor)
                picked = cp.ascontiguousarray(block[..., ::reg.nrj, ::reg.nri])
                target.write(key, window, picked)
    return dict(fields=names, sync_bytes=(source.host_to_device_bytes
                + target.host_to_device_bytes + target.device_to_host_bytes))
