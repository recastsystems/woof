"""WRF v4.7.1 Noah mosaic initialisation, restart=False.

Category arrays retain only the first mosaic_cat of WRF's NLCAT entries.
full=True exposes both full arrays for independent oracle verification.
EM initialisation passes ZNT as Z0, so both roughness tile arrays copy ZNT.
RC/LAI mosaic and RS/XLAIDYN are deliberately absent: glacial RC in WRF
is undefined or stale (D3), and these outputs have no woof consumer.
The kernel transcribes lsm_mosaic and the ordinary/glacial SFLX subtrees.
D1 uses 4*t+ns in the final-tile soil reduction. D2 weights per-tile increments
before adding once to cell accumulators. D4: a tiled cell whose tile weights are
all zero (WRF's area averages then divide 0 by 0) runs its own IVGTYP as one
full-weight tile; see landless_tile_cells. These corrections are graded against
independent WRF controls and single-tile increment oracles. The GPU gate was
made to fail by restoring WRF's defective NS*mosaic_i grid SMOIS index, and
again by reintroducing lsm's SWDOWN>10 condition into mosaic snow damping.
All non-held-out output words match the committed -O0 gfortran fixtures.
CuPy FTZ divergences are pinned by exact column/field set and GPU word hashes.
The CPU oracle gate was mutation-tested on test host A: replacing WRF's reciprocal
then multiply with per-entry division failed all 16 initialisation fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

MOSAIC_TILE_FIELDS: tuple[str, ...] = tuple((
    "tsk qsfc canwat snow snowh snowc albedo albbck emiss embck znt z0 "
    "hfx qfx lh grdflx snotime").split())
MOSAIC_TILE_FIELDS = tuple(n + "_mosaic" for n in MOSAIC_TILE_FIELDS)
MOSAIC_SOIL_FIELDS: tuple[str, ...] = ("tslb_mosaic", "smois_mosaic", "sh2o_mosaic")
MOSAIC_CATEGORY_FIELDS: dict[str, str] = {"mosaic_cat_index": "int32", "landusef2": "float32"}
MOSAIC_URBAN_TILE_FIELDS = tuple(n + "_urb2d_mosaic" for n in
    "tr tb tg tc qc uc ts ts_rul sh lh g rn".split())
MOSAIC_URBAN_TILE_FIELDS = tuple("ts_rul2d_mosaic" if n == "ts_rul_urb2d_mosaic" else n
                                for n in MOSAIC_URBAN_TILE_FIELDS)
MOSAIC_URBAN_SOIL_FIELDS = tuple(n + "_urb3d_mosaic" for n in ("trl", "tbl", "tgl"))


@dataclass(frozen=True)
class MosaicCategories:
    isurban: int
    isice: int
    iswater: int
    natural: int
    lcz: tuple[int, ...]


def load_mosaic_categories(mminlu: str, *, isurban: int, isice: int,
                           iswater: int, tbl_dir=None) -> MosaicCategories:
    from .noah import TBL_DIR
    from woof.tables import table_path
    lines = table_path("noah", Path(tbl_dir or TBL_DIR) / "VEGPARM.TBL").read_text(
        encoding="ascii").splitlines()
    try:
        start = next(i for i, s in enumerate(lines) if s.strip() == mminlu)
    except StopIteration:
        raise ValueError(f"{mminlu}: no VEGPARM block; category identities are undefined") from None
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() == "Vegetation Parameters"), len(lines))
    values = {}
    for i in range(start + 1, end - 1):
        key = lines[i].strip()
        if key == "NATURAL" or key.startswith("LCZ_"):
            values[key] = int(lines[i + 1].strip())
    if "NATURAL" not in values:
        raise ValueError(f"{mminlu}: missing NATURAL prevents urban category remapping")
    return MosaicCategories(isurban, isice, iswater, values["NATURAL"],
                            tuple(values[f"LCZ_{i}"] for i in range(1, 12)
                                  if f"LCZ_{i}" in values))


def mosaic_array_shapes(mosaic_cat: int, ny: int, nx: int, *, urban: bool = False) -> dict[str, tuple[tuple[int, ...], str]]:
    if mosaic_cat < 1:
        raise ValueError("mosaic_cat < 1 leaves no dominant tile")
    if ny < 1 or nx < 1:
        raise ValueError("empty horizontal extent leaves no land columns")
    return {**{n: ((mosaic_cat, ny, nx), "float32") for n in
               (*MOSAIC_TILE_FIELDS, *(MOSAIC_URBAN_TILE_FIELDS if urban else ()))},
            **{n: ((4 * mosaic_cat, ny, nx), "float32") for n in
               (*MOSAIC_SOIL_FIELDS, *(MOSAIC_URBAN_SOIL_FIELDS if urban else ()))},
            **{n: ((mosaic_cat, ny, nx), d) for n, d in MOSAIC_CATEGORY_FIELDS.items()}}


def real_exe_landusef(landusef, *, landmask, xice, iswater: int, islake: int,
                      isice: int, fractional_seaice: bool) -> np.ndarray:
    """Apply real.exe's LANDUSEF edits, in source order, to (category, y, x).

    Only the edits real.exe makes on woof's own category route,
    ``surface_input_source = 3`` (Registry.EM_COMMON:2506 default; IVGTYP from
    LU_INDEX, see :func:`woof.core.landuse.soil_category_matched_to_land`):

    * the lake merge, module_initialize_real.F:2863-2869 (only when ISLAKE is
      set, i.e. a land-use table with a lake category);
    * adjust_for_seaice_pre's LSMSCHEME arm, module_soil_pre.F:149-157, which
      clears XICE over LANDMASK > 0.5 before anything reads it;
    * adjust_for_seaice_post's LSMSCHEME arm, module_soil_pre.F:258-275: a
      sea-ice point becomes a one-hot ISICE fraction vector.

    The exact-50% water fix in process_percent_cat_new (module_soil_pre.F:
    455-472) is NOT applied: real.exe reaches it only under
    ``surface_input_source = 1`` (module_initialize_real.F:3033-3050), a route
    woof does not take.  This API has no TSK, so the post arm's second
    trigger (a cold open-water point, TSK below SEAICE_THRESHOLD) is not
    applied either; woof's land-use initialisation derives sea ice from XICE
    alone (:func:`woof.core.landuse._derive_categories`), and the tiles must
    agree with the categories the driver runs.
    """
    f = np.array(landusef, dtype=np.float32, copy=True)
    if f.ndim != 3:
        raise ValueError("landusef must be (nlcat, ny, nx) for category indexing")
    landmask = np.asarray(landmask, dtype=np.float32)
    xice = np.asarray(xice, dtype=np.float32)
    if landmask.shape != f.shape[1:] or xice.shape != f.shape[1:]:
        raise ValueError("LANDMASK/XICE shape differs from LANDUSEF; real.exe edits would address other columns")
    if islake >= 0:
        water, lake = f[iswater - 1], f[islake - 1]
        merged = water + lake
        # Geogrid multiplies category counts by a float32 reciprocal.
        # An entirely water-covered cell with 1 and 6 of 7 source pixels
        # therefore merges to the next float above one. Its combined area
        # is exactly one. Preserve ordinary merge bits and repair only this
        # endpoint when both source weights are valid and no land is present.
        rounded_full_water = (
            (merged == np.nextafter(np.float32(1), np.float32(np.inf)))
            & (water >= np.float32(0)) & (water <= np.float32(1))
            & (lake >= np.float32(0)) & (lake <= np.float32(1)))
        for category in range(f.shape[0]):
            if category not in (iswater - 1, islake - 1):
                rounded_full_water &= f[category] == np.float32(0)
        merged[rounded_full_water] = np.float32(1)
        f[iswater - 1] = merged
        f[islake - 1] = np.float32(0)
    # adjust_for_seaice_pre, module_soil_pre.F:149-157, runs before post.
    xice = np.where(landmask > np.float32(.5), np.float32(0), xice)
    ice = np.asarray(xice) >= np.float32(.02 if fractional_seaice else .5)
    f[:, ice] = np.float32(0)
    f[isice - 1, ice] = np.float32(1)
    return f


def _lsm_mosaic_init_reference(landusef, *, ivgtyp, xland, xice, mosaic_cat: int,
                    iswater: int, isice: int, fractional_seaice: bool,
                    tsk, tslb, smois, sh2o, snow, snowc, snowh, canwat,
                    albedo, albbck, emiss, embck, znt, full: bool = False) -> dict:
    """Transcribe module_sf_noahdrv.F:5052-5330 with scalar float32 ops."""
    f = np.array(landusef, dtype=np.float32, copy=True)
    if f.ndim != 3:
        raise ValueError("landusef must be (nlcat, ny, nx) for WRF category indexing")
    nlcat, ny, nx = f.shape
    if not 1 <= mosaic_cat <= nlcat:
        raise ValueError("mosaic_cat outside 1..nlcat reads beyond WRF LANDUSEF2")
    for name, value in (("ivgtyp", ivgtyp), ("xland", xland), ("xice", xice)):
        if np.asarray(value).shape != (ny, nx):
            raise ValueError(f"{name}: expected {(ny, nx)}; category elimination would misindex columns")
    ivgtyp = np.asarray(ivgtyp, dtype=np.int32)
    xland, xice = np.asarray(xland, dtype=np.float32), np.asarray(xice, dtype=np.float32)
    idx = np.broadcast_to(np.arange(1, nlcat + 1, dtype=np.int32)[:, None, None], f.shape).copy()
    threshold = np.float32(.02 if fractional_seaice else .5)
    for x in range(nx):
        for y in range(ny):
            # Strict swaps preserve tie order, including signed zero and NaN.
            pairs = nlcat - 1
            while pairs:
                last = 1
                for t in range(pairs):
                    if f[t, y, x] < f[t + 1, y, x]:
                        f[t, y, x], f[t + 1, y, x] = f[t + 1, y, x], f[t, y, x]
                        idx[t, y, x], idx[t + 1, y, x] = idx[t + 1, y, x], idx[t, y, x]
                        last = t + 1
                pairs = last - 1

            def rotate(t):
                value, category = f[t, y, x], idx[t, y, x]
                f[t:-1, y, x] = f[t + 1:, y, x].copy()
                idx[t:-1, y, x] = idx[t + 1:, y, x].copy()
                f[-1, y, x], idx[-1, y, x] = value, category

            if xland[y, x] < np.float32(1.5):
                if xice[y, x] >= threshold:
                    if ivgtyp[y, x] != isice:
                        for t in range(1, mosaic_cat):
                            if idx[t, y, x] == isice:
                                rotate(t)
                elif idx[0, y, x] == iswater:
                    if ivgtyp[y, x] != iswater:
                        rotate(0)
                else:
                    for t in range(1, mosaic_cat):
                        if idx[t, y, x] == iswater:
                            rotate(t)
            total = np.float32(0)
            for t in range(mosaic_cat):
                total = np.float32(total + f[t, y, x])
            if total < np.float32(1e-5):
                total = np.float32(1e-5)
            reciprocal = np.float32(np.float32(1) / total)
            for t in range(mosaic_cat):
                f[t, y, x] = np.float32(f[t, y, x] * reciprocal)
    out = {"landusef2": f[:mosaic_cat].copy(), "mosaic_cat_index": idx[:mosaic_cat].copy()}
    grid = dict(tsk=tsk, canwat=canwat, snow=snow, snowh=snowh, snowc=snowc,
                albedo=albedo, albbck=albbck, emiss=emiss, embck=embck, znt=znt, z0=znt)
    for name, value in grid.items():
        a = np.asarray(value, dtype=np.float32)
        if a.shape != (ny, nx):
            raise ValueError(f"{name}: expected grid shape {(ny, nx)}; tile copies would misindex")
        out[name + "_mosaic"] = np.broadcast_to(a, (mosaic_cat, ny, nx)).copy()
    for name, value in dict(tslb=tslb, smois=smois, sh2o=sh2o).items():
        a = np.asarray(value, dtype=np.float32)
        if a.shape != (4, ny, nx):
            raise ValueError(f"{name}: expected four soil layers; tile layers would misindex")
        out[name + "_mosaic"] = np.tile(a, (mosaic_cat, 1, 1))
    # WRF init does not assign these; explicit zero storage for later first call.
    for name in MOSAIC_TILE_FIELDS:
        out.setdefault(name, np.zeros((mosaic_cat, ny, nx), dtype=np.float32))
    if full:
        out["landusef2_full"], out["mosaic_cat_index_full"] = f, idx
        # module_sf_noahdrv.F:5302-5330, optional urban arrays assigned by init.
        for name in ("tr", "tb", "tg", "tc", "ts", "ts_rul"):
            out[name + "_urb2d_mosaic"] = out["tsk_mosaic"].copy()
        for name, value in (("qc", .01), ("sh", 0), ("lh", 0), ("g", 0), ("rn", 0)):
            out[name + "_urb2d_mosaic"] = np.full((mosaic_cat, ny, nx), value, dtype=np.float32)
        st = np.asarray(tslb, dtype=np.float32)
        wall = np.stack((st[0] + np.float32(0),
                         np.float32(.5) * (st[0] + st[1]),
                         st[1] + np.float32(0),
                         st[1] + (st[2] - st[1]) * np.float32(.29)))
        for name in ("trl", "tbl"):
            out[name + "_urb3d_mosaic"] = np.tile(wall, (mosaic_cat, 1, 1))
        out["tgl_urb3d_mosaic"] = out["tslb_mosaic"].copy()
    return out


def landless_tile_cells(tile_fractions, *, xland, xice,
                        fractional_seaice: bool) -> np.ndarray:
    """Cells lsm_mosaic tiles whose normalised tile weights are all zero.

    lsm_mosaic tiles land cells below the sea-ice threshold
    (module_sf_noahdrv.F:3169); ``tile_fractions`` is the normalised
    ``(mosaic_cat, ny, nx)`` LANDUSEF2 prefix.  WRF's own init floors the
    total at 1e-5 (:5246-5268) but a total of exactly zero stays zero, so
    these are the cells where WRF's area averages divide 0 by 0.
    """
    f = np.asarray(tile_fractions, dtype=np.float32)
    xland = np.asarray(xland, dtype=np.float32)
    xice = np.asarray(xice, dtype=np.float32)
    tiled = (xland < np.float32(1.5)) & (
        xice < np.float32(.02 if fractional_seaice else .5))
    return tiled & np.all(f == np.float32(0), axis=0)


def lsm_mosaic_init(landusef, *, ivgtyp, xland, xice, mosaic_cat: int,
                    iswater: int, isice: int, fractional_seaice: bool,
                    tsk, tslb, smois, sh2o, snow, snowc, snowh, canwat,
                    albedo, albbck, emiss, embck, znt, full: bool = False) -> dict:
    """WRF init with stable sorting and masked rotations over all cells.

    Bitwise :func:`_lsm_mosaic_init_reference` (the scalar transcription of
    WRF's restart=.false. arm) everywhere except the cells
    :func:`landless_tile_cells` names, where WRF leaves every tile at zero
    weight and its first step divides zero by zero; there the first tile
    takes the cell's own IVGTYP at weight 1.
    """
    f = np.array(landusef, dtype=np.float32, copy=True)
    if f.ndim != 3:
        raise ValueError("landusef must be (nlcat, ny, nx) for WRF category indexing")
    nlcat, ny, nx = f.shape
    if not 1 <= mosaic_cat <= nlcat:
        raise ValueError("mosaic_cat outside 1..nlcat reads beyond WRF LANDUSEF2")
    for name, value in (("ivgtyp", ivgtyp), ("xland", xland), ("xice", xice)):
        if np.asarray(value).shape != (ny, nx):
            raise ValueError(f"{name}: expected {(ny, nx)}; category elimination would misindex columns")
    ivgtyp = np.asarray(ivgtyp, dtype=np.int32)
    xland, xice = np.asarray(xland, dtype=np.float32), np.asarray(xice, dtype=np.float32)
    if np.isnan(f).any():
        raise ValueError("NaN LANDUSEF has no category order: stable sort differs from WRF strict-< swaps")
    order = np.argsort(-f, axis=0, kind="stable")
    f = np.take_along_axis(f, order, axis=0)
    idx = (order + 1).astype(np.int32)
    land = xland < np.float32(1.5)
    ice = land & (xice >= np.float32(.02 if fractional_seaice else .5))
    ordinary = land & ~ice
    water_dominant = idx[0] == iswater

    def rotate(t, mask):
        if not mask.any():
            return
        value, category = f[t, mask].copy(), idx[t, mask].copy()
        f[t:-1, mask] = f[t+1:, mask]
        idx[t:-1, mask] = idx[t+1:, mask]
        f[-1, mask], idx[-1, mask] = value, category

    rotate(0, ordinary & water_dominant & (ivgtyp != iswater))
    for t in range(1, mosaic_cat):
        rotate(t, (ice & (ivgtyp != isice) & (idx[t] == isice)) |
                  (ordinary & ~water_dominant & (idx[t] == iswater)))
    total = np.zeros((ny, nx), dtype=np.float32)
    for t in range(mosaic_cat):
        total = total + f[t]
    total = np.maximum(total, np.float32(1e-5))
    reciprocal = np.float32(1) / total
    f[:mosaic_cat] = f[:mosaic_cat] * reciprocal[None]
    # DEFINED WHERE WRF DIVIDES ZERO BY ZERO.  A tiled cell (land, not sea
    # ice) whose first mosaic_cat fractions are all zero -- a point the
    # land/soil reconciliation turned from water into land, whose LANDUSEF is
    # all water -- keeps zero weights in WRF's init, and lsm_mosaic then
    # forms TSK = (0/0)**0.25 and ZNT = EXP(0/0) (module_sf_noahdrv.F:4195,
    # 4186), a NaN the health gate stops the run on.  Here the cell runs as
    # its own dominant category, one tile at full weight: what the
    # dominant-category Noah column runs there.  See landless_tile_cells.
    landless = landless_tile_cells(f[:mosaic_cat], xland=xland, xice=xice,
                                   fractional_seaice=fractional_seaice)
    f[0, landless] = np.float32(1)
    idx[0, landless] = ivgtyp[landless]
    out = {"landusef2": f[:mosaic_cat].copy(), "mosaic_cat_index": idx[:mosaic_cat].copy()}
    grid = dict(tsk=tsk, canwat=canwat, snow=snow, snowh=snowh, snowc=snowc,
                albedo=albedo, albbck=albbck, emiss=emiss, embck=embck, znt=znt, z0=znt)
    for name, value in grid.items():
        a = np.asarray(value, dtype=np.float32)
        if a.shape != (ny, nx):
            raise ValueError(f"{name}: expected grid shape {(ny, nx)}; tile copies would misindex")
        out[name + "_mosaic"] = np.broadcast_to(a, (mosaic_cat, ny, nx)).copy()
    for name, value in dict(tslb=tslb, smois=smois, sh2o=sh2o).items():
        a = np.asarray(value, dtype=np.float32)
        if a.shape != (4, ny, nx):
            raise ValueError(f"{name}: expected four soil layers; tile layers would misindex")
        out[name + "_mosaic"] = np.tile(a, (mosaic_cat, 1, 1))
    # WRF init does not assign these; explicit zero storage for later first call.
    for name in MOSAIC_TILE_FIELDS:
        out.setdefault(name, np.zeros((mosaic_cat, ny, nx), dtype=np.float32))
    if full:
        out["landusef2_full"], out["mosaic_cat_index_full"] = f, idx
        # module_sf_noahdrv.F:5302-5330, optional urban arrays assigned by init.
        for name in ("tr", "tb", "tg", "tc", "ts", "ts_rul"):
            out[name + "_urb2d_mosaic"] = out["tsk_mosaic"].copy()
        for name, value in (("qc", .01), ("sh", 0), ("lh", 0), ("g", 0), ("rn", 0)):
            out[name + "_urb2d_mosaic"] = np.full((mosaic_cat, ny, nx), value, dtype=np.float32)
        st = np.asarray(tslb, dtype=np.float32)
        wall = np.stack((st[0] + np.float32(0),
                         np.float32(.5) * (st[0] + st[1]),
                         st[1] + np.float32(0),
                         st[1] + (st[2] - st[1]) * np.float32(.29)))
        for name in ("trl", "tbl"):
            out[name + "_urb3d_mosaic"] = np.tile(wall, (mosaic_cat, 1, 1))
        out["tgl_urb3d_mosaic"] = out["tslb_mosaic"].copy()
    return out


def attach_noah_mosaic(fields: dict, *, landusef, mosaic_cat: int,
                       categories: MosaicCategories, fractional_seaice: bool,
                       lucats: int, urban: bool = False) -> None:
    names = (*MOSAIC_TILE_FIELDS, *MOSAIC_SOIL_FIELDS, *MOSAIC_CATEGORY_FIELDS,
             *(MOSAIC_URBAN_TILE_FIELDS if urban else ()),
             *(MOSAIC_URBAN_SOIL_FIELDS if urban else ()))
    if any(n in fields for n in names):
        raise ValueError("mosaic arrays already present: double attach would destroy evolved tile state")
    def host(a):
        return a.get() if hasattr(a, "get") else np.asarray(a)
    state = {n: host(fields[n]) for n in (
        "ivgtyp xland xice tsk tslb smois sh2o snow snowc snowh canwat "
        "albedo albbck emiss embck znt").split()}
    fractions = host(landusef)
    if fractions.ndim != 3 or fractions.shape[1:] != state["tsk"].shape:
        raise ValueError("landusef not (nlcat, ny, nx): WRF category indexing would misread columns")
    tiles = lsm_mosaic_init(fractions, **state, mosaic_cat=mosaic_cat,
                            iswater=categories.iswater, isice=categories.isice,
                            fractional_seaice=fractional_seaice, full=urban)
    if urban:
        tiles.pop("landusef2_full")
        tiles.pop("mosaic_cat_index_full")
        tiles["ts_rul2d_mosaic"] = tiles.pop("ts_rul_urb2d_mosaic")
        # UC is Registry zero, not assigned by lsm_mosaic_init:5302-5330.
        tiles["uc_urb2d_mosaic"] = np.zeros_like(tiles["tsk_mosaic"])
    cat = tiles["mosaic_cat_index"]
    urban_tiles = np.isin(cat, (categories.isurban, *categories.lcz))
    land = (state["xland"] < np.float32(1.5)) & (state["xice"] < np.float32(.02 if fractional_seaice else .5))
    if np.any((cat > lucats) & ~urban_tiles & land[None]):
        raise ValueError("land tile category beyond LUCATS: WRF would read VEGPARM past its rows")
    if hasattr(fields["tsk"], "get"):
        import cupy as cp
        tiles = {n: cp.asarray(a) for n, a in tiles.items()}
    fields.update(tiles)


def mosaic_tile_utype_lookup(categories: MosaicCategories,
                             use_wudapt_lcz: int) -> np.ndarray:
    """Category -> the UTYPE lsm_mosaic gives an urban tile (0 = not urban).

    module_sf_noahdrv.F:3745-3767 as noah_mosaic.cu decides it: ISURBAN is
    type 5 with the LCZ table and 2 without, LCZ_n is type n, and an LCZ
    match is taken after the ISURBAN test.
    """
    size = max((categories.isurban, *categories.lcz)) + 1
    table = np.zeros(size, dtype=np.int32)
    table[categories.isurban] = 5 if int(use_wudapt_lcz) else 2
    for n, lcz in enumerate(categories.lcz, start=1):
        table[lcz] = n
    return table


def every_tile_urban_fraction(frc_urb2d, *, utype_urb2d, mosaic_cat_index,
                              landusef2, utype_lookup, frc_table, land):
    """FRC_URB2D under ``mosaic_urban_canopy = "every_tile"`` (host arrays).

    WRF's urban_var_init leaves FRC_URB2D at the table or input fraction
    where the cell's DOMINANT category is urban and sets it to 0 everywhere
    else (module_sf_urban.F:2811-2821); lsm_mosaic then blends each urban
    tile's canopy with that one grid fraction (module_sf_noahdrv.F:
    3941-3955), so a town tile in a mostly rural cell is blended at weight 0
    and runs as vegetation.  The town rule keeps every dominant-urban cell
    exactly as WRF left it and gives each other LAND cell that carries an
    urban tile of positive weight its urban type's own table fraction,
    URBPARM's FRC_URB.  The type is the one lsm_mosaic gives the tile
    (``utype_lookup``, the kernel's ISURBAN -> 2 or 5 and LCZ_n -> n); a
    cell with two urban tiles takes the larger one's, the first in
    lsm_mosaic_init's descending order, as WRF's dominant cells share one
    fraction between their urban tiles.

    ``frc_urb2d``/``utype_urb2d``/``land`` are (ny, nx); ``mosaic_cat_index``
    and ``landusef2`` are (mosaic_cat, ny, nx).  Returns a new float32 array.
    """
    frc = np.array(frc_urb2d, dtype=np.float32, copy=True)
    cat = np.asarray(mosaic_cat_index).astype(np.int64)
    weight = np.asarray(landusef2, dtype=np.float32)
    lookup = np.asarray(utype_lookup, dtype=np.int32)
    table = np.asarray(frc_table, dtype=np.float32)
    inside = (cat >= 0) & (cat < lookup.size)
    tile_utype = np.where(inside, lookup[np.clip(cat, 0, lookup.size - 1)], 0)
    present = (tile_utype > 0) & (weight > np.float32(0.0))
    town = (np.asarray(utype_urb2d) == 0) & np.asarray(land, bool) & present.any(axis=0)
    first = np.argmax(present, axis=0)
    utype = np.take_along_axis(tile_utype, first[None], axis=0)[0]
    if np.any(town & (utype > table.size)):
        raise ValueError(
            f"an urban tile of type {int(utype[town].max())} has no row in "
            f"the {table.size}-row urban table: the town rule would read its "
            "fraction past the end of URBPARM (set use_wudapt_lcz = 1 for "
            "Local Climate Zone land cover, WRF's own fatal for this table)")
    frc[town] = table[utype[town] - 1]
    return frc


def launch_noah_mosaic(dev: dict, params, dt: float, dzs, *, mosaic_cat: int,
                       categories: MosaicCategories, xice_threshold: float,
                       frpcpn: bool, usemonalb: bool, rdlai2d: bool,
                       opt_thcnd: int, itimestep: int, urban=None,
                       atmosphere=None, solar=None, use_wudapt_lcz: int = 0,
                       urban_water_balance=None) -> None:
    """Launch one thread per column, tile work in a device function.

    WRF sf_urban_physics=0 or 1, four soil layers. Glacial tiles and the mosaic
    LAI/SOILW and CQS2/CHS2 carries are included. UA_PHYS, FASDAS,
    WRF-Hydro and RC/LAI mosaic consumer outputs are outside this API.
    No reslin or ebal array is read or written.
    """
    import cupy as cp
    from .noah import _F2D, _F3D, _device_tables
    grid = tuple(n for n in _F2D if n != "reslin")
    ny, nx = dev["tsk"].shape
    if mosaic_cat < 1:
        raise ValueError("mosaic_cat < 1 leaves no dominant tile")
    if len(dzs) != 4:
        raise ValueError("mosaic soil state has four layers; other depths would misindex the tile body")
    args = []
    shapes = {n: ((ny, nx), "float32") for n in grid}
    shapes.update({n: ((4, ny, nx), "float32") for n in _F3D})
    shapes.update({n: ((ny, nx), "int32") for n in ("ivgtyp", "isltyp")})
    shapes.update(mosaic_array_shapes(mosaic_cat, ny, nx))
    for n in ("ivgtyp", "isltyp", *grid, *_F3D, *MOSAIC_TILE_FIELDS, *MOSAIC_SOIL_FIELDS,
              "mosaic_cat_index", "landusef2"):
        shape, dtype = shapes[n]
        if dev[n].shape != shape or dev[n].dtype != np.dtype(dtype) or not dev[n].flags.c_contiguous:
            raise ValueError(f"{n}: expected contiguous {shape} {dtype}; CUDA column indexing would misread state")
        args.append(dev[n])
    key = tuple(categories.lcz)
    if key not in _LCZ_DEVICE:
        _LCZ_DEVICE[key] = cp.asarray(key, dtype=cp.int32)
    args += [_LCZ_DEVICE[key], np.int32(len(key)), *_device_tables(params, dzs),
             np.float32(dt), np.int32(params.lucats), np.int32(params.slcats),
             np.int32(categories.isurban), np.int32(categories.isice), np.float32(xice_threshold),
             np.int32(itimestep), np.int32(frpcpn), np.int32(usemonalb), np.int32(rdlai2d),
             np.int32(opt_thcnd), np.int32(mosaic_cat), np.int32(ny), np.int32(nx)]
    if urban is None:
        function = _mosaic_kernel()
    else:
        from .urban_ucm import (
            _checked, _pointer_table, _state_arrays, _device_params,
            _status_word, _jmonth, _raise_on, water_balance_params)
        for n, (shape, dtype) in mosaic_array_shapes(mosaic_cat, ny, nx, urban=True).items():
            _checked(dev[n], shape, n, getattr(cp, dtype))
        sp = _pointer_table(_state_arrays(urban, ny, nx))
        tp = _pointer_table([dev[n] for n in (*MOSAIC_URBAN_TILE_FIELDS, *MOSAIC_URBAN_SOIL_FIELDS)])
        tab, glob, isw = _device_params(water_balance_params(
            urban.params, urban_water_balance))
        err = _status_word(urban)
        # The same UrbanCoupler-held solar object as the ordinary UCM route.
        # noahdrv.F:3790-3794 sets LSOLAR=.false.; urban's active arm reads
        # HRANG and calendar month, not DECLIN/COSZEN/XLAT.
        args += [sp, tp, tab, glob, isw, urban.fields["frc_urb2d"],
                 cp.ascontiguousarray(atmosphere["u"][0]),
                 cp.ascontiguousarray(atmosphere["v"][0]),
                 _checked(solar.hrang, (ny, nx), "hrang"), dev["ust"],
                 np.int32(categories.natural), np.int32(use_wudapt_lcz),
                 np.int32(_jmonth(solar)), err]
        function = _mosaic_ucm_module().get_function("noah_mosaic_ucm_column")
    function(((ny*nx + 31)//32,), (32,), tuple(args))
    if urban is not None:
        _raise_on(err)


_LCZ_DEVICE = {}

#: NVRTC options for the mosaic module.  Every float operation in the kernel
#: is already an explicit round-to-nearest intrinsic; ``--fmad=false`` is the
#: second lock that keeps any expression written without one from being
#: contracted into an FMA, which gfortran -O0 never forms.
MOSAIC_NVRTC_OPTIONS = ("-std=c++17", "--fmad=false")


@lru_cache(maxsize=None)
def _mosaic_module():
    """Compile ``noah_mosaic.cu`` once, through the recorded, observed path.

    A dedicated loader (as :func:`woof.core.nest_interp._nest_module`)
    because this module alone needs ``--fmad=false`` beside the loader's
    ``-std=c++17``; the source is :func:`woof.core.kernels.module_source`'s,
    so the prepended glibc header is the one the allow-list names.
    """
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    from woof.core.kernels import MODULE_KEY_ROOT, _compile_observed, module_source

    source = module_source("noah_mosaic")
    module = cp.RawModule(code=source, options=MOSAIC_NVRTC_OPTIONS,
                          name_expressions=None)
    key = f"{MODULE_KEY_ROOT}:noah_mosaic"
    _compile_observed(module, key)
    record_module(key, source=source, options=MOSAIC_NVRTC_OPTIONS,
                  module=module)
    return module


def _mosaic_kernel():
    return _mosaic_module().get_function("noah_mosaic_column")


def mosaic_ucm_source():
    """One glibc header, the unchanged UCM, then the mosaic tile loop."""
    from .kernels import _KDIR, module_source
    return ("#define NOAH_MOSAIC_UCM 1\n" + module_source("urban_ucm")
            + "\n" + (_KDIR / "noah_mosaic.cu").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _mosaic_ucm_module():
    import cupy as cp
    from woof.cuda_subnormals import install
    install()
    from woof.certify.kernel_manifest import record_module
    from .kernels import MODULE_KEY_ROOT, _compile_observed
    source = mosaic_ucm_source()
    module = cp.RawModule(code=source, options=MOSAIC_NVRTC_OPTIONS,
                          name_expressions=None)
    key = f"{MODULE_KEY_ROOT}:noah_mosaic_ucm"
    _compile_observed(module, key)
    record_module(key, source=source, options=MOSAIC_NVRTC_OPTIONS, module=module)
    return module
