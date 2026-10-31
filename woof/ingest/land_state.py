"""Soil moisture across land-surface models: index remap and same-model donors.

The defect this module fixes (found 2026-10-26): a run
started from ERA5 copied ERA5's HTESSEL volumetric soil moisture straight into
RUC.  HTESSEL rates the Oklahoma soils of 2024-05-06 at field capacity (top
level 0.36); HRRR's own cycled RUC analysis puts the same cells at 0.21.  With
that much water evaporation took the afternoon sun (latent heat 438 W/m2,
sensible 5 W/m2 at +1 h), T2 ran 1-2 C cold against ASOS from the first hour
and no storms formed.  Swapping only the soil moisture for HRRR's removed the
cold bias and let the storm line form.

Volumetric soil moisture is specific to each land model's own climate: the
same field is "wet" in one model and "half way" in another.  Two rules follow,
both read from ``woof/data/soil/land-state.v1.json`` so a new source is a
table row, never a code path (the arbitrary acceptance test):

1. **Soil-moisture index remap** (every route, every source whose soil model
   differs from the target's, except a pair the table's ``copy_pairs``
   copies: RUC into Noah, where the remap dried Noah's soil and the 2 m dew
   point against ASOS, WOOF NYC 2026-10-28 12Z).  The source model is recognised by the depth
   ladder its soil fields carry (HTESSEL's 0-7-28-100-289 cm, Noah's
   0-10-40-100-200 cm, RUC's nine nodes).  Each layer is carried across as
   the soil-moisture index ``SMI = (theta - WP_s) / (FC_s - WP_s)`` and
   rebuilt on the target's hydraulics ``theta = WP_t + SMI (FC_t - WP_t)``,
   bounded by the target's air-dry and saturation values.  Layers are mapped
   by depth afterwards by the existing ``init_soil_*_real`` remaps.
2. **Same-model donor** (where one covers the domain).  A cycled analysis
   from the TARGET's own land model -- HRRR's RUC soil for a RUC run over
   CONUS -- replaces the converted moisture on every target land cell it
   covers.  The index remap alone recovered about 14 % of the measured error
   (lane SMI probe); the same-model analysis recovered all of it.  Cells the
   donor does not cover keep the remapped value.

``WOOF_SOIL_LAND_STATE=off`` is the opt-out (the stock-WRF copy).  A donor
that cannot be fetched or decoded falls back to the remapped value for that
run and says so on stderr; it never stops a preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Mapping

import numpy as np

_TABLE = Path(__file__).resolve().parent.parent / "data" / "soil" / "land-state.v1.json"
_SCHEMA = "gpuwm-land-state-v1"
_LAYER_NAME = re.compile(r"^(?P<prefix>(?:[A-Z0-9]+_)?)SM(?P<top>\d{3})(?P<bottom>\d{3})$")
_STAS_WATER = 14


@dataclass(frozen=True)
class LandStateContext:
    """Where and when the target domain is, for a donor analysis.

    The mass-point latitudes and longitudes are read from the grid only when
    a donor needs them: a start that copies or remaps its own soil never asks,
    and every preparation door builds this context (2.9.9 pass 1: an eager
    ``grid.latlon_mass()`` refused the doors' own grid stand-ins, 166 tests).
    """

    grid: object
    valid_time: datetime
    cache_dir: Path | None = None

    @property
    def latitude(self) -> np.ndarray:
        return self._latlon()[0]

    @property
    def longitude(self) -> np.ndarray:
        return self._latlon()[1]

    def _latlon(self):
        cached = self.__dict__.get("_latlon_cache")
        if cached is None:
            lat, lon = self.grid.latlon_mass()
            cached = (np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64))
            object.__setattr__(self, "_latlon_cache", cached)
        return cached


def land_state_context(grid, valid_time, cache_dir=None) -> LandStateContext | None:
    """Context from a projected grid (``latlon_mass``, read when a donor needs it) and the start time.

    ``None`` when the valid time is unknown (the donor then says so)."""

    if valid_time is None:
        return None
    return LandStateContext(grid, valid_time, cache_dir)


_RULES: dict | None = None


def load_rules(path: Path | None = None) -> dict:
    global _RULES
    if path is None and _RULES is not None:
        return _RULES
    rules = json.loads(Path(path or _TABLE).read_text(encoding="utf-8"))
    if rules.get("schema") != _SCHEMA:
        raise ValueError(f"{path or _TABLE}: schema is not {_SCHEMA}")
    models = rules["soil_models"]
    for target in rules["targets"].values():
        if target not in models:
            raise ValueError(f"land-state target model {target!r} has no soil_models row")
    for donor in rules["donors"]:
        if donor["soil_model"] not in models:
            raise ValueError(f"donor {donor['id']}: unknown soil model {donor['soil_model']!r}")
        if not models[donor["soil_model"]].get("node_ladders_cm"):
            # A donor is written onto the target's own levels unchanged, so
            # only a node model (whose nodes ARE the target levels) can be
            # one; a layer donor would need a vertical remap this seam does
            # not run.
            raise ValueError(f"donor {donor['id']}: soil model {donor['soil_model']!r} "
                             "publishes no node ladder, so it cannot be written "
                             "onto the target levels unchanged")
        if list(donor["fields"]) != ["soil_moisture"]:
            raise ValueError(f"donor {donor['id']}: only soil_moisture is implemented")
    for pair in rules.get("copy_pairs", ()):
        for side in ("source", "target"):
            if pair[side] not in models:
                raise ValueError(f"copy pair {pair['source']}->{pair['target']}: "
                                 f"unknown soil model {pair[side]!r}")
    if path is None:
        _RULES = rules
    return rules


def _disabled() -> bool:
    return os.environ.get(load_rules()["opt_out_env"], "").strip().lower() in {
        "off", "0", "false", "no", "copy"}


def _host(value) -> np.ndarray:
    from woof.ingest.soil import _host as soil_host
    return soil_host(value)


def _note(message: str, *, warning: bool = False) -> None:
    print(("warning: " if warning else "note: ") + message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Source and target soil models.
# ---------------------------------------------------------------------------

def _model_for_ladder(rules, *, layers=None, nodes=None) -> str | None:
    for name, row in rules["soil_models"].items():
        if layers is not None:
            for ladder in row.get("layer_ladders_cm", ()):
                if [list(map(int, pair)) for pair in ladder] == [list(pair) for pair in layers]:
                    return name
        if nodes is not None:
            for ladder in row.get("node_ladders_cm", ()):
                if list(map(int, ladder)) == list(nodes):
                    return name
    return None


def identify_source_soil(fields: Mapping[str, object], soil_layer_contract=None):
    """Return ``(soil_model or None, ladder description, moisture keys)``.

    ``moisture keys`` is a tuple of ``(field name, leading index or None)``:
    a per-layer field (``SM000007``) is one key with ``None``; a stacked
    field (``SOILW`` nodes, the mapped moisture array) is one key whose
    leading axis is the ladder.
    """

    from woof.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                            soil_layer_bounds, soil_node_depths)

    rules = load_rules()
    if soil_layer_contract is not None and MAPPED_SOIL_MOISTURE in fields:
        if "source_nodes" in soil_layer_contract:
            nodes = [int(round(depth * 100.0)) for depth in soil_node_depths(soil_layer_contract)]
            return (_model_for_ladder(rules, nodes=nodes), f"nodes {nodes} cm",
                    ((MAPPED_SOIL_MOISTURE, len(nodes)),))
        bounds = [(int(round(top * 100.0)), int(round(bottom * 100.0)))
                  for top, bottom in soil_layer_bounds(soil_layer_contract, "source_layers")]
        return (_model_for_ladder(rules, layers=bounds), f"layers {bounds} cm",
                ((MAPPED_SOIL_MOISTURE, len(bounds)),))
    if "SOILW" in fields and "SOILT" in fields:
        from woof.ingest.soil import HRRR_SOIL_NODE_DEPTHS_M
        nodes = [int(round(float(depth) * 100.0)) for depth in HRRR_SOIL_NODE_DEPTHS_M]
        return (_model_for_ladder(rules, nodes=nodes), f"nodes {nodes} cm",
                (("SOILW", len(nodes)),))
    groups: dict[str, list[tuple[int, int, str]]] = {}
    for name in fields:
        match = _LAYER_NAME.match(str(name))
        if match:
            groups.setdefault(match["prefix"], []).append(
                (int(match["top"]), int(match["bottom"]), str(name)))
    if len(groups) == 1:
        layers = sorted(next(iter(groups.values())))
        bounds = [(top, bottom) for top, bottom, _ in layers]
        return (_model_for_ladder(rules, layers=bounds), f"layers {bounds} cm",
                tuple((name, None) for _, _, name in layers))
    return None, "no recognised soil moisture ladder", ()


def target_soil_model(sf_surface_physics: int) -> str | None:
    return load_rules()["targets"].get(str(int(sf_surface_physics)))


def _hydraulics(rules, model: str, soil_class: np.ndarray):
    """Per-cell ``(wilting point, field capacity, air dry, saturation)``.

    ``soil_class`` is the target grid's STAS class.  NaN where the model has
    no value for that class (water, or a class its table does not map).
    """

    spec = rules["soil_models"][model]["hydraulics"]
    nan = np.full(soil_class.shape, np.nan)
    if spec.startswith("SOILPARM:"):
        from woof.core.noah import load_tables
        table = load_tables(mminsl=spec.split(":", 1)[1])
        count = len(table.wltsmc)
        valid = (soil_class >= 1) & (soil_class <= count) & (soil_class != _STAS_WATER)
        index = np.clip(soil_class - 1, 0, count - 1)
        out = []
        for name in ("wltsmc", "refsmc", "drysmc", "maxsmc"):
            values = np.asarray(getattr(table, name), dtype=np.float64)
            out.append(np.where(valid, values[index], np.nan))
        return tuple(out)
    block = rules["hydraulics"][spec]
    mapping = block["class_from_stas"]
    wp, fc = nan.copy(), nan.copy()
    for stas, cls in mapping.items():
        if stas == "comment":
            continue
        row = block["classes"][str(cls)]
        cells = soil_class == int(stas)
        wp[cells] = row["wilting_point"]
        fc[cells] = row["field_capacity"]
    return wp, fc, nan, nan


def convert_source_soil_moisture(fields, *, sf_surface_physics, soil_type,
                                 landmask=None, lake_mask=None,
                                 soil_layer_contract=None):
    """The soil-moisture index remap on the source moisture fields.

    Returns ``(fields, receipt)``.  ``fields`` is the input mapping itself
    when nothing changes (same soil model, unknown model, opt-out), else a
    shallow copy with the moisture arrays replaced.  Temperature fields are
    never touched.
    """

    rules = load_rules()
    target = target_soil_model(sf_surface_physics)
    source, ladder, keys = identify_source_soil(fields, soil_layer_contract)
    receipt: dict[str, object] = {
        "schema": "gpuwm-land-state-receipt-v1",
        "table": _TABLE.name,
        "source_soil_model": source,
        "source_ladder": ladder,
        "target_soil_model": target,
    }
    if _disabled():
        receipt["moisture"] = f"copied ({rules['opt_out_env']}=off)"
        return fields, receipt
    if target is None or source is None:
        receipt["moisture"] = "copied: no soil model row for " + (
            "the target land surface" if target is None else "this source ladder")
        if source is None and keys == () and target is not None:
            receipt["moisture"] = "copied: no soil moisture ladder in the source fields"
        return fields, receipt
    if source == target:
        receipt["moisture"] = "copied: source and target share a soil model"
        return fields, receipt
    if any(pair["source"] == source and pair["target"] == target
           for pair in rules.get("copy_pairs", ())):
        # A pair the table copies (RUC into Noah: the index remap scored worse
        # against ASOS than the stock-WRF copy, WOOF NYC 2026-10-28 12Z).
        receipt["moisture"] = f"copied: the table's copy pair {source} -> {target}"
        return fields, receipt

    soil_class = np.rint(np.asarray(_host(soil_type), dtype=np.float64)).astype(np.int64)
    wp_s, fc_s, _, _ = _hydraulics(rules, source, soil_class)
    wp_t, fc_t, dry_t, sat_t = _hydraulics(rules, target, soil_class)
    if np.isnan(dry_t).all():
        raise ValueError(f"target soil model {target!r} must carry air-dry/saturation values")
    land = np.ones(soil_class.shape, dtype=bool)
    if landmask is not None:
        land &= _host(landmask) >= 0.5
    if lake_mask is not None:
        land &= ~_host(lake_mask).astype(bool)
    usable = (land & np.isfinite(wp_s) & np.isfinite(fc_s) & (fc_s > wp_s)
              & np.isfinite(wp_t) & np.isfinite(fc_t))
    out = dict(fields)
    before, after, clipped = [], [], 0
    for name, _count in keys:
        values = _host(fields[name])
        dtype = values.dtype if values.dtype.kind == "f" else np.float32
        theta = np.asarray(values, dtype=np.float64)
        cells = usable & np.isfinite(theta).all(axis=0) if theta.ndim == 3 else usable & np.isfinite(theta)
        smi = (theta - wp_s) / np.where(cells, fc_s - wp_s, 1.0)
        raw = wp_t + smi * (fc_t - wp_t)
        bounded = np.clip(raw, dry_t, sat_t)
        mask = np.broadcast_to(cells, theta.shape)
        clipped += int(np.count_nonzero(mask & (bounded != raw)))
        converted = np.where(mask, bounded, theta).astype(dtype)
        if theta.ndim == 3:
            before += [float(theta[k][cells].mean()) for k in range(theta.shape[0])] if cells.any() else []
            after += [float(converted[k][cells].mean()) for k in range(theta.shape[0])] if cells.any() else []
        elif cells.any():
            before.append(float(theta[cells].mean()))
            after.append(float(converted[cells].mean()))
        out[name] = converted
    receipt.update({
        "moisture": "soil-moisture index remap",
        "formula": "theta_t = WP_t + (theta_s - WP_s)/(FC_s - WP_s) * (FC_t - WP_t), bounded by the target air-dry/saturation",
        "land_cells_converted": int(np.count_nonzero(usable)),
        "land_cells_without_hydraulics": int(np.count_nonzero(land & ~usable)),
        "layer_mean_before": [round(value, 4) for value in before],
        "layer_mean_after": [round(value, 4) for value in after],
        "values_bounded": clipped,
    })
    if before:
        _note(f"soil moisture: {source} -> {target} soil-moisture index remap on "
              f"{receipt['land_cells_converted']} land cell(s); top layer mean "
              f"{before[0]:.3f} -> {after[0]:.3f} (woof/data/soil/land-state.v1.json)")
    return out, receipt


# ---------------------------------------------------------------------------
# Same-model donor analyses.
# ---------------------------------------------------------------------------

def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _cache_root(context: LandStateContext) -> Path:
    if context.cache_dir is not None:
        return Path(context.cache_dir)
    explicit = os.environ.get("WOOF_LAND_STATE_CACHE")
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "woof" / "land-state"


def _fetch_donor_file(donor, cycle: datetime, root: Path) -> Path:
    from tools import download_hrrr_native_subset as transport

    url = donor["transport"]["url"].format(cycle=cycle)
    folder = root / donor["id"] / f"{cycle:%Y%m%d%H}"
    destination = folder / "soil.grib2"
    if destination.is_file():
        with destination.open("rb") as stream:
            head = stream.read(4)
            stream.seek(-4, os.SEEK_END)
            tail = stream.read(4)
        if head == b"GRIB" and tail == b"7777":
            return destination
        destination.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder / "soil.grib2.idx"
    if index_path.exists():
        index_path.unlink()
    transport._download_subset(
        url=url, index_url=url + donor["transport"]["index_suffix"],
        index_path=index_path, destination=destination,
        kind=donor["transport"]["kind"], workers=8, retries=5)
    return destination


def _decode_donor(donor, path: Path):
    """``(moisture (L, ny, nx) float32, level depths m, projection)``."""

    from woof.grib2_stack import stack_grib2

    spec = donor["grib2"]["soil_moisture"]
    selector = dict(key="soil_moisture", discipline=spec["discipline"],
                    category=spec["category"], parameter=spec["parameter"],
                    constituent_type=None, aerosol_type=None,
                    level_type=spec["level_type"], levels=None)
    scratch = Path(tempfile.mkdtemp(prefix=".land-state-", dir=path.parent))
    try:
        (field,) = stack_grib2(path, [selector], scratch / "decoded")
        if field.projection is None:
            raise ValueError(f"donor {donor['id']}: expected a Lambert grid")
        levels = np.asarray(field.levels, dtype=np.float64)
        order = np.argsort(levels)
        values = np.array(field.array, dtype=np.float32)[order]
        projection = dict(field.projection)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return values, levels[order], projection


def _donor_grid(projection, earth_radius_m: float):
    from woof.static.lambert import LambertGrid
    from woof.static.projection import EARTH_RADIUS_M

    def wrap(lon):
        return ((float(lon) + 180.0) % 360.0) - 180.0

    scale = EARTH_RADIUS_M / float(earth_radius_m)
    dx = float(projection["dlon"]) * scale
    dy = float(projection["dlat"]) * scale
    return LambertGrid(
        ref_lat=float(projection["lat1"]), ref_lon=wrap(projection["lon1"]),
        truelat1=float(projection["latin1"]), truelat2=float(projection["latin2"]),
        stand_lon=wrap(projection["lov"]), dx=dx, dy=dy,
        e_we=int(projection["nx"]) + 1, e_sn=int(projection["ny"]) + 1,
        known_x=1.0, known_y=1.0)


def _sample_land(values, donor_land, x, y, radius: int):
    """Land-only bilinear, then nearest donor land within ``radius`` cells."""

    levels, ny, nx = values.shape
    inside = (x >= 0.0) & (x <= nx - 1) & (y >= 0.0) & (y <= ny - 1)
    i0 = np.floor(np.where(inside, x, 0.0)).astype(np.int64)
    j0 = np.floor(np.where(inside, y, 0.0)).astype(np.int64)
    fx = np.where(inside, x - i0, 0.0)
    fy = np.where(inside, y - j0, 0.0)
    total = np.zeros((levels,) + x.shape)
    weight = np.zeros(x.shape)
    for dj, di, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                      (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        jj = np.clip(j0 + dj, 0, ny - 1)
        ii = np.clip(i0 + di, 0, nx - 1)
        ok = inside & donor_land[jj, ii] & (w > 0.0)
        ww = np.where(ok, w, 0.0)
        total += values[:, jj, ii] * ww
        weight += ww
    found = weight > 0.0
    out = np.where(found, total / np.where(found, weight, 1.0), np.nan)
    need = inside & ~found
    if need.any() and radius > 0:
        ci = np.rint(x).astype(np.int64)
        cj = np.rint(y).astype(np.int64)
        offsets = sorted(((dj, di) for dj in range(-radius, radius + 1)
                          for di in range(-radius, radius + 1)),
                         key=lambda o: o[0] * o[0] + o[1] * o[1])
        for dj, di in offsets:
            if not need.any():
                break
            jj = np.clip(cj + dj, 0, ny - 1)
            ii = np.clip(ci + di, 0, nx - 1)
            hit = need & donor_land[jj, ii]
            if hit.any():
                out[:, hit] = values[:, jj[hit], ii[hit]]
                need &= ~hit
                found |= hit
    return out.astype(np.float32), found


def _donor_reason(donor, context, start) -> str | None:
    if start < datetime.fromisoformat(donor["valid_from"].replace("Z", "+00:00")):
        return f"start {start:%Y-%m-%d %H}Z is before the donor archive ({donor['valid_from']})"
    if start.minute or start.second or start.microsecond:
        return f"start {start:%H:%M:%S}Z is not on an analysis hour"
    box = donor.get("coverage_box")
    if box is not None:
        lat = np.asarray(context.latitude)
        lon = ((np.asarray(context.longitude) + 180.0) % 360.0) - 180.0
        inside = ((lat >= box["lat"][0]) & (lat <= box["lat"][1])
                  & (lon >= box["lon"][0]) & (lon <= box["lon"][1]))
        if not inside.any():
            return "the domain lies outside the donor's coverage"
    return None


def apply_land_state_donor(state, *, sf_surface_physics, context, receipt,
                           landmask=None, lake_mask=None):
    """Replace converted moisture with a same-model analysis where one covers.

    ``state`` is the router's soil state; its ``level_depths`` must be the
    donor's nodes.  Returns ``(state, receipt)``.
    """

    import dataclasses

    rules = load_rules()
    target = target_soil_model(sf_surface_physics)
    receipt = dict(receipt)
    donors = [row for row in rules["donors"] if row["soil_model"] == target]
    if not donors or _disabled() or receipt.get("source_soil_model") == target:
        return state, receipt
    if context is None:
        receipt["donor"] = "none: this route passed no domain context"
        return state, receipt
    start = _utc(context.valid_time)
    level_depths = getattr(state, "level_depths", None)
    land = _host(landmask if landmask is not None else state.landmask) >= 0.5
    if lake_mask is not None:
        land &= ~_host(lake_mask).astype(bool)
    reasons = []
    for donor in donors:
        reason = _donor_reason(donor, context, start)
        if reason is not None:
            reasons.append(f"{donor['id']}: {reason}")
            continue
        try:
            path = _fetch_donor_file(donor, start.replace(tzinfo=None), _cache_root(context))
            values, depths, projection = _decode_donor(donor, path)
            if level_depths is None or values.shape[0] != len(level_depths) or not np.allclose(
                    depths, np.asarray(level_depths, dtype=np.float64), atol=1e-4):
                raise ValueError(
                    f"donor nodes {depths.tolist()} m are not the target levels "
                    f"{None if level_depths is None else np.asarray(level_depths).tolist()}")
            grid = _donor_grid(projection, donor["grib2"]["earth_radius_m"])
            x, y = grid.latlon_to_ij(context.latitude, context.longitude)
            x = np.asarray(x) - 1.0
            y = np.asarray(y) - 1.0
            water = float(donor["water_value"])
            donor_land = (np.isfinite(values).all(axis=0) & (values > 0.0).all(axis=0)
                          & (values < water - 1e-6).all(axis=0))
            sampled, found = _sample_land(values, donor_land, x, y,
                                          int(donor["search_radius_cells"]))
        except Exception as error:  # a per-run fallback, loudly
            reasons.append(f"{donor['id']}: {type(error).__name__}: {error}")
            _note(f"soil moisture: same-model donor {donor['id']} unavailable for this run "
                  f"({type(error).__name__}: {error}); the index-remapped source moisture stays",
                  warning=True)
            continue
        covered = land & found
        if not covered.any():
            reasons.append(f"{donor['id']}: covers none of this domain's land")
            continue
        moisture = np.array(state.soil_moisture, copy=True)
        before_top = float(moisture[0][covered].mean())
        moisture[:, covered] = sampled[:, covered].astype(moisture.dtype)
        # RUC's SH2O is a copy of SMOIS that ruclsminit recomputes; only the
        # covered land cells change, so a sea-ice column keeps its SH2O = 0.
        liquid = np.array(state.liquid_moisture, copy=True)
        liquid[:, covered] = moisture[:, covered]
        receipt["donor"] = {
            "id": donor["id"],
            "cycle": f"{start:%Y-%m-%dT%H}Z",
            "grib": str(path),
            "land_cells_covered": int(np.count_nonzero(covered)),
            "land_cells_not_covered": int(np.count_nonzero(land & ~found)),
            "top_level_mean_before": round(before_top, 4),
            "top_level_mean_after": round(float(moisture[0][covered].mean()), 4),
            "column_mean_after": round(float(moisture[:, covered].mean()), 4),
        }
        _note(f"soil moisture: {donor['id']} {start:%Y-%m-%d %H}Z (the target land model's own "
              f"analysis) on {receipt['donor']['land_cells_covered']} of "
              f"{int(np.count_nonzero(land))} land cell(s); top level mean "
              f"{before_top:.3f} -> {receipt['donor']['top_level_mean_after']:.3f}")
        updates = {"soil_moisture": moisture, "liquid_moisture": liquid}
        return dataclasses.replace(state, **updates), receipt
    receipt["donor"] = "none: " + "; ".join(reasons)
    return state, receipt


__all__ = [
    "LandStateContext",
    "apply_land_state_donor",
    "convert_source_soil_moisture",
    "identify_source_soil",
    "land_state_context",
    "load_rules",
    "target_soil_model",
]
