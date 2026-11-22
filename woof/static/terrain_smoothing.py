"""Per-domain terrain smoothing: WPS GEOGRID.TBL ``smooth_option`` and
``smooth_passes`` for HGT_M, with WPS's own arithmetic.

WPS geogrid smooths HGT_M with whatever its GEOGRID.TBL names (the stock
table: one ``smth-desmth_special`` pass).  This module is the static
builders' owner of that choice, one setting per domain:

* ``smth-desmth_special`` x1 is the default and keeps the builders'
  historical float64 arithmetic and code path exactly
  (:func:`woof.static.build.smth_desmth_special`), so a configuration
  that says nothing, or says the default, builds the bytes it always did.
  ``smooth_precision = "wps-float32"`` is the option that runs this same
  setting through WPS's single-precision sweeps below instead: the float64
  smoother's largest distance from geogrid.exe's HGT_M was 1.2 to 1.5 mm
  on the Alpine domains measured, and the sweeps reproduce it exactly
  (``SMOOTH_PRECISIONS``).
* ``none`` keeps the sampled halo-extended field as it is.
* Every other setting runs WPS v4.6.0's single-precision sweeps
  (geogrid/src/smooth_module.F, called by process_tile_module.F:884-914 on
  the memory array widened by HALO_WIDTH = 3, which is woof's ``HALO``):

  - a smoothing sweep is ``S = 0.5*A + 0.25*(west + east)`` over every row
    and the interior columns, then ``A = 0.5*S + 0.25*(south + north)``
    over the interior rows and columns, so the one-cell outer ring of the
    extended array never changes;
  - ``smth-desmth`` follows each smoothing sweep with a desmoothing sweep
    of the same shape, ``1.52*A - 0.26*(pair)``;
  - ``smth-desmth_special`` then gives every point that went negative
    from a non-negative start its start value back (after all passes);
  - every operation is rounded to float32 in Fortran's left-to-right
    order, never fused, and subnormals are kept.

  The result is cast back to float64 and cropped like the default.
  The Python arithmetic here and the Rust entry point behind
  :func:`smooth_terrain` are both held bit for bit to WPS itself
  (tests/test_terrain_smoothing.py, tools/wps_smooth_v460_oracle/).

The setting travels on :class:`woof.static.highres_production.
HighresStaticConfig` (``terrain_smoothing`` rows for the non-default
domains only, so every default identity and receipt is unchanged), and
reaches each builder through :class:`woof.static.build.GeogSelection`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

#: The GEOGRID.TBL smoothers (source_data_module.F:444-458) and ``none``.
SMOOTH_OPTIONS = ("smth-desmth_special", "smth-desmth", "1-2-1", "none")

#: The arithmetic a smoother runs in (``smooth_precision``).  ``float64``
#: is the builders' historical smoother, which only the default
#: ``smth-desmth_special`` x1 has, and stays that setting's default;
#: ``wps-float32`` is WPS's own single-precision sweeps, the only
#: arithmetic every other smoother has.  ``none`` runs no arithmetic and
#: keeps the float64 samples.
SMOOTH_PRECISIONS = ("float64", "wps-float32")

#: Cells the static builders sample beyond each domain edge before
#: smoothing (woof.static.build.HALO, WPS's HALO_WIDTH).
STATIC_HALO = 3


@dataclass(frozen=True)
class TerrainSmoothing:
    """One domain's HGT_M smoother, its pass count and its arithmetic.

    ``precision`` left as ``None`` takes the setting's own arithmetic
    (:attr:`native_precision`), so ``TerrainSmoothing("1-2-1", 3)`` and
    ``TerrainSmoothing("1-2-1", 3, "wps-float32")`` are the same setting.
    Only ``smth-desmth_special`` x1 has a choice.
    """

    option: str = "smth-desmth_special"
    passes: int = 1
    precision: str | None = None

    def __post_init__(self):
        if self.option not in SMOOTH_OPTIONS:
            raise ValueError(
                f"unknown terrain smoother {self.option!r} (one of "
                f"{', '.join(SMOOTH_OPTIONS)}); refusing to build terrain "
                "with a smoother nobody asked for")
        if self.option == "none":
            object.__setattr__(self, "passes", 0)
        elif type(self.passes) is not int or self.passes < 1:
            raise ValueError(
                "terrain passes must be an integer >= 1; write "
                'smooth_option = "none" to disable smoothing')
        native = self.native_precision
        if self.precision is None:
            object.__setattr__(self, "precision", native)
        elif self.precision not in SMOOTH_PRECISIONS:
            raise ValueError(
                f"unknown terrain smoothing precision {self.precision!r} "
                f"(one of {', '.join(SMOOTH_PRECISIONS)}); refusing to "
                "build terrain in an arithmetic nobody asked for")
        elif self.precision != native and not self.precision_is_a_choice:
            if self.option == "none":
                raise ValueError(
                    f"smooth_precision = {self.precision!r} names a "
                    "smoother's arithmetic and smooth_option = \"none\" "
                    "runs no smoother; refusing a precision that would be "
                    "ignored")
            raise ValueError(
                f"smooth_precision = {self.precision!r}: only the default "
                "smth-desmth_special x1 has the float64 smoother, and "
                f"{self.option} x{self.passes} runs WPS's float32 "
                "arithmetic only; refusing a precision that would be "
                "ignored")

    @property
    def precision_is_a_choice(self) -> bool:
        """Whether this smoother has both arithmetics (special x1 only)."""
        return self.option == "smth-desmth_special" and self.passes == 1

    @property
    def native_precision(self) -> str:
        """The arithmetic this smoother runs when none is named."""
        if self.option == "none" or self.precision_is_a_choice:
            return "float64"
        return "wps-float32"

    @property
    def is_default(self) -> bool:
        return self.precision_is_a_choice and self.precision == "float64"

    @property
    def reach(self) -> int:
        """Cells the frozen outer ring's influence travels inward.

        Each sweep moves it one cell: one sweep per pass for 1-2-1, two
        (smooth, then desmooth) for the smoother-desmoothers.
        """
        if self.option == "none":
            return 0
        return self.passes * (1 if self.option == "1-2-1" else 2)

    @property
    def names_precision(self) -> bool:
        """Whether the precision differs from the smoother's own.

        Only then do the echo, the carrier row and the ``static`` line
        spell it, so every setting that existed before the option keeps
        its bytes in receipts, cache identities and emitted TOML.
        """
        return self.precision != self.native_precision

    def echo(self) -> dict[str, object]:
        echoed: dict[str, object] = {"smooth_option": self.option,
                                     "smooth_passes": self.passes}
        if self.names_precision:
            echoed["smooth_precision"] = self.precision
        return echoed

    def row(self, grid_id: int) -> tuple:
        """This setting's carrier row: ``(grid_id, option, passes)``, then
        the precision when :attr:`names_precision`."""
        row = (int(grid_id), self.option, self.passes)
        return row + (self.precision,) if self.names_precision else row

    @classmethod
    def from_row(cls, row) -> tuple[int, "TerrainSmoothing"]:
        """``(grid_id, setting)`` from a carrier row or its echoed list."""
        row = tuple(row)
        if len(row) not in (3, 4):
            raise ValueError(
                f"terrain smoothing row {list(row)!r} is not [grid_id, "
                "smooth_option, smooth_passes] with an optional "
                "smooth_precision; refusing a row whose setting would be "
                "guessed")
        grid_id, option, passes = row[:3]
        setting = cls(option, passes, row[3] if len(row) == 4 else None)
        if len(row) == 4 and not setting.names_precision:
            raise ValueError(
                f"terrain smoothing row {list(row)!r} spells the precision "
                f"its smoother always runs; the canonical row is "
                f"{list(setting.row(grid_id))!r}")
        return int(grid_id), setting

    def label(self) -> str:
        """``none`` or ``OPTION xPASSES`` (plus a named precision), for
        messages and audits."""
        if self.option == "none":
            return "none"
        text = f"{self.option} x{self.passes}"
        return f"{text} {self.precision}" if self.names_precision else text


WPS_DEFAULT = TerrainSmoothing()

#: WPS's stock setting in WPS's own arithmetic: geogrid.exe's HGT_M.
WPS_EXACT_DEFAULT = TerrainSmoothing(precision="wps-float32")


def with_precision(setting: TerrainSmoothing, precision) -> TerrainSmoothing:
    """``setting`` in ``precision`` where the smoother has that choice.

    The door-wide flags apply a precision to every domain: a domain whose
    smoother has one arithmetic (every smoother but the default, and
    ``none``) keeps its setting, which already is WPS's arithmetic.
    ``None`` leaves ``setting`` as it is.
    """
    if precision is None:
        return setting
    if precision not in SMOOTH_PRECISIONS:
        raise ValueError(
            f"unknown terrain smoothing precision {precision!r} (one of "
            f"{', '.join(SMOOTH_PRECISIONS)})")
    if not setting.precision_is_a_choice:
        return setting
    return replace(setting, precision=precision)


# ---------------------------------------------------------------------------
# Configuration: [[domain]] static tables and the woof domain flag.
# ---------------------------------------------------------------------------

def parse_domain_static(table, *, source, grid_id) -> TerrainSmoothing:
    """Validate one ``[[domain]] static = {...}`` table."""
    from woof.experiment import did_you_mean

    where = f"[[domain]] grid_id = {grid_id} static of {source}"
    if not isinstance(table, Mapping):
        raise ValueError(
            f"{where} must be a table; refusing to discard its terrain "
            "smoothing")
    known = ("smooth_option", "smooth_passes", "smooth_precision")
    unknown = sorted(set(table) - set(known))
    if unknown:
        names = ", ".join(f"{k!r}{did_you_mean(k, known)}" for k in unknown)
        raise ValueError(
            f"{where}: unknown keys {names}; refusing to ignore them and "
            "build default terrain")
    # smooth_precision alone names the default smoother's arithmetic; a
    # pass count without its smoother stays ambiguous.
    if "smooth_option" not in table and (
            "smooth_passes" in table or "smooth_precision" not in table):
        raise ValueError(
            f"{where} requires smooth_option; a pass count alone leaves the "
            "smoother ambiguous")
    option = table.get("smooth_option", WPS_DEFAULT.option)
    if option == "none" and "smooth_passes" in table:
        raise ValueError(
            f"{where}: a pass count for no smoother would be ignored")
    if option == "none" and "smooth_precision" in table:
        raise ValueError(
            f"{where}: a precision for no smoother would be ignored")
    try:
        return TerrainSmoothing(option, table.get("smooth_passes", 1),
                                table.get("smooth_precision"))
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def parse_smoothing_spec(text) -> tuple[TerrainSmoothing, ...]:
    """``woof domain --terrain-smoothing``: ``none,1-2-1:3,...``.

    One item per domain in domain order; the last one repeats for the
    remaining domains (WRF's namelist convention).  Each item is
    ``none``, ``OPTION`` or ``OPTION:PASSES``.
    """
    result = []
    for item in str(text).split(","):
        parts = item.strip().split(":")
        if (len(parts) > 2 or not parts[0]
                or (parts[0] == "none" and len(parts) > 1)):
            raise ValueError(
                f"invalid terrain smoothing item {item!r} (none, OPTION or "
                "OPTION:PASSES); refusing a setting that would be ignored")
        passes = 1
        if len(parts) == 2:
            try:
                passes = int(parts[1])
            except ValueError:
                raise ValueError(
                    f"invalid terrain pass count {parts[1]!r}") from None
        result.append(TerrainSmoothing(parts[0], passes))
    return tuple(result)


def static_inline(setting: TerrainSmoothing) -> str:
    """The ``static = {...}`` line a domain carries for ``setting``."""
    text = f'static = {{ smooth_option = "{setting.option}"'
    if setting.option != "none":
        text += f", smooth_passes = {setting.passes}"
    if setting.names_precision:
        text += f', smooth_precision = "{setting.precision}"'
    return text + " }"


def emit_smoothing(text: str, settings) -> str:
    """Add a ``static`` line under each ``[[domain]]`` header of ``text``.

    Only table-header lines count (a comment mentioning ``[[domain]]`` is
    not a domain); default settings add nothing, so an all-default list
    returns ``text`` unchanged.
    """
    headings = tuple(re.finditer(r"(?m)^\[\[domain\]\][ \t]*$", text))
    if len(settings) > len(headings):
        raise ValueError(
            f"the terrain smoothing list ({len(settings)} settings) is "
            f"longer than the domain count ({len(headings)}); refusing "
            "settings that belong to no domain")
    if not settings:
        return text
    for index, heading in reversed(tuple(enumerate(headings))):
        setting = settings[min(index, len(settings) - 1)]
        if not setting.is_default:
            offset = heading.end()
            text = text[:offset] + "\n" + static_inline(setting) + text[offset:]
    return text


def refuse_moving_reach(domain_tables, experiment, *, source) -> None:
    """Refuse a smoother that reaches past the halo on a moving domain.

    WPS leaves the extended array's outer ring untouched, so once a
    setting's :attr:`TerrainSmoothing.reach` passes the ``STATIC_HALO``
    cells sampled beyond the domain, the smoothed edge rows depend on
    where the footprint sits (1-2-1 from 4 passes, the smoother-
    desmoothers from 2; measured, tests/test_terrain_smoothing.py).  A
    moved domain rebuilds its statics per footprint, and
    :func:`woof.ingest.relocation_init.overlap_statics_mismatches`
    refuses the first move whose shared ground differs.
    """
    from woof.static.corridor import relocating_subtree_grid_ids

    moving = set(relocating_subtree_grid_ids(experiment))
    if not moving:
        return
    for table in domain_tables or ():
        if "static" not in table:
            continue
        grid_id = int(table["grid_id"])
        setting = parse_domain_static(table["static"], source=source,
                                      grid_id=grid_id)
        if grid_id in moving and setting.reach > STATIC_HALO:
            raise ValueError(
                f"[[domain]] grid_id = {grid_id} of {source} moves, and "
                f"terrain smoothing {setting.label()} reaches {setting.reach} cells, past the {STATIC_HALO}-cell "
                "halo the static build smooths over: the edge rows then "
                "depend on where the footprint sits, the rebuilt terrain "
                "differs from the outgoing footprint's on shared ground, and "
                "the first relocation is refused. On a moving domain use "
                "none, 1-2-1 with at most 3 passes, or smth-desmth(_special) "
                "with 1 pass.")


def resolve_domain_smoothing(raw, config, *, source):
    """``config`` carrying the non-default ``[[domain]] static`` rows.

    All-default returns ``config`` itself (``None`` stays ``None``); a
    configuration with no ``[static.highres]`` gets a disabled carrier,
    which applies no overlay.
    """
    rows = []
    for index, domain in enumerate(raw.get("domain", ()) or ()):
        if "static" in domain:
            grid_id = int(domain.get("grid_id", index + 1))
            setting = parse_domain_static(domain["static"], source=source,
                                          grid_id=grid_id)
            if not setting.is_default:
                rows.append(setting.row(grid_id))
    if not rows:
        return config
    from .highres_production import (HighresStaticConfig,
                                     default_highres_cache_root)
    if config is None:
        config = HighresStaticConfig(
            enabled=False, cache_root=default_highres_cache_root())
    return replace(config, terrain_smoothing=tuple(sorted(rows)))


def smoothing_rows_from_echo(rows, *, source) -> tuple[tuple, ...]:
    """The carrier rows a sealed echo recorded, exactly as the carrier held them.

    A sealed preparation records its carrier through
    :meth:`woof.static.highres_production.HighresStaticConfig.echo`:
    ``[[grid_id, option, passes], ...]``, a fourth ``smooth_precision``
    column on a row whose precision is not its smoother's own
    (:meth:`TerrainSmoothing.row`), sorted, one row per non-default domain.
    Anything else was not written by that echo, and rebuilding from it
    would build terrain under a setting the seal does not hold (a reader
    that kept three columns would rebuild ``smooth_precision =
    "wps-float32"`` in float64), so it is refused.
    """
    where = f"terrain_smoothing of {source}"
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(
            f"{where} must be a non-empty list of [grid_id, smooth_option, "
            f"smooth_passes] rows, got {rows!r}; refusing to rebuild terrain "
            "under a smoothing the seal does not record")
    parsed = []
    for row in rows:
        if (not isinstance(row, (list, tuple)) or len(row) not in (3, 4)
                or type(row[0]) is not int or row[0] < 1):
            raise ValueError(
                f"{where} holds {row!r}, not a [grid_id, smooth_option, "
                "smooth_passes] row with an optional smooth_precision; "
                "refusing to rebuild terrain under a smoothing the seal does "
                "not record")
        try:
            setting = TerrainSmoothing(*row[1:])
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        if setting.is_default or setting.row(row[0]) != tuple(row):
            raise ValueError(
                f"{where} holds {row!r}, which the carrier never records "
                "(it keeps one row per non-default domain, none with a pass "
                "count of 0, and a precision only where it is not the "
                "smoother's own); refusing to rebuild terrain under a "
                "smoothing the seal does not record")
        parsed.append(setting.row(row[0]))
    ids = [row[0] for row in parsed]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError(
            f"{where} names a domain twice or out of order ({rows!r}); "
            "refusing to pick one of two settings for a domain's terrain")
    return tuple(parsed)


def smoothing_for(config, domain_id) -> TerrainSmoothing:
    """The setting ``config`` (a carrier or ``None``) holds for a domain."""
    for row in getattr(config, "terrain_smoothing", ()):
        grid_id, setting = TerrainSmoothing.from_row(row)
        if grid_id == domain_id:
            return setting
    return WPS_DEFAULT


def selection_carrier(config):
    """``config`` when a GEOG selection reads it, else ``None``.

    :meth:`woof.static.build.GeogSelection.from_case_data` reads two
    things from a carrier: a domain's non-default terrain smoothing, and
    the land cover an enabled ``[static.highres]`` block builds, which
    admits that collection's ``geog_data_res`` token (``cglc_modis_lcz``)
    where the block builds it.  A catalog that resolves selections must
    carry such a carrier, or that token is refused on a run that declared
    the block.  Any other carrier (disabled, terrain only, the engine's
    default terrain row) selects exactly what no carrier selects, so it
    stays off the catalog and those routes keep the calls they made.
    """
    if config is None:
        return None
    if getattr(config, "terrain_smoothing", ()):
        return config
    if getattr(config, "static_source", None) is not None:
        # A configuration's static source (woof.static.external_source)
        # is read by every selection the carrier reaches.
        return config
    if (getattr(config, "enabled", False)
            and getattr(config, "fields", "auto") != "terrain"):
        return config
    return None


def selection_carrier_kwargs(config) -> dict[str, object]:
    """``{"static_highres": config}`` for a catalog call when
    :func:`selection_carrier` keeps it, else ``{}``."""
    carrier = selection_carrier(config)
    return {} if carrier is None else {"static_highres": carrier}


def catalog_with_smoothing(inner, config):
    """``inner`` (a static catalog) viewed with ``config``'s settings.

    Returns ``inner`` itself unless ``config`` is a
    :func:`selection_carrier` (a non-default smoothing row, or a land
    cover its block builds) and ``inner`` carries none, or ``config``
    carries smoothing rows and ``inner`` carries none, so every route
    whose selection reads nothing from ``config`` is unchanged.
    """
    held = None if inner is None else getattr(inner, "static_highres", None)
    if inner is not None and selection_carrier(config) is not None and (
            selection_carrier(held) is None
            or (getattr(config, "terrain_smoothing", ())
                and not getattr(held, "terrain_smoothing", ()))
            or (getattr(config, "static_source", None) is not None
                and getattr(held, "static_source", None) is None)):
        return SimpleNamespace(**{**vars(inner), "static_highres": config})
    return inner


def smoothing_receipt(config) -> dict[str, dict[str, object]]:
    """The receipt rows attesting ``config``'s non-default settings."""
    settings = (TerrainSmoothing.from_row(row)
                for row in getattr(config, "terrain_smoothing", ()))
    return {f"d{gid:02d}": setting.echo() for gid, setting in settings}


def require_root_smoothing(config, domain_id, receipt) -> None:
    """The terrain-smoothing root seam.

    A route that built or loaded a root's statics without carrying its
    setting has default terrain; integrating it would silently drop the
    configuration.  Accepted only when the receipt (or its nested
    ``baseline``) attests the setting.  The ERA5, GFS, mapped-source and
    native HRRR roots built from WPS_GEOG carry and attest it; what this
    still stops is a prebuilt static cache that does not record its
    smoothing.
    """
    setting = smoothing_for(config, domain_id)
    if setting.is_default:
        return
    while isinstance(receipt, Mapping):
        attested = receipt.get("terrain_smoothing")
        if (isinstance(attested, Mapping)
                and attested.get(f"d{domain_id:02d}") == setting.echo()):
            return
        receipt = receipt.get("baseline")
    raise ValueError(
        f"terrain-smoothing root seam: d{domain_id:02d} asks for terrain "
        f"smoothing {setting.label()}, but this route built its "
        "terrain with the default smth-desmth_special x1 or loaded it from a "
        "prebuilt cache that does not record its smoothing; the forecast "
        "would integrate terrain the configuration did not ask for. Build "
        "the root from WPS_GEOG on the ERA5, GFS, native HRRR or a mapped "
        "source's route (not from a prebuilt static cache), put the setting "
        "on a nest, or drop the setting")


# ---------------------------------------------------------------------------
# GEOGRID.TBL import.
# ---------------------------------------------------------------------------

#: The first six branches of the GEOGRID.TBL keyword chain, in order
#: (source_data_module.F:398-465).  A key is classified by the FIRST
#: keyword that contains it (``index(keyword, key) /= 0``).
_TABLE_KEYWORDS = ("name", "priority", "dest_type", "interp_option",
                   "smooth_option", "smooth_passes")


def _despace(line: str) -> str:
    """WPS ``despace`` (module_stringutil.F:51-80): drop spaces and tabs
    outside quotes."""
    out, quoted = [], False
    for char in line:
        if char in "\"'":
            quoted = not quoted
        if quoted or char not in " \t":
            out.append(char)
    return "".join(out)


def geogrid_tbl_smoothing(path) -> TerrainSmoothing:
    """HGT_M's setting in a GEOGRID.TBL (the file or its directory).

    WPS semantics: ``#`` starts a comment, ``;`` separates specifications,
    ``=====`` starts a new entry, option values must match exactly
    (source_data_module.F:444-458), and ``get_smooth_option``
    (:2711-2744) takes the FIRST entry named exactly ``HGT_M`` that
    carries a recognised option, with 1 pass when none is given.  No such
    entry, or ``smooth_passes <= 0`` (``do ipass=1,npass``), is no
    smoothing.

    One key is WOOF-ArWen's, not WPS's: ``smooth_precision = wps-float32`` in
    that entry selects WPS's own arithmetic for the default smoother
    (:data:`SMOOTH_PRECISIONS`).  It is contained in none of WPS's
    keywords, so geogrid.exe logs "unrecognized option" for it and smooths
    in float32 as it always does; the table stays usable by both.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "GEOGRID.TBL"
    entries, entry = [], {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = _despace(line)
        if line.startswith("#"):
            continue
        if "=====" in line:
            if entry:
                entries.append(entry)
                entry = {}
            continue
        for spec in line.split("#", 1)[0].split(";"):
            if "=" not in spec:
                continue
            key, value = spec.split("=", 1)
            branch = next((k for k in _TABLE_KEYWORDS if key in k), None)
            if branch == "name":
                entry["name"] = value
            elif branch == "smooth_option" and value in SMOOTH_OPTIONS[:3]:
                entry["option"] = value
            elif branch == "smooth_passes":
                entry["passes"] = int(value[:10])  # read(..., '(i10)')
            elif branch is None and key == "smooth_precision":
                entry["precision"] = value
    entries.append(entry)
    for entry in entries:
        if entry.get("name") == "HGT_M" and "option" in entry:
            passes = entry.get("passes", 1)
            setting = (TerrainSmoothing("none") if passes <= 0 else
                       TerrainSmoothing(entry["option"], passes))
            if "precision" not in entry:
                return setting
            if setting.option == "none":
                raise ValueError(
                    f"{path}: HGT_M's smooth_precision = "
                    f"{entry['precision']} names a smoother's arithmetic "
                    f"and smooth_passes = {passes} runs no smoother; "
                    "refusing a precision that would be ignored")
            try:
                return replace(setting, precision=entry["precision"])
            except ValueError as error:
                raise ValueError(f"{path}: HGT_M entry: {error}") from None
    return TerrainSmoothing("none")


# ---------------------------------------------------------------------------
# Arithmetic.
# ---------------------------------------------------------------------------

def _wps_smooth_f32(extended, smoothing: TerrainSmoothing) -> np.ndarray:
    """WPS v4.6.0 smooth_module.F in float32 (see the module docstring)."""
    original = np.array(extended, dtype=np.float32, copy=True)
    a = original.copy()
    sweeps = [(np.float32(0.5), np.float32(0.25), False)]
    if smoothing.option != "1-2-1":
        sweeps.append((np.float32(1.52), np.float32(0.26), True))
    for _ in range(smoothing.passes):
        for c1, c2, desmooth in sweeps:
            s = a.copy()
            t = c1 * a[:, 1:-1]
            v = c2 * (a[:, :-2] + a[:, 2:])
            s[:, 1:-1] = t - v if desmooth else t + v
            t = c1 * s[1:-1, 1:-1]
            v = c2 * (s[:-2, 1:-1] + s[2:, 1:-1])
            a[1:-1, 1:-1] = t - v if desmooth else t + v
    if smoothing.option == "smth-desmth_special":
        restore = (a < 0) & (original >= 0)
        a[restore] = original[restore]
    return a.astype(np.float64)


def smooth_terrain_reference(extended, smoothing=WPS_DEFAULT) -> np.ndarray:
    """The Python reference: halo-extended float64 in, float64 out."""
    if smoothing.is_default:
        from .build import smth_desmth_special
        return smth_desmth_special(extended, passes=1)
    if smoothing.option == "none":
        return np.array(extended, dtype=np.float64, copy=True)
    return _wps_smooth_f32(extended, smoothing)


def smooth_terrain(extended, smoothing=WPS_DEFAULT) -> np.ndarray:
    """The production entry for callers outside the Rust field build.

    The default takes the legacy call unchanged; any other setting runs
    the Rust entry point (``WOOF_STATIC_PYTHON=1`` routes it to the
    Python reference, with the route's WORKAROUND line).
    """
    if smoothing.is_default:
        return smooth_terrain_reference(extended, smoothing)
    from . import rust_bridge
    bridge = rust_bridge.route("terrain_smooth")
    if bridge is not None:
        return bridge.terrain_smooth(extended, smoothing)
    return smooth_terrain_reference(extended, smoothing)
