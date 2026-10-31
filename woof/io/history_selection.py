"""Which variables the history tape carries: the ``[output]`` surface.

A woof run writes its whole history inventory every period, and on a
four-domain tree at forecast cadence that is the thing that fills a
disk.  WRF's own answer is ``iofields_filename``; this is woof's, and
it is deliberately a SELECTION over the inventory the run already
produces rather than a second inventory of its own:

* the vocabulary is the UNION of the wrfout schema tables, both of them
  in :mod:`woof.io.wrf_output_schema` (``HISTORY_FIELDS_BY_NETCDF_NAME``
  and ``REGISTRY_VAR_META``, which the writer binds as ``_VAR_META``),
  so a scheme that adds a history field joins the vocabulary with its
  schema row and no code here changes -- the arbitrary-model test
  applied to variables;
* the presets are TABLES of names, so a new preset, or a new member of
  one, is a row;
* the filter is one predicate applied to the frame the writer already
  assembled, so there is no per-variable code path to grow.

THE DEFAULT IS THE FULL SET.  ``[output]`` absent, or ``preset =
"full"``, writes exactly what every run before this feature wrote --
same inventory, same order, same header -- and stamps no attribute of
its own.  Trimming is a thing a user asks for, never something a
default does to them.

WHAT IS NOT DROPPABLE, and why.  :data:`STRUCTURAL_FIELDS` is the
georeference, the vertical coordinate and the time coordinate: without
them no reader can place a value in space or time, so a file missing
one is not a smaller wrfout, it is an unreadable one.  Naming one in
``history_drop`` is refused with that sentence rather than served.

THE CHECKPOINT STREAM IS UNAFFECTED.  Checkpoints are separate files
written by :mod:`woof.io.restart` from model state, not from the
history frame, so a run that trims its history restarts exactly as it
would have.  ``tests/test_restart.py`` proves it, and the selection is
deliberately kept out of the restart identity for the same reason
``[tiles]`` is: it changes no number the model computes.
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib

from woof.explain import warn
from woof.io.wrf_output_schema import (HISTORY_FIELDS_BY_NETCDF_NAME,
                                        REGISTRY_VAR_META)


#: The names the writer emits for every frame regardless of physics, plus
#: the time coordinate it derives.  Kept here rather than read off
#: ``state_frame`` because that function needs a live GPU state to answer
#: and this table has to be readable at config-resolution time.
_CORE_DYNAMICS_FIELDS = (
    "Times", "XTIME", "ITIMESTEP",
    "T", "U", "V", "W", "PH", "PHB", "MU", "MUB", "HGT",
    "P", "PB", "PSFC", "P_TOP", "ZNU", "ZNW",
    "QVAPOR", "QCLOUD", "QRAIN",
    "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA",
    "LANDMASK", "LU_INDEX",
)


def _writer_metadata_names() -> frozenset[str]:
    """Every name the writer carries Registry metadata for.

    Read from :mod:`woof.io.wrf_output_schema`, which OWNS the table,
    and never from :mod:`woof.io.wrfout`, which merely binds it.  This
    function is called at module scope to build
    :data:`HISTORY_VOCABULARY`, so importing the writer here imports the
    writer whenever anything reads a config -- and the writer imports
    :mod:`woof.supervisor`, ``netCDF4`` and the runtime behind them.
    MEASURED: ``import woof.era5_direct`` pulled the whole forecast
    output side in through case_data -> experiment -> here, and the
    RW-WPS preprocessing wheel's no-CuPy/no-executor import proof went
    red on it.  ``tests/test_history_selection.py`` pins the boundary.
    """

    return frozenset(REGISTRY_VAR_META)


#: Every variable name a woof wrfout can carry: the union of the two
#: schema tables plus the core dynamics/time names.  A ``[output]`` list
#: may only name members of this set, and an unknown name is refused with
#: the whole set printed -- there is no other place a user can read it.
HISTORY_VOCABULARY: frozenset[str] = frozenset(
    set(_CORE_DYNAMICS_FIELDS)
    | set(HISTORY_FIELDS_BY_NETCDF_NAME)
    | _writer_metadata_names()
)

#: The fields no selection may drop, and the sentence that says why.
#:
#: These are not "important" fields -- ``REFL_10CM`` is important and is
#: droppable.  These are the ones whose absence stops a reader from
#: placing ANY value: the horizontal georeference (``XLAT``/``XLONG``),
#: the terrain and mass/vertical coordinate the eta levels are defined
#: against (``HGT``, ``ZNU``, ``ZNW``, ``P_TOP``, ``MUB``, ``PHB``), and
#: the time coordinate WRF tooling reads (``Times``, ``XTIME``,
#: ``ITIMESTEP``).  A wrfout missing one of these is not a smaller
#: wrfout; it is a file no tool in the estate can open as one.
#:
#: ``T`` IS IN THIS SET, AND IT IS HERE BECAUSE IT WAS MEASURED.  A
#: two-domain proof run whose 1 km child took ``preset = "minimal"``
#: wrote a d02 tape that ``rw_wrfbatch`` refused outright -- "Open WRF
#: ... failed and the file is not a supported post-processed WRF
#: archive" -- not a product skipped, the whole FILE rejected.  The
#: ladder was then measured against the real binary on a real frame,
#: one field at a time (``rw_wrfbatch --list-products``):
#:
#:   minimal (57 vars, no 3-D)      rc 1, the file does not open
#:   minimal + T (58)               rc 0,  30 products renderable
#:   minimal + T,P,PB (60)          rc 0,  38
#:   minimal + T,P,PB,PH (61)       rc 0,  42
#:   minimal + ...,QVAPOR (62)      rc 0,  66
#:   minimal + ...,U,V,W (65)       rc 0, 160
#:   the full inventory (74)        rc 0, 162
#:
#: So ``T`` is the field that makes the difference between a readable
#: wrfout and a file the estate's own reader will not open, which is
#: exactly what this set is for.  It costs one 3-D volume, and a preset
#: that shed it would be selling a file nothing can read.
STRUCTURAL_FIELDS: frozenset[str] = frozenset({
    "Times", "XTIME", "ITIMESTEP",
    "XLAT", "XLONG",
    "HGT", "ZNU", "ZNW", "P_TOP", "MUB", "PHB",
    "T",
})

#: Grid-relative wind components, and the rotation that makes them
#: earth-relative.  On a projected grid U10/V10 (and U/V) point along the
#: grid's own axes; without SINALPHA/COSALPHA beside them no reader can
#: say which way the wind blows.  The frozen station scoreboard refuses
#: such a wind rather than guess it is east-north (rw-scoreboard
#: ``extract.rs``: "a projected wind with no orientation is absent"), so
#: a run whose ``history_vars`` named U10 and V10 but not the rotation got
#: no 10 m wind score against a single station (the 2026-10-07
#: boundary-head proof: wind10_rmse empty for both WOOF arms, scored for
#: HRRR).  A kept wind therefore keeps its rotation, the way a kept field
#: keeps its georeference.
PROJECTED_WIND_FIELDS: frozenset[str] = frozenset({"U10", "V10", "U", "V"})
WIND_ROTATION_FIELDS: frozenset[str] = frozenset({"SINALPHA", "COSALPHA"})

#: The ``minimal`` preset's science: the 2-D surface state and the
#: accumulators every surface product is drawn from.  One row per field;
#: adding a surface product's input to the preset is adding a row.
#:
#: It carries no 3-D volume of its own.  The one it does write --
#: ``T``, plus the ``PHB`` column -- comes from
#: :data:`STRUCTURAL_FIELDS`, which is where the measured reason lives:
#: without ``T`` the file is not a wrfout any reader in the estate will
#: open.
_MINIMAL_FIELDS: frozenset[str] = frozenset({
    # Surface state and near-surface diagnostics.
    "T2", "Q2", "TH2", "PSFC", "U10", "V10", "TSK", "MU",
    # Surface energy budget and boundary-layer depth.
    "HFX", "LH", "QFX", "GRDFLX", "SWDOWN", "GLW", "OLR", "PBLH",
    "UST", "PSIM", "PSIH",
    # Precipitation and snow accumulators.
    "RAINC", "RAINNC", "RAINSH", "SNOWNC", "GRAUPELNC", "HAILNC",
    "SNOW", "SNOWC", "SNOWH",
    # Column extremes a severe-weather surface panel reads.
    "UP_HELI_MAX",
    # Grid metadata: 2-D, cheap, and what a staggered reader needs.
    "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA",
    # Land identity: what the ground IS, which every child grid needs.
    "LANDMASK", "LU_INDEX", "ISLTYP", "IVGTYP", "TMN", "VEGFRA", "SEAICE",
})

#: What ``severe`` adds to ``minimal``: the storm-scale volumes.  It is
#: the full inventory MINUS the restart-only scheme carriers no render
#: product reads -- stated as an explicit add-list rather than a
#: subtract-list so a new scheme's carriers do not silently join it.
_SEVERE_EXTRA_FIELDS: frozenset[str] = frozenset({
    # Dynamics volumes (``T`` is structural and rides along regardless).
    "U", "V", "W", "PH", "P", "PB",
    # Hydrometeors: mixing ratios and the number concentrations a
    # double-moment scheme carries.
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL",
    "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL", "QNHAIL",
    # Simulated reflectivity, the volume every radar product is built on.
    "REFL_10CM",
    # The radiation cloud fraction the GRIB2 post's cloud cover, base,
    # top and ceiling are drawn from.
    "CLDFRA",
})

#: Named presets: name -> the set of droppable fields it keeps, or
#: ``None`` for "everything this run produces".  :data:`STRUCTURAL_FIELDS`
#: is added to every non-``None`` set at selection time.
HISTORY_PRESETS: dict[str, frozenset[str] | None] = {
    "full": None,
    "minimal": _MINIMAL_FIELDS,
    "severe": _MINIMAL_FIELDS | _SEVERE_EXTRA_FIELDS,
}

#: What each named preset costs in RENDER PRODUCTS, not in variables.
#:
#: The ladder at the head of this module was measured one field at a
#: time against the real binary (``rw_wrfbatch --list-products``) on one
#: 1 km frame of the proof run; these are its rungs read as products
#: rather than as prose.  ``severe`` keeps every volume a render product
#: reads and sheds only the restart-only scheme carriers none of them
#: touches, so it prices at the full inventory's count.
#:
#: The figure is computed from THIS table inside
#: :meth:`HistorySelection.warn_lost_products`, never from what a caller
#: passes as ``produced``, so the two plan-review doors print the same
#: number for one configuration no matter what either of them hands in.
PRESET_PRODUCT_COUNTS: dict[str, int] = {
    "full": 162,
    "severe": 162,
    "minimal": 30,
}

#: Which fields the loss is attributable to, largest first, read off the
#: same ladder by subtraction: 160 - 66 for ``U,V,W``, 66 - 42 for
#: ``QVAPOR``, 38 - 30 for ``P,PB``, 42 - 38 for ``PH``.  The remainder
#: is whatever the rest of the full inventory serves, and
#: :meth:`HistorySelection.warn_lost_products` names it as such rather
#: than leaving a reader to find that the figures do not add up.
PRODUCT_LOSS_ATTRIBUTION: tuple[tuple[tuple[str, ...], int], ...] = (
    (("U", "V", "W"), 94),
    (("QVAPOR",), 24),
    (("P", "PB"), 8),
    (("PH",), 4),
)

#: How the counts above were obtained, in one clause a warning can carry.
PRODUCT_COUNT_BASIS = (
    "measured with rw_wrfbatch --list-products on one 1 km frame of the "
    "proof run (49 levels, WSM6, RRTMG), one field at a time")

#: The keys ``[output]`` accepts.
OUTPUT_KEYS: frozenset[str] = frozenset(
    {"formats", "icechunk_store", "preset", "history_vars", "history_drop",
     "full_history_interval_s", "between_preset", "between_history_add"})

#: How many dropped names the plan-time warning spells out before it
#: switches to a count.  See :meth:`HistorySelection.warn_lost_products`.
_NAMED_DROP_LIMIT = 12


#: woof's render product catalog, as the wrfout variables each product
#: is drawn from.  ``A|B`` means either input satisfies the requirement.
#:
#: ONE table, read by three callers: the matplotlib engine's
#: availability listing and render pre-check
#: (:func:`woof.render.missing_declared_inputs`), the plan-time
#: dependency warning below, and the render front door's named refusal.
#: Two tables describing one dependency is how a listing and a render
#: come to disagree about whether a product can be drawn.
#:
#: The rust-catalog rows are the slugs
#: ``woof.render.RUST_PRODUCT_ALIASES`` maps the shared names onto, plus
#: the generic ``var:`` token for a stored variable the engine draws
#: directly; their inputs are the wrfout variables whose absence the
#: renderer itself reports as missing-fields.  Slugs beyond these are
#: still not claimed: the renderer owns its whole catalog, it answers
#: for their inputs against a real file through ``--list-products``, and
#: inventing rows here for recipes this module has not verified would be
#: the correct-looking-wrong-mapping defect that ``wind10`` already cost
#: once.  What a caller with a WIDER verified mapping does is pass it to
#: :func:`lost_products`, which is the seam that exists for exactly that
#: and keeps these rows as the answer when nobody has one.
PRODUCT_HISTORY_INPUTS: dict[str, tuple[str, ...]] = {
    # The shared product names (the matplotlib engine's whole catalog).
    "refl": ("REFL_10CM",),
    "t2": ("T2",),
    "wind10": ("U10", "V10"),
    "precip": ("RAINC|RAINNC",),
    # OLR is present exactly when the attached longwave scheme produced
    # it, so a radiation-off run lists this product as missing-fields
    # rather than failing -- which is the accurate report of that run.
    "olr": ("OLR",),
    # The GENERIC token for the same field.  The rust engine imports
    # WRF's OLR by default and draws any stored 2-D variable through
    # ``--products var:<name>``, so ``var:wrf_olr`` is how OLR is asked
    # for on that lane -- and without this row a run that shed OLR met a
    # bare engine skip after its run directory had been claimed instead
    # of the named [output] refusal at the door.
    "var:wrf_olr": ("OLR",),
    # The rust catalog slugs the shared names resolve to.
    "composite_reflectivity": ("REFL_10CM",),
    "2m_temperature": ("T2",),
    "10m_wind_speed_and_direction": ("U10", "V10"),
    "total_qpf": ("RAINC|RAINNC",),
}


def _requirement_lost(requirement: str, dropped: frozenset[str]) -> bool:
    """True when every option of one ``A|B`` requirement was dropped."""
    return all(option in dropped for option in requirement.split("|"))


def lost_products(dropped, requirements=None) -> dict[str, tuple[str, ...]]:
    """Which catalog products these dropped variables kill, and by what.

    The value is the requirement(s) that can no longer be met, spelled
    the way the table spells them (``RAINC or RAINNC`` for an either-or
    row), so the reader gets the mapping and not merely a count.

    ``requirements`` replaces :data:`PRODUCT_HISTORY_INPUTS` for a
    caller that HAS a wider verified mapping, in the same
    ``slug -> ("A|B", ...)`` grammar over WRFOUT VARIABLE NAMES.  It is
    deliberately not the renderer's own requirement rows: those are
    spelled in the store's selector vocabulary, and a selector is not a
    netCDF variable name -- the hop between them runs through wrf-core
    diagnostics that declare no input list at all, so joining them here
    would invent exactly the kind of correct-looking wrong mapping this
    table's own note refuses.  Absent, the verified rows answer, which
    is what every caller in the tree does today.
    """
    dropped = frozenset(dropped)
    table = PRODUCT_HISTORY_INPUTS if requirements is None else dict(requirements)
    lost: dict[str, tuple[str, ...]] = {}
    for product, requirements in table.items():
        missing = tuple(" or ".join(requirement.split("|"))
                        for requirement in requirements
                        if _requirement_lost(requirement, dropped))
        if missing:
            lost[product] = missing
    return lost


def _did_you_mean(name: str, known) -> str:
    close = difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.6)
    return "" if not close else f"  Did you mean {', '.join(close)}?"


@dataclass(frozen=True)
class HistorySelection:
    """One resolved ``[output]`` block: which history variables to write.

    ``preset`` names the base set (``full`` -- the DEFAULT -- keeps
    everything the run produces).  ``history_vars`` replaces the base
    with an explicit include list; ``history_drop`` subtracts from it.
    The two lists are mutually exclusive, and so are ``preset`` and
    ``history_vars``: they are two spellings of the same include list
    and there is no rule for which wins.
    """

    preset: str = "full"
    history_vars: tuple[str, ...] = ()
    history_drop: tuple[str, ...] = ()
    #: Sub-hourly output the way HRRR writes it: a full history frame only
    #: at leads that are whole multiples of ``full_history_interval_s``,
    #: and between them the 2-D subset ``between_preset`` plus
    #: ``between_history_add`` (HRRR's 15-minute wrfsubh files are 2-D
    #: fields; its 3-D history is hourly).  0 (the default) writes every
    #: frame with this selection, as before.
    #:
    #: Breakage it prevents (V61 8x RTX 5090, 2026-10-21): the HRRR v4
    #: recipe's 15-minute history wrote an 18.7 GB full frame every 15
    #: model minutes; the writer lands one in about 40 s while the model
    #: makes 15 minutes in about 15 s, so frames queued behind each other
    #: and the step loop blocked 10-30 s per write, growing over the run
    #: (fh3 171 s/fh against about 62 without output).
    full_history_interval_s: float = 0.0
    between_preset: str = "minimal"
    between_history_add: tuple[str, ...] = ()
    #: Where this block was read from, for refusal text only.
    source: str = "<config>"
    formats: tuple[str, ...] = ("netcdf",)
    icechunk_store: str | None = None

    # ``_include`` -- the resolved include set, or None for "everything
    # produced" -- is NOT a dataclass field, deliberately.
    # ``dataclasses.asdict`` walks every field, and both the
    # restart-identity payload and the resolved-TOML round trip walk the
    # experiment that way; a frozenset in that walk is not
    # JSON-serializable.  It is derived state, so __post_init__ sets it
    # as a plain attribute.

    def __post_init__(self) -> None:
        from woof.output_formats import selected
        selected(self.formats)
        if self.preset not in HISTORY_PRESETS:
            raise ValueError(
                f"[output] preset = {self.preset!r} in {self.source} is not "
                f"a known history preset; choose one of "
                f"{sorted(HISTORY_PRESETS)}.  'full' is the DEFAULT and "
                "writes every variable the run produces; 'minimal' keeps "
                "the 2-D surface state and the accumulators; 'severe' adds "
                "the storm-scale volumes (winds, theta, hydrometeors, "
                "REFL_10CM).")
        if self.history_vars and self.history_drop:
            raise ValueError(
                f"[output] of {self.source} sets both history_vars and "
                "history_drop.  They are an include list and an exclude "
                "list over the same inventory and there is no rule for "
                "which wins -- choose one: history_vars to name what the "
                "tape keeps, history_drop to name what it sheds.")
        if self.preset != "full" and self.history_vars:
            raise ValueError(
                f"[output] of {self.source} sets preset = "
                f"{self.preset!r} together with history_vars.  Both are "
                "include lists over the same inventory -- choose one: the "
                "preset for a named set, history_vars for your own.  A "
                "preset MAY be trimmed further with history_drop.")
        for key, names in (("history_vars", self.history_vars),
                           ("history_drop", self.history_drop)):
            for name in names:
                if name not in HISTORY_VOCABULARY:
                    raise ValueError(
                        f"[output] {key} of {self.source} names "
                        f"{name!r}, which is not a wrfout history "
                        f"variable.{_did_you_mean(name, HISTORY_VOCABULARY)}"
                        "  The valid set is: "
                        f"{', '.join(sorted(HISTORY_VOCABULARY))}.")
        if (WIND_ROTATION_FIELDS & set(self.history_drop)
                and not PROJECTED_WIND_FIELDS <= set(self.history_drop)):
            raise ValueError(
                f"[output] history_drop of {self.source} drops "
                f"{sorted(WIND_ROTATION_FIELDS & set(self.history_drop))} "
                "while the tape keeps a grid-relative wind "
                f"({sorted(PROJECTED_WIND_FIELDS - set(self.history_drop))}): "
                "without SINALPHA and COSALPHA no reader can turn that wind "
                "earth-relative, and station scoring drops it.  Keep the "
                "rotation, or drop U10, V10, U and V as well.")
        for name in self.history_drop:
            if name in STRUCTURAL_FIELDS:
                raise ValueError(
                    f"[output] history_drop of {self.source} names "
                    f"{name!r}, which is structural: it is part of the "
                    "georeference, the vertical coordinate or the time "
                    "coordinate, and a wrfout without it is not a smaller "
                    "file but an unreadable one -- no reader can place a "
                    "value in space or time.  The structural set is "
                    f"{sorted(STRUCTURAL_FIELDS)}.  Drop the science "
                    "fields you do not need instead, or use preset = "
                    "'minimal'.")
        if self.full_history_interval_s < 0 or (
                self.full_history_interval_s != self.full_history_interval_s):
            raise ValueError(
                f"[output] full_history_interval_s of {self.source} must be "
                f"a non-negative number of seconds, got "
                f"{self.full_history_interval_s!r}.")
        if self.between_preset not in HISTORY_PRESETS:
            raise ValueError(
                f"[output] between_preset = {self.between_preset!r} in "
                f"{self.source} is not a known history preset; choose one "
                f"of {sorted(HISTORY_PRESETS)}.")
        for name in self.between_history_add:
            if name not in HISTORY_VOCABULARY:
                raise ValueError(
                    f"[output] between_history_add of {self.source} names "
                    f"{name!r}, which is not a wrfout history variable."
                    f"{_did_you_mean(name, HISTORY_VOCABULARY)}")
        if (self.between_preset != "minimal" or self.between_history_add) \
                and not self.full_history_interval_s:
            raise ValueError(
                f"[output] of {self.source} sets between_preset or "
                "between_history_add without full_history_interval_s: the "
                "between-hours subset applies only to frames between the "
                "full-history leads, so name that interval (3600 for "
                "hourly full history).")
        include: frozenset[str] | None
        if self.history_vars:
            include = frozenset(self.history_vars)
            if include & PROJECTED_WIND_FIELDS:
                include |= WIND_ROTATION_FIELDS
        else:
            include = HISTORY_PRESETS[self.preset]
        object.__setattr__(self, "_include", include)

    @classmethod
    def from_mapping(cls, table, *, source: str = "<config>"
                     ) -> "HistorySelection":
        """Validate one ``[output]`` table.  ``None`` gives :data:`FULL`."""
        if table is None:
            return FULL
        if not isinstance(table, dict):
            raise ValueError(
                f"[output] of {source} must be a table, e.g. "
                '[output] preset = "minimal", got '
                f"{table!r}.")
        unknown = sorted(set(table) - OUTPUT_KEYS)
        if unknown:
            suggestions = "".join(
                _did_you_mean(key, OUTPUT_KEYS) for key in unknown)
            raise ValueError(
                f"unknown key(s) {unknown} in [output] of {source}; known "
                f"keys: {sorted(OUTPUT_KEYS)}.{suggestions}")
        for key in ("history_vars", "history_drop"):
            value = table.get(key)
            if value is None:
                continue
            if isinstance(value, str) or not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"[output] {key} of {source} must be a LIST of wrfout "
                    f'variable names, e.g. {key} = ["QCLOUD", "QRAIN"], '
                    f"got {value!r}.")
            for name in value:
                if not isinstance(name, str):
                    raise ValueError(
                        f"[output] {key} of {source} must contain variable "
                        f"NAMES; {name!r} is not a string.")
            if not value:
                # WRITTEN AND EMPTY, which is not the same as absent: an
                # empty include list would write a bare structural
                # skeleton with no science in it, and an empty exclude
                # list is a key that does nothing.  Both are far more
                # likely to be a half-finished edit than an intent.
                raise ValueError(
                    f"[output] {key} of {source} is an empty list.  An "
                    "empty history_vars would write a structural skeleton "
                    "with no science in it, and an empty history_drop "
                    "sheds nothing -- name the variables you mean, or "
                    "delete the key.")
        add = table.get("between_history_add", ()) or ()
        if isinstance(add, str) or not isinstance(add, (list, tuple)):
            raise ValueError(
                f"[output] between_history_add of {source} must be a LIST of "
                f"wrfout variable names, got {add!r}.")
        interval = table.get("full_history_interval_s", 0.0) or 0.0
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise ValueError(
                f"[output] full_history_interval_s of {source} must be a "
                f"number of seconds, got {interval!r}.")
        formats = table.get("formats", ("netcdf",))
        if not isinstance(formats, (list, tuple)):
            raise ValueError(f"[output] formats of {source} must be a list of row names")
        return cls(
            formats=tuple(formats),
            icechunk_store=table.get("icechunk_store"),
            preset=str(table.get("preset", "full")),
            history_vars=tuple(table.get("history_vars", ()) or ()),
            history_drop=tuple(table.get("history_drop", ()) or ()),
            full_history_interval_s=float(interval),
            between_preset=str(table.get("between_preset", "minimal")),
            between_history_add=tuple(add),
            source=source,
        )

    def for_lead(self, lead_seconds) -> "HistorySelection":
        """The selection one frame at ``lead_seconds`` is written with.

        Itself at a full-history lead (or with no sub-hourly subset set, or
        an unknown lead); otherwise the between-hours subset.
        """
        interval = float(self.full_history_interval_s or 0.0)
        if not interval or lead_seconds is None:
            return self
        steps = float(lead_seconds) / interval
        if abs(steps - round(steps)) * interval < 0.5:
            return self
        return _between_selection(self.between_preset,
                                  self.between_history_add, self.source)

    @property
    def subhourly(self) -> bool:
        """Whether some frames are written with the between-hours subset."""
        return bool(self.full_history_interval_s)

    @property
    def writes_everything(self) -> bool:
        """``True`` when this selection removes nothing from any frame."""
        return self._include is None and not self.history_drop

    def keeps(self, name: str) -> bool:
        """Whether one produced variable survives this selection."""
        if name in STRUCTURAL_FIELDS:
            return True
        if name in self.history_drop:
            return False
        return self._include is None or name in self._include

    def select(self, produced) -> tuple[str, ...]:
        """The kept subset of ``produced``, IN THE PRODUCED ORDER.

        Order is not cosmetic: HDF5 lays its name heap out in
        variable-creation order, so the same numbers written in a
        different order give a file that hashes differently while every
        variable in it compares equal.  The filter may only remove.
        """
        return tuple(name for name in produced if self.keeps(name))

    def dropped(self, produced) -> tuple[str, ...]:
        """The subset of ``produced`` this selection removes."""
        return tuple(name for name in produced if not self.keeps(name))

    def wrfout_attrs(self, produced) -> dict[str, str]:
        """Global attributes recording what this selection did.

        A trimmed file must SAY it was trimmed and name what it shed:
        that is what lets the render front door answer "this run dropped
        REFL_10CM" instead of the much weaker "this file has no
        REFL_10CM".  A full-default run stamps NOTHING, so every wrfout
        written before this feature keeps its exact header.
        """
        if self.writes_everything:
            return {}
        dropped = self.dropped(produced)
        return {
            "GPUWM_HISTORY_PRESET": self.preset,
            "GPUWM_HISTORY_SELECTION": self.spelling(),
            "GPUWM_HISTORY_DROPPED": ",".join(dropped),
        }

    def apply(self, frame) -> tuple[dict, dict[str, str]]:
        """(kept frame, history attributes) for an ALREADY-ASSEMBLED frame.

        The seam for a writer that has the whole host frame in hand --
        ``woof downscale``'s offline child, which builds one frame and
        writes it synchronously.  The asynchronous per-domain writer does
        NOT use this: it has to know the kept set BEFORE it stages, so a
        dropped volume never crosses the bus, and it resolves the same
        selection through its own ``_history_plan``.

        A full-default selection returns the frame object unchanged and
        an empty attribute set, so nothing about a default run moves.
        """
        if self.writes_everything:
            return frame, {}
        produced = tuple(frame)
        return ({name: frame[name] for name in self.select(produced)},
                self.wrfout_attrs(produced))

    def spelling(self) -> str:
        """The one-line form of this block, for receipts and messages."""
        parts = [f"preset={self.preset}"]
        if self.history_vars:
            parts.append("history_vars=" + ",".join(self.history_vars))
        if self.history_drop:
            parts.append("history_drop=" + ",".join(self.history_drop))
        return " ".join(parts)

    def product_cost(self) -> tuple[int, int, bool]:
        """``(products kept, products in the full inventory, exact)``.

        Computed from :data:`PRESET_PRODUCT_COUNTS` INSIDE this object,
        never from what a caller passes as ``produced``: the two
        plan-review doors hand in different inventories, and a
        configuration that prints two different figures depending on
        which door read it is the disagreement this lane exists to end.

        ``exact`` is False for a selection that is not exactly a named
        preset.  ``history_vars`` and ``history_drop`` narrow the tape
        further than the preset they start from, so the preset's figure
        is an UPPER BOUND there, and the caller says so rather than
        printing a measured-looking number nobody measured.
        """

        total = PRESET_PRODUCT_COUNTS["full"]
        kept = PRESET_PRODUCT_COUNTS.get(self.preset, total)
        return kept, total, not (self.history_vars or self.history_drop)

    def warn_lost_products(self, produced, *, where: str) -> None:
        """Name the render products this selection kills, at plan time.

        The user asked for storage back; what they get in exchange is a
        set of pictures they can no longer draw, and they should read
        that BEFORE the run rather than discover it at render time.  It
        is a warning and not a refusal: shedding a product deliberately
        is exactly what this feature is for.

        It LEADS WITH PRODUCTS.  A variable count answers "how much disk
        did I save"; the reader is choosing between pictures and disk,
        and the product figure is the other half of that trade.  It used
        to be absent entirely, so ``preset = "minimal"`` reported two
        products by name and a bare variable count while the measured
        ladder in this module recorded that the same preset leaves 30 of
        162 renderable.
        """
        dropped = self.dropped(produced)
        if not dropped:
            return
        lost = lost_products(dropped)
        if not lost:
            return
        kept, total, exact = self.product_cost()
        if kept < total:
            # The named rungs do not exhaust the loss: the ladder's own
            # 162 -> 160 step belongs to the rest of the inventory, and
            # a sentence whose figures do not add up reads as an error
            # in the measurement rather than as what it is.
            named = [f"{count} to {','.join(fields)}"
                     for fields, count in PRODUCT_LOSS_ATTRIBUTION]
            remainder = (total - kept) - sum(
                count for _, count in PRODUCT_LOSS_ATTRIBUTION)
            if remainder > 0:
                named.append(f"{remainder} to the rest of the full "
                             f"inventory")
            attribution = ", ".join(named)
            cost = (f"keeps {kept} of {total} render products, "
                    f"{total - kept} lost: {attribution}"
                    if exact else
                    f"keeps at most {kept} of {total} render products "
                    f"(an upper bound: this selection narrows "
                    f"preset={self.preset} further)")
        else:
            cost = (f"keeps all {total} render products"
                    if exact else
                    f"keeps at most {total} render products "
                    "(an upper bound: this selection narrows the preset "
                    "it starts from)")
        mapping = "; ".join(
            f"{product} needs {', '.join(missing)}"
            for product, missing in sorted(lost.items()))
        # NAME the drops while naming them is readable; COUNT them once
        # it is a wall.  A preset sheds well over a hundred scheme
        # carriers, and printing all of them (measured: >1,700 characters
        # on one real d02) buries the actionable half -- which product
        # dies of which variable -- under state nobody was reading.
        # Sorted, not in produced order: the plan-time caller passes the
        # VOCABULARY (a frozenset), whose iteration order varies per
        # process, and a warning that reorders itself between two runs of
        # the same config is a warning a reader cannot diff.
        shed = (", ".join(sorted(dropped))
                if len(dropped) <= _NAMED_DROP_LIMIT
                else f"{len(dropped)} variables")
        warn(
            f"{where}: [output] {self.spelling()} {cost}.  It drops "
            f"{shed}, so these render products can no longer be drawn "
            f"from this run's history: {mapping}.  Keep the variable(s) "
            "if you want the product(s); otherwise this is the storage "
            "you asked for.",
            "The product figure comes from this module's own measured "
            f"ladder ({PRODUCT_COUNT_BASIS}), read as products rather "
            "than as prose.  The NAMED mapping beside it is a narrower "
            "thing: woof.io.history_selection.PRODUCT_HISTORY_INPUTS "
            f"carries {len(PRODUCT_HISTORY_INPUTS)} verified rows, so of "
            f"the {PRESET_PRODUCT_COUNTS['full']} products the measured "
            "full inventory renders, roughly "
            f"{PRESET_PRODUCT_COUNTS['full'] - len(PRODUCT_HISTORY_INPUTS)} "
            "were NOT checked by name here.  The renderer answers for "
            "those itself, against a real file, through --list-products: "
            "a product named above will refuse by name at render time, "
            "and one that is not named may still be lost.")


#: The default: write everything the run produces, stamp nothing.
FULL = HistorySelection()


_BETWEEN_CACHE: dict = {}


#: What a QUICK-LOOK frame need not carry although every full frame must:
#: ``T`` (the one 3-D volume a reader needed to open a wrfout -- the vendored
#: wrf-core now takes the grid's dimensions from ZNU and XLAT when T is
#: absent) and ``PHB`` (the base geopotential, which never changes over a
#: fixed domain and is in every full frame).  A quick-look frame keeps either
#: one only when its own variable list names it.
#:
#: Breakage it prevents (WOOF run_0e9d51235e54e762, 2026-10-28): 22
#: quick-look frames of 200 x 150 x 49 for eight chosen fields were 26.4 MB
#: each, and T and PHB -- written whatever the user chose -- were about 45 %
#: of every frame.
QUICKLOOK_UNFORCED_FIELDS: frozenset[str] = frozenset({"T", "PHB"})

#: The global attribute a quick-look frame of a FIXED domain carries: the
#: path, relative to the frame's own folder, of the domain's first full
#: frame, where a reader finds the static fields the quick-look frame left
#: out (wrf-core ``WrfFile`` reads PHB, PB, MUB and HGT there).
STATIC_FRAME_ATTR = "GPUWM_STATIC_FRAME"


@dataclass(frozen=True)
class QuickLookSelection(HistorySelection):
    """A quick-look stream's selection (``quicklook = {...}`` on a domain).

    The same fields and validation as :class:`HistorySelection` (so a resolved
    document and every fingerprint read from it keep their bytes); it only
    stops forcing :data:`QUICKLOOK_UNFORCED_FIELDS` into the frame.
    """

    def keeps(self, name: str) -> bool:
        if (name in QUICKLOOK_UNFORCED_FIELDS and self._include is not None
                and name not in self._include):
            return False
        return super().keeps(name)

    def on_moving_domain(self) -> "HistorySelection":
        """The selection for a domain that moves: PHB changes with every
        move, so it stays in each frame (T may still be left out)."""
        return _MovingQuickLookSelection(**{
            name: getattr(self, name)
            for name in self.__dataclass_fields__})


@dataclass(frozen=True)
class _MovingQuickLookSelection(QuickLookSelection):
    def keeps(self, name: str) -> bool:
        if name == "PHB":
            return True
        return super().keeps(name)

    def on_moving_domain(self) -> "HistorySelection":
        return self


def _between_selection(preset, add, source) -> HistorySelection:
    """The between-hours subset as an ordinary selection (cached)."""
    key = (preset, tuple(add), source)
    found = _BETWEEN_CACHE.get(key)
    if found is None:
        base = HISTORY_PRESETS[preset]
        names = (tuple(sorted(base | frozenset(add))) if base is not None
                 else ())
        found = (HistorySelection(history_vars=names,
                                  source=f"{source} (between full-history leads)")
                 if names else FULL)
        _BETWEEN_CACHE[key] = found
    return found


def resolve(tree, domain) -> HistorySelection:
    """One domain's selection: its own block, else the tree-wide one.

    A domain that says nothing takes the tree-wide table; a domain that
    speaks overrides it ENTIRELY rather than merging key by key, on the
    ``[tiles]`` precedent -- a half-inherited selection ("the preset from
    the tree, the drop list from the domain") is a configuration nobody
    can read off the file.
    """
    if domain is not None:
        return domain
    return FULL if tree is None else tree


__all__ = [
    "FULL",
    "HISTORY_PRESETS",
    "PRESET_PRODUCT_COUNTS",
    "PRODUCT_COUNT_BASIS",
    "PRODUCT_LOSS_ATTRIBUTION",
    "HISTORY_VOCABULARY",
    "HistorySelection",
    "OUTPUT_KEYS",
    "PRODUCT_HISTORY_INPUTS",
    "STRUCTURAL_FIELDS",
    "QUICKLOOK_UNFORCED_FIELDS",
    "QuickLookSelection",
    "STATIC_FRAME_ATTR",
    "lost_products",
    "resolve",
]
