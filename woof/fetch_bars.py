"""Record-count bars: derived from the live inventory, tripwired by the
certified constants.

A downloaded subset has to carry exactly the records the fail-closed
bridges select, and ``woof fetch`` refuses anything else.  For a long
time the bar was a hardcoded integer -- 124 for the GFS NOMADS subset,
561 and 18 for the HRRR native and soil subsets.  That is correct right
up until NCEP changes an inventory, at which point a hardcode turns a
routine upstream change into a mystery breakage with no remedy in the
error text.

So the bar is now **derived** from the index of the very object being
fetched, and the certified constants stay on as a **tripwire**:

* derived == certified -- the ordinary case, nothing is said;
* derived != certified -- a loud, specific warning naming both numbers
  and what changed, and the fetch refuses unless the operator passes
  ``--accept-inventory-change``.

Never silently adapt (a quietly smaller subset is a corrupt
initialization), and never mystery-break (an operator who has read the
warning can proceed in one flag).  Accepting a change is a deliberate
re-certification event and the manifest records that it happened.
"""

from __future__ import annotations

from dataclasses import dataclass

from woof.explain import layered

#: The record counts this ArWen was certified against.  Each is the
#: exact inventory its fail-closed bridge selects; see
#: ``tools/grib1_bridge/src/bin/{gfs,hrrr}_grib2_bridge.rs``.
CERTIFIED_RECORD_BARS = {
    #: 21 pressure levels x {HGT, TMP, RH, UGRD, VGRD} + 11
    #: surface/near-surface + 8 soil-layer records.
    "gfs": 124,
    #: 11 hybrid fields on all 50 levels + 11 surface/near-surface.
    "hrrr-atmosphere": 561,
    #: The same selection with the analyzed aerosol number pair
    #: (QNWFA/QNIFA) on its 50 levels, for a preparation that reads it.
    "hrrr-atmosphere-aerosol": 661,
    #: TSOIL + SOILW at the nine RUC level depths.
    "hrrr-soil": 18,
}

#: The flag that turns a tripped tripwire from a refusal into a
#: recorded, deliberate re-certification.
ACCEPT_FLAG = "--accept-inventory-change"

#: What each bar counts, for the warning text.
_BAR_SUBJECTS = {
    "gfs": "the GFS pgrb2.0p25 variable/level selection",
    "hrrr-atmosphere": "the HRRR wrfnat atmosphere selection",
    "hrrr-atmosphere-aerosol": ("the HRRR wrfnat atmosphere selection with "
                                "the analyzed aerosol numbers"),
    "hrrr-soil": "the HRRR wrfprs soil selection",
}


@dataclass(frozen=True)
class RecordBar:
    """One resolved bar and the story behind it."""

    kind: str
    #: The count actually applied to the downloaded file.
    expected: int
    #: What this ArWen was certified against.
    certified: int
    #: What the live index says the selection yields, when it could be
    #: read.  ``None`` means the index was unavailable and the certified
    #: constant is standing in.
    derived: int | None

    @property
    def tripped(self) -> bool:
        return self.derived is not None and self.derived != self.certified

    @property
    def source(self) -> str:
        if self.derived is None:
            return "certified constant (live index unavailable)"
        if self.tripped:
            return "live index, accepted over the certified constant"
        return "live index, matching the certified constant"

    def as_manifest(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "expected": self.expected,
            "certified": self.certified,
            "derived": self.derived,
            "source": self.source,
            "inventory_change_accepted": self.tripped,
        }


#: What a refusal says about the disk when the bar was resolved BEFORE
#: anything was transferred.  Callers that resolve the bar afterwards
#: pass ``on_refusal`` and say what is actually there.
NOTHING_DOWNLOADED = "Nothing was downloaded."


def resolve_bar(kind: str, derived: int | None, *,
                accept_inventory_change: bool = False,
                progress=print,
                on_refusal=None,
                certified: int | None = None) -> RecordBar:
    """Resolve the record-count bar for ``kind``.

    Raises :class:`ValueError` when the live inventory disagrees with
    the certified constant and the operator has not accepted the change.

    ``certified`` overrides the table entry for requests whose census is
    a known function of the request itself rather than a fixed number.
    The GFS selection is the case: it takes five records per isobaric
    level, so a run that asks for a model top above 100 hPa fetches a
    longer ladder and legitimately expects more than 124 records.  The
    tripwire still fires -- it is simply aimed at the count this
    request's ladder implies, not at the count the default ladder
    implies, which would otherwise refuse every deeper top as an
    upstream inventory change.

    ``on_refusal`` is for callers that can only derive the bar *after*
    the payload has landed -- the Rust HRRR transport reads the census
    out of the index it kept beside the object it just wrote.  Such a
    caller passes a zero-argument callable that disposes of the payload
    and returns one sentence saying what is now on disk; it runs before
    the refusal is raised, and its sentence replaces
    :data:`NOTHING_DOWNLOADED`.  Without it the refusal claims nothing
    was downloaded, which is only true when the bar came first.
    """

    if kind not in CERTIFIED_RECORD_BARS:
        raise ValueError(
            f"unknown record bar {kind!r}; known: "
            f"{sorted(CERTIFIED_RECORD_BARS)}")
    if certified is None:
        certified = CERTIFIED_RECORD_BARS[kind]
    if derived is None:
        return RecordBar(kind=kind, expected=certified, certified=certified,
                         derived=None)
    if derived == certified:
        return RecordBar(kind=kind, expected=certified, certified=certified,
                         derived=derived)

    subject = _BAR_SUBJECTS[kind]
    headline = (
        f"fetch: UPSTREAM INVENTORY CHANGE -- {subject} now yields "
        f"{derived} records, but this WOOF was certified against "
        f"{certified}.")
    if not accept_inventory_change:
        disposition = (NOTHING_DOWNLOADED if on_refusal is None
                       else str(on_refusal()))
        raise ValueError(layered(
            f"{headline}\n"
            f"  {disposition}\n"
            f"  remedy: confirm the change against the provider's "
            f"announcement, then re-run with {ACCEPT_FLAG} to fetch "
            f"against the live count of {derived} and record the "
            f"acceptance in the manifest.",
            "  why: a changed inventory is a re-certification event, not "
            "a transient error -- the fail-closed bridge downstream "
            "selects by exact field identity and will reject a file "
            "whose census it does not recognise."))
    progress(f"{headline}  Proceeding with {derived} because "
             f"{ACCEPT_FLAG} was given; the manifest records the "
             f"acceptance.")
    return RecordBar(kind=kind, expected=derived, certified=certified,
                     derived=derived)


# ──────────────────────────────────────────────────────────
# Deriving the bar from a live index
# ──────────────────────────────────────────────────────────

#: Where the NOMADS grib-filter names a GRIB2 parameter differently from
#: the ``.idx`` published beside the object: the filter keeps the older
#: wgrib2 table.  Cloud mixing ratio (0.1.22) is ``var_CLWMR`` to the
#: filter and ``CLMR`` in the index; the filter answers ``var_CLMR`` with
#: HTTP 500 "invalid parameter: var_CLMR" (filter_gdas_0p25.pl,
#: 2026-10-29).  Keyed filter name -> index name.
FILTER_TO_IDX_NAMES = {"CLWMR": "CLMR"}


def filter_variable(idx_name: str) -> str:
    """The grib-filter's ``var_`` parameter for an ``.idx`` variable name."""

    for filter_name, name in FILTER_TO_IDX_NAMES.items():
        if name == idx_name:
            return f"var_{filter_name}"
    return f"var_{idx_name}"


def nomads_selector_pairs(variables: tuple[str, ...],
                          levels: tuple[str, ...],
                          pressure_levels_hpa: tuple[int, ...],
                          ) -> frozenset[tuple[str, str]]:
    """The ``(variable, level)`` pairs a NOMADS CGI query asks for.

    The CGI's own vocabulary maps mechanically onto the ``.idx``
    columns: ``var_TSOIL`` is the ``TSOIL`` column, and
    ``lev_0.1-0.4_m_below_ground`` is the ``0.1-0.4 m below ground``
    column, except where :data:`FILTER_TO_IDX_NAMES` says the two name
    a parameter differently.  Building the pair set this way means the
    derived bar comes from the same declaration the request does --
    there is no second table to drift.
    """

    names = tuple(FILTER_TO_IDX_NAMES.get(name.removeprefix("var_"),
                                          name.removeprefix("var_"))
                  for name in variables)
    level_names = [level.removeprefix("lev_").replace("_", " ")
                   for level in levels]
    # ``format(..., "g")`` because the index spells a level without
    # trailing zeros -- "50 mb", never "50.0 mb" -- and the ladder is a
    # float tuple once a requested model top has extended it upward.
    level_names += [f"{format(float(value), 'g')} mb"
                    for value in pressure_levels_hpa]
    return frozenset(
        (name, level) for name in names for level in level_names)


def count_index_selection(index_text: str,
                          wanted: frozenset[tuple[str, str]]) -> int:
    """Count ``.idx`` records whose ``(variable, level)`` is wanted.

    Deliberately exact on both columns.  A substring level match would
    let ``1 mb`` swallow ``100 mb``, ``150 mb`` and ``1000 mb``, which
    is precisely the over-count a record bar exists to notice.
    """

    matched = 0
    for line in index_text.splitlines():
        fields = line.strip().rstrip(":").split(":")
        if len(fields) < 6:
            continue
        if (fields[3], fields[4]) in wanted:
            matched += 1
    return matched


__all__ = [
    "ACCEPT_FLAG", "CERTIFIED_RECORD_BARS", "FILTER_TO_IDX_NAMES",
    "NOTHING_DOWNLOADED", "filter_variable",
    "RecordBar", "count_index_selection", "nomads_selector_pairs",
    "resolve_bar",
]
