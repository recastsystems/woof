"""WP-12b -- the mp=28 validation CAMPAIGN roll-up, and the evidence ratchet.

Every other file in the mp=28 suite gates one thing: a device helper, a
kernel, the adapter's composition, the driver wiring.  This one owns the
properties of the campaign AS A WHOLE, which is where a port quietly rots:

1.  THE G3 RESIDUAL MATRIX IS A RATCHET.  Nineteen committed WRF column
    fixtures times sixteen compared fields.  ``_PUBLISHED_G3`` records what
    every non-zero residual above 2e-6 measured on the tree WP-12b published
    from, and this file fails if ANY of them grows.  That is strictly
    stronger than the pass/fail gate in
    ``tests/test_thompson_aerosol_adapter.py``: a change that moves
    ``aero-nc-sed`` from 1.1e-6 to 1.9e-6 still passes the 2e-6 gate and
    still gets caught here.

2.  THE PUBLISHED EVIDENCE CANNOT DRIFT FROM THE MEASUREMENT.
    ``docs/public/wrf-comparison/mp28-column-evidence.md`` is the document a user
    will quote back.  Its per-fixture table is recomputed here and compared.
    A fixture that IMPROVES also fails this file, deliberately: an evidence
    document that understates the port is still a document nobody re-read.

3.  NO NEW SILENT SKIP.  Seven silent skips are how two critical defects
    survived wave 2 of this port.  A skipped test is an unmeasured claim
    wearing a green tick.  ``_SKIP_SITES`` is the complete, audited census of
    every place the mp=28 suite can skip, xfail or importorskip, and this
    file fails when one is ADDED as loudly as when one is removed.  The
    census is static (AST over the test sources), so it is deterministic and
    costs no device time.

4.  THE mp=8 FREEZE IS RE-ASSERTED AT THIS PACKAGE'S TIP.  MP28_PORT_SPEC.md
    makes the WP-00 receipt the merge criterion for WP-12, "re-run at the tip
    of every package branch".  This is that re-run.

5.  CCN_ACTIVATE.BIN SHIPS, AND STAYS OUT OF THE CLASSIC SET.  Port HARD
    RULE 5 was "stays unvendored"; the owner reversed that on 2026-08-01
    once WRF's public-domain dedication was checked.  The assertion is
    inverted rather than dropped, and is still made from inside the campaign
    file, because both directions are invariants whose violation would be
    invisible in a passing test suite and fatal at release: an unshipped
    table breaks mp=28 for every user, and a table inside
    CLASSIC_TABLE_ASSETS breaks mp=8.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
import re

import pytest

from conftest import requires_gpu

_TESTS = pathlib.Path(__file__).resolve().parent
_ROOT = _TESTS.parent
EVIDENCE_DOC = (_ROOT / "docs" / "public" / "wrf-comparison"
                / "mp28-column-evidence.md")

#: The gate every fixture is measured against.  Transcribed from
#: tests/test_thompson_aerosol_adapter.py::_END_TO_END_DEFAULT_BOUND; this
#: file never defines its own, looser, copy.
G3_GATE = 2.0e-6


# ---------------------------------------------------------------------------
# 1.  The G3 residual matrix, pinned.
# ---------------------------------------------------------------------------

#: MEASURED on an RTX 5090 (CUDA 12, cupy 14.1.1, nvrtc defaults) by
#: WP-12c, from a full re-drive of the shipped adapter over every fixture in
#: ``woof/data/thompson/oracle-aero/``.  ONLY the residuals that exceed
#: :data:`G3_GATE` are listed; every other (fixture, field) cell of the
#: 22 x 16 matrix is at or below it and is covered by
#: :data:`_G3_CLEAN_FIXTURES` and by the adapter's own gate.
#:
#: These are UPPER BOUNDS, not expected values.  The assertion is
#: ``measured <= published * 1.05``.  Nothing here is a tolerance in the
#: sense of "how close is close enough"; the tolerance is G3_GATE and these
#: are the accurate, published distances from it.
#:
#: WHAT MOVED IN WP-13, ALL RE-MEASURED HERE:
#:   * WP-13a restored WRF's LEVEL-WISE working rain density for sedimentation
#:     (module_mp_thompson.F:3237-3238 from the :3193 TAU+1 density everywhere,
#:     :3568-3570 from the :3490 post-condensation density only inside the
#:     :3501-3502 gate; woof/core/kernels/thompson_aerosol_sat.cu wrote the
#:     post-condensation one unconditionally).  ``aero-drop-evap`` and
#:     ``aero-ice-demott-idxin`` LEFT this table completely;
#:     ``aero-cloud-freeze-nc`` lost qr, nr_per_kg and rainnc_mm and kept only
#:     qc; ``aero-reduces-to-classic`` went qr 7.813e-05 -> 1.788e-07 (inside
#:     the gate, so it is gone) and nr_per_kg 4.832e-05 -> 5.700e-06.
#:   * WP-13b pinned the contraction of the source networks' terminal
#:     multiply-add (WRF's :3973-4023 `q1d + qten*DT`, which the gfortran -O2
#:     oracle cannot fuse and nvrtc did).  ``aero-nc-cap``'s qc and nc and
#:     ``aero-ice-demott-idxin``'s surface accumulators became BITWISE exact.
#:   * ONE CELL GREW AND IS PUBLISHED AS GROWN: ``aero-cold-overlap`` qr
#:     3.6666e-05 -> 4.4426e-05.  Bisected to the cold network's own qr apply;
#:     reverting that single pin restores the old number and simultaneously
#:     un-does four aero-ice-demott-idxin improvements, two of them exact.  In
#:     the units the adapter's _RESIDUAL_ATTRIBUTION records this cell at --
#:     ulps of the entry value, because 99.5% of the level's rain is consumed
#:     in the step -- it moved 1.477 -> 1.789 ulp.
#:
#: WHAT MOVED IN THE PUBLICATION BEFORE:
#:   * ``aero-ice-koop`` LEFT the table.  ni 1.764e-03 / qi 1.612e-03 /
#:     effi 5.093e-05 -- what three auditors called the port's largest genuine
#:     physics gap -- now measure 3.396e-07 / 1.534e-07 / 1.886e-07, inside
#:     the gate.  WP-06 closed it.
#:   * ``aero-cold-overlap`` GAINED qc 1.000e+00, nc_per_kg 1.000e+00 and
#:     effc_m 8.102e-01 at 0-based level 4.  MEASURED: WRF ends the step there
#:     with qc = 1.4551915228366852e-11 kg/kg -- exactly 2**-36, and exactly
#:     ONE float32 ulp of the 2.325216e-04 kg/kg the level entered with -- and
#:     nc = 1.833336 per kg, while woof ends at exactly zero, so the relative
#:     metric reports 1.0 on an absolute difference of one ulp.  Recorded as a
#:     MISS, not allowanced.  See the attribution note on
#:     ``tests/test_thompson_aerosol_adapter.py::_G3_RESIDUALS`` for what is
#:     and is not claimed about where the change came from.
#:   * three WP-08 sedimentation columns joined the deck; ``wp08-freeze`` and
#:     ``wp08-nusweep`` carried residuals 1.4x and 2.3x the gate.
#:   * ``wp08-freeze`` LEFT the table when the rain fallout was handed WRF's
#:     L_qr (nr 2.7239e-06 -> 4.006e-07 at level 0 on the RTX 4090 and the
#:     RTX 5090; 8.012e-08 on the host build of the kernels).
#:   * ``aero-cloud-freeze-nc``'s effc_m and ``aero-ice-demott-idxin``'s qc
#:     fell inside the gate and left the table.
#:
#: AND NOW EMPTY.  The 2.8.6 accumulator rework (WRF's qcten, qrten, nrten,
#: qiten and niten carried through every stage and applied once, mp=28's
#: own tendency-form rain and ice fallout, the cold network's REAL
#: rain-conservation ratio) closed the three rows that were left:
#: aero-cloud-freeze-nc qc 4.9256e-06 -> 0, aero-cold-overlap
#: qc / nc_per_kg / effc_m / nr_per_kg / qr -> 0, and wp08-nusweep qr
#: 4.6424e-06 -> 5.532e-07, all on an RTX 5090 and an RTX 4090 alike.
#:
#: ONE ROW AGAIN, AND IT IS NOT A RESIDUAL: ``aero-cold-overlap`` is the
#: declared rain-graupel divergence.  WRF v4.6.1 reads its rain-graupel
#: collision tables out of bounds when the scheme is not hail aware
#: (module_mp_thompson.F:465, :607-615, :2527-2545) and WOOF reads the slab
#: the tables hold; with WRF's read emulated in a measurement copy of the
#: tree the fixture clears the gate.  Measured on an RTX PRO 6000 Blackwell
#: (sm_120).  Re-measured by development branch after WRF's snow and graupel
#: accumulators (:4054-4059): qs 2.2658e-06 -> 1.9259e-06 (inside the gate),
#: qg 8.7700e-05 -> 8.7763e-05, effs_m 4.6974e-06 -> 6.5743e-06.
_PUBLISHED_G3: dict[str, dict[str, float]] = {
    "aero-cold-overlap": {
        "qr": 5.0700e-04, "qi": 1.6180e-04, "qs": 1.9259e-06,
        "qg": 8.7763e-05, "ni_per_kg": 1.4643e-05, "nr_per_kg": 5.0656e-04,
        "effi_m": 5.3941e-05, "effs_m": 6.5743e-06,
    },
}

#: Fixtures that clear :data:`G3_GATE` on EVERY compared field.  Membership is
#: asserted for equality, not containment: a fixture that starts clearing the
#: gate must be moved here in the same change that updates the evidence
#: document, or the document silently understates the port.
#:
#: 21 of 22 since the cold and warm source networks were re-transcribed in
#: WRF's arithmetic order: ``aero-cold-overlap`` left for the declared
#: rain-graupel divergence (see :data:`_PUBLISHED_G3`).
#: 22 of 22 from the 2.8.6 accumulator rework, which added
#: ``aero-cloud-freeze-nc``, ``aero-cold-overlap`` and ``wp08-nusweep``.
#: Before it: 19 of 22.  ``wp08-freeze`` joined when the rain fallout was handed WRF's
#: L_qr.  ``aero-drop-evap`` and ``aero-ice-demott-idxin`` joined at
#: WP-13a.  ``aero-reduces-to-classic`` joined at the 1.4.1 merge, and it is
#: worth being exact about why, because this file used to name it as the
#: standing example of the two files disagreeing: this file applies the FLAT
#: gate and never honoured the adapter's ``_END_TO_END_BOUNDS`` carve-out,
#: so it kept the fixture out while the adapter counted it clean.  The
#: carve-out is now RETIRED -- the merge inherited the mp=8 lane's two
#: sedimentation reconciliations and level 5's nr_per_kg went 5.7005e-06 to
#: 4.146e-07 -- so both files agree on it for the first time, at the flat
#: gate, with nothing carved out.
_G3_CLEAN_FIXTURES = (
    "aero-ccn-activate",
    "aero-ccn-sweep",
    "aero-cloud-freeze-nc",
    "aero-drop-evap",
    "aero-ice-demott-dep",
    "aero-ice-demott-idxin",
    "aero-ice-koop",
    "aero-init-profile",
    "aero-nc-accrete",
    "aero-nc-auto",
    "aero-nc-cap",
    "aero-nc-effrad",
    "aero-nc-sed",
    "aero-reduces-to-classic",
    "aero-scav-frozen",
    "aero-scav-rain",
    "aero-sfc-emit",
    "aero-warm-overlap",
    "wp08-freeze",
    "wp08-melt",
    "wp08-nusweep",
)

#: THE FIXTURE DECK, PARTITIONED, because the deck grew and the published
#: narrative did not.  ``woof/data/thompson/oracle-aero/*-column.csv`` now
#: resolves to TWENTY-TWO columns: the nineteen scenarios MP28_PORT_SPEC.md
#: specifies (ids 101-119, all named ``aero-*``) plus three WP-08
#: sedimentation columns from the same oracle build.  The evidence document's
#: Class A table is about the nineteen and says so in its heading, and its row
#: regex only ever matched ``aero-`` names, so the three below were never
#: capable of being checked against it.  Naming them makes that explicit and
#: makes a FOURTH un-documented fixture fail loudly instead of vanishing.
_FIXTURES_OUTSIDE_THE_DOCUMENT = (
    "wp08-freeze",
    "wp08-melt",
    "wp08-nusweep",
)


def _adapter_module():
    """The WP-09 harness, imported read-only.

    Re-implementing the entry-state reconstruction here would fork the one
    piece of this port that must not be forked: the search for a ``thp``
    whose float32 round trip reproduces the fixture temperature BIT-exactly.
    Two copies of that would drift, and the drift would show up as a physics
    residual.  The import is the lockstep.
    """
    import test_thompson_aerosol_adapter as adapter

    for name in ("_FIXTURES", "_END_TO_END_FIELDS", "_run_g3",
                 "_END_TO_END_DEFAULT_BOUND"):
        assert hasattr(adapter, name), (
            f"tests/test_thompson_aerosol_adapter.py no longer exposes "
            f"{name}; the campaign roll-up drives the G3 harness through it "
            "and cannot measure the matrix without it")
    assert adapter._END_TO_END_DEFAULT_BOUND == G3_GATE, (
        f"the adapter's G3 gate moved to "
        f"{adapter._END_TO_END_DEFAULT_BOUND}; this file pins {G3_GATE} and "
        "must not carry a different one")
    return adapter


_MATRIX: dict[str, dict[str, float]] = {}


def _g3_matrix(cp):
    """Measure the whole 22 x 16 matrix once per session."""
    if _MATRIX:
        return _MATRIX
    _tables_or_skip()
    adapter = _adapter_module()
    fields = list(adapter._END_TO_END_FIELDS) + ["rainnc_mm"]
    for scenario in adapter._FIXTURES:
        measured, _ = adapter._run_g3(cp, scenario)
        _MATRIX[scenario] = {name: float(measured[name]) for name in fields}
    return _MATRIX


@requires_gpu
def test_g3_residual_matrix_never_grows_beyond_the_published_values():
    """The ratchet.  Every published residual is an upper bound.

    Runs the same adapter calls the G3 gate runs and compares the complete
    matrix -- including the cells that are comfortably inside the gate --
    against what the evidence document publishes.  A regression that doubles a
    1e-6 residual is invisible to a 2e-6 pass/fail gate and is caught here.

    THE DECK IS 22, NOT 19, AND THAT IS ASSERTED.  The glob over
    ``woof/data/thompson/oracle-aero/`` picks up MP28_PORT_SPEC.md's nineteen
    ``aero-*`` scenarios plus the three WP-08 sedimentation columns in
    :data:`_FIXTURES_OUTSIDE_THE_DOCUMENT`.  Both counts are pinned so neither
    a lost fixture nor a silently added one can pass.
    """
    import cupy as cp

    matrix = _g3_matrix(cp)
    adapter = _adapter_module()
    assert set(matrix) == set(adapter._FIXTURES)
    aero = sorted(name for name in matrix if name.startswith("aero-"))
    assert len(aero) == 19, f"expected 19 spec'd fixtures, measured {aero}"
    assert sorted(set(matrix) - set(aero)) == sorted(
        _FIXTURES_OUTSIDE_THE_DOCUMENT), sorted(set(matrix) - set(aero))
    assert len(matrix) == 22, f"expected 22 fixtures, measured {len(matrix)}"

    grew = []
    for scenario, published in _PUBLISHED_G3.items():
        assert scenario in matrix, scenario
        for field, bound in published.items():
            got = matrix[scenario][field]
            if got > bound * 1.05:
                grew.append(f"{scenario}.{field}: {got:.4e} > "
                            f"{bound:.4e} (published)")
    # Every cell NOT named in _PUBLISHED_G3 must be inside the gate.
    escaped = []
    for scenario, row in matrix.items():
        published = _PUBLISHED_G3.get(scenario, {})
        for field, value in row.items():
            if field in published:
                continue
            if value > G3_GATE:
                escaped.append(f"{scenario}.{field}: {value:.4e} > "
                               f"{G3_GATE:.1e} and is not a published "
                               "residual")
    assert not grew and not escaped, "\n".join(grew + escaped)


@requires_gpu
def test_the_set_of_fixtures_clearing_the_gate_is_exactly_what_is_published():
    """Improvements must be published too.

    An evidence document that says eight fixtures miss when only five do is
    just as wrong as one that says none miss.  This asserts SET EQUALITY, so
    closing a residual fails here until ``_G3_CLEAN_FIXTURES``, the
    ``_PUBLISHED_G3`` row and the document's table are all updated together.

    MEASURED: 22 of 22 clean under this file's gate since the 2.8.6
    accumulator rework, which also retired the near-cancellation exclusion
    discussed below, so this file's count and the adapter's unexceptioned
    count coincide again.  Before it: 19 of 22 clean under this file's gate, 17 of the 19 spec'd
    fixtures plus ``wp08-freeze`` and ``wp08-melt``.  ``wp08-freeze`` joined
    when the rain fallout was handed WRF's L_qr.  ``aero-drop-evap`` and
    ``aero-ice-demott-idxin`` joined when WP-13a restored WRF's level-wise
    sedimentation density; ``aero-ice-koop`` joined earlier, when WP-06 closed
    the homogeneous-haze-freezing rate gap.

    WHAT THIS FILE'S GATE IS, EXACTLY, BECAUSE IT IS NOT THE STRICTEST ONE IN
    THE PORT AND USED TO SAY IT WAS.  It is the flat 2.0e-06 applied to the
    SIXTEEN-field matrix ``_g3_matrix`` measures -- the 15 prognostic columns
    plus ``rainnc_mm``.  Two consequences, both stated rather than implied:

      * it does NOT honour the adapter's ``_END_TO_END_BOUNDS`` carve-out.
        That USED TO BE the reason ``aero-reduces-to-classic`` was counted
        MISSING here while the adapter's own gate passed it.  The 1.4.1
        merge retired that carve-out -- the inherited mp=8 sedimentation
        reconciliations took its level 5 from 5.700e-06 to 4.146e-07 -- so
        the two gates now agree on the fixture and it is counted clean HERE
        while the adapter's UNEXCEPTIONED table still reports it missing,
        which is the opposite asymmetry and comes from the second bullet;
      * it DOES inherit the adapter's ``_NEAR_CANCELLATION_LEVELS`` exclusion,
        because ``_run_g3`` applies it in both modes.  ``aero-reduces-to-
        classic``'s 0-based level 6 is therefore out of the qr and nr numbers
        in :data:`_PUBLISHED_G3` -- which is why its qr row reads 1.788e-07
        rather than the 3.155e-03 the adapter's UNEXCEPTIONED table reports.
        That level is not unmeasured: it is bounded in ulps of the entry value
        by ``tests/test_thompson_aerosol_adapter.py::
        test_the_near_cancellation_level_is_bounded_in_ulps_not_excluded`` (retired with the exclusion by the 2.8.6 accumulator rework)
        and published in ULP terms by the same file's
        ``test_every_g3_residual_is_published_in_ulps_as_well_as_relative``.

    The genuinely unexceptioned count -- 23 quantities, no exclusion, no
    carve-out -- is 18 of 22, and is pinned in the adapter file by
    ``test_the_unexceptioned_g3_table_is_printed_and_its_count_pinned``.  The
    two counts USED TO coincide and no longer do: 19 here, 18 there.  They
    were always DIFFERENT MEASUREMENTS asserted separately on purpose, and
    the divergence is the level-6 exclusion this file inherits and that one
    does not.
    """
    import cupy as cp

    matrix = _g3_matrix(cp)
    clean = {name for name, row in matrix.items()
             if all(value <= G3_GATE for value in row.values())}
    expected = set(_G3_CLEAN_FIXTURES)
    assert clean == expected, (
        f"newly clean: {sorted(clean - expected)}; no longer clean: "
        f"{sorted(expected - clean)}. Update _G3_CLEAN_FIXTURES, "
        "_PUBLISHED_G3 and docs/public/wrf-comparison/mp28-column-evidence.md "
        "in one change.")
    assert set(_PUBLISHED_G3) | expected == set(matrix), (
        "every fixture must be either clean or carry a published residual")
    assert not set(_PUBLISHED_G3) & expected, (
        "a fixture is published both clean and with a residual: "
        f"{sorted(set(_PUBLISHED_G3) & expected)}")
    assert len(clean) == 21 and len(matrix) == 22, (len(clean), len(matrix))
    aero = sorted(name for name in clean if name.startswith("aero-"))
    assert len(aero) == 18, aero


# ---------------------------------------------------------------------------
# 2.  The evidence document cannot drift from the measurement.
# ---------------------------------------------------------------------------

_DOC_ROW = re.compile(
    r"^\|\s*`(aero-[a-z0-9-]+)`\s*\|\s*(PASS|MISS)\s*\|\s*"
    r"`?([A-Za-z0-9_.-]+)`?\s*\|\s*([0-9.eE+-]+)\s*\|", re.MULTILINE)

#: A row of the document's per-card-class table: the same row led by the
#: card class it was measured on (``| sm_89 | `aero-...` | PASS | ...``).
_DOC_CARD_ROW = re.compile(
    r"^\|\s*sm_([0-9]+)\s*\|\s*`(aero-[a-z0-9-]+)`\s*\|\s*(PASS|MISS)"
    r"\s*\|\s*`?([A-Za-z0-9_.-]+)`?\s*\|\s*([0-9.eE+-]+)\s*\|",
    re.MULTILINE)

#: THE DOCUMENT'S NUMBERS ARE PER CARD CLASS.  Its Class A table is the
#: sm_120 measurement (an RTX 5090 and an RTX 5070 Ti read it alike); every
#: other measured class publishes the rows that read differently on it, in
#: the per-card table.  A card class in neither is held to the verdicts
#: only, and the test says so.
_DOCUMENT_CARD_CLASS = "12.0"
_MEASURED_CARD_CLASSES = ("12.0", "8.9")


def _document_rows(card_class: str | None = None):
    """The document's Class A rows, as published for ``card_class``.

    ``None`` or the document's own class returns the Class A table; another
    measured class returns it with that class's per-card rows laid over it.
    """
    assert EVIDENCE_DOC.exists(), (
        f"missing {EVIDENCE_DOC}; WP-12b owns it and the campaign is not "
        "evidenced without it")
    text = EVIDENCE_DOC.read_text(encoding="utf-8")
    rows = {m.group(1): (m.group(2), m.group(3), float(m.group(4)))
            for m in _DOC_ROW.finditer(text)}
    if card_class is not None and card_class != _DOCUMENT_CARD_CLASS:
        tag = card_class.replace(".", "")
        for m in _DOC_CARD_ROW.finditer(text):
            if m.group(1) == tag:
                rows[m.group(2)] = (m.group(3), m.group(4),
                                    float(m.group(5)))
    return text, rows


@requires_gpu
def test_the_evidence_document_publishes_all_nineteen_measured_fixtures():
    """Doc/measurement lockstep, fixture by fixture.

    For each of the nineteen scenarios the document's Class A table covers:
    the document must name it, say whether it clears the gate, name its WORST
    field and give that field's measured residual.  ``rtol=0.05`` is the
    printing tolerance of a 4-significant-figure table, not a physics
    tolerance.

    THE THREE FIXTURES THE DOCUMENT DOES NOT COVER ARE NAMED, NOT DROPPED.
    :data:`_FIXTURES_OUTSIDE_THE_DOCUMENT` is asserted to be exactly the
    complement of the documented set, so the partition is complete and a
    fourth un-documented fixture fails here.  Those three are still fully
    measured and ratcheted by ``_PUBLISHED_G3`` above; what they are outside
    is the DOCUMENT, not the gate.
    """
    import cupy as cp

    matrix = _g3_matrix(cp)
    card = _adapter_module()._card_class(cp)
    measured_class = card in _MEASURED_CARD_CLASSES
    _text, rows = _document_rows(card if measured_class else None)
    if not measured_class:
        print(f"card class sm_{card.replace('.', '')} has no published row: "
              "the document's verdicts are asserted, its numbers are not "
              f"(measured classes: {list(_MEASURED_CARD_CLASSES)})")

    documented = set(matrix) - set(_FIXTURES_OUTSIDE_THE_DOCUMENT)
    assert documented == {name for name in matrix
                          if name.startswith("aero-")}, sorted(documented)
    assert len(documented) == 19, sorted(documented)

    missing = documented - set(rows)
    assert not missing, (
        f"the evidence document does not carry a row for {sorted(missing)}")
    extra = set(rows) - documented
    assert not extra, f"the document carries unknown fixtures {sorted(extra)}"

    problems = []
    for scenario, row in sorted(
            (name, matrix[name]) for name in documented):
        verdict, field, published = rows[scenario]
        clears = all(value <= G3_GATE for value in row.values())
        if clears != (verdict == "PASS"):
            problems.append(
                f"{scenario}: document says {verdict}, measurement "
                f"{'clears' if clears else 'misses'} {G3_GATE:.0e}")
            continue
        if not measured_class:
            continue
        worst_field = max(row, key=lambda name: row[name])
        worst = row[worst_field]
        if worst == 0.0:
            # An all-zero row has no "worst field"; the document says "-".
            if field != "-" or published != 0.0:
                problems.append(
                    f"{scenario}: every compared field is bit-identical to "
                    f"the fixture, so the document must record '-' and 0.0, "
                    f"not {field} / {published}")
            continue
        if field != worst_field:
            problems.append(
                f"{scenario}: document names {field} as the worst field, "
                f"measurement says {worst_field} ({worst:.3e})")
            continue
        if not published == pytest.approx(worst, rel=0.05):
            problems.append(
                f"{scenario}.{field}: document says {published:.3e}, "
                f"measurement says {worst:.3e}")
    assert not problems, "\n".join(problems)


def test_the_evidence_document_states_the_maturity_it_may_claim():
    """The promotion criterion, written where it cannot drift.

    MP28_PORT_SPEC.md: thompson-aerosol-mp28 may reach
    'implemented-unverified' when tiers 1-3 pass and the measured distances
    are published here; it may NOT claim 'wrf-matched-run-candidate' without a
    ratified reference comparison, nor 'wrf-matched-run' without a matched
    multi-hour ArWen-vs-WRF forecast.  A public evidence document that omits
    the ceiling on its own claim is the exact failure this port is trying to
    avoid.
    """
    text, _rows = _document_rows()
    lowered = text.lower()
    assert "implemented-unverified" in lowered
    # Both spellings: the registry's current labels and the older ones that
    # 2.8.3 and earlier records still carry (read as aliases), so a reader
    # holding either finds the ceiling stated.
    for forbidden in ("wrf-matched-run", "wrf-matched-run-candidate",
                      "model-validated", "validation-candidate"):
        assert forbidden in lowered, (
            f"the document must say explicitly that it does NOT claim "
            f"'{forbidden}'")
    assert "not " in lowered


# ---------------------------------------------------------------------------
# 3.  No new silent skip.
# ---------------------------------------------------------------------------

#: Every mp=28-owned test module.  A module that leaves this list also leaves
#: the skip census, so the list is asserted against the tree.
_SUITE_MODULES = (
    "test_kernel_loader_inert.py",
    "test_mp28_forecast_smoke.py",
    "test_mp28_runnable.py",
    "test_mp8_frozen.py",
    "test_thompson_aerosol_adapter.py",
    "test_thompson_aerosol_cold_gpu.py",
    "test_thompson_aerosol_contract.py",
    "test_thompson_aerosol_device_helpers.py",
    "test_thompson_aerosol_gpu.py",
    "test_thompson_aerosol_sat_gpu.py",
    "test_thompson_aerosol_sed_gpu.py",
    "test_thompson_aerosol_state_gpu.py",
    "test_thompson_aerosol_warm_gpu.py",
)

#: The COMPLETE audited census of every skip/xfail/importorskip site in the
#: mp=28 suite, as ``(module, enclosing def, kind)``.  Line numbers are
#: deliberately absent: they churn on every edit and would make this a
#: nuisance instead of a gate.  ``@requires_gpu`` is excluded -- it is the
#: repository-wide device marker applied by conftest and audited by
#: tests/test_gpu_marker_discipline.py, not an mp=28 claim.
#:
#: Each entry is explained in the skip census of
#: docs/public/wrf-comparison/mp28-column-evidence.md.  ADD ONE HERE AND YOU MUST
#: ADD IT THERE: the point of this gate is that no skip is invisible.
_SKIP_SITES = frozenset({
    # Structural: the six aerosol translation units are allow-listed to
    # receive thompson_aerosol_common.cuh, so "source is byte-identical to
    # preamble + file" is FALSE for them by construction.  12 runtime skips
    # (6 modules x 2 parametrized tests); not an unmeasured claim -- the
    # positive property is asserted by the same file's own aerosol cases.
    ("test_kernel_loader_inert.py",
     "test_non_aerosol_module_source_is_byte_identical", "call:skip"),
    ("test_kernel_loader_inert.py",
     "test_non_aerosol_int_define_source_is_byte_identical", "call:skip"),
    # Opt-in empirical mp=8 gate: needs gfortran, the pristine WRF tree and
    # ~380 MB of regenerated tables.  WP-12b RAN it; see the document.
    # One launcher-shape test per arithmetic leg (2.9.1 intake of the WRF
    # exact chain): a default process runs 2.9.1's mp=8 adapter, which
    # still calls the reused thompson.cu launchers, so _default compares
    # live; under GPUWM_WRF_EXACT=1 mp=8 is the classic unit and _strict
    # holds the pinned shape.  Exactly one of the two runs on every leg.
    ("test_thompson_aerosol_adapter.py",
     "test_reused_classic_launchers_receive_the_mp8_argument_shape_default",
     "marker:skipif"),
    ("test_thompson_aerosol_adapter.py",
     "test_reused_classic_launchers_receive_the_mp8_argument_shape_strict",
     "marker:skipif"),
    # Renamed by the WRF exact chain (2.9.1 intake): it reproduces the
    # committed oracle on the receipt's libc, bounded on another.
    ("test_mp8_frozen.py",
     "test_clean_oracle_rebuild_reproduces_the_committed_oracle",
     "marker:skipif"),
    # CCN_ACTIVATE.BIN ships as of 2026-08-01, so on a clean checkout these
    # do NOT fire.  They are kept as defence: a tree where the asset was
    # deleted or an override points elsewhere must skip by name rather than
    # fail obscurely.
    ("test_mp28_forecast_smoke.py", "_tables_or_skip", "call:skip"),
    ("test_mp28_runnable.py", "_tables_or_skip", "call:skip"),
    ("test_thompson_aerosol_adapter.py", "_tables_or_skip", "call:skip"),
    ("test_thompson_aerosol_gpu.py", "_tables_or_skip", "call:skip"),
    ("test_thompson_aerosol_device_helpers.py",
     "test_activ_ncloud_matches_the_wrf_probe_exactly", "marker:skipif"),
    ("test_thompson_aerosol_device_helpers.py",
     "test_activ_ncloud_never_activates_more_than_the_available_aerosol",
     "marker:skipif"),
    ("test_thompson_aerosol_contract.py",
     "test_parsed_table_reproduces_the_fortran_activ_ncloud_probe",
     "call:skip"),
    # No CUDA device present.
    ("test_mp28_runnable.py",
     "test_microphysics_init_fills_wrfs_synthetic_ccn_profile_for_mp28",
     "call:skip"),
    ("test_mp28_runnable.py",
     "test_the_unfilled_aerosol_profile_is_physics_visible_not_cosmetic",
     "call:skip"),
    ("test_mp28_runnable.py",
     "test_refl_10cm_is_bit_identical_under_two_very_different_nc_fields",
     "call:skip"),
    ("test_mp28_runnable.py",
     "test_mp28_echo_is_its_own_transcription_and_mp8s_stays_frozen",
     "call:skip"),
    ("test_mp28_runnable.py",
     "test_microphysics_apply_runs_mp28_end_to_end_on_a_real_domainstate",
     "call:skip"),
    ("test_mp28_runnable.py",
     "test_mp28_specified_zone_ring_is_bit_restored_including_the_aerosols",
     "call:skip"),
    ("test_mp28_runnable.py",
     "test_a_real_mp28_state_reaches_the_file_through_state_frame",
     "call:skip"),
    ("test_mp28_runnable.py",
     "test_an_output_due_mp28_call_produces_a_real_radar_field", "call:skip"),
    ("test_mp28_runnable.py", "_refl_oracle_harness", "call:skip"),
    ("test_thompson_aerosol_adapter.py", "_require_device", "call:skip"),
    ("test_thompson_aerosol_cold_gpu.py", "classic_tables", "call:skip"),
    ("test_thompson_aerosol_contract.py",
     "test_aerosol_tables_upload_and_read_back_on_a_real_cuda_device",
     "call:importorskip"),
    ("test_thompson_aerosol_contract.py",
     "test_aerosol_tables_upload_and_read_back_on_a_real_cuda_device",
     "call:skip"),
    ("test_thompson_aerosol_sed_gpu.py", "<module>", "call:importorskip"),
    ("test_thompson_aerosol_sed_gpu.py", "_require_device", "call:skip"),
    # Fixture/oracle-availability skips inside individual gates.
    ("test_thompson_aerosol_sed_gpu.py",
     "test_embedded_seed_matches_the_committed_fixture", "call:skip"),
    # RETIRED, not relaxed.  These two carried
    #     if _LIBM is None: pytest.skip("libm.so.6 unavailable; ...")
    # because they need the glibc ``powf`` gfortran linked the oracle
    # against, and the suite reached it with ctypes.CDLL("libm.so.6") -- so
    # off a glibc host they did not run at all.  They now call
    # woof.core.noahmp_libm.powf, a bit-exact transcription of the same
    # glibc 2.39 e_powf.c, so the comparison is identical and available
    # everywhere.  The skips are gone because the condition is gone.
    ("test_thompson_aerosol_device_helpers.py",
     "test_local_reimplementations_of_shared_helpers_have_not_drifted",
     "call:skip"),
})


def _tables_or_skip():
    """See tests/test_mp28_forecast_smoke.py::_tables_or_skip.

    Present in this module so the device gates above name the missing asset
    instead of failing with a table-load traceback.  Enumerated in the skip
    census above and in the evidence document.
    """
    from woof.core.thompson_aerosol_contract import (
        MissingAerosolTableAsset, resolve_aerosol_table_root,
        resolve_ccn_activation_path)
    try:
        resolve_ccn_activation_path(None, resolve_aerosol_table_root(None))
    except MissingAerosolTableAsset as exc:                # pragma: no cover
        pytest.skip(f"CCN_ACTIVATE.BIN unavailable: {exc}")


def _skip_sites_in(path: pathlib.Path) -> set[tuple[str, str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner: dict[int, str] = {}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for inner in ast.walk(fn):
            owner.setdefault(id(inner), fn.name)

    found: set[tuple[str, str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else getattr(func, "id", ""))
            if name in ("skip", "importorskip", "xfail"):
                found.add((path.name, owner.get(id(node), "<module>"),
                           "call:" + name))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                target = deco.func if isinstance(deco, ast.Call) else deco
                attr = (target.attr if isinstance(target, ast.Attribute)
                        else getattr(target, "id", ""))
                if attr in ("skipif", "skip", "xfail"):
                    found.add((path.name, node.name, "marker:" + attr))
    return found


def test_the_mp28_suite_has_no_unaudited_skip_site():
    """A skip is an unmeasured claim.  This is the complete census.

    Static, over the test sources.  Fails when a skip, xfail or
    importorskip is ADDED anywhere in the mp=28 suite, and when one is
    removed without updating the census -- both directions matter, because
    the census is also what the public evidence document enumerates.
    """
    present = [name for name in _SUITE_MODULES if (_TESTS / name).is_file()]
    assert present == list(_SUITE_MODULES), (
        "an mp=28 test module disappeared: "
        f"{sorted(set(_SUITE_MODULES) - set(present))}")

    found: set[tuple[str, str, str]] = set()
    for name in _SUITE_MODULES:
        found |= _skip_sites_in(_TESTS / name)

    added = sorted(found - _SKIP_SITES)
    removed = sorted(_SKIP_SITES - found)
    assert not added, (
        "NEW skip site(s) in the mp=28 suite, none of them audited:\n  "
        + "\n  ".join(map(str, added))
        + "\nAdd each to _SKIP_SITES here AND to the skip census in "
          "docs/public/wrf-comparison/mp28-column-evidence.md, with the reason "
          "it is not an unmeasured claim.")
    assert not removed, (
        "skip site(s) gone; update the census and the document:\n  "
        + "\n  ".join(map(str, removed)))


def test_no_mp28_test_is_marked_xfail():
    """Port HARD RULE 7, asserted rather than trusted.

    ``xfail`` converts a red gate into a green one with no reader-visible
    difference.  Nothing in this port is allowed to use it; a residual that
    will not close is REPORTED as a measured number in the evidence
    document, which is what the document is for.
    """
    offenders = []
    for name in _SUITE_MODULES:
        for site in _skip_sites_in(_TESTS / name):
            if site[2].endswith("xfail"):
                offenders.append(site)
    assert not offenders, offenders


def test_the_evidence_document_enumerates_every_skip_site():
    """The census in the document must name every module that can skip."""
    text, _rows = _document_rows()
    modules = {site[0] for site in _SKIP_SITES}
    missing = sorted(name for name in modules if name not in text)
    assert not missing, (
        "the evidence document's skip census does not mention "
        f"{missing}; every module that can skip must be enumerated there")


# ---------------------------------------------------------------------------
# 4.  The mp=8 freeze, re-asserted at this package's tip.
# ---------------------------------------------------------------------------

def test_mp8_freeze_receipt_still_holds_at_the_wp12b_tip():
    """MP28_PORT_SPEC.md makes the WP-00 receipt WP-12's merge criterion.

    Three things, cheaply: thompson.cu's bytes, the EXACT source string cupy
    would compile for module ``thompson`` (which catches a preamble or
    CUDA_DEFINES change the file hash alone would not), and the classic
    table-asset set.  The full receipt has its own file; this is the
    tip-of-branch re-run the spec asks for, in the campaign file, so the
    campaign cannot be declared green while mp=8 has moved.
    """
    import test_mp8_frozen as frozen          # WP-00's pins, read-only

    from woof.core import kernels as kernel_loader
    from woof.core import thompson_contract

    cu = _ROOT / "woof" / "core" / "kernels" / "thompson.cu"
    digest = hashlib.sha256(cu.read_bytes()).hexdigest()
    assert digest == frozen.THOMPSON_CU_SHA256, (
        "woof/core/kernels/thompson.cu has changed; the entire mp=8 "
        "numerics guarantee rests on it being byte-frozen")

    assembled = kernel_loader.module_source("thompson")
    assert hashlib.sha256(assembled.encode("utf-8")).hexdigest() == \
        frozen.THOMPSON_COMPILED_SOURCE_SHA256, (
        "the assembled nvrtc source string for module 'thompson' changed "
        "even though thompson.cu did not -- a preamble or CUDA_DEFINES "
        "change would do this and would move mp=8's PTX")

    thompson_py = _ROOT / "woof" / "core" / "thompson.py"
    assert hashlib.sha256(thompson_py.read_bytes()).hexdigest() == \
        frozen.THOMPSON_PY_SHA256, "woof/core/thompson.py has changed"

    names = {asset.filename for asset in
             thompson_contract.CLASSIC_TABLE_ASSETS}
    assert "CCN_ACTIVATE.BIN" not in names, (
        "CCN_ACTIVATE.BIN entered the CLASSIC table set; every existing "
        "mp=8 launch would then fail closed on a file it never needed")
    assert thompson_contract.TABLE_SET_ID == frozen.TABLE_SET_ID_PIN, (
        "the classic Thompson TABLE_SET_ID moved; every mp=8 launch resolves "
        "its packaged tables through it")


# ---------------------------------------------------------------------------
# 5.  CCN_ACTIVATE.BIN ships, and still never joins the classic set.
# ---------------------------------------------------------------------------

def test_ccn_activate_bin_is_committed_and_manifested_but_not_classic():
    """Port HARD RULE 5, from inside the campaign, after its reversal.

    The asset is third-party parcel-model output redistributed by WRF and
    NOT generated by ``thompson_init``.  The port's v1 posture was that it
    stayed gitignored and unvendored; the licence question was answered on
    2026-08-01 (WRF's LICENSE.txt is a public-domain dedication and the
    committed copy is WRF v4.6.1's ``run/`` file bit for bit), and the
    decision was reversed by the owner.  The half of the rule that did NOT
    move is the half this test now exists for: the blob must be reachable
    from a clean checkout AND must never enter the classic mp=8 inventory,
    because that inventory is what every mp=8 launch validates.
    """
    import subprocess

    from woof.physics_compat import packaged_thompson_table_root

    manifest = packaged_thompson_table_root() / "MANIFEST.sha256"
    assert manifest.is_file()
    assert "CCN_ACTIVATE.BIN" in manifest.read_text(encoding="utf-8")

    # encoding named: git writes path names as UTF-8 regardless of the
    # console codepage, and text=True alone would decode them with the host
    # locale (cp1252 on Windows).
    tracked = subprocess.run(
        ["git", "ls-files", "--",
         "recast-woof-data/woof_data/data/thompson/tables"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(_ROOT))
    if tracked.returncode == 0 and tracked.stdout.strip():
        assert "CCN_ACTIVATE.BIN" in tracked.stdout, (
            "CCN_ACTIVATE.BIN is not tracked by git; a clean checkout then "
            "cannot run any mp=28 device gate")

    from woof.core import thompson_aerosol_contract as aerosol_contract
    from woof.core import thompson_contract as classic_contract

    assert aerosol_contract.AEROSOL_ASSET_REDISTRIBUTED is True
    assert "CCN_ACTIVATE.BIN" not in {
        asset.filename for asset in classic_contract.CLASSIC_TABLE_ASSETS}
    assert aerosol_contract.AEROSOL_TABLE_SET_ID != \
        classic_contract.TABLE_SET_ID, (
        "the aerosol table set must carry its OWN id so an mp=8 launch is "
        "never asked for the aerosol asset")
