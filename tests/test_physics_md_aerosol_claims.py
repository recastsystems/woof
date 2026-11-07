"""The mp=28 publication layer must be DERIVED from the gate, not retyped.

WHY THIS FILE EXISTS
--------------------
``docs/public/PHYSICS.md`` is the page a user reads before selecting
``mp_physics = 28``; ``docs/public/wrf-comparison/mp28-column-evidence.md``,
``PROVENANCE.md``, ``CHANGELOG.md`` and ``woof/physics_registry_v2.json``
are the four texts it points at.  Five waves of this port have shipped at
least one of those five quoting a residual, a count or a call graph the
code no longer produced.  It has happened every wave, it has been caught
every wave by a different auditor, and patching the literals again would
only reset the clock.

So this file does not check prose.  It DERIVES the publishable facts from
the objects that own them and asserts the documents match:

======================================  =====================================
fact                                    authority it is derived from
======================================  =====================================
is the synthetic aerosol profile        an AST-free source scan for callers
installed on a production run?          of ``microphysics_init`` in ``gpuwm/``
which fixtures clear a flat gate,       ``tests/test_thompson_aerosol_adapter
which clear it only allowanced,         .py``'s ``_G3_UNEXCEPTIONED_CLEAN``,
which miss, and by how much             ``_G3_GATED_CLEAN``, ``_G3_RESIDUALS``
the carve-out and its bound             the same file's ``_END_TO_END_BOUNDS``
what the aerosol initial condition      ``tests/test_mp28_forecast_smoke.py``'s
is worth to a forecast                  ``_forecast``, re-run on the device
what closed ``aero-ice-koop``           the committed fixture deck itself,
                                        re-inverted under both Exner constants
the profile values the page quotes      the ``aero-init-profile`` fixture
======================================  =====================================

Every one of those is two-directional.  A document may not understate the
port either: an evidence page that still calls a closed gap open is exactly
as false as one that calls an open gap closed, and it is the failure mode a
"the page must warn" test rots into the day the gap shuts.

THE PAIR THIS FILE USED TO CARRY, AND WHY IT WAS UNSATISFIABLE
--------------------------------------------------------------
Wave 4 wrote this module when ``microphysics_init`` had no production
caller.  ``PHYSICS.md`` printed a manual workaround --
``woof.core.microphysics.microphysics_init(state, cfg)`` -- and
``test_the_workaround_the_page_prints_is_real`` asserted the workaround was
importable.  Wave 5 WIRED the call
(``woof/core/physics.py::initialize_physics``), at which point
``test_physics_md_does_not_claim_the_synthetic_profile_is_installed``
started requiring that the page NOT contain ``microphysics_init(state,
cfg)`` -- a substring of the string the other test required.  No page
satisfied both.

Both tests are still here, both still fail in both directions, and neither
was weakened to make the pair satisfiable.  What changed is that the second
one now gates the REAL call site instead of a workaround for its absence --
it asserts strictly more than it did (the hook's signature and off-mp=28
no-op behaviour are still checked, plus the caller, plus that the caller is
unique, plus that the page prints no workaround while a caller exists), so
it is renamed to ``test_the_production_call_site_the_page_names_is_real``
to stop the name lying about what it covers.

NOT a style test.  Every assertion corresponds to a statement that was
false on some tree this port actually shipped.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
PHYSICS_MD = ROOT / "docs" / "public" / "PHYSICS.md"
EVIDENCE_MD = (ROOT / "docs" / "public" / "wrf-comparison"
               / "mp28-column-evidence.md")
PROVENANCE_MD = ROOT / "PROVENANCE.md"
CHANGELOG_MD = ROOT / "CHANGELOG.md"
REGISTRY_JSON = ROOT / "woof" / "physics_registry_v2.json"
BUILD_REGISTRY = ROOT / "tools" / "build_registry.py"

#: ``mp\w*\s*=?\s*28`` so that the sweep below sees "every run",
#: "every mp=28 run" and the registry's own "every mp_physics=28 run" alike.
_MP28 = r"(?:mp\w*\s*=?\s*28\s+)?"

#: Sentences that assert the synthetic profile IS installed on every run.
#: These were the shapes of the FALSE claim in waves 1-4 and are the shapes
#: of the TRUE claim now; the polarity is decided by
#: :func:`_production_callers_of_microphysics_init`, never by this tuple.
_ASSERTS_THE_PROFILE_IS_INSTALLED = (
    r"(?i)\bevery\s+" + _MP28 + r"run\s+takes\b",
    r"(?i)\bevery\s+" + _MP28 + r"run\s+(?:gets|uses|receives|is\s+given)\b",
    r"(?i)\bevery\s+" + _MP28 + r"run\s+starts\s+(?:from|with)\s+"
    r"(?:the\s+)?(?:\*?synthetic|`?thompson_init)",
    r"(?i)\bruns?\s+on\s+`?thompson_init`?'s\s+\*?synthetic",
)

#: The unambiguous, machine-checkable marker that a text claims the profile
#: is installed: it names the production caller.  Prose can be paraphrased;
#: a module-qualified call site cannot, and a text that claims installation
#: without naming who does it is not useful to a reader anyway.
_NAMES_THE_CALL_SITE = r"physics\.py::initialize_physics"

#: The manual workaround the page printed while the hook had no caller.
#: It must be absent exactly while a caller exists.
_MANUAL_WORKAROUND = "woof.core.microphysics.microphysics_init(state, cfg)"

#: The exact sentence fragment the page must carry while -- and only while --
#: ``woof/physics_registry_v2.json``'s own mp=28 warning disagrees with the
#: page about whether the profile is installed.  A marker, not prose
#: policing: it is what makes the disagreement a two-directional gate instead
#: of a one-way ratchet.
STALE_REGISTRY_NOTE = "the registry warning is not yet corrected"

#: The same device for the OTHER registry/page disagreement this tree had:
#: the ``aero-reduces-to-classic`` carve-out was tightened 25x before the
#: registry was regenerated, and while that window was open the page
#: published both numbers and said which was live.
STALE_CARVE_OUT_NOTE = "The registry has not been re-measured since that fix"


# ---------------------------------------------------------------------------
# The authorities.
# ---------------------------------------------------------------------------

def _read(path: pathlib.Path) -> str:
    assert path.is_file(), f"{path} is missing"
    return path.read_text(encoding="utf-8")


def _live(path: pathlib.Path) -> str:
    """The part of ``path`` that makes a claim about THIS tree.

    Every document is live in full except ``CHANGELOG.md``, whose released
    sections record what each release shipped and are history by
    construction: "18 of 22" under an older release heading was true of that
    release.  Its live part is ``## Unreleased`` when that carries entries,
    else the newest release section.  A lane cannot add changelog entries
    without a version bump (tests/test_release_version_declaration.py), so
    the counts are not REQUIRED in the changelog; when its live part quotes
    one, it must be the gate's.
    """
    text = _read(path)
    if path != CHANGELOG_MD:
        return text
    starts = [m.start() for m in re.finditer(r"(?m)^## ", text)]
    if not starts:
        return text
    sections = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    for section in sections:
        heading, _, body = section.partition("\n")
        if re.fullmatch(r"##\s+Unreleased\s*", heading, re.IGNORECASE) \
                and body.strip():
            return section
    for section in sections:
        heading = section.partition("\n")[0]
        if not re.fullmatch(r"##\s+Unreleased\s*", heading, re.IGNORECASE):
            return section
    return ""


_ADAPTER_MODULE = None


def _adapter():
    """``tests/test_thompson_aerosol_adapter.py`` as an importable module.

    It is the OWNER of the G3 partition: which fixtures are clean, which are
    clean only under an allowance, which miss and by how much.  Importing it
    costs nothing (it opens no device at import time) and it is what stops
    this file from becoming a second, drifting copy of those numbers.
    """
    global _ADAPTER_MODULE
    if _ADAPTER_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "_mp28_adapter_gate_docs",
            TESTS / "test_thompson_aerosol_adapter.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("_mp28_adapter_gate_docs", module)
        spec.loader.exec_module(module)
        _ADAPTER_MODULE = module
    return _ADAPTER_MODULE


def _registry_mp28_option() -> dict:
    registry = json.loads(_read(REGISTRY_JSON))

    def _find(node):
        if isinstance(node, dict):
            if node.get("selectors") == {"mp_physics": 28}:
                return node
            for value in node.values():
                found = _find(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = _find(value)
                if found is not None:
                    return found
        return None

    option = _find(registry)
    assert option is not None, "the registry has no mp_physics=28 option"
    return option


def _production_callers_of_microphysics_init() -> list[str]:
    """Every ``gpuwm/`` module that CALLS ``microphysics_init``.

    Deliberately the same scan as
    ``test_mp28_forecast_smoke.py``'s
    ``test_microphysics_init_has_a_production_call_site``
    so that the documentation gate and the code gate can never disagree
    about whether the hook is wired.  The definition site itself is not a
    caller: in ``microphysics.py`` only text BEFORE ``def microphysics_init``
    counts.

    POSIX-spelled, like every other path this suite reports: the caller is a
    location in the source tree, and ``core/physics.py`` is the same location
    whichever separator the host writes it with.  ``str()`` on the relative
    path would make this gate compare ``core\\physics.py`` against the
    ``core/physics.py`` its assertions name, and fail on Windows for a reason
    that has nothing to do with where the hook is called from.
    """
    root = ROOT / "woof"
    callers: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "microphysics.py" and "def microphysics_init" in text:
            body = text.split("def microphysics_init", 1)[0]
            if "microphysics_init(" in body:
                callers.append(path.relative_to(root).as_posix())
            continue
        if re.search(r"\bmicrophysics_init\s*\(", text):
            callers.append(path.relative_to(root).as_posix())
    return callers


def _section(text: str, start: str, end: str) -> str:
    """The slice of ``text`` from heading ``start`` up to heading ``end``."""
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def _section_6_1() -> str:
    """The evidence document's section 6.1, verbatim."""
    return _section(_read(EVIDENCE_MD), "### 6.1", "### 6.2")


#: How the adapter spells a compared quantity, and how a published table
#: does.  The gate compares 23 quantities; the documents publish the
#: registry's 16-field shape and drop the unit suffixes.
_FIELD_LABEL = {
    "nr_per_kg": "nr", "nc_per_kg": "nc", "ni_per_kg": "ni",
    "nwfa_per_kg": "nwfa", "nifa_per_kg": "nifa",
    "effc_m": "effc", "effi_m": "effi", "effs_m": "effs",
    "rainnc_mm": "rainnc", "rainncv_mm": "rainncv",
    "snownc_mm": "snownc", "snowncv_mm": "snowncv",
    "graupelnc_mm": "graupelnc", "graupelncv_mm": "graupelncv",
    "temp_k": "temp", "refl_dbz_db": "refl",
}


def _table_rows(text: str, header: str) -> dict[str, str]:
    """The markdown table that follows ``header``, keyed by fixture name.

    ``header`` must be present; a renamed header is a documentation change
    that has to come back through this file, which is the point.
    """
    assert header in text, (
        f"the published table headed {header!r} is gone; the counts and "
        "residuals this file derives are no longer findable")
    body = text[text.index(header) + len(header):]
    rows: dict[str, str] = {}
    # The table ends at its first non-table line.  It used to end at the
    # first non-table line AFTER A FIXTURE ROW, which read straight past an
    # EMPTY table (header and separator only) into whichever table came next
    # on the page; an empty miss table is the published state since the
    # 2.8.6 accumulator rework.  ``[1:]`` drops the rest of the header line.
    for line in body.splitlines()[1:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        match = re.match(r"\|\s*`([A-Za-z0-9_-]+)`\s*\|", stripped)
        if match:
            rows[match.group(1)] = stripped
    return rows


# ---------------------------------------------------------------------------
# 1. Does the page's story about the aerosol initial condition track the code?
# ---------------------------------------------------------------------------

def test_physics_md_does_not_claim_the_synthetic_profile_is_installed():
    """The page's claim must match whether the hook actually has a caller.

    The name is the wave-4 name and is kept so the history is traceable; the
    test has always been two-directional and what it asserts flips with the
    code.

    BEFORE THIS TEST: ``PHYSICS.md:87`` read "Every run takes
    ``thompson_init``'s *synthetic* CCN/IN profile" while
    ``_production_callers_of_microphysics_init()`` returned ``[]``.
    """
    page = _read(PHYSICS_MD)
    callers = _production_callers_of_microphysics_init()

    if callers:
        # THE GAP IS CLOSED.  The page must say so, name who closed it, and
        # stop describing a live run as starting from zero aerosol.  A test
        # that only forbade the old false sentence would let the page keep
        # warning about a defect that no longer exists -- the same failure in
        # the other direction.
        assert re.search(_NAMES_THE_CALL_SITE, page) is not None, (
            f"microphysics_init now has production callers ({callers}), so "
            "docs/public/PHYSICS.md must say the synthetic CCN/IN profile IS "
            "installed and name the caller "
            "(woof/core/physics.py::initialize_physics)")
        assert _MANUAL_WORKAROUND not in page, (
            "the page still prints the manual workaround "
            f"{_MANUAL_WORKAROUND!r} while initialize_physics already makes "
            "the call; following it would overwrite an initialised aerosol "
            "field with the synthetic profile a second time")
        required = {
            "the hook is named": "microphysics_init",
            "who calls it": "initialize_physics",
            "the WRF fill it implements": "module_mp_thompson.F:493-515",
            "the WRF clamps a run now starts ABOVE":
                "module_mp_thompson.F:3979-3982",
            "the code gate that pins the call site":
                "test_microphysics_init_has_a_production_call_site",
            "where the measurement lives":
                "wrf-comparison/mp28-column-evidence.md",
        }
        missing = sorted(what for what, token in required.items()
                         if token not in page)
        assert missing == [], (
            "the profile is installed but docs/public/PHYSICS.md does not "
            f"say the whole of it: {missing}")
        return

    # THE GAP IS OPEN.  Restore the wave-4 assertions exactly.
    for pattern in _ASSERTS_THE_PROFILE_IS_INSTALLED:
        match = re.search(pattern, page)
        assert match is None, (
            "docs/public/PHYSICS.md asserts that the synthetic CCN/IN "
            f"profile is installed ({match.group(0)!r}), but nothing in "
            "gpuwm/ calls microphysics_init, so every mp=28 run starts from "
            "nwfa = nifa = 0 and is clamped to WRF's floors "
            "(module_mp_thompson.F:3979-3982). See "
            "docs/public/wrf-comparison/mp28-column-evidence.md section 6.1")
    required = {
        "the hook is named": "microphysics_init",
        "the state it actually starts from": "nwfa = nifa = 0",
        "the WRF fill it is skipping": "module_mp_thompson.F:493-515",
        "the WRF clamp that hides it": "module_mp_thompson.F:3979-3982",
        "the code gate that pins the same fact":
            "test_gap_microphysics_init_has_no_production_call_site",
        "where the measurement lives":
            "wrf-comparison/mp28-column-evidence.md",
    }
    missing = sorted(what for what, token in required.items()
                     if token not in page)
    assert missing == [], (
        "docs/public/PHYSICS.md omits part of the largest measured error in "
        f"the port: {missing}")


def test_the_production_call_site_the_page_names_is_real():
    """The caller the page names must exist, be unique, and be the hook the
    page describes: ``(state, cfg)``, and the documented no-op away from 28.

    WAVE 4's VERSION OF THIS TEST, AND WHY IT IS GONE.  It was
    ``test_the_workaround_the_page_prints_is_real`` and it required
    ``PHYSICS.md`` to print
    ``woof.core.microphysics.microphysics_init(state, cfg)`` -- correct
    while nothing called the hook, and unsatisfiable the moment
    ``initialize_physics`` did, because the sibling test above then required
    the page NOT to contain ``microphysics_init(state, cfg)``.  Everything
    that version asserted is still asserted here (the hook exists, the
    signature is ``(state, cfg)``, it returns ``{}`` for mp_physics=8); what
    is added is the half that only became checkable once the call landed.

    BOTH DIRECTIONS.  With no production caller this test requires the
    workaround back, verbatim -- so a refactor that silently drops the call
    site cannot be papered over by deleting a sentence.
    """
    import inspect

    page = _read(PHYSICS_MD)
    callers = _production_callers_of_microphysics_init()

    from woof.core import microphysics

    signature = inspect.signature(microphysics.microphysics_init)
    assert list(signature.parameters) == ["state", "cfg"], (
        "microphysics_init's signature is not the (state, cfg) the "
        "publications describe")

    class _NotMp28:
        mp_physics = 8
        # MAD-WRF's cloud initialisation shares the hook (RunConfig defaults).
        madwrf_opt = 0
        madwrf_cldinit = 0

    receipt = microphysics.microphysics_init(object(), _NotMp28())
    assert receipt == {}, (
        "the publications say the hook refuses to act on any scheme other "
        f"than 28; it returned {receipt!r} for mp_physics=8")

    if not callers:
        assert _MANUAL_WORKAROUND in page, (
            "nothing in gpuwm/ calls microphysics_init, so the page must "
            "print the exact call a user needs, not a description of it")
        return

    # The caller is exactly one place.  Once-per-domain silently becoming
    # once-per-step would overwrite an advected, activated and scavenged
    # aerosol field with the synthetic profile, with no NaN and no warning.
    assert callers == ["core/physics.py"], (
        "microphysics_init's production callers are "
        f"{callers}; WRF calls thompson_init from mp_init and nowhere else, "
        "and the publications say the call is made once per domain")

    physics_source = _read(ROOT / "woof" / "core" / "physics.py")
    assert len(re.findall(r"\bmicrophysics_init\s*\(", physics_source)) == 1, (
        "woof/core/physics.py calls microphysics_init more than once")
    assert re.search(r"def\s+initialize_physics\b", physics_source), (
        "the page names woof/core/physics.py::initialize_physics as the "
        "caller and that function does not exist")

    # And the page must not be telling a reader to make the call themselves.
    assert _MANUAL_WORKAROUND not in page, (
        "the page prints a manual call to a hook initialize_physics already "
        "calls")


# ---------------------------------------------------------------------------
# 2. The G3 partition: derived from the gate, in both directions.
# ---------------------------------------------------------------------------

#: The two published tables this file derives, and the header each sits
#: under.  Keyed by the path so a failure names the file to fix.
_MISS_TABLES = (
    (PHYSICS_MD,
     "| fixture | quantities that miss, with the measured maximum "
     "relative difference |"),
    (EVIDENCE_MD, "| fixture | fields above 2.0e-6 |"),
)


def test_the_published_clean_counts_are_the_gates_own_counts():
    """"15 of 22" and "16 of 22" are computed here, not retyped.

    ``_G3_UNEXCEPTIONED_CLEAN`` (a flat gate, nothing held out),
    ``_G3_GATED_CLEAN`` (the three named allowances applied) and
    ``_FIXTURES`` are the adapter gate's own pinned data and are themselves
    re-measured on the device by
    ``test_the_g3_residual_ratchet_holds_in_both_directions``.  Every
    publication must quote what they say.

    BEFORE THIS TEST: the counts were four independent literals in
    PHYSICS.md, the evidence document, PROVENANCE.md and the registry, and
    two waves shipped with them disagreeing with the gate.
    """
    adapter = _adapter()
    total = len(adapter._FIXTURES)
    flat = len(adapter._G3_UNEXCEPTIONED_CLEAN)
    gated = len(adapter._G3_GATED_CLEAN)
    missing = len(adapter._G3_RESIDUALS)

    # The partition has to be a partition before anything is published.
    assert gated + missing == total, (
        f"{gated} gated-clean + {missing} residual fixtures != {total} "
        "fixtures; the gate's own partition is inconsistent")
    assert set(adapter._G3_RESIDUALS) & set(adapter._G3_GATED_CLEAN) == set()

    flat_count = f"{flat} of {total}"
    gated_count = f"{gated} of {total}"

    for path in (PHYSICS_MD, EVIDENCE_MD, PROVENANCE_MD):
        text = _read(path)
        assert flat_count in text, (
            f"{path.name} does not publish the gate's unexceptioned clean "
            f"count ({flat_count})")
        assert gated_count in text, (
            f"{path.name} does not publish the gate's clean-as-gated count "
            f"({gated_count})")

    # The one-line summary in PHYSICS.md's own microphysics table is the
    # sentence most readers will see, and it carried its own literals.
    page = _read(PHYSICS_MD)
    summary = next((line for line in page.splitlines()
                    if line.startswith("| Thompson aerosol-aware | 28 |")),
                   None)
    assert summary is not None, (
        "docs/public/PHYSICS.md's microphysics table has no mp=28 row")
    for phrase in (f"{flat} clear a flat", f"{missing} do not",
                   f"{len(adapter._G3_ALLOWANCE_ONLY_CLEAN)} clears only "
                   "under"):
        assert phrase in summary, (
            f"the mp=28 maturity row does not say {phrase!r}: {summary}")

    registry = " ".join(_registry_mp28_option()["warnings"])
    assert flat_count in registry, (
        f"the registry mp=28 warnings do not publish {flat_count}")

    evidence = _registry_mp28_option()["extensions"]["column_oracle_evidence"]
    published_clean = set(evidence["clean_fixtures"])
    measured_clean = set(adapter._G3_UNEXCEPTIONED_CLEAN)
    assert published_clean == measured_clean, (
        "the registry's clean_fixtures list is not the gate's "
        "_G3_UNEXCEPTIONED_CLEAN: only in registry "
        f"{sorted(published_clean - measured_clean)}, only in gate "
        f"{sorted(measured_clean - published_clean)}")


def test_every_published_residual_is_the_gates_own_number():
    """Every miss table is derived from ``_G3_RESIDUALS``, both ways.

    A fixture the gate misses must be listed with every quantity it misses
    on and every distinct value it misses by, at the ``%.3e`` precision the
    port publishes; a fixture the gate clears must not be listed at all,
    unless it is the one fixture that clears only through an allowance, and
    then its row must say so.  The clean half is the one that matters when a
    sibling package closes a residual: the closure has to reach the page or
    this goes red.

    BEFORE THIS TEST: the evidence document and PHYSICS.md both listed
    ``aero-ice-koop`` ``qi`` 1.612e-03 as an open residual for two waves
    after it stopped being one, and both collapsed
    ``rainnc``/``rainncv``/``sr`` -- three quantities the gate compares
    separately -- into a single unlabelled ``rainnc`` row.
    """
    adapter = _adapter()
    residuals = adapter._G3_RESIDUALS
    clean = set(adapter._G3_GATED_CLEAN)
    allowanced = set(adapter._G3_ALLOWANCE_ONLY_CLEAN)

    failures: list[str] = []
    for path, header in _MISS_TABLES:
        rows = _table_rows(_read(path), header)
        # The table IS the count.  Deriving the six-of-twenty-two figure
        # from the set of rows, rather than checking a literal "6", is what
        # makes a closed residual impossible to leave behind and an invented
        # one impossible to add.
        if set(rows) - allowanced != set(residuals):
            failures.append(
                f"{path.name}: the miss table lists "
                f"{sorted(set(rows) - allowanced)}; the gate misses "
                f"{sorted(residuals)}")
        for name in sorted(set(rows) & clean):
            if name in allowanced:
                # Publishing the allowanced fixture beside the misses is
                # informative, but it must be marked, not left to look like
                # a seventh failure.
                if "allowance" not in rows[name]:
                    failures.append(
                        f"{path.name}: lists {name} among the misses without "
                        "saying it clears the gate under an allowance")
                continue
            failures.append(
                f"{path.name}: lists {name} as missing the gate, but the "
                "gate now clears it")
        for name, fields in sorted(residuals.items()):
            row = rows.get(name)
            if row is None:
                failures.append(
                    f"{path.name}: does not publish the residual fixture "
                    f"{name}")
                continue
            for value in sorted({f"{v:.3e}" for v in fields.values()}):
                if value not in row:
                    failures.append(
                        f"{path.name}: {name}'s row does not carry the "
                        f"measured {value}")
            for field in fields:
                label = _FIELD_LABEL.get(field, field)
                if f"`{label}`" not in row:
                    failures.append(
                        f"{path.name}: {name}'s row does not name the "
                        f"quantity `{label}`")
    assert failures == [], "\n".join(failures)


#: Words that mark a number as HISTORY or as a COUNTERFACTUAL rather than as
#: a live claim about the tree.  The sweep below is deliberately narrow --
#: it looks only at blocks that name a fixture the gate CLEARS -- so this
#: vocabulary does not have to anticipate every sentence in the port; it has
#: to cover the ways the port talks about a residual that is gone.
_NOT_A_CURRENT_RESIDUAL_CLAIM = re.compile(
    r"(?i)\b(was|were|had been|used to|previously|superseded|until|pre-fix|old|"
    r"earlier|former|closed|no longer|published|removing|without|"
    r"returns? to|moved|fell|rose|now measures?|now measure)\b|->|→")


def test_no_publication_calls_a_clean_fixture_a_residual():
    """A fixture the gate clears may not still be published as missing it.

    This is the defect that has recurred every wave, in prose rather than in
    a table: the tables get re-measured and a paragraph three sections away
    keeps quoting the number that made the port look worse -- or, exactly as
    bad, keeps quoting one that made it look better.

    The sweep is scoped to blocks that NAME a fixture the gate clears on a
    flat 2.0e-6, and a block is allowed to carry an above-gate number only
    if it also says the number is history or a counterfactual
    (:data:`_NOT_A_CURRENT_RESIDUAL_CLAIM`).  That keeps §6.6's "remove the
    RSLF pin and aero-ice-koop returns to ..." legitimate while catching
    "aero-ice-koop measures qi 1.612e-03".
    """
    adapter = _adapter()
    gate = adapter._END_TO_END_DEFAULT_BOUND
    clean = tuple(adapter._G3_UNEXCEPTIONED_CLEAN)
    # Blocks naming a fixture that DOES miss, or the allowanced one, are
    # about that fixture; they are covered by the table gate above.
    noisy = tuple(adapter._G3_RESIDUALS) + tuple(
        adapter._G3_ALLOWANCE_ONLY_CLEAN)
    # The gate and the bounds themselves are numbers in this range and are
    # not residuals.  Comparing the parsed float, not the spelling, so
    # "2.0e-6" and "2.000e-06" are both recognised.
    thresholds = {gate, adapter._REFL_DB_GATE}
    thresholds |= {v for f in adapter._END_TO_END_BOUNDS.values()
                   for v in f.values()}
    thresholds |= set(adapter._REFL_DB_BOUNDS.values())

    failures = []
    for path in (PHYSICS_MD, EVIDENCE_MD, PROVENANCE_MD):
        text = _read(path)
        for block in re.split(r"\n\s*\n|\n(?=\|)", text):
            named = [name for name in clean if name in block]
            if not named or any(name in block for name in noisy):
                continue
            if _NOT_A_CURRENT_RESIDUAL_CLAIM.search(block):
                continue
            # Only sub-unit, negative-exponent values can be a relative
            # residual; the profile's 1.478987e+08 kg-1 is a concentration.
            for value in re.findall(r"\b\d\.\d+e-\d+\b", block):
                number = float(value)
                if number <= gate or number in thresholds:
                    continue
                failures.append(
                    f"{path.name}: {value} presented as a current "
                    f"measurement in a block naming {named}: "
                    f"{block.strip()[:160]!r}")
    assert failures == [], (
        "these fixtures clear the gate but are still published carrying a "
        "residual above it:\n  " + "\n  ".join(failures))


def test_the_page_publishes_the_carve_out_the_gate_actually_applies():
    """The carved-out fixture's numbers must be the ones on THIS tree.

    ``tests/test_thompson_aerosol_adapter.py`` owns the bound; the registry
    republishes a measurement taken against whatever bound was live when it
    was last regenerated.  While the two disagree the page must carry BOTH
    and say which is live, and when the registry is regenerated the page
    must stop calling it stale.

    BEFORE THIS TEST: the page published only ``qr`` 1.915e-03 / ``nr``
    1.922e-03 at a 2.5e-3 bound, while the gate on this tree applies
    1.0e-4 and the fixture measures 7.813e-05 / 4.832e-05.
    """
    page = _read(PHYSICS_MD)
    bounds = _adapter()._END_TO_END_BOUNDS
    published = _registry_mp28_option()["extensions"][
        "column_oracle_evidence"]["carved_out_bound"]

    # The page must name the live bound, whatever it is.  Both the padded
    # exponent ("1.0e-04") and the page's own trimmed spelling ("1.0e-4")
    # count; nothing else does, so the bound cannot be paraphrased away.
    for name, fields in bounds.items():
        assert f"`{name}`" in page, f"the page does not name {name}"
        for bound in sorted(set(fields.values())):
            padded = f"{bound:.1e}"
            trimmed = re.sub(r"e([+-])0(\d)", r"e\1\2", padded)
            assert padded in page or trimmed in page, (
                f"the page does not publish {name}'s live bound {padded}")

    registry_is_stale = any(
        published.get(name, {}).get(field, 0.0) > bound
        for name, fields in bounds.items()
        for field, bound in fields.items())

    if registry_is_stale:
        assert STALE_CARVE_OUT_NOTE in page, (
            "the registry publishes a carved-out residual larger than the "
            "bound the gate now applies, so the page must say in these "
            f"words that the registry is behind: {STALE_CARVE_OUT_NOTE!r}")
    else:
        assert STALE_CARVE_OUT_NOTE not in page, (
            "the registry has been re-measured, so the page must stop "
            "describing its carved-out numbers as stale")


#: Every gate constant this port has EVER used as a departure from the flat
#: G3 gate.  ``_G3_ALLOWANCES`` says which of them are live TODAY; the
#: difference between the two sets is the set of RETIRED allowances, and a
#: page that keeps printing a retired one as live is publishing a tolerance
#: the gate does not apply -- which reads as a weaker port than the tree is.
#:
#: This tuple only ever grows.  Removing a name from it would make a retired
#: allowance invisible to :func:`test_the_published_allowance_tables_name_
#: exactly_the_live_allowances`, which is the drift the whole tuple exists
#: to catch, so it is asserted to be a superset of the live set below.
_ALLOWANCE_CONSTANTS_EVER = (
    "_END_TO_END_BOUNDS",
    "_NEAR_CANCELLATION_LEVELS",
    "_REFL_DB_BOUNDS",
)

#: How many, spelled, because the published tables say "The two allowances"
#: rather than "2 allowances" and the count is the sentence a reader takes
#: away.
_SPELLED = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

#: A published allowance row that has been marked retired.  Either spelling
#: counts; what must not happen is the row surviving with no marker at all.
_RETIRED_MARKER = re.compile(r"(?i)\bretired\b|~~")


def _allowance_rows(text: str) -> dict[str, str]:
    """Every markdown table row that names an allowance gate constant."""
    rows: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        for name in _ALLOWANCE_CONSTANTS_EVER:
            if f"`{name}`" in stripped:
                rows[name] = stripped
    return rows


def test_the_published_allowance_tables_name_exactly_the_live_allowances():
    """The allowance tables are derived from ``_G3_ALLOWANCES``, both ways.

    An allowance the gate applies must appear as a live row; an allowance
    the gate has RETIRED must either be gone or be marked retired, and the
    spelled count in the heading above the table must be the gate's own
    count.

    BEFORE THIS TEST, and this is the fourth wave the same class of drift
    landed: ``_REFL_DB_BOUNDS`` was retired by the gate -- the reflectivity
    residual it existed for fell to 3.242e-05 dB, inside the flat 2.0e-4 dB
    gate, and the constant is now an empty dict -- while ``PHYSICS.md`` and
    the evidence document both still printed it as a live "1.0e-3 dB, 10x
    stricter" row under a heading that said "The three allowances".  Nothing
    in the tree could see that: the existing carve-out gate checks the BOUND
    VALUE the page prints, not the SET of allowances it claims to apply.
    """
    adapter = _adapter()
    live = {name for _label, name, _fixtures, _why in adapter._G3_ALLOWANCES}
    assert live <= set(_ALLOWANCE_CONSTANTS_EVER), (
        "the gate applies an allowance this file has never heard of: "
        f"{sorted(live - set(_ALLOWANCE_CONSTANTS_EVER))}.  Add it to "
        "_ALLOWANCE_CONSTANTS_EVER so a future retirement is visible here.")
    retired = set(_ALLOWANCE_CONSTANTS_EVER) - live

    failures: list[str] = []
    for path in (PHYSICS_MD, EVIDENCE_MD):
        text = _read(path)
        # No live allowance is published as "No allowance remains", not as
        # "The no allowance".
        heading = (f"The {_SPELLED[len(live)]} allowance" if live
                   else "No allowance remains")
        if heading not in text:
            failures.append(
                f"{path.name}: does not say {heading!r}; the gate applies "
                f"{len(live)} allowance(s)")
        rows = _allowance_rows(text)
        for name in sorted(live):
            row = rows.get(name)
            if row is None:
                failures.append(
                    f"{path.name}: publishes no row for the live allowance "
                    f"{name}")
            elif _RETIRED_MARKER.search(row):
                failures.append(
                    f"{path.name}: marks {name} retired, but the gate still "
                    "applies it")
        for name in sorted(retired):
            row = rows.get(name)
            if row is not None and not _RETIRED_MARKER.search(row):
                failures.append(
                    f"{path.name}: still publishes the RETIRED allowance "
                    f"{name} as live: {row[:120]!r}")
    assert failures == [], "\n".join(failures)


def test_the_published_clean_fixture_list_is_the_gates_own_list():
    """The named clean set, and the aero-only subtotal, are derived.

    ``test_the_published_clean_counts_are_the_gates_own_counts`` checks the
    NUMBER.  This checks the LIST and the subtotal beside it, which drifted
    independently: the page carried "15 of 22" and a fifteen-name list, and
    when the gate closed two fixtures the number and the names had to move
    together and neither did.

    BEFORE THIS TEST: ``PHYSICS.md``'s clean row omitted ``aero-drop-evap``
    and ``aero-ice-demott-idxin`` -- both of which clear the flat gate on all
    23 quantities -- and said "14 of the 19 spec'd", where the gate says 16.
    """
    adapter = _adapter()
    clean = set(adapter._G3_UNEXCEPTIONED_CLEAN)
    spec = [name for name in adapter._FIXTURES if name.startswith("aero-")]
    aero_clean = [name for name in clean if name.startswith("aero-")]
    subtotal = f"{len(aero_clean)} of the {len(spec)} spec'd"

    page = _read(PHYSICS_MD)
    row = next((line.strip() for line in page.splitlines()
                if line.strip().startswith("| clear a **flat** gate")), None)
    assert row is not None, (
        "docs/public/PHYSICS.md has no flat-gate row in its result table; "
        "the clean list this test derives is no longer findable")
    named = {token for token in re.findall(r"`([A-Za-z0-9_*-]+)`", row)
             if token in set(adapter._FIXTURES)}
    assert named == clean, (
        "docs/public/PHYSICS.md's flat-gate row is not the gate's clean set. "
        f"Listed but not clean: {sorted(named - clean)}; clean but not "
        f"listed: {sorted(clean - named)}.")

    failures = [path.name for path in (PHYSICS_MD, EVIDENCE_MD)
                if subtotal not in _read(path)]
    assert failures == [], (
        f"{failures} do not publish the gate's aero-only subtotal "
        f"({subtotal!r})")


def test_no_publication_states_a_clean_count_the_gate_does_not_produce():
    """Any "N of 22" is the gate's own N, or is marked history.

    The counts live in prose as well as in tables -- a counterfactual in
    section 6.6, a "what moved" paragraph, a changelog bullet -- and the
    prose is where they rot.  Every ``N of <deck size>`` literal in the four
    documents must be one of the gate's two counts, unless its block is
    explicitly historical or counterfactual, which is the same vocabulary
    :data:`_NOT_A_CURRENT_RESIDUAL_CLAIM` already defines.

    BEFORE THIS TEST: "15 of 22" and "16 of 22" appeared as live claims in
    the result table of ``PHYSICS.md``, in the evidence document's result
    sentence, in ``PROVENANCE.md`` D9k and in ``CHANGELOG.md``, three waves
    running, while the gate produced 17 and 18.  ``CHANGELOG.md``'s open-items
    list additionally said "Six of 22 fixtures miss", spelled out, where the
    gate misses four -- which is why the sweep reads the spelled forms too.
    """
    # The pluperfect is history too.  CHANGELOG.md's 2.7.6 section says "9 of
    # 22 profiles had been refused on the namelist route", a count of physics
    # profiles a fix closed; the vocabulary knew "was" and "were" but not
    # "had been", so this sweep read it as a live fixture claim and stage 1
    # stayed red on text that claims nothing about the deck.  A bare count
    # still reads as live.
    assert _NOT_A_CURRENT_RESIDUAL_CLAIM.search(
        "9 of 22 profiles had been refused on the namelist route")
    assert not _NOT_A_CURRENT_RESIDUAL_CLAIM.search(
        "17 of 22 WRF column fixtures clear the flat gate")
    adapter = _adapter()
    total = len(adapter._FIXTURES)
    # The three counts the gate's own partition produces, and nothing else:
    # unexceptioned clean, clean as gated, and the number that miss.
    allowed = {len(adapter._G3_UNEXCEPTIONED_CLEAN),
               len(adapter._G3_GATED_CLEAN),
               len(adapter._G3_RESIDUALS)}
    spelled = {word: value for value, word in _SPELLED.items()}
    spelled.update({"six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                    "eleven": 11, "twelve": 12, "thirteen": 13,
                    "fourteen": 14, "fifteen": 15, "sixteen": 16,
                    "seventeen": 17, "eighteen": 18, "nineteen": 19,
                    "twenty": 20})
    words = "|".join(sorted(spelled, key=len, reverse=True))
    pattern = re.compile(rf"\b(\d+|{words}) of (?:the )?{total}\b", re.I)

    failures: list[str] = []
    for path in (PHYSICS_MD, EVIDENCE_MD, PROVENANCE_MD, CHANGELOG_MD):
        for block in re.split(r"\n\s*\n|\n(?=\|)", _live(path)):
            found = {int(match) if match.isdigit() else spelled[match.lower()]
                     for match in pattern.findall(block)}
            if not found - allowed:
                continue
            if _NOT_A_CURRENT_RESIDUAL_CLAIM.search(block):
                continue
            failures.append(
                f"{path.name}: states {sorted(found - allowed)} of {total} "
                f"as a live count where the gate produces {sorted(allowed)}: "
                f"{block.strip()[:160]!r}")
    assert failures == [], "\n".join(failures)


def test_no_miss_table_row_names_a_quantity_the_gate_says_agrees():
    """A residual row may not carry a quantity that is now inside the gate.

    ``test_every_published_residual_is_the_gates_own_number`` asserts the
    row carries every quantity the gate says misses.  This asserts the other
    direction, which is the one that goes stale when a sibling package
    closes a residual: a row may not name a quantity the gate no longer
    reports for that fixture.

    The one fixture that clears only under an allowance is out of scope by
    construction -- its row is explicitly about what the FLAT gate would say
    -- and it is named here rather than filtered silently.

    BEFORE THIS TEST: ``aero-cloud-freeze-nc``'s row in both published miss
    tables carried ``qr``, ``nr``, ``rainnc``, ``rainncv`` and ``sr``
    alongside its ``qc``, months after the sedimentation-density fix took
    all five inside the gate (the surface three to bitwise equality).
    """
    adapter = _adapter()
    residuals = adapter._G3_RESIDUALS
    allowanced = set(adapter._G3_ALLOWANCE_ONLY_CLEAN)
    quantities = {_FIELD_LABEL.get(field, field)
                  for field in (tuple(adapter._END_TO_END_FIELDS)
                                + tuple(adapter._END_TO_END_SURFACE_FIELDS)
                                + (adapter._REFL_FIELD,))}

    failures: list[str] = []
    for path, header in _MISS_TABLES:
        rows = _table_rows(_read(path), header)
        for name, row in sorted(rows.items()):
            if name in allowanced or name not in residuals:
                continue
            expected = {_FIELD_LABEL.get(field, field)
                        for field in residuals[name]}
            named = {token for token in re.findall(r"`([A-Za-z0-9_]+)`", row)
                     if token in quantities}
            extra = named - expected
            if extra:
                failures.append(
                    f"{path.name}: {name}'s row names {sorted(extra)}, which "
                    f"the gate reports inside the flat gate; it misses only "
                    f"on {sorted(expected)}")
    assert failures == [], "\n".join(failures)


def test_the_published_maturity_label_is_the_registrys_own():
    """The page's maturity claim is read out of the registry, not asserted.

    PHYSICS.md's own opening paragraph tells the reader the registry is the
    authority.  If the two ever disagree, the page is a false statement
    about its own authority.
    """
    label = _registry_mp28_option()["maturity"]
    page = _read(PHYSICS_MD)
    evidence = _read(EVIDENCE_MD)
    assert label == "implemented-unverified", (
        f"the registry now publishes mp=28 at maturity {label!r}; every "
        "publication's headline claim has to be re-derived")
    assert f"| 28 | {label} |" in page, (
        f"the PHYSICS.md microphysics table does not carry mp=28 at {label}")
    assert f"**`{label}`**" in evidence, (
        "the evidence document does not open on the registry's own label")


# ---------------------------------------------------------------------------
# 3. What closed aero-ice-koop.  It was the measuring stick, not the kernel.
# ---------------------------------------------------------------------------

#: The residual ``aero-ice-koop`` carried while it was published, by four
#: waves, as the port's largest genuine physics gap.
KOOP_SUPERSEDED = ("1.612e-03", "1.764e-03", "5.093e-05")

#: What a text must carry if it mentions that closure at all.  These are the
#: cause, not a description of the effect: the oracle harness's Exner
#: constant.  A port that says "we fixed our measuring stick" is more
#: trustworthy than one that claims a physics fix it did not make.
KOOP_ATTRIBUTION = ("run_column_aero.F90", "287.0/1004.0")

#: Claims of a kernel/physics closure that are FALSE for this fixture.
#: The negative lookbehind is not a loophole: a text is allowed -- and this
#: package's texts do -- to say "it was NOT closed by a kernel".
KOOP_FALSE_ATTRIBUTIONS = (
    r"(?i)WP-0?6\s+closed\s+it",
    r"(?i)homogeneous haze freezing is no longer a gap",
    r"(?i)(?<!not )\bclosed by (?:a|the) (?:kernel|physics|cold[- ]network)",
)


def test_the_koop_closure_is_attributed_to_the_oracle_not_to_a_kernel():
    """``aero-ice-koop`` was closed by correcting the HARNESS, not the port.

    ``tools/thompson_wrf461_oracle/run_column_aero.F90`` built the Exner
    function with ``rd_over_cp = 287.0/1004.0``.  WRF's own ``rcp`` is
    ``r_d/cp`` with ``r_d = 287.`` and ``cp = 7.*r_d/2.`` (=1004.5), i.e.
    exactly ``2/7`` -- ``share/module_model_constants.F:19,:20,:31``.  The
    fixtures were therefore recorded with a ``(p, T)`` pair the port could
    not invert exactly, the adapter had to perturb the entry pressure to
    recover the recorded temperature, and the perturbed pressure drove
    genuinely different microphysics.  The port was being measured against a
    yardstick that was not WRF.

    Any publication that mentions the superseded numbers must say that, in
    those terms, and must not claim a kernel fix it did not make.  The
    MECHANISM is measured separately, next.
    """
    failures = []
    for path in (PHYSICS_MD, EVIDENCE_MD, PROVENANCE_MD, CHANGELOG_MD,
                 BUILD_REGISTRY):
        text = _read(path)
        if not any(value in text for value in KOOP_SUPERSEDED):
            continue
        missing = [token for token in KOOP_ATTRIBUTION if token not in text]
        if missing:
            failures.append(
                f"{path.name} publishes the superseded aero-ice-koop "
                f"residual without attributing the closure to the oracle "
                f"harness's Exner constant (missing {missing})")
        for pattern in KOOP_FALSE_ATTRIBUTIONS:
            match = re.search(pattern, text)
            if match is not None:
                failures.append(
                    f"{path.name} claims {match.group(0)!r}; the fixture was "
                    "closed by correcting run_column_aero.F90's Exner "
                    "constant, with no change to any kernel")

    registry = " ".join(_registry_mp28_option()["warnings"])
    if any(value in registry for value in KOOP_SUPERSEDED):
        missing = [token for token in KOOP_ATTRIBUTION
                   if token not in registry]
        assert not missing, (
            "the registry's mp=28 warnings publish the superseded "
            f"aero-ice-koop residual without the attribution ({missing})")

    assert failures == [], "\n".join(failures)


def test_the_committed_deck_carries_wrfs_own_exner_constant():
    """THE MECHANISM, MEASURED, on the committed fixtures themselves.

    Every fixture records ``p_pa`` and ``temp_k``.  ``run_column_aero.F90``
    hands ``mp_gt_driver`` the pair ``(theta, pii)`` with
    ``pii = (p/p0)**rd_over_cp``, so the adapter has to solve for a float32
    ``theta`` that reproduces ``temp_k`` bitwise under WOOF's own ``RCP``.

    Under WRF's ``r_d/cp`` that solve succeeds at EVERY level of EVERY
    fixture.  Under the superseded ``287.0/1004.0`` it fails at 47 of the
    528 entry levels (first published as 40 -- the count is the host
    libm's, re-pinned 2026-08-03; the evidence document carries the
    correction note) -- which is precisely why the adapter used to perturb
    the entry pressure, and why fixtures including ``aero-ice-koop`` were
    driven from a state WRF never produced.

    This is what makes the attribution above a measurement rather than a
    story: no kernel is involved, and the affected fixture set is read off
    the committed CSVs.
    """
    import csv
    import struct

    import numpy as np

    from woof.core import constants as c

    f32 = np.float32
    wrf_rcp = f32(f32(287.0) / f32(7.0 * 287.0 / 2.0))
    old_rcp = f32(f32(287.0) / f32(1004.0))

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]

    assert wrf_rcp == f32(2.0 / 7.0), (
        "WRF's r_d/cp is not 2/7 in float32; the attribution's premise is "
        "wrong")
    assert wrf_rcp == f32(c.RCP), (
        "woof's RCP is no longer WRF's r_d/cp, so the fixtures and the port "
        "no longer share an Exner function")
    assert bits(old_rcp) - bits(wrf_rcp) == 4774, (
        "the superseded constant is not 4774 ulps from WRF's; the published "
        "distance is wrong")

    from woof.core.noahmp_libm import powf

    def invertible(pressure, temperature, rcp):
        # The oracle's own ``(p/p0)**rd_over_cp`` is gfortran calling glibc
        # ``powf``, so the Exner values come from the glibc 2.39 transcription
        # element by element, not from NumPy's float32 ``power``: NumPy
        # 2.5's AVX-512 loop rounds 110 of these 1056 powers differently
        # and put the count below at 40 on an AVX-512 Linux machine.
        base = (pressure / f32(100000.0)).astype(np.float32)
        pii = np.array([powf(value, rcp) for value in base],
                       dtype=np.float32)
        theta = (temperature / pii).astype(np.float32)
        ok = (theta * pii).astype(np.float32) == temperature
        for candidate in (np.nextafter(theta, f32(np.inf), dtype=np.float32),
                          np.nextafter(theta, f32(-np.inf),
                                       dtype=np.float32)):
            ok |= (candidate * pii).astype(np.float32) == temperature
        return ok

    root = ROOT / "woof" / "data" / "thompson" / "oracle-aero"
    levels = 0
    broken: dict[str, int] = {}
    for path in sorted(root.glob("*-column.csv")):
        name = path.name[: -len("-column.csv")]
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row["phase"] == "before"]
        pressure = np.array([float(row["p_pa"]) for row in rows], np.float32)
        temperature = np.array([float(row["temp_k"]) for row in rows],
                               np.float32)
        levels += pressure.size
        assert bool(invertible(pressure, temperature, wrf_rcp).all()), (
            f"{name} does not invert exactly under WRF's r_d/cp; the "
            "committed deck was not generated with WRF's own Exner constant")
        bad = int((~invertible(pressure, temperature, old_rcp)).sum())
        if bad:
            broken[name] = bad

    assert levels == 528, levels
    # RE-PINNED 40 -> 47, owner-approved 2026-08-03, correction note in the
    # evidence document.  The deck bytes never changed and the essential
    # half of this test never moved: 528 of 528 levels invert under WRF's
    # own constant, asserted above.  The count under the SUPERSEDED constant
    # is a property of the host libm's float32 ``power``, not of the deck:
    # the original 2026-08-01 environment measured 40; every current
    # environment measures 47 with identical per-fixture counts (measured
    # 2026-08-03 on glibc 2.43 / numpy 2.5.1 and on MSVC / numpy 2.2.6).
    # 40 is NumPy 2.5's AVX-512 float32 ``power``; glibc's ``powf``, which
    # the oracle calls and which ``invertible`` now takes from the
    # transcription, gives 47 on every host.
    assert sum(broken.values()) == 47, (
        "the superseded Exner constant now breaks "
        f"{sum(broken.values())} of {levels} entry levels, not the "
        f"re-pinned 47: {broken}")
    assert "aero-ice-koop" in broken, (
        "aero-ice-koop is not among the fixtures the superseded constant "
        f"perturbs, so the published attribution is wrong: {sorted(broken)}")
    # The exact seven, with their level counts, as the evidence document
    # publishes them.  Two-directional: a fixture may neither join nor
    # leave this set without the document being re-derived.  The seven
    # NAMES are environment-invariant -- 40 and 47 land in the same
    # fixtures -- and that invariance is what the attribution rests on.
    assert broken == {
        "aero-cloud-freeze-nc": 8, "aero-cold-overlap": 8,
        "aero-ice-demott-dep": 8, "aero-ice-demott-idxin": 8,
        "aero-ice-koop": 6, "wp08-freeze": 6,
        "aero-reduces-to-classic": 3}, broken
    evidence = _read(EVIDENCE_MD)
    for name in broken:
        assert f"`{name}`" in evidence, (
            f"the evidence document does not name {name} among the fixtures "
            "the superseded Exner constant affects")


# ---------------------------------------------------------------------------
# 4. What the aerosol initial condition is worth -- bound to a live run.
# ---------------------------------------------------------------------------

def test_physics_md_publishes_the_measured_cost_verbatim():
    """Every number section 6.1 measured must appear on the page, spelled
    the same way.  One-directional between the two documents: the page may
    say more, never less, and may never round.

    BEFORE THIS TEST: PHYSICS.md contained none of these numbers.
    """
    page = _read(PHYSICS_MD)
    section = _section_6_1()

    # The measured quantities of section 6.1's table, as the evidence
    # document spells them.  The multiplication sign is left out of this
    # sweep on purpose -- the evidence document writes "5.6x" with U+00D7 --
    # and is asserted separately just below.
    numbers = sorted(set(re.findall(r"\d+\.\d+e[+-]\d+", section))
                     | set(re.findall(r"\d+\.\d+(?= mm)", section))
                     | {"+" + m + "%"
                        for m in re.findall(r"\+(\d+(?:\.\d+)?)%", section)})
    assert len(numbers) >= 8, (
        "section 6.1 of the evidence document no longer publishes the "
        f"measured table this page mirrors (found {numbers})")

    missing = [n for n in numbers if n not in page]
    assert missing == [], (
        "docs/public/PHYSICS.md does not republish what "
        "docs/public/wrf-comparison/mp28-column-evidence.md section 6.1 "
        f"measured: {missing}")

    ratio = re.search(r"(\d+\.\d+)\s*[x×]\s*fewer droplets", section)
    assert ratio is not None, (
        "section 6.1 no longer states the droplet-count ratio")
    assert re.search(ratio.group(1) + r"\s*[x×]\s*fewer droplets",
                     page) is not None, (
        f"the page does not republish the measured {ratio.group(1)}x fewer "
        "droplets")


def test_the_published_aerosol_sensitivity_is_a_live_measurement():
    """§6.1's table is RE-RUN here, not cross-referenced.

    This is the number the port advertises as the largest thing it has
    measured about its own behaviour, and it is the one that was published
    for two waves as the COST of a missing call site after the call site
    landed.  Every other gate on it compares one written-down number against
    another.  This one drives the two 150-step forecasts and rebuilds the
    table.

    Exact comparison at the published precision is legitimate rather than
    flaky: both forecasts were repeated on this device and every value was
    bit-identical across repeats.  If that stops holding the right response
    is to publish the spread, not to widen this.
    """
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:      # pragma: no cover
            pytest.skip("no CUDA device")
    except Exception:                                 # pragma: no cover
        pytest.skip("no CUDA device")

    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    spec = importlib.util.spec_from_file_location(
        "_mp28_forecast_smoke_docs", TESTS / "test_mp28_forecast_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_forecast_smoke_docs", smoke)
    spec.loader.exec_module(smoke)
    smoke._tables_or_skip()

    _cfg, filled = smoke._forecast(cp, bubble=True, initialise=True)
    _cfg2, stripped = smoke._forecast(cp, bubble=True, initialise=False)

    assert filled["init_receipt"] == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}, (
        "the control run is not the production init path, so this is not "
        "the comparison §6.1 publishes")
    assert stripped["init_receipt"] == {}
    assert float(stripped["nwfa_initial"].max()) == 0.0

    rain_filled = filled["rain_sum"][-1]
    rain_stripped = stripped["rain_sum"][-1]
    measured = {
        "initial mean nwfa": f"{float(filled['nwfa_initial'].mean()):.3e}",
        "final interior nwfa (filled)":
            f"{filled['nwfa_interior_mean'][-1]:.3e}",
        "final interior nwfa (stripped)":
            f"{stripped['nwfa_interior_mean'][-1]:.3e}",
        "domain-total RAINNC (filled)": f"{rain_filled:.3f}",
        "domain-total RAINNC (stripped)": f"{rain_stripped:.3f}",
        "peak RAINNC (filled)": f"{max(filled['rain_max']):.3f}",
        "peak RAINNC (stripped)": f"{max(stripped['rain_max']):.3f}",
        "peak nc (filled)": f"{max(filled['nc_max']):.3e}",
        "peak nc (stripped)": f"{max(stripped['nc_max']):.3e}",
        "rain excess": f"{rain_stripped / rain_filled - 1.0:+.1%}",
        "droplet ratio":
            f"{max(filled['nc_max']) / max(stripped['nc_max']):.1f}",
        "peak rain change": "{:+.1%}".format(
            max(stripped["rain_max"]) / max(filled["rain_max"]) - 1.0),
    }

    section = _section_6_1()
    page = _read(PHYSICS_MD)
    print("\n§6.1 RE-MEASURED FOR THE PUBLICATION GATE")
    for what, value in measured.items():
        print(f"  {what:32s} {value}")

    missing = {
        what: value for what, value in measured.items()
        if value not in section or value not in page}
    assert missing == {}, (
        "docs/public/wrf-comparison/mp28-column-evidence.md §6.1 and/or "
        "docs/public/PHYSICS.md publish an aerosol-sensitivity number this "
        f"run does not reproduce: {missing}.  Re-measure and republish; "
        "never round toward the published value.")

    # The direction of the sensitivity is part of the claim.
    assert rain_stripped > rain_filled, (
        "removing CCN reduced surface rain, which inverts the published "
        "direction of the sensitivity")
    assert max(stripped["nc_max"]) < max(filled["nc_max"])


#: The lateral-boundary depletion numbers ``PHYSICS.md`` republishes from
#: the evidence document's §5.1 table, and the metric each one is.  The page
#: rounds them differently from the table, so the check is "does the page's
#: spelling round-trip to the live measurement at the page's OWN precision",
#: not string equality -- that is what lets a page quote 19.8638 while the
#: table quotes 19.863753181374097 without either being unbound.
_PAGE_DEPLETION_CLAIMS = {
    "front_speed_ms": "20.0909",
    "front_speed_ratio": "1.00454",
    "nwfa_retained": "0.3314",
    "nifa_retained": "0.3272",
    "surface_emission_per_kg_s": "5540.14",
}


def test_the_page_republishes_the_measured_depletion_numbers():
    """``PHYSICS.md``'s lateral-boundary bullet is bound to the same live
    run the evidence document's §5.1 table is.

    ``test_mp28_forecast_smoke.py``'s
    ``test_the_depletion_measurement_is_published_in_the_evidence_document``
    binds the TABLE. Nothing bound the page, which restates the same numbers
    at different precisions and is the document a user actually reads.

    BEFORE THIS TEST: the page's 19.8638 / 0.4566 / 0.3363 / 5540.14 were
    hand-copied and nothing anywhere would have noticed them going stale.
    """
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:      # pragma: no cover
            pytest.skip("no CUDA device")
    except Exception:                                 # pragma: no cover
        pytest.skip("no CUDA device")

    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    spec = importlib.util.spec_from_file_location(
        "_mp28_forecast_smoke_docs", TESTS / "test_mp28_forecast_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_forecast_smoke_docs", smoke)
    spec.loader.exec_module(smoke)
    smoke._tables_or_skip()

    cfg, record = smoke._forecast(cp, bubble=False)
    metrics = smoke._depletion_metrics(cfg, record)

    page = _read(PHYSICS_MD)
    drift = []
    for metric, spelling in _PAGE_DEPLETION_CLAIMS.items():
        if spelling not in page:
            drift.append(f"the page no longer publishes {metric} as "
                         f"{spelling}")
            continue
        decimals = len(spelling.split(".")[1])
        if f"{metrics[metric]:.{decimals}f}" != spelling:
            drift.append(
                f"{metric}: the page says {spelling}, this run measures "
                f"{metrics[metric]:.{decimals}f}")
    assert drift == [], (
        "docs/public/PHYSICS.md's lateral-boundary numbers no longer match "
        "the run that produced them:\n  " + "\n  ".join(drift))


def test_the_published_profile_values_are_the_fixtures_own():
    """The profile numbers the page prints come from WRF, not from prose.

    The page quotes what ``thompson_init`` fills on the
    ``aero-init-profile`` fixture, and the ratio between that and the floor
    a run would be clamped to without it.  Both are read back out of the
    committed fixture here so the page cannot drift from the oracle -- and
    so the tempting round number (``naCCN1 + naCCN0`` = 350e6) can never
    creep back in: WRF's lowest level carries the first layer's thickness in
    the exponent (``module_mp_thompson.F:508``) and is nowhere near it.

    BEFORE THIS TEST: the page said the profile "runs from 350e6 at the
    lowest level", which the fixture contradicts -- it fills 1.478987e+08.
    """
    import csv

    page = _read(PHYSICS_MD)
    fixture = (ROOT / "woof" / "data" / "thompson" / "oracle-aero"
               / "aero-init-profile-column.csv")
    with fixture.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["phase"] == "before"]
    assert rows, "the aero-init-profile fixture has no 'before' rows"

    nwfa = [float(row["nwfa_per_kg"]) for row in rows]
    nifa = [float(row["nifa_per_kg"]) for row in rows]

    for label, value in (("nwfa bottom", nwfa[0]), ("nwfa top", nwfa[-1]),
                         ("nifa bottom", nifa[0]), ("nifa top", nifa[-1])):
        assert f"{value:.6e}" in page, (
            f"the page does not publish the fixture's {label} "
            f"({value:.6e})")

    # The bare ceiling must not be quoted as the lowest-level value.
    assert nwfa[0] < 3.5e8, "the fixture changed shape"
    assert "350e6 at the lowest" not in page, (
        "the page quotes naCCN1 + naCCN0 as the lowest-level value; WRF's "
        f":508 exponent puts it at {nwfa[0]:.6e} on this fixture")

    # And the clamp ratios the page uses to say how much the profile buys.
    ccn_ratio = nwfa[0] / 11.1e6            # module_mp_thompson.F:3979
    in_ratio = nifa[0] / 5.0e3              # :3981, naIN1*0.01
    assert f"{ccn_ratio:.1f}x" in page, (
        f"the page does not publish the CCN floor ratio {ccn_ratio:.1f}x")
    assert f"{in_ratio:.0f}x" in page, (
        f"the page does not publish the IN floor ratio {in_ratio:.0f}x")


# ---------------------------------------------------------------------------
# 5. The page against its siblings.
# ---------------------------------------------------------------------------

#: The sibling texts that carried the false "the profile is installed"
#: sentence when this file was written, mapped to the name the page uses for
#: each and to the regexes that detect the claim in that text's own wording.
#: Kept separate from :data:`_ASSERTS_THE_PROFILE_IS_INSTALLED` because these
#: are broader -- they must catch "runs it on", "always starts from the
#: synthetic profile" and "comes from thompson_init's synthetic ... profile",
#: phrasings PHYSICS.md must remain free to discuss in the negative.
#:
#: THE POLARITY IS THE CODE'S, NOT THIS TABLE'S.  While the hook had no
#: caller these sentences were false and the page had to name whoever still
#: carried them.  Now that the call is wired they are TRUE, so the page must
#: name nobody -- and the day the call site is dropped, the list comes back.
_SIBLING_TEXTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "CHANGELOG.md": ("CHANGELOG.md", (
        r"(?i)\bevery\s+" + _MP28 + r"run\s+takes\b",
    )),
    "PROVENANCE.md": ("PROVENANCE.md", (
        r"(?i)\bon\s+`?thompson_init`?'s\s+\*?synthetic",
        r"(?i)\balways\s+starts\s+from\s+the\s+\*?synthetic",
    )),
    "docs/public/CONFIGURATION.md": ("CONFIGURATION.md", (
        r"(?i)\bruns?\s+it\s+on\s+`?thompson_init`?'s\s+\*?synthetic",
    )),
    # RETIRED (development branch).  This row scanned the imported constant
    # woof.config.MP28_AEROSOL_SOURCE_DEVIATION for the sentence "comes
    # from thompson_init's synthetic profile".  The constant no longer
    # exists -- the default it described was flipped to WRF's monthly WIF
    # aerosol climatology, and fixing a defect retires its guards.  The
    # sentence it carried did not vanish; it moved to
    # MP28_AEROSOL_SYNTHETIC_FALLBACK, where it is TRUE of the branch it
    # names, so scanning it as a wrongness claim would now be wrong itself.
    # The fallback constant's own wording is pinned by
    # tests/test_mp28_runnable.py instead.
}


def _sibling_text(key: str) -> str:
    """The text to scan for ``key`` -- a file, or an imported constant."""
    # (The imported-constant branch retired with
    # MP28_AEROSOL_SOURCE_DEVIATION; every remaining key is a file.)
    return _read(ROOT / key)


def test_the_page_names_every_sibling_text_that_is_still_wrong():
    """The page lists the other published texts that contradict the code.

    That list must be exact in both directions.  A stale name is a second
    false statement; a missing name sends a reader to a document that will
    tell them the opposite of the measurement.  Neither is acceptable, and
    neither is detectable by reading one file.

    BEFORE THIS TEST: the page named none of them, because it agreed with
    them.
    """
    page = _read(PHYSICS_MD)
    marker = "have **not** caught up and still say the profile is"
    installed = bool(_production_callers_of_microphysics_init())

    disagreeing, agreeing = [], []
    for key, (name, patterns) in _SIBLING_TEXTS.items():
        text = _sibling_text(key)
        # Two independent detectors, because the two errors are different
        # sentences.  ``patterns`` are the wave-1..4 shapes of "the profile
        # is installed" -- FALSE then, TRUE now.  The call-site marker is a
        # claim that a specific function makes the call -- TRUE now, and a
        # fabrication the day that function stops making it.  A text that is
        # simply silent about the aerosol source is not wrong either way.
        asserts_installed = any(re.search(p, text) for p in patterns)
        names_the_caller = re.search(_NAMES_THE_CALL_SITE, text) is not None
        wrong = (asserts_installed or names_the_caller) and not installed
        (disagreeing if wrong else agreeing).append((key, name))

    if not disagreeing:
        assert marker not in page, (
            "every sibling text agrees with the shipped call graph, so the "
            f"page must stop listing them: {[name for _, name in agreeing]}")
        return

    assert marker in page, (
        "these published texts disagree with the shipped call graph and the "
        f"page does not say so: {[key for key, _ in disagreeing]}")
    missing = [key for key, name in disagreeing if name not in page]
    assert missing == [], (
        f"the page's list of texts that are still wrong omits {missing}")
    lingering = [key for key, name in agreeing
                 if name in page.split(marker, 1)[1].split("\n\n", 1)[0]]
    assert lingering == [], (
        f"the page still lists {lingering} as wrong; they agree with the "
        "code")


def test_the_page_does_not_launder_the_registry_disagreement():
    """Page and registry must agree with the CODE, and say so when they do
    not agree with each other.

    PHYSICS.md's own first paragraph tells the reader the registry is the
    authority.  A page that quietly contradicts the authority it names is
    worse than one that is merely wrong.

    BEFORE THIS TEST: the page and the registry agreed -- on a false
    statement.
    """
    page = _read(PHYSICS_MD)
    warnings = " ".join(_registry_mp28_option()["warnings"])
    installed = bool(_production_callers_of_microphysics_init())

    page_says_installed = re.search(_NAMES_THE_CALL_SITE, page) is not None
    registry_says_installed = (
        re.search(_NAMES_THE_CALL_SITE, warnings) is not None)

    if page_says_installed != registry_says_installed:
        assert "physics_registry_v2.json" in page, (
            "the page and the registry disagree about whether the synthetic "
            "profile is installed; the page must name the disagreement and "
            "where it is filed, because its own opening paragraph tells the "
            "reader the registry is the authority")
        assert STALE_REGISTRY_NOTE in page, (
            "the page must state, in these words, that the registry warning "
            f"is stale: {STALE_REGISTRY_NOTE!r}")
        assert re.search(r"(?i)all of these are also registry warnings",
                         page) is None, (
            "the page still claims every deviation bullet is also a registry "
            "warning while one of them contradicts its registry warning")
        return

    assert STALE_REGISTRY_NOTE not in page, (
        "the page and the registry agree, so the page must stop telling the "
        "reader that the registry contradicts it")
    assert page_says_installed == installed, (
        "the page and the registry agree with each other and BOTH disagree "
        f"with gpuwm/: microphysics_init callers = "
        f"{_production_callers_of_microphysics_init()}")


# ---------------------------------------------------------------------------
# 6. The deviations a reader must be able to find.
# ---------------------------------------------------------------------------

#: Every deviation or coupling a user selecting mp=28 would care about, and
#: a token that proves the page states it.  This list is the outcome of
#: reading the page as a hostile outside reader; each entry is something an
#: auditor asked "where does it say that?" about.
_MUST_BE_FINDABLE = {
    "no WIF / GOCART aerosol ingest": "wif_input_opt",
    "no black-carbon species": "nbca",
    "aerosol-free lateral-boundary inflow": "flow-dependent boundaries",
    "the spec_zone ring ends at exactly zero aerosol": "spec_zone",
    "MYNN does not mix nc/nwfa/nifa": "bl_mynn_mixscalars",
    "MYNN's snow contract for mp=28": "flag_qs",
    # Audit R-004 ratified the mixed edge, so what a reader must find is
    # no longer a refusal but the CLOSURE: the values a mixed edge seeds
    # into an mp=28 child, and the fact that they are the scheme's own
    # rather than the parent's aerosol field.
    "mixed mp8/mp28 nesting seeds WRF's own non-aerosol-aware values":
        "is_aerosol_aware",
    "mp=28's RSLF is contraction-pinned while mp=8's is not": "RSLF",
    "WRF real.exe refuses this configuration":
        "module_initialize_real.F:2734-2736",
    "the RTE+RRTMGP cloud-optics coupling": "re_cloud",
    "CCN_ACTIVATE.BIN is distributed, and a different copy is refused":
        "CCN_ACTIVATE.BIN",
    # These two replace a single entry that pinned the exact sentence "No
    # forecast has ever been validated against WRF".  On 2026-08-01 that
    # sentence stopped being true: a matched IDEALIZED single-domain
    # doubly-periodic forecast was run against WRF v4.6.1 and published in
    # wrf-comparison/mp28-matched-trajectory.md.  Pinning a sentence is what
    # broke -- the sentence was rewritten and the token was not -- so the
    # replacements obey this file's own rule and name durable things: the
    # registry maturity state, and the evidence document.  Both are still
    # true after that comparison landed, which is the property a token
    # needs.  What the reader must not miss is unchanged in substance: the
    # scheme's label did not move, and the comparison that exists is
    # idealized, not real-data and not nested.
    "the maturity label is unverified, and the idealized comparison did "
    "not raise it": "implemented-unverified",
    "where the one matched forecast comparison lives, so its limits and "
    "its failed condition are one click away":
        "wrf-comparison/mp28-matched-trajectory.md",
}


def test_the_page_states_every_deviation_a_user_would_care_about():
    """The mp=28 section must be readable once, by a skeptic, with nothing
    material left to discover elsewhere.

    Each token below stands for a deviation an auditor of this port asked
    "where does the page say that?" about.  Tokens rather than sentences:
    the page must be free to phrase them, and a token that survives a
    rewrite is a token that names a WRF symbol, a namelist key or a gate.
    """
    page = _read(PHYSICS_MD)
    start = page.index("### Thompson aerosol-aware (`mp_physics = 28`)")
    end = page.index("## Planetary boundary layer")
    section = page[start:end]

    missing = sorted(what for what, token in _MUST_BE_FINDABLE.items()
                     if token not in section)
    assert missing == [], (
        "a reader who reads only the mp=28 section of docs/public/"
        f"PHYSICS.md would not learn: {missing}")


def test_the_page_does_not_assert_a_mynn_snow_deviation_the_tree_closed():
    """MYNN's ``FLAG_QS`` deviation is closed; the page must not still
    publish it as open.

    ``woof/core/mynn_pbl_runtime.py``'s ``MYNN_SNOW_MICROPHYSICS`` carries
    every WRF Registry package whose moist state declares ``F_QS`` --
    including 28 (``Registry.EM_COMMON:3036``) -- so snow reaches MYNN's
    condensation ``rh_hack`` for every snow-carrying scheme.  The page said
    the opposite, and attributed the claim to a registry entry that does not
    make it.

    Two-directional: if the flag is ever withdrawn for a snow scheme the
    page has to say so again.
    """
    from woof.core.mynn_pbl_runtime import mynn_flag_qs

    page = _read(PHYSICS_MD)
    snow_schemes = (6, 8, 10, 18, 28)
    mixed = {mp: mynn_flag_qs(mp) for mp in snow_schemes}

    registry_warnings = json.dumps(json.loads(_read(REGISTRY_JSON)))
    claim = ("snow mixing ratio is not passed to the condensation RH "
             "adjustment")

    if all(mixed.values()):
        assert claim not in page, (
            "docs/public/PHYSICS.md still publishes MYNN's snow deviation as "
            f"open, but mynn_flag_qs is true for every snow scheme {mixed}")
        assert "rh_hack" not in registry_warnings, (
            "the registry still carries the closed rh_hack deviation")
    else:                                             # pragma: no cover
        assert claim in page, (
            "mynn_flag_qs is false for "
            f"{sorted(k for k, v in mixed.items() if not v)}, "
            "so the page must publish the deviation again")


#: THE SKIP CENSUS FOR THIS FILE, because the mp=28 suite's own census
#: (``test_thompson_aerosol_gpu.py::test_the_mp28_suite_has_no_unaudited_skip_site``)
#: covers thirteen modules and this is not one of them -- it gates the
#: PUBLICATIONS, not the physics.  A skip is an unmeasured claim wherever it
#: lives, so the census is here instead of nowhere.  Each entry is
#: ``(enclosing def, kind)``.
_SKIP_SITES = frozenset({
    ("test_the_published_aerosol_sensitivity_is_a_live_measurement",
     "importorskip"),
    ("test_the_published_aerosol_sensitivity_is_a_live_measurement", "skip"),
    ("test_the_page_republishes_the_measured_depletion_numbers",
     "importorskip"),
    ("test_the_page_republishes_the_measured_depletion_numbers", "skip"),
})


def test_this_publication_gate_has_no_unaudited_skip_site():
    """Every skip in this file is device availability, and only that.

    The documentary assertions -- the counts, the residual tables, the
    attribution, the maturity label, the deviation census -- must never be
    skippable, because they are exactly the checks a machine without a GPU
    still needs to make before shipping a page. Fifteen of the seventeen
    tests here are host-only by construction and this proves it.
    """
    import ast

    tree = ast.parse(_read(pathlib.Path(__file__)))
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)]
    found = set()
    for function in functions:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = getattr(target, "attr", None)
            if name in ("skip", "importorskip", "xfail"):
                found.add((function.name, name))

    assert found == _SKIP_SITES, (
        "the skip census for this file is stale. Added: "
        f"{sorted(found - _SKIP_SITES)}; removed: "
        f"{sorted(_SKIP_SITES - found)}. Every skip here must be device "
        "availability and must be named in "
        "docs/public/wrf-comparison/mp28-column-evidence.md section 8.")

    tests = [function.name for function in functions
             if function.name.startswith("test_")]
    skipping = {name for name, _ in _SKIP_SITES}
    # Five documentary assertions leave the snapshot with their excluded inputs.
    # Keep exact counts in both trees; every retained public assertion still runs.
    expected_host_only = 20 if (ROOT / "RELEASE-EXCLUDE.txt").is_file() else 15
    assert len(tests) - len(skipping) == expected_host_only, (
        f"{len(tests)} tests, {len(skipping)} of which can skip; "
        f"expected {expected_host_only} host-only assertions in this tree")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
