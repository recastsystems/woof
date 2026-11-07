"""Every composition the loader accepts must also PRICE.

THE BLIND SPOT THIS CLOSES
    ``tools/report_physics_composition_walk.py`` measures one gate:
    :func:`woof.experiment.build_experiment`.  A row it marks ACCEPTED
    means the loader resolved the TOML into per-domain ``RunConfig``s with
    every selector preserved -- and nothing more.  The memory preflight is
    a SECOND gate, reached on every real run:
    ``woof/core/model.py`` calls
    :func:`woof.core.preflight.estimate_experiment` unconditionally while
    building the model, and ``woof check`` is that call and little else.
    Its selector tables (``_MICROPHYSICS_KERNEL_MODULES``,
    ``_PBL_KERNEL_MODULES``, ``_SURFACE_LAYER_KERNEL_MODULES``, ...) are
    FAIL-CLOSED: a selector value with no kernel-module row raises rather
    than pricing zero.

    So a scheme could be admitted by the loader, advertised in the schema
    menu, swept by the composition walk, recorded ACCEPTED in a published
    receipt -- and still be unable to run, because nothing ever asked the
    preflight to price it.  That is not hypothetical.  It shipped twice
    into the 1.9 assembly:

      * ``mp_physics = 50`` (P3 one-category) -- accepted at
        woof/config.py's inline mp check, absent from
        ``_MICROPHYSICS_KERNEL_MODULES``;
      * ``bl_pbl_physics = 2`` / ``sf_sfclay_physics = 2`` (MYJ + the Eta
        surface layer) -- accepted via ``PBL_SCHEMES`` /
        ``SURFACE_LAYER_SCHEMES``, absent from both PBL tables.

    Both were caught by a human reading the tables, which is the failure
    mode this file exists to retire.  A new scheme that reaches the loader
    without a kernel-module row now fails HERE, on every cut, by name.

WHAT IS MEASURED
    The receipt's ``accepted_combinations`` -- the measured accepted set,
    regenerated and byte-compared by
    ``tests/test_physics_composition_walk.py`` -- is decoded back into
    combinations, rebuilt through the same front door, and pushed through
    ``estimate_experiment``.  Decoding is not trusted: every key must
    round-trip through the walk's own ``key_of``, and every rebuild must
    still be ACCEPTED, so a lossy decode fails loudly instead of quietly
    testing something else.

WHAT CARD IT PRICES ON
    Every composition is priced on ONE explicit profile,
    :data:`PRICING_PROFILE`: the reference card's geometry carrying the
    compile platform of the tree's first Noah-MP composed-unit recording
    (:data:`woof.core.kernel_frame_recordings.NOAHMP_COMPOSED_FRAME_RECORDINGS`).
    Given a profile, the estimator reads no device, so the walk's outcome
    is the same on a box with a card, on a box without one, and under
    ``GPUWM_NO_LOCAL_GPU`` -- and ``sf_surface_physics = 4`` prices from
    its recorded row instead of refusing because the machine running the
    suite was never read.  (Before this profile the walk priced with no
    profile at all, which for scheme 4 means "read this machine's card":
    green under the switch only because a declared rule excused the
    refusal, red the moment a recorded card was visible.)  The estimator's
    card-class refusals for scheme 4 -- absent, unread, unrecorded, stale
    -- are held by ``tests/test_noahmp_frame_provenance.py``, not here.

WHAT A REFUSAL IS ALLOWED TO BE
    Not every accepted composition has to price.  A preflight refusal is
    legitimate when it is a DECLARED rule: a stated, reasoned refusal that
    names the way through, rather than a missing table row.  Those live in
    :data:`DECLARED_PRICING_RULES`, one entry each, with the predicate
    that says which compositions it may fire on.  A refusal matching no
    declared rule is a defect.  A declared rule that fires on nothing is
    dead text and also fails -- the escape hatch is not allowed to grow
    quietly.  The list is empty right now: every composition the loader
    accepts prices.
"""
from __future__ import annotations

import json
import re
import tomllib
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import requires_cupy

from woof.core.kernel_frame_recordings import NOAHMP_COMPOSED_FRAME_RECORDINGS
from woof.core.preflight import (
    MEASURED_LOCAL_MEMORY_PROFILE, estimate_experiment, physics_kernel_modules)
from woof.experiment import build_experiment
from tools import report_physics_composition_walk as walk

MODEL = Path(__file__).resolve().parents[1]
RECEIPT = (
    MODEL / "docs" / "public" / "receipts" / "physics-composition-walk.json"
)

#: The walk's compact key tags, inverted.  ``raagg`` is the legacy
#: ``ra_physics`` aggregate rather than an axis, so it is mapped by hand.
_TAG_TO_AXIS = {tag: name for name, tag in walk._TAG.items()}
_PART = re.compile(r"([a-z]+)(-?\d+)")


def decode(key: str) -> dict[str, object]:
    """One accepted key back into the combination that produced it."""

    combination: dict[str, object] = {}
    for part in key.split("."):
        if part == "ack":
            combination["km_opt_zero_acknowledgement"] = walk.KM_OPT_ZERO_ACK
            continue
        if part == "noack":
            combination["_ack_withheld"] = True
            combination["km_opt_zero_acknowledgement"] = ""
            continue
        if part == "noradack":
            combination["_radiation_ack_withheld"] = True
            continue
        if "=" in part:
            # key_of spells the radiation-variant settings as name=value
            # (ra_rrtmg_variant, wrf_rrtmg_compatibility).
            name, value = part.split("=", 1)
            combination[name] = value
            continue
        matched = _PART.fullmatch(part)
        if matched is None:
            raise AssertionError(
                f"unrecognised accepted-combination key segment {part!r} in "
                f"{key!r}; this decoder and "
                "tools/report_physics_composition_walk.py:key_of have "
                "drifted apart")
        tag, value = matched.group(1), int(matched.group(2))
        if tag == "raagg":
            combination["ra_physics"] = value
        else:
            combination[_TAG_TO_AXIS[tag]] = value
    return combination


def build(combination: dict[str, object]):
    """Push one combination through the production front door."""

    text = walk.render_config(combination)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_experiment(
            tomllib.loads(text), source="composition-pricing.toml")


# ---------------------------------------------------------------------------
# The declared refusals.
# ---------------------------------------------------------------------------

#: Preflight refusals that are RULES, not gaps.  Each entry is
#: ``(id, predicate, signature)``: the predicate says which compositions
#: the rule may fire on, and the signature is a fragment the refusal's own
#: message must contain.  Both halves matter -- a rule that fires outside
#: its predicate is a different defect wearing a declared rule's name.
#:
#: Empty: nothing the loader accepts is refused by the estimator.  The
#: Noah-MP entry that lived here (``sf_surface_physics = 4`` refused for
#: want of a measured composite frame, then for want of a read card) left
#: with the per-platform recording that prices scheme 4 and with
#: :data:`PRICING_PROFILE`, which hands the walk that platform explicitly.
DECLARED_PRICING_RULES: tuple[tuple[str, object, str], ...] = ()


#: The card every composition is priced on.  A present-card profile in
#: the estimator's own terms: the reference geometry with the compile
#: platform of the first recorded Noah-MP row, so the walk asks "does
#: this composition price on a card whose platform has a reading" and
#: never "was the machine running this suite read".
assert NOAHMP_COMPOSED_FRAME_RECORDINGS, (
    "the tree carries no Noah-MP composed-unit recording, so no card can "
    "price sf_surface_physics = 4 and this walk has no platform to hand "
    "the estimator")
PRICING_PROFILE = replace(
    MEASURED_LOCAL_MEMORY_PROFILE,
    name="composition-pricing walk card (reference geometry, recorded platform)",
    compile_platform=NOAHMP_COMPOSED_FRAME_RECORDINGS[0].platform_key)


@pytest.fixture(scope="module")
def accepted_keys() -> tuple[str, ...]:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    keys = tuple(receipt["accepted_combinations"])
    assert keys, "the composition walk recorded no accepted combination"
    return keys


def _price_batch(keys) -> tuple[list, list, list]:
    """``(ok, refused, rebuild_refused)`` for ``keys``, in their order."""

    ok: list[str] = []
    refused: list[tuple[str, dict[str, object], str]] = []
    rebuild_refused: list[tuple[str, str]] = []
    for key in keys:
        combination = decode(key)
        assert walk.key_of(combination) == key, (
            f"decoding {key!r} did not round-trip through key_of; the "
            "pricing walk would be measuring a different composition than "
            "the one the receipt accepted")
        try:
            experiment = build(combination)
        except Exception as error:  # noqa: BLE001 -- the verdict IS the error
            rebuild_refused.append((key, f"{type(error).__name__}: {error}"))
            continue
        try:
            estimate_experiment(experiment, profile=PRICING_PROFILE)
        except Exception as error:  # noqa: BLE001 -- the verdict IS the error
            refused.append((key, combination, str(error)))
        else:
            ok.append(key)
    return ok, refused, rebuild_refused


@pytest.fixture(scope="module")
def priced(accepted_keys) -> dict[str, object]:
    """Build and price every accepted composition, once, for the file,
    on :data:`PRICING_PROFILE` -- no device is read.

    The 71,068 compositions are priced in bounded batches across the walk's
    own worker count (``WOOF_COMPOSITION_WORKERS``, default 8, capped at the
    cores this process may use), and joined in receipt order, so the verdict
    is the one a single process gives.  Breakage (seal3 stage1, 2026-11-03):
    priced on one core this fixture took 1,671 s, the tail every other stage1
    worker sat idle behind; a single test past the 1,500 s pytest-timeout
    there ends its xdist worker, and the replacement re-collects all 962 files
    before anything runs again.
    """

    import itertools
    import os
    keys = list(accepted_keys)
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    jobs = max(1, min(int(os.environ.get("WOOF_COMPOSITION_WORKERS", "8")), available))
    batches = [keys[start:start + 256] for start in range(0, len(keys), 256)]
    if jobs == 1:
        parts = [_price_batch(batch) for batch in batches]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            parts = list(pool.map(_price_batch, batches))
    ok, refused, rebuild_refused = (tuple(itertools.chain.from_iterable(part[index] for part in parts))
                                    for index in range(3))
    return {"ok": ok, "refused": refused, "rebuild_refused": rebuild_refused}


def test_every_accepted_composition_rebuilds(priced):
    """A receipt row that no longer builds means the decode is wrong."""

    assert priced["rebuild_refused"] == (), (
        "the receipt records these combinations as ACCEPTED but rebuilding "
        "them through build_experiment refused:\n  "
        + "\n  ".join(f"{key}: {message}"
                      for key, message in priced["rebuild_refused"][:10]))


# NEEDS CUPY INSTALLED, and opens no device: this test prices every
# composition the loader accepts, which imports the kernel modules; without
# cupy 235 of them report the absent module rather than a pricing gap.
@requires_cupy
def test_every_accepted_composition_can_be_priced(priced):
    """THE INSTRUMENT.  Loader-accepted implies preflight-priceable.

    An unpriceable composition is a scheme a user can select, that
    ``woof check`` refuses and ``woof run`` dies on.  The only refusals
    allowed through are the declared rules above, and only on the
    compositions their predicate admits.
    """

    undeclared = []
    for key, combination, message in priced["refused"]:
        flat = " ".join(message.split())
        matches = [rule_id
                   for rule_id, predicate, signature in DECLARED_PRICING_RULES
                   if signature in flat and predicate(combination)]
        if not matches:
            undeclared.append((key, flat))
    assert not undeclared, (
        f"{len(undeclared)} composition(s) the loader ACCEPTS cannot be "
        "priced by woof/core/preflight.py, and the refusal is not a "
        "declared composition rule.  Every one of these is a scheme a user "
        "can select and cannot run: `woof check` refuses it and "
        "woof/core/model.py dies on the same call while building the "
        "model.  Add the missing kernel-module row (with accurate pricing "
        "derived from the scheme's actual kernels), or declare the refusal "
        "in DECLARED_PRICING_RULES with its reason.\n  "
        + "\n  ".join(f"{key}: {message}" for key, message in undeclared[:10]))


def test_no_declared_pricing_rule_is_dead_text(priced):
    """An escape hatch nothing uses is an escape hatch nobody audits."""

    fired = {rule_id
             for key, combination, message in priced["refused"]
             for rule_id, predicate, signature in DECLARED_PRICING_RULES
             if signature in " ".join(message.split())
             and predicate(combination)}
    declared = {rule_id for rule_id, _, _ in DECLARED_PRICING_RULES}
    assert declared == fired, (
        "these declared pricing rules fired on nothing and are now dead "
        f"text: {sorted(declared - fired)}.  If the underlying refusal was "
        "resolved, delete the entry in the same commit that resolved it.")


def test_the_walk_priced_a_meaningful_share_of_its_accepted_set(priced,
                                                                accepted_keys):
    """Guard against a green that comes from measuring almost nothing."""

    assert len(priced["ok"]) > len(accepted_keys) // 2, (
        f"only {len(priced['ok'])} of {len(accepted_keys)} accepted "
        "compositions priced; a majority behind declared refusals means the "
        "declared-rule list has become the measurement")


# ---------------------------------------------------------------------------
# The two schemes this file was written for, named, so the failure reads.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,combination,expected_modules",
    [
        # P3 one-category on the shipped YSU/MM5/Noah anchor with the
        # Dudhia radiation pair -- kept on Dudhia so this row prices the
        # microphysics alone.  (The RTE+RRTMGP pairing is coupled now via
        # rrtmgp's ``50: "p3"`` row; the refusal this comment used to cite
        # retired with that fix.)
        #
        # P3 is a HOST transcription and launches no kernel of its own, so
        # what its row must produce is the shared moisture validator -- and
        # NOT ``refl``, whose absence is the other half of the accurate price.
        ("mp_physics=50 (P3 one-category)",
         walk._suite(mp_physics=50, ra_lw_physics=0, ra_sw_physics=1),
         {"present": ("microphysics_validation",), "absent": ("refl",)}),
        # MYJ with its own Eta surface layer, the pairing
        # woof.config.validate_myj_pairing enforces in both directions.
        ("bl_pbl_physics=2 + sf_sfclay_physics=2 (MYJ + Eta)",
         walk._suite(bl_pbl_physics=2, sf_sfclay_physics=2),
         {"present": ("myjpbl", "myjsfc"), "absent": ("ysu", "sfclay")}),
    ],
)
def test_the_1_9_resurrected_schemes_price(label, combination,
                                           expected_modules):
    """mp=50 and MYJ, priced by name, against the modules they launch.

    These two are covered by the receipt walk above; they are ALSO spelled
    out here so that dropping their kernel-module row fails with the
    scheme's name in the test id rather than as one line inside a list of
    991.  Both were accepted-but-unpriceable at b557703ba.

    The module assertions are what stop the blocker from being "fixed" by
    an empty row: a scheme priced at nothing prices fine and reserves
    nothing, which is the under-pricing direction that put a run 1,630 MiB
    over.  ``absent`` pins the two accurate exclusions -- P3's ``refl`` and
    MYJ's YSU/MM5 modules -- so a copy-paste row fails too.
    """

    experiment = build(combination)
    modules = physics_kernel_modules(experiment)
    missing = [name for name in expected_modules["present"]
               if name not in modules]
    assert not missing, f"{label}: unpriced kernel module(s) {missing}"
    surplus = [name for name in expected_modules["absent"] if name in modules]
    assert not surplus, (
        f"{label}: priced kernel module(s) {surplus} the scheme never "
        "launches; over-pricing a rail gate refuses runs that would fit")
    estimate = estimate_experiment(experiment)
    assert estimate.alloc_estimate_bytes > 0, label
