"""No shipped config may seed inflow turbulence it cannot run.

``inflow_perturbation`` takes its vertical extent from the PARENT's
diagnosed PBLH, so a parent running ``bl_pbl_physics = 0`` has nothing
to define it with and the generator refuses.  That is the mechanism's
applicability boundary and it is ratified as G4 in
``docs/internal/specs/P6-LES-DECISIONS-RATIFIED-2026-08-05.md``: ON at
the first LES domain under a PARAMETERIZED-turbulence parent, OFF below
it, because a PBL-off parent is itself LES and its RESOLVED eddies
already ARE the child's inflow turbulence.

The refusal was real and it worked -- G4 records it killing the G5 LES
smoke twelve seconds after the tree reached the card.  What did not work
is WHERE it lives: at ``NestCoupler`` construction, downstream of a
fetch, two preparations and a whole prepared tree, for a fact that was
legible in the TOML the whole time.  So a config shipped tripping it
(``les_tornado_100m_dodgecity_20160524.toml``, whose d04 sits under a
PBL-off d03) and nothing in either tree noticed.

This file is the durable half of that fix: the sweep is the test, and
the config edit is one instance of it.  Everything here is CPU-only.
"""
from __future__ import annotations

import io
import tomllib
from pathlib import Path

import pytest

from woof.core import inflow_perturbation as ip
from woof.experiment import build_experiment, load_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

#: Companion tables of the one-file case schema, split off by every file
#: loader before ``build_experiment`` sees the dict (experiment.py:2352).
_COMPANION_TABLES = ("fetch", "case_data", "static", "ingest")

#: A legal shipped tree to mutate: d03 (LES) seeded under d02 (YSU), and
#: d04 (LES) deliberately unseeded -- G4's ruling, in a shipped file.
_LEGAL_TREE = "les_tornado_100m_mayfield_20211210.toml"


@pytest.fixture
def version_identity_bound(monkeypatch):
    """Stand down ONE unrelated, pre-existing refusal for these loads.

    ``build_experiment`` calls
    :func:`woof.provenance_gate.require_version_identity`, which refuses
    whenever the running tree's ``pyproject.toml`` version and the
    version the installed distribution reports disagree.  Running a
    suite out of a worktree is exactly that case on this machine, so
    EVERY experiment-config load refuses regardless of what the config
    says.  It is a real refusal doing its real job and it is not about
    inflow seeding.  Same idiom, same wording, as
    ``tests/test_p3_front_door.py`` and ``tests/test_chain_deep_ladder``
    already use, and self-retiring: once the tree under test is bound to
    its own metadata the early return fires and nothing is patched.
    """
    import woof.provenance_gate as gate

    if gate.version_identity_refusal() is None:
        return
    monkeypatch.setattr(gate, "version_identity_refusal",
                        lambda prov=None: None)


def _parent_of(experiment, domain):
    """The domain's parent config, or ``None`` for the root.

    The root carries ``parent_id == 0`` (experiment.py:2775-2780), which
    is not a grid id, so the lookup answers ``None`` there by itself.
    """
    by_id = {int(dc.grid_id): dc for dc in experiment.domains}
    return by_id.get(int(domain.parent_id))


def _offenders(experiment):
    """(grid_id, parent_id, refusal) for every unrunnable pairing."""
    out = []
    for domain in experiment.domains:
        if not getattr(domain.run, "inflow_perturbation", False):
            continue
        parent = _parent_of(experiment, domain)
        refusal = ip.parent_pbl_refusal(
            None if parent is None else parent.run.bl_pbl_physics)
        if refusal is not None:
            out.append((int(domain.grid_id), int(domain.parent_id), refusal))
    return out


def _raw_tables(name):
    """One shipped config as ``build_experiment`` wants it: split."""
    raw = tomllib.load(io.BytesIO((CONFIGS / name).read_bytes()))
    for table in _COMPANION_TABLES:
        raw.pop(table, None)
    return raw


def _domain_table(raw, grid_id):
    for table in raw["domain"]:
        if int(table["grid_id"]) == int(grid_id):
            return table
    raise AssertionError(f"{_LEGAL_TREE} has no grid_id = {grid_id}")


def test_no_shipped_config_declares_inflow_seeding_under_a_pbl_off_parent(
        version_identity_bound):
    """The sweep. One config shipped in this state; none may again.

    A config that will not load for an unrelated reason (an unset
    ``${GPUWM_*}`` path, a legacy single-domain schema) is reported as
    skipped rather than silently dropped -- a sweep that covers nothing
    passes for the wrong reason, so the covered count is asserted too.
    """
    offenders, skipped, covered = [], [], 0
    for path in sorted(CONFIGS.glob("*.toml")):
        try:
            experiment = load_experiment(path)
        except Exception as exc:                        # noqa: BLE001
            if ip.PARENT_PBL_REFUSAL_MARKER in str(exc):
                # OUR refusal, fired at load.  That is the config being
                # wrong, not the environment, and it is never a skip.
                offenders.append(f"{path.name}: refused at load -- "
                                 f"{str(exc).splitlines()[0]}")
            else:
                skipped.append(f"{path.name}: {type(exc).__name__}")
            continue
        covered += 1
        for grid_id, parent_id, refusal in _offenders(experiment):
            offenders.append(
                f"{path.name}: d{grid_id:02d} (parent d{parent_id:02d}) "
                f"declares inflow_perturbation = true -- {refusal}")
    assert not offenders, (
        "shipped configs that cannot run their own inflow seeding:\n  "
        + "\n  ".join(offenders))
    assert covered >= 60, (
        f"the sweep only loaded {covered} configs ({len(skipped)} skipped: "
        f"{skipped[:5]}); it is not covering the tree it claims to")


def test_build_experiment_refuses_the_pairing_by_name(version_identity_bound):
    """The check belongs at the shared load, not at coupler construction.

    Every front door reaches ``build_experiment`` -- ``load_experiment``,
    the LES route in ``hrrr_hierarchy_direct``, ``core.preflight``, the
    domain wizard, the namelist importer -- so this is the one seam
    where a door nobody remembered to wire still inherits the refusal.
    """
    raw = _raw_tables(_LEGAL_TREE)
    _domain_table(raw, 4)["inflow_perturbation"] = True
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<inflow-applicability-mutation>")
    text = str(caught.value)
    assert ip.PARENT_PBL_REFUSAL_MARKER in text
    assert "grid_id = 4" in text                 # the domain that is wrong
    assert "grid_id = 3" in text                 # the parent that decides it
    assert "bl_pbl_physics = 0" in text          # the value that decides it
    assert "inflow_perturbation = false" in text  # the remedy, spelled out
    assert "P6-LES-DECISIONS-RATIFIED-2026-08-05" in text   # the ruling


def test_a_legal_pairing_is_still_admitted(version_identity_bound):
    """The deliberately-wrong input for the new refusal: a LEGAL tree.

    Unmutated, ``les_tornado_100m_mayfield_20211210.toml`` seeds d03
    under a ``bl_pbl_physics = 1`` parent, which is the pairing G4 rules
    ON.  If this ever fails the new check is refusing everything, and
    the LES campaign's own configuration with it.
    """
    experiment = build_experiment(_raw_tables(_LEGAL_TREE),
                                  source="<inflow-applicability-control>")
    seeded = [int(dc.grid_id) for dc in experiment.domains
              if dc.run.inflow_perturbation]
    assert seeded == [3]
    assert _offenders(experiment) == []


def test_the_two_seams_speak_with_one_voice():
    """One copy of the predicate, read by both places that enforce it.

    ``build_inflow_perturbation`` refuses at coupler construction and
    ``build_experiment`` refuses at the config load; a second spelling
    of the same rule is a second thing to keep true.
    """
    assert ip.parent_pbl_refusal(1) is None
    assert ip.parent_pbl_refusal(11) is None
    for absent in (0, None):
        refusal = ip.parent_pbl_refusal(absent)
        assert refusal is not None
        assert ip.PARENT_PBL_REFUSAL_MARKER in refusal
        assert "bl_pbl_physics=0" in refusal
