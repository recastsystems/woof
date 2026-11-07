"""ACCEPTED IMPLIES CHECKPOINTABLE: every admitted scheme can be NAMED.

WHAT THIS PREVENTS, concretely and measured on 2026-09-10 07:39Z: a
single-domain ERA5 forecast configured with ``mp_physics = 9`` -- a value
``validate_run_config`` accepts and ``woof/core/milbrandt2.py`` integrates
-- ran for 59 minutes and then died writing its first hourly checkpoint
with

    RestartManifestError: cannot identify unsupported microphysics
    scheme 9

because ``MICROPHYSICS_ALGORITHM_IDENTITIES`` had no row for it.  The
forecast was lost.  Nothing earlier in the run asked whether the
checkpoint writer could name the scheme, so nothing earlier could refuse.

Two rungs are held here.  The first is the TABLES: for every scheme id
each registry declares implemented, the identity table that a checkpoint
resolves it through has a row.  The second is the GATE: the question is
asked at plan review, by ``validate_run_config``, on a bare default
configuration with no flag to set -- so a configuration that could not be
checkpointed is refused before any device work instead of an hour into
the forecast.

A THIRD rung was added on 2026-09-10: WHERE the tables live.  The gate is
called from ``validate_run_config``, and ``woof/config.py`` ships in
distributions that stage no ``woof/io`` -- the standalone RW-WPS
preprocessing project stages ``woof/*.py`` plus a named handful of
``woof/io`` modules, and ``restart.py`` is deliberately not one of them.
Reading the gate out of ``woof.io.restart`` therefore made plan review
itself unimportable there.  The tables and the gate now live in
``woof.checkpoint_identity``, which imports neither ``woof.io`` nor
CuPy, and ``woof.io.restart`` imports the same objects back under the
names the tree already spells -- so a table is monkeypatched at its
definition below, and the re-export is asserted to BE that object rather
than a second copy that could drift from it.

CPU-only: identity tables and configuration validation, no runtime.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import textwrap

import pytest

from woof import checkpoint_identity
from woof import config as config_mod
from woof.config import (CU_SCHEMES, LAND_SURFACE_SCHEMES,
                          MP_PHYSICS_ACCEPTED, PBL_SCHEMES,
                          RA_LW_PHYSICS_ACCEPTED, RA_PHYSICS_ACCEPTED,
                          RA_SW_PHYSICS_ACCEPTED, SURFACE_LAYER_SCHEMES,
                          RunConfig, validate_run_config)
from woof.io import restart


_TINY = dict(nx=8, ny=6, nz=5, dx=2000.0, dy=2000.0, ztop=8000.0,
             dt=10.0, run_seconds=10.0)


def _cfg(**overrides) -> RunConfig:
    values = dict(_TINY)
    values.update(overrides)
    return RunConfig(**values)


# ---------------------------------------------------------------------------
# Rung 1: the tables cover every registry the loader gates on.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme_id", MP_PHYSICS_ACCEPTED)
def test_every_accepted_microphysics_scheme_has_an_identity(scheme_id):
    assert scheme_id in restart.MICROPHYSICS_ALGORITHM_IDENTITIES, (
        f"mp_physics={scheme_id} is accepted by the loader "
        "(woof.config.MP_PHYSICS_ACCEPTED) but has no checkpoint identity "
        "row; a run configured with it dies at its first restart interval")


@pytest.mark.parametrize("scheme_id", MP_PHYSICS_ACCEPTED)
def test_published_microphysics_identity_matches_checkpoint_owner(scheme_id):
    """Published preparation identity must name the restored algorithm."""
    from woof.physics_registry import consumer_row_for_selector

    published = consumer_row_for_selector(
        "microphysics", "restart_algorithm_identity", scheme_id)
    assert published == checkpoint_identity.MICROPHYSICS_ALGORITHM_IDENTITIES[scheme_id]


def test_milbrandt_yau_is_identifiable():
    """The 2026-09-10 defect, pinned by number rather than by iteration."""
    identity = restart.MICROPHYSICS_ALGORITHM_IDENTITIES[9]
    assert "milbrandt" in identity
    # Named at the mp=8/28/50 granularity: the trajectory-defining
    # configuration, not the scheme name alone.
    assert "wrf-v4.6.1" in identity
    assert "six-category-2mom" in identity


@pytest.mark.parametrize("scheme_id", SURFACE_LAYER_SCHEMES)
def test_every_accepted_surface_layer_scheme_has_an_identity(scheme_id):
    assert scheme_id in restart.SURFACE_LAYER_ALGORITHM_IDENTITIES


@pytest.mark.parametrize("scheme_id", LAND_SURFACE_SCHEMES)
def test_every_accepted_land_surface_scheme_has_an_identity(scheme_id):
    assert scheme_id in restart.LAND_SURFACE_ALGORITHM_IDENTITIES


@pytest.mark.parametrize("scheme_id", PBL_SCHEMES)
def test_every_accepted_pbl_scheme_has_an_identity(scheme_id):
    assert scheme_id in restart.PBL_ALGORITHM_IDENTITIES


@pytest.mark.parametrize("scheme_id", CU_SCHEMES)
def test_every_accepted_cumulus_scheme_has_an_identity(scheme_id):
    assert scheme_id in restart.CUMULUS_ALGORITHM_IDENTITIES


@pytest.mark.parametrize("scheme_id", RA_LW_PHYSICS_ACCEPTED)
def test_every_accepted_longwave_scheme_has_an_identity(scheme_id):
    assert scheme_id in restart.LONGWAVE_ALGORITHM_IDENTITIES
    assert scheme_id in restart.LONGWAVE_ABOVE_ATMOSPHERE_POLICIES


@pytest.mark.parametrize("scheme_id", RA_SW_PHYSICS_ACCEPTED)
def test_every_accepted_shortwave_scheme_has_an_identity(scheme_id):
    assert scheme_id in restart.SHORTWAVE_ALGORITHM_IDENTITIES
    assert scheme_id in restart.SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES


@pytest.mark.parametrize("scheme_id", RA_PHYSICS_ACCEPTED)
def test_every_accepted_coupled_radiation_selector_has_an_identity(scheme_id):
    assert scheme_id in restart.RADIATION_ALGORITHM_IDENTITIES
    assert scheme_id in restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES


# ---------------------------------------------------------------------------
# Rung 2: the gate asks the question at plan review.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mp_physics", MP_PHYSICS_ACCEPTED)
def test_the_gate_passes_every_accepted_microphysics_configuration(mp_physics):
    cfg = _cfg(moist=(mp_physics != 0), mp_physics=mp_physics)
    assert restart.unidentifiable_checkpoint_schemes(cfg) == []
    restart.require_identifiable_checkpoint_schemes(cfg)


def test_the_gate_passes_a_bare_default_configuration():
    cfg = _cfg()
    assert restart.unidentifiable_checkpoint_schemes(cfg) == []


def test_a_missing_row_is_refused_and_names_the_scheme(monkeypatch):
    """Remove mp=9's row and the gate reproduces the defect's diagnosis."""
    table = dict(restart.MICROPHYSICS_ALGORITHM_IDENTITIES)
    table.pop(9)
    monkeypatch.setattr(checkpoint_identity,
                        "MICROPHYSICS_ALGORITHM_IDENTITIES", table)

    cfg = _cfg(moist=True, mp_physics=9)
    gaps = restart.unidentifiable_checkpoint_schemes(cfg)
    assert gaps == [
        "microphysics scheme 9 (mp_physics) has no row in the checkpoint "
        "identity table"]
    with pytest.raises(ValueError) as excinfo:
        restart.require_identifiable_checkpoint_schemes(cfg)
    message = str(excinfo.value)
    assert not isinstance(excinfo.value, restart.RestartManifestError)
    assert "microphysics scheme 9" in message
    # The refusal names the concrete breakage it prevents.
    assert "first restart interval" in message
    assert "discards the forecast" in message


def test_plan_review_refuses_before_the_run_starts(monkeypatch):
    """validate_run_config -- what every door runs -- carries the gate.

    This is the property that turns the defect from a lost forecast into
    a refused configuration: no flag, no opt-in, and no device work
    between the answer and the question.
    """
    table = dict(restart.MICROPHYSICS_ALGORITHM_IDENTITIES)
    table.pop(9)
    monkeypatch.setattr(checkpoint_identity,
                        "MICROPHYSICS_ALGORITHM_IDENTITIES", table)

    # ValueError, the type every door already renders as a refusal with
    # exit 2 rather than a traceback (woof/cli.py) -- a plan-review
    # refusal has to READ like the loader's other refusals.
    with pytest.raises(ValueError, match="microphysics scheme 9"):
        validate_run_config(_cfg(moist=True, mp_physics=9))


def test_plan_review_admits_milbrandt_yau_as_shipped():
    """The bare default run of the defect's configuration is accepted."""
    cfg = validate_run_config(_cfg(moist=True, mp_physics=9,
                                   restart_interval_s=600.0))
    assert cfg.mp_physics == 9


@pytest.mark.parametrize(
    ("table_name", "field", "value", "label"),
    [("SURFACE_LAYER_ALGORITHM_IDENTITIES", "sf_sfclay_physics", 1,
      "surface layer"),
     ("LAND_SURFACE_ALGORITHM_IDENTITIES", "sf_surface_physics", 2,
      "land surface"),
     ("LAND_SURFACE_PARAMETER_SOURCES", "sf_surface_physics", 2,
      "land-surface parameter bundle"),
     ("PBL_ALGORITHM_IDENTITIES", "bl_pbl_physics", 1, "PBL"),
     ("CUMULUS_ALGORITHM_IDENTITIES", "cu_physics", 1, "cumulus"),
     ("LONGWAVE_ALGORITHM_IDENTITIES", "ra_lw_physics", 4,
      "longwave radiation"),
     ("LONGWAVE_ABOVE_ATMOSPHERE_POLICIES", "ra_lw_physics", 4,
      "longwave above-atmosphere policy"),
     ("SHORTWAVE_ALGORITHM_IDENTITIES", "ra_sw_physics", 1,
      "shortwave radiation"),
     ("SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES", "ra_sw_physics", 1,
      "shortwave above-atmosphere policy")])
def test_every_configured_family_is_gated(monkeypatch, table_name, field,
                                          value, label):
    """Microphysics is not a special case: each table is asked about."""
    table = dict(getattr(restart, table_name))
    table.pop(value)
    monkeypatch.setattr(checkpoint_identity, table_name, table)

    overrides = {field: value}
    if field in ("ra_lw_physics", "ra_sw_physics"):
        # A MIXED pair, because that is the branch these two tables are
        # read on: a coupled selection resolves through the RADIATION_*
        # pair instead, which has its own cases below.
        overrides = {"ra_lw_physics": 4, "ra_sw_physics": 1, "icloud": 1}
    cfg = _cfg(moist=True, mp_physics=10, **overrides)
    gaps = restart.unidentifiable_checkpoint_schemes(cfg)
    assert any(gap.startswith(f"{label} scheme {value}") for gap in gaps), gaps


def test_the_noahmp_snow_age_generation_is_gated(monkeypatch):
    """The generation-keyed table is asked about by generation name."""
    cfg = _cfg(moist=True, mp_physics=10)
    assert restart.unidentifiable_checkpoint_schemes(cfg) == []
    table = dict(restart.NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES)
    table.pop(cfg.noahmp_snow_age)
    monkeypatch.setattr(checkpoint_identity,
                        "NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES", table)
    gaps = restart.unidentifiable_checkpoint_schemes(cfg)
    assert any(gap.startswith("Noah-MP snow aging generation") for gap in gaps), gaps


_IDENTITY_TABLE_NAME = re.compile(
    r"^[A-Z0-9_]+_(ALGORITHM_IDENTITIES|ABOVE_ATMOSPHERE_POLICIES"
    r"|PARAMETER_SOURCES)$")


def _tables_the_writer_reads():
    """Identity tables reachable from ``physics_setup_identity``, from SOURCE.

    Walks the call graph out of the checkpoint's identity entry point and
    collects every module-level identity table any function in it names.
    DERIVED, not restated: a table added to the writer, or a new helper
    that reads one, turns up here without anyone editing this file, which
    is the only way this comparison can catch the case it exists for.
    """
    tree = ast.parse(inspect.getsource(restart))
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    tables = {}
    seen = set()
    pending = ["physics_setup_identity"]
    while pending:
        name = pending.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        for node in ast.walk(functions[name]):
            if (isinstance(node, ast.Name)
                    and _IDENTITY_TABLE_NAME.match(node.id)):
                tables.setdefault(node.id, set()).add(name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                pending.append(node.func.id)
    # The walk found the entry point and followed a call out of it; if it
    # ever stops doing either, the comparison below is vacuous.
    assert "physics_setup_identity" in seen
    assert "_radiation_setup_identity" in seen, sorted(seen)
    assert "_land_surface_parameters_identity" in seen, sorted(seen)
    return tables


def _tables_the_gate_asks_about():
    """Identity tables the plan-review gate consults, read off the gate.

    The unconditional rows carry their table name.  The radiation rows are
    chosen per configuration, so every branch is collected by walking the
    string constants in ``_radiation_identity_table_rows`` -- derived for
    the same reason the writer side is.
    """
    names = {name for name, _, _
             in checkpoint_identity._CONFIGURED_IDENTITY_TABLES}
    radiation = ast.parse(textwrap.dedent(
        inspect.getsource(
            checkpoint_identity._radiation_identity_table_rows)))
    for node in ast.walk(radiation):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and _IDENTITY_TABLE_NAME.match(node.value)):
            names.add(node.value)
    return names


def test_the_gated_tables_are_the_tables_a_checkpoint_reads():
    """The gate's table set is the writer's table set, computed from both.

    A checkpoint resolves an identity out of one table per family plus the
    per-scheme parameter bundles and above-atmosphere policies those
    resolutions reach.  A table read on the way to a checkpoint and NOT
    asked about at plan review is the 2026-09-10 defect with a different
    table name on it: the loader admits a scheme, the forecast integrates
    it, and the first restart interval raises ``RestartManifestError``
    with an hour of forecast in it.  Both sides are computed -- the
    writer's from its call graph, the gate's from its own rows -- so
    neither can drift from a literal written here, because there is none.
    """
    writer = _tables_the_writer_reads()
    gated = _tables_the_gate_asks_about()
    unasked = {name: sorted(where) for name, where in writer.items()
               if name not in gated}
    assert unasked == {}, (
        "the checkpoint writer resolves " + str(sorted(unasked)) + " from "
        "configuration, and plan review never asks whether this "
        "configuration has a row in them")
    # And every name the gate carries is a real table, so a typo in a row
    # cannot pass as coverage.
    for name in gated:
        assert isinstance(getattr(restart, name), dict), name


def test_the_gated_attributes_are_real_configuration_fields():
    cfg = _cfg(moist=True, mp_physics=10)
    fields = {field.name for field in dataclasses.fields(cfg)}
    for _, attribute, _ in (
            checkpoint_identity._CONFIGURED_IDENTITY_TABLES):
        assert attribute in fields
    for _, attribute, _, _ in (
            checkpoint_identity._radiation_identity_table_rows(cfg)):
        assert attribute in fields


@pytest.mark.parametrize("scheme_id",
                         [value for value in LAND_SURFACE_SCHEMES if value])
def test_every_active_land_surface_scheme_has_a_parameter_bundle(scheme_id):
    """The SECOND table a land-surface scheme is named through.

    ``_land_surface_parameters_identity`` resolves the packed bundle
    through ``LAND_SURFACE_PARAMETER_SOURCES`` and refuses a scheme with
    no row there -- the same failure, at the same point in the run, as the
    microphysics lookup that lost a forecast.
    """
    assert scheme_id in restart.LAND_SURFACE_PARAMETER_SOURCES


def test_a_land_surface_scheme_without_a_parameter_bundle_is_refused(
        monkeypatch):
    """An LSM in the algorithm table but not the bundle table is a gap."""
    sources = dict(restart.LAND_SURFACE_PARAMETER_SOURCES)
    sources.pop(2)
    monkeypatch.setattr(checkpoint_identity,
                        "LAND_SURFACE_PARAMETER_SOURCES", sources)

    cfg = _cfg(moist=True, sf_surface_physics=2)
    gaps = restart.unidentifiable_checkpoint_schemes(cfg)
    assert gaps == [
        "land-surface parameter bundle scheme 2 (sf_surface_physics) has "
        "no row in the checkpoint identity table"]
    with pytest.raises(ValueError, match="land-surface parameter bundle"):
        restart.require_identifiable_checkpoint_schemes(cfg)


def test_no_land_surface_needs_no_parameter_bundle(monkeypatch):
    """Scheme 0 resolves no bundle, so the gate must not invent one.

    ``physics_setup_identity`` reaches the bundle only when a land surface
    runs.  A gate that refused ``sf_surface_physics = 0`` for having no
    row in a table it never reads would block a run the writer would have
    written -- a refusal naming a breakage that cannot happen.
    """
    sources = dict(restart.LAND_SURFACE_PARAMETER_SOURCES)
    sources.pop(2, None)
    monkeypatch.setattr(checkpoint_identity,
                        "LAND_SURFACE_PARAMETER_SOURCES", sources)

    cfg = _cfg(sf_surface_physics=0, sf_sfclay_physics=0, bl_pbl_physics=0,
               ra_lw_physics=0, ra_sw_physics=0)
    assert restart.unidentifiable_checkpoint_schemes(cfg) == []


def test_the_coupled_radiation_policy_table_is_gated(monkeypatch):
    """A coupled setup is named through the RADIATION_* pair, both halves.

    ``_radiation_setup_identity`` looks the above-atmosphere POLICY up
    beside the algorithm, and a missing policy row raises exactly like a
    missing algorithm row.
    """
    policies = dict(restart.RADIATION_ABOVE_ATMOSPHERE_POLICIES)
    policies.pop(90)
    monkeypatch.setattr(checkpoint_identity,
                        "RADIATION_ABOVE_ATMOSPHERE_POLICIES", policies)

    cfg = _cfg(moist=True, ra_lw_physics=90, ra_sw_physics=90,
               sf_sfclay_physics=91, sf_surface_physics=4, bl_pbl_physics=11)
    gaps = restart.unidentifiable_checkpoint_schemes(cfg)
    assert gaps == [
        "radiation above-atmosphere policy scheme 90 (ra_lw_physics) has "
        "no row in the checkpoint identity table"]


def test_a_mixed_radiation_pair_is_gated_on_the_split_tables(monkeypatch):
    """A lw != sw setup is named through LONGWAVE_*/SHORTWAVE_*.

    Which tables a radiation setup is looked up in is decided by the
    setup, so the gate follows the writer's branch instead of always
    asking about one pair.
    """
    policies = dict(restart.SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES)
    policies.pop(1)
    monkeypatch.setattr(checkpoint_identity,
                        "SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES", policies)

    cfg = _cfg(moist=True, ra_lw_physics=4, ra_sw_physics=1)
    gaps = restart.unidentifiable_checkpoint_schemes(cfg)
    assert gaps == [
        "shortwave above-atmosphere policy scheme 1 (ra_sw_physics) has "
        "no row in the checkpoint identity table"]


def test_the_accepted_registries_are_the_loader_menus():
    """The tuples this suite iterates are the ones the loader gates on."""
    from woof.microphysics_schemes import NAMED_MP_IDS
    # 5 is Ferrier-Aligo (its own identity row).  30 is FAST_SBM, which joined with its own restart identity in 2.8.9.
    assert config_mod.MP_PHYSICS_ACCEPTED == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 50, 51, 52, 53, 55, 56, 95, 96, 97, 106, 900)
    assert config_mod.RA_LW_PHYSICS_ACCEPTED == (0, 1, 3, 4, 5, 7, 31, 90, 99)
    assert config_mod.RA_SW_PHYSICS_ACCEPTED == (0, 1, 2, 3, 4, 5, 7, 90, 99)
    validate_run_config(_cfg(moist=True, mp_physics=55))
    with pytest.raises(ValueError, match="mp_physics must be"):
        validate_run_config(_cfg(moist=True, mp_physics=9999))
    with pytest.raises(ValueError, match="ra_lw_physics must be"):
        validate_run_config(_cfg(ra_lw_physics=6, ra_sw_physics=0))


# ---------------------------------------------------------------------------
# Rung 3: one copy of each table, in a module a preparation-only install has.
# ---------------------------------------------------------------------------

_GATED_TABLES = (
    "CUMULUS_ALGORITHM_IDENTITIES",
    "LAND_SURFACE_ALGORITHM_IDENTITIES",
    "LAND_SURFACE_PARAMETER_SOURCES",
    "LONGWAVE_ABOVE_ATMOSPHERE_POLICIES",
    "LONGWAVE_ALGORITHM_IDENTITIES",
    "MICROPHYSICS_ALGORITHM_IDENTITIES",
    "PBL_ALGORITHM_IDENTITIES",
    "RADIATION_ABOVE_ATMOSPHERE_POLICIES",
    "RADIATION_ALGORITHM_IDENTITIES",
    "SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES",
    "SHORTWAVE_ALGORITHM_IDENTITIES",
    "SURFACE_LAYER_ALGORITHM_IDENTITIES",
)


@pytest.mark.parametrize("name", _GATED_TABLES)
def test_the_restart_spelling_is_the_same_object_not_a_copy(name):
    """One table, two names for it.

    Every reader in the tree spells these through ``woof.io.restart``,
    which is where they were written.  A second copy there would be a
    table that can disagree with the one plan review reads -- a run
    refused for a row the writer has, or admitted for one it does not.
    """

    assert getattr(restart, name) is getattr(checkpoint_identity, name)


def test_the_gate_module_reaches_neither_restart_nor_the_device():
    """The property that lets plan review run in a preparation-only install.

    ``woof/config.py`` calls this gate, and it ships where ``woof/io``
    does not.  Read off the source rather than off ``sys.modules``, so an
    import added inside a function -- which no import-time check would see
    -- fails here too.
    """

    import woof.checkpoint_identity as module

    tree = ast.parse(inspect.getsource(module))
    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            reached.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            reached.add(node.module)
    forbidden = sorted(
        name for name in reached
        if name == "cupy" or name.startswith("cupy.")
        or name == "woof.io" or name.startswith("woof.io."))
    assert forbidden == [], (
        "woof/checkpoint_identity.py is imported by "
        "woof.config.validate_run_config, which runs in distributions "
        f"that stage neither of these: {forbidden}")
    # And what it DOES reach is reachable there: the two configuration
    # helpers the radiation branch needs, resolved inside that function.
    assert {"woof.config", "woof.physics_compat"} <= reached


@pytest.mark.parametrize("surface,albedo", [(0, 3), (4, 2), (4, 3)])
@pytest.mark.parametrize("name,value", [
    ("snicar_snowshape_opt", 1), ("snicar_snowbc_intmix", False),
    ("snicar_snowdust_intmix", True), ("snicar_bc_ppb", 2.0),
    ("snicar_dust_ppm", 1.0), ("snicar_ice_density", 850.0),
    ("snicar_bubble_radius_um", 800.0),
])
def test_snow_controls_preserve_inactive_headers_and_bind_active_resumes(
        surface, albedo, name, value, tmp_path):
    cfg = _cfg(sf_surface_physics=surface, opt_alb=albedo)
    changed = dataclasses.replace(cfg, **{name: value})
    echo = restart.configuration_echo(cfg)
    fingerprint = restart._configuration_fingerprint(cfg)
    if surface == 4 and albedo == 3:
        assert name in echo
        assert restart._configuration_fingerprint(changed) != fingerprint
        with pytest.raises(restart.RestartMismatchError, match=name):
            restart._require_config_match(echo, changed, tmp_path)
    else:
        assert name not in echo
        assert restart.configuration_echo(changed) == echo
        assert restart._configuration_fingerprint(changed) == fingerprint
        restart._require_config_match(echo, changed, tmp_path)
