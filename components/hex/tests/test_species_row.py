"""The species row is the door's one source of truth about scalars.

These gates hold the invariants every consumer in the v841 chain will read
instead of restating: vapour at index 0, the mass-loading species as a
contiguous prefix, the class tags that keep number and volume state out of
the moisture loading, and the declared-extras hook a later lane appends
through.  The rows themselves are checked against the PINNED ENGINE's own
species table, so hex and the engine cannot drift apart silently.
"""

from __future__ import annotations

import pytest

from woof.hex import species_row as sr
from woof.hex.errors import ConfigurationRefusal


def test_only_morrisons_native_buckets_declare_signed_fallout():
    for mp in (1, 6, 8, 10, 50):
        row = sr.engine_species_row(
            scheme_name=f"test-mp{mp}", mp_physics=mp,
            species=("qv", "qc", "qr", "qi", "qs", "qg"),
            mass_species=("qv", "qc", "qr", "qi", "qs", "qg"),
            surface_accumulators=("RAINNC", "SNOWNC", "GRAUPELNC", "CUSTOM"))
        assert [v.signed_fallout for v in row.surface_accumulators] == [mp == 10]*3 + [False]
        extra = sr.SurfaceAccumulatorDeclaration("extra", "float32", "EXTRA", "extra")
        assert not row.with_extras((), accumulators=(extra,)).surface_accumulators[-1].signed_fallout


# ---------------------------------------------------------------------------
# The declared rows.
# ---------------------------------------------------------------------------


def test_wsm6_row_is_the_six_in_order():
    row = sr.WSM6_SPECIES_ROW
    assert row.names() == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert row.mp_physics == 6
    assert row.engine_scheme == "wsm6"


def test_every_wsm6_species_loads_the_air():
    """Why the weld was invisible: on WSM6 the loading set IS the row."""

    row = sr.WSM6_SPECIES_ROW
    assert row.mass_loading_names() == row.names()
    assert row.n_mass_loading == row.n_species() == 6


def test_p3_row_is_wrfs_eight_in_engine_order():
    row = sr.P3_SPECIES_ROW
    assert row.names() == (
        "qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib")
    assert row.mp_physics == 50
    assert row.engine_scheme == "p3"
    # P3's single ice category writes neither snow nor graupel.
    assert "qs" not in row.names()
    assert "qg" not in row.names()


def test_p3_loading_set_excludes_number_volume_and_the_rime_component():
    """The defect this row exists to prevent, stated as a gate.

    ni and nr are concentrations, qib is a volume, and qir is the rime mass
    held INSIDE qi.  All four ride transport; none of them may reach qtot.
    """

    row = sr.P3_SPECIES_ROW
    assert row.mass_loading_names() == ("qv", "qc", "qr", "qi")
    assert row.n_mass_loading == 4
    assert row.n_species() == 8
    for name in ("ni", "nr", "qir", "qib"):
        assert not row.declaration(name).is_mass_loading
        assert row.declaration(name).transported
    assert row.declaration("qir").species_class == sr.MASS_COMPONENT
    assert row.declaration("qib").species_class == sr.VOLUME
    assert row.names_of_class(sr.NUMBER) == ("ni", "nr")


def test_thompson_aerosol_row_is_wrfs_eleven_in_engine_order():
    """The mp=28 refusal retired the way its own text said it would.

    From 2026-09-01 this row was declared and refused because the pinned
    engine carried no ``thompson_aero`` row; gpuwm 2.6.3 published one, the
    two-rows gate below failed, and the row took the engine's scheme name
    and the engine's number-block order.  What stays gated: six loading
    masses first, five number species after, none of them in qtot.
    """

    row = sr.THOMPSON_AEROSOL_SPECIES_ROW
    assert row.mp_physics == 28
    assert row.engine_scheme == "thompson_aero"
    assert row.require_engine_scheme() == "thompson_aero"
    assert row.names() == (
        "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa")
    assert row.n_species() == 11
    assert row.n_mass_loading == 6
    assert row.mass_loading_names() == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert row.names_of_class(sr.NUMBER) == ("ni", "nr", "nc", "nwfa", "nifa")
    for name in ("ni", "nr", "nc", "nwfa", "nifa"):
        assert not row.declaration(name).is_mass_loading
        assert row.declaration(name).transported
    # mp=28 has a graupel category: three buckets, WSM6's three radii.
    assert [item.name for item in row.surface_accumulators] == [
        "rainnc", "snownc", "graupelnc"]
    assert row.radius_names == ("effc", "effi", "effs")


def test_the_rows_once_refused_for_want_of_an_engine_row_name_the_selector():
    """mp=1 and mp=8 were declared-and-refused while the frozen column batch
    was the only seam and built four schemes.  The generic column seam takes
    the engine's selector itself, so the rows name it and nothing refuses."""

    assert sr.KESSLER_SPECIES_ROW.require_engine_scheme() == "mp_physics=1"
    assert sr.THOMPSON_SPECIES_ROW.require_engine_scheme() == "mp_physics=8"
    for row in sr.registered_species_rows().values():
        assert row.engine_scheme is not None, row.scheme_name


def test_the_loading_prefix_is_contiguous_in_every_declared_row():
    """The invariant that lets the kernels take one int32 loop bound."""

    for name, row in sr.registered_species_rows().items():
        loading = [item.is_mass_loading for item in row.species]
        assert loading == sorted(loading, reverse=True), name
        assert row.mass_loading_names() == row.names()[: row.n_mass_loading]


def test_every_declared_species_is_transported_and_writes_under_a_name():
    for name, row in sr.registered_species_rows().items():
        history = row.history_names()
        assert set(history) == set(row.names()), name
        assert len(set(history.values())) == len(history), name
        for item in row.species:
            assert item.transported, (name, item.name)


# ---------------------------------------------------------------------------
# The row agrees with the PINNED ENGINE, or the door is building on sand.
# ---------------------------------------------------------------------------


def _pinned_engine_column_batch():
    """Grade every supported engine, while excluding unrelated checkouts."""

    from woof.hex import engine_pin

    woof = pytest.importorskip("woof")
    mcb = pytest.importorskip("woof.core.mpas_column_batch")
    version = getattr(woof, "__version__", "unknown")
    pinned = _pinned_engine_range()
    admitted = {row.version for row in engine_pin.ADMITTED_ENGINES}
    if version not in pinned and version not in admitted:
        pytest.skip(
            f"the importable woof is {version} at {woof.__file__}, which "
            f"is outside the supported engine range ({pinned}) and explicit "
            f"admissions ({sorted(admitted)}); its species table is not a "
            "supported counterparty"
        )
    return mcb


def _pinned_engine_range():
    """Read the full dependency range, including supported patch releases."""

    import re
    import pathlib

    from packaging.specifiers import SpecifierSet

    text = (pathlib.Path(__file__).resolve().parents[1]
            / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"woof(>=[0-9.]+,<[0-9.]+)"', text)
    assert match, "the packaging no longer pins woof with a >=,< range"
    return SpecifierSet(match.group(1))


def test_rows_match_the_pinned_engine_species_table():
    """Cross-lane law: grade against the other side's real writer.

    The door hands species to ``woof.core.mpas_column_batch``; if hex's row
    and the engine's ``_SPECIES_BY_SCHEME`` disagree on order or membership,
    the door feeds the seam the wrong array with no error anywhere.
    """

    table = getattr(_pinned_engine_column_batch(), "_SPECIES_BY_SCHEME")
    engine_selectors = getattr(
        _pinned_engine_column_batch(), "_MP_PHYSICS_BY_SCHEME")
    for row in sr.registered_species_rows().values():
        if row.engine_scheme is None:
            # The row is declared and refused; the engine must indeed not
            # carry it, or the refusal is stale and must be retired.
            assert row.scheme_name not in table
            assert row.mp_physics not in engine_selectors.values(), (
                f"the engine now carries a row for mp_physics={row.mp_physics}; "
                f"retire {row.scheme_name}'s engine_absent_reason and give "
                "it an engine_scheme"
            )
            continue
        if row.engine_scheme.startswith("mp_physics="):
            # A row the generic column seam runs: the engine's own plan
            # (column_physics_plan) is its species authority, not the frozen
            # batch's four-scheme table.
            continue
        if row.engine_scheme not in table and row.engine_scheme in _ENGINE_PORT_ROWS:
            # An engine before this scheme's port (TEMPO arrives with the
            # RRFS composition, after gpuwm 2.8.7).  The door refuses such a
            # backend by name before the card (forecast_door, "carries the
            # TEMPO port"); there is no engine row here to grade it against.
            continue
        assert row.names() == tuple(table[row.engine_scheme]), row.scheme_name
        assert engine_selectors[row.engine_scheme] == row.mp_physics


#: Rows whose engine scheme arrived with an engine port later than some
#: engines this tree admits.  Present in the engine, they are graded exactly
#: like every other row; absent, the door's own refusal covers them.
_ENGINE_PORT_ROWS = frozenset({"tempo"})


def test_the_engine_carries_only_the_reviewed_rows():
    """The named deferrals' own retirement condition, as a gate.

    When this fails, the engine has gained a scheme and one of the door's
    remaining refused rows (mp=1 Kessler, mp=8 Thompson) is ready to be
    retired.  It has fired once already: the 2.6.3 engine took this table
    from two rows to three and the mp=28 refusal retired on that failure.
    """

    table = getattr(_pinned_engine_column_batch(), "_SPECIES_BY_SCHEME")
    ported = sorted(_ENGINE_PORT_ROWS & set(table))
    # The engine's predicted-number P3 row is not registered by this door.
    # Keep its exact order reviewed without enabling another hex physics row.
    engine_only = {"p3_nc": ("qv", "qc", "qr", "qi", "nc", "ni", "nr", "qir", "qib")}
    for name, expected in engine_only.items():
        assert tuple(table[name]) == expected
    assert sorted(table) == sorted(["p3", "thompson_aero", "wsm6", *ported, *engine_only]), (
        "the PINNED engine has gained a microphysics row.  Re-check every "
        "declared row's species order against it, then retire that row's "
        "engine_absent_reason and give it an engine_scheme")


# ---------------------------------------------------------------------------
# Construction-time refusals.
# ---------------------------------------------------------------------------


def _decl(name, species_class, history="X"):
    return sr.SpeciesDeclaration(
        name=name,
        species_class=species_class,
        transported=True,
        history_name=history,
        description="test species",
    )


def test_a_row_that_does_not_start_with_vapour_is_refused():
    with pytest.raises(ValueError, match="index 0"):
        sr.MicrophysicsSpeciesRow(
            scheme_name="mp_test",
            mp_physics=999,
            engine_scheme="test",
            species=(_decl("qc", sr.MASS_LOADING),
                     _decl("qv", sr.MASS_LOADING)),
            surface_accumulators=(),
            radius_names=(),
            restart_extra_names=(),
        )


def test_a_row_with_a_loading_species_after_a_number_is_refused():
    """The silent-omission defect the prefix invariant prevents."""

    with pytest.raises(ValueError, match="contiguous LEADING run"):
        sr.MicrophysicsSpeciesRow(
            scheme_name="mp_test",
            mp_physics=999,
            engine_scheme="test",
            species=(
                _decl("qv", sr.MASS_LOADING),
                _decl("ni", sr.NUMBER),
                _decl("qc", sr.MASS_LOADING),
            ),
            surface_accumulators=(),
            radius_names=(),
            restart_extra_names=(),
        )


def test_an_unknown_species_class_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        _decl("qx", "mass")


def test_a_row_with_no_engine_and_no_reason_is_refused():
    with pytest.raises(ValueError, match="no reason"):
        sr.MicrophysicsSpeciesRow(
            scheme_name="mp_test",
            mp_physics=999,
            engine_scheme=None,
            species=(_decl("qv", sr.MASS_LOADING),),
            surface_accumulators=(),
            radius_names=(),
            restart_extra_names=(),
        )


def test_a_repeated_species_name_is_refused():
    with pytest.raises(ValueError, match="repeats a species name"):
        sr.MicrophysicsSpeciesRow(
            scheme_name="mp_test",
            mp_physics=999,
            engine_scheme="test",
            species=(_decl("qv", sr.MASS_LOADING),
                     _decl("qv", sr.MASS_LOADING)),
            surface_accumulators=(),
            radius_names=(),
            restart_extra_names=(),
        )


# ---------------------------------------------------------------------------
# Declared extras: the hook a later lane adds scalars through.
# ---------------------------------------------------------------------------


def test_an_extra_scalar_appends_without_moving_the_schemes_indices():
    base = sr.P3_SPECIES_ROW
    extra = _decl("ntest", sr.NUMBER, history="QNTEST")
    extended = base.with_extras(species=(extra,))
    assert extended.names() == base.names() + ("ntest",)
    for name in base.names():
        assert extended.index_of(name) == base.index_of(name)
    assert extended.n_species() == base.n_species() + 1
    # The whole point: the moisture loading is untouched.
    assert extended.n_mass_loading == base.n_mass_loading
    assert extended.mass_loading_names() == base.mass_loading_names()


def test_an_extra_is_marked_as_declared_and_findable():
    extra = _decl("ntest", sr.NUMBER, history="QNTEST")
    extended = sr.WSM6_SPECIES_ROW.with_extras(species=(extra,))
    assert [item.name for item in extended.declared_extras()] == ["ntest"]
    assert extended.declaration("ntest").declared_extra
    # A scheme's own species are never marked as extras.
    assert not extended.declaration("qv").declared_extra


def test_a_mass_loading_extra_is_refused_by_name():
    """A loading extra would be transported and silently left out of qtot."""

    with pytest.raises(ConfigurationRefusal) as excinfo:
        sr.WSM6_SPECIES_ROW.with_extras(
            species=(_decl("qextra", sr.MASS_LOADING),))
    message = str(excinfo.value)
    assert "outside that prefix" in message
    assert "silently omitted from qtot" in message


def test_an_extra_surface_accumulator_may_be_fp64():
    """A bucket integrating a rate over a forecast needs the width."""

    accumulator = sr.SurfaceAccumulatorDeclaration(
        name="testmass",
        dtype="float64",
        history_name="TESTMASS",
        description="test accumulated mass",
    )
    extended = sr.WSM6_SPECIES_ROW.with_extras(accumulators=(accumulator,))
    extras = extended.declared_extra_accumulators()
    assert [item.name for item in extras] == ["testmass"]
    assert extras[0].dtype == "float64"
    assert extras[0].declared_extra
    # The scheme's own accumulators are unchanged and still first.
    assert extended.surface_accumulators[: len(
        sr.WSM6_SPECIES_ROW.surface_accumulators)] == \
        sr.WSM6_SPECIES_ROW.surface_accumulators


def test_extras_do_not_mutate_the_registered_row():
    before = sr.WSM6_SPECIES_ROW.names()
    sr.WSM6_SPECIES_ROW.with_extras(
        species=(_decl("ntest", sr.NUMBER, history="QNTEST"),))
    assert sr.WSM6_SPECIES_ROW.names() == before
    assert sr.registered_species_rows()["mp_wsm6"].names() == before


def test_an_accumulator_of_an_unknown_width_is_refused():
    with pytest.raises(ValueError, match="only 'float32' and 'float64'"):
        sr.SurfaceAccumulatorDeclaration(
            name="testmass",
            dtype="float16",
            history_name="TESTMASS",
            description="test",
        )


# ---------------------------------------------------------------------------
# Lookup.
# ---------------------------------------------------------------------------


def _column_suite_only_schemes() -> dict[str, int]:
    """Rows the column-suite seam carries but the MPAS-name seam never routes.

    TEMPO (WOOF id 900) reaches the run only through the column-suite table
    (``mpas-physics-suites.v1.json``, "column-suite seam only"); MPAS-A has no
    spelling for it and ``physics_seam`` has no route.  Derived from the
    table rather than listed, so a new column-suite scheme is classified by
    the same data that admits it.
    """

    from woof.hex import physics_seam as ps
    from woof.hex import physics_suites

    routed_names = set(ps._ROUTES["config_microp_scheme"])
    return {
        name: sr.species_row_for_scheme(name).mp_physics
        for name in physics_suites._microphysics_spellings()
        if name not in routed_names
    }


def test_the_row_table_is_exhaustive_over_the_seams_routed_domain():
    """Every scheme the seams can route to has a row, and vice versa.

    The seam refuses a selector with no requirement row because silence
    there declares qv alone -- a state substitution.  The door has the same
    hazard one layer down: a routed scheme with no species row would reach
    the door and find nothing to size its scalar block from.  Deriving both
    from the same table is only safe while the table covers the domain.

    The domain is the MPAS-name seam's routes plus the column-suite seam's
    own schemes (TEMPO).  A declared row reachable from neither seam is an
    orphan and still fails here; that is what the RRFS merge's unconditional
    TEMPO row looked like to the old one-seam form of this test.
    """

    from woof.hex import physics_seam as ps

    routed = {int(value) for value in
              ps._ROUTES["config_microp_scheme"].values()}
    routed.discard(0)  # 'off' transports nothing and needs no row
    column_only = set(_column_suite_only_schemes().values())
    assert not routed & column_only
    # The rows WRITTEN in species_row.py.  A row derived from the engine's
    # plan (generic column seam) is the engine's inventory, not this
    # table's, and is held by tests/test_arbitrary_physics.py.
    declared = {row.mp_physics
                for row in sr.declared_species_rows().values()}
    assert declared == routed | column_only


def test_every_row_matches_the_seams_own_requirement_ladder():
    """The two answers to "which scalars" must be one answer.

    ``physics_seam._required_scalar_names`` is the seam's hand-written
    ladder and this table was derived independently from the engine's
    species table and WRF's mp driver arms.  They agree today; this gate is
    what keeps them agreeing.

    A column-suite-only row has no ladder row by construction; the hazard
    the ladder exists for (a selection accepted declaring qv alone) is then
    closed by the MPAS-name seam refusing that name outright, which is
    asserted instead, together with the row naming its engine scheme.
    """

    from woof.hex import physics_seam as ps

    column_only = _column_suite_only_schemes()
    for name, row in sr.declared_species_rows().items():
        if name in column_only:
            assert row.engine_scheme, name
            with pytest.raises(ConfigurationRefusal, match=name):
                ps.resolve_mpas_physics(config_microp_scheme=name)
            continue
        selection = ps.resolve_mpas_physics(config_microp_scheme=name)
        assert ps._required_scalar_names(selection) ==             row.required_scalar_names(), name


def test_a_block_with_extras_resolves_to_its_LONGEST_matching_row():
    """First-match would resolve an aerosol block to the warm-rain row.

    mp_kessler declares (qv, qc, qr), which is a prefix of every other row.
    A first-match resolver hands an eleven-species block plus a lane's
    appended scalars back as the three-species row, and the caller then
    sums three species into qtot and prepares three of eleven -- with no
    error anywhere.  Measured against the in-tree provider block, which is
    exactly that shape.
    """

    base = sr.THOMPSON_AEROSOL_SPECIES_ROW
    block = base.names() + ("extra_a", "extra_b")
    assert sr.species_row_for_names(block) is base
    assert sr.species_row_for_names(base.names()) is base
    # The short row still resolves for its own block and for its own extras.
    kessler = sr.KESSLER_SPECIES_ROW
    assert sr.species_row_for_names(kessler.names()) is kessler
    assert sr.species_row_for_names(
        kessler.names() + ("nextra",)) is kessler


def test_lookup_by_scheme_name_and_selector_agree():
    for row in sr.registered_species_rows().values():
        assert sr.species_row_for_scheme(row.scheme_name) is row
        assert sr.species_row_for_mp_physics(row.mp_physics) is row


def test_an_undeclared_scheme_is_refused_with_the_declared_set_named():
    with pytest.raises(ConfigurationRefusal) as excinfo:
        sr.species_row_for_scheme("mp_morrison")
    assert "mp_wsm6" in str(excinfo.value)


def test_an_undeclared_selector_is_refused():
    with pytest.raises(ConfigurationRefusal):
        sr.species_row_for_mp_physics(987654)
