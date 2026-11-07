"""A nest born mid-run gets its land state the way WRF gives one.

THE PROBLEM.  A spawn-at-trigger nest is materialized from the LIVE
parent (that is the point: it is born inside the storm its trigger saw)
on its OWN-GRID statics.  Neither ingredient carries soil.  The parent's
soil is on the parent's grid; the own-grid statics are climatological
ground, not state.  Until this landed, the real-data spawn route refused
rather than integrate a nest on undefined land.

THE PRACTICE, and it is not WOOF's invention.  WRF has exactly two
routes for a nest that starts later than its parent (Users' Guide chapter
5, "Nesting"):

* ``fine_input_stream = 2`` -- 3-D meteorology interpolated from the
  parent, static AND MASKED SURFACE fields (soil temperature and moisture
  among them) read from the nest's own ``wrfinput``.  That file is a
  ``real.exe`` product at the nest's footprint and start time, so the
  route presupposes knowing both in advance.
* ``input_from_file = .false.`` -- "the model interpolates all variables
  required in the nest from the coarse domain fields".

A trigger-spawned nest picks its footprint mid-run, from a storm the
analysis does not contain, so the first route cannot exist for it.  The
second is therefore the route -- and it is the same operator WRF runs
unconditionally at EVERY nest birth (``med_nest_initial`` calls
``med_interp_domain(parent, nest)`` before any input file is consulted,
share/mediation_integrate.F:670) and, via the identical call after each
``shift_domain_em``, at every moving-nest leading edge
(share/mediation_nest_move.F:186).  The surface/soil family does not go
through a plain interpolator on that path: the Registry names a
landmask-aware one per field
(``i02rhd=(interp_mask_field:lu_index,iswater)``, Registry.EM_COMMON:790
and :839-841/:868-872/:1417; ``,isice`` for XICE at :842).

These tests pin the operator against the Fortran and the spawn-time
assembly against the receipt discipline ``donor_fill_plan`` set: every
land/water conflict COUNTED, never silent.  Everything here is host
NumPy -- the operator is a one-shot at birth, and a CPU test is the
accurate instrument for it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.core.nest_interp import (MASK_INTERP_BRANCHES, _donor_maps,
                                    interp_mask_field, mask_donor_index)
from woof.ingest.nest_spawn_init import (SEA_ICE_MASKED_FIELDS,
                                          SpawnInitRefusal,
                                          spawn_land_state_from_parent)
from woof.ingest.relocation_init import LAND_SURFACE_CONTINUATION_FIELDS
from woof.core.physics import DECLARED_CONSTANT_GLW_WM2  # noqa: E402

#: The idealised constant downward longwave these fixtures declare.
#:
#: ``woof.core.physics.initialize_physics`` no longer defaults ``glw``
#: (300.0 through 1.8.7): a land-surface suite with no longwave scheme
#: must state where its downward longwave comes from instead of being
#: handed a plausible-looking 300 W m-2 nobody chose.  These are
#: idealised columns; the constant is the right answer for them and this
#: is where they say so.  The VALUE is 1.8.7's default, so every fixture
#: below integrates exactly the numbers it always did.
_IDEALISED_GLW = DECLARED_CONSTANT_GLW_WM2

ISWATER = 17          # the MODIS water category, the usual iswater
ISICE = 15
LAND_CAT = 5.0
_PNY = _PNX = 14
_CNY = _CNX = 9
_RATIO = 3


def _parent_landuse(*, coast_col=None, water=False, ice=False):
    """All land, all water, or land east of ``coast_col``."""
    if coast_col is None:
        value = ISICE if ice else ISWATER if water else LAND_CAT
        return np.full((_PNY, _PNX), float(value))
    return (np.where(np.arange(_PNX)[None, :] < coast_col, ISWATER, LAND_CAT)
            * np.ones((_PNY, 1)))


def _child_landuse(value=LAND_CAT):
    return np.full((_CNY, _CNX), float(value))


# ---------------------------------------------------------------------------
# The operator: interp_fcn.F:4075-4275, transcribed by hand
# ---------------------------------------------------------------------------

def test_mask_donor_index_is_the_fortran_pickup_arithmetic():
    """interp_fcn.F:4140-4155, both parities.

    Odd ratios land the middle child cell exactly on the parent centre
    (offset 0); even ratios never do.  This is NOT the SINT donor map --
    that one takes the CONTAINING cell, and the difference is why both
    are transliterated rather than shared.
    """
    donor, frac = mask_donor_index(9, 3, 3)
    assert donor.tolist() == [1, 2, 2, 2, 3, 3, 3, 4, 4]
    assert np.allclose(frac, [2 / 3, 0, 1 / 3, 2 / 3, 0, 1 / 3,
                              2 / 3, 0, 1 / 3])
    donor2, frac2 = mask_donor_index(6, 2, 3)
    assert donor2.tolist() == [1, 2, 2, 3, 3, 4]
    assert np.allclose(frac2, [0.75, 0.25, 0.75, 0.25, 0.75, 0.25])
    sint_ci, _ip = _donor_maps(9, 3, 3, 0, _PNX, "x")
    assert sint_ci.tolist() == [2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert sint_ci.tolist() != donor.tolist()


def test_mask_field_is_bitwise_bilinear_where_the_classes_agree():
    """Branch 1 (:4224-4232), evaluated in the field's own FP32 as WRF's
    REAL arithmetic does -- bitwise, not merely close."""
    rng = np.random.default_rng(11)
    parent = rng.uniform(280.0, 300.0, (_PNY, _PNX)).astype(np.float32)
    out, counts = interp_mask_field(
        parent, nri=_RATIO, nrj=_RATIO, i_parent_start=3, j_parent_start=3,
        child_landuse=_child_landuse(), parent_landuse=_parent_landuse(),
        flag_category=ISWATER)
    assert counts["same_class_bilinear"] == _CNY * _CNX
    assert counts["class_matched_average"] == 0
    assert counts["opposite_class_bilinear"] == 0
    assert out.dtype == np.float32

    ci, dx = mask_donor_index(_CNX, _RATIO, 3)
    cj, dy = mask_donor_index(_CNY, _RATIO, 3)
    one = np.float32(1.0)
    for j, i in ((0, 0), (4, 7), (8, 8), (3, 1)):
        a, b = np.float32(dx[i]), np.float32(dy[j])
        cix, cjy = ci[i], cj[j]
        manual = ((one - a) * ((one - b) * parent[cjy, cix]
                               + b * parent[cjy + 1, cix])
                  + a * ((one - b) * parent[cjy, cix + 1]
                         + b * parent[cjy + 1, cix + 1]))
        assert manual.view(np.uint32) == out[j, i].view(np.uint32)


def test_mask_field_never_pulls_land_state_across_a_coast():
    """Branch 2 (:4266-4290), and the whole reason the mask exists.

    A child that is all land over a parent coast must take LAND values
    wherever a land corner exists.  A plain bilinear blends the sea in and
    hands Noah a soil column 25 K wrong; the negative control shows it
    doing exactly that over the same geometry.
    """
    plu = _parent_landuse(coast_col=7)
    parent = np.where(plu == ISWATER, 275.0, 300.0).astype(np.float32)
    common = dict(nri=_RATIO, nrj=_RATIO, i_parent_start=6, j_parent_start=4,
                  child_landuse=_child_landuse(), flag_category=ISWATER)
    out, counts = interp_mask_field(parent, parent_landuse=plu, **common)
    assert counts["class_matched_average"] > 0, (
        "the fixture must straddle the parent coast or it proves nothing")
    assert counts["same_class_bilinear"] > 0
    assert set(np.unique(out).tolist()) == {275.0, 300.0}

    plain, plain_counts = interp_mask_field(
        parent, parent_landuse=np.full_like(plu, LAND_CAT), **common)
    assert plain_counts["class_matched_average"] == 0
    blended = (plain > 275.0) & (plain < 300.0)
    assert blended.any(), "negative control never blended; fixture is inert"


def test_mask_field_counts_the_class_conflict_instead_of_hiding_it():
    """The island/lake case, where WRF's own comment (:4232) says it has
    "no better way".  WOOF does the same thing and COUNTS it."""
    parent = np.full((_PNY, _PNX), 300.0, dtype=np.float32)
    out, counts = interp_mask_field(
        parent, nri=_RATIO, nrj=_RATIO, i_parent_start=3, j_parent_start=3,
        child_landuse=_child_landuse(ISWATER),
        parent_landuse=_parent_landuse(), flag_category=ISWATER)
    cells = _CNY * _CNX
    assert counts["opposite_class_bilinear"] == cells
    assert counts["same_class_bilinear"] == 0
    assert counts["child_flag_class_cells"] == cells
    assert np.allclose(out, 300.0)
    assert sum(counts[name] for name in MASK_INTERP_BRANCHES) == counts["cells"]


def test_mask_field_decides_once_per_column_for_a_soil_stack():
    """WRF's nk loop re-decides nothing: the mask is 2-D, the field is not."""
    plu = _parent_landuse(coast_col=7)
    layer = np.where(plu == ISWATER, 275.0, 300.0).astype(np.float32)
    stack = np.stack([layer + 2.0 * k for k in range(4)]).astype(np.float32)
    common = dict(nri=_RATIO, nrj=_RATIO, i_parent_start=6, j_parent_start=4,
                  child_landuse=_child_landuse(), parent_landuse=plu,
                  flag_category=ISWATER)
    flat, flat_counts = interp_mask_field(layer, **common)
    out, counts = interp_mask_field(stack, **common)
    assert out.shape == (4, _CNY, _CNX)
    assert counts == flat_counts
    for k in range(4):
        assert np.array_equal(out[k], flat + np.float32(2.0 * k))


def test_mask_field_masks_sea_ice_on_its_own_flag_category():
    """XICE carries ``interp_mask_field:lu_index,isice`` (Registry:842),
    not iswater.  Same operator, same geometry, different flag, and it
    must reach a different answer -- three parent classes are needed to
    see it, because a two-class fixture makes the two flags degenerate.
    """
    column = np.arange(_PNX)[None, :]
    plu = (np.where(column < 6, ISICE,
                    np.where(column < 9, ISWATER, LAND_CAT))
           * np.ones((_PNY, 1)))
    parent = np.where(plu == ISICE, 1.0,
                      np.where(plu == ISWATER, 0.5, 0.0)).astype(np.float32)
    common = dict(nri=_RATIO, nrj=_RATIO, i_parent_start=9, j_parent_start=4,
                  child_landuse=_child_landuse(ISICE), parent_landuse=plu)
    on_ice, ice_counts = interp_mask_field(
        parent, flag_category=ISICE, **common)
    on_water, water_counts = interp_mask_field(
        parent, flag_category=ISWATER, **common)
    assert ice_counts != water_counts
    assert not np.array_equal(on_ice, on_water)
    # Keyed on ISICE the child's class is absent from every coarse cell,
    # so WRF's uniform branch blends water and land.  Keyed on ISWATER the
    # child is simply not-water, so the mixed cells take the land corners
    # alone and never blend the sea in.
    assert ice_counts["opposite_class_bilinear"] == _CNY * _CNX
    assert water_counts["class_matched_average"] > 0
    assert ((on_ice > 0.0) & (on_ice < 0.5)).any()
    assert set(np.unique(on_water).tolist()) == {0.0, 0.5}


def test_mask_field_refuses_an_off_grid_placement():
    with pytest.raises(ValueError, match="outside the parent extent"):
        interp_mask_field(
            np.zeros((_PNY, _PNX), np.float32), nri=_RATIO, nrj=_RATIO,
            i_parent_start=12, j_parent_start=3,
            child_landuse=_child_landuse(),
            parent_landuse=_parent_landuse(), flag_category=ISWATER)


def test_mask_field_refuses_a_mask_that_does_not_match_its_field():
    with pytest.raises(ValueError, match="parent land-use mask"):
        interp_mask_field(
            np.zeros((10, 10), np.float32), nri=_RATIO, nrj=_RATIO,
            i_parent_start=3, j_parent_start=3,
            child_landuse=_child_landuse(),
            parent_landuse=_parent_landuse(), flag_category=ISWATER)


# ---------------------------------------------------------------------------
# The spawn-time assembly
# ---------------------------------------------------------------------------

def _driver_fields(*, layers=4, omit=()):
    """A parent driver's land inventory, each field distinctly valued."""
    rng = np.random.default_rng(5)
    fields = {}
    for index, name in enumerate(LAND_SURFACE_CONTINUATION_FIELDS):
        if name in omit:
            continue
        shape = ((layers, _PNY, _PNX)
                 if name in ("tslb", "smois", "sh2o", "smcrel")
                 else (_PNY, _PNX))
        fields[name] = (rng.uniform(0.0, 1.0, shape).astype(np.float32)
                        + np.float32(index))
    return fields


def _parent_node(fields, *, grid_id=1):
    return SimpleNamespace(
        cfg=SimpleNamespace(grid_id=grid_id),
        state=SimpleNamespace(physics=SimpleNamespace(fields=fields)))


def _child_dc(*, i=6, j=4, grid_id=2):
    return SimpleNamespace(run=SimpleNamespace(sf_surface_physics=2, sf_lake_physics=0),
                           grid_id=grid_id, parent_grid_ratio=_RATIO,
                           i_parent_start=i, j_parent_start=j)


_ATTRS = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
          "ISWATER": ISWATER, "ISICE": ISICE, "ISLAKE": 21}


def _spawn_land(*, child_lu=None, parent_lu=None, fields=None, **kwargs):
    return spawn_land_state_from_parent(
        _child_dc(**kwargs), _parent_node(fields or _driver_fields()),
        static_fields={"LU_INDEX": (_child_landuse() if child_lu is None
                                    else child_lu)},
        parent_static_fields={"LU_INDEX": (_parent_landuse(coast_col=7)
                                           if parent_lu is None
                                           else parent_lu)},
        landuse_attrs=_ATTRS)


def test_spawn_land_state_covers_the_whole_continuation_inventory():
    """Every field a relocation carries, a birth interpolates.  The two
    events differ in WHERE the state comes from, never in WHICH state."""
    product = _spawn_land()
    fields, receipt = product["fields"], product["receipt"]
    assert set(fields) == set(LAND_SURFACE_CONTINUATION_FIELDS)
    assert receipt["fields_absent"] == []
    assert receipt["fields_shape_skipped"] == {}
    for name, value in fields.items():
        expected = ((4, _CNY, _CNX)
                    if name in ("tslb", "smois", "sh2o", "smcrel")
                    else (_CNY, _CNX))
        assert value.shape == expected, name
        assert np.isfinite(value).all(), name
    assert receipt["operator"].startswith("interp_mask_field")
    assert receipt["parent_grid_ratio"] == _RATIO
    assert receipt["placement"] == [6, 4]
    assert receipt["accumulators_reinitialized"] is True


def test_the_sea_ice_family_is_keyed_on_isice_and_the_rest_on_iswater():
    """The Registry spells the flag per field; the receipt shows both
    families keyed separately rather than one blanket mask."""
    receipt = _spawn_land()["receipt"]
    by_flag = receipt["by_mask_flag"]
    assert set(by_flag) == {"iswater", "isice"}
    assert by_flag["isice"]["flag_category"] == ISICE
    assert by_flag["iswater"]["flag_category"] == ISWATER
    assert set(by_flag["isice"]["fields"]) == (
        SEA_ICE_MASKED_FIELDS & set(LAND_SURFACE_CONTINUATION_FIELDS))
    assert "tslb" in by_flag["iswater"]["fields"]
    assert "xice" not in by_flag["iswater"]["fields"]


def test_the_land_class_conflicts_are_counted_not_swallowed():
    """A child whose finer statics resolve land where the parent said sea
    has cells with no same-class donor.  They are reported, with a number.
    """
    all_sea = _parent_landuse(water=True)
    conflicted = _spawn_land(parent_lu=all_sea)["receipt"]
    assert conflicted["land_class_conflict_cells"] > 0
    assert (conflicted["by_mask_flag"]["iswater"]["counts"]
            ["opposite_class_bilinear"] == _CNY * _CNX)
    # Control: same child over a parent that agrees with it has none.
    agreed = _spawn_land(parent_lu=_parent_landuse())["receipt"]
    assert agreed["land_class_conflict_cells"] == 0
    assert (agreed["by_mask_flag"]["iswater"]["counts"]
            ["same_class_bilinear"] == _CNY * _CNX)


def test_a_field_the_parent_never_allocated_is_named_not_invented():
    """The transplant's rule, kept: skipped by name, listed in the receipt."""
    product = _spawn_land(fields=_driver_fields(omit=("smcrel", "snotime")))
    assert product["receipt"]["fields_absent"] == ["smcrel", "snotime"]
    assert "smcrel" not in product["fields"]
    assert "tslb" in product["fields"]


def test_a_six_layer_soil_column_rides_through_whole():
    """RUC's nine layers and Noah's four are the same operator: the layer
    axis is not the mask's business."""
    product = _spawn_land(fields=_driver_fields(layers=9))
    assert product["fields"]["tslb"].shape == (9, _CNY, _CNX)
    assert product["fields"]["tsk"].shape == (_CNY, _CNX)


def test_a_field_off_the_parent_mass_grid_is_skipped_by_name():
    fields = _driver_fields()
    fields["ust"] = np.zeros((_PNY, _PNX + 1), dtype=np.float32)
    product = _spawn_land(fields=fields)
    assert product["receipt"]["fields_shape_skipped"] == {
        "ust": [_PNY, _PNX + 1]}
    assert "ust" not in product["fields"]


def test_spawn_land_state_refuses_a_parent_with_no_driver():
    with pytest.raises(SpawnInitRefusal, match="no physics driver"):
        spawn_land_state_from_parent(
            _child_dc(),
            SimpleNamespace(cfg=SimpleNamespace(grid_id=1),
                            state=SimpleNamespace(physics=None)),
            static_fields={"LU_INDEX": _child_landuse()},
            parent_static_fields={"LU_INDEX": _parent_landuse()},
            landuse_attrs=_ATTRS)


@pytest.mark.parametrize("missing", ["child", "parent"])
def test_spawn_land_state_refuses_without_land_use_categories(missing):
    """WRF's masked interpolator keys on LU_INDEX; without one there is no
    defined behaviour, so this refuses instead of falling back to plain."""
    statics = {"LU_INDEX": _child_landuse()}
    parent_statics = {"LU_INDEX": _parent_landuse()}
    if missing == "child":
        statics = {"HGT_M": np.zeros((_CNY, _CNX))}
    else:
        parent_statics = {"HGT_M": np.zeros((_PNY, _PNX))}
    with pytest.raises(SpawnInitRefusal, match="no LU_INDEX"):
        spawn_land_state_from_parent(
            _child_dc(), _parent_node(_driver_fields()),
            static_fields=statics, parent_static_fields=parent_statics,
            landuse_attrs=_ATTRS)


def test_the_interpolated_state_is_the_parents_where_they_agree():
    """The calibration point: a uniform-class parent field comes across
    unchanged in value, so nothing about the mask perturbs a clean case."""
    fields = _driver_fields()
    fields["tslb"] = np.full((4, _PNY, _PNX), 287.5, dtype=np.float32)
    fields["tsk"] = np.full((_PNY, _PNX), 291.25, dtype=np.float32)
    product = _spawn_land(parent_lu=_parent_landuse(), fields=fields)
    assert np.all(product["fields"]["tslb"] == np.float32(287.5))
    assert np.all(product["fields"]["tsk"] == np.float32(291.25))


# ---------------------------------------------------------------------------
# The birth certificate: the land accounting must reach the receipt
# ---------------------------------------------------------------------------

def test_the_spawn_receipt_carries_the_preparers_land_accounting():
    """The preparer owns the land half of the birth, so its receipt has to
    travel -- through the same duck-typed ``last_receipt`` seam the
    relocation runner reads.  A route with no preparer carries None rather
    than failing, which is what keeps the idealized path unaffected.
    """
    from test_nest_spawn_init import _experiment, _live_parent

    from woof.ingest.nest_spawn_init import spawn_child_from_parent

    exp = _experiment()
    parent, _grids = _live_parent(exp)
    child_dc = exp.domains[1]

    class _Preparer:
        last_receipt = {"land_surface": {"land_class_conflict_cells": 7},
                        "driver_rebuild_seconds": 0.5}

        def __call__(self, initialized, dc, parent_node):
            assert initialized.state is not None

    receipt = spawn_child_from_parent(
        child_dc, parent, array_module=np, on_child_built=_Preparer())
    assert receipt["land_surface"] == _Preparer.last_receipt
    assert receipt["land_surface"]["land_surface"][
        "land_class_conflict_cells"] == 7

    bare = spawn_child_from_parent(
        child_dc, parent, array_module=np, on_child_built=lambda *a: None)
    assert bare["land_surface"] is None


@requires_gpu
def test_the_newborns_driver_comes_up_on_the_card_carrying_parent_soil():
    """The seam that was refusing, run on the artifact.

    Not a mock: a real ``initialize_physics`` allocation on the device,
    fed the parent-interpolated soil a spawn produces, on a child grid
    that straddles the parent's coast.  The device arrays must carry the
    masked answer bitwise -- and must DIFFER from the plain-bilinear one,
    or the mask rode along without doing anything (the exact-0.0-delta
    trap).
    """
    import cupy as cp

    from woof.config import validate_run_config
    from woof.core.physics import initialize_physics
    from test_physics import _balanced_state, _mp_only_cfg, _tables_or_skip

    _tables_or_skip()
    plu = _parent_landuse(coast_col=7)
    fields = _driver_fields()
    # A coast the parent resolves and the child does not agree with: cold
    # sea, warm land, and a child that is land the whole way across.
    fields["tslb"] = np.stack([
        np.where(plu == ISWATER, 274.0 + k, 299.0 + k) for k in range(4)
    ]).astype(np.float32)
    fields["tsk"] = np.where(plu == ISWATER, 274.0, 299.0).astype(np.float32)
    product = _spawn_land(parent_lu=plu, fields=fields)
    masked = product["fields"]
    assert product["receipt"]["by_mask_flag"]["iswater"]["counts"][
        "class_matched_average"] > 0

    cfg = validate_run_config(_mp_only_cfg(
        nx=_CNX, ny=_CNY, sf_sfclay_physics=1, sf_surface_physics=2,
        num_soil_layers=4))
    state = _balanced_state(cp, cfg)
    assert tuple(state.mup.shape) == (_CNY, _CNX)
    driver = initialize_physics(
        state, cfg, tsk=masked["tsk"],
        soil_temperature=masked["tslb"], soil_moisture=masked["smois"],
        liquid_moisture=masked["sh2o"], glw=_IDEALISED_GLW)
    on_card = cp.asnumpy(driver.fields["tslb"])
    assert on_card.shape == (4, _CNY, _CNX)
    assert np.array_equal(on_card, masked["tslb"].astype(np.float32))
    assert np.array_equal(cp.asnumpy(driver.fields["tsk"]),
                          masked["tsk"].astype(np.float32))
    # The masked column never took a sea value where a land donor existed.
    assert on_card[0].min() == np.float32(274.0)      # the island cells
    assert on_card[0].max() == np.float32(299.0)

    # A/B: the same geometry with the mask disabled must land DIFFERENT
    # bytes on the card, or this test proves nothing about the mask.
    from woof.core.nest_interp import interp_mask_field

    plain, _counts = interp_mask_field(
        fields["tslb"], nri=_RATIO, nrj=_RATIO,
        i_parent_start=6, j_parent_start=4,
        child_landuse=_child_landuse(),
        parent_landuse=np.full_like(plu, LAND_CAT), flag_category=ISWATER)
    control = initialize_physics(
        state, cfg, tsk=masked["tsk"], soil_temperature=plain,
        soil_moisture=masked["smois"], liquid_moisture=masked["sh2o"],
        glw=_IDEALISED_GLW)
    delta = np.abs(cp.asnumpy(control.fields["tslb"]) - on_card)
    assert delta.max() > 1.0, (
        "masked and plain interpolation agreed exactly; the treatment "
        "never ran")


def test_the_spawn_preparer_refuses_before_it_can_invent_land_state():
    """Both inputs the masked interpolator needs are named, not defaulted:
    the newborn's own-grid categories and the parent's."""
    from woof import runtime

    initialized = SimpleNamespace(static_fields=None, grid=None, state=None)
    model = SimpleNamespace(_prepared_by_grid_id={}, _activation_context={})
    preparer = runtime.RealSpawnChildPreparer(
        exp=SimpleNamespace(), data=SimpleNamespace(), model=model)
    parent = SimpleNamespace(cfg=SimpleNamespace(grid_id=1))
    with pytest.raises(SpawnInitRefusal, match="no static fields"):
        preparer(initialized, _child_dc(), parent)

    initialized = SimpleNamespace(
        static_fields={"LU_INDEX": _child_landuse()}, grid=None, state=None)
    with pytest.raises(SpawnInitRefusal, match="no statics on record"):
        preparer(initialized, _child_dc(), parent)
