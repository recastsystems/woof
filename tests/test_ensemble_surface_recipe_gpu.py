"""Every surface-state member matches its standalone GPU initialization."""
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _state(cp, *, shape=(3, 5), frozen=False):
    fields = {"landmask": cp.ones(shape, dtype=cp.float32),
              "lakemask": cp.zeros(shape, dtype=cp.float32),
              "xice": cp.zeros(shape, dtype=cp.float32),
              "smois": cp.full((2, *shape), 0.4, dtype=cp.float32),
              "sh2o": cp.full((2, *shape), 0.2, dtype=cp.float32),
              "smcrel": cp.full((2, *shape), 0.5, dtype=cp.float32),
              "tsk": cp.full(shape, 288, dtype=cp.float32),
              "tsk_save": cp.full(shape, 288, dtype=cp.float32),
              "tsk_sea": cp.full(shape, 288, dtype=cp.float32)}
    fields["landmask"][0] = 0
    fields["lakemask"][0, 1] = 1
    fields["xice"][0, 2] = 1
    if frozen:
        fields["smfr3d"] = cp.full((2, *shape), 0.125, dtype=cp.float32)
    return SimpleNamespace(physics=SimpleNamespace(fields=fields, ruc_params=None), elapsed_seconds=0)


@pytest.mark.parametrize("members", [4, 8])
def test_nested_member_initialization_is_byte_identical_to_same_member_alone(members):
    cp = pytest.importorskip("cupy")
    from woof.ensemble.surface_recipe import surface_initialization_callback
    from woof.ensemble.seeds import member_seed
    options = {"kind": "surface-state", "soil_moisture_scale": [0.8, 1.2], "sst_offset_k": [-1, 1]}
    request = SimpleNamespace(perturbation=options)
    realized = set()
    for member in reversed(range(members)):
        seed = member_seed(713, member)
        nodes = [SimpleNamespace(cfg=SimpleNamespace(grid_id=index), state=_state(cp, shape=shape, frozen=True))
                 for index, shape in ((1, (3, 5)), (2, (6, 7)))]
        model = SimpleNamespace(walk_parent_first=lambda: iter(nodes))
        receipts = []
        callback = surface_initialization_callback(request, member_id=member, seed=seed, record=receipts.append)
        callback(model=model)
        realization = receipts[-1]["domains"][0]["realized_fp32_hex"]
        assert all(row["realized_fp32_hex"] == realization for row in receipts[-1]["domains"])
        realized.add(realization)
        standalone_nodes = [SimpleNamespace(cfg=SimpleNamespace(grid_id=index), state=_state(cp, shape=shape, frozen=True))
                            for index, shape in ((1, (3, 5)), (2, (6, 7)))]
        standalone_model = SimpleNamespace(walk_parent_first=lambda: iter(standalone_nodes))
        alone_receipts = []
        surface_initialization_callback(request, member_id=member, seed=seed,
            record=alone_receipts.append)(model=standalone_model)
        for original, alone in zip(nodes, standalone_nodes):
            for name, values in original.state.physics.fields.items():
                np.testing.assert_array_equal(values.get().view(np.uint32),
                    alone.state.physics.fields[name].get().view(np.uint32))
        assert receipts == alone_receipts
        before_repeat = {name: values.get() for name, values in nodes[0].state.physics.fields.items()}
        callback(model=model)
        for name, before in before_repeat.items():
            np.testing.assert_array_equal(before, nodes[0].state.physics.fields[name].get())
    assert len(realized) == members


def test_fixed_scaling_preserves_frozen_partition_and_open_ocean_masks():
    cp = pytest.importorskip("cupy")
    from woof.ensemble.surface_recipe import apply_surface_recipe
    state = _state(cp, frozen=True)
    receipt = apply_surface_recipe(state, value={"kind": "surface-state", "soil_moisture_scale": 2,
        "sst_offset_k": 1.5}, member_id=0, seed=913, domain_id=1)
    f = state.physics.fields
    assert float(f["smois"][0, 1, 0].get()) == np.float32(0.8)
    assert float(f["sh2o"][0, 1, 0].get()) == np.float32(0.4)
    assert float(f["smfr3d"][0, 1, 0].get()) == np.float32(0.25)
    assert float(f["smcrel"][0, 1, 0].get()) == np.float32(0.5)
    assert "smcrel" not in receipt["attributes"]
    assert float(f["smois"][0, 0, 0].get()) == np.float32(0.4)
    for name in ("tsk", "tsk_save", "tsk_sea"):
        assert float(f[name][0, 0].get()) == 289.5
        assert float(f[name][0, 1].get()) == 288
        assert float(f[name][0, 2].get()) == 288
        assert float(f[name][1, 0].get()) == 288
    assert receipt["realized"] == {"soil_moisture_scale": 2, "sst_offset_k": 1.5}
    assert receipt["changed_surface"]


def test_validation_does_not_half_apply_soil_when_sst_would_be_invalid():
    cp = pytest.importorskip("cupy")
    from woof.ensemble.surface_recipe import apply_surface_recipe
    state = _state(cp, frozen=True)
    before = {name: values.get() for name, values in state.physics.fields.items()}
    with pytest.raises(ValueError, match="before mutation.*ocean temperature"):
        apply_surface_recipe(state, value={"kind": "surface-state", "soil_moisture_scale": 2,
            "sst_offset_k": 300}, member_id=0, seed=913)
    for name, values in before.items():
        np.testing.assert_array_equal(values.view(np.uint32), state.physics.fields[name].get().view(np.uint32))


def test_sst_offset_refusal_names_cells_field_domain_and_zero_k_source():
    # WOOF 2026-11-06 Pickens: 0 K sst on coastal water refused a member with
    # no count, field or cause; the refusal now names them.
    cp = pytest.importorskip("cupy")
    from woof.ensemble.surface_recipe import apply_surface_recipe
    state = _state(cp)
    state.physics.fields["sst"] = cp.full((3, 5), 290, dtype=cp.float32)
    state.physics.fields["sst"][0, 3:] = 0
    with pytest.raises(ValueError, match=r"on domain 1 \(sst: 2 of 3 open-water cells, initialized "
                       r"0\.\.0 K before the \+0\.500 K offset; the initialization left 0 K there"):
        apply_surface_recipe(state, value={"kind": "surface-state", "sst_offset_k": 0.5},
                             member_id=1, seed=913, domain_id=1)


def test_ruc_availability_recomputed_from_perturbed_top_soil_on_gpu():
    cp = pytest.importorskip("cupy")
    from woof.ensemble.surface_recipe import apply_surface_recipe
    state = _state(cp, frozen=True)
    state.physics.ruc_params = SimpleNamespace(bundle=SimpleNamespace(soil=SimpleNamespace(rows=(
        SimpleNamespace(values=(1, 0.1, 0, 1, 0.5)),))))
    state.physics.fields["isltyp"] = cp.ones((3, 5), dtype=cp.int32)
    state.physics.fields["mavail"] = cp.full((3, 5), 0.75, dtype=cp.float32)
    apply_surface_recipe(state, value={"kind": "surface-state", "soil_moisture_scale": 0.5},
                         member_id=0, seed=913)
    assert float(state.physics.fields["mavail"][1, 0].get()) == np.float32(0.25)
    assert float(state.physics.fields["mavail"][0, 0].get()) == np.float32(0.75)


def test_entirely_liquid_alias_and_saturated_profile_scale_once():
    cp = pytest.importorskip("cupy")
    from woof.ensemble.surface_recipe import apply_surface_recipe
    state = _state(cp)
    fields = state.physics.fields
    fields["sh2o"] = fields["smois"]
    apply_surface_recipe(state, value={"kind": "surface-state", "soil_moisture_scale": 2},
                         member_id=0, seed=913)
    assert float(fields["smois"][0, 1, 0].get()) == np.float32(0.8)
    assert float(fields["sh2o"][0, 1, 0].get()) == np.float32(0.8)
    saturated = _state(cp, frozen=True)
    apply_surface_recipe(saturated, value={"kind": "surface-state", "soil_moisture_scale": 4},
                         member_id=0, seed=913)
    assert float(saturated.physics.fields["smois"][0, 1, 0].get()) == 1
    assert float(saturated.physics.fields["sh2o"][0, 1, 0].get()) == 0.5
    assert float(saturated.physics.fields["smfr3d"][0, 1, 0].get()) == 0.3125
    assert float(saturated.physics.fields["smcrel"][0, 1, 0].get()) == 0.5


def test_noah_scaling_preserves_existing_air_dry_floor_and_water_category_fallback():
    cp = pytest.importorskip("cupy")
    from woof.ensemble.surface_recipe import apply_surface_recipe
    from woof.core.noah import SOIL_COLS
    state = _state(cp)
    table = np.zeros((2, len(SOIL_COLS)), dtype=np.float32)
    table[0, SOIL_COLS.index("smcdry")] = np.float32(0.1)
    state.physics.noah_params = SimpleNamespace(soil=table)
    state.physics.fields["isltyp"] = cp.ones((3, 5), dtype=cp.int32)
    state.physics.fields["isltyp"][1, 1] = 2
    receipt = apply_surface_recipe(state, value={"kind": "surface-state", "soil_moisture_scale": 1e-40},
                                   member_id=0, seed=913)
    assert float(state.physics.fields["smois"][0, 1, 0].get()) == np.float32(0.1)
    assert float(state.physics.fields["sh2o"][0, 1, 0].get()) == np.float32(0.05)
    assert float(state.physics.fields["smois"][0, 1, 1].get()) == np.float32(0.005)
    assert float(state.physics.fields["smcrel"][0, 1, 0].get()) == np.float32(0.5)
    assert "SMCDRY" in receipt["soil_moisture_floor"]
