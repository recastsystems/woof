"""Local time stepping: classing, derivation fidelity, and the default-off shape.

These are CPU-only checks.  The bit-identity and conservation gates are run
against the exe on a real mesh; nothing here substitutes for those.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex.config_lts import (
    DEFAULT_LOCAL_TIMESTEP_RATES,
    V841LocalTimestepDryConfig,
    V841LocalTimestepGwdoConfig,
    V841LocalTimestepSmagorinskyGwdoConfig,
    local_timestep_enabled,
)
from woof.hex.config_v841 import (
    V841DryDycoreConfig,
    V841MpasColumnPhysicsGwdoConfig,
    V841MpasColumnPhysicsSmagorinskyGwdoConfig,
)
from woof.hex.errors import ConfigurationRefusal
from woof.hex.lts_v841 import (
    admissible_rates,
    cell_min_spacing,
    classify_from_cell_rates,
    classify_local_timestep,
)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------
def test_released_schedule_admits_one_and_three_only():
    # (1, 3, 6) is the released schedule from integration.RKSchedule.from_mpas.
    # 2 and 4 do not divide the RK2 stage's three sub-steps, so a rate-2 or
    # rate-4 class would have to change that stage's sub-step count.
    assert admissible_rates((1, 3, 6)) == (1, 3)


def test_single_substep_stage_does_not_collapse_the_ladder():
    # RK1 runs one sub-step; requiring r | 1 would leave only rate 1 and turn
    # the whole feature into a silent no-op.
    assert admissible_rates((1, 1, 4)) == (1, 2, 4)


def test_ladder_refuses_a_zero_substep_stage():
    with pytest.raises(ConfigurationRefusal):
        admissible_rates((1, 0, 6))


# --------------------------------------------------------------------------
# h_c
# --------------------------------------------------------------------------
def test_cell_min_spacing_hand_case():
    dc = np.array([3.0, 7.0, 11.0])
    edges_on_cell = np.array([[1, 2, 0], [2, 3, 0]])  # 1-based, 0 padded
    counts = np.array([2, 2])
    h = cell_min_spacing(dc, edges_on_cell, counts)
    assert h.tolist() == [3.0, 7.0]


def test_cell_min_spacing_ignores_padding_beyond_the_count():
    dc = np.array([3.0, 7.0, 1.0])
    edges_on_cell = np.array([[1, 2, 3]])
    counts = np.array([2])  # slot 3 (dcEdge 1.0) is past the count
    assert cell_min_spacing(dc, edges_on_cell, counts).tolist() == [3.0]


# --------------------------------------------------------------------------
# classing
# --------------------------------------------------------------------------
def _chain(spacings):
    """A 1-D chain of cells: cell i shares edge i with cell i+1."""

    n_cells = len(spacings)
    dc = np.asarray(spacings, dtype=float)
    n_edges = n_cells - 1
    cells_on_edge = np.array(
        [[i + 1, i + 2] for i in range(n_edges)], dtype=np.int64
    )
    edges_on_cell = np.zeros((n_cells, 2), dtype=np.int64)
    counts = np.zeros(n_cells, dtype=np.int64)
    for edge in range(n_edges):
        for cell in (edge, edge + 1):
            edges_on_cell[cell, counts[cell]] = edge + 1
            counts[cell] += 1
    return dc[:n_edges], edges_on_cell, counts, cells_on_edge


def test_uniform_mesh_is_a_single_class_with_the_identity_permutation():
    dc, eoc, counts, coe = _chain([100.0] * 9)
    classing = classify_local_timestep(
        dc_edge=dc,
        edges_on_cell=eoc,
        n_edges_on_cell=counts,
        cells_on_edge=coe,
        rates=(1, 3),
    )
    assert classing.is_single_class
    assert classing.identity_permutation
    assert classing.interface_edges.size == 0
    assert classing.arithmetic_acoustic_saving((1, 3, 6)) == 0.0


def test_a_ratio_below_the_next_rung_stays_in_the_fine_class():
    # 2.9x is not admissible for a rate-3 sub-step.  Rounding it up would hand
    # a cell a step its own spacing does not permit.
    dc, eoc, counts, coe = _chain([10.0, 29.0, 29.0, 29.0])
    classing = classify_local_timestep(
        dc_edge=dc,
        edges_on_cell=eoc,
        n_edges_on_cell=counts,
        cells_on_edge=coe,
        rates=(1, 3),
    )
    assert classing.is_single_class


def test_a_coarse_run_earns_a_coarse_class_and_an_interface():
    dc, eoc, counts, coe = _chain([10.0] + [50.0] * 8)
    classing = classify_local_timestep(
        dc_edge=dc,
        edges_on_cell=eoc,
        n_edges_on_cell=counts,
        cells_on_edge=coe,
        rates=(1, 3),
        buffer_rings=1,
    )
    assert classing.rates == (1, 3)
    assert classing.interface_edges.size >= 1
    # An edge always runs at the finer of its two cells' rates, so the fine
    # side's flux is sampled at the fine rate and the reflux has something to
    # accumulate.
    c0 = coe[:, 0] - 1
    c1 = coe[:, 1] - 1
    expected = np.minimum(classing.cell_rate[c0], classing.cell_rate[c1])
    assert np.array_equal(classing.edge_rate, expected)
    assert classing.arithmetic_acoustic_saving((1, 3, 6)) > 0.0


def test_the_buffer_ring_widens_the_fine_class():
    dc, eoc, counts, coe = _chain([10.0] + [50.0] * 8)
    narrow = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    wide = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=3,
    )
    assert int((wide.cell_rate == 1).sum()) >= int((narrow.cell_rate == 1).sum())


def test_no_cell_is_handed_a_step_its_own_spacing_forbids():
    rng = np.random.default_rng(20260821)
    spacings = rng.uniform(10.0, 300.0, size=200)
    dc, eoc, counts, coe = _chain(spacings.tolist())
    classing = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    ratio = classing.h_cell / classing.h_min
    assert np.all(classing.cell_rate <= ratio + 1e-9)


def test_classing_refuses_a_ladder_that_does_not_start_at_one():
    dc, eoc, counts, coe = _chain([10.0] * 5)
    with pytest.raises(ConfigurationRefusal):
        classify_local_timestep(
            dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
            cells_on_edge=coe, rates=(2, 4),
        )


def test_classing_refuses_a_zero_buffer():
    dc, eoc, counts, coe = _chain([10.0] * 5)
    with pytest.raises(ConfigurationRefusal):
        classify_local_timestep(
            dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
            cells_on_edge=coe, rates=(1, 3), buffer_rings=0,
        )


# --------------------------------------------------------------------------
# the option shape
# --------------------------------------------------------------------------
def test_the_switch_is_off_by_default():
    for config in (
        V841LocalTimestepDryConfig(),
        V841LocalTimestepGwdoConfig(),
        V841LocalTimestepSmagorinskyGwdoConfig(),
    ):
        assert config.config_local_timestep is False
        assert local_timestep_enabled(config) is False
        config.validate()


def test_the_switch_off_leaves_every_pinned_knob_where_it_was():
    for lts_type, parent_type in (
        (V841LocalTimestepDryConfig, V841DryDycoreConfig),
        (V841LocalTimestepGwdoConfig, V841MpasColumnPhysicsGwdoConfig),
        (
            V841LocalTimestepSmagorinskyGwdoConfig,
            V841MpasColumnPhysicsSmagorinskyGwdoConfig,
        ),
    ):
        lts = lts_type()
        parent = parent_type()
        for field in parent.__dataclass_fields__:
            assert getattr(lts, field) == getattr(parent, field), field
        # config_dt in particular: local time stepping does not move the model
        # timestep, it changes how many acoustic sub-steps a column runs.
        assert lts.config_dt == parent.config_dt

    # The column-physics lane's dt is pinned at 120 s and the subtype inherits
    # it unchanged; the dry lane leaves dt free, which is what lets the referee
    # arm run the same mesh at a globally smaller acoustic step.
    assert V841LocalTimestepSmagorinskyGwdoConfig().config_dt == 120.0
    assert V841LocalTimestepDryConfig(config_dt=7.5).config_dt == 7.5


def test_the_default_ladder_is_two_classes():
    assert DEFAULT_LOCAL_TIMESTEP_RATES == (1, 3)
    assert len(DEFAULT_LOCAL_TIMESTEP_RATES) == 2


def test_the_dry_lane_carries_the_block_and_stays_dry():
    # The conservation gate is defined for a no-physics run with qv as a
    # passive scalar; a subtype that quietly turned physics back on would be
    # measuring the microphysics budget instead of the transport.
    config = V841LocalTimestepDryConfig(config_local_timestep=True)
    config.validate()
    assert config.config_physics_suite == "none"
    assert config.config_moist_physics is False
    assert local_timestep_enabled(config) is True


def test_the_declared_off_arm_is_the_subtype_with_the_switch_off():
    # GATE 2's arm: the local-timestep configuration TYPE with the switch off.
    # Selecting the parent type instead would prove nothing, because then the
    # option's own code never entered the run at all.
    arm = V841LocalTimestepDryConfig(config_local_timestep=False)
    arm.validate()
    assert type(arm) is not V841DryDycoreConfig
    assert isinstance(arm, V841DryDycoreConfig)
    assert local_timestep_enabled(arm) is False


def test_a_ladder_that_skips_rate_one_is_refused():
    with pytest.raises(ConfigurationRefusal):
        V841LocalTimestepGwdoConfig(
            config_local_timestep=True, config_local_timestep_rates=(3, 6)
        ).validate()


def test_a_safety_factor_above_one_is_refused():
    with pytest.raises(ConfigurationRefusal):
        V841LocalTimestepGwdoConfig(
            config_local_timestep=True,
            config_local_timestep_safety_factor=1.5,
        ).validate()


# --------------------------------------------------------------------------
# derivation fidelity: the LTS kernel is its ancestor plus the gather, nothing else
# --------------------------------------------------------------------------
_GATHERED = (
    ("acoustic_coefficients_v841", "cell", "ncells"),
    ("acoustic_ru_v841", "edge", "nedges"),
    ("acoustic_prepare_v841", "cell", "ncells"),
    ("acoustic_rs_ts_v841", "cell", "ncells"),
    ("acoustic_column_solve_v841", "cell", "ncells"),
)


def test_every_derived_acoustic_kernel_is_its_ancestor_plus_the_gather():
    from woof.hex import cuda_acoustic_lts as lts
    from woof.hex import cuda_acoustic_v841

    source = cuda_acoustic_v841._CUDA_SOURCE
    for name, index, bound in _GATHERED:
        original = lts._extract_kernel(source, name)
        derived = lts._gather(
            original,
            name=name,
            index=index,
            bound=bound,
            new_name=name.replace("_v841", "_lts"),
        )
        # Undo exactly the three declared edits and require the ancestor back.
        restored = derived.replace(
            "const int lts_slot = blockDim.x * blockIdx.x + threadIdx.x;\n"
            "    if (lts_slot >= lts_n_active) return;\n"
            f"    const int {index} = lts_active[lts_slot];",
            f"const int {index} = blockDim.x * blockIdx.x + threadIdx.x;\n"
            f"    if ({index} >= {bound}) return;",
        )
        restored = restored.replace(
            ",\n    const int *lts_active, const int lts_n_active", ""
        )
        restored = restored.replace(name.replace("_v841", "_lts"), name)
        assert restored == original, name


def test_the_damping_kernel_is_its_ancestor_plus_the_gather():
    from woof.hex import cuda_acoustic_lts as lts
    from woof.hex import cuda_horizontal

    original = lts._extract_kernel(cuda_horizontal._CUDA_SOURCE, "divergence_damping_f32")
    derived = lts._gather(
        original,
        name="divergence_damping_f32",
        index="edge",
        bound="nedges",
        new_name="divergence_damping_lts",
    )
    restored = derived.replace(
        "const int lts_slot = blockDim.x * blockIdx.x + threadIdx.x;\n"
        "    if (lts_slot >= lts_n_active) return;\n"
        "    const int edge = lts_active[lts_slot];",
        "const int edge = blockDim.x * blockIdx.x + threadIdx.x;\n"
        "    if (edge >= nedges) return;",
    )
    restored = restored.replace(
        ",\n    const int *lts_active, const int lts_n_active", ""
    )
    restored = restored.replace("divergence_damping_lts", "divergence_damping_f32")
    assert restored == original


def test_a_moved_anchor_refuses_instead_of_silently_deriving_a_whole_domain_launch():
    from woof.hex import cuda_acoustic_lts as lts

    drifted = (
        'extern "C" __global__ void made_up(const int ncells)\n'
        "{\n"
        "    const int cell = threadIdx.x;\n"
        "    if (cell >= ncells) return;\n"
        "}\n"
    )
    with pytest.raises(lts.LocalTimestepSourceDrift):
        lts._gather(
            lts._extract_kernel(drifted, "made_up"),
            name="made_up",
            index="cell",
            bound="ncells",
            new_name="made_up_lts",
        )


def test_a_flux_corrected_limiter_is_refused_alongside_the_option():
    # Not reachable today -- the v8.4.1 CUDA lane refuses both knobs outright --
    # but admitting the FCT branch later must not let the option ride along on
    # an interface that was never proved with a limiter crossing it.
    import types

    from woof.hex import cuda_driver_lts

    for knob in ("config_monotonic", "config_positive_definite"):
        config = types.SimpleNamespace(
            config_local_timestep=True,
            config_monotonic=False,
            config_positive_definite=False,
            config_time_integration_order=3,
            config_number_of_sub_steps=6,
            config_local_timestep_rates=(1, 3),
            config_local_timestep_buffer_rings=1,
            config_local_timestep_safety_factor=1.0,
        )
        setattr(config, knob, True)
        driver = types.SimpleNamespace(
            config=config, v841_context=object(), halo_exchanger_v841=None
        )
        with pytest.raises(ConfigurationRefusal) as caught:
            cuda_driver_lts.attach_local_timestep(driver)
        assert knob in str(caught.value)


def test_the_reflux_settle_carries_no_atomics():
    # A coarse cell can own more than one interface edge -- 404 of 529 do on
    # the published x4.163842 mesh -- so a per-edge settle has to atomicAdd
    # into the shared cell.  Binary32 addition is not associative, and two
    # identical runs then differ: measured over 120 steps that seed reached
    # 59-99% of the domain.  The port's corruption screen on cards with no ECC
    # is the dual-run byte comparison, which a nondeterministic path defeats.
    from woof.hex import cuda_acoustic_lts as lts

    import re

    source = lts.local_timestep_cuda_source()
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    # The stripper has to work, or the test passes for the wrong reason.
    assert "atomicAdd" in source, "the explanatory comment names the hazard"
    assert "ONE THREAD PER COARSE CELL" not in code, "comments were not stripped"
    # No accumulating atomic anywhere in the translation unit.  The atomics
    # that remain are the ancestors' ``atomicExch(singular, 1)`` (the global
    # coefficient kernel) and ``atomicExch(invalid, 1)`` (the regional
    # momentum kernel's validity flag), each of which writes the same constant
    # from every racing thread and so has one outcome whatever the order; both
    # are pinned text and are not touched here.
    assert "atomicAdd" not in code
    assert "atomicSub" not in code
    exchanges = re.findall(r"atomicExch\([^)]*\)", code)
    assert sorted(exchanges) == ["atomicExch(invalid, 1)", "atomicExch(singular, 1)"]

    settle = lts._extract_kernel(source, "lts_reflux_settle")
    accumulate = lts._extract_kernel(source, "lts_reflux_accumulate")
    for name, body in (("settle", settle), ("accumulate", accumulate)):
        assert "atomic" not in body, name


def test_the_settle_grouping_keeps_every_interface_edge_exactly_once():
    import numpy as _np

    from woof.hex.lts_v841 import classify_local_timestep

    dc, eoc, counts, coe = _chain([10.0] * 3 + [50.0] * 6)
    classing = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    iface = _np.asarray(classing.interface_edges)
    assert iface.size >= 1
    # Every interface edge has exactly one coarse side, so a grouping by coarse
    # cell partitions them.  Losing or duplicating one would silently drop or
    # double count mass at the boundary.
    c0 = coe[iface, 0] - 1
    c1 = coe[iface, 1] - 1
    coarse = _np.where(
        classing.cell_rate[c1] > classing.cell_rate[c0], c1, c0
    )
    order = _np.argsort(coarse, kind="stable")
    cells, starts = _np.unique(coarse[order], return_index=True)
    offsets = _np.append(starts, coarse.size)
    assert int(offsets[-1]) == int(iface.size)
    assert int(_np.diff(offsets).sum()) == int(iface.size)
    assert cells.size == _np.unique(coarse).size


def test_the_derived_translation_unit_declares_every_kernel_once():
    from woof.hex import cuda_acoustic_lts as lts

    source = lts.local_timestep_cuda_source()
    for name in (
        "acoustic_cofrz_v841",
        "acoustic_coefficients_lts",
        "acoustic_ru_lts",
        "acoustic_prepare_lts",
        "acoustic_rs_ts_lts",
        "acoustic_column_solve_lts",
        "divergence_damping_lts",
        "lts_reflux_accumulate",
        "lts_reflux_settle",
        "lts_recover_edges",
        "lts_recover_interfaces",
    ):
        assert source.count(f"void {name}(") == 1, name


# --------------------------------------------------------------------------
# limited-area culls: the driven-zone hold, one-cell edges, explicit rates
# --------------------------------------------------------------------------
def _chain_with_open_end(spacings):
    """The chain, plus one ring-7 style edge on the last cell with no second cell."""

    dc, eoc, counts, coe = _chain(spacings)
    n_cells = len(spacings)
    open_edge = len(dc)  # 0-based index of the new edge
    dc = np.append(dc, dc[-1])
    coe = np.vstack([coe, [[n_cells, 0]]])  # 1-based: last cell, then 0 = no cell
    eoc = np.hstack([eoc, np.zeros((n_cells, 1), dtype=np.int64)])
    eoc[n_cells - 1, counts[n_cells - 1]] = open_edge + 1
    counts = counts.copy()
    counts[n_cells - 1] += 1
    return dc, eoc, counts, coe


def test_a_one_cell_edge_takes_its_present_cells_rate_and_is_never_an_interface():
    # Before 2026-09-14 the classing raised on the 0 a limited-area grid file
    # writes for the missing cell of a ring-7 edge, so --local-timestep could
    # not run on any cull at all.
    dc, eoc, counts, coe = _chain_with_open_end([10.0] + [50.0] * 8)
    classing = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    open_edge = len(dc) - 1
    assert open_edge not in set(classing.interface_edges.tolist())
    assert classing.edge_rate[open_edge] == classing.cell_rate[-1]
    assert classing.cell_rate[-1] == 3


def test_an_edge_with_no_cell_at_all_is_refused():
    dc, eoc, counts, coe = _chain([10.0] * 4)
    coe = np.vstack([coe, [[0, 0]]])
    dc = np.append(dc, 10.0)
    with pytest.raises(ValueError):
        classify_local_timestep(
            dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
            cells_on_edge=coe, rates=(1, 3),
        )


def test_every_driven_zone_cell_is_held_at_rate_one_and_the_hold_is_recorded():
    # Nine cells, the last four in driven rings 1..4 (a cull's boundary).  All
    # nine but the first would earn rate 3 from spacing; the driven four are
    # held, the buffer then demotes the interior cell next to ring 1.
    dc, eoc, counts, coe = _chain([10.0] + [50.0] * 8)
    ring = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4])
    classing = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
        driven_ring=ring, driven_zone_source="bdyMaskCell",
    )
    assert np.all(classing.cell_rate[ring > 0] == 1)
    summary = classing.summary()
    assert summary["driven_zone_source"] == "bdyMaskCell"
    # Cells 0 and 1 share the 10 m edge, so both earn rate 1 from spacing.
    assert summary["cells_qualified_by_spacing"] == {1: 2, 3: 7}
    assert summary["cells_requested"] is None
    assert summary["cells_held_in_driven_zone"] == 4
    assert summary["cells_held_in_driven_zone_by_ring"] == {1: 1, 2: 1, 3: 1, 4: 1}
    # The interior cell adjacent to ring 1 is demoted by the buffer, and the
    # cell adjacent to the fine cell 0 likewise; both are charged to the
    # buffer, not to the driven zone.
    assert summary["cells_demoted_by_buffer"] == 2
    # No interface may touch a driven cell.
    cells = np.asarray(classing.interface_cells)
    assert cells.size and np.all(ring[cells] == 0)


def test_the_driven_hold_leaves_a_global_mesh_alone():
    dc, eoc, counts, coe = _chain([10.0] + [50.0] * 8)
    without = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    with_zero_ring = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
        driven_ring=np.zeros(9, dtype=np.int32),
    )
    assert np.array_equal(without.cell_rate, with_zero_ring.cell_rate)
    assert without.summary()["driven_zone_source"] is None
    assert without.summary()["cells_held_in_driven_zone"] is None
    assert with_zero_ring.summary()["cells_held_in_driven_zone"] == 0


def test_explicit_rates_class_the_interior_where_spacing_would_not():
    # The instrument: a uniform chain, which spacing classes into one class,
    # handed rate 3 on its right half.  The interface exists, the driven hold
    # and the buffer still apply, and the result says it is explicit.
    dc, eoc, counts, coe = _chain([10.0] * 9)
    requested = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3], dtype=np.int32)
    ring = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    classing = classify_from_cell_rates(
        cell_rate=requested, dc_edge=dc, edges_on_cell=eoc,
        n_edges_on_cell=counts, cells_on_edge=coe, rates=(1, 3),
        buffer_rings=1, driven_ring=ring, driven_zone_source="bdyMaskCell",
    )
    assert classing.rate_source == "explicit"
    assert classing.rates == (1, 3)
    assert classing.cell_rate[-1] == 1  # held: driven
    assert classing.cell_rate[-2] == 1  # buffer next to the held cell
    assert classing.cell_rate[4] == 1  # buffer next to the fine half
    assert classing.cell_rate.tolist() == [1, 1, 1, 1, 1, 3, 3, 1, 1]
    assert classing.interface_edges.size == 2
    summary = classing.summary()
    assert summary["rate_source"] == "explicit"
    # The receipt must not present the instrument's request as something the
    # spacing said: the requested rates go out under their own key and the
    # spacing key is empty for an explicit classing.
    assert summary["cells_qualified_by_spacing"] is None
    assert summary["cells_requested"] == {1: 4, 3: 5}
    # Two cells were demoted from the request by the buffer (cell 4 beside the
    # fine half, cell 7 beside the held driven cell); the driven cell itself is
    # charged to the hold, not to the buffer.
    assert summary["cells_demoted_by_buffer"] == 2
    assert summary["cells_held_in_driven_zone"] == 1


def test_explicit_rates_off_the_ladder_are_refused():
    dc, eoc, counts, coe = _chain([10.0] * 5)
    with pytest.raises(ConfigurationRefusal):
        classify_from_cell_rates(
            cell_rate=np.array([1, 2, 1, 1, 1]), dc_edge=dc, edges_on_cell=eoc,
            n_edges_on_cell=counts, cells_on_edge=coe, rates=(1, 3),
        )


def test_an_interface_touching_a_driven_cell_is_refused_on_the_driver():
    from woof.hex.cuda_driver_lts import refuse_interface_in_driven_zone

    dc, eoc, counts, coe = _chain([10.0] * 9)
    requested = np.array([1, 1, 1, 1, 3, 3, 3, 3, 3], dtype=np.int32)
    # Built with no driven mask, so the coarse class reaches the last cell.
    classing = classify_from_cell_rates(
        cell_rate=requested, dc_edge=dc, edges_on_cell=eoc,
        n_edges_on_cell=counts, cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    assert classing.cell_rate[-1] == 3
    # The driver's own mask says the last two cells are driven: refused.
    with pytest.raises(ConfigurationRefusal) as caught:
        refuse_interface_in_driven_zone(classing, np.array([0, 0, 0, 0, 0, 0, 0, 1, 2]))
    assert "driven" in str(caught.value)
    # An interface next to a driven cell with every driven cell at rate 1
    # cannot come out of either classing route (the buffer ring moves the
    # jump one cell inward), so the second condition is exercised on a
    # classing whose interface is set by hand.
    import dataclasses

    held = classify_from_cell_rates(
        cell_rate=np.array([1, 1, 1, 1, 3, 3, 3, 3, 1], dtype=np.int32),
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
        driven_ring=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
    )
    assert held.cell_rate.tolist() == [1, 1, 1, 1, 1, 3, 3, 1, 1]
    # Interior interfaces, driven cell at rate 1: admitted.
    refuse_interface_in_driven_zone(held, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]))
    forged = dataclasses.replace(
        held,
        interface_edges=np.array([7], dtype=np.int32),
        interface_cells=np.array([[7, 8]], dtype=np.int32),
    )
    with pytest.raises(ConfigurationRefusal) as caught:
        refuse_interface_in_driven_zone(forged, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert "interface" in str(caught.value)


def test_the_derived_regional_kernels_are_their_ancestors_plus_the_gather():
    from woof.hex import cuda_acoustic_lts as lts
    from woof.hex import cuda_regional_v841

    source = cuda_regional_v841._CUDA_SOURCE
    assert len(lts.REGIONAL_GATHERED) == 3
    forms = {}
    for name, index, bound in lts.REGIONAL_GATHERED:
        original = lts._extract_kernel(source, name)
        derived = lts._gather(
            original, name=name, index=index, bound=bound,
            new_name=name.replace("_v841", "_lts"),
        )
        # Undo exactly the declared edits, in whichever preamble form the
        # ancestor has, and require the ancestor back.
        if lts._gathered_element_preamble(index) in derived:
            forms[name] = "element"
            restored = derived.replace(
                lts._gathered_element_preamble(index), lts._element_preamble(index, bound)
            )
        else:
            forms[name] = "owner"
            restored = derived.replace(
                lts._gathered_owner_preamble(index), lts._owner_preamble(index, bound)
            )
        restored = restored.replace(
            ",\n    const int *lts_active, const int lts_n_active", ""
        )
        restored = restored.replace(name.replace("_v841", "_lts"), name)
        assert restored == original, name
        # The specified-zone masking survives the gather byte for byte.
        assert "spec_zone_mask" in derived
    # The two level-independent entrypoints walk elements; the column solve
    # keeps its per-column form.
    assert forms == {
        "acoustic_ru_regional_v841": "element",
        "acoustic_rs_ts_regional_v841": "element",
        "acoustic_column_solve_regional_v841": "owner",
    }


def test_the_derived_unit_carries_the_regional_element_loop_verbatim():
    from woof.hex import cuda_acoustic_lts as lts
    from woof.hex import cuda_regional_v841

    macro = lts._extract_macro(cuda_regional_v841._CUDA_SOURCE, "REGIONAL_ELEMENT_LOOP")
    assert macro.startswith("#define REGIONAL_ELEMENT_LOOP(total)")
    assert "element += gridDim.x * blockDim.x" in macro
    source = lts.local_timestep_cuda_source()
    assert source.count(macro) == 1
    assert source.count("REGIONAL_ELEMENT_LOOP(nlev * lts_n_active)") == 2


def test_a_kernel_with_both_preamble_forms_is_refused():
    from woof.hex import cuda_acoustic_lts as lts

    both = (
        'extern "C" __global__ void twice(const int nlev, const int ncells)\n{\n    '
        + lts._owner_preamble("cell", "ncells")
        + "\n    "
        + lts._element_preamble("cell", "ncells")
        + "\n    }\n}"
    )
    with pytest.raises(lts.LocalTimestepSourceDrift):
        lts._gather(both, name="twice", index="cell", bound="ncells", new_name="twice_lts")


def test_the_derived_translation_unit_declares_the_regional_kernels_once():
    from woof.hex import cuda_acoustic_lts as lts

    source = lts.local_timestep_cuda_source()
    for name in (
        "acoustic_ru_regional_lts",
        "acoustic_rs_ts_regional_lts",
        "acoustic_column_solve_regional_lts",
    ):
        assert source.count(f"void {name}(") == 1, name


def test_the_regional_sub_step_is_rebound_and_restored():
    # The pinned driver imports the regional sub-step from its module at every
    # sub-step, so the module attribute is where the attachment must bind;
    # rebinding only the global name leaves a regional run on the global rate
    # under class-rate coefficients.  Checked without a card: the attach and
    # detach methods are exercised on a stand-in with the device work stubbed.
    import types

    from woof.hex import cuda_driver as driver_module
    from woof.hex import cuda_driver_lts
    from woof.hex import cuda_regional_forecast_v841 as regional_module

    before_global = driver_module.advance_acoustic_step_cuda_v841
    before_regional = regional_module.advance_acoustic_step_regional_cuda_v841
    attachment = cuda_driver_lts.LocalTimestepAttachment.__new__(
        cuda_driver_lts.LocalTimestepAttachment
    )
    horizontal = types.SimpleNamespace(divergence_damping_3d=lambda *a, **k: None)
    driver = types.SimpleNamespace(
        horizontal=horizontal, _recover_candidate=lambda *a, **k: None
    )
    attachment.driver = driver
    attachment._saved = {}
    attachment.attach()
    assert driver_module.advance_acoustic_step_cuda_v841 == attachment.advance_acoustic_step
    assert (
        regional_module.advance_acoustic_step_regional_cuda_v841
        == attachment.advance_acoustic_step_regional
    )
    attachment.detach()
    assert driver_module.advance_acoustic_step_cuda_v841 is before_global
    assert regional_module.advance_acoustic_step_regional_cuda_v841 is before_regional


def test_a_regional_driver_counts_its_padding_in_the_mesh_check():
    # A regional driver's ncells/nedges carry one garbage element each.  The
    # attachment is exercised up to the mesh check with a classing one short
    # on edges, so the refusal names the padded expectation.
    import types

    from woof.hex import cuda_driver_lts

    dc, eoc, counts, coe = _chain([10.0] * 5)
    classing = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3),
    )
    config = types.SimpleNamespace(
        config_local_timestep=True, config_monotonic=False,
        config_positive_definite=False, config_time_integration_order=3,
        config_number_of_sub_steps=6, config_local_timestep_rates=(1, 3),
        config_local_timestep_buffer_rings=1,
        config_local_timestep_safety_factor=1.0,
    )
    regional = types.SimpleNamespace(
        masks_host=types.SimpleNamespace(bdy_mask_cell=np.zeros(5, dtype=np.int32))
    )
    # Global shape: 5 cells, 4 edges, no padding -> passes the check.
    driver = types.SimpleNamespace(
        config=config, v841_context=object(), halo_exchanger_v841=None,
        regional_v841=regional, ncells=5, nedges=4,
    )
    with pytest.raises(ConfigurationRefusal) as caught:
        cuda_driver_lts.attach_local_timestep(driver, classing=classing)
    assert "padded" in str(caught.value)
    assert "4 cells and 3 edges" in str(caught.value)



def test_the_damping_override_takes_every_keyword_the_acoustic_loop_passes():
    """THE BREAKAGE (hex 0.6.1; fork jacobbieker/woof PR 447): the v8.4.1 acoustic loop calls
    ``horizontal.divergence_damping_3d(..., native_stored_inverse=True)``, and under --local-timestep that
    method is the LocalTimestepAttachment's override, which lacked the keyword: every --local-timestep run
    stopped at step 0 with a TypeError."""
    import inspect

    from woof.hex.cuda_driver_lts import LocalTimestepAttachment
    from woof.hex.cuda_horizontal import CudaHorizontal

    def keywords(function):
        return {name for name, parameter in inspect.signature(function).parameters.items()
                if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not name.startswith("_")}

    assert keywords(CudaHorizontal.divergence_damping_3d) <= keywords(LocalTimestepAttachment.divergence_damping_3d)
