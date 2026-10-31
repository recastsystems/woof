"""The element-parallel launch geometry of the level-independent kernels.

CPU-testable by design.  The bits are proved on a card by the regional
contract deck (``tools/run_cuda_regional_contract.py``) and by the forecast
frames of the point cull; what these tests hold is the shape of the claim:
that every kernel this tree re-mapped walks its work as a grid-stride loop
over a flat element index (so any launch geometry computes every element
and the deck's one-thread-per-owner launch and the forecast's
one-thread-per-element launch cannot disagree), that no re-mapped kernel
still carries the per-owner early-return guard that would make the extra
threads of an element launch skip work, that the loop macros of the eight
translation units are one and the same text, and that the forecast's launch
sites pass element counts rather than owner counts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from woof.hex import (
    cuda_acoustic,
    cuda_driver,
    cuda_dynamics_v841,
    cuda_horizontal,
    cuda_horizontal_v841,
    cuda_regional_v841,
    cuda_transport,
)
from woof.hex.cuda_backend import recovery
from _layout import PACKAGE_DIR

ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_DIR

#: (translation unit source, loop macro, the kernels re-mapped in it)
UNITS = {
    "cuda_regional_v841": (
        cuda_regional_v841.CUDA_REGIONAL_SOURCE,
        "REGIONAL_ELEMENT_LOOP",
        (
            "regional_lbc_rho_edge_v841",
            "regional_speczone_tend_cell_v841",
            "regional_speczone_tend_edge_v841",
            "regional_relaxzone_rayleigh_cell_v841",
            "regional_relaxzone_rayleigh_edge_v841",
            "regional_relaxzone_filter_cell_v841",
            "regional_relaxzone_filter_edge_v841",
            "regional_speczone_u_ru_v841",
            "regional_zero_speczone_w_v841",
            "regional_reset_speczone_values_v841",
            "regional_bdy_adjust_scalars_compute_v841",
            "regional_bdy_adjust_scalars_copyback_v841",
            "regional_bdy_set_scalars_v841",
            "acoustic_ru_regional_v841",
            "acoustic_rs_ts_regional_v841",
            "transport_edge_values_regional_v841",
            "transport_standard_finish_regional_v841",
        ),
    ),
    "cuda_dynamics_v841": (
        cuda_dynamics_v841._CUDA_SOURCE,
        "DYNAMICS_ELEMENT_LOOP",
        (
            "reference_wind_edge_v841_f32",
            "vector_momentum_v841_f32",
            "theta_finish_v841_f32",
            "w_finish_v841_f32",
            "split_flux_first_v841_f32",
            "split_flux_add_v841_f32",
            "split_flux_finish_v841_f32",
        ),
    ),
    "cuda_driver": (
        cuda_driver._CUDA_SOURCE,
        "DRIVER_ELEMENT_LOOP",
        (
            "euler_w_f32",
            "vertical_u_flux_f32",
            "vertical_u_finish_f32",
            "theta_edge_flux_f32",
            "theta_vertical_flux_f32",
            "w_edge_flux_f32",
            "w_vertical_flux_f32",
            "add_inplace_f32",
            "scale_f32",
            "recover_cells_f32",
            "recover_edges_f32",
            "recover_interfaces_f32",
        ),
    ),
    "cuda_acoustic": (
        cuda_acoustic._CUDA_SOURCE,
        "ACOUSTIC_ELEMENT_LOOP",
        ("tendency_w_to_omega",),
    ),
    "cuda_horizontal": (
        cuda_horizontal._CUDA_SOURCE,
        "HORIZONTAL_ELEMENT_LOOP",
        (
            "tangential_velocity_f32",
            "laplacian_divergence_f32",
            "theta_filter_lap2_f32",
            "theta_filter_lap4_f32",
            "w_filter_lap2_f32",
            "w_filter_lap4_f32",
        ),
    ),
    "cuda_horizontal_v841": (
        cuda_horizontal_v841._CUDA_SOURCE,
        "HORIZONTAL_ELEMENT_LOOP",
        (
            "vertex_diagnostics_v841_f32",
            "cell_diagnostics_v841_f32",
            "smagorinsky_v841_f32",
            "pv_cell_v841_f32",
            "pv_apvm_v841_f32",
            "mass_flux_divergence_v841_f32",
        ),
    ),
    "cuda_transport": (
        cuda_transport._CUDA_SOURCE,
        "TRANSPORT_ELEMENT_LOOP",
        ("transport_vertical_flux",),
    ),
    "cuda_backend.recovery": (
        recovery.RECOVERY_CUDA_SOURCE,
        "RECOVERY_ELEMENT_LOOP",
        (
            "recover_pressure_f32",
            "recover_edge_velocity_f32",
            "recover_flat_w_f32",
            "recover_terrain_w_f32",
        ),
    ),
}

#: The one grid-stride form every translation unit's macro must expand to.
LOOP_BODY = (
    "for (int element = blockDim.x * blockIdx.x + threadIdx.x;"
    " element < (total); element += gridDim.x * blockDim.x)"
)

OWNER_GUARD = re.compile(
    r"const int (cell|edge|vertex|owner|slot) = blockDim\.x \* blockIdx\.x"
    r" \+ threadIdx\.x;\s*if \(\1 >= \w+\) return;"
)


def _kernel_text(source: str, name: str) -> str:
    """The text of one ``extern "C" __global__`` definition, macro bodies
    included for the macro-generated recovery kernels."""

    start = source.find(f'extern "C" __global__ void {name}(')
    if start < 0:
        # A macro-declared kernel: DECLARE_*(NAME, T) expands the body.
        macro = re.search(r"DECLARE_(\w+)_KERNEL\(%s, " % re.escape(name), source)
        assert macro is not None, f"{name} is neither defined nor declared"
        start = source.find(f"#define DECLARE_{macro.group(1)}_KERNEL(")
        assert start >= 0
        end = source.find("\n}\n", start)
        return source[start:end]
    end = source.find("\n}\n", start)
    return source[start:end]


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_the_loop_macro_is_the_same_grid_stride_text_in_every_unit(unit):
    source, macro, _ = UNITS[unit]
    definition = re.search(
        r"#define %s\(total\) \\\n(.*?)\n(?=\S)" % macro, source, re.S
    )
    assert definition is not None, f"{unit} lacks #define {macro}(total)"
    collapsed = " ".join(
        line.strip().rstrip("\\").strip() for line in definition.group(1).splitlines()
    )
    assert collapsed == LOOP_BODY, collapsed


@pytest.mark.parametrize(
    "unit,name",
    [(unit, name) for unit, (_, _, names) in UNITS.items() for name in names],
)
def test_every_remapped_kernel_walks_elements_and_keeps_no_owner_guard(unit, name):
    source, macro, _ = UNITS[unit]
    text = _kernel_text(source, name)
    assert f"{macro}(" in text, f"{unit}:{name} does not use {macro}"
    assert OWNER_GUARD.search(text) is None, (
        f"{unit}:{name} still guards on a per-owner thread index; an "
        "element launch would leave that work to threads that exit"
    )


def test_the_divergence_damping_kernel_keeps_the_text_the_local_timestep_unit_derives_from():
    """``cuda_acoustic_lts`` builds its damping kernel by extracting this
    kernel's text and rewriting its thread preamble into an index gather; a
    re-map here would strand that derivation, so the kernel keeps its
    per-owner form until the derivation moves with it."""

    text = _kernel_text(cuda_horizontal._CUDA_SOURCE, "divergence_damping_f32")
    assert "HORIZONTAL_ELEMENT_LOOP(" not in text
    assert "const int edge = blockDim.x * blockIdx.x + threadIdx.x;" in text


def test_the_column_solve_keeps_its_per_column_form():
    """The vertically implicit solve is a recurrence in k: a level-parallel
    map would reorder its arithmetic.  It is deliberately the one regional
    kernel left on one thread per column, and this test refuses the day
    someone folds it in without a new proof."""

    text = _kernel_text(
        cuda_regional_v841.CUDA_REGIONAL_SOURCE,
        "acoustic_column_solve_regional_v841",
    )
    assert "REGIONAL_ELEMENT_LOOP(" not in text
    assert "if (cell >= ncells) return;" in text


def test_the_regional_kernel_set_is_unchanged_by_the_remap():
    """No entrypoint was added or renamed: every deck still names a kernel
    and every kernel still has a deck."""

    defined = set(re.findall(r"__global__ void (\w+)", cuda_regional_v841.CUDA_REGIONAL_SOURCE))
    assert defined == set(cuda_regional_v841.REGIONAL_KERNELS)
    assert len(cuda_regional_v841.REGIONAL_KERNELS) == 22


def test_the_momentum_kernel_gathers_the_reference_wind_it_used_to_compute():
    """``u_init*cos(angleEdge) - v_init*sin(angleEdge)`` is evaluated by a
    kernel of the same translation unit, once per run, and the momentum
    kernel no longer calls cosf/sinf itself."""

    source = cuda_dynamics_v841._CUDA_SOURCE
    momentum = _kernel_text(source, "vector_momentum_v841_f32")
    assert "cosf(" not in momentum and "sinf(" not in momentum
    assert "reference_u[E2(k, neighbor, nedges)]" in momentum
    reference = _kernel_text(source, "reference_wind_edge_v841_f32")
    assert (
        "mpas_sub(\n            mpas_mul(u_init[k], cosf(angle_edge[edge])),\n"
        "            mpas_mul(v_init[k], sinf(angle_edge[edge])))"
    ) in reference
    assert callable(cuda_dynamics_v841.reference_wind_edge_cuda_v841)


FORECAST = (SRC / "cuda_regional_forecast_v841.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name,count",
    [
        ("acoustic_ru_regional_v841", "nlev * nedges"),
        ("acoustic_rs_ts_regional_v841", "nlev * ncells"),
        ("acoustic_column_solve_regional_v841", "ncells"),
        ("transport_edge_values_regional_v841", "group * nlev * nedges"),
        ("transport_standard_finish_regional_v841", "group * nlev * ncells"),
        ("regional_relaxzone_filter_cell_v841", "max(nlev * masks.n_relax_cells, 1)"),
        ("regional_relaxzone_filter_edge_v841", "max(nlev * masks.n_relax_edges, 1)"),
        ("regional_relaxzone_rayleigh_cell_v841", "max(nlev * masks.n_relax_cells, 1)"),
        ("regional_relaxzone_rayleigh_edge_v841", "max(nlev * masks.n_relax_edges, 1)"),
        ("regional_speczone_tend_cell_v841", "max(nlev * masks.n_spec_cells, 1)"),
        ("regional_speczone_tend_edge_v841", "max(nlev * masks.n_spec_edges, 1)"),
        ("regional_speczone_u_ru_v841", "max(self.nlev * masks.n_spec_edges, 1)"),
        ("regional_zero_speczone_w_v841", "max((self.nlev + 1) * masks.n_spec_cells, 1)"),
        ("regional_bdy_adjust_scalars_copyback_v841", "max(ntracers * self.nlev * masks.n_nudged_cells, 1)"),
        ("regional_reset_speczone_values_v841", "max(self.nlev * masks.n_spec_cells, 1)"),
    ],
)
def test_the_forecast_launches_the_regional_kernels_per_element(name, count):
    pattern = re.compile(r'"%s",\n\s+%s,' % (re.escape(name), re.escape(count)))
    assert pattern.search(FORECAST), f"{name} is not launched with {count}"


def test_the_forecast_never_launches_a_remapped_regional_kernel_per_owner():
    """A per-owner count on a re-mapped kernel is not wrong -- the
    grid-stride loop computes everything -- but it is the pre-remap
    geometry, one thread per column, and the point of the remap is gone."""

    _, _, remapped = UNITS["cuda_regional_v841"]
    for name in remapped:
        if name in ("regional_lbc_rho_edge_v841",):
            continue  # launched from cuda_regional_v841 itself
        for match in re.finditer(r'"%s",\n\s+([^\n]+),' % re.escape(name), FORECAST):
            count = match.group(1)
            assert "nlev" in count or "ntracers" in count, (name, count)
