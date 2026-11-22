"""The limited-area PHYSICS seam: one route, and the four things it had to fix.

the regional physics work, 2026-08-26.  Before this lane the limited-area
runtime was a dry dycore with a single passive moisture variable: it produced
wind, temperature, density and pressure and nothing renderable.  Joining it to
the full WOOF-ArWen stack was not wiring, but it was also not a second path -- the
physics geometry constructors turned out to be cell-count agnostic, so the
SAME constructors the global stack calls are called on a mesh view with one
more cell in it.

These tests hold the four host-side conventions that made that possible, and
the three defects the join uncovered.  None of them needs a card; the run they
describe is `evidence/regional-physics-20260826/`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from woof.hex import regional_v841
from woof.hex.cuda_regional_forecast_v841 import (
    PaddedRegionalHostMesh,
    SELF_MANAGED_GARBAGE_MODULES,
    pad_regional_physics_host,
)
from _layout import PACKAGE_DIR

ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_DIR


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def _mesh_binding_module():
    spec = importlib.util.spec_from_file_location(
        "_regional_physics_mesh_binding", PACKAGE_DIR / "drivers" / "mpas_mesh_binding.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


class _SyntheticCull:
    """Three cells, four edges, two vertices, with ring-7 sentinels."""

    def __init__(self) -> None:
        n_cells, n_edges, n_vertices = 3, 4, 2
        self.arrays: dict[str, object] = {
            "cellsOnEdge": np.array(
                [[0, 1], [1, 2], [2, -1], [0, 2]], dtype=np.int64
            ),
            "cellsOnCell": np.array(
                [[1, 2, -1], [0, 2, -1], [0, 1, -1]], dtype=np.int64
            ),
            "edgesOnCell": np.array(
                [[0, 3, 0], [0, 1, 0], [1, 3, 2]], dtype=np.int64
            ),
            "nEdgesOnCell": np.array([2, 2, 3], dtype=np.int64),
            "edgesOnEdge": np.array(
                [[1, 3, -1, -1], [0, 3, -1, -1], [-1, -1, -1, -1],
                 [0, 1, -1, -1]],
                dtype=np.int64,
            ),
            "nEdgesOnEdge": np.array([2, 2, 0, 2], dtype=np.int64),
            "verticesOnEdge": np.array(
                [[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int64
            ),
            "verticesOnCell": np.array(
                [[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int64
            ),
            "edgesOnVertex": np.array([[0, 1, 3], [0, 1, -1]], dtype=np.int64),
            "cellsOnVertex": np.array([[0, 1, 2], [0, 1, -1]], dtype=np.int64),
            "dcEdge": np.full(n_edges, 2.0, dtype=np.float64),
            "dvEdge": np.full(n_edges, 3.0, dtype=np.float64),
            "areaCell": np.full(n_cells, 5.0, dtype=np.float64),
            "areaTriangle": np.full(n_vertices, 7.0, dtype=np.float64),
            "weightsOnEdge": np.zeros((n_edges, 4), dtype=np.float64),
            "kiteAreasOnVertex": np.ones((n_vertices, 3), dtype=np.float64),
            "latCell": np.zeros(n_cells, dtype=np.float64),
            "lonCell": np.zeros(n_cells, dtype=np.float64),
            "latEdge": np.zeros(n_edges, dtype=np.float64),
            "lonEdge": np.zeros(n_edges, dtype=np.float64),
            "angleEdge": np.zeros(n_edges, dtype=np.float64),
            "meshDensity": np.ones(n_cells, dtype=np.float64),
            "fVertex": np.zeros(n_vertices, dtype=np.float64),
            "fEdge": np.zeros(n_edges, dtype=np.float64),
            "nominalMinDc": np.float64(2.0),
            "coeffs_reconstruct": (
                np.arange(n_cells * 3 * 3, dtype=np.float32).reshape(n_cells, 3, 3)
                + np.float32(1.0)
            ),
            "edgeNormalVectors": np.ones((n_edges, 3), dtype=np.float32),
        }
        self.attrs: dict[str, object] = {}

    def __getattr__(self, name: str):
        arrays = self.__dict__.get("arrays")
        if arrays is not None and name in arrays:
            return arrays[name]
        raise AttributeError(name)


def _masks() -> regional_v841.RegionalMasks:
    n_cells, n_edges, n_vertices = 3, 4, 2
    dtype = np.dtype(np.float32)
    return regional_v841.RegionalMasks(
        bdy_mask_cell=np.array([0, 6, 7], dtype=np.int64),
        bdy_mask_edge=np.array([0, 6, 7, 7], dtype=np.int64),
        bdy_mask_vertex=np.zeros(n_vertices, dtype=np.int64),
        spec_zone_mask_cell=np.array([0.0, 1.0, 1.0], dtype=dtype),
        spec_zone_mask_edge=np.array([0.0, 1.0, 1.0, 1.0], dtype=dtype),
        spec_zone_mask_vertex=np.zeros(n_vertices, dtype=dtype),
        nearest_relaxation_cell=np.full(n_cells, n_cells, dtype=np.int64),
        spec_cells=np.array([1, 2], dtype=np.int64),
        spec_edges=np.array([1, 2, 3], dtype=np.int64),
        relax_cells=np.zeros(0, dtype=np.int64),
        relax_edges=np.zeros(0, dtype=np.int64),
        nudged_cells=np.array([1, 2], dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# the padded mesh view the physics constructors are handed
# ---------------------------------------------------------------------------


def test_the_padded_view_restores_canonical_padding_in_inactive_edge_slots():
    """The prep's convention and the dycore's, on one array.

    ``PaddedRegionalMesh`` sends every negative connectivity entry to the
    garbage element, which is right for the sentinel slots the dycore gathers
    through and wrong for the slots past ``nEdgesOnCell``, where the loader's
    canonical padding is -1 and ``CudaMpasToPhysGeometryV841.from_host``
    requires exactly that.  The claim held here is that restoring them moves
    NO active entry: both sides loop ``slot < nEdgesOnCell``, and
    ``cuda_driver``'s own host validation checks ``[cell, :count]`` only.
    """

    mesh = _SyntheticCull()
    padded = PaddedRegionalHostMesh(mesh, _masks())
    counts = np.asarray(padded.nEdgesOnCell)
    edges = np.asarray(padded.edgesOnCell)
    active = np.arange(edges.shape[1])[None, :] < counts[:, None]

    assert set(np.unique(edges[~active]).tolist()) == {-1}
    assert np.all(edges[active] >= 0)
    assert np.all(edges[active] < padded.n_edges)

    base_edges = np.asarray(mesh.edgesOnCell)
    base_counts = np.asarray(mesh.nEdgesOnCell)
    base_active = np.arange(base_edges.shape[1])[None, :] < base_counts[:, None]
    assert np.array_equal(
        base_edges[base_active], edges[: base_edges.shape[0]][base_active]
    )
    # The garbage row is a cell with no edges, so every slot on it is
    # inactive and it reconstructs nothing.
    assert int(counts[-1]) == 0


def test_the_padded_view_keeps_every_cells_on_edge_entry_in_range():
    """``tend_toEdges`` refuses a negative endpoint, and would read cell -1."""

    padded = PaddedRegionalHostMesh(_SyntheticCull(), _masks())
    cells_on_edge = np.asarray(padded.cellsOnEdge)
    assert cells_on_edge.min() >= 0
    assert cells_on_edge.max() < padded.n_cells
    # The acoustic lane still needs the sentinel form, and keeps its own.
    sentinel = np.asarray(padded.arrays["cellsOnEdgeSentinel"])
    assert sentinel.shape == (padded.n_edges, 2)


def test_the_padded_view_carries_what_the_physics_seam_reads():
    """Reconstruction carriers ride along; ``DeviceMesh`` uploads neither."""

    from woof.hex.cuda_backend.containers import DeviceMesh

    mesh = _SyntheticCull()
    padded = PaddedRegionalHostMesh(mesh, _masks())

    coeffs = np.asarray(padded.coeffs_reconstruct)
    assert coeffs.shape == (4, 3, 3)
    counts = np.asarray(padded.nEdgesOnCell)
    active = np.arange(3)[None, :] < counts[:, None]
    # The prep contract: inactive slots are bitwise +0, and the garbage row
    # is entirely inactive.
    assert not np.any(coeffs[~active])
    assert not np.any(coeffs[-1])
    assert np.asarray(padded.edgeNormalVectors).shape == (5, 3)
    # DeviceMesh has no field for either, so carrying them costs the dycore
    # no device byte.  If that ever changes, this says so.
    fields = set(DeviceMesh.__dataclass_fields__)
    assert "coeffs_reconstruct" not in fields
    assert "edge_normal_vectors" not in fields


def test_the_padded_view_is_optional_about_the_reconstruction_carriers():
    """The dry lane's mesh carries neither, and must still build."""

    mesh = _SyntheticCull()
    del mesh.arrays["coeffs_reconstruct"]
    del mesh.arrays["edgeNormalVectors"]
    padded = PaddedRegionalHostMesh(mesh, _masks())
    assert not hasattr(padded, "coeffs_reconstruct")
    assert padded.n_cells == 4


def test_the_padded_view_declares_its_garbage_column_and_its_rings():
    padded = PaddedRegionalHostMesh(_SyntheticCull(), _masks())
    assert padded.garbage_cell == padded.n_cells - 1
    assert padded.is_regional is True
    assert padded.attrs["on_a_sphere"] is True
    for name in ("bdyMaskCell", "bdyMaskEdge", "bdyMaskVertex"):
        assert np.asarray(getattr(padded, name)).dtype == np.int32
    # The garbage element is ring 0: it is not a boundary element, it is not
    # an element at all.
    assert int(np.asarray(padded.bdyMaskCell)[-1]) == 0
    assert np.array_equal(
        np.asarray(padded.bdyMaskCell)[:-1], _masks().bdy_mask_cell
    )


# ---------------------------------------------------------------------------
# the ArWen statics on the padded extent
# ---------------------------------------------------------------------------


def test_the_physics_statics_pad_by_duplicating_a_real_column():
    """Pool zeros are not a column, and every physics table refuses them."""

    solve = 5
    values = {
        "n_columns": solve,
        "n_levels": 55,
        "landmask": np.arange(solve, dtype=np.float32),
        "soil_temperature": np.arange(4 * solve, dtype=np.float32).reshape(4, solve),
        "dx_column_m": np.full(solve, 4000.0, dtype=np.float32),
        "z_interface_nominal_m": np.arange(56, dtype=np.float32),
        "start_time": "not an array",
    }
    gwdo = {
        "meshDensity": np.linspace(0.1, 0.5, solve, dtype=np.float32),
        "nominalMinDc": np.float32(4000.0),
    }
    padded_values, padded_gwdo, receipt = pad_regional_physics_host(
        values, gwdo, n_cells_solve=solve
    )

    assert padded_values["n_columns"] == solve + 1
    assert padded_values["landmask"].shape == (solve + 1,)
    assert padded_values["landmask"][-1] == padded_values["landmask"][-2]
    assert padded_values["soil_temperature"].shape == (4, solve + 1)
    assert np.array_equal(
        padded_values["soil_temperature"][:, -1],
        padded_values["soil_temperature"][:, -2],
    )
    # dx_column_m must stay strictly positive or SealedArwenConstructorV841
    # refuses; meshDensity must stay in (0,1] or native_cell_dx_m does.
    assert float(padded_values["dx_column_m"][-1]) > 0.0
    assert 0.0 < float(padded_gwdo["meshDensity"][-1]) <= 1.0
    # A column PROFILE is shared, not per column, and must not be padded.
    assert padded_values["z_interface_nominal_m"].shape == (56,)
    assert padded_values["start_time"] == "not an array"
    assert receipt["convention"] == "duplicate-last-real-column"
    assert "landmask" in receipt["constructor_arrays_padded"]
    assert "meshDensity" in receipt["gwdo_arrays_padded"]
    assert "z_interface_nominal_m" not in receipt["constructor_arrays_padded"]


def test_the_physics_pad_refuses_a_mapping_that_is_not_the_solve_extent():
    with pytest.raises(ValueError, match="solve column count"):
        pad_regional_physics_host(
            {"n_columns": 9, "landmask": np.zeros(9, np.float32)},
            {},
            n_cells_solve=5,
        )


# ---------------------------------------------------------------------------
# the three defects the join uncovered
# ---------------------------------------------------------------------------


def test_the_garbage_discipline_leaves_the_physics_seam_alone():
    """One cache serves the dycore and the seam; only one wants scrubbing.

    THE BREAKAGE THIS PREVENTS, measured in
    ``evidence/regional-physics-20260826/``: the discipline rewrote the
    padded column of every float32 argument of every physics launch, so the
    preparation's own well-posed garbage column was zeroed between the kernel
    that wrote it and the validation that read it -- and its SOURCE arrays
    were zeroed too, because they are arguments.  No limited-area
    full-physics run could take one step.
    """

    # The fifth member is the solve-region check unit: its kernels read the
    # state (envelope, validations) or are the restoration itself (the fused
    # scrub), so a scrub after one of its launches would restore nothing and,
    # after the health gate, would write the default pool over rho_zz's unit
    # pool between the commit and the next begin_step.
    assert SELF_MANAGED_GARBAGE_MODULES == frozenset(
        {
            "hexcore.cuda_physics_prep_v841",
            "hexcore.cuda_physics_v841",
            "hexcore.cuda_gwdo_v841",
            "hexcore.cuda_arwen_physics_v841",
            "hexcore.cuda_solve_region_v841",
        }
    )
    # Every module in the set that resolves kernels does so under its own
    # name, so the discriminator is the module key and not a kernel-name
    # prefix that a new kernel could slip past.
    for module in ("cuda_physics_prep_v841", "cuda_physics_v841", "cuda_gwdo_v841"):
        source = (SRC / f"{module}.py").read_text(encoding="utf-8")
        assert f'module_key="hexcore.{module}"' in source, module

    body = _between(
        (SRC / "cuda_regional_forecast_v841.py").read_text(encoding="utf-8"),
        "    def scrub(",
        "    def receipt(",
    )
    assert "if module_key in SELF_MANAGED_GARBAGE_MODULES:" in body


def test_the_launch_observer_carries_the_module_that_launched():
    """The observer cannot discriminate without being told whose launch it is."""

    source = (SRC / "cuda_backend" / "runtime.py").read_text(encoding="utf-8")
    assert "hook(self._name, args, module_key=self._module_key)" in source
    assert "self._observe(kernel_key, kernel, name, stable_module_key)" in source
    assert "self._observe(kernel_key, cached, name, stable_module_key)" in source


def test_one_bdy_mask_digest_definition_survives():
    """Two classifiers that disagree about a mesh is the defect shape.

    ``cuda_driver.regional_bdy_mask_digest``'s docstring promised the mesh
    contract's digest and computed a second one.  MEASURED on r4.75.11020 the
    two spellings of the same three arrays were 2baf091d... and 0c2d9feb...;
    the anchor rows carried one and the registry rows the other, so
    ``require_regional_anchor``'s name/digest cross-check could never be
    satisfied by a registry-derived digest.
    """

    from woof.hex.cuda_backend.regional_admission import ADMITTED_REGIONS
    from woof.hex.cuda_driver import regional_bdy_mask_digest
    from woof.hex.mesh import (
        REGIONAL_BOUNDARY_MASK_NAMES,
        regional_boundary_mask_digest,
    )

    rng = np.random.default_rng(20260826)
    masks = {
        name: rng.integers(0, 8, size=17 + index).astype(np.int32)
        for index, name in enumerate(REGIONAL_BOUNDARY_MASK_NAMES)
    }

    class _Mesh:
        def __init__(self, arrays):
            self.arrays = dict(arrays)

        def __getattr__(self, name):
            arrays = self.__dict__.get("arrays")
            if arrays is not None and name in arrays:
                return arrays[name]
            raise AttributeError(name)

    assert regional_bdy_mask_digest(_Mesh(masks)) == regional_boundary_mask_digest(
        masks
    )
    # A closed mesh still reports None rather than the digest of nothing.
    assert regional_bdy_mask_digest(_Mesh({})) is None
    # A partial triple cannot identify the rings, and is refused rather than
    # quietly hashed to something.
    partial = {name: masks[name] for name in REGIONAL_BOUNDARY_MASK_NAMES[:2]}
    with pytest.raises(Exception):
        regional_bdy_mask_digest(_Mesh(partial))

    binding = _mesh_binding_module()
    for name, anchor in ADMITTED_REGIONS.items():
        assert (
            anchor.bdy_mask_sha256 == binding.MESH_BINDINGS[name].bdy_mask_sha256
        ), name


def test_the_boundary_nudge_only_drives_species_the_stream_carries():
    """Five planes past the end of the driving array is what this prevents.

    A dry regional run carries one passive qv and the stream carries one, so
    the model count and the driving count agreed and nothing was ever wrong.
    A full-physics run carries six WSM6 species and rw_mpas_lbc writes three,
    so the nudge read past the end of the driving array and wrote whatever
    was there into qc..qg at every cell with bdyMask > 1.  MEASURED on
    r4.75.11020: the forecast committed one step and the second reported
    |w| = 179.1 m/s at boundary ring 5.
    """

    from woof.hex.lbc import LBC_REQUIRED_VARIABLES, LBC_OPTIONAL_SCALAR_VARIABLES

    source = (SRC / "cuda_regional_forecast_v841.py").read_text(encoding="utf-8")
    body = _between(
        source, "    def _driven_tracer_count(", "    def moist_coefficients("
    )
    assert "if self.driven_scalars > model:" in body
    assert "return self.driven_scalars" in body
    # Both scalar boundary sites go through it; neither uses the model count.
    adjust = _between(
        source, "    def bdy_adjust_scalars(", "    def clamp_negative_scalars("
    )
    assert "ntracers = self._driven_tracer_count(scalars)" in adjust
    assert "int(scalars.shape[0])" not in adjust
    reset = _between(
        source, "    def reset_speczone_values(", "    def moist_coefficients("
    )
    assert "self._driven_tracer_count(scalars)" in reset
    # The stream's own contract says which species exist to be driven, so the
    # forecast door derives the list rather than hard-coding three.
    assert "lbc_qv" in LBC_REQUIRED_VARIABLES
    # Native Registry.xml:2154-2161 activates condensates by package.
    assert {"lbc_qc", "lbc_qr"} <= set(LBC_OPTIONAL_SCALAR_VARIABLES)
    assert "lbc_qi" not in LBC_REQUIRED_VARIABLES


# ---------------------------------------------------------------------------
# one route, not two
# ---------------------------------------------------------------------------


def test_the_physics_stack_is_constructed_once_for_both_lanes():
    """A parallel physics construction is the trap this lane exists to avoid.

    The regional branch in ``_construct_device_stack`` may choose the DRIVER
    (a limited-area driver carries a boundary runtime and a padded
    atmosphere) and the MESH VIEW it hands on.  It may not choose a different
    physics geometry, a different sealed constructor or a different backend:
    two constructions drift, and the drift arrives as a physics difference
    nobody meant to make.
    """

    source = (PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py").read_text(
        encoding="utf-8"
    )
    body = _between(
        source, "def _construct_device_stack(", "def _previous_surface_updates("
    )
    for constructor in (
        "CudaMpasToPhysGeometryV841.from_host(physics_mesh)",
        "CudaPhysicsGeometryV841.from_host(physics_mesh)",
        'CudaYsuGwdoStaticV841.from_host(host["gwdo_host"])',
        'SealedArwenConstructorV841.from_mapping(host["constructor_values"])',
    ):
        assert body.count(constructor) == 1, constructor
    # Exactly one branch, and it is about the driver and the mesh view.
    assert body.count("if regional is None:") == 1
    assert body.count("PersistentTwoPhaseCudaPhysicsBackendV841(") == 1


def test_the_step_loop_is_the_same_loop():
    """The boundary is a property of the driver, not a second step loop."""

    forecast = (PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py").read_text(
        encoding="utf-8"
    )
    body = _between(forecast, "def execute_forecast(", "def parse_args(")
    assert body.count("proof.execute_composite_step(") == 1
    # No regional branch inside the loop at all: every boundary site lives
    # behind ``driver.regional_v841`` in cuda_driver, which both lanes run.
    # (The receipt block AFTER the loop does read the runtime, to record what
    # drove the boundary; that is provenance, not a second execution path.)
    loop = _between(body, "for step in range(start_step + 1, steps + 1):", "    loop_seconds =")
    assert "regional" not in loop
    assert "lbc" not in loop


def test_a_history_frame_publishes_the_domain_not_the_allocation():
    """The garbage column is stripped before anything sees a frame."""

    source = (PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py").read_text(
        encoding="utf-8"
    )
    capture = _between(source, "def capture_snapshot(", "def _phase_from_receipt(")
    assert "if solve_cells is not None:" in capture
    assert "arrays[name] = np.ascontiguousarray(array[..., :keep])" in capture
    # And the strip happens before the physical gate, the hash projection and
    # the netCDF writer, which all read ``arrays``.
    assert capture.index("if solve_cells is not None:") < capture.index(
        "f000_overlay = None"
    )


# ---------------------------------------------------------------------------
# The scalar-requirement ladder fails closed over its own domain.
# ---------------------------------------------------------------------------

def _selection(mp_physics: int):
    from woof.hex.physics_seam import GPUWMSchemeSelection

    return GPUWMSchemeSelection(
        mp_physics=mp_physics, cu_physics=0, sf_surface_physics=0,
        bl_pbl_physics=0, sf_sfclay_physics=0, ra_lw_physics=0,
        ra_sw_physics=0, ra_rrtmg_variant="rte-rrtmgp",
        wrf_rrtmg_compatibility="none", cloud_fraction_scheme="off",
        mpas_names={})


def test_an_unrouted_selector_refuses_instead_of_declaring_qv_alone():
    """A selector with no scalar-requirement row must refuse, not fall through.

    THE BREAKAGE (measured 2026-08-29): the ladder's silence was not
    neutral -- a selector with no row fell through every arm and declared
    ``{"qv"}`` alone, so ``_validate_requirements`` accepted a batch with
    no cloud, no rain and no ice state, and the backend ran on state that
    was substituted rather than supplied, with no error anywhere.
    """
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex import physics_seam as ps

    # A selector with no reason row gets the generic fall-through text.
    # (The mp=50 reason row RETIRED 2026-08-31 with the defect it named:
    # the rime pair is now seam vocabulary, the mp_p3 route exists, and
    # the ladder's 50-arm requires all eight P3 scalars -- see
    # test_the_p3_row_requires_wrfs_exact_eight below.)
    with pytest.raises(ConfigurationRefusal) as caught:
        ps._required_scalar_names(_selection(10))
    assert "no scalar" in str(caught.value)
    assert "state substitution" in str(caught.value)


def test_every_routed_selector_still_declares_its_rows():
    """The refusal covers exactly the complement of the routed domain."""
    from woof.hex import physics_seam as ps

    assert ps._ROUTED_MP_PHYSICS == frozenset(
        ps._ROUTES["config_microp_scheme"].values())
    for mp in sorted(ps._ROUTED_MP_PHYSICS):
        names = ps._required_scalar_names(_selection(mp))
        if mp:
            assert "qv" in names
    assert 50 in ps._ROUTED_MP_PHYSICS


def test_the_p3_row_requires_wrfs_exact_eight():
    """mp=50 requires WRF's own P3 transport set -- no more, no less.

    module_microphysics_driver.F:1569-1602 binds EIGHT scalars into
    mp_p3_wrapper_wrf: qv, qc, qr, qi, ni, nr and the rime pair
    qir/qib.  A row missing the rime pair would accept a batch carrying
    no rime state (the state substitution the retired refusal existed
    to stop); a row demanding qs or qg would require two species P3's
    single ice category does not have.
    """
    from woof.hex import physics_seam as ps

    assert ps._required_scalar_names(_selection(50)) == frozenset(
        ("qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"))


def test_mp_p3_routes_as_declared_hexcore_vocabulary():
    """The mp_p3 name arrives, routes to 50, and stays out of MPAS's
    own inventory.

    The name is post-divergence hexcore vocabulary (the 2026-08-20
    physics-goes-its-own-way ruling), NOT an MPAS name: it must resolve
    through the seam, and it must never leak into
    MPAS_SELECTOR_INVENTORY, which records what MPAS itself accepts --
    a leak would make the frozen-Registry refusal text lie.
    """
    from woof.hex import physics_seam as ps

    selection = ps.resolve_mpas_physics(config_microp_scheme="mp_p3")
    assert selection.mp_physics == 50
    assert selection.mpas_names["config_microp_scheme"] == "mp_p3"
    assert "mp_p3" not in ps.MPAS_SELECTOR_INVENTORY["config_microp_scheme"]
    assert "mp_p3" in ps.HEXCORE_SELECTOR_EXTENSIONS["config_microp_scheme"]
    # The full requirement path is live end to end: mp-only selection,
    # so the backend requirements are the P3 scalars plus vertical wind.
    requirements = ps._gpuwm_requirements(selection)
    assert requirements.scalars == frozenset(
        ("qv", "qc", "qr", "qi", "ni", "nr", "qir", "qib"))
    assert "w" in requirements.atmosphere


def test_the_rime_pair_aliases_are_canonical():
    """qir/qib arrive under WRF Registry and module_mp_p3.F spellings."""
    from woof.hex import physics_seam as ps

    for alias, canonical in (("qir", "qir"), ("qirim", "qir"),
                             ("QIR", "qir"), ("qib", "qib"),
                             ("birim", "qib"), ("QIB", "qib")):
        assert ps._canonical_scalar_name(alias) == canonical


def test_an_unknown_scheme_name_still_refuses_with_both_vocabularies():
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex import physics_seam as ps

    with pytest.raises(ConfigurationRefusal) as caught:
        ps.resolve_mpas_physics(config_microp_scheme="mp_morrison")
    message = str(caught.value)
    assert "MPAS-A v8.2.3" in message
    assert "mp_p3" in message  # the extension vocabulary is named too
