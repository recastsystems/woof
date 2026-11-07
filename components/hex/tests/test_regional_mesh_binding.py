"""Regional row fields in the mesh-binding registry: table work, not code.

Registering a regional cull is a :class:`MeshBinding` row whose three
regional slots are filled -- ``boundary_zone_width`` (7 on every measured
native cull), ``bdy_mask_sha256`` (the digest of the grid's
``bdyMaskCell/Edge/Vertex`` triple), and ``lbc_source`` (nullable today).
``admit_regional_row`` cross-examines every row against the inspected grid
before any constant is rebound, and each refusal names the breakage.

When ``WOOF_HEX_REGIONAL_REFERENCE_DIR`` points at the native-culled
reference set, the digest and inspection legs run against the real bytes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from _layout import PACKAGE_DIR

TOOLS = Path(__file__).resolve().parents[1] / "tools"
DRIVERS = PACKAGE_DIR / "drivers"

REFERENCE_DIR_VARIABLE = "WOOF_HEX_REGIONAL_REFERENCE_DIR"


def _load_binding() -> object:
    name = "_test_regional_mesh_binding_module"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, DRIVERS / "mpas_mesh_binding.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def binding_module() -> object:
    return _load_binding()


def _row(binding_module: object, **overrides: object) -> object:
    fields = dict(
        name="regional-test-row",
        n_cells=2_971,
        n_edges=9_116,
        n_levels=55,
        n_interfaces=56,
        n_soil_levels=4,
        nominal_dx_m=120_000.0,
        dt_seconds=120.0,
        grid_bytes=1,
        grid_sha256="0" * 64,
        static_bytes=1,
        static_sha256="0" * 64,
    )
    fields.update(overrides)
    return binding_module.MeshBinding(**fields)


_REGIONAL_OBSERVED = {
    "regional_masks": {"present": True, "zone_width": 7, "sha256": "a" * 64}
}
_GLOBAL_OBSERVED = {"regional_masks": {"present": False}}


def test_a_regional_registration_is_a_row(binding_module: object) -> None:
    row = _row(
        binding_module,
        boundary_zone_width=7,
        bdy_mask_sha256="a" * 64,
        lbc_source=None,
    )
    assert row.regional
    assert not _row(binding_module).regional


def test_verified_atmospheric_extent_reaches_both_runtime_modules(
    binding_module: object,
) -> None:
    row = _row(binding_module, n_levels=59, n_interfaces=60)
    proof = SimpleNamespace(N_LEVELS=55, N_INTERFACES=56, N_SOIL_LEVELS=4)
    forecast = SimpleNamespace(N_LEVELS=55, N_INTERFACES=56, N_SOIL_LEVELS=4)
    binding_module._admit_vertical_dimensions(proof, row)
    receipt = binding_module._rebind_atmospheric_dimensions(proof, row, forecast)
    assert (proof.N_LEVELS, proof.N_INTERFACES) == (59, 60)
    assert (forecast.N_LEVELS, forecast.N_INTERFACES) == (59, 60)
    assert (proof.N_SOIL_LEVELS, forecast.N_SOIL_LEVELS) == (4, 4)
    assert receipt == {"N_LEVELS": [55, 59], "N_INTERFACES": [56, 60]}


@pytest.mark.parametrize(
    "overrides,breakage",
    [
        ({"n_levels": 2, "n_interfaces": 3}, "at least three layers"),
        ({"n_levels": 59, "n_interfaces": 59}, "one more interface"),
        ({"n_soil_levels": 9}, "wrong soil depths"),
        ({"frozen_native": True, "n_levels": 59, "n_interfaces": 60}, "no-op"),
    ],
)
def test_vertical_dimension_refusals_keep_the_runtime_unchanged(
    binding_module: object, overrides: dict[str, object], breakage: str,
) -> None:
    proof = SimpleNamespace(N_LEVELS=55, N_INTERFACES=56, N_SOIL_LEVELS=4)
    with pytest.raises(binding_module.MeshBindingMismatch, match=breakage):
        binding_module._admit_vertical_dimensions(proof, _row(binding_module, **overrides))
    assert (proof.N_LEVELS, proof.N_INTERFACES, proof.N_SOIL_LEVELS) == (55, 56, 4)


def test_imported_atmosphere_rebinds_the_gwd_carrier_guard(
    binding_module: object, monkeypatch,
) -> None:
    import numpy as np
    from woof.hex import cuda_gwdo_v841 as gwdo

    monkeypatch.setattr(gwdo, "X4_N_VERT_LEVELS", 55)
    arrays = {name: SimpleNamespace(shape=(59, 2), dtype=np.dtype("float32"))
              for name in ("u_p", "v_p", "t_p", "qv_p", "pres_hydd_p", "pi_p", "zmid_p")}
    columns = gwdo.CudaYsuGwdoColumnViewV841(
        **arrays, pres2_hydd_p=SimpleNamespace(shape=(60, 2), dtype=np.dtype("float32")),
        time_seconds=0.0, prep_contract_sha256=gwdo.CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
        _seal=gwdo._COLUMN_VIEW_SEAL,
    )
    with pytest.raises(ValueError, match="exact x4 YSU-GWDO needs"):
        columns.validate()
    checked = []
    def resident_shape(name, array, *, dtype, shape):
        assert array.dtype == np.dtype(dtype)
        assert array.shape == shape
        checked.append(shape)
    monkeypatch.setattr(gwdo, "require_resident_array", resident_shape)
    row = _row(binding_module, n_levels=59, n_interfaces=60)
    assert binding_module._rebind_gwdo_vertical_dimension(gwdo, row) == [55, 59]
    columns.validate()
    assert checked == [(59, 2)] * 7 + [(60, 2)]


def test_frozen_native_gwd_dimension_is_a_noop(binding_module: object) -> None:
    gwdo = SimpleNamespace(X4_N_VERT_LEVELS=55)
    row = _row(binding_module, frozen_native=True)
    assert binding_module._rebind_gwdo_vertical_dimension(gwdo, row) == [55, 55]
    assert gwdo.X4_N_VERT_LEVELS == 55
    with pytest.raises(binding_module.MeshBindingMismatch, match="layer guard moved"):
        binding_module._rebind_gwdo_vertical_dimension(
            SimpleNamespace(X4_N_VERT_LEVELS=59), row,
        )


def test_verified_gwd_extent_reaches_cuda_storage_and_receipt_hashes(
    binding_module: object, monkeypatch,
) -> None:
    import hashlib
    from woof.hex import cuda_gwdo_v841 as gwdo

    original = gwdo._CUDA_SOURCE
    before_sha = hashlib.sha256(original.encode()).hexdigest()
    monkeypatch.setattr(gwdo, "_CUDA_SOURCE", original)
    monkeypatch.setattr(gwdo, "CUDA_GWDO_V841_KERNEL_SHA256", before_sha)
    stale_kernels = {7: object()}
    monkeypatch.setattr(gwdo, "_KERNELS", stale_kernels)
    physics = SimpleNamespace(CUDA_GWDO_V841_KERNEL_SHA256=before_sha)
    config = SimpleNamespace(V841_GWDO_KERNEL_SHA256=before_sha)
    row = _row(binding_module, n_levels=59, n_interfaces=60)
    receipt = binding_module._rebind_gwdo_kernel_dimension(gwdo, row, physics, config)
    expected = original.replace("#define GWDO_NLEV 55", "#define GWDO_NLEV 59", 1)
    assert gwdo._CUDA_SOURCE == expected
    assert expected.replace("#define GWDO_NLEV 59", "#define GWDO_NLEV 55", 1) == original
    after_sha = hashlib.sha256(expected.encode()).hexdigest()
    assert receipt == {"layers": [55, 59], "sha256": [before_sha, after_sha]}
    assert gwdo.CUDA_GWDO_V841_KERNEL_SHA256 == after_sha
    assert physics.CUDA_GWDO_V841_KERNEL_SHA256 == after_sha
    assert config.V841_GWDO_KERNEL_SHA256 == after_sha
    assert stale_kernels == {}
    # Repeating an admitted binding preserves the compiled kernel.
    stale_kernels[8] = object()
    binding_module._rebind_gwdo_kernel_dimension(gwdo, row, physics, config)
    assert 8 in stale_kernels
    with pytest.raises(binding_module.MeshBindingMismatch, match="native GWD CUDA extent"):
        binding_module._rebind_gwdo_kernel_dimension(
            gwdo, _row(binding_module, frozen_native=True), physics, config,
        )


def test_frozen_native_gwd_cuda_storage_is_a_noop(binding_module: object) -> None:
    import hashlib
    from woof.hex import cuda_gwdo_v841 as gwdo

    original = gwdo._CUDA_SOURCE
    kernels = dict(gwdo._KERNELS)
    physics = SimpleNamespace(CUDA_GWDO_V841_KERNEL_SHA256=gwdo.CUDA_GWDO_V841_KERNEL_SHA256)
    config = SimpleNamespace(V841_GWDO_KERNEL_SHA256=gwdo.CUDA_GWDO_V841_KERNEL_SHA256)
    receipt = binding_module._rebind_gwdo_kernel_dimension(
        gwdo, _row(binding_module, frozen_native=True), physics, config,
    )
    digest = hashlib.sha256(original.encode()).hexdigest()
    assert receipt == {"layers": [55, 55], "sha256": [digest, digest]}
    assert gwdo._CUDA_SOURCE == original
    assert gwdo._KERNELS == kernels


def test_bound_gwd_hash_reaches_forecast_config_and_execution_validation(
    binding_module: object, monkeypatch,
) -> None:
    from woof.hex import config_v841, cuda_arwen_physics_v841, cuda_gwdo_v841
    from woof.hex.cuda_physics_v841 import CudaPhaseOneExecutionProvenanceV841, TransferStats
    from woof.hex.forecast_door import DRIVERS_DIR, _load_drivers

    original_sha = cuda_gwdo_v841.CUDA_GWDO_V841_KERNEL_SHA256
    for module, name in (
        (cuda_gwdo_v841, "_CUDA_SOURCE"),
        (cuda_gwdo_v841, "CUDA_GWDO_V841_KERNEL_SHA256"),
        (cuda_arwen_physics_v841, "CUDA_GWDO_V841_KERNEL_SHA256"),
        (config_v841, "V841_GWDO_KERNEL_SHA256"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(cuda_gwdo_v841, "_KERNELS", {})
    binding_module._rebind_gwdo_kernel_dimension(
        cuda_gwdo_v841, _row(binding_module, n_levels=59, n_interfaces=60),
        cuda_arwen_physics_v841, config_v841,
    )
    _, forecast = _load_drivers(DRIVERS_DIR)
    config = forecast.build_forecast_config(dt_seconds=120)
    config.validate()
    assert config.gwdo_kernel_sha256 == cuda_gwdo_v841.CUDA_GWDO_V841_KERNEL_SHA256
    kwargs = dict(
        aggregate_executed=True, gwdo_executed=True, gwdo_composed_once=True,
        gwdo_result_module=config_v841.V841_GWDO_RESULT_MODULE,
        gwdo_result_class=config_v841.V841_GWDO_RESULT_CLASS,
        gwdo_contract_sha256=config_v841.V841_GWDO_CONTRACT_SHA256,
        gwdo_kernel_sha256=config.gwdo_kernel_sha256,
        gwdo_validation_d2h=TransferStats(4, 0.0),
        gwdo_input_du_is_arwen_output=True, gwdo_input_dv_is_arwen_output=True,
        raw_du_is_gwdo_output=True, raw_dv_is_gwdo_output=True,
    )
    CudaPhaseOneExecutionProvenanceV841.arwen_with_external_gwdo(**kwargs).validate()
    kwargs["gwdo_kernel_sha256"] = original_sha
    with pytest.raises(ValueError, match="gwdo_kernel_sha256 changed"):
        CudaPhaseOneExecutionProvenanceV841.arwen_with_external_gwdo(**kwargs)


@pytest.mark.parametrize("max_edges", [6, 10])
def test_reconstruction_width_is_read_from_the_supplied_grid(
    binding_module: object, tmp_path: Path, max_edges: int,
) -> None:
    netcdf = pytest.importorskip("netCDF4")
    row = _row(binding_module)
    grid = tmp_path / "grid.nc"
    with netcdf.Dataset(grid, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.createDimension("nCells", row.n_cells)
        dataset.createDimension("nEdges", row.n_edges)
        dataset.createDimension("maxEdges", max_edges)
        dataset.createDimension("TWO", 2)
        counts = dataset.createVariable("nEdgesOnCell", "i4", ("nCells",))
        counts[:] = 6
        cells = dataset.createVariable("cellsOnEdge", "i4", ("nEdges", "TWO"))
        cells[:] = 1
    observed = binding_module._inspect_grid(grid, row)
    assert observed["maxEdges"] == max_edges
    assert observed["active_slots"] == row.n_cells * 6


def test_global_row_with_regional_grid_is_refused_by_name(
    binding_module: object,
) -> None:
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(_row(binding_module), _REGIONAL_OBSERVED)
    assert "declares no boundary zone" in str(refusal.value)
    assert "unforced boundary" in str(refusal.value)


def test_regional_row_with_global_grid_is_refused_by_name(
    binding_module: object,
) -> None:
    row = _row(
        binding_module, boundary_zone_width=7, bdy_mask_sha256="a" * 64
    )
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _GLOBAL_OBSERVED)
    assert "carries no bdyMask triple" in str(refusal.value)


def test_mask_digest_mismatch_is_refused_by_name(binding_module: object) -> None:
    row = _row(
        binding_module, boundary_zone_width=7, bdy_mask_sha256="b" * 64
    )
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert "bdyMask triple SHA-256" in str(refusal.value)
    assert "wrong cells" in str(refusal.value)


def test_zone_width_mismatch_is_refused_by_name(binding_module: object) -> None:
    row = _row(
        binding_module, boundary_zone_width=5, bdy_mask_sha256="a" * 64
    )
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert "outermost ring" in str(refusal.value)


def test_half_declared_regional_row_is_refused_by_name(
    binding_module: object,
) -> None:
    row = _row(binding_module, boundary_zone_width=7)
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert "all-or-nothing" in str(refusal.value)


def test_empty_lbc_slot_refuses_execution_by_name(binding_module: object) -> None:
    row = _row(
        binding_module,
        boundary_zone_width=7,
        bdy_mask_sha256="a" * 64,
        lbc_source=None,
    )
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert "lbc_source slot is empty" in str(refusal.value)
    assert "unforced boundary" in str(refusal.value)


def test_filled_regional_row_admits(binding_module: object) -> None:
    row = _row(
        binding_module,
        boundary_zone_width=7,
        bdy_mask_sha256="a" * 64,
        lbc_source="registered-boundary-stream",
    )
    receipt = binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert receipt["regional"] is True
    assert receipt["lbc_source"] == "registered-boundary-stream"


def test_global_row_with_global_grid_is_untouched(binding_module: object) -> None:
    receipt = binding_module.admit_regional_row(
        _row(binding_module), _GLOBAL_OBSERVED
    )
    assert receipt == {"regional": False}


# ---------------------------------------------------------------------------
# the real bytes: digest and inspection against the native-culled reference
# ---------------------------------------------------------------------------
def _reference_grid() -> Path:
    root = os.environ.get(REFERENCE_DIR_VARIABLE, "")
    if not root or not Path(root).is_dir():
        pytest.skip(
            f"{REFERENCE_DIR_VARIABLE} does not point at the native-culled "
            "regional reference set"
        )
    candidates = sorted((Path(root) / "cull-x1").glob("*.nc"))
    if len(candidates) != 1:
        pytest.skip("reference set does not hold exactly one quick-cull grid")
    return candidates[0]


def test_regional_mask_digest_is_computable_from_the_real_grid(
    binding_module: object,
) -> None:
    digest = binding_module.regional_mask_digest(_reference_grid())
    assert len(digest) == 64 and digest == digest.lower()
    # deterministic: the digest a row pins is reproducible read-over-read
    assert binding_module.regional_mask_digest(_reference_grid()) == digest


def test_inspect_grid_observes_the_real_mask_triple(binding_module: object) -> None:
    grid = _reference_grid()
    row = _row(binding_module)
    observed = binding_module._inspect_grid(grid, row)
    masks = observed["regional_masks"]
    assert masks["present"] is True
    assert masks["zone_width"] == 7
    assert masks["sha256"] == binding_module.regional_mask_digest(grid)
    # and the admission path sees exactly what a registered row would verify
    regional_row = _row(
        binding_module,
        boundary_zone_width=7,
        bdy_mask_sha256=masks["sha256"],
        lbc_source="registered-boundary-stream",
    )
    receipt = binding_module.admit_regional_row(regional_row, observed)
    assert receipt["bdy_mask_sha256"] == masks["sha256"]


def test_global_digest_request_is_refused_by_name(
    binding_module: object, tmp_path: Path
) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    path = tmp_path / "not-regional.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", 3)
        variable = dataset.createVariable("nEdgesOnCell", "i4", ("nCells",))
        variable[:] = [6, 6, 6]
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.regional_mask_digest(path)
    assert "not a regional cull" in str(refusal.value)


_GF_CLAIM = (
    "one uninterrupted real-initialized conus12km MPAS-A v8.4.2 CUDA forecast "
    "using WSM6 + GF + YSU + external YSU-GWDO + revised-MO + NoahMP"
)


def test_claim_names_new_tiedtke_when_the_decision_selects_it(binding_module: object) -> None:
    # Breakage: the rrfs-h-h2-score conus12km New Tiedtke run's receipt claimed GF.
    claim = binding_module.claim_for_convection(_GF_CLAIM, "cu_ntiedtke")
    assert "WSM6 + New Tiedtke + YSU" in claim
    assert "GF" not in claim


def test_claim_is_byte_identical_for_gf_and_drops_the_term_when_off(binding_module: object) -> None:
    assert binding_module.claim_for_convection(_GF_CLAIM, "cu_grell_freitas") is _GF_CLAIM
    assert "WSM6 + YSU" in binding_module.claim_for_convection(_GF_CLAIM, "off")


def test_claim_refuses_a_scheme_it_cannot_name(binding_module: object) -> None:
    # A name no selection can resolve.  It was cu_kain_fritsch, which passes
    # only until another test in the same process resolves a Kain-Fritsch
    # selection (physics_suites registers its label): order-dependent
    # (release review M4).
    with pytest.raises(ValueError, match="no claim label"):
        binding_module.claim_for_convection(_GF_CLAIM, "cu_never_resolved_by_any_selection")
