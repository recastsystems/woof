from __future__ import annotations

import dataclasses
from contextlib import contextmanager
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import weakref

import numpy as np
import pytest
from _layout import PACKAGE_DIR


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py"



@pytest.fixture(autouse=True)
def fake_phase_release(monkeypatch: pytest.MonkeyPatch):
    """NumPy transaction fakes must not query a real CUDA stream.

    The memory-release implementation has its own stand-in CuPy contract tests
    in test_device_memory_release.py. This file exercises transaction ownership
    and rollback with host arrays, including when real CuPy is installed.
    """
    from woof.hex import device_memory_release

    boundaries: list[tuple[str, str]] = []

    @contextmanager
    def phase(name: str):
        boundaries.append(("enter", name))
        try:
            yield
        finally:
            boundaries.append(("exit", name))

    monkeypatch.setattr(device_memory_release, "phase_release", phase)
    return boundaries


def _load_runner() -> object:
    name = "_test_run_cuda_v841_full_physics_x4"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_exact_release_scope_and_all_execution_sources_are_frozen() -> None:
    runner = _load_runner()
    assert str(runner.SRC) in sys.path
    assert runner.SOURCE_RELEASE == "v8.4.1"
    assert runner.ARWEN_COMMIT == "26daaab7ef5c1104166fe61503cdd9487750f1af"
    assert (runner.N_CELLS, runner.N_EDGES, runner.N_LEVELS) == (
        163_842,
        491_520,
        55,
    )
    assert runner.SCALAR_NAMES == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert runner.DT_SECONDS == 120.0
    assert runner.FULL_STEPS == 30
    assert runner.CHECKPOINT_STEP == 15
    assert runner.ARWEN_CONTRACT_SURFACE_SHA256 == (
        "45de852c6f8a8951d7e3f65b2b1e64462cf213d50707db4e52cb7eba17491ff2"
    )
    assert runner.ARWEN_GLACIER_COMPOSED_TU_SHA256 == (
        "edafcac585d4786c0cdfddf07f8e767b64d0d40b6db0e4da3dc3b2fa8c21fb59"
    )
    assert float(runner.ARWEN_XICE_THRESHOLD) == float(np.float32(0.02))
    assert runner.NEGATIVE_QV_PIN["negative_count"] == 215
    assert runner.NEGATIVE_QV_PIN["full_qv_sha256"] == (
        "c0180afa0fa99253f414472d8f767421c85b1324f67e161fdef9f6244b565099"
    )
    assert runner.NEGATIVE_QV_PIN["negative_indices_sha256"] == (
        "b622daead41e76d988e5419d92bac9b28a2390e8cfdb619d38ff19c1188a10a4"
    )
    assert runner.NEGATIVE_QV_PIN["negative_values_sha256"] == (
        "192e06fa6a1302eb06f7148e0cb85483ca61a21caf0eed4bc95ccfcb3452eb4b"
    )
    assert runner.unresolved_source_pins() == ()
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in runner.EXECUTION_SOURCE_PINS.values()
    )
    receipt = runner.require_frozen_execution_sources()
    assert set(receipt["files"]) == set(runner.EXECUTION_SOURCE_PINS)


def test_frozen_source_hashes_and_adapter_contract_are_exact() -> None:
    """The runner's twenty-three pins, spelled out a second time on purpose.

    This mirror is what makes a pin move REVIEWABLE: the runner's table can be
    edited by anybody, and without a second copy in the test suite the edit and
    its justification travel in one diff hunk that a reviewer skims.  The two
    copies are re-derived from the TREE by ``tools/repin_source_tables.py``,
    each independently and never one from the other -- a mirror generated from
    its original agrees with it by construction and gates nothing.
    """

    runner = _load_runner()
    assert runner.EXECUTION_SOURCE_PINS == {
        "src/hexcore/diagnostic_export.py": "e8d86569ff5f592465d00a6764eb354fb2e178d297edf479185b907529c18d16",
        "src/hexcore/snapshot_transfer.py": "7064e74719933894c9d92dbfe06190b404801cd72ecc5e4fe2447b66124c5efb",
        "src/hexcore/history_ready.py": "287840b7ece3cb97185340f3e161c997bbfd9a11233ff5da4198cca42e1f9523",
        "src/hexcore/history_writer.py": "570c9124745a8d8325a35c11da17e58036bfcf2e3117a5be502dec196a9905f0",
        "src/hexcore/cuda_arwen_helper_scope_v2.py": "641e8b77427a7c2f8eff374daa37e09a034fc45271d1dac67c3e39d03e3814e9",
        "src/hexcore/radiation_source_sets.json": "4fdb41a8102a0c4891771ed476d2eb7cf870e89a66c1a2dd4a92b8cd42dac802",
        "src/hexcore/data/profiles/engine-helper-sets.v1.json": "fb9988bda21d9447b6aba277c6b394da0ffc1727e4662c894bd6279b92abda86",
        "src/hexcore/step_anchors.py": (
            "13f8bf3361b3fc5c25805d05d48e3c59841050e3d372c03c96f1498133399b72"
        ),
        "src/hexcore/data/dycore-step-anchors.v1.json": (
            "f9a102a3f4fd848e95bf9fabe5a5ccd0579ba01de739de9e7da8e4ed7930e62d"
        ),
        "src/hexcore/native/upper_damping.rs": "5483b2c1e7da45317ffe4213c5879b583070d9e28d591ec52a9c697a052e81f1",
        "src/hexcore/upper_damping_v841.py": "947f477d5f56b41d95fb5a0b92adb21d68be95b5fac19ba524181cb373710860",
        "src/hexcore/cuda_upper_damping_v841.py": "04f0b5b450c088575e589de943b7b44fff10b442d01ae0c0acc0a5e848748545",
        "src/hexcore/cuda_physics_prep_v841.py": (
            "fffb12b21f67e51e045c50dc739e2259b962ad28cf99d829211beb92c72b8ef3"
        ),
        "src/hexcore/cuda_gwdo_v841.py": (
            "8dc6e9a3d8e88c231bc8d738f80a1655839aed2478b56ccc5038ab4daba7cfb5"
        ),
        "src/hexcore/cuda_physics_v841.py": (
            "18ac060a3adf6884eef7b853cb4b8fab76b37a6ec561c545372515d7f7e71672"
        ),
        # Re-frozen 2026-08-26 for the convection ruling: the physics cadence
        # table reports convection: None when no scheme is selected; and again
        # 2026-08-26 for the limited-area physics lane, which retired the
        # second bdyMask digest convention.  Both rationales live beside the
        # runner's pin.
        # Re-frozen 2026-08-27 for the provenance scrub: provenance is
        # now rewritten out of the shipped source in THIS tree rather than on
        # the public copy, because the copy carries these pins with it and one
        # of the two digest gates is an admission-time refusal, not a test.
        # Measured inert by AST; the rationale lives beside the runner's pin.
        # Re-frozen 2026-10-05 for the startup lane (host-preparation cache);
        # the rationale lives beside the runner's pin.
        "src/hexcore/cuda_driver.py": (
            "e8d3c8484928c85f126dfce6bacb35ba215cf59dc7005ded780b96aa639f0f42"
        ),
        # New pin (startup lane, 2026-10-05): the frozen driver executes host
        # preparation through this module; the rationale lives beside the
        # runner's pin.
        "src/hexcore/prep_cache.py": (
            "9a02aded2bf3f708589aa1006c2ac042cffa145d48e58841b062f1f253280bcf"
        ),
        "src/hexcore/cuda_backend/recovery.py": (
            "9765e0d72853a2af1d4d7acf18db45e2c09e9a89ed3098e392a8ec8ebaab98ab"
        ),
        # Re-frozen for the regional CPU authority lane, again
        # for timestep admission, and again 2026-08-26 for the
        # convection ruling; the rationale lives beside the runner's pin.
        # Re-frozen 2026-08-27 for the provenance scrub: provenance is
        # now rewritten out of the shipped source in THIS tree rather than on
        # the public copy, because the copy carries these pins with it and one
        # of the two digest gates is an admission-time refusal, not a test.
        # Measured inert by AST; the rationale lives beside the runner's pin.
        "src/hexcore/config_v841.py": (
            "1cee9d6702785a4226805ab133aa7e7112ac6fcdb234cf3eef44a87e2c3a2933"
        ),
        # New pin (convection ruling, 2026-08-26): the frozen config admits
        # config_convection_scheme from this module and the timestep registry
        # keys on the selection it produces; the rationale lives beside the
        # runner's pin.
        # Re-frozen 2026-08-27 for the provenance scrub: provenance is
        # now rewritten out of the shipped source in THIS tree rather than on
        # the public copy, because the copy carries these pins with it and one
        # of the two digest gates is an admission-time refusal, not a test.
        # Measured inert by AST; the rationale lives beside the runner's pin.
        # Re-frozen 2026-10-11: variable-resolution meshes decided per cell;
        # the rationale lives beside the runner's pin.
        "src/hexcore/convection_admission.py": (
            "33372d55a763e50428d3742af895d19c9ff8c000ef7814ee52ac757554e602f0"
        ),
        # New pin (the timestep-anchor campaign): the frozen config ADMITS config_dt from this
        # module, so the pinned config's meaning depends on these bytes.
        # Re-frozen 2026-08-26: ADMITTED_TIMESTEPS gained the four anchors
        # ruled for that day (100, 75, 20, 5 s), each naming the campaign
        # that earned it and each recording native_reference=None.  Table
        # only; no executable behaviour outside it changed.
        # Re-frozen again 2026-08-26: the registry is keyed by the
        # CONFIGURATION (dt, cumulus selection), not by the timestep alone.
        # Re-frozen a third time 2026-08-26: that key gained its third
        # fragment, the surface/PBL cadence, so a run holding the cadence
        # cannot occupy the welded row's slot; the rationale lives beside
        # the runner's pin.
        # Re-frozen 2026-08-27 for the provenance scrub: provenance is
        # now rewritten out of the shipped source in THIS tree rather than on
        # the public copy, because the copy carries these pins with it and one
        # of the two digest gates is an admission-time refusal, not a test.
        # Measured inert by AST; the rationale lives beside the runner's pin.
        # Re-frozen 2026-10-14 (release hygiene N7): the 75 s anchor's evidence
        # citations name the lab host neutrally; the rationale lives beside the
        # runner's pin.
        "src/hexcore/dt_admission.py": (
            "77abfa300cf131704d56914011771993ee4574b489f6a4b54d32a8f742b17d20"
        ),
        # New pin (surface/PBL cadence, 2026-08-26): this module decides
        # config_bldt_seconds, a knob the frozen registry keys on; the
        # rationale lives beside the runner's pin.
        # Re-frozen 2026-08-27 for the provenance scrub: provenance is
        # now rewritten out of the shipped source in THIS tree rather than on
        # the public copy, because the copy carries these pins with it and one
        # of the two digest gates is an admission-time refusal, not a test.
        # Measured inert by AST; the rationale lives beside the runner's pin.
        "src/hexcore/pbl_cadence.py": (
            "f9c86d3ec05d98b4e80df0270d3fde1b31a5949512772f1177c4b34f7eeb82d3"
        ),
        # Re-frozen 2026-08-28 for the 2.5.8 engine repin: 77f831b rewrote
        # this module's own ARWEN_SOURCE_MANIFEST/ARWEN_BUILD_COMMIT and did
        # not carry its new digest into either copy of the table, which left
        # the x4 proof refusing before CUDA.  Re-frozen again 2026-09-13
        # when the checkout guard learned to resolve an admitted engine's
        # identity.  Re-frozen again 2026-09-27 when the per-step boundary
        # export became declinable (default armed).  The rationale lives
        # beside the runner's pin.
        "src/hexcore/cuda_arwen_physics_v841.py": (
            "58f169eb8841f81e693bfa5297e67ee309dd61a552413e8b4da234325a50ea99"
        ),
        # New pin 2026-10-05 (New Tiedtke row); the rationale lives beside the
        # runner's pin.
        "src/hexcore/cuda_ntiedtke_column_v841.py": (
            "5b2016c35060ba14b17bbc7356e6cf2f8b9c59cd5fc5517f1c99ebe4620d531c"
        ),
        "src/hexcore/cuda_seam_snapshot_v841.py": (
            "a05fdda5ef82b0d77b1af63d4354953660f3d5b80470d7915bb647f41d191019"
        ),
        # Re-frozen for the regional CPU authority lane; the
        # rationale lives beside the runner's pin.
        "src/hexcore/mixing.py": (
            "1fd749a6c2a8599acf9bac7b2fc9764c0cb8ceb428be50b19a4e93e3a094d554"
        ),
        # Re-frozen for the regional CPU authority lane, and again
        # 2026-10-05 for the startup lane (host-preparation cache); the
        # rationale lives beside the runner's pin.
        "src/hexcore/mixing_v841.py": (
            "023df21823186b0820c21a1e15f8a9ceb662bd7c77fc4e86e469202c4ec7c749"
        ),
        "src/hexcore/cuda_horizontal.py": (
            "f94137176db65d24df0d9ff1d491c278d6c835bb08cd76dc43c5cfcbf323a345"
        ),
        "src/hexcore/cuda_horizontal_v841.py": (
            "37714ab89935af5b02d79eeea4c06cfad038de457502116c9dad27a8bdd55465"
        ),
        # Current d15ea3482cc5 cache-key correction; archived proofs unchanged.
        "src/hexcore/cuda_transport_v841.py": (
            "178246f5032b512717edf764edaa6d1ddec6f04241f329e6cb6acffe9112acb6"
        ),
        "src/hexcore/cuda_transport.py": (
            "013ed46f8ef3bcc1a445b2e81135e262c23c49950e52e5b62b22e4274644b15e"
        ),
        "src/hexcore/cuda_fp32.py": (
            "ba6b962cac42454846b96d87bba73ef3ceedbf69cdf8ae6a7f880b31f25cc28e"
        ),
        "src/hexcore/partition_assets_v841.py": (
            "ddc7054ba296f5ae6d3e17a601276f698e37f622beb066e17a47f919ba79040b"
        ),
        "src/hexcore/partition_local_mesh_v841.py": (
            "3d860e5bc9695173bcf05efaaea40fdca8e545ae0f88a26a250befbacb3ee3fe"
        ),
        "src/hexcore/partition_state_v841.py": (
            "a504e6f5c5abc2014d40e4a8e3e89885f97d6456ff428c585fbaf57910eccbe8"
        ),
        "src/hexcore/partition_device_scheduler_v841.py": (
            # re-frozen 2026-08-25: admission floors rerouted through
            # device_admission (stale-guard audit, finding 5)
            "0ad0d541335ffaef95d385c81fdc01c9eaa2d65dff0b3a6d8b442f2812fdfeeb"
        ),
        "src/hexcore/partition_executor_v841.py": (
            "9821e7e3b7c7238d48e40edadbc2c99afbf7e157d08b078649b51cef2c445009"
        ),
        "src/hexcore/partition_net_v841.py": (
            "707a7c455641c45a7e6c42b72c40272cfb184258177feba7a6c8e70414d45134"
        ),
    }


def test_every_restated_contract_digest_equals_the_module_that_computes_it() -> None:
    """``KNOWN_CONTRACT_PINS`` and ``config_v841`` measured, not restated.

    THE BREAKAGE THIS PREVENTS, and it is not hypothetical: this test used to
    assert three of those rows against a second copy of the same literals, so
    it compared the table to itself and passed while every row in it was
    wrong.  All four contract digests are computed at import over a document
    that frames a MODULE NAME into itself, the 0.2.0 rename moved all four,
    and nothing here noticed.  What did notice was a card: the first composite
    step of every full-physics forecast refused with

        ValueError: external GWDO provenance gwdo_contract_sha256 changed:
        <what the module computed> != <what config_v841 restated>

    raised by ``CudaPhaseOneExecutionProvenanceV841.validate`` against
    ``config_v841.V841_GWDO_CONTRACT_SHA256``.  A pin that only a card-tier
    gate reads is a pin that goes stale silently and fails on the node at the
    start of a booked run, so this one now has a CPU-tier gate.

    Two of the six rows the sweep moved had been stale since BEFORE the
    rename and are recorded here so nobody re-derives them a second time
    wondering what happened.  ``coupling_kernel_sha256`` spelled a value
    that appears in no source and in no receipt in this tree, while all 564
    archived receipts carrying that field report the computed one; and
    ``adapter_contract_sha256`` spelled the value that was live until the
    2.5.8 engine re-pin moved the adapter authority document under it.

    The re-derivation lives in ``tools/repin_source_tables.py``
    (``RESTATED_CONTRACT_DIGESTS``); this is its gate.
    """

    from woof.hex import config_v841
    from woof.hex import cuda_arwen_physics_v841
    from woof.hex import cuda_gwdo_v841
    from woof.hex import cuda_physics_prep_v841
    from woof.hex import cuda_physics_v841

    runner = _load_runner()

    assert runner.KNOWN_CONTRACT_PINS["prep_contract_sha256"] == (
        cuda_physics_prep_v841.CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256
    )
    assert runner.KNOWN_CONTRACT_PINS["prep_kernel_sha256"] == (
        cuda_physics_prep_v841.CUDA_PHYSICS_PREP_V841_KERNEL_SHA256
    )
    assert runner.KNOWN_CONTRACT_PINS["gwdo_contract_sha256"] == (
        cuda_gwdo_v841.CUDA_GWDO_V841_CONTRACT_SHA256
    )
    assert runner.KNOWN_CONTRACT_PINS["gwdo_kernel_sha256"] == (
        cuda_gwdo_v841.CUDA_GWDO_V841_KERNEL_SHA256
    )
    assert runner.KNOWN_CONTRACT_PINS["coupling_contract_sha256"] == (
        cuda_physics_v841.CUDA_PHYSICS_V841_CONTRACT_SHA256
    )
    assert runner.KNOWN_CONTRACT_PINS["coupling_kernel_sha256"] == (
        cuda_physics_v841.CUDA_PHYSICS_V841_KERNEL_SHA256
    )
    assert runner.KNOWN_CONTRACT_PINS["adapter_contract_sha256"] == (
        cuda_arwen_physics_v841.CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256
    )

    # The one the forecast lane actually reads, and the reason this test
    # exists: ``validate`` compares the provenance the adapter built against
    # THIS constant, so a disagreement here is a dead forecast lane.
    assert config_v841.V841_GWDO_CONTRACT_SHA256 == (
        cuda_gwdo_v841.CUDA_GWDO_V841_CONTRACT_SHA256
    )
    assert config_v841.V841_GWDO_KERNEL_SHA256 == (
        cuda_gwdo_v841.CUDA_GWDO_V841_KERNEL_SHA256
    )

    # The module the document names is the module the document lives in.
    # This is what made all four digests move at the rename, and it is what
    # a future rename must keep true.
    assert config_v841.V841_GWDO_RESULT_MODULE == cuda_gwdo_v841.__name__


def test_landmask_cast_and_exhaustive_sealed_constructor_audit_are_exact() -> None:
    from netCDF4 import Dataset
    from woof.hex import cuda_arwen_physics_v841 as adapter

    runner = _load_runner()
    pin = runner.LANDMASK_CONSTRUCTOR_CAST_PIN
    assert pin["source_dimensions"] == ("nCells",)
    assert pin["shape"] == (163_842,)
    assert pin["source_dtype"] == "<i4"
    assert pin["source_array_sha256"] == (
        "6aee8da961b605b9aa21362840aec214beaa384d42041e232ff8f8a9f60b22b5"
    )
    assert pin["source_unique_values"] == (0, 1)
    assert pin["target_dtype"] == "<f4"
    assert pin["target_array_sha256"] == (
        "31a7e78b7f0a6b10d719df443b0e2d629e60028c8ac628c5ad8556a87c5eae65"
    )
    assert pin["target_uint32_values"] == (0, 1_065_353_216)
    assert "landmask" in adapter._SURFACE_FLOAT_FIELDS
    assert adapter._SURFACE_INT_FIELDS == ("ivgtyp", "isltyp")
    assert adapter._SOIL_FIELDS == ("soil_temperature", "soil_moisture")
    assert set(adapter._CONSTRUCTOR_ARRAY_FIELDS) == (
        set(adapter._SURFACE_FLOAT_FIELDS)
        | set(adapter._SURFACE_INT_FIELDS)
        | set(adapter._SOIL_FIELDS)
        | {"z_interface_nominal_m", "dx_column_m"}
    )
    # GF's native geometry/shallow boundary is part of the sealed mapping:
    # a per-cell dx vector and the hardwired ishallow=1.
    assert "dx_column_m" in adapter._CONSTRUCTOR_KEYS
    assert "gf_ishallow" in adapter._CONSTRUCTOR_KEYS

    with Dataset(runner.default_authority_paths()["init"], "r") as dataset:
        variable = dataset.variables["landmask"]
        variable.set_auto_maskandscale(False)
        source = np.ascontiguousarray(variable[...])
    target = np.ascontiguousarray(source, dtype=np.float32)
    assert source.dtype.str == pin["source_dtype"]
    assert runner.array_sha256(source) == pin["source_array_sha256"]
    assert tuple(int(value) for value in np.unique(source)) == (0, 1)
    assert target.dtype.str == pin["target_dtype"]
    assert runner.array_sha256(target) == pin["target_array_sha256"]
    assert tuple(int(value) for value in np.unique(target.view(np.uint32))) == (
        0,
        1_065_353_216,
    )
    assert np.array_equal(target.astype(np.int32), source)

    builder = inspect.getsource(runner.build_arwen_constructor_values)
    for token in (
        "source_landmask_sha256",
        "source_unique_values",
        "target_uint32_values",
        "value_preserving_exact_fp32_cast",
        '"landmask": landmask',
        '"xland": xland',
        '"xice_threshold": float(ARWEN_XICE_THRESHOLD)',
        "surface_classification_receipt",
    ):
        assert token in builder
    preparation = inspect.getsource(runner._prepare_host_execution)
    mapping = preparation.index("build_arwen_constructor_values")
    audit = preparation.index("SealedArwenConstructorV841.from_mapping")
    returned = preparation.index('"constructor_values": constructor_values')
    assert mapping < audit < returned
    assert "all_required_keys_dtypes_shapes_validated" in preparation

def test_real_x4_native_xland_classification_and_cuda_census_are_exact() -> None:
    from netCDF4 import Dataset

    runner = _load_runner()
    with Dataset(runner.default_authority_paths()["init"], "r") as dataset:
        for name in ("xland", "xice", "ivgtyp", "landmask"):
            dataset.variables[name].set_auto_maskandscale(False)
        xland_source = np.ascontiguousarray(dataset.variables["xland"][...])
        xice_source = np.ascontiguousarray(dataset.variables["xice"][...])
        ivgtyp = np.ascontiguousarray(dataset.variables["ivgtyp"][...])
        landmask = np.ascontiguousarray(dataset.variables["landmask"][...])
    assert xland_source.dtype == np.dtype(np.float32)
    assert xland_source.shape == (1, runner.N_CELLS)
    assert runner.array_sha256(xland_source) == runner.NATIVE_XLAND_SOURCE_SHA256
    xland = np.ascontiguousarray(xland_source[0])
    xice = np.ascontiguousarray(xice_source[0])
    assert runner.array_sha256(xland) == runner.NATIVE_XLAND_FLAT_SHA256
    sea_ice = xice >= runner.ARWEN_XICE_THRESHOLD
    open_water = (xland >= np.float32(1.5)) & ~sea_ice
    land = ~(sea_ice | open_water)
    glacier = land & (ivgtyp == np.int32(15))
    sflx_land = land & ~glacier
    observed = {
        "xland_source": "native",
        "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
        "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
        "xice_threshold": float(runner.ARWEN_XICE_THRESHOLD),
        "sea_ice_columns": int(np.count_nonzero(sea_ice)),
        "open_water_columns": int(np.count_nonzero(open_water)),
        "sflx_land_columns": int(np.count_nonzero(sflx_land)),
        "glacier_columns": int(np.count_nonzero(glacier)),
    }
    assert observed == dict(runner.EXPECTED_SURFACE_CLASSIFICATION)
    glacier_indices = np.ascontiguousarray(np.flatnonzero(glacier))
    sea_ice_indices = np.ascontiguousarray(np.flatnonzero(sea_ice))
    delta_indices = np.ascontiguousarray(
        np.flatnonzero(
            (xice >= runner.ARWEN_XICE_THRESHOLD) & (xice < np.float32(0.5))
        )
    )
    assert int(glacier_indices[0]) == 30
    assert runner.array_sha256(glacier_indices) == runner.GLACIER_INDEX_SHA256
    assert runner.array_sha256(sea_ice_indices) == runner.SEA_ICE_INDEX_SHA256
    assert runner.array_sha256(delta_indices) == runner.THRESHOLD_DELTA_INDEX_SHA256
    assert np.all(ivgtyp[sea_ice] == 15)
    assert np.all(landmask[sea_ice] == 0)
    assert np.all(xland[sea_ice] == np.float32(1.0))
    assert np.all(landmask[glacier] == 1)
    assert np.all(xland[glacier] == np.float32(1.0))
    f000 = runner.require_arwen_v2_surface_execution(
        {"surface_classification": observed, "last_noahmp_census": None},
        executed=False,
        label="F000",
    )
    assert f000["last_noahmp_census"] is None
    executed = runner.require_arwen_v2_surface_execution(
        {
            "surface_classification": observed,
            "last_noahmp_census": dict(runner.EXPECTED_NOAHMP_CENSUS),
        },
        executed=True,
        label="F001",
    )
    assert executed["last_noahmp_census"]["glacier_path"] == (
        runner.ARWEN_GLACIER_CUDA_PROVENANCE
    )
    bad = dict(runner.EXPECTED_NOAHMP_CENSUS)
    bad["glacier_path"] = "noahmp-glacier/host"
    with pytest.raises(ValueError, match="census/provenance"):
        runner.require_arwen_v2_surface_execution(
            {"surface_classification": observed, "last_noahmp_census": bad},
            executed=True,
            label="mutant",
        )


def test_p_top_is_exact_f000_area_weighted_native_derivation() -> None:
    runner = _load_runner()
    expected = np.float32(1_159.38818359375)
    assert runner.EXPECTED_ARWEN_P_TOP_PA_F32.view(np.uint32) == expected.view(
        np.uint32
    )
    assert runner.EXPECTED_TOP_PRESSURE_RANGE_PA == (
        592.24884,
        1_233.00952,
        1_342.08362,
    )
    source = inspect.getsource(runner.derive_area_weighted_p_top_v841)
    for token in (
        "pressure_base",
        "pressure_perturbation",
        "zgrid",
        "area_cell",
        "np.float32",
        "weighted_mean64",
        "cavallo_buffer_layers",
        "claimed_native_identity",
    ):
        assert token in source
    assert "ARWEN_P_TOP_PA = 5_000" not in RUNNER_PATH.read_text(encoding="utf-8")


def test_exact_init_reconstruction_overlay_replaces_only_static_placeholder() -> None:
    runner = _load_runner()
    pin = runner.INIT_RECONSTRUCTION_COEFFICIENTS_PIN
    assert pin["dimensions"] == ("nCells", "maxEdges", "R3")
    assert pin["shape"] == (163_842, 10, 3)
    assert pin["static_placeholder_raw_sha256"] == (
        "b3b09d26538fe509096884906c93ff8b8d2c794300bafcab91449eee1c7bd31c"
    )
    assert pin["init_carrier_raw_sha256"] == (
        "1d25d2439a6cdcc3cc4a3cabfb5b6720730bb548f4cd88208340cca9df883350"
    )
    assert pin["active_slots"] == 983_040
    assert pin["active_components"] == 2_949_120
    assert pin["nonzero_components"] == 2_949_120

    overlay = inspect.getsource(
        runner.overlay_exact_init_reconstruction_coefficients
    )
    for token in (
        'prior_source != "static"',
        "init/static nEdgesOnCell topology differs",
        "init/static edgesOnCell topology differs",
        "active_every_component_nonzero",
        "bitwise_positive_zero",
        "source_files_mutated",
        "dynamics mesh in memory only",
    ):
        assert token in overlay
    preparation = inspect.getsource(runner._prepare_host_execution)
    overlay_call = preparation.index(
        "overlay_exact_init_reconstruction_coefficients"
    )
    prepared_seal = preparation.index("PreparedCudaInputs.validated")
    assert overlay_call < prepared_seal
    assert '"reconstruction_coefficients": reconstruction_overlay' in preparation


def test_exact_init_edge_normal_overlay_closes_all_geometry_carriers() -> None:
    runner = _load_runner()
    pin = runner.INIT_EDGE_NORMAL_VECTORS_PIN
    assert pin["dimensions"] == ("nEdges", "R3")
    assert pin["shape"] == (491_520, 3)
    assert pin["dtype"] == "<f4"
    assert pin["static_placeholder_raw_sha256"] == (
        "91e0c31eb2d6776a903dc5456c5e72e1e447c2d835cebbcde87581738cac735b"
    )
    assert pin["init_carrier_raw_sha256"] == (
        "25d9ef5c70b38a2e7d2c9c60456d9835a1e0fc790b60a3c98d1dbb44482d41da"
    )
    assert pin["nonzero_components"] == 1_474_550
    assert pin["exact_zero_components"] == 10
    assert pin["zero_rows"] == 0
    assert pin["float64_norm_min"] == 0.999999867111912
    assert pin["float64_norm_max"] == 1.000000135860109

    carriers = runner.PHYSICS_GEOMETRY_CARRIER_PIN
    assert carriers["cellsOnEdge"]["raw_sha256"] == (
        "a53b1c9bf9e5c0e026c7253b9026f4711bc973acd376f70a6e996adfa584b2d3"
    )
    assert carriers["cellsOnEdge"]["source_roles"] == ("grid", "static", "init")
    assert carriers["east_north"]["absent_source_roles"] == (
        "grid",
        "static",
        "init",
    )
    assert carriers["east_north"]["fallback_source_fields"] == (
        "lonCell",
        "latCell",
    )
    assert carriers["edgeNormalVectors"] == {
        "grid_present": False,
        "static_present": True,
        "init_present": True,
    }

    overlay = inspect.getsource(runner.overlay_exact_init_edge_normal_vectors)
    for token in (
        'prior_source != "static"',
        "grid/{role} raw cellsOnEdge topology differs",
        "absent_in_grid_static_init_and_prepared_mesh",
        "zonal_meridional_vectors(lonCell, latCell)",
        "only_zero_placeholder_trap",
        "float64_norm_min",
        "source_files_mutated",
        "dynamics mesh in memory only",
    ):
        assert token in overlay
    preparation = inspect.getsource(runner._prepare_host_execution)
    reconstruction_call = preparation.index(
        "overlay_exact_init_reconstruction_coefficients"
    )
    edge_normal_call = preparation.index("overlay_exact_init_edge_normal_vectors")
    prepared_seal = preparation.index("PreparedCudaInputs.validated")
    assert reconstruction_call < edge_normal_call < prepared_seal
    assert '"edge_normal_vectors": edge_normal_overlay' in preparation


def test_real_edge_normal_overlay_uses_sealed_init_bytes_without_source_mutation() -> None:
    from netCDF4 import Dataset

    runner = _load_runner()
    paths = runner.default_authority_paths()
    before = {
        role: (path.stat().st_size, path.stat().st_mtime_ns)
        for role, path in paths.items()
        if role in {"grid", "static", "init"}
    }
    with Dataset(paths["static"], "r") as dataset:
        normal_variable = dataset.variables["edgeNormalVectors"]
        normal_variable.set_auto_maskandscale(False)
        static_normals = np.ascontiguousarray(normal_variable[...])
        cells_variable = dataset.variables["cellsOnEdge"]
        cells_variable.set_auto_maskandscale(False)
        raw_cells = np.ascontiguousarray(cells_variable[...])
    with Dataset(paths["grid"], "r") as dataset:
        lon_variable = dataset.variables["lonCell"]
        lat_variable = dataset.variables["latCell"]
        lon_variable.set_auto_maskandscale(False)
        lat_variable.set_auto_maskandscale(False)
        lon = np.ascontiguousarray(lon_variable[...])
        lat = np.ascontiguousarray(lat_variable[...])

    mesh = SimpleNamespace(
        arrays={
            "edgeNormalVectors": static_normals,
            "cellsOnEdge": np.ascontiguousarray(raw_cells.astype(np.int64) - 1),
            "lonCell": lon,
            "latCell": lat,
        },
        variable_sources={
            "edgeNormalVectors": "static",
            "cellsOnEdge": "static",
            "lonCell": "grid_binary64_test_carrier",
            "latCell": "grid_binary64",
        },
        variable_dimensions={},
        variable_attrs={},
        provenance={},
    )
    receipt = runner.overlay_exact_init_edge_normal_vectors(
        mesh,
        grid_path=paths["grid"],
        static_path=paths["static"],
        init_path=paths["init"],
    )
    assert receipt["static_placeholder"]["bitwise_positive_zero"] is True
    assert receipt["init_carrier"]["raw_c_sha256"] == (
        runner.INIT_EDGE_NORMAL_VECTORS_PIN["init_carrier_raw_sha256"]
    )
    assert receipt["init_carrier"]["nonzero_components"] == 1_474_550
    assert receipt["init_carrier"]["exact_positive_zero_components"] == 10
    assert receipt["init_carrier"]["zero_rows"] == 0
    audit = receipt["cuda_physics_geometry_carrier_audit"]
    assert audit["cellsOnEdge"]["grid_static_init_raw_bit_identical"] is True
    assert audit["east_north"][
        "absent_in_grid_static_init_and_prepared_mesh"
    ] is True
    assert audit["edgeNormalVectors"]["only_zero_placeholder_trap"] is True
    assert mesh.variable_sources["edgeNormalVectors"] == (
        "init_exact_in_memory_physics_edge_normal_overlay"
    )
    assert np.count_nonzero(mesh.arrays["edgeNormalVectors"]) == 1_474_550
    after = {
        role: (path.stat().st_size, path.stat().st_mtime_ns)
        for role, path in paths.items()
        if role in {"grid", "static", "init"}
    }
    assert after == before

def test_arwen_v2_git_and_source_pin_precede_cuda_backend_and_kernel_cache() -> None:
    runner = _load_runner()
    main_source = inspect.getsource(runner.main)
    # The source pins verify FIRST: the checkout guard imports the seam
    # manifest from a pinned module, so the module's bytes must be proven
    # before any of its constants are trusted.
    assert main_source.index("require_frozen_execution_sources") < main_source.index(
        "verify_arwen_checkout_git"
    )
    source = inspect.getsource(runner._execute_full_proof)
    pin_import = source.index("pin_arwen_physics_v841")
    pin_call = source.index("arwen_pin = dict(pin_arwen_physics_v841")
    backend_import = source.index("from woof.hex.cuda_backend import")
    cache = source.index("cache = KernelCache(")
    assert pin_import < pin_call < backend_import < cache


class _FakeBackend:
    def __init__(
        self,
        log: list[str],
        *,
        commit_fails: bool = False,
        finish_auto_rolls_back: bool = False,
        rollback_boundary_valid: bool = True,
    ) -> None:
        self.log = log
        self.phase = "boundary"
        self.commit_fails = commit_fails
        self.finish_auto_rolls_back = finish_auto_rolls_back
        self.rollback_boundary_valid = rollback_boundary_valid
        self._at_boundary = True
        self._start_time = 0.0

    def begin_step(self, **kwargs: object) -> object:
        self.log.append("begin")
        self._start_time = float(kwargs["atmosphere"].state.time_seconds)
        self.phase = "begun"
        self._at_boundary = False
        return object()

    def finish_step(self, **_: object) -> object:
        self.log.append("phase2")
        if self.finish_auto_rolls_back:
            self.phase = "automatic_rollback"
            self._at_boundary = self.rollback_boundary_valid
            raise FloatingPointError("phase-two numeric refusal")
        self.phase = "finished_unpublished"
        return object()

    def commit_step(self) -> None:
        self.log.append("adapter_commit")
        if self.commit_fails:
            raise RuntimeError("adapter commit failure")
        self.phase = "complete"
        self._at_boundary = True

    def abort_step(self) -> None:
        if self.phase not in ("begun", "finished_unpublished"):
            raise RuntimeError("abort requires an active transaction")
        self.log.append("adapter_abort")
        self.phase = "rolled_back"
        self._at_boundary = True

    def step_receipt(self) -> dict[str, object]:
        return {
            "schema": "fake-transaction/v1",
            "adapter_contract_sha256": "fake-adapter",
            "constructor": {"identity_sha256": "fake-constructor"},
            "phase": self.phase,
            "start_time_seconds": self._start_time,
        }

    def restart_state(self) -> dict[str, object]:
        if not self._at_boundary:
            raise RuntimeError("restart state is not at a boundary")
        return {
            "schema": "fake-transaction/v1",
            "identity": {
                "adapter_contract_sha256": "fake-adapter",
                "constructor_identity_sha256": "fake-constructor",
            },
            "seam": {
                "identity": {"fake": True},
                "arrays": {},
                "scalars": {"elapsed_seconds": self._start_time},
            },
            "adapter": {},
        }


class _FakeDriver:
    def __init__(self, log: list[str], *, rollback_changes_state: bool = False) -> None:
        self.log = log
        self.rollback_changes_state = rollback_changes_state
        start = SimpleNamespace(
            time_seconds=0.0,
            rho=np.ones((1, 1), dtype=np.float32),
            rho_u=np.ones((1, 1), dtype=np.float32),
            scalars=np.zeros((6, 1, 1), dtype=np.float32),
        )
        self.atmosphere = SimpleNamespace(state=start)
        self.horizontal = SimpleNamespace(
            recover_edge_fields=lambda *_: SimpleNamespace(
                rho_edge=np.ones((1, 1), dtype=np.float32)
            )
        )

    def step_device_with_physics(self, _: object) -> object:
        self.log.append("moist_rk")
        endpoint = SimpleNamespace(
            time_seconds=120.0,
            scalars=np.zeros((6, 1, 1), dtype=np.float32),
        )
        return SimpleNamespace(atmosphere=SimpleNamespace(state=endpoint))

    def commit_post_wsm6_candidate(self, candidate: object, recovery: object) -> object:
        self.log.append("driver_commit")
        return SimpleNamespace(
            atmosphere=candidate.atmosphere,
            recovery=recovery,
            surface_updates={},
        )

    def abort_post_wsm6_candidate(self, _: object) -> None:
        self.log.append("driver_abort")
        if self.rollback_changes_state:
            self.atmosphere = SimpleNamespace(
                state=SimpleNamespace(time_seconds=0.0)
            )


def _transaction_callables(log: list[str], *, recovery_fails: bool = False):
    def couple(*_: object, **__: object) -> object:
        log.append("couple")
        return object()

    def clamp(*_: object, **__: object) -> object:
        log.append("clamp")
        return object()

    def recover(*_: object, **kwargs: object) -> object:
        log.append("recover")
        assert kwargs["phase2_dt_seconds"] == 120.0
        if recovery_fails:
            raise RuntimeError("recovery failure")
        return object()

    return couple, clamp, recover


def test_composite_transaction_order_is_atomic_and_staged(fake_phase_release) -> None:
    runner = _load_runner()
    log: list[str] = []
    couple, clamp, recover = _transaction_callables(log)
    result = runner.execute_composite_step(
        driver=_FakeDriver(log),
        backend=_FakeBackend(log),
        scalar_names=runner.SCALAR_NAMES,
        physics_geometry=object(),
        kernel_cache=object(),
        previous_surface_updates=None,
        couple=couple,
        clamp=clamp,
        recover=recover,
    )
    assert log == [
        "begin",
        "couple",
        "moist_rk",
        "clamp",
        "phase2",
        "recover",
        "driver_commit",
        "adapter_commit",
    ]
    assert result.committed.atmosphere.state.time_seconds == 120.0
    assert fake_phase_release == [
        ("enter", "arwen:phase1"), ("exit", "arwen:phase1"),
        ("enter", "arwen:phase2"), ("exit", "arwen:phase2"),
    ]


def test_composite_transaction_aborts_both_owners_before_publication() -> None:
    runner = _load_runner()
    log: list[str] = []
    couple, clamp, recover = _transaction_callables(log, recovery_fails=True)
    with pytest.raises(runner.CompositeTransactionError, match="aborted without publication"):
        runner.execute_composite_step(
            driver=_FakeDriver(log),
            backend=_FakeBackend(log),
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=couple,
            clamp=clamp,
            recover=recover,
        )
    assert log[-2:] == ["driver_abort", "adapter_abort"]
    assert "driver_commit" not in log


@pytest.mark.parametrize("recovery_fails", [False, True])
@pytest.mark.parametrize("parked", [False, True])
def test_composite_releases_consumed_carriers_before_next_phase(
    recovery_fails: bool, parked: bool,
) -> None:
    runner = _load_runner()
    log: list[str] = []
    refs = {}

    class Carrier:
        pass

    def edges(*_):
        result = Carrier()
        result.rho_edge = np.ones((1, 1), dtype=np.float32)
        result.normal_velocity = np.ones((1, 1), dtype=np.float32)
        refs["edges"] = weakref.ref(result)
        refs["rho_edge"] = weakref.ref(result.rho_edge)
        refs["normal_velocity"] = weakref.ref(result.normal_velocity)
        return result

    class Backend(_FakeBackend):
        def begin_step(self, **kwargs):
            super().begin_step(**kwargs)
            raw = Carrier()
            refs["raw"] = weakref.ref(raw)
            return raw

        def finish_step(self, **kwargs):
            assert refs["held"]() is None
            super().finish_step(**kwargs)
            update = Carrier()
            update.theta = np.ones((1, 1), dtype=np.float32)
            update.surface = {"rainnc": np.zeros((1,), dtype=np.float32)}
            update.radii = {"effc": np.zeros((1, 1), dtype=np.float32)}
            refs["update"] = weakref.ref(update)
            refs["update_theta"] = weakref.ref(update.theta)
            refs["surface"] = weakref.ref(update.surface["rainnc"])
            refs["radius"] = weakref.ref(update.radii["effc"])
            return update

    class Driver(_FakeDriver):
        def step_device_with_physics(self, held):
            assert all(refs[name]() is None for name in (
                "edges", "rho_edge", "normal_velocity", "raw"))
            assert refs["held"]() is held
            return super().step_device_with_physics(held)

        def commit_post_wsm6_candidate(self, candidate, recovery):
            assert refs["update"]() is None
            assert refs["update_theta"]() is None
            assert recovery.surface_updates["rainnc"] is refs["surface"]()
            assert recovery.effective_radii["effc"] is refs["radius"]()
            return super().commit_post_wsm6_candidate(candidate, recovery)

    class Park:
        is_parked = False

        def park(self, *, shared_roots):
            assert shared_roots[1] is refs["raw"]()
            assert shared_roots[2] is refs["held"]()
            assert shared_roots[3] is refs["edges"]()
            self.is_parked = True

        def unpark(self):
            self.is_parked = False

    def couple(raw, **kwargs):
        assert raw is refs["raw"]()
        assert kwargs["rho_edge"] is refs["rho_edge"]()
        held = Carrier()
        refs["held"] = weakref.ref(held)
        log.append("couple")
        return held

    def forcing(held, *_, **kwargs):
        assert held is refs["held"]()
        assert kwargs["normal_velocity"] is refs["normal_velocity"]()
        assert kwargs["rho_edge"] is refs["rho_edge"]()

    driver = Driver(log)
    driver.horizontal.recover_edge_fields = edges
    _, clamp, recover_original = _transaction_callables(
        log, recovery_fails=recovery_fails)

    def recover(state, update, **kwargs):
        assert refs["update"]() is update
        assert refs["update_theta"]() is update.theta
        recover_original(state, update, **kwargs)
        return SimpleNamespace(
            state=state, surface_updates=dict(update.surface),
            effective_radii=dict(update.radii))

    kwargs = dict(
        driver=driver, backend=Backend(log), scalar_names=runner.SCALAR_NAMES,
        physics_geometry=object(), kernel_cache=object(),
        previous_surface_updates=None, couple=couple, clamp=clamp,
        recover=recover, held_forcing=forcing,
        physics_park=Park() if parked else None,
    )
    if recovery_fails:
        with pytest.raises(runner.CompositeTransactionError,
                           match="aborted without publication"):
            runner.execute_composite_step(**kwargs)
        assert log[-2:] == ["driver_abort", "adapter_abort"]
    else:
        runner.execute_composite_step(**kwargs)
        assert log[-2:] == ["driver_commit", "adapter_commit"]


def test_physics_ownership_refresh_precedes_phase_two_and_is_atomic() -> None:
    runner = _load_runner()
    log = []
    couple, clamp, recover = _transaction_callables(log)
    def refresh(atmosphere):
        assert atmosphere.state.time_seconds == 120.0
        log.append("physics_owner_truth")
    runner.execute_composite_step(
        driver=_FakeDriver(log), backend=_FakeBackend(log),
        scalar_names=runner.SCALAR_NAMES, physics_geometry=object(),
        kernel_cache=object(), previous_surface_updates=None,
        couple=couple, clamp=clamp, recover=recover, before_phase_two=refresh)
    assert log.index("clamp") < log.index("physics_owner_truth") < log.index("phase2")
    assert log.index("phase2") < log.index("driver_commit")
    log.clear()
    def refuse(_):
        raise RuntimeError("physics halo ownership failed")
    with pytest.raises(runner.CompositeTransactionError):
        runner.execute_composite_step(
            driver=_FakeDriver(log), backend=_FakeBackend(log),
            scalar_names=runner.SCALAR_NAMES, physics_geometry=object(),
            kernel_cache=object(), previous_surface_updates=None,
            couple=couple, clamp=clamp, recover=recover, before_phase_two=refuse)
    assert log[-2:] == ["driver_abort", "adapter_abort"]
    assert "phase2" not in log and "driver_commit" not in log


def test_composite_transaction_accepts_verified_finish_automatic_rollback() -> None:
    runner = _load_runner()
    log: list[str] = []
    backend = _FakeBackend(log, finish_auto_rolls_back=True)
    couple, clamp, recover = _transaction_callables(log)
    with pytest.raises(runner.CompositeTransactionError, match="aborted without publication"):
        runner.execute_composite_step(
            driver=_FakeDriver(log),
            backend=backend,
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=couple,
            clamp=clamp,
            recover=recover,
        )
    assert backend.phase == "automatic_rollback"
    assert log == ["begin", "couple", "moist_rk", "clamp", "phase2", "driver_abort"]
    assert "adapter_abort" not in log


def test_composite_transaction_rejects_unverified_automatic_rollback() -> None:
    runner = _load_runner()
    log: list[str] = []
    couple, clamp, recover = _transaction_callables(log)
    with pytest.raises(
        runner.CompositeTransactionError,
        match="rollback was incomplete.*restart state is not at a boundary",
    ):
        runner.execute_composite_step(
            driver=_FakeDriver(log),
            backend=_FakeBackend(
                log,
                finish_auto_rolls_back=True,
                rollback_boundary_valid=False,
            ),
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=couple,
            clamp=clamp,
            recover=recover,
        )
    assert "adapter_abort" not in log


def test_composite_transaction_rejects_incomplete_driver_rollback() -> None:
    runner = _load_runner()
    log: list[str] = []
    couple, clamp, recover = _transaction_callables(log, recovery_fails=True)
    with pytest.raises(
        runner.CompositeTransactionError,
        match="rollback was incomplete.*exact start-state identity",
    ):
        runner.execute_composite_step(
            driver=_FakeDriver(log, rollback_changes_state=True),
            backend=_FakeBackend(log),
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=couple,
            clamp=clamp,
            recover=recover,
        )
    assert log[-2:] == ["driver_abort", "adapter_abort"]


def test_native_gate_includes_physical_edge_u_interface_w_and_gwdo() -> None:
    runner = _load_runner()
    assert runner.NATIVE_FIELD_MAP["normal_u"] == ("u", "level_edge")
    assert runner.NATIVE_FIELD_MAP["w"] == ("w", "interface_cell")
    for name in (
        *runner.SCALAR_NAMES,
        "rho",
        "theta",
        "pressure",
        "surface_pressure",
        "normal_u",
        "w",
        "smois",
        "tslb",
        "rainc",
        "rainnc",
        "dusfcg",
        "dvsfcg",
        "dtaux3d",
        "dtauy3d",
        "rubldiff",
        "rvbldiff",
    ):
        assert name in runner.NATIVE_RMSE_LIMITS


def test_diagnostic_dataclass_and_restart_hash_projection_are_exact() -> None:
    runner = _load_runner()
    groups = {
        name: {"x": np.array([1.0], dtype=np.float32)}
        for name in ("surface", "soil", "precipitation", "gwdo")
    }
    snapshot = SimpleNamespace(
        **groups,
        metadata={"time_seconds": 0.0},
        receipt={"phase": "boundary"},
    )
    mapped = runner.backend_diagnostic_mapping(snapshot)
    assert set(mapped) == {
        "surface",
        "soil",
        "precipitation",
        "gwdo",
        "metadata",
        "receipt",
    }
    direct = {"arrays": {"qv": np.array([1.0], dtype=np.float32), "q2": np.array([-1.0], dtype=np.float32)}}
    resumed = {"arrays": {"qv": np.array([1.0], dtype=np.float32), "q2": np.array([9.0], dtype=np.float32)}}
    identity = runner.require_bitwise_restart_identity(direct, resumed)
    assert identity["bitwise_identical"] is True
    assert identity["q2_excluded_from_publication_projection"] is True


def test_restart_fingerprint_identity_names_exact_numerical_leaf() -> None:
    runner = _load_runner()
    direct = runner.fingerprint_nested_arrays(
        {
            "state": {"rho": np.array([1.0, 2.0], dtype=np.float32)},
            "step": 16,
        }
    )
    identical = runner.fingerprint_nested_arrays(
        {
            "state": {"rho": np.array([1.0, 2.0], dtype=np.float32)},
            "step": 16,
        }
    )
    receipt = runner.require_fingerprint_identity(
        "step 16 atmosphere", direct, identical
    )
    assert receipt["bitwise_identical"] is True
    assert receipt["sha256"] == direct["sha256"]

    changed = runner.fingerprint_nested_arrays(
        {
            "state": {"rho": np.array([1.0, 3.0], dtype=np.float32)},
            "step": 16,
        }
    )
    with pytest.raises(
        RuntimeError,
        match=r"step 16 atmosphere differs.*arrays/state/rho/sha256",
    ):
        runner.require_fingerprint_identity("step 16 atmosphere", direct, changed)


def test_restart_localization_gates_precede_first_resumed_step() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner._execute_full_proof)
    worker_spawn = source.index("worker = _spawn_restart_worker(")
    restored_f030 = source.index('restored_f030 = worker["restored_f030"]')
    step16 = source.index('"first resumed step 16 MPAS atmosphere"')
    f001 = source.index("identity = require_bitwise_restart_identity(")
    assert worker_spawn < restored_f030 < step16 < f001
    assert '"F030 restored MPAS atmosphere"' in source
    assert '"F030 restored WOOF backend"' in source
    assert '"F030 restored GF advective forcing"' in source
    assert '"first resumed step 16 WOOF backend"' in source
    assert '"restart_arm_fresh_process": True' in source

    worker_source = inspect.getsource(runner._execute_restart_worker)
    construct = worker_source.index("stack = _construct_device_stack(")
    rehydrate = worker_source.index(
        "restored_f030 = fingerprint_execution_boundary(stack)"
    )
    identity = worker_source.index(
        '"F030 restored MPAS atmosphere (fresh restart process)"'
    )
    forcing_identity = worker_source.index(
        '"F030 restored GF advective forcing (fresh restart process)"'
    )
    steps = worker_source.index(
        "restart_snapshots, _, restart_receipts = _run_steps("
    )
    assert construct < rehydrate < identity < forcing_identity < steps
    assert "boundary_observer=capture_step16" in worker_source

    loop_source = inspect.getsource(runner._run_steps)
    receipt = loop_source.index("receipts.append(")
    observer = loop_source.index("boundary_observer(step, stack)")
    capture = loop_source.index("if step in capture_steps:", receipt)
    assert receipt < observer < capture


def test_checkpoint_carries_and_restores_the_gf_advective_forcing() -> None:
    """The step-16 restart mechanism, pinned: GF's rthdynten/rqvdynten pair is per-step
    carried state formed by each step's dynamics and consumed by the NEXT
    begin_step -- outside both the MPAS atmosphere and the Arwen backend
    restart payload.  The v2 checkpoint never carried it, so every restored
    arm re-entered step 16 on zero forcing lanes while the unbroken run fed
    the real step-15 pair: a deterministic all-fields divergence at exactly
    step 16, on every arm.  Schema v3 captures the pair at F030, refuses a
    checkpoint written without it, and re-seeds it on restore under its own
    fingerprint identity."""

    runner = _load_runner()
    assert runner.CHECKPOINT_SCHEMA.endswith("-checkpoint/v3")
    fields = {field.name for field in dataclasses.fields(runner.HostDriverCheckpoint)}
    assert {"gf_dynamics_tendencies", "gf_forcing_fingerprint"} <= fields

    # Writing a checkpoint without the carrier refuses by name, before any
    # device access.
    with pytest.raises(RuntimeError, match="GF advective-forcing"):
        runner.download_driver_checkpoint(
            object(), object(), dynamics_tendencies=None
        )

    # The step loop seeds the carrier from the stack, so a restored stack
    # that carries it continues exactly where the unbroken run is.
    loop_source = inspect.getsource(runner._run_steps)
    assert 'stack.get("gf_dynamics_tendencies")' in loop_source
    proof_source = inspect.getsource(runner._execute_full_proof)
    assert 'dynamics_tendencies=stack.get("gf_dynamics_tendencies")' in proof_source

    # The restore path uploads the pair back onto the stack.
    construct_source = inspect.getsource(runner._construct_device_stack)
    assert "CudaV841GfDynamicsTendencies(" in construct_source
    assert 'stack["gf_dynamics_tendencies"] = carrier' in construct_source

    # The worker refuses a pre-v3 checkpoint by name and returns the
    # restored-forcing fingerprint the parent gates on.
    worker_source = inspect.getsource(runner._execute_restart_worker)
    assert "predates the GF advective-forcing capture" in worker_source
    assert "gf_dynamics_tendencies=forcing" in worker_source
    spawn_source = inspect.getsource(runner._spawn_restart_worker)
    assert '"restored_gf_forcing"' in spawn_source

    # A stale pickle (pre-v3 HostDriverCheckpoint with the slot unset) at the
    # correct F030 time is refused by the worker before any CUDA probing.
    import pickle

    stale = object.__new__(runner.HostDriverCheckpoint)
    for name, value in {
        "state": None,
        "saved_diagnostics": None,
        "backend_state": {},
        "atmosphere_fingerprint": {},
        "backend_fingerprint": {},
        "model_time_seconds": runner.CHECKPOINT_STEP * runner.DT_SECONDS,
        # A v2 pickle leaves these slots unset; the closest picklable
        # simulation is the carrier explicitly absent.
        "gf_dynamics_tendencies": None,
        "gf_forcing_fingerprint": None,
    }.items():
        object.__setattr__(stale, name, value)
    job = {
        "schema": runner.RESTART_WORKER_SCHEMA,
        "arwen_commit": runner.ARWEN_COMMIT,
        "checkpoint": stale,
    }
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as scratch:
        stale_path = _Path(scratch) / "stale.pkl"
        with stale_path.open("wb") as stream:
            pickle.dump(job, stream, protocol=4)
        with pytest.raises(
            RuntimeError, match="predates the GF advective-forcing capture"
        ):
            runner._execute_restart_worker(
                stale_path, _Path(scratch) / "never.pkl"
            )


def test_restart_arm_is_a_genuinely_fresh_worker_process(tmp_path: Path) -> None:
    """The measured defect: a seam aggregate reconstructed inside the live
    baseline process reproduces frozen legacy-RRTMG longwave heating only to
    within 1 ULP (raw phase-1 dtheta, carrier rthratenlw), while the same
    restore in a fresh process continues bit-identically.  The restart arm
    must therefore execute in a fresh worker process."""

    import pickle

    runner = _load_runner()
    # Worker CLI flags travel only as a pair.
    with pytest.raises(SystemExit):
        runner.parse_args(["--restart-worker-input", str(tmp_path / "in.pkl")])
    with pytest.raises(SystemExit):
        runner.parse_args(["--restart-worker-output", str(tmp_path / "out.pkl")])
    args = runner.parse_args(
        [
            "--restart-worker-input", str(tmp_path / "in.pkl"),
            "--restart-worker-output", str(tmp_path / "out.pkl"),
        ]
    )
    assert args.restart_worker_input == tmp_path / "in.pkl"
    assert args.restart_worker_output == tmp_path / "out.pkl"
    # The spawn launches a separate interpreter on this exact tool file.
    spawn_source = inspect.getsource(runner._spawn_restart_worker)
    assert "sys.executable" in spawn_source
    assert "subprocess.run" in spawn_source
    assert '"--restart-worker-input"' in spawn_source
    assert '"--restart-worker-output"' in spawn_source
    # The spawn refuses non-fresh or foreign worker payloads.
    assert 'results.get("fresh_process") is not True' in spawn_source
    assert 'results.get("arwen_commit") != ARWEN_COMMIT' in spawn_source
    # The worker refuses foreign payload schemas before any CUDA probing.
    bad = tmp_path / "bad.pkl"
    with bad.open("wb") as stream:
        pickle.dump({"schema": "not-the-restart-worker-schema"}, stream, protocol=4)
    with pytest.raises(RuntimeError, match="schema"):
        runner._execute_restart_worker(bad, tmp_path / "never.pkl")
    # The worker refuses a non-F030 checkpoint before any CUDA probing.
    wrong_time = SimpleNamespace(model_time_seconds=16 * 120.0)
    job = {
        "schema": runner.RESTART_WORKER_SCHEMA,
        "arwen_commit": runner.ARWEN_COMMIT,
        "checkpoint": wrong_time,
    }
    wrong = tmp_path / "wrong-time.pkl"
    with wrong.open("wb") as stream:
        pickle.dump(job, stream, protocol=4)
    with pytest.raises(TypeError, match="HostDriverCheckpoint"):
        runner._execute_restart_worker(wrong, tmp_path / "never2.pkl")


def test_gf_deviation_is_explicit_and_never_claims_native_parity() -> None:
    runner = _load_runner()
    assert any("GF" in item and "zero" in item for item in runner.NONCLAIMS)
    source = inspect.getsource(runner._execute_full_proof)
    assert '"fa35_rthften_rqvften": "zero"' in source
    assert '"native_gf_parity_claim": False' in source
    assert '"f001_restart_snapshot_receipt"' in source


def test_baseline_diagnostic_cli_is_isolated_and_precedes_restart(tmp_path: Path) -> None:
    runner = _load_runner()
    cache = tmp_path / "cache"
    output = tmp_path / "release-output"
    diagnostic = tmp_path / "baseline-output"
    args = runner.parse_args(
        [
            "--cache-root", str(cache),
            "--output", str(output),
            "--baseline-diagnostic-output", str(diagnostic),
        ]
    )
    assert args.baseline_diagnostic_output == diagnostic
    admitted_cache, admitted_output = runner.validate_destination(cache, output, ())
    admitted_diagnostic = runner.validate_baseline_diagnostic_destination(
        diagnostic,
        cache_root=admitted_cache,
        output_root=admitted_output,
        protected=(),
    )
    assert admitted_diagnostic == diagnostic.absolute()
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_baseline_diagnostic_destination(
            output / "nested",
            cache_root=admitted_cache,
            output_root=admitted_output,
            protected=(),
        )
    source = inspect.getsource(runner._execute_full_proof)
    diagnostic_write = source.index("_write_baseline_diagnostic_output(")
    release_uninterrupted = source.index("# Release the uninterrupted device graph")
    restart_worker = source.index("worker = _spawn_restart_worker(")
    assert diagnostic_write < release_uninterrupted < restart_worker
    assert runner.BASELINE_DIAGNOSTIC_WARNING == (
        "ENGINEERING BASELINE; restart proof pending; NOT RELEASE"
    )
    assert runner.BASELINE_DIAGNOSTIC_STATUS == (
        "uninterrupted_baseline_passed_restart_not_evaluated_non_release"
    )


def test_f000_optional_surface_diagnostics_are_exact_init_snapshot_only() -> None:
    from netCDF4 import Dataset

    runner = _load_runner()
    paths = runner.default_authority_paths()
    carrier = runner.load_f000_initialized_surface_diagnostics(paths["init"])
    assert set(carrier) == {"arrays", "receipt"}
    assert set(carrier["arrays"]) == {"t2", "u10", "v10"}
    with Dataset(paths["native_f000"], "r") as native:
        for target, pin in runner.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items():
            source = carrier["arrays"][target]
            authority = np.ascontiguousarray(
                np.asarray(native.variables[str(pin["source"])][0]), dtype=np.float32
            )
            np.testing.assert_array_equal(source, authority)
            assert runner.array_sha256(source) == pin["sha256"]

    placeholders = {
        name: np.zeros(runner.N_CELLS, dtype=np.float32)
        for name in carrier["arrays"]
    }
    receipt = runner.overlay_f000_initialized_surface_diagnostics(
        placeholders, carrier["arrays"]
    )
    assert receipt["applied"] is True
    assert "snapshot only" in receipt["scope"]
    for name, source in carrier["arrays"].items():
        np.testing.assert_array_equal(placeholders[name], source)
        assert receipt["fields"][name]["source_correct"] is True

    nonzero = {
        name: np.zeros(runner.N_CELLS, dtype=np.float32)
        for name in carrier["arrays"]
    }
    nonzero["t2"][0] = np.float32(1.0)
    with pytest.raises(ValueError, match=r"not the exact \+0 FP32 placeholder"):
        runner.overlay_f000_initialized_surface_diagnostics(
            nonzero, carrier["arrays"]
        )

    capture_source = inspect.getsource(runner.capture_snapshot)
    assert capture_source.index("arrays.update(diagnostics)") < capture_source.index(
        "overlay_f000_initialized_surface_diagnostics("
    )
    assert "if step == 0:" in capture_source


def test_baseline_writer_validates_nested_exact_capture_receipt_schema() -> None:
    runner = _load_runner()
    for label, step in (
        ("F000", 0),
        ("F030", runner.CHECKPOINT_STEP),
        ("F001", runner.FULL_STEPS),
    ):
        nested = {
            "surface_classification": dict(
                runner.EXPECTED_SURFACE_CLASSIFICATION
            ),
            "last_noahmp_census": (
                None if step == 0 else dict(runner.EXPECTED_NOAHMP_CENSUS)
            ),
        }
        capture_receipt = {
            "schema": runner.SNAPSHOT_SCHEMA,
            "label": label,
            "step": step,
            "time_seconds": step * runner.DT_SECONDS,
            "arwen_v2_surface_execution": nested,
            "audit_d2h_outside_step_receipt": True,
            "prep": {},
            "backend": {},
            "backend_diagnostic_metadata": {},
            "f000_initialized_surface_diagnostics": (
                {} if step == 0 else None
            ),
            "arrays": {},
        }
        assert set(capture_receipt) == {
            "schema", "label", "step", "time_seconds",
            "arwen_v2_surface_execution", "audit_d2h_outside_step_receipt",
            "prep", "backend", "backend_diagnostic_metadata",
            "f000_initialized_surface_diagnostics", "arrays",
        }
        observed = runner.require_snapshot_receipt_surface_execution(
            capture_receipt, label=label
        )
        assert observed == nested

        flattened = dict(capture_receipt)
        flattened.pop("arwen_v2_surface_execution")
        flattened.update(nested)
        with pytest.raises(ValueError, match="lacks nested"):
            runner.require_snapshot_receipt_surface_execution(
                flattened, label=label
            )

    writer = inspect.getsource(runner._write_baseline_diagnostic_output)
    assert "require_snapshot_receipt_surface_execution(" in writer
    assert "snapshot_receipts[label], label=label" in writer
