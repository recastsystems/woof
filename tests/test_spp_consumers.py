"""Nonzero SPP against independent, unmodified WRF v4.6.1 modules.

The oracle builder supplies patterns at Fortran interfaces. It never edits
the reference physics. GF uses CUP_gf because stock GFDRV drops rand_clos.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from woof.core.fp32_ulp import fp32_ulp_distance
import test_mynn_pbl as pbl
import test_mynn_surface_water as surface

DATA = Path(__file__).resolve().parents[1] / "woof" / "data" / "spp"
# The native captures ship in the recast-woof-data companion since 2.8.7
# (woof.data_assets.COMPANION_TREES); the ULP receipts stay in DATA.
NATIVE_ZIP = "spp/native-wrf461.zip"
# Rebuilt by development branch on the program's oracle platform
# (gfortran 13.3.0, glibc 2.39, the toolchain of every other WRF column
# oracle) with tools/spp_wrf461_oracle/build.py and package.py.  The previous
# archive (9bc14a88...) was built with gfortran 15.2.0 on glibc 2.43; its
# MYNN PBL and GF captures differ from this platform's in the float32 libm
# words, which the per-element budgets below used to absorb.  2.9.1: the GF
# captures were re-run (receipt.json gf_rebuild, same platform) on the GF
# harness after d612ddee205 corrected its GFS constants; built before it,
# their stock capture differed from GFDRV on 14 of 216 columns by 1-4 ULP.
NATIVE_ZIP_SHA256 = "201025994da48275c72fb06d37e38074c924780139a1aed8ed4409682b3a688e"


@pytest.fixture(scope="session")
def native_directory(tmp_path_factory):
    directory = os.environ.get("WOOF_SPP_ORACLE")
    if directory:
        return Path(directory)
    from woof.data_assets import data_path
    archive = data_path(NATIVE_ZIP)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == NATIVE_ZIP_SHA256
    target = tmp_path_factory.mktemp("spp-native")
    with zipfile.ZipFile(archive) as source:
        assert all(Path(name).name == name for name in source.namelist())
        source.extractall(target)
    receipt = json.loads((target / "receipt.json").read_text())
    for path in target.glob("*.csv"):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["files"][path.name]["sha256"]
    return target


@pytest.fixture
def oracle(monkeypatch, native_directory):
    root = native_directory
    for attr, filename in [("TURBULENCE_ORACLE", "turbulence-spp.csv"),
                           ("CONDENSATION_ORACLE", "condensation-spp.csv"),
                           ("DMP_MF_ORACLE", "dmp_mf-spp.csv"),
                           ("DRIVER_ORACLE", "driver-spp.csv")]:
        monkeypatch.setattr(pbl, attr, root / filename)
    return root


@pytest.fixture
def backend():
    from woof.core import mynn_pbl
    return "cpu", mynn_pbl, np.asarray, surface.mynn_surface_layer_default, np.asarray


def _call(backend, name, values, **kwargs):
    kind, module, host = backend[:3]
    result = getattr(module, name + ("_cuda" if kind == "gpu" else ""))(values, **kwargs)
    def get(key):
        return host(result[key] if isinstance(result, dict) else getattr(result, key))
    return get


def _measure(actual, expected):
    return {name: int(fp32_ulp_distance(actual[name], value).max(initial=0))
            for name, value in expected.items()}


def _gate(label, actual, expected):
    metrics = _measure(actual, expected)
    print("SPP_NATIVE " + json.dumps({"name": label, "ulp": {k:v for k,v in metrics.items() if v}, "fields": len(metrics)}, sort_keys=True))
    for name in expected:
        assert np.isfinite(actual[name]).all(), (label, name)
    # Every MYNN PBL label is 0 ULP against the capture rebuilt on the oracle
    # platform, CPU and GPU, default and strict arithmetic (measured on W1,
    # RTX PRO 6000, by development branch).  The budgets this gate
    # used to read for them measured the old capture's foreign libm words,
    # so a regression of up to 18 ULP in the SPP cloud diagnosis could pass.
    if not label.split(".", 1)[1].startswith("surface"):
        assert {name: value for name, value in metrics.items() if value} == {}, label
        return
    # The MYNN surface layer keeps one measured bound per element: on this
    # base its kernel is not yet WRF-exact.  development branch
    # makes it 0 ULP with spp_pbl=1 on the same platform; this budget
    # retires when that fix lands (woof/data/spp/*-ulp.json "why").
    with np.load(DATA / (label.split(".",1)[0] + "-ulp.npz"), allow_pickle=False) as budget:
        for name,value in expected.items():
            distance = fp32_ulp_distance(actual[name],value)
            bound = budget[label+"/"+name]
            assert distance.shape == bound.shape
            assert np.isfinite(actual[name]).all(), (label,name)
            assert np.all(distance <= bound), (label,name,metrics[name],int(bound.max(initial=0)))


def test_mynn_turbulence_spp_native(oracle, backend):
    _, fields = pbl._turbulence_oracle()
    inputs = {name: fields[name] for name in pbl.MYNN_TURBULENCE_INPUTS}
    for name in ("xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl", "psig_shcu"):
        inputs[name] = fields[name][:, 0]
    inputs["rstoch"] = fields["rstoch"]
    get = _call(backend, "mynn_turbulence_default", inputs, spp_pbl=1)
    actual = {name: get(name) for name in pbl.TURBULENCE_OUTPUTS}
    expected = {name: fields[name] for name in actual}
    _gate(backend[0] + ".turbulence", actual, expected)


def test_mynn_condensation_spp_native(oracle, backend):
    _, fields = pbl._condensation_oracle()
    get = _call(backend, "mynn_condensation_default", pbl._condensation_inputs(fields), spp_pbl=1)
    names = ("qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm")
    expected = {name: fields[name + ("_after" if name in ("vt", "vq", "sgm") else "")]
                for name in names}
    _gate(backend[0] + ".condensation", {name: get(name) for name in names}, expected)


def test_mynn_mass_flux_spp_native(oracle, backend):
    rows = pbl._dmp_mf_oracle()
    actual, expected = {}, {}
    for case in pbl.DMP_MF_CASES:
        selected = pbl._dmp_mf_case(rows, case)
        get = _call(backend, "mynn_dmp_mf", pbl._dmp_mf_inputs(selected), spp_pbl=1)
        for name, value in pbl._dmp_mf_expected(selected).items():
            got = get(name)
            key = case + "/" + name
            actual[key] = got[0] if got.ndim == 2 else got
            expected[key] = value
    _gate(backend[0] + ".mass_flux", actual, expected)


@pytest.mark.parametrize("step", [1, 2])
def test_mynn_driver_spp_native(oracle, backend, step):
    blocks, values, initflag, delt = pbl._driver_step(step)
    values["rstoch"] = np.broadcast_to(
        np.float32(.1) * (np.arange(5, dtype=np.float32)[:, None] - np.float32(2)),
        (5, pbl.DRIVER_NZ)).copy()
    get = _call(backend, "mynn_bl_driver", values, initflag=initflag, delt=delt,
                flag_qs=True, spp_pbl=1)
    expected = {name: np.asarray([[np.float32(row[pbl.DRIVER_OUTPUT_CSV.get(name, name)])
                                  for row in block] for block in blocks])
                for name in pbl.DRIVER_PROFILE_OUTPUTS}
    expected.update({name: np.asarray([np.float32(block[0][name]) for block in blocks])
                     for name in ("pblh", "rmol", "maxwidth", "maxmf", "ztop_plume")})
    for name in ("kpbl", "ktop_plume"):
        np.testing.assert_array_equal(get(name).reshape(-1), [int(block[0][name]) for block in blocks])
    actual = {name: get(name).reshape(value.shape) for name, value in expected.items()}
    _gate(f"{backend[0]}.driver.step{step}", actual, expected)


def test_mynn_surface_spp_native(oracle, backend):
    # The gpu.surface budget was re-measured by development branch
    # (gpu-ulp.json "gpu.surface_remeasured"; no other label moved).  This
    # capture was built with gfortran 15.2.0 on glibc 2.43; on the program's
    # oracle platform (gfortran 13.3.0, glibc 2.39) the CUDA kernel is 0 ULP
    # on every output with spp_pbl=1 as well as 0
    # (tests/test_mynn_sfclay_wrf461_column_oracle_gpu.py).  So the budget
    # here measures the two platforms' float32 libm words, chiefly psim (up
    # to 51 ULP), not a kernel residual.
    with (oracle / "surface-spp.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    actual, expected = {}, {}
    for flux in range(4):
        for step in (1, 2):
            selected = [r for r in rows if int(r["isftcflx"]) == flux and int(r["itimestep"]) == step]
            def field(name):
                return np.asarray([[np.float32(row[name]) for row in selected]])
            values = {name: field(surface.INPUT_ALIASES.get(name, name)) for name in surface.INPUT_NAMES}
            kwargs = dict(dx=float(selected[0]["dx"]), itimestep=step, isfflx=1, isftcflx=flux,
                          mol=field("mol_input"), ustm=field("ustm_input"), spp_pbl=1,
                          pattern_spp_pbl=field("rstoch"))
            kwargs["pattern_spp_pbl"] = backend[4](kwargs["pattern_spp_pbl"])
            if backend[0] == "cpu":
                values = {key: value.ravel() for key,value in values.items()}
                for key in ("mol", "ustm", "pattern_spp_pbl"):
                    kwargs[key] = kwargs[key].ravel()
            result = backend[3](values, **kwargs)
            for name in surface.OUTPUT_NAMES:
                key = f"{flux}.{step}.{name}"
                actual[key] = backend[2](result[name] if isinstance(result, dict) else getattr(result, name)).reshape(1,-1)
                expected[key] = field(name)
    _gate(backend[0] + ".surface", actual, expected)


def test_spp_specializations_leave_default_compiler_sources_frozen():
    from woof.core.kernels import module_source
    from woof.core.spp_kernel_sources import specialized_source
    pins = {
            # RE-PINNED for 2.9.0 (development branch 0e7c7f03d, exact chain
            # 7cdf42951): CUP_gf keeps descending after a valid
            # downdraft-origin adjustment, as WRF does; the same change
            # re-pinned gf in tests/test_kernel_source_freeze_per_module.py.
            # Previously 53785cdb.
            # RE-PINNED for 2.9.1 (the WRF exact chain): float32-widened GFS
            # constants, the PBL closure bound, the lower-face W and the FMA exp
            # form are default-on GF corrections. Previously ada52295.
            # RE-PINNED for 3.0.0, default compiled program unchanged:
            # 6d79b6e9c5d dropped gf.cu's GPUWM_WRF_EXACT-only normalizer
            # branch and reworded its header comment (the default #else arm
            # is the retained code), and 51bb93111ee reworded two comments in
            # glibc_flt32.cuh. Previously b5b0d38c.
            "gf": "704111403acd51eeae88e0715108bf8633cf3830f8a606d12cfec721bc81aa19",
            # Accepted staging adds source-selected GSD41 entry points.
            # This pins the actual default compiler string and checks that
            # constructing an SPP specialization leaves that string intact.
            # It does not claim unchanged default PTX from the older source.
            # P23 (development branch) adds MYNN_GSD41 arms only; the default
            # PTX is byte-identical (test_mp8_frozen's mynn_pbl row).
            # RE-PINNED by development branch (the YSU recipe on MYNN), MOVES ANSWERS
            # for bl_pbl_physics = 5: --fmad=false, glibc 2.39 expf/log10f and the
            # FMA-ifunc logf/powf/expf in the unit, fdlibm tanhf at every site;
            # under MYNN_GSD41 the fork's a2den algebra and 0.608 thetav.  Reading
            # (RTX 5090 sm_120, NVRTC 12.9): every MYNN CUDA leaf and both drivers 0
            # ULP against WRF v4.6.1, six column families x 12 steps free-running 0
            # ULP (tests/test_mynn_wrf461_exact_gpu.py); gsd_41 0 ULP against the
            # fork's driver (tests/test_mynn_gsd41_driver_exact_gpu.py). Previously
            # 185b36a7.
            # Review: the changed CASE(1) is excluded from the stock/SPP build.
            # Source-only synchronization to the selected R1 capture in
            # test_mp8_frozen.FROZEN_MODULE_DIGESTS. This does not establish new PTX or
            # numerical replay identity for the assembled MYNN source.
            # Slimming recorder follows the reviewed R1 export-arm fold.
            # RE-PINNED for 3.0.0, synchronized with test_mp8_frozen:
            # 6d79b6e9c5d adds MYNN_GSD41-only DMP cloud-base outputs and
            # GRIMS diagnostics, off in the default string. Previously
            # 1f5e6f64.
            "mynn_pbl": 'cf6678981492e29c5d58e8c269f6cf247d5b00118cffa47e97d71014cd471342',
            # Match test_mp8_frozen's accepted staging source: selectable
            # GSL WRF 3.9 stability solver plus explicitly rounded ordinary
            # mynn_table interpolation. The SPP construction remains inert
            # for the default compiler string; the table source has changed.
            # development branch (8cd59d74c15) adds the OCEAN_WAVE_ROUGHNESS
            # block, off in the default string; synchronized with
            # test_mp8_frozen.FROZEN_MODULE_DIGESTS.
            # RE-PINNED for 3.0.0: 51bb93111ee reworded one comment.
            # Previously 5b294157.
            "mynn_surface": "a9abd5020db8a204e3f538e9af69ebb903c2b329c56c258d5286bfdf18cc5ddf"}
    for name, pin in pins.items():
        baseline = module_source(name)
        assert hashlib.sha256(baseline.encode()).hexdigest() == pin
        assert specialized_source(name) != baseline
        assert module_source(name) == baseline


def test_enabled_spp_compiler_inputs_are_in_the_literal_division_census():
    from tools.literal_division_census import production_units
    from woof.core.spp_kernel_sources import specialized_source
    units = {unit.key: unit for unit in production_units(DATA.parents[2])}
    for name,capacity in (("gf",40),("gf",60),("mynn_pbl",40),("mynn_surface",40)):
        assert units[f"spp:{name}[capacity={capacity}]"].source == specialized_source(name,capacity=capacity)
    assert "kernels:ruc_spp" in units


def test_native_spp_controls_exercise_the_parameter_sites(oracle):
    def rows(name):
        with (oracle / name).open(newline="") as stream:
            return list(csv.DictReader(stream))
    def values(data, field):
        return np.asarray([row[field] for row in data], dtype=np.float32)
    for stem, field in (("turbulence", "dfm"), ("condensation", "cldfra"),
                        ("dmp_mf", "edmf_ent"), ("surface", "hfx")):
        on, off = rows(stem + "-spp.csv"), rows(stem + "-off.csv")
        assert np.any(values(on,field) != values(off,field)), (stem,field)
        if stem == "turbulence":
            np.testing.assert_array_equal(values(on,"dfq"), values(off,"dfq"))
        if stem == "surface":
            first = [k for k,row in enumerate(on) if int(row["itimestep"]) == 1]
            np.testing.assert_array_equal(values(on,"znt")[first], values(off,"znt")[first])


def test_disabled_spp_config_preserves_old_prepared_cache_identity():
    from woof.ingest.prepared_cache import compare_prepared_domain_config
    fields = {"spp_conv":0,"spp_pbl":0,"spp_lsm":0}
    defaults = {"run."+key:value for key,value in fields.items()}
    cached = {"run":{"dx":3000.0}}
    live = {"run":{"dx":3000.0,**fields}}
    tolerated,differing = compare_prepared_domain_config(cached,live,not_in_use=defaults)
    assert set(tolerated) == set(defaults)
    assert differing == []
    for key in fields:
        live["run"][key] = 1
        assert compare_prepared_domain_config(cached,live,not_in_use=defaults)[1] == ["run."+key]
        live["run"][key] = 0


def test_gpu_surface_bounds_never_exceed_the_cpu_authority():
    """Every gpu.surface element bound is at most the CPU port's bound.

    The breakage: the gpu.surface rows were re-recorded on 2.8.8 (the MYNN
    surface unit lost flush to zero and took WOOF's float32 libm), which
    raised 46 per-element bounds.  Each raised bound equals the CPU float32
    port's bound for the same element, so the card reproduces a residual the
    CPU port already carries against WRF.  A re-record that raised a bound
    past the CPU's would let the GPU acquire a residual the CPU port lacks,
    and the native gate above would accept it because it reads only
    gpu-ulp.npz.
    """
    receipt = json.loads((DATA / "gpu-ulp.json").read_text(encoding="utf-8"))
    assert "cpu-ulp" in receipt["metadata"]["rerecorded_rows"]["gpu.surface"]["bounds"]
    with np.load(DATA / "gpu-ulp.npz", allow_pickle=False) as gpu, \
            np.load(DATA / "cpu-ulp.npz", allow_pickle=False) as cpu:
        rows = [name for name in gpu.files if name.startswith("gpu.surface/")]
        assert len(rows) > 0
        for name in rows:
            twin = "cpu.surface/" + name.split("/", 1)[1]
            assert twin in cpu.files, name
            assert gpu[name].shape == cpu[twin].shape, name
            above = gpu[name] > cpu[twin]
            assert not above.any(), (name, gpu[name][above].tolist(),
                                     cpu[twin][above].tolist())
