"""A set changes only its named sites, and those sites affect live kernels.

The source checks enumerate every translation unit independently of the
editor. Device checks use the existing wide column fixtures, retaining
unchanged intermediates, water columns, inactive plumes and CASE(2) as
negative controls. A successfully compiled source alone is insufficient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof import physics_params as pp
from woof.core import kernels

KDIR = Path(kernels._KDIR)
VALUES = {
    "mynn.czil": 0.16,
    "mynn.prandtl": 0.9,
    "mynn.cns": 4.5,
    "mynn.alp1": 0.32,
    "mynn.edmf_entrainment": 0.48,
    "ruc.z0.tall": 1.5,
    "ruc.z0.short": 0.6,
    "ruc.rs": 1.5,
}


@pytest.fixture(autouse=True)
def _binding(monkeypatch):
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    pp.reset_for_tests()
    yield
    pp.reset_for_tests()


def _source(module):
    source = "mynn_pbl" if module == "mynn_dmp_sibling" else module
    return (KDIR / f"{source}.cu").read_text(encoding="utf-8")


def _expected_source(row, module, source, value):
    """Reconstruct with literal positions, without calling the editor."""
    positions = []
    for site in row.sites:
        if site.module != module:
            continue
        start = 0
        found = []
        while (offset := source.find(site.anchor, start)) >= 0:
            literal = offset + site.anchor.index(site.literal)
            found.append((literal, literal + len(site.literal)))
            start = offset + len(site.anchor)
        assert len(found) == site.count, (row.name, module, site.anchor)
        positions.extend(found)
    result = source
    replacement = pp.c_float_literal(value)
    for start, end in sorted(positions, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result, positions


@pytest.mark.parametrize("name", sorted(VALUES))
def test_each_single_parameter_changes_only_its_registered_literal_spans(name):
    pset = pp.make_set("single", {name: VALUES[name]})
    pp.declare(pset, source="isolation")
    row = pp.registry()[name]
    changed_modules = set()
    for module in sorted({p.stem for p in KDIR.glob("*.cu")} | {"mynn_dmp_sibling"}):
        source = _source(module)
        expected, positions = _expected_source(row, module, source,
                                                pset.value(name))
        actual = pp.edit_kernel_source(module, source)
        assert actual == expected, (name, module)
        if positions:
            changed_modules.add(module)
            assert actual != source
        else:
            assert actual is source
    assert changed_modules == {site.module for site in row.sites}


def test_case1_length_rows_leave_new_case2_body_byte_identical():
    source = _source("mynn_pbl")
    start = source.index("__device__ void mynn_mym_length_local_column(")
    end = source.index("// module_bl_mynn.F:1999-2098", start)
    baseline = source[start:end]
    assert "MYNN_MUL(3.5f," in baseline
    assert "MYNN_MUL(0.22f," in baseline
    pp.declare(pp.make_set("length", {"mynn.cns": 4.5, "mynn.alp1": 0.32}),
               source="isolation")
    edited = pp.edit_kernel_source("mynn_pbl", source)
    assert baseline in edited
    assert "MYNN_MUL(4.5f," in edited[end:]
    assert "MYNN_MUL(0.32f," in edited[end:]
    for name in ("mynn.cns", "mynn.alp1"):
        assert pp.registry()[name].requires["bl_mynn_mixlength"] == (1,)


@pytest.mark.parametrize("name", ("mynn.cns", "mynn.alp1"))
def test_case1_only_set_refuses_an_experiment_running_only_case2(name):
    from types import SimpleNamespace

    pset = pp.make_set("length", {name: VALUES[name]})
    run = SimpleNamespace(bl_pbl_physics=5, bl_mynn_mixlength=2)
    with pytest.raises(pp.PhysicsParamsError, match="no domain runs that scheme"):
        pp.check_schemes(pset, [run], source="isolation")


def test_default_and_default_valued_sets_preserve_both_loader_source_routes():
    paths = sorted(KDIR.glob("*.cu"))
    baseline = {}
    for path in paths:
        module = path.stem
        raw = path.read_text(encoding="utf-8")
        # A default process compiles the 2.9.1 unit of the exact chain's
        # default-source modules (kernels._DEFAULT_SOURCE_MODULES).
        from woof import wrf_exact
        default_unit = KDIR / f"{module}_default.cu"
        if (not wrf_exact.ENABLED and module in kernels._DEFAULT_SOURCE_MODULES
                and default_unit.is_file()):
            raw = default_unit.read_text(encoding="utf-8")
        if wrf_exact.ENABLED and module == "noah":
            raw = "#define NOAH_WRF_SCALAR_RCT 1\n" + raw   # kernels._unit_text
            if wrf_exact.referee_glibc() >= (2, 43):
                raw = "#define NOAH_WRF_MODERN_LOG10 1\n" + raw
        # The remaining mode-selected unit texts of kernels._unit_text.
        from woof.numerics import is_strict
        if module == "gsfcsw" and not is_strict():
            raw = (KDIR / "gsfcsw_fast.cuh").read_text(encoding="utf-8")
        if module == "slab":
            raw = f"#define GPUWM_SLAB_STRICT {int(is_strict())}\n" + raw
        if module == "ishmael":
            if wrf_exact.ENABLED:
                log10 = KDIR.parents[1] / "hydro/noahowp/log10.cu"
                raw = (log10.read_text(encoding="utf-8")
                       + "\n#define gfk_log10 owp_log10\n" + raw)
            else:
                raw = "#define ISHMAEL_FAST_NUMERICS 1\n" + raw
        if module == "trajectory" and wrf_exact.ENABLED:
            raw = "#define WRF_EXACT 1\n" + raw
        expected = kernels._preamble() + kernels._extra_header_text(module) + raw
        # kernels.module_source's own prefixes, ahead of the preamble.
        if module == "pxsfclay":
            from woof.core.pxsfclay import source_constant_prefix
            expected = source_constant_prefix() + expected
        if module in ("mskf", "mskf_aerosol_replay") and not is_strict():
            expected = "#define GPUWM_MSKF_FAST 1\n" + expected
        if module in kernels._URBAN_NO_FTZ_MODULES:
            # The SLUCM/BEP units retain subnormals through a source marker
            # (woof/cuda_subnormals.py); no other unit's source moves.
            from woof.cuda_subnormals import MARKER
            expected = MARKER + expected
        baseline[module] = expected
        assert kernels.module_source(module) == expected
        defines = (("PHYSICS_PARAMS_SOURCE_CONTROL", 1),)
        tier = "#define PHYSICS_PARAMS_SOURCE_CONTROL 1\n"
        # Goddard (golden/ports-rad) sizes its band headers from the same tier
        # (kernels._HEADERS_NEED_INT_DEFINES), so its tier precedes them too.
        # RCON and UDM size their generated headers from the tier as thompson_column does.
        if module in ("thompson_column", "rcon", "udm") or module in kernels._HEADERS_NEED_INT_DEFINES:
            # thompson-v44 (effc0a99a): the integer tier sizes the generated
            # header's arrays, so it precedes that header as well as the
            # unit (tests/test_thompson_generation_selector.py pins it).
            defined = (kernels._preamble() + tier
                       + kernels._extra_header_text(module) + raw)
        else:
            defined = (kernels._preamble() + kernels._extra_header_text(module)
                       + tier + raw)
        assert kernels.module_source_int_defines(module, defines) == defined
    pp.declare(pp.make_set("defaults", {
        name: row.default for name, row in pp.registry().items()}), source="isolation")
    for module, expected in baseline.items():
        assert kernels.module_source(module) == expected


def test_roughness_registry_excludes_every_seasonal_crop():
    from woof.core.ruc import load_ruc_parameters

    bundle = load_ruc_parameters()
    for name in ("ruc.z0.tall", "ruc.z0.short"):
        row = pp.registry()[name]
        for section, categories in row.categories.items():
            table = bundle.vegetation[section]
            assert len(set(categories)) == len(categories)
            for category in categories:
                assert 1 <= category <= len(table.rows)
                assert table.rows[category - 1].ifor != 7, (name, section, category)


def _compile_pair(cp, name, module, defines=()):
    baseline = (kernels.module_source_int_defines(module, defines)
                if defines else kernels.module_source(module))
    pp.declare(pp.make_set("single", {name: VALUES[name]}), source="device isolation")
    edited = (kernels.module_source_int_defines(module, defines)
              if defines else kernels.module_source(module))
    assert edited != baseline
    modules = [cp.RawModule(code=source, options=kernels.module_options(module),
                            name_expressions=None)
               for source in (baseline, edited)]
    for item in modules:
        item.compile()
    return modules


def _host_result(cp, result, names):
    return {name: cp.asnumpy(getattr(result, name)).copy() for name in names}


def _same_words(left, right, name):
    assert left.dtype == right.dtype and left.shape == right.shape, name
    assert left.tobytes() == right.tobytes(), name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("module", ("mynn_pbl", "mynn_dmp_sibling"))
def test_prandtl_changes_stability_only_and_keeps_kinematic_inputs(module):
    import cupy as cp

    from woof.core.mynn_pbl import MYNN_LEVEL2_INPUTS, MYNN_LEVEL2_OUTPUTS
    from test_mynn_pbl_gpu import _oracle_fields

    _, fields = _oracle_fields()
    inputs = {name: cp.ascontiguousarray(cp.asarray(fields[name].reshape(-1)))
              for name in MYNN_LEVEL2_INPUTS}
    count = inputs["dz"].size
    outputs = []
    for unit in _compile_pair(cp, "mynn.prandtl", module):
        result = {name: cp.empty(count, dtype=cp.float32)
                  for name in MYNN_LEVEL2_OUTPUTS}
        unit.get_function("mynn_level2_pairs")(
            ((count + 127) // 128,), (128,),
            (*(inputs[name] for name in MYNN_LEVEL2_INPUTS),
             *(result[name] for name in MYNN_LEVEL2_OUTPUTS), np.int32(count)))
        outputs.append({name: cp.asnumpy(value).copy()
                        for name, value in result.items()})
    for name in ("dtl", "dqw", "dtv", "gm", "gh"):
        _same_words(outputs[0][name], outputs[1][name], name)
    assert any(outputs[0][name].tobytes() != outputs[1][name].tobytes()
               for name in ("sm", "sh"))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", ("mynn.cns", "mynn.alp1"))
@pytest.mark.parametrize("module", ("mynn_pbl", "mynn_dmp_sibling"))
def test_length_parameter_reaches_case1_and_preserves_controls(monkeypatch, name, module):
    import cupy as cp

    from woof.core import mynn_pbl_gpu
    from test_mynn_pbl import _mixlength_oracle

    _, fields = _mixlength_oracle()
    columns = ("dz", "zw", "u", "v", "qke", "dtv", "theta", "vt", "vq",
               "cldfra", "edmf_w", "edmf_a")
    scalars = ("xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl")
    inputs = {key: cp.asarray(fields[key]) for key in columns}
    inputs.update({key: cp.asarray(fields[key][:, 0]) for key in scalars})
    results = []
    for unit in _compile_pair(cp, name, module):
        def function(_, symbol, shape=(), *, selected=unit):
            # Both logical modules now share the main unit's helper ABI.
            # The sibling is dispatched only for its DMP plume exports;
            # this isolation check compiles its shared helper directly.
            return selected.get_function(symbol)

        monkeypatch.setattr(mynn_pbl_gpu, "get_kernel", function)
        options = (1, 2) if module == "mynn_pbl" else (1,)
        results.append({option: _host_result(
            cp, mynn_pbl_gpu.mynn_mixlength_default_cuda(
                inputs, bl_mynn_mixlength=option), ("el", "qkw"))
            for option in options})
    assert results[0][1]["el"].tobytes() != results[1][1]["el"].tobytes()
    _same_words(results[0][1]["qkw"], results[1][1]["qkw"], "CASE(1) qkw")
    if module == "mynn_pbl":
        for key in ("el", "qkw"):
            _same_words(results[0][2][key], results[1][2][key], f"CASE(2) {key}")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("variant", ("wrf_461", "gsl_wrf39"))
def test_czil_reaches_both_surface_variants_and_preserves_water(monkeypatch, variant):
    import cupy as cp

    from woof.core import mynn_sfclay
    from test_mynn_surface_gpu import _oracle_fields, _device_inputs

    _, fields = _oracle_fields()
    definitions = mynn_sfclay.MYNN_SFCLAY_DEFINES[variant]
    results = []
    for unit in _compile_pair(cp, "mynn.czil", "mynn_surface", definitions):
        monkeypatch.setattr(mynn_sfclay, "get_kernel",
                            lambda _, symbol, shape=(), *, selected=unit: selected.get_function(symbol))
        monkeypatch.setattr(mynn_sfclay, "get_kernel_int_defines",
                            lambda _, symbol, __, shape=(), *, selected=unit: selected.get_function(symbol))
        result = mynn_sfclay.mynn_surface_layer(_device_inputs(fields),
                                               variant=variant)
        results.append(_host_result(cp, result, mynn_sfclay.MYNN_SURFACE_OUTPUTS))
    water = fields["xland"] >= 1.5
    for key in mynn_sfclay.MYNN_SURFACE_OUTPUTS:
        _same_words(results[0][key][water], results[1][key][water], f"water {key}")
    land = ~water
    assert any(results[0][key][land].tobytes() != results[1][key][land].tobytes()
               for key in ("flhc", "hfx", "chs"))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("scalar_mixing", (0, 1))
def test_entrainment_reaches_active_and_sibling_plumes_and_preserves_inactive(
        monkeypatch, scalar_mixing):
    import cupy as cp

    from woof.core import mynn_pbl_gpu
    from woof.core.mynn_pbl import MYNN_DMP_MF_QN_COLUMN_INPUTS
    from test_mynn_pbl import (
        _dmp_mf_oracle, _dmp_mf_case, _dmp_mf_inputs, _dmp_mf_all_outputs)

    module = "mynn_dmp_sibling" if scalar_mixing else "mynn_pbl"
    rows = _dmp_mf_oracle()
    original = mynn_pbl_gpu.get_kernel
    results = []
    for unit in _compile_pair(cp, "mynn.edmf_entrainment", module):
        monkeypatch.setattr(mynn_pbl_gpu, "get_kernel", lambda target, symbol, shape=(), *, selected=unit:
                            selected.get_function(symbol) if target == module
                            else original(target, symbol, shape))
        cases = {}
        for case in ("land_cumulus", "stable_off"):
            values = _dmp_mf_inputs(_dmp_mf_case(rows, case))
            device = {key: cp.asarray(value) for key, value in values.items()}
            if scalar_mixing:
                device.update({key: cp.zeros_like(device["dz"])
                               for key in MYNN_DMP_MF_QN_COLUMN_INPUTS})
            actual = mynn_pbl_gpu.mynn_dmp_mf_cuda(device, bl_mynn_mixscalars=scalar_mixing)
            cases[case] = _host_result(cp, actual, _dmp_mf_all_outputs())
        results.append(cases)
    for key in _dmp_mf_all_outputs():
        _same_words(results[0]["stable_off"][key], results[1]["stable_off"][key], key)
    assert any(results[0]["land_cumulus"][key].tobytes()
               != results[1]["land_cumulus"][key].tobytes()
               for key in _dmp_mf_all_outputs())
