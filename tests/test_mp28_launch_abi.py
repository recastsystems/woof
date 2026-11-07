"""CPU checks of the production Python launch boundary against the CUDA ABI."""

import ast
import inspect
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _parameters(module, kernel):
    text = (ROOT / "woof/core/kernels" / f"{module}.cu").read_text()
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    match = re.search(r"\bvoid\s+" + kernel + r"\s*\((.*?)\)\s*\{", text, re.S)
    assert match, kernel
    parameters = match.group(1).strip()
    if re.fullmatch(r"[A-Z_]+", parameters):
        macro = re.search(r"#define\s+" + parameters + r"\s+((?:[^\n]*\\\n)*[^\n]*)", text)
        assert macro, parameters
        parameters = macro.group(1).replace("\\\n", " ")
    return [part.strip().split()[-1].lstrip("*")
            for part in parameters.split(",")]


def _table(shape):
    # No large table allocation is needed to verify the launch boundary.
    return SimpleNamespace(shape=shape, dtype=np.dtype(np.float64),
                           flags=SimpleNamespace(f_contiguous=True))


@pytest.mark.parametrize("family", ["cold", "warm"])
@pytest.mark.parametrize("frozen", [False, True])
def test_production_network_tuple_matches_the_compiled_parameter_order(monkeypatch, family, frozen):
    from woof.core import thompson_aerosol_cold as cold
    from woof.core import thompson_aerosol_warm as warm

    owner = cold if family == "cold" else warm
    launcher = (cold.launch_aa_cold_network if family == "cold"
                else warm.launch_aerosol_warm_source_network)
    kernel = ("thompson_aa_cold_network" if family == "cold"
              else "thompson_aa_warm_source_network")
    module = f"thompson_aerosol_{family}"
    captured = []

    def capture(module_name, kernel_name):
        assert (module_name, kernel_name) == (module, kernel)
        return lambda grid, block, arguments: captured.append(arguments)

    # Replace only the native launch boundary. Production argument validation,
    # table expansion, tuple construction and optional-buffer selection run.
    monkeypatch.setattr(owner, "aerosol_kernel", capture)
    values = {}
    for name, parameter in inspect.signature(launcher).parameters.items():
        if name == "dt":
            values[name] = 20.0
        elif name in ("rain_snow_tables", "rain_graupel_tables"):
            table_names = getattr(owner, name.upper().replace("TABLES", "TABLE_NAMES"))
            shape = getattr(owner, "_" + name.replace("tables", "shape"))()
            values[name] = tuple(_table(shape) for _ in table_names)
        elif name in ("rain_freezing_tables", "cloud_freezing_tables"):
            table_names = getattr(cold, name.upper().replace("TABLES", "TABLE_NAMES"))
            shape = getattr(cold, name.upper().replace("TABLES", "TABLE_SHAPE"))
            values[name] = tuple(_table(shape) for _ in table_names)
        elif name in ("ice_deposition_partition", "ice_to_snow_mass", "ice_to_snow_number"):
            values[name] = _table(cold.ICE_PARTITION_SHAPE)
        elif name in ("rain_cloud_efficiency", "snow_cloud_efficiency"):
            values[name] = _table((100, 100))
        elif name in ("graupel_volume", "qbten"):
            values[name] = None
        elif name in ("qsten", "qgten", "ngten") and not frozen:
            values[name] = None
        else:
            values[name] = np.zeros((1,), dtype=np.float32)

    launcher(**values)
    assert len(captured) == 1
    arguments = captured[0]
    parameters = _parameters(module, kernel)
    assert len(arguments) == len(parameters), (kernel, len(arguments), len(parameters))
    assert parameters[-7:] == ["qsten", "qgten", "ngten", "graupel_volume", "qbten", "dt", "size"]
    for name in ("qsten", "qgten", "ngten"):
        assert arguments[parameters.index(name)] is values[name]
    if family == "cold":
        assert arguments[parameters.index("snow_cloud_efficiency")] is values["snow_cloud_efficiency"]


def test_verified_owner_wrappers_forward_all_frozen_buffers():
    for filename, function in (
            ("thompson_aerosol_cold.py", "launch_aa_cold_network_from_owner"),
            ("thompson_aerosol_warm.py", "launch_aerosol_warm_source_network_from_owner")):
        tree = ast.parse((ROOT / "woof/core" / filename).read_text())
        wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == function)
        forwarded = {keyword.arg for node in ast.walk(wrapper)
                     if isinstance(node, ast.Call) for keyword in node.keywords}
        assert {"qsten", "qgten", "ngten"} <= forwarded, function


@pytest.mark.parametrize("family", ["snow", "graupel"])
@pytest.mark.parametrize("frozen", [False, True])
def test_sedimentation_tuple_matches_serial_and_parallel_cuda_abis(monkeypatch, family, frozen):
    from woof.core import thompson_aerosol_sed as sed

    launcher = getattr(sed, f"launch_aa_{family}_sedimentation")
    captured = []
    monkeypatch.setattr(sed, "_column_launch",
                        lambda kernel, nz, ny, nx, arguments: captured.append((kernel, arguments)))
    values = {}
    for name in inspect.signature(launcher).parameters:
        if name == "dt":
            values[name] = 20.0
        elif name in ("snow_precipitation", "graupel_precipitation", "active_columns"):
            values[name] = np.zeros((1, 1), dtype=np.float32)
        elif name in ("graupel_volume", "qbten"):
            values[name] = None
        elif name in ("qsten", "qgten", "ngten") and not frozen:
            values[name] = None
        else:
            values[name] = np.zeros((2, 1, 1), dtype=np.float32)
    launcher(**values)
    assert len(captured) == 1
    kernel, arguments = captured[0]
    for suffix in ("_64", "_levels_64", "_256"):
        parameters = _parameters("thompson_aerosol_sed", kernel + suffix)
        assert len(arguments) == len(parameters), (kernel + suffix, len(arguments), len(parameters))
        carriers = ("qsten",) if family == "snow" else ("qgten", "ngten")
        for name in carriers:
            assert arguments[parameters.index(name)] is values[name]


@pytest.mark.parametrize("stage", ["saturation", "rain", "phase"])
@pytest.mark.parametrize("heat", [False, True])
def test_heat_carrier_launch_closure_matches_cuda_abi(monkeypatch, stage, heat):
    from woof.core import thompson_aerosol_sat as sat
    from woof.core import thompson_aerosol_sed as sed

    owner, launcher, kernel, module = {
        "saturation": (sat, sat.launch_aerosol_saturation_adjust,
                       "thompson_aa_saturation_adjust", "thompson_aerosol_sat"),
        "rain": (sat, sat.launch_aerosol_rain_evaporation,
                 "thompson_aa_rain_evaporation", "thompson_aerosol_sat"),
        "phase": (sed, sed.launch_aa_final_phase_cleanup,
                  "thompson_aa_final_phase_cleanup", "thompson_aerosol_sed"),
    }[stage]
    captured = []

    def capture(module_name, kernel_name):
        assert (module_name, kernel_name) == (module, kernel)
        return lambda grid, block, arguments: captured.append(arguments)

    monkeypatch.setattr(owner, "aerosol_kernel", capture)
    values = {}
    for name in inspect.signature(launcher).parameters:
        if name == "dt":
            values[name] = 20.0
        elif name == "tnccn_act":
            values[name] = _table(sat.CCN_ACTIVATION_SHAPE)
        elif name == "tnc_wev":
            values[name] = _table(sat.DROP_EVAP_SHAPE)
        elif not heat and (name in ("heat_ocp", "heat_lvap", "tau1_temperature")
                           or stage == "phase" and name in ("tten", "theta", "exner")):
            values[name] = None
        else:
            values[name] = np.zeros((1,), dtype=np.float32)
    launcher(**values)
    assert len(captured) == 1
    arguments = captured[0]
    parameters = _parameters(module, kernel)
    assert len(arguments) == len(parameters), (kernel, len(arguments), len(parameters))
    for name in ("heat_ocp", "heat_lvap", "tau1_temperature"):
        if name in values:
            assert arguments[parameters.index(name)] is values[name]


def test_production_adapter_zeroes_and_forwards_the_frozen_carriers():
    text = (ROOT / "woof/core/microphysics_aerosol.py").read_text()
    tree = ast.parse(text)
    for carrier in ("qsten", "qgten", "ngten"):
        assert f'"mp_thompson_aero_{carrier}"' in text
    network_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id in (
                         "launch_aa_cold_network_from_owner",
                         "launch_aerosol_warm_source_network_from_owner")]
    assert len(network_calls) == 2
    for call in network_calls:
        assert any(keyword.arg is None and any(isinstance(node, ast.Name)
                   and node.id == "frozen_tendencies" for node in ast.walk(keyword.value))
                   for keyword in call.keywords)
    accumulator_loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == "accumulator"]
    assert any({"qsten", "qgten", "ngten"} <= {
        node.id for node in ast.walk(loop.iter) if isinstance(node, ast.Name)}
        for loop in accumulator_loops)
