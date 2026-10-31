"""Validate the MYNN driver contract against the pinned native source."""
from dataclasses import fields
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest

from woof.hex.mpasphys.bl_mynn_v842 import MynnOptions, MYNN_ARGUMENTS, WRAPPER_TO_MMM


def test_sheet_history_retention_survives_verified_output_retirement(tmp_path):
    """The observer keeps the actual inode and authenticates the final bytes."""
    import hashlib
    import importlib.util
    import time
    helper = Path(__file__).parent/'mpas_mynn_v842/retain_sheet_inputs.py'
    spec = importlib.util.spec_from_file_location('mynn_sheet_inputs', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path/'forecast'
    output.mkdir()
    observer = module.SheetInputs(output, tmp_path/'retained', '2023-03-31_12:00:00')
    observer.start()
    hashes = {}
    try:
        for name in observer.names:
            with (output/name).open('wb') as stream:
                stream.write(b'first part')
                deadline = time.monotonic()+5
                while not (observer.retained/name).exists():
                    assert time.monotonic()<deadline
                    time.sleep(0.01)
                stream.write(b' final part')
            hashes[name] = hashlib.sha256((output/name).read_bytes()).hexdigest()
            (output/name).unlink()
        assert observer.finish({'provenance': {'hex_output_sha256': hashes}}) == hashes
    finally:
        observer.stop.set()
        observer.thread.join()


def test_sheet_history_retention_rejects_a_provenance_mismatch(tmp_path):
    import importlib.util
    import time
    helper = Path(__file__).parent/'mpas_mynn_v842/retain_sheet_inputs.py'
    spec = importlib.util.spec_from_file_location('mynn_sheet_inputs', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path/'forecast'
    output.mkdir()
    observer = module.SheetInputs(output, tmp_path/'retained', '2023-03-31_12:00:00')
    observer.start()
    try:
        for name in observer.names:
            (output/name).write_bytes(b'original forecast bytes')
        deadline = time.monotonic()+5
        while not all((observer.retained/name).exists() for name in observer.names):
            assert time.monotonic()<deadline
            time.sleep(0.01)
        with pytest.raises(ValueError, match='differs from the actual word proof'):
            observer.finish({'provenance': {'hex_output_sha256': {
                name: '0'*64 for name in observer.names}}})
    finally:
        observer.stop.set()
        observer.thread.join()


@pytest.fixture(scope="module")
def mpas():
    root = os.environ.get("MPAS_REF")
    if not root:
        pytest.skip("MPAS_REF is required for pinned source contract tests")
    return Path(root) / "src/core_atmosphere"


def test_every_registry_option_and_default_is_preserved(mpas):
    defaults = MynnOptions()
    options = {node.attrib["name"].removeprefix("config_mynn_"): node.attrib
               for node in ET.parse(mpas / "Registry.xml").iter("nml_option")
               if node.attrib.get("name", "").startswith("config_mynn_")}
    assert set(options) == {field.name for field in fields(defaults)}
    kinds = {"integer": int, "real": float, "logical": lambda v: v == "true"}
    for name, option in options.items():
        assert getattr(defaults, name) == kinds[option["type"]](option["default_value"])


def test_wrapper_switches_use_exact_native_integer_comparison(mpas):
    wrapper = (mpas / "physics/physics_wrf/module_bl_mynn.F").read_text().lower()
    switches = re.findall(r"if\(bl_mynn_(\w+)\s*==\s*1\)", wrapper)
    aliases = {"output": "edmf_output", "cloudmix": "mixclouds"}
    converted = {"dheat_opt": "dheatopt"}
    for source_name in switches:
        field = aliases.get(source_name, source_name)
        dest = converted.get(source_name, source_name)
        for value in (-1, 0, 1, 2):
            options = MynnOptions.from_namelist({"config_mynn_"+field: value})
            assert options.mmm_arguments()["bl_mynn_"+dest] is (value == 1)


def test_mmm_argument_names_match_complete_native_run_interface(mpas):
    source = (mpas / "physics/physics_mmm/bl_mynn.F90").read_text().lower()
    signature = source.split("subroutine bl_mynn_run", 1)[1].split("!input arguments:", 1)[0]
    signature = re.sub(r"!.*", "", signature)
    native = set(re.findall(r"\bbl_mynn_\w+\b", signature))
    assert set(MynnOptions().mmm_arguments()) == native


def test_wrapper_launcher_covers_the_actual_nonchemistry_interface(mpas):
    source = (mpas / "physics/physics_wrf/module_bl_mynn.F").read_text().lower()
    signature = source.split("subroutine mynn_bl_driver", 1)[1].split("!--- input arguments:", 1)[0]
    signature = signature.split("#if", 1)[0]
    native_names = set(re.findall(r"\b[a-z]\w*\b", re.sub(r"!.*", "", signature)))
    local_specific = {"sqv", "sqc", "sqi", "sqs"}
    absent_dd = {name for name in MYNN_ARGUMENTS if name.startswith("edmf_") and name.endswith("_dd")}
    assert len(absent_dd) == 6
    for name in MYNN_ARGUMENTS:
        if name in local_specific | absent_dd or name.startswith("bl_mynn_"):
            continue
        assert WRAPPER_TO_MMM.get(name, name) in native_names, name
    call = source.split("call bl_mynn_run", 1)[1].split("!--- conversion of tendencies", 1)[0]
    for name in absent_dd:
        assert not re.search(r"\b"+name+r"\s*=", call)


def test_native_pool_binding_covers_complete_pinned_pbl_call(mpas):
    from woof.hex.mpasphys.bl_mynn_v842 import (
        _MYNN_PREPARED, _MYNN_CONSTANTS, _MYNN_FLAGS, _MYNN_BOUNDS,
        _DIAGNOSTICS, _BUDGETS, _ZERO_TENDENCIES,
    )
    source = (mpas / 'physics/mpas_atmphys_driver_pbl.F').read_text().lower()
    call = source.split('call mynn_bl_driver(', 1)[1].split('call mpas_timer_stop', 1)[0]
    native = set(re.findall(r'\b([a-z]\w*)\s*=', re.sub(r'!.*', '', call)))
    option_names = {'bl_mynn_'+({'edmf_output':'output', 'mixclouds':'cloudmix'}.get(f.name, f.name))
                    for f in fields(MynnOptions)}
    bound = (set(_MYNN_PREPARED) | set(_MYNN_FLAGS) | set(_MYNN_BOUNDS)
             | set(_DIAGNOSTICS.values()) | set(_BUDGETS) | set(_ZERO_TENDENCIES)
             | option_names | {'icloud_bl','delt','dx','xland','ts','qsfc','ust','ch',
                'hfx','qfx','rmol','wspd','znt','uoce','voce','rthraten','pblh','kpbl',
                'maxwidth','maxmf','ktop_plume','ztop_plume','spp_pbl','pattern_spp',
                'do_restart','do_dacycling','initflag','errmsg','errflg'})
    assert not (native-bound)
    # tkepbl_p is copied through by the driver but deliberately absent from MYNN.
    assert bound-native == {'tke_pbl'}
    for wrapper, prepared in _MYNN_PREPARED.items():
        assert re.search(r'\b'+wrapper+r'\s*=\s*'+prepared+r'\b', call)
    init = source.split('call bl_mynn_init(', 1)[1].split(')', 1)[0]
    assert tuple(x.strip() for x in init.replace('&','').replace('\n','').split(',')[:-2]) == _MYNN_CONSTANTS


@pytest.mark.parametrize('columns,stride,processors,free,expected', [
    (25, 100000, 170, 8*1024**3, 64),
    (99274, 100000, 170, 16*1024**3, 10880),
    (99274, 100000, 170, 0, 64),
    (99274, 100000, 2, 16*1024**3, 128),
])
def test_mynn_batch_respects_available_memory_and_device_width(columns, stride, processors, free, expected):
    from woof.hex.mpasphys.bl_mynn_v842 import _column_batch
    assert _column_batch(columns, stride, processors, free, shared_card=True) == expected


def test_worker_scratch_is_released_without_repeating_device_queries():
    """A completed call must let the pool lend its workspace to the dycore."""
    from types import SimpleNamespace
    import weakref
    import numpy as np
    from woof.hex.mpasphys.bl_mynn_v842 import MynnCudaModule
    queries = []
    def properties(device):
        queries.append(('properties', device))
        return {'multiProcessorCount': 2}
    def memory():
        queries.append(('memory',))
        return 1 << 30, 2 << 30
    module = MynnCudaModule.__new__(MynnCudaModule)
    module.cp = SimpleNamespace(empty=np.empty, uint8=np.uint8, int32=np.int32,
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(ptr=0),
            runtime=SimpleNamespace(getDevice=lambda: 0, getDeviceProperties=properties,
                                    memGetInfo=memory)))
    module._workers = None
    module._shared_card_memory = True
    workspace, errors, batch = module._storage(128, 16)
    held = weakref.ref(workspace)
    assert workspace.size == batch * 16 and errors.shape == (128,)
    del workspace
    assert held() is None
    other, next_errors, next_batch = module._storage(128, 16)
    assert next_batch == batch and next_errors is not errors
    assert queries == [('properties', 0), ('memory',)]
