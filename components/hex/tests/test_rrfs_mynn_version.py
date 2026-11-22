from pathlib import Path
"""config_mynn_version: the selectable MYNN suite versions on the rrfs-44c0849 row.

The default is mynn-v6.1 (MYNN-EDMF v6.1.1 + MYNN-SFC v6.1.0, the latest upstream releases;
since 2026-11-08) wherever both MYNN slots run on hex-cuda.  'pinned' is the 44c0849 MYNN, selected by name for the
bitwise claims (its output unchanged: the same module, the same source text), and taken by a module replay or a
run that puts a MYNN slot on the native library.  mynn-sfc-v6.1.0 / mynn-v6.1 select sf_mynnsfc_v610, whose
strict module replay against standalone native MYNN-SFC v6.1.0 is tools/rrfs_mynn_v61_replay (no GPU or native
build here).
"""
import hashlib
import importlib
import re

import pytest

from woof.hex.mpasphys.driver_v842 import slot_cuda_entry
from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist
from woof.hex.physics_suites import load_table

KEY = 'mpasphys_rrfs_44c0849'
HRRRV5 = '&physics config_physics_suite="hrrrv5" /'


def resolve(version=None, mode='correct'):
    text = HRRRV5 + (f"\n&hex_mpasphys config_mynn_version='{version}' /" if version else '')
    return resolve_namelist(text, table_key=KEY, default_backend='hex-cuda', physics_mode=mode)


def test_pinned_is_the_row_entry_unchanged():
    rows = load_table()[KEY]['slots']
    config = resolve('pinned')
    assert config.mynn_version == 'pinned'
    assert slot_cuda_entry(config, rows, 'surface_layer', 'sf_mynnsfclay') is rows['surface_layer']['cuda_entries']['sf_mynnsfclay']
    assert rows['surface_layer']['cuda_entries']['sf_mynnsfclay']['module'] == 'rrfs.sf_mynnsfc_v600'
    assert 'mynn_version' not in config.receipt()
    assert config.mynn_version_line().startswith('MYNN SUITE VERSION: pinned (MYNN-EDMF c6d08709 on hex-cuda, '
                                                 'MYNN-SFC b66187f5 on hex-cuda); the RRFS 44c0849 pin')


def test_default_is_mynn_v61_on_hex_cuda():
    # The default a user gets is the latest upstream release; the pin stays by name.
    assert load_table()[KEY]['mynn_versions']['default'] == 'mynn-v6.1'
    config = resolve()
    assert config.mynn_version == 'mynn-v6.1'
    assert config.receipt() == resolve('mynn-v6.1').receipt() != resolve('pinned').receipt()
    assert config.receipt()['mynn_version_modules'] == {'pbl/bl_mynnedmf': 'rrfs.bl_mynnedmf_v611',
                                                        'surface_layer/sf_mynnsfclay': 'rrfs.sf_mynnsfc_v610'}
    line = config.mynn_version_line()
    assert line.startswith('MYNN SUITE VERSION: mynn-v6.1 (MYNN-EDMF 2b2da61b on hex-cuda, MYNN-SFC a038a9ea on hex-cuda)')
    assert "config_mynn_version='pinned' selects the 44c0849 referee" in line


def test_default_is_the_pin_where_the_pin_runs():
    # A module replay replays the native library, which is the 44c0849 pin.
    assert resolve_namelist(HRRRV5, table_key=KEY).mynn_version == 'pinned'
    # A run naming the native referee for a MYNN slot runs the 44c0849 MYNN there: the run is the pin and says so.
    text = HRRRV5 + "\n&hex_mpasphys config_sfclayer_backend='native-cpu' /"
    config = resolve_namelist(text, table_key=KEY, default_backend='hex-cuda', physics_mode='correct')
    assert config.mynn_version == 'pinned' and 'mynn_version' not in config.receipt()
    assert config.mynn_version_line().startswith('MYNN SUITE VERSION: pinned (MYNN-EDMF c6d08709 on hex-cuda, '
                                                 'MYNN-SFC b66187f5 on native-cpu)')
    # No MYNN slot active: nothing to overlay, no version named, no line.
    off = resolve_namelist('&physics config_physics_suite="none" /', table_key=KEY, default_backend='hex-cuda')
    assert off.mynn_version == 'pinned' and off.mynn_version_line() is None


def test_sfc_version_selects_v610_and_records_its_pin():
    rows = load_table()[KEY]['slots']
    config = resolve('mynn-sfc-v6.1.0')
    entry = slot_cuda_entry(config, rows, 'surface_layer', 'sf_mynnsfclay')
    assert entry['module'] == 'rrfs.sf_mynnsfc_v610'
    assert {k: v for k, v in entry.items() if k != 'module'} == {
        k: v for k, v in rows['surface_layer']['cuda_entries']['sf_mynnsfclay'].items() if k != 'module'}
    assert slot_cuda_entry(config, rows, 'pbl', 'bl_mynnedmf') is rows['pbl']['cuda_entries']['bl_mynnedmf']
    receipt = config.receipt()
    assert receipt['mynn_version'] == 'mynn-sfc-v6.1.0'
    assert receipt['submodule_pins']['MYNN-SFC'] == 'a038a9ea35a12f0bf879fac8b64c99a77ffdae1c'
    assert receipt['submodule_pins']['MYNN-EDMF'] == 'c6d08709e83fbc6245e91a65600fd78c7219f1ee'
    assert receipt['mynn_version_modules'] == {'surface_layer/sf_mynnsfclay': 'rrfs.sf_mynnsfc_v610'}


def test_version_refusals_name_the_breakage():
    with pytest.raises(ValueError, match='unknown config_mynn_version'):
        resolve('mynn-v9')
    text = HRRRV5 + "\n&hex_mpasphys config_mynn_version='mynn-sfc-v6.1.0' config_sfclayer_backend='native-cpu' /"
    with pytest.raises(ValueError, match='hex-cuda only'):
        resolve_namelist(text, table_key=KEY, default_backend='hex-cuda')
    # mynn-v6.1 carries both of its required slots (EDMF v6.1.1 and SFC v6.1.0), so it resolves.
    assert resolve('mynn-v6.1').mynn_version == 'mynn-v6.1'


def test_v610_source_is_the_v600_source_plus_its_cited_edits():
    v600 = importlib.import_module('woof.hex.mpasphys.rrfs.sf_mynnsfc_v600')
    v610 = importlib.import_module('woof.hex.mpasphys.rrfs.sf_mynnsfc_v610')
    assert v610.SOURCE == v610._apply(v600.SOURCE) != v600.SOURCE
    assert (v610.FIELDS, v610.OPTIONS, v610.OUTPUTS, v610.OPTIONAL, v610.MPAS_PRESENT) == (
        v600.FIELDS, v600.OPTIONS, v600.OUTPUTS, v600.OPTIONAL, v600.MPAS_PRESENT)
    cites = re.findall(r'// MYNN-SFC v6\.1\.0 src/\S+:\S+ @a038a9e', v610.SOURCE)
    assert len(cites) >= 13 and not re.findall(r'@a038a9e', v600.SOURCE)
    # The waves variant composes with v6.1.0: both of its anchors still occur once.
    from woof.hex.wave_surface import rrfs_mynn_source
    composed = rrfs_mynn_source(v610.SOURCE)
    assert composed.count('if (ocean_z0>0.f) znt=ocean_z0;') == 1
    # The host glue is the v6.0.0 code, resolving this module's image.
    for name in ('run_packed', 'from_mpas', 'to_mpas', 'driver'):
        bound = getattr(v610, name)
        assert bound.__code__ is getattr(v600, name).__code__ and bound.__globals__ is vars(v610)


#: The differentiable engine's parameter hook, the last lines of each pinned module since 0.5.1. With
#: WOOF_HEX_AD_PARAMS unset (the default) it changes nothing; the route's own text is everything before it.
_AD_PARAMS_HOOK = ('\n\n# WOOF_HEX_AD_PARAMS (default unset: nothing changes) runs the forward at chosen values of the '
                   'carried constants.\nfrom .. import ad_params_override as _ad_params_override\n'
                   '_ad_params_override.patch_module(globals(), "{scheme}")\n')


def _pinned_module_body(path, scheme):
    """The pinned route's text: the module less the parameter hook, which must be its last lines, once, naming this
    scheme. THE BREAKAGE THIS ANSWERS: the pins were read against the whole file, so the hook 0.5.1 appended failed
    both tests on every build since (0.5.1, dist45, dist46, dist47), with the route's text unchanged."""
    text = Path(path).read_bytes()
    hook = _AD_PARAMS_HOOK.format(scheme=scheme).encode()
    assert text.endswith(hook) and text.count(hook) == 1, f'{path}: the parameter hook is not its last lines, once'
    return text[:-len(hook)]


@pytest.mark.parametrize('module, scheme', [('sf_mynnsfc_v600', 'mynnsfc'), ('bl_mynnedmf_c6d08709', 'mynnedmf')])
def test_the_pinned_modules_parameter_hook_changes_nothing_when_unset(module, scheme, monkeypatch):
    from woof.hex.mpasphys import ad_params_override
    monkeypatch.delenv('WOOF_HEX_AD_PARAMS', raising=False)
    mod = importlib.import_module(f'woof.hex.mpasphys.rrfs.{module}')
    before = dict(vars(mod))
    ad_params_override.patch_module(vars(mod), scheme)
    assert all(vars(mod)[k] is v for k, v in before.items()) and set(vars(mod)) == set(before)


def test_pinned_module_text_is_untouched():
    # The pinned route's CUDA text never changes: sha256 of SOURCE, read at 3c9b42ac0, where the whole file still
    # had its woof hex 0.4.15 (4eee442f) hash 2333e36e.  Exact mode and every caller that names no fork id
    # compile exactly this text.
    mod = importlib.import_module('woof.hex.mpasphys.rrfs.sf_mynnsfc_v600')
    assert hashlib.sha256(mod.SOURCE.encode()).hexdigest() == (
        '98df75d230195487a65c9f944be38aae0ac5b6c064b7a2e83f3f7248e7bf22f4')
    assert mod.cuda_source(corrections=('MY1', 'TP4')) == mod.cuda_source() != mod.cuda_source(corrections=('SF1',))
    # The file around that text: re-pinned once for correct mode's SF1 text pair and its keyword
    # (fork_corrections); any later edit to this file still fails here.
    assert hashlib.sha256(_pinned_module_body(mod.__file__, 'mynnsfc')).hexdigest() == (
        'b752b99a945f3605ecf6b6554af016abd09f463c8f390c8778f801f7a2833f6c')


def test_v61_selects_edmf_v611_with_its_own_image():
    rows = load_table()[KEY]['slots']
    pinned = rows['pbl']['cuda_entries']['bl_mynnedmf']
    config = resolve('mynn-v6.1')
    entry = slot_cuda_entry(config, rows, 'pbl', 'bl_mynnedmf')
    assert entry['module'] == entry['resources']['module'] == 'rrfs.bl_mynnedmf_v611'
    assert entry['resources']['packet_key'] == 'bl_mynnedmf_v611' != pinned['resources']['packet_key']
    assert (entry['call'], entry['bind'], entry['resources']['attribute']) == (
        pinned['call'], pinned['bind'], pinned['resources']['attribute'])
    receipt = config.receipt()
    assert receipt['submodule_pins']['MYNN-EDMF'] == '2b2da61b05a8ef9f9b06795c816dc01e347e0736'
    assert receipt['mynn_version_modules']['pbl/bl_mynnedmf'] == 'rrfs.bl_mynnedmf_v611'
    # 'pinned' keeps the row's own entry (c6d08709).
    assert slot_cuda_entry(resolve('pinned'), rows, 'pbl', 'bl_mynnedmf') is pinned
    assert pinned['module'] == 'rrfs.bl_mynnedmf_c6d08709'


def test_every_edmf_image_the_table_names_is_packed():
    # THE BREAKAGE: an image the packs do not carry compiles through NVRTC on one host thread (~11 min for v6.1.1)
    # before the first step.
    from woof.hex.mpasphys.resources_build import rrfs_mynnedmf_images
    images = dict(rrfs_mynnedmf_images())
    assert images['bl_mynnedmf_v611'] == 'rrfs.bl_mynnedmf_v611'
    assert images['bl_mynnedmf_c6d08709'] == 'rrfs.bl_mynnedmf_c6d08709'


def test_pinned_edmf_module_text_is_untouched():
    # The pinned route's CUDA text never changes: sha256 of MODULE_CUDA and ARRAY_SUPPORT, read at 3c9b42ac0,
    # where the whole file still had its woof hex 0.4.15 (4eee442f) hash 00ebad3f.
    mod = importlib.import_module('woof.hex.mpasphys.rrfs.bl_mynnedmf_c6d08709')
    assert hashlib.sha256(mod.MODULE_CUDA.encode()).hexdigest() == (
        '32ce71b78a8438c208a62e375ff44eab3e2aa57cfdabdee8e4f62ab11c228c43')
    assert hashlib.sha256(mod.ARRAY_SUPPORT.encode()).hexdigest() == (
        'd9410165f9d1be5ed13fbc3bd315698474ae8484387b37866dd1dc54a24d2ebe')
    assert mod.module_cuda() is mod.MODULE_CUDA and mod.module_cuda(('SF1', 'TP4')) is mod.MODULE_CUDA
    # The file around that text: re-pinned for correct mode's MY1 and MY2 text pairs, then MY3 to MY5 (fork_corrections);
    # any later edit to this file still fails here.
    assert hashlib.sha256(_pinned_module_body(mod.__file__, 'mynnedmf')).hexdigest() == (
        'd5f3b5d3ac397b0788c6da9990525de095222d3787819240530116f6e3739d3a')


def test_v611_is_its_own_unit_on_the_44c0849_host():
    c6d = importlib.import_module('woof.hex.mpasphys.rrfs.bl_mynnedmf_c6d08709')
    v611 = importlib.import_module('woof.hex.mpasphys.rrfs.bl_mynnedmf_v611')
    assert v611.CUDA_OPTIONS == c6d.CUDA_OPTIONS
    assert v611.ARRAY_SUPPORT.startswith(c6d.ARRAY_SUPPORT) and 'MynnOptionalRef' in v611.ARRAY_SUPPORT
    assert v611.MODULE_CUDA.count('NCAR/MYNN-EDMF@2b2da61 src/module_bl_mynnedmf_driver.F90') > 100
    assert 'MPAS-Model-rrfs' not in v611.MODULE_CUDA
    # The v6 driver's interface pressure is the host's pres2_hyd_p (noaa/develop de016e0 driver_pbl.F:1748-1828).
    assert v611.NATIVE_ARRAY_BINDINGS['pint'] == 'pres2_hyd_p' and v611.NATIVE_ARRAY_BINDINGS['tk'] == 't_p'
    assert {'pint', 'restart', 'cycling', 'tk', 'bl_mynn_diags2d'} <= set(v611.ARGUMENTS)
    assert set(v611.HOST_FLAGS) == {n for n in c6d.ARGUMENTS if n.startswith('f_')}


def test_v611_arena_holds_one_chemistry_species_without_chemistry():
    # THE BREAKAGE: the v6 driver calls the core with nchem1 = ndvel1 = 1 when chemistry is off, so the core's
    # (kts:kte, nchem) arrays hold one species; a slot sized with the host's num_chem = 0 was short and plume
    # columns overwrote the next warp tile's column (x1.2562 hrrrv5 step 1, strict: 20 columns, NaN tendencies).
    v611 = importlib.import_module('woof.hex.mpasphys.rrfs.bl_mynnedmf_v611')
    off = dict(kts=1, kte=59, kme=60, nchem=0, ndvel=0)
    assert v611.workspace_bytes(off) == v611.workspace_bytes(dict(off, nchem=1, ndvel=1))


def test_proofs_against_native_44c0849_name_the_pin(tmp_path):
    # verify_mpas compares against native 44c0849, which runs the pinned MYNN: a deck that names no version gets
    # config_mynn_version=pinned on the hex side; a deck that names one keeps it; other routes are unchanged.
    from woof.hex.verify_mpas import proof_referee_versions
    namelist = tmp_path / 'namelist.atmosphere'
    namelist.write_text(HRRRV5 + '\n')
    base = ['woof hex', 'forecast', '--physics-backend', 'mpasphys']
    run, record = proof_referee_versions(base, namelist)
    assert run == base + ['--physics', 'config_mynn_version=pinned'] and record['appended'] == run[len(base):]
    named = base + ['--physics', 'config_mynn_version=mynn-v6.1']
    assert proof_referee_versions(named, namelist)[0] == named
    namelist.write_text(HRRRV5 + "\n&hex_mpasphys config_mynn_version='mynn-v6.1' /\n")
    assert proof_referee_versions(base, namelist)[0] == base
    namelist.write_text('&physics config_physics_suite="convection_permitting" /\n')
    assert proof_referee_versions(base, namelist)[0] == base


def test_exact_mode_takes_the_pin_unless_named():
    # Exact mode reproduces the pinned native fork: a deck that names no version runs the 44c0849 MYNN, as TEMPO's
    # exact_mode_defaults does; a named version is kept.
    assert resolve(mode='exact').mynn_version == 'pinned'
    assert resolve('mynn-v6.1', mode='exact').mynn_version == 'mynn-v6.1'
    assert resolve(mode='correct').mynn_version == 'mynn-v6.1'
