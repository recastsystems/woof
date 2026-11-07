"""Eta Ferrier (mp_physics=95): card-free checks of the port's inputs.

* the committed oracle fixtures are the bytes their manifest names;
* the packaged ETAMPNEW_DATA.expanded_rain is WRF v4.6.1's own file, its
  twelve records parse to the lengths ETANEWinit reads, and a file with
  other bytes or a truncated record is refused by name;
* the folded constants kernels/etanew.cu carries as hexadecimal literals
  (powers of literals gfortran evaluates at compile time) are recomputed
  here with exact rational arithmetic, rounded once to binary32.
"""

from __future__ import annotations

import hashlib
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from woof.core import etanew_tables as tables

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "oracles" / "etanew"
KERNEL = ROOT / "woof" / "core" / "kernels" / "etanew.cu"


def test_oracle_fixtures_match_their_manifest():
    for line in (FIXTURES / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        got = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        assert got == digest, name


def test_packaged_table_is_wrfs_file_and_parses():
    t = tables.load_tables()
    assert t.sha256 == tables.TABLE_SHA256
    assert t.flat.size == tables.TABLE_WORDS == 11 * 951 + 328
    for name, words in tables.RECORDS:
        assert t[name].size == words
        assert np.all(np.isfinite(t[name]))
    # spot values a mis-ordered parse could not reproduce
    assert t["VRAIN"][0] == np.float32(0.7277349)
    assert t["MASSI"][-1] == np.float32(6.063416e-08)
    assert t["VEL_RF"][0] == np.float32(1.0)


def test_table_with_other_bytes_is_refused(tmp_path):
    raw = bytearray((tables.table_root() / tables.TABLE_FILENAME)
                    .read_bytes())
    raw[100] ^= 0x01
    (tmp_path / tables.TABLE_FILENAME).write_bytes(bytes(raw))
    with pytest.raises(tables.EtampnewTableError, match="sha256"):
        tables.load_tables(tmp_path)


def test_truncated_record_is_refused_by_name():
    raw = (tables.table_root() / tables.TABLE_FILENAME).read_bytes()
    # a file cut inside the first record, and a first record whose
    # leading marker names the wrong length, are both refused by name
    with pytest.raises(tables.EtampnewTableError, match="sha256"):
        tables.parse_table_bytes(raw[:2000])
    bad = bytearray(raw)
    bad[0:4] = (4 * 950).to_bytes(4, "big")
    with pytest.raises(tables.EtampnewTableError, match="sha256"):
        tables.parse_table_bytes(bytes(bad))


def _rn32(value: Fraction) -> np.float32:
    """Round an exact rational to the nearest binary32 (ties to even)."""
    x = np.float32(float(value))
    cands = (np.nextafter(x, np.float32(-np.inf)), x,
             np.nextafter(x, np.float32(np.inf)))
    return min(cands, key=lambda c: (abs(Fraction(float(c)) - value),
                                     int(np.asarray(c).view(np.uint32)) & 1))


def _kernel_literal(name: str) -> np.float32:
    text = KERNEL.read_text(encoding="utf-8")
    m = re.search(rf"#define {name} (0x[0-9a-fA-Fp.+-]+)f", text)
    assert m, name
    return np.float32(float.fromhex(m.group(1)))


@pytest.mark.parametrize("name,base,power", [
    ("ETA_DMRMIN4", .05e-3, 4),
    ("ETA_DMRMAX4", 1.e-3, 4),
    ("ETA_RFMAX_FOLD", 1.1, 40),
    ("ETA_C20E6_3", 20.e-6, 3),
])
def test_folded_integer_powers(name, base, power):
    exact = Fraction(float(np.float32(base))) ** power
    assert _kernel_literal(name) == _rn32(exact)


def test_folded_real_power_of_a_literal():
    # (1.E5)**C1, C1 = 1./3. in binary32: correctly rounded, which is what
    # gfortran's MPFR folding produces.  Bracket the real value tightly
    # with float64 logs and check the literal is the nearest binary32.
    c1 = np.float32(1.0) / np.float32(3.0)
    value = 1.0e5 ** float(c1)
    lit = _kernel_literal("ETA_1E5_POW_C1")
    lo = np.nextafter(lit, np.float32(-np.inf))
    hi = np.nextafter(lit, np.float32(np.inf))
    assert abs(float(lit) - value) < min(abs(float(lo) - value),
                                         abs(float(hi) - value))


def test_pi_and_table_end_literals():
    assert _kernel_literal("ETA_PI") == np.float32(np.pi)
    assert _kernel_literal("ETA_XMRMAX") == np.float32(1.e6) * np.float32(1.e-3)


def _compare_module():
    import importlib.util
    path = ROOT / 'tools/etanew_wrf461_oracle/compare_etanew.py'
    spec = importlib.util.spec_from_file_location('eta_compare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparator_refuses_absent_results(tmp_path):
    assert not _compare_module().compare(tmp_path, tmp_path)[0]


@pytest.mark.parametrize('missing', ['levels.csv', 'surface.csv', 'init.csv', 'tables.csv'])
def test_comparator_requires_every_result(tmp_path, missing):
    import shutil
    for name in ('levels.csv', 'surface.csv', 'init.csv', 'tables.csv'):
        if name != missing:
            shutil.copyfile(FIXTURES / name, tmp_path / name)
    assert not _compare_module().compare(FIXTURES, tmp_path)[0]


def test_comparator_detects_one_bit_in_carried_state(tmp_path):
    import shutil
    for name in ('levels.csv', 'surface.csv', 'init.csv', 'tables.csv'):
        shutil.copyfile(FIXTURES / name, tmp_path / name)
    p = tmp_path / 'levels.csv'
    rows = p.read_text().splitlines()
    cells = rows[1].split(',')
    cells[5] = f'{int(cells[5], 16) ^ 1:08X}'
    rows[1] = ','.join(cells)
    p.write_text('\n'.join(rows) + '\n')
    ok, fields = _compare_module().compare(FIXTURES, tmp_path)
    assert not ok
    row = next(r for r in fields if r['field'] == 'qt')
    assert row['differ'] == 1 and row['max_ulp'] == 1


def test_mp95_passes_the_actual_config_door():
    from woof.config import RunConfig, validate_run_config
    from woof.physics_registry import implemented_selector_values
    cfg=RunConfig(nx=8,ny=6,nz=16,dx=3000.,dy=3000.,ztop=12000.,
                  dt=10.,run_seconds=20.,mp_physics=95,moist=True)
    validate_run_config(cfg)
    assert implemented_selector_values('microphysics')[95]=='etanew-mp95'


def test_eta_restart_fields_are_serialized_and_rk_copy_is_rebuilt():
    from woof.io.restart import classify_state_attr
    for name in ('qt','f_ice_phy','f_rain_phy','f_rimef_phy',
                 'etanew_restart_state','etanew_tbpvs','etanew_tbpvs0',
                 'etanew_restart_dt'):
        assert classify_state_attr(name)=='serialize',name
    assert classify_state_attr('qt0')=='rebuild'


def test_native_input_inventory_carries_total_condensate():
    from types import SimpleNamespace
    from woof.ingest.wrfinput import active_moisture_map, active_moisture_inventory
    cfg=SimpleNamespace(mp_physics=95,moist=True)
    required,allowed=active_moisture_inventory(cfg)
    assert required=={'QVAPOR','QCLOUD','QRAIN','QSNOW'}
    assert allowed==required|{'CWM'}
    assert dict(active_moisture_map(cfg))=={'QVAPOR':'qv','QCLOUD':'qc',
                'QRAIN':'qr','QSNOW':'qs','CWM':'qt'}


def test_memory_manifest_matches_real_state_allocation(monkeypatch):
    from test_restart import _cfg, _shim_state
    from woof.core.device_inventory import state_array_shapes
    cfg=_cfg(moist=True,mp_physics=95)
    state=_shim_state(cfg,monkeypatch)
    planned=state_array_shapes(cfg)
    for name in ('qt','qs','qt0','qs0','f_ice_phy','f_rain_phy','f_rimef_phy',
                 'etanew_restart_state','etanew_tbpvs','etanew_tbpvs0','etanew_restart_dt'):
        assert getattr(state,name).shape==planned[name],name


def test_stock_transport_holds_partitioned_condensate():
    from types import SimpleNamespace
    from woof.core.moist import moist_species
    assert moist_species(SimpleNamespace(qt=object(),etanew_restart_state=object()))==('qv','qt')


def test_eta_history_state_has_wrfs_names_and_units():
    from types import SimpleNamespace
    from woof.io.history_layout import live_state_history_fields
    from woof.io.wrf_output_schema import REGISTRY_VAR_META
    values={k:np.zeros((2,2,2),np.float32) for k in ('qt','f_ice_phy','f_rain_phy','f_rimef_phy')}
    fields=live_state_history_fields(SimpleNamespace(**values,etanew_restart_state=object()))
    assert set(fields)=={'CWM','F_ICE_PHY','F_RAIN_PHY','F_RIMEF_PHY'}
    assert REGISTRY_VAR_META['CWM']==('Total condensate mixing ratio','kg kg-1')
    assert REGISTRY_VAR_META['F_ICE_PHY']==('FRACTION OF ICE','')
    assert REGISTRY_VAR_META['F_RAIN_PHY']==('FRACTION OF RAIN ','')
    assert REGISTRY_VAR_META['F_RIMEF_PHY']==('MASS RATIO OF RIMED ICE ','')


def test_lookup_path_is_shared_with_mp5(monkeypatch,tmp_path):
    monkeypatch.delenv('WOOF_ETAMPNEW_TABLE_ROOT',raising=False)
    monkeypatch.delenv('WOOF_FERRIER_TABLE_ROOT',raising=False)
    assert tables.table_root()==ROOT/'woof/data/ferrier/tables'
    monkeypatch.setenv('WOOF_FERRIER_TABLE_ROOT',str(tmp_path))
    assert tables.table_root()==tmp_path
    preferred=tmp_path/'eta'
    monkeypatch.setenv('WOOF_ETAMPNEW_TABLE_ROOT',str(preferred))
    assert tables.table_root()==preferred

def test_both_public_table_variants_share_the_lookup_loader():
    from woof.core.etanew_tables import load_tables
    original=load_tables(variant='original_rain')
    expanded=load_tables()
    assert original.flat.size==7489 and expanded.flat.size==10789
    assert original['VENTR1'].size==401 and original['VSNOWI'].size==951
    assert original.sha256=='a913785f0bf5f75f8428ede706ce67ae804c3b285b4a34aeb533c4ad5cd9bea2'
    assert expanded.sha256=='28e60e6c83eb55889c4ab65eaa33e9fc3b557e5ab45c3dde7ee7153c2ef95a13'
    # The original file has distinct low bits even in some ice records.
    # Keep each variant's own seed rather than borrowing expanded values.
    assert original['VENTI1'].tobytes()!=expanded['VENTI1'].tobytes()
    import json
    registry=json.loads((Path(__file__).resolve().parents[1]/'woof/physics_registry_v2.json').read_text())
    metadata=registry['components']['microphysics']['options']['etanew-mp95']['extensions']
    assert metadata['table']['sha256']==expanded.sha256
    assert metadata['table_variants']['original_rain']['sha256']==original.sha256
