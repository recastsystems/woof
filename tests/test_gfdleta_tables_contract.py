from pathlib import Path
import shutil
import pytest
from woof.table_assets import physics_table_root,TableAssetError

def test_packaged_tables_are_byte_pinned():
 root=physics_table_root('gfdleta')
 assert (root/'co2_trans').stat().st_size>10000

def test_one_bit_transmission_mutation_is_refused(tmp_path):
 root=physics_table_root('gfdleta')
 for p in root.iterdir():shutil.copyfile(p,tmp_path/p.name)
 p=tmp_path/'co2-raw.npy';data=bytearray(p.read_bytes());data[128+4*27]^=1;p.write_bytes(data)
 with pytest.raises(TableAssetError,match='corrupted radiation coefficients'):physics_table_root('gfdleta',tmp_path)


def test_override_cannot_redefine_the_coefficient_pin(tmp_path):
 import hashlib,json
 root=physics_table_root('gfdleta')
 for p in root.iterdir():shutil.copyfile(p,tmp_path/p.name)
 p=tmp_path/'co2-raw.npy';data=bytearray(p.read_bytes());data[160]^=1;p.write_bytes(data)
 manifest=tmp_path/'MANIFEST.json';pin=json.loads(manifest.read_text())
 pin['assets'][p.name]['sha256']=hashlib.sha256(data).hexdigest()
 manifest.write_text(json.dumps(pin))
 with pytest.raises(TableAssetError,match='manifest differs from the package pin'):
  physics_table_root('gfdleta',tmp_path)


def test_stock_namelist_table_admission_checks_gfdl_assets(monkeypatch,tmp_path):
 from woof import tables
 cfg={'ra_lw_physics':99,'ra_sw_physics':99}
 assert tables.status(cfg).ready
 monkeypatch.setenv('WOOF_TABLE_ROOT',str(tmp_path))
 report=tables.status(cfg)
 assert not report.ready
 assert {p.scheme for p in report.missing}=={'gfdleta'}
 assert 'co2-raw.npy' in {p.path.name for p in report.missing}


def test_gfdl_preflight_prices_every_production_module():
 from woof.config import RunConfig
 from woof.experiment import DomainConfig
 from woof.core.preflight import domain_kernel_modules
 cfg=RunConfig(nx=24,ny=24,nz=16,dx=1000,dy=1000,ztop=8000,dt=1,run_seconds=10,ra_lw_physics=99,ra_sw_physics=99)
 domain=DomainConfig(grid_id=1,parent_id=0,i_parent_start=1,j_parent_start=1,
                     parent_grid_ratio=1,parent_time_step_ratio=1,
                     history_interval_s=3600,run=cfg,time_step=cfg.dt)
 assert {'gfdleta_column','gfdleta_co2_init','gfdleta_init',
         'gfdleta_table_runtime','gfdleta_tables'} <= domain_kernel_modules(domain,prices_refl=False)
