"""Stock output controls reach the amount provider without changing off bytes."""
from pathlib import Path
import importlib.util,tomllib
import pytest
from conftest import requires_gpu
from woof.fortran_namelist import parse_namelist_text
from woof.namelist_import import import_parsed_namelists
from woof.core.noahmp_output import ACCUMULATORS,CARRIERS,shapes
from woof.core.noahmp_mynn_contract import NOAHMP_PACKAGE_STATE_FIELDS
from test_restart import _cfg


def test_native_amount_fields_are_package_carriers_without_an_extra_clock():
 assert len(ACCUMULATORS)==58
 assert {CARRIERS[n] for n in ACCUMULATORS} <= set(NOAHMP_PACKAGE_STATE_FIELDS)
 assert not shapes(1,(3,4))
 assert not shapes(2,(3,4))
 assert shapes(3,(3,4))['acsagv']==(3,4)
 # The owner's update uses the domain-relative atmospheric clock directly.
 import inspect
 from woof.core import noahmp_output
 assert 'domain_start_offset' in inspect.getsource(noahmp_output.update)
 import json
 registry=json.loads((Path(__file__).parents[1]/'woof/physics_registry_v2.json').read_text())
 declared=registry['diagnostic_state']
 assert 'noahmp_output_clock' not in declared
 assert {CARRIERS[n] for n in ACCUMULATORS} <= declared['noahmp_output_fields'].keys()


def test_stock_noahmp_output_and_interval_survive_import_and_default_bytes():
 spec=importlib.util.spec_from_file_location('noahmp_diag_census',Path(__file__).parents[1]/'tools/diag_wrf471_oracle/census.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);fixture,_=m.fixture()
 def read(extra):
  inp=parse_namelist_text(fixture['INPUT_TEXT']);inp['noah_mp']=extra
  return import_parsed_namelists(parse_namelist_text(fixture['WPS_TEXT']),inp,wps_path='namelist.wps',input_path='namelist.input',name='noahmp-diagnostic',source_text=None)[0]
 original=read({});defaults=read({'noahmp_output':[1],'noahmp_acc_dt':[0.]})
 assert original==defaults
 shared=tomllib.loads(read({'noahmp_output':[3],'noahmp_acc_dt':[60.]}))['shared']
 assert shared['noahmp_output']==3 and shared['noahmp_acc_dt']==60.


@pytest.mark.gpu
@requires_gpu
def test_actual_allocation_covers_kernel_inputs_and_amounts_without_off_changes():
 import numpy as np
 from test_noahmp_runtime import _build
 from woof.core.noahmp_output import INPUTS
 from woof.io.wrf_output_schema import NOAHMP_OUTPUT_FIELDS
 from woof.ingest.wrfrst import fields_restore_plan
 _,cfg,driver=_build(noahmp_output=3,noahmp_acc_dt=60.)
 _,_,off=_build()
 required={CARRIERS[n] for n in (*INPUTS,*ACCUMULATORS)}
 assert required<=driver.fields.keys()
 for name,shape in shapes(3,(cfg.ny,cfg.nx)).items():
  assert driver.fields[name].shape==shape and driver.fields[name].dtype==np.float32
  assert name not in off.fields
  if name == 'swnorm':
   # This radiation input is not a Noah-MP history variable. WRF rewrites
   # it at the next radiation call; its explicit absent rule starts at zero.
   rule=fields_restore_plan(name,set(),cfg)
   assert rule[:2]==('cold',0.0) and 'SWNORM' in rule[2]
  else:
   external=NOAHMP_OUTPUT_FIELDS[name].netcdf_name
   assert fields_restore_plan(name,{external},cfg)==('fill',external)


def test_registry_flags_and_external_names_ignore_interpolation_operators():
 import json
 from woof.io.wrf_output_schema import NOAHMP_OUTPUT_FIELDS
 registry=json.loads((Path(__file__).parents[1]/'woof/physics_registry_v2.json').read_text())
 fields=registry['diagnostic_state']['noahmp_output_fields']
 # Native registry stream flags are separate from its interpolation names.
 # In particular interp_fcn:m=... contains m but does not change h/r flags.
 assert NOAHMP_OUTPUT_FIELDS['aceflxb'].netcdf_name=='ACEFLXB'
 assert fields['qintsxy']['restart'] and fields['qintsxy']['history']
 assert fields['acsagv']['restart'] and fields['acsagv']['history']
 assert NOAHMP_OUTPUT_FIELDS['qintsxy'].netcdf_name=='QINTS'


def test_native_restart_alias_uses_the_selected_noahmp_schema():
 from dataclasses import replace
 from woof.ingest.wrfrst import fields_restore_plan
 cfg=replace(_cfg(),sf_surface_physics=4)
 assert fields_restore_plan('sagxy',{'SAG'},cfg)==('fill','SAG')
 assert fields_restore_plan('aceflxb',{'ACEFLXB'},cfg)==('fill','ACEFLXB')
 assert fields_restore_plan('sagxy',{'SAG'},replace(cfg,sf_surface_physics=2)) is None
 assert fields_restore_plan('sagxy',set(),cfg) is None
