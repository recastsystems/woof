"""Public route and gate regression tests; no GPU or shared-data mutation."""
import copy,json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from woof.hex import physics_default
from woof.hex.physics_suites import load_table
from woof.hex.mpasphys.forecast_v842 import native_route
from woof.hex.mpasphys.resident_v842 import resident_requested, ported_route_refusal, TARGET
from woof.hex.mpasphys.profiles_v842 import finish_profile,resolve_profile

DIMS=dict(nCells=2,nCellsSolve=2,nEdges=3,nEdgesSolve=3)
def resolve(**kw):return native_route(SimpleNamespace(**kw),default_mode='correct')

class Profiles(unittest.TestCase):
 def test_default_unchanged(self):
  args=SimpleNamespace()
  physics_default.apply(args)
  c,t=resolve(**vars(args))
  self.assertEqual(c.selectors['microphysics'],'mp_thompson')
  self.assertNotIn('config_hex_icloud_bl',c.options)
  self.assertNotIn('hex_physics_profile',c.receipt()['options'])
  self.assertTrue(resident_requested(c,DIMS))
 def test_rows_and_override_precedence(self):
  a,_=resolve(physics_profile='hrrr-like-aerosol')
  b,t=resolve(physics_profile='hrrr-like')
  self.assertEqual(a.options['config_mynn_closure'],2.5)
  self.assertEqual((b.options['config_mynn_mixlength'],b.options['config_mynn_closure'],b.options['config_mynn_edmf_mom']),(1,2.6,1))
  for c in (a,b):
   self.assertTrue(c.options['config_hex_icloud_bl'])
   self.assertEqual(c.selectors['microphysics'],'mp_thompson_aerosols')
   self.assertTrue(resident_requested(c,DIMS))
   self.assertIsNone(ported_route_refusal(c))
  self.assertNotIn('hex_physics_profile',t)
  c,_=resolve(physics_profile='hrrr-like',physics=['config_mynn_mixlength=2'])
  self.assertEqual(c.options['config_mynn_mixlength'],2)
  self.assertEqual(TARGET['microphysics'],'mp_thompson')
 def test_guardrails(self):
  for opts in [dict(physics_profile='missing'),dict(physics_profile='hrrr-like',physics_mode='exact'),dict(physics_profile='hrrr-like',physics_suite='hrrrv5'),dict(physics_profile='hrrr-like',physics=['config_pbl_scheme=bl_ysu']),dict(physics_profile='hrrr-like',physics=['nonexistent_option=1']),dict(physics_profile='hrrr-like',physics_default_backend='native-cpu')]:
   with self.subTest(opts=opts),self.assertRaises(ValueError):resolve(**opts)
 def test_prepared_input_required(self):
  c,_=resolve(physics_profile='hrrr-like')
  row=resolve_profile('hrrr-like','correct')
  fake=SimpleNamespace(variables={'qv':None})
  with patch('netCDF4.Dataset') as ds:
   ds.return_value.__enter__.return_value=fake
   with self.assertRaisesRegex(ValueError,'nifa'):finish_profile('hrrr-like',row,c,Path('init.nc'))
   fake.variables.update(dict(nc=None,nifa=None,nwfa=None))
   self.assertEqual(finish_profile('hrrr-like',row,c,Path('init.nc')).selectors,c.selectors)
 def test_arbitrary_table_row(self):
  table=json.loads((Path(__file__).parents[1]/'../../woof/hex/data/profiles/mpas-physics-suites.v1.json').read_text())
  table['experimental_profiles']['new-row']=copy.deepcopy(table['experimental_profiles']['hrrr-like'])
  table['experimental_profiles']['new-row']['options']['config_mynn_mixlength']=2
  with patch('woof.hex.mpasphys.profiles_v842.load_table',return_value=table):
   c,_=resolve(physics_profile='new-row')
   self.assertEqual(c.options['config_mynn_mixlength'],2)
   self.assertTrue(resident_requested(c,DIMS))
 def test_profile_counts_as_explicit_physics(self):
  args=SimpleNamespace(physics_profile='hrrr-like')
  self.assertIsNone(physics_default.apply(args))
  self.assertEqual(physics_default.ensure(['--physics-profile=hrrr-like']),['--physics-profile=hrrr-like'])

if __name__=='__main__':unittest.main()
