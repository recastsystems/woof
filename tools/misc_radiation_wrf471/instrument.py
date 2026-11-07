"""Retain the real WRF 4.7.1 inputs and exits around both radiation arms."""
from pathlib import Path
import json
import sys

source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
bounds = 'ids ide jds jde kds kde ims ime jms jme kms kme its ite jts jte kts kte'.split()
integers = bounds + 'itimestep lw_physics sw_physics yr julday icloud o3input ghg_input aer_opt mp_physics no_src_types has_reqc has_reqi has_reqs'.split()
reals = 'r_d g solcon xtime gmt radt degrad declin julian'.split()
logical = 'warm_rain is_cammgmp_used'.split()
optional_logical = 'f_qnc f_qv f_qc f_qr f_qi f_qs f_qg f_qndrop'.split()
mass = 't rho p pi dz8w re_cloud re_ice re_snow o3rad'.split()
interface = 't8w p8w'.split()
optional_mass = 'qv qc qr qi qs qg qnc_curr qndrop cldfra f_ice_phy f_rain_phy lradius iradius'.split()
surface = 'albedo emiss tsk xland xice snow xlat xlong coszr'.split()
optional_surface = 'coszen alswvisdir alswvisdif alswnirdir alswnirdif'.split()
out_mass = 'rthraten rthratenlw rthratensw rthratenlwc rthratenswc'.split()
out_interface = 'lwupflx lwupflxc lwdnflx lwdnflxc swupflx swupflxc swdnflx swdnflxc'.split()
out_surface = 'gsw glw swdown rlwtoa rswtoa swddir swddni swddif'.split()
out_optional_surface = 'lwupt lwuptc lwdnt lwdntc lwupb lwupbc lwdnb lwdnbc olr lwcf swupt swuptc swdnt swdntc swupb swupbc swdnb swdnbc swcf swvisdir swvisdif swnirdir swnirdif swdownc swddnic swddirc'.split()
schema = dict(header_i4=integers, header_f4=reals, logical=logical,
              optional_logical=optional_logical)
groups = dict(mass=mass, interface=interface, optional_mass=optional_mass,
              surface=surface, optional_surface=optional_surface,
              out_mass=out_mass, out_interface=out_interface,
              out_surface=out_surface, out_optional_surface=out_optional_surface)
schema.update(groups)
def write_array(name, section, optional=False):
    s = f'   WRITE(mrad_unit) {name}({section})\n'
    if optional:
        s = f'   WRITE(mrad_unit) MERGE(1,0,PRESENT({name}))\n   IF(PRESENT({name})) THEN\n' + s + '   END IF\n'
    return s
def stream(suffix):
    s = f'''   IF(mrad_capture) THEN
   WRITE(mrad_file,'(A,A,I4.4,A,I6.6,A,I6.6,A)') &
      TRIM(mrad_dir),'/rad-s',mrad_call,'-i',its,'-j',jts,'{suffix}'
   OPEN(NEWUNIT=mrad_unit,FILE=TRIM(mrad_file),ACCESS='STREAM', &
      FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
'''
    if suffix == '.input':
        s += '   WRITE(mrad_unit) ' + ','.join(integers) + '\n'
        s += '   WRITE(mrad_unit) ' + ','.join(reals) + '\n'
        for n in logical:
            s += f'   WRITE(mrad_unit) MERGE(1,0,{n})\n'
        for n in optional_logical:
            s += f'   WRITE(mrad_unit) MERGE(1,0,PRESENT({n}))\n   IF(PRESENT({n})) WRITE(mrad_unit) MERGE(1,0,{n})\n'
        for n in mass: s += write_array(n,'its:ite,kts:kte,jts:jte')
        for n in interface: s += write_array(n,'its:ite,kts:kte+1,jts:jte')
        for n in optional_mass: s += write_array(n,'its:ite,kts:kte,jts:jte', True)
        for n in surface: s += write_array(n,'its:ite,jts:jte')
        for n in optional_surface: s += write_array(n,'its:ite,jts:jte', True)
        s += write_array('aerod','its:ite,kts:kte,jts:jte,1:no_src_types', True)
    for n in out_mass: s += write_array(n,'its:ite,kts:kte,jts:jte')
    for n in out_interface: s += write_array(n,'its:ite,kts:kte+1,jts:jte', True)
    for n in out_surface: s += write_array(n,'its:ite,jts:jte')
    for n in out_optional_surface: s += write_array(n,'its:ite,jts:jte', True)
    return s + '   CLOSE(mrad_unit)\n   END IF\n'
s = source.read_text()
marker = '   logical, save :: feedback_restart, direct_sw_feedback'
assert s.count(marker) == 1
s = s.replace(marker, marker + '''
   INTEGER :: mrad_unit
   INTEGER,SAVE :: mrad_call=0
   LOGICAL :: mrad_capture
   CHARACTER(LEN=512) :: mrad_dir,mrad_file
''')
marker = '     lwrad_select: SELECT CASE(lw_physics)'
assert s.count(marker) == 1
s = s.replace(marker, '''   CALL GET_ENVIRONMENT_VARIABLE('MISC_RAD_CAPTURE_DIR',mrad_dir)
   mrad_call=mrad_call+1
   mrad_capture=LEN_TRIM(mrad_dir)>0 &
     .AND.(mrad_call==1.OR.mrad_call==2.OR.mrad_call==4.OR.mrad_call==8)
''' + stream('.input') + marker)
marker = '     END SELECT swrad_select'
assert s.count(marker) == 1
s = s.replace(marker, marker + '\n' + stream('.output'))
(out / source.name).write_text(s)
(out / 'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
