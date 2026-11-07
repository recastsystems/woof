"""Bounded, passive native-call captures. Run only on an owned WRF tree."""
from pathlib import Path
import hashlib, json, re

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wrf-source',required=True,type=Path)
parser.add_argument('--schema',required=True,type=Path)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
TREE=args.wrf_source.resolve()
manifest=json.loads((ROOT/'sources.json').read_text())
# Read and verify the release licence before any source modification.
assert hashlib.sha256((TREE/'LICENSE.txt').read_bytes()).hexdigest()==manifest['wrf']['licence_sha256']
for row in manifest['sources']:
 if row['path'].startswith('phys/'):
  assert hashlib.sha256((TREE/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
SURFACE = 'psfc tsk t2 th2 q2 u10 v10 hfx lh swdnb glw lwupb swupb swupt swdnt lwupt lwdnt'.split()
SOLAR_OUT = 'clrnidx sza ghi_accum cldfrac2d wvp2d lwp2d iwp2d swp2d wp2d_sum lwp2d_tot iwp2d_tot wp2d_tot_sum re_cloud_path re_ice_path re_snow_path re_cloud_path_tot re_ice_path_tot tau_qc tau_qi tau_qs tau_qc_tot tau_qi_tot cbase ctop cbase_tot ctop_tot'.split()
SCHEMAS = {}
DECL = '''
      INTEGER, SAVE :: gc_count=0
      INTEGER :: gc_unit,gc_status,gc_i,gc_j,gc_ie,gc_je
      LOGICAL :: gc_write
      CHARACTER(LEN=2048) :: gc_dir,gc_file
'''

def instrument(source, routine, marker, inputs, outputs, scalars=(), clock=False):
    path = TREE / 'phys' / source
    text = path.read_text()
    start = re.search(r'(?i)subroutine\s+'+routine+r'\s*\(',text).start()
    end = text.lower().index('end subroutine '+routine.lower(), start)
    part = text[start:end]
    at = part.index(marker)
    pre = DECL + '''
      CALL GET_ENVIRONMENT_VARIABLE('WOOF_SURFACE_CAPTURE_DIR',gc_dir,STATUS=gc_status)
      gc_write=gc_status==0 .AND. LEN_TRIM(gc_dir)>0 .AND. gc_count<3
      IF(gc_write) THEN
        gc_count=gc_count+1
'''
    if clock:
        pre += '        gc_i=i_start(1); gc_j=j_start(1)\n'
    else:
        pre += '        gc_i=its; gc_j=jts\n'
    pre += '''        gc_ie=MIN(gc_i+3,ide-1); gc_je=MIN(gc_j+3,jde-1)
        WRITE(gc_file,'(A,"/'''+routine+'''-",I6.6,"-",I6.6,"-",I2.2,".bin")') &
          TRIM(gc_dir),gc_i,gc_j,gc_count
        OPEN(NEWUNIT=gc_unit,FILE=TRIM(gc_file),ACCESS='STREAM',FORM='UNFORMATTED', &
          STATUS='NEW',CONVERT='LITTLE_ENDIAN')
        WRITE(gc_unit) 47102,gc_ie-gc_i+1,gc_je-gc_j+1,kme-kms+1,kts,kte,kms,kde
'''
    if source == 'module_diag_rasm.F':
        pre = pre.replace('kts,kte,kms,kde','kps,kpe,kms,kde')
    schema=[]
    def write_rows(rows, label):
        s=''
        for name, dim in rows:
            sel = {'2':'(gc_i:gc_ie,gc_j:gc_je)',
                   '3':'(gc_i:gc_ie,kms:kme,gc_j:gc_je)',
                   'u':'(gc_i:gc_ie+1,kms:kme,gc_j:gc_je)',
                   'v':'(gc_i:gc_ie,kms:kme,gc_j:gc_je+1)',
                   'd':'(gc_i:gc_ie,1:8,gc_j:gc_je)', 's':''}[dim]
            s += f'        WRITE(gc_unit) {name}{sel}\n'
            schema.append({'name':name,'shape':dim,'phase':label})
        return s
    pre += write_rows(scalars, 'scalar') + write_rows(inputs, 'input') + write_rows(outputs, 'before') + '      END IF\n'
    post='\n      IF(gc_write) THEN\n'+write_rows(outputs,'after')+'        CLOSE(gc_unit)\n      END IF\n  '
    text=text[:start]+part[:at]+pre+part[at:]+post+text[end:]
    path.write_text(text)
    SCHEMAS[routine]={'source':source,'rows':schema,'instrumented_sha256':hashlib.sha256(text.encode()).hexdigest()}

instrument('module_diag_solar.F','solar_diag','       DO ij = 1, num_tiles',
    [(s,'3') for s in 'rho dz8w ph phb cldfrac3d qv qc qi qs qc_tot qi_tot re_cloud re_ice re_snow'.split()]+
    [(s,'2') for s in 'coszen swdnb swdnt'.split()],[(s,'2') for s in SOLAR_OUT],
    [(s,'s') for s in 'dt param_first_scalar p_qc p_qi p_qs has_reqc has_reqi has_reqs f_qv f_qc f_qi f_qs'.split()],True)
means=['psfc','tsk','pmsl',*SURFACE[2:]]
instrument('module_diag_rasm.F','mean_output_calc','    IF ( output_freq .eq. MONTHLY)',
    [(s,'2') for s in SURFACE]+[(s,'3') for s in 't p pb moist'.split()]+[('ht','2')],
    [(s+'_mean','2') for s in means]+[('nsteps','s')],
    [('dt','s'),('xtime','s'),('stats_interval','s'),('output_freq','s')],True)
instrument('module_diag_rasm.F','diurnalcycle_output_calc','    ! initialize frequencies and intervals',
    [(s,'2') for s in SURFACE],
    [(s+'_dtmp','2') for s in SURFACE]+[(s+'_diurn','d') for s in SURFACE]+[('avg_nsteps','s'),('diurnal_nsteps','s')],
    [('dt','s'),('xtime','s')],True)
args.schema.write_text(json.dumps(SCHEMAS,indent=2)+'\n')
