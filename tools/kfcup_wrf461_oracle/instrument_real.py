"""Add binary entry/exit capture to an owned copy of pinned native WRF.

The arithmetic is unchanged. Each MPI tile captures 64 actual columns at its
first call. The raw native inputs and reference outputs stay with their manifest; a JSON
receipt does not replace replay evidence.
"""
from pathlib import Path
import hashlib
import sys

PIN = 'd5773c66c9e353b7838bc14e59d66905091425ce449e84cea58f14e7725b074c'
path = Path(sys.argv[1]) / 'phys/module_cu_kfcup.F'
source = path.read_bytes()
assert hashlib.sha256(source).hexdigest() == PIN, 'native source pin moved'
s = source.decode()
declarations = '''
   integer, save :: lane_capture_count = 0
   integer :: lane_unit, lane_col, lane_i, lane_j, lane_flags
   logical :: lane_capture
   character(len=128) :: lane_path
'''
s = s.replace('   DXSQ=DX*DX', declarations + '\n   DXSQ=DX*DX', 1)
level_in = ['u','v','th','t','dz8w','Pcps','pi','rho','qv','z','z_at_w','w0avg']
scalars = ['ht','hfx','qfx','xland','pblh']
held = ['nca','raincv','cutop','cubot','shall','slopesfc','slopeez','sigmasfc',
        'sigmaez','taucloud','tactive','tstar','lnint','activefrac','wcloudbase',
        'wact_cup','wulcl_cup','tcloud_cup']
outputs = ['rthcuten','rqvcuten','rqccuten','rqrcuten','rqicuten','rqscuten',
           'cldfra_cup','wup_cup','qc_ic_cup','qndrop_ic_cup','qc_iu_cup',
           'fcvt_qc_to_pr_cup','fcvt_qc_to_qi_cup','fcvt_qi_to_pr_cup',
           'mfup_cup','mfup_ent_cup','mfdn_cup','mfdn_ent_cup','updfra_cup']

def column_loop(names, surface=False):
    return ''.join(f'      write(lane_unit) {n}(lane_i,lane_j)\n' if surface else
                   f'      write(lane_unit) {n}(lane_i,kts:kte,lane_j)\n' for n in names)
coords = '''
      lane_i=its+mod(lane_col-1,8)*(ite-its)/7
      lane_j=jts+((lane_col-1)/8)*(jte-jts)/7
'''
held_write = column_loop(held, True) + '''      write(lane_unit) merge(1,0,cupflag(lane_i,lane_j)), cu_act_flag(lane_i,lane_j)
''' + column_loop(outputs)
entry = '''
   lane_capture = lane_capture_count == 0 .and. ktau >= 1
   if (lane_capture) then
      lane_capture_count=lane_capture_count+1
      lane_flags=merge(1,0,warm_rain)+merge(2,0,flag_qr)+merge(4,0,flag_qi)+merge(8,0,flag_qs)
      write(lane_path,'(a,i0,a,i0,a)') 'kfcup-real-',its,'-',jts,'.bin'
      open(newunit=lane_unit,file=trim(lane_path),access='stream',form='unformatted',status='replace')
      write(lane_unit) 766721, 1, kte-kts+1, 64, ktau, stepcu, numbins, lane_flags
      write(lane_unit) dt,dx,thbinsize,rbinsize,mindeepfreq,minshallowfreq
      do lane_col=1,64
''' + coords + column_loop(level_in) + column_loop(scalars, True) + '''      write(lane_unit) kpbl(lane_i,lane_j)
      write(lane_unit) w(lane_i,kts:kte+1,lane_j)
''' + held_write + '''      end do
   end if
'''
s=s.replace('   idiagff = 0 ; idiagee = 0',entry+'\n   idiagff = 0 ; idiagee = 0',1)
exit = '''   if (lane_capture) then
      do lane_col=1,64
''' + coords + column_loop(['w0avg']) + held_write + '''      end do
      close(lane_unit)
   end if
'''
s=s.replace('   END SUBROUTINE KF_cup_CPS',exit+'\n   END SUBROUTINE KF_cup_CPS',1)
path.write_text(s)
print(path, 'instrumented; pristine SHA256', PIN)
