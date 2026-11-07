from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text();native=s
anchor='      save time_tke,hour'
s=s.replace(anchor,anchor+'''
      logical, save :: lane_recorded = .false.
      logical :: lane_capture
      integer :: lane_in, lane_rec, lane_i, lane_n, lane_nz, lane_flag
      character(len=128) :: lane_name
''',1)
start='!here I fix the value of the reference state'
inp=['dz8w','u_phy','v_phy','th_phy','qv_curr','qc_curr','rho','tke']
state=['u_phy','v_phy','th_phy','qv_curr','qc_curr','tke','dlg_bep']
bep=['a_u_bep','a_v_bep','a_t_bep','a_q_bep','a_e_bep','b_u_bep','b_v_bep','b_t_bep','b_q_bep','b_e_bep','dlg_bep','dl_u_bep','vl_bep']
out=['rublten','rvblten','rthblten','rqvblten','rqcblten','tke','dlk','exch_h','exch_m','wu','wv','wt','wq','dlg_bep']
def arr(a,unit,face=False):return f'        write({unit}) {a}(its:its+lane_n-1,kts:kte'+('+1' if face else '')+',jts)\n'
def surf(a,unit):return f'        write({unit}) {a}(its:its+lane_n-1,jts)\n'
cap='''! Lane-only tap: stream the first real module invocation on each MPI tile.
      lane_capture = .not. lane_recorded
      if (lane_capture) then
        lane_n = min(32,ite-its+1)
        lane_nz = kte-kts+1
        lane_flag = 0
        if (flag_bep) lane_flag = 1
        write(lane_name,'("boulac-",I0,"-",I0)') its,jts
        open(newunit=lane_in,file=trim(lane_name)//'.in.bin', &
          access='stream',form='unformatted',status='new',convert='little_endian')
        open(newunit=lane_rec,file=trim(lane_name)//'.rec.bin', &
          access='stream',form='unformatted',status='new',convert='little_endian')
        write(lane_in) 11579816,lane_nz,lane_n,1,lane_flag,idiff
        write(lane_in) dt,cp,g
        write(lane_rec) 11579816,lane_nz,lane_n,1,lane_flag,idiff
        write(lane_rec) dt,cp,g
'''
cap+=''.join(arr(a,'lane_in') for a in inp)
cap+=''.join(surf(a,'lane_in') for a in ['hfx','qfx','ustar','frc_urb2d','pblh'])
cap+=''.join(arr(a,'lane_in') for a in bep)+arr('sf_bep','lane_in',True)
cap+='        close(lane_in)\n'
cap+=''.join(arr(a,'lane_rec') for a in state)+surf('pblh','lane_rec')+'      endif\n\n'
s=s.replace(start,cap+start,1)
end='''      if (lane_capture) then
'''+''.join(arr(a,'lane_rec') for a in out)+surf('pblh','lane_rec')+'''        close(lane_rec)
        lane_recorded = .true.
      endif
'''
s=s.replace('      return\n      end subroutine boulac',end+'      return\n      end subroutine boulac',1)
assert s!=native
p.write_text(s)
