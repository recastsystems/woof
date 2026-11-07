"""Add diagnostic streams to a private, pinned native WRF tree."""
from pathlib import Path
import argparse,hashlib
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('wrf',type=Path)
parser.add_argument('--source',type=Path,help='pristine pinned file; defaults to the private tree source')
parser.add_argument('--ktau',type=int,default=4)
args=parser.parse_args()
if args.ktau < 1: parser.error('--ktau must be positive')
root=args.wrf; p=root/'phys/module_fddaobs_rtfdda.F'; src=args.source or p
s=src.read_text(); assert hashlib.sha256(src.read_bytes()).hexdigest()=='1ca4065b3aba0f926a7ded65f9eb92e1c6a7da678401271c9441bc93dda946c0'
a=s.index('SUBROUTINE errob('); b=s.index('END SUBROUTINE errob',a)
e=s[a:b]
e=e.replace('  IMPLICIT NONE','''  IMPLICIT NONE
  integer :: lane_u
  logical, save :: lane_done=.false.
  character(len=128) :: lane_file
''',1)
e=e.replace('  NSTA=NSTAT','''  if (.not.lane_done.and.ktau.eq.4) then
    write(lane_file,'(a,i4.4,a,i4.4,a)') 'lane-errob-',its,'-',jts,'-in.bin'
    open(newunit=lane_u,file=trim(lane_file),access='stream',form='unformatted',status='replace',convert='little_endian')
    write(lane_u) ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
    write(lane_u) nstat,nndgv,nerrf,iswind,istemp,ismois,inest
    write(lane_u) t0,rcp
    write(lane_u) ub,vb,tb,qvb,pbase,pp,z,uratx,vratx,tratx,kpbl
    write(lane_u) rio(:nstat),rjo(:nstat),rko(:nstat),varobs(:,:nstat)
    close(lane_u)
  endif
  NSTA=NSTAT''',1)
needle='  CALL get_full_obs_vector'
e=e.replace(needle,'''  if (.not.lane_done.and.ktau.eq.4) then
    write(lane_file,'(a,i4.4,a,i4.4,a)') 'lane-errob-',its,'-',jts,'-out.bin'
    open(newunit=lane_u,file=trim(lane_file),access='stream',form='unformatted',status='replace',convert='little_endian')
    write(lane_u) rko(:nstat),varobs(:,:nstat),errf(:,:nstat)
    close(lane_u)
    lane_done=.true.
  endif
'''+needle,1)
s=s[:a]+e+s[b:]
a=s.index('SUBROUTINE nudob(');b=s.index('END SUBROUTINE nudob',a)
e=s[a:b]
e=e.replace('  IMPLICIT NONE','''  IMPLICIT NONE
  integer :: lane_u
  logical :: lane_capture
  logical, save :: lane_done(4)=.false.
  character(len=128) :: lane_file
''',1)
e=e.replace('  if(NSCAN.ne.0) then','''  lane_capture=.not.lane_done(ivar).and.ktau.eq.4.and.j.eq.(jts+jte)/2
  if(lane_capture) then
    write(lane_file,'(a,i4.4,a,i4.4,a,i1,a)') 'lane-nudob-',its,'-',j,'-',ivar,'-in.bin'
    open(newunit=lane_u,file=trim(lane_file),access='stream',form='unformatted',status='replace',convert='little_endian')
    write(lane_u) ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte
    write(lane_u) nstat,nndgv,nerrf,ivar,inest,ktau,ktaur,merge(1,0,ifrest),ionf*(iratio**levidn(inest)), &
       merge(1,0,nudge_pbl),sfc_scheme_horiz,sfc_scheme_vert,obs_scl_neg_qv_innov,j
    write(lane_u) xtime,dtmin,rinxy,dx,twindo,sfcfact,sfcfacr,rinfmn,rinfmx,pfree,dcon,tfaci,giv,git,giq,maxsnd_gap
    write(lane_u) aten,savwt,mu,c1h,c2h,msfx,msfy
    write(lane_u) rio(:nstat),rjo(:nstat),rko(:nstat),timeob(:nstat),varobs(:,:nstat),errf(:,:nstat), &
       nlevs_ob(:nstat),lev_in_ob(:nstat)
    write(lane_u) pbase,pp,kpblt,vih1,vih2,terrh,zslab,qvb(:,:,j)
    close(lane_u)
  endif
  if(NSCAN.ne.0) then''',1)
cap='''  if(lane_capture) then
    write(lane_file,'(a,i4.4,a,i4.4,a,i1,a)') 'lane-nudob-',its,'-',j,'-',ivar,'-out.bin'
    open(newunit=lane_u,file=trim(lane_file),access='stream',form='unformatted',status='replace',convert='little_endian')
    write(lane_u) aten,savwt
    close(lane_u)
    lane_done(ivar)=.true.
  endif
'''
# two native return paths after output, plus early boundary returns untouched.
e=e.replace('  RETURN\n\n170 CONTINUE',cap+'  RETURN\n\n170 CONTINUE',1)
e=e.rstrip()+'\n' # final RETURN native is uppercase
idx=e.rfind('  RETURN');e=e[:idx]+cap+e[idx:]
s=s[:a]+e+s[b:];s=s.replace('ktau.eq.4',f'ktau.eq.{args.ktau}');p.write_text(s)
print(p)
