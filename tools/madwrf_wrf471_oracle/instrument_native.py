from pathlib import Path
import sys
root=Path(sys.argv[1])
p=root/'phys/module_madwrf.F';s=p.read_text()
s=s.replace('    implicit none','    implicit none\n    logical, external :: wrf_dm_on_monitor',1)
anchor='      logical, parameter :: LOCAL_DEBUG = .false.'
begin='''
      logical, save :: capture_cloud_done = .false.
      logical :: capture_cloud
      integer :: ci0, ci1, cj0, cj1

      capture_cloud = .not. capture_cloud_done .and. wrf_dm_on_monitor()
      ci0=its; ci1=min(ite,ide-1); cj0=jts; cj1=min(jte,jde-1)
      if (capture_cloud) then
        open(91,file='mad-cloud.in',access='stream',form='unformatted',status='replace')
        write(91) ci1-ci0+1,cj1-cj0+1,kte-kts+1,flag_cldmask,flag_cldtopz,flag_cldbasez,flag_brtemp,em_width,merge(1,0,hold_ups)
        write(91) dx,dy,p00
        write(91) t(ci0:ci1,kts:kte,cj0:cj1),p(ci0:ci1,kts:kte,cj0:cj1)
        write(91) ph_2(ci0:ci1,kts:kte,cj0:cj1),phb(ci0:ci1,kts:kte,cj0:cj1)
        write(91) moist(ci0:ci1,kts:kte,cj0:cj1,p_qv),moist(ci0:ci1,kts:kte,cj0:cj1,p_qc)
        write(91) moist(ci0:ci1,kts:kte,cj0:cj1,p_qi),moist(ci0:ci1,kts:kte,cj0:cj1,p_qs)
        write(91) xland(ci0:ci1,cj0:cj1),ht(ci0:ci1,cj0:cj1),cldmask(ci0:ci1,cj0:cj1)
        write(91) cldtopz(ci0:ci1,cj0:cj1),cldbasez(ci0:ci1,cj0:cj1),brtemp(ci0:ci1,cj0:cj1)
        close(91)
      endif
'''
assert s.count(anchor)>=1;s=s.replace(anchor,anchor+'\n'+begin,1)
end='''
      if (capture_cloud) then
        open(92,file='mad-cloud.out',access='stream',form='unformatted',status='replace')
        write(92) cldfra(ci0:ci1,kts:kte,cj0:cj1)
        write(92) moist(ci0:ci1,kts:kte,cj0:cj1,p_qv),moist(ci0:ci1,kts:kte,cj0:cj1,p_qc)
        write(92) moist(ci0:ci1,kts:kte,cj0:cj1,p_qi),moist(ci0:ci1,kts:kte,cj0:cj1,p_qs)
        write(92) cldmask(ci0:ci1,cj0:cj1),cldtopz(ci0:ci1,cj0:cj1)
        close(92)
        capture_cloud_done=.true.
      endif
'''
s=s.replace('    end subroutine Init_madwrf_clouds',end+'\n    end subroutine Init_madwrf_clouds',1)
p.write_text(s)
p=root/'dyn_em/module_big_step_utilities_em.F';s=p.read_text()
start=s.index('  subroutine Cloud_tracer_nudge ');stop=s.index('  end subroutine Cloud_tracer_nudge',start)
body=s[start:stop]
body=body.replace('    implicit none','    implicit none\n    logical, external :: wrf_dm_on_monitor',1)
body=body.replace('    integer :: i, j, k, i_start, i_end, j_start, j_end, k_start, k_end',
'''    integer :: i, j, k, i_start, i_end, j_start, j_end, k_start, k_end
    logical, save :: capture_nudge_done = .false.
    logical :: capture_nudge''')
entry='''
    capture_nudge = .not. capture_nudge_done .and. wrf_dm_on_monitor()
    if (capture_nudge) then
      open(93,file='mad-nudge.in',access='stream',form='unformatted',status='replace')
      write(93) i_end-i_start+1,j_end-j_start+1,k_end-k_start+1
      write(93) dt,dt_relax,dt_nudge,xtime
      write(93) qc(i_start:i_end,k_start:k_end,j_start:j_end),qi(i_start:i_end,k_start:k_end,j_start:j_end)
      write(93) qs(i_start:i_end,k_start:k_end,j_start:j_end),tr_qc(i_start:i_end,k_start:k_end,j_start:j_end)
      write(93) tr_qi(i_start:i_end,k_start:k_end,j_start:j_end),tr_qs(i_start:i_end,k_start:k_end,j_start:j_end)
      close(93)
    endif
'''
body=body.replace('      ! Nudge',entry+'\n      ! Nudge',1)
body+='''
    if (capture_nudge) then
      open(94,file='mad-nudge.out',access='stream',form='unformatted',status='replace')
      write(94) qc(i_start:i_end,k_start:k_end,j_start:j_end),qi(i_start:i_end,k_start:k_end,j_start:j_end)
      write(94) qs(i_start:i_end,k_start:k_end,j_start:j_end)
      close(94)
      capture_nudge_done=.true.
    endif
'''
s=s[:start]+body+s[stop:];p.write_text(s)
(root/'mad-capture-cloud.patch-note').write_text('Observer only: entry/exit stream writes on monitor, first cloud call. Arithmetic body unchanged.\n')
