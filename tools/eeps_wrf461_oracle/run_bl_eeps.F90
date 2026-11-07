! Column oracle for WRF v4.6.1 phys/module_bl_eepsilon.F (bl_pbl_physics=16).
!
! Reads inputs.bin from make_columns.py, runs EEPSINIT (cold arm) and then
! NSTEPS calls of the real EEPS driver on one tile of NCOL columns, and
! writes trace.bin: for every step the full input set the call saw and the
! full output set it produced.  Between steps the oracle advances the state
! with the tendencies it just got and applies a fixed vertical stand-in for
! scalar advection to PEK_ADV/PEP_ADV, so every step after the first reads
! advected TKE/epsilon that is NOT the copy EEPS wrote: that is the handoff
! the GPU replay must reproduce from its own outputs.
!
! Layout: every 3-D record is field(1:ncol, 1:nz, 1) in Fortran order (i
! fastest), i.e. numpy (nz, ncol); EXCH_H/EXCH_M are (nz+1, ncol); 2-D
! records are (ncol); KPBL is int32.
program run_bl_eeps
  use module_bl_eeps, only: eeps, eepsinit
  use module_model_constants, only: r_d, r_v, prandtl, ep_1, ep_2, g, &
       karman, cp, rcp, xlv, xls, xlf, p1000mb
  implicit none
  integer :: ncol, nz, nsteps, it0, has_qsqg, step, i, k, uin, uout, ucon
  integer :: itimestep
  real :: dt
  real, allocatable, dimension(:,:,:) :: u3d, v3d, t3d, qv3d, qc3d, qi3d, &
       qr3d, qs3d, qg3d, p3d, pi3d, rho3d, rthraten, dz8w
  real, allocatable, dimension(:,:,:) :: rublten, rvblten, rthblten, &
       rqvblten, rqcblten, rqiblten, pek, pep, pek_adv, pep_adv, exch_h, exch_m
  real, allocatable, dimension(:,:) :: psfc, qsfc, tsk, ust, rmol, wspd, &
       xland, hfx, qfx, pblh, tmp, work
  integer, allocatable :: kpbl(:,:)
  character(len=1024) :: fin, fout, fcon
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, fcon)

  open(newunit=ucon, file=trim(fcon), status='replace', action='write')
  write(ucon, '(A,Z8.8)') 'r_d     ', transfer(r_d, 0)
  write(ucon, '(A,Z8.8)') 'r_v     ', transfer(r_v, 0)
  write(ucon, '(A,Z8.8)') 'prandtl ', transfer(prandtl, 0)
  write(ucon, '(A,Z8.8)') 'ep_1    ', transfer(ep_1, 0)
  write(ucon, '(A,Z8.8)') 'ep_2    ', transfer(ep_2, 0)
  write(ucon, '(A,Z8.8)') 'g       ', transfer(g, 0)
  write(ucon, '(A,Z8.8)') 'karman  ', transfer(karman, 0)
  write(ucon, '(A,Z8.8)') 'cp      ', transfer(cp, 0)
  write(ucon, '(A,Z8.8)') 'rcp     ', transfer(rcp, 0)
  write(ucon, '(A,Z8.8)') 'xlv     ', transfer(xlv, 0)
  write(ucon, '(A,Z8.8)') 'xls     ', transfer(xls, 0)
  write(ucon, '(A,Z8.8)') 'xlf     ', transfer(xlf, 0)
  write(ucon, '(A,Z8.8)') 'p1000mb ', transfer(p1000mb, 0)
  close(ucon)

  open(newunit=uin, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  read(uin) ncol, nz, nsteps, it0, has_qsqg
  read(uin) dt

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1; kte = nz

  allocate(u3d(ims:ime,kms:kme,jms:jme), v3d(ims:ime,kms:kme,jms:jme), &
       t3d(ims:ime,kms:kme,jms:jme), qv3d(ims:ime,kms:kme,jms:jme), &
       qc3d(ims:ime,kms:kme,jms:jme), qi3d(ims:ime,kms:kme,jms:jme), &
       qr3d(ims:ime,kms:kme,jms:jme), qs3d(ims:ime,kms:kme,jms:jme), &
       qg3d(ims:ime,kms:kme,jms:jme), p3d(ims:ime,kms:kme,jms:jme), &
       pi3d(ims:ime,kms:kme,jms:jme), rho3d(ims:ime,kms:kme,jms:jme), &
       rthraten(ims:ime,kms:kme,jms:jme), dz8w(ims:ime,kms:kme,jms:jme))
  allocate(rublten(ims:ime,kms:kme,jms:jme), rvblten(ims:ime,kms:kme,jms:jme), &
       rthblten(ims:ime,kms:kme,jms:jme), rqvblten(ims:ime,kms:kme,jms:jme), &
       rqcblten(ims:ime,kms:kme,jms:jme), rqiblten(ims:ime,kms:kme,jms:jme), &
       pek(ims:ime,kms:kme,jms:jme), pep(ims:ime,kms:kme,jms:jme), &
       pek_adv(ims:ime,kms:kme,jms:jme), pep_adv(ims:ime,kms:kme,jms:jme), &
       exch_h(ims:ime,kms:kme,jms:jme), exch_m(ims:ime,kms:kme,jms:jme))
  allocate(psfc(ims:ime,jms:jme), qsfc(ims:ime,jms:jme), tsk(ims:ime,jms:jme), &
       ust(ims:ime,jms:jme), rmol(ims:ime,jms:jme), wspd(ims:ime,jms:jme), &
       xland(ims:ime,jms:jme), hfx(ims:ime,jms:jme), qfx(ims:ime,jms:jme), &
       pblh(ims:ime,jms:jme), kpbl(ims:ime,jms:jme))
  allocate(tmp(ncol, nz), work(ncol, nz))

  call read3(u3d); call read3(v3d); call read3(t3d); call read3(qv3d)
  call read3(qc3d); call read3(qi3d); call read3(qr3d); call read3(qs3d)
  call read3(qg3d); call read3(p3d); call read3(pi3d); call read3(rho3d)
  call read3(rthraten); call read3(dz8w)
  call read2(psfc); call read2(qsfc); call read2(tsk); call read2(ust)
  call read2(rmol); call read2(wspd); call read2(xland); call read2(hfx)
  call read2(qfx)
  ! Starting carried state; the cold arm overwrites it below.
  call read3(pek); call read3(pep); call read3(pek_adv); call read3(pep_adv)
  close(uin)

  rublten = 0.; rvblten = 0.; rthblten = 0.; rqvblten = 0.
  rqcblten = 0.; rqiblten = 0.; exch_h = 0.; exch_m = 0.
  if (it0 == 1) then
     ! WRF's cold start: real.exe's zero-initialised scalar array (the
     ! Registry 'i0' flag on pek_adv/pep_adv) and EEPSINIT's minpek/minpep.
     pek_adv = 0.; pep_adv = 0.
     call eepsinit(rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, &
          2, 1, pek, pep, .false., .true., &
          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
          its, ite, jts, jte, kts, kte)
  end if

  open(newunit=uout, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write')
  write(uout) ncol, nz, nsteps, it0, has_qsqg
  write(uout) dt

  do step = 1, nsteps
     itimestep = it0 + step - 1
     ! ---- inputs this call sees
     call write3(u3d); call write3(v3d); call write3(t3d); call write3(qv3d)
     call write3(qc3d); call write3(qi3d); call write3(qr3d); call write3(qs3d)
     call write3(qg3d); call write3(p3d); call write3(pi3d); call write3(rho3d)
     call write3(rthraten); call write3(dz8w)
     call write2(psfc); call write2(qsfc); call write2(tsk); call write2(ust)
     call write2(rmol); call write2(wspd); call write2(xland); call write2(hfx)
     call write2(qfx)
     call write3(pek); call write3(pep); call write3(pek_adv); call write3(pep_adv)
     write(uout) itimestep

     if (has_qsqg == 1) then
        call eeps(u3d, v3d, t3d, qv3d, qc3d, qi3d, qr3d, qs3d, qg3d, p3d, pi3d, &
             rho3d, rthraten, rublten, rvblten, rthblten, &
             rqvblten, rqcblten, rqiblten, pek, pep, &
             dz8w, psfc, qsfc, tsk, ust, rmol, wspd, xland, hfx, qfx, &
             dt, 3000., itimestep, exch_h, exch_m, pblh, kpbl, &
             pek_adv, pep_adv, &
             ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
             its, ite, jts, jte, kts, kte)
     else
        call eeps(u3d, v3d, t3d, qv3d, qc3d, qi3d, qr3d=qr3d, p3d=p3d, &
             pi3d=pi3d, rho3d=rho3d, rthraten=rthraten, rublten=rublten, &
             rvblten=rvblten, rthblten=rthblten, rqvblten=rqvblten, &
             rqcblten=rqcblten, rqiblten=rqiblten, pek=pek, pep=pep, &
             dz8w=dz8w, psfc=psfc, qsfc=qsfc, tsk=tsk, ust=ust, rmol=rmol, &
             wspd=wspd, xland=xland, hfx=hfx, qfx=qfx, dt=dt, dx=3000., &
             itimestep=itimestep, exch_h=exch_h, exch_m=exch_m, pblh=pblh, &
             kpbl=kpbl, pek_adv=pek_adv, pep_adv=pep_adv, &
             ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
             ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
             its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)
     end if

     ! ---- outputs
     call write3(rublten); call write3(rvblten); call write3(rthblten)
     call write3(rqvblten); call write3(rqcblten); call write3(rqiblten)
     call write3(pek); call write3(pep); call write3(pek_adv); call write3(pep_adv)
     call write3f(exch_h); call write3f(exch_m)
     call write2(pblh)
     write(uout) kpbl(ims:ime, 1)

     ! ---- advance the state with the tendencies just produced
     do k = kts, kte
        do i = its, ite
           u3d(i,k,1) = u3d(i,k,1) + dt*rublten(i,k,1)
           v3d(i,k,1) = v3d(i,k,1) + dt*rvblten(i,k,1)
           t3d(i,k,1) = (t3d(i,k,1)/pi3d(i,k,1) + dt*rthblten(i,k,1))*pi3d(i,k,1)
           qv3d(i,k,1) = max(0., qv3d(i,k,1) + dt*rqvblten(i,k,1))
           qc3d(i,k,1) = max(0., qc3d(i,k,1) + dt*rqcblten(i,k,1))
           qi3d(i,k,1) = max(0., qi3d(i,k,1) + dt*rqiblten(i,k,1))
        end do
     end do
     ! ---- stand-in for scalar advection of the carried pair
     call blend(pek_adv); call blend(pep_adv)
  end do
  close(uout)

contains

  subroutine read3(a)
    real, intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    read(uin) tmp
    a(ims:ime, kts:kte, 1) = tmp
    a(ims:ime, kte+1, 1) = tmp(:, nz)
  end subroutine read3

  subroutine read2(a)
    real, intent(inout) :: a(ims:ime,jms:jme)
    read(uin) a(ims:ime, 1)
  end subroutine read2

  subroutine write3(a)
    real, intent(in) :: a(ims:ime,kms:kme,jms:jme)
    write(uout) a(ims:ime, kts:kte, 1)
  end subroutine write3

  subroutine write3f(a)
    real, intent(in) :: a(ims:ime,kms:kme,jms:jme)
    write(uout) a(ims:ime, kts:kte+1, 1)
  end subroutine write3f

  subroutine write2(a)
    real, intent(in) :: a(ims:ime,jms:jme)
    write(uout) a(ims:ime, 1)
  end subroutine write2

  subroutine blend(a)
    ! work(k) = 0.5*a(k) + 0.25*(a(k-1) + a(k+1)) on interior levels.
    real, intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    integer :: ii, kk
    do kk = kts + 1, kte - 1
       do ii = its, ite
          work(ii, kk) = 0.5*a(ii,kk,1) + 0.25*(a(ii,kk-1,1) + a(ii,kk+1,1))
       end do
    end do
    do kk = kts + 1, kte - 1
       do ii = its, ite
          a(ii,kk,1) = work(ii, kk)
       end do
    end do
  end subroutine blend

end program run_bl_eeps
