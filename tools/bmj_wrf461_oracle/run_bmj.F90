! Column oracle for WRF v4.6.1 phys/module_cu_bmj.F (cu_physics = 2).
!
! Drives the byte-unmodified BMJINIT and BMJDRV over the columns in
! cases.bin (gen_cases.py) as ONE ROW of a WRF tile: i = 1..ncol, j = 1,
! k = 1..nz mass levels with PINT carrying the nz+1 interfaces, exactly the
! index ranges module_cumulus_driver.F hands BMJDRV (KTE = kde-1, so
! PINT(I,KTE+1,J) is the model-top interface).
!
! Two arms, each four consecutive convective calls so that the carried
! CLDEFI (precipitation efficiency, Registry `ij misc` restart state) is
! read back by the scheme that wrote it:
!   arm 1: bmj_rad_feedback = .false., DT = 60 s, STEPCU = 1
!   arm 2: bmj_rad_feedback = .true.,  DT = 30 s, STEPCU = 3 (DTCNVC = 90 s)
! Between calls the harness (NOT the scheme) applies the scheme's own
! tendencies over DTCNVC plus a small low-level warming/moistening so the
! columns keep evolving.  Every step's inputs are dumped beside its outputs,
! so the GPU side replays the oracle's own inputs and needs to reproduce
! none of this harness arithmetic -- only the carried CLDEFI is its own.
!
! Output (stream, little-endian float32/int32, level-major a(i,k) = i
! fastest, which is the engine's a[k*ncol+i]):
!   per arm, per step, in this order:
!     inputs : t, qv, th, pi, pmid, rho, dz8w (nz each), pint (nz+1),
!              xland, cldefi_in (1 each), kpbl (int32)
!     outputs: rthcuten, rqvcuten, ccldfra, qcconv, qiconv (nz each),
!              raincv, pratec, cutop, cubot, cldefi, convcld (1 each),
!              cu_act_flag (int32)
! and tables.bin: QS0, SQS, PTBL, THE0, STHE, TTBL, THE0Q, STHEQ, TTBLQ,
! then the public QS0_EXP, SQS_EXP, PTBL_EXP shares.  The private tables are
! visible because build.sh compiles the module with -DPRIVATE=PUBLIC, a cpp
! token substitution on the unmodified file that changes no arithmetic.
program run_bmj
  use module_model_constants, only: cp, r_d, xlv, xls, g, svpt0, ep_1
  use module_cu_bmj
  implicit none
  integer :: ncol, nz, ios, i, k, step, arm, nstep, stepcu
  real :: dt
  logical :: feedback
  real, allocatable :: in_pint0(:,:), in_pmid0(:,:), in_t0(:,:), in_qv0(:,:), in_rho0(:,:), in_dz0(:,:)
  real, allocatable :: xland(:), xland2(:,:)
  integer, allocatable :: in_kpbl0(:), fam(:)
  ! WRF-shaped arrays (ims:ime, kms:kme, jms:jme) = (1:ncol, 1:nz+1, 1:1)
  real, allocatable :: t(:,:,:), qv(:,:,:), th(:,:,:), pi3(:,:,:), pmid(:,:,:), &
       rho(:,:,:), dz8w(:,:,:), pint(:,:,:), ccldfra(:,:,:), qcconv(:,:,:), &
       qiconv(:,:,:), rthcuten(:,:,:), rqvcuten(:,:,:), rqccuten(:,:,:), &
       rqrcuten(:,:,:)
  real, allocatable :: cldefi(:,:), raincv(:,:), pratec(:,:), convcld(:,:), &
       cutop(:,:), cubot(:,:), cldefi_in(:,:)
  integer, allocatable :: kpbl(:,:), lowlyr(:,:)
  logical, allocatable :: cu_act(:,:)
  integer, allocatable :: iact(:,:)
  character(len=256) :: cases_path, out_prefix, tables_path
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
             its, ite, jts, jte, kts, kte
  real :: dtcnvc

  if (kind(1.0) /= 4) stop 'default real is not 4 bytes'
  call get_command_argument(1, cases_path)
  call get_command_argument(2, out_prefix)
  call get_command_argument(3, tables_path)

  open(10, file=trim(cases_path), access='stream', form='unformatted', status='old')
  read(10) ncol, nz
  allocate(in_pint0(nz+1,ncol), in_pmid0(nz,ncol), in_t0(nz,ncol), in_qv0(nz,ncol), &
           in_rho0(nz,ncol), in_dz0(nz,ncol), xland(ncol), in_kpbl0(ncol), fam(ncol))
  read(10) in_pint0, in_pmid0, in_t0, in_qv0, in_rho0, in_dz0, xland, in_kpbl0, fam
  close(10)

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1; kte = nz

  allocate(t(ims:ime,kms:kme,jms:jme), qv(ims:ime,kms:kme,jms:jme), &
           th(ims:ime,kms:kme,jms:jme), pi3(ims:ime,kms:kme,jms:jme), &
           pmid(ims:ime,kms:kme,jms:jme), rho(ims:ime,kms:kme,jms:jme), &
           dz8w(ims:ime,kms:kme,jms:jme), pint(ims:ime,kms:kme,jms:jme), &
           ccldfra(ims:ime,kms:kme,jms:jme), qcconv(ims:ime,kms:kme,jms:jme), &
           qiconv(ims:ime,kms:kme,jms:jme), rthcuten(ims:ime,kms:kme,jms:jme), &
           rqvcuten(ims:ime,kms:kme,jms:jme), rqccuten(ims:ime,kms:kme,jms:jme), &
           rqrcuten(ims:ime,kms:kme,jms:jme))
  allocate(cldefi(ims:ime,jms:jme), raincv(ims:ime,jms:jme), &
           pratec(ims:ime,jms:jme), convcld(ims:ime,jms:jme), &
           cutop(ims:ime,jms:jme), cubot(ims:ime,jms:jme), &
           cldefi_in(ims:ime,jms:jme), xland2(ims:ime,jms:jme))
  allocate(kpbl(ims:ime,jms:jme), lowlyr(ims:ime,jms:jme), iact(ims:ime,jms:jme))
  allocate(cu_act(ims:ime,jms:jme))

  nstep = 4
  do arm = 1, 2
    if (arm == 1) then
      feedback = .false.; dt = 60.; stepcu = 1
    else
      feedback = .true.;  dt = 30.; stepcu = 3
    end if
    dtcnvc = dt * real(stepcu)
    ! fresh state per arm
    t = 0.; qv = 0.; th = 0.; pi3 = 0.; pmid = 0.; rho = 0.; dz8w = 0.; pint = 0.
    ccldfra = -999.; qcconv = -999.; qiconv = -999.
    rthcuten = 0.; rqvcuten = 0.
    do i = 1, ncol
      do k = 1, nz
        t(i,k,1) = in_t0(k,i); qv(i,k,1) = in_qv0(k,i); pmid(i,k,1) = in_pmid0(k,i)
        rho(i,k,1) = in_rho0(k,i); dz8w(i,k,1) = in_dz0(k,i)
      end do
      do k = 1, nz + 1
        pint(i,k,1) = in_pint0(k,i)
      end do
      xland2(i,1) = xland(i)
      kpbl(i,1) = in_kpbl0(i)
    end do
    do i = 1, ncol
      do k = 1, nz
        pi3(i,k,1) = (pmid(i,k,1) / 1.e5) ** (r_d / cp)
        th(i,k,1) = t(i,k,1) / pi3(i,k,1)
      end do
    end do
    lowlyr = -7
    call BMJINIT(rthcuten, rqvcuten, rqccuten, rqrcuten, cldefi, lowlyr, &
                 cp, r_d, .false., .true., &
                 ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                 its, ite, jts, jte, kts, kte)
    if (arm == 1) call dump_tables(trim(tables_path))
    open(20, file=trim(out_prefix)//'-arm'//achar(48+arm)//'.bin', &
         access='stream', form='unformatted', status='replace')
    do step = 1, nstep
      cldefi_in = cldefi
      ! ---- inputs -----------------------------------------------------
      call wlev(t); call wlev(qv); call wlev(th); call wlev(pi3)
      call wlev(pmid); call wlev(rho); call wlev(dz8w)
      write(20) ((pint(i,k,1), i = 1, ncol), k = 1, nz + 1)
      write(20) (xland2(i,1), i = 1, ncol)
      write(20) (cldefi_in(i,1), i = 1, ncol)
      write(20) (kpbl(i,1), i = 1, ncol)
      ! poison the outputs so an unwritten slot cannot pass as zero
      raincv = -999.; pratec = -999.; convcld = -999.; cutop = -999.; cubot = -999.
      call BMJDRV(ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte, &
                  dt, step, stepcu, ccldfra, convcld, &
                  raincv, pratec, cutop, cubot, kpbl, &
                  th, t, qv, qcconv, qiconv, feedback, &
                  pint, pmid, pi3, rho, dz8w, &
                  cp, r_d, xlv, xls, g, svpt0, ep_1, &
                  cldefi, lowlyr, xland2, cu_act, &
                  rthcuten, rqvcuten)
      ! ---- outputs ----------------------------------------------------
      call wlev(rthcuten); call wlev(rqvcuten); call wlev(ccldfra)
      call wlev(qcconv); call wlev(qiconv)
      write(20) (raincv(i,1), i = 1, ncol)
      write(20) (pratec(i,1), i = 1, ncol)
      write(20) (cutop(i,1), i = 1, ncol)
      write(20) (cubot(i,1), i = 1, ncol)
      write(20) (cldefi(i,1), i = 1, ncol)
      write(20) (convcld(i,1), i = 1, ncol)
      do i = 1, ncol
        iact(i,1) = merge(1, 0, cu_act(i,1))
      end do
      write(20) (iact(i,1), i = 1, ncol)
      ! ---- harness evolution (not the scheme) -------------------------
      do i = 1, ncol
        do k = 1, nz
          t(i,k,1) = t(i,k,1) + rthcuten(i,k,1) * pi3(i,k,1) * dtcnvc
          qv(i,k,1) = max(qv(i,k,1) + rqvcuten(i,k,1) * dtcnvc, 1.e-9)
          if (k <= 3) then
            t(i,k,1) = t(i,k,1) + 0.35
            qv(i,k,1) = qv(i,k,1) * 1.015
          end if
          th(i,k,1) = t(i,k,1) / pi3(i,k,1)
        end do
      end do
    end do
    close(20)
  end do
  write(*,'(a,i0,a,i0,a)') 'bmj oracle: ', ncol, ' columns x ', nz, ' levels, 2 arms x 4 steps'

contains

  subroutine wlev(a)
    real, intent(in) :: a(ims:ime,kms:kme,jms:jme)
    integer :: ii, kk
    write(20) ((a(ii,kk,1), ii = 1, ncol), kk = 1, nz)
  end subroutine wlev

  subroutine dump_tables(path)
    character(len=*), intent(in) :: path
    open(30, file=path, access='stream', form='unformatted', status='replace')
    write(30) QS0, SQS, PTBL, THE0, STHE, TTBL, THE0Q, STHEQ, TTBLQ
    write(30) QS0_EXP, SQS_EXP, PTBL_EXP
    close(30)
  end subroutine dump_tables

end program run_bmj
