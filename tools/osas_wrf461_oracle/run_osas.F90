! run_osas.F90 -- column oracle driver for WRF v4.6.1 CU_OSAS (cu_physics = 95).
!
! The scheme under test is the byte-pinned phys/module_cu_osas.F, run through
! WRF's own cpp pipeline (build.sh), with exactly three harness edits made by
! sed and nothing else:
!
!   1. `call init_random_seed()` -> `call osas_oracle_seed()` (a no-op), and
!      `call random_number(XKT2)` -> `call osas_oracle_xkt2(XKT2)`, which hands
!      CU_OSAS the XKT2 row this driver pinned.  WRF draws XKT2 from a
!      clock-seeded random_number on every call, so stock WRF is not
!      reproducible; pinning the draw is the only way to grade the scheme.
!   2. A hook call on each side of `CALL SHALCV(...)` that writes T1, Q1, QL,
!      RN, KBOT, KTOP, KUO and CLDWRK as OSASCNV left them, and T1, Q1 as
!      SHALCV left them (stage evidence for the deep and shallow halves).
!   3. module_gfs_funcphys gains a public routine that writes the gfuncphys
!      lookup tables and their index coefficients, so the GPU table build is
!      graded word for word.
!
! Input file (stream, little-endian):
!   int32 im, jm, kx, ncall
!   per call: real32 dt; int32 stepcu, itimestep, chain;
!             real32 t, qv, qc, qi, u, v, p, pi, rho, dz8w   (im, kx, jm) each
!             real32 w, p8w                                   (im, kx+1, jm)
!             real32 xland (im, jm); real64 xkt2 (im, jm)
!   chain = 1 replaces t, qv, qc, qi by the previous call's state advanced by
!   its own tendencies over DT*STEPCU (T by RTHCUTEN*PI), so a sequence of
!   calls exercises the scheme on states it produced itself.
! Output file: per call, the inputs actually used (same layout), then
!   rthcuten, rqvcuten, rqccuten, rqicuten (im, kx, jm) real32,
!   raincv, pratec, htop, hbot (im, jm) real32.
module osas_oracle_hooks
  implicit none
  integer, parameter :: r8 = selected_real_kind(13, 60)
  real(r8), allocatable :: xkt2_grid(:, :)
  integer :: xkt2_row = 0
  integer :: stage_unit = 0
  logical :: stage_on = .false.
end module osas_oracle_hooks

subroutine osas_oracle_seed()
end subroutine osas_oracle_seed

subroutine osas_oracle_xkt2(x)
  use osas_oracle_hooks
  implicit none
  real(r8), intent(out) :: x(*)
  integer :: i
  xkt2_row = xkt2_row + 1
  do i = 1, size(xkt2_grid, 1)
    x(i) = xkt2_grid(i, xkt2_row)
  end do
end subroutine osas_oracle_xkt2

subroutine osas_oracle_after_deep(im, kx, t1, q1, ql, rn, kbot, ktop, kuo, cldwrk)
  use osas_oracle_hooks
  implicit none
  integer, intent(in) :: im, kx, kbot(im), ktop(im), kuo(im)
  real(r8), intent(in) :: t1(im, kx), q1(im, kx), ql(im, kx, 2), rn(im), cldwrk(im)
  if (stage_on) write(stage_unit) t1, q1, ql, rn, cldwrk, kbot, ktop, kuo
end subroutine osas_oracle_after_deep

subroutine osas_oracle_after_shallow(im, kx, t1, q1)
  use osas_oracle_hooks
  implicit none
  integer, intent(in) :: im, kx
  real(r8), intent(in) :: t1(im, kx), q1(im, kx)
  if (stage_on) write(stage_unit) t1, q1
end subroutine osas_oracle_after_shallow

program run_osas
  use module_cu_osas, only: cu_osas
  use module_gfs_funcphys, only: gfuncphys, osas_oracle_dump_tables
  use osas_oracle_hooks
  implicit none
  character(len=1024) :: inpath, outpath, stagepath, tablepath
  integer :: uin, uout, im, jm, kx, ncall, icall, stepcu, itimestep, chain
  real :: dt, span
  real, allocatable, dimension(:, :, :) :: t, qv, qc, qi, u, v, p, pii, rho, dz8w
  real, allocatable, dimension(:, :, :) :: w, p8w
  real, allocatable, dimension(:, :, :) :: rthcuten, rqvcuten, rqccuten, rqicuten
  real, allocatable, dimension(:, :, :) :: rucuten, rvcuten
  real, allocatable, dimension(:, :) :: xland, raincv, pratec, htop, hbot
  real, allocatable, dimension(:, :, :) :: tprev, qvprev, qcprev, qiprev
  real, allocatable, dimension(:, :, :) :: rthprev, rqvprev, rqcprev, rqiprev, piprev
  real(r8), allocatable :: xkt2(:, :)
  logical, allocatable :: cu_act_flag(:, :)
  integer :: i, k, j
  real :: spanprev

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, stagepath)
  call get_command_argument(4, tablepath)

  call gfuncphys()
  if (len_trim(tablepath) > 0 .and. trim(tablepath) /= '-') then
    call osas_oracle_dump_tables(trim(tablepath))
  end if

  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', status='old')
  read(uin) im, jm, kx, ncall
  ! WRF memory extents: (ims:ime, kms:kme, jms:jme) with kme = kx + 1.
  allocate(t(im, kx + 1, jm), qv(im, kx + 1, jm), qc(im, kx + 1, jm), qi(im, kx + 1, jm))
  allocate(u(im, kx + 1, jm), v(im, kx + 1, jm), p(im, kx + 1, jm), pii(im, kx + 1, jm))
  allocate(rho(im, kx + 1, jm), dz8w(im, kx + 1, jm), w(im, kx + 1, jm), p8w(im, kx + 1, jm))
  allocate(rthcuten(im, kx + 1, jm), rqvcuten(im, kx + 1, jm))
  allocate(rqccuten(im, kx + 1, jm), rqicuten(im, kx + 1, jm))
  allocate(rucuten(im, jm, kx + 1), rvcuten(im, jm, kx + 1))
  allocate(xland(im, jm), raincv(im, jm), pratec(im, jm), htop(im, jm), hbot(im, jm))
  allocate(tprev(im, kx, jm), qvprev(im, kx, jm), qcprev(im, kx, jm), qiprev(im, kx, jm))
  allocate(rthprev(im, kx, jm), rqvprev(im, kx, jm), rqcprev(im, kx, jm), rqiprev(im, kx, jm))
  allocate(piprev(im, kx, jm))
  allocate(xkt2(im, jm), xkt2_grid(im, jm), cu_act_flag(im, jm))
  spanprev = 0.

  open(newunit=uout, file=trim(outpath), access='stream', form='unformatted', status='replace')
  if (len_trim(stagepath) > 0 .and. trim(stagepath) /= '-') then
    open(newunit=stage_unit, file=trim(stagepath), access='stream', form='unformatted', &
         status='replace')
    stage_on = .true.
  end if
  write(uout) im, jm, kx, ncall

  do icall = 1, ncall
    read(uin) dt, stepcu, itimestep, chain
    t = 0.; qv = 0.; qc = 0.; qi = 0.; u = 0.; v = 0.; p = 0.; pii = 0.; rho = 0.; dz8w = 0.
    read(uin) t(:, 1:kx, :), qv(:, 1:kx, :), qc(:, 1:kx, :), qi(:, 1:kx, :)
    read(uin) u(:, 1:kx, :), v(:, 1:kx, :), p(:, 1:kx, :), pii(:, 1:kx, :)
    read(uin) rho(:, 1:kx, :), dz8w(:, 1:kx, :)
    read(uin) w, p8w, xland, xkt2
    if (chain == 1 .and. icall > 1) then
      ! The previous call's own result, advanced over its DT*STEPCU.
      do j = 1, jm
        do k = 1, kx
          do i = 1, im
            t(i, k, j) = tprev(i, k, j) + rthprev(i, k, j) * piprev(i, k, j) * spanprev
            qv(i, k, j) = max(qvprev(i, k, j) + rqvprev(i, k, j) * spanprev, 0.)
            qc(i, k, j) = max(qcprev(i, k, j) + rqcprev(i, k, j) * spanprev, 0.)
            qi(i, k, j) = max(qiprev(i, k, j) + rqiprev(i, k, j) * spanprev, 0.)
          end do
        end do
      end do
    end if
    xkt2_grid = xkt2
    xkt2_row = 0
    rthcuten = 0.; rqvcuten = 0.; rqccuten = 0.; rqicuten = 0.
    rucuten = 0.; rvcuten = 0.; raincv = 0.; pratec = 0.; htop = 0.; hbot = 0.
    cu_act_flag = .false.

    call cu_osas(DT=dt, ITIMESTEP=itimestep, STEPCU=stepcu,                  &
                 RTHCUTEN=rthcuten, RQVCUTEN=rqvcuten, RQCCUTEN=rqccuten,     &
                 RQICUTEN=rqicuten, RUCUTEN=rucuten, RVCUTEN=rvcuten,         &
                 RAINCV=raincv, PRATEC=pratec, HTOP=htop, HBOT=hbot,          &
                 U3D=u, V3D=v, W=w, T3D=t, QV3D=qv, QC3D=qc, QI3D=qi,         &
                 PI3D=pii, RHO3D=rho, DZ8W=dz8w, PCPS=p, P8W=p8w,             &
                 XLAND=xland, CU_ACT_FLAG=cu_act_flag,                        &
                 P_QC=2, P_QI=3, P_FIRST_SCALAR=2,                            &
                 ids=1, ide=im + 1, jds=1, jde=jm + 1, kds=1, kde=kx + 1,     &
                 ims=1, ime=im, jms=1, jme=jm, kms=1, kme=kx + 1,             &
                 its=1, ite=im, jts=1, jte=jm, kts=1, kte=kx)

    write(uout) dt, stepcu, itimestep, chain
    write(uout) t(:, 1:kx, :), qv(:, 1:kx, :), qc(:, 1:kx, :), qi(:, 1:kx, :)
    write(uout) u(:, 1:kx, :), v(:, 1:kx, :), p(:, 1:kx, :), pii(:, 1:kx, :)
    write(uout) rho(:, 1:kx, :), dz8w(:, 1:kx, :)
    write(uout) w, p8w, xland, xkt2
    write(uout) rthcuten(:, 1:kx, :), rqvcuten(:, 1:kx, :), rqccuten(:, 1:kx, :), &
                rqicuten(:, 1:kx, :)
    write(uout) raincv, pratec, htop, hbot

    span = dt * stepcu
    spanprev = span
    tprev = t(:, 1:kx, :); qvprev = qv(:, 1:kx, :)
    qcprev = qc(:, 1:kx, :); qiprev = qi(:, 1:kx, :)
    rthprev = rthcuten(:, 1:kx, :); rqvprev = rqvcuten(:, 1:kx, :)
    rqcprev = rqccuten(:, 1:kx, :); rqiprev = rqicuten(:, 1:kx, :)
    piprev = pii(:, 1:kx, :)
  end do
  close(uout)
  close(uin)
  if (stage_on) close(stage_unit)
end program run_osas
