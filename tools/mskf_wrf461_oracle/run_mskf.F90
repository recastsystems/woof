! WRF v4.6.1 Multiscale Kain-Fritsch (cu_physics = 11) column oracle.
!
! Drives the byte-unmodified MSKF_CPS (phys/module_cu_mskf.F:3424) on the
! columns make_cases.py wrote, exactly as module_cumulus_driver.F:1051-1094
! calls it: every optional argument present, the WRF model constants of
! share/module_model_constants.F, trigger = kfeta_trigger, the CMT on
! (it is a hard-wired .TRUE. inside MSKF_CPS), KF_EDRATES = 1 so the
! entrainment and detrainment rates are captured too.
!
! Each case runs twice:
!   * alone, in a one-column tile -- this is the fixture;
!   * in the middle of a three-column tile between two OTHER columns, with
!     the same scalars -- the isolation check.  MSKF at trigger = 1 has no
!     horizontal coupling; the count of differing words is recorded per case
!     and build.sh fails on any nonzero count.
!
! Every output array is pre-filled with the sentinel -7777.0 so that a
! column MSKF does not touch (an NCA-held column, levels above LTOP, fields
! written only on the aercu path) is visible as untouched, and the port is
! graded on leaving them untouched too.
!
! Output: out_f32.bin, little-endian float32, one record per case of
! NF3 x nz words (3-D fields, field-major) then NF2 words (2-D fields), in
! the order named in out_fields.txt; isolation.txt, one line per case.
program run_mskf
  use module_cu_mskf, only: mskf_cps, mskf_init
  implicit none

  integer, parameter :: nz = 40
  integer, parameter :: nint_ = 9, nscal = 10, nlev = 11, nsrc = 10
  integer, parameter :: nf3 = 27, nf2 = 8
  real, parameter :: sentinel = -7777.0

  ! WRF v4.6.1 share/module_model_constants.F
  real, parameter :: g = 9.81, r_d = 287., cp = 7.*r_d/2., r_v = 461.6
  real, parameter :: ep_1 = r_v/r_d - 1., ep_2 = r_d/r_v
  real, parameter :: xlv0 = 3.15e6, xlv1 = 2370., xls0 = 2.905e6, &
                     xls1 = 259.532
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65, &
                     svpt0 = 273.15

  integer :: ncase, recl, c, k, n, ios, ndiff, nb1, nb2
  integer, allocatable :: ints(:,:)
  real, allocatable :: rec(:,:)
  real, allocatable :: single(:), triple(:)
  character(len=16) :: arg
  integer :: unit_out, unit_iso

  ! one-column scratch for mskf_init
  real :: z3(1, nz+1, 1), z2(1, 1)

  call get_command_argument(1, arg)
  read(arg, *) ncase
  recl = nscal + nlev*nz + (nz+1) + nsrc*nz

  allocate(ints(nint_, ncase), rec(recl, ncase))
  open(newunit=n, file='cases_i32.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(n) ints
  close(n)
  open(newunit=n, file='cases_f32.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(n) rec
  close(n)

  ! MSKF_LUTAB is built by mskf_init (module_cu_mskf.F:7417), the WRF call
  ! path module_physics_init.F:4070.  restart = .false. also seeds NCA = -100
  ! and W0AVG = 0 on the scratch arrays, which the cases then overwrite.
  z3 = 0.; z2 = 0.
  call mskf_init(z3, z3, z3, z3, z3, z3, z2, z3, 2, 2, &
                 svp1, svp2, svp3, svpt0, 1, .false., .true., &
                 1, 2, 1, 2, 1, nz+1, &
                 1, 1, 1, 1, 1, nz+1, &
                 1, 1, 1, 1, 1, nz, z3, z3)

  allocate(single(nf3*nz + nf2), triple(nf3*nz + nf2))
  open(newunit=unit_out, file='out_f32.bin', access='stream', &
       form='unformatted', status='replace', convert='little_endian')
  open(newunit=unit_iso, file='isolation.txt', status='replace')
  do c = 1, ncase
    nb1 = mod(c, ncase) + 1
    nb2 = mod(c + 7, ncase) + 1
    call run_tile(c, (/ c /), 1, 1, single)
    call run_tile(c, (/ nb1, c, nb2 /), 3, 2, triple)
    ndiff = 0
    do k = 1, nf3*nz + nf2
      if (transfer(single(k), 0) /= transfer(triple(k), 0)) ndiff = ndiff + 1
    end do
    write(unit_iso, '(I4,1X,I8)') c, ndiff
    write(unit_out) single
  end do
  close(unit_out)
  close(unit_iso)

contains

  subroutine run_tile(cs, cols, nx, pick, outv)
    ! cs: the case whose SCALARS and control words drive the tile;
    ! cols: which case's column sits at each i.
    integer, intent(in) :: cs, nx, pick
    integer, intent(in) :: cols(nx)
    real, intent(out) :: outv(:)

    real, dimension(nx, nz+1, 1) :: u, v, th, t, w, qv, dz8w, pcps, rho, pi3, &
         w0avg, rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten, &
         rqvften, rucuten, rvcuten, cldfra_dp, cldfra_sh, qc_kf, qi_kf, &
         qr_kf, qs_kf, nc_kf, ni_kf, nr_kf, ns_kf, ccn_kf, efcs, efis, efss, &
         udr_kf, ddr_kf, uer_kf, der_kf
    real, dimension(nx, nz+1, 1, nsrc) :: aerocu
    real, dimension(nx, 1) :: raincv, pratec, nca, cutop, cubot, ainc_frac, &
         timec_kf, zol, hfx, ust, pblh, xland
    logical, dimension(nx, 1) :: cu_act_flag
    real :: dx, dt, cudt, aercu_fct
    logical :: warm_rain, f_qr, f_qi, f_qs, adapt
    integer :: i, cc, f, s, stepcu, aercu_opt, kf_edrates, trigger

    warm_rain = ints(1, cs) /= 0
    f_qr = ints(2, cs) /= 0
    f_qi = ints(3, cs) /= 0
    f_qs = ints(4, cs) /= 0
    aercu_opt = ints(5, cs)
    kf_edrates = ints(6, cs)
    adapt = ints(7, cs) /= 0
    stepcu = ints(8, cs)
    trigger = ints(9, cs)
    dx = rec(1, cs); dt = rec(2, cs); cudt = rec(3, cs)
    aercu_fct = rec(4, cs)

    ! everything the scheme may write starts as the sentinel
    rthcuten = sentinel; rqvcuten = sentinel; rqccuten = sentinel
    rqrcuten = sentinel; rqicuten = sentinel; rqscuten = sentinel
    rucuten = sentinel; rvcuten = sentinel
    cldfra_dp = sentinel; cldfra_sh = sentinel; qc_kf = sentinel
    qi_kf = sentinel; qr_kf = sentinel; qs_kf = sentinel; nc_kf = sentinel
    ni_kf = sentinel; nr_kf = sentinel; ns_kf = sentinel; ccn_kf = sentinel
    efcs = sentinel; efis = sentinel; efss = sentinel
    udr_kf = sentinel; ddr_kf = sentinel; uer_kf = sentinel; der_kf = sentinel
    raincv = sentinel; pratec = sentinel; cutop = sentinel; cubot = sentinel
    ainc_frac = sentinel; timec_kf = sentinel
    cu_act_flag = .false.
    u = 0.; v = 0.; th = 0.; t = 0.; w = 0.; qv = 0.; dz8w = 0.; pcps = 0.
    rho = 0.; pi3 = 0.; w0avg = 0.; rqvften = 0.; aerocu = 0.

    do i = 1, nx
      cc = cols(i)
      call level(cc, 1, u(i,:,1)); call level(cc, 2, v(i,:,1))
      call level(cc, 3, t(i,:,1)); call level(cc, 4, th(i,:,1))
      call level(cc, 5, qv(i,:,1)); call level(cc, 6, pcps(i,:,1))
      call level(cc, 7, rho(i,:,1)); call level(cc, 8, pi3(i,:,1))
      call level(cc, 9, dz8w(i,:,1))
      call level(cc, 10, w0avg(i,:,1))
      call level(cc, 11, rqvften(i,:,1))
      w(i, 1:nz+1, 1) = rec(nscal + nlev*nz + 1 : nscal + nlev*nz + nz + 1, cc)
      do s = 1, nsrc
        aerocu(i, 1:nz, 1, s) = rec(nscal + nlev*nz + nz + 1 + (s-1)*nz + 1 : &
                                    nscal + nlev*nz + nz + 1 + s*nz, cc)
      end do
      nca(i, 1) = rec(5, cs)
      hfx(i, 1) = rec(6, cc); ust(i, 1) = rec(7, cc); pblh(i, 1) = rec(8, cc)
      zol(i, 1) = rec(9, cc); xland(i, 1) = rec(10, cc)
    end do

    call mskf_cps( &
         1, nx+1, 1, 2, 1, nz+1, &
         1, nx, 1, 1, 1, nz+1, &
         1, nx, 1, 1, 1, nz, &
         trigger, &
         dt, 1, dx, cudt, adapt, &
         rho, raincv, pratec, nca, &
         u, v, th, t, w, dz8w, pcps, pi3, &
         w0avg, xlv0, xlv1, xls0, xls1, cp, r_d, g, ep_1, &
         ep_2, svp1, svp2, svp3, svpt0, &
         stepcu, cu_act_flag, warm_rain, cutop, cubot, &
         qv, &
         .true., .true., f_qr, f_qi, f_qs, &
         rthcuten, rqvcuten, rqccuten, rqrcuten, &
         rqicuten, rqscuten, rqvften, &
         cldfra_dp, cldfra_sh, &
         qc_kf, qi_kf, qr_kf, qs_kf, &
         nc_kf, ni_kf, nr_kf, ns_kf, &
         ccn_kf, ainc_frac, &
         udr_kf, ddr_kf, &
         uer_kf, der_kf, &
         timec_kf, kf_edrates, &
         zol, hfx, ust, pblh, &
         aerocu, nsrc, aercu_fct, aercu_opt, &
         efcs, efis, efss, &
         rucuten, rvcuten, xland)

    f = 0
    call put3(outv, f, pick, rthcuten); call put3(outv, f, pick, rqvcuten); call put3(outv, f, pick, rqccuten)
    call put3(outv, f, pick, rqrcuten); call put3(outv, f, pick, rqicuten); call put3(outv, f, pick, rqscuten)
    call put3(outv, f, pick, rucuten); call put3(outv, f, pick, rvcuten)
    call put3(outv, f, pick, cldfra_dp); call put3(outv, f, pick, cldfra_sh); call put3(outv, f, pick, qc_kf)
    call put3(outv, f, pick, qi_kf); call put3(outv, f, pick, qr_kf); call put3(outv, f, pick, qs_kf); call put3(outv, f, pick, nc_kf)
    call put3(outv, f, pick, ni_kf); call put3(outv, f, pick, nr_kf); call put3(outv, f, pick, ns_kf); call put3(outv, f, pick, ccn_kf)
    call put3(outv, f, pick, efcs); call put3(outv, f, pick, efis); call put3(outv, f, pick, efss)
    call put3(outv, f, pick, udr_kf); call put3(outv, f, pick, ddr_kf); call put3(outv, f, pick, uer_kf); call put3(outv, f, pick, der_kf)
    call put3(outv, f, pick, w0avg)
    outv(f+1) = raincv(pick, 1); outv(f+2) = pratec(pick, 1)
    outv(f+3) = nca(pick, 1); outv(f+4) = cutop(pick, 1)
    outv(f+5) = cubot(pick, 1)
    outv(f+6) = merge(1., 0., cu_act_flag(pick, 1))
    outv(f+7) = ainc_frac(pick, 1); outv(f+8) = timec_kf(pick, 1)

  end subroutine run_tile

  subroutine level(cc, idx, dest)
    integer, intent(in) :: cc, idx
    real, intent(out) :: dest(:)
    dest(1:nz) = rec(nscal + (idx-1)*nz + 1 : nscal + idx*nz, cc)
    dest(nz+1) = 0.
  end subroutine level

  subroutine put3(outv, f, pick, a)
    real, intent(inout) :: outv(:)
    integer, intent(inout) :: f
    integer, intent(in) :: pick
    real, intent(in) :: a(:,:,:)
    outv(f+1 : f+nz) = a(pick, 1:nz, 1)
    f = f + nz
  end subroutine put3

end program run_mskf
