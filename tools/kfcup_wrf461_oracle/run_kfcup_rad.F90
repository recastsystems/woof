module kfcup_rad_extracts
  ! WRF's own radiation-driver code for the KF-CuP cloud override, and the
  ! KF-family arm of advance_ppt, compiled from WRF's preprocessed sources.
  !
  ! build.sh cuts three pieces out of the WRF build's .f90 files between
  ! textual markers and records their digests; nothing below is retyped:
  !   cal_cldfra1.inc          module_radiation_driver: SUBROUTINE cal_cldfra1
  !                            (the icloud = 1 fraction and its cldfra1_flag)
  !   kfcup_merge_block.inc    module_radiation_driver: the
  !                            `if(present(cldfra_cup))` block that overrides
  !                            CLDFRA and floors QC for cu_physics = 10
  !   kfcup_advance_block.inc  module_physics_addtendc advance_ppt: the
  !                            CASE (KFETASCHEME, MSKFSCHEME, KFCUPSCHEME) arm
  ! The wrappers supply only the declarations the pieces read, under the
  ! names WRF gives them.
  implicit none
contains

  include 'cal_cldfra1.inc'

  subroutine kfcup_merge(cu_physics, icloud, shallowcu_forced_ra, cldfra1_flag, shall, cubot, &
       cutop, cldfra, qc, cldfra_cup, its, ite, jts, jte, kts, kte, ims, ime, jms, jme, kms, kme)
    use module_state_description, only: KFCUPSCHEME
    integer, intent(in) :: cu_physics, icloud, its, ite, jts, jte, kts, kte, ims, ime, jms, jme, kms, kme
    logical, intent(in) :: shallowcu_forced_ra
    integer, intent(in) :: cldfra1_flag(ims:ime, kms:kme, jms:jme)
    real, intent(in) :: shall(ims:ime, jms:jme), cubot(ims:ime, jms:jme), cutop(ims:ime, jms:jme)
    real, intent(inout) :: cldfra(ims:ime, kms:kme, jms:jme), qc(ims:ime, kms:kme, jms:jme)
    real, optional, intent(inout) :: cldfra_cup(ims:ime, kms:kme, jms:jme)
    real :: cldfra_cup_mod
    logical :: compute_cldfra_cup
    integer :: i, j, k
    include 'kfcup_merge_block.inc'
  end subroutine kfcup_merge

  subroutine kfcup_advance(cu_physics_in, rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten, &
       cldfra_cup, nca, htop, hbot, cutop, cubot, dt, p_qi, p_qs, &
       i_start, i_end, j_start, j_end, k_start, k_end, kde, ims, ime, jms, jme, kms, kme)
    use module_state_description, only: KFETASCHEME, MSKFSCHEME, KFCUPSCHEME, PARAM_FIRST_SCALAR
    integer, intent(in) :: cu_physics_in, p_qi, p_qs, i_start, i_end, j_start, j_end, k_start, k_end
    integer, intent(in) :: kde, ims, ime, jms, jme, kms, kme
    real, intent(in) :: dt
    real, dimension(ims:ime, kms:kme, jms:jme), intent(inout) :: rthcuten, rqvcuten, rqccuten, &
         rqrcuten, rqicuten, rqscuten, cldfra_cup
    real, dimension(ims:ime, jms:jme), intent(inout) :: nca, htop, hbot
    real, dimension(ims:ime, jms:jme), intent(in) :: cutop, cubot
    type :: flags_t
       integer :: cu_physics
    end type flags_t
    type(flags_t) :: config_flags
    integer :: i, j, k, ncutop, ncubot
    config_flags%cu_physics = cu_physics_in
    select case (config_flags%cu_physics)
    case (KFETASCHEME, MSKFSCHEME, KFCUPSCHEME)
       include 'kfcup_advance_block.inc'
    end select
  end subroutine kfcup_advance

end module kfcup_rad_extracts

program run_kfcup_rad
  ! The radiation-side half of KF-CuP, captured from WRF's own code.
  !
  !   kfcup-rad-levels.csv  per (arm, case, k): cal_cldfra1's CLDFRA and
  !                         cldfra1_flag, then CLDFRA, QC and CLDFRA_CUP after
  !                         the KF-CuP override; arms are icloud (1 or 3,
  !                         whose override is the same with no flag gating)
  !                         x shallowcu_forced_ra (false, true)
  !   kfcup-advance.csv     advance_ppt's KF-family arm for cu_physics = 10:
  !                         NCA after the step and whether the stored rates
  !                         and CLDFRA_CUP were cleared, over an NCA x DT grid
  !
  ! The KF-CuP state the override reads (SHALL, CUBOT, CUTOP, CLDFRA_CUP) is
  ! the end-to-end capture's own output for configuration 1, so this is the
  ! radiation step that follows that cumulus call.
  use iso_fortran_env, only: int32
  use module_state_description, only: KFCUPSCHEME
  use kfcup_fixture
  use kfcup_rad_extracts
  implicit none
  integer :: n, k, ic, iarm, ufile, uadv
  real :: u(nz), v(nz), th(nz), t(nz), w(nz+1), dz(nz), p(nz), pii(nz), rho(nz), qv(nz)
  real :: z(nz), zw(nz+1), w0avg(nz)
  real :: psfc, ht, tsk, hfx, qfx, xland, mavail, br, regime, pblh, t2, q2
  integer :: kpbl
  real, allocatable :: t3(:,:,:), p3(:,:,:), qv3(:,:,:), qc3(:,:,:), qi3(:,:,:), qs3(:,:,:)
  real, allocatable :: cf(:,:,:), cf0(:,:,:), qcm(:,:,:), cup3(:,:,:), cupm(:,:,:), fice(:,:,:), frain(:,:,:)
  integer, allocatable :: flag3(:,:,:)
  real, allocatable :: shall(:,:), cubot(:,:), cutop(:,:)
  character(len=512) :: line
  integer :: icfg, icase, kk, ios
  character(len=8) :: words(32)
  integer :: icl, ifr
  logical :: fr

  call build_table()
  allocate(t3(ncase,nz+1,1), p3(ncase,nz+1,1), qv3(ncase,nz+1,1), qc3(ncase,nz+1,1), &
       qi3(ncase,nz+1,1), qs3(ncase,nz+1,1), cf(ncase,nz+1,1), cf0(ncase,nz+1,1), &
       qcm(ncase,nz+1,1), cup3(ncase,nz+1,1), cupm(ncase,nz+1,1), fice(ncase,nz+1,1), &
       frain(ncase,nz+1,1), flag3(ncase,nz+1,1), shall(ncase,1), cubot(ncase,1), cutop(ncase,1))
  t3 = 250.; p3 = 5.e4; qv3 = 0.; qc3 = 0.; qi3 = 0.; qs3 = 0.; fice = 0.; frain = 0.
  do n = 1, ncase
     call build_column(cases(n), 60., u, v, th, t, w, dz, p, pii, rho, qv, z, zw, w0avg, &
          psfc, ht, tsk, hfx, qfx, xland, mavail, br, regime, pblh, kpbl, t2, q2)
     t3(n,1:nz,1) = t; p3(n,1:nz,1) = p; qv3(n,1:nz,1) = qv
     ! A resolved cloud pattern that varies by case: none, a saturated deck,
     ! a sub-saturated wisp, and ice aloft, so every cal_cldfra1 branch
     ! (flag 1, 2, 3) meets every KF-CuP state.
     do k = 1, nz
        select case (mod(n + k, 5))
        case (0)
           qc3(n,k,1) = 0.
        case (1)
           qc3(n,k,1) = 2.e-4
           qv3(n,k,1) = qv(k)*1.08
        case (2)
           qc3(n,k,1) = 3.e-6
        case (3)
           qi3(n,k,1) = 5.e-6; qs3(n,k,1) = 1.e-5
        case default
           qc3(n,k,1) = 1.e-13
        end select
     end do
  end do
  ! The cumulus state: configuration 1 of the end-to-end capture.
  open(newunit=ufile, file='kfcup-surface.csv', status='old')
  read(ufile,'(a)') line
  do
     read(ufile,'(a)', iostat=ios) line
     if (ios /= 0) exit
     call split(line, words)
     read(words(1),*) icfg; read(words(2),*) icase
     if (icfg /= 1) cycle
     shall(icase,1) = word(words(7)); cutop(icase,1) = word(words(5)); cubot(icase,1) = word(words(6))
  end do
  close(ufile)
  open(newunit=ufile, file='kfcup-levels.csv', status='old')
  read(ufile,'(a)') line
  cup3 = 0.
  do
     read(ufile,'(a)', iostat=ios) line
     if (ios /= 0) exit
     call split(line, words)
     read(words(1),*) icfg; read(words(2),*) icase; read(words(3),*) kk
     if (icfg /= 1) cycle
     cup3(icase,kk,1) = word(words(11))
  end do
  close(ufile)

  open(newunit=ufile, file='kfcup-rad-levels.csv', status='replace')
  write(ufile,'(a)') 'icloud,shallowcu_forced_ra,case,k,shall,cubot,cutop,qc_in,cldfra_cup_in,cldfra1,cldfra1_flag,cldfra,qc,cldfra_cup'
  do icl = 1, 2
     do ifr = 0, 1
        fr = ifr == 1
        cf = 0.; flag3 = 0
        call cal_cldfra1(cf, qv3, qc3, qi3, qs3, .true., .true., .true., .true., t3, p3, fice, frain, &
             8, flag3, 1, ncase+1, 1, 2, 1, nz+1, 1, ncase, 1, 1, 1, nz+1, 1, ncase, 1, 1, 1, nz)
        cf0 = cf; qcm = qc3; cupm = cup3
        call kfcup_merge(KFCUPSCHEME, merge(1, 3, icl == 1), fr, flag3, shall, cubot, cutop, cf, qcm, &
             cupm, 1, ncase, 1, 1, 1, nz, 1, ncase, 1, 1, 1, nz+1)
        do n = 1, ncase
           do k = 1, nz
              write(ufile,'(i0,",",l1,",",i0,",",i0,6(",",a),",",i0,3(",",a))') merge(1, 3, icl == 1), fr, n, k, &
                   hx(shall(n,1)), hx(cubot(n,1)), hx(cutop(n,1)), hx(qc3(n,k,1)), hx(cup3(n,k,1)), &
                   hx(cf0(n,k,1)), flag3(n,k,1), hx(cf(n,k,1)), hx(qcm(n,k,1)), hx(cupm(n,k,1))
           end do
        end do
     end do
  end do
  close(ufile)

  call advance_grid()

contains

  real function word(s)
    character(len=*), intent(in) :: s
    integer(int32) :: iw
    read(s,'(z8)') iw
    word = transfer(iw, 1.0)
  end function word

  subroutine split(s, out)
    character(len=*), intent(in) :: s
    character(len=8), intent(out) :: out(:)
    integer :: a, b, m
    a = 1; m = 0
    out = ''
    do
       b = index(s(a:), ',')
       m = m + 1
       if (m > size(out)) exit
       if (b == 0) then
          out(m) = adjustl(s(a:))
          exit
       end if
       out(m) = adjustl(s(a:a+b-2))
       a = a + b
    end do
  end subroutine split

  subroutine advance_grid()
    ! advance_ppt's KF-family arm on a 1 x 1 x 3 column, once per (DT, NCA).
    real :: dts(4) = [20., 45., 60., 120.]
    real :: fr(15) = [-100., -1., 0., 0.4, 0.49, 0.5, 0.51, 1.0, 1.49, 1.5, 1.51, 2.0, 2.5, 3.0, 30.]
    real :: rth(1,3,1), rqv(1,3,1), rqc(1,3,1), rqr(1,3,1), rqi(1,3,1), rqs(1,3,1), cup(1,3,1)
    real :: nca(1,1), htop(1,1), hbot(1,1), ctop(1,1), cbot(1,1), nca0
    integer :: id, ifr, uadv
    open(newunit=uadv, file='kfcup-advance.csv', status='replace')
    write(uadv,'(a)') 'dt,nca_in,nca_out,rates_cleared,cldfra_cup_cleared,htop,hbot'
    do id = 1, 4
       do ifr = 1, 15
          rth = 1.; rqv = 1.; rqc = 1.; rqr = 1.; rqi = 1.; rqs = 1.; cup = 0.5
          nca0 = fr(ifr)*dts(id)
          if (fr(ifr) <= 0.) nca0 = fr(ifr)
          nca = nca0; htop = 7.; hbot = 9.; ctop = 12.; cbot = 4.
          call kfcup_advance(KFCUPSCHEME, rth, rqv, rqc, rqr, rqi, rqs, cup, nca, htop, hbot, ctop, cbot, &
               dts(id), 2, 2, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 3)
          write(uadv,'(a,",",a,",",a,",",i0,",",i0,2(",",a))') hx(dts(id)), hx(nca0), hx(nca(1,1)), &
               merge(1, 0, rth(1,1,1) == 0.), merge(1, 0, cup(1,1,1) == 0.), hx(htop(1,1)), hx(hbot(1,1))
       end do
    end do
    close(uadv)
  end subroutine advance_grid

end program run_kfcup_rad
