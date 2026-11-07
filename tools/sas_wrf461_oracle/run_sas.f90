! WRF v4.6.1 SAS (cu_physics = 94) column oracle.
!
! Linked against the module objects of a BUILT WRF tree (strict or scalar,
! /work/wrf/build/<tree>/WRF-4.6.1/phys/module_*.o), so the arithmetic graded is the
! arithmetic wrf.exe runs: module_cu_sas.o, module_gfs_funcphys.o,
! module_gfs_physcons.o, module_gfs_machine.o, unmodified.  This file only
! drives them.
!
! Per column (cases.bin, see gen_cases.py):
!   stage 3  CU_SAS itself on a one-column tile -> stage3.bin
!   stage 1  SASCNVN on the kind_phys inputs CU_SAS forms (the prep below is
!            CU_SAS :335-406 verbatim, compiled with the same flags; the
!            chained check at the end proves it reproduces CU_SAS)  -> stage1.bin
!   stage 2  OLD_ARW_SHALCV on the post-SASCNVN state            -> stage2.bin
! and the funcphys tables after GFUNCPHYS (tables.bin, sas_tables_dump.c).
!
! Two self-checks print to stdout and fail the run when nonzero:
!   chained   stage1+stage2 recombined into tendencies the way CU_SAS does,
!             against CU_SAS's own outputs, word for word;
!   row       every column re-run inside one multi-column CU_SAS row call per
!             level count, against the one-column run, word for word (the
!             GPU runs one thread per column; WRF runs a tile row).
program run_sas
  use module_gfs_machine, only: kind_phys
  use module_gfs_physcons, only: grav => con_g, rd => con_rd, rovcp => con_rocp
  use module_gfs_funcphys, only: gfuncphys
  use module_cu_sas, only: cu_sas, sascnvn, old_arw_shalcv
  use iso_c_binding
  implicit none

  interface
    subroutine sas_dump_tables(path) bind(C, name='sas_dump_tables')
      import :: c_char
      character(kind=c_char), dimension(*), intent(in) :: path
    end subroutine
  end interface

  character(len=512) :: dir
  integer :: ncase, ic, nz, stepcu, cat, k, i, nbad_chain, nbad_row, n
  real :: dt, pgcon, xland
  real, allocatable :: t3d(:), qv(:), qc(:), qi(:), u(:), v(:), w(:), pcps(:), &
                       pi3(:), rho(:), dz8w(:), p8w(:)
  integer, allocatable :: allnz(:)
  real, allocatable :: keep(:,:)            ! per-case stage-3 words for the row check
  integer :: maxnz, nrec
  real, allocatable :: o3(:)

  call get_command_argument(1, dir)
  call gfuncphys
  call sas_dump_tables(trim(dir)//'/tables.bin'//c_null_char)

  open(10, file=trim(dir)//'/cases.bin', access='stream', form='unformatted', status='old')
  open(21, file=trim(dir)//'/stage1.bin', access='stream', form='unformatted', status='replace')
  open(22, file=trim(dir)//'/stage2.bin', access='stream', form='unformatted', status='replace')
  open(23, file=trim(dir)//'/stage3.bin', access='stream', form='unformatted', status='replace')
  open(24, file=trim(dir)//'/stage3_shaloff.bin', access='stream', form='unformatted', status='replace')
  read(10) ncase
  write(24) ncase
  write(21) ncase; write(22) ncase; write(23) ncase
  allocate(allnz(ncase))
  maxnz = 0
  nbad_chain = 0
  ! first pass only to size the row-check store
  do ic = 1, ncase
    call read_case()
    allnz(ic) = nz
    maxnz = max(maxnz, nz)
  end do
  nrec = 4*maxnz + 4
  allocate(keep(nrec, ncase))
  rewind(10)
  read(10) ncase
  do ic = 1, ncase
    call read_case()
    call one_column(ic)
  end do
  close(21); close(22); close(23); close(24)

  nbad_row = 0
  call row_check()
  write(*,'(a,i0,a,i0,a,i0)') 'cases ', ncase, ' chained-mismatch ', nbad_chain, &
       ' row-mismatch ', nbad_row
  if (nbad_chain /= 0 .or. nbad_row /= 0) stop 3

contains

  subroutine read_case()
    if (allocated(t3d)) deallocate(t3d, qv, qc, qi, u, v, w, pcps, pi3, rho, dz8w, p8w)
    read(10) nz, stepcu
    read(10) dt, pgcon, xland
    allocate(t3d(nz), qv(nz), qc(nz), qi(nz), u(nz), v(nz), w(nz+1), pcps(nz), &
             pi3(nz), rho(nz), dz8w(nz), p8w(nz+1))
    read(10) t3d, qv, qc, qi, u, v, w, pcps, pi3, rho, dz8w, p8w
    read(10) cat
  end subroutine

  subroutine fill3(a, src, m)
    real, intent(out) :: a(1, nz+1, 1)
    integer, intent(in) :: m
    real, intent(in) :: src(m)
    a = 0.
    a(1, 1:m, 1) = src
  end subroutine

  subroutine one_column(ic)
    integer, intent(in) :: ic
    ! ---- stage 3: CU_SAS, one-column tile -----------------------------------
    real :: a_t(1,nz+1,1), a_qv(1,nz+1,1), a_qc(1,nz+1,1), a_qi(1,nz+1,1), &
            a_u(1,nz+1,1), a_v(1,nz+1,1), a_w(1,nz+1,1), a_p(1,nz+1,1), &
            a_pi(1,nz+1,1), a_rho(1,nz+1,1), a_dz(1,nz+1,1), a_p8w(1,nz+1,1)
    real :: rth(1,nz+1,1), rqv(1,nz+1,1), rqc(1,nz+1,1), rqi(1,nz+1,1)
    real :: ruc(1,1,nz+1), rvc(1,1,nz+1)
    real :: raincv(1,1), pratec(1,1), hbot(1,1), htop(1,1), xl(1,1)
    logical :: act(1,1)
    ! ---- stage 1/2: CU_SAS's kind_phys locals -------------------------------
    real(kind_phys) :: zi(1,nz), zl(1,nz), ps(1), rcs(1), slimsk(1), prsi(1,nz+1)
    real(kind_phys) :: prsl(1,nz), phil(1,nz), dot(1,nz), del(1,nz), u1(1,nz), &
                       v1(1,nz), q1(1,nz), t1(1,nz), ql(1,nz,2), prslk(1,nz)
    real(kind_phys) :: q1in(1,nz), t1in(1,nz), qlin(1,nz,2), u1in(1,nz), v1in(1,nz)
    real(kind_phys) :: cldwrk(1), rn(1), delt, rdelt, dpshc, pgcon_use, massf
    integer :: kbot(1), ktop(1), kcnv(1), km, kp, j
    real :: psfc, rth2(nz), rqv2(nz), rqc2(nz), rqi2(nz), raincv1

    call fill3(a_t, t3d, nz);  call fill3(a_qv, qv, nz); call fill3(a_qc, qc, nz)
    call fill3(a_qi, qi, nz);  call fill3(a_u, u, nz);   call fill3(a_v, v, nz)
    call fill3(a_w, w, nz+1);  call fill3(a_p, pcps, nz); call fill3(a_pi, pi3, nz)
    call fill3(a_rho, rho, nz); call fill3(a_dz, dz8w, nz); call fill3(a_p8w, p8w, nz+1)
    rth = 0.; rqv = 0.; rqc = 0.; rqi = 0.; ruc = 0.; rvc = 0.
    raincv = 0.; pratec = 0.; hbot = 0.; htop = 0.
    xl(1,1) = xland
    call cu_sas(DT=dt, ITIMESTEP=1, STEPCU=stepcu,                         &
         RTHCUTEN=rth, RQVCUTEN=rqv, RQCCUTEN=rqc, RQICUTEN=rqi,           &
         RUCUTEN=ruc, RVCUTEN=rvc, RAINCV=raincv, PRATEC=pratec,           &
         HTOP=htop, HBOT=hbot, U3D=a_u, V3D=a_v, W=a_w, T3D=a_t,           &
         QV3D=a_qv, QC3D=a_qc, QI3D=a_qi, PI3D=a_pi, RHO3D=a_rho,          &
         DZ8W=a_dz, PCPS=a_p, P8W=a_p8w, XLAND=xl, CU_ACT_FLAG=act,        &
         P_QC=2, PGCON=pgcon, pert_sas=.false., ens_random_seed=0,         &
         ens_sasamp=0., P_QI=3, P_FIRST_SCALAR=2,                          &
         ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz+1,                      &
         ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz+1,                      &
         its=1, ite=1, jts=1, jte=1, kts=1, kte=nz)
    write(23) nz
    write(23) rth(1,1:nz,1), rqv(1,1:nz,1), rqc(1,1:nz,1), rqi(1,1:nz,1)
    write(23) raincv(1,1), pratec(1,1), hbot(1,1), htop(1,1)
    keep(:, ic) = 0.
    keep(1:nz, ic) = rth(1,1:nz,1)
    keep(maxnz+1:maxnz+nz, ic) = rqv(1,1:nz,1)
    keep(2*maxnz+1:2*maxnz+nz, ic) = rqc(1,1:nz,1)
    keep(3*maxnz+1:3*maxnz+nz, ic) = rqi(1,1:nz,1)
    keep(4*maxnz+1:4*maxnz+4, ic) = (/ raincv(1,1), pratec(1,1), hbot(1,1), htop(1,1) /)

    ! ---- CU_SAS :312-406, verbatim on one column ----------------------------
    dpshc = 30_kind_phys
    delt = dt*stepcu
    rdelt = 1./delt
    pgcon_use = pgcon
    massf = 9e9
    psfc = p8w(1)
    i = 1
    j = 1
    zi(i,1) = 0.0
    do k = 2, nz
      km = k-1
      zi(i,k) = zi(i,km) + dz8w(km)
    end do
    do k = 2, nz
      km = k-1
      zl(i,km) = (zi(i,k)+zi(i,km))*0.5
    end do
    zl(i,nz) = 2.*zi(i,nz) - zl(i,nz-1)
    ps(i) = psfc*.001
    rcs(i) = 1.
    slimsk(i) = abs(xland-2.)
    prsi(i,1) = ps(i)
    do k = 1, nz
      kp = k+1
      prsl(i,k) = pcps(k)*.001
      phil(i,k) = zl(i,k)*grav
      dot(i,k) = -5.0e-4*grav*rho(k)*(w(k)+w(kp))
    end do
    do k = 1, nz
      del(i,k) = prsl(i,k)*grav/rd*dz8w(k)/t3d(k)
      u1(i,k) = u(k)
      v1(i,k) = v(k)
      q1(i,k) = qv(k)/(1.+qv(k))
      t1(i,k) = t3d(k)
      ql(i,k,1) = qi(k)/(1.+qi(k))
      ql(i,k,2) = qc(k)/(1.+qc(k))
      prslk(i,k) = (prsl(i,k)*.01)**rovcp
    end do
    do k = 2, nz+1
      km = k-1
      prsi(i,k) = prsi(i,km) - del(i,km)
    end do
    q1in = q1; t1in = t1; qlin = ql; u1in = u1; v1in = v1

    call sascnvn(1, 1, nz, 126, delt, del, prsl, ps, phil, ql, q1, t1, u1, v1, &
                 rcs, cldwrk, rn, kbot, ktop, kcnv, slimsk, dot, 1, pgcon_use, &
                 massf, .false., 0, 0.)
    write(21) nz
    write(21) delt, pgcon_use, massf, ps(1), slimsk(1)
    write(21) del(1,:), prsl(1,:), phil(1,:), qlin(1,:,1), qlin(1,:,2), &
              q1in(1,:), t1in(1,:), u1in(1,:), v1in(1,:), dot(1,:)
    write(21) ql(1,:,1), ql(1,:,2), q1(1,:), t1(1,:), u1(1,:), v1(1,:)
    write(21) cldwrk(1), rn(1)
    write(21) kbot(1), ktop(1), kcnv(1)

    write(22) nz, kcnv(1)
    write(22) delt, dpshc
    write(22) del(1,:), prsi(1,:), prsl(1,:), prslk(1,:), q1(1,:), t1(1,:)
    call old_arw_shalcv(1, 1, nz, delt, del, prsi, prsl, prslk, kcnv, q1, t1, dpshc)
    write(22) q1(1,:), t1(1,:)

    ! ---- chained: CU_SAS :413-496 on the staged result ----------------------
    raincv1 = rn(1)*1000./stepcu
    do k = 1, nz
      rth2(k) = (t1(1,k)-t3d(k))/pi3(k)*rdelt
      rqv2(k) = (q1(1,k)/(1.-q1(1,k))-qv(k))*rdelt
      rqc2(k) = (ql(1,k,2)/(1.-ql(1,k,2))-qc(k))*rdelt
      rqi2(k) = (ql(1,k,1)/(1.-ql(1,k,1))-qi(k))*rdelt
    end do
    n = count(transfer(rth2, 1, nz) /= transfer(rth(1,1:nz,1), 1, nz)) &
      + count(transfer(rqv2, 1, nz) /= transfer(rqv(1,1:nz,1), 1, nz)) &
      + count(transfer(rqc2, 1, nz) /= transfer(rqc(1,1:nz,1), 1, nz)) &
      + count(transfer(rqi2, 1, nz) /= transfer(rqi(1,1:nz,1), 1, nz))
    if (transfer(raincv1, 1) /= transfer(raincv(1,1), 1)) n = n + 1
    if (real(kbot(1)) /= hbot(1,1) .or. real(ktop(1)) /= htop(1,1)) n = n + 1
    if (n /= 0) then
      write(*,'(a,i0,a,i0)') 'chained mismatch case ', ic, ' words ', n
      nbad_chain = nbad_chain + 1
    end if
    ! Direct optional shallow-off driver call.
    rth = 0.; rqv = 0.; rqc = 0.; rqi = 0.; ruc = 0.; rvc = 0.
    raincv = 0.; pratec = 0.; hbot = 0.; htop = 0.
    xl(1,1) = xland
    call cu_sas(DT=dt, ITIMESTEP=1, STEPCU=stepcu,                         &
         RTHCUTEN=rth, RQVCUTEN=rqv, RQCCUTEN=rqc, RQICUTEN=rqi,           &
         RUCUTEN=ruc, RVCUTEN=rvc, RAINCV=raincv, PRATEC=pratec,           &
         HTOP=htop, HBOT=hbot, U3D=a_u, V3D=a_v, W=a_w, T3D=a_t,           &
         QV3D=a_qv, QC3D=a_qc, QI3D=a_qi, PI3D=a_pi, RHO3D=a_rho,          &
         DZ8W=a_dz, PCPS=a_p, P8W=a_p8w, XLAND=xl, CU_ACT_FLAG=act,        &
         P_QC=2, PGCON=pgcon, pert_sas=.false., ens_random_seed=0,         &
         ens_sasamp=0., P_QI=3, P_FIRST_SCALAR=2,                          &
         ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz+1,                      &
         ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz+1,                      &
         its=1, ite=1, jts=1, jte=1, kts=1, kte=nz, shalconv=0)
    write(24) nz
    write(24) rth(1,1:nz,1), rqv(1,1:nz,1), rqc(1,1:nz,1), rqi(1,1:nz,1)
    write(24) raincv(1,1), pratec(1,1), hbot(1,1), htop(1,1)


  end subroutine

  ! Every distinct level count: all its columns in ONE row of CU_SAS.
  subroutine row_check()
    integer :: nzv, m, ii, jc, idx(ncase)
    real, allocatable :: b_t(:,:,:), b_qv(:,:,:), b_qc(:,:,:), b_qi(:,:,:), &
         b_u(:,:,:), b_v(:,:,:), b_w(:,:,:), b_p(:,:,:), b_pi(:,:,:), b_rho(:,:,:), &
         b_dz(:,:,:), b_p8w(:,:,:), b_rth(:,:,:), b_rqv(:,:,:), b_rqc(:,:,:), &
         b_rqi(:,:,:), b_ruc(:,:,:), b_rvc(:,:,:), b_rain(:,:), b_prat(:,:), &
         b_hb(:,:), b_ht(:,:), b_xl(:,:)
    logical, allocatable :: b_act(:,:)
    integer :: dts, sts
    real :: dtv, pgv
    logical :: done(ncase)
    done = .false.
    do jc = 1, ncase
      if (done(jc)) cycle
      ! group: same nz, dt, stepcu and pgcon (CU_SAS takes them per call)
      rewind(10); read(10) m
      do ii = 1, jc
        call read_case()
      end do
      nzv = nz; dts = stepcu; dtv = dt; pgv = pgcon
      m = 0
      rewind(10); read(10) ii
      do ii = 1, ncase
        call read_case()
        if (.not. done(ii) .and. nz == nzv .and. stepcu == dts .and. dt == dtv &
            .and. pgcon == pgv) then
          m = m + 1; idx(m) = ii
        end if
      end do
      allocate(b_t(m,nzv+1,1), b_qv(m,nzv+1,1), b_qc(m,nzv+1,1), b_qi(m,nzv+1,1), &
               b_u(m,nzv+1,1), b_v(m,nzv+1,1), b_w(m,nzv+1,1), b_p(m,nzv+1,1), &
               b_pi(m,nzv+1,1), b_rho(m,nzv+1,1), b_dz(m,nzv+1,1), b_p8w(m,nzv+1,1), &
               b_rth(m,nzv+1,1), b_rqv(m,nzv+1,1), b_rqc(m,nzv+1,1), b_rqi(m,nzv+1,1), &
               b_ruc(m,1,nzv+1), b_rvc(m,1,nzv+1), b_rain(m,1), b_prat(m,1), &
               b_hb(m,1), b_ht(m,1), b_xl(m,1), b_act(m,1))
      b_t = 0.; b_qv = 0.; b_qc = 0.; b_qi = 0.; b_u = 0.; b_v = 0.; b_w = 0.
      b_p = 0.; b_pi = 0.; b_rho = 0.; b_dz = 0.; b_p8w = 0.
      rewind(10); read(10) ii
      jj: do ii = 1, ncase
        call read_case()
        do k = 1, m
          if (idx(k) == ii) then
            b_t(k,1:nz,1) = t3d; b_qv(k,1:nz,1) = qv; b_qc(k,1:nz,1) = qc
            b_qi(k,1:nz,1) = qi; b_u(k,1:nz,1) = u; b_v(k,1:nz,1) = v
            b_w(k,1:nz+1,1) = w; b_p(k,1:nz,1) = pcps; b_pi(k,1:nz,1) = pi3
            b_rho(k,1:nz,1) = rho; b_dz(k,1:nz,1) = dz8w; b_p8w(k,1:nz+1,1) = p8w
            b_xl(k,1) = xland
          end if
        end do
      end do jj
      b_rth = 0.; b_rqv = 0.; b_rqc = 0.; b_rqi = 0.; b_ruc = 0.; b_rvc = 0.
      b_rain = 0.; b_prat = 0.; b_hb = 0.; b_ht = 0.
      call cu_sas(DT=dtv, ITIMESTEP=1, STEPCU=dts,                              &
           RTHCUTEN=b_rth, RQVCUTEN=b_rqv, RQCCUTEN=b_rqc, RQICUTEN=b_rqi,      &
           RUCUTEN=b_ruc, RVCUTEN=b_rvc, RAINCV=b_rain, PRATEC=b_prat,          &
           HTOP=b_ht, HBOT=b_hb, U3D=b_u, V3D=b_v, W=b_w, T3D=b_t,              &
           QV3D=b_qv, QC3D=b_qc, QI3D=b_qi, PI3D=b_pi, RHO3D=b_rho,             &
           DZ8W=b_dz, PCPS=b_p, P8W=b_p8w, XLAND=b_xl, CU_ACT_FLAG=b_act,       &
           P_QC=2, PGCON=pgv, pert_sas=.false., ens_random_seed=0,              &
           ens_sasamp=0., P_QI=3, P_FIRST_SCALAR=2,                             &
           ids=1, ide=m+1, jds=1, jde=2, kds=1, kde=nzv+1,                      &
           ims=1, ime=m, jms=1, jme=1, kms=1, kme=nzv+1,                         &
           its=1, ite=m, jts=1, jte=1, kts=1, kte=nzv)
      do k = 1, m
        ii = idx(k)
        done(ii) = .true.
        n = count(transfer(b_rth(k,1:nzv,1), 1, nzv) /= transfer(keep(1:nzv,ii), 1, nzv)) &
          + count(transfer(b_rqv(k,1:nzv,1), 1, nzv) /= transfer(keep(maxnz+1:maxnz+nzv,ii), 1, nzv)) &
          + count(transfer(b_rqc(k,1:nzv,1), 1, nzv) /= transfer(keep(2*maxnz+1:2*maxnz+nzv,ii), 1, nzv)) &
          + count(transfer(b_rqi(k,1:nzv,1), 1, nzv) /= transfer(keep(3*maxnz+1:3*maxnz+nzv,ii), 1, nzv))
        if (transfer(b_rain(k,1), 1) /= transfer(keep(4*maxnz+1,ii), 1)) n = n + 1
        if (transfer(b_prat(k,1), 1) /= transfer(keep(4*maxnz+2,ii), 1)) n = n + 1
        if (b_hb(k,1) /= keep(4*maxnz+3,ii) .or. b_ht(k,1) /= keep(4*maxnz+4,ii)) n = n + 1
        if (n /= 0) then
          write(*,'(a,i0,a,i0,a,i0)') 'row mismatch case ', ii, ' row width ', m, ' words ', n
          nbad_row = nbad_row + 1
        end if
      end do
      deallocate(b_t, b_qv, b_qc, b_qi, b_u, b_v, b_w, b_p, b_pi, b_rho, b_dz, b_p8w, &
                 b_rth, b_rqv, b_rqc, b_rqi, b_ruc, b_rvc, b_rain, b_prat, b_hb, b_ht, &
                 b_xl, b_act)
    end do
  end subroutine

end program run_sas
