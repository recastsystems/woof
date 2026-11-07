! Column driver for the WRF v4.6.1 Goddard GCE microphysics oracle
! (mp_physics = 97, phys/module_mp_gsfcgce.F).
!
!   run_gsfcgce <in.bin> <out.bin> <ncol> <nk> <nsteps> <dt> <ihail> <ice2>
!
! The columns are laid out as WRF's i dimension (one j row), levels as k,
! and gsfcgce is called exactly as module_microphysics_driver.F:2030-2053
! calls it, once per step for nsteps steps, with diagflag and
! do_radar_ref on so REFL_10CM is produced every step.
!
! Input  (little-endian float32, Fortran order (ncol,nk) per 3-D field):
!   th qv qc qr qi qs qg rho pii p z dz8w   then (ncol) ht rainnc snownc graupelnc
! Output (nsteps+1 frames; frame 0 is the input state):
!   th qv qc qr qi qs qg refl   then (ncol) rainnc rainncv snownc snowncv
!   graupelnc graupelncv sr
!
! One process per (ihail, ice2) combination: gsfcgce initializes the radar
! module once per process (its SAVEd NCALL), from the first call's ihail.
program run_gsfcgce
  use module_mp_gsfcgce, only: gsfcgce
  implicit none
  integer :: ncol, nk, nsteps, ihail, ice2, it, u, v
  real :: dt
  real, parameter :: grav = 9.81, rhowater = 1000., rhosnow = 100.
  real, allocatable, dimension(:,:,:) :: th, qv, qc, qr, qi, qs, qg, &
       rho, pii, p, z, dz8w, refl
  real, allocatable, dimension(:,:) :: ht, rainnc, rainncv, snownc, &
       snowncv, graupelnc, graupelncv, sr
  character(len=1024) :: fin, fout, arg
  logical :: diagflag
  integer :: do_radar_ref

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, arg); read(arg, *) ncol
  call get_command_argument(4, arg); read(arg, *) nk
  call get_command_argument(5, arg); read(arg, *) nsteps
  call get_command_argument(6, arg); read(arg, *) dt
  call get_command_argument(7, arg); read(arg, *) ihail
  call get_command_argument(8, arg); read(arg, *) ice2

  allocate(th(ncol,nk,1), qv(ncol,nk,1), qc(ncol,nk,1), qr(ncol,nk,1), &
           qi(ncol,nk,1), qs(ncol,nk,1), qg(ncol,nk,1), rho(ncol,nk,1), &
           pii(ncol,nk,1), p(ncol,nk,1), z(ncol,nk,1), dz8w(ncol,nk,1), &
           refl(ncol,nk,1))
  allocate(ht(ncol,1), rainnc(ncol,1), rainncv(ncol,1), snownc(ncol,1), &
           snowncv(ncol,1), graupelnc(ncol,1), graupelncv(ncol,1), sr(ncol,1))

  open(newunit=u, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  read(u) th, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz8w
  read(u) ht, rainnc, snownc, graupelnc
  close(u)
  rainncv = 0.; snowncv = 0.; graupelncv = 0.; sr = 0.; refl = 0.

  open(newunit=v, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write')
  call dump()
  diagflag = .true.
  do_radar_ref = 1
  do it = 1, nsteps
     call gsfcgce(TH=th, QV=qv, QL=qc, QR=qr, QI=qi, QS=qs,              &
          RHO=rho, PII=pii, P=p, DT_IN=dt, Z=z,                            &
          HT=ht, DZ8W=dz8w, GRAV=grav,                                     &
          RHOWATER=rhowater, RHOSNOW=rhosnow,                              &
          ITIMESTEP=it,                                                    &
          IDS=1, IDE=ncol+1, JDS=1, JDE=2, KDS=1, KDE=nk+1,                &
          IMS=1, IME=ncol, JMS=1, JME=1, KMS=1, KME=nk,                    &
          ITS=1, ITE=ncol, JTS=1, JTE=1, KTS=1, KTE=nk,                    &
          RAINNC=rainnc, RAINNCV=rainncv,                                  &
          SNOWNC=snownc, SNOWNCV=snowncv, SR=sr,                           &
          GRAUPELNC=graupelnc, GRAUPELNCV=graupelncv,                      &
          REFL_10CM=refl, DIAGFLAG=diagflag, DO_RADAR_REF=do_radar_ref,    &
          F_QG=.true., QG=qg, IHAIL=ihail, ICE2=ice2)
     call dump()
  end do
  close(v)

contains

  subroutine dump()
    write(v) th, qv, qc, qr, qi, qs, qg, refl
    write(v) rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr
  end subroutine dump

end program run_gsfcgce
