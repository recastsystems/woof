! Column driver for WRF v4.6.1 Ferrier-Aligo microphysics, advected-rime
! variant (mp_physics=15, package fer_mp_hires_advect,
! phys/module_mp_fer_hires.F subroutine FER_HIRES_ADVECT, byte-unmodified).
!
! For every column in columns.in it does what WRF does for a run at that
! column's time step: fer_hires_init (GSMDT=0, so DTPH=DT; reads
! ETAMPNEW_DATA.expanded_rain, builds the saturation tables), then
! FER_HIRES_ADVECT once per step for NSTEPS steps, carrying exactly the state
! WRF carries for this package between calls: theta, qv, qc, qr, qi, the
! advected scalar QRIMEF (rime factor times qi) and RAINNC.  QT, F_ICE,
! F_RAIN and F_RIMEF are locals of FER_HIRES_ADVECT, rebuilt every call
! (F_RIMEF = QRIMEF/QI where QI > EPSQ), so nothing else is carried.
!
! The patch is one point wide (its=ite=1, jts=jte=1), kts=1..kte=NZ with
! kde=NZ+1, as WRF's microphysics driver passes it.
!
! Outputs (floats as binary32 bit patterns, hex):
!   levels.csv  : col,step,k,th,qv,qc,qr,qi,qrimef
!   surface.csv : col,step,rainnc,rainncv,sr
! for step 1 and step NSTEPS.
!
! The main program must be compiled with -fconvert=big-endian: WRF reads
! the unformatted table with that runtime conversion.
program run_ferrier_advect
  use module_mp_fer_hires, only: fer_hires_advect, fer_hires_init
  implicit none

  integer :: ncol, nz, ic, k, step, nsteps, gid, iu, ul, us
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: hdt, hdx, hrain
  integer :: w(10)
  real :: dt, dx
  real, allocatable, dimension(:,:,:) :: p, pi, th, qv, qc, qr, qi, &
       qrimef, dz8w, rho
  real, dimension(1,1) :: rainnc, rainncv, sr
  integer, dimension(1,1) :: lowlyr

  if (kind(1.0) /= 4) stop 'default REAL is not binary32'

  open(newunit=iu, file='columns.in', status='old', action='read', &
       form='formatted')
  read(iu, *) ncol, nz

  ids = 1; ide = 2; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz

  allocate(p(1,kms:kme,1), pi(1,kms:kme,1), th(1,kms:kme,1), &
       qv(1,kms:kme,1), qc(1,kms:kme,1), qr(1,kms:kme,1), &
       qi(1,kms:kme,1), qrimef(1,kms:kme,1), dz8w(1,kms:kme,1), &
       rho(1,kms:kme,1))

  open(newunit=ul, file='levels.csv', status='replace', action='write')
  open(newunit=us, file='surface.csv', status='replace', action='write')
  write(ul, '(A)') 'col,step,k,th,qv,qc,qr,qi,qrimef'
  write(us, '(A)') 'col,step,rainnc,rainncv,sr'

  do ic = 0, ncol - 1
     read(iu, '(I6,1X,I6,3(1X,Z8))') gid, nsteps, hdt, hdx, hrain
     dt = transfer(hdt, dt)
     dx = transfer(hdx, dx)
     rainnc(1,1) = transfer(hrain, dx)
     rainncv = 0.0
     sr = 0.0
     p = 0.0; pi = 1.0; th = 0.0; qv = 0.0; qc = 0.0; qr = 0.0
     qi = 0.0; qrimef = 0.0; dz8w = 0.0; rho = 0.0
     do k = 1, nz
        read(iu, '(10(Z8,1X))') w
        p(1,k,1)      = transfer(w(1), dt)
        pi(1,k,1)     = transfer(w(2), dt)
        th(1,k,1)     = transfer(w(3), dt)
        qv(1,k,1)     = transfer(w(4), dt)
        qc(1,k,1)     = transfer(w(5), dt)
        qr(1,k,1)     = transfer(w(6), dt)
        qi(1,k,1)     = transfer(w(7), dt)
        qrimef(1,k,1) = transfer(w(8), dt)
        dz8w(1,k,1)   = transfer(w(9), dt)
        rho(1,k,1)    = transfer(w(10), dt)
     end do

     ! WRF calls the init once per run with the model time step
     ! (module_physics_init.F:4499 runs fer_hires_init for both
     ! FER_MP_HIRES and FER_MP_HIRES_ADVECT).  The optional F_* arrays are
     ! not passed: the advected package carries no F_* state at all.
     call fer_hires_init(0.0, dt, dx, dx, lowlyr, .false., .true., &
          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
          its, ite, jts, jte, kts, kte)

     do step = 1, nsteps
        call fer_hires_advect(step, dt, dx, dx, gid, rainnc, rainncv, &
             dz8w, rho, p, pi, th, qv, lowlyr, sr, &
             qc, qr, qi, qrimef, &
             ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
             its, ite, jts, jte, kts, kte)
        if (step == 1 .or. step == nsteps) then
           do k = 1, nz
              write(ul, '(I0,",",I0,",",I0,6(",",Z8.8))') ic, step, k, &
                   th(1,k,1), qv(1,k,1), qc(1,k,1), qr(1,k,1), &
                   qi(1,k,1), qrimef(1,k,1)
           end do
           write(us, '(I0,",",I0,3(",",Z8.8))') ic, step, rainnc(1,1), &
                rainncv(1,1), sr(1,1)
        end if
     end do
  end do
  close(iu); close(ul); close(us)
end program run_ferrier_advect
