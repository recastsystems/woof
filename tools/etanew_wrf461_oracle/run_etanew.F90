! Column driver for WRF v4.6.1 Eta Ferrier microphysics (mp_physics=95,
! ETAMPNEW, phys/module_mp_etanew.F, byte-unmodified).
!
! For every column in columns.in it does what WRF does for a run at that
! column's time step: ETANEWinit (GSMDT=0, so DTPH=DT; restart=.false.; reads
! ETAMPNEW_DATA.expanded_rain, builds the saturation tables and fills
! MP_RESTART_STATE / TBPVS_STATE / TBPVS0_STATE), then ETAMP_NEW once per step
! for NSTEPS steps, carrying the state (theta, qv, qt, F_ICE_PHY, F_RAIN_PHY,
! F_RIMEF_PHY, RAINNC and the restart vectors) from step to step exactly as
! WRF's state arrays would.  qc/qr/qs are outputs only: ETAMP_NEW rebuilds
! them from qt and the fractions on exit.  qt is carried unchanged between
! steps (no advection in a column).
!
! ETANEWinit with restart=.false. resets F_ICE/F_RAIN/F_RIMEF to 0/0/1; the
! column's own fractions are installed after the init so the carried state
! a column taken mid-run would hold is what the first step sees.
!
! The patch is one point wide (its=ite=1, jts=jte=1), kts=1..kte=NZ with
! kde=NZ+1, as WRF's microphysics driver passes it.
!
! Outputs (floats as binary32 bit patterns, hex), every step:
!   levels.csv  : col,step,k,th,qv,qt,qc,qr,qs,f_ice,f_rain,f_rimef
!   surface.csv : col,step,rainnc,rainncv,sr
!   init.csv    : col,name,index,value   (MP_RESTART_STATE 1..43 and the
!                 two 7501-entry saturation tables, for the first column of
!                 each distinct time step)
!
! The main program must be compiled with -fconvert=big-endian: WRF reads
! the unformatted table with that runtime conversion.
program run_etanew
  use module_mp_etanew, only: etamp_new, etanewinit, nx
  implicit none

  integer, parameter :: nstate = 43
  integer :: ncol, nz, ic, k, step, nsteps, iu, ul, us, ui, n, j
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: hdt, hdx, hrain
  integer :: w(10)
  real :: dt, dx
  real, allocatable, dimension(:,:,:) :: p, pi, th, qv, qt, qc, qr, qs, &
       dz8w, rho, f_ice, f_rain, f_rimef, in_ice, in_rain, in_rimef
  real, dimension(1,1) :: rainnc, rainncv, sr
  integer, dimension(1,1) :: lowlyr
  real :: mp_restart_state(nstate), tbpvs_state(nx), tbpvs0_state(nx)
  real :: dumped(16)
  integer :: ndumped
  logical :: seen

  if (kind(1.0) /= 4) stop 'default REAL is not binary32'

  open(newunit=iu, file='columns.in', status='old', action='read', &
       form='formatted')
  read(iu, *) ncol, nz

  ids = 1; ide = 2; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz

  allocate(p(1,kms:kme,1), pi(1,kms:kme,1), th(1,kms:kme,1), &
       qv(1,kms:kme,1), qt(1,kms:kme,1), qc(1,kms:kme,1), &
       qr(1,kms:kme,1), qs(1,kms:kme,1), dz8w(1,kms:kme,1), &
       rho(1,kms:kme,1), f_ice(1,kms:kme,1), f_rain(1,kms:kme,1), &
       f_rimef(1,kms:kme,1), in_ice(1,kms:kme,1), in_rain(1,kms:kme,1), &
       in_rimef(1,kms:kme,1))

  open(newunit=ul, file='levels.csv', status='replace', action='write')
  open(newunit=us, file='surface.csv', status='replace', action='write')
  open(newunit=ui, file='init.csv', status='replace', action='write')
  write(ul, '(A)') 'col,step,k,th,qv,qt,qc,qr,qs,f_ice,f_rain,f_rimef'
  write(us, '(A)') 'col,step,rainnc,rainncv,sr'
  write(ui, '(A)') 'col,name,index,value'
  ndumped = 0

  do ic = 0, ncol - 1
     read(iu, '(I6,3(1X,Z8))') nsteps, hdt, hdx, hrain
     dt = transfer(hdt, dt)
     dx = transfer(hdx, dx)
     rainnc(1,1) = transfer(hrain, dx)
     rainncv = 0.0
     sr = 0.0
     p = 0.0; pi = 1.0; th = 0.0; qv = 0.0; qt = 0.0; qc = 0.0; qr = 0.0
     qs = 0.0; dz8w = 0.0; rho = 0.0; f_ice = 0.0; f_rain = 0.0
     f_rimef = 1.0; in_ice = 0.0; in_rain = 0.0; in_rimef = 1.0
     do k = 1, nz
        read(iu, '(10(Z8,1X))') w
        p(1,k,1)        = transfer(w(1), dt)
        pi(1,k,1)       = transfer(w(2), dt)
        th(1,k,1)       = transfer(w(3), dt)
        qv(1,k,1)       = transfer(w(4), dt)
        qt(1,k,1)       = transfer(w(5), dt)
        dz8w(1,k,1)     = transfer(w(6), dt)
        rho(1,k,1)      = transfer(w(7), dt)
        in_ice(1,k,1)   = transfer(w(8), dt)
        in_rain(1,k,1)  = transfer(w(9), dt)
        in_rimef(1,k,1) = transfer(w(10), dt)
     end do

     mp_restart_state = 0.0
     tbpvs_state = 0.0
     tbpvs0_state = 0.0
     call etanewinit(0.0, dt, dx, dx, lowlyr, .false., &
          f_ice, f_rain, f_rimef, &
          mp_restart_state, tbpvs_state, tbpvs0_state, .true., &
          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
          its, ite, jts, jte, kts, kte)
     f_ice = in_ice
     f_rain = in_rain
     f_rimef = in_rimef

     seen = .false.
     do j = 1, ndumped
        if (dumped(j) == dt) seen = .true.
     end do
     if (.not. seen) then
        ndumped = ndumped + 1
        dumped(ndumped) = dt
        do n = 1, nstate
           write(ui, '(I0,",mp_restart_state,",I0,",",Z8.8)') ic, n, &
                mp_restart_state(n)
        end do
        do n = 1, nx
           write(ui, '(I0,",tbpvs,",I0,",",Z8.8)') ic, n, tbpvs_state(n)
        end do
        do n = 1, nx
           write(ui, '(I0,",tbpvs0,",I0,",",Z8.8)') ic, n, tbpvs0_state(n)
        end do
     end if

     do step = 1, nsteps
        call etamp_new(step, dt, dx, dx, &
             dz8w, rho, p, pi, th, qv, qt, &
             lowlyr, sr, f_ice, f_rain, f_rimef, &
             qc, qr, qs, &
             mp_restart_state, tbpvs_state, tbpvs0_state, &
             rainnc, rainncv, &
             ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
             its, ite, jts, jte, kts, kte)
        do k = 1, nz
           write(ul, '(I0,",",I0,",",I0,9(",",Z8.8))') ic, step, k, &
                th(1,k,1), qv(1,k,1), qt(1,k,1), qc(1,k,1), qr(1,k,1), &
                qs(1,k,1), f_ice(1,k,1), f_rain(1,k,1), f_rimef(1,k,1)
        end do
        write(us, '(I0,",",I0,3(",",Z8.8))') ic, step, rainnc(1,1), &
             rainncv(1,1), sr(1,1)
     end do
  end do
  close(iu); close(ul); close(us); close(ui)
end program run_etanew
