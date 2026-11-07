! Hail-aware (mp_physics=38) sibling of generate_tables.F90.
!
! WRF v4.6.1 decides the graupel-density table shape from ONE thing: whether
! thompson_init is handed the optional graupel number ``ng``
! (module_mp_thompson.F:459-466).  Present, is_hail_aware becomes true,
! dimNRHG = NRHG = 9 and qr_acr_qg (:4091) computes and writes
! qr_acr_qg_mp38V1.dat with one slab per graupel density rho_g(1:9);
! absent, it is the one-slab qr_acr_qg_V4.dat of mp=8/28.  Nothing else in
! the table set changes: qr_acr_qsV2.dat and freezeH2O.dat do not read a
! graupel coefficient, and build_hail.sh verifies both against the classic
! pins so a reader can see that.
!
! Linked against stub_wrf_hail.F90, whose write_thompson_mp38table switch is
! true; with the classic stub WRF refuses to compute the mp=38 table at all
! (:4163-4165).
program generate_thompson_hail_tables
  use module_mp_thompson, only: thompson_init, tps_iaus, tni_iaus, tpi_ide, t_Efrw, t_Efsw, tpc_wev, tnc_wev
  implicit none
  real :: hgt(2,2,2), ng(2,2,2)

  hgt = 0.0
  hgt(:,1,:) = 100.0
  hgt(:,2,:) = 200.0
  ng = 0.0
  call thompson_init(                                                &
       hgt=hgt, ng=ng,                                               &
       ids=1, ide=3, jds=1, jde=3, kds=1, kde=2,                    &
       ims=1, ime=2, jms=1, jme=2, kms=1, kme=2,                    &
       its=1, ite=2, jts=1, jte=2, kts=1, kte=2)
  open(71, file='thompson_aux_tables.dat', form='unformatted', status='replace')
  write(71) tps_iaus
  write(71) tni_iaus
  write(71) tpi_ide
  write(71) t_Efrw
  write(71) t_Efsw
  write(71) tpc_wev
  write(71) tnc_wev
  close(71)
  print '(A)', 'THOMPSON_HAIL_TABLE_GENERATION_COMPLETE'
end program generate_thompson_hail_tables
