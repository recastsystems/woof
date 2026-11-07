program dump_morrison_aero_constants
  ! Print every module constant MORR_TWO_MOMENT_INIT_AERO computes, as the
  ! FP32 bit pattern WRF holds after the call.  Linked against a copy of
  ! module_mp_morr_two_moment_aero.F whose only change is that the PRIVATE
  ! visibility attributes are removed (build.sh makes that copy with sed and
  ! checks that the copy differs from the pinned source in those lines only),
  ! so the init arithmetic is the pinned WRF statement sequence.  The GPU
  ! port reads these words from gpuwm/data/morrison/constants.toml; the
  ! column oracle then checks them end to end.
  !
  ! usage: dump_constants RIMED > constants-RIMED.txt
  use module_model_constants, only: cp, g, r_d, r_v, ep_2, t0, svp1, svp2, svp3
  use module_mp_morr_two_moment_aero
  implicit none
  character(len=32) :: arg
  integer :: rimed, m, s

  call get_command_argument(1, arg)
  read(arg, *) rimed
  call morr_two_moment_init_aero(rimed)

  call p('CP', cp); call p('G', g); call p('R', r_d); call p('RV', r_v)
  call p('EP_2', ep_2); call p('T0', t0)
  call p('SVP1', svp1); call p('SVP2', svp2); call p('SVP3', svp3)
  call p('PI', pi); call p('XXX', xxx); call p('REAL_TINY', real_tiny)
  call p('NDCNST', ndcnst)
  call p('AI', ai); call p('AC', ac); call p('AS', as); call p('AR', ar)
  call p('AG', ag); call p('BI', bi); call p('BC', bc); call p('BS', bs)
  call p('BR', br); call p('BG', bg)
  call p('RHOSU', rhosu); call p('RHOW', rhow); call p('RHOI', rhoi)
  call p('RHOSN', rhosn); call p('RHOG', rhog); call p('AIMM', aimm)
  call p('BIMM', bimm); call p('ECR', ecr); call p('DCS', dcs)
  call p('MI0', mi0); call p('MG0', mg0); call p('F1S', f1s)
  call p('F2S', f2s); call p('F1R', f1r); call p('F2R', f2r)
  call p('QSMALL', qsmall); call p('CI', ci); call p('DI', di)
  call p('CS', cs); call p('DS', ds); call p('CG', cg); call p('DG', dg)
  call p('EII', eii); call p('ECI', eci); call p('RIN', rin)
  call p('CPW', cpw); call p('C1', c1); call p('K1', k1); call p('MW', mw)
  call p('OSM', osm); call p('VI', vi); call p('EPSM', epsm)
  call p('RHOA', rhoa); call p('MAP', map); call p('MA', ma)
  call p('RR', rr); call p('BACT', bact); call p('RM1', rm1)
  call p('RM2', rm2); call p('NANEW1', nanew1); call p('NANEW2', nanew2)
  call p('SIG1', sig1); call p('SIG2', sig2); call p('F11', f11)
  call p('F12', f12); call p('F21', f21); call p('F22', f22)
  call p('MMULT', mmult)
  call p('LAMMAXI', lammaxi); call p('LAMMINI', lammini)
  call p('LAMMAXR', lammaxr); call p('LAMMINR', lamminr)
  call p('LAMMAXS', lammaxs); call p('LAMMINS', lammins)
  call p('LAMMAXG', lammaxg); call p('LAMMING', lamming)
  call p('CONS1', cons1); call p('CONS2', cons2); call p('CONS3', cons3)
  call p('CONS4', cons4); call p('CONS5', cons5); call p('CONS6', cons6)
  call p('CONS7', cons7); call p('CONS8', cons8); call p('CONS9', cons9)
  call p('CONS10', cons10); call p('CONS11', cons11)
  call p('CONS12', cons12); call p('CONS13', cons13)
  call p('CONS14', cons14); call p('CONS15', cons15)
  call p('CONS16', cons16); call p('CONS17', cons17)
  call p('CONS18', cons18); call p('CONS19', cons19)
  call p('CONS20', cons20); call p('CONS21', cons21)
  call p('CONS22', cons22); call p('CONS23', cons23)
  call p('CONS24', cons24); call p('CONS25', cons25)
  call p('CONS26', cons26); call p('CONS27', cons27)
  call p('CONS28', cons28); call p('CONS29', cons29)
  call p('CONS30', cons30); call p('CONS31', cons31)
  call p('CONS32', cons32); call p('CONS33', cons33)
  call p('CONS34', cons34); call p('CONS35', cons35)
  call p('CONS36', cons36); call p('CONS37', cons37)
  call p('CONS38', cons38); call p('CONS39', cons39)
  call p('CONS40', cons40); call p('CONS41', cons41)
  call p('ATEN', aten_pamdm); call p('ALOGATEN', alogaten_pamdm)
  call p('ALOG2', alog2_pamdm); call p('ALOG3', alog3_pamdm)
  call p('SURFTEN', surften_pamdm); call p('THIRD', third_pamdm)
  call p('SQ2', sq2_pamdm)
  do m = 1, naer_cu
    call pi1('ALOGSIG', m, alogsig_pamdm(m))
    call pi1('EXP45LOGSIG', m, exp45logsig_pamdm(m))
    call pi1('ARGFACTOR', m, argfactor_pamdm(m))
    call pi1('AMCUBE', m, amcube_pamdm(m))
    call pi1('SMCRIT', m, smcrit_pamdm(m))
    call pi1('LNSM', m, lnsm_pamdm(m))
    call pi1('AMCUBEFACTOR', m, amcubefactor_pamdm(m))
    call pi1('SMCRITFACTOR', m, smcritfactor_pamdm(m))
    call pi1('F1AER', m, f1_pamdm(m))
    call pi1('F2AER', m, f2_pamdm(m))
    call pi1('DRYRAD', m, dryrad_aer(m))
    call pi1('DENSITY', m, density_aer(m))
    call pi1('HYGRO', m, hygro_aer(m))
    call pi1('DISPERSION', m, dispersion_aer(m))
    call pi1('NUM_TO_MASS', m, num_to_mass_aer(m))
    do s = 1, psat_pamdm
      write(*, '(A,I0,A,I0,A,Z8.8)') 'CCNFACT_', s, '_', m, ' ', &
        transfer(ccnfact_pamdm(s, m), 0)
    end do
  end do
  do s = 1, psat_pamdm
    call pi1('SUPER', s, super_pamdm(s))
  end do

contains

  subroutine p(name, value)
    character(len=*), intent(in) :: name
    real, intent(in) :: value
    write(*, '(A,1X,Z8.8)') name, transfer(value, 0)
  end subroutine p

  subroutine pi1(name, idx, value)
    character(len=*), intent(in) :: name
    integer, intent(in) :: idx
    real, intent(in) :: value
    write(*, '(A,"_",I0,1X,Z8.8)') name, idx, transfer(value, 0)
  end subroutine pi1

end program dump_morrison_aero_constants
