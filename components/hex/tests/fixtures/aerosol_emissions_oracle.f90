! Extracted assignments: MPAS-A v8.4.2 mpas_atmphys_init_microphysics.F:239-245.
! Standalone test wrapper added 2026-10-28. Original source has the MPAS BSD licence.
subroutine emissions(n,nwfa,rhozz,zz,z0,z1,area,result) bind(C)
 use iso_c_binding
 implicit none
 integer(c_int),value::n
 real(c_float),intent(in)::nwfa(n),rhozz(n),zz(n),z0(n),z1(n),area(n)
 real(c_float),intent(out)::result(n)
 real(c_float)::airmass
 integer::iCell
 do iCell=1,n
  airmass=rhozz(iCell)*zz(iCell)
  airmass=airmass*(z1(iCell)-z0(iCell))*area(iCell)
  result(iCell)=nwfa(iCell)*0.000196*airmass*0.5e-10
 enddo
end subroutine
