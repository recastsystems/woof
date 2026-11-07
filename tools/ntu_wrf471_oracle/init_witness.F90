! An analytic initialization diagnostic, not a captured native forecast.
! Every caller-supplied array is filled. The unmodified module's automatic
! RHO/P_PHY/DZ8W arrays retain their unfilled top level. The compiler's
! signaling-NaN initialization makes the first use observable.
program init_witness
  use module_mp_ntu, only: ntu_init
  implicit none
  integer, parameter :: nz=12, nk=nz+1
  real :: phb(1,nk,1), ph(1,nk,1), p(1,nk,1), pb(1,nk,1)
  real :: alt(1,nk,1), qv(1,nk,1), qdcn(1,nk,1), qtcn(1,nk,1)
  real :: qccn(1,nk,1), qrcn(1,nk,1), qnin(1,nk,1), xland(1,1)
  integer :: k
  ph=0.; pb=0.; alt=1.; qv=0.005
  qdcn=0.; qtcn=0.; qccn=0.; qrcn=0.; qnin=0.; xland=1.
  do k=1,nk
     phb(1,k,1)=9.8*500.*real(k-1)
     p(1,k,1)=100000.*exp(-500.*real(k-1)/8000.)
  enddo
  call ntu_init(phb,ph,p,pb,alt,qv,qdcn,qtcn,qccn,qrcn,qnin,xland,2,.false., &
       1,2,1,2,1,nk,1,1,1,1,1,nk,1,1,1,1,1,nk)
  print *, 'NTU_INIT returned'
end program init_witness
