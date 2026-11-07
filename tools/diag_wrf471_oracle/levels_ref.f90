! ABI adapter only. The pinned WRF modules are compiled without modification.
! Common native arrays have (nx+1,nz+1,ny+1) Fortran storage, with unused
! staggered/halo slots provided by the caller. Outputs retain those halo slots.
subroutine levels_ref(kind,nx,ny,nz,nl,fields,terrain,levels,pressure_kind,extrap,missing,axis,out) bind(C)
  use iso_c_binding
  use module_diag_pld, only: pld
  use module_diag_zld, only: zld
  implicit none
  integer(c_int),value :: kind,nx,ny,nz,nl,pressure_kind,extrap
  real(c_float),value :: missing
  real(c_float),intent(in) :: fields(nx+1,nz+1,ny+1,11),terrain(nx+1,ny+1),levels(nl)
  real(c_float),intent(out) :: axis(nl),out(nx+1,nl,ny+1,9)
  real(c_float) :: one(nx+1,ny+1)
  one=1.
  out=missing
  if(kind==1) then
    call pld(fields(:,:,:,1),fields(:,:,:,2),fields(:,:,:,3), &
      fields(:,:,:,4),fields(:,:,:,5),fields(:,:,:,6),fields(:,:,:,7), &
      fields(:,:,:,8),fields(:,:,:,9),fields(:,:,:,10),fields(:,:,:,11), &
      one,one,one,one,one,one,one,one,pressure_kind,extrap,missing,nl,nl,levels, &
      axis,out(:,:,:,1),out(:,:,:,2),out(:,:,:,3),out(:,:,:,4), &
      out(:,:,:,5),out(:,:,:,6),out(:,:,:,7),out(:,:,:,8), &
      1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz+1)
  else
    call zld(fields(:,:,:,1),fields(:,:,:,2),fields(:,:,:,3), &
      fields(:,:,:,4),fields(:,:,:,5),fields(:,:,:,6),fields(:,:,:,7), &
      fields(:,:,:,8),fields(:,:,:,9),fields(:,:,:,10),fields(:,:,:,11), &
      one,one,one,one,one,one,one,one,terrain,pressure_kind,extrap,missing,nl,nl,levels, &
      axis,out(:,:,:,1),out(:,:,:,2),out(:,:,:,3),out(:,:,:,4), &
      out(:,:,:,5),out(:,:,:,6),out(:,:,:,7),out(:,:,:,8),out(:,:,:,9), &
      1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz+1)
  endif
end subroutine
