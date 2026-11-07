! Driver only. Link the unchanged, licence-reviewed WRF V3.5 modules.
program ideal_real_capture
 use module_wind_generic
 use module_wind_fitch
 implicit none
 integer nx,ny,nz,nt,t,k,unit
 integer gi
 real meta(3),row(10)
 real,allocatable :: phb(:,:,:),u(:,:,:),v(:,:,:),dz(:,:,:),qke(:,:,:),qa(:,:,:),du(:,:,:),dv(:,:,:)
 real,allocatable :: latu(:,:),lonu(:,:),latv(:,:),lonv(:,:)
 open(newunit=unit,file='legacy-native.in',access='stream',form='unformatted')
 read(unit) nx,ny,nz,nt
 read(unit) meta
 do t=1,nt
  read(unit) gi,row
 enddo
 allocate(phb(nx+1,nz+1,ny+1),u(nx+1,nz+1,ny+1),v(nx+1,nz+1,ny+1),dz(nx+1,nz+1,ny+1))
 allocate(qke(nx+1,nz+1,ny+1),qa(nx+1,nz+1,ny+1),du(nx+1,nz+1,ny+1),dv(nx+1,nz+1,ny+1))
 allocate(latu(nx+1,ny+1),lonu(nx+1,ny+1),latv(nx+1,ny+1),lonv(nx+1,ny+1))
 phb=0.;u=0.;v=0.;dz=0.;qke=0.;qa=0.;du=0.;dv=0.;latu=0.;lonu=0.;latv=0.;lonv=0.
 read(unit) u(1:nx,1:nz,1:ny),v(1:nx,1:nz,1:ny),dz(1:nx,1:nz,1:ny)
 read(unit) phb(1:nx,1:nz+1,1:ny)
 read(unit) du(1:nx,1:nz,1:ny),dv(1:nx,1:nz,1:ny),qke(1:nx,1:nz,1:ny),qa(1:nx,1:nz,1:ny)
 read(unit) latu(1:nx+1,1:ny),lonu(1:nx+1,1:ny),latv(1:nx,1:ny+1),lonv(1:nx,1:ny+1)
 close(unit)
 call init_module_wind_generic
 call init_module_wind_fitch
 call turbine_drag(1,phb,u,v,latu,lonu,latv,lonv,meta(1),dz,meta(2),qke,qa,.true.,du,dv, &
  1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz)
 open(newunit=unit,file='ideal-native.out',access='stream',form='unformatted',status='replace')
 write(unit) du(1:nx,1:nz,1:ny),dv(1:nx,1:nz,1:ny),qke(1:nx,1:nz,1:ny),qa(1:nx,1:nz,1:ny)
 close(unit)
 open(newunit=unit,file='ideal-native.geometry',access='stream',form='unformatted',status='replace')
 write(unit) nwindturbines
 do t=1,nwindturbines
  write(unit) windturbines(t)
 enddo
 close(unit)
end program
logical function wrf_dm_on_monitor()
 wrf_dm_on_monitor=.true.
end function
subroutine wrf_dm_bcast_integer(x,n)
 integer x,n
end subroutine
subroutine wrf_dm_bcast_real(x,n)
 real x
 integer n
end subroutine
subroutine wrf_message(message)
 character(*) message
end subroutine
subroutine wrf_error_fatal(message)
 character(*) message
 print *,message
 stop 3
end subroutine
subroutine nl_get_windturbines_spec(id,value)
 integer id
 character(*) value
 value='ideal'
end subroutine
subroutine nl_get_td_turbgridid(id,value)
 integer id,value
 value=1
end subroutine
subroutine nl_get_td_ewfx(id,value)
 integer id,value
 value=2
end subroutine
subroutine nl_get_td_ewfy(id,value)
 integer id,value
 value=2
end subroutine
subroutine nl_get_td_pwfx(id,value)
 integer id,value
 value=18
end subroutine
subroutine nl_get_td_pwfy(id,value)
 integer id,value
 value=22
end subroutine
subroutine nl_get_td_turbpercell(id,value)
 integer id
 real value
 value=2.5
end subroutine
subroutine nl_get_td_hubheight(id,value)
 integer id
 real value
 value=100.
end subroutine
subroutine nl_get_td_diameter(id,value)
 integer id
 real value
 value=60.
end subroutine
subroutine nl_get_td_power(id,value)
 integer id
 real value
 value=2.
end subroutine
subroutine nl_get_td_cutinspeed(id,value)
 integer id
 real value
 value=4.
end subroutine
subroutine nl_get_td_cutoutspeed(id,value)
 integer id
 real value
 value=27.
end subroutine
subroutine nl_get_td_stdthrcoef(id,value)
 integer id
 real value
 value=.158
end subroutine
