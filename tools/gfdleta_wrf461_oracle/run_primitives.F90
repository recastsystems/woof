program run_primitives
 use iso_fortran_env, only:int32
 use module_ra_gfdleta, only: gfdletainit,gausin,solard,r1,clo89
 use module_configure
 implicit none
 integer,parameter::nc=64,n=12,nv=n+1
 type(grid_config_rec_type)::cfg
 real::sf(nv),sh(nv),em(nc,1),ca(nc,nv),fac(nc,nv,nv)
 integer::counts(nc),kt(nc,nv),kb(nc,nv)
 integer::i,j,k,q,u,v,year,month,day,hour
 real::x
 sf=0.;sh=0.
 call gfdletainit(em,sf,sh,1.,2001,3,21,12.,cfg,.false., &
 1,nc+1,1,2,1,nv, 1,nc,1,1,1,nv, 1,nc,1,1,1,n)
 open(newunit=u,file='primitives-inputs.csv',status='replace')
 open(newunit=v,file='primitives-expected.csv',status='replace')
 write(u,'(A)')'field,column,index,bits'
 write(v,'(A)')'field,column,index,bits'
 do i=1,310
  x=real(i)*0.01
  write(u,'(A,",",I0,",0,",Z8.8)')'GAUSIN',i,transfer(x,0_int32)
  write(v,'(A,",",I0,",0,",Z8.8)')'GAUSIN',i,transfer(gausin(x),0_int32)
 enddo
 do i=1,nc
  year=1901+mod(i*17,198);month=1+mod(i*7,12);day=1+mod(i*11,28);hour=mod(i*5,24)
  write(u,'(A,",",I0,",0,",Z8.8)')'YEAR',i,year
  write(u,'(A,",",I0,",0,",Z8.8)')'MONTH',i,month
  write(u,'(A,",",I0,",0,",Z8.8)')'DAY',i,day
  write(u,'(A,",",I0,",0,",Z8.8)')'HOUR',i,hour
  call solard(hour,day,month,year)
  write(v,'(A,",",I0,",0,",Z8.8)')'SOLARD',i,transfer(r1,0_int32)
 enddo
 ca=0.;kt=0;kb=0
 do i=1,nc
  counts(i)=mod(i-1,4)
  write(u,'(A,",",I0,",0,",Z8.8)')'NCLDS',i,counts(i)
  do q=1,counts(i)
   ca(i,q+1)=real(mod(i*13+q*7,101))/100.
   kt(i,q+1)=1+mod(q*3+i,8)
   kb(i,q+1)=min(n,kt(i,q+1)+mod(i+q,4))
  enddo
  do q=1,nv
   write(u,'(A,",",I0,",",I0,",",Z8.8)')'CAMT',i,q,transfer(ca(i,q),0_int32)
   write(u,'(A,",",I0,",",I0,",",Z8.8)')'KTOP',i,q,kt(i,q)
   write(u,'(A,",",I0,",",I0,",",Z8.8)')'KBTM',i,q,kb(i,q)
  enddo
 enddo
 call clo89(fac,ca,counts,kb,kt, 1,nc+1,1,2,1,nv, 1,nc,1,1,1,nv, 1,nc,1,1,1,n)
 do i=1,nc
  do j=1,nv
   do k=1,nv
    write(v,'(A,",",I0,",",I0,",",Z8.8)')'CLO89',i,k+(j-1)*nv,transfer(fac(i,k,j),0_int32)
   enddo
  enddo
 enddo
 close(u);close(v)
end program
