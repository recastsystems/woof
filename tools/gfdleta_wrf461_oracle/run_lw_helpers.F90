program run_lw_helpers
 use iso_fortran_env,only:int32
 use module_ra_gfdleta,only:table,e3v88,e290,e2spec,em3v,t1,t2,t4,e1e290,em1v,em1vw
 implicit none
 integer,parameter::nc=64,n=12,nv=n+1,nw=2*n+1
 real::tv(nc,nw),av(nc,nw),out(nc,nw),phi(nc,nv),fx(nc,nv),dt(nc,nv)
 real::f1(nc,nv),d1(nc,nv),g1(nc,nv),g2(nc,n),g3(nc,nv),g4(nc,nv),g5(nc,n)
 real::em(nc,nv),eb(nc,nv),fsp(nc,2),dsp(nc,2)
 integer::i,k,u,v,q,kl
 call table
 open(newunit=u,file='lw-inputs.csv',status='replace')
 open(newunit=v,file='lw-expected.csv',status='replace')
 write(u,'(A)')'field,column,index,bits';write(v,'(A)')'field,column,index,bits'
 do q=1,5040
  call input('EM1',1,q,em1v(q));call input('EM1VW',1,q,em1vw(q));call input('EM3',1,q,em3v(q));call input('T1',1,q,t1(q));call input('T2',1,q,t2(q));call input('T4',1,q,t4(q))
 enddo
 do i=1,nc
  do k=1,nw
   tv(i,k)=170.+real(mod(i*11+k*7,170))+.125*mod(k,5)
   av(i,k)=10.**(-12.+real(mod(i*17+k*3,120))/10.)
   call input('TV',i,k,tv(i,k));call input('AV',i,k,av(i,k))
  enddo
  do k=1,nv
   phi(i,k)=av(i,k);fx(i,k)=aint(tv(i,k)*.1)-9.;dt(i,k)=tv(i,k)-10.*aint(tv(i,k)*.1)
   f1(i,k)=fx(i,k)+1.;d1(i,k)=mod(dt(i,k)+.375,10.)
   call input('FX1',i,k,f1(i,k));call input('DT1',i,k,d1(i,k))
   call input('PHI',i,k,phi(i,k));call input('FX',i,k,fx(i,k));call input('DT',i,k,dt(i,k))
  enddo
  fsp(i,:)=fx(i,1:2);dsp(i,:)=dt(i,1:2)
 enddo
 call e3v88(out,tv,av,10.,.1,28.,16., 1,nc+1,1,2,1,nv, 1,nc,1,1,1,nv, 1,nc,1,1,1,n)
 do i=1,nc
  do k=1,nw
   call expected('E3V88',i,k,out(i,k))
  enddo
 enddo
 em=0.
 call e2spec(em,phi,fsp,dsp,16.,10.,28.,.1, 1,nc+1,1,2,1,nv, 1,nc,1,1,1,nv, 1,nc,1,1,1,n)
 do i=1,nc
  do k=1,2
   call expected('E2SPEC',i,k,em(i,k))
  enddo
 enddo
 kl=3;em=0.;eb=0.
 call e290(eb,em,phi,kl,fx,dt,16.,.1,28.,.5,10., 1,nc+1,1,2,1,nv, 1,nc,1,1,1,nv, 1,nc,1,1,1,n)
 do i=1,nc
  do k=kl,nv
   call expected('E290',i,k,em(i,k))
   if(k<=n)call expected('E290B',i,k,eb(i,k))
  enddo
 enddo
 call e1e290(g1,g2,g3,g4,g5,em,f1,d1,fx,dt,phi,tv(:,1:nv),tv(:,1:nv),16.,10.,.1,28.,.5, &
 1,nc+1,1,2,1,nv, 1,nc,1,1,1,nv, 1,nc,1,1,1,n)
 do i=1,nc
  do k=1,nv
   call expected('G1',i,k,g1(i,k));call expected('G3',i,k,g3(i,k));call expected('G4',i,k,g4(i,k))
   call expected('E1EMISS',i,k,em(i,k))
   if(k<=n)then
    call expected('G2',i,k,g2(i,k));call expected('G5',i,k,g5(i,k))
   endif
  enddo
 enddo
 close(u);close(v)
contains
 subroutine input(name,c,j,x)
  character(*)::name
  integer::c,j
  real::x
  write(u,'(A,",",I0,",",I0,",",Z8.8)')name,c,j,transfer(x,0_int32)
 end subroutine
 subroutine expected(name,c,j,x)
  character(*)::name
  integer::c,j
  real::x
  write(v,'(A,",",I0,",",I0,",",Z8.8)')name,c,j,transfer(x,0_int32)
 end subroutine
end program
