program run_flg
  use iso_fortran_env, only: int32
  use para_file
  use control_para
  use band_new, only: hk_1_new
  use module_ra_FLG
  implicit none
  integer, parameter :: nv=24, nv1=nv+1
  integer :: c,s,a,ng,nice_i,k,u,uc,ui,ud,jday
  character(256) :: output, mutation
  real :: pp(nv1),pt(nv1),ph(nv1),po(nv1),as(mbs),ee(mbir),u0,pts
  real :: iwc(nv),de(nv),lwc(nv),re(nv),rwc(nv),gwc(nv),cf(nv),ci(nv)
  real :: aw(mxat,mxac),at(mxat,mxac),ap(nvx,mxac)
  real :: fds(nv1),fus(nv1),dts(nv),fdir(nv1),fuir(nv1),dtir(nv)
  real :: fd(nv1),fu(nv1),dtt(nv),direct(nv1),diffuse(nv1)
  real :: pev(1,nv1,1),pod(1,nv1,1),temp(1,nv1,1),t8(1,nv1,1),pi(1,nv1,1)
  real :: qv(1,nv1,1),qi(1,nv1,1),qc(1,nv1,1),qr(1,nv1,1),qg(1,nv1,1),qs(1,nv1,1)
  real :: ozone3(1,nv1,1),cldf(1,nv1,1),rho(1,nv1,1),dz3(1,nv1,1)
  real :: alb(1,1),tsk(1,1),em(1,1),lon(1,1),lat(1,1),swtop(1,1),lwtop(1,1)
  real :: gsw(1,1),glw(1,1),swd(1,1),dhi(1,1),dif(1,1),dni(1,1)
  real :: tend(1,nv1,1),tsw(1,nv1,1),tlw(1,nv1,1),gmt1,x,factor
  call get_command_argument(1,output)
  if (len_trim(output)==0) error stop 'usage: run_flg OUTPUT_DIR [flip-coefficient]'
  call get_command_argument(2,mutation)
  if (len_trim(mutation)>0) then
    if (trim(mutation)/='flip-coefficient') error stop 'unknown mutation'
    hk_1_new(2)=transfer(ieor(transfer(hk_1_new(2),0_int32),1_int32),0.0)
  endif
  open(newunit=u,file=trim(output)//'/columns.csv',status='replace')
  open(newunit=uc,file=trim(output)//'/cases.csv',status='replace')
  open(newunit=ui,file=trim(output)//'/inputs.csv',status='replace')
  open(newunit=ud,file=trim(output)//'/driver.csv',status='replace')
  write(u,'(A)') 'case,field,index,bits'
  write(ui,'(A)') 'case,field,index,bits'
  write(ud,'(A)') 'case,field,index,bits'
  write(uc,'(A)') 'case,ice,scenario,aerosol,new_gases'
  c=0
  do nice_i=0,7
    do s=0,7
      do a=0,1
        do ng=0,1
          c=c+1
          write(uc,'(I0,4(A,I0))') c,',',nice_i,',',s,',',a,',',ng
          call reset_controls()
          nice=nice_i; naero=a; ngas=ng
          nsubcld=(nv-1)/ngroup; nclouds=nsubcld*ngroup
          as=0.12; ee=0.98; pts=294.; u0=0.75
          if(s==0) pts=280.
          if(s==6) u0=-0.25
          if(s==7) then
            u0=0.02; as=0.85
          endif
          iwc=0.; lwc=0.; rwc=0.; gwc=0.; cf=0.; ci=1.; re=10.; de=85.
          do k=1,nv1
            x=real(k-1)/real(nv)
            pp(k)=10.+990.*x*x
            pt(k)=210.+84.*x
            ph(k)=0.00001+0.014*x*x*x
            if(s==0) ph(k)=0.00000001+0.00001*x
            po(k)=0.000002*(1.-x)+0.00000005
          enddo
          do k=8,nv-2
            if(s==2.or.s==4.or.s==5) lwc(k)=0.10
            if(s==3.or.s==4.or.s==5) iwc(k)=0.05
            if(s==5) then
              lwc(k)=0.5; iwc(k)=0.3; rwc(k)=0.1; gwc(k)=0.05
            endif
            if(lwc(k)+iwc(k)>0.) cf(k)=0.6
          enddo
          aw=0.; at=0.; ap=0.
          if(a==1) then
            itps(2)=1
            aw(1,2)=0.53; at(1,2)=0.2
            call aer_scale_hgt(nv,pp,3.,ap(1:nv,2))
          endif
          call emit(ui,'pp',pp); call emit(ui,'pt',pt); call emit(ui,'ph',ph); call emit(ui,'po',po)
          call emit(ui,'as',as); call emit(ui,'ee',ee); call emit(ui,'u0',[u0]); call emit(ui,'pts',[pts])
          call emit(ui,'iwc',iwc); call emit(ui,'de',de); call emit(ui,'lwc',lwc); call emit(ui,'re',re)
          call emit(ui,'rwc',rwc); call emit(ui,'gwc',gwc); call emit(ui,'cf',cf); call emit(ui,'ci',ci)
          call rad_all(nv,nv1,as,u0,1361.,pts,ee,pp,pt,ph,po,iwc,de,lwc,re,rwc,gwc,cf,ci, &
                       aw,at,ap,fds,fus,dts,fdir,fuir,dtir,fd,fu,dtt,direct,diffuse)
          call emit(u,'SW_DOWN',fds); call emit(u,'SW_UP',fus)
          call emit(u,'LW_DOWN',fdir); call emit(u,'LW_UP',fuir)
          call emit(u,'NET_DOWN',fd); call emit(u,'NET_UP',fu)
          call emit(u,'SW_HEATING',dts); call emit(u,'LW_HEATING',dtir); call emit(u,'HEATING',dtt)
          call emit(u,'DIRECT',direct); call emit(u,'DIFFUSE',diffuse)
          call emit(u,'GLW',[fdir(nv1)]); call emit(u,'GSW',[fds(nv1)-fus(nv1)])
          call emit(u,'SWDOWN',[fds(nv1)]); call emit(u,'OLR',[fuir(1)])
          ! Independent RAD_FLG driver deck, using the same vertical atmosphere.
          ! The driver derives cloud sizes and ozone itself; fluxes above are the core deck.
          if(a==0.and.ng==0) then
            call reset_controls(); nice=nice_i
            do k=1,nv1
              pev(1,k,1)=pp(nv1-k+1)*100.
              pod(1,k,1)=pev(1,k,1)
              t8(1,k,1)=pt(nv1-k+1); temp(1,k,1)=t8(1,k,1)
              pi(1,k,1)=(pod(1,k,1)/100000.)**(287./1004.)
              qv(1,k,1)=ph(nv1-k+1)
            enddo
            qi=0.; qc=0.; qr=0.; qg=0.; qs=0.; cldf=0.; rho=1.; dz3=500.; ozone3=0.
            do k=1,nv
              qi(1,k,1)=iwc(nv-k+1)/1000.; qc(1,k,1)=lwc(nv-k+1)/1000.
              qr(1,k,1)=rwc(nv-k+1)/1000.; qg(1,k,1)=gwc(nv-k+1)/1000.
              cldf(1,k,1)=cf(nv-k+1)
            enddo
            alb=as(1); tsk=pts; em=ee(1); lon=0.; lat=45.; jday=180
            if(s==0) then
              lat=-70.; jday=10
            endif
            gmt1=12.; if(s==6) gmt1=0.
            if(s==7) gmt1=6.1
            tend=0.; tsw=0.; tlw=0.
            call RAD_FLG(pev,pod,t8,temp,pi,ozone3,9.81,1004.,alb,tsk,qv,qi,qc,qr,qg,qs, &
              .true.,.true.,.true.,.true.,.true.,.true.,.false.,cldf,em,rho,dz3,1361.,0.0, &
              0.0,lon,lat,jday,gmt1,0.0,0.0174532925,60., &
              1,2,1,2,1,nv1,1,1,1,1,1,nv1,1,1,1,1,1,nv, &
              swtop,lwtop,gsw,glw,swd,tend,tsw,tlw,dhi,dif,dni)
            call emit(ud,'RTHRATEN',tend(1,1:nv,1))
            call emit(ud,'RTHRATENSW',tsw(1,1:nv,1)); call emit(ud,'RTHRATENLW',tlw(1,1:nv,1))
            call emit(ud,'GLW',[glw(1,1)]); call emit(ud,'GSW',[gsw(1,1)])
            call emit(ud,'SWDOWN',[swd(1,1)]); call emit(ud,'RLWTOA',[lwtop(1,1)])
          endif
        enddo
      enddo
    enddo
  enddo
  close(u); close(uc); close(ui); close(ud)
contains
  subroutine reset_controls()
    nice=2; naero=0; ngas=0; nfract=0; ninho=0; npde=2; nozone=1
    d4s=.true.; d2s=.false.; d4ir=.false.; d2ir=.true.; pderandom=.false.
    itps=0; nfraca=1; ivd=1; ifg=0; iaform=3; n_atau=1; ngroup=3
    umco2=345.; umch4=1.6; umn2o=.28; umco=.16; umo2=2.0948e5
    umno=.0005; umso2=.001; umno2=.001; umch3cl=.5e-3; umcfc11=.22e-3; umcfc12=.375e-3
  end subroutine
  subroutine emit(unit1,field,values)
    integer,intent(in)::unit1
    character(*),intent(in)::field
    real,intent(in)::values(:)
    integer::idx
    do idx=1,size(values)
      write(unit1,'(I0,A,A,A,I0,A,Z8.8)') c,',',trim(field),',',idx,',',transfer(values(idx),0_int32)
    enddo
  end subroutine
end program
