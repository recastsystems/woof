! Capture wrapper for the WRF v4.8.0 GFL driver (capture copy only; the referee binary stays pristine).
! Inserted by make_capture_source.py into a copy of phys/GFL/WRF/module_cu_gfl_wrf.F90 after the
! original routine has been renamed gfl_referee_body. It writes every argument word before the call
! (PRE) and every inout word after it (POST), then returns. Arithmetic is the original routine's.
 subroutine cu_grell_freitas_li(                                &
               itimestep,dt,dxcell                     &
              ,u,v,w,t,q,rho,p,pi,p8w,dz8w                      &
              ,ht,xland,hfx,qfx,gsw,rqvften,rthften             &
              ,rthblten,rqvblten,rthraten,kpbl,xlv,cp,g,r_v     &
              ,ichoice_deep,ishallow_g3         &
              ,htop,hbot,k22_shallow,kbcon_shallow,ktop_shallow &
              ,xmb_shallow,raincv,pratec,gdc,gdc2     &
              ,rthcuten,rqvcuten,rqccuten,rqicuten              &
              ,rucuten,rvcuten                                  &
              ,ims, ime, jms, jme, kms,kme                      &
              ,ids, ide, jds, jde, kds,kde                      &
              ,its, ite, jts, jte, kts,kte)
 integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte
 integer,intent(in):: ichoice_deep,itimestep
 integer,intent(in):: ishallow_g3
 integer,dimension(ims:ime,jms:jme ),intent(in):: kpbl
 real(kind=kind_phys),intent(in):: dt
 real(kind=kind_phys),intent(in):: xlv,r_v,cp,g
 real(kind=kind_phys),intent(in):: dxCell
 real(kind=kind_phys),dimension(ims:ime,jms:jme),intent(in):: hfx,qfx,gsw,ht,xland
 real(kind=kind_phys),dimension(ims:ime,kms:kme,jms:jme),intent(in):: u,v,w,p,pi,q,rho,t
 real(kind=kind_phys),dimension(ims:ime,kms:kme,jms:jme),intent(in):: dz8w,p8w
 real(kind=kind_phys),dimension(ims:ime,kms:kme,jms:jme),intent(in):: rqvblten,rthblten,rthraten
 real(kind=kind_phys),dimension(ims:ime,kms:kme,jms:jme),intent(in),optional:: rthften,rqvften
 integer,dimension(ims:ime,jms:jme),intent(inout):: k22_shallow,kbcon_shallow,ktop_shallow
 real(kind=kind_phys),dimension(ims:ime,jms:jme),intent(inout):: hbot,htop,raincv,pratec,xmb_shallow
 real(kind=kind_phys),dimension(ims:ime,kms:kme,jms:jme),intent(inout):: rthcuten,rqvcuten,rqccuten,rqicuten
 real(kind=kind_phys),dimension(ims:ime,kms:kme,jms:jme),intent(inout):: rucuten,rvcuten
 real(kind=kind_phys),dimension(ims:ime,kms:kme,jms:jme),intent(inout),optional:: gdc,gdc2
 character(len=512):: dir, steps, path
 character(len=32):: tag
 integer:: unit, ios, n
 logical:: want
 integer, save:: calls = 0

 want = .false.
 call get_environment_variable('GFL_CAPTURE_DIR', dir, status=ios)
 if (ios == 0) then
    call get_environment_variable('GFL_CAPTURE_STEPS', steps, status=ios)
    if (ios == 0) then
       write(tag,'(a,i0,a)') ',', itimestep, ','
       want = index(','//trim(steps)//',', trim(tag)) > 0
    endif
 endif
 calls = calls + 1
 if (want) then
    write(path,'(a,a,i6.6,a,i6.6,a)') trim(dir), '/pre_', itimestep, '_', calls, '.bin'
    open(newunit=unit, file=trim(path), access='stream', form='unformatted', status='replace', convert='little_endian')
    write(unit) ims,ime,jms,jme,kms,kme,ids,ide,jds,jde,kds,kde,its,ite,jts,jte,kts,kte, itimestep,ichoice_deep,ishallow_g3
    write(unit) merge(1,0,present(rthften)), merge(1,0,present(rqvften)), merge(1,0,present(gdc)), merge(1,0,present(gdc2))
    write(unit) dt, dxcell, xlv, cp, g, r_v
    write(unit) u,v,w,t,q,rho,p,pi,p8w,dz8w
    if (present(rqvften)) write(unit) rqvften
    if (present(rthften)) write(unit) rthften
    write(unit) rthblten,rqvblten,rthraten
    write(unit) ht,xland,hfx,qfx,gsw
    write(unit) kpbl
    call put_inout(unit)
    close(unit)
 endif
 call gfl_referee_body(itimestep,dt,dxcell,u,v,w,t,q,rho,p,pi,p8w,dz8w        &
              ,ht,xland,hfx,qfx,gsw,rqvften,rthften                         &
              ,rthblten,rqvblten,rthraten,kpbl,xlv,cp,g,r_v                 &
              ,ichoice_deep,ishallow_g3                                    &
              ,htop,hbot,k22_shallow,kbcon_shallow,ktop_shallow             &
              ,xmb_shallow,raincv,pratec,gdc,gdc2                           &
              ,rthcuten,rqvcuten,rqccuten,rqicuten,rucuten,rvcuten          &
              ,ims, ime, jms, jme, kms,kme, ids, ide, jds, jde, kds,kde     &
              ,its, ite, jts, jte, kts,kte)
 if (want) then
    write(path,'(a,a,i6.6,a,i6.6,a)') trim(dir), '/post_', itimestep, '_', calls, '.bin'
    open(newunit=unit, file=trim(path), access='stream', form='unformatted', status='replace', convert='little_endian')
    call put_inout(unit)
    close(unit)
 endif
 contains
 subroutine put_inout(u8)
 integer, intent(in):: u8
 write(u8) htop,hbot,xmb_shallow,raincv,pratec
 write(u8) k22_shallow,kbcon_shallow,ktop_shallow
 if (present(gdc)) write(u8) gdc
 if (present(gdc2)) write(u8) gdc2
 write(u8) rthcuten,rqvcuten,rqccuten,rqicuten,rucuten,rvcuten
 end subroutine put_inout
 end subroutine cu_grell_freitas_li
