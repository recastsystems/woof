! NCAR/MYNN-SFC v6.1.0 src/module_sf_mynnsfc_driver.F90:103-833 @a038a9e (adapted from the v6.0.0 bridge).
! flag_iter is no longer an argument; iter, dry/wet/icy, uoce/voce and taux/tauy are absent (the RRFS MPAS host passes none).
! Complete original driver interface; captured full profiles and bounds, one call for all columns.
subroutine native_mynnsfc_observed(f,options,vegtype,profiles,domains,threshold,mask,n,levels,error_text,error_flag) bind(c)
 use iso_c_binding
 use module_sf_mynnsfc_driver
 implicit none
 integer(c_int),value::n,levels
 integer(c_int),target::options(*),vegtype(n)
 real(c_float),target::f(n,68)
 real(c_float),value::threshold
 integer(c_int64_t),value::mask
 integer::err,k
 integer(c_int),intent(in)::domains(18)
 integer(c_int8_t),intent(out)::error_text(512)
 integer(c_int),intent(out)::error_flag
 character(len=512)::errmsg
 character(len=12),target::lsm_value
 character(len=:),pointer::p_flagc_lsm
 real(c_float),target::profiles(n,levels,1,9)
 integer,pointer::p_ivegsrc
 logical,target::v_redrag
 logical,pointer::p_redrag
 integer,pointer::p_sfc_z0_type
 integer,pointer::p_spp_pbl
 integer,pointer::p_sf_mynn_sfcflux_water
 integer,pointer::p_sf_mynn_sfcflux_land
 integer,pointer::p_shalwater_z0
 logical,target::v_restart
 logical,pointer::p_restart
 logical,target::v_cycling
 logical,pointer::p_cycling
 integer,pointer::p_initflag
 real(c_float),pointer::p_sigmaf(:,:)
 real(c_float),pointer::p_shdmax(:,:)
 real(c_float),pointer::p_z0pert(:,:)
 real(c_float),pointer::p_ztpert(:,:)
 real(c_float),pointer::p_ustm(:,:)
 real(c_float),pointer::p_lakemask(:,:)
 real(c_float),pointer::p_ck(:,:)
 real(c_float),pointer::p_cka(:,:)
 real(c_float),pointer::p_cd(:,:)
 real(c_float),pointer::p_cda(:,:)
 real(c_float),pointer::p_stress(:,:)
 real(c_float),pointer::p_hflx(:,:)
 real(c_float),pointer::p_qflx(:,:)
 real(c_float),pointer::p_cm(:,:)
 real(c_float),pointer::p_fm(:,:)
 real(c_float),pointer::p_fh(:,:)
 real(c_float),pointer::p_fm10(:,:)
 real(c_float),pointer::p_fh2(:,:)
 real(c_float),pointer::p_tsurf(:,:)
 real(c_float),pointer::p_water_depth(:,:)
 real(c_float),pointer::p_pattern_spp_pbl(:,:,:)
 integer,pointer::p_vegtype(:,:)
 real(c_float),target::pattern(n,levels,1)
 integer,target::vegetation(n,1)
vegetation(:,1)=vegtype
pattern=profiles(:,:,:,9)
nullify(p_ivegsrc)
if(btest(mask,8)) p_ivegsrc=>options(1)
v_redrag=options(2)/=0
nullify(p_redrag)
if(btest(mask,10)) p_redrag=>v_redrag
nullify(p_sfc_z0_type)
if(btest(mask,9)) p_sfc_z0_type=>options(3)
nullify(p_spp_pbl)
if(btest(mask,7)) p_spp_pbl=>options(4)
nullify(p_sf_mynn_sfcflux_water)
if(btest(mask,3)) p_sf_mynn_sfcflux_water=>options(5)
nullify(p_sf_mynn_sfcflux_land)
if(btest(mask,4)) p_sf_mynn_sfcflux_land=>options(6)
nullify(p_shalwater_z0)
if(btest(mask,5)) p_shalwater_z0=>options(7)
v_restart=options(9)/=0
nullify(p_restart)
if(btest(mask,0)) p_restart=>v_restart
v_cycling=options(10)/=0
nullify(p_cycling)
if(btest(mask,1)) p_cycling=>v_cycling
nullify(p_initflag)
if(btest(mask,2)) p_initflag=>options(11)
nullify(p_flagc_lsm)
select case(options(13))
case(1);lsm_value='sf_noaa'
case(2);lsm_value='sf_noahmp'
case(3);lsm_value='sf_ruc'
case default;lsm_value='other'
end select
if(btest(mask,6)) p_flagc_lsm=>lsm_value
nullify(p_vegtype)
if(btest(mask,12))p_vegtype=>vegetation
nullify(p_sigmaf)
if(btest(mask,13))p_sigmaf=>f(:,12:12)
nullify(p_shdmax)
if(btest(mask,14))p_shdmax=>f(:,13:13)
nullify(p_z0pert)
if(btest(mask,15))p_z0pert=>f(:,14:14)
nullify(p_ztpert)
if(btest(mask,16))p_ztpert=>f(:,15:15)
nullify(p_ustm)
if(btest(mask,26))p_ustm=>f(:,24:24)
nullify(p_lakemask)
if(btest(mask,19))p_lakemask=>f(:,45:45)
nullify(p_ck)
if(btest(mask,22))p_ck=>f(:,53:53)
nullify(p_cka)
if(btest(mask,23))p_cka=>f(:,54:54)
nullify(p_cd)
if(btest(mask,24))p_cd=>f(:,55:55)
nullify(p_cda)
if(btest(mask,25))p_cda=>f(:,56:56)
nullify(p_stress)
if(btest(mask,33))p_stress=>f(:,57:57)
nullify(p_hflx)
if(btest(mask,31))p_hflx=>f(:,58:58)
nullify(p_qflx)
if(btest(mask,32))p_qflx=>f(:,59:59)
nullify(p_cm)
if(btest(mask,21))p_cm=>f(:,60:60)
nullify(p_fm)
if(btest(mask,27))p_fm=>f(:,61:61)
nullify(p_fh)
if(btest(mask,28))p_fh=>f(:,62:62)
nullify(p_fm10)
if(btest(mask,29))p_fm10=>f(:,63:63)
nullify(p_fh2)
if(btest(mask,30))p_fh2=>f(:,64:64)
nullify(p_tsurf)
if(btest(mask,18))p_tsurf=>f(:,65:65)
nullify(p_water_depth)
if(btest(mask,20))p_water_depth=>f(:,66:66)
nullify(p_pattern_spp_pbl)
if(btest(mask,17))p_pattern_spp_pbl=>pattern
call mynnsfc_driver( &
u3d=profiles(:,:,:,1), &
v3d=profiles(:,:,:,2), &
t3d=profiles(:,:,:,3), &
qv3d=profiles(:,:,:,4), &
p3d=profiles(:,:,:,5), &
dz8w=profiles(:,:,:,6), &
th3d=profiles(:,:,:,7), &
rho3d=profiles(:,:,:,8), &
sigmaf=p_sigmaf, &
vegtype=p_vegtype, &
shdmax=p_shdmax, &
ivegsrc=p_ivegsrc, &
z0pert=p_z0pert, &
ztpert=p_ztpert, &
redrag=p_redrag, &
sfc_z0_type=p_sfc_z0_type, &
psfcpa=f(:,16:16), &
chs=f(:,17:17), &
chs2=f(:,18:18), &
cqs=f(:,19:19), &
cqs2=f(:,20:20), &
cpm=f(:,21:21), &
znt=f(:,22:22), &
ust=f(:,23:23), &
ustm=p_ustm, &
pblh=f(:,25:25), &
mavail=f(:,26:26), &
zol=f(:,27:27), &
mol=f(:,28:28), &
rmol=f(:,29:29), &
psim=f(:,30:30), &
psih=f(:,31:31), &
xland=f(:,32:32), &
qgh=f(:,33:33), &
hfx=f(:,34:34), &
qfx=f(:,35:35), &
lh=f(:,36:36), &
tsk=f(:,37:37), &
flhc=f(:,38:38), &
flqc=f(:,39:39), &
qsfc=f(:,40:40), &
u10=f(:,41:41), &
v10=f(:,42:42), &
th2=f(:,43:43), &
t2=f(:,44:44), &
lakemask=p_lakemask, &
q2=f(:,46:46), &
snowh=f(:,47:47), &
gz1oz0=f(:,48:48), &
wspd=f(:,49:49), &
br=f(:,50:50), &
dx=f(:,51:51), &
ch=f(:,52:52), &
ck=p_ck, &
cka=p_cka, &
cd=p_cd, &
cda=p_cda, &
stress=p_stress, &
hflx=p_hflx, &
qflx=p_qflx, &
cm=p_cm, &
fm=p_fm, &
fh=p_fh, &
fm10=p_fm10, &
fh2=p_fh2, &
tsurf=p_tsurf, &
water_depth=p_water_depth, &
xice=f(:,67:67), &
xice_threshold=threshold, &
spp_pbl=p_spp_pbl, &
pattern_spp_pbl=p_pattern_spp_pbl, &
sf_mynn_sfcflux_water=p_sf_mynn_sfcflux_water, &
sf_mynn_sfcflux_land=p_sf_mynn_sfcflux_land, &
shalwater_z0=p_shalwater_z0, &
isfflx=options(8), &
restart=p_restart, &
cycling=p_cycling, &
initflag=p_initflag, &
flagc_lsm=p_flagc_lsm, &
itimestep=options(14), &
ids=domains(1), &
ide=domains(2), &
jds=domains(3), &
jde=domains(4), &
kds=domains(5), &
kde=domains(6), &
ims=domains(7), &
ime=domains(8), &
jms=domains(9), &
jme=domains(10), &
kms=domains(11), &
kme=domains(12), &
its=domains(13), &
ite=domains(14), &
jts=domains(15), &
jte=domains(16), &
kts=domains(17), &
kte=domains(18), &
errflg=err, &
errmsg=errmsg)
error_text=transfer(errmsg,error_text)
error_flag=err
end subroutine

subroutine native_mynnsfc_init() bind(c)
 use module_sf_mynnsfc_driver,only:mynnsfc_init
 implicit none
 character(len=512)::errmsg
 integer::err
 call mynnsfc_init(.true.,errmsg,err)
end subroutine
