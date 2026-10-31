"""Pinned RRFS-SD subsystem; sequential tile launcher preserves native loop order.

SMOKE 9652bbf in RRFS MPAS-Model-rrfs 1d6e370e. All arithmetic runs on
the device. Callers supply the native wrapper arguments in Fortran order;
registry/stream decoding belongs to the Rust driver. Persistent plume and
SAVE state belong to one launcher instance, matching one native tile/thread.
This module is not an accepted whole-forecast identity delivery yet.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics
import json
from pathlib import Path
import numpy as np
FORK_PIN = "1d6e370e7b43a332074809f170532a216c2212fb"
SMOKE_PIN = "9652bbf152f8b946bf04f72a2b48ff67e15d7973"
# Literal payloads, including spaces, at the original wrapper call sites.
# MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:262-600 @1d6e370e.
WRAPPER_LOG_MESSAGES = {262: ' Beginning Aerosol Driver', 290: ' Calling smoke prep', 312: ' Initializing dry deposition parameterss ', 314: ' Initializing radiation feedback parameterss ', 321: ' Calling module wildfire smoke emissions ', 452: ' Calling ebu_driver', 477: ' Calling add_emis_burn', 513: ' Calling dust driver', 538: ' Calling anthro emis driver', 555: ' Calling dry_dep_driver_emerson', 577: 'Not performing dry deposition', 583: ' Calling wetdep_ls', 595: ' Calculating AOD ', 600: ' Calculating VIS '}
ROUTINES = json.loads('{"gocart_seasalt_driver":{"module":"ssalt_mod","args":["dt","alt","t_phy","u_phy","v_phy","num_chem","chem","rho_phy","dz8w","u10","v10","ustar","p8w","tsk","xland","xlat","xlong","area","g","e_ss_out","num_e_ss_out","index_e_ss_out_ssalt_fine","index_e_ss_out_ssalt_coarse","pi","num_emis_seas","seas_opt","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in   ) :: dt,g,pi","str_len":"100"},"alt":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: alt, t_phy, dz8w,p8w, u_phy,v_phy,rho_phy","str_len":"100"},"t_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: alt, t_phy, dz8w,p8w, u_phy,v_phy,rho_phy","str_len":"100"},"u_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: alt, t_phy, dz8w,p8w, u_phy,v_phy,rho_phy","str_len":"100"},"v_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: alt, t_phy, dz8w,p8w, u_phy,v_phy,rho_phy","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme, 1:num_chem ), intent(inout ) ::                                   chem","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: alt, t_phy, dz8w,p8w, u_phy,v_phy,rho_phy","str_len":"100"},"dz8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: alt, t_phy, dz8w,p8w, u_phy,v_phy,rho_phy","str_len":"100"},"u10":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )                   , intent(in   ) :: u10, v10, ustar,tsk, xland, xlat, xlong,area","str_len":"100"},"v10":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )                   , intent(in   ) :: u10, v10, ustar,tsk, xland, xlat, xlong,area","str_len":"100"},"ustar":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )                   , intent(in   ) :: u10, v10, ustar,tsk, xland, xlat, xlong,area","str_len":"100"},"p8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: alt, t_phy, dz8w,p8w, u_phy,v_phy,rho_phy","str_len":"100"},"tsk":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )                   , intent(in   ) :: u10, v10, ustar,tsk, xland, xlat, xlong,area","str_len":"100"},"xland":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )                   , intent(in   ) :: u10, v10, ustar,tsk, xland, xlat, xlong,area","str_len":"100"},"xlat":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )                   , intent(in   ) :: u10, v10, ustar,tsk, xland, xlat, xlong,area","str_len":"100"},"xlong":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )                   , intent(in   ) :: u10, v10, ustar,tsk, xland, xlat, xlong,area","str_len":"100"},"area":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )                   , intent(in   ) :: u10, v10, ustar,tsk, xland, xlat, xlong,area","str_len":"100"},"g":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in   ) :: dt,g,pi","str_len":"100"},"e_ss_out":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_e_ss_out"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme,1:num_e_ss_out), intent(inout   ) :: e_ss_out","str_len":"100"},"num_e_ss_out":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) ::  num_e_ss_out, index_e_ss_out_ssalt_fine, index_e_ss_out_ssalt_coarse","str_len":"100"},"index_e_ss_out_ssalt_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) ::  num_e_ss_out, index_e_ss_out_ssalt_fine, index_e_ss_out_ssalt_coarse","str_len":"100"},"index_e_ss_out_ssalt_coarse":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) ::  num_e_ss_out, index_e_ss_out_ssalt_fine, index_e_ss_out_ssalt_coarse","str_len":"100"},"pi":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in   ) :: dt,g,pi","str_len":"100"},"num_emis_seas":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"seas_opt":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: num_emis_seas,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte,seas_opt","str_len":"100"}},"source":"ssalt_mod.F90","first_line":25,"last_line":158},"source_ss":{"module":"ssalt_mod","args":["imx","jmx","lmx","nmx","dt1","tc","pi","ilwi","dxy","w10m","airmas","bems","ipr"],"symbols":{"imx":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in)    :: nmx,imx,jmx,lmx,ipr","str_len":"100"},"jmx":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in)    :: nmx,imx,jmx,lmx,ipr","str_len":"100"},"lmx":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in)    :: nmx,imx,jmx,lmx,ipr","str_len":"100"},"nmx":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in)    :: nmx,imx,jmx,lmx,ipr","str_len":"100"},"dt1":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: dt1,fudge_fac","str_len":"100"},"tc":{"type":"float","dim":["imx","jmx","lmx","nmx"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(inout) :: tc(imx,jmx,lmx,nmx)","str_len":"100"},"pi":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)    :: dxy(jmx), w10m(imx,jmx), pi","str_len":"100"},"ilwi":{"type":"int","dim":["imx","jmx"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in)    :: ilwi(imx,jmx)","str_len":"100"},"dxy":{"type":"float","dim":["jmx"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)    :: dxy(jmx), w10m(imx,jmx), pi","str_len":"100"},"w10m":{"type":"float","dim":["imx","jmx"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)    :: dxy(jmx), w10m(imx,jmx), pi","str_len":"100"},"airmas":{"type":"float","dim":["imx","jmx","lmx"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)    :: airmas(imx,jmx,lmx)","str_len":"100"},"bems":{"type":"float","dim":["imx","jmx","nmx"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(out)   :: bems(imx,jmx,nmx)","str_len":"100"},"ipr":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in)    :: nmx,imx,jmx,lmx,ipr","str_len":"100"}},"source":"ssalt_mod.F90","first_line":160,"last_line":320},"ebu_driver":{"module":"module_plumerise","args":["flam_frac","kfire","theta_phy","q_vap","rho_phy","vvel","u_phy","v_phy","pi_phy","wind_phy","z_at_w","z","g","con_cp","con_rd","frp_inst","k_min","k_max","wind_eff_opt","do_plumerise","kpbl_thetav","kpbl","curr_secs","xlat","xlong","uspdavg2d","hpbl2d","alpha","frp_min","frp_wthreshold","zpbl_lim","uspd_lim","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte","errmsg","errflg","ebu_in","ebu","ebu_in_coarse","ebu_coarse","ebu_in_ch4","ebu_ch4"],"symbols":{"flam_frac":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime, jms:jme ), intent(out ) :: flam_frac","str_len":"100"},"kfire":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: kfire","str_len":"100"},"theta_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"q_vap":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"vvel":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"u_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"v_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"pi_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"wind_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"z_at_w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"z":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime , kms:kme , jms:jme )         , intent(in   ) ::   z,z_at_w,vvel,u_phy,v_phy,rho_phy,pi_phy,q_vap,theta_phy,wind_phy","str_len":"100"},"g":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), intent(in )  :: g, con_cp, con_rd","str_len":"100"},"con_cp":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), intent(in )  :: g, con_cp, con_rd","str_len":"100"},"con_rd":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), intent(in )  :: g, con_cp, con_rd","str_len":"100"},"frp_inst":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime, jms:jme), intent(in ) :: frp_inst","str_len":"100"},"k_min":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, dimension(ims:ime, jms:jme), intent (inout) :: k_min, k_max","str_len":"100"},"k_max":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, dimension(ims:ime, jms:jme), intent (inout) :: k_min, k_max","str_len":"100"},"wind_eff_opt":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: wind_eff_opt","str_len":"100"},"do_plumerise":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical, intent(in) :: do_plumerise","str_len":"100"},"kpbl_thetav":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,         dimension(ims:ime, jms:jme),     intent(in)    :: kpbl, kpbl_thetav","str_len":"100"},"kpbl":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,         dimension(ims:ime, jms:jme),     intent(in)    :: kpbl, kpbl_thetav","str_len":"100"},"curr_secs":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: curr_secs","str_len":"100"},"xlat":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in) ::  xlat,xlong","str_len":"100"},"xlong":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in) ::  xlat,xlong","str_len":"100"},"uspdavg2d":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real, dimension(ims:ime, jms:jme), intent (in) :: uspdavg2d, hpbl2d","str_len":"100"},"hpbl2d":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real, dimension(ims:ime, jms:jme), intent (in) :: uspdavg2d, hpbl2d","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: alpha","str_len":"100"},"frp_min":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: frp_min, frp_wthreshold, zpbl_lim, uspd_lim","str_len":"100"},"frp_wthreshold":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: frp_min, frp_wthreshold, zpbl_lim, uspd_lim","str_len":"100"},"zpbl_lim":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: frp_min, frp_wthreshold, zpbl_lim, uspd_lim","str_len":"100"},"uspd_lim":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: frp_min, frp_wthreshold, zpbl_lim, uspd_lim","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"errmsg":{"type":"FStr","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"character(*), intent(inout) :: errmsg","str_len":"100"},"errflg":{"type":"int","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(inout) :: errflg","str_len":"100"},"ebu_in":{"type":"float","dim":["ims:ime","1:kfire","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime, 1:kfire, jms:jme ), intent(in ), optional  :: ebu_in, ebu_in_coarse, ebu_in_ch4","str_len":"100"},"ebu":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ), optional ::  ebu, ebu_coarse, ebu_ch4","str_len":"100"},"ebu_in_coarse":{"type":"float","dim":["ims:ime","1:kfire","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime, 1:kfire, jms:jme ), intent(in ), optional  :: ebu_in, ebu_in_coarse, ebu_in_ch4","str_len":"100"},"ebu_coarse":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ), optional ::  ebu, ebu_coarse, ebu_ch4","str_len":"100"},"ebu_in_ch4":{"type":"float","dim":["ims:ime","1:kfire","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime, 1:kfire, jms:jme ), intent(in ), optional  :: ebu_in, ebu_in_coarse, ebu_in_ch4","str_len":"100"},"ebu_ch4":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ), optional ::  ebu, ebu_coarse, ebu_ch4","str_len":"100"}},"source":"module_plumerise.F90","first_line":17,"last_line":180},"mpas_aod_diag":{"module":"module_smoke_diagnostics","args":["chem","aod3d","rho_phy","dz8w","num_chem","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme,1:num_chem), intent(in) :: chem","str_len":"100"},"aod3d":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent(inout) :: aod3d","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent(in) :: rho_phy, dz8w","str_len":"100"},"dz8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent(in) :: rho_phy, dz8w","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_chem","str_len":"100"}},"source":"module_smoke_diagnostics.F90","first_line":20,"last_line":53},"mpas_visibility_diag":{"module":"module_smoke_diagnostics","args":["qcloud","qrain","qice","qsnow","qgrpl","blcldw","blcldi","rho_phy","wind10m","wind","rh2m","rh","qv","t2m","t","coszen","extcoef55","vis","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"qcloud":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: qcloud,qrain,qice,qsnow,qgrpl","str_len":"100"},"qrain":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: qcloud,qrain,qice,qsnow,qgrpl","str_len":"100"},"qice":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: qcloud,qrain,qice,qsnow,qgrpl","str_len":"100"},"qsnow":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: qcloud,qrain,qice,qsnow,qgrpl","str_len":"100"},"qgrpl":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: qcloud,qrain,qice,qsnow,qgrpl","str_len":"100"},"blcldw":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: blcldi,blcldw","str_len":"100"},"blcldi":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: blcldi,blcldw","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: rho_phy,wind,rh,qv,t,extcoef55","str_len":"100"},"wind10m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,jms:jme),intent(in)     :: wind10m,rh2m,t2m,coszen","str_len":"100"},"wind":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: rho_phy,wind,rh,qv,t,extcoef55","str_len":"100"},"rh2m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,jms:jme),intent(in)     :: wind10m,rh2m,t2m,coszen","str_len":"100"},"rh":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: rho_phy,wind,rh,qv,t,extcoef55","str_len":"100"},"qv":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: rho_phy,wind,rh,qv,t,extcoef55","str_len":"100"},"t2m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,jms:jme),intent(in)     :: wind10m,rh2m,t2m,coszen","str_len":"100"},"t":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: rho_phy,wind,rh,qv,t,extcoef55","str_len":"100"},"coszen":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,jms:jme),intent(in)     :: wind10m,rh2m,t2m,coszen","str_len":"100"},"extcoef55":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in)   :: rho_phy,wind,rh,qv,t,extcoef55","str_len":"100"},"vis":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,jms:jme), intent(out)   :: vis","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"}},"source":"module_smoke_diagnostics.F90","first_line":55,"last_line":184},"compute_emission_factors":{"module":"module_fire_emissions","args":["efs_map","vegfrac","bb_beta","eco_id","efs_smold","efs_flam","efs_rsmold","fmc_avg","wetness","nlcat","dt","gmt","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"efs_map":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(out) :: efs_map","str_len":"100"},"vegfrac":{"type":"float","dim":["ims:ime","nlcat","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in), dimension(ims:ime, nlcat,jms:jme) :: vegfrac","str_len":"100"},"bb_beta":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: bb_beta, nlcat","str_len":"100"},"eco_id":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), dimension(ims:ime,jms:jme) :: eco_id","str_len":"100"},"efs_smold":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in) :: efs_smold, efs_flam, efs_rsmold, wetness, fmc_avg","str_len":"100"},"efs_flam":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in) :: efs_smold, efs_flam, efs_rsmold, wetness, fmc_avg","str_len":"100"},"efs_rsmold":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in) :: efs_smold, efs_flam, efs_rsmold, wetness, fmc_avg","str_len":"100"},"fmc_avg":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in) :: efs_smold, efs_flam, efs_rsmold, wetness, fmc_avg","str_len":"100"},"wetness":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in) :: efs_smold, efs_flam, efs_rsmold, wetness, fmc_avg","str_len":"100"},"nlcat":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: bb_beta, nlcat","str_len":"100"},"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dt, gmt","str_len":"100"},"gmt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dt, gmt","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"}},"source":"module_fire_emissions.F90","first_line":24,"last_line":191},"calculate_smoke_emissions":{"module":"module_fire_emissions","args":["dt","julday","nlcat","efs_map","fre_avg","ebb_dcycle","area","nblocks","ktau","bb_input_prevh","ebu","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dt","str_len":"100"},"julday":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: julday, nlcat","str_len":"100"},"nlcat":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: julday, nlcat","str_len":"100"},"efs_map":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in),   dimension(ims:ime, jms:jme)            :: efs_map","str_len":"100"},"fre_avg":{"type":"float","dim":["ims:ime","jms:jme","nblocks"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in),   dimension(ims:ime, jms:jme, nblocks)   :: fre_avg","str_len":"100"},"ebb_dcycle":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ebb_dcycle, nblocks, bb_input_prevh","str_len":"100"},"area":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in),   dimension(ims:ime, jms:jme)            :: area","str_len":"100"},"nblocks":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ebb_dcycle, nblocks, bb_input_prevh","str_len":"100"},"ktau":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ktau","str_len":"100"},"bb_input_prevh":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ebb_dcycle, nblocks, bb_input_prevh","str_len":"100"},"ebu":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(inout),dimension(ims:ime, kms:kme, jms:jme)   :: ebu","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ids, ide, jds, jde, kds, kde","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: ims, ime, jms, jme, kms, kme","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: its, ite, jts, jte, kts, kte","str_len":"100"}},"source":"module_fire_emissions.F90","first_line":203,"last_line":257},"diurnal_cycle":{"module":"module_fire_emissions","args":["dtstep","dz8w","rho_phy","pi","ebb_min","chem","num_chem","julday","gmt","xlat","xlong","fire_end_hr","peak_hr","time_int","coef_bb_dc","fire_hist","hwp","hwp_avg","hwp_prev_day","vegfrac","eco_id","nblocks","lu_nofire","lu_qfire","lu_sfire","swdown","ebb_dcycle","ebu","fire_type","q_vap","add_fire_moist_flux","plume_beta_qv","hwp_alpha","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"dtstep":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  dtstep, gmt","str_len":"100"},"dz8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent(in) :: dz8w,rho_phy","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent(in) :: dz8w,rho_phy","str_len":"100"},"pi":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  time_int, pi, ebb_min","str_len":"100"},"ebb_min":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  time_int, pi, ebb_min","str_len":"100"},"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","num_chem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme, num_chem ), intent(inout ) ::                                   chem","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"julday":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"gmt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  dtstep, gmt","str_len":"100"},"xlat":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: xlat,xlong, swdown","str_len":"100"},"xlong":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: xlat,xlong, swdown","str_len":"100"},"fire_end_hr":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp, peak_hr, fire_end_hr","str_len":"100"},"peak_hr":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp, peak_hr, fire_end_hr","str_len":"100"},"time_int":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  time_int, pi, ebb_min","str_len":"100"},"coef_bb_dc":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(out)    :: lu_nofire, lu_qfire, lu_sfire, coef_bb_dc","str_len":"100"},"fire_hist":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(inout)    :: fire_hist","str_len":"100"},"hwp":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp, peak_hr, fire_end_hr","str_len":"100"},"hwp_avg":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp_avg","str_len":"100"},"hwp_prev_day":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp_prev_day","str_len":"100"},"vegfrac":{"type":"float","dim":["ims:ime","jms:jme","20"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  intent(in), dimension(ims:ime,jms:jme,20) :: vegfrac","str_len":"100"},"eco_id":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in), dimension(ims:ime,jms:jme) :: eco_id","str_len":"100"},"nblocks":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) ::  nblocks","str_len":"100"},"lu_nofire":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(out)    :: lu_nofire, lu_qfire, lu_sfire, coef_bb_dc","str_len":"100"},"lu_qfire":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(out)    :: lu_nofire, lu_qfire, lu_sfire, coef_bb_dc","str_len":"100"},"lu_sfire":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(out)    :: lu_nofire, lu_qfire, lu_sfire, coef_bb_dc","str_len":"100"},"swdown":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: xlat,xlong, swdown","str_len":"100"},"ebb_dcycle":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) ::  ebb_dcycle","str_len":"100"},"ebu":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ) ::                                   ebu, q_vap","str_len":"100"},"fire_type":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, dimension(ims:ime,jms:jme), intent(out) :: fire_type","str_len":"100"},"q_vap":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ) ::                                   ebu, q_vap","str_len":"100"},"add_fire_moist_flux":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical, intent(in)  :: add_fire_moist_flux","str_len":"100"},"plume_beta_qv":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: plume_beta_qv","str_len":"100"},"hwp_alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: hwp_alpha","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) ::  julday,num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"}},"source":"module_fire_emissions.F90","first_line":266,"last_line":438},"add_emis_burn":{"module":"module_add_emiss_burn","args":["dtstep","dz8w","rho_phy","pi","ebb_min","chem","num_chem","julday","gmt","xlat","xlong","fire_end_hr","peak_hr","time_int","coef_bb_dc","fire_hist","hwp","hwp_prevd","swdown","ebb_dcycle","ebu","ebu_coarse","ebu_ch4","fire_type","q_vap","add_fire_moist_flux","sc_factor","aod3d","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"dtstep":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  dtstep, gmt","str_len":"100"},"dz8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent(in) :: dz8w,rho_phy","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent(in) :: dz8w,rho_phy","str_len":"100"},"pi":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  time_int, pi, ebb_min","str_len":"100"},"ebb_min":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  time_int, pi, ebb_min","str_len":"100"},"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme, 1:num_chem ), intent(inout ) ::                                   chem","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"julday":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"gmt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  dtstep, gmt","str_len":"100"},"xlat":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: xlat,xlong, swdown","str_len":"100"},"xlong":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: xlat,xlong, swdown","str_len":"100"},"fire_end_hr":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp, peak_hr, fire_end_hr","str_len":"100"},"peak_hr":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp, peak_hr, fire_end_hr","str_len":"100"},"time_int":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) ::  time_int, pi, ebb_min","str_len":"100"},"coef_bb_dc":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(inout)  :: coef_bb_dc","str_len":"100"},"fire_hist":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(inout) :: fire_hist","str_len":"100"},"hwp":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp, peak_hr, fire_end_hr","str_len":"100"},"hwp_prevd":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: hwp_prevd","str_len":"100"},"swdown":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,jms:jme), intent(in)     :: xlat,xlong, swdown","str_len":"100"},"ebb_dcycle":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) ::  ebb_dcycle","str_len":"100"},"ebu":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ) ::                                   ebu, ebu_coarse, ebu_ch4, q_vap","str_len":"100"},"ebu_coarse":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ) ::                                   ebu, ebu_coarse, ebu_ch4, q_vap","str_len":"100"},"ebu_ch4":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ) ::                                   ebu, ebu_coarse, ebu_ch4, q_vap","str_len":"100"},"fire_type":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, dimension(ims:ime,jms:jme), intent(in) :: fire_type","str_len":"100"},"q_vap":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme ), intent(inout ) ::                                   ebu, ebu_coarse, ebu_ch4, q_vap","str_len":"100"},"add_fire_moist_flux":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical, intent(in)  :: add_fire_moist_flux","str_len":"100"},"sc_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: sc_factor","str_len":"100"},"aod3d":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real, dimension(ims:ime,kms:kme,jms:jme), intent(out) ::  aod3d","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"}},"source":"module_add_emiss_burn.F90","first_line":10,"last_line":107},"plumerise":{"module":"module_smoke_plumerise","args":["m1","m2","m3","ia","iz","ja","jz","up","vp","wp","theta","pp","dn0","rv","zt_rams","zm_rams","wind_eff_opt","frp_inst","k1","k2","dbg_opt","g","cp","rgas","cpor","errmsg","errflg","icall","lat","long","curr_secs","alpha","frp_min"],"symbols":{"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ng,m1,m2,m3,ia,iz,ja,jz,ibcon,mynum,i,j,k,imm,ixx,ispc","str_len":"100"},"m2":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ng,m1,m2,m3,ia,iz,ja,jz,ibcon,mynum,i,j,k,imm,ixx,ispc","str_len":"100"},"m3":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ng,m1,m2,m3,ia,iz,ja,jz,ibcon,mynum,i,j,k,imm,ixx,ispc","str_len":"100"},"ia":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ng,m1,m2,m3,ia,iz,ja,jz,ibcon,mynum,i,j,k,imm,ixx,ispc","str_len":"100"},"iz":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ng,m1,m2,m3,ia,iz,ja,jz,ibcon,mynum,i,j,k,imm,ixx,ispc","str_len":"100"},"ja":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ng,m1,m2,m3,ia,iz,ja,jz,ibcon,mynum,i,j,k,imm,ixx,ispc","str_len":"100"},"jz":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ng,m1,m2,m3,ia,iz,ja,jz,ibcon,mynum,i,j,k,imm,ixx,ispc","str_len":"100"},"up":{"type":"float","dim":["m1","m2","m3"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1,m2,m3) :: up, vp, wp,theta,pp,dn0,rv","str_len":"100"},"vp":{"type":"float","dim":["m1","m2","m3"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1,m2,m3) :: up, vp, wp,theta,pp,dn0,rv","str_len":"100"},"wp":{"type":"float","dim":["m1","m2","m3"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1,m2,m3) :: up, vp, wp,theta,pp,dn0,rv","str_len":"100"},"theta":{"type":"float","dim":["m1","m2","m3"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1,m2,m3) :: up, vp, wp,theta,pp,dn0,rv","str_len":"100"},"pp":{"type":"float","dim":["m1","m2","m3"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1,m2,m3) :: up, vp, wp,theta,pp,dn0,rv","str_len":"100"},"dn0":{"type":"float","dim":["m1","m2","m3"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1,m2,m3) :: up, vp, wp,theta,pp,dn0,rv","str_len":"100"},"rv":{"type":"float","dim":["m1","m2","m3"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1,m2,m3) :: up, vp, wp,theta,pp,dn0,rv","str_len":"100"},"zt_rams":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1)       :: zt_rams,zm_rams","str_len":"100"},"zm_rams":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    dimension(m1)       :: zt_rams,zm_rams","str_len":"100"},"wind_eff_opt":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent (in) :: wind_eff_opt","str_len":"100"},"frp_inst":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: frp_inst","str_len":"100"},"k1":{"type":"int","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent (out) :: k1,k2","str_len":"100"},"k2":{"type":"int","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent (out) :: k1,k2","str_len":"100"},"dbg_opt":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical, intent (in) :: dbg_opt","str_len":"100"},"g":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: g, cp, rgas, cpor","str_len":"100"},"cp":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: g, cp, rgas, cpor","str_len":"100"},"rgas":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: g, cp, rgas, cpor","str_len":"100"},"cpor":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: g, cp, rgas, cpor","str_len":"100"},"errmsg":{"type":"FStr","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"character(*), intent(inout) :: errmsg","str_len":"100"},"errflg":{"type":"int","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(inout) :: errflg","str_len":"100"},"icall":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: icall","str_len":"100"},"lat":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  intent(in) ::  lat,long, curr_secs, alpha","str_len":"100"},"long":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  intent(in) ::  lat,long, curr_secs, alpha","str_len":"100"},"curr_secs":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  intent(in) ::  lat,long, curr_secs, alpha","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  intent(in) ::  lat,long, curr_secs, alpha","str_len":"100"},"frp_min":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: frp_min","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":11,"last_line":115},"get_env_condition":{"module":"module_smoke_plumerise","args":["coms","k1","k2","kmt","wind_eff","g","cp","rgas","cpor","errmsg","errflg"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"k1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: k1,k2,k,kcon,klcl,kmt,nk,nkmid,i","str_len":"100"},"k2":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: k1,k2,k,kcon,klcl,kmt,nk,nkmid,i","str_len":"100"},"kmt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: k1,k2,k,kcon,klcl,kmt,nk,nkmid,i","str_len":"100"},"wind_eff":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: wind_eff","str_len":"100"},"g":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: g,cp,rgas,cpor","str_len":"100"},"cp":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: g,cp,rgas,cpor","str_len":"100"},"rgas":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: g,cp,rgas,cpor","str_len":"100"},"cpor":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: g,cp,rgas,cpor","str_len":"100"},"errmsg":{"type":"FStr","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"character(*), intent(inout) :: errmsg","str_len":"100"},"errflg":{"type":"int","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(inout) :: errflg","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":118,"last_line":198},"set_grid":{"module":"module_smoke_plumerise","args":["coms"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":202,"last_line":236},"set_flam_vert":{"module":"module_smoke_plumerise","args":["ztopmax","k1","k2","nkp","zzcon"],"symbols":{"ztopmax":{"type":"float","dim":["2"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)  :: ztopmax(2)","str_len":"100"},"k1":{"type":"int","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer , intent(out) :: k1,k2","str_len":"100"},"k2":{"type":"int","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer , intent(out) :: k1,k2","str_len":"100"},"nkp":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer , intent(in)  :: nkp","str_len":"100"},"zzcon":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)  :: zzcon(nkp)","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":239,"last_line":275},"get_fire_properties":{"module":"module_smoke_plumerise","args":["coms","imm","burnt_area","frp","errmsg","errflg"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"imm":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer ::  moist,  i,  icount,imm","str_len":"100"},"burnt_area":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)::   bfract,  effload,  heat,  hinc ,burnt_area,heat_fluxw,frp","str_len":"100"},"frp":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)::   bfract,  effload,  heat,  hinc ,burnt_area,heat_fluxw,frp","str_len":"100"},"errmsg":{"type":"FStr","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"character(*), intent(inout) :: errmsg","str_len":"100"},"errflg":{"type":"int","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(inout) :: errflg","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":278,"last_line":382},"makeplume":{"module":"module_smoke_plumerise","args":["coms","kmt","ztopmax","ixx","imm","alpha"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"kmt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer ::  izprint, iconv,  itime, k, kk, kkmax, deltak,ilastprint,kmt ,ixx,nrectotal,i_micro,n_sub_step","str_len":"100"},"ztopmax":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) ::  vc, g,  r,  cp,  eps, tmelt,  heatsubl,  heatfus,  heatcond, tfreeze, ztopmax, wmax, rmaxtime, es, esat, heat,dt_save, alpha","str_len":"100"},"ixx":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer ::  izprint, iconv,  itime, k, kk, kkmax, deltak,ilastprint,kmt ,ixx,nrectotal,i_micro,n_sub_step","str_len":"100"},"imm":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer     :: imm, dtknt","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) ::  vc, g,  r,  cp,  eps, tmelt,  heatsubl,  heatfus,  heatcond, tfreeze, ztopmax, wmax, rmaxtime, es, esat, heat,dt_save, alpha","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":385,"last_line":690},"burn":{"module":"module_smoke_plumerise","args":["coms","eflux","water"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"eflux":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: eflux,water","str_len":"100"},"water":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: eflux,water","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":693,"last_line":726},"lbound":{"module":"module_smoke_plumerise","args":["coms","alpha"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: alpha","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":729,"last_line":830},"initial":{"module":"module_smoke_plumerise","args":["coms","kmt","alpha"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"kmt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer ::  isub,  k,  n1,  n2,  n3,  lbuoy,  itmp,  isubm1 ,kmt","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: alpha","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":833,"last_line":901},"damp_grav_wave":{"module":"module_smoke_plumerise","args":["ifrom","nm1","deltak","dt","zt","zm","w","t","tt","qv","qh","qi","qc","te","pe","qvenv"],"symbols":{"ifrom":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer nm1,ifrom,deltak","str_len":"100"},"nm1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer nm1,ifrom,deltak","str_len":"100"},"deltak":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer nm1,ifrom,deltak","str_len":"100"},"dt":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) dt","str_len":"100"},"zt":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"zm":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"w":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"t":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"tt":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"qv":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"qh":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"qi":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"qc":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"te":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"pe":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"},"qvenv":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: w,t,tt,qv,qh,qi,qc,te,pe,qvenv,dummy,zt,zm","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":904,"last_line":922},"friction":{"module":"module_smoke_plumerise","args":["ifrom","nm1","deltak","dt","zt","zm","var1","vart","var2"],"symbols":{"ifrom":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer k,nfpt,kf,nm1,ifrom,deltak","str_len":"100"},"nm1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer k,nfpt,kf,nm1,ifrom,deltak","str_len":"100"},"deltak":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer k,nfpt,kf,nm1,ifrom,deltak","str_len":"100"},"dt":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) zmkf,ztop,distim,c1,c2,dt","str_len":"100"},"zt":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: var1,var2,vart,zt,zm","str_len":"100"},"zm":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: var1,var2,vart,zt,zm","str_len":"100"},"var1":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: var1,var2,vart,zt,zm","str_len":"100"},"vart":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: var1,var2,vart,zt,zm","str_len":"100"},"var2":{"type":"float","dim":["nm1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(nm1) :: var1,var2,vart,zt,zm","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":925,"last_line":957},"vel_advectc_plumerise":{"module":"module_smoke_plumerise","args":["m1","wc","wt","rho","dzm"],"symbols":{"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: k,m1","str_len":"100"},"wc":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: wc,wt,flxw,dzm,rho","str_len":"100"},"wt":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: wc,wt,flxw,dzm,rho","str_len":"100"},"rho":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: wc,wt,flxw,dzm,rho","str_len":"100"},"dzm":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: wc,wt,flxw,dzm,rho","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":960,"last_line":993},"hadvance_plumerise":{"module":"module_smoke_plumerise","args":["iac","m1","dt","wc","wt","wp","mintime"],"symbols":{"iac":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: k,iac","str_len":"100"},"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: m1,mintime","str_len":"100"},"dt":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) eps,dt","str_len":"100"},"wc":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: dummy, wc,wt,wp","str_len":"100"},"wt":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: dummy, wc,wt,wp","str_len":"100"},"wp":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: dummy, wc,wt,wp","str_len":"100"},"mintime":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: m1,mintime","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":996,"last_line":1019},"predict_plumerise":{"module":"module_smoke_plumerise","args":["npts","ac","ap","fa","af","iac","dtlp","epsu"],"symbols":{"npts":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: npts,iac,m","str_len":"100"},"ac":{"type":"float","dim":["*"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(*) :: ac,ap,fa,af","str_len":"100"},"ap":{"type":"float","dim":["*"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(*) :: ac,ap,fa,af","str_len":"100"},"fa":{"type":"float","dim":["*"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(*) :: ac,ap,fa,af","str_len":"100"},"af":{"type":"float","dim":["*"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(*) :: ac,ap,fa,af","str_len":"100"},"iac":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: npts,iac,m","str_len":"100"},"dtlp":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: epsu,dtlp","str_len":"100"},"epsu":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: epsu,dtlp","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1022,"last_line":1065},"buoyancy_plumerise":{"module":"module_smoke_plumerise","args":["m1","t","te","qv","qvenv","qh","qi","qc","wt","scr1"],"symbols":{"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: k,m1","str_len":"100"},"t":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"},"te":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"},"qv":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"},"qvenv":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"},"qh":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"},"qi":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"},"qc":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"},"wt":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"},"scr1":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: t, te, qv, qvenv, qh, qi, qc, wt, scr1","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1068,"last_line":1101},"entrainment":{"module":"module_smoke_plumerise","args":["coms","m1","w","wt","radius","alpha"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: k,m1","str_len":"100"},"w":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: w,wt,radius","str_len":"100"},"wt":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: w,wt,radius","str_len":"100"},"radius":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: w,wt,radius","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) dmdtm,wbar,radius_bar,umgamai,dyn_entr,alpha","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1104,"last_line":1146},"scl_advectc_plumerise":{"module":"module_smoke_plumerise","args":["coms","varn","mzp"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"varn":{"type":"FStr","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"character(len=*) :: varn","str_len":"100"},"mzp":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: mzp","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1149,"last_line":1275},"fa_zc_plumerise":{"module":"module_smoke_plumerise","args":["m1","scp","scr1","vt3dc","vt3df","vt3dg","vt3dk","vctr1","vctr2"],"symbols":{"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: m1,k","str_len":"100"},"scp":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,scr1,vt3dc,vt3df,vt3dg,vt3dk","str_len":"100"},"scr1":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,scr1,vt3dc,vt3df,vt3dg,vt3dk","str_len":"100"},"vt3dc":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,scr1,vt3dc,vt3df,vt3dg,vt3dk","str_len":"100"},"vt3df":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,scr1,vt3dc,vt3df,vt3dg,vt3dk","str_len":"100"},"vt3dg":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,scr1,vt3dc,vt3df,vt3dg,vt3dk","str_len":"100"},"vt3dk":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,scr1,vt3dc,vt3df,vt3dg,vt3dk","str_len":"100"},"vctr1":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: vctr1,vctr2","str_len":"100"},"vctr2":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: vctr1,vctr2","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1278,"last_line":1322},"advtndc_plumerise":{"module":"module_smoke_plumerise","args":["m1","scp","sca","sct","dtl"],"symbols":{"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: m1,k","str_len":"100"},"scp":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,sca,sct","str_len":"100"},"sca":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,sca,sct","str_len":"100"},"sct":{"type":"float","dim":["m1"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind), dimension(m1) :: scp,sca,sct","str_len":"100"},"dtl":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) :: dtl,dtli","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1325,"last_line":1336},"tend0_plumerise":{"module":"module_smoke_plumerise","args":["coms"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1339,"last_line":1351},"scl_misc":{"module":"module_smoke_plumerise","args":["coms","m1","alpha"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer m1,k","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: alpha","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1355,"last_line":1390},"scl_dyn_entrain":{"module":"module_smoke_plumerise","args":["m1","nkp","wbar","w","adiabat","alpha","radius","tt","t","te","qvt","qv","qvenv","qct","qc","qht","qh","qit","qi","vel_e","vel_p","vel_t","rad_p","rad_t"],"symbols":{"m1":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer , intent(in)    :: m1","str_len":"100"},"nkp":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer , intent(in)    :: nkp","str_len":"100"},"wbar":{"type":"float","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: wbar","str_len":"100"},"w":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: w(nkp)","str_len":"100"},"adiabat":{"type":"float","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: adiabat","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: alpha","str_len":"100"},"radius":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: radius(nkp)","str_len":"100"},"tt":{"type":"float","dim":["nkp"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: tt(nkp)","str_len":"100"},"t":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: t(nkp)","str_len":"100"},"te":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: te(nkp)","str_len":"100"},"qvt":{"type":"float","dim":["nkp"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: qvt(nkp)","str_len":"100"},"qv":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: qv(nkp)","str_len":"100"},"qvenv":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: qvenv(nkp)","str_len":"100"},"qct":{"type":"float","dim":["nkp"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: qct(nkp)","str_len":"100"},"qc":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: qc(nkp)","str_len":"100"},"qht":{"type":"float","dim":["nkp"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: qht(nkp)","str_len":"100"},"qh":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: qh(nkp)","str_len":"100"},"qit":{"type":"float","dim":["nkp"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: qit(nkp)","str_len":"100"},"qi":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: qi(nkp)","str_len":"100"},"vel_e":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: vel_e(nkp)","str_len":"100"},"vel_p":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: vel_p(nkp)","str_len":"100"},"vel_t":{"type":"float","dim":["nkp"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: vel_t(nkp)","str_len":"100"},"rad_p":{"type":"float","dim":["nkp"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(in)    :: rad_p(nkp)","str_len":"100"},"rad_t":{"type":"float","dim":["nkp"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind)    , intent(inout) :: rad_t(nkp)","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1393,"last_line":1459},"visc_w":{"module":"module_smoke_plumerise","args":["coms","m1","deltak","kmt"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer m1,k,deltak,kmt,m2","str_len":"100"},"deltak":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer m1,k,deltak,kmt,m2","str_len":"100"},"kmt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer m1,k,deltak,kmt,m2","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1463,"last_line":1514},"update_plumerise":{"module":"module_smoke_plumerise","args":["coms","m1","varn"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer m1,k","str_len":"100"},"varn":{"type":"FStr","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"character(len=*) :: varn","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1518,"last_line":1561},"fallpart":{"module":"module_smoke_plumerise","args":["coms","m1"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"},"m1":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer m1,k","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1564,"last_line":1627},"waterbal":{"module":"module_smoke_plumerise","args":["coms"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1630,"last_line":1655},"evaporate":{"module":"module_smoke_plumerise","args":["coms"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1657,"last_line":1870},"convert":{"module":"module_smoke_plumerise","args":["coms"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1873,"last_line":1947},"sublimate":{"module":"module_smoke_plumerise","args":["coms"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":1951,"last_line":2000},"glaciate":{"module":"module_smoke_plumerise","args":["coms"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":2004,"last_line":2054},"melt":{"module":"module_smoke_plumerise","args":["coms"],"symbols":{"coms":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"type(plumegen_coms), pointer :: coms","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":2058,"last_line":2100},"htint":{"module":"module_smoke_plumerise","args":["nzz1","vctra","eleva","nzz2","vctrb","elevb","errmsg","errflg"],"symbols":{"nzz1":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: nzz1","str_len":"100"},"vctra":{"type":"float","dim":["nzz1"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    intent(in ) :: vctra(nzz1)","str_len":"100"},"eleva":{"type":"float","dim":["nzz1"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    intent(in ) :: eleva(nzz1)","str_len":"100"},"nzz2":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: nzz2","str_len":"100"},"vctrb":{"type":"float","dim":["nzz2"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    intent(out) :: vctrb(nzz2)","str_len":"100"},"elevb":{"type":"float","dim":["nzz2"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),    intent(in ) :: elevb(nzz2)","str_len":"100"},"errmsg":{"type":"FStr","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"character(*), intent(inout) :: errmsg","str_len":"100"},"errflg":{"type":"int","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(inout) :: errflg","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":2102,"last_line":2143},"esat_pr":{"module":"module_smoke_plumerise","args":["tem"],"symbols":{"tem":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind) temc , tem,esatm","str_len":"100"}},"source":"module_smoke_plumerise.F90","first_line":2145,"last_line":2173},"dry_dep_driver_simple":{"module":"dep_dry_simple_mod","args":["rmol","ust","ndvel","ddvel","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"rmol":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )        , intent(inout) :: ust, rmol","str_len":"100"},"ust":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),  dimension( ims:ime , jms:jme )        , intent(inout) :: ust, rmol","str_len":"100"},"ndvel":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ddvel":{"type":"float","dim":["its:ite","jts:jte","ndvel"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( its:ite, jts:jte, ndvel ), intent(inout) ::   ddvel","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ndvel, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"}},"source":"dep_dry_simple_mod.F90","first_line":16,"last_line":62},"aero_rad_init":{"module":"rad_data_mod","args":[],"symbols":{},"source":"rad_data_mod.F90","first_line":17,"last_line":64},"aero_dry_dep_init":{"module":"dep_data_mod","args":[],"symbols":{},"source":"dep_data_mod.F90","first_line":59,"last_line":105},"gocart_dust_fengsha_driver":{"module":"dust_fengsha_mod","args":["dt","ktau","chem","rho_phy","smois","stemp","p8w","isltyp","snowh","xland","area","g","ust","znt","clay","sand","uthr","uthr_sg","albedo_drag","feff","sep","e_dust_out","num_e_dust_out","index_e_dust_out_dust_fine","index_e_dust_out_dust_coarse","num_emis_dust","num_chem","num_soil_layers","dust_alpha","dust_gamma","dust_drylimit_factor","dust_moist_correction","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dt","str_len":"100"},"ktau":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme, 1:num_chem ), intent(inout) :: chem","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , kms:kme , jms:jme ), intent(in) :: rho_phy","str_len":"100"},"smois":{"type":"float","dim":["ims:ime","1:num_soil_layers","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, 1:num_soil_layers, jms:jme ), intent(in) :: smois, stemp","str_len":"100"},"stemp":{"type":"float","dim":["ims:ime","1:num_soil_layers","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, 1:num_soil_layers, jms:jme ), intent(in) :: smois, stemp","str_len":"100"},"p8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , kms:kme , jms:jme ), intent(in) :: p8w","str_len":"100"},"isltyp":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,         dimension( ims:ime , jms:jme ), intent(in) :: isltyp","str_len":"100"},"snowh":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: snowh","str_len":"100"},"xland":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: xland","str_len":"100"},"area":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: area","str_len":"100"},"g":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: g","str_len":"100"},"ust":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: ust","str_len":"100"},"znt":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: znt","str_len":"100"},"clay":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: clay","str_len":"100"},"sand":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: sand","str_len":"100"},"uthr":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: uthr","str_len":"100"},"uthr_sg":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: uthr_sg","str_len":"100"},"albedo_drag":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: albedo_drag","str_len":"100"},"feff":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: feff","str_len":"100"},"sep":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: sep","str_len":"100"},"e_dust_out":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_e_dust_out"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme,1:num_e_dust_out), intent(inout) :: e_dust_out","str_len":"100"},"num_e_dust_out":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"index_e_dust_out_dust_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"index_e_dust_out_dust_coarse":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"num_emis_dust":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"num_soil_layers":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"dust_alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dust_alpha, dust_gamma, dust_drylimit_factor, dust_moist_correction","str_len":"100"},"dust_gamma":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dust_alpha, dust_gamma, dust_drylimit_factor, dust_moist_correction","str_len":"100"},"dust_drylimit_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dust_alpha, dust_gamma, dust_drylimit_factor, dust_moist_correction","str_len":"100"},"dust_moist_correction":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dust_alpha, dust_gamma, dust_drylimit_factor, dust_moist_correction","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_emis_dust,num_chem,num_soil_layers, num_e_dust_out, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse,ktau","str_len":"100"}},"source":"dust_fengsha_mod.F90","first_line":24,"last_line":218},"source_dust":{"module":"dust_fengsha_mod","args":["imx","jmx","lmx","nmx","dt1","tc","ustar","massfrac","erod","dxy","smois","airden","airmas","bems","g0","alpha","gamma","r","uthres","source_out_diag","dust_drylimit_factor"],"symbols":{"imx":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,     intent(in)    :: imx,jmx,lmx,nmx","str_len":"100"},"jmx":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,     intent(in)    :: imx,jmx,lmx,nmx","str_len":"100"},"lmx":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,     intent(in)    :: imx,jmx,lmx,nmx","str_len":"100"},"nmx":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,     intent(in)    :: imx,jmx,lmx,nmx","str_len":"100"},"dt1":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: dt1","str_len":"100"},"tc":{"type":"float","dim":["nmx"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(inout) :: tc(nmx)","str_len":"100"},"ustar":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: ustar","str_len":"100"},"massfrac":{"type":"float","dim":["3"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: massfrac(3)","str_len":"100"},"erod":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: erod","str_len":"100"},"dxy":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: dxy","str_len":"100"},"smois":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: smois","str_len":"100"},"airden":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: airden","str_len":"100"},"airmas":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: airmas","str_len":"100"},"bems":{"type":"float","dim":["nmx"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(inout)   :: bems(nmx)","str_len":"100"},"g0":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: g0","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: alpha","str_len":"100"},"gamma":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: gamma","str_len":"100"},"r":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: r","str_len":"100"},"uthres":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: uthres","str_len":"100"},"source_out_diag":{"type":"float","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(out) :: source_out_diag","str_len":"100"},"dust_drylimit_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: dust_drylimit_factor","str_len":"100"}},"source":"dust_fengsha_mod.F90","first_line":221,"last_line":394},"fengsha_drag":{"module":"dust_fengsha_mod","args":["z0","r"],"symbols":{"z0":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: z0","str_len":"100"},"r":{"type":"float","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(out) :: r","str_len":"100"}},"source":"dust_fengsha_mod.F90","first_line":397,"last_line":418},"dustemissionfengsha":{"module":"dust_fengsha_mod","args":["slc","clay","sand","silt","sep","feff","airdens","ustar","uthrs","alpha","gamma","kvhmax","grav","rhop","emissions","reason","dust_drylimit_factor"],"symbols":{"slc":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: slc","str_len":"100"},"clay":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: clay","str_len":"100"},"sand":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: sand","str_len":"100"},"silt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: silt","str_len":"100"},"sep":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: sep","str_len":"100"},"feff":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: feff","str_len":"100"},"airdens":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: airdens","str_len":"100"},"ustar":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: ustar","str_len":"100"},"uthrs":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: uthrs","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: alpha","str_len":"100"},"gamma":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: gamma","str_len":"100"},"kvhmax":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: kvhmax","str_len":"100"},"grav":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: grav","str_len":"100"},"rhop":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: rhop","str_len":"100"},"emissions":{"type":"float","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(inout) :: emissions","str_len":"100"},"reason":{"type":"float","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(out)     :: reason","str_len":"100"},"dust_drylimit_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dust_drylimit_factor","str_len":"100"}},"source":"dust_fengsha_mod.F90","first_line":420,"last_line":521},"soilmoistureconvertvol2grav":{"module":"dust_fengsha_mod","args":["vsoil","sandfrac"],"symbols":{"vsoil":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: vsoil","str_len":"100"},"sandfrac":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: sandfrac","str_len":"100"}},"source":"dust_fengsha_mod.F90","first_line":523,"last_line":556},"moisturecorrectionfecan":{"module":"dust_fengsha_mod","args":["slc","sand","clay","dust_drylimit_factor"],"symbols":{"slc":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: slc","str_len":"100"},"sand":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: sand","str_len":"100"},"clay":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: clay","str_len":"100"},"dust_drylimit_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dust_drylimit_factor","str_len":"100"}},"source":"dust_fengsha_mod.F90","first_line":558,"last_line":593},"moisturecorrectionshao":{"module":"dust_fengsha_mod","args":["slc"],"symbols":{"slc":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: slc","str_len":"100"}},"source":"dust_fengsha_mod.F90","first_line":595,"last_line":624},"dustfluxv2hratiomb95":{"module":"dust_fengsha_mod","args":["clay","kvhmax"],"symbols":{"clay":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: clay","str_len":"100"},"kvhmax":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: kvhmax","str_len":"100"}},"source":"dust_fengsha_mod.F90","first_line":626,"last_line":656},"wetdep_ls":{"module":"module_wetdep_ls","args":["dt","gravity","var","rain","moist","rho","nchem","dz8w","vvel","p_phy","alpha","wetdep_resolved","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dt, gravity","str_len":"100"},"gravity":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dt, gravity","str_len":"100"},"var":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:nchem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme,1:nchem),intent(inout) :: var","str_len":"100"},"rain":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, jms:jme),         intent(in) :: rain","str_len":"100"},"moist":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme),intent(in) :: moist, p_phy","str_len":"100"},"rho":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme),intent(in) :: rho,dz8w,vvel","str_len":"100"},"nchem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"dz8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme),intent(in) :: rho,dz8w,vvel","str_len":"100"},"vvel":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme),intent(in) :: rho,dz8w,vvel","str_len":"100"},"p_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme),intent(in) :: moist, p_phy","str_len":"100"},"alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: alpha","str_len":"100"},"wetdep_resolved":{"type":"float","dim":["ims:ime","jms:jme","1:nchem"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, jms:jme, 1:nchem ),       intent(out)   ::  wetdep_resolved","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in) :: nchem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"}},"source":"module_wetdep_ls.F90","first_line":10,"last_line":89},"set_scalar_indices":{"module":"mpas_smoke_init","args":["chemistry_start","index_smoke_fine","index_smoke_coarse","index_polp_tree","index_polp_grass","index_polp_weed","index_pols_tree","index_pols_grass","index_pols_weed","index_pols_all","index_polp_all","index_unspc_fine","index_unspc_coarse","index_ssalt_fine","index_ssalt_coarse","index_no3_a_fine","index_so4_a_fine","index_nh4_a_fine","index_so2","index_nh3","index_ch4","index_dust_fine","index_dust_coarse"],"symbols":{"chemistry_start":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_smoke_fine":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_smoke_coarse":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_polp_tree":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_polp_grass":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_polp_weed":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_pols_tree":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_pols_grass":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_pols_weed":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_pols_all":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_polp_all":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_unspc_fine":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_unspc_coarse":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_ssalt_fine":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_ssalt_coarse":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_no3_a_fine":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_so4_a_fine":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_nh4_a_fine":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_so2":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_nh3":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_ch4":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_dust_fine":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"},"index_dust_coarse":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: chemistry_start, index_smoke_fine, index_smoke_coarse, index_dust_fine, index_dust_coarse, index_polp_tree, index_polp_grass, index_polp_weed, index_pols_tree, index_pols_grass, index_pols_weed, index_pols_all,  index_polp_all, index_unspc_fine, index_unspc_coarse, index_ssalt_fine, index_ssalt_coarse, index_no3_a_fine, index_so4_a_fine, index_nh4_a_fine, index_so2, index_nh3, index_ch4","str_len":"100"}},"source":"mpas_smoke_init.F90","first_line":43,"last_line":90},"seasaltemission":{"module":"seas_ngac_mod","args":["rlow","rup","method","w10m","ustar","pi","memissions","nemissions","rc"],"symbols":{"rlow":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)           :: rlow, rup","str_len":"100"},"rup":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)           :: rlow, rup","str_len":"100"},"method":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in)           :: method","str_len":"100"},"w10m":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)           :: w10m","str_len":"100"},"ustar":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)           :: ustar","str_len":"100"},"pi":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(in)           :: pi","str_len":"100"},"memissions":{"type":"float","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(inout)        :: memissions","str_len":"100"},"nemissions":{"type":"float","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),    intent(inout)        :: nemissions","str_len":"100"},"rc":{"type":"int","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(out)          :: rc","str_len":"100"}},"source":"seas_ngac_mod.F90","first_line":58,"last_line":162},"seasaltemissiongong":{"module":"seas_ngac_mod","args":["r","dr","w","scalefac","afac","bfac","rpow","exppow","wpow"],"symbols":{"r":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: r, dr","str_len":"100"},"dr":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: r, dr","str_len":"100"},"w":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: w","str_len":"100"},"scalefac":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: scalefac, afac, bfac, rpow, exppow, wpow","str_len":"100"},"afac":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: scalefac, afac, bfac, rpow, exppow, wpow","str_len":"100"},"bfac":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: scalefac, afac, bfac, rpow, exppow, wpow","str_len":"100"},"rpow":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: scalefac, afac, bfac, rpow, exppow, wpow","str_len":"100"},"exppow":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: scalefac, afac, bfac, rpow, exppow, wpow","str_len":"100"},"wpow":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: scalefac, afac, bfac, rpow, exppow, wpow","str_len":"100"}},"source":"seas_ngac_mod.F90","first_line":170,"last_line":186},"get_thread_coms":{"module":"module_zero_plumegen_coms","args":[],"symbols":{},"source":"module_zero_plumegen_coms.F90","first_line":66,"last_line":74},"plumegen_coms_zero":{"module":"module_zero_plumegen_coms","args":["this"],"symbols":{"this":{"type":"Plume*","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"class(plumegen_coms) :: this","str_len":"100"}},"source":"module_zero_plumegen_coms.F90","first_line":76,"last_line":186},"mpas_smoke_anthro_emis_driver":{"module":"module_anthro_emissions","args":["dt","gmt","julday","kemit","xlat","xlong","chem","num_chem","dz8w","t_phy","rho_phy","e_ant_in","e_ant_out","num_e_ant_in","num_e_ant_out","index_e_ant_in_unspc_fine","index_e_ant_in_unspc_coarse","index_e_ant_out_unspc_fine","index_e_ant_out_unspc_coarse","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in    ) :: dt,gmt","str_len":"100"},"gmt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in    ) :: dt,gmt","str_len":"100"},"julday":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"kemit":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"xlat":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,jms:jme),intent(in) :: xlat,xlong","str_len":"100"},"xlong":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,jms:jme),intent(in) :: xlat,xlong","str_len":"100"},"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme,1:num_chem), intent(inout)     :: chem","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"dz8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in) :: dz8w,rho_phy,t_phy","str_len":"100"},"t_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in) :: dz8w,rho_phy,t_phy","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme),intent(in) :: dz8w,rho_phy,t_phy","str_len":"100"},"e_ant_in":{"type":"float","dim":["ims:ime","1:kemit","jms:jme","1:num_e_ant_in"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,1:kemit,jms:jme,1:num_e_ant_in), intent(in)    :: e_ant_in","str_len":"100"},"e_ant_out":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_e_ant_out"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),dimension(ims:ime,kms:kme,jms:jme,1:num_e_ant_out),intent(inout) :: e_ant_out","str_len":"100"},"num_e_ant_in":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"num_e_ant_out":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"index_e_ant_in_unspc_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"index_e_ant_in_unspc_coarse":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"index_e_ant_out_unspc_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"index_e_ant_out_unspc_coarse":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,      intent(in   ) :: julday, num_chem, kemit, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte, num_e_ant_in, num_e_ant_out, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"}},"source":"module_anthro_emissions.F90","first_line":21,"last_line":83},"dry_dep_driver_emerson":{"module":"dep_dry_mod_emerson","args":["rmol","ustar","znt","num_chem","ddvel","chem","delz","snowh","t_phy","p_phy","rho_phy","ivgtyp","gravity","dt","drydep_flux","tend_chem_settle","dbg_opt","settling_opt","vg","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte","curr_secs"],"symbols":{"rmol":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme )        , intent(in) :: ustar, rmol, znt, snowh","str_len":"100"},"ustar":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme )        , intent(in) :: ustar, rmol, znt, snowh","str_len":"100"},"znt":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme )        , intent(in) :: ustar, rmol, znt, snowh","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ddvel":{"type":"float","dim":["ims:ime","jms:jme","1:num_chem"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, jms:jme, 1:num_chem ), intent(out)         :: ddvel","str_len":"100"},"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme, 1:num_chem ), intent(in ) :: chem","str_len":"100"},"delz":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: t_phy, rho_phy, p_phy, delz","str_len":"100"},"snowh":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , jms:jme )        , intent(in) :: ustar, rmol, znt, snowh","str_len":"100"},"t_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: t_phy, rho_phy, p_phy, delz","str_len":"100"},"p_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: t_phy, rho_phy, p_phy, delz","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime , kms:kme , jms:jme ), intent(in   ) :: t_phy, rho_phy, p_phy, delz","str_len":"100"},"ivgtyp":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, dimension(ims:ime,jms:jme), intent(in) ::  ivgtyp","str_len":"100"},"gravity":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: gravity,dt","str_len":"100"},"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: gravity,dt","str_len":"100"},"drydep_flux":{"type":"float","dim":["ims:ime","jms:jme","1:num_chem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, jms:jme, 1:num_chem ), intent(inout)       :: drydep_flux","str_len":"100"},"tend_chem_settle":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme, 1:num_chem), intent(out) :: tend_chem_settle, vg","str_len":"100"},"dbg_opt":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical, intent(in) :: dbg_opt","str_len":"100"},"settling_opt":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: settling_opt","str_len":"100"},"vg":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension( ims:ime, kms:kme, jms:jme, 1:num_chem), intent(out) :: tend_chem_settle, vg","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,  intent(in   ) ::  num_chem, ids,ide, jds,jde, kds,kde, ims,ime, jms,jme, kms,kme, its,ite, jts,jte, kts,kte","str_len":"100"},"curr_secs":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: curr_secs","str_len":"100"}},"source":"dep_dry_mod_emerson.F90","first_line":23,"last_line":229},"depvel":{"module":"dep_dry_mod_emerson","args":["rmol","zr","z0","ustar","aer_res"],"symbols":{"rmol":{"type":"float","dim":[],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(inout) :: rmol","str_len":"100"},"zr":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: ustar, z0, zr","str_len":"100"},"z0":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: ustar, z0, zr","str_len":"100"},"ustar":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in)    :: ustar, z0, zr","str_len":"100"},"aer_res":{"type":"float","dim":[],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(out)   :: aer_res","str_len":"100"}},"source":"dep_dry_mod_emerson.F90","first_line":233,"last_line":298},"particle_settling_wrapper":{"module":"dep_dry_mod_emerson","args":["tend_chem_settle","chem","rho_phy","delz_flip","vg","dt","kts","kte","its","ite","jts","jte","num_chem","ims","ime","jms","jme","kms","kme"],"symbols":{"tend_chem_settle":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme,1:num_chem), intent(inout) :: tend_chem_settle","str_len":"100"},"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme,1:num_chem), intent(in) :: chem","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent (in)  :: rho_phy","str_len":"100"},"delz_flip":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme), intent (in)  :: delz_flip","str_len":"100"},"vg":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), dimension(ims:ime,kms:kme,jms:jme,1:num_chem), intent(in) :: vg","str_len":"100"},"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind), intent(in) :: dt","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in ) :: kts, kte,its,ite,jts,jte,num_chem,ims,ime, jms,jme, kms,kme","str_len":"100"}},"source":"dep_dry_mod_emerson.F90","first_line":350,"last_line":419},"smoke_from_mpas_fields":{"module":"smoke_driver_slice","args":["add_fire_heat_flux","add_fire_moist_flux","albedo_drag","albedo_drag_p","aod3d_p","area_p","areacell","chem","chem_p","clayfrac_in","clayfrac_in_p","cldfrac","cldfrac_p","coef_bb_dc","coef_bb_dc_p","config_anthro_scheme","config_convection_scheme","config_dust_scheme","config_rwc_scheme","config_smoke_scheme","coszr","coszr_p","ddvel_p","degrad","dpt2m_p","drydep_flux","drydep_flux_p","drydep_opt","dx_p","e_ant_in","e_ant_in_p","e_ant_out_p","e_bb_in","e_bb_in_p","e_bb_out","e_bb_out_p","e_dust_out","e_dust_out_p","ebb_dcycle","eco_id","eco_id_p","efs_flam","efs_flam_p","efs_rsmold","efs_rsmold_p","efs_smold","efs_smold_p","feff","feff_p","fire_ef_out","fire_ef_out_p","fire_end_hr","fire_end_hr_p","fmc_avg_p","fmc_prev24","frac_grid_burned","frac_grid_burned_p","fre_avg_p","fre_in","fre_in_p","fre_out","fre_out_p","fre_prev24","frp_avg_p","frp_in","frp_in_p","frp_out","frp_out_p","frp_prev24","hfx_bb_p","hpbl","hpbl_p","hwp","hwp_avg_p","hwp_p","hwp_prev24","isltyp","isltyp_p","ite","its","ivgtyp","ivgtyp_p","jte","jts","kanthro","kpbl","kpbl_p","kreswoodcomb","kte","ktop_deep","ktop_deep_p","kts","kwildfire","landusef","landusef_p","latcell","len_disp","loncell","mavail","mavail_p","max_bb_plume_p","meshdensity","min_bb_plume_p","nblocks","num_chem","num_e_ant_in","num_e_ant_out","num_e_bb_in","num_e_bb_out","num_e_dust_out","num_landcat","num_soils","plumerise_opt","qc_bl","qcbl_p","qfx_bb_p","qi_bl","qibl_p","raincv","raincv_p","rainncv","rainncv_p","rmol","rmol_p","rwc_annual_sum","rwc_annual_sum_p","rwc_annual_sum_smoke_coarse","rwc_annual_sum_smoke_coarse_p","rwc_annual_sum_smoke_fine","rwc_annual_sum_smoke_fine_p","rwc_annual_sum_unspc_coarse","rwc_annual_sum_unspc_coarse_p","rwc_annual_sum_unspc_fine","rwc_annual_sum_unspc_fine_p","rwc_denominator","rwc_denominator_p","sandfrac_in","sandfrac_in_p","sep_in","sep_in_p","skintemp","smois","smois_p","snowh","snowh_p","t_size","tend_chem_settle_p","totprcp_prev24","totprcp_prev24_p","tsk_p","tslb","tslb_p","u10","u10_p","ust","ust_p","uthres_in","uthres_in_p","uthres_sg_in","uthres_sg_in_p","v10","v10_p","vegfra","vegfra_p","wetdep_ls_opt","wetdep_resolved","wetdep_resolved_p","xland","xland_p","xlat_p","xlon_p","z0","z0_p","zgrid","zgrid_p","znt","znt_p"],"symbols":{"add_fire_heat_flux":{"type":"bool","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"logical :: add_fire_heat_flux","str_len":"100"},"add_fire_moist_flux":{"type":"bool","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"logical :: add_fire_moist_flux","str_len":"100"},"albedo_drag":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: albedo_drag(:)","str_len":"100"},"albedo_drag_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: albedo_drag_p(:,:)","str_len":"100"},"aod3d_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod3d_p(:,:,:)","str_len":"100"},"area_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: area_p(:,:)","str_len":"100"},"areacell":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: areacell(:)","str_len":"100"},"chem":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: chem(:,:,:)","str_len":"100"},"chem_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: chem_p(:,:,:,:)","str_len":"100"},"clayfrac_in":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: clayfrac_in(:)","str_len":"100"},"clayfrac_in_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: clayfrac_in_p(:,:)","str_len":"100"},"cldfrac":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: cldfrac(:,:)","str_len":"100"},"cldfrac_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: cldfrac_p(:,:,:)","str_len":"100"},"coef_bb_dc":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: coef_bb_dc(:)","str_len":"100"},"coef_bb_dc_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: coef_bb_dc_p(:,:)","str_len":"100"},"config_anthro_scheme":{"type":"FStr","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"character(100) :: config_anthro_scheme","str_len":"100"},"config_convection_scheme":{"type":"FStr","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"character(100) :: config_convection_scheme","str_len":"100"},"config_dust_scheme":{"type":"FStr","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"character(100) :: config_dust_scheme","str_len":"100"},"config_rwc_scheme":{"type":"FStr","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"character(100) :: config_rwc_scheme","str_len":"100"},"config_smoke_scheme":{"type":"FStr","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"character(100) :: config_smoke_scheme","str_len":"100"},"coszr":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: coszr(:)","str_len":"100"},"coszr_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: coszr_p(:,:)","str_len":"100"},"ddvel_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: ddvel_p(:,:,:)","str_len":"100"},"degrad":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: degrad","str_len":"100"},"dpt2m_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: dpt2m_p(:,:)","str_len":"100"},"drydep_flux":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: drydep_flux(:,:)","str_len":"100"},"drydep_flux_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: drydep_flux_p(:,:,:)","str_len":"100"},"drydep_opt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: drydep_opt","str_len":"100"},"dx_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: dx_p(:,:)","str_len":"100"},"e_ant_in":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_ant_in(:,:,:)","str_len":"100"},"e_ant_in_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_ant_in_p(:,:,:,:)","str_len":"100"},"e_ant_out_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_ant_out_p(:,:,:,:)","str_len":"100"},"e_bb_in":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_bb_in(:,:,:)","str_len":"100"},"e_bb_in_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_bb_in_p(:,:,:,:)","str_len":"100"},"e_bb_out":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_bb_out(:,:,:)","str_len":"100"},"e_bb_out_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_bb_out_p(:,:,:,:)","str_len":"100"},"e_dust_out":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_dust_out(:,:,:)","str_len":"100"},"e_dust_out_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_dust_out_p(:,:,:,:)","str_len":"100"},"ebb_dcycle":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ebb_dcycle","str_len":"100"},"eco_id":{"type":"int","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: eco_id(:)","str_len":"100"},"eco_id_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: eco_id_p(:,:)","str_len":"100"},"efs_flam":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: efs_flam(:)","str_len":"100"},"efs_flam_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: efs_flam_p(:,:)","str_len":"100"},"efs_rsmold":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: efs_rsmold(:)","str_len":"100"},"efs_rsmold_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: efs_rsmold_p(:,:)","str_len":"100"},"efs_smold":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: efs_smold(:)","str_len":"100"},"efs_smold_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: efs_smold_p(:,:)","str_len":"100"},"feff":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: feff(:)","str_len":"100"},"feff_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: feff_p(:,:)","str_len":"100"},"fire_ef_out":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fire_ef_out(:)","str_len":"100"},"fire_ef_out_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fire_ef_out_p(:,:)","str_len":"100"},"fire_end_hr":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fire_end_hr(:)","str_len":"100"},"fire_end_hr_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fire_end_hr_p(:,:)","str_len":"100"},"fmc_avg_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fmc_avg_p(:,:)","str_len":"100"},"fmc_prev24":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fmc_prev24(:,:)","str_len":"100"},"frac_grid_burned":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frac_grid_burned(:)","str_len":"100"},"frac_grid_burned_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frac_grid_burned_p(:,:)","str_len":"100"},"fre_avg_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fre_avg_p(:,:,:)","str_len":"100"},"fre_in":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fre_in(:)","str_len":"100"},"fre_in_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fre_in_p(:,:)","str_len":"100"},"fre_out":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fre_out(:)","str_len":"100"},"fre_out_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fre_out_p(:,:)","str_len":"100"},"fre_prev24":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fre_prev24(:,:)","str_len":"100"},"frp_avg_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frp_avg_p(:,:,:)","str_len":"100"},"frp_in":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frp_in(:)","str_len":"100"},"frp_in_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frp_in_p(:,:)","str_len":"100"},"frp_out":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frp_out(:)","str_len":"100"},"frp_out_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frp_out_p(:,:)","str_len":"100"},"frp_prev24":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frp_prev24(:,:)","str_len":"100"},"hfx_bb_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hfx_bb_p(:,:)","str_len":"100"},"hpbl":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hpbl(:)","str_len":"100"},"hpbl_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hpbl_p(:,:)","str_len":"100"},"hwp":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hwp(:)","str_len":"100"},"hwp_avg_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hwp_avg_p(:,:,:)","str_len":"100"},"hwp_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hwp_p(:,:)","str_len":"100"},"hwp_prev24":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hwp_prev24(:,:)","str_len":"100"},"isltyp":{"type":"int","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: isltyp(:)","str_len":"100"},"isltyp_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: isltyp_p(:,:)","str_len":"100"},"ite":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ite","str_len":"100"},"its":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: its","str_len":"100"},"ivgtyp":{"type":"int","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ivgtyp(:)","str_len":"100"},"ivgtyp_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ivgtyp_p(:,:)","str_len":"100"},"jte":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: jte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: jts","str_len":"100"},"kanthro":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kanthro","str_len":"100"},"kpbl":{"type":"int","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kpbl(:)","str_len":"100"},"kpbl_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kpbl_p(:,:)","str_len":"100"},"kreswoodcomb":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kreswoodcomb","str_len":"100"},"kte":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kte","str_len":"100"},"ktop_deep":{"type":"int","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ktop_deep(:)","str_len":"100"},"ktop_deep_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ktop_deep_p(:,:)","str_len":"100"},"kts":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kts","str_len":"100"},"kwildfire":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kwildfire","str_len":"100"},"landusef":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: landusef(:,:)","str_len":"100"},"landusef_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: landusef_p(:,:,:)","str_len":"100"},"latcell":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: latcell(:)","str_len":"100"},"len_disp":{"type":"float","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: len_disp","str_len":"100"},"loncell":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: loncell(:)","str_len":"100"},"mavail":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: mavail(:)","str_len":"100"},"mavail_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: mavail_p(:,:)","str_len":"100"},"max_bb_plume_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: max_bb_plume_p(:,:)","str_len":"100"},"meshdensity":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: meshdensity(:)","str_len":"100"},"min_bb_plume_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: min_bb_plume_p(:,:)","str_len":"100"},"nblocks":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: nblocks","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_chem","str_len":"100"},"num_e_ant_in":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_e_ant_in","str_len":"100"},"num_e_ant_out":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_e_ant_out","str_len":"100"},"num_e_bb_in":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_e_bb_in","str_len":"100"},"num_e_bb_out":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_e_bb_out","str_len":"100"},"num_e_dust_out":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_e_dust_out","str_len":"100"},"num_landcat":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_landcat","str_len":"100"},"num_soils":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_soils","str_len":"100"},"plumerise_opt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: plumerise_opt","str_len":"100"},"qc_bl":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: qc_bl(:,:)","str_len":"100"},"qcbl_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: qcbl_p(:,:,:)","str_len":"100"},"qfx_bb_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: qfx_bb_p(:,:)","str_len":"100"},"qi_bl":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: qi_bl(:,:)","str_len":"100"},"qibl_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: qibl_p(:,:,:)","str_len":"100"},"raincv":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: raincv(:)","str_len":"100"},"raincv_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: raincv_p(:,:)","str_len":"100"},"rainncv":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rainncv(:)","str_len":"100"},"rainncv_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rainncv_p(:,:)","str_len":"100"},"rmol":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rmol(:)","str_len":"100"},"rmol_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rmol_p(:,:)","str_len":"100"},"rwc_annual_sum":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum(:,:)","str_len":"100"},"rwc_annual_sum_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_p(:,:,:)","str_len":"100"},"rwc_annual_sum_smoke_coarse":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_smoke_coarse(:,:)","str_len":"100"},"rwc_annual_sum_smoke_coarse_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_smoke_coarse_p(:,:,:)","str_len":"100"},"rwc_annual_sum_smoke_fine":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_smoke_fine(:,:)","str_len":"100"},"rwc_annual_sum_smoke_fine_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_smoke_fine_p(:,:,:)","str_len":"100"},"rwc_annual_sum_unspc_coarse":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_unspc_coarse(:,:)","str_len":"100"},"rwc_annual_sum_unspc_coarse_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_unspc_coarse_p(:,:,:)","str_len":"100"},"rwc_annual_sum_unspc_fine":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_unspc_fine(:,:)","str_len":"100"},"rwc_annual_sum_unspc_fine_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_annual_sum_unspc_fine_p(:,:,:)","str_len":"100"},"rwc_denominator":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_denominator(:)","str_len":"100"},"rwc_denominator_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rwc_denominator_p(:,:)","str_len":"100"},"sandfrac_in":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: sandfrac_in(:)","str_len":"100"},"sandfrac_in_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: sandfrac_in_p(:,:)","str_len":"100"},"sep_in":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: sep_in(:)","str_len":"100"},"sep_in_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: sep_in_p(:,:)","str_len":"100"},"skintemp":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: skintemp(:)","str_len":"100"},"smois":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: smois(:,:)","str_len":"100"},"smois_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: smois_p(:,:,:)","str_len":"100"},"snowh":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: snowh(:)","str_len":"100"},"snowh_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: snowh_p(:,:)","str_len":"100"},"t_size":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: t_size","str_len":"100"},"tend_chem_settle_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: tend_chem_settle_p(:,:,:,:)","str_len":"100"},"totprcp_prev24":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: totprcp_prev24(:,:)","str_len":"100"},"totprcp_prev24_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: totprcp_prev24_p(:,:)","str_len":"100"},"tsk_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: tsk_p(:,:)","str_len":"100"},"tslb":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: tslb(:,:)","str_len":"100"},"tslb_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: tslb_p(:,:,:)","str_len":"100"},"u10":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: u10(:)","str_len":"100"},"u10_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: u10_p(:,:)","str_len":"100"},"ust":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: ust(:)","str_len":"100"},"ust_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: ust_p(:,:)","str_len":"100"},"uthres_in":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: uthres_in(:)","str_len":"100"},"uthres_in_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: uthres_in_p(:,:)","str_len":"100"},"uthres_sg_in":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: uthres_sg_in(:)","str_len":"100"},"uthres_sg_in_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: uthres_sg_in_p(:,:)","str_len":"100"},"v10":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: v10(:)","str_len":"100"},"v10_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: v10_p(:,:)","str_len":"100"},"vegfra":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: vegfra(:)","str_len":"100"},"vegfra_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: vegfra_p(:,:)","str_len":"100"},"wetdep_ls_opt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: wetdep_ls_opt","str_len":"100"},"wetdep_resolved":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: wetdep_resolved(:,:)","str_len":"100"},"wetdep_resolved_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: wetdep_resolved_p(:,:,:)","str_len":"100"},"xland":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: xland(:)","str_len":"100"},"xland_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: xland_p(:,:)","str_len":"100"},"xlat_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: xlat_p(:,:)","str_len":"100"},"xlon_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: xlon_p(:,:)","str_len":"100"},"z0":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: z0(:)","str_len":"100"},"z0_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: z0_p(:,:)","str_len":"100"},"zgrid":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: zgrid(:,:)","str_len":"100"},"zgrid_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: zgrid_p(:,:,:)","str_len":"100"},"znt":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: znt(:)","str_len":"100"},"znt_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: znt_p(:,:)","str_len":"100"}},"source":"driver_slice.F90","first_line":500,"last_line":774},"smoke_to_mpas_fields":{"module":"smoke_driver_slice","args":["add_fire_heat_flux","add_fire_moist_flux","aero_emis_for_enhmix","aod3d","aod3d_p","aod3d_smoke","aod3d_smoke_p","aod550","aod550_p","chem","chem_p","coef_bb_dc","coef_bb_dc_p","config_smoke_scheme","ddvel","ddvel_p","drydep_flux","drydep_flux_p","drydep_opt","e_ant_out","e_ant_out_p","e_bb_out","e_bb_out_p","e_dust_out","e_dust_out_p","ebb_dcycle","fire_ef_out","fire_ef_out_p","frac_grid_burned","frac_grid_burned_p","fre_out","fre_out_p","frp_out","frp_out_p","hfx_bb","hfx_bb_p","hwp","hwp_p","ite","its","jte","jts","kte","kts","max_bb_plume","max_bb_plume_p","min_bb_plume","min_bb_plume_p","num_chem","num_e_ant_out","num_e_bb_out","num_e_dust_out","plumerise_opt","pm10","pm10_p","pm2_5","pm2_5_p","qfx_bb","qfx_bb_p","tend_chem_settle","tend_chem_settle_p","vis","vis_p","wetdep_ls_opt","wetdep_resolved","wetdep_resolved_p"],"symbols":{"add_fire_heat_flux":{"type":"bool","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"logical :: add_fire_heat_flux","str_len":"100"},"add_fire_moist_flux":{"type":"bool","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"logical :: add_fire_moist_flux","str_len":"100"},"aero_emis_for_enhmix":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aero_emis_for_enhmix(:)","str_len":"100"},"aod3d":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod3d(:,:)","str_len":"100"},"aod3d_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod3d_p(:,:,:)","str_len":"100"},"aod3d_smoke":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod3d_smoke(:,:)","str_len":"100"},"aod3d_smoke_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod3d_smoke_p(:,:,:)","str_len":"100"},"aod550":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod550(:)","str_len":"100"},"aod550_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod550_p(:,:)","str_len":"100"},"chem":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: chem(:,:,:)","str_len":"100"},"chem_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: chem_p(:,:,:,:)","str_len":"100"},"coef_bb_dc":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: coef_bb_dc(:)","str_len":"100"},"coef_bb_dc_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: coef_bb_dc_p(:,:)","str_len":"100"},"config_smoke_scheme":{"type":"FStr","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"character(100) :: config_smoke_scheme","str_len":"100"},"ddvel":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: ddvel(:,:)","str_len":"100"},"ddvel_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: ddvel_p(:,:,:)","str_len":"100"},"drydep_flux":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: drydep_flux(:,:)","str_len":"100"},"drydep_flux_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: drydep_flux_p(:,:,:)","str_len":"100"},"drydep_opt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: drydep_opt","str_len":"100"},"e_ant_out":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_ant_out(:,:,:)","str_len":"100"},"e_ant_out_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_ant_out_p(:,:,:,:)","str_len":"100"},"e_bb_out":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_bb_out(:,:,:)","str_len":"100"},"e_bb_out_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_bb_out_p(:,:,:,:)","str_len":"100"},"e_dust_out":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_dust_out(:,:,:)","str_len":"100"},"e_dust_out_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: e_dust_out_p(:,:,:,:)","str_len":"100"},"ebb_dcycle":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ebb_dcycle","str_len":"100"},"fire_ef_out":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fire_ef_out(:)","str_len":"100"},"fire_ef_out_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fire_ef_out_p(:,:)","str_len":"100"},"frac_grid_burned":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frac_grid_burned(:)","str_len":"100"},"frac_grid_burned_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frac_grid_burned_p(:,:)","str_len":"100"},"fre_out":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fre_out(:)","str_len":"100"},"fre_out_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: fre_out_p(:,:)","str_len":"100"},"frp_out":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frp_out(:)","str_len":"100"},"frp_out_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: frp_out_p(:,:)","str_len":"100"},"hfx_bb":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hfx_bb(:)","str_len":"100"},"hfx_bb_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hfx_bb_p(:,:)","str_len":"100"},"hwp":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hwp(:)","str_len":"100"},"hwp_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: hwp_p(:,:)","str_len":"100"},"ite":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ite","str_len":"100"},"its":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: its","str_len":"100"},"jte":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: jte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: jts","str_len":"100"},"kte":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kts","str_len":"100"},"max_bb_plume":{"type":"int","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: max_bb_plume(:)","str_len":"100"},"max_bb_plume_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: max_bb_plume_p(:,:)","str_len":"100"},"min_bb_plume":{"type":"int","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: min_bb_plume(:)","str_len":"100"},"min_bb_plume_p":{"type":"int","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: min_bb_plume_p(:,:)","str_len":"100"},"num_chem":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_chem","str_len":"100"},"num_e_ant_out":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_e_ant_out","str_len":"100"},"num_e_bb_out":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_e_bb_out","str_len":"100"},"num_e_dust_out":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: num_e_dust_out","str_len":"100"},"plumerise_opt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: plumerise_opt","str_len":"100"},"pm10":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: pm10(:,:)","str_len":"100"},"pm10_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: pm10_p(:,:,:)","str_len":"100"},"pm2_5":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: pm2_5(:,:)","str_len":"100"},"pm2_5_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: pm2_5_p(:,:,:)","str_len":"100"},"qfx_bb":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: qfx_bb(:)","str_len":"100"},"qfx_bb_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: qfx_bb_p(:,:)","str_len":"100"},"tend_chem_settle":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: tend_chem_settle(:,:,:)","str_len":"100"},"tend_chem_settle_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: tend_chem_settle_p(:,:,:,:)","str_len":"100"},"vis":{"type":"float","dim":[":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: vis(:)","str_len":"100"},"vis_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: vis_p(:,:)","str_len":"100"},"wetdep_ls_opt":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: wetdep_ls_opt","str_len":"100"},"wetdep_resolved":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: wetdep_resolved(:,:)","str_len":"100"},"wetdep_resolved_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: wetdep_resolved_p(:,:,:)","str_len":"100"}},"source":"driver_slice.F90","first_line":881,"last_line":994},"smoke_driver_diagnostics":{"module":"smoke_driver_slice","args":["aod3d_p","aod550_p","chem_p","chemistry_start","dz_p","index_dust_coarse","index_dust_fine","index_smoke_fine","index_unspc_fine","ite","its","jte","jts","kte","kts","pm10_p","pm2_5_p","rho_p"],"symbols":{"aod3d_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod3d_p(:,:,:)","str_len":"100"},"aod550_p":{"type":"float","dim":[":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: aod550_p(:,:)","str_len":"100"},"chem_p":{"type":"float","dim":[":",":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: chem_p(:,:,:,:)","str_len":"100"},"chemistry_start":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: chemistry_start","str_len":"100"},"dz_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: dz_p(:,:,:)","str_len":"100"},"index_dust_coarse":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: index_dust_coarse","str_len":"100"},"index_dust_fine":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: index_dust_fine","str_len":"100"},"index_smoke_fine":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: index_smoke_fine","str_len":"100"},"index_unspc_fine":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: index_unspc_fine","str_len":"100"},"ite":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: ite","str_len":"100"},"its":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: its","str_len":"100"},"jte":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: jte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: jts","str_len":"100"},"kte":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"integer :: kts","str_len":"100"},"pm10_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: pm10_p(:,:,:)","str_len":"100"},"pm2_5_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: pm2_5_p(:,:,:)","str_len":"100"},"rho_p":{"type":"float","dim":[":",":",":"],"intent":null,"optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind) :: rho_p(:,:,:)","str_len":"100"}},"source":"driver_slice.F90","first_line":1259,"last_line":1291},"mpas_smoke_driver":{"module":"mpas_smoke_wrapper","args":["num_chem","chemistry_start","chem","kanthro","kbio","kfire","kvol","krwc","index_smoke_fine","index_smoke_coarse","index_dust_fine","index_dust_coarse","index_unspc_fine","index_ch4","index_e_bb_in_ch4","index_e_bb_out_ch4","index_e_bb_in_smoke_fine","index_e_bb_in_smoke_coarse","index_e_ant_in_unspc_fine","index_e_ant_in_unspc_coarse","index_e_bb_out_smoke_fine","index_e_bb_out_smoke_coarse","index_e_dust_out_dust_fine","index_e_dust_out_dust_coarse","index_e_ant_out_unspc_fine","index_e_ant_out_unspc_coarse","frp_in","frp_out","fre_in","fre_out","hwp","coef_bb_dc","totprcp_prev24","hwp_avg","frp_avg","fre_avg","fire_end_hr","nblocks","efs_map","eco_id","efs_smold","efs_flam","efs_rsmold","fmc_avg","hfx_bb","qfx_bb","frac_grid_burned","min_bb_plume","max_bb_plume","sandfrac_in","clayfrac_in","uthres_in","uthres_sg_in","albedo_drag_in","feff_in","sep_in","e_ant_in","e_ant_out","num_e_ant_in","num_e_ant_out","e_bb_in","e_bb_out","e_dust_out","num_e_bb_in","num_e_bb_out","num_e_dust_out","drydep_flux","ddvel","wetdep_resolved","tend_chem_settle","do_mpas_smoke","do_mpas_dust","do_mpas_anthro","do_mpas_rwc","calc_bb_emis_online","bb_beta","hwp_method","hwp_alpha","wetdep_ls_opt","wetdep_ls_alpha","plumerise_opt","plume_wind_eff","plume_alpha","bb_emis_scale_factor","ebb_dcycle","drydep_opt","pm_settling","add_fire_heat_flux","add_fire_moist_flux","plumerisefire_frq","bb_qv_scale_factor","dust_alpha","dust_gamma","dust_drylimit_factor","dust_moist_correction","bb_input_prevh","rwc_emis_scale_factor","rwc_denominator","rwc_annual_sum","rwc_annual_sum_smoke_fine","rwc_annual_sum_smoke_coarse","rwc_annual_sum_unspc_fine","rwc_annual_sum_unspc_coarse","vis","qc_vis","qr_vis","qi_vis","qs_vis","qg_vis","blcldw_vis","blcldi_vis","coszen","aod3d_smoke","aod3d","ktau","dt","dxcell","area","xland","u10","v10","ust","xlat","xlong","tskin","pblh","t2m","p8w","dz8w","z_at_w","p_phy","t_phy","u_phy","v_phy","qv","vvel","pi_phy","rho_phy","kpbl","nsoil","smois","tslb","ivgtyp","isltyp","nlcat","swdown","z0","snowh","julian","rmol","raincv","rainncv","dpt2m","znt","mavail","g","vegfra","landusef","cldfrac","ktop_deep","cp","rd","gmt","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"num_chem":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: nsoil, nlcat, num_chem, chemistry_start","str_len":"100"},"chemistry_start":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: nsoil, nlcat, num_chem, chemistry_start","str_len":"100"},"chem":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, kms:kme, jms:jme, 1:num_chem)                :: chem","str_len":"100"},"kanthro":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: kanthro, kbio, kfire, kvol, krwc","str_len":"100"},"kbio":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: kanthro, kbio, kfire, kvol, krwc","str_len":"100"},"kfire":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: kanthro, kbio, kfire, kvol, krwc","str_len":"100"},"kvol":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: kanthro, kbio, kfire, kvol, krwc","str_len":"100"},"krwc":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: kanthro, kbio, kfire, kvol, krwc","str_len":"100"},"index_smoke_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_smoke_fine, index_dust_fine,  index_dust_coarse, index_unspc_fine","str_len":"100"},"index_smoke_coarse":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: index_smoke_coarse, index_e_bb_in_smoke_coarse, index_e_bb_out_smoke_coarse","str_len":"100"},"index_dust_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_smoke_fine, index_dust_fine,  index_dust_coarse, index_unspc_fine","str_len":"100"},"index_dust_coarse":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_smoke_fine, index_dust_fine,  index_dust_coarse, index_unspc_fine","str_len":"100"},"index_unspc_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_smoke_fine, index_dust_fine,  index_dust_coarse, index_unspc_fine","str_len":"100"},"index_ch4":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: index_ch4, index_e_bb_in_ch4, index_e_bb_out_ch4","str_len":"100"},"index_e_bb_in_ch4":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: index_ch4, index_e_bb_in_ch4, index_e_bb_out_ch4","str_len":"100"},"index_e_bb_out_ch4":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: index_ch4, index_e_bb_in_ch4, index_e_bb_out_ch4","str_len":"100"},"index_e_bb_in_smoke_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_e_bb_in_smoke_fine, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse","str_len":"100"},"index_e_bb_in_smoke_coarse":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: index_smoke_coarse, index_e_bb_in_smoke_coarse, index_e_bb_out_smoke_coarse","str_len":"100"},"index_e_ant_in_unspc_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_e_bb_in_smoke_fine, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse","str_len":"100"},"index_e_ant_in_unspc_coarse":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_e_bb_in_smoke_fine, index_e_ant_in_unspc_fine, index_e_ant_in_unspc_coarse","str_len":"100"},"index_e_bb_out_smoke_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_e_bb_out_smoke_fine, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"index_e_bb_out_smoke_coarse":{"type":"int","dim":[],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in), optional :: index_smoke_coarse, index_e_bb_in_smoke_coarse, index_e_bb_out_smoke_coarse","str_len":"100"},"index_e_dust_out_dust_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_e_bb_out_smoke_fine, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"index_e_dust_out_dust_coarse":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_e_bb_out_smoke_fine, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"index_e_ant_out_unspc_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_e_bb_out_smoke_fine, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"index_e_ant_out_unspc_coarse":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer, intent(in) :: index_e_bb_out_smoke_fine, index_e_dust_out_dust_fine, index_e_dust_out_dust_coarse, index_e_ant_out_unspc_fine, index_e_ant_out_unspc_coarse","str_len":"100"},"frp_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime, jms:jme),optional        :: frp_in, fre_in","str_len":"100"},"frp_out":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime, jms:jme),optional :: frp_out, fre_out, efs_map","str_len":"100"},"fre_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime, jms:jme),optional        :: frp_in, fre_in","str_len":"100"},"fre_out":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime, jms:jme),optional :: frp_out, fre_out, efs_map","str_len":"100"},"hwp":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime, jms:jme),optional :: hwp, coef_bb_dc","str_len":"100"},"coef_bb_dc":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime, jms:jme),optional :: hwp, coef_bb_dc","str_len":"100"},"totprcp_prev24":{"type":"float","dim":["ims:ims","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme), optional      :: totprcp_prev24, fire_end_hr,fmc_avg, efs_smold, efs_flam, efs_rsmold","str_len":"100"},"hwp_avg":{"type":"float","dim":["ims:ims","jms:jme","nblocks"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme, nblocks), optional      :: hwp_avg, fre_avg, frp_avg","str_len":"100"},"frp_avg":{"type":"float","dim":["ims:ims","jms:jme","nblocks"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme, nblocks), optional      :: hwp_avg, fre_avg, frp_avg","str_len":"100"},"fre_avg":{"type":"float","dim":["ims:ims","jms:jme","nblocks"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme, nblocks), optional      :: hwp_avg, fre_avg, frp_avg","str_len":"100"},"fire_end_hr":{"type":"float","dim":["ims:ims","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme), optional      :: totprcp_prev24, fire_end_hr,fmc_avg, efs_smold, efs_flam, efs_rsmold","str_len":"100"},"nblocks":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ktau,nblocks","str_len":"100"},"efs_map":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime, jms:jme),optional :: frp_out, fre_out, efs_map","str_len":"100"},"eco_id":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in), dimension(ims:ime,jms:jme),optional            :: eco_id","str_len":"100"},"efs_smold":{"type":"float","dim":["ims:ims","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme), optional      :: totprcp_prev24, fire_end_hr,fmc_avg, efs_smold, efs_flam, efs_rsmold","str_len":"100"},"efs_flam":{"type":"float","dim":["ims:ims","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme), optional      :: totprcp_prev24, fire_end_hr,fmc_avg, efs_smold, efs_flam, efs_rsmold","str_len":"100"},"efs_rsmold":{"type":"float","dim":["ims:ims","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme), optional      :: totprcp_prev24, fire_end_hr,fmc_avg, efs_smold, efs_flam, efs_rsmold","str_len":"100"},"fmc_avg":{"type":"float","dim":["ims:ims","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme), optional      :: totprcp_prev24, fire_end_hr,fmc_avg, efs_smold, efs_flam, efs_rsmold","str_len":"100"},"hfx_bb":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime, jms:jme),optional :: hfx_bb, qfx_bb, frac_grid_burned","str_len":"100"},"qfx_bb":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime, jms:jme),optional :: hfx_bb, qfx_bb, frac_grid_burned","str_len":"100"},"frac_grid_burned":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime, jms:jme),optional :: hfx_bb, qfx_bb, frac_grid_burned","str_len":"100"},"min_bb_plume":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer,intent(inout),dimension(ims:ime,jms:jme),optional      :: min_bb_plume, max_bb_plume","str_len":"100"},"max_bb_plume":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer,intent(inout),dimension(ims:ime,jms:jme),optional      :: min_bb_plume, max_bb_plume","str_len":"100"},"sandfrac_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme),optional  :: sandfrac_in, clayfrac_in, uthres_in, uthres_sg_in, albedo_drag_in, feff_in, sep_in","str_len":"100"},"clayfrac_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme),optional  :: sandfrac_in, clayfrac_in, uthres_in, uthres_sg_in, albedo_drag_in, feff_in, sep_in","str_len":"100"},"uthres_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme),optional  :: sandfrac_in, clayfrac_in, uthres_in, uthres_sg_in, albedo_drag_in, feff_in, sep_in","str_len":"100"},"uthres_sg_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme),optional  :: sandfrac_in, clayfrac_in, uthres_in, uthres_sg_in, albedo_drag_in, feff_in, sep_in","str_len":"100"},"albedo_drag_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme),optional  :: sandfrac_in, clayfrac_in, uthres_in, uthres_sg_in, albedo_drag_in, feff_in, sep_in","str_len":"100"},"feff_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme),optional  :: sandfrac_in, clayfrac_in, uthres_in, uthres_sg_in, albedo_drag_in, feff_in, sep_in","str_len":"100"},"sep_in":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme),optional  :: sandfrac_in, clayfrac_in, uthres_in, uthres_sg_in, albedo_drag_in, feff_in, sep_in","str_len":"100"},"e_ant_in":{"type":"float","dim":["ims:ime","1:kanthro","jms:jme","1:num_e_ant_in"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime,1:kanthro,jms:jme,1:num_e_ant_in),optional :: e_ant_in","str_len":"100"},"e_ant_out":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_e_ant_out"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, kms:kme, jms:jme,1:num_e_ant_out),optional   :: e_ant_out","str_len":"100"},"num_e_ant_in":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: num_e_bb_in, num_e_bb_out, num_e_dust_out, num_e_ant_in, num_e_ant_out","str_len":"100"},"num_e_ant_out":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: num_e_bb_in, num_e_bb_out, num_e_dust_out, num_e_ant_in, num_e_ant_out","str_len":"100"},"e_bb_in":{"type":"float","dim":["ims:ime","1:kfire","jms:jme","1:num_e_bb_in"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime,1:kfire,jms:jme,1:num_e_bb_in),optional  :: e_bb_in","str_len":"100"},"e_bb_out":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_e_bb_out"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, kms:kme, jms:jme,1:num_e_bb_out),optional    :: e_bb_out","str_len":"100"},"e_dust_out":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_e_dust_out"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, kms:kme, jms:jme,1:num_e_dust_out),optional  :: e_dust_out","str_len":"100"},"num_e_bb_in":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: num_e_bb_in, num_e_bb_out, num_e_dust_out, num_e_ant_in, num_e_ant_out","str_len":"100"},"num_e_bb_out":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: num_e_bb_in, num_e_bb_out, num_e_dust_out, num_e_ant_in, num_e_ant_out","str_len":"100"},"num_e_dust_out":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: num_e_bb_in, num_e_bb_out, num_e_dust_out, num_e_ant_in, num_e_ant_out","str_len":"100"},"drydep_flux":{"type":"float","dim":["ims:ime","jms:jme","1:num_chem"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, jms:jme, 1:num_chem),optional :: drydep_flux","str_len":"100"},"ddvel":{"type":"float","dim":["ims:ime","jms:jme","1:num_chem"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, jms:jme, 1:num_chem),optional :: ddvel","str_len":"100"},"wetdep_resolved":{"type":"float","dim":["ims:ime","jms:jme","1:num_chem"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, jms:jme, 1:num_chem),optional :: wetdep_resolved","str_len":"100"},"tend_chem_settle":{"type":"float","dim":["ims:ime","kms:kme","jms:jme","1:num_chem"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, kms:kme, jms:jme, 1:num_chem),optional       :: tend_chem_settle","str_len":"100"},"do_mpas_smoke":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical,intent(in)                :: do_mpas_smoke","str_len":"100"},"do_mpas_dust":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical,intent(in)                :: do_mpas_dust","str_len":"100"},"do_mpas_anthro":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical,intent(in)                :: do_mpas_anthro","str_len":"100"},"do_mpas_rwc":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical,intent(in)                :: do_mpas_rwc","str_len":"100"},"calc_bb_emis_online":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical,intent(in)                :: calc_bb_emis_online","str_len":"100"},"bb_beta":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: bb_beta","str_len":"100"},"hwp_method":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: hwp_method","str_len":"100"},"hwp_alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in)            :: hwp_alpha","str_len":"100"},"wetdep_ls_opt":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: wetdep_ls_opt","str_len":"100"},"wetdep_ls_alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),intent(in)       :: wetdep_ls_alpha","str_len":"100"},"plumerise_opt":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: plumerise_opt","str_len":"100"},"plume_wind_eff":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: plume_wind_eff","str_len":"100"},"plume_alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),intent(in)       :: plume_alpha","str_len":"100"},"bb_emis_scale_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),intent(in)       :: bb_emis_scale_factor, bb_qv_scale_factor","str_len":"100"},"ebb_dcycle":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: ebb_dcycle","str_len":"100"},"drydep_opt":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: drydep_opt","str_len":"100"},"pm_settling":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: pm_settling","str_len":"100"},"add_fire_heat_flux":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical,intent(in)                :: add_fire_heat_flux","str_len":"100"},"add_fire_moist_flux":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical,intent(in)                :: add_fire_moist_flux","str_len":"100"},"plumerisefire_frq":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: plumerisefire_frq","str_len":"100"},"bb_qv_scale_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),intent(in)       :: bb_emis_scale_factor, bb_qv_scale_factor","str_len":"100"},"dust_alpha":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in)            :: dust_alpha, dust_gamma","str_len":"100"},"dust_gamma":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in)            :: dust_alpha, dust_gamma","str_len":"100"},"dust_drylimit_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in)            :: dust_drylimit_factor, dust_moist_correction","str_len":"100"},"dust_moist_correction":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in)            :: dust_drylimit_factor, dust_moist_correction","str_len":"100"},"bb_input_prevh":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)                :: bb_input_prevh","str_len":"100"},"rwc_emis_scale_factor":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(kind=rkind),intent(in)      :: rwc_emis_scale_factor","str_len":"100"},"rwc_denominator":{"type":"float","dim":["ims:ims","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, jms:jme),optional    :: rwc_denominator","str_len":"100"},"rwc_annual_sum":{"type":"float","dim":["ims:ims","1:krwc","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, 1:krwc,jms:jme),optional :: rwc_annual_sum, rwc_annual_sum_smoke_fine, rwc_annual_sum_smoke_coarse, rwc_annual_sum_unspc_fine, rwc_annual_sum_unspc_coarse","str_len":"100"},"rwc_annual_sum_smoke_fine":{"type":"float","dim":["ims:ims","1:krwc","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, 1:krwc,jms:jme),optional :: rwc_annual_sum, rwc_annual_sum_smoke_fine, rwc_annual_sum_smoke_coarse, rwc_annual_sum_unspc_fine, rwc_annual_sum_unspc_coarse","str_len":"100"},"rwc_annual_sum_smoke_coarse":{"type":"float","dim":["ims:ims","1:krwc","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, 1:krwc,jms:jme),optional :: rwc_annual_sum, rwc_annual_sum_smoke_fine, rwc_annual_sum_smoke_coarse, rwc_annual_sum_unspc_fine, rwc_annual_sum_unspc_coarse","str_len":"100"},"rwc_annual_sum_unspc_fine":{"type":"float","dim":["ims:ims","1:krwc","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, 1:krwc,jms:jme),optional :: rwc_annual_sum, rwc_annual_sum_smoke_fine, rwc_annual_sum_smoke_coarse, rwc_annual_sum_unspc_fine, rwc_annual_sum_unspc_coarse","str_len":"100"},"rwc_annual_sum_unspc_coarse":{"type":"float","dim":["ims:ims","1:krwc","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ims, 1:krwc,jms:jme),optional :: rwc_annual_sum, rwc_annual_sum_smoke_fine, rwc_annual_sum_smoke_coarse, rwc_annual_sum_unspc_fine, rwc_annual_sum_unspc_coarse","str_len":"100"},"vis":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, jms:jme),optional                   :: vis","str_len":"100"},"qc_vis":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime,kms:kme,jms:jme),optional :: qc_vis, qr_vis, qi_vis, qs_vis, qg_vis, blcldw_vis, blcldi_vis","str_len":"100"},"qr_vis":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime,kms:kme,jms:jme),optional :: qc_vis, qr_vis, qi_vis, qs_vis, qg_vis, blcldw_vis, blcldi_vis","str_len":"100"},"qi_vis":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime,kms:kme,jms:jme),optional :: qc_vis, qr_vis, qi_vis, qs_vis, qg_vis, blcldw_vis, blcldi_vis","str_len":"100"},"qs_vis":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime,kms:kme,jms:jme),optional :: qc_vis, qr_vis, qi_vis, qs_vis, qg_vis, blcldw_vis, blcldi_vis","str_len":"100"},"qg_vis":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime,kms:kme,jms:jme),optional :: qc_vis, qr_vis, qi_vis, qs_vis, qg_vis, blcldw_vis, blcldi_vis","str_len":"100"},"blcldw_vis":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime,kms:kme,jms:jme),optional :: qc_vis, qr_vis, qi_vis, qs_vis, qg_vis, blcldw_vis, blcldi_vis","str_len":"100"},"blcldi_vis":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),dimension(ims:ime,kms:kme,jms:jme),optional :: qc_vis, qr_vis, qi_vis, qs_vis, qg_vis, blcldw_vis, blcldi_vis","str_len":"100"},"coszen":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: coszen","str_len":"100"},"aod3d_smoke":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, kms:kme, jms:jme)                            :: aod3d_smoke, aod3d","str_len":"100"},"aod3d":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, kms:kme, jms:jme)                            :: aod3d_smoke, aod3d","str_len":"100"},"ktau":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ktau,nblocks","str_len":"100"},"dt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in) :: dt, julian, g, cp, rd, gmt","str_len":"100"},"dxcell":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: xlat, xlong, dxcell, area, xland","str_len":"100"},"area":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: xlat, xlong, dxcell, area, xland","str_len":"100"},"xland":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: xlat, xlong, dxcell, area, xland","str_len":"100"},"u10":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: u10, v10","str_len":"100"},"v10":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: u10, v10","str_len":"100"},"ust":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, jms:jme)          :: rmol, ust","str_len":"100"},"xlat":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: xlat, xlong, dxcell, area, xland","str_len":"100"},"xlong":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: xlat, xlong, dxcell, area, xland","str_len":"100"},"tskin":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: tskin, t2m, dpt2m","str_len":"100"},"pblh":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: pblh","str_len":"100"},"t2m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: tskin, t2m, dpt2m","str_len":"100"},"p8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"dz8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"z_at_w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"p_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"t_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"u_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"v_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"qv":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, kms:kme, jms:jme)           :: qv","str_len":"100"},"vvel":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"pi_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"kpbl":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in), dimension(ims:ime, jms:jme)                 :: kpbl","str_len":"100"},"nsoil":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: nsoil, nlcat, num_chem, chemistry_start","str_len":"100"},"smois":{"type":"float","dim":["ims:ime","1:nsoil","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime,1:nsoil, jms:jme)               :: smois, tslb","str_len":"100"},"tslb":{"type":"float","dim":["ims:ime","1:nsoil","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime,1:nsoil, jms:jme)               :: smois, tslb","str_len":"100"},"ivgtyp":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in), dimension(ims:ime, jms:jme)                 :: isltyp, ivgtyp","str_len":"100"},"isltyp":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in), dimension(ims:ime, jms:jme)                 :: isltyp, ivgtyp","str_len":"100"},"nlcat":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: nsoil, nlcat, num_chem, chemistry_start","str_len":"100"},"swdown":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: swdown, z0, snowh, znt","str_len":"100"},"z0":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: swdown, z0, snowh, znt","str_len":"100"},"snowh":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: swdown, z0, snowh, znt","str_len":"100"},"julian":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in) :: dt, julian, g, cp, rd, gmt","str_len":"100"},"rmol":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout), dimension(ims:ime, jms:jme)          :: rmol, ust","str_len":"100"},"raincv":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: raincv, rainncv, mavail","str_len":"100"},"rainncv":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: raincv, rainncv, mavail","str_len":"100"},"dpt2m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: tskin, t2m, dpt2m","str_len":"100"},"znt":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: swdown, z0, snowh, znt","str_len":"100"},"mavail":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: raincv, rainncv, mavail","str_len":"100"},"g":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in) :: dt, julian, g, cp, rd, gmt","str_len":"100"},"vegfra":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, jms:jme)             :: vegfra","str_len":"100"},"landusef":{"type":"float","dim":["ims:ime","1:nlcat","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime,1:nlcat, jms:jme)               :: landusef","str_len":"100"},"cldfrac":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme)    :: p8w,    dz8w,    z_at_w, cldfrac, p_phy,  t_phy,   u_phy,  v_phy, pi_phy, rho_phy, vvel","str_len":"100"},"ktop_deep":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in), dimension(ims:ime, jms:jme),optional        :: ktop_deep","str_len":"100"},"cp":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in) :: dt, julian, g, cp, rd, gmt","str_len":"100"},"rd":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in) :: dt, julian, g, cp, rd, gmt","str_len":"100"},"gmt":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in) :: dt, julian, g, cp, rd, gmt","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"}},"source":"mpas_smoke_wrapper.F90","first_line":34,"last_line":638},"mpas_smoke_prep":{"module":"mpas_smoke_wrapper","args":["do_mpas_smoke","ktau","nlcat","cp","ebb_dcycle","ebb_min","xland","xlat","xlong","ivgtyp","isltyp","vegfrac","snowh","u10","v10","wind10m","t2m","dpt2m","wetness","hwp","hwp_day_avg","index_e_bb_in_smoke_fine","num_e_bb_in","kfire","e_bb_in","t_phy","u_phy","v_phy","p_phy","pi_phy","z_at_w","dz8w","dz8w_flip","rho_phy","qv","relhum","rh2m","rri","total_flashrate","wind_phy","theta_phy","zmid","kpbl_thetav","peak_hr","coef_bb_dc","fire_hist","lu_nofire","lu_qfire","lu_sfire","fire_type","ids","ide","jds","jde","kds","kde","ims","ime","jms","jme","kms","kme","its","ite","jts","jte","kts","kte"],"symbols":{"do_mpas_smoke":{"type":"bool","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"logical,intent(in)     :: do_mpas_smoke","str_len":"100"},"ktau":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)     :: ktau, nlcat,ebb_dcycle, kfire, num_e_bb_in, index_e_bb_in_smoke_fine","str_len":"100"},"nlcat":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)     :: ktau, nlcat,ebb_dcycle, kfire, num_e_bb_in, index_e_bb_in_smoke_fine","str_len":"100"},"cp":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in) :: cp, ebb_min","str_len":"100"},"ebb_dcycle":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)     :: ktau, nlcat,ebb_dcycle, kfire, num_e_bb_in, index_e_bb_in_smoke_fine","str_len":"100"},"ebb_min":{"type":"float","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in) :: cp, ebb_min","str_len":"100"},"xland":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"xlat":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"xlong":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"ivgtyp":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in), dimension(ims:ime, jms:jme) :: isltyp, ivgtyp","str_len":"100"},"isltyp":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in), dimension(ims:ime, jms:jme) :: isltyp, ivgtyp","str_len":"100"},"vegfrac":{"type":"float","dim":["ims:ime","nlcat","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, nlcat, jms:jme) :: vegfrac","str_len":"100"},"snowh":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"u10":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"v10":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"wind10m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: total_flashrate, wind10m, rh2m","str_len":"100"},"t2m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"dpt2m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"wetness":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, jms:jme) :: xland, xlat, xlong, snowh, u10, v10, t2m, dpt2m, wetness","str_len":"100"},"hwp":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime,jms:jme),optional :: hwp, coef_bb_dc","str_len":"100"},"hwp_day_avg":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: fire_hist, peak_hr, hwp_day_avg","str_len":"100"},"index_e_bb_in_smoke_fine":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)     :: ktau, nlcat,ebb_dcycle, kfire, num_e_bb_in, index_e_bb_in_smoke_fine","str_len":"100"},"num_e_bb_in":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)     :: ktau, nlcat,ebb_dcycle, kfire, num_e_bb_in, index_e_bb_in_smoke_fine","str_len":"100"},"kfire":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in)     :: ktau, nlcat,ebb_dcycle, kfire, num_e_bb_in, index_e_bb_in_smoke_fine","str_len":"100"},"e_bb_in":{"type":"float","dim":["ims:ime","1:kfire","jms:jme","1:num_e_bb_in"],"intent":"in","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime,1:kfire,jms:jme,1:num_e_bb_in),optional :: e_bb_in","str_len":"100"},"t_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"u_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"v_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"p_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"pi_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"z_at_w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"dz8w":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"dz8w_flip":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, kms:kme, jms:jme) :: zmid, wind_phy,theta_phy, relhum, rri, dz8w_flip","str_len":"100"},"rho_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"qv":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(in),   dimension(ims:ime, kms:kme, jms:jme) :: qv, z_at_w, p_phy, t_phy, u_phy, v_phy, pi_phy, rho_phy, dz8w","str_len":"100"},"relhum":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, kms:kme, jms:jme) :: zmid, wind_phy,theta_phy, relhum, rri, dz8w_flip","str_len":"100"},"rh2m":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: total_flashrate, wind10m, rh2m","str_len":"100"},"rri":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, kms:kme, jms:jme) :: zmid, wind_phy,theta_phy, relhum, rri, dz8w_flip","str_len":"100"},"total_flashrate":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: total_flashrate, wind10m, rh2m","str_len":"100"},"wind_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, kms:kme, jms:jme) :: zmid, wind_phy,theta_phy, relhum, rri, dz8w_flip","str_len":"100"},"theta_phy":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, kms:kme, jms:jme) :: zmid, wind_phy,theta_phy, relhum, rri, dz8w_flip","str_len":"100"},"zmid":{"type":"float","dim":["ims:ime","kms:kme","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, kms:kme, jms:jme) :: zmid, wind_phy,theta_phy, relhum, rri, dz8w_flip","str_len":"100"},"kpbl_thetav":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer    ,intent(out),  dimension(ims:ime, jms:jme) :: fire_type, kpbl_thetav","str_len":"100"},"peak_hr":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: fire_hist, peak_hr, hwp_day_avg","str_len":"100"},"coef_bb_dc":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"inout","optional":true,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(inout),dimension(ims:ime,jms:jme),optional :: hwp, coef_bb_dc","str_len":"100"},"fire_hist":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: fire_hist, peak_hr, hwp_day_avg","str_len":"100"},"lu_nofire":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: lu_nofire, lu_qfire, lu_sfire","str_len":"100"},"lu_qfire":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: lu_nofire, lu_qfire, lu_sfire","str_len":"100"},"lu_sfire":{"type":"float","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"real(rkind),intent(out),  dimension(ims:ime, jms:jme) :: lu_nofire, lu_qfire, lu_sfire","str_len":"100"},"fire_type":{"type":"int","dim":["ims:ime","jms:jme"],"intent":"out","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer    ,intent(out),  dimension(ims:ime, jms:jme) :: fire_type, kpbl_thetav","str_len":"100"},"ids":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"ide":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kds":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kde":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"ims":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"ime":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kms":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kme":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"its":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"ite":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"jte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kts":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"},"kte":{"type":"int","dim":[],"intent":"in","optional":false,"parameter":false,"init":null,"save":false,"decl":"integer,intent(in):: ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, its,ite,jts,jte,kts,kte","str_len":"100"}},"source":"mpas_smoke_wrapper.F90","first_line":640,"last_line":835}}')
SOURCE_FILES = json.loads('[{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/ssalt_mod.F90","sha256":"5d3b7eaa0aef9c34b1e03c68e0bbaf28c1fd982b720a048a4022608c79e4b2de","lines":322},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/module_plumerise.F90","sha256":"9a1916475f23b8a02f01fe92ab418c86f391e3e65c9a7bdc995574041052be1c","lines":182},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_diagnostics.F90","sha256":"3a5adf290f7d334837325459a9ca91a00f5a9af8c539729937b8bd4630b826fd","lines":186},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/module_fire_emissions.F90","sha256":"192bebf1ca2a059e0fcc0fc141fd02afab08dda42c89fa32a90df33812c3c739","lines":440},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/module_add_emiss_burn.F90","sha256":"668c09bb4d8247d5cb3f0793cbcb90782c26121238ed0c10868e314b94d9e1c9","lines":110},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90","sha256":"b0110f5e43d6de6a083fb088b9b9dac948f7a4145e1e5453f142817a80743016","lines":2177},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90","sha256":"ad0509551a38cccc82965b3db83ff75ab138e931b73f4d45e061e3b9caeac54f","lines":32},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/dep_dry_simple_mod.F90","sha256":"3ff6801a5f80030af2d2d4102a274b2831857f1266242bda973fbf05b3bbf15b","lines":64},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/rad_data_mod.F90","sha256":"fc433c1a005978a5a13b32ca1753cb722588ca188b6842898497cc2e9f127713","lines":66},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/seas_data_mod.F90","sha256":"770a7ef6c7eb3caedeb195dbfd5388c5bd54a3e1fca7a9de27810bd4b226b0d5","lines":21},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/dust_data_mod.F90","sha256":"5a08bceeaf0719e12ea6563f553bd494b05a0fb9f5e11b45f8b5a8eb3f7b1aa4","lines":97},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90","sha256":"5716e81d9e9bc88e21eb7756c852ae03e8ee5808add556b17fa5d4be38be87fb","lines":107},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90","sha256":"64e288d939e6c92998a52eaf56e0aa3f27be81ceeab95739247dfe897ba74378","lines":658},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/module_wetdep_ls.F90","sha256":"11ec1db3c4a9fb51c907b944927f89b6da57f19c2ae4726fec0f1a5ef6a21169","lines":90},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90","sha256":"0292cbd547c1ad4d033e2650ad749a80278c1487e6ea589bb337f949984b8309","lines":93},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/seas_ngac_mod.F90","sha256":"74a87c55b0a1afe4ba77abf2c28a1aee1b1b53b346e0bc329ca0d7aeff83ff2a","lines":189},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90","sha256":"d7cce067c8af92f443f064827b63ef409bc860763ed95390f0dee9f3556ea478","lines":187},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/module_anthro_emissions.F90","sha256":"96d606996b36ef6640b39637b6187e38e20e64c30b198ba834ebc97c7edb5951","lines":85},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/dep_dry_mod_emerson.F90","sha256":"e92f62b74340893cb4db8efed624e4c4d5be0f2ee0766f7c97306ed5cd960b6d","lines":420},{"path":"src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90","sha256":"21f89e7e66d09909abb18ed7f77e1e4481672a14e2ead9e6c1bea62e8e57a3ed","lines":839}]')
CHEMISTRY_SCHEMAS = json.loads('{\n  "registry": {\n    "atmosphere": {\n      "path": "src/core_atmosphere/physics/registry.chemistry.xml",\n      "sha256": "899ca8417d9c48e9dbdf92bbbe1ea69c20eaa7fe4dfc1aaaa3565f3732099a98",\n      "tree": {\n        "tag": "registry",\n        "attributes": {},\n        "children": [\n          {\n            "tag": "dims",\n            "attributes": {},\n            "children": [\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nChem",\n                  "definition": "namelist:num_chem",\n                  "description": "Number of chemical tracers (nscalar minus nmoist)"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkanthro",\n                  "definition": "namelist:kanthro",\n                  "description": "Number of vertical levels in anthropogenic emission file"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkwildfire",\n                  "definition": "namelist:kwildfire",\n                  "description": "Number of vertical levels in biomass burning emission file"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkbiogenic",\n                  "definition": "namelist:kbiogenic",\n                  "description": "Number of vertical levels in biogenic emission file"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkvolcanic",\n                  "definition": "namelist:kvolcanic",\n                  "description": "Number of vertical levels in volcanic emission file"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "TWENTYFOUR",\n                  "definition": "24",\n                  "description": "A constant value of 24"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkreswoodcomb",\n                  "definition": "namelist:kreswoodcomb",\n                  "description": "Number of vertical levels in residential wood combustion emission file"\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "packages",\n            "attributes": {},\n            "children": [\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_smoke_fine",\n                  "active_when": "config_smoke_scheme .ne. \'off\' .or. config_rwc_scheme .ne. \'off\'",\n                  "description": "parameterization of fine smoke"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_smoke_in",\n                  "active_when": "config_smoke_scheme .ne. \'off\'",\n                  "description": "parameterization of basic wildfire simulation"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_dust_in",\n                  "active_when": "config_dust_scheme .ne. \'off\'",\n                  "description": "parameterization of fine and coarse dust"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_anthro_in",\n                  "active_when": "config_anthro_scheme .ne. \'off\' .or. config_rwc_scheme .ne. \'off\'",\n                  "description": "parameterization of anthropogenic emissions"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_anthro_aero_in",\n                  "active_when": "( ANY( [ CHARACTER(len=20) :: \'simple_aero\', \'simple_gasaero\', \'acomp_gasaero\' ] == TRIM(config_anthro_scheme) ))",\n                  "description": "parameterization of emisssions of anthropogenic aerosols"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_rwc_in",\n                  "active_when": "config_rwc_scheme .ne. \'off\'",\n                  "description": "parameterization of residential wood combustion"\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "nml_record",\n            "attributes": {\n              "name": "chemistry",\n              "in_defaults": "true"\n            },\n            "children": [\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "config_smoke_scheme",\n                  "type": "character",\n                  "default_value": "off",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "configuration for mpas smoke model",\n                  "possible_values": "`on\',`off\'"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "config_dust_scheme",\n                  "type": "character",\n                  "default_value": "off",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "configuration for mpas dust model",\n                  "possible_values": "`on\',`off\'"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "config_anthro_scheme",\n                  "type": "character",\n                  "default_value": "off",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "configuration for mpas anthropogenic emissions model",\n                  "possible_values": "`on\',`off\',`simple_aero\'"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "config_rwc_scheme",\n                  "type": "character",\n                  "default_value": "off",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "Option for online residential wood burning emissions",\n                  "possible_values": "`on\',`off\'"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "rwc_emis_scale_factor",\n                  "type": "real",\n                  "default_value": "1.0",\n                  "units": "-",\n                  "description": "Scale factor applied to residnetial wood fire emissions",\n                  "possible_values": "Positive real values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "num_chem",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Number of chemical tracers",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kanthro",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in anthropogenic emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kwildfire",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in biomass burning emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kbiogenic",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in biogenic emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kvolcanic",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in volcanic emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kreswoodcomb",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in volcanic emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "hwp_method",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Algorithm choice for the hourly wildfire potential",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "hwp_alpha",\n                  "type": "real",\n                  "default_value": "1.0",\n                  "units": "-",\n                  "description": "Scale factor for the hourly wildfire potential",\n                  "possible_values": "Positive real values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "bb_input_prevh",\n                  "type": "integer",\n                  "default_value": "24",\n                  "units": "-",\n                  "description": "Number of previous hours to average smoke input variables",\n                  "possible_values": "factors of 24 (1,2,3,4,6,8,12,24)"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "bb_beta",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Emission factor dataset choice",\n                  "possible_values": "1,2"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "wetdep_ls_opt",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Flag to turn on offline wet deposition",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "wetdep_ls_alpha",\n                  "type": "real",\n                  "default_value": "0.0",\n                  "units": "-",\n                  "description": "Removal fraction for offline wet deposition",\n                  "possible_values": "Real values in the range [0,1]"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "plumerise_opt",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Algorithm choice for plumerise",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "plume_wind_eff",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Algorithm choice for wind effects in plumerise",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "plume_alpha",\n                  "type": "real",\n                  "default_value": "0.0",\n                  "units": "-",\n                  "description": "Entrainment constant for plumerise scheme",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "bb_qv_scale_factor",\n                  "type": "real",\n                  "default_value": "1.0",\n                  "units": "-",\n                  "description": "Scaling factor for water vapor wildfire emissions",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "ebb_dcycle",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Algorithm choice wildfire emission forecast",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "add_fire_heat_flux",\n                  "type": "logical",\n                  "default_value": "false",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "Logical flag to turn on/off fire heat flux to the atmosphere",\n                  "possible_values": ".true. or .false."\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "add_fire_moist_flux",\n                  "type": "logical",\n                  "default_value": "false",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "Logical flag to turn on/off fire heat moisture to the atmosphere",\n                  "possible_values": ".true. or .false."\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "calc_bb_emis_online",\n                  "type": "logical",\n                  "default_value": "false",\n                  "units": "-",\n                  "description": "Flag to calculate emissions online",\n                  "possible_values": "true or false"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "bb_emis_scale_factor",\n                  "type": "real",\n                  "default_value": "1.0",\n                  "units": "-",\n                  "description": "Scale factor applied to biomass burning fire emissions",\n                  "possible_values": "Positive real values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "drydep_opt",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Algorithm choice for dry deposition",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "pm_settling",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Algorithm choice for particulate matter settling",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "dust_alpha",\n                  "type": "real",\n                  "default_value": "1.0",\n                  "units": "-",\n                  "description": "Scaling (linear) factor for dust emissions",\n                  "possible_values": "Positive real values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "dust_gamma",\n                  "type": "real",\n                  "default_value": "1.0",\n                  "units": "-",\n                  "description": "Scaling (exponential) factor for sediment supply",\n                  "possible_values": "Positive real values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "dust_moist_correction",\n                  "type": "real",\n                  "default_value": "1.0",\n                  "units": "-",\n                  "description": "Moisture correction factor for FENGSHA dust",\n                  "possible_values": "Positive real values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "dust_drylimit_factor",\n                  "type": "real",\n                  "default_value": "1.0",\n                  "units": "-",\n                  "description": "soil moisture dry limit factor for FENGSHA dust",\n                  "possible_values": "Positive real values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "plumerisefire_frq",\n                  "type": "integer",\n                  "default_value": "60",\n                  "in_defaults": "false",\n                  "units": "minutes",\n                  "description": "frequency of call to fire plumerise",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "aero_dir_rad_fdb",\n                  "type": "logical",\n                  "default_value": "false",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "Flag that controls the direct aerosol effect",\n                  "possbile_values": ".true. or .false."\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "config_mynn_enh_mix",\n                  "type": "logical",\n                  "default_value": "false",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "Flag that controls artificially enhancing the PBL exhange coefs",\n                  "possbile_values": ".true. or .false."\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "streams",\n            "attributes": {},\n            "children": [\n              {\n                "tag": "stream",\n                "attributes": {\n                  "name": "output",\n                  "type": "output",\n                  "filename_template": "history.$Y-$M-$D_$h.$m.$s.nc",\n                  "output_interval": "6:00:00",\n                  "runtime_format": "separate_file"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "aero_emis_for_enhmix"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "swdnb"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "swdnbc"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "aod3d_smoke"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "aod3d"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "aod550"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "vis"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "ddvel"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "drydep_flux"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "wetdep_resolved"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "hwp"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "frp_out"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "fre_out"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "hfx_bb"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "qfx_bb"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "PM2_5"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "PM10"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "max_bb_plume"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "min_bb_plume"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "coef_bb_dc"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "fire_hist"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "fire_peak_hr"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "fire_type"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var_array",\n                    "attributes": {\n                      "name": "e_bb_out"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var_array",\n                    "attributes": {\n                      "name": "e_dust_out"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "stream",\n                "attributes": {\n                  "name": "smoke_retro_input",\n                  "type": "input",\n                  "filename_template": "smoke.init.retro.$Y-$M-$D_$h.$m.$s.nc",\n                  "filename_interval": "input_interval",\n                  "input_interval": "1:00:00",\n                  "immutable": "true",\n                  "in_defaults": "false"\n                },\n                "children": [\n                  {\n                    "tag": "var_array",\n                    "attributes": {\n                      "name": "e_bb_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "frp_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "fre_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "stream",\n                "attributes": {\n                  "name": "smoke_input",\n                  "type": "input",\n                  "filename_template": "x1.40962.bb.init.nc",\n                  "input_interval": "initial_only",\n                  "immutable": "true",\n                  "in_defaults": "false"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "hwp_prev24"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "totprcp_prev24"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "frp_prev24"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "fre_prev24"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "fmc_prev24"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "stream",\n                "attributes": {\n                  "name": "dust_input",\n                  "type": "input",\n                  "filename_template": "x1.40962.dust.init.nc",\n                  "input_interval": "initial_only",\n                  "immutable": "true",\n                  "in_defaults": "false"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "sandfrac_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "clayfrac_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "uthres_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "uthres_sg_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "sep_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "albedo_drag_m_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "feff_m_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "stream",\n                "attributes": {\n                  "name": "anthro_input",\n                  "type": "input",\n                  "filename_template": "anthro.init.$Y-$M-$D_$h.$m.$s.nc",\n                  "filename_interval": "input_interval",\n                  "input_interval": "1:00:00",\n                  "immutable": "true",\n                  "in_defaults": "false"\n                },\n                "children": [\n                  {\n                    "tag": "var_array",\n                    "attributes": {\n                      "name": "e_ant_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "stream",\n                "attributes": {\n                  "name": "rwc_input",\n                  "type": "input",\n                  "filename_template": "rwc.init.nc",\n                  "input_interval": "intial_only",\n                  "immutable": "true",\n                  "in_defaults": "false"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "RWC_annual_sum"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "RWC_denominator"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "RWC_annual_sum_smoke_fine"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "RWC_annual_sum_smoke_coarse"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "RWC_annual_sum_unspc_fine"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "RWC_annual_sum_unspc_coarse"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "state",\n              "time_levs": "2"\n            },\n            "children": [\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "scalars",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "smoke_fine",\n                      "array_group": "chemistry",\n                      "units": "kg kg^{-1}",\n                      "description": "fine smoke mixing ratio",\n                      "packages": "mpas_smoke_fine"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "dust_fine",\n                      "array_group": "chemistry",\n                      "units": "kg kg^{-1}",\n                      "description": "fine dust mixing ratio",\n                      "packages": "mpas_dust_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "dust_coarse",\n                      "array_group": "chemistry",\n                      "units": "kg kg^{-1}",\n                      "description": "coarse dust mixing ratio",\n                      "packages": "mpas_dust_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "unspc_fine",\n                      "array_group": "chemistry",\n                      "units": "ug kg^{-1}",\n                      "description": "fine anthropogenic primary aerosol mixing ration",\n                      "packages": "mpas_anthro_aero_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "emission_input",\n              "time_levs": "1"\n            },\n            "children": [\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "e_bb_in",\n                  "type": "real",\n                  "dimensions": "nkwildfire nCells Time",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "e_bb_in_smoke_fine",\n                      "name_in_code": "e_bb_in_smoke_fine",\n                      "array_group": "bb_in_aerosols",\n                      "units": "ug m^{-2} s^{-1}",\n                      "description": "input biomass burning fine smoke emissions",\n                      "packages": "mpas_smoke_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "e_ant_in",\n                  "type": "real",\n                  "dimensions": "nkanthro nCells Time",\n                  "packages": "mpas_anthro_in"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "e_ant_in_unspc_fine",\n                      "name_in_code": "e_ant_in_unspc_fine",\n                      "array_group": "ant_in_aerosols",\n                      "units": "ug m^{-2} s^{-1}",\n                      "description": "input anthropogenic fine unspc emissions",\n                      "packages": "mpas_anthro_aero_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "e_ant_in_unspc_coarse",\n                      "name_in_code": "e_ant_in_unspc_coarse",\n                      "array_group": "ant_in_aerosols",\n                      "units": "ug m^{-2} s^{-1}",\n                      "description": "input anthropogenic coarse unspc emissions",\n                      "packages": "mpas_anthro_aero_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "diag_physics",\n              "time_levs": "1"\n            },\n            "children": [\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "e_bb_out",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "e_bb_out_smoke_fine",\n                      "name_in_code": "e_bb_out_smoke_fine",\n                      "array_group": "bb_out_aerosols",\n                      "units": "ug m^{-2} s^{-1}",\n                      "description": "output biomass burning fine smoke emissions",\n                      "packages": "mpas_smoke_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "e_ant_out",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time",\n                  "packages": "mpas_anthro_in"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "e_ant_out_unspc_fine",\n                      "name_in_code": "e_ant_out_unspc_fine",\n                      "array_group": "ant_out_aerosols",\n                      "units": "ug m^{-2} s^{-1}",\n                      "description": "output anthropogenic fine unspc emissions",\n                      "packages": "mpas_anthro_aero_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "e_ant_out_unspc_coarse",\n                      "name_in_code": "e_ant_out_unspc_coarse",\n                      "array_group": "ant_out_aerosols",\n                      "units": "ug m^{-2} s^{-1}",\n                      "description": "output anthropogenic coarse unspc emissions",\n                      "packages": "mpas_anthro_aero_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "e_dust_out",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "e_dust_out_dust_fine",\n                      "name_in_code": "e_dust_out_dust_fine",\n                      "array_group": "dust_out_aerosols",\n                      "units": "ug m^{-2} s^{-1}",\n                      "description": "emissions of fine dust",\n                      "packages": "mpas_dust_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "e_dust_out_dust_coarse",\n                      "name_in_code": "e_dust_out_dust_coarse",\n                      "array_group": "dust_out_aerosols",\n                      "units": "ug m^{-2} s^{-1}",\n                      "description": "emissions of fine dust",\n                      "packages": "mpas_dust_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "vis",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "m",\n                  "description": "Visibility"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "frac_grid_burned",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "fraction of grid cell area burned",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "hfx_bb",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "W m^{-2}",\n                  "description": "heat flux from biomass burning",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "qfx_bb",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "kg m^{-2}",\n                  "description": "water vapor flux from biomass burning",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "flam_frac",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "fraction of flaming bb emissions",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "min_bb_plume",\n                  "type": "integer",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "minimumum biomass burning plumerise injection level",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "max_bb_plume",\n                  "type": "integer",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "maximumum biomass burning plumerise injection level",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "fire_type",\n                  "type": "integer",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "type of fire",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "fire_peak_hr",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "hr",\n                  "description": "peak hour of wildfire",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "fire_hist",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "coefficient to scale the fire activity depending on the fire duration",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "coef_bb_dc",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "coefficient of diurnal cycle of biomass burning emissions",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "frp_in",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "MW",\n                  "description": "Input fire radiative power",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "frp_out",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "MW",\n                  "description": "Output fire radiative power",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "fre_in",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "MJs",\n                  "description": "Input fire radiative energy",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "fre_out",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "MJs",\n                  "description": "Output fire radiative energy",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "aero_emis_for_enhmix",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "ug m^{-2} s^{-1}",\n                  "description": "total output aerosol emissions for use in enhancxed mixing"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "hwp",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "Hourly Wildfire Potential",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "hwp_prev24",\n                  "type": "real",\n                  "dimensions": "nCells TWENTYFOUR",\n                  "units": "-",\n                  "description": "Hourly Wildfire Potential over the last 24 hours",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "frp_prev24",\n                  "type": "real",\n                  "dimensions": "nCells TWENTYFOUR",\n                  "units": "MW",\n                  "description": "Fire Radiative Power over the last 24 hours",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "fre_prev24",\n                  "type": "real",\n                  "dimensions": "nCells TWENTYFOUR",\n                  "units": "MJs",\n                  "description": "Fire Radiative Energy over the last 24 hours",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "fmc_prev24",\n                  "type": "real",\n                  "dimensions": "nCells TWENTYFOUR",\n                  "units": "MJs",\n                  "description": "Fuel Moisture Content over the last 24 hours",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "totprcp_prev24",\n                  "type": "real",\n                  "dimensions": "nCells TWENTYFOUR",\n                  "units": "m",\n                  "description": "Total precipitation over the last 24 hours",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "fire_end_hr",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "hr",\n                  "description": "Hours since fire was last detected",\n                  "packages": "mpas_smoke_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "ddvel",\n                  "type": "real",\n                  "dimensions": "nChem nCells Time",\n                  "units": "cm s^{-1}",\n                  "description": "dry deposition velocity"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "drydep_flux",\n                  "type": "real",\n                  "dimensions": "nChem nCells Time",\n                  "units": "ug m^{-2} s^{-1}",\n                  "description": "dry deposition mass flux"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "wetdep_resolved",\n                  "type": "real",\n                  "dimensions": "nChem nCells Time",\n                  "units": "ug m^{-2} s^{-1}",\n                  "description": "resolved wet deposition flux"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "PM2_5",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time",\n                  "units": "ug m^{-3}",\n                  "description": "particulate matter less than or equal to 2.5 micrometers in diameter"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "PM10",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time",\n                  "units": "ug m^{-3}",\n                  "description": "particulate matter less than or equal to 10 micrometers in diameter"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "feff",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "drag partition (alternate)",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "albedo_drag",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "drag partition",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "uthres_in",\n                  "type": "real",\n                  "dimensions": "nCells",\n                  "units": "-",\n                  "description": "threshold friction velocity",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "uthres_sg_in",\n                  "type": "real",\n                  "dimensions": "nCells",\n                  "units": "-",\n                  "description": "threshold friction velocity (alternate)",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "sandfrac_in",\n                  "type": "real",\n                  "dimensions": "nCells",\n                  "units": "-",\n                  "description": "sand fraction",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "clayfrac_in",\n                  "type": "real",\n                  "dimensions": "nCells",\n                  "units": "-",\n                  "description": "clay fraction",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "albedo_drag_m_in",\n                  "type": "real",\n                  "dimensions": "nMonths nCells",\n                  "units": "-",\n                  "description": "drag partition",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "feff_m_in",\n                  "type": "real",\n                  "dimensions": "nMonths nCells",\n                  "units": "-",\n                  "description": "drag partition (alternate)",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "sep_in",\n                  "type": "real",\n                  "dimensions": "nCells",\n                  "units": "-",\n                  "description": "soil erosion potential",\n                  "packages": "mpas_dust_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "aod3d_smoke",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time",\n                  "units": "-",\n                  "description": "aerosol optical depth from smoke"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "aod3d",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time",\n                  "units": "-",\n                  "description": "aerosol optical depth from all aerosols"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "aod550",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "-",\n                  "description": "aerosol optical depth (2D) from all aerosols"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "RWC_annual_sum",\n                  "type": "real",\n                  "dimensions": "nkreswoodcomb nCells Time",\n                  "units": "ug m^{-2}",\n                  "description": "Annual sum of residential wood burning emissions",\n                  "packages": "mpas_rwc_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "RWC_annual_sum_smoke_fine",\n                  "type": "real",\n                  "dimensions": "nkreswoodcomb nCells Time",\n                  "units": "ug m^{-2}",\n                  "description": "Annual sum of residential wood burning fine smoke emissions",\n                  "packages": "mpas_rwc_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "RWC_annual_sum_smoke_coarse",\n                  "type": "real",\n                  "dimensions": "nkreswoodcomb nCells Time",\n                  "units": "ug m^{-2}",\n                  "description": "Annual sum of residential wood burning coarse smoke emissions",\n                  "packages": "mpas_rwc_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "RWC_annual_sum_unspc_fine",\n                  "type": "real",\n                  "dimensions": "nkreswoodcomb nCells Time",\n                  "units": "ug m^{-2}",\n                  "description": "Annual sum of residential wood burning fine unspc emissions",\n                  "packages": "mpas_rwc_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "RWC_annual_sum_unspc_coarse",\n                  "type": "real",\n                  "dimensions": "nkreswoodcomb nCells Time",\n                  "units": "ug m^{-2}",\n                  "description": "Annual sum of residential wood burning coarse unspc emissions",\n                  "packages": "mpas_rwc_in"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "RWC_denominator",\n                  "type": "real",\n                  "dimensions": "nCells Time",\n                  "units": "F",\n                  "description": "Annual Sum of (42.12 - 0.79*daily_minimum_temperature)",\n                  "packages": "mpas_rwc_in"\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "tend",\n              "time_levs": "1"\n            },\n            "children": [\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "scalars_tend",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "tend_smoke_fine",\n                      "name_in_code": "smoke_fine",\n                      "array_group": "chemistry",\n                      "units": "kg m^{-3} s^{-1}",\n                      "description": "Tendency of fine smoke mass per unit volume divided by d(zeta)/dz",\n                      "packages": "mpas_smoke_fine"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "tend_dust_fine",\n                      "name_in_code": "dust_fine",\n                      "array_group": "chemistry",\n                      "units": "kg m^{-3} s^{-1}",\n                      "description": "Tendency of fine dust mass per unit volume divided by d(zeta)/dz",\n                      "packages": "mpas_dust_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "tend_dust_coarse",\n                      "name_in_code": "dust_coarse",\n                      "array_group": "chemistry",\n                      "units": "kg m^{-3} s^{-1}",\n                      "description": "Tendency of fine dust mass per unit volume divided by d(zeta)/dz",\n                      "packages": "mpas_dust_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "tend_unspc_fine",\n                      "name_in_code": "unspc_fine",\n                      "array_group": "chemistry",\n                      "units": "kg m^{-3} s^{-1}",\n                      "description": "Tendency of fine unspeciated aerosol mass per unit volume divided by d(zeta)/dz",\n                      "packages": "mpas_anthro_aero_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              }\n            ],\n            "text": "#ifndef MPAS_CAM_DYCORE"\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "lbc",\n              "time_levs": "2",\n              "packages": "limited_area"\n            },\n            "children": [\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "lbc_scalars",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "lbc_smoke_fine",\n                      "name_in_code": "smoke_fine",\n                      "array_group": "chemistry",\n                      "packages": "mpas_smoke_fine",\n                      "units": "kg kg^{-1} s^{-1}",\n                      "description": "Lateral boundary tendency of fine smoke mixing ratio"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "lbc_dust_fine",\n                      "name_in_code": "dust_fine",\n                      "array_group": "chemistry",\n                      "packages": "mpas_dust_in",\n                      "units": "kg kg^{-1} s^{-1}",\n                      "description": "Lateral boundary tendency of fine dust mixing ratio"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "lbc_dust_coarse",\n                      "name_in_code": "dust_coarse",\n                      "array_group": "chemistry",\n                      "packages": "mpas_dust_in",\n                      "units": "kg kg^{-1} s^{-1}",\n                      "description": "Lateral boundary tendency of coarse dust mixing ratio"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "lbc_unspc_fine",\n                      "name_in_code": "unspc_fine",\n                      "array_group": "chemistry",\n                      "packages": "mpas_anthro_aero_in",\n                      "units": "kg kg^{-1} s^{-1}",\n                      "description": "Lateral boundary tendency of fine primary aerosol mixing ratio"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "tend_physics",\n              "time_levs": "1"\n            },\n            "children": [\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "tend_chem_settle",\n                  "type": "real",\n                  "dimensions": "nChem nVertLevels nCells Time",\n                  "units": "kg kg^{-1} s^{-1}",\n                  "description": "tendency of chemistry due to gravitational settling"\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          }\n        ],\n        "text": ""\n      }\n    },\n    "init": {\n      "path": "src/core_init_atmosphere/registry.chemistry.xml",\n      "sha256": "a7a16fd5d868da42ec00eaa54c60137aac90125037bd2fc5c62dd870120ed68e",\n      "tree": {\n        "tag": "registry",\n        "attributes": {},\n        "children": [\n          {\n            "tag": "dims",\n            "attributes": {},\n            "children": [\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nChem",\n                  "definition": "namelist:num_chem",\n                  "description": "Number of chemical tracers (nscalar minus nmoist)"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkanthro",\n                  "definition": "namelist:kanthro",\n                  "description": "Number of vertical levels in anthropogenic emission file"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkwildfire",\n                  "definition": "namelist:kwildfire",\n                  "description": "Number of vertical levels in biomass burning emission file"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkbiogenic",\n                  "definition": "namelist:kbiogenic",\n                  "description": "Number of vertical levels in biogenic emission file"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "dim",\n                "attributes": {\n                  "name": "nkvolcanic",\n                  "definition": "namelist:kvolcanic",\n                  "description": "Number of vertical levels in volcanic emission file"\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "nml_record",\n            "attributes": {\n              "name": "chemistry",\n              "in_defaults": "true"\n            },\n            "children": [\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "config_smoke_scheme",\n                  "type": "character",\n                  "default_value": "off",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "configuration for mpas smoke model",\n                  "possible_values": "`on\',`off\'"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "config_dust_scheme",\n                  "type": "character",\n                  "default_value": "off",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "configuration for mpas dust model",\n                  "possible_values": "`on\',`off\'"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "config_anthro_scheme",\n                  "type": "character",\n                  "default_value": "off",\n                  "in_defaults": "false",\n                  "units": "-",\n                  "description": "configuration for mpas anthropogenic emissions model",\n                  "possible_values": "`on\',`off\'"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "num_chem",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Number of chemical tracers",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kanthro",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in anthropogenic emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kwildfire",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in biomaass burning emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kbiogenic",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in biogenic emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "kvolcanic",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Number of vertical levels in volcanic emission input",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "hwp_method",\n                  "type": "integer",\n                  "default_value": "1",\n                  "units": "-",\n                  "description": "Algorithm choice for the hourly wildfire potential",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "nml_option",\n                "attributes": {\n                  "name": "ebb_dcycle",\n                  "type": "integer",\n                  "default_value": "0",\n                  "units": "-",\n                  "description": "Algorithm choice wildfire emission forecast",\n                  "possible_values": "Positive integer values"\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "packages",\n            "attributes": {},\n            "children": [\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_smoke_in",\n                  "description": "parameterization of smoke."\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_dust_in",\n                  "description": "parameterization of dust"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_anthro_in",\n                  "description": "parameterization of anthropogenic emissions for mpas-smoke"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "package",\n                "attributes": {\n                  "name": "mpas_aerosols_in",\n                  "description": "parameterization of mpas aerosols"\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "state",\n              "time_levs": "1"\n            },\n            "children": [\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "scalars",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time",\n                  "packages": "met_stage_out"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "smoke_fine",\n                      "array_group": "chemistry",\n                      "units": "kg kg^{-1}",\n                      "description": "fine smoke mixing ratio",\n                      "packages": "mpas_smoke_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "dust_fine",\n                      "array_group": "chemistry",\n                      "units": "kg kg^{-1}",\n                      "description": "fine dust mixing ratio",\n                      "packages": "mpas_dust_in;mpas_anthro_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "dust_coarse",\n                      "array_group": "chemistry",\n                      "units": "kg kg^{-1}",\n                      "description": "coarse dust mixing ratio",\n                      "packages": "mpas_dust_in;mpas_anthro_in"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "lbc_state",\n              "time_levs": "1"\n            },\n            "children": [\n              {\n                "tag": "var_array",\n                "attributes": {\n                  "name": "lbc_scalars",\n                  "type": "real",\n                  "dimensions": "nVertLevels nCells Time",\n                  "packages": "lbcs"\n                },\n                "children": [\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "lbc_smoke_fine",\n                      "array_group": "chemistry",\n                      "packages": "mpas_smoke_in",\n                      "units": "kg kg^{-1} s^{-1}",\n                      "description": "Lateral boundary tendency of fine smoke mixing ratio"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "lbc_dust_fine",\n                      "array_group": "chemistry",\n                      "packages": "mpas_dust_in",\n                      "units": "kg kg^{-1} s^{-1}",\n                      "description": "Lateral boundary tendency of fine dust mixing ratio"\n                    },\n                    "children": [],\n                    "text": ""\n                  },\n                  {\n                    "tag": "var",\n                    "attributes": {\n                      "name": "lbc_dust_coarse",\n                      "array_group": "chemistry",\n                      "packages": "mpas_dust_in",\n                      "units": "kg kg^{-1} s^{-1}",\n                      "description": "Lateral boundary tendency of coarse dust mixing ratio"\n                    },\n                    "children": [],\n                    "text": ""\n                  }\n                ],\n                "text": ""\n              }\n            ],\n            "text": ""\n          },\n          {\n            "tag": "var_struct",\n            "attributes": {\n              "name": "fg",\n              "time_levs": "1"\n            },\n            "children": [\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "smoke_fine_fg",\n                  "name_in_code": "smoke_fine",\n                  "type": "real",\n                  "dimensions": "nFGLevels nCells Time",\n                  "units": "kg^{-1}",\n                  "packages": "mpas_smoke_in;mpas_anthro_in",\n                  "description": "First guess fine smoke aerosols"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "dust_fine_fg",\n                  "name_in_code": "dust_fine",\n                  "type": "real",\n                  "dimensions": "nFGLevels nCells Time",\n                  "units": "kg^{-1}",\n                  "packages": "mpas_dust_in;mpas_anthro_in",\n                  "description": "First guess fine dust aerosols"\n                },\n                "children": [],\n                "text": ""\n              },\n              {\n                "tag": "var",\n                "attributes": {\n                  "name": "dust_coarse_fg",\n                  "name_in_code": "dust_coarse",\n                  "type": "real",\n                  "dimensions": "nFGLevels nCells Time",\n                  "units": "kg^{-1}",\n                  "packages": "mpas_dust_in;mpas_anthro_in",\n                  "description": "First guess coarse dust aerosols"\n                },\n                "children": [],\n                "text": ""\n              }\n            ],\n            "text": ""\n          }\n        ],\n        "text": ""\n      }\n    }\n  },\n  "workflow_streams": {\n    "streams.atmosphere.anthro": {\n      "sha256": "bd32c7379cfaf439fa949335b92c2c1ef535d61dc14df14373884c1436c2d2f1",\n      "tree": {\n        "tag": "streams",\n        "attributes": {},\n        "children": [\n          {\n            "tag": "immutable_stream",\n            "attributes": {\n              "name": "anthro_input",\n              "type": "input",\n              "filename_template": "anthro.init.$Y-$M-$D_$h.$m.$s.nc",\n              "input_interval": "1:00:00",\n              "io_type": "netcdf,netcdf4",\n              "filename_interval": "input_interval"\n            },\n            "children": [],\n            "text": ""\n          }\n        ],\n        "text": ""\n      }\n    },\n    "streams.atmosphere.rwc": {\n      "sha256": "cc0a0563d7573cb9985af4791733ecdf2313b97005daa1891bf71f4c3f4c04c3",\n      "tree": {\n        "tag": "streams",\n        "attributes": {},\n        "children": [\n          {\n            "tag": "immutable_stream",\n            "attributes": {\n              "name": "rwc_input",\n              "type": "input",\n              "filename_template": "rwc.init.nc",\n              "io_type": "netcdf,netcdf4",\n              "input_interval": "initial_only"\n            },\n            "children": [],\n            "text": ""\n          }\n        ],\n        "text": ""\n      }\n    },\n    "streams.atmosphere.dust": {\n      "sha256": "a958c61293914d7bd0eacc4982efa07d4f1b1f0735703617dbe63cf3c41bdd65",\n      "tree": {\n        "tag": "streams",\n        "attributes": {},\n        "children": [\n          {\n            "tag": "immutable_stream",\n            "attributes": {\n              "name": "dust_input",\n              "type": "input",\n              "filename_template": "dust.init.nc",\n              "io_type": "netcdf,netcdf4",\n              "input_interval": "initial_only"\n            },\n            "children": [],\n            "text": ""\n          }\n        ],\n        "text": ""\n      }\n    },\n    "streams.atmosphere.smoke_forecast": {\n      "sha256": "2aaf4dedec49d85e0b558e4f781a970673cd6499a5fbc20dafe0269798010f60",\n      "tree": {\n        "tag": "streams",\n        "attributes": {},\n        "children": [\n          {\n            "tag": "immutable_stream",\n            "attributes": {\n              "name": "smoke_input",\n              "type": "input",\n              "filename_template": "smoke.init.nc",\n              "io_type": "netcdf,netcdf4",\n              "input_interval": "initial_only"\n            },\n            "children": [],\n            "text": ""\n          }\n        ],\n        "text": ""\n      }\n    },\n    "streams.atmosphere.pollen": {\n      "sha256": "d5c4d33cd34a3c40ca4cbd922c86eef0da1d0fa293981c1319dd6497d1826608",\n      "tree": {\n        "tag": "streams",\n        "attributes": {},\n        "children": [\n          {\n            "tag": "immutable_stream",\n            "attributes": {\n              "name": "bio_input",\n              "type": "input",\n              "filename_template": "bio.init.nc",\n              "io_type": "netcdf,netcdf4",\n              "input_interval": "initial_only"\n            },\n            "children": [],\n            "text": ""\n          }\n        ],\n        "text": ""\n      }\n    },\n    "streams.atmosphere.smoke_retro": {\n      "sha256": "52217312775d71449e01399abd3960e313d8567c975dc5385fd0888b2e07e99f",\n      "tree": {\n        "tag": "streams",\n        "attributes": {},\n        "children": [\n          {\n            "tag": "immutable_stream",\n            "attributes": {\n              "name": "smoke_retro_input",\n              "type": "input",\n              "filename_template": "smoke.init.retro.$Y-$M-$D_$h.$m.$s.nc",\n              "io_type": "netcdf,netcdf4",\n              "input_interval": "1:00:00",\n              "filename_interval": "input_interval"\n            },\n            "children": [],\n            "text": ""\n          }\n        ],\n        "text": ""\n      }\n    }\n  }\n}\n')
DRIVER_BINDINGS = json.loads('{\n  "source_path": "src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F",\n  "source_sha256": "30c381d3098709d78e78a3eb514ab4822686b6015b112c49e68c7efbc3095a6a",\n  "pool_bindings": {\n    "allocate_smoke": [\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_smoke_scheme",\n        "local": "config_smoke_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_dust_scheme",\n        "local": "config_dust_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_anthro_scheme",\n        "local": "config_anthro_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_rwc_scheme",\n        "local": "config_rwc_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "ebb_dcycle",\n        "local": "ebb_dcycle",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "plumerise_opt",\n        "local": "plumerise_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "add_fire_heat_flux",\n        "local": "add_fire_heat_flux",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "add_fire_moist_flux",\n        "local": "add_fire_moist_flux",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "bb_input_prevh",\n        "local": "bb_input_prevh",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "wetdep_ls_opt",\n        "local": "wetdep_ls_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "drydep_opt",\n        "local": "drydep_opt",\n        "time_level": null\n      }\n    ],\n    "deallocate_smoke": [],\n    "smoke_from_mpas": [\n      {\n        "kind": "array",\n        "pool": "state",\n        "field": "scalars",\n        "local": "scalars",\n        "time_level": "time_lev"\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "chemistry_start",\n        "local": "chemistry_start",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "chemistry_end",\n        "local": "chemistry_end",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_smoke_scheme",\n        "local": "config_smoke_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_dust_scheme",\n        "local": "config_dust_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_anthro_scheme",\n        "local": "config_anthro_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_rwc_scheme",\n        "local": "config_rwc_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "wetdep_ls_opt",\n        "local": "wetdep_ls_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "drydep_opt",\n        "local": "drydep_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_convection_scheme",\n        "local": "config_convection_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "plumerise_opt",\n        "local": "plumerise_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "add_fire_heat_flux",\n        "local": "add_fire_heat_flux",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "add_fire_moist_flux",\n        "local": "add_fire_moist_flux",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "ebb_dcycle",\n        "local": "ebb_dcycle",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_len_disp",\n        "local": "len_disp",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "mesh",\n        "field": "meshDensity",\n        "local": "meshDensity",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "mesh",\n        "field": "latCell",\n        "local": "latCell",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "mesh",\n        "field": "lonCell",\n        "local": "lonCell",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "mesh",\n        "field": "areaCell",\n        "local": "areaCell",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "xland",\n        "local": "xland",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "ust",\n        "local": "ust",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "coszr",\n        "local": "coszr",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "u10",\n        "local": "u10",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "v10",\n        "local": "v10",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "skintemp",\n        "local": "skintemp",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "hpbl",\n        "local": "hpbl",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "kpbl",\n        "local": "kpbl",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "znt",\n        "local": "znt",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "qc_bl",\n        "local": "qc_bl",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "qi_bl",\n        "local": "qi_bl",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "ktop_deep",\n        "local": "ktop_deep",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "mesh",\n        "field": "zgrid",\n        "local": "zgrid",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "smois",\n        "local": "smois",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "tslb",\n        "local": "tslb",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "isltyp",\n        "local": "isltyp",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "ivgtyp",\n        "local": "ivgtyp",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "vegfra",\n        "local": "vegfra",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "landusef",\n        "local": "landusef",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "z0",\n        "local": "z0",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "sfc_input",\n        "field": "snowh",\n        "local": "snowh",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "rmol",\n        "local": "rmol",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "raincv",\n        "local": "raincv",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "rainncv",\n        "local": "rainncv",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "mavail",\n        "local": "mavail",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "cldfrac",\n        "local": "cldfrac",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "wetdep_resolved",\n        "local": "wetdep_resolved",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "drydep_flux",\n        "local": "drydep_flux",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "clayfrac_in",\n        "local": "clayfrac_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "sandfrac_in",\n        "local": "sandfrac_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "uthres_in",\n        "local": "uthres_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "uthres_sg_in",\n        "local": "uthres_sg_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "albedo_drag",\n        "local": "albedo_drag",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "feff",\n        "local": "feff",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "sep_in",\n        "local": "sep_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "emission_input",\n        "field": "e_bb_in",\n        "local": "e_bb_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "emission_input",\n        "field": "e_ant_in",\n        "local": "e_ant_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "RWC_denominator",\n        "local": "RWC_denominator",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "RWC_annual_sum",\n        "local": "RWC_annual_sum",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "RWC_annual_sum_smoke_fine",\n        "local": "RWC_annual_sum_smoke_fine",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "RWC_annual_sum_smoke_coarse",\n        "local": "RWC_annual_sum_smoke_coarse",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "RWC_annual_sum_unspc_fine",\n        "local": "RWC_annual_sum_unspc_fine",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "RWC_annual_sum_unspc_coarse",\n        "local": "RWC_annual_sum_unspc_coarse",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "frp_out",\n        "local": "frp_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "fre_out",\n        "local": "fre_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "hwp",\n        "local": "hwp",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "coef_bb_dc",\n        "local": "coef_bb_dc",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "frp_in",\n        "local": "frp_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "fre_in",\n        "local": "fre_in",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "totprcp_prev24",\n        "local": "totprcp_prev24",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "hwp_prev24",\n        "local": "hwp_prev24",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "frp_prev24",\n        "local": "frp_prev24",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "fre_prev24",\n        "local": "fre_prev24",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "fire_end_hr",\n        "local": "fire_end_hr",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "fmc_prev24",\n        "local": "fmc_prev24",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "eco_id",\n        "local": "eco_id",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "efs_smold",\n        "local": "efs_smold",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "efs_flam",\n        "local": "efs_flam",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "efs_rsmold",\n        "local": "efs_rsmold",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "fire_ef_out",\n        "local": "fire_ef_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "min_bb_plume",\n        "local": "min_bb_plume",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "max_bb_plume",\n        "local": "max_bb_plume",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "hfx_bb",\n        "local": "hfx_bb",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "frac_grid_burned",\n        "local": "frac_grid_burned",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "qfx_bb",\n        "local": "qfx_bb",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "e_bb_out",\n        "local": "e_bb_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "e_dust_out",\n        "local": "e_dust_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "e_ant_out",\n        "local": "e_ant_out",\n        "time_level": null\n      }\n    ],\n    "smoke_to_mpas": [\n      {\n        "kind": "array",\n        "pool": "state",\n        "field": "scalars",\n        "local": "scalars",\n        "time_level": "time_lev"\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "PM2_5",\n        "local": "PM2_5",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "PM10",\n        "local": "PM10",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "aod3d",\n        "local": "aod3d",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "aod550",\n        "local": "aod550",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "ddvel",\n        "local": "ddvel",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "wetdep_resolved",\n        "local": "wetdep_resolved",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "drydep_flux",\n        "local": "drydep_flux",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "vis",\n        "local": "vis",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "aero_emis_for_enhmix",\n        "local": "aero_emis_for_enhmix",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "tend_physics",\n        "field": "tend_chem_settle",\n        "local": "tend_chem_settle",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "chemistry_start",\n        "local": "chemistry_start",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "chemistry_end",\n        "local": "chemistry_end",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_smoke_scheme",\n        "local": "config_smoke_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_dust_scheme",\n        "local": "config_dust_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_anthro_scheme",\n        "local": "config_anthro_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "ebb_dcycle",\n        "local": "ebb_dcycle",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "wetdep_ls_opt",\n        "local": "wetdep_ls_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "drydep_opt",\n        "local": "drydep_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "plumerise_opt",\n        "local": "plumerise_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "add_fire_heat_flux",\n        "local": "add_fire_heat_flux",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "add_fire_moist_flux",\n        "local": "add_fire_moist_flux",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "e_bb_out",\n        "local": "e_bb_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "e_dust_out",\n        "local": "e_dust_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "e_ant_out",\n        "local": "e_ant_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "aod3d_smoke",\n        "local": "aod3d_smoke",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "frp_out",\n        "local": "frp_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "fre_out",\n        "local": "fre_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "hwp",\n        "local": "hwp",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "coef_bb_dc",\n        "local": "coef_bb_dc",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "fire_ef_out",\n        "local": "fire_ef_out",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "min_bb_plume",\n        "local": "min_bb_plume",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "max_bb_plume",\n        "local": "max_bb_plume",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "hfx_bb",\n        "local": "hfx_bb",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "frac_grid_burned",\n        "local": "frac_grid_burned",\n        "time_level": null\n      },\n      {\n        "kind": "array",\n        "pool": "diag_physics",\n        "field": "qfx_bb",\n        "local": "qfx_bb",\n        "time_level": null\n      }\n    ],\n    "init_smoke": [],\n    "driver_smoke": [\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_do_restart",\n        "local": "config_do_restart",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_smoke_scheme",\n        "local": "config_smoke_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_dust_scheme",\n        "local": "config_dust_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_anthro_scheme",\n        "local": "config_anthro_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "hwp_method",\n        "local": "hwp_method",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "hwp_alpha",\n        "local": "hwp_alpha",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "plumerise_opt",\n        "local": "plumerise_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "plume_wind_eff",\n        "local": "plume_wind_eff",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "plume_alpha",\n        "local": "plume_alpha",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "bb_emis_scale_factor",\n        "local": "bb_emis_scale_factor",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "ebb_dcycle",\n        "local": "ebb_dcycle",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "bb_input_prevh",\n        "local": "bb_input_prevh",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "add_fire_heat_flux",\n        "local": "add_fire_heat_flux",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "add_fire_moist_flux",\n        "local": "add_fire_moist_flux",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "plumerisefire_frq",\n        "local": "plumerisefire_frq",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "calc_bb_emis_online",\n        "local": "calc_bb_emis_online",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "bb_beta",\n        "local": "bb_beta",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "bb_qv_scale_factor",\n        "local": "bb_qv_scale_factor",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "config_rwc_scheme",\n        "local": "config_rwc_scheme",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "rwc_emis_scale_factor",\n        "local": "rwc_emis_scale_factor",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "wetdep_ls_opt",\n        "local": "wetdep_ls_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "wetdep_ls_alpha",\n        "local": "wetdep_ls_alpha",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "drydep_opt",\n        "local": "drydep_opt",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "pm_settling",\n        "local": "pm_settling",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "dust_alpha",\n        "local": "dust_alpha",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "dust_gamma",\n        "local": "dust_gamma",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "dust_drylimit_factor",\n        "local": "dust_drylimit_factor",\n        "time_level": null\n      },\n      {\n        "kind": "config",\n        "pool": "configs",\n        "field": "dust_moist_correction",\n        "local": "dust_moist_correction",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "emission_input",\n        "field": "index_e_bb_in_smoke_fine",\n        "local": "index_e_bb_in_smoke_fine",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "emission_input",\n        "field": "index_e_ant_in_unspc_fine",\n        "local": "index_e_ant_in_unspc_fine",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "emission_input",\n        "field": "index_e_ant_in_unspc_coarse",\n        "local": "index_e_ant_in_unspc_coarse",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "diag_physics",\n        "field": "index_e_bb_out_smoke_fine",\n        "local": "index_e_bb_out_smoke_fine",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "diag_physics",\n        "field": "index_e_dust_out_dust_fine",\n        "local": "index_e_dust_out_dust_fine",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "diag_physics",\n        "field": "index_e_dust_out_dust_coarse",\n        "local": "index_e_dust_out_dust_coarse",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "diag_physics",\n        "field": "index_e_ant_out_unspc_fine",\n        "local": "index_e_ant_out_unspc_fine",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "diag_physics",\n        "field": "index_e_ant_out_unspc_coarse",\n        "local": "index_e_ant_out_unspc_coarse",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "chemistry_start",\n        "local": "chemistry_start",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "index_smoke_fine",\n        "local": "index_smoke_fine",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "index_dust_fine",\n        "local": "index_dust_fine",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "index_dust_coarse",\n        "local": "index_dust_coarse",\n        "time_level": null\n      },\n      {\n        "kind": "dimension",\n        "pool": "state",\n        "field": "index_unspc_fine",\n        "local": "index_unspc_fine",\n        "time_level": null\n      }\n    ]\n  },\n  "wrapper_bindings": {\n    "chem": "chem_p",\n    "num_chem": "num_chem",\n    "chemistry_start": "chemistry_start",\n    "kanthro": "kanthro",\n    "kfire": "kwildfire",\n    "kbio": "kbiogenic",\n    "kvol": "kvolcanic",\n    "krwc": "kreswoodcomb",\n    "e_bb_in": "e_bb_in_p",\n    "e_bb_out": "e_bb_out_p",\n    "e_ant_in": "e_ant_in_p",\n    "e_ant_out": "e_ant_out_p",\n    "e_dust_out": "e_dust_out_p",\n    "num_e_ant_in": "num_e_ant_in",\n    "num_e_bb_in": "num_e_bb_in",\n    "num_e_ant_out": "num_e_ant_out",\n    "num_e_bb_out": "num_e_bb_out",\n    "num_e_dust_out": "num_e_dust_out",\n    "index_smoke_fine": "index_smoke_fine",\n    "index_dust_fine": "index_dust_fine",\n    "index_dust_coarse": "index_dust_coarse",\n    "index_unspc_fine": "index_unspc_fine",\n    "index_e_bb_in_smoke_fine": "index_e_bb_in_smoke_fine",\n    "index_e_ant_in_unspc_fine": "index_e_ant_in_unspc_fine",\n    "index_e_ant_in_unspc_coarse": "index_e_ant_in_unspc_coarse",\n    "index_e_bb_out_smoke_fine": "index_e_bb_out_smoke_fine",\n    "index_e_dust_out_dust_fine": "index_e_dust_out_dust_fine",\n    "index_e_dust_out_dust_coarse": "index_e_dust_out_dust_coarse",\n    "index_e_ant_out_unspc_fine": "index_e_ant_out_unspc_fine",\n    "index_e_ant_out_unspc_coarse": "index_e_ant_out_unspc_coarse",\n    "hwp": "hwp_p",\n    "hwp_method": "hwp_method",\n    "hwp_alpha": "hwp_alpha",\n    "frp_in": "frp_in_p",\n    "frp_out": "frp_out_p",\n    "fre_in": "fre_in_p",\n    "fre_out": "fre_out_p",\n    "totprcp_prev24": "totprcp_prev24_p",\n    "hwp_avg": "hwp_avg_p",\n    "frp_avg": "frp_avg_p",\n    "fre_avg": "fre_avg_p",\n    "fire_end_hr": "fire_end_hr_p",\n    "eco_id": "eco_id_p",\n    "calc_bb_emis_online": "calc_bb_emis_online",\n    "efs_smold": "efs_smold_p",\n    "efs_flam": "efs_flam_p",\n    "efs_rsmold": "efs_rsmold_p",\n    "fmc_avg": "fmc_avg_p",\n    "hfx_bb": "hfx_bb_p",\n    "qfx_bb": "qfx_bb_p",\n    "frac_grid_burned": "frac_grid_burned_p",\n    "min_bb_plume": "min_bb_plume_p",\n    "max_bb_plume": "max_bb_plume_p",\n    "coef_bb_dc": "coef_bb_dc_p",\n    "nblocks": "nblocks",\n    "sandfrac_in": "sandfrac_in_p",\n    "clayfrac_in": "clayfrac_in_p",\n    "uthres_in": "uthres_in_p",\n    "uthres_sg_in": "uthres_sg_in_p",\n    "albedo_drag_in": "albedo_drag_p",\n    "feff_in": "feff_p",\n    "sep_in": "sep_in_p",\n    "wetdep_ls_opt": "wetdep_ls_opt",\n    "drydep_flux": "drydep_flux_p",\n    "tend_chem_settle": "tend_chem_settle_p",\n    "ddvel": "ddvel_p",\n    "wetdep_resolved": "wetdep_resolved_p",\n    "do_mpas_smoke": "do_mpas_smoke",\n    "do_mpas_dust": "do_mpas_dust",\n    "do_mpas_anthro": "do_mpas_anthro",\n    "do_mpas_rwc": "do_mpas_rwc",\n    "plume_wind_eff": "plume_wind_eff",\n    "plume_alpha": "plume_alpha",\n    "bb_emis_scale_factor": "bb_emis_scale_factor",\n    "bb_qv_scale_factor": "bb_qv_scale_factor",\n    "ebb_dcycle": "ebb_dcycle",\n    "add_fire_heat_flux": "add_fire_heat_flux",\n    "add_fire_moist_flux": "add_fire_moist_flux",\n    "plumerisefire_frq": "plumerisefire_frq",\n    "bb_beta": "bb_beta",\n    "bb_input_prevh": "bb_input_prevh",\n    "wetdep_ls_alpha": "wetdep_ls_alpha",\n    "plumerise_opt": "plumerise_opt",\n    "drydep_opt": "drydep_opt",\n    "pm_settling": "pm_settling",\n    "dust_alpha": "dust_alpha",\n    "dust_gamma": "dust_gamma",\n    "dust_drylimit_factor": "dust_drylimit_factor",\n    "dust_moist_correction": "dust_moist_correction",\n    "rwc_emis_scale_factor": "rwc_emis_scale_factor",\n    "ktau": "itimestep",\n    "dt": "dt_dyn",\n    "dxcell": "dx_p",\n    "area": "area_p",\n    "xland": "xland_p",\n    "u10": "u10_p",\n    "v10": "v10_p",\n    "ust": "ust_p",\n    "xlat": "xlat_p",\n    "xlong": "xlon_p",\n    "tskin": "tsk_p",\n    "pblh": "hpbl_p",\n    "t2m": "t2m_p",\n    "p8w": "pres2_hyd_p",\n    "dz8w": "dz_p",\n    "z_at_w": "zgrid_p",\n    "p_phy": "pres_hyd_p",\n    "t_phy": "t_p",\n    "u_phy": "u_p",\n    "v_phy": "v_p",\n    "qv": "qv_p",\n    "vvel": "w_p",\n    "qc_vis": "qc_p",\n    "qr_vis": "qr_p",\n    "qi_vis": "qi_p",\n    "qs_vis": "qs_p",\n    "qg_vis": "qg_p",\n    "blcldw_vis": "qcbl_p",\n    "blcldi_vis": "qibl_p",\n    "coszen": "coszr_p",\n    "aod3d_smoke": "aod3d_smoke_p",\n    "aod3d": "aod3d_p",\n    "vis": "vis_p",\n    "pi_phy": "pi_p",\n    "rho_phy": "rho_p",\n    "kpbl": "kpbl_p",\n    "nsoil": "num_soils",\n    "smois": "smois_p",\n    "tslb": "tslb_p",\n    "ivgtyp": "ivgtyp_p",\n    "isltyp": "isltyp_p",\n    "nlcat": "num_landcat",\n    "swdown": "swdown_p",\n    "z0": "z0_p",\n    "snowh": "snowh_p",\n    "julian": "curr_julday",\n    "rmol": "rmol_p",\n    "raincv": "raincv_p",\n    "rainncv": "rainncv_p",\n    "dpt2m": "dpt2m_p",\n    "znt": "znt_p",\n    "mavail": "mavail_p",\n    "g": "gravity",\n    "vegfra": "vegfra_p",\n    "landusef": "landusef_p",\n    "cldfrac": "cldfrac_p",\n    "ktop_deep": "ktop_deep_p",\n    "cp": "cp",\n    "rd": "r_d",\n    "gmt": "gmt",\n    "ids": "ids",\n    "ide": "ide",\n    "jds": "jds",\n    "jde": "jde",\n    "kds": "kds",\n    "kde": "kde",\n    "ims": "ims",\n    "ime": "ime",\n    "jms": "jms",\n    "jme": "jme",\n    "kms": "kds",\n    "kme": "kme",\n    "its": "its",\n    "ite": "ite",\n    "jts": "jts",\n    "jte": "jte",\n    "kts": "kts",\n    "kte": "kte"\n  },\n  "allocations": {\n    "dx_p": "ims:ime,jms:jme",\n    "area_p": "ims:ime,jms:jme",\n    "xland_p": "ims:ime,jms:jme",\n    "u10_p": "ims:ime,jms:jme",\n    "v10_p": "ims:ime,jms:jme",\n    "ust_p": "ims:ime,jms:jme",\n    "xlat_p": "ims:ime,jms:jme",\n    "xlon_p": "ims:ime,jms:jme",\n    "vegfra_p": "ims:ime,jms:jme",\n    "ivgtyp_p": "ims:ime,jms:jme",\n    "landusef_p": "ims:ime,1:num_landcat,jms:jme",\n    "coszr_p": "ims:ime,jms:jme",\n    "tsk_p": "ims:ime,jms:jme",\n    "hpbl_p": "ims:ime,jms:jme",\n    "t2m_p": "ims:ime,jms:jme",\n    "isltyp_p": "ims:ime,jms:jme",\n    "swdown_p": "ims:ime,jms:jme",\n    "z0_p": "ims:ime,jms:jme",\n    "snowh_p": "ims:ime,jms:jme",\n    "rmol_p": "ims:ime,jms:jme",\n    "raincv_p": "ims:ime,jms:jme",\n    "rainncv_p": "ims:ime,jms:jme",\n    "kpbl_p": "ims:ime,jms:jme",\n    "znt_p": "ims:ime,jms:jme",\n    "dpt2m_p": "ims:ime,jms:jme",\n    "mavail_p": "ims:ime,jms:jme",\n    "qcbl_p": "ims:ime,kms:kme,jms:jme",\n    "qibl_p": "ims:ime,kms:kme,jms:jme",\n    "cldfrac_p": "ims:ime,kms:kme,jms:jme",\n    "ktop_deep_p": "ims:ime,jms:jme",\n    "smois_p": "ims:ime,1:num_soils,jms:jme",\n    "tslb_p": "ims:ime,1:num_soils,jms:jme",\n    "zgrid_p": "ims:ime,kms:kme,jms:jme",\n    "vis_p": "ims:ime,jms:jme",\n    "aod550_p": "ims:ime,jms:jme",\n    "aod3d_p": "ims:ime,kms:kme,jms:jme",\n    "frp_out_p": "ims:ime,jms:jme",\n    "fre_out_p": "ims:ime,jms:jme",\n    "hwp_p": "ims:ime,jms:jme",\n    "coef_bb_dc_p": "ims:ime,jms:jme",\n    "aod3d_smoke_p": "ims:ime,kms:kme,jms:jme",\n    "frp_in_p": "ims:ime,jms:jme",\n    "fre_in_p": "ims:ime,jms:jme",\n    "totprcp_prev24_p": "ims:ime,jms:jme",\n    "hwp_avg_p": "ims:ime,jms:jme,nblocks",\n    "frp_avg_p": "ims:ime,jms:jme,nblocks",\n    "fre_avg_p": "ims:ime,jms:jme,nblocks",\n    "fire_end_hr_p": "ims:ime,jms:jme",\n    "fmc_avg_p": "ims:ime,jms:jme",\n    "eco_id_p": "ims:ime,jms:jme",\n    "efs_smold_p": "ims:ime,jms:jme",\n    "efs_flam_p": "ims:ime,jms:jme",\n    "efs_rsmold_p": "ims:ime,jms:jme",\n    "fire_ef_out_p": "ims:ime,jms:jme",\n    "hfx_bb_p": "ims:ime,jms:jme",\n    "frac_grid_burned_p": "ims:ime,jms:jme",\n    "qfx_bb_p": "ims:ime,jms:jme",\n    "min_bb_plume_p": "ims:ime,jms:jme",\n    "max_bb_plume_p": "ims:ime,jms:jme",\n    "clayfrac_in_p": "ims:ime,jms:jme",\n    "sandfrac_in_p": "ims:ime,jms:jme",\n    "uthres_in_p": "ims:ime,jms:jme",\n    "uthres_sg_in_p": "ims:ime,jms:jme",\n    "albedo_drag_p": "ims:ime,jms:jme",\n    "sep_in_p": "ims:ime,jms:jme",\n    "feff_p": "ims:ime,jms:jme",\n    "wetdep_resolved_p": "ims:ime,jms:jme,1:num_chem",\n    "tend_chem_settle_p": "ims:ime,kms:kme,jms:jme,1:num_chem",\n    "ddvel_p": "ims:ime,jms:jme,1:num_chem",\n    "drydep_flux_p": "ims:ime,jms:jme,1:num_chem",\n    "chem_p": "ims:ime,kms:kme,jms:jme,1:num_chem",\n    "pm2_5_p": "ims:ime,kms:kme,jms:jme",\n    "pm10_p": "ims:ime,kms:kme,jms:jme",\n    "e_bb_in_p": "ims:ime,1:kwildfire,jms:jme,1:num_e_bb_in",\n    "e_ant_in_p": "ims:ime,1:kanthro,jms:jme,1:num_e_ant_in",\n    "e_bb_out_p": "ims:ime,kms:kme,jms:jme,1:num_e_bb_out",\n    "e_dust_out_p": "ims:ime,kms:kme,jms:jme,1:num_e_dust_out",\n    "e_ant_out_p": "ims:ime,kms:kme,jms:jme,1:num_e_ant_out",\n    "rwc_denominator_p": "ims:ime,jms:jme",\n    "rwc_annual_sum_p": "ims:ime,1:kreswoodcomb,jms:jme",\n    "rwc_annual_sum_smoke_fine_p": "ims:ime,1:kreswoodcomb,jms:jme",\n    "rwc_annual_sum_smoke_coarse_p": "ims:ime,1:kreswoodcomb,jms:jme",\n    "rwc_annual_sum_unspc_fine_p": "ims:ime,1:kreswoodcomb,jms:jme",\n    "rwc_annual_sum_unspc_coarse_p": "ims:ime,1:kreswoodcomb,jms:jme"\n  }\n}\n')
CUDA_SOURCE = r'''
#ifndef SDDEV
#define SDDEV __device__
#endif
// RRFS src/core_atmosphere/physics/physics_noaa/SMOKE/*.F90: array and RKIND operation lowering.
const int SD_END = 2147483647;
struct Slice {int lo,hi;SDDEV Slice():lo(SD_END),hi(SD_END) {} SDDEV Slice(int a,int b):lo(a),hi(b) {}};
template<class T> struct FArray {
 T *p=nullptr; int rank=0;int lo[4]={1,1,1,1},hi[4]={0,0,0,0},stride[4]={1,1,1,1};
 SDDEV int size() const {int n=1;for(int d=0;d<rank;d++)n*=hi[d]-lo[d]+1;return rank?n:0;}
 SDDEV T &linear(int n) const {int off=0;for(int d=0;d<rank;d++){int z=hi[d]-lo[d]+1;off+=(n%z)*stride[d];n/=z;}return p[off];}
 SDDEV T &operator()(int a) const {return p[(a-lo[0])*stride[0]];}
 SDDEV T &operator()(int a,int b) const {return p[(a-lo[0])*stride[0]+(b-lo[1])*stride[1]];}
 SDDEV T &operator()(int a,int b,int c) const {return p[(a-lo[0])*stride[0]+(b-lo[1])*stride[1]+(c-lo[2])*stride[2]];}
 SDDEV T &operator()(int a,int b,int c,int e) const {return p[(a-lo[0])*stride[0]+(b-lo[1])*stride[1]+(c-lo[2])*stride[2]+(e-lo[3])*stride[3]];}
 SDDEV void add_section(FArray &r,int &d,int val) const {r.p+=(val-lo[d])*stride[d];d++;}
 SDDEV void add_section(FArray &r,int &d,Slice val) const {int l=val.lo==SD_END?lo[d]:val.lo,h=val.hi==SD_END?hi[d]:val.hi;r.p+=(l-lo[d])*stride[d];r.lo[r.rank]=1;r.hi[r.rank]=h-l+1;r.stride[r.rank++]=stride[d++];}
 template<class... A> SDDEV FArray section(A... a) const {FArray r;r.p=p;int d=0;(add_section(r,d,a),...);return r;}
 template<class... A> SDDEV void bind(A... a) {int v[]={int(a)...};int r=sizeof...(a)/2;int base=stride[0];bool reshape=rank!=r;for(int d=0;d<r && !reshape;d++)reshape=(hi[d]-lo[d]+1 != v[2*d+1]-v[2*d]+1);if(reshape){int st=base;for(int d=0;d<r;d++){stride[d]=st;st*=v[2*d+1]-v[2*d]+1;}}rank=r;for(int d=0;d<r;d++){lo[d]=v[2*d];hi[d]=v[2*d+1];}}
};
template<class T,class... I> SDDEV FArray<T> sd_sequence(FArray<T> a,I... i){FArray<T> r;r.p=&a(i...);r.bind(1,1);return r;}
// Port arena defect: writing beyond the allocated device workspace corrupts
// subsequent state. Launcher reserves from the complete tile descriptors;
// this assertion diagnoses a bad reservation, never a native undefined read.
SDDEV void sd_capacity_failure() {
#ifdef __CUDA_ARCH__
 asm("trap;");
#else
 __builtin_trap();
#endif
}
struct Arena {float *p;int used=0;int capacity;template<class T,class... A> SDDEV FArray<T> alloc(A... a) {FArray<T> r;r.p=(T*)(p+used);r.bind(a...);int words=r.size();if(words<0 || used>capacity-words)sd_capacity_failure();used+=words;return r;}};
struct ArenaScope {Arena &arena;int mark;SDDEV ArenaScope(Arena &a):arena(a),mark(a.used){} SDDEV ~ArenaScope(){arena.used=mark;}};
struct FStr {
 char *p;int len;const char *lit;
 SDDEV FStr():p(nullptr),len(0),lit(nullptr){}
 SDDEV FStr(const char *s):p(nullptr),len(0),lit(s){while(s[len])len++;}
 SDDEV FStr(char *s,int n):p(s),len(n),lit(nullptr){}
 SDDEV char at(int i) const {return i<len?(p?p[i]:lit[i]):' ';}
 SDDEV FStr &operator=(const FStr &s){if(p){for(int i=0;i<len;i++)p[i]=s.at(i);}else{lit=s.lit;len=s.len;p=s.p;}return *this;}
};
SDDEV bool operator==(FStr a,FStr b){int n=a.len>b.len?a.len:b.len;for(int i=0;i<n;i++)if(a.at(i)!=b.at(i))return false;return true;}
SDDEV bool operator!=(FStr a,FStr b){return !(a==b);}
// Native Fortran unary REAL negation is a sign-bit XOR, including sNaNs.
template<class T> SDDEV T sd_neg(T a){return -a;}
SDDEV float sd_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
template<class T> SDDEV T sd_add(T a,T b){return a+b;}
template<class T> SDDEV T sd_sub(T a,T b){return a-b;}
template<class T> SDDEV T sd_mul(T a,T b){return a*b;}
template<class T> SDDEV T sd_div(T a,T b){return a/b;}
// Strict GNU scalar SSE operand semantics: first operand NaN wins, signaling
// payloads are quieted, and invalid operations yield the negative indefinite.
// Finite results still come from the explicit round-to-nearest CUDA operations.
// This is arithmetic at the operation, not a comparison-time NaN substitution.
SDDEV bool sd_float_nan_word(unsigned int u){return (u&0x7fffffffu)>0x7f800000u;}
SDDEV float sd_sse_binary(float a,float b,float result){
#ifdef HEX_NUMERICS_FAST
 // Fast mode retains IEEE results without GNU SSE NaN payload repair.
 return result;
#else
 unsigned int x=__float_as_uint(a),y=__float_as_uint(b);
 if(sd_float_nan_word(x))return __uint_as_float(x|0x00400000u);
 if(sd_float_nan_word(y))return __uint_as_float(y|0x00400000u);
 return sd_float_nan_word(__float_as_uint(result))?__uint_as_float(0xffc00000u):result;
#endif
}
SDDEV float sd_add(float a,float b){return sd_sse_binary(a,b,__fadd_rn(a,b));} SDDEV double sd_add(double a,double b){return __dadd_rn(a,b);}
SDDEV float sd_sub(float a,float b){return sd_sse_binary(a,b,__fsub_rn(a,b));} SDDEV double sd_sub(double a,double b){return __dsub_rn(a,b);}
SDDEV float sd_mul(float a,float b){return sd_sse_binary(a,b,__fmul_rn(a,b));} SDDEV double sd_mul(double a,double b){return __dmul_rn(a,b);}
SDDEV float sd_div(float a,float b){
#ifdef HEX_NUMERICS_FAST
 // Avoid the out-of-line precise division fallback for IEEE exceptional
 // operands. Finite nonzero division retains the same CUDA intrinsic.
 unsigned int x=__float_as_uint(a),y=__float_as_uint(b);
 unsigned int ax=x&0x7fffffffu,ay=y&0x7fffffffu,sign=(x^y)&0x80000000u;
 // Preserve the compiler's FTZ policy for subnormal operand combinations.
 if((ax && ax<0x00800000u)||(ay && ay<0x00800000u))return __fdiv_rn(a,b);
 if(ax>0x7f800000u || ay>0x7f800000u || (ax==0 && ay==0) ||
    (ax==0x7f800000u && ay==0x7f800000u))return __uint_as_float(0x7fffffffu);
 if(ay==0 || ax==0x7f800000u)return __uint_as_float(sign|0x7f800000u);
 if(ax==0 || ay==0x7f800000u)return __uint_as_float(sign);
#endif
 return sd_sse_binary(a,b,__fdiv_rn(a,b));
} SDDEV double sd_div(double a,double b){return __ddiv_rn(a,b);}
#define SD_MIXED(name) template<class A,class B> SDDEV auto name(A a,B b)->decltype(a+b){using T=decltype(a+b);return name(T(a),T(b));}
SD_MIXED(sd_add) SD_MIXED(sd_sub) SD_MIXED(sd_mul) SD_MIXED(sd_div)
template<class T> SDDEV T sd_min(T a,T b){return a<b?a:b;} template<class T> SDDEV T sd_max(T a,T b){return a>b?a:b;}
SD_MIXED(sd_min) SD_MIXED(sd_max)
SDDEV float sd_sqrt(float a){
#ifdef HEX_NUMERICS_FAST
 return __fsqrt_rn(a);
#else
 unsigned int x=__float_as_uint(a);
 if(sd_float_nan_word(x))return __uint_as_float(x|0x00400000u);
 float result=__fsqrt_rn(a);
 return sd_float_nan_word(__float_as_uint(result))?__uint_as_float(0xffc00000u):result;
#endif
}SDDEV double sd_sqrt(double a){return __dsqrt_rn(a);}
SDDEV float sd_exp(float a){return gfk_exp_fma(a);}SDDEV float sd_log(float a){return gfk_log(a);}SDDEV float sd_pow(float a,float b){return gfk_pow(a,b);}
SDDEV double sd_exp(double a){return glibc_exp(a);}SDDEV double sd_log(double a){return glibc_log(a);}SDDEV double sd_pow(double a,double b){return glibc_pow(a,b);}
SD_MIXED(sd_pow)
SDDEV float sd_log10(float a){return thompson_aa_log10f(a);}SDDEV double sd_log10(double a){return thompson_aa_log10(a);}
SDDEV float sd_erf(float a){return smoke_shared_erf(a);}
SDDEV float sd_cos(float a){return glibc_cosf(a);}SDDEV float sd_sin(float a){return glibc_sinf(a);}
SDDEV float sd_abs(float a){return __uint_as_float(__float_as_uint(a)&0x7fffffffu);} SDDEV double sd_abs(double a){return fabs(a);}SDDEV int sd_abs(int a){return a<0?-a:a;}
SDDEV float sd_aint(float a){unsigned int u=__float_as_uint(a);return sd_float_nan_word(u)?__uint_as_float(u):truncf(a);}SDDEV int sd_floor(float a){return int(floorf(a));}SDDEV int sd_nint(float a){return a>=0?int(a+0.5f):int(a-0.5f);}
SDDEV double sd_dble(float a){return double(a);}SDDEV float sd_mod(float a,float b){return fmodf(a,b);}SDDEV int sd_mod(int a,int b){return a%b;}
SDDEV float sd_sign(float a,float b){return copysignf(a,b);}SDDEV bool sd_isnan(float a){return isnan(a);}
template<class T> SDDEV T sd_ipow(T a,int b){if(b==0)return T(1);if(b==1)return a;T x=sd_ipow(a,b/2);T y=sd_mul(x,x);return b%2?sd_mul(y,a):y;}
template<class T> struct ScalarExpr {T v; SDDEV T linear(int) const{return v;}};
template<class T> SDDEV ScalarExpr<T> sd_expr(T v){return {v};}template<class T> SDDEV FArray<T> sd_expr(FArray<T> v){return v;}
struct OpAdd {template<class A,class B> SDDEV static auto op(A a,B b){return sd_add(a,b);}};
struct OpSub {template<class A,class B> SDDEV static auto op(A a,B b){return sd_sub(a,b);}};
struct OpMul {template<class A,class B> SDDEV static auto op(A a,B b){return sd_mul(a,b);}};
struct OpDiv {template<class A,class B> SDDEV static auto op(A a,B b){return sd_div(a,b);}};
template<class A,class B,class Op> struct BinaryExpr {A a;B b;SDDEV auto linear(int i) const {return Op::op(a.linear(i),b.linear(i));}};
template<class A,class B,class O> SDDEV BinaryExpr<A,B,O> sd_expr(BinaryExpr<A,B,O> e){return e;}
#define SD_ARRAY_OP(name,op) \
 template<class T,class B> SDDEV auto name(FArray<T> a,B b){return BinaryExpr<FArray<T>,decltype(sd_expr(b)),op>{a,sd_expr(b)};} \
 template<class A,class T> SDDEV auto name(A a,FArray<T> b){return BinaryExpr<decltype(sd_expr(a)),FArray<T>,op>{sd_expr(a),b};} \
 template<class T,class U> SDDEV auto name(FArray<T> a,FArray<U> b){return BinaryExpr<FArray<T>,FArray<U>,op>{a,b};} \
 template<class A,class B,class O,class C> SDDEV auto name(BinaryExpr<A,B,O> a,C b){return BinaryExpr<BinaryExpr<A,B,O>,decltype(sd_expr(b)),op>{a,sd_expr(b)};}
SD_ARRAY_OP(sd_add,OpAdd) SD_ARRAY_OP(sd_sub,OpSub) SD_ARRAY_OP(sd_mul,OpMul) SD_ARRAY_OP(sd_div,OpDiv)
template<class T,class E> SDDEV void sd_assign(FArray<T> a,E rhs,Arena &arena){ArenaScope scope(arena);auto b=sd_expr(rhs);auto t=arena.alloc<T>(1,a.size());for(int i=0;i<a.size();i++)t.linear(i)=b.linear(i);for(int i=0;i<a.size();i++)a.linear(i)=t.linear(i);}
template<class T> SDDEV T sd_sum(FArray<T> a){T v=0;for(int i=0;i<a.size();i++)v=sd_add(v,a.linear(i));return v;}
template<class T> SDDEV T sd_minval(FArray<T> a){T v=a.linear(0);for(int i=1;i<a.size();i++)v=sd_min(v,a.linear(i));return v;}
template<class T> SDDEV T sd_maxval(FArray<T> a){T v=a.linear(0);for(int i=1;i<a.size();i++)v=sd_max(v,a.linear(i));return v;}

struct Plume {
bool initialized;
FArray<float> w;
FArray<float> t;
FArray<float> qv;
FArray<float> qc;
FArray<float> qh;
FArray<float> qi;
FArray<float> sc;
FArray<float> vth;
FArray<float> vti;
FArray<float> rho;
FArray<float> txs;
FArray<float> est;
FArray<float> qsat;
FArray<float> wc;
FArray<float> wt;
FArray<float> tt;
FArray<float> qvt;
FArray<float> qct;
FArray<float> qht;
FArray<float> qit;
FArray<float> sct;
FArray<float> dzm;
FArray<float> dzt;
FArray<float> zm;
FArray<float> zt;
FArray<float> vctr1;
FArray<float> vctr2;
FArray<float> vt3dc;
FArray<float> vt3df;
FArray<float> vt3dk;
FArray<float> vt3dg;
FArray<float> scr1;
FArray<float> pke;
FArray<float> the;
FArray<float> thve;
FArray<float> thee;
FArray<float> pe;
FArray<float> te;
FArray<float> qvenv;
FArray<float> dne;
FArray<float> ucon;
FArray<float> vcon;
FArray<float> thtcon;
FArray<float> rvcon;
FArray<float> picon;
FArray<float> tmpcon;
FArray<float> zcon;
FArray<float> zzcon;
float dz;
float dqsdz;
FArray<float> visc;
float viscosity;
float tstpf;
int n;
int nm1;
int l;
FArray<float> cvh;
FArray<float> cvi;
float adiabat;
float wbar;
float vhrel;
float virel;
float zsurf;
float ztop;
float area;
float rsurf;
float alpha;
FArray<float> radius;
FArray<float> heating;
float fmoist;
float bload;
float dt;
float time;
float tdur;
int mintime;
int mdur;
int maxtime;
FArray<float> upe;
FArray<float> vpe;
FArray<float> vel_e;
FArray<float> vel_p;
FArray<float> rad_p;
FArray<float> vel_t;
FArray<float> rad_t;
FArray<float> ztop_;
int testval;
};
struct State {
Plume *plume_store;
bool framework_log_active; int framework_log_output_count;
int wrapper_log_count; int wrapper_log_sites[14];
float absent_float; int absent_int; bool absent_bool; FStr absent_FStr;
struct {
int chem_opt_gocart;
int chem_opt;
int emission_scheme;
} ssalt_mod;
struct {
} module_plumerise;
struct {
} module_smoke_diagnostics;
struct {
} module_fire_emissions;
struct {
} module_add_emiss_burn;
struct {
} module_smoke_plumerise;
struct {
float epsilc;
float pi;
int seas_opt;
int addsmoke_flag;
int n_dbg_lines;
int nfire_types;
bool dbg_opt;
bool aero_ind_fdb;
int chem_opt_gocart;
int num_moist;
int num_emis_seas;
int num_emis_dust;
int p_dust_1;
int p_dust_2;
int p_dust_3;
int p_dust_4;
int p_dust_5;
int p_edust1;
int p_edust2;
int p_edust3;
int p_edust4;
int p_edust5;
int ndvel;
int numgas;
} mpas_smoke_config;
struct {
} dep_dry_simple_mod;
struct {
FArray<float> sc_eff;
FArray<float> ab_eff;
} rad_data_mod;
struct {
int number_ss_bins;
FArray<float> ra;
FArray<float> rb;
FArray<float> emission_scale;
FArray<float> den_seas;
FArray<float> reff_seas;
} seas_data_mod;
struct {
int ndust;
int ndcls;
int ndsrc;
int maxstypes;
int nsalt;
float dyn_visc;
FArray<float> den_dust;
FArray<float> reff_dust;
FArray<float> frac_s;
FArray<float> lo_dust;
FArray<float> up_dust;
int fengsha_maxstypes;
FArray<float> dust_uthres;
int dust_calcdrag;
int dust_moist_opt;
FArray<int> spoint;
FArray<float> reff_salt;
FArray<float> den_salt;
FArray<float> frac_salt;
int max_soiltyp;
FArray<float> maxsmc;
} dust_data_mod;
struct {
float max_dep_vel;
float dep_ref_hgt;
float pi;
float threepi;
float gravity;
float grav;
float boltzmann;
float rgasuniv;
float avo;
float cb;
float cim;
float alpha;
float cin;
float vv;
float a_for;
float a_grs;
float a_wat;
float eps0_for;
float eps0_grs;
float eps0_wat;
float one3;
float two3;
float sqrt2;
float sqrtpi;
float karman;
float conmin;
float pirs;
float f6dpi;
float f6dpim9;
float sigma1;
float mean_diameter1;
float fact_wfa;
float dgmin;
float densmin;
FArray<float> aero_dens;
FArray<float> aero_diam;
} dep_data_mod;
struct {
} dust_fengsha_mod;
struct {
} module_wetdep_ls;
struct {
int p_smoke_fine;
int p_smoke_coarse;
int p_dust_fine;
int p_dust_coarse;
int p_polp_all;
int p_polp_tree;
int p_polp_grass;
int p_polp_weed;
int p_pols_tree;
int p_pols_grass;
int p_pols_weed;
int p_pols_all;
int p_unspc_fine;
int p_unspc_coarse;
int p_ssalt_fine;
int p_ssalt_coarse;
int p_no3_a_fine;
int p_so4_a_fine;
int p_nh4_a_fine;
int p_so2;
int p_nh3;
int p_ch4;
int p_mold_fine;
int p_bact_fine;
int p_plastic_fine;
int p_dms;
} mpas_smoke_init;
struct {
float r80fac;
float rhop;
} seas_ngac_mod;
struct {
int nkp;
int ntime;
Plume* private_thread_coms;
} module_zero_plumegen_coms;
struct {
} module_anthro_emissions;
struct {
} dep_dry_mod_emerson;
struct {
} smoke_driver_slice;
struct {
} mpas_smoke_wrapper;
int saved_diurnal_cycle_icall;
int saved_add_emis_burn_icall;
float saved_makeplume_delz_thresold;
bool saved_visc_w_printed;
int saved_dry_dep_driver_emerson_icall;
};
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:262-600 @1d6e370e.
// Every one of the fourteen literal log sites runs at most once per wrapper
// invocation. Preserve calls even when output is inactive: the original native
// logger owns activity, prefixes, trimming, file I/O and its native counter.
// MPAS-Model-rrfs/src/framework/mpas_log.F:565-595 @1d6e370e.
SDDEV void sd_framework_log(State &state,int site) {
if(state.wrapper_log_count < 14) state.wrapper_log_sites[state.wrapper_log_count] = site;
++state.wrapper_log_count;
if(state.framework_log_active) ++state.framework_log_output_count;
}
SDDEV void gocart_seasalt_driver(State &state,Arena &arena,float dt,FArray<float> alt,FArray<float> t_phy,FArray<float> u_phy,FArray<float> v_phy,int num_chem,FArray<float> chem,FArray<float> rho_phy,FArray<float> dz8w,FArray<float> u10,FArray<float> v10,FArray<float> ustar,FArray<float> p8w,FArray<float> tsk,FArray<float> xland,FArray<float> xlat,FArray<float> xlong,FArray<float> area,float g,FArray<float> e_ss_out,int num_e_ss_out,int index_e_ss_out_ssalt_fine,int index_e_ss_out_ssalt_coarse,float pi,int num_emis_seas,int seas_opt,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void source_ss(State &state,Arena &arena,int imx,int jmx,int lmx,int nmx,float dt1,FArray<float> tc,float pi,FArray<int> ilwi,FArray<float> dxy,FArray<float> w10m,FArray<float> airmas,FArray<float> bems,int ipr);
SDDEV void ebu_driver(State &state,Arena &arena,FArray<float> flam_frac,int kfire,FArray<float> theta_phy,FArray<float> q_vap,FArray<float> rho_phy,FArray<float> vvel,FArray<float> u_phy,FArray<float> v_phy,FArray<float> pi_phy,FArray<float> wind_phy,FArray<float> z_at_w,FArray<float> z,float g,float con_cp,float con_rd,FArray<float> frp_inst,FArray<int> k_min,FArray<int> k_max,int wind_eff_opt,bool do_plumerise,FArray<int> kpbl_thetav,FArray<int> kpbl,float curr_secs,FArray<float> xlat,FArray<float> xlong,FArray<float> uspdavg2d,FArray<float> hpbl2d,float alpha,float frp_min,float frp_wthreshold,float zpbl_lim,float uspd_lim,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,FStr & errmsg,int & errflg,FArray<float> ebu_in,FArray<float> ebu,FArray<float> ebu_in_coarse,FArray<float> ebu_coarse,FArray<float> ebu_in_ch4,FArray<float> ebu_ch4,bool has_ebu_in,bool has_ebu,bool has_ebu_in_coarse,bool has_ebu_coarse,bool has_ebu_in_ch4,bool has_ebu_ch4);
SDDEV void mpas_aod_diag(State &state,Arena &arena,FArray<float> chem,FArray<float> aod3d,FArray<float> rho_phy,FArray<float> dz8w,int num_chem,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void mpas_visibility_diag(State &state,Arena &arena,FArray<float> qcloud,FArray<float> qrain,FArray<float> qice,FArray<float> qsnow,FArray<float> qgrpl,FArray<float> blcldw,FArray<float> blcldi,FArray<float> rho_phy,FArray<float> wind10m,FArray<float> wind,FArray<float> rh2m,FArray<float> rh,FArray<float> qv,FArray<float> t2m,FArray<float> t,FArray<float> coszen,FArray<float> extcoef55,FArray<float> vis,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void compute_emission_factors(State &state,Arena &arena,FArray<float> efs_map,FArray<float> vegfrac,int bb_beta,FArray<int> eco_id,FArray<float> efs_smold,FArray<float> efs_flam,FArray<float> efs_rsmold,FArray<float> fmc_avg,FArray<float> wetness,int nlcat,float dt,float gmt,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void calculate_smoke_emissions(State &state,Arena &arena,float dt,int julday,int nlcat,FArray<float> efs_map,FArray<float> fre_avg,int ebb_dcycle,FArray<float> area,int nblocks,int ktau,int bb_input_prevh,FArray<float> ebu,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void diurnal_cycle(State &state,Arena &arena,float dtstep,FArray<float> dz8w,FArray<float> rho_phy,float pi,float ebb_min,FArray<float> chem,int num_chem,int julday,float gmt,FArray<float> xlat,FArray<float> xlong,FArray<float> fire_end_hr,FArray<float> peak_hr,float time_int,FArray<float> coef_bb_dc,FArray<float> fire_hist,FArray<float> hwp,FArray<float> hwp_avg,FArray<float> hwp_prev_day,FArray<float> vegfrac,FArray<int> eco_id,int nblocks,FArray<float> lu_nofire,FArray<float> lu_qfire,FArray<float> lu_sfire,FArray<float> swdown,int ebb_dcycle,FArray<float> ebu,FArray<int> fire_type,FArray<float> q_vap,bool add_fire_moist_flux,float plume_beta_qv,float hwp_alpha,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void add_emis_burn(State &state,Arena &arena,float dtstep,FArray<float> dz8w,FArray<float> rho_phy,float pi,float ebb_min,FArray<float> chem,int num_chem,int julday,float gmt,FArray<float> xlat,FArray<float> xlong,FArray<float> fire_end_hr,FArray<float> peak_hr,float time_int,FArray<float> coef_bb_dc,FArray<float> fire_hist,FArray<float> hwp,FArray<float> hwp_prevd,FArray<float> swdown,int ebb_dcycle,FArray<float> ebu,FArray<float> ebu_coarse,FArray<float> ebu_ch4,FArray<int> fire_type,FArray<float> q_vap,bool add_fire_moist_flux,float sc_factor,FArray<float> aod3d,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void plumerise(State &state,Arena &arena,int m1,int m2,int m3,int ia,int iz,int ja,int jz,FArray<float> up,FArray<float> vp,FArray<float> wp,FArray<float> theta,FArray<float> pp,FArray<float> dn0,FArray<float> rv,FArray<float> zt_rams,FArray<float> zm_rams,int wind_eff_opt,float frp_inst,int & k1,int & k2,bool dbg_opt,float g,float cp,float rgas,float cpor,FStr & errmsg,int & errflg,int icall,float lat,float f_long,float curr_secs,float alpha,float frp_min);
SDDEV void get_env_condition(State &state,Arena &arena,Plume* coms,int k1,int k2,int & kmt,int wind_eff,float g,float cp,float rgas,float cpor,FStr & errmsg,int & errflg);
SDDEV void set_grid(State &state,Arena &arena,Plume* coms);
SDDEV void set_flam_vert(State &state,Arena &arena,FArray<float> ztopmax,int & k1,int & k2,int nkp,FArray<float> zzcon);
SDDEV void get_fire_properties(State &state,Arena &arena,Plume* coms,int imm,float burnt_area,float frp,FStr & errmsg,int & errflg);
SDDEV void makeplume(State &state,Arena &arena,Plume* coms,int kmt,float & ztopmax,int ixx,int imm,float alpha);
SDDEV void burn(State &state,Arena &arena,Plume* coms,float & eflux,float & water);
SDDEV void lbound(State &state,Arena &arena,Plume* coms,float alpha);
SDDEV void initial(State &state,Arena &arena,Plume* coms,int kmt,float alpha);
SDDEV void damp_grav_wave(State &state,Arena &arena,int ifrom,int nm1,int deltak,float dt,FArray<float> zt,FArray<float> zm,FArray<float> w,FArray<float> t,FArray<float> tt,FArray<float> qv,FArray<float> qh,FArray<float> qi,FArray<float> qc,FArray<float> te,FArray<float> pe,FArray<float> qvenv);
SDDEV void friction(State &state,Arena &arena,int ifrom,int nm1,int deltak,float dt,FArray<float> zt,FArray<float> zm,FArray<float> var1,FArray<float> vart,FArray<float> var2);
SDDEV void vel_advectc_plumerise(State &state,Arena &arena,int m1,FArray<float> wc,FArray<float> wt,FArray<float> rho,FArray<float> dzm);
SDDEV void hadvance_plumerise(State &state,Arena &arena,int iac,int m1,float dt,FArray<float> wc,FArray<float> wt,FArray<float> wp,int mintime);
SDDEV void predict_plumerise(State &state,Arena &arena,int npts,FArray<float> ac,FArray<float> ap,FArray<float> fa,FArray<float> af,int iac,float dtlp,float epsu);
SDDEV void buoyancy_plumerise(State &state,Arena &arena,int m1,FArray<float> t,FArray<float> te,FArray<float> qv,FArray<float> qvenv,FArray<float> qh,FArray<float> qi,FArray<float> qc,FArray<float> wt,FArray<float> scr1);
SDDEV void entrainment(State &state,Arena &arena,Plume* coms,int m1,FArray<float> w,FArray<float> wt,FArray<float> radius,float alpha);
SDDEV void scl_advectc_plumerise(State &state,Arena &arena,Plume* coms,FStr varn,int mzp);
SDDEV void fa_zc_plumerise(State &state,Arena &arena,int m1,FArray<float> scp,FArray<float> scr1,FArray<float> vt3dc,FArray<float> vt3df,FArray<float> vt3dg,FArray<float> vt3dk,FArray<float> vctr1,FArray<float> vctr2);
SDDEV void advtndc_plumerise(State &state,Arena &arena,int m1,FArray<float> scp,FArray<float> sca,FArray<float> sct,float dtl);
SDDEV void tend0_plumerise(State &state,Arena &arena,Plume* coms);
SDDEV void scl_misc(State &state,Arena &arena,Plume* coms,int m1,float alpha);
SDDEV void scl_dyn_entrain(State &state,Arena &arena,int m1,int nkp,float & wbar,FArray<float> w,float & adiabat,float alpha,FArray<float> radius,FArray<float> tt,FArray<float> t,FArray<float> te,FArray<float> qvt,FArray<float> qv,FArray<float> qvenv,FArray<float> qct,FArray<float> qc,FArray<float> qht,FArray<float> qh,FArray<float> qit,FArray<float> qi,FArray<float> vel_e,FArray<float> vel_p,FArray<float> vel_t,FArray<float> rad_p,FArray<float> rad_t);
SDDEV void visc_w(State &state,Arena &arena,Plume* coms,int m1,int deltak,int kmt);
SDDEV void update_plumerise(State &state,Arena &arena,Plume* coms,int m1,FStr varn);
SDDEV void fallpart(State &state,Arena &arena,Plume* coms,int m1);
SDDEV void waterbal(State &state,Arena &arena,Plume* coms);
SDDEV void evaporate(State &state,Arena &arena,Plume* coms);
SDDEV void convert(State &state,Arena &arena,Plume* coms);
SDDEV void sublimate(State &state,Arena &arena,Plume* coms);
SDDEV void glaciate(State &state,Arena &arena,Plume* coms);
SDDEV void melt(State &state,Arena &arena,Plume* coms);
SDDEV void htint(State &state,Arena &arena,int nzz1,FArray<float> vctra,FArray<float> eleva,int nzz2,FArray<float> vctrb,FArray<float> elevb,FStr & errmsg,int & errflg);
SDDEV float esat_pr(State &state,Arena &arena,float tem);
SDDEV void dry_dep_driver_simple(State &state,Arena &arena,FArray<float> rmol,FArray<float> ust,int ndvel,FArray<float> ddvel,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void aero_rad_init(State &state,Arena &arena);
SDDEV void aero_dry_dep_init(State &state,Arena &arena);
SDDEV void gocart_dust_fengsha_driver(State &state,Arena &arena,float dt,int ktau,FArray<float> chem,FArray<float> rho_phy,FArray<float> smois,FArray<float> stemp,FArray<float> p8w,FArray<int> isltyp,FArray<float> snowh,FArray<float> xland,FArray<float> area,float g,FArray<float> ust,FArray<float> znt,FArray<float> clay,FArray<float> sand,FArray<float> uthr,FArray<float> uthr_sg,FArray<float> albedo_drag,FArray<float> feff,FArray<float> sep,FArray<float> e_dust_out,int num_e_dust_out,int index_e_dust_out_dust_fine,int index_e_dust_out_dust_coarse,int num_emis_dust,int num_chem,int num_soil_layers,float dust_alpha,float dust_gamma,float dust_drylimit_factor,float dust_moist_correction,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void source_dust(State &state,Arena &arena,int imx,int jmx,int lmx,int nmx,float dt1,FArray<float> tc,float ustar,FArray<float> massfrac,float erod,float dxy,float smois,float airden,float airmas,FArray<float> bems,float g0,float alpha,float gamma,float r,float uthres,float & source_out_diag,float dust_drylimit_factor);
SDDEV void fengsha_drag(State &state,Arena &arena,float z0,float & r);
SDDEV void dustemissionfengsha(State &state,Arena &arena,float slc,float clay,float sand,float silt,float sep,float feff,float airdens,float ustar,float uthrs,float alpha,float gamma,float kvhmax,float grav,float rhop,float & emissions,float & reason,float dust_drylimit_factor);
SDDEV float soilmoistureconvertvol2grav(State &state,Arena &arena,float vsoil,float sandfrac);
SDDEV float moisturecorrectionfecan(State &state,Arena &arena,float slc,float sand,float clay,float dust_drylimit_factor);
SDDEV float moisturecorrectionshao(State &state,Arena &arena,float slc);
SDDEV float dustfluxv2hratiomb95(State &state,Arena &arena,float clay,float kvhmax);
SDDEV void wetdep_ls(State &state,Arena &arena,float dt,float gravity,FArray<float> var,FArray<float> rain,FArray<float> moist,FArray<float> rho,int nchem,FArray<float> dz8w,FArray<float> vvel,FArray<float> p_phy,float alpha,FArray<float> wetdep_resolved,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void set_scalar_indices(State &state,Arena &arena,int chemistry_start,int index_smoke_fine,int index_smoke_coarse,int index_polp_tree,int index_polp_grass,int index_polp_weed,int index_pols_tree,int index_pols_grass,int index_pols_weed,int index_pols_all,int index_polp_all,int index_unspc_fine,int index_unspc_coarse,int index_ssalt_fine,int index_ssalt_coarse,int index_no3_a_fine,int index_so4_a_fine,int index_nh4_a_fine,int index_so2,int index_nh3,int index_ch4,int index_dust_fine,int index_dust_coarse,bool has_chemistry_start,bool has_index_smoke_fine,bool has_index_smoke_coarse,bool has_index_polp_tree,bool has_index_polp_grass,bool has_index_polp_weed,bool has_index_pols_tree,bool has_index_pols_grass,bool has_index_pols_weed,bool has_index_pols_all,bool has_index_polp_all,bool has_index_unspc_fine,bool has_index_unspc_coarse,bool has_index_ssalt_fine,bool has_index_ssalt_coarse,bool has_index_no3_a_fine,bool has_index_so4_a_fine,bool has_index_nh4_a_fine,bool has_index_so2,bool has_index_nh3,bool has_index_ch4,bool has_index_dust_fine,bool has_index_dust_coarse);
SDDEV void seasaltemission(State &state,Arena &arena,float rlow,float rup,int method,float w10m,float ustar,float pi,float & memissions,float & nemissions,int & rc);
SDDEV float seasaltemissiongong(State &state,Arena &arena,float r,float dr,float w,float scalefac,float afac,float bfac,float rpow,float exppow,float wpow);
SDDEV Plume* get_thread_coms(State &state,Arena &arena);
SDDEV void plumegen_coms_zero(State &state,Arena &arena,Plume* f_this);
SDDEV void mpas_smoke_anthro_emis_driver(State &state,Arena &arena,float dt,float gmt,int julday,int kemit,FArray<float> xlat,FArray<float> xlong,FArray<float> chem,int num_chem,FArray<float> dz8w,FArray<float> t_phy,FArray<float> rho_phy,FArray<float> e_ant_in,FArray<float> e_ant_out,int num_e_ant_in,int num_e_ant_out,int index_e_ant_in_unspc_fine,int index_e_ant_in_unspc_coarse,int index_e_ant_out_unspc_fine,int index_e_ant_out_unspc_coarse,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte);
SDDEV void dry_dep_driver_emerson(State &state,Arena &arena,FArray<float> rmol,FArray<float> ustar,FArray<float> znt,int num_chem,FArray<float> ddvel,FArray<float> chem,FArray<float> delz,FArray<float> snowh,FArray<float> t_phy,FArray<float> p_phy,FArray<float> rho_phy,FArray<int> ivgtyp,float gravity,float dt,FArray<float> drydep_flux,FArray<float> tend_chem_settle,bool dbg_opt,int settling_opt,FArray<float> vg,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,float curr_secs);
SDDEV void depvel(State &state,Arena &arena,float & rmol,float zr,float z0,float ustar,float & aer_res);
SDDEV void particle_settling_wrapper(State &state,Arena &arena,FArray<float> tend_chem_settle,FArray<float> chem,FArray<float> rho_phy,FArray<float> delz_flip,FArray<float> vg,float dt,int kts,int kte,int its,int ite,int jts,int jte,int num_chem,int ims,int ime,int jms,int jme,int kms,int kme);
SDDEV void smoke_from_mpas_fields(State &state,Arena &arena,bool add_fire_heat_flux,bool add_fire_moist_flux,FArray<float> albedo_drag,FArray<float> albedo_drag_p,FArray<float> aod3d_p,FArray<float> area_p,FArray<float> areacell,FArray<float> chem,FArray<float> chem_p,FArray<float> clayfrac_in,FArray<float> clayfrac_in_p,FArray<float> cldfrac,FArray<float> cldfrac_p,FArray<float> coef_bb_dc,FArray<float> coef_bb_dc_p,FStr config_anthro_scheme,FStr config_convection_scheme,FStr config_dust_scheme,FStr config_rwc_scheme,FStr config_smoke_scheme,FArray<float> coszr,FArray<float> coszr_p,FArray<float> ddvel_p,float degrad,FArray<float> dpt2m_p,FArray<float> drydep_flux,FArray<float> drydep_flux_p,int drydep_opt,FArray<float> dx_p,FArray<float> e_ant_in,FArray<float> e_ant_in_p,FArray<float> e_ant_out_p,FArray<float> e_bb_in,FArray<float> e_bb_in_p,FArray<float> e_bb_out,FArray<float> e_bb_out_p,FArray<float> e_dust_out,FArray<float> e_dust_out_p,int ebb_dcycle,FArray<int> eco_id,FArray<int> eco_id_p,FArray<float> efs_flam,FArray<float> efs_flam_p,FArray<float> efs_rsmold,FArray<float> efs_rsmold_p,FArray<float> efs_smold,FArray<float> efs_smold_p,FArray<float> feff,FArray<float> feff_p,FArray<float> fire_ef_out,FArray<float> fire_ef_out_p,FArray<float> fire_end_hr,FArray<float> fire_end_hr_p,FArray<float> fmc_avg_p,FArray<float> fmc_prev24,FArray<float> frac_grid_burned,FArray<float> frac_grid_burned_p,FArray<float> fre_avg_p,FArray<float> fre_in,FArray<float> fre_in_p,FArray<float> fre_out,FArray<float> fre_out_p,FArray<float> fre_prev24,FArray<float> frp_avg_p,FArray<float> frp_in,FArray<float> frp_in_p,FArray<float> frp_out,FArray<float> frp_out_p,FArray<float> frp_prev24,FArray<float> hfx_bb_p,FArray<float> hpbl,FArray<float> hpbl_p,FArray<float> hwp,FArray<float> hwp_avg_p,FArray<float> hwp_p,FArray<float> hwp_prev24,FArray<int> isltyp,FArray<int> isltyp_p,int ite,int its,FArray<int> ivgtyp,FArray<int> ivgtyp_p,int jte,int jts,int kanthro,FArray<int> kpbl,FArray<int> kpbl_p,int kreswoodcomb,int kte,FArray<int> ktop_deep,FArray<int> ktop_deep_p,int kts,int kwildfire,FArray<float> landusef,FArray<float> landusef_p,FArray<float> latcell,float len_disp,FArray<float> loncell,FArray<float> mavail,FArray<float> mavail_p,FArray<int> max_bb_plume_p,FArray<float> meshdensity,FArray<int> min_bb_plume_p,int nblocks,int num_chem,int num_e_ant_in,int num_e_ant_out,int num_e_bb_in,int num_e_bb_out,int num_e_dust_out,int num_landcat,int num_soils,int plumerise_opt,FArray<float> qc_bl,FArray<float> qcbl_p,FArray<float> qfx_bb_p,FArray<float> qi_bl,FArray<float> qibl_p,FArray<float> raincv,FArray<float> raincv_p,FArray<float> rainncv,FArray<float> rainncv_p,FArray<float> rmol,FArray<float> rmol_p,FArray<float> rwc_annual_sum,FArray<float> rwc_annual_sum_p,FArray<float> rwc_annual_sum_smoke_coarse,FArray<float> rwc_annual_sum_smoke_coarse_p,FArray<float> rwc_annual_sum_smoke_fine,FArray<float> rwc_annual_sum_smoke_fine_p,FArray<float> rwc_annual_sum_unspc_coarse,FArray<float> rwc_annual_sum_unspc_coarse_p,FArray<float> rwc_annual_sum_unspc_fine,FArray<float> rwc_annual_sum_unspc_fine_p,FArray<float> rwc_denominator,FArray<float> rwc_denominator_p,FArray<float> sandfrac_in,FArray<float> sandfrac_in_p,FArray<float> sep_in,FArray<float> sep_in_p,FArray<float> skintemp,FArray<float> smois,FArray<float> smois_p,FArray<float> snowh,FArray<float> snowh_p,int t_size,FArray<float> tend_chem_settle_p,FArray<float> totprcp_prev24,FArray<float> totprcp_prev24_p,FArray<float> tsk_p,FArray<float> tslb,FArray<float> tslb_p,FArray<float> u10,FArray<float> u10_p,FArray<float> ust,FArray<float> ust_p,FArray<float> uthres_in,FArray<float> uthres_in_p,FArray<float> uthres_sg_in,FArray<float> uthres_sg_in_p,FArray<float> v10,FArray<float> v10_p,FArray<float> vegfra,FArray<float> vegfra_p,int wetdep_ls_opt,FArray<float> wetdep_resolved,FArray<float> wetdep_resolved_p,FArray<float> xland,FArray<float> xland_p,FArray<float> xlat_p,FArray<float> xlon_p,FArray<float> z0,FArray<float> z0_p,FArray<float> zgrid,FArray<float> zgrid_p,FArray<float> znt,FArray<float> znt_p);
SDDEV void smoke_to_mpas_fields(State &state,Arena &arena,bool add_fire_heat_flux,bool add_fire_moist_flux,FArray<float> aero_emis_for_enhmix,FArray<float> aod3d,FArray<float> aod3d_p,FArray<float> aod3d_smoke,FArray<float> aod3d_smoke_p,FArray<float> aod550,FArray<float> aod550_p,FArray<float> chem,FArray<float> chem_p,FArray<float> coef_bb_dc,FArray<float> coef_bb_dc_p,FStr config_smoke_scheme,FArray<float> ddvel,FArray<float> ddvel_p,FArray<float> drydep_flux,FArray<float> drydep_flux_p,int drydep_opt,FArray<float> e_ant_out,FArray<float> e_ant_out_p,FArray<float> e_bb_out,FArray<float> e_bb_out_p,FArray<float> e_dust_out,FArray<float> e_dust_out_p,int ebb_dcycle,FArray<float> fire_ef_out,FArray<float> fire_ef_out_p,FArray<float> frac_grid_burned,FArray<float> frac_grid_burned_p,FArray<float> fre_out,FArray<float> fre_out_p,FArray<float> frp_out,FArray<float> frp_out_p,FArray<float> hfx_bb,FArray<float> hfx_bb_p,FArray<float> hwp,FArray<float> hwp_p,int ite,int its,int jte,int jts,int kte,int kts,FArray<int> max_bb_plume,FArray<int> max_bb_plume_p,FArray<int> min_bb_plume,FArray<int> min_bb_plume_p,int num_chem,int num_e_ant_out,int num_e_bb_out,int num_e_dust_out,int plumerise_opt,FArray<float> pm10,FArray<float> pm10_p,FArray<float> pm2_5,FArray<float> pm2_5_p,FArray<float> qfx_bb,FArray<float> qfx_bb_p,FArray<float> tend_chem_settle,FArray<float> tend_chem_settle_p,FArray<float> vis,FArray<float> vis_p,int wetdep_ls_opt,FArray<float> wetdep_resolved,FArray<float> wetdep_resolved_p);
SDDEV void smoke_driver_diagnostics(State &state,Arena &arena,FArray<float> aod3d_p,FArray<float> aod550_p,FArray<float> chem_p,int chemistry_start,FArray<float> dz_p,int index_dust_coarse,int index_dust_fine,int index_smoke_fine,int index_unspc_fine,int ite,int its,int jte,int jts,int kte,int kts,FArray<float> pm10_p,FArray<float> pm2_5_p,FArray<float> rho_p);
SDDEV void mpas_smoke_driver(State &state,Arena &arena,int num_chem,int chemistry_start,FArray<float> chem,int kanthro,int kbio,int kfire,int kvol,int krwc,int index_smoke_fine,int index_smoke_coarse,int index_dust_fine,int index_dust_coarse,int index_unspc_fine,int index_ch4,int index_e_bb_in_ch4,int index_e_bb_out_ch4,int index_e_bb_in_smoke_fine,int index_e_bb_in_smoke_coarse,int index_e_ant_in_unspc_fine,int index_e_ant_in_unspc_coarse,int index_e_bb_out_smoke_fine,int index_e_bb_out_smoke_coarse,int index_e_dust_out_dust_fine,int index_e_dust_out_dust_coarse,int index_e_ant_out_unspc_fine,int index_e_ant_out_unspc_coarse,FArray<float> frp_in,FArray<float> frp_out,FArray<float> fre_in,FArray<float> fre_out,FArray<float> hwp,FArray<float> coef_bb_dc,FArray<float> totprcp_prev24,FArray<float> hwp_avg,FArray<float> frp_avg,FArray<float> fre_avg,FArray<float> fire_end_hr,int nblocks,FArray<float> efs_map,FArray<int> eco_id,FArray<float> efs_smold,FArray<float> efs_flam,FArray<float> efs_rsmold,FArray<float> fmc_avg,FArray<float> hfx_bb,FArray<float> qfx_bb,FArray<float> frac_grid_burned,FArray<int> min_bb_plume,FArray<int> max_bb_plume,FArray<float> sandfrac_in,FArray<float> clayfrac_in,FArray<float> uthres_in,FArray<float> uthres_sg_in,FArray<float> albedo_drag_in,FArray<float> feff_in,FArray<float> sep_in,FArray<float> e_ant_in,FArray<float> e_ant_out,int num_e_ant_in,int num_e_ant_out,FArray<float> e_bb_in,FArray<float> e_bb_out,FArray<float> e_dust_out,int num_e_bb_in,int num_e_bb_out,int num_e_dust_out,FArray<float> drydep_flux,FArray<float> ddvel,FArray<float> wetdep_resolved,FArray<float> tend_chem_settle,bool do_mpas_smoke,bool do_mpas_dust,bool do_mpas_anthro,bool do_mpas_rwc,bool calc_bb_emis_online,int bb_beta,int hwp_method,float hwp_alpha,int wetdep_ls_opt,float wetdep_ls_alpha,int plumerise_opt,int plume_wind_eff,float plume_alpha,float bb_emis_scale_factor,int ebb_dcycle,int drydep_opt,int pm_settling,bool add_fire_heat_flux,bool add_fire_moist_flux,int plumerisefire_frq,float bb_qv_scale_factor,float dust_alpha,float dust_gamma,float dust_drylimit_factor,float dust_moist_correction,int bb_input_prevh,float rwc_emis_scale_factor,FArray<float> rwc_denominator,FArray<float> rwc_annual_sum,FArray<float> rwc_annual_sum_smoke_fine,FArray<float> rwc_annual_sum_smoke_coarse,FArray<float> rwc_annual_sum_unspc_fine,FArray<float> rwc_annual_sum_unspc_coarse,FArray<float> vis,FArray<float> qc_vis,FArray<float> qr_vis,FArray<float> qi_vis,FArray<float> qs_vis,FArray<float> qg_vis,FArray<float> blcldw_vis,FArray<float> blcldi_vis,FArray<float> coszen,FArray<float> aod3d_smoke,FArray<float> aod3d,int ktau,float dt,FArray<float> dxcell,FArray<float> area,FArray<float> xland,FArray<float> u10,FArray<float> v10,FArray<float> ust,FArray<float> xlat,FArray<float> xlong,FArray<float> tskin,FArray<float> pblh,FArray<float> t2m,FArray<float> p8w,FArray<float> dz8w,FArray<float> z_at_w,FArray<float> p_phy,FArray<float> t_phy,FArray<float> u_phy,FArray<float> v_phy,FArray<float> qv,FArray<float> vvel,FArray<float> pi_phy,FArray<float> rho_phy,FArray<int> kpbl,int nsoil,FArray<float> smois,FArray<float> tslb,FArray<int> ivgtyp,FArray<int> isltyp,int nlcat,FArray<float> swdown,FArray<float> z0,FArray<float> snowh,float julian,FArray<float> rmol,FArray<float> raincv,FArray<float> rainncv,FArray<float> dpt2m,FArray<float> znt,FArray<float> mavail,float g,FArray<float> vegfra,FArray<float> landusef,FArray<float> cldfrac,FArray<int> ktop_deep,float cp,float rd,float gmt,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,bool has_index_smoke_coarse,bool has_index_ch4,bool has_index_e_bb_in_ch4,bool has_index_e_bb_out_ch4,bool has_index_e_bb_in_smoke_coarse,bool has_index_e_bb_out_smoke_coarse,bool has_frp_in,bool has_frp_out,bool has_fre_in,bool has_fre_out,bool has_hwp,bool has_coef_bb_dc,bool has_totprcp_prev24,bool has_hwp_avg,bool has_frp_avg,bool has_fre_avg,bool has_fire_end_hr,bool has_efs_map,bool has_eco_id,bool has_efs_smold,bool has_efs_flam,bool has_efs_rsmold,bool has_fmc_avg,bool has_hfx_bb,bool has_qfx_bb,bool has_frac_grid_burned,bool has_min_bb_plume,bool has_max_bb_plume,bool has_sandfrac_in,bool has_clayfrac_in,bool has_uthres_in,bool has_uthres_sg_in,bool has_albedo_drag_in,bool has_feff_in,bool has_sep_in,bool has_e_ant_in,bool has_e_ant_out,bool has_e_bb_in,bool has_e_bb_out,bool has_e_dust_out,bool has_drydep_flux,bool has_ddvel,bool has_wetdep_resolved,bool has_tend_chem_settle,bool has_rwc_denominator,bool has_rwc_annual_sum,bool has_rwc_annual_sum_smoke_fine,bool has_rwc_annual_sum_smoke_coarse,bool has_rwc_annual_sum_unspc_fine,bool has_rwc_annual_sum_unspc_coarse,bool has_vis,bool has_qc_vis,bool has_qr_vis,bool has_qi_vis,bool has_qs_vis,bool has_qg_vis,bool has_blcldw_vis,bool has_blcldi_vis,bool has_ktop_deep);
SDDEV void mpas_smoke_prep(State &state,Arena &arena,bool do_mpas_smoke,int ktau,int nlcat,float cp,int ebb_dcycle,float ebb_min,FArray<float> xland,FArray<float> xlat,FArray<float> xlong,FArray<int> ivgtyp,FArray<int> isltyp,FArray<float> vegfrac,FArray<float> snowh,FArray<float> u10,FArray<float> v10,FArray<float> wind10m,FArray<float> t2m,FArray<float> dpt2m,FArray<float> wetness,FArray<float> hwp,FArray<float> hwp_day_avg,int index_e_bb_in_smoke_fine,int num_e_bb_in,int kfire,FArray<float> e_bb_in,FArray<float> t_phy,FArray<float> u_phy,FArray<float> v_phy,FArray<float> p_phy,FArray<float> pi_phy,FArray<float> z_at_w,FArray<float> dz8w,FArray<float> dz8w_flip,FArray<float> rho_phy,FArray<float> qv,FArray<float> relhum,FArray<float> rh2m,FArray<float> rri,FArray<float> total_flashrate,FArray<float> wind_phy,FArray<float> theta_phy,FArray<float> zmid,FArray<int> kpbl_thetav,FArray<float> peak_hr,FArray<float> coef_bb_dc,FArray<float> fire_hist,FArray<float> lu_nofire,FArray<float> lu_qfire,FArray<float> lu_sfire,FArray<int> fire_type,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,bool has_hwp,bool has_e_bb_in,bool has_coef_bb_dc);

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/ssalt_mod.F90:25-158 @1d6e370e; SMOKE @9652bbf
SDDEV void gocart_seasalt_driver(State &state,Arena &arena,float dt,FArray<float> alt,FArray<float> t_phy,FArray<float> u_phy,FArray<float> v_phy,int num_chem,FArray<float> chem,FArray<float> rho_phy,FArray<float> dz8w,FArray<float> u10,FArray<float> v10,FArray<float> ustar,FArray<float> p8w,FArray<float> tsk,FArray<float> xland,FArray<float> xlat,FArray<float> xlong,FArray<float> area,float g,FArray<float> e_ss_out,int num_e_ss_out,int index_e_ss_out_ssalt_fine,int index_e_ss_out_ssalt_coarse,float pi,int num_emis_seas,int seas_opt,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
alt.bind(ims,ime,kms,kme,jms,jme);
t_phy.bind(ims,ime,kms,kme,jms,jme);
u_phy.bind(ims,ime,kms,kme,jms,jme);
v_phy.bind(ims,ime,kms,kme,jms,jme);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
dz8w.bind(ims,ime,kms,kme,jms,jme);
u10.bind(ims,ime,jms,jme);
v10.bind(ims,ime,jms,jme);
ustar.bind(ims,ime,jms,jme);
p8w.bind(ims,ime,kms,kme,jms,jme);
tsk.bind(ims,ime,jms,jme);
xland.bind(ims,ime,jms,jme);
xlat.bind(ims,ime,jms,jme);
xlong.bind(ims,ime,jms,jme);
area.bind(ims,ime,jms,jme);
e_ss_out.bind(ims,ime,kms,kme,jms,jme,1,num_e_ss_out);
int p_seas_1 = 15;
int p_seas_2 = 16;
int p_seas_3 = 17;
int p_seas_4 = 18;
int p_seas_5 = 19;
int p_eseas1 = 1;
int p_eseas2 = 2;
int p_eseas3 = 3;
int p_eseas4 = 4;
int p_eseas5 = 5;
int ipr;
int i;
int j;
int imx;
int jmx;
int lmx;
int n;
int rc;
auto ilwi = arena.alloc<int>(1,1,1,1);
float fsstemis;
float memissions;
float nemissions;
float tskin_c;
float ws10m;
float delp;
auto tc = arena.alloc<float>(1,state.seas_data_mod.number_ss_bins);
auto bems = arena.alloc<float>(1,state.seas_data_mod.number_ss_bins);
auto w10m = arena.alloc<float>(1,1,1,1);
auto airmas = arena.alloc<float>(1,1,1,1);
auto tskin = arena.alloc<float>(1,1,1,1);
auto dxy = arena.alloc<float>(1,1);
auto airmas1 = arena.alloc<float>(1,1,1,1,1,1);
auto tc1 = arena.alloc<float>(1,1,1,1,1,1,1,state.seas_data_mod.number_ss_bins);
auto bems1 = arena.alloc<float>(1,1,1,1,1,state.seas_data_mod.number_ss_bins);
float conver = __uint_as_float(0x3089705FU);
float converi = __uint_as_float(0x4E6E6B28U);
imx = 1; // L102
jmx = 1; // L103
lmx = 1; // L104
for (j=jts;j<=jte;j+=1) { // L107
for (i=its;i<=ite;i+=1) { // L108
if ((sd_sub(xland(i,j),1.5f) > 0.f)) { // L111
delp = sd_sub(p8w(i,kts,j),p8w(i,(kts + 1),j)); // L114
if ((dz8w(i,kts,j) < 12.f)) { // L115
sd_assign(w10m,sd_sqrt(sd_add(sd_mul(u_phy(i,kts,j),u_phy(i,kts,j)),sd_mul(v_phy(i,kts,j),v_phy(i,kts,j)))),arena); // L116
} else { // L117
sd_assign(w10m,sd_sqrt(sd_add(sd_mul(u10(i,j),u10(i,j)),sd_mul(v10(i,j),v10(i,j)))),arena); // L118
} // L119
ilwi(1,1) = 0; // L121
sd_assign(tc,0.f,arena); // L122
tskin(1,1) = tsk(i,j); // L123
airmas(1,1) = sd_div(sd_mul(area(i,j),delp),g); // L124
dxy(1) = area(i,j); // L125
ipr = 0; // L126
airmas1(1,1,1) = airmas(1,1); // L128
sd_assign(tc1.section(1,1,1,Slice()),tc,arena); // L129
sd_assign(bems1.section(1,1,Slice()),bems,arena); // L130
source_ss(state,arena,imx,jmx,lmx,state.seas_data_mod.number_ss_bins,dt,tc1,pi,ilwi,dxy,w10m,airmas1,bems1,ipr); // L131
sd_assign(tc,tc1.section(1,1,1,Slice()),arena); // L132
sd_assign(bems,bems1.section(1,1,Slice()),arena); // L133
chem(i,kts,j,state.mpas_smoke_init.p_ssalt_fine) = sd_add(chem(i,kts,j,state.mpas_smoke_init.p_ssalt_fine),sd_mul(sd_add(tc(1),sd_mul(0.286f,tc(2))),converi)); // L135
chem(i,kts,j,state.mpas_smoke_init.p_ssalt_coarse) = sd_add(chem(i,kts,j,state.mpas_smoke_init.p_ssalt_coarse),sd_mul(sd_add(sd_add(sd_mul(0.714f,tc(2)),tc(3)),tc(4)),converi)); // L137
e_ss_out(i,1,j,index_e_ss_out_ssalt_fine) = sd_mul(converi,sd_add(bems(1),sd_mul(0.286f,bems(2)))); // L141
e_ss_out(i,1,j,index_e_ss_out_ssalt_coarse) = sd_mul(converi,sd_add(sd_add(sd_mul(0.714f,bems(2)),bems(3)),bems(4))); // L142
} // L152
loop2_continue: ; } loop2_exit: ; // L154
loop1_continue: ; } loop1_exit: ; // L155
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/ssalt_mod.F90:160-320 @1d6e370e; SMOKE @9652bbf
SDDEV void source_ss(State &state,Arena &arena,int imx,int jmx,int lmx,int nmx,float dt1,FArray<float> tc,float pi,FArray<int> ilwi,FArray<float> dxy,FArray<float> w10m,FArray<float> airmas,FArray<float> bems,int ipr) {
ArenaScope scope(arena);
tc.bind(1,imx,1,jmx,1,lmx,1,nmx);
ilwi.bind(1,imx,1,jmx);
dxy.bind(1,jmx);
w10m.bind(1,imx,1,jmx);
airmas.bind(1,imx,1,jmx,1,lmx);
bems.bind(1,imx,1,jmx,1,nmx);
auto c0 = arena.alloc<float>(1,5);
auto b0 = arena.alloc<float>(1,2);
auto c_old = arena.alloc<float>(1,5);
c_old(1) = __uint_as_float(0x3FAFBE77U);
c_old(2) = __uint_as_float(0x404CCCCDU);
c_old(3) = __uint_as_float(0x3D6978D5U);
c_old(4) = __uint_as_float(0x3F866666U);
c_old(5) = __uint_as_float(0x3F9851ECU);
auto c_new = arena.alloc<float>(1,5);
c_new(1) = __uint_as_float(0x3FAFBE77U);
c_new(2) = __uint_as_float(0x404CCCCDU);
c_new(3) = __uint_as_float(0x3D6978D5U);
c_new(4) = __uint_as_float(0x405CCCCDU);
c_new(5) = __uint_as_float(0x3FCDB22DU);
auto b_old = arena.alloc<float>(1,2);
b_old(1) = __uint_as_float(0x3EC28F5CU);
b_old(2) = __uint_as_float(0x3F266666U);
auto b_new = arena.alloc<float>(1,2);
b_new(1) = __uint_as_float(0x3EDDB22DU);
b_new(2) = __uint_as_float(0x3EDDB22DU);
float dr = __uint_as_float(0x3D4CCCCDU);
float theta = __uint_as_float(0x41F00000U);
float frh = __uint_as_float(0x40000000U);
bool old = true;
bool f_new = false;
float rho_d;
float r0;
float r1;
float r;
float r_w;
float a;
float b;
float dfn;
float r_d;
float dfm;
float src;
int i;
int j;
int n;
int nr;
int ir;
float fudge_fac;
auto tcmw = arena.alloc<float>(1,nmx);
auto ar = arena.alloc<float>(1,nmx);
auto tcvv = arena.alloc<float>(1,nmx);
auto ar_wetdep = arena.alloc<float>(1,nmx);
auto kc = arena.alloc<float>(1,nmx);
auto tcname = arena.alloc<FStr>(1,nmx);
auto tcunits = arena.alloc<FStr>(1,nmx);
auto aerosol = arena.alloc<bool>(1,nmx);
auto tc1 = arena.alloc<float>(1,imx,1,jmx,1,lmx,1,nmx);
auto tcms = arena.alloc<float>(1,imx,1,jmx,1,lmx,1,nmx);
auto tcgm = arena.alloc<float>(1,imx,1,jmx,1,lmx,1,nmx);
auto e_an = arena.alloc<float>(1,imx,1,jmx,1,2,1,nmx);
auto e_bb = arena.alloc<float>(1,imx,1,jmx,1,nmx);
auto e_ac = arena.alloc<float>(1,imx,1,jmx,1,lmx,1,nmx);
auto tmas0 = arena.alloc<float>(1,nmx);
auto tmas1 = arena.alloc<float>(1,nmx);
auto dtems = arena.alloc<float>(1,nmx);
auto dttrp = arena.alloc<float>(1,nmx);
auto dtdif = arena.alloc<float>(1,nmx);
auto dtcnv = arena.alloc<float>(1,nmx);
auto dtwet = arena.alloc<float>(1,nmx);
auto dtdry = arena.alloc<float>(1,nmx);
auto dtstl = arena.alloc<float>(1,nmx);
auto dtems2 = arena.alloc<float>(1,nmx);
auto dttrp2 = arena.alloc<float>(1,nmx);
auto dtdif2 = arena.alloc<float>(1,nmx);
auto dtcnv2 = arena.alloc<float>(1,nmx);
auto dtwet2 = arena.alloc<float>(1,nmx);
auto dtdry2 = arena.alloc<float>(1,nmx);
auto dtstl2 = arena.alloc<float>(1,nmx);
auto ems_an = arena.alloc<float>(1,imx,1,jmx,1,nmx);
auto ems_bb = arena.alloc<float>(1,imx,1,jmx,1,nmx);
auto ems_tp = arena.alloc<float>(1,imx,1,jmx);
auto ems_ac = arena.alloc<float>(1,imx,1,jmx,1,lmx,1,nmx);
auto ems_co = arena.alloc<float>(1,imx,1,jmx,1,nmx);
fudge_fac = .25f; // L275
for (n=1;n<=nmx;n+=1) { // L277
sd_assign(bems.section(Slice(),Slice(),n),0.0f,arena); // L278
rho_d = state.seas_data_mod.den_seas(n); // L279
r0 = sd_mul(state.seas_data_mod.ra(n),frh); // L280
r1 = sd_mul(state.seas_data_mod.rb(n),frh); // L281
r = r0; // L282
nr = int(sd_add(sd_div(sd_sub(r1,r0),dr),.001f)); // L283
for (ir=1;ir<=nr;ir+=1) { // L284
r_w = sd_add(r,sd_mul(dr,0.5f)); // L285
r = sd_add(r,dr); // L286
if (f_new) { // L287
a = sd_mul(4.7f,sd_pow(sd_add(1.0f,sd_mul(theta,r_w)),sd_mul(sd_neg(0.017f),sd_pow(r_w,sd_neg(1.44f))))); // L288
sd_assign(c0,c_new,arena); // L289
sd_assign(b0,b_new,arena); // L290
} else { // L291
a = 3.0f; // L292
sd_assign(c0,c_old,arena); // L293
sd_assign(b0,b_old,arena); // L294
} // L295
b = sd_div(sd_sub(b0(1),sd_log10(r_w)),b0(2)); // L297
dfn = sd_mul(sd_mul(sd_div(c0(1),sd_pow(r_w,a)),sd_add(1.0f,sd_mul(c0(3),sd_pow(r_w,c0(4))))),sd_pow(10,sd_mul(c0(5),sd_exp(sd_neg(sd_ipow(b,2)))))); // L298
r_d = sd_mul(sd_div(r_w,frh),1.0e-6); // L301
dfm = sd_mul(sd_mul(sd_mul(sd_mul(sd_mul(sd_mul(sd_mul(sd_div(4.0f,3.0f),pi),sd_ipow(r_d,3)),rho_d),frh),dfn),dr),dt1); // L302
for (i=1;i<=imx;i+=1) { // L303
for (j=1;j<=jmx;j+=1) { // L304
if ((ilwi(i,j) == 0)) { // L306
src = sd_mul(sd_mul(dfm,dxy(j)),sd_pow(w10m(i,j),c0(2))); // L308
tc(i,j,1,n) = sd_add(tc(i,j,1,n),sd_div(sd_mul(fudge_fac,src),airmas(i,j,1))); // L310
} else { // L311
src = 0.0f; // L312
} // L313
bems(i,j,n) = sd_add(bems(i,j,n),sd_div(sd_mul(src,fudge_fac),sd_mul(dxy(j),dt1))); // L314
loop4_continue: ; } loop4_exit: ; // L315
loop3_continue: ; } loop3_exit: ; // L316
loop2_continue: ; } loop2_exit: ; // L317
loop1_continue: ; } loop1_exit: ; // L318
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_plumerise.F90:17-180 @1d6e370e; SMOKE @9652bbf
SDDEV void ebu_driver(State &state,Arena &arena,FArray<float> flam_frac,int kfire,FArray<float> theta_phy,FArray<float> q_vap,FArray<float> rho_phy,FArray<float> vvel,FArray<float> u_phy,FArray<float> v_phy,FArray<float> pi_phy,FArray<float> wind_phy,FArray<float> z_at_w,FArray<float> z,float g,float con_cp,float con_rd,FArray<float> frp_inst,FArray<int> k_min,FArray<int> k_max,int wind_eff_opt,bool do_plumerise,FArray<int> kpbl_thetav,FArray<int> kpbl,float curr_secs,FArray<float> xlat,FArray<float> xlong,FArray<float> uspdavg2d,FArray<float> hpbl2d,float alpha,float frp_min,float frp_wthreshold,float zpbl_lim,float uspd_lim,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,FStr & errmsg,int & errflg,FArray<float> ebu_in,FArray<float> ebu,FArray<float> ebu_in_coarse,FArray<float> ebu_coarse,FArray<float> ebu_in_ch4,FArray<float> ebu_ch4,bool has_ebu_in,bool has_ebu,bool has_ebu_in_coarse,bool has_ebu_coarse,bool has_ebu_in_ch4,bool has_ebu_ch4) {
ArenaScope scope(arena);
flam_frac.bind(ims,ime,jms,jme);
theta_phy.bind(ims,ime,kms,kme,jms,jme);
q_vap.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
vvel.bind(ims,ime,kms,kme,jms,jme);
u_phy.bind(ims,ime,kms,kme,jms,jme);
v_phy.bind(ims,ime,kms,kme,jms,jme);
pi_phy.bind(ims,ime,kms,kme,jms,jme);
wind_phy.bind(ims,ime,kms,kme,jms,jme);
z_at_w.bind(ims,ime,kms,kme,jms,jme);
z.bind(ims,ime,kms,kme,jms,jme);
frp_inst.bind(ims,ime,jms,jme);
k_min.bind(ims,ime,jms,jme);
k_max.bind(ims,ime,jms,jme);
kpbl_thetav.bind(ims,ime,jms,jme);
kpbl.bind(ims,ime,jms,jme);
xlat.bind(ims,ime,jms,jme);
xlong.bind(ims,ime,jms,jme);
uspdavg2d.bind(ims,ime,jms,jme);
hpbl2d.bind(ims,ime,jms,jme);
ebu_in.bind(ims,ime,1,kfire,jms,jme);
ebu.bind(ims,ime,kms,kme,jms,jme);
ebu_in_coarse.bind(ims,ime,1,kfire,jms,jme);
ebu_coarse.bind(ims,ime,kms,kme,jms,jme);
ebu_in_ch4.bind(ims,ime,1,kfire,jms,jme);
ebu_ch4.bind(ims,ime,kms,kme,jms,jme);
int nv;
int i;
int j;
int k;
int kp1;
int kp2;
int icall;
auto u_in = arena.alloc<float>(1,kte);
auto v_in = arena.alloc<float>(1,kte);
auto w_in = arena.alloc<float>(1,kte);
auto theta_in = arena.alloc<float>(1,kte);
auto pi_in = arena.alloc<float>(1,kte);
auto rho_phyin = arena.alloc<float>(1,kte);
auto qv_in = arena.alloc<float>(1,kte);
auto zmid = arena.alloc<float>(1,kte);
auto z_lev = arena.alloc<float>(1,kte);
auto uspd = arena.alloc<float>(1,kte);
float dz_plume;
float cpor;
float con_rocp;
cpor = sd_div(con_cp,con_rd); // L77
con_rocp = sd_div(con_rd,con_cp); // L78
if ((state.mpas_smoke_config.dbg_opt && (sd_mod(int(curr_secs),1800) == 0))) { // L80
icall = 0; // L81
} // L82
for (j=jts;j<=jte;j+=1) { // L86
for (k=kts;k<=kte;k+=1) { // L87
for (i=its;i<=ite;i+=1) { // L88
ebu(i,k,j) = 0.f; // L89
if (has_ebu_in_coarse) { ebu_coarse(i,k,j) = 0.f; } // L90
if (has_ebu_in_ch4) { ebu_ch4(i,k,j) = 0.f; } // L91
loop3_continue: ; } loop3_exit: ; // L92
loop2_continue: ; } loop2_exit: ; // L93
loop1_continue: ; } loop1_exit: ; // L94
if (do_plumerise) { // L99
for (j=jts;j<=jte;j+=1) { // L100
for (i=its;i<=ite;i+=1) { // L101
for (k=kts;k<=kte;k+=1) { // L103
u_in(k) = u_phy(i,k,j); // L104
v_in(k) = v_phy(i,k,j); // L105
w_in(k) = vvel(i,k,j); // L106
qv_in(k) = q_vap(i,k,j); // L107
pi_in(k) = pi_phy(i,k,j); // L108
zmid(k) = sd_sub(z(i,k,j),z_at_w(i,kts,j)); // L109
z_lev(k) = sd_sub(z_at_w(i,k,j),z_at_w(i,kts,j)); // L110
rho_phyin(k) = rho_phy(i,k,j); // L111
theta_in(k) = theta_phy(i,k,j); // L112
loop6_continue: ; } loop6_exit: ; // L114
plumerise(state,arena,kte,1,1,1,1,1,1,u_in,v_in,w_in,theta_in,pi_in,rho_phyin,qv_in,zmid,z_lev,wind_eff_opt,frp_inst(i,j),k_min(i,j),k_max(i,j),state.mpas_smoke_config.dbg_opt,g,con_cp,con_rd,cpor,errmsg,errflg,icall,xlat(i,j),xlong(i,j),curr_secs,alpha,frp_min); // L117
if ((errflg != 0)) { return; } // L126
kp1 = k_min(i,j); // L128
kp2 = k_max(i,j); // L129
if ((frp_inst(i,j) <= frp_min)) { // L138
flam_frac(i,j) = 0.f; // L141
} else if (((((frp_inst(i,j) <= frp_wthreshold) && (uspdavg2d(i,1) >= uspd_lim)) && (hpbl2d(i,1) > zpbl_lim)) && (wind_eff_opt == 1))) { // L142
kp1 = 2; // L144
kp2 = sd_max(3,sd_nint(sd_div(float(kpbl(i,j)),3.f))); // L145
flam_frac(i,j) = 0.85f; // L146
} else { // L147
flam_frac(i,j) = 0.9f; // L148
} // L149
dz_plume = sd_sub(z_at_w(i,kp2,j),z_at_w(i,kp1,j)); // L153
for (k=kp1;k<=(kp2 - 1);k+=1) { // L154
ebu(i,k,j) = sd_div(sd_mul(sd_mul(flam_frac(i,j),ebu_in(i,kts,j)),sd_sub(z_at_w(i,(k + 1),j),z_at_w(i,k,j))),dz_plume); // L155
if (has_ebu_in_coarse) { ebu_coarse(i,k,j) = sd_div(sd_mul(sd_mul(flam_frac(i,j),ebu_in_coarse(i,kts,j)),sd_sub(z_at_w(i,(k + 1),j),z_at_w(i,k,j))),dz_plume); } // L156
if (has_ebu_in_ch4) { ebu_ch4(i,k,j) = sd_div(sd_mul(sd_mul(flam_frac(i,j),ebu_in_ch4(i,kts,j)),sd_sub(z_at_w(i,(k + 1),j),z_at_w(i,k,j))),dz_plume); } // L157
loop7_continue: ; } loop7_exit: ; // L158
ebu(i,kts,j) = sd_mul(sd_sub(1.f,flam_frac(i,j)),ebu_in(i,kts,j)); // L159
if (has_ebu_in_coarse) { ebu_coarse(i,kts,j) = sd_mul(sd_sub(1.f,flam_frac(i,j)),ebu_in_coarse(i,kts,j)); } // L160
if (has_ebu_in_ch4) { ebu_ch4(i,kts,j) = sd_mul(sd_sub(1.f,flam_frac(i,j)),ebu_in_ch4(i,kts,j)); } // L161
k_min(i,j) = kp1; // L165
k_max(i,j) = kp2; // L166
if (((state.mpas_smoke_config.dbg_opt && (icall <= state.mpas_smoke_config.n_dbg_lines)) && (frp_inst(i,j) >= frp_min))) { // L168
if ((frp_inst(i,j) >= 2.e+10f)) { // L169
} // L170
icall = (icall + 1); // L171
} // L172
loop5_continue: ; } loop5_exit: ; // L175
loop4_continue: ; } loop4_exit: ; // L176
} // L178
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_diagnostics.F90:20-53 @1d6e370e; SMOKE @9652bbf
SDDEV void mpas_aod_diag(State &state,Arena &arena,FArray<float> chem,FArray<float> aod3d,FArray<float> rho_phy,FArray<float> dz8w,int num_chem,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
aod3d.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
dz8w.bind(ims,ime,kms,kme,jms,jme);
float ext;
int i;
int k;
int j;
int nv;
sd_assign(aod3d.section(Slice(),Slice(),Slice()),0.f,arena); // L40
for (nv=1;nv<=num_chem;nv+=1) { // L41
if ((nv == state.mpas_smoke_init.p_ch4)) { goto loop1_continue; } // L42
ext = sd_add(state.rad_data_mod.sc_eff(nv),state.rad_data_mod.ab_eff(nv)); // L43
for (j=jts;j<=jte;j+=1) { // L44
for (k=kts;k<=kte;k+=1) { // L45
for (i=its;i<=ite;i+=1) { // L46
aod3d(i,k,j) = sd_add(aod3d(i,k,j),sd_mul(sd_mul(sd_mul(sd_mul(1.e-6f,ext),chem(i,k,j,nv)),rho_phy(i,k,j)),dz8w(i,k,j))); // L47
loop4_continue: ; } loop4_exit: ; // L48
loop3_continue: ; } loop3_exit: ; // L49
loop2_continue: ; } loop2_exit: ; // L50
loop1_continue: ; } loop1_exit: ; // L51
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_diagnostics.F90:55-184 @1d6e370e; SMOKE @9652bbf
SDDEV void mpas_visibility_diag(State &state,Arena &arena,FArray<float> qcloud,FArray<float> qrain,FArray<float> qice,FArray<float> qsnow,FArray<float> qgrpl,FArray<float> blcldw,FArray<float> blcldi,FArray<float> rho_phy,FArray<float> wind10m,FArray<float> wind,FArray<float> rh2m,FArray<float> rh,FArray<float> qv,FArray<float> t2m,FArray<float> t,FArray<float> coszen,FArray<float> extcoef55,FArray<float> vis,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
qcloud.bind(ims,ime,kms,kme,jms,jme);
qrain.bind(ims,ime,kms,kme,jms,jme);
qice.bind(ims,ime,kms,kme,jms,jme);
qsnow.bind(ims,ime,kms,kme,jms,jme);
qgrpl.bind(ims,ime,kms,kme,jms,jme);
blcldw.bind(ims,ime,kms,kme,jms,jme);
blcldi.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
wind10m.bind(ims,ime,jms,jme);
wind.bind(ims,ime,kms,kme,jms,jme);
rh2m.bind(ims,ime,jms,jme);
rh.bind(ims,ime,kms,kme,jms,jme);
qv.bind(ims,ime,kms,kme,jms,jme);
t2m.bind(ims,ime,jms,jme);
t.bind(ims,ime,kms,kme,jms,jme);
coszen.bind(ims,ime,jms,jme);
extcoef55.bind(ims,ime,kms,kme,jms,jme);
vis.bind(ims,ime,jms,jme);
int i;
int j;
int k;
int d;
float visfactor = __uint_as_float(0x407A5E35U);
float bc;
float br;
float bi;
float bs;
float hydro_extcoeff;
float extcoeff;
float vis_haze;
float tvd;
float qrh;
float prob_ext_coeff_gt_p29;
float haze_ext_coeff;
float vis_hydlith;
float alpha_haze;
float rhmax;
float bg;
float qcloud2;
float blcldw2;
float qrain2;
float qice2;
float blcldi2;
float qsnow2;
float qgrpl2;
float extcoeff552;
float vis_night;
float zen_fac;
float tiny_number;
auto vis_alpha = arena.alloc<float>(its,ite,jts,jte);
tiny_number = 1.e-12f; // L86
for (j=jts;j<=jte;j+=1) { // L88
for (i=its;i<=ite;i+=1) { // L89
qcloud2 = 0.f; // L91
blcldw2 = 0.f; // L92
qrain2 = 0.f; // L93
qice2 = 0.f; // L94
blcldi2 = 0.f; // L95
qsnow2 = 0.f; // L96
qgrpl2 = 0.f; // L97
extcoeff552 = 0.f; // L98
k = kts; // L101
qcloud2 = sd_mul(sd_mul(qcloud(i,k,j),rho_phy(i,k,j)),1000.f); // L103
blcldw2 = sd_mul(sd_mul(blcldw(i,k,j),rho_phy(i,k,j)),1000.f); // L104
qrain2 = sd_mul(sd_mul(qrain(i,k,j),rho_phy(i,k,j)),1000.f); // L105
qice2 = sd_mul(sd_mul(qice(i,k,j),rho_phy(i,k,j)),1000.f); // L106
blcldi2 = sd_mul(sd_mul(blcldi(i,k,j),rho_phy(i,k,j)),1000.f); // L107
qsnow2 = sd_mul(sd_mul(qsnow(i,k,j),rho_phy(i,k,j)),1000.f); // L108
qgrpl2 = sd_mul(sd_mul(qgrpl(i,k,j),rho_phy(i,k,j)),1000.f); // L109
extcoeff552 = sd_mul(extcoef55(i,k,j),1.e-3f); // L110
bc = sd_mul(144.7f,sd_pow(sd_add(qcloud2,blcldw2),0.88f)); // L113
br = sd_mul(2.24f,sd_pow(qrain2,0.75f)); // L114
bi = sd_mul(327.8f,sd_pow(sd_add(qice2,blcldi2),1.00f)); // L115
bs = sd_mul(10.36f,sd_pow(qsnow2,0.7776f)); // L116
bg = sd_mul(8.0f,sd_pow(qgrpl2,0.7500f)); // L117
hydro_extcoeff = sd_mul(sd_add(sd_add(sd_add(sd_add(bc,br),bi),bs),bg),1.e-3f); // L119
vis_haze = 999999.f; // L122
if ((qv(i,1,j) > 0.f)) { // L123
vis_haze = sd_mul(sd_mul(1500.f,sd_sub(105.f,rh2m(i,j))),sd_div(5.f,sd_min(sd_mul(1000.f,qv(i,1,j)),5.f))); // L124
} // L125
rhmax = sd_max(rh(i,1,j),rh(i,2,j)); // L128
qrh = sd_max(0.f,sd_min(0.8f,sd_sub(sd_mul(rhmax,0.01f),0.15f))); // L129
vis_haze = sd_mul(90000.f,sd_exp(sd_mul(sd_neg(2.5f),qrh))); // L130
extcoeff = sd_add(hydro_extcoeff,extcoeff552); // L155
if ((extcoeff > tiny_number)) { // L156
vis_hydlith = sd_min(sd_div(visfactor,extcoeff),999999.f); // L157
} else { // L158
vis_hydlith = 999999.f; // L159
} // L160
vis_night = sd_mul(sd_mul(1.69f,sd_pow(sd_div(vis_hydlith,1.609f),0.86f)),1.609f); // L164
zen_fac = sd_mul(sd_min(0.1f,sd_max(coszen(i,j),0.f)),10.f); // L165
vis_hydlith = sd_add(sd_mul(zen_fac,vis_hydlith),sd_mul(sd_sub(1.f,zen_fac),vis_night)); // L166
if ((vis_hydlith < vis_haze)) { // L173
vis(i,j) = vis_hydlith; // L174
} else { // L176
vis(i,j) = vis_haze; // L177
} // L179
loop2_continue: ; } loop2_exit: ; // L181
loop1_continue: ; } loop1_exit: ; // L182
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_fire_emissions.F90:24-191 @1d6e370e; SMOKE @9652bbf
SDDEV void compute_emission_factors(State &state,Arena &arena,FArray<float> efs_map,FArray<float> vegfrac,int bb_beta,FArray<int> eco_id,FArray<float> efs_smold,FArray<float> efs_flam,FArray<float> efs_rsmold,FArray<float> fmc_avg,FArray<float> wetness,int nlcat,float dt,float gmt,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
efs_map.bind(ims,ime,jms,jme);
vegfrac.bind(ims,ime,1,nlcat,jms,jme);
eco_id.bind(ims,ime,jms,jme);
efs_smold.bind(ims,ime,jms,jme);
efs_flam.bind(ims,ime,jms,jme);
efs_rsmold.bind(ims,ime,jms,jme);
fmc_avg.bind(ims,ime,jms,jme);
wetness.bind(ims,ime,jms,jme);
int i;
int j;
int n;
int eco;
float ef;
float frac;
float beta_val;
float sum_frac;
auto ef_flm = arena.alloc<float>(1,8);
ef_flm(1) = __uint_as_float(0x41980000U);
ef_flm(2) = __uint_as_float(0x41166666U);
ef_flm(3) = __uint_as_float(0x4169999AU);
ef_flm(4) = __uint_as_float(0x4114CCCDU);
ef_flm(5) = __uint_as_float(0x412B3333U);
ef_flm(6) = __uint_as_float(0x4154CCCDU);
ef_flm(7) = __uint_as_float(0x412CCCCDU);
ef_flm(8) = __uint_as_float(0x413CCCCDU);
auto ef_sml = arena.alloc<float>(1,8);
ef_sml(1) = __uint_as_float(0x41E00000U);
ef_sml(2) = __uint_as_float(0x4216CCCDU);
ef_sml(3) = __uint_as_float(0x418CCCCDU);
ef_sml(4) = __uint_as_float(0x42126666U);
ef_sml(5) = __uint_as_float(0x4212CCCDU);
ef_sml(6) = __uint_as_float(0x4219999AU);
ef_sml(7) = __uint_as_float(0x41A5999AU);
ef_sml(8) = __uint_as_float(0x41BA6666U);
auto ef_east = arena.alloc<float>(1,3);
ef_east(1) = __uint_as_float(0x4114A3D7U);
ef_east(2) = __uint_as_float(0x413E8F5CU);
ef_east(3) = __uint_as_float(0x41033333U);
auto beta_kaiser = arena.alloc<float>(1,7);
beta_kaiser(1) = __uint_as_float(0x3F47AE14U);
beta_kaiser(2) = __uint_as_float(0x3E851EB8U);
beta_kaiser(3) = __uint_as_float(0x3E947AE1U);
beta_kaiser(4) = __uint_as_float(0x3E051EB8U);
beta_kaiser(5) = __uint_as_float(0x3EFAE148U);
beta_kaiser(6) = __uint_as_float(0x3FC66666U);
beta_kaiser(7) = __uint_as_float(0x40BBD70AU);
float beta_wooster = __uint_as_float(0x3EC28F5CU);
sd_assign(efs_map,0.0f,arena); // L64
for (j=jts;j<=jte;j+=1) { // L66
for (i=its;i<=ite;i+=1) { // L67
eco = eco_id(i,j); // L68
sum_frac = 0.0f; // L69
for (n=1;n<=20;n+=1) { // L70
sum_frac = sd_add(sum_frac,vegfrac(i,j,n)); // L71
loop3_continue: ; } loop3_exit: ; // L72
if ((sum_frac > 1.01f)) { // L73
; /* Framework diagnostic I/O; no field mutation. */ // L74
} // L75
for (n=1;n<=20;n+=1) { // L77
frac = vegfrac(i,j,n); // L78
{ auto sel1 = n; // L80
if (sel1 == 1) { // L81
ef = sd_add(sd_mul(0.75f,ef_flm(1)),sd_mul(0.25f,ef_sml(1))); // L82
if (((bb_beta == 1) && (((eco == 2) || (eco == 3)) || (eco == 4)))) { // L83
beta_val = beta_kaiser(6); // L84
} else { // L85
beta_val = beta_wooster; // L86
} // L87
} else if (sel1 == 2) { // L89
ef = sd_add(sd_mul(0.75f,ef_flm(1)),sd_mul(0.25f,ef_sml(1))); // L90
if (((bb_beta == 1) && (eco == 2))) { // L91
beta_val = beta_kaiser(6); // L92
} else if (((bb_beta == 1) && (eco == 5))) { // L93
beta_val = beta_kaiser(5); // L94
} else { // L95
beta_val = beta_wooster; // L96
} // L97
} else if (sel1 == 3) { // L99
ef = sd_add(sd_mul(0.80f,ef_flm(2)),sd_mul(0.20f,ef_sml(2))); // L100
if (((bb_beta == 1) && (eco == 2))) { // L101
beta_val = beta_kaiser(6); // L102
} else if (((bb_beta == 1) && (eco == 5))) { // L103
beta_val = beta_kaiser(5); // L104
} else { // L105
beta_val = beta_wooster; // L106
} // L107
} else if (sel1 == 4) { // L109
ef = sd_add(sd_mul(0.80f,ef_flm(2)),sd_mul(0.20f,ef_sml(2))); // L110
if (((bb_beta == 1) && (eco == 2))) { // L111
beta_val = beta_kaiser(6); // L112
} else if (((bb_beta == 1) && (eco == 5))) { // L113
beta_val = beta_kaiser(5); // L114
} else { // L115
beta_val = beta_wooster; // L116
} // L117
} else if (sel1 == 5) { // L119
ef = sd_add(sd_mul(0.85f,ef_flm(3)),sd_mul(0.15f,ef_sml(3))); // L120
if (((bb_beta == 1) && (eco == 2))) { // L121
beta_val = beta_kaiser(6); // L122
} else if (((bb_beta == 1) && (eco == 5))) { // L123
beta_val = beta_kaiser(5); // L124
} else { // L125
beta_val = beta_wooster; // L126
} // L127
} else if (sel1 == 6 || sel1 == 7) { // L129
ef = sd_add(sd_mul(0.95f,ef_flm(4)),sd_mul(0.05f,ef_sml(4))); // L130
if (((bb_beta == 1) && (((eco == 2) || (eco == 3)) || (eco == 4)))) { // L131
beta_val = beta_kaiser(2); // L132
} else { // L133
beta_val = beta_wooster; // L134
} // L135
} else if (sel1 == 8) { // L137
ef = sd_add(sd_mul(0.95f,ef_flm(5)),sd_mul(0.05f,ef_sml(5))); // L138
beta_val = beta_wooster; // L139
} else if (sel1 == 9) { // L141
ef = sd_add(sd_mul(0.95f,ef_flm(6)),sd_mul(0.05f,ef_sml(6))); // L142
beta_val = beta_wooster; // L143
} else if (sel1 == 10) { // L145
if ((eco == 10)) { // L146
ef = ef_east(1); // L147
} else { // L148
ef = ef_flm(6); // L149
} // L150
beta_val = beta_wooster; // L151
} else if (sel1 == 11) { // L153
ef = 18.9f; // L154
beta_val = beta_wooster; // L155
} else if (sel1 == 12) { // L157
if (((eco == 8) || (eco == 9))) { // L158
ef = ef_east(2); // L159
} else { // L160
ef = 8.2f; // L161
} // L162
beta_val = beta_wooster; // L163
} else if (sel1 == 14) { // L165
if (((eco == 8) || (eco == 9))) { // L166
ef = ef_east(3); // L167
} else { // L168
ef = 8.2f; // L169
} // L170
beta_val = beta_wooster; // L171
} else if (sel1 == 18) { // L173
ef = sd_add(sd_mul(0.7f,ef_flm(8)),sd_mul(0.3f,ef_sml(8))); // L174
beta_val = beta_wooster; // L175
} else if (sel1 == 19) { // L177
ef = sd_add(sd_mul(0.7f,ef_flm(7)),sd_mul(0.3f,ef_sml(7))); // L178
beta_val = beta_wooster; // L179
} else { // L181
ef = 0.0f; // L182
beta_val = 0.0f; // L183
} } // L184
efs_map(i,j) = sd_add(efs_map(i,j),sd_mul(sd_mul(frac,ef),beta_val)); // L186
loop4_continue: ; } loop4_exit: ; // L187
loop2_continue: ; } loop2_exit: ; // L188
loop1_continue: ; } loop1_exit: ; // L189
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_fire_emissions.F90:203-257 @1d6e370e; SMOKE @9652bbf
SDDEV void calculate_smoke_emissions(State &state,Arena &arena,float dt,int julday,int nlcat,FArray<float> efs_map,FArray<float> fre_avg,int ebb_dcycle,FArray<float> area,int nblocks,int ktau,int bb_input_prevh,FArray<float> ebu,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
efs_map.bind(ims,ime,jms,jme);
fre_avg.bind(ims,ime,jms,jme,1,nblocks);
area.bind(ims,ime,jms,jme);
ebu.bind(ims,ime,kms,kme,jms,jme);
int i;
int j;
int blk;
int hour_int;
float hour_tmp;
float fg_to_ug = __uint_as_float(0x49742400U);
float to_s = __uint_as_float(0x45610000U);
if ((ebb_dcycle == 2)) { // L232
hour_int = sd_floor(sd_div(sd_mul(ktau,dt),3600.f)); // L234
hour_tmp = (sd_mod(hour_int,24) / bb_input_prevh); // L236
blk = (sd_floor(hour_tmp) + 1); // L238
if ((blk < 1)) { blk = 1; } // L239
if ((blk > (24 / bb_input_prevh))) { blk = (24 / bb_input_prevh); } // L240
} else { // L241
blk = 1; // L243
} // L244
for (j=jts;j<=jte;j+=1) { // L247
for (i=its;i<=ite;i+=1) { // L248
if ((fre_avg(i,j,blk) > 0.0f)) { // L249
ebu(i,kts,j) = sd_div(sd_mul(sd_mul(fre_avg(i,j,blk),efs_map(i,j)),fg_to_ug),sd_mul(area(i,j),to_s)); // L250
} else { // L251
ebu(i,kts,j) = 0.0f; // L252
} // L253
loop2_continue: ; } loop2_exit: ; // L254
loop1_continue: ; } loop1_exit: ; // L255
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_fire_emissions.F90:266-438 @1d6e370e; SMOKE @9652bbf
SDDEV void diurnal_cycle(State &state,Arena &arena,float dtstep,FArray<float> dz8w,FArray<float> rho_phy,float pi,float ebb_min,FArray<float> chem,int num_chem,int julday,float gmt,FArray<float> xlat,FArray<float> xlong,FArray<float> fire_end_hr,FArray<float> peak_hr,float time_int,FArray<float> coef_bb_dc,FArray<float> fire_hist,FArray<float> hwp,FArray<float> hwp_avg,FArray<float> hwp_prev_day,FArray<float> vegfrac,FArray<int> eco_id,int nblocks,FArray<float> lu_nofire,FArray<float> lu_qfire,FArray<float> lu_sfire,FArray<float> swdown,int ebb_dcycle,FArray<float> ebu,FArray<int> fire_type,FArray<float> q_vap,bool add_fire_moist_flux,float plume_beta_qv,float hwp_alpha,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
dz8w.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
xlat.bind(ims,ime,jms,jme);
xlong.bind(ims,ime,jms,jme);
fire_end_hr.bind(ims,ime,jms,jme);
peak_hr.bind(ims,ime,jms,jme);
coef_bb_dc.bind(ims,ime,jms,jme);
fire_hist.bind(ims,ime,jms,jme);
hwp.bind(ims,ime,jms,jme);
hwp_avg.bind(ims,ime,jms,jme);
hwp_prev_day.bind(ims,ime,jms,jme);
vegfrac.bind(ims,ime,jms,jme,1,20);
eco_id.bind(ims,ime,jms,jme);
lu_nofire.bind(ims,ime,jms,jme);
lu_qfire.bind(ims,ime,jms,jme);
lu_sfire.bind(ims,ime,jms,jme);
swdown.bind(ims,ime,jms,jme);
ebu.bind(ims,ime,kms,kme,jms,jme);
fire_type.bind(ims,ime,jms,jme);
q_vap.bind(ims,ime,kms,kme,jms,jme);
int i;
int j;
int k;
int n;
int m;
int blk;
auto &icall = state.saved_diurnal_cycle_icall;
float conv_rho;
float conv;
float dm_smoke;
float dc_hwp;
float dc_gp;
float dc_fn;
int kfire_max = 51;
float ef_h2o = __uint_as_float(0x43A21C29U);
float coef_con;
float timeq;
float fire_age;
float age_hr;
float dt1;
float dt2;
float dtm;
float rinti = __uint_as_float(0x32BB615AU);
float ax2 = __uint_as_float(0x45548000U);
float const2 = __uint_as_float(0x43020000U);
float coef2 = __uint_as_float(0x3A8BDEEEU);
float cx2 = __uint_as_float(0x45E10000U);
float timeq_max = __uint_as_float(0x47A8C000U);
auto avg_fire_dur = arena.alloc<float>(1,5);
avg_fire_dur(1) = __uint_as_float(0x410E6666U);
avg_fire_dur(2) = __uint_as_float(0x40866666U);
avg_fire_dur(3) = __uint_as_float(0x40533333U);
avg_fire_dur(4) = __uint_as_float(0x40400000U);
avg_fire_dur(5) = __uint_as_float(0x3FB33333U);
auto sigma_fire_dur = arena.alloc<float>(1,5);
sigma_fire_dur(1) = __uint_as_float(0x410B3333U);
sigma_fire_dur(2) = __uint_as_float(0x40C00000U);
sigma_fire_dur(3) = __uint_as_float(0x40B00000U);
sigma_fire_dur(4) = __uint_as_float(0x40A66666U);
sigma_fire_dur(5) = __uint_as_float(0x4019999AU);
float avgx1 = __uint_as_float(0x00000000U);
float sigmx1 = __uint_as_float(0x400CCCCDU);
float c1 = __uint_as_float(0x3E4CCCCDU);
float avgx2 = __uint_as_float(0x3F000000U);
float sigmx2 = __uint_as_float(0x3F4CCCCDU);
float c2 = __uint_as_float(0x3F8CCCCDU);
timeq = sd_add(sd_mul(gmt,3600.f),float(time_int)); // L336
timeq = sd_mod(timeq,timeq_max); // L337
coef_con = sd_div(1.f,sd_pow(sd_mul(2.f,pi),0.5f)); // L338
sd_assign(fire_type,0,arena); // L341
sd_assign(lu_qfire,0.f,arena); // L342
sd_assign(lu_sfire,0.f,arena); // L343
sd_assign(lu_nofire,0.f,arena); // L344
if ((ebb_dcycle == 2)) { // L348
for (j=jts;j<=jte;j+=1) { // L349
for (i=its;i<=ite;i+=1) { // L350
if ((ebu(i,kts,j) < ebb_min)) { // L351
fire_type(i,j) = 0; // L352
lu_nofire(i,j) = 1.0f; // L353
} else { // L354
lu_qfire(i,j) = sd_add(sd_add(sd_add(sd_add(lu_nofire(i,j),vegfrac(i,12,j)),vegfrac(i,13,j)),vegfrac(i,14,j)),vegfrac(i,16,j)); // L357
lu_sfire(i,j) = sd_add(sd_add(sd_add(lu_qfire(i,j),vegfrac(i,8,j)),vegfrac(i,9,j)),vegfrac(i,10,j)); // L359
if ((lu_nofire(i,j) > 0.95f)) { // L360
fire_type(i,j) = 0; // L361
} else if ((lu_qfire(i,j) > 0.9f)) { // L362
fire_type(i,j) = 1; // L363
} else if ((eco_id(i,j) == 8)) { // L364
fire_type(i,j) = 2; // L365
if (((sd_mul(sd_mul(ebu(i,kts,j),3600),sd_div(1,0.416f)) >= 1.e6f) && (fire_end_hr(i,j) <= 8))) { // L367
fire_type(i,j) = 4; // L368
} // L369
} else if ((lu_sfire(i,j) > 0.8f)) { // L370
fire_type(i,j) = 3; // L371
} else { // L372
fire_type(i,j) = 4; // L373
} // L374
} // L375
loop2_continue: ; } loop2_exit: ; // L376
loop1_continue: ; } loop1_exit: ; // L377
} // L378
if ((ebb_dcycle == 2)) { // L381
for (j=jts;j<=jte;j+=1) { // L383
for (i=its;i<=ite;i+=1) { // L384
fire_age = sd_max(0.01f,sd_add(sd_div(time_int,3600.f),sd_sub(fire_end_hr(i,j),1.0f))); // L385
{ auto sel1 = fire_type(i,j); // L387
if (sel1 == 1) { // L388
coef_bb_dc(i,j) = sd_mul(sd_div(c1,sd_mul(sigmx1,fire_age)),sd_exp(sd_div(sd_neg(sd_ipow(sd_sub(sd_log(fire_age),avgx1),2)),sd_mul(2.f,sd_ipow(sigmx1,2))))); // L392
if ((state.mpas_smoke_config.dbg_opt && (time_int < 5000.f))) { // L394
; /* Framework diagnostic I/O; no field mutation. */ // L395
; /* Framework diagnostic I/O; no field mutation. */ // L396
} // L397
} else if (sel1 == 2) { // L399
coef_bb_dc(i,j) = sd_mul(sd_div(c2,sd_mul(sigmx2,fire_age)),sd_exp(sd_div(sd_neg(sd_ipow(sd_sub(sd_log(fire_age),avgx2),2)),sd_mul(2.f,sd_ipow(sigmx2,2))))); // L402
if ((state.mpas_smoke_config.dbg_opt && (time_int < 5000.f))) { // L404
; /* Framework diagnostic I/O; no field mutation. */ // L405
; /* Framework diagnostic I/O; no field mutation. */ // L406
} // L407
} else if (sel1 == 3 || sel1 == 4) { // L409
if ((((swdown(i,j) < .1f) && (fire_age > 6.f)) && (fire_hist(i,j) > 0.75f))) { // L410
fire_hist(i,j) = 0.75f; // L411
} // L412
if ((((swdown(i,j) < .1f) && (fire_age > 18.f)) && (fire_hist(i,j) > 0.5f))) { // L413
fire_hist(i,j) = 0.5f; // L414
} // L415
if ((((swdown(i,j) < .1f) && (fire_age > 36.f)) && (fire_hist(i,j) > 0.25f))) { // L416
fire_hist(i,j) = 0.25f; // L417
} // L418
dc_hwp = sd_add(sd_div(sd_mul(hwp_alpha,hwp(i,j)),sd_max(10.f,hwp_prev_day(i,j))),sd_mul(sd_sub(1,hwp_alpha),sd_div(hwp(i,j),sd_max(10.f,hwp_avg(i,j))))); // L420
dc_hwp = sd_max(0.f,dc_hwp); // L421
dc_hwp = sd_min(20.f,dc_hwp); // L422
coef_bb_dc(i,j) = sd_mul(fire_hist(i,j),dc_hwp); // L425
if ((state.mpas_smoke_config.dbg_opt && (time_int < 5000.f))) { // L427
; /* Framework diagnostic I/O; no field mutation. */ // L428
; /* Framework diagnostic I/O; no field mutation. */ // L429
} // L430
} else { // L432
} } // L433
loop4_continue: ; } loop4_exit: ; // L435
loop3_continue: ; } loop3_exit: ; // L436
} // L437
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_add_emiss_burn.F90:10-107 @1d6e370e; SMOKE @9652bbf
SDDEV void add_emis_burn(State &state,Arena &arena,float dtstep,FArray<float> dz8w,FArray<float> rho_phy,float pi,float ebb_min,FArray<float> chem,int num_chem,int julday,float gmt,FArray<float> xlat,FArray<float> xlong,FArray<float> fire_end_hr,FArray<float> peak_hr,float time_int,FArray<float> coef_bb_dc,FArray<float> fire_hist,FArray<float> hwp,FArray<float> hwp_prevd,FArray<float> swdown,int ebb_dcycle,FArray<float> ebu,FArray<float> ebu_coarse,FArray<float> ebu_ch4,FArray<int> fire_type,FArray<float> q_vap,bool add_fire_moist_flux,float sc_factor,FArray<float> aod3d,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
dz8w.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
xlat.bind(ims,ime,jms,jme);
xlong.bind(ims,ime,jms,jme);
fire_end_hr.bind(ims,ime,jms,jme);
peak_hr.bind(ims,ime,jms,jme);
coef_bb_dc.bind(ims,ime,jms,jme);
fire_hist.bind(ims,ime,jms,jme);
hwp.bind(ims,ime,jms,jme);
hwp_prevd.bind(ims,ime,jms,jme);
swdown.bind(ims,ime,jms,jme);
ebu.bind(ims,ime,kms,kme,jms,jme);
ebu_coarse.bind(ims,ime,kms,kme,jms,jme);
ebu_ch4.bind(ims,ime,kms,kme,jms,jme);
fire_type.bind(ims,ime,jms,jme);
q_vap.bind(ims,ime,kms,kme,jms,jme);
aod3d.bind(ims,ime,kms,kme,jms,jme);
int i;
int j;
int k;
int n;
int m;
auto &icall = state.saved_add_emis_burn_icall;
float conv_gas;
float conv_rho;
float conv;
float dm_smoke;
float dm_smoke_coarse;
float dm_ch4;
float dc_hwp;
float dc_gp;
float dc_fn;
float ext2;
int kfire_max = 51;
float ef_h2o = __uint_as_float(0x43A21C29U);
float sc_me = __uint_as_float(0x40800000U);
float ab_me = __uint_as_float(0x3F000000U);
if ((sd_mod(int(time_int),1800) == 0)) { // L58
icall = 0; // L59
} // L60
ext2 = sd_add(sc_me,ab_me); // L62
for (j=jts;j<=jte;j+=1) { // L63
for (i=its;i<=ite;i+=1) { // L64
for (k=kts;k<=kfire_max;k+=1) { // L65
if ((ebu(i,k,j) < ebb_min)) { goto loop3_continue; } // L66
if ((ebb_dcycle == 1)) { // L68
conv = sd_div(dtstep,sd_mul(rho_phy(i,k,j),dz8w(i,k,j))); // L69
conv_gas = sd_div(sd_mul(dtstep,0.02897f),sd_mul(rho_phy(i,k,j),dz8w(i,k,j))); // L70
} else if ((ebb_dcycle == 2)) { // L71
conv = sd_div(sd_mul(coef_bb_dc(i,j),dtstep),sd_mul(rho_phy(i,k,j),dz8w(i,k,j))); // L72
} // L73
if ((state.mpas_smoke_init.p_smoke_fine > 0)) { // L76
dm_smoke = sd_mul(conv,ebu(i,k,j)); // L77
chem(i,k,j,state.mpas_smoke_init.p_smoke_fine) = sd_add(chem(i,k,j,state.mpas_smoke_init.p_smoke_fine),dm_smoke); // L78
chem(i,k,j,state.mpas_smoke_init.p_smoke_fine) = sd_min(sd_max(chem(i,k,j,state.mpas_smoke_init.p_smoke_fine),state.mpas_smoke_config.epsilc),5.e+3f); // L79
aod3d(i,k,j) = sd_mul(sd_mul(sd_mul(sd_mul(1.e-6f,ext2),chem(i,k,j,state.mpas_smoke_init.p_smoke_fine)),rho_phy(i,k,j)),dz8w(i,k,j)); // L80
} // L81
if ((state.mpas_smoke_init.p_smoke_coarse > 0)) { // L83
dm_smoke_coarse = sd_mul(conv,ebu_coarse(i,k,j)); // L84
chem(i,k,j,state.mpas_smoke_init.p_smoke_coarse) = sd_add(chem(i,k,j,state.mpas_smoke_init.p_smoke_coarse),dm_smoke_coarse); // L85
chem(i,k,j,state.mpas_smoke_init.p_smoke_coarse) = sd_min(sd_max(chem(i,k,j,state.mpas_smoke_init.p_smoke_coarse),state.mpas_smoke_config.epsilc),5.e+3f); // L86
} // L87
if ((state.mpas_smoke_init.p_ch4 > 0)) { // L88
dm_ch4 = sd_mul(conv_gas,ebu_ch4(i,k,j)); // L89
chem(i,k,j,state.mpas_smoke_init.p_ch4) = sd_add(chem(i,k,j,state.mpas_smoke_init.p_ch4),dm_ch4); // L90
chem(i,k,j,state.mpas_smoke_init.p_ch4) = sd_min(sd_max(chem(i,k,j,state.mpas_smoke_init.p_ch4),state.mpas_smoke_config.epsilc),5.e+3f); // L91
} // L92
if (add_fire_moist_flux) { // L95
q_vap(i,k,j) = sd_add(q_vap(i,k,j),sd_mul(sd_mul(dm_smoke,ef_h2o),1.e-9f)); // L96
q_vap(i,k,j) = sd_min(sd_max(q_vap(i,k,j),0.f),1.e+3f); // L97
} // L98
if (((state.mpas_smoke_config.dbg_opt && ((k == kts) || (k == kfire_max))) && (icall <= state.mpas_smoke_config.n_dbg_lines))) { // L100
} // L101
loop3_continue: ; } loop3_exit: ; // L102
icall = (icall + 1); // L103
loop2_continue: ; } loop2_exit: ; // L104
loop1_continue: ; } loop1_exit: ; // L105
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:11-115 @1d6e370e; SMOKE @9652bbf
SDDEV void plumerise(State &state,Arena &arena,int m1,int m2,int m3,int ia,int iz,int ja,int jz,FArray<float> up,FArray<float> vp,FArray<float> wp,FArray<float> theta,FArray<float> pp,FArray<float> dn0,FArray<float> rv,FArray<float> zt_rams,FArray<float> zm_rams,int wind_eff_opt,float frp_inst,int & k1,int & k2,bool dbg_opt,float g,float cp,float rgas,float cpor,FStr & errmsg,int & errflg,int icall,float lat,float f_long,float curr_secs,float alpha,float frp_min) {
ArenaScope scope(arena);
up.bind(1,m1,1,m2,1,m3);
vp.bind(1,m1,1,m2,1,m3);
wp.bind(1,m1,1,m2,1,m3);
theta.bind(1,m1,1,m2,1,m3);
pp.bind(1,m1,1,m2,1,m3);
dn0.bind(1,m1,1,m2,1,m3);
rv.bind(1,m1,1,m2,1,m3);
zt_rams.bind(1,m1);
zm_rams.bind(1,m1);
int ng;
int ibcon;
int mynum;
int i;
int j;
int k;
int imm;
int ixx;
int ispc;
int kmt;
float burnt_area;
float dzi;
float frp;
auto ztopmax = arena.alloc<float>(1,2);
float q_smold_kgm2;
int wind_eff;
Plume* coms;
wind_eff = wind_eff_opt; // L49
coms = get_thread_coms(state,arena); // L53
j = 1; // L56
i = 1; // L57
for (k=1;k<=m1;k+=1) { // L64
coms->ucon(k) = up(k,i,j); // L65
coms->vcon(k) = vp(k,i,j); // L66
coms->thtcon(k) = theta(k,i,j); // L68
coms->picon(k) = pp(k,i,j); // L69
coms->rvcon(k) = rv(k,i,j); // L73
coms->zcon(k) = zt_rams(k); // L74
coms->zzcon(k) = zm_rams(k); // L75
loop1_continue: ; } loop1_exit: ; // L76
get_env_condition(state,arena,coms,1,m1,kmt,wind_eff,g,cp,rgas,cpor,errmsg,errflg); // L78
if ((errflg != 0)) { return; } // L79
for (imm=1;imm<=2;imm+=1) { // L89
if ((imm == 1)) { // L90
burnt_area = sd_mul(sd_mul(0.7f,0.0006f),frp_inst); // L91
} else if ((imm == 2)) { // L92
burnt_area = sd_mul(sd_mul(1.3f,0.0006f),frp_inst); // L93
} // L94
burnt_area = sd_max(1.0e4f,burnt_area); // L95
if ((frp_inst < frp_min)) { // L97
k1 = 1; // L98
k2 = 2; // L99
return; // L100
} // L101
get_fire_properties(state,arena,coms,imm,burnt_area,frp_inst,errmsg,errflg); // L104
if ((errflg != 0)) { return; } // L105
makeplume(state,arena,coms,kmt,ztopmax(imm),ixx,imm,alpha); // L108
lp_minmax_continue: ; } lp_minmax_exit: ; // L110
set_flam_vert(state,arena,ztopmax,k1,k2,state.module_zero_plumegen_coms.nkp,coms->zzcon); // L113
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:118-198 @1d6e370e; SMOKE @9652bbf
SDDEV void get_env_condition(State &state,Arena &arena,Plume* coms,int k1,int k2,int & kmt,int wind_eff,float g,float cp,float rgas,float cpor,FStr & errmsg,int & errflg) {
ArenaScope scope(arena);
int k;
int kcon;
int klcl;
int nk;
int nkmid;
int i;
float p1000mb = __uint_as_float(0x47C35000U);
float p00 = __uint_as_float(0x47C35000U);
float znz;
float themax;
float tlll;
float plll;
float rlll;
float zlll;
float dzdd;
float dzlll;
float tlcl;
float plcl;
float dzlcl;
float dummy;
if ((!coms->initialized)) { // L134
set_grid(state,arena,coms); // L136
} // L139
znz = coms->zcon(k2); // L141
errflg = 1; // L142
for (k=state.module_zero_plumegen_coms.nkp;k>=1;k+=(-1)) { // L143
if ((coms->zt(k) < znz)) { // L144
errflg = 0; // L145
goto loop1_exit; // L146
} // L147
loop1_continue: ; } loop1_exit: ; // L148
if ((errflg != 0)) { // L149
errmsg = FStr(" envir stop 12"); // L150
return; // L151
} // L152
kmt = sd_min(k,(state.module_zero_plumegen_coms.nkp - 1)); // L154
nk = ((k2 - k1) + 1); // L156
htint(state,arena,nk,coms->ucon,coms->zcon,kmt,coms->upe,coms->zt,errmsg,errflg); // L159
if ((errflg != 0)) { return; } // L160
htint(state,arena,nk,coms->vcon,coms->zcon,kmt,coms->vpe,coms->zt,errmsg,errflg); // L161
if ((errflg != 0)) { return; } // L162
htint(state,arena,nk,coms->thtcon,coms->zcon,kmt,coms->the,coms->zt,errmsg,errflg); // L163
if ((errflg != 0)) { return; } // L164
htint(state,arena,nk,coms->rvcon,coms->zcon,kmt,coms->qvenv,coms->zt,errmsg,errflg); // L165
if ((errflg != 0)) { return; } // L166
for (k=1;k<=kmt;k+=1) { // L167
coms->qvenv(k) = sd_max(coms->qvenv(k),1e-8f); // L168
loop2_continue: ; } loop2_exit: ; // L169
coms->pke(1) = coms->picon(1); // L171
for (k=1;k<=kmt;k+=1) { // L172
coms->thve(k) = sd_mul(coms->the(k),sd_add(1.f,sd_mul(.61f,coms->qvenv(k)))); // L173
loop3_continue: ; } loop3_exit: ; // L174
for (k=2;k<=kmt;k+=1) { // L175
coms->pke(k) = sd_sub(coms->pke((k - 1)),sd_div(sd_mul(sd_mul(g,2.f),sd_sub(coms->zt(k),coms->zt((k - 1)))),sd_add(coms->thve(k),coms->thve((k - 1))))); // L176
loop4_continue: ; } loop4_exit: ; // L178
for (k=1;k<=kmt;k+=1) { // L179
coms->te(k) = sd_div(sd_mul(coms->the(k),coms->pke(k)),cp); // L180
coms->pe(k) = sd_mul(sd_pow(sd_div(coms->pke(k),cp),cpor),p00); // L181
coms->dne(k) = sd_div(coms->pe(k),sd_mul(sd_mul(rgas,coms->te(k)),sd_add(1.f,sd_mul(.61f,coms->qvenv(k))))); // L182
coms->vel_e(k) = sd_sqrt(sd_add(sd_ipow(coms->upe(k),2),sd_ipow(coms->vpe(k),2))); // L185
loop5_continue: ; } loop5_exit: ; // L187
if ((wind_eff < 1)) { sd_assign(coms->vel_e.section(Slice(1,kmt)),0.f,arena); } // L190
for (k=1;k<=kmt;k+=1) { // L193
coms->pe(k) = sd_mul(coms->pe(k),1.e-3f); // L194
loop6_continue: ; } loop6_exit: ; // L195
return; // L197
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:202-236 @1d6e370e; SMOKE @9652bbf
SDDEV void set_grid(State &state,Arena &arena,Plume* coms) {
ArenaScope scope(arena);
int k;
int mzp;
coms->dz = 100.f; // L208
mzp = state.module_zero_plumegen_coms.nkp; // L210
coms->zt(1) = coms->zsurf; // L211
coms->zm(1) = coms->zsurf; // L212
coms->zt(2) = sd_add(coms->zt(1),sd_mul(0.5f,coms->dz)); // L213
coms->zm(2) = sd_add(coms->zm(1),coms->dz); // L214
for (k=3;k<=mzp;k+=1) { // L215
coms->zt(k) = sd_add(coms->zt((k - 1)),coms->dz); // L216
coms->zm(k) = sd_add(coms->zm((k - 1)),coms->dz); // L217
loop1_continue: ; } loop1_exit: ; // L218
for (k=1;k<=(mzp - 1);k+=1) { // L221
coms->dzm(k) = sd_div(1.f,sd_sub(coms->zt((k + 1)),coms->zt(k))); // L222
loop2_continue: ; } loop2_exit: ; // L223
coms->dzm(mzp) = coms->dzm((mzp - 1)); // L224
for (k=2;k<=mzp;k+=1) { // L226
coms->dzt(k) = sd_div(1.f,sd_sub(coms->zm(k),coms->zm((k - 1)))); // L227
loop3_continue: ; } loop3_exit: ; // L228
coms->dzt(1) = sd_div(sd_mul(coms->dzt(2),coms->dzt(2)),coms->dzt(3)); // L229
coms->initialized = true; // L231
return; // L235
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:239-275 @1d6e370e; SMOKE @9652bbf
SDDEV void set_flam_vert(State &state,Arena &arena,FArray<float> ztopmax,int & k1,int & k2,int nkp,FArray<float> zzcon) {
ArenaScope scope(arena);
ztopmax.bind(1,2);
zzcon.bind(1,nkp);
int imm;
int k;
auto k_lim = arena.alloc<int>(1,2);
for (imm=1;imm<=2;imm+=1) { // L258
for (k=1;k<=(nkp - 1);k+=1) { // L261
if ((zzcon(k) > ztopmax(imm))) { goto loop2_exit; } // L262
loop2_continue: ; } loop2_exit: ; // L263
k_lim(imm) = k; // L264
loop1_continue: ; } loop1_exit: ; // L265
k1 = sd_min(sd_max(4,k_lim(1)),51); // L266
k2 = sd_min(51,k_lim(2)); // L267
if ((k2 <= k1)) { // L269
k2 = (k1 + 1); // L272
} // L273
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:278-382 @1d6e370e; SMOKE @9652bbf
SDDEV void get_fire_properties(State &state,Arena &arena,Plume* coms,int imm,float burnt_area,float frp,FStr & errmsg,int & errflg) {
ArenaScope scope(arena);
int moist;
int i;
int icount;
float bfract;
float effload;
float heat;
float hinc;
float heat_fluxw;
int use_last = 1;
float beta = __uint_as_float(0x3F6147AEU);
coms->area = burnt_area; // L292
heat_fluxw = sd_div(sd_mul(beta,sd_div(frp,coms->area)),0.55f); // L296
coms->mdur = 53; // L306
coms->bload = 10.f; // L307
moist = 10; // L308
coms->maxtime = (coms->mdur + 2); // L309
heat = 19.3e6f; // L312
coms->maxtime = (coms->maxtime * 60); // L320
coms->rsurf = sd_sqrt(sd_div(coms->area,3.14159f)); // L322
coms->fmoist = sd_div(moist,100.f); // L324
for (i=1;i<=state.module_zero_plumegen_coms.ntime;i+=1) { // L331
coms->heating(i) = 0.0001f; // L332
loop1_continue: ; } loop1_exit: ; // L333
coms->tdur = sd_mul(coms->mdur,60.f); // L335
bfract = 1.f; // L337
effload = sd_mul(coms->bload,bfract); // L339
icount = 1; // L343
if ((coms->mdur > state.module_zero_plumegen_coms.ntime)) { // L345
errmsg = FStr("Increase time duration (ntime) in min - see file \"module_zero_plumegen_coms.F90\""); // L346
errflg = 1; // L347
return; // L348
} // L349
while ((icount <= coms->mdur)) { // L351
coms->heating(icount) = sd_mul(heat_fluxw,0.55f); // L355
icount = (icount + 1); // L356
loop2_continue: ; } loop2_exit: ; // L357
if ((use_last != 1)) { // L359
hinc = sd_div(coms->heating(1),4.f); // L361
coms->heating(1) = 0.1f; // L362
coms->heating(2) = hinc; // L363
coms->heating(3) = sd_mul(2.f,hinc); // L364
coms->heating(4) = sd_mul(3.f,hinc); // L365
} else { // L366
hinc = sd_div(coms->heating(1),4.f); // L367
if ((imm == 1)) { // L368
coms->heating(1) = 0.1f; // L370
coms->heating(2) = hinc; // L371
coms->heating(3) = sd_mul(2.f,hinc); // L372
coms->heating(4) = sd_mul(3.f,hinc); // L373
} else { // L374
coms->heating(2) = sd_add(coms->heating(1),hinc); // L375
coms->heating(3) = sd_add(coms->heating(2),hinc); // L376
coms->heating(4) = sd_add(coms->heating(3),hinc); // L377
} // L378
} // L379
return; // L381
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:385-690 @1d6e370e; SMOKE @9652bbf
SDDEV void makeplume(State &state,Arena &arena,Plume* coms,int kmt,float & ztopmax,int ixx,int imm,float alpha) {
ArenaScope scope(arena);
FStr varn;
int izprint;
int iconv;
int itime;
int k;
int kk;
int kkmax;
int deltak;
int ilastprint;
int nrectotal;
int i_micro;
int n_sub_step;
float vc = __uint_as_float(0x40A36FB7U);
float g = __uint_as_float(0x411CED67U);
float r = __uint_as_float(0x438F851FU);
float cp = __uint_as_float(0x447B0000U);
float eps = __uint_as_float(0x3F1F3B64U);
float tmelt = __uint_as_float(0x4388A666U);
float heatsubl = __uint_as_float(0x4A2CF940U);
float heatfus = __uint_as_float(0x48A31600U);
float heatcond = __uint_as_float(0x4A18A620U);
float tfreeze = __uint_as_float(0x4386A666U);
float wmax;
float rmaxtime;
float es;
float esat;
float heat;
float dt_save;
FStr cixx;
auto &delz_thresold = state.saved_makeplume_delz_thresold;
int dtknt;
coms->tstpf = 2.0f; // L472
coms->viscosity = 500.f; // L473
nrectotal = 150; // L475
dtknt = 0; // L476
coms->mintime = 1; // L479
ztopmax = 0.f; // L480
coms->ztop = 0.f; // L481
coms->time = 0.f; // L482
coms->dt = 1.f; // L483
wmax = 1.f; // L484
kkmax = 10; // L485
deltak = 20; // L486
ilastprint = 0; // L487
coms->l = 1; // L488
initial(state,arena,coms,kmt,alpha); // L491
izprint = 0; // L494
rmaxtime = float(coms->maxtime); // L507
while ((coms->time <= rmaxtime)) { // L511
coms->nm1 = sd_min(kmt,(kkmax + deltak)); // L516
coms->dt = sd_max(sd_min(sd_div(sd_sub(coms->zm(2),coms->zm(1)),sd_mul(wmax,coms->tstpf)),5.f),0.01f); // L521
dtknt = (dtknt + 1); // L522
coms->time = sd_add(coms->dt,coms->time); // L526
coms->mintime = (1 + (int(coms->time) / 60)); // L528
wmax = 1.f; // L529
tend0_plumerise(state,arena,coms); // L533
coms->l = 1; // L536
lbound(state,arena,coms,alpha); // L537
vel_advectc_plumerise(state,arena,coms->nm1,coms->wc,coms->wt,coms->rho,coms->dzm); // L542
scl_advectc_plumerise(state,arena,coms,FStr("SC"),coms->nm1); // L545
scl_misc(state,arena,coms,coms->nm1,alpha); // L551
scl_dyn_entrain(state,arena,coms->nm1,state.module_zero_plumegen_coms.nkp,coms->wbar,coms->w,coms->adiabat,alpha,coms->radius,coms->tt,coms->t,coms->te,coms->qvt,coms->qv,coms->qvenv,coms->qct,coms->qc,coms->qht,coms->qh,coms->qit,coms->qi,coms->vel_e,coms->vel_p,coms->vel_t,coms->rad_p,coms->rad_t); // L554
damp_grav_wave(state,arena,1,coms->nm1,deltak,coms->dt,coms->zt,coms->zm,coms->w,coms->t,coms->tt,coms->qv,coms->qh,coms->qi,coms->qc,coms->te,coms->pe,coms->qvenv); // L558
dt_save = coms->dt; // L562
n_sub_step = 3; // L563
coms->dt = sd_div(coms->dt,float(n_sub_step)); // L564
for (i_micro=1;i_micro<=n_sub_step;i_micro+=1) { // L566
fallpart(state,arena,coms,coms->nm1); // L568
coms->l = 2; // L570
while ((coms->l <= (coms->nm1 - 1))) { // L571
coms->wbar = sd_mul(sd_add(coms->w(coms->l),coms->w((coms->l - 1))),0.5f); // L573
es = esat_pr(state,arena,coms->t(coms->l)); // L574
coms->qsat(coms->l) = sd_div(sd_mul(eps,es),sd_sub(coms->pe(coms->l),es)); // L575
coms->est(coms->l) = es; // L576
coms->rho(coms->l) = sd_div(sd_mul(coms->pe(coms->l),3483.8f),coms->t(coms->l)); // L581
if ((coms->w(coms->l) >= 0.f)) { // L585
coms->dqsdz = sd_div(sd_sub(coms->qsat((coms->l + 1)),coms->qsat((coms->l - 1))),sd_sub(coms->zt((coms->l + 1)),coms->zt((coms->l - 1)))); // L586
} else { // L587
coms->dqsdz = sd_div(sd_sub(coms->qsat((coms->l + 1)),coms->qsat((coms->l - 1))),sd_sub(coms->zt((coms->l + 1)),coms->zt((coms->l - 1)))); // L588
} // L589
waterbal(state,arena,coms); // L591
coms->l = (coms->l + 1); // L592
loop3_continue: ; } loop3_exit: ; // L593
loop2_continue: ; } loop2_exit: ; // L594
coms->dt = dt_save; // L595
label_101: ; ; // L597
visc_w(state,arena,coms,coms->nm1,deltak,kmt); // L600
update_plumerise(state,arena,coms,coms->nm1,FStr("S")); // L603
hadvance_plumerise(state,arena,1,coms->nm1,coms->dt,coms->wc,coms->wt,coms->w,coms->mintime); // L605
buoyancy_plumerise(state,arena,coms->nm1,coms->t,coms->te,coms->qv,coms->qvenv,coms->qh,coms->qi,coms->qc,coms->wt,coms->scr1); // L608
entrainment(state,arena,coms,coms->nm1,coms->w,coms->wt,coms->radius,alpha); // L611
update_plumerise(state,arena,coms,coms->nm1,FStr("W")); // L614
hadvance_plumerise(state,arena,2,coms->nm1,coms->dt,coms->wc,coms->wt,coms->w,coms->mintime); // L616
for (k=2;k<=coms->nm1;k+=1) { // L620
es = esat_pr(state,arena,coms->t(k)); // L624
coms->qsat(k) = sd_div(sd_mul(eps,es),sd_sub(coms->pe(k),es)); // L625
coms->est(k) = es; // L626
coms->txs(k) = sd_sub(coms->t(k),coms->te(k)); // L627
coms->rho(k) = sd_div(sd_mul(coms->pe(k),3483.8f),coms->t(k)); // L632
if ((sd_abs(coms->wc(k)) > wmax)) { wmax = sd_abs(coms->wc(k)); } // L634
loop4_continue: ; } loop4_exit: ; // L635
damp_grav_wave(state,arena,2,coms->nm1,deltak,coms->dt,coms->zt,coms->zm,coms->w,coms->t,coms->tt,coms->qv,coms->qh,coms->qi,coms->qc,coms->te,coms->pe,coms->qvenv); // L638
for (k=2;k<=coms->nm1;k+=1) { // L641
coms->radius(k) = coms->rad_p(k); // L642
loop5_continue: ; } loop5_exit: ; // L643
kk = 1; // L645
while ((coms->w(kk) > 1.f)) { // L646
kk = (kk + 1); // L647
coms->ztop = coms->zm(kk); // L648
loop6_continue: ; } loop6_exit: ; // L650
coms->ztop_(coms->mintime) = coms->ztop; // L652
ztopmax = sd_max(ztopmax,coms->ztop); // L653
kkmax = sd_max(kk,kkmax); // L654
if ((coms->mintime > 10)) { // L659
if ((sd_abs(sd_sub(coms->ztop_(coms->mintime),coms->ztop_((coms->mintime - 10)))) < delz_thresold)) { // L662
goto loop1_exit; // L667
} // L668
} // L669
loop1_continue: ; } loop1_exit: ; // L677
return; // L689
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:693-726 @1d6e370e; SMOKE @9652bbf
SDDEV void burn(State &state,Arena &arena,Plume* coms,float & eflux,float & water) {
ArenaScope scope(arena);
float heat = __uint_as_float(0x4B933F50U);
if ((coms->time > coms->tdur)) { // L711
eflux = 0.000001f; // L712
water = 0.f; // L713
return; // L714
} else { // L715
eflux = coms->heating(coms->mintime); // L717
water = sd_div(sd_mul(sd_mul(eflux,sd_div(coms->dt,heat)),sd_add(0.5f,coms->fmoist)),0.55f); // L719
water = sd_mul(water,1000.f); // L720
} // L723
return; // L725
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:729-830 @1d6e370e; SMOKE @9652bbf
SDDEV void lbound(State &state,Arena &arena,Plume* coms,float alpha) {
ArenaScope scope(arena);
float g = __uint_as_float(0x411CED67U);
float r = __uint_as_float(0x438F851FU);
float cp = __uint_as_float(0x447B2666U);
float eps = __uint_as_float(0x3F1F3B64U);
float tmelt = __uint_as_float(0x4388A666U);
float tfreeze = __uint_as_float(0x4386A666U);
float pi = __uint_as_float(0x40490FD0U);
float e1 = __uint_as_float(0x3EAAAAABU);
float e2 = __uint_as_float(0x3FD55555U);
float es;
float esat;
float eflux;
float water;
float pres;
float c1;
float c2;
float f;
float zv;
float denscor;
float xwater;
coms->qh(1) = coms->qh(2); // L753
coms->qi(1) = coms->qi(2); // L754
coms->qc(1) = 0.f; // L755
burn(state,arena,coms,eflux,water); // L758
pres = sd_mul(coms->pe(1),1000.f); // L762
c1 = sd_div(5.f,sd_mul(6.f,alpha)); // L764
c2 = sd_mul(0.9f,alpha); // L766
f = sd_div(eflux,sd_mul(sd_mul(pres,cp),pi)); // L768
f = sd_mul(sd_mul(sd_mul(g,r),f),coms->area); // L770
zv = sd_mul(c1,coms->rsurf); // L772
coms->w(1) = sd_div(sd_mul(c1,sd_pow(sd_mul(c2,f),e1)),sd_pow(zv,e1)); // L774
denscor = sd_div(sd_div(sd_div(sd_mul(c1,f),g),sd_pow(sd_mul(c2,f),e1)),sd_pow(zv,e2)); // L776
coms->t(1) = sd_div(coms->te(1),sd_sub(1.f,denscor)); // L778
coms->wc(1) = coms->w(1); // L781
coms->vel_p(1) = 0.f; // L782
coms->rad_p(1) = coms->rsurf; // L783
coms->vth(1) = sd_neg(4.f); // L797
coms->vti(1) = sd_neg(3.f); // L798
coms->txs(1) = sd_sub(coms->t(1),coms->te(1)); // L799
coms->visc(1) = coms->viscosity; // L801
coms->rho(1) = sd_div(sd_mul(3483.8f,coms->pe(1)),coms->t(1)); // L807
xwater = sd_div(water,sd_max(1e-20f,sd_mul(sd_mul(coms->w(1),coms->dt),coms->rho(1)))); // L809
coms->qv(1) = sd_add(xwater,coms->qvenv(1)); // L811
es = esat_pr(state,arena,coms->t(1)); // L817
coms->est(1) = es; // L819
coms->qsat(1) = sd_div(sd_mul(eps,es),sd_max(1e-20f,sd_sub(coms->pe(1),es))); // L820
if ((coms->qv(1) > coms->qsat(1))) { // L822
coms->qc(1) = sd_add(sd_sub(coms->qv(1),coms->qsat(1)),coms->qc(1)); // L823
coms->qv(1) = coms->qsat(1); // L824
} // L825
waterbal(state,arena,coms); // L827
return; // L829
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:833-901 @1d6e370e; SMOKE @9652bbf
SDDEV void initial(State &state,Arena &arena,Plume* coms,int kmt,float alpha) {
ArenaScope scope(arena);
float tfreeze = __uint_as_float(0x4386A666U);
int isub;
int k;
int n1;
int n2;
int n3;
int lbuoy;
int itmp;
int isubm1;
float xn1;
float xi;
float es;
float esat;
coms->n = kmt; // L844
for (k=1;k<=coms->n;k+=1) { // L846
coms->txs(k) = 0.0f; // L847
coms->w(k) = 0.0f; // L848
coms->t(k) = coms->te(k); // L849
coms->wc(k) = 0.0f; // L850
coms->wt(k) = 0.0f; // L851
coms->qv(k) = coms->qvenv(k); // L852
coms->vth(k) = 0.f; // L853
coms->vti(k) = 0.f; // L854
coms->qh(k) = 0.f; // L855
coms->qi(k) = 0.f; // L856
coms->qc(k) = 0.f; // L857
es = esat_pr(state,arena,coms->t(k)); // L861
coms->est(k) = es; // L862
coms->qsat(k) = sd_div(sd_mul(.622f,es),sd_sub(coms->pe(k),es)); // L863
coms->rho(k) = sd_div(sd_mul(3483.8f,coms->pe(k)),coms->t(k)); // L868
coms->vel_p(k) = 0.f; // L869
coms->rad_p(k) = 0.f; // L870
loop1_continue: ; } loop1_exit: ; // L871
coms->radius(1) = coms->rsurf; // L874
for (k=2;k<=coms->n;k+=1) { // L875
coms->radius(k) = sd_add(coms->radius((k - 1)),sd_mul(sd_mul(sd_div(6.f,5.f),alpha),sd_sub(coms->zt(k),coms->zt((k - 1))))); // L876
loop2_continue: ; } loop2_exit: ; // L877
coms->radius(1) = coms->rsurf; // L879
coms->rad_p(1) = coms->rsurf; // L880
for (k=2;k<=coms->n;k+=1) { // L881
coms->radius(k) = sd_add(coms->radius((k - 1)),sd_mul(sd_mul(sd_div(6.f,5.f),alpha),sd_sub(coms->zt(k),coms->zt((k - 1))))); // L882
coms->rad_p(k) = coms->radius(k); // L883
loop3_continue: ; } loop3_exit: ; // L884
coms->visc(1) = coms->viscosity; // L887
for (k=2;k<=coms->n;k+=1) { // L888
coms->visc(k) = sd_max(1.e-3f,sd_sub(coms->visc((k - 1)),sd_div(sd_mul(1.f,coms->viscosity),float(state.module_zero_plumegen_coms.nkp)))); // L890
loop4_continue: ; } loop4_exit: ; // L891
lbound(state,arena,coms,alpha); // L898
return; // L900
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:904-922 @1d6e370e; SMOKE @9652bbf
SDDEV void damp_grav_wave(State &state,Arena &arena,int ifrom,int nm1,int deltak,float dt,FArray<float> zt,FArray<float> zm,FArray<float> w,FArray<float> t,FArray<float> tt,FArray<float> qv,FArray<float> qh,FArray<float> qi,FArray<float> qc,FArray<float> te,FArray<float> pe,FArray<float> qvenv) {
ArenaScope scope(arena);
zt.bind(1,nm1);
zm.bind(1,nm1);
w.bind(1,nm1);
t.bind(1,nm1);
tt.bind(1,nm1);
qv.bind(1,nm1);
qh.bind(1,nm1);
qi.bind(1,nm1);
qc.bind(1,nm1);
te.bind(1,nm1);
pe.bind(1,nm1);
qvenv.bind(1,nm1);
auto dummy = arena.alloc<float>(1,nm1);
if ((ifrom == 1)) { // L910
friction(state,arena,ifrom,nm1,deltak,dt,zt,zm,t,tt,te); // L911
return; // L913
} // L914
sd_assign(dummy.section(Slice()),0.f,arena); // L916
if ((ifrom == 2)) { friction(state,arena,ifrom,nm1,deltak,dt,zt,zm,w,dummy,dummy); } // L917
return; // L921
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:925-957 @1d6e370e; SMOKE @9652bbf
SDDEV void friction(State &state,Arena &arena,int ifrom,int nm1,int deltak,float dt,FArray<float> zt,FArray<float> zm,FArray<float> var1,FArray<float> vart,FArray<float> var2) {
ArenaScope scope(arena);
zt.bind(1,nm1);
zm.bind(1,nm1);
var1.bind(1,nm1);
vart.bind(1,nm1);
var2.bind(1,nm1);
int k;
int nfpt;
int kf;
float zmkf;
float ztop;
float distim;
float c1;
float c2;
kf = (nm1 - int(deltak)); // L934
zmkf = zm(kf); // L936
ztop = zm(nm1); // L937
distim = sd_min(sd_mul(3.f,dt),60.f); // L940
c1 = sd_div(1.f,sd_mul(sd_sub(ztop,zmkf),distim)); // L942
c2 = sd_mul(dt,c1); // L943
if ((ifrom == 1)) { // L945
for (k=nm1;k>=2;k+=(-1)) { // L946
if ((zt(k) <= zmkf)) { goto loop1_continue; } // L947
vart(k) = sd_add(sd_mul(sd_sub(var2(k),var1(k)),sd_mul(sd_sub(zt(k),zmkf),c1)),vart(k)); // L948
loop1_continue: ; } loop1_exit: ; // L949
} else if ((ifrom == 2)) { // L950
for (k=nm1;k>=2;k+=(-1)) { // L951
if ((zt(k) <= zmkf)) { goto loop2_continue; } // L952
var1(k) = sd_add(sd_mul(sd_mul(sd_sub(zt(k),zmkf),c2),sd_sub(var2(k),var1(k))),var1(k)); // L953
loop2_continue: ; } loop2_exit: ; // L954
} // L955
return; // L956
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:960-993 @1d6e370e; SMOKE @9652bbf
SDDEV void vel_advectc_plumerise(State &state,Arena &arena,int m1,FArray<float> wc,FArray<float> wt,FArray<float> rho,FArray<float> dzm) {
ArenaScope scope(arena);
wc.bind(1,m1);
wt.bind(1,m1);
rho.bind(1,m1);
dzm.bind(1,m1);
int k;
auto flxw = arena.alloc<float>(1,m1);
auto dn0 = arena.alloc<float>(1,m1);
float c1z;
sd_assign(dn0.section(Slice(1,m1)),sd_mul(rho.section(Slice(1,m1)),1.e-3f),arena); // L970
flxw(1) = sd_mul(dn0(1),wc(1)); // L972
for (k=2;k<=(m1 - 1);k+=1) { // L974
flxw(k) = sd_mul(sd_mul(wc(k),.5f),sd_add(dn0(k),dn0((k + 1)))); // L975
loop1_continue: ; } loop1_exit: ; // L976
c1z = .5f; // L980
for (k=2;k<=(m1 - 2);k+=1) { // L982
wt(k) = sd_add(sd_mul(sd_add(sd_sub(sd_mul(sd_add(flxw(k),flxw((k - 1))),sd_add(wc((k - 1)),wc(k))),sd_mul(sd_add(wc((k + 1)),wc(k)),sd_add(flxw(k),flxw((k + 1))))),sd_mul(sd_add(sd_sub(flxw((k + 1)),flxw((k - 1))),sd_sub(flxw((k + 1)),flxw((k - 1)))),wc(k))),sd_div(sd_mul(dzm(k),c1z),sd_add(dn0(k),dn0((k + 1))))),wt(k)); // L984
loop2_continue: ; } loop2_exit: ; // L990
return; // L992
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:996-1019 @1d6e370e; SMOKE @9652bbf
SDDEV void hadvance_plumerise(State &state,Arena &arena,int iac,int m1,float dt,FArray<float> wc,FArray<float> wt,FArray<float> wp,int mintime) {
ArenaScope scope(arena);
wc.bind(1,m1);
wt.bind(1,m1);
wp.bind(1,m1);
int k;
auto dummy = arena.alloc<float>(1,m1);
float eps;
eps = .2f; // L1008
if ((mintime == 1)) { eps = 0.5f; } // L1009
predict_plumerise(state,arena,m1,wc,wp,wt,dummy,iac,sd_mul(2.f,dt),eps); // L1013
return; // L1018
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1022-1065 @1d6e370e; SMOKE @9652bbf
SDDEV void predict_plumerise(State &state,Arena &arena,int npts,FArray<float> ac,FArray<float> ap,FArray<float> fa,FArray<float> af,int iac,float dtlp,float epsu) {
ArenaScope scope(arena);
ac.bind(1,1);
ap.bind(1,1);
fa.bind(1,1);
af.bind(1,1);
int m;
if ((iac == 1)) { // L1041
for (m=1;m<=npts;m+=1) { // L1042
ac(m) = sd_add(ac(m),sd_mul(epsu,sd_sub(ap(m),sd_mul(2.f,ac(m))))); // L1043
loop1_continue: ; } loop1_exit: ; // L1044
return; // L1045
} else if ((iac == 2)) { // L1046
for (m=1;m<=npts;m+=1) { // L1047
af(m) = ap(m); // L1048
ap(m) = sd_add(ac(m),sd_mul(epsu,af(m))); // L1049
loop2_continue: ; } loop2_exit: ; // L1050
} // L1059
for (m=1;m<=npts;m+=1) { // L1061
ac(m) = af(m); // L1062
loop3_continue: ; } loop3_exit: ; // L1063
return; // L1064
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1068-1101 @1d6e370e; SMOKE @9652bbf
SDDEV void buoyancy_plumerise(State &state,Arena &arena,int m1,FArray<float> t,FArray<float> te,FArray<float> qv,FArray<float> qvenv,FArray<float> qh,FArray<float> qi,FArray<float> qc,FArray<float> wt,FArray<float> scr1) {
ArenaScope scope(arena);
t.bind(1,m1);
te.bind(1,m1);
qv.bind(1,m1);
qvenv.bind(1,m1);
qh.bind(1,m1);
qi.bind(1,m1);
qc.bind(1,m1);
wt.bind(1,m1);
scr1.bind(1,m1);
int k;
float g = __uint_as_float(0x411CCCCDU);
float eps = __uint_as_float(0x3F1F3B64U);
float gama = __uint_as_float(0x3F000000U);
float tv;
float tve;
float qwtotl;
float umgamai;
float mu = __uint_as_float(0x3E19999AU);
umgamai = sd_div(1.f,sd_add(1.f,gama)); // L1077
for (k=2;k<=(m1 - 1);k+=1) { // L1083
tv = sd_div(sd_mul(sd_add(sd_div(qv(k),eps),1.f),t(k)),sd_add(qv(k),1.f)); // L1085
tve = sd_div(sd_mul(sd_add(sd_div(qvenv(k),eps),1.f),te(k)),sd_add(qvenv(k),1.f)); // L1086
qwtotl = sd_add(sd_add(qh(k),qi(k)),qc(k)); // L1088
scr1(k) = sd_mul(sd_sub(sd_div(sd_sub(tv,tve),tve),qwtotl),sd_mul(g,umgamai)); // L1091
loop1_continue: ; } loop1_exit: ; // L1094
for (k=2;k<=(m1 - 2);k+=1) { // L1096
wt(k) = sd_add(sd_mul(sd_add(scr1(k),scr1((k + 1))),0.5f),wt(k)); // L1097
loop2_continue: ; } loop2_exit: ; // L1099
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1104-1146 @1d6e370e; SMOKE @9652bbf
SDDEV void entrainment(State &state,Arena &arena,Plume* coms,int m1,FArray<float> w,FArray<float> wt,FArray<float> radius,float alpha) {
ArenaScope scope(arena);
w.bind(1,m1);
wt.bind(1,m1);
radius.bind(1,m1);
int k;
float dmdtm;
float wbar;
float radius_bar;
float umgamai;
float dyn_entr;
float mu = __uint_as_float(0x3E19999AU);
float gama = __uint_as_float(0x3F000000U);
umgamai = sd_div(1.f,sd_add(1.f,gama)); // L1117
for (k=2;k<=(m1 - 1);k+=1) { // L1122
wbar = w(k); // L1126
radius_bar = sd_mul(sd_add(radius((k - 1)),radius(k)),0.5f); // L1127
dmdtm = sd_div(sd_mul(sd_mul(sd_mul(umgamai,2.f),alpha),sd_abs(wbar)),radius_bar); // L1130
wt(k) = sd_sub(wt(k),sd_mul(dmdtm,sd_abs(wbar))); // L1133
dyn_entr = sd_div(sd_mul(sd_abs(sd_sub(sd_add(sd_sub(coms->vel_p(k),coms->vel_e(k)),coms->vel_p((k - 1))),coms->vel_e((k - 1)))),sd_mul(sd_div(2.f,3.1416f),0.5f)),radius_bar); // L1139
wt(k) = sd_sub(wt(k),sd_mul(dyn_entr,sd_abs(wbar))); // L1141
loop1_continue: ; } loop1_exit: ; // L1145
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1149-1275 @1d6e370e; SMOKE @9652bbf
SDDEV void scl_advectc_plumerise(State &state,Arena &arena,Plume* coms,FStr varn,int mzp) {
ArenaScope scope(arena);
float dtlto2;
int k;
dtlto2 = sd_mul(coms->dt,0.5f); // L1160
coms->vt3dc(1) = sd_mul(sd_mul(sd_mul(sd_add(coms->w(1),coms->wc(1)),dtlto2),coms->rho(1)),1.e-3f); // L1162
coms->vt3df(1) = sd_mul(sd_mul(sd_mul(sd_add(coms->w(1),coms->wc(1)),0.5f),dtlto2),coms->dzm(1)); // L1163
for (k=2;k<=mzp;k+=1) { // L1165
coms->vt3dc(k) = sd_mul(sd_mul(sd_add(coms->rho(k),coms->rho((k + 1))),sd_mul(sd_mul(sd_add(coms->w(k),coms->wc(k)),dtlto2),0.5f)),1.e-3f); // L1167
coms->vt3df(k) = sd_mul(sd_mul(sd_mul(sd_add(coms->w(k),coms->wc(k)),dtlto2),.5f),coms->dzm(k)); // L1168
loop1_continue: ; } loop1_exit: ; // L1170
for (k=1;k<=mzp;k+=1) { // L1175
coms->vctr1(k) = sd_mul(sd_sub(coms->zt((k + 1)),coms->zm(k)),coms->dzm(k)); // L1176
coms->vctr2(k) = sd_mul(sd_sub(coms->zm(k),coms->zt(k)),coms->dzm(k)); // L1177
coms->vt3dk(k) = sd_div(coms->dzt(k),sd_mul(coms->rho(k),1.e-3f)); // L1179
loop2_continue: ; } loop2_exit: ; // L1181
sd_assign(coms->scr1,coms->t,arena); // L1187
fa_zc_plumerise(state,arena,mzp,coms->t,sd_sequence(coms->scr1,1),sd_sequence(coms->vt3dc,1),sd_sequence(coms->vt3df,1),sd_sequence(coms->vt3dg,1),sd_sequence(coms->vt3dk,1),coms->vctr1,coms->vctr2); // L1188
advtndc_plumerise(state,arena,mzp,coms->t,sd_sequence(coms->scr1,1),coms->tt,coms->dt); // L1194
sd_assign(coms->scr1,coms->qv,arena); // L1197
fa_zc_plumerise(state,arena,mzp,coms->qv,sd_sequence(coms->scr1,1),sd_sequence(coms->vt3dc,1),sd_sequence(coms->vt3df,1),sd_sequence(coms->vt3dg,1),sd_sequence(coms->vt3dk,1),coms->vctr1,coms->vctr2); // L1198
advtndc_plumerise(state,arena,mzp,coms->qv,sd_sequence(coms->scr1,1),coms->qvt,coms->dt); // L1204
sd_assign(coms->scr1,coms->qc,arena); // L1207
fa_zc_plumerise(state,arena,mzp,coms->qc,sd_sequence(coms->scr1,1),sd_sequence(coms->vt3dc,1),sd_sequence(coms->vt3df,1),sd_sequence(coms->vt3dg,1),sd_sequence(coms->vt3dk,1),coms->vctr1,coms->vctr2); // L1208
advtndc_plumerise(state,arena,mzp,coms->qc,sd_sequence(coms->scr1,1),coms->qct,coms->dt); // L1214
sd_assign(coms->scr1,coms->qi,arena); // L1217
fa_zc_plumerise(state,arena,mzp,coms->qi,sd_sequence(coms->scr1,1),sd_sequence(coms->vt3dc,1),sd_sequence(coms->vt3df,1),sd_sequence(coms->vt3dg,1),sd_sequence(coms->vt3dk,1),coms->vctr1,coms->vctr2); // L1218
advtndc_plumerise(state,arena,mzp,coms->qi,sd_sequence(coms->scr1,1),coms->qit,coms->dt); // L1224
sd_assign(coms->scr1,coms->qh,arena); // L1229
fa_zc_plumerise(state,arena,mzp,coms->qh,sd_sequence(coms->scr1,1),sd_sequence(coms->vt3dc,1),sd_sequence(coms->vt3df,1),sd_sequence(coms->vt3dg,1),sd_sequence(coms->vt3dk,1),coms->vctr1,coms->vctr2); // L1230
advtndc_plumerise(state,arena,mzp,coms->qh,sd_sequence(coms->scr1,1),coms->qht,coms->dt); // L1236
sd_assign(coms->scr1,coms->vel_p,arena); // L1239
fa_zc_plumerise(state,arena,mzp,coms->vel_p,sd_sequence(coms->scr1,1),sd_sequence(coms->vt3dc,1),sd_sequence(coms->vt3df,1),sd_sequence(coms->vt3dg,1),sd_sequence(coms->vt3dk,1),coms->vctr1,coms->vctr2); // L1240
advtndc_plumerise(state,arena,mzp,coms->vel_p,sd_sequence(coms->scr1,1),coms->vel_t,coms->dt); // L1246
sd_assign(coms->scr1,coms->rad_p,arena); // L1250
fa_zc_plumerise(state,arena,mzp,coms->rad_p,sd_sequence(coms->scr1,1),sd_sequence(coms->vt3dc,1),sd_sequence(coms->vt3df,1),sd_sequence(coms->vt3dg,1),sd_sequence(coms->vt3dk,1),coms->vctr1,coms->vctr2); // L1251
advtndc_plumerise(state,arena,mzp,coms->rad_p,sd_sequence(coms->scr1,1),coms->rad_t,coms->dt); // L1257
return; // L1260
sd_assign(coms->scr1,coms->sc,arena); // L1264
fa_zc_plumerise(state,arena,mzp,coms->sc,sd_sequence(coms->scr1,1),sd_sequence(coms->vt3dc,1),sd_sequence(coms->vt3df,1),sd_sequence(coms->vt3dg,1),sd_sequence(coms->vt3dk,1),coms->vctr1,coms->vctr2); // L1265
advtndc_plumerise(state,arena,mzp,coms->sc,sd_sequence(coms->scr1,1),coms->sct,coms->dt); // L1271
return; // L1274
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1278-1322 @1d6e370e; SMOKE @9652bbf
SDDEV void fa_zc_plumerise(State &state,Arena &arena,int m1,FArray<float> scp,FArray<float> scr1,FArray<float> vt3dc,FArray<float> vt3df,FArray<float> vt3dg,FArray<float> vt3dk,FArray<float> vctr1,FArray<float> vctr2) {
ArenaScope scope(arena);
scp.bind(1,m1);
scr1.bind(1,m1);
vt3dc.bind(1,m1);
vt3df.bind(1,m1);
vt3dg.bind(1,m1);
vt3dk.bind(1,m1);
vctr1.bind(1,m1);
vctr2.bind(1,m1);
int k;
float dfact;
dfact = .5f; // L1286
for (k=1;k<=(m1 - 1);k+=1) { // L1289
vt3dg(k) = sd_mul(sd_add(sd_add(sd_mul(vctr2(k),scr1((k + 1))),sd_mul(vctr1(k),scr1(k))),sd_mul(sd_sub(scr1(k),scr1((k + 1))),vt3df(k))),vt3dc(k)); // L1290
loop1_continue: ; } loop1_exit: ; // L1294
for (k=1;k<=(m1 - 1);k+=1) { // L1302
if ((vt3dc(k) > 0.f)) { // L1303
if ((sd_mul(vt3dg(k),vt3dk(k)) > sd_mul(dfact,scr1(k)))) { // L1304
vt3dg(k) = sd_mul(vt3dc(k),scr1(k)); // L1305
} // L1306
} else if ((vt3dc(k) < 0.f)) { // L1307
if ((sd_mul(sd_neg(vt3dg(k)),vt3dk((k + 1))) > sd_mul(dfact,scr1((k + 1))))) { // L1308
vt3dg(k) = sd_mul(vt3dc(k),scr1((k + 1))); // L1309
} // L1310
} // L1311
loop2_continue: ; } loop2_exit: ; // L1313
for (k=2;k<=(m1 - 1);k+=1) { // L1316
scr1(k) = sd_add(sd_mul(sd_add(sd_mul(sd_sub(vt3dc(k),vt3dc((k - 1))),scp(k)),sd_sub(vt3dg((k - 1)),vt3dg(k))),vt3dk(k)),scr1(k)); // L1317
loop3_continue: ; } loop3_exit: ; // L1320
return; // L1321
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1325-1336 @1d6e370e; SMOKE @9652bbf
SDDEV void advtndc_plumerise(State &state,Arena &arena,int m1,FArray<float> scp,FArray<float> sca,FArray<float> sct,float dtl) {
ArenaScope scope(arena);
scp.bind(1,m1);
sca.bind(1,m1);
sct.bind(1,m1);
int k;
float dtli;
dtli = sd_div(1.f,dtl); // L1331
for (k=2;k<=(m1 - 1);k+=1) { // L1332
sct(k) = sd_add(sd_mul(sd_sub(sca(k),scp(k)),dtli),sct(k)); // L1333
loop1_continue: ; } loop1_exit: ; // L1334
return; // L1335
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1339-1351 @1d6e370e; SMOKE @9652bbf
SDDEV void tend0_plumerise(State &state,Arena &arena,Plume* coms) {
ArenaScope scope(arena);
sd_assign(coms->wt.section(Slice(1,coms->nm1)),0.f,arena); // L1342
sd_assign(coms->tt.section(Slice(1,coms->nm1)),0.f,arena); // L1343
sd_assign(coms->qvt.section(Slice(1,coms->nm1)),0.f,arena); // L1344
sd_assign(coms->qct.section(Slice(1,coms->nm1)),0.f,arena); // L1345
sd_assign(coms->qht.section(Slice(1,coms->nm1)),0.f,arena); // L1346
sd_assign(coms->qit.section(Slice(1,coms->nm1)),0.f,arena); // L1347
sd_assign(coms->vel_t.section(Slice(1,coms->nm1)),0.f,arena); // L1348
sd_assign(coms->rad_t.section(Slice(1,coms->nm1)),0.f,arena); // L1349
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1355-1390 @1d6e370e; SMOKE @9652bbf
SDDEV void scl_misc(State &state,Arena &arena,Plume* coms,int m1,float alpha) {
ArenaScope scope(arena);
float g = __uint_as_float(0x411CF5C3U);
float cp = __uint_as_float(0x447B0000U);
int k;
float dmdtm;
for (k=2;k<=(m1 - 1);k+=1) { // L1363
coms->wbar = sd_mul(0.5f,sd_add(coms->w(k),coms->w((k - 1)))); // L1364
coms->adiabat = sd_neg(sd_div(sd_mul(coms->wbar,g),cp)); // L1366
dmdtm = sd_div(sd_mul(sd_abs(coms->wbar),sd_add(alpha,alpha)),coms->radius(k)); // L1369
coms->tt(k) = sd_sub(sd_sub(coms->tt(k),sd_div(sd_mul(coms->wbar,g),cp)),sd_mul(sd_sub(coms->t(k),coms->te(k)),dmdtm)); // L1372
coms->qvt(k) = sd_sub(coms->qvt(k),sd_mul(sd_sub(coms->qv(k),coms->qvenv(k)),dmdtm)); // L1375
coms->qct(k) = sd_sub(coms->qct(k),sd_mul(coms->qc(k),dmdtm)); // L1377
coms->qht(k) = sd_sub(coms->qht(k),sd_mul(coms->qh(k),dmdtm)); // L1378
coms->qit(k) = sd_sub(coms->qit(k),sd_mul(coms->qi(k),dmdtm)); // L1379
coms->vel_t(k) = sd_sub(coms->vel_t(k),sd_mul(sd_sub(coms->vel_p(k),coms->vel_e(k)),dmdtm)); // L1382
coms->rad_t(k) = sd_add(sd_mul(sd_mul(sd_mul(dmdtm,0.5f),sd_div(6.f,5.f)),coms->radius(k)),coms->rad_t(k)); // L1385
loop1_continue: ; } loop1_exit: ; // L1389
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1393-1459 @1d6e370e; SMOKE @9652bbf
SDDEV void scl_dyn_entrain(State &state,Arena &arena,int m1,int nkp,float & wbar,FArray<float> w,float & adiabat,float alpha,FArray<float> radius,FArray<float> tt,FArray<float> t,FArray<float> te,FArray<float> qvt,FArray<float> qv,FArray<float> qvenv,FArray<float> qct,FArray<float> qc,FArray<float> qht,FArray<float> qh,FArray<float> qit,FArray<float> qi,FArray<float> vel_e,FArray<float> vel_p,FArray<float> vel_t,FArray<float> rad_p,FArray<float> rad_t) {
ArenaScope scope(arena);
w.bind(1,nkp);
radius.bind(1,nkp);
tt.bind(1,nkp);
t.bind(1,nkp);
te.bind(1,nkp);
qvt.bind(1,nkp);
qv.bind(1,nkp);
qvenv.bind(1,nkp);
qct.bind(1,nkp);
qc.bind(1,nkp);
qht.bind(1,nkp);
qh.bind(1,nkp);
qit.bind(1,nkp);
qi.bind(1,nkp);
vel_e.bind(1,nkp);
vel_p.bind(1,nkp);
vel_t.bind(1,nkp);
rad_p.bind(1,nkp);
rad_t.bind(1,nkp);
float g = __uint_as_float(0x411CF5C3U);
float cp = __uint_as_float(0x447B0000U);
float pi = __uint_as_float(0x40490FF9U);
int k;
float dmdtm;
for (k=2;k<=(m1 - 1);k+=1) { // L1430
rad_t(k) = sd_add(sd_div(sd_abs(sd_sub(vel_e(k),vel_p(k))),pi),rad_t(k)); // L1434
dmdtm = sd_div(sd_mul(sd_abs(sd_sub(vel_e(k),vel_p(k))),sd_div(2.f,3.1416f)),radius(k)); // L1438
vel_t(k) = sd_sub(vel_t(k),sd_mul(sd_sub(vel_p(k),vel_e(k)),dmdtm)); // L1441
tt(k) = sd_sub(tt(k),sd_mul(sd_sub(t(k),te(k)),dmdtm)); // L1446
qvt(k) = sd_sub(qvt(k),sd_mul(sd_sub(qv(k),qvenv(k)),dmdtm)); // L1449
qct(k) = sd_sub(qct(k),sd_mul(qc(k),dmdtm)); // L1451
qht(k) = sd_sub(qht(k),sd_mul(qh(k),dmdtm)); // L1452
qit(k) = sd_sub(qit(k),sd_mul(dmdtm,qi(k))); // L1453
loop1_continue: ; } loop1_exit: ; // L1458
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1463-1514 @1d6e370e; SMOKE @9652bbf
SDDEV void visc_w(State &state,Arena &arena,Plume* coms,int m1,int deltak,int kmt) {
ArenaScope scope(arena);
int k;
int m2;
float dz1t;
float dz1m;
float dz2t;
float dz2m;
float d2wdz;
float d2tdz;
float d2qvdz;
float d2qhdz;
float d2qcdz;
float d2qidz;
float d2scdz;
float d2vel_pdz;
float d2rad_dz;
auto &printed = state.saved_visc_w_printed;
m2 = sd_min(m1,kmt); // L1476
for (k=2;k<=(m2 - 1);k+=1) { // L1479
dz1t = sd_mul(0.5f,sd_sub(coms->zt((k + 1)),coms->zt((k - 1)))); // L1480
dz2t = sd_div(coms->visc(k),sd_mul(dz1t,dz1t)); // L1481
dz1m = sd_mul(0.5f,sd_sub(coms->zm((k + 1)),coms->zm((k - 1)))); // L1482
dz2m = sd_div(coms->visc(k),sd_mul(dz1m,dz1m)); // L1483
d2wdz = sd_mul(dz2m,sd_add(sd_sub(coms->w((k + 1)),sd_add(coms->w(k),coms->w(k))),coms->w((k - 1)))); // L1484
d2tdz = sd_mul(sd_add(coms->t((k - 1)),sd_sub(coms->t((k + 1)),sd_add(coms->t(k),coms->t(k)))),dz2t); // L1485
d2qvdz = sd_mul(sd_add(sd_sub(coms->qv((k + 1)),sd_add(coms->qv(k),coms->qv(k))),coms->qv((k - 1))),dz2t); // L1486
d2qhdz = sd_mul(sd_add(sd_sub(coms->qh((k + 1)),sd_add(coms->qh(k),coms->qh(k))),coms->qh((k - 1))),dz2t); // L1487
d2qcdz = sd_mul(sd_add(sd_sub(coms->qc((k + 1)),sd_add(coms->qc(k),coms->qc(k))),coms->qc((k - 1))),dz2t); // L1488
d2qidz = sd_mul(sd_add(sd_sub(coms->qi((k + 1)),sd_add(coms->qi(k),coms->qi(k))),coms->qi((k - 1))),dz2t); // L1489
d2vel_pdz = sd_mul(sd_add(sd_sub(coms->vel_p((k + 1)),sd_add(coms->vel_p(k),coms->vel_p(k))),coms->vel_p((k - 1))),dz2t); // L1491
d2rad_dz = sd_mul(sd_add(sd_sub(coms->rad_p((k + 1)),sd_add(coms->rad_p(k),coms->rad_p(k))),coms->rad_p((k - 1))),dz2t); // L1492
coms->wt(k) = sd_add(d2wdz,coms->wt(k)); // L1494
coms->tt(k) = sd_add(d2tdz,coms->tt(k)); // L1496
coms->qvt(k) = sd_add(d2qvdz,coms->qvt(k)); // L1504
coms->qct(k) = sd_add(d2qcdz,coms->qct(k)); // L1505
coms->qht(k) = sd_add(d2qhdz,coms->qht(k)); // L1506
coms->qit(k) = sd_add(d2qidz,coms->qit(k)); // L1507
coms->vel_t(k) = sd_add(d2vel_pdz,coms->vel_t(k)); // L1508
coms->rad_t(k) = sd_add(d2rad_dz,coms->rad_t(k)); // L1509
loop1_continue: ; } loop1_exit: ; // L1512
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1518-1561 @1d6e370e; SMOKE @9652bbf
SDDEV void update_plumerise(State &state,Arena &arena,Plume* coms,int m1,FStr varn) {
ArenaScope scope(arena);
int k;
if ((varn == FStr("W"))) { // L1526
for (k=2;k<=(m1 - 1);k+=1) { // L1528
coms->w(k) = sd_add(sd_mul(coms->wt(k),coms->dt),coms->w(k)); // L1529
loop1_continue: ; } loop1_exit: ; // L1530
return; // L1531
} else { // L1533
for (k=2;k<=(m1 - 1);k+=1) { // L1534
coms->t(k) = sd_add(sd_mul(coms->tt(k),coms->dt),coms->t(k)); // L1536
coms->qv(k) = sd_add(sd_mul(coms->qvt(k),coms->dt),coms->qv(k)); // L1542
coms->qc(k) = sd_add(sd_mul(coms->qct(k),coms->dt),coms->qc(k)); // L1544
coms->qh(k) = sd_add(sd_mul(coms->qht(k),coms->dt),coms->qh(k)); // L1545
coms->qi(k) = sd_add(sd_mul(coms->qit(k),coms->dt),coms->qi(k)); // L1546
coms->qv(k) = sd_max(coms->qv(k),0.f); // L1550
coms->qc(k) = sd_max(coms->qc(k),0.f); // L1551
coms->qh(k) = sd_max(coms->qh(k),0.f); // L1552
coms->qi(k) = sd_max(coms->qi(k),0.f); // L1553
coms->vel_p(k) = sd_add(sd_mul(coms->vel_t(k),coms->dt),coms->vel_p(k)); // L1555
coms->rad_p(k) = sd_add(sd_mul(coms->rad_t(k),coms->dt),coms->rad_p(k)); // L1556
loop2_continue: ; } loop2_exit: ; // L1559
} // L1560
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1564-1627 @1d6e370e; SMOKE @9652bbf
SDDEV void fallpart(State &state,Arena &arena,Plume* coms,int m1) {
ArenaScope scope(arena);
int k;
float vtc;
float dfhz;
float dfiz;
float dz1;
float vconst = __uint_as_float(0x40A36FB7U);
float eps = __uint_as_float(0x3F1F3B64U);
float f0 = __uint_as_float(0x3F400000U);
float g = __uint_as_float(0x411CF5C3U);
float cp = __uint_as_float(0x447B0000U);
for (k=2;k<=(m1 - 1);k+=1) { // L1583
vtc = sd_mul(vconst,sd_pow(coms->rho(k),.125f)); // L1588
coms->vth(k) = sd_neg(4.f); // L1592
coms->vhrel = sd_add(coms->w(k),coms->vth(k)); // L1594
coms->cvh(k) = sd_add(1.6f,sd_mul(0.57e-3f,sd_pow(sd_abs(coms->vhrel),1.5f))); // L1597
coms->vti(k) = sd_neg(3.f); // L1600
coms->virel = sd_add(coms->w(k),coms->vti(k)); // L1602
coms->cvi(k) = sd_add(1.6f,sd_div(sd_mul(0.57e-3f,sd_pow(sd_abs(coms->virel),1.5f)),f0)); // L1605
if ((coms->vhrel >= 0.0f)) { // L1608
dfhz = sd_div(sd_mul(coms->qh(k),sd_sub(sd_mul(coms->rho(k),coms->vth(k)),sd_mul(coms->rho((k - 1)),coms->vth((k - 1))))),coms->rho((k - 1))); // L1609
} else { // L1610
dfhz = sd_div(sd_mul(coms->qh(k),sd_sub(sd_mul(coms->rho((k + 1)),coms->vth((k + 1))),sd_mul(coms->rho(k),coms->vth(k)))),coms->rho(k)); // L1611
} // L1612
if ((coms->virel >= 0.0f)) { // L1615
dfiz = sd_div(sd_mul(coms->qi(k),sd_sub(sd_mul(coms->rho(k),coms->vti(k)),sd_mul(coms->rho((k - 1)),coms->vti((k - 1))))),coms->rho((k - 1))); // L1616
} else { // L1617
dfiz = sd_div(sd_mul(coms->qi(k),sd_sub(sd_mul(coms->rho((k + 1)),coms->vti((k + 1))),sd_mul(coms->rho(k),coms->vti(k)))),coms->rho(k)); // L1618
} // L1619
dz1 = sd_sub(coms->zm(k),coms->zm((k - 1))); // L1621
coms->qht(k) = sd_sub(coms->qht(k),sd_div(dfhz,dz1)); // L1623
coms->qit(k) = sd_sub(coms->qit(k),sd_div(dfiz,dz1)); // L1624
loop1_continue: ; } loop1_exit: ; // L1626
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1630-1655 @1d6e370e; SMOKE @9652bbf
SDDEV void waterbal(State &state,Arena &arena,Plume* coms) {
ArenaScope scope(arena);
if ((coms->qc(coms->l) <= 1.0e-10f)) { coms->qc(coms->l) = 0.f; } // L1637
if ((coms->qh(coms->l) <= 1.0e-10f)) { coms->qh(coms->l) = 0.f; } // L1638
if ((coms->qi(coms->l) <= 1.0e-10f)) { coms->qi(coms->l) = 0.f; } // L1639
evaporate(state,arena,coms); // L1641
sublimate(state,arena,coms); // L1643
glaciate(state,arena,coms); // L1645
melt(state,arena,coms); // L1647
convert(state,arena,coms); // L1650
return; // L1654
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1657-1870 @1d6e370e; SMOKE @9652bbf
SDDEV void evaporate(State &state,Arena &arena,Plume* coms) {
ArenaScope scope(arena);
float herc = __uint_as_float(0x3A0E9B39U);
float cp = __uint_as_float(0x3F808312U);
float heatcond = __uint_as_float(0x451C4000U);
float heatsubl = __uint_as_float(0x45312000U);
float tmelt = __uint_as_float(0x43888000U);
float tfreeze = __uint_as_float(0x4386A666U);
float frc = __uint_as_float(0x451BA0A4U);
float src = __uint_as_float(0x45306B59U);
float evhdt;
float evidt;
float evrate;
float evap;
float sd;
float quant;
float dividend;
float divisor;
float devidt;
sd = sd_sub(coms->qsat(coms->l),coms->qv(coms->l)); // L1677
if ((sd == 0.0f)) { return; } // L1678
evhdt = 0.f; // L1682
evidt = 0.f; // L1683
evrate = sd_abs(sd_mul(coms->wbar,coms->dqsdz)); // L1686
evap = sd_mul(evrate,coms->dt); // L1687
if ((sd <= 0.0f)) { // L1690
if ((evap >= sd_abs(sd))) { // L1692
coms->qc(coms->l) = sd_sub(coms->qc(coms->l),sd); // L1694
coms->qv(coms->l) = coms->qsat(coms->l); // L1695
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(sd,frc)); // L1696
return; // L1698
} else { // L1700
coms->qc(coms->l) = sd_add(coms->qc(coms->l),evap); // L1702
coms->qv(coms->l) = sd_sub(coms->qv(coms->l),evap); // L1703
coms->t(coms->l) = sd_add(coms->t(coms->l),sd_mul(evap,frc)); // L1704
return; // L1706
} // L1708
} else { // L1710
if ((evap <= coms->qc(coms->l))) { // L1715
if ((sd <= evap)) { // L1718
coms->qc(coms->l) = sd_sub(coms->qc(coms->l),sd); // L1720
coms->qv(coms->l) = coms->qsat(coms->l); // L1721
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(sd,frc)); // L1722
return; // L1723
} else { // L1726
sd = sd_sub(sd,evap); // L1728
coms->qv(coms->l) = sd_add(coms->qv(coms->l),evap); // L1729
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(evap,frc)); // L1730
coms->qc(coms->l) = sd_sub(coms->qc(coms->l),evap); // L1731
} // L1733
} else { // L1735
if ((sd <= coms->qc(coms->l))) { // L1737
coms->qv(coms->l) = coms->qsat(coms->l); // L1739
coms->qc(coms->l) = sd_sub(coms->qc(coms->l),sd); // L1740
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(sd,frc)); // L1741
return; // L1742
} else { // L1744
sd = sd_sub(sd,coms->qc(coms->l)); // L1745
coms->qv(coms->l) = sd_add(coms->qv(coms->l),coms->qc(coms->l)); // L1746
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(coms->qc(coms->l),frc)); // L1747
coms->qc(coms->l) = 0.0f; // L1748
} // L1750
} // L1751
if ((coms->qh(coms->l) > 1.e-10f)) { // L1762
quant = sd_mul(sd_sub(sd_sub(coms->qsat(coms->l),coms->qc(coms->l)),coms->qv(coms->l)),coms->rho(coms->l)); // L1766
evhdt = sd_div(sd_mul(sd_mul(sd_mul(coms->dt,herc),quant),sd_pow(sd_mul(coms->qh(coms->l),coms->rho(coms->l)),.65f)),coms->rho(coms->l)); // L1768
if ((evhdt <= coms->qh(coms->l))) { // L1771
if ((sd <= evhdt)) { // L1773
coms->qh(coms->l) = sd_sub(coms->qh(coms->l),sd); // L1774
coms->qv(coms->l) = coms->qsat(coms->l); // L1775
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(sd,frc)); // L1776
return; // L1778
} else { // L1780
sd = sd_sub(sd,evhdt); // L1781
coms->qv(coms->l) = sd_add(coms->qv(coms->l),evhdt); // L1782
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(evhdt,frc)); // L1783
coms->qh(coms->l) = sd_sub(coms->qh(coms->l),evhdt); // L1784
} // L1786
} else { // L1788
if ((sd <= coms->qh(coms->l))) { // L1790
coms->qv(coms->l) = coms->qsat(coms->l); // L1791
coms->qh(coms->l) = sd_sub(coms->qh(coms->l),sd); // L1792
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(sd,frc)); // L1793
return; // L1794
} else { // L1796
sd = sd_sub(sd,coms->qh(coms->l)); // L1797
coms->qv(coms->l) = sd_add(coms->qv(coms->l),coms->qh(coms->l)); // L1798
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(coms->qh(coms->l),frc)); // L1799
coms->qh(coms->l) = 0.0f; // L1800
} // L1802
} // L1805
} // L1811
if ((coms->qi(coms->l) <= 1.e-10f)) { return; } // L1812
dividend = sd_mul(sd_mul(sd_mul(sd_sub(sd_div(sd,coms->qsat(coms->l)),1.f),sd_pow(sd_div(1.e6f,coms->rho(coms->l)),0.475f)),sd_pow(coms->qi(coms->l),0.525f)),1.13f); // L1814
divisor = sd_add(sd_div(4.1e6f,sd_mul(10.f,coms->est(coms->l))),7.e5f); // L1816
devidt = sd_neg(sd_div(sd_mul(dividend,coms->cvi(coms->l)),divisor)); // L1818
evidt = sd_mul(devidt,coms->dt); // L1820
if ((evidt <= coms->qi(coms->l))) { // L1826
if ((sd <= evidt)) { // L1829
coms->qi(coms->l) = sd_sub(coms->qi(coms->l),sd); // L1830
coms->qv(coms->l) = coms->qsat(coms->l); // L1831
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(sd,src)); // L1832
return; // L1834
} else { // L1837
sd = sd_sub(sd,evidt); // L1839
coms->qv(coms->l) = sd_add(coms->qv(coms->l),evidt); // L1840
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(evidt,src)); // L1841
coms->qi(coms->l) = sd_sub(coms->qi(coms->l),evidt); // L1842
} // L1844
} else { // L1846
if ((sd <= coms->qi(coms->l))) { // L1848
coms->qv(coms->l) = coms->qsat(coms->l); // L1850
coms->qi(coms->l) = sd_sub(coms->qi(coms->l),sd); // L1851
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(sd,src)); // L1852
return; // L1854
} else { // L1856
sd = sd_sub(sd,coms->qi(coms->l)); // L1857
coms->qv(coms->l) = sd_add(coms->qv(coms->l),coms->qi(coms->l)); // L1858
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(coms->qi(coms->l),src)); // L1859
coms->qi(coms->l) = 0.0f; // L1860
} // L1862
} // L1864
} // L1866
return; // L1868
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1873-1947 @1d6e370e; SMOKE @9652bbf
SDDEV void convert(State &state,Arena &arena,Plume* coms) {
ArenaScope scope(arena);
float ak1 = __uint_as_float(0x3A83126FU);
float ak2 = __uint_as_float(0x3BAA64C3U);
float th = __uint_as_float(0x3F000000U);
int iconv = 1;
float anbase = __uint_as_float(0x47C35000U);
float bdisp = __uint_as_float(0x3E158106U);
float tfreeze = __uint_as_float(0x4386A666U);
float accrete;
float con;
float q;
float h;
float bc1;
float bc2;
float total;
if ((coms->t(coms->l) <= tfreeze)) { return; } // L1896
if ((coms->qc(coms->l) == 0.f)) { return; } // L1898
accrete = 0.f; // L1900
con = 0.f; // L1901
q = sd_mul(coms->rho(coms->l),coms->qc(coms->l)); // L1902
h = sd_mul(coms->rho(coms->l),coms->qh(coms->l)); // L1903
if ((coms->qh(coms->l) > 0.f)) { accrete = sd_mul(sd_mul(ak2,q),sd_pow(h,.875f)); } // L1908
if ((iconv != 0)) { // L1910
con = sd_div(sd_mul(sd_mul(sd_mul(q,q),q),bdisp),sd_mul(60.f,sd_add(sd_mul(sd_mul(5.f,q),bdisp),sd_mul(0.0366f,anbase)))); // L1916
} else { // L1918
con = sd_max(0.f,sd_mul(ak1,sd_sub(q,th))); // L1924
} // L1926
total = sd_div(sd_mul(sd_add(con,accrete),coms->dt),coms->rho(coms->l)); // L1929
if ((total < coms->qc(coms->l))) { // L1932
coms->qc(coms->l) = sd_sub(coms->qc(coms->l),total); // L1934
coms->qh(coms->l) = sd_add(coms->qh(coms->l),total); // L1935
return; // L1936
} else { // L1938
coms->qh(coms->l) = sd_add(coms->qh(coms->l),coms->qc(coms->l)); // L1940
coms->qc(coms->l) = 0.0f; // L1941
} // L1943
return; // L1945
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:1951-2000 @1d6e370e; SMOKE @9652bbf
SDDEV void sublimate(State &state,Arena &arena,Plume* coms) {
ArenaScope scope(arena);
float eps = __uint_as_float(0x3F1F3B64U);
float heatfus = __uint_as_float(0x43A70000U);
float heatsubl = __uint_as_float(0x45312000U);
float cp = __uint_as_float(0x3F808312U);
float src = __uint_as_float(0x45306B59U);
float frc = __uint_as_float(0x43A655ADU);
float tmelt = __uint_as_float(0x4388A666U);
float tfreeze = __uint_as_float(0x4386A666U);
float dtsubh;
float dividend;
float divisor;
float subl;
dtsubh = 0.f; // L1965
if ((coms->t(coms->l) > tfreeze)) { return; } // L1968
if ((coms->qv(coms->l) <= coms->qsat(coms->l))) { return; } // L1969
dividend = sd_mul(sd_mul(sd_mul(sd_pow(sd_div(1.e6f,coms->rho(coms->l)),0.475f),sd_sub(sd_div(coms->qv(coms->l),coms->qsat(coms->l)),1)),sd_pow(coms->qi(coms->l),0.525f)),1.13f); // L1973
divisor = sd_add(7.e5f,sd_div(4.1e6f,sd_mul(10.f,coms->est(coms->l)))); // L1975
dtsubh = sd_abs(sd_div(dividend,divisor)); // L1978
subl = sd_mul(dtsubh,coms->dt); // L1979
if ((subl < coms->qv(coms->l))) { // L1983
coms->qv(coms->l) = sd_sub(coms->qv(coms->l),subl); // L1985
coms->qi(coms->l) = sd_add(coms->qi(coms->l),subl); // L1986
coms->t(coms->l) = sd_add(coms->t(coms->l),sd_mul(subl,src)); // L1987
return; // L1989
} else { // L1991
coms->qi(coms->l) = coms->qv(coms->l); // L1993
coms->t(coms->l) = sd_add(coms->t(coms->l),sd_mul(coms->qv(coms->l),src)); // L1994
coms->qv(coms->l) = 0.0f; // L1995
} // L1997
return; // L1999
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:2004-2054 @1d6e370e; SMOKE @9652bbf
SDDEV void glaciate(State &state,Arena &arena,Plume* coms) {
ArenaScope scope(arena);
float heatfus = __uint_as_float(0x43A70000U);
float cp = __uint_as_float(0x3F808312U);
float eps = __uint_as_float(0x3F1F3B64U);
float heatsubl = __uint_as_float(0x45312000U);
float frc = __uint_as_float(0x43A655ADU);
float frs = __uint_as_float(0x45306B59U);
float tfreeze = __uint_as_float(0x4386A666U);
float glconst = __uint_as_float(0x3CCCCCCDU);
float dfrzh;
dfrzh = 0.f; // L2020
if ((coms->qh(coms->l) <= 0.f)) { return; } // L2023
if ((coms->qv(coms->l) < coms->qsat(coms->l))) { return; } // L2024
if ((coms->t(coms->l) > tfreeze)) { return; } // L2025
dfrzh = sd_mul(sd_mul(coms->dt,glconst),coms->qh(coms->l)); // L2031
if ((dfrzh < coms->qh(coms->l))) { // L2033
coms->qi(coms->l) = sd_add(coms->qi(coms->l),dfrzh); // L2035
coms->qh(coms->l) = sd_sub(coms->qh(coms->l),dfrzh); // L2036
coms->t(coms->l) = sd_add(coms->t(coms->l),sd_mul(frc,dfrzh)); // L2037
return; // L2040
} else { // L2042
coms->qi(coms->l) = sd_add(coms->qi(coms->l),coms->qh(coms->l)); // L2044
coms->t(coms->l) = sd_add(coms->t(coms->l),sd_mul(frc,coms->qh(coms->l))); // L2045
coms->qh(coms->l) = 0.0f; // L2046
} // L2050
return; // L2052
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:2058-2100 @1d6e370e; SMOKE @9652bbf
SDDEV void melt(State &state,Arena &arena,Plume* coms) {
ArenaScope scope(arena);
float frc = __uint_as_float(0x43A6228FU);
float tmelt = __uint_as_float(0x43888000U);
float f0 = __uint_as_float(0x3F400000U);
float dtmelt;
dtmelt = 0.f; // L2069
if ((coms->qi(coms->l) <= 0.0f)) { return; } // L2072
if ((coms->t(coms->l) < tmelt)) { return; } // L2073
dtmelt = sd_mul(sd_mul(sd_mul(sd_mul(sd_mul(coms->dt,sd_div(2.27f,coms->rho(coms->l))),coms->cvi(coms->l)),sd_sub(coms->t(coms->l),tmelt)),sd_pow(sd_mul(sd_mul(coms->rho(coms->l),coms->qi(coms->l)),1.e-6f),0.525f)),sd_pow(f0,sd_neg(0.42f))); // L2076
if ((dtmelt < coms->qi(coms->l))) { // L2082
coms->qh(coms->l) = sd_add(coms->qh(coms->l),dtmelt); // L2084
coms->qi(coms->l) = sd_sub(coms->qi(coms->l),dtmelt); // L2085
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(frc,dtmelt)); // L2086
return; // L2088
} else { // L2090
coms->qh(coms->l) = sd_add(coms->qh(coms->l),coms->qi(coms->l)); // L2092
coms->t(coms->l) = sd_sub(coms->t(coms->l),sd_mul(frc,coms->qi(coms->l))); // L2093
coms->qi(coms->l) = 0.0f; // L2094
} // L2096
return; // L2098
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:2102-2143 @1d6e370e; SMOKE @9652bbf
SDDEV void htint(State &state,Arena &arena,int nzz1,FArray<float> vctra,FArray<float> eleva,int nzz2,FArray<float> vctrb,FArray<float> elevb,FStr & errmsg,int & errflg) {
ArenaScope scope(arena);
vctra.bind(1,nzz1);
eleva.bind(1,nzz1);
vctrb.bind(1,nzz2);
elevb.bind(1,nzz2);
int l;
int k;
int kk;
float wt;
l = 1; // L2117
for (k=1;k<=nzz2;k+=1) { // L2119
for (;;) { // L2120
if (((elevb(k) < eleva(1)) || ((elevb(k) >= eleva(l)) && (elevb(k) <= eleva((l + 1)))))) { // L2121
wt = sd_div(sd_sub(elevb(k),eleva(l)),sd_sub(eleva((l + 1)),eleva(l))); // L2123
vctrb(k) = sd_add(vctra(l),sd_mul(sd_sub(vctra((l + 1)),vctra(l)),wt)); // L2124
goto loop2_exit; // L2125
} else if ((elevb(k) > eleva(nzz1))) { // L2126
wt = sd_div(sd_sub(elevb(k),eleva(nzz1)),sd_sub(eleva((nzz1 - 1)),eleva(nzz1))); // L2127
vctrb(k) = sd_add(vctra(nzz1),sd_mul(sd_sub(vctra((nzz1 - 1)),vctra(nzz1)),wt)); // L2128
goto loop2_exit; // L2129
} // L2130
l = (l + 1); // L2132
if ((l == nzz1)) { // L2133
; /* Framework diagnostic I/O; no field mutation. */ // L2134
for (kk=1;kk<=l;kk+=1) { // L2135
; /* Framework diagnostic I/O; no field mutation. */ // L2136
loop3_continue: ; } loop3_exit: ; // L2137
errmsg = FStr("htint assertion failure (see print for details)"); // L2138
errflg = 1; // L2139
} // L2140
loop2_continue: ; } loop2_exit: ; // L2141
loop1_continue: ; } loop1_exit: ; // L2142
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_smoke_plumerise.F90:2145-2173 @1d6e370e; SMOKE @9652bbf
SDDEV float esat_pr(State &state,Arena &arena,float tem) {
ArenaScope scope(arena);
float ci1 = __uint_as_float(0x40C39168U);
float ci2 = __uint_as_float(0x41B45604U);
float ci3 = __uint_as_float(0x4388BD71U);
float cw1 = __uint_as_float(0x40C39653U);
float cw2 = __uint_as_float(0x4195D4FEU);
float cw3 = __uint_as_float(0x4380EF5CU);
float cw4 = __uint_as_float(0x43634CCDU);
float tmelt = __uint_as_float(0x4388A666U);
float esat_pr;
float temc;
float esatm;
temc = sd_sub(tem,tmelt); // L2161
if ((temc <= sd_neg(40.0f))) { // L2162
esatm = sd_mul(ci1,sd_exp(sd_div(sd_mul(ci2,temc),sd_add(temc,ci3)))); // L2163
esat_pr = sd_div(esatm,10.f); // L2164
return esat_pr; // L2166
} // L2167
esatm = sd_mul(cw1,sd_exp(sd_div(sd_mul(sd_sub(cw2,sd_div(temc,cw4)),temc),sd_add(temc,cw3)))); // L2169
esat_pr = sd_div(esatm,10.f); // L2171
return esat_pr; // L2172
return esat_pr;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_dry_simple_mod.F90:16-62 @1d6e370e; SMOKE @9652bbf
SDDEV void dry_dep_driver_simple(State &state,Arena &arena,FArray<float> rmol,FArray<float> ust,int ndvel,FArray<float> ddvel,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
rmol.bind(ims,ime,jms,jme);
ust.bind(ims,ime,jms,jme);
ddvel.bind(its,ite,jts,jte,1,ndvel);
float kpart = __uint_as_float(0x43FA0000U);
float dvpart;
int i;
int j;
int k;
int nv;
sd_assign(ddvel.section(Slice(),Slice(),Slice()),0.0f,arena); // L47
for (nv=1;nv<=ndvel;nv+=1) { // L48
for (j=jts;j<=jte;j+=1) { // L49
for (i=its;i<=ite;i+=1) { // L50
dvpart = sd_div(ust(i,j),kpart); // L51
if ((rmol(i,j) < 0.f)) { // L53
dvpart = sd_mul(dvpart,sd_add(1.f,sd_pow(sd_mul(sd_neg(300.f),rmol(i,j)),0.66667f))); // L54
} // L55
ddvel(i,j,nv) = sd_min(0.50f,dvpart); // L57
loop3_continue: ; } loop3_exit: ; // L58
loop2_continue: ; } loop2_exit: ; // L59
loop1_continue: ; } loop1_exit: ; // L60
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/rad_data_mod.F90:17-64 @1d6e370e; SMOKE @9652bbf
SDDEV void aero_rad_init(State &state,Arena &arena) {
ArenaScope scope(arena);
sd_assign(state.rad_data_mod.sc_eff.section(Slice()),0.f,arena); // L20
sd_assign(state.rad_data_mod.ab_eff.section(Slice()),0.f,arena); // L21
if ((state.mpas_smoke_init.p_smoke_fine > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_smoke_fine) = 3.9f; } // L24
if ((state.mpas_smoke_init.p_smoke_coarse > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_smoke_coarse) = 2.6f; } // L25
if ((state.mpas_smoke_init.p_dust_fine > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_dust_fine) = 3.3f; } // L26
if ((state.mpas_smoke_init.p_unspc_fine > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_unspc_fine) = 3.3f; } // L27
if ((state.mpas_smoke_init.p_dust_coarse > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_dust_coarse) = 0.7f; } // L28
if ((state.mpas_smoke_init.p_unspc_coarse > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_unspc_coarse) = 0.7f; } // L29
if ((state.mpas_smoke_init.p_ssalt_fine > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_ssalt_fine) = 4.5f; } // L30
if ((state.mpas_smoke_init.p_ssalt_coarse > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_ssalt_coarse) = 1.0f; } // L31
if ((state.mpas_smoke_init.p_polp_all > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_polp_all) = 2.6f; } // L33
if ((state.mpas_smoke_init.p_polp_tree > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_polp_tree) = 2.6f; } // L34
if ((state.mpas_smoke_init.p_polp_grass > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_polp_grass) = 2.6f; } // L35
if ((state.mpas_smoke_init.p_polp_weed > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_polp_weed) = 2.6f; } // L36
if ((state.mpas_smoke_init.p_pols_all > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_pols_all) = 3.9f; } // L38
if ((state.mpas_smoke_init.p_pols_tree > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_pols_tree) = 3.9f; } // L39
if ((state.mpas_smoke_init.p_pols_grass > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_pols_grass) = 3.9f; } // L40
if ((state.mpas_smoke_init.p_pols_weed > 0)) { state.rad_data_mod.sc_eff(state.mpas_smoke_init.p_pols_weed) = 3.9f; } // L41
if ((state.mpas_smoke_init.p_smoke_fine > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_smoke_fine) = 0.5f; } // L43
if ((state.mpas_smoke_init.p_smoke_coarse > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_smoke_coarse) = 0.0f; } // L44
if ((state.mpas_smoke_init.p_dust_fine > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_dust_fine) = 0.0f; } // L45
if ((state.mpas_smoke_init.p_unspc_fine > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_unspc_fine) = 0.0f; } // L46
if ((state.mpas_smoke_init.p_dust_coarse > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_dust_coarse) = 0.0f; } // L47
if ((state.mpas_smoke_init.p_unspc_coarse > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_unspc_coarse) = 0.0f; } // L48
if ((state.mpas_smoke_init.p_ssalt_fine > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_ssalt_fine) = 0.0f; } // L49
if ((state.mpas_smoke_init.p_ssalt_coarse > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_ssalt_coarse) = 0.0f; } // L50
if ((state.mpas_smoke_init.p_polp_all > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_polp_all) = 0.0f; } // L52
if ((state.mpas_smoke_init.p_polp_tree > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_polp_tree) = 0.0f; } // L53
if ((state.mpas_smoke_init.p_polp_grass > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_polp_grass) = 0.0f; } // L54
if ((state.mpas_smoke_init.p_polp_weed > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_polp_weed) = 0.0f; } // L55
if ((state.mpas_smoke_init.p_pols_all > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_pols_all) = 0.0f; } // L57
if ((state.mpas_smoke_init.p_pols_tree > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_pols_tree) = 0.0f; } // L58
if ((state.mpas_smoke_init.p_pols_grass > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_pols_grass) = 0.0f; } // L59
if ((state.mpas_smoke_init.p_pols_weed > 0)) { state.rad_data_mod.ab_eff(state.mpas_smoke_init.p_pols_weed) = 0.0f; } // L60
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_data_mod.F90:59-105 @1d6e370e; SMOKE @9652bbf
SDDEV void aero_dry_dep_init(State &state,Arena &arena) {
ArenaScope scope(arena);
sd_assign(state.dep_data_mod.aero_dens.section(Slice()),1.e3f,arena); // L63
sd_assign(state.dep_data_mod.aero_diam.section(Slice()),1.e-6f,arena); // L64
if ((state.mpas_smoke_init.p_smoke_fine > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_smoke_fine) = 1.4e3f; } // L67
if ((state.mpas_smoke_init.p_smoke_coarse > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_smoke_coarse) = 1.4e3f; } // L68
if ((state.mpas_smoke_init.p_dust_fine > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_dust_fine) = 2.6e3f; } // L69
if ((state.mpas_smoke_init.p_unspc_fine > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_unspc_fine) = 2.6e3f; } // L70
if ((state.mpas_smoke_init.p_dust_coarse > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_dust_coarse) = 2.6e3f; } // L71
if ((state.mpas_smoke_init.p_unspc_coarse > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_unspc_coarse) = 2.6e3f; } // L72
if ((state.mpas_smoke_init.p_ssalt_fine > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_ssalt_fine) = 2.2e3f; } // L73
if ((state.mpas_smoke_init.p_ssalt_coarse > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_ssalt_coarse) = 2.2e3f; } // L74
if ((state.mpas_smoke_init.p_polp_all > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_polp_all) = 1.2e3f; } // L76
if ((state.mpas_smoke_init.p_polp_tree > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_polp_tree) = 1.2e3f; } // L77
if ((state.mpas_smoke_init.p_polp_grass > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_polp_grass) = 1.0e3f; } // L78
if ((state.mpas_smoke_init.p_polp_weed > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_polp_weed) = 1.3e3f; } // L79
if ((state.mpas_smoke_init.p_pols_all > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_pols_all) = 1.425e3f; } // L81
if ((state.mpas_smoke_init.p_pols_tree > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_pols_tree) = 1.425e3f; } // L82
if ((state.mpas_smoke_init.p_pols_grass > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_pols_grass) = 1.425e3f; } // L83
if ((state.mpas_smoke_init.p_pols_weed > 0)) { state.dep_data_mod.aero_dens(state.mpas_smoke_init.p_pols_weed) = 1.425e3f; } // L84
if ((state.mpas_smoke_init.p_smoke_fine > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_smoke_fine) = 4e-8f; } // L86
if ((state.mpas_smoke_init.p_smoke_coarse > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_smoke_coarse) = 10e-6f; } // L87
if ((state.mpas_smoke_init.p_dust_fine > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_dust_fine) = 1e-6f; } // L88
if ((state.mpas_smoke_init.p_unspc_fine > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_unspc_fine) = 1e-7f; } // L89
if ((state.mpas_smoke_init.p_dust_coarse > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_dust_coarse) = 4.5e-6f; } // L90
if ((state.mpas_smoke_init.p_unspc_coarse > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_unspc_coarse) = 4.5e-6f; } // L91
if ((state.mpas_smoke_init.p_ssalt_fine > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_ssalt_fine) = 6.32e-7f; } // L92
if ((state.mpas_smoke_init.p_ssalt_coarse > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_ssalt_coarse) = 5.632e-6f; } // L93
if ((state.mpas_smoke_init.p_polp_all > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_polp_all) = 30e-6f; } // L95
if ((state.mpas_smoke_init.p_polp_tree > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_polp_tree) = 35e-6f; } // L96
if ((state.mpas_smoke_init.p_polp_grass > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_polp_grass) = 35e-6f; } // L97
if ((state.mpas_smoke_init.p_polp_weed > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_polp_weed) = 20e-6f; } // L98
if ((state.mpas_smoke_init.p_pols_all > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_pols_all) = 1.5e-7f; } // L100
if ((state.mpas_smoke_init.p_pols_tree > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_pols_tree) = 1.5e-7f; } // L101
if ((state.mpas_smoke_init.p_pols_grass > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_pols_grass) = 1.5e-7f; } // L102
if ((state.mpas_smoke_init.p_pols_weed > 0)) { state.dep_data_mod.aero_diam(state.mpas_smoke_init.p_pols_weed) = 1.5e-7f; } // L103
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:24-218 @1d6e370e; SMOKE @9652bbf
SDDEV void gocart_dust_fengsha_driver(State &state,Arena &arena,float dt,int ktau,FArray<float> chem,FArray<float> rho_phy,FArray<float> smois,FArray<float> stemp,FArray<float> p8w,FArray<int> isltyp,FArray<float> snowh,FArray<float> xland,FArray<float> area,float g,FArray<float> ust,FArray<float> znt,FArray<float> clay,FArray<float> sand,FArray<float> uthr,FArray<float> uthr_sg,FArray<float> albedo_drag,FArray<float> feff,FArray<float> sep,FArray<float> e_dust_out,int num_e_dust_out,int index_e_dust_out_dust_fine,int index_e_dust_out_dust_coarse,int num_emis_dust,int num_chem,int num_soil_layers,float dust_alpha,float dust_gamma,float dust_drylimit_factor,float dust_moist_correction,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
smois.bind(ims,ime,1,num_soil_layers,jms,jme);
stemp.bind(ims,ime,1,num_soil_layers,jms,jme);
p8w.bind(ims,ime,kms,kme,jms,jme);
isltyp.bind(ims,ime,jms,jme);
snowh.bind(ims,ime,jms,jme);
xland.bind(ims,ime,jms,jme);
area.bind(ims,ime,jms,jme);
ust.bind(ims,ime,jms,jme);
znt.bind(ims,ime,jms,jme);
clay.bind(ims,ime,jms,jme);
sand.bind(ims,ime,jms,jme);
uthr.bind(ims,ime,jms,jme);
uthr_sg.bind(ims,ime,jms,jme);
albedo_drag.bind(ims,ime,jms,jme);
feff.bind(ims,ime,jms,jme);
sep.bind(ims,ime,jms,jme);
e_dust_out.bind(ims,ime,kms,kme,jms,jme,1,num_e_dust_out);
int nmx;
int i;
int j;
int k;
int imx;
int jmx;
int lmx;
int ilwi;
float airden;
float airmas;
float dxy;
float conver;
float converi;
float r;
float ustar;
auto tc = arena.alloc<float>(1,num_emis_dust);
auto bems = arena.alloc<float>(1,num_emis_dust);
auto distribution = arena.alloc<float>(1,num_emis_dust);
auto massfrac = arena.alloc<float>(1,3);
float erodtot;
float moist_volumetric;
float reason_nodust;
conver = 1.e-9f; // L92
converi = 1.e9f; // L93
imx = 1; // L97
jmx = 1; // L98
lmx = 1; // L99
nmx = state.dust_data_mod.ndust; // L100
k = kts; // L102
for (j=jts;j<=jte;j+=1) { // L103
for (i=its;i<=ite;i+=1) { // L104
ilwi = 0; // L107
if ((sd_sub(xland(i,j),1.5f) < 0.f)) { // L108
ilwi = 1; // L109
tc(1) = 0.f; // L116
tc(2) = 0.f; // L117
tc(3) = 0.f; // L118
tc(4) = 0.f; // L119
tc(5) = 0.f; // L120
airmas = sd_div(sd_mul(sd_neg(sd_sub(p8w(i,(kts + 1),j),p8w(i,kts,j))),area(i,j)),g); // L124
airden = rho_phy(i,kts,j); // L125
ustar = ust(i,j); // L126
dxy = area(i,j); // L127
massfrac(1) = clay(i,j); // L130
massfrac(2) = sd_sub(1.f,sd_add(clay(i,j),sand(i,j))); // L131
massfrac(3) = sand(i,j); // L132
erodtot = sep(i,j); // L137
if ((znt(i,j) > 0.2f)) { // L147
ilwi = 0; // L148
reason_nodust = 0.f; // L149
} // L150
if ((snowh(i,j) > 1.e-6f)) { // L154
ilwi = 0; // L155
reason_nodust = 1.f; // L156
} // L157
if ((stemp(i,1,j) < 268.f)) { // L160
ilwi = 0; // L161
reason_nodust = 2.f; // L162
} // L163
if ((((isltyp(i,j) == 15) || (isltyp(i,j) == 16.f)) || (isltyp(i,j) == 18))) { // L166
ilwi = 0; // L168
reason_nodust = 3.f; // L169
} // L170
if ((isltyp(i,j) == 0)) { // L171
ilwi = 0; // L172
reason_nodust = 4.f; // L173
} // L174
if (((ilwi == 0) && (!(((isltyp(i,j) == 15) || (isltyp(i,j) == 16.f)) || (isltyp(i,j) == 18))))) { // L175
reason_nodust = 5.f; // L177
} // L178
if ((state.dust_data_mod.dust_calcdrag != 1)) { // L183
fengsha_drag(state,arena,znt(i,j),r); // L184
} else { // L185
if ((feff(i,j) > 1.e-10f)) { // L189
r = float(feff(i,j)); // L190
} else { // L191
reason_nodust = 6.f; // L192
goto loop2_continue; // L193
} // L194
} // L195
moist_volumetric = sd_mul(dust_moist_correction,smois(i,2,j)); // L198
sd_assign(bems.section(Slice()),0.f,arena); // L201
source_dust(state,arena,imx,jmx,lmx,nmx,dt,tc,ustar,massfrac,erodtot,dxy,moist_volumetric,airden,airmas,bems,g,dust_alpha,dust_gamma,r,uthr(i,j),reason_nodust,dust_drylimit_factor); // L202
chem(i,kts,j,state.mpas_smoke_init.p_dust_fine) = sd_add(chem(i,kts,j,state.mpas_smoke_init.p_dust_fine),sd_mul(tc(1),converi)); // L208
chem(i,kts,j,state.mpas_smoke_init.p_dust_coarse) = sd_add(chem(i,kts,j,state.mpas_smoke_init.p_dust_coarse),sd_mul(tc(5),converi)); // L209
e_dust_out(i,kts,j,index_e_dust_out_dust_fine) = reason_nodust; // L211
e_dust_out(i,kts,j,index_e_dust_out_dust_coarse) = bems(5); // L212
} // L213
loop2_continue: ; } loop2_exit: ; // L214
loop1_continue: ; } loop1_exit: ; // L215
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:221-394 @1d6e370e; SMOKE @9652bbf
SDDEV void source_dust(State &state,Arena &arena,int imx,int jmx,int lmx,int nmx,float dt1,FArray<float> tc,float ustar,FArray<float> massfrac,float erod,float dxy,float smois,float airden,float airmas,FArray<float> bems,float g0,float alpha,float gamma,float r,float uthres,float & source_out_diag,float dust_drylimit_factor) {
ArenaScope scope(arena);
tc.bind(1,nmx);
massfrac.bind(1,3);
bems.bind(1,nmx);
auto dvol = arena.alloc<float>(1,nmx);
auto distr_dust = arena.alloc<float>(1,nmx);
auto dlndp = arena.alloc<float>(1,nmx);
float dsrc;
float dvol_tot;
float emit;
float emit_vol;
float rhoa;
float reason;
int i;
int j;
int n;
float cmb = __uint_as_float(0x3F800000U);
float kvhmax = __uint_as_float(0x3951B717U);
float mmd_dust = __uint_as_float(0x36642B8EU);
float gsd_dust = __uint_as_float(0x40400000U);
float lambda = __uint_as_float(0x3749539CU);
float cv = __uint_as_float(0x3753BA7DU);
float rhosoil = __uint_as_float(0x4525A000U);
emit = 0.f; // L333
dustemissionfengsha(state,arena,smois,massfrac(1),massfrac(3),massfrac(2),erod,r,airden,ustar,uthres,alpha,gamma,kvhmax,g0,rhosoil,emit,reason,dust_drylimit_factor); // L335
dvol_tot = 0.f; // L352
for (n=1;n<=nmx;n+=1) { // L353
dlndp(n) = sd_log(sd_div(state.dust_data_mod.up_dust(n),state.dust_data_mod.lo_dust(n))); // L354
dvol(n) = sd_mul(sd_mul(sd_mul(sd_div(sd_mul(2.0f,state.dust_data_mod.reff_dust(n)),cv),sd_add(1.f,sd_erf(sd_div(sd_log(sd_div(sd_mul(2.0f,state.dust_data_mod.reff_dust(n)),mmd_dust)),sd_mul(sd_sqrt(2.f),sd_log(gsd_dust)))))),sd_exp(sd_neg(sd_ipow(sd_div(sd_mul(2.0f,state.dust_data_mod.reff_dust(n)),lambda),3)))),dlndp(n)); // L355
dvol_tot = sd_add(dvol_tot,dvol(n)); // L357
loop1_continue: ; } loop1_exit: ; // L360
for (n=1;n<=nmx;n+=1) { // L361
distr_dust(n) = sd_div(dvol(n),dvol_tot); // L362
loop2_continue: ; } loop2_exit: ; // L364
for (n=1;n<=nmx;n+=1) { // L368
dsrc = sd_mul(sd_mul(sd_mul(emit,distr_dust(n)),dxy),dt1); // L370
if ((dsrc < 0.f)) { dsrc = 0.f; } // L371
tc(n) = sd_add(tc(n),sd_div(dsrc,airmas)); // L374
bems(n) = sd_div(sd_mul(1.e+9f,dsrc),sd_mul(dxy,dt1)); // L377
source_out_diag = reason; // L378
loop3_continue: ; } loop3_exit: ; // L380
tc(1) = sd_add(tc(1),sd_mul(0.286f,tc(2))); // L382
tc(5) = sd_add(sd_add(sd_mul(0.714f,tc(2)),tc(3)),tc(4)); // L383
tc(2) = 0.f; // L384
tc(3) = 0.f; // L385
tc(4) = 0.f; // L386
bems(1) = sd_add(bems(1),sd_mul(0.286f,bems(2))); // L388
bems(5) = sd_add(sd_add(sd_mul(0.714f,bems(2)),bems(3)),bems(4)); // L389
bems(2) = 0.f; // L390
bems(3) = 0.f; // L391
bems(4) = 0.f; // L392
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:397-418 @1d6e370e; SMOKE @9652bbf
SDDEV void fengsha_drag(State &state,Arena &arena,float z0,float & r) {
ArenaScope scope(arena);
float z0s = __uint_as_float(0x38D1B717U);
r = sd_sub(1.0f,sd_div(sd_log(sd_div(z0,z0s)),sd_log(sd_mul(0.7f,sd_pow(sd_div(12255.f,z0s),0.8f))))); // L411
return; // L417
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:420-521 @1d6e370e; SMOKE @9652bbf
SDDEV void dustemissionfengsha(State &state,Arena &arena,float slc,float clay,float sand,float silt,float sep,float feff,float airdens,float ustar,float uthrs,float alpha,float gamma,float kvhmax,float grav,float rhop,float & emissions,float & reason,float dust_drylimit_factor) {
ArenaScope scope(arena);
float alpha_grav;
float h;
float kvh;
float q;
float rustar;
float total_emissions;
float u_sum;
float u_thresh;
emissions = 0.f; // L471
alpha_grav = sd_div(alpha,grav); // L475
kvh = dustfluxv2hratiomb95(state,arena,clay,kvhmax); // L479
if (sd_isnan(kvh)) { // L480
; /* Framework diagnostic I/O; no field mutation. */ // L481
kvh = kvhmax; // L482
} // L483
emissions = sd_mul(sd_mul(sd_mul(alpha_grav,sd_pow(sep,gamma)),airdens),kvh); // L487
rustar = sd_mul(feff,ustar); // L491
if ((state.dust_data_mod.dust_moist_opt == 1)) { // L496
h = moisturecorrectionfecan(state,arena,slc,sand,clay,dust_drylimit_factor); // L500
} else { // L501
h = moisturecorrectionshao(state,arena,slc); // L503
} // L504
u_thresh = sd_mul(uthrs,h); // L507
u_sum = sd_add(rustar,u_thresh); // L509
q = sd_mul(sd_mul(sd_max(0.f,sd_sub(rustar,u_thresh)),u_sum),u_sum); // L513
if ((sd_sub(rustar,u_thresh) < 0.f)) { reason = sd_neg(2.f); } // L516
emissions = sd_mul(emissions,q); // L518
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:523-556 @1d6e370e; SMOKE @9652bbf
SDDEV float soilmoistureconvertvol2grav(State &state,Arena &arena,float vsoil,float sandfrac) {
ArenaScope scope(arena);
float vsat;
float rhow = __uint_as_float(0x447A0000U);
float rhop = __uint_as_float(0x44D48000U);
float soilmoistureconvertvol2grav;
vsat = sd_sub(0.489f,sd_mul(0.126f,sandfrac)); // L550
soilmoistureconvertvol2grav = sd_mul(100.f,sd_div(sd_div(sd_mul(vsoil,rhow),rhop),sd_sub(1.f,vsat))); // L554
return soilmoistureconvertvol2grav;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:558-593 @1d6e370e; SMOKE @9652bbf
SDDEV float moisturecorrectionfecan(State &state,Arena &arena,float slc,float sand,float clay,float dust_drylimit_factor) {
ArenaScope scope(arena);
float grvsoilm;
float drylimit;
float moisturecorrectionfecan;
grvsoilm = soilmoistureconvertvol2grav(state,arena,slc,sand); // L585
drylimit = sd_mul(sd_mul(dust_drylimit_factor,clay),sd_add(sd_mul(14.0f,clay),17.0f)); // L588
moisturecorrectionfecan = sd_sqrt(sd_add(1.0f,sd_mul(1.21f,sd_pow(sd_max(0.f,sd_sub(grvsoilm,drylimit)),0.68f)))); // L591
return moisturecorrectionfecan;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:595-624 @1d6e370e; SMOKE @9652bbf
SDDEV float moisturecorrectionshao(State &state,Arena &arena,float slc) {
ArenaScope scope(arena);
float grvsoilm;
float drylimit;
float moisturecorrectionshao;
if ((slc < 0.03f)) { // L618
moisturecorrectionshao = sd_exp(sd_mul(22.7f,slc)); // L619
} else { // L620
moisturecorrectionshao = sd_exp(sd_sub(sd_mul(95.3f,slc),2.029f)); // L621
} // L622
return moisturecorrectionshao;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:626-656 @1d6e370e; SMOKE @9652bbf
SDDEV float dustfluxv2hratiomb95(State &state,Arena &arena,float clay,float kvhmax) {
ArenaScope scope(arena);
float clay_thresh = __uint_as_float(0x3E4CCCCDU);
float dustfluxv2hratiomb95;
if ((clay > clay_thresh)) { // L650
dustfluxv2hratiomb95 = kvhmax; // L651
} else { // L652
dustfluxv2hratiomb95 = sd_pow(10.0f,sd_sub(sd_mul(13.4f,clay),6.0f)); // L653
} // L654
return dustfluxv2hratiomb95;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_wetdep_ls.F90:10-89 @1d6e370e; SMOKE @9652bbf
SDDEV void wetdep_ls(State &state,Arena &arena,float dt,float gravity,FArray<float> var,FArray<float> rain,FArray<float> moist,FArray<float> rho,int nchem,FArray<float> dz8w,FArray<float> vvel,FArray<float> p_phy,float alpha,FArray<float> wetdep_resolved,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
var.bind(ims,ime,kms,kme,jms,jme,1,nchem);
rain.bind(ims,ime,jms,jme);
moist.bind(ims,ime,kms,kme,jms,jme);
rho.bind(ims,ime,kms,kme,jms,jme);
dz8w.bind(ims,ime,kms,kme,jms,jme);
vvel.bind(ims,ime,kms,kme,jms,jme);
p_phy.bind(ims,ime,kms,kme,jms,jme);
wetdep_resolved.bind(ims,ime,jms,jme,1,nchem);
auto var_sum = arena.alloc<float>(its,ite,jts,jte);
auto var_rmv = arena.alloc<float>(its,ite,jts,jte);
auto var_rmvl = arena.alloc<float>(its,ite,kts,kte,jts,jte);
auto frc = arena.alloc<float>(its,ite,jts,jte);
auto var_sum_clw = arena.alloc<float>(its,ite,jts,jte);
auto rain_clw = arena.alloc<float>(its,ite,jts,jte);
float dvar;
float factor;
float clsum;
float factor_diag;
float one_over_dt;
float one_over_gravdt;
int nv;
int i;
int j;
int k;
int km;
int kb;
int kbeg;
one_over_dt = sd_div(1.f,dt); // L37
one_over_gravdt = sd_div(sd_div(1.f,gravity),dt); // L38
for (nv=1;nv<=nchem;nv+=1) { // L40
if ((nv == state.mpas_smoke_init.p_ch4)) { goto loop1_continue; } // L41
for (i=its;i<=ite;i+=1) { // L42
for (j=jts;j<=jte;j+=1) { // L43
var_sum_clw(i,j) = 0.f; // L44
var_sum(i,j) = 0.f; // L45
sd_assign(var_rmvl.section(i,Slice(),j),0.f,arena); // L46
frc(i,j) = 0.f; // L47
rain_clw(i,j) = 0.f; // L48
if ((rain(i,j) > 1.e-10f)) { // L49
rain_clw(i,j) = sd_mul(rain(i,j),one_over_dt); // L52
for (k=1;k<=(kte - 1);k+=1) { // L55
dvar = sd_max(0.f,sd_mul(sd_mul(sd_mul(moist(i,k,j),rho(i,k,j)),vvel(i,k,j)),dz8w(i,k,j))); // L56
var_sum_clw(i,j) = sd_add(var_sum_clw(i,j),dvar); // L57
var_sum(i,j) = sd_add(var_sum(i,j),sd_mul(var(i,k,j,nv),rho(i,k,j))); // L58
loop4_continue: ; } loop4_exit: ; // L59
if (((var_sum(i,j) > 1.e-10f) && (var_sum_clw(i,j) > 1.e-10f))) { // L60
frc(i,j) = sd_div(rain_clw(i,j),var_sum_clw(i,j)); // L63
frc(i,j) = sd_max(1.e-6f,sd_min(frc(i,j),.005f)); // L64
} // L65
} // L66
loop3_continue: ; } loop3_exit: ; // L67
loop2_continue: ; } loop2_exit: ; // L68
for (i=its;i<=ite;i+=1) { // L72
for (j=jts;j<=jte;j+=1) { // L73
if ((((rain(i,j) > 1.e-10f) && (var_sum(i,j) > 1.e-10f)) && (var_sum_clw(i,j) > 1.e-10f))) { // L74
for (k=kts;k<=(kte - 2);k+=1) { // L75
if (((var(i,k,j,nv) > 1.e-16f) && (moist(i,k,j) > 0.f))) { // L76
factor = sd_max(0.f,sd_mul(sd_mul(sd_mul(frc(i,j),rho(i,k,j)),dz8w(i,k,j)),vvel(i,k,j))); // L77
dvar = sd_mul(sd_div(sd_mul(alpha,factor),sd_add(1,factor)),var(i,k,j,nv)); // L78
factor_diag = sd_mul(sd_mul(sd_mul(dvar,sd_sub(p_phy(i,(k + 1),j),p_phy(i,k,j))),100.f),one_over_gravdt); // L79
wetdep_resolved(i,j,nv) = sd_add(wetdep_resolved(i,j,nv),factor_diag); // L81
var(i,k,j,nv) = sd_max(1.e-16f,sd_sub(var(i,k,j,nv),dvar)); // L82
} // L83
loop7_continue: ; } loop7_exit: ; // L84
} // L85
loop6_continue: ; } loop6_exit: ; // L86
loop5_continue: ; } loop5_exit: ; // L87
loop1_continue: ; } loop1_exit: ; // L88
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:43-90 @1d6e370e; SMOKE @9652bbf
SDDEV void set_scalar_indices(State &state,Arena &arena,int chemistry_start,int index_smoke_fine,int index_smoke_coarse,int index_polp_tree,int index_polp_grass,int index_polp_weed,int index_pols_tree,int index_pols_grass,int index_pols_weed,int index_pols_all,int index_polp_all,int index_unspc_fine,int index_unspc_coarse,int index_ssalt_fine,int index_ssalt_coarse,int index_no3_a_fine,int index_so4_a_fine,int index_nh4_a_fine,int index_so2,int index_nh3,int index_ch4,int index_dust_fine,int index_dust_coarse,bool has_chemistry_start,bool has_index_smoke_fine,bool has_index_smoke_coarse,bool has_index_polp_tree,bool has_index_polp_grass,bool has_index_polp_weed,bool has_index_pols_tree,bool has_index_pols_grass,bool has_index_pols_weed,bool has_index_pols_all,bool has_index_polp_all,bool has_index_unspc_fine,bool has_index_unspc_coarse,bool has_index_ssalt_fine,bool has_index_ssalt_coarse,bool has_index_no3_a_fine,bool has_index_so4_a_fine,bool has_index_nh4_a_fine,bool has_index_so2,bool has_index_nh3,bool has_index_ch4,bool has_index_dust_fine,bool has_index_dust_coarse) {
ArenaScope scope(arena);
if (has_index_smoke_fine) { state.mpas_smoke_init.p_smoke_fine = ((index_smoke_fine - chemistry_start) + 1); } // L67
if (has_index_smoke_coarse) { state.mpas_smoke_init.p_smoke_coarse = ((index_smoke_coarse - chemistry_start) + 1); } // L68
if (has_index_dust_fine) { state.mpas_smoke_init.p_dust_fine = ((index_dust_fine - chemistry_start) + 1); } // L69
if (has_index_dust_coarse) { state.mpas_smoke_init.p_dust_coarse = ((index_dust_coarse - chemistry_start) + 1); } // L70
if (has_index_polp_all) { state.mpas_smoke_init.p_polp_all = ((index_polp_all - chemistry_start) + 1); } // L71
if (has_index_polp_tree) { state.mpas_smoke_init.p_polp_tree = ((index_polp_tree - chemistry_start) + 1); } // L72
if (has_index_polp_grass) { state.mpas_smoke_init.p_polp_grass = ((index_polp_grass - chemistry_start) + 1); } // L73
if (has_index_polp_weed) { state.mpas_smoke_init.p_polp_weed = ((index_polp_weed - chemistry_start) + 1); } // L74
if (has_index_pols_tree) { state.mpas_smoke_init.p_pols_tree = ((index_pols_tree - chemistry_start) + 1); } // L75
if (has_index_pols_grass) { state.mpas_smoke_init.p_pols_grass = ((index_pols_grass - chemistry_start) + 1); } // L76
if (has_index_pols_weed) { state.mpas_smoke_init.p_pols_weed = ((index_pols_weed - chemistry_start) + 1); } // L77
if (has_index_pols_all) { state.mpas_smoke_init.p_pols_all = ((index_pols_all - chemistry_start) + 1); } // L78
if (has_index_unspc_fine) { state.mpas_smoke_init.p_unspc_fine = ((index_unspc_fine - chemistry_start) + 1); } // L79
if (has_index_unspc_coarse) { state.mpas_smoke_init.p_unspc_coarse = ((index_unspc_coarse - chemistry_start) + 1); } // L80
if (has_index_ssalt_fine) { state.mpas_smoke_init.p_ssalt_fine = ((index_ssalt_fine - chemistry_start) + 1); } // L81
if (has_index_ssalt_coarse) { state.mpas_smoke_init.p_ssalt_coarse = ((index_ssalt_coarse - chemistry_start) + 1); } // L82
if (has_index_no3_a_fine) { state.mpas_smoke_init.p_no3_a_fine = ((index_no3_a_fine - chemistry_start) + 1); } // L83
if (has_index_so4_a_fine) { state.mpas_smoke_init.p_so4_a_fine = ((index_so4_a_fine - chemistry_start) + 1); } // L84
if (has_index_nh4_a_fine) { state.mpas_smoke_init.p_nh4_a_fine = ((index_nh4_a_fine - chemistry_start) + 1); } // L85
if (has_index_so2) { state.mpas_smoke_init.p_so2 = ((index_so2 - chemistry_start) + 1); } // L86
if (has_index_nh3) { state.mpas_smoke_init.p_nh3 = ((index_nh3 - chemistry_start) + 1); } // L87
if (has_index_ch4) { state.mpas_smoke_init.p_ch4 = ((index_ch4 - chemistry_start) + 1); } // L88
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/seas_ngac_mod.F90:58-162 @1d6e370e; SMOKE @9652bbf
SDDEV void seasaltemission(State &state,Arena &arena,float rlow,float rup,int method,float w10m,float ustar,float pi,float & memissions,float & nemissions,int & rc) {
ArenaScope scope(arena);
int ir;
float w;
float r;
float dr;
float rwet;
float drwet;
float afac;
float bfac;
float scalefac;
float rpow;
float exppow;
float wpow;
int nr = 10;
FStr myname = FStr("SeasaltEmission");
dr = sd_div(sd_sub(rup,rlow),nr); // L101
r = sd_add(rlow,sd_mul(0.5f,dr)); // L102
nemissions = 0.f; // L105
memissions = 0.f; // L106
for (ir=1;ir<=nr;ir+=1) { // L108
rwet = sd_mul(state.seas_ngac_mod.r80fac,r); // L110
drwet = sd_mul(state.seas_ngac_mod.r80fac,dr); // L111
{ auto sel1 = method; // L113
if (sel1 == 1) { // L115
afac = sd_mul(4.7f,sd_pow(sd_add(1.f,sd_mul(30.f,rwet)),sd_mul(sd_neg(0.017f),sd_pow(rwet,sd_neg(1.44f))))); // L116
bfac = sd_div(sd_sub(0.433f,sd_log10(rwet)),0.433f); // L117
scalefac = 1.f; // L118
rpow = 3.45f; // L119
exppow = 1.607f; // L120
wpow = 3.41f; // L121
w = w10m; // L122
} else if (sel1 == 2) { // L124
afac = 3.f; // L125
bfac = sd_div(sd_sub(0.380f,sd_log10(rwet)),0.650f); // L126
scalefac = 1.f; // L127
rpow = 1.05f; // L128
exppow = 1.19f; // L129
wpow = 3.41f; // L130
w = w10m; // L131
} else if (sel1 == 3) { // L133
afac = sd_mul(4.7f,sd_pow(sd_add(1.f,sd_mul(30.f,rwet)),sd_mul(sd_neg(0.017f),sd_pow(rwet,sd_neg(1.44f))))); // L134
bfac = sd_div(sd_sub(0.433f,sd_log10(rwet)),0.433f); // L135
scalefac = 33.0e3f; // L136
rpow = 3.45f; // L137
exppow = 1.607f; // L138
wpow = sd_sub(3.41f,1.f); // L139
w = ustar; // L140
} else { // L142
rc = 1; // L144
return; // L145
} } // L147
nemissions = sd_add(nemissions,seasaltemissiongong(state,arena,rwet,drwet,w,scalefac,afac,bfac,rpow,exppow,wpow)); // L151
scalefac = sd_mul(sd_mul(sd_mul(sd_mul(sd_div(sd_mul(scalefac,4.f),3.f),pi),state.seas_ngac_mod.rhop),sd_ipow(r,3)),1.e-18f); // L153
memissions = sd_add(memissions,seasaltemissiongong(state,arena,rwet,drwet,w,scalefac,afac,bfac,rpow,exppow,wpow)); // L154
r = sd_add(r,dr); // L156
loop1_continue: ; } loop1_exit: ; // L158
rc = 0; // L160
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/seas_ngac_mod.F90:170-186 @1d6e370e; SMOKE @9652bbf
SDDEV float seasaltemissiongong(State &state,Arena &arena,float r,float dr,float w,float scalefac,float afac,float bfac,float rpow,float exppow,float wpow) {
ArenaScope scope(arena);
float seasaltemissiongong;
seasaltemissiongong = 0.f; // L178
seasaltemissiongong = sd_mul(sd_mul(sd_mul(sd_mul(sd_mul(scalefac,1.373f),sd_pow(r,sd_neg(afac))),sd_add(1.f,sd_mul(0.057f,sd_pow(r,rpow)))),sd_pow(10,sd_mul(exppow,sd_exp(sd_neg(sd_ipow(bfac,2)))))),dr); // L181
seasaltemissiongong = sd_mul(sd_pow(w,wpow),seasaltemissiongong); // L184
return seasaltemissiongong;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90:66-74 @1d6e370e; SMOKE @9652bbf
SDDEV Plume* get_thread_coms(State &state,Arena &arena) {
ArenaScope scope(arena);
Plume* coms;
if ((!(state.module_zero_plumegen_coms.private_thread_coms != nullptr))) { // L69
state.module_zero_plumegen_coms.private_thread_coms = state.plume_store; // L70
plumegen_coms_zero(state,arena,state.module_zero_plumegen_coms.private_thread_coms); // L71
} // L72
coms = state.module_zero_plumegen_coms.private_thread_coms; // L73
return coms;
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90:76-186 @1d6e370e; SMOKE @9652bbf
SDDEV void plumegen_coms_zero(State &state,Arena &arena,Plume* f_this) {
ArenaScope scope(arena);
f_this->initialized = false; // L80
sd_assign(f_this->w,0.0f,arena); // L82
sd_assign(f_this->t,0.0f,arena); // L83
sd_assign(f_this->qv,0.0f,arena); // L84
sd_assign(f_this->qc,0.0f,arena); // L85
sd_assign(f_this->qh,0.0f,arena); // L86
sd_assign(f_this->qi,0.0f,arena); // L87
sd_assign(f_this->sc,0.0f,arena); // L88
sd_assign(f_this->vth,0.0f,arena); // L89
sd_assign(f_this->vti,0.0f,arena); // L90
sd_assign(f_this->rho,0.0f,arena); // L91
sd_assign(f_this->txs,0.0f,arena); // L92
sd_assign(f_this->est,0.0f,arena); // L93
sd_assign(f_this->qsat,0.0f,arena); // L94
sd_assign(f_this->wc,0.0f,arena); // L97
sd_assign(f_this->wt,0.0f,arena); // L98
sd_assign(f_this->tt,0.0f,arena); // L99
sd_assign(f_this->qvt,0.0f,arena); // L100
sd_assign(f_this->qct,0.0f,arena); // L101
sd_assign(f_this->qht,0.0f,arena); // L102
sd_assign(f_this->qit,0.0f,arena); // L103
sd_assign(f_this->sct,0.0f,arena); // L104
sd_assign(f_this->dzm,0.0f,arena); // L105
sd_assign(f_this->dzt,0.0f,arena); // L106
sd_assign(f_this->zm,0.0f,arena); // L107
sd_assign(f_this->zt,0.0f,arena); // L108
sd_assign(f_this->vctr1,0.0f,arena); // L109
sd_assign(f_this->vctr2,0.0f,arena); // L110
sd_assign(f_this->vt3dc,0.0f,arena); // L111
sd_assign(f_this->vt3df,0.0f,arena); // L112
sd_assign(f_this->vt3dk,0.0f,arena); // L113
sd_assign(f_this->vt3dg,0.0f,arena); // L114
sd_assign(f_this->scr1,0.0f,arena); // L115
sd_assign(f_this->pke,0.0f,arena); // L116
sd_assign(f_this->the,0.0f,arena); // L117
sd_assign(f_this->thve,0.0f,arena); // L118
sd_assign(f_this->thee,0.0f,arena); // L119
sd_assign(f_this->pe,0.0f,arena); // L120
sd_assign(f_this->te,0.0f,arena); // L121
sd_assign(f_this->qvenv,0.0f,arena); // L122
sd_assign(f_this->dne,0.0f,arena); // L124
sd_assign(f_this->ucon,0.0f,arena); // L126
sd_assign(f_this->vcon,0.0f,arena); // L127
sd_assign(f_this->thtcon,0.0f,arena); // L129
sd_assign(f_this->rvcon,0.0f,arena); // L130
sd_assign(f_this->picon,0.0f,arena); // L131
sd_assign(f_this->tmpcon,0.0f,arena); // L132
sd_assign(f_this->zcon,0.0f,arena); // L135
sd_assign(f_this->zzcon,0.0f,arena); // L136
f_this->dz = 0.0f; // L138
f_this->dqsdz = 0.0f; // L139
sd_assign(f_this->visc,0.0f,arena); // L140
f_this->viscosity = 0.0f; // L141
f_this->tstpf = 0.0f; // L142
sd_assign(f_this->cvh,0.0f,arena); // L149
sd_assign(f_this->cvi,0.0f,arena); // L150
f_this->adiabat = 0.0f; // L151
f_this->wbar = 0.0f; // L152
f_this->vhrel = 0.0f; // L154
f_this->virel = 0.0f; // L155
f_this->zsurf = 0.0f; // L156
f_this->ztop = 0.0f; // L158
f_this->area = 0.0f; // L159
f_this->rsurf = 0.0f; // L160
f_this->alpha = 0.0f; // L161
sd_assign(f_this->radius,0.0f,arena); // L162
sd_assign(f_this->heating,0.0f,arena); // L163
f_this->fmoist = 0.0f; // L164
f_this->bload = 0.0f; // L165
f_this->dt = 0.0f; // L166
f_this->time = 0.0f; // L167
f_this->tdur = 0.0f; // L168
sd_assign(f_this->ztop_,0.0f,arena); // L169
sd_assign(f_this->upe,0.0f,arena); // L170
sd_assign(f_this->vpe,0.0f,arena); // L171
sd_assign(f_this->vel_e,0.0f,arena); // L172
sd_assign(f_this->vel_p,0.0f,arena); // L173
sd_assign(f_this->rad_p,0.0f,arena); // L174
sd_assign(f_this->vel_t,0.0f,arena); // L175
sd_assign(f_this->rad_t,0.0f,arena); // L176
f_this->n = 0; // L179
f_this->nm1 = 0; // L180
f_this->l = 0; // L181
f_this->mintime = 0; // L183
f_this->mdur = 0; // L184
f_this->maxtime = 0; // L185
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_anthro_emissions.F90:21-83 @1d6e370e; SMOKE @9652bbf
SDDEV void mpas_smoke_anthro_emis_driver(State &state,Arena &arena,float dt,float gmt,int julday,int kemit,FArray<float> xlat,FArray<float> xlong,FArray<float> chem,int num_chem,FArray<float> dz8w,FArray<float> t_phy,FArray<float> rho_phy,FArray<float> e_ant_in,FArray<float> e_ant_out,int num_e_ant_in,int num_e_ant_out,int index_e_ant_in_unspc_fine,int index_e_ant_in_unspc_coarse,int index_e_ant_out_unspc_fine,int index_e_ant_out_unspc_coarse,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte) {
ArenaScope scope(arena);
xlat.bind(ims,ime,jms,jme);
xlong.bind(ims,ime,jms,jme);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
dz8w.bind(ims,ime,kms,kme,jms,jme);
t_phy.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
e_ant_in.bind(ims,ime,1,kemit,jms,jme,1,num_e_ant_in);
e_ant_out.bind(ims,ime,kms,kme,jms,jme,1,num_e_ant_out);
int i;
int j;
int k;
int n;
float conv_aer;
float conv_gas;
float emis;
float rwc_t_thresh = __uint_as_float(0x438D9333U);
for (j=jts;j<=jte;j+=1) { // L55
for (k=kts;k<=kemit;k+=1) { // L56
for (i=its;i<=ite;i+=1) { // L57
conv_aer = sd_div(dt,sd_mul(rho_phy(i,k,j),dz8w(i,k,j))); // L60
conv_gas = sd_div(sd_mul(sd_mul(sd_mul(60.f,1.e6f),4.828e-4f),dt),sd_mul(rho_phy(i,k,j),dz8w(i,k,j))); // L62
if (((state.mpas_smoke_init.p_unspc_fine > 0) && (index_e_ant_in_unspc_fine > 0))) { // L64
emis = sd_mul(conv_aer,e_ant_in(i,k,j,index_e_ant_in_unspc_fine)); // L65
chem(i,k,j,state.mpas_smoke_init.p_unspc_fine) = sd_add(chem(i,k,j,state.mpas_smoke_init.p_unspc_fine),emis); // L66
e_ant_out(i,k,j,index_e_ant_out_unspc_fine) = sd_add(e_ant_out(i,k,j,index_e_ant_out_unspc_fine),emis); // L67
} // L68
if ((index_e_ant_in_unspc_coarse > 0)) { // L69
emis = sd_mul(conv_aer,e_ant_in(i,k,j,index_e_ant_in_unspc_coarse)); // L70
if ((state.mpas_smoke_init.p_unspc_coarse > 0)) { // L71
chem(i,k,j,state.mpas_smoke_init.p_unspc_coarse) = sd_add(chem(i,k,j,state.mpas_smoke_init.p_unspc_coarse),emis); // L72
} else if ((state.mpas_smoke_init.p_dust_coarse > 0)) { // L73
chem(i,k,j,state.mpas_smoke_init.p_dust_coarse) = sd_add(chem(i,k,j,state.mpas_smoke_init.p_dust_coarse),emis); // L74
} // L75
e_ant_out(i,k,j,index_e_ant_out_unspc_coarse) = sd_add(e_ant_out(i,k,j,index_e_ant_out_unspc_coarse),emis); // L76
} // L77
loop3_continue: ; } loop3_exit: ; // L79
loop2_continue: ; } loop2_exit: ; // L80
loop1_continue: ; } loop1_exit: ; // L81
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_dry_mod_emerson.F90:23-229 @1d6e370e; SMOKE @9652bbf
SDDEV void dry_dep_driver_emerson(State &state,Arena &arena,FArray<float> rmol,FArray<float> ustar,FArray<float> znt,int num_chem,FArray<float> ddvel,FArray<float> chem,FArray<float> delz,FArray<float> snowh,FArray<float> t_phy,FArray<float> p_phy,FArray<float> rho_phy,FArray<int> ivgtyp,float gravity,float dt,FArray<float> drydep_flux,FArray<float> tend_chem_settle,bool dbg_opt,int settling_opt,FArray<float> vg,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,float curr_secs) {
ArenaScope scope(arena);
rmol.bind(ims,ime,jms,jme);
ustar.bind(ims,ime,jms,jme);
znt.bind(ims,ime,jms,jme);
ddvel.bind(ims,ime,jms,jme,1,num_chem);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
delz.bind(ims,ime,kms,kme,jms,jme);
snowh.bind(ims,ime,jms,jme);
t_phy.bind(ims,ime,kms,kme,jms,jme);
p_phy.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
ivgtyp.bind(ims,ime,jms,jme);
drydep_flux.bind(ims,ime,jms,jme,1,num_chem);
tend_chem_settle.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
vg.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
auto airkinvisc = arena.alloc<float>(its,ite,kts,kte,jts,jte);
auto freepath = arena.alloc<float>(its,ite,kts,kte,jts,jte);
auto aer_res = arena.alloc<float>(its,ite,jts,jte);
auto a = arena.alloc<float>(its,ite,jts,jte);
auto eps0 = arena.alloc<float>(its,ite,jts,jte);
float cc;
float ddp;
float eb;
float eim;
float ein;
float sc;
float st;
float dp;
float aerodens;
float rs;
float growth_fac;
float vsettl;
float dtmax;
float conver;
float converi;
float dzmin;
float rmol_local;
int i;
int j;
int k;
int ntdt;
int nv;
auto &icall = state.saved_dry_dep_driver_emerson_icall;
float rsi = __uint_as_float(0x4105083CU);
float four_ninths;
float g100;
bool do_timing = true;
int max_iter_settle = 5;
float ust;
sd_assign(tend_chem_settle.section(Slice(),Slice(),Slice(),Slice()),0.f,arena); // L88
sd_assign(ddvel.section(Slice(),Slice(),Slice()),0.f,arena); // L89
sd_assign(vg.section(Slice(),Slice(),Slice(),Slice()),0.f,arena); // L90
conver = 1.e-9f; // L92
converi = 1.e9f; // L93
g100 = sd_mul(gravity,1.e2f); // L95
ntdt = int(dt); // L96
if ((sd_mod(int(curr_secs),1800) == 0)) { // L98
icall = 0; // L99
} // L100
for (j=jts;j<=jte;j+=1) { // L108
for (i=its;i<=ite;i+=1) { // L109
if (((ivgtyp(i,j) == 13) || (ivgtyp(i,j) <= 5))) { // L110
a(i,j) = state.dep_data_mod.a_for; // L111
eps0(i,j) = state.dep_data_mod.eps0_for; // L112
} else if ((ivgtyp(i,j) == 17)) { // L113
a(i,j) = state.dep_data_mod.a_wat; // L114
eps0(i,j) = state.dep_data_mod.eps0_wat; // L115
} else { // L116
a(i,j) = state.dep_data_mod.a_grs; // L117
eps0(i,j) = state.dep_data_mod.eps0_grs; // L118
} // L119
rmol_local = rmol(i,j); // L120
depvel(state,arena,rmol_local,state.dep_data_mod.dep_ref_hgt,znt(i,j),ustar(i,j),aer_res(i,j)); // L121
loop2_continue: ; } loop2_exit: ; // L122
loop1_continue: ; } loop1_exit: ; // L123
for (j=jts;j<=jte;j+=1) { // L127
for (k=kts;k<=kte;k+=1) { // L128
for (i=its;i<=ite;i+=1) { // L129
airkinvisc(i,k,j) = sd_div(sd_mul(sd_mul(1.8325e-4f,sd_div(416.16f,sd_add(t_phy(i,k,j),120.f))),sd_pow(sd_div(t_phy(i,k,j),296.16f),1.5f)),rho_phy(i,k,j)); // L130
freepath(i,k,j) = sd_div(sd_mul(7.39758e-4f,airkinvisc(i,k,j)),sd_sqrt(t_phy(i,k,j))); // L133
loop5_continue: ; } loop5_exit: ; // L135
loop4_continue: ; } loop4_exit: ; // L136
loop3_continue: ; } loop3_exit: ; // L137
for (nv=1;nv<=num_chem;nv+=1) { // L163
if ((nv == state.mpas_smoke_init.p_ch4)) { goto loop6_continue; } // L164
dp = sd_mul(state.dep_data_mod.aero_diam(nv),100.f); // L166
aerodens = sd_mul(state.dep_data_mod.aero_dens(nv),1.e-3f); // L167
for (j=jts;j<=jte;j+=1) { // L168
for (k=kts;k<=kte;k+=1) { // L169
for (i=its;i<=ite;i+=1) { // L170
cc = sd_add(1.f,sd_mul(sd_div(sd_mul(2.f,freepath(i,k,j)),dp),sd_add(1.257f,sd_mul(0.4f,sd_exp(sd_div(sd_mul(sd_neg(0.55f),dp),freepath(i,k,j))))))); // L172
vg(i,k,j,nv) = sd_div(sd_mul(sd_mul(sd_mul(sd_mul(aerodens,dp),dp),g100),cc),sd_mul(sd_mul(18.f,airkinvisc(i,k,j)),rho_phy(i,k,j))); // L174
loop9_continue: ; } loop9_exit: ; // L176
loop8_continue: ; } loop8_exit: ; // L177
loop7_continue: ; } loop7_exit: ; // L178
loop6_continue: ; } loop6_exit: ; // L179
k = kts; // L182
for (nv=1;nv<=num_chem;nv+=1) { // L183
if ((nv == state.mpas_smoke_init.p_ch4)) { goto loop10_continue; } // L184
dp = sd_mul(state.dep_data_mod.aero_diam(nv),100.f); // L186
aerodens = sd_mul(state.dep_data_mod.aero_dens(nv),1.e-3f); // L187
for (j=jts;j<=jte;j+=1) { // L188
for (i=its;i<=ite;i+=1) { // L189
ust = sd_mul(ustar(i,j),1.e2f); // L190
cc = sd_add(1.f,sd_mul(sd_div(sd_mul(2.f,freepath(i,k,j)),dp),sd_add(1.257f,sd_mul(0.4f,sd_exp(sd_div(sd_mul(sd_neg(0.55f),dp),freepath(i,k,j))))))); // L192
ddp = sd_div(sd_mul(sd_mul(state.dep_data_mod.boltzmann,t_phy(i,k,j)),cc),sd_mul(sd_mul(sd_mul(sd_mul(3.f,state.dep_data_mod.pi),airkinvisc(i,k,j)),rho_phy(i,k,j)),dp)); // L194
sc = sd_div(airkinvisc(i,k,j),ddp); // L196
eb = sd_mul(state.dep_data_mod.cb,sd_pow(sc,sd_neg(0.666666667f))); // L198
st = sd_div(sd_div(sd_mul(sd_mul(ust,ust),vg(i,k,j,nv)),airkinvisc(i,k,j)),g100); // L200
eim = sd_mul(state.dep_data_mod.cim,sd_pow(sd_div(st,sd_add(state.dep_data_mod.alpha,st)),1.7f)); // L202
ein = sd_mul(state.dep_data_mod.cin,sd_pow(sd_div(dp,a(i,j)),state.dep_data_mod.vv)); // L204
rs = sd_div(1.f,sd_mul(sd_mul(ust,sd_add(sd_add(eb,eim),ein)),eps0(i,j))); // L206
if ((settling_opt > 0)) { // L209
ddvel(i,j,nv) = sd_max(sd_min(sd_mul(sd_add(vg(i,k,j,nv),sd_div(1.f,sd_add(aer_res(i,j),rs))),1.e-2f),state.dep_data_mod.max_dep_vel),0.f); // L210
} else { // L211
ddvel(i,j,nv) = sd_max(sd_min(sd_mul(sd_div(1.f,sd_add(aer_res(i,j),rs)),1.e-2f),state.dep_data_mod.max_dep_vel),0.f); // L212
} // L213
if ((dbg_opt && (icall <= state.mpas_smoke_config.n_dbg_lines))) { // L214
icall = (icall + 1); // L215
} // L216
drydep_flux(i,j,nv) = sd_add(drydep_flux(i,j,nv),sd_mul(sd_mul(sd_mul(sd_mul(chem(i,k,j,nv),rho_phy(i,k,j)),ddvel(i,j,nv)),dt),10.f)); // L217
loop12_continue: ; } loop12_exit: ; // L218
loop11_continue: ; } loop11_exit: ; // L219
loop10_continue: ; } loop10_exit: ; // L220
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_dry_mod_emerson.F90:233-298 @1d6e370e; SMOKE @9652bbf
SDDEV void depvel(State &state,Arena &arena,float & rmol,float zr,float z0,float ustar,float & aer_res) {
ArenaScope scope(arena);
int l;
float ao;
float ar;
float polint;
float vk;
vk = state.dep_data_mod.karman; // L279
if ((sd_abs(rmol) < 1.e-6f)) { rmol = 0.f; } // L286
if ((rmol < 0.f)) { // L287
ar = sd_ipow(sd_add(sd_pow(sd_sub(1.f,sd_mul(sd_mul(9.f,zr),rmol)),0.25f),0.001f),2); // L288
ao = sd_ipow(sd_add(sd_pow(sd_sub(1.f,sd_mul(sd_mul(9.f,z0),rmol)),0.25f),0.001f),2); // L289
polint = sd_mul(0.74f,sd_sub(sd_log(sd_div(sd_sub(ar,1.f),sd_add(ar,1.f))),sd_log(sd_div(sd_sub(ao,1.f),sd_add(ao,1.f))))); // L290
} else if ((rmol == 0.f)) { // L291
polint = sd_mul(0.74f,sd_log(sd_div(zr,z0))); // L292
} else { // L293
polint = sd_add(sd_mul(0.74f,sd_log(sd_div(zr,z0))),sd_mul(sd_mul(4.7f,rmol),sd_sub(zr,z0))); // L294
} // L295
aer_res = sd_mul(sd_div(polint,sd_mul(state.dep_data_mod.karman,sd_max(ustar,1.0e-4f))),1.e-2f); // L297
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dep_dry_mod_emerson.F90:350-419 @1d6e370e; SMOKE @9652bbf
SDDEV void particle_settling_wrapper(State &state,Arena &arena,FArray<float> tend_chem_settle,FArray<float> chem,FArray<float> rho_phy,FArray<float> delz_flip,FArray<float> vg,float dt,int kts,int kte,int its,int ite,int jts,int jte,int num_chem,int ims,int ime,int jms,int jme,int kms,int kme) {
ArenaScope scope(arena);
tend_chem_settle.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
delz_flip.bind(ims,ime,kms,kme,jms,jme);
vg.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
float dt_settl;
float growth_fac;
float four_ninths;
float dzmin;
float vsettl;
float dtmax;
int ntdt;
int ndt_settl;
int max_iter_settle = 10;
float one_over_dyn_visc = __uint_as_float(0x47C35000U);
int k;
int n;
int l2;
int i;
int j;
int nv;
float temp_tc;
float transfer_to_below_level;
float vd_wrk1;
growth_fac = 1.f; // L371
four_ninths = sd_div(4.f,9.f); // L372
ntdt = int(dt); // L373
for (nv=1;nv<=num_chem;nv+=1) { // L375
if ((nv == state.mpas_smoke_init.p_ch4)) { goto loop1_continue; } // L376
vsettl = sd_mul(sd_mul(sd_mul(sd_mul(four_ninths,state.dep_data_mod.gravity),state.dep_data_mod.aero_dens(nv)),sd_ipow(sd_mul(growth_fac,sd_mul(0.5f,state.dep_data_mod.aero_diam(nv))),2)),one_over_dyn_visc); // L378
for (j=jte;j<=jte;j+=1) { // L380
for (i=its;i<=ite;i+=1) { // L381
dzmin = delz_flip(i,kte,j); // L383
dtmax = sd_div(dzmin,vsettl); // L389
ndt_settl = sd_max(1,int(sd_div(ntdt,dtmax))); // L391
if ((ndt_settl > max_iter_settle)) { ndt_settl = max_iter_settle; } // L393
dt_settl = sd_div(float(ntdt),float(ndt_settl)); // L394
for (n=1;n<=ndt_settl;n+=1) { // L396
transfer_to_below_level = 0.f; // L397
for (k=kte;k>=kts;k+=(-1)) { // L399
l2 = ((kte - k) + 1); // L401
temp_tc = chem(i,k,j,nv); // L403
vd_wrk1 = sd_div(sd_mul(sd_mul(dt_settl,vg(i,k,j,nv)),1.e-2f),delz_flip(i,l2,j)); // L405
temp_tc = sd_add(sd_mul(temp_tc,sd_sub(1.f,vd_wrk1)),transfer_to_below_level); // L407
if ((k > kts)) { // L408
transfer_to_below_level = sd_mul(sd_mul(chem(i,k,j,nv),vd_wrk1),sd_div(sd_mul(delz_flip(i,l2,j),rho_phy(i,k,j)),sd_mul(delz_flip(i,(l2 + 1),j),rho_phy(i,(k - 1),j)))); // L409
} // L411
tend_chem_settle(i,k,j,nv) = sd_add(tend_chem_settle(i,k,j,nv),sd_sub(temp_tc,chem(i,k,j,nv))); // L412
loop5_continue: ; } loop5_exit: ; // L414
loop4_continue: ; } loop4_exit: ; // L415
loop3_continue: ; } loop3_exit: ; // L416
loop2_continue: ; } loop2_exit: ; // L417
loop1_continue: ; } loop1_exit: ; // L418
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:500-774 @1d6e370e; SMOKE @9652bbf
SDDEV void smoke_from_mpas_fields(State &state,Arena &arena,bool add_fire_heat_flux,bool add_fire_moist_flux,FArray<float> albedo_drag,FArray<float> albedo_drag_p,FArray<float> aod3d_p,FArray<float> area_p,FArray<float> areacell,FArray<float> chem,FArray<float> chem_p,FArray<float> clayfrac_in,FArray<float> clayfrac_in_p,FArray<float> cldfrac,FArray<float> cldfrac_p,FArray<float> coef_bb_dc,FArray<float> coef_bb_dc_p,FStr config_anthro_scheme,FStr config_convection_scheme,FStr config_dust_scheme,FStr config_rwc_scheme,FStr config_smoke_scheme,FArray<float> coszr,FArray<float> coszr_p,FArray<float> ddvel_p,float degrad,FArray<float> dpt2m_p,FArray<float> drydep_flux,FArray<float> drydep_flux_p,int drydep_opt,FArray<float> dx_p,FArray<float> e_ant_in,FArray<float> e_ant_in_p,FArray<float> e_ant_out_p,FArray<float> e_bb_in,FArray<float> e_bb_in_p,FArray<float> e_bb_out,FArray<float> e_bb_out_p,FArray<float> e_dust_out,FArray<float> e_dust_out_p,int ebb_dcycle,FArray<int> eco_id,FArray<int> eco_id_p,FArray<float> efs_flam,FArray<float> efs_flam_p,FArray<float> efs_rsmold,FArray<float> efs_rsmold_p,FArray<float> efs_smold,FArray<float> efs_smold_p,FArray<float> feff,FArray<float> feff_p,FArray<float> fire_ef_out,FArray<float> fire_ef_out_p,FArray<float> fire_end_hr,FArray<float> fire_end_hr_p,FArray<float> fmc_avg_p,FArray<float> fmc_prev24,FArray<float> frac_grid_burned,FArray<float> frac_grid_burned_p,FArray<float> fre_avg_p,FArray<float> fre_in,FArray<float> fre_in_p,FArray<float> fre_out,FArray<float> fre_out_p,FArray<float> fre_prev24,FArray<float> frp_avg_p,FArray<float> frp_in,FArray<float> frp_in_p,FArray<float> frp_out,FArray<float> frp_out_p,FArray<float> frp_prev24,FArray<float> hfx_bb_p,FArray<float> hpbl,FArray<float> hpbl_p,FArray<float> hwp,FArray<float> hwp_avg_p,FArray<float> hwp_p,FArray<float> hwp_prev24,FArray<int> isltyp,FArray<int> isltyp_p,int ite,int its,FArray<int> ivgtyp,FArray<int> ivgtyp_p,int jte,int jts,int kanthro,FArray<int> kpbl,FArray<int> kpbl_p,int kreswoodcomb,int kte,FArray<int> ktop_deep,FArray<int> ktop_deep_p,int kts,int kwildfire,FArray<float> landusef,FArray<float> landusef_p,FArray<float> latcell,float len_disp,FArray<float> loncell,FArray<float> mavail,FArray<float> mavail_p,FArray<int> max_bb_plume_p,FArray<float> meshdensity,FArray<int> min_bb_plume_p,int nblocks,int num_chem,int num_e_ant_in,int num_e_ant_out,int num_e_bb_in,int num_e_bb_out,int num_e_dust_out,int num_landcat,int num_soils,int plumerise_opt,FArray<float> qc_bl,FArray<float> qcbl_p,FArray<float> qfx_bb_p,FArray<float> qi_bl,FArray<float> qibl_p,FArray<float> raincv,FArray<float> raincv_p,FArray<float> rainncv,FArray<float> rainncv_p,FArray<float> rmol,FArray<float> rmol_p,FArray<float> rwc_annual_sum,FArray<float> rwc_annual_sum_p,FArray<float> rwc_annual_sum_smoke_coarse,FArray<float> rwc_annual_sum_smoke_coarse_p,FArray<float> rwc_annual_sum_smoke_fine,FArray<float> rwc_annual_sum_smoke_fine_p,FArray<float> rwc_annual_sum_unspc_coarse,FArray<float> rwc_annual_sum_unspc_coarse_p,FArray<float> rwc_annual_sum_unspc_fine,FArray<float> rwc_annual_sum_unspc_fine_p,FArray<float> rwc_denominator,FArray<float> rwc_denominator_p,FArray<float> sandfrac_in,FArray<float> sandfrac_in_p,FArray<float> sep_in,FArray<float> sep_in_p,FArray<float> skintemp,FArray<float> smois,FArray<float> smois_p,FArray<float> snowh,FArray<float> snowh_p,int t_size,FArray<float> tend_chem_settle_p,FArray<float> totprcp_prev24,FArray<float> totprcp_prev24_p,FArray<float> tsk_p,FArray<float> tslb,FArray<float> tslb_p,FArray<float> u10,FArray<float> u10_p,FArray<float> ust,FArray<float> ust_p,FArray<float> uthres_in,FArray<float> uthres_in_p,FArray<float> uthres_sg_in,FArray<float> uthres_sg_in_p,FArray<float> v10,FArray<float> v10_p,FArray<float> vegfra,FArray<float> vegfra_p,int wetdep_ls_opt,FArray<float> wetdep_resolved,FArray<float> wetdep_resolved_p,FArray<float> xland,FArray<float> xland_p,FArray<float> xlat_p,FArray<float> xlon_p,FArray<float> z0,FArray<float> z0_p,FArray<float> zgrid,FArray<float> zgrid_p,FArray<float> znt,FArray<float> znt_p) {
ArenaScope scope(arena);
int blk;
int i;
int j;
int k;
int last_obs;
int n;
int n_active;
int t;
int t_end;
int t_start;
float sum_fre;
float sum_frp;
float sum_hwp;
float sum_prcp;
for (n=1;n<=num_chem;n+=1) { // L500
for (j=jts;j<=jte;j+=1) { // L501
for (k=kts;k<=kte;k+=1) { // L502
for (i=its;i<=ite;i+=1) { // L503
chem_p(i,k,j,n) = sd_max(1.e-12f,chem(n,k,i)); // L504
if ((drydep_opt != 0)) { // L505
tend_chem_settle_p(i,k,j,n) = 0.f; // L506
} // L507
loop4_continue: ; } loop4_exit: ; // L508
loop3_continue: ; } loop3_exit: ; // L509
loop2_continue: ; } loop2_exit: ; // L510
loop1_continue: ; } loop1_exit: ; // L511
for (j=jts;j<=jte;j+=1) { // L513
for (k=kts;k<=kte;k+=1) { // L514
for (i=its;i<=ite;i+=1) { // L515
zgrid_p(i,k,j) = zgrid(k,i); // L516
cldfrac_p(i,k,j) = cldfrac(k,i); // L517
qcbl_p(i,k,j) = qc_bl(k,i); // L518
qibl_p(i,k,j) = qi_bl(k,i); // L519
aod3d_p(i,k,j) = 0.f; // L520
loop7_continue: ; } loop7_exit: ; // L521
loop6_continue: ; } loop6_exit: ; // L522
loop5_continue: ; } loop5_exit: ; // L523
if ((num_e_bb_in > 0)) { // L526
for (j=jts;j<=jte;j+=1) { // L527
for (k=1;k<=kwildfire;k+=1) { // L528
for (i=its;i<=ite;i+=1) { // L529
for (n=1;n<=num_e_bb_in;n+=1) { // L530
e_bb_in_p(i,k,j,n) = e_bb_in(n,k,i); // L531
loop11_continue: ; } loop11_exit: ; // L532
loop10_continue: ; } loop10_exit: ; // L533
loop9_continue: ; } loop9_exit: ; // L534
loop8_continue: ; } loop8_exit: ; // L535
} // L536
if (((num_e_ant_in > 0) && (config_anthro_scheme != FStr("off")))) { // L538
for (n=1;n<=num_e_ant_in;n+=1) { // L539
for (j=jts;j<=jte;j+=1) { // L540
for (k=1;k<=kanthro;k+=1) { // L541
for (i=its;i<=ite;i+=1) { // L542
e_ant_in_p(i,k,j,n) = e_ant_in(n,k,i); // L543
loop15_continue: ; } loop15_exit: ; // L544
loop14_continue: ; } loop14_exit: ; // L545
loop13_continue: ; } loop13_exit: ; // L546
loop12_continue: ; } loop12_exit: ; // L547
} // L548
for (j=jts;j<=jte;j+=1) { // L550
for (n=1;n<=num_soils;n+=1) { // L551
for (i=its;i<=ite;i+=1) { // L552
smois_p(i,n,j) = smois(n,i); // L553
tslb_p(i,n,j) = tslb(n,i); // L554
loop18_continue: ; } loop18_exit: ; // L555
loop17_continue: ; } loop17_exit: ; // L556
loop16_continue: ; } loop16_exit: ; // L557
if ((config_smoke_scheme != FStr("off"))) { // L562
for (i=its;i<=ite;i+=1) { // L563
for (j=jts;j<=jte;j+=1) { // L564
frp_out_p(i,j) = frp_out(i); // L565
fre_out_p(i,j) = fre_out(i); // L566
hwp_p(i,j) = hwp(i); // L567
coef_bb_dc_p(i,j) = coef_bb_dc(i); // L568
if ((ebb_dcycle == 1)) { // L569
frp_in_p(i,j) = frp_in(i); // L570
fre_in_p(i,j) = fre_in(i); // L571
} else { // L572
eco_id_p(i,j) = eco_id(i); // L573
efs_smold_p(i,j) = efs_smold(i); // L574
efs_flam_p(i,j) = efs_flam(i); // L575
efs_rsmold_p(i,j) = efs_rsmold(i); // L576
fire_ef_out_p(i,j) = fire_ef_out(i); // L577
fire_end_hr_p(i,j) = fire_end_hr(i); // L578
} // L579
if ((plumerise_opt > 0)) { // L580
min_bb_plume_p(i,j) = 0; // L581
max_bb_plume_p(i,j) = 0; // L582
} // L583
if (add_fire_heat_flux) { // L584
hfx_bb_p(i,j) = 0.f; // L585
frac_grid_burned_p(i,j) = frac_grid_burned(i); // L586
} // L587
if (add_fire_moist_flux) { // L588
qfx_bb_p(i,j) = 0.f; // L589
} // L590
loop20_continue: ; } loop20_exit: ; // L591
loop19_continue: ; } loop19_exit: ; // L592
} // L593
if ((config_rwc_scheme != FStr("off"))) { // L595
for (j=jts;j<=jte;j+=1) { // L596
for (k=1;k<=kreswoodcomb;k+=1) { // L597
for (i=its;i<=ite;i+=1) { // L598
rwc_annual_sum_p(i,k,j) = rwc_annual_sum(k,i); // L599
rwc_annual_sum_smoke_fine_p(i,k,j) = rwc_annual_sum_smoke_fine(k,i); // L600
rwc_annual_sum_smoke_coarse_p(i,k,j) = rwc_annual_sum_smoke_coarse(k,i); // L601
rwc_annual_sum_unspc_fine_p(i,k,j) = rwc_annual_sum_unspc_fine(k,i); // L602
rwc_annual_sum_unspc_coarse_p(i,k,j) = rwc_annual_sum_unspc_coarse(k,i); // L603
loop23_continue: ; } loop23_exit: ; // L604
loop22_continue: ; } loop22_exit: ; // L605
loop21_continue: ; } loop21_exit: ; // L606
for (j=jts;j<=jte;j+=1) { // L607
for (i=its;i<=ite;i+=1) { // L608
rwc_denominator_p(i,j) = rwc_denominator(i); // L609
loop25_continue: ; } loop25_exit: ; // L610
loop24_continue: ; } loop24_exit: ; // L611
} // L612
if (((config_smoke_scheme != FStr("off")) && (ebb_dcycle == 2))) { // L614
for (blk=1;blk<=nblocks;blk+=1) { // L615
t_start = (((blk - 1) * t_size) + 1); // L616
t_end = (blk * t_size); // L617
for (i=its;i<=ite;i+=1) { // L618
for (j=jts;j<=jte;j+=1) { // L619
sum_frp = 0.0f; // L620
sum_fre = 0.0f; // L621
sum_hwp = 0.0f; // L622
n_active = 0; // L624
last_obs = (-1); // L625
for (t=t_start;t<=t_end;t+=1) { // L626
if ((frp_prev24(i,t) > 0.0f)) { // L627
sum_frp = sd_add(sum_frp,frp_prev24(i,t)); // L628
sum_fre = sd_add(sum_fre,fre_prev24(i,t)); // L629
sum_hwp = sd_add(sum_hwp,hwp_prev24(i,t)); // L630
n_active = (n_active + 1); // L631
last_obs = t; // L632
} // L633
sum_prcp = sd_add(sum_prcp,totprcp_prev24(i,t)); // L634
loop29_continue: ; } loop29_exit: ; // L635
if ((n_active > 0)) { // L637
frp_avg_p(i,j,blk) = sd_div(sum_frp,n_active); // L638
fre_avg_p(i,j,blk) = sd_div(sum_fre,n_active); // L639
hwp_avg_p(i,j,blk) = sd_div(sum_hwp,n_active); // L640
} else { // L641
frp_avg_p(i,j,blk) = 0.0f; // L642
fre_avg_p(i,j,blk) = 0.0f; // L643
hwp_avg_p(i,j,blk) = 0.0f; // L644
} // L645
if ((blk == nblocks)) { // L647
totprcp_prev24_p(i,j) = sum_prcp; // L648
fire_end_hr_p(i,j) = 24; // L649
if ((last_obs > 0)) { // L650
fire_end_hr_p(i,j) = ((24 - last_obs) + 1); // L651
fmc_avg_p(i,j) = fmc_prev24(i,t_end); // L652
} // L653
} // L654
loop28_continue: ; } loop28_exit: ; // L655
loop27_continue: ; } loop27_exit: ; // L656
loop26_continue: ; } loop26_exit: ; // L657
} // L658
for (j=jts;j<=jte;j+=1) { // L660
for (i=its;i<=ite;i+=1) { // L661
dx_p(i,j) = sd_div(len_disp,sd_pow(meshdensity(i),0.25f)); // L663
area_p(i,j) = areacell(i); // L664
xland_p(i,j) = xland(i); // L665
ust_p(i,j) = ust(i); // L666
xlat_p(i,j) = sd_div(latcell(i),degrad); // L667
xlon_p(i,j) = sd_div(loncell(i),degrad); // L668
isltyp_p(i,j) = isltyp(i); // L669
ivgtyp_p(i,j) = ivgtyp(i); // L670
vegfra_p(i,j) = vegfra(i); // L671
coszr_p(i,j) = coszr(i); // L672
kpbl_p(i,j) = kpbl(i); // L674
if ((config_convection_scheme == FStr("cu_grell_freitas_li"))) { // L675
ktop_deep_p(i,j) = ktop_deep(i); // L676
} else { // L677
ktop_deep_p(i,j) = 1; // L678
} // L679
u10_p(i,j) = u10(i); // L680
v10_p(i,j) = v10(i); // L681
znt_p(i,j) = znt(i); // L682
tsk_p(i,j) = skintemp(i); // L683
hpbl_p(i,j) = hpbl(i); // L684
z0_p(i,j) = z0(i); // L685
snowh_p(i,j) = snowh(i); // L686
rmol_p(i,j) = rmol(i); // L687
if ((config_convection_scheme == FStr("cu_grell_freitas_li"))) { // L688
raincv_p(i,j) = raincv(i); // L689
} else { // L690
raincv_p(i,j) = 0.f; // L691
} // L692
rainncv_p(i,j) = rainncv(i); // L693
dpt2m_p(i,j) = 280.f; // L694
mavail_p(i,j) = mavail(i); // L695
loop31_continue: ; } loop31_exit: ; // L696
loop30_continue: ; } loop30_exit: ; // L697
if ((config_dust_scheme != FStr("off"))) { // L701
for (j=jts;j<=jte;j+=1) { // L702
for (i=its;i<=ite;i+=1) { // L703
sandfrac_in_p(i,j) = sandfrac_in(i); // L704
clayfrac_in_p(i,j) = clayfrac_in(i); // L705
uthres_in_p(i,j) = uthres_in(i); // L706
uthres_sg_in_p(i,j) = uthres_sg_in(i); // L707
albedo_drag_p(i,j) = albedo_drag(i); // L708
feff_p(i,j) = feff(i); // L709
sep_in_p(i,j) = sep_in(i); // L710
loop33_continue: ; } loop33_exit: ; // L711
loop32_continue: ; } loop32_exit: ; // L712
} // L713
for (n=1;n<=num_chem;n+=1) { // L716
for (j=jts;j<=jte;j+=1) { // L717
for (i=its;i<=ite;i+=1) { // L718
if ((drydep_opt != 0)) { // L719
ddvel_p(i,j,n) = 0.f; // L720
drydep_flux_p(i,j,n) = drydep_flux(n,i); // L721
} // L722
if ((wetdep_ls_opt != 0)) { // L723
wetdep_resolved_p(i,j,n) = wetdep_resolved(n,i); // L724
} // L725
loop36_continue: ; } loop36_exit: ; // L726
loop35_continue: ; } loop35_exit: ; // L727
loop34_continue: ; } loop34_exit: ; // L728
for (j=jts;j<=jte;j+=1) { // L731
for (n=1;n<=num_landcat;n+=1) { // L732
for (i=its;i<=ite;i+=1) { // L733
landusef_p(i,n,j) = landusef(n,i); // L734
loop39_continue: ; } loop39_exit: ; // L735
loop38_continue: ; } loop38_exit: ; // L736
loop37_continue: ; } loop37_exit: ; // L737
if (((num_e_ant_out > 0) && (config_anthro_scheme != FStr("off")))) { // L739
for (n=1;n<=num_e_ant_out;n+=1) { // L740
for (j=jts;j<=jte;j+=1) { // L741
for (k=kts;k<=kte;k+=1) { // L742
for (i=its;i<=ite;i+=1) { // L743
e_ant_out_p(i,k,j,n) = 0.f; // L744
loop43_continue: ; } loop43_exit: ; // L745
loop42_continue: ; } loop42_exit: ; // L746
loop41_continue: ; } loop41_exit: ; // L747
loop40_continue: ; } loop40_exit: ; // L748
} // L749
if (((num_e_bb_out > 0) && (config_smoke_scheme != FStr("off")))) { // L751
for (n=1;n<=num_e_bb_out;n+=1) { // L752
for (j=jts;j<=jte;j+=1) { // L753
for (k=kts;k<=kte;k+=1) { // L754
for (i=its;i<=ite;i+=1) { // L755
e_bb_out_p(i,k,j,n) = e_bb_out(n,k,i); // L756
loop47_continue: ; } loop47_exit: ; // L757
loop46_continue: ; } loop46_exit: ; // L758
loop45_continue: ; } loop45_exit: ; // L759
loop44_continue: ; } loop44_exit: ; // L760
} // L761
if (((num_e_dust_out > 0) && (config_dust_scheme != FStr("off")))) { // L763
for (n=1;n<=num_e_dust_out;n+=1) { // L764
for (j=jts;j<=jte;j+=1) { // L765
for (k=kts;k<=kte;k+=1) { // L766
for (i=its;i<=ite;i+=1) { // L767
e_dust_out_p(i,k,j,n) = e_dust_out(n,k,i); // L768
loop51_continue: ; } loop51_exit: ; // L769
loop50_continue: ; } loop50_exit: ; // L770
loop49_continue: ; } loop49_exit: ; // L771
loop48_continue: ; } loop48_exit: ; // L772
} // L773
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:881-994 @1d6e370e; SMOKE @9652bbf
SDDEV void smoke_to_mpas_fields(State &state,Arena &arena,bool add_fire_heat_flux,bool add_fire_moist_flux,FArray<float> aero_emis_for_enhmix,FArray<float> aod3d,FArray<float> aod3d_p,FArray<float> aod3d_smoke,FArray<float> aod3d_smoke_p,FArray<float> aod550,FArray<float> aod550_p,FArray<float> chem,FArray<float> chem_p,FArray<float> coef_bb_dc,FArray<float> coef_bb_dc_p,FStr config_smoke_scheme,FArray<float> ddvel,FArray<float> ddvel_p,FArray<float> drydep_flux,FArray<float> drydep_flux_p,int drydep_opt,FArray<float> e_ant_out,FArray<float> e_ant_out_p,FArray<float> e_bb_out,FArray<float> e_bb_out_p,FArray<float> e_dust_out,FArray<float> e_dust_out_p,int ebb_dcycle,FArray<float> fire_ef_out,FArray<float> fire_ef_out_p,FArray<float> frac_grid_burned,FArray<float> frac_grid_burned_p,FArray<float> fre_out,FArray<float> fre_out_p,FArray<float> frp_out,FArray<float> frp_out_p,FArray<float> hfx_bb,FArray<float> hfx_bb_p,FArray<float> hwp,FArray<float> hwp_p,int ite,int its,int jte,int jts,int kte,int kts,FArray<int> max_bb_plume,FArray<int> max_bb_plume_p,FArray<int> min_bb_plume,FArray<int> min_bb_plume_p,int num_chem,int num_e_ant_out,int num_e_bb_out,int num_e_dust_out,int plumerise_opt,FArray<float> pm10,FArray<float> pm10_p,FArray<float> pm2_5,FArray<float> pm2_5_p,FArray<float> qfx_bb,FArray<float> qfx_bb_p,FArray<float> tend_chem_settle,FArray<float> tend_chem_settle_p,FArray<float> vis,FArray<float> vis_p,int wetdep_ls_opt,FArray<float> wetdep_resolved,FArray<float> wetdep_resolved_p) {
ArenaScope scope(arena);
int blk;
int i;
int j;
int k;
int last_obs;
int n;
int n_active;
int t;
int t_end;
int t_start;
float sum_fre;
float sum_frp;
float sum_hwp;
float sum_prcp;
for (j=jts;j<=jte;j+=1) { // L881
for (k=kts;k<=kte;k+=1) { // L882
for (i=its;i<=ite;i+=1) { // L883
pm2_5(k,i) = pm2_5_p(i,k,j); // L885
pm10(k,i) = pm10_p(i,k,j); // L886
aod3d(k,i) = aod3d_p(i,k,j); // L888
if ((config_smoke_scheme != FStr("off"))) { // L890
aod3d_smoke(k,i) = aod3d_smoke_p(i,k,j); // L891
} // L892
loop3_continue: ; } loop3_exit: ; // L893
loop2_continue: ; } loop2_exit: ; // L894
loop1_continue: ; } loop1_exit: ; // L895
for (n=1;n<=num_chem;n+=1) { // L897
for (j=jts;j<=jte;j+=1) { // L898
for (k=kts;k<=kte;k+=1) { // L899
for (i=its;i<=ite;i+=1) { // L900
chem(n,k,i) = chem_p(i,k,j,n); // L901
if ((drydep_opt != 0)) { // L904
tend_chem_settle(n,k,i) = tend_chem_settle_p(i,k,j,n); // L905
} // L906
loop7_continue: ; } loop7_exit: ; // L907
loop6_continue: ; } loop6_exit: ; // L908
loop5_continue: ; } loop5_exit: ; // L909
loop4_continue: ; } loop4_exit: ; // L910
for (n=1;n<=num_chem;n+=1) { // L912
for (j=jts;j<=jte;j+=1) { // L913
for (i=its;i<=kte;i+=1) { // L914
if ((drydep_opt != 0)) { // L915
ddvel(n,i) = ddvel_p(i,j,n); // L916
drydep_flux(n,i) = drydep_flux_p(i,j,n); // L917
} // L918
if ((wetdep_ls_opt > 0)) { // L919
wetdep_resolved(n,i) = wetdep_resolved_p(i,j,n); // L920
} // L921
loop10_continue: ; } loop10_exit: ; // L922
loop9_continue: ; } loop9_exit: ; // L923
loop8_continue: ; } loop8_exit: ; // L924
for (i=its;i<=ite;i+=1) { // L927
aero_emis_for_enhmix(i) = 0.f; // L928
loop11_continue: ; } loop11_exit: ; // L929
if ((num_e_ant_out > 0)) { // L931
for (n=1;n<=num_e_ant_out;n+=1) { // L932
for (j=jts;j<=jte;j+=1) { // L933
for (k=kts;k<=kte;k+=1) { // L934
for (i=its;i<=ite;i+=1) { // L935
e_ant_out(n,k,i) = e_ant_out_p(i,k,j,n); // L936
aero_emis_for_enhmix(i) = sd_add(aero_emis_for_enhmix(i),e_ant_out_p(i,k,j,n)); // L937
loop15_continue: ; } loop15_exit: ; // L938
loop14_continue: ; } loop14_exit: ; // L939
loop13_continue: ; } loop13_exit: ; // L940
loop12_continue: ; } loop12_exit: ; // L941
} // L942
if ((num_e_bb_out > 0)) { // L944
for (n=1;n<=num_e_bb_out;n+=1) { // L945
for (j=jts;j<=jte;j+=1) { // L946
for (k=kts;k<=kte;k+=1) { // L947
for (i=its;i<=ite;i+=1) { // L948
e_bb_out(n,k,i) = e_bb_out_p(i,k,j,n); // L949
loop19_continue: ; } loop19_exit: ; // L950
loop18_continue: ; } loop18_exit: ; // L951
loop17_continue: ; } loop17_exit: ; // L952
loop16_continue: ; } loop16_exit: ; // L953
} // L954
if ((num_e_dust_out > 0)) { // L956
for (n=1;n<=num_e_dust_out;n+=1) { // L957
for (j=jts;j<=jte;j+=1) { // L958
for (k=kts;k<=kte;k+=1) { // L959
for (i=its;i<=ite;i+=1) { // L960
e_dust_out(n,k,i) = e_dust_out_p(i,k,j,n); // L961
loop23_continue: ; } loop23_exit: ; // L962
loop22_continue: ; } loop22_exit: ; // L963
loop21_continue: ; } loop21_exit: ; // L964
loop20_continue: ; } loop20_exit: ; // L965
} // L966
for (j=jts;j<=jte;j+=1) { // L968
for (i=its;i<=ite;i+=1) { // L969
vis(i) = vis_p(i,j); // L970
aod550(i) = aod550_p(i,j); // L971
if ((config_smoke_scheme != FStr("off"))) { // L972
frp_out(i) = frp_out_p(i,j); // L973
fre_out(i) = fre_out_p(i,j); // L974
hwp(i) = hwp_p(i,j); // L975
coef_bb_dc(i) = coef_bb_dc_p(i,j); // L976
if ((ebb_dcycle == 2)) { // L977
fire_ef_out(i) = fire_ef_out_p(i,j); // L978
} // L979
if ((plumerise_opt > 0)) { // L980
min_bb_plume(i) = min_bb_plume_p(i,j); // L981
max_bb_plume(i) = max_bb_plume_p(i,j); // L982
} // L983
if (add_fire_heat_flux) { // L984
hfx_bb(i) = hfx_bb_p(i,j); // L985
frac_grid_burned(i) = frac_grid_burned_p(i,j); // L986
} // L987
if (add_fire_moist_flux) { // L988
qfx_bb(i) = qfx_bb_p(i,j); // L989
} // L990
} // L991
loop25_continue: ; } loop25_exit: ; // L992
loop24_continue: ; } loop24_exit: ; // L993
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:1259-1291 @1d6e370e; SMOKE @9652bbf
SDDEV void smoke_driver_diagnostics(State &state,Arena &arena,FArray<float> aod3d_p,FArray<float> aod550_p,FArray<float> chem_p,int chemistry_start,FArray<float> dz_p,int index_dust_coarse,int index_dust_fine,int index_smoke_fine,int index_unspc_fine,int ite,int its,int jte,int jts,int kte,int kts,FArray<float> pm10_p,FArray<float> pm2_5_p,FArray<float> rho_p) {
ArenaScope scope(arena);
int blk;
int i;
int j;
int k;
int last_obs;
int n;
int n_active;
int t;
int t_end;
int t_start;
float sum_fre;
float sum_frp;
float sum_hwp;
float sum_prcp;
for (j=jts;j<=jte;j+=1) { // L1260
for (k=kts;k<=kte;k+=1) { // L1261
for (i=its;i<=ite;i+=1) { // L1262
pm2_5_p(i,k,j) = 0.0f; // L1263
if ((index_smoke_fine > 0)) { pm2_5_p(i,k,j) = sd_add(pm2_5_p(i,k,j),chem_p(i,k,j,((index_smoke_fine - chemistry_start) + 1))); } // L1264
if ((index_dust_fine > 0)) { pm2_5_p(i,k,j) = sd_add(pm2_5_p(i,k,j),chem_p(i,k,j,((index_dust_fine - chemistry_start) + 1))); } // L1265
if ((index_unspc_fine > 0)) { pm2_5_p(i,k,j) = sd_add(pm2_5_p(i,k,j),chem_p(i,k,j,((index_unspc_fine - chemistry_start) + 1))); } // L1266
pm2_5_p(i,k,j) = sd_mul(pm2_5_p(i,k,j),rho_p(i,k,j)); // L1267
pm10_p(i,k,j) = 0.0f; // L1269
if ((index_dust_coarse > 0)) { pm10_p(i,k,j) = sd_add(pm10_p(i,k,j),chem_p(i,k,j,((index_dust_coarse - chemistry_start) + 1))); } // L1271
pm10_p(i,k,j) = sd_mul(pm10_p(i,k,j),rho_p(i,k,j)); // L1272
loop3_continue: ; } loop3_exit: ; // L1273
loop2_continue: ; } loop2_exit: ; // L1274
loop1_continue: ; } loop1_exit: ; // L1275
for (j=jts;j<=jte;j+=1) { // L1277
for (i=its;i<=ite;i+=1) { // L1278
aod550_p(i,j) = 0.0f; // L1279
loop5_continue: ; } loop5_exit: ; // L1280
loop4_continue: ; } loop4_exit: ; // L1281
for (j=jts;j<=jte;j+=1) { // L1283
for (k=kts;k<=kte;k+=1) { // L1284
for (i=its;i<=ite;i+=1) { // L1285
aod550_p(i,j) = sd_add(aod550_p(i,j),sd_div(sd_mul(aod3d_p(i,k,j),dz_p(i,k,j)),1000.f)); // L1286
loop8_continue: ; } loop8_exit: ; // L1287
loop7_continue: ; } loop7_exit: ; // L1288
loop6_continue: ; } loop6_exit: ; // L1289
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e; SMOKE @9652bbf
SDDEV void mpas_smoke_driver(State &state,Arena &arena,int num_chem,int chemistry_start,FArray<float> chem,int kanthro,int kbio,int kfire,int kvol,int krwc,int index_smoke_fine,int index_smoke_coarse,int index_dust_fine,int index_dust_coarse,int index_unspc_fine,int index_ch4,int index_e_bb_in_ch4,int index_e_bb_out_ch4,int index_e_bb_in_smoke_fine,int index_e_bb_in_smoke_coarse,int index_e_ant_in_unspc_fine,int index_e_ant_in_unspc_coarse,int index_e_bb_out_smoke_fine,int index_e_bb_out_smoke_coarse,int index_e_dust_out_dust_fine,int index_e_dust_out_dust_coarse,int index_e_ant_out_unspc_fine,int index_e_ant_out_unspc_coarse,FArray<float> frp_in,FArray<float> frp_out,FArray<float> fre_in,FArray<float> fre_out,FArray<float> hwp,FArray<float> coef_bb_dc,FArray<float> totprcp_prev24,FArray<float> hwp_avg,FArray<float> frp_avg,FArray<float> fre_avg,FArray<float> fire_end_hr,int nblocks,FArray<float> efs_map,FArray<int> eco_id,FArray<float> efs_smold,FArray<float> efs_flam,FArray<float> efs_rsmold,FArray<float> fmc_avg,FArray<float> hfx_bb,FArray<float> qfx_bb,FArray<float> frac_grid_burned,FArray<int> min_bb_plume,FArray<int> max_bb_plume,FArray<float> sandfrac_in,FArray<float> clayfrac_in,FArray<float> uthres_in,FArray<float> uthres_sg_in,FArray<float> albedo_drag_in,FArray<float> feff_in,FArray<float> sep_in,FArray<float> e_ant_in,FArray<float> e_ant_out,int num_e_ant_in,int num_e_ant_out,FArray<float> e_bb_in,FArray<float> e_bb_out,FArray<float> e_dust_out,int num_e_bb_in,int num_e_bb_out,int num_e_dust_out,FArray<float> drydep_flux,FArray<float> ddvel,FArray<float> wetdep_resolved,FArray<float> tend_chem_settle,bool do_mpas_smoke,bool do_mpas_dust,bool do_mpas_anthro,bool do_mpas_rwc,bool calc_bb_emis_online,int bb_beta,int hwp_method,float hwp_alpha,int wetdep_ls_opt,float wetdep_ls_alpha,int plumerise_opt,int plume_wind_eff,float plume_alpha,float bb_emis_scale_factor,int ebb_dcycle,int drydep_opt,int pm_settling,bool add_fire_heat_flux,bool add_fire_moist_flux,int plumerisefire_frq,float bb_qv_scale_factor,float dust_alpha,float dust_gamma,float dust_drylimit_factor,float dust_moist_correction,int bb_input_prevh,float rwc_emis_scale_factor,FArray<float> rwc_denominator,FArray<float> rwc_annual_sum,FArray<float> rwc_annual_sum_smoke_fine,FArray<float> rwc_annual_sum_smoke_coarse,FArray<float> rwc_annual_sum_unspc_fine,FArray<float> rwc_annual_sum_unspc_coarse,FArray<float> vis,FArray<float> qc_vis,FArray<float> qr_vis,FArray<float> qi_vis,FArray<float> qs_vis,FArray<float> qg_vis,FArray<float> blcldw_vis,FArray<float> blcldi_vis,FArray<float> coszen,FArray<float> aod3d_smoke,FArray<float> aod3d,int ktau,float dt,FArray<float> dxcell,FArray<float> area,FArray<float> xland,FArray<float> u10,FArray<float> v10,FArray<float> ust,FArray<float> xlat,FArray<float> xlong,FArray<float> tskin,FArray<float> pblh,FArray<float> t2m,FArray<float> p8w,FArray<float> dz8w,FArray<float> z_at_w,FArray<float> p_phy,FArray<float> t_phy,FArray<float> u_phy,FArray<float> v_phy,FArray<float> qv,FArray<float> vvel,FArray<float> pi_phy,FArray<float> rho_phy,FArray<int> kpbl,int nsoil,FArray<float> smois,FArray<float> tslb,FArray<int> ivgtyp,FArray<int> isltyp,int nlcat,FArray<float> swdown,FArray<float> z0,FArray<float> snowh,float julian,FArray<float> rmol,FArray<float> raincv,FArray<float> rainncv,FArray<float> dpt2m,FArray<float> znt,FArray<float> mavail,float g,FArray<float> vegfra,FArray<float> landusef,FArray<float> cldfrac,FArray<int> ktop_deep,float cp,float rd,float gmt,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,bool has_index_smoke_coarse,bool has_index_ch4,bool has_index_e_bb_in_ch4,bool has_index_e_bb_out_ch4,bool has_index_e_bb_in_smoke_coarse,bool has_index_e_bb_out_smoke_coarse,bool has_frp_in,bool has_frp_out,bool has_fre_in,bool has_fre_out,bool has_hwp,bool has_coef_bb_dc,bool has_totprcp_prev24,bool has_hwp_avg,bool has_frp_avg,bool has_fre_avg,bool has_fire_end_hr,bool has_efs_map,bool has_eco_id,bool has_efs_smold,bool has_efs_flam,bool has_efs_rsmold,bool has_fmc_avg,bool has_hfx_bb,bool has_qfx_bb,bool has_frac_grid_burned,bool has_min_bb_plume,bool has_max_bb_plume,bool has_sandfrac_in,bool has_clayfrac_in,bool has_uthres_in,bool has_uthres_sg_in,bool has_albedo_drag_in,bool has_feff_in,bool has_sep_in,bool has_e_ant_in,bool has_e_ant_out,bool has_e_bb_in,bool has_e_bb_out,bool has_e_dust_out,bool has_drydep_flux,bool has_ddvel,bool has_wetdep_resolved,bool has_tend_chem_settle,bool has_rwc_denominator,bool has_rwc_annual_sum,bool has_rwc_annual_sum_smoke_fine,bool has_rwc_annual_sum_smoke_coarse,bool has_rwc_annual_sum_unspc_fine,bool has_rwc_annual_sum_unspc_coarse,bool has_vis,bool has_qc_vis,bool has_qr_vis,bool has_qi_vis,bool has_qs_vis,bool has_qg_vis,bool has_blcldw_vis,bool has_blcldi_vis,bool has_ktop_deep) {
ArenaScope scope(arena);
chem.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
frp_in.bind(ims,ime,jms,jme);
frp_out.bind(ims,ime,jms,jme);
fre_in.bind(ims,ime,jms,jme);
fre_out.bind(ims,ime,jms,jme);
hwp.bind(ims,ime,jms,jme);
coef_bb_dc.bind(ims,ime,jms,jme);
totprcp_prev24.bind(ims,ims,jms,jme);
hwp_avg.bind(ims,ims,jms,jme,1,nblocks);
frp_avg.bind(ims,ims,jms,jme,1,nblocks);
fre_avg.bind(ims,ims,jms,jme,1,nblocks);
fire_end_hr.bind(ims,ims,jms,jme);
efs_map.bind(ims,ime,jms,jme);
eco_id.bind(ims,ime,jms,jme);
efs_smold.bind(ims,ims,jms,jme);
efs_flam.bind(ims,ims,jms,jme);
efs_rsmold.bind(ims,ims,jms,jme);
fmc_avg.bind(ims,ims,jms,jme);
hfx_bb.bind(ims,ime,jms,jme);
qfx_bb.bind(ims,ime,jms,jme);
frac_grid_burned.bind(ims,ime,jms,jme);
min_bb_plume.bind(ims,ime,jms,jme);
max_bb_plume.bind(ims,ime,jms,jme);
sandfrac_in.bind(ims,ime,jms,jme);
clayfrac_in.bind(ims,ime,jms,jme);
uthres_in.bind(ims,ime,jms,jme);
uthres_sg_in.bind(ims,ime,jms,jme);
albedo_drag_in.bind(ims,ime,jms,jme);
feff_in.bind(ims,ime,jms,jme);
sep_in.bind(ims,ime,jms,jme);
e_ant_in.bind(ims,ime,1,kanthro,jms,jme,1,num_e_ant_in);
e_ant_out.bind(ims,ime,kms,kme,jms,jme,1,num_e_ant_out);
e_bb_in.bind(ims,ime,1,kfire,jms,jme,1,num_e_bb_in);
e_bb_out.bind(ims,ime,kms,kme,jms,jme,1,num_e_bb_out);
e_dust_out.bind(ims,ime,kms,kme,jms,jme,1,num_e_dust_out);
drydep_flux.bind(ims,ime,jms,jme,1,num_chem);
ddvel.bind(ims,ime,jms,jme,1,num_chem);
wetdep_resolved.bind(ims,ime,jms,jme,1,num_chem);
tend_chem_settle.bind(ims,ime,kms,kme,jms,jme,1,num_chem);
rwc_denominator.bind(ims,ims,jms,jme);
rwc_annual_sum.bind(ims,ims,1,krwc,jms,jme);
rwc_annual_sum_smoke_fine.bind(ims,ims,1,krwc,jms,jme);
rwc_annual_sum_smoke_coarse.bind(ims,ims,1,krwc,jms,jme);
rwc_annual_sum_unspc_fine.bind(ims,ims,1,krwc,jms,jme);
rwc_annual_sum_unspc_coarse.bind(ims,ims,1,krwc,jms,jme);
vis.bind(ims,ime,jms,jme);
qc_vis.bind(ims,ime,kms,kme,jms,jme);
qr_vis.bind(ims,ime,kms,kme,jms,jme);
qi_vis.bind(ims,ime,kms,kme,jms,jme);
qs_vis.bind(ims,ime,kms,kme,jms,jme);
qg_vis.bind(ims,ime,kms,kme,jms,jme);
blcldw_vis.bind(ims,ime,kms,kme,jms,jme);
blcldi_vis.bind(ims,ime,kms,kme,jms,jme);
coszen.bind(ims,ime,jms,jme);
aod3d_smoke.bind(ims,ime,kms,kme,jms,jme);
aod3d.bind(ims,ime,kms,kme,jms,jme);
dxcell.bind(ims,ime,jms,jme);
area.bind(ims,ime,jms,jme);
xland.bind(ims,ime,jms,jme);
u10.bind(ims,ime,jms,jme);
v10.bind(ims,ime,jms,jme);
ust.bind(ims,ime,jms,jme);
xlat.bind(ims,ime,jms,jme);
xlong.bind(ims,ime,jms,jme);
tskin.bind(ims,ime,jms,jme);
pblh.bind(ims,ime,jms,jme);
t2m.bind(ims,ime,jms,jme);
p8w.bind(ims,ime,kms,kme,jms,jme);
dz8w.bind(ims,ime,kms,kme,jms,jme);
z_at_w.bind(ims,ime,kms,kme,jms,jme);
p_phy.bind(ims,ime,kms,kme,jms,jme);
t_phy.bind(ims,ime,kms,kme,jms,jme);
u_phy.bind(ims,ime,kms,kme,jms,jme);
v_phy.bind(ims,ime,kms,kme,jms,jme);
qv.bind(ims,ime,kms,kme,jms,jme);
vvel.bind(ims,ime,kms,kme,jms,jme);
pi_phy.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
kpbl.bind(ims,ime,jms,jme);
smois.bind(ims,ime,1,nsoil,jms,jme);
tslb.bind(ims,ime,1,nsoil,jms,jme);
ivgtyp.bind(ims,ime,jms,jme);
isltyp.bind(ims,ime,jms,jme);
swdown.bind(ims,ime,jms,jme);
z0.bind(ims,ime,jms,jme);
snowh.bind(ims,ime,jms,jme);
rmol.bind(ims,ime,jms,jme);
raincv.bind(ims,ime,jms,jme);
rainncv.bind(ims,ime,jms,jme);
dpt2m.bind(ims,ime,jms,jme);
znt.bind(ims,ime,jms,jme);
mavail.bind(ims,ime,jms,jme);
vegfra.bind(ims,ime,jms,jme);
landusef.bind(ims,ime,1,nlcat,jms,jme);
cldfrac.bind(ims,ime,kms,kme,jms,jme);
ktop_deep.bind(ims,ime,jms,jme);
auto rri = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto wind_phy = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto theta_phy = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto zmid = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto t8w = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto relhum = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto rh2m = arena.alloc<float>(ims,ime,jms,jme);
int julday;
auto vegfrac = arena.alloc<float>(ims,ime,1,nlcat,jms,jme);
auto total_flashrate = arena.alloc<float>(ims,ime,jms,jme);
auto ebu = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto ebu_coarse = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto ebu_ch4 = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto flam_frac = arena.alloc<float>(ims,ime,jms,jme);
auto fire_hist = arena.alloc<float>(ims,ime,jms,jme);
auto peak_hr = arena.alloc<float>(ims,ime,jms,jme);
auto hwp_day_avg = arena.alloc<float>(ims,ime,jms,jme);
auto uspdavg2d = arena.alloc<float>(ims,ime,jms,jme);
auto hpbl2d = arena.alloc<float>(ims,ime,jms,jme);
auto wind10m = arena.alloc<float>(ims,ime,jms,jme);
auto lu_nofire = arena.alloc<float>(ims,ime,jms,jme);
auto lu_qfire = arena.alloc<float>(ims,ime,jms,jme);
auto lu_sfire = arena.alloc<float>(ims,ime,jms,jme);
auto fire_type = arena.alloc<int>(ims,ime,jms,jme);
auto kpbl_thetav = arena.alloc<int>(ims,ime,jms,jme);
bool call_plume;
float conv_frpi = __uint_as_float(0x358637BDU);
float conv_frei = __uint_as_float(0x358637BDU);
auto vg = arena.alloc<float>(ims,ime,kms,kme,jms,jme,1,num_chem);
auto thetav = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
auto dz8w_flip = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
float theta;
float curr_secs;
int nbegin;
int nv;
int i;
int j;
int k;
int kp;
int n;
FStr errmsg;
int errflg;
bool do_plumerise;
float frp_min = __uint_as_float(0x4B189680U);
float frp_max = __uint_as_float(0x509502F9U);
float fre_min = __uint_as_float(0xC479C000U);
float fre_max = __uint_as_float(0xC479C000U);
float zpbl_threshold = __uint_as_float(0x44FA0000U);
float uspd_threshold = __uint_as_float(0x40A00000U);
float frp_wthreshold = __uint_as_float(0x4E6E6B28U);
float fre_wthreshold = __uint_as_float(0x4E6E6B28U);
float ebb_min = __uint_as_float(0x3A83126FU);
bool do_timing = false;
errmsg = FStr(""); // L259
errflg = 0; // L260
sd_framework_log(state,262); // L262
if ((((!do_mpas_smoke) && (!do_mpas_dust)) && (!do_mpas_anthro))) { return; } // L265
if ((ktau == 1)) { // L270
set_scalar_indices(state,arena,chemistry_start,index_smoke_fine,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,index_unspc_fine,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,state.absent_int,index_dust_fine,index_dust_coarse,true,true,false,false,false,false,false,false,false,false,false,true,false,false,false,false,false,false,false,false,false,true,true); // L271
} // L276
sd_assign(uspdavg2d,0.f,arena); // L281
sd_assign(hpbl2d,0.f,arena); // L282
sd_assign(peak_hr,0.f,arena); // L283
sd_assign(flam_frac,0.f,arena); // L284
sd_assign(fire_type,0,arena); // L285
curr_secs = sd_mul(ktau,dt); // L287
julday = int(julian); // L288
sd_framework_log(state,290); // L290
if (do_timing) { ; /* Framework timing only. */ } // L291
mpas_smoke_prep(state,arena,do_mpas_smoke,ktau,nlcat,cp,ebb_dcycle,ebb_min,xland,xlat,xlong,ivgtyp,isltyp,landusef,snowh,u10,v10,wind10m,t2m,dpt2m,mavail,hwp,hwp_day_avg,index_e_bb_in_smoke_fine,num_e_bb_in,kfire,e_bb_in,t_phy,u_phy,v_phy,p_phy,pi_phy,z_at_w,dz8w,dz8w_flip,rho_phy,qv,relhum,rh2m,rri,total_flashrate,wind_phy,theta_phy,zmid,kpbl_thetav,peak_hr,coef_bb_dc,fire_hist,lu_nofire,lu_qfire,lu_sfire,fire_type,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,has_hwp,has_e_bb_in,has_coef_bb_dc); // L293
if (do_timing) { ; /* Framework timing only. */ } // L309
if ((ktau == 1)) { // L311
sd_framework_log(state,312); // L312
aero_dry_dep_init(state,arena); // L313
sd_framework_log(state,314); // L314
aero_rad_init(state,arena); // L315
} // L316
if (do_mpas_smoke) { // L318
if (calc_bb_emis_online) { // L320
sd_framework_log(state,321); // L321
compute_emission_factors(state,arena,efs_map,vegfrac,bb_beta,eco_id,efs_smold,efs_flam,efs_rsmold,fmc_avg,mavail,nlcat,dt,gmt,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L323
if ((ebb_dcycle == 2)) { // L331
calculate_smoke_emissions(state,arena,dt,julday,nlcat,efs_map,fre_avg,ebb_dcycle,area,nblocks,ktau,bb_input_prevh,ebu,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L332
} else { // L338
calculate_smoke_emissions(state,arena,dt,julday,nlcat,efs_map,fre_in,ebb_dcycle,area,1,ktau,1,ebu,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L339
} // L346
} else { // L347
if ((ktau == 1)) { // L348
for (j=jts;j<=jte;j+=1) { // L349
for (i=its;i<=ite;i+=1) { // L350
ebu(i,kts,j) = e_bb_in(i,kts,j,index_e_bb_in_smoke_fine); // L351
if ((state.mpas_smoke_init.p_smoke_coarse > 0)) { // L352
ebu_coarse(i,kts,j) = e_bb_in(i,kts,j,index_e_bb_in_smoke_coarse); // L353
} // L354
if ((state.mpas_smoke_init.p_ch4 > 0)) { // L355
ebu_ch4(i,kts,j) = e_bb_in(i,kts,j,index_e_bb_in_ch4); // L356
} // L357
for (k=(kts + 1);k<=kte;k+=1) { // L358
ebu(i,k,j) = 0.f; // L359
ebu_coarse(i,k,j) = 0.f; // L360
ebu_ch4(i,k,j) = 0.f; // L361
loop3_continue: ; } loop3_exit: ; // L362
loop2_continue: ; } loop2_exit: ; // L363
loop1_continue: ; } loop1_exit: ; // L364
} else { // L365
for (j=jts;j<=jte;j+=1) { // L366
for (k=kts;k<=kte;k+=1) { // L367
for (i=its;i<=ite;i+=1) { // L368
ebu(i,k,j) = sd_div(e_bb_out(i,k,j,index_e_bb_out_smoke_fine),sd_max(1.e-4f,coef_bb_dc(i,j))); // L370
if ((has_index_e_bb_out_smoke_coarse && (state.mpas_smoke_init.p_smoke_coarse > 0))) { // L372
ebu_coarse(i,k,j) = sd_div(e_bb_out(i,k,j,index_e_bb_out_smoke_coarse),sd_max(1.e-4f,coef_bb_dc(i,j))); // L373
} // L375
if ((has_index_e_bb_out_ch4 && (state.mpas_smoke_init.p_ch4 > 0))) { // L376
ebu_ch4(i,k,j) = sd_div(e_bb_out(i,k,j,index_e_bb_out_ch4),sd_max(1.e-4f,coef_bb_dc(i,j))); // L377
} // L379
loop6_continue: ; } loop6_exit: ; // L380
loop5_continue: ; } loop5_exit: ; // L381
loop4_continue: ; } loop4_exit: ; // L382
} // L383
} // L384
if (add_fire_heat_flux) { // L387
for (j=jts;j<=jte;j+=1) { // L388
for (i=its;i<=ite;i+=1) { // L389
if ((sd_mul(coef_bb_dc(i,j),frp_in(i,j)) >= 1.e7f)) { // L390
hfx_bb(i,j) = sd_min(sd_max(0.f,sd_div(sd_div(sd_mul(sd_mul(0.88f,coef_bb_dc(i,j)),frp_in(i,j)),0.55f),area(i,j))),5000.f); // L391
frac_grid_burned(i,j) = sd_min(sd_max(0.f,sd_div(sd_mul(sd_mul(sd_mul(1.3f,0.0006f),coef_bb_dc(i,j)),frp_in(i,j)),area(i,j))),1.f); // L393
} else { // L396
hfx_bb(i,j) = 0.f; // L397
frac_grid_burned(i,j) = 0.f; // L398
} // L399
loop8_continue: ; } loop8_exit: ; // L400
loop7_continue: ; } loop7_exit: ; // L401
} // L402
if (add_fire_moist_flux) { // L404
for (j=jts;j<=jte;j+=1) { // L405
for (i=its;i<=ite;i+=1) { // L406
if ((sd_mul(coef_bb_dc(i,j),frp_in(i,j)) >= 1.e7f)) { // L407
qfx_bb(i,j) = 0.f; // L408
} else { // L409
qfx_bb(i,j) = 0.f; // L410
} // L411
loop10_continue: ; } loop10_exit: ; // L412
loop9_continue: ; } loop9_exit: ; // L413
} // L414
if ((ebb_dcycle == 2)) { // L416
diurnal_cycle(state,arena,dt,dz8w,rho_phy,state.mpas_smoke_config.pi,ebb_min,chem,num_chem,julday,gmt,xlat,xlong,fire_end_hr,peak_hr,curr_secs,coef_bb_dc,fire_hist,hwp,hwp_avg,hwp_day_avg,vegfrac,eco_id,nblocks,lu_nofire,lu_qfire,lu_sfire,swdown,ebb_dcycle,ebu,fire_type,qv,add_fire_moist_flux,bb_qv_scale_factor,hwp_alpha,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L417
} // L429
for (j=jts;j<=jte;j+=1) { // L432
for (i=its;i<=ite;i+=1) { // L433
if ((fire_type(i,j) == 4)) { // L434
frp_out(i,j) = sd_min(sd_mul(sd_mul(bb_emis_scale_factor,frp_in(i,j)),coef_bb_dc(i,j)),frp_max); // L435
} else { // L436
frp_out(i,j) = sd_min(sd_mul(frp_in(i,j),coef_bb_dc(i,j)),frp_max); // L437
} // L438
loop12_continue: ; } loop12_exit: ; // L439
loop11_continue: ; } loop11_exit: ; // L440
do_plumerise = false; // L443
if ((plumerise_opt > 0)) { do_plumerise = true; } // L444
call_plume = (do_plumerise && (plumerisefire_frq > 0)); // L446
if (call_plume) { call_plume = ((sd_mod(int(curr_secs),sd_max(1,(60 * plumerisefire_frq))) == 0) || (ktau == 2)); } // L447
if (do_timing) { ; /* Framework timing only. */ } // L450
if (call_plume) { // L451
sd_framework_log(state,452); // L452
ebu_driver(state,arena,flam_frac,kfire,theta_phy,qv,rho_phy,vvel,u_phy,v_phy,pi_phy,wind_phy,z_at_w,zmid,g,cp,rd,frp_out,min_bb_plume,max_bb_plume,plume_wind_eff,do_plumerise,kpbl_thetav,kpbl,curr_secs,xlat,xlong,uspdavg2d,hpbl2d,plume_alpha,frp_min,frp_wthreshold,zpbl_threshold,uspd_threshold,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,errmsg,errflg,e_bb_in.section(Slice(),Slice(),Slice(),index_e_bb_in_smoke_fine),ebu,FArray<float>(),FArray<float>(),FArray<float>(),FArray<float>(),true,true,false,false,false,false); // L453
if ((errflg != 0)) { return; } // L470
} // L471
if (do_timing) { ; /* Framework timing only. */ } // L472
if ((state.mpas_smoke_config.addsmoke_flag == 1)) { // L475
if (do_timing) { ; /* Framework timing only. */ } // L476
sd_framework_log(state,477); // L477
add_emis_burn(state,arena,dt,dz8w,rho_phy,state.mpas_smoke_config.pi,ebb_min,chem,num_chem,julday,gmt,xlat,xlong,fire_end_hr,peak_hr,curr_secs,coef_bb_dc,fire_hist,hwp,hwp_day_avg,swdown,ebb_dcycle,ebu,ebu_coarse,ebu_ch4,fire_type,qv,add_fire_moist_flux,bb_emis_scale_factor,aod3d_smoke,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L478
} // L490
if (do_timing) { ; /* Framework timing only. */ } // L491
} // L492
if (do_mpas_dust) { // L511
if (do_timing) { ; /* Framework timing only. */ } // L512
sd_framework_log(state,513); // L513
gocart_dust_fengsha_driver(state,arena,dt,ktau,chem,rho_phy,smois,tslb,p8w,isltyp,snowh,xland,area,g,ust,znt,clayfrac_in,sandfrac_in,uthres_in,uthres_sg_in,albedo_drag_in,feff_in,sep_in,e_dust_out,num_e_dust_out,index_e_dust_out_dust_fine,index_e_dust_out_dust_coarse,state.mpas_smoke_config.num_emis_dust,num_chem,nsoil,dust_alpha,dust_gamma,dust_drylimit_factor,dust_moist_correction,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L516
if (do_timing) { ; /* Framework timing only. */ } // L533
} // L534
if (do_mpas_anthro) { // L536
if (do_timing) { ; /* Framework timing only. */ } // L537
sd_framework_log(state,538); // L538
mpas_smoke_anthro_emis_driver(state,arena,dt,gmt,julday,kanthro,xlat,xlong,chem,num_chem,dz8w,t_phy,rho_phy,e_ant_in,e_ant_out,num_e_ant_in,num_e_ant_out,index_e_ant_in_unspc_fine,index_e_ant_in_unspc_coarse,index_e_ant_out_unspc_fine,index_e_ant_out_unspc_coarse,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L539
if (do_timing) { ; /* Framework timing only. */ } // L547
} // L548
if ((drydep_opt == 1)) { // L552
if (do_timing) { ; /* Framework timing only. */ } // L553
sd_framework_log(state,555); // L555
dry_dep_driver_emerson(state,arena,rmol,ust,znt,num_chem,ddvel,chem,dz8w,snowh,t_phy,p_phy,rho_phy,ivgtyp,g,dt,drydep_flux,tend_chem_settle,state.mpas_smoke_config.dbg_opt,pm_settling,vg,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,curr_secs); // L556
if (do_timing) { ; /* Framework timing only. */ } // L563
if (do_timing) { ; /* Framework timing only. */ } // L564
if ((pm_settling > 0)) { // L565
particle_settling_wrapper(state,arena,tend_chem_settle,chem,rho_phy,dz8w_flip,vg,dt,kts,kte,its,ite,jts,jte,num_chem,ims,ime,jms,jme,kms,kme); // L566
} // L568
if (do_timing) { ; /* Framework timing only. */ } // L569
} else if ((drydep_opt == 2)) { // L571
dry_dep_driver_simple(state,arena,rmol,ust,state.mpas_smoke_config.ndvel,ddvel,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L572
} else { // L576
sd_framework_log(state,577); // L577
} // L578
if ((wetdep_ls_opt == 1)) { // L581
if (do_timing) { ; /* Framework timing only. */ } // L582
sd_framework_log(state,583); // L583
wetdep_ls(state,arena,dt,g,chem,rainncv,qv,rho_phy,num_chem,dz8w,vvel,p_phy,wetdep_ls_alpha,wetdep_resolved,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L584
if (do_timing) { ; /* Framework timing only. */ } // L591
} // L592
sd_framework_log(state,595); // L595
mpas_aod_diag(state,arena,chem,aod3d,rho_phy,dz8w,num_chem,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L596
sd_framework_log(state,600); // L600
mpas_visibility_diag(state,arena,qc_vis,qr_vis,qi_vis,qs_vis,qg_vis,blcldw_vis,blcldi_vis,rho_phy,wind10m,wind_phy,rh2m,relhum,qv,t2m,t_phy,coszen,aod3d,vis,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // L601
for (j=jts;j<=jte;j+=1) { // L613
for (i=its;i<=ite;i+=1) { // L614
if ((fire_type(i,j) == 4)) { // L615
frp_out(i,j) = sd_mul(1.e-6f,sd_min(sd_mul(sd_mul(bb_emis_scale_factor,frp_in(i,j)),coef_bb_dc(i,j)),frp_max)); // L616
} else { // L617
frp_out(i,j) = sd_mul(1.e-6f,sd_min(sd_mul(frp_in(i,j),coef_bb_dc(i,j)),frp_max)); // L618
} // L619
loop14_continue: ; } loop14_exit: ; // L620
loop13_continue: ; } loop13_exit: ; // L621
for (j=jts;j<=jte;j+=1) { // L623
for (k=kts;k<=kte;k+=1) { // L624
for (i=its;i<=ite;i+=1) { // L625
if (do_mpas_smoke) { // L626
e_bb_out(i,k,j,index_e_bb_out_smoke_fine) = sd_mul(ebu(i,k,j),coef_bb_dc(i,j)); // L627
if (has_index_e_bb_out_smoke_coarse) { e_bb_out(i,k,j,index_e_bb_out_smoke_coarse) = sd_mul(ebu_coarse(i,k,j),coef_bb_dc(i,j)); } // L628
} // L629
loop17_continue: ; } loop17_exit: ; // L634
loop16_continue: ; } loop16_exit: ; // L635
loop15_continue: ; } loop15_exit: ; // L636
}

// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:640-835 @1d6e370e; SMOKE @9652bbf
SDDEV void mpas_smoke_prep(State &state,Arena &arena,bool do_mpas_smoke,int ktau,int nlcat,float cp,int ebb_dcycle,float ebb_min,FArray<float> xland,FArray<float> xlat,FArray<float> xlong,FArray<int> ivgtyp,FArray<int> isltyp,FArray<float> vegfrac,FArray<float> snowh,FArray<float> u10,FArray<float> v10,FArray<float> wind10m,FArray<float> t2m,FArray<float> dpt2m,FArray<float> wetness,FArray<float> hwp,FArray<float> hwp_day_avg,int index_e_bb_in_smoke_fine,int num_e_bb_in,int kfire,FArray<float> e_bb_in,FArray<float> t_phy,FArray<float> u_phy,FArray<float> v_phy,FArray<float> p_phy,FArray<float> pi_phy,FArray<float> z_at_w,FArray<float> dz8w,FArray<float> dz8w_flip,FArray<float> rho_phy,FArray<float> qv,FArray<float> relhum,FArray<float> rh2m,FArray<float> rri,FArray<float> total_flashrate,FArray<float> wind_phy,FArray<float> theta_phy,FArray<float> zmid,FArray<int> kpbl_thetav,FArray<float> peak_hr,FArray<float> coef_bb_dc,FArray<float> fire_hist,FArray<float> lu_nofire,FArray<float> lu_qfire,FArray<float> lu_sfire,FArray<int> fire_type,int ids,int ide,int jds,int jde,int kds,int kde,int ims,int ime,int jms,int jme,int kms,int kme,int its,int ite,int jts,int jte,int kts,int kte,bool has_hwp,bool has_e_bb_in,bool has_coef_bb_dc) {
ArenaScope scope(arena);
xland.bind(ims,ime,jms,jme);
xlat.bind(ims,ime,jms,jme);
xlong.bind(ims,ime,jms,jme);
ivgtyp.bind(ims,ime,jms,jme);
isltyp.bind(ims,ime,jms,jme);
vegfrac.bind(ims,ime,1,nlcat,jms,jme);
snowh.bind(ims,ime,jms,jme);
u10.bind(ims,ime,jms,jme);
v10.bind(ims,ime,jms,jme);
wind10m.bind(ims,ime,jms,jme);
t2m.bind(ims,ime,jms,jme);
dpt2m.bind(ims,ime,jms,jme);
wetness.bind(ims,ime,jms,jme);
hwp.bind(ims,ime,jms,jme);
hwp_day_avg.bind(ims,ime,jms,jme);
e_bb_in.bind(ims,ime,1,kfire,jms,jme,1,num_e_bb_in);
t_phy.bind(ims,ime,kms,kme,jms,jme);
u_phy.bind(ims,ime,kms,kme,jms,jme);
v_phy.bind(ims,ime,kms,kme,jms,jme);
p_phy.bind(ims,ime,kms,kme,jms,jme);
pi_phy.bind(ims,ime,kms,kme,jms,jme);
z_at_w.bind(ims,ime,kms,kme,jms,jme);
dz8w.bind(ims,ime,kms,kme,jms,jme);
dz8w_flip.bind(ims,ime,kms,kme,jms,jme);
rho_phy.bind(ims,ime,kms,kme,jms,jme);
qv.bind(ims,ime,kms,kme,jms,jme);
relhum.bind(ims,ime,kms,kme,jms,jme);
rh2m.bind(ims,ime,jms,jme);
rri.bind(ims,ime,kms,kme,jms,jme);
total_flashrate.bind(ims,ime,jms,jme);
wind_phy.bind(ims,ime,kms,kme,jms,jme);
theta_phy.bind(ims,ime,kms,kme,jms,jme);
zmid.bind(ims,ime,kms,kme,jms,jme);
kpbl_thetav.bind(ims,ime,jms,jme);
peak_hr.bind(ims,ime,jms,jme);
coef_bb_dc.bind(ims,ime,jms,jme);
fire_hist.bind(ims,ime,jms,jme);
lu_nofire.bind(ims,ime,jms,jme);
lu_qfire.bind(ims,ime,jms,jme);
lu_sfire.bind(ims,ime,jms,jme);
fire_type.bind(ims,ime,jms,jme);
float delta_theta4gust = __uint_as_float(0x3F000000U);
float theta;
float wdgust;
float snoweq;
float term2;
float term3;
auto thetav = arena.alloc<float>(ims,ime,kms,kme,jms,jme);
int i;
int j;
int k;
int k1;
int nv;
if (do_mpas_smoke) { // L690
if ((ktau == 1)) { // L691
for (j=jts;j<=jte;j+=1) { // L692
for (i=its;i<=ite;i+=1) { // L693
fire_hist(i,j) = 1.f; // L694
coef_bb_dc(i,j) = 1.f; // L695
if ((xlong(i,j) < 230.f)) { // L696
peak_hr(i,j) = sd_mul(0.0f,3600.f); // L697
} else if ((xlong(i,j) < 245.f)) { // L698
peak_hr(i,j) = sd_mul(23.0f,3600.f); // L699
} else if ((xlong(i,j) < 260.f)) { // L700
peak_hr(i,j) = sd_mul(22.0f,3600.f); // L701
} else if ((xlong(i,j) < 275.f)) { // L702
peak_hr(i,j) = sd_mul(21.0f,3600.f); // L703
} else if ((xlong(i,j) < 290.f)) { // L704
peak_hr(i,j) = sd_mul(20.0f,3600.f); // L705
} else { // L706
peak_hr(i,j) = sd_mul(19.0f,3600.f); // L707
} // L708
loop2_continue: ; } loop2_exit: ; // L709
loop1_continue: ; } loop1_exit: ; // L710
} // L711
} // L712
for (j=jts;j<=jte;j+=1) { // L714
for (k=kts;k<=kte;k+=1) { // L715
for (i=its;i<=ite;i+=1) { // L716
zmid(i,k,j) = z_at_w(i,k,j); // L717
loop5_continue: ; } loop5_exit: ; // L718
loop4_continue: ; } loop4_exit: ; // L719
loop3_continue: ; } loop3_exit: ; // L720
for (j=jts;j<=jte;j+=1) { // L722
for (k=kts;k<=kte;k+=1) { // L723
for (i=its;i<=ite;i+=1) { // L724
theta_phy(i,k,j) = sd_mul(sd_div(t_phy(i,k,j),pi_phy(i,k,j)),cp); // L725
wind_phy(i,k,j) = sd_sqrt(sd_add(sd_ipow(u_phy(i,k,j),2),sd_ipow(v_phy(i,k,j),2))); // L726
rri(i,k,j) = sd_div(1.f,rho_phy(i,k,j)); // L727
loop8_continue: ; } loop8_exit: ; // L728
loop7_continue: ; } loop7_exit: ; // L729
loop6_continue: ; } loop6_exit: ; // L730
for (j=jts;j<=jte;j+=1) { // L733
for (k=kts;k<=kte;k+=1) { // L734
for (i=its;i<=ite;i+=1) { // L735
relhum(i,k,j) = sd_max(.1f,sd_min(.95f,sd_div(qv(i,k,j),sd_div(sd_mul(3.80f,sd_exp(sd_div(sd_mul(17.27f,sd_sub(t_phy(i,k,j),273.f)),sd_sub(t_phy(i,k,j),36.f)))),sd_mul(.01f,p_phy(i,k,j)))))); // L736
loop11_continue: ; } loop11_exit: ; // L739
loop10_continue: ; } loop10_exit: ; // L740
loop9_continue: ; } loop9_exit: ; // L741
for (j=jts;j<=jte;j+=1) { // L744
for (i=its;i<=ite;i+=1) { // L745
total_flashrate(i,j) = 0.f; // L746
rh2m(i,j) = relhum(i,1,j); // L747
loop13_continue: ; } loop13_exit: ; // L751
loop12_continue: ; } loop12_exit: ; // L752
for (j=jts;j<=jte;j+=1) { // L757
for (k=kts;k<=kte;k+=1) { // L758
for (i=its;i<=ite;i+=1) { // L759
theta = sd_mul(t_phy(i,k,j),sd_pow(sd_div(1.e5f,p_phy(i,k,j)),0.286f)); // L760
thetav(i,k,j) = sd_mul(theta,sd_add(1.f,sd_mul(0.61f,qv(i,k,j)))); // L761
dz8w_flip(i,k,j) = dz8w(i,((kte - k) + kts),j); // L762
loop16_continue: ; } loop16_exit: ; // L763
loop15_continue: ; } loop15_exit: ; // L764
loop14_continue: ; } loop14_exit: ; // L765
for (i=its;i<=ite;i+=1) { // L767
for (j=jts;j<=jte;j+=1) { // L768
if ((thetav(i,(kts + 1),j) < sd_add(thetav(i,kts,j),delta_theta4gust))) { // L769
for (k=(kts + 1);k<=kte;k+=1) { // L770
k1 = k; // L771
if ((thetav(i,((kts + k) - 1),j) > sd_add(thetav(i,kts,j),delta_theta4gust))) { // L773
goto loop19_exit; // L774
} // L775
loop19_continue: ; } loop19_exit: ; // L776
kpbl_thetav(i,j) = k1; // L777
} else { // L778
kpbl_thetav(i,j) = (kts + 1); // L779
} // L780
loop18_continue: ; } loop18_exit: ; // L781
loop17_continue: ; } loop17_exit: ; // L782
if (do_mpas_smoke) { // L784
if ((ebb_dcycle == 2)) { // L787
for (j=jts;j<=jte;j+=1) { // L788
for (i=its;i<=ite;i+=1) { // L789
if ((e_bb_in(i,1,j,index_e_bb_in_smoke_fine) < ebb_min)) { // L790
fire_type(i,j) = 0; // L791
lu_nofire(i,j) = 1.0f; // L792
} else { // L793
lu_nofire(i,j) = sd_add(sd_add(sd_add(vegfrac(i,11,j),vegfrac(i,15,j)),vegfrac(i,17,j)),vegfrac(i,20,j)); // L795
lu_qfire(i,j) = sd_add(sd_add(sd_add(sd_add(lu_nofire(i,j),vegfrac(i,12,j)),vegfrac(i,13,j)),vegfrac(i,14,j)),vegfrac(i,16,j)); // L798
lu_sfire(i,j) = sd_add(sd_add(sd_add(lu_qfire(i,j),vegfrac(i,8,j)),vegfrac(i,9,j)),vegfrac(i,10,j)); // L800
if ((lu_nofire(i,j) > 0.95f)) { // L801
fire_type(i,j) = 0; // L802
} else if ((lu_qfire(i,j) > 0.9f)) { // L803
fire_type(i,j) = 1; // L804
} else if ((((xlong(i,j) > 260.f) && (xlat(i,j) > 25.f)) && (xlat(i,j) < 41.f))) { // L805
fire_type(i,j) = 2; // L806
} else if ((lu_sfire(i,j) > 0.8f)) { // L807
fire_type(i,j) = 3; // L808
} else { // L809
fire_type(i,j) = 4; // L810
} // L811
} // L812
loop21_continue: ; } loop21_exit: ; // L813
loop20_continue: ; } loop20_exit: ; // L814
} // L815
for (i=its;i<=ite;i+=1) { // L820
for (j=jts;j<=jte;j+=1) { // L821
wind10m(i,j) = sd_sqrt(sd_add(sd_ipow(u10(i,j),2),sd_ipow(v10(i,j),2))); // L822
wdgust = sd_max(wind10m(i,j),3.f); // L823
snoweq = sd_max(sd_div(sd_sub(25.f,snowh(i,j)),25.f),0.f); // L824
term2 = 15.f; // L826
term3 = sd_pow(sd_sub(1.f,wetness(i,j)),0.4f); // L827
hwp(i,j) = sd_mul(sd_mul(sd_mul(sd_mul(0.177f,sd_pow(wdgust,0.97f)),term2),sd_pow(sd_sub(1.f,wetness(i,j)),0.4f)),snoweq); // L828
hwp_day_avg(i,j) = hwp(i,j); // L829
loop23_continue: ; } loop23_exit: ; // L830
loop22_continue: ; } loop22_exit: ; // L831
} // L832
}
SDDEV void initialize(State &state, Arena &storage,Plume &plume) {
state.plume_store = &plume;
state.framework_log_active = false; state.framework_log_output_count = 0;
state.wrapper_log_count = 0;
state.seas_data_mod.number_ss_bins = 5;
state.seas_data_mod.ra = storage.alloc<float>(1,state.seas_data_mod.number_ss_bins);
state.seas_data_mod.ra(1) = __uint_as_float(0x3CF5C28FU);
state.seas_data_mod.ra(2) = __uint_as_float(0x3DCCCCCDU);
state.seas_data_mod.ra(3) = __uint_as_float(0x3F000000U);
state.seas_data_mod.ra(4) = __uint_as_float(0x3FC00000U);
state.seas_data_mod.ra(5) = __uint_as_float(0x40A00000U);
state.seas_data_mod.rb = storage.alloc<float>(1,state.seas_data_mod.number_ss_bins);
state.seas_data_mod.rb(1) = __uint_as_float(0x3DCCCCCDU);
state.seas_data_mod.rb(2) = __uint_as_float(0x3F000000U);
state.seas_data_mod.rb(3) = __uint_as_float(0x3FC00000U);
state.seas_data_mod.rb(4) = __uint_as_float(0x40A00000U);
state.seas_data_mod.rb(5) = __uint_as_float(0x41200000U);
state.seas_data_mod.emission_scale = storage.alloc<float>(1,state.seas_data_mod.number_ss_bins);
state.seas_data_mod.emission_scale(1) = __uint_as_float(0x3F800000U);
state.seas_data_mod.emission_scale(2) = __uint_as_float(0x3F800000U);
state.seas_data_mod.emission_scale(3) = __uint_as_float(0x3F800000U);
state.seas_data_mod.emission_scale(4) = __uint_as_float(0x3F800000U);
state.seas_data_mod.emission_scale(5) = __uint_as_float(0x3F800000U);
state.seas_data_mod.den_seas = storage.alloc<float>(1,state.seas_data_mod.number_ss_bins);
state.seas_data_mod.den_seas(1) = __uint_as_float(0x45098000U);
state.seas_data_mod.den_seas(2) = __uint_as_float(0x45098000U);
state.seas_data_mod.den_seas(3) = __uint_as_float(0x45098000U);
state.seas_data_mod.den_seas(4) = __uint_as_float(0x45098000U);
state.seas_data_mod.den_seas(5) = __uint_as_float(0x45098000U);
state.seas_data_mod.reff_seas = storage.alloc<float>(1,state.seas_data_mod.number_ss_bins);
state.seas_data_mod.reff_seas(1) = __uint_as_float(0x33A9A6B6U);
state.seas_data_mod.reff_seas(2) = __uint_as_float(0x34A9A6B6U);
state.seas_data_mod.reff_seas(3) = __uint_as_float(0x3596308CU);
state.seas_data_mod.reff_seas(4) = __uint_as_float(0x363D1CDFU);
state.seas_data_mod.reff_seas(5) = __uint_as_float(0x3702647CU);
state.seas_ngac_mod.r80fac = __uint_as_float(0x3FD33333U);
state.seas_ngac_mod.rhop = __uint_as_float(0x45098000U);
state.mpas_smoke_init.p_smoke_fine = -1;
state.mpas_smoke_init.p_smoke_coarse = -1;
state.mpas_smoke_init.p_dust_fine = -1;
state.mpas_smoke_init.p_dust_coarse = -1;
state.mpas_smoke_init.p_polp_all = -1;
state.mpas_smoke_init.p_polp_tree = -1;
state.mpas_smoke_init.p_polp_grass = -1;
state.mpas_smoke_init.p_polp_weed = -1;
state.mpas_smoke_init.p_pols_tree = -1;
state.mpas_smoke_init.p_pols_grass = -1;
state.mpas_smoke_init.p_pols_weed = -1;
state.mpas_smoke_init.p_pols_all = -1;
state.mpas_smoke_init.p_unspc_fine = -1;
state.mpas_smoke_init.p_unspc_coarse = -1;
state.mpas_smoke_init.p_ssalt_fine = -1;
state.mpas_smoke_init.p_ssalt_coarse = -1;
state.mpas_smoke_init.p_no3_a_fine = -1;
state.mpas_smoke_init.p_so4_a_fine = -1;
state.mpas_smoke_init.p_nh4_a_fine = -1;
state.mpas_smoke_init.p_so2 = -1;
state.mpas_smoke_init.p_nh3 = -1;
state.mpas_smoke_init.p_ch4 = -1;
state.mpas_smoke_init.p_mold_fine = -1;
state.mpas_smoke_init.p_bact_fine = -1;
state.mpas_smoke_init.p_plastic_fine = -1;
state.mpas_smoke_init.p_dms = -1;
state.ssalt_mod.chem_opt_gocart = 300;
state.ssalt_mod.chem_opt = 300;
state.ssalt_mod.emission_scheme = 3;
state.rad_data_mod.sc_eff = storage.alloc<float>(1,1000);
sd_assign(state.rad_data_mod.sc_eff,0,storage);
state.rad_data_mod.ab_eff = storage.alloc<float>(1,1000);
sd_assign(state.rad_data_mod.ab_eff,0,storage);
state.mpas_smoke_config.epsilc = __uint_as_float(0x2B8CBCCCU);
state.mpas_smoke_config.pi = __uint_as_float(0x40490FDAU);
state.mpas_smoke_config.seas_opt = 1;
state.mpas_smoke_config.addsmoke_flag = 1;
state.mpas_smoke_config.n_dbg_lines = 3;
state.mpas_smoke_config.nfire_types = 5;
state.mpas_smoke_config.dbg_opt = false;
state.mpas_smoke_config.aero_ind_fdb = false;
state.mpas_smoke_config.chem_opt_gocart = 1;
state.mpas_smoke_config.num_moist = 2;
state.mpas_smoke_config.num_emis_seas = 5;
state.mpas_smoke_config.num_emis_dust = 5;
state.mpas_smoke_config.p_dust_1 = 10;
state.mpas_smoke_config.p_dust_2 = 11;
state.mpas_smoke_config.p_dust_3 = 12;
state.mpas_smoke_config.p_dust_4 = 13;
state.mpas_smoke_config.p_dust_5 = 14;
state.mpas_smoke_config.p_edust1 = 1;
state.mpas_smoke_config.p_edust2 = 2;
state.mpas_smoke_config.p_edust3 = 3;
state.mpas_smoke_config.p_edust4 = 4;
state.mpas_smoke_config.p_edust5 = 5;
state.mpas_smoke_config.ndvel = 20;
state.mpas_smoke_config.numgas = 0;
state.module_zero_plumegen_coms.nkp = 200;
state.module_zero_plumegen_coms.ntime = 200;
state.module_zero_plumegen_coms.private_thread_coms = nullptr;
state.dust_data_mod.ndust = 5;
state.dust_data_mod.ndcls = 3;
state.dust_data_mod.ndsrc = 1;
state.dust_data_mod.maxstypes = 100;
state.dust_data_mod.nsalt = 9;
state.dust_data_mod.dyn_visc = __uint_as_float(0x377BA882U);
state.dust_data_mod.den_dust = storage.alloc<float>(1,state.dust_data_mod.ndust);
state.dust_data_mod.den_dust(1) = __uint_as_float(0x451C4000U);
state.dust_data_mod.den_dust(2) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_dust(3) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_dust(4) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_dust(5) = __uint_as_float(0x4525A000U);
state.dust_data_mod.reff_dust = storage.alloc<float>(1,state.dust_data_mod.ndust);
state.dust_data_mod.reff_dust(1) = __uint_as_float(0x3543F538U);
state.dust_data_mod.reff_dust(2) = __uint_as_float(0x35BBE7A2U);
state.dust_data_mod.reff_dust(3) = __uint_as_float(0x36210FB0U);
state.dust_data_mod.reff_dust(4) = __uint_as_float(0x3696FEB5U);
state.dust_data_mod.reff_dust(5) = __uint_as_float(0x370637BDU);
state.dust_data_mod.frac_s = storage.alloc<float>(1,state.dust_data_mod.ndust);
state.dust_data_mod.frac_s(1) = __uint_as_float(0x3DCCCCCDU);
state.dust_data_mod.frac_s(2) = __uint_as_float(0x3E800000U);
state.dust_data_mod.frac_s(3) = __uint_as_float(0x3E800000U);
state.dust_data_mod.frac_s(4) = __uint_as_float(0x3E800000U);
state.dust_data_mod.frac_s(5) = __uint_as_float(0x3E800000U);
state.dust_data_mod.lo_dust = storage.alloc<float>(1,state.dust_data_mod.ndust);
state.dust_data_mod.lo_dust(1) = __uint_as_float(0x33D6BF95U);
state.dust_data_mod.lo_dust(2) = __uint_as_float(0x358637BDU);
state.dust_data_mod.lo_dust(3) = __uint_as_float(0x35F19787U);
state.dust_data_mod.lo_dust(4) = __uint_as_float(0x3649539CU);
state.dust_data_mod.lo_dust(5) = __uint_as_float(0x36C9539CU);
state.dust_data_mod.up_dust = storage.alloc<float>(1,state.dust_data_mod.ndust);
state.dust_data_mod.up_dust(1) = __uint_as_float(0x358637BDU);
state.dust_data_mod.up_dust(2) = __uint_as_float(0x35F19787U);
state.dust_data_mod.up_dust(3) = __uint_as_float(0x3649539CU);
state.dust_data_mod.up_dust(4) = __uint_as_float(0x36C9539CU);
state.dust_data_mod.up_dust(5) = __uint_as_float(0x3727C5ACU);
state.dust_data_mod.fengsha_maxstypes = 13;
state.dust_data_mod.dust_uthres = storage.alloc<float>(1,state.dust_data_mod.fengsha_maxstypes);
state.dust_data_mod.dust_uthres(1) = __uint_as_float(0x3D851EB8U);
state.dust_data_mod.dust_uthres(2) = __uint_as_float(0x3E3851ECU);
state.dust_data_mod.dust_uthres(3) = __uint_as_float(0x3E8A3D71U);
state.dust_data_mod.dust_uthres(4) = __uint_as_float(0x3E99999AU);
state.dust_data_mod.dust_uthres(5) = __uint_as_float(0x3EB33333U);
state.dust_data_mod.dust_uthres(6) = __uint_as_float(0x3EC28F5CU);
state.dust_data_mod.dust_uthres(7) = __uint_as_float(0x3EB33333U);
state.dust_data_mod.dust_uthres(8) = __uint_as_float(0x3ED1EB85U);
state.dust_data_mod.dust_uthres(9) = __uint_as_float(0x3ED1EB85U);
state.dust_data_mod.dust_uthres(10) = __uint_as_float(0x3EE66666U);
state.dust_data_mod.dust_uthres(11) = __uint_as_float(0x3F000000U);
state.dust_data_mod.dust_uthres(12) = __uint_as_float(0x3EE66666U);
state.dust_data_mod.dust_uthres(13) = __uint_as_float(0x461C3C00U);
state.dust_data_mod.dust_calcdrag = 1;
state.dust_data_mod.dust_moist_opt = 1;
state.dust_data_mod.spoint = storage.alloc<int>(1,state.dust_data_mod.nsalt);
state.dust_data_mod.spoint(1) = 1;
state.dust_data_mod.spoint(2) = 2;
state.dust_data_mod.spoint(3) = 2;
state.dust_data_mod.spoint(4) = 2;
state.dust_data_mod.spoint(5) = 2;
state.dust_data_mod.spoint(6) = 2;
state.dust_data_mod.spoint(7) = 3;
state.dust_data_mod.spoint(8) = 3;
state.dust_data_mod.spoint(9) = 3;
state.dust_data_mod.reff_salt = storage.alloc<float>(1,state.dust_data_mod.nsalt);
state.dust_data_mod.reff_salt(1) = __uint_as_float(0x353E96D4U);
state.dust_data_mod.reff_salt(2) = __uint_as_float(0x35B7E0D7U);
state.dust_data_mod.reff_salt(3) = __uint_as_float(0x36307F0EU);
state.dust_data_mod.reff_salt(4) = __uint_as_float(0x36A7C5ACU);
state.dust_data_mod.reff_salt(5) = __uint_as_float(0x371F6230U);
state.dust_data_mod.reff_salt(6) = __uint_as_float(0x3797D574U);
state.dust_data_mod.reff_salt(7) = __uint_as_float(0x3810B418U);
state.dust_data_mod.reff_salt(8) = __uint_as_float(0x38895D0BU);
state.dust_data_mod.reff_salt(9) = __uint_as_float(0x3903126FU);
state.dust_data_mod.den_salt = storage.alloc<float>(1,state.dust_data_mod.nsalt);
state.dust_data_mod.den_salt(1) = __uint_as_float(0x451C4000U);
state.dust_data_mod.den_salt(2) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_salt(3) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_salt(4) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_salt(5) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_salt(6) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_salt(7) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_salt(8) = __uint_as_float(0x4525A000U);
state.dust_data_mod.den_salt(9) = __uint_as_float(0x4525A000U);
state.dust_data_mod.frac_salt = storage.alloc<float>(1,state.dust_data_mod.nsalt);
state.dust_data_mod.frac_salt(1) = __uint_as_float(0x3F800000U);
state.dust_data_mod.frac_salt(2) = __uint_as_float(0x3E4CCCCDU);
state.dust_data_mod.frac_salt(3) = __uint_as_float(0x3E4CCCCDU);
state.dust_data_mod.frac_salt(4) = __uint_as_float(0x3E4CCCCDU);
state.dust_data_mod.frac_salt(5) = __uint_as_float(0x3E4CCCCDU);
state.dust_data_mod.frac_salt(6) = __uint_as_float(0x3E4CCCCDU);
state.dust_data_mod.frac_salt(7) = __uint_as_float(0x3EAA7EFAU);
state.dust_data_mod.frac_salt(8) = __uint_as_float(0x3EAA7EFAU);
state.dust_data_mod.frac_salt(9) = __uint_as_float(0x3EAA7EFAU);
state.dust_data_mod.max_soiltyp = 30;
state.dust_data_mod.maxsmc = storage.alloc<float>(1,state.dust_data_mod.max_soiltyp);
state.dust_data_mod.maxsmc(1) = __uint_as_float(0x3ED78D50U);
state.dust_data_mod.maxsmc(2) = __uint_as_float(0x3EED9168U);
state.dust_data_mod.maxsmc(3) = __uint_as_float(0x3EEF9DB2U);
state.dust_data_mod.maxsmc(4) = __uint_as_float(0x3EDE353FU);
state.dust_data_mod.maxsmc(5) = __uint_as_float(0x3ECFDF3BU);
state.dust_data_mod.maxsmc(6) = __uint_as_float(0x3EEE147BU);
state.dust_data_mod.maxsmc(7) = __uint_as_float(0x3ECED917U);
state.dust_data_mod.maxsmc(8) = __uint_as_float(0x3EE0C49CU);
state.dust_data_mod.maxsmc(9) = __uint_as_float(0x3ED78D50U);
state.dust_data_mod.maxsmc(10) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(11) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(12) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(13) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(14) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(15) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(16) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(17) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(18) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(19) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(20) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(21) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(22) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(23) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(24) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(25) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(26) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(27) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(28) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(29) = __uint_as_float(0x00000000U);
state.dust_data_mod.maxsmc(30) = __uint_as_float(0x00000000U);
state.dep_data_mod.max_dep_vel = __uint_as_float(0x3BA3D70AU);
state.dep_data_mod.dep_ref_hgt = __uint_as_float(0x40000000U);
state.dep_data_mod.pi = __uint_as_float(0x40490FDBU);
state.dep_data_mod.threepi = __uint_as_float(0x4116CBE4U);
state.dep_data_mod.gravity = __uint_as_float(0x411CF5C3U);
state.dep_data_mod.grav = __uint_as_float(0x411CE647U);
state.dep_data_mod.boltzmann = __uint_as_float(0x251F2F12U);
state.dep_data_mod.rgasuniv = __uint_as_float(0x4105083CU);
state.dep_data_mod.avo = __uint_as_float(0x66FF0C23U);
state.dep_data_mod.cb = __uint_as_float(0x40000000U);
state.dep_data_mod.cim = __uint_as_float(0x3ECCCCCDU);
state.dep_data_mod.alpha = __uint_as_float(0x3F4CCCCDU);
state.dep_data_mod.cin = __uint_as_float(0x40200000U);
state.dep_data_mod.vv = __uint_as_float(0x3F4CCCCDU);
state.dep_data_mod.a_for = __uint_as_float(0x3DCCCCCDU);
state.dep_data_mod.a_grs = __uint_as_float(0x3E4CCCCDU);
state.dep_data_mod.a_wat = __uint_as_float(0x42C80000U);
state.dep_data_mod.eps0_for = __uint_as_float(0x3C03126FU);
state.dep_data_mod.eps0_grs = __uint_as_float(0x3B83126FU);
state.dep_data_mod.eps0_wat = __uint_as_float(0x3BC49BA6U);
state.dep_data_mod.one3 = __uint_as_float(0x3EAAAAABU);
state.dep_data_mod.two3 = __uint_as_float(0x3F2AAAABU);
state.dep_data_mod.sqrt2 = __uint_as_float(0x3FB504F3U);
state.dep_data_mod.sqrtpi = __uint_as_float(0x3FE2DFC5U);
state.dep_data_mod.karman = __uint_as_float(0x3ECCCCCDU);
state.dep_data_mod.conmin = __uint_as_float(0x24E69595U);
state.dep_data_mod.pirs = __uint_as_float(0x40490FDBU);
state.dep_data_mod.f6dpi = __uint_as_float(0x3FF47645U);
state.dep_data_mod.f6dpim9 = __uint_as_float(0x31033E99U);
state.dep_data_mod.sigma1 = __uint_as_float(0x3FE66666U);
state.dep_data_mod.mean_diameter1 = __uint_as_float(0x332BCC77U);
state.dep_data_mod.fact_wfa = __uint_as_float(0x57007A44U);
state.dep_data_mod.dgmin = __uint_as_float(0x3089705FU);
state.dep_data_mod.densmin = __uint_as_float(0x447A0000U);
state.dep_data_mod.aero_dens = storage.alloc<float>(1,1000);
sd_assign(state.dep_data_mod.aero_dens,0,storage);
state.dep_data_mod.aero_diam = storage.alloc<float>(1,1000);
sd_assign(state.dep_data_mod.aero_diam,0,storage);
plume.initialized = false;
plume.w = storage.alloc<float>(1,200);
plume.t = storage.alloc<float>(1,200);
plume.qv = storage.alloc<float>(1,200);
plume.qc = storage.alloc<float>(1,200);
plume.qh = storage.alloc<float>(1,200);
plume.qi = storage.alloc<float>(1,200);
plume.sc = storage.alloc<float>(1,200);
plume.vth = storage.alloc<float>(1,200);
plume.vti = storage.alloc<float>(1,200);
plume.rho = storage.alloc<float>(1,200);
plume.txs = storage.alloc<float>(1,200);
plume.est = storage.alloc<float>(1,200);
plume.qsat = storage.alloc<float>(1,200);
plume.wc = storage.alloc<float>(1,200);
plume.wt = storage.alloc<float>(1,200);
plume.tt = storage.alloc<float>(1,200);
plume.qvt = storage.alloc<float>(1,200);
plume.qct = storage.alloc<float>(1,200);
plume.qht = storage.alloc<float>(1,200);
plume.qit = storage.alloc<float>(1,200);
plume.sct = storage.alloc<float>(1,200);
plume.dzm = storage.alloc<float>(1,200);
plume.dzt = storage.alloc<float>(1,200);
plume.zm = storage.alloc<float>(1,200);
plume.zt = storage.alloc<float>(1,200);
plume.vctr1 = storage.alloc<float>(1,200);
plume.vctr2 = storage.alloc<float>(1,200);
plume.vt3dc = storage.alloc<float>(1,200);
plume.vt3df = storage.alloc<float>(1,200);
plume.vt3dk = storage.alloc<float>(1,200);
plume.vt3dg = storage.alloc<float>(1,200);
plume.scr1 = storage.alloc<float>(1,200);
plume.pke = storage.alloc<float>(1,200);
plume.the = storage.alloc<float>(1,200);
plume.thve = storage.alloc<float>(1,200);
plume.thee = storage.alloc<float>(1,200);
plume.pe = storage.alloc<float>(1,200);
plume.te = storage.alloc<float>(1,200);
plume.qvenv = storage.alloc<float>(1,200);
plume.dne = storage.alloc<float>(1,200);
plume.ucon = storage.alloc<float>(1,200);
plume.vcon = storage.alloc<float>(1,200);
plume.thtcon = storage.alloc<float>(1,200);
plume.rvcon = storage.alloc<float>(1,200);
plume.picon = storage.alloc<float>(1,200);
plume.tmpcon = storage.alloc<float>(1,200);
plume.zcon = storage.alloc<float>(1,200);
plume.zzcon = storage.alloc<float>(1,200);
plume.visc = storage.alloc<float>(1,200);
plume.cvh = storage.alloc<float>(1,200);
plume.cvi = storage.alloc<float>(1,200);
plume.radius = storage.alloc<float>(1,200);
plume.heating = storage.alloc<float>(1,200);
plume.upe = storage.alloc<float>(1,200);
plume.vpe = storage.alloc<float>(1,200);
plume.vel_e = storage.alloc<float>(1,200);
plume.vel_p = storage.alloc<float>(1,200);
plume.rad_p = storage.alloc<float>(1,200);
plume.vel_t = storage.alloc<float>(1,200);
plume.rad_t = storage.alloc<float>(1,200);
plume.ztop_ = storage.alloc<float>(1,200);
state.saved_diurnal_cycle_icall = 0;
state.saved_add_emis_burn_icall = 0;
state.saved_makeplume_delz_thresold = 100.f;
state.saved_visc_w_printed = false;
state.saved_dry_dep_driver_emerson_icall = 0;
}
// st: element strides of a pool view (the provider's own words); zero means a contiguous Fortran array.
struct Argument {void *data; int lo[4],hi[4]; int ivalue;float fvalue;int present;int st[4];};
SDDEV FArray<float> sd_arg_float(Argument &a,int rank) {FArray<float> r;r.p=(float*)a.data;r.rank=rank;int st=1;for(int i=0;i<rank;i++){r.lo[i]=a.lo[i];r.hi[i]=a.hi[i];r.stride[i]=a.st[0]?a.st[i]:st;st*=a.hi[i]-a.lo[i]+1;}return r;}
SDDEV FArray<int> sd_arg_int(Argument &a,int rank) {FArray<int> r;r.p=(int*)a.data;r.rank=rank;int st=1;for(int i=0;i<rank;i++){r.lo[i]=a.lo[i];r.hi[i]=a.hi[i];r.stride[i]=a.st[0]?a.st[i]:st;st*=a.hi[i]-a.lo[i]+1;}return r;}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:262-600 @1d6e370e.
extern "C" SDKERNEL void sd_export_wrapper_logs(State *state,int *out) {SDGUARD;
out[0]=state->wrapper_log_count;
for(int i=0;i<state->wrapper_log_count && i<14;++i) out[i+1]=state->wrapper_log_sites[i];
state->wrapper_log_count=0;
}
extern "C" SDKERNEL void sd_sizes(int *out) {SDGUARD;out[0]=sizeof(State);out[1]=sizeof(Plume);out[2]=sizeof(Argument);}
extern "C" SDKERNEL void sd_initialize(State *state,Plume *plume,float *persistent,int capacity) {SDGUARD;Arena arena{persistent,0,capacity};initialize(*state,arena,*plume);}
extern "C" SDKERNEL void sd_run_gocart_seasalt_driver(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float dt = args[0].fvalue;
auto alt = sd_arg_float(args[1],3);
auto t_phy = sd_arg_float(args[2],3);
auto u_phy = sd_arg_float(args[3],3);
auto v_phy = sd_arg_float(args[4],3);
int num_chem = args[5].ivalue;
auto chem = sd_arg_float(args[6],4);
auto rho_phy = sd_arg_float(args[7],3);
auto dz8w = sd_arg_float(args[8],3);
auto u10 = sd_arg_float(args[9],2);
auto v10 = sd_arg_float(args[10],2);
auto ustar = sd_arg_float(args[11],2);
auto p8w = sd_arg_float(args[12],3);
auto tsk = sd_arg_float(args[13],2);
auto xland = sd_arg_float(args[14],2);
auto xlat = sd_arg_float(args[15],2);
auto xlong = sd_arg_float(args[16],2);
auto area = sd_arg_float(args[17],2);
float g = args[18].fvalue;
auto e_ss_out = sd_arg_float(args[19],4);
int num_e_ss_out = args[20].ivalue;
int index_e_ss_out_ssalt_fine = args[21].ivalue;
int index_e_ss_out_ssalt_coarse = args[22].ivalue;
float pi = args[23].fvalue;
int num_emis_seas = args[24].ivalue;
int seas_opt = args[25].ivalue;
int ids = args[26].ivalue;
int ide = args[27].ivalue;
int jds = args[28].ivalue;
int jde = args[29].ivalue;
int kds = args[30].ivalue;
int kde = args[31].ivalue;
int ims = args[32].ivalue;
int ime = args[33].ivalue;
int jms = args[34].ivalue;
int jme = args[35].ivalue;
int kms = args[36].ivalue;
int kme = args[37].ivalue;
int its = args[38].ivalue;
int ite = args[39].ivalue;
int jts = args[40].ivalue;
int jte = args[41].ivalue;
int kts = args[42].ivalue;
int kte = args[43].ivalue;
gocart_seasalt_driver(*state,arena,dt,alt,t_phy,u_phy,v_phy,num_chem,chem,rho_phy,dz8w,u10,v10,ustar,p8w,tsk,xland,xlat,xlong,area,g,e_ss_out,num_e_ss_out,index_e_ss_out_ssalt_fine,index_e_ss_out_ssalt_coarse,pi,num_emis_seas,seas_opt,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_source_ss(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int imx = args[0].ivalue;
int jmx = args[1].ivalue;
int lmx = args[2].ivalue;
int nmx = args[3].ivalue;
float dt1 = args[4].fvalue;
auto tc = sd_arg_float(args[5],4);
float pi = args[6].fvalue;
auto ilwi = sd_arg_int(args[7],2);
auto dxy = sd_arg_float(args[8],1);
auto w10m = sd_arg_float(args[9],2);
auto airmas = sd_arg_float(args[10],3);
auto bems = sd_arg_float(args[11],3);
int ipr = args[12].ivalue;
source_ss(*state,arena,imx,jmx,lmx,nmx,dt1,tc,pi,ilwi,dxy,w10m,airmas,bems,ipr);
}
extern "C" SDKERNEL void sd_run_ebu_driver(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto flam_frac = sd_arg_float(args[0],2);
int kfire = args[1].ivalue;
auto theta_phy = sd_arg_float(args[2],3);
auto q_vap = sd_arg_float(args[3],3);
auto rho_phy = sd_arg_float(args[4],3);
auto vvel = sd_arg_float(args[5],3);
auto u_phy = sd_arg_float(args[6],3);
auto v_phy = sd_arg_float(args[7],3);
auto pi_phy = sd_arg_float(args[8],3);
auto wind_phy = sd_arg_float(args[9],3);
auto z_at_w = sd_arg_float(args[10],3);
auto z = sd_arg_float(args[11],3);
float g = args[12].fvalue;
float con_cp = args[13].fvalue;
float con_rd = args[14].fvalue;
auto frp_inst = sd_arg_float(args[15],2);
auto k_min = sd_arg_int(args[16],2);
auto k_max = sd_arg_int(args[17],2);
int wind_eff_opt = args[18].ivalue;
bool do_plumerise = args[19].ivalue;
auto kpbl_thetav = sd_arg_int(args[20],2);
auto kpbl = sd_arg_int(args[21],2);
float curr_secs = args[22].fvalue;
auto xlat = sd_arg_float(args[23],2);
auto xlong = sd_arg_float(args[24],2);
auto uspdavg2d = sd_arg_float(args[25],2);
auto hpbl2d = sd_arg_float(args[26],2);
float alpha = args[27].fvalue;
float frp_min = args[28].fvalue;
float frp_wthreshold = args[29].fvalue;
float zpbl_lim = args[30].fvalue;
float uspd_lim = args[31].fvalue;
int ids = args[32].ivalue;
int ide = args[33].ivalue;
int jds = args[34].ivalue;
int jde = args[35].ivalue;
int kds = args[36].ivalue;
int kde = args[37].ivalue;
int ims = args[38].ivalue;
int ime = args[39].ivalue;
int jms = args[40].ivalue;
int jme = args[41].ivalue;
int kms = args[42].ivalue;
int kme = args[43].ivalue;
int its = args[44].ivalue;
int ite = args[45].ivalue;
int jts = args[46].ivalue;
int jte = args[47].ivalue;
int kts = args[48].ivalue;
int kte = args[49].ivalue;
FStr errmsg((char*)args[50].data,100);
int errflg = args[51].ivalue;
auto ebu_in = sd_arg_float(args[52],3);
auto ebu = sd_arg_float(args[53],3);
auto ebu_in_coarse = sd_arg_float(args[54],3);
auto ebu_coarse = sd_arg_float(args[55],3);
auto ebu_in_ch4 = sd_arg_float(args[56],3);
auto ebu_ch4 = sd_arg_float(args[57],3);
ebu_driver(*state,arena,flam_frac,kfire,theta_phy,q_vap,rho_phy,vvel,u_phy,v_phy,pi_phy,wind_phy,z_at_w,z,g,con_cp,con_rd,frp_inst,k_min,k_max,wind_eff_opt,do_plumerise,kpbl_thetav,kpbl,curr_secs,xlat,xlong,uspdavg2d,hpbl2d,alpha,frp_min,frp_wthreshold,zpbl_lim,uspd_lim,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,errmsg,errflg,ebu_in,ebu,ebu_in_coarse,ebu_coarse,ebu_in_ch4,ebu_ch4,bool(args[52].present),bool(args[53].present),bool(args[54].present),bool(args[55].present),bool(args[56].present),bool(args[57].present));
args[51].ivalue = errflg;
}
extern "C" SDKERNEL void sd_run_mpas_aod_diag(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto chem = sd_arg_float(args[0],4);
auto aod3d = sd_arg_float(args[1],3);
auto rho_phy = sd_arg_float(args[2],3);
auto dz8w = sd_arg_float(args[3],3);
int num_chem = args[4].ivalue;
int ids = args[5].ivalue;
int ide = args[6].ivalue;
int jds = args[7].ivalue;
int jde = args[8].ivalue;
int kds = args[9].ivalue;
int kde = args[10].ivalue;
int ims = args[11].ivalue;
int ime = args[12].ivalue;
int jms = args[13].ivalue;
int jme = args[14].ivalue;
int kms = args[15].ivalue;
int kme = args[16].ivalue;
int its = args[17].ivalue;
int ite = args[18].ivalue;
int jts = args[19].ivalue;
int jte = args[20].ivalue;
int kts = args[21].ivalue;
int kte = args[22].ivalue;
mpas_aod_diag(*state,arena,chem,aod3d,rho_phy,dz8w,num_chem,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_mpas_visibility_diag(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto qcloud = sd_arg_float(args[0],3);
auto qrain = sd_arg_float(args[1],3);
auto qice = sd_arg_float(args[2],3);
auto qsnow = sd_arg_float(args[3],3);
auto qgrpl = sd_arg_float(args[4],3);
auto blcldw = sd_arg_float(args[5],3);
auto blcldi = sd_arg_float(args[6],3);
auto rho_phy = sd_arg_float(args[7],3);
auto wind10m = sd_arg_float(args[8],2);
auto wind = sd_arg_float(args[9],3);
auto rh2m = sd_arg_float(args[10],2);
auto rh = sd_arg_float(args[11],3);
auto qv = sd_arg_float(args[12],3);
auto t2m = sd_arg_float(args[13],2);
auto t = sd_arg_float(args[14],3);
auto coszen = sd_arg_float(args[15],2);
auto extcoef55 = sd_arg_float(args[16],3);
auto vis = sd_arg_float(args[17],2);
int ids = args[18].ivalue;
int ide = args[19].ivalue;
int jds = args[20].ivalue;
int jde = args[21].ivalue;
int kds = args[22].ivalue;
int kde = args[23].ivalue;
int ims = args[24].ivalue;
int ime = args[25].ivalue;
int jms = args[26].ivalue;
int jme = args[27].ivalue;
int kms = args[28].ivalue;
int kme = args[29].ivalue;
int its = args[30].ivalue;
int ite = args[31].ivalue;
int jts = args[32].ivalue;
int jte = args[33].ivalue;
int kts = args[34].ivalue;
int kte = args[35].ivalue;
mpas_visibility_diag(*state,arena,qcloud,qrain,qice,qsnow,qgrpl,blcldw,blcldi,rho_phy,wind10m,wind,rh2m,rh,qv,t2m,t,coszen,extcoef55,vis,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_compute_emission_factors(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto efs_map = sd_arg_float(args[0],2);
auto vegfrac = sd_arg_float(args[1],3);
int bb_beta = args[2].ivalue;
auto eco_id = sd_arg_int(args[3],2);
auto efs_smold = sd_arg_float(args[4],2);
auto efs_flam = sd_arg_float(args[5],2);
auto efs_rsmold = sd_arg_float(args[6],2);
auto fmc_avg = sd_arg_float(args[7],2);
auto wetness = sd_arg_float(args[8],2);
int nlcat = args[9].ivalue;
float dt = args[10].fvalue;
float gmt = args[11].fvalue;
int ids = args[12].ivalue;
int ide = args[13].ivalue;
int jds = args[14].ivalue;
int jde = args[15].ivalue;
int kds = args[16].ivalue;
int kde = args[17].ivalue;
int ims = args[18].ivalue;
int ime = args[19].ivalue;
int jms = args[20].ivalue;
int jme = args[21].ivalue;
int kms = args[22].ivalue;
int kme = args[23].ivalue;
int its = args[24].ivalue;
int ite = args[25].ivalue;
int jts = args[26].ivalue;
int jte = args[27].ivalue;
int kts = args[28].ivalue;
int kte = args[29].ivalue;
compute_emission_factors(*state,arena,efs_map,vegfrac,bb_beta,eco_id,efs_smold,efs_flam,efs_rsmold,fmc_avg,wetness,nlcat,dt,gmt,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_calculate_smoke_emissions(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float dt = args[0].fvalue;
int julday = args[1].ivalue;
int nlcat = args[2].ivalue;
auto efs_map = sd_arg_float(args[3],2);
auto fre_avg = sd_arg_float(args[4],3);
int ebb_dcycle = args[5].ivalue;
auto area = sd_arg_float(args[6],2);
int nblocks = args[7].ivalue;
int ktau = args[8].ivalue;
int bb_input_prevh = args[9].ivalue;
auto ebu = sd_arg_float(args[10],3);
int ids = args[11].ivalue;
int ide = args[12].ivalue;
int jds = args[13].ivalue;
int jde = args[14].ivalue;
int kds = args[15].ivalue;
int kde = args[16].ivalue;
int ims = args[17].ivalue;
int ime = args[18].ivalue;
int jms = args[19].ivalue;
int jme = args[20].ivalue;
int kms = args[21].ivalue;
int kme = args[22].ivalue;
int its = args[23].ivalue;
int ite = args[24].ivalue;
int jts = args[25].ivalue;
int jte = args[26].ivalue;
int kts = args[27].ivalue;
int kte = args[28].ivalue;
calculate_smoke_emissions(*state,arena,dt,julday,nlcat,efs_map,fre_avg,ebb_dcycle,area,nblocks,ktau,bb_input_prevh,ebu,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_diurnal_cycle(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float dtstep = args[0].fvalue;
auto dz8w = sd_arg_float(args[1],3);
auto rho_phy = sd_arg_float(args[2],3);
float pi = args[3].fvalue;
float ebb_min = args[4].fvalue;
auto chem = sd_arg_float(args[5],4);
int num_chem = args[6].ivalue;
int julday = args[7].ivalue;
float gmt = args[8].fvalue;
auto xlat = sd_arg_float(args[9],2);
auto xlong = sd_arg_float(args[10],2);
auto fire_end_hr = sd_arg_float(args[11],2);
auto peak_hr = sd_arg_float(args[12],2);
float time_int = args[13].fvalue;
auto coef_bb_dc = sd_arg_float(args[14],2);
auto fire_hist = sd_arg_float(args[15],2);
auto hwp = sd_arg_float(args[16],2);
auto hwp_avg = sd_arg_float(args[17],2);
auto hwp_prev_day = sd_arg_float(args[18],2);
auto vegfrac = sd_arg_float(args[19],3);
auto eco_id = sd_arg_int(args[20],2);
int nblocks = args[21].ivalue;
auto lu_nofire = sd_arg_float(args[22],2);
auto lu_qfire = sd_arg_float(args[23],2);
auto lu_sfire = sd_arg_float(args[24],2);
auto swdown = sd_arg_float(args[25],2);
int ebb_dcycle = args[26].ivalue;
auto ebu = sd_arg_float(args[27],3);
auto fire_type = sd_arg_int(args[28],2);
auto q_vap = sd_arg_float(args[29],3);
bool add_fire_moist_flux = args[30].ivalue;
float plume_beta_qv = args[31].fvalue;
float hwp_alpha = args[32].fvalue;
int ids = args[33].ivalue;
int ide = args[34].ivalue;
int jds = args[35].ivalue;
int jde = args[36].ivalue;
int kds = args[37].ivalue;
int kde = args[38].ivalue;
int ims = args[39].ivalue;
int ime = args[40].ivalue;
int jms = args[41].ivalue;
int jme = args[42].ivalue;
int kms = args[43].ivalue;
int kme = args[44].ivalue;
int its = args[45].ivalue;
int ite = args[46].ivalue;
int jts = args[47].ivalue;
int jte = args[48].ivalue;
int kts = args[49].ivalue;
int kte = args[50].ivalue;
diurnal_cycle(*state,arena,dtstep,dz8w,rho_phy,pi,ebb_min,chem,num_chem,julday,gmt,xlat,xlong,fire_end_hr,peak_hr,time_int,coef_bb_dc,fire_hist,hwp,hwp_avg,hwp_prev_day,vegfrac,eco_id,nblocks,lu_nofire,lu_qfire,lu_sfire,swdown,ebb_dcycle,ebu,fire_type,q_vap,add_fire_moist_flux,plume_beta_qv,hwp_alpha,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_add_emis_burn(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float dtstep = args[0].fvalue;
auto dz8w = sd_arg_float(args[1],3);
auto rho_phy = sd_arg_float(args[2],3);
float pi = args[3].fvalue;
float ebb_min = args[4].fvalue;
auto chem = sd_arg_float(args[5],4);
int num_chem = args[6].ivalue;
int julday = args[7].ivalue;
float gmt = args[8].fvalue;
auto xlat = sd_arg_float(args[9],2);
auto xlong = sd_arg_float(args[10],2);
auto fire_end_hr = sd_arg_float(args[11],2);
auto peak_hr = sd_arg_float(args[12],2);
float time_int = args[13].fvalue;
auto coef_bb_dc = sd_arg_float(args[14],2);
auto fire_hist = sd_arg_float(args[15],2);
auto hwp = sd_arg_float(args[16],2);
auto hwp_prevd = sd_arg_float(args[17],2);
auto swdown = sd_arg_float(args[18],2);
int ebb_dcycle = args[19].ivalue;
auto ebu = sd_arg_float(args[20],3);
auto ebu_coarse = sd_arg_float(args[21],3);
auto ebu_ch4 = sd_arg_float(args[22],3);
auto fire_type = sd_arg_int(args[23],2);
auto q_vap = sd_arg_float(args[24],3);
bool add_fire_moist_flux = args[25].ivalue;
float sc_factor = args[26].fvalue;
auto aod3d = sd_arg_float(args[27],3);
int ids = args[28].ivalue;
int ide = args[29].ivalue;
int jds = args[30].ivalue;
int jde = args[31].ivalue;
int kds = args[32].ivalue;
int kde = args[33].ivalue;
int ims = args[34].ivalue;
int ime = args[35].ivalue;
int jms = args[36].ivalue;
int jme = args[37].ivalue;
int kms = args[38].ivalue;
int kme = args[39].ivalue;
int its = args[40].ivalue;
int ite = args[41].ivalue;
int jts = args[42].ivalue;
int jte = args[43].ivalue;
int kts = args[44].ivalue;
int kte = args[45].ivalue;
add_emis_burn(*state,arena,dtstep,dz8w,rho_phy,pi,ebb_min,chem,num_chem,julday,gmt,xlat,xlong,fire_end_hr,peak_hr,time_int,coef_bb_dc,fire_hist,hwp,hwp_prevd,swdown,ebb_dcycle,ebu,ebu_coarse,ebu_ch4,fire_type,q_vap,add_fire_moist_flux,sc_factor,aod3d,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_plumerise(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int m1 = args[0].ivalue;
int m2 = args[1].ivalue;
int m3 = args[2].ivalue;
int ia = args[3].ivalue;
int iz = args[4].ivalue;
int ja = args[5].ivalue;
int jz = args[6].ivalue;
auto up = sd_arg_float(args[7],3);
auto vp = sd_arg_float(args[8],3);
auto wp = sd_arg_float(args[9],3);
auto theta = sd_arg_float(args[10],3);
auto pp = sd_arg_float(args[11],3);
auto dn0 = sd_arg_float(args[12],3);
auto rv = sd_arg_float(args[13],3);
auto zt_rams = sd_arg_float(args[14],1);
auto zm_rams = sd_arg_float(args[15],1);
int wind_eff_opt = args[16].ivalue;
float frp_inst = args[17].fvalue;
int k1 = args[18].ivalue;
int k2 = args[19].ivalue;
bool dbg_opt = args[20].ivalue;
float g = args[21].fvalue;
float cp = args[22].fvalue;
float rgas = args[23].fvalue;
float cpor = args[24].fvalue;
FStr errmsg((char*)args[25].data,100);
int errflg = args[26].ivalue;
int icall = args[27].ivalue;
float lat = args[28].fvalue;
float f_long = args[29].fvalue;
float curr_secs = args[30].fvalue;
float alpha = args[31].fvalue;
float frp_min = args[32].fvalue;
plumerise(*state,arena,m1,m2,m3,ia,iz,ja,jz,up,vp,wp,theta,pp,dn0,rv,zt_rams,zm_rams,wind_eff_opt,frp_inst,k1,k2,dbg_opt,g,cp,rgas,cpor,errmsg,errflg,icall,lat,f_long,curr_secs,alpha,frp_min);
args[18].ivalue = k1;
args[19].ivalue = k2;
args[26].ivalue = errflg;
}
extern "C" SDKERNEL void sd_run_set_flam_vert(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto ztopmax = sd_arg_float(args[0],1);
int k1 = args[1].ivalue;
int k2 = args[2].ivalue;
int nkp = args[3].ivalue;
auto zzcon = sd_arg_float(args[4],1);
set_flam_vert(*state,arena,ztopmax,k1,k2,nkp,zzcon);
args[1].ivalue = k1;
args[2].ivalue = k2;
}
extern "C" SDKERNEL void sd_run_damp_grav_wave(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int ifrom = args[0].ivalue;
int nm1 = args[1].ivalue;
int deltak = args[2].ivalue;
float dt = args[3].fvalue;
auto zt = sd_arg_float(args[4],1);
auto zm = sd_arg_float(args[5],1);
auto w = sd_arg_float(args[6],1);
auto t = sd_arg_float(args[7],1);
auto tt = sd_arg_float(args[8],1);
auto qv = sd_arg_float(args[9],1);
auto qh = sd_arg_float(args[10],1);
auto qi = sd_arg_float(args[11],1);
auto qc = sd_arg_float(args[12],1);
auto te = sd_arg_float(args[13],1);
auto pe = sd_arg_float(args[14],1);
auto qvenv = sd_arg_float(args[15],1);
damp_grav_wave(*state,arena,ifrom,nm1,deltak,dt,zt,zm,w,t,tt,qv,qh,qi,qc,te,pe,qvenv);
}
extern "C" SDKERNEL void sd_run_friction(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int ifrom = args[0].ivalue;
int nm1 = args[1].ivalue;
int deltak = args[2].ivalue;
float dt = args[3].fvalue;
auto zt = sd_arg_float(args[4],1);
auto zm = sd_arg_float(args[5],1);
auto var1 = sd_arg_float(args[6],1);
auto vart = sd_arg_float(args[7],1);
auto var2 = sd_arg_float(args[8],1);
friction(*state,arena,ifrom,nm1,deltak,dt,zt,zm,var1,vart,var2);
}
extern "C" SDKERNEL void sd_run_vel_advectc_plumerise(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int m1 = args[0].ivalue;
auto wc = sd_arg_float(args[1],1);
auto wt = sd_arg_float(args[2],1);
auto rho = sd_arg_float(args[3],1);
auto dzm = sd_arg_float(args[4],1);
vel_advectc_plumerise(*state,arena,m1,wc,wt,rho,dzm);
}
extern "C" SDKERNEL void sd_run_hadvance_plumerise(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int iac = args[0].ivalue;
int m1 = args[1].ivalue;
float dt = args[2].fvalue;
auto wc = sd_arg_float(args[3],1);
auto wt = sd_arg_float(args[4],1);
auto wp = sd_arg_float(args[5],1);
int mintime = args[6].ivalue;
hadvance_plumerise(*state,arena,iac,m1,dt,wc,wt,wp,mintime);
}
extern "C" SDKERNEL void sd_run_predict_plumerise(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int npts = args[0].ivalue;
auto ac = sd_arg_float(args[1],1);
auto ap = sd_arg_float(args[2],1);
auto fa = sd_arg_float(args[3],1);
auto af = sd_arg_float(args[4],1);
int iac = args[5].ivalue;
float dtlp = args[6].fvalue;
float epsu = args[7].fvalue;
predict_plumerise(*state,arena,npts,ac,ap,fa,af,iac,dtlp,epsu);
}
extern "C" SDKERNEL void sd_run_buoyancy_plumerise(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int m1 = args[0].ivalue;
auto t = sd_arg_float(args[1],1);
auto te = sd_arg_float(args[2],1);
auto qv = sd_arg_float(args[3],1);
auto qvenv = sd_arg_float(args[4],1);
auto qh = sd_arg_float(args[5],1);
auto qi = sd_arg_float(args[6],1);
auto qc = sd_arg_float(args[7],1);
auto wt = sd_arg_float(args[8],1);
auto scr1 = sd_arg_float(args[9],1);
buoyancy_plumerise(*state,arena,m1,t,te,qv,qvenv,qh,qi,qc,wt,scr1);
}
extern "C" SDKERNEL void sd_run_fa_zc_plumerise(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int m1 = args[0].ivalue;
auto scp = sd_arg_float(args[1],1);
auto scr1 = sd_arg_float(args[2],1);
auto vt3dc = sd_arg_float(args[3],1);
auto vt3df = sd_arg_float(args[4],1);
auto vt3dg = sd_arg_float(args[5],1);
auto vt3dk = sd_arg_float(args[6],1);
auto vctr1 = sd_arg_float(args[7],1);
auto vctr2 = sd_arg_float(args[8],1);
fa_zc_plumerise(*state,arena,m1,scp,scr1,vt3dc,vt3df,vt3dg,vt3dk,vctr1,vctr2);
}
extern "C" SDKERNEL void sd_run_advtndc_plumerise(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int m1 = args[0].ivalue;
auto scp = sd_arg_float(args[1],1);
auto sca = sd_arg_float(args[2],1);
auto sct = sd_arg_float(args[3],1);
float dtl = args[4].fvalue;
advtndc_plumerise(*state,arena,m1,scp,sca,sct,dtl);
}
extern "C" SDKERNEL void sd_run_scl_dyn_entrain(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int m1 = args[0].ivalue;
int nkp = args[1].ivalue;
float wbar = args[2].fvalue;
auto w = sd_arg_float(args[3],1);
float adiabat = args[4].fvalue;
float alpha = args[5].fvalue;
auto radius = sd_arg_float(args[6],1);
auto tt = sd_arg_float(args[7],1);
auto t = sd_arg_float(args[8],1);
auto te = sd_arg_float(args[9],1);
auto qvt = sd_arg_float(args[10],1);
auto qv = sd_arg_float(args[11],1);
auto qvenv = sd_arg_float(args[12],1);
auto qct = sd_arg_float(args[13],1);
auto qc = sd_arg_float(args[14],1);
auto qht = sd_arg_float(args[15],1);
auto qh = sd_arg_float(args[16],1);
auto qit = sd_arg_float(args[17],1);
auto qi = sd_arg_float(args[18],1);
auto vel_e = sd_arg_float(args[19],1);
auto vel_p = sd_arg_float(args[20],1);
auto vel_t = sd_arg_float(args[21],1);
auto rad_p = sd_arg_float(args[22],1);
auto rad_t = sd_arg_float(args[23],1);
scl_dyn_entrain(*state,arena,m1,nkp,wbar,w,adiabat,alpha,radius,tt,t,te,qvt,qv,qvenv,qct,qc,qht,qh,qit,qi,vel_e,vel_p,vel_t,rad_p,rad_t);
args[2].fvalue = wbar;
args[4].fvalue = adiabat;
}
extern "C" SDKERNEL void sd_run_htint(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int nzz1 = args[0].ivalue;
auto vctra = sd_arg_float(args[1],1);
auto eleva = sd_arg_float(args[2],1);
int nzz2 = args[3].ivalue;
auto vctrb = sd_arg_float(args[4],1);
auto elevb = sd_arg_float(args[5],1);
FStr errmsg((char*)args[6].data,100);
int errflg = args[7].ivalue;
htint(*state,arena,nzz1,vctra,eleva,nzz2,vctrb,elevb,errmsg,errflg);
args[7].ivalue = errflg;
}
extern "C" SDKERNEL void sd_run_dry_dep_driver_simple(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto rmol = sd_arg_float(args[0],2);
auto ust = sd_arg_float(args[1],2);
int ndvel = args[2].ivalue;
auto ddvel = sd_arg_float(args[3],3);
int ids = args[4].ivalue;
int ide = args[5].ivalue;
int jds = args[6].ivalue;
int jde = args[7].ivalue;
int kds = args[8].ivalue;
int kde = args[9].ivalue;
int ims = args[10].ivalue;
int ime = args[11].ivalue;
int jms = args[12].ivalue;
int jme = args[13].ivalue;
int kms = args[14].ivalue;
int kme = args[15].ivalue;
int its = args[16].ivalue;
int ite = args[17].ivalue;
int jts = args[18].ivalue;
int jte = args[19].ivalue;
int kts = args[20].ivalue;
int kte = args[21].ivalue;
dry_dep_driver_simple(*state,arena,rmol,ust,ndvel,ddvel,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_aero_rad_init(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
aero_rad_init(*state,arena);
}
extern "C" SDKERNEL void sd_run_aero_dry_dep_init(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
aero_dry_dep_init(*state,arena);
}
extern "C" SDKERNEL void sd_run_gocart_dust_fengsha_driver(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float dt = args[0].fvalue;
int ktau = args[1].ivalue;
auto chem = sd_arg_float(args[2],4);
auto rho_phy = sd_arg_float(args[3],3);
auto smois = sd_arg_float(args[4],3);
auto stemp = sd_arg_float(args[5],3);
auto p8w = sd_arg_float(args[6],3);
auto isltyp = sd_arg_int(args[7],2);
auto snowh = sd_arg_float(args[8],2);
auto xland = sd_arg_float(args[9],2);
auto area = sd_arg_float(args[10],2);
float g = args[11].fvalue;
auto ust = sd_arg_float(args[12],2);
auto znt = sd_arg_float(args[13],2);
auto clay = sd_arg_float(args[14],2);
auto sand = sd_arg_float(args[15],2);
auto uthr = sd_arg_float(args[16],2);
auto uthr_sg = sd_arg_float(args[17],2);
auto albedo_drag = sd_arg_float(args[18],2);
auto feff = sd_arg_float(args[19],2);
auto sep = sd_arg_float(args[20],2);
auto e_dust_out = sd_arg_float(args[21],4);
int num_e_dust_out = args[22].ivalue;
int index_e_dust_out_dust_fine = args[23].ivalue;
int index_e_dust_out_dust_coarse = args[24].ivalue;
int num_emis_dust = args[25].ivalue;
int num_chem = args[26].ivalue;
int num_soil_layers = args[27].ivalue;
float dust_alpha = args[28].fvalue;
float dust_gamma = args[29].fvalue;
float dust_drylimit_factor = args[30].fvalue;
float dust_moist_correction = args[31].fvalue;
int ids = args[32].ivalue;
int ide = args[33].ivalue;
int jds = args[34].ivalue;
int jde = args[35].ivalue;
int kds = args[36].ivalue;
int kde = args[37].ivalue;
int ims = args[38].ivalue;
int ime = args[39].ivalue;
int jms = args[40].ivalue;
int jme = args[41].ivalue;
int kms = args[42].ivalue;
int kme = args[43].ivalue;
int its = args[44].ivalue;
int ite = args[45].ivalue;
int jts = args[46].ivalue;
int jte = args[47].ivalue;
int kts = args[48].ivalue;
int kte = args[49].ivalue;
gocart_dust_fengsha_driver(*state,arena,dt,ktau,chem,rho_phy,smois,stemp,p8w,isltyp,snowh,xland,area,g,ust,znt,clay,sand,uthr,uthr_sg,albedo_drag,feff,sep,e_dust_out,num_e_dust_out,index_e_dust_out_dust_fine,index_e_dust_out_dust_coarse,num_emis_dust,num_chem,num_soil_layers,dust_alpha,dust_gamma,dust_drylimit_factor,dust_moist_correction,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_source_dust(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int imx = args[0].ivalue;
int jmx = args[1].ivalue;
int lmx = args[2].ivalue;
int nmx = args[3].ivalue;
float dt1 = args[4].fvalue;
auto tc = sd_arg_float(args[5],1);
float ustar = args[6].fvalue;
auto massfrac = sd_arg_float(args[7],1);
float erod = args[8].fvalue;
float dxy = args[9].fvalue;
float smois = args[10].fvalue;
float airden = args[11].fvalue;
float airmas = args[12].fvalue;
auto bems = sd_arg_float(args[13],1);
float g0 = args[14].fvalue;
float alpha = args[15].fvalue;
float gamma = args[16].fvalue;
float r = args[17].fvalue;
float uthres = args[18].fvalue;
float source_out_diag = args[19].fvalue;
float dust_drylimit_factor = args[20].fvalue;
source_dust(*state,arena,imx,jmx,lmx,nmx,dt1,tc,ustar,massfrac,erod,dxy,smois,airden,airmas,bems,g0,alpha,gamma,r,uthres,source_out_diag,dust_drylimit_factor);
args[19].fvalue = source_out_diag;
}
extern "C" SDKERNEL void sd_run_fengsha_drag(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float z0 = args[0].fvalue;
float r = args[1].fvalue;
fengsha_drag(*state,arena,z0,r);
args[1].fvalue = r;
}
extern "C" SDKERNEL void sd_run_dustemissionfengsha(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float slc = args[0].fvalue;
float clay = args[1].fvalue;
float sand = args[2].fvalue;
float silt = args[3].fvalue;
float sep = args[4].fvalue;
float feff = args[5].fvalue;
float airdens = args[6].fvalue;
float ustar = args[7].fvalue;
float uthrs = args[8].fvalue;
float alpha = args[9].fvalue;
float gamma = args[10].fvalue;
float kvhmax = args[11].fvalue;
float grav = args[12].fvalue;
float rhop = args[13].fvalue;
float emissions = args[14].fvalue;
float reason = args[15].fvalue;
float dust_drylimit_factor = args[16].fvalue;
dustemissionfengsha(*state,arena,slc,clay,sand,silt,sep,feff,airdens,ustar,uthrs,alpha,gamma,kvhmax,grav,rhop,emissions,reason,dust_drylimit_factor);
args[14].fvalue = emissions;
args[15].fvalue = reason;
}
extern "C" SDKERNEL void sd_run_wetdep_ls(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float dt = args[0].fvalue;
float gravity = args[1].fvalue;
auto var = sd_arg_float(args[2],4);
auto rain = sd_arg_float(args[3],2);
auto moist = sd_arg_float(args[4],3);
auto rho = sd_arg_float(args[5],3);
int nchem = args[6].ivalue;
auto dz8w = sd_arg_float(args[7],3);
auto vvel = sd_arg_float(args[8],3);
auto p_phy = sd_arg_float(args[9],3);
float alpha = args[10].fvalue;
auto wetdep_resolved = sd_arg_float(args[11],3);
int ids = args[12].ivalue;
int ide = args[13].ivalue;
int jds = args[14].ivalue;
int jde = args[15].ivalue;
int kds = args[16].ivalue;
int kde = args[17].ivalue;
int ims = args[18].ivalue;
int ime = args[19].ivalue;
int jms = args[20].ivalue;
int jme = args[21].ivalue;
int kms = args[22].ivalue;
int kme = args[23].ivalue;
int its = args[24].ivalue;
int ite = args[25].ivalue;
int jts = args[26].ivalue;
int jte = args[27].ivalue;
int kts = args[28].ivalue;
int kte = args[29].ivalue;
wetdep_ls(*state,arena,dt,gravity,var,rain,moist,rho,nchem,dz8w,vvel,p_phy,alpha,wetdep_resolved,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_set_scalar_indices(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int chemistry_start = args[0].ivalue;
int index_smoke_fine = args[1].ivalue;
int index_smoke_coarse = args[2].ivalue;
int index_polp_tree = args[3].ivalue;
int index_polp_grass = args[4].ivalue;
int index_polp_weed = args[5].ivalue;
int index_pols_tree = args[6].ivalue;
int index_pols_grass = args[7].ivalue;
int index_pols_weed = args[8].ivalue;
int index_pols_all = args[9].ivalue;
int index_polp_all = args[10].ivalue;
int index_unspc_fine = args[11].ivalue;
int index_unspc_coarse = args[12].ivalue;
int index_ssalt_fine = args[13].ivalue;
int index_ssalt_coarse = args[14].ivalue;
int index_no3_a_fine = args[15].ivalue;
int index_so4_a_fine = args[16].ivalue;
int index_nh4_a_fine = args[17].ivalue;
int index_so2 = args[18].ivalue;
int index_nh3 = args[19].ivalue;
int index_ch4 = args[20].ivalue;
int index_dust_fine = args[21].ivalue;
int index_dust_coarse = args[22].ivalue;
set_scalar_indices(*state,arena,chemistry_start,index_smoke_fine,index_smoke_coarse,index_polp_tree,index_polp_grass,index_polp_weed,index_pols_tree,index_pols_grass,index_pols_weed,index_pols_all,index_polp_all,index_unspc_fine,index_unspc_coarse,index_ssalt_fine,index_ssalt_coarse,index_no3_a_fine,index_so4_a_fine,index_nh4_a_fine,index_so2,index_nh3,index_ch4,index_dust_fine,index_dust_coarse,bool(args[0].present),bool(args[1].present),bool(args[2].present),bool(args[3].present),bool(args[4].present),bool(args[5].present),bool(args[6].present),bool(args[7].present),bool(args[8].present),bool(args[9].present),bool(args[10].present),bool(args[11].present),bool(args[12].present),bool(args[13].present),bool(args[14].present),bool(args[15].present),bool(args[16].present),bool(args[17].present),bool(args[18].present),bool(args[19].present),bool(args[20].present),bool(args[21].present),bool(args[22].present));
}
extern "C" SDKERNEL void sd_run_seasaltemission(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float rlow = args[0].fvalue;
float rup = args[1].fvalue;
int method = args[2].ivalue;
float w10m = args[3].fvalue;
float ustar = args[4].fvalue;
float pi = args[5].fvalue;
float memissions = args[6].fvalue;
float nemissions = args[7].fvalue;
int rc = args[8].ivalue;
seasaltemission(*state,arena,rlow,rup,method,w10m,ustar,pi,memissions,nemissions,rc);
args[6].fvalue = memissions;
args[7].fvalue = nemissions;
args[8].ivalue = rc;
}
extern "C" SDKERNEL void sd_run_mpas_smoke_anthro_emis_driver(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float dt = args[0].fvalue;
float gmt = args[1].fvalue;
int julday = args[2].ivalue;
int kemit = args[3].ivalue;
auto xlat = sd_arg_float(args[4],2);
auto xlong = sd_arg_float(args[5],2);
auto chem = sd_arg_float(args[6],4);
int num_chem = args[7].ivalue;
auto dz8w = sd_arg_float(args[8],3);
auto t_phy = sd_arg_float(args[9],3);
auto rho_phy = sd_arg_float(args[10],3);
auto e_ant_in = sd_arg_float(args[11],4);
auto e_ant_out = sd_arg_float(args[12],4);
int num_e_ant_in = args[13].ivalue;
int num_e_ant_out = args[14].ivalue;
int index_e_ant_in_unspc_fine = args[15].ivalue;
int index_e_ant_in_unspc_coarse = args[16].ivalue;
int index_e_ant_out_unspc_fine = args[17].ivalue;
int index_e_ant_out_unspc_coarse = args[18].ivalue;
int ids = args[19].ivalue;
int ide = args[20].ivalue;
int jds = args[21].ivalue;
int jde = args[22].ivalue;
int kds = args[23].ivalue;
int kde = args[24].ivalue;
int ims = args[25].ivalue;
int ime = args[26].ivalue;
int jms = args[27].ivalue;
int jme = args[28].ivalue;
int kms = args[29].ivalue;
int kme = args[30].ivalue;
int its = args[31].ivalue;
int ite = args[32].ivalue;
int jts = args[33].ivalue;
int jte = args[34].ivalue;
int kts = args[35].ivalue;
int kte = args[36].ivalue;
mpas_smoke_anthro_emis_driver(*state,arena,dt,gmt,julday,kemit,xlat,xlong,chem,num_chem,dz8w,t_phy,rho_phy,e_ant_in,e_ant_out,num_e_ant_in,num_e_ant_out,index_e_ant_in_unspc_fine,index_e_ant_in_unspc_coarse,index_e_ant_out_unspc_fine,index_e_ant_out_unspc_coarse,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" SDKERNEL void sd_run_dry_dep_driver_emerson(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto rmol = sd_arg_float(args[0],2);
auto ustar = sd_arg_float(args[1],2);
auto znt = sd_arg_float(args[2],2);
int num_chem = args[3].ivalue;
auto ddvel = sd_arg_float(args[4],3);
auto chem = sd_arg_float(args[5],4);
auto delz = sd_arg_float(args[6],3);
auto snowh = sd_arg_float(args[7],2);
auto t_phy = sd_arg_float(args[8],3);
auto p_phy = sd_arg_float(args[9],3);
auto rho_phy = sd_arg_float(args[10],3);
auto ivgtyp = sd_arg_int(args[11],2);
float gravity = args[12].fvalue;
float dt = args[13].fvalue;
auto drydep_flux = sd_arg_float(args[14],3);
auto tend_chem_settle = sd_arg_float(args[15],4);
bool dbg_opt = args[16].ivalue;
int settling_opt = args[17].ivalue;
auto vg = sd_arg_float(args[18],4);
int ids = args[19].ivalue;
int ide = args[20].ivalue;
int jds = args[21].ivalue;
int jde = args[22].ivalue;
int kds = args[23].ivalue;
int kde = args[24].ivalue;
int ims = args[25].ivalue;
int ime = args[26].ivalue;
int jms = args[27].ivalue;
int jme = args[28].ivalue;
int kms = args[29].ivalue;
int kme = args[30].ivalue;
int its = args[31].ivalue;
int ite = args[32].ivalue;
int jts = args[33].ivalue;
int jte = args[34].ivalue;
int kts = args[35].ivalue;
int kte = args[36].ivalue;
float curr_secs = args[37].fvalue;
dry_dep_driver_emerson(*state,arena,rmol,ustar,znt,num_chem,ddvel,chem,delz,snowh,t_phy,p_phy,rho_phy,ivgtyp,gravity,dt,drydep_flux,tend_chem_settle,dbg_opt,settling_opt,vg,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,curr_secs);
}
extern "C" SDKERNEL void sd_run_depvel(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
float rmol = args[0].fvalue;
float zr = args[1].fvalue;
float z0 = args[2].fvalue;
float ustar = args[3].fvalue;
float aer_res = args[4].fvalue;
depvel(*state,arena,rmol,zr,z0,ustar,aer_res);
args[0].fvalue = rmol;
args[4].fvalue = aer_res;
}
extern "C" SDKERNEL void sd_run_particle_settling_wrapper(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto tend_chem_settle = sd_arg_float(args[0],4);
auto chem = sd_arg_float(args[1],4);
auto rho_phy = sd_arg_float(args[2],3);
auto delz_flip = sd_arg_float(args[3],3);
auto vg = sd_arg_float(args[4],4);
float dt = args[5].fvalue;
int kts = args[6].ivalue;
int kte = args[7].ivalue;
int its = args[8].ivalue;
int ite = args[9].ivalue;
int jts = args[10].ivalue;
int jte = args[11].ivalue;
int num_chem = args[12].ivalue;
int ims = args[13].ivalue;
int ime = args[14].ivalue;
int jms = args[15].ivalue;
int jme = args[16].ivalue;
int kms = args[17].ivalue;
int kme = args[18].ivalue;
particle_settling_wrapper(*state,arena,tend_chem_settle,chem,rho_phy,delz_flip,vg,dt,kts,kte,its,ite,jts,jte,num_chem,ims,ime,jms,jme,kms,kme);
}
extern "C" SDKERNEL void sd_run_smoke_from_mpas_fields(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
bool add_fire_heat_flux = args[0].ivalue;
bool add_fire_moist_flux = args[1].ivalue;
auto albedo_drag = sd_arg_float(args[2],1);
auto albedo_drag_p = sd_arg_float(args[3],2);
auto aod3d_p = sd_arg_float(args[4],3);
auto area_p = sd_arg_float(args[5],2);
auto areacell = sd_arg_float(args[6],1);
auto chem = sd_arg_float(args[7],3);
auto chem_p = sd_arg_float(args[8],4);
auto clayfrac_in = sd_arg_float(args[9],1);
auto clayfrac_in_p = sd_arg_float(args[10],2);
auto cldfrac = sd_arg_float(args[11],2);
auto cldfrac_p = sd_arg_float(args[12],3);
auto coef_bb_dc = sd_arg_float(args[13],1);
auto coef_bb_dc_p = sd_arg_float(args[14],2);
FStr config_anthro_scheme((char*)args[15].data,100);
FStr config_convection_scheme((char*)args[16].data,100);
FStr config_dust_scheme((char*)args[17].data,100);
FStr config_rwc_scheme((char*)args[18].data,100);
FStr config_smoke_scheme((char*)args[19].data,100);
auto coszr = sd_arg_float(args[20],1);
auto coszr_p = sd_arg_float(args[21],2);
auto ddvel_p = sd_arg_float(args[22],3);
float degrad = args[23].fvalue;
auto dpt2m_p = sd_arg_float(args[24],2);
auto drydep_flux = sd_arg_float(args[25],2);
auto drydep_flux_p = sd_arg_float(args[26],3);
int drydep_opt = args[27].ivalue;
auto dx_p = sd_arg_float(args[28],2);
auto e_ant_in = sd_arg_float(args[29],3);
auto e_ant_in_p = sd_arg_float(args[30],4);
auto e_ant_out_p = sd_arg_float(args[31],4);
auto e_bb_in = sd_arg_float(args[32],3);
auto e_bb_in_p = sd_arg_float(args[33],4);
auto e_bb_out = sd_arg_float(args[34],3);
auto e_bb_out_p = sd_arg_float(args[35],4);
auto e_dust_out = sd_arg_float(args[36],3);
auto e_dust_out_p = sd_arg_float(args[37],4);
int ebb_dcycle = args[38].ivalue;
auto eco_id = sd_arg_int(args[39],1);
auto eco_id_p = sd_arg_int(args[40],2);
auto efs_flam = sd_arg_float(args[41],1);
auto efs_flam_p = sd_arg_float(args[42],2);
auto efs_rsmold = sd_arg_float(args[43],1);
auto efs_rsmold_p = sd_arg_float(args[44],2);
auto efs_smold = sd_arg_float(args[45],1);
auto efs_smold_p = sd_arg_float(args[46],2);
auto feff = sd_arg_float(args[47],1);
auto feff_p = sd_arg_float(args[48],2);
auto fire_ef_out = sd_arg_float(args[49],1);
auto fire_ef_out_p = sd_arg_float(args[50],2);
auto fire_end_hr = sd_arg_float(args[51],1);
auto fire_end_hr_p = sd_arg_float(args[52],2);
auto fmc_avg_p = sd_arg_float(args[53],2);
auto fmc_prev24 = sd_arg_float(args[54],2);
auto frac_grid_burned = sd_arg_float(args[55],1);
auto frac_grid_burned_p = sd_arg_float(args[56],2);
auto fre_avg_p = sd_arg_float(args[57],3);
auto fre_in = sd_arg_float(args[58],1);
auto fre_in_p = sd_arg_float(args[59],2);
auto fre_out = sd_arg_float(args[60],1);
auto fre_out_p = sd_arg_float(args[61],2);
auto fre_prev24 = sd_arg_float(args[62],2);
auto frp_avg_p = sd_arg_float(args[63],3);
auto frp_in = sd_arg_float(args[64],1);
auto frp_in_p = sd_arg_float(args[65],2);
auto frp_out = sd_arg_float(args[66],1);
auto frp_out_p = sd_arg_float(args[67],2);
auto frp_prev24 = sd_arg_float(args[68],2);
auto hfx_bb_p = sd_arg_float(args[69],2);
auto hpbl = sd_arg_float(args[70],1);
auto hpbl_p = sd_arg_float(args[71],2);
auto hwp = sd_arg_float(args[72],1);
auto hwp_avg_p = sd_arg_float(args[73],3);
auto hwp_p = sd_arg_float(args[74],2);
auto hwp_prev24 = sd_arg_float(args[75],2);
auto isltyp = sd_arg_int(args[76],1);
auto isltyp_p = sd_arg_int(args[77],2);
int ite = args[78].ivalue;
int its = args[79].ivalue;
auto ivgtyp = sd_arg_int(args[80],1);
auto ivgtyp_p = sd_arg_int(args[81],2);
int jte = args[82].ivalue;
int jts = args[83].ivalue;
int kanthro = args[84].ivalue;
auto kpbl = sd_arg_int(args[85],1);
auto kpbl_p = sd_arg_int(args[86],2);
int kreswoodcomb = args[87].ivalue;
int kte = args[88].ivalue;
auto ktop_deep = sd_arg_int(args[89],1);
auto ktop_deep_p = sd_arg_int(args[90],2);
int kts = args[91].ivalue;
int kwildfire = args[92].ivalue;
auto landusef = sd_arg_float(args[93],2);
auto landusef_p = sd_arg_float(args[94],3);
auto latcell = sd_arg_float(args[95],1);
float len_disp = args[96].fvalue;
auto loncell = sd_arg_float(args[97],1);
auto mavail = sd_arg_float(args[98],1);
auto mavail_p = sd_arg_float(args[99],2);
auto max_bb_plume_p = sd_arg_int(args[100],2);
auto meshdensity = sd_arg_float(args[101],1);
auto min_bb_plume_p = sd_arg_int(args[102],2);
int nblocks = args[103].ivalue;
int num_chem = args[104].ivalue;
int num_e_ant_in = args[105].ivalue;
int num_e_ant_out = args[106].ivalue;
int num_e_bb_in = args[107].ivalue;
int num_e_bb_out = args[108].ivalue;
int num_e_dust_out = args[109].ivalue;
int num_landcat = args[110].ivalue;
int num_soils = args[111].ivalue;
int plumerise_opt = args[112].ivalue;
auto qc_bl = sd_arg_float(args[113],2);
auto qcbl_p = sd_arg_float(args[114],3);
auto qfx_bb_p = sd_arg_float(args[115],2);
auto qi_bl = sd_arg_float(args[116],2);
auto qibl_p = sd_arg_float(args[117],3);
auto raincv = sd_arg_float(args[118],1);
auto raincv_p = sd_arg_float(args[119],2);
auto rainncv = sd_arg_float(args[120],1);
auto rainncv_p = sd_arg_float(args[121],2);
auto rmol = sd_arg_float(args[122],1);
auto rmol_p = sd_arg_float(args[123],2);
auto rwc_annual_sum = sd_arg_float(args[124],2);
auto rwc_annual_sum_p = sd_arg_float(args[125],3);
auto rwc_annual_sum_smoke_coarse = sd_arg_float(args[126],2);
auto rwc_annual_sum_smoke_coarse_p = sd_arg_float(args[127],3);
auto rwc_annual_sum_smoke_fine = sd_arg_float(args[128],2);
auto rwc_annual_sum_smoke_fine_p = sd_arg_float(args[129],3);
auto rwc_annual_sum_unspc_coarse = sd_arg_float(args[130],2);
auto rwc_annual_sum_unspc_coarse_p = sd_arg_float(args[131],3);
auto rwc_annual_sum_unspc_fine = sd_arg_float(args[132],2);
auto rwc_annual_sum_unspc_fine_p = sd_arg_float(args[133],3);
auto rwc_denominator = sd_arg_float(args[134],1);
auto rwc_denominator_p = sd_arg_float(args[135],2);
auto sandfrac_in = sd_arg_float(args[136],1);
auto sandfrac_in_p = sd_arg_float(args[137],2);
auto sep_in = sd_arg_float(args[138],1);
auto sep_in_p = sd_arg_float(args[139],2);
auto skintemp = sd_arg_float(args[140],1);
auto smois = sd_arg_float(args[141],2);
auto smois_p = sd_arg_float(args[142],3);
auto snowh = sd_arg_float(args[143],1);
auto snowh_p = sd_arg_float(args[144],2);
int t_size = args[145].ivalue;
auto tend_chem_settle_p = sd_arg_float(args[146],4);
auto totprcp_prev24 = sd_arg_float(args[147],2);
auto totprcp_prev24_p = sd_arg_float(args[148],2);
auto tsk_p = sd_arg_float(args[149],2);
auto tslb = sd_arg_float(args[150],2);
auto tslb_p = sd_arg_float(args[151],3);
auto u10 = sd_arg_float(args[152],1);
auto u10_p = sd_arg_float(args[153],2);
auto ust = sd_arg_float(args[154],1);
auto ust_p = sd_arg_float(args[155],2);
auto uthres_in = sd_arg_float(args[156],1);
auto uthres_in_p = sd_arg_float(args[157],2);
auto uthres_sg_in = sd_arg_float(args[158],1);
auto uthres_sg_in_p = sd_arg_float(args[159],2);
auto v10 = sd_arg_float(args[160],1);
auto v10_p = sd_arg_float(args[161],2);
auto vegfra = sd_arg_float(args[162],1);
auto vegfra_p = sd_arg_float(args[163],2);
int wetdep_ls_opt = args[164].ivalue;
auto wetdep_resolved = sd_arg_float(args[165],2);
auto wetdep_resolved_p = sd_arg_float(args[166],3);
auto xland = sd_arg_float(args[167],1);
auto xland_p = sd_arg_float(args[168],2);
auto xlat_p = sd_arg_float(args[169],2);
auto xlon_p = sd_arg_float(args[170],2);
auto z0 = sd_arg_float(args[171],1);
auto z0_p = sd_arg_float(args[172],2);
auto zgrid = sd_arg_float(args[173],2);
auto zgrid_p = sd_arg_float(args[174],3);
auto znt = sd_arg_float(args[175],1);
auto znt_p = sd_arg_float(args[176],2);
smoke_from_mpas_fields(*state,arena,add_fire_heat_flux,add_fire_moist_flux,albedo_drag,albedo_drag_p,aod3d_p,area_p,areacell,chem,chem_p,clayfrac_in,clayfrac_in_p,cldfrac,cldfrac_p,coef_bb_dc,coef_bb_dc_p,config_anthro_scheme,config_convection_scheme,config_dust_scheme,config_rwc_scheme,config_smoke_scheme,coszr,coszr_p,ddvel_p,degrad,dpt2m_p,drydep_flux,drydep_flux_p,drydep_opt,dx_p,e_ant_in,e_ant_in_p,e_ant_out_p,e_bb_in,e_bb_in_p,e_bb_out,e_bb_out_p,e_dust_out,e_dust_out_p,ebb_dcycle,eco_id,eco_id_p,efs_flam,efs_flam_p,efs_rsmold,efs_rsmold_p,efs_smold,efs_smold_p,feff,feff_p,fire_ef_out,fire_ef_out_p,fire_end_hr,fire_end_hr_p,fmc_avg_p,fmc_prev24,frac_grid_burned,frac_grid_burned_p,fre_avg_p,fre_in,fre_in_p,fre_out,fre_out_p,fre_prev24,frp_avg_p,frp_in,frp_in_p,frp_out,frp_out_p,frp_prev24,hfx_bb_p,hpbl,hpbl_p,hwp,hwp_avg_p,hwp_p,hwp_prev24,isltyp,isltyp_p,ite,its,ivgtyp,ivgtyp_p,jte,jts,kanthro,kpbl,kpbl_p,kreswoodcomb,kte,ktop_deep,ktop_deep_p,kts,kwildfire,landusef,landusef_p,latcell,len_disp,loncell,mavail,mavail_p,max_bb_plume_p,meshdensity,min_bb_plume_p,nblocks,num_chem,num_e_ant_in,num_e_ant_out,num_e_bb_in,num_e_bb_out,num_e_dust_out,num_landcat,num_soils,plumerise_opt,qc_bl,qcbl_p,qfx_bb_p,qi_bl,qibl_p,raincv,raincv_p,rainncv,rainncv_p,rmol,rmol_p,rwc_annual_sum,rwc_annual_sum_p,rwc_annual_sum_smoke_coarse,rwc_annual_sum_smoke_coarse_p,rwc_annual_sum_smoke_fine,rwc_annual_sum_smoke_fine_p,rwc_annual_sum_unspc_coarse,rwc_annual_sum_unspc_coarse_p,rwc_annual_sum_unspc_fine,rwc_annual_sum_unspc_fine_p,rwc_denominator,rwc_denominator_p,sandfrac_in,sandfrac_in_p,sep_in,sep_in_p,skintemp,smois,smois_p,snowh,snowh_p,t_size,tend_chem_settle_p,totprcp_prev24,totprcp_prev24_p,tsk_p,tslb,tslb_p,u10,u10_p,ust,ust_p,uthres_in,uthres_in_p,uthres_sg_in,uthres_sg_in_p,v10,v10_p,vegfra,vegfra_p,wetdep_ls_opt,wetdep_resolved,wetdep_resolved_p,xland,xland_p,xlat_p,xlon_p,z0,z0_p,zgrid,zgrid_p,znt,znt_p);
}
extern "C" SDKERNEL void sd_run_smoke_to_mpas_fields(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
bool add_fire_heat_flux = args[0].ivalue;
bool add_fire_moist_flux = args[1].ivalue;
auto aero_emis_for_enhmix = sd_arg_float(args[2],1);
auto aod3d = sd_arg_float(args[3],2);
auto aod3d_p = sd_arg_float(args[4],3);
auto aod3d_smoke = sd_arg_float(args[5],2);
auto aod3d_smoke_p = sd_arg_float(args[6],3);
auto aod550 = sd_arg_float(args[7],1);
auto aod550_p = sd_arg_float(args[8],2);
auto chem = sd_arg_float(args[9],3);
auto chem_p = sd_arg_float(args[10],4);
auto coef_bb_dc = sd_arg_float(args[11],1);
auto coef_bb_dc_p = sd_arg_float(args[12],2);
FStr config_smoke_scheme((char*)args[13].data,100);
auto ddvel = sd_arg_float(args[14],2);
auto ddvel_p = sd_arg_float(args[15],3);
auto drydep_flux = sd_arg_float(args[16],2);
auto drydep_flux_p = sd_arg_float(args[17],3);
int drydep_opt = args[18].ivalue;
auto e_ant_out = sd_arg_float(args[19],3);
auto e_ant_out_p = sd_arg_float(args[20],4);
auto e_bb_out = sd_arg_float(args[21],3);
auto e_bb_out_p = sd_arg_float(args[22],4);
auto e_dust_out = sd_arg_float(args[23],3);
auto e_dust_out_p = sd_arg_float(args[24],4);
int ebb_dcycle = args[25].ivalue;
auto fire_ef_out = sd_arg_float(args[26],1);
auto fire_ef_out_p = sd_arg_float(args[27],2);
auto frac_grid_burned = sd_arg_float(args[28],1);
auto frac_grid_burned_p = sd_arg_float(args[29],2);
auto fre_out = sd_arg_float(args[30],1);
auto fre_out_p = sd_arg_float(args[31],2);
auto frp_out = sd_arg_float(args[32],1);
auto frp_out_p = sd_arg_float(args[33],2);
auto hfx_bb = sd_arg_float(args[34],1);
auto hfx_bb_p = sd_arg_float(args[35],2);
auto hwp = sd_arg_float(args[36],1);
auto hwp_p = sd_arg_float(args[37],2);
int ite = args[38].ivalue;
int its = args[39].ivalue;
int jte = args[40].ivalue;
int jts = args[41].ivalue;
int kte = args[42].ivalue;
int kts = args[43].ivalue;
auto max_bb_plume = sd_arg_int(args[44],1);
auto max_bb_plume_p = sd_arg_int(args[45],2);
auto min_bb_plume = sd_arg_int(args[46],1);
auto min_bb_plume_p = sd_arg_int(args[47],2);
int num_chem = args[48].ivalue;
int num_e_ant_out = args[49].ivalue;
int num_e_bb_out = args[50].ivalue;
int num_e_dust_out = args[51].ivalue;
int plumerise_opt = args[52].ivalue;
auto pm10 = sd_arg_float(args[53],2);
auto pm10_p = sd_arg_float(args[54],3);
auto pm2_5 = sd_arg_float(args[55],2);
auto pm2_5_p = sd_arg_float(args[56],3);
auto qfx_bb = sd_arg_float(args[57],1);
auto qfx_bb_p = sd_arg_float(args[58],2);
auto tend_chem_settle = sd_arg_float(args[59],3);
auto tend_chem_settle_p = sd_arg_float(args[60],4);
auto vis = sd_arg_float(args[61],1);
auto vis_p = sd_arg_float(args[62],2);
int wetdep_ls_opt = args[63].ivalue;
auto wetdep_resolved = sd_arg_float(args[64],2);
auto wetdep_resolved_p = sd_arg_float(args[65],3);
smoke_to_mpas_fields(*state,arena,add_fire_heat_flux,add_fire_moist_flux,aero_emis_for_enhmix,aod3d,aod3d_p,aod3d_smoke,aod3d_smoke_p,aod550,aod550_p,chem,chem_p,coef_bb_dc,coef_bb_dc_p,config_smoke_scheme,ddvel,ddvel_p,drydep_flux,drydep_flux_p,drydep_opt,e_ant_out,e_ant_out_p,e_bb_out,e_bb_out_p,e_dust_out,e_dust_out_p,ebb_dcycle,fire_ef_out,fire_ef_out_p,frac_grid_burned,frac_grid_burned_p,fre_out,fre_out_p,frp_out,frp_out_p,hfx_bb,hfx_bb_p,hwp,hwp_p,ite,its,jte,jts,kte,kts,max_bb_plume,max_bb_plume_p,min_bb_plume,min_bb_plume_p,num_chem,num_e_ant_out,num_e_bb_out,num_e_dust_out,plumerise_opt,pm10,pm10_p,pm2_5,pm2_5_p,qfx_bb,qfx_bb_p,tend_chem_settle,tend_chem_settle_p,vis,vis_p,wetdep_ls_opt,wetdep_resolved,wetdep_resolved_p);
}
extern "C" SDKERNEL void sd_run_smoke_driver_diagnostics(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
auto aod3d_p = sd_arg_float(args[0],3);
auto aod550_p = sd_arg_float(args[1],2);
auto chem_p = sd_arg_float(args[2],4);
int chemistry_start = args[3].ivalue;
auto dz_p = sd_arg_float(args[4],3);
int index_dust_coarse = args[5].ivalue;
int index_dust_fine = args[6].ivalue;
int index_smoke_fine = args[7].ivalue;
int index_unspc_fine = args[8].ivalue;
int ite = args[9].ivalue;
int its = args[10].ivalue;
int jte = args[11].ivalue;
int jts = args[12].ivalue;
int kte = args[13].ivalue;
int kts = args[14].ivalue;
auto pm10_p = sd_arg_float(args[15],3);
auto pm2_5_p = sd_arg_float(args[16],3);
auto rho_p = sd_arg_float(args[17],3);
smoke_driver_diagnostics(*state,arena,aod3d_p,aod550_p,chem_p,chemistry_start,dz_p,index_dust_coarse,index_dust_fine,index_smoke_fine,index_unspc_fine,ite,its,jte,jts,kte,kts,pm10_p,pm2_5_p,rho_p);
}
extern "C" SDKERNEL void sd_run_mpas_smoke_driver(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
int num_chem = args[0].ivalue;
int chemistry_start = args[1].ivalue;
auto chem = sd_arg_float(args[2],4);
int kanthro = args[3].ivalue;
int kbio = args[4].ivalue;
int kfire = args[5].ivalue;
int kvol = args[6].ivalue;
int krwc = args[7].ivalue;
int index_smoke_fine = args[8].ivalue;
int index_smoke_coarse = args[9].ivalue;
int index_dust_fine = args[10].ivalue;
int index_dust_coarse = args[11].ivalue;
int index_unspc_fine = args[12].ivalue;
int index_ch4 = args[13].ivalue;
int index_e_bb_in_ch4 = args[14].ivalue;
int index_e_bb_out_ch4 = args[15].ivalue;
int index_e_bb_in_smoke_fine = args[16].ivalue;
int index_e_bb_in_smoke_coarse = args[17].ivalue;
int index_e_ant_in_unspc_fine = args[18].ivalue;
int index_e_ant_in_unspc_coarse = args[19].ivalue;
int index_e_bb_out_smoke_fine = args[20].ivalue;
int index_e_bb_out_smoke_coarse = args[21].ivalue;
int index_e_dust_out_dust_fine = args[22].ivalue;
int index_e_dust_out_dust_coarse = args[23].ivalue;
int index_e_ant_out_unspc_fine = args[24].ivalue;
int index_e_ant_out_unspc_coarse = args[25].ivalue;
auto frp_in = sd_arg_float(args[26],2);
auto frp_out = sd_arg_float(args[27],2);
auto fre_in = sd_arg_float(args[28],2);
auto fre_out = sd_arg_float(args[29],2);
auto hwp = sd_arg_float(args[30],2);
auto coef_bb_dc = sd_arg_float(args[31],2);
auto totprcp_prev24 = sd_arg_float(args[32],2);
auto hwp_avg = sd_arg_float(args[33],3);
auto frp_avg = sd_arg_float(args[34],3);
auto fre_avg = sd_arg_float(args[35],3);
auto fire_end_hr = sd_arg_float(args[36],2);
int nblocks = args[37].ivalue;
auto efs_map = sd_arg_float(args[38],2);
auto eco_id = sd_arg_int(args[39],2);
auto efs_smold = sd_arg_float(args[40],2);
auto efs_flam = sd_arg_float(args[41],2);
auto efs_rsmold = sd_arg_float(args[42],2);
auto fmc_avg = sd_arg_float(args[43],2);
auto hfx_bb = sd_arg_float(args[44],2);
auto qfx_bb = sd_arg_float(args[45],2);
auto frac_grid_burned = sd_arg_float(args[46],2);
auto min_bb_plume = sd_arg_int(args[47],2);
auto max_bb_plume = sd_arg_int(args[48],2);
auto sandfrac_in = sd_arg_float(args[49],2);
auto clayfrac_in = sd_arg_float(args[50],2);
auto uthres_in = sd_arg_float(args[51],2);
auto uthres_sg_in = sd_arg_float(args[52],2);
auto albedo_drag_in = sd_arg_float(args[53],2);
auto feff_in = sd_arg_float(args[54],2);
auto sep_in = sd_arg_float(args[55],2);
auto e_ant_in = sd_arg_float(args[56],4);
auto e_ant_out = sd_arg_float(args[57],4);
int num_e_ant_in = args[58].ivalue;
int num_e_ant_out = args[59].ivalue;
auto e_bb_in = sd_arg_float(args[60],4);
auto e_bb_out = sd_arg_float(args[61],4);
auto e_dust_out = sd_arg_float(args[62],4);
int num_e_bb_in = args[63].ivalue;
int num_e_bb_out = args[64].ivalue;
int num_e_dust_out = args[65].ivalue;
auto drydep_flux = sd_arg_float(args[66],3);
auto ddvel = sd_arg_float(args[67],3);
auto wetdep_resolved = sd_arg_float(args[68],3);
auto tend_chem_settle = sd_arg_float(args[69],4);
bool do_mpas_smoke = args[70].ivalue;
bool do_mpas_dust = args[71].ivalue;
bool do_mpas_anthro = args[72].ivalue;
bool do_mpas_rwc = args[73].ivalue;
bool calc_bb_emis_online = args[74].ivalue;
int bb_beta = args[75].ivalue;
int hwp_method = args[76].ivalue;
float hwp_alpha = args[77].fvalue;
int wetdep_ls_opt = args[78].ivalue;
float wetdep_ls_alpha = args[79].fvalue;
int plumerise_opt = args[80].ivalue;
int plume_wind_eff = args[81].ivalue;
float plume_alpha = args[82].fvalue;
float bb_emis_scale_factor = args[83].fvalue;
int ebb_dcycle = args[84].ivalue;
int drydep_opt = args[85].ivalue;
int pm_settling = args[86].ivalue;
bool add_fire_heat_flux = args[87].ivalue;
bool add_fire_moist_flux = args[88].ivalue;
int plumerisefire_frq = args[89].ivalue;
float bb_qv_scale_factor = args[90].fvalue;
float dust_alpha = args[91].fvalue;
float dust_gamma = args[92].fvalue;
float dust_drylimit_factor = args[93].fvalue;
float dust_moist_correction = args[94].fvalue;
int bb_input_prevh = args[95].ivalue;
float rwc_emis_scale_factor = args[96].fvalue;
auto rwc_denominator = sd_arg_float(args[97],2);
auto rwc_annual_sum = sd_arg_float(args[98],3);
auto rwc_annual_sum_smoke_fine = sd_arg_float(args[99],3);
auto rwc_annual_sum_smoke_coarse = sd_arg_float(args[100],3);
auto rwc_annual_sum_unspc_fine = sd_arg_float(args[101],3);
auto rwc_annual_sum_unspc_coarse = sd_arg_float(args[102],3);
auto vis = sd_arg_float(args[103],2);
auto qc_vis = sd_arg_float(args[104],3);
auto qr_vis = sd_arg_float(args[105],3);
auto qi_vis = sd_arg_float(args[106],3);
auto qs_vis = sd_arg_float(args[107],3);
auto qg_vis = sd_arg_float(args[108],3);
auto blcldw_vis = sd_arg_float(args[109],3);
auto blcldi_vis = sd_arg_float(args[110],3);
auto coszen = sd_arg_float(args[111],2);
auto aod3d_smoke = sd_arg_float(args[112],3);
auto aod3d = sd_arg_float(args[113],3);
int ktau = args[114].ivalue;
float dt = args[115].fvalue;
auto dxcell = sd_arg_float(args[116],2);
auto area = sd_arg_float(args[117],2);
auto xland = sd_arg_float(args[118],2);
auto u10 = sd_arg_float(args[119],2);
auto v10 = sd_arg_float(args[120],2);
auto ust = sd_arg_float(args[121],2);
auto xlat = sd_arg_float(args[122],2);
auto xlong = sd_arg_float(args[123],2);
auto tskin = sd_arg_float(args[124],2);
auto pblh = sd_arg_float(args[125],2);
auto t2m = sd_arg_float(args[126],2);
auto p8w = sd_arg_float(args[127],3);
auto dz8w = sd_arg_float(args[128],3);
auto z_at_w = sd_arg_float(args[129],3);
auto p_phy = sd_arg_float(args[130],3);
auto t_phy = sd_arg_float(args[131],3);
auto u_phy = sd_arg_float(args[132],3);
auto v_phy = sd_arg_float(args[133],3);
auto qv = sd_arg_float(args[134],3);
auto vvel = sd_arg_float(args[135],3);
auto pi_phy = sd_arg_float(args[136],3);
auto rho_phy = sd_arg_float(args[137],3);
auto kpbl = sd_arg_int(args[138],2);
int nsoil = args[139].ivalue;
auto smois = sd_arg_float(args[140],3);
auto tslb = sd_arg_float(args[141],3);
auto ivgtyp = sd_arg_int(args[142],2);
auto isltyp = sd_arg_int(args[143],2);
int nlcat = args[144].ivalue;
auto swdown = sd_arg_float(args[145],2);
auto z0 = sd_arg_float(args[146],2);
auto snowh = sd_arg_float(args[147],2);
float julian = args[148].fvalue;
auto rmol = sd_arg_float(args[149],2);
auto raincv = sd_arg_float(args[150],2);
auto rainncv = sd_arg_float(args[151],2);
auto dpt2m = sd_arg_float(args[152],2);
auto znt = sd_arg_float(args[153],2);
auto mavail = sd_arg_float(args[154],2);
float g = args[155].fvalue;
auto vegfra = sd_arg_float(args[156],2);
auto landusef = sd_arg_float(args[157],3);
auto cldfrac = sd_arg_float(args[158],3);
auto ktop_deep = sd_arg_int(args[159],2);
float cp = args[160].fvalue;
float rd = args[161].fvalue;
float gmt = args[162].fvalue;
int ids = args[163].ivalue;
int ide = args[164].ivalue;
int jds = args[165].ivalue;
int jde = args[166].ivalue;
int kds = args[167].ivalue;
int kde = args[168].ivalue;
int ims = args[169].ivalue;
int ime = args[170].ivalue;
int jms = args[171].ivalue;
int jme = args[172].ivalue;
int kms = args[173].ivalue;
int kme = args[174].ivalue;
int its = args[175].ivalue;
int ite = args[176].ivalue;
int jts = args[177].ivalue;
int jte = args[178].ivalue;
int kts = args[179].ivalue;
int kte = args[180].ivalue;
mpas_smoke_driver(*state,arena,num_chem,chemistry_start,chem,kanthro,kbio,kfire,kvol,krwc,index_smoke_fine,index_smoke_coarse,index_dust_fine,index_dust_coarse,index_unspc_fine,index_ch4,index_e_bb_in_ch4,index_e_bb_out_ch4,index_e_bb_in_smoke_fine,index_e_bb_in_smoke_coarse,index_e_ant_in_unspc_fine,index_e_ant_in_unspc_coarse,index_e_bb_out_smoke_fine,index_e_bb_out_smoke_coarse,index_e_dust_out_dust_fine,index_e_dust_out_dust_coarse,index_e_ant_out_unspc_fine,index_e_ant_out_unspc_coarse,frp_in,frp_out,fre_in,fre_out,hwp,coef_bb_dc,totprcp_prev24,hwp_avg,frp_avg,fre_avg,fire_end_hr,nblocks,efs_map,eco_id,efs_smold,efs_flam,efs_rsmold,fmc_avg,hfx_bb,qfx_bb,frac_grid_burned,min_bb_plume,max_bb_plume,sandfrac_in,clayfrac_in,uthres_in,uthres_sg_in,albedo_drag_in,feff_in,sep_in,e_ant_in,e_ant_out,num_e_ant_in,num_e_ant_out,e_bb_in,e_bb_out,e_dust_out,num_e_bb_in,num_e_bb_out,num_e_dust_out,drydep_flux,ddvel,wetdep_resolved,tend_chem_settle,do_mpas_smoke,do_mpas_dust,do_mpas_anthro,do_mpas_rwc,calc_bb_emis_online,bb_beta,hwp_method,hwp_alpha,wetdep_ls_opt,wetdep_ls_alpha,plumerise_opt,plume_wind_eff,plume_alpha,bb_emis_scale_factor,ebb_dcycle,drydep_opt,pm_settling,add_fire_heat_flux,add_fire_moist_flux,plumerisefire_frq,bb_qv_scale_factor,dust_alpha,dust_gamma,dust_drylimit_factor,dust_moist_correction,bb_input_prevh,rwc_emis_scale_factor,rwc_denominator,rwc_annual_sum,rwc_annual_sum_smoke_fine,rwc_annual_sum_smoke_coarse,rwc_annual_sum_unspc_fine,rwc_annual_sum_unspc_coarse,vis,qc_vis,qr_vis,qi_vis,qs_vis,qg_vis,blcldw_vis,blcldi_vis,coszen,aod3d_smoke,aod3d,ktau,dt,dxcell,area,xland,u10,v10,ust,xlat,xlong,tskin,pblh,t2m,p8w,dz8w,z_at_w,p_phy,t_phy,u_phy,v_phy,qv,vvel,pi_phy,rho_phy,kpbl,nsoil,smois,tslb,ivgtyp,isltyp,nlcat,swdown,z0,snowh,julian,rmol,raincv,rainncv,dpt2m,znt,mavail,g,vegfra,landusef,cldfrac,ktop_deep,cp,rd,gmt,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,bool(args[9].present),bool(args[13].present),bool(args[14].present),bool(args[15].present),bool(args[17].present),bool(args[21].present),bool(args[26].present),bool(args[27].present),bool(args[28].present),bool(args[29].present),bool(args[30].present),bool(args[31].present),bool(args[32].present),bool(args[33].present),bool(args[34].present),bool(args[35].present),bool(args[36].present),bool(args[38].present),bool(args[39].present),bool(args[40].present),bool(args[41].present),bool(args[42].present),bool(args[43].present),bool(args[44].present),bool(args[45].present),bool(args[46].present),bool(args[47].present),bool(args[48].present),bool(args[49].present),bool(args[50].present),bool(args[51].present),bool(args[52].present),bool(args[53].present),bool(args[54].present),bool(args[55].present),bool(args[56].present),bool(args[57].present),bool(args[60].present),bool(args[61].present),bool(args[62].present),bool(args[66].present),bool(args[67].present),bool(args[68].present),bool(args[69].present),bool(args[97].present),bool(args[98].present),bool(args[99].present),bool(args[100].present),bool(args[101].present),bool(args[102].present),bool(args[103].present),bool(args[104].present),bool(args[105].present),bool(args[106].present),bool(args[107].present),bool(args[108].present),bool(args[109].present),bool(args[110].present),bool(args[159].present));
}
extern "C" SDKERNEL void sd_run_mpas_smoke_prep(State *state,float *scratch,int capacity,Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};
bool do_mpas_smoke = args[0].ivalue;
int ktau = args[1].ivalue;
int nlcat = args[2].ivalue;
float cp = args[3].fvalue;
int ebb_dcycle = args[4].ivalue;
float ebb_min = args[5].fvalue;
auto xland = sd_arg_float(args[6],2);
auto xlat = sd_arg_float(args[7],2);
auto xlong = sd_arg_float(args[8],2);
auto ivgtyp = sd_arg_int(args[9],2);
auto isltyp = sd_arg_int(args[10],2);
auto vegfrac = sd_arg_float(args[11],3);
auto snowh = sd_arg_float(args[12],2);
auto u10 = sd_arg_float(args[13],2);
auto v10 = sd_arg_float(args[14],2);
auto wind10m = sd_arg_float(args[15],2);
auto t2m = sd_arg_float(args[16],2);
auto dpt2m = sd_arg_float(args[17],2);
auto wetness = sd_arg_float(args[18],2);
auto hwp = sd_arg_float(args[19],2);
auto hwp_day_avg = sd_arg_float(args[20],2);
int index_e_bb_in_smoke_fine = args[21].ivalue;
int num_e_bb_in = args[22].ivalue;
int kfire = args[23].ivalue;
auto e_bb_in = sd_arg_float(args[24],4);
auto t_phy = sd_arg_float(args[25],3);
auto u_phy = sd_arg_float(args[26],3);
auto v_phy = sd_arg_float(args[27],3);
auto p_phy = sd_arg_float(args[28],3);
auto pi_phy = sd_arg_float(args[29],3);
auto z_at_w = sd_arg_float(args[30],3);
auto dz8w = sd_arg_float(args[31],3);
auto dz8w_flip = sd_arg_float(args[32],3);
auto rho_phy = sd_arg_float(args[33],3);
auto qv = sd_arg_float(args[34],3);
auto relhum = sd_arg_float(args[35],3);
auto rh2m = sd_arg_float(args[36],2);
auto rri = sd_arg_float(args[37],3);
auto total_flashrate = sd_arg_float(args[38],2);
auto wind_phy = sd_arg_float(args[39],3);
auto theta_phy = sd_arg_float(args[40],3);
auto zmid = sd_arg_float(args[41],3);
auto kpbl_thetav = sd_arg_int(args[42],2);
auto peak_hr = sd_arg_float(args[43],2);
auto coef_bb_dc = sd_arg_float(args[44],2);
auto fire_hist = sd_arg_float(args[45],2);
auto lu_nofire = sd_arg_float(args[46],2);
auto lu_qfire = sd_arg_float(args[47],2);
auto lu_sfire = sd_arg_float(args[48],2);
auto fire_type = sd_arg_int(args[49],2);
int ids = args[50].ivalue;
int ide = args[51].ivalue;
int jds = args[52].ivalue;
int jde = args[53].ivalue;
int kds = args[54].ivalue;
int kde = args[55].ivalue;
int ims = args[56].ivalue;
int ime = args[57].ivalue;
int jms = args[58].ivalue;
int jme = args[59].ivalue;
int kms = args[60].ivalue;
int kme = args[61].ivalue;
int its = args[62].ivalue;
int ite = args[63].ivalue;
int jts = args[64].ivalue;
int jte = args[65].ivalue;
int kts = args[66].ivalue;
int kte = args[67].ivalue;
mpas_smoke_prep(*state,arena,do_mpas_smoke,ktau,nlcat,cp,ebb_dcycle,ebb_min,xland,xlat,xlong,ivgtyp,isltyp,vegfrac,snowh,u10,v10,wind10m,t2m,dpt2m,wetness,hwp,hwp_day_avg,index_e_bb_in_smoke_fine,num_e_bb_in,kfire,e_bb_in,t_phy,u_phy,v_phy,p_phy,pi_phy,z_at_w,dz8w,dz8w_flip,rho_phy,qv,relhum,rh2m,rri,total_flashrate,wind_phy,theta_phy,zmid,kpbl_thetav,peak_hr,coef_bb_dc,fire_hist,lu_nofire,lu_qfire,lu_sfire,fire_type,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,bool(args[19].present),bool(args[24].present),bool(args[44].present));
}
'''

_COMPILED_MODULES = {}

# Native storage census: original SAVE/THREADPRIVATE state, not a checkpoint claim.
NATIVE_STORAGE = json.loads('{"state.ssalt_mod.chem_opt_gocart":{"expression":"state->ssalt_mod.chem_opt_gocart","type":"int","rank":0},"state.ssalt_mod.chem_opt":{"expression":"state->ssalt_mod.chem_opt","type":"int","rank":0},"state.ssalt_mod.emission_scheme":{"expression":"state->ssalt_mod.emission_scheme","type":"int","rank":0},"state.mpas_smoke_config.epsilc":{"expression":"state->mpas_smoke_config.epsilc","type":"float","rank":0},"state.mpas_smoke_config.pi":{"expression":"state->mpas_smoke_config.pi","type":"float","rank":0},"state.mpas_smoke_config.seas_opt":{"expression":"state->mpas_smoke_config.seas_opt","type":"int","rank":0},"state.mpas_smoke_config.addsmoke_flag":{"expression":"state->mpas_smoke_config.addsmoke_flag","type":"int","rank":0},"state.mpas_smoke_config.n_dbg_lines":{"expression":"state->mpas_smoke_config.n_dbg_lines","type":"int","rank":0},"state.mpas_smoke_config.nfire_types":{"expression":"state->mpas_smoke_config.nfire_types","type":"int","rank":0},"state.mpas_smoke_config.dbg_opt":{"expression":"state->mpas_smoke_config.dbg_opt","type":"bool","rank":0},"state.mpas_smoke_config.aero_ind_fdb":{"expression":"state->mpas_smoke_config.aero_ind_fdb","type":"bool","rank":0},"state.mpas_smoke_config.chem_opt_gocart":{"expression":"state->mpas_smoke_config.chem_opt_gocart","type":"int","rank":0},"state.mpas_smoke_config.num_moist":{"expression":"state->mpas_smoke_config.num_moist","type":"int","rank":0},"state.mpas_smoke_config.num_emis_seas":{"expression":"state->mpas_smoke_config.num_emis_seas","type":"int","rank":0},"state.mpas_smoke_config.num_emis_dust":{"expression":"state->mpas_smoke_config.num_emis_dust","type":"int","rank":0},"state.mpas_smoke_config.p_dust_1":{"expression":"state->mpas_smoke_config.p_dust_1","type":"int","rank":0},"state.mpas_smoke_config.p_dust_2":{"expression":"state->mpas_smoke_config.p_dust_2","type":"int","rank":0},"state.mpas_smoke_config.p_dust_3":{"expression":"state->mpas_smoke_config.p_dust_3","type":"int","rank":0},"state.mpas_smoke_config.p_dust_4":{"expression":"state->mpas_smoke_config.p_dust_4","type":"int","rank":0},"state.mpas_smoke_config.p_dust_5":{"expression":"state->mpas_smoke_config.p_dust_5","type":"int","rank":0},"state.mpas_smoke_config.p_edust1":{"expression":"state->mpas_smoke_config.p_edust1","type":"int","rank":0},"state.mpas_smoke_config.p_edust2":{"expression":"state->mpas_smoke_config.p_edust2","type":"int","rank":0},"state.mpas_smoke_config.p_edust3":{"expression":"state->mpas_smoke_config.p_edust3","type":"int","rank":0},"state.mpas_smoke_config.p_edust4":{"expression":"state->mpas_smoke_config.p_edust4","type":"int","rank":0},"state.mpas_smoke_config.p_edust5":{"expression":"state->mpas_smoke_config.p_edust5","type":"int","rank":0},"state.mpas_smoke_config.ndvel":{"expression":"state->mpas_smoke_config.ndvel","type":"int","rank":0},"state.mpas_smoke_config.numgas":{"expression":"state->mpas_smoke_config.numgas","type":"int","rank":0},"state.rad_data_mod.sc_eff":{"expression":"state->rad_data_mod.sc_eff","type":"float","rank":1},"state.rad_data_mod.ab_eff":{"expression":"state->rad_data_mod.ab_eff","type":"float","rank":1},"state.seas_data_mod.number_ss_bins":{"expression":"state->seas_data_mod.number_ss_bins","type":"int","rank":0},"state.seas_data_mod.ra":{"expression":"state->seas_data_mod.ra","type":"float","rank":1},"state.seas_data_mod.rb":{"expression":"state->seas_data_mod.rb","type":"float","rank":1},"state.seas_data_mod.emission_scale":{"expression":"state->seas_data_mod.emission_scale","type":"float","rank":1},"state.seas_data_mod.den_seas":{"expression":"state->seas_data_mod.den_seas","type":"float","rank":1},"state.seas_data_mod.reff_seas":{"expression":"state->seas_data_mod.reff_seas","type":"float","rank":1},"state.dust_data_mod.ndust":{"expression":"state->dust_data_mod.ndust","type":"int","rank":0},"state.dust_data_mod.ndcls":{"expression":"state->dust_data_mod.ndcls","type":"int","rank":0},"state.dust_data_mod.ndsrc":{"expression":"state->dust_data_mod.ndsrc","type":"int","rank":0},"state.dust_data_mod.maxstypes":{"expression":"state->dust_data_mod.maxstypes","type":"int","rank":0},"state.dust_data_mod.nsalt":{"expression":"state->dust_data_mod.nsalt","type":"int","rank":0},"state.dust_data_mod.dyn_visc":{"expression":"state->dust_data_mod.dyn_visc","type":"float","rank":0},"state.dust_data_mod.den_dust":{"expression":"state->dust_data_mod.den_dust","type":"float","rank":1},"state.dust_data_mod.reff_dust":{"expression":"state->dust_data_mod.reff_dust","type":"float","rank":1},"state.dust_data_mod.frac_s":{"expression":"state->dust_data_mod.frac_s","type":"float","rank":1},"state.dust_data_mod.lo_dust":{"expression":"state->dust_data_mod.lo_dust","type":"float","rank":1},"state.dust_data_mod.up_dust":{"expression":"state->dust_data_mod.up_dust","type":"float","rank":1},"state.dust_data_mod.fengsha_maxstypes":{"expression":"state->dust_data_mod.fengsha_maxstypes","type":"int","rank":0},"state.dust_data_mod.dust_uthres":{"expression":"state->dust_data_mod.dust_uthres","type":"float","rank":1},"state.dust_data_mod.dust_calcdrag":{"expression":"state->dust_data_mod.dust_calcdrag","type":"int","rank":0},"state.dust_data_mod.dust_moist_opt":{"expression":"state->dust_data_mod.dust_moist_opt","type":"int","rank":0},"state.dust_data_mod.spoint":{"expression":"state->dust_data_mod.spoint","type":"int","rank":1},"state.dust_data_mod.reff_salt":{"expression":"state->dust_data_mod.reff_salt","type":"float","rank":1},"state.dust_data_mod.den_salt":{"expression":"state->dust_data_mod.den_salt","type":"float","rank":1},"state.dust_data_mod.frac_salt":{"expression":"state->dust_data_mod.frac_salt","type":"float","rank":1},"state.dust_data_mod.max_soiltyp":{"expression":"state->dust_data_mod.max_soiltyp","type":"int","rank":0},"state.dust_data_mod.maxsmc":{"expression":"state->dust_data_mod.maxsmc","type":"float","rank":1},"state.dep_data_mod.max_dep_vel":{"expression":"state->dep_data_mod.max_dep_vel","type":"float","rank":0},"state.dep_data_mod.dep_ref_hgt":{"expression":"state->dep_data_mod.dep_ref_hgt","type":"float","rank":0},"state.dep_data_mod.pi":{"expression":"state->dep_data_mod.pi","type":"float","rank":0},"state.dep_data_mod.threepi":{"expression":"state->dep_data_mod.threepi","type":"float","rank":0},"state.dep_data_mod.gravity":{"expression":"state->dep_data_mod.gravity","type":"float","rank":0},"state.dep_data_mod.grav":{"expression":"state->dep_data_mod.grav","type":"float","rank":0},"state.dep_data_mod.boltzmann":{"expression":"state->dep_data_mod.boltzmann","type":"float","rank":0},"state.dep_data_mod.rgasuniv":{"expression":"state->dep_data_mod.rgasuniv","type":"float","rank":0},"state.dep_data_mod.avo":{"expression":"state->dep_data_mod.avo","type":"float","rank":0},"state.dep_data_mod.cb":{"expression":"state->dep_data_mod.cb","type":"float","rank":0},"state.dep_data_mod.cim":{"expression":"state->dep_data_mod.cim","type":"float","rank":0},"state.dep_data_mod.alpha":{"expression":"state->dep_data_mod.alpha","type":"float","rank":0},"state.dep_data_mod.cin":{"expression":"state->dep_data_mod.cin","type":"float","rank":0},"state.dep_data_mod.vv":{"expression":"state->dep_data_mod.vv","type":"float","rank":0},"state.dep_data_mod.a_for":{"expression":"state->dep_data_mod.a_for","type":"float","rank":0},"state.dep_data_mod.a_grs":{"expression":"state->dep_data_mod.a_grs","type":"float","rank":0},"state.dep_data_mod.a_wat":{"expression":"state->dep_data_mod.a_wat","type":"float","rank":0},"state.dep_data_mod.eps0_for":{"expression":"state->dep_data_mod.eps0_for","type":"float","rank":0},"state.dep_data_mod.eps0_grs":{"expression":"state->dep_data_mod.eps0_grs","type":"float","rank":0},"state.dep_data_mod.eps0_wat":{"expression":"state->dep_data_mod.eps0_wat","type":"float","rank":0},"state.dep_data_mod.one3":{"expression":"state->dep_data_mod.one3","type":"float","rank":0},"state.dep_data_mod.two3":{"expression":"state->dep_data_mod.two3","type":"float","rank":0},"state.dep_data_mod.sqrt2":{"expression":"state->dep_data_mod.sqrt2","type":"float","rank":0},"state.dep_data_mod.sqrtpi":{"expression":"state->dep_data_mod.sqrtpi","type":"float","rank":0},"state.dep_data_mod.karman":{"expression":"state->dep_data_mod.karman","type":"float","rank":0},"state.dep_data_mod.conmin":{"expression":"state->dep_data_mod.conmin","type":"float","rank":0},"state.dep_data_mod.pirs":{"expression":"state->dep_data_mod.pirs","type":"float","rank":0},"state.dep_data_mod.f6dpi":{"expression":"state->dep_data_mod.f6dpi","type":"float","rank":0},"state.dep_data_mod.f6dpim9":{"expression":"state->dep_data_mod.f6dpim9","type":"float","rank":0},"state.dep_data_mod.sigma1":{"expression":"state->dep_data_mod.sigma1","type":"float","rank":0},"state.dep_data_mod.mean_diameter1":{"expression":"state->dep_data_mod.mean_diameter1","type":"float","rank":0},"state.dep_data_mod.fact_wfa":{"expression":"state->dep_data_mod.fact_wfa","type":"float","rank":0},"state.dep_data_mod.dgmin":{"expression":"state->dep_data_mod.dgmin","type":"float","rank":0},"state.dep_data_mod.densmin":{"expression":"state->dep_data_mod.densmin","type":"float","rank":0},"state.dep_data_mod.aero_dens":{"expression":"state->dep_data_mod.aero_dens","type":"float","rank":1},"state.dep_data_mod.aero_diam":{"expression":"state->dep_data_mod.aero_diam","type":"float","rank":1},"state.mpas_smoke_init.p_smoke_fine":{"expression":"state->mpas_smoke_init.p_smoke_fine","type":"int","rank":0},"state.mpas_smoke_init.p_smoke_coarse":{"expression":"state->mpas_smoke_init.p_smoke_coarse","type":"int","rank":0},"state.mpas_smoke_init.p_dust_fine":{"expression":"state->mpas_smoke_init.p_dust_fine","type":"int","rank":0},"state.mpas_smoke_init.p_dust_coarse":{"expression":"state->mpas_smoke_init.p_dust_coarse","type":"int","rank":0},"state.mpas_smoke_init.p_polp_all":{"expression":"state->mpas_smoke_init.p_polp_all","type":"int","rank":0},"state.mpas_smoke_init.p_polp_tree":{"expression":"state->mpas_smoke_init.p_polp_tree","type":"int","rank":0},"state.mpas_smoke_init.p_polp_grass":{"expression":"state->mpas_smoke_init.p_polp_grass","type":"int","rank":0},"state.mpas_smoke_init.p_polp_weed":{"expression":"state->mpas_smoke_init.p_polp_weed","type":"int","rank":0},"state.mpas_smoke_init.p_pols_tree":{"expression":"state->mpas_smoke_init.p_pols_tree","type":"int","rank":0},"state.mpas_smoke_init.p_pols_grass":{"expression":"state->mpas_smoke_init.p_pols_grass","type":"int","rank":0},"state.mpas_smoke_init.p_pols_weed":{"expression":"state->mpas_smoke_init.p_pols_weed","type":"int","rank":0},"state.mpas_smoke_init.p_pols_all":{"expression":"state->mpas_smoke_init.p_pols_all","type":"int","rank":0},"state.mpas_smoke_init.p_unspc_fine":{"expression":"state->mpas_smoke_init.p_unspc_fine","type":"int","rank":0},"state.mpas_smoke_init.p_unspc_coarse":{"expression":"state->mpas_smoke_init.p_unspc_coarse","type":"int","rank":0},"state.mpas_smoke_init.p_ssalt_fine":{"expression":"state->mpas_smoke_init.p_ssalt_fine","type":"int","rank":0},"state.mpas_smoke_init.p_ssalt_coarse":{"expression":"state->mpas_smoke_init.p_ssalt_coarse","type":"int","rank":0},"state.mpas_smoke_init.p_no3_a_fine":{"expression":"state->mpas_smoke_init.p_no3_a_fine","type":"int","rank":0},"state.mpas_smoke_init.p_so4_a_fine":{"expression":"state->mpas_smoke_init.p_so4_a_fine","type":"int","rank":0},"state.mpas_smoke_init.p_nh4_a_fine":{"expression":"state->mpas_smoke_init.p_nh4_a_fine","type":"int","rank":0},"state.mpas_smoke_init.p_so2":{"expression":"state->mpas_smoke_init.p_so2","type":"int","rank":0},"state.mpas_smoke_init.p_nh3":{"expression":"state->mpas_smoke_init.p_nh3","type":"int","rank":0},"state.mpas_smoke_init.p_ch4":{"expression":"state->mpas_smoke_init.p_ch4","type":"int","rank":0},"state.mpas_smoke_init.p_mold_fine":{"expression":"state->mpas_smoke_init.p_mold_fine","type":"int","rank":0},"state.mpas_smoke_init.p_bact_fine":{"expression":"state->mpas_smoke_init.p_bact_fine","type":"int","rank":0},"state.mpas_smoke_init.p_plastic_fine":{"expression":"state->mpas_smoke_init.p_plastic_fine","type":"int","rank":0},"state.mpas_smoke_init.p_dms":{"expression":"state->mpas_smoke_init.p_dms","type":"int","rank":0},"state.seas_ngac_mod.r80fac":{"expression":"state->seas_ngac_mod.r80fac","type":"float","rank":0},"state.seas_ngac_mod.rhop":{"expression":"state->seas_ngac_mod.rhop","type":"float","rank":0},"state.module_zero_plumegen_coms.nkp":{"expression":"state->module_zero_plumegen_coms.nkp","type":"int","rank":0},"state.module_zero_plumegen_coms.ntime":{"expression":"state->module_zero_plumegen_coms.ntime","type":"int","rank":0},"state.module_zero_plumegen_coms.private_thread_coms.present":{"expression":"state->module_zero_plumegen_coms.private_thread_coms","type":"pointer_present","rank":0},"state.module_zero_plumegen_coms.private_thread_coms.alias":{"expression":"state->module_zero_plumegen_coms.private_thread_coms","type":"pointer_alias","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%initialized":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->initialized","type":"bool","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%w":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->w","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%t":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->t","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qv":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qv","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qc":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qc","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qh":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qh","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qi":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qi","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%sc":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->sc","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vth":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vth","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vti":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vti","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%rho":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->rho","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%txs":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->txs","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%est":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->est","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qsat":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qsat","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%wc":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->wc","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%wt":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->wt","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%tt":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->tt","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qvt":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qvt","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qct":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qct","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qht":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qht","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qit":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qit","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%sct":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->sct","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%dzm":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->dzm","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%dzt":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->dzt","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%zm":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->zm","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%zt":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->zt","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vctr1":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vctr1","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vctr2":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vctr2","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vt3dc":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vt3dc","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vt3df":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vt3df","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vt3dk":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vt3dk","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vt3dg":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vt3dg","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%scr1":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->scr1","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%pke":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->pke","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%the":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->the","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%thve":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->thve","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%thee":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->thee","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%pe":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->pe","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%te":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->te","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%qvenv":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->qvenv","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%dne":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->dne","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%ucon":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->ucon","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vcon":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vcon","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%thtcon":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->thtcon","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%rvcon":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->rvcon","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%picon":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->picon","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%tmpcon":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->tmpcon","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%zcon":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->zcon","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%zzcon":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->zzcon","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%dz":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->dz","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%dqsdz":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->dqsdz","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%visc":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->visc","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%viscosity":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->viscosity","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%tstpf":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->tstpf","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%n":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->n","type":"int","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%nm1":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->nm1","type":"int","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%l":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->l","type":"int","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%cvh":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->cvh","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%cvi":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->cvi","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%adiabat":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->adiabat","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%wbar":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->wbar","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%vhrel":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vhrel","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%virel":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->virel","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%zsurf":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->zsurf","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%ztop":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->ztop","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%area":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->area","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%rsurf":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->rsurf","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%alpha":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->alpha","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%radius":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->radius","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%heating":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->heating","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%fmoist":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->fmoist","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%bload":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->bload","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%dt":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->dt","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%time":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->time","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%tdur":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->tdur","type":"float","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%mintime":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->mintime","type":"int","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%mdur":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->mdur","type":"int","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%maxtime":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->maxtime","type":"int","rank":0},"state.module_zero_plumegen_coms.private_thread_coms%upe":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->upe","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vpe":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vpe","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vel_e":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vel_e","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vel_p":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vel_p","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%rad_p":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->rad_p","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%vel_t":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->vel_t","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%rad_t":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->rad_t","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%ztop_":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->ztop_","type":"float","rank":1},"state.module_zero_plumegen_coms.private_thread_coms%testval":{"expression":"state->module_zero_plumegen_coms.private_thread_coms->testval","type":"int","rank":0},"local.module_fire_emissions.diurnal_cycle.icall":{"expression":"state->saved_diurnal_cycle_icall","type":"int","rank":0},"local.module_add_emiss_burn.add_emis_burn.icall":{"expression":"state->saved_add_emis_burn_icall","type":"int","rank":0},"local.module_smoke_plumerise.makeplume.delz_thresold":{"expression":"state->saved_makeplume_delz_thresold","type":"float","rank":0},"local.module_smoke_plumerise.visc_w.printed":{"expression":"state->saved_visc_w_printed","type":"bool","rank":0},"local.dep_dry_mod_emerson.dry_dep_driver_emerson.icall":{"expression":"state->saved_dry_dep_driver_emerson_icall","type":"int","rank":0},"state.mpas_log.mpas_log_info%outputlog%isactive":{"expression":"state->framework_log_active","type":"bool","rank":0},"state.mpas_log.mpas_log_info%outputmessagecount":{"expression":"state->framework_log_output_count","type":"int","rank":0}}')
_STORAGE_SLOTS = {name: slot for slot, name in enumerate(NATIVE_STORAGE)}

_ARGUMENT_DTYPE = np.dtype([
    ('data', np.uint64), ('lo', np.int32, (4,)), ('hi', np.int32, (4,)),
    ('ivalue', np.int32), ('fvalue', np.float32), ('present', np.int32),
    ('st', np.int32, (4,)),
], align=True)

# The fork driver's pool pointers (mpas_atmphys_driver_smoke.F:290-496,777-1020): assumed-shape dummies that the
# boundaries only index element by element (no reshape, section or sequence association), so they may be the
# provider's own words in any layout, passed with their strides.
_POOL_VIEW_ROUTINES = frozenset(('smoke_from_mpas_fields', 'smoke_to_mpas_fields'))

def shared_math_source(kernel_directory: Path) -> str:
    """Read the pinned engine's common device-math implementations.

    The caller must add the shared binary32 erf entrypoint as
    ``smoke_shared_erf(float)``. FENGSHA reaches it at
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/dust_fengsha_mod.F90:355-356.
    No replacement erf implementation is supplied by this scheme.
    """
    names = ('glibc_flt32.cuh', 'flt32_expf_fma.cuh', 'glibc_flt64.cuh',
             'glibc_trig_flt32.cuh', 'thompson_aerosol_libm.cuh')
    return '\n'.join((kernel_directory / name).read_text() for name in names)


def native_storage_source():
    """Copy complete native storage descriptors without numerical processing.

    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:648-814 @1d6e370e.
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90:14-187 @1d6e370e.
    Association records precede aliases and members. An absent parent, changed
    allocation bounds or unknown field reports an error instead of inventing
    values. Array descriptors retain device allocations owned by SmokeState.
    """
    lines = ['extern "C" SDKERNEL void sd_capture_storage(State *state, int slot, int action, Argument *args, int *status) {SDGUARD; *status=0; switch(slot) {']
    for slot, field in enumerate(NATIVE_STORAGE.values()):
        target, kind, rank = field['expression'], field['type'], field['rank']
        lines.append(f'case {slot}: {{')
        # Transport defect: the module-to-member separator is '.', so the
        # old '->private_thread_coms->' search missed every private member
        # and dereferenced an explicitly absent native association.
        if '.private_thread_coms->' in target:
            parent = target.rsplit('->', 1)[0]
            lines.append(f'if (!{parent}) {{*status=-1; break;}}')
        if kind == 'pointer_present':
            lines.append(f'if(action==0) {target}=args->ivalue ? state->plume_store : nullptr; else args->ivalue=({target}!=nullptr);')
        elif kind == 'pointer_alias':
            lines.append(f'if(action==0) {{if(args->ivalue!=({target} ? 1 : 0)) *status=-2;}} else args->ivalue=({target} ? 1 : 0);')
        elif rank:
            lines.append(f'auto buffer=sd_arg_{kind}(*args,{rank});')
            lines.append(f'if(action==0) {target}=buffer; else {{')
            lines.append(f'if({target}.rank!={rank}) {{*status=-3; break;}}')
            lines.append(f'for(int d=0;d<{rank};++d) if({target}.lo[d]!=buffer.lo[d] || {target}.hi[d]!=buffer.hi[d]) {{*status=-3; break;}}')
            lines.append(f'if(*status==0) for(int i=0;i<buffer.size();++i) buffer.linear(i)={target}.linear(i); }}')
        else:
            member = 'fvalue' if kind == 'float' else 'ivalue'
            lines.append(f'if(action==0) {target}=args->{member}; else args->{member}={target};')
        lines.append('break;}')
    return '\n'.join(lines + ['default: *status=-4; } }'])

CUDA_OPTIONS = ('--std=c++17', '--fmad=false', '--ftz=false',
                '--prec-div=true', '--prec-sqrt=true')
#: The column graph's header paths under cupy-cuda12x 14.2, in the packs' portable form (measured on that wheel).
NO_CUPY_COLUMN_INCLUDES = ('-I@SITE@/cupy/_core/include/cupy/_cccl/cub', '-I@SITE@/cupy/_core/include/cupy/_cccl/thrust',
                           '-I@SITE@/cupy/_core/include/cupy/_cccl/libcudacxx', '-I@SITE@/cupy/_core/include',
                           '-I@SITE@/nvidia/cuda_runtime/include')


def cuda_options(*, columns=False):
    """Resolve standard CUDA headers from the installed CuPy environment."""
    if not columns:
        return CUDA_OPTIONS
    import importlib.util
    if importlib.util.find_spec('cupy') is None:
        # THE BREAKAGE THIS ENDS: on a host with no CuPy (the public CPU CI, the Linux CPU tier) the AOT
        # physics_inventory raised ModuleNotFoundError here, so every inventory test failed. Such a host
        # compiles nothing; it lists the cupy-cuda12x wheel layout in the kernel packs' portable form
        # (cuda_backend.aot.portable_options), which is what the cuda12 pack holds for these rows.
        return CUDA_OPTIONS + NO_CUPY_COLUMN_INCLUDES
    from pathlib import Path
    import cupy
    from cupy._core import core
    site = Path(cupy.__file__).resolve().parent.parent
    # Raw NVRTC does not add CuPy's header paths. Keep only paths supplied by
    # the installed environment, so a system toolkit cannot become required.
    includes = tuple(str(option) for option in core.assemble_cupy_compiler_options(())
                     if str(option).startswith('-I')
                     and Path(str(option)[2:]).resolve().is_relative_to(site))
    if not any((Path(option[2:]) / 'cuda/barrier').is_file() for option in includes):
        raise RuntimeError('RRFS smoke requires cuda/barrier from the installed CuPy headers')
    return CUDA_OPTIONS + includes


def cuda_source(math_source, storage_transport_source=None, *, columns=False):
    """Exact source shared by CPU compilation and the resident loader."""
    storage = native_storage_source() if storage_transport_source is None else storage_transport_source
    prefix = '#define SDKERNEL __global__\n#define SDGUARD if(blockIdx.x || threadIdx.x) return\n'
    graph = CUDA_SOURCE
    if columns:
        from .smoke_columns import source
        graph = source(graph, ROUTINES, _NATIVE_BOUNDARIES, automatic=COLUMN_AUTOMATIC_ARRAYS)
    return math_source + '\n' + prefix + graph + '\n' + storage


# Scratch words every column slot has beyond twice its share of the call's arguments.
COLUMN_SCRATCH_FLOOR_WORDS = 2_048


def column_persistent_words(source):
    """Words initialize() takes from a column slot's SAVE arena, read from the graph itself.

    Each allocation there is rank one with a literal extent or a module
    constant that initialize() itself assigns (sea-salt and dust tables, the
    radiation and deposition tables: 4,129 words at 9652bbf and 5d2c71e2),
    rounded up to 64 words. Breakage it prevents: a fixed 16,384-word arena
    (sized when the 61 plume vectors, 12,200 words, still lived there) held
    49 KiB of nothing per slot, ~2.5 GiB across a 3 km tile.
    """
    import re
    start = source.index('SDDEV void initialize(State &state, Arena &storage,Plume &plume) {')
    body = source[start:source.index('\n}\n', start)]
    constants = {name: int(value) for name, value in
                 re.findall(r'^(state\.\w+\.\w+) = (\d+);$', body, re.M)}
    words = 0
    for extent in re.findall(r'storage\.alloc<(?:float|int)>\(([^)]*)\)', body):
        lower, upper = (part.strip() for part in extent.split(','))
        if lower != '1' or not (upper.isdigit() or upper in constants):
            raise ValueError('Pinned SMOKE SAVE allocation changed: storage.alloc(' + extent + ')')
        words += int(upper) if upper.isdigit() else constants[upper]
    if 'storage' in re.sub(r'storage\.alloc<|,storage\)|Arena &storage', '', body):
        raise ValueError('Pinned SMOKE initialize() passes its SAVE arena on')
    return -(-words // 64) * 64


# Every kernel a column-mode state launches besides the wrapper phases (smoke_phases.PHASES); their
# widest static frame sizes the stack limit.
COLUMN_KERNELS = ('sd_sizes', 'sd_initialize', 'sd_initialize_columns', 'sd_export_wrapper_logs',
                  'sd_capture_storage', 'sd_run_smoke_from_mpas_fields', 'sd_run_smoke_to_mpas_fields',
                  'sd_run_smoke_driver_diagnostics')
# Full-grid automatic arrays (ims:ime first) the column lowering makes private to one column.
COLUMN_AUTOMATIC_ARRAYS = 28
# Columns per launch tile when no storage budget is set (see _column_tile).
COLUMN_TILE_COLUMNS = 65_536


def _column_tile(columns, slot_bytes):
    """Columns per launch tile: as many as keep the card busy, never more than the budget holds.

    One column is one thread of a phase kernel, so a tile of COLUMN_TILE_COLUMNS
    already fills every SM of a 188-SM card; a wider tile only holds more
    private storage (state, plume, SAVE arena and scratch per slot). Tiles are
    whole 128-column blocks of equal width, so the last tile is not a sliver.
    WOOF_HEX_SMOKE_TILE_MIB sets a storage budget instead (default: at most a
    quarter of free device memory, at most 16 GiB). Breakage it prevents: one
    private slot per column is ~90 GB at 3 km with the old scratch floor, and
    sizing tiles from free memory alone held 16 GiB for what ~4 GiB does as fast.
    """
    import os
    mib = os.environ.get('WOOF_HEX_SMOKE_TILE_MIB')
    if mib:
        budget = int(mib) << 20
    else:
        import cupy as cp
        budget = min(16 << 30, cp.cuda.runtime.memGetInfo()[0] // 4)
    tile = max(128, budget // max(1, slot_bytes) // 128 * 128)
    if not mib:
        tile = min(tile, COLUMN_TILE_COLUMNS)
    if tile >= columns:
        return columns
    tiles = -(-columns // tile)
    return (-(-columns // tiles) + 127) // 128 * 128


class SmokeState:
    """Native tile state; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e."""

    def __init__(self, *, math_source: str, workspace_words: int = 1_000_000,
                 persistent_words: int = 50_000, storage_transport_source: str | None = None,
                 columns: bool = False):
        """Initialize SAVE state; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:648-814 @1d6e370e."""
        import cupy as cp
        from cupy.cuda import function
        from woof.hex.cuda_backend.aot import raw_image
        # RawKernel injects --ftz=true. Native GNU preserves subnormals; use
        # the low-level compiler with an explicit --ftz=false instead.
        import hashlib
        source = cuda_source(math_source, storage_transport_source, columns=columns)
        self.columns = columns
        self._column_count = 1
        self._tile = 1
        self._column_persistent_words = column_persistent_words(source) if columns else 0
        self._persistent_words = persistent_words
        digest = hashlib.sha256(source.encode()).hexdigest()
        architecture = cp.cuda.Device().compute_capability
        key = (cp.cuda.runtime.getDevice(), architecture, digest, _hex_numerics.mode())
        if key not in _COMPILED_MODULES:
            image = raw_image(source, _hex_numerics.options(cuda_options(columns=columns)),
                              architecture, filename='rrfs_smoke.cu')
            module = function.Module()
            module.load(image)
            _COMPILED_MODULES[key] = module
        self._module = _COMPILED_MODULES[key]
        self.compile_metadata = dict(source_sha256=digest, ftz=not _hex_numerics.is_strict(), fmad=not _hex_numerics.is_strict(), numerics=_hex_numerics.mode(),
                                     architecture=architecture)
        self._cp = cp
        self._functions = {}
        self._storage_buffers = {}
        from collections import deque
        self._pending_logs = deque()
        self._wrapper_logs = cp.empty(1 + len(WRAPPER_LOG_MESSAGES), dtype=cp.int32)
        self._log_host = None
        self._log_event = None
        self._inflight = None
        # The whole native call graph passes many Fortran descriptors through
        # nested calls. CUDA's default per-thread stack is only 1 KiB.
        limit = cp.cuda.runtime.cudaLimitStackSize
        current_stack = cp.cuda.runtime.deviceGetLimit(limit)
        if columns:
            # Column launches inline the whole graph (SDDEV is __forceinline__ and the graph has
            # no recursion), so each entry's static frame is all the stack it uses. Reserve the
            # widest entry frame. The driver's local-memory store is that frame times every thread
            # the card can hold, so a fixed 16 KiB floor held ~4.4 GiB on a 188-SM card.
            from cupy.cuda import driver
            from .smoke_phases import PHASES
            wanted = max(int(driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                                     self._function(name).ptr))
                         for name in COLUMN_KERNELS + tuple('sd_run_smoke_phase_' + phase for phase in PHASES))
        else:
            wanted = 16_384
        if wanted > current_stack:
            cp.cuda.runtime.deviceSetLimit(limit, wanted)
        self.compile_metadata['stack_bytes'] = cp.cuda.runtime.deviceGetLimit(limit)
        self.compile_metadata['frame_bytes'] = wanted
        sizes = cp.empty(3, dtype=cp.int32)
        self._function('sd_sizes')((1,), (1,), (sizes,))
        state_size, plume_size, arg_size = map(int, sizes.get())
        self._state_size, self._plume_size = state_size, plume_size
        if arg_size != _ARGUMENT_DTYPE.itemsize:
            raise RuntimeError(f'SMOKE descriptor ABI differs: {arg_size}')
        self._state = cp.empty(state_size, dtype=cp.uint8)
        self._plume = cp.empty(plume_size, dtype=cp.uint8)
        self._persistent = cp.empty(persistent_words, dtype=cp.float32)
        self._workspace = cp.empty(workspace_words, dtype=cp.float32)
        self._function('sd_initialize')((1,), (1,), (
            self._state, self._plume, self._persistent, np.int32(persistent_words),
        ))

    def _function(self, name: str):
        """Resolve graph entry; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e."""
        if name not in self._functions:
            self._functions[name] = self._module.get_function(name)
        return self._functions[name]

    def call(self, routine: str, arguments: dict, *, lower_bounds: dict | None = None,
             forwarded_absent: frozenset[str] = frozenset()):
        """Launch one complete routine graph using its original driver arguments.

        ``None`` denotes a missing OPTIONAL argument. Scalars and array
        descriptors are packed on the host; weather arrays remain on device.
        The returned descriptor buffer carries scalar INOUT/OUT values.
        MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e.
        """
        spec = ROUTINES[routine]
        cp = self._cp
        lower_bounds = lower_bounds or {}
        # The wrapper's automatic arrays scale with the native tile, including
        # its full chemistry tensor (wrapper:210-240). Reserve from descriptor
        # sizes; the initial workspace is a floor, not a supported tile limit.
        required_words = 50_000 + 2 * sum(
            value.size for value in arguments.values() if isinstance(value, cp.ndarray)
        )
        columns = 1
        if self.columns:
            if routine not in _NATIVE_BOUNDARIES.values():
                raise ValueError('Column state accepts the four complete driver boundaries')
            columns = (int(arguments['ime']) - int(arguments['ims']) + 1
                       if 'ims' in arguments else arguments['chem_p'].shape[0])
            # A column's automatic arrays and assignment temporaries are its own slices of
            # these arguments' shapes (levels, species, bins), never the whole tile. Measured
            # at 3 km (206,726 columns, 59 levels, 348 fire cells, every call of 1 h): the
            # most any column held at once was 1,680 words against a 1,903-word share. Twice
            # the share plus a floor keeps that margin as levels and species grow; a short
            # arena still traps (sd_capacity_failure) rather than write past its slot.
            required_words = COLUMN_SCRATCH_FLOOR_WORDS + (required_words-50_000+columns-1)//columns
            if self._column_count != columns:
                if self._column_count != 1:
                    raise ValueError('SMOKE native tile shape changed during a run')
                self._column_count = columns
                # A column slot's SAVE arena holds exactly what initialize()
                # allocates (the 61 plume vectors live in the Plume struct); a
                # short arena traps at the first call.
                self._persistent_words = self._column_persistent_words
                self._tile = _column_tile(columns, self._state_size + self._plume_size
                                          + 4*(self._persistent_words + required_words))
                # One more state slot than the tile when tiled: slot 0 belongs
                # to the first column alone (its wrapper log events).
                slots = self._tile + (self._tile < columns)
                self._state = cp.empty(slots*self._state_size, dtype=cp.uint8)
                self._plume = cp.empty(slots*self._plume_size, dtype=cp.uint8)
                self._persistent = cp.empty(slots*self._persistent_words, dtype=cp.float32)
                self._function('sd_initialize_columns')(((slots+127)//128,), (128,), (
                    self._state, self._plume, self._persistent,
                    np.int32(self._persistent_words), np.int32(slots)))
                import logging
                logging.getLogger(__name__).info(
                    'smoke column tiles: %d columns, tile %d, %d slots; per slot state %d B, plume %d B, SAVE %d words, '
                    'scratch %d words', columns, self._tile, slots, self._state_size, self._plume_size,
                    self._persistent_words, required_words)
        slots = self._tile if self.columns else 1
        if self._workspace.size < required_words*slots:
            self._workspace = cp.empty(required_words*slots, dtype=cp.float32)
        if any(n not in spec['args'] or arguments.get(n) is not None for n in forwarded_absent):
            raise ValueError('forwarded absence must name an original absent actual')
        descriptor = np.zeros(len(spec['args']), dtype=_ARGUMENT_DTYPE)
        keep = []
        for index, name in enumerate(spec['args']):
            symbol = spec['symbols'][name]
            if name not in arguments and not symbol['optional']:
                raise ValueError(f'{routine} requires {name}')
            value = arguments.get(name)
            if value is None:
                if not symbol['optional'] and name not in forwarded_absent:
                    raise ValueError(f'{routine} requires {name}')
                continue
            descriptor['present'][index] = 1
            if symbol['dim']:
                dtype = cp.float32 if symbol['type'] == 'float' else cp.int32
                if not isinstance(value, cp.ndarray) or value.dtype != dtype:
                    raise TypeError(f'{name} must be a device {dtype} array')
                if value.ndim != len(symbol['dim']):
                    raise ValueError(f'{name} requires rank {len(symbol["dim"])} in Fortran order')
                if not value.flags.f_contiguous:
                    # A pool view: the provider's words with their own strides (no copy in or out).
                    if routine not in _POOL_VIEW_ROUTINES or any(d != ':' for d in symbol['dim']):
                        raise ValueError(f'{name} requires rank {len(symbol["dim"])} in Fortran order')
                    steps = [stride // value.itemsize for stride in value.strides]
                    if (any(stride % value.itemsize or stride <= 0 for stride in value.strides)
                            or sum((n - 1) * st for n, st in zip(value.shape, steps)) >= 2**31):
                        raise ValueError(f'{name} pool view strides are not positive element strides')
                    descriptor['st'][index, :value.ndim] = steps
                lo = lower_bounds.get(name, (1,) * value.ndim)
                if len(lo) != value.ndim:
                    raise ValueError(f'{name} lower bounds have the wrong rank')
                descriptor['data'][index] = value.data.ptr
                descriptor['lo'][index, :value.ndim] = lo
                descriptor['hi'][index, :value.ndim] = tuple(l + n - 1 for l, n in zip(lo, value.shape))
                keep.append(value)
            elif symbol['type'] == 'FStr':
                if not isinstance(value, cp.ndarray) or value.dtype != cp.uint8:
                    raise TypeError(f'{name} must be a device character buffer')
                descriptor['data'][index] = value.data.ptr
                keep.append(value)
            elif symbol['type'] == 'float':
                descriptor['fvalue'][index] = value
            else:
                descriptor['ivalue'][index] = value
        device_descriptor = cp.asarray(descriptor)
        if self.columns:
            # Column tiles in order; a tile runs every phase before the next
            # tile reuses its slots (the phases share a column's scratch).
            from .smoke_phases import PHASES
            names = (['sd_run_smoke_phase_' + phase for phase in PHASES]
                     if routine == 'mpas_smoke_driver' else ['sd_run_' + routine])
            capacity = np.int32(self._workspace.size//self._tile)
            for offset in range(0, columns, self._tile):
                count = min(self._tile, columns - offset)
                for name in names:
                    plume = name == 'sd_run_smoke_phase_plume'
                    self._function(name)((count,) if plume else ((count+127)//128,), (64,) if plume else (128,), (
                        self._state, self._workspace, capacity, device_descriptor, np.int32(offset),
                    ), shared_mem=2048 if plume else 0)
        else:
            self._function('sd_run_' + routine)((1,), (1,), (
                self._state, self._workspace, np.int32(self._workspace.size), device_descriptor,
            ))
        # Device work owns its arguments until it completes. Stream order
        # keeps them alive for the next call's work; the host does not wait
        # (no per-scheme host synchronization on the step path).
        self._inflight = (keep, device_descriptor)
        if routine == 'mpas_smoke_driver':
            self._collect_wrapper_logs()
        return device_descriptor

    def _collect_wrapper_logs(self):
        """Export original call order and literal payloads.

        MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:262-600 @1d6e370e.
        """
        # The previous call's events are read here, one call late: its copy
        # finished long ago, so reading it never stalls the step. This call's
        # events are exported and copied asynchronously behind its kernels.
        sites = self._drain_wrapper_logs()
        self._function('sd_export_wrapper_logs')((1,), (1,), (self._state, self._wrapper_logs))
        if self._log_host is None:
            import cupyx
            self._log_host = cupyx.empty_pinned(self._wrapper_logs.shape, dtype=np.int32)
            self._log_event = None
        self._wrapper_logs.get(stream=self._cp.cuda.get_current_stream(), out=self._log_host, blocking=False)
        self._log_event = self._cp.cuda.get_current_stream().record()
        return sites

    def _drain_wrapper_logs(self):
        """Deliver the exported events of the last call (waits only on that copy)."""
        if self._log_host is None or self._log_event is None:
            return ()
        self._log_event.synchronize()
        self._log_event = None
        records = self._log_host
        count = int(records[0])
        # Transport defect: an overflowing or unknown event would otherwise
        # silently lose an original framework side effect.
        if not 0 <= count <= len(WRAPPER_LOG_MESSAGES):
            raise RuntimeError('SMOKE wrapper log transport overflow')
        sites = tuple(map(int, records[1:count + 1]))
        if any(site not in WRAPPER_LOG_MESSAGES for site in sites):
            raise RuntimeError('SMOKE wrapper log transport contains an unknown native site')
        self._pending_logs.extend(WRAPPER_LOG_MESSAGES[site] for site in sites)
        return sites

    def finish_logs(self):
        """Deliver the last call's events (end of run, or a caller that needs them now)."""
        self._drain_wrapper_logs()
        return tuple(self._pending_logs)

    @property
    def pending_logs(self):
        """Literal native payloads awaiting framework delivery.

        MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:262-600 @1d6e370e.
        """
        return tuple(self._pending_logs)

    def flush_logs(self, native):
        """Deliver literal payloads through the original native C logger.

        MPAS-Model-rrfs/src/framework/mpas_log.F:565-595,1072-1114 @1d6e370e.
        A failed write leaves that message and subsequent messages pending.
        Native logging owns formatting, filtering, counters and file effects.
        """
        delivered = 0
        while self._pending_logs:
            native.log_write(self._pending_logs[0])
            self._pending_logs.popleft()
            delivered += 1
        return delivered

    def storage_record(self, label: str, value, *, lower_bounds=(), action='load'):
        """Hydrate or export one original native SAVE/THREADPRIVATE record.

        MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:648-814 @1d6e370e.
        MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90:14-187 @1d6e370e.
        Scalars are original typed host words. Arrays are full, typed Fortran
        device allocations; export copies into the supplied allocation. Neither
        unknown fields nor absent associations receive replacement values.
        This API transports the records supplied by the caller. It does not
        declare that a native capture includes every hidden field.
        """
        if action not in ('load', 'export'):
            raise ValueError('unknown native SMOKE storage action: ' + str(action))
        field = NATIVE_STORAGE[label]
        cp, rank, kind = self._cp, field['rank'], field['type']
        dtype = np.dtype(np.float32 if kind == 'float' else np.int32)
        if len(lower_bounds) != rank:
            raise ValueError('native SMOKE storage lower-bound rank differs: ' + label)
        descriptor = np.zeros(1, dtype=_ARGUMENT_DTYPE)
        descriptor['present'][0] = 1
        if rank:
            if (not isinstance(value, cp.ndarray) or value.dtype != dtype
                    or value.ndim != rank or not value.flags.f_contiguous):
                raise ValueError('native SMOKE storage type/rank/order differs: ' + label)
            lower = tuple(lower_bounds)
            upper = tuple(lo + size - 1 for lo, size in zip(lower, value.shape))
            if any(not isinstance(bound, (int, np.integer)) or not -(2**31) <= bound < 2**31
                   for bound in (*lower, *upper)):
                raise ValueError('native SMOKE storage bounds exceed INTEGER: ' + label)
            descriptor['data'][0] = value.data.ptr
            descriptor['lo'][0, :rank] = lower
            descriptor['hi'][0, :rank] = upper
        else:
            scalar = np.asarray(value)
            if scalar.ndim != 0 or scalar.dtype != dtype:
                raise ValueError('native SMOKE storage scalar precision differs: ' + label)
            if kind in ('pointer_present', 'pointer_alias') and scalar[()] not in (0, 1):
                raise ValueError('native SMOKE storage association/logical word differs: ' + label)
            member = 'fvalue' if kind == 'float' else 'ivalue'
            if action == 'load':
                descriptor[member][0] = scalar[()]
        device_descriptor = cp.asarray(descriptor)
        status = cp.empty(1, dtype=cp.int32)
        self._function('sd_capture_storage')((1,), (1,), (
            self._state, np.int32(_STORAGE_SLOTS[label]), np.int32(action == 'export'),
            device_descriptor, status))
        if int(status.get()[0]):
            raise ValueError('native state association or shape differs: ' + label)
        if rank:
            if action == 'load':
                self._storage_buffers[label] = value
            return value
        row = device_descriptor.get()[0]
        return np.asarray(row['fvalue' if kind == 'float' else 'ivalue'], dtype=dtype)

    def advance(self, arguments: dict, *, lower_bounds: dict | None = None,
                forwarded_absent: frozenset[str] = frozenset()):
        """RRFS driver; MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:34-638 @1d6e370e."""
        return self.call('mpas_smoke_driver', arguments, lower_bounds=lower_bounds,
                         forwarded_absent=forwarded_absent)


def _schema_nodes(node):
    """Registry traversal; MPAS-Model-rrfs/src/core_atmosphere/physics/registry.chemistry.xml:1-548."""
    yield node
    for child in node['children']:
        yield from _schema_nodes(child)


def chemistry_defaults():
    """Native metadata defaults; MPAS-Model-rrfs/src/core_atmosphere/physics/registry.chemistry.xml:56-271."""
    values = {}
    for node in _schema_nodes(CHEMISTRY_SCHEMAS['registry']['atmosphere']['tree']):
        if node['tag'] != 'nml_option':
            continue
        attr = node['attributes']; raw = attr['default_value']; kind = attr['type']
        values[attr['name']] = (int(raw) if kind == 'integer' else float(raw) if kind == 'real'
                               else raw.lower() == 'true' if kind == 'logical' else raw)
    return values


def chemistry_packages(options):
    """Native package predicates; MPAS-Model-rrfs/src/core_atmosphere/physics/registry.chemistry.xml:31-51."""
    o = {**chemistry_defaults(), **options}
    smoke = o['config_smoke_scheme'].rstrip(' ') != 'off'
    dust = o['config_dust_scheme'].rstrip(' ') != 'off'
    anthro = o['config_anthro_scheme'].rstrip(' ')
    rwc = o['config_rwc_scheme'].rstrip(' ') != 'off'
    return dict(mpas_smoke_fine=smoke or rwc, mpas_smoke_in=smoke, mpas_dust_in=dust,
                mpas_anthro_in=anthro != 'off' or rwc,
                mpas_anthro_aero_in=anthro in ('simple_aero', 'simple_gasaero', 'acomp_gasaero'),
                mpas_rwc_in=rwc)


def chemistry_scalar_registry(options):
    """Native scalar order/units; MPAS-Model-rrfs/src/core_atmosphere/physics/registry.chemistry.xml:309-328."""
    packages = chemistry_packages(options)
    for node in _schema_nodes(CHEMISTRY_SCHEMAS['registry']['atmosphere']['tree']):
        if node['tag'] == 'var_array' and node['attributes']['name'] == 'scalars':
            return tuple(dict(c['attributes']) for c in node['children']
                         if any(packages.get(p, False) for p in c['attributes']['packages'].split(';')))
    raise RuntimeError('pinned chemistry scalar registry is absent')


class SmokeSlot:
    """Complete driver callback with bindings supplied by the pool adapter.

    ``bindings(call)`` yields tiles with ``key``, ``from_mpas``, ``wrapper``,
    ``diagnostics`` and ``to_mpas`` mappings, plus optional ``lower_bounds``.
    The pool adapter binds fields/characters and native dimensions; emission
    decode and interpolation stay in Rust. State persists under each tile
    key, matching native module SAVE/THREADPRIVATE ownership.
    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:1025-1299 @1d6e370e.
    """
    def __init__(self, bindings, *, math_source, workspace_words=1_000_000):
        """Tile ownership; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:35-192,1025-1299 @1d6e370e."""
        self._bindings = bindings
        self._math_source = math_source
        self._workspace_words = workspace_words
        self._states = {}

    def __call__(self, call):
        """Native driver order; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:1079-1299 @1d6e370e."""
        for tile in self._bindings(call):
            key = tile['key']
            if key not in self._states:
                self._states[key] = SmokeState(math_source=self._math_source,
                                               workspace_words=self._workspace_words)
            state = self._states[key]
            lo = tile.get('lower_bounds', {})
            absent = tile.get('forwarded_absent', {})
            state.call('smoke_from_mpas_fields', tile['from_mpas'], lower_bounds=lo,
                       forwarded_absent=absent.get('from_mpas', frozenset()))
            state.advance(tile['wrapper'], lower_bounds=lo,
                          forwarded_absent=absent.get('wrapper', frozenset()))
            state.call('smoke_driver_diagnostics', tile['diagnostics'], lower_bounds=lo,
                       forwarded_absent=absent.get('diagnostics', frozenset()))
            state.call('smoke_to_mpas_fields', tile['to_mpas'], lower_bounds=lo,
                       forwarded_absent=absent.get('to_mpas', frozenset()))


def register_slot(registry, bindings, *, math_source, workspace_words=1_000_000):
    """Register the fork slot; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:334-351 @1d6e370e."""
    callback = SmokeSlot(bindings, math_source=math_source, workspace_words=workspace_words)
    registry.register('smoke', 'on', 'hex-cuda', callback)
    return callback


# FENGSHA inputs the driver fetches only with dust on (driver_smoke.F:429-436 @1d6e370e).
DUST_INPUT_FIELDS = ('sandfrac_in', 'clayfrac_in', 'uthres_in', 'uthres_sg_in', 'sep_in', 'albedo_drag', 'feff')

_NATIVE_BOUNDARIES = {
    'from_mpas': 'smoke_from_mpas_fields', 'wrapper': 'mpas_smoke_driver',
    'diagnostics': 'smoke_driver_diagnostics', 'to_mpas': 'smoke_to_mpas_fields',
}


def _native_smoke_field_active(name, boundary, options, scalars):
    """Getter predicates; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:403-496,823-878 @1d6e370e."""
    smoke = options['config_smoke_scheme'].rstrip() != 'off'
    dust = options['config_dust_scheme'].rstrip() != 'off'
    anthro = options['config_anthro_scheme'].rstrip() != 'off'
    # raincv is fetched unconditionally by the native pool API but is read
    # only in the GF-LI branch (driver_smoke.F:688-692). Other suites have
    # no allocation and the CUDA graph writes raincv_p = 0, as native does.
    if name in ('ktop_deep', 'raincv'):
        return options['config_convection_scheme'].rstrip() == 'cu_grell_freitas_li'
    if name in DUST_INPUT_FIELDS:
        return dust
    if name in ('nwfa2d', 'nifa2d'):
        # 44c0849 driver_smoke.F:505-508,899-902: fetched only for TEMPO aerosol-aware microphysics.
        return (options['config_microp_scheme'].rstrip() == 'mp_tempo'
                and bool(options['config_tempo_aerosolaware']))
    if name.startswith('rwc_'):
        return options['config_rwc_scheme'].rstrip() != 'off'
    if name in ('wetdep_resolved', 'drydep_flux') and boundary == 'from_mpas':
        return options['wetdep_ls_opt' if name == 'wetdep_resolved' else 'drydep_opt'] != 0
    if name in ('e_bb_in', 'e_ant_in'):
        return (smoke if name == 'e_bb_in' else anthro) and scalars['num_' + name] > 0
    if name in ('e_bb_out', 'e_dust_out', 'e_ant_out'):
        return scalars['num_' + name] > 0 and (name != 'e_ant_out' or boundary == 'to_mpas' or anthro)
    if name in ('frp_in', 'fre_in'):
        return smoke and options['ebb_dcycle'] == 1
    if name in ('totprcp_prev24', 'hwp_prev24', 'frp_prev24', 'fre_prev24',
                'fire_end_hr', 'fmc_prev24', 'eco_id', 'efs_smold', 'efs_flam', 'efs_rsmold'):
        return smoke and options['ebb_dcycle'] != 1
    if name == 'fire_ef_out':
        return smoke and (options['ebb_dcycle'] == 2 if boundary == 'to_mpas'
                          else options['ebb_dcycle'] != 1)
    if name in ('min_bb_plume', 'max_bb_plume'):
        return smoke and options['plumerise_opt'] > 0
    if name in ('hfx_bb', 'frac_grid_burned', 'qfx_bb'):
        return smoke and options['add_fire_moist_flux' if name == 'qfx_bb' else 'add_fire_heat_flux']
    if name in ('frp_out', 'fre_out', 'hwp', 'coef_bb_dc', 'aod3d_smoke'):
        return smoke
    return True


def bind_native_pools(provider, call):
    """Bind the whole native driver, retaining allocations and chemistry aliases.

    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:290-496,777-880,1020-1299 @1d6e370e.
    ``provider._rrfs_smoke_runtime`` supplies framework/entry-state transport,
    not scientific routines. Its ``entry(native, call)`` returns separate
    from_mpas/wrapper local scalar mappings. Both actual nblocks values are
    required: the fork only assigns these stack locals conditionally. Reading
    an exported global or replacing them with a configured default would not
    preserve the original call boundary. All numerical processing remains in
    the four CUDA graphs, and emission decoding remains in Rust.
    """
    native, cp = provider.native, provider.cp
    runtime = getattr(provider, '_rrfs_smoke_runtime', None)
    if runtime is None:
        raise RuntimeError('native SMOKE entry/framework transport is not installed')
    entry = runtime['entry'](native, call)
    for boundary in ('from_mpas', 'wrapper'):
        if 'nblocks' not in entry[boundary]:
            raise KeyError('native SMOKE boundary scalar is not exported: ' + boundary + '.nblocks')
    options = dict(provider.session.layout.configs)
    options.update(call.options)
    metadata, variables, variable_lower, fields = {}, {}, {}, {}
    views, sizes = set(), {}
    # Read full allocations once. Do not crop resident halo or species axes.
    needed = set()
    for boundary, routine in _NATIVE_BOUNDARIES.items():
        mapping = DRIVER_BINDINGS['wrapper_bindings'] if boundary == 'wrapper' else {}
        for name in ROUTINES[routine]['args']:
            actual = mapping.get(name, name)
            if actual in native.abi['variables']:
                needed.add(actual)
    aliased = set()
    for name in sorted(needed):
        alias = _resident_alias(native, name)
        if alias is not None:
            # The provider's own words in Fortran order: nothing to copy in or write back.
            variables[name], variable_lower[name] = alias
            aliased.add(name)
            continue
        value = native.get_variable(name)
        if value is None:
            variables[name] = None
        elif value[0].ndim:
            host, bounds = value
            variables[name] = cp.asarray(host, order='F')
            variable_lower[name] = bounds
        else:
            metadata[name] = value[0][()]
    # These are driver arguments, not substitutes for native module storage.
    metadata.update(its=provider.session.arguments.its, ite=provider.session.arguments.ite,
                    itimestep=call.itimestep)
    for name in ('gmt', 'curr_julday'):
        value = native.get_table('mpas_atmphys_manager.' + name)
        if value is None or value[0].ndim:
            raise ValueError('native SMOKE manager scalar is absent: ' + name)
        metadata[name] = value[0][()]
    getter_rows = {}
    for source in ('smoke_from_mpas', 'smoke_to_mpas', 'driver_smoke'):
        for row in DRIVER_BINDINGS['pool_bindings'][source]:
            name = row['local'].lower()
            if row['kind'] == 'config':
                value = options[row['field']]
                metadata[name] = (native.get_real_config(row['field'])
                                  if isinstance(value, (float, np.floating)) else value)
            elif row['kind'] == 'dimension':
                key = row['pool'], row['field']
                if key in provider.session.layout.indices:
                    metadata[name] = provider.session.layout.indices[key]
                else:
                    metadata[name] = provider.session.layout.dimensions[row['field']]
            elif row['kind'] == 'array':
                getter_rows.setdefault(name, []).append((source, row))
    for name, rows in getter_rows.items():
        active = [(source, row) for source, row in rows
                  if _native_smoke_field_active(name,
                         'to_mpas' if source == 'smoke_to_mpas' else 'from_mpas', options, metadata)]
        if not active:
            variables[name] = None
            continue
        row = active[0][1]
        # The original driver receives state time level one in this slot.
        level = provider.level if row['time_level'] else 1
        key = row['pool'], row['field'], level
        if key not in fields:
            view = _resident_view(native, row['pool'], row['field'], level)
            if view is not None:
                fields[key], sizes[name] = view
                views.add(key)
            else:
                host = native.get_field(row['pool'], row['field'], level=level)
                fields[key] = cp.asarray(host, order='F')
        variables[name] = fields[key]
        sizes.setdefault(name, variables[name].size)
    first, last = int(metadata['chemistry_start']), int(metadata['chemistry_end'])
    scalars = variables['scalars']
    if not 1 <= first <= last <= scalars.shape[0] or last - first + 1 != metadata['num_chem']:
        raise ValueError('native chemistry scalar interval differs from num_chem')
    if scalars.flags.f_contiguous:
        # A species slice usually has gaps between columns. Pack it for the
        # Fortran descriptors and scatter it back into the SAME full allocation.
        variables['chem'] = cp.asfortranarray(scalars[first - 1:last, :, :])
    else:
        # chem => scalars(chemistry_start:chemistry_end,:,:) (driver_smoke.F:499,881): the provider's own words.
        variables['chem'] = scalars[first - 1:last, :, :]
    # The native allocation's size (its dummy column included), never a view's owned extent.
    metadata['t_size'] = sizes['meshdensity']
    metadata['do_mpas_smoke'] = (options['config_smoke_scheme'].rstrip() != 'off'
                               and metadata['index_smoke_fine'] > 0)
    metadata['do_mpas_dust'] = (options['config_dust_scheme'].rstrip() != 'off'
                              and (metadata['index_dust_fine'] > 0 or metadata['index_dust_coarse'] > 0))
    metadata['do_mpas_anthro'] = (options['config_anthro_scheme'].rstrip() != 'off'
                                and metadata['index_unspc_fine'] > 0)
    metadata['do_mpas_rwc'] = (options['config_rwc_scheme'].rstrip() != 'off'
                             and (metadata['index_smoke_fine'] > 0 or metadata['index_unspc_fine'] > 0))
    if not getattr(provider, '_rrfs_smoke_bind_reported', False):
        # One line per run: what each step binds in place and what it still copies (named, with words).
        import logging
        provider._rrfs_smoke_bind_reported = True
        copied = sorted((n, int(v.size)) for n, v in variables.items()
                        if n in variable_lower and n not in aliased and v is not None)
        logging.getLogger(__name__).info(
            'smoke binding: %d module arrays aliased, %d pool fields in place; copied module arrays %s; '
            'copied pool fields %s', len(aliased), len(views), copied,
            sorted('.'.join(map(str, k)) for k in fields if k not in views))
    tile = {'key': id(native), 'lower_bounds': {}, 'forwarded_absent': {}}
    for boundary, routine in _NATIVE_BOUNDARIES.items():
        arguments, bounds = {}, {}
        mapping = DRIVER_BINDINGS['wrapper_bindings'] if boundary == 'wrapper' else {}
        for name in ROUTINES[routine]['args']:
            symbol = ROUTINES[routine]['symbols'][name]
            if boundary == 'wrapper' and name not in mapping:
                if not symbol['optional']:
                    raise KeyError('native wrapper omits a mandatory actual: ' + name)
                continue
            actual = mapping.get(name, name)
            if symbol['dim']:
                source = 'smoke_to_mpas' if boundary == 'to_mpas' else 'smoke_from_mpas'
                is_local_pointer = any(s == source for s, _ in getter_rows.get(actual, ()))
                arguments[name] = (None if is_local_pointer and not
                    _native_smoke_field_active(actual, boundary, options, metadata)
                    else variables[actual])
                if actual in variable_lower:
                    bounds[name] = variable_lower[actual]
            elif name in entry.get(boundary, {}):
                arguments[name] = entry[boundary][name]
            elif actual in metadata:
                value = metadata[actual]
                arguments[name] = (cp.asarray(np.frombuffer(value.ljust(512).encode('ascii'), dtype=np.uint8))
                                   if symbol['type'] == 'FStr' else value)
            elif actual in native.abi['constants']:
                arguments[name] = native.get_constant(actual)
            elif not symbol['optional']:
                raise KeyError('native SMOKE driver actual is not bound: ' + boundary + '.' + name)
        # Only the source's explicitly forwarded unallocated arrays may be
        # absent mandatory dummies. Never create emission or deposition zeros.
        tile[boundary] = arguments
        tile['lower_bounds'][boundary] = bounds
        tile['forwarded_absent'][boundary] = frozenset(
            name for name, value in arguments.items()
            if value is None and not ROUTINES[routine]['symbols'][name]['optional'])
    return {'smoke_native': {'provider': provider, 'native': native, 'runtime': runtime,
            'tile': tile, 'variables': variables, 'variable_lower': variable_lower,
            'fields': fields, 'chemistry_slice': slice(first - 1, last), 'aliased': frozenset(aliased),
            'views': frozenset(views)}}


def _resident_view(native, pool, name, level):
    """A pool field as the resident authority's own words (no copy) with its native allocation size, or None.

    THE COST THIS REMOVES: the binding copied every pool field into Fortran order (the facade's get_field assembled a
    whole allocation, the binding transposed it), wrote the outputs back through put_field, and copied the chemistry
    species of scalars a third time: at 3 km (mv3km, 206,726 columns, V8X6000 2026-10-29) a 40.3 ms whole-scalars
    transpose before the smoke kernels, a 14.4 ms one after them, and 3-9 ms slices, for the same words. The driver
    boundaries index the pool pointers element by element over the owned columns its..ite, so they take the
    authority's (k,i) words through strided descriptors and write their outputs in place. Words beyond the owned
    extent (the dummy column, allocation tails) are never read or written there.
    """
    import math
    from .resident_rrfs import ResidentFrameworkFacade
    if not isinstance(native, ResidentFrameworkFacade):
        return None
    dtype, shape, _ = native._host._fields[pool, name]
    packed = native._authority(pool, name, level)
    if (packed.dtype != dtype or packed.ndim != len(shape) or packed.shape[:-1] != tuple(shape[:-1])
            or not 0 < packed.shape[-1] <= shape[-1]):
        return None
    return packed, math.prod(shape)


def _resident_alias(native, name):
    """A covered module array as the provider's k,i words seen in Fortran i,k,j order, or None.

    The resident facade's get_variable returns a fresh whole copy assembled by a transposed core copy, the binding
    then made a Fortran-order copy of it, and put_variable wrote every bound array back by another transposed copy:
    about 100 ms of copies per call at 3 km (mv3km, 206,726 columns, V8X6000 2026-10-29), for the same words. When
    the provider's k,i view is the array's whole allocation (as resident_pbl_rrfs._alias for MYNN-EDMF), the kernels
    read and write those words in place; anything else keeps the facade's copies.
    """
    import math
    from .resident_rrfs import ResidentFrameworkFacade
    if not isinstance(native, ResidentFrameworkFacade):
        return None
    stored = native._host.variable_storage(name)
    if stored is None:
        return None
    host, lower = stored
    if not host.ndim or not native._covers(host):
        return None
    _, view = native._view(name)
    if (view is None or view.dtype != host.dtype or not view.flags.c_contiguous
            or view.size != math.prod(host.shape) or view.size == 0):
        return None
    cp = native._cp
    strides, step = [], view.itemsize
    for extent in host.shape:
        strides.append(step)
        step *= extent
    whole = cp.ndarray(host.shape, dtype=host.dtype, memptr=view.data, strides=tuple(strides))
    core = native._core(whole, host)
    if core.shape != view.shape or core.strides != view.strides or core.data.ptr != view.data.ptr:
        return None
    return whole, lower


def native_public_storage(native):
    """Observe readable original module storage without changing native state.

    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_config.F90:14-30 @1d6e370e.
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/mpas_smoke_init.F90:13-42 @1d6e370e.
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90:60-74 @1d6e370e.
    The manifest defines readable public tables. Missing private/local SAVE,
    framework and boundary words remain missing; get_thread_coms is never called.
    Logical storage is copied as raw INTEGER-sized words, without normalizing it.
    The shared native lock covers the whole snapshot, not individual tables.
    """
    import ctypes as ct
    from ..native_backend import _LOCK
    records = {}
    with _LOCK:
        native._check()
        for label, field in NATIVE_STORAGE.items():
            name = label.removeprefix('state.')
            if name not in native.abi['tables']:
                continue
            observed = native.get_table(name)
            if observed is None:
                raise RuntimeError('native SMOKE public allocation is absent: ' + name)
            value, lower = observed
            records[label] = {'value': value, 'lower_bounds': lower}
        for name in ('dbg_opt', 'aero_ind_fdb'):
            word = ct.c_int32.in_dll(native.lib, '__mpas_smoke_config_MOD_' + name)
            value = np.frombuffer(ct.string_at(ct.addressof(word), 4), dtype=np.int32).copy().reshape(())
            records['state.mpas_smoke_config.' + name] = {'value': value, 'lower_bounds': ()}
    return records


def hydrate_native_storage(state, records, *, require_complete=True):
    """Load one observed native snapshot, retaining original types and bounds.

    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/module_zero_plumegen_coms.F90:14-74 @1d6e370e.
    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/SMOKE/MPAS/mpas_smoke_wrapper.F90:648-814 @1d6e370e.
    Validate the whole host snapshot before launching any transfer. Completeness
    is required for associated storage by default; explicit partial mode returns the exact missing
    inventory for transport qualification, without creating replacement records.
    This function does not certify the provenance of caller-supplied observations.
    """
    unknown = set(records) - NATIVE_STORAGE.keys()
    if unknown:
        raise ValueError('unknown native SMOKE storage records: ' + ', '.join(sorted(unknown)))
    missing = tuple(label for label in NATIVE_STORAGE if label not in records)
    parent = 'state.module_zero_plumegen_coms.private_thread_coms.present'
    members = 'state.module_zero_plumegen_coms.private_thread_coms%'
    observed_parent = records.get(parent, {}).get('value')
    absent_parent = (isinstance(observed_parent, np.ndarray)
                     and observed_parent.dtype == np.dtype(np.int32)
                     and observed_parent.ndim == 0 and observed_parent[()] == 0)
    # An observed unassociated original pointer has no readable members.
    # Do not demand, synthesize or load storage behind that null association.
    if absent_parent and any(label.startswith(members) for label in records):
        raise ValueError('native SMOKE snapshot includes members of an absent parent')
    required_missing = tuple(label for label in missing
                             if not (absent_parent and label.startswith(members)))
    if require_complete and required_missing:
        raise ValueError('native SMOKE snapshot is incomplete: ' + ', '.join(required_missing))
    validated = []
    for label in NATIVE_STORAGE:
        if label not in records:
            continue
        field, record = NATIVE_STORAGE[label], records[label]
        value, lower = record['value'], tuple(record['lower_bounds'])
        dtype = np.dtype(np.float32 if field['type'] == 'float' else np.int32)
        if (not isinstance(value, np.ndarray) or value.dtype != dtype
                or value.ndim != field['rank'] or not value.flags.f_contiguous
                or len(lower) != field['rank']):
            raise ValueError('native SMOKE snapshot type/rank/order differs: ' + label)
        upper = tuple(lo + size - 1 for lo, size in zip(lower, value.shape))
        if any(not isinstance(bound, (int, np.integer)) or not -(2**31) <= bound < 2**31
               for bound in (*lower, *upper)):
            raise ValueError('native SMOKE snapshot bounds exceed INTEGER: ' + label)
        if field['type'] in ('pointer_present', 'pointer_alias') and value[()] not in (0, 1):
            raise ValueError('native SMOKE snapshot association word differs: ' + label)
        validated.append((label, value, lower))
    # Original descriptor order establishes the private parent before members.
    # Array owners are retained by storage_record until the state is retired.
    for label, value, lower in validated:
        device = state._cp.asarray(value, order='F') if value.ndim else value
        state.storage_record(label, device, lower_bounds=lower)
    return missing


def native_timer(native, name, start):
    """Call the published GNU native framework, with clear_timer absent.

    MPAS-Model-rrfs/src/framework/mpas_timer.F:88-229,243-310 @1d6e370e.
    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:1158,1256 @1d6e370e.
    The library owns its timer context and compiled timer selection. GNU's
    assumed-length character ABI places a size_t length after explicit
    arguments; its absent optional logical is a null pointer. No timer state
    is initialized, cleared or reconstructed here.
    """
    import ctypes as ct
    from ..native_backend import _LOCK
    encoded = name.encode('ascii')
    with _LOCK:
        native._check()
        if start:
            function = native._bind('__mpas_timer_MOD_mpas_timer_start',
                                    [ct.c_char_p, ct.c_void_p, ct.c_size_t])
            function(encoded, None, len(encoded))
        else:
            function = native._bind('__mpas_timer_MOD_mpas_timer_stop',
                                    [ct.c_char_p, ct.c_size_t])
            function(encoded, len(encoded))


def run_slot(call):
    """Execute and commit the whole slot; MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F:1079-1299 @1d6e370e.

    The route installs a runtime with new_state(native, call), entry(native,
    call), optionally timer(native, name, start) and flush_logs(state, native). new_state
    hydrates actual native module/framework entry state and supplies shared
    device math. The runtime must not execute native SMOKE science. Timers
    remain native framework services through native_timer unless overridden.
    Without an override, literal log events
    are delivered by SmokeState.flush_logs through NativePhysics.log_write.
    """
    context = call.pools['smoke_native']
    provider, native, runtime = context['provider'], context['native'], context['runtime']
    if not hasattr(provider, '_rrfs_smoke_state'):
        provider._rrfs_smoke_state = runtime['new_state'](native, call)
    state, tile = provider._rrfs_smoke_state, context['tile']
    timer = runtime.get('timer', native_timer)
    for boundary, routine in _NATIVE_BOUNDARIES.items():
        if boundary == 'wrapper':
            timer(native, 'mpas_smoke', True)
        state.call(routine, tile[boundary], lower_bounds=tile['lower_bounds'][boundary],
                   forwarded_absent=tile['forwarded_absent'][boundary])
        if boundary == 'wrapper':
            if 'flush_logs' in runtime:
                runtime['flush_logs'](state, native)
            else:
                state.flush_logs(native)
            timer(native, 'mpas_smoke', False)
    variables = context['variables']
    views = context.get('views', frozenset())
    if not any(key[:2] == ('state', 'scalars') for key in views):
        variables['scalars'][context['chemistry_slice'], :, :] = variables['chem']
    # Write complete original allocations with their bounds and time levels;
    # the native to-MPAS loop (including its kte horizontal limit) owns which
    # words change. No host interpolation or field arithmetic occurs here.
    for name, bounds in context['variable_lower'].items():
        if name in context.get('aliased', ()):
            continue
        value = variables[name] if getattr(provider, 'resident', False) else provider.cp.asnumpy(variables[name])
        native.put_variable(name, value, lower=bounds)
    output_fields = {(row['pool'], row['field']) for row in
                     DRIVER_BINDINGS['pool_bindings']['smoke_to_mpas'] if row['kind'] == 'array'}
    chemistry = context['chemistry_slice']
    for (pool, name, level), value in context['fields'].items():
        if (pool, name, level) in views:
            continue  # the boundaries wrote the authority's own words
        if (pool, name) == ('state', 'scalars') and getattr(provider, 'resident', False) and hasattr(native, '_authority'):
            # The to-MPAS boundary and the slot change only the chemistry species of scalars (the slice assignment
            # above); every other species is the authority's own words. Write back that slice alone: the whole
            # (num_scalars, levels, cells) transposed copy was the largest copy of the slot at 3 km.
            target = native._authority(pool, name, level)
            if target.shape == value.shape:
                target[chemistry] = value[chemistry]
                continue
        if (pool, name) in output_fields:
            native.put_field(pool, name, value if getattr(provider, 'resident', False) else provider.cp.asnumpy(value), level=level,
                             nlevels=native._fields[pool, name][2])
    return state
