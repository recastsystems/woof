"""MPAS-A v8.4.2 Noah and glacier surface solver.

Authority: MPAS f34984b, whole module_sf_noahlsm.F and
module_sf_noahlsm_glacial_only.F. Packed replay interface follows their
argument order; numerical state stays on the device. Driver registration
belongs to the MPAS physics seam. WOOF's default backend is untouched.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hip_backend import runtime_limit as _runtime_limit
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from pathlib import Path
import numpy as np

MPAS_SOURCE_COMMIT = "f34984b2c89353cf31c235b08d3b4acd96b37bb3"
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")

COLUMN_LAYOUTS = {'noah': {'float': ['FFROZP', 'DT', 'ZLVL', 'SLDPTH[0]', 'SLDPTH[1]', 'SLDPTH[2]', 'SLDPTH[3]', 'LWDN', 'SOLDN', 'SOLNET', 'SFCPRS', 'PRCP', 'SFCTMP', 'Q2', 'SFCSPD', 'COSZ', 'PRCPRAIN', 'SOLARDIRECT', 'TH2', 'Q2SAT', 'DQSDT2', 'SHDFAC', 'SHDMIN', 'SHDMAX', 'ALB', 'SNOALB', 'TBOT', 'Z0BRD', 'Z0', 'EMISSI', 'EMBRD', 'CMC', 'T1', 'STC[0]', 'STC[1]', 'STC[2]', 'STC[3]', 'SMC[0]', 'SMC[1]', 'SMC[2]', 'SMC[3]', 'SH2O[0]', 'SH2O[1]', 'SH2O[2]', 'SH2O[3]', 'SNOWH', 'SNEQV', 'ALBEDO', 'CH', 'CM', 'ETA', 'SHEAT', 'ETA_KINEMATIC', 'FDOWN', 'EC', 'EDIR', 'ET[0]', 'ET[1]', 'ET[2]', 'ET[3]', 'ETT', 'ESNOW', 'DRIP', 'DEW', 'BETA', 'ETP', 'SSOIL', 'FLX1', 'FLX2', 'FLX3', 'FLX4', 'FVB', 'FBUR', 'FGSN', 'SNOMLT', 'SNCOVR', 'RUNOFF1', 'RUNOFF2', 'RUNOFF3', 'RC', 'PC', 'RSMIN', 'XLAI', 'RCS', 'RCT', 'RCQ', 'RCSOIL', 'SOILW', 'SOILM', 'Q1', 'SMAV[0]', 'SMAV[1]', 'SMAV[2]', 'SMAV[3]', 'SNOTIME1', 'RIBB', 'SMCWLT', 'SMCDRY', 'SMCREF', 'SMCMAX', 'SFHEAD1RT', 'INFXS1RT', 'ETPND1', 'AOASIS', 'XSDA_QFX', 'HFX_PHY', 'QFX_PHY', 'XQNORM', 'HCPCT_FASDAS', 'IRRIGATION_CHANNEL'], 'int': ['IILOC', 'JJLOC', 'ISURBAN', 'NSOIL', 'LOCAL', 'VEGTYP', 'SOILTYP', 'SLOPETYP', 'UA_PHYS', 'RDLAI2D', 'USEMONALB', 'NROOT', 'OPT_THCND', 'FASDAS'], 'entry': 'SFLX'}, 'glacial': {'float': ['FFROZP', 'DT', 'ZLVL', 'SLDPTH[0]', 'SLDPTH[1]', 'SLDPTH[2]', 'SLDPTH[3]', 'LWDN', 'SOLNET', 'SFCPRS', 'PRCP', 'SFCTMP', 'Q2', 'TH2', 'Q2SAT', 'DQSDT2', 'ALB', 'SNOALB', 'TBOT', 'Z0BRD', 'Z0', 'EMISSI', 'EMBRD', 'T1', 'STC[0]', 'STC[1]', 'STC[2]', 'STC[3]', 'SNOWH', 'SNEQV', 'ALBEDO', 'CH', 'ETA', 'SHEAT', 'ETA_KINEMATIC', 'FDOWN', 'ESNOW', 'DEW', 'ETP', 'SSOIL', 'FLX1', 'FLX2', 'FLX3', 'SNOMLT', 'SNCOVR', 'RUNOFF1', 'Q1', 'SNOTIME1', 'RIBB'], 'int': ['IILOC', 'JJLOC', 'ISICE', 'NSOIL'], 'entry': 'SFLX_GLACIAL'}, 'driver': {'float': ['DZ8W[0]', 'DZ8W[1]', 'QV3D[0]', 'QV3D[1]', 'P8W3D[0]', 'P8W3D[1]', 'T3D[0]', 'T3D[1]', 'TSK', 'HFX', 'QFX', 'LH', 'GRDFLX', 'QGH', 'SWDOWN', 'GLW', 'SMSTAV', 'SMSTOT', 'SFCRUNOFF', 'UDRUNOFF', 'VEGFRA', 'ALBEDO', 'ALBBCK', 'ZNT', 'Z0', 'TMN', 'XLAND', 'XICE', 'EMISS', 'EMBCK', 'SNOWC', 'QSFC', 'RAINBL', 'DT', 'DZS[0]', 'DZS[1]', 'DZS[2]', 'DZS[3]', 'SMOIS[0]', 'SMOIS[1]', 'SMOIS[2]', 'SMOIS[3]', 'TSLB[0]', 'TSLB[1]', 'TSLB[2]', 'TSLB[3]', 'SNOW', 'CANWAT', 'CHS', 'CHS2', 'CQS2', 'SR', 'CHKLOWQ', 'LAI', 'QZ0', 'SH2O[0]', 'SH2O[1]', 'SH2O[2]', 'SH2O[3]', 'SNOWH', 'SNOALB', 'SHDMIN', 'SHDMAX', 'SNOTIME', 'ACSNOM', 'ACSNOW', 'SNOPCX', 'POTEVP', 'SMCREL[0]', 'SMCREL[1]', 'SMCREL[2]', 'SMCREL[3]', 'XICE_THRESHOLD', 'RIB', 'NOAHRES', 'FLX4_2D', 'FVB_2D', 'FBUR_2D', 'FGSN_2D'], 'int': ['IVGTYP', 'ISLTYP', 'ISURBAN', 'ISICE', 'ITIMESTEP', 'USEMONALB', 'OPT_THCND'], 'entry': 'DRIVER_COLUMN'}, 'init': {'float': ['TSLB[0]', 'TSLB[1]', 'TSLB[2]', 'TSLB[3]', 'SMOIS[0]', 'SMOIS[1]', 'SMOIS[2]', 'SMOIS[3]', 'SH2O[0]', 'SH2O[1]', 'SH2O[2]', 'SH2O[3]', 'SNOALB', 'SNOW', 'SNOWH'], 'int': ['IVGTYP', 'ISLTYP', 'RESTART', 'INPUT_SFC_ALBEDO'], 'entry': 'INIT_COLUMN'}}
COLUMN_LAYOUTS['sfcdif'] = {'float':['ZLM','Z0','THZ0','THLM','SFCSPD','CZIL','AKMS','AKHS'],'int':[],'entry':'SFCDIF_off'}
TABLE_SCHEMA = {'RD': {'type': 'float', 'dims': [], 'init': '287.04', 'parameter': True, 'optional': False, 'intent': ''}, 'SIGMA': {'type': 'float', 'dims': [], 'init': '5.67E-8', 'parameter': True, 'optional': False, 'intent': ''}, 'CPH2O': {'type': 'float', 'dims': [], 'init': '4.218E+3', 'parameter': True, 'optional': False, 'intent': ''}, 'CPICE': {'type': 'float', 'dims': [], 'init': '2.106E+3', 'parameter': True, 'optional': False, 'intent': ''}, 'LSUBF': {'type': 'float', 'dims': [], 'init': '3.335E+5', 'parameter': True, 'optional': False, 'intent': ''}, 'EMISSI_S': {'type': 'float', 'dims': [], 'init': '0.95', 'parameter': True, 'optional': False, 'intent': ''}, 'LUCATS': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'BARE': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'NATURAL': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_1': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_2': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_3': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_4': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_5': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_6': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_7': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_8': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_9': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_10': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LCZ_11': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'NLUS': {'type': 'int', 'dims': [], 'init': '50', 'parameter': True, 'optional': False, 'intent': ''}, 'LUTYPE': {'type': 'char', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'NROTBL': {'type': 'int', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SNUPTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'RSTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'RGLTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'HSTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SHDTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'MAXALB': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'EMISSMINTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'EMISSMAXTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LAIMINTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LAIMAXTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'Z0MINTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'Z0MAXTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'ALBEDOMINTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'ALBEDOMAXTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'ZTOPVTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'ZBOTVTBL': {'type': 'float', 'dims': ['1:NLUS'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'TOPT_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'CMCMAX_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'CFACTR_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'RSMAX_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SLCATS': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'NSLTYPE': {'type': 'int', 'dims': [], 'init': '30', 'parameter': True, 'optional': False, 'intent': ''}, 'SLTYPE': {'type': 'char', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'BB': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'DRYSMC': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'F11': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'MAXSMC': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'REFSMC': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SATPSI': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SATDK': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SATDW': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'WLTSMC': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'QTZ': {'type': 'float', 'dims': ['1:NSLTYPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SLPCATS': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'NSLOPE': {'type': 'int', 'dims': [], 'init': '30', 'parameter': True, 'optional': False, 'intent': ''}, 'SLOPE_DATA': {'type': 'float', 'dims': ['1:NSLOPE'], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SBETA_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'FXEXP_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'CSOIL_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SALP_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'REFDK_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'REFKDT_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'FRZK_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'ZBOT_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SMLOW_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'SMHIGH_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'CZIL_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'LVCOEF_DATA': {'type': 'float', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'ERR_MESSAGE': {'type': 'char', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'ILOC': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'JLOC': {'type': 'int', 'dims': [], 'init': None, 'parameter': False, 'optional': False, 'intent': ''}, 'CP': {'type': 'float', 'dims': [], 'init': '1004.5', 'parameter': True, 'optional': False, 'intent': ''}, 'R_D': {'type': 'float', 'dims': [], 'init': '287.0', 'parameter': True, 'optional': False, 'intent': ''}, 'XLF': {'type': 'float', 'dims': [], 'init': '3.50E5', 'parameter': True, 'optional': False, 'intent': ''}, 'XLV': {'type': 'float', 'dims': [], 'init': '2.50E6', 'parameter': True, 'optional': False, 'intent': ''}, 'RHOWATER': {'type': 'float', 'dims': [], 'init': '1000.0', 'parameter': True, 'optional': False, 'intent': ''}, 'STBOLT': {'type': 'float', 'dims': [], 'init': '5.67051E-8', 'parameter': True, 'optional': False, 'intent': ''}, 'KARMAN': {'type': 'float', 'dims': [], 'init': '0.4', 'parameter': True, 'optional': False, 'intent': ''}, 'GRAVITY': {'type': 'float', 'dims': [], 'init': '9.80616', 'parameter': True, 'optional': False, 'intent': ''}}

_CORE_CUDA = r"""
struct NoahTables {
float RD;
float SIGMA;
float CPH2O;
float CPICE;
float LSUBF;
float EMISSI_S;
int LUCATS;
int BARE;
int NATURAL;
int LCZ_1;
int LCZ_2;
int LCZ_3;
int LCZ_4;
int LCZ_5;
int LCZ_6;
int LCZ_7;
int LCZ_8;
int LCZ_9;
int LCZ_10;
int LCZ_11;
int NLUS;
char LUTYPE[256];
int NROTBL[50];
float SNUPTBL[50];
float RSTBL[50];
float RGLTBL[50];
float HSTBL[50];
float SHDTBL[50];
float MAXALB[50];
float EMISSMINTBL[50];
float EMISSMAXTBL[50];
float LAIMINTBL[50];
float LAIMAXTBL[50];
float Z0MINTBL[50];
float Z0MAXTBL[50];
float ALBEDOMINTBL[50];
float ALBEDOMAXTBL[50];
float ZTOPVTBL[50];
float ZBOTVTBL[50];
float TOPT_DATA;
float CMCMAX_DATA;
float CFACTR_DATA;
float RSMAX_DATA;
int SLCATS;
int NSLTYPE;
char SLTYPE[256];
float BB[30];
float DRYSMC[30];
float F11[30];
float MAXSMC[30];
float REFSMC[30];
float SATPSI[30];
float SATDK[30];
float SATDW[30];
float WLTSMC[30];
float QTZ[30];
int SLPCATS;
int NSLOPE;
float SLOPE_DATA[30];
float SBETA_DATA;
float FXEXP_DATA;
float CSOIL_DATA;
float SALP_DATA;
float REFDK_DATA;
float REFKDT_DATA;
float FRZK_DATA;
float ZBOT_DATA;
float SMLOW_DATA;
float SMHIGH_DATA;
float CZIL_DATA;
float LVCOEF_DATA;
char ERR_MESSAGE[256];
int ILOC;
int JLOC;
float CP;
float R_D;
float XLF;
float XLV;
float RHOWATER;
float STBOLT;
float KARMAN;
float GRAVITY;
int error;
};
template<class T> __device__ T noah_min(T a,T b){return a<b?a:b;}
template<class T> __device__ T noah_max(T a,T b){return a>b?a:b;}
template<class T> __device__ T noah_abs(T a){return a<0?-a:a;}
__device__ float noah_sign(float a,float b){return b<0?-noah_abs(a):noah_abs(a);}
__device__ int noah_mod(int a,int b){return a%b;}
__device__ float noah_powi(float a,int n){if(n<0)return DV(1.0f,noah_powi(a,-n));float r=1.0f;while(n){if(n&1)r=MU(r,a);n>>=1;if(n)a=MU(a,a);}return r;}
__device__ void noah_sflx(NoahTables& tables, int& IILOC, int& JJLOC, float& FFROZP, int& ISURBAN, float& DT, float& ZLVL, int& NSOIL, float* SLDPTH, bool& LOCAL, const char* LLANDUSE, const char* LSOIL, float& LWDN, float& SOLDN, float& SOLNET, float& SFCPRS, float& PRCP, float& SFCTMP, float& Q2, float& SFCSPD, float& COSZ, float& PRCPRAIN, float& SOLARDIRECT, float& TH2, float& Q2SAT, float& DQSDT2, int& VEGTYP, int& SOILTYP, int& SLOPETYP, float& SHDFAC, float& SHDMIN, float& SHDMAX, float& ALB, float& SNOALB, float& TBOT, float& Z0BRD, float& Z0, float& EMISSI, float& EMBRD, float& CMC, float& T1, float* STC, float* SMC, float* SH2O, float& SNOWH, float& SNEQV, float& ALBEDO, float& CH, float& CM, float& ETA, float& SHEAT, float& ETA_KINEMATIC, float& FDOWN, float& EC, float& EDIR, float* ET, float& ETT, float& ESNOW, float& DRIP, float& DEW, float& BETA, float& ETP, float& SSOIL, float& FLX1, float& FLX2, float& FLX3, float& FLX4, float& FVB, float& FBUR, float& FGSN, bool& UA_PHYS, float& SNOMLT, float& SNCOVR, float& RUNOFF1, float& RUNOFF2, float& RUNOFF3, float& RC, float& PC, float& RSMIN, float& XLAI, float& RCS, float& RCT, float& RCQ, float& RCSOIL, float& SOILW, float& SOILM, float& Q1, float* SMAV, bool& RDLAI2D, bool& USEMONALB, float& SNOTIME1, float& RIBB, float& SMCWLT, float& SMCDRY, float& SMCREF, float& SMCMAX, int& NROOT, float& SFHEAD1RT, float& INFXS1RT, float& ETPND1, int& OPT_THCND, float& AOASIS, float& XSDA_QFX, float& HFX_PHY, float& QFX_PHY, float& XQNORM, int& FASDAS, float& HCPCT_FASDAS, float& IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL);
__device__ void noah_alcalc(NoahTables& tables, float& ALB, float& SNOALB, float& EMBRD, float& SHDFAC, float& SHDMIN, float& SNCOVR, float& TSNOW, float& ALBEDO, float& EMISSI, float& DT, bool& SNOWNG, float& SNOTIME1, float& LVCOEF);
__device__ void noah_canres(NoahTables& tables, float& SOLAR, float& CH, float& SFCTMP, float& Q2, float& SFCPRS, float* SMC, float* ZSOIL, int& NSOIL, float& SMCWLT, float& SMCREF, float& RSMIN, float& RC, float& PC, int& NROOT, float& Q2SAT, float& DQSDT2, float& TOPT, float& RSMAX, float& RGL, float& HS, float& XLAI, float& RCS, float& RCT, float& RCQ, float& RCSOIL, float& EMISSI);
__device__ void noah_csnow(NoahTables& tables, float& SNCOND, float& DSNOW);
__device__ void noah_devap(NoahTables& tables, float& EDIR, float& ETP1, float& SMC, float& ZSOIL, float& SHDFAC, float& SMCMAX, float& BEXP, float& DKSAT, float& DWSAT, float& SMCDRY, float& SMCREF, float& SMCWLT, float& FXEXP);
__device__ void noah_devap_hydro(NoahTables& tables, float& EDIR, float& ETP1, float& SMC, float& ZSOIL, float& SHDFAC, float& SMCMAX, float& BEXP, float& DKSAT, float& DWSAT, float& SMCDRY, float& SMCREF, float& SMCWLT, float& FXEXP, float& SFHEAD1RT, float& ETPND1, float& DT);
__device__ void noah_evapo(NoahTables& tables, float& ETA1, float* SMC, int& NSOIL, float& CMC, float& ETP1, float& DT, float* ZSOIL, float* SH2O, float& SMCMAX, float& BEXP, float& PC, float& SMCWLT, float& DKSAT, float& DWSAT, float& SMCREF, float& SHDFAC, float& CMCMAX, float& SMCDRY, float& CFACTR, float& EDIR, float& EC, float* ET, float& ETT, float& SFCTMP, float& Q2, int& NROOT, float* RTDIS, float& FXEXP, float& SFHEAD1RT, float& ETPND1);
__device__ void noah_fac2mit(NoahTables& tables, float& SMCMAX, float& FLIMIT);
__device__ void noah_frh2o(NoahTables& tables, float& FREE, float& TKELV, float& SMC, float& SH2O, float& SMCMAX, float& BEXP, float& PSIS);
__device__ void noah_hrt(NoahTables& tables, float* RHSTS, float* STC, float* SMC, float& SMCMAX, int& NSOIL, float* ZSOIL, float& YY, float& ZZ1, float& TBOT, float& ZBOT, float& PSISAT, float* SH2O, float& DT, float& BEXP, int& SOILTYP, int& OPT_THCND, float& F1, float& DF1, float& QUARTZ, float& CSOIL, float* AI, float* BI, float* CI, int& VEGTYP, int& ISURBAN, float& HCPCT_FASDAS);
__device__ void noah_hstep(NoahTables& tables, float* STCOUT, float* STCIN, float* RHSTS, float& DT, int& NSOIL, float* AI, float* BI, float* CI);
__device__ void noah_nopac(NoahTables& tables, float& ETP, float& ETA, float& PRCP, float* SMC, float& SMCMAX, float& SMCWLT, float& SMCREF, float& SMCDRY, float& CMC, float& CMCMAX, int& NSOIL, float& DT, float& SHDFAC, float& SBETA, float& Q2, float& T1, float& SFCTMP, float& T24, float& TH2, float& FDOWN, float& F1, float& EMISSI, float& SSOIL, float* STC, float& EPSCA, float& BEXP, float& PC, float& RCH, float& RR, float& CFACTR, float* SH2O, float& SLOPE, float& KDT, float& FRZFACT, float& PSISAT, float* ZSOIL, float& DKSAT, float& DWSAT, float& TBOT, float& ZBOT, float& RUNOFF1, float& RUNOFF2, float& RUNOFF3, float& EDIR, float& EC, float* ET, float& ETT, int& NROOT, float* RTDIS, float& QUARTZ, float& FXEXP, float& CSOIL, float& BETA, float& DRIP, float& DEW, float& FLX1, float& FLX3, int& VEGTYP, int& ISURBAN, float& SFHEAD1RT, float& INFXS1RT, float& ETPND1, int& SOILTYP, int& OPT_THCND, float& XSDA_QFX, float& QFX_PHY, float& XQNORM, int& FASDAS, float& HCPCT_FASDAS, float& IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL);
__device__ void noah_penman(NoahTables& tables, float& SFCTMP, float& SFCPRS, float& CH, float& T2V, float& TH2, float& PRCP, float& FDOWN, float& T24, float& SSOIL, float& Q2, float& Q2SAT, float& ETP, float& RCH, float& EPSCA, float& RR, bool& SNOWNG, bool& FRZGRA, float& DQSDT2, float& FLX2, float& EMISSI_IN, float& SNEQV, float& T1, float& SNCOVR, float& AOASIS, float& ALBEDO, float& SOLDN, float& FVB, float& GAMA, float& STC1, float& ETPN, float& FLX4, bool& UA_PHYS);
__device__ void noah_redprm(NoahTables& tables, int& VEGTYP, int& SOILTYP, int& SLOPETYP, float& CFACTR, float& CMCMAX, float& RSMAX, float& TOPT, float& REFKDT, float& KDT, float& SBETA, float& SHDFAC, float& RSMIN, float& RGL, float& HS, float& ZBOT, float& FRZX, float& PSISAT, float& SLOPE, float& SNUP, float& SALP, float& BEXP, float& DKSAT, float& DWSAT, float& SMCMAX, float& SMCWLT, float& SMCREF, float& SMCDRY, float& F1, float& QUARTZ, float& FXEXP, float* RTDIS, float* SLDPTH, float* ZSOIL, int& NROOT, int& NSOIL, float& CZIL, float& LAIMIN, float& LAIMAX, float& EMISSMIN, float& EMISSMAX, float& ALBEDOMIN, float& ALBEDOMAX, float& Z0MIN, float& Z0MAX, float& CSOIL, float& PTU, const char* LLANDUSE, const char* LSOIL, bool& LOCAL, float& LVCOEF, float& ZTOPV, float& ZBOTV);
__device__ void noah_rosr12(NoahTables& tables, float* P, float* A, float* B, float* C, float* D, float* DELTA, int& NSOIL);
__device__ void noah_shflx(NoahTables& tables, float& SSOIL, float* STC, float* SMC, float& SMCMAX, int& NSOIL, float& T1, float& DT, float& YY, float& ZZ1, float* ZSOIL, float& TBOT, float& ZBOT, float& SMCWLT, float& PSISAT, float* SH2O, float& BEXP, float& F1, float& DF1, float& QUARTZ, float& CSOIL, int& VEGTYP, int& ISURBAN, int& SOILTYP, int& OPT_THCND, float& HCPCT_FASDAS);
__device__ void noah_smflx(NoahTables& tables, float* SMC, int& NSOIL, float& CMC, float& DT, float& PRCP1, float* ZSOIL, float* SH2O, float& SLOPE, float& KDT, float& FRZFACT, float& SMCMAX, float& BEXP, float& SMCWLT, float& DKSAT, float& DWSAT, float& SHDFAC, float& CMCMAX, float& RUNOFF1, float& RUNOFF2, float& RUNOFF3, float& EDIR, float& EC, float* ET, float& DRIP, float& SFHEAD1RT, float& INFXS1RT, float& IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL);
__device__ void noah_snfrac(NoahTables& tables, float& SNEQV, float& SNUP, float& SALP, float& SNOWH, float& SNCOVR, float& XLAI, float& SHDFAC, float& FVB, float& GAMA, float& FBUR, float& FGSN, float& ZTOPV, float& ZBOTV, bool& UA_PHYS);
__device__ void noah_snksrc(NoahTables& tables, float& TSNSR, float& TAVG, float& SMC, float& SH2O, float* ZSOIL, int& NSOIL, float& SMCMAX, float& PSISAT, float& BEXP, float& DT, int& K, float& QTOT);
__device__ void noah_snopac(NoahTables& tables, float& ETP, float& ETA, float& PRCP, float& PRCPF, bool& SNOWNG, float* SMC, float& SMCMAX, float& SMCWLT, float& SMCREF, float& SMCDRY, float& CMC, float& CMCMAX, int& NSOIL, float& DT, float& SBETA, float& DF1, float& Q2, float& T1, float& SFCTMP, float& T24, float& TH2, float& FDOWN, float& F1, float& SSOIL, float* STC, float& EPSCA, float& SFCPRS, float& BEXP, float& PC, float& RCH, float& RR, float& CFACTR, float& SNCOVR, float& ESD, float& SNDENS, float& SNOWH, float* SH2O, float& SLOPE, float& KDT, float& FRZFACT, float& PSISAT, float* ZSOIL, float& DWSAT, float& DKSAT, float& TBOT, float& ZBOT, float& SHDFAC, float& RUNOFF1, float& RUNOFF2, float& RUNOFF3, float& EDIR, float& EC, float* ET, float& ETT, int& NROOT, float& SNOMLT, float* RTDIS, float& QUARTZ, float& FXEXP, float& CSOIL, float& BETA, float& DRIP, float& DEW, float& FLX1, float& FLX2, float& FLX3, float& ESNOW, float& ETNS, float& EMISSI, float& RIBB, float& SOLDN, int& ISURBAN, int& VEGTYP, float& ETPN, float& FLX4, bool& UA_PHYS, float& SFHEAD1RT, float& INFXS1RT, float& ETPND1, int& SOILTYP, int& OPT_THCND, float& QFX_PHY, int& FASDAS, float& HCPCT_FASDAS);
__device__ void noah_snowpack(NoahTables& tables, float& ESD, float& DTSEC, float& SNOWH, float& SNDENS, float& TSNOW, float& TSOIL, float& SNOMLT, bool& UA_PHYS);
__device__ void noah_snowz0(NoahTables& tables, float& SNCOVR, float& Z0, float& Z0BRD, float& SNOWH, float& FBUR, float& FGSN, float& SHDMAX, bool& UA_PHYS);
__device__ void noah_snow_new(NoahTables& tables, float& TEMP, float& NEWSN, float& SNOWH, float& SNDENS);
__device__ void noah_srt(NoahTables& tables, float* RHSTT, float& EDIR, float* ET, float* SH2O, float* SH2OA, int& NSOIL, float& PCPDRP, float* ZSOIL, float& DWSAT, float& DKSAT, float& SMCMAX, float& BEXP, float& RUNOFF1, float& RUNOFF2, float& DT, float& SMCWLT, float& SLOPE, float& KDT, float& FRZX, float* SICE, float* AI, float* BI, float* CI, float& SFHEAD1RT, float& INFXS1RT);
__device__ void noah_sstep(NoahTables& tables, float* SH2OOUT, float* SH2OIN, float& CMC, float* RHSTT, float& RHSCT, float& DT, int& NSOIL, float& SMCMAX, float& CMCMAX, float& RUNOFF3, float* ZSOIL, float* SMC, float* SICE, float* AI, float* BI, float* CI, float& INFXS1RT);
__device__ void noah_tbnd(NoahTables& tables, float& TU, float& TB, float* ZSOIL, float& ZBOT, int& K, int& NSOIL, float& TBND1);
__device__ void noah_tdfcnd(NoahTables& tables, float& DF, float& SMC, float& QZ, float& SMCMAX, float& SH2O, float& BEXP, float& PSISAT, int& SOILTYP, int& OPT_THCND);
__device__ void noah_tmpavg(NoahTables& tables, float& TAVG, float& TUP, float& TM, float& TDN, float* ZSOIL, int& NSOIL, int& K);
__device__ void noah_transp(NoahTables& tables, float* ET, int& NSOIL, float& ETP1, float* SMC, float& CMC, float* ZSOIL, float& SHDFAC, float& SMCWLT, float& CMCMAX, float& PC, float& CFACTR, float& SMCREF, float& SFCTMP, float& Q2, int& NROOT, float* RTDIS);
__device__ void noah_wdfcnd(NoahTables& tables, float& WDF, float& WCND, float& SMC, float& SMCMAX, float& BEXP, float& DKSAT, float& DWSAT, float& SICEMAX);
__device__ void noah_sfcdif_off(NoahTables& tables, float& ZLM, float& Z0, float& THZ0, float& THLM, float& SFCSPD, float& CZIL, float& AKMS, float& AKHS);
__device__ void gl_sflx_glacial(NoahTables& tables, int& IILOC, int& JJLOC, int& ISICE, float& FFROZP, float& DT, float& ZLVL, int& NSOIL, float* SLDPTH, float& LWDN, float& SOLNET, float& SFCPRS, float& PRCP, float& SFCTMP, float& Q2, float& TH2, float& Q2SAT, float& DQSDT2, float& ALB, float& SNOALB, float& TBOT, float& Z0BRD, float& Z0, float& EMISSI, float& EMBRD, float& T1, float* STC, float& SNOWH, float& SNEQV, float& ALBEDO, float& CH, float& ETA, float& SHEAT, float& ETA_KINEMATIC, float& FDOWN, float& ESNOW, float& DEW, float& ETP, float& SSOIL, float& FLX1, float& FLX2, float& FLX3, float& SNOMLT, float& SNCOVR, float& RUNOFF1, float& Q1, float& SNOTIME1, float& RIBB);
__device__ void gl_alcalc(NoahTables& tables, float& ALB, float& SNOALB, float& EMBRD, float& TSNOW, float& ALBEDO, float& EMISSI, float& DT, bool& SNOWNG, float& SNOTIME1);
__device__ void gl_csnow(NoahTables& tables, float& SNCOND, float& DSNOW);
__device__ void gl_hrtice(NoahTables& tables, float* RHSTS, float* STC, float& TBOT, int& NSOIL, float* ZSOIL, float& YY, float& ZZ1, float& DF1, float* AI, float* BI, float* CI);
__device__ void gl_hstep(NoahTables& tables, float* STCOUT, float* STCIN, float* RHSTS, float& DT, int& NSOIL, float* AI, float* BI, float* CI);
__device__ void gl_penman(NoahTables& tables, float& SFCTMP, float& SFCPRS, float& CH, float& TH2, float& PRCP, float& FDOWN, float& T24, float& SSOIL, float& Q2, float& Q2SAT, float& ETP, float& RCH, float& RR, bool& SNOWNG, bool& FRZGRA, float& DQSDT2, float& FLX2, float& EMISSI, float& T1);
__device__ void gl_shflx(NoahTables& tables, float* STC, int& NSOIL, float& DT, float& YY, float& ZZ1, float* ZSOIL, float& TBOT, float& DF1);
__device__ void gl_snopac(NoahTables& tables, float& ETP, float& ETA, float& PRCP, float& PRCPF, bool& SNOWNG, int& NSOIL, float& DT, float& DF1, float& Q2, float& T1, float& SFCTMP, float& T24, float& TH2, float& FDOWN, float& SSOIL, float* STC, float& SFCPRS, float& RCH, float& RR, float& SNEQV, float& SNDENS, float& SNOWH, float* ZSOIL, float& TBOT, float& SNOMLT, float& DEW, float& FLX1, float& FLX2, float& FLX3, float& ESNOW, float& EMISSI, float& RIBB);
__device__ void gl_snowpack(NoahTables& tables, float& SNEQV, float& DTSEC, float& SNOWH, float& SNDENS, float& TSNOW, float& TSOIL);
__device__ void gl_snowz0(NoahTables& tables, float& Z0, float& Z0BRD, float& SNOWH);
__device__ void gl_snow_new(NoahTables& tables, float& TEMP, float& NEWSN, float& SNOWH, float& SNDENS);
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:73-892
__device__ void noah_sflx(NoahTables& tables, int& IILOC, int& JJLOC, float& FFROZP, int& ISURBAN, float& DT, float& ZLVL, int& NSOIL, float* SLDPTH, bool& LOCAL, const char* LLANDUSE, const char* LSOIL, float& LWDN, float& SOLDN, float& SOLNET, float& SFCPRS, float& PRCP, float& SFCTMP, float& Q2, float& SFCSPD, float& COSZ, float& PRCPRAIN, float& SOLARDIRECT, float& TH2, float& Q2SAT, float& DQSDT2, int& VEGTYP, int& SOILTYP, int& SLOPETYP, float& SHDFAC, float& SHDMIN, float& SHDMAX, float& ALB, float& SNOALB, float& TBOT, float& Z0BRD, float& Z0, float& EMISSI, float& EMBRD, float& CMC, float& T1, float* STC, float* SMC, float* SH2O, float& SNOWH, float& SNEQV, float& ALBEDO, float& CH, float& CM, float& ETA, float& SHEAT, float& ETA_KINEMATIC, float& FDOWN, float& EC, float& EDIR, float* ET, float& ETT, float& ESNOW, float& DRIP, float& DEW, float& BETA, float& ETP, float& SSOIL, float& FLX1, float& FLX2, float& FLX3, float& FLX4, float& FVB, float& FBUR, float& FGSN, bool& UA_PHYS, float& SNOMLT, float& SNCOVR, float& RUNOFF1, float& RUNOFF2, float& RUNOFF3, float& RC, float& PC, float& RSMIN, float& XLAI, float& RCS, float& RCT, float& RCQ, float& RCSOIL, float& SOILW, float& SOILM, float& Q1, float* SMAV, bool& RDLAI2D, bool& USEMONALB, float& SNOTIME1, float& RIBB, float& SMCWLT, float& SMCDRY, float& SMCREF, float& SMCMAX, int& NROOT, float& SFHEAD1RT, float& INFXS1RT, float& ETPND1, int& OPT_THCND, float& AOASIS, float& XSDA_QFX, float& HFX_PHY, float& QFX_PHY, float& XQNORM, int& FASDAS, float& HCPCT_FASDAS, float& IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL){
auto& ILOC=tables.ILOC;
auto& JLOC=tables.JLOC;
auto& CP=tables.CP;
bool FRZGRA;
bool SNOWNG;
int KZ;
int K;
int IOUT;
float RTDIS[20];
float ZSOIL[20];
float ZTOPV;
float ZBOTV;
float GAMA;
float FNET;
float ETPN;
float RU;
float BEXP;
float CFACTR;
float CMCMAX;
float CSOIL;
float CZIL;
float DF1;
float DF1H;
float DF1A;
float DKSAT;
float DWSAT;
float DSOIL;
float DTOT;
float FRCSNO;
float FRCSOI;
float EPSCA;
float F1;
float FXEXP;
float FRZX;
float HS;
float KDT;
float LVH2O = 0x1.314c400000000p+21f;
float PRCP1;
float PSISAT;
float QUARTZ;
float R = 0x1.1f0a3e0000000p+8f;
float RCH;
float REFKDT;
float RR;
float RGL;
float RSMAX;
float RSNOW;
float SNDENS;
float SNCOND;
float SBETA;
float SN_NEW;
float SLOPE;
float SNUP;
float SALP;
float SOILWM;
float SOILWW;
float T1V;
float T24;
float T2V;
float TH2V;
float TOPT;
float TFREEZ = 0x1.1126660000000p+8f;
float TSNOW;
float ZBOT;
float PRCPF;
float ETNS;
float PTU;
float LSUBS = 0x1.5975800000000p+21f;
float LVCOEF;
float INTERP_FRACTION;
float LAIMIN;
float LAIMAX;
float ALBEDOMIN;
float ALBEDOMAX;
float EMISSMIN;
float EMISSMAX;
float Z0MIN;
float Z0MAX;
 // module_sf_noahlsm.F:299
ILOC = IILOC; // module_sf_noahlsm.F:398
JLOC = JJLOC; // module_sf_noahlsm.F:399
RUNOFF1 = 0x0.0p+0f; // module_sf_noahlsm.F:401
RUNOFF2 = 0x0.0p+0f; // module_sf_noahlsm.F:402
RUNOFF3 = 0x0.0p+0f; // module_sf_noahlsm.F:403
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm.F:404
if ((!UA_PHYS)) { // module_sf_noahlsm.F:406
FLX4 = 0x0.0p+0f; // module_sf_noahlsm.F:407
FVB = 0x0.0p+0f; // module_sf_noahlsm.F:408
FBUR = 0x0.0p+0f; // module_sf_noahlsm.F:409
FGSN = 0x0.0p+0f; // module_sf_noahlsm.F:410
} // module_sf_noahlsm.F:411
ZSOIL[(1)-(1)] = (-SLDPTH[(1)-(1)]); // module_sf_noahlsm.F:418
for (KZ=2; KZ<=NSOIL; KZ+=1) { // module_sf_noahlsm.F:419
ZSOIL[(KZ)-(1)] = AD((-SLDPTH[(KZ)-(1)]),ZSOIL[((KZ - 1))-(1)]); // module_sf_noahlsm.F:420
} // module_sf_noahlsm.F:421
{  noah_redprm(tables,VEGTYP,SOILTYP,SLOPETYP,CFACTR,CMCMAX,RSMAX,TOPT,REFKDT,KDT,SBETA,SHDFAC,RSMIN,RGL,HS,ZBOT,FRZX,PSISAT,SLOPE,SNUP,SALP,BEXP,DKSAT,DWSAT,SMCMAX,SMCWLT,SMCREF,SMCDRY,F1,QUARTZ,FXEXP,RTDIS,SLDPTH,ZSOIL,NROOT,NSOIL,CZIL,LAIMIN,LAIMAX,EMISSMIN,EMISSMAX,ALBEDOMIN,ALBEDOMAX,Z0MIN,Z0MAX,CSOIL,PTU,LLANDUSE,LSOIL,LOCAL,LVCOEF,ZTOPV,ZBOTV); if(tables.error)return; } // module_sf_noahlsm.F:427
if ((VEGTYP == ISURBAN)) { // module_sf_noahlsm.F:437
SHDFAC = 0x1.99999a0000000p-5f; // module_sf_noahlsm.F:438
RSMIN = 0x1.9000000000000p+8f; // module_sf_noahlsm.F:439
SMCMAX = 0x1.cccccc0000000p-2f; // module_sf_noahlsm.F:440
SMCREF = 0x1.ae147a0000000p-2f; // module_sf_noahlsm.F:441
SMCWLT = 0x1.99999a0000000p-2f; // module_sf_noahlsm.F:442
SMCDRY = 0x1.99999a0000000p-2f; // module_sf_noahlsm.F:443
} // module_sf_noahlsm.F:444
if ((SHDFAC >= SHDMAX)) { // module_sf_noahlsm.F:446
EMBRD = EMISSMAX; // module_sf_noahlsm.F:447
if ((!RDLAI2D)) { // module_sf_noahlsm.F:448
XLAI = LAIMAX; // module_sf_noahlsm.F:449
} // module_sf_noahlsm.F:450
if ((!USEMONALB)) { // module_sf_noahlsm.F:451
ALB = ALBEDOMIN; // module_sf_noahlsm.F:452
} // module_sf_noahlsm.F:453
Z0BRD = Z0MAX; // module_sf_noahlsm.F:454
} else if ((SHDFAC <= SHDMIN)) { // module_sf_noahlsm.F:455
EMBRD = EMISSMIN; // module_sf_noahlsm.F:456
if ((!RDLAI2D)) { // module_sf_noahlsm.F:457
XLAI = LAIMIN; // module_sf_noahlsm.F:458
} // module_sf_noahlsm.F:459
if ((!USEMONALB)) { // module_sf_noahlsm.F:460
ALB = ALBEDOMAX; // module_sf_noahlsm.F:461
} // module_sf_noahlsm.F:462
Z0BRD = Z0MIN; // module_sf_noahlsm.F:463
} else { // module_sf_noahlsm.F:464
if ((SHDMAX > SHDMIN)) { // module_sf_noahlsm.F:466
INTERP_FRACTION = DV(SU(SHDFAC,SHDMIN),SU(SHDMAX,SHDMIN)); // module_sf_noahlsm.F:468
INTERP_FRACTION = noah_min(INTERP_FRACTION,0x1.0000000000000p+0f); // module_sf_noahlsm.F:470
INTERP_FRACTION = noah_max(INTERP_FRACTION,0x0.0p+0f); // module_sf_noahlsm.F:471
EMBRD = AD(MU(SU(0x1.0000000000000p+0f,INTERP_FRACTION),EMISSMIN),MU(INTERP_FRACTION,EMISSMAX)); // module_sf_noahlsm.F:473
if ((!RDLAI2D)) { // module_sf_noahlsm.F:474
XLAI = AD(MU(SU(0x1.0000000000000p+0f,INTERP_FRACTION),LAIMIN),MU(INTERP_FRACTION,LAIMAX)); // module_sf_noahlsm.F:475
} // module_sf_noahlsm.F:476
if ((!USEMONALB)) { // module_sf_noahlsm.F:477
ALB = AD(MU(SU(0x1.0000000000000p+0f,INTERP_FRACTION),ALBEDOMAX),MU(INTERP_FRACTION,ALBEDOMIN)); // module_sf_noahlsm.F:478
} // module_sf_noahlsm.F:479
Z0BRD = AD(MU(SU(0x1.0000000000000p+0f,INTERP_FRACTION),Z0MIN),MU(INTERP_FRACTION,Z0MAX)); // module_sf_noahlsm.F:480
} else { // module_sf_noahlsm.F:482
EMBRD = AD(MU(0x1.0000000000000p-1f,EMISSMIN),MU(0x1.0000000000000p-1f,EMISSMAX)); // module_sf_noahlsm.F:484
if ((!RDLAI2D)) { // module_sf_noahlsm.F:485
XLAI = AD(MU(0x1.0000000000000p-1f,LAIMIN),MU(0x1.0000000000000p-1f,LAIMAX)); // module_sf_noahlsm.F:486
} // module_sf_noahlsm.F:487
if ((!USEMONALB)) { // module_sf_noahlsm.F:488
ALB = AD(MU(0x1.0000000000000p-1f,ALBEDOMIN),MU(0x1.0000000000000p-1f,ALBEDOMAX)); // module_sf_noahlsm.F:489
} // module_sf_noahlsm.F:490
Z0BRD = AD(MU(0x1.0000000000000p-1f,Z0MIN),MU(0x1.0000000000000p-1f,Z0MAX)); // module_sf_noahlsm.F:491
} // module_sf_noahlsm.F:493
} // module_sf_noahlsm.F:495
SNOWNG = false; // module_sf_noahlsm.F:499
FRZGRA = false; // module_sf_noahlsm.F:500
if ((SNEQV <= 0x1.ad7f2a0000000p-24f)) { // module_sf_noahlsm.F:507
SNEQV = 0x0.0p+0f; // module_sf_noahlsm.F:508
SNDENS = 0x0.0p+0f; // module_sf_noahlsm.F:509
SNOWH = 0x0.0p+0f; // module_sf_noahlsm.F:510
SNCOND = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:511
} else { // module_sf_noahlsm.F:512
SNDENS = DV(SNEQV,SNOWH); // module_sf_noahlsm.F:513
if ((SNDENS > 0x1.0000000000000p+0f)) { // module_sf_noahlsm.F:514
tables.error = 515; return; // module_sf_noahlsm.F:515
} // module_sf_noahlsm.F:516
{  noah_csnow(tables,SNCOND,SNDENS); if(tables.error)return; } // module_sf_noahlsm.F:517
} // module_sf_noahlsm.F:518
if ((PRCP > 0x0.0p+0f)) { // module_sf_noahlsm.F:525
if ((FFROZP > 0x1.0000000000000p-1f)) { // module_sf_noahlsm.F:528
SNOWNG = true; // module_sf_noahlsm.F:529
} else { // module_sf_noahlsm.F:530
if ((T1 <= TFREEZ)) { FRZGRA = true; } // module_sf_noahlsm.F:531
} // module_sf_noahlsm.F:532
} // module_sf_noahlsm.F:533
if ((SNOWNG || FRZGRA)) { // module_sf_noahlsm.F:541
SN_NEW = MU(MU(PRCP,DT),0x1.0624de0000000p-10f); // module_sf_noahlsm.F:542
SNEQV = AD(SNEQV,SN_NEW); // module_sf_noahlsm.F:543
PRCPF = 0x0.0p+0f; // module_sf_noahlsm.F:544
{  noah_snow_new(tables,SFCTMP,SN_NEW,SNOWH,SNDENS); if(tables.error)return; } // module_sf_noahlsm.F:550
{  noah_csnow(tables,SNCOND,SNDENS); if(tables.error)return; } // module_sf_noahlsm.F:551
} else { // module_sf_noahlsm.F:558
PRCPF = PRCP; // module_sf_noahlsm.F:559
} // module_sf_noahlsm.F:560
if ((SNEQV == 0x0.0p+0f)) { // module_sf_noahlsm.F:567
SNCOVR = 0x0.0p+0f; // module_sf_noahlsm.F:568
ALBEDO = ALB; // module_sf_noahlsm.F:569
EMISSI = EMBRD; // module_sf_noahlsm.F:570
if (UA_PHYS) { FGSN = 0x0.0p+0f; } // module_sf_noahlsm.F:571
if (UA_PHYS) { FVB = 0x0.0p+0f; } // module_sf_noahlsm.F:572
if (UA_PHYS) { FBUR = 0x0.0p+0f; } // module_sf_noahlsm.F:573
} else { // module_sf_noahlsm.F:574
{  noah_snfrac(tables,SNEQV,SNUP,SALP,SNOWH,SNCOVR,XLAI,SHDFAC,FVB,GAMA,FBUR,FGSN,ZTOPV,ZBOTV,UA_PHYS); if(tables.error)return; } // module_sf_noahlsm.F:579
if (UA_PHYS) { // module_sf_noahlsm.F:583
if ((SFCTMP <= T1)) { // module_sf_noahlsm.F:584
RU = 0x0.0p+0f; // module_sf_noahlsm.F:585
} else { // module_sf_noahlsm.F:586
RU = MU(MU(MU(MU(0x1.9000000000000p+6f,SHDFAC),FGSN),noah_min(DV(SU(SFCTMP,T1),0x1.4000000000000p+2f),0x1.0000000000000p+0f)),SU(0x1.0000000000000p+0f,r_exp((-XLAI)))); // module_sf_noahlsm.F:587
} // module_sf_noahlsm.F:588
CH = DV(CH,AD(0x1.0000000000000p+0f,MU(RU,CH))); // module_sf_noahlsm.F:589
} // module_sf_noahlsm.F:590
SNCOVR = noah_min(SNCOVR,0x1.f5c2900000000p-1f); // module_sf_noahlsm.F:592
{  noah_alcalc(tables,ALB,SNOALB,EMBRD,SHDFAC,SHDMIN,SNCOVR,T1,ALBEDO,EMISSI,DT,SNOWNG,SNOTIME1,LVCOEF); if(tables.error)return; } // module_sf_noahlsm.F:594
} // module_sf_noahlsm.F:596
{  noah_tdfcnd(tables,DF1,SMC[(1)-(1)],QUARTZ,SMCMAX,SH2O[(1)-(1)],BEXP,PSISAT,SOILTYP,OPT_THCND); if(tables.error)return; } // module_sf_noahlsm.F:621
if ((VEGTYP == ISURBAN)) { DF1 = 0x1.9eb8520000000p+1f; } // module_sf_noahlsm.F:624
DF1 = MU(DF1,r_exp(MU(SBETA,SHDFAC))); // module_sf_noahlsm.F:626
if ((SNCOVR > 0x1.f0a3d80000000p-1f)) { // module_sf_noahlsm.F:631
DF1 = SNCOND; // module_sf_noahlsm.F:632
} // module_sf_noahlsm.F:633
DSOIL = (-MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm.F:641
if ((SNEQV == 0x0.0p+0f)) { // module_sf_noahlsm.F:642
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DSOIL); // module_sf_noahlsm.F:643
} else { // module_sf_noahlsm.F:644
DTOT = AD(SNOWH,DSOIL); // module_sf_noahlsm.F:645
FRCSNO = DV(SNOWH,DTOT); // module_sf_noahlsm.F:646
FRCSOI = DV(DSOIL,DTOT); // module_sf_noahlsm.F:650
DF1H = DV(MU(SNCOND,DF1),AD(MU(FRCSOI,SNCOND),MU(FRCSNO,DF1))); // module_sf_noahlsm.F:653
DF1A = AD(MU(FRCSNO,SNCOND),MU(FRCSOI,DF1)); // module_sf_noahlsm.F:660
DF1 = AD(MU(DF1A,SNCOVR),MU(DF1,SU(0x1.0000000000000p+0f,SNCOVR))); // module_sf_noahlsm.F:667
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm.F:668
} // module_sf_noahlsm.F:669
if ((SNCOVR > 0x0.0p+0f)) { // module_sf_noahlsm.F:674
{  noah_snowz0(tables,SNCOVR,Z0,Z0BRD,SNOWH,FBUR,FGSN,SHDMAX,UA_PHYS); if(tables.error)return; } // module_sf_noahlsm.F:675
} else { // module_sf_noahlsm.F:676
Z0 = Z0BRD; // module_sf_noahlsm.F:677
if (UA_PHYS) { {  noah_snowz0(tables,SNCOVR,Z0,Z0BRD,SNOWH,FBUR,FGSN,SHDMAX,UA_PHYS); if(tables.error)return; } } // module_sf_noahlsm.F:678
} // module_sf_noahlsm.F:680
FDOWN = AD(SOLNET,LWDN); // module_sf_noahlsm.F:720
T2V = MU(SFCTMP,AD(0x1.0000000000000p+0f,MU(0x1.3851ec0000000p-1f,Q2))); // module_sf_noahlsm.F:724
IOUT = 0; // module_sf_noahlsm.F:726
if ((IOUT == 1)) { // module_sf_noahlsm.F:727
// Native diagnostic formatting is handled by the host error receipt. // module_sf_noahlsm.F:728
// Native diagnostic formatting is handled by the host error receipt. // module_sf_noahlsm.F:729
} // module_sf_noahlsm.F:737
{  noah_penman(tables,SFCTMP,SFCPRS,CH,T2V,TH2,PRCP,FDOWN,T24,SSOIL,Q2,Q2SAT,ETP,RCH,EPSCA,RR,SNOWNG,FRZGRA,DQSDT2,FLX2,EMISSI,SNEQV,T1,SNCOVR,AOASIS,ALBEDO,SOLDN,FVB,GAMA,STC[(1)-(1)],ETPN,FLX4,UA_PHYS); if(tables.error)return; } // module_sf_noahlsm.F:739
if (((SHDFAC > 0x0.0p+0f) && (XLAI > 0x0.0p+0f))) { // module_sf_noahlsm.F:753
{  noah_canres(tables,SOLDN,CH,SFCTMP,Q2,SFCPRS,SH2O,ZSOIL,NSOIL,SMCWLT,SMCREF,RSMIN,RC,PC,NROOT,Q2SAT,DQSDT2,TOPT,RSMAX,RGL,HS,XLAI,RCS,RCT,RCQ,RCSOIL,EMISSI); if(tables.error)return; } // module_sf_noahlsm.F:754
} else { // module_sf_noahlsm.F:758
RC = 0x0.0p+0f; // module_sf_noahlsm.F:759
} // module_sf_noahlsm.F:760
ESNOW = 0x0.0p+0f; // module_sf_noahlsm.F:765
if ((SNEQV == 0x0.0p+0f)) { // module_sf_noahlsm.F:766
{  noah_nopac(tables,ETP,ETA,PRCP,SMC,SMCMAX,SMCWLT,SMCREF,SMCDRY,CMC,CMCMAX,NSOIL,DT,SHDFAC,SBETA,Q2,T1,SFCTMP,T24,TH2,FDOWN,F1,EMISSI,SSOIL,STC,EPSCA,BEXP,PC,RCH,RR,CFACTR,SH2O,SLOPE,KDT,FRZX,PSISAT,ZSOIL,DKSAT,DWSAT,TBOT,ZBOT,RUNOFF1,RUNOFF2,RUNOFF3,EDIR,EC,ET,ETT,NROOT,RTDIS,QUARTZ,FXEXP,CSOIL,BETA,DRIP,DEW,FLX1,FLX3,VEGTYP,ISURBAN,SFHEAD1RT,INFXS1RT,ETPND1,SOILTYP,OPT_THCND,XSDA_QFX,QFX_PHY,XQNORM,FASDAS,HCPCT_FASDAS,IRRIGATION_CHANNEL,present_IRRIGATION_CHANNEL); if(tables.error)return; } // module_sf_noahlsm.F:767
ETA_KINEMATIC = ETA; // module_sf_noahlsm.F:781
} else { // module_sf_noahlsm.F:782
{  noah_snopac(tables,ETP,ETA,PRCP,PRCPF,SNOWNG,SMC,SMCMAX,SMCWLT,SMCREF,SMCDRY,CMC,CMCMAX,NSOIL,DT,SBETA,DF1,Q2,T1,SFCTMP,T24,TH2,FDOWN,F1,SSOIL,STC,EPSCA,SFCPRS,BEXP,PC,RCH,RR,CFACTR,SNCOVR,SNEQV,SNDENS,SNOWH,SH2O,SLOPE,KDT,FRZX,PSISAT,ZSOIL,DWSAT,DKSAT,TBOT,ZBOT,SHDFAC,RUNOFF1,RUNOFF2,RUNOFF3,EDIR,EC,ET,ETT,NROOT,SNOMLT,RTDIS,QUARTZ,FXEXP,CSOIL,BETA,DRIP,DEW,FLX1,FLX2,FLX3,ESNOW,ETNS,EMISSI,RIBB,SOLDN,ISURBAN,VEGTYP,ETPN,FLX4,UA_PHYS,SFHEAD1RT,INFXS1RT,ETPND1,SOILTYP,OPT_THCND,QFX_PHY,FASDAS,HCPCT_FASDAS); if(tables.error)return; } // module_sf_noahlsm.F:783
ETA_KINEMATIC = SU(AD(ESNOW,ETNS),MU(0x1.f400000000000p+9f,DEW)); // module_sf_noahlsm.F:799
} // module_sf_noahlsm.F:800
Q1 = AD(Q2,DV(MU(ETA_KINEMATIC,CP),RCH)); // module_sf_noahlsm.F:805
SHEAT = MU(DV((-MU(MU(CH,CP),SFCPRS)),MU(R,T2V)),SU(TH2,T1)); // module_sf_noahlsm.F:811
if (UA_PHYS) { SHEAT = AD(SHEAT,FLX4); } // module_sf_noahlsm.F:812
if ((FASDAS == 1)) { // module_sf_noahlsm.F:816
HFX_PHY = SHEAT; // module_sf_noahlsm.F:817
} // module_sf_noahlsm.F:818
EDIR = MU(EDIR,LVH2O); // module_sf_noahlsm.F:825
EC = MU(EC,LVH2O); // module_sf_noahlsm.F:826
for (K=1; K<=4; K+=1) { // module_sf_noahlsm.F:827
ET[(K)-(1)] = MU(ET[(K)-(1)],LVH2O); // module_sf_noahlsm.F:828
} // module_sf_noahlsm.F:829
ETT = MU(ETT,LVH2O); // module_sf_noahlsm.F:830
ETPND1 = MU(ETPND1,LVH2O); // module_sf_noahlsm.F:832
ESNOW = MU(ESNOW,LSUBS); // module_sf_noahlsm.F:834
ETP = MU(ETP,AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),LVH2O),MU(SNCOVR,LSUBS))); // module_sf_noahlsm.F:835
if (UA_PHYS) { ETPN = MU(ETPN,AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),LVH2O),MU(SNCOVR,LSUBS))); } // module_sf_noahlsm.F:836
if ((ETP > 0x0.0p+0f)) { // module_sf_noahlsm.F:837
ETA = AD(AD(AD(EDIR,EC),ETT),ESNOW); // module_sf_noahlsm.F:838
} else { // module_sf_noahlsm.F:839
ETA = ETP; // module_sf_noahlsm.F:840
} // module_sf_noahlsm.F:841
if ((ETP == 0x0.0p+0f)) { // module_sf_noahlsm.F:845
BETA = 0x0.0p+0f; // module_sf_noahlsm.F:846
} else { // module_sf_noahlsm.F:847
BETA = DV(ETA,ETP); // module_sf_noahlsm.F:848
} // module_sf_noahlsm.F:849
SSOIL = MU(-0x1.0000000000000p+0f,SSOIL); // module_sf_noahlsm.F:856
RUNOFF3 = DV(RUNOFF3,DT); // module_sf_noahlsm.F:864
RUNOFF2 = AD(RUNOFF2,RUNOFF3); // module_sf_noahlsm.F:865
SOILM = MU(MU(-0x1.0000000000000p+0f,SMC[(1)-(1)]),ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:866
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:867
SOILM = AD(SOILM,MU(SMC[(K)-(1)],SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]))); // module_sf_noahlsm.F:868
} // module_sf_noahlsm.F:869
SOILWM = MU(MU(-0x1.0000000000000p+0f,SU(SMCMAX,SMCWLT)),ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:870
SOILWW = MU(MU(-0x1.0000000000000p+0f,SU(SMC[(1)-(1)],SMCWLT)),ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:871
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:873
SMAV[(K)-(1)] = DV(SU(SMC[(K)-(1)],SMCWLT),SU(SMCMAX,SMCWLT)); // module_sf_noahlsm.F:874
} // module_sf_noahlsm.F:875
if ((NROOT >= 2)) { // module_sf_noahlsm.F:877
for (K=2; K<=NROOT; K+=1) { // module_sf_noahlsm.F:878
SOILWM = AD(SOILWM,MU(SU(SMCMAX,SMCWLT),SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]))); // module_sf_noahlsm.F:879
SOILWW = AD(SOILWW,MU(SU(SMC[(K)-(1)],SMCWLT),SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]))); // module_sf_noahlsm.F:880
} // module_sf_noahlsm.F:881
} // module_sf_noahlsm.F:882
if ((SOILWM < 0x1.0c6f7a0000000p-20f)) { // module_sf_noahlsm.F:883
SOILWM = 0x0.0p+0f; // module_sf_noahlsm.F:884
SOILW = 0x0.0p+0f; // module_sf_noahlsm.F:885
SOILM = 0x0.0p+0f; // module_sf_noahlsm.F:886
} else { // module_sf_noahlsm.F:887
SOILW = DV(SOILWW,SOILWM); // module_sf_noahlsm.F:888
} // module_sf_noahlsm.F:889
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:895-1010
__device__ void noah_alcalc(NoahTables& tables, float& ALB, float& SNOALB, float& EMBRD, float& SHDFAC, float& SHDMIN, float& SNCOVR, float& TSNOW, float& ALBEDO, float& EMISSI, float& DT, bool& SNOWNG, float& SNOTIME1, float& LVCOEF){
auto& EMISSI_S=tables.EMISSI_S;
float SNOALB2;
float TM;
float SNOALB1;
float SNACCA = 0x1.e147ae0000000p-1f;
float SNACCB = 0x1.28f5c20000000p-1f;
float SNTHWA = 0x1.a3d70a0000000p-1f;
float SNTHWB = 0x1.d70a3e0000000p-2f;
 // module_sf_noahlsm.F:908
ALBEDO = AD(ALB,MU(SNCOVR,SU(SNOALB,ALB))); // module_sf_noahlsm.F:928
EMISSI = AD(EMBRD,MU(SNCOVR,SU(EMISSI_S,EMBRD))); // module_sf_noahlsm.F:929
SNOALB1 = AD(SNOALB,MU(LVCOEF,SU(0x1.b333340000000p-1f,SNOALB))); // module_sf_noahlsm.F:983
SNOALB2 = SNOALB1; // module_sf_noahlsm.F:984
if (SNOWNG) { // module_sf_noahlsm.F:986
SNOTIME1 = 0x0.0p+0f; // module_sf_noahlsm.F:987
} else { // module_sf_noahlsm.F:988
SNOTIME1 = AD(SNOTIME1,DT); // module_sf_noahlsm.F:989
SNOALB2 = MU(SNOALB1,r_pow(SNACCA,r_pow(DV(SNOTIME1,0x1.5180000000000p+16f),SNACCB))); // module_sf_noahlsm.F:991
} // module_sf_noahlsm.F:995
SNOALB2 = noah_max(SNOALB2,ALB); // module_sf_noahlsm.F:997
ALBEDO = AD(ALB,MU(SNCOVR,SU(SNOALB2,ALB))); // module_sf_noahlsm.F:998
if ((ALBEDO > SNOALB2)) { ALBEDO = SNOALB2; } // module_sf_noahlsm.F:999
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1013-1149
__device__ void noah_canres(NoahTables& tables, float& SOLAR, float& CH, float& SFCTMP, float& Q2, float& SFCPRS, float* SMC, float* ZSOIL, int& NSOIL, float& SMCWLT, float& SMCREF, float& RSMIN, float& RC, float& PC, int& NROOT, float& Q2SAT, float& DQSDT2, float& TOPT, float& RSMAX, float& RGL, float& HS, float& XLAI, float& RCS, float& RCT, float& RCQ, float& RCSOIL, float& EMISSI){
auto& RD=tables.RD;
auto& SIGMA=tables.SIGMA;
auto& CP=tables.CP;
int K;
float DELTA;
float FF;
float GX;
float P;
float RR;
float PART[20];
float SLV = 0x1.314c400000000p+21f;
 // module_sf_noahlsm.F:1054
RCS = 0x0.0p+0f; // module_sf_noahlsm.F:1070
RCT = 0x0.0p+0f; // module_sf_noahlsm.F:1071
RCQ = 0x0.0p+0f; // module_sf_noahlsm.F:1072
RCSOIL = 0x0.0p+0f; // module_sf_noahlsm.F:1073
RC = 0x0.0p+0f; // module_sf_noahlsm.F:1078
FF = DV(MU(0x1.19999a0000000p+0f,SOLAR),MU(RGL,XLAI)); // module_sf_noahlsm.F:1079
RCS = DV(AD(FF,DV(RSMIN,RSMAX)),AD(0x1.0000000000000p+0f,FF)); // module_sf_noahlsm.F:1080
RCS = noah_max(RCS,0x1.a36e2e0000000p-14f); // module_sf_noahlsm.F:1086
RCT = SU(0x1.0000000000000p+0f,MU(0x1.a36e2e0000000p-10f,MU(SU(TOPT,SFCTMP),SU(TOPT,SFCTMP)))); // module_sf_noahlsm.F:1087; strict GNU folds real exponent 2 into multiplication
RCT = noah_max(RCT,0x1.a36e2e0000000p-14f); // module_sf_noahlsm.F:1093
RCQ = DV(0x1.0000000000000p+0f,AD(0x1.0000000000000p+0f,MU(HS,SU(Q2SAT,Q2)))); // module_sf_noahlsm.F:1094
RCQ = noah_max(RCQ,0x1.47ae140000000p-7f); // module_sf_noahlsm.F:1100
GX = DV(SU(SMC[(1)-(1)],SMCWLT),SU(SMCREF,SMCWLT)); // module_sf_noahlsm.F:1101
if ((GX > 0x1.0000000000000p+0f)) { GX = 0x1.0000000000000p+0f; } // module_sf_noahlsm.F:1102
if ((GX < 0x0.0p+0f)) { GX = 0x0.0p+0f; } // module_sf_noahlsm.F:1103
PART[(1)-(1)] = MU(DV(ZSOIL[(1)-(1)],ZSOIL[(NROOT)-(1)]),GX); // module_sf_noahlsm.F:1112
for (K=2; K<=NROOT; K+=1) { // module_sf_noahlsm.F:1113
GX = DV(SU(SMC[(K)-(1)],SMCWLT),SU(SMCREF,SMCWLT)); // module_sf_noahlsm.F:1114
if ((GX > 0x1.0000000000000p+0f)) { GX = 0x1.0000000000000p+0f; } // module_sf_noahlsm.F:1115
if ((GX < 0x0.0p+0f)) { GX = 0x0.0p+0f; } // module_sf_noahlsm.F:1116
PART[(K)-(1)] = MU(DV(SU(ZSOIL[(K)-(1)],ZSOIL[((K - 1))-(1)]),ZSOIL[(NROOT)-(1)]),GX); // module_sf_noahlsm.F:1124
} // module_sf_noahlsm.F:1125
for (K=1; K<=NROOT; K+=1) { // module_sf_noahlsm.F:1126
RCSOIL = AD(RCSOIL,PART[(K)-(1)]); // module_sf_noahlsm.F:1127
} // module_sf_noahlsm.F:1128
RCSOIL = noah_max(RCSOIL,0x1.a36e2e0000000p-14f); // module_sf_noahlsm.F:1137
RC = DV(RSMIN,MU(MU(MU(MU(XLAI,RCS),RCT),RCQ),RCSOIL)); // module_sf_noahlsm.F:1139
RR = AD(DV(MU(DV(MU(MU(MU(0x1.0000000000000p+2f,EMISSI),SIGMA),RD),CP),r_pow(SFCTMP,0x1.0000000000000p+2f)),MU(SFCPRS,CH)),0x1.0000000000000p+0f); // module_sf_noahlsm.F:1141
DELTA = MU(DV(SLV,CP),DQSDT2); // module_sf_noahlsm.F:1144
PC = DV(AD(RR,DELTA),AD(MU(RR,AD(0x1.0000000000000p+0f,MU(RC,CH))),DELTA)); // module_sf_noahlsm.F:1146
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1152-1191
__device__ void noah_csnow(NoahTables& tables, float& SNCOND, float& DSNOW){
float C;
float UNIT = 0x1.dc67e00000000p-4f;
 // module_sf_noahlsm.F:1160
C = MU(0x1.4fdf3c0000000p-2f,r_pow(10,MU(0x1.2000000000000p+1f,DSNOW))); // module_sf_noahlsm.F:1171
SNCOND = MU(MU(0x1.0000000000000p+1f,UNIT),C); // module_sf_noahlsm.F:1188
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1193-1232
__device__ void noah_devap(NoahTables& tables, float& EDIR, float& ETP1, float& SMC, float& ZSOIL, float& SHDFAC, float& SMCMAX, float& BEXP, float& DKSAT, float& DWSAT, float& SMCDRY, float& SMCREF, float& SMCWLT, float& FXEXP){
float FX;
float SRATIO;
 // module_sf_noahlsm.F:1202
SRATIO = DV(SU(SMC,SMCDRY),SU(SMCMAX,SMCDRY)); // module_sf_noahlsm.F:1218
if ((SRATIO > 0x0.0p+0f)) { // module_sf_noahlsm.F:1219
FX = r_pow(SRATIO,FXEXP); // module_sf_noahlsm.F:1220
FX = noah_max(noah_min(FX,0x1.0000000000000p+0f),0x0.0p+0f); // module_sf_noahlsm.F:1221
} else { // module_sf_noahlsm.F:1222
FX = 0x0.0p+0f; // module_sf_noahlsm.F:1223
} // module_sf_noahlsm.F:1224
EDIR = MU(MU(FX,SU(0x1.0000000000000p+0f,SHDFAC)),ETP1); // module_sf_noahlsm.F:1229
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1234-1324
__device__ void noah_devap_hydro(NoahTables& tables, float& EDIR, float& ETP1, float& SMC, float& ZSOIL, float& SHDFAC, float& SMCMAX, float& BEXP, float& DKSAT, float& DWSAT, float& SMCDRY, float& SMCREF, float& SMCWLT, float& FXEXP, float& SFHEAD1RT, float& ETPND1, float& DT){
float FX;
float SRATIO;
float EDIRTMP;
 // module_sf_noahlsm.F:1244
SRATIO = DV(SU(SMC,SMCDRY),SU(SMCMAX,SMCDRY)); // module_sf_noahlsm.F:1265
if ((SRATIO > 0x0.0p+0f)) { // module_sf_noahlsm.F:1266
FX = r_pow(SRATIO,FXEXP); // module_sf_noahlsm.F:1267
FX = noah_max(noah_min(FX,0x1.0000000000000p+0f),0x0.0p+0f); // module_sf_noahlsm.F:1268
} else { // module_sf_noahlsm.F:1269
FX = 0x0.0p+0f; // module_sf_noahlsm.F:1270
} // module_sf_noahlsm.F:1271
EDIRTMP = 0x0.0p+0f; // module_sf_noahlsm.F:1274
ETPND1 = 0x0.0p+0f; // module_sf_noahlsm.F:1275
EDIRTMP = MU(EDIRTMP,DT); // module_sf_noahlsm.F:1284
SFHEAD1RT = MU(SFHEAD1RT,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:1287
if ((EDIRTMP > 0x0.0p+0f)) { // module_sf_noahlsm.F:1292
if ((EDIRTMP > SFHEAD1RT)) { // module_sf_noahlsm.F:1293
ETPND1 = SFHEAD1RT; // module_sf_noahlsm.F:1294
SFHEAD1RT = 0x0.0p+0f; // module_sf_noahlsm.F:1295
EDIRTMP = SU(EDIRTMP,ETPND1); // module_sf_noahlsm.F:1296
} else { // module_sf_noahlsm.F:1297
ETPND1 = EDIRTMP; // module_sf_noahlsm.F:1298
EDIRTMP = 0x0.0p+0f; // module_sf_noahlsm.F:1299
SFHEAD1RT = SU(SFHEAD1RT,ETPND1); // module_sf_noahlsm.F:1300
} // module_sf_noahlsm.F:1301
} // module_sf_noahlsm.F:1302
if ((SFHEAD1RT != 0x0.0p+0f)) { SFHEAD1RT = MU(SFHEAD1RT,0x1.f400000000000p+9f); } // module_sf_noahlsm.F:1305
ETPND1 = DV(ETPND1,DT); // module_sf_noahlsm.F:1308
EDIRTMP = DV(EDIRTMP,DT); // module_sf_noahlsm.F:1309
EDIR = MU(FX,EDIRTMP); // module_sf_noahlsm.F:1318
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1327-1425
__device__ void noah_evapo(NoahTables& tables, float& ETA1, float* SMC, int& NSOIL, float& CMC, float& ETP1, float& DT, float* ZSOIL, float* SH2O, float& SMCMAX, float& BEXP, float& PC, float& SMCWLT, float& DKSAT, float& DWSAT, float& SMCREF, float& SHDFAC, float& CMCMAX, float& SMCDRY, float& CFACTR, float& EDIR, float& EC, float* ET, float& ETT, float& SFCTMP, float& Q2, int& NROOT, float* RTDIS, float& FXEXP, float& SFHEAD1RT, float& ETPND1){
int I;
int K;
float CMC2MS;
 // module_sf_noahlsm.F:1344
EDIR = 0x0.0p+0f; // module_sf_noahlsm.F:1361
EC = 0x0.0p+0f; // module_sf_noahlsm.F:1362
ETT = 0x0.0p+0f; // module_sf_noahlsm.F:1363
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1364
ET[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1365
} // module_sf_noahlsm.F:1366
if ((ETP1 > 0x0.0p+0f)) { // module_sf_noahlsm.F:1373
if ((SHDFAC < 0x1.0000000000000p+0f)) { // module_sf_noahlsm.F:1374
{  noah_devap(tables,EDIR,ETP1,SMC[(1)-(1)],ZSOIL[(1)-(1)],SHDFAC,SMCMAX,BEXP,DKSAT,DWSAT,SMCDRY,SMCREF,SMCWLT,FXEXP); if(tables.error)return; } // module_sf_noahlsm.F:1387
} // module_sf_noahlsm.F:1390
if ((SHDFAC > 0x0.0p+0f)) { // module_sf_noahlsm.F:1396
{  noah_transp(tables,ET,NSOIL,ETP1,SH2O,CMC,ZSOIL,SHDFAC,SMCWLT,CMCMAX,PC,CFACTR,SMCREF,SFCTMP,Q2,NROOT,RTDIS); if(tables.error)return; } // module_sf_noahlsm.F:1397
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1399
ETT = AD(ETT,ET[(K)-(1)]); // module_sf_noahlsm.F:1400
} // module_sf_noahlsm.F:1401
if ((CMC > 0x0.0p+0f)) { // module_sf_noahlsm.F:1406
EC = MU(MU(SHDFAC,r_pow(DV(CMC,CMCMAX),CFACTR)),ETP1); // module_sf_noahlsm.F:1407
} else { // module_sf_noahlsm.F:1408
EC = 0x0.0p+0f; // module_sf_noahlsm.F:1409
} // module_sf_noahlsm.F:1410
CMC2MS = DV(CMC,DT); // module_sf_noahlsm.F:1415
EC = noah_min(CMC2MS,EC); // module_sf_noahlsm.F:1416
} // module_sf_noahlsm.F:1417
} // module_sf_noahlsm.F:1418
ETA1 = AD(AD(EDIR,ETT),EC); // module_sf_noahlsm.F:1422
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1428-1448
__device__ void noah_fac2mit(NoahTables& tables, float& SMCMAX, float& FLIMIT){
 // module_sf_noahlsm.F:1429
FLIMIT = 0x1.cccccc0000000p-1f; // module_sf_noahlsm.F:1433
if ((SMCMAX == 0x1.947ae20000000p-2f)) { // module_sf_noahlsm.F:1435
FLIMIT = 0x1.2e147a0000000p-1f; // module_sf_noahlsm.F:1436
} else if (((SMCMAX == 0x1.bc6a7e0000000p-2f) || (SMCMAX == 0x1.9db22e0000000p-2f))) { // module_sf_noahlsm.F:1437
FLIMIT = 0x1.b333340000000p-1f; // module_sf_noahlsm.F:1438
} else if (((SMCMAX == 0x1.dc28f60000000p-2f) || (SMCMAX == 0x1.9fbe760000000p-2f))) { // module_sf_noahlsm.F:1439
FLIMIT = 0x1.b851ec0000000p-1f; // module_sf_noahlsm.F:1440
} else if (((SMCMAX == 0x1.e76c8c0000000p-2f) || (SMCMAX == 0x1.c189380000000p-2f))) { // module_sf_noahlsm.F:1441
FLIMIT = 0x1.7ae1480000000p-1f; // module_sf_noahlsm.F:1442
} else if (((SMCMAX == 0x1.99999a0000000p-3f) || (SMCMAX == 0x1.db22d00000000p-2f))) { // module_sf_noahlsm.F:1443
FLIMIT = 0x1.99999a0000000p-1f; // module_sf_noahlsm.F:1444
} // module_sf_noahlsm.F:1445
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1451-1589
__device__ void noah_frh2o(NoahTables& tables, float& FREE, float& TKELV, float& SMC, float& SH2O, float& SMCMAX, float& BEXP, float& PSIS){
float BX;
float DENOM;
float DF;
float DSWL;
float FK;
float SWL;
float SWLK;
int NLOG;
int KCOUNT;
float CK = 0x1.0000000000000p+3f;
float BLIM = 0x1.6000000000000p+2f;
float ERROR = 0x1.47ae140000000p-8f;
float HLICE = 0x1.45af000000000p+18f;
float GS = 0x1.39eb860000000p+3f;
float DICE = 0x1.cc00000000000p+9f;
float DH2O = 0x1.f400000000000p+9f;
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm.F:1482
BX = BEXP; // module_sf_noahlsm.F:1497
if ((BEXP > BLIM)) { BX = BLIM; } // module_sf_noahlsm.F:1502
NLOG = 0; // module_sf_noahlsm.F:1503
KCOUNT = 0; // module_sf_noahlsm.F:1508
if ((TKELV > SU(T0,0x1.0624de0000000p-10f))) { // module_sf_noahlsm.F:1510
FREE = SMC; // module_sf_noahlsm.F:1511
} else { // module_sf_noahlsm.F:1512
if ((CK != 0x0.0p+0f)) { // module_sf_noahlsm.F:1520
SWL = SU(SMC,SH2O); // module_sf_noahlsm.F:1521
if ((SWL > SU(SMC,0x1.47ae140000000p-6f))) { SWL = SU(SMC,0x1.47ae140000000p-6f); } // module_sf_noahlsm.F:1525
if ((SWL < 0x0.0p+0f)) { SWL = 0x0.0p+0f; } // module_sf_noahlsm.F:1530
label_1001:; ; // module_sf_noahlsm.F:1531
if ((!((NLOG < 10) && (KCOUNT == 0)))) { goto label_1002; } // module_sf_noahlsm.F:1532
NLOG = (NLOG + 1); // module_sf_noahlsm.F:1533
DF = SU(r_log(MU(MU(DV(MU(PSIS,GS),HLICE),MU(AD(0x1.0000000000000p+0f,MU(CK,SWL)),AD(0x1.0000000000000p+0f,MU(CK,SWL)))),r_pow(DV(SMCMAX,SU(SMC,SWL)),BX))),r_log(DV((-SU(TKELV,T0)),TKELV))); // module_sf_noahlsm.F:1534; strict GNU folds real exponent 2 into multiplication
DENOM = AD(DV(MU(0x1.0000000000000p+1f,CK),AD(0x1.0000000000000p+0f,MU(CK,SWL))),DV(BX,SU(SMC,SWL))); // module_sf_noahlsm.F:1537
SWLK = SU(SWL,DV(DF,DENOM)); // module_sf_noahlsm.F:1538
if ((SWLK > SU(SMC,0x1.47ae140000000p-6f))) { SWLK = SU(SMC,0x1.47ae140000000p-6f); } // module_sf_noahlsm.F:1542
if ((SWLK < 0x0.0p+0f)) { SWLK = 0x0.0p+0f; } // module_sf_noahlsm.F:1543
DSWL = noah_abs(SU(SWLK,SWL)); // module_sf_noahlsm.F:1548
SWL = SWLK; // module_sf_noahlsm.F:1554
if ((DSWL <= ERROR)) { // module_sf_noahlsm.F:1555
KCOUNT = (KCOUNT + 1); // module_sf_noahlsm.F:1556
} // module_sf_noahlsm.F:1557
goto label_1001; // module_sf_noahlsm.F:1564
label_1002:; ; // module_sf_noahlsm.F:1565
FREE = SU(SMC,SWL); // module_sf_noahlsm.F:1566
} // module_sf_noahlsm.F:1567
if ((KCOUNT == 0)) { // module_sf_noahlsm.F:1576
// Native diagnostic formatting is handled by the host error receipt. // module_sf_noahlsm.F:1577
FK = MU(r_pow(MU(DV(HLICE,MU(GS,(-PSIS))),DV(SU(TKELV,T0),TKELV)),DV(-1,BX)),SMCMAX); // module_sf_noahlsm.F:1578
if ((FK < 0x1.47ae140000000p-6f)) { FK = 0x1.47ae140000000p-6f; } // module_sf_noahlsm.F:1581
FREE = noah_min(FK,SMC); // module_sf_noahlsm.F:1582
} // module_sf_noahlsm.F:1586
} // module_sf_noahlsm.F:1587
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1592-1854
__device__ void noah_hrt(NoahTables& tables, float* RHSTS, float* STC, float* SMC, float& SMCMAX, int& NSOIL, float* ZSOIL, float& YY, float& ZZ1, float& TBOT, float& ZBOT, float& PSISAT, float* SH2O, float& DT, float& BEXP, int& SOILTYP, int& OPT_THCND, float& F1, float& DF1, float& QUARTZ, float& CSOIL, float* AI, float* BI, float* CI, int& VEGTYP, int& ISURBAN, float& HCPCT_FASDAS){
bool ITAVG;
int I;
int K;
float DDZ;
float DDZ2;
float DENOM;
float DF1N;
float DF1K;
float DTSDZ;
float DTSDZ2;
float HCPCT;
float QTOT;
float SSOIL;
float SICE;
float TAVG;
float TBK;
float TBK1;
float TSNSR;
float TSURF;
float CSOIL_LOC;
float T0 = 0x1.1126660000000p+8f;
float CAIR = 0x1.f600000000000p+9f;
float CICE = 0x1.0114800000000p+21f;
float CH2O = 0x1.0059000000000p+22f;
 // module_sf_noahlsm.F:1604
if ((VEGTYP == ISURBAN)) { // module_sf_noahlsm.F:1632
CSOIL_LOC = 0x1.6e36000000000p+21f; // module_sf_noahlsm.F:1633
} else { // module_sf_noahlsm.F:1634
CSOIL_LOC = CSOIL; // module_sf_noahlsm.F:1635
} // module_sf_noahlsm.F:1636
ITAVG = true; // module_sf_noahlsm.F:1641
HCPCT = AD(AD(AD(MU(SH2O[(1)-(1)],CH2O),MU(SU(0x1.0000000000000p+0f,SMCMAX),CSOIL_LOC)),MU(SU(SMCMAX,SMC[(1)-(1)]),CAIR)),MU(SU(SMC[(1)-(1)],SH2O[(1)-(1)]),CICE)); // module_sf_noahlsm.F:1647
HCPCT_FASDAS = HCPCT; // module_sf_noahlsm.F:1653
DDZ = DV(0x1.0000000000000p+0f,MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm.F:1660
AI[(1)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1661
CI[(1)-(1)] = DV(MU(DF1,DDZ),MU(ZSOIL[(1)-(1)],HCPCT)); // module_sf_noahlsm.F:1662
BI[(1)-(1)] = AD((-CI[(1)-(1)]),DV(DF1,MU(MU(MU(MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),ZSOIL[(1)-(1)]),HCPCT),ZZ1))); // module_sf_noahlsm.F:1670
DTSDZ = DV(SU(STC[(1)-(1)],STC[(2)-(1)]),MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm.F:1672
SSOIL = DV(MU(DF1,SU(STC[(1)-(1)],YY)),MU(MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),ZZ1)); // module_sf_noahlsm.F:1673
DENOM = MU(ZSOIL[(1)-(1)],HCPCT); // module_sf_noahlsm.F:1675
RHSTS[(1)-(1)] = DV(SU(MU(DF1,DTSDZ),SSOIL),DENOM); // module_sf_noahlsm.F:1683
QTOT = MU(MU(-0x1.0000000000000p+0f,RHSTS[(1)-(1)]),DENOM); // module_sf_noahlsm.F:1688
SICE = SU(SMC[(1)-(1)],SH2O[(1)-(1)]); // module_sf_noahlsm.F:1700
if (ITAVG) { // module_sf_noahlsm.F:1701
TSURF = DV(AD(YY,MU(SU(ZZ1,1),STC[(1)-(1)])),ZZ1); // module_sf_noahlsm.F:1702
{ int tmp0_4=1; noah_tbnd(tables,STC[(1)-(1)],STC[(2)-(1)],ZSOIL,ZBOT,tmp0_4,NSOIL,TBK); if(tables.error)return; } // module_sf_noahlsm.F:1709
if (((((SICE > 0x0.0p+0f) || (STC[(1)-(1)] < T0)) || (TSURF < T0)) || (TBK < T0))) { // module_sf_noahlsm.F:1710
{ int tmp1_6=1; noah_tmpavg(tables,TAVG,TSURF,STC[(1)-(1)],TBK,ZSOIL,NSOIL,tmp1_6); if(tables.error)return; } // module_sf_noahlsm.F:1713
{ int tmp2_10=1; noah_snksrc(tables,TSNSR,TAVG,SMC[(1)-(1)],SH2O[(1)-(1)],ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,tmp2_10,QTOT); if(tables.error)return; } // module_sf_noahlsm.F:1714
RHSTS[(1)-(1)] = SU(RHSTS[(1)-(1)],DV(TSNSR,DENOM)); // module_sf_noahlsm.F:1717
} // module_sf_noahlsm.F:1718
} else { // module_sf_noahlsm.F:1719
if (((SICE > 0x0.0p+0f) || (STC[(1)-(1)] < T0))) { // module_sf_noahlsm.F:1721
{ int tmp3_10=1; noah_snksrc(tables,TSNSR,STC[(1)-(1)],SMC[(1)-(1)],SH2O[(1)-(1)],ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,tmp3_10,QTOT); if(tables.error)return; } // module_sf_noahlsm.F:1722
RHSTS[(1)-(1)] = SU(RHSTS[(1)-(1)],DV(TSNSR,DENOM)); // module_sf_noahlsm.F:1725
} // module_sf_noahlsm.F:1726
} // module_sf_noahlsm.F:1730
DDZ2 = 0x0.0p+0f; // module_sf_noahlsm.F:1735
DF1K = DF1; // module_sf_noahlsm.F:1736
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1744
HCPCT = AD(AD(AD(MU(SH2O[(K)-(1)],CH2O),MU(SU(0x1.0000000000000p+0f,SMCMAX),CSOIL_LOC)),MU(SU(SMCMAX,SMC[(K)-(1)]),CAIR)),MU(SU(SMC[(K)-(1)],SH2O[(K)-(1)]),CICE)); // module_sf_noahlsm.F:1745
if ((K != NSOIL)) { // module_sf_noahlsm.F:1752
{  noah_tdfcnd(tables,DF1N,SMC[(K)-(1)],QUARTZ,SMCMAX,SH2O[(K)-(1)],BEXP,PSISAT,SOILTYP,OPT_THCND); if(tables.error)return; } // module_sf_noahlsm.F:1757
if ((VEGTYP == ISURBAN)) { DF1N = 0x1.9eb8520000000p+1f; } // module_sf_noahlsm.F:1760
DENOM = MU(0x1.0000000000000p-1f,SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)])); // module_sf_noahlsm.F:1762
DTSDZ2 = DV(SU(STC[(K)-(1)],STC[((K + 1))-(1)]),DENOM); // module_sf_noahlsm.F:1767
DDZ2 = DV(0x1.0000000000000p+1f,SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)])); // module_sf_noahlsm.F:1768
CI[(K)-(1)] = DV(MU((-DF1N),DDZ2),MU(SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]),HCPCT)); // module_sf_noahlsm.F:1774
if (ITAVG) { // module_sf_noahlsm.F:1776
{  noah_tbnd(tables,STC[(K)-(1)],STC[((K + 1))-(1)],ZSOIL,ZBOT,K,NSOIL,TBK1); if(tables.error)return; } // module_sf_noahlsm.F:1777
} // module_sf_noahlsm.F:1778
} else { // module_sf_noahlsm.F:1780
{  noah_tdfcnd(tables,DF1N,SMC[(K)-(1)],QUARTZ,SMCMAX,SH2O[(K)-(1)],BEXP,PSISAT,SOILTYP,OPT_THCND); if(tables.error)return; } // module_sf_noahlsm.F:1789
if ((VEGTYP == ISURBAN)) { DF1N = 0x1.9eb8520000000p+1f; } // module_sf_noahlsm.F:1793
DENOM = SU(MU(0x1.0000000000000p-1f,AD(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)])),ZBOT); // module_sf_noahlsm.F:1795
DTSDZ2 = DV(SU(STC[(K)-(1)],TBOT),DENOM); // module_sf_noahlsm.F:1800
CI[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1806
if (ITAVG) { // module_sf_noahlsm.F:1807
{  noah_tbnd(tables,STC[(K)-(1)],TBOT,ZSOIL,ZBOT,K,NSOIL,TBK1); if(tables.error)return; } // module_sf_noahlsm.F:1808
} // module_sf_noahlsm.F:1809
} // module_sf_noahlsm.F:1812
DENOM = MU(SU(ZSOIL[(K)-(1)],ZSOIL[((K - 1))-(1)]),HCPCT); // module_sf_noahlsm.F:1816
RHSTS[(K)-(1)] = DV(SU(MU(DF1N,DTSDZ2),MU(DF1K,DTSDZ)),DENOM); // module_sf_noahlsm.F:1817
QTOT = MU(MU(-0x1.0000000000000p+0f,DENOM),RHSTS[(K)-(1)]); // module_sf_noahlsm.F:1818
SICE = SU(SMC[(K)-(1)],SH2O[(K)-(1)]); // module_sf_noahlsm.F:1820
if (ITAVG) { // module_sf_noahlsm.F:1821
{  noah_tmpavg(tables,TAVG,TBK,STC[(K)-(1)],TBK1,ZSOIL,NSOIL,K); if(tables.error)return; } // module_sf_noahlsm.F:1822
if (((((SICE > 0x0.0p+0f) || (STC[(K)-(1)] < T0)) || (TBK < T0)) || (TBK1 < T0))) { // module_sf_noahlsm.F:1824
{  noah_snksrc(tables,TSNSR,TAVG,SMC[(K)-(1)],SH2O[(K)-(1)],ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,K,QTOT); if(tables.error)return; } // module_sf_noahlsm.F:1826
RHSTS[(K)-(1)] = SU(RHSTS[(K)-(1)],DV(TSNSR,DENOM)); // module_sf_noahlsm.F:1828
} // module_sf_noahlsm.F:1829
} else { // module_sf_noahlsm.F:1830
if (((SICE > 0x0.0p+0f) || (STC[(K)-(1)] < T0))) { // module_sf_noahlsm.F:1832
{  noah_snksrc(tables,TSNSR,STC[(K)-(1)],SMC[(K)-(1)],SH2O[(K)-(1)],ZSOIL,NSOIL,SMCMAX,PSISAT,BEXP,DT,K,QTOT); if(tables.error)return; } // module_sf_noahlsm.F:1833
RHSTS[(K)-(1)] = SU(RHSTS[(K)-(1)],DV(TSNSR,DENOM)); // module_sf_noahlsm.F:1835
} // module_sf_noahlsm.F:1836
} // module_sf_noahlsm.F:1837
AI[(K)-(1)] = DV(MU((-DF1K),DDZ),MU(SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]),HCPCT)); // module_sf_noahlsm.F:1842
BI[(K)-(1)] = (-AD(AI[(K)-(1)],CI[(K)-(1)])); // module_sf_noahlsm.F:1847
TBK = TBK1; // module_sf_noahlsm.F:1848
DF1K = DF1N; // module_sf_noahlsm.F:1849
DTSDZ = DTSDZ2; // module_sf_noahlsm.F:1850
DDZ = DDZ2; // module_sf_noahlsm.F:1851
} // module_sf_noahlsm.F:1852
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1857-1905
__device__ void noah_hstep(NoahTables& tables, float* STCOUT, float* STCIN, float* RHSTS, float& DT, int& NSOIL, float* AI, float* BI, float* CI){
int K;
float RHSTSIN[20];
float CIIN[20];
 // module_sf_noahlsm.F:1864
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1879
RHSTS[(K)-(1)] = MU(RHSTS[(K)-(1)],DT); // module_sf_noahlsm.F:1880
AI[(K)-(1)] = MU(AI[(K)-(1)],DT); // module_sf_noahlsm.F:1881
BI[(K)-(1)] = AD(0x1.0000000000000p+0f,MU(BI[(K)-(1)],DT)); // module_sf_noahlsm.F:1882
CI[(K)-(1)] = MU(CI[(K)-(1)],DT); // module_sf_noahlsm.F:1883
} // module_sf_noahlsm.F:1884
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1888
RHSTSIN[(K)-(1)] = RHSTS[(K)-(1)]; // module_sf_noahlsm.F:1889
} // module_sf_noahlsm.F:1890
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1891
CIIN[(K)-(1)] = CI[(K)-(1)]; // module_sf_noahlsm.F:1892
} // module_sf_noahlsm.F:1893
{  noah_rosr12(tables,CI,AI,BI,CIIN,RHSTSIN,RHSTS,NSOIL); if(tables.error)return; } // module_sf_noahlsm.F:1897
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1901
STCOUT[(K)-(1)] = AD(STCIN[(K)-(1)],CI[(K)-(1)]); // module_sf_noahlsm.F:1902
} // module_sf_noahlsm.F:1903
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:1908-2196
__device__ void noah_nopac(NoahTables& tables, float& ETP, float& ETA, float& PRCP, float* SMC, float& SMCMAX, float& SMCWLT, float& SMCREF, float& SMCDRY, float& CMC, float& CMCMAX, int& NSOIL, float& DT, float& SHDFAC, float& SBETA, float& Q2, float& T1, float& SFCTMP, float& T24, float& TH2, float& FDOWN, float& F1, float& EMISSI, float& SSOIL, float* STC, float& EPSCA, float& BEXP, float& PC, float& RCH, float& RR, float& CFACTR, float* SH2O, float& SLOPE, float& KDT, float& FRZFACT, float& PSISAT, float* ZSOIL, float& DKSAT, float& DWSAT, float& TBOT, float& ZBOT, float& RUNOFF1, float& RUNOFF2, float& RUNOFF3, float& EDIR, float& EC, float* ET, float& ETT, int& NROOT, float* RTDIS, float& QUARTZ, float& FXEXP, float& CSOIL, float& BETA, float& DRIP, float& DEW, float& FLX1, float& FLX3, int& VEGTYP, int& ISURBAN, float& SFHEAD1RT, float& INFXS1RT, float& ETPND1, int& SOILTYP, int& OPT_THCND, float& XSDA_QFX, float& QFX_PHY, float& XQNORM, int& FASDAS, float& HCPCT_FASDAS, float& IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL){
auto& SIGMA=tables.SIGMA;
auto& CPH2O=tables.CPH2O;
int K;
float ET1[20];
float EC1;
float EDIR1;
float ETT1;
float DF1;
float ETA1;
float ETP1;
float PRCP1;
float YY;
float YYNUM;
float ZZ1;
float EFT[20];
float WETTY[20];
float EFDIR;
float EFC;
float EALL_NOW;
 // module_sf_noahlsm.F:1929
PRCP1 = MU(PRCP,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:1969
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:1970
DEW = 0x0.0p+0f; // module_sf_noahlsm.F:1971
QFX_PHY = 0x0.0p+0f; // module_sf_noahlsm.F:1978
EDIR = 0x0.0p+0f; // module_sf_noahlsm.F:1982
EDIR1 = 0x0.0p+0f; // module_sf_noahlsm.F:1983
EC1 = 0x0.0p+0f; // module_sf_noahlsm.F:1984
EC = 0x0.0p+0f; // module_sf_noahlsm.F:1985
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:1986
ET[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1987
ET1[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:1988
WETTY[(K)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:1992
} // module_sf_noahlsm.F:1996
ETT = 0x0.0p+0f; // module_sf_noahlsm.F:1997
ETT1 = 0x0.0p+0f; // module_sf_noahlsm.F:1998
ETPND1 = 0x0.0p+0f; // module_sf_noahlsm.F:2001
if ((ETP > 0x0.0p+0f)) { // module_sf_noahlsm.F:2004
{  noah_evapo(tables,ETA1,SMC,NSOIL,CMC,ETP1,DT,ZSOIL,SH2O,SMCMAX,BEXP,PC,SMCWLT,DKSAT,DWSAT,SMCREF,SHDFAC,CMCMAX,SMCDRY,CFACTR,EDIR1,EC1,ET1,ETT1,SFCTMP,Q2,NROOT,RTDIS,FXEXP,SFHEAD1RT,ETPND1); if(tables.error)return; } // module_sf_noahlsm.F:2005
if ((FASDAS == 1)) { // module_sf_noahlsm.F:2015
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2016
QFX_PHY = AD(QFX_PHY,ET1[(K)-(1)]); // module_sf_noahlsm.F:2017
if (((SMC[(K)-(1)] >= SMCREF) && (XSDA_QFX > 0x0.0p+0f))) { WETTY[(K)-(1)] = 0x0.0p+0f; } // module_sf_noahlsm.F:2019
} // module_sf_noahlsm.F:2020
QFX_PHY = AD(AD(EDIR1,EC1),QFX_PHY); // module_sf_noahlsm.F:2021
EALL_NOW = QFX_PHY; // module_sf_noahlsm.F:2022
QFX_PHY = MU(QFX_PHY,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2023
if ((EALL_NOW != 0x0.0p+0f)) { // module_sf_noahlsm.F:2025
EFDIR = MU(MU(MU(DV(EDIR1,EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2026
EFDIR = MU(EFDIR,WETTY[(1)-(1)]); // module_sf_noahlsm.F:2027
EDIR1 = AD(EDIR1,EFDIR); // module_sf_noahlsm.F:2029
EFC = MU(MU(MU(DV(EC1,EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2031
EC1 = AD(EC1,EFC); // module_sf_noahlsm.F:2033
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2036
EFT[(K)-(1)] = MU(MU(MU(DV(ET1[(K)-(1)],EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2037
EFT[(K)-(1)] = MU(EFT[(K)-(1)],WETTY[(K)-(1)]); // module_sf_noahlsm.F:2038
ET1[(K)-(1)] = AD(ET1[(K)-(1)],EFT[(K)-(1)]); // module_sf_noahlsm.F:2040
} // module_sf_noahlsm.F:2041
} // module_sf_noahlsm.F:2044
} else { // module_sf_noahlsm.F:2045
QFX_PHY = 0x0.0p+0f; // module_sf_noahlsm.F:2046
} // module_sf_noahlsm.F:2047
{  noah_smflx(tables,SMC,NSOIL,CMC,DT,PRCP1,ZSOIL,SH2O,SLOPE,KDT,FRZFACT,SMCMAX,BEXP,SMCWLT,DKSAT,DWSAT,SHDFAC,CMCMAX,RUNOFF1,RUNOFF2,RUNOFF3,EDIR1,EC1,ET1,DRIP,SFHEAD1RT,INFXS1RT,IRRIGATION_CHANNEL,present_IRRIGATION_CHANNEL); if(tables.error)return; } // module_sf_noahlsm.F:2051
ETA = MU(ETA1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2063
} else { // module_sf_noahlsm.F:2069
DEW = (-ETP1); // module_sf_noahlsm.F:2070
PRCP1 = AD(PRCP1,DEW); // module_sf_noahlsm.F:2076
if ((FASDAS == 1)) { // module_sf_noahlsm.F:2080
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2081
QFX_PHY = AD(QFX_PHY,ET1[(K)-(1)]); // module_sf_noahlsm.F:2082
if (((SMC[(K)-(1)] >= SMCREF) && (XSDA_QFX > 0x0.0p+0f))) { WETTY[(K)-(1)] = 0x0.0p+0f; } // module_sf_noahlsm.F:2084
} // module_sf_noahlsm.F:2085
QFX_PHY = AD(AD(EDIR1,EC1),QFX_PHY); // module_sf_noahlsm.F:2086
EALL_NOW = QFX_PHY; // module_sf_noahlsm.F:2087
QFX_PHY = MU(QFX_PHY,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2088
if ((EALL_NOW != 0x0.0p+0f)) { // module_sf_noahlsm.F:2090
EFDIR = MU(MU(MU(DV(EDIR1,EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2091
EFDIR = MU(EFDIR,WETTY[(1)-(1)]); // module_sf_noahlsm.F:2092
EDIR1 = AD(EDIR1,EFDIR); // module_sf_noahlsm.F:2094
EFC = MU(MU(MU(DV(EC1,EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2096
EC1 = AD(EC1,EFC); // module_sf_noahlsm.F:2098
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2100
EFT[(K)-(1)] = MU(MU(MU(DV(ET1[(K)-(1)],EALL_NOW),XSDA_QFX),0x1.0624de0000000p-10f),XQNORM); // module_sf_noahlsm.F:2101
EFT[(K)-(1)] = MU(EFT[(K)-(1)],WETTY[(K)-(1)]); // module_sf_noahlsm.F:2102
ET1[(K)-(1)] = AD(ET1[(K)-(1)],EFT[(K)-(1)]); // module_sf_noahlsm.F:2104
} // module_sf_noahlsm.F:2105
} // module_sf_noahlsm.F:2107
} else { // module_sf_noahlsm.F:2108
QFX_PHY = 0x0.0p+0f; // module_sf_noahlsm.F:2109
} // module_sf_noahlsm.F:2110
{  noah_smflx(tables,SMC,NSOIL,CMC,DT,PRCP1,ZSOIL,SH2O,SLOPE,KDT,FRZFACT,SMCMAX,BEXP,SMCWLT,DKSAT,DWSAT,SHDFAC,CMCMAX,RUNOFF1,RUNOFF2,RUNOFF3,EDIR1,EC1,ET1,DRIP,SFHEAD1RT,INFXS1RT,IRRIGATION_CHANNEL,present_IRRIGATION_CHANNEL); if(tables.error)return; } // module_sf_noahlsm.F:2114
} // module_sf_noahlsm.F:2126
if ((ETP <= 0x0.0p+0f)) { // module_sf_noahlsm.F:2132
BETA = 0x0.0p+0f; // module_sf_noahlsm.F:2133
ETA = ETP; // module_sf_noahlsm.F:2134
if ((ETP < 0x0.0p+0f)) { // module_sf_noahlsm.F:2135
BETA = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:2136
} // module_sf_noahlsm.F:2137
} else { // module_sf_noahlsm.F:2138
BETA = DV(ETA,ETP); // module_sf_noahlsm.F:2139
} // module_sf_noahlsm.F:2140
EDIR = MU(EDIR1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2145
EC = MU(EC1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2146
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2147
ET[(K)-(1)] = MU(ET1[(K)-(1)],0x1.f400000000000p+9f); // module_sf_noahlsm.F:2148
} // module_sf_noahlsm.F:2149
ETT = MU(ETT1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:2150
{  noah_tdfcnd(tables,DF1,SMC[(1)-(1)],QUARTZ,SMCMAX,SH2O[(1)-(1)],BEXP,PSISAT,SOILTYP,OPT_THCND); if(tables.error)return; } // module_sf_noahlsm.F:2158
if ((VEGTYP == ISURBAN)) { DF1 = 0x1.9eb8520000000p+1f; } // module_sf_noahlsm.F:2161
DF1 = MU(DF1,r_exp(MU(SBETA,SHDFAC))); // module_sf_noahlsm.F:2171
YYNUM = SU(FDOWN,MU(MU(EMISSI,SIGMA),T24)); // module_sf_noahlsm.F:2176
YY = AD(SFCTMP,DV(SU(SU(AD(DV(YYNUM,RCH),TH2),SFCTMP),MU(BETA,EPSCA)),RR)); // module_sf_noahlsm.F:2177
ZZ1 = AD(DV(DF1,MU(MU(MU(-0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),RCH),RR)),0x1.0000000000000p+0f); // module_sf_noahlsm.F:2179
{  noah_shflx(tables,SSOIL,STC,SMC,SMCMAX,NSOIL,T1,DT,YY,ZZ1,ZSOIL,TBOT,ZBOT,SMCWLT,PSISAT,SH2O,BEXP,F1,DF1,QUARTZ,CSOIL,VEGTYP,ISURBAN,SOILTYP,OPT_THCND,HCPCT_FASDAS); if(tables.error)return; } // module_sf_noahlsm.F:2182
FLX1 = MU(MU(CPH2O,PRCP),SU(T1,SFCTMP)); // module_sf_noahlsm.F:2192
FLX3 = 0x0.0p+0f; // module_sf_noahlsm.F:2193
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2199-2316
__device__ void noah_penman(NoahTables& tables, float& SFCTMP, float& SFCPRS, float& CH, float& T2V, float& TH2, float& PRCP, float& FDOWN, float& T24, float& SSOIL, float& Q2, float& Q2SAT, float& ETP, float& RCH, float& EPSCA, float& RR, bool& SNOWNG, bool& FRZGRA, float& DQSDT2, float& FLX2, float& EMISSI_IN, float& SNEQV, float& T1, float& SNCOVR, float& AOASIS, float& ALBEDO, float& SOLDN, float& FVB, float& GAMA, float& STC1, float& ETPN, float& FLX4, bool& UA_PHYS){
auto& RD=tables.RD;
auto& SIGMA=tables.SIGMA;
auto& CPH2O=tables.CPH2O;
auto& CPICE=tables.CPICE;
auto& LSUBF=tables.LSUBF;
float A;
float DELTA;
float FNET;
float RAD;
float RHO;
float EMISSI;
float ELCP1;
float LVS;
float TOTABS;
float UCABS;
float SIGNCK;
float FNETN;
float RADN;
float EPSCAN;
float ELCP = 0x1.37199a0000000p+11f;
float LSUBC = 0x1.314c400000000p+21f;
float CP = 0x1.f64ccc0000000p+9f;
float LSUBS = 0x1.5975800000000p+21f;
float ALGDSN = 0x1.0000000000000p-1f;
float ALVGSN = 0x1.0a3d700000000p-3f;
 // module_sf_noahlsm.F:2211
EMISSI = EMISSI_IN; // module_sf_noahlsm.F:2235
ELCP1 = AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),ELCP),DV(MU(MU(SNCOVR,ELCP),LSUBS),LSUBC)); // module_sf_noahlsm.F:2236
LVS = AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),LSUBC),MU(SNCOVR,LSUBS)); // module_sf_noahlsm.F:2237
FLX2 = 0x0.0p+0f; // module_sf_noahlsm.F:2239
DELTA = MU(ELCP1,DQSDT2); // module_sf_noahlsm.F:2241
T24 = MU(MU(MU(SFCTMP,SFCTMP),SFCTMP),SFCTMP); // module_sf_noahlsm.F:2242
RR = AD(DV(MU(MU(EMISSI,T24),0x1.16505a0000000p-24f),MU(SFCPRS,CH)),0x1.0000000000000p+0f); // module_sf_noahlsm.F:2244
RHO = DV(SFCPRS,MU(RD,T2V)); // module_sf_noahlsm.F:2245
RCH = MU(MU(RHO,CP),CH); // module_sf_noahlsm.F:2251
if ((!SNOWNG)) { // module_sf_noahlsm.F:2252
if ((PRCP > 0x0.0p+0f)) { RR = AD(RR,DV(MU(CPH2O,PRCP),RCH)); } // module_sf_noahlsm.F:2253
} else { // module_sf_noahlsm.F:2254
RR = AD(RR,DV(MU(CPICE,PRCP),RCH)); // module_sf_noahlsm.F:2255
} // module_sf_noahlsm.F:2256
FNET = SU(SU(FDOWN,MU(MU(EMISSI,SIGMA),T24)),SSOIL); // module_sf_noahlsm.F:2263
FLX4 = 0x0.0p+0f; // module_sf_noahlsm.F:2265
if (UA_PHYS) { // module_sf_noahlsm.F:2266
if ((((SNEQV > 0x0.0p+0f) && (FNET > 0x0.0p+0f)) && (SOLDN > 0x0.0p+0f))) { // module_sf_noahlsm.F:2267
TOTABS = MU(MU(SU(0x1.0000000000000p+0f,ALBEDO),SOLDN),FVB); // module_sf_noahlsm.F:2268
UCABS = noah_min(TOTABS,MU(MU(MU(MU(SU(0x1.0000000000000p+0f,ALGDSN),SU(0x1.0000000000000p+0f,ALVGSN)),SOLDN),GAMA),FVB)); // module_sf_noahlsm.F:2270
FLX4 = noah_min(SU(TOTABS,UCABS),noah_min(0x1.f400000000000p+7f,MU(MU(0x1.0000000000000p-1f,SU(0x1.0000000000000p+0f,ALBEDO)),SOLDN))); // module_sf_noahlsm.F:2275
} // module_sf_noahlsm.F:2276
SIGNCK = MU(SU(STC1,0x1.1126660000000p+8f),SU(SFCTMP,0x1.1126660000000p+8f)); // module_sf_noahlsm.F:2278
if (((FLX4 > 0x0.0p+0f) && ((SIGNCK <= 0x0.0p+0f) || (STC1 < 0x1.1126660000000p+8f)))) { // module_sf_noahlsm.F:2280
if ((FNET >= FLX4)) { // module_sf_noahlsm.F:2281
FNETN = SU(FNET,FLX4); // module_sf_noahlsm.F:2282
} else { // module_sf_noahlsm.F:2283
FLX4 = FNET; // module_sf_noahlsm.F:2284
FNETN = 0x0.0p+0f; // module_sf_noahlsm.F:2285
} // module_sf_noahlsm.F:2286
} else { // module_sf_noahlsm.F:2287
FLX4 = 0x0.0p+0f; // module_sf_noahlsm.F:2288
FNETN = 0x0.0p+0f; // module_sf_noahlsm.F:2289
} // module_sf_noahlsm.F:2290
} // module_sf_noahlsm.F:2291
if (FRZGRA) { // module_sf_noahlsm.F:2293
FLX2 = MU((-LSUBF),PRCP); // module_sf_noahlsm.F:2294
FNET = SU(FNET,FLX2); // module_sf_noahlsm.F:2295
if (UA_PHYS) { FNETN = SU(FNETN,FLX2); } // module_sf_noahlsm.F:2296
} // module_sf_noahlsm.F:2300
RAD = SU(AD(DV(FNET,RCH),TH2),SFCTMP); // module_sf_noahlsm.F:2301
A = MU(ELCP1,SU(Q2SAT,Q2)); // module_sf_noahlsm.F:2303
EPSCA = DV(AD(MU(A,RR),MU(RAD,DELTA)),AD(DELTA,RR)); // module_sf_noahlsm.F:2304
if ((EPSCA > 0x0.0p+0f)) { EPSCA = MU(EPSCA,AOASIS); } // module_sf_noahlsm.F:2306
ETP = DV(MU(EPSCA,RCH),LVS); // module_sf_noahlsm.F:2308
if (UA_PHYS) { // module_sf_noahlsm.F:2310
RADN = SU(AD(DV(FNETN,RCH),TH2),SFCTMP); // module_sf_noahlsm.F:2311
EPSCAN = DV(AD(MU(A,RR),MU(RADN,DELTA)),AD(DELTA,RR)); // module_sf_noahlsm.F:2312
ETPN = DV(MU(EPSCAN,RCH),LVS); // module_sf_noahlsm.F:2313
} // module_sf_noahlsm.F:2314
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2319-2539
__device__ void noah_redprm(NoahTables& tables, int& VEGTYP, int& SOILTYP, int& SLOPETYP, float& CFACTR, float& CMCMAX, float& RSMAX, float& TOPT, float& REFKDT, float& KDT, float& SBETA, float& SHDFAC, float& RSMIN, float& RGL, float& HS, float& ZBOT, float& FRZX, float& PSISAT, float& SLOPE, float& SNUP, float& SALP, float& BEXP, float& DKSAT, float& DWSAT, float& SMCMAX, float& SMCWLT, float& SMCREF, float& SMCDRY, float& F1, float& QUARTZ, float& FXEXP, float* RTDIS, float* SLDPTH, float* ZSOIL, int& NROOT, int& NSOIL, float& CZIL, float& LAIMIN, float& LAIMAX, float& EMISSMIN, float& EMISSMAX, float& ALBEDOMIN, float& ALBEDOMAX, float& Z0MIN, float& Z0MAX, float& CSOIL, float& PTU, const char* LLANDUSE, const char* LSOIL, bool& LOCAL, float& LVCOEF, float& ZTOPV, float& ZBOTV){
auto& LUCATS=tables.LUCATS;
auto& BARE=tables.BARE;
auto& NROTBL=tables.NROTBL;
auto& SNUPTBL=tables.SNUPTBL;
auto& RSTBL=tables.RSTBL;
auto& RGLTBL=tables.RGLTBL;
auto& HSTBL=tables.HSTBL;
auto& EMISSMINTBL=tables.EMISSMINTBL;
auto& EMISSMAXTBL=tables.EMISSMAXTBL;
auto& LAIMINTBL=tables.LAIMINTBL;
auto& LAIMAXTBL=tables.LAIMAXTBL;
auto& Z0MINTBL=tables.Z0MINTBL;
auto& Z0MAXTBL=tables.Z0MAXTBL;
auto& ALBEDOMINTBL=tables.ALBEDOMINTBL;
auto& ALBEDOMAXTBL=tables.ALBEDOMAXTBL;
auto& ZTOPVTBL=tables.ZTOPVTBL;
auto& ZBOTVTBL=tables.ZBOTVTBL;
auto& TOPT_DATA=tables.TOPT_DATA;
auto& CMCMAX_DATA=tables.CMCMAX_DATA;
auto& CFACTR_DATA=tables.CFACTR_DATA;
auto& RSMAX_DATA=tables.RSMAX_DATA;
auto& SLCATS=tables.SLCATS;
auto& BB=tables.BB;
auto& DRYSMC=tables.DRYSMC;
auto& F11=tables.F11;
auto& MAXSMC=tables.MAXSMC;
auto& REFSMC=tables.REFSMC;
auto& SATPSI=tables.SATPSI;
auto& SATDK=tables.SATDK;
auto& SATDW=tables.SATDW;
auto& WLTSMC=tables.WLTSMC;
auto& QTZ=tables.QTZ;
auto& SLPCATS=tables.SLPCATS;
auto& SLOPE_DATA=tables.SLOPE_DATA;
auto& SBETA_DATA=tables.SBETA_DATA;
auto& FXEXP_DATA=tables.FXEXP_DATA;
auto& CSOIL_DATA=tables.CSOIL_DATA;
auto& SALP_DATA=tables.SALP_DATA;
auto& REFDK_DATA=tables.REFDK_DATA;
auto& REFKDT_DATA=tables.REFKDT_DATA;
auto& FRZK_DATA=tables.FRZK_DATA;
auto& ZBOT_DATA=tables.ZBOT_DATA;
auto& CZIL_DATA=tables.CZIL_DATA;
auto& LVCOEF_DATA=tables.LVCOEF_DATA;
const char* ERR_MESSAGE=tables.ERR_MESSAGE;
int MAX_SLOPETYP = 30;
int MAX_SOILTYP = 30;
int MAX_VEGTYP = 30;
int I;
float FRZFACT;
float FRZK;
float REFDK;
 // module_sf_noahlsm.F:2329
if ((SOILTYP > SLCATS)) { // module_sf_noahlsm.F:2437
tables.error = 2438; return; // module_sf_noahlsm.F:2438
} // module_sf_noahlsm.F:2439
if ((VEGTYP > LUCATS)) { // module_sf_noahlsm.F:2440
tables.error = 2441; return; // module_sf_noahlsm.F:2441
} // module_sf_noahlsm.F:2442
if ((SLOPETYP > SLPCATS)) { // module_sf_noahlsm.F:2443
tables.error = 2444; return; // module_sf_noahlsm.F:2444
} // module_sf_noahlsm.F:2445
CSOIL = CSOIL_DATA; // module_sf_noahlsm.F:2450
BEXP = BB[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2451
DKSAT = SATDK[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2452
DWSAT = SATDW[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2453
F1 = F11[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2454
PSISAT = SATPSI[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2455
QUARTZ = QTZ[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2456
SMCDRY = DRYSMC[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2457
SMCMAX = MAXSMC[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2458
SMCREF = REFSMC[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2459
SMCWLT = WLTSMC[(SOILTYP)-(1)]; // module_sf_noahlsm.F:2460
ZBOT = ZBOT_DATA; // module_sf_noahlsm.F:2465
SALP = SALP_DATA; // module_sf_noahlsm.F:2466
SBETA = SBETA_DATA; // module_sf_noahlsm.F:2467
REFDK = REFDK_DATA; // module_sf_noahlsm.F:2468
FRZK = FRZK_DATA; // module_sf_noahlsm.F:2469
FXEXP = FXEXP_DATA; // module_sf_noahlsm.F:2470
REFKDT = REFKDT_DATA; // module_sf_noahlsm.F:2471
PTU = 0x0.0p+0f; // module_sf_noahlsm.F:2472
KDT = DV(MU(REFKDT,DKSAT),REFDK); // module_sf_noahlsm.F:2473
CZIL = CZIL_DATA; // module_sf_noahlsm.F:2474
SLOPE = SLOPE_DATA[(SLOPETYP)-(1)]; // module_sf_noahlsm.F:2475
LVCOEF = LVCOEF_DATA; // module_sf_noahlsm.F:2476
FRZFACT = MU(DV(SMCMAX,SMCREF),0x1.c2bc2c0000000p-1f); // module_sf_noahlsm.F:2481
FRZX = MU(FRZK,FRZFACT); // module_sf_noahlsm.F:2482
TOPT = TOPT_DATA; // module_sf_noahlsm.F:2487
CMCMAX = CMCMAX_DATA; // module_sf_noahlsm.F:2488
CFACTR = CFACTR_DATA; // module_sf_noahlsm.F:2489
RSMAX = RSMAX_DATA; // module_sf_noahlsm.F:2490
NROOT = NROTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2491
SNUP = SNUPTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2492
RSMIN = RSTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2493
RGL = RGLTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2494
HS = HSTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2495
EMISSMIN = EMISSMINTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2496
EMISSMAX = EMISSMAXTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2497
LAIMIN = LAIMINTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2498
LAIMAX = LAIMAXTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2499
Z0MIN = Z0MINTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2500
Z0MAX = Z0MAXTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2501
ALBEDOMIN = ALBEDOMINTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2502
ALBEDOMAX = ALBEDOMAXTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2503
ZTOPV = ZTOPVTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2504
ZBOTV = ZBOTVTBL[(VEGTYP)-(1)]; // module_sf_noahlsm.F:2505
if ((VEGTYP == BARE)) { SHDFAC = 0x0.0p+0f; } // module_sf_noahlsm.F:2507
if ((NROOT > NSOIL)) { // module_sf_noahlsm.F:2508
// Native diagnostic formatting is handled by the host error receipt. // module_sf_noahlsm.F:2509
tables.error = 2511; return; // module_sf_noahlsm.F:2511
} // module_sf_noahlsm.F:2516
for (I=1; I<=NROOT; I+=1) { // module_sf_noahlsm.F:2517
RTDIS[(I)-(1)] = DV((-SLDPTH[(I)-(1)]),ZSOIL[(NROOT)-(1)]); // module_sf_noahlsm.F:2518
} // module_sf_noahlsm.F:2522
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2541-2600
__device__ void noah_rosr12(NoahTables& tables, float* P, float* A, float* B, float* C, float* D, float* DELTA, int& NSOIL){
int K;
int KK;
 // module_sf_noahlsm.F:2561
C[(NSOIL)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:2572
P[(1)-(1)] = DV((-C[(1)-(1)]),B[(1)-(1)]); // module_sf_noahlsm.F:2573
DELTA[(1)-(1)] = DV(D[(1)-(1)],B[(1)-(1)]); // module_sf_noahlsm.F:2581
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2582
P[(K)-(1)] = MU((-C[(K)-(1)]),DV(0x1.0000000000000p+0f,AD(B[(K)-(1)],MU(A[(K)-(1)],P[((K - 1))-(1)])))); // module_sf_noahlsm.F:2583
DELTA[(K)-(1)] = MU(SU(D[(K)-(1)],MU(A[(K)-(1)],DELTA[((K - 1))-(1)])),DV(0x1.0000000000000p+0f,AD(B[(K)-(1)],MU(A[(K)-(1)],P[((K - 1))-(1)])))); // module_sf_noahlsm.F:2584
} // module_sf_noahlsm.F:2586
P[(NSOIL)-(1)] = DELTA[(NSOIL)-(1)]; // module_sf_noahlsm.F:2590
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2595
KK = ((NSOIL - K) + 1); // module_sf_noahlsm.F:2596
P[(KK)-(1)] = AD(MU(P[(KK)-(1)],P[((KK + 1))-(1)]),DELTA[(KK)-(1)]); // module_sf_noahlsm.F:2597
} // module_sf_noahlsm.F:2598
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2604-2670
__device__ void noah_shflx(NoahTables& tables, float& SSOIL, float* STC, float* SMC, float& SMCMAX, int& NSOIL, float& T1, float& DT, float& YY, float& ZZ1, float* ZSOIL, float& TBOT, float& ZBOT, float& SMCWLT, float& PSISAT, float* SH2O, float& BEXP, float& F1, float& DF1, float& QUARTZ, float& CSOIL, int& VEGTYP, int& ISURBAN, int& SOILTYP, int& OPT_THCND, float& HCPCT_FASDAS){
int I;
float AI[20];
float BI[20];
float CI[20];
float STCF[20];
float RHSTS[20];
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm.F:2616
{  noah_hrt(tables,RHSTS,STC,SMC,SMCMAX,NSOIL,ZSOIL,YY,ZZ1,TBOT,ZBOT,PSISAT,SH2O,DT,BEXP,SOILTYP,OPT_THCND,F1,DF1,QUARTZ,CSOIL,AI,BI,CI,VEGTYP,ISURBAN,HCPCT_FASDAS); if(tables.error)return; } // module_sf_noahlsm.F:2645
{  noah_hstep(tables,STCF,STC,RHSTS,DT,NSOIL,AI,BI,CI); if(tables.error)return; } // module_sf_noahlsm.F:2650
for (I=1; I<=NSOIL; I+=1) { // module_sf_noahlsm.F:2652
STC[(I)-(1)] = STCF[(I)-(1)]; // module_sf_noahlsm.F:2653
} // module_sf_noahlsm.F:2654
T1 = DV(AD(YY,MU(SU(ZZ1,0x1.0000000000000p+0f),STC[(1)-(1)])),ZZ1); // module_sf_noahlsm.F:2666
SSOIL = DV(MU(DF1,SU(STC[(1)-(1)],T1)),MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm.F:2667
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2673-2817
__device__ void noah_smflx(NoahTables& tables, float* SMC, int& NSOIL, float& CMC, float& DT, float& PRCP1, float* ZSOIL, float* SH2O, float& SLOPE, float& KDT, float& FRZFACT, float& SMCMAX, float& BEXP, float& SMCWLT, float& DKSAT, float& DWSAT, float& SHDFAC, float& CMCMAX, float& RUNOFF1, float& RUNOFF2, float& RUNOFF3, float& EDIR, float& EC, float* ET, float& DRIP, float& SFHEAD1RT, float& INFXS1RT, float& IRRIGATION_CHANNEL, bool present_IRRIGATION_CHANNEL){
int I;
int K;
float AI[20];
float BI[20];
float CI[20];
float STCF[20];
float RHSTS[20];
float RHSTT[20];
float SICE[20];
float SH2OA[20];
float SH2OFG[20];
float SH2OIN[20];
float DUMMY;
float EXCESS;
float PCPDRP;
float RHSCT;
float TRHSCT;
float FAC2;
float FLIMIT;
 // module_sf_noahlsm.F:2691
DUMMY = 0x0.0p+0f; // module_sf_noahlsm.F:2716
RHSCT = SU(MU(SHDFAC,PRCP1),EC); // module_sf_noahlsm.F:2723
DRIP = 0x0.0p+0f; // module_sf_noahlsm.F:2724
TRHSCT = MU(DT,RHSCT); // module_sf_noahlsm.F:2725
EXCESS = AD(CMC,TRHSCT); // module_sf_noahlsm.F:2726
if ((EXCESS > CMCMAX)) { DRIP = SU(EXCESS,CMCMAX); } // module_sf_noahlsm.F:2732
PCPDRP = AD(MU(SU(0x1.0000000000000p+0f,SHDFAC),PRCP1),DV(DRIP,DT)); // module_sf_noahlsm.F:2733
if (present_IRRIGATION_CHANNEL) { // module_sf_noahlsm.F:2734
if ((IRRIGATION_CHANNEL != 0x0.0p+0f)) { PCPDRP = AD(PCPDRP,MU(0x1.0624de0000000p-10f,IRRIGATION_CHANNEL)); } // module_sf_noahlsm.F:2735
} // module_sf_noahlsm.F:2736
for (I=1; I<=NSOIL; I+=1) { // module_sf_noahlsm.F:2740
SICE[(I)-(1)] = SU(SMC[(I)-(1)],SH2O[(I)-(1)]); // module_sf_noahlsm.F:2741
} // module_sf_noahlsm.F:2742
FAC2 = 0x0.0p+0f; // module_sf_noahlsm.F:2766
for (I=1; I<=NSOIL; I+=1) { // module_sf_noahlsm.F:2767
FAC2 = noah_max(FAC2,DV(SH2O[(I)-(1)],SMCMAX)); // module_sf_noahlsm.F:2768
} // module_sf_noahlsm.F:2769
{  noah_fac2mit(tables,SMCMAX,FLIMIT); if(tables.error)return; } // module_sf_noahlsm.F:2770
if (((MU(PCPDRP,DT) > MU(MU(0x1.9999980000000p-4f,(-ZSOIL[(1)-(1)])),SMCMAX)) || (FAC2 > FLIMIT))) { // module_sf_noahlsm.F:2785
{  noah_srt(tables,RHSTT,EDIR,ET,SH2O,SH2O,NSOIL,PCPDRP,ZSOIL,DWSAT,DKSAT,SMCMAX,BEXP,RUNOFF1,RUNOFF2,DT,SMCWLT,SLOPE,KDT,FRZFACT,SICE,AI,BI,CI,SFHEAD1RT,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2787
{  noah_sstep(tables,SH2OFG,SH2O,DUMMY,RHSTT,RHSCT,DT,NSOIL,SMCMAX,CMCMAX,RUNOFF3,ZSOIL,SMC,SICE,AI,BI,CI,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2791
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:2793
SH2OA[(K)-(1)] = MU(AD(SH2O[(K)-(1)],SH2OFG[(K)-(1)]),0x1.0000000000000p-1f); // module_sf_noahlsm.F:2794
} // module_sf_noahlsm.F:2795
{  noah_srt(tables,RHSTT,EDIR,ET,SH2O,SH2OA,NSOIL,PCPDRP,ZSOIL,DWSAT,DKSAT,SMCMAX,BEXP,RUNOFF1,RUNOFF2,DT,SMCWLT,SLOPE,KDT,FRZFACT,SICE,AI,BI,CI,SFHEAD1RT,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2796
for(int ac=1;ac<=NSOIL;++ac) SH2OIN[ac-(1)] = SH2O[ac-(1)]; // module_sf_noahlsm.F:2800
{  noah_sstep(tables,SH2O,SH2OIN,CMC,RHSTT,RHSCT,DT,NSOIL,SMCMAX,CMCMAX,RUNOFF3,ZSOIL,SMC,SICE,AI,BI,CI,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2801
} else { // module_sf_noahlsm.F:2804
{  noah_srt(tables,RHSTT,EDIR,ET,SH2O,SH2O,NSOIL,PCPDRP,ZSOIL,DWSAT,DKSAT,SMCMAX,BEXP,RUNOFF1,RUNOFF2,DT,SMCWLT,SLOPE,KDT,FRZFACT,SICE,AI,BI,CI,SFHEAD1RT,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2805
for(int ac=1;ac<=NSOIL;++ac) SH2OIN[ac-(1)] = SH2O[ac-(1)]; // module_sf_noahlsm.F:2809
{  noah_sstep(tables,SH2O,SH2OIN,CMC,RHSTT,RHSCT,DT,NSOIL,SMCMAX,CMCMAX,RUNOFF3,ZSOIL,SMC,SICE,AI,BI,CI,INFXS1RT); if(tables.error)return; } // module_sf_noahlsm.F:2810
} // module_sf_noahlsm.F:2814
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2821-2923
__device__ void noah_snfrac(NoahTables& tables, float& SNEQV, float& SNUP, float& SALP, float& SNOWH, float& SNCOVR, float& XLAI, float& SHDFAC, float& FVB, float& GAMA, float& FBUR, float& FGSN, float& ZTOPV, float& ZBOTV, bool& UA_PHYS){
float RSNOW;
float Z0N;
float SNUPGRD = 0x1.47ae140000000p-6f;
 // module_sf_noahlsm.F:2834
if ((SNEQV < SNUP)) { // module_sf_noahlsm.F:2855
RSNOW = DV(SNEQV,SNUP); // module_sf_noahlsm.F:2856
SNCOVR = SU(0x1.0000000000000p+0f,SU(r_exp(MU((-SALP),RSNOW)),MU(RSNOW,r_exp((-SALP))))); // module_sf_noahlsm.F:2857
} else { // module_sf_noahlsm.F:2858
SNCOVR = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:2859
} // module_sf_noahlsm.F:2860
if (UA_PHYS) { // module_sf_noahlsm.F:2870
if ((SNEQV < SNUPGRD)) { // module_sf_noahlsm.F:2875
FGSN = DV(SNEQV,SNUPGRD); // module_sf_noahlsm.F:2876
} else { // module_sf_noahlsm.F:2877
FGSN = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:2878
} // module_sf_noahlsm.F:2879
if (((ZBOTV > 0x0.0p+0f) && (SNOWH > ZBOTV))) { // module_sf_noahlsm.F:2886
if ((ZBOTV <= 0x1.0000000000000p-1f)) { // module_sf_noahlsm.F:2887
FBUR = DV(SU(SNOWH,MU(0x1.99999a0000000p-2f,ZBOTV)),MU(0x1.99999a0000000p-2f,SU(ZTOPV,ZBOTV))); // module_sf_noahlsm.F:2888
} else { // module_sf_noahlsm.F:2889
FBUR = DV(SU(SNOWH,ZBOTV),SU(ZTOPV,ZBOTV)); // module_sf_noahlsm.F:2890
} // module_sf_noahlsm.F:2891
} else { // module_sf_noahlsm.F:2892
FBUR = 0x0.0p+0f; // module_sf_noahlsm.F:2893
} // module_sf_noahlsm.F:2894
FBUR = noah_min(noah_max(FBUR,0x0.0p+0f),0x1.0000000000000p+0f); // module_sf_noahlsm.F:2896
XLAI = MU(XLAI,SU(0x1.0000000000000p+0f,FBUR)); // module_sf_noahlsm.F:2899
FVB = MU(MU(SHDFAC,FGSN),SU(0x1.0000000000000p+0f,FBUR)); // module_sf_noahlsm.F:2909
GAMA = r_exp(MU(-0x1.0000000000000p+0f,XLAI)); // module_sf_noahlsm.F:2913
} else { // module_sf_noahlsm.F:2914
FVB = 0x0.0p+0f; // module_sf_noahlsm.F:2916
GAMA = 0x0.0p+0f; // module_sf_noahlsm.F:2917
FBUR = 0x0.0p+0f; // module_sf_noahlsm.F:2918
FGSN = 0x0.0p+0f; // module_sf_noahlsm.F:2919
} // module_sf_noahlsm.F:2920
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:2926-3011
__device__ void noah_snksrc(NoahTables& tables, float& TSNSR, float& TAVG, float& SMC, float& SH2O, float* ZSOIL, int& NSOIL, float& SMCMAX, float& PSISAT, float& BEXP, float& DT, int& K, float& QTOT){
float DF;
float DZ;
float DZH;
float FREE;
float TDN;
float TM;
float TUP;
float TZ;
float X0;
float XDN;
float XH2O;
float XUP;
float DH2O = 0x1.f400000000000p+9f;
float HLICE = 0x1.45af000000000p+18f;
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm.F:2934
if ((K == 1)) { // module_sf_noahlsm.F:2949
DZ = (-ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:2950
} else { // module_sf_noahlsm.F:2951
DZ = SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]); // module_sf_noahlsm.F:2952
} // module_sf_noahlsm.F:2953
{  noah_frh2o(tables,FREE,TAVG,SMC,SH2O,SMCMAX,BEXP,PSISAT); if(tables.error)return; } // module_sf_noahlsm.F:2972
XH2O = AD(SH2O,DV(MU(QTOT,DT),MU(MU(DH2O,HLICE),DZ))); // module_sf_noahlsm.F:2979
if (((XH2O < SH2O) && (XH2O < FREE))) { // module_sf_noahlsm.F:2980
if ((FREE > SH2O)) { // module_sf_noahlsm.F:2981
XH2O = SH2O; // module_sf_noahlsm.F:2982
} else { // module_sf_noahlsm.F:2983
XH2O = FREE; // module_sf_noahlsm.F:2984
} // module_sf_noahlsm.F:2985
} // module_sf_noahlsm.F:2986
if (((XH2O > SH2O) && (XH2O > FREE))) { // module_sf_noahlsm.F:2992
if ((FREE < SH2O)) { // module_sf_noahlsm.F:2993
XH2O = SH2O; // module_sf_noahlsm.F:2994
} else { // module_sf_noahlsm.F:2995
XH2O = FREE; // module_sf_noahlsm.F:2996
} // module_sf_noahlsm.F:2997
} // module_sf_noahlsm.F:2998
if ((XH2O < 0x0.0p+0f)) { XH2O = 0x0.0p+0f; } // module_sf_noahlsm.F:3005
if ((XH2O > SMC)) { XH2O = SMC; } // module_sf_noahlsm.F:3006
TSNSR = DV(MU(MU(MU((-DH2O),HLICE),DZ),SU(XH2O,SH2O)),DT); // module_sf_noahlsm.F:3007
SH2O = XH2O; // module_sf_noahlsm.F:3008
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3014-3417
__device__ void noah_snopac(NoahTables& tables, float& ETP, float& ETA, float& PRCP, float& PRCPF, bool& SNOWNG, float* SMC, float& SMCMAX, float& SMCWLT, float& SMCREF, float& SMCDRY, float& CMC, float& CMCMAX, int& NSOIL, float& DT, float& SBETA, float& DF1, float& Q2, float& T1, float& SFCTMP, float& T24, float& TH2, float& FDOWN, float& F1, float& SSOIL, float* STC, float& EPSCA, float& SFCPRS, float& BEXP, float& PC, float& RCH, float& RR, float& CFACTR, float& SNCOVR, float& ESD, float& SNDENS, float& SNOWH, float* SH2O, float& SLOPE, float& KDT, float& FRZFACT, float& PSISAT, float* ZSOIL, float& DWSAT, float& DKSAT, float& TBOT, float& ZBOT, float& SHDFAC, float& RUNOFF1, float& RUNOFF2, float& RUNOFF3, float& EDIR, float& EC, float* ET, float& ETT, int& NROOT, float& SNOMLT, float* RTDIS, float& QUARTZ, float& FXEXP, float& CSOIL, float& BETA, float& DRIP, float& DEW, float& FLX1, float& FLX2, float& FLX3, float& ESNOW, float& ETNS, float& EMISSI, float& RIBB, float& SOLDN, int& ISURBAN, int& VEGTYP, float& ETPN, float& FLX4, bool& UA_PHYS, float& SFHEAD1RT, float& INFXS1RT, float& ETPND1, int& SOILTYP, int& OPT_THCND, float& QFX_PHY, int& FASDAS, float& HCPCT_FASDAS){
auto& SIGMA=tables.SIGMA;
auto& CPH2O=tables.CPH2O;
auto& CPICE=tables.CPICE;
auto& LSUBF=tables.LSUBF;
int K;
int IT16;
float ET1[20];
float DENOM;
float DSOIL;
float DTOT;
float EC1;
float EDIR1;
float ESDFLX;
float ETT1;
float ESNOW1;
float ESNOW2;
float ETA1;
float ETP1;
float ETP2;
float ETP3;
float ETNS1;
float ETANRG;
float ETAX;
float EX;
float FLX3X;
float FRCSNO;
float FRCSOI;
float PRCP1;
float QSAT;
float RSNOW;
float SEH;
float SNCOND;
float SSOIL1;
float T11;
float T12;
float T12A;
float T12AX;
float T12B;
float T14;
float YY;
float ZZ1;
float T15;
float T16;
float DTOT2;
float ESDMIN = 0x1.0c6f7a0000000p-20f;
float LSUBC = 0x1.314c400000000p+21f;
float LSUBS = 0x1.5975800000000p+21f;
float TFREEZ = 0x1.1126660000000p+8f;
float SNOEXP = 0x1.0000000000000p+1f;
float ETP1N;
 // module_sf_noahlsm.F:3037
DEW = 0x0.0p+0f; // module_sf_noahlsm.F:3110
EDIR = 0x0.0p+0f; // module_sf_noahlsm.F:3111
EDIR1 = 0x0.0p+0f; // module_sf_noahlsm.F:3112
EC1 = 0x0.0p+0f; // module_sf_noahlsm.F:3113
EC = 0x0.0p+0f; // module_sf_noahlsm.F:3114
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3117
ET[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:3118
ET1[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:3119
} // module_sf_noahlsm.F:3120
ETT = 0x0.0p+0f; // module_sf_noahlsm.F:3121
ETT1 = 0x0.0p+0f; // module_sf_noahlsm.F:3122
ETPND1 = 0x0.0p+0f; // module_sf_noahlsm.F:3125
ETNS = 0x0.0p+0f; // module_sf_noahlsm.F:3128
ETNS1 = 0x0.0p+0f; // module_sf_noahlsm.F:3129
ESNOW = 0x0.0p+0f; // module_sf_noahlsm.F:3130
ESNOW1 = 0x0.0p+0f; // module_sf_noahlsm.F:3131
ESNOW2 = 0x0.0p+0f; // module_sf_noahlsm.F:3132
PRCP1 = MU(PRCPF,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:3137
BETA = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3141
if ((ETP <= 0x0.0p+0f)) { // module_sf_noahlsm.F:3142
if (((RIBB >= 0x1.99999a0000000p-4f) && (FDOWN > 0x1.2c00000000000p+7f))) { // module_sf_noahlsm.F:3143
ETP = DV(AD(DV(MU(noah_min(MU(ETP,SU(0x1.0000000000000p+0f,RIBB)),0x0.0p+0f),SNCOVR),0x1.f5c2900000000p-1f),MU(ETP,SU(0x1.f5c2900000000p-1f,SNCOVR))),0x1.f5c2900000000p-1f); // module_sf_noahlsm.F:3144
} // module_sf_noahlsm.F:3145
if ((ETP == 0x0.0p+0f)) { BETA = 0x0.0p+0f; } // module_sf_noahlsm.F:3146
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:3147
if (UA_PHYS) { ETP1N = MU(ETPN,0x1.0624de0000000p-10f); } // module_sf_noahlsm.F:3148
DEW = (-ETP1); // module_sf_noahlsm.F:3149
ESNOW2 = MU(ETP1,DT); // module_sf_noahlsm.F:3150
ETANRG = MU(ETP,AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),LSUBC),MU(SNCOVR,LSUBS))); // module_sf_noahlsm.F:3151
} else { // module_sf_noahlsm.F:3152
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:3153
if (UA_PHYS) { ETP1N = MU(ETPN,0x1.0624de0000000p-10f); } // module_sf_noahlsm.F:3154
if ((SNCOVR < 0x1.0000000000000p+0f)) { // module_sf_noahlsm.F:3156
{  noah_evapo(tables,ETNS1,SMC,NSOIL,CMC,ETP1,DT,ZSOIL,SH2O,SMCMAX,BEXP,PC,SMCWLT,DKSAT,DWSAT,SMCREF,SHDFAC,CMCMAX,SMCDRY,CFACTR,EDIR1,EC1,ET1,ETT1,SFCTMP,Q2,NROOT,RTDIS,FXEXP,SFHEAD1RT,ETPND1); if(tables.error)return; } // module_sf_noahlsm.F:3157
EDIR1 = MU(EDIR1,SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3165
EC1 = MU(EC1,SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3166
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3167
ET1[(K)-(1)] = MU(ET1[(K)-(1)],SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3168
} // module_sf_noahlsm.F:3169
ETT1 = MU(ETT1,SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3170
ETNS1 = MU(ETNS1,SU(0x1.0000000000000p+0f,SNCOVR)); // module_sf_noahlsm.F:3172
EDIR = MU(EDIR1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3174
EC = MU(EC1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3175
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3176
ET[(K)-(1)] = MU(ET1[(K)-(1)],0x1.f400000000000p+9f); // module_sf_noahlsm.F:3177
} // module_sf_noahlsm.F:3178
if ((FASDAS == 1)) { // module_sf_noahlsm.F:3182
QFX_PHY = AD(EDIR,EC); // module_sf_noahlsm.F:3183
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3184
QFX_PHY = AD(QFX_PHY,ET[(K)-(1)]); // module_sf_noahlsm.F:3185
} // module_sf_noahlsm.F:3186
} // module_sf_noahlsm.F:3187
ETT = MU(ETT1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3191
ETNS = MU(ETNS1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3192
ETPND1 = MU(ETPND1,0x1.f400000000000p+9f); // module_sf_noahlsm.F:3196
} // module_sf_noahlsm.F:3201
ESNOW = MU(ETP,SNCOVR); // module_sf_noahlsm.F:3202
if (UA_PHYS) { ESNOW = MU(ETPN,SNCOVR); } // module_sf_noahlsm.F:3203
ESNOW1 = MU(ESNOW,0x1.0624de0000000p-10f); // module_sf_noahlsm.F:3204
ESNOW2 = MU(ESNOW1,DT); // module_sf_noahlsm.F:3205
ETANRG = AD(MU(ESNOW,LSUBS),MU(ETNS,LSUBC)); // module_sf_noahlsm.F:3206
} // module_sf_noahlsm.F:3207
FLX1 = 0x0.0p+0f; // module_sf_noahlsm.F:3215
if (SNOWNG) { // module_sf_noahlsm.F:3216
FLX1 = MU(MU(CPICE,PRCP),SU(T1,SFCTMP)); // module_sf_noahlsm.F:3217
} else { // module_sf_noahlsm.F:3218
if ((PRCP > 0x0.0p+0f)) { FLX1 = MU(MU(CPH2O,PRCP),SU(T1,SFCTMP)); } // module_sf_noahlsm.F:3219
} // module_sf_noahlsm.F:3228
DSOIL = (-MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm.F:3229
DTOT = AD(SNOWH,DSOIL); // module_sf_noahlsm.F:3230
DENOM = AD(0x1.0000000000000p+0f,DV(DF1,MU(MU(DTOT,RR),RCH))); // module_sf_noahlsm.F:3231
T12A = DV(SU(SU(AD(DV(SU(SU(SU(FDOWN,FLX1),FLX2),MU(MU(EMISSI,SIGMA),T24)),RCH),TH2),SFCTMP),DV(ETANRG,RCH)),RR); // module_sf_noahlsm.F:3236
T12B = DV(MU(DF1,STC[(1)-(1)]),MU(MU(DTOT,RR),RCH)); // module_sf_noahlsm.F:3239
T12 = DV(AD(AD(SFCTMP,T12A),T12B),DENOM); // module_sf_noahlsm.F:3253
if ((T12 <= TFREEZ)) { // module_sf_noahlsm.F:3254
T1 = T12; // module_sf_noahlsm.F:3255
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm.F:3256
ESD = noah_max(0x0.0p+0f,SU(ESD,ESNOW2)); // module_sf_noahlsm.F:3258
FLX3 = 0x0.0p+0f; // module_sf_noahlsm.F:3259
EX = 0x0.0p+0f; // module_sf_noahlsm.F:3260
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm.F:3262
if (UA_PHYS) { FLX4 = 0x0.0p+0f; } // module_sf_noahlsm.F:3263
} else { // module_sf_noahlsm.F:3279
// GNU strict folds real PARAMETER SNOEXP=2 to one rounded multiply; module_sf_noahlsm.F:3282.
T1 = AD(MU(TFREEZ,noah_max(0x1.47ae140000000p-7f,MU(SNCOVR,SNCOVR))),MU(T12,SU(0x1.0000000000000p+0f,noah_max(0x1.47ae140000000p-7f,MU(SNCOVR,SNCOVR))))); // module_sf_noahlsm.F:3282
BETA = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3283
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm.F:3290
if ((SU(ESD,ESNOW2) <= ESDMIN)) { // module_sf_noahlsm.F:3291
ESD = 0x0.0p+0f; // module_sf_noahlsm.F:3292
EX = 0x0.0p+0f; // module_sf_noahlsm.F:3293
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm.F:3294
FLX3 = 0x0.0p+0f; // module_sf_noahlsm.F:3295
if (UA_PHYS) { FLX4 = 0x0.0p+0f; } // module_sf_noahlsm.F:3296
} else { // module_sf_noahlsm.F:3301
ESD = SU(ESD,ESNOW2); // module_sf_noahlsm.F:3302
ETP3 = MU(ETP,LSUBC); // module_sf_noahlsm.F:3303
SEH = MU(RCH,SU(T1,TH2)); // module_sf_noahlsm.F:3304
T14 = MU(T1,T1); // module_sf_noahlsm.F:3305
T14 = MU(T14,T14); // module_sf_noahlsm.F:3306
FLX3 = SU(SU(SU(SU(SU(SU(FDOWN,FLX1),FLX2),MU(MU(EMISSI,SIGMA),T14)),SSOIL),SEH),ETANRG); // module_sf_noahlsm.F:3310
if ((FLX3 <= 0x0.0p+0f)) { FLX3 = 0x0.0p+0f; } // module_sf_noahlsm.F:3311
if (((UA_PHYS && (FLX4 > 0x0.0p+0f)) && (FLX3 > 0x0.0p+0f))) { // module_sf_noahlsm.F:3313
if ((FLX3 >= FLX4)) { // module_sf_noahlsm.F:3314
FLX3 = SU(FLX3,FLX4); // module_sf_noahlsm.F:3315
} else { // module_sf_noahlsm.F:3316
FLX4 = FLX3; // module_sf_noahlsm.F:3317
FLX3 = 0x0.0p+0f; // module_sf_noahlsm.F:3318
} // module_sf_noahlsm.F:3319
} else { // module_sf_noahlsm.F:3320
FLX4 = 0x0.0p+0f; // module_sf_noahlsm.F:3321
} // module_sf_noahlsm.F:3322
EX = DV(MU(FLX3,0x1.0624de0000000p-10f),LSUBF); // module_sf_noahlsm.F:3327
SNOMLT = MU(EX,DT); // module_sf_noahlsm.F:3333
if ((SU(ESD,SNOMLT) >= ESDMIN)) { // module_sf_noahlsm.F:3334
ESD = SU(ESD,SNOMLT); // module_sf_noahlsm.F:3335
} else { // module_sf_noahlsm.F:3339
EX = DV(ESD,DT); // module_sf_noahlsm.F:3340
FLX3 = MU(MU(EX,0x1.f400000000000p+9f),LSUBF); // module_sf_noahlsm.F:3341
SNOMLT = ESD; // module_sf_noahlsm.F:3342
ESD = 0x0.0p+0f; // module_sf_noahlsm.F:3344
} // module_sf_noahlsm.F:3348
} // module_sf_noahlsm.F:3349
PRCP1 = AD(PRCP1,EX); // module_sf_noahlsm.F:3361
} // module_sf_noahlsm.F:3369
{ float tmp1_26=0; noah_smflx(tables,SMC,NSOIL,CMC,DT,PRCP1,ZSOIL,SH2O,SLOPE,KDT,FRZFACT,SMCMAX,BEXP,SMCWLT,DKSAT,DWSAT,SHDFAC,CMCMAX,RUNOFF1,RUNOFF2,RUNOFF3,EDIR1,EC1,ET1,DRIP,SFHEAD1RT,INFXS1RT,tmp1_26,false); if(tables.error)return; } // module_sf_noahlsm.F:3370
ZZ1 = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3385
YY = SU(STC[(1)-(1)],DV(MU(MU(MU(0x1.0000000000000p-1f,SSOIL),ZSOIL[(1)-(1)]),ZZ1),DF1)); // module_sf_noahlsm.F:3386
T11 = T1; // module_sf_noahlsm.F:3395
{  noah_shflx(tables,SSOIL1,STC,SMC,SMCMAX,NSOIL,T11,DT,YY,ZZ1,ZSOIL,TBOT,ZBOT,SMCWLT,PSISAT,SH2O,BEXP,F1,DF1,QUARTZ,CSOIL,VEGTYP,ISURBAN,SOILTYP,OPT_THCND,HCPCT_FASDAS); if(tables.error)return; } // module_sf_noahlsm.F:3396
if ((ESD > 0x0.0p+0f)) { // module_sf_noahlsm.F:3406
{  noah_snowpack(tables,ESD,DT,SNOWH,SNDENS,T1,YY,SNOMLT,UA_PHYS); if(tables.error)return; } // module_sf_noahlsm.F:3407
} else { // module_sf_noahlsm.F:3408
ESD = 0x0.0p+0f; // module_sf_noahlsm.F:3409
SNOWH = 0x0.0p+0f; // module_sf_noahlsm.F:3410
SNDENS = 0x0.0p+0f; // module_sf_noahlsm.F:3411
SNCOND = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3412
SNCOVR = 0x0.0p+0f; // module_sf_noahlsm.F:3413
} // module_sf_noahlsm.F:3414
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3421-3553
__device__ void noah_snowpack(NoahTables& tables, float& ESD, float& DTSEC, float& SNOWH, float& SNDENS, float& TSNOW, float& TSOIL, float& SNOMLT, bool& UA_PHYS){
int IPOL;
int J;
float BFAC;
float DSX;
float DTHR;
float DW;
float SNOWHC;
float PEXP;
float TAVGC;
float TSNOWC;
float TSOILC;
float ESDC;
float ESDCX;
float C1 = 0x1.47ae140000000p-7f;
float C2 = 0x1.5000000000000p+4f;
float G = 0x1.39eb860000000p+3f;
float KN = 0x1.f400000000000p+11f;
float SNOMLTC;
 // module_sf_noahlsm.F:3440
SNOWHC = MU(SNOWH,0x1.9000000000000p+6f); // module_sf_noahlsm.F:3455
ESDC = MU(ESD,0x1.9000000000000p+6f); // module_sf_noahlsm.F:3456
if (UA_PHYS) { SNOMLTC = MU(SNOMLT,0x1.9000000000000p+6f); } // module_sf_noahlsm.F:3457
DTHR = DV(DTSEC,0x1.c200000000000p+11f); // module_sf_noahlsm.F:3458
TSNOWC = SU(TSNOW,0x1.1126660000000p+8f); // module_sf_noahlsm.F:3459
TSOILC = SU(TSOIL,0x1.1126660000000p+8f); // module_sf_noahlsm.F:3460
TAVGC = MU(0x1.0000000000000p-1f,AD(TSNOWC,TSOILC)); // module_sf_noahlsm.F:3474
if ((ESDC > 0x1.47ae140000000p-7f)) { // module_sf_noahlsm.F:3475
ESDCX = ESDC; // module_sf_noahlsm.F:3476
} else { // module_sf_noahlsm.F:3477
ESDCX = 0x1.47ae140000000p-7f; // module_sf_noahlsm.F:3478
} // module_sf_noahlsm.F:3479
BFAC = MU(MU(DTHR,C1),r_exp(SU(MU(0x1.47ae140000000p-4f,TAVGC),MU(C2,SNDENS)))); // module_sf_noahlsm.F:3499
IPOL = 4; // module_sf_noahlsm.F:3500
PEXP = 0x0.0p+0f; // module_sf_noahlsm.F:3501
for (J=IPOL; J>=1; J+=-1) { // module_sf_noahlsm.F:3503
PEXP = DV(MU(MU(AD(0x1.0000000000000p+0f,PEXP),BFAC),ESDCX),((float)((J + 1)))); // module_sf_noahlsm.F:3504
} // module_sf_noahlsm.F:3505
PEXP = AD(PEXP,0x1.0000000000000p+0f); // module_sf_noahlsm.F:3507
DSX = MU(SNDENS,PEXP); // module_sf_noahlsm.F:3528
if ((DSX > 0x1.99999a0000000p-2f)) { DSX = 0x1.99999a0000000p-2f; } // module_sf_noahlsm.F:3529
if ((DSX < 0x1.99999a0000000p-5f)) { DSX = 0x1.99999a0000000p-5f; } // module_sf_noahlsm.F:3530
SNDENS = DSX; // module_sf_noahlsm.F:3536
if ((TSNOWC >= 0x0.0p+0f)) { // module_sf_noahlsm.F:3537
DW = DV(MU(0x1.0a3d700000000p-3f,DTHR),0x1.8000000000000p+4f); // module_sf_noahlsm.F:3538
if ((UA_PHYS && (TSOILC >= 0x0.0p+0f))) { // module_sf_noahlsm.F:3539
DW = noah_min(DW,DV(MU(0x1.0a3d700000000p-3f,SNOMLTC),AD(ESDCX,MU(0x1.0a3d700000000p-3f,SNOMLTC)))); // module_sf_noahlsm.F:3540
} // module_sf_noahlsm.F:3541
SNDENS = AD(MU(SNDENS,SU(0x1.0000000000000p+0f,DW)),DW); // module_sf_noahlsm.F:3542
if ((SNDENS >= 0x1.99999a0000000p-2f)) { SNDENS = 0x1.99999a0000000p-2f; } // module_sf_noahlsm.F:3543
} // module_sf_noahlsm.F:3548
SNOWHC = DV(ESDC,SNDENS); // module_sf_noahlsm.F:3549
SNOWH = MU(SNOWHC,0x1.47ae140000000p-7f); // module_sf_noahlsm.F:3550
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3556-3601
__device__ void noah_snowz0(NoahTables& tables, float& SNCOVR, float& Z0, float& Z0BRD, float& SNOWH, float& FBUR, float& FGSN, float& SHDMAX, bool& UA_PHYS){
float Z0S = 0x1.0624de0000000p-10f;
float BURIAL;
float Z0EFF;
float Z0G = 0x1.47ae140000000p-7f;
float FV;
float A1;
float A2;
 // module_sf_noahlsm.F:3566
if (UA_PHYS) { // module_sf_noahlsm.F:3580
FV = MU(SHDMAX,SU(0x1.0000000000000p+0f,FBUR)); // module_sf_noahlsm.F:3582
A1 = MU(noah_powi(SU(0x1.0000000000000p+0f,FV),2),AD(MU(SU(0x1.0000000000000p+0f,noah_powi(FGSN,2)),r_log(Z0G)),MU(noah_powi(FGSN,2),r_log(Z0S)))); // module_sf_noahlsm.F:3583
A2 = MU(SU(0x1.0000000000000p+0f,noah_powi(SU(0x1.0000000000000p+0f,FV),2)),r_log(Z0BRD)); // module_sf_noahlsm.F:3584
Z0 = r_exp(AD(A1,A2)); // module_sf_noahlsm.F:3585
} else { // module_sf_noahlsm.F:3587
BURIAL = SU(MU(0x1.c000000000000p+2f,Z0BRD),SNOWH); // module_sf_noahlsm.F:3590
if ((BURIAL <= 0x1.6f00680000000p-11f)) { // module_sf_noahlsm.F:3591
Z0EFF = Z0S; // module_sf_noahlsm.F:3592
} else { // module_sf_noahlsm.F:3593
Z0EFF = DV(BURIAL,0x1.c000000000000p+2f); // module_sf_noahlsm.F:3594
} // module_sf_noahlsm.F:3595
Z0 = AD(MU(SU(0x1.0000000000000p+0f,SNCOVR),Z0BRD),MU(SNCOVR,Z0EFF)); // module_sf_noahlsm.F:3597
} // module_sf_noahlsm.F:3599
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3605-3654
__device__ void noah_snow_new(NoahTables& tables, float& TEMP, float& NEWSN, float& SNOWH, float& SNDENS){
float DSNEW;
float HNEWC;
float SNOWHC;
float NEWSNC;
float TEMPC;
 // module_sf_noahlsm.F:3618
SNOWHC = MU(SNOWH,0x1.9000000000000p+6f); // module_sf_noahlsm.F:3626
NEWSNC = MU(NEWSN,0x1.9000000000000p+6f); // module_sf_noahlsm.F:3627
TEMPC = SU(TEMP,0x1.1126660000000p+8f); // module_sf_noahlsm.F:3635
if ((TEMPC <= -0x1.e000000000000p+3f)) { // module_sf_noahlsm.F:3636
DSNEW = 0x1.99999a0000000p-5f; // module_sf_noahlsm.F:3637
} else { // module_sf_noahlsm.F:3638
DSNEW = AD(0x1.99999a0000000p-5f,MU(0x1.bda5120000000p-10f,r_pow(AD(TEMPC,0x1.e000000000000p+3f),0x1.8000000000000p+0f))); // module_sf_noahlsm.F:3639
} // module_sf_noahlsm.F:3640
HNEWC = DV(NEWSNC,DSNEW); // module_sf_noahlsm.F:3644
if ((AD(SNOWHC,HNEWC) < 0x1.0624de0000000p-10f)) { // module_sf_noahlsm.F:3645
SNDENS = noah_max(DSNEW,SNDENS); // module_sf_noahlsm.F:3646
} else { // module_sf_noahlsm.F:3647
SNDENS = DV(AD(MU(SNOWHC,SNDENS),MU(HNEWC,DSNEW)),AD(SNOWHC,HNEWC)); // module_sf_noahlsm.F:3648
} // module_sf_noahlsm.F:3649
SNOWHC = AD(SNOWHC,HNEWC); // module_sf_noahlsm.F:3650
SNOWH = MU(SNOWHC,0x1.47ae140000000p-7f); // module_sf_noahlsm.F:3651
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3657-3951
__device__ void noah_srt(NoahTables& tables, float* RHSTT, float& EDIR, float* ET, float* SH2O, float* SH2OA, int& NSOIL, float& PCPDRP, float* ZSOIL, float& DWSAT, float& DKSAT, float& SMCMAX, float& BEXP, float& RUNOFF1, float& RUNOFF2, float& DT, float& SMCWLT, float& SLOPE, float& KDT, float& FRZX, float* SICE, float* AI, float* BI, float* CI, float& SFHEAD1RT, float& INFXS1RT){
int IALP1;
int IOHINF;
int J;
int JJ;
int K;
int KS;
float SFCWATR;
float CHCKSM;
float DMAX[20];
float ACRT;
float DD;
float DDT;
float DDZ;
float DDZ2;
float DENOM;
float DENOM2;
float DICE;
float DSMDZ;
float DSMDZ2;
float DT1;
float FCR;
float INFMAX;
float MXSMC;
float MXSMC2;
float NUMER;
float PDDUM;
float PX;
float SICEMAX;
float SLOPX;
float SMCAV;
float SSTT;
float SUM;
float VAL;
float WCND;
float WCND2;
float WDF;
float WDF2;
int CVFRZ = 3;
 // module_sf_noahlsm.F:3669
IOHINF = 1; // module_sf_noahlsm.F:3714
SICEMAX = 0x0.0p+0f; // module_sf_noahlsm.F:3715
for (KS=1; KS<=NSOIL; KS+=1) { // module_sf_noahlsm.F:3716
if ((SICE[(KS)-(1)] > SICEMAX)) { SICEMAX = SICE[(KS)-(1)]; } // module_sf_noahlsm.F:3717
} // module_sf_noahlsm.F:3721
PDDUM = PCPDRP; // module_sf_noahlsm.F:3732
RUNOFF1 = 0x0.0p+0f; // module_sf_noahlsm.F:3733
if ((PCPDRP != 0x0.0p+0f)) { // module_sf_noahlsm.F:3748
DT1 = DV(DT,0x1.5180000000000p+16f); // module_sf_noahlsm.F:3750
SMCAV = SU(SMCMAX,SMCWLT); // module_sf_noahlsm.F:3751
DMAX[(1)-(1)] = MU((-ZSOIL[(1)-(1)]),SMCAV); // module_sf_noahlsm.F:3756
DICE = MU((-ZSOIL[(1)-(1)]),SICE[(1)-(1)]); // module_sf_noahlsm.F:3758
DMAX[(1)-(1)] = MU(DMAX[(1)-(1)],SU(0x1.0000000000000p+0f,DV(SU(AD(SH2OA[(1)-(1)],SICE[(1)-(1)]),SMCWLT),SMCAV))); // module_sf_noahlsm.F:3759
DD = DMAX[(1)-(1)]; // module_sf_noahlsm.F:3762
for (KS=2; KS<=NSOIL; KS+=1) { // module_sf_noahlsm.F:3767
DICE = AD(DICE,MU(SU(ZSOIL[((KS - 1))-(1)],ZSOIL[(KS)-(1)]),SICE[(KS)-(1)])); // module_sf_noahlsm.F:3769
DMAX[(KS)-(1)] = MU(SU(ZSOIL[((KS - 1))-(1)],ZSOIL[(KS)-(1)]),SMCAV); // module_sf_noahlsm.F:3770
DMAX[(KS)-(1)] = MU(DMAX[(KS)-(1)],SU(0x1.0000000000000p+0f,DV(SU(AD(SH2OA[(KS)-(1)],SICE[(KS)-(1)]),SMCWLT),SMCAV))); // module_sf_noahlsm.F:3771
DD = AD(DD,DMAX[(KS)-(1)]); // module_sf_noahlsm.F:3773
} // module_sf_noahlsm.F:3778
VAL = SU(0x1.0000000000000p+0f,r_exp(MU((-KDT),DT1))); // module_sf_noahlsm.F:3779
DDT = MU(DD,VAL); // module_sf_noahlsm.F:3780
PX = MU(PCPDRP,DT); // module_sf_noahlsm.F:3786
if ((PX < 0x0.0p+0f)) { PX = 0x0.0p+0f; } // module_sf_noahlsm.F:3788
INFMAX = DV(MU(PX,DV(DDT,AD(PX,DDT))),DT); // module_sf_noahlsm.F:3796
FCR = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3797
if ((DICE > 0x1.47ae140000000p-7f)) { // module_sf_noahlsm.F:3798
ACRT = DV(MU(CVFRZ,FRZX),DICE); // module_sf_noahlsm.F:3799
SUM = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3800
IALP1 = (CVFRZ - 1); // module_sf_noahlsm.F:3801
for (J=1; J<=IALP1; J+=1) { // module_sf_noahlsm.F:3802
K = 1; // module_sf_noahlsm.F:3803
for (JJ=(J + 1); JJ<=IALP1; JJ+=1) { // module_sf_noahlsm.F:3804
K = (K * JJ); // module_sf_noahlsm.F:3805
} // module_sf_noahlsm.F:3806
SUM = AD(SUM,DV(noah_powi(ACRT,(CVFRZ - J)),((float)(K)))); // module_sf_noahlsm.F:3807
} // module_sf_noahlsm.F:3808
FCR = SU(0x1.0000000000000p+0f,MU(r_exp((-ACRT)),SUM)); // module_sf_noahlsm.F:3809
} // module_sf_noahlsm.F:3810
INFMAX = MU(INFMAX,FCR); // module_sf_noahlsm.F:3818
MXSMC = SH2OA[(1)-(1)]; // module_sf_noahlsm.F:3820
{  noah_wdfcnd(tables,WDF,WCND,MXSMC,SMCMAX,BEXP,DKSAT,DWSAT,SICEMAX); if(tables.error)return; } // module_sf_noahlsm.F:3821
INFMAX = noah_max(INFMAX,WCND); // module_sf_noahlsm.F:3823
INFMAX = noah_min(INFMAX,DV(PX,DT)); // module_sf_noahlsm.F:3825
if ((PCPDRP > INFMAX)) { // module_sf_noahlsm.F:3833
RUNOFF1 = SU(PCPDRP,INFMAX); // module_sf_noahlsm.F:3834
INFXS1RT = MU(MU(RUNOFF1,DT),0x1.f400000000000p+9f); // module_sf_noahlsm.F:3836
PDDUM = INFMAX; // module_sf_noahlsm.F:3837
} // module_sf_noahlsm.F:3838
} // module_sf_noahlsm.F:3845
MXSMC = SH2OA[(1)-(1)]; // module_sf_noahlsm.F:3847
{  noah_wdfcnd(tables,WDF,WCND,MXSMC,SMCMAX,BEXP,DKSAT,DWSAT,SICEMAX); if(tables.error)return; } // module_sf_noahlsm.F:3848
DDZ = DV(0x1.0000000000000p+0f,MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm.F:3853
AI[(1)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:3854
BI[(1)-(1)] = DV(MU(WDF,DDZ),(-ZSOIL[(1)-(1)])); // module_sf_noahlsm.F:3855
CI[(1)-(1)] = (-BI[(1)-(1)]); // module_sf_noahlsm.F:3861
DSMDZ = DV(SU(SH2O[(1)-(1)],SH2O[(2)-(1)]),MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm.F:3862
RHSTT[(1)-(1)] = DV(AD(AD(SU(AD(MU(WDF,DSMDZ),WCND),PDDUM),EDIR),ET[(1)-(1)]),ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:3863
SSTT = AD(AD(AD(MU(WDF,DSMDZ),WCND),EDIR),ET[(1)-(1)]); // module_sf_noahlsm.F:3868
DDZ2 = 0x0.0p+0f; // module_sf_noahlsm.F:3873
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3874
DENOM2 = SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]); // module_sf_noahlsm.F:3875
if ((K != NSOIL)) { // module_sf_noahlsm.F:3876
SLOPX = 0x1.0000000000000p+0f; // module_sf_noahlsm.F:3883
MXSMC2 = SH2OA[(K)-(1)]; // module_sf_noahlsm.F:3885
{  noah_wdfcnd(tables,WDF2,WCND2,MXSMC2,SMCMAX,BEXP,DKSAT,DWSAT,SICEMAX); if(tables.error)return; } // module_sf_noahlsm.F:3886
DENOM = SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)]); // module_sf_noahlsm.F:3891
DSMDZ2 = DV(SU(SH2O[(K)-(1)],SH2O[((K + 1))-(1)]),MU(DENOM,0x1.0000000000000p-1f)); // module_sf_noahlsm.F:3896
DDZ2 = DV(0x1.0000000000000p+1f,DENOM); // module_sf_noahlsm.F:3897
CI[(K)-(1)] = DV(MU((-WDF2),DDZ2),DENOM2); // module_sf_noahlsm.F:3898
} else { // module_sf_noahlsm.F:3900
SLOPX = SLOPE; // module_sf_noahlsm.F:3909
{  noah_wdfcnd(tables,WDF2,WCND2,SH2OA[(NSOIL)-(1)],SMCMAX,BEXP,DKSAT,DWSAT,SICEMAX); if(tables.error)return; } // module_sf_noahlsm.F:3910
DSMDZ2 = 0x0.0p+0f; // module_sf_noahlsm.F:3920
CI[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:3921
} // module_sf_noahlsm.F:3925
NUMER = AD(SU(SU(AD(MU(WDF2,DSMDZ2),MU(SLOPX,WCND2)),MU(WDF,DSMDZ)),WCND),ET[(K)-(1)]); // module_sf_noahlsm.F:3926
RHSTT[(K)-(1)] = DV(NUMER,(-DENOM2)); // module_sf_noahlsm.F:3932
AI[(K)-(1)] = DV(MU((-WDF),DDZ),DENOM2); // module_sf_noahlsm.F:3933
BI[(K)-(1)] = (-AD(AI[(K)-(1)],CI[(K)-(1)])); // module_sf_noahlsm.F:3939
if ((K == NSOIL)) { // module_sf_noahlsm.F:3940
RUNOFF2 = MU(SLOPX,WCND2); // module_sf_noahlsm.F:3941
} // module_sf_noahlsm.F:3942
if ((K != NSOIL)) { // module_sf_noahlsm.F:3943
WDF = WDF2; // module_sf_noahlsm.F:3944
WCND = WCND2; // module_sf_noahlsm.F:3945
DSMDZ = DSMDZ2; // module_sf_noahlsm.F:3946
DDZ = DDZ2; // module_sf_noahlsm.F:3947
} // module_sf_noahlsm.F:3948
} // module_sf_noahlsm.F:3949
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:3954-4081
__device__ void noah_sstep(NoahTables& tables, float* SH2OOUT, float* SH2OIN, float& CMC, float* RHSTT, float& RHSCT, float& DT, int& NSOIL, float& SMCMAX, float& CMCMAX, float& RUNOFF3, float* ZSOIL, float* SMC, float* SICE, float* AI, float* BI, float* CI, float& INFXS1RT){
int I;
int K;
int KK11;
float AVAIL;
float RHSTTIN[20];
float CIIN[20];
float DDZ;
float STOT;
float WPLUS;
 // module_sf_noahlsm.F:3964
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3987
RHSTT[(K)-(1)] = MU(RHSTT[(K)-(1)],DT); // module_sf_noahlsm.F:3988
AI[(K)-(1)] = MU(AI[(K)-(1)],DT); // module_sf_noahlsm.F:3989
BI[(K)-(1)] = AD(0x1.0000000000000p+0f,MU(BI[(K)-(1)],DT)); // module_sf_noahlsm.F:3990
CI[(K)-(1)] = MU(CI[(K)-(1)],DT); // module_sf_noahlsm.F:3991
} // module_sf_noahlsm.F:3992
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3996
RHSTTIN[(K)-(1)] = RHSTT[(K)-(1)]; // module_sf_noahlsm.F:3997
} // module_sf_noahlsm.F:3998
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:3999
CIIN[(K)-(1)] = CI[(K)-(1)]; // module_sf_noahlsm.F:4000
} // module_sf_noahlsm.F:4001
{  noah_rosr12(tables,CI,AI,BI,CIIN,RHSTTIN,RHSTT,NSOIL); if(tables.error)return; } // module_sf_noahlsm.F:4005
WPLUS = 0x0.0p+0f; // module_sf_noahlsm.F:4011
RUNOFF3 = 0x0.0p+0f; // module_sf_noahlsm.F:4012
DDZ = (-ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:4014
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:4015
if ((K != 1)) { DDZ = SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]); } // module_sf_noahlsm.F:4016
SH2OOUT[(K)-(1)] = AD(AD(SH2OIN[(K)-(1)],CI[(K)-(1)]),DV(WPLUS,DDZ)); // module_sf_noahlsm.F:4017
STOT = AD(SH2OOUT[(K)-(1)],SICE[(K)-(1)]); // module_sf_noahlsm.F:4018
if ((STOT > SMCMAX)) { // module_sf_noahlsm.F:4019
if ((K == 1)) { // module_sf_noahlsm.F:4020
DDZ = (-ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:4021
} else { // module_sf_noahlsm.F:4022
KK11 = (K - 1); // module_sf_noahlsm.F:4023
DDZ = AD((-ZSOIL[(K)-(1)]),ZSOIL[(KK11)-(1)]); // module_sf_noahlsm.F:4024
} // module_sf_noahlsm.F:4025
WPLUS = MU(SU(STOT,SMCMAX),DDZ); // module_sf_noahlsm.F:4026
} else { // module_sf_noahlsm.F:4027
WPLUS = 0x0.0p+0f; // module_sf_noahlsm.F:4028
} // module_sf_noahlsm.F:4029
SMC[(K)-(1)] = noah_max(noah_min(STOT,SMCMAX),0x1.47ae140000000p-6f); // module_sf_noahlsm.F:4030
SH2OOUT[(K)-(1)] = noah_max(SU(SMC[(K)-(1)],SICE[(K)-(1)]),0x0.0p+0f); // module_sf_noahlsm.F:4031
} // module_sf_noahlsm.F:4032
RUNOFF3 = WPLUS; // module_sf_noahlsm.F:4075
CMC = AD(CMC,MU(DT,RHSCT)); // module_sf_noahlsm.F:4076
if ((CMC < 0x1.79ca100000000p-67f)) { CMC = 0x0.0p+0f; } // module_sf_noahlsm.F:4077
CMC = noah_min(CMC,CMCMAX); // module_sf_noahlsm.F:4078
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4084-4124
__device__ void noah_tbnd(NoahTables& tables, float& TU, float& TB, float* ZSOIL, float& ZBOT, int& K, int& NSOIL, float& TBND1){
float ZB;
float ZUP;
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm.F:4092
if ((K == 1)) { // module_sf_noahlsm.F:4104
ZUP = 0x0.0p+0f; // module_sf_noahlsm.F:4105
} else { // module_sf_noahlsm.F:4106
ZUP = ZSOIL[((K - 1))-(1)]; // module_sf_noahlsm.F:4107
} // module_sf_noahlsm.F:4108
if ((K == NSOIL)) { // module_sf_noahlsm.F:4113
ZB = SU(MU(0x1.0000000000000p+1f,ZBOT),ZSOIL[(K)-(1)]); // module_sf_noahlsm.F:4114
} else { // module_sf_noahlsm.F:4115
ZB = ZSOIL[((K + 1))-(1)]; // module_sf_noahlsm.F:4116
} // module_sf_noahlsm.F:4117
TBND1 = AD(TU,DV(MU(SU(TB,TU),SU(ZUP,ZSOIL[(K)-(1)])),SU(ZUP,ZB))); // module_sf_noahlsm.F:4122
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4128-4251
__device__ void noah_tdfcnd(NoahTables& tables, float& DF, float& SMC, float& QZ, float& SMCMAX, float& SH2O, float& BEXP, float& PSISAT, int& SOILTYP, int& OPT_THCND){
float AKE;
float GAMMD;
float THKDRY;
float THKICE;
float THKO;
float THKQTZ;
float THKSAT;
float THKS;
float THKW;
float SATRATIO;
float XU;
float XUNFROZ;
float AKEI;
float AKEL;
float PSIF;
float PF;
 // module_sf_noahlsm.F:4139
if (((OPT_THCND == 1) || ((OPT_THCND == 2) && ((SOILTYP != 4) && (SOILTYP != 3))))) { // module_sf_noahlsm.F:4177
SATRATIO = DV(SMC,SMCMAX); // module_sf_noahlsm.F:4184
THKICE = 0x1.19999a0000000p+1f; // module_sf_noahlsm.F:4186
THKW = 0x1.23d70a0000000p-1f; // module_sf_noahlsm.F:4188
THKO = 0x1.0000000000000p+1f; // module_sf_noahlsm.F:4191
THKQTZ = 0x1.eccccc0000000p+2f; // module_sf_noahlsm.F:4193
THKS = MU(r_pow(THKQTZ,QZ),r_pow(THKO,SU(0x1.0000000000000p+0f,QZ))); // module_sf_noahlsm.F:4195
XUNFROZ = DV(SH2O,SMC); // module_sf_noahlsm.F:4198
XU = MU(XUNFROZ,SMCMAX); // module_sf_noahlsm.F:4200
THKSAT = MU(MU(r_pow(THKS,SU(0x1.0000000000000p+0f,SMCMAX)),r_pow(THKICE,SU(SMCMAX,XU))),r_pow(THKW,XU)); // module_sf_noahlsm.F:4203
GAMMD = MU(SU(0x1.0000000000000p+0f,SMCMAX),0x1.5180000000000p+11f); // module_sf_noahlsm.F:4207
THKDRY = DV(AD(MU(0x1.147ae20000000p-3f,GAMMD),0x1.02cccc0000000p+6f),SU(0x1.5180000000000p+11f,MU(0x1.e4dd300000000p-1f,GAMMD))); // module_sf_noahlsm.F:4210
AKEI = SATRATIO; // module_sf_noahlsm.F:4212
if ((SATRATIO > 0x1.99999a0000000p-4f)) { // module_sf_noahlsm.F:4220
AKEL = AD(r_log10(SATRATIO),0x1.0000000000000p+0f); // module_sf_noahlsm.F:4222
} else { // module_sf_noahlsm.F:4225
AKEL = 0x0.0p+0f; // module_sf_noahlsm.F:4227
} // module_sf_noahlsm.F:4228
AKE = DV(AD(MU(SU(SMC,SH2O),AKEI),MU(SH2O,AKEL)),SMC); // module_sf_noahlsm.F:4229
DF = AD(MU(AKE,SU(THKSAT,THKDRY)),THKDRY); // module_sf_noahlsm.F:4233
} else { // module_sf_noahlsm.F:4235
PSIF = MU(MU(PSISAT,0x1.9000000000000p+6f),r_pow(DV(SMCMAX,SMC),BEXP)); // module_sf_noahlsm.F:4239
PF = r_log10(noah_abs(PSIF)); // module_sf_noahlsm.F:4241
if ((PF <= 0x1.4666660000000p+2f)) { // module_sf_noahlsm.F:4243
DF = MU(0x1.a400000000000p+8f,r_exp((-AD(PF,0x1.59999a0000000p+1f)))); // module_sf_noahlsm.F:4244
} else { // module_sf_noahlsm.F:4245
DF = 0x1.652bd40000000p-3f; // module_sf_noahlsm.F:4246
} // module_sf_noahlsm.F:4247
} // module_sf_noahlsm.F:4249
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4254-4356
__device__ void noah_tmpavg(NoahTables& tables, float& TAVG, float& TUP, float& TM, float& TDN, float* ZSOIL, int& NSOIL, int& K){
float DZ;
float DZH;
float T0 = 0x1.1126660000000p+8f;
float X0;
float XDN;
float XUP;
 // module_sf_noahlsm.F:4264
if ((K == 1)) { // module_sf_noahlsm.F:4283
DZ = (-ZSOIL[(1)-(1)]); // module_sf_noahlsm.F:4284
} else { // module_sf_noahlsm.F:4285
DZ = SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]); // module_sf_noahlsm.F:4286
} // module_sf_noahlsm.F:4287
DZH = MU(DZ,0x1.0000000000000p-1f); // module_sf_noahlsm.F:4289
if ((TUP < T0)) { // module_sf_noahlsm.F:4290
if ((TM < T0)) { // module_sf_noahlsm.F:4291
if ((TDN < T0)) { // module_sf_noahlsm.F:4295
TAVG = DV(AD(AD(TUP,MU(0x1.0000000000000p+1f,TM)),TDN),0x1.0000000000000p+2f); // module_sf_noahlsm.F:4296
} else { // module_sf_noahlsm.F:4300
X0 = DV(MU(SU(T0,TM),DZH),SU(TDN,TM)); // module_sf_noahlsm.F:4301
TAVG = DV(MU(0x1.0000000000000p-1f,AD(AD(MU(TUP,DZH),MU(TM,AD(DZH,X0))),MU(T0,SU(MU(0x1.0000000000000p+1f,DZH),X0)))),DZ); // module_sf_noahlsm.F:4302
} // module_sf_noahlsm.F:4304
} else { // module_sf_noahlsm.F:4305
if ((TDN < T0)) { // module_sf_noahlsm.F:4309
XUP = DV(MU(SU(T0,TUP),DZH),SU(TM,TUP)); // module_sf_noahlsm.F:4310
XDN = SU(DZH,DV(MU(SU(T0,TM),DZH),SU(TDN,TM))); // module_sf_noahlsm.F:4311
TAVG = DV(MU(0x1.0000000000000p-1f,AD(AD(MU(TUP,XUP),MU(T0,SU(SU(MU(0x1.0000000000000p+1f,DZ),XUP),XDN))),MU(TDN,XDN))),DZ); // module_sf_noahlsm.F:4312
} else { // module_sf_noahlsm.F:4317
XUP = DV(MU(SU(T0,TUP),DZH),SU(TM,TUP)); // module_sf_noahlsm.F:4318
TAVG = DV(MU(0x1.0000000000000p-1f,AD(MU(TUP,XUP),MU(T0,SU(MU(0x1.0000000000000p+1f,DZ),XUP)))),DZ); // module_sf_noahlsm.F:4319
} // module_sf_noahlsm.F:4320
} // module_sf_noahlsm.F:4321
} else { // module_sf_noahlsm.F:4322
if ((TM < T0)) { // module_sf_noahlsm.F:4323
if ((TDN < T0)) { // module_sf_noahlsm.F:4327
XUP = SU(DZH,DV(MU(SU(T0,TUP),DZH),SU(TM,TUP))); // module_sf_noahlsm.F:4328
TAVG = DV(MU(0x1.0000000000000p-1f,AD(AD(MU(T0,SU(DZ,XUP)),MU(TM,AD(DZH,XUP))),MU(TDN,DZH))),DZ); // module_sf_noahlsm.F:4329
} else { // module_sf_noahlsm.F:4334
XUP = SU(DZH,DV(MU(SU(T0,TUP),DZH),SU(TM,TUP))); // module_sf_noahlsm.F:4335
XDN = DV(MU(SU(T0,TM),DZH),SU(TDN,TM)); // module_sf_noahlsm.F:4336
TAVG = DV(MU(0x1.0000000000000p-1f,AD(MU(T0,SU(SU(MU(0x1.0000000000000p+1f,DZ),XUP),XDN)),MU(TM,AD(XUP,XDN)))),DZ); // module_sf_noahlsm.F:4337
} // module_sf_noahlsm.F:4339
} else { // module_sf_noahlsm.F:4340
if ((TDN < T0)) { // module_sf_noahlsm.F:4344
XDN = SU(DZH,DV(MU(SU(T0,TM),DZH),SU(TDN,TM))); // module_sf_noahlsm.F:4345
TAVG = DV(AD(MU(T0,SU(DZ,XDN)),MU(MU(0x1.0000000000000p-1f,AD(T0,TDN)),XDN)),DZ); // module_sf_noahlsm.F:4346
} else { // module_sf_noahlsm.F:4350
TAVG = DV(AD(AD(TUP,MU(0x1.0000000000000p+1f,TM)),TDN),0x1.0000000000000p+2f); // module_sf_noahlsm.F:4351
} // module_sf_noahlsm.F:4352
} // module_sf_noahlsm.F:4353
} // module_sf_noahlsm.F:4354
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4359-4462
__device__ void noah_transp(NoahTables& tables, float* ET, int& NSOIL, float& ETP1, float* SMC, float& CMC, float* ZSOIL, float& SHDFAC, float& SMCWLT, float& CMCMAX, float& PC, float& CFACTR, float& SMCREF, float& SFCTMP, float& Q2, int& NROOT, float* RTDIS){
int I;
int K;
float DENOM;
float ETP1A;
float GX[20];
float RTX;
float SGX;
 // module_sf_noahlsm.F:4368
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm.F:4398
ET[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm.F:4399
} // module_sf_noahlsm.F:4407
if ((CMC != 0x0.0p+0f)) { // module_sf_noahlsm.F:4408
ETP1A = MU(MU(MU(SHDFAC,PC),ETP1),SU(0x1.0000000000000p+0f,r_pow(DV(CMC,CMCMAX),CFACTR))); // module_sf_noahlsm.F:4409
} else { // module_sf_noahlsm.F:4410
ETP1A = MU(MU(SHDFAC,PC),ETP1); // module_sf_noahlsm.F:4411
} // module_sf_noahlsm.F:4412
SGX = 0x0.0p+0f; // module_sf_noahlsm.F:4413
for (I=1; I<=NROOT; I+=1) { // module_sf_noahlsm.F:4414
GX[(I)-(1)] = DV(SU(SMC[(I)-(1)],SMCWLT),SU(SMCREF,SMCWLT)); // module_sf_noahlsm.F:4415
GX[(I)-(1)] = noah_max(noah_min(GX[(I)-(1)],0x1.0000000000000p+0f),0x0.0p+0f); // module_sf_noahlsm.F:4416
SGX = AD(SGX,GX[(I)-(1)]); // module_sf_noahlsm.F:4417
} // module_sf_noahlsm.F:4418
SGX = DV(SGX,NROOT); // module_sf_noahlsm.F:4420
DENOM = 0x0.0p+0f; // module_sf_noahlsm.F:4421
for (I=1; I<=NROOT; I+=1) { // module_sf_noahlsm.F:4422
RTX = SU(AD(RTDIS[(I)-(1)],GX[(I)-(1)]),SGX); // module_sf_noahlsm.F:4423
GX[(I)-(1)] = MU(GX[(I)-(1)],noah_max(RTX,0x0.0p+0f)); // module_sf_noahlsm.F:4424
DENOM = AD(DENOM,GX[(I)-(1)]); // module_sf_noahlsm.F:4425
} // module_sf_noahlsm.F:4426
if ((DENOM <= 0x0.0p+0f)) { DENOM = 0x1.0000000000000p+0f; } // module_sf_noahlsm.F:4428
for (I=1; I<=NROOT; I+=1) { // module_sf_noahlsm.F:4429
ET[(I)-(1)] = DV(MU(ETP1A,GX[(I)-(1)]),DENOM); // module_sf_noahlsm.F:4430
} // module_sf_noahlsm.F:4460
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4465-4523
__device__ void noah_wdfcnd(NoahTables& tables, float& WDF, float& WCND, float& SMC, float& SMCMAX, float& BEXP, float& DKSAT, float& DWSAT, float& SICEMAX){
float EXPON;
float FACTR1;
float FACTR2;
float VKWGT;
 // module_sf_noahlsm.F:4473
FACTR1 = DV(0x1.99999a0000000p-5f,SMCMAX); // module_sf_noahlsm.F:4490
FACTR2 = DV(SMC,SMCMAX); // module_sf_noahlsm.F:4495
FACTR1 = noah_min(FACTR1,FACTR2); // module_sf_noahlsm.F:4496
EXPON = AD(BEXP,0x1.0000000000000p+1f); // module_sf_noahlsm.F:4497
WDF = MU(DWSAT,r_pow(FACTR2,EXPON)); // module_sf_noahlsm.F:4511
if ((SICEMAX > 0x0.0p+0f)) { // module_sf_noahlsm.F:4512
VKWGT = DV(0x1.0000000000000p+0f,AD(0x1.0000000000000p+0f,r_pow(MU(0x1.f400000000000p+8f,SICEMAX),0x1.8000000000000p+1f))); // module_sf_noahlsm.F:4513
WDF = AD(MU(VKWGT,WDF),MU(MU(SU(0x1.0000000000000p+0f,VKWGT),DWSAT),r_pow(FACTR1,EXPON))); // module_sf_noahlsm.F:4514
} // module_sf_noahlsm.F:4518
EXPON = AD(MU(0x1.0000000000000p+1f,BEXP),0x1.8000000000000p+1f); // module_sf_noahlsm.F:4519
WCND = MU(DKSAT,r_pow(FACTR2,EXPON)); // module_sf_noahlsm.F:4520
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4526-4760
__device__ void noah_sfcdif_off(NoahTables& tables, float& ZLM, float& Z0, float& THZ0, float& THLM, float& SFCSPD, float& CZIL, float& AKMS, float& AKHS){
float WWST = 0x1.3333340000000p+0f;
float WWST2 = MU(WWST,WWST);
float G = 0x1.39999a0000000p+3f;
float VKRM = 0x1.99999a0000000p-2f;
float EXCM = 0x1.0624de0000000p-10f;
float BETA = 0x1.e573ac0000000p-9f;
float BTG = MU(BETA,G);
float ELFC = MU(VKRM,BTG);
float WOLD = 0x1.3333340000000p-3f;
float WNEW = SU(0x1.0000000000000p+0f,WOLD);
float PIHF = 0x1.921fb60000000p+0f;
float EPSU2 = 0x1.a36e2e0000000p-14f;
float EPSUST = 0x1.1eb8520000000p-4f;
float EPSIT = 0x1.a36e2e0000000p-14f;
float EPSA = 0x1.5798ee0000000p-27f;
float ZTMIN = -0x1.4000000000000p+2f;
float ZTMAX = 0x1.0000000000000p+0f;
float HPBL = 0x1.f400000000000p+9f;
float SQVISC = 0x1.0233340000000p+8f;
float RIC = 0x1.76c8b40000000p-3f;
float RRIC = DV(0x1.0000000000000p+0f,RIC);
float FHNEU = 0x1.99999a0000000p-1f;
float RFC = 0x1.872b020000000p-3f;
float RFAC = DV(RIC,MU(MU(FHNEU,RFC),RFC));
float ZZ;
float XX;
float YY;
float ZILFC;
float ZU;
float ZT;
float RDZ;
float CXCH;
float DTHV;
float DU2;
float BTGH;
float WSTAR2;
float USTAR;
float ZSLU;
float ZSLT;
float RLOGU;
float RLOGT;
float RLMO;
float ZETALT;
float ZETALU;
float ZETAU;
float ZETAT;
float XLU4;
float XLT4;
float XU4;
float XT4;
float XLU;
float XLT;
float XU;
float XT;
float PSMZ;
float SIMM;
float PSHZ;
float SIMH;
float USTARK;
float RLMN;
float RLMA;
int ITRMX = 5;
int ILECH;
int ITR;
 // module_sf_noahlsm.F:4535
auto PSLMU = [&] (float ZZ) { return MU(-0x1.eb851e0000000p-1f,r_log(SU(0x1.0000000000000p+0f,MU(0x1.2000000000000p+2f,ZZ)))); }; // module_sf_noahlsm.F:4570
auto PSLMS = [&] (float ZZ) { return SU(MU(ZZ,RRIC),MU(0x1.09ba5e0000000p+1f,SU(0x1.0000000000000p+0f,DV(0x1.0000000000000p+0f,AD(ZZ,0x1.0000000000000p+0f))))); }; // module_sf_noahlsm.F:4571
auto PSLHU = [&] (float ZZ) { return MU(-0x1.eb851e0000000p-1f,r_log(SU(0x1.0000000000000p+0f,MU(0x1.2000000000000p+2f,ZZ)))); }; // module_sf_noahlsm.F:4572
auto PSLHS = [&] (float ZZ) { return SU(MU(ZZ,RFAC),MU(0x1.09ba5e0000000p+1f,SU(0x1.0000000000000p+0f,DV(0x1.0000000000000p+0f,AD(ZZ,0x1.0000000000000p+0f))))); }; // module_sf_noahlsm.F:4577
auto PSPMU = [&] (float XX) { return SU(AD(SU(MU(-0x1.0000000000000p+1f,r_log(MU(AD(XX,0x1.0000000000000p+0f),0x1.0000000000000p-1f))),r_log(MU(AD(MU(XX,XX),0x1.0000000000000p+0f),0x1.0000000000000p-1f))),MU(0x1.0000000000000p+1f,r_atan(XX))),PIHF); }; // module_sf_noahlsm.F:4578
auto PSPMS = [&] (float YY) { return MU(0x1.4000000000000p+2f,YY); }; // module_sf_noahlsm.F:4581
auto PSPHU = [&] (float XX) { return MU(-0x1.0000000000000p+1f,r_log(MU(AD(MU(XX,XX),0x1.0000000000000p+0f),0x1.0000000000000p-1f))); }; // module_sf_noahlsm.F:4582
auto PSPHS = [&] (float YY) { return MU(0x1.4000000000000p+2f,YY); }; // module_sf_noahlsm.F:4588
ILECH = 0; // module_sf_noahlsm.F:4595
ZILFC = MU(MU((-CZIL),VKRM),SQVISC); // module_sf_noahlsm.F:4598
ZU = Z0; // module_sf_noahlsm.F:4600
RDZ = DV(0x1.0000000000000p+0f,ZLM); // module_sf_noahlsm.F:4601
CXCH = MU(EXCM,RDZ); // module_sf_noahlsm.F:4602
DTHV = SU(THLM,THZ0); // module_sf_noahlsm.F:4603
DU2 = noah_max(MU(SFCSPD,SFCSPD),EPSU2); // module_sf_noahlsm.F:4608
BTGH = MU(BTG,HPBL); // module_sf_noahlsm.F:4610
if ((MU(MU(BTGH,AKHS),DTHV) != 0x0.0p+0f)) { // module_sf_noahlsm.F:4611
WSTAR2 = MU(WWST2,r_pow(noah_abs(MU(MU(BTGH,AKHS),DTHV)),0x1.5555560000000p-1f)); // module_sf_noahlsm.F:4612
} else { // module_sf_noahlsm.F:4613
WSTAR2 = 0x0.0p+0f; // module_sf_noahlsm.F:4614
} // module_sf_noahlsm.F:4615
USTAR = noah_max(__fsqrt_rn(MU(AKMS,__fsqrt_rn(AD(DU2,WSTAR2)))),EPSUST); // module_sf_noahlsm.F:4620
ZT = MU(r_exp(MU(ZILFC,__fsqrt_rn(MU(USTAR,Z0)))),Z0); // module_sf_noahlsm.F:4623
ZSLU = AD(ZLM,ZU); // module_sf_noahlsm.F:4624
ZSLT = AD(ZLM,ZT); // module_sf_noahlsm.F:4629
RLOGU = r_log(DV(ZSLU,ZU)); // module_sf_noahlsm.F:4630
RLOGT = r_log(DV(ZSLT,ZT)); // module_sf_noahlsm.F:4632
RLMO = DV(MU(MU(ELFC,AKHS),DTHV),noah_powi(USTAR,3)); // module_sf_noahlsm.F:4639
for (ITR=1; ITR<=ITRMX; ITR+=1) { // module_sf_noahlsm.F:4643
ZETALT = noah_max(MU(ZSLT,RLMO),ZTMIN); // module_sf_noahlsm.F:4644
RLMO = DV(ZETALT,ZSLT); // module_sf_noahlsm.F:4645
ZETALU = MU(ZSLU,RLMO); // module_sf_noahlsm.F:4646
ZETAU = MU(ZU,RLMO); // module_sf_noahlsm.F:4647
ZETAT = MU(ZT,RLMO); // module_sf_noahlsm.F:4649
if ((ILECH == 0)) { // module_sf_noahlsm.F:4650
if ((RLMO < 0x0.0p+0f)) { // module_sf_noahlsm.F:4651
XLU4 = SU(0x1.0000000000000p+0f,MU(0x1.0000000000000p+4f,ZETALU)); // module_sf_noahlsm.F:4652
XLT4 = SU(0x1.0000000000000p+0f,MU(0x1.0000000000000p+4f,ZETALT)); // module_sf_noahlsm.F:4653
XU4 = SU(0x1.0000000000000p+0f,MU(0x1.0000000000000p+4f,ZETAU)); // module_sf_noahlsm.F:4654
XT4 = SU(0x1.0000000000000p+0f,MU(0x1.0000000000000p+4f,ZETAT)); // module_sf_noahlsm.F:4656
XLU = __fsqrt_rn(__fsqrt_rn(XLU4)); // module_sf_noahlsm.F:4657
XLT = __fsqrt_rn(__fsqrt_rn(XLT4)); // module_sf_noahlsm.F:4658
XU = __fsqrt_rn(__fsqrt_rn(XU4)); // module_sf_noahlsm.F:4659
XT = __fsqrt_rn(__fsqrt_rn(XT4)); // module_sf_noahlsm.F:4661
PSMZ = PSPMU(XU); // module_sf_noahlsm.F:4667
SIMM = AD(SU(PSPMU(XLU),PSMZ),RLOGU); // module_sf_noahlsm.F:4668
PSHZ = PSPHU(XT); // module_sf_noahlsm.F:4669
SIMH = AD(SU(PSPHU(XLT),PSHZ),RLOGT); // module_sf_noahlsm.F:4670
} else { // module_sf_noahlsm.F:4671
ZETALU = noah_min(ZETALU,ZTMAX); // module_sf_noahlsm.F:4672
ZETALT = noah_min(ZETALT,ZTMAX); // module_sf_noahlsm.F:4673
PSMZ = PSPMS(ZETAU); // module_sf_noahlsm.F:4679
SIMM = AD(SU(PSPMS(ZETALU),PSMZ),RLOGU); // module_sf_noahlsm.F:4680
PSHZ = PSPHS(ZETAT); // module_sf_noahlsm.F:4681
SIMH = AD(SU(PSPHS(ZETALT),PSHZ),RLOGT); // module_sf_noahlsm.F:4682
} // module_sf_noahlsm.F:4683
} else { // module_sf_noahlsm.F:4687
if ((RLMO < 0x0.0p+0f)) { // module_sf_noahlsm.F:4688
PSMZ = PSLMU(ZETAU); // module_sf_noahlsm.F:4694
SIMM = AD(SU(PSLMU(ZETALU),PSMZ),RLOGU); // module_sf_noahlsm.F:4695
PSHZ = PSLHU(ZETAT); // module_sf_noahlsm.F:4696
SIMH = AD(SU(PSLHU(ZETALT),PSHZ),RLOGT); // module_sf_noahlsm.F:4697
} else { // module_sf_noahlsm.F:4698
ZETALU = noah_min(ZETALU,ZTMAX); // module_sf_noahlsm.F:4699
ZETALT = noah_min(ZETALT,ZTMAX); // module_sf_noahlsm.F:4701
PSMZ = PSLMS(ZETAU); // module_sf_noahlsm.F:4707
SIMM = AD(SU(PSLMS(ZETALU),PSMZ),RLOGU); // module_sf_noahlsm.F:4708
PSHZ = PSLHS(ZETAT); // module_sf_noahlsm.F:4709
SIMH = AD(SU(PSLHS(ZETALT),PSHZ),RLOGT); // module_sf_noahlsm.F:4710
} // module_sf_noahlsm.F:4711
} // module_sf_noahlsm.F:4715
USTAR = noah_max(__fsqrt_rn(MU(AKMS,__fsqrt_rn(AD(DU2,WSTAR2)))),EPSUST); // module_sf_noahlsm.F:4720
ZT = MU(r_exp(MU(ZILFC,__fsqrt_rn(MU(USTAR,Z0)))),Z0); // module_sf_noahlsm.F:4722
ZSLT = AD(ZLM,ZT); // module_sf_noahlsm.F:4723
RLOGT = r_log(DV(ZSLT,ZT)); // module_sf_noahlsm.F:4725
USTARK = MU(USTAR,VKRM); // module_sf_noahlsm.F:4726
AKMS = noah_max(DV(USTARK,SIMM),CXCH); // module_sf_noahlsm.F:4727
AKHS = noah_max(DV(USTARK,SIMH),CXCH); // module_sf_noahlsm.F:4731
if ((MU(MU(BTGH,AKHS),DTHV) != 0x0.0p+0f)) { // module_sf_noahlsm.F:4732
WSTAR2 = MU(WWST2,r_pow(noah_abs(MU(MU(BTGH,AKHS),DTHV)),0x1.5555560000000p-1f)); // module_sf_noahlsm.F:4733
} else { // module_sf_noahlsm.F:4734
WSTAR2 = 0x0.0p+0f; // module_sf_noahlsm.F:4735
} // module_sf_noahlsm.F:4736
RLMN = DV(MU(MU(ELFC,AKHS),DTHV),noah_powi(USTAR,3)); // module_sf_noahlsm.F:4738
RLMA = AD(MU(RLMO,WOLD),MU(RLMN,WNEW)); // module_sf_noahlsm.F:4742
RLMO = RLMA; // module_sf_noahlsm.F:4744
} // module_sf_noahlsm.F:4758
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:35-413
__device__ void gl_sflx_glacial(NoahTables& tables, int& IILOC, int& JJLOC, int& ISICE, float& FFROZP, float& DT, float& ZLVL, int& NSOIL, float* SLDPTH, float& LWDN, float& SOLNET, float& SFCPRS, float& PRCP, float& SFCTMP, float& Q2, float& TH2, float& Q2SAT, float& DQSDT2, float& ALB, float& SNOALB, float& TBOT, float& Z0BRD, float& Z0, float& EMISSI, float& EMBRD, float& T1, float* STC, float& SNOWH, float& SNEQV, float& ALBEDO, float& CH, float& ETA, float& SHEAT, float& ETA_KINEMATIC, float& FDOWN, float& ESNOW, float& DEW, float& ETP, float& SSOIL, float& FLX1, float& FLX2, float& FLX3, float& SNOMLT, float& SNCOVR, float& RUNOFF1, float& Q1, float& SNOTIME1, float& RIBB){
auto& RD=tables.RD;
auto& ILOC=tables.ILOC;
auto& JLOC=tables.JLOC;
auto& CP=tables.CP;
bool FRZGRA;
bool SNOWNG;
int KZ;
float ZSOIL[20];
float DF1;
float DSOIL;
float DTOT;
float FRCSNO;
float FRCSOI;
float PRCP1;
float RCH;
float RR;
float RSNOW;
float SNDENS;
float SNCOND;
float SN_NEW;
float T1V;
float T24;
float T2V;
float TH2V;
float TSNOW;
float PRCPF;
float RHO;
float TFREEZ = 0x1.1126660000000p+8f;
float LVH2O = 0x1.314c400000000p+21f;
float LSUBS = 0x1.5975800000000p+21f;
float R = 0x1.1f0a3e0000000p+8f;
 // module_sf_noahlsm_glacial_only.F:165
ILOC = IILOC; // module_sf_noahlsm_glacial_only.F:209
JLOC = JJLOC; // module_sf_noahlsm_glacial_only.F:210
ZSOIL[(1)-(1)] = (-SLDPTH[(1)-(1)]); // module_sf_noahlsm_glacial_only.F:212
for (KZ=2; KZ<=NSOIL; KZ+=1) { // module_sf_noahlsm_glacial_only.F:213
ZSOIL[(KZ)-(1)] = AD((-SLDPTH[(KZ)-(1)]),ZSOIL[((KZ - 1))-(1)]); // module_sf_noahlsm_glacial_only.F:214
} // module_sf_noahlsm_glacial_only.F:215
if ((SNEQV < 0x1.99999a0000000p-4f)) { // module_sf_noahlsm_glacial_only.F:221
SNEQV = 0x1.99999a0000000p-4f; // module_sf_noahlsm_glacial_only.F:222
SNOWH = 0x1.0000000000000p-1f; // module_sf_noahlsm_glacial_only.F:223
} // module_sf_noahlsm_glacial_only.F:224
SNDENS = DV(SNEQV,SNOWH); // module_sf_noahlsm_glacial_only.F:229
if ((SNDENS > 0x1.0000000000000p+0f)) { // module_sf_noahlsm_glacial_only.F:230
tables.error = 231; return; // module_sf_noahlsm_glacial_only.F:231
} // module_sf_noahlsm_glacial_only.F:232
{  gl_csnow(tables,SNCOND,SNDENS); } // module_sf_noahlsm_glacial_only.F:234
SNOWNG = false; // module_sf_noahlsm_glacial_only.F:242
FRZGRA = false; // module_sf_noahlsm_glacial_only.F:243
if ((PRCP > 0x0.0p+0f)) { // module_sf_noahlsm_glacial_only.F:244
if ((FFROZP > 0x1.0000000000000p-1f)) { // module_sf_noahlsm_glacial_only.F:249
SNOWNG = true; // module_sf_noahlsm_glacial_only.F:250
} else { // module_sf_noahlsm_glacial_only.F:251
if ((T1 <= TFREEZ)) { FRZGRA = true; } // module_sf_noahlsm_glacial_only.F:252
} // module_sf_noahlsm_glacial_only.F:253
} // module_sf_noahlsm_glacial_only.F:254
if ((SNOWNG || FRZGRA)) { // module_sf_noahlsm_glacial_only.F:262
SN_NEW = MU(MU(PRCP,DT),0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:263
SNEQV = AD(SNEQV,SN_NEW); // module_sf_noahlsm_glacial_only.F:264
PRCPF = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:265
{  gl_snow_new(tables,SFCTMP,SN_NEW,SNOWH,SNDENS); } // module_sf_noahlsm_glacial_only.F:271
if ((SNCOVR > 0x1.fae1480000000p-1f)) { // module_sf_noahlsm_glacial_only.F:279
if ((STC[(1)-(1)] < SU(TFREEZ,0x1.4000000000000p+2f))) { SNDENS = 0x1.99999a0000000p-3f; } // module_sf_noahlsm_glacial_only.F:280
if (((SNOWNG && (T1 < 0x1.1100000000000p+8f)) && (SFCTMP < 0x1.1100000000000p+8f))) { SNDENS = 0x1.99999a0000000p-3f; } // module_sf_noahlsm_glacial_only.F:281
} // module_sf_noahlsm_glacial_only.F:282
{  gl_csnow(tables,SNCOND,SNDENS); } // module_sf_noahlsm_glacial_only.F:284
} else { // module_sf_noahlsm_glacial_only.F:290
PRCPF = PRCP; // module_sf_noahlsm_glacial_only.F:291
} // module_sf_noahlsm_glacial_only.F:292
SNCOVR = 0x1.0000000000000p+0f; // module_sf_noahlsm_glacial_only.F:299
{  gl_alcalc(tables,ALB,SNOALB,EMBRD,T1,ALBEDO,EMISSI,DT,SNOWNG,SNOTIME1); } // module_sf_noahlsm_glacial_only.F:305
DF1 = SNCOND; // module_sf_noahlsm_glacial_only.F:311
DSOIL = (-MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm_glacial_only.F:313
DTOT = AD(SNOWH,DSOIL); // module_sf_noahlsm_glacial_only.F:314
FRCSNO = DV(SNOWH,DTOT); // module_sf_noahlsm_glacial_only.F:315
FRCSOI = DV(DSOIL,DTOT); // module_sf_noahlsm_glacial_only.F:319
DF1 = AD(MU(FRCSNO,SNCOND),MU(FRCSOI,DF1)); // module_sf_noahlsm_glacial_only.F:323
if ((DTOT > MU(0x1.0000000000000p+1f,DSOIL))) { // module_sf_noahlsm_glacial_only.F:330
DTOT = MU(0x1.0000000000000p+1f,DSOIL); // module_sf_noahlsm_glacial_only.F:331
} // module_sf_noahlsm_glacial_only.F:332
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm_glacial_only.F:333
{  gl_snowz0(tables,Z0,Z0BRD,SNOWH); } // module_sf_noahlsm_glacial_only.F:340
FDOWN = AD(SOLNET,LWDN); // module_sf_noahlsm_glacial_only.F:347
T2V = MU(SFCTMP,AD(0x1.0000000000000p+0f,MU(0x1.3851ec0000000p-1f,Q2))); // module_sf_noahlsm_glacial_only.F:354
RHO = DV(SFCPRS,MU(RD,T2V)); // module_sf_noahlsm_glacial_only.F:355
RCH = MU(MU(RHO,0x1.f64ccc0000000p+9f),CH); // module_sf_noahlsm_glacial_only.F:356
T24 = MU(MU(MU(SFCTMP,SFCTMP),SFCTMP),SFCTMP); // module_sf_noahlsm_glacial_only.F:357
{  gl_penman(tables,SFCTMP,SFCPRS,CH,TH2,PRCP,FDOWN,T24,SSOIL,Q2,Q2SAT,ETP,RCH,RR,SNOWNG,FRZGRA,DQSDT2,FLX2,EMISSI,T1); } // module_sf_noahlsm_glacial_only.F:366
{  gl_snopac(tables,ETP,ETA,PRCP,PRCPF,SNOWNG,NSOIL,DT,DF1,Q2,T1,SFCTMP,T24,TH2,FDOWN,SSOIL,STC,SFCPRS,RCH,RR,SNEQV,SNDENS,SNOWH,ZSOIL,TBOT,SNOMLT,DEW,FLX1,FLX2,FLX3,ESNOW,EMISSI,RIBB); } // module_sf_noahlsm_glacial_only.F:370
ETA_KINEMATIC = ETP; // module_sf_noahlsm_glacial_only.F:376
Q1 = AD(Q2,DV(MU(ETA_KINEMATIC,CP),RCH)); // module_sf_noahlsm_glacial_only.F:381
SHEAT = MU(DV((-MU(MU(CH,CP),SFCPRS)),MU(R,T2V)),SU(TH2,T1)); // module_sf_noahlsm_glacial_only.F:386
ESNOW = MU(ESNOW,LSUBS); // module_sf_noahlsm_glacial_only.F:391
ETP = MU(ETP,LSUBS); // module_sf_noahlsm_glacial_only.F:392
if ((ETP > 0x0.0p+0f)) { // module_sf_noahlsm_glacial_only.F:393
ETA = ESNOW; // module_sf_noahlsm_glacial_only.F:394
} else { // module_sf_noahlsm_glacial_only.F:395
ETA = ETP; // module_sf_noahlsm_glacial_only.F:396
} // module_sf_noahlsm_glacial_only.F:397
SSOIL = MU(-0x1.0000000000000p+0f,SSOIL); // module_sf_noahlsm_glacial_only.F:404
RUNOFF1 = DV(SNOMLT,DT); // module_sf_noahlsm_glacial_only.F:410
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:416-527
__device__ void gl_alcalc(NoahTables& tables, float& ALB, float& SNOALB, float& EMBRD, float& TSNOW, float& ALBEDO, float& EMISSI, float& DT, bool& SNOWNG, float& SNOTIME1){
auto& EMISSI_S=tables.EMISSI_S;
auto& LVCOEF_DATA=tables.LVCOEF_DATA;
float SNOALB2;
float TM;
float SNOALB1;
float SNACCA = 0x1.e147ae0000000p-1f;
float SNACCB = 0x1.28f5c20000000p-1f;
float SNTHWA = 0x1.a3d70a0000000p-1f;
float SNTHWB = 0x1.d70a3e0000000p-2f;
 // module_sf_noahlsm_glacial_only.F:426
ALBEDO = AD(ALB,SU(SNOALB,ALB)); // module_sf_noahlsm_glacial_only.F:445
EMISSI = AD(EMBRD,SU(EMISSI_S,EMBRD)); // module_sf_noahlsm_glacial_only.F:446
SNOALB1 = AD(SNOALB,MU(LVCOEF_DATA,SU(0x1.b333340000000p-1f,SNOALB))); // module_sf_noahlsm_glacial_only.F:500
SNOALB2 = SNOALB1; // module_sf_noahlsm_glacial_only.F:501
if (SNOWNG) { // module_sf_noahlsm_glacial_only.F:503
SNOTIME1 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:504
} else { // module_sf_noahlsm_glacial_only.F:505
SNOTIME1 = AD(SNOTIME1,DT); // module_sf_noahlsm_glacial_only.F:506
SNOALB2 = MU(SNOALB1,r_pow(SNACCA,r_pow(DV(SNOTIME1,0x1.5180000000000p+16f),SNACCB))); // module_sf_noahlsm_glacial_only.F:508
} // module_sf_noahlsm_glacial_only.F:512
SNOALB2 = noah_max(SNOALB2,ALB); // module_sf_noahlsm_glacial_only.F:514
ALBEDO = AD(ALB,SU(SNOALB2,ALB)); // module_sf_noahlsm_glacial_only.F:515
if ((ALBEDO > SNOALB2)) { ALBEDO = SNOALB2; } // module_sf_noahlsm_glacial_only.F:516
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:530-566
__device__ void gl_csnow(NoahTables& tables, float& SNCOND, float& DSNOW){
float C;
float UNIT = 0x1.dc67e00000000p-4f;
 // module_sf_noahlsm_glacial_only.F:535
C = MU(0x1.4fdf3c0000000p-2f,r_pow(10,MU(0x1.2000000000000p+1f,DSNOW))); // module_sf_noahlsm_glacial_only.F:546
SNCOND = MU(MU(0x1.0000000000000p+1f,UNIT),C); // module_sf_noahlsm_glacial_only.F:563
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:569-711
__device__ void gl_hrtice(NoahTables& tables, float* RHSTS, float* STC, float& TBOT, int& NSOIL, float* ZSOIL, float& YY, float& ZZ1, float& DF1, float* AI, float* BI, float* CI){
int K;
float DDZ;
float DDZ2;
float DENOM;
float DTSDZ;
float DTSDZ2;
float SSOIL;
float HCPCT;
float DF1K;
float DF1N;
float ZMD;
float ZBOT = -0x1.9000000000000p+4f;
 // module_sf_noahlsm_glacial_only.F:580
HCPCT = MU(0x1.e848000000000p+19f,SU(0x1.a388660000000p-1f,MU(0x1.0c154c0000000p-4f,ZSOIL[(1)-(1)]))); // module_sf_noahlsm_glacial_only.F:604
DF1K = DF1; // module_sf_noahlsm_glacial_only.F:605
DDZ = DV(0x1.0000000000000p+0f,MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm_glacial_only.F:619
AI[(1)-(1)] = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:620
CI[(1)-(1)] = DV(MU(DF1,DDZ),MU(ZSOIL[(1)-(1)],HCPCT)); // module_sf_noahlsm_glacial_only.F:621
BI[(1)-(1)] = AD((-CI[(1)-(1)]),DV(DF1,MU(MU(MU(MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),ZSOIL[(1)-(1)]),HCPCT),ZZ1))); // module_sf_noahlsm_glacial_only.F:628
DTSDZ = DV(SU(STC[(1)-(1)],STC[(2)-(1)]),MU(-0x1.0000000000000p-1f,ZSOIL[(2)-(1)])); // module_sf_noahlsm_glacial_only.F:630
SSOIL = DV(MU(DF1,SU(STC[(1)-(1)],YY)),MU(MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)]),ZZ1)); // module_sf_noahlsm_glacial_only.F:631
RHSTS[(1)-(1)] = DV(SU(MU(DF1,DTSDZ),SSOIL),MU(ZSOIL[(1)-(1)],HCPCT)); // module_sf_noahlsm_glacial_only.F:636
DDZ2 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:641
DF1K = DF1; // module_sf_noahlsm_glacial_only.F:642
DF1N = DF1; // module_sf_noahlsm_glacial_only.F:643
for (K=2; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:644
ZMD = MU(0x1.0000000000000p-1f,AD(ZSOIL[(K)-(1)],ZSOIL[((K - 1))-(1)])); // module_sf_noahlsm_glacial_only.F:646
HCPCT = MU(0x1.e848000000000p+19f,SU(0x1.a388660000000p-1f,MU(0x1.0c154c0000000p-3f,ZMD))); // module_sf_noahlsm_glacial_only.F:656
DF1N = SU(0x1.4b17040000000p-2f,MU(0x1.9c97100000000p-4f,ZMD)); // module_sf_noahlsm_glacial_only.F:665
if ((K != NSOIL)) { // module_sf_noahlsm_glacial_only.F:669
DENOM = MU(0x1.0000000000000p-1f,SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)])); // module_sf_noahlsm_glacial_only.F:670
DTSDZ2 = DV(SU(STC[(K)-(1)],STC[((K + 1))-(1)]),DENOM); // module_sf_noahlsm_glacial_only.F:675
DDZ2 = DV(0x1.0000000000000p+1f,SU(ZSOIL[((K - 1))-(1)],ZSOIL[((K + 1))-(1)])); // module_sf_noahlsm_glacial_only.F:676
CI[(K)-(1)] = DV(MU((-DF1N),DDZ2),MU(SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]),HCPCT)); // module_sf_noahlsm_glacial_only.F:677
} else { // module_sf_noahlsm_glacial_only.F:682
DTSDZ2 = DV(SU(STC[(K)-(1)],TBOT),SU(MU(0x1.0000000000000p-1f,AD(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)])),ZBOT)); // module_sf_noahlsm_glacial_only.F:687
CI[(K)-(1)] = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:689
} // module_sf_noahlsm_glacial_only.F:693
DENOM = MU(SU(ZSOIL[(K)-(1)],ZSOIL[((K - 1))-(1)]),HCPCT); // module_sf_noahlsm_glacial_only.F:694
RHSTS[(K)-(1)] = DV(SU(MU(DF1N,DTSDZ2),MU(DF1K,DTSDZ)),DENOM); // module_sf_noahlsm_glacial_only.F:699
AI[(K)-(1)] = DV(MU((-DF1K),DDZ),MU(SU(ZSOIL[((K - 1))-(1)],ZSOIL[(K)-(1)]),HCPCT)); // module_sf_noahlsm_glacial_only.F:700
BI[(K)-(1)] = (-AD(AI[(K)-(1)],CI[(K)-(1)])); // module_sf_noahlsm_glacial_only.F:705
DF1K = DF1N; // module_sf_noahlsm_glacial_only.F:706
DTSDZ = DTSDZ2; // module_sf_noahlsm_glacial_only.F:707
DDZ = DDZ2; // module_sf_noahlsm_glacial_only.F:708
} // module_sf_noahlsm_glacial_only.F:709
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:714-759
__device__ void gl_hstep(NoahTables& tables, float* STCOUT, float* STCIN, float* RHSTS, float& DT, int& NSOIL, float* AI, float* BI, float* CI){
float RHSTSIN[20];
float CIIN[20];
int K;
 // module_sf_noahlsm_glacial_only.F:719
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:733
RHSTS[(K)-(1)] = MU(RHSTS[(K)-(1)],DT); // module_sf_noahlsm_glacial_only.F:734
AI[(K)-(1)] = MU(AI[(K)-(1)],DT); // module_sf_noahlsm_glacial_only.F:735
BI[(K)-(1)] = AD(0x1.0000000000000p+0f,MU(BI[(K)-(1)],DT)); // module_sf_noahlsm_glacial_only.F:736
CI[(K)-(1)] = MU(CI[(K)-(1)],DT); // module_sf_noahlsm_glacial_only.F:737
} // module_sf_noahlsm_glacial_only.F:738
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:742
RHSTSIN[(K)-(1)] = RHSTS[(K)-(1)]; // module_sf_noahlsm_glacial_only.F:743
} // module_sf_noahlsm_glacial_only.F:744
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:745
CIIN[(K)-(1)] = CI[(K)-(1)]; // module_sf_noahlsm_glacial_only.F:746
} // module_sf_noahlsm_glacial_only.F:747
{  noah_rosr12(tables,CI,AI,BI,CIIN,RHSTSIN,RHSTS,NSOIL); if(tables.error)return; } // module_sf_noahlsm_glacial_only.F:751
for (K=1; K<=NSOIL; K+=1) { // module_sf_noahlsm_glacial_only.F:755
STCOUT[(K)-(1)] = AD(STCIN[(K)-(1)],CI[(K)-(1)]); // module_sf_noahlsm_glacial_only.F:756
} // module_sf_noahlsm_glacial_only.F:757
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:762-825
__device__ void gl_penman(NoahTables& tables, float& SFCTMP, float& SFCPRS, float& CH, float& TH2, float& PRCP, float& FDOWN, float& T24, float& SSOIL, float& Q2, float& Q2SAT, float& ETP, float& RCH, float& RR, bool& SNOWNG, bool& FRZGRA, float& DQSDT2, float& FLX2, float& EMISSI, float& T1){
auto& SIGMA=tables.SIGMA;
auto& CPH2O=tables.CPH2O;
auto& CPICE=tables.CPICE;
auto& LSUBF=tables.LSUBF;
float A;
float DELTA;
float FNET;
float RAD;
float ELCP1;
float LVS;
float EPSCA;
float ELCP = 0x1.37199a0000000p+11f;
float LSUBC = 0x1.314c400000000p+21f;
float LSUBS = 0x1.5975800000000p+21f;
 // module_sf_noahlsm_glacial_only.F:771
if ((T1 > 0x1.1126660000000p+8f)) { // module_sf_noahlsm_glacial_only.F:785
ELCP1 = ELCP; // module_sf_noahlsm_glacial_only.F:786
LVS = LSUBC; // module_sf_noahlsm_glacial_only.F:787
} else { // module_sf_noahlsm_glacial_only.F:788
ELCP1 = DV(MU(ELCP,LSUBS),LSUBC); // module_sf_noahlsm_glacial_only.F:789
LVS = LSUBS; // module_sf_noahlsm_glacial_only.F:790
} // module_sf_noahlsm_glacial_only.F:791
DELTA = MU(ELCP1,DQSDT2); // module_sf_noahlsm_glacial_only.F:792
A = MU(ELCP1,SU(Q2SAT,Q2)); // module_sf_noahlsm_glacial_only.F:793
RR = AD(DV(MU(MU(EMISSI,T24),0x1.16505a0000000p-24f),MU(SFCPRS,CH)),0x1.0000000000000p+0f); // module_sf_noahlsm_glacial_only.F:794
if ((!SNOWNG)) { // module_sf_noahlsm_glacial_only.F:800
if ((PRCP > 0x0.0p+0f)) { RR = AD(RR,DV(MU(CPH2O,PRCP),RCH)); } // module_sf_noahlsm_glacial_only.F:801
} else { // module_sf_noahlsm_glacial_only.F:802
RR = AD(RR,DV(MU(CPICE,PRCP),RCH)); // module_sf_noahlsm_glacial_only.F:803
} // module_sf_noahlsm_glacial_only.F:804
if (FRZGRA) { // module_sf_noahlsm_glacial_only.F:810
FLX2 = MU((-LSUBF),PRCP); // module_sf_noahlsm_glacial_only.F:811
} else { // module_sf_noahlsm_glacial_only.F:812
FLX2 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:813
} // module_sf_noahlsm_glacial_only.F:814
FNET = SU(SU(SU(FDOWN,MU(MU(EMISSI,SIGMA),T24)),SSOIL),FLX2); // module_sf_noahlsm_glacial_only.F:815
RAD = SU(AD(DV(FNET,RCH),TH2),SFCTMP); // module_sf_noahlsm_glacial_only.F:820
EPSCA = DV(AD(MU(A,RR),MU(RAD,DELTA)),AD(DELTA,RR)); // module_sf_noahlsm_glacial_only.F:821
ETP = DV(MU(EPSCA,RCH),LVS); // module_sf_noahlsm_glacial_only.F:822
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:828-857
__device__ void gl_shflx(NoahTables& tables, float* STC, int& NSOIL, float& DT, float& YY, float& ZZ1, float* ZSOIL, float& TBOT, float& DF1){
float AI[20];
float BI[20];
float CI[20];
float STCF[20];
float RHSTS[20];
int I;
float T0 = 0x1.1126660000000p+8f;
 // module_sf_noahlsm_glacial_only.F:834
{  gl_hrtice(tables,RHSTS,STC,TBOT,NSOIL,ZSOIL,YY,ZZ1,DF1,AI,BI,CI); } // module_sf_noahlsm_glacial_only.F:849
{  gl_hstep(tables,STCF,STC,RHSTS,DT,NSOIL,AI,BI,CI); } // module_sf_noahlsm_glacial_only.F:851
for (I=1; I<=NSOIL; I+=1) { // module_sf_noahlsm_glacial_only.F:853
STC[(I)-(1)] = STCF[(I)-(1)]; // module_sf_noahlsm_glacial_only.F:854
} // module_sf_noahlsm_glacial_only.F:855
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:860-1077
__device__ void gl_snopac(NoahTables& tables, float& ETP, float& ETA, float& PRCP, float& PRCPF, bool& SNOWNG, int& NSOIL, float& DT, float& DF1, float& Q2, float& T1, float& SFCTMP, float& T24, float& TH2, float& FDOWN, float& SSOIL, float* STC, float& SFCPRS, float& RCH, float& RR, float& SNEQV, float& SNDENS, float& SNOWH, float* ZSOIL, float& TBOT, float& SNOMLT, float& DEW, float& FLX1, float& FLX2, float& FLX3, float& ESNOW, float& EMISSI, float& RIBB){
auto& SIGMA=tables.SIGMA;
auto& CPH2O=tables.CPH2O;
auto& CPICE=tables.CPICE;
auto& LSUBF=tables.LSUBF;
float ET1[20];
int K;
float DENOM;
float DSOIL;
float DTOT;
float ESDFLX;
float ESNOW1;
float ESNOW2;
float ETA1;
float ETP1;
float ETP2;
float ETP3;
float ETANRG;
float EX;
float FRCSNO;
float FRCSOI;
float PRCP1;
float QSAT;
float RSNOW;
float SEH;
float SNCOND;
float T12;
float T12A;
float T12B;
float T14;
float YY;
float ZZ1;
float ESDMIN = 0x1.0c6f7a0000000p-20f;
float LSUBC = 0x1.314c400000000p+21f;
float LSUBS = 0x1.5975800000000p+21f;
float TFREEZ = 0x1.1126660000000p+8f;
float SNOEXP = 0x1.0000000000000p+1f;
 // module_sf_noahlsm_glacial_only.F:870
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:907
DEW = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:908
ESNOW = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:909
ESNOW1 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:910
ESNOW2 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:911
PRCP1 = MU(PRCPF,0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:916
if ((ETP <= 0x0.0p+0f)) { // module_sf_noahlsm_glacial_only.F:920
if (((RIBB >= 0x1.99999a0000000p-4f) && (FDOWN > 0x1.2c00000000000p+7f))) { // module_sf_noahlsm_glacial_only.F:921
ETP = DV(AD(DV(noah_min(MU(ETP,SU(0x1.0000000000000p+0f,RIBB)),0x0.0p+0f),0x1.f5c2900000000p-1f),MU(ETP,-0x1.47ae000000000p-6f)),0x1.f5c2900000000p-1f); // module_sf_noahlsm_glacial_only.F:922
} // module_sf_noahlsm_glacial_only.F:923
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:924
DEW = (-ETP1); // module_sf_noahlsm_glacial_only.F:925
ESNOW2 = MU(ETP1,DT); // module_sf_noahlsm_glacial_only.F:926
ETANRG = MU(ETP,LSUBS); // module_sf_noahlsm_glacial_only.F:927
} else { // module_sf_noahlsm_glacial_only.F:928
ETP1 = MU(ETP,0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:929
ESNOW = ETP; // module_sf_noahlsm_glacial_only.F:930
ESNOW1 = MU(ESNOW,0x1.0624de0000000p-10f); // module_sf_noahlsm_glacial_only.F:931
ESNOW2 = MU(ESNOW1,DT); // module_sf_noahlsm_glacial_only.F:932
ETANRG = MU(ESNOW,LSUBS); // module_sf_noahlsm_glacial_only.F:933
} // module_sf_noahlsm_glacial_only.F:934
FLX1 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:942
if (SNOWNG) { // module_sf_noahlsm_glacial_only.F:943
FLX1 = MU(MU(CPICE,PRCP),SU(T1,SFCTMP)); // module_sf_noahlsm_glacial_only.F:944
} else { // module_sf_noahlsm_glacial_only.F:945
if ((PRCP > 0x0.0p+0f)) { FLX1 = MU(MU(CPH2O,PRCP),SU(T1,SFCTMP)); } // module_sf_noahlsm_glacial_only.F:946
} // module_sf_noahlsm_glacial_only.F:947
DSOIL = (-MU(0x1.0000000000000p-1f,ZSOIL[(1)-(1)])); // module_sf_noahlsm_glacial_only.F:956
DTOT = AD(SNOWH,DSOIL); // module_sf_noahlsm_glacial_only.F:957
DENOM = AD(0x1.0000000000000p+0f,DV(DF1,MU(MU(DTOT,RR),RCH))); // module_sf_noahlsm_glacial_only.F:958
T12A = DV(SU(SU(AD(DV(SU(SU(SU(FDOWN,FLX1),FLX2),MU(MU(EMISSI,SIGMA),T24)),RCH),TH2),SFCTMP),DV(ETANRG,RCH)),RR); // module_sf_noahlsm_glacial_only.F:959
T12B = DV(MU(DF1,STC[(1)-(1)]),MU(MU(DTOT,RR),RCH)); // module_sf_noahlsm_glacial_only.F:961
T12 = DV(AD(AD(SFCTMP,T12A),T12B),DENOM); // module_sf_noahlsm_glacial_only.F:963
if ((T12 <= TFREEZ)) { // module_sf_noahlsm_glacial_only.F:964
T1 = T12; // module_sf_noahlsm_glacial_only.F:978
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm_glacial_only.F:979
SNEQV = noah_max(0x0.0p+0f,SU(SNEQV,ESNOW2)); // module_sf_noahlsm_glacial_only.F:980
FLX3 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:981
EX = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:982
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:983
} else { // module_sf_noahlsm_glacial_only.F:984
T1 = TFREEZ; // module_sf_noahlsm_glacial_only.F:1000
if ((DTOT > MU(0x1.0000000000000p+1f,DSOIL))) { // module_sf_noahlsm_glacial_only.F:1001
DTOT = MU(0x1.0000000000000p+1f,DSOIL); // module_sf_noahlsm_glacial_only.F:1002
} // module_sf_noahlsm_glacial_only.F:1003
SSOIL = DV(MU(DF1,SU(T1,STC[(1)-(1)])),DTOT); // module_sf_noahlsm_glacial_only.F:1004
if ((SU(SNEQV,ESNOW2) <= ESDMIN)) { // module_sf_noahlsm_glacial_only.F:1005
SNEQV = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1006
EX = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1007
SNOMLT = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1008
FLX3 = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1009
} else { // module_sf_noahlsm_glacial_only.F:1014
SNEQV = SU(SNEQV,ESNOW2); // module_sf_noahlsm_glacial_only.F:1015
ETP3 = MU(ETP,LSUBC); // module_sf_noahlsm_glacial_only.F:1016
SEH = MU(RCH,SU(T1,TH2)); // module_sf_noahlsm_glacial_only.F:1017
T14 = MU(MU(T1,T1),MU(T1,T1)); // module_sf_noahlsm_glacial_only.F:1018
FLX3 = SU(SU(SU(SU(SU(SU(FDOWN,FLX1),FLX2),MU(MU(EMISSI,SIGMA),T14)),SSOIL),SEH),ETANRG); // module_sf_noahlsm_glacial_only.F:1019
if ((FLX3 <= 0x0.0p+0f)) { FLX3 = 0x0.0p+0f; } // module_sf_noahlsm_glacial_only.F:1020
EX = DV(MU(FLX3,0x1.0624de0000000p-10f),LSUBF); // module_sf_noahlsm_glacial_only.F:1021
SNOMLT = MU(EX,DT); // module_sf_noahlsm_glacial_only.F:1022
if ((SU(SNEQV,SNOMLT) >= ESDMIN)) { // module_sf_noahlsm_glacial_only.F:1027
SNEQV = SU(SNEQV,SNOMLT); // module_sf_noahlsm_glacial_only.F:1028
} else { // module_sf_noahlsm_glacial_only.F:1029
EX = DV(SNEQV,DT); // module_sf_noahlsm_glacial_only.F:1033
FLX3 = MU(MU(EX,0x1.f400000000000p+9f),LSUBF); // module_sf_noahlsm_glacial_only.F:1034
SNOMLT = SNEQV; // module_sf_noahlsm_glacial_only.F:1035
SNEQV = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1037
} // module_sf_noahlsm_glacial_only.F:1038
} // module_sf_noahlsm_glacial_only.F:1039
} // module_sf_noahlsm_glacial_only.F:1047
ZZ1 = 0x1.0000000000000p+0f; // module_sf_noahlsm_glacial_only.F:1056
YY = SU(STC[(1)-(1)],DV(MU(MU(MU(0x1.0000000000000p-1f,SSOIL),ZSOIL[(1)-(1)]),ZZ1),DF1)); // module_sf_noahlsm_glacial_only.F:1057
{  gl_shflx(tables,STC,NSOIL,DT,YY,ZZ1,ZSOIL,TBOT,DF1); } // module_sf_noahlsm_glacial_only.F:1062
if ((SNEQV >= 0x1.99999a0000000p-4f)) { // module_sf_noahlsm_glacial_only.F:1068
{  gl_snowpack(tables,SNEQV,DT,SNOWH,SNDENS,T1,YY); } // module_sf_noahlsm_glacial_only.F:1069
} else { // module_sf_noahlsm_glacial_only.F:1070
SNEQV = 0x1.99999a0000000p-4f; // module_sf_noahlsm_glacial_only.F:1071
SNOWH = 0x1.0000000000000p-1f; // module_sf_noahlsm_glacial_only.F:1072
} // module_sf_noahlsm_glacial_only.F:1075
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1080-1203
__device__ void gl_snowpack(NoahTables& tables, float& SNEQV, float& DTSEC, float& SNOWH, float& SNDENS, float& TSNOW, float& TSOIL){
int IPOL;
int J;
float BFAC;
float DSX;
float DTHR;
float DW;
float SNOWHC;
float PEXP;
float TAVGC;
float TSNOWC;
float TSOILC;
float ESDC;
float ESDCX;
float C1 = 0x1.47ae140000000p-7f;
float C2 = 0x1.5000000000000p+4f;
float G = 0x1.39eb860000000p+3f;
float KN = 0x1.f400000000000p+11f;
 // module_sf_noahlsm_glacial_only.F:1097
SNOWHC = MU(SNOWH,0x1.9000000000000p+6f); // module_sf_noahlsm_glacial_only.F:1109
ESDC = MU(SNEQV,0x1.9000000000000p+6f); // module_sf_noahlsm_glacial_only.F:1110
DTHR = DV(DTSEC,0x1.c200000000000p+11f); // module_sf_noahlsm_glacial_only.F:1111
TSNOWC = SU(TSNOW,0x1.1126660000000p+8f); // module_sf_noahlsm_glacial_only.F:1112
TSOILC = SU(TSOIL,0x1.1126660000000p+8f); // module_sf_noahlsm_glacial_only.F:1113
TAVGC = MU(0x1.0000000000000p-1f,AD(TSNOWC,TSOILC)); // module_sf_noahlsm_glacial_only.F:1127
if ((ESDC > 0x1.47ae140000000p-7f)) { // module_sf_noahlsm_glacial_only.F:1128
ESDCX = ESDC; // module_sf_noahlsm_glacial_only.F:1129
} else { // module_sf_noahlsm_glacial_only.F:1130
ESDCX = 0x1.47ae140000000p-7f; // module_sf_noahlsm_glacial_only.F:1131
} // module_sf_noahlsm_glacial_only.F:1132
BFAC = MU(MU(DTHR,C1),r_exp(SU(MU(0x1.47ae140000000p-4f,TAVGC),MU(C2,SNDENS)))); // module_sf_noahlsm_glacial_only.F:1152
IPOL = 4; // module_sf_noahlsm_glacial_only.F:1153
PEXP = 0x0.0p+0f; // module_sf_noahlsm_glacial_only.F:1154
for (J=IPOL; J>=1; J+=-1) { // module_sf_noahlsm_glacial_only.F:1156
PEXP = DV(MU(MU(AD(0x1.0000000000000p+0f,PEXP),BFAC),ESDCX),((float)((J + 1)))); // module_sf_noahlsm_glacial_only.F:1157
} // module_sf_noahlsm_glacial_only.F:1158
PEXP = AD(PEXP,0x1.0000000000000p+0f); // module_sf_noahlsm_glacial_only.F:1160
DSX = MU(SNDENS,PEXP); // module_sf_noahlsm_glacial_only.F:1181
if ((DSX > 0x1.99999a0000000p-2f)) { DSX = 0x1.99999a0000000p-2f; } // module_sf_noahlsm_glacial_only.F:1182
if ((DSX < 0x1.99999a0000000p-5f)) { DSX = 0x1.99999a0000000p-5f; } // module_sf_noahlsm_glacial_only.F:1183
SNDENS = DSX; // module_sf_noahlsm_glacial_only.F:1189
if ((TSNOWC >= 0x0.0p+0f)) { // module_sf_noahlsm_glacial_only.F:1190
DW = DV(MU(0x1.0a3d700000000p-3f,DTHR),0x1.8000000000000p+4f); // module_sf_noahlsm_glacial_only.F:1191
SNDENS = AD(MU(SNDENS,SU(0x1.0000000000000p+0f,DW)),DW); // module_sf_noahlsm_glacial_only.F:1192
if ((SNDENS >= 0x1.99999a0000000p-2f)) { SNDENS = 0x1.99999a0000000p-2f; } // module_sf_noahlsm_glacial_only.F:1193
} // module_sf_noahlsm_glacial_only.F:1198
SNOWHC = DV(ESDC,SNDENS); // module_sf_noahlsm_glacial_only.F:1199
SNOWH = MU(SNOWHC,0x1.47ae140000000p-7f); // module_sf_noahlsm_glacial_only.F:1200
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1206-1230
__device__ void gl_snowz0(NoahTables& tables, float& Z0, float& Z0BRD, float& SNOWH){
float Z0S = 0x1.0624de0000000p-10f;
float BURIAL;
float Z0EFF;
 // module_sf_noahlsm_glacial_only.F:1212
BURIAL = SU(MU(0x1.c000000000000p+2f,Z0BRD),SNOWH); // module_sf_noahlsm_glacial_only.F:1220
if ((BURIAL <= 0x1.6f00680000000p-11f)) { // module_sf_noahlsm_glacial_only.F:1221
Z0EFF = Z0S; // module_sf_noahlsm_glacial_only.F:1222
} else { // module_sf_noahlsm_glacial_only.F:1223
Z0EFF = DV(BURIAL,0x1.c000000000000p+2f); // module_sf_noahlsm_glacial_only.F:1224
} // module_sf_noahlsm_glacial_only.F:1225
Z0 = Z0EFF; // module_sf_noahlsm_glacial_only.F:1227
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:1233-1281
__device__ void gl_snow_new(NoahTables& tables, float& TEMP, float& NEWSN, float& SNOWH, float& SNDENS){
float DSNEW;
float HNEWC;
float SNOWHC;
float NEWSNC;
float TEMPC;
 // module_sf_noahlsm_glacial_only.F:1244
TEMPC = SU(TEMP,0x1.1126660000000p+8f); // module_sf_noahlsm_glacial_only.F:1255
if ((TEMPC <= -0x1.e000000000000p+3f)) { // module_sf_noahlsm_glacial_only.F:1256
DSNEW = 0x1.99999a0000000p-5f; // module_sf_noahlsm_glacial_only.F:1257
} else { // module_sf_noahlsm_glacial_only.F:1258
DSNEW = AD(0x1.99999a0000000p-5f,MU(0x1.bda5120000000p-10f,r_pow(AD(TEMPC,0x1.e000000000000p+3f),0x1.8000000000000p+0f))); // module_sf_noahlsm_glacial_only.F:1259
} // module_sf_noahlsm_glacial_only.F:1260
SNOWHC = MU(SNOWH,0x1.9000000000000p+6f); // module_sf_noahlsm_glacial_only.F:1265
NEWSNC = MU(NEWSN,0x1.9000000000000p+6f); // module_sf_noahlsm_glacial_only.F:1266
HNEWC = DV(NEWSNC,DSNEW); // module_sf_noahlsm_glacial_only.F:1271
if ((AD(SNOWHC,HNEWC) < 0x1.0624de0000000p-10f)) { // module_sf_noahlsm_glacial_only.F:1272
SNDENS = noah_max(DSNEW,SNDENS); // module_sf_noahlsm_glacial_only.F:1273
} else { // module_sf_noahlsm_glacial_only.F:1274
SNDENS = DV(AD(MU(SNOWHC,SNDENS),MU(HNEWC,DSNEW)),AD(SNOWHC,HNEWC)); // module_sf_noahlsm_glacial_only.F:1275
} // module_sf_noahlsm_glacial_only.F:1276
SNOWHC = AD(SNOWHC,HNEWC); // module_sf_noahlsm_glacial_only.F:1277
SNOWH = MU(SNOWHC,0x1.47ae140000000p-7f); // module_sf_noahlsm_glacial_only.F:1278
}
// MPAS v8.4.2 physics_wrf/module_sf_noahdrv.F:44-1590; MPAS urban=0 slice.
__device__ void noah_driver_column(NoahTables& tables, float* DZ8W, float* QV3D, float* P8W3D, float* T3D, float& TSK, float& HFX, float& QFX, float& LH, float& GRDFLX, float& QGH, float& SWDOWN, float& GLW, float& SMSTAV, float& SMSTOT, float& SFCRUNOFF, float& UDRUNOFF, int& IVGTYP, int& ISLTYP, int& ISURBAN, int& ISICE, float& VEGFRA, float& ALBEDO, float& ALBBCK, float& ZNT, float& Z0, float& TMN, float& XLAND, float& XICE, float& EMISS, float& EMBCK, float& SNOWC, float& QSFC, float& RAINBL, float& DT, float* DZS, int& ITIMESTEP, float* SMOIS, float* TSLB, float& SNOW, float& CANWAT, float& CHS, float& CHS2, float& CQS2, float& SR, float& CHKLOWQ, float& LAI, float& QZ0, float* SH2O, float& SNOWH, float& SNOALB, float& SHDMIN, float& SHDMAX, float& SNOTIME, float& ACSNOM, float& ACSNOW, float& SNOPCX, float& POTEVP, float* SMCREL, float& XICE_THRESHOLD, bool& USEMONALB, float& RIB, float& NOAHRES, int& OPT_THCND, float& FLX4_2D, float& FVB_2D, float& FBUR_2D, float& FGSN_2D, int I = 1, int J = 1, int* called = nullptr){
auto& LCZ_1=tables.LCZ_1;
auto& LCZ_2=tables.LCZ_2;
auto& LCZ_3=tables.LCZ_3;
auto& LCZ_4=tables.LCZ_4;
auto& LCZ_5=tables.LCZ_5;
auto& LCZ_6=tables.LCZ_6;
auto& LCZ_7=tables.LCZ_7;
auto& LCZ_8=tables.LCZ_8;
auto& LCZ_9=tables.LCZ_9;
auto& LCZ_10=tables.LCZ_10;
auto& LCZ_11=tables.LCZ_11;
const char* LUTYPE=tables.LUTYPE;
const char* SLTYPE=tables.SLTYPE;
auto& CP=tables.CP;
auto& R_D=tables.R_D;
auto& XLF=tables.XLF;
auto& XLV=tables.XLV;
auto& RHOWATER=tables.RHOWATER;
auto& STBOLT=tables.STBOLT;
int KTS = 1;
float ETPND1_HYDRO;
float SFCHEADRT_HYDRO;
float INFXSRT_HYDRO;
int NUM_SOIL_LAYERS = 4;
bool UA_PHYS = false;
float FLX4;
float FVB;
float FBUR;
float FGSN;
float ET[20];
float SMAV[20];
float BETA;
float ETP;
float SSOIL;
float EC;
float EDIR;
float ESNOW;
float ETT;
float FLX1;
float FLX2;
float FLX3;
float DRIP;
float DEW;
float FDOWN;
float RC;
float PC;
float RSMIN;
float XLAI;
float RCS;
float RCT;
float RCQ;
float RCSOIL;
float FFROZP;
bool MYJ = false;
bool FRPCPN = false;
bool LOCAL = false;
bool IPRINT;
int ICE;
int NSOIL;
int SLOPETYP;
int SOILTYP;
int VEGTYP;
int NROOT;
int NS;
float SHMIN;
float SHMAX;
float DQSDT2;
float LWDN;
float PRCP;
float Q2SAT;
float Q2SATI;
float SFCPRS;
float SFCTMP;
float SHDFAC;
float SNOALB1;
float SOLDN;
float TBOT;
float ZLVL;
float Q2K;
float ALBBRD;
float ALBEDOK;
float ETA;
float ETA_KINEMATIC;
float EMBRD;
float Z0K;
float RUNOFF1;
float RUNOFF2;
float RUNOFF3;
float SHEAT;
float SOLNET;
float E2SAT;
float SFCTSNO;
float SATFLG;
float FDTLIW;
float RIBB;
float FDTW;
float EMISSI;
float SNCOVR;
float SNEQV;
float SNOWHK;
float CMC;
float CHK;
float TH2;
float SMCDRY;
float SMCMAX;
float SMCREF;
float SMCWLT;
float SNOMLT;
float SOILM;
float SOILW;
float Q1;
float T1;
float SNOTIME1;
float DUMMY;
float Z0BRD;
float SLDPTH[20];
float STC[20];
float SMC[20];
float SWC[20];
float TRESH = 0x1.e666660000000p-1f;
float A2 = 0x1.1ab8520000000p+4f;
float A3 = 0x1.1126660000000p+8f;
float A4 = 0x1.da66660000000p+4f;
float T0 = 0x1.1128f60000000p+8f;
float ELWV = 0x1.312d000000000p+21f;
float A23M4 = MU(A2,SU(A3,A4));
float ROW = 0x1.f400000000000p+9f;
float ELIW = XLF;
float ROWLIW = MU(ROW,ELIW);
bool RDLAI2D = false;
float AOASIS;
float SIGMA_SB;
float CAPA = DV(R_D,CP);
float APELM;
float APES;
float SFCTH2;
float PSFC;
int FASDAS = 0;
float XSDA_HFX;
float XSDA_QFX;
float XQNORM = 0.0f; // module_sf_noahdrv.F:718
float HFX_PHY = 0.0f; // module_sf_noahdrv.F:716
float QFX_PHY = 0.0f; // module_sf_noahdrv.F:717
float HCPCT_FASDAS;
float IRRIGATION_CHANNEL = 0.0f; // module_sf_noahdrv.F:715
XSDA_HFX = 0x0.0p+0f; // module_sf_noahdrv.F:719
XSDA_QFX = 0x0.0p+0f; // module_sf_noahdrv.F:720
SIGMA_SB = 0x1.e70c9e0000000p-25f; // module_sf_noahdrv.F:724
FDTLIW = DV(DT,ROWLIW); // module_sf_noahdrv.F:727
FDTW = DV(DT,MU(XLV,RHOWATER)); // module_sf_noahdrv.F:729
IPRINT = false; // module_sf_noahdrv.F:731
SLOPETYP = 1; // module_sf_noahdrv.F:734
NSOIL = NUM_SOIL_LAYERS; // module_sf_noahdrv.F:738
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:740
SLDPTH[(NS)-(1)] = DZS[(NS)-(1)]; // module_sf_noahdrv.F:741
} // module_sf_noahdrv.F:742
if ((ITIMESTEP == 1)) { // module_sf_noahdrv.F:746
if ((SU(XLAND,0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // module_sf_noahdrv.F:759
SMSTAV = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:765
SMSTOT = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:766
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:767
SMOIS[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:768
TSLB[(NS)-(1)] = 0x1.1128f60000000p+8f; // module_sf_noahdrv.F:769
SMCREL[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:770
} // module_sf_noahdrv.F:771
} else { // module_sf_noahdrv.F:772
if ((XICE >= XICE_THRESHOLD)) { // module_sf_noahdrv.F:773
SMSTAV = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:775
SMSTOT = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:776
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:777
SMOIS[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:778
SMCREL[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:779
} // module_sf_noahdrv.F:780
} // module_sf_noahdrv.F:781
} // module_sf_noahdrv.F:782
label_50:; ; // module_sf_noahdrv.F:784
} // module_sf_noahdrv.F:785
PSFC = P8W3D[(1)-(1)]; // module_sf_noahdrv.F:790
SFCPRS = MU(AD(P8W3D[((KTS + 1))-(1)],P8W3D[(KTS)-(1)]),0x1.0000000000000p-1f); // module_sf_noahdrv.F:792
Q2K = DV(QV3D[(1)-(1)],AD(0x1.0000000000000p+0f,QV3D[(1)-(1)])); // module_sf_noahdrv.F:794
SOILW = 0x0.0p+0f; // module_sf_noahdrv.F:796
FLX4 = 0x0.0p+0f; // module_sf_noahdrv.F:797
FVB = 0x0.0p+0f; // module_sf_noahdrv.F:798
FBUR = 0x0.0p+0f; // module_sf_noahdrv.F:799
FGSN = 0x0.0p+0f; // module_sf_noahdrv.F:800
Q2SAT = DV(QGH,AD(0x1.0000000000000p+0f,QGH)); // module_sf_noahdrv.F:803
if (((MYJ && (Q2K >= MU(Q2SAT,TRESH))) && (Q2K < QZ0))) { // module_sf_noahdrv.F:806
SATFLG = 0x0.0p+0f; // module_sf_noahdrv.F:807
CHKLOWQ = 0x0.0p+0f; // module_sf_noahdrv.F:808
} else { // module_sf_noahdrv.F:809
SATFLG = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:810
CHKLOWQ = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:811
} // module_sf_noahdrv.F:812
SFCTMP = T3D[(1)-(1)]; // module_sf_noahdrv.F:814
ZLVL = MU(0x1.0000000000000p-1f,DZ8W[(1)-(1)]); // module_sf_noahdrv.F:815
APES = r_pow(DV(0x1.86a0000000000p+16f,PSFC),CAPA); // module_sf_noahdrv.F:819
APELM = r_pow(DV(0x1.86a0000000000p+16f,SFCPRS),CAPA); // module_sf_noahdrv.F:820
SFCTH2 = MU(SFCTMP,APELM); // module_sf_noahdrv.F:821
TH2 = DV(SFCTH2,APES); // module_sf_noahdrv.F:822
EMISSI = EMISS; // module_sf_noahdrv.F:824
LWDN = MU(GLW,EMISSI); // module_sf_noahdrv.F:825
SOLDN = SWDOWN; // module_sf_noahdrv.F:827
SOLNET = MU(SOLDN,SU(0x1.0000000000000p+0f,ALBEDO)); // module_sf_noahdrv.F:831
PRCP = DV(RAINBL,DT); // module_sf_noahdrv.F:832
VEGTYP = IVGTYP; // module_sf_noahdrv.F:840
SOILTYP = ISLTYP; // module_sf_noahdrv.F:841
SHDFAC = DV(VEGFRA,0x1.9000000000000p+6f); // module_sf_noahdrv.F:842
T1 = TSK; // module_sf_noahdrv.F:843
CHK = CHS; // module_sf_noahdrv.F:844
SHMIN = DV(SHDMIN,0x1.9000000000000p+6f); // module_sf_noahdrv.F:845
SHMAX = DV(SHDMAX,0x1.9000000000000p+6f); // module_sf_noahdrv.F:846
SNEQV = MU(SNOW,0x1.0624de0000000p-10f); // module_sf_noahdrv.F:848
SNOWHK = SNOWH; // module_sf_noahdrv.F:850
SNCOVR = SNOWC; // module_sf_noahdrv.F:851
if (FRPCPN) { // module_sf_noahdrv.F:856
FFROZP = SR; // module_sf_noahdrv.F:857
} else { // module_sf_noahdrv.F:858
if ((SFCTMP <= 0x1.1126660000000p+8f)) { // module_sf_noahdrv.F:859
FFROZP = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:860
} else { // module_sf_noahdrv.F:861
FFROZP = 0x0.0p+0f; // module_sf_noahdrv.F:862
} // module_sf_noahdrv.F:863
} // module_sf_noahdrv.F:864
if ((SU(XLAND,0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // module_sf_noahdrv.F:866
} else { // module_sf_noahdrv.F:875
if ((XICE >= XICE_THRESHOLD)) { // module_sf_noahdrv.F:878
ICE = 1; // module_sf_noahdrv.F:880
} else if ((VEGTYP == ISICE)) { // module_sf_noahdrv.F:881
ICE = -1; // module_sf_noahdrv.F:883
} else { // module_sf_noahdrv.F:884
ICE = 0; // module_sf_noahdrv.F:886
} // module_sf_noahdrv.F:887
DQSDT2 = DV(MU(Q2SAT,A23M4),noah_powi(SU(SFCTMP,A4),2)); // module_sf_noahdrv.F:888
if ((SNOW > 0x0.0p+0f)) { // module_sf_noahdrv.F:890
SFCTSNO = SFCTMP; // module_sf_noahdrv.F:892
E2SAT = MU(0x1.31999a0000000p+9f,r_exp(MU(0x1.81e0000000000p+12f,SU(0x1.dfda840000000p-9f,DV(0x1.0000000000000p+0f,SFCTSNO))))); // module_sf_noahdrv.F:893
Q2SATI = DV(MU(0x1.3e76c80000000p-1f,E2SAT),SU(SFCPRS,E2SAT)); // module_sf_noahdrv.F:894
Q2SATI = DV(Q2SATI,AD(0x1.0000000000000p+0f,Q2SATI)); // module_sf_noahdrv.F:895
if ((T1 > 0x1.1123d80000000p+8f)) { // module_sf_noahdrv.F:896
Q2SAT = AD(MU(Q2SAT,SU(0x1.0000000000000p+0f,SNOWC)),MU(Q2SATI,SNOWC)); // module_sf_noahdrv.F:898
DQSDT2 = AD(MU(DQSDT2,SU(0x1.0000000000000p+0f,SNOWC)),MU(DV(MU(Q2SATI,0x1.81e0000000000p+12f),noah_powi(SFCTSNO,2)),SNOWC)); // module_sf_noahdrv.F:899
} else { // module_sf_noahdrv.F:900
Q2SAT = Q2SATI; // module_sf_noahdrv.F:902
DQSDT2 = DV(MU(Q2SATI,0x1.81e0000000000p+12f),noah_powi(SFCTSNO,2)); // module_sf_noahdrv.F:903
} // module_sf_noahdrv.F:904
if ((((T1 > 0x1.1100000000000p+8f) && (SNOWC > 0x0.0p+0f)) && (SWDOWN > 0x1.4000000000000p+3f))) { DQSDT2 = MU(DQSDT2,SU(0x1.0000000000000p+0f,SNOWC)); } // module_sf_noahdrv.F:907
} // module_sf_noahdrv.F:908
TBOT = TMN; // module_sf_noahdrv.F:911
if ((ISURBAN == 1)) { // module_sf_noahdrv.F:913
if ((VEGTYP == 25)) { SHDFAC = 0x0.0p+0f; } // module_sf_noahdrv.F:915
if ((VEGTYP == 26)) { SHDFAC = 0x0.0p+0f; } // module_sf_noahdrv.F:916
if ((VEGTYP == 27)) { SHDFAC = 0x0.0p+0f; } // module_sf_noahdrv.F:917
} // module_sf_noahdrv.F:918
if (((SOILTYP == 14) && (XICE == 0x0.0p+0f))) { // module_sf_noahdrv.F:919
SOILTYP = 7; // module_sf_noahdrv.F:924
} // module_sf_noahdrv.F:925
SNOALB1 = SNOALB; // module_sf_noahdrv.F:926
CMC = DV(CANWAT,0x1.f400000000000p+9f); // module_sf_noahdrv.F:928
ALBBRD = ALBBCK; // module_sf_noahdrv.F:941
Z0BRD = Z0; // module_sf_noahdrv.F:942
EMBRD = EMBCK; // module_sf_noahdrv.F:943
SNOTIME1 = SNOTIME; // module_sf_noahdrv.F:944
RIBB = RIB; // module_sf_noahdrv.F:945
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:948
SMC[(NS)-(1)] = SMOIS[(NS)-(1)]; // module_sf_noahdrv.F:949
STC[(NS)-(1)] = TSLB[(NS)-(1)]; // module_sf_noahdrv.F:950
SWC[(NS)-(1)] = SH2O[(NS)-(1)]; // module_sf_noahdrv.F:951
} // module_sf_noahdrv.F:952
if ((((SNEQV != 0x0.0p+0f) && (SNOWHK == 0x0.0p+0f)) || (SNOWHK <= SNEQV))) { // module_sf_noahdrv.F:954
SNOWHK = MU(0x1.4000000000000p+2f,SNEQV); // module_sf_noahdrv.F:955
} // module_sf_noahdrv.F:956
if (((((((((((((IVGTYP == ISURBAN) || (IVGTYP == LCZ_1)) || (IVGTYP == LCZ_2)) || (IVGTYP == LCZ_3)) || (IVGTYP == LCZ_4)) || (IVGTYP == LCZ_5)) || (IVGTYP == LCZ_6)) || (IVGTYP == LCZ_7)) || (IVGTYP == LCZ_8)) || (IVGTYP == LCZ_9)) || (IVGTYP == LCZ_10)) || (IVGTYP == LCZ_11))) { // module_sf_noahdrv.F:986
VEGTYP = ISURBAN; // module_sf_noahdrv.F:990
} // module_sf_noahdrv.F:991
AOASIS = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:996
if ((ICE == 1)) { // module_sf_noahdrv.F:1064
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:1068
SH2O[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:1069
} // module_sf_noahdrv.F:1070
LAI = 0x1.47ae140000000p-7f; // module_sf_noahdrv.F:1071
return; // module_sf_noahdrv.F:1073
} else if ((ICE == 0)) { // module_sf_noahdrv.F:1075
ETPND1_HYDRO = 0x0.0p+0f; // module_sf_noahdrv.F:1098
SFCHEADRT_HYDRO = 0x0.0p+0f; // module_sf_noahdrv.F:1099
INFXSRT_HYDRO = 0x0.0p+0f; // module_sf_noahdrv.F:1100
{ if(called)*called=1; noah_sflx(tables,I,J,FFROZP,ISURBAN,DT,ZLVL,NSOIL,SLDPTH,LOCAL,LUTYPE,SLTYPE,LWDN,SOLDN,SOLNET,SFCPRS,PRCP,SFCTMP,Q2K,DUMMY,DUMMY,DUMMY,DUMMY,TH2,Q2SAT,DQSDT2,VEGTYP,SOILTYP,SLOPETYP,SHDFAC,SHMIN,SHMAX,ALBBRD,SNOALB1,TBOT,Z0BRD,Z0K,EMISSI,EMBRD,CMC,T1,STC,SMC,SWC,SNOWHK,SNEQV,ALBEDOK,CHK,DUMMY,ETA,SHEAT,ETA_KINEMATIC,FDOWN,EC,EDIR,ET,ETT,ESNOW,DRIP,DEW,BETA,ETP,SSOIL,FLX1,FLX2,FLX3,FLX4,FVB,FBUR,FGSN,UA_PHYS,SNOMLT,SNCOVR,RUNOFF1,RUNOFF2,RUNOFF3,RC,PC,RSMIN,XLAI,RCS,RCT,RCQ,RCSOIL,SOILW,SOILM,Q1,SMAV,RDLAI2D,USEMONALB,SNOTIME1,RIBB,SMCWLT,SMCDRY,SMCREF,SMCMAX,NROOT,SFCHEADRT_HYDRO,INFXSRT_HYDRO,ETPND1_HYDRO,OPT_THCND,AOASIS,XSDA_QFX,HFX_PHY,QFX_PHY,XQNORM,FASDAS,HCPCT_FASDAS,IRRIGATION_CHANNEL,true); if(tables.error)return; } // module_sf_noahdrv.F:1102
} else if ((ICE == -1)) { // module_sf_noahdrv.F:1137
SOILM = 0x0.0p+0f; // module_sf_noahdrv.F:1143
XLAI = 0x1.47ae140000000p-7f; // module_sf_noahdrv.F:1144
RUNOFF2 = 0x0.0p+0f; // module_sf_noahdrv.F:1145
RUNOFF3 = 0x0.0p+0f; // module_sf_noahdrv.F:1146
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:1147
SWC[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:1148
SMC[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:1149
SMAV[(NS)-(1)] = 0x1.0000000000000p+0f; // module_sf_noahdrv.F:1150
} // module_sf_noahdrv.F:1151
{ if(called)*called=2; gl_sflx_glacial(tables,I,J,ISICE,FFROZP,DT,ZLVL,NSOIL,SLDPTH,LWDN,SOLNET,SFCPRS,PRCP,SFCTMP,Q2K,TH2,Q2SAT,DQSDT2,ALBBRD,SNOALB1,TBOT,Z0BRD,Z0K,EMISSI,EMBRD,T1,STC,SNOWHK,SNEQV,ALBEDOK,CHK,ETA,SHEAT,ETA_KINEMATIC,FDOWN,ESNOW,DEW,ETP,SSOIL,FLX1,FLX2,FLX3,SNOMLT,SNCOVR,RUNOFF1,Q1,SNOTIME1,RIBB); } // module_sf_noahdrv.F:1167
} // module_sf_noahdrv.F:1182
LAI = XLAI; // module_sf_noahdrv.F:1184
CANWAT = MU(CMC,0x1.f400000000000p+9f); // module_sf_noahdrv.F:1217
SNOW = MU(SNEQV,0x1.f400000000000p+9f); // module_sf_noahdrv.F:1218
SNOWH = SNOWHK; // module_sf_noahdrv.F:1220
ALBEDO = ALBEDOK; // module_sf_noahdrv.F:1221
ALBBCK = ALBBRD; // module_sf_noahdrv.F:1223
Z0 = Z0BRD; // module_sf_noahdrv.F:1224
EMISS = EMISSI; // module_sf_noahdrv.F:1225
ZNT = Z0K; // module_sf_noahdrv.F:1228
TSK = T1; // module_sf_noahdrv.F:1245
HFX = SHEAT; // module_sf_noahdrv.F:1252
POTEVP = AD(POTEVP,MU(ETP,FDTW)); // module_sf_noahdrv.F:1255
QFX = ETA_KINEMATIC; // module_sf_noahdrv.F:1256
LH = ETA; // module_sf_noahdrv.F:1267
GRDFLX = SSOIL; // module_sf_noahdrv.F:1269
SNOWC = SNCOVR; // module_sf_noahdrv.F:1271
CHS2 = CQS2; // module_sf_noahdrv.F:1272
SNOTIME = SNOTIME1; // module_sf_noahdrv.F:1273
QSFC = DV(Q1,SU(0x1.0000000000000p+0f,Q1)); // module_sf_noahdrv.F:1283
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:1288
SMOIS[(NS)-(1)] = SMC[(NS)-(1)]; // module_sf_noahdrv.F:1289
TSLB[(NS)-(1)] = STC[(NS)-(1)]; // module_sf_noahdrv.F:1290
SH2O[(NS)-(1)] = SWC[(NS)-(1)]; // module_sf_noahdrv.F:1291
} // module_sf_noahdrv.F:1292
FLX4_2D = FLX4; // module_sf_noahdrv.F:1295
FVB_2D = FVB; // module_sf_noahdrv.F:1296
FBUR_2D = FBUR; // module_sf_noahdrv.F:1297
FGSN_2D = FGSN; // module_sf_noahdrv.F:1298
if (UA_PHYS) { // module_sf_noahdrv.F:1304
NOAHRES = SU(SU(SU(SU(SU(SU(AD(SU(AD(SOLNET,LWDN),SHEAT),SSOIL),ETA),MU(MU(EMISSI,STBOLT),noah_powi(T1,4))),FLX1),FLX2),FLX3),FLX4); // module_sf_noahdrv.F:1305
} else { // module_sf_noahdrv.F:1308
NOAHRES = SU(SU(SU(SU(SU(AD(SU(AD(SOLNET,LWDN),SHEAT),SSOIL),ETA),MU(MU(EMISSI,STBOLT),noah_powi(T1,4))),FLX1),FLX2),FLX3); // module_sf_noahdrv.F:1309
} // module_sf_noahdrv.F:1311
SMSTAV = SOILW; // module_sf_noahdrv.F:1570
SMSTOT = MU(SOILM,0x1.f400000000000p+9f); // module_sf_noahdrv.F:1571
for (NS=1; NS<=NSOIL; NS+=1) { // module_sf_noahdrv.F:1572
SMCREL[(NS)-(1)] = SMAV[(NS)-(1)]; // module_sf_noahdrv.F:1573
} // module_sf_noahdrv.F:1574
SFCRUNOFF = AD(SFCRUNOFF,MU(MU(RUNOFF1,DT),0x1.f400000000000p+9f)); // module_sf_noahdrv.F:1578
UDRUNOFF = AD(UDRUNOFF,MU(MU(RUNOFF2,DT),0x1.f400000000000p+9f)); // module_sf_noahdrv.F:1579
if ((FFROZP > 0x1.0000000000000p-1f)) { // module_sf_noahdrv.F:1581
ACSNOW = AD(ACSNOW,MU(PRCP,DT)); // module_sf_noahdrv.F:1582
} // module_sf_noahdrv.F:1583
if ((SNOW > 0x0.0p+0f)) { // module_sf_noahdrv.F:1584
ACSNOM = AD(ACSNOM,MU(SNOMLT,0x1.f400000000000p+9f)); // module_sf_noahdrv.F:1585
SNOPCX = SU(SNOPCX,DV(SNOMLT,FDTLIW)); // module_sf_noahdrv.F:1587
} // module_sf_noahdrv.F:1588
} // module_sf_noahdrv.F:1590
}
// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:76-207. FNDSNOWH=.true. in MPAS.
__device__ void noah_init_column(NoahTables& tables, float* TSLB, float* SMOIS, float* SH2O, float& SNOALB, float& SNOW, float& SNOWH, int& IVGTYP, int& ISLTYP, bool& RESTART, bool& INPUT_SFC_ALBEDO){
auto& MAXALB=tables.MAXALB;
auto& BB=tables.BB;
auto& MAXSMC=tables.MAXSMC;
auto& SATPSI=tables.SATPSI;
auto& GRAVITY=tables.GRAVITY;
int NS;
int NSOILLEVELS = 4;
float BX;
float FK;
float SMCMAX;
float PSISAT;
float FREE;
float BLIM = 0x1.6000000000000p+2f;
float HLICE = 0x1.45af000000000p+18f;
float T0 = 0x1.1126660000000p+8f;
if (RESTART) { return; } // mpas_atmphys_lsm_noahinit.F:136
if ((ISLTYP < 1)) { tables.error = 140; return; } // mpas_atmphys_lsm_noahinit.F:140
if ((!INPUT_SFC_ALBEDO)) { SNOALB = MU(MAXALB[(IVGTYP)-(1)],0x1.47ae140000000p-7f); } // mpas_atmphys_lsm_noahinit.F:146
BX = BB[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:154
SMCMAX = MAXSMC[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:155
PSISAT = SATPSI[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:156
if ((((BX > 0x0.0p+0f) && (SMCMAX > 0x0.0p+0f)) && (PSISAT > 0x0.0p+0f))) { // mpas_atmphys_lsm_noahinit.F:157
for (NS=1; NS<=NSOILLEVELS; NS+=1) { // mpas_atmphys_lsm_noahinit.F:158
if ((TSLB[(NS)-(1)] < 0x1.1126240000000p+8f)) { // mpas_atmphys_lsm_noahinit.F:161
BX = BB[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:166
SMCMAX = MAXSMC[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:167
PSISAT = SATPSI[(ISLTYP)-(1)]; // mpas_atmphys_lsm_noahinit.F:168
if ((BX > BLIM)) { BX = BLIM; } // mpas_atmphys_lsm_noahinit.F:169
FK = MU(r_pow(MU(DV(HLICE,MU(GRAVITY,(-PSISAT))),DV(SU(TSLB[(NS)-(1)],T0),TSLB[(NS)-(1)])),DV(-1,BX)),SMCMAX); // mpas_atmphys_lsm_noahinit.F:170
if ((FK < 0x1.47ae140000000p-6f)) { FK = 0x1.47ae140000000p-6f; } // mpas_atmphys_lsm_noahinit.F:172
SH2O[(NS)-(1)] = noah_min(FK,SMOIS[(NS)-(1)]); // mpas_atmphys_lsm_noahinit.F:173
{  noah_frh2o(tables,FREE,TSLB[(NS)-(1)],SMOIS[(NS)-(1)],SH2O[(NS)-(1)],SMCMAX,BX,PSISAT); if(tables.error)return; } // mpas_atmphys_lsm_noahinit.F:178
SH2O[(NS)-(1)] = FREE; // mpas_atmphys_lsm_noahinit.F:180
} else { // mpas_atmphys_lsm_noahinit.F:181
SH2O[(NS)-(1)] = SMOIS[(NS)-(1)]; // mpas_atmphys_lsm_noahinit.F:184
} // mpas_atmphys_lsm_noahinit.F:186
} // mpas_atmphys_lsm_noahinit.F:187
} else { // mpas_atmphys_lsm_noahinit.F:188
for (NS=1; NS<=NSOILLEVELS; NS+=1) { // mpas_atmphys_lsm_noahinit.F:189
SH2O[(NS)-(1)] = SMOIS[(NS)-(1)]; // mpas_atmphys_lsm_noahinit.F:190
} // mpas_atmphys_lsm_noahinit.F:191
} // mpas_atmphys_lsm_noahinit.F:192
}
"""

_ENTRY_CUDA = r"""
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:73-892
extern "C" __global__ void replay_noah(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*110;int* ix=ixs+c*14;NoahTables tables=*ts;tables.error=0;
// Four packed soil slots: prevent out-of-bounds module_sf_noahlsm.F:426 access.
if(ix[3]!=4){status[c]=426;return;}
int IILOC_arg=ix[0];int JJLOC_arg=ix[1];float FFROZP_arg=x[0];int ISURBAN_arg=ix[2];float DT_arg=x[1];float ZLVL_arg=x[2];int NSOIL_arg=ix[3];bool LOCAL_arg=ix[4];float LWDN_arg=x[7];float SOLDN_arg=x[8];float SOLNET_arg=x[9];float SFCPRS_arg=x[10];float PRCP_arg=x[11];float SFCTMP_arg=x[12];float Q2_arg=x[13];float SFCSPD_arg=x[14];float COSZ_arg=x[15];float PRCPRAIN_arg=x[16];float SOLARDIRECT_arg=x[17];float TH2_arg=x[18];float Q2SAT_arg=x[19];float DQSDT2_arg=x[20];int VEGTYP_arg=ix[5];int SOILTYP_arg=ix[6];int SLOPETYP_arg=ix[7];float SHDFAC_arg=x[21];float SHDMIN_arg=x[22];float SHDMAX_arg=x[23];float ALB_arg=x[24];float SNOALB_arg=x[25];float TBOT_arg=x[26];float Z0BRD_arg=x[27];float Z0_arg=x[28];float EMISSI_arg=x[29];float EMBRD_arg=x[30];float CMC_arg=x[31];float T1_arg=x[32];float SNOWH_arg=x[45];float SNEQV_arg=x[46];float ALBEDO_arg=x[47];float CH_arg=x[48];float CM_arg=x[49];float ETA_arg=x[50];float SHEAT_arg=x[51];float ETA_KINEMATIC_arg=x[52];float FDOWN_arg=x[53];float EC_arg=x[54];float EDIR_arg=x[55];float ETT_arg=x[60];float ESNOW_arg=x[61];float DRIP_arg=x[62];float DEW_arg=x[63];float BETA_arg=x[64];float ETP_arg=x[65];float SSOIL_arg=x[66];float FLX1_arg=x[67];float FLX2_arg=x[68];float FLX3_arg=x[69];float FLX4_arg=x[70];float FVB_arg=x[71];float FBUR_arg=x[72];float FGSN_arg=x[73];bool UA_PHYS_arg=ix[8];float SNOMLT_arg=x[74];float SNCOVR_arg=x[75];float RUNOFF1_arg=x[76];float RUNOFF2_arg=x[77];float RUNOFF3_arg=x[78];float RC_arg=x[79];float PC_arg=x[80];float RSMIN_arg=x[81];float XLAI_arg=x[82];float RCS_arg=x[83];float RCT_arg=x[84];float RCQ_arg=x[85];float RCSOIL_arg=x[86];float SOILW_arg=x[87];float SOILM_arg=x[88];float Q1_arg=x[89];bool RDLAI2D_arg=ix[9];bool USEMONALB_arg=ix[10];float SNOTIME1_arg=x[94];float RIBB_arg=x[95];float SMCWLT_arg=x[96];float SMCDRY_arg=x[97];float SMCREF_arg=x[98];float SMCMAX_arg=x[99];int NROOT_arg=ix[11];float SFHEAD1RT_arg=x[100];float INFXS1RT_arg=x[101];float ETPND1_arg=x[102];int OPT_THCND_arg=ix[12];float AOASIS_arg=x[103];float XSDA_QFX_arg=x[104];float HFX_PHY_arg=x[105];float QFX_PHY_arg=x[106];float XQNORM_arg=x[107];int FASDAS_arg=ix[13];float HCPCT_FASDAS_arg=x[108];float IRRIGATION_CHANNEL_arg=x[109];
noah_sflx(tables,IILOC_arg,JJLOC_arg,FFROZP_arg,ISURBAN_arg,DT_arg,ZLVL_arg,NSOIL_arg,x+3,LOCAL_arg,tables.LUTYPE,tables.SLTYPE,LWDN_arg,SOLDN_arg,SOLNET_arg,SFCPRS_arg,PRCP_arg,SFCTMP_arg,Q2_arg,SFCSPD_arg,COSZ_arg,PRCPRAIN_arg,SOLARDIRECT_arg,TH2_arg,Q2SAT_arg,DQSDT2_arg,VEGTYP_arg,SOILTYP_arg,SLOPETYP_arg,SHDFAC_arg,SHDMIN_arg,SHDMAX_arg,ALB_arg,SNOALB_arg,TBOT_arg,Z0BRD_arg,Z0_arg,EMISSI_arg,EMBRD_arg,CMC_arg,T1_arg,x+33,x+37,x+41,SNOWH_arg,SNEQV_arg,ALBEDO_arg,CH_arg,CM_arg,ETA_arg,SHEAT_arg,ETA_KINEMATIC_arg,FDOWN_arg,EC_arg,EDIR_arg,x+56,ETT_arg,ESNOW_arg,DRIP_arg,DEW_arg,BETA_arg,ETP_arg,SSOIL_arg,FLX1_arg,FLX2_arg,FLX3_arg,FLX4_arg,FVB_arg,FBUR_arg,FGSN_arg,UA_PHYS_arg,SNOMLT_arg,SNCOVR_arg,RUNOFF1_arg,RUNOFF2_arg,RUNOFF3_arg,RC_arg,PC_arg,RSMIN_arg,XLAI_arg,RCS_arg,RCT_arg,RCQ_arg,RCSOIL_arg,SOILW_arg,SOILM_arg,Q1_arg,x+90,RDLAI2D_arg,USEMONALB_arg,SNOTIME1_arg,RIBB_arg,SMCWLT_arg,SMCDRY_arg,SMCREF_arg,SMCMAX_arg,NROOT_arg,SFHEAD1RT_arg,INFXS1RT_arg,ETPND1_arg,OPT_THCND_arg,AOASIS_arg,XSDA_QFX_arg,HFX_PHY_arg,QFX_PHY_arg,XQNORM_arg,FASDAS_arg,HCPCT_FASDAS_arg,IRRIGATION_CHANNEL_arg,true);
ix[0]=IILOC_arg;ix[1]=JJLOC_arg;x[0]=FFROZP_arg;ix[2]=ISURBAN_arg;x[1]=DT_arg;x[2]=ZLVL_arg;ix[3]=NSOIL_arg;ix[4]=LOCAL_arg;x[7]=LWDN_arg;x[8]=SOLDN_arg;x[9]=SOLNET_arg;x[10]=SFCPRS_arg;x[11]=PRCP_arg;x[12]=SFCTMP_arg;x[13]=Q2_arg;x[14]=SFCSPD_arg;x[15]=COSZ_arg;x[16]=PRCPRAIN_arg;x[17]=SOLARDIRECT_arg;x[18]=TH2_arg;x[19]=Q2SAT_arg;x[20]=DQSDT2_arg;ix[5]=VEGTYP_arg;ix[6]=SOILTYP_arg;ix[7]=SLOPETYP_arg;x[21]=SHDFAC_arg;x[22]=SHDMIN_arg;x[23]=SHDMAX_arg;x[24]=ALB_arg;x[25]=SNOALB_arg;x[26]=TBOT_arg;x[27]=Z0BRD_arg;x[28]=Z0_arg;x[29]=EMISSI_arg;x[30]=EMBRD_arg;x[31]=CMC_arg;x[32]=T1_arg;x[45]=SNOWH_arg;x[46]=SNEQV_arg;x[47]=ALBEDO_arg;x[48]=CH_arg;x[49]=CM_arg;x[50]=ETA_arg;x[51]=SHEAT_arg;x[52]=ETA_KINEMATIC_arg;x[53]=FDOWN_arg;x[54]=EC_arg;x[55]=EDIR_arg;x[60]=ETT_arg;x[61]=ESNOW_arg;x[62]=DRIP_arg;x[63]=DEW_arg;x[64]=BETA_arg;x[65]=ETP_arg;x[66]=SSOIL_arg;x[67]=FLX1_arg;x[68]=FLX2_arg;x[69]=FLX3_arg;x[70]=FLX4_arg;x[71]=FVB_arg;x[72]=FBUR_arg;x[73]=FGSN_arg;ix[8]=UA_PHYS_arg;x[74]=SNOMLT_arg;x[75]=SNCOVR_arg;x[76]=RUNOFF1_arg;x[77]=RUNOFF2_arg;x[78]=RUNOFF3_arg;x[79]=RC_arg;x[80]=PC_arg;x[81]=RSMIN_arg;x[82]=XLAI_arg;x[83]=RCS_arg;x[84]=RCT_arg;x[85]=RCQ_arg;x[86]=RCSOIL_arg;x[87]=SOILW_arg;x[88]=SOILM_arg;x[89]=Q1_arg;ix[9]=RDLAI2D_arg;ix[10]=USEMONALB_arg;x[94]=SNOTIME1_arg;x[95]=RIBB_arg;x[96]=SMCWLT_arg;x[97]=SMCDRY_arg;x[98]=SMCREF_arg;x[99]=SMCMAX_arg;ix[11]=NROOT_arg;x[100]=SFHEAD1RT_arg;x[101]=INFXS1RT_arg;x[102]=ETPND1_arg;ix[12]=OPT_THCND_arg;x[103]=AOASIS_arg;x[104]=XSDA_QFX_arg;x[105]=HFX_PHY_arg;x[106]=QFX_PHY_arg;x[107]=XQNORM_arg;ix[13]=FASDAS_arg;x[108]=HCPCT_FASDAS_arg;x[109]=IRRIGATION_CHANNEL_arg;
status[c]=tables.error;
}

// MPAS v8.4.2 physics_wrf/module_sf_noahlsm_glacial_only.F:35-413
extern "C" __global__ void replay_glacial(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*49;int* ix=ixs+c*4;NoahTables tables=*ts;tables.error=0;
// Four packed soil slots: prevent out-of-bounds module_sf_noahlsm_glacial_only.F:211 access.
if(ix[3]!=4){status[c]=211;return;}
int IILOC_arg=ix[0];int JJLOC_arg=ix[1];int ISICE_arg=ix[2];float FFROZP_arg=x[0];float DT_arg=x[1];float ZLVL_arg=x[2];int NSOIL_arg=ix[3];float LWDN_arg=x[7];float SOLNET_arg=x[8];float SFCPRS_arg=x[9];float PRCP_arg=x[10];float SFCTMP_arg=x[11];float Q2_arg=x[12];float TH2_arg=x[13];float Q2SAT_arg=x[14];float DQSDT2_arg=x[15];float ALB_arg=x[16];float SNOALB_arg=x[17];float TBOT_arg=x[18];float Z0BRD_arg=x[19];float Z0_arg=x[20];float EMISSI_arg=x[21];float EMBRD_arg=x[22];float T1_arg=x[23];float SNOWH_arg=x[28];float SNEQV_arg=x[29];float ALBEDO_arg=x[30];float CH_arg=x[31];float ETA_arg=x[32];float SHEAT_arg=x[33];float ETA_KINEMATIC_arg=x[34];float FDOWN_arg=x[35];float ESNOW_arg=x[36];float DEW_arg=x[37];float ETP_arg=x[38];float SSOIL_arg=x[39];float FLX1_arg=x[40];float FLX2_arg=x[41];float FLX3_arg=x[42];float SNOMLT_arg=x[43];float SNCOVR_arg=x[44];float RUNOFF1_arg=x[45];float Q1_arg=x[46];float SNOTIME1_arg=x[47];float RIBB_arg=x[48];
gl_sflx_glacial(tables,IILOC_arg,JJLOC_arg,ISICE_arg,FFROZP_arg,DT_arg,ZLVL_arg,NSOIL_arg,x+3,LWDN_arg,SOLNET_arg,SFCPRS_arg,PRCP_arg,SFCTMP_arg,Q2_arg,TH2_arg,Q2SAT_arg,DQSDT2_arg,ALB_arg,SNOALB_arg,TBOT_arg,Z0BRD_arg,Z0_arg,EMISSI_arg,EMBRD_arg,T1_arg,x+24,SNOWH_arg,SNEQV_arg,ALBEDO_arg,CH_arg,ETA_arg,SHEAT_arg,ETA_KINEMATIC_arg,FDOWN_arg,ESNOW_arg,DEW_arg,ETP_arg,SSOIL_arg,FLX1_arg,FLX2_arg,FLX3_arg,SNOMLT_arg,SNCOVR_arg,RUNOFF1_arg,Q1_arg,SNOTIME1_arg,RIBB_arg);
ix[0]=IILOC_arg;ix[1]=JJLOC_arg;ix[2]=ISICE_arg;x[0]=FFROZP_arg;x[1]=DT_arg;x[2]=ZLVL_arg;ix[3]=NSOIL_arg;x[7]=LWDN_arg;x[8]=SOLNET_arg;x[9]=SFCPRS_arg;x[10]=PRCP_arg;x[11]=SFCTMP_arg;x[12]=Q2_arg;x[13]=TH2_arg;x[14]=Q2SAT_arg;x[15]=DQSDT2_arg;x[16]=ALB_arg;x[17]=SNOALB_arg;x[18]=TBOT_arg;x[19]=Z0BRD_arg;x[20]=Z0_arg;x[21]=EMISSI_arg;x[22]=EMBRD_arg;x[23]=T1_arg;x[28]=SNOWH_arg;x[29]=SNEQV_arg;x[30]=ALBEDO_arg;x[31]=CH_arg;x[32]=ETA_arg;x[33]=SHEAT_arg;x[34]=ETA_KINEMATIC_arg;x[35]=FDOWN_arg;x[36]=ESNOW_arg;x[37]=DEW_arg;x[38]=ETP_arg;x[39]=SSOIL_arg;x[40]=FLX1_arg;x[41]=FLX2_arg;x[42]=FLX3_arg;x[43]=SNOMLT_arg;x[44]=SNCOVR_arg;x[45]=RUNOFF1_arg;x[46]=Q1_arg;x[47]=SNOTIME1_arg;x[48]=RIBB_arg;
status[c]=tables.error;
}
extern "C" __global__ void replay_driver(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*79;int* ix=ixs+c*7;NoahTables tables=*ts;tables.error=0;
float TSK_arg=x[8];float HFX_arg=x[9];float QFX_arg=x[10];float LH_arg=x[11];float GRDFLX_arg=x[12];float QGH_arg=x[13];float SWDOWN_arg=x[14];float GLW_arg=x[15];float SMSTAV_arg=x[16];float SMSTOT_arg=x[17];float SFCRUNOFF_arg=x[18];float UDRUNOFF_arg=x[19];int IVGTYP_arg=ix[0];int ISLTYP_arg=ix[1];int ISURBAN_arg=ix[2];int ISICE_arg=ix[3];float VEGFRA_arg=x[20];float ALBEDO_arg=x[21];float ALBBCK_arg=x[22];float ZNT_arg=x[23];float Z0_arg=x[24];float TMN_arg=x[25];float XLAND_arg=x[26];float XICE_arg=x[27];float EMISS_arg=x[28];float EMBCK_arg=x[29];float SNOWC_arg=x[30];float QSFC_arg=x[31];float RAINBL_arg=x[32];float DT_arg=x[33];int ITIMESTEP_arg=ix[4];float SNOW_arg=x[46];float CANWAT_arg=x[47];float CHS_arg=x[48];float CHS2_arg=x[49];float CQS2_arg=x[50];float SR_arg=x[51];float CHKLOWQ_arg=x[52];float LAI_arg=x[53];float QZ0_arg=x[54];float SNOWH_arg=x[59];float SNOALB_arg=x[60];float SHDMIN_arg=x[61];float SHDMAX_arg=x[62];float SNOTIME_arg=x[63];float ACSNOM_arg=x[64];float ACSNOW_arg=x[65];float SNOPCX_arg=x[66];float POTEVP_arg=x[67];float XICE_THRESHOLD_arg=x[72];bool USEMONALB_arg=ix[5];float RIB_arg=x[73];float NOAHRES_arg=x[74];int OPT_THCND_arg=ix[6];float FLX4_2D_arg=x[75];float FVB_2D_arg=x[76];float FBUR_2D_arg=x[77];float FGSN_2D_arg=x[78];
noah_driver_column(tables,x+0,x+2,x+4,x+6,TSK_arg,HFX_arg,QFX_arg,LH_arg,GRDFLX_arg,QGH_arg,SWDOWN_arg,GLW_arg,SMSTAV_arg,SMSTOT_arg,SFCRUNOFF_arg,UDRUNOFF_arg,IVGTYP_arg,ISLTYP_arg,ISURBAN_arg,ISICE_arg,VEGFRA_arg,ALBEDO_arg,ALBBCK_arg,ZNT_arg,Z0_arg,TMN_arg,XLAND_arg,XICE_arg,EMISS_arg,EMBCK_arg,SNOWC_arg,QSFC_arg,RAINBL_arg,DT_arg,x+34,ITIMESTEP_arg,x+38,x+42,SNOW_arg,CANWAT_arg,CHS_arg,CHS2_arg,CQS2_arg,SR_arg,CHKLOWQ_arg,LAI_arg,QZ0_arg,x+55,SNOWH_arg,SNOALB_arg,SHDMIN_arg,SHDMAX_arg,SNOTIME_arg,ACSNOM_arg,ACSNOW_arg,SNOPCX_arg,POTEVP_arg,x+68,XICE_THRESHOLD_arg,USEMONALB_arg,RIB_arg,NOAHRES_arg,OPT_THCND_arg,FLX4_2D_arg,FVB_2D_arg,FBUR_2D_arg,FGSN_2D_arg);
x[8]=TSK_arg;x[9]=HFX_arg;x[10]=QFX_arg;x[11]=LH_arg;x[12]=GRDFLX_arg;x[13]=QGH_arg;x[14]=SWDOWN_arg;x[15]=GLW_arg;x[16]=SMSTAV_arg;x[17]=SMSTOT_arg;x[18]=SFCRUNOFF_arg;x[19]=UDRUNOFF_arg;ix[0]=IVGTYP_arg;ix[1]=ISLTYP_arg;ix[2]=ISURBAN_arg;ix[3]=ISICE_arg;x[20]=VEGFRA_arg;x[21]=ALBEDO_arg;x[22]=ALBBCK_arg;x[23]=ZNT_arg;x[24]=Z0_arg;x[25]=TMN_arg;x[26]=XLAND_arg;x[27]=XICE_arg;x[28]=EMISS_arg;x[29]=EMBCK_arg;x[30]=SNOWC_arg;x[31]=QSFC_arg;x[32]=RAINBL_arg;x[33]=DT_arg;ix[4]=ITIMESTEP_arg;x[46]=SNOW_arg;x[47]=CANWAT_arg;x[48]=CHS_arg;x[49]=CHS2_arg;x[50]=CQS2_arg;x[51]=SR_arg;x[52]=CHKLOWQ_arg;x[53]=LAI_arg;x[54]=QZ0_arg;x[59]=SNOWH_arg;x[60]=SNOALB_arg;x[61]=SHDMIN_arg;x[62]=SHDMAX_arg;x[63]=SNOTIME_arg;x[64]=ACSNOM_arg;x[65]=ACSNOW_arg;x[66]=SNOPCX_arg;x[67]=POTEVP_arg;x[72]=XICE_THRESHOLD_arg;ix[5]=USEMONALB_arg;x[73]=RIB_arg;x[74]=NOAHRES_arg;ix[6]=OPT_THCND_arg;x[75]=FLX4_2D_arg;x[76]=FVB_2D_arg;x[77]=FBUR_2D_arg;x[78]=FGSN_2D_arg;
status[c]=tables.error;
}
// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:76-207.
extern "C" __global__ void replay_init(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*15;int* ix=ixs+c*4;NoahTables tables=*ts;tables.error=0;
// mpas_atmphys_lsm_noahinit.F:140-150: native rejects only ISLTYP < 1.
// Preserve its unchecked table reads; no additional upper/category refusal.
if(!ix[2] && ix[1]<1){status[c]=140;return;}
bool restart=ix[2]!=0,albedo=ix[3]!=0;
noah_init_column(tables,x,x+4,x+8,x[12],x[13],x[14],ix[0],ix[1],restart,albedo);
status[c]=tables.error;
}
// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4526-4760.
extern "C" __global__ void replay_sfcdif(float* xs,int* ixs,const NoahTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*8;NoahTables tables=*ts;tables.error=0;
noah_sfcdif_off(tables,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
status[c]=tables.error;
}
"""


# MPAS v8.4.2 module_sf_noahlsm.F:398-399 and
# module_sf_noahlsm_glacial_only.F:209-210. A native serial tile leaves each
# module's diagnostic indices at its last invocation. Preserve tile order on
# device after parallel columns complete, retaining untouched module state.
_CORE_CUDA += r"""
extern "C" __device__ void noah_finish_module_locations(const int* locations,
    const int* calls, int* state, int n){
    for(int c=0;c<n;++c){
        if(calls[c]){
            int at=2*(calls[c]-1);
            state[at]=locations[2*c];state[at+1]=locations[2*c+1];
        }
    }
}
"""
_driver_entry = _ENTRY_CUDA[_ENTRY_CUDA.index('extern "C" __global__ void replay_driver('):
                            _ENTRY_CUDA.index('// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:76-207.')]
_ENTRY_CUDA += _driver_entry.replace(
    'void replay_driver(', 'void replay_driver_with_state(').replace(
    'int* status,int n){', 'int* status,int n,const int* locations,int* calls){').replace(
    'FGSN_2D_arg);', 'FGSN_2D_arg,locations[2*c],locations[2*c+1],calls+c);')
_STATE_CUDA = r"""
// MPAS v8.4.2 module_sf_noahlsm.F:398-399;
// module_sf_noahlsm_glacial_only.F:209-210.
extern "C" __global__ void finish_module_locations(const int* locations,
    const int* calls,int* state,int n){
    if(blockIdx.x || threadIdx.x)return;
    noah_finish_module_locations(locations,calls,state,n);
}
"""


def table_dtype():
    """C ABI for tables decoded by the native oracle or Rust input bridge."""
    fields = []
    capacities = {"NLUS": 50, "NSLTYPE": 30, "NSLOPE": 30}
    for name, spec in TABLE_SCHEMA.items():
        dtype = {"float": np.float32, "int": np.int32, "bool": np.bool_, "char": "S256"}[spec["type"]]
        if spec["dims"] and spec["type"] != "char":
            bound = spec["dims"][0].split(":")[-1]
            size = capacities[bound] if bound in capacities else int(bound)
            fields.append((name, dtype, (size,)))
        else:
            fields.append((name, dtype))
    fields.append(("error", np.int32))
    return np.dtype(fields, align=True)


def load_tables(directory, landuse, soil_type, *, decoder=None, device=None):
    """Upload tables decoded by the Rust MPAS Noah input bridge.

    MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:210-597 and
    mpas_atmphys_landuse.F:177-227. The decoder writes binary ABI bytes to
    stdout; Python only starts the bridge and transfers its buffers.
    """
    import os
    import shutil
    import subprocess
    if device is None:
        import cupy as device
    decoder = decoder or os.environ.get('WOOF_HEX_NOAH_TABLE_DECODER') or shutil.which('rw-mpas-noah-tables')
    if not decoder:
        raise RuntimeError('Rust MPAS Noah table decoder rw-mpas-noah-tables is required')
    payload = subprocess.check_output((str(decoder), str(directory), str(landuse), str(soil_type)))
    size = table_dtype().itemsize
    if len(payload) != size + 7*12*100*4:
        raise ValueError('Rust MPAS Noah table decoder returned a different ABI size')
    return (device.asarray(np.frombuffer(payload, dtype=np.uint8, count=size)),
            device.asarray(np.frombuffer(payload, dtype=np.uint8, offset=size)))


def _device_math_source():
    # Reuse shared engine routines verbatim. glibc 2.43 changed atanf/log10f
    # from the versions embedded in the old Noah-MP kernels.
    import woof
    root = Path(woof.__file__).resolve().parent / "core" / "kernels"
    basic = (root / 'glibc_flt32.cuh').read_text()
    trig = (root / 'glibc_trig_flt32.cuh').read_text()
    urban = (root / 'urban_bem.cuh').read_text()
    # Only the existing libm routine and its notice/tables, no urban physics.
    log10 = urban[urban.index('// log10f: CORE-MATH'):urban.index('__device__ void bem_bem')]
    aliases = '''
typedef float real;
#define AD(a,b) FADD(a,b)
#define SU(a,b) FSUB(a,b)
#define MU(a,b) FMUL(a,b)
#define DV(a,b) FDIV(a,b)
#define r_log gfk_log
#define r_exp gfk_exp
#define r_pow gfk_pow
#define r_atan glibc_atanf
#define r_log10 ubm_log10f
'''
    return basic + '\n' + trig + '\n' + log10 + aliases


def _compile_module(source, reserve_stack=True):
    """Compile without CuPy's appended FTZ flag, then load its device binary.

    RawModule(code=...) appends -ftz=true even when callers request false.
    That loses subnormal drainage in Noah WDFCND/SRT. The NVRTC entry below
    preserves the complete explicit options; RawModule(path=...) only loads.
    """
    import hashlib
    import os
    import cupy as cp
    from cupy.cuda import compiler, nvrtc
    # The complete solver's out-of-line CUDA call chain exceeds the 1 KiB
    # runtime default on sm_89. The unchanged cubin faults at 1 KiB and matches
    # every synthetic word at 8 KiB (stack-limit witness in lane receipts).
    # Reserve stack before loading either the production or helper module;
    # this changes device storage only, never the native arithmetic.
    # The production module in shared-tables mode has no recursion, so ptxas
    # sizes its whole chain and the driver reserves it (see _module_for).
    limit = cp.cuda.runtime.cudaLimitStackSize
    if reserve_stack and _runtime_limit(limit) < 8192:
        _runtime_limit(limit, 8192)
    arch = _compilation_target(xp=cp)
    from ..cuda_backend.aot import compiler_identity
    identity = repr((_hex_numerics.options(CUDA_OPTIONS, keep_ftz=True), arch, compiler_identity())) + source
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory = Path(os.environ.get('CUPY_CACHE_DIR', Path.home() / '.cupy/kernel_cache')) / 'hex-noah'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + '.cubin')
    if not path.exists():
        from ..cuda_backend.aot import direct_image
        binary = direct_image(source, _hex_numerics.options(CUDA_OPTIONS, keep_ftz=True), arch)
        # Process AND thread in the name.  THE BREAKAGE A PID ALONE CAUSED (2026-10-25, MPAS-suite LTS, 3 rate
        # classes on a cold cache): the class threads of one process compiled this module at once, wrote the same
        # .<pid>.tmp, one renamed it away and the next rename failed (FileNotFoundError), and another thread loaded
        # a half-written cubin (CUDA_ERROR_INVALID_IMAGE).  Each writer now owns its temporary; os.replace is atomic.
        import threading
        temporary = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
        temporary.write_bytes(binary)
        temporary.replace(path)
    return cp.RawModule(path=str(path))


#: Shared tables (memory lane, 2026-10-14). Every entry kernel began with
#: ``NoahTables tables=*ts``: each thread copied the whole table struct (land-use,
#: soil and slope tables, three 256-byte names; about 5.7 KB) into its own local
#: frame and then only read it. The CUDA driver sizes local memory as the widest
#: frame launched in the context times every thread the card can hold (RTX 5090:
#: 170 SMs x 1536) and never shrinks it, so that copy alone held about 1.5 GB of
#: every card. In shared mode each thread keeps only the members the solver writes
#: (the error line and the ILOC/JLOC diagnostic indices both solvers set on entry)
#: and reads every table through a
#: const reference to the one device copy. The compiler enforces the split: a
#: write through any other member is a const error. Every statement and operand
#: order is unchanged, so only storage moves.
#:
#: Shared mode also gives noah_powi a loop body without self-recursion (the same
#: multiplications and the same final DV), so ptxas sizes each entry's complete
#: call chain statically. The 8 KiB stack reservation in _compile_module covered
#: only that recursion (the sm_89 1 KiB fault); with a static chain the driver
#: reserves exactly the launched frame, so the production module skips it. The
#: helper replay modules still reserve 8 KiB. WOOF_HEX_NOAH_SHARED_TABLES=0
#: restores the per-thread copy, the recursive noah_powi and the reservation.
NOAH_THREAD_MEMBERS = ("ILOC", "JLOC", "error")
_POWI_RECURSIVE = ("__device__ float noah_powi(float a,int n){if(n<0)return DV(1.0f,noah_powi(a,-n));"
                   "float r=1.0f;while(n){if(n&1)r=MU(r,a);n>>=1;if(n)a=MU(a,a);}return r;}")
_POWI_STATIC = ("__device__ __forceinline__ float noah_powi_n(float a,int n){float r=1.0f;"
                "while(n){if(n&1)r=MU(r,a);n>>=1;if(n)a=MU(a,a);}return r;}\n"
                "__device__ float noah_powi(float a,int n){if(n<0)return DV(1.0f,noah_powi_n(a,-n));"
                "return noah_powi_n(a,n);}")
_THREAD_STRUCT = """
// Per-thread Noah module state; tables stay in the one device copy (memory lane).
struct NoahThread {
    const NoahTables& g; int ILOC; int JLOC; int error;
    __device__ __forceinline__ explicit NoahThread(const NoahTables& t):g(t),ILOC(t.ILOC),JLOC(t.JLOC),error(0) {}
};
"""


def shared_tables_enabled() -> bool:
    import os
    value = os.environ.get("WOOF_HEX_NOAH_SHARED_TABLES", "1").strip().lower()
    if value in ("1", "on", "true", "yes", ""):
        return True
    if value in ("0", "off", "false", "no", "copy"):
        return False
    raise ValueError("WOOF_HEX_NOAH_SHARED_TABLES must be 1/on or 0/off")


def _shared_tables_source(source: str) -> str:
    """Read Noah tables through a const reference instead of a per-thread copy."""
    import re
    end = source.index("};", source.index("struct NoahTables {")) + 2
    source = source[:end] + _THREAD_STRUCT + source[end:]
    if source.count(_POWI_RECURSIVE) != 1:
        raise ValueError("Noah noah_powi definition not found exactly once")
    source = source.replace(_POWI_RECURSIVE, _POWI_STATIC)
    copies = source.count("NoahTables tables=*ts;tables.error=0;")
    if copies != 6:
        raise ValueError(f"expected six Noah entry table copies, found {copies}")
    source = source.replace("NoahTables tables=*ts;tables.error=0;", "NoahThread tables(*ts);")
    source = source.replace("NoahTables& tables", "NoahThread& tables")
    keep = "|".join(NOAH_THREAD_MEMBERS)
    source = re.sub(r"\btables\.(?!(?:" + keep + r")\b)(?=[A-Za-z_])", "tables.g.", source)
    if re.search(r"\bNoahTables(?:\s+|&\s*)tables\b", source):
        raise ValueError("a Noah table copy escaped the shared-tables transformation")
    return source


@lru_cache(maxsize=None)
def _module_for(device: int, shared: bool):
    """One module per card: the stack limit is a per-device setting, so a
    process-wide cache reserved 8 KiB only on the first card and the second
    card faulted at the 1 KiB default with shared tables off."""
    import cupy as cp
    source = _device_math_source() + _CORE_CUDA + _ENTRY_CUDA + _LANDUSE_CUDA + _MPAS_IO_CUDA + _STATE_CUDA
    with cp.cuda.Device(device):
        if shared:
            return _compile_module(_shared_tables_source(source), reserve_stack=False)
        return _compile_module(source)


def _module():
    import cupy as cp
    return _module_for(int(cp.cuda.Device().id), shared_tables_enabled())


def replay_columns(scheme, floats, integers, tables, *, stream=None,
                   locations=None, module_locations=None):
    """Run complete SFLX or SFLX_GLACIAL in place; return device error lines.

    Inputs are contiguous device arrays (columns, packed arguments), float32
    and int32 respectively. ``tables`` is a one-element device array with
    ``table_dtype()`` ABI, or equivalent uint8 bytes. Optional irrigation is
    present in this replay contract. NSOIL is four, as in MPAS's driver.
    Driver ``locations`` are native tile (i,j) coordinates per column.
    ``module_locations`` retains four mutable globals in Noah i/j, glacier
    i/j order, updated by the last invocation of each solver in tile order.
    """
    import cupy as cp
    if scheme not in COLUMN_LAYOUTS:
        raise ValueError(f"unknown Noah entry: {scheme}")
    layout = COLUMN_LAYOUTS[scheme]
    for array, kind, width in ((floats, cp.float32, len(layout["float"])),
                               (integers, cp.int32, len(layout["int"]))):
        if not isinstance(array, cp.ndarray) or array.dtype != kind:
            raise TypeError("Noah columns require float32/int32 device arrays")
        if array.ndim != 2 or array.shape[1] != width or not array.flags.c_contiguous:
            raise ValueError("Noah argument packing differs from COLUMN_LAYOUTS")
    if floats.shape[0] != integers.shape[0]:
        raise ValueError("Noah float and integer column counts differ")
    if not isinstance(tables, cp.ndarray) or tables.nbytes != table_dtype().itemsize:
        raise ValueError("Noah table buffer must match table_dtype ABI")
    n = floats.shape[0]
    status = cp.empty(n, dtype=cp.int32)
    stateful = locations is not None or module_locations is not None
    if stateful:
        if scheme != 'driver':
            raise ValueError('module location export requires the MPAS driver entry')
        for a, shape in ((locations, (n, 2)), (module_locations, (4,))):
            if not isinstance(a, cp.ndarray) or a.dtype != cp.int32 or a.shape != shape or not a.flags.c_contiguous:
                raise ValueError('module location state requires contiguous int32 device buffers')
    if n:
        with stream or cp.cuda.get_current_stream():
            calls = cp.zeros(n, dtype=cp.int32) if stateful else None
        name = "replay_" + scheme + ('_with_state' if stateful else '')
        args = (floats, integers, tables, status, np.int32(n))
        if stateful:
            args += (locations, calls)
        _module().get_function(name)(((n + 63) // 64,), (64,), args, stream=stream)
        if stateful:
            _module().get_function('finish_module_locations')(
                (1,), (1,), (locations, calls, module_locations, np.int32(n)), stream=stream)
    return status

# MPAS v8.4.2 mpas_atmphys_landuse.F:257-314.
LANDUSE_FLOATS = ('LATCELL', 'SNOALB', 'SNOWC', 'XICE', 'ALBBCK', 'Z0', 'ZNT',
                  'EMBCK', 'SFC_ALBEDO', 'SFC_EMISS', 'THC', 'MAVAIL', 'UST', 'XICE_THRESHOLD')
LANDUSE_INTEGERS = ('IVGTYP', 'JULDAY', 'ISWATER', 'ISICE', 'RESTART',
                    'SFC_ALBEDO_INPUT', 'FRAC_SEAICE', 'HAS_MAVAIL', 'HAS_UST')
LANDUSE_TABLE_FIELDS = ('ALBD', 'SLMO', 'SFEM', 'SFZ0', 'THERIN', 'SCFX', 'SFHC')
LANDUSE_URBAN_INDEX = {'OLD': 1, 'USGS': 1, 'MODIFIED_IGBP_MODIS_NOAH': 13, 'SiB': 11, 'LW12': 1}

_LANDUSE_CUDA = r"""
// MPAS v8.4.2 mpas_atmphys_landuse.F:257-314.
// Fortran table order: (100 categories,12 seasons,7 table variables).
struct LanduseTables { float ALBD[1200],SLMO[1200],SFEM[1200],SFZ0[1200],THERIN[1200],SCFX[1200],SFHC[1200]; };
extern "C" __global__ void replay_landuse(float* xs,int* ixs,const LanduseTables* ts,int* status,int n){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
float* x=xs+c*14;int* ix=ixs+c*9;
status[c]=0;
if(ix[4])return; // mpas_atmphys_landuse.F:250-254.
int season=(ix[1]<105 || ix[1]>=288)?2:1;
if(x[0]<0.0f)season=3-season;
int category=ix[0]==0?ix[2]:ix[0];
// mpas_atmphys_landuse.F:268-273: native table category reads are unchecked.
int is=(season-1)*100+category-1,ice=(season-1)*100+ix[3]-1;
if(!ix[5])x[4]=DV(ts->ALBD[is],100.0f);
x[8]=x[4];
if(x[2]>0.5f){
 x[4]=DV(ts->ALBD[ice],100.0f);
 x[8]=ix[5]?x[1]:DV(x[4],AD(1.0f,ts->SCFX[is]));
}
x[10]=DV(ts->THERIN[is],100.0f);x[5]=DV(ts->SFZ0[is],100.0f);x[6]=x[5];
x[7]=ts->SFEM[is];x[9]=x[7];
if(ix[7])x[11]=ts->SLMO[is];
if(ix[8])x[12]=0.0001f;
if(x[3]>=x[13]){
 x[4]=DV(ts->ALBD[ice],100.0f);x[7]=ts->SFEM[ice];
 if(ix[6]){
  x[8]=AD(MU(x[3],x[4]),MU(SU(1.0f,x[3]),0.08f));
  x[9]=AD(MU(x[3],x[7]),MU(SU(1.0f,x[3]),0.98f));
 }else{x[8]=x[4];x[9]=x[7];}
 x[10]=DV(ts->THERIN[ice],100.0f);x[5]=DV(ts->SFZ0[ice],100.0f);x[6]=x[5];
 if(ix[7])x[11]=ts->SLMO[ice];
}
}
"""


def initialize_landuse(floats, integers, tables, *, stream=None):
    """Apply decoded LANDUSE.TBL properties, including restart and sea ice.

    Arrays use LANDUSE_FLOATS/INTEGERS order. ``tables`` is float32 with shape
    (7,12,100), contiguous; Rust owns file decoding. This follows MPAS's
    two-season selection exactly, including its day thresholds 105 and 288.
    """
    import cupy as cp
    for a, kind, width in ((floats, cp.float32, 14), (integers, cp.int32, 9)):
        if not isinstance(a, cp.ndarray) or a.dtype != kind:
            raise TypeError('landuse initialization requires float32/int32 device arrays')
        if a.ndim != 2 or a.shape[1] != width or not a.flags.c_contiguous:
            raise ValueError('landuse argument packing differs from documented ABI')
    if floats.shape[0] != integers.shape[0]:
        raise ValueError('landuse column counts differ')
    if not isinstance(tables, cp.ndarray) or tables.dtype != cp.float32 or tables.shape != (7,12,100) or not tables.flags.c_contiguous:
        raise ValueError('landuse table ABI is (7,12,100) contiguous float32')
    n = floats.shape[0]
    status = cp.empty(n, dtype=cp.int32)
    if n:
        _module().get_function('replay_landuse')(((n + 63)//64,), (64,),
                                               (floats, integers, tables, status, np.int32(n)), stream=stream)
    return status

# MPAS v8.4.2 mpas_atmphys_driver_lsm.F:357-630.
_MPAS_INPUT_FIELDS = {'dz8w': 'atmosphere', 'qv': 'atmosphere', 'pres2_hyd': 'pressure', 't': 'atmosphere', 'skintemp': 'surface', 'hfx': 'surface', 'qfx': 'surface', 'lh': 'surface', 'grdflx': 'surface', 'qgh': 'surface', 'glw': 'surface', 'smstav': 'surface', 'smstot': 'surface', 'sfcrunoff': 'surface', 'udrunoff': 'surface', 'vegfra': 'surface', 'sfc_albedo': 'surface', 'sfc_albbck': 'surface', 'znt': 'surface', 'z0': 'surface', 'tmn': 'surface', 'xland': 'surface', 'xice': 'surface', 'sfc_emiss': 'surface', 'sfc_emibck': 'surface', 'snowc': 'surface', 'qsfc': 'surface', 'dzs': 'soil', 'smois': 'soil', 'tslb': 'soil', 'snow': 'surface', 'canwat': 'surface', 'chs': 'surface', 'chs2': 'surface', 'cqs2': 'surface', 'chklowq': 'surface', 'lai': 'surface', 'sh2o': 'soil', 'snowh': 'surface', 'snoalb': 'surface', 'shdmin': 'surface', 'shdmax': 'surface', 'snotime': 'surface', 'acsnom': 'surface', 'acsnow': 'surface', 'snopcx': 'surface', 'potevp': 'surface', 'smcrel': 'soil', 'br': 'surface', 'noahres': 'surface', 'gsw': 'surface', 'rainncv': 'optional_surface', 'raincv': 'optional_surface', 'sr': 'optional_surface', 'ivgtyp': 'integer_surface', 'isltyp': 'integer_surface'}
_MPAS_OUTPUT_FIELDS = ['skintemp', 'hfx', 'qfx', 'lh', 'grdflx', 'smstav', 'smstot', 'sfcrunoff', 'udrunoff', 'sfc_albedo', 'sfc_albbck', 'znt', 'z0', 'sfc_emiss', 'snowc', 'qsfc', 'smois', 'tslb', 'snow', 'canwat', 'chs', 'chs2', 'cqs2', 'chklowq', 'lai', 'sh2o', 'snowh', 'snoalb', 'snotime', 'acsnom', 'acsnow', 'snopcx', 'potevp', 'smcrel', 'noahres']
_MPAS_IO_CUDA = r"""
// MPAS v8.4.2 mpas_atmphys_driver_lsm.F:357-480 (lsm_from_MPAS).
extern "C" __global__ void noah_from_mpas(float* xs,int* ixs,const float* dz8w,const float* qv,const float* pres2_hyd,const float* t,const float* skintemp,const float* hfx,const float* qfx,const float* lh,const float* grdflx,const float* qgh,const float* glw,const float* smstav,const float* smstot,const float* sfcrunoff,const float* udrunoff,const float* vegfra,const float* sfc_albedo,const float* sfc_albbck,const float* znt,const float* z0,const float* tmn,const float* xland,const float* xice,const float* sfc_emiss,const float* sfc_emibck,const float* snowc,const float* qsfc,const float* dzs,const float* smois,const float* tslb,const float* snow,const float* canwat,const float* chs,const float* chs2,const float* cqs2,const float* chklowq,const float* lai,const float* sh2o,const float* snowh,const float* snoalb,const float* shdmin,const float* shdmax,const float* snotime,const float* acsnom,const float* acsnow,const float* snopcx,const float* potevp,const float* smcrel,const float* br,const float* noahres,const float* gsw,const float* rainncv,const float* raincv,const float* sr,const int* ivgtyp,const int* isltyp,int nv,int npres,int start,int n,float dt,float xice_threshold,int micro,int convection,int isurban,int isice,int step,int usemonalb){
int local=blockIdx.x*blockDim.x+threadIdx.x;if(local>=n)return;int c=start+local;
float* x=xs+local*79;int* ix=ixs+local*7;
float rain=0.0f;if(micro)rain=AD(rain,rainncv[c]);if(convection)rain=AD(rain,raincv[c]);
x[0]=dz8w[c*nv+0];
x[1]=dz8w[c*nv+1];
x[2]=qv[c*nv+0];
x[3]=qv[c*nv+1];
x[4]=pres2_hyd[c*npres+0];
x[5]=pres2_hyd[c*npres+1];
x[6]=t[c*nv+0];
x[7]=t[c*nv+1];
x[8]=skintemp[c];
x[9]=hfx[c];
x[10]=qfx[c];
x[11]=lh[c];
x[12]=grdflx[c];
x[13]=qgh[c];
x[14]=DV(gsw[c],SU(1.0f,sfc_albedo[c]));
x[15]=glw[c];
x[16]=smstav[c];
x[17]=smstot[c];
x[18]=sfcrunoff[c];
x[19]=udrunoff[c];
x[20]=vegfra[c];
x[21]=sfc_albedo[c];
x[22]=sfc_albbck[c];
x[23]=znt[c];
x[24]=z0[c];
x[25]=tmn[c];
x[26]=xland[c];
x[27]=xice[c];
x[28]=sfc_emiss[c];
x[29]=sfc_emibck[c];
x[30]=snowc[c];
x[31]=qsfc[c];
x[32]=rain;
x[33]=dt;
x[34]=dzs[start*4+0];
x[35]=dzs[start*4+1];
x[36]=dzs[start*4+2];
x[37]=dzs[start*4+3];
x[38]=smois[c*4+0];
x[39]=smois[c*4+1];
x[40]=smois[c*4+2];
x[41]=smois[c*4+3];
x[42]=tslb[c*4+0];
x[43]=tslb[c*4+1];
x[44]=tslb[c*4+2];
x[45]=tslb[c*4+3];
x[46]=snow[c];
x[47]=canwat[c];
x[48]=chs[c];
x[49]=chs2[c];
x[50]=cqs2[c];
x[51]=micro?sr[c]:0.0f;
x[52]=chklowq[c];
x[53]=lai[c];
x[54]=0.0f;
x[55]=sh2o[c*4+0];
x[56]=sh2o[c*4+1];
x[57]=sh2o[c*4+2];
x[58]=sh2o[c*4+3];
x[59]=snowh[c];
x[60]=snoalb[c];
x[61]=shdmin[c];
x[62]=shdmax[c];
x[63]=snotime[c];
x[64]=acsnom[c];
x[65]=acsnow[c];
x[66]=snopcx[c];
x[67]=potevp[c];
x[68]=smcrel[c*4+0];
x[69]=smcrel[c*4+1];
x[70]=smcrel[c*4+2];
x[71]=smcrel[c*4+3];
x[72]=xice_threshold;
x[73]=br[c];
x[74]=noahres[c];
x[75]=0.0f;
x[76]=0.0f;
x[77]=0.0f;
x[78]=0.0f;
ix[0]=ivgtyp[c];
ix[1]=isltyp[c];
ix[2]=isurban;
ix[3]=isice;
ix[4]=step;
ix[5]=usemonalb;
ix[6]=1;
}
// MPAS v8.4.2 mpas_atmphys_driver_lsm.F:563-630 (lsm_to_MPAS).
extern "C" __global__ void noah_to_mpas(const float* xs,const int* status,float* skintemp,float* hfx,float* qfx,float* lh,float* grdflx,float* smstav,float* smstot,float* sfcrunoff,float* udrunoff,float* sfc_albedo,float* sfc_albbck,float* znt,float* z0,float* sfc_emiss,float* snowc,float* qsfc,float* smois,float* tslb,float* snow,float* canwat,float* chs,float* chs2,float* cqs2,float* chklowq,float* lai,float* sh2o,float* snowh,float* snoalb,float* snotime,float* acsnom,float* acsnow,float* snopcx,float* potevp,float* smcrel,float* noahres,int start,int n){
int local=blockIdx.x*blockDim.x+threadIdx.x;if(local>=n || status[local])return;int c=start+local;
const float* x=xs+local*79;
skintemp[c]=x[8];
hfx[c]=x[9];
qfx[c]=x[10];
lh[c]=x[11];
grdflx[c]=x[12];
smstav[c]=x[16];
smstot[c]=x[17];
sfcrunoff[c]=x[18];
udrunoff[c]=x[19];
sfc_albedo[c]=x[21];
sfc_albbck[c]=x[22];
znt[c]=x[23];
z0[c]=x[24];
sfc_emiss[c]=x[28];
snowc[c]=x[30];
qsfc[c]=x[31];
smois[c*4+0]=x[38];
smois[c*4+1]=x[39];
smois[c*4+2]=x[40];
smois[c*4+3]=x[41];
tslb[c*4+0]=x[42];
tslb[c*4+1]=x[43];
tslb[c*4+2]=x[44];
tslb[c*4+3]=x[45];
snow[c]=x[46];
canwat[c]=x[47];
chs[c]=x[48];
chs2[c]=x[49];
cqs2[c]=x[50];
chklowq[c]=x[52];
lai[c]=x[53];
sh2o[c*4+0]=x[55];
sh2o[c*4+1]=x[56];
sh2o[c*4+2]=x[57];
sh2o[c*4+3]=x[58];
snowh[c]=x[59];
snoalb[c]=x[60];
snotime[c]=x[63];
acsnom[c]=x[64];
acsnow[c]=x[65];
snopcx[c]=x[66];
potevp[c]=x[67];
smcrel[c*4+0]=x[68];
smcrel[c*4+1]=x[69];
smcrel[c*4+2]=x[70];
smcrel[c*4+3]=x[71];
noahres[c]=x[74];
}
"""


def run_mpas_surface(fields, tables, *, dt_pbl, itimestep, isurban, isice,
                     config_sfc_albedo, xice_threshold, microphysics_enabled,
                     convection_enabled, start=0, count=None, stream=None,
                     level_first=False, module_locations=None):
    """Run the MPAS Noah slot and its exact from/to-MPAS conversions.

    ``fields`` contains device arrays named after MPAS pools. Atmospheric
    inputs t/qv/dz8w and pres2_hyd are prepared by the common physics driver,
    with shape (cells,levels). Soil arrays have shape (cells,4); surface arrays
    have shape (cells,). DZS follows MPAS's first-active-cell convention.
    All arithmetic and field packing run in RawKernels. Sea ice exits Noah's
    land step as native does; the separate MPAS seaice slot follows it.
    ``level_first=True`` accepts f4's contiguous (levels,cells) pool layout;
    CuPy copies transposed device views for the native packed kernels and
    copies their soil outputs back to the original shared pools.
    Optional ``module_locations`` retains native diagnostic indices on device
    across calls; this MPAS tile uses i=cell+1, j=1.
    """
    import cupy as cp
    if level_first:
        with stream or cp.cuda.get_current_stream():
            converted = dict(fields)
            for key, shape in _MPAS_INPUT_FIELDS.items():
                if shape in ('soil', 'atmosphere', 'pressure'):
                    if not isinstance(fields[key], cp.ndarray) or fields[key].ndim != 2 or not fields[key].flags.c_contiguous:
                        raise ValueError(f'MPAS Noah level-first field {key} requires a contiguous device matrix')
                    converted[key] = cp.ascontiguousarray(fields[key].T)
            status = run_mpas_surface(
                converted, tables, dt_pbl=dt_pbl, itimestep=itimestep,
                isurban=isurban, isice=isice, config_sfc_albedo=config_sfc_albedo,
                xice_threshold=xice_threshold, microphysics_enabled=microphysics_enabled,
                convection_enabled=convection_enabled, start=start, count=count, stream=stream,
                module_locations=module_locations)
            for key in _MPAS_OUTPUT_FIELDS:
                if _MPAS_INPUT_FIELDS[key] == 'soil':
                    cp.copyto(fields[key], converted[key].T)
            return status
    ncells = fields['skintemp'].size
    count = ncells-start if count is None else count
    if start < 0 or count < 0 or start+count > ncells:
        raise ValueError('MPAS Noah active cell interval is outside the arrays')
    nv = fields['t'].shape[1]
    npres = fields['pres2_hyd'].shape[1]
    args = []
    for key, shape in _MPAS_INPUT_FIELDS.items():
        enabled = microphysics_enabled if key in ('rainncv','sr') else convection_enabled
        if shape == 'optional_surface' and not enabled:
            a = fields['skintemp']  # valid device pointer, never read when disabled
        else:
            a = fields[key]
        kind = cp.int32 if shape == 'integer_surface' else cp.float32
        expected = (ncells,4) if shape == 'soil' else (ncells,nv) if shape == 'atmosphere' else (ncells,npres) if shape == 'pressure' else (ncells,)
        if not isinstance(a, cp.ndarray) or a.dtype != kind or a.shape != expected or not a.flags.c_contiguous:
            raise ValueError(f'MPAS Noah field {key} requires contiguous {kind} with shape {expected}')
        args.append(a)
    if nv < 2 or npres < 2:
        raise ValueError('MPAS Noah reads the lowest layer and two pressure interfaces')
    x = cp.empty((count,79),dtype=cp.float32)
    ix = cp.empty((count,7),dtype=cp.int32)
    if not count:
        return cp.empty(0,dtype=cp.int32)
    grid = ((count+63)//64,)
    _module().get_function('noah_from_mpas')(grid,(64,),
        (x,ix,*args,np.int32(nv),np.int32(npres),np.int32(start),np.int32(count),
         np.float32(dt_pbl),np.float32(xice_threshold),np.int32(microphysics_enabled),
         np.int32(convection_enabled),np.int32(isurban),np.int32(isice),
         np.int32(itimestep),np.int32(config_sfc_albedo)),stream=stream)
    locations = None
    if module_locations is not None:
        with stream or cp.cuda.get_current_stream():
            locations = cp.empty((count,2),dtype=cp.int32)
            locations[:,0] = cp.arange(start+1,start+count+1,dtype=cp.int32)
            locations[:,1] = 1
    status = replay_columns('driver',x,ix,tables,stream=stream,
                            locations=locations,module_locations=module_locations)
    _module().get_function('noah_to_mpas')(grid,(64,),
        (x,status,*(fields[k] for k in _MPAS_OUTPUT_FIELDS),np.int32(start),np.int32(count)),stream=stream)
    return status


def register_hex_cuda(registry, arguments):
    """Register sf_noah through f4's SlotRegistry without retaining field state.

    ``arguments(call)`` is the framework pool adapter. It supplies ``fields``,
    ``tables``, category constants, active extents and the optional CUDA stream
    in run_mpas_surface's documented ABI. The common adapter owns pool layout
    and restart storage; this callback owns only the native LSM driver slice.
    """
    # Native land-slot call and its argument list; mpas_atmphys_driver_lsm.F:663-750.
    def launch(call):
        kwargs = dict(arguments(call))
        kwargs.update(dt_pbl=kwargs.get('dt_pbl', call.dt), itimestep=call.itimestep,
                      config_sfc_albedo=call.options['config_sfc_albedo'],
                      microphysics_enabled=call.options['config_microp_scheme'] != 'off',
                      convection_enabled=call.options['config_convection_scheme'] != 'off')
        status = run_mpas_surface(**kwargs)
        # Native fatal conditions abort before the next slot. Prevent continuing
        # a forecast after the CUDA equivalent has refused a column.
        if status.size:
            import cupy as cp
            from .deferred_checks_v842 import defer
            # Read off the step path inside a forecast (deferred_checks_v842, U-MPASPHYS-1).
            with kwargs.get('stream') or cp.cuda.get_current_stream():
                defer(status, 'MPAS Noah refused a column at native source line {value}', error=RuntimeError)
        return status

    # Native driver dispatch position; mpas_atmphys_driver.F:286-308.
    registry.register('land_surface', 'sf_noah', 'hex-cuda', launch)


# Native pool gets; MPAS v8.4.2 mpas_atmphys_driver_lsm.F:300-358,450-480.
_NATIVE_POOL_FIELDS = {'skintemp': ('sfc_input', 'skintemp'), 'hfx': ('diag_physics', 'hfx'), 'qfx': ('diag_physics', 'qfx'), 'lh': ('diag_physics', 'lh'), 'grdflx': ('diag_physics', 'grdflx'), 'qgh': ('diag_physics', 'qgh'), 'glw': ('diag_physics', 'glw'), 'smstav': ('diag_physics', 'smstav'), 'smstot': ('diag_physics', 'smstot'), 'sfcrunoff': ('diag_physics', 'sfcrunoff'), 'udrunoff': ('diag_physics', 'udrunoff'), 'vegfra': ('sfc_input', 'vegfra'), 'sfc_albedo': ('diag_physics', 'sfc_albedo'), 'sfc_albbck': ('sfc_input', 'sfc_albbck'), 'znt': ('diag_physics', 'znt'), 'z0': ('diag_physics', 'z0'), 'tmn': ('sfc_input', 'tmn'), 'xland': ('sfc_input', 'xland'), 'xice': ('sfc_input', 'xice'), 'sfc_emiss': ('diag_physics', 'sfc_emiss'), 'sfc_emibck': ('diag_physics', 'sfc_emibck'), 'snowc': ('sfc_input', 'snowc'), 'qsfc': ('diag_physics', 'qsfc'), 'dzs': ('sfc_input', 'dzs'), 'smois': ('sfc_input', 'smois'), 'tslb': ('sfc_input', 'tslb'), 'snow': ('sfc_input', 'snow'), 'canwat': ('diag_physics', 'canwat'), 'chs': ('diag_physics', 'chs'), 'chs2': ('diag_physics', 'chs2'), 'cqs2': ('diag_physics', 'cqs2'), 'chklowq': ('diag_physics', 'chklowq'), 'lai': ('diag_physics', 'lai'), 'sh2o': ('sfc_input', 'sh2o'), 'snowh': ('sfc_input', 'snowh'), 'snoalb': ('sfc_input', 'snoalb'), 'shdmin': ('sfc_input', 'shdmin'), 'shdmax': ('sfc_input', 'shdmax'), 'snotime': ('diag_physics', 'snotime'), 'acsnom': ('diag_physics', 'acsnom'), 'acsnow': ('diag_physics', 'acsnow'), 'snopcx': ('diag_physics', 'snopcx'), 'potevp': ('diag_physics', 'potevp'), 'smcrel': ('sfc_input', 'smcrel'), 'br': ('diag_physics', 'br'), 'noahres': ('diag_physics', 'noahres'), 'gsw': ('diag_physics', 'gsw'), 'rainncv': ('diag_physics', 'rainncv'), 'raincv': ('diag_physics', 'raincv'), 'sr': ('diag_physics', 'sr'), 'ivgtyp': ('sfc_input', 'ivgtyp'), 'isltyp': ('sfc_input', 'isltyp')}

# MPAS v8.4.2 mpas_atmphys_driver_lsm.F:706-707.
_NATIVE_PREPARED_FIELDS = {'dz8w': 'dz_p', 'qv': 'qv_p', 'pres2_hyd': 'pres2_hyd_p', 't': 't_p'}

# MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:210-597; mpas_atmphys_landuse.F:177-227.
TABLE_FILE_SHA256 = {
    'LANDUSE.TBL': '7f661318ff5f06aed6b4b5508e4d077dc794cff03d5fb4447fe59c88d0ed2ece',
    'VEGPARM.TBL': 'ed5478afbe49af51492256c1eb6cf88b3948590308525b32df1ddec28687b40e',
    'SOILPARM.TBL': '1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f',
    'GENPARM.TBL': '9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2',
}


def load_native_resources(resources, device):
    """Persistent Rust-decoded native tables for the foundation slot descriptor.

    MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:210-597;
    mpas_atmphys_landuse.F:177-227. Python authenticates and transports bytes.
    """
    import hashlib
    directory = Path(resources['table_directory'])
    decoder = resources['rust_decoder']
    path = Path(decoder['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != decoder['sha256']:
        raise ValueError('Noah Rust table decoder certificate differs')
    for name, expected in TABLE_FILE_SHA256.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Noah pinned table certificate differs: {name}')
    tables, landuse_tables = load_tables(directory, resources['landuse'],
                                        resources['soil_type'], decoder=path, device=device)
    return dict(tables=tables, landuse_tables=landuse_tables,
                landuse=resources['landuse'], isurban=resources.get('isurban'))



def _native_device_input(provider, pool, name):
    """Transport native solve words; MPAS v8.4.2 mpas_atmphys_driver_lsm.F:300-480.

    Foundation's generic provider currently calls CuPy.ascontiguousarray on
    NumPy views. CuPy requires an explicit upload first. Cache that upload in
    the existing provider transaction, so its normal commit preserves halos.
    This copies bytes only; native preparation and all arithmetic are unchanged.
    """
    if getattr(provider, 'resident', False):
        return provider.prepared(name) if pool == 'prepared' else provider.field(pool, name)
    count = provider.session.layout.dimensions['nCellsSolve']
    if pool == 'prepared':
        key = (pool, name)
        if key not in provider.buffers:
            value = provider.native.get_variable(name)
            if value is None:
                raise RuntimeError(f'native prepared array {name} is not allocated before Noah')
            host, lower = value
            if host.ndim != 3 or host.shape[-1] != 1 or lower[0] != 1:
                raise ValueError('Noah prepared profiles require native i,k,one-j bounds')
            packed = provider.cp.ascontiguousarray(provider.cp.asarray(host[:count, :, 0].T))
            provider.buffers[key] = (host, packed, lower)
    else:
        key = (pool, name, 1)
        if key not in provider.buffers:
            host = provider.native.get_field(pool, name, level=1)
            selection = (..., slice(0, count))
            packed = provider.cp.ascontiguousarray(provider.cp.asarray(host[selection]))
            provider.buffers[key] = (host, packed, selection)
    return provider.buffers[key][1]


def bind_native_pools(provider, call):
    """Bind the complete MPAS driver interface to foundation's solve-cell views.

    MPAS v8.4.2 mpas_atmphys_driver_lsm.F:300-480,694-737;
    mpas_atmphys_landuse.F:222-239. Native initialization owns initial pools.
    """
    resources = provider.noah_v842_resources
    landuse = provider.native.get_text_field('sfc_input', 'mminlu').strip()
    if landuse != resources['landuse']:
        raise ValueError('Noah table classification differs from native mminlu')
    # Transport the public native word, including the unchanged value for
    # an unassigned classification; mpas_atmphys_landuse.F:27,222-243.
    # This is the pinned GNU INTEGER ABI, not a replacement table assignment.
    if getattr(provider.native, 'framework_only', False):
        isurban = provider.native.landuse_state['isurban']
    else:
        import ctypes
        isurban = ctypes.c_int.in_dll(
            provider.native.lib, '__mpas_atmphys_landuse_MOD_isurban').value
    fields = {name: _native_device_input(provider, 'prepared', native_name)
              for name, native_name in _NATIVE_PREPARED_FIELDS.items()}
    for name, (pool, native_name) in _NATIVE_POOL_FIELDS.items():
        if name in ('rainncv', 'sr') and call.options['config_microp_scheme'] == 'off':
            continue
        if name == 'raincv' and call.options['config_convection_scheme'] == 'off':
            continue
        fields[name] = _native_device_input(provider, pool, native_name)
    return dict(fields=fields, tables=resources['tables'], level_first=True,
                start=0, count=provider.session.layout.dimensions['nCellsSolve'],
                dt_pbl=provider['scalar_state']['dt_pbl'],
                xice_threshold=provider['scalar_state']['xice_threshold'],
                isice=provider.native.get_field('sfc_input', 'isice').item(),
                isurban=isurban)


def run_slot(call):
    """Foundation callback for the entire Noah land-surface slot.

    MPAS v8.4.2 mpas_atmphys_driver_lsm.F:663-759. The common driver supplies
    native initialization, sea ice and surface diagnostics separately.
    """
    arguments = dict(call.pools)
    arguments.update(itimestep=call.itimestep,
                     config_sfc_albedo=call.options['config_sfc_albedo'],
                     microphysics_enabled=call.options['config_microp_scheme'] != 'off',
                     convection_enabled=call.options['config_convection_scheme'] != 'off')
    status = run_mpas_surface(**arguments)
    if status.size:
        import cupy as cp
        from .deferred_checks_v842 import defer
        # Read off the step path inside a forecast (deferred_checks_v842, U-MPASPHYS-1).
        with arguments.get('stream') or cp.cuda.get_current_stream():
            defer(status, 'MPAS Noah native fatal condition at source line {value}', error=RuntimeError)
    return status
