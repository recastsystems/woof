// WRF v4 public domain; licenses/LICENSE-WRF-public-domain.txt.
namespace noahmp_crop {
using noahmp_carbon::mn;using noahmp_carbon::mx;
struct Params {
float q10mr;
float foln_mx;
float lfmr25;
float rtmr25;
float stmr25;
float grainmr25;
float fra_gr;
float lefreez;
float mrp;
float bio2lai;
float gddtbase;
float gddtcut;
float gdds1;
float gdds2;
float gdds3;
float gdds4;
float gdds5;
float pltday;
float hsday;
float lfpt[8];
float stpt[8];
float rtpt[8];
float grainpt[8];
float lf_ovrc[8];
float rt_ovrc[8];
float st_ovrc[8];
float dile_fw[8];
float dile_fc[8];
float lfct[8];
float stct[8];
float rtct[8];
float smcmax[4]; int nroot,iswater,isbarren,isice;bool urban_flag;
};
// Complete native GROWING_GDD; module_sf_noahmplsm.F:10704
__device__ void growing_gdd(const Params&p,float t2m,float dt,float julian,float&gdd,int&ipa,int&iha,int&pgs) {
    float gddday;
    float dayofs2;
    float tdiff;
    float tc;
    tc=SU(t2m,273.15f);
    ipa=1.0f;
    iha=1.0f;
    if((julian<p.pltday)) ipa=0.0f;
    if((julian>=p.hsday)) iha=0.0f;
    if((tc<p.gddtbase)) {
    tdiff=0.0f;
    } else if((tc>=p.gddtcut)) {
    tdiff=SU(p.gddtcut,p.gddtbase);
    } else {
    tdiff=SU(tc,p.gddtbase);
    }
    gdd=MU(MU(AD(gdd,DV(MU(tdiff,dt),86400.0f)),ipa),iha);
    gddday=gdd;
    pgs=1.0f;
    if((gddday>0.0f)) pgs=2.0f;
    if((gddday>=p.gdds1)) pgs=3.0f;
    if((gddday>=p.gdds2)) pgs=4.0f;
    if((gddday>=p.gdds3)) pgs=5.0f;
    if((gddday>=p.gdds4)) pgs=6.0f;
    if((gddday>=p.gdds5)) pgs=7.0f;
    if((julian>=p.hsday)) pgs=8.0f;
    if((julian<p.pltday)) pgs=1.0f;
}

// Complete native CO2FLUX_CROP; module_sf_noahmplsm.F:10421
__device__ void co2flux_crop(const Params&p,float dt,float stc,float psn,float tv,float wroot,float wstres,float foln,int ipa,int iha,int pgs,float&xlai,float&xsai,float&lfmass,float&rtmass,float&stmass,float&fastcp,float&stblcp,float&wood,float&grain,float&gdd,float&gpp,float&npp,float&nee,float&autors,float&heters,float&totsc,float&totlb) {
    float cflux;
    float lfmsmn;
    float rswood;
    float rsleaf;
    float rsroot;
    float rsgrain;
    float nppl;
    float nppr;
    float nppw;
    float npps;
    float nppg;
    float dielf;
    float addnpplf;
    float addnppst;
    float carbfx;
    float cbhydrafx;
    float grleaf;
    float grroot;
    float grwood;
    float grstem;
    float grgrain;
    float leafpt;
    float lfdel;
    float lftovr;
    float sttovr;
    float wdtovr;
    float grtovr;
    float rssoil;
    float rttovr;
    float stablc;
    float woodf;
    float nonlef;
    float resp;
    float rsstem;
    float fsw;
    float fst;
    float fnf;
    float tf;
    float stdel;
    float stmsmn;
    float sapm;
    float diest;
    float lfconvert;
    float stconvert;
    float rtconvert;
    float bf;
    float rswoodc;
    float stovrc;
    float rsdryc;
    float rtovrc;
    float wstrc;
    float laimin;
    float xsamin;
    float sc;
    float sd;
    float vegfrac;
    float temp;
    rsdryc=40.0f;
    rswoodc=3e-10f;
    bf=0.9f;
    wstrc=100.0f;
    laimin=0.05f;
    xsamin=0.05f;
    sapm=MU(3.0f,0.001f);
    lfmsmn=DV(laimin,0.035f);
    stmsmn=DV(xsamin,sapm);
    carbfx=MU(psn,1.2e-05f);
    cbhydrafx=MU(psn,3e-05f);
    fnf=mn(DV(foln,mx(1e-06f,p.foln_mx)),1.0f);
    tf=r_pow(p.q10mr,DV(SU(tv,298.16f),10.0f));
    resp=MU(MU(MU(MU(p.lfmr25,tf),fnf),xlai),SU(1.0f,wstres));
    rsleaf=mn(DV(SU(lfmass,lfmsmn),dt),MU(resp,3e-05f));
    rsroot=MU(MU(MU(p.rtmr25,MU(rtmass,0.001f)),tf),3e-05f);
    rsstem=MU(MU(MU(p.stmr25,MU(stmass,0.001f)),tf),3e-05f);
    rsgrain=MU(MU(MU(p.grainmr25,MU(grain,0.001f)),tf),3e-05f);
    grleaf=mx(0.0f,MU(p.fra_gr,SU(MU(p.lfpt[(int)(pgs)-1],cbhydrafx),rsleaf)));
    grstem=mx(0.0f,MU(p.fra_gr,SU(MU(p.stpt[(int)(pgs)-1],cbhydrafx),rsstem)));
    grroot=mx(0.0f,MU(p.fra_gr,SU(MU(p.rtpt[(int)(pgs)-1],cbhydrafx),rsroot)));
    grgrain=mx(0.0f,MU(p.fra_gr,SU(MU(p.grainpt[(int)(pgs)-1],cbhydrafx),rsgrain)));
    lftovr=MU(MU(p.lf_ovrc[(int)(pgs)-1],1e-06f),lfmass);
    rttovr=MU(MU(p.rt_ovrc[(int)(pgs)-1],1e-06f),rtmass);
    sttovr=MU(MU(p.st_ovrc[(int)(pgs)-1],1e-06f),stmass);
    sc=MU(r_exp(MU(-(0.3f),mx(0.0f,SU(tv,p.lefreez)))),DV(lfmass,120.0f));
    sd=r_exp(MU(SU(wstres,1.0f),wstrc));
    dielf=MU(MU(lfmass,1e-06f),AD(MU(p.dile_fw[(int)(pgs)-1],sd),MU(p.dile_fc[(int)(pgs)-1],sc)));
    addnpplf=SU(SU(MU(p.lfpt[(int)(pgs)-1],cbhydrafx),grleaf),rsleaf);
    addnppst=SU(SU(MU(p.stpt[(int)(pgs)-1],cbhydrafx),grstem),rsstem);
    lfdel=DV(SU(lfmass,lfmsmn),dt);
    stdel=DV(SU(stmass,stmsmn),dt);
    lftovr=mn(lftovr,AD(lfdel,addnpplf));
    sttovr=mn(sttovr,AD(stdel,addnppst));
    dielf=mn(dielf,SU(AD(lfdel,addnpplf),lftovr));
    nppl=addnpplf;
    npps=addnppst;
    nppr=SU(SU(MU(p.rtpt[(int)(pgs)-1],cbhydrafx),rsroot),grroot);
    nppg=SU(SU(MU(p.grainpt[(int)(pgs)-1],cbhydrafx),rsgrain),grgrain);
    lfmass=AD(lfmass,MU(SU(SU(nppl,lftovr),dielf),dt));
    stmass=AD(stmass,MU(SU(npps,sttovr),dt));
    rtmass=AD(rtmass,MU(SU(nppr,rttovr),dt));
    grain=AD(grain,MU(nppg,dt));
    gpp=MU(cbhydrafx,0.4f);
    lfconvert=0.0f;
    stconvert=0.0f;
    rtconvert=0.0f;
    lfconvert=MU(lfmass,DV(MU(p.lfct[(int)(pgs)-1],dt),3600.0f));
    stconvert=MU(stmass,DV(MU(p.stct[(int)(pgs)-1],dt),3600.0f));
    rtconvert=MU(rtmass,DV(MU(p.rtct[(int)(pgs)-1],dt),3600.0f));
    lfmass=SU(lfmass,lfconvert);
    stmass=SU(stmass,stconvert);
    rtmass=SU(rtmass,rtconvert);
    grain=AD(AD(AD(grain,stconvert),rtconvert),lfconvert);
    if((rtmass<0.0f)) {
    rttovr=nppr;
    rtmass=0.0f;
    }
    if((grain<0.0f)) {
    grain=0.0f;
    }
    fastcp=AD(fastcp,MU(MU(AD(AD(AD(rttovr,lftovr),sttovr),dielf),dt),0.4f));
    fst=r_pow(2.0f,DV(SU(stc,283.16f),10.0f));
    fsw=DV(MU(DV(wroot,AD(0.2f,wroot)),0.23f),AD(0.23f,wroot));
    rssoil=MU(MU(MU(MU(fsw,fst),p.mrp),mx(0.0f,MU(fastcp,0.001f))),3e-05f);
    stablc=MU(0.1f,rssoil);
    fastcp=SU(fastcp,MU(MU(AD(rssoil,stablc),dt),0.4f));
    stblcp=AD(stblcp,MU(MU(stablc,dt),0.4f));
    cflux=AD(-(carbfx),MU(AD(AD(AD(AD(AD(AD(AD(AD(rsleaf,rsroot),rsstem),rsgrain),MU(0.9f,rssoil)),grleaf),grroot),grstem),grgrain),0.4f));
    npp=MU(AD(AD(AD(nppl,npps),nppr),nppg),0.4f);
    autors=MU(AD(AD(AD(AD(AD(AD(AD(rsroot,rsgrain),rsleaf),rsstem),grleaf),grroot),grgrain),grstem),0.4f);
    heters=MU(MU(0.9f,rssoil),0.4f);
    nee=DV(MU(SU(AD(autors,heters),gpp),44.0f),12.0f);
    totsc=AD(fastcp,stblcp);
    totlb=MU(AD(AD(AD(lfmass,rtmass),stmass),grain),0.4f);
    xlai=mx(MU(lfmass,p.bio2lai),laimin);
    xsai=mx(MU(stmass,sapm),xsamin);
    if(((pgs==8.0f)&&((grain>0.0f)||(lfmass>0.0f)||(stmass>0.0f)||(rtmass>0.0f)))) {
    xlai=0.05f;
    xsai=0.05f;
    lfmass=lfmsmn;
    stmass=stmsmn;
    rtmass=0.0f;
    grain=0.0f;
    }
}

// Complete native CARBON_CROP; module_sf_noahmplsm.F:10301
__device__ void carbon_crop(const Params&p,int vegtyp,float dt,const float*zsoil,float julian,const float*dzsnso,const float*stc,const float*smc,float tv,float psn,float foln,float btran,float soldn,float t2m,float&lfmass,float&rtmass,float&stmass,float&wood,float&stblcp,float&fastcp,float&grain,float&xlai,float&xsai,float&gdd,float&gpp,float&npp,float&nee,float&autors,float&heters,float&totsc,float&totlb,int&pgs) {
    int nsnow;
    int nsoil;
    int j;
    float wroot;
    float wstres;
    int ipa;
    int iha;
    if(((vegtyp==p.iswater)||(vegtyp==p.isbarren)||(vegtyp==p.isice)||p.urban_flag)) {
    xlai=0.0f;
    xsai=0.0f;
    gpp=0.0f;
    npp=0.0f;
    nee=0.0f;
    autors=0.0f;
    heters=0.0f;
    totsc=0.0f;
    totlb=0.0f;
    lfmass=0.0f;
    rtmass=0.0f;
    stmass=0.0f;
    wood=0.0f;
    stblcp=0.0f;
    fastcp=0.0f;
    grain=0.0f;
    return;
    }
    wstres=SU(1.0f,btran);
    wroot=0.0f;
    for(j=(int)(1.0f);j<=(int)(p.nroot);++j) {
    wroot=AD(wroot,DV(MU(DV(smc[(int)(j)-1+0],p.smcmax[(int)(j)-1]),dzsnso[(int)(j)-1+3]),-(zsoil[(int)(p.nroot)-1+0])));
    }
    growing_gdd(p,t2m,dt,julian,gdd,ipa,iha,pgs);
    co2flux_crop(p,dt,stc[(int)(1.0f)-1+3],psn,tv,wroot,wstres,foln,ipa,iha,pgs,xlai,xsai,lfmass,rtmass,stmass,fastcp,stblcp,wood,grain,gdd,gpp,npp,nee,autors,heters,totsc,totlb);
}
}

extern "C" __global__ void noahmp_crop_flux(const float*x,const int*ix,const float*pars,float*y,int*yi,int n) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 x+=c*44;ix+=c*8;pars+=c*115;y+=c*17;
 noahmp_crop::Params p;
p.q10mr=pars[0];
p.foln_mx=pars[1];
p.lfmr25=pars[2];
p.rtmr25=pars[3];
p.stmr25=pars[4];
p.grainmr25=pars[5];
p.fra_gr=pars[6];
p.lefreez=pars[7];
p.mrp=pars[8];
p.bio2lai=pars[9];
p.gddtbase=pars[10];
p.gddtcut=pars[11];
p.gdds1=pars[12];
p.gdds2=pars[13];
p.gdds3=pars[14];
p.gdds4=pars[15];
p.gdds5=pars[16];
p.pltday=pars[17];
p.hsday=pars[18];
for(int k=0;k<8;++k)p.lfpt[k]=pars[19+k];
for(int k=0;k<8;++k)p.stpt[k]=pars[27+k];
for(int k=0;k<8;++k)p.rtpt[k]=pars[35+k];
for(int k=0;k<8;++k)p.grainpt[k]=pars[43+k];
for(int k=0;k<8;++k)p.lf_ovrc[k]=pars[51+k];
for(int k=0;k<8;++k)p.rt_ovrc[k]=pars[59+k];
for(int k=0;k<8;++k)p.st_ovrc[k]=pars[67+k];
for(int k=0;k<8;++k)p.dile_fw[k]=pars[75+k];
for(int k=0;k<8;++k)p.dile_fc[k]=pars[83+k];
for(int k=0;k<8;++k)p.lfct[k]=pars[91+k];
for(int k=0;k<8;++k)p.stct[k]=pars[99+k];
for(int k=0;k<8;++k)p.rtct[k]=pars[107+k];

 for(int k=0;k<4;++k)p.smcmax[k]=x[40+k];
 p.nroot=ix[1];p.iswater=ix[2];p.isbarren=ix[3];p.isice=ix[4];p.urban_flag=ix[5];
 float lai=x[8],sai=x[9],lfmass=x[10],rtmass=x[11],stmass=x[12],wood=x[13],stblcp=x[14],fastcp=x[15],grain=x[16],gdd=x[17];
 float gpp=0.f,npp=0.f,nee=0.f,autors=0.f,heters=0.f,totsc=0.f,totlb=0.f;int pgs=ix[7];
 if(ix[6]>0)noahmp_crop::carbon_crop(p,ix[0],x[0],x+18,x[5],x+22,x+29,x+36,x[1],x[2],x[3],x[4],x[6],x[7],lfmass,rtmass,stmass,wood,stblcp,fastcp,grain,lai,sai,gdd,gpp,npp,nee,autors,heters,totsc,totlb,pgs);
 const float values[17]={lai,sai,lfmass,rtmass,stmass,wood,stblcp,fastcp,gpp,npp,nee,autors,heters,totsc,totlb,grain,gdd};
 for(int k=0;k<17;++k)y[k]=values[k];yi[c]=pgs;
}
