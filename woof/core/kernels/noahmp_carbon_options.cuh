// WRF Noah-MP v4 848f54ad3d28c4303151fe5ad83724e232694422.
// module_sf_noahmplsm.F:9917-10298, complete CARBON/CO2FLUX.
// Public domain: licenses/LICENSE-WRF-public-domain.txt. No v5/MPAS code.
namespace noahmp_carbon {
__device__ float mn(float a,float b) {return a<b?a:b;}
__device__ float mx(float a,float b) {return a>b?a:b;}
}
extern "C" __global__ void noahmp_carbon_flux(
    const float* x,const int* ix,float* y,int n) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    x+=c*47;ix+=c*6;y+=c*15;
    using noahmp_carbon::mn;using noahmp_carbon::mx;
    if(ix[0]==ix[2]||ix[0]==ix[3]||ix[0]==ix[4]||ix[5]) {
        for(int k=0;k<15;++k)y[k]=0.0f;return;
    }
    const float dt=x[0],tv=x[1],psn=x[2],foln=x[3],btran=x[4],igs=x[5];
    float lai=x[6],sai=x[7],lfmass=x[8],rtmass=x[9],stmass=x[10],wood=x[11],stblcp=x[12],fastcp=x[13];
    const float sla=x[14],folnmx=x[15],arm=x[16],rmf25=x[17],rmr25=x[18],rms25=x[19],wdpool=x[20],wrrat=x[21],ltovrc=x[22],tdlef=x[23],dilefw=x[24],dilefc=x[25],fragr=x[26],tmin=x[27],mrp=x[28];
    const float lapm=DV(sla,1000.0f),sapm=.003f;
    float wroot=0.0f;
    for(int j=0;j<ix[1];++j)wroot=AD(wroot,DV(MU(DV(x[29+j],x[33+j]),x[37+j]),-x[41+ix[1]-1]));
    const float wstres=SU(1.0f,btran),lfmsmn=DV(.05f,lapm),stmsmn=DV(.05f,sapm);
    const float rf=igs==0.0f?.5f:1.0f;
    const float fnf=mn(DV(foln,mx(1.0e-6f,folnmx)),1.0f);
    const float tf=r_pow(arm,DV(SU(tv,298.16f),10.0f));
    float resp=MU(MU(MU(MU(MU(rmf25,tf),fnf),lai),rf),SU(1.0f,wstres));
    float rsleaf=mn(DV(SU(lfmass,lfmsmn),dt),MU(resp,12.0e-6f));
    float rsroot=MU(MU(MU(MU(rmr25,MU(rtmass,1.0e-3f)),tf),rf),12.0e-6f);
    float rsstem=MU(MU(MU(MU(rms25,MU(SU(stmass,stmsmn),1.0e-3f)),tf),rf),12.0e-6f);
    float rswood=MU(MU(MU(3.0e-10f,r_exp(MU(.08f,SU(tv,298.16f)))),wood),wdpool);
    float carbfx=MU(psn,12.0e-6f);
    float leafpt=r_exp(MU(MU(.01f,SU(1.0f,r_exp(MU(.75f,lai)))),lai));
    if(ix[0]==(int)x[46])leafpt=r_exp(MU(MU(.01f,SU(1.0f,r_exp(MU(.50f,lai)))),lai));
    float nonlef=SU(1.0f,leafpt),stempt=MU(DV(lai,10.0f),leafpt);leafpt=SU(leafpt,stempt);
    float woodf=wood>1.0e-6f?MU(SU(1.0f,DV(r_exp(MU(-.90f,DV(MU(wrrat,rtmass),wood))),.90f)),wdpool):wdpool;
    float rootpt=MU(nonlef,SU(1.0f,woodf)),woodpt=MU(nonlef,woodf);
    float lftovr=MU(MU(ltovrc,5.0e-7f),lfmass),sttovr=MU(MU(ltovrc,5.0e-7f),stmass),rttovr=MU(2.0e-8f,rtmass),wdtovr=MU(9.5e-10f,wood);
    float sc=MU(r_exp(MU(-.3f,mx(0.0f,SU(tv,tdlef)))),DV(lfmass,120.0f));
    float sd=r_exp(MU(SU(wstres,1.0f),100.0f));
    float dielf=MU(MU(lfmass,1.0e-6f),AD(MU(dilefw,sd),MU(dilefc,sc)));
    float diest=MU(MU(stmass,1.0e-6f),AD(MU(dilefw,sd),MU(dilefc,sc)));
    float grleaf=mx(0.0f,MU(fragr,SU(MU(leafpt,carbfx),rsleaf)));
    float grstem=mx(0.0f,MU(fragr,SU(MU(stempt,carbfx),rsstem)));
    float grroot=mx(0.0f,MU(fragr,SU(MU(rootpt,carbfx),rsroot)));
    float grwood=mx(0.0f,MU(fragr,SU(MU(woodpt,carbfx),rswood)));
    float addnpplf=mx(0.0f,SU(SU(MU(leafpt,carbfx),grleaf),rsleaf));
    float addnppst=mx(0.0f,SU(SU(MU(stempt,carbfx),grstem),rsstem));
    if(tv<tmin){addnpplf=0.0f;addnppst=0.0f;}
    float lfdel=DV(SU(lfmass,lfmsmn),dt),stdel=DV(SU(stmass,stmsmn),dt);
    dielf=mn(dielf,SU(AD(lfdel,addnpplf),lftovr));diest=mn(diest,SU(AD(stdel,addnppst),sttovr));
    float nppl=mx(addnpplf,-lfdel),npps=mx(addnppst,-stdel),nppr=SU(SU(MU(rootpt,carbfx),rsroot),grroot),nppw=SU(SU(MU(woodpt,carbfx),rswood),grwood);
    lfmass=AD(lfmass,MU(SU(SU(nppl,lftovr),dielf),dt));
    stmass=AD(stmass,MU(SU(SU(npps,sttovr),diest),dt));
    rtmass=AD(rtmass,MU(SU(nppr,rttovr),dt));
    if(rtmass<0.0f){rttovr=nppr;rtmass=0.0f;}
    wood=MU(AD(wood,MU(SU(nppw,wdtovr),dt)),wdpool);
    fastcp=AD(fastcp,MU(AD(AD(AD(AD(AD(rttovr,lftovr),sttovr),wdtovr),dielf),diest),dt));
    float fst=r_pow(2.0f,DV(SU(x[45],283.16f),10.0f));
    float fsw=DV(MU(DV(wroot,AD(.20f,wroot)),.23f),AD(.23f,wroot));
    float rssoil=MU(MU(MU(MU(fsw,fst),mrp),mx(0.0f,MU(fastcp,1.0e-3f))),12.0e-6f);
    float stablc=MU(.1f,rssoil);fastcp=SU(fastcp,MU(AD(rssoil,stablc),dt));stblcp=AD(stblcp,MU(stablc,dt));
    float gpp=carbfx,npp=AD(AD(AD(nppl,nppw),nppr),npps);
    float autors=AD(AD(AD(AD(AD(AD(AD(rsroot,rswood),rsleaf),rsstem),grleaf),grroot),grwood),grstem);
    float heters=MU(.9f,rssoil),nee=DV(MU(SU(AD(autors,heters),gpp),44.0f),12.0f);
    float totsc=AD(fastcp,stblcp),totlb=AD(AD(AD(lfmass,rtmass),stmass),wood);
    lai=mx(MU(lfmass,lapm),.05f);sai=mx(MU(stmass,sapm),.05f);
    y[0]=lai;y[1]=sai;y[2]=lfmass;y[3]=rtmass;y[4]=stmass;y[5]=wood;y[6]=stblcp;y[7]=fastcp;
    y[8]=gpp;y[9]=npp;y[10]=nee;y[11]=autors;y[12]=heters;y[13]=totsc;y[14]=totlb;
}
