namespace noahmp_water_options {
// WRF public domain; licenses/LICENSE-WRF-public-domain.txt.
// Complete native SHALLOWWATERTABLE; source line 9773
__device__ void groundwater_shallowwatertable(const float*smcmax,const float*psisat,const float*bexp,const float*zsoil,const float*dzsnso,const float*smceq,float dt,float*smc,float&wtd,float&smcwtd,float&rech,float&qdrain) {
    int iz;
    int iwtd;
    int kwtd;
    float wtdold;
    float dzup;
    float smceqdeep;
    float zsoil0[NSOIL+1];
    for(int lane_k=0;lane_k<NSOIL;++lane_k)zsoil0[lane_k+1]=zsoil[lane_k];
    zsoil0[(int)(0.0f)-0]=0.0f;
    for(iz=(int)(NSOIL);iz>=(int)(1.0f);iz --) {
    if((FADD(wtd,1e-06f)<zsoil0[(int)(iz)-0])) break;
    }
    iwtd=iz;
    kwtd=FADD(iwtd,1.0f);
    if((kwtd<=NSOIL)) {
    wtdold=wtd;
    if((smc[(int)(kwtd)-1]>smceq[(int)(kwtd)-1])) {
    if((smc[(int)(kwtd)-1]==smcmax[(int)(kwtd)-1])) {
    wtd=zsoil0[(int)(iwtd)-0];
    rech=FMUL(f_neg(FSUB(wtdold,wtd)),FSUB(smcmax[(int)(kwtd)-1],smceq[(int)(kwtd)-1]));
    iwtd=FSUB(iwtd,1.0f);
    kwtd=FSUB(kwtd,1.0f);
    if((kwtd>=1.0f)) {
    if((smc[(int)(kwtd)-1]>smceq[(int)(kwtd)-1])) {
    wtdold=wtd;
    wtd=f_min(FDIV(FADD(FSUB(FMUL(smc[(int)(kwtd)-1],dzsnso[(int)(kwtd)-1]),FMUL(smceq[(int)(kwtd)-1],zsoil0[(int)(iwtd)-0])),FMUL(smcmax[(int)(kwtd)-1],zsoil0[(int)(kwtd)-0])),FSUB(smcmax[(int)(kwtd)-1],smceq[(int)(kwtd)-1])),zsoil0[(int)(iwtd)-0]);
    rech=FSUB(rech,FMUL(FSUB(wtdold,wtd),FSUB(smcmax[(int)(kwtd)-1],smceq[(int)(kwtd)-1])));
    }
    }
    } else {
    wtd=f_min(FDIV(FADD(FSUB(FMUL(smc[(int)(kwtd)-1],dzsnso[(int)(kwtd)-1]),FMUL(smceq[(int)(kwtd)-1],zsoil0[(int)(iwtd)-0])),FMUL(smcmax[(int)(kwtd)-1],zsoil0[(int)(kwtd)-0])),FSUB(smcmax[(int)(kwtd)-1],smceq[(int)(kwtd)-1])),zsoil0[(int)(iwtd)-0]);
    rech=FMUL(f_neg(FSUB(wtdold,wtd)),FSUB(smcmax[(int)(kwtd)-1],smceq[(int)(kwtd)-1]));
    }
    } else {
    wtd=zsoil0[(int)(kwtd)-0];
    rech=FMUL(f_neg(FSUB(wtdold,wtd)),FSUB(smcmax[(int)(kwtd)-1],smceq[(int)(kwtd)-1]));
    kwtd=FADD(kwtd,1.0f);
    iwtd=FADD(iwtd,1.0f);
    if((kwtd<=NSOIL)) {
    wtdold=wtd;
    if((smc[(int)(kwtd)-1]>smceq[(int)(kwtd)-1])) {
    wtd=f_min(FDIV(FADD(FSUB(FMUL(smc[(int)(kwtd)-1],dzsnso[(int)(kwtd)-1]),FMUL(smceq[(int)(kwtd)-1],zsoil0[(int)(iwtd)-0])),FMUL(smcmax[(int)(kwtd)-1],zsoil0[(int)(kwtd)-0])),FSUB(smcmax[(int)(kwtd)-1],smceq[(int)(kwtd)-1])),zsoil0[(int)(iwtd)-0]);
    } else {
    wtd=zsoil0[(int)(kwtd)-0];
    }
    rech=FSUB(rech,FMUL(FSUB(wtdold,wtd),FSUB(smcmax[(int)(kwtd)-1],smceq[(int)(kwtd)-1])));
    } else {
    wtdold=wtd;
    smceqdeep=FMUL(smcmax[(int)(NSOIL)-1],runoff_powf(FDIV(f_neg(psisat[(int)(NSOIL)-1]),FSUB(f_neg(psisat[(int)(NSOIL)-1]),dzsnso[(int)(NSOIL)-1])),FDIV(1.0f,bexp[(int)(NSOIL)-1])));
    wtd=f_min(FDIV(FADD(FSUB(FMUL(smcwtd,dzsnso[(int)(NSOIL)-1]),FMUL(smceqdeep,zsoil0[(int)(NSOIL)-0])),FMUL(smcmax[(int)(NSOIL)-1],FSUB(zsoil0[(int)(NSOIL)-0],dzsnso[(int)(NSOIL)-1]))),FSUB(smcmax[(int)(NSOIL)-1],smceqdeep)),zsoil0[(int)(NSOIL)-0]);
    rech=FSUB(rech,FMUL(FSUB(wtdold,wtd),FSUB(smcmax[(int)(NSOIL)-1],smceqdeep)));
    }
    }
    } else if((wtd>=FSUB(zsoil0[(int)(NSOIL)-0],dzsnso[(int)(NSOIL)-1]))) {
    wtdold=wtd;
    smceqdeep=FMUL(smcmax[(int)(NSOIL)-1],runoff_powf(FDIV(f_neg(psisat[(int)(NSOIL)-1]),FSUB(f_neg(psisat[(int)(NSOIL)-1]),dzsnso[(int)(NSOIL)-1])),FDIV(1.0f,bexp[(int)(NSOIL)-1])));
    if((smcwtd>smceqdeep)) {
    wtd=f_min(FDIV(FADD(FSUB(FMUL(smcwtd,dzsnso[(int)(NSOIL)-1]),FMUL(smceqdeep,zsoil0[(int)(NSOIL)-0])),FMUL(smcmax[(int)(NSOIL)-1],FSUB(zsoil0[(int)(NSOIL)-0],dzsnso[(int)(NSOIL)-1]))),FSUB(smcmax[(int)(NSOIL)-1],smceqdeep)),zsoil0[(int)(NSOIL)-0]);
    rech=FMUL(f_neg(FSUB(wtdold,wtd)),FSUB(smcmax[(int)(NSOIL)-1],smceqdeep));
    } else {
    rech=FMUL(f_neg(FSUB(wtdold,FSUB(zsoil0[(int)(NSOIL)-0],dzsnso[(int)(NSOIL)-1]))),FSUB(smcmax[(int)(NSOIL)-1],smceqdeep));
    wtdold=FSUB(zsoil0[(int)(NSOIL)-0],dzsnso[(int)(NSOIL)-1]);
    dzup=FDIV(FMUL(FSUB(smceqdeep,smcwtd),dzsnso[(int)(NSOIL)-1]),FSUB(smcmax[(int)(NSOIL)-1],smceqdeep));
    wtd=FSUB(wtdold,dzup);
    rech=FSUB(rech,FMUL(FSUB(smcmax[(int)(NSOIL)-1],smceqdeep),dzup));
    smcwtd=smceqdeep;
    }
    }
    if(((iwtd<NSOIL)&&(iwtd>0.0f))) {
    smcwtd=smcmax[(int)(iwtd)-1];
    } else if(((iwtd<NSOIL)&&(iwtd<=0.0f))) {
    smcwtd=smcmax[(int)(1.0f)-1];
    }
}
// Complete native UPDATEWTD; source line 302
__device__ void groundwater_updatewtd(const float*dzs,const float*zsoil,const float*smceq,float smcmax,float smcwlt,float psisat,float bexp,float&totwater,float&wtd,float*smc,float*sh2o,float&smcwtd,float&qspring) {
    int k;
    int k1;
    int iwtd;
    int kwtd;
    float maxwatup;
    float maxwatdw;
    float wtdold;
    float wgpmid;
    float syielddw;
    float dzup;
    float smceqdeep;
    float sice[NSOIL];
    qspring=0.0f;
    for(int lane_k=1;lane_k<=NSOIL;++lane_k) sice[(int)(lane_k)-1]=FSUB(smc[(int)(lane_k)-1],sh2o[(int)(lane_k)-1]);
    iwtd=1.0f;
    if((totwater>0.0f)) {
    if((wtd>=zsoil[(int)(NSOIL)-0])) {
    for(k=(int)(FSUB(NSOIL,1.0f));k>=(int)(1.0f);k --) {
    if((wtd<zsoil[(int)(k)-0])) break;
    }
    iwtd=k;
    kwtd=FADD(iwtd,1.0f);
    maxwatup=FMUL(dzs[(int)(kwtd)-1],FSUB(smcmax,smc[(int)(kwtd)-1]));
    if((totwater<=maxwatup)) {
    smc[(int)(kwtd)-1]=FADD(smc[(int)(kwtd)-1],FDIV(totwater,dzs[(int)(kwtd)-1]));
    smc[(int)(kwtd)-1]=f_min(smc[(int)(kwtd)-1],smcmax);
    if((smc[(int)(kwtd)-1]>smceq[(int)(kwtd)-1])) wtd=f_min(FDIV(FADD(FSUB(FMUL(smc[(int)(kwtd)-1],dzs[(int)(kwtd)-1]),FMUL(smceq[(int)(kwtd)-1],zsoil[(int)(iwtd)-0])),FMUL(smcmax,zsoil[(int)(kwtd)-0])),FSUB(smcmax,smceq[(int)(kwtd)-1])),zsoil[(int)(iwtd)-0]);
    totwater=0.0f;
    } else {
    smc[(int)(kwtd)-1]=smcmax;
    totwater=FSUB(totwater,maxwatup);
    k1=iwtd;
    for(k=(int)(k1);k>=(int)(0.0f);k --) {
    wtd=zsoil[(int)(k)-0];
    iwtd=FSUB(k,1.0f);
    if((k==0.0f)) break;
    maxwatup=FMUL(dzs[(int)(k)-1],FSUB(smcmax,smc[(int)(k)-1]));
    if((totwater<=maxwatup)) {
    smc[(int)(k)-1]=FADD(smc[(int)(k)-1],FDIV(totwater,dzs[(int)(k)-1]));
    smc[(int)(k)-1]=f_min(smc[(int)(k)-1],smcmax);
    if((smc[(int)(k)-1]>smceq[(int)(k)-1])) wtd=f_min(FDIV(FADD(FSUB(FMUL(smc[(int)(k)-1],dzs[(int)(k)-1]),FMUL(smceq[(int)(k)-1],zsoil[(int)(iwtd)-0])),FMUL(smcmax,zsoil[(int)(k)-0])),FSUB(smcmax,smceq[(int)(k)-1])),zsoil[(int)(iwtd)-0]);
    totwater=0.0f;
    break;
    } else {
    smc[(int)(k)-1]=smcmax;
    totwater=FSUB(totwater,maxwatup);
    }
    }
    }
    } else if((wtd>=FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]))) {
    smceqdeep=FMUL(smcmax,runoff_powf(FDIV(psisat,FSUB(psisat,dzs[(int)(NSOIL)-1])),FDIV(1.0f,bexp)));
    smceqdeep=f_max(smceqdeep,0.0001f);
    maxwatup=FMUL(FSUB(smcmax,smcwtd),dzs[(int)(NSOIL)-1]);
    if((totwater<=maxwatup)) {
    smcwtd=FADD(smcwtd,FDIV(totwater,dzs[(int)(NSOIL)-1]));
    smcwtd=f_min(smcwtd,smcmax);
    if((smcwtd>smceqdeep)) wtd=f_min(FDIV(FADD(FSUB(FMUL(smcwtd,dzs[(int)(NSOIL)-1]),FMUL(smceqdeep,zsoil[(int)(NSOIL)-0])),FMUL(smcmax,FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]))),FSUB(smcmax,smceqdeep)),zsoil[(int)(NSOIL)-0]);
    totwater=0.0f;
    } else {
    smcwtd=smcmax;
    totwater=FSUB(totwater,maxwatup);
    for(k=(int)(NSOIL);k>=(int)(0.0f);k --) {
    wtd=zsoil[(int)(k)-0];
    iwtd=FSUB(k,1.0f);
    if((k==0.0f)) break;
    maxwatup=FMUL(dzs[(int)(k)-1],FSUB(smcmax,smc[(int)(k)-1]));
    if((totwater<=maxwatup)) {
    smc[(int)(k)-1]=f_min(FADD(smc[(int)(k)-1],FDIV(totwater,dzs[(int)(k)-1])),smcmax);
    if((smc[(int)(k)-1]>smceq[(int)(k)-1])) wtd=f_min(FDIV(FADD(FSUB(FMUL(smc[(int)(k)-1],dzs[(int)(k)-1]),FMUL(smceq[(int)(k)-1],zsoil[(int)(iwtd)-0])),FMUL(smcmax,zsoil[(int)(k)-0])),FSUB(smcmax,smceq[(int)(k)-1])),zsoil[(int)(iwtd)-0]);
    totwater=0.0f;
    break;
    } else {
    smc[(int)(k)-1]=smcmax;
    totwater=FSUB(totwater,maxwatup);
    }
    }
    }
    } else {
    maxwatup=FMUL(FSUB(smcmax,smcwtd),FSUB(FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]),wtd));
    if((totwater<=maxwatup)) {
    wtd=FADD(wtd,FDIV(totwater,FSUB(smcmax,smcwtd)));
    totwater=0.0f;
    } else {
    totwater=FSUB(totwater,maxwatup);
    wtd=FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]);
    maxwatup=FMUL(FSUB(smcmax,smcwtd),dzs[(int)(NSOIL)-1]);
    if((totwater<=maxwatup)) {
    smceqdeep=FMUL(smcmax,runoff_powf(FDIV(psisat,FSUB(psisat,dzs[(int)(NSOIL)-1])),FDIV(1.0f,bexp)));
    smceqdeep=f_max(smceqdeep,0.0001f);
    smcwtd=FADD(smcwtd,FDIV(totwater,dzs[(int)(NSOIL)-1]));
    smcwtd=f_min(smcwtd,smcmax);
    wtd=FDIV(FADD(FSUB(FMUL(smcwtd,dzs[(int)(NSOIL)-1]),FMUL(smceqdeep,zsoil[(int)(NSOIL)-0])),FMUL(smcmax,FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]))),FSUB(smcmax,smceqdeep));
    totwater=0.0f;
    } else {
    smcwtd=smcmax;
    totwater=FSUB(totwater,maxwatup);
    for(k=(int)(NSOIL);k>=(int)(0.0f);k --) {
    wtd=zsoil[(int)(k)-0];
    iwtd=FSUB(k,1.0f);
    if((k==0.0f)) break;
    maxwatup=FMUL(dzs[(int)(k)-1],FSUB(smcmax,smc[(int)(k)-1]));
    if((totwater<=maxwatup)) {
    smc[(int)(k)-1]=FADD(smc[(int)(k)-1],FDIV(totwater,dzs[(int)(k)-1]));
    smc[(int)(k)-1]=f_min(smc[(int)(k)-1],smcmax);
    if((smc[(int)(k)-1]>smceq[(int)(k)-1])) wtd=FDIV(FADD(FSUB(FMUL(smc[(int)(k)-1],dzs[(int)(k)-1]),FMUL(smceq[(int)(k)-1],zsoil[(int)(iwtd)-0])),FMUL(smcmax,zsoil[(int)(k)-0])),FSUB(smcmax,smceq[(int)(k)-1]));
    totwater=0.0f;
    break;
    } else {
    smc[(int)(k)-1]=smcmax;
    totwater=FSUB(totwater,maxwatup);
    }
    }
    }
    }
    }
    qspring=totwater;
    } else if((totwater<0.0f)) {
    if((wtd>=zsoil[(int)(NSOIL)-0])) {
    for(k=(int)(FSUB(NSOIL,1.0f));k>=(int)(1.0f);k --) {
    if((wtd<zsoil[(int)(k)-0])) break;
    }
    iwtd=k;
    k1=FADD(iwtd,1.0f);
    for(kwtd=(int)(k1);kwtd<=(int)(NSOIL);kwtd ++) {
    maxwatdw=FMUL(dzs[(int)(kwtd)-1],FSUB(smc[(int)(kwtd)-1],f_max(smceq[(int)(kwtd)-1],sice[(int)(kwtd)-1])));
    if((f_neg(totwater)<=maxwatdw)) {
    smc[(int)(kwtd)-1]=FADD(smc[(int)(kwtd)-1],FDIV(totwater,dzs[(int)(kwtd)-1]));
    if((smc[(int)(kwtd)-1]>smceq[(int)(kwtd)-1])) {
    wtd=FDIV(FADD(FSUB(FMUL(smc[(int)(kwtd)-1],dzs[(int)(kwtd)-1]),FMUL(smceq[(int)(kwtd)-1],zsoil[(int)(iwtd)-0])),FMUL(smcmax,zsoil[(int)(kwtd)-0])),FSUB(smcmax,smceq[(int)(kwtd)-1]));
    } else {
    wtd=zsoil[(int)(kwtd)-0];
    iwtd=FADD(iwtd,1.0f);
    }
    totwater=0.0f;
    break;
    } else {
    wtd=zsoil[(int)(kwtd)-0];
    iwtd=FADD(iwtd,1.0f);
    if((maxwatdw>=0.0f)) {
    smc[(int)(kwtd)-1]=FADD(smc[(int)(kwtd)-1],FDIV(maxwatdw,dzs[(int)(kwtd)-1]));
    totwater=FADD(totwater,maxwatdw);
    }
    }
    }
    if(((iwtd==NSOIL)&&(totwater<0.0f))) {
    smceqdeep=FMUL(smcmax,runoff_powf(FDIV(psisat,FSUB(psisat,dzs[(int)(NSOIL)-1])),FDIV(1.0f,bexp)));
    smceqdeep=f_max(smceqdeep,0.0001f);
    maxwatdw=FMUL(dzs[(int)(NSOIL)-1],FSUB(smcwtd,smceqdeep));
    if((f_neg(totwater)<=maxwatdw)) {
    smcwtd=FADD(smcwtd,FDIV(totwater,dzs[(int)(NSOIL)-1]));
    wtd=f_max(FDIV(FADD(FSUB(FMUL(smcwtd,dzs[(int)(NSOIL)-1]),FMUL(smceqdeep,zsoil[(int)(NSOIL)-0])),FMUL(smcmax,FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]))),FSUB(smcmax,smceqdeep)),FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]));
    } else {
    wtd=FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]);
    smcwtd=FADD(smcwtd,FDIV(totwater,dzs[(int)(NSOIL)-1]));
    dzup=FDIV(FMUL(FSUB(smceqdeep,smcwtd),dzs[(int)(NSOIL)-1]),FSUB(smcmax,smceqdeep));
    wtd=FSUB(wtd,dzup);
    smcwtd=smceqdeep;
    }
    }
    } else if((wtd>=FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]))) {
    smceqdeep=FMUL(smcmax,runoff_powf(FDIV(psisat,FSUB(psisat,dzs[(int)(NSOIL)-1])),FDIV(1.0f,bexp)));
    smceqdeep=f_max(smceqdeep,0.0001f);
    maxwatdw=FMUL(dzs[(int)(NSOIL)-1],FSUB(smcwtd,smceqdeep));
    if((f_neg(totwater)<=maxwatdw)) {
    smcwtd=FADD(smcwtd,FDIV(totwater,dzs[(int)(NSOIL)-1]));
    wtd=f_max(FDIV(FADD(FSUB(FMUL(smcwtd,dzs[(int)(NSOIL)-1]),FMUL(smceqdeep,zsoil[(int)(NSOIL)-0])),FMUL(smcmax,FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]))),FSUB(smcmax,smceqdeep)),FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]));
    } else {
    wtd=FSUB(zsoil[(int)(NSOIL)-0],dzs[(int)(NSOIL)-1]);
    smcwtd=FADD(smcwtd,FDIV(totwater,dzs[(int)(NSOIL)-1]));
    dzup=FDIV(FMUL(FSUB(smceqdeep,smcwtd),dzs[(int)(NSOIL)-1]),FSUB(smcmax,smceqdeep));
    wtd=FSUB(wtd,dzup);
    smcwtd=smceqdeep;
    }
    } else {
    wgpmid=FMUL(smcmax,runoff_powf(FDIV(psisat,FSUB(psisat,FSUB(zsoil[(int)(NSOIL)-0],wtd))),FDIV(1.0f,bexp)));
    wgpmid=f_max(wgpmid,0.0001f);
    syielddw=FSUB(smcmax,wgpmid);
    wtdold=wtd;
    wtd=FADD(wtdold,FDIV(totwater,syielddw));
    smcwtd=FDIV(FADD(FMUL(smcwtd,FSUB(zsoil[(int)(NSOIL)-0],wtdold)),FMUL(wgpmid,FSUB(wtdold,wtd))),FSUB(zsoil[(int)(NSOIL)-0],wtd));
    }
    qspring=0.0f;
    }
    for(int lane_k=1;lane_k<=NSOIL;++lane_k) sh2o[(int)(lane_k)-1]=FSUB(smc[(int)(lane_k)-1],sice[(int)(lane_k)-1]);
}
// Complete native EQSMOISTURE; source line 2759
__device__ void groundwater_eqsmoisture(const float*zsoil,float smcmax,float smcwlt,float dwsat,float dksat,float bexp,float*smceq) {
    int k;
    int iter;
    float ddz;
    float smc;
    float func;
    float dfunc;
    float aa;
    float bb;
    float expon;
    float dx;
    for(k=(int)(1.0f);k<=(int)(NSOIL);k ++) {
    if((k==1.0f)) {
    ddz=FMUL(f_neg(zsoil[(int)(FADD(k,1.0f))-1]),0.5f);
    } else if((k<NSOIL)) {
    ddz=FMUL(FSUB(zsoil[(int)(FSUB(k,1.0f))-1],zsoil[(int)(FADD(k,1.0f))-1]),0.5f);
    } else {
    ddz=FSUB(zsoil[(int)(FSUB(k,1.0f))-1],zsoil[(int)(k)-1]);
    }
    expon=FADD(bexp,1.0f);
    aa=FDIV(dwsat,ddz);
    bb=FDIV(dksat,runoff_powf(smcmax,expon));
    smc=FMUL(0.5f,smcmax);
    for(iter =(int)(1.0f);iter <=(int)(100.0f);iter  ++) {
    func=FADD(FMUL(FSUB(smc,smcmax),aa),FMUL(bb,runoff_powf(smc,expon)));
    dfunc=FADD(aa,FMUL(FMUL(bb,expon),runoff_powf(smc,bexp)));
    dx=FDIV(func,dfunc);
    smc=FSUB(smc,dx);
    if((fabsf(dx)<1e-06f)) break;
    }
    smceq[(int)(k)-1]=f_min(f_max(smc,0.0001f),FMUL(smcmax,0.99f));
    }
}
// Complete native GW_COLD; source line 1
__device__ void groundwater_gw_cold(const float*zsoil,const float*dzs,float bexp,float smcmax,float smcwlt,float dwsat,float dksat,float psisat,float deltat,float qlat,float qrf,float&wtd,float*smceq,float*smoi,float*sh2o,float&smcwtd) {
    int k;
    int iter;
    float frliq;
    float smceqdeep;
    float aa;
    float bbb;
    float cc;
    float dd;
    float dx;
    float func;
    float dfunc;
    float ddz;
    float expon;
    float smc;
    float flux;
    if(((bexp>0.0f)&&(smcmax>0.0f)&&(f_neg(psisat)>0.0f))) {
    groundwater_eqsmoisture(zsoil,smcmax,smcwlt,dwsat,dksat,bexp,smceq);
    if((wtd<FSUB(zsoil[(int)(NSOIL)-1],dzs[(int)(NSOIL)-1]))) {
    expon=FADD(FMUL(2.0f,bexp),3.0f);
    ddz=FSUB(zsoil[(int)(NSOIL)-1],wtd);
    cc=FDIV(psisat,ddz);
    flux=FDIV(FSUB(qlat,qrf),deltat);
    smc=FMUL(0.5f,smcmax);
    for(iter =(int)(1.0f);iter <=(int)(100.0f);iter  ++) {
    dd=FDIV(FADD(smc,smcmax),FMUL(2.0f,smcmax));
    aa=FMUL(f_neg(dksat),runoff_powf(dd,expon));
    bbb=FADD(FMUL(cc,FSUB(runoff_powf(FDIV(smcmax,smc),bexp),1.0f)),1.0f);
    func=FSUB(FMUL(aa,bbb),flux);
    dfunc=FADD(FMUL(FMUL(FMUL(f_neg(dksat),FDIV(expon,FMUL(2.0f,smcmax))),runoff_powf(dd,FSUB(expon,1.0f))),bbb),FMUL(FMUL(FMUL(FMUL(aa,cc),f_neg(bexp)),runoff_powf(smcmax,bexp)),runoff_powf(smc,FSUB(f_neg(bexp),1.0f))));
    dx=FDIV(func,dfunc);
    smc=FSUB(smc,dx);
    if((fabsf(dx)<1e-06f)) break;
    }
    smcwtd=f_max(smc,0.0001f);
    } else if((wtd<zsoil[(int)(NSOIL)-1])) {
    smceqdeep=FMUL(smcmax,runoff_powf(FDIV(psisat,FSUB(psisat,dzs[(int)(NSOIL)-1])),FDIV(1.0f,bexp)));
    smceqdeep=f_max(smceqdeep,0.0001f);
    smcwtd=FADD(FMUL(smcmax,FSUB(wtd,FSUB(zsoil[(int)(NSOIL)-1],dzs[(int)(NSOIL)-1]))),FMUL(smceqdeep,FSUB(zsoil[(int)(NSOIL)-1],wtd)));
    } else {
    smcwtd=smcmax;
    for(k=(int)(NSOIL);k>=(int)(2.0f);k --) {
    if((wtd>=zsoil[(int)(FSUB(k,1.0f))-1])) {
    frliq=FDIV(sh2o[(int)(k)-1],smoi[(int)(k)-1]);
    smoi[(int)(k)-1]=smcmax;
    sh2o[(int)(k)-1]=FMUL(smcmax,frliq);
    } else {
    if((smoi[(int)(k)-1]<smceq[(int)(k)-1])) {
    wtd=zsoil[(int)(k)-1];
    } else {
    wtd=FDIV(FADD(FSUB(FMUL(smoi[(int)(k)-1],dzs[(int)(k)-1]),FMUL(smceq[(int)(k)-1],zsoil[(int)(FSUB(k,1.0f))-1])),FMUL(smcmax,zsoil[(int)(k)-1])),FSUB(smcmax,smceq[(int)(k)-1]));
    }
    break;
    }
    }
    }
    } else {
    for(int lane_k=0;lane_k<NSOIL;++lane_k)smceq[lane_k]=smcmax;
    smcwtd=smcmax;
    wtd=0.0f;
    }
}
}
// Complete spatial WRF-v4 Miguez-Macho/Fan groundwater coupling.
// Original WRF public domain. Source pin: noahmp_options_wrf461_oracle.
namespace noahmp_water_options {
__constant__ float groundwater_klatfactor[19]={2,3,4,10,10,12,14,20,24,28,40,48,2,0,10,0,20,2,2};
__device__ float groundwater_river(const float*s,float wtd,float dt) {
 float rc=s[6];if(wtd>s[5]&&s[3]>s[5])rc=FMUL(rc,glibc_expf(FMUL(s[4],FSUB(wtd,s[3]))));
 return f_max(FDIV(FMUL(FMUL(rc,FSUB(wtd,s[5])),dt),s[1]),0.0f);
}
}
// Static rows: FDEPTH,AREA,TOPO,EQWTD,PEXP,RIVERBED,RIVERCOND,RECHCLIM,XLAND,XICE.
// Parameter rows: BEXP,DKSAT,negative PSISAT,SMCMAX,SMCWLT,DWSAT.
// State rows: WTD,SMCWTD,DEEPRECH,QSLAT,QRFS,QSPRINGS,RECH,QLAT,QRF,QSPRING.
extern "C" __global__ void k_groundwater_heads(const float*p,const float*s,const float*y,const int*soil,float*kh,int n) {
 using namespace noahmp_water_options;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 const float*a=s+(size_t)c*10;float w=y[(size_t)c*10],k=0.0f;
 if(a[0]>0.0f){float kl=FMUL(p[(size_t)c*6+1],groundwater_klatfactor[soil[c]-1]);k=w<-1.5f?FMUL(FMUL(a[0],kl),glibc_expf(FDIV(FADD(w,1.5f),a[0]))):FMUL(kl,FADD(FADD(w,1.5f),a[0]));}
 kh[(size_t)c*2]=k;kh[(size_t)c*2+1]=FADD(a[2],w);
}
extern "C" __global__ void k_groundwater_lateral(const float*s,const float*kh,const int*land,float*y,int nx,int ny,float dt) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nx*ny)return;int i=c%nx,j=c/nx;float q=0.0f;
 if(i>0&&j>0&&i<nx-1&&j<ny-1&&land[c]){
  int offsets[8]={nx-1,-1,-nx-1,nx,-nx,nx+1,1,-nx+1};
  for(int k=0;k<8;++k){int nb=c+offsets[k];float f=FMUL(FADD(kh[(size_t)nb*2],kh[(size_t)c*2]),FSUB(kh[(size_t)nb*2+1],kh[(size_t)c*2+1]));if(k==0||k==2||k==5||k==7)f=FDIV(f,1.41421353816986083984375f);q=FADD(q,f);}
  q=FDIV(FMUL(FMUL(0.22754493f,q),dt),s[(size_t)c*10+1]);
 }
 y[(size_t)c*10+7]=q;
}
extern "C" __global__ void k_groundwater_update(const float*p,const float*s,const float*dz,const int*land,float*y,float*z,int n,float dt) {
 using namespace noahmp_water_options;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 const float*a=s+(size_t)c*10;const float*b=p+(size_t)c*6;float*r=y+(size_t)c*10,*v=z+(size_t)c*12;
 float zsoil[5]={0.0f};for(int k=1;k<=4;++k)zsoil[k]=FSUB(zsoil[k-1],dz[k-1]);
 r[8]=land[c]?groundwater_river(a,r[0],dt):0.0f;
 if(land[c]){
  if(r[0]<FSUB(zsoil[4],dz[3])){
   float ddz=FSUB(zsoil[4],r[0]),mid=FMUL(0.5f,FADD(r[1],b[3]));
   float psi=FMUL(b[2],runoff_powf(FDIV(b[3],r[1]),b[0]));
   float wc=FMUL(b[1],runoff_powf(FDIV(mid,b[3]),FADD(FMUL(2.0f,b[0]),3.0f)));
   float flux=FMUL(FMUL(f_neg(dt),wc),FSUB(FDIV(FSUB(b[2],psi),ddz),1.0f));
   r[1]=FADD(r[1],FDIV(FSUB(r[2],flux),ddz));
   float plus=FMUL(f_max(FSUB(r[1],b[3]),0.0f),ddz),minus=FMUL(f_max(FSUB(1.0e-4f,r[1]),0.0f),ddz);
   r[1]=f_max(f_min(r[1],b[3]),1.0e-4f);r[2]=FSUB(FADD(flux,plus),minus);
  }
  float water=FADD(FSUB(r[7],r[8]),r[2]);
  groundwater_updatewtd(dz,zsoil,v+8,b[3],b[4],b[2],b[0],water,r[0],v,v+4,r[1],r[9]);
 }
 r[3]=FADD(r[3],FMUL(r[7],1000.0f));r[4]=FADD(r[4],FMUL(r[8],1000.0f));r[5]=FADD(r[5],FMUL(r[9],1000.0f));r[6]=FADD(r[6],FMUL(r[2],1000.0f));r[2]=0.0f;
}
extern "C" __global__ void k_groundwater_cold_adjust(float*y,const int*land,int n) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n&&land[c]&&y[(size_t)c*10+7]>0.01f)y[(size_t)c*10]=f_min(FADD(y[(size_t)c*10],0.25f),0.0f);
}
extern "C" __global__ void k_groundwater_cold_river(float*s,const float*y,int n) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;float*a=s+(size_t)c*10;const float*r=y+(size_t)c*10;
 a[3]=r[0];float ddz=FSUB(a[3],FSUB(a[5],a[2]));if(ddz<0.0f){a[5]=FADD(a[2],a[3]);ddz=0.0f;}
 float total=FDIV(FMUL(a[1],FADD(r[7],FMUL(a[7],0.001f))),31536000.0f);
 if(total>0.0f)a[6]=FDIV(total,f_max(ddz,0.05f));else{a[6]=0.01f;a[5]=FADD(a[2],a[3]);}
 a[5]=f_min(FSUB(a[5],a[2]),0.0f);
}
extern "C" __global__ void k_groundwater_cold_soil(const float*p,const float*s,const float*dz,const int*land,float*y,float*z,int n,float dt) {
 using namespace noahmp_water_options;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 const float*a=s+(size_t)c*10,*b=p+(size_t)c*6;float*r=y+(size_t)c*10,*v=z+(size_t)c*12;
 float zsoil[4];zsoil[0]=f_neg(dz[0]);for(int k=1;k<4;++k)zsoil[k]=FSUB(zsoil[k-1],dz[k]);
 float qrf=land[c]?groundwater_river(a,r[0],dt):0.0f;
 groundwater_gw_cold(zsoil,dz,b[0],b[3],b[4],b[5],b[1],b[2],dt,r[7],qrf,r[0],v+8,v,v+4,r[1]);
 for(int k=2;k<10;++k)r[k]=0.0f;
}
