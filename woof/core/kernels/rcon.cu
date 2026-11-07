// WRF 4.7.1 module_mp_rcon.F:1071-1513; one full native driver column.
extern "C" __global__ void rcon_step(
    float* th,const float* pii,const float* p,const float* w,const float* dz,
    float* qv,float* qc,float* qi,float* qr,float* qs,float* qg,
    float* ni,float* nr,float* nc,float* nwfa,float* nifa,float* nbca,
    const float* nwfa2d,const float* nifa2d,const float* nbca2d,
    float* rainnc,float* rainncv,float* cloudnc,float* snownc,float* snowncv,
    float* graupelnc,float* graupelncv,float* sr,float* refl,
    float* effc,float* effi,float* effs,
    int nz,int columns,float dt,int aer,int wif,int do_refl,int do_radii,int ke_diag,
    const double** tables)
{
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns)return;
    RconColumn m;m.nz=nz;
    // Table bindings are emitted from the retained table schema.
    m.t_efrw=(const double*)tables[0];
m.t_efsw=(const double*)tables[1];
m.tcg_racg=(const double*)tables[2];
m.tcr_gacr=(const double*)tables[3];
m.tcr_sacr1=(const double*)tables[4];
m.tcr_sacr2=(const double*)tables[5];
m.tcs_racs1=(const double*)tables[6];
m.tcs_racs2=(const double*)tables[7];
m.tmr_racg=(const double*)tables[8];
m.tmr_racs1=(const double*)tables[9];
m.tmr_racs2=(const double*)tables[10];
m.tms_sacr1=(const double*)tables[11];
m.tms_sacr2=(const double*)tables[12];
m.tnc_wev=(const double*)tables[13];
m.tnccn_act=(const float*)tables[14];
m.tni_iaus=(const double*)tables[15];
m.tni_qcfz=(const double*)tables[16];
m.tni_qrfz=(const double*)tables[17];
m.tnr_gacr=(const double*)tables[18];
m.tnr_qrfz=(const double*)tables[19];
m.tnr_racg=(const double*)tables[20];
m.tnr_racs1=(const double*)tables[21];
m.tnr_racs2=(const double*)tables[22];
m.tnr_rev=(const double*)tables[23];
m.tnr_sacr1=(const double*)tables[24];
m.tnr_sacr2=(const double*)tables[25];
m.tpc_wev=(const double*)tables[26];
m.tpg_qrfz=(const double*)tables[27];
m.tpi_ide=(const double*)tables[28];
m.tpi_qcfz=(const double*)tables[29];
m.tpi_qrfz=(const double*)tables[30];
m.tps_iaus=(const double*)tables[31];
    float qv1d[RCON_COLUMN_KMAX],qc1d[RCON_COLUMN_KMAX],qi1d[RCON_COLUMN_KMAX];
    float qr1d[RCON_COLUMN_KMAX],qs1d[RCON_COLUMN_KMAX],qg1d[RCON_COLUMN_KMAX];
    float qb1d[RCON_COLUMN_KMAX],ng1d[RCON_COLUMN_KMAX],ni1d[RCON_COLUMN_KMAX];
    float nr1d[RCON_COLUMN_KMAX],nc1d[RCON_COLUMN_KMAX],nwfa1d[RCON_COLUMN_KMAX];
    float nifa1d[RCON_COLUMN_KMAX],nbca1d[RCON_COLUMN_KMAX],t1d[RCON_COLUMN_KMAX];
    float p1d[RCON_COLUMN_KMAX],w1d[RCON_COLUMN_KMAX],dzq[RCON_COLUMN_KMAX];
    float rho[RCON_COLUMN_KMAX],re_c[RCON_COLUMN_KMAX],re_i[RCON_COLUMN_KMAX];
    float re_s[RCON_COLUMN_KMAX],dbz[RCON_COLUMN_KMAX];
    for(int k=0;k<nz;k++) {
        const int idx=k*columns+col;
        t1d[k]=th[idx]*pii[idx];p1d[k]=p[idx];w1d[k]=w[idx];dzq[k]=dz[idx];
        qv1d[k]=qv[idx];qc1d[k]=qc[idx];qi1d[k]=qi[idx];qr1d[k]=qr[idx];
        qs1d[k]=qs[idx];qg1d[k]=qg[idx];ni1d[k]=ni[idx];nr1d[k]=nr[idx];
        rho[k]=tc_div(0.622f*p1d[k],(m.r*t1d[k])*(qv1d[k]+0.622f));
        nc1d[k]=nc[idx];nwfa1d[k]=nwfa[idx];nifa1d[k]=nifa[idx];
        nbca1d[k]=(wif==2)?nbca[idx]:0.0f;
    }
    for(int k=nz-1;k>=0;k--) {
        if(qg1d[k]>m.r1) {
            const float ygra=rcon_log10(tc_max(1.e-9f,qg1d[k]*rho[k]));
            float zans=3.0f+(2.0f/7.0f)*(ygra+8.0f);
            zans=tc_max(2.0f,tc_min(zans,6.0f));
            const double n0=rcon_pow(10.0f,zans);
            const double lam_exp=rcon_pow(tc_div(n0*m.am_g[m.idx_bg1-1]*m.cgg[0],
                                              (double)(rho[k]*qg1d[k])),(double)m.oge1);
            const double lamg=lam_exp*rcon_pow(m.cgg[2]*m.ogg2*m.ogg1,m.obmg);
            ng1d[k]=tc_div((double)(m.cgg[1]*m.ogg3*rho[k]*qg1d[k])*
                               rcon_pow(lamg,(double)m.bm_g),(double)m.am_g[m.idx_bg1-1]);
            ng1d[k]=tc_max(m.r2,tc_div(ng1d[k],rho[k]));
            qb1d[k]=tc_div(qg1d[k],m.rho_g[m.idx_bg1-1]);
        } else {ng1d[k]=0.0f;qb1d[k]=0.0f;}
    }
    float rain=0.0f,cloud=0.0f,snow=0.0f,graupel=0.0f,ice=0.0f;
    m.mp_rcon(aer,wif,qv1d,qc1d,qi1d,qr1d,qs1d,qg1d,qb1d,ni1d,nr1d,nc1d,
              ng1d,nwfa1d,nifa1d,nbca1d,t1d,p1d,w1d,dzq,
              rain,cloud,snow,graupel,ice,1,nz,dt,1,1);
    rainncv[col]=rain+snow+graupel+ice+cloud;
    rainnc[col]=rainnc[col]+rain+snow+graupel+ice+cloud;
    cloudnc[col]=cloudnc[col]+cloud;
    if(snownc&&snowncv){snowncv[col]=snow+ice;snownc[col]=snownc[col]+snow+ice;}
    if(graupelnc&&graupelncv){graupelncv[col]=graupel;graupelnc[col]=graupelnc[col]+graupel;}
    sr[col]=tc_div(snow+graupel+ice,rainncv[col]+1.e-12f);
    nwfa1d[0]=nwfa1d[0]+nwfa2d[col]*dt;
    nifa1d[0]=nifa1d[0]+nifa2d[col]*dt;
    nbca1d[0]=(wif==2)?nbca1d[0]+nbca2d[col]*dt:0.0f;
    for(int k=0;k<nz;k++) {
        const int idx=k*columns+col;
        nc[idx]=nc1d[k];nwfa[idx]=nwfa1d[k];nifa[idx]=nifa1d[k];
        if(nbca)nbca[idx]=(wif==2)?nbca1d[k]:0.0f;
        qv[idx]=qv1d[k];qc[idx]=qc1d[k];qi[idx]=qi1d[k];qr[idx]=qr1d[k];
        qs[idx]=qs1d[k];qg[idx]=qg1d[k];ni[idx]=ni1d[k];nr[idx]=nr1d[k];
        th[idx]=tc_div(t1d[k],pii[idx]);
        if(qv1d[k]<0.0f) {
            qv[idx]=(k>1&&k<nz-3)?tc_max(1.e-7f,0.5f*(qv[idx-columns]+qv[idx+columns])):1.e-7f;
        }
    }
    if(do_refl) {
        m.calc_refl10cm(qv1d,qc1d,qr1d,nr1d,qs1d,qg1d,ng1d,qb1d,t1d,p1d,dbz,1,nz,1,1,ke_diag);
        for(int k=0;k<nz;k++)refl[k*columns+col]=tc_max(-35.0f,dbz[k]);
    }
    if(do_radii) {
        for(int k=0;k<nz;k++){re_c[k]=2.49e-6f;re_i[k]=4.99e-6f;re_s[k]=9.99e-6f;}
        m.calc_effectrad(t1d,p1d,qv1d,qc1d,nc1d,qi1d,ni1d,qs1d,re_c,re_i,re_s,1,nz);
        for(int k=0;k<nz;k++) {
            const int idx=k*columns+col;
            effc[idx]=tc_max(2.49e-6f,tc_min(re_c[k],75.e-6f));
            effi[idx]=tc_max(4.99e-6f,tc_min(re_i[k],125.e-6f));
            effs[idx]=tc_max(9.99e-6f,tc_min(re_s[k],999.e-6f));
        }
    }
}
