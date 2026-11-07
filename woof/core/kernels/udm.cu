// WRF 4.7.1 module_mp_udm.F:310-420, complete written driver boundary.
// Radius coupling is refused: native passes uninitialized den1d at :411.
extern "C" __global__ void udm_step(float** f,const float* s,int nz,int columns,
    int first,int radar,const double** tables) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns)return;
    UdmColumn m;m.nz=nz;
    m.tbaut_nc_sub=(const double*)tables[0];
m.tbaut_nc_tot=(const double*)tables[1];
m.tbaut_qc_sub=(const double*)tables[2];
m.tbaut_qc_tot=(const double*)tables[3];
m.tbshape=(const float*)tables[4];
m.tbsvp=(const float*)tables[5];
m.tbsvpw=(const float*)tables[6];
    float t[UDM_COLUMN_KMAX],q[UDM_COLUMN_KMAX];
    float qci[2*UDM_COLUMN_KMAX],qrs[4*UDM_COLUMN_KMAX],ncr[3*UDM_COLUMN_KMAX];
    float den[UDM_COLUMN_KMAX],p[UDM_COLUMN_KMAX],dz[UDM_COLUMN_KMAX];
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        t[k]=f[0][idx]*f[12][idx];q[k]=f[1][idx];
        qci[k]=f[2][idx];qci[k+nz]=f[4][idx];
        qrs[k]=f[3][idx];qrs[k+nz]=f[5][idx];
        qrs[k+2*nz]=f[6][idx];qrs[k+3*nz]=f[7][idx];
        ncr[k]=first?s[4]:f[8][idx];ncr[k+nz]=f[9][idx];ncr[k+2*nz]=f[10][idx];
        den[k]=f[11][idx];p[k]=f[13][idx];dz[k]=f[14][idx];
    }
    int mask=f[18][col]>1.f?0:1;if(f[19][col]>.5f)mask=2;
    float precip[9];for(int c=0;c<9;c++)precip[c]=f[c<3?20+c:21+c][col];
    m.udm2d(t,q,qci,qrs,ncr,den,p,dz,
        s[0],s[1],s[2],s[3],s[5],s[6],s[7],s[8],s[9],s[10],
        s[11],s[12],s[13],s[14],s[15],s[16],s[17],s[18],
        1,&mask,&precip[0],&precip[1],&precip[2],
        1,2,1,2,1,nz+1,1,1,1,1,1,nz,1,1,1,1,1,nz,
        &precip[3],&precip[4],&precip[5],&precip[6],&precip[7],&precip[8]);
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        f[0][idx]=tc_div(t[k],f[12][idx]);f[1][idx]=q[k];
        f[2][idx]=qci[k];f[4][idx]=qci[k+nz];
        f[3][idx]=qrs[k];f[5][idx]=qrs[k+nz];
        f[6][idx]=qrs[k+2*nz];f[7][idx]=qrs[k+3*nz];
        f[8][idx]=ncr[k];f[9][idx]=ncr[k+nz];f[10][idx]=ncr[k+2*nz];
        t[k]=f[0][idx]*f[12][idx];
    }
    for(int c=0;c<9;c++)f[c<3?20+c:21+c][col]=precip[c];
    if(radar) {
        float dbz[UDM_COLUMN_KMAX];
        m.udm_mp_reflectivity(q,qrs,qrs+nz,qrs+2*nz,ncr+2*nz,qrs+3*nz,t,p,dbz,1,nz,1,1);
        for(int k=0;k<nz;k++)f[23][k*columns+col]=tc_max(-35.f,dbz[k]);
    }
}
