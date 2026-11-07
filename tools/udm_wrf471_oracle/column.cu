// Complete UDM2D call for each independent native horizontal column.
extern "C" __global__ void udm_columns(float** fields,const int* land,
    const float* scalars,int nz,int columns,const double** tables) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=columns)return;
    UdmColumn m;m.nz=nz;
    // BIND_TABLES
    float t[UDM_COLUMN_KMAX],q[UDM_COLUMN_KMAX];
    float qci[2*UDM_COLUMN_KMAX],qrs[4*UDM_COLUMN_KMAX],ncr[3*UDM_COLUMN_KMAX];
    float den[UDM_COLUMN_KMAX],p[UDM_COLUMN_KMAX],dz[UDM_COLUMN_KMAX];
    for(int k=0;k<nz;k++) {
        t[k]=fields[0][k*columns+col];q[k]=fields[1][k*columns+col];
        for(int c=0;c<2;c++)qci[k+c*nz]=fields[2][(c*nz+k)*columns+col];
        for(int c=0;c<4;c++)qrs[k+c*nz]=fields[3][(c*nz+k)*columns+col];
        for(int c=0;c<3;c++)ncr[k+c*nz]=fields[4][(c*nz+k)*columns+col];
        den[k]=fields[5][k*columns+col];p[k]=fields[6][k*columns+col];
        dz[k]=fields[7][k*columns+col];
    }
    float precip[9];for(int i=0;i<9;i++)precip[i]=fields[8+i][col];
    int mask=land[col];
    m.udm2d(t,q,qci,qrs,ncr,den,p,dz,
        scalars[0],scalars[1],scalars[2],scalars[3],scalars[4],scalars[5],
        scalars[6],scalars[7],scalars[8],scalars[9],scalars[10],scalars[11],
        scalars[12],scalars[13],scalars[14],scalars[15],scalars[16],scalars[17],
        1,&mask,&precip[0],&precip[1],&precip[2],
        1,2,1,2,1,nz+1,1,1,1,1,1,nz,1,1,1,1,1,nz,
        &precip[3],&precip[4],&precip[5],&precip[6],&precip[7],&precip[8]);
    for(int k=0;k<nz;k++) {
        fields[0][k*columns+col]=t[k];fields[1][k*columns+col]=q[k];
        for(int c=0;c<2;c++)fields[2][(c*nz+k)*columns+col]=qci[k+c*nz];
        for(int c=0;c<4;c++)fields[3][(c*nz+k)*columns+col]=qrs[k+c*nz];
        for(int c=0;c<3;c++)fields[4][(c*nz+k)*columns+col]=ncr[k+c*nz];
    }
    for(int i=0;i<9;i++)fields[8+i][col]=precip[i];
}
