// Native qftisf after qccgts: the two-quadrature-angle source-function
// sweeps. The same rounded operations run in the same order for every
// value; the angles share one downward and one upward pass, recurrences
// stay in registers, and fuq2/fj/fk/fy/fid and alfa are formed where they
// are used instead of being stored. ffu/ffd keep native order (0 + w1*f1
// then + w2*f2).
{
const float flgw_ug0=(tables+table_offsets[179])[0];
const float flgw_ug1=(tables+table_offsets[179])[1];
const float flgw_w0=(tables+table_offsets[180])[0];
const float flgw_w1=(tables+table_offsets[180])[1];
const float flgw_xu=FG_DV(1.0f,ugts1);
float flgw_fid0=0.0f,flgw_fid1=0.0f;
ffd[0]=FG_AD(FG_AD(0.0f,FG_MU(flgw_w0,flgw_fid0)),FG_MU(flgw_w1,flgw_fid1));
float flgw_alfa=FG_MU(6.2832f,bf[0]);
t0=0.0f;
for(i=1;i<=n;i+=1){
const float lam=lamdan[(i-1)];
const float bet=beta[(i-1)];
const float wi=w[(i-1)];
xgy=FG_SU(FG_MU(lam,lam),FG_MU(bet,bet));
if((fabsf(xgy)<1.e-4f)){
if((xgy>=0.f)){
xgy=1.e-4f;
}
if((xgy<0.f)){
xgy=(-1.e-4f);
}
}
x=FG_DV(FG_MU(FG_MU(2.0f,FG_SU(1.0f,wi)),wi),xgy);
y1=FG_DV(w1[(i-1)],3.0f);
y=FG_MU(2.0f,FG_SU(1.0f,FG_MU(wi,y1)));
z=FG_MU((-y1),bet);
fuq1[(i-1)]=FG_SU(FG_AD(FG_MU(x,FG_SU(y,z)),1.0f),wi);
const float q2v=FG_SU(FG_AD(FG_MU(x,FG_AD(y,z)),1.0f),wi);
const float gam=gamman[(i-1)];
const float a_=aa[(i-1)];
const float b_=bb[(i-1)];
y=FG_MU(gam,FG_AD(flgw_xu,lam));
z=FG_SU(flgw_xu,lam);
fg[(i-1)]=FG_MU(a_,z);
fh[(i-1)]=FG_MU(b_,y);
const float fjv=FG_MU(a_,y);
const float fkv=FG_MU(b_,z);
const float alfa_next=FG_MU(6.2832f,bf[i]);
const float ti=t[(i-1)];
const float fyv=expn[(i-1)];
#pragma unroll
for(j=0;j<2;++j){
const float ug=j?flgw_ug1:flgw_ug0;
const float fxv=gfk_exp(FG_DV((-FG_SU(ti,t0)),ug));
xx=FG_MU(lam,ug);
tempugbeta=FG_AD(FG_MU(ug,bet),1.0f);
tempxxp1=FG_AD(xx,1.0f);
tempxxm1=FG_SU(xx,1.0f);
if((tempugbeta==0.f)){
tempugbeta=1.e-4f;
}
if((tempxxp1==0.f)){
tempxxp1=1.e-4f;
}
if((tempxxm1==0.f)){
tempxxm1=1.e-4f;
}
const float fz1v=FG_DV(FG_SU(1.0f,FG_MU(fxv,fyv)),tempxxp1);
const float fz2v=FG_DV(FG_SU(fxv,fyv),tempxxm1);
fx[(i-1)+j*nv]=fxv;
fz1[(i-1)+j*nv]=fz1v;
fz2[(i-1)+j*nv]=fz2v;
float& fidj=j?flgw_fid1:flgw_fid0;
fidj=FG_AD(FG_AD(FG_AD(FG_MU(fidj,fxv),FG_MU(fjv,fz1v)),FG_MU(fkv,fz2v)),FG_MU(FG_MU(FG_DV(1.0f,tempugbeta),FG_SU(alfa_next,FG_MU(flgw_alfa,fxv))),q2v));
}
ffd[i]=FG_AD(FG_AD(0.0f,FG_MU(flgw_w0,flgw_fid0)),FG_MU(flgw_w1,flgw_fid1));
flgw_alfa=alfa_next;
t0=ti;
}
yy=0.0f;
yy=FG_AD(yy,FG_MU(flgw_w0,flgw_fid0));
yy=FG_AD(yy,FG_MU(flgw_w1,flgw_fid1));
xx=FG_AD(FG_MU(FG_MU(yy,FG_SU(1.0f,ee)),2.0f),FG_MU(FG_MU(6.2831854f,ee),bs));
float flgw_fiu0=xx,flgw_fiu1=xx;
ffu[(nv1-1)]=FG_AD(FG_AD(0.0f,FG_MU(flgw_w0,flgw_fiu0)),FG_MU(flgw_w1,flgw_fiu1));
float flgw_alfa_up=FG_MU(6.2832f,bf[(nv1-1)]);
for(i=(nv1-1);i>=1;i+=(-1)){
const float alfa_i=FG_MU(6.2832f,bf[(i-1)]);
const float bet=beta[(i-1)];
const float fgi=fg[(i-1)];
const float fhi=fh[(i-1)];
const float q1v=fuq1[(i-1)];
#pragma unroll
for(j=0;j<2;++j){
const float ug=j?flgw_ug1:flgw_ug0;
ugbeta=FG_MU(ug,bet);
if((ugbeta==1.f)){
ugbeta=1.0001f;
}
const float fxv=fx[(i-1)+j*nv];
float& fiuj=j?flgw_fiu1:flgw_fiu0;
fiuj=FG_AD(FG_AD(FG_AD(FG_MU(fiuj,fxv),FG_MU(fgi,fz2[(i-1)+j*nv])),FG_MU(fhi,fz1[(i-1)+j*nv])),FG_MU(FG_MU(FG_DV(1.0f,FG_SU(ugbeta,1.0f)),FG_SU(FG_MU(flgw_alfa_up,fxv),alfa_i)),q1v));
}
ffu[(i-1)]=FG_AD(FG_AD(0.0f,FG_MU(flgw_w0,flgw_fiu0)),FG_MU(flgw_w1,flgw_fiu1));
flgw_alfa_up=alfa_i;
}
}
return;
}
