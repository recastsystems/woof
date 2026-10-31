// Generated from public-domain NOAA-EMC/HRRR v4.1.21 (WRFV3.9) HAILCAST.
// Notice: tools/hail_wrf461/upstream/LICENSE.txt
// SHA256: 4c43129a05b9a4686be08682ddf2d6b664e7c22bcf588e70acc58208baa491b1
template<int NZ> struct HailColumnWrf39 {
__device__ void hailstone_driver(const HailArray<float>& tca,const HailArray<float>& h1d,float ht,const HailArray<float>& pa,const HailArray<float>& rho1d,const HailArray<float>& ra,const HailArray<float>& qi1d,const HailArray<float>& qc1d,const HailArray<float>& qr1d,const HailArray<float>& qs1d,const HailArray<float>& qg1d,const HailArray<float>& vuu,float wdur,int nz,float& dhail1,float& dhail2,float& dhail3,float& dhail4,float& dhail5) {
float zbas = 0;
float tbas = 0;
float wbasp = 0;
float rbas = 0;
float cwitot = 0;
int kbas = 0;
float tk_embryo = 0;
float zfzl = 0;
float tfzl = 0;
float wfzlp = 0;
float rfzl = 0;
float vufzl = 0;
float densafzl = 0;
int kfzl = 0;
int nofroze = 0;
int cloudon = 0;
float rtime = 0;
float tau = 0;
float tau_1 = 0;
float tau_2 = 0;
float deltau = 0;
float g = 0;
float r_d = 0;
double dd = 0;
double d = 0;
double d_ice = 0;
float vt = 0;
float v = 0;
float ts = 0;
float tsm1 = 0;
float tsm2 = 0;
float fw = 0;
float water = 0;
float crit = 0;
float dense = 0;
int itype = 0;
HailStorage<float,NZ> ria;
HailStorage<float,NZ> rwa;
HailStorage<float,NZ> vuu_pert;
HailStorage<float,NZ> rwa_adiabat;
HailStorage<float,NZ> rwa_new;
HailStorage<float,NZ> esva;
HailStorage<float,NZ> rsa;
float p = 0;
float rs = 0;
float ri = 0;
float rw = 0;
float xi = 0;
float xw = 0;
float pc = 0;
float tc = 0;
float vu = 0;
float vumax = 0;
float vucore = 0;
float densa = 0;
float z = 0;
float delrw = 0;
HailStorage<float,5> dhails;
float tlayer = 0;
float rlayer = 0;
float player = 0;
float tsum = 0;
float rsum = 0;
float psum = 0;
float ldepth = 0;
float gm = 0;
float gm1 = 0;
float gmw = 0;
float gmi = 0;
float dgm = 0;
float dgmw = 0;
float dgmi = 0;
float dgmv = 0;
float di = 0;
float anu = 0;
float re = 0;
float ae = 0;
float dum = 0;
float icefactor = 0;
float adiabatic_factor = 0;
float sec = 0;
float secdel = 0;
int i = 0;
int j = 0;
int k = 0;
int ifout = 0;
HailStorage<int,1> ind;
// WRF:593
secdel = 5.0f;
// WRF:594
g = 9.81f;
// WRF:595
r_d = 287.f;
// WRF:601
tau = 7200.f;
// WRF:605
for(i=1; i<=nz; i+=1) {
// WRF:606
vuu_pert(i) = (vuu(i)*1.f);
// WRF:607
}
// WRF:611
for(i=1; i<=5; i+=1) {
// WRF:612
dhails(i) = 0.f;
// WRF:613
}
// WRF:616
if((wdur>2000)) {
// WRF:617
rtime = 2000.f;
// WRF:618
} else {
// WRF:619
rtime = wdur;
// WRF:620
}
// WRF:627
kbas = nz;
// WRF:629
for(k=1; k<=nz; k+=1) {
// WRF:630
cwitot = (qi1d(k)+qc1d(k));
// WRF:633
ria(k) = (qi1d(k)+qs1d(k));
// WRF:635
rwa(k) = qc1d(k);
// WRF:640
if((((cwitot>=1.e-12f))&&((k<kbas)))) {
// WRF:641
kbas = k;
// WRF:642
}
// WRF:643
}
// WRF:650
zbas = h1d(kbas);
// WRF:651
tbas = tca(kbas);
// WRF:652
wbasp = pa(kbas);
// WRF:653
rbas = ra(kbas);
// WRF:660
for(k=1; k<=nz; k+=1) {
// WRF:661
rwa_new(k) = rwa(k);
// WRF:662
if((k<kbas)) {
// WRF:663
rwa_adiabat(k) = 0.f;
// WRF:664
continue;
// WRF:665
}
// WRF:668
esva(k) = (611.2f*lake_exp(lake_div((17.67f*((tca(k)-273.155f))),((tca(k)-29.655f)))));
// WRF:670
rsa(k) = lake_div((0.62197f*esva(k)),((pa(k)-esva(k))));
// WRF:673
if((tca(k)>242.155f)) {
// WRF:674
icefactor = 1.f;
// WRF:675
} else if((((tca(k)<=242.155f))&&((tca(k)>235.155f)))) {
// WRF:676
icefactor = ((1-lake_div(((242.155f-tca(k))),5.f)));
// WRF:677
} else {
// WRF:678
icefactor = 0.f;
// WRF:679
}
// WRF:681
if((rbas>rsa(k))) {
// WRF:682
rwa_adiabat(k) = (((rbas-rsa(k)))*icefactor);
// WRF:683
} else {
// WRF:684
rwa_adiabat(k) = rwa(k);
// WRF:685
}
// WRF:688
if((k==kbas)) {
// WRF:689
rwa_new(k) = rwa_adiabat(k);
// WRF:690
} else if((((k>=(kbas+1)))&&((rwa_adiabat(k)>=1.e-12f)))) {
// WRF:691
rwa_new(k) = ((rwa_adiabat(k)*((h1d(k)-h1d((k-1)))))-rwa_new((k-1)));
// WRF:692
if((rwa_new(k)<0)) { rwa_new(k) = 0.f; }
// WRF:693
}
// WRF:701
}
// WRF:703
for(k=(kbas+1); k<=nz; k+=1) {
// WRF:704
rwa_new(k) = lake_div(rwa_new(k),((h1d(k)-h1d((k-1)))));
// WRF:705
}
// WRF:716
for(i=1; i<=5; i+=1) {
// WRF:717
switch(i) {
// WRF:718
case 1:
// WRF:720
dd = 5.e-3f;
// WRF:721
tk_embryo = 265.155f;
// WRF:722
break; case 2:
// WRF:723
dd = 7.5e-3f;
// WRF:724
tk_embryo = 265.155f;
// WRF:725
break; case 3:
// WRF:726
dd = 5.e-3f;
// WRF:727
tk_embryo = 260.155f;
// WRF:728
break; case 4:
// WRF:729
dd = 7.5e-3f;
// WRF:730
tk_embryo = 260.155f;
// WRF:731
break; case 5:
// WRF:732
tk_embryo = 260.155f;
// WRF:733
dd = 1.e-2f;
// WRF:734
break; }
// WRF:737
rtime = 2000.f;
// WRF:738
if((wdur<rtime)) { rtime = wdur; }
sec=0; d=0;
// WRF:740
tfzl = tk_embryo;
// WRF:741
interpp(pa,wfzlp,tca,tk_embryo,ifout,nz);
if(ifout) { dhails(i)=0; continue; }
// WRF:742
interp(h1d,zfzl,wfzlp,ifout,pa,nz);
// WRF:743
interp(ra,rfzl,wfzlp,ifout,pa,nz);
// WRF:744
interp(vuu_pert,vufzl,wfzlp,ifout,pa,nz);
// WRF:745
interp(rho1d,densafzl,wfzlp,ifout,pa,nz);
// WRF:749
if((zfzl<(zbas-1000))) { goto hail_label_400; }
// WRF:753
sec = 0.f;
// WRF:760
p = wfzlp;
// WRF:761
rs = rfzl;
// WRF:762
tc = tfzl;
// WRF:763
vu = vufzl;
// WRF:764
z = (zfzl-ht);
// WRF:765
ldepth = z;
// WRF:766
densa = densafzl;
// WRF:769
nofroze = 1;
// WRF:770
ts = tc;
// WRF:771
tsm1 = ts;
// WRF:772
tsm2 = ts;
// WRF:773
d = dd;
// WRF:774
fw = 0.0f;
// WRF:775
dense = 500.f;
// WRF:776
itype = 1.f;
// WRF:777
cloudon = 1;
// WRF:780
while((sec<tau)) {
// WRF:781
sec = (sec+secdel);
// WRF:787
interp(vuu_pert,vumax,p,ifout,pa,nz);
// WRF:791
if((ifout==1)) { goto hail_label_100; }
// WRF:794
if(((sec>0.0f)&&(sec<rtime))) {
// WRF:795
vucore = ((vumax*(((0.5f*lake_sin(lake_div(((3.14159f*sec)),(rtime))))+0.5f)))*1.2f);
// WRF:796
vu = vucore;
// WRF:797
} else if((sec>=rtime)) {
// WRF:798
vu = 0.0f;
// WRF:799
cloudon = 0;
// WRF:800
}
// WRF:804
terminl(densa,dense,d,vt,tc);
// WRF:807
v = (vu-vt);
// WRF:810
p = (p-(((densa*g)*v)*secdel));
// WRF:811
z = (z+(v*secdel));
// WRF:814
interp(tca,tc,p,ifout,pa,nz);
// WRF:815
interp(ra,rs,p,ifout,pa,nz);
// WRF:818
densa = lake_div(p,(((r_d*((1.f+lake_div((0.609f*rs),((1.f+rs))))))*tc)));
// WRF:821
interp(ria,ri,p,ifout,pa,nz);
// WRF:822
interp(rwa_new,rw,p,ifout,pa,nz);
// WRF:823
xi = ((ri*densa)*cloudon);
// WRF:824
xw = ((rw*densa)*cloudon);
// WRF:825
if((((xw+xi))>0)) {
// WRF:826
pc = lake_div(xi,((xw+xi)));
// WRF:827
} else {
// WRF:828
pc = 1.f;
// WRF:829
}
// WRF:833
vaporclose(delrw,pc,ts,tc,itype);
// WRF:837
massagr(d,gm,gm1,gmw,gmi,dgm,dgmw,dgmi,dgmv,di,anu,re,ae,tc,ts,p,dense,densa,fw,vt,xw,xi,secdel,itype,delrw);
// WRF:842
heatbud(ts,tsm1,tsm2,fw,tc,vt,delrw,d,densa,gm1,gm,dgm,dgmw,dgmv,dgmi,gmw,gmi,di,anu,re,ae,secdel,itype,p);
// WRF:849
water = (fw*gm);
// WRF:851
crit = 2.0e-4f;
// WRF:852
if((water>crit)) {
// WRF:853
breakup(dense,d,gm,fw,crit);
// WRF:854
}
// WRF:858
if((z<=zbas)) { goto hail_label_200; }
// WRF:861
d_ice = lake_pow((lake_div((((6*gm)*((1.f-fw)))),((3.141592654f*dense)))),0.33333333f);
// WRF:864
if((((d_ice<1.e-8f))&&((tc>273.155f)))) { goto hail_label_300; }
// WRF:867
tsm1 = ts;
// WRF:868
tsm2 = tsm1;
// WRF:870
}
// WRF:872
hail_label_100:;
// WRF:873
hail_label_200:;
// WRF:874
hail_label_300:;
// WRF:879
if((p<pa(nz))) {
// WRF:880
d = 0.0f;
// WRF:882
} else if((lake_abs((fw-1.0f))<0.001f)) {
// WRF:883
d = 0.0f;
// WRF:884
} else if((z>0)) {
// WRF:889
tsum = 0.f;
// WRF:890
rsum = 0.f;
// WRF:891
psum = 0.f;
// WRF:892
for(k=1; k<=kbas; k+=1) {
// WRF:893
tsum = (tsum+tca(k));
// WRF:894
psum = (psum+pa(k));
// WRF:895
rsum = (rsum+ra(k));
// WRF:896
}
// WRF:897
tlayer = lake_div(tsum,kbas);
// WRF:898
player = lake_div(psum,kbas);
// WRF:899
rlayer = lake_div(rsum,kbas);
// WRF:904
d_ice = lake_pow((lake_div((((6*gm)*((1.f-fw)))),((3.141592654f*dense)))),0.33333333f);
// WRF:905
d = d_ice;
// WRF:906
melt(d,tlayer,player,rlayer,ldepth,vt);
// WRF:908
}
// WRF:911
hail_label_400:;
// WRF:914
if((sec<60)) { d = 0.f; }
// WRF:917
if((d>0.254f)) { d = 0.f; }
// WRF:920
dhails(i) = (d*1000);
// WRF:922
}
// WRF:925
dhail1 = dhails(1);
// WRF:926
dhail2 = dhails(2);
// WRF:927
dhail3 = dhails(3);
// WRF:928
dhail4 = dhails(4);
// WRF:929
dhail5 = dhails(5);
}
__device__ void interpp(const HailArray<float>& pa,float& pval,const HailArray<float>& ta,float& tval,int& ifout,int& itel) {
int i = 0;
float fract = 0;
// WRF:957
ifout = 1;
// WRF:959
for(i=1; i<=(itel-1); i+=1) {
// WRF:960
if(((((tval<ta(i))&&(tval>=ta((i+1)))))||(((tval>ta(i))&&(tval<=ta((i+1))))))) {
// WRF:963
fract = lake_div(((ta(i)-tval)),((ta(i)-ta((i+1)))));
// WRF:965
pval = (((((1.0f-fract))*pa(i)))+((fract*pa((i+1)))));
// WRF:968
ifout = 0;
// WRF:969
break;
// WRF:970
}
// WRF:971
}
}
__device__ void interp(const HailArray<float>& aa,float& a,float& p,int& ifout,const HailArray<float>& pa,int& itel) {
int i = 0;
float pdiff = 0;
float vdiff = 0;
float rdiff = 0;
float verh = 0;
float adiff = 0;
// WRF:1000
ifout = 1;
// WRF:1002
for(i=1; i<=(itel-1); i+=1) {
// WRF:1003
if(((p<=pa(i))&&(p>pa((i+1))))) {
// WRF:1005
pdiff = (pa(i)-pa((i+1)));
// WRF:1006
vdiff = (pa(i)-p);
// WRF:1007
verh = lake_div(vdiff,pdiff);
// WRF:1010
rdiff = (aa((i+1))-aa(i));
// WRF:1013
a = (aa(i)+(rdiff*verh));
// WRF:1016
ifout = 0;
// WRF:1017
break;
// WRF:1018
}
// WRF:1019
}
}
__device__ void terminl(float& densa,float& dense,double& d,float& vt,float& tc) {
float gmass = 0;
float gx = 0;
float re = 0;
float w = 0;
float y = 0;
const float pi = 3.141592654f;
const float g = 9.78956f;
float anu = 0;
// WRF:1045
gmass = lake_div((((dense*pi)*(lake_pow(d,3.f)))),6.f);
// WRF:1048
anu = (lake_div(((0.00001718f)*((273.155f+120.f))),((tc+120.f)))*lake_pow((lake_div(tc,273.155f)),(1.5f)));
// WRF:1051
gx = lake_div(((((8.0f*gmass)*g)*densa)),((pi*((anu*anu)))));
// WRF:1052
re = lake_pow((lake_div(gx,0.6f)),0.5f);
// WRF:1056
if((gx<550)) {
// WRF:1057
w = lake_log10(gx);
// WRF:1058
y = (((-1.7095f)+(1.33438f*w))-(0.11591f*(lake_pow(w,2.0f))));
// WRF:1059
re = lake_pow(10,y);
// WRF:1060
vt = lake_div((anu*re),((d*densa)));
// WRF:1061
} else if(((gx>=550)&&(gx<1800))) {
// WRF:1062
w = lake_log10(gx);
// WRF:1063
y = ((((-1.81391f)+(1.34671f*w))-(0.12427f*(lake_pow(w,2.0f))))+(0.0063f*(lake_pow(w,3.0f))));
// WRF:1064
re = lake_pow(10,y);
// WRF:1065
vt = lake_div((anu*re),((d*densa)));
// WRF:1066
} else if(((gx>=1800)&&(gx<3.45e08f))) {
// WRF:1067
re = (0.4487f*(lake_pow(gx,0.5536f)));
// WRF:1068
vt = lake_div((anu*re),((d*densa)));
// WRF:1069
} else {
// WRF:1070
re = lake_pow((lake_div(gx,0.6f)),0.5f);
// WRF:1071
vt = lake_div((anu*re),((d*densa)));
// WRF:1072
}
}
__device__ void vaporclose(float& delrw,float& pc,float& ts,float& tc,int& itype) {
float rv = 0;
float alv = 0;
float als = 0;
float ratio = 0;
float esat = 0;
float rhokor = 0;
float esatw = 0;
float rhoomgw = 0;
float esati = 0;
float rhoomgi = 0;
float rhoomg = 0;
// WRF:1099
rv = 461.48f; alv = 2500000.f; als = 2836050.f;
// WRF:1103
ratio = lake_div(1.f,273.155f);
// WRF:1104
if((itype==2)) {
// WRF:1105
esat = (611.f*lake_exp((lake_div(alv,rv)*((ratio-lake_div(1.f,ts))))));
// WRF:1106
} else {
// WRF:1107
esat = (611.f*lake_exp((lake_div(als,rv)*((ratio-lake_div(1.f,ts))))));
// WRF:1108
}
// WRF:1109
rhokor = lake_div(esat,((rv*ts)));
// WRF:1112
esatw = (611.f*lake_exp((lake_div(alv,rv)*((ratio-lake_div(1.f,tc))))));
// WRF:1113
rhoomgw = lake_div(esatw,((rv*tc)));
// WRF:1114
esati = (611.f*lake_exp((lake_div(als,rv)*((ratio-lake_div(1.f,tc))))));
// WRF:1115
rhoomgi = lake_div(esati,((rv*tc)));
// WRF:1117
rhoomg = rhoomgi;
// WRF:1121
delrw = ((rhokor-rhoomg));
}
__device__ void massagr(double& d,float& gm,float& gm1,float& gmw,float& gmi,float& dgm,float& dgmw,float& dgmi,float& dgmv,float& di,float& anu,float& re,float& ae,float& tc,float& ts,float& p,float& dense,float& densa,float& fw,float& vt,float& xw,float& xi,float& sekdel,int& itype,float& delrw) {
float pi = 0;
float d0 = 0;
float gmw2 = 0;
float gmi2 = 0;
float ew = 0;
float ei = 0;
float densel = 0;
float denseli = 0;
float denselw = 0;
float dc = 0;
float voll = 0;
float volt = 0;
float vol1 = 0;
float dgmw_nosoak = 0;
float soak = 0;
float soakm = 0;
float densac = 0;
float e = 0;
float afactor = 0;
float ns = 0;
float tscelsius = 0;
float vimp = 0;
float w = 0;
// WRF:1145
pi = 3.141592654f;
// WRF:1148
d0 = (0.226f*1.e-4f);
// WRF:1149
di = ((d0*lake_pow((lake_div(tc,273.155f)),1.81f))*(lake_div(100000.f,p)));
// WRF:1162
ew = 1.0f;
// WRF:1164
if((tc>=273.155f)) {
// WRF:1165
ei = 1.00f;
// WRF:1166
} else if((tc>=233.155f)) {
// WRF:1167
ei = (1.0f-(lake_div(((273.155f-ts)),40.f)));
// WRF:1168
} else {
// WRF:1169
ei = 0.0f;
// WRF:1170
}
// WRF:1175
densac = ((densa*(1.e3f))*(1.e-6f));
// WRF:1177
anu = ((1.717e-4f*(lake_div(393.0f,((tc+120.0f)))))*lake_pow((lake_div(tc,273.155f)),1.5f));
// WRF:1179
re = lake_div(((d*vt)*densac),anu);
// WRF:1180
e = (lake_pow((0.60f),(0.333333333f))*(lake_pow(re,0.50f)));
// WRF:1182
if((re<6000.0f)) {
// WRF:1183
ae = (0.78f+(0.308f*e));
// WRF:1184
} else if(((re>=6000.0f)&&(re<20000.0f))) {
// WRF:1185
ae = (0.76f*e);
// WRF:1186
} else if((re>=20000.0f)) {
// WRF:1187
ae = (((0.57f+(9.0e-6f*re)))*e);
// WRF:1188
}
// WRF:1193
gm = ((lake_div(pi,6.f)*(lake_pow(d,3.f)))*dense);
// WRF:1194
gmw = (fw*gm);
// WRF:1195
gmi = (gm-gmw);
// WRF:1198
gm1 = gm;
// WRF:1205
gmw2 = (gmw+(sekdel*(((((lake_div(pi,4.f)*lake_pow(d,2.f))*vt)*xw)*ew))));
// WRF:1206
dgmw = (gmw2-gmw);
// WRF:1207
gmw = gmw2;
// WRF:1210
gmi2 = (gmi+(sekdel*(((((lake_div(pi,4.f)*lake_pow(d,2.f))*vt)*xi)*ei))));
// WRF:1211
dgmi = (gmi2-gmi);
// WRF:1212
gmi = gmi2;
// WRF:1216
dgmv = ((((((sekdel*2)*pi)*d)*ae)*di)*delrw);
// WRF:1217
if((dgmv<0)) { dgmv = 0; }
// WRF:1220
dgm = ((dgmw+dgmi)+dgmv);
// WRF:1222
denselw = 900.f;
// WRF:1223
denseli = 700.f;
// WRF:1225
if((itype==1)) {
// WRF:1228
if((((dgmw>0))||((dgmv>0)))) {
// WRF:1230
dc = (lake_pow((lake_div((0.74f*xw),(((3.14159f*1000.f)*3.e8f)))),0.33333333f)*1.e6f);
// WRF:1232
ns = lake_div((((2*vt)*100.f)*lake_pow(((dc*1.e-4f)),2.f)),((((9*anu)*d)*50)));
// WRF:1234
if((ns<0.1f)) {
// WRF:1235
w = (-1.f);
// WRF:1236
} else {
// WRF:1237
w = lake_log10(ns);
// WRF:1238
}
// WRF:1239
if((re>200)) {
// WRF:1240
if((ns<0.1f)) {
// WRF:1241
vimp = 0.0f;
// WRF:1242
} else if((((ns>=0.1f))&&((ns<=10)))) {
// WRF:1243
vimp = ((((((0.356f+(0.4738f*w))-(0.1233f*lake_pow(w,2.f)))-(0.1618f*lake_pow(w,3.f)))+(0.0807f*lake_pow(w,4.f))))*vt);
// WRF:1245
} else if((ns>10)) {
// WRF:1246
vimp = (0.63f*vt);
// WRF:1247
}
// WRF:1248
} else if((((re>65))&&((re<=200)))) {
// WRF:1249
if((ns<0.1f)) {
// WRF:1250
vimp = 0.0f;
// WRF:1251
} else if((((ns>=0.1f))&&((ns<=10)))) {
// WRF:1252
vimp = ((((((0.3272f+(0.4907f*w))-(0.09452f*lake_pow(w,2.f)))-(0.1906f*lake_pow(w,3.f)))+(0.07105f*lake_pow(w,4.f))))*vt);
// WRF:1254
} else if((ns>10)) {
// WRF:1255
vimp = (0.61f*vt);
// WRF:1256
}
// WRF:1257
} else if((((re>20))&&((re<=65)))) {
// WRF:1258
if((ns<0.1f)) {
// WRF:1259
vimp = 0.0f;
// WRF:1260
} else if((((ns>=0.1f))&&((ns<=10)))) {
// WRF:1261
vimp = ((((((0.2927f+(0.5085f*w))-(0.03453f*lake_pow(w,2.f)))-(0.2184f*lake_pow(w,3.f)))+(0.03595f*lake_pow(w,4.f))))*vt);
// WRF:1263
} else if((ns>10)) {
// WRF:1264
vimp = (0.59f*vt);
// WRF:1265
}
// WRF:1266
} else if((re<=20)) {
// WRF:1267
if((ns<0.4f)) {
// WRF:1268
vimp = 0.0f;
// WRF:1269
} else if((((ns>=0.4f))&&((ns<=10)))) {
// WRF:1270
vimp = ((((((0.1701f+(0.7246f*w))+(0.2257f*lake_pow(w,2.f)))-(1.13f*lake_pow(w,3.f)))+(0.5756f*lake_pow(w,4.f))))*vt);
// WRF:1272
} else if((ns>10)) {
// WRF:1273
vimp = (0.57f*vt);
// WRF:1274
}
// WRF:1275
}
// Physical domain of the empirical impact-speed fit.
vimp = lake_max(vimp,0.f);
// WRF:1278
tscelsius = (ts-273.16f);
// WRF:1279
afactor = lake_div(((-dc)*vimp),tscelsius);
// WRF:1280
if((((tscelsius<=(-5.f)))||((afactor>=(-1.60f))))) {
// WRF:1281
denselw = (0.30f*lake_pow((afactor),0.44f));
// WRF:1282
} else if((tscelsius>(-5.f))) {
// WRF:1283
denselw = lake_exp(((((-0.03115f)-(1.7030f*afactor))+(0.9116f*lake_pow(afactor,2.f)))-(0.1224f*lake_pow(afactor,3.f))));
// WRF:1285
}
// WRF:1287
denselw = (denselw*1000.f);
// WRF:1289
if((denselw<100)) { denselw = 100; }
// WRF:1290
if((denselw>900)) { denselw = 900; }
// WRF:1293
}
// WRF:1294
if((dgmi>0)) {
// WRF:1296
denseli = 700.f;
// WRF:1297
}
// WRF:1300
dgmw_nosoak = dgmw;
// WRF:1303
} else {
// WRF:1307
vol1 = lake_div(gm,dense);
// WRF:1310
soak = ((900*vol1)-gm);
// WRF:1312
soakm = dgmw;
// WRF:1314
if((soakm>soak)) { soakm = soak; }
// WRF:1315
gm = (gm+soakm);
// WRF:1317
dense = lake_div(gm,vol1);
// WRF:1320
dgmw_nosoak = (dgmw-soakm);
// WRF:1323
denselw = 900.f;
// WRF:1324
denseli = 900.f;
// WRF:1326
}
// WRF:1332
if((dgmi<=0)) {
// WRF:1333
voll = lake_div(((dgmw_nosoak+dgmv)),denselw);
// WRF:1334
} else if((dgmw<=0)) {
// WRF:1335
voll = lake_div((dgmi),denseli);
// WRF:1336
} else {
// WRF:1337
voll = (lake_div((dgmi),denseli)+lake_div(((dgmw_nosoak+dgmv)),denselw));
// WRF:1338
}
// WRF:1341
volt = (voll+lake_div(gm,dense));
// WRF:1343
dense = lake_div(((((gm+dgmi)+dgmv)+dgmw_nosoak)),volt);
// WRF:1345
gm = (((gm+dgmi)+dgmw_nosoak)+dgmv);
// WRF:1346
d = lake_pow((lake_div(((6*gm)),((pi*dense)))),0.33333333f);
}
__device__ void heatbud(float& ts,float& tsm1,float& tsm2,float& fw,float& tc,float& vt,float& delrw,double& d,float& densa,float& gm1,float& gm,float& dgm,float& dgmw,float& dgmv,float& dgmi,float& gmw,float& gmi,float& di,float& anu,float& re,float& ae,float& sekdel,int& itype,float& p) {
float rv = 0;
float rd = 0;
float g = 0;
float pi = 0;
float alf = 0;
float alv = 0;
float als = 0;
float ci = 0;
float cw = 0;
float ak = 0;
float h = 0;
float ah = 0;
float tcc = 0;
float tsc = 0;
float delrwc = 0;
float densac = 0;
float tdiff = 0;
float dmlt = 0;
float tscm1 = 0;
float tscm2 = 0;
// WRF:1380
rv = 461.48f; rd = 287.04f; g = 9.78956f;
// WRF:1381
pi = 3.141592654f; alf = 79.7f; alv = 597.3f;
// WRF:1382
als = 677.0f; ci = 0.5f; cw = 1.f;
// WRF:1385
tsc = (ts-273.155f);
// WRF:1386
tscm1 = (tsm1-273.155f);
// WRF:1387
tscm2 = (tsm2-273.155f);
// WRF:1388
tcc = (tc-273.155f);
// WRF:1389
delrwc = ((delrw*(1.e3f))*(1.e-6f));
// WRF:1390
densac = ((densa*(1.e3f))*(1.e-6f));
// WRF:1395
ak = (((5.8f+(0.0184f*tcc)))*1.e-5f);
// WRF:1402
h = (lake_pow((0.71f),(0.333333333f))*(lake_pow(re,0.50f)));
// WRF:1406
if((re<6000.0f)) {
// WRF:1407
ah = (0.78f+(0.308f*h));
// WRF:1409
} else if(((re>=6000.0f)&&(re<20000.0f))) {
// WRF:1410
ah = (0.76f*h);
// WRF:1412
} else if((re>=20000.0f)) {
// WRF:1413
ah = (((0.57f+(9.0e-6f*re)))*h);
// WRF:1415
}
// WRF:1421
if((itype==1)) {
// WRF:1427
tsc = (((0.6f*(((tsc-lake_div((tsc*dgm),gm1))+(lake_div(sekdel,((gm1*ci)))*((((((2.f*pi)*d)*((((ah*ak)*((tcc-tsc)))-(((ae*als)*di)*delrwc))))+(lake_div(dgmw,sekdel)*((alf+(cw*tcc)))))+((lake_div(dgmi,sekdel)*ci)*tcc)))))))+(0.2f*tscm1))+(0.2f*tscm2));
// WRF:1432
ts = (tsc+273.155f);
// WRF:1433
if((ts>=273.155f)) {
// WRF:1434
ts = 273.155f;
// WRF:1435
}
// WRF:1436
tdiff = lake_abs((ts-273.155f));
// WRF:1437
if((tdiff<=1.e-6f)) { itype = 2; }
// WRF:1439
} else if((itype==2)) {
// WRF:1442
if((tcc<0.f)) {
// WRF:1444
fw = ((fw-lake_div((fw*dgm),gm1))+(lake_div(sekdel,((gm1*alf)))*((((((2.f*pi)*d)*((((ah*ak)*tcc)-(((ae*alv)*di)*delrwc))))+(lake_div(dgmw,sekdel)*((alf+(cw*tcc)))))+((lake_div(dgmi,sekdel)*ci)*tcc)))));
// WRF:1447
} else {
// WRF:1449
dmlt = lake_div(((((((((2.f*pi)*d)*ah)*ak)*tcc)+((((((2.f*pi)*d)*ae)*alv)*di)*delrwc))+((lake_div(dgmw,sekdel)*cw)*tcc))),alf);
// WRF:1451
fw = lake_div((((fw*gm)+dmlt)),gm);
// WRF:1452
}
// WRF:1454
if((fw>1.f)) { fw = 1.f; }
// WRF:1455
if((fw<0.f)) { fw = 0.f; }
// WRF:1458
if((fw<=1.e-6f)) {
// WRF:1459
itype = 1;
// WRF:1460
}
// WRF:1462
}
}
__device__ void breakup(float& dense,double& d,float& gm,float& fw,float& crit) {
float water = 0;
float gmi = 0;
float wat = 0;
float pi = 0;
// WRF:1479
pi = 3.141592654f;
// WRF:1481
water = (fw*gm);
// WRF:1492
wat = (water-crit);
// WRF:1493
gm = (gm-wat);
// WRF:1494
fw = lake_div((crit),gm);
// WRF:1496
if((fw>1.0f)) { fw = 1.0f; }
// WRF:1497
if((fw<0.0f)) { fw = 0.0f; }
// WRF:1501
d = lake_pow((lake_div((6.f*gm),((pi*dense)))),(0.333333333f));
}
__device__ void melt(double& d,float& tlayer,float& player,float& rlayer,float& ldepth,float& vt) {
float eenv = 0;
float delta = 0;
float ewet = 0;
float de = 0;
float der = 0;
float wetold = 0;
float wetbulb = 0;
float wetbulbk = 0;
float tdclayer = 0;
float tclayer = 0;
float eps = 0;
float b = 0;
float hplayer = 0;
double a = 0;
float sd = 0;
float lt = 0;
float ka = 0;
float lf = 0;
float lv = 0;
float t0 = 0;
float dv = 0;
float pi = 0;
float rv = 0;
float rhoice = 0;
float tres = 0;
float re = 0;
float delt = 0;
float esenv = 0;
float rhosenv = 0;
float essfc = 0;
float rhosfc = 0;
float dsig = 0;
float dmdt = 0;
float mass = 0;
float massorg = 0;
float newmass = 0;
float gamma = 0;
float r = 0;
float rho = 0;
int wcnt = 0;
// WRF:1532
eps = 0.622f;
// WRF:1533
tclayer = (tlayer-273.155f);
// WRF:1534
a = 2.53e11f;
// WRF:1535
b = 5.42e3f;
// WRF:1536
tdclayer = lake_div(b,lake_log(lake_div((a*eps),((rlayer*player)))));
// WRF:1537
hplayer = lake_div(player,100.f);
// WRF:1540
eenv = lake_div(((player*rlayer)),((rlayer+eps)));
// WRF:1541
eenv = lake_div(eenv,100.f);
// WRF:1544
gamma = (6.6e-4f*player);
// WRF:1545
delta = lake_div(((4098.0f*eenv)),((((tdclayer+237.7f))*((tdclayer+237.7f)))));
// WRF:1546
wetbulb = lake_div(((((gamma*tclayer))+((delta*tdclayer)))),((gamma+delta)));
// WRF:1549
wcnt = 0;
// WRF:1550
while((wcnt<11)) {
// WRF:1551
ewet = (6.108f*(lake_exp(lake_div(((17.27f*wetbulb)),((237.3f+wetbulb))))));
// WRF:1552
de = ((((0.0006355f*hplayer)*((tclayer-wetbulb))))-((ewet-eenv)));
// WRF:1553
der = (((ewet*((.0091379024f-(lake_div(6106.396f,lake_pow(((273.155f+wetbulb)),2)))))))-((0.0006355f*hplayer)));
// WRF:1555
wetold = wetbulb;
// WRF:1556
wetbulb = (wetbulb-lake_div(de,der));
// WRF:1557
wcnt = (wcnt+1);
// WRF:1558
if(((lake_div(lake_abs((wetbulb-wetold)),wetbulb)>0.0001f))) {
// WRF:1559
break;
// WRF:1560
}
// WRF:1561
}
// WRF:1563
wetbulbk = (wetbulb+273.155f);
// WRF:1564
ka = .02f;
// WRF:1565
lf = 3.34e5f;
// WRF:1566
lv = 2.5e6f;
// WRF:1567
t0 = 273.155f;
// WRF:1568
dv = 0.25e-4f;
// WRF:1569
pi = 3.1415927f;
// WRF:1570
rv = (1004.f-287.f);
// WRF:1571
rhoice = 917.0f;
// WRF:1572
r = lake_div(d,2.f);
// WRF:1575
tres = lake_div(ldepth,vt);
// WRF:1580
rho = lake_div(85000.f,((287.f*tlayer)));
// WRF:1581
re = lake_div((((rho*r)*vt)*.01f),1.7e-5f);
// WRF:1584
delt = wetbulb;
// WRF:1589
esenv = (610.8f*(lake_exp(lake_div(((17.27f*wetbulb)),((237.3f+wetbulb))))));
// WRF:1591
rhosenv = lake_div(esenv,((rv*wetbulbk)));
// WRF:1592
essfc = (610.8f*(lake_exp(lake_div(((17.27f*((t0-273.155f)))),((237.3f+((t0-273.155f))))))));
// WRF:1594
rhosfc = lake_div(essfc,((rv*t0)));
// WRF:1595
dsig = (rhosenv-rhosfc);
// WRF:1598
dmdt = ((lake_div(((((-1.7f)*pi)*r)*(lake_pow(re,0.5f))),lf))*((((ka*delt))+(((((lv-lf))*dv)*dsig)))));
// WRF:1599
if((dmdt>0.f)) { dmdt = 0; }
// WRF:1600
mass = (dmdt*tres);
// WRF:1603
massorg = (((((1.33333333f*pi)*r)*r)*r)*rhoice);
// WRF:1604
newmass = (massorg+mass);
// WRF:1605
if((newmass<0.0f)) { newmass = 0.0f; }
// WRF:1606
d = (2.f*lake_pow((lake_div((0.75f*newmass),((pi*rhoice)))),0.333333333f));
}
};
