// Written from papers, not bitwise to WRF. SNICAR-AD Appendix A
// (10.5194/gmd-14-7673-2021), Dang 2019, Whicker 2022, He 2017/2019.
// One GPU block per column; wavelengths distributed over 64 threads.
#define NR 32
#define NW 480
#define OP 1504
#define FR (OP+(5*NR+2)*NW*3)
#define MX (FR+NW*4)
#define AG (MX+32*NW*3)
__device__ float clip(float x,float a,float b){return fminf(b,fmaxf(a,x));}
struct Slab {double tau,gam,ws,gs,rf,tf;};
struct RT {double r,t,b,ru,rd,tu,td;};
__device__ Slab layer(double tau,double absorption,double g) {
    Slab s;double w=1.-absorption,scale=fmax(1e-15,1.-w*g*g),a=absorption/scale;
    s.tau=tau*scale;s.ws=1.-a;s.gs=g/(1.+g);s.gam=sqrt(3.*a*(1.-s.ws*s.gs));
    if(s.gam<1e-9){double z=1.5*(1.-s.gs)*s.tau;s.rf=z/(2.+z);s.tf=2./(2.+z);}
    else {double u=1.5*(1.-s.ws*s.gs)/s.gam,e=exp(-s.gam*s.tau);
        double d=(u+1.)*(u+1.)-(u-1.)*(u-1.)*e*e;
        s.rf=(u*u-1.)*(-expm1(-2.*s.gam*s.tau))/d;s.tf=4.*u*e/d;}
    return s;
}
__device__ void direct(const Slab &s,double mu,double &r,double &t) {
    double a=1.-s.ws,A=.75*s.ws*(1.+s.gs*a),E=A*mu;
    double G=.5*s.ws*(1.+3.*s.gs*a*mu*mu),B=exp(-s.tau/mu);
    double D=1.-s.gam*s.gam*mu*mu;
    if(fabs(D)>1e-7){
        r=((E+G)*s.rf+(E-G)*(s.tf*B-1.))/D;
        t=((E+G)*s.tf+((E-G)*s.rf-E-G)*B)/D+B;
    }else{
        // Exact removable-singularity limit by l'Hopital; no perturbed mu.
        double Ep=A,Gp=3.*s.ws*s.gs*a*mu,Bp=B*s.tau/(mu*mu),Dp=-2.*s.gam*s.gam*mu;
        r=((Ep+Gp)*s.rf+(Ep-Gp)*(s.tf*B-1.)+(E-G)*s.tf*Bp)/Dp;
        t=((Ep+Gp)*s.tf+((Ep-Gp)*s.rf-Ep-Gp)*B+((E-G)*s.rf-E-G)*Bp)/Dp+B;
    }
    // Bounds remove roundoff at pure scattering/opaque limits only.
    r=fmin(1.,fmax(0.,r));t=fmin(1.-r,fmax(0.,t));
}
__constant__ double mu8[8]={.0198550717512319,.101666761293187,.237233795041836,.408282678752175,.591717321247825,.762766204958164,.898333238706813,.980144928248768};
__constant__ double w8[8]={.0506142681451881,.111190517226687,.156853322938944,.181341891689181,.181341891689181,.156853322938944,.111190517226687,.0506142681451881};
__device__ RT slab(double tau,double ab,double g,double mu) {
    Slab s=layer(tau,ab,g);RT o;direct(s,mu,o.r,o.t);o.b=exp(-s.tau/mu);o.ru=0.;o.td=0.;
    for(int j=0;j<8;++j){double r,t;direct(s,mu8[j],r,t);o.ru+=2.*w8[j]*mu8[j]*r;o.td+=2.*w8[j]*mu8[j]*t;}
    o.rd=o.ru;o.tu=o.td;return o;
}
// Add two possibly asymmetric layers: A above B. A's beam is retained
// independently of its direct-to-diffuse transmission (Appendix A13–A16).
__device__ RT add(RT A,RT B) {
    double d=fmax(1e-15,1.-A.rd*B.ru);RT o;
    o.r=A.r+A.tu*((A.t-A.b)*B.ru+A.b*B.r)/d;
    o.t=A.b*B.t+B.td*((A.t-A.b)+A.b*B.r*A.rd)/d;
    o.ru=A.ru+A.tu*A.td*B.ru/d;o.rd=B.rd+B.td*B.tu*A.rd/d;
    o.td=A.td*B.td/d;o.tu=A.tu*B.tu/d;o.b=A.b*B.b;return o;
}
__device__ RT transparent(){RT a={0,1,1,0,0,1,1};return a;}
// Appendix A17--A20. Net downward flux at an interface for unit
// illumination above A, with the remaining column B underneath.
__device__ void netflux(RT A,RT B,double &beam,double &diffuse){
    double den=fmax(1e-15,1.-A.rd*B.ru);
    double scattered=A.t-A.b;
    double down=A.b+(scattered+A.b*B.r*A.rd)/den;
    double up=(A.b*B.r+scattered*B.ru)/den;
    beam=down-up;diffuse=A.td*(1.-B.ru)/den;
}
__device__ int bracket(const float *a,int n,float q,float &f) {
    int i=0;while(i<n-2&&q>a[i+1])++i;f=clip((q-a[i])/(a[i+1]-a[i]),0,1);return i;
}
__device__ float age(const float *tab,float r,float T,float grad,float density,float liquid,float dt) {
    const float *a=tab+AG;float ft,fg,fd;
    int it=bracket(a,8,T,ft),ig=bracket(a+8,7,grad,fg),id=bracket(a+15,8,density,fd);
    double tau=0.,k=0.;for(int i=0;i<2;++i)for(int j=0;j<2;++j)for(int l=0;l<2;++l){
        int ix=((it+i)*7+ig+j)*8+id+l;double w=(i?ft:1-ft)*(j?fg:1-fg)*(l?fd:1-fd);
        tau+=w*a[23+ix*2];k+=w*a[24+ix*2];}
    double r0=3e6/(917.*60.),x=fmax(1.,r/r0),delta=dt/(3600.*fmax(tau,1e-6));
    // Log form avoids overflow at cold, low-growth high-kappa cells.
    double lr=log(x),lk=k*lr;
    double logsum=lk+log1p(delta*exp(-lk));r=r0*exp(logsum/fmax(k,1e-6));
    // Brun 1989 wet metamorphism as used by Flanner 2007: dr^3/dt
    // =4.22e-13 f_liq^3 m3/s. Radii here are micrometres.
    r=cbrt((double)r*r*r+4.22e5*pow((double)liquid,3.)*dt);
    return clip(r,30,1500);
}
__device__ void optics(const float *tab,int row,float r,int iw,bool bubble,double &ex,double &ab,double &g) {
    const float *axis=tab+(bubble?NR:0);int i=0;while(i<NR-2&&r>axis[i+1])++i;
    double f=clip(logf(r/axis[i])/logf(axis[i+1]/axis[i]),0,1);
    const float *v=tab+OP+((row+i)*NW+iw)*3,*u=v+NW*3;
    ex=v[0]+f*(u[0]-v[0]);ab=v[1]+f*(u[1]-v[1]);g=v[2]+f*(u[2]-v[2]);
}
__device__ double enhance(const float *tab,int iw,int species,double c) {
    if(c<=0)return 1.;double z=log10(c/.01)/(species==0?6.:5.)*31.;
    int i=(int)fmin(30.,fmax(0.,floor(z)));double f=fmin(1.,fmax(0.,z-i));
    double a=tab[MX+(i*NW+iw)*3+species],b=tab[MX+((i+1)*NW+iw)*3+species];
    return exp(log(a)+f*log(b/a));
}
__device__ double fresnel(double mu,double n,double k) {
    // Whicker 2022 eqs 7–10: adjusted real index at an absorbing interface.
    double si=1.-mu*mu,nr=sqrt(.5*(n*n-k*k+si+sqrt(pow(n*n-k*k-si,2.)+4.*n*n*k*k)));
    double ct=sqrt(fmax(0.,1.-si/(nr*nr)));
    double rs=(mu-nr*ct)/(mu+nr*ct),rp=(nr*mu-ct)/(nr*mu+ct);return .5*(rs*rs+rp*rp);
}
// State is three radiative snow layers, seven words each:
// mass, effective radius, liquid mass, external/internal BC and dust.
// Impurity and liquid carriers are kg/m2.
extern "C" __global__ void snow_papers_albedo(const float *in,const int *ist,
 const float *layers,const float *previous,const float *forcing,const float *tab,
 const float *ice,float *out,float *next,float *heating,int n,int shape) {
    int c=blockIdx.x,tid=threadIdx.x;if(c>=n)return;const float *p=in+(size_t)c*56;
    const float *lp=layers+(size_t)c*12,*old=previous+(size_t)c*21,*src=forcing+(size_t)c*6;
    __shared__ float st[21],mass[3],dz[3],temp[3],liq[3];__shared__ float sums[64][36];
    if(tid==0){
        float total=0,ot=0;for(int j=0;j<3;++j){mass[j]=fmaxf(0,lp[4*j])+fmaxf(0,lp[4*j+1]);dz[j]=fmaxf(1e-6,lp[4*j+2]);temp[j]=lp[4*j+3];liq[j]=mass[j]>0?lp[4*j+1]/mass[j]:0;total+=mass[j];ot+=old[7*j];}
        if(total<=0&&p[34]>0){mass[2]=p[34];dz[2]=fmaxf(.001,p[30]);temp[2]=p[28];total=p[34];}
        float fresh=fmaxf(0,total-ot),top=0;
        // Lin et al. 2025 section 2b2: the published Noah-MP fresh-snow
        // ramp is 54.526--204.526 um between -30 and 0 C.
        float r0=54.526f+150.f*clip((src[5]-243.15f)/30.f,0,1);
        for(int j=0;j<3;++j){float bottom=top+mass[j],radius_mass=0;float q[4]={0,0,0,0};
            if(ot<=0){float r=p[42]<0?-p[42]:r0;radius_mass=mass[j]*r;for(int a=0;a<4;++a)q[a]=mass[j]*src[a];}
            else {float f=fmaxf(0,fminf(bottom,fresh)-top);radius_mass=f*r0;for(int a=0;a<4;++a)q[a]=f*src[a];
                float pos=fresh;for(int k=0;k<3;++k){float m=old[7*k],end=pos+m;float overlap=fmaxf(0,fminf(bottom,end)-fmaxf(top,pos));
                    if(m>0&&overlap>0){radius_mass+=overlap*fmaxf(30,old[7*k+1]);for(int a=0;a<4;++a)q[a]+=old[7*k+3+a]*overlap/m;}pos=end;}}
            float r=mass[j]>0?radius_mass/mass[j]:r0;
            // Resolved refreezing is applied after Noah-MP PHASECHANGE;
            // liquid drainage must not be mistaken for refreezing here.
            // Two-sided layer-center gradient, including the ground boundary.
            int other=j<2?j+1:j-1;float grad=fabsf(temp[j]-temp[other])/fmaxf(.001,.5f*(dz[j]+dz[other]));
            if(mass[other]<=0)grad=fabsf(temp[j]-p[28])/fmaxf(.001,.5f*dz[j]);
            if(mass[j]>0)r=p[30]<.025f?r0:age(tab,r,temp[j],grad,mass[j]/dz[j],liq[j],fmaxf(0,p[24]));
            st[7*j]=mass[j];st[7*j+1]=mass[j]>0?r:0;st[7*j+2]=fmaxf(0,lp[4*j+1]);for(int a=0;a<4;++a)st[7*j+3+a]=q[a];top=bottom;
        }
        for(int j=0;j<21;++j)next[(size_t)c*21+j]=st[j];
    }__syncthreads();
    // Transfer remains f64, including conservative/opaque/singular limits.
    // Broadband integration is f32, the Noah-MP output precision. Each
    // thread sums at most eight bins before the balanced block reduction.
    float acc[36]={0};bool glacier=src[4]>0;
    if(p[25]>0&&(p[34]>0||glacier))for(int iw=tid;iw<NW;iw+=64){
        double wd=tab[2*NR+NW+iw],wi=tab[2*NR+2*NW+iw];
        // G173 has finite spectral support. These bins contribute exactly
        // zero to every broadband output, for either illumination.
        if(wd==0.&&wi==0.)continue;
        int band=tab[2*NR+iw]<.7f?0:1;double mu=fmax(.001,(double)p[25]);
        double inc=fmax(.11-.4*p[37],0.);double ground=glacier?0.:(ist[c]==1?fmin(p[5+band]+inc,(double)p[7+band]):p[9+band]);
        // Only internally mixed dust has distinct clear/overcast optics.
        // With none present the same layer solution supplies both beams.
        // Retain the two solves whenever any layer contains internal dust.
        int skies=(st[6]>0||st[13]>0||st[20]>0)?2:1;
        for(int sky=0;sky<skies;++sky){
        RT ground_rt={ground,0,0,ground,ground,0,0};
        RT unit[8],below[9];int ids[8],count=0;
        for(int j=0;j<3;++j){if(mass[j]<=0)continue;double ex,ab,g;optics(tab,shape*NR,st[7*j+1],iw,false,ex,ab,g);
            double bcint=st[7*j+4]/mass[j]*1e9,duint=st[7*j+6]/mass[j]*1e6;
            ab*=enhance(tab,iw,0,bcint)*enhance(tab,iw,1+sky,duint);
            ab=fmin(ex,ab);double scat=ex-ab,moment=scat*g;
            for(int a=0;a<2;++a){const float *q=tab+OP+((5*NR+a)*NW+iw)*3;double frac=st[7*j+3+2*a]/mass[j];
                ex+=q[0]*frac;ab+=q[1]*frac;moment+=(q[0]-q[1])*q[2]*frac;scat+=(q[0]-q[1])*frac;}
            g=moment/fmax(1e-20,scat);double m=mass[j]/fmax(.01,(double)p[31]);unit[count]=slab(ex*m,ab/ex,g,mu);ids[count++]=j;
        }
        if(glacier){
            const float *fr=tab+FR+iw*4;
            double f=fresnel(mu,fr[0],fr[1]);RT face={f,1.-f,1.-f,fr[2],fr[3],1.-fr[3],1.-fr[2]};unit[count]=face;ids[count++]=-1;
            for(int j=0;j<4;++j){const float *ip=ice+((size_t)c*4+j)*3;
            double rho=ip[1],air=(917.-rho)/917.;
            double ex,ab,g;optics(tab,4*NR,ip[2],iw,true,ex,ab,g);
            double beta=4.*3.141592653589793*fr[1]/(tab[2*NR+iw]*1e-6);
            double e=ex*air+beta*rho/917.,a=beta*rho/917.;double si=1.-mu*mu,nn=fr[0]*fr[0]-fr[1]*fr[1];double nr2=.5*(nn+si+sqrt((nn-si)*(nn-si)+4.*fr[0]*fr[0]*fr[1]*fr[1]));double mut=sqrt(fmax(1e-10,1.-si/nr2));
            // Externally mixed impurities remain independent scatterers in ice.
            double moment=(e-a)*g;for(int sp=0;sp<2;++sp){const float *q=tab+OP+((5*NR+sp)*NW+iw)*3;
                double frac=src[2*sp]+src[2*sp+1];e+=q[0]*frac*rho;a+=q[1]*frac*rho;moment+=(q[0]-q[1])*q[2]*frac*rho;}
            g=moment/fmax(1e-20,e-a);unit[count]=slab(e*ip[0],a/e,g,mut);ids[count++]=3+j;
            }
        }
        below[count]=ground_rt;for(int j=count-1;j>=0;--j)below[j]=add(unit[j],below[j+1]);RT stack=below[0];
        if(sky==0){acc[band]+=wd*stack.r;acc[4+band]+=wd;}if(sky==1||skies==1){acc[2+band]+=wi*stack.ru;acc[6+band]+=wi;}
        RT above=transparent();double nd=1.-stack.r,ni=1.-stack.ru;
        for(int j=0;j<count;++j){above=add(above,unit[j]);double bd,bi;netflux(above,below[j+1],bd,bi);
            if(ids[j]>=0){int id=ids[j];if(sky==0)acc[8+id*4+band]+=wd*fmax(0.,nd-bd);if(sky==1||skies==1)acc[8+id*4+2+band]+=wi*fmax(0.,ni-bi);}nd=bd;ni=bi;}
        // The underlying boundary is the top soil, or the bottom ice layer.
        int id=glacier?6:3;if(sky==0)acc[8+id*4+band]+=wd*fmax(0.,nd);if(sky==1||skies==1)acc[8+id*4+2+band]+=wi*fmax(0.,ni);
        }
    }
    for(int k=0;k<36;++k)sums[tid][k]=acc[k];__syncthreads();
    for(int stride=32;stride;stride/=2){if(tid<stride)for(int k=0;k<36;++k)sums[tid][k]+=sums[tid+stride][k];__syncthreads();}
    if(tid==0){float *o=out+(size_t)c*6;for(int k=0;k<4;++k)o[k]=sums[0][4+k]>0?clip(sums[0][k]/sums[0][4+k],0,1):0;
        for(int j=0;j<7;++j)for(int k=0;k<4;++k)heating[((size_t)c*7+j)*4+k]=sums[0][4+k]>0?sums[0][8+j*4+k]/sums[0][4+k]:0;
        float m=0,a=0;for(int j=0;j<3;++j){m+=st[7*j];if(st[7*j]>0)a+=st[7*j]/st[7*j+1];}o[4]=a>0?-m/a:0;
        float flux=p[52]+p[53]+p[54]+p[55];o[5]=flux>0?(p[52]*o[0]+p[53]*o[1]+p[54]*o[2]+p[55]*o[3])/flux:p[41];}
}
// Physical solver diagnostic used by tests, on the same compiled device code.
extern "C" __global__ void snow_ad_probe(const double *x,double *y,int n){int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 RT a=slab(x[4*i],x[4*i+1],x[4*i+2],x[4*i+3]);RT h=slab(x[4*i]*.5,x[4*i+1],x[4*i+2],x[4*i+3]);RT b=add(h,h);
 double *o=y+10*i;o[0]=a.r;o[1]=a.t;o[2]=a.ru;o[3]=a.td;o[4]=b.r;o[5]=b.t;o[6]=b.ru;o[7]=b.td;o[8]=a.b;o[9]=b.b;}

// Packet assembly only. The spectral solver and its arithmetic are unchanged.
extern "C" __global__ void snow_forcing_pack(
 const float *in,const float *air,const float *carbon,const float *glacier,
 float *forcing,float *ice,int n,int bc_slot,int dust_slot,float bc,float dust,
 float rho,float radius) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    float *f=forcing+(size_t)c*6;
    for(int k=0;k<6;++k)f[k]=0.f;
    f[5]=air?air[c]:in[(size_t)c*56+28];
    f[bc_slot]=bc;f[dust_slot]=dust;
    if(carbon)f[0]+=carbon[c];
    if(glacier)f[4]=glacier[c];
    if(ice)for(int j=0;j<4;++j){
        float *p=ice+((size_t)c*4+j)*3;
        p[0]=j==0?.1f:j==1?.3f:j==2?.6f:1.f;p[1]=rho;p[2]=radius;
    }
}

extern "C" __global__ void snow_glacier_pack(
 const bool *active,const float *cosz,const float *tg,const float *snowh,
 const float *sneqvo,const float *snow,const float *qsnow,const float *albold,
 const float *tauss,const float *swdown,const float *snice,const float *snliq,
 const float *z,const float *tsnow,const float *state,const float *zsoil,
 float *packed,float *layers,float *previous,float *ice,int n,float dt,
 float rho,float radius) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    float *p=packed+(size_t)c*56;
    for(int k=0;k<56;++k)p[k]=0.f;
    p[24]=dt;p[25]=cosz[c];p[28]=tg[c];p[30]=snowh[c];p[31]=1.f;
    p[33]=sneqvo[c];p[34]=active[c]?snow[c]:0.f;p[35]=qsnow[c];
    p[41]=albold[c];p[42]=tauss[c];
    p[5]=p[7]=.8f;p[6]=p[8]=.55f;
    p[52]=p[53]=swdown[c]*.35f;p[54]=p[55]=swdown[c]*.15f;
    for(int j=0;j<3;++j){
        float *l=layers+((size_t)c*3+j)*4;
        l[0]=snice[(size_t)j*n+c];l[1]=snliq[(size_t)j*n+c];
        l[2]=j==0?-z[c]:z[(size_t)(j-1)*n+c]-z[(size_t)j*n+c];
        l[3]=tsnow[(size_t)j*n+c];
    }
    for(int j=0;j<21;++j)previous[(size_t)c*21+j]=state[(size_t)j*n+c];
    for(int j=0;j<4;++j){
        float *l=ice+((size_t)c*4+j)*3;
        l[0]=j==0?-zsoil[0]:zsoil[j-1]-zsoil[j];l[1]=rho;l[2]=radius;
    }
}
