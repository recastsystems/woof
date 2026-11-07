/* Observation FDDA, translated only from public-domain WRF v4.7.1.
 * Source hashes and the complete notice: tools/fddaobs_wrf471_oracle/.
 * Array ABI deliberately retains native Fortran (i,k,j) ordering and bounds.
 * Qualification scope is recorded beside the pinned native oracle.
 */
struct Frame { int first,last,enabled,ktau,update; float xtime,tfaci,dtmin; };
struct Grid {
 int ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte;
 __device__ int xy(int i,int j) const {return i-ims+(ime-ims+1)*(j-jms);}
 __device__ int at(int i,int k,int j) const {return i-ims+(ime-ims+1)*(k-kms+(kme-kms+1)*(j-jms));}
};
__device__ int bound(int a,int lo,int hi){return max(lo,min(a,hi));}
__device__ int nintf(float x){return x>=0.f?int(x+0.5f):int(x-0.5f);}
__device__ float bilinear(const float*a,Grid g,int i,int k,int j,float dx,float dy){
 return (1.f-dy)*((1.f-dx)*a[g.at(i,k,j)]+dx*a[g.at(i+1,k,j)])
             +dy*((1.f-dx)*a[g.at(i,k,j+1)]+dx*a[g.at(i+1,k,j+1)]);
}
__device__ float linear2(const float*a,Grid g,int i,int j,float dx,float dy){
 return (1.f-dy)*((1.f-dx)*a[g.xy(i,j)]+dx*a[g.xy(i+1,j)])
             +dy*((1.f-dx)*a[g.xy(i,j+1)]+dx*a[g.xy(i+1,j+1)]);
}
__device__ float trilinear(const float*a,Grid g,int i,int k,int j,int kp,float dx,float dy,float dz){
 return (1.f-dz)*bilinear(a,g,i,k,j,dx,dy)+dz*bilinear(a,g,i,kp,j,dx,dy);
}
extern "C" __global__ void fddaobs_ratios(const float*ur,const float*vr,float*up,float*vp,Grid g){
 int i=int(blockIdx.x*blockDim.x+threadIdx.x)+g.ims;
 int j=int(blockIdx.y*blockDim.y+threadIdx.y)+g.jms;
 if(i>g.ime||j>g.jme)return;
 int is=max(g.its-1,g.ids),ie=min(g.ite+1,g.ide-1);
 int js=max(g.jts-1,g.jds),je=min(g.jte+1,g.jde-1);
 if(j>=js&&j<=je){
  if(i>=max(2,is)&&i<=ie)up[g.xy(i,j)]=.5f*(ur[g.xy(i,j)]+ur[g.xy(i-1,j)]);
  if(is==g.ids&&i==is)up[g.xy(i,j)]=ur[g.xy(i,j)];
  if(ie==g.ide-1&&i==ie+1)up[g.xy(i,j)]=ur[g.xy(ie,j)];
 }
 if(i>=is&&i<=ie){
  if(j>=max(2,js)&&j<=je)vp[g.xy(i,j)]=.5f*(vr[g.xy(i,j)]+vr[g.xy(i,j-1)]);
  if(js==g.jds&&j==js)vp[g.xy(i,j)]=vr[g.xy(i,j)];
  if(je==g.jde-1&&j==je+1)vp[g.xy(i,j)]=vr[g.xy(i,je)];
 }
}
/* One thread per report: no atomic reductions or observation-order changes. */
extern "C" __global__ void fddaobs_errob(
 const float*ub,const float*vb,const float*tb,const float*qv,const float*pb,const float*pp,
 const float*z,const float*ur,const float*vr,const float*tr,const int*kpbl,
 const float*rio,const float*rjo,float*rko,float*obs,float*err,
 int count,int nvar,int nerr,int wind,int temp,int mois,float t0,float rcp,Grid g,const Frame*frame){
 int n=int(blockIdx.x*blockDim.x+threadIdx.x);if(n>=count)return;
 if(frame&&(!frame->enabled||!frame->update||n<frame->first||n>=frame->last))return;
 float*e=err+n*nerr;float*o=obs+n*nvar;for(int v=0;v<nerr;v++)e[v]=0.f;
 int kend=min(g.kte,g.kde-1);
 float mx=rio[n]-.5f,my=rjo[n]-.5f;int mi=int(mx),mj=int(my);
 float mdx=mx-float(mi),mdy=my-float(mj);mi=bound(mi,1,g.ide-1);mj=bound(mj,1,g.jde-1);
 for(int typ=1;typ<=3;typ++){
  float gx=typ==1?0.f:.5f,gy=typ==2?0.f:.5f;
  float ra=rio[n]-gx,rb=rjo[n]-gy;int ia=int(ra),ib=int(rb);
  if(fabsf(rko[n]+99.f)<1.f){
   float pob=o[4];
   if(pob==-888888.f&&o[5]>-800000.f){
    int klo=g.kts;
    for(int k=g.kts+1;k<=kend;k++){klo=k-1;if(o[5]<=bilinear(z,g,mi,k,mj,mdx,mdy))break;}
    float zlo=bilinear(z,g,mi,klo,mj,mdx,mdy),zhi=bilinear(z,g,mi,klo+1,mj,mdx,mdy);
    float lp0=logf(.001f*bilinear(pb,g,mi,klo+1,mj,mdx,mdy)+.001f*bilinear(pp,g,mi,klo+1,mj,mdx,mdy));
    float lp1=logf(.001f*bilinear(pb,g,mi,klo,mj,mdx,mdy)+.001f*bilinear(pp,g,mi,klo,mj,mdx,mdy));
    float lp=o[5]<=zlo?lp1:o[5]>=zhi?lp0:lp0+(lp1-lp0)*((zhi-o[5])/(zhi-zlo));pob=expf(lp);
   }
   if(pob>-800000.f){
    int kb=1;for(int k=kend-1;k>=1;k--){kb=k;if(pob<=.001f*bilinear(pb,g,mi,k,mj,mdx,mdy)+.001f*bilinear(pp,g,mi,k,mj,mdx,mdy))break;}
    float hi=.001f*bilinear(pb,g,mi,kb+1,mj,mdx,mdy),hp=.001f*bilinear(pp,g,mi,kb+1,mj,mdx,mdy);
    rko[n]=max(float(kb+1)-(pob-hi-hp)/(.001f*bilinear(pb,g,mi,kb,mj,mdx,mdy)+.001f*bilinear(pp,g,mi,kb,mj,mdx,mdy)-hi-hp),1.f);
   }
   if(o[4]==-888888.f&&o[5]>-800000.f)o[4]=pob;
  }
  int ic=int(rko[n]);float dx=ra-float(ia),dy=rb-float(ib),dz=rko[n]-float(ic);
  float dummy=0.f;
  if(ra-1.f<0.f)dummy+=99999.f;if(rb-1.f<0.f)dummy+=99999.f;
  if(float(g.ide)-2.f*gx-ra-1.e-10f<0.f)dummy+=99999.f;
  if(float(g.jde)-2.f*gy-rb-1.e-10f<0.f)dummy+=99999.f;
  if(rko[n]<1.f)dummy+=99999.f;
  int lo=typ==1?0:typ==2?1:2,hi=typ<3?lo:4;for(int v=lo;v<=hi;v++)e[v]+=dummy;
  // Native permits out-of-tile observations; the owning tile performs interpolation.
  int i=bound(ia,1,g.ide-1),j=bound(ib,1,g.jde-1);
  if(!(i>=g.its&&i<=g.ite&&j>=g.jts&&j<=g.jte))continue;
  if(typ==1)e[8]=rko[n];
  int k=min(kend,ic),kp=min(k+1,kend);
  if(typ<=2&&wind==1){
   i=bound(ia,2,g.ide-1);j=bound(ib,2,g.jde-1);
   int im=max(1,i-1),ip=min(g.ide-1,i+1),jm=max(1,j-1),jp=min(g.jde-1,j+1);
   const float*a=typ==1?ub:vb;const float*r=typ==1?ur:vr;
   float ratio=1.f;if(rko[n]==1.f){ratio=(1.f-dy)*((1.f-dx)*r[g.xy(i,j)]+dx*r[g.xy(ip,j)])+dy*((1.f-dx)*r[g.xy(i,jp)]+dx*r[g.xy(ip,jp)]);}
   if(fabsf(ratio)<1.e-3f)ratio=1.f;
   e[typ-1]=e[typ-1]+ratio*o[typ-1]-trilinear(a,g,i,k,j,kp,dx,dy,dz);
   if(typ==1){float p1=.5f*(pb[g.at(im,1,j)]+pb[g.at(i,1,j)]),p2=.5f*(pb[g.at(i,1,j)]+pb[g.at(ip,1,j)]);e[6]=.001f*((1.f-dx)*p1+dx*p2);}
   else{float p1=.5f*(pb[g.at(i,1,jm)]+pb[g.at(i,1,j)]),p2=.5f*(pb[g.at(i,1,j)]+pb[g.at(i,1,jp)]);e[7]=.001f*((1.f-dy)*p1+dy*p2);}
  }
  if(typ==3){
   if(temp==1||mois==1){
    float ratio=rko[n]==1.f?linear2(tr,g,i,j,dx,dy):1.f;if(fabsf(ratio)<1.e-3f)ratio=1.f;
    float theta=-888888.f;if(o[2]>-800000.f&&o[4]>-800000.f)theta=o[2]*powf(100.f/o[4],rcp)-t0;
    e[2]=e[2]+ratio*theta-trilinear(tb,g,i,k,j,kp,dx,dy,dz);
    float modelq=trilinear(qv,g,i,k,j,kp,dx,dy,dz);
    e[3]=e[3]+o[3]-modelq;e[9]=e[9]+modelq;e[5]=.001f*bilinear(pb,g,i,1,j,dx,dy);
   }
   e[4]=float(kpbl[g.xy(i+nintf(dx),j+nintf(dy))]);
  }
 }
 if(fabsf(e[2])>20.f&&e[2]>-800000.f){for(int v=0;v<4;v++){e[v]=-888888.f;o[v]=-888888.f;}}
}
__device__ float timewt(float xtime,float dtmin,float window,float scale,float obtime){
 float fdtim=xtime-dtmin,tw1=window/2.f*60.f*scale,tw2=window*60.f*scale;
 float tc=1.f/tw1,ttim=obtime*60.f,dift=fabsf(fdtim-ttim),w=0.f;
 if(dift<=tw1)w=1.f;
 if(dift>tw1&&dift<=tw2){if(fdtim<ttim)w=(fdtim-(ttim-tw2))*tc;if(fdtim>ttim)w=((ttim+tw2)-fdtim)*tc;}
 return w;
}
struct Controls {
 int inest,ktau,ktaur,restart,interval,nudge_pbl,horiz,vert,negative_qv;
 float xtime,dtmin,rinxy,dx,window,sfcfact,sfcfacr,rinfmn,rinfmx,pfree,dcon,tfaci,coefficient,maxsnd_gap;
};
/* Parallel columns retain the native observation and vertical accumulation
 * order. With negative-qv scaling the surface innovation recurs across i;
 * that native dependency uses the row schedule (blockDim.y == 1). */
extern "C" __global__ void fddaobs_nudob_rows(
 float*aten,float*savwt,float*wt,float*w2,
 const float*mu,const float*c1h,const float*c2h,const float*msfx,const float*msfy,
 const float*rio,const float*rjo,const float*rko,const float*timeob,const float*obs,const float*err,
 const float*nlevs,const float*lev,const float*pb,const float*pp,const int*kpbl,
 const float*vih1,const float*vih2,const float*terrain,const float*z,const float*qv,
 int count,int nvar,int nerr,int ivar,Controls c,Grid g,const Frame*frame,const int*surface_top){
 bool cells=blockDim.z>1;
 bool columns=blockDim.y>1||cells;
 int kcell=int(blockIdx.z*blockDim.z+threadIdx.z)+g.kts;
 if(cells&&kcell>g.kte)return;
 int kbeg=cells?kcell:1,kend=cells?kcell:g.kte;
 float local_wt=0.f,local_w2=0.f;
 int j=int(columns?blockIdx.y*blockDim.y+threadIdx.y:blockIdx.x*blockDim.x+threadIdx.x)+g.jts;
 int column=int(blockIdx.x*blockDim.x+threadIdx.x)+g.its;
 if(j>g.jte||(columns&&column>g.ite))return;
 int ibeg=columns?column:g.its,iend=columns?column:g.ite;
 if(frame){if(!frame->enabled||frame->first==frame->last)return; c.ktau=frame->ktau;c.xtime=frame->xtime;c.tfaci=frame->tfaci;c.dtmin=frame->dtmin;}
 if(c.inest==1 && ((ivar<3&&(j<=2||j>=g.jde-1))||(ivar>=3&&(j<=2||j>=g.jde-2))))return;
 bool update=frame?frame->update:(c.ktau%c.interval==0||(c.restart&&c.ktau==c.ktaur));
 int nx=g.ime-g.ims+1,nk=g.kme-g.kms+1;
 #define A(i,k) g.at(i,k,j)
 #define XY(i) g.xy(i,j)
 #define E(v,n) err[(n)*nerr+(v)-1]
 #define O(v,n) obs[(n)*nvar+(v)-1]
 #define WT(a) (cells?local_wt:wt[(a)])
 #define W2(a) (cells?local_w2:w2[(a)])
 if(update){
  for(int i=ibeg;i<=iend;i++)for(int k=kbeg;k<=kend;k++){WT(A(i,k))=0.f;W2(A(i,k))=0.f;}
  float gx=ivar==1?0.f:.5f,gy=ivar==2?0.f:.5f;
  float rindx=c.rinxy*1000.f/c.dx;
  int maxj=j+int(rindx*c.rinfmx+.99f),minj=j-int(rindx*c.rinfmx+.99f);
  for(int n=frame?frame->first:0;n<(frame?frame->last:count);){
   int ns=int(nlevs[n]-lev[n])+1, next=n+ns;
   float ra=rio[n]-gx,rb=rjo[n]-gy,pob=O(5,n);
   if((fabsf(E(ivar,n))>9.e4f&&ns==1)||rb<float(minj)||rb>float(maxj)){n=next;continue;}
   int kob=bound(nintf(rko[n]+.45f),1,g.kte);
   float erf=E(ivar,n),qob=E(10,n),psurf=E(ivar>=3?6:ivar==1?7:8,n);
   bool surface=kob==1&&nlevs[n]<1.5f;
   float tw=timewt(c.xtime,c.dtmin,c.window,surface?c.sfcfact:1.f,timeob[n]);
   float radius=c.sfcfacr;
   if(!surface){float slope=(c.rinfmn-c.rinfmx)/(psurf-c.pfree);radius=min(max(slope*pob+c.rinfmx-slope*c.pfree,c.rinfmn),c.rinfmx);if(ns>1)radius=c.rinfmx;}
   int maxi=min(g.ide-1,int(ra+.99f+rindx*radius));
   int mini=max(surface?2:1,int(ra-rindx*radius-.99f));
   if(!(ra>=0.f-rindx*radius/3.f&&ra<=float(g.ide)+rindx*radius/3.f&&rb>=0.f-rindx*radius/3.f&&rb<=float(g.jde)+rindx*radius/3.f)){n=next;continue;}
   if(surface&&!c.nudge_pbl){n=next;continue;}
   float rx=float(j)-rb;
   for(int i=max(ibeg,mini);i<=min(iend,maxi);i++){
    float ry=float(i)-ra,rsq=rx*rx+ry*ry,ris;
    if(surface)ris=rindx*rindx*c.sfcfacr*c.sfcfacr;else ris=rindx*radius*rindx*radius;
    float wh;
    if(surface&&c.horiz==1){float dprim=sqrtf(rsq);float d=dprim+rindx*c.dcon*fabsf(psurf-.001f*pb[A(i,1)]);float dsq=d*d;wh=max(0.f,(ris-dsq)/(ris+dsq));}
    else{wh=(ris-rsq)/(ris+rsq);if(surface){float scratch=fabsf(psurf-.001f*pb[A(i,1)])*c.dcon;float pdfac=1.f-min(1.f,scratch);wh=wh*pdfac;}wh=max(0.f,wh);}
    if(surface){
     int komax=g.kte;
     if(c.vert==1){komax=max(3,kpbl[XY(i)]);if(kpbl[XY(i)]>25)komax=3;}
     float dzr=1.f/max(1.f,vih2[XY(i)]-vih1[XY(i)]);
     if(cells&&c.vert!=1)komax=min(komax,surface_top[XY(i)]);
     for(int k=max(c.vert==1?1:g.kts,kbeg);k<=min(komax,kend);k++){
      float wv=c.vert==1?float(komax-k+1)/float(komax):max(0.f,min(1.f,1.f+(vih1[XY(i)]-z[A(i,k)]+terrain[XY(i)])*dzr));
      if(c.vert!=1&&wv<=0.f)break;
      if(cells&&k!=kcell)continue;
      WT(A(i,k))=WT(A(i,k))+tw*wv*wh;
      if(ivar==4&&c.negative_qv>0){float cur=max(qv[A(i,k)],0.f);if(erf<0.f&&cur<qob)erf=erf*min(1.f,fabsf(cur/erf));}
      W2(A(i,k))=W2(A(i,k))+tw*tw*wh*wh*wv*wv*erf;
     }
    }else if(ns==1){
     int komin=1,komax=g.kte;
     for(int k=g.kte;k>=1;k--)if(.001f*(pb[A(i,k)]+pp[A(i,k)])>=pob+7.5f){komin=k;break;}
     for(int k=3;k<=g.kte;k++)if(.001f*(pb[A(i,k)]+pp[A(i,k)])<=pob-7.5f){komax=k;break;}
     if(kpbl[XY(i)]<=komax&&kpbl[XY(i)]>=komin){
      int kobs=1;for(int k=komin;k<=komax;k++)if(pob>.001f*(pb[A(i,k)]+pp[A(i,k)])){kobs=k;break;}
      if(kobs>int(E(5,n)))komin=max(kobs,komin);else komax=min(kpbl[XY(i)],komax);
     }
     if(c.nudge_pbl||komin>=kpbl[XY(i)])for(int k=max(komin,kbeg);k<=min(komax,kend);k++){
      float pijk=.001f*(pb[A(i,k)]+pp[A(i,k)]),res=erf,wv=max(1.f-fabsf(pijk-pob)/7.5f,0.f);
      WT(A(i,k))=WT(A(i,k))+tw*wh*wv;
      if(ivar==4&&c.negative_qv>0){float cur=max(qv[A(i,k)],0.f);if(erf<0.f&&cur<qob)res=res*min(1.f,fabsf(cur/erf));}
      W2(A(i,k))=W2(A(i,k))+tw*tw*wh*wh*res*wv*wv;
     }
    }else{
     for(int k=min(g.kte-2,kend);k>=kbeg;k--){
      float pijk=.001f*(pb[A(i,k)]+pp[A(i,k)]);if(pijk>O(5,n)||pijk<=O(5,n+ns-1))continue;
      float slope=(c.rinfmn-c.rinfmx)/(.001f*pb[A(i,1)]-c.pfree);
      float rf=min(max(slope*pijk+c.rinfmx-slope*c.pfree,c.rinfmn),c.rinfmx);
      ris=rindx*rf*rindx*rf;wh=max(0.f,(ris-rsq)/(ris+rsq));
      int nhi=-1,nlo=-1;float phi=0.f,plo=0.f,above=c.maxsnd_gap,below=c.maxsnd_gap;
      for(int nn=1;nn<ns;nn++)if(O(ivar,n+nn)>-800000.f&&O(5,n+nn)>-800000.f){phi=O(5,n+nn);above=fabsf(phi-pijk);if(phi<=pijk&&above<=c.maxsnd_gap){nhi=n+nn;break;}}
      if(nhi<0)continue;
      for(int nn=nhi-1;nn>=n;nn--)if(O(ivar,nn)>-800000.f&&O(5,nn)>-800000.f){plo=O(5,nn);below=fabsf(plo-pijk);if(plo>=pijk&&below<=c.maxsnd_gap){nlo=nn;break;}}
      if(nlo<0||above+below>c.maxsnd_gap)continue;
      float pdiff=fabsf(phi-plo)<.000001f?0.f:logf(pijk/plo)/logf(phi/plo);
      float res=E(ivar,nlo)+(E(ivar,nhi)-E(ivar,nlo))*pdiff;
      if(ivar==4&&c.negative_qv>0){float cur=qv[A(i,k)],ob=E(10,nlo)+(E(10,nhi)-E(10,nlo))*pdiff;if(res<0.f&&cur<ob)res=res*min(1.f,fabsf(cur/res));}
      if(c.nudge_pbl||k>kpbl[XY(i)]){float wv=1.f;WT(A(i,k))=WT(A(i,k))+tw*wh*wv;W2(A(i,k))=W2(A(i,k))+tw*tw*wh*wh*res*wv*wv;}
     }
    }
   }
   n=next;
  }
  for(int k=kbeg;k<=kend;k++)for(int i=ibeg;i<=iend;i++){
   if(WT(A(i,k))==0.f){W2(A(i,k))=0.f;WT(A(i,k))=1.f;}
   savwt[(A(i,k))*nvar+ivar-1]=W2(A(i,k))/WT(A(i,k));
  }
 }
 int is=c.inest==1?max(g.its,2):max(g.its,3),ie=c.inest==1?min(g.ite,g.ide-1):min(g.ite,g.ide-2);
 for(int k=kbeg;k<=kend;k++)for(int i=max(is,ibeg);i<=min(ie,iend);i++){
  float w=savwt[A(i,k)*nvar+ivar-1];
  if(ivar<3){float scale=mu[XY(i)]/(ivar==1?msfy[XY(i)]:msfx[XY(i)]);aten[A(i,k)]=aten[A(i,k)]+c.coefficient*scale*w*c.tfaci*1.f*1.f;}
  else aten[A(i,k)]=aten[A(i,k)]+c.coefficient*(c1h[k-g.kms]*mu[XY(i)]+c2h[k-g.kms])*w*c.tfaci*1.f*1.f;
 }
 #undef A
 #undef XY
 #undef E
 #undef WT
 #undef W2
 #undef O
}
extern "C" __global__ void fddaobs_vif(const float*vif,const float*regime,const float*pblh,float*h1,float*h2,
 float hmax,float fullmin,float rampmin,Grid g,unsigned int*error,const float*z,const float*terrain,int*top){
 int i=int(blockIdx.x*blockDim.x+threadIdx.x)+g.its,j=int(blockIdx.y*blockDim.y+threadIdx.y)+g.jts;if(i>g.ite||j>g.jte)return;
 int a=g.xy(i,j);
 if(!(regime[a]==1.f||regime[a]==2.f||regime[a]==3.f||regime[a]==4.f)){atomicOr(error,1u);h1[a]=__int_as_float(0x7fc00000);h2[a]=h1[a];return;}
 int r=regime[a]==1.f?0:regime[a]==2.f?2:4;
 float full=vif[r],ramp=vif[r+1],lo=full>=0.f?full:pblh[a]-(full+5000.f);lo=max(fullmin,lo);
 float hi=ramp>=0.f?lo+max(rampmin,ramp):max(lo+rampmin,pblh[a]-(ramp+5000.f));
 h1[a]=min(lo,hmax-rampmin);h2[a]=min(hi,hmax);
 float dzr=1.f/max(1.f,h2[a]-h1[a]);int last=g.kts-1;
 for(int k=g.kts;k<=g.kte;k++){
  float wv=max(0.f,min(1.f,1.f+(h1[a]-z[g.at(i,k,j)]+terrain[a])*dzr));
  if(wv<=0.f)break;last=k;
 }
 top[a]=last;
}
