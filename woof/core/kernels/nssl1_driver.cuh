// Public native one-moment driver: column sedimentation, cell source processes.
namespace nssl1 {
template<int NZ> __device__ __forceinline__ void one_velocity(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol) {
int cell=blockDim.x*blockIdx.x+threadIdx.x;if(cell>=ncol*NZ)return;int col=cell;Column<1> ph;
Storage<float,1*Column<1>::na> an{1,1,1,1,1,1,1,Column<1>::na};Storage<float,Column<1>::na> xfall{1,1,1,1,1,Column<1>::na};
Storage<float,1> dn{1,1,1,1,1,1};
Storage<float,1> gz{1,1,1,1,1,1};
Storage<float,1> gzinv{1,1,1,1,1,1};
Storage<float,1> t0{1,1,1,1,1,1};
Storage<float,1> t7{1,1,1,1,1,1};
for(int j=0;j<1*ph.na;++j)an.data[j]=0;for(int j=0;j<ph.na;++j)xfall.data[j]=0;for(int k=1;k<=1;++k){
an(1,1,k,ph.lt)=in_theta[cell];
an(1,1,k,ph.lv)=in_qv[cell];
an(1,1,k,ph.lc)=in_qc[cell];
an(1,1,k,ph.lr)=in_qr[cell];
an(1,1,k,ph.li)=in_qi[cell];
an(1,1,k,ph.ls)=in_qs[cell];
an(1,1,k,ph.lh)=in_qg[cell];
if(ph.lhl>1)an(1,1,k,ph.lhl)=in_qh[cell];if(ph.lvh>0)an(1,1,k,ph.lvh)=in_qvolg[cell]*in_rho[cell];dn(1,1,k)=in_rho[cell];gz(1,1,k)=in_dz[cell];gzinv(1,1,k)=lake_div(1.f,in_dz[cell]);t0(1,1,k)=in_theta[cell]*in_exner[cell];t7(1,1,k)=::primary_ice_target(in_qv[cell],t0(1,1,k),dn(1,1,k),in_pressure[cell]);}
Storage<float,2*3*(Column<1>::lhab-Column<1>::lc+1)> vt{1,2,1,1,1,3,Column<1>::lc,Column<1>::lhab};
ph.velocity1d(dtp,1,1,1,an,ph.na,0,0,xfall,dn,gz,gzinv,t0,t7,2,1,1,1,0.,0.,0.,0.,0.,vt);
for(int l=1;l<=ph.na;++l)cells[(l-1)*NZ*ncol+cell]=an(1,1,1,l);
for(int l=ph.lc;l<=ph.lhab;++l)for(int v=1;v<=3;++v)cells[(ph.na+(l-ph.lc)*3+v-1)*NZ*ncol+cell]=vt(1,1,v,l);
}
template<int NZ> __device__ __forceinline__ void one_sed(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol) {
if constexpr(NZ>256){
int task=blockDim.x*blockIdx.x+threadIdx.x;Column<NZ> ph;int nspecies=ph.lhab-ph.lc+1;if(task>=ncol*nspecies)return;int col=task%ncol,selected=ph.lc+task/ncol;
Storage<float,NZ*Column<NZ>::na> an{1,1,1,1,1,NZ,1,Column<NZ>::na};Storage<float,Column<NZ>::na> xfall{1,1,1,1,1,Column<NZ>::na};
Storage<float,NZ> dn{1,1,1,1,1,NZ},gz{1,1,1,1,1,NZ},gzinv{1,1,1,1,1,NZ},t0{1,1,1,1,1,NZ},t7{1,1,1,1,1,NZ};
for(int j=0;j<ph.na;++j)xfall.data[j]=0;for(int k=1;k<=NZ;++k){int cell=(k-1)*ncol+col;an(1,1,k,selected)=cells[((selected-1)*NZ+k-1)*ncol+col];int volume=ph.lvol(selected);if(volume>1)an(1,1,k,volume)=cells[((volume-1)*NZ+k-1)*ncol+col];dn(1,1,k)=in_rho[cell];gz(1,1,k)=in_dz[cell];gzinv(1,1,k)=lake_div(1.f,in_dz[cell]);t0(1,1,k)=in_theta[cell]*in_exner[cell];t7(1,1,k)=0.f;}
ph.sediment1d(dtp,1,1,NZ,an,ph.na,0,0,xfall,dn,gz,gzinv,t0,t7,2,1,1,1,0.,0.,0.,0.,0.,cells+ph.na*NZ*ncol,ncol,col,selected);
for(int k=1;k<=NZ;++k){cells[((selected-1)*NZ+k-1)*ncol+col]=an(1,1,k,selected);int volume=ph.lvol(selected);if(volume>1)cells[((volume-1)*NZ+k-1)*ncol+col]=an(1,1,k,volume);}
cells[(ph.na+(selected-ph.lc)*3)*NZ*ncol+col]=xfall(1,1,selected);
} else {
// Native fallout is simultaneous in k: form every face flux, then update.
// A shared tile keeps the complete old-level fluxes until all lanes consume them.
constexpr int NC=4, NK=(NZ<=32?32:NZ<=64?64:NZ<=128?128:256);

int lane=threadIdx.x%NC,k=threadIdx.x/NC,col=blockIdx.x*NC+lane;
Column<NZ> ph;int selected=ph.lc+blockIdx.y,volume=ph.lvol(selected);
__shared__ float fm[(NK+1)*NC],fv[(NK+1)*NC],vm[NK*NC];
__shared__ int steps[NC];
int pos=k*NC+lane,cell=k*ncol+col;
bool valid=col<ncol&&k<NZ;
float mass=valid?cells[(selected-1)*NZ*ncol+cell]:0.f;
float vol=valid&&volume>1?cells[(volume-1)*NZ*ncol+cell]:0.f;
float rho=valid?in_rho[cell]:1.f,dzinv=valid?lake_div(1.f,in_dz[cell]):0.f;
float v1=0.f,v2=0.f,v3=0.f;
if(valid){int off=(ph.na+(selected-ph.lc)*3)*NZ*ncol+cell;
v1=lake_min(ph.vtmaxsed,cells[off]);v2=lake_min(ph.vtmaxsed,cells[off+NZ*ncol]);v3=lake_min(ph.vtmaxsed,cells[off+2*NZ*ncol]);}
float maxv=lake_max(0.f,v1*dzinv);maxv=lake_max(maxv,v2*dzinv);maxv=lake_max(maxv,v3*dzinv);
vm[pos]=maxv;__syncthreads();
for(int stride=NK/2;stride>0;stride/=2){if(k<stride)vm[pos]=lake_max(vm[pos],vm[pos+stride*NC]);__syncthreads();}
float vmax=vm[lane];int count=0;
if(ph.ido(selected)!=0&&vmax!=0.f){if(dtp*vmax<.7f)count=1;else if(dtp>20.f)count=lake_max(2,int(lake_div(dtp*vmax,.7f))+1);else count=1+int(dtp*vmax+.301f);}
if(k==0)steps[lane]=count;__syncthreads();
int maxsteps=0;for(int c=0;c<NC;++c)maxsteps=max(maxsteps,steps[c]);
float dtlocal=count>1?lake_div(dtp,float(count)):dtp,frac=lake_div(dtlocal,dtp),precip=0.f;
float dtrho=dzinv*lake_div(1.f,rho);
for(int n=0;n<maxsteps;++n){
 fm[pos]=(mass*v1)*rho;fv[pos]=vol*v1;
 if(k==0&&NZ==NK){fm[NZ*NC+lane]=0.f;fv[NZ*NC+lane]=0.f;}
 __syncthreads();
 if(valid&&n<count){
 // WRF's kmn test includes the first two levels; all-zero columns add zero.
 if(k==0)precip=precip+((mass*v1)*frac);
 mass=mass+((dtlocal*dtrho)*(fm[pos+NC]-fm[pos]));
 if(volume>1)vol=vol+((dtlocal*dzinv)*(fv[pos+NC]-fv[pos]));
 }
 __syncthreads();
}
if(valid){cells[(selected-1)*NZ*ncol+cell]=mass;if(volume>1)cells[(volume-1)*NZ*ncol+cell]=vol;}
if(k==0&&col<ncol)cells[(ph.na+(selected-ph.lc)*3)*NZ*ncol+col]=precip;
}

}
template<int NZ> __device__ __forceinline__ void one_precip(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol) {
int col=blockDim.x*blockIdx.x+threadIdx.x;if(col>=ncol)return;Column<1> ph;
Storage<float,Column<1>::na> xfall{1,1,1,1,1,Column<1>::na};Storage<float,1> dn{1,1,1,1,1,1};dn(1,1,1)=in_rho[col];
for(int l=ph.lc;l<=ph.lhab;++l)xfall(1,1,l)=cells[(ph.na+(l-ph.lc)*3)*NZ*ncol+col];
float sr=dtp*dn(1,1,1)*xfall(1,1,ph.ls)*1000.f/ph.xdn0(ph.lr);float gr=dtp*dn(1,1,1)*xfall(1,1,ph.lh)*1000.f/ph.xdn0(ph.lr);float hr=ph.lhl>1 ? dtp*dn(1,1,1)*xfall(1,1,ph.lhl)*1000.f/ph.xdn0(ph.lr):0.f;float rain=dtp*dn(1,1,1)*(xfall(1,1,ph.lr)+xfall(1,1,ph.ls)*1000.f/ph.xdn0(ph.lr)+xfall(1,1,ph.lh)*1000.f/ph.xdn0(ph.lr)+(ph.lhl>1?xfall(1,1,ph.lhl)*1000.f/ph.xdn0(ph.lr):0.f));
out_rainnc[col]+= rain;
out_rainncv[col]= rain;
out_snownc[col]+= sr;
out_snowncv[col]= sr;
out_graupelnc[col]+= gr;
out_graupelncv[col]= gr;
out_hailnc[col]+= hr;
out_hailncv[col]= hr;
out_sr[col]= (sr+hr+gr)/(rain+1.e-12f);
}
template<int NZ> __device__ __forceinline__ void one_cell(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol,int due) {
int cell=blockDim.x*blockIdx.x+threadIdx.x;if(cell>=ncol*NZ)return;Column<1> ph;Storage<float,Column<1>::na> an{1,1,1,1,1,1,1,Column<1>::na};for(int l=1;l<=ph.na;++l)an(1,1,1,l)=cells[(l-1)*NZ*ncol+cell];
Storage<float,1> t0{1,1,1,1,1,1};t0.data[0]=0;
Storage<float,1> t1{1,1,1,1,1,1};t1.data[0]=0;
Storage<float,1> t2{1,1,1,1,1,1};t2.data[0]=0;
Storage<float,1> t3{1,1,1,1,1,1};t3.data[0]=0;
Storage<float,1> t4{1,1,1,1,1,1};t4.data[0]=0;
Storage<float,1> t5{1,1,1,1,1,1};t5.data[0]=0;
Storage<float,1> t6{1,1,1,1,1,1};t6.data[0]=0;
Storage<float,1> t7{1,1,1,1,1,1};t7.data[0]=0;
Storage<float,1> t8{1,1,1,1,1,1};t8.data[0]=0;
Storage<float,1> t9{1,1,1,1,1,1};t9.data[0]=0;
Storage<float,1> pn{1,1,1,1,1,1};pn.data[0]=0;
Storage<float,1> dn{1,1,1,1,1,1};dn.data[0]=0;
Storage<float,1> gz{1,1,1,1,1,1};gz.data[0]=0;
Storage<float,1> gzinv{1,1,1,1,1,1};gzinv.data[0]=0;
Storage<float,1> w{1,1,1,1,1,1};w.data[0]=0;
Storage<float,1> t00{1,1,1,1,1,1};t00.data[0]=0;
Storage<float,1> t77{1,1,1,1,1,1};t77.data[0]=0;
Storage<float,1> dbz{1,1,1,1,1,1};dbz.data[0]=0;
Storage<float,1> tmp{1,1,1,1,1,1};tmp.data[0]=0;
Storage<float,1> axtra{1,1,1,1,1,1};axtra.data[0]=0;
Storage<float,1> rainprod{1,1,1,1,1,1};rainprod.data[0]=0;
Storage<float,1> evapprod{1,1,1,1,1,1};evapprod.data[0]=0;
Storage<float,1> thproc{1,1,1,1,1,1};thproc.data[0]=0;
Storage<float,1> tke{1,1,1,1,1,1};tke.data[0]=0;
Storage<float,1> elec{1,1,1,1,1,1};elec.data[0]=0;
Storage<float,3> alpha2d{1,1,1,1,1,3};alpha2d.data[0]=alpha2d.data[1]=alpha2d.data[2]=0;
pn.data[0]=in_pressure[cell];
dn.data[0]=in_rho[cell];
gz.data[0]=in_dz[cell];
w.data[0]=in_w[cell];
t77.data[0]=in_exner[cell];
t0.data[0]=an(1,1,1,ph.lt)*t77.data[0];t00.data[0]=lake_div(380.f,pn.data[0]);t7.data[0]=::primary_ice_target(an(1,1,1,ph.lv),t0.data[0],dn.data[0],pn.data[0]);
int k=cell/ncol,col=cell%ncol,km=max(k-1,0),km2=max(k-2,0),kp=min(k+1,NZ-1),kpn=min(k+1,NZ-2);ph.global_k=k+1;ph.global_nz=NZ;ph.below_t=cells[km*ncol+col]*in_exner[km*ncol+col];ph.below2_t=cells[km2*ncol+col]*in_exner[km2*ncol+col];ph.below_w=in_w[km*ncol+col];ph.above_w=in_w[kp*ncol+col];ph.below_in=::primary_ice_target(cells[(ph.lv-1)*NZ*ncol+km*ncol+col],ph.below_t,in_rho[km*ncol+col],in_pressure[km*ncol+col]);ph.above_in=::primary_ice_target(cells[(ph.lv-1)*NZ*ncol+kpn*ncol+col],cells[kpn*ncol+col]*in_exner[kpn*ncol+col],in_rho[kpn*ncol+col],in_pressure[kpn*ncol+col]);ph.below_qc=cells[(ph.lc-1)*NZ*ncol+km*ncol+col];
ph.nssl_2mom_gs(1,1,1,ph.na,1,0,0,dtp,gz,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,an,dn,t77,pn,w,0,t00,t77,ph.ventr,0.f,ph.c1sw,1,ph.ido,ph.xdnmx,ph.xdnmn,ph.cdx,ph.xdn0,dbz,tke,thproc,1,1000.f,1000.f,1,0.,axtra,true,false,rainprod,evapprod,alpha2d,elec,1,1,1,1,1);
ph.nucond(1,1,1,ph.na,1,0,0,dtp,1,gz,t0,t9,an,dn,t77,pn,w,1,axtra,true,tmp,t00,t77,false);
if(due){ph.radardd02(1,1,1,0,ph.na,an,t0,dbz,dn,1,ph.cnoh,ph.rho_qh,ph.ipconc,1,0);out_refl_10cm[cell]=dbz.data[0];}
out_theta[cell]=an(1,1,1,ph.lt);
out_qv[cell]=an(1,1,1,ph.lv);
out_qc[cell]=an(1,1,1,ph.lc);
out_qr[cell]=an(1,1,1,ph.lr);
out_qi[cell]=an(1,1,1,ph.li);
out_qs[cell]=an(1,1,1,ph.ls);
out_qg[cell]=an(1,1,1,ph.lh);
if(ph.lhl>1)out_qh[cell]=an(1,1,1,ph.lhl);if(ph.lvh>0)out_qvolg[cell]=lake_div(an(1,1,1,ph.lvh),dn.data[0]);
}
}
extern "C" __global__ void nssl1_sed(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol){nssl1::one_sed<NSSL1_NZ>(in_theta,in_qv,in_qc,in_qr,in_qi,in_qs,in_qg,in_qh,in_qndrop,in_qnr,in_qni,in_qns,in_qng,in_qnh,in_qnn,in_qvolg,in_qvolh,in_qzr,in_qzg,in_qzh,in_pressure,in_exner,in_rho,in_dz,in_w,out_theta,out_qv,out_qc,out_qr,out_qi,out_qs,out_qg,out_qh,out_qndrop,out_qnr,out_qni,out_qns,out_qng,out_qnh,out_qnn,out_qvolg,out_qvolh,out_qzr,out_qzg,out_qzh,out_refl_10cm,out_effc_m,out_effi_m,out_effs_m,out_rainnc,out_rainncv,out_snownc,out_snowncv,out_graupelnc,out_graupelncv,out_hailnc,out_hailncv,out_sr,cells,dtp,ncol);}
extern "C" __global__ void nssl1_process(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol,int due){nssl1::one_cell<NSSL1_NZ>(in_theta,in_qv,in_qc,in_qr,in_qi,in_qs,in_qg,in_qh,in_qndrop,in_qnr,in_qni,in_qns,in_qng,in_qnh,in_qnn,in_qvolg,in_qvolh,in_qzr,in_qzg,in_qzh,in_pressure,in_exner,in_rho,in_dz,in_w,out_theta,out_qv,out_qc,out_qr,out_qi,out_qs,out_qg,out_qh,out_qndrop,out_qnr,out_qni,out_qns,out_qng,out_qnh,out_qnn,out_qvolg,out_qvolh,out_qzr,out_qzg,out_qzh,out_refl_10cm,out_effc_m,out_effi_m,out_effs_m,out_rainnc,out_rainncv,out_snownc,out_snowncv,out_graupelnc,out_graupelncv,out_hailnc,out_hailncv,out_sr,cells,dtp,ncol,due);}
namespace nssl1 { template<int NZ> __device__ __forceinline__ void one_radar(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol){int cell=blockDim.x*blockIdx.x+threadIdx.x;if(cell>=ncol*NZ)return;Column<1> ph;Storage<float,Column<1>::na> an{1,1,1,1,1,1,1,Column<1>::na};for(int l=1;l<=ph.na;++l)an(1,1,1,l)=0;an(1,1,1,ph.lt)=in_theta[cell];an(1,1,1,ph.lv)=in_qv[cell];an(1,1,1,ph.lc)=in_qc[cell];an(1,1,1,ph.lr)=in_qr[cell];an(1,1,1,ph.li)=in_qi[cell];an(1,1,1,ph.ls)=in_qs[cell];an(1,1,1,ph.lh)=in_qg[cell];if(ph.lhl>1)an(1,1,1,ph.lhl)=in_qh[cell];if(ph.lvh>0)an(1,1,1,ph.lvh)=in_qvolg[cell]*in_rho[cell];Storage<float,1> temp{1,1,1,1,1,1},dbz{1,1,1,1,1,1},rho{1,1,1,1,1,1};temp.data[0]=in_theta[cell]*in_exner[cell];rho.data[0]=in_rho[cell];ph.radardd02(1,1,1,0,ph.na,an,temp,dbz,rho,1,ph.cnoh,ph.rho_qh,0,1,0);out_refl_10cm[cell]=dbz.data[0];}}
extern "C" __global__ void nssl1_radar(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol){nssl1::one_radar<NSSL1_NZ>(in_theta,in_qv,in_qc,in_qr,in_qi,in_qs,in_qg,in_qh,in_qndrop,in_qnr,in_qni,in_qns,in_qng,in_qnh,in_qnn,in_qvolg,in_qvolh,in_qzr,in_qzg,in_qzh,in_pressure,in_exner,in_rho,in_dz,in_w,out_theta,out_qv,out_qc,out_qr,out_qi,out_qs,out_qg,out_qh,out_qndrop,out_qnr,out_qni,out_qns,out_qng,out_qnh,out_qnn,out_qvolg,out_qvolh,out_qzr,out_qzg,out_qzh,out_refl_10cm,out_effc_m,out_effi_m,out_effs_m,out_rainnc,out_rainncv,out_snownc,out_snowncv,out_graupelnc,out_graupelncv,out_hailnc,out_hailncv,out_sr,cells,dtp,ncol);}
extern "C" __global__ void nssl1_velocity(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol){nssl1::one_velocity<NSSL1_NZ>(in_theta,in_qv,in_qc,in_qr,in_qi,in_qs,in_qg,in_qh,in_qndrop,in_qnr,in_qni,in_qns,in_qng,in_qnh,in_qnn,in_qvolg,in_qvolh,in_qzr,in_qzg,in_qzh,in_pressure,in_exner,in_rho,in_dz,in_w,out_theta,out_qv,out_qc,out_qr,out_qi,out_qs,out_qg,out_qh,out_qndrop,out_qnr,out_qni,out_qns,out_qng,out_qnh,out_qnn,out_qvolg,out_qvolh,out_qzr,out_qzg,out_qzh,out_refl_10cm,out_effc_m,out_effi_m,out_effs_m,out_rainnc,out_rainncv,out_snownc,out_snowncv,out_graupelnc,out_graupelncv,out_hailnc,out_hailncv,out_sr,cells,dtp,ncol);}
extern "C" __global__ void nssl1_precip(const float* in_theta,const float* in_qv,const float* in_qc,const float* in_qr,const float* in_qi,const float* in_qs,const float* in_qg,const float* in_qh,const float* in_qndrop,const float* in_qnr,const float* in_qni,const float* in_qns,const float* in_qng,const float* in_qnh,const float* in_qnn,const float* in_qvolg,const float* in_qvolh,const float* in_qzr,const float* in_qzg,const float* in_qzh,const float* in_pressure,const float* in_exner,const float* in_rho,const float* in_dz,const float* in_w,float* out_theta,float* out_qv,float* out_qc,float* out_qr,float* out_qi,float* out_qs,float* out_qg,float* out_qh,float* out_qndrop,float* out_qnr,float* out_qni,float* out_qns,float* out_qng,float* out_qnh,float* out_qnn,float* out_qvolg,float* out_qvolh,float* out_qzr,float* out_qzg,float* out_qzh,float* out_refl_10cm,float* out_effc_m,float* out_effi_m,float* out_effs_m,float* out_rainnc,float* out_rainncv,float* out_snownc,float* out_snowncv,float* out_graupelnc,float* out_graupelncv,float* out_hailnc,float* out_hailncv,float* out_sr,float* cells,float dtp,int ncol){nssl1::one_precip<NSSL1_NZ>(in_theta,in_qv,in_qc,in_qr,in_qi,in_qs,in_qg,in_qh,in_qndrop,in_qnr,in_qni,in_qns,in_qng,in_qnh,in_qnn,in_qvolg,in_qvolh,in_qzr,in_qzg,in_qzh,in_pressure,in_exner,in_rho,in_dz,in_w,out_theta,out_qv,out_qc,out_qr,out_qi,out_qs,out_qg,out_qh,out_qndrop,out_qnr,out_qni,out_qns,out_qng,out_qnh,out_qnn,out_qvolg,out_qvolh,out_qzr,out_qzg,out_qzh,out_refl_10cm,out_effc_m,out_effi_m,out_effs_m,out_rainnc,out_rainncv,out_snownc,out_snowncv,out_graupelnc,out_graupelncv,out_hailnc,out_hailncv,out_sr,cells,dtp,ncol);}
