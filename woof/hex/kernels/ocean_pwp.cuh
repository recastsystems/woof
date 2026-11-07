// Original implementation from equations, Apache-2.0.
// Price, Weller and Pinkel (1986), DOI:10.1029/JC091iC07p08411.
// Accessible algorithm: HYCOM PWP technical note, pp. 1-2,
// https://www.hycom.org/attachments/067_pwp.pdf (accessed 2026-10-23).
// Cell-local T/S/u/v layers; no horizontally staggered momentum interpolation.
// Linear EOS and single-exponential sunlight are explicit idealizations.
__device__ double pwp_density(const double* x, int n, int k, const double* p) {
 return -p[4]*(x[k]-p[12])+p[5]*(x[n+k]-p[13]); // rho anomaly / rho0
}
__device__ void pwp_mix(double* x,int n,int end) {
 for(int v=0;v<4;v++) {
   double sum=0; for(int k=0;k<end;k++) sum+=x[v*n+k];
   double mean=sum/end; for(int k=0;k<end;k++) x[v*n+k]=mean;
 }
}
__device__ double pwp_ri(const double* x,int n,int a,int b,double h,const double* p) {
 double du=x[2*n+b]-x[2*n+a],dv=x[3*n+b]-x[3*n+a];
 double dr=pwp_density(x,n,b,p)-pwp_density(x,n,a,p);
 if(dr>=0 && du*du+dv*dv<1.e-20) return 1.e30;
 return 9.81*h*dr/fmax(1.e-20,du*du+dv*dv);
}
__device__ void pwp_advance(double* x,const double* p,double dt,double f,
 double tx,double ty,double sw,double non,double freshwater,double* diagnostic) {
 const int n=int(p[0]); const double dz=p[1],rho=p[2],cw=p[3];
 double before=0; for(int k=0;k<n;k++) before+=x[k];
 // Surface freshwater uses the virtual salt flux convention. All temperature
 // fluxes are W/m2 into the ocean; freshwater is m/s into the ocean.
 x[n]-=dt*freshwater*x[n]/dz;
 double c=cos(f*dt),s=sin(f*dt);
 for(int k=0;k<n;k++) {
   x[k]+=dt*(sw*(exp(-k*dz/p[10])-exp(-(k+1)*dz/p[10]))+(k==0?non:0))/(rho*cw*dz);
   double u=x[2*n+k]+(k==0?dt*tx/(2*rho*dz):0);
   double v=x[3*n+k]+(k==0?dt*ty/(2*rho*dz):0);
   x[2*n+k]=c*u+s*v+(k==0?dt*tx/(2*rho*dz):0);
   x[3*n+k]=c*v-s*u+(k==0?dt*ty/(2*rho*dz):0);
 }
 // First relieve static instability, then diagnose the mixed layer from
 // adjacent density jumps, as separate operations in the specification.
 int h=1;
 while(h<n && pwp_density(x,n,h,p)<pwp_density(x,n,0,p)) {
   ++h; pwp_mix(x,n,h);
 }
 h=1;
 while(h<n && (pwp_density(x,n,h,p)-pwp_density(x,n,h-1,p))*rho<=p[9]) ++h;
 pwp_mix(x,n,h);
 while(h<n && pwp_ri(x,n,0,h,h*dz,p)<p[6]) { ++h; pwp_mix(x,n,h); }
 // Least-stable-interface relaxation. Equal-thickness pair mixing scales
 // density and velocity jumps by r, so Ri_new=Ri_old/r. Negative jumps
 // are fully homogenized. All four integrals are conserved by each pair.
 int iter=0;
 for(;iter<int(p[11]);iter++) {
   double minimum=p[7]; int a=-1;
   for(int k=h-1;k<n-1;k++) {
     double ri=pwp_ri(x,n,k,k+1,dz,p);
     if(ri<minimum) { minimum=ri; a=k; }
   }
   if(a<0) break;
   double fraction=minimum<=0 ? .5 : .5*(1-minimum/p[8]);
   for(int v=0;v<4;v++) {
     double transfer=fraction*(x[v*n+a+1]-x[v*n+a]);
     x[v*n+a]+=transfer; x[v*n+a+1]-=transfer;
   }
 }
 h=1;
 while(h<n && (pwp_density(x,n,h,p)-pwp_density(x,n,h-1,p))*rho<=p[9]) ++h;
 // Final homogenization follows the technical note. Co-located momentum
 // permits the same conservative operation on all four variables.
 pwp_mix(x,n,h);
 double after=0; for(int k=0;k<n;k++) after+=x[k];
 diagnostic[0]=h*dz;
 diagnostic[1]=(after-before)*rho*cw*dz-dt*(non+sw*(1-exp(-n*dz/p[10])));
 diagnostic[2]+=iter==int(p[11]); // recorded, never silently called converged
}
