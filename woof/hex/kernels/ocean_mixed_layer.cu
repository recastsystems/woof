// Original conservative column coupling, Apache-2.0.
// Wind entrainment uses the WRF public-domain OML bulk-energy family.
// No horizontal ocean or ice dynamics; columns are native Voronoi cells.
extern "C" __global__ void ocean_step(
 int n, float dt, float threshold, const float* ocean,
 double* temp, double* depth, double* hu, double* hv,
 const double* reference_temp, const double* reference_depth,
 double* volume, double* area, double* ice_temp, double* snow_water,
 double* energy, double* entrainment, double* residual, float* melt,
 float* basal, float* net_flux, const float* coriolis,
 const float* sw, const float* lw, float* ice_hfx, float* ice_lh,
 const float* water_hfx, const float* water_lh, const float* ust,
 const float* wind_u, const float* wind_v, const float* tair,
 const float* pressure, const float* chs, const float* cqs,
 const float* rain, const float* frozen, const double* p
#ifdef HEX_PWP
 , double* columns, const double* pp, double* diagnostics
#endif
) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n || ocean[i]<0.5f) return;
#ifdef HEX_PWP
 // PWP is a tropical open-water option. Initially icy columns retain the
 // existing thermodynamic ice model throughout this run.
 if(diagnostics[4*i+3]>0.5) {
   const double rhoair=pressure[i]/(287.04*fmax(180.0,double(tair[i])));
   const double speed=fmax(1.e-10,hypot(double(wind_u[i]),double(wind_v[i])));
   const double stress=rhoair*double(ust[i])*ust[i];
   const double t=temp[i];
   const double qsw=(1-p[11])*sw[i];
   const double qnon=p[14]*(lw[i]-5.67051e-8*t*t*t*t)-water_hfx[i]-water_lh[i];
   const double freshwater=double(rain[i])/1000.0/dt-water_lh[i]/(2.5e6*1000.0);
   pwp_advance(columns+((long long)i)*4*int(pp[0]), pp, dt, coriolis[i],
       stress*wind_u[i]/speed, stress*wind_v[i]/speed, qsw,qnon,freshwater,
       diagnostics+4*i);
   temp[i]=columns[((long long)i)*4*int(pp[0])];
   depth[i]=diagnostics[4*i];
   hu[i]=columns[((long long)i)*4*int(pp[0])+2*int(pp[0])]*depth[i];
   hv[i]=columns[((long long)i)*4*int(pp[0])+3*int(pp[0])]*depth[i];
   const double absorbed=qnon+qsw*(1-exp(-pp[0]*pp[1]/pp[10]));
   energy[i]+=absorbed*dt; residual[i]+=diagnostics[4*i+1];
   net_flux[i]=absorbed; ice_temp[i]=temp[i];
   return;
 }
#endif
 const double rho=p[0], cw=p[1], rhoi=p[2], L=p[3], ki=p[4], ks=p[5];
 const double tf=273.15-0.054*p[6], gamma=p[7], maxh=p[8];
 const double hnew=p[9], cb=p[10], aw=p[11], ai=p[12], as=p[13], eps=p[14];
 const double sigma=5.67051e-8;
 double T=temp[i], H=depth[i], V=volume[i], A=area[i], Ti=ice_temp[i];
 const double oldE=rho*cw*H*(T-tf)-rhoi*L*V;
 const double oldV=V;
 // Zero-heat-capacity conductive sea ice with prognostic snow.
 double swe=snow_water[i];
 if(A>0) swe+=fmax(0.0,double(rain[i])*double(frozen[i]))/1000.0;
 const double snowh=swe*1000.0/p[15];
 const double alpha=ai+(as-ai)*fmin(1.0,swe/0.01);
 const double resistance=fmax(0.01,V/fmax(A,1.e-12))/ki+snowh/ks;
 const double density=pressure[i]/(287.04*fmax(180.0,double(tair[i])));
 const double saturation=0.622*611.2*exp(22.46*(Ti-273.15)/(Ti-0.55))/pressure[i];
 const double dqs=saturation*22.46*272.6/((Ti-0.55)*(Ti-0.55));
 const double turbulent_derivative=density*(1004.5*fmax(0.0,double(chs[i]))+2830000.0*fmax(0.0,double(cqs[i]))*dqs);
 double Ts=Ti;
 for(int k=0;k<6;k++) {
   const double q=(1-alpha)*sw[i]+eps*(lw[i]-sigma*Ts*Ts*Ts*Ts)-ice_hfx[i]-ice_lh[i]
       -turbulent_derivative*(Ts-Ti)+(tf-Ts)/resistance;
   const double d=turbulent_derivative+4*eps*sigma*Ts*Ts*Ts+1/resistance;
   Ts=fmin(273.15,fmax(150.0,Ts+q/fmax(d,1.e-6)));
 }
 const double qa=(1-alpha)*sw[i]+eps*(lw[i]-sigma*Ts*Ts*Ts*Ts)-ice_hfx[i]-ice_lh[i]
       -turbulent_derivative*(Ts-Ti);
 ice_hfx[i]+=density*1004.5*fmax(0.0,double(chs[i]))*(Ts-Ti);
 ice_lh[i]+=density*2830000.0*fmax(0.0,double(cqs[i]))*dqs*(Ts-Ti);
 const double conductive=(tf-Ts)/resistance;
 double surface_melt=fmax(0.0,(qa+conductive)*dt);
 const double snow_melt=fmin(swe,surface_melt/(1000*L));
 swe-=snow_melt; surface_melt-=snow_melt*1000*L;
 const double waterq=(1-aw)*sw[i]+eps*(lw[i]-sigma*T*T*T*T)-water_hfx[i]-water_lh[i];
 const double qb=fmin(fmax(0.0,rho*cw*cb*fmax(0.001,double(ust[i]))*(T-tf)),
                       fmax(0.0,rho*cw*H*(T-tf)/dt));
 double ocean_heat=((1-A)*waterq-A*qb)*dt;
 const double ice_heat=A*(qb*dt+surface_melt-conductive*dt);
 V-=ice_heat/(rhoi*L);
 if(V<0) { ocean_heat-=V*rhoi*L; V=0; }
 T+=ocean_heat/(rho*cw*H);
 // Exact inertial rotation avoids secular explicit-Euler energy growth.
 const double speed=fmax(1.e-10,hypot(double(wind_u[i]),double(wind_v[i])));
 const double tau=0.25*double(ust[i])*ust[i]/rho;
 const double angle=coriolis[i]*dt, c=cos(angle), s=sin(angle);
 const double ux=hu[i]+0.5*dt*tau*wind_u[i]/speed;
 const double vy=hv[i]+0.5*dt*tau*wind_v[i]/speed;
 const double U=c*ux+s*vy+0.5*dt*tau*wind_u[i]/speed;
 const double W=c*vy-s*ux+0.5*dt*tau*wind_v[i]/speed;
 // OML bulk potential-energy inversion and heat-conserving entrainment.
 const double alphaT=fmax((T-273.15)*1.e-5,1.e-6);
 const double thp=reference_temp[i]-gamma*(H-reference_depth[i]);
 const double a2=(T-thp)*H-0.5*gamma*H*H;
 const double b2=U*U+W*W;
 const double h2=-a2/gamma+sqrt(a2*a2/(gamma*gamma)+2*b2/(alphaT*9.81*gamma));
 const double nextH=fmin(maxh,fmax(H,sqrt(fmax(0.0,h2))));
 const double dh=nextH-H;
 const double below=fmax(tf,thp-0.5*gamma*dh);
 const double added=rho*cw*dh*(below-tf);
 T=(H*T+dh*below)/nextH; H=nextH;
 if(T<tf) { V+=rho*cw*H*(tf-T)/(rhoi*L); T=tf; }
 double nextA;
 if(V<=0) { V=0; nextA=0; swe=0; Ts=T; }
 else if(V<oldV && oldV>0) nextA=A*V/oldV;
 else nextA=fmin(1.0,A+(V-oldV)/hnew);
 if(V>0) nextA=fmax(double(threshold),fmin(1.0,nextA));
 if(nextA>0) swe=swe*A/nextA;
 temp[i]=T; depth[i]=H; hu[i]=U; hv[i]=W;
 volume[i]=V; area[i]=nextA; ice_temp[i]=Ts; snow_water[i]=swe;
 basal[i]=A*qb;
 const double external=((1-A)*waterq+A*qa)*dt-A*snow_melt*1000*L;
 net_flux[i]=external/dt;
 energy[i]+=external; entrainment[i]+=added;
 const double newE=rho*cw*double(depth[i])*(double(temp[i])-tf)-rhoi*L*double(volume[i]);
 residual[i]+=newE-oldE-external-added;
 melt[i]+=fmax(0.0,oldV-V)*rhoi;
}
