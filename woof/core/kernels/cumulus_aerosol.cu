/* WRF 4.6.1 module_cumulus_driver.F:1681-1931.
 * Monthly native-grid fields: (species,month,level,column), pressure in Pa.
 * Model fields: (level,column), output (species,level,column). */
extern "C" __global__ void cumulus_aerosol(
    const float* monthly,const float* pressure_monthly,const float* pressure,
    float* aerosol,int* status,float julian,int nlev,int nz,int ncol) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=ncol) return;
    const int date[12]={16,45,75,105,136,166,197,228,258,289,319,350};
    float day=julian+1.0f;
    int whole=(int)day;
    day=day-(float)whole;
    whole=whole%365;
    if(whole==0) whole=365;
    day=day+(float)whole;
    int np=0; bool found=false;
    for(int m=0;m<12;++m) if((float)date[m]>day && !found) {np=m;found=true;}
    int nm=np>0?np-1:11;
    float plus=(float)date[np],minus=(float)date[nm],f1,f2,delta;
    if(np==0) {
        delta=plus+365.0f-minus;
        if(day>plus) {f1=(plus+365.0f-day)/delta;f2=(day-minus)/delta;}
        else {f1=(plus-day)/delta;f2=(day+365.0f-minus)/delta;}
    } else {delta=plus-minus;f1=(plus-day)/delta;f2=(day-minus)/delta;}
    float pin[30];
    if(nlev!=30) {atomicOr(status,1);return;}
    for(int l=0;l<30;++l) {
        pin[l]=pressure_monthly[((size_t)nm*nlev+l)*ncol+col]*f1
              +pressure_monthly[((size_t)np*nlev+l)*ncol+col]*f2;
        if(!isfinite(pin[l]) || pin[l]<=0.0f || (l>0 && pin[l]<=pin[l-1])) {
            atomicOr(status,2);return;
        }
    }
    for(int s=0;s<10;++s) {
        int upper=0;
        for(int k=nz-1;k>=0;--k) {
            float p=pressure[(size_t)k*ncol+col];
            int matches=0;
            for(int l=upper;l<nlev-1;++l) if(pin[l]<p && p<=pin[l+1]) {upper=l;++matches;}
            if(matches>1) {atomicOr(status,2);return;}
            size_t base=(size_t)s*12*nlev*ncol;
            auto sample=[&](int l) {
                return monthly[base+((size_t)nm*nlev+l)*ncol+col]*f1
                      +monthly[base+((size_t)np*nlev+l)*ncol+col]*f2;
            };
            float value;
            if(matches==0 && p<pin[0]) value=sample(0)*p/pin[0];
            else if(matches==0 && p>pin[nlev-1]) value=sample(nlev-1);
            else {
                float up=p-pin[upper],down=pin[upper+1]-p;
                value=(sample(upper)*down+sample(upper+1)*up)/(down+up);
            }
            aerosol[((size_t)s*nz+k)*ncol+col]=value;
        }
    }
}
