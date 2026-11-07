/* WRF 4.6.1 module_radiation_driver.F:1336-1395.
 * All fields are profiles in the radiation caller's contiguous layout. */
extern "C" __global__ void cumulus_feedback(
    float* qc,float* qi,float* qs,float* cloud,
    const float* deep,const float* shallow,
    const float* cu_qc,const float* cu_qi,const float* cu_qs,
    const float* cu_ec,const float* cu_ei,const float* cu_es,
    float* ec,float* ei,float* es,
    int aerosol,int radii_meters,long long count) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    float cf=deep[i]+shallow[i];
    cloud[i]=(1.0f-cf)*cloud[i];
    cloud[i]=cloud[i]+cf;
    cloud[i]=fminf(1.0f,cloud[i]);
    qc[i]=qc[i]+cu_qc[i]*cf;
    qi[i]=qi[i]+cu_qi[i]*cf;
    if(aerosol) {
        float wc,wi,ws;
        if(qc[i]==0.0f && cu_qc[i]>0.0f) wc=1.0f;
        else if(cu_qc[i]==0.0f) wc=0.0f;
        else wc=(cu_qc[i]*cf)/(qc[i]+cu_qc[i]*cf);
        if(qi[i]==0.0f && cu_qi[i]>0.0f) wi=1.0f;
        else if(cu_qi[i]==0.0f) wi=0.0f;
        else wi=(cu_qi[i]*cf)/(qi[i]+cu_qi[i]*cf);
        if(qs[i]==0.0f && cu_qs[i]>0.0f) ws=1.0f;
        else if(cu_qs[i]==0.0f) ws=0.0f;
        else ws=(cu_qs[i]*cf)/(qs[i]+cu_qs[i]*cf);
        float bc=ec[i],bi=ei[i],bs=es[i];
        if(radii_meters) {bc=bc*1.e6f;bi=bi*1.e6f;bs=bs*1.e6f;}
        ec[i]=cu_ec[i]*wc+bc*(1.0f-wc);
        ei[i]=cu_ei[i]*wi+bi*(1.0f-wi);
        es[i]=cu_es[i]*ws+bs*(1.0f-ws);
        if(radii_meters) {ec[i]=ec[i]*1.e-6f;ei[i]=ei[i]*1.e-6f;es[i]=es[i]*1.e-6f;}
        qs[i]=qs[i]+cu_qs[i]*cf;
    }
}
