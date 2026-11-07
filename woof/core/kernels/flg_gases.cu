// Full WRF FLG gas drivers; generated helpers retain source association.
extern "C" __global__ void flg_gas_driver(int ncol,int nv1,int npoints,
    const int* points,const float* pressure,const float* temperature,
    const float* humidity,const float* ozone,const float* tables,const int* offsets,
    const float* gas_values,const int* switches,float* optical_depth,float* weights,float* temperature_out)
{
    int task=blockIdx.x*blockDim.x+threadIdx.x;
    if(task>=ncol*npoints) return;
    int col=task/npoints,point=task%npoints,nv=nv1-1;
    const int* meta=points+point*4;
    float pt[101];
    for(int i=0;i<nv1;++i) pt[i]=temperature[col*nv1+i];
    float hk=0.0f;
    if(meta[0]==0)
        flg_gases(nv,nv1,meta[1],meta[2],hk,pressure+col*nv1,pt,humidity+col*nv1,ozone+col*nv1,
                  optical_depth+task*nv,tables,offsets,gas_values+col*11,switches+col*20);
    else
        flg_gases_new(nv,nv1,meta[1],meta[2],meta[3],hk,pressure+col*nv1,pt,humidity+col*nv1,ozone+col*nv1,
                      optical_depth+task*nv,tables,offsets,gas_values+col*11,switches+col*20);
    weights[task]=hk;
    for(int i=0;i<nv1;++i) temperature_out[task*nv1+i]=pt[i];
}
