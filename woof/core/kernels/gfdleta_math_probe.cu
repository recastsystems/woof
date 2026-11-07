extern "C" __global__ void gd_math_probe(const int *kind,const float *inputs,float *outputs,int n){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;
 float x=inputs[3*k],y=inputs[3*k+1],v=0.f;
 switch(kind[k]){
 case 0:v=gd_exp(x);break;case 1:v=gd_log(x);break;case 2:v=gd_log10(x);break;
 case 3:v=gd_sqrt(x);break;case 4:v=gd_sin(x);break;case 5:v=gd_cos(x);break;
 case 6:v=gd_asin(x);break;case 7:v=gd_acos(x);break;case 8:v=gd_tan(x);break;
 case 9:v=gd_atan(x);break;case 10:v=gd_real_pow(x,y);break;
 }
 outputs[k]=v;
}
