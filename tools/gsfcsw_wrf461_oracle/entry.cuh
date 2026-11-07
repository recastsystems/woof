
// Bottom-to-top packed layers. All optional water fields are present.
extern "C" __global__ void gsfcsw_columns_kernel(
    const float *tables, float *inputs, float *outputs, float *workspace,
    int ncol, int nlay, int warm, int fqi, int fqn,
    int julday, float solcon, float latitude) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    int n = nlay + 1, one = 1, two = 2, yes = 1, no = 0;
    float *a = inputs + col*(9*n+2), *b = outputs + col*(3*n+2);
    float zero[129] = {}, cp = 1004.5f, g = 9.81f, obscur = 0.0f;
    gsfcswrad_(workspace + (col/32)*32*GSFC_WORKSPACE_WORDS + col%32, tables,
        b, b+3*n, zero, zero, a+9*n+1, a, a+4*n, a+5*n, zero,
        a+6*n, zero, zero, a+8*n, a+n, a+2*n, a+3*n, a+7*n,
        b+3*n+1, &cp, &g, &julday, &solcon, b+n, b+2*n, &warm,
        zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,
        &no, &yes,&yes,&no,&fqi,&no,&no,&fqn,
        a+9*n,&obscur,
        &one,&two,&one,&two,&one,&n,
        &one,&one,&one,&one,&one,&n,
        &one,&one,&one,&one,&one,&nlay,&latitude);
}

extern "C" __global__ void gsfcsw_math_probe(const float *rows,float *out,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    int kind=(int)rows[4*i]; float a=rows[4*i+1],b=rows[4*i+2];
    if(kind==0)out[i]=gfk_exp(a);
    if(kind==1)out[i]=gfk_log(a);
    if(kind==2)out[i]=gfk_pow(a,b);
    if(kind==3)out[i]=r_lg10(&a);
}
