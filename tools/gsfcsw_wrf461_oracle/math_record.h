#include "cpu_shim.h"
#include <cstdio>
#include <cstdlib>
static float gsfc_math_record(int kind,float a,float b,float value) {
    static FILE *file=fopen(getenv("GSFC_MATH_RECORD_FILE"),"wb");
    float row[4]={(float)kind,a,b,value}; fwrite(row,sizeof(float),4,file);fflush(file);return value;
}
static float gsfc_exp_record(float x){return gsfc_math_record(0,x,0,expf(x));}
static float gsfc_log_record(float x){return gsfc_math_record(1,x,0,logf(x));}
static float gsfc_pow_record(float x,float y){return gsfc_math_record(2,x,y,powf(x,y));}
static float gsfc_log10_record(float x){return gsfc_math_record(3,x,0,log10f(x));}
#undef gfk_exp
#undef gfk_log
#undef gfk_pow
#define gfk_exp gsfc_exp_record
#define gfk_log gsfc_log_record
#define gfk_pow gsfc_pow_record
#define log10f gsfc_log10_record
