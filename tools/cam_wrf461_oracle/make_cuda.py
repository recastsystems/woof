"""Emit device C from the validated WRF statement translation, without a runtime library."""
from pathlib import Path
import argparse,json,re
from make_driver import arguments
from transcribe import declarations
from specialize_vertical import local_arrays,external_arrays,count_expression,value,explicit_column_frames
from optimize_columns import optimize_columns
from stage_absorption import stage_absorption
from compact_shortwave import compact_shortwave, coalesce_flux, bound_frames

NAMES=('radems','trcplk','trcems','psi','phi','fh2oself','radabs','trcab','trcabn','radclwmx','radtpl','radoz2','aer_pth','aer_trn','sortarray','trcpth','radcswmx','raddedmx','findvalue')
HEADER='''#ifndef CAM_LEVELS
#define CAM_LEVELS 32
#endif
// Generated from pinned WRF 4.6.1 numeric statements by the lane tools.
// WOOF's own math routines provide exp, log and pow. No other math source is included.
typedef int integer;
typedef int logical;
typedef int ftnlen;
typedef float real;
typedef double doublereal;
#define TRUE_ 1
#define FALSE_ 0
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define abs(a) ((a)>=0?(a):-(a))
// Fast forecasts use CUDA binary64 libm. Strict replay retains the pinned math.
#if defined(GPUWM_WRF_EXACT) && GPUWM_WRF_EXACT
#define cam_exp64 glibc_exp
#define cam_log64 glibc_log
#define cam_pow64 glibc_pow
#else
#define cam_exp64 exp
#define cam_log64 log
#define cam_pow64 pow
#endif
__device__ double d_int(double *x){return trunc(*x);}
__device__ int cam_floor__(double *x){return int(floor(*x));}
__device__ double pow_dd(double *x,double *y){return cam_pow64(*x,*y);}
__device__ double pow_di(double *a,int *b){int n=*b;double x=*a,result=1;unsigned int m=n<0?-n:n;if(n<0)x=1/x;while(m){if(m&1)result*=x;m>>=1;if(m)x*=x;}return result;}
// WOOF base-ten reduction: log10(x)=exponent*log10(2)+log(m)/log(10).
// Keep the mantissa on the same side of unity as x to limit cancellation.
// The low term is the residual after rounding log10(2) to the chosen high word.
__device__ double d_lg10(double *x){
#if !defined(GPUWM_WRF_EXACT) || !GPUWM_WRF_EXACT
 return log10(*x);
#else
 int e;double m=frexp(*x,&e);if(e>0){m*=2;--e;}
 const double hi=0x1.3441350900000p-2,lo=0x1.ef3fde623e256p-35;
 return double(e)*hi+(double(e)*lo+0.43429448190325182765*cam_log64(m));
#endif
}
__device__ int cam_overlap_failure__(){asm("trap;");return 0;}
__device__ int endrun_(){asm("trap;");return 0;}
__device__ int cam_optics_warning__(){printf("RADCSWMX: aerosol optics outside physical range\\n");return 0;}
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('translation',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--driver',action='store_true')
    args=parser.parse_args()
    names=list(NAMES)
    if args.driver:names+=['param_cldoptics_calc','radctl','camrad','radinp','radozn','oznint','get_aerosol','aerosol_indirect','trcmix_clwrf','aqsat','cldefr','cldems','cldovrlap','getfactors','get_rf_scales','get_int_scales','vert_interpolate','background','scale_aerosols','estblf','validfactors','reltab','reitab','trcmix']
    pieces=[];prototypes=[];common=None;workspace_sizes=[]
    for name in names:
        code=(args.translation/(name+'.c')).read_text()
        code=re.sub(r'/\*.*?\*/','',code,flags=re.S)
        code=re.sub(r'^\s*#include.*$','',code,flags=re.M)
        match=re.search(r'struct\s*\{.*?\}\s*cam_state__;',code,re.S)
        if common is None:
            common=match[0].replace('struct {','struct CamState {').replace('} cam_state__;','};')
        code=code[:match.start()]+code[match.end():]
        code=re.sub(r'^\s*#define cam_state.*$','',code,flags=re.M)
        # F2C scalar constants remain device data, uniquely named per unit.
        function=re.search(r'\b(int|integer|logical|doublereal) '+name+r'_+\(.*?\)\s*\{',code,re.S)
        constants=code[:function.start()]
        body=code[function.start():]
        constnames=re.findall(r'static\s+(?:integer|doublereal)\s+(\w+)',constants)
        for constant in constnames:
            code=re.sub(r'\b'+constant+r'\b',name+'_'+constant,code)
        function=re.search(r'\b(int|integer|logical|doublereal) '+name+r'_+\(.*?\)\s*\{',code,re.S)
        constants=code[:function.start()].replace('static ','__device__ ')
        body=code[function.start():]
        body=re.sub(r'\s*double\s+(?:exp|d_int|d_lg10|sqrt|log|pow_dd|cos|sin|r_mod)\s*\(.*?;', '',body,flags=re.S)
        body=re.sub(r'\s*extern\s+.*?;', '',body,flags=re.S)
        body=re.sub(r'\s*integer i_dnnt\(.*?;', '',body,flags=re.S)
        body=body.replace('static ','')
        local_layout=json.loads((args.translation/'local-layout.json').read_text())
        body=local_arrays(body,local_layout[name])
        def initialize(match):
            dtype,values=match.groups()
            if '=' in values or '(' in values:return match[0]
            items=values.split(',')
            return '\n    '+dtype+' '+','.join(v.strip()+(' = {}' if '[' in v else ' = 0') for v in items)+';'
        brace=body.index('{')+1
        body=body[:brace]+re.sub(r'(?m)^\s*(integer|doublereal|real|logical)\s+([^;]+);',initialize,body[brace:])
        body,sizes=external_arrays(body,local_layout[name]);workspace_sizes.extend(sizes)
        body=body.replace('cam_state__1.','state->')
        signature=body[:body.index('{')].strip()
        signature=re.sub(r'\b'+name+r'_+\(',name+'_(CamState* state, ',signature)
        prototypes.append('__device__ '+signature+';')
        body=signature+'\n'+body[body.index('{'):]
        for callee in names:
            if callee!=name:
                body=re.sub(r'\b'+callee+r'_+\(',callee+'_(state, ',body)
        if args.driver:body=body.replace('read_camgases__(', 'read_camgases__(state, ')
        body=body.replace('exp(', 'cam_exp64(').replace('log(', 'cam_log64(')
        pieces.append(constants+'\n__device__ '+body)
    # Shared tables stay in device memory. Scalars belong to each column, so
    # driver gas updates cannot race between columns.
    layout=json.loads((args.translation/'state-layout.json').read_text())
    setup=['CamState column_state = {};','CamState* state=&column_state;']
    for field in layout:
        dtype={'double precision':'double','real':'float','integer':'int','logical':'int'}[field['dtype']]
        member='cg_'+field['name']+'__'
        if field['bounds']:
            common=re.sub(r'\b'+member+r'\[\s*\d+\s*\]', '*'+member,common)
            setup.append('state->'+member+'=('+dtype+'*)(bytes+'+str(field['offset'])+');')
        else:setup.append('state->'+member+'=*('+dtype+'*)(bytes+'+str(field['offset'])+');')
    common=common.replace('};','double driver_gases[5]; unsigned char* scratch; unsigned long long scratch_offset;\n};')
    budget='((64+'+'+'.join(workspace_sizes)+'+7)&~7)' # Align every column base, including odd columns.
    common+='\n#define CAM_WORK_BYTES ('+budget+')\n'+r'''
struct CamScratchFrame{CamState* state;unsigned long long saved;__device__ CamScratchFrame(CamState* s):state(s),saved(s->scratch_offset){} __device__ ~CamScratchFrame(){state->scratch_offset=saved;}};
template<typename T> __device__ T* cam_alloc(CamState* s,unsigned long long n){unsigned long long p=(s->scratch_offset+7)&~7ULL;s->scratch_offset=p+n*sizeof(T);T* out=(T*)(s->scratch+p);for(unsigned long long i=0;i<n;++i)out[i]=T{};return out;}
'''
    state_setup='\n'.join(setup)
    wrappers=[];metadata={}
    source=args.source.read_text()
    entries=[('radems','cam_emissivity'),('radclwmx','cam_lw'),('radcswmx','cam_sw')]
    if args.driver:entries.append(('camrad','cam_driver'))
    for routine,kernel in entries:
        formal,decl=arguments(source,routine)
        profile=[];pointerargs=[];parameters=[];fields=[]
        for name in formal:
            if name in ('lchnk','ncol','pcols','pver','pverp'):
                pointerargs.append('&'+name);continue
            declaration=decl[name]
            dtype='int' if declaration.startswith(('integer','logical')) else 'float' if re.match(r'real\b(?!\s*\(r8\))',declaration) else 'double'
            entity=declaration.split('::')[1].strip()
            shape=entity[entity.index('(')+1:entity.rindex(')')] if '(' in entity else (re.search(r'dimension\((.*?)\)',declaration)[1] if re.search(r'dimension\((.*?)\)',declaration) else '')
            shape_list=declarations(declaration)[0][2]
            count_expr=count_expression(shape_list)
            count=1
            from make_driver import split
            for dimension in split(shape) if shape else []:
                values={'pcols':1,'pver':32,'pverp':33,'naer_all':13,'nspint':19,'naer_groups':7,'nbands':2,'bnd_nbr_lw':7,'ims':1,'ime':1,'kms':1,'kme':33,'jms':1,'jme':1,'levsiz':59,'num_months':13,'paerlev':29,'naer_c':13,'cam_abs_dim1':4,'cam_abs_dim2':33}
                expression=re.sub(r'\b\w+\b',lambda m:str(values.get(m[0],m[0])),dimension)
                if ':' in expression:
                    lo,hi=expression.split(':');count*=eval(hi)-eval(lo)+1
                else:count*=eval(expression)
            profile.append(name);fields.append({'name':name,'dtype':dtype,'count':count,'output':'intent(out)' in declaration,'count_expression':count_expr})
            pointerargs.append('a'+str(len(profile)-1)+'+column*('+count_expr+')')
            parameters.append(dtype+'* a'+str(len(profile)-1))
        if routine=='camrad':
            profile.append('gases');fields.append({'name':'gases','dtype':'double','count':5,'output':False});parameters.append('double* gases')
            column_setup=state_setup+'\n'+''.join('state->driver_gases['+str(i)+']=gases[column*5+'+str(i)+'];\n' for i in range(5))
        else:column_setup=state_setup
        if routine == 'camrad':
            schedule = 'int worker=blockIdx.x*blockDim.x+threadIdx.x;\nint workers=gridDim.x*blockDim.x;\nfor(int column=worker;column<columns;column+=workers){\n'
            scratch_index = 'worker'
            close = '}\n}\n'
        else:
            schedule = 'int column=blockIdx.x*blockDim.x+threadIdx.x;\nif(column>=columns)return;\n'
            scratch_index = 'column'
            close = '}\n'
        wrappers.append('extern "C" __global__ __launch_bounds__(32) void '+kernel+'(int columns,const unsigned char* bytes,unsigned char* workspace,'+','.join(parameters)+'){\n'+schedule+'int lchnk=1,ncol=1,pcols=1,pver=CAM_LEVELS,pverp=CAM_LEVELS+1;\n'+column_setup+'\nstate->scratch=workspace+'+scratch_index+'*CAM_WORK_BYTES; state->scratch_offset=0;\n'+routine+'_(state,'+','.join(pointerargs)+');\n'+close)
        metadata[routine]={'workspace_expression':budget,'fields':profile,'parameters':fields,'levels':32,'source':'WRF v4.6.1 '+routine}
    wrapper='\n'.join(wrappers)
    wrapper+='extern "C" __global__ void cam_log10_probe(int count,const double* input,double* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){double x=input[i];output[i]=d_lg10(&x);}}\n'
    wrapper+=r'''extern "C" __global__ void cam_log10_variants(int count,const double* input,double* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 double x=input[i],old=x;
 output[i*5]=d_lg10(&old);
 int exponent;double m=frexp(x,&exponent);
 const double inv=0.43429448190325182765;
 const double hi=0x1.3441350900000p-2,lo=0x1.ef3fde623e256p-35;
 output[i*5+1]=double(exponent)*hi+(double(exponent)*lo+inv*cam_log64(m));
 if(exponent>0){m*=2;--exponent;}
 output[i*5+2]=double(exponent)*hi+(double(exponent)*lo+inv*cam_log64(m));
 m=frexp(x,&exponent);if(m<0.70710678118654752440){m*=2;--exponent;}
 output[i*5+3]=double(exponent)*hi+(double(exponent)*lo+inv*cam_log64(m));
 double a=double(exponent)*0.30102999566398119521,b=inv*cam_log64(m);
 double residual=__fma_rn(double(exponent),0.30102999566398119521,-a)+__fma_rn(inv,cam_log64(m),-b);
 output[i*5+4]=a+(b+residual);
}
'''
    support='''
__device__ double r_mod(float* a,float* b){return fmodf(*a,*b);}
__device__ integer i_dnnt(double* x){return int(round(*x));}
__device__ int cam_driver_failure__(){printf("CAM driver: invalid ozone pressure order, interpolation factors or aerosol state\\n");asm("trap;");return 0;}
__device__ int read_camgases__(CamState* state,int* year,float* julian,int* read,char* label,double* co2,double* n2o,double* ch4,double* f11,double* f12,ftnlen length){*co2=state->driver_gases[0];*n2o=state->driver_gases[1];*ch4=state->driver_gases[2];*f11=state->driver_gases[3];*f12=state->driver_gases[4];return 0;}
''' if args.driver else ''
    assembled=HEADER+common+support+'\n'+'\n'.join(prototypes)+'\n'+'\n'.join(pieces)+'\n'+wrapper
    assembled,metadata=explicit_column_frames(assembled,metadata)
    assembled=compact_shortwave(optimize_columns(assembled))
    if args.driver:
        assembled,metadata=stage_absorption(assembled,metadata)
        assembled,metadata=coalesce_flux(assembled,metadata)
        assembled,metadata=bound_frames(assembled,metadata)
    args.output.write_text('\n'.join(line.expandtabs(4).rstrip() for line in assembled.splitlines())+'\n')
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
