"""Regenerate the column ABI from an explicitly selected public WRF source."""
from pathlib import Path
import argparse,json,re,importlib.util

ROOT=Path(__file__).parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wrf-source',type=Path,required=True)
parser.add_argument('--scratch-layout',type=Path,default=ROOT/'woof/core/gfdleta_column_layout.json')
parser.add_argument('--output-root',type=Path,default=ROOT)
options=parser.parse_args()
path=options.output_root/'woof/core/kernels/gfdleta_column.cuh'
s=path.read_text()
m=re.search(r'GD_DEVICE int etara_\((.*?)\)\s*\{',s,re.S)
args=[];typed=m[1].replace('\n',' ').split(',')
for x in typed:
 typ,name=x.strip().split('*');args.append((typ.strip(),name.strip()))
spec=importlib.util.spec_from_file_location('lower',Path(__file__).with_name('lower_column.py'))
lower=importlib.util.module_from_spec(spec);spec.loader.exec_module(lower)
source=options.wrf_source.read_text()
units=lower.routines(lower.logical_lines(source));decls={}
for _,line in units['ETARA']:
 if '::' not in line:continue
 lhs,rhs=line.split('::',1);d=re.search(r'DIMENSION\((.*?)\)',lhs,re.I)
 if d:
  dims=lower.bounds(d[1]);sizes=[]
  for dim in lower.split_top(dims):
   if ':' in dim:
    a,b=dim.split(':');sizes.append(eval(b.replace('GDMAX','81'))-eval(a.replace('GDMAX','81'))+1)
   else:sizes.append(eval(dim.replace('GDMAX','81')))
  size=__import__('math').prod(sizes)
 else:size=1
 for name in lower.split_top(rhs):decls[name.strip().lower()]=size
offsets={'real':0,'integer':0,'logical':0};fields=[];expr=[]
for typ,name in args:
 if typ=='GDTable':expr.append('table');continue
 if typ=='GDClock':expr.append('&clock');continue
 if typ=='GDScratch':expr.append('work');continue
 name0=name.rstrip('_')
 n=decls.get(name0,1);group='real' if typ=='real' else 'integer'
 off=offsets[group];offsets[group]+=n
 fields.append({'name':name0,'cname':name,'kind':group,'offset':off,'count':n})
 expr.append(('f' if group=='real' else 'ints')+'+'+str(off))
scratch=json.loads(options.scratch_layout.read_text())
schema={'fields':fields,'float_count':offsets['real'],'int_count':offsets['integer'],'tables':lower.GLOBALS,'clock':lower.CLOCK,'scratch_bytes':scratch.get('scratch_bytes',scratch.get('bytes'))}
if not isinstance(schema['scratch_bytes'],int) or schema['scratch_bytes']<1:
 raise ValueError('scratch layout must declare positive scratch_bytes or bytes')
(options.output_root/'woof/core/gfdleta_column_layout.json').write_text(json.dumps(schema,indent=2)+'\n')
calls='etara_('+','.join(expr)+');'
lw=next(x for x in fields if x['name']=='gfdl_lw');sw=next(x for x in fields if x['name']=='gfdl_sw')
body=f'''GD_DEVICE void gd_column_one(GDTable *table,float *f,int *ints,GDScratch *work,int longwave=1,int shortwave=1) {{
 GDClock clock={{0.0f,0.0f,0.0f,0.0f}};
 ints[{lw['offset']}]=longwave;ints[{sw['offset']}]=longwave?0:shortwave;
 {calls}
 if(!(longwave&&shortwave))return;
 ints[{sw['offset']}]=1;
 {calls}
}}
#ifndef GD_HOST_DIAGNOSTIC
extern "C" __global__ void gd_radiation_columns(GDTable *table,float *floats,int *integers,GDScratch *work,int ncol) {{
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<ncol)gd_column_one(table,floats+c*{offsets['real']},integers+c*{offsets['integer']},work+c);
}}

extern "C" __global__ void gd_radiation_columns_pooled(GDTable *table,float *floats,int *integers,GDScratch *work,int ncol,int slots,int longwave,int shortwave) {{
 int slot=blockIdx.x*blockDim.x+threadIdx.x;
 if(slot>=slots)return;
 for(int c=slot;c<ncol;c+=slots)
  gd_column_one(table,floats+c*{offsets['real']},integers+c*{offsets['integer']},work+slot,longwave,shortwave);
}}
extern "C" __global__ void gd_radiation_call(GDTable *table,float *floats,int *integers,GDScratch *scratch,GDClock *clocks,int ncol) {{
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=ncol)return;
 float *f=floats+c*{offsets['real']};int *ints=integers+c*{offsets['integer']};
 GDScratch *work=scratch+c;GDClock &clock=clocks[c];
 {calls}
}}
#endif
'''
(options.output_root/'woof/core/kernels/gfdleta_column.cu').write_text(body)
print(schema['float_count'],schema['int_count'],[x['name'] for x in fields])
