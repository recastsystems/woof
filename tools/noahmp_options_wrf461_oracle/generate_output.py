"""Offline expression transcription from the pinned public-domain WRF source."""
from pathlib import Path
import ast,hashlib,re

def generate(data):
    assert hashlib.sha256(data).hexdigest()=="1a6429debc72d2d9092d66e1d6cd4dbc04978bc1a21673eb48adcf15e2d60952"
    text=data.decode()
    part=text[text.index('   subroutine noahmp_output_calc('):text.index('   end subroutine noahmp_output_calc')]
    flat=part[:part.index('!------')].replace('&','').replace('\n',' ')
    args=re.search(r'noahmp_output_calc\((.*)\)',flat).group(1).replace(' ','').split(',')
    acc=args[args.index('acsagv'):args.index('sag')];acc.remove('isnow')
    inputs=args[args.index('sag'):]+['isnow']
    # Strict translation by Python AST retains the native left-to-right FP32 expression tree.
    def expression(x):
     node=ast.parse(x,mode='eval').body
     def emit(n):
      if isinstance(n,ast.Name):return n.id if n.id=='kjconvert' else n.id+'[c]'
      if isinstance(n,ast.Constant):return repr(float(n.value))+'f'
      if isinstance(n,ast.BinOp):
       op={ast.Add:'AD',ast.Sub:'SU',ast.Mult:'MU',ast.Div:'DV'}[type(n.op)]
       return f'{op}({emit(n.left)},{emit(n.right)})'
      raise ValueError(ast.dump(n))
     return emit(node)
    body=['float kjconvert=DV(dt,1000.0f);'];depth=0
    for raw in part[part.index('         kjconvert ='):part.index('      enddo')].splitlines()[1:]:
     line=raw.split('!')[0].strip().lower()
     if not line:continue
     if line.startswith('if(opt_run'):body.append('if(opt_run==5){');continue
     if line.startswith('if(mod'):body.append('if(groundwater_due){');continue
     if line.startswith('if(isnow'):body.append('if(isnow[c]<0 && isnow[c]>-5){');continue
     if line=='endif':body.append('}');continue
     m=re.fullmatch(r'(\w+)\(i,j\)\s*=\s*(.*)',line);assert m,line
     rhs=re.sub(r'\(i,j\)','',m[2]);rhs=re.sub(r'\bdt\b','timestep',rhs)
     body.append(m[1]+'[c]='+expression(rhs).replace('timestep[c]','dt')+';')
    header='''// WRF v4.6.1 d66e442f, phys/module_diag_misc.F:641-846.
    // Original notice: licenses/LICENSE-WRF-public-domain.txt.
    // Generated offline from pinned expressions. Native FP32 tree retained.
    #ifdef WRF_EXACT
    #define AD(a,b) __fadd_rn(a,b)
    #define SU(a,b) __fsub_rn(a,b)
    #define MU(a,b) __fmul_rn(a,b)
    #define DV(a,b) __fdiv_rn(a,b)
    #else
    #define AD(a,b) ((a)+(b))
    #define SU(a,b) ((a)-(b))
    #define MU(a,b) ((a)*(b))
    #define DV(a,b) ((a)/(b))
    #endif
    '''
    header="\n".join(line.lstrip() for line in header.splitlines()).rstrip()+"\n"
    params=[f'float* {n}' for n in acc]+[f'const {"int" if n=="isnow" else "float"}* {n}' for n in inputs]+['float dt','int n','int reset','int opt_run','int groundwater_due']
    source=header+'extern "C" __global__ void noahmp_output_calc(\n'+',\n'.join(params)+'){\nint c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;\nif(reset){\n'+'\n'.join(n+'[c]=0.0f;' for n in acc)+'\n}\n'+'\n'.join(body)+'\n}\n'
    return source

if __name__=="__main__":
    import sys
    sys.stdout.write(generate(Path(sys.argv[1]).read_bytes()))
