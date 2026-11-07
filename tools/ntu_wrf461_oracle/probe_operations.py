"""Control CUDA operator lowering using explicit IEEE arithmetic intrinsics."""
import argparse,json,sys,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.ntu_wrf461_oracle import f2cu

def generate(source,parameters,out):
    original=f2cu.Emitter.emit_bin
    def emit_bin(self,e,dry):
        if e.op not in ('+','-','*','/'):
            return original(self,e,dry)
        a,ta=self.emit(e.a,dry); b,tb=self.emit(e.b,dry); t=self.maxt(ta,tb)
        if t not in ('f','d'): return original(self,e,dry)
        a=self.conv(a,ta,t); b=self.conv(b,tb,t)
        name=('F' if t=='f' else 'D')+{'+':'ADD','-':'SUB','*':'MUL','/':'DIV'}[e.op]
        return f'{name}({a}, {b})',t
    f2cu.Emitter.emit_bin=emit_bin
    previous=sys.argv
    try:
        sys.argv=['f2cu.py',str(source),'--params-in',str(parameters),'--out',str(out)]
        f2cu.main()
    finally: sys.argv=previous

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('source',type=Path);ap.add_argument('parameters',type=Path);ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    gen=args.output.with_suffix('.cuh'); generate(args.source,args.parameters,gen)
    import cupy as cp
    from woof.core import ntu
    from tools.ntu_wrf461_oracle.run_gpu import main as run
    original=ntu.module_source
    def altered(*a,**kw):
        text=original(*a,**kw); start=text.index('// ---- BEGIN transcription (f2cu.py) ----');end=text.index('// ---- END transcription ----',start)
        return text[:start]+gen.read_text()+'\n'+text[end:]
    ntu.module_source=altered
    run(args.input,args.output)
    receipt={'source_sha256':hashlib.sha256(gen.read_bytes()).hexdigest(),'options':ntu.compile_options()}
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
