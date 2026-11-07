"""Reproduce native bounds failure; replay defined native and repaired rows.

Run on the strict native host, after building the whole DFI crate with cargo.
All native inputs, output words, source, commands and hashes stay in --output.
"""
from pathlib import Path
import argparse, ctypes as C, difflib, hashlib, json, subprocess, time

PIN = 'c7b94989ccdf637ebe756b9dcdff26d5b9ba798852ec74c3415a853a32a4ae0f'
HERE = Path(__file__).resolve().parent
def bounded_source(text):
    """Repair only scratch sizing and writes beyond the output contract."""
    assert text.count('integer, PARAMETER  :: NOMAX=100') == 1
    text = text.replace('integer, PARAMETER  :: NOMAX=100', 'integer :: NOMAX')
    # Automatic arrays need an expression in their declaration. Retain the
    # native NOMAX argument of RHOCOF with its matching actual work extent.
    text = text.replace('dimension(0:NOMAX)', 'dimension(0:MAX(nstep,norder))')
    text = text.replace('dimension(0:NOMAX,0:NOMAX)',
                        'dimension(0:MAX(nstep,norder),0:MAX(nstep,norder))')
    text = text.replace('      DTT = ABS(DT)',
                        '      NOMAX = MAX(nstep,norder)\n      DTT = ABS(DT)')
    text = text.replace('       FROW (NC) = 0.', '')
    text = text.replace('!     Fill up the Filter Matrix.',
                        '    FROW(:) = 0.\n\n!     Fill up the Filter Matrix.')
    text = text.replace('      DO NC=0,NSTEP\n        FROW(NC)',
                        '      DO NC=0,MIN(NSTEP,NOSIZE)\n        FROW(NC)')
    return text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    source=a.source.read_bytes();assert hashlib.sha256(source).hexdigest()==PIN
    text=source.decode(); original=text[text.index('   SUBROUTINE dfcoef ('):text.index('    SUBROUTINE optfil (')]
    (out/'dfcoef-original.F90').write_text(original)
    fixed=bounded_source(original);(out/'dfcoef-bounded.F90').write_text(fixed)
    (out/'native-bounds.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),fixed.splitlines(True),fromfile='dfcoef-original.F90',tofile='dfcoef-bounded.F90')))
    driver=HERE/'probe.F90'
    flags=['-cpp','-DWRFPLUS=0','-ffree-form','-ffree-line-length-none','-O2',
           '-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize','-fcheck=bounds','-fbacktrace']
    commands=[]
    def run(args,label,expect=0):
        start=time.monotonic();args=list(map(str,args))
        r=subprocess.run(args,cwd=out,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/(label+'.log')).write_text(r.stdout)
        commands.append(dict(argv=args,exit=r.returncode,seconds=time.monotonic()-start,log=label+'.log'))
        assert (r.returncode==0)==(expect==0),(args,r.stdout)
        return r
    for name in ('original','bounded'):
        run(['gfortran',*flags,out/f'dfcoef-{name}.F90',driver,'-o',out/name],'build-'+name)
    failure=run([out/'original','dfcoef',10,8,12,3600,out/'failure.bin'],'native-failure',expect=1)
    assert 'above upper bound' in failure.stdout
    lib=C.CDLL(str(a.library.resolve())); fn=lib.aw_dfi_weights
    fn.argtypes=[C.c_int,C.c_float,C.c_float,C.c_int,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
    receipt=dict(source_sha256=PIN,scope='coefficient replay; original native safe calls distinguished from bounded native repair',
                 library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),cases=[],words=0,differing_words=0,commands=commands)
    def words(data):return [int.from_bytes(data[i:i+4],'little') for i in range(0,len(data),4)]
    for count in (3,4,9,10,21,22,51,52,101,102,301,302,451):
        half=(count+1)//2; steps=half-1
        for dt in (-12,12):
            for nf in (range(9) if count in (9,10,21,22) else (8,)):
                name=f'{count}-{dt}-{nf}';path=out/(name+'.bin')
                arm='original' if nf<8 else 'bounded'
                run([out/arm,'dfcoef',steps,nf,dt,3600,path],name)
                data=path.read_bytes();prefix=words(data)
                if nf==8 and 2*steps<=100:
                    safe=out/(name+'-safe.bin')
                    run([out/'original','safe',2*steps,nf,dt,3600,safe],name+'-safe')
                    assert safe.read_bytes()[:len(data)]==data,'defined native RHOFIL differs from repair'
                    arm='original-safe-RHOFIL-prefix'
                expected=prefix+list(reversed(prefix[:-1] if count%2 else prefix))
                result=(C.c_float*count)();written=C.c_size_t()
                status=fn((count-1)*abs(dt),dt,3600,nf,result,count,C.byref(written))
                assert status==0 and written.value==count
                (out/(name+'-rust.bin')).write_bytes(bytes(result))
                different=sum(x!=y for x,y in zip(expected,words(bytes(result))))
                receipt['cases'].append(dict(samples=count,dt=dt,filter=nf,native_arm=arm,words=count,differing_words=different))
                receipt['words']+=count;receipt['differing_words']+=different
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('cases','commands')},indent=2))
    return int(receipt['differing_words']!=0)
if __name__=='__main__':raise SystemExit(main())
