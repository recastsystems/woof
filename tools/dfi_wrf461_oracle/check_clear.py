#!/usr/bin/env python3
"""Witness native clear's live-radius behavior with nonzero bounded controls.

These controls are synthetic, separate from the real native weather captures.
The complete original clear routine is compiled without arithmetic edits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from check_gpu import original_driver, FIELDS, DYNAMICS, ORDINARY_RADII, GSFC_RADII

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import cupy as cp
    source=a.source.read_bytes()
    pin=json.loads((Path(__file__).parent/'SOURCE.json').read_text())['files']['share/dfi.F']
    assert hashlib.sha256(source).hexdigest()==pin
    text=source.decode()
    anchor='\n   SUBROUTINE dfi_clear_accumulation( grid )\n'
    assert text.count(anchor)==1
    start=text.index(anchor)+1
    end=text.index('   END SUBROUTINE dfi_clear_accumulation',start)
    clear=text[start:end+len('   END SUBROUTINE dfi_clear_accumulation')]
    driver=original_driver(text,a.output)
    program=driver.read_text().replace('program array_driver',clear+'\nprogram array_driver',1)
    program=program.replace(' type domain\n',' type domain\n integer :: sf_surface_physics=2\n',1)
    program=program.replace('module module_state_description\n',
        'module module_state_description\n integer, parameter :: RUCLSMSCHEME=3\n',1)
    # Only fixture setup and invocation change; the extracted operator is
    # retained byte for byte. All saved arrays start nonzero in this control.
    for n in FIELDS:
        program=program.replace(f'grid%{("dfi_"+(n[:-2] if n.endswith("_2") else n))}=0.',
            f'grid%{("dfi_"+(n[:-2] if n.endswith("_2") else n))}=grid%d_{n}')
    marker=' end interface\n allocate(grid)'
    interface=''' subroutine dfi_clear_accumulation(grid)
 use module_domain
 type(domain), pointer :: grid
 end subroutine
'''
    assert program.count(marker)==1
    program=program.replace(marker,interface+marker)
    begin=program.index(' do i=0,59\n')
    end=program.index(' call normalize(grid)\n',begin)+len(' call normalize(grid)\n')
    program=program[:begin]+''' call get_command_argument(6,arg)
 read(arg,*) grid%dfi_savehydmeteors
 grid%hcoeff_tot=9.5
 call dfi_clear_accumulation(grid)
'''+program[end:]
    driver.write_text(program)
    assert clear in program and 'grid%dfi_mu(:,:)' in clear
    command=['gfortran','-cpp','-DEM_CORE=1','-ffree-form','-ffree-line-length-none',
        '-O0','-fno-fast-math','-ffp-contract=off','-fcheck=bounds',str(driver),
        '-o',str(a.output/'clear-native')]
    subprocess.run(command,check=True,cwd=a.output)
    module=cp.RawModule(code=(ROOT/'woof/core/kernels/dfi.cu').read_text(),
        options=('-std=c++11','--fmad=false','--ftz=false'))
    kernel=module.get_function('dfi_clear_fields')
    live={};saved={}
    rng=np.random.default_rng(461)
    with (a.output/'input.bin').open('wb') as f:
        for n in FIELDS:
            count=1031*(2 if n in ('moist','scalar') else 1)
            live[n]=rng.uniform(1,3,count).astype(np.float32)
            saved[n]=rng.uniform(4,6,count).astype(np.float32)
            f.write(live[n].tobytes());f.write(saved[n].tobytes())
    np.ones(60,np.float32).tofile(a.output/'weights.bin')
    receipt=dict(scope='synthetic nonzero native clear controls; no weather or forecast',
        source_sha256=pin,native_build=command,cases=[])
    for mp,save in ((6,0),(7,0),(6,1),(7,1)):
        native=[str(a.output/'clear-native'),str(a.output/'input.bin'),str(a.output/'weights.bin'),
            str(a.output/'saved.bin'),str(a.output/'live.bin'),str(mp),str(save)]
        subprocess.run(native,check=True,cwd=a.output)
        current={n:cp.asarray(x) for n,x in live.items()}
        sums={n:cp.asarray(x) for n,x in saved.items()}
        targets=[sums[n] for n in DYNAMICS]
        if save:
            targets+=[sums['moist'][:1031]]
        else:
            targets+=[sums['moist'],sums['scalar']]
            targets += [current[n] for n in (GSFC_RADII if mp==7 else ORDINARY_RADII)]
        rows=cp.asarray(np.asarray([[0,x.data.ptr,x.size,0] for x in targets],np.uint64))
        kernel(((max(x.size for x in targets)+255)//256,len(targets)),(256,),(rows,))
        expected_saved=np.fromfile(a.output/'saved.bin',np.float32)
        expected_live=np.fromfile(a.output/'live.bin',np.float32)
        actual_saved=np.concatenate([cp.asnumpy(sums[n]) for n in FIELDS]+[np.zeros(1,np.float32)])
        actual_live=np.concatenate([cp.asnumpy(current[n]) for n in FIELDS])
        bad=sum(int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32)))
                for x,y in ((actual_saved,expected_saved),(actual_live,expected_live)))
        receipt['cases'].append(dict(mp=mp,save_hydrometeors=save,
            native_command=native,words=int(actual_saved.size+actual_live.size),differing_words=bad))
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert all(x['differing_words']==0 for x in receipt['cases']),receipt['cases']
    print(json.dumps(dict(words=sum(x['words'] for x in receipt['cases']),differing_words=0)))


if __name__=='__main__':main()
