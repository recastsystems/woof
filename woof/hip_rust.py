"""Source-bound HIP compilation and the narrow Rust driver adapter.

This process boundary keeps the single portable translator in Python. Rust
embeds its canonical kernel source, and consumes the same bound AOT cache as
CuPy. No generated kernel fork or external compiler binary is packaged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def driver():
    from woof.hip_backend import toolchain_identity,compiler_dependencies,nvidia_banner
    hipcc,banner=toolchain_identity()
    nvidia=os.environ.get('WOOF_BACKEND')=='hip-nvidia'
    compiler=os.environ.get('HIPCC_NVIDIA',hipcc) if nvidia else hipcc
    source=Path(__file__).with_name('hip_driver.cpp')
    cuda=Path(os.environ.get('WOOF_HIP_NVIDIA_CUDA_PATH',os.environ.get('CUDA_PATH','/usr/local/cuda')))
    identity=dict(source=hashlib.sha256(source.read_bytes()).hexdigest(),compiler=banner,
                  dependencies=compiler_dependencies(hipcc),nvidia=nvidia,
                  nvidia_compiler=nvidia_banner(compiler) if nvidia else None,
                  nvidia_cuda_path=str(cuda.resolve()) if nvidia else None)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    root=Path(os.environ.get('CUPY_CACHE_DIR',str(Path.home()/'.cache/cupy')))/'gpuwm-hip'
    root.mkdir(parents=True,exist_ok=True)
    image=root/('rust-driver-'+key+'.so');receipt=image.with_suffix('.json')
    if image.exists() and receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('identity')==identity and saved.get('sha256')==hashlib.sha256(image.read_bytes()).hexdigest():return image
    with tempfile.TemporaryDirectory(prefix='gpuwm-hip-driver-',dir=root) as tmp:
        target=Path(tmp)/'driver.so'
        options=['-shared','-O2','-std=c++17']
        options+=['-Xcompiler=-fPIC'] if nvidia else ['-fPIC','--offload-arch=gfx942']
        command=[compiler,*options,str(source),'-o',str(target)]
        if nvidia:
            command+=['-L'+str(cuda/'lib'),'-L'+str(cuda/'lib/stubs'),'-L'+str(cuda/'lib64/stubs'),'-lcuda']
        env=dict(os.environ,HIP_PLATFORM='nvidia' if nvidia else 'amd')
        result=subprocess.run(command,env=env,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('HIP Rust driver build failed: '+result.stdout+result.stderr)
        data=target.read_bytes();record=dict(identity=identity,command=command,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        os.replace(target,image)
        receipt.write_text(json.dumps(record,indent=2)+'\n')
    return image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--driver',action='store_true');a=p.parse_args()
    if a.driver:print(driver().resolve());return 0
    request=json.load(sys.stdin)
    import cupy as cp
    from woof.hip_backend import architecture,compile_aot
    from woof.numerics import effective_options,mode
    # Rust postprocessing has its own pinned arithmetic in fast forecasts too.
    options=tuple(request.get('options',()))
    if mode()=='strict':options=effective_options(options)
    image,receipt=compile_aot(request['source'],options,architecture(cp))
    sys.stdout.buffer.write(image)
    return 0

if __name__=='__main__':raise SystemExit(main())
