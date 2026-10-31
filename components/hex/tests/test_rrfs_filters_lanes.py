"""Real-forecast per-kernel A/B, invoked as the recipe's LAUNCHER.

FILTER_IMAGES points at CPU-built base/tip cubins and their source/option hashes.
FILTER_AB_OUT enables replay at actual outer steps 1, 12, 24 (plus hourly output
step 5 and following radiation step 6). Without it this is a plain whole run.
Every pointer argument is copied, restored between arms, and compared as words,
including untouched boundary rows. Baseline grids use 7876465's owner counts.
Timing excludes restores and uses nine event samples from the same saved state.
Vertical mixing defaults off on this deck: exercise both full/perturbation and
momentum/theta branches on live saved fields in private shadow tendencies.
"""
from __future__ import annotations
import hashlib
from functools import wraps
import json
import os
from pathlib import Path
import statistics
import sys

# Kernel: (baseline owner argument, output argument indices).
HORIZONTAL = {
    'laplacian_divergence_v841_f32': (7, (10,)),
    'laplacian_vorticity_v841_f32': (7, (9,)),
    'momentum_filter_seed_v841_f32': (11, (13, 14)),
    'momentum_filter_lap4_v841_f32': (12, (14,)),
    'theta_filter_lap2_v841_f32': (11, (14, 15)),
    'theta_filter_lap4_v841_f32': (10, (12,)),
    'w_filter_lap2_v841_f32': (11, (14, 15)),
    'w_filter_lap4_v841_f32': (10, (12,)),
    'pressure_gradient_v841_f32': (9, (10,)),
}
VERTICAL = {'vertical_u_rrfs': (3, (14,)), 'vertical_cells_rrfs': (2, (13, 14))}
STEPS = {1, 5, 6, 12, 24}


def install_images():
    """Serve exact source+option CPU builds through the normal compile observer."""
    from cupy.cuda import compiler
    folder = Path(os.environ['FILTER_IMAGES'])
    rows = json.loads((folder/'manifest.json').read_text())
    lookup = {(r['source_sha256'], tuple(r['options'])): r for r in rows}
    original = compiler._compile_using_nvrtc_no_warning
    def compiled(source, options=(), *args, **kwargs):
        row = lookup.get((hashlib.sha256(source.encode()).hexdigest(), tuple(options)))
        if row is None:
            return original(source, options, *args, **kwargs)
        blob = (folder/row['image']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == row['image_sha256']
        return blob, None
    compiler._compile_using_nvrtc_no_warning = compiled
    return folder, rows


class Replay:
    def __init__(self, folder, rows):
        import cupy as cp
        from cupy.cuda import function
        from woof.hex import numerics
        self.cp = cp
        self.step = 0
        self.driver = None
        self.shadow = False
        self.counts = {}
        self.modules = {}
        self.rows = rows
        self.out = Path(os.environ['FILTER_AB_OUT'])
        for row in rows:
            if row['mode'] == numerics.mode():
                m = function.Module(); m.load((folder/row['image']).read_bytes())
                self.modules[row['arm'], row['unit']] = m
        self.write(dict(kind='identity', numerics=numerics.mode(), sources=[r for r in rows if r['mode']==numerics.mode()]))
        import subprocess
        p = subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'], capture_output=True,text=True,timeout=10)
        self.write(dict(kind='neighbours', output=p.stdout, error=p.stderr))

    def write(self, row):
        with self.out.open('a') as f:f.write(json.dumps(row)+'\n')

    def compare(self, name, args, grid, block, unit, live):
        cp = self.cp
        spec = (VERTICAL if self.shadow else HORIZONTAL)[name]
        key = (self.step, name, self.shadow)
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.step not in STEPS:
            return live()
        # Preserve aliases exactly, reject partial overlaps rather than silently
        # converting them into independent arrays in the proof.
        arrays = {}
        for a in args:
            if isinstance(a, cp.ndarray):
                assert a.flags.c_contiguous
                arrays.setdefault((a.data.ptr,a.nbytes),a)
        spans = sorted(arrays)
        assert all(a[0]+a[1] <= b[0] for a,b in zip(spans,spans[1:]))
        saved = {key:a.copy() for key,a in arrays.items()}
        def restore():
            for key,a in arrays.items():a[...] = saved[key]
        old = self.modules['base',unit].get_function(name)
        new = self.modules['tip',unit].get_function(name)
        oldgrid = ((int(args[spec[0]])+127)//128,)
        old(oldgrid, block, args)
        previous = {key:a.copy() for key,a in arrays.items()}
        restore()
        live()
        different = words = output_words = 0
        for key,a in arrays.items():
            different += int(cp.count_nonzero(a.view(cp.uint32) != previous[key].view(cp.uint32)).get())
            words += a.nbytes//4
        output_words = sum(args[i].nbytes//4 for i in spec[1])
        row = dict(kind='kernel', step=self.step, name=name, module=unit,
                   call=self.counts[(self.step,name,self.shadow)],
                   shadow=self.shadow, words=words, output_words=output_words,
                   different=different, baseline_grid=oldgrid, tip_grid=grid)
        # Time one representative call per kernel and selected step, all nine
        # samples restored. No host work between event start, kernel, stop.
        if self.counts[(self.step,name,self.shadow)] == 1:
            times={}
            for arm,kernel,g in [('base',old,oldgrid),('tip',new,grid)]:
                samples=[]
                for _ in range(9):
                    restore()
                    start,end=cp.cuda.Event(),cp.cuda.Event()
                    start.record();kernel(g,block,args);end.record();end.synchronize()
                    samples.append(float(cp.cuda.get_elapsed_time(start,end)))
                times[arm]=dict(median_ms=statistics.median(samples),samples_ms=samples)
            row['timing']=times
            restore();live()
        self.write(row)
        if different:raise AssertionError(f'{name}: {different}/{words} words differ at step {self.step}')

    def install(self):
        from woof.hex.cuda_driver_rrfs import CudaRRFSDycoreDriver
        from woof.hex.cuda_horizontal_rrfs import CudaHorizontalRRFS
        from woof.hex import cuda_vertical_mixing_rrfs as vm
        step = CudaRRFSDycoreDriver._step_device_v841
        @wraps(step)
        def stepped(driver,*args,**kw):
            self.step += 1; self.driver = driver
            return step(driver,*args,**kw)
        CudaRRFSDycoreDriver._step_device_v841 = stepped
        launch = CudaHorizontalRRFS._launch_v841
        def horizontal(obj,name,total,args):
            if name not in HORIZONTAL:return launch(obj,name,total,args)
            self.compare(name,args,((total+127)//128,),(128,),'cuda_horizontal_rrfs',lambda:launch(obj,name,total,args))
        CudaHorizontalRRFS._launch_v841 = horizontal
        kernel = vm.kernel
        def vertical(source,module,name,cache=None):
            target=kernel(source,module,name,cache)
            def call(grid,block,args):
                return self.compare(name,args,grid,block,'cuda_vertical_mixing_rrfs',lambda:target(grid,block,args))
            return call
        vm.kernel = vertical
        mixing = CudaHorizontalRRFS.compute_dry_mixing_tendencies_v841
        seen=set()
        def mixed(obj,*args,**kw):
            result=mixing(obj,*args,**kw)
            if self.step not in STEPS or self.step in seen:return result
            seen.add(self.step)
            # Saved rho belongs to this exact dynamics substep, not an inferred
            # reconstruction from the current driver's public state.
            saved_state=sys._getframe(1).f_locals['saved_state']
            driver=self.driver
            from dataclasses import replace
            self.shadow=True
            try:
                for full in (False,True):
                    for momentum,theta in ((0.1,0.0),(0.0,0.1),(0.1,0.1)):
                        cfg=replace(driver.config,config_mix_full=full,config_v_mom_eddy_visc2=momentum,config_v_theta_eddy_visc2=theta)
                        self.write(dict(kind='vertical_shadow',step=self.step,full=full,vm=momentum,vt=theta,cell_solve=obj.ncells-1,edge_solve=obj.nedges-1))
                        vm.apply_vertical_mixing_cuda_rrfs(obj.mesh,driver.v841_context,cfg,
                            zgrid=driver.atmosphere.vertical.zgrid,normal=args[0],vertical=args[2],theta=args[3],
                            rho=saved_state.rho,rho_edge=args[4],rdzw=driver.atmosphere.vertical.rdzw,
                            rdzu=driver.atmosphere.vertical.rdzu,theta_init=driver._rrfs_theta_init,
                            edge_cos=driver._rrfs_edge_cos,edge_sin=driver._rrfs_edge_sin,
                            tend_u=result.tend_u_euler.copy(),tend_w=result.tend_w_euler.copy(),
                            tend_theta=result.tend_theta_euler.copy(),cell_solve_count=obj.ncells-1,
                            edge_solve_count=obj.nedges-1,kernel_cache=obj.kernel_cache)
            finally:self.shadow=False
            return result
        CudaHorizontalRRFS.compute_dry_mixing_tendencies_v841 = mixed

    def finish(self):
        required={(s,n) for s in STEPS for n in (*HORIZONTAL,*VERTICAL)}
        actual={(s,n) for (s,n,shadow),count in self.counts.items() if count}
        missing=sorted(required-actual)
        self.write(dict(kind='coverage',missing=missing,counts=[dict(step=s,name=n,shadow=h,calls=c) for (s,n,h),c in self.counts.items()]))
        assert not missing, missing


def test_kernel_output_contracts():
    """The proof must compare all writable pointers, not just selected fields."""
    import re
    from woof.hex.cuda_horizontal_rrfs import _CUDA_SOURCE
    from woof.hex.cuda_vertical_mixing_rrfs import _SOURCE
    for source,table in [(_CUDA_SOURCE,HORIZONTAL),(_SOURCE,VERTICAL)]:
        for name,(_,outputs) in table.items():
            signature=re.search(r'void\s+'+name+r'\s*\((.*?)\)\s*\{',source,re.S)[1]
            params=signature.split(',')
            writable={i for i,p in enumerate(params) if '*' in p and 'const' not in p}
            assert writable==set(outputs),(name,writable,outputs)


if __name__ == '__main__':
    from woof.hex import cli
    args=cli.build_parser().parse_args(sys.argv[1:]);cli._select_numerics(args)
    folder,rows=install_images()
    replay=Replay(folder,rows) if os.environ.get('FILTER_AB_OUT') else None
    if replay:replay.install()
    rc=int(args.handler(args))
    if replay:replay.finish()
    sys.exit(rc)
