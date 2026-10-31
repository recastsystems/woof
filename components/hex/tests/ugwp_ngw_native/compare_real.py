"""Replay real native-prepared UGWP columns; Rust owns capture/NetCDF decoding.

This debug receipt complements f1's eventual inner-slot capture and real door
acceptance. Orographic additive wind baselines cannot be inferred from the
combined parent capture, so native standalone uses explicit baselines. NGW
temperature and wind diagnostic rates are compared with the actual parent post.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


class Transport:
    def __init__(self, path):
        self.lib = C.CDLL(str(path.resolve()))
        self.handles = []
        for name, signature in {
            'bridge_oracle': [C.c_char_p, C.c_void_p, C.c_size_t],
            'bridge_netcdf': [C.c_char_p, C.c_char_p, C.c_void_p, C.c_size_t],
            'bridge_sample': [C.c_void_p, C.c_size_t],
            'bridge_levels': [C.c_void_p, C.c_size_t],
            'bridge_unmarked_cells': [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t],
        }.items():
            fn = getattr(self.lib, name); fn.argtypes = signature; fn.restype = C.c_void_p
        self.lib.bridge_error.restype = C.c_char_p
        self.lib.bridge_data.argtypes = [C.c_void_p]; self.lib.bridge_data.restype = C.POINTER(C.c_uint32)
        self.lib.bridge_ids.argtypes = [C.c_void_p]; self.lib.bridge_ids.restype = C.POINTER(C.c_int32)
        self.lib.bridge_meta.argtypes = [C.c_void_p, C.c_size_t]; self.lib.bridge_meta.restype = C.c_size_t
        self.lib.bridge_free.argtypes = [C.c_void_p]

    def own(self, handle):
        if not handle:
            raise ValueError(self.lib.bridge_error().decode())
        self.handles.append(handle)
        return handle

    def view(self, handle):
        meta = lambda i: self.lib.bridge_meta(handle, i)
        shape = tuple(meta(4+i) for i in range(meta(1)))[::-1]
        while len(shape) > 1 and shape[0] == 1:
            shape = shape[1:]  # native synthetic j dimension, preserve singleton cell axis
        dtype = 'f4' if meta(3) == 1 else 'i4'
        # Memory views over decoded Rust-owned words; no Python file processing.
        return np.ctypeslib.as_array(self.lib.bridge_data(handle), shape=(meta(0),)).view(dtype).reshape(shape)

    def ids(self, handle):
        return np.ctypeslib.as_array(self.lib.bridge_ids(handle), shape=(self.lib.bridge_meta(handle, 2),))

    def oracle(self, path, ids=None):
        handle = self.own(self.lib.bridge_oracle(str(path).encode(), ptr(ids) if ids is not None else None, len(ids) if ids is not None else 0))
        return handle, self.view(handle)

    def netcdf(self, path, variable, ids=None):
        handle = self.own(self.lib.bridge_netcdf(str(path).encode(), variable.encode(), ptr(ids) if ids is not None else None, len(ids) if ids is not None else 0))
        return self.view(handle)

    def close(self):
        for handle in self.handles: self.lib.bridge_free(handle)
        self.handles.clear()


def ptr(array):
    return array.ctypes.data_as(C.c_void_p)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dump', 'initial', 'climate', 'work', 'transport', 'receipt', 'native_manifest'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--columns', type=int, default=24)
    parser.add_argument('--amplitude', type=float, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--orographic-baseline',type=Path)
    parser.add_argument('--baseline-receipt',type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
    from woof.hex.mpasphys import ugwp_ngw_v842 as port
    transport = Transport(args.transport)
    records = []
    captures = sorted(args.dump.glob('gwdo_ugwp*.pre.u3d.bin'))
    if not captures: raise ValueError('native never called the UGWP parent; no replay permitted')
    native = C.CDLL(str((args.work/'native/liboracle.so').resolve())).oracle
    native.restype = None
    native.argtypes = [C.c_int,C.c_int,C.c_float,C.c_float,C.c_float]+[C.c_void_p]*6+[C.c_int,C.c_int]+[C.c_void_p]*24
    host = C.CDLL(str((args.work/'libhost.so').resolve())).host_run
    host.restype = None
    host.argtypes = [C.c_int]*3+[C.c_float]*3+[C.c_void_p]*29
    names = ['rublten','rvblten','rthblten','dudt_ngw','dvdt_ngw','dtdt_ngw','dked','zngw','initial_state','latitude_indices','latitude_weights']
    try:
        for capture in captures:
            prefix = capture.name.removesuffix('.pre.u3d.bin')
            reference, _ = transport.oracle(capture)
            sample = transport.own(transport.lib.bridge_sample(reference, args.columns))
            ids = transport.ids(sample)
            hashes = {}
            def field(name, phase='pre', unmarked=False, levels=None):
                path = args.dump/(prefix+'.'+phase+'.'+name+'.bin')
                hashes[path.name] = sha(path)
                handle, value = transport.oracle(path, None if unmarked else ids)
                if unmarked:
                    handle = transport.own(transport.lib.bridge_unmarked_cells(handle, reference, ptr(ids), len(ids)))
                    value = transport.view(handle)
                if levels is not None:
                    handle = transport.own(transport.lib.bridge_levels(handle, levels))
                    value = transport.view(handle)
                return value
            dtp, day = float(field('dt')), float(field('r_doy'))
            if not bool(field('ngw_scheme')) or not bool(field('ugwp_diags')):
                raise ValueError('NGW/diagnostics must be active in the captured native call')
            rdzw = transport.netcdf(args.initial, 'rdzw')
            nl = len(rdzw)
            dzu = transport.netcdf(args.initial, 'dzu')
            lat = transport.netcdf(args.initial, 'latCell', ids)
            taulat = transport.netcdf(args.climate, 'LATS')
            days, tau = field('days_limb'), field('tau_limb')
            input_names = ('u3d','v3d','t3d','qv3d','p3d','pi3d','p3di','z','zi','raincv','rainncv')
            fields = [field(name, levels=nl+1 if name in ('p3di','zi') else nl if i<9 else None) for i,name in enumerate(input_names)]
            nc = len(ids)
            inputs = [rdzw,dzu,lat,taulat,days,tau]+fields
            baselines=[]
            if args.orographic_baseline:
                if not args.baseline_receipt: raise ValueError('native parent baseline requires its provenance receipt')
                for name in names[:3]:
                    path=args.orographic_baseline/(prefix+'.post.'+name+'.bin')
                    hashes['orographic_baseline/'+path.name]=sha(path)
                    handle,_=transport.oracle(path,ids)
                    handle=transport.own(transport.lib.bridge_levels(handle,nl))
                    baselines.append(transport.view(handle))
            def outputs():
                tendency = [a.copy() for a in baselines] if baselines else [np.full((nl,nc),np.float32(.00013*(i+1)),dtype='f4') for i in range(3)]
                return tendency+[np.zeros((nl,nc),dtype='f4') for _ in range(4)]+[np.zeros(nc,dtype='f4'),np.zeros(4*58+2*(nl+1),dtype='f4'),np.zeros((2,nc),dtype='i4'),np.zeros((2,nc),dtype='f4')]
            n, h = outputs(), outputs()
            status = np.zeros(nc, dtype='i4')
            native(nl,nc,dtp,args.amplitude,day,*[ptr(a) for a in inputs[:6]],len(taulat),len(days),*[ptr(a) for a in inputs[6:]],*[ptr(a) for a in n[:9]],ptr(n[9][0]),ptr(n[9][1]),ptr(n[10][0]),ptr(n[10][1]))
            if args.gpu:
                import cupy as cp
                columns = dict(zip(('um','vm','tm','qm','prsl','prslk','prsi','zmet','zmeti','raincv','rainncv'),map(cp.asarray,fields)))
                columns['lat_r'] = cp.asarray(lat)
                for name, array in zip(names[:3],outputs()[:3]): columns[name] = cp.asarray(array)
                climate = dict(zip(('rdzw','dzu','LATS','DAYS','ABSMF'),map(cp.asarray,inputs[:2]+inputs[3:6])))
                result = port.run(columns,climate,dtp=dtp,amplitude=args.amplitude,day=day)
                h = [cp.asnumpy(getattr(result,name)) for name in names]
                status = cp.asnumpy(result.status)
            else:
                host(nc,len(taulat),len(days),dtp,args.amplitude,day,*[ptr(a) for a in inputs],*[ptr(a) for a in h],ptr(status))
            comparisons = {}
            def compare(name, a, b):
                if a.shape != b.shape: raise ValueError((name, a.shape, b.shape))
                diff = a.view('u4') != b.view('u4')
                comparisons[name] = dict(words=int(a.size), differing_words=int(diff.sum()), native_sha256=hashlib.sha256(a.tobytes()).hexdigest(),candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest())
                if diff.any():
                    print(prefix, name, int(diff.sum()), [(index.tolist(),a[tuple(index)].item(),b[tuple(index)].item()) for index in np.argwhere(diff)[:3]])
            for name, a, b in zip(names,n,h): compare(name,a,b)
            if baselines:
                for name,a in zip(names[:3],n[:3]): compare('actual_parent_post_'+name,field(name,'post',levels=nl),a)
            for name, a in zip(names[3:6],n[3:6]): compare('actual_parent_post_'+name,field(name,'post',unmarked=True,levels=nl),a)
            for i, name in enumerate(('jindx1_tau','jindx2_tau')): compare('actual_parent_pre_'+name,field(name),n[9][i])
            for i, name in enumerate(('ddy_j1tau','ddy_j2tau')): compare('actual_parent_pre_'+name,field(name),n[10][i])
            records.append(dict(capture=prefix,global_cell_ids=ids.tolist(),dtp=dtp,day=day,amplitude=args.amplitude,status=status.tolist(),input_sha256=hashes,arrays=comparisons))
    finally:
        transport.close()
    provenance = dict(port_sha256=sha(port.__file__),native_binary_sha256=sha(args.work/'native/liboracle.so'),host_witness_sha256=sha(args.work/'libhost.so'),transport_source_sha256=sha(Path(__file__).with_name('oracle_bridge.rs')),transport_binary_sha256=sha(args.transport),replay_script_sha256=sha(__file__),initial_sha256=sha(args.initial),climatology_sha256=sha(args.climate),native_forecast_manifest=json.loads(args.native_manifest.read_text()))
    if args.gpu:
        provenance.update(shared_math_sha256=hashlib.sha256(port.shared_math_header().encode()).hexdigest(),cupy_version=cp.__version__,runtime_version=cp.cuda.runtime.runtimeGetVersion(),driver_version=cp.cuda.runtime.driverGetVersion(),compiler_options=['--std=c++17','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'])
    if args.baseline_receipt: provenance['native_orographic_baseline'] = json.loads(args.baseline_receipt.read_text())
    receipt = dict(schema='p15.ugwp-ngw-real-debug/v1',acceptance=False,backend='hex-cuda' if args.gpu else 'host-glibc witness',provenance=provenance,cases=records,additive_baseline='native parent with embedded NGW false' if args.orographic_baseline else 'explicit standalone baselines; actual parent includes prior orographic contribution')
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    count = sum(v['differing_words'] for r in records for v in r['arrays'].values())
    print('real columns',sum(len(r['global_cell_ids']) for r in records),'differing words',count)
    assert count == 0
    assert all(code == 0 for r in records for code in r['status'])


if __name__ == '__main__': main()
