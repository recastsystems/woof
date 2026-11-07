"""Full-volume WDM6 byte audit for the real CLI forecast verification."""
import os
if os.environ.get("WOOF_WDM6_IDENTITY_RECEIPT"):
    import hashlib
    import json
    from pathlib import Path
    import cupy as cp
    from woof.wrf_exact import ENABLED, install
    install()
    import woof.core.wdm6 as adapter
    from woof.core.kernels import _preamble, module_options
    ROOT=Path(__file__).resolve().parents[3]
    receipt=Path(os.environ["WOOF_WDM6_IDENTITY_RECEIPT"])
    before=(ROOT/"tests/data/wdm6_before_family.cu").read_text()
    original=adapter.launch_wdm6
    names=("theta","qv","qc","qr","qi","qs","qg","nn","nc","nr","rho",
           "pii","pressure","dz","xland","rainnc","rainncv","snownc",
           "snowncv","graupelnc","graupelncv","sr","dt")
    modules={}
    calls=0
    def audit(*args,**kwargs):
        global calls
        nz=args[0].shape[0];tier=64 if nz<=64 else 80
        key=(tier,int(cp.cuda.Device().id))
        if key not in modules:
            modules[key]=cp.RawModule(code=f"#define WDM6_KMAX {tier}\n"+_preamble()+before,
                                      options=module_options("wdm6"))
        module=modules[key]
        twin=tuple(a.copy() if isinstance(a,cp.ndarray) else a for a in args)
        twin_kw={n:a.copy() if isinstance(a,cp.ndarray) else a for n,a in kwargs.items()}
        get,defined=adapter.get_kernel,adapter.get_kernel_int_defines
        try:
            adapter.get_kernel=lambda unit,symbol:module.get_function(symbol)
            adapter.get_kernel_int_defines=lambda unit,symbol,defines:module.get_function(symbol)
            original(*twin,**twin_kw)
        finally:
            adapter.get_kernel=get;adapter.get_kernel_int_defines=defined
        result=original(*args,**kwargs)
        fields={}
        pairs=[(n,a,b) for n,a,b in zip(names,args,twin) if isinstance(a,cp.ndarray)]
        pairs += [(n,a,twin_kw[n]) for n,a in kwargs.items() if isinstance(a,cp.ndarray)]
        for n,a,b in pairs:
            aa=cp.asnumpy(a).tobytes();bb=cp.asnumpy(b).tobytes()
            ha,hb=hashlib.sha256(aa).hexdigest(),hashlib.sha256(bb).hexdigest()
            fields[n]={"current_sha256":ha,"previous_sha256":hb,"bytes":len(aa)}
            if aa!=bb:
                with receipt.open("a") as f:f.write(json.dumps(dict(call=calls,field=n,match=False,hashes=fields[n]))+"\n")
                raise AssertionError(f"WDM6 full-volume byte mismatch at call {calls}, field {n}: {ha} != {hb}")
        with receipt.open("a") as f:f.write(json.dumps(dict(call=calls,strict=bool(ENABLED),shape=list(args[0].shape),
                                                     match=True,fields=fields))+"\n")
        calls+=1
        return result
    adapter.launch_wdm6=audit
