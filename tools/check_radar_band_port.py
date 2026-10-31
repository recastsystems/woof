"""Compare native PSD fixtures with the original GPU scalar implementation.

Run the CUDA portion under a GPU lease. Reference sources and table archives
are explicit inputs; this is a ten-minute numerical smoke, not weather proof.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-kernel", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    # The engine's compile-policy hook otherwise relaxes division even when
    # a reference kernel explicitly asks for precise division. This process
    # is a scalar-equation reference, not a forecast numerics selection.
    os.environ["GPUWM_NUMERICS"] = "strict"
    import cupy as cp
    import numpy as np
    from woof.core.p3_tables import load_lookup_table_1, p3_table_root
    source = a.reference_kernel.read_text()
    source = source.replace("j<5", "j<7").replace("float acc[5]={0,0,0,0,0}", "float acc[7]={0,0,0,0,0,0,0}")
    source = source.replace("float dry[5]", "float dry[7]").replace("float wet[5]", "float wet[7]")
    source = source.replace("*nd+id+c)*5)", "*nd+id+c)*7)")
    source += r'''
extern "C" __global__ void p3_diagnostic(const float* tab,float q,float n,float rim,float rd,
 float rho,int nd,float lm,float dl,float* out){
 float lo=p3lookup(tab,q,n,rim,rd,6),hi=p3lookup(tab,q,n,rim,rd,7);
 n=fmaxf(fminf(n,lo*q),hi*q);
 float lam=p3lookup(tab,q,n,rim,rd,12),mu=p3lookup(tab,q,n,rim,rd,13);
 IceLaw law=p3law(rim,rd);float integral=0;
 for(int k=0;k<nd;++k){float d=expf(lm+k*dl)*.001f;
  integral+=expf(mu*logf(d)-lam*d)*p3mass(law,d)*d*dl*((k==0||k==nd-1)?.5f:1.f);}
 out[0]=lo;out[1]=hi;out[2]=n;out[3]=lam;out[4]=mu;out[5]=integral;
 out[6]=logf(q*rho/fmaxf(integral,1e-38f));out[7]=law.cg;out[8]=law.d1;out[9]=law.d2;
}
'''
    # The P3 mass normalization reaches subnormal float32 integrals. Fast
    # division flushes those denominators to zero and produces infinity.
    # Use the original equations with IEEE underflow/division for the port
    # reference; retain the same comparison tolerance.
    # CuPy appends -ftz=true after RawModule options. The engine's direct
    # NVRTC loader preserves subnormals and records the actual cache options.
    from woof.core.ieee_module import compile_module
    module = compile_module(source, ("--std=c++17", "--fmad=false",
        "--ftz=false", "--prec-div=true", "--prec-sqrt=true"), "radar_band_reference")
    kernel = module.get_function("polar_integrate")
    p3 = cp.asarray(load_lookup_table_1(p3_table_root())[0])
    p6 = np.pi/6
    rows = {
        8: ((1000*p6,3,0,1,1000),(.069,2,0,1,100),(890*p6,3,0,1,890),(400*p6,3,0,1,400),(900*p6,3,0,1,900)),
        9: ((1000*p6,3,0,1,1000),(.1597,2.078,0,1,100),(500*p6,3,0,1,500),(400*p6,3,0,1,400),(900*p6,3,0,1,900)),
        18: ((1000*p6,3,0,1,1000),(.106214,2,-.4,3,100),(900*p6,3,2,3,900),(500*p6,3,0,1,500),(900*p6,3,1,1,900)),
        50: ((1000*p6,3,0,1,1000),(.069,2,0,1,100),(917*p6,3,0,1,917),(400*p6,3,0,1,400),(900*p6,3,0,1,900)),
    }
    rows[51] = rows[50]
    results = []
    for fixture in json.loads(a.fixtures.read_text()):
        with np.load(a.tables/f"{fixture['band']}.npz", allow_pickle=False) as table:
            arrays = {k: table[k] for k in ("values", "diameter_mm", "density", "liquid")}
        nd, nw, nr = len(arrays["diameter_mm"]), len(arrays["liquid"]), len(arrays["density"])
        f32 = lambda x: cp.asarray(x, dtype="f4")
        q, number, volume = (f32(fixture[k]) for k in ("mass", "number", "volume"))
        rime, rv, rho, temp = (f32([fixture[k]]) for k in ("rime", "rime_volume", "density", "temperature"))
        coefficients = f32(rows[fixture["scheme"]])
        da, wa = f32(arrays["density"]), f32(arrays["liquid"])
        actual = []
        for elevation in range(5):
            values = f32(np.ascontiguousarray(arrays["values"][:,:,:,elevation]))
            output, audit = cp.empty(7,dtype="f4"), cp.empty(2,dtype="f4")
            kernel((1,),(128,),(q,number,volume,rime,rv,rho,temp,coefficients,p3,values,da,wa,output,audit,
                np.int32(1),np.int32(fixture["scheme"]),np.int32(nd),np.int32(nw),np.int32(nr),
                np.float32(np.log(arrays["diameter_mm"][0])),np.float32(np.log(arrays["diameter_mm"][-1]/arrays["diameter_mm"][0])/(nd-1)),
                np.float32(.5),np.int32(1)))
            actual.append(cp.asnumpy(output).tolist())
        reference, native = np.array(actual), np.array(fixture["moments"])
        # Relative error is meaningless in a cancellation-zero covariance.
        tolerance = 2e-3*np.maximum(np.abs(reference),1e-7)
        passed = bool(np.all(np.isfinite(native)) and np.all(np.abs(native-reference)<=tolerance))
        diagnostic = None
        if not passed and fixture["scheme"] >= 50 and fixture["mass"][2] > 0:
            diagnostic_values = cp.empty(10,dtype="f4")
            module.get_function("p3_diagnostic")((1,),(1,),(p3,
                np.float32(fixture["mass"][2]),np.float32(fixture["number"][2]),
                np.float32(fixture["rime"]/fixture["mass"][2]),np.float32(fixture["rime"]/fixture["rime_volume"]),
                np.float32(fixture["density"]),np.int32(nd),np.float32(np.log(arrays["diameter_mm"][0])),
                np.float32(np.log(arrays["diameter_mm"][-1]/arrays["diameter_mm"][0])/(nd-1)),diagnostic_values))
            diagnostic = cp.asnumpy(diagnostic_values).tolist()
        results.append(dict(band=fixture["band"],scheme=fixture["scheme"],label=fixture["label"],
                            passed=passed,max_normalized_error=float(np.max(np.abs(native-reference)/tolerance)),
                            native=native.tolist(),reference=reference.tolist(),p3_diagnostic=diagnostic))
    receipt = dict(passed=all(r["passed"] for r in results),cases=len(results),
                   tolerance="0.2 percent with 1e-7 component floor",results=results)
    a.out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="results"}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
