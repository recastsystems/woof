import importlib.util
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
from woof.certify.kernel_manifest import kernel_manifest
from woof.wrf_exact import ENABLED

root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('mp40_validator',root/'tools/morrison_aero_wrf461_oracle/validate_morrison_aero_oracle.py')
validator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
results={}
failed=False
planted="--plant-one-bit" in sys.argv
for variant,tag in validator.FIXTURES:
    meta,inputs,expected=validator.load_fixture(variant,tag)
    got=validator.run_port(meta,inputs)
    if planted and not results:
        got["th"].view(np.uint32).flat[0] ^= np.uint32(1)
    report=validator.compare(expected,got)
    accepted_reference = expected
    declaration = None
    if variant == "t0fix":
        declaration="Explicit column-only melting-point activation arm; compared with the patched source, not used by forecast defaults."
    acceptance=validator.compare(accepted_reference,got)
    failed |= any(item["differ"] for item in acceptance.values())
    for name,item in report.items():
        if item['differ']:
            item['columns']=np.count_nonzero(expected[name].view(np.uint32)!=got[name].view(np.uint32),axis=(0,1)).tolist() if got[name].ndim==3 else []
    results[variant+'-'+tag]={'meta':meta,'fields':report,'acceptance':acceptance,'declared_divergence':declaration,'full_field_sha256':{name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in got.items()}}
    print(variant+'-'+tag, {k:{n:v[n] for n in ('differ','max_ulp','max_abs')} for k,v in report.items() if v['differ']},flush=True)
manifest=kernel_manifest()
for name,row in manifest.items():
    if ":morrison_aero" in name:
        failed |= ("--ftz=false" not in row["options"] or "--fmad=false" not in row["options"] or row["compiled_image"]["status"]!="resolved")
receipt={'strict':ENABLED,'results':results,'kernel_manifest':kernel_manifest(),'pass':not failed,'planted_one_bit':planted,'default_tolerance':'0 ULP against the declared reference on these 84-column fixtures'}
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(int(failed))
