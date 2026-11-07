"""Refresh implementation identity after independent strict production tests.

Only implementation hashes change. Frozen WRF source, harness and reference
fixture hashes must remain valid. A failed physics test refuses publication.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "woof/data/nssl2/oracle/production-evidence-v1.json"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--junit",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--variant-comparison",type=Path,
                    help="strict current-libm variant receipt, required for coefficient/radar changes")
    a=ap.parse_args()
    j=json.loads(MANIFEST.read_text())
    b=j["oracle_build"]
    inputs={"build.sh":b["build_sh_sha256"],"visibility.patch":b["visibility_patch_sha256"],"nucond_production.F90":b["nucond_harness_sha256"],"radardd02.F90":b["radardd02_harness_sha256"]}
    for name,h in inputs.items():
        assert digest(ROOT/"tools/nssl2_wrf461_production_oracle"/name)==h,name
    for name,h in {"nucond-production.csv":"1b1b472a832a6c9b9968e218d43a61b06c924880254ae0ea049b67e74eff4911","radardd02.csv":"37b4173833d9b1213da323ccdff1edf7cf5f79ba8b8f1eff62fc8ac91d9e65b7"}.items():
        assert digest(MANIFEST.parent/name)==h,name
    cases=list(ET.parse(a.junit).getroot().iter("testcase"))
    assert len(cases)==14,len(cases)
    for c in cases:
        assert c.find("error") is None and c.find("skipped") is None,c.attrib
        failure=c.find("failure")
        assert failure is None or c.attrib["name"]=="test_production_evidence_manifest_pins_every_tracked_input",c.attrib
    before=dict(j["implementation"])
    after={rel:digest(ROOT/rel) for rel in before}
    changed={rel:{"before":h,"after":after[rel]} for rel,h in before.items() if h!=after[rel]}
    allowed={"woof/core/nssl2_nucond.py"}
    variant_receipt=None
    if a.variant_comparison is not None:
        comparison=json.loads(a.variant_comparison.read_text())
        row=comparison["cccn-low"]
        assert row["arm"]=="strict" and row["passed"] and row["max_ulp"]==0
        assert sum(f["cells"] for f in row["fields"].values())==10614
        assert all(f["cells_differing"]==0 and f["nonfinite_cells"]==0
                   for f in row["fields"].values())
        allowed.update({"woof/core/kernels/nssl2_nucond.cu",
                        "woof/core/nssl2_diagnostics.py",
                        "woof/core/kernels/nssl2_diagnostics.cu"})
        variant_receipt={"sha256":digest(a.variant_comparison),
                         "words":10614,"differing_words":0,
                         "native_glibc":"2.43"}
    assert set(changed) <= allowed,changed
    old=digest(MANIFEST)
    revision=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    j["implementation"]=after
    j["exact_chain_mode_contract"]={"strict":"incoming WRF math and frozen-water arguments","default":"base cd336e62d6 CUDA source and original call arity","physics_tests_passed":13,"source_revision":revision,"junit_sha256":digest(a.junit),"refresh_owner":"tools/nssl2_wrf461_production_oracle/refresh_implementation_evidence.py","historical_validation":"Earlier validation and default-on notes above describe their recorded source revisions."}
    if variant_receipt is not None:
        j["exact_chain_mode_contract"]["cccn_current_libm"]=variant_receipt
    MANIFEST.write_text(json.dumps(j,indent=2)+"\n")
    receipt={"before_sha256":old,"after_sha256":digest(MANIFEST),"changed_implementation":changed,"frozen_oracles_unchanged":True,"source_revision":revision,"junit":str(a.junit),"junit_sha256":digest(a.junit)}
    a.out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))
if __name__=="__main__":main()
