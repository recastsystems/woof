"""Closed translations of WRF HAILCAST, public domain.
Reuses the repository's checked Fortran expression parser. Unknown statements
fail generation. Every emitted statement names its upstream line.

Two generations (woof.core.hail, ``hailcast_version``):
  wrf_461      WRF v4.6.1 phys/module_diag_hailcast.F (upstream/).
  wrf_39_noaa  The operational HRRR v4 branch, NOAA-EMC/HRRR tag v4.1.21
               (commit 40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827),
               sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_diag_hailcast.F
               (upstream_hrrr_v4121/). It predates Adams-Selin's 2019-2022
               ventilation, heat-budget and melting changes (Reynolds number
               of the melting stone, stone density and surface temperature).
"""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hail_lake_parser", ROOT/"tools/lake_wrf461_oracle/translate.py")
lake = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lake
spec.loader.exec_module(lake)
SOURCE = Path(__file__).parent/"upstream/module_diag_hailcast.F"
PIN = "8fe1d76319ddcb63473752165485ca3cd2e30e62545620d16a2ee0c9f2721f14"
# form -> source, pin, hailstone_driver..MELT line range, the line after which
# the impact-speed bound goes (the ENDIF closing massagr's VIMP fits), the
# generated struct and header.
FORMS = {
    "wrf_461": dict(source=SOURCE, pin=PIN, lines=(472, 1601), vimp_after=1244,
                    struct="HailColumn", header="hailcast_wrf.cuh",
                    title="WRF v4.6.1 HAILCAST",
                    notice="tools/hail_wrf461/upstream/LICENSE.txt"),
    "wrf_39_noaa": dict(source=Path(__file__).parent/"upstream_hrrr_v4121/module_diag_hailcast.F",
                        pin="4c43129a05b9a4686be08682ddf2d6b664e7c22bcf588e70acc58208baa491b1",
                        lines=(498, 1607), vimp_after=1275,
                        struct="HailColumnWrf39", header="hailcast_wrf39.cuh",
                        title="NOAA-EMC/HRRR v4.1.21 (WRFV3.9) HAILCAST",
                        notice="tools/hail_wrf461/upstream/LICENSE.txt"),
}
def statements(form="wrf_461"):
    f = FORMS[form]; first, last = f["lines"]
    carry = ""
    for n, raw in enumerate(f["source"].read_text().splitlines(), 1):
        if not first <= n <= last:
            continue
        line = raw.split("!",1)[0].strip().lower()
        if not line: continue
        if not carry: start=n
        carry += " "+line.lstrip("&").rstrip("&").strip()
        if not line.endswith("&"):
            yield start, re.sub(r"(\.[a-z]+\.)", r" \1 ", carry.strip())
            carry=""
    assert not carry
def declaration(line):
    m=re.match(r"(real\*8|real|integer)\s+(.*)",line)
    if m and "::" not in line:
        line=m[1]+"::"+m[2]
    vars=lake.declaration(line)
    if vars and line.startswith("real*8"):
        for v in vars: v.typ="double"
    return vars
def action(line,symbols,n):
    if line.startswith("goto "): return "goto hail_label_"+line[5:].strip()+";"
    if line.startswith("call "): return lake.expr(line[5:],symbols)+";"
    return lake.action(line,symbols,n)
def generate(form="wrf_461"):
    f = FORMS[form]
    assert hashlib.sha256(f["source"].read_bytes()).hexdigest()==f["pin"], "Upstream changed"
    routines=[]; current=None
    for n,line in statements(form):
        m=re.match(r"subroutine\s+(\w+)\s*\((.*)\)",line)
        if m:
            current=dict(name=m[1],args=lake.split(m[2]),vars={},body=[])
            routines.append(current); continue
        if line.startswith("end subroutine"):
            current=None; continue
        if current is None or line.startswith(("implicit ","character")): continue
        decl=declaration(line)
        if decl is not None:
            for v in decl: current["vars"][v.name]=v
        else: current["body"].append((n,line))
    out=["// Generated from public-domain "+f["title"]+".",
         "// Notice: "+f["notice"],
         "// SHA256: "+f["pin"], "template<int NZ> struct "+f["struct"]+" {"]
    for r in routines:
        symbols=r["vars"]; args=[]
        for name in r["args"]:
            v=symbols[name]
            typ=f"HailArray<{v.typ}>" if v.dims else v.typ
            args.append(("const "+typ+"& " if v.dims else typ+(" " if v.intent=="in" else "& "))+name)
        out.append("__device__ void "+r["name"]+"("+",".join(args)+") {")
        for name,v in symbols.items():
            if name in r["args"] or v.typ=="char": continue
            if v.dims:
                assert len(v.dims)==1
                cap="NZ" if v.dims[0]=="nz" else v.dims[0]
                out.append(f"HailStorage<{v.typ},{cap}> {name};")
            else:
                init=" = "+lake.expr(v.initial,symbols) if v.initial else " = 0"
                out.append(("const " if v.const else "")+v.typ+" "+name+init+";")
        in_switch=False
        for n,line in r["body"]:
            if line.startswith("data "):
                pairs=re.findall(r"(\w+)\s*/([^/]+)/",line[5:])
                assert pairs
                value=" ".join(k+" = "+lake.expr(v,symbols)+";" for k,v in pairs)
            elif line.startswith("select case"):
                value="switch"+line[len("select case"):].strip()+" {"; in_switch=True
            elif line.startswith("case "):
                value=("break; " if "case (1)" not in line else "")+"case "+line.split("(")[1].split(")")[0]+":"
            elif line=="end select":
                value="break; }"; in_switch=False
            elif re.match(r"\d+\s+continue",line):
                value="hail_label_"+line.split()[0]+":;"
            elif line.startswith("call "):
                value=action(line,symbols,n)
            elif line.startswith(("if ","if(")) and "then" not in line:
                start=line.index("("); depth=0
                for end in range(start,len(line)):
                    depth+=(line[end]=="(")-(line[end]==")")
                    if not depth: break
                value="if("+lake.expr(line[start+1:end],symbols)+") { "+action(line[end+1:].strip(),symbols,n)+" }"
            else:
                value=lake.body_line(n,line,symbols,None).replace("; if(error) return;",";")
            # WRF has undefined outputs when embryo interpolation fails, or
            # embryo starts below cloud base. Define these as no surviving stone.
            if r["name"]=="hailstone_driver" and line=="tfzl = tk_embryo":
                out.append("sec=0; d=0;")
            out.append("// WRF:"+str(n)+"\n"+value)
            if r["name"]=="massagr" and n==f["vimp_after"]:
                # The fitted impact-speed polynomials dip below zero near
                # NS=0.1 (e.g. RE=21.4069, NS=0.104819). A speed cannot be
                # negative: otherwise AFACTOR**0.44 makes the density NaN.
                out.append("// Physical domain of the empirical impact-speed fit.\nvimp = lake_max(vimp,0.f);")
            if r["name"]=="hailstone_driver" and line.startswith("call interpp("):
                out.append("if(ifout) { dhails(i)=0; continue; }")
        out.append("}")
    out.append("};")
    return "\n".join(out)+"\n"
if __name__=="__main__":
    for form, f in FORMS.items():
        (ROOT/"woof/core/kernels"/f["header"]).write_bytes(generate(form).encode())
