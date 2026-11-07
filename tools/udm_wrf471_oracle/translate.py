"""Lower the pinned UDM statement programs, keeping native kinds and axes.

This is generation tooling, not a forecast data path. Initialized module
values come from the kept native forecast. Unknown syntax fails closed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "udm_common_translation", ROOT / "tools/thompson_generation_oracle/translate.py")
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)
_base_expr = t.expr
_function = None
_routines = {}

ROUTINES = (
    "udm2d", "slope_rain", "slope_graupel", "slope_hail", "slope_cloud",
    "slope_snow", "semi_lagrangian", "cldf_diag", "adjust_number_concent",
    "find_cloud_top",
    "fshape", "funct_aut_qc_tot", "funct_aut_qc_sub",
    "funct_aut_nc_tot", "funct_aut_nc_sub", "fsvp", "fsvp_water",
    "udm_mp_reflectivity", "udm_mp_effective_radius")


def radar_declarations(source):
    result = {}
    for n, line in t.statements(source):
        if line == "contains":
            break
        if line.startswith(("public", "private", "use ", "module ")):
            continue
        if line.startswith("character"):
            if "::" in line:
                for name in t.lake.split(line.split("::",1)[1]):
                    result[name.strip()] = t.lake.Variable(name.strip(),"int",[],initial="0",const=True)
            continue
        if line.startswith("complex*16"):
            for name in t.lake.split(line.split("::",1)[1]):
                result[name.strip()] = t.lake.Variable(name.strip(),"cd",[])
            continue
        decl = t.declaration(line)
        if decl:
            result.update((v.name,v) for v in decl)
    return result


def parse(source, radar_source=None):
    module, routines, current, module_zone = {}, {}, None, True
    for n, line in t.statements(source):
        if line == "contains":
            module_zone = False
        start = re.fullmatch(
            r"(?:(real|double precision)\s+)?(subroutine|function)\s+(\w+)\s*\((.*)\)", line)
        if start:
            current = dict(name=start[3], ret=("double" if start[1] == "double precision"
                else "float") if start[2] == "function" else "void",
                args=t.lake.split(start[4]), vars={}, body=[])
            routines[current["name"]] = current
            continue
        if line.startswith(("end subroutine", "end function")):
            current = None
            continue
        if line.startswith(("module ", "end module", "contains", "implicit ", "use ")):
            continue
        declarations = t.declaration(line)
        if declarations is not None:
            if current is None and not module_zone:
                continue
            for v in declarations:
                # module_gfs_machine defines kind_phys independently of
                # WRF's default REAL. The source witness must travel with
                # any generated result using this declaration.
                if "kind=kind_phys" in line.replace(" ", ""):
                    v.typ = "double"
                (module if current is None else current["vars"])[v.name] = v
        elif current is not None:
            current["body"].append((n, line))
    for name, (typ, dims, initial) in t.RADAR_FIELDS.items():
        module[name] = t.lake.Variable(name, typ, dims, initial=initial, const=bool(initial))
    for name in t.RADAR_STRINGS:
        module[name] = t.lake.Variable(name, "int", [], initial="0", const=True)
    # UDM's hail radar uses the same initialized quadrature axes as the
    # common radar module, with a third hail diameter axis.
    for name in ("xxdh", "xdth"):
        module[name] = t.lake.Variable(name, "double", ["50"])
    if radar_source:
        module.update(radar_declarations(radar_source))
    return module, {name: routines[name] for name in ROUTINES}


def calls(text, name):
    """Yield balanced argument spans, outside quoted strings."""
    pattern = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    pos = 0
    while match := pattern.search(text, pos):
        begin, end, depth = match.end(), match.end(), 1
        while depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        yield match.start(), end, t.lake.split(text[begin:end-1])
        pos = end


def expression(source, symbols, slices=None):
    source = source.strip()
    if _function:
        source = re.sub(r"\b"+_function+r"\b(?!\s*\()", "result", source)
    for intrinsic in ("min", "max"):
        for begin, end, args in reversed(list(calls(source,intrinsic))):
            if len(args)>3:
                value=args[0]
                for arg in args[1:]:
                    value=intrinsic+"("+value+","+arg+")"
                source=source[:begin]+value+source[end:]
    substitutions = {}
    def save(value):
        name = "udmplaceholder" + str(len(substitutions))
        substitutions[name] = value
        return name
    # Native MAXVAL traverses the complete vertical section. Its argument
    # here contains only elementwise products and division.
    for begin, end, args in reversed(list(calls(source, "maxval"))):
        assert len(args) == 1
        value = expression(args[0], symbols, ["udmreducek"])
        reduced = ("([&]() { float udmmax=-__int_as_float(0x7f800000); "
            "for(int udmreducek=1;udmreducek<=nz;++udmreducek) "
            f"udmmax=tc_max(udmmax,{value}); return udmmax; }}())")
        source = source[:begin] + save(reduced) + source[end:]
    for name, v in sorted(symbols.items(), key=lambda item: -len(item[0])):
        if not v.dims:
            continue
        for begin, end, args in reversed(list(calls(source, name))):
            if not any(":" in a for a in args):
                continue
            assert len(args) == len(v.dims), (name, args, v.dims)
            indices, axis = [], 0
            for arg, dim in zip(args, v.dims):
                if ":" in arg:
                    lo, hi = t.lake.bounds(dim)
                    lo = arg.split(":")[0].strip() or lo
                    indices.append(slices[axis] if slices else lo)
                    axis += 1
                else:
                    indices.append(arg)
            value = _base_expr(name + "(" + ",".join(indices) + ")", symbols)
            # UDM passes contiguous vertical sections. The generated
            # column has singleton horizontal axes. No strided view is
            # silently treated as contiguous.
            if not slices:
                colon_axes = [i for i, a in enumerate(args) if ":" in a]
                assert colon_axes == list(range(len(colon_axes))), (name,args)
                value = "(&" + value + ")"
            source = source[:begin] + save(value) + source[end:]
        if slices:
            dims = [f"{name}({','.join(slices)})"]
            if len(v.dims) == len(slices):
                source = re.sub(r"\b" + name + r"\b(?!\s*\()",
                    lambda m: save(_base_expr(dims[0], symbols)), source)
    value = _base_expr(source, symbols)
    value = re.sub(r"\blake_(exp|log|log10)\b", r"tc_\1", value)
    value = re.sub(r"\bceiling\(", "udm_ceiling(", value)
    for name, replacement in substitutions.items():
        value = re.sub(r"\b" + name + r"\b", lambda m: replacement, value)
    return value


def body_line(n, line, symbols, function):
    if line.startswith("call "):
        name=line[5:].split("(",1)[0].strip()
        if name == "rayleigh_soak_wetgraupel":
            return name+"("+",".join(expression(a,symbols) for a in next(calls(line,name))[2])+");"
        assert name in _routines,(n,name)
        routine=_routines[name]
        args=next(calls(line,name))[2]
        assert len(args)==len(routine["args"]),(n,name,args)
        values=[]
        for actual,formal in zip(args,routine["args"]):
            value=expression(actual,symbols)
            if routine["vars"][formal].dims and "(" in actual and ":" not in actual:
                value="(&"+value+")"
            values.append(value)
        return name+"("+",".join(values)+");"
    if "=" in line and not line.startswith(("if", "else", "do ", "call ")):
        lhs, rhs = line.split("=", 1)
        base = lhs.strip().split("(")[0]
        if base in symbols and symbols[base].dims and (":" in lhs or "(" not in lhs):
            v = symbols[base]
            args = next(calls(lhs, base))[2] if "(" in lhs else [":"]*len(v.dims)
            axes, loops = [], []
            for axis, (arg, dim) in enumerate(zip(args, v.dims)):
                if ":" not in arg:
                    continue
                lo, hi = t.lake.bounds(dim)
                pair = arg.split(":")
                lo, hi = pair[0].strip() or lo, pair[1].strip() or hi
                var = "udmslice"+str(len(axes))
                axes.append(var)
                loops.append(f"for(int {var}={expression(lo,symbols)};"
                    f"{var}<={expression(hi,symbols)};++{var}) ")
            return "".join(loops) + expression(lhs,symbols,axes) + " = " + expression(rhs,symbols,axes) + ";"
    return t.body_line(n, line, symbols, function)


def normalized_body(routine):
    stack, out = [], []
    for n, line in routine["body"]:
        named = re.fullmatch(r"(\w+)\s*:\s*(do .*?)", line)
        if named:
            stack.append(named[1])
            out.append((n,named[2]))
        elif line.startswith("do "):
            stack.append(None)
            out.append((n,line))
        elif re.fullmatch(r"end\s*do(?:\s+\w+)?",line):
            name = stack.pop()
            if name:
                out.append((n,"udm_label_cycle_"+name))
            out.append((n,"enddo"))
            if name:
                out.append((n,"udm_label_exit_"+name))
        else:
            line = re.sub(r"\b(cycle|exit)\s+(\w+)",r"go to udm_\1_\2",line)
            out.append((n,line))
    assert not stack
    return out


def generate(source, directory, output, radar_source=None):
    global _function,_routines
    import numpy as np
    module, routines = parse(source, radar_source)
    _routines=routines
    schema = json.loads((directory/"module-schema.json").read_text())
    state_path = next((directory/"captures").glob("*-module.bin"))
    raw, offset, values = state_path.read_bytes(), 0, {}
    dims = {"kts":1,"kte":64,"kms":1,"kme":64,"km":64,"kdim":64,
        "ims":1,"ime":1,"its":1,"ite":1,"im":1,"idim":1}
    unresolved = {n:v.initial for n,v in module.items() if v.typ=="int" and v.const and v.initial}
    while unresolved:
        before = len(unresolved)
        for name, initial in list(unresolved.items()):
            try:
                dims[name] = eval(initial,{"__builtins__":{}},dims)
            except NameError:
                continue
            del unresolved[name]
        if len(unresolved)==before:
            raise ValueError(unresolved)
    for name, v in schema.items():
        count = 1
        for dim in v["dims"]:
            lo,hi=t.lake.bounds(dim)
            count *= eval(hi,{"__builtins__":{}},dims)-eval(lo,{"__builtins__":{}},dims)+1
        typ={"float":"<f4","double":"<f8","int":"<i4","bool":"<i4"}[v["typ"]]
        a=np.frombuffer(raw,typ,count,offset)
        values[name]=a.copy();offset+=a.nbytes
    assert offset==len(raw),(offset,len(raw))
    if radar_source:
        radar_schema=json.loads((directory/"radar-schema.json").read_text())
        radar_raw=(directory/"radar-state.bin").read_bytes();offset=0
        for name,v in radar_schema.items():
            dtype="<f8" if v["typ"] in ("double","cd") else "<f4" if v["typ"]=="float" else "<i4"
            count=v["bytes"]//np.dtype(dtype).itemsize
            values[name]=np.frombuffer(radar_raw,dtype,count,offset).copy();offset+=v["bytes"]
        assert offset==len(radar_raw)
    used=t.referenced(module,routines)
    used.update(("pi5","lamda4"))
    tables={n:module[n] for n in used if n in values and values[n].nbytes>4096}
    lines=["// Generated from pinned WRF 4.7.1 UDM and native initialized module state.",
        "#ifndef UDM_COLUMN_KMAX\n#define UDM_COLUMN_KMAX 64\n#endif",
        "__device__ int udm_ceiling(float x) { return __float2int_ru(x); }",
        "__device__ int udm_ceiling(double x) { return __double2int_ru(x); }",
        "struct UdmColumn { int nz;"]
    table_raw=b"";table_schema={}
    for name in sorted(used):
        v=module[name]
        if name in tables:
            a=values[name];table_schema[name] = dict(typ=v.typ,dims=v.dims,shape=[
                eval(t.lake.bounds(dim)[1],{"__builtins__":{}},dims)
                -eval(t.lake.bounds(dim)[0],{"__builtins__":{}},dims)+1 for dim in v.dims])
            table_raw+=a.tobytes();lines.append(f"const {v.typ}* {name};")
        elif name in values:
            def literal(x):
                return str(int(x)) if v.typ=="int" else ("true" if x else "false") if v.typ=="bool" else float(x).hex()+("f" if v.typ=="float" else "")
            vs=values[name]
            if v.typ == "cd":
                lines.append(f"cd {name}={{"+",".join(float(x).hex() for x in vs)+"};")
            elif v.dims:
                lines.insert(2,f"__device__ __constant__ {v.typ} udm_{name}[{len(vs)}] = {{"+",".join(map(literal,vs))+"};")
                lines.append(f"const {v.typ}* {name}=udm_{name};")
            else:
                lines.append(f"static constexpr {v.typ} {name}={literal(vs[0])};")
        elif v.initial:
            lines.append(f"static constexpr {v.typ} {name}={expression(v.initial,module)};")
        else:
            raise ValueError(("missing initialized module state",name))
    lines.append("__device__ void rayleigh_soak_wetgraupel(double x,double a,double b,double fm,double mo,cd mw,cd mi,double lambda,double& cback,int,int,int,int,int,int) { cback=refl_rayleigh_soak_wetgraupel(x,a,b,fm,mo,mw,mi,pi5,lamda4); }")
    for routine in routines.values():
        symbols=module|routine["vars"]
        args=[]
        for name in routine["args"]:
            v=routine["vars"][name]
            typ=v.typ+("*" if v.dims else "&" if v.intent in ("out","inout") else "")
            args.append(typ+" "+name)
        lines.append("__device__ __forceinline__ "+routine["ret"]+" "+routine["name"]+"("+",".join(args)+") {")
        function=routine["name"] if routine["ret"]!="void" else None
        _function=function
        if function:
            lines.append(routine["ret"]+" result;")
        statement_functions={}
        for n,line in routine["body"]:
            match=re.fullmatch(r"(\w+)\(([^()]*)\)\s*=\s*(.*)",line)
            if match and match[1] in symbols and not symbols[match[1]].dims:
                statement_functions[match[1]]=match
        for name,v in routine["vars"].items():
            if name in routine["args"] or name in statement_functions:
                continue
            if v.dims:
                factors=[]
                for dim in v.dims:
                    lo,hi=t.lake.bounds(dim)
                    size=eval(hi,{"__builtins__":{}},dims)-eval(lo,{"__builtins__":{}},dims)+1
                    dynamic=any(re.search(r"\b"+axis+r"\b",dim) for axis in ("kte","kme","km","kdim"))
                    factors.append("(UDM_COLUMN_KMAX+"+str(size-64)+")" if dynamic else str(size))
                initial=""
                if v.initial:
                    assert v.initial.startswith("(/") and v.initial.endswith("/)")
                    initial="={"+",".join(expression(x,symbols) for x in t.lake.split(v.initial[2:-2]))+"}"
                lines.append(v.typ+" "+name+"["+"*".join(factors)+"]"+initial+";")
            else:
                lines.append(v.typ+" "+name+("="+expression(v.initial,symbols) if v.initial else "")+";")
        for n,line in normalized_body(routine):
            match=re.fullmatch(r"(\w+)\(([^()]*)\)\s*=\s*(.*)",line)
            if match and match[1] in statement_functions:
                args=",".join(symbols[a.strip()].typ+" "+a.strip() for a in t.lake.split(match[2]))
                body="auto "+match[1]+"=[&]("+args+"){return "+expression(match[3],symbols)+";};"
            elif line.startswith("udm_label_"):
                body=line.replace("udm_label_","udm_")+": ;"
            elif re.fullmatch(r"go to udm_\w+",line):
                body="goto "+line[6:]+";"
            elif "go to udm_" in line:
                # A named CYCLE/EXIT can be a single-line IF body.
                pos=line.index("go to udm_")
                condition=line[:pos].strip()
                assert condition.startswith("if")
                body="if("+expression(condition[condition.index("(")+1:-1],symbols)+") goto "+line[pos+6:]+";"
            else:
                try:
                    old=t.expr;t.expr=expression
                    body=body_line(n,line,symbols,function)
                except Exception as exc:
                    raise ValueError((routine["name"],n,line)) from exc
                finally:
                    t.expr=old
            lines.append(f"// source:{n}\n"+body)
        if function:
            lines.append("return result;")
        lines.append("}")
    lines.append("};")
    output.write_text(re.sub(r"\btc_(exp|log|log10|pow)\b", r"udm_\1", "\n".join(lines))+"\n")
    (directory/"udm-tables.bin").write_bytes(table_raw)
    (directory/"table-schema.json").write_text(json.dumps(table_schema,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("directory",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--radar-source",type=Path,required=True)
    a=p.parse_args()
    generate(a.source,a.directory,a.output,a.radar_source)
