"""Generate complete CUDA DVIC procedures from the pinned public-domain WRF v4.

This offline generator preserves the native expression tree and every branch,
including native iteration limits and labels. It runs only during development.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import re

from .build import LSM_SHA256

PARAMETERS = {name: f"p[P_{name}+({{index}})-1]" for name in
              ("SMCMAX", "SMCWLT", "DKSAT", "DWSAT", "BEXP")}
SCALARS = {name: f"p[W_P_STRIDE+{k}]" for k, name in
           enumerate(("BBVIC", "BDVIC", "GDVIC"))}
SIGNATURES = {
    "RR1": "const float*p,float i_0,float i_max,float yd,float&r1",
    "RR2": "float yd,float y0,float r1,float fmax,float fsur,float dt,float dp,float bb,float&r2",
    "PHILIP_INFIL": "const float*p,const float*smc,const float*sice,float qinsur,float dt,float&facc,float&fsur,int inflmax",
    "GREEN_AMPT_INFIL": "const float*p,const float*zsoil,const float*smc,const float*sice,float qinsur,float&facc,float&fsur,int inflmax",
    "SMITH_PARLANGE_INFIL": "const float*p,const float*zsoil,const float*smc,const float*sice,float qinsur,float&facc,float&fsur,int inflmax",
    "DYNAMIC_VIC": "const float*p,float dt,const float*smc,const float*sh2o,const float*sice,float sicemax,const float*zsoil,float qinsur,float&facc,float&pddum,float&runsrf,int opt_infdv",
}


def expression(text):
    text = text.upper().replace("PARAMETERS%", "")
    for old, new in ((".GT.", ">"), (".GE.", ">="), (".LT.", "<"),
                     (".LE.", "<="), (".EQ.", "=="), (".NE.", "!="),
                     (".AND.", " and "), (".OR.", " or ")):
        text = text.replace(old, new)
    node = ast.parse(text.strip(), mode="eval").body

    def emit(n):
        if isinstance(n, ast.Name):
            return SCALARS.get(n.id, "NSOIL" if n.id == "NSOIL" else n.id.lower())
        if isinstance(n, ast.Constant):
            return repr(float(n.value)) + "f"
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return f"f_neg({emit(n.operand)})"
        if isinstance(n, ast.BinOp):
            fn = {ast.Add: "FADD", ast.Sub: "FSUB", ast.Mult: "FMUL",
                  ast.Div: "FDIV", ast.Pow: "runoff_powf"}[type(n.op)]
            return f"{fn}({emit(n.left)},{emit(n.right)})"
        if isinstance(n, ast.Compare):
            assert len(n.ops) == 1
            op = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=",
                  ast.Eq: "==", ast.NotEq: "!="}[type(n.ops[0])]
            return f"({emit(n.left)}{op}{emit(n.comparators[0])})"
        if isinstance(n, ast.BoolOp):
            op = "&&" if isinstance(n.op, ast.And) else "||"
            return "(" + op.join(emit(v) for v in n.values) + ")"
        if isinstance(n, ast.Call):
            name = n.func.id
            if name in PARAMETERS:
                return PARAMETERS[name].format(index=emit(n.args[0]))
            if name in ("SMC", "SICE", "SH2O", "ZSOIL"):
                return f"{name.lower()}[(int)({emit(n.args[0])})-1]"
            fn = {"MIN": "f_min", "MAX": "f_max", "ABS": "fabsf",
                  "SQRT": "sqrtf", "EXP": "glibc_expf"}[name]
            return fn + "(" + ",".join(emit(v) for v in n.args) + ")"
        raise ValueError(ast.dump(n))

    # Array indices are integers; the scientific arithmetic remains real.
    return re.sub(r"p\[(P_\w+)\+\(([^()]*)\)-1\]",
                  r"p[\1+(int)(\2)-1]", emit(node))


def split_args(text):
    depth = 0
    first = 0
    out = []
    for k, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            out.append(text[first:k].strip())
            first = k + 1
    return out + [text[first:].strip()]


def statement(text):
    if text.startswith("CALL "):
        name, args = text[5:].split("(", 1)
        name = name.strip()
        args = split_args(args.rsplit(")", 1)[0])
        if name == "WDFCND2":
            _, wdf, wcnd, smc, sice, layer = args
            k = f"(int)({expression(layer)})-1"
            return (f"d_wdfcnd2({expression(smc)},{expression(sice)},"
                    + ",".join(f"p[P_{n}+{k}]" for n in
                               ("SMCMAX", "BEXP", "DWSAT", "DKSAT"))
                    + f",&{wdf.lower()},&{wcnd.lower()});")
        args = ["p" if a == "PARAMETERS" else expression(a) for a in args if a != "NSOIL"]
        return name.lower() + "(" + ",".join(args) + ");"
    if text.startswith("GOTO "):
        return "goto label_" + text[5:].strip() + ";"
    if text == "EXIT":
        return "break;"
    left, right = text.split("=", 1)
    left = left.strip().lower()
    return left + "=" + (right.strip() if left in ("isoil", "izmax", "inflmax")
                         else expression(right)) + ";"


def routine(source, name):
    match = re.search(r"^[ \t]*SUBROUTINE\s+" + name + r"\s*\(", source, re.M | re.I)
    assert match, name
    end = re.search(r"END SUBROUTINE\s+" + name, source[match.end():], re.I)
    assert end, name
    text = source[match.start():match.end()+end.start()]
    line0 = source[:match.start()].count("\n") + 1
    lines = []
    pending = ""
    locals_ = []
    begun = False
    for line in text.splitlines():
        raw = line.split("!", 1)[0].strip().upper()
        if not raw:
            continue
        if "::" in raw:
            if "INTENT" not in raw and "TYPE" not in raw:
                kind = "int" if raw.startswith("INTEGER") else "float"
                locals_.append(kind + " " + raw.split("::", 1)[1].strip().lower() + ";")
            continue
        if raw.startswith(("SUBROUTINE", "IMPLICIT")):
            continue
        if raw.endswith("&"):
            pending += raw[:-1]
            continue
        raw = pending + raw.lstrip("&")
        pending = ""
        begun = True
        label = re.match(r"^(\d+)\s+", raw)
        if label:
            lines.append("label_" + label[1] + ": ;")
            raw = raw[label.end():]
        if raw == "ELSE":
            lines.append("} else {")
        elif raw in ("END IF", "ENDIF", "END DO", "ENDDO"):
            lines.append("}")
        elif re.match(r"(?:ELSE\s+)?IF\s*\(", raw):
            start = raw.index("(")
            depth = 1
            stop = start + 1
            while depth:
                depth += (raw[stop] == "(") - (raw[stop] == ")")
                stop += 1
            prefix = "} else " if raw.startswith("ELSE") else ""
            cond = expression(raw[start+1:stop-1])
            tail = raw[stop:].strip()
            lines.append(prefix + "if(" + cond + ")" +
                         (" {" if tail == "THEN" else " " + statement(tail)))
        elif raw.startswith("DO "):
            loop, bounds = raw[3:].strip().split("=", 1)
            first, last = split_args(bounds)
            loop = loop.strip().lower()
            # Native bounds are integer expressions.
            last = last.lower().replace("nsoil", "NSOIL")
            lines.append(f"for({loop}={first};{loop}<={last};++{loop}) {{")
        else:
            lines.append(statement(raw))
    assert begun and not pending
    if name == "DYNAMIC_VIC":
        # Native YD is unassigned before two RR1 calls. Retain a quiet-NaN
        # residue rather than invent a hydrologic value. The native witness
        # dvic_undefined.py documents this input-dependent undefined path.
        locals_.append("yd=__int_as_float(0x7fc00000);")
    return (f"// Native module_sf_noahmplsm.F:{line0}, complete {name}.\n"
            f"__device__ void {name.lower()}({SIGNATURES[name]}) {{\n"
            + "\n".join("    " + line for line in locals_+lines) + "\n}\n")


def generate(source):
    assert hashlib.sha256(source).hexdigest() == LSM_SHA256
    text = source.decode()
    order = ("RR1", "RR2", "PHILIP_INFIL", "GREEN_AMPT_INFIL",
             "SMITH_PARLANGE_INFIL", "DYNAMIC_VIC")
    return ("// BEGIN GENERATED NATIVE DYNAMIC VIC\n"
            "// Public domain WRF v4; licenses/LICENSE-WRF-public-domain.txt.\n"
            "// Generated by tools/noahmp_options_wrf461_oracle/generate_dynamic_vic.py.\n"
            + "\n".join(routine(text, name) for name in order)
            + "// END GENERATED NATIVE DYNAMIC VIC\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_bytes()))


if __name__ == "__main__":
    main()
