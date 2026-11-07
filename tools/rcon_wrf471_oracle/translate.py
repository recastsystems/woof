"""Lower WRF 4.7.1 RCON statements, retaining its REAL kinds and table axes.

The common Thompson parser is used unchanged. RCON has a density axis even
when hail is off, and removes the unused tmg_gacr table. Coefficients and
tables are captured after native initialization, rather than borrowed from
another Thompson generation.
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
    "rcon_thompson_translation", ROOT / "tools/thompson_generation_oracle/translate.py")
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)
t.ROUTINES = tuple("mp_rcon" if n == "mp_thompson" else n for n in t.ROUTINES) + (
    "theta_e", "t_lcl", "t_dew", "theta_wetb", "compt_fr_the")
t.TABLE_SHAPES.pop("tmg_gacr")
for name in ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr"):
    t.TABLE_SHAPES[name] = [37, 37, 1, 37, 37]
t.TABLE_SHAPES["tnr_rev"] = [100, 37, 37]
_statements = t.statements
def statements(path):
    for n, line in _statements(path):
        yield n, "end function" if line == "end" else line
t.statements = statements
_declaration = t.declaration
def declaration(line):
    return _declaration(re.sub(r"^real\*8\s+", "double precision ", line))
t.declaration = declaration
_parse_source = t.parse_source
def parse_source(path):
    module, routines = _parse_source(path)
    for routine in routines.values():
        body = []
        for n, line in routine["body"]:
            data = re.fullmatch(r"data\s+(\w+)\s*/(.*)/", line)
            if data:
                routine["vars"][data[1]].initial = "(/" + data[2] + "/)"
            else:
                body.append((n, line))
        routine["body"] = body
    return module, routines
t.parse_source = parse_source


def coefficient_names(module, routines):
    return sorted(t.referenced(module, routines) - set(t.TABLE_SHAPES)
                  - set(t.RADAR_STRINGS) - {"re_qc_bg", "re_qi_bg", "re_qs_bg"})


def instrument(source, directory):
    module, routines = t.parse_source(source)
    names = coefficient_names(module, routines)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "coefficient-schema.json").write_text(
        json.dumps({n: vars(module[n]) for n in names}, indent=2) + "\n")
    (directory / "table-schema.json").write_text(json.dumps({
        n: dict(typ=module[n].typ, shape=shape)
        for n, shape in sorted(t.TABLE_SHAPES.items())}, indent=2) + "\n")
    # Export one initialized native module and real forecast column calls.
    # Selecting global ii,jj prevents eight copies of the large tables.
    s = source.read_text()
    marker = "      subroutine mp_rcon ("
    begin = s.index(marker)
    declare = s.index("      debug_flag = .false.", begin)
    arrays = [n for n in routines["mp_rcon"]["args"]
              if routines["mp_rcon"]["vars"][n].dims]
    outputs = [n for n in arrays
               if routines["mp_rcon"]["vars"][n].intent == "inout"]
    precip = ("pptrain", "pptcloud", "pptsnow", "pptgraul", "pptice")
    def stream_write(names):
        return "".join(f"      WRITE(rcon_unit) {n}\n" for n in names)
    def open_capture(suffix):
        return f"""      WRITE(rcon_file,'(A,A,I6.6,A,I6.6,A,I4.4,A)') &
        TRIM(rcon_dir),'/rcon-i',ii,'-j',jj,'-s',rcon_step,'{suffix}'
      OPEN(NEWUNIT=rcon_unit,FILE=TRIM(rcon_file),ACCESS='STREAM', &
        FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
"""
    s = s[:declare] + """      INTEGER :: rcon_unit
      INTEGER, SAVE :: rcon_calls(128,128)=0
      INTEGER :: rcon_step
      LOGICAL :: rcon_capture,rcon_selected
      LOGICAL, SAVE :: rcon_dumped=.FALSE.
      CHARACTER(LEN=512) :: rcon_dir,rcon_file
      CALL GET_ENVIRONMENT_VARIABLE('MISC_RCON_CAPTURE_DIR',rcon_dir)
      IF (LEN_TRIM(rcon_dir)>0 .AND. .NOT.rcon_dumped &
          .AND. ii==2 .AND. jj==2) THEN
      WRITE(rcon_file,'(A,A,I6.6,A,I6.6,A)') &
        TRIM(rcon_dir),'/rcon-i',ii,'-j',jj,'-coefficients.bin'
      OPEN(NEWUNIT=rcon_unit,FILE=TRIM(rcon_file),ACCESS='STREAM', &
        FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
""" + "".join(f"      WRITE(rcon_unit) {n}\n" for n in names) + """      CLOSE(rcon_unit)
      WRITE(rcon_file,'(A,A,I6.6,A,I6.6,A)') &
        TRIM(rcon_dir),'/rcon-i',ii,'-j',jj,'-tables.bin'
      OPEN(NEWUNIT=rcon_unit,FILE=TRIM(rcon_file),ACCESS='STREAM', &
        FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
""" + "".join(f"      WRITE(rcon_unit) {n}\n" for n in sorted(t.TABLE_SHAPES)) + """      CLOSE(rcon_unit)
      rcon_dumped=.TRUE.
      END IF
      rcon_selected=(ii==2.OR.ii==25.OR.ii==49.OR.ii==73.OR.ii==95) &
        .AND. (jj==2.OR.jj==20.OR.jj==40.OR.jj==60.OR.jj==79)
      rcon_step=0
      IF (rcon_selected) THEN
        rcon_calls(ii,jj)=rcon_calls(ii,jj)+1
        rcon_step=rcon_calls(ii,jj)
      END IF
      rcon_capture=LEN_TRIM(rcon_dir)>0 .AND. rcon_selected &
         .AND. (rcon_step==1 .OR. rcon_step==2 .OR. &
                rcon_step==30 .OR. rcon_step==100)
      IF (rcon_capture) THEN
""" + open_capture(".input") + """      WRITE(rcon_unit) kts,kte,ii,jj,rcon_step,aer_init_opt,wif_input_opt
      WRITE(rcon_unit) dt
""" + stream_write(arrays + list(precip)) + """      CLOSE(rcon_unit)
      END IF
""" + s[declare:]
    post = "      IF (rcon_capture) THEN\n" + open_capture(".output")
    post += stream_write(outputs + list(precip))
    post += "      CLOSE(rcon_unit)\n      END IF\n"
    end = "      end subroutine mp_rcon"
    assert s.count(end) == 1
    # The dry-column exit is a normal native path; retain its output too.
    early = "      if (no_micro) return"
    assert s.count(early) == 1
    s = s.replace(early, "      if (no_micro) then\n" + post +
                  "      return\n      end if")
    s = s.replace(end, post + end)
    (directory / "column-schema.json").write_text(json.dumps({
        "header_i4": ["kts", "kte", "ii", "jj", "call", "aer_init_opt", "wif_input_opt"],
        "header_f4": ["dt"], "input_arrays_f4": arrays,
        "output_arrays_f4": outputs, "precip_f4": precip}, indent=2) + "\n")
    (directory / "module_mp_rcon.F").write_text(s)


def fuse_process_loops(s):
    """Keep level-local process rates in registers, at native REAL kinds.

    Native 2201-3231 computes warm rates, frozen rates, depletion limits,
    and tendencies in four separate level loops. Each array access in this
    region is to the current level. Loop induction and temporary scalars
    are private; no reduction or vertical transport runs here. Preserve
    statement order within each level, and retain every array used after
    this region (including later condensation/evaporation rates).

    The source-line guards deliberately bind this transform to the pinned
    WRF 4.7.1 source. The complete-driver strict replay gates the emitted
    program, including radar and effective-radius outputs.
    """
    start = s.index("__device__ __forceinline__ void mp_rcon(")
    stop = s.index("\n__device__", start + 1)
    prefix, routine, suffix = s[:start], s[start:stop], s[stop:]
    begin = routine.index("// source:2201\n")
    end = routine.index("// source:3236\n", begin)
    region = routine[begin:end]
    replacements = (
        ("// source:2201\nfor(k=kts; k<=kte; k+=1) {\n", ""),
        ("// source:2273\n}\n", ""),
        ("// source:2281\nfor(k=kts; k<=kte; k+=1) {\n", ""),
        ("// source:2889\n}\n", ""),
        ("// source:2895\nfor(k=kts; k<=kte; k+=1) {\n", ""),
        ("// source:2991\n}\n", ""),
        ("// source:2997\nfor(k=kts; k<=kte; k+=1) {\n", ""),
        ("// source:3231\n}\n", ""),
    )
    for before, after in replacements:
        assert region.count(before) == 1, before
        region = region.replace(before, after)
    # Check every process-rate reference, not just names selected by a
    # manual list. An array needed outside this region stays an array.
    index = "[0+(k-(kts))*(1)]"
    names = re.findall(r"double (p[rnb][a-z]_\w+)\[RCON_COLUMN_KMAX\];", routine)
    locals_ = []
    before, after = routine[:begin], routine[end:]
    for name in names:
        declaration = f"double {name}[RCON_COLUMN_KMAX];\n"
        zero = f"{name}{index} = 0.f;\n"
        outside = before.replace(declaration, "").replace(zero, "") + after
        if re.search(r"\b" + name + r"\b", outside):
            continue
        if not re.search(r"\b" + name + r"\b", region):
            continue
        assert before.count(zero) == 1, name
        references = re.findall(r"\b" + name + r"(\[[^\]]+\])", region)
        assert references and all(ref == index for ref in references), name
        region = region.replace(name + index, name)
        before = before.replace(declaration, "").replace(zero, "")
        locals_.append(name)
    assert len(locals_) >= 60, locals_
    fused = ("// Native 2201-3231: level-local rates, native statement order.\n"
             "for(k=kts; k<=kte; k+=1) {\n" +
             "".join(f"double {name} = 0.0;\n" for name in locals_) +
             region + "}\n")
    return prefix + before + fused + after + suffix


def generate(source, directory, output):
    t.generate(source, directory, output)
    s = output.read_text().replace("ThompsonColumn", "RconColumn")
    s = s.replace("THOMPSON_COLUMN_KMAX", "RCON_COLUMN_KMAX")
    s = s.replace("Generated Thompson numerical generation", "Generated WRF 4.7.1 RCON numerical generation")
    # RCON uses the plain EXP/LOG spellings. The common expression parser
    # gives these a lake_ prefix before Thompson's intrinsic mapping runs.
    s = re.sub(r"\blake_(exp|log|log10)\b", r"tc_\1", s)
    s = re.sub(r"\btc_(exp|log|log10|pow)\b", r"rcon_\1", s)
    # Unlike the common Thompson generation, RCON updates these module
    # work scalars inside the column. They are private mutable members.
    for name in ("rho_s", "t1_qg_qc", "t2_qg_sd", "t2_qg_me"):
        s = re.sub(r"static constexpr (float|double) " + name + r"\b",
                   r"\1 " + name, s)
    s = "#ifndef RCON_COLUMN_KMAX\n#define RCON_COLUMN_KMAX 64\n#endif\n" + s
    s = fuse_process_loops(s)
    output.write_text(s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("instrument", "generate", "check"))
    p.add_argument("source", type=Path)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.action == "instrument":
        instrument(a.source, a.directory)
    elif a.action == "generate":
        generate(a.source, a.directory, a.output)
    else:
        m, rs = t.parse_source(a.source)
        for r in rs.values():
            for n, line in r["body"]:
                t.body_line(n, line, m | r["vars"], r["name"] if r["ret"] != "void" else None)
        print(f"Checked {len(rs)} RCON statement programs")
