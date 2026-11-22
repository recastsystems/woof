"""Instrument a COPY of a WRF source tree to capture scheme-wrapper arguments.

The method is the tools/wrf_instrumented one (a separate instrumented copy whose
added code only reads state and writes dumps; the referee binary stays pristine),
generalised from one hand-written hook to any list of wrapper subroutines:

    python3 tools/wrf480_capture/instrument.py WRF_COPY TARGETS.json

For each target {"file": "phys/module_bl_ysu.F", "subroutine": "ysu"} it parses the
dummy-argument list and the declarations of those dummies (REAL/INTEGER/LOGICAL of
any kind, rank 0-4, OPTIONAL respected), then inserts

  * ``USE module_gpuwm_capture`` after the SUBROUTINE statement,
  * an ``entry`` dump of every declared dummy before the first executable statement,
  * an ``exit`` dump before every RETURN and before END SUBROUTINE / CONTAINS.

Dummies inside ``#if`` blocks, CHARACTER and derived-type dummies are not dumped and
are listed in the receipt.  ``frame/module_gpuwm_capture.F`` is added to the frame
library and ``solve_em`` sets the step after its ``itimestep`` increment.
Nothing assigns to model state: the instrumented history must equal the pristine
referee history word for word, which the capture run checks.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PP_OPEN, PP_CLOSE = " @@<@@ ", " @@>@@ "
PP_SEGMENT = re.compile(r"@@<@@(.*?)@@>@@", re.S)


def drop_pp(text):
    """Text with #if-block segments removed, and the names those segments held."""
    held = []
    for seg in PP_SEGMENT.findall(text):
        held += re.findall(r"[A-Za-z_]\w*", seg)
    return PP_SEGMENT.sub(" ", text), held
TYPE_RE = re.compile(
    r"^(real|integer|logical|double\s*precision)\b(\s*\*\s*\d+|\s*\([^)]*\))?", re.I)
SPEC_START = re.compile(
    r"^(use|implicit|integer|real|logical|character|double\s*precision|complex"
    r"|type\s*\(|class\s*\(|parameter|dimension|save|data|external|intrinsic"
    r"|intent|optional|namelist|equivalence|common|interface|end\s*interface"
    r"|procedure|import|private|public|allocatable|target|pointer)\b", re.I)


def _strip_comment(line):
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _skippable(line):
    s = line.strip()
    return not s or s.startswith("!") or s.startswith("#")


def statements(lines, start):
    """Yield (first, last, text, in_pp) logical statements from line ``start``.

    ``in_pp`` is None for a preprocessor line, else whether the statement sits
    in (or spans) a ``#if`` block.
    """
    i, depth_pp = start, 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("#"):
            if re.match(r"#\s*if", s):
                depth_pp += 1
            elif re.match(r"#\s*endif", s):
                depth_pp -= 1
            yield i, i, s, None
            i += 1
            continue
        if not s or s.startswith("!"):
            i += 1
            continue
        first, parts, pp_inside, local = i, [], False, 0
        while True:
            body = _strip_comment(lines[i]).rstrip()
            cont = body.endswith("&")
            if cont:
                body = body[:-1]
            b = body.strip()
            if b.startswith("&"):
                b = b[1:]
            # text inside a #if block that opened within this statement is
            # wrapped so callers can drop exactly that text and keep the rest
            parts.append(f"{PP_OPEN}{b}{PP_CLOSE}" if local > 0 else b)
            if not cont:
                break
            i += 1
            while _skippable(lines[i]):
                s2 = lines[i].strip()
                if s2.startswith("#"):
                    pp_inside = True
                    if re.match(r"#\s*if", s2):
                        local += 1
                    elif re.match(r"#\s*endif", s2):
                        local -= 1
                i += 1
        yield first, i, " ".join(parts), (depth_pp > 0 or pp_inside)
        i += 1


def split_top(s, sep=","):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [x.strip() for x in out]


def parse_decl(text):
    """Declaration -> [(name, base, rank, optional)] (PARAMETERs ignored)."""
    text = text.strip()
    if "::" in text:
        spec, ents = text.split("::", 1)
    else:
        m = TYPE_RE.match(text)
        if not m:
            return []
        spec, ents = text[:m.end()], text[m.end():]
    m = TYPE_RE.match(spec.strip())
    if not m:
        return []
    base = m.group(1).lower().replace(" ", "")
    attrs = split_top(spec.strip()[m.end():].lstrip(", "))
    rank, guards = 0, set()
    for a in attrs:
        al = a.lower().replace(" ", "")
        if al.startswith("dimension"):
            rank = len(split_top(a[a.index("(") + 1:a.rindex(")")]))
        if al in ("optional", "pointer", "allocatable"):
            guards.add(al)
        if al == "parameter":
            return []
    optional = frozenset(guards)
    out = []
    for e in split_top(ents):
        mm = re.match(r"([A-Za-z_]\w*)\s*(\((.*)\))?", e)
        if not mm:
            continue
        r = len(split_top(mm.group(3))) if mm.group(2) else rank
        out.append((mm.group(1), base, r, optional))
    return out


def instrument_file(path: Path, sub: str, receipt: dict):
    lines = path.read_text(errors="replace").split("\n")
    sub_re = re.compile(r"^\s*((recursive|pure|elemental)\s+)*subroutine\s+"
                        + re.escape(sub) + r"\b", re.I)
    head = next(i for i, line in enumerate(lines) if sub_re.match(line))
    stmts = statements(lines, head)
    _first, last, text, _ = next(stmts)
    text, pp_dummies = drop_pp(text)
    arglist = text[text.index("(") + 1:text.rindex(")")] if "(" in text else ""
    dummies = [a for a in split_top(arglist) if re.fullmatch(r"[A-Za-z_]\w*", a)]
    lower = {d.lower(): d for d in dummies}
    decls, first_exec, skipped_pp, optional_stmt = {}, None, set(), set()
    depth, opened_at = 0, None   # #if depth within the specification part
    for f, _l, t, pp in stmts:
        if pp is None:
            if re.match(r"#\s*if", t):
                if depth == 0:
                    opened_at = f
                depth += 1
            elif re.match(r"#\s*endif", t):
                depth -= 1
            continue
        tl, held = drop_pp(t)
        tl = tl.strip()
        skipped_pp.update(h.lower() for h in held if h.lower() in lower)
        m_opt = re.match(r"^optional\b\s*(::)?\s*(.*)$", tl, re.I)
        if m_opt:
            # attribute statement (RRTMG_SWRAD: "OPTIONAL :: tauaer3d_sw, ...");
            # an absent optional must never be read, so it gets a PRESENT guard
            optional_stmt.update(n.lower() for n in re.findall(r"[A-Za-z_]\w*", m_opt.group(2)))
            continue
        if SPEC_START.match(tl) and not re.search(r"\bfunction\b", tl, re.I):
            for name, base, rank, opt in parse_decl(tl):
                key = name.lower()
                if key in lower:
                    if pp:
                        skipped_pp.add(key)
                    else:
                        decls[key] = (lower[key], base, rank, opt)
            continue
        first_exec = f
        break
    # The first executable statement may sit inside a #if block the build does
    # not compile (RRTMG_SWRAD opens its body with WRF_CHEM-only code): the
    # entry dump then goes before that block's #if, where every build runs it.
    entry_at = opened_at if depth > 0 else first_exec
    for key in optional_stmt & set(decls):
        nm, base, rank, opt = decls[key]
        decls[key] = (nm, base, rank, frozenset(opt) | {"optional"})
    names = [d for d in dummies if d.lower() in decls]
    undeclared = [d for d in dummies if d.lower() not in decls]
    tag = sub.lower()

    def dump(phase, indent="      "):
        out = [f"{indent}IF (gpcap_begin('{tag}','{phase}')) THEN"]
        for d in names:
            nm, _base, rank, opt = decls[d.lower()]
            lb = f", LBOUND({nm})" if rank else ""
            call = f"CALL gpcap_put('{nm.lower()}', {nm}{lb})"
            # absent optionals and unassociated pointers are never touched
            # (RRTMG_SWRAD's aerosol pointers are null with aer_opt=0)
            tests = ([f"PRESENT({nm})"] if "optional" in opt else []) +                     ([f"ASSOCIATED({nm})"] if "pointer" in opt else []) +                     ([f"ALLOCATED({nm})"] if "allocatable" in opt else [])
            if not tests:
                out.append(f"{indent}  {call}")
            else:
                for i, test in enumerate(tests):
                    out.append(f"{indent}  {'  ' * i}IF ({test}) THEN")
                out.append(f"{indent}  {'  ' * len(tests)}{call}")
                for i in reversed(range(len(tests))):
                    out.append(f"{indent}  {'  ' * i}END IF")
        out.append(f"{indent}  CALL gpcap_end()")
        out.append(f"{indent}END IF")
        return out

    end_re = re.compile(r"^end(\s*subroutine\b.*)?$", re.I)
    edits, stop = [], None
    for f, l, t, pp in statements(lines, first_exec):
        if pp is None:
            continue
        tl = t.strip()
        low = tl.lower()
        if re.match(r"^contains\b", low) or end_re.match(tl):
            edits.append((f, f, None))
            stop = f
            break
        if re.match(r"^(\d+\s+)?return$", low):
            edits.append((f, f, None))
            continue
        m = re.match(r"^(\d+\s+)?if\s*\((.*)\)\s*return$", tl, re.I)
        if m:
            ind = re.match(r"\s*", lines[f]).group(0)
            label = m.group(1) or ""
            edits.append((f, l, [f"{ind}{label}IF ({m.group(2)}) THEN"]
                          + dump("exit", ind + "  ")
                          + [f"{ind}  RETURN", f"{ind}END IF"]))
    if stop is None:
        raise RuntimeError(f"{sub}: no END SUBROUTINE found")
    for f, l, rep in sorted(edits, key=lambda e: -e[0]):
        if rep is None:
            lines[f:f] = dump("exit")
        else:
            lines[f:l + 1] = rep
    lines[entry_at:entry_at] = dump("entry")
    lines[last + 1:last + 1] = [
        "      USE module_gpuwm_capture, ONLY: gpcap_begin, gpcap_end, gpcap_put"]
    path.write_text("\n".join(lines))
    receipt[f"{path.name}:{sub}"] = {
        "dumped": [decls[d.lower()][0] for d in names],
        "not_dumped_undeclared_or_unsupported": undeclared,
        "not_dumped_preprocessor_blocks": sorted(set(pp_dummies) | skipped_pp),
        "exit_points": len(edits),
        "entry_moved_before_if_block": depth > 0,
        "optional_by_attribute_statement": sorted(optional_stmt & set(decls)),
    }


def hook_framework(tree: Path):
    shutil.copy(HERE / "module_gpuwm_capture.F", tree / "frame" / "module_gpuwm_capture.F")
    mk = tree / "frame" / "Makefile"
    t = mk.read_text()
    anchor = "MODULES =       module_driver_constants.o  \\\n"
    assert t.count(anchor) == 1, "frame/Makefile MODULES anchor"
    mk.write_text(t.replace(
        anchor, "MODULES =       module_gpuwm_capture.o module_driver_constants.o  \\\n"))
    se = tree / "dyn_em" / "solve_em.F"
    t = se.read_text()
    anchor = "   grid%itimestep = grid%itimestep + 1\n"
    assert t.count(anchor) == 1, "solve_em itimestep anchor"
    t = t.replace(anchor, anchor + "   CALL gpcap_set_step(grid%itimestep)\n")
    m = re.search(r"(?im)^\s*SUBROUTINE\s+solve_em\b.*?\n([ \t]*USE\s)", t, re.S)
    t = t[:m.start(1)] + "   USE module_gpuwm_capture, ONLY: gpcap_set_step\n" + t[m.start(1):]
    se.write_text(t)


def main():
    tree, targets = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    receipt = {}
    hook_framework(tree)
    for tgt in targets:
        instrument_file(tree / tgt["file"], tgt["subroutine"], receipt)
    (tree / "GPUWM-CAPTURE-RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    for k, v in receipt.items():
        print(k, "dumped", len(v["dumped"]), "not", v["not_dumped_undeclared_or_unsupported"],
              "pp", v["not_dumped_preprocessor_blocks"], "exits", v["exit_points"])


if __name__ == "__main__":
    main()
