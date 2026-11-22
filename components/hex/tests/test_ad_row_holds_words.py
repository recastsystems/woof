"""Each row of the door table says, in its own words, what its default route holds; the words follow the fields.

THE BREAKAGE THIS PREVENTS: four check rounds of the 0.6.0 candidate in a row (16 Nov 2026) each found row texts
that contradicted the row's own fields, after a hand sweep each time. Rows that also hold the boundary-layer plume,
the mountain-drag closure and the transport limiter's coefficients opened "every physics derivative live except
radiation"; the 3 km and 12 km example rows said "B9 holds the MYNN plume" while their default (tape-free) route
differentiates the plume's transport; the 60 km class said "every physics derivative live except radiation" over a
run that holds the drag closure and the limiter; the 3 km row named a kernel set and a window length its default
route does not run. The receipts print the row's text as ``row_what``, so each was a false statement in every
receipt of that row.

This test derives, from each row's own machine fields, what its DEFAULT route holds: the scheme bundle and the
default boundary-layer route's images (which parts of MYNN-EDMF are held), the route's held outputs
(``AD_P3_HOLD_OUT``), the default cloud route, the limiter flag (``--fct-frozen``), the refuse list (radiation),
the mesh kind (lateral boundaries), the schemes held by name (``derivative_holds``), the kernel set, the window and
the host need. It reads it for every shipped row, and for a row made from every window class of every suite (the
door's own ``make_row``), and fails when the row's words (its ``what``, which the receipts carry as ``row_what``;
the param-grad ``row_what``; the known limits, the held-scheme lines and the cautions the door prints)

(a) say "every", "all" or "everything" about derivatives (or "nothing held") while a piece is held and the
    sentence's own "except ..." does not name it;
(b) state a whole hold the default route does not have (outside a clause about another route), or leave out a
    piece the default route holds;
(c) name a kernel set, a window length or a host need other than the field's;
(d) (each boundary-layer and cloud route's own sentences, read against the pieces that route holds) call a held piece
    differentiated, hold a part of a piece the route differentiates, or say the default route was not read through
    the command on this example while the row's known limits give that reading.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import rows

# ------------------------------------------------------------------------------------------------ the pieces
# piece -> (the words that name it, whether it is a physics derivative)
PIECES = {
    "radiation": (r"radiation", True),
    "limiter": (r"limiter", True),
    "drag closure": (r"drag closure|GSL drag", True),
    "boundary-layer closure": (r"turbulence closure|boundary layer's mixing|held boundary layer|whole boundary layer",
                               True),
    "plume": (r"plume", True),
    "mixing length": (r"mixing length", True),
    "plume's own speed": (r"plume's own speed", True),
    "species outputs": (r"species outputs", True),
    "cumulus": (r"cumulus", True),
    "riming": (r"riming|rime ratio", True),
    "lateral boundaries": (r"lateral[- ]boundar", False),
}
# a statement that a piece is held whole (checked against the route the clause is about)
WHOLE_HOLD = {
    "plume": re.compile(r"(?:B9 holds the (?:MYNN(?:-EDMF)? )?plume|with its plume held|freezes the plume"
                        r"|(?:MYNN(?:-EDMF)? )?(?:mass-flux )?plume (?:is |are )?(?:held|frozen)"
                        r"|holds? the (?:MYNN(?:-EDMF)? )?(?:mass-flux )?plume)(?!'s)"),
    "radiation": re.compile(r"radiation (?:is )?(?:held|refused)"),
    "limiter": re.compile(r"limiter(?:'s)? coefficients (?:are |is )?(?:held|frozen)"),
    "drag closure": re.compile(r"drag closure(?: \(GSL drag\))? (?:is )?held"),
    "boundary-layer closure": re.compile(r"turbulence closure(?:, [\w -]+)* (?:is |are )?held"),
}
# a clause about another route than the default (or about the param-grad bundle) is read against that route, not here
OTHER_ROUTE = re.compile(r"--boundary-layer taped|taped route|the exact route|held form|param-grad|parameter-carrier"
                         r"|before this build|--cloud-derivative|bundle_params|guarded")
# in an except-list, the words that name a piece whole (the plume's own coefficients are not the plume)
EXCEPT_NAMES = {"plume": r"plume(?!'s own)", "boundary-layer closure": r"turbulence closure|boundary layer's mixing"}
EVERY = re.compile(r"\b(?:every|all)\b(?:\s+[\w-]+){0,2}?\s+derivatives?\b"
                   r"|\bdifferentiat\w*\s+(?:everything|all\b|every\b)"
                   r"|\b(?:everything|all)\b(?:\s+else)?\s+(?:is\s+|are\s+)?(?:differentiated|live)\b"
                   r"|\ball-live\b|\bnothing\s+(?:is\s+)?held\b|\bno simplification")
KSET = re.compile(r"\b(kset-d\d+|storm-h-ws|storm-h|storm-bs\d[\w-]*?|b10-[a-z0-9]+(?:-[a-z0-9]+)*?)(?=[\s,;:.)]|-v\d|$)")


def _bundle(path: str | None) -> str:
    return Path(path or "").name


def bundle_bl_holds(name: str) -> set[str]:
    """What a scheme bundle's own boundary-layer images hold. B9: the held boundary layer (its turbulence closure,
    mass-flux plume and returned height at the forecast's values: the 12 km row's taped route). gen-A (the exact
    composition) and phys- (the full-physics bundle): nothing of the boundary layer."""
    if name.startswith(("gen-A", "genA", "phys-")):
        return set()
    if "B9" in name:
        return {"boundary-layer closure", "plume"}
    raise AssertionError(f"bundle {name!r}: this test has no hold words for its family; add them to bundle_bl_holds")


def route_bl_holds(row: dict, route_name: str | None = None, bundle: str | None = None) -> set[str]:
    bl = d.row_boundary_layer(row)
    name = route_name or bl.get("default") or "taped"
    route = (bl.get("routes") or {}).get(name) or {}
    over = (route.get("scheme_bundle") or {}).get("name") or ""
    if "bl-closure" in over:      # the closure-live form: closure, plume transport and height differentiated
        held = {"mixing length", "plume's own speed"}
    elif "bl-guarded" in over or "q1floor" in over:
        held = set()
    else:
        held = bundle_bl_holds(bundle or _bundle(row["assets"].get("bundle")))
    if (route.get("window_env") or {}).get("AD_P3_HOLD_OUT"):
        held.add("species outputs")
    return held


def refuse_holds(table: dict, row: dict) -> set[str]:
    """The refuse list's pieces. The template list (refuse.txt, in the pinned row archive) holds 64 kernel kinds of
    radiation and its cloud fraction; a whole limited-area class carries its own list in the table (refuse_list)."""
    name = Path(row["assets"].get("refuse") or "").name
    if not name:
        return set()
    if name == "refuse.txt":
        return {"radiation"}
    own = {Path(c["assets"]["refuse"]).name: c.get("refuse_list") for s in table["suites"].values()
           for c in s.get("window_classes") or [] if c.get("whole") and (c.get("assets") or {}).get("refuse")}
    assert name in own, f"refuse list {name}: neither the template's nor a class's own"
    assert own[name] == "", f"refuse list {name} refuses {own[name]!r}: name its pieces in refuse_holds"
    return set()


def _flag(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


def default_holds(table: dict, row: dict, door: str | None = None) -> set[str]:
    """The pieces the row's default route holds (door 'param-grad': the parameter-carrier bundle's)."""
    held = set(refuse_holds(table, row))
    if _flag(list(row.get("window_args") or []), "--fct-frozen") == "1":
        held.add("limiter")
    if row.get("mesh_kind") == "limited-area":
        held.add("lateral boundaries")
    held |= set(row.get("derivative_holds") or {})
    if door == "param-grad":
        bundle = (row.get("param_bundle") or {}).get("name") or ""
        held |= bundle_bl_holds(bundle)
    else:
        bundle = _bundle(row["assets"].get("bundle"))
        held |= route_bl_holds(row)
        cloud = (row.get("gradient_line") or {}).get("cloud_derivative") or {}
        if cloud.get("default") in ("number", "gated"):
            held.add("riming")
    if "gfrz" in bundle:
        held.add("drag closure")
    return held


def default_kset(row: dict) -> str:
    ks = (row.get("gradient_line") or {}).get("kernel_set") or {}
    if ks:
        if d.row_boundary_layer(row).get("default") == "tape-free" and ks.get("tape_free_route"):
            return ks["tape_free_route"]["set"]
        return ks["set"]
    return Path(row["assets"]["kernels"]).parent.name


# ------------------------------------------------------------------------------------------------ the reading
def _clauses(text: str) -> list[str]:
    return [c for c in re.split(r"(?<=[.;])\s+|;\s*", text) if c.strip()]


def _names(text: str, piece: str) -> bool:
    return re.search(PIECES[piece][0], text, flags=re.I) is not None


def read_words(label: str, text: str, held: set[str], *, whole: bool, row: dict | None = None) -> list[str]:
    """The flags of one text against the held pieces of the route it describes (whole: a row's own label, which
    must name every held piece)."""
    out = []
    for m in EVERY.finditer(text):      # (a)
        after = text[m.end():m.end() + 400]
        exc = re.match(r"[\w\s-]{0,40}?\bexcept\b(.*?)(?:\(|;|:|\.\s|$)", after, flags=re.S)
        physics = "physics" in m.group(0)
        need = {p for p in held if PIECES[p][1] or not physics}
        missed = sorted(p for p in need if not (exc and _names(exc.group(1), p)))
        extra = sorted(p for p in PIECES if exc and p not in held
                       and re.search(EXCEPT_NAMES.get(p, PIECES[p][0]), exc.group(1), flags=re.I))
        if missed:
            out.append(f"{label}: (a) {m.group(0)!r} while {', '.join(missed)} held")
        if extra:
            out.append(f"{label}: (a) {m.group(0)!r} except-list names {', '.join(extra)}, which the route does not hold")
    for clause in _clauses(text):       # (b) whole holds the route does not have
        if OTHER_ROUTE.search(clause):
            continue
        for piece, rx in WHOLE_HOLD.items():
            if piece not in held and rx.search(clause):
                out.append(f"{label}: (b) says {rx.search(clause).group(0)!r}; the default route does not hold the "
                           f"{piece}: {clause[:160]!r}")
    if whole:                           # (b) every held piece named
        missing = sorted(p for p in held if not _names(text, p))
        if missing:
            out.append(f"{label}: (b) leaves out {', '.join(missing)}, which the default route holds")
    if row is not None:                 # (c) kernel set, window, limiter minutes
        known = default_kset(row)
        for clause in _clauses(text):
            other = re.search(r"\bwas\b|exact route|taped|single-arithmetic", clause)
            for k in ([] if other else KSET.findall(clause)):
                if k != known:
                    out.append(f"{label}: (c) names kernel set {k}; the default route runs {known}")
            for n in re.findall(r"(?<!measured on a )\b(\d+)-step window", clause):
                ok = {int(row.get("default_steps") or 0), *[int(v) for v in (row.get("door_steps") or {}).values()]}
                if int(n) not in ok:
                    out.append(f"{label}: (c) names a {n}-step window; the row's window is {sorted(ok)}")
            for lo, hi in re.findall(r"\b(\d+) to (\d+) steps", clause):
                top = row.get("example_plan_steps")
                if top and int(hi) != int(top):
                    out.append(f"{label}: (c) names {lo} to {hi} steps; the example plan covers 1 to {top}")
            if row.get("fct_live_max_s") and "limiter" in clause:
                for n in re.findall(r"\b(\d+) minutes", clause):
                    if int(n) * 60 != int(row["fct_live_max_s"]):
                        out.append(f"{label}: (c) names {n} minutes for the limiter; fct_live_max_s is "
                                   f"{row['fct_live_max_s']} s")
    return out


# ------------------------------------------------------------------------------------------------ the route sentences
# THE BREAKAGE (confirmation 4, 16 Nov 2026): the 3 km row's default boundary-layer route said, in its own `error`
# sentence that every 3 km adjoint-test and sensitivity prints and both receipts store, that the route "was not measured
# through this command on this example", three lines above the same output's known limits that give that measurement
# (PASS). The row test read the row's `what` and limits but not a route's own sentences, and it did not read a sentence
# that calls a held piece differentiated ("radiation is differentiated" on a row that refuses it) or a partial hold of a
# piece the route differentiates ("the plume's transport is held" on the closure-live route).
NOT_READ = re.compile(r"\bnot (?:been )?(?:measured|read|run|checked) through this command on this (?:example|row|box)\b"
                      r"|\bnever (?:measured|read|run) through this command\b")
# "<piece>'s <part> is held": the part a route may hold without holding the piece whole
PARTIAL = re.compile(r"\b(plume|boundary layer|turbulence closure|limiter|cumulus scheme)'s "
                     r"((?:own )?[\w-]+(?:(?:, |,? and )(?!the\b)[\w-]+){0,4}) (?:is|are) (?:held|frozen)\b")
PARTS = {"plume": ("plume's own speed", r"(?:own )?(?:speed|coefficients|area|condensation|top)\b"),
         "boundary layer": ("mixing length", r"mixing length\b")}
WHOLE_OF = {"plume": "plume", "boundary layer": "boundary-layer closure", "turbulence closure": "boundary-layer closure",
            "limiter": "limiter", "cumulus scheme": "cumulus"}
DIFF_NAMES = {"radiation": r"radiation", "limiter": r"limiter", "drag closure": r"drag closure",
              "boundary-layer closure": r"turbulence closure|whole boundary layer", "plume": r"plume(?!'s)",
              "mixing length": r"mixing length", "plume's own speed": r"plume's own speed",
              "species outputs": r"species outputs", "cumulus": r"cumulus(?: scheme)?"}


def held_differentiated(label: str, text: str, held: set[str]) -> list[str]:
    """'<held piece> is differentiated' (or 'differentiates the <held piece>') outside a clause about another route."""
    out = []
    for clause in _clauses(text):
        if OTHER_ROUTE.search(clause) or re.search(r"\bexcept\b", clause):
            continue
        for piece in sorted(held & set(DIFF_NAMES)):
            words = DIFF_NAMES[piece]
            m = (re.search(rf"\b(?:{words})\b(?:(?!held|frozen|but)[^;:.()]){{0,30}}?\b(?:is|are) (?:differentiated|live)\b",
                           clause, flags=re.I)
                 or re.search(rf"\bdifferentiat(?:es|ed|ing) (?:the )?(?:\w+'s )?(?:{words})\b", clause, flags=re.I))
            if m:
                out.append(f"{label}: (d) says {m.group(0)!r}; the route holds the {piece}: {clause[:160]!r}")
    return out


def partial_holds(label: str, text: str, held: set[str]) -> list[str]:
    """'<piece>'s <part> is held' where the route neither holds the piece whole nor holds that part."""
    out = []
    for clause in _clauses(text):
        if OTHER_ROUTE.search(clause):
            continue
        for m in PARTIAL.finditer(clause):
            piece, part = m.group(1), m.group(2)
            if WHOLE_OF[piece] in held:
                continue
            own = PARTS.get(piece)
            if own and own[0] in held and re.match(own[1], part):
                continue
            out.append(f"{label}: (d) says {m.group(0)!r}; the route does not hold that part: {clause[:160]!r}")
    return out


def route_texts(route: dict) -> list[tuple[str, str]]:
    texts = [(k, route[k]) for k in ("what", "error", "runs", "says", "window_limit_warning")
             if isinstance(route.get(k), str)]
    form = route.get("form")
    if isinstance(form, dict) and isinstance(form.get("what"), str):
        texts.append(("form.what", form["what"]))
    return texts


def read_routes(table: dict, label: str, row: dict) -> list[str]:
    """Each boundary-layer and cloud route's own sentences, against the pieces THAT route holds, and the default routes'
    against the row's known limits: a sentence may not say the route was never read through the command on this
    example when the row's known limits give that reading."""
    flags = []
    base = default_holds(table, row)
    bl = d.row_boundary_layer(row)
    limits = d.known_limits(row, None) if bl or row.get("gradient_line") else {}
    read_here = any(re.search(r"\bPASS\b|\bINSIDE\b", item.get("text") or "") for item in limits.values())
    for name, route in (bl.get("routes") or {}).items():
        held = (base - route_bl_holds(row)) | route_bl_holds(row, name)
        for key, text in route_texts(route):
            where = f"{label} boundary-layer route {name} {key}"
            flags += held_differentiated(where, text, held) + partial_holds(where, text, held)
            if name == bl.get("default"):
                flags += read_words(where, text, held, whole=False)
                if read_here and NOT_READ.search(text):
                    flags.append(f"{where}: (d) says {NOT_READ.search(text).group(0)!r}; the row's known limits give "
                                 "this route's reading through this command on this example")
    cloud = (row.get("gradient_line") or {}).get("cloud_derivative") or {}
    for name, route in (cloud.get("routes") or {}).items():
        held = (base - {"riming"}) | ({"riming"} if name in ("number", "gated") else set())
        for key, text in route_texts(route):
            where = f"{label} cloud route {name} {key}"
            flags += held_differentiated(where, text, held) + partial_holds(where, text, held)
            if name == cloud.get("default") and read_here and NOT_READ.search(text):
                flags.append(f"{where}: (d) says {NOT_READ.search(text).group(0)!r}; the row's known limits give "
                             "this route's reading through this command on this example")
    return flags


def _host_flags(label: str, row: dict) -> list[str]:
    """(c) a host-need source states the need the row asks."""
    out = []
    needs = {int(row["need_host_mib"])} if row.get("need_host_mib") else set()
    needs |= {int(v) for v in (row.get("need_host_mib_by_door") or {}).values()}
    src = row.get("need_host_source")
    for need in sorted(needs):
        if src and f"{need:,} MiB" not in src and not (need % 1024 == 0 and f"{need // 1024} GiB" in src):
            out.append(f"{label}: (c) need_host_source does not state the need {need:,} MiB")
    return out


def read_row(table: dict, label: str, row: dict) -> list[str]:
    held = default_holds(table, row)
    flags = read_words(f"{label} what", row["what"], held, whole=True, row=row)
    flags += held_differentiated(f"{label} what", row["what"], held) + partial_holds(f"{label} what", row["what"], held)
    flags += read_routes(table, label, row)
    if "param-grad" in (row.get("doors") or []) and (row.get("param_bundle") or {}).get("instead_of_row"):
        own = row["param_bundle"]["instead_of_row"]
        pg = default_holds(table, row, "param-grad")
        flags += read_words(f"{label} param-grad instead_of_row", own, pg, whole=False)
        combined = f"{row['what']}. IN param-grad: {own}"
        flags += [f"{label} param-grad row_what: (b) leaves out {p}, which the param-grad bundle's route holds"
                  for p in sorted(pg) if not _names(combined, p)]
        bl = {"boundary-layer closure", "plume"} & pg   # the bundle's own pieces are said in its own sentence
        flags += [f"{label} param-grad: (b) instead_of_row leaves out the {p}, which its bundle holds"
                  for p in sorted(bl) if not _names(own, p)]
        if "as in the row" in own:
            for piece, rx in WHOLE_HOLD.items():
                if rx.search(own) and piece not in held:
                    flags.append(f"{label} param-grad: (b) says the {piece} is held 'as in the row'; the row's "
                                 "default route does not hold it")
    for key, item in d.known_limits(row, None).items():
        flags += read_words(f"{label} known limit {key}", item["text"], held, whole=False, row=row)
    for key, hold in (row.get("derivative_holds") or {}).items():
        flags += read_words(f"{label} held-scheme line {key}", hold["what"], held, whole=False, row=row)
    for i, note in enumerate(row.get("accuracy_notes") or []):
        flags += read_words(f"{label} caution {i}", note, held, whole=False, row=row)
    return flags + _host_flags(label, row)


def _mesh(root: Path, name: str) -> None:
    folder = root / "meshes" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.grid.nc").write_bytes(b"x")
    (folder / f"{name}.ugwp_oro_data.nc").write_bytes(b"x")


def made_rows(table: dict, tmp_path: Path, monkeypatch) -> list[tuple[str, dict]]:
    """One row made by the door's own make_row from every window class of every suite (a mesh of the class's size
    and kind)."""
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    out = []
    for suite_name, suite in table["suites"].items():
        lower = {"global": 0, "limited-area": 0}
        for cls in suite.get("window_classes") or []:
            kind = cls.get("mesh_kind", "global")
            cells = int(cls["max_cells"]) if int(cls["max_cells"]) > lower[kind] else lower[kind] + 1
            lower[kind] = max(lower[kind], int(cls["max_cells"]))
            name = f"m{len(out)}"
            _mesh(tmp_path, name)
            facts = {"cells": cells, "edges": 3 * cells, "levels": 59,
                     "nominal_dc_m": 3000.0 if kind == "limited-area" else 120000.0,
                     **({"limited_area": True} if kind == "limited-area" else {})}
            monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None, f=facts: dict(f))
            _, row = rows.make_row(copy.deepcopy(table), suite_name, rows.find_mesh(name), dt=60.0)
            made = (row.get("generated") or {}).get("window_class") or {}
            assert made.get("from_recipe") == cls.get("from_recipe"), (suite_name, cls.get("from_recipe"), made)
            out.append((f"suite {suite_name} class {cls.get('from_recipe')}", row))
    return out


def read_suites(table: dict, made: list[tuple[str, dict]]) -> list[str]:
    """A suite's own words: a hold the suite states must be held by every class it makes, or say which meshes."""
    flags = []
    for suite_name, suite in table["suites"].items():
        mine = [r for label, r in made if label.startswith(f"suite {suite_name} ")]
        every = set.intersection(*[default_holds(table, r) for r in mine]) if mine else set()
        for clause in _clauses(suite.get("what") or ""):
            if not re.search(r"\bheld\b|\bholds?\b", clause) or re.search(r"global|limited-area|3 km", clause):
                continue
            head = clause.split("differentiated", 1)[-1]
            for piece in ("radiation", "limiter", "drag closure", "plume"):
                if _names(head, piece) and piece not in every:
                    flags.append(f"suite {suite_name} what: (b) states the {piece} held without naming the meshes; "
                                 "a class of this suite does not hold it")
    return flags


def check_table(table: dict, tmp_path: Path, monkeypatch) -> tuple[int, int, list[str]]:
    flags = []
    for name, row in table["rows"].items():
        flags += read_row(table, f"row {name}", row)
    made = made_rows(table, tmp_path, monkeypatch)
    for label, row in made:
        flags += read_row(table, label, row)
    flags += read_suites(table, made)
    return len(table["rows"]), len(made), flags


# ------------------------------------------------------------------------------------------------ the tests
def test_every_row_and_window_class_says_what_its_default_route_holds(tmp_path, monkeypatch):
    table = d.load_table()
    n_rows, n_classes, flags = check_table(table, tmp_path, monkeypatch)
    assert n_rows == 7 and n_classes == 12, (n_rows, n_classes)
    assert not flags, "\n".join(flags)


def test_the_reading_catches_the_texts_the_checks_found(tmp_path, monkeypatch):
    """The four texts the hand checks found, put back: each is flagged (the test is not blind to its own class)."""
    table = d.load_table()
    q3, q12 = "hrrrv5-q3.96.38278.n40.60w91.30", "hrrrv5-q12.96.2890.n40.60w91.30"
    cases = [
        ("rows", "hrrrv5-x1.40962", "what", "RRFS hrrrv5 suite with every physics derivative live except radiation; "
         "the mountain-drag closure held; the plume held; FCT limiter coefficients frozen", "(a)"),
        ("rows", q12, "what", table["rows"][q12]["what"] + ". B9 holds the MYNN plume.", "(b) says"),
        ("rows", q3, "what", table["rows"][q3]["what"] + ". Kernel set storm-bs5-lean2-astra2-dyn6.", "(c) names kernel"),
        ("rows", q3, "what", table["rows"][q3]["what"] + ". It runs a 4-step window.", "(c) names a 4-step"),
        ("rows", q12, "what", table["rows"][q12]["what"].replace("mixing length", "length"), "(b) leaves out"),
    ]
    for _, row, key, text, tag in cases:
        bad = copy.deepcopy(table)
        bad["rows"][row][key] = text
        flags = read_row(bad, f"row {row}", bad["rows"][row])
        assert any(tag in f for f in flags), (row, tag, flags)
    # confirmation 4's texts: a route sentence that denies the row's own reading, a held piece called differentiated,
    # a partial hold of a part the route differentiates (each was missed by the reading before 0.6.0's last seal)
    tf = table["rows"][q3]["gradient_line"]["boundary_layer"]["routes"]["tape-free"]
    later = [
        (q3, ["gradient_line", "boundary_layer", "routes", "tape-free", "error"],
         tf["error"] + " This route was not measured through this command on this example.", "known limits give"),
        (q12, ["what"], table["rows"][q12]["what"] + "; the plume's transport is held", "does not hold that part"),
        ("hrrrv5-x1.40962", ["what"], table["rows"]["hrrrv5-x1.40962"]["what"] + ". Radiation is differentiated.",
         "holds the radiation"),
        (q12, ["what"], table["rows"][q12]["what"] + ". The cumulus scheme is differentiated.", "holds the cumulus"),
    ]
    for row, path, text, tag in later:
        bad = copy.deepcopy(table)
        node = bad["rows"][row]
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = text
        flags = read_row(bad, f"row {row}", bad["rows"][row])
        assert any(tag in f for f in flags), (row, tag, flags)
    bad = copy.deepcopy(table)
    sixty = [c for c in bad["suites"]["hrrrv5"]["window_classes"] if c["from_recipe"] == "60"][0]
    sixty["what"] = "RRFS hrrrv5 suite with every physics derivative live except radiation, on the 60 km recipe's sizes"
    _, _, flags = check_table(bad, tmp_path, monkeypatch)
    assert any("class 60 what: (a)" in f for f in flags), flags


def test_a_regenerated_table_keeps_the_checked_words(tmp_path, monkeypatch):
    """THE BREAKAGE: the generator wrote the rows' and classes' old words (and a recipe's development description)
    over the table on a re-run, bringing back texts the checks had removed. Re-run on the published set the table
    follows, it keeps every row's and class's words and images, and the result reads clean."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "ad_doors_from_latest.py"
    spec = importlib.util.spec_from_file_location("ad_doors_from_latest", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    table = d.load_table()
    lid = table["rows"]["hrrrv5-any"]["latest_id"]

    def recipe(bundle, kset="kset-d12", refuse="a,b", tlf="tlfd-fp32.cubin", what=None, args=()):
        r = {"env": {"AD_SCHEMES": "{L}/bundle/" + bundle, "AD_WIN_TEMPO_TLF_CUBIN": "{L}/img/" + tlf,
                     "AD_TAPE_PACK": "1"},
             "args": [["kernels", "{L}/kset/" + kset + "/kernels.json"], ["cubins", "{L}/kset/" + kset + "/cubins"],
                      ["schemes", "{L}/aux/schemes18.json"], ["tempo-cubin", "{L}/aux/split2.cubin"],
                      ["fct-frozen", "1"], *([["refuse", refuse]] if refuse is not None else []), *args]}
        if what:
            r["what"] = what
        return r

    old = "3 km storm box, 4-step window on ONE card (MYNN B9 with its plume held); B9 holds the MYNN plume"
    manifest = {"id": lid, "recipes": {
        "480": recipe("gen-A", tlf="ti2.tlf.cubin"), "480-best": recipe("gen-B9", args=[["tape-gib", "2"]]),
        "120": recipe("gen-B9", args=[["tape-gib", "6"]]),
        "60": {**recipe("genA-gfrz", tlf="ti2.tlf.cubin", args=[["tape-gib", "7"]]), "needs": {"cards": 1}},
        "3km": recipe("gen-B9", kset="storm-rev4", refuse="", what=old, args=[["local-topology", "1"]]),
        "3km-exact": recipe("gen-A", kset="storm-h", refuse="", tlf="ti2.tlf.cubin", what=old)}}
    made = gen.build(manifest, copy.deepcopy(table))["table"]
    for name in ("hrrrv5-any", "hrrrv5-any-exact"):
        assert made["rows"][name]["what"] == table["rows"][name]["what"], name
        assert made["rows"][name]["assets"]["bundle"] == table["rows"][name]["assets"]["bundle"], name
    for suite in ("hrrrv5", "hrrrv5-exact"):
        assert made["suites"][suite]["what"] == table["suites"][suite]["what"], suite
        for new, was in zip(made["suites"][suite]["window_classes"], table["suites"][suite]["window_classes"]):
            assert new.get("what") == was.get("what"), (suite, was["from_recipe"])
    _, _, flags = check_table(made, tmp_path, monkeypatch)
    assert not flags, "\n".join(flags)
