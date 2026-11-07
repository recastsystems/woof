"""Write the licence notice for the Rust crates the wheel's compiled helpers link.

    python tools/helper_licences.py --engine GPUWM_SOURCE [--scorer-engine GPUWM_SOURCE]
    python tools/helper_licences.py --engine GPUWM_SOURCE --check  # exit 1 if it would change

THE BREAKAGE THIS PREVENTS (release readiness B12, 2026-10-14): the wheel ships
hex_obs_scorecard, rw_mpas_geometry, rw_mpas_stream and rw_mpas_static (the
last since 2026-10-15), which link third-party
crates whose MIT, BSD, ISC, Zlib, Unicode and Apache grants each condition
redistribution in binary form on the crate's notice travelling with the copy.
NOTICE pointed at woof's bridge-bundle notice for those texts, and no woof
wheel carries that file, so the wheel shipped the binaries without the texts.

The texts come from woof's own notice for its bridge bundle
(licenses/THIRD-PARTY-LICENSES-bridge-binaries.txt in GPUWM_SOURCE), restricted
to the crates the inventoried programs link. New versions absent from that
notice use the full texts in their exact resolved crate source.  That file also covers code this wheel
does not carry, so it is not copied whole.  The crate lists come from cargo:
the scorer's private project (tools/build_obs_scorecard.py) and the rw-mpas
package, each resolved against GPUWM_SOURCE's tools/rustwx/Cargo.lock offline.
A crate the programs link that the source notice does not inventory stops the
tool by name: its text would be missing.  Run it against the woof source the
release's helpers are built from (src/hexcore/data/engine-bridges.json names
the bridges' commit; the scorer is built against the engine release the
declaration's floor names).  Needs cargo and git on PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib

TREE = Path(__file__).resolve().parents[1]
OUTPUT = TREE / "licenses" / "THIRD-PARTY-LICENSES-helper-binaries.txt"
SOURCE_NOTICE = "licenses/THIRD-PARTY-LICENSES-bridge-binaries.txt"
RULE = "=" * 78
DASH = "-" * 78
# The source notice's section 2 subsections (ported third-party code inside
# ArWen's own crates), each reproduced when a program links the crate named.
SECTION_TWO = {"2.1": ("region_global_dealias", "region-global-dealias"),
               "2.2": ("rustwx-render",), "2.3": ("rw-mpas",), "2.4": ("static-fields",), "2.5": ("rw-libm",)}
PROGRAMS = (("S", "hex_obs_scorecard", "the observation scorer (rust/obs-scorecard)"),
            ("B", "rw_mpas_geometry, rw_mpas_stream, rw_mpas_static, rw_mpas_mesh, librw_surface.so, rw_mpas_lbc and the RRFS programs "
                  "rw_mpas_rrfs_connectivity, rw_mpas_rrfs_damping and rw_mpas_rrfs_deformation",
             "the bundled engine bridges (crate rw-mpas)"),
            ("O", "rw_igra2", "the IGRA2 radiosonde door (crate rw-obs)"),
            ("W", "rw_wrfbatch and rw_ensbatch", "the renderer and the ensemble renderer (crate rw-wrfbatch)"),
            ("C", "rw_mpas_convert", "the chemistry-aware history converter (crate rw-mpas)"),
            ("F", "rw_sfc_series", "the surface series producer (crate rw-sfc-series)"),
            ("P", "rw_mpas_physics_pool", "the Registry pool reader (rust/physics-pool)"),
            # THE BREAKAGE THIS PREVENTS (0.4.2 licence census, 2026-10-22): the wheel ships rw_s2s (this tree's
            # tools/rustwx workspace, crate rw-s2s) and rw_hex_chem_sources (rust/chem-sources), which link rayon,
            # serde, serde_json and chrono at versions no other helper links, while this notice said the remaining
            # helpers link the Rust standard library only, so those versions' texts did not travel with the wheel.
            ("R", "rw_s2s", "the subseasonal reducer (crate rw-s2s, this tree's tools/rustwx)"),
            ("H", "rw_hex_chem_sources", "the chemistry source preparer (rust/chem-sources)"),
            ("E", "rw_s2s_energy", "the S2S energy tables' reducer (crate rw-s2s-energy, this tree's tools/rustwx)"),
            # THE BREAKAGE THIS PREVENTS (hex 0.4.5 licence census, 2026-10-25): the wheel ships rw_hex_ugwp_oro
            # (rust/ugwp-oro), which links rayon and serde_json, and no program row listed it, so their texts did
            # not travel with the binary and the census refused it as an unrecognized ELF.
            ("U", "rw_hex_ugwp_oro", "the GSL orographic drag statistics builder (rust/ugwp-oro)"),
            ("V", "rw_hex_wave_products and wave_audit", "the wave maps and audit (rust/wave-products)"))


def cargo_tree(cwd: Path, *args: str, linked: bool, target: str = "all") -> list[tuple[str, str, str | None]]:
    """(name, version, source-or-None) for every package one cargo tree prints."""
    edges = "normal,no-proc-macro" if linked else "normal"
    result = subprocess.run(["cargo", "tree", "--offline", "--target", target, "-e", edges, "--prefix", "none",
                           "--format", "{p}", *args], cwd=cwd, capture_output=True,
                          text=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    text = result.stdout
    rows = set()
    for line in text.splitlines():
        match = re.fullmatch(r"(\S+) v(\S+)((?: \([^()]*\))*)", line.strip())
        if match is None:
            raise SystemExit(f"cannot read cargo tree line: {line!r}")
        sources = [s for s in re.findall(r"\(([^()]*)\)", match.group(3)) if s not in ("proc-macro", "*")]
        rows.add((match.group(1), match.group(2), sources[0] if sources else None))
    return sorted(rows, key=lambda row: (row[0], row[1], row[2] or ""))


def closures(engines: dict[str, Path], work: Path) -> dict[str, dict[str, list]]:
    spec = importlib.util.spec_from_file_location("build_obs_scorecard", TREE / "tools/build_obs_scorecard.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    manifest = scorer.write_project(engines["S"], work / "scorer")
    sfc = work / "sfc"
    sfc.mkdir()
    text = (TREE / "tools/rustwx/crates/rw-sfc-series/Cargo.toml").read_text()
    for old, new in (("../rw-mpas", "tools/rustwx/crates/rw-mpas"),
                     ("../rw-store", "tools/rustwx/crates/rw-store"),
                     ("../../vendor/netcrust", "tools/rustwx/vendor/netcrust"),
                     ("../../../grib1_bridge/vendor/grib-core", "tools/grib1_bridge/vendor/grib-core")):
        text = text.replace('"' + old + '"', '"' + str(engines["F"] / new) + '"')
    text = text.replace('"src/main.rs"', '"' + str(TREE / "tools/rustwx/crates/rw-sfc-series/src/main.rs") + '"')
    (sfc / "Cargo.toml").write_text(text)
    (sfc / "Cargo.lock").write_bytes((TREE / "tools/rustwx/crates/rw-sfc-series/Cargo.lock").read_bytes())
    energy = work / "energy"
    energy.mkdir()
    crate = TREE / "tools/rustwx/crates/rw-s2s-energy"
    text = (crate / "Cargo.toml").read_text()
    for old, new in (("../../vendor/netcrust", "tools/rustwx/vendor/netcrust"),
                     ("../rw-store", "tools/rustwx/crates/rw-store")):
        text = text.replace('"' + old + '"', '"' + str(engines["E"] / new) + '"')
    text = text.replace('"src/main.rs"', '"' + str(crate / "src/main.rs") + '"')
    (energy / "Cargo.toml").write_text(text)
    (energy / "Cargo.lock").write_bytes((crate / "Cargo.lock").read_bytes())
    out = {}
    for key, args, locked in (("S", ("--manifest-path", str(manifest)), ("--locked",)),
                              ("B", ("-p", "rw-mpas"), ("--locked",)),
                              ("O", ("-p", "rw-obs"), ("--locked",)),
                              ("W", ("-p", "rw-wrfbatch"), ("--locked",)),
                              ("C", ("-p", "rw-mpas"), ("--locked",)),
                              ("F", ("--manifest-path", str(sfc / "Cargo.toml")), ("--locked",)),
                              ("P", ("--manifest-path", str(TREE / "rust/physics-pool/Cargo.toml")), ("--locked",)),
                              ("R", ("-p", "rw-s2s"), ("--locked",)),
                              ("H", ("--manifest-path", str(TREE / "rust/chem-sources/Cargo.toml")), ("--locked",)),
                              ("U", ("--manifest-path", str(TREE / "rust/ugwp-oro/Cargo.toml")), ("--locked",)),
                              ("V", ("--manifest-path", str(TREE / "rust/wave-products/Cargo.toml")), ("--locked",)),
                              ("E", ("--manifest-path", str(energy / "Cargo.toml")), ("--locked",))):
        # E resolves from the engine workspace directory so its vendored sources apply, exactly as
        # tools/build_s2s_energy.sh builds it.
        root = {"F": sfc, "P": TREE / "rust/physics-pool", "R": TREE / "tools/rustwx",
                "H": TREE / "rust/chem-sources", "U": TREE / "rust/ugwp-oro",
                "V": TREE / "rust/wave-products"}.get(key, engines[key] / "tools/rustwx")
        target = "x86_64-unknown-linux-gnu" if key in ("F", "P", "R", "H", "E", "U", "V") else "all"
        linked = cargo_tree(root, *locked, *args, linked=True, target=target)
        every = cargo_tree(root, *locked, *args, linked=False, target=target)
        out[key] = {"linked": linked,
                    "compile_only": sorted({(n, v) for n, v, _ in every} - {(n, v) for n, v, _ in linked})}
    # THE BREAKAGE THIS PREVENTS (hex 0.4.5 release review, 2026-10-24): rw_mpas_mesh is built from woof
    # 3c41c9c551, whose rw-mpas links static-fields and through it chrono 0.4.45, while [B] was listed from
    # the stream's workspace only, so the wheel would ship the mesh bridge without that version's text.
    # Each --bridge-engine (a bridge row's own gpuwm_source_rev) adds its rw-mpas closure to [B].
    for extra in engines.get("B+", ()):
        root = extra / "tools/rustwx"
        linked = set(out["B"]["linked"]) | set(cargo_tree(root, "--locked", "-p", "rw-mpas", linked=True))
        every = {(n, v) for n, v, _ in cargo_tree(root, "--locked", "-p", "rw-mpas", linked=False)}
        every |= {(n, v) for n, v, _ in out["B"]["linked"]} | set(out["B"]["compile_only"])
        out["B"] = {"linked": sorted(linked, key=lambda row: (row[0], row[1], row[2] or "")),
                    "compile_only": sorted(every - {(n, v) for n, v, _ in linked})}
    return out


def own_crate(source: str | None, engines: dict[str, Path], work: Path) -> bool:
    """WOOF's own rustwx crates and the scorer's private project; vendored crates are not own."""
    if source is None or not source.startswith("/"):
        return False
    path = Path(source)
    return path.is_relative_to(work) or path.is_relative_to(TREE / "rust") or path.is_relative_to(
        TREE / "tools/rustwx/crates") or any(path.is_relative_to(engine / "tools/rustwx/crates")
                                            for engine in _engine_paths(engines))


def _engine_paths(engines: dict) -> list[Path]:
    return [path for value in engines.values() for path in (value if isinstance(value, tuple) else (value,))]


def parse_notice(text: str) -> dict:
    """Section 2 subsections, section 3 rows and section 4 texts of the source notice."""
    s2 = text.index("2.  NOTICES THAT ARE NOT CRATE LICENCES")
    s3 = text.index("3.  VENDORED RUST CRATES -- INVENTORY")
    s4 = text.index("4.  VENDORED RUST CRATES -- LICENCE TEXTS")
    heads = list(re.finditer(r"^2\.(\d)  .*$", text[s2:s3], re.M))
    section_two = {}
    for i, head in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else text[s2:s3].rindex(RULE)
        section_two[f"2.{head.group(1)}"] = text[s2 + head.start(): s2 + stop].rstrip() + "\n"
    inventory, untexted = {}, {}
    block = text[s3:s4]
    main, _, rest = block.partition("Crates that vendored no licence text of their own:")
    for line in main.splitlines():
        # A long grant can run into its text numbers with no space between.
        match = re.fullmatch(r"  (\S+) (\d\S*) +(.*?)(?: *\[([\d,]+)\])? *", line)
        if match:
            refs = tuple(int(n) for n in match.group(4).split(",")) if match.group(4) else ()
            inventory[(match.group(1), match.group(2))] = (match.group(3), refs)
    for line in rest.split("\n\n", 2)[1].splitlines():
        match = re.fullmatch(r"  (\S+) (\d\S*) +(.*?) {2,}(.*?) *", line)
        if match:
            untexted[(match.group(1), match.group(2))] = match.group(4)
    stop = re.compile(r"^={70,}\n(?=[A-Za-z0-9])", re.M).search(text, s4 + len(RULE)).start()
    head = re.compile(r"^-{78}\n\[(\d+)\]  as carried by: .*\n     sha256 ([0-9a-f]{64})\n-{78}\n", re.M)
    found = list(head.finditer(text, s4, stop))
    texts = {}
    for i, match in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else stop
        texts[int(match.group(1))] = (match.group(2), text[match.end():end].rstrip("\n") + "\n")
    # The source states its own counts; a parse that disagrees with them has misread the file.
    pairs = int(re.search(r"(\d+) distinct crate name-and-version pairs", block).group(1))
    distinct = int(re.search(r"(\d+) distinct texts", text[s4:stop]).group(1))
    if len(inventory) != pairs or len(texts) != distinct or any("[" in g for g, _ in inventory.values()):
        raise SystemExit(f"read {len(inventory)} crates and {len(texts)} texts from the source notice, "
                         f"which states {pairs} and {distinct}")
    # A crate section 3 does not inventory can still be in the notice: the radar appendix inventories the
    # crates the simulated-radar dependency closure added, with texts of its own (RADAR-DEPS-n).  Section 3
    # wins where both list a crate.
    extra_inventory, extra_untexted, extra_texts = radar_appendix(text)
    for crate, row in extra_inventory.items():
        inventory.setdefault(crate, row)
    for crate, reason in extra_untexted.items():
        if crate not in untexted and not inventory[crate][1]:
            untexted[crate] = reason
    texts.update(extra_texts)
    return {"section_two": section_two, "inventory": inventory, "untexted": untexted, "texts": texts}


RADAR_HEAD = "WOOF SIMULATED RADAR -- ADDED DEPENDENCY LICENCE TEXTS"
RADAR_END = "END WOOF SIMULATED RADAR DEPENDENCY LICENCE TEXTS"


def radar_appendix(text: str) -> tuple[dict, dict, dict]:
    """Inventory rows, text-less reasons and texts of the source notice's radar appendix.

    THE BREAKAGE THIS PREVENTS (hex 0.4.0 repin to gpuwm 2.9.0, 2026-10-14): at the 2.9.0 release commit
    rw-mpas links libloading 0.9.0, so the scorer publish.yml builds against the floor's engine links it
    too.  The source notice carries libloading's ISC text only in this appendix, so section 3 alone found
    no row for it and the helper notice could not be written: the wheel would have shipped the scorer
    without that text.  An appendix that is absent contributes nothing; one whose count disagrees with what
    was read stops the tool.
    """
    start = text.find(RADAR_HEAD)
    if start < 0:
        return {}, {}, {}
    end = text.index(RADAR_END, start)
    block = text[start:end]
    inventory, untexted = {}, {}
    lines = block.splitlines()
    for i, line in enumerate(lines):
        match = re.fullmatch(r"  (\S+) (\d\S*) \| (.+?) \| \[([A-Z0-9,-]*)\]", line)
        if not match:
            continue
        refs = tuple(ref for ref in match.group(4).split(",") if ref)
        inventory[(match.group(1), match.group(2))] = (match.group(3), refs)
        if not refs:
            reason = []
            for follow in lines[i + 1:]:
                if not follow.startswith("    "):
                    break
                reason.append(follow.strip())
            untexted[(match.group(1), match.group(2))] = " ".join(reason)
    head = re.compile(r"^-{78}\n\[(RADAR-DEPS-\d+)\] sha256 ([0-9a-f]{64})\nAs carried by:\n(?:  .*\n)+-{78}\n",
                      re.M)
    found = list(head.finditer(text, start, end))
    stop = text.rindex(RULE, start, end)
    texts = {}
    for i, match in enumerate(found):
        stop_at = found[i + 1].start() if i + 1 < len(found) else stop
        texts[match.group(1)] = (match.group(2), text[match.end():stop_at].strip("\n") + "\n")
    stated = re.search(r"(\d+) distinct grant and attribution texts follow", block)
    if stated is None or int(stated.group(1)) != len(texts):
        raise SystemExit(f"read {len(texts)} texts from the source notice's radar appendix, which states "
                         f"{stated.group(1) if stated else 'no count'}")
    return inventory, untexted, texts


def _ref_key(ref: int | str) -> tuple:
    """Section 4's numbered texts first, in number order, then the radar appendix's named ones."""
    return (0, ref, "") if isinstance(ref, int) else (1, 0, ref)


def identity(engine: Path) -> tuple[str, str]:
    """(commit, Cargo.lock sha256) of a woof source."""
    lock = hashlib.sha256((engine / "tools/rustwx/Cargo.lock").read_bytes()).hexdigest()
    if not (engine / ".git").exists():
        # The release build mounts a source export. Authenticate its identity
        # against the source and lock hashes already pinned with the bridge.
        import json
        pins = json.loads((TREE / "src/hexcore/data/engine-bridges.json").read_text())
        for row in pins["bridges"].values():
            if (row.get("cargo_lock_sha256") == lock and row.get("gpuwm_source_rev")
                    and hashlib.sha256((engine / row["source"]).read_bytes()).hexdigest() == row["source_sha256"]):
                return row["gpuwm_source_rev"], lock
        raise SystemExit("source export does not match a pinned source and Cargo.lock identity")
    rev = subprocess.run(["git", "-C", str(engine), "rev-parse", "HEAD"], check=True, capture_output=True,
                         text=True).stdout.strip()
    return rev, lock


def crate_source(name, version, source, engines):
    """Find the exact resolved manifest, including registry crates added by hex."""
    candidates = [Path(source)] if source and source.startswith("/") else []
    for engine in set(_engine_paths(engines)):
        vendor = engine / "tools/rustwx/vendor/crates-io"
        candidates.extend((vendor / f"{name}-{version}", vendor / name))
    home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    candidates.extend((home / "registry/src").glob(f"*/{name}-{version}"))
    for path in candidates:
        manifest = path / "Cargo.toml"
        if manifest.is_file():
            package = tomllib.loads(manifest.read_text())["package"]
            if package["name"] == name and package["version"] == version:
                if isinstance(package.get("license"), dict):
                    for parent in path.parents:
                        candidate = parent / "Cargo.toml"
                        if candidate.is_file():
                            workspace = tomllib.loads(candidate.read_text()).get("workspace", {}).get("package", {})
                            if isinstance(workspace.get("license"), str):
                                package["license"] = workspace["license"]
                                break
                return path, package
    raise SystemExit(f"resolved crate source unavailable: {name} {version}")


def supplement_notice(notice, sets, engines, work):
    """New helper versions need their own source texts, not another version's grant."""
    for closure in sets.values():
        for name, version, source in closure["linked"]:
            if own_crate(source, engines, work) or (name, version) in notice["inventory"]:
                continue
            path, package = crate_source(name, version, source, engines)
            grant = package.get("license")
            if not isinstance(grant, str):
                raise SystemExit(f"{name} {version}: no SPDX licence in resolved manifest")
            texts = [p for p in path.iterdir() if p.is_file() and
                     re.match(r"(?i)^(?:licen[cs]e|copying|copyright|notice)(?:[._-]|$)", p.name)]
            if package.get("license-file"):
                texts.append(path / package["license-file"])
            if not texts:
                raise SystemExit(f"{name} {version}: licence text absent from resolved source {path}")
            refs = []
            for text in sorted(set(texts)):
                body = text.read_text().rstrip("\n") + "\n"
                digest = hashlib.sha256(body.encode()).hexdigest()
                ref = "SOURCE-" + digest[:16]
                refs.append(ref)
                notice["texts"][ref] = (digest, body)
            notice["inventory"][(name, version)] = (grant, tuple(refs))


def render(engines: dict[str, Path], notice_sha: str, notice: dict, sets: dict, work: Path) -> str:
    linked: dict[tuple[str, str], set[str]] = {}
    own: set[tuple[str, str]] = set()
    for key, closure in sets.items():
        for name, version, source in closure["linked"]:
            if own_crate(source, engines, work):
                if not Path(source).is_relative_to(work):
                    own.add((name, version))
                continue
            linked.setdefault((name, version), set()).add(key)
    missing = sorted(set(linked) - set(notice["inventory"]))
    if missing:
        raise SystemExit("linked but not inventoried in the source notice, so its text would be missing: "
                         + ", ".join(f"{n} {v}" for n, v in missing))
    refs = sorted({ref for crate in linked for ref in notice["inventory"][crate][1]}, key=_ref_key)
    absent = [ref for ref in refs if ref not in notice["texts"]]
    if absent:
        raise SystemExit(f"the source notice's section 4 lacks texts {absent}")
    names = {name for name, _ in linked} | {name for name, _ in own}
    two = [key for key, crates in sorted(SECTION_TWO.items()) if names & set(crates)]
    compile_only = sorted({crate for closure in sets.values() for crate in closure["compile_only"]}
                          - set(linked) - own)
    out = [f"THIRD-PARTY LICENCES FOR THE GPUWM-HEX WHEEL'S COMPILED HELPERS\n{RULE}\n\n",
           "Written by tools/helper_licences.py; do not edit it by hand.  The tool's own\n"
           "docstring says why this file exists and how to regenerate or check it.\n\n",
           "The programs below, which the wheel carries in woof/hex/libexec, are built\n"
           "from WOOF's Rust workspace (tools/rustwx in the woof repository) and link\n"
           "third-party "
           "crates.  Their grants condition redistribution in binary form on the\n"
           "notices below travelling with the copy; this file is that.\n\n"]
    out += [f"  [{key}]  {label}: {what}\n" for key, label, what in PROGRAMS]
    out.append("\nThe remaining Rust helpers link the Rust standard library only.\n"
               "NOTICE section 12 also names the C libraries the helpers link.\n")
    bridges, scorer = identity(engines["B"]), identity(engines["S"])
    pin_sha = hashlib.sha256((TREE / "src/hexcore/data/engine-bridges.json").read_bytes()).hexdigest()
    out.append(f"\nWHAT IT COVERS\n{'-' * 14}\n"
               f"  woof commit             {bridges[0]}\n"
               f"  tools/rustwx/Cargo.lock  sha256 {bridges[1]}\n"
               f"  engine-bridges.json      sha256 {pin_sha}\n"
               f"  source notice            {SOURCE_NOTICE} at that commit,\n"
               f"                           sha256 {notice_sha}\n")
    out.append("\nThe source identity above is the resolved build workspace. The hashed bridge\n"
               "manifest retains each binary's own source identity and lock override;\n"
               "its top-level identity is the fallback for entries without an override.\n")
    if scorer != bridges:
        out.append(f"  scorer resolved against  woof commit {scorer[0]},\n"
                   f"                           Cargo.lock sha256 {scorer[1]}\n")
    energy = identity(engines["E"])
    if energy != scorer:
        out.append(f"  [E] resolved against     woof commit {energy[0]},\n"
                   f"                           Cargo.lock sha256 {energy[1]}\n")
    for extra in engines.get("B+", ()):
        commit, lock = identity(extra)
        out.append(f"  [B] also resolved against woof commit {commit},\n"
                   f"                           Cargo.lock sha256 {lock}\n")
    prebuilt = identity(engines["W"])
    if prebuilt != bridges:
        out.append(f"  [W][C] resolved against   woof commit {prebuilt[0]},\n"
                   f"                           Cargo.lock sha256 {prebuilt[1]}\n")
    out.append("\nThe crate lists use cargo tree -e normal,no-proc-macro, with all targets\n"
               "for the engine workspace and x86_64-unknown-linux-gnu for the separate\n"
               "surface-series and pool-reader projects. These are conservative package\n"
               "closures; linker removal can leave fewer crates in a binary. Numbered\n"
               "texts come from the engine notice. SOURCE references come from the exact\n"
               "resolved crate source when that notice does not inventory its version.\n")
    if any(isinstance(ref, str) for ref in refs):
        out.append("A [RADAR-DEPS-n] reference is a text the source notice carries in its WOOF\n"
                   "simulated radar appendix, which is where it inventories that crate.\n")
    out.append(f"\n{RULE}\n1.  THE PROGRAMS' OWN CODE\n{DASH}\n\n"
               "rust/obs-scorecard is this project's own code (Apache-2.0, LICENSE).  The\n"
               "WOOF crates below are the rustwx workspace's own under MIT; its text ships as\n"
               "woof/hex/data/licenses/rustwx-LICENSE-MIT.\n\n")
    out.append("rw-sfc-series is project code with MPAS-derived portions under Apache-2.0\n"
               "AND BSD-3-Clause. The Registry pool reader is MIT. rw-s2s, rw-s2s-energy and\n"
               "rust/chem-sources (hex-chem-sources), rust/ugwp-oro (hex-ugwp-oro) and\n"
               "rust/wave-products (hex-wave-products) are this project's own code (Apache-2.0).\n\n")
    out += [f"  {name} {version}\n" for name, version in sorted(own)]
    if two:
        out.append(f"\n{RULE}\n2.  NOTICES THAT ARE NOT CRATE LICENCES\n{DASH}\n\n"
                   "Some of WOOF's own crates above port third-party code into themselves.\n"
                   "The source notice's paragraphs for the ones these programs link follow\n"
                   "unchanged; where they say \"this bundle\" they mean WOOF's bridge bundle,\n"
                   "and here they cover the programs in this wheel that link the crate named.\n\n")
        out += [notice["section_two"][key] + "\n" for key in two]
    out.append(f"{RULE}\n3.  RUST CRATES -- INVENTORY\n{DASH}\n\n"
               f"{len(linked)} crate name-and-version pairs.  The column after the grant\n"
               "names the programs that link the crate (key above); the bracketed numbers\n"
               "point at the texts in section 4.  A crate with no number vendored no\n"
               "licence file of its own, and its grant is the SPDX expression its\n"
               "Cargo.toml declares.  Where a crate offers a choice of grants, the\n"
               "permissive one is taken; no copyleft branch is taken.\n\n")
    name_width = max(len(f"{n} {v}") for n, v in linked) + 2
    grant_width = max(len(notice["inventory"][crate][0]) for crate in linked) + 2
    for crate in sorted(linked):
        grant, numbers = notice["inventory"][crate]
        cited = f"[{','.join(map(str, numbers))}]" if numbers else ""
        out.append(f"  {crate[0] + ' ' + crate[1]:<{name_width}}{grant:<{grant_width}}"
                   f"{','.join(sorted(linked[crate])):<6}{cited}".rstrip() + "\n")
    untexted = [crate for crate in sorted(linked) if crate in notice["untexted"]]
    if untexted:
        out.append("\nCrates that vendored no licence text of their own, as the source notice\n"
                   "records them:\n\n")
        out += [f"  {crate[0] + ' ' + crate[1]:<32}{notice['inventory'][crate][0]:<42}"
                f"{notice['untexted'][crate]}".rstrip() + "\n" for crate in untexted]
    if compile_only:
        out.append("\nRun only while compiling and not linked into any of the programs (procedural\n"
                   "macros and what they alone need):\n\n")
        out += [f"  {name} {version}\n" for name, version in compile_only]
    out.append(f"\n{RULE}\n4.  RUST CRATES -- LICENCE TEXTS\n{DASH}\n\n"
               f"{len(refs)} distinct texts.  Each is headed by the crates here that carried it\n"
               "and the SHA-256 of the file as the crate vendored it.\n")
    for ref in refs:
        carriers = ", ".join(f"{n} {v}" for n, v in sorted(c for c in linked if ref in notice["inventory"][c][1]))
        sha, body = notice["texts"][ref]
        out.append(f"\n{DASH}\n[{ref}]  as carried by: {carriers}\n     sha256 {sha}\n{DASH}\n{body}")
    out.append(f"\n{RULE}\nEND OF THIRD-PARTY LICENCES FOR THE GPUWM-HEX WHEEL'S COMPILED HELPERS\n")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", type=Path, required=True,
                        help="the woof source the bridges are built from (engine-bridges.json's commit); "
                             "its notice supplies every text")
    parser.add_argument("--scorer-engine", type=Path,
                        help="the woof source the scorer is built against, when it is another one "
                             "(publish.yml builds it against the engine release the declared floor names)")
    parser.add_argument("--energy-engine", type=Path,
                        help="the woof source the S2S energy helper was built against, when it differs "
                             "from the cached scorer's source")
    parser.add_argument("--prebuilt-engine", type=Path,
                        help="the woof source the renderer prebuilts are built from "
                             "(engine-bridges.json source_prebuilts.gpuwm_source_rev), when it is another one")
    parser.add_argument("--bridge-engine", type=Path, action="append", default=[],
                        help="another woof source a bundled bridge is built from (a row's own gpuwm_source_rev "
                             "in engine-bridges.json, e.g. rw_mpas_mesh); its rw-mpas closure joins [B]")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--source-notice", type=Path,
                        help="bridge notice from the sealed engine wheel when the build source export omits licenses/")
    parser.add_argument("--closures-json", type=Path, help="write the per-program Cargo dependency closures")
    parser.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    args = parser.parse_args(argv)
    engine = args.engine.resolve()
    engines = {key: engine for key, _, _ in PROGRAMS}
    engines["S"] = (args.scorer_engine or args.engine).resolve()
    # THE BREAKAGE THIS PREVENTS: the S2S/CF renderer (rw_wrfbatch, rw_ensbatch) and the chemistry converter
    # (rw_mpas_convert) are built from a newer
    # engine source than the bridges (source_prebuilts); listing their crates from the bridges' source would
    # ship texts for crates the renderer no longer links and none for those it now does.
    engines["W"] = engines["C"] = (args.prebuilt_engine or args.engine).resolve()
    # A retained scorer and a newly built energy helper can have different engine inputs.
    engines["E"] = (args.energy_engine or args.scorer_engine or args.engine).resolve()
    extras = tuple(path.resolve() for path in args.bridge_engine if path.resolve() != engine)
    if extras:
        engines["B+"] = extras
    rev = identity(engine)[0]
    notice_bytes = (args.source_notice or engine / SOURCE_NOTICE).read_bytes()
    with tempfile.TemporaryDirectory(prefix="helper-licences-") as scratch:
        work = Path(scratch).resolve()
        sets = closures(engines, work)
        notice = parse_notice(notice_bytes.decode("utf-8"))
        supplement_notice(notice, sets, engines, work)
        # The sealed engine NOTICE records both as first-party MIT despite
        # their unchanged Cargo metadata. Release ruling 2026-10-20 accepts it.
        for name in ('grib-core', 'wrf-core'):
            key = (name, '0.1.0')
            if key in notice['inventory']:
                notice['inventory'][key] = ('MIT', notice['inventory'][key][1])
                notice['untexted'][key] = 'first-party MIT; sealed engine NOTICE, Vendored Rust crates; release ruling 2026-10-20'
        if '2.2' in notice['section_two']:
            notice['section_two']['2.2'] = (TREE / 'licenses/LICENSE-embedded-fonts-OFL-1.1.txt').read_text()
        text = render(engines, hashlib.sha256(notice_bytes).hexdigest(), notice, sets, work)
        if args.closures_json:
            import json
            rows = {key: [{"name": n, "version": v, "source": s,
                           "licence": (notice["inventory"][(n, v)][0] if (n, v) in notice["inventory"] else
                                       ("Apache-2.0" if n == "hex-obs-scorecard" else
                                        "Apache-2.0 AND BSD-3-Clause" if n == "rw-sfc-series" else
                                        "Apache-2.0" if n in ("rw-s2s", "rw-s2s-energy", "hex-chem-sources", "hex-ugwp-oro", "hex-wave-products") else
                                        crate_source(n, v, s, engines)[1].get("license", "UNKNOWN")))}
                          for n, v, s in closure["linked"]] for key, closure in sets.items()}
            args.closures_json.write_text(json.dumps({"engine": identity(engine), "programs": rows}, indent=2) + "\n")
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        if current != text:
            print(f"{args.out} differs from what {rev} gives; rerun without --check", file=sys.stderr)
            return 1
        print(f"{args.out} matches woof {rev}")
        return 0
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} from woof {rev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
