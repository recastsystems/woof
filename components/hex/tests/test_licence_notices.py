"""The licence notices name every upstream text, and the wheel carries them.

THE BREAKAGES THESE PREVENT
---------------------------
1. A licence text that NOTICE or THIRD-PARTY-NOTICES.txt points at is not in
   licenses/, or one in licenses/ is named by neither: the reader is sent to
   a file that does not exist, or a text ships that nothing explains.
2. pyproject's license-files leaves a text out, so the wheel's
   dist-info/licenses/ lacks it.  That happened: through 0.3.x the wheel
   carried only LICENSE and NOTICE, while src/hexcore/native_powf.py ships
   Arm's MIT tables whose permission notice lived in LICENSE.arm-powf, which
   the wheel did not carry.
3. The Noah-MP v5.0 port reaches the public distribution, or a private build
   that keeps it presents it under Apache-2.0 alone.  UCAR's Noah-MP terms
   forbid selling it or any work containing it, so the public distribution
   withholds it (owner ruling 2026-10-14: it stays Apache-2.0 and sellable), NOTICE
   says so, and a build that keeps it carries the terms, the notices they
   require and a module header saying those terms govern it.
4. The licence holds stop matching the tree: a hold names a path that was
   renamed, or a tier list names a held test that the export then drops,
   leaving tests/test_tier_membership.py red only on the export.
5. THIRD-PARTY-NOTICES.txt lists a withheld port as if it shipped.
6. The UGWP ports ship again (owner ruling 2026-10-14: UGWP and the RRFS fork's NOAA GSL physics are treated as
   US Government code) while NOTICE or THIRD-PARTY-NOTICES.txt still call them withheld, or a hold still keeps
   them home: the reader is told the opposite of what the distribution carries.

Pure file reads; no device, no network, no port imports.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LICENSES = ROOT / "licenses"
NOTICE = ROOT / "NOTICE"
THIRD_PARTY = ROOT / "THIRD-PARTY-NOTICES.txt"
NOAHMP_CLAUSE = ("You shall not sell, license or transfer for a fee the Software, or any work that in any\n"
                 "manner contains the Software.")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _named_licence_files() -> set[str]:
    names = set()
    for source in (NOTICE, THIRD_PARTY):
        names.update(re.findall(r"licenses/([A-Za-z0-9._-]+\.txt)", _text(source)))
    return names


def _held_texts() -> set[str]:
    """Licence texts that travel only with the ports they cover (private builds)."""
    return {Path(p).name for hold in _holds() for p in hold["paths"] if p.startswith("licenses/")}


def test_every_named_licence_text_exists_and_every_text_is_named() -> None:
    named = _named_licence_files()
    present = {p.name for p in LICENSES.glob("*.txt")}
    assert named, "NOTICE and THIRD-PARTY-NOTICES.txt name no licence text"
    assert named - _held_texts() - present == set(), (
        f"named but missing from licenses/: {sorted(named - _held_texts() - present)}")
    assert present - named == set(), (
        f"in licenses/ but named by neither NOTICE nor THIRD-PARTY-NOTICES.txt: "
        f"{sorted(present - named)}")


def test_the_wheel_carries_every_licence_text() -> None:
    project = tomllib.loads(_text(ROOT / "pyproject.toml"))["project"]
    declared = project.get("license-files", [])
    for required in ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.txt", "LICENSE.arm-powf"):
        assert required in declared, (
            f"license-files does not ship {required}; the wheel's dist-info/licenses/ "
            "would lack it")
        assert (ROOT / required).is_file(), f"{required} is declared but missing"
    assert "licenses/*" in declared, (
        "license-files does not ship licenses/*: the upstream texts NOTICE relies on "
        "would be missing from the wheel")


def test_the_noahmp_v5_port_is_withheld_or_carries_its_terms() -> None:
    notice = re.sub(r"\s+", " ", _text(NOTICE))
    assert "Noah-MP v5.0 (UCAR) -- withheld; not in this distribution" in notice
    assert re.sub(r"\s+", " ", NOAHMP_CLAUSE) in notice, "NOTICE no longer quotes the clause it withholds for"
    assert "--licence-opt-in ucar-noahmp" in notice
    hold = next(h for h in _holds() if h["id"] == "ucar-noahmp-v5")
    module = ROOT / next(p for p in hold["paths"] if p.startswith("src/"))
    copy = LICENSES / "LICENSE-Noah-MP-UCAR.txt"
    private = LICENSES / "NOTICE-Noah-MP-UCAR-private-builds.txt"
    if not module.exists():
        assert not copy.exists() and not private.exists(), (
            "a distribution without the Noah-MP v5.0 port carries its terms, so it reads as if it shipped it")
        return
    assert NOAHMP_CLAUSE in _text(copy), "the Noah-MP licence copy lost its no-sale clause"
    terms = re.sub(r"\s+", " ", _text(private))
    assert "UCAR's Noah-MP terms" in terms and "not by the Apache License 2.0" in terms
    assert "Research Applications Laboratory" in terms, "the notice section 9 of the terms requires is missing"
    docstring = re.sub(r"\s+", " ", module.read_text(encoding="utf-8").split('"""')[1])
    assert "LICENSE-Noah-MP-UCAR.txt" in docstring and "not by the Apache License 2.0" in docstring, (
        "the Noah-MP port's module header no longer says that UCAR's Noah-MP terms govern it")


def test_third_party_lists_withheld_ports_only_as_withheld() -> None:
    """Every held src/ or tests/ path that THIRD-PARTY names sits in a 'Withheld' block."""
    text = _text(THIRD_PARTY)
    blocks = re.split(r"\n(?=\S)|\n\n", text)
    names = {Path(p.rstrip("/")).name for hold in _holds() for p in hold["paths"]
             if p.startswith(("src/", "tests/"))}
    shipped_text = "\n".join(block for block in blocks if "Withheld from the public distribution" not in block)
    listed = sorted(name for name in names if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", shipped_text))
    assert listed == [], f"THIRD-PARTY-NOTICES.txt lists withheld paths as shipped: {listed}"


def test_the_helper_crate_notice_follows_the_bridge_pin() -> None:
    """The bundled bridges' crate texts come from the commit the bridges are pinned to.

    THE BREAKAGE: the bridges are re-pinned to another woof commit (or its
    Cargo.lock moves) and licenses/THIRD-PARTY-LICENSES-helper-binaries.txt
    still lists the old commit's crates, so the wheel ships texts for crates
    it no longer links and none for those it now does (readiness B12).
    Rerun tools/helper_licences.py against the new commit.
    """
    text = _text(LICENSES / "THIRD-PARTY-LICENSES-helper-binaries.txt")
    pin_bytes = (ROOT / ".." / ".." / "woof" / "hex" / "data" / "engine-bridges.json").read_bytes()
    pins = json.loads(pin_bytes)
    assert "Written by tools/helper_licences.py" in text
    commit = re.search(r"^  woof commit +([0-9a-f]{40})$", text, re.M)
    lock = re.search(r"^  tools/rustwx/Cargo\.lock +sha256 ([0-9a-f]{64})$", text, re.M)
    # Entries can override the manifest default. rw_mpas_stream does so:
    # treating the old top-level pin as every binary's build identity rejected
    # a correctly generated notice for the actual release build workspace.
    identities = {(pins['gpuwm_source_rev'], pins['cargo_lock_sha256'])}
    identities.update((row.get('gpuwm_source_rev', pins['gpuwm_source_rev']),
                       row.get('cargo_lock_sha256', pins['cargo_lock_sha256']))
                      for row in pins['bridges'].values())
    # A source prebuilt built from its own engine commit (rw_sfc_series keeps its aaec6b4 build after the
    # bridges moved to woof 2.9.7) is the notice's resolving engine when it names its own commit and lock.
    identities.update((row['gpuwm_source_rev'], row['cargo_lock_sha256'])
                      for row in pins['source_prebuilts']['files'].values()
                      if 'gpuwm_source_rev' in row and 'cargo_lock_sha256' in row)
    pin_hash = re.search(r"^  engine-bridges\.json +sha256 ([0-9a-f]{64})$", text, re.M)
    assert pin_hash and pin_hash.group(1) == hashlib.sha256(pin_bytes).hexdigest(), "regenerate the helper crate notice"
    assert commit and lock and (commit.group(1), lock.group(1)) in identities, "regenerate the helper crate notice"
    for crate in ("rw-mpas", "hdf5-reader", "netcdf-reader", "serde_json"):
        assert re.search(rf"^  {re.escape(crate)} ", text, re.M), f"{crate} is not in the notice"


def test_notice_reproduces_the_mpas_licence_byte_for_byte() -> None:
    copy = _text(LICENSES / "LICENSE-MPAS-BSD-3-Clause.txt")
    body = copy.split("------------------------------------------------------------------------\n\n", 1)[1]
    assert body.startswith("Copyright (c) 2013-2019,  Los Alamos National Security")
    assert body in _text(NOTICE), "NOTICE's MPAS licence text differs from licenses/ copy"


def _holds() -> list[dict]:
    data = json.loads(_text(ROOT / "tools" / "licence_holds.json"))
    assert data["schema"] == "gpuwm-hex.licence-holds/v1"
    return data["holds"]


def test_every_hold_states_why_and_when_it_ends() -> None:
    for hold in _holds():
        for key in ("id", "upstream", "upstream_commit", "reason", "release_when", "paths"):
            assert hold.get(key), f"hold {hold.get('id')!r} lacks {key}"
        assert hold["upstream_commit"] in _text(NOTICE), (
            f"NOTICE does not name the upstream commit of hold {hold['id']}")


def test_held_paths_are_either_all_present_or_all_absent() -> None:
    """Internal tree: every held path exists.  Export: none does.  Mixed is a rename."""
    for hold in _holds():
        present = [p for p in hold["paths"] if (ROOT / p.rstrip("/")).exists()]
        assert present in ([], hold["paths"]), (
            f"hold {hold['id']}: only {present} of {hold['paths']} exist; a held path "
            "was renamed or removed without updating tools/licence_holds.json")


def test_the_export_tool_drops_exactly_the_tier_rows_of_held_tests() -> None:
    spec = importlib.util.spec_from_file_location("licence_export", ROOT / "tools" / "licence_export.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    paths = tool.held_paths(_holds())
    held_tests = {p for p in paths if p.startswith("tests/") and p.endswith(".py")}
    rows = set()
    for tier in tool.TIER_LISTS:
        path = ROOT / tier
        if path.is_file():
            for line in _text(path).splitlines():
                entry = line.strip()
                if entry and not entry.startswith("#") and tool.is_held(entry, paths):
                    rows.add(entry)
    held_folders = [p for p in paths if p.startswith("tests/") and p.endswith("/")]
    if all((ROOT / p).exists() for p in held_tests):
        assert held_tests <= rows, (
            f"held test files {sorted(held_tests - rows)} are in no tier list; the export's tier "
            "check would not see them dropped")
        stray = sorted(r for r in rows - held_tests if not any(r.startswith(f) for f in held_folders))
        assert stray == [], f"tier rows naming held paths outside the held files and folders: {stray}"
    else:
        assert rows == set(), f"the export still names held tests: {sorted(rows)}"


def test_the_ugwp_ports_ship_as_us_government_code() -> None:
    held = {p for hold in _holds() for p in hold["paths"]}
    ports = ("src/hexcore/mpasphys/ugwp_oro_v842.py", "src/hexcore/mpasphys/ugwp_ngw_v842.py",
             "tests/test_mpas_ugwp_oro_v842.py", "tests/test_ugwp_ngw_v842.py",
             "tests/ugwp_ngw_native/", "tests/ugwp_oro_oracle/")
    assert held.isdisjoint(ports), f"a licence hold still keeps UGWP home: {sorted(held & set(ports))}"
    assert all(hold["id"] not in ("ugwp", "noaa-gsl-ugwp") for hold in _holds())
    assert all((ROOT / port.rstrip("/")).exists() for port in ports), "a UGWP port or its tests are missing"
    notice = re.sub(r"\s+", " ", _text(NOTICE))
    assert "8. NOAA GSL Unified Gravity Wave Physics (UGWP) -- US Government code" in notice
    assert "c1c893edcf171af5639af60e3a3a528816f6cc2b" in notice, "NOTICE lost UGWP's upstream commit"
    assert "noaa-gsl-unlicensed" not in notice and "ugwp-held" not in notice, "NOTICE names a retired opt-in"
    assert "UGWP ports are withheld" not in notice and "excludes the held UGWP" not in notice
    third = _text(THIRD_PARTY)
    shipped = "\n".join(block for block in re.split(r"\n(?=\S)|\n\n", third)
                        if "Withheld from the public distribution" not in block)
    for name in ("ugwp_oro_v842.py", "ugwp_ngw_v842.py", "tests/ugwp_oro_oracle/"):
        assert name in shipped, f"THIRD-PARTY-NOTICES.txt does not list {name} as shipped"
