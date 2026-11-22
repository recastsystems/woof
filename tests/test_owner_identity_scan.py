"""The release packet's owner-identity gate reads every member of every asset.

The breakage it prevents: the owner's e-mail address or home directory
published on PyPI or a GitHub release (owner ruling 2026-09-10).  Nested
archives are opened and binaries are read as bytes.  The gate lives in the
private cut tooling, which the public snapshot does not carry, so a published
tree skips this module.

The owner's first name as a word is NOT read, in any spelling.  It was from
2.9.0 cut r7 to 2.9.2 with a 264-phrase allowance for the past tense of
"draw", and the 2.9.2 cut refused a source archive over nine renderer
sentences.  It was retired on 2026-10-17.  :func:`test_the_owners_first_name_is_a_word_not_a_hit`
keeps it retired; it replaces test_the_name_is_read_in_every_case,
test_an_allowance_forgives_only_its_exact_phrase and
test_the_allowance_names_a_reason_for_every_phrase_and_no_identity, and
:func:`test_other_users_homes_are_not_the_owner` replaces
test_words_and_other_users_are_not_the_owner.

Every identity string below comes from the never-shipped
work/release_private_markers.py (``private_test_controls``): this file ships
in the sdist, where the packet gate reads it, and must carry none of them.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVERS = REPO / "tools" / "release" / "cut" / "drivers"
SCANNER = DRIVERS / "owner_identity_scan.py"

from tools.release_exclusions import private_test_controls

_CONTROLS = private_test_controls()

pytestmark = pytest.mark.skipif(
    not SCANNER.is_file() or not _CONTROLS,
    reason="the release cut tooling is not in this tree (published snapshot)")

NAME = _CONTROLS.get("owner_name", "")
USER = NAME.lower()
CAPS = USER.upper()
MAIL = _CONTROLS.get("owner_mail", "")
BS = chr(92)


def _scanner():
    spec = importlib.util.spec_from_file_location("owner_identity_scan", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _wheel(path: Path, members: dict[str, bytes]) -> Path:
    path.write_bytes(_zip_bytes(members))
    return path


def _sdist(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_every_identity_shape_is_found_in_code_data_and_binaries(tmp_path):
    scan = _scanner()
    wheel = _wheel(tmp_path / "pkg-1.0-py3-none-any.whl", {
        "pkg/receipt.json": ('{"by": "' + MAIL + '"}\n').encode(),
        "pkg/plan.json": ('{"dir": "C:' + BS * 2 + "Users" + BS * 2 + USER
                          + BS * 2 + 'deck"}\n').encode(),
        "pkg/bin/tool": (b"\x7fELF\x00\x01/ho" + b"me/" + USER.encode()
                         + b"/.cargo/registry/src/x.rs\x00\x02"),
        "pkg/mnt.txt": ("/mnt/c/Users/" + USER + "/notes\n").encode(),
        "pkg/clean.py": b"x = 1\n",
    })
    rows = scan.scan_asset(wheel)
    found = sorted((row["member"], row["kind"]) for row in rows)
    assert found == [
        ("pkg/bin/tool", "owner's home directory"),
        ("pkg/mnt.txt", "owner's home directory"),
        ("pkg/plan.json", "owner's home directory"),
        ("pkg/receipt.json", "owner's e-mail"),
    ]
    assert all(row["asset"] == wheel.name and row["where"] == "member-body" for row in rows)


def test_the_owners_first_name_is_a_word_not_a_hit(tmp_path):
    """Owner ruling 2026-10-17: the first name, in any spelling, is never read.

    Every shape the retired rule matched: the capitalised word in a comment,
    the all-capitals word in a native binary, the lower-case word as a verb,
    inside an identifier and in a holder string, and in a member name.
    """

    scan = _scanner()
    wheel = _wheel(tmp_path / "pkg-1.0-py3-none-manylinux_2_28_x86_64.whl", {
        "pkg/namelist_chem.py": ("# " + NAME + "'s ruling (2026-10-12)\n").encode(),
        "pkg/libexec/bridges/rw_static": (
            b"\x7fELF\x00\x00EXCLUSIONS PROPOSED FOR " + CAPS.encode()
            + b": case-specific sign-off required\x00"),
        "pkg/render.py": ("# the renderer " + USER + " 43 pictures\n").encode(),
        "pkg/ident.py": ("x_" + USER + "_y = 1\n").encode(),
        "pkg/lane.py": ('HOLDER = "' + USER + '-gui"\n').encode(),
        "pkg/notes/" + NAME + "-ruling.md": b"text\n",
    })
    assert scan.scan_asset(wheel) == []
    assert not hasattr(scan, "NAME") and not hasattr(scan, "load_allowance")
    assert not (DRIVERS / "owner_identity_allow.json").exists()


def test_other_users_homes_are_not_the_owner(tmp_path):
    scan = _scanner()
    sdist = _sdist(tmp_path / "pkg-1.0.tar.gz", {
        "pkg-1.0/paths.txt": ("/ho" + "me/somebody/wrf\n/ho" + "me/" + USER
                              + "ess/x\nC:/Users/" + USER + "lab/y\n").encode(),
        "pkg-1.0/shell.py": b'NAMES = ["owner\'s case", "Kai\\u2019s case"]\n',
    })
    assert scan.scan_asset(sdist) == []


def test_archives_inside_members_are_opened(tmp_path):
    """A wheel inside a tarball, an .npz and a .gz inside a wheel: their members are read."""

    scan = _scanner()
    home = ("/ho" + "me/" + USER + "/x").encode()
    inner = _zip_bytes({"pkg/bin/tool": b"\x00" + home + b"\x00"})
    npz = _zip_bytes({"table.npy": b"\x93NUMPY\x00" + MAIL.encode() + b"\x00"})
    wheel = _zip_bytes({"pkg/data/table.npz": npz,
                        "pkg/data/rows.csv.gz": gzip.compress(b"a," + home + b",b\n")})
    outer = _sdist(tmp_path / "Desktop-1.0-Linux-x64-pip.tar.gz", {
        "Desktop-1.0/wheels/pkg-1.0-py3-none-any.whl": inner,
        "Desktop-1.0/wheels/data-1.0-py3-none-any.whl": wheel,
    })
    rows = scan.scan_asset(outer)
    assert sorted(row["member"] for row in rows) == [
        "Desktop-1.0/wheels/data-1.0-py3-none-any.whl!pkg/data/rows.csv.gz!<gunzipped>",
        "Desktop-1.0/wheels/data-1.0-py3-none-any.whl!pkg/data/table.npz!table.npy",
        "Desktop-1.0/wheels/pkg-1.0-py3-none-any.whl!pkg/bin/tool",
    ]


def test_member_names_are_read_as_well_as_bodies(tmp_path):
    scan = _scanner()
    sdist = _sdist(tmp_path / "pkg-1.0.tar.gz", {
        "pkg-1.0/ho" + "me/" + USER + "/notes.md": b"text\n"})
    rows = scan.scan_asset(sdist)
    assert [(row["where"], row["kind"]) for row in rows] == [
        ("member-name", "owner's home directory")]


def test_the_command_line_exits_nonzero_on_a_hit(tmp_path, capsys):
    scan = _scanner()
    clean = _wheel(tmp_path / "clean-1.0-py3-none-any.whl", {"a/b.py": b"x = 1\n"})
    dirty = _wheel(tmp_path / "dirty-1.0-py3-none-any.whl",
                   {"a/b.py": ("# " + MAIL + "\n").encode()})
    assert scan.main([str(clean)]) == 0
    assert scan.main([str(clean), str(dirty)]) == 1
    assert '"hits": 1' in capsys.readouterr().out


def test_the_gate_is_wired_into_the_final_packet_verdict():
    """A hit must BLOCK the packet, not merely be recorded."""

    source = (DRIVERS / "verify_final_packet.py").read_text(encoding="utf-8")
    assert "import owner_identity_scan as owner_scan" in source
    assert "owner_rows.extend(owner_scan.scan_asset(path))" in source
    assert "ALLOWANCE" not in source and "owner_forgiven" not in source
    assert "if owner_rows:\n    blockers.append({'gate': 'owner_identity'," in source


def test_the_built_wheels_are_read_with_the_same_gate():
    """packages.py reads each wheel and sdist the moment it is built, before the packet."""

    source = (REPO / "tools" / "release" / "cut" / "packages.py").read_text(encoding="utf-8")
    assert "from .drivers.owner_identity_scan import scan_asset" in source
    assert "if hits:\n        raise ValueError('built artifacts contain private identity" in source
