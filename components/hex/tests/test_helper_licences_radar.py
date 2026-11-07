"""tools/helper_licences.py reads crates the source notice inventories only in its radar appendix.

THE BREAKAGE THIS PREVENTS (hex 0.4.0 repin to gpuwm 2.9.0): at the 2.9.0 release commit rw-mpas links
libloading 0.9.0, so the observation scorer publish.yml builds against the declared floor links it.  The
engine's bridge notice carries libloading's ISC text only in its WOOF simulated radar appendix, so the tool
stopped with "linked but not inventoried in the source notice" and the wheel's helper notice could not
carry that text.  CPU tier: a synthetic notice, no cargo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("helper_licences", ROOT / "tools" / "helper_licences.py")
helper_licences = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_licences)

RULE, DASH = "=" * 78, "-" * 78
SHA_A, SHA_B, SHA_R = "a" * 64, "b" * 64, "c" * 64


def _notice(radar_count: int = 1) -> str:
    return (
        f"HEAD\n{RULE}\n2.  NOTICES THAT ARE NOT CRATE LICENCES\n{DASH}\n\n2.1  Something\n\npara\n\n"
        f"{RULE}\n3.  VENDORED RUST CRATES -- INVENTORY\n{DASH}\n\n"
        "2 distinct crate name-and-version pairs across\n\n"
        "  libc 0.2.189   MIT OR Apache-2.0   [1,2]\n"
        "  libm 0.2.16    MIT                 [2]\n\n"
        "Crates that vendored no licence text of their own:\n\n  none\n\n"
        f"{RULE}\n4.  VENDORED RUST CRATES -- LICENCE TEXTS\n{DASH}\n\n2 distinct texts follow.\n\n"
        f"{DASH}\n[1]  as carried by: libc 0.2.189\n     sha256 {SHA_A}\n{DASH}\nApache text\n\n"
        f"{DASH}\n[2]  as carried by: libc 0.2.189, libm 0.2.16\n     sha256 {SHA_B}\n{DASH}\nMIT text\n\n"
        f"{RULE}\nWOOF SIMULATED RADAR -- ADDED DEPENDENCY LICENCE TEXTS\n\n"
        "Packages and grant expressions:\n"
        "  wrf-core 0.1.0 | MIT | []\n"
        "    https://example.invalid/pyproject.toml\n"
        "    declares MIT in pyproject.toml.\n"
        "  libloading 0.9.0 | ISC | [RADAR-DEPS-028]\n"
        "  libc 0.2.189 | MIT | [RADAR-DEPS-028]\n\n"
        f"{radar_count} distinct grant and attribution texts follow.\n\n"
        f"{DASH}\n[RADAR-DEPS-028] sha256 {SHA_R}\nAs carried by:\n"
        "  tools/rustwx/vendor/crates-io/libloading-0.9.0/LICENSE\n"
        f"{DASH}\n\nCopyright (c) 2015, Simonas Kazlauskas\n\nISC text\n\n\n"
        f"{RULE}\nEND WOOF SIMULATED RADAR DEPENDENCY LICENCE TEXTS\n\n"
        f"{RULE}\n5.  WOOF TUI\n"
    )


def test_a_crate_only_the_radar_appendix_inventories_is_read_with_its_text():
    notice = helper_licences.parse_notice(_notice())
    assert notice["inventory"][("libloading", "0.9.0")] == ("ISC", ("RADAR-DEPS-028",))
    sha, body = notice["texts"]["RADAR-DEPS-028"]
    assert sha == SHA_R and body.startswith("Copyright (c) 2015") and body.endswith("ISC text\n")
    # section 3 wins where both list a crate; numbered texts are untouched
    assert notice["inventory"][("libc", "0.2.189")] == ("MIT OR Apache-2.0", (1, 2))
    assert notice["texts"][2] == (SHA_B, "MIT text\n")
    # a radar row with no text keeps its stated reason
    assert notice["untexted"][("wrf-core", "0.1.0")].startswith("https://example.invalid")


def test_numbered_texts_sort_before_named_ones():
    assert sorted({"RADAR-DEPS-028", 2, 1}, key=helper_licences._ref_key) == [1, 2, "RADAR-DEPS-028"]


def test_an_appendix_that_disagrees_with_its_own_count_stops_the_tool():
    with pytest.raises(SystemExit, match="radar appendix"):
        helper_licences.parse_notice(_notice(radar_count=2))
