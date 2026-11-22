"""The publish-time platform-tag check refuses a wheel older hosts cannot install.

THE BREAKAGE (a release review of 0.4.0, 2026-10-12): a 0.4.0 build's wheel
was tagged ``manylinux_2_35_x86_64`` because it was built on a glibc 2.35+
host, so pip on glibc 2.28-2.34 hosts skipped it and fell back to an sdist
build that needs cargo and an engine checkout.  ``tools/check_wheel_platform.py``
runs in publish.yml after auditwheel; these tests pin what it refuses.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_wheel_platform.py"
WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "publish.yml"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("_check_wheel_platform_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_elf(glibc: str | None) -> bytes:
    tail = b"" if glibc is None else b"\0GLIBC_2.2.5\0GLIBC_" + glibc.encode() + b"\0"
    return b"\x7fELF\x02\x01\x01" + b"\0" * 64 + tail


def make_wheel(tmp_path: Path, platform: str, *, elf_glibc: dict[str, str | None] | None = None,
               tag_lines: list[str] | None = None, mode: int = 0o755, pure: bool = False) -> Path:
    name = f"gpuwm_hex-0.4.0-py3-none-{platform}.whl"
    path = tmp_path / name
    tags = tag_lines if tag_lines is not None else [f"py3-none-{p}" for p in platform.split(".")]
    wheel = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: {}\n{}\n".format(
        "true" if pure else "false", "\n".join(f"Tag: {t}" for t in tags))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("woof/hex/__init__.py", "")
        archive.writestr("gpuwm_hex-0.4.0.dist-info/WHEEL", wheel)
        for member, glibc in (elf_glibc or {"hex_contract_hostprep": "2.17"}).items():
            info = zipfile.ZipInfo(f"woof/hex/libexec/{member}")
            info.external_attr = (0o100000 | mode) << 16
            archive.writestr(info, fake_elf(glibc))
    return path


def test_a_manylinux_2_28_wheel_with_old_enough_helpers_passes(check, tmp_path):
    wheel = make_wheel(tmp_path, "manylinux_2_28_x86_64",
                       elf_glibc={"hex_contract_hostprep": "2.17", "hex_obs_scorecard": "2.28"})
    report = check.check_wheel(wheel, "manylinux_2_28_x86_64", ["hex_contract_hostprep", "hex_obs_scorecard"])
    assert report.passed, report.findings
    assert report.elf["woof/hex/libexec/hex_obs_scorecard"] == "2.28"


def test_the_runner_glibc_tag_of_the_0_4_0_candidate_is_refused(check, tmp_path):
    wheel = make_wheel(tmp_path, "manylinux_2_35_x86_64", elf_glibc={"hex_contract_hostprep": "2.35"})
    report = check.check_wheel(wheel, "manylinux_2_28_x86_64", [])
    assert not report.passed
    assert any("newer than the manylinux_2_28_x86_64 policy" in f for f in report.findings)


def test_a_bare_linux_tag_is_refused(check, tmp_path):
    report = check.check_wheel(make_wheel(tmp_path, "linux_x86_64"), "manylinux_2_28_x86_64", [])
    assert any("is not manylinux" in f for f in report.findings)


def test_a_tag_older_than_its_own_helper_is_refused(check, tmp_path):
    # auditwheel would never write this, but a hand retag would: pip installs
    # it on glibc 2.17 and the helper then fails to load.
    wheel = make_wheel(tmp_path, "manylinux2014_x86_64", elf_glibc={"hex_contract_hostprep": "2.28"})
    report = check.check_wheel(wheel, "manylinux_2_28_x86_64", [])
    assert any("needs glibc 2.28 but the wheel claims 2.17" in f for f in report.findings)


def test_a_missing_or_non_executable_libexec_helper_is_refused(check, tmp_path):
    wheel = make_wheel(tmp_path, "manylinux_2_28_x86_64", mode=0o644)
    report = check.check_wheel(wheel, "manylinux_2_28_x86_64", ["hex_contract_hostprep", "gpuwm-hex-netcdf-copy"])
    assert any("hex_contract_hostprep is not executable" in f for f in report.findings)
    assert any("gpuwm-hex-netcdf-copy is missing" in f for f in report.findings)


def test_a_grafted_shared_library_is_refused_unless_allowed(check, tmp_path):
    # The copier once linked the system libnetcdf and auditwheel grafted 31
    # libraries (HDF5, curl, OpenSSL, Kerberos ...) into the wheel.
    wheel = make_wheel(tmp_path, "manylinux_2_28_x86_64")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("gpuwm_hex.libs/libnetcdf-3915b03a.so.15.0.1", fake_elf("2.14"))
    assert any("1 grafted shared libraries" in f for f in check.check_wheel(wheel, "manylinux_2_28_x86_64", []).findings)
    assert check.check_wheel(wheel, "manylinux_2_28_x86_64", [], allow_grafted=True).passed


def test_wheel_metadata_must_agree_with_the_file_name(check, tmp_path):
    wheel = make_wheel(tmp_path, "manylinux_2_28_x86_64", tag_lines=["py3-none-linux_x86_64"])
    report = check.check_wheel(wheel, "manylinux_2_28_x86_64", [])
    assert any("disagree with the file name" in f for f in report.findings)


def test_a_compressed_tag_set_is_checked_tag_by_tag(check, tmp_path):
    good = make_wheel(tmp_path, "manylinux_2_17_x86_64.manylinux2014_x86_64")
    assert check.check_wheel(good, "manylinux_2_28_x86_64", ["hex_contract_hostprep"]).passed
    bad = make_wheel(tmp_path, "manylinux_2_28_x86_64.manylinux_2_39_x86_64",
                     elf_glibc={"hex_contract_hostprep": "2.28"})
    assert not check.check_wheel(bad, "manylinux_2_28_x86_64", []).passed


def test_the_command_line_fails_on_no_wheels_and_on_a_finding(check, tmp_path, capsys):
    assert check.main([str(tmp_path / "gpuwm_hex-0.4.0.tar.gz")]) == 1
    assert check.main([str(make_wheel(tmp_path, "manylinux_2_35_x86_64"))]) == 1
    assert check.main([str(make_wheel(tmp_path, "manylinux_2_28_x86_64"))]) == 0
    assert "PASS" in capsys.readouterr().out


def test_publish_builds_in_manylinux_2_28_repairs_and_checks_before_upload():
    if not WORKFLOW.is_file():
        pytest.skip("publish.yml is outside this tree (sdist install)")
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "quay.io/pypa/manylinux_2_28_x86_64@sha256:" in text, "the build image must be pinned by digest"
    run = "check_wheel_platform.py dist/*.whl"
    assert "auditwheel repair --plat" in text
    assert text.count(run) >= 2, "check after the repair AND on the downloaded files right before the upload"
    first, last = text.index(run), text.rindex(run)
    assert text.index("auditwheel repair --plat") < first
    assert last < text.index("pypa/gh-action-pypi-publish")
    for helper in ("hex_contract_hostprep", "hex_obs_scorecard", "gpuwm-hex-netcdf-copy", "hex_gpmetis"):
        assert f"--require-libexec {helper}" in text, helper


def test_publish_stamps_the_repaired_wheel_with_one_fixed_date():
    """THE BREAKAGE THIS PREVENTS: two release builds of one commit gave wheels equal in every RECORD entry and
    different in sha256, because the repair step stamped each zip member with the build's clock, so a rebuilt
    wheel could not be told from another build by its hash."""
    if not WORKFLOW.is_file():
        pytest.skip("publish.yml is outside this tree (sdist install)")
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    dates = [line.split(":", 1)[1].strip().strip('"') for line in lines if line.startswith("  WHEEL_DATE_EPOCH:")]
    assert len(dates) == 1 and dates[0].isdigit(), "one fixed date in the workflow's own env, not the clock"
    assert int(dates[0]) >= 315532800, "a zip member cannot carry a date before 1980"
    repairs = [line.strip() for line in lines if "auditwheel repair --plat" in line]
    assert repairs == ['SOURCE_DATE_EPOCH="${WHEEL_DATE_EPOCH}" auditwheel repair --plat "${MANYLINUX_POLICY}" '
                       '--wheel-dir dist raw/*.whl'], repairs
