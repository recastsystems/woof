"""What a forecast prints and writes names each gap and each service in words a user can follow.

THE BREAKAGE: 0.6.0 printed a bare development label as a correct-mode gap's owner on every MPAS-suite forecast and
wrote it into the run folder's receipts, and its finished forecast's record line named the surface service by a
label of the same kind.  The correction tables now say, in a user's words, which part of hex closes each gap, and the
surface service names the MPAS service it ports.  These tests read every owner the tables carry at run time and build
the record the way a forecast builds it (its physics block from the resolved suite, its surface service block from
the service's own receipt), so a bare label put back in either table or in the service fails them.

The maintainers' release tools keep a further list of shapes that must not appear in printed text; when
``HEX_INTERNAL_SHAPES`` names that file, the same records, the regional classes' printed evidence, the derivative
door table and the wheel's sources are read against it as well.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import warnings

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import correct_mode  # noqa: E402
from woof.hex.cuda_backend import regional_admission  # noqa: E402
from woof.hex.mpasphys import driver_v842, surface_update_v842  # noqa: E402
from woof.hex.mpasphys.rrfs import driver_rrfs, fork_corrections  # noqa: E402


def _shapes(group):
    """The compiled shapes of one group of the release tools' pattern file, or None when it is not named."""
    path = os.environ.get("HEX_INTERNAL_SHAPES")
    if not path:
        return None
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [re.compile(item["re"], re.I if item.get("i") else 0) for item in doc["guard"][group]]


def _hits(text, group):
    shapes = _shapes(group)
    if shapes is None:
        return []
    return sorted({m.group(0) for shape in shapes for m in shape.finditer(text)})


def _need(group):
    if _shapes(group) is None:
        pytest.skip("HEX_INTERNAL_SHAPES does not name the release tools' pattern file")


def _deck(suite, mode="correct"):
    return (f"&nhyd_model\nconfig_physics_mode = {mode!r}\n/\n"
            f"&physics\nconfig_physics_suite = {suite!r}\n/")


def _owners():
    """Every owner string the correction tables carry (the MPAS suite table, the routes outside it, the RRFS fork
    table, the restart gap and the RRFS suite row's inventory gap)."""
    names = set()
    for row in driver_v842.CORRECTIONS_V842.values():
        for value in (row.get("owners") or {}).values():
            names |= set(value.values()) if isinstance(value, dict) else {value}
        route = row.get("hex_route") or {}
        if route.get("owner"):
            names.add(route["owner"])
    for row in correct_mode.FINDINGS.values():
        if row.get("owner"):
            names.add(row["owner"])
    names.add(fork_corrections.OWNER)
    names.add(driver_v842.RP6_RESTART_GAP["owner"])
    rrfs = driver_rrfs.route_namelist("&physics\nconfig_physics_suite = 'hrrrv5'\n/", physics_mode="correct")
    names |= {gap.owner for gap in rrfs.gaps}
    return {str(name) for name in names if name}


def _surface_service_block():
    """The ``surface_service`` block a finished MPAS-suite forecast writes: the service's own receipt, after one call
    of each operation (the forecast calls ``receipt()`` on its session's service)."""
    service = SimpleNamespace(module_sha256="0" * 64,
                              executions={name: 1 for name in surface_update_v842.NativeSurfaceService.OPERATIONS})
    return surface_update_v842.NativeSurfaceService.receipt(service)


def _forecast_record():
    """A finished MPAS-suite forecast's record, built as runtime_v842 builds it: ``physics`` is the resolved suite's
    receipt (its ``correct_mode`` block with the gap rows), ``surface_service`` the service's receipt."""
    config = driver_v842.resolve_namelist(_deck("convection_permitting"))
    return {"schema": "mpas-native-forecast-v1", "physics": config.receipt(),
            "surface_service": _surface_service_block()}


def _bare_label(text):
    """A single token with no space: what a user cannot follow as the name of a part of hex."""
    return " " not in text.strip()


def test_every_owner_in_the_tables_is_plain_words():
    owners = _owners()
    assert len(owners) >= 8, owners
    for owner in owners:
        assert not _bare_label(owner), f"{owner!r} is a bare label, not words a user can follow"
        assert not _hits(owner, "printed"), owner


def test_the_surface_service_names_the_mpas_service():
    block = _surface_service_block()
    assert block["unit"] == surface_update_v842.SERVICE_UNIT and block["backend"] == "hex-cuda"
    assert not _hits(json.dumps(block), "printed"), _hits(json.dumps(block), "printed")


def test_the_finished_forecast_line_and_its_receipt_name_each_gap_in_words():
    record = _forecast_record()
    gaps = record["physics"]["correct_mode"]["gaps"]
    assert gaps and all(row["owner"] for row in gaps), "no gap row with an owner: this test reads nothing"
    written = json.dumps(record, default=str)
    printed = json.dumps(correct_mode.terminal_record(record), default=str)
    assert '"surface_service"' in printed and '"owner"' not in printed
    for owner in _owners():
        assert not _bare_label(owner) or (owner not in printed and owner not in written), owner
    assert _hits(printed, "printed") == [], ("the terminal line", _hits(printed, "printed"))
    assert _hits(written, "printed") == [], ("the receipt file", _hits(written, "printed"))
    assert record["physics"]["correct_mode"]["gaps"][0]["owner"], "the record itself is not changed"


def test_the_mpas_suite_warning_names_the_gap_and_its_reason():
    printed = 0
    for suite in ("mesoscale_reference", "convection_permitting"):
        config = driver_v842.resolve_namelist(_deck(suite))
        warning = config.gap_warning()
        if warning is None:
            continue
        printed += 1
        assert "owner" not in warning.lower() and not _hits(warning, "printed"), warning
        assert all(row["owner"] for row in config.correct_mode_record()["gaps"])
    assert printed, "no MPAS suite printed a correct-mode gap: this test reads nothing"


def test_the_rrfs_suite_warning_and_receipt_name_each_gap_in_words():
    config = driver_rrfs.route_namelist("&physics\nconfig_physics_suite = 'hrrrv5'\n/", physics_mode="correct")
    warning = config.gap_warning()
    assert warning and "CORRECT MODE: named gaps" in warning
    assert "owner" not in warning.lower() and not _hits(warning, "printed"), warning
    assert not _hits(json.dumps(config.receipt(), default=str), "printed")


def test_the_arwen_column_route_prints_the_gap():
    finding = next(name for name, row in correct_mode.FINDINGS.items() if row.get("owner"))
    stream = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stderr(stream):
            rows = correct_mode.report_gaps([correct_mode.CorrectionGap(finding, "a test column", "not built here")])
    printed = stream.getvalue()
    assert finding in printed and "NOT APPLIED" in printed
    assert "owner" not in printed.lower() and not _hits(printed, "printed"), printed
    assert rows[0]["owner"], "the receipt row keeps its owner field"


def _regional_texts():
    """Every text a regional forecast prints or writes about the class and the geometry it was admitted on."""
    texts = {}
    for cid, klass in regional_admission.ADMITTED_CLASSES.items():
        texts[f"class {cid}"] = json.dumps(klass.as_dict() if hasattr(klass, "as_dict") else klass.basis, default=str)
        texts[f"class {cid} basis"] = klass.basis
    for digest, contract in regional_admission.SHIPPED_CONTRACTS.items():
        texts[f"geometry {digest[:12]}"] = contract.basis
    return texts


def test_every_regional_class_states_its_measurement():
    texts = _regional_texts()
    assert len(texts) >= 12, sorted(texts)
    for name, text in texts.items():
        if name.endswith(" basis") and name.startswith("class "):
            assert "22 boundary kernels" in text and "byte-identity" in text, name


def test_every_regional_class_text_names_nothing_internal():
    _need("printed")
    found = {name: _hits(text, "printed") for name, text in _regional_texts().items()}
    assert {k: v for k, v in found.items() if v} == {}


def test_the_derivative_runner_writes_its_approximation_only():
    # The runner once wrote an internal decision note, with its date, into every window.json's `approximation`
    # field. The runner sources are what the wheel compiles into woof/hex/libexec/ad-binder; the runner's bytes follow
    # its source text, comments included, so its comments change only with a runner rebuilt on purpose.
    window_rs = ROOT / "rust" / "hexad" / "src" / "runner" / "window.rs"
    if not window_rs.is_file():
        import pytest
        pytest.skip("the runner's Rust source is not part of this tree (its ad-binder ships prebuilt)")
    window = window_rs.read_text(encoding="utf-8")
    assert "fct_minmax_source differentiates source_old only. This operator is not the exact derivative" in window
    _need("runner")
    sources = [ROOT / "rust" / "hexad" / "src", ROOT / "ad" / "phys" / "binder" / "runtime" / "src"]
    read = 0
    for base in sources:
        for path in base.rglob("*.rs"):
            text = path.read_text(encoding="utf-8", errors="replace")
            read += 1
            assert not _hits(text, "runner"), path
    assert read > 10, "no runner source was read"


def test_the_door_table_prints_nothing_internal():
    # The derivative doors print their rows' measured sentences (where, source, speed, why) and copy them into receipts.
    # File and folder names inside the pinned downloads, and kernel-set names, are part of published asset names.
    _need("printed")
    text = (ROOT / ".." / ".." / "woof" / "hex" / "data" / "ad-doors.v1.json").read_text(encoding="utf-8")
    text = re.sub(r"kernel sets? [\w.-]+|[\w.-]+\.tar\.gz|[\w.-]*/[\w./-]*", " ", text)
    assert _hits(text, "printed") == []


def test_every_forecast_printer_uses_the_terminal_record():
    from woof.hex import forecast_door
    package = Path(forecast_door.__file__).resolve().parent   # the package under test, wherever it is laid out
    for rel in ("mpasphys/runtime_v842.py", "cuda_regional_forecast_v842.py", "forecast_door.py"):
        text = (package / rel).read_text(encoding="utf-8")
        assert "print(json.dumps(terminal_record(receipt)" in text, rel
        assert "print(json.dumps(receipt" not in text, rel


#: The one shipped module whose text is attested by hash against its generator (tests/rrfs_gfl/attest_generated.py);
#: it changes only with a regeneration.
_GENERATED = {"src/hexcore/mpasphys/rrfs/cu_gfl_bf6a4b0.py"}


def test_the_wheel_sources_name_nothing_internal():
    # Every Python module and table the wheel carries (the compiled kernels' recipe tables hold kernel source, whose
    # bytes are pinned by the kernel packs).
    _need("sources")
    found = {}
    for path in sorted((ROOT / ".." / ".." / "woof" / "hex").rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in (".py", ".json", ".toml") or "compiler-recipes" in path.name or rel in _GENERATED:
            continue
        hits = _hits(path.read_text(encoding="utf-8", errors="replace"), "sources")
        if hits:
            found[rel] = hits
    assert found == {}, found
