"""Static gates for the bit-preserving capacity copy-elision patch.

These tests do not make a hardware or output-equivalence claim. They ensure the
specific read-only bindings, named refusal, and exact source-pin migration do
not silently disappear before the required CUDA A/B runs are performed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from _layout import PACKAGE_DIR


ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_DIR

EXPECTED_SOURCE_SHA256 = {
    # Moved by the NVRTC reciprocal-rewrite lane: the flux3/flux4
    # denominator in the three vertical-flux kernels is a translation-unit
    # constant rather than a source literal, so NVRTC cannot rewrite the
    # division as a reciprocal multiply on a compute_100-or-above target.
    # Moved again 2026-08-26 by the convection ruling: the per-component
    # physics cadence table reports convection: None when no cumulus scheme
    # is selected.  The reciprocal-rewrite migration this table guards is
    # untouched -- the flux3/flux4 denominator is still the translation-unit
    # constant, checked by name in the tests below.
    # Moved again 2026-08-26 by the limited-area physics lane: the bdyMask
    # digest has one definition again and cuda_driver delegates to it, and
    # the full-physics commit's recovered-state gate is routed through the
    # regional runtime like every other recovered-state gate already was.
    # Moved again 2026-08-27 by the provenance scrub: three comment
    # and docstring lines naming a person and a machine were rewritten so the
    # public assembly no longer has to move bytes that sit under a pin.  The
    # elisions this table guards are untouched, which is checked by name in
    # the tests below rather than asserted.
    # Moved again 2026-08-28 by the 0.2.0 package rename: all three files are
    # byte-identical to their pre-rename selves once the token mpas_port is
    # substituted with hexcore, so no elision this table guards moved at all --
    # which the tests below check by name rather than assert.  Re-derived by
    # tools/repin_source_tables.py, never by hand.
    # Re-derived 2026-10-17 for the optional final-acoustic output callback.
    # It retains original device allocations for due restart streams without
    # changing dynamics arithmetic or the read-only lifetimes checked below.
    # Re-derived 2026-10-19 after f65ab11c specialized immutable kernel
    # dimensions. The read-only velocity alias checks below still apply.
    SRC / "cuda_driver.py": (
        "e8d3c8484928c85f126dfce6bacb35ba215cf59dc7005ded780b96aa639f0f42"
    ),
    SRC / "cuda_horizontal.py": (
        "f94137176db65d24df0d9ff1d491c278d6c835bb08cd76dc43c5cfcbf323a345"
    ),
    SRC / "cuda_horizontal_v841.py": (
        "37714ab89935af5b02d79eeea4c06cfad038de457502116c9dad27a8bdd55465"
    ),
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_modified_execution_sources_match_the_migration_table() -> None:
    for path, expected in EXPECTED_SOURCE_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_recovery_velocities_only_mode_refuses_incomplete_host_publish() -> None:
    source = _source(SRC / "cuda_backend" / "recovery.py")
    assert "include_pressure: bool = True" in source
    assert "if include_pressure:" in source
    assert "recover_state(include_pressure=False) did not compute" in source
    assert "to_host() would publish a field set that silently" in source
    assert '"pressure": pressure_timing' in source
    assert "if include_pressure\n            else" in source


def test_candidate_recovery_binds_scalars_and_skips_discarded_fields() -> None:
    source = _source(SRC / "cuda_driver.py")
    body = _between(source, "    def _recover_candidate(", "    def _copy_state(")
    assert "self._device_state(rho, rtheta, ru, rw, scalars)" in body
    assert "cp.array(scalars, copy=True)" not in body
    assert "include_pressure=False" in body


def test_dynamics_subcycle_uses_read_only_aliases_only_at_proven_lifetimes() -> None:
    source = _source(SRC / "cuda_driver.py")
    copy_body = _between(source, "    def _copy_state(", "    def _copy_saved(")
    assert "share_scalars: bool = False" in copy_body
    assert "source.scalars if share_scalars else cp.array(source.scalars, copy=True)" in copy_body

    subcycle = _between(
        source,
        "    def _advance_dynamics_subcycle_v841(",
        "    def _step_device_v841(",
    )
    assert "saved_state = self._copy_state(state, share_scalars=True)" in subcycle
    assert "current_state = saved_state" in subcycle
    assert "current_diag = saved_diag" in subcycle
    assert "current_state = self._copy_state(saved_state)" not in subcycle
    assert "current_diag = self._copy_saved(saved_diag)" not in subcycle


def test_cached_tangential_velocity_is_bound_not_copied() -> None:
    for name in ("cuda_horizontal.py", "cuda_horizontal_v841.py"):
        source = _source(SRC / name)
        # The window opens at the guard above the binding so that it covers the
        # comment as well as the call.  Anchoring it on the _edge_field argument
        # instead would start below the comment and could only ever see the
        # closing paren.
        cached = _between(
            source,
            'raise ValueError("RK1/RK2 diagnostics require cached tangential velocity")',
            "        vorticity =",
        )
        assert ").copy()" not in cached
        assert "read" in cached or "const float*" in cached
