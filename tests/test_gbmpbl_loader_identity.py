"""Protect the established loaders and the original GBM compiled bytes."""
import hashlib
from pathlib import Path
from woof.core import gbmpbl


def test_gbm_source_and_options_are_unchanged_by_loader_isolation():
    source = gbmpbl.module_source().encode()
    assert len(source) == 65978
    assert hashlib.sha256(source).hexdigest() == "b9cd4cfaf4d7f29e7531f05c5aa48b440163ec6698e3877619006b3b35ae9ecc"
    assert gbmpbl.MODULE_OPTIONS == ("-std=c++17", "-fmad=false", "--ftz=false")


def test_shared_loader_is_pinned_after_qualified_port_merges():
    # GBM still composes the literal bytes pinned above. The shared loader
    # also owns the merged GCE, FLG, CuP, TEMF, km_opt=5, SBU, WDM and SST declarations.
    # test_flg_wiring checks its unchanged source/options for legacy modules.
    root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root / "woof/core/kernels/__init__.py").read_bytes()).hexdigest() == "c2eff811878fd46042ea94611d63b1c5dd6ee442bbd708f377f914bc7bba0ed3"


def test_fast_transcendentals_leave_the_strict_replay_image_unchanged(monkeypatch):
    from woof import numerics

    monkeypatch.setattr(numerics, "mode", lambda: "strict")
    strict = gbmpbl.compile_source()
    assert strict == gbmpbl.module_source()
    monkeypatch.setattr(numerics, "mode", lambda: "fast")
    fast = gbmpbl.compile_source()
    assert fast != strict
    # Aliases must begin after the helpers, preserving their declarations.
    assert fast.index("#define gfk_exp expf") > fast.index("__device__ float gfk_pow(")
    assert fast.index("#define gfk_pow powf") < fast.index("struct GbCol")
