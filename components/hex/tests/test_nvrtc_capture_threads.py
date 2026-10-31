"""The NVRTC capture never serializes compiles across threads.

THE BREAKAGE (VRFS 8x RTX 5090, woof hex 0.4.9, 2026-10-28): the capture held a
process-wide lock for the whole compile, so the 8 rank threads of a multi-card
run compiled their 48 mesh-specific first-use modules one at a time (a 1.05 s
staircase in fast numerics, ~4 s per step in strict: 127 s first strict step).
"""
import hashlib
import threading

import pytest

compiler = pytest.importorskip("cupy.cuda.compiler")
from woof.hex.cuda_backend import compile_contract  # noqa: E402
from woof.hex.cuda_backend.compile_contract import NvrtcCompileCapture  # noqa: E402


def image(source):
    return b"\x7fELF" + source.encode(), {}


def test_two_threads_compile_at_once_and_each_capture_sees_only_its_own(monkeypatch):
    both_inside = threading.Barrier(2, timeout=20)

    def nvrtc(source, options=(), *args, **kwargs):
        # A lock held across the compile never lets the second thread in: the barrier breaks.
        both_inside.wait()
        return image(source)

    monkeypatch.setattr(compiler, "_compile_using_nvrtc_no_warning", nvrtc)
    captures, errors = {}, []

    def rank(name):
        try:
            with NvrtcCompileCapture() as capture:
                compiler._compile_using_nvrtc_no_warning(name, ("--fmad=false", "-I/x"))
            captures[name] = capture
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=rank, args=(name,)) for name in ("rank0", "rank1")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not errors, errors
    for name, capture in captures.items():
        assert [o.source_sha256 for o in capture.observations] == [
            hashlib.sha256(name.encode()).hexdigest()]
        assert capture.observations[0].effective_flags == ("--fmad=false",)
        assert capture.observations[0].include_path_count == 1
        assert capture.evidence(hashlib.sha256(name.encode()).hexdigest())["status"] == "resolved"


def test_nested_and_late_wrappers_record_each_compile_once(monkeypatch):
    calls = []

    def nvrtc(source, options=(), *args, **kwargs):
        calls.append(source)
        return image(source)

    monkeypatch.setattr(compiler, "_compile_using_nvrtc_no_warning", nvrtc)
    with NvrtcCompileCapture() as outer:
        with NvrtcCompileCapture() as inner:
            compiler._compile_using_nvrtc_no_warning("nested", ())
    assert len(outer.observations) == len(inner.observations) == 1

    # A wrapper installed on top after the hook (the kernel plan funnel, the warm
    # store) is covered by a new outer hook at the next capture; still one record.
    below = compiler._compile_using_nvrtc_no_warning

    def funnel(source, options=(), *args, **kwargs):
        return below(source, options, *args, **kwargs)

    monkeypatch.setattr(compiler, "_compile_using_nvrtc_no_warning", funnel)
    with NvrtcCompileCapture() as late:
        compiler._compile_using_nvrtc_no_warning("late", ())
    assert [o.source_sha256 for o in late.observations] == [hashlib.sha256(b"late").hexdigest()]

    # Outside any capture the hooks only pass through.
    compiler._compile_using_nvrtc_no_warning("plain", ())
    assert calls == ["nested", "late", "plain"]
    assert len(late.observations) == 1
    assert not getattr(compile_contract._CAPTURE_THREAD, "stack", [])
