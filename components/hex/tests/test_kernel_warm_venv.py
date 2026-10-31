"""The warm kernel store is keyed by content: a second venv of one release finds the images, without a GPU."""
import pytest

from woof.hex.kernel_warm import Store


def _install(tmp_path, name, header="#define CUPY_VALUE 1\n"):
    """A site-packages tree with CuPy's include directory, as a venv of one release carries it."""
    root = tmp_path / name / "lib" / "python3.14t" / "site-packages"
    include = root / "cupy" / "_core" / "include"
    include.mkdir(parents=True)
    (include / "cupy_value.h").write_text(header)
    return str(root.resolve()), include


def _cupy_request(include, program, source="#include <cupy_value.h>\nextern \"C\" __global__ void k() {}"):
    return dict(source=source, options=["-I" + str(include), "--std=c++17", "-ftz=true"], arch="120",
                filename=program, name_expressions=None, cache_in_memory=False, method=None)


def test_a_second_venv_of_the_same_release_reuses_the_image(tmp_path, monkeypatch):
    """THE BREAKAGE THIS PREVENTS (VRFS 8x RTX 5090, 2026-10-28, pc-after-8c): a run from a new venv path, on a
    box whose store held every image, compiled 53 modules inline on one core (first step 37.3 s against 4.3 s warm):
    the key carried the venv's site-packages -I paths, its header-tree paths and CuPy's program name derived from
    them.  A second install with the same header bytes now finds the image."""
    import woof.hex.kernel_warm as warm
    identity = dict(nvrtc=[13, 4], library="compiler-a")
    first_root, first_include = _install(tmp_path, "venv-a")
    second_root, second_include = _install(tmp_path, "venv-b")
    monkeypatch.setattr(warm, "_install_roots", lambda: (first_root,))
    first = Store(tmp_path / "cache", identity, "shape", workers=2)
    assert first.call("nvrtc", _cupy_request(first_include, "a" * 40 + ".cubin.cu"),
                      lambda: (b"\x7fELF image", None)) == (b"\x7fELF image", None)
    first.flush()
    first.pool.shutdown()
    monkeypatch.setattr(warm, "_install_roots", lambda: (second_root,))
    second = Store(tmp_path / "cache", identity, "shape", workers=2)
    assert second.call("nvrtc", _cupy_request(second_include, "b" * 40 + ".cubin.cu"),
                       lambda: pytest.fail("a second venv recompiled a cached request")) == (b"\x7fELF image", None)
    assert second.receipt()["hits"] == 1
    second.pool.shutdown()


@pytest.mark.parametrize("change", ["header", "explicit_name", "name_sensitive"])
def test_a_second_venv_still_misses_on_what_can_change_the_image(tmp_path, monkeypatch, change):
    """Only where the install lives is dropped: other header bytes, a caller's explicit program name, and CuPy's
    derived name for a source that can embed it (``__FILE__``) still bind the key."""
    import woof.hex.kernel_warm as warm
    identity = dict(nvrtc=[13, 4], library="compiler-a")
    first_root, first_include = _install(tmp_path, "venv-a")
    second_root, second_include = _install(tmp_path, "venv-b",
                                           "#define CUPY_VALUE 2\n" if change == "header" else "#define CUPY_VALUE 1\n")
    source = "#include <cupy_value.h>\nconst char *f = __FILE__;" if change == "name_sensitive" else None
    names = ("k.cu", "other.cu") if change == "explicit_name" else ("a" * 40 + ".cubin.cu", "b" * 40 + ".cubin.cu")
    def request(include, program):
        return _cupy_request(include, program, source) if source else _cupy_request(include, program)
    monkeypatch.setattr(warm, "_install_roots", lambda: (first_root,))
    first = Store(tmp_path / "cache", identity, "shape", workers=2)
    first.call("nvrtc", request(first_include, names[0]), lambda: (b"old", None))
    first.flush()
    first.pool.shutdown()
    monkeypatch.setattr(warm, "_install_roots", lambda: (second_root,))
    second = Store(tmp_path / "cache", identity, "shape", workers=2)
    assert second.call("nvrtc", request(second_include, names[1]), lambda: (b"new", None)) == (b"new", None)
    second.flush()
    second.pool.shutdown()
