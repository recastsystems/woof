"""The integer tier records the options passed to its ordinary compiler."""
import sys
import importlib
from types import SimpleNamespace

from woof.core import kernels
from woof import physics_params
kernel_manifest = importlib.import_module("woof.certify.kernel_manifest")


def test_integer_tier_records_the_compiled_options(monkeypatch):
    compiled = []
    recorded = []
    def raw_module(**kwargs):
        compiled.append(kwargs)
        return object()
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(RawModule=raw_module))
    monkeypatch.setattr(kernels, "_compile_observed", lambda *args: None)
    monkeypatch.setattr(physics_params, "note_compiled", lambda *args: None)
    monkeypatch.setattr(kernel_manifest, "record_module", lambda *args, **kwargs: recorded.append((args, kwargs)))
    factory = kernels.load_module_int_defines.__wrapped__
    module = factory("record_select", (("NLEVELS", 3),))
    assert len(compiled) == len(recorded) == 1
    assert compiled[0]["options"] == ("-std=c++17",)
    assert recorded[0][1]["options"] == compiled[0]["options"]
    assert recorded[0][1]["source"] == compiled[0]["code"]
    assert recorded[0][1]["module"] is module
    assert recorded[0][0] == ("woof.core.kernels:record_select[NLEVELS=3]",)
