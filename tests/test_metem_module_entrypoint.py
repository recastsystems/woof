"""The native met_em CLI must see its complete preparation namespace."""
import runpy
import sys
from pathlib import Path


def test_native_module_entrypoint_has_preparation_helpers():
    path = str(Path(__file__).resolve().parents[1] / "woof/metem_forecast.py")
    class EntrypointObserved(Exception):
        pass
    def observe(frame, event, argument):
        if event == "call" and frame.f_code.co_filename == path and frame.f_code.co_name == "main":
            assert callable(frame.f_globals.get("_smooth_cg_case")), "native CLI entered main before its terrain preparation helper was defined"
            raise EntrypointObserved
        return observe
    previous = sys.gettrace()
    try:
        sys.settrace(observe)
        try:
            runpy.run_module("woof.metem_forecast", run_name="__main__")
        except EntrypointObserved:
            pass
        else:
            raise AssertionError("the real native module did not enter its CLI")
    finally:
        sys.settrace(previous)
