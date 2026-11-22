#!/usr/bin/env python3
"""Short whole-model run of a native wrfinput handoff through the verify-exact door.

    one_step.py ENGINE HANDOFF WORK SECONDS [key=value ...]

ENGINE is a woof source tree put first on sys.path (the tip or the branch);
HANDOFF holds wrfinput_d01, wrfbdy_d01 and namelist.input.  key=value pairs
override the handoff's run TOML (ints, floats, true/false or strings).  Every
correction key is false; history is written at every frame.  Used for the
4.7.1-option identity (tip vs branch, same handoff, same settings) and for
the 4.8.0 smoke.
"""
import dataclasses
import json
import os
import sys
import time
import tomllib
from pathlib import Path



def _value(v):
    if v in ("true", "false"):
        return v == "true"
    for kind in (int, float):
        try:
            return kind(v)
        except ValueError:
            pass
    return v


def main():
    engine, handoff, work = map(Path, sys.argv[1:4])
    seconds = float(sys.argv[4])
    # The engine under test comes from PYTHONPATH (set by the caller), so
    # spawned kernel-plan workers import the same tree.
    import woof
    assert Path(woof.__file__).resolve().parent.parent == engine.resolve(), woof.__file__
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.correct_mode import CORRECTION_KEYS
    from woof.experiment import build_experiment
    from woof.verify_exact.replay import set_toml_key, integrate
    resolved = resolve_wrfinput_run(handoff)
    text = resolved.toml_text
    for key in CORRECTION_KEYS:
        text = set_toml_key(text, key, False)
    doc = tomllib.loads(text)
    for item in sys.argv[5:]:
        key, value = item.split("=", 1)
        table = next((t for t in ("physics", "run", "domain") if isinstance(doc.get(t), dict) and key in doc[t]), None)
        try:
            text = set_toml_key(text, key, _value(value))
        except Exception:  # a key the handoff TOML does not spell yet
            target = table or ("physics" if "physics" in doc else "run")
            v = _value(value)
            lit = json.dumps(v) if isinstance(v, str) else (str(v).lower() if isinstance(v, bool) else str(v))
            marker = f"[{target}]\n"
            assert marker in text, (target, key)
            text = text.replace(marker, marker + f"{key} = {lit}\n", 1)
    resolved = dataclasses.replace(resolved, toml_text=text,
                                   experiment=build_experiment(tomllib.loads(text), source=f"{handoff}:l09"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "resolved.toml").write_text(resolved.toml_text)
    t0 = time.perf_counter()
    integrate(resolved, work / "out", run_seconds=seconds)
    print(json.dumps(dict(engine=str(engine), integrate_wall_s=round(time.perf_counter() - t0, 2),
                          exact=os.environ.get("GPUWM_WRF_EXACT", ""))))


if __name__ == "__main__":
    main()
