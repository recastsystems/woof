"""WOOF-ArWen strict replay of the native 4.8.0 dycore case (Kessler, everything else off).

Uses the verify-exact door and integrator: real.exe handoff -> resolve_wrfinput_run ->
exact (every correction key false) -> integrate with history at every step.
"""
import os, sys, time, shutil, json
from pathlib import Path

def main():
    engine, handoff, work = map(Path, sys.argv[1:4])
    sys.path.insert(0, str(engine))
    assert os.environ.get("GPUWM_WRF_EXACT") == "1"
    from woof.wrfinput_door import resolve_wrfinput_run
    from woof.correct_mode import CORRECTION_KEYS
    from woof.experiment import build_experiment
    from woof.verify_exact.replay import set_toml_key, integrate
    import dataclasses, tomllib

    resolved = resolve_wrfinput_run(handoff)
    text = resolved.toml_text
    # The handoff namelist carries the producer's physics so the door's wrfinput
    # identity check passes (as verify-exact stages it); the dycore case's physics
    # (Kessler, everything else off, as native 4.8.0 ran it) is then set key by key.
    for key, value in (("mp_physics", 1), ("sf_sfclay_physics", 0), ("sf_surface_physics", 0),
                       ("bl_pbl_physics", 0), ("ra_physics", 0), ("do_radar_ref", 0),
                       ("wrf_rrtmg_compatibility", "none")):
        text = set_toml_key(text, key, value)
    for key in CORRECTION_KEYS:
        text = set_toml_key(text, key, False)
    if text != resolved.toml_text:
        resolved = dataclasses.replace(resolved, toml_text=text,
                                       experiment=build_experiment(tomllib.loads(text), source=f"{handoff}:l15"))
    work.mkdir(parents=True, exist_ok=True)
    if os.environ.get("L15_CHECK_ONLY"):
        print(resolved.toml_text); return
    (work / "resolved.toml").write_text(resolved.toml_text)
    t0 = time.perf_counter()
    integrate(resolved, work / "out", run_seconds=120.0)
    print(json.dumps(dict(integrate_wall_s=round(time.perf_counter() - t0, 2))))


if __name__ == "__main__":  # spawned pool workers re-import this file
    main()
