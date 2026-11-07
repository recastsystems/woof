"""The managed Earth2Studio environment `woof setup --with-earth2` builds; no network or GPU."""

from woof import earth2_env


def _built_argv(monkeypatch, tmp_path, recipe):
    monkeypatch.setenv(earth2_env.ROOT_ENV, str(tmp_path / "env"))
    monkeypatch.setattr(earth2_env, "recipe_name", lambda: recipe)
    monkeypatch.setattr(earth2_env, "_uv", lambda base, log: "uv")
    calls = []
    envs = []

    def run(argv, log, env=None):
        calls.append([str(a) for a in argv])
        envs.append(env)

    monkeypatch.setattr(earth2_env, "_run", run)
    receipt = earth2_env._build_python(earth2_env.table()["default_extras"], lambda line: None,
                                       tmp_path / "build.log")
    return calls, receipt, envs


def test_setup_builds_the_default_models_extra(monkeypatch, tmp_path):
    # Defect (2.9.4 landing, test host 2026-10-20): `woof setup --with-earth2` built
    # earth2studio[cyclone,windgust-afno,precip-afno] only, so the environment had no
    # torch_harmonics or makani and every FCN3 run (the table's default model) died at import.
    from woof.earth2_run import model_table
    table = model_table()
    default_extra = table["models"][table["runtime"]["default_model"]]["earth2studio_extra"]
    assert default_extra == "fcn3"
    assert default_extra in earth2_env.table()["default_extras"]
    assert {"torch_harmonics", "makani"} <= set(earth2_env.required_modules(["fcn3"]))
    for recipe in ("cu12", "cu13"):
        calls, receipt, envs = _built_argv(monkeypatch, tmp_path, recipe)
        install = next(argv for argv in calls if any(a.startswith("earth2studio") for a in argv))
        spec = next(a for a in install if a.startswith("earth2studio"))
        # makani is not on PyPI: earth2studio[fcn3] cannot resolve, so the pinned sources install.
        assert "fcn3" not in spec, (recipe, spec)
        assert "--no-build-isolation" in install
        pins = [a for a in install if " @ git+" in a]
        assert any(a.startswith("torch-harmonics @ git+") and
                   a.endswith("@a632ca748a12bd9f74dbc1e00653317810991f74") for a in pins), recipe
        assert any(a.startswith("makani @ git+") and
                   a.endswith("@b38fcb2799d7dbc146fa60459f3f9823394a8bf1") for a in pins), recipe
        # makani builds with setuptools-scm, and --no-build-isolation needs it installed first.
        before = calls[:calls.index(install)]
        assert any("setuptools-scm" in argv for argv in before), recipe
        assert "fcn3" in receipt["extras"]


def test_cu13_builds_fcn3_extensions_against_its_private_compiler(monkeypatch, tmp_path):
    # Defect (test host 2026-10-21, the first real setup of this branch): torch-harmonics built against
    # the box's /usr/local/cuda 12.8 under the cu13 torch and refused (CUDA 12.8 vs 13.0).
    from woof.earth2_env import runtime
    recipe = runtime()["install_recipes"]["cu13"]
    calls, _receipt, envs = _built_argv(monkeypatch, tmp_path, "cu13")
    install = next(i for i, argv in enumerate(calls) if any(a.startswith("earth2studio") for a in argv))
    assert all(p in calls[install - 1] for p in recipe["compiler_packages"])
    from pathlib import Path
    home = str(Path(recipe["cuda_home"].replace("ENV", str(tmp_path / "env" / "venv"))))
    assert envs[install]["CUDA_HOME"] == home
    assert envs[install]["PATH"].startswith(home)
    calls, _receipt, envs = _built_argv(monkeypatch, tmp_path, "cu12")
    assert all(env is None for env in envs)


def test_cu13_private_compiler_header_takes_glibc_rsqrt_noexcept(tmp_path):
    # Defect (2.9.4 R15, Ubuntu 26.04 / glibc 2.43, 2026-10-22): nvcc 13.0.88 failed torch-harmonics on
    # "exception specification is incompatible with that of previous function rsqrt" (bits/mathcalls.h).
    from woof.earth2_env import _glibc_rsqrt_compat
    header = tmp_path / "include" / "crt" / "math_functions.h"
    header.parent.mkdir(parents=True)
    header.write_text(
        "/* doc */\n"
        "extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x);\n"
        "extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x);\n",
        encoding="utf-8")
    assert _glibc_rsqrt_compat(tmp_path) is True
    text = header.read_text(encoding="utf-8")
    assert "rsqrt(double x) _NV_RSQRT_SPECIFIER;" in text and "rsqrtf(float x) _NV_RSQRT_SPECIFIER;" in text
    assert text.index("#if !defined(_NV_RSQRT_SPECIFIER)") < text.index("rsqrt(double x)")
    assert _glibc_rsqrt_compat(tmp_path) is False  # a guarded header (13.2+, or a second build) is left alone
    assert header.read_text(encoding="utf-8") == text


def test_setup_with_earth2_fails_when_the_earth2_build_fails(monkeypatch, capsys):
    # Defect (2.9.4 R15, RTX 4090 2026-10-22): "FAILED earth2" and still exit 0 through the
    # estate-complete note, so the FCN3 fetch after it failed on "earth2studio is not installed".
    import argparse
    from woof import doctor, earth2_env, setup_cli

    def broken(**_kwargs):
        raise RuntimeError("building the managed Earth2Studio environment failed")

    monkeypatch.setattr(setup_cli, "_STEPS", [])
    monkeypatch.setattr(earth2_env, "ensure", broken)
    monkeypatch.setattr(doctor, "collect_checks", lambda: [])
    monkeypatch.setattr(doctor, "blocking_gaps", lambda checks: [])
    monkeypatch.setattr(doctor, "format_brief", lambda checks: "")
    args = argparse.Namespace(with_earth2=True, with_geog=False, from_dir=None, explain=False)
    assert setup_cli.setup_main(args) == 2
    assert "FAILED  earth2" in capsys.readouterr().out
