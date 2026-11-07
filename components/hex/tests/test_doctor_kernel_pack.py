"""``woof hex doctor`` warns when the loaded NVRTC matches no pinned kernel pack.

THE BREAKAGE (woof hex 0.4.12 post-publish check, 2026-11-05): ``cupy-cuda13x[ctk]`` pulled NVRTC 13.5.37,
published after the gate.  No pack was pinned for it, so a fresh ``[gpu-cu13]`` install compiled every kernel
through NVRTC for about 100 s before its first step, and doctor printed "Every check passed".
"""

from __future__ import annotations

from woof.hex import assets, doctor


def _identity(nvrtc, library, package):
    return {"cupy": "14.2.0", "nvrtc": list(nvrtc), "cupy_package": "cupy-cuda13x==14.2.0",
            "nvrtc_package": package, "nvrtc_library_sha256": library,
            "cache_key": "87417f5ddafec4c853dab1863f8bb308dafc152c", "use_ptx": False, "use_pch": False,
            "debug": "0"}


#: An NVRTC published after the newest pack (a stand-in library digest): what 13.5.37 was to 0.4.12.
NVRTC_UNPINNED = _identity((13, 6), "0" * 64, "nvidia-cuda-nvrtc==13.6.0")


def test_an_nvrtc_without_a_pinned_pack_is_a_warning_and_not_every_check_passed(monkeypatch):
    monkeypatch.setattr(doctor, "_compiler_identity_probe", lambda: {"identity": NVRTC_UNPINNED})
    [finding] = doctor.check_kernel_pack()
    assert finding.status == doctor.WARN
    assert not finding.required
    assert "13.6" in finding.detail and "13.5 (nvidia-cuda-nvrtc==13.5.37)" in finding.detail
    assert finding.remedy.startswith('pip install "nvidia-cuda-nvrtc==')
    assert doctor.summarise([finding]) != "\nEvery check passed."
    assert doctor.render([finding], explain=False).startswith("WARN    prebuilt kernel pack")


#: What the ``[gpu-cu13]`` extra installs (nvidia-cuda-nvrtc<=13.5.37), measured on a fresh install 2026-11-05.
NVRTC_135 = _identity((13, 5), "98a317ff5d2176dee5a04235c6663f8d553d5ed95b3104cbf7fb198cbf76a10d",
                      "nvidia-cuda-nvrtc==13.5.37")
#: The NVRTC 0.4.13's capped extra installed; its pack stays pinned for installs that hold it.
NVRTC_134 = _identity((13, 4), "d8f82e7019fe990bcd9efe4c5c444c036ddf2d5df95090d3965ae62be2009a2c",
                      "nvidia-cuda-nvrtc==13.4.92")


def test_the_nvrtc_a_fresh_cu13_install_loads_has_its_pack(monkeypatch):
    monkeypatch.setattr(doctor, "_compiler_identity_probe", lambda: {"identity": NVRTC_135})
    [finding] = doctor.check_kernel_pack()
    assert finding.status == doctor.VERIFIED, finding.detail
    name, row, reason = assets.compatible_pack(NVRTC_135)
    assert reason is None and name == "kernel-pack-cuda13.json"
    assert row["compiler"]["nvrtc_package"] == "nvidia-cuda-nvrtc==13.5.37"
    name, row, reason = assets.compatible_pack(NVRTC_134)
    assert reason is None and name == "kernel-pack-cuda13.4.json"


def test_no_compiler_is_information_only(monkeypatch):
    monkeypatch.setattr(doctor, "_compiler_identity_probe", lambda: {"error": "ModuleNotFoundError: cupy"})
    [finding] = doctor.check_kernel_pack()
    assert finding.status == doctor.INFO and not finding.required


def test_the_runtime_and_doctor_share_one_rule():
    path, reason = assets.resolve_pack(NVRTC_UNPINNED)
    assert path is None and "matches no NVRTC 13 pin" in reason
