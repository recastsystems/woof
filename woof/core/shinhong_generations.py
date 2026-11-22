"""Shin-Hong (bl_pbl_physics = 11) source generations.

Kept free of CuPy so configuration, registry and restart code can read it
on a host without a device.
"""

#: ``wrf_461``: WRF v4.6.1 phys/module_bl_shinhong.F (sha256 99f44dbe...),
#: byte-identical in v4.7.1; kernels/shinhong.cu.  ``wrf_480``: WRF v4.8.0
#: phys/physics_mmm/bl_shinhong.F90 (MMM 550b5b49, sha256 7d258836...) with
#: phys/module_bl_shinhong.F (sha256 c7bfe363...); kernels/shinhong_480.cu.
SHINHONG_GENERATIONS = ("wrf_480", "wrf_461")
SHINHONG_DEFAULT_GENERATION = "wrf_480"
SHINHONG_SOURCES = {
    "wrf_461": {
        "kernel_module": "shinhong",
        "source_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "source_sha256": "99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56",
    },
    "wrf_480": {
        "kernel_module": "shinhong_480",
        "source_commit": "06d4240ae989cc3e50af412bb472df3d9048783c",
        "physics_mmm_commit": "550b5b494758ad2756394e93cb08b5709b8feb83",
        "source_sha256": "7d2588364d95519024d4914273e6a4953f1baa393261e0035f9a7198fbc9d961",
        "wrapper_sha256": "c7bfe36353a9db6f4fd471a5ca13287efb4b92d5c12ff2797af685eb536cb5dc",
        "restart_algorithm_identity": "shinhong-wrf-v4.8.0-v1",
    },
}
