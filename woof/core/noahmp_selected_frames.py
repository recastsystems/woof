"""Source-bound frame prices for the selected Noah-MP translation units.

Read on test host B, sm_120 / NVRTC 12.9.86, 2026-11-01. All exported
functions were read through the production loader in fast and strict modes;
the readings agree. An unread platform uses the measured ceiling, as the
existing composed-unit estimator does. Changed sources use the tree's
assumed bound until a new reading is added. Default-option pricing is separate.
"""
from woof.core.noahmp_kernel_sources import runtime_unit
from woof.core.kernel_frame_recordings import assumed_frame_bound

# name, pricing key, complete translation-unit identity, maximum frame bytes.
MEASURED_PLATFORM = ("120", "12.9.86")
READINGS = (
    ('noahmp_vegprecip', 'noahmp_vegprecip', 'fe5fcc483c9bd173753ab7e428ba373f0bfd0d1c3b9afd2e1f4ca75090fc9c54', 0),
    ('noahmp_leaves', 'noahmp_leaves',
     '673c29ed4b0add71ee3a570c91887160730e0b1c4ff02a9bcb627e7b6d0db3fa', 496),
    ('noahmp_glacier', 'noahmp_glacier_composed',
     '81078c0a5ac38c358052cc3baf7dc9d6df9dcf22b2026ee3e0d9e344032e5e07', 792),
    ('noahmp_radiation', 'noahmp_radiation',
     '4e355d5c9e9b8b7da38e24047bd3fd6d255c552bb96406402874feddcea180cc', 88),
    ('noahmp_bareflux', 'noahmp_bareflux',
     'd8a44aa348151a0f991dca61919cfcaadde39e11dbfadbb9c674965d65d7bdf4', 0),
    ('noahmp_vegeflux', 'noahmp_vegeflux_runtime',
     '080700d6932eab2b60c290a825e16add570911345181ee80f8746d7f00e7c3ea', 0),
    ('noahmp_water', 'noahmp_water',
     '910e8dd2fec6ef8bcc5c0be50f5c13942aea09ef1b4ffd138f9809371ab2f445', 336),
    ('noahmp_thermal', 'noahmp_thermal_composed',
     '8d026f9cc33a9711112647e1b50c4c97ac8c2ebf98affa096cf06b9be363c79b', 416),
)


def selected_frames(profile=None, *, snow_papers=False):
    """Return prices and a readable measurement basis, without a CUDA import."""
    frames = {}
    stale = []
    for name, key, identity, frame in READINGS:
        if runtime_unit(name, selected_options=True).identity()["identity_sha256"] == identity:
            frames[key] = frame
        else:
            frames[key] = assumed_frame_bound()
            stale.append(name)
    if snow_papers:
        import hashlib
        from pathlib import Path
        source = Path(__file__).with_name("kernels") / "snow_papers.cu"
        # Production loader: C++17, --ftz=false. Read with NVRTC 13.4.92,
        # sm_120, fast and strict: solver 1136 B local/thread, 9348 B
        # shared/block, 138 registers; both packing exports have zero local.
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest == "6910e7580f9d4dcf6c7ee9b53f349ee3c982f075d4d7f2b8cf924ee19ffa6ad6":
            frames["snow_papers"] = 1136
        else:
            frames["snow_papers"] = assumed_frame_bound()
            stale.append("snow_papers")
    if stale:
        basis = ("selected Noah-MP frame sources changed: " + ", ".join(stale)
                 + "; priced at the assumed frame bound until remeasured")
    elif getattr(profile, "compile_platform", None) == MEASURED_PLATFORM:
        basis = "selected Noah-MP frames measured on sm_120 / NVRTC 12.9.86, fast and strict"
    else:
        basis = ("selected Noah-MP frames priced from the sm_120 / NVRTC 12.9.86 "
                 "measured ceiling; this compile platform was not read")
    return frames, basis


def selected_experiment(exp):
    from woof.core.noahmp_surface_device import selected_options
    return any(dc.run.sf_surface_physics == 4 and selected_options(dc.run)
               for dc in exp.domains)


# Complete GECROS image, all 19 exports, release-box sm_120 / NVRTC 13.0.88.
# Evidence: port-gecros/gecros-frame-only.json; 2026-11-02.
GECROS_IDENTITY = "c60f43ecdaa87ef636b1b34634e74ea50fcd5638238450f7b2dbb4d53012fe52"


def gecros_frame(profile=None):
    """Source-bound opt-in price; default Noah-MP prices remain untouched."""
    current = runtime_unit("noahmp_gecros").identity()["identity_sha256"]
    if current != GECROS_IDENTITY:
        return assumed_frame_bound(), "GECROS source changed; assumed frame bound until remeasured", False
    if getattr(profile, "compile_platform", None) == ("120", "13.0.88"):
        basis = "GECROS 0 B/thread measured across all exports on sm_120 / NVRTC 13.0.88"
    else:
        basis = "GECROS priced from sm_120 / NVRTC 13.0.88 measured ceiling (0 B/thread); not measured on this card"
    return 0, basis, True
