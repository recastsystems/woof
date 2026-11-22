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
# 2026-11-09: the glacier and thermal rows were re-read on test host B (RTX 5090,
# sm_120, NVRTC 12.9) for the sources at rel/2.9.13; their identities had
# moved since the 11-01 reading but the frames read the same (792, 416).
# With the stale identities every selected Noah-MP option was priced at the
# assumed frame bound: a 150x120 3 km domain priced 15.34 GiB instead of
# about 4.7, and was refused on a 24 GB card.
# 2026-11-09 (fix/noahmp-small-defects): every default Noah-MP run now steps
# these selected images (noahmp_surface_device.device_route), so a stale row
# would price every Noah-MP run at the assumed bound.  Re-read on test host B
# (RTX 5090, sm_120, NVRTC 12.9.86, production loader, fresh process, no
# launch) after the SN1 snowmelt-reset edit: every frame below read exactly
# as recorded, and bareflux/vegeflux read 0 B/thread with and without
# GPUWM_NOAHMP_MELT_RESET_ESAT.  The glacier and thermal identities are the
# rel/2.9.13 sources (the compaction port at 2f9a3c6a17 read the same).
# 3.0.0 fold: glacier and thermal re-read after opt_tksno entered both units
# (tools/measure_selected_noahmp_frames.py, test host B, NVRTC 12.9.86):
# 792 and 416 B, unchanged; every other row read as recorded.
MEASURED_PLATFORM = ("120", "12.9.86")
READINGS = (
    ('noahmp_vegprecip', 'noahmp_vegprecip', 'fe5fcc483c9bd173753ab7e428ba373f0bfd0d1c3b9afd2e1f4ca75090fc9c54', 0),
    ('noahmp_leaves', 'noahmp_leaves',
     '673c29ed4b0add71ee3a570c91887160730e0b1c4ff02a9bcb627e7b6d0db3fa', 496),
    ('noahmp_glacier', 'noahmp_glacier_composed',
     '9478c566dc0d4106aa8624d487c37595a1bba8acfc0702bd69c7627748eed032', 792),
    ('noahmp_radiation', 'noahmp_radiation',
     '4e355d5c9e9b8b7da38e24047bd3fd6d255c552bb96406402874feddcea180cc', 88),
    ('noahmp_bareflux', 'noahmp_bareflux',
     '118664a5d89aba3ee8e0d98e4ea35cf42b0fa522c84010b93325092c047ec971', 0),
    ('noahmp_vegeflux', 'noahmp_vegeflux_runtime',
     'a7e1964cd19c059babbd9406b940089530425d8710a36e1a0106bd9530584405', 0),
    # Re-read 2026-11-09 on test host B (RTX 5090, sm_120, NVRTC 12.9) after the
    # snow-compaction option (opt_snow_compact=2) entered the selected unit;
    # the same read of the previous identity 910e8dd2... gave its recorded 336.
    ('noahmp_water', 'noahmp_water',
     # 3.0.0 fix batch 4 (wetland tail): read 272 on sm_120 with NVRTC 13.4.92
     # (fast and strict); the 352 read on NVRTC 12.9.86 for the previous source
     # stays the price until this source is read on the table's platform.
     '01d3d469f9b12b85500acb3f25f12ef3950955e41956042aa0a2bbb7483948b5', 352),
    ('noahmp_thermal', 'noahmp_thermal_composed',
     # Unchanged source: NVRTC 13.4.92 reads 368, the table platform read 416.
     '5d0d3fc9d1602f3786558f17a5a554d39449861308d417a922bde6c656036b91', 416),
)


def selected_frames(profile=None, *, snow_papers=False, snow_layers=False, snow_lap=False):
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
        # sm_120 and sm_89, and with NVRTC 13.4.59, sm_120, fast and strict:
        # default solver 1136 B local/thread, Toon solver 1984 B. Both use
        # 9348 B shared/block; register counts vary by target.
        # tools/measure_snicar_frames.py records the complete source hash
        # and the attributes of every exported function
        # (docs/receipts/snicar-solver-fold-20261109/).
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest == "1352163f7a42109d3bfcd3ae8f96eced72153034bd597f811fda9548572f2923":
            frames["snow_papers"] = 1984 if snow_layers else 1136
        else:
            frames["snow_papers"] = assumed_frame_bound()
            stale.append("snow_papers")
    if snow_papers and snow_lap:
        # Bytes, not text: a locale-decoded read (Windows cp1252) re-encodes the
        # units' non-ASCII comment characters differently, so the identity would
        # never match there and every LAP run would be priced at the assumed bound.
        lap_source = source.read_bytes() + b"\n" + source.with_name("snow_lap.cu").read_bytes()
        if hashlib.sha256(lap_source).hexdigest() == LAP_SOURCE_SHA256:
            frames["snow_papers"] = max(frames["snow_papers"], LAP_FRAME_CEILING)
        else:
            frames["snow_papers"] = max(frames["snow_papers"], assumed_frame_bound())
            stale.append("snow_lap")
    if stale:
        basis = ("selected Noah-MP frame sources changed: " + ", ".join(stale)
                 + "; priced at the assumed frame bound until remeasured")
    elif getattr(profile, "compile_platform", None) == MEASURED_PLATFORM:
        basis = "selected Noah-MP frames measured on sm_120 / NVRTC 12.9.86, fast and strict"
    else:
        basis = ("selected Noah-MP frames priced from the sm_120 / NVRTC 12.9.86 "
                 "measured ceiling; this compile platform was not read")
    if snow_papers and snow_lap and "snow_lap" not in stale:
        basis += ("; LAP shared solver: 2096 B/thread, source-bound ceiling "
                  "read on sm_120 / NVRTC 13.4.59 and sm_89 / NVRTC 13.4.92")
    return frames, basis


def selected_experiment(exp):
    """Any Noah-MP domain: every one steps the selected (resident) images
    from 2.9.14 (noahmp_surface_device.device_route)."""
    from woof.core.noahmp_surface_device import device_route
    return any(dc.run.sf_surface_physics == 4 and device_route(dc.run)
               for dc in exp.domains)


# Complete GECROS image, all 19 exports, release-box sm_120 / NVRTC 13.0.88.
# Evidence: port-gecros/gecros-frame-only.json; 2026-11-02.
# 2026-11-09: SN1 edit to noahmp_gecros_flux.cuh; re-read 0 B/thread on
# sm_120 / NVRTC 13.4.59, with and without GPUWM_NOAHMP_MELT_RESET_ESAT.
GECROS_IDENTITY = "91e814bd8b0172f36f2bcb81845d51726154641fe9455da411c74da41c94a0d1"


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

# Complete snow_papers + snow_lap translation unit, measured by
# tools/measure_snicar_frames.py; both compiler modes and both target cards.
LAP_SOURCE_SHA256 = "1991d6f510748bd8468949550595a43c6bcc8622c94c435d24217d2d053701f5"
LAP_FRAME_CEILING = 2096
