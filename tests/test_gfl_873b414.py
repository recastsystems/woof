"""cu_physics=3 under WRF v4.8.0: Grell-Freitas-Li (GFL 873b414) and the previous-pin selector.

The breakage each test prevents:

1. ``test_default_is_gfl_and_the_previous_pin_is_selectable`` -- a default
   that silently stays on the deleted GFDRV would leave the 4.8.0 pin
   claiming a scheme it does not run; a selector that cannot reach GFDRV
   would take the previous pin away.
2. ``test_gf_only_capabilities_keep_gfdrv`` -- WRF v4.8.0's GFL call carries
   no SPP pattern and never writes KTOP_DEEP: under GFL, ``spp_conv`` would
   be dropped (so it is refused by name, and binding it never switches the
   scheme) and PR92 lightning would read a cloud top that stays 0 (so that
   run keeps GFDRV).
3. ``test_unknown_generation_is_refused_by_name``.
4. ``test_dispatch_resolves_the_versioned_callable`` -- the physics driver
   resolves cu_physics=3 through CU_VERSIONED_CALLABLES; a missing row
   would run GFDRV under the GFL name.
5. ``test_previous_pin_headers_keep_their_identity`` -- a checkpoint written
   before the selector ran GFDRV; the echo drops ``wrf_461`` (and the key
   on any cu_physics other than 3), so those headers keep their bytes.
6. ``test_generated_module_is_the_pinned_transcription`` -- the generated
   module must be the WRF_CHEM=0 WRF driver's transcription of the pinned
   sources (transcription.json), not an MPAS-driver or hand-edited one.
7. ``test_lane_placement_changes_no_operation`` -- the kernel runs that
   transcription lane-interleaved; spelled back, the lowered scheme must be
   the pinned one byte for byte (placement moves, no operation does).
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(nx=20, ny=20, nz=30, dx=12000.0, dy=12000.0, ztop=20000.0, dt=60.0, run_seconds=600.0)


def _cfg(**kw):
    from woof.config import RunConfig
    return RunConfig(**{**BASE, **kw})


def test_default_is_gfl_and_the_previous_pin_is_selectable():
    assert _cfg(cu_physics=3).cu_gf_version == "gfl_873b414"
    assert _cfg(cu_physics=3, cu_gf_version="wrf_461").cu_gf_version == "wrf_461"


def test_gf_only_capabilities_keep_gfdrv():
    assert _cfg(cu_physics=3, lightning_option=11).cu_gf_version == "wrf_461"
    from woof.config import validate_spp_config
    validate_spp_config(_cfg(cu_physics=3, cu_gf_version="wrf_461", spp_conv=1))
    with pytest.raises(Exception, match="cu_gf_version=wrf_461"):
        validate_spp_config(_cfg(cu_physics=3, spp_conv=1))
    # Binding SPP never changes the scheme (an ensemble's control and its
    # perturbed members run one scheme).
    assert _cfg(cu_physics=3, spp_conv=1).cu_gf_version == "gfl_873b414"


def test_unknown_generation_is_refused_by_name():
    from woof.config import validate_run_config
    valid = dict(cu_physics=3, moist=True, bl_pbl_physics=1, sf_sfclay_physics=1, cudt_minutes=0.0)
    for version in ("gfl_873b414", "wrf_461"):
        validate_run_config(_cfg(**valid, cu_gf_version=version))
    with pytest.raises(Exception, match="cu_gf_version"):
        validate_run_config(_cfg(**valid, cu_gf_version="gf_wrf471"))


def test_dispatch_resolves_the_versioned_callable():
    from woof.config import CU_STOCK_CALLABLES, CU_VERSIONED_CALLABLES
    assert CU_VERSIONED_CALLABLES[(3, "gfl_873b414")] == "woof.core.gfl.GrellFreitasLi"
    assert (3, "wrf_461") not in CU_VERSIONED_CALLABLES
    assert CU_STOCK_CALLABLES[3] == "woof.core.gf.GrellFreitas"
    import importlib
    module, _, name = CU_VERSIONED_CALLABLES[(3, "gfl_873b414")].rpartition(".")
    assert callable(getattr(importlib.import_module(module), name))


def test_previous_pin_headers_keep_their_identity():
    from woof.io.restart import ECHO_ABSENT_RUN_VALUES
    assert ECHO_ABSENT_RUN_VALUES["cu_gf_version"] == "wrf_461"
    source = (ROOT / "woof/io/restart.py").read_text(encoding="utf-8")
    assert 'values.get("cu_gf_version", "wrf_461") == "wrf_461"' in source


def test_generated_module_is_the_pinned_transcription():
    from woof.core import gfl_873b414 as gfl
    pins = json.loads((ROOT / "tools/gfl_873b414/transcription.json").read_text(encoding="utf-8"))
    assert pins["gfl"] == gfl.GFL_PIN
    by_source = {m["source"]: m for m in pins["modules"]}
    for path, sha in gfl.SOURCE_HASHES.items():
        assert by_source[path]["source_sha256"] == sha
    # One lowered piece per module, joined in MODULES order by generate_wrf.py.
    assert hashlib.sha256(gfl._SCHEME_SOURCE.encode()).hexdigest() == pins["scheme_source_sha256"]
    assert "areacell" not in gfl.ARGUMENTS and "pbl_scheme" not in gfl.ARGUMENTS
    assert "num_chem" not in gfl.ARGUMENTS and len(gfl.FIELDS) == 46
    assert "RRFS" not in gfl._SCHEME_SOURCE


def _spell_back(lane):
    """Undo gfl_lanes.lane_scheme's spellings: lane pointers, byte displacements, character storage."""
    import re
    out, pos = [], 0
    head = re.compile(r'gf_lane_plus<GFReal>\(([A-Za-z_][\w.>-]*), ')
    while (m := head.search(lane, pos)):
        depth, j = 1, m.end()
        while depth:
            depth += {'(': 1, ')': -1}.get(lane[j], 0)
            j += 1
        out.append(lane[pos:m.start()] + '((GFReal *) ((char *) ' + m[1] + ' + ' + lane[m.end():j - 1] + '))')
        pos = j
    out.append(lane[pos:])
    text = ''.join(out)
    text = text.replace('GFP ', 'GFReal *').replace('GIP ', 'int *').replace('GBP ', 'bool *')
    return text.replace(' = (char *) workspace.allocate_bytes (', ' = (char *) workspace.allocate (')


def test_lane_placement_changes_no_operation():
    """The kernel runs the pinned transcription lane-interleaved (woof/core/gfl_lanes.py).

    The breakage it prevents: a lane lowering that edits anything but pointer
    placement (an operation, its order, an index expression or a constant) would
    move words while the scheme hash still matched. Spelled back, the lowered
    scheme must be the pinned _SCHEME_SOURCE byte for byte, and the kernel source
    must be built from it.
    """
    from woof.core import gfl_873b414 as gfl
    from woof.core import gfl_lanes
    lane = gfl_lanes.lane_scheme(gfl._SCHEME_SOURCE)
    assert lane != gfl._SCHEME_SOURCE
    assert _spell_back(lane) == gfl._SCHEME_SOURCE
    source = gfl_lanes.lower(gfl._SUPPORT, gfl._SCHEME_SOURCE, gfl._ENTRY)
    assert lane in source and 'gfl_lane_columns' in source
    assert gfl.LANES == gfl_lanes.LANES and gfl.CHARACTER_BYTES == gfl_lanes.CHARACTER_BYTES


def test_lane_scratch_admission_matches_the_kernel_bounds():
    """The admission prices GFL scratch at the blocks per SM the kernel is compiled for."""
    from woof.core import gfl_lanes
    assert "#define GFL_MIN_BLOCKS %d\n" % gfl_lanes.MIN_BLOCKS_PER_SM in gfl_lanes._DEFINES


def test_lane_scratch_is_bounded_by_the_domain_and_the_cap():
    """A small domain holds only its own slabs; a CONUS 12 km domain stops at the 4 GiB cap.

    Breakage prevented: before 2026-11-09 a 150 x 150 domain held a whole card's
    resident slabs (4 GiB on a 188-SM card) and the admission priced none of it.
    """
    from woof.core.gfl_lanes import LANES, CHARACTER_BYTES, SCRATCH_CAP_BYTES, scratch_bytes, share_bytes
    per_slab = (share_bytes(50, 49) + CHARACTER_BYTES) * LANES
    assert scratch_bytes(49, 150 * 150, 188) == 176 * per_slab          # 22,500 columns = 176 slabs
    big = scratch_bytes(49, 424 * 299, 188)                             # 991 slabs > 3 x 188 resident
    assert big == min(3 * 188, SCRATCH_CAP_BYTES // per_slab) * per_slab <= SCRATCH_CAP_BYTES
    assert scratch_bytes(49, 424 * 299, 188, free=8 << 30) == (2 << 30) // per_slab * per_slab


def test_preflight_prices_gfl_lane_scratch_not_gfdrv_tiles():
    from types import SimpleNamespace
    from woof.core.gfl_lanes import scratch_bytes
    from woof.core.preflight import gf_column_workspace_bytes, MEASURED_LOCAL_MEMORY_PROFILE

    def exp(version, nx=150, ny=150, nz=49):
        run = SimpleNamespace(cu_physics=3, cu_gf_version=version, nx=nx, ny=ny, nz=nz)
        return SimpleNamespace(domains=[SimpleNamespace(run=run)])
    sms = MEASURED_LOCAL_MEMORY_PROFILE.multiprocessor_count
    assert gf_column_workspace_bytes(exp("gfl_873b414")) == scratch_bytes(49, 22500, sms)
    assert gf_column_workspace_bytes(exp("wrf_461")) != scratch_bytes(49, 22500, sms)
