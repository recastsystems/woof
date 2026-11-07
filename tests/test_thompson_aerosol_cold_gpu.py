"""WP-06 GPU gates for the aerosol-aware Thompson cold network (mp=28).

Three families of gate, in increasing strength:

1.  SOURCE GATES (no GPU, no tables).  ``iceDeMott`` *replaces* Cooper
    nucleation under mp=28; a half-converted kernel that still carries the
    Cooper expression, or the frozen ``nuclei_bin = 27`` / ``cloud_number_bin
    = 65`` literals, runs and stays numerically stable while being silently
    wrong.  These gates read the translation unit and refuse those literals.

2.  SYNTHETIC-TABLE GATES (GPU, no coefficient assets).  A hand-built
    freezing table with a different value on every ``idx_IN`` slice proves
    the kernel reads the slice ``thompson_aa_in_bin`` selects, not slice 27.
    This is the one gate that runs on a tree whose 380 MB classic assets are
    not staged.

3.  ORACLE GATES (GPU + the canonical classic tables).  Direct comparison
    against ``woof/data/thompson/oracle-aero/``, generated from
    ``wrf461-pristine/phys/module_mp_thompson.F`` (WRF v4.6.1,
    commit d66e442) by ``tools/thompson_wrf461_oracle/build_aero.sh``.

    Two forms are used, and the difference matters:

    * END-TO-END, against the committed fixture CSVs.  In three fixtures the
      aerosol species are touched by this kernel and by nothing else in the
      whole ``mp_gt_driver`` call, so ``clamp(nwfa_before + nwfaten*dt)``
      *is* the committed ``after`` column.  ``nwfa``/``nifa`` have no
      sedimentation term anywhere in module_mp_thompson.F, which is what
      makes this exact rather than approximate.

    * PER-LEVEL, against ``_WRF_COLD_REFERENCE``.  Fixtures that also run
      warm-section number sinks, CCN activation or hydrometeor fallout
      cannot be closed end to end by one kernel.  For those, the reference
      is WRF's own per-level tendency, obtained by the procedure documented
      on ``_WRF_COLD_REFERENCE``.

TOLERANCE POLICY.  Bounds here are MEASURED, not chosen for convenience, and
each non-default one names its cause.  The one loose bound is ``iceKoop``'s:
``prob_h = 1 - exp(-J*V*dt)`` with ``J*V*dt ~ 3e-5`` is quantized to
multiples of 2^-24 by the float32 subtraction, so a sub-ulp difference in
``qvs`` moves the answer by whole grid steps of 1.8e-3 relative.  See
``test_koop_haze_freezing_matches_committed_oracle`` for the measurement
that attributes it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

_REPO = Path(__file__).parents[1]
_ORACLE = _REPO / "woof" / "data" / "thompson" / "oracle-aero"
_KERNEL_SOURCE = (
    _REPO / "woof" / "core" / "kernels" / "thompson_aerosol_cold.cu")

_LEVELS = 24

#: WRF's terminal aerosol clamps, module_mp_thompson.F:3976-3981.  Both
#: compare a PER-KILOGRAM quantity against a per-cubic-metre constant.  That
#: unit inconsistency is WRF's, is visible in fixture 111 (levels 17-24 of
#: nifa land exactly on 9999e6), and is reproduced literally.
_NWFA_FLOOR = np.float32(11.1e6)
_NIFA_FLOOR = np.float32(5.0e3)
_AERO_CEIL = np.float32(9999.0e6)


# ---------------------------------------------------------------------------
# Fixture plumbing.
# ---------------------------------------------------------------------------

def _column(scenario):
    """Return the fixture's ``(before, after)`` 24-row halves."""
    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * _LEVELS
    return rows[:_LEVELS], rows[_LEVELS:]


def _host(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float32)


def _classic_table_root():
    """Locate a directory holding all four canonical classic assets.

    ``freezeH2O.dat`` is 255 MB and is not committed, so this returns
    ``None`` when the assets have not been staged.  Callers skip rather than
    substitute a synthetic table: the whole point of these gates is that the
    kernel indexes the REAL ``idx_IN`` axis.
    """
    import os

    from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS

    candidates = []
    override = os.environ.get("WOOF_THOMPSON_TABLE_ROOT")
    if override:
        candidates.append(Path(override))
    # Resolved rather than joined: the packaged classic-table directory
    # ships in the recast-woof-data companion distribution since 2.5.0.
    from woof.physics_compat import packaged_thompson_table_root
    candidates.append(packaged_thompson_table_root())
    for root in candidates:
        if all((root / asset.filename).is_file()
               for asset in CLASSIC_TABLE_ASSETS):
            return root
    return None


@pytest.fixture(scope="module")
def classic_tables():
    from woof.core.thompson_runtime import load_classic_device_tables

    root = _classic_table_root()
    if root is None:
        pytest.skip(
            "canonical classic Thompson tables (including the 255 MB "
            "freezeH2O.dat) are not staged; set WOOF_THOMPSON_TABLE_ROOT")
    return load_classic_device_tables(str(root))


def _run_cold_network(scenario, dt, table_owner):
    """Launch the cold network once on a fixture's entry column.

    Returns ``(state, accumulators, entry)`` as host arrays.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import (
        launch_aa_cold_network_from_owner)
    from woof.core.thompson_aerosol_state import (
        launch_aa_graupel_number_init)

    before, _ = _column(scenario)

    def device(name):
        return cp.asarray(_host(before, name)[:, None, None].copy())

    fields = {
        "qi": device("qi"), "ni": device("ni_per_kg"),
        "qs": device("qs"), "qg": device("qg"),
        "qr": device("qr"), "nr": device("nr_per_kg"),
        "qc": device("qc"),
        "temperature": device("temp_k"), "qv": device("qv"),
    }
    pressure = device("p_pa")
    entry = {
        "nc": device("nc_per_kg"),
        "nwfa": device("nwfa_per_kg"),
        "nifa": device("nifa_per_kg"),
    }
    entry_copy = {k: v.copy() for k, v in entry.items()}
    zeros = cp.zeros_like(fields["qi"])
    acc = {"ncten": zeros.copy(), "nwfaten": zeros.copy(),
           "nifaten": zeros.copy()}
    shadow = zeros.copy()
    # mp_gt_driver's per-call graupel number (:1267-1281), as the adapter
    # forms it before the cold network: the graupel distribution of every
    # graupel rate is built from it.
    launch_aa_graupel_number_init(fields["qg"], fields["temperature"],
                                  pressure, fields["qv"], shadow)
    # Deliberately seeded to a non-neutral value so the unconditional reset
    # at kernel entry (WRF's vts_boost = 1.0 at :2243) is observable.
    boost = cp.full_like(fields["qi"], 7.0)

    launch_aa_cold_network_from_owner(
        fields["qi"], fields["ni"], fields["qs"], fields["qg"],
        fields["qr"], fields["nr"], fields["qc"],
        fields["temperature"], pressure, fields["qv"],
        entry["nc"], entry["nwfa"], entry["nifa"],
        acc["ncten"], acc["nwfaten"], acc["nifaten"],
        shadow, boost, table_owner, dt)
    cp.cuda.Stream.null.synchronize()

    # THE ACCUMULATOR CONTRACT.  nc/nwfa/nifa state is read-only entry state
    # for the whole mp=28 call; only the scratch accumulators may move.
    for name, value in entry.items():
        assert cp.array_equal(value, entry_copy[name]), (
            f"the cold network wrote into the read-only {name} entry array")

    host = {k: cp.asnumpy(v).ravel() for k, v in fields.items()}
    host_acc = {k: cp.asnumpy(v).ravel() for k, v in acc.items()}
    host["snow_velocity_boost"] = cp.asnumpy(boost).ravel()
    host["graupel_number_shadow"] = cp.asnumpy(shadow).ravel()
    return host, host_acc, {k: cp.asnumpy(v).ravel()
                            for k, v in entry.items()}


def _max_relative(got, expected, *, floor_fraction=1.0e-7):
    """Largest relative difference over levels that carry real signal.

    Levels whose reference value is below ``floor_fraction`` of the column
    maximum are excluded: they are at the float32 noise floor of the field
    and a relative test there measures rounding, not physics.
    """
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected)))
    if scale == 0.0:
        assert np.array_equal(got, expected)
        return 0.0
    mask = np.abs(expected) > floor_fraction * scale
    relative = np.abs(got - expected) / np.maximum(np.abs(expected), 1e-300)
    return float(relative[mask].max()) if mask.any() else 0.0


def _apply_terminal_clamp(entry_per_kg, tendency, dt, floor):
    """module_mp_thompson.F:3977-3981, in float32 and in WRF's own units."""
    updated = np.float32(entry_per_kg) + np.float32(tendency) * np.float32(dt)
    return np.maximum(floor, np.minimum(_AERO_CEIL, updated))


# ---------------------------------------------------------------------------
# 1. Source gates.  These need neither a GPU nor the coefficient assets.
# ---------------------------------------------------------------------------

def test_cooper_nucleation_is_absent_from_the_aerosol_cold_kernel():
    """iceDeMott REPLACES Cooper; the Cooper path must be unreachable.

    module_mp_thompson.F:2622-2626 is an if/else on
    ``dustyIce .AND. is_aerosol_aware``.  thompson.cu evaluates the
    non-aerosol branch ``MIN(250.E3, TNO*EXP(ATO*(T_0-temp)))`` at :6645-6646
    and :7189-7200.  Leaving that expression in the mp=28 kernel -- even
    behind a flag -- is the single most plausible way to ship a scheme that
    runs, stays stable, and is wrong.
    """
    source = _KERNEL_SOURCE.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("//"))
    for literal in ("250.0e3f", "0.304f"):
        assert literal not in code, (
            f"Cooper (1986) nucleation literal {literal} is still present in "
            "thompson_aerosol_cold.cu; mp=28 must use iceDeMott instead")
    assert "thompson_ice_demott(" in code


def test_frozen_lookup_indices_are_not_hardcoded():
    """``idx_IN`` and ``idx_n`` must be live, not mp=8's 27 and 65."""
    source = _KERNEL_SOURCE.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("//"))
    assert "nuclei_bin = 27" not in code
    assert "cloud_number_bin = 65" not in code
    assert "thompson_aa_in_bin(xni_demott)" in code
    assert "thompson_aa_droplet_bin(nc_work)" in code
    # The 12 hardcoded droplet numbers of mp=8 must all be gone.
    assert "100.0e6f" not in code, (
        "a constant Nt_c survives in the mp=28 cold kernel; nc is prognostic")


def test_identity_gates_reduce_to_the_values_mp8_froze():
    """thompson_aa_droplet_bin(100e6) == 65 and thompson_aa_in_bin(1000) == 27.

    These are exactly the constants thompson.cu:7005 and :6820/:7008 embed.
    If the generalized index formulas do not reproduce them, they are wrong
    in a way that would ALSO have broken mp=8.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_launch import (
        probe_droplet_bin, probe_in_bin)

    assert int(probe_droplet_bin(
        cp.asarray(np.float32([100.0e6])))[0]) == 65
    assert int(probe_in_bin(cp.asarray(np.float32([1000.0])))[0]) == 27


# ---------------------------------------------------------------------------
# 2. Synthetic-table gates.  GPU, but no coefficient assets required.
# ---------------------------------------------------------------------------

def _synthetic_tables(cp, *, marked_in_axis=None, cloud_mass_value=0.0):
    """Zero coefficient tables of the production shapes.

    ``marked_in_axis`` paints a distinguishable value on every ``idx_IN``
    slice of ``cloud_to_ice_number`` -- slice ``s`` carries ``(s+1)*value``
    -- so the slice the kernel reads is recoverable from its output.
    ``cloud_mass_value`` fills ``cloud_to_ice_mass`` uniformly, which is
    needed to get past WRF's ``pri_wfz/(2*xm0i)`` term at :2611 without
    making that term the binding one.
    """
    def zeros(shape):
        return cp.zeros(shape, dtype=cp.float64, order="F")

    cloud_shape = (37, 100, 45, 55)
    cloud_mass = np.full(cloud_shape, cloud_mass_value,
                         dtype=np.float64, order="F")
    cloud_number = np.zeros(cloud_shape, dtype=np.float64, order="F")
    if marked_in_axis is not None:
        cloud_number[...] = (np.arange(55, dtype=np.float64) + 1.0)[
            None, None, None, :] * marked_in_axis

    return {
        "ice_deposition_partition": zeros((64, 55)),
        "ice_to_snow_mass": zeros((64, 55)),
        "ice_to_snow_number": zeros((64, 55)),
        "rain_cloud_efficiency": zeros((100, 100)),
        "snow_cloud_efficiency": zeros((100, 100)),
        "rain_snow": tuple(zeros((37, 9, 37, 37)) for _ in range(12)),
        "rain_graupel": tuple(zeros((37, 37, 1, 37, 37)) for _ in range(5)),
        "rain_freezing": tuple(zeros((37, 37, 45, 55)) for _ in range(4)),
        "cloud_freezing": (cp.asarray(cloud_mass, order="F"),
                           cp.asarray(cloud_number, order="F")),
    }


def test_idx_in_selects_the_slice_thompson_aa_in_bin_returns():
    """The freezeH2O ``idx_IN`` axis is genuinely indexed, not pinned at 27.

    mp=8 has only ever read slice 27 of a 55-slice axis.  This gate paints a
    different value on all 55 slices of ``tni_qcfz`` and checks, per level,
    that the droplet-number sink WRF forms from it
    (``ncten -= pni_wfz*orho``, :2991-2994) matches the slice
    ``thompson_aa_in_bin(iceDeMott(...))`` selects.

    ``ncten`` is the right observable: it is not rescaled by the cloud-water
    mass limiter at :2880-2890 and is not touched by any hydrometeor bound,
    so the table value survives to the output unmodified.  Snow, graupel and
    rain are absent, so pnc_scw, pnc_gcw and pnc_rcw are identically zero;
    the cloud water is above the 0.01e-3 autoconversion threshold, so
    pnc_wau (:2192-2193) is the one other member of the :2993-2995 sum and
    it is subtracted out here using the readback probe.  Its own agreement
    with WRF is gated by
    ``test_working_nu_c_is_recomputed_from_the_rediagnosed_nc``.

    Synthetic tables are used deliberately: this gate must run on a tree
    where the 255 MB freezeH2O.dat has not been staged.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import (
        launch_aa_cold_network, probe_cold_warm_loop)
    from woof.core.thompson_aerosol_launch import (
        probe_ice_demott, probe_in_bin)

    levels = 8
    dt = 10.0
    step = 1.0e3
    temperature = np.full(levels, 250.0, dtype=np.float32)
    pressure = np.full(levels, 80000.0, dtype=np.float32)
    qv = np.full(levels, 3.0e-4, dtype=np.float32)
    qc = np.full(levels, 3.0e-4, dtype=np.float32)
    # Six decades of ice-friendly aerosol -> six different idx_IN values.
    nifa_m3 = np.float32([5.0e3, 5.0e4, 5.0e5, 5.0e6,
                          5.0e7, 5.0e8, 5.0e9, 9.0e9])
    rho = (np.float32(0.622) * pressure
           / (np.float32(287.04) * temperature
              * (np.maximum(np.float32(1.0e-10), qv) + np.float32(0.622))))
    nifa = (nifa_m3 / rho).astype(np.float32)
    nc = (np.float32(1.0e8) / rho).astype(np.float32)

    def device(values):
        return cp.asarray(np.asarray(values, dtype=np.float32).copy())

    qi = device(np.zeros(levels)); ni = device(np.zeros(levels))
    qs = device(np.zeros(levels)); qg = device(np.zeros(levels))
    qr = device(np.zeros(levels)); nr = device(np.zeros(levels))
    d_qc = device(qc)
    d_t = device(temperature); d_p = device(pressure); d_qv = device(qv)
    nc_entry = device(nc)
    nwfa_entry = device(np.full(levels, 1.0e9, dtype=np.float32))
    nifa_entry = device(nifa)
    ncten = device(np.zeros(levels))
    nwfaten = device(np.zeros(levels))
    nifaten = device(np.zeros(levels))
    shadow = device(np.zeros(levels))
    boost = device(np.zeros(levels))

    tables = _synthetic_tables(
        cp, marked_in_axis=step, cloud_mass_value=1.0e-5)
    launch_aa_cold_network(
        qi, ni, qs, qg, qr, nr, d_qc, d_t, d_p, d_qv,
        nc_entry, nwfa_entry, nifa_entry, ncten, nwfaten, nifaten,
        shadow, boost,
        tables["ice_deposition_partition"], tables["ice_to_snow_mass"],
        tables["ice_to_snow_number"], tables["rain_snow"],
        tables["rain_graupel"], tables["rain_freezing"],
        tables["rain_cloud_efficiency"], tables["cloud_freezing"], dt,
        snow_cloud_efficiency=tables["snow_cloud_efficiency"])
    cp.cuda.Stream.null.synchronize()

    xni = cp.asnumpy(probe_ice_demott(
        device(temperature - np.float32(273.15)), device(rho),
        device(nifa_m3)))
    selected = cp.asnumpy(probe_in_bin(device(xni)))

    # pnc_wau also feeds ncten at :2993, and this column autoconverts.
    readback = probe_cold_warm_loop(
        device(qc), device(nc), device(np.zeros(levels)),
        device(np.zeros(levels)),
        device(np.full(levels, 1.0e9)), device(nifa),
        device(temperature), device(pressure), device(qv),
        tables["rain_cloud_efficiency"], dt)
    cp.cuda.Stream.null.synchronize()
    pnc_wau = cp.asnumpy(readback["pnc_wau"]).astype(np.float64)
    assert np.all(pnc_wau > 0.0), (
        "this fixture is supposed to autoconvert; if pnc_wau is zero the "
        "gate below is no longer subtracting anything")

    got = cp.asnumpy(ncten).astype(np.float64)
    orho = (np.float32(1.0) / rho).astype(np.float64)
    predicted = -(((selected.astype(np.float64) + 1.0) * step / dt)
                  + pnc_wau) * orho

    assert len(set(selected.tolist())) >= 6, (
        "the aerosol ladder must span several idx_IN slices for this gate "
        f"to mean anything; got {sorted(set(selected.tolist()))}")
    assert set(selected.tolist()) - {27}, (
        "no slice other than mp=8's frozen 27 was exercised")
    np.testing.assert_allclose(got, predicted, rtol=2.0e-6)


# ---------------------------------------------------------------------------
# 3. Oracle gates.
# ---------------------------------------------------------------------------
#
# ``_WRF_COLD_REFERENCE`` holds WRF v4.6.1's own per-level answers for the
# quantities this kernel is solely responsible for, at the point in
# module_mp_thompson.F where the cold network has finished
# (immediately after the tendency loop closes at line 3183, i.e. before the
# TAU+1 refresh at 3186).  It was produced like this, and is reproducible:
#
#   1. Copy the pristine module_mp_thompson.F to a scratch directory.  Do not
#      modify anything under wrf461-pristine.
#   2. Insert, immediately after the ``enddo`` at line 3183, a write of
#      qiten/niten, of nc(k)/mvd_c(k), and of the individual rate arrays
#      pnc_wau, pnc_rcw, pna_rca, pnd_rcd, pni_wfz, pnc_scw, pnc_gcw,
#      pna_sca, pnd_scd, pna_gca, pnd_gcd, pni_inu, pni_iha, prr_wau,
#      prr_rcw, pnr_wau, pnr_rcr.
#   3. Build with tools/thompson_wrf461_oracle/build_aero.sh's exact flags
#      (gfortran -O2 -cpp -DWRF_CHEM=0 -ffree-form) and run
#      run_column_aero.F90 unmodified for each scenario.
#   4. VERIFY THE INSTRUMENTATION IS INERT: the resulting -column.csv and
#      -surface.csv must be byte-identical to the committed fixtures under
#      woof/data/thompson/oracle-aero/.  They were, for all six scenarios.
#   5. qi/ni = before + {qi,ni}ten*dt; the accumulators are assembled with
#      WRF's own grouping and ``orho`` in float32, because WRF's
#      ``orho = 1./rho(k)`` is REAL(4) (:2959):
#          ncten   = -(pnc_wau + pnc_rcw + pni_wfz + pnc_scw + pnc_gcw)*orho
#          nwfaten = -(pna_rca + pna_sca + pna_gca + pni_iha)*orho
#          nifaten = -(pnd_rcd + pnd_scd + pnd_gcd + pni_inu)*orho
#      i.e. WRF's COMPLETE sums at :2964-2995 for a sub-freezing level.
#
# CORRECTION, WAVE 3, AND WHY THE VALUES MOVED.  Wave 2 assembled this table
# with the :2157-2234 members (pnc_wau, pnc_rcw, pna_rca, pnd_rcd) REMOVED,
# on the grounds that they "belong to WP-07's kernel".  They do -- for cells
# whose ENTRY temperature was >= 273.15 K.  Every level of all six aerosol
# fixtures is sub-freezing (230 K, 240 K or 260 K), so WP-07's kernel returns
# immediately on all of them and those four rates reached NOBODY.  Removing
# them from the reference made a reference that agreed with the defect.
# Restoring them moves ncten by up to 65% on aero-cold-overlap and 62% on
# aero-ice-demott-idxin, nwfaten by 1.3% and nifaten by 100% at levels where
# rain scavenging is the only ice-nuclei sink.  Cross-check that the
# re-derivation is sound: rebuilding the OLD three-term sums from this same
# instrumentation reproduces the wave-2 committed numbers to 2.4e-13
# relative, which is the 12-significant-digit CSV round trip.
#
# Values are the float64 print of WRF's REAL(4)/DOUBLE arrays.
_WRF_COLD_REFERENCE = {
  "aero-ice-demott-idxin": {
    "qi": (
        4.180941232335e-05, 7.071090749378e-05, 1.398117728968e-04,
        1.116437852033e-04, 2.011003743974e-05, 4.691016499692e-06,
        1.203758017709e-06, 1.287454324483e-05, 2.891982209263e-09,
        4.401593256542e-09, 2.656988762340e-07, 1.603875716683e-05,
        2.891982209263e-09, 5.483495879588e-09, 3.310112361987e-07,
        1.998154402827e-05, 2.891982209263e-09, 6.831620269487e-09,
        4.123954155943e-07, 2.489449343557e-05, 2.891982486819e-09,
        8.511690796631e-09, 5.138228331703e-07, 3.101795300609e-05,
    ),
    "ni": (
        7.103125781250e+05, 7.561414843750e+05, 8.049287500000e+05,
        8.568650781250e+05, 9.121539843750e+05, 5.404126464844e+04,
        2.132821875000e+05, 1.100374375000e+06, 2.891982116699e+03,
        4.401593322754e+03, 2.656988671875e+05, 1.413135625000e+06,
        2.891982116699e+03, 5.483495483398e+03, 3.310112500000e+05,
        1.814876562500e+06, 2.891982421875e+03, 6.831620483398e+03,
        4.123954296875e+05, 2.330965937500e+06, 2.891982421875e+03,
        8.511691284180e+03, 5.138228515625e+05, 2.994038125000e+06,
    ),
    "ncten": (
        -4.364549250415e+05, -6.288402376569e+05, -2.771781640357e+06,
        -3.612752072946e+06, -1.410580469697e+05, -1.383075428475e+04,
        -1.686331500239e+03, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
    ),
    "nwfaten": (
        -6.016964331911e+02, -9.349410844293e+02, -9.610409554730e+02,
        -6.535066804368e+02, -2.939739642987e+02, -8.748193126563e+01,
        -1.722179843110e+01, -2.242793352934e+00, -1.932193546923e-01,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
    ),
    "nifaten": (
        -2.725379817981e-03, -4.207668121754e-01, -4.298128695215e+01,
        -2.904958485347e+03, -1.299038720204e-03, -3.843483033202e-02,
        -1.991219037711e+04, -1.287464129659e+06, -2.891982270377e+02,
        -4.401593285734e+02, -2.656988727877e+04, -1.603875703816e+06,
        -2.891982121973e+02, -5.483495748108e+02, -3.310112345931e+04,
        -1.998154494066e+06, -2.891982321888e+02, -6.831620517526e+02,
        -4.123954282031e+04, -2.489449373039e+06, -2.891982369899e+02,
        -8.511691076051e+02, -5.138228385621e+04, -3.101795262015e+06,
    ),
  },
  "aero-cloud-freeze-nc": {
    "qi": (
        2.420336613795e-05, 2.258929953314e-05, 1.218150487148e-05,
        2.068772602115e-06, 4.159124955549e-07, 3.748435517537e-07,
        6.064253543769e-09, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
    ),
    "ni": (
        7.102527343750e+05, 7.560736718750e+05, 8.048518750000e+05,
        8.567779687500e+05, 2.079562500000e+05, 1.874217773438e+05,
        3.032126770020e+03, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
    ),
    "ncten": (
        -3.779872030682e+05, -6.282249021874e+05, -6.090752598806e+05,
        -1.034386323043e+05, -2.079562512607e+04, -1.874217730627e+04,
        -3.032126804492e+02, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
    ),
    "nwfaten": (
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
    ),
    "nifaten": (
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
    ),
  },
  "aero-cold-overlap": {
    "qi": (
        0.000000000000e+00, 0.000000000000e+00, 1.605258527029e-04,
        1.015553446222e-04, 4.903340949114e-05, 1.249578360052e-05,
        1.006156472055e-06, 5.861954343999e-07, 2.380513531733e-07,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
    ),
    "ni": (
        0.000000000000e+00, 0.000000000000e+00, 8.049286854172e+05,
        8.568650448208e+05, 9.121538617682e+05, 9.710122378919e+05,
        3.372172329009e+05, 2.253264224219e+05, 2.380512009176e+05,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
    ),
    "ncten": (
        -0.000000000000e+00, -0.000000000000e+00, -2.684460146297e+06,
        -2.568337252732e+06, -1.759104235085e+06, -4.886020150066e+05,
        -4.406556387812e+04, -8.359065623472e+02, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
    ),
    "nwfaten": (
        -0.000000000000e+00, -0.000000000000e+00, -3.628333134091e+05,
        -3.059338810013e+05, -1.846954716926e+05, -7.984512058839e+04,
        -2.472296210475e+04, -5.484886242696e+03, -8.691409989667e+02,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
    ),
    "nifaten": (
        -0.000000000000e+00, -0.000000000000e+00, -1.053834652388e+03,
        -8.906413200441e+02, -5.416853610657e+02, -2.371900626900e+02,
        -7.483736489962e+01, -4.226909900815e+03, -4.763820601415e+03,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
        -0.000000000000e+00, -0.000000000000e+00, -0.000000000000e+00,
    ),
  },
}


def test_frozen_aerosol_scavenging_matches_committed_oracle(classic_tables):
    """Fixture 111 closes END TO END on both aerosol species.

    aero-scav-frozen is 260 K with ssati == 0 exactly, snow and graupel but
    no cloud, no rain and no ice.  Every vapour rate, every nucleation rate
    and every warm-section aerosol sink is therefore identically zero, and
    ``nwfa``/``nifa`` have no sedimentation term anywhere in
    module_mp_thompson.F.  The committed ``after`` column is exactly
    ``clamp(before + nwfaten*dt)`` with this kernel's four scavenging rates
    -- pna_sca, pnd_scd, pna_gca and pnd_gcd -- and nothing else.

    MEASURED: bit-identical in float32 on both species, all 24 levels.
    """
    scenario, dt = "aero-scav-frozen", 10.0
    before, after = _column(scenario)
    _, acc, _ = _run_cold_network(scenario, dt, classic_tables)

    for name, tendency, floor in (
            ("nwfa_per_kg", acc["nwfaten"], _NWFA_FLOOR),
            ("nifa_per_kg", acc["nifaten"], _NIFA_FLOOR)):
        got = _apply_terminal_clamp(
            _host(before, name), tendency, dt, floor)
        expected = _host(after, name)
        np.testing.assert_array_equal(got, expected)

    # The scavenging must actually have done something, on both species, and
    # only where the collector exists.
    snow = _host(before, "qs")
    assert np.any(acc["nwfaten"] < 0.0)
    assert np.any(acc["nifaten"] < 0.0)
    assert np.all(acc["nwfaten"][snow == 0.0] == 0.0)
    assert np.all(acc["nifaten"][snow == 0.0] == 0.0)
    # Bigg freezing consumes no aerosol and there is no cloud water, so the
    # droplet-number accumulator must be untouched.
    assert np.all(acc["ncten"] == 0.0)


def test_demott_nucleation_matches_committed_oracle(classic_tables):
    """Fixture 112 closes END TO END on nifa: pni_inu is its only sink.

    aero-ice-demott-dep is 240 K at 30% ice supersaturation with no
    condensate at all, so deposition nucleation is the only active process
    and ``nifaten = -pni_inu*orho`` (module_mp_thompson.F:2974-2976, dustyIce
    is a PARAMETER .true.).  Nothing else in the call touches nifa.

    MEASURED: bit-identical in float32, all 24 levels.
    """
    scenario, dt = "aero-ice-demott-dep", 10.0
    before, after = _column(scenario)
    state, acc, _ = _run_cold_network(scenario, dt, classic_tables)

    got = _apply_terminal_clamp(
        _host(before, "nifa_per_kg"), acc["nifaten"], dt, _NIFA_FLOOR)
    np.testing.assert_array_equal(got, _host(after, "nifa_per_kg"))

    # DeMott, not Cooper.  At 240 K Cooper would ask for
    # MIN(250e3, 5*exp(0.304*33.15)) = 1.14e5 crystals per cubic metre
    # independent of aerosol; DeMott at nifa ~ 1e6 m^-3 gives far fewer.
    # The produced ice number must follow the aerosol, not the temperature.
    ice_number_m3 = state["ni"] * (
        np.float32(0.622) * _host(before, "p_pa")
        / (np.float32(287.04) * _host(before, "temp_k")
           * (np.maximum(np.float32(1.0e-10), _host(before, "qv"))
              + np.float32(0.622))))
    cooper = 5.0 * np.exp(0.304 * (273.15 - 240.0))
    assert np.all(ice_number_m3 < 0.5 * cooper), (
        "the nucleated crystal number is at the Cooper level; mp=28 must "
        "use iceDeMott")
    assert np.all(acc["nwfaten"] == 0.0)
    assert np.all(acc["ncten"] == 0.0)


def test_koop_haze_freezing_matches_committed_oracle(classic_tables):
    """Fixture 114 closes END TO END on nwfa: pni_iha is its only sink.

    aero-ice-koop is 230 K with satw = 0.975 and ssati = 0.493 -- above the
    0.4 Koop gate, below liquid saturation so nothing condenses -- and no
    condensate, so ``nwfaten = -pni_iha*orho`` (:2964-2965: Koop consumes
    WATER-friendly aerosol, not ice nuclei) and ``nifaten = -pni_inu*orho``.
    The CCN ladder makes the fixture self-evidencing: every level shares one
    temperature and one saturation, so all level-to-level variation in the
    ice produced is attributable to iceKoop.

    TOLERANCE, MEASURED AND ATTRIBUTED.  nifa is bit-identical.  nwfa agrees
    to 1.8e-7 relative, and the residual is NOT in this kernel:

      * ``prob_h = MIN(1 - exp(-J*ar_volume*dt), 1)`` has ``J*V*dt ~ 3.4e-5``
        here, so the float32 subtraction quantizes prob_h to multiples of
        2^-24, i.e. a relative grid of 1.8e-3.
      * Feeding ``thompson_ice_koop`` the qvs that gfortran's RSLF produces
        reproduces WRF's pni_iha to 5.7e-8 on all 24 levels.  Feeding it the
        device ``thompson_rslf``'s qvs moves 6 of 24 levels by 1 to 4 grid
        steps, i.e. up to 7.1e-3 relative in the Koop ice number.
      * ``thompson_rslf`` in thompson_aerosol_common.cuh deliberately keeps
        thompson.cu's FMA-contracted Horner so mp=8 and mp=28 evaluate the
        shared fit identically; it differs from gfortran's REAL(4) Horner by
        up to 195 ulps here (and thompson_rsif by up to 400).

    Diluted into the nwfa field the effect is 1.8e-7, which is why the
    end-to-end aerosol gate is still tight; the ice number itself is not,
    and that is reported rather than absorbed.
    """
    scenario, dt = "aero-ice-koop", 10.0
    before, after = _column(scenario)
    _, acc, _ = _run_cold_network(scenario, dt, classic_tables)

    nifa = _apply_terminal_clamp(
        _host(before, "nifa_per_kg"), acc["nifaten"], dt, _NIFA_FLOOR)
    np.testing.assert_array_equal(nifa, _host(after, "nifa_per_kg"))

    nwfa = _apply_terminal_clamp(
        _host(before, "nwfa_per_kg"), acc["nwfaten"], dt, _NWFA_FLOOR)
    assert _max_relative(nwfa, _host(after, "nwfa_per_kg")) <= 5.0e-7

    # Koop must actually have fired, and its sink must track the CCN ladder
    # rather than being constant.
    assert np.all(acc["nwfaten"] < 0.0)
    assert len(set(np.round(acc["nwfaten"], 6).tolist())) >= 5


#: THE PER-CELL RATCHET on the comparison below: the largest relative
#: difference this kernel shows against ``_WRF_COLD_REFERENCE``, per fixture
#: and per field.  The gate is 1.0e-6; these are the accurate distances from it
#: and they are asserted individually so a residual cannot triple and still
#: pass.
#:
#: RE-DERIVED IN WP-14, AND THE NUMBERS MOVED.  They moved because WP-13b
#: contraction-pinned the source networks' terminal multiply-add (WRF's
#: :3973-4023 ``q1d + qten*DT``, which the gfortran -O2 oracle cannot fuse
#: and nvrtc did), which changes woof/core/kernels/thompson_aerosol_cold.cu's
#: own arithmetic.  The reference values themselves did NOT move: they are
#: WRF's, taken from the instrumented build described above, and every one of
#: them still reproduces at or under 1.7e-07.  The previous prose (wave 3)
#: read qi 1.20e-07 / ni 6.60e-08 on aero-ice-demott-idxin, qi 9.95e-08 /
#: ncten 3.43e-07 / nwfaten 2.55e-07 / nifaten 5.16e-07 on aero-cold-overlap.
#: Net: six of the fifteen cells improved (aero-cold-overlap's three
#: accumulators by factors of 4 to 9), three grew inside the same decade, and
#: the gate never moved.
#:
#: RE-MEASURED when the cold network was re-transcribed in WRF's arithmetic
#: order (thompson_aerosol_common.cuh, ThompsonAaLevel): aero-ice-demott-idxin
#: qi 7.064e-08 -> 4.722e-08 and ni 8.442e-08 -> 4.175e-08; aero-cold-overlap
#: ni 1.515e-07 -> 6.692e-08 and ncten 5.450e-08 -> 3.863e-08.  What is left
#: is the float32 round trip of the reference CSV.  aero-cold-overlap's qi is
#: the declared rain-graupel divergence (:data:`_COLD_DECLARED_DIVERGENCE`).
_COLD_PER_LEVEL_MEASURED = {
    "aero-ice-demott-idxin": {
        "qi": 4.722e-08, "ni": 4.175e-08, "ncten": 3.956e-08,
        "nwfaten": 4.224e-08, "nifaten": 5.239e-08},
    "aero-cloud-freeze-nc": {
        "qi": 2.013e-08, "ni": 3.647e-08, "ncten": 4.328e-08,
        "nwfaten": 0.0, "nifaten": 0.0},
    "aero-cold-overlap": {
        "qi": 1.619e-04, "ni": 6.692e-08, "ncten": 3.863e-08,
        "nwfaten": 4.259e-08, "nifaten": 5.509e-08},
}

#: The cells held to their pin but not to the 1e-6 gate, each a declared
#: divergence from WRF v4.6.1.  aero-cold-overlap carries rain and graupel
#: together below 0 C; WRF reads its rain-graupel collision tables out of
#: bounds when the scheme is not hail aware (dimNRHG = 1, idx_bg = 5,
#: module_mp_thompson.F:465, :607-615, :2527-2545) and WOOF reads the one slab
#: the tables hold, and the rain the two codes then freeze into ice differs.
_COLD_DECLARED_DIVERGENCE = {("aero-cold-overlap", "qi")}


@pytest.mark.parametrize(
    "scenario,dt", (("aero-ice-demott-idxin", 10.0),
                    ("aero-cloud-freeze-nc", 10.0),
                    ("aero-cold-overlap", 50.0)))
def test_cold_network_matches_wrf_per_level(scenario, dt, classic_tables):
    """Per-level comparison against WRF's own post-cold-network answers.

    These three fixtures cannot be closed end to end by one kernel -- they
    run warm-section number sinks, CCN activation and hydrometeor fallout as
    well -- so the reference is ``_WRF_COLD_REFERENCE``, whose derivation is
    documented above it.

    ``aero-ice-demott-idxin`` is the ACCEPTANCE fixture for this package: its
    ice-nuclei ladder (5e3, 5e5, 5e7, 5e9 per cubic metre) drives iceDeMott
    across four decades, which moves ``idx_IN`` off 27 and onto table slices
    woof has never read, and it carries supercooled cloud AND rain so both
    freezeH2O families are indexed on that axis.  ``aero-cloud-freeze-nc``
    drives the droplet bin ``idx_n`` across the nc ladder (30, 100, 300,
    1000, 1800 per cc) that mp=8 pins at 65.

    MEASURED maxima are in :data:`_COLD_PER_LEVEL_MEASURED` and are asserted
    cell by cell, not just against the 1.0e-6 gate -- so a change that doubles
    a 4e-08 residual and still clears the gate fails here.  The numbers below
    are the values that dict carries; both are re-derived, not re-pinned,
    whenever this file is re-measured.
    """
    state, acc, _ = _run_cold_network(scenario, dt, classic_tables)
    reference = _WRF_COLD_REFERENCE[scenario]
    source = dict(state)
    source.update(acc)
    recorded = _COLD_PER_LEVEL_MEASURED[scenario]
    assert set(recorded) == set(reference), sorted(
        set(recorded) ^ set(reference))
    for field, expected in reference.items():
        got = source[field]
        assert len(expected) == _LEVELS
        worst = _max_relative(got, expected)
        assert worst <= 1.0e-6 or (scenario, field) in (
            _COLD_DECLARED_DIVERGENCE), (
            f"{scenario} {field}: max relative difference {worst:.3e} "
            "against WRF v4.6.1")
        # ...and the RATCHET, which is the strictly stronger statement.
        pin = recorded[field]
        if pin == 0.0:
            assert worst == 0.0, (
                f"{scenario} {field} is published as BIT-EXACT against WRF "
                f"and now measures {worst:.3e}")
        else:
            assert worst <= pin * 1.01, (
                f"{scenario} {field}: grew from the published {pin:.3e} to "
                f"{worst:.3e}.  Re-derive the number before updating it: a "
                "stale literal and a regression look identical here.")


def test_vts_boost_and_entry_state_contract(classic_tables):
    """vts_boost is reset for EVERY cell; entry aerosol state never moves.

    WRF sets vts_boost(k) = 1.0 unconditionally at the top of the cold loop
    (:2243).  woof's column sedimentation consumes the complete field, so a
    warm or hydrometeor-free cell that kept a stale value would silently
    accelerate snow fallout somewhere else in the column.
    """
    state, _, _ = _run_cold_network("aero-cold-overlap", 50.0, classic_tables)
    boost = state["snow_velocity_boost"]
    assert np.all(boost >= 1.0)
    assert np.all(boost <= 1.5)
    # The seed was 7.0; nothing may survive.
    assert not np.any(boost == 7.0)




# ---------------------------------------------------------------------------
# 4. WAVE-3 REGRESSION GATES.
# ---------------------------------------------------------------------------
#
# Both defects these gates cover were SILENT: the kernel ran, stayed finite,
# stayed stable, and the whole wave-2 suite passed.  Neither is observable in
# any committed column fixture, which is precisely why they survived.
#
# ``_WRF_COLD_WARM_LOOP`` is 54 rows drawn from an 11340-row Fortran oracle
# produced by tools/thompson_wrf461_oracle/probe_cold_warm_loop_aero.F90,
# which is COMMITTED and is rebuilt and rerun by build_aero_probes.sh (it used
# to live only in an agent scratch directory, and this note used to call it "a
# scratch program").  It LINKS the same compiled module_mp_thompson.o
# tools/thompson_wrf461_oracle/build_aero.sh builds from
# wrf461-pristine/phys/module_mp_thompson.F, calls thompson_init
# exactly as run_column_aero.F90 does, and then evaluates
# module_mp_thompson.F:1826-1842 (BOTH nu_c stages) and :2144-2232 verbatim
# with the module's own public WGAMMA, Eff_aero, t_Efrw and Dr.  ccg/ocg1/
# ocg2/cce are PRIVATE in the module and are rebuilt from thompson_init's own
# expressions at :671-685.  The regenerated CSV reproduces the SHA-256 pinned
# in tools/thompson_wrf461_oracle/PROBE_ORACLE_RECEIPTS.md.
#
# Every row is SUB-FREEZING (272, 265, 255, 245 or 232 K), which is the point:
# WRF's warm-rain loop at :2157-2234 has no temperature guard, so every rate
# in it runs at these levels, and woof gives every one of these levels to
# thompson_aa_cold_network.  WP-07's own oracle ladder is entirely at or above
# 275 K, so before wave 3 nothing had ever gated these four rates on the cold
# side.
#
# 34 of the 54 rows are states where WRF's :1832 nu_c and its :2170 nu_c
# DISAGREE.  The full 11340-row sweep has 3528 such states, 2058 of them with
# prr_wau > 0 (i.e. where the disagreement is observable at all).
#
# MEASURED over the full 11340 rows on an RTX 5090 (CuPy 14.1.1, nvrtc
# -std=c++17), GPU against gfortran -O2 REAL(4)/DOUBLE, on the REGENERATED
# oracle (which reproduces its committed SHA-256):
#     nu_c_entry, nu_c_working      EXACT on all 11340
#     nc_m3, mvd_c, mvd_r           EXACT
#     prr_wau, pnr_wau, pnc_wau     EXACT
#     pnc_rcw, pna_rca, pnd_rcd     <= 7.6e-16   (the CSV round trip)
#
# THAT IS A CHANGE, AND THE CAUSE WAS UPSTREAM.  These lines used to read
#     mvd_c <= 1.3e-7 ; nc_m3, pnc_rcw <= 3.3e-7 ;
#     pnc_wau <= 2.0e-6 (1.973e-6) ; prr_wau, pnr_wau <= 2.4e-6 (2.311e-6)
# and were attributed here, correctly, to thompson_aa_cloud_dist in
# thompson_aerosol_common.cuh -- CUDA's powf (~2 ulp) where gfortran lowers
# REAL(4)**REAL(4) to glibc's correctly rounded powf, amplified by the Dc_b
# cancellation at :2182, since this ladder reaches rc = 0.15 kg m^-3 where
# Dc_b subtracts two sixth powers that agree to five digits.  That helper is
# now correctly-rounded AND contraction-pinned in the shared header, and the
# whole of the residual it was carrying is gone: this file's four rates are
# bit-exact against compiled WRF on 11340 states.  Contraction-pinning
# everything this file owns had already taken prr_wau from 1.03e-5 to
# 2.31e-6; the header's repair took the remaining 2.31e-6 to zero.

# case, pres, temp, qv, qc, nc_per_kg, qr, nr_per_kg, nwfa_per_kg, nifa_per_kg,
#   nu_c_entry, nu_c_working, nc_m3, mvd_c, mvd_r,
#   prr_wau, pnr_wau, pnc_wau, pnc_rcw, pna_rca, pnd_rcd
_WRF_COLD_WARM_LOOP_DT = 30.0
#: One row per state:
#   p_pa, temp_k, qv, qc, nc_per_kg, qr, nr_per_kg, nwfa_per_kg,
#   nifa_per_kg, nu_c_entry(:1832), nu_c_working(:2170),
#   nc_m3(:1840), mvd_c(:2174), mvd_r(:2149), prr_wau(:2189),
#   pnr_wau(:2191), pnc_wau(:2192), pnc_rcw(:2205), pna_rca(:2214),
#   pnd_rcd(:2219)
_WRF_COLD_WARM_LOOP = (
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 0.0000000000e+00, 0.0000000000e+00, 8.2261950000e+06, 4.1130976562e+03,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 2.507182303816e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 0.0000000000e+00, 0.0000000000e+00, 2.4678585600e+08, 8.2261950000e+05,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 2.507182303816e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 0.0000000000e+00, 0.0000000000e+00, 2.4678584320e+09, 4.1130974720e+09,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 2.507182303816e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 9.9999997474e-06, 7.2963281250e+05, 8.2261950000e+06, 4.1130976562e+03,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 5.999998029438e-05, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.729732519703e+01, 1.999400010008e-03),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 9.9999997474e-06, 7.2963281250e+05, 2.4678584320e+09, 4.1130974720e+09,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 5.999998029438e-05, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 4.674953007184e+03, 1.999400051354e+03),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 9.9999997474e-06, 2.4625107422e+03, 2.4678584320e+09, 4.1130974720e+09,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 3.999998734798e-04, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 5.295853253061e+02, 2.341964582957e+02),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 9.9999997474e-06, 4.6696502686e+01, 8.2261950000e+06, 4.1130976562e+03,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 1.499999547377e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 4.544458052685e-01, 4.979925462871e-05),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 9.9999997474e-06, 4.6696502686e+01, 2.4678584320e+09, 4.1130974720e+09,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 1.499999547377e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.228231920515e+02, 4.979925783426e+01),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 0.0000000000e+00, 1.6452389956e+00, 2.0000000950e-03, 1.4592657600e+08, 2.4678585600e+08, 8.2261950000e+05,
     0, 0,
     2.000000000000e+00, 9.999999974752e-07, 5.999997665640e-05, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 9.349905041947e+04, 7.997599281176e+01),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.9999999920e-12, 1.6452389956e+00, 0.0000000000e+00, 0.0000000000e+00, 8.2261950000e+06, 4.1130976562e+03,
     15, 15,
     2.000002622604e+00, 1.456053087168e-05, 2.507182303816e-03, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 9.9999997474e-05, 1.6452389956e+00, 9.9999997474e-06, 4.6696502686e+01, 8.2261950000e+06, 4.1130976562e+03,
     15, 15,
     3.252526875000e+05, 4.999999873689e-05, 1.499999547377e-03, 1.148169189946e-06, 1.169515479128e+02, 1.084175683594e+04, 1.986845350661e+01, 4.544458052685e-01, 4.979925462871e-05),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 1.6452389956e+00, 0.0000000000e+00, 0.0000000000e+00, 8.2261950000e+06, 4.1130976562e+03,
     15, 6,
     2.602021600000e+08, 4.999999873689e-05, 2.507182303816e-03, 3.241677070037e-03, 8.254863051064e+05, 8.673406000000e+06, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 1.6452389956e+00, 9.9999997474e-06, 7.2963281250e+05, 8.2261950000e+06, 4.1130976562e+03,
     15, 6,
     2.602021600000e+08, 4.999999873689e-05, 5.999998029438e-05, 3.241677070037e-03, 8.254863051064e+05, 8.673406000000e+06, 2.132409687849e+04, 1.729732519703e+01, 1.999400010008e-03),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 1.6452389956e+00, 2.0000000950e-03, 1.4592657600e+08, 8.2261950000e+06, 4.1130976562e+03,
     15, 6,
     2.602021600000e+08, 4.999999873689e-05, 5.999997665640e-05, 3.241677070037e-03, 8.254863051064e+05, 8.673406000000e+06, 4.264819103455e+06, 3.459464818574e+03, 3.998799764755e-01),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 1.6452389956e+00, 0.0000000000e+00, 0.0000000000e+00, 8.2261950000e+06, 4.1130976562e+03,
     15, 4,
     4.878790720000e+08, 4.999999873689e-05, 2.507182303816e-03, 6.078144535422e-03, 2.321680244229e+06, 1.626263700000e+07, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 1.6452389956e+00, 0.0000000000e+00, 0.0000000000e+00, 2.4678585600e+08, 8.2261950000e+05,
     15, 4,
     4.878790720000e+08, 4.999999873689e-05, 2.507182303816e-03, 6.078144535422e-03, 2.321680244229e+06, 1.626263700000e+07, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 1.6452389956e+00, 9.9999997474e-06, 7.2963281250e+05, 8.2261950000e+06, 4.1130976562e+03,
     15, 4,
     4.878790720000e+08, 4.999999873689e-05, 5.999998029438e-05, 6.078144535422e-03, 2.321680244229e+06, 1.626263700000e+07, 3.998268390061e+04, 1.729732519703e+01, 1.999400010008e-03),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 1.6452389956e+00, 2.0000000950e-03, 1.4592657600e+08, 8.2261950000e+06, 4.1130976562e+03,
     15, 4,
     4.878790720000e+08, 4.999999873689e-05, 5.999997665640e-05, 6.078144535422e-03, 2.321680244229e+06, 1.626263700000e+07, 7.996536269666e+06, 3.459464818574e+03, 3.998799764755e-01),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 9.9999997474e-05, 8.2261953125e+03, 0.0000000000e+00, 0.0000000000e+00, 2.4678585600e+08, 8.2261950000e+05,
     15, 15,
     3.252526875000e+05, 4.999999873689e-05, 2.507182303816e-03, 1.148169189946e-06, 1.169515479128e+02, 1.084175683594e+04, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.0000000475e-03, 8.2261953125e+03, 2.0000000950e-03, 4.9250218750e+05, 2.4678585600e+08, 8.2261950000e+05,
     15, 15,
     3.252527250000e+06, 4.999999873689e-05, 3.999998734798e-04, 4.052096483065e-05, 4.127431393703e+03, 1.084175781250e+05, 4.963896133688e+04, 1.059170731290e+04, 9.367859673343e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 3.9999999106e-02, 8.2261953125e+03, 0.0000000000e+00, 0.0000000000e+00, 2.4678584320e+09, 4.1130974720e+09,
     15, 10,
     1.301010800000e+08, 4.999999873689e-05, 2.507182303816e-03, 1.620838535018e-03, 2.476458747287e+05, 4.336703000000e+06, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 3.9999999106e-02, 8.2261950000e+05, 9.9999997474e-06, 2.4625107422e+03, 8.2261950000e+06, 4.1130976562e+03,
     15, 10,
     1.301010800000e+08, 4.999999873689e-05, 3.999998734798e-04, 1.620838535018e-03, 2.476458747287e+05, 4.336703000000e+06, 9.927790402018e+03, 1.959465591798e+00, 2.341964454302e-04),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 3.9999999106e-02, 8.2261950000e+05, 2.0000000950e-03, 1.4592657600e+08, 8.2261950000e+06, 4.1130976562e+03,
     15, 10,
     1.301010800000e+08, 4.999999873689e-05, 5.999997665640e-05, 1.620838535018e-03, 2.476458747287e+05, 4.336703000000e+06, 2.132409551727e+06, 3.459464818574e+03, 3.998799764755e-01),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 8.2261950000e+05, 0.0000000000e+00, 0.0000000000e+00, 8.2261950000e+06, 4.1130976562e+03,
     15, 6,
     2.602021600000e+08, 4.999999873689e-05, 2.507182303816e-03, 3.241677070037e-03, 8.254863051064e+05, 8.673406000000e+06, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 9.9999997474e-05, 2.4678584000e+07, 0.0000000000e+00, 0.0000000000e+00, 8.2261950000e+06, 4.1130976562e+03,
     15, 15,
     3.000003800000e+07, 2.175054214604e-05, 2.507182303816e-03, 6.384891226840e-11, 6.503596497561e-03, 1.185071592617e+01, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 3.9999999106e-02, 8.2261952000e+07, 2.0000000950e-03, 4.9250218750e+05, 8.2261950000e+06, 4.1130976562e+03,
     12, 9,
     1.393349440000e+08, 4.999999873689e-05, 3.999998734798e-04, 1.620838535018e-03, 2.751621120745e+05, 4.644498500000e+06, 2.126482322810e+06, 3.918931643844e+02, 4.683929458695e-02),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 8.2261952000e+07, 0.0000000000e+00, 0.0000000000e+00, 8.2261950000e+06, 4.1130976562e+03,
     12, 6,
     2.786698880000e+08, 4.999999873689e-05, 2.507182303816e-03, 3.241677070037e-03, 8.254863051064e+05, 9.288997000000e+06, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 8.2261952000e+07, 9.9999997474e-06, 7.2963281250e+05, 8.2261950000e+06, 4.1130976562e+03,
     12, 6,
     2.786698880000e+08, 4.999999873689e-05, 5.999998029438e-05, 3.241677070037e-03, 8.254863051064e+05, 9.288997000000e+06, 2.283756399980e+04, 1.729732519703e+01, 1.999400010008e-03),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 8.2261952000e+07, 2.0000000950e-03, 1.4592657600e+08, 8.2261950000e+06, 4.1130976562e+03,
     12, 6,
     2.786698880000e+08, 4.999999873689e-05, 5.999997665640e-05, 3.241677070037e-03, 8.254863051064e+05, 9.288997000000e+06, 4.567512508396e+06, 3.459464818574e+03, 3.998799764755e-01),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 8.2261952000e+07, 0.0000000000e+00, 0.0000000000e+00, 8.2261950000e+06, 4.1130976562e+03,
     12, 4,
     5.225060160000e+08, 4.999999873689e-05, 2.507182303816e-03, 6.078144535422e-03, 2.321680244229e+06, 1.741686800000e+07, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 8.2261952000e+07, 9.9999997474e-06, 7.2963281250e+05, 8.2261950000e+06, 4.1130976562e+03,
     12, 4,
     5.225060160000e+08, 4.999999873689e-05, 5.999998029438e-05, 6.078144535422e-03, 2.321680244229e+06, 1.741686800000e+07, 4.282043300040e+04, 1.729732519703e+01, 1.999400010008e-03),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 8.2261952000e+07, 2.0000000950e-03, 1.4592657600e+08, 8.2261950000e+06, 4.1130976562e+03,
     12, 4,
     5.225060160000e+08, 4.999999873689e-05, 5.999997665640e-05, 6.078144535422e-03, 2.321680244229e+06, 1.741686800000e+07, 8.564086053395e+06, 3.459464818574e+03, 3.998799764755e-01),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 1.2339292800e+08, 0.0000000000e+00, 0.0000000000e+00, 2.4678584320e+09, 4.1130974720e+09,
     9, 4,
     5.796281600000e+08, 4.999999873689e-05, 2.507182303816e-03, 6.078144535422e-03, 2.321680244229e+06, 1.932094000000e+07, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 1.6452390400e+08, 9.9999997474e-06, 2.4625107422e+03, 8.2261950000e+06, 4.1130976562e+03,
     7, 5,
     3.433506240000e+08, 4.999999873689e-05, 3.999998734798e-04, 3.241677070037e-03, 9.905834989148e+05, 1.144502100000e+07, 2.620049810196e+04, 1.959465591798e+00, 2.341964454302e-04),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 1.5000000596e-01, 2.4678585600e+08, 9.9999997474e-06, 7.2963281250e+05, 8.2261950000e+06, 4.1130976562e+03,
     5, 3,
     7.555825920000e+08, 4.999999873689e-05, 5.999998029438e-05, 6.078144535422e-03, 3.095573658971e+06, 2.518608800000e+07, 6.192153033720e+04, 1.729732519703e+01, 1.999400010008e-03),
    (9.5000000000e+04, 2.7200000000e+02, 5.8926310157e-04, 7.9999998212e-02, 4.1130976000e+08, 2.0000000950e-03, 9.3393007812e+03, 8.2261950000e+06, 4.1130976562e+03,
     4, 4,
     5.000004800000e+08, 4.999999873689e-05, 1.499999663793e-03, 3.241677070037e-03, 1.238229457660e+06, 1.666668400000e+07, 6.108628898024e+06, 9.088918085663e+01, 9.959851869575e-03),
    (8.0000000000e+04, 2.6500000000e+02, 6.9974997314e-04, 3.9999999106e-02, 2.8556692000e+07, 9.9999997474e-06, 7.2963281250e+05, 2.8556692480e+09, 4.7594485760e+09,
     15, 11,
     1.124328560000e+08, 4.999999873689e-05, 5.999998029438e-05, 1.400722539984e-03, 1.945587493551e+05, 3.747762000000e+06, 8.565627513680e+03, 4.302539801646e+03, 1.830823154843e+03),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 9.9999997474e-05, 2.4434998035e+00, 2.0000000950e-03, 4.9250218750e+05, 3.6652496000e+08, 1.2217498750e+06,
     15, 15,
     2.189967187500e+05, 4.999999873689e-05, 3.999999025837e-04, 5.205220077187e-07, 5.301993387251e+01, 7.299891113281e+03, 2.742507824380e+03, 8.462859600304e+03, 7.296144419005e+00),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 7.9999998212e-02, 2.4434998035e+00, 0.0000000000e+00, 0.0000000000e+00, 1.2217499000e+07, 6.1087495117e+03,
     15, 8,
     1.751973760000e+08, 4.999999873689e-05, 2.507182303816e-03, 2.182661788538e-03, 4.168577042508e+05, 5.839913000000e+06, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 7.9999998212e-02, 2.4434998035e+00, 9.9999997474e-06, 7.2963281250e+05, 1.2217499000e+07, 6.1087495117e+03,
     15, 8,
     1.751973760000e+08, 4.999999873689e-05, 5.999998029438e-05, 2.182661788538e-03, 4.168577042508e+05, 5.839913000000e+06, 1.178136924459e+04, 1.383173273464e+01, 1.577123956323e-03),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 7.9999998212e-02, 2.4434998035e+00, 2.0000000950e-03, 1.4592657600e+08, 1.2217499000e+07, 6.1087495117e+03,
     15, 8,
     1.751973760000e+08, 4.999999873689e-05, 5.999998029438e-05, 2.182661788538e-03, 4.168577042508e+05, 5.839913000000e+06, 2.356273987011e+06, 2.766346709054e+03, 3.154248097506e-01),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 1.5000000596e-01, 2.4434998035e+00, 0.0000000000e+00, 0.0000000000e+00, 1.2217499000e+07, 6.1087495117e+03,
     15, 5,
     3.284951040000e+08, 4.999999873689e-05, 2.507182303816e-03, 4.092491231859e-03, 1.250573143514e+06, 1.094983700000e+07, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 1.5000000596e-01, 2.4434998035e+00, 9.9999997474e-06, 7.2963281250e+05, 1.2217499000e+07, 6.1087495117e+03,
     15, 5,
     3.284951040000e+08, 4.999999873689e-05, 5.999998029438e-05, 4.092491231859e-03, 1.250573143514e+06, 1.094983700000e+07, 2.209006868229e+04, 1.383173273464e+01, 1.577123956323e-03),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 1.5000000596e-01, 2.4434998035e+00, 2.0000000950e-03, 1.4592657600e+08, 1.2217499000e+07, 6.1087495117e+03,
     15, 5,
     3.284951040000e+08, 4.999999873689e-05, 5.999998029438e-05, 4.092491231859e-03, 1.250573143514e+06, 1.094983700000e+07, 4.418013995382e+06, 2.766346709054e+03, 3.154248097506e-01),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 1.5000000596e-01, 1.2217499200e+08, 0.0000000000e+00, 0.0000000000e+00, 1.2217499000e+07, 6.1087495117e+03,
     12, 5,
     3.518098560000e+08, 4.999999873689e-05, 2.507182303816e-03, 4.092491231859e-03, 1.250573143514e+06, 1.172699600000e+07, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 1.5000000596e-01, 1.2217499200e+08, 9.9999997474e-06, 7.2963281250e+05, 1.2217499000e+07, 6.1087495117e+03,
     12, 5,
     3.518098560000e+08, 4.999999873689e-05, 5.999998029438e-05, 4.092491231859e-03, 1.250573143514e+06, 1.172699600000e+07, 2.365789910515e+04, 1.383173273464e+01, 1.577123956323e-03),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 1.5000000596e-01, 1.2217499200e+08, 2.0000000950e-03, 1.4592657600e+08, 1.2217499000e+07, 6.1087495117e+03,
     12, 5,
     3.518098560000e+08, 4.999999873689e-05, 5.999998029438e-05, 4.092491231859e-03, 1.250573143514e+06, 1.172699600000e+07, 4.731580098331e+06, 2.766346709054e+03, 3.154248097506e-01),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 7.9999998212e-02, 3.6652496000e+08, 9.9999997474e-06, 2.4625107422e+03, 1.2217499000e+07, 6.1087495117e+03,
     5, 5,
     3.000003520000e+08, 4.999999873689e-05, 3.999999025837e-04, 2.182661788538e-03, 6.669722815460e+05, 1.000001200000e+07, 1.878460102878e+04, 1.565628850928e+00, 1.824035863697e-04),
    (6.0000000000e+04, 2.5500000000e+02, 9.3299994478e-04, 7.9999998212e-02, 6.1087494400e+08, 2.0000000950e-03, 9.3393007812e+03, 3.6652496000e+08, 1.2217498750e+06,
     4, 4,
     5.000004800000e+08, 4.999999873689e-05, 1.499999663793e-03, 2.182661788538e-03, 8.337154085016e+05, 1.666668400000e+07, 5.012473686504e+06, 1.962548977887e+03, 1.555904256998e+00),
    (4.0000000000e+04, 2.4500000000e+02, 1.3994999463e-03, 1.5000000596e-01, 3.5241518021e+00, 0.0000000000e+00, 0.0000000000e+00, 1.7620758000e+07, 8.8103789062e+03,
     15, 6,
     2.277647840000e+08, 4.999999873689e-05, 2.507182303816e-03, 2.837562467903e-03, 7.225793644866e+05, 7.592160000000e+06, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (4.0000000000e+04, 2.4500000000e+02, 1.3994999463e-03, 1.5000000596e-01, 3.5241518021e+00, 9.9999997474e-06, 7.2963281250e+05, 1.7620758000e+07, 8.8103789062e+03,
     15, 6,
     2.277647840000e+08, 4.999999873689e-05, 5.999997665640e-05, 2.837562467903e-03, 7.225793644866e+05, 7.592160000000e+06, 1.275362117436e+04, 1.128071917935e+01, 1.270714872118e-03),
    (4.0000000000e+04, 2.4500000000e+02, 1.3994999463e-03, 1.5000000596e-01, 3.5241518021e+00, 2.0000000950e-03, 1.4592657600e+08, 1.7620758000e+07, 8.8103789062e+03,
     15, 6,
     2.277647840000e+08, 4.999999873689e-05, 5.999997665640e-05, 2.837562467903e-03, 7.225793644866e+05, 7.592160000000e+06, 2.550724488973e+06, 2.256144060626e+03, 2.541429997412e-01),
    (4.0000000000e+04, 2.4500000000e+02, 1.3994999463e-03, 1.0000000475e-03, 2.6431137600e+08, 2.0000000950e-03, 1.4592657600e+08, 5.2862274560e+09, 8.8103792640e+09,
     9, 9,
     1.500001760000e+08, 2.233307714050e-05, 5.999997665640e-05, 2.669538368139e-09, 4.531949356897e-01, 4.577115053738e+02, 6.779988473002e+05, 6.097686424459e+05, 2.541429929635e+05),
    (2.5000000000e+04, 2.3200000000e+02, 2.2391998209e-03, 9.9999997474e-05, 5.3466415405e+00, 9.9999997474e-06, 4.6696502686e+01, 2.6733208000e+07, 1.3366604492e+04,
     15, 15,
     1.000849687500e+05, 4.999999873689e-05, 1.499999663793e-03, 1.087180336867e-07, 1.107392746386e+01, 1.661089197066e+03, 3.391455494494e+00, 2.347045259871e-01, 2.420943676232e-05),
)


_COLD_WARM_LOOP_INPUTS = (
    "p_pa", "temp_k", "qv", "qc", "nc_per_kg", "qr", "nr_per_kg",
    "nwfa_per_kg", "nifa_per_kg")
_COLD_WARM_LOOP_OUTPUTS = (
    "nc_m3", "mvd_c", "mvd_r",
    "prr_wau", "pnr_wau", "pnc_wau", "pnc_rcw", "pna_rca", "pnd_rcd")


def _cold_warm_loop_columns():
    """Unpack ``_WRF_COLD_WARM_LOOP`` into named float64 arrays."""
    table = np.asarray(
        [row[:9] + row[11:] for row in _WRF_COLD_WARM_LOOP], dtype=np.float64)
    ints = np.asarray(
        [row[9:11] for row in _WRF_COLD_WARM_LOOP], dtype=np.int32)
    columns = {name: table[:, i]
               for i, name in enumerate(_COLD_WARM_LOOP_INPUTS)}
    for j, name in enumerate(_COLD_WARM_LOOP_OUTPUTS):
        columns[name] = table[:, 9 + j]
    columns["nu_c_entry"] = ints[:, 0]
    columns["nu_c_working"] = ints[:, 1]
    return columns


def _run_cold_warm_loop_probe(columns, efficiency_table):
    import cupy as cp

    from woof.core.thompson_aerosol_cold import probe_cold_warm_loop

    def device(name):
        return cp.asarray(columns[name].astype(np.float32).copy())

    out = probe_cold_warm_loop(
        device("qc"), device("nc_per_kg"), device("qr"), device("nr_per_kg"),
        device("nwfa_per_kg"), device("nifa_per_kg"),
        device("temp_k"), device("p_pa"), device("qv"),
        efficiency_table, _WRF_COLD_WARM_LOOP_DT)
    cp.cuda.Stream.null.synchronize()
    return {k: cp.asnumpy(v) for k, v in out.items()}


def _rowwise_max_relative(got, expected):
    """Relative difference over rows whose reference is non-negligible."""
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected)))
    if scale == 0.0:
        assert np.array_equal(got, expected)
        return 0.0
    mask = np.abs(expected) > 1.0e-12 * scale
    relative = np.abs(got - expected) / np.maximum(np.abs(expected), 1e-300)
    return float(relative[mask].max()) if mask.any() else 0.0


def test_working_nu_c_is_recomputed_from_the_rediagnosed_nc(classic_tables):
    """CRITICAL: nu_c must come from :2170, never from :1832.

    module_mp_thompson.F computes nu_c TWICE.  :1832 uses the PRE-rediagnosis
    nc of :1829 and exists only to build the entry lamc and drive the
    :1834-1838 droplet-size clamp.  :1840 then REDIAGNOSES nc from that
    clamped lamc, and :2170 recomputes nu_c from the rediagnosed value.  The
    :2170 answer is what feeds lamc (:2173), mvd_c (:2174), Dc_g (:2181),
    pnr_wau (:2191) and pnc_wau (:2192).

    ``thompson_aa_cloud_dist`` RETURNS the rediagnosed nc and writes the
    ENTRY nu_c to an out-parameter.  Taking nu_c from that out-parameter --
    which is what this kernel used to do -- is finite, stable and grossly
    wrong wherever the clamp engages.

    34 of the 54 oracle rows are such states, and this gate pins BOTH stages
    against WRF, so a kernel that swaps them fails on the working column even
    though the entry column still matches.
    """
    columns = _cold_warm_loop_columns()
    got = _run_cold_warm_loop_probe(
        columns, classic_tables.cold_source_tables.rain_cloud_efficiency)

    divergent = columns["nu_c_entry"] != columns["nu_c_working"]
    assert int(divergent.sum()) >= 30, (
        "the oracle subset must span the states where the two nu_c stages "
        f"disagree; only {int(divergent.sum())} of {len(divergent)} do")

    np.testing.assert_array_equal(
        got["nu_c_working"], columns["nu_c_working"],
        err_msg="nu_c must be recomputed from the POST-rediagnosis nc "
                "(module_mp_thompson.F:2170), not reused from :1832")
    np.testing.assert_array_equal(
        got["nu_c_entry"], columns["nu_c_entry"],
        err_msg="the :1832 entry-stage nu_c no longer matches WRF")

    # The consumers.  mvd_c is deliberately included even though it cannot
    # discriminate (see below) so that a regression in it is still caught.
    #
    # TIGHTENED by up to 5,000,000x, never widened.  These were
    # (5.0e-7, 5.0e-7, 5.0e-6, 5.0e-6) while thompson_aa_cloud_dist used
    # CUDA's powf on an unpinned float32 chain; with the shared helper
    # correctly-rounded and contraction-pinned the measured maxima on these
    # 54 rows are 1.8496e-13, 2.2339e-13, 2.5504e-13 and 3.9369e-13 -- the
    # CSV round trip of the embedded literals and nothing else.  On the FULL
    # 11340-row oracle all four are BIT-EXACT.
    for name, bound in (("nc_m3", 1.0e-12), ("mvd_c", 1.0e-12),
                        ("prr_wau", 1.0e-12), ("pnr_wau", 1.0e-12)):
        worst = _rowwise_max_relative(got[name], columns[name])
        assert worst <= bound, (
            f"{name}: max relative difference {worst:.3e} against WRF v4.6.1")


def test_entry_stage_nu_c_would_disagree_with_wrf_by_whole_factors():
    """The staging is not a rounding question; it is a physics question.

    ``pnr_wau = prr_wau/(am_r*nu_c*10*D0r**3)`` (:2191) is LINEAR in nu_c, so
    on a row where the two stages differ, substituting the entry value
    rescales the rain-number source by exactly ``nu_c_working/nu_c_entry``.
    This gate reads only the committed WRF table -- no GPU -- and states the
    size of the error the previous kernel carried, so that "the tolerance
    absorbed it" can never be a defence.
    """
    columns = _cold_warm_loop_columns()
    active = columns["prr_wau"] > 0.0
    divergent = active & (columns["nu_c_entry"] != columns["nu_c_working"])
    assert int(divergent.sum()) >= 20

    entry = columns["nu_c_entry"][divergent].astype(np.float64)
    working = columns["nu_c_working"][divergent].astype(np.float64)
    ratio = working / entry
    # MEASURED on this subset: every observable divergent row is wrong by at
    # least 11%, and the worst by 89%.  Over the full 11340-row sweep the
    # extremes are nu_c_entry = 15 against nu_c_working = 4 and
    # nu_c_entry = 3 against nu_c_working = 15.
    assert float(np.abs(ratio - 1.0).min()) >= 0.09, (
        "the divergent rows must carry a physically large error, not a "
        "rounding one")
    assert float(np.abs(ratio - 1.0).max()) >= 0.7

    # mvd_c CANNOT discriminate, and that is structural rather than lucky:
    # if the :1834-1838 clamp engaged then lamc = cce(2,nu)/D0c or
    # cce(2,nu)/(2*D0r), so mvd_c = (3.672+nu)/(4+nu) times D0c or 2*D0r,
    # which lies below D0c in the first case and above D0r in the second for
    # every nu in [2,15] -- the :2175 clamp swallows it either way.  This
    # assertion records why a state-level fixture comparison could never have
    # found the defect.
    clamped = np.isclose(columns["mvd_c"][divergent], 1.0e-6, rtol=1e-6) | \
        np.isclose(columns["mvd_c"][divergent], 50.0e-6, rtol=1e-6)
    assert clamped.all(), (
        "a divergent state with an unclamped mvd_c would be a NEW and "
        "stronger observable; update this test rather than deleting it")


def test_sub_freezing_rain_scavenging_reaches_the_accumulators():
    """CRITICAL: pna_rca and pnd_rcd must run below freezing.

    WRF's rain aerosol-scavenging block at :2211-2222 sits inside the k-loop
    opened at :2157, which precedes the ``.not. iiwarm`` gate at :2239 AND
    the ``temp(k).lt.T_0`` guard at :2554.  It therefore runs at EVERY level.
    Its gate is ``L_qr .and. mvd_r > D0r`` alone -- no cloud water required --
    so rain scavenges CCN and IN out of clear supercooled air.

    thompson_aerosol_warm.cu computes these two rates only for cells whose
    ENTRY temperature was >= 273.15 K, and this kernel owns exactly the
    complement, so before wave 3 nothing supplied them anywhere below
    freezing: ``nwfaten`` and ``nifaten`` came back identically zero on a
    raining supercooled column.

    The column here has NO cloud, NO ice, NO snow and NO graupel, and sits at
    272 K where the ladder's vapour is far below ice saturation, so both
    iceDeMott deposition nucleation (gated at :2620-2621 on ssati >= 0.25 or
    water supersaturation below 253.15 K) and Koop haze freezing (gated at
    :2635-2636 on temp < 238 K and ssati >= 0.4) are off.  pna_rca and
    pnd_rcd are therefore the ONLY aerosol sinks in the entire scheme on this
    column and the accumulators must equal ``-rate*orho``.  Synthetic zero
    coefficient tables are enough: with qc = 0 the t_Efrw lookup is never
    reached, and Eff_aero is a device helper, not a table.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import launch_aa_cold_network

    columns = _cold_warm_loop_columns()
    # Rows with rain, aerosol scavenging active, no cloud, and no nucleation.
    rows = np.nonzero((columns["qc"] == 0.0)
                      & (columns["pna_rca"] > 0.0)
                      & (columns["temp_k"] == 272.0))[0]
    assert rows.size >= 4, "the oracle subset must carry clear-air rain rows"
    dt = _WRF_COLD_WARM_LOOP_DT

    def device(name):
        return cp.asarray(columns[name][rows].astype(np.float32).copy())

    levels = int(rows.size)
    zeros = cp.zeros(levels, dtype=cp.float32)
    qi = zeros.copy(); ni = zeros.copy()
    qs = zeros.copy(); qg = zeros.copy()
    ncten = zeros.copy(); nwfaten = zeros.copy(); nifaten = zeros.copy()
    shadow = zeros.copy(); boost = zeros.copy()
    tables = _synthetic_tables(cp)

    launch_aa_cold_network(
        qi, ni, qs, qg, device("qr"), device("nr_per_kg"), device("qc"),
        device("temp_k"), device("p_pa"), device("qv"),
        device("nc_per_kg"), device("nwfa_per_kg"), device("nifa_per_kg"),
        ncten, nwfaten, nifaten, shadow, boost,
        tables["ice_deposition_partition"], tables["ice_to_snow_mass"],
        tables["ice_to_snow_number"], tables["rain_snow"],
        tables["rain_graupel"], tables["rain_freezing"],
        tables["rain_cloud_efficiency"], tables["cloud_freezing"], dt,
        snow_cloud_efficiency=tables["snow_cloud_efficiency"])
    cp.cuda.Stream.null.synchronize()

    rho = (np.float32(0.622) * columns["p_pa"][rows].astype(np.float32)
           / (np.float32(287.04) * columns["temp_k"][rows].astype(np.float32)
              * (np.maximum(np.float32(1.0e-10),
                            columns["qv"][rows].astype(np.float32))
                 + np.float32(0.622))))
    orho = (np.float32(1.0) / rho).astype(np.float64)

    got_nwfa = cp.asnumpy(nwfaten).astype(np.float64)
    got_nifa = cp.asnumpy(nifaten).astype(np.float64)
    assert np.all(got_nwfa < 0.0), (
        "rain wet-scavenging of water-friendly aerosol is missing below "
        "freezing (module_mp_thompson.F:2212-2216)")
    assert np.all(got_nifa < 0.0), (
        "rain wet-scavenging of ice-friendly aerosol is missing below "
        "freezing (module_mp_thompson.F:2218-2221)")

    for name, got, rate in (("nwfaten", got_nwfa, "pna_rca"),
                            ("nifaten", got_nifa, "pnd_rcd")):
        expected = -columns[rate][rows] * orho
        worst = _rowwise_max_relative(got, expected)
        # TIGHTENED from 5.0e-7: measured 3.001e-08 (nwfaten) with the shared
        # droplet diagnosis repaired.  The residual is the host float32
        # reconstruction of orho, not the kernel.
        assert worst <= 5.0e-8, (
            f"{name}: max relative difference {worst:.3e} against WRF v4.6.1")

    # Bigg freezing consumes no aerosol and there is no cloud water, so the
    # droplet accumulator must not have moved.
    assert np.all(cp.asnumpy(ncten) == 0.0)


def test_sub_freezing_autoconversion_debits_droplet_number():
    """CRITICAL: pnc_wau must run below freezing.

    :2192-2193 sits in the same unguarded k-loop.  This kernel already
    computed its MASS partner prr_wau (:2189-2190) at sub-freezing levels --
    it has to, because prr_wau competes for cloud water with riming and Bigg
    freezing in the :2878-2890 conservation pass -- but it discarded the
    number.  Cloud mass left the droplet population without taking any
    droplets with it.

    The column here has cloud water above the 0.01e-3 autoconversion
    threshold and nothing else at all, and sits at 272 K where neither
    iceDeMott nor Koop can fire, so ``ncten`` must be exactly
    ``-pnc_wau*orho``.  The Bigg cloud-freezing lookup IS reached (rc is well
    above r_c(1)) but the synthetic tpi_qcfz/tni_qcfz are identically zero,
    so pni_wfz is zero by construction; 272 K is also far above HGFR =
    235.16 K, so the :2613-2616 all-at-once branch cannot fire either.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import launch_aa_cold_network

    columns = _cold_warm_loop_columns()
    rows = np.nonzero((columns["qr"] == 0.0)
                      & (columns["pnc_wau"] > 0.0)
                      & (columns["temp_k"] == 272.0))[0]
    assert rows.size >= 3, (
        "the oracle subset must carry rain-free autoconverting rows")
    dt = _WRF_COLD_WARM_LOOP_DT

    def device(name):
        return cp.asarray(columns[name][rows].astype(np.float32).copy())

    levels = int(rows.size)
    zeros = cp.zeros(levels, dtype=cp.float32)
    ncten = zeros.copy(); nwfaten = zeros.copy(); nifaten = zeros.copy()
    tables = _synthetic_tables(cp)

    launch_aa_cold_network(
        zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(),
        device("qr"), device("nr_per_kg"), device("qc"),
        device("temp_k"), device("p_pa"), device("qv"),
        device("nc_per_kg"), device("nwfa_per_kg"), device("nifa_per_kg"),
        ncten, nwfaten, nifaten, zeros.copy(), zeros.copy(),
        tables["ice_deposition_partition"], tables["ice_to_snow_mass"],
        tables["ice_to_snow_number"], tables["rain_snow"],
        tables["rain_graupel"], tables["rain_freezing"],
        tables["rain_cloud_efficiency"], tables["cloud_freezing"], dt,
        snow_cloud_efficiency=tables["snow_cloud_efficiency"])
    cp.cuda.Stream.null.synchronize()

    rho = (np.float32(0.622) * columns["p_pa"][rows].astype(np.float32)
           / (np.float32(287.04) * columns["temp_k"][rows].astype(np.float32)
              * (np.maximum(np.float32(1.0e-10),
                            columns["qv"][rows].astype(np.float32))
                 + np.float32(0.622))))
    orho = (np.float32(1.0) / rho).astype(np.float64)

    got = cp.asnumpy(ncten).astype(np.float64)
    assert np.all(got < 0.0), (
        "autoconversion is removing cloud mass without removing droplet "
        "number (module_mp_thompson.F:2192-2193)")
    expected = -columns["pnc_wau"][rows] * orho
    worst = _rowwise_max_relative(got, expected)
    # TIGHTENED from 5.0e-6 by 100x: measured 4.241e-08.  pnc_wau itself is
    # bit-exact against WRF on all 11340 oracle rows now; what is left is the
    # host float32 reconstruction of orho on the expectation side.
    assert worst <= 5.0e-8, (
        f"ncten: max relative difference {worst:.3e} against WRF v4.6.1")


def test_cold_and_warm_temperature_masks_are_exact_complements():
    """No cell may be claimed by both networks, and none by neither.

    Ten of WRF's rates run at every level.  woof evaluates each of them in
    two kernels and relies on the two temperature masks partitioning the
    domain:

    * this kernel returns at once for ``temperature >= 273.15f``, and the
      adapter launches it BEFORE anything has moved the temperature field,
      so its gate sees the entry column;
    * thompson_aerosol_warm.cu returns unless the held entry mask
      ``cp.greater_equal(temperature, 273.15)`` -- exactly what
      microphysics.py:503-504 seeds for mp=8 -- is set.

    A gap or an overlap at 273.15 K itself is the obvious way this goes
    wrong, so the kernel is run at 273.15 and at both bracketing float32
    values.

    The observable is the SOURCE-TERM set, not ``snow_velocity_boost``: WRF
    resets vts_boost unconditionally at :2243 and this kernel deliberately
    writes it for every cell, warm ones included, BEFORE its own temperature
    return, because the later column sedimentation kernel consumes the whole
    field.  A cell is "claimed" here exactly when the kernel added something
    to one of the three accumulators, which is the quantity that would be
    double-counted or dropped.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import launch_aa_cold_network

    freezing = np.float32(273.15)
    temperatures = np.asarray([
        np.nextafter(freezing, np.float32(-np.inf), dtype=np.float32),
        freezing,
        np.nextafter(freezing, np.float32(np.inf), dtype=np.float32),
        np.float32(250.0),
        np.float32(300.0),
    ], dtype=np.float32)
    levels = temperatures.size

    def full(value):
        return cp.asarray(np.full(levels, value, dtype=np.float32))

    ncten = full(0.0); nwfaten = full(0.0); nifaten = full(0.0)
    boost = full(7.0)
    tables = _synthetic_tables(cp)
    launch_aa_cold_network(
        full(0.0), full(0.0), full(0.0), full(0.0), full(1.0e-3),
        full(1.0e4), full(1.0e-3),
        cp.asarray(temperatures.copy()), full(80000.0), full(3.0e-4),
        full(1.0e8), full(1.0e9), full(1.0e6),
        ncten, nwfaten, nifaten, full(0.0), boost,
        tables["ice_deposition_partition"], tables["ice_to_snow_mass"],
        tables["ice_to_snow_number"], tables["rain_snow"],
        tables["rain_graupel"], tables["rain_freezing"],
        tables["rain_cloud_efficiency"], tables["cloud_freezing"], 30.0,
        snow_cloud_efficiency=tables["snow_cloud_efficiency"])
    cp.cuda.Stream.null.synchronize()

    # vts_boost is reset for EVERY cell by design, so it cannot serve as the
    # claim marker; the accumulators can.
    assert np.all(cp.asnumpy(boost) == np.float32(1.0)), (
        "WRF resets vts_boost unconditionally at :2243")

    cold_claimed = ((cp.asnumpy(ncten) != 0.0)
                    | (cp.asnumpy(nwfaten) != 0.0)
                    | (cp.asnumpy(nifaten) != 0.0))
    warm_claimed = cp.asnumpy(
        cp.greater_equal(cp.asarray(temperatures.copy()),
                         cp.float32(273.15))).astype(bool)

    np.testing.assert_array_equal(
        cold_claimed, ~warm_claimed,
        err_msg="the cold and warm entry-temperature masks are not exact "
                "complements; every :2157-2234 and :2402-2478 rate is "
                "evaluated in BOTH kernels and would be double-counted or "
                "dropped")
    # Spelled out at the boundary so a future reader does not have to derive
    # it: 273.15 K itself belongs to the WARM network.
    assert not cold_claimed[1]
    assert cold_claimed[0] and not cold_claimed[2]
    # And the sinks must actually have fired on the cell the cold side owns.
    assert cp.asnumpy(ncten)[0] < 0.0
    assert cp.asnumpy(nwfaten)[0] < 0.0
    assert cp.asnumpy(nifaten)[0] < 0.0


def test_probe_and_production_kernel_agree_on_the_four_new_rates(
        classic_tables):
    """The readback probe may not drift away from the production kernel.

    The probe exists because these rates have no end-to-end observable, which
    also means nothing else would notice if it stopped matching.  Here the
    two are run on ONE column that carries cloud, rain and aerosol but no ice,
    snow or graupel, at 272 K where no nucleation path can fire, so ``ncten``
    is exactly ``-(pnc_wau + pnc_rcw)*orho`` and the aerosol accumulators are
    exactly the two rain-scavenging terms.

    The coefficient tables are the SYNTHETIC zero set except for
    ``rain_cloud_efficiency``, which is the canonical ``t_Efrw`` taken from
    the verified classic owner: pnc_rcw genuinely reads it, and zeroing the
    freezing tables is what leaves pni_wfz out of the sum.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import (
        launch_aa_cold_network, probe_cold_warm_loop)

    columns = _cold_warm_loop_columns()
    rows = np.nonzero((columns["pnc_rcw"] > 0.0)
                      & (columns["pna_rca"] > 0.0)
                      & (columns["pnc_wau"] > 0.0)
                      & (columns["temp_k"] == 272.0))[0]
    assert rows.size >= 3
    dt = _WRF_COLD_WARM_LOOP_DT

    def device(name):
        return cp.asarray(columns[name][rows].astype(np.float32).copy())

    efrw = classic_tables.cold_source_tables.rain_cloud_efficiency
    tables = _synthetic_tables(cp)
    levels = int(rows.size)
    zeros = cp.zeros(levels, dtype=cp.float32)
    ncten = zeros.copy(); nwfaten = zeros.copy(); nifaten = zeros.copy()
    launch_aa_cold_network(
        zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(),
        device("qr"), device("nr_per_kg"), device("qc"),
        device("temp_k"), device("p_pa"), device("qv"),
        device("nc_per_kg"), device("nwfa_per_kg"), device("nifa_per_kg"),
        ncten, nwfaten, nifaten, zeros.copy(), zeros.copy(),
        tables["ice_deposition_partition"], tables["ice_to_snow_mass"],
        tables["ice_to_snow_number"], tables["rain_snow"],
        tables["rain_graupel"], tables["rain_freezing"],
        efrw, tables["cloud_freezing"], dt,
        snow_cloud_efficiency=tables["snow_cloud_efficiency"])
    cp.cuda.Stream.null.synchronize()

    probe = probe_cold_warm_loop(
        device("qc"), device("nc_per_kg"), device("qr"), device("nr_per_kg"),
        device("nwfa_per_kg"), device("nifa_per_kg"),
        device("temp_k"), device("p_pa"), device("qv"), efrw, dt)
    cp.cuda.Stream.null.synchronize()

    rho = (np.float32(0.622) * columns["p_pa"][rows].astype(np.float32)
           / (np.float32(287.04) * columns["temp_k"][rows].astype(np.float32)
              * (np.maximum(np.float32(1.0e-10),
                            columns["qv"][rows].astype(np.float32))
                 + np.float32(0.622))))
    orho = (np.float32(1.0) / rho).astype(np.float64)

    pnc_wau = cp.asnumpy(probe["pnc_wau"])
    pnc_rcw = cp.asnumpy(probe["pnc_rcw"])
    pna_rca = cp.asnumpy(probe["pna_rca"])
    pnd_rcd = cp.asnumpy(probe["pnd_rcd"])
    assert np.all(pnc_wau > 0.0) and np.all(pnc_rcw > 0.0)
    assert np.all(pna_rca > 0.0) and np.all(pnd_rcd > 0.0)

    np.testing.assert_allclose(
        cp.asnumpy(ncten).astype(np.float64),
        np.float32(-(pnc_wau + pnc_rcw) * orho).astype(np.float64),
        rtol=1.0e-6,
        err_msg="ncten does not equal -(pnc_wau + pnc_rcw)*orho")
    np.testing.assert_allclose(
        cp.asnumpy(nwfaten).astype(np.float64),
        np.float32(-pna_rca * orho).astype(np.float64), rtol=1.0e-6)
    np.testing.assert_allclose(
        cp.asnumpy(nifaten).astype(np.float64),
        np.float32(-pnd_rcd * orho).astype(np.float64), rtol=1.0e-6)


def test_shared_device_helpers_carry_no_local_copy():
    """MINOR 3: one definition per helper, in the shared header.

    thompson_aerosol_common.cuh owns thompson_aa_entry_rain_distribution,
    thompson_aa_bound_rain_number and thompson_aa_bound_ice_number.  For the
    two ``void`` helpers a surviving local copy is a hard nvrtc redefinition
    error, but the shared rain distribution takes SEVEN parameters and the
    old local copy took six -- so it would quietly OVERLOAD rather than
    collide, and every call in this file would keep resolving to the local,
    un-pinned, N0_r-less version.  Only a source gate catches that.
    """
    source = _KERNEL_SOURCE.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("//"))
    for helper in ("thompson_aa_entry_rain_distribution",
                   "thompson_aa_bound_rain_number",
                   "thompson_aa_bound_ice_number"):
        assert f"void {helper}(" not in code and f"bool {helper}(" not in code, (
            f"{helper} is defined locally in thompson_aerosol_cold.cu; it "
            "belongs to thompson_aerosol_common.cuh and nowhere else")
        assert f"{helper}(" in code, f"{helper} is no longer called at all"
    # The seven-output form is the one that publishes N0_r, and the four
    # rain-collection rates must consume it rather than a re-derived lambda.
    assert "rain_intercept_n0" in code


def test_no_frozen_droplet_shape_constant_survives_as_an_initialiser():
    """MINOR 4: ``nu_c`` may not be seeded with mp=8's frozen 12.

    ``12`` is exactly ``MIN(15, NINT(1000e6/100e6) + 2)``, i.e. the shape
    parameter mp=8 froze for Nt_c.  A declaration ``int nu_c = 12;`` in a
    kernel where nc is prognostic is one dropped guard away from becoming the
    answer, and it would look right.  The sentinel is 0, whose gamma-table
    column is the deliberate zero pad, so a dropped guard divides by zero
    instead of silently returning mp=8 physics.
    """
    source = _KERNEL_SOURCE.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("//"))
    assert "nu_c = 12" not in code
    assert "int nu_c = 0;" in code


# ---------------------------------------------------------------------------
# Launcher contract.
# ---------------------------------------------------------------------------

def test_launcher_rejects_wrong_shapes_dtypes_and_tables():
    import cupy as cp

    from woof.core.thompson_aerosol_cold import launch_aa_cold_network

    tables = _synthetic_tables(cp)
    good = [cp.zeros((4, 1, 1), dtype=cp.float32) for _ in range(18)]

    def call(fields=None, **overrides):
        args = list(fields if fields is not None else good)
        table_args = {
            "ice_deposition_partition": tables["ice_deposition_partition"],
            "ice_to_snow_mass": tables["ice_to_snow_mass"],
            "ice_to_snow_number": tables["ice_to_snow_number"],
            "rain_snow_tables": tables["rain_snow"],
            "rain_graupel_tables": tables["rain_graupel"],
            "rain_freezing_tables": tables["rain_freezing"],
            "rain_cloud_efficiency": tables["rain_cloud_efficiency"],
            "cloud_freezing_tables": tables["cloud_freezing"],
            "dt": 10.0,
        }
        table_args.update(overrides)
        return launch_aa_cold_network(
            *args,
            table_args["ice_deposition_partition"],
            table_args["ice_to_snow_mass"],
            table_args["ice_to_snow_number"],
            table_args["rain_snow_tables"],
            table_args["rain_graupel_tables"],
            table_args["rain_freezing_tables"],
            table_args["rain_cloud_efficiency"],
            table_args["cloud_freezing_tables"],
            table_args["dt"],
            snow_cloud_efficiency=tables["snow_cloud_efficiency"])

    call()  # the baseline must succeed

    mismatched = list(good)
    mismatched[5] = cp.zeros((5, 1, 1), dtype=cp.float32)
    with pytest.raises(ValueError):
        call(mismatched)

    wrong_dtype = list(good)
    wrong_dtype[2] = cp.zeros((4, 1, 1), dtype=cp.float64)
    with pytest.raises(TypeError):
        call(wrong_dtype)

    with pytest.raises(ValueError):
        call(dt=0.0)
    with pytest.raises(ValueError):
        call(rain_snow_tables=tables["rain_snow"][:5])
    with pytest.raises(ValueError):
        call(cloud_freezing_tables=(
            cp.zeros((37, 100, 45, 54), dtype=cp.float64, order="F"),
            tables["cloud_freezing"][1]))
    with pytest.raises(TypeError):
        call(rain_freezing_tables=tuple(
            cp.zeros((37, 37, 45, 55), dtype=cp.float32, order="F")
            for _ in range(4)))


def test_launcher_rejects_an_unverified_table_owner():
    from woof.core.thompson_aerosol_cold import (
        launch_aa_cold_network_from_owner)

    with pytest.raises(TypeError):
        launch_aa_cold_network_from_owner(
            *([None] * 18), object(), 10.0)


# ===========================================================================
# WAVE 4.  Three gates that close audit gaps rather than add coverage.
#
# PROVENANCE OF EVERY REFERENCE NUMBER BELOW.  All three tables come from
# ONE scratch build under mp28-oracle-work/wp06/ that compiles a
# COPY of wrf461-pristine/phys/module_mp_thompson.F carrying only
# added ``write`` statements and three added PUBLIC probe subroutines whose
# bodies are verbatim transcriptions of WRF statements already in the file.
# The pristine reference is never modified.  The build links the same
# stub_wrf.F90 / module_mp_radar.F / run_column_aero.F90 that
# tools/thompson_wrf461_oracle/build_aero.sh uses, with the same
# ``gfortran -O2 -ffree-form`` flags and the same four table assets, and it
# REPRODUCES ALL 38 COMMITTED aerosol fixture CSVs BYTE FOR BYTE (19 column
# + 19 surface, verified by diff).  That byte-for-byte reproduction is what
# makes numbers taken from its instrumentation admissible as WRF ground
# truth: the instrumented module computes the same trajectory as the module
# the committed fixtures came from.
# ===========================================================================

#: module_mp_thompson.F:1826-1842 (BOTH nu_c stages) and :2168-2193, over a
#: 10 x 12 (nc1d, qc1d) grid at p = 85000 Pa, T = 265 K, qv = 2.0e-3,
#: dtsave = 30 s.  Columns:
#:
#:     qc (kg/kg), nc1d (kg^-1), nu_c_entry(:1832), nu_c_working(:2170),
#:     prr_wau(:2189), pnr_wau(:2191), pnc_wau(:2192)
#:
#: WHY THIS GRID.  The staging of nu_c is invisible in ``mvd_c`` (the
#: :2175 clamp swallows it) and invisible in every committed fixture,
#: because the only unclamped consumers -- ``Dc_g``, ``prr_wau``,
#: ``pnr_wau``, ``pnc_wau`` -- are gated on ``rc > 0.01e-3`` and the lowest
#: rc at which the two stages disagree AND that gate is open is about
#: 3.4e-2 kg m^-3.  MEASURED on device over a 60 x 80 (nc, qc) sweep:
#: 1259 of 4800 states are divergent and every one of them has
#: prr_wau > 0; the smallest such rc is 3.4365e-2 kg m^-3.  The grid below
#: reaches that regime deliberately.  36 of its 120 rows are divergent with
#: prr_wau > 0, spanning nu_c_working/nu_c_entry from 0.200 to 0.867.
_WRF_NU_C_STAGE = (
    (9.9999997474e-06, 1.0000000000e+06, 15, 15, 9.359213057336e-12, 9.533215695788e-04, 7.038984934770e-01),
    (9.9999997474e-05, 1.0000000000e+06, 15, 15, 9.343838058840e-08, 9.517554851646e+00, 1.427633294343e+03),
    (1.0000000475e-03, 1.0000000000e+06, 15, 15, 3.712916804943e-05, 3.781945826556e+03, 9.934250781250e+04),
    (4.9999998882e-03, 1.0000000000e+06, 15, 15, 1.856458256952e-04, 1.890972765053e+04, 4.967125000000e+05),
    (9.9999997765e-03, 1.0000000000e+06, 15, 15, 3.712916513905e-04, 3.781945530107e+04, 9.934250000000e+05),
    (1.9999999553e-02, 1.0000000000e+06, 15, 15, 7.425833027810e-04, 7.563891060214e+04, 1.986850000000e+06),
    (3.0999999493e-02, 1.0000000000e+06, 15, 13, 1.151004107669e-03, 1.352772867033e+05, 3.079617500000e+06),
    (5.0000000745e-02, 1.0000000000e+06, 15,  9, 1.856458256952e-03, 3.151621607734e+05, 4.967125500000e+06),
    (1.0000000149e-01, 1.0000000000e+06, 15,  5, 3.712916513905e-03, 1.134583659032e+06, 9.934251000000e+06),
    (3.0000001192e-01, 1.0000000000e+06, 15,  3, 1.113874930888e-02, 5.672918561499e+06, 2.980275400000e+07),
    (1.0000000000e+00, 1.0000000000e+06, 15,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 1.0000000000e+06, 15,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 5.0000000000e+06, 15, 15, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 5.0000000000e+06, 15, 15, 4.357514349351e-09, 4.438527463300e-01, 1.638624939463e+02),
    (1.0000000475e-03, 5.0000000000e+06, 15, 15, 3.220329745091e-05, 3.280200790754e+03, 1.856460468750e+05),
    (4.9999998882e-03, 5.0000000000e+06, 15, 15, 1.856458256952e-04, 1.890972765053e+04, 4.967125000000e+05),
    (9.9999997765e-03, 5.0000000000e+06, 15, 15, 3.712916513905e-04, 3.781945530107e+04, 9.934250000000e+05),
    (1.9999999553e-02, 5.0000000000e+06, 15, 15, 7.425833027810e-04, 7.563891060214e+04, 1.986850000000e+06),
    (3.0999999493e-02, 5.0000000000e+06, 15, 13, 1.151004107669e-03, 1.352772867033e+05, 3.079617500000e+06),
    (5.0000000745e-02, 5.0000000000e+06, 15,  9, 1.856458256952e-03, 3.151621607734e+05, 4.967125500000e+06),
    (1.0000000149e-01, 5.0000000000e+06, 15,  5, 3.712916513905e-03, 1.134583659032e+06, 9.934251000000e+06),
    (3.0000001192e-01, 5.0000000000e+06, 15,  3, 1.113874930888e-02, 5.672918561499e+06, 2.980275400000e+07),
    (1.0000000000e+00, 5.0000000000e+06, 15,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 5.0000000000e+06, 15,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 2.0000000000e+07, 15, 15, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 2.0000000000e+07, 15, 15, 1.264592319750e-10, 1.288103099861e-02, 1.902178115076e+01),
    (1.0000000475e-03, 2.0000000000e+07, 15, 15, 2.613991227918e-06, 2.662589477339e+02, 3.993884402316e+04),
    (4.9999998882e-03, 2.0000000000e+07, 15, 15, 1.856458256952e-04, 1.890972765053e+04, 7.425841250000e+05),
    (9.9999997765e-03, 2.0000000000e+07, 15, 15, 3.712916513905e-04, 3.781945530107e+04, 9.934250000000e+05),
    (1.9999999553e-02, 2.0000000000e+07, 15, 15, 7.425833027810e-04, 7.563891060214e+04, 1.986850000000e+06),
    (3.0999999493e-02, 2.0000000000e+07, 15, 13, 1.151004107669e-03, 1.352772867033e+05, 3.079617500000e+06),
    (5.0000000745e-02, 2.0000000000e+07, 15,  9, 1.856458256952e-03, 3.151621607734e+05, 4.967125500000e+06),
    (1.0000000149e-01, 2.0000000000e+07, 15,  5, 3.712916513905e-03, 1.134583659032e+06, 9.934251000000e+06),
    (3.0000001192e-01, 2.0000000000e+07, 15,  3, 1.113874930888e-02, 5.672918561499e+06, 2.980275400000e+07),
    (1.0000000000e+00, 2.0000000000e+07, 15,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 2.0000000000e+07, 15,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 5.0000000000e+07, 15, 15, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 5.0000000000e+07, 15, 15, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (1.0000000475e-03, 5.0000000000e+07, 15, 15, 4.357517298104e-07, 4.438530466874e+01, 1.638625513862e+04),
    (4.9999998882e-03, 5.0000000000e+07, 15, 15, 1.856458256952e-04, 1.890972765053e+04, 1.856460250000e+06),
    (9.9999997765e-03, 5.0000000000e+07, 15, 15, 3.712916513905e-04, 3.781945530107e+04, 1.856460250000e+06),
    (1.9999999553e-02, 5.0000000000e+07, 15, 15, 7.425833027810e-04, 7.563891060214e+04, 1.986850000000e+06),
    (3.0999999493e-02, 5.0000000000e+07, 15, 13, 1.151004107669e-03, 1.352772867033e+05, 3.079617500000e+06),
    (5.0000000745e-02, 5.0000000000e+07, 15,  9, 1.856458256952e-03, 3.151621607734e+05, 4.967125500000e+06),
    (1.0000000149e-01, 5.0000000000e+07, 15,  5, 3.712916513905e-03, 1.134583659032e+06, 9.934251000000e+06),
    (3.0000001192e-01, 5.0000000000e+07, 15,  3, 1.113874930888e-02, 5.672918561499e+06, 2.980275400000e+07),
    (1.0000000000e+00, 5.0000000000e+07, 15,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 5.0000000000e+07, 15,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 1.0000000000e+08, 11, 11, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 1.0000000000e+08, 11, 11, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (1.0000000475e-03, 1.0000000000e+08, 11, 11, 1.286747561835e-07, 1.787277560117e+01, 8.897726601632e+03),
    (4.9999998882e-03, 1.0000000000e+08, 11, 11, 8.347804396180e-05, 1.159500426973e+04, 1.275450560637e+06),
    (9.9999997765e-03, 1.0000000000e+08, 11, 11, 3.712916513905e-04, 5.157198322900e+04, 3.712920250000e+06),
    (1.9999999553e-02, 1.0000000000e+08, 11, 11, 7.425833027810e-04, 1.031439664580e+05, 3.712920250000e+06),
    (3.0999999493e-02, 1.0000000000e+08, 11, 11, 1.151004107669e-03, 1.598731463929e+05, 3.712920250000e+06),
    (5.0000000745e-02, 1.0000000000e+08, 11,  8, 1.856458256952e-03, 3.545574175049e+05, 5.479080000000e+06),
    (1.0000000149e-01, 1.0000000000e+08, 11,  5, 3.712916513905e-03, 1.134583659032e+06, 1.095816000000e+07),
    (3.0000001192e-01, 1.0000000000e+08, 11,  3, 1.113874930888e-02, 5.672918561499e+06, 3.287447800000e+07),
    (1.0000000000e+00, 1.0000000000e+08, 11,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 1.0000000000e+08, 11,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 2.0000000000e+08,  6,  6, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 2.0000000000e+08,  6,  6, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (1.0000000475e-03, 2.0000000000e+08,  6,  6, 4.357067595606e-08, 1.109518176206e+01, 4.854080107020e+03),
    (4.9999998882e-03, 2.0000000000e+08,  6,  6, 3.660342190415e-05, 9.320985048509e+03, 8.155758363458e+05),
    (9.9999997765e-03, 2.0000000000e+08,  6,  6, 3.712916513905e-04, 9.454864466797e+04, 5.672918559791e+06),
    (1.9999999553e-02, 2.0000000000e+08,  6,  6, 7.425833027810e-04, 1.890972893359e+05, 7.425841000000e+06),
    (3.0999999493e-02, 2.0000000000e+08,  6,  6, 1.151004107669e-03, 2.931007955062e+05, 7.425840500000e+06),
    (5.0000000745e-02, 2.0000000000e+08,  6,  6, 1.856458256952e-03, 4.727432233399e+05, 7.425841000000e+06),
    (1.0000000149e-01, 2.0000000000e+08,  6,  4, 3.712916513905e-03, 1.418229670020e+06, 1.406973900000e+07),
    (3.0000001192e-01, 2.0000000000e+08,  6,  3, 1.113874930888e-02, 5.672918561499e+06, 4.220922000000e+07),
    (1.0000000000e+00, 2.0000000000e+08,  6,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 2.0000000000e+08,  6,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 5.0000000000e+08,  4,  4, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 5.0000000000e+08,  4,  4, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (1.0000000475e-03, 5.0000000000e+08,  4,  4, 2.553233846569e-09, 9.752635110814e-01, 5.936838779862e+02),
    (4.9999998882e-03, 5.0000000000e+08,  4,  4, 8.229865670728e-06, 3.143577193506e+03, 3.827255667623e+05),
    (9.9999997765e-03, 5.0000000000e+08,  4,  4, 1.313204411417e-04, 5.016071468609e+04, 3.053493222769e+06),
    (1.9999999553e-02, 5.0000000000e+08,  4,  4, 7.425833027810e-04, 2.836459340039e+05, 1.134583711958e+07),
    (3.0999999493e-02, 5.0000000000e+08,  4,  4, 1.151004107669e-03, 4.396511932593e+05, 1.758604735748e+07),
    (5.0000000745e-02, 5.0000000000e+08,  4,  4, 1.856458256952e-03, 7.091148350098e+05, 1.856460200000e+07),
    (1.0000000149e-01, 5.0000000000e+08,  4,  4, 3.712916513905e-03, 1.418229670020e+06, 1.856460200000e+07),
    (3.0000001192e-01, 5.0000000000e+08,  4,  3, 1.113874930888e-02, 5.672918561499e+06, 5.186666400000e+07),
    (1.0000000000e+00, 5.0000000000e+08,  4,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 5.0000000000e+08,  4,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 1.0000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 1.0000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (1.0000000475e-03, 1.0000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (4.9999998882e-03, 1.0000000000e+09,  3,  3, 2.257763753732e-06, 1.149869662280e+03, 1.824411393837e+05),
    (9.9999997765e-03, 1.0000000000e+09,  3,  3, 4.169817111688e-05, 2.123670462004e+04, 1.684733216715e+06),
    (1.9999999553e-02, 1.0000000000e+09,  3,  3, 6.517867441289e-04, 3.319522700774e+05, 1.316708930497e+07),
    (3.0999999493e-02, 1.0000000000e+09,  3,  3, 1.151004107669e-03, 5.862015910124e+05, 1.758604735748e+07),
    (5.0000000745e-02, 1.0000000000e+09,  3,  3, 1.856458256952e-03, 9.454864466797e+05, 2.836459279896e+07),
    (1.0000000149e-01, 1.0000000000e+09,  3,  3, 3.712916513905e-03, 1.890972893359e+06, 3.712920800000e+07),
    (3.0000001192e-01, 1.0000000000e+09,  3,  3, 1.113874930888e-02, 5.672918561499e+06, 6.080659200000e+07),
    (1.0000000000e+00, 1.0000000000e+09,  3,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 1.0000000000e+09,  3,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 1.5000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 1.5000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (1.0000000475e-03, 1.5000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (4.9999998882e-03, 1.5000000000e+09,  3,  3, 7.758142146486e-07, 3.951189434746e+02, 9.403578716914e+04),
    (9.9999997765e-03, 1.5000000000e+09,  3,  3, 1.757137579261e-05, 8.949028398141e+03, 1.064905602190e+06),
    (1.9999999553e-02, 1.5000000000e+09,  3,  3, 2.975611714646e-04, 1.515466634514e+05, 9.016782812762e+06),
    (3.0999999493e-02, 1.5000000000e+09,  3,  3, 1.151004107669e-03, 5.862015910124e+05, 2.250197118792e+07),
    (5.0000000745e-02, 1.5000000000e+09,  3,  3, 1.856458256952e-03, 9.454864466797e+05, 2.836459279896e+07),
    (1.0000000149e-01, 1.5000000000e+09,  3,  3, 3.712916513905e-03, 1.890972893359e+06, 5.569380800000e+07),
    (3.0000001192e-01, 1.5000000000e+09,  3,  3, 1.113874930888e-02, 5.672918561499e+06, 6.080659200000e+07),
    (1.0000000000e+00, 1.5000000000e+09,  3,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 1.5000000000e+09,  3,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
    (9.9999997474e-06, 3.0000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (9.9999997474e-05, 3.0000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (1.0000000475e-03, 3.0000000000e+09,  3,  3, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00),
    (4.9999998882e-03, 3.0000000000e+09,  3,  3, 4.451680979400e-07, 2.267222554131e+02, 6.455710602101e+04),
    (9.9999997765e-03, 3.0000000000e+09,  3,  3, 1.168115795736e-05, 5.949165023691e+03, 8.469853929231e+05),
    (1.9999999553e-02, 3.0000000000e+09,  3,  3, 2.079823461827e-04, 1.059245413830e+05, 7.540263124843e+06),
    (3.0999999493e-02, 3.0000000000e+09,  3,  3, 1.151004107669e-03, 5.862015910124e+05, 2.692187377889e+07),
    (5.0000000745e-02, 3.0000000000e+09,  3,  3, 1.856458256952e-03, 9.454864466797e+05, 2.836459279896e+07),
    (1.0000000149e-01, 3.0000000000e+09,  3,  3, 3.712916513905e-03, 1.890972893359e+06, 5.672918559791e+07),
    (3.0000001192e-01, 3.0000000000e+09,  3,  3, 1.113874930888e-02, 5.672918561499e+06, 6.663334000000e+07),
    (1.0000000000e+00, 3.0000000000e+09,  3,  3, 3.712916746736e-02, 1.890973011939e+07, 6.663334000000e+07),
    (2.0000000000e+00, 3.0000000000e+09,  3,  3, 7.425833493471e-02, 3.781946023878e+07, 6.663334000000e+07),
)

_WRF_NU_C_STAGE_DT = 30.0
_WRF_NU_C_STAGE_P = 85000.0
_WRF_NU_C_STAGE_T = 265.0
_WRF_NU_C_STAGE_QV = 2.0e-3


#: module_mp_thompson.F:2029-2088 verbatim: ``(temp_k, smob, smoc, ns)`` for
#: 23 temperatures from -0.1 C to -70 C crossed with four snow contents
#: (1e-12, 1e-8, 1e-5, 1e-2 kg m^-3), ns spanning 2.3e-1 to 2.5e+08 m^-3.
#: The full sweep run was 23 x 19 = 437 states; this is the committed
#: subsample and the device helper is bit-exact on ALL of them.
_WRF_SNOW_NUMBER = (
    (2.7304998779e+02, 1.449275421894e-11, 9.064083852137e-18, 9.575814208984e+02),
    (2.7304998779e+02, 1.449275401910e-07, 4.325675234673e-12, 4.204513183594e+03),
    (2.7304998779e+02, 1.449275296181e-04, 7.854189476575e-08, 1.275324511719e+04),
    (2.7304998779e+02, 1.449275314808e-01, 1.426096539944e-03, 3.868348437500e+04),
    (2.7214999390e+02, 1.449275421894e-11, 1.054789780573e-17, 7.071176757812e+02),
    (2.7214999390e+02, 1.449275401910e-07, 4.566699015468e-12, 3.772408691406e+03),
    (2.7214999390e+02, 1.449275296181e-04, 7.707786409128e-08, 1.324231835938e+04),
    (2.7214999390e+02, 1.449275314808e-01, 1.300939242356e-03, 4.648462890625e+04),
    (2.7114999390e+02, 1.449275421894e-11, 1.244434170890e-17, 5.080186767578e+02),
    (2.7114999390e+02, 1.449275401910e-07, 4.843326260884e-12, 3.353791503906e+03),
    (2.7114999390e+02, 1.449275296181e-04, 7.546968561201e-08, 1.381269238281e+04),
    (2.7114999390e+02, 1.449275314808e-01, 1.175984041765e-03, 5.688798828125e+04),
    (2.7014999390e+02, 1.449275421894e-11, 1.463343714972e-17, 3.673929443359e+02),
    (2.7014999390e+02, 1.449275401910e-07, 5.128886898520e-12, 2.990730468750e+03),
    (2.7014999390e+02, 1.449275296181e-04, 7.388064204861e-08, 1.441325488281e+04),
    (2.7014999390e+02, 1.449275314808e-01, 1.064236741513e-03, 6.946192968750e+04),
    (2.6814999390e+02, 1.449275421894e-11, 2.003648992105e-17, 1.959657592773e+02),
    (2.6814999390e+02, 1.449275401910e-07, 5.725449194610e-12, 2.399962158203e+03),
    (2.6814999390e+02, 1.449275296181e-04, 7.076206287593e-08, 1.571167382812e+04),
    (2.6814999390e+02, 1.449275314808e-01, 8.745637023821e-04, 1.028585859375e+05),
    (2.6614999390e+02, 1.449275421894e-11, 2.707606990347e-17, 1.073129959106e+02),
    (2.6614999390e+02, 1.449275401910e-07, 6.352693978501e-12, 1.949430053711e+03),
    (2.6614999390e+02, 1.449275296181e-04, 6.772314264936e-08, 1.715335742188e+04),
    (2.6614999390e+02, 1.449275314808e-01, 7.219653343782e-04, 1.509352968750e+05),
    (2.6414999390e+02, 1.449275421894e-11, 3.611116521104e-17, 6.033100128174e+01),
    (2.6414999390e+02, 1.449275401910e-07, 7.005996911086e-12, 1.602815795898e+03),
    (2.6414999390e+02, 1.449275296181e-04, 6.476512481868e-08, 1.875603320312e+04),
    (2.6414999390e+02, 1.449275314808e-01, 5.987045587972e-04, 2.194817031250e+05),
    (2.6214999390e+02, 1.449275421894e-11, 4.753243835781e-17, 3.482117080688e+01),
    (2.6214999390e+02, 1.449275401910e-07, 7.679739656707e-12, 1.333922363281e+03),
    (2.6214999390e+02, 1.449275296181e-04, 6.188903256543e-08, 2.053978906250e+04),
    (2.6214999390e+02, 1.449275314808e-01, 4.987478023395e-04, 3.162725000000e+05),
    (2.6014999390e+02, 1.449275421894e-11, 6.174896655263e-17, 2.063308715820e+01),
    (2.6014999390e+02, 1.449275401910e-07, 8.367317255731e-12, 1.123701782227e+03),
    (2.6014999390e+02, 1.449275296181e-04, 5.909539524396e-08, 2.252766406250e+04),
    (2.6014999390e+02, 1.449275314808e-01, 4.173698544037e-04, 4.516284062500e+05),
    (2.5814999390e+02, 1.449275421894e-11, 7.917013140683e-17, 1.255165100098e+01),
    (2.5814999390e+02, 1.449275401910e-07, 9.061288178125e-12, 9.581724243164e+02),
    (2.5814999390e+02, 1.449275296181e-04, 5.638470668146e-08, 2.474575781250e+04),
    (2.5814999390e+02, 1.449275314808e-01, 3.508591034915e-04, 6.390840000000e+05),
    (2.5514999390e+02, 1.449275421894e-11, 1.121390756988e-16, 6.256185054779e+00),
    (2.5514999390e+02, 1.449275401910e-07, 1.009608976282e-11, 7.718219604492e+02),
    (2.5514999390e+02, 1.449275296181e-04, 5.247474632597e-08, 2.857082226562e+04),
    (2.5514999390e+02, 1.449275314808e-01, 2.727391838562e-04, 1.057617000000e+06),
    (2.5314999390e+02, 1.449275421894e-11, 1.391287747114e-16, 4.064336299896e+00),
    (2.5314999390e+02, 1.449275401910e-07, 1.076875828093e-11, 6.784100341797e+02),
    (2.5314999390e+02, 1.449275296181e-04, 4.997203717494e-08, 3.150426367188e+04),
    (2.5314999390e+02, 1.449275314808e-01, 2.318934857612e-04, 1.463006750000e+06),
    (2.5014999390e+02, 1.449275421894e-11, 1.875884428312e-16, 2.235688924789e+00),
    (2.5014999390e+02, 1.449275401910e-07, 1.172850878917e-11, 5.719235229492e+02),
    (2.5014999390e+02, 1.449275296181e-04, 4.637354678039e-08, 3.658329687500e+04),
    (2.5014999390e+02, 1.449275314808e-01, 1.833571441239e-04, 2.340064000000e+06),
    (2.4814999390e+02, 1.449275421894e-11, 2.252152610048e-16, 1.551057577133e+00),
    (2.4814999390e+02, 1.449275401910e-07, 1.232156737752e-11, 5.181930541992e+02),
    (2.4814999390e+02, 1.449275296181e-04, 4.407750253677e-08, 4.049389062500e+04),
    (2.4814999390e+02, 1.449275314808e-01, 1.576768991072e-04, 3.164370000000e+06),
    (2.4514999390e+02, 1.449275421894e-11, 2.890552077971e-16, 9.415901899338e-01),
    (2.4514999390e+02, 1.449275401910e-07, 1.311767621609e-11, 4.572037353516e+02),
    (2.4514999390e+02, 1.449275296181e-04, 4.078626503201e-08, 4.729285937500e+04),
    (2.4514999390e+02, 1.449275314808e-01, 1.268151099794e-04, 4.891943500000e+06),
    (2.4314999390e+02, 1.449275421894e-11, 3.358153704827e-16, 6.976254582405e-01),
    (2.4314999390e+02, 1.449275401910e-07, 1.357340802255e-11, 4.270175170898e+02),
    (2.4314999390e+02, 1.449275296181e-04, 3.869279652235e-08, 5.254884765625e+04),
    (2.4314999390e+02, 1.449275314808e-01, 1.102989626816e-04, 6.466668000000e+06),
    (2.4014999390e+02, 1.449275421894e-11, 4.102779045269e-16, 4.673769772053e-01),
    (2.4014999390e+02, 1.449275401910e-07, 1.412519146787e-11, 3.943073425293e+02),
    (2.4014999390e+02, 1.449275296181e-04, 3.570109541329e-08, 6.172489843750e+04),
    (2.4014999390e+02, 1.449275314808e-01, 9.023369057104e-05, 9.662422000000e+06),
    (2.3814999390e+02, 1.449275421894e-11, 4.612411505759e-16, 3.698004484177e-01),
    (2.3814999390e+02, 1.449275401910e-07, 1.439580746276e-11, 3.796220703125e+02),
    (2.3814999390e+02, 1.449275296181e-04, 3.380391433438e-08, 6.884770312500e+04),
    (2.3814999390e+02, 1.449275314808e-01, 7.937760528876e-05, 1.248611900000e+07),
    (2.3314999390e+02, 1.449275421894e-11, 5.835478906596e-16, 2.310311347246e-01),
    (2.3314999390e+02, 1.449275401910e-07, 1.469958009481e-11, 3.640940551758e+02),
    (2.3314999390e+02, 1.449275296181e-04, 2.939180099304e-08, 9.106908593750e+04),
    (2.3314999390e+02, 1.449275314808e-01, 5.876889554202e-05, 2.277867000000e+07),
    (2.2814999390e+02, 1.449275421894e-11, 6.800640656861e-16, 1.701076477766e-01),
    (2.2814999390e+02, 1.449275401910e-07, 1.445102024156e-11, 3.767268066406e+02),
    (2.2814999390e+02, 1.449275296181e-04, 2.543372801256e-08, 1.216195390625e+05),
    (2.2814999390e+02, 1.449275314808e-01, 4.476324465941e-05, 3.926271600000e+07),
    (2.2314999390e+02, 1.449275421894e-11, 7.300427153432e-16, 1.476138085127e-01),
    (2.2314999390e+02, 1.449275401910e-07, 1.367779674244e-11, 4.205244140625e+02),
    (2.2314999390e+02, 1.449275296181e-04, 2.190367531796e-08, 1.639794531250e+05),
    (2.2314999390e+02, 1.449275314808e-01, 3.507663859637e-05, 6.394218400000e+07),
    (2.1314999390e+02, 1.449275421894e-11, 6.575231425311e-16, 1.819707006216e-01),
    (2.1314999390e+02, 1.449275401910e-07, 1.093502198513e-11, 6.579368286133e+02),
    (2.1314999390e+02, 1.449275296181e-04, 1.601404697738e-08, 3.067758750000e+05),
    (2.1314999390e+02, 1.449275314808e-01, 2.345214670640e-05, 1.430402400000e+08),
    (2.0314999390e+02, 1.449275421894e-11, 4.263567339180e-16, 4.327901005745e-01),
    (2.0314999390e+02, 1.449275401910e-07, 7.511401223237e-12, 1.394381469727e+03),
    (2.0314999390e+02, 1.449275296181e-04, 1.148634343195e-08, 5.962931875000e+05),
    (2.0314999390e+02, 1.449275314808e-01, 1.756477831805e-05, 2.549988000000e+08),
)

#: The two states used by
#: :func:`test_two_gamma_snow_number_and_not_smo0_decides_the_koop_gate`:
#: ``(rs, smob, smoc, ns, smo0)`` at T = 233.15 K.  ``ns`` is WRF's explicit
#: two-gamma integral (:2081-2088); ``smo0`` is the zeroth power-law moment
#: (:2051-2054) that the mp=8 snow closure carries.  999e3 m^-3 is WRF's
#: Koop gate at :2635.  ns STRADDLES that gate across these two states and
#: smo0 does NOT -- both smo0 values are below it -- so the two closures
#: give OPPOSITE answers at the second state.
_WRF_SNOW_KOOP_GATE = (
    (9.708938887343e-05, 1.407092669979e-03, 3.584415537716e-07, 5.604051875000e+05, 3.870780468750e+04),
    (5.003837286495e-04, 7.251938339323e-03, 2.177617488996e-06, 2.078595000000e+06, 1.656056875000e+05),
)

#: ``(qiten, niten)`` per kilogram per second at every level of
#: ``aero-ice-koop``, read out of module_mp_thompson.F's own tendency loop
#: immediately after the ice mass/number balance at :3035-3055 -- i.e. after
#: every cold source and BEFORE the condensation and sedimentation blocks,
#: which is exactly the state ``thompson_aa_cold_network`` is responsible
#: for.  dtsave = 10 s.
_WRF_ICE_KOOP_COLD_TENDENCY = (
    (5.8204235875436439E-10, 6.0556982421875000E+02),
    (6.0445515259743843E-10, 8.2969763183593750E+02),
    (8.4538098743536239E-10, 3.2389560546875000E+03),
    (1.4272832915551703E-09, 9.0579775390625000E+03),
    (3.5824017086127924E-09, 3.0609162109375000E+04),
    (5.8297078275870717E-10, 6.1485345458984375E+02),
    (6.1328342404465275E-10, 9.1797943115234375E+02),
    (9.3918961496797237E-10, 4.1770419921875000E+03),
    (1.7263277474199867E-09, 1.2048422851562500E+04),
    (4.6343879844812363E-09, 4.1129027343750000E+04),
    (5.8421112392181840E-10, 6.2726196289062500E+02),
    (6.2522098609463228E-10, 1.0373602294921875E+03),
    (1.0643244063857082E-09, 5.4283896484375000E+03),
    (2.1224735302638464E-09, 1.6009880859375000E+04),
    (6.0444587113295256E-09, 5.5229738281250000E+04),
    (5.8586296924900694E-10, 6.4377954101562500E+02),
    (6.4070210248345916E-10, 1.1921628417968750E+03),
    (1.2305209073915080E-09, 7.0903544921875000E+03),
    (2.6549815679288713E-09, 2.1334966796875000E+04),
    (7.8903443778699511E-09, 7.3688593750000000E+04),
    (5.8802068769736593E-10, 6.6535229492187500E+02),
    (6.6137950671674162E-10, 1.3989407958984375E+03),
    (1.4470037390523771E-09, 9.2551835937500000E+03),
    (3.3347207217815367E-09, 2.8132359375000000E+04),
)


def test_two_gamma_snow_number_matches_the_wrf_fortran_integral():
    """``thompson_aa_snow_number`` against Fortran, not against a host copy.

    The helper is the ONLY consumer of WRF's explicit bimodal snow-number
    integral (:2081-2088) and its only use in the whole routine is the Koop
    gate at :2635.  Until now nothing in the suite compared it to a compiled
    WRF -- the shared header's own gate is a host NumPy transcription of the
    same expressions, which cannot catch a transcription error.  This does.

    MEASURED, and reported rather than rounded off: over a WIDE sweep of
    3721 states (61 temperatures from 273.05 K to 201.05 K crossed with 61
    log-spaced snow contents from 1e-12 to 1e-2 kg m^-3, ns spanning 1.5e-1
    to 2.8e+08 m^-3) the device helper is BIT-EXACT on 3719 and one float32
    ulp away on exactly 2 -- (268.25 K, 6.813e-6) and (265.85 K, 4.642e-5),
    max 6.924e-08 relative.  Rebuilding the helper with every operation
    contraction-pinned and both powers correctly rounded reproduces the
    SAME 3719/3721, so those two are not a CUDA powf artefact and there is
    nothing left to repair.  The 92 states committed below -- 23
    temperatures x four snow contents -- are all bit-exact, which is why
    this gate is written as an equality.

    ``ns`` is NOT ``smo0``: across these states ns/smo0 runs from 0.83 to
    1226, so substituting the mp=8 zeroth-moment snow closure is a
    three-order-of-magnitude error at the cold end, not a rounding one.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_launch import probe_snow_number

    table = np.asarray(_WRF_SNOW_NUMBER, dtype=np.float64)
    smob = cp.asarray(table[:, 1].astype(np.float32).copy())
    smoc = cp.asarray(table[:, 2].astype(np.float32).copy())
    got = cp.asnumpy(probe_snow_number(smob, smoc))
    want = table[:, 3].astype(np.float32)

    bad = np.nonzero(got != want)[0]
    assert bad.size == 0, (
        "thompson_aa_snow_number is not bit-exact against WRF v4.6.1's "
        ":2081-2088 at "
        + ", ".join(f"T={table[i, 0]:.2f} got={got[i]!r} want={want[i]!r}"
                    for i in bad[:6]))


#: The two states, out of a 3721-state Fortran sweep, where
#: ``thompson_aa_snow_number`` used to miss compiled WRF by one float32 ulp.
#: Columns: temp_k, rs [kg m^-3], smob, smoc, ns [m^-3] -- all five as WRF's
#: own ``probe_snow_moments`` (a verbatim copy of :2028-2088 compiled INTO
#: module_mp_thompson, so Kap0/Lam0/csg(15) are thompson_init's) produced them.
_WRF_SNOW_NUMBER_FORMER_SURVIVORS = (
    (268.25, 6.8129234e-06, 9.8738026e-05, 4.199831e-08, 14104.59),
    (265.84998, 4.641592e-05, 0.0006726945, 5.257931e-07, 28457.361),
)


def test_the_two_former_snow_number_survivors_are_bit_exact():
    """Both one-ulp survivors are gone, and must stay gone.

    They were the double-rounding limit of the old helper, which evaluated
    ``REAL(4)**REAL(4)`` as ``(float)pow((double)x,(double)y)`` where gfortran
    lowers it to the C library's singly rounded ``powf``.  The helper now
    calls WOOF's own powf word (thompson_aerosol_libm.cuh), which returns the
    oracle host's word, and both states are bit-exact; the ratchet that
    pinned them at one ulp retires with the cause it accepted
    (development branch).
    """
    import cupy as cp

    from woof.core.thompson_aerosol_launch import probe_snow_number

    table = np.asarray(_WRF_SNOW_NUMBER_FORMER_SURVIVORS, dtype=np.float64)
    smob = cp.asarray(table[:, 2].astype(np.float32).copy())
    smoc = cp.asarray(table[:, 3].astype(np.float32).copy())
    got = cp.asnumpy(probe_snow_number(smob, smoc))
    want = table[:, 4].astype(np.float32)
    assert np.array_equal(got.view(np.int32), want.view(np.int32)), (got, want)


def test_snow_free_levels_contribute_exactly_zero_to_the_koop_gate(
        classic_tables):
    """WRF's ``ns(k)`` is EXACTLY ZERO where there is no snow, and it matters.

    ``ns(k) = 0.`` at module_mp_thompson.F:1790, and the only place it is ever
    overwritten is the moment loop at :2026-2088, which OPENS with
    ``if (.not. L_qs(k)) CYCLE`` (:2027).  ``L_qs`` is set at :1906-1913 from
    ``qs1d(k) .gt. R1``.  So at a snow-free level the Koop gate at :2634 reads
    ``xni = 0. + ni(k) + ...`` and the threshold is at ``ni = 999.E3`` exactly.

    A port that dropped the guard would not blow up: WRF leaves the sentinel
    ``rs(k) = R1`` there (:1912), and feeding that through the Field fits and
    the two-gamma integral gives a SMALL but nonzero ns -- MEASURED with the
    device helpers at this cell's temperature, 0.2310 m^-3 (0.1869 at 230 K,
    0.1469 at 220 K).  That is exactly the kind of defect that survives every
    column fixture and then bites once, so it is gated numerically here rather
    than by reading the source.

    THE DISCRIMINATING CELL is ``ni`` chosen so the gate's ``xni`` lands
    0.1 m^-3 BELOW 999.E3.  With WRF's ``ns = 0`` the gate is OPEN and Koop
    fires; with the sentinel-derived 0.2310 it would be SHUT.  The cells
    either side pin the threshold itself at 999.E3 to within 0.1 m^-3, which
    is 4000x finer than the perturbation being excluded.

    No cloud, rain or graupel, so ``pni_rfz``/``pni_wfz``/``pni_ihm`` are
    zero, and ``pni_inu`` contributes nothing here (iceDeMott's ``xnc`` is
    far below ``xni`` at this loading, so :2628's
    ``0.5*(xnc-xni+|xnc-xni|)`` is identically zero).  The entry ice carries
    mass (50 micron crystals): WRF reads ``ni(k)`` only where ``qi1d`` passed
    R1 and otherwise sets it to R2 (:1871-1874), so ice number without ice
    mass never reaches the gate.  The synthetic tables are zero, so the
    deposition stays in snow (``tpi_ide = 0``) and nothing autoconverts, and
    the entire ``qi`` response is the Koop term.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import launch_aa_cold_network
    from woof.core.thompson_aerosol_launch import (
        probe_field_ab, probe_snow_number)

    f32 = np.float32
    temp = f32(233.15)
    pres = f32(50000.0)
    qv = f32(2.3454424e-4)          # 0.99 * RSLF(50000, 233.15)
    dt = 30.0
    rho = f32(f32(0.622) * pres
              / (f32(287.04) * temp * (qv + f32(0.622))))

    # (1) What a DROPPED :2027 guard would inject, measured on device from
    #     WRF's own sentinel rs(k) = R1.
    tc0 = f32(min(-0.1, float(temp) - 273.15))
    smob = f32(f32(1.0e-12) * f32(1.0 / 0.069))
    a_dev, b_dev = probe_field_ab(cp.asarray(np.asarray([tc0], f32)),
                                  cp.asarray(np.asarray([3.0], f32)))
    a_ = float(cp.asnumpy(a_dev)[0])
    b_ = float(cp.asnumpy(b_dev)[0])
    smoc = f32(a_ * float(smob) ** b_)
    sentinel_ns = float(cp.asnumpy(probe_snow_number(
        cp.asarray(np.asarray([smob], f32)),
        cp.asarray(np.asarray([smoc], f32))))[0])
    assert 0.2 < sentinel_ns < 0.3, sentinel_ns

    # (2) The ladder.  Targets are per-m3; the kernel forms ni*rho itself.
    targets = np.asarray(
        [998000.0, 998999.9, 999000.0, 999000.5, 1000000.0])
    ni_entry = (targets / float(rho)).astype(f32)
    n = targets.size

    def const(value):
        return cp.asarray(np.full(n, value, f32))

    def zeros():
        return cp.zeros(n, dtype=cp.float32)

    # 50 micron mean crystals at the threshold number (:1860-1869 leave
    # them alone): ri = am_i*6*ni*(D/4)**3.
    am_i = np.pi * 890.0 / 6.0
    qi_entry = f32(am_i * 6.0 * 999000.0 * (50.0e-6 / 4.0) ** 3
                   / float(rho))
    qi = const(qi_entry)
    ni = cp.asarray(ni_entry.copy())
    ncten, nwfaten, nifaten = zeros(), zeros(), zeros()
    launch_aa_cold_network(
        qi, ni, zeros(), zeros(), zeros(), zeros(), zeros(),
        const(temp), const(pres), const(qv),
        zeros(), const(5.0e9), const(1.0e6),
        ncten, nwfaten, nifaten, zeros(), zeros(),
        *_synthetic_table_arguments(cp, classic_tables), dt,
        snow_cloud_efficiency=classic_tables.t_Efsw)
    cp.cuda.Stream.null.synchronize()

    qi_out = cp.asnumpy(qi).astype(np.float64)
    fired = qi_out > float(qi_entry)
    # The gate is OPEN at and below 999.E3 and SHUT above it.  Index 1 is the
    # discriminating cell: 0.1 m^-3 below the threshold, i.e. inside the
    # 0.2310 m^-3 a dropped guard would have added.
    assert fired.tolist() == [True, True, True, False, False], (
        f"Koop fired at {fired.tolist()} for xni targets {targets.tolist()}; "
        "with ns(k) = 0 the threshold is exactly 999.E3")
    assert targets[2] - targets[1] < sentinel_ns, (
        "the discriminating cell is no longer inside the perturbation a "
        "dropped :2027 guard would inject; re-choose it")
    # All three open cells must produce the SAME ice, because ns is zero at
    # every one of them rather than merely small.
    assert qi_out[0] == qi_out[1] == qi_out[2] > float(qi_entry), (
        qi_out.tolist())

    # (3) And the structural statement, so a future reader does not have to
    #     re-derive it: the level transcription initialises ns to zero and
    #     assigns it ONLY inside the L_qs branch.
    source = (_REPO / "woof" / "core" / "kernels"
              / "thompson_aerosol_common.cuh").read_text(
        encoding="utf-8")
    assert "    L->ns = 0.0f;\n    if (L->L_qs) {" in source
    assert source.count("L->ns = thompson_aa_snow_number(") == 1


def test_two_gamma_snow_number_and_not_smo0_decides_the_koop_gate(
        classic_tables):
    """ns(k) is PRODUCTION-observable, and the mp=8 closure fails here.

    Until this gate, ``thompson_aa_snow_number`` had no consumer any test
    could see: the Koop gate at :2634-2637 either opens or it does not, and
    no committed fixture puts a snow-bearing, ice-supersaturated, below-238 K
    level near the 999e3 m^-3 threshold.  A kernel that substituted the mp=8
    zeroth-moment snow number ``smo0`` for WRF's explicit integral would have
    passed every previous test in this file.

    The column below is three cells that differ ONLY in ``qs``:

      * ``qs = 0``     -- ns is identically zero, the gate is open;
      * ``rs = 9.709e-5`` -- ns = 5.604e5 < 999e3, the gate is open;
      * ``rs = 5.004e-4`` -- ns = 2.079e6 > 999e3, THE GATE IS SHUT.

    WRF's ``smo0`` at those same two states is 3.87e4 and 1.66e5, BOTH below
    999e3, so the mp=8 closure leaves the gate open in the third cell and
    the last assertion below fails.

    Cloud water, rain, graupel and entry cloud ice are all zero, which makes
    the ice number budget exactly ``(pni_inu + pni_iha)*orho*dt``: Hallett-
    Mossop needs cloud water, ``pni_ide`` needs entry ice, and ``pni_rfz``/
    ``pni_wfz`` need rain/cloud.  ``pni_inu`` (iceDeMott) does not depend on
    qs at all, so the whole qs dependence of ``ni`` is the Koop term.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import launch_aa_cold_network
    from woof.core.thompson_aerosol_launch import (
        probe_ice_koop, probe_saturation, probe_snow_number)

    f32 = np.float32
    temp = f32(233.15)
    pres = f32(50000.0)
    qv = f32(2.3454424e-4)          # 0.99 * RSLF(50000, 233.15)
    dt = 30.0
    rho = f32(f32(0.622) * pres
              / (f32(287.04) * temp * (qv + f32(0.622))))
    qs_column = np.asarray([0.0, 1.3e-4, 6.7e-4], f32)
    rs_column = (qs_column * rho).astype(f32)

    gate = np.asarray(_WRF_SNOW_KOOP_GATE, dtype=np.float64)
    np.testing.assert_array_equal(
        rs_column[1:].astype(np.float64),
        gate[:, 0].astype(f32).astype(np.float64),
        err_msg="the column no longer forms the rs values the WRF snow "
                "probe was run at; regenerate _WRF_SNOW_KOOP_GATE")

    # The premise, stated as an assertion so it cannot rot: the two closures
    # disagree at the third cell.
    ns_lo, ns_hi = gate[0, 3], gate[1, 3]
    smo0_lo, smo0_hi = gate[0, 4], gate[1, 4]
    assert ns_lo < 999.0e3 < ns_hi, (ns_lo, ns_hi)
    assert smo0_lo < 999.0e3 and smo0_hi < 999.0e3, (smo0_lo, smo0_hi)

    # And the device helper reproduces WRF's ns at exactly those states.
    got_ns = cp.asnumpy(probe_snow_number(
        cp.asarray(gate[:, 1].astype(f32).copy()),
        cp.asarray(gate[:, 2].astype(f32).copy())))
    np.testing.assert_array_equal(got_ns, gate[:, 3].astype(f32))

    n = qs_column.size

    def const(value):
        return cp.asarray(np.full(n, value, f32))

    def zeros():
        return cp.zeros(n, dtype=cp.float32)

    qi = zeros()
    ni = zeros()
    ncten, nwfaten, nifaten = zeros(), zeros(), zeros()
    launch_aa_cold_network(
        qi, ni, cp.asarray(qs_column.copy()), zeros(), zeros(), zeros(),
        zeros(), const(temp), const(pres), const(qv),
        zeros(), const(5.0e9), const(1.0e6),
        ncten, nwfaten, nifaten, zeros(), zeros(),
        *_synthetic_table_arguments(cp, classic_tables), dt,
        snow_cloud_efficiency=classic_tables.t_Efsw)
    cp.cuda.Stream.null.synchronize()

    ni_out = cp.asnumpy(ni).astype(np.float64)
    qi_out = cp.asnumpy(qi).astype(np.float64)

    # Cell 2 (gate shut): every crystal is an iceDeMott crystal, and WRF
    # gives those the deposition-nucleated mass xm0i = 1e-12 exactly.  The
    # ratio is divided by xm0i before the comparison ON PURPOSE: pytest's
    # approx carries a DEFAULT ABSOLUTE tolerance of 1e-12, which would
    # swallow this entire assertion if it were written against 1e-12.
    mean_mass = qi_out[2] / ni_out[2]
    assert mean_mass / 1.0e-12 == pytest.approx(1.0, rel=1.0e-6), (
        "the Koop gate did not shut at ns = 2.079e6 m^-3; a snow number "
        f"below 999e3 was used.  qi/ni = {mean_mass:.6e}, and Koop crystals "
        "(xm0i*0.1) drag that ratio ten-fold below xm0i")

    # Cells 0 and 1 (gate open) carry the Koop crystals, whose mass is
    # xm0i*0.1 -- ten times smaller -- so the mean crystal mass drops.
    assert ni_out[0] == ni_out[1], (
        "ns = 5.604e5 < 999e3 must leave the gate open exactly as ns = 0 "
        "does")
    koop_number = float(cp.asnumpy(probe_ice_koop(
        const(temp), const(qv),
        cp.asarray(cp.asnumpy(probe_saturation(
            const(pres), const(temp))[0])[:1].repeat(n)),
        const(min(np.float32(9999.0e6), f32(f32(5.0e9) * rho))),
        const(dt)))[0])
    expected_gap = koop_number / float(rho)
    measured_gap = ni_out[1] - ni_out[2]
    assert measured_gap == pytest.approx(expected_gap, rel=5.0e-7), (
        f"the qs dependence of ni is {measured_gap:.6e} kg^-1 but iceKoop "
        f"accounts for {expected_gap:.6e}")
    # And it is a large signal, not a rounding one.
    assert measured_gap > 5.0 * ni_out[2]


def _synthetic_table_arguments(cp, classic_tables):
    """The synthetic zero table set, with the canonical ``t_Efrw``.

    ``rain_cloud_efficiency`` has to be the real table because ``pnc_rcw``
    genuinely indexes it; every other coefficient set is zeroed so the only
    live paths are the ones the calling gate is about.
    """
    tables = _synthetic_tables(cp)
    return (
        tables["ice_deposition_partition"], tables["ice_to_snow_mass"],
        tables["ice_to_snow_number"], tables["rain_snow"],
        tables["rain_graupel"], tables["rain_freezing"],
        classic_tables.cold_source_tables.rain_cloud_efficiency,
        tables["cloud_freezing"])


def test_production_kernel_uses_the_working_stage_nu_c(classic_tables):
    """CRITICAL, and previously unguarded: the PRODUCTION kernel's own output.

    ``test_working_nu_c_is_recomputed_from_the_rediagnosed_nc`` drives the
    READBACK PROBE, and
    ``test_probe_and_production_kernel_agree_on_the_four_new_rates`` compares
    the production kernel against that probe using ``ncten`` -- which on
    every staging-divergent state is pinned to the ``nc*odts`` cap at :2192
    and therefore cannot tell the two stages apart.  So nothing failed if
    ``thompson_aerosol_cold.cu``'s production kernel regressed from
    :2170's nu_c to :1832's.

    This gate closes that.  It drives the PRODUCTION kernel on WRF's own
    (nc1d, qc1d) grid with no rain, no ice, no snow and no graupel, so the
    only live rates are the :2178-2193 autoconversion group, and compares
    ``qr``, ``nr`` and ``ncten`` against the Fortran table.

    ``nr`` is the discriminator, and the reason is WRF's own algebra:
    ``pnr_wau = prr_wau/(am_r*nu_c*10*D0r**3)`` (:2191) is LINEAR in nu_c
    while ``prr_wau`` is pinned to ``rc*odts`` by the MIN at :2190 across
    this whole regime.  MEASURED with line 404 reverted to the entry stage:
    ``qr`` and ``ncten`` do not move at all and ``nr`` is wrong by 25.0% to
    80.0% on all 36 divergent rows.

    THE UPSTREAM CAUSE OF THE OLD 5e-7 BOUND IS GONE, so the bound moved.
    This docstring used to record a wider 31 x 41 sweep whose tail --
    2.817e-06 on qr, 2.806e-06 on nr, 2.443e-06 on ncten -- occurred ONLY on
    NON-divergent states, where ``prr_wau`` is not cap-pinned and the
    Berry-Reinhardt ``Dc_b`` cancellation at :2182 amplifies
    ``thompson_aa_cloud_dist``'s CUDA-powf residual; it named
    ``thompson_aerosol_common.cuh`` as the owner.  That helper is now
    correctly-rounded AND contraction-pinned, and on the committed grid
    ``qr`` is BIT-EXACT while ``nr`` measures 1.2613e-07 and ``ncten``
    4.8050e-08.
    The same repair took the full 11340-row cold-warm-loop oracle's prr_wau /
    pnr_wau / pnc_wau / nc_m3 / mvd_c from 1.23e-07..2.31e-06 to EXACT.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import launch_aa_cold_network

    table = np.asarray(_WRF_NU_C_STAGE, dtype=np.float64)
    qc = table[:, 0]
    nc = table[:, 1]
    nu_entry = table[:, 2]
    nu_working = table[:, 3]
    prr_wau = table[:, 4]
    pnr_wau = table[:, 5]
    pnc_wau = table[:, 6]

    observable = (nu_entry != nu_working) & (prr_wau > 0.0)
    assert int(observable.sum()) >= 30, (
        "the grid must reach the regime where the staging is observable at "
        f"all; only {int(observable.sum())} of {table.shape[0]} rows are")
    ratio = nu_working[observable] / nu_entry[observable]
    assert float(np.abs(ratio - 1.0).min()) >= 0.13
    assert float(np.abs(ratio - 1.0).max()) >= 0.75

    f32 = np.float32
    n = table.shape[0]
    dt = _WRF_NU_C_STAGE_DT

    def const(value):
        return cp.asarray(np.full(n, value, f32))

    def zeros():
        return cp.zeros(n, dtype=cp.float32)

    qr = zeros()
    nr = zeros()
    ncten, nwfaten, nifaten = zeros(), zeros(), zeros()
    launch_aa_cold_network(
        zeros(), zeros(), zeros(), zeros(), qr, nr,
        cp.asarray(qc.astype(f32).copy()),
        const(_WRF_NU_C_STAGE_T), const(_WRF_NU_C_STAGE_P),
        const(_WRF_NU_C_STAGE_QV),
        cp.asarray(nc.astype(f32).copy()), const(1.0e9), const(1.0e6),
        ncten, nwfaten, nifaten, zeros(), zeros(),
        *_synthetic_table_arguments(cp, classic_tables), dt,
        snow_cloud_efficiency=classic_tables.t_Efsw)
    cp.cuda.Stream.null.synchronize()

    rho = f32(f32(0.622) * f32(_WRF_NU_C_STAGE_P)
              / (f32(287.04) * f32(_WRF_NU_C_STAGE_T)
                 * (f32(_WRF_NU_C_STAGE_QV) + f32(0.622))))
    orho = f32(f32(1.0) / rho)
    want_qr = (f32(prr_wau.astype(f32) * orho) * f32(dt)).astype(np.float64)
    want_nr = (f32(pnr_wau.astype(f32) * orho) * f32(dt)).astype(np.float64)
    want_ncten = -pnc_wau * np.float64(orho)

    active = prr_wau > 0.0
    # TIGHTENED from 5.0e-7 on all three.  qr is now BIT-EXACT against WRF
    # on every active row; nr measures 1.2613e-07 (WRF's :2191 divides the
    # now-exact prr_wau by am_r*nu_c*10*D0r**3, and that float32 quotient is
    # all that is left) and ncten 4.8050e-08 (its reference is formed in
    # float64 on the host, so it cannot be bitwise by construction).  The
    # cause of the old bound was upstream and is named in the docstring.
    for name, got, want, bound in (
            ("qr", cp.asnumpy(qr), want_qr, 0.0),
            ("nr", cp.asnumpy(nr), want_nr, 2.0e-7),
            ("ncten", cp.asnumpy(ncten), want_ncten, 1.0e-7)):
        got = got.astype(np.float64)[active]
        reference = want[active]
        worst = float(np.max(np.abs(got - reference)
                             / np.maximum(np.abs(reference), 1.0e-300)))
        assert worst <= bound, (
            f"{name}: max relative difference {worst:.4e} against WRF "
            f"v4.6.1's :2168-2193 (bound {bound:.1e}).  If only `nr` moved, "
            "the production kernel is using the :1832 entry-stage nu_c "
            "instead of :2170's -- see thompson_aerosol_cold.cu:404")

    # The gate provably discriminates: the entry-stage answer for nr is
    # pnr_wau*(nu_working/nu_entry), which is 15% to 400% away.
    entry_stage_nr = want_nr[observable] * ratio
    departure = np.abs(entry_stage_nr - want_nr[observable]) / np.abs(
        want_nr[observable])
    assert float(departure.min()) >= 0.13, (
        "every divergent row must put the entry-stage answer far outside "
        "the 5e-7 bound above, or this gate does not discriminate")


def test_cold_network_reproduces_wrfs_own_ice_koop_tendency(classic_tables):
    """The Koop fixture's ice budget, against WRF's own tendency loop.

    ``aero-ice-koop`` is the only committed fixture whose ice comes almost
    entirely from homogeneous haze freezing: at level 15 WRF's
    ``pri_iha = 3.3434858788e-9`` kg m^-3 s^-1 is 90% of ``qiten``, the rest
    being iceDeMott.

    The reference is WRF's ``qiten``/``niten`` read immediately after the
    :3035-3055 ice mass/number balance, which is the last thing that touches
    them before condensation and sedimentation.  ``aero-ice-koop`` has no
    cloud water, no rain, no snow and no graupel, so the cold network is the
    ONLY producer of ice in the whole call and the comparison is exact
    rather than partial.

    REGENERATED against a fresh instrumented build (the oracle harness
    repaired its own ``pii``, which moved this fixture's entry column by ulps
    at 11 of 24 levels).  MEASURED on the regenerated reference: 1.149897e-07
    on qiten and 8.481638e-08 on niten over all 24 levels, i.e. at the float32
    noise floor of the fixture's own CSV round trip -- both bounds below are
    UNCHANGED at 2.0e-07.

    THE LEVEL-15 CLAIM IS NOW MADE ON THE STATE, WHICH IS STRICTLY STRONGER.
    It used to assert that the float64 finite difference
    ``(qi_after - qi_entry)/dt`` equalled WRF's ``qiten`` bit for bit.  That
    quantity is a reconstruction WOOF never forms and WRF never stores, and
    on the regenerated column it sits 4.408228e-08 away -- one ulp of the
    float32 ``qi`` it is differenced from, not a kernel change.  What the
    kernel actually produces is the STATE, so the assertion is now that
    ``qi`` and ``ni`` equal ``float32(entry + WRF_tendency*dt)`` BITWISE:
    23 of 24 levels on each field, and level 15 among them.

    The second half is a SENSITIVITY measurement, no longer an attribution.
    Rerunning on the SAME column with ``p`` raised by ONE float32 ulp at
    level 15 moves ``qiten`` there by 1.6116e-03 and moves NOTHING else --
    four orders of magnitude above the noise floor above, which is what makes
    ``iceKoop``'s ``prob_h`` quantisation (see the test below) visible at all.
    It used to be quoted as the explanation for a 1.6117e-03 G3 residual on
    this fixture; that residual is GONE -- the harness's ``pii`` repair
    removed the need for the reconstruction's ulp shift -- and the
    perturbation is kept because the sensitivity itself is the physics.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_cold import (
        launch_aa_cold_network_from_owner)

    dt = 10.0
    scenario = "aero-ice-koop"
    host, _, _ = _run_cold_network(scenario, dt, classic_tables)
    before, _ = _column(scenario)
    reference = np.asarray(_WRF_ICE_KOOP_COLD_TENDENCY, dtype=np.float64)

    qi_entry = _host(before, "qi").astype(np.float64)
    ni_entry = _host(before, "ni_per_kg").astype(np.float64)
    got_qiten = (host["qi"].astype(np.float64) - qi_entry) / dt
    got_niten = (host["ni"].astype(np.float64) - ni_entry) / dt

    for name, got, want, bound in (
            ("qiten", got_qiten, reference[:, 0], 2.0e-7),
            ("niten", got_niten, reference[:, 1], 2.0e-7)):
        active = np.abs(want) > 0.0
        worst = float(np.max(np.abs(got[active] - want[active])
                             / np.abs(want[active])))
        assert worst <= bound, (
            f"{name}: max relative difference {worst:.4e} against WRF "
            f"v4.6.1's own tendency loop (bound {bound:.1e})")

    # THE STATE, BITWISE.  qi/ni are what the kernel writes; the finite
    # difference above is not.  WRF's own tendency applied to the fixture's
    # own entry value, rounded to float32 exactly as the kernel must:
    level = 14                                    # 0-based; fixture level 15
    want_qi = (qi_entry + reference[:, 0] * dt).astype(np.float32)
    want_ni = (ni_entry + reference[:, 1] * dt).astype(np.float32)
    assert host["qi"][level] == want_qi[level], (
        "level 15 qi must be BIT-EXACT on the fixture's own pressure: got "
        f"{host['qi'][level]!r} want {want_qi[level]!r}")
    assert host["ni"][level] == want_ni[level], (
        f"level 15 ni: got {host['ni'][level]!r} want {want_ni[level]!r}")
    # 24 of 24 since the cold network runs WRF's arithmetic order (it was
    # 23 of 24).
    assert int((host["qi"] == want_qi).sum()) == 24, (
        int((host["qi"] == want_qi).sum()))
    assert int((host["ni"] == want_ni).sum()) == 24, (
        int((host["ni"] == want_ni).sum()))

    # ---- the attribution ------------------------------------------------
    def device(name):
        return cp.asarray(_host(before, name)[:, None, None].copy())

    pressure = _host(before, "p_pa")
    perturbed = pressure.copy()
    perturbed[level] = np.nextafter(
        pressure[level], np.float32(np.inf), dtype=np.float32)
    assert perturbed[level] != pressure[level]

    qi = device("qi")
    ni = device("ni_per_kg")
    zero = cp.zeros_like(qi)
    launch_aa_cold_network_from_owner(
        qi, ni, device("qs"), device("qg"), device("qr"),
        device("nr_per_kg"), device("qc"), device("temp_k"),
        cp.asarray(perturbed[:, None, None].copy()), device("qv"),
        device("nc_per_kg"), device("nwfa_per_kg"), device("nifa_per_kg"),
        zero.copy(), zero.copy(), zero.copy(), zero.copy(),
        cp.full_like(qi, 1.0), classic_tables, dt)
    cp.cuda.Stream.null.synchronize()
    shifted = (cp.asnumpy(qi).ravel().astype(np.float64) - qi_entry) / dt

    moved = abs(shifted[level] - reference[level, 0]) / abs(
        reference[level, 0])
    assert moved == pytest.approx(1.6116e-3, rel=1.0e-3), (
        f"one ulp of pressure at level 15 moved qiten by {moved:.6e}; the "
        "measured value on the regenerated column is 1.6116e-03")
    # ...and that is four orders of magnitude above the kernel's own agreement
    # with WRF on the unperturbed column, which is the point.
    assert moved > 1.0e4 * 1.149897e-07
    # Every other level is untouched -- the perturbation is local, so the
    # adapter's spill into level 14 is its own ice sedimentation, not this
    # kernel's.
    others = [k for k in range(24) if k != level]
    np.testing.assert_array_equal(shifted[others], got_qiten[others])


def test_ice_koop_is_quantised_and_one_pressure_ulp_moves_it_one_quantum():
    """Why ``aero-ice-koop`` misses G3, stated as an executable measurement.

    ``iceKoop`` ends in ``prob_h = 1. - exp(-J_rate*ar_volume*DT)``
    (:5539).  ``exp(...)`` is just under 1, so its float32 ulp is 2^-24 and
    the subtraction is EXACT -- which means ``prob_h`` can only take values
    that are integer multiples of 2^-24.  At ``aero-ice-koop``'s level 15
    the multiple is 561, so ONE step of that grid is 1/561 = 1.78e-3 of the
    answer, and the returned crystal count is a step function of the inputs
    at that granularity.

    ``log_J_rate`` is a cubic in ``delta_aw`` whose four terms are
    -906.7, +2.63e3, -2.59e3 and +8.7e2 and sum to about 10.7, so
    d(log10 J)/d(satw) is about 222 there; one float32 ulp of ``pres``
    (9.7e-8 relative) moves ``qvs``, hence ``satw``, by about the same, and
    that is enough to cross a 2^-24 boundary.

    MEASURED HERE, and this is the whole of the fixture's G3 residual:
    the adapter harness's entry-state reconstruction
    (``tests/test_thompson_aerosol_adapter.py::_reconstruct_entry_state``)
    perturbs ``p`` by exactly +1 ulp at level 15 in order to find a float32
    theta that reproduces ``temp_k`` exactly, and that single ulp costs
    595.97 crystals m^-3 out of 334348.6 -- 1.7825e-3, which is the
    1.612e-3 qi / 1.764e-3 ni the adapter reports.  Driven on the fixture's
    OWN pressure this kernel is bit-exact against WRF (see
    ``test_cold_network_reproduces_wrfs_own_ice_koop_tendency``).

    This is a gate, not a note: if anyone "smooths" iceKoop -- by evaluating
    ``-expm1`` instead of ``1 - exp``, or by widening the subtraction to
    double -- prob_h stops being a multiple of 2^-24 and this test fails.
    WRF's REAL(4) arithmetic is the authority and it is not smooth.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_launch import (
        probe_ice_koop, probe_saturation)

    f32 = np.float32
    # aero-ice-koop level 15 (1-based), straight from the fixture.
    pres = f32(4.0403652343750000e+004)
    temp = f32(2.3000000000000000e+002)
    qv = f32(2.0552294154185802e-004)
    nwfa_per_kg = f32(1.6343673856000000e+010)
    dt = f32(10.0)
    perturbed = np.nextafter(pres, f32(np.inf))

    def col(values):
        return cp.asarray(np.asarray(values, f32))

    pressures = col([pres, perturbed])
    temps = col([temp, temp])
    qvs = cp.asnumpy(probe_saturation(pressures, temps)[0])
    rho = (f32(0.622) * cp.asnumpy(pressures)
           / (f32(287.04) * cp.asnumpy(temps) * (qv + f32(0.622)))).astype(f32)
    nwfa_work = np.minimum(f32(9999.0e6), (nwfa_per_kg * rho).astype(f32))

    xni = cp.asnumpy(probe_ice_koop(
        temps, col([qv, qv]), col(qvs), col(nwfa_work), col([dt, dt])))

    # WRF's own value at this state, from its tendency loop:
    #   pri_iha = 3.3434858788274056e-09  =>  xnc = pri_iha/1e-13*dtsave
    wrf_xni = f32(3.3434858788274056e-09 / 1.0e-13 * 10.0)
    assert abs(float(xni[0]) - float(wrf_xni)) / float(wrf_xni) <= 2.0e-7, (
        f"unperturbed pressure must reproduce WRF's iceKoop: got {xni[0]!r} "
        f"want {wrf_xni!r}")

    # prob_h is an integer multiple of 2^-24, so the returned crystal count
    # is BITWISE reconstructible from that integer alone.  Asserted as an
    # exact float32 identity rather than as a tolerance: a smoothed prob_h
    # (``-expm1``, or a double-precision subtraction) lands off this grid at
    # a spacing of 3.05e-5 steps, and no tolerance separates the two
    # reliably.
    quantum = np.float64(2.0) ** -24

    def quantised(steps, naero):
        return f32(f32(np.float64(steps) * quantum) * naero)

    assert f32(xni[0]) == quantised(561, nwfa_work[0]), (
        f"iceKoop returned {xni[0]!r}, which is not "
        f"float32(561*2^-24 * naero) = {quantised(561, nwfa_work[0])!r}; "
        "the float32 `1. - exp(...)` at module_mp_thompson.F:5539 has been "
        "changed and prob_h is no longer quantised to 2^-24")
    assert f32(xni[0]) != quantised(560, nwfa_work[0])
    assert f32(xni[0]) != quantised(562, nwfa_work[0])

    # And one pressure ulp is exactly ONE step down that grid.
    assert f32(xni[1]) == quantised(560, nwfa_work[1]), (
        f"one ulp of pressure gave {xni[1]!r}, not the next quantum down "
        f"{quantised(560, nwfa_work[1])!r}")
    moved = (np.float64(xni[1]) - np.float64(xni[0])) / np.float64(xni[0])
    assert abs(moved) == pytest.approx(1.0 / 561.0, rel=2.0e-3), (
        f"one ulp of pressure moved iceKoop by {moved:.6e}; the documented "
        "quantum is 1/561 = 1.78e-3")
