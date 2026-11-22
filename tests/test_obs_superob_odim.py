"""Superobbing an ODIM pack: the clear-air regime, and the silent-empty refusal.

The cross-lane proof, a real KNMI Den Helder volume through the real
``rw_odim`` writer, through this stage, onto a real wrfout, is recorded in
the ODIM front-door lane record. It needs a 26 MB volume off a
24-hour feed and a 700 MB wrfout, so it cannot live in the suite. What lives
here is what would otherwise be exercised only by that file.

The refusal in :func:`test_a_volume_that_yields_no_gate_is_refused` is not
hypothetical and is why this file exists. ``rw_odim`` wrote ODIM's own
``DBZH``/``VRADH`` moment names where :mod:`woof.obs.superob` filters for
``REF``/``VEL``; every one of 8.5 million gates was skipped, ``sweeps_used``
still counted fifteen, the observation file wrote successfully, and its record
was entirely zeros. Nothing failed. There was simply nothing in it.

The grid here is a stub rather than a real :class:`~woof.obs.target_grid.
TargetGrid`, because a TargetGrid is built from a wrfout and none of these
tests is about geometry. Everything they assert, the census, the vocabulary,
the refusal, is settled per gate, before or independently of where the gate
lands.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from woof.obs.superob import (CLEAR_AIR_SOURCE, CLEAR_AIR_SOURCE_CENSOR,
                               CLEAR_AIR_SOURCE_ODIM, CLEAR_AIR_SOURCES,
                               SuperobParams, superob_volume)
from woof.obs.sweeps import (SWEEPS_SCHEMA_CENSOR, SWEEPS_SCHEMA_ODIM, Censor,
                              read_sweep_pack)


def _write_pack(path: Path, schema: str, *,
                products: dict[str, list[int]], measured=-10.0, unit="dBZ") -> Path:
    """One sweep, one radial, ``products`` mapping moment name to censor codes.

    The only hand-built pack in this file. It exists to put codes in the plane
    that no European volume this lane could fetch happens to carry today:
    ``SENTINEL_AMBIGUOUS`` is a Finnish ``VRADH`` state, and ``NODATA`` did not
    appear in the Dutch or German volumes measured.
    """

    payload = b""
    arrays: dict[str, dict] = {}

    def push(name: str, array: np.ndarray) -> str:
        nonlocal payload
        raw = array.tobytes()
        arrays[name] = {
            "dtype": "<f4" if array.dtype == np.float32 else "|u1",
            "shape": list(array.shape), "offset": len(payload),
            "bytes": len(raw)}
        payload += raw
        return name

    moments = []
    for index, (product, codes) in enumerate(products.items()):
        censor = np.array([codes], dtype=np.uint8)
        # Only a MEASURED gate carries a number; every censored code is NaN,
        # which is what the real writer does.
        data = np.where(censor == Censor.MEASURED, measured, np.nan).astype("<f4")
        moments.append({
            "product": product, "unit": unit, "gate_count": len(codes),
            "first_gate_range_m": 125.0, "gate_size_m": 250.0,
            "array": push(f"v{index}", data),
            "censor_array": push(f"c{index}", censor)})

    azimuth = push("az", np.array([0.0], dtype="<f4"))
    elevation = push("el", np.array([0.5], dtype="<f4"))
    meta = {
        "schema": schema, "status": "READY",
        "site": {"id": "site", "name": "Site", "lat_deg": 52.95,
                 "lon_deg": 4.79, "alt_m": 55.0, "source": "odim:/where"},
        "volume": {"file": "v.h5", "bytes": 1, "sha256": "0" * 64,
                   "station_id": "site",
                   "valid_time": "2026-08-14T13:00:00+00:00",
                   "volume_date": 0, "volume_time_ms": 0},
        "params": {"moments": [], "max_range_km": 200.0,
                   "max_elevation_deg": 25.0, "censor_flags": True},
        "sweeps": [{
            "sweep_index": 0, "elevation_number": 0,
            "elevation_angle_deg": 0.5, "nyquist_velocity_ms": 8.0,
            "nyquist_granularity": "sweep",
            "start_status": 0, "end_status": 0, "cut_sector": 0,
            "complete": True,
            "radial_count": 1, "azimuth_array": azimuth,
            "elevation_array": elevation, "moments": moments}],
        "arrays": arrays, "payload_bytes": len(payload),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "dropped_sweeps": 0, "dropped_moments": 0, "trimmed_gates": 0}

    blob = json.dumps(meta).encode()
    header = (b"GPWMRDR1" + (1).to_bytes(4, "little")
              + len(blob).to_bytes(4, "little")
              + len(payload).to_bytes(8, "little"))
    path.write_bytes(header.ljust(64, b"\0") + blob + payload)
    return path


@pytest.fixture(autouse=True)
def _numpy_reference(monkeypatch):
    """These tests place gates on duck-typed test grids.

    A test double's placement methods are Python, and the Rust superob
    places gates with a TargetGrid's projection and column interfaces, so
    it refuses these objects by name.  The census semantics are pinned
    here on the numpy reference, opted into explicitly, and the Rust
    route's census is held to that reference on a real TargetGrid by
    tests/test_obs_superob_rust_parity.py.
    """
    monkeypatch.setenv("WOOF_SUPEROB_PYTHON", "1")


class _OffGrid:
    """A grid every gate misses. Reaching it at all is the point."""

    nz, ny, nx = 1, 3, 3
    # The accumulators now cover only the cells the radar can reach, and
    # that window is measured against the grid's own mass-point
    # coordinates.  Sitting these on the pack's antenna keeps the window
    # the whole 3x3 -- the dense case this fixture has always exercised --
    # so what the test measures is unchanged; putting them elsewhere would
    # shrink the window and quietly change the subject.
    dx_m = dy_m = 3000.0
    lat = np.full((3, 3), 52.95)
    lon = np.full((3, 3), 4.79)

    def mass_index(self, lat, lon):
        return np.full(lat.shape, -5.0), np.full(lat.shape, -5.0)

    def inside(self, i, j):
        return np.zeros(i.shape, dtype=bool)

    def level_index(self, i, j, height):
        return np.full(i.shape, -1)


def _params() -> SuperobParams:
    return SuperobParams(max_range_km=200.0, max_elevation_deg=25.0)


# ------------------------------------------------- the allow-list itself

def test_the_odim_regime_is_on_the_allow_list_and_is_its_own_value():
    """Adding a regime must be a deliberate edit, and a distinguishable one."""

    assert CLEAR_AIR_SOURCE_ODIM in CLEAR_AIR_SOURCES
    assert CLEAR_AIR_SOURCE_ODIM not in (CLEAR_AIR_SOURCE,
                                         CLEAR_AIR_SOURCE_CENSOR)
    assert len(set(CLEAR_AIR_SOURCES)) == len(CLEAR_AIR_SOURCES)


def test_every_radar_consumer_reads_the_regime_the_odim_gridder_writes():
    """The superob writes ``undetect_and_finite_below_floor`` for an ODIM
    pack, but the reflectivity adapter and the heating builder each held an
    allow-list without it, so a European volume gridded with its clear air
    was refused by the filter, the precipitation analysis and the heating
    builder alike and the documented European route ended at the file.  The
    adapter went with the local data assimilation in gpuwm 3.0.3; the heating
    builder reads exactly the regimes the superob writes."""
    from woof.da import radar_tten

    assert CLEAR_AIR_SOURCE_ODIM in radar_tten._CLEAR_AIR_SOURCES
    assert set(radar_tten._CLEAR_AIR_SOURCES) == set(CLEAR_AIR_SOURCES)


def test_the_regime_is_chosen_by_the_pack_not_by_the_caller(tmp_path):
    """The flag says "use the decoder's codes"; the file says whose codes."""

    codes = [Censor.MEASURED, Censor.BELOW_THRESHOLD]
    odim = read_sweep_pack(_write_pack(
        tmp_path / "v3.pack", SWEEPS_SCHEMA_ODIM, products={"REF": codes}))
    nexrad = read_sweep_pack(_write_pack(
        tmp_path / "v2.pack", SWEEPS_SCHEMA_CENSOR, products={"REF": codes}))

    assert superob_volume(odim, _OffGrid(), params=_params(),
                          clear_air_from_censor=True
                          ).clear_air_source == CLEAR_AIR_SOURCE_ODIM
    assert superob_volume(nexrad, _OffGrid(), params=_params(),
                          clear_air_from_censor=True
                          ).clear_air_source == CLEAR_AIR_SOURCE_CENSOR
    # The measurement-only regime is not a property of the pack.
    assert superob_volume(odim, _OffGrid(), params=_params()
                          ).clear_air_source == CLEAR_AIR_SOURCE


def test_the_census_closes_over_an_odim_pack(tmp_path):
    """Every reflectivity gate is accounted for under exactly one code.

    Without the two ODIM-only counters the census fell short of the gate count
    by exactly the gates that were refused, which is how a refusal becomes
    invisible: the numbers still look like numbers.
    """

    codes = [Censor.MEASURED, Censor.BELOW_THRESHOLD, Censor.NODATA,
             Censor.SENTINEL_AMBIGUOUS, Censor.NOT_COLLECTED]
    volume = read_sweep_pack(_write_pack(
        tmp_path / "v3.pack", SWEEPS_SCHEMA_ODIM, products={"REF": codes}))
    census = superob_volume(volume, _OffGrid(), params=_params(),
                            clear_air_from_censor=True).counts.censor

    assert census.reflectivity_measured == 1
    assert census.reflectivity_below_threshold == 1
    assert census.reflectivity_nodata == 1
    assert census.reflectivity_sentinel_ambiguous == 1
    assert census.reflectivity_not_collected == 1
    assert census.reflectivity_range_folded == 0     # ODIM never mints it
    assert census.sentinel_ambiguous_gates_refused == 1

    accounted = (census.reflectivity_measured
                 + census.reflectivity_below_threshold
                 + census.reflectivity_range_folded
                 + census.reflectivity_not_collected
                 + census.reflectivity_nodata
                 + census.reflectivity_sentinel_ambiguous)
    assert accounted == len(codes), "the census must close over every gate"


def test_the_odim_only_counters_stay_out_of_a_nexrad_census(tmp_path):
    """Zero and absent are different statements; the schema decides which.

    A ``v2`` pack cannot mint codes 4 or 5, so reporting them as zero would
    claim "we looked and there were none". Keeping the keys out is also what
    holds every committed ``v2`` observation digest byte-identical.
    """

    volume = read_sweep_pack(_write_pack(
        tmp_path / "v2.pack", SWEEPS_SCHEMA_CENSOR,
        products={"REF": [Censor.MEASURED, Censor.BELOW_THRESHOLD]}))
    counts = superob_volume(volume, _OffGrid(), params=_params(),
                            clear_air_from_censor=True).counts

    assert counts.censor.reflectivity_nodata is None
    assert counts.censor.sentinel_ambiguous_gates_refused is None
    payload = counts.to_payload()["censor"]
    assert "reflectivity_nodata" not in payload
    assert "reflectivity_sentinel_ambiguous" not in payload
    assert "sentinel_ambiguous_gates_refused" not in payload
    assert payload["reflectivity_below_threshold"] == 1


def test_no_admission_path_can_reach_the_two_refused_codes():
    """Clear air is established by equality with ONE code, structurally.

    The behavioural half is above -- the census proves 4 and 5 were seen and
    that ``clear_air_gates_admitted`` did not grow for them. This half pins the
    mechanism, because a future edit to "not an echo" would pass every count
    assertion on a volume that happens to carry neither code.
    """

    import woof.obs.superob as module

    body = Path(module.__file__).read_text(
        encoding="utf-8").partition("def superob_volume(")[2]
    assert "clear_flag = codes == Censor.BELOW_THRESHOLD" in body
    for forbidden in ("clear_flag = codes == Censor.NODATA",
                      "clear_flag = codes == Censor.SENTINEL_AMBIGUOUS",
                      "clear_flag = codes != Censor.MEASURED",
                      "clear_flag = np.isin(codes"):
        assert forbidden not in body, (
            f"{forbidden!r} would admit a censored state as clear air")


def test_admitted_clear_air_counts_only_the_below_threshold_gate(tmp_path):
    """The behavioural half of the same claim, on a gate of every code."""

    codes = [Censor.MEASURED, Censor.BELOW_THRESHOLD, Censor.NODATA,
             Censor.SENTINEL_AMBIGUOUS, Censor.NOT_COLLECTED]

    class _OnGrid(_OffGrid):
        def mass_index(self, lat, lon):
            return np.ones(lat.shape), np.ones(lat.shape)

        def inside(self, i, j):
            return np.ones(i.shape, dtype=bool)

        def level_index(self, i, j, height):
            return np.zeros(i.shape, dtype=np.intp)

    volume = read_sweep_pack(_write_pack(
        tmp_path / "v3.pack", SWEEPS_SCHEMA_ODIM, products={"REF": codes}))
    census = superob_volume(volume, _OnGrid(), params=_params(),
                            clear_air_from_censor=True).counts.censor
    assert census.clear_air_gates_admitted == 1


# ------------------------------------------------ the silent-empty refusal

def test_a_volume_that_yields_no_gate_is_refused(tmp_path):
    """The instrument in the direction that matters: it must fire.

    A pack whose moments no consumer recognises used to produce a well-formed,
    entirely empty observation file. Now it refuses, and names both
    vocabularies, which is the whole diagnosis.
    """

    volume = read_sweep_pack(_write_pack(
        tmp_path / "alien.pack", SWEEPS_SCHEMA_ODIM,
        products={"DBZH": [Censor.MEASURED],
                  "VRADH": [Censor.MEASURED]}))
    assert set(volume.sweeps[0].moments) == {"DBZH", "VRADH"}

    with pytest.raises(ValueError) as raised:
        superob_volume(volume, _OffGrid(), params=_params())
    message = str(raised.value)
    assert "not one gate was considered" in message
    assert "DBZH" in message and "VRADH" in message
    assert "REF" in message and "VEL" in message


def test_the_refusal_is_silent_when_the_gates_were_read_and_then_dropped(
        tmp_path):
    """The other direction: gates read and legitimately discarded must pass.

    The refusal is keyed on "sweeps used and no gate considered", so a volume
    whose every gate falls outside the domain -- an ordinary outcome for a
    radar near a domain edge -- must reach the end normally.
    """

    volume = read_sweep_pack(_write_pack(
        tmp_path / "canonical.pack", SWEEPS_SCHEMA_ODIM,
        products={"REF": [Censor.MEASURED], "VEL": [Censor.MEASURED]}))
    counts = superob_volume(volume, _OffGrid(), params=_params()).counts
    assert counts.gates_considered == 2
    assert counts.gates_out_of_grid == 2
