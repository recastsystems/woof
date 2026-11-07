"""RUC runs from every source WOOF can drive, and the matrix is pinned here.

The claim this file holds is the arbitrary acceptance test applied to one
scheme: **adding a future model must be metadata/table work**.  For the RUC
LSM's soil ingest that means every source in the registry is either

* RUNS -- its declared soil geometry selects a row of
  :data:`woof.ingest.soil_contract.RUC_REMAP_POLICIES`, that row selects an
  arm of WRF's own ``init_soil_3_real``, and a real nine-level float32
  column comes out; or
* a NAMED physical impossibility -- the source does not publish enough soil
  for a 3 m column to exist, and the refusal cites WRF's own line for what
  it would otherwise have to invent.

There is no third answer, and "this source publishes it a different way" is
not one of the two.  The whole matrix is asserted below rather than
described, so a future source cannot be added into a silent refusal: an
adapter with a mapped soil contract that is neither RUNS nor one of the
pinned impossibilities fails this file.

The oracle half is the other guard.  The mapped LAYER route is not a second
implementation of the layer arm -- it is the SAME
``remap_soil_to_ruc_levels`` call the native ERA5/GFS field-name dispatch
makes, and :func:`test_the_mapped_layer_route_reproduces_wrfs_own_numbers`
proves that by replaying WRF v4.6.1's own ``init_soil_3_real`` rows through
the mapped contract path and requiring 0 ULP.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from woof.ingest.ruc_soil import _source_soil_profiles, remap_soil_to_ruc_levels
from woof.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                        MAPPED_SOIL_TEMPERATURE,
                                        RUC_REMAP_POLICIES,
                                        ruc_soil_remap_policy)
from woof.mapped_composition import load_composition
from woof.source_adapters import source_adapters
from woof.source_authorities import packaged_authorities

ROOT = Path(__file__).parents[1]
ORACLE = ROOT / "woof" / "data" / "ruc" / "oracle" / "soil_ingest.csv"
NSOIL = 9

#: Every source whose mapped composition declares a soil contract, and the
#: RUC policy row its declaration selects.  This is the matrix.  A new
#: packaged profile with soil MUST appear here, and it must appear with a
#: policy name rather than a refusal unless its impossibility is named in
#: :data:`NAMED_IMPOSSIBILITIES` below.
EXPECTED_POLICY = {
    "20crv3": "layer_midpoint_samples",
    # 3.0.0 hurricane line (fold/tc-arwen-3.0): the HAFS parents and the GFS
    # pressure-level source the storm merge reads, on the same RUC policy.
    "gfs-pressure": "layer_midpoint_samples",
    "hafs-a": "layer_midpoint_samples",
    "hafs-b": "layer_midpoint_samples",
    "20crv3-cf": "layer_midpoint_samples",
    "aifs": "layer_midpoint_samples",
    "aigefs": "layer_midpoint_samples",
    "aigfs": "layer_midpoint_samples",
    "ecmwf-ens": "layer_midpoint_samples",
    "ecmwf-open-data": "layer_midpoint_samples",
    # E-SRC (src/any-init-sources 94841c57ed): IFS HRES open data at 0.1
    # degree, decoded by the 0.25-degree oper profile's soil layers.
    "ecmwf-open-data-0p1": "layer_midpoint_samples",
    "era5-l137": "layer_midpoint_samples",
    # The admitted GEM profiles declare the authenticated deep-soil donor.
    "gem-gdps": "layer_midpoint_samples",
    "gem-hrdps": "layer_midpoint_samples",
    "gem-rdps": "layer_midpoint_samples",
    "gdas": "layer_midpoint_samples",
    "gefs": "layer_midpoint_samples",
    # development branch (6a3b8fd50bf): the indexed GEFS retrospective
    # members carry the GFS Noah four-layer soil, the layer arm.
    "gefs-reforecast": "layer_midpoint_samples",
    "gefs-reforecast-weekly": "layer_midpoint_samples",
    "gefs-v11": "layer_midpoint_samples",
    "gefs-v11-flat": "layer_midpoint_samples",
    # 2.9.2: a woof hex run as an ArWen parent. Its history carries MPAS
    # Noah's four layers (0-10, 10-40, 40-100, 100-200 cm), the GFS ladder.
    "hex-global": "layer_midpoint_samples",
    "hrrr-native": "node_point_samples",
    "hrrr-prs": "node_point_samples",
    "icon-d2": "node_point_samples",
    "icon-eu": "node_point_samples",
    "icon-global": "node_point_samples",
    "rap": "node_point_samples",
    "rap-native": "node_point_samples",
    "rrfs": "node_point_samples",
    # S9 (development branch): the RRFS ensemble member's soil comes from its
    # declared deterministic RRFS companion, the same RUC nodes as rrfs.
    "rrfs-ens": "node_point_samples",
}

# Parameterized models reuse the certified land-surface profile. Each new
# profile still needs an independently pinned RUC policy, while another model
# row on an existing profile needs no duplicate source-name list.
MODEL_PROFILE_POLICY = {
    "earth2-pl13-q-netcdf-v1": "layer_midpoint_samples",
    "earth2-pl13-rh-netcdf-v1": "layer_midpoint_samples",
    # StormCast rows borrow the same-cycle HRRR native soil (RUC nodes),
    # so they carry HRRR's node-point policy.
    "earth2-stormcast-hrrr-netcdf-v1": "node_point_samples",
    "earth2-stormcast-conus-hrrr-netcdf-v1": "node_point_samples",
}
EXPECTED_POLICY.update({row.source_id: MODEL_PROFILE_POLICY[row.packaged_profile]
    for row in source_adapters()
    if row.model_table and row.packaged_profile in MODEL_PROFILE_POLICY})

# The deep-soil donor now supplies the GEM column; no declared source is impossible.
NAMED_IMPOSSIBILITIES = {}

#: The integer scale each source's ladder is carried in.  100 is WRF's own
#: centimetres, and every source metgrid can express keeps it -- which is
#: what makes the oracle bit-identity below a statement about all of them.
EXPECTED_SCALE = {
    # The DWD rows carry TERRA's node depths, whose shallowest is 0.005 m:
    # centimetres cannot express it, so the ladder rides in millimetres.
    "icon-d2": 1000,
    "icon-eu": 1000,
    "icon-global": 1000,
}


def _soil_contract(profile_id):
    authorities = packaged_authorities(profile_id)
    composition = load_composition(
        authorities["composition"], authorities["mapping"])
    return composition.get("soil_layers")


def _sources_with_soil():
    for adapter in sorted(source_adapters(), key=lambda a: a.source_id):
        if adapter.packaged_profile is None:
            continue
        try:
            contract = _soil_contract(adapter.packaged_profile)
        except ValueError:
            # A PENDING profile refuses to compose at all; it has no soil
            # to classify and its own tests hold that refusal.
            continue
        if contract is None:
            continue
        yield adapter.source_id, contract


def test_the_matrix_is_exactly_runs_or_a_named_impossibility():
    """No cell is blank, and no cell is a style objection."""

    seen: dict[str, str] = {}
    for source_id, contract in _sources_with_soil():
        try:
            policy = ruc_soil_remap_policy(contract)
        except ValueError as refusal:
            assert source_id in NAMED_IMPOSSIBILITIES, (
                f"{source_id} is refused for RUC and is not a named physical "
                f"impossibility: {refusal}.  A source may be refused only "
                "where the soil it publishes cannot constrain RUC's column, "
                "never because it publishes in a different geometry -- both "
                "geometries have a row in RUC_REMAP_POLICIES")
            assert NAMED_IMPOSSIBILITIES[source_id] in str(refusal)
            assert "init_soil_3_real" in str(refusal), (
                f"{source_id}'s refusal must cite the WRF line it would "
                "otherwise have to invent a value for")
            seen[source_id] = "REFUSED"
            continue
        assert policy["policy"] in RUC_REMAP_POLICIES
        seen[source_id] = policy["policy"]

    assert seen == dict(
        EXPECTED_POLICY,
        **{name: "REFUSED" for name in NAMED_IMPOSSIBILITIES}), (
        "the RUC source matrix moved.  Every registry source with a mapped "
        "soil contract must be pinned here as RUNS-with-a-policy or as a "
        "named impossibility; a new source silently landing in either bucket "
        "is the regression this test exists to catch")


@pytest.mark.parametrize(
    "source_id", sorted(EXPECTED_POLICY), ids=sorted(EXPECTED_POLICY))
def test_every_running_source_produces_a_real_nine_level_column(source_id):
    """RUNS means numbers came out, not that a validator said yes."""

    contract = dict(_sources_with_soil())[source_id]
    policy = ruc_soil_remap_policy(contract)
    assert policy["policy"] == EXPECTED_POLICY[source_id]
    assert policy["depth_scale_per_m"] == EXPECTED_SCALE.get(source_id, 100)
    assert policy["arm"] in {"flag_soil_levels", "flag_soil_layers"}

    count = len(policy["sample_depths"])
    shape = (3, 4)
    rng = np.random.default_rng(19740403)
    temperature = (np.float32(280.0)
                   + rng.random((count,) + shape).astype(np.float32)
                   * np.float32(20.0))
    # Monotone with depth so the source profile is a plausible soil column
    # rather than noise; the vertical remap is linear either way.
    moisture = np.linspace(
        0.12, 0.34, count, dtype=np.float32)[:, None, None] * np.ones(
            shape, dtype=np.float32)
    skin = np.full(shape, np.float32(288.0), dtype=np.float32)
    deep = np.full(shape, np.float32(285.0), dtype=np.float32)
    landmask = np.ones(shape, dtype=np.float32)
    landmask[0, 0] = np.float32(0.0)

    columns = remap_soil_to_ruc_levels(
        source_temperature=temperature,
        source_moisture=moisture,
        source_levels_cm=np.array(policy["sample_depths"], dtype=np.int64),
        source_geometry=str(policy["geometry"]),
        skin_temperature=skin,
        deep_temperature=deep,
        landmask=landmask,
        num_soil_layers=NSOIL,
        source_depth_scale=int(policy["depth_scale_per_m"]),
    )

    assert columns.soil_temperature.shape == (NSOIL,) + shape
    assert columns.soil_moisture.shape == (NSOIL,) + shape
    assert columns.soil_temperature.dtype == np.float32
    assert columns.soil_moisture.dtype == np.float32
    assert np.isfinite(columns.soil_temperature).all()
    assert np.isfinite(columns.soil_moisture).all()
    # The land columns carry the source's own profile, bounded by it; the
    # one water column is WRF's :2131-2157 fill.
    land = np.asarray(columns.soil_temperature)[:, landmask > 0.5]
    assert float(land.min()) >= 279.0
    assert float(land.max()) <= 301.0
    # RUC's own level table came back, whatever the source's ladder was.
    assert [round(float(v), 5) for v in columns.level_depths] == [
        0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0]


def _oracle_rows():
    with ORACLE.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


def _ulp(got, want):
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    a = got.view(np.int32).astype(np.int64)
    b = want.view(np.int32).astype(np.int64)
    a = np.where(a < 0, np.int64(np.iinfo(np.int32).min) - a, a)
    b = np.where(b < 0, np.int64(np.iinfo(np.int32).min) - b, b)
    return np.abs(a - b)


#: The two oracle experiments whose source level sets are exactly what a
#: mapped LAYER contract declares: ERA5/IFS 0-7/7-28/28-100/100-289 cm
#: (midpoints 3/17/64/194) and the GFS-family 0-10/10-40/40-100/100-200 cm
#: (midpoints 5/25/70/150).  ``ecmwf-open-data`` and ``gdas`` are the
#: shipped compositions that declare them.
LAYER_ORACLE_CASES = (
    ("era5_layers_noadj_nosst", "ecmwf-open-data", (3, 17, 64, 194)),
    ("noah_layers_adj_nosst", "gdas", (5, 25, 70, 150)),
)


@pytest.mark.parametrize(
    "experiment,source_id,levels", LAYER_ORACLE_CASES,
    ids=[case[0] for case in LAYER_ORACLE_CASES])
def test_the_mapped_layer_route_reproduces_wrfs_own_numbers(
        experiment, source_id, levels):
    """WRF v4.6.1's own rows, replayed through the MAPPED contract path.

    This is the field-for-field oracle for the route this lane opened.  The
    rows are ``woof/data/ruc/oracle/soil_ingest.csv``, produced by
    ``tools/ruc_soil_ingest_wrf461_oracle`` driving byte-unmodified
    ``share/module_soil_pre.F`` at the pinned WRF commit; the level sets
    are the ones the shipped ECMWF and GDAS compositions declare.  What is
    measured here is not the arm -- ``validate_ruc_soil_ingest_oracle.py``
    already measures that -- but that a mapped contract reaches it: the
    sample depths the contract produces, and the profiles the mapped
    ``RW_SOIL_*`` arrays carry into it, land on WRF's numbers at 0 ULP.

    ``noah_layers_adj_nosst`` column 10 is the row WRF answers with
    ``MAX(0.02, NaN)``; it is refused rather than reproduced, for the
    reason ``woof/ingest/ruc_soil.py`` gives, and is skipped here.
    """

    contract = dict(_sources_with_soil())[source_id]
    policy = ruc_soil_remap_policy(contract)
    assert policy["policy"] == "layer_midpoint_samples"
    assert tuple(policy["sample_depths"]) == levels, (
        f"{source_id}'s declared layer bounds no longer sample at the "
        f"oracle's {levels} cm; the comparison below would be against a "
        "different source geometry than WRF ran")

    rows = [row for row in _oracle_rows() if row["experiment"] == experiment]
    assert rows, f"{experiment} has no rows in {ORACLE}"

    worst_t = worst_m = worst_tsk = 0
    compared = 0
    for row in rows:
        nlev = int(row["nlev"])
        assert nlev == len(levels)
        if (experiment, int(row["col"])) == ("noah_layers_adj_nosst", 10):
            continue
        temperature = np.array(
            [np.float32(row[f"st_src_{k}"]) for k in range(1, nlev + 1)],
            dtype=np.float32)[:, None]
        moisture = np.array(
            [np.float32(row[f"sm_src_{k}"]) for k in range(1, nlev + 1)],
            dtype=np.float32)[:, None]

        # THE POINT: the source profiles enter through the mapped
        # RW_SOIL_TEMPERATURE / RW_SOIL_MOISTURE arrays and the shipped
        # contract, not through ERA5's ST000007 field names.
        got_t, got_m, got_levels, geometry, scale = _source_soil_profiles(
            {MAPPED_SOIL_TEMPERATURE: temperature,
             MAPPED_SOIL_MOISTURE: moisture}, contract)
        assert geometry == "layers"
        assert scale == 100
        assert list(got_levels) == list(levels)

        columns = remap_soil_to_ruc_levels(
            source_temperature=got_t,
            source_moisture=got_m,
            source_levels_cm=got_levels,
            source_geometry=geometry,
            skin_temperature=np.array(
                [np.float32(row["tsk_in"])], dtype=np.float32),
            deep_temperature=np.array(
                [np.float32(row["tmn"])], dtype=np.float32),
            landmask=np.array(
                [np.float32(row["landmask"])], dtype=np.float32),
            sea_surface_temperature=(
                np.array([np.float32(row["sst"])], dtype=np.float32)
                if row["flag_sst"] == "1" else None),
            num_soil_layers=NSOIL,
            moisture_adjustment=row["flag_sm_adj"] == "1",
            source_depth_scale=scale,
        )

        want_t = np.array(
            [np.float32(row[f"tslb_{k}"]) for k in range(1, NSOIL + 1)],
            dtype=np.float32)
        want_m = np.array(
            [np.float32(row[f"smois_{k}"]) for k in range(1, NSOIL + 1)],
            dtype=np.float32)
        worst_t = max(
            worst_t, int(_ulp(columns.soil_temperature[:, 0], want_t).max()))
        worst_m = max(
            worst_m, int(_ulp(columns.soil_moisture[:, 0], want_m).max()))
        worst_tsk = max(
            worst_tsk,
            int(_ulp(columns.skin_temperature[0],
                     np.float32(row["tsk_out"])).max()))
        compared += 1

    assert compared >= 19
    assert (worst_t, worst_m, worst_tsk) == (0, 0, 0), (
        f"{experiment} through the mapped contract: TSLB {worst_t} ULP, "
        f"SMOIS {worst_m} ULP, TSK {worst_tsk} ULP from WRF v4.6.1's own "
        "init_soil_3_real")


def test_the_policy_table_is_the_dispatch_and_names_no_model():
    """A future model is a row, not a branch.

    The arbitrary acceptance test in prose is "a per-model adapter file
    fails".  Here it is as a grep: the RUC ingest may not contain a
    producer's name in its dispatch, and the two policies must be the two
    geometries the contract language can declare -- so a new producer is
    covered by whichever it already declares.
    """

    assert set(RUC_REMAP_POLICIES) == {
        "node_point_samples", "layer_midpoint_samples"}
    assert {row["declaration"] for row in RUC_REMAP_POLICIES.values()} == {
        "source_nodes", "source_layers"}
    assert {row["arm"] for row in RUC_REMAP_POLICIES.values()} == {
        "flag_soil_levels", "flag_soil_layers"}

    source = (ROOT / "woof" / "ingest" / "ruc_soil.py").read_text(
        encoding="utf-8")
    mapped_arm = source[
        source.index("The declarative mapped-source arm."):
        source.index('if "SOILT" in fields and "SOILW" in fields:')]
    for producer in ("ICON", "icon", "ECMWF", "ecmwf", "AIFS", "aifs",
                     "GEFS", "gefs", "20CR", "RRFS", "rrfs", "GDAS", "gdas"):
        assert producer not in mapped_arm, (
            f"the mapped RUC arm names {producer!r}; adding a model must be "
            "a composition document plus a RUC_REMAP_POLICIES row, never a "
            "branch here")
