"""A row's surface input: a companion analysis the init reads with its own land mask.

The breakage: ECMWF open data (0.25 and 0.1 degree) publishes sea-ice thickness and no concentration, so every
IFS init started with no sea ice anywhere.  The rows now take the same cycle's GFS analysis ICEC (with the LAND
mask it is interpolated with) as MPAS's surface input (--sfc-met).  It must never enter the primary decode: the
init shares the primary LANDSEA by array index for soil and ice, and a GFS LAND record there would replace the
IFS mask.
"""

from __future__ import annotations

import argparse

import pytest

from woof.hex import init_sources as src


@pytest.mark.parametrize("source,member", [("ifs", None), ("ifs-0p1", None), ("aifs", None), ("ifs-ens", 7),
                                           ("aifs-ens", 0), ("aifs-ens", 12)])
@pytest.mark.parametrize("lead", [0, 72])
def test_ecmwf_rows_plan_the_same_cycle_gfs_sea_ice_analysis_at_any_start_lead(source, member, lead):
    plans = src.plan_products(src.resolve_source(src.load_table(), source), src.Cycle.parse("2026-10-28T00"),
                              member, lead)
    companions = [plan for plan in plans if plan.role == "surface-input"]
    assert len(companions) == 1
    assert companions[0].url.endswith("/gfs.20261028/00/atmos/gfs.t00z.pgrb2.0p25.f000")
    assert companions[0].index_select["records"] == [["^(LAND|ICEC)$", "^surface$"]]


def test_the_surface_input_stays_out_of_the_primary_decode_and_reaches_the_init(tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    def acquire(plans, dest, cycle, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        out = []
        for plan in plans:
            (dest / plan.filename).write_bytes(b"x")
            out.append({"role": plan.role, "file": str(dest / plan.filename), "sha256": "-"})
        return out

    def decode(row, inputs, cycle, out_dir, writer):
        seen["primary"] = list(inputs)
        met = out_dir / "MET:2026-10-28_00"
        met.write_bytes(b"met")
        return {"met": str(met), "met_sha256": src._sha256_file(met)}

    def decode_surface_input(row, inputs, analysis, out_dir, writer):
        seen["surface"] = list(inputs)
        met = out_dir / "SFC:2026-10-28_00"
        met.write_bytes(b"sfc")
        return {"met": str(met), "met_sha256": src._sha256_file(met)}

    def run_init(argv):
        seen["init"] = list(argv)
        return 0

    for name, value in (("acquire", acquire), ("decode", decode), ("decode_surface_input", decode_surface_input),
                        ("run_init", run_init), ("resolve_writer", lambda explicit: tmp_path / "writer")):
        monkeypatch.setattr(src, name, value)
    arguments = argparse.Namespace(
        table=None, list=False, source="ifs-0p1", cycle="2026-10-28T00", member=None, lead_hours=0,
        out_dir=tmp_path / "out", grid=tmp_path / "g.nc", static=tmp_path / "s.nc", vertical_spec=tmp_path / "v.json",
        staged=[], writer=None, remapper=None, jobs=1, stop_after="init", order_map=None, hex_order=None,
        case_template=None, keep_unordered=False)
    assert src.run_init_source_command(arguments) == 0
    assert seen["primary"] and not [name for name in seen["primary"] if "gfs" in name]
    assert [name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for name in seen["surface"]] == [
        "gfs.t00z.pgrb2.0p25.f000.land-icec"]
    init = seen["init"]
    assert init[init.index("--sfc-met") + 1].endswith("SFC:2026-10-28_00")
