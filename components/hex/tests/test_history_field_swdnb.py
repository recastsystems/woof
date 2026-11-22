"""--history-field adds a Registry field to MPAS's default output stream; the S2S door asks for swdnb on the
MPAS-physics route (hex 0.4.15 fit16-rows).

THE BREAKAGE THESE PIN (verification host, 2026-11-07): a 7-day 60 km S2S run on the MPAS-physics route (the hex default)
finished its forecast and export, then its energy products stopped with
'missing field ["swdown", "swdnb", "SWDOWN"]': MPAS-A's default output list carries no downward surface shortwave.
"""
import pytest

from woof.hex import s2s_door
from woof.hex.cli import build_parser


def _layout():
    from woof.hex.mpasphys import resolve_namelist
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = '&physics\nconfig_physics_suite="convection_permitting"\n/'
    return RegistryLayout(registry_source(), resolve_namelist(text), text,
        dict(nCells=2, nEdges=3, nVertices=2, maxEdges=6, maxEdges2=12, vertexDegree=3, nVertLevels=55))


def test_added_history_field_is_in_the_output_stream_only(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    layout = _layout()
    plain = {s.name: s for s in read_streams(write_default_streams(layout, tmp_path / "a" / "streams.atmosphere"),
                                             layout)}
    added = {s.name: s for s in read_streams(write_default_streams(layout, tmp_path / "b" / "streams.atmosphere",
                                                                   add_output=("swdnb",)), layout)}
    assert "swdnb" not in plain["output"].fields
    assert set(added["output"].fields) == set(plain["output"].fields) | {"swdnb"}
    assert {k: v.fields for k, v in added.items() if k != "output"} == \
        {k: v.fields for k, v in plain.items() if k != "output"}
    with pytest.raises(ValueError, match="not a field of the Registry"):
        write_default_streams(layout, tmp_path / "c" / "streams.atmosphere", add_output=("no_such_field",))


def test_s2s_mpas_route_members_write_swdnb_and_the_arwen_route_does_not(tmp_path):
    def forecast(*extra):
        args = build_parser().parse_args(["s2s", "--row", "global-120km", "--source", "gfs", "--cycle", "2026-10-15T00",
                                          "--members", "1", "--days", "7", "--work-dir", str(tmp_path), *extra])
        return s2s_door.plan(args)["commands"]["forecast"]
    argv = forecast()
    assert argv[argv.index("--history-field") + 1] == "swdnb"
    assert "--history-field" not in forecast("--physics-row", "arwen")
