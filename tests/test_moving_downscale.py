"""woof downscale --follow: the plan a moving step-down runs on.

The window cut and the warm start were proven on a real storm run (Milton,
2024-10-09 12Z); these
pin the arithmetic they rest on.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from woof.downscale import _centered_placement
from woof.moving_downscale import (
    Footprint, MovingDownscaleRefusal, _start_offset, child_targets,
    plan_segments)


@pytest.mark.parametrize("span", range(10, 82))
def test_start_offset_is_the_doors_own_centring(span):
    # The point handed to each segment is the child's centre cell, so the
    # door's centring must put the child's first cell exactly at the plan's.
    ratio = 3
    parent = {"nx": 400, "ny": 400}
    for i0 in (span, span + 1, 150):
        placed = _centered_placement(parent, j0=i0, i0=i0, ratio=ratio,
                                     child_nx=span * ratio,
                                     child_ny=span * ratio)
        assert placed.i_parent_start - 1 == i0 - _start_offset(span)


def _moving(n=9, step=(4, 2), every=4, nx=200, ny=200):
    start = datetime(2024, 10, 9, 13)
    rows = []
    for k in range(n):
        moves = k // every
        rows.append(Footprint(path=Path(f"wrfout_d02_{k}"),
                              valid_time=start + timedelta(minutes=15 * k),
                              origin_i=moves * step[0], origin_j=moves * step[1],
                              nx=nx, ny=ny))
    return rows


class _Lattice:
    """A lattice whose mass index is the degree (1-based)."""

    def latlon_to_ij(self, lat, lon):
        return float(lon) + 1.0, float(lat) + 1.0


def test_a_child_following_its_parent_moves_with_it_and_stays_inside():
    frames = _moving()
    targets, basis = child_targets(frames, _Lattice(), mode="parent",
                                   point=(100.0, 100.0))
    assert basis == "its place in the moving parent"
    segments = plan_segments(frames, targets, span_i=60, span_j=60,
                             start=frames[0].valid_time, seconds=7200,
                             every=3600, edge=6)
    assert len(segments) == 2
    first, second = segments
    # One parent move (4, 2) inside the second hour, and the child follows
    # the mean place of the parent over its stretch.
    assert (second.child_i - first.child_i, second.child_j - first.child_j) in (
        (4, 2), (3, 2), (4, 1), (3, 1))
    for segment in segments:
        for frame in segment.frames:
            assert segment.window_i >= frame.origin_i + 6
            assert segment.window_i + segment.window_nx <= frame.origin_i + frame.nx - 6
            assert segment.window_j >= frame.origin_j + 6
            assert segment.window_j + segment.window_ny <= frame.origin_j + frame.ny - 6


def test_a_fixed_child_at_the_parent_edge_is_clamped_inside_every_frame():
    frames = _moving(step=(20, 0))
    targets, _ = child_targets(frames, _Lattice(), mode="fixed",
                               point=(100.0, 30.0))
    segments = plan_segments(frames, targets, span_i=60, span_j=60,
                             start=frames[0].valid_time, seconds=7200,
                             every=3600, edge=6)
    assert segments[1].clamped
    for frame in segments[1].frames:
        assert segments[1].window_i >= frame.origin_i + 6


def test_a_child_too_big_for_the_moving_parent_is_refused_by_name():
    frames = _moving(step=(40, 0))
    targets, _ = child_targets(frames, _Lattice(), mode="parent",
                               point=(100.0, 100.0))
    with pytest.raises(MovingDownscaleRefusal, match="never leaves room"):
        plan_segments(frames, targets, span_i=150, span_j=60,
                      start=frames[0].valid_time, seconds=7200, every=3600,
                      edge=6)


def test_segments_end_on_frames():
    frames = _moving()
    targets, _ = child_targets(frames, _Lattice(), mode="parent",
                               point=(100.0, 100.0))
    with pytest.raises(MovingDownscaleRefusal, match="frame"):
        plan_segments(frames, targets, span_i=60, span_j=60,
                      start=frames[0].valid_time, seconds=7200, every=1000,
                      edge=6)


def test_strip_rain_is_interpolated_from_the_parent_cells():
    import numpy as np

    from woof.moving_downscale import _bilinear_to_child

    # A field linear in the parent index is reproduced at every child centre.
    span, ratio = 4, 3
    jj, ii = np.meshgrid(np.arange(-1, span + 1), np.arange(-1, span + 1),
                         indexing="ij")
    field = 2.0 * ii + 5.0 * jj
    child = _bilinear_to_child(field.astype(float), ratio=ratio, span_i=span,
                               span_j=span)
    x = (np.arange(span * ratio) + 0.5) / ratio - 0.5
    assert np.allclose(child, 2.0 * x[np.newaxis, :] + 5.0 * x[:, np.newaxis])


def test_the_earlier_childs_specified_ring_takes_the_parents_rain():
    from woof.core.nest_relocation import Placement, plan_relocation
    from woof.offline_child_warm import _strip_mask

    plan = plan_relocation(
        placement_from=Placement(grid_id=3, i_parent_start=10, j_parent_start=10),
        placement_to=Placement(grid_id=3, i_parent_start=13, j_parent_start=10),
        parent_grid_ratio=1, child_nx=12, child_ny=8)
    mask = _strip_mask(plan, (8, 12), ring=1)
    # The 3 new columns on the east, and the old west edge column... is left
    # behind; the old top and bottom rows still inside, and the old east
    # column, now 3 cells in, take the parent's rain.
    assert mask[:, 9:].all()
    assert mask[0, :].all() and mask[7, :].all()
    assert mask[:, 8].all()           # old east edge (11) -> 11 - 3 = 8
    assert not mask[1:7, 0:8].any()


# ---------------------------------------------------------------------------
# The frames are ONE moving nest's history (production run_54ce6aa98bdfdfe9:
# no 1 h rain on the moving 1 km and 333 m grids at any hour)
# ---------------------------------------------------------------------------

def _frame_attrs(monkeypatch, tmp_path, **moving):
    """The global attributes one offline-child frame is written with."""
    from types import SimpleNamespace

    import woof.io.wrfout as wrfout
    import woof.offline_child_run as run

    seen = {}

    class Writer:
        def __init__(self, path, **kwargs):
            seen.update(kwargs["global_attrs"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_frame(self, *args):
            pass

    monkeypatch.setattr(wrfout, "WrfoutWriter", Writer)
    monkeypatch.setattr(run, "_output_fields", lambda *a, **k: {})
    monkeypatch.setattr(run, "soil_layer_count", lambda cfg: 4)
    cfg = SimpleNamespace(grid_id=2, dt=6.0, hybrid_opt=2, etac=0.2, nx=24,
                          ny=24, nz=10, dx=1000.0, dy=1000.0)
    run._write_frame(
        tmp_path / "wrfout_d02_2026-10-28_22:00:00", None, cfg,
        SimpleNamespace(valid_time=datetime(2026, 10, 28, 21, 30)),
        datetime(2026, 10, 28, 22), {}, SimpleNamespace(
            i_parent_start=7, j_parent_start=7, parent_grid_ratio=3),
        **moving)
    return seen


def test_a_segment_writes_the_childs_start_and_its_place_on_the_lattice(
        monkeypatch, tmp_path):
    # The segment ran 21:30-22:00 at (7, 7) in its cropped window; the child
    # began at 21:00 at lattice (131, 88) of the real parent.
    attrs = _frame_attrs(monkeypatch, tmp_path,
                         simulation_start=datetime(2026, 10, 28, 21),
                         history_parent_start=(131, 88))
    assert attrs["START_DATE"] == "2026-10-28_21:00:00"
    assert attrs["SIMULATION_START_DATE"] == "2026-10-28_21:00:00"
    assert (attrs["I_PARENT_START"], attrs["J_PARENT_START"]) == (131, 88)
    assert attrs["PARENT_GRID_RATIO"] == 3


def test_an_ordinary_child_writes_its_own_start_and_place(monkeypatch, tmp_path):
    attrs = _frame_attrs(monkeypatch, tmp_path)
    assert attrs["START_DATE"] == "2026-10-28_21:30:00"
    assert attrs["SIMULATION_START_DATE"] == "2026-10-28_21:30:00"
    assert (attrs["I_PARENT_START"], attrs["J_PARENT_START"]) == (7, 7)


def test_a_run_folder_frame_is_a_file_of_the_run_folder(tmp_path):
    # The renderer keys a series on the folder of the resolved file, so a
    # symlink into segments/sNN made every segment its own run.
    from woof.moving_downscale import _link_frame

    segment = tmp_path / "segments" / "s01"
    segment.mkdir(parents=True)
    frame = segment / "wrfout_d02_2026-10-28_22_00_00"
    frame.write_bytes(b"CDF\x02 frame")
    out = tmp_path / "wrfout_d02_2026-10-28_22_00_00"
    _link_frame(frame, out)
    assert not out.is_symlink()
    assert out.resolve().parent == tmp_path.resolve()
    assert out.read_bytes() == frame.read_bytes()


def test_the_run_windows_are_drawn_beside_every_earlier_frame(monkeypatch, tmp_path):
    import woof.go_cli as go_cli
    import woof.moving_downscale as moving
    from contextlib import contextmanager

    names = [f"wrfout_d02_2026-10-28_{h:02d}_{m:02d}_00"
             for h in (21, 22) for m in (0, 15, 30, 45)][:5]
    for name in names:
        (tmp_path / name).write_bytes(b"CDF")
    (tmp_path / "wrfout_d01_2026-10-28_21_00_00").write_bytes(b"CDF")

    @contextmanager
    def no_file():
        yield None

    commands = []

    class Done:
        returncode, stderr = 0, ""

    monkeypatch.setattr(go_cli, "render_inputs_file", no_file)
    monkeypatch.setattr(moving.subprocess, "run",
                        lambda command, **kw: commands.append(command) or Done())
    closing = [tmp_path / names[-1]]
    record = moving.draw_run_windows(tmp_path, closing, "all")
    (command,) = commands
    assert command[command.index("--products") + 1] == "windowed"
    context = [command[i + 1] for i, part in enumerate(command)
               if part == "--context-wrfout"]
    assert context == [str(tmp_path / name) for name in names[:-1]]
    assert str(tmp_path / names[-1]) in command
    assert record["context"] == 4 and record["rc"] == 0

    # A request with no picture draws nothing.
    commands.clear()
    moving.draw_run_windows(tmp_path, closing, "none")
    assert commands == []


# ---------------------------------------------------------------------------
# A run-total window on ground the grid reached during it (regional-storm
# test, HRRR 2025-05-19: the moving 333 m grid showed 1 h rain on 22 % of
# itself, and the storms, all on newly covered ground, not at all)
# ---------------------------------------------------------------------------

def _history(path, when, *, fields, nx, ny, attrs=None, xtime=0.0):
    """A one-time classic history frame the Rust reader opens."""
    import netCDF4
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("west_east", nx)
        ds.createDimension("south_north", ny)
        for name, value in (attrs or {}).items():
            ds.setncattr(name, value)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = netCDF4.stringtochar(
            np.array([when.strftime("%Y-%m-%d_%H:%M:%S")], dtype="S19"))[0]
        xt = ds.createVariable("XTIME", "f4", ("Time",))
        xt.units = "minutes since 2025-05-19 19:00:00"
        xt[0] = xtime
        for name, values in fields.items():
            var = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            var.units = "mm"
            var[0] = values
    return path


def test_a_window_start_is_the_grids_own_where_it_was_and_the_parents_elsewhere(tmp_path):
    import netCDF4
    import numpy as np

    from woof.moving_downscale import (
        reseeded_after, window_start_values, write_window_start)
    from woof.render import history_series_groups

    ratio, span = 3, 4
    nx = ny = span * ratio
    t_child, t0, t1 = (datetime(2025, 5, 19, 19), datetime(2025, 5, 19, 20),
                       datetime(2025, 5, 19, 21))
    # The parent moves too: lattice (0, 0) at the child's start, (2, 1) at 20Z.
    jj, ii = np.mgrid[0:14, 0:16].astype(float)
    first = Footprint(path=_history(tmp_path / "p" / "wrfout_d02_19", t_child,
                                    fields={"RAINNC": 1.0 + 0 * ii,
                                            "RAINC": 0.5 + 0 * ii},
                                    nx=16, ny=14),
                      valid_time=t_child, origin_i=0, origin_j=0, nx=16, ny=14)
    # At 20Z lattice cell (i, j) has had 1 + 2 i + 3 j since 19Z.
    lat_i, lat_j = ii + 2, jj + 1
    now = Footprint(path=_history(tmp_path / "p" / "wrfout_d02_20", t0,
                                  fields={"RAINNC": 2.0 + 2 * lat_i + 3 * lat_j,
                                          "RAINC": 0.5 + 0 * ii},
                                  nx=16, ny=14),
                    valid_time=t0, origin_i=2, origin_j=1, nx=16, ny=14)
    own = 1000.0 + np.arange(nx * ny, dtype=float).reshape(ny, nx)
    start_frame = _history(tmp_path / "run" / "wrfout_d03_2025-05-19_20_00_00", t0,
                           fields={"RAINNC": own, "RAINC": 0 * own,
                                   "RAINSH": 0 * own}, nx=nx, ny=ny)
    # The grid was at lattice (4, 4) at 20Z and is at (6, 5) when the hour ends.
    values, cells = window_start_values(
        start_frame, start_place=(4, 4), end_place=(6, 5), nx=nx, ny=ny,
        ratio=ratio, parent_now=now, parent_first=first)
    rainnc = values["RAINNC"]
    # Kept ground: the grid's own 20Z values, moved by (6, 3) cells.
    assert np.array_equal(rainnc[:ny - 3, :nx - 6], own[3:, 6:])
    # New ground: the parent's rain since 19Z (exact for a linear field).
    x = 6 + (np.arange(nx) + 0.5) / ratio - 0.5
    y = 5 + (np.arange(ny) + 0.5) / ratio - 0.5
    want = 1.0 + 2 * x[np.newaxis, :] + 3 * y[:, np.newaxis]
    new = np.ones((ny, nx), dtype=bool)
    new[:ny - 3, :nx - 6] = False
    assert np.allclose(rainnc[new], want[new], atol=1e-4)
    assert np.allclose(values["RAINC"][new], 0.0)
    assert np.all(values["RAINSH"][new] == 0.0)   # the parent has none: zero
    assert cells == {"own": (nx - 6) * (ny - 3),
                     "parent": nx * ny - (nx - 6) * (ny - 3), "missing": 0}

    # The warm start at the move also gave the parent's rain to the earlier
    # grid's specified ring (spec_zone 1): its last column (end-grid i = 5)
    # and row (j = 8).  Kept, the grid's own value there drew a line of too
    # much rain along the old edge, so the ring starts from the parent too.
    reseeded = reseeded_after([((4, 4), (6, 5))], (6, 5), nx=nx, ny=ny,
                              ratio=ratio, ring=1)
    ringed, cells = window_start_values(
        start_frame, start_place=(4, 4), end_place=(6, 5), nx=nx, ny=ny,
        ratio=ratio, parent_now=now, parent_first=first, reseeded=reseeded)
    kept = np.zeros((ny, nx), dtype=bool)
    kept[:ny - 4, :nx - 7] = True
    assert np.array_equal(ringed["RAINNC"][kept], rainnc[kept])
    assert np.allclose(ringed["RAINNC"][~kept], want[~kept], atol=1e-4)
    assert cells["own"] == (nx - 7) * (ny - 4)

    attrs = {"START_DATE": "2025-05-19_19:00:00",
             "SIMULATION_START_DATE": "2025-05-19_19:00:00", "GRID_ID": 3,
             "DX": 333.33, "DY": 333.33, "I_PARENT_START": 7,
             "J_PARENT_START": 6, "PARENT_GRID_RATIO": 3}
    lat = 36.0 + np.zeros((ny, nx))
    end = _history(tmp_path / "run" / "wrfout_d03_2025-05-19_21_00_00", t1,
                   fields={"RAINNC": rainnc + 4.0, "RAINC": 0 * own,
                           "RAINSH": 0 * own, "T2": 300 + 0 * own,
                           "XLAT": lat, "XLONG": lat - 134.0},
                   nx=nx, ny=ny, attrs=attrs, xtime=120.0)
    start = write_window_start(end, tmp_path / "starts" / start_frame.name,
                               when=t0, values=values)
    with netCDF4.Dataset(start) as ds, netCDF4.Dataset(end) as closing:
        assert netCDF4.chartostring(ds["Times"][:])[0] == "2025-05-19_20:00:00"
        assert float(ds["XTIME"][0]) == 60.0
        assert np.allclose(ds["RAINNC"][0], rainnc.astype(np.float32))
        assert np.array_equal(ds["T2"][0], closing["T2"][0])
        assert {a: closing.getncattr(a) for a in closing.ncattrs()} == {
            a: ds.getncattr(a) for a in ds.ncattrs()}
    # The renderer's grouping joins it, from its own folder, to the closing
    # frame's series as the window's start.
    ((series, context),) = history_series_groups([start, end],
                                                 context_paths=[start])
    assert [p.name for p in series] == [start.name, end.name]
    assert [p.resolve() for p in context] == [start.resolve()]


def test_the_reseeded_cells_are_the_warm_starts_own_mask_carried_to_the_end():
    import numpy as np

    from woof.core.nest_relocation import Placement, plan_relocation
    from woof.moving_downscale import reseeded_after
    from woof.offline_child_warm import _strip_mask

    nx, ny, ratio = 12, 9, 3

    def warm(shift_i, shift_j):
        base = 1 + max(abs(shift_i), abs(shift_j))
        plan = plan_relocation(
            placement_from=Placement(grid_id=3, i_parent_start=base, j_parent_start=base),
            placement_to=Placement(grid_id=3, i_parent_start=base + shift_i,
                                   j_parent_start=base + shift_j),
            parent_grid_ratio=1, child_nx=nx, child_ny=ny)
        return _strip_mask(plan, (ny, nx), ring=1)

    # One move, ending where it landed: exactly the warm start's mask.
    one = reseeded_after([((10, 10), (11, 10))], (11, 10), nx=nx, ny=ny,
                         ratio=ratio, ring=1)
    assert np.array_equal(one, warm(3, 0))
    # Two moves: the first move's mask, carried one more move east, joins
    # the second's.
    two = reseeded_after([((10, 10), (11, 10)), ((11, 10), (12, 10))], (12, 10),
                         nx=nx, ny=ny, ratio=ratio, ring=1)
    first = np.zeros((ny, nx), dtype=bool)
    first[:, :nx - 3] = warm(3, 0)[:, 3:]
    assert np.array_equal(two, first | warm(3, 0))


def test_ground_the_parent_did_not_cover_either_stays_missing(tmp_path):
    import numpy as np

    from woof.moving_downscale import parent_accumulations_since

    t_child, t0 = datetime(2025, 5, 19, 19), datetime(2025, 5, 19, 20)
    field = {"RAINNC": np.full((6, 6), 5.0)}
    first = Footprint(path=_history(tmp_path / "a", t_child, fields=field, nx=6, ny=6),
                      valid_time=t_child, origin_i=0, origin_j=0, nx=6, ny=6)
    now = Footprint(path=_history(tmp_path / "b", t0, fields=field, nx=6, ny=6),
                    valid_time=t0, origin_i=0, origin_j=0, nx=6, ny=6)
    # A 4 x 4-cell child at lattice (3, 1) reaches past the parent's east edge.
    out = parent_accumulations_since(now, first, child_i=3, child_j=1,
                                     span_i=4, span_j=4, ratio=1,
                                     names=("RAINNC",))
    rain = out["RAINNC"]
    assert np.all(rain[:, :2] == 0.0)
    assert np.isnan(rain[:, 3]).all()


def test_only_windows_whose_start_the_grid_left_are_drawn_again(monkeypatch, tmp_path):
    from contextlib import contextmanager

    import woof.go_cli as go_cli
    import woof.moving_downscale as moving
    from woof.moving_downscale import Segment

    for h, m in ((19, 0), (19, 30), (20, 0), (20, 30), (21, 0)):
        (tmp_path / f"wrfout_d03_2025-05-19_{h:02d}_{m:02d}_00").write_bytes(b"CDF")
    start = datetime(2025, 5, 19, 19)

    def segment(k, place):
        return Segment(index=k, start=start + timedelta(minutes=30 * k),
                       end=start + timedelta(minutes=30 * (k + 1)), frames=(),
                       child_i=place[0], child_j=place[1], span_i=80, span_j=80,
                       window_i=0, window_j=0, window_nx=0, window_ny=0,
                       clamped=False, target=(0.0, 0.0))

    built, commands = [], []

    def values(frame, **kw):
        built.append((frame.name, kw["start_place"], kw["end_place"]))
        return {}, {"own": 0, "parent": 0, "missing": 0}

    def write(end, target, **kw):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"CDF")
        return target

    @contextmanager
    def no_file():
        yield None

    class Done:
        returncode, stderr = 0, ""

    monkeypatch.setattr(moving, "window_start_values", values)
    monkeypatch.setattr(moving, "write_window_start", write)
    monkeypatch.setattr(go_cli, "render_inputs_file", no_file)
    monkeypatch.setattr(moving.subprocess, "run",
                        lambda command, **kw: commands.append(command) or Done())
    scratch = tmp_path / "segments" / "s03-window-starts"
    closing = [tmp_path / "wrfout_d03_2025-05-19_21_00_00"]
    # The grid moved at 19:30, 20:00 and 20:30: the 20Z frame the run folder
    # keeps (segment 1's) is at another place than the 21Z one.
    moved = [segment(0, (10, 10)), segment(1, (18, 12)), segment(2, (26, 20)),
             segment(3, (34, 22))]
    record = moving.draw_window_starts(
        tmp_path, closing, "all", segments=moved, parent_frames={},
        parent_first=None, ratio=3, nx=240, ny=240, scratch=scratch)
    assert built == [("wrfout_d03_2025-05-19_20_00_00", (18, 12), (34, 22))]
    (command,) = commands
    assert command[command.index("--products") + 1] == "qpf_1h"
    context = [command[i + 1] for i, part in enumerate(command)
               if part == "--context-wrfout"]
    assert context == [str(scratch / "wrfout_d03_2025-05-19_20_00_00")]
    assert str(closing[0]) in command
    assert not scratch.exists() and record["draws"][0]["rc"] == 0

    # A grid that kept its place over the hour: the run folder's own frames
    # close the window, and nothing is drawn again.
    built.clear()
    commands.clear()
    still = [segment(k, (10, 10)) for k in range(4)]
    moving.draw_window_starts(
        tmp_path, closing, "all", segments=still, parent_frames={},
        parent_first=None, ratio=3, nx=240, ny=240, scratch=scratch)
    assert built == [] and commands == []


# ---------------------------------------------------------------------------
# A moving grid cut from a compressed parent (2026-11-19: every moving grid
# cut from a deflate-compressed history, the default history format, failed
# at its first cut with "numpy dtype dtype('O') has no classic NetCDF
# external type", and the run lost its moving grid.  The Rust reader reports
# a NetCDF-4 frame's character Times as String, which the bridge maps to
# numpy object, and _write_like declared the cut's Times with that type.)
# ---------------------------------------------------------------------------

_PARENT_FORMATS = ("NETCDF3_64BIT_OFFSET", "NETCDF4_CLASSIC", "NETCDF4")


def _parent_frame(path, when, *, nx, ny, file_format, nz=2, xtime=0.0):
    """A one-time history frame on the C grid, deflated unless classic."""
    import netCDF4
    import numpy as np

    deflate = ({} if file_format.startswith("NETCDF3")
               else {"zlib": True, "complevel": 1, "shuffle": True})
    rng = np.random.default_rng(7)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format=file_format) as ds:
        for name, size in (("Time", None), ("DateStrLen", 19),
                           ("west_east", nx), ("south_north", ny),
                           ("west_east_stag", nx + 1),
                           ("south_north_stag", ny + 1), ("bottom_top", nz)):
            ds.createDimension(name, size)
        ds.setncatts({"WEST-EAST_GRID_DIMENSION": np.int32(nx + 1),
                      "SOUTH-NORTH_GRID_DIMENSION": np.int32(ny + 1),
                      "DX": np.float32(3000.0), "DY": np.float32(3000.0),
                      "TITLE": " OUTPUT FROM GPUWM"})
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = netCDF4.stringtochar(
            np.array([when.strftime("%Y-%m-%d_%H:%M:%S")], dtype="S19"))[0]
        xt = ds.createVariable("XTIME", "f4", ("Time",))
        xt.units = "minutes since 2026-11-18 20:00:00"
        xt[0] = xtime
        step = ds.createVariable("ITIMESTEP", "i4", ("Time",))
        step[0] = 360
        for name, dims, units in (
                ("T2", ("Time", "south_north", "west_east"), "K"),
                ("U", ("Time", "bottom_top", "south_north", "west_east_stag"),
                 "m s-1"),
                ("V", ("Time", "bottom_top", "south_north_stag", "west_east"),
                 "m s-1")):
            var = ds.createVariable(name, "f4", dims, **deflate)
            var.units = units
            var[0] = rng.random(tuple(len(ds.dimensions[d]) for d in dims[1:]),
                                dtype=np.float32)
    return path


@pytest.mark.parametrize("file_format", _PARENT_FORMATS)
def test_a_moving_cut_from_a_compressed_parent_writes_its_times_as_characters(
        tmp_path, file_format):
    import netCDF4
    import numpy as np

    from woof.moving_downscale import crop_frame

    source = _parent_frame(tmp_path / "p" / "wrfout_d02_2026-11-18_21_00_00",
                           datetime(2026, 11, 18, 21), nx=20, ny=16,
                           file_format=file_format)
    target = tmp_path / "cut" / source.name
    target.parent.mkdir()
    crop_frame(source, target, origin_i=0, origin_j=0, window_i=2,
               window_j=3, nx=10, ny=8)
    with netCDF4.Dataset(target) as cut, netCDF4.Dataset(source) as parent:
        assert cut.file_format.startswith("NETCDF3")
        assert cut["Times"].dtype == np.dtype("S1")
        assert netCDF4.chartostring(cut["Times"][:])[0] == "2026-11-18_21:00:00"
        assert np.array_equal(cut["T2"][0], parent["T2"][0, 3:11, 2:12])
        assert np.array_equal(cut["U"][0], parent["U"][0, :, 3:11, 2:13])
        assert np.array_equal(cut["V"][0], parent["V"][0, :, 3:12, 2:12])
        assert cut["ITIMESTEP"].dtype == np.int32
        assert int(cut["ITIMESTEP"][0]) == 360
        assert cut["U"].units == "m s-1"
        assert int(cut.getncattr("WEST-EAST_GRID_DIMENSION")) == 11
        assert int(cut.getncattr("SOUTH-NORTH_GRID_DIMENSION")) == 9
        assert list(cut.getncattr("GPUWM_MOVING_WINDOW")) == [2, 3, 10, 8]


@pytest.mark.parametrize("file_format", _PARENT_FORMATS)
def test_a_window_start_from_a_compressed_frame_writes_its_times_as_characters(
        tmp_path, file_format):
    import netCDF4
    import numpy as np

    from woof.moving_downscale import write_window_start

    end = _parent_frame(tmp_path / "run" / "wrfout_d03_2026-11-18_22_00_00",
                        datetime(2026, 11, 18, 22), nx=12, ny=12,
                        file_format=file_format, xtime=120.0)
    t2 = 280.0 + np.arange(144, dtype=np.float64).reshape(12, 12)
    start = write_window_start(
        end, tmp_path / "starts" / "wrfout_d03_2026-11-18_21_00_00",
        when=datetime(2026, 11, 18, 21), values={"T2": t2})
    with netCDF4.Dataset(start) as ds, netCDF4.Dataset(end) as closing:
        assert ds.file_format.startswith("NETCDF3")
        assert ds["Times"].dtype == np.dtype("S1")
        assert netCDF4.chartostring(ds["Times"][:])[0] == "2026-11-18_21:00:00"
        assert float(ds["XTIME"][0]) == 60.0
        assert np.array_equal(ds["T2"][0], t2.astype(np.float32))
        assert np.array_equal(ds["U"][0], closing["U"][0])
        assert int(ds["ITIMESTEP"][0]) == 360


def test_a_move_larger_than_the_grid_starts_cold_from_the_parent(tmp_path):
    """A moving 1 km grid can jump 60 km at one move, and a 96-cell 333 m
    grid inside it is 32 km across: the inner grid's next segment shares no
    ground with its last.  That used to refuse ("leaves no ground shared")
    and end the run without the grid.  Nothing can be carried, so the grid
    keeps its cold start from the parent at the new place, and every cell
    takes the parent's rain since the grid began, none of the old grid's."""
    import json
    from types import SimpleNamespace

    import numpy as np

    from woof.io.restart import _HEADER_KEY
    from woof.offline_child_warm import _strip_mask, apply_warm_start
    from woof.moving_downscale import reseeded_after

    nx = ny = 96
    header = {"config": {"nx": nx, "ny": ny, "nz": 50, "dx": 1000 / 3,
                         "dy": 1000 / 3},
              "valid_time": "2026-11-19T07:00:00Z"}
    checkpoint = tmp_path / "wrfrst_d03.npz"
    np.savez(checkpoint, **{
        _HEADER_KEY: np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
        "scratch/mp_rainnc": np.full((ny, nx), 7.0, dtype=np.float32),
        "state/u": np.ones((50, ny, nx + 1), dtype=np.float32)})
    parent_rain = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) / 100
    strip = tmp_path / "strip-accumulations.npz"
    np.savez(strip, mp_rainnc=parent_rain)
    rain = np.zeros((ny, nx), dtype=np.float32)
    child = SimpleNamespace(
        existing_scratch=lambda slot: rain if slot == "mp_rainnc" else None)
    cfg = SimpleNamespace(nx=nx, ny=ny, nz=50, dx=1000 / 3, dy=1000 / 3,
                          grid_id=3, spec_zone=1, hypsometric_opt=2)

    # 60 km of the 1 km parent is 60 parent cells, 180 cells of this grid.
    receipt = apply_warm_start(child, cfg, SimpleNamespace(fields={}),
                               checkpoint=checkpoint, shift=(180, 0),
                               strip=strip)
    assert receipt["start"] == "cold"
    assert receipt["overlap_cells"] == 0
    assert receipt["strip_cells"] == nx * ny
    assert receipt["state_fields_stamped"] == 0
    assert receipt["accumulators"]["mp_rainnc"] == {"overlap_cells": 0,
                                                    "strip_cells": nx * ny}
    np.testing.assert_array_equal(rain, parent_rain)
    # The run's window starts count the same cells as reseeded.
    assert reseeded_after([((10, 10), (70, 10))], (70, 10), nx=nx, ny=ny,
                          ratio=3, ring=1).all()
    # One cell short of the grid's width still shares ground and is not cold.
    from woof.core.nest_relocation import Placement, plan_relocation
    near = plan_relocation(
        placement_from=Placement(grid_id=3, i_parent_start=97, j_parent_start=97),
        placement_to=Placement(grid_id=3, i_parent_start=97 + 95,
                               j_parent_start=97),
        parent_grid_ratio=1, child_nx=nx, child_ny=ny)
    assert not near.disjoint and not _strip_mask(near, (ny, nx)).all()
