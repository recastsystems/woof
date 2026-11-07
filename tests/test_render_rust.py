"""``woof render --engine rust`` and ``--pair`` -- CPU-only tests.

The rust-engine tests run against the real vendored renderer binary
(``tools/rustwx``) and skip with a stated reason when it is not built; nothing here
mocks the engine.  The fixture wrfout is written by the project's own
``WrfoutWriter`` with the production global-attribute profile
(``wrf_global_attrs``: Lambert projection, START_DATE, domain
topology), because that is what ``woof run`` writes and what the rust
importer's fail-closed preflight requires.  Its second frame is on the
half hour -- every rust-engine test doubles as the exact-time
(sub-hourly) regression test that the source renderer refused.

PNG dimensions are read from the IHDR chunk directly so the assertions
add no image-library dependency.
"""

from __future__ import annotations

import datetime
import re
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import woof.cli as cli
from woof import render_layout, run_stamp, rustwx
from woof.io.wrfout import WrfoutWriter, wrf_global_attrs
from woof.render import parse_products_rust, parse_size

_NZ, _NY, _NX = 4, 12, 16
_STAMPS = ("1974-04-03_18:00:00", "1974-04-03_18:30:00")
_LEADS = ("lead_000h00m00s", "lead_000h30m00s")

RENDERER = rustwx.find_renderer()


def _renderer_gate() -> tuple[bool, str]:
    """(usable, why not) for THIS tree's renderer -- contract, not stat().

    Task #106.  The gate used to be ``find_renderer() is None``, which is
    a question about a filename.  The resolution ladder's last rung is
    ``~/.woof/bridges``, so on any box that has ever installed a bundle
    the answer was yes -- and the binary answering it was whichever build
    happened to be staged there.  A stale one turned this suite red on a
    correct tree (catalog 153 against 168, zero generic rows) and would
    just as readily have turned a broken tree green, because none of
    these tests can tell which engine drew the plots.

    So the gate asks the contract question instead, and it asks it
    through :func:`woof.render.renderer_refusal` -- the one function the
    render path itself reads, covering both the
    :func:`woof.rustwx.probe_renderer` ``--abi`` handshake ``woof
    doctor`` reports and the provenance clause that catches a sibling
    checkout answering the same contract correctly.  Reading the
    product's own gate rather than re-deriving it is the point: a gate
    the suite defines for itself can admit a renderer the render path
    then refuses, and the suite would report that as a failure of the
    code under test.  A foreign engine SKIPS with its mismatch named
    rather than silently standing in for this tree's.
    """

    if RENDERER is None:
        return False, ("rust renderer not built (cd tools/rustwx && cargo "
                       "build --release --locked --offline)")
    from woof.render import renderer_refusal

    reason = renderer_refusal(RENDERER)
    if reason is None:
        return True, ""
    return False, (
        f"the renderer resolved at {RENDERER} is not this tree's: "
        f"{reason}.  These tests assert THIS tree's product catalog, so "
        "a foreign engine is skipped, not substituted")


_RENDERER_USABLE, _RENDERER_SKIP_REASON = _renderer_gate()
needs_renderer = pytest.mark.skipif(
    not _RENDERER_USABLE, reason=_RENDERER_SKIP_REASON)


def _checkout_build() -> Path | None:
    """THIS checkout's own rw_wrfbatch, or None if it is not built.

    Deliberately not :func:`woof.rustwx.find_renderer`, whose ladder
    ends at ``~/.woof/bridges``: the marker test below is about drift
    between this tree's Rust and this tree's Python, and a binary
    installed from some other release cannot answer that question either
    way.  Whoever has a stale bundle staged learns it from ``woof
    doctor``, which now reports it with the rebuild remedy; they do not
    learn it from a red in a suite they cannot fix by editing code.
    """

    from woof.bridges import executable_name

    built = (rustwx.crate_dir() / "target" / "release"
             / executable_name(rustwx.RENDERER_NAME))
    return built if built.is_file() else None


def test_the_pinned_abi_marker_is_the_built_renderer_s_own_answer():
    """The Python constant against the Rust writer, not against itself.

    A marker both halves read from one Python string proves nothing.
    This runs the binary THIS checkout built and compares its stdout to
    :data:`woof.rustwx.RENDERER_ABI_MARKER` byte for byte, so editing
    the contract on one side and not the other is red rather than a
    silent skip of the whole rust lane.

    Skipped only when this checkout has no build at all -- and that skip
    cannot hide a mismatch, because there is nothing of this tree's to
    mismatch with.
    """

    built = _checkout_build()
    if built is None:
        pytest.skip("this checkout's rust renderer is not built (cd "
                    "tools/rustwx && cargo build --release --locked "
                    "--offline)")
    import subprocess

    probe = subprocess.run([str(built), "--abi"], capture_output=True,
                           text=True, errors="replace", timeout=60)
    assert probe.returncode == 0, (
        f"{built} --abi exited {probe.returncode}: "
        f"{(probe.stderr or '').strip()!r} -- this build predates the "
        "renderer contract handshake; rebuild it from this checkout")
    assert probe.stdout.strip() == rustwx.RENDERER_ABI_MARKER, (
        f"{built} answers a different render contract than "
        "woof.rustwx.RENDERER_ABI_MARKER pins")


def test_every_bundled_rustwx_binary_pins_a_contract_marker():
    """The renderer was the odd one out; nothing may be the odd one again.

    ``rw_fetch`` and ``rw_nexrad`` have always pinned an exact ``--abi``
    line in their Python wrapper.  ``rw_wrfbatch`` did not, which is how
    two builds with different md5s both reported ``verified``.  This
    holds the three together structurally, so the next binary added to
    the workspace cannot arrive without a contract to check it against.
    """

    from woof import rustwx_fetch
    from woof.obs import nexrad

    markers = {
        "rw_fetch": rustwx_fetch.FETCH_ABI_MARKER,
        "rw_nexrad": nexrad.NEXRAD_ABI_MARKER,
        "rw_wrfbatch": rustwx.RENDERER_ABI_MARKER,
    }
    for name, marker in markers.items():
        assert isinstance(marker, str) and marker.strip(), name
        assert "\t" in marker, (
            f"{name}'s marker is not a tab-separated contract line: "
            f"{marker!r}")
    assert len(set(markers.values())) == len(markers), (
        "two bundled binaries pin the SAME marker, so one of them would "
        f"verify against the other's contract: {markers}")


# ---- the probe's decision table -------------------------------------------
#
# Against the real binary above; here against the two answers a stale
# build gives, which cannot be produced without shipping a stale binary.
# What is stubbed is the subprocess layer, never the renderer's meaning:
# these assert what probe_renderer DECIDES, given each observable.

def _stub_probe(monkeypatch, *, abi_returncode, abi_stdout):
    from woof import bridges

    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))

    def run(command, **_kwargs):
        if command[1] == "--help":
            return SimpleNamespace(
                returncode=0, stdout="usage: rw_wrfbatch --store-root DIR",
                stderr="")
        assert command[1] == "--abi", command
        return SimpleNamespace(returncode=abi_returncode, stdout=abi_stdout,
                               stderr="")

    monkeypatch.setattr(rustwx.subprocess, "run", run)


def test_a_binary_that_launches_but_predates_the_handshake_is_refused(
        monkeypatch, tmp_path):
    """`unknown option --abi`, exit 2: every build older than the contract."""

    _stub_probe(monkeypatch, abi_returncode=2, abi_stdout="")
    ok, evidence = rustwx.probe_renderer(tmp_path / "rw_wrfbatch.exe")
    assert ok is False
    assert "--abi does not match the render contract" in evidence
    assert "REBUILD" in evidence


def test_a_binary_answering_another_contract_is_refused(monkeypatch,
                                                        tmp_path):
    """A build that answers --abi, with somebody else's grammar."""

    _stub_probe(monkeypatch, abi_returncode=0,
                abi_stdout="gpuwm-rw-wrfbatch-catalog-v0\tPRODUCT\tslug\n")
    ok, evidence = rustwx.probe_renderer(tmp_path / "rw_wrfbatch.exe")
    assert ok is False
    assert "gpuwm-rw-wrfbatch-catalog-v0" in evidence


def test_the_probe_accepts_the_contract_it_pins(monkeypatch, tmp_path):
    """The negative test: the guard does NOT fire on the right answer."""

    _stub_probe(monkeypatch, abi_returncode=0,
                abi_stdout=rustwx.RENDERER_ABI_MARKER + "\n")
    ok, evidence = rustwx.probe_renderer(tmp_path / "rw_wrfbatch.exe")
    assert ok is True, evidence
    assert "--abi matches the render contract" in evidence


def _frame(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    return {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "REFL_10CM": rng.uniform(-20.0, 65.0,
                                 (_NZ, _NY, _NX)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
        "Q2": rng.uniform(0.004, 0.012, (_NY, _NX)).astype(np.float32),
        "PSFC": rng.uniform(96000.0, 98000.0,
                            (_NY, _NX)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }


def _write_wrfout(path, stamps, *, grid_id=2, dx=1000.0, seed_offset=0,
                  model_label=None, t2_offset=0.0):
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(1974, 4, 3, 18), grid_id=grid_id,
        parent_id=max(grid_id - 1, 1), i_parent_start=5, j_parent_start=5,
        parent_grid_ratio=3, dt=6.0)
    if model_label is not None:
        attrs["GPUWM_MODEL_LABEL"] = model_label
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=dx, dy=dx,
                      global_attrs=attrs) as writer:
        for index, stamp in enumerate(stamps):
            fields = _frame(seed=7 + index + seed_offset)
            fields["T2"] = (fields["T2"] + np.float32(t2_offset)).astype(np.float32)
            writer.write_frame(stamp, fields)
    return path


@pytest.fixture(scope="module")
def wrfout(tmp_path_factory):
    """Two frames, the second on the half hour (exact-time axis)."""

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust") /
        "wrfout_d02_1974-04-03_18-00-00.nc", _STAMPS)


@pytest.fixture(scope="module")
def wrfout_hourly(tmp_path_factory):
    """Two WHOLE-hour frames -- the windowed lane's admissible axis."""

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust-hourly") /
        "wrfout_d02_1974-04-03_18-00-00.nc",
        ("1974-04-03_18:00:00", "1974-04-03_19:00:00"))


@pytest.fixture(scope="module")
def wrfout_hourly_d03(tmp_path_factory):
    """The SAME init and the same whole-hour frames, one nest down.

    Sub-kilometre spacing (333 m), so its resolution token exercises the
    integer-metre branch while its lead/valid times are identical to
    ``wrfout_hourly``'s -- everything the old filename carried.
    """

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust-hourly-d03") /
        "wrfout_d03_1974-04-03_18-00-00.nc",
        ("1974-04-03_18:00:00", "1974-04-03_19:00:00"),
        grid_id=3, dx=333.3333)


def _png_size(path) -> tuple[int, int]:
    """(width, height) from the PNG IHDR chunk; no image library.

    Read through ``render_layout.fs_path`` for the same reason the
    placement seam writes through it: ``--out`` plus a run folder plus
    ``<domain>/<product>/<valid-day>`` plus an exact-time product name
    passes Windows' MAX_PATH from an ordinary temp root, and a reader
    that is less long-path aware than the writer reports a correctly
    filed frame as missing.
    """

    data = Path(render_layout.fs_path(path)).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _delivered(out: Path) -> list[str]:
    """Every delivered PNG under ``out``, as a relative posix path.

    The PATH, not the bare name.  The delivered filename no longer
    carries the domain and product tokens -- the two folders above it
    spell exactly those, and a frame repeating them ran real deliveries
    past Windows' 260-character ceiling -- so a test asking "is this a
    d03-333m reflectivity frame?" reads one and two folders up.  Folder
    plus name carries what the v2.4.1 filename carried, so what these
    tests pin is unchanged.
    """

    return sorted(p.relative_to(out).as_posix()
                  for p in out.rglob("*.png"))


def _domain_of(delivered: str) -> str:
    """The domain token of one delivered path from :func:`_delivered`.

    Third from the end: ``<domain>/<product>/<valid-day>/<file>.png``.
    Counted from the RIGHT because what is above the domain is the
    caller's -- the run-stamped folder, and whatever ``--out`` was.
    """

    return Path(delivered).parts[-4]


@needs_renderer
def test_rust_engine_renders_every_frame_including_half_hourly(
        wrfout, tmp_path, capsys):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--out", str(out)])
    assert rc == 0
    assert "render: engine rust" in capsys.readouterr().out
    produced = _delivered(out)
    assert produced, "the rust engine wrote no PNGs"
    # Both frames rendered -- the :30 frame is the sub-hourly lead the
    # source renderer refused (exact-time ordinal axis).
    for lead in _LEADS:
        matching = [name for name in produced if lead in name]
        assert matching, (lead, produced)
    # 'all' renders the catalog the fixture's fields prove out, which
    # must include the reflectivity composite at minimum.
    assert any("composite_reflectivity" in name for name in produced)
    for png in out.rglob("*.png"):
        assert Path(render_layout.fs_path(png)).stat().st_size > 5_000, (
            png.name)
        width, height = _png_size(png)
        # The default is the fixture's projected domain shape, with one
        # canvas shared by its products. The old landscape size fitted
        # the map into padding; an explicit --size still pins fixed pixels.
        assert (width, height) == (954, 806), png.name


@needs_renderer
def test_rust_engine_timeidx_selects_the_half_hourly_frame(
        wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl", "--timeidx", "1",
                   "--out", str(out)])
    assert rc == 0
    produced = _delivered(out)
    assert len(produced) == 1, produced
    assert "composite_reflectivity" in produced[0]
    assert _LEADS[1] in produced[0], produced
    assert _LEADS[0] not in produced[0]


@needs_renderer
def test_rust_engine_maps_shared_product_names(wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl,t2", "--timeidx", "0",
                   "--size", "800x600", "--out", str(out)])
    assert rc == 0
    produced = _delivered(out)
    assert len(produced) == 2, produced
    assert any("composite_reflectivity" in name for name in produced)
    assert any("2m_temperature" in name for name in produced)
    for png in out.rglob("*.png"):
        assert _png_size(png) == (800, 600)


@needs_renderer
def test_wind10_renders_the_wind_not_a_pressure_chart(wrfout, tmp_path):
    """``--products wind10`` must draw the wind.

    It did not.  ``RUST_PRODUCT_ALIASES["wind10"]`` pointed at
    ``mslp_10m_winds`` -- a mean-sea-level PRESSURE analysis with 10 m
    barbs over it -- under a source comment calling that "the catalog's
    standalone surface-wind product".  Asking the tuned engine for wind
    returned pressure, so during a wildfire an operator went to the
    matplotlib fallback (whose own ``wind10`` IS a wind map) to get one.

    A test comparing ``RUST_PRODUCT_ALIASES["wind10"]`` to a hardcoded
    string would have passed on the broken code, which is why this one
    goes to the artifact instead: render it, and separately ask the
    renderer's own catalog what that product FILLS with.
    """

    from woof import render as render_module

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "wind10", "--timeidx", "0",
                   "--size", "800x600", "--out", str(out)])
    assert rc == 0
    produced = _delivered(out)
    assert len(produced) == 1, produced
    assert "10m_wind_speed_and_direction" in produced[0], produced
    assert "mslp" not in produced[0], (
        "wind10 must not return a mean-sea-level pressure chart")

    rows, _summary = rustwx.list_products(
        RENDERER, wrfout, store_root=tmp_path / "store")
    detail = {slug: text for slug, _kind, _status, text in rows}
    status = {slug: state for slug, _kind, state, _text in rows}

    # What each shared name actually DRAWS, in the renderer's own words.
    # A renderable row states its fill as "[fill: <selector>]"; a row the
    # store cannot serve names the same selector as the field it lacks.
    # Either way the answer to "what is this a chart OF?" is in the row.
    for alias, selector_key in (
            render_module.RUST_PRODUCT_ALIAS_FILL_SELECTORS.items()):
        slug = render_module.RUST_PRODUCT_ALIASES[alias]
        assert slug in detail, f"{alias} -> {slug} is not in the catalog"
        assert selector_key in detail[slug], (
            f"--products {alias} draws {detail[slug]!r}, which is not "
            f"a chart of {selector_key}")

    # The wind one must additionally be renderable from this store --
    # three instantaneous 10 m planes, one frame, no neighbours.
    wind_slug = render_module.RUST_PRODUCT_ALIASES["wind10"]
    assert status[wind_slug] == "renderable", detail[wind_slug]

    # And the product it must never be confused with again: whatever
    # `mslp_10m_winds` fills with, it is pressure, and it is not wind10.
    assert "pressure_reduced_to_mean_sea_level" in detail["mslp_10m_winds"]
    assert wind_slug != "mslp_10m_winds"


# ---------------------------------------------------------------------------
# Domain + resolution tokens: two domains at one lead must not collide
# ---------------------------------------------------------------------------

@needs_renderer
def test_two_domains_at_one_lead_do_not_overwrite_each_other(
        wrfout_hourly, wrfout_hourly_d03, tmp_path):
    """The data-loss bug this token exists to prevent.

    Two nests of one run share model, init cycle, and forecast hour.  On
    the whole-hour axis nothing else distinguished them, so the second
    domain's PNG landed on the first domain's path and one of the two
    forecasts vanished with no error, no warning, and an exit code of 0.
    """

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout_hourly), str(wrfout_hourly_d03),
                   "--engine", "rust", "--products", "refl",
                   "--timeidx", "0", "--out", str(out)])
    assert rc == 0
    produced = _delivered(out)
    assert len(produced) == 2, (
        "two domains at one lead collapsed to one file", produced)
    # The distinguishing token is the domain FOLDER now.  It has to stay
    # distinguishing: the two frames share model, cycle and lead, so
    # without it they are one path and one forecast silently replaces
    # the other, which is the data loss this test exists for.
    assert any(_domain_of(name) == "d02-1km" for name in produced), produced
    assert any(_domain_of(name) == "d03-333m" for name in produced), produced
    # Both survive as real images, not one truncated overwrite.
    for png in out.rglob("*.png"):
        assert png.stat().st_size > 5_000, png.name


@needs_renderer
def test_domain_and_resolution_token_on_the_exact_time_axis(
        wrfout, tmp_path):
    """The sub-hourly axis carries the same token as the whole-hour one."""

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl", "--out", str(out)])
    assert rc == 0
    produced = _delivered(out)
    assert len(produced) == 2, produced
    for name in produced:
        assert _domain_of(name) == "d02-1km", name
        assert "native_grid" not in name, name


@needs_renderer
def test_unknown_domain_identity_keeps_the_native_grid_token(
        tmp_path):
    """No GRID_ID, no wrfout_dNN name: the token degrades, never guesses.

    A file whose domain identity cannot be established keeps the
    renderer's generic ``native_grid`` slug rather than being labelled
    ``d01`` on no evidence.
    """

    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(grid, datetime.datetime(1974, 4, 3, 18),
                             dt=6.0)
    assert "GRID_ID" not in attrs
    path = tmp_path / "some_model_output.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0,
                      dy=1000.0, global_attrs=attrs) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=21))
    out = tmp_path / "png"
    rc = cli.main(["render", str(path), "--engine", "rust",
                   "--products", "refl", "--out", str(out)])
    assert rc == 0
    produced = _delivered(out)
    assert produced, "the anonymous-domain file rendered nothing"
    for name in produced:
        assert _domain_of(name) == "native_grid", name


@needs_renderer
def test_a_delayed_start_nest_draws_every_frame_on_the_runs_lead_clock(
        tmp_path):
    """A137: a nest that starts an hour after the forecast drew nothing.

    WRF, and this writer, put a nest's own start in START_DATE and the
    run's in SIMULATION_START_DATE, so a delayed nest's START_DATE is
    later by design.  The renderer refused every one of its frames as
    "conflicting WRF references", live and through ``woof render``.  Its
    frames draw now, their leads measured from the run's start, so the
    nest's 14Z frame is f002 exactly as its parent's 14Z frame is.
    """

    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    run_start = datetime.datetime(2026, 5, 20, 12)
    nest_start = run_start + datetime.timedelta(hours=1)
    attrs = wrf_global_attrs(
        grid, nest_start, grid_id=3, parent_id=2, i_parent_start=5,
        j_parent_start=5, parent_grid_ratio=3, dt=2.0,
        simulation_start_time=run_start)
    assert attrs["START_DATE"] == "2026-05-20_13:00:00"
    assert attrs["SIMULATION_START_DATE"] == "2026-05-20_12:00:00"
    path = tmp_path / "wrfout_d03_2026-05-20_13-00-00.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=333.3333,
                      dy=333.3333, global_attrs=attrs) as writer:
        for index, stamp in enumerate(("2026-05-20_13:00:00",
                                       "2026-05-20_14:00:00")):
            writer.write_frame(stamp, _frame(seed=31 + index))
    out = tmp_path / "png"
    rc = cli.main(["render", str(path), "--engine", "rust",
                   "--products", "refl", "--out", str(out)])
    assert rc == 0
    produced = _delivered(out)
    assert len(produced) == 2, produced
    clocks = sorted(re.search(r"_(\d{1,2})z_f(\d{3})", Path(name).name).groups()
                    for name in produced)
    assert clocks == [("12", "001"), ("12", "002")], produced
    for name in produced:
        assert _domain_of(name) == "d03-333m", name


def test_the_engines_skip_line_is_read_and_is_not_a_failure(monkeypatch,
                                                            tmp_path):
    """`rw_wrfbatch` has always emitted three verdicts; two were read.

    A product whose stored fields are absent is `SKIPPED <slug>
    <reason>` on stdout and is NOT counted in `summary.failed`, so the
    process still exits 0.  Reading only RENDERED/FAILED made that
    verdict invisible to every caller, which is why a render could
    quietly draw one image fewer than the catalog and say nothing.

    Third arm is the negative control: FAILED is still a failure.
    """

    import subprocess

    class Result:
        returncode = 0
        stdout = "\n".join((
            "RENDERED t2 /tmp/png/t2.png",
            "SKIPPED refl store carries no REFL_10CM",
            ""))
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    written, failures, skipped = rustwx.run_renderer(
        tmp_path / "rw_wrfbatch", tmp_path / "wrfout_d01_x.nc",
        store_root=tmp_path / "store", out_dir=tmp_path / "png",
        products="all", frames="all", width=800, height=600)
    assert [p.name for p in written] == ["t2.png"]
    assert failures == []
    assert len(skipped) == 1
    product, detail = skipped[0]
    assert product == "refl"
    assert "store carries no REFL_10CM" in detail

    class Failed:
        returncode = 1
        stdout = "RENDERED t2 /tmp/png/t2.png\n"
        stderr = "FAILED refl contour engine panicked\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Failed())
    _written, failures, skipped = rustwx.run_renderer(
        tmp_path / "rw_wrfbatch", tmp_path / "wrfout_d01_x.nc",
        store_root=tmp_path / "store", out_dir=tmp_path / "png",
        products="all", frames="all", width=800, height=600)
    assert skipped == []
    assert len(failures) == 1
    assert "contour engine panicked" in failures[0]


def test_source_label_reaches_the_renderer_invocation(monkeypatch,
                                                      tmp_path):
    """A locally imported run is not a GDEX fetch, and a stock-WRF file
    is not ours: both are one flag away from being labelled accurately."""

    import subprocess

    from woof import render as render_module

    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def spy(command, **kwargs):
        seen.append([str(part) for part in command])
        return Result()

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr("woof.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch.exe")
    monkeypatch.setattr("woof.rustwx.probe_renderer",
                        lambda path: (True, "stubbed"))
    # A stub under tmp_path belongs to no tree, so the bridge gate would
    # refuse it -- correctly.  Declaring it through the override is the
    # documented way to say "this binary is the one I mean", and is what
    # a person pointing woof at a hand-built renderer does.
    monkeypatch.setenv(rustwx.RENDERER_ENV,
                       str((tmp_path / "rw_wrfbatch.exe").resolve()))

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="composite_reflectivity",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600))
    assert seen, "no renderer invocation was made"
    command = seen[-1]
    assert "--source-label" in command
    # The default label is the brand plus the EXECUTING version, and it
    # is asked for rather than transcribed so the assertion survives a
    # release cut.
    assert (command[command.index("--source-label") + 1]
            == render_module.default_source_label())
    assert command[command.index("--source-label") + 1].startswith("WOOF ")

    seen.clear()
    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="composite_reflectivity",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        source_label="WRF-ARW 4.6.1")
    command = seen[-1]
    assert command[command.index("--source-label") + 1] == "WRF-ARW 4.6.1"


def _renderer_spy(monkeypatch, tmp_path):
    """Capture the argv `woof render --engine rust` hands the binary."""

    import subprocess

    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def spy(command, **kwargs):
        seen.append([str(part) for part in command])
        return Result()

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr("woof.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch.exe")
    monkeypatch.setattr("woof.rustwx.probe_renderer",
                        lambda path: (True, "stubbed"))
    monkeypatch.setenv(rustwx.RENDERER_ENV,
                       str((tmp_path / "rw_wrfbatch.exe").resolve()))
    return seen


def test_streamlines_have_a_front_door_flag(monkeypatch, tmp_path):
    """Wind streamlines were reachable only through an environment
    variable named in no help text -- which under the
    ship-only-what-users-can-reach rule means they were not shipped.

    Both spellings have to reach the engine, and saying neither has to
    leave the invocation byte-identical to every earlier release, so a
    render that never mentions the wind layer is unchanged.
    """

    from woof import render as render_module

    seen = _renderer_spy(monkeypatch, tmp_path)

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="wind10",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600))
    assert "--streamlines" not in seen[-1]
    assert "--barbs" not in seen[-1]

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="wind10",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        streamlines=True)
    assert "--streamlines" in seen[-1]

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="wind10",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        streamlines=False)
    assert "--barbs" in seen[-1]


def test_the_streamlines_flag_is_on_the_render_parser(monkeypatch,
                                                      tmp_path):
    """The flag has to be typed by a user, not only passed by a caller:
    `woof render --streamlines` is the door, and `--barbs` is how a
    user overrules an inherited `RUSTWX_WIND_STREAMLINES=1`."""

    seen = _renderer_spy(monkeypatch, tmp_path)
    wrfout = tmp_path / "wrfout_d02_2026-08-19_00_00_00"
    wrfout.write_bytes(b"")

    parser = cli.build_parser()
    args = parser.parse_args(
        ["render", str(wrfout), "--engine", "rust", "--streamlines",
         "--out", str(tmp_path / "png")])
    assert args.streamlines is True

    args = parser.parse_args(
        ["render", str(wrfout), "--engine", "rust", "--barbs",
         "--out", str(tmp_path / "png")])
    assert args.streamlines is False

    args = parser.parse_args(
        ["render", str(wrfout), "--engine", "rust",
         "--out", str(tmp_path / "png")])
    assert args.streamlines is None
    assert not seen


def test_engine_outputs_are_rebranded_to_the_product_prefix(monkeypatch,
                                                            tmp_path,
                                                            capsys):
    """rustwx_* out of the engine becomes arwen_* out of `woof render`.

    The vendored engine hardcodes ``rustwx_`` into every output
    filename (rustwx-products' format strings); the product shipping
    those files is WOOF, and the field run's first question about its
    own pictures was what "rustwx" meant.  The rename happens at the
    one seam every wheel render flows through, so the engine binary
    stays byte-identical to its campaign builds.  Filenames only:
    a name the engine did not brand passes through untouched, and a
    RENDERED line naming a file that is not on disk keeps its reported
    path rather than failing the render over branding.
    """

    import subprocess

    from woof import render as render_module

    outdir = tmp_path / "png"
    branded = "rustwx_wrf_19740403_12z_f003_d02-3km_sbcape.png"
    ghost = "rustwx_wrf_19740403_12z_f003_d02-3km_ghost.png"

    def fake_renderer(command, **kwargs):
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / branded).write_bytes(b"\x89PNG")
        (out / "unbranded_extra.png").write_bytes(b"\x89PNG")

        class Result:
            returncode = 0
            stdout = (f"RENDERED sbcape {out / branded}\n"
                      f"RENDERED extra {out / 'unbranded_extra.png'}\n"
                      f"RENDERED ghost {out / ghost}\n")
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_renderer)
    monkeypatch.setattr("woof.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch.exe")
    # The renderer gate, satisfied both ways, exactly as the source-label
    # test above does it: the CONTRACT half is stubbed because this
    # tmp_path stub is a name and not an executable, and the PROVENANCE
    # half is declared because a stub under tmp_path belongs to no tree
    # and the bridge gate is right to refuse one nobody named.  What this
    # test pins is the rebranding of the engine's output, not which
    # engine is admitted.
    monkeypatch.setattr("woof.rustwx.probe_renderer",
                        lambda path: (True, "stubbed"))
    monkeypatch.setenv(rustwx.RENDERER_ENV,
                       str((tmp_path / "rw_wrfbatch.exe").resolve()))

    written, failures, _skipped = render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="sbcape", timeidx=0,
        outdir=outdir, size=(800, 600))
    delivered = "arwen_wrf_19740403_12z_f003.png"
    assert failures == []
    assert sorted(p.name for p in written) == [
        delivered,
        ghost,
        "unbranded_extra.png",
    ]
    # On disk: the branded file moved and nothing else changed.  Since
    # 2.5.0 it moves TWICE -- rebranded, then filed under the render
    # layout (woof.render_layout) -- and a name the engine's grammar
    # does not produce is left exactly where the engine put it, because
    # nothing here knows its product or its valid time.  The filed name
    # drops the domain and product tokens, which are the two folders it
    # lands in; carrying them twice is what ran delivered paths past the
    # Windows ceiling.
    assert (outdir / "d02-3km" / "sbcape" / "1974-04-03"
            / delivered).is_file()
    assert not (outdir / branded).exists()
    assert (outdir / "unbranded_extra.png").is_file()
    # The per-file console lines name what is actually on disk.
    transcript = capsys.readouterr().out
    assert delivered in transcript
    assert branded not in transcript


@needs_renderer
def test_a_picture_warning_from_the_engine_reaches_the_reader(
        wrfout, tmp_path, capsys):
    """The engine says when it cut a subtitle; the render says it too.

    At the smallest size the door allows, the provenance line does not
    fit the header row and the engine cuts it, warning once on its
    stderr.  The picture still renders and the render still succeeds.
    """

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "t2", "--timeidx", "0",
                   "--size", "320x240", "--out", str(out)])
    assert rc == 0
    assert len(_delivered(out)) == 1
    err = capsys.readouterr().err
    assert err.count("warning: the left subtitle does not fit") == 1, err


@needs_renderer
def test_rust_engine_unknown_slug_fails_loudly(wrfout, tmp_path, capsys):
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "definitely_not_a_product",
                   "--out", str(tmp_path / "png")])
    # rc 1 is `woof render`'s, not the renderer's.  rw_wrfbatch exits 2 on a
    # bad command line -- matching what matplotlib's engine costs for the same
    # typo -- but woof.rustwx collects any nonzero exit as a render failure
    # and woof.cli reports failures as 1, so the two engines agree here
    # whichever one runs.  The pin is on the CLI's contract, so it is
    # unchanged by the renderer's exit code.
    assert rc == 1
    assert not list((tmp_path / "png").rglob("*.png"))
    # ...and the reason reaches the user.  woof.rustwx surfaces the LAST
    # non-empty stderr line, which the renderer used to make the usage line:
    # every arg mistake reported the same unactionable sentence.
    reported = capsys.readouterr()
    transcript = reported.out + reported.err
    assert "definitely_not_a_product" in transcript, transcript
    assert "--list-products" in transcript, transcript
    assert not transcript.rstrip().endswith("wrfout..."), (
        "the usage line is back as the reported cause", transcript)


def test_engine_rust_unbuilt_is_a_documented_refusal(
        wrfout, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("woof.rustwx.find_renderer", lambda: None)
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--out", str(tmp_path / "png")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not built" in err
    assert "cargo build --release --locked --offline" in err


def test_engine_auto_refuses_when_the_renderer_is_not_built(
        wrfout, tmp_path, monkeypatch, capsys):
    """`auto` used to degrade here; the render law says it may not.

    Weather-field product plots come from ``rw_wrfbatch`` (the render law,
    2026-08-06), and the one permitted fallback --
    ``da_nowcast_render.py`` -- draws none of this door's products.  So
    with nothing staged the answer is a refusal naming the artifact and
    the staging remedy, at a nonzero exit; drawing five matplotlib
    weather fields and reporting success is the defect (audit F7).
    """

    monkeypatch.setattr("woof.rustwx.find_renderer", lambda: None)
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--products", "t2",
                   "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "rw_wrfbatch" in err and "fetch-bridges" in err
    assert not out.exists() or not list(out.rglob("*.png"))


def test_engine_matplotlib_is_reachable_only_by_name(
        wrfout, tmp_path, capsys):
    """And says WORKAROUND every time it draws."""

    pytest.importorskip(
        "wrf", reason="the matplotlib workaround needs the render extra")
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "matplotlib",
                   "--products", "t2", "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr()
    assert "render: engine matplotlib" in printed.out
    assert "WORKAROUND:" in printed.err
    assert list(out.rglob("t2_*.png"))


# --------------------------------------------------------------------------
# Task #106, the half `--engine auto` did not cover: an EXPLICIT engine
# request must pass the same contract, or the one caller who pinned
# `--engine rust` to be sure of the real renderer is the one caller who
# still gets a foreign one.
# --------------------------------------------------------------------------

def _stage_foreign_renderer(monkeypatch, tmp_path):
    """A renderer that resolves and fails the ``--abi`` contract.

    The subprocess handshake itself is proved against real binaries --
    ``test_the_pinned_abi_marker_is_the_built_renderer_s_own_answer``
    runs this checkout's build, and a foreign build was run by hand for
    the incident.  What is pinned here is what the CALLERS do with a
    failing probe, which must be assertable on a clean checkout with no
    build staged, since that is where a regression would land.
    """

    staged = tmp_path / "bridges" / rustwx.executable_name(
        rustwx.RENDERER_NAME)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"MZ not this tree's build")
    evidence = ("launches, but --abi does not match the render contract "
                "this woof expects (exit 2 with no --abi line) -- it is a "
                "build from another checkout")
    monkeypatch.setattr("woof.rustwx.find_renderer", lambda: staged)
    monkeypatch.setattr("woof.rustwx.probe_renderer",
                        lambda path: (False, evidence))
    return staged, evidence


def test_engine_rust_refuses_a_renderer_that_fails_the_contract(
        wrfout, tmp_path, monkeypatch, capsys):
    staged, _ = _stage_foreign_renderer(monkeypatch, tmp_path)
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "t2", "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--engine rust" in err
    assert str(staged) in err
    assert "--abi does not match the render contract" in err
    # An explicit request is a statement about which engine must draw:
    # no silent substitution, and no fallback either.
    assert not out.exists() or not list(out.rglob("*.png"))
    # ... and the exit is named (1.8.8 refusal sweep).  Refusing to
    # choose for the caller is not the same as refusing to tell them
    # what the choices are.  `--engine auto` is no longer one of those
    # choices: it refuses this same renderer now, because degrading to
    # matplotlib weather fields is what the render law forbids.
    assert "cargo build --release" in err
    assert "woof fetch-bridges" in err
    assert "--engine matplotlib to take the named workaround" in err
    assert "fall back" not in err


def test_a_missing_renderer_override_names_the_three_ways_out(
        tmp_path, monkeypatch, capsys):
    """The other dead end on the same door: the variable names nothing.

    ``WOOF_RW_WRFBATCH`` pointing at a path that does not exist used to
    refuse with the variable, the path, and nothing else -- no
    suggestion to repoint it, to unset it and take the vendored ladder,
    or to ask for the fallback engine.  A reader who inherited that
    variable from a shell profile had a diagnosis and no instruction.
    """

    missing = tmp_path / "nonexistent-renderer.exe"
    monkeypatch.setenv(rustwx.RENDERER_ENV, str(missing))
    assert cli.main(["render", "--list-products", "--engine", "rust"]) == 2

    err = capsys.readouterr().err
    assert "names a missing file" in err
    assert str(missing) in err
    assert "Point it at a built rw_wrfbatch binary" in err
    assert f"unset {rustwx.RENDERER_ENV}" in err
    assert "--engine matplotlib" in err


def test_engine_auto_refuses_the_same_renderer_engine_rust_refuses(
        wrfout, tmp_path, monkeypatch, capsys):
    """Same staged renderer, two request forms, ONE outcome now.

    This is the pair the original defect was invisible to: with a stale
    bridge staged, ``auto`` fell back naming the mismatch while ``rust``
    ran the stale build without a word.  The contract check closed the
    second half; the render law closes the first, because "fell back"
    meant matplotlib drew the weather fields.  What must stay asserted
    is that both forms still NAME the mismatch -- a refusal that hides
    which binary was rejected is the outage the probe exists to prevent.
    """

    _stage_foreign_renderer(monkeypatch, tmp_path)
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "auto",
                   "--products", "t2", "--out", str(out)])
    assert rc == 2
    printed = capsys.readouterr()
    assert "--abi does not match the render contract" in (
        printed.out + printed.err)
    assert not out.exists() or not list(out.rglob("*.png"))


def test_engine_rust_accepts_the_renderer_that_passes_the_contract(
        tmp_path, monkeypatch):
    """The other direction: a passing probe still resolves to rust.

    A refusal that fires on everything is not a contract check, it is an
    outage.  Both request forms take the rust path on a probe that
    passes, and neither consults anything else to decide.
    """

    from woof.render import _resolve_engine

    staged = tmp_path / rustwx.executable_name(rustwx.RENDERER_NAME)
    staged.write_bytes(b"MZ this tree's build")
    monkeypatch.setattr("woof.rustwx.find_renderer", lambda: staged)
    # The renderer gate has TWO clauses since 1.8.8 and this test is
    # about the first one, so the second is satisfied here the same way
    # the rebranding sibling satisfies it: the CONTRACT half is stubbed
    # passing, and the PROVENANCE half is stubbed silent because a
    # binary under tmp_path belongs to no tree and the bridge gate is
    # right to refuse one nobody named.  Left unstubbed, whether this
    # test passes depends on how the RUNNER was installed -- a wheel or
    # a non-git source root makes bridge_tree_match unanswerable and the
    # test green, an editable checkout makes the same tmp_path binary
    # foreign and the test red -- which is a property of the box, not of
    # the code under test.
    monkeypatch.setattr("woof.rustwx.probe_renderer",
                        lambda path: (True, "--abi matches"))
    monkeypatch.setattr("woof.provenance_gate.renderer_bridge_refusal",
                        lambda bridge, **kwargs: None)
    assert _resolve_engine("rust") == ("rust", str(staged))
    assert _resolve_engine("auto") == ("rust", str(staged))

    monkeypatch.setattr("woof.rustwx.probe_renderer",
                        lambda path: (False, "stale"))
    # Both request forms refuse a failing contract now, and both name
    # the evidence.  `auto` used to answer ("matplotlib", ...) here.
    with pytest.raises(RuntimeError, match="stale"):
        _resolve_engine("auto")
    with pytest.raises(RuntimeError, match="stale"):
        _resolve_engine("rust")


def test_the_product_catalog_refuses_an_engine_it_cannot_verify(
        tmp_path, monkeypatch, capsys):
    """``--list-products`` with no wrfout resolves the engine of its own.

    It is the one entry point that reaches ``_resolve_engine`` outside
    the render path's try, so the refusal has to be caught here too or a
    documented refusal arrives as a traceback.
    """

    _stage_foreign_renderer(monkeypatch, tmp_path)
    assert cli.main(["render", "--list-products", "--engine", "rust"]) == 2
    assert "--abi does not match the render contract" in (
        capsys.readouterr().err)


def test_both_engine_resolvers_treat_an_explicit_request_the_same(
        tmp_path, monkeypatch):
    """The renderer was the odd one out here too; hold the pair together.

    ``woof fetch`` has always resolved ``--engine rust`` by probing and
    then refusing an explicit request while degrading an automatic one.
    The render resolver skipped the probe for an explicit request, which
    is the whole defect.  Asserting both in one place is what makes the
    next divergence a red rather than a discovery in the field.

    The two doors diverge on ONE point deliberately, and it is a law and
    not an oversight: ``fetch --engine auto`` still degrades to the
    Python engine, because moving bytes on Python is a workaround, while
    ``render --engine auto`` refuses, because DRAWING a weather field on
    matplotlib is not permitted at all (the render law,
    2026-08-06).
    """

    from woof import fetch as fetch_module
    from woof.render import _resolve_engine

    # render: staged binary, failing contract.  Both forms refuse.
    _stage_foreign_renderer(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        _resolve_engine("rust")
    with pytest.raises(RuntimeError):
        _resolve_engine("auto")

    # fetch: staged backbone, failing contract, same two answers.
    backbone = tmp_path / "rw_fetch"
    backbone.write_bytes(b"MZ not this tree's build")
    monkeypatch.setattr("woof.rustwx_fetch.find_fetch_bin",
                        lambda: backbone)
    monkeypatch.setattr("woof.rustwx_fetch.probe_fetch_bin",
                        lambda path: (False, "abi mismatch"))
    with pytest.raises(ValueError):
        fetch_module.resolve_fetch_engine("rust", progress=lambda *a: None)
    assert fetch_module.resolve_fetch_engine(
        "auto", progress=lambda *a: None)[0] == "python"


def test_doctor_reports_a_renderer_that_fails_the_contract(
        tmp_path, monkeypatch):
    """The doctor branch for a foreign build, without staging one.

    ``test_doctor_reports_the_rust_renderer`` covers this outcome only
    when a foreign build happens to be resolvable, which on a clean
    checkout it is not -- so the branch that reports a stale bridge went
    unexecuted in the ordinary battery, which is exactly where a
    regression would be caught.
    """

    from woof.doctor import _rust_renderer_check

    staged, evidence = _stage_foreign_renderer(monkeypatch, tmp_path)
    check = _rust_renderer_check()
    assert check.status == "missing"
    assert check.blocking is False
    assert str(staged) in check.detail
    assert evidence in check.detail
    assert check.remedy and "cargo build" in check.remedy


def test_parse_products_rust_mapping():
    # String equality only.  This test PASSED while `wind10` returned a
    # pressure chart, which is why it is not the guard against that bug
    # -- `test_wind10_renders_the_wind_not_a_pressure_chart` is, by
    # going to the rendered artifact and to the renderer's own statement
    # of what each product fills with.
    assert parse_products_rust("all") == "all"
    assert parse_products_rust("refl,t2,wind10,precip") == (
        "composite_reflectivity,2m_temperature,"
        "10m_wind_speed_and_direction,total_qpf")
    # Raw catalog slugs pass through for the renderer's own validation.
    assert parse_products_rust("sbcape,refl") == (
        "sbcape,composite_reflectivity")
    assert parse_products_rust("refl,refl") == "composite_reflectivity"
    with pytest.raises(ValueError, match="no products"):
        parse_products_rust(",")


def test_parse_size():
    assert parse_size("1200x900") == (1200, 900)
    assert parse_size("800X600") == (800, 600)
    for bad in ("1200", "axb", "100x100", "1200x"):
        with pytest.raises(ValueError):
            parse_size(bad)


# ---------------------------------------------------------------------------
# --list-products: the catalog with per-file availability
# ---------------------------------------------------------------------------

@needs_renderer
def test_list_products_reports_the_full_catalog(wrfout, tmp_path, capsys):
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--list-products", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # The complete catalog is enumerated, not just what renders.
    # 364 = 163 + the standalone 10 m wind chart + this fixture's 23
    # generic ``var:`` rows (stored 2-D planes no named product claims;
    # the generic family is store-dependent, so the count is the
    # FIXTURE's, not the build's) + the 172 ensemble/probabilistic
    # rows, which stay outside ``all`` but are LISTED with their field
    # truth and the opt-in code instead of being skipped, so a reader
    # can see what naming one of them would need.  It was 168 with 15
    # generic rows until the science import started carrying EVERY
    # stored ``(Time, south_north, west_east)`` plane -- the eight rows
    # that added are this fixture's own surface planes, which the two
    # fixed catalogs did not name and which therefore used to be
    # unrenderable.  The named rows went from 152 to 163 when the
    # wrfout import gained the eleven column products (three isotherm
    # heights, three supercooled water paths, five hydrometeor column
    # maxima); ``rw_wrfbatch --list-products`` counts the same eleven,
    # ``selectable_slugs`` 322 to 333. Five optional AQ rows add surface
    # smoke, dry PM2.5, dust, ozone and aerosol optical depth; the existing
    # canonical smoke and column recipes remain in the original catalog.
    # The coupled fire's three maps (fire_perimeter, fire_ros,
    # fire_heat_flux, development branch) are listed too, excluded on this
    # fixture because it carries no refined fire grid: 364 + 3 = 367.
    # Three interval winter products add snowfall_window, snow_10to1_window
    # and snow_kuchera_window to that complete catalog: 367 + 3 = 370.
    # The radiation cloud-fraction column maximum (cldfra_max, golden/ports-rad,
    # 2.9.0) is the twelfth column product: 370 + 1 = 371.
    # The Earth-2 export (189e1776dc6) adds three direct products,
    # 100m_wind_speed, 500mb_omega and 850mb_specific_humidity (rw_wrfbatch
    # --list-products: selectable_slugs 345 -> 348): 371 + 3 = 374.
    # The 2.9.4 capability lanes add 35 named rows (selectable_slugs 348 ->
    # 382): cap-ivt's eleven (ivt twice, ivt_vectors, ivt_full_column, the
    # four eastward/northward components, iwv, ar_outline, freezing_level),
    # cap-hail's eight hail_max_* maps, the heat/fire/lightning diagnostics'
    # nine (haines_low/mid/high, hot_dry_windy, lightning_threat_1..3,
    # pr92_flash_rate, wbgt_2m), cap-snow's three (snow_depth_change,
    # snow_depth_total, snowfall_liquid_equivalent), cap-fire-fuels'
    # fire_fuel and fire_terrain, cap-visibility's fog_flag and the hex-ivt
    # engine's 925mb_height_winds: 374 + 35 = 409.
    # The hex IVT companion's model-column recipe (bc8db45d3fe) renamed its
    # duplicate "ivt" to ivt_column and gave its two component planes their
    # own rows, ivt_column_eastward and ivt_column_northward: 409 + 2 = 411.
    # The two fixed-height W products from development branch
    # (1km_vertical_velocity, 3km_vertical_velocity; catalog fixture
    # b0790a03c8): 411 + 2 = 413.
    # The mp55 (ISHMAEL) habit columns add ice_water_path (d765178a42,
    # a blocked catalog row on this fixture): 413 + 1 = 414.
    # The 0-1/0-2/0-3 km UH layers (dce81682858, 3.0.0) add ten direct
    # recipes: uh_0to1km/0to2km/0to3km, composite_reflectivity_uh0to1km/
    # 0to2km/0to3km and 1km_reflectivity_uh2to5km/0to1km/0to2km/0to3km:
    # 414 + 10 = 424.
    assert "total=424" in out
    for slug in ("uh_0to1km", "uh_0to2km", "uh_0to3km",
                 "composite_reflectivity_uh0to1km",
                 "composite_reflectivity_uh0to2km",
                 "composite_reflectivity_uh0to3km",
                 "1km_reflectivity_uh2to5km", "1km_reflectivity_uh0to1km",
                 "1km_reflectivity_uh0to2km", "1km_reflectivity_uh0to3km"):
        rows = [line for line in out.splitlines()
                if len(line.split()) >= 3 and line.split()[2] == slug]
        assert len(rows) == 1, (slug, rows)
    for slug in ("ivt", "ivt_column", "ivt_column_eastward",
                 "ivt_column_northward"):
        rows = [line for line in out.splitlines()
                if len(line.split()) >= 3 and line.split()[2] == slug]
        assert len(rows) == 1, (slug, rows)
    for slug in ("snowfall_window", "snow_10to1_window", "snow_kuchera_window"):
        rows = [line for line in out.splitlines()
                if len(line.split()) >= 3 and line.split()[2] == slug]
        assert len(rows) == 1, (slug, rows)
    assert "renderable" in out and "excluded" in out
    for slug in ("smoke_near_surface", "pm25_near_surface",
                 "dust_near_surface", "ozone_near_surface", "aod_550"):
        rows = [line for line in out.splitlines()
                if len(line.split()) >= 3 and line.split()[2] == slug]
        assert len(rows) == 1, (slug, rows)
        assert rows[0].lstrip().startswith("missing-fields"), rows[0]
        assert "not stored:" in rows[0], rows[0]
    for slug in ("smoke_pm25_native", "smoke_column"):
        assert any(len(line.split()) >= 3 and line.split()[2] == slug
                   for line in out.splitlines()), slug
    # The generic rows are part of the catalog, not a side channel: every
    # stored plane without a named product renders as ``var:<name>``.
    #
    # Twenty-nine rows print where the engine's own catalog holds
    # twenty-three of them.  The other six are the planes that ARE a
    # named product's sole source: the engine leaves those out of its
    # catalog, because listing one would draw the same grid twice under
    # two slugs, and prints each on stderr as ``GENERIC_EXCLUDED ...
    # already rendered by a named product``.  The door folds them back
    # into the listing as renderable (``generic-deduped``) so that a
    # request for a spelling this build accepts is not refused by the
    # listing that describes it.  They are also why ``total=`` still
    # says 364 while 370 rows print: the total is the ENGINE's catalog
    # and these six are the door's addition to it.
    generic_rows = [line for line in out.splitlines()
                    if " generic " in line and " var:" in line]
    deduped = [line for line in generic_rows
               if "already rendered by a named product" in line]
    #
    # 33 and 10 in 2.9.4: the heat/fire diagnostics lane (8c96bae6bf2) stores
    # heat_index_2m, apparent_temperature_2m, wetbulb_2m and
    # fire_weather_composite planes, each the sole source of its named
    # product, so each prints as one more deduplicated generic row.
    # 34 and 11: development branch (51fabe790f, merged 054770cd6e)
    # stores the native WRF vpd_2m plane, the sole source of its named VPD
    # product and of the day-window VPD rows, so it is one more
    # deduplicated generic row.
    assert len(generic_rows) == 34, out
    assert len(deduped) == 11, deduped
    assert sorted(re.search(r"var:(\S+)", line).group(1)
                  for line in deduped) == [
        "apcp", "apparent_temperature_2m", "composite_reflectivity",
        "dewpoint_2m", "fire_weather_composite", "heat_index_2m", "orography",
        "relative_humidity_2m", "temperature_2m", "vpd_2m",
        "wetbulb_2m"], deduped
    assert all("renderable" in line for line in generic_rows), generic_rows
    # The fixture's fields prove out the reflectivity composite ...
    assert any("composite_reflectivity" in line and "renderable" in line
               for line in out.splitlines())
    # ... and on its exact-time (half-hourly) axis the fixed-hour windows
    # are blocked with the reason spelled out: its frames end at +30 min,
    # so no whole hour of it closes a window.
    assert any("qpf_total" in line and "blocked" in line
               and "stored frames end at F000" in line
               for line in out.splitlines())
    # Every non-renderable row carries a reason string.
    for line in out.splitlines():
        if line.lstrip().startswith("missing-fields"):
            assert "not stored:" in line, line


@needs_renderer
def test_terrain_product_is_renderable_from_a_wrfout(wrfout, tmp_path, capsys):
    """Orography is in every wrfout; the catalog must offer it as a product.

    The negative control this pins is a silent skip: before the recipe
    existed the terrain plane sat in the store as a browse-only raw field
    and no catalog row mentioned it at all, so "there is no terrain frame"
    and "the terrain frame failed" were indistinguishable.
    """
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--list-products", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if "terrain_height" in line]
    assert rows, out
    assert all("renderable" in row for row in rows), rows


@needs_renderer
@pytest.mark.parametrize("single_frame", [False, True],
                         ids=["minute-axis", "first-frame"])
def test_general_products_skip_unavailable_subhour_windows(
        single_frame, wrfout, tmp_path, capsys):
    """The default TUI request must not fail a good subhour forecast.

    Its one-hour rain product needs a whole-hour window. The real native
    availability catalog must supply that skip; no shorter accumulation
    may be drawn or labelled as an hour, and available products still run.
    """

    import json
    from woof import render as render_module

    presets = json.loads((Path(render_module.__file__).parent /
                          "data/tui/plot-presets.json").read_text())
    general = next(row["products"] for row in presets["presets"]
                   if row["id"] == presets["default"])
    # 22: simulated_ir_satellite left the general and hurricane presets
    # when the lane record gained its reason, and 10m_wind_gusts and
    # precipitation_type left general when every run of it was measured
    # drawing 20 of 24 (no wrfout carries their fields).  26 in 2.9.4: cap-ivt
    # added ivt, iwv, freezing_level and ar_outline, drawn from the stored
    # moisture, wind and geopotential every history carries.
    assert len(general) == 26 and "qpf_1h" in general
    if single_frame:
        wrfout = _write_wrfout(tmp_path / "first-wrfout.nc", _STAMPS[:1])
    frame_idx = 0 if single_frame else 1
    reason = ("more than one stored whole-hour frame" if single_frame
              else "stored frames end at F000")
    out = tmp_path / "general"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", ",".join(general), "--timeidx", str(frame_idx),
                   "--size", "400x300", "--out", str(out),
                   "--run-stamp", "off", "--explain"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    produced = _delivered(out)
    assert any("composite_reflectivity" in name for name in produced)
    if single_frame:
        assert all(name.endswith("_f000.png") for name in produced), produced
    else:
        assert all(_LEADS[frame_idx] in name for name in produced), produced
    assert not any("qpf_1h" in name for name in produced), produced
    assert "qpf_1h" in captured.err and reason in captured.err
    assert "render FAIL:" not in captured.err


@needs_renderer
def test_only_unavailable_subhour_window_still_returns_nonzero(
        wrfout, tmp_path, capsys):
    """A named skip is not a successful command when no image was drawn."""

    out = tmp_path / "unavailable-only"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "qpf_1h", "--out", str(out),
                   "--run-stamp", "off", "--explain"])
    captured = capsys.readouterr()
    assert rc == 1 and not _delivered(out)
    assert ("qpf_1h" in captured.err
            and "stored frames end at F000" in captured.err), captured.err
    assert "Nothing else was drawn" in captured.err
    assert "render FAIL:" not in captured.err


@needs_renderer
def test_native_exact_time_window_is_never_drawn_short(wrfout, tmp_path):
    """A half-hour store holds no hour: the engine draws no 1 h window.

    The engine serves windows on an exact-time axis from each frame's
    lead, and one ends only on a whole-hour frame with the frame an hour
    before it stored.  This fixture's frames are at +0 and +30 min, so
    the one window anchor (+0) is refused by name, the +30 min frame is
    no window anchor at all, and nothing shorter is drawn as an hour.
    """

    written, failures, skipped = rustwx.run_renderer(
        RENDERER, wrfout, store_root=tmp_path / "strict-store",
        out_dir=tmp_path / "strict-png", products="qpf_1h", frames="all",
        width=400, height=300)
    assert not written
    assert [slug for slug, _reason in skipped] == ["qpf_1h"], skipped
    assert "F000:" in skipped[0][1] and "forecast hour >= 1" in skipped[0][1]
    # A render that drew nothing still exits nonzero, and its count says
    # it was one skip and no failed item.
    assert failures and all("rendered=0 skipped=1 failed=0" in row
                            for row in failures), failures


@needs_renderer
@pytest.mark.parametrize("problem", ["unknown-product", "corrupt-input"])
def test_window_availability_does_not_hide_real_render_failures(
        problem, wrfout, tmp_path, capsys):
    """Only declared time-axis unavailability may become a skip."""

    source = wrfout
    products = "composite_reflectivity,qpf_1h"
    if problem == "unknown-product":
        products += ",not_a_registered_weather_product"
    else:
        source = tmp_path / "broken-wrfout.nc"
        source.write_bytes(b"not a NetCDF file")
    out = tmp_path / "failed"
    rc = cli.main(["render", str(source), "--engine", "rust",
                   "--products", products, "--out", str(out),
                   "--run-stamp", "off", "--explain"])
    captured = capsys.readouterr()
    assert rc != 0 and "render FAIL:" in captured.err
    assert not _delivered(out)


@needs_renderer
def test_list_products_whole_hour_axis_serves_windowed(
        wrfout_hourly, tmp_path, capsys):
    rc = cli.main(["render", str(wrfout_hourly), "--engine", "rust",
                   "--list-products", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # Run-total QPF folds the wrfout lane's apcp plane; the 1 h wind max
    # realizes through the documented hypot(u_10m, v_10m) fallback.
    assert any("qpf_total" in line and "renderable" in line
               for line in out.splitlines()), out
    assert any("10m_wind_1h_max" in line and "renderable" in line
               for line in out.splitlines()), out
    # 24/48 h windows stay blocked with their window arithmetic.
    assert any("2m_temp_0_24h_max" in line and "blocked" in line
               for line in out.splitlines()), out


@needs_renderer
def test_an_hourly_wind_snapshot_is_not_listed_as_a_maximum(
        wrfout_hourly, tmp_path):
    """A wrfout with no sub-hourly 10 m wind maximum (no WSPD10MAX, which
    is every WOOF history and this fixture's) gives the 1 h and run
    maxima on whole hours only top-of-hour snapshots.  They were drawn
    under "(1 h max)" and "(run max)" titles and only the catalog detail
    said "lower bound"; the detail now names the fold and why it is not
    the maximum, and the picture is titled the same way (``rusty-weather``
    ``windowed_store`` tests pin the title)."""

    rows, _summary = rustwx.list_products(
        RENDERER, wrfout_hourly, store_root=tmp_path / "wind-store")
    detail = {slug: (status, detail) for slug, _kind, status, detail in rows}
    for slug in ("10m_wind_1h_max", "10m_wind_run_max"):
        status, text = detail[slug]
        assert status == "renderable", (slug, text)
        assert "top-of-hour 10 m wind speed" in text, (slug, text)
        assert "WSPD10MAX" in text and "not the maximum" in text, (slug, text)
        assert "stored sub-hourly 1 h max" not in text, (slug, text)


@needs_renderer
def test_a_wrfout_that_stores_wspd10max_draws_the_wind_maximum(tmp_path):
    """WRF written with nwp_diagnostics = 1 stores WSPD10MAX, the 10 m wind
    maximum over each history interval, and the import keeps it as
    ``wrf_wspd10max``.  The 10 m wind windows read only the GRIB lane's
    ``wind_speed_10m_max_1h`` and then top-of-hour U10/V10 speeds, so such
    a history was drawn as snapshots under "no stored max" titles and its
    detail said it stores no WSPD10MAX.  The windows now read it as the
    maximum it is and keep their maximum titles (``rusty-weather``
    ``windowed_store`` tests pin the titles and values)."""

    netCDF4 = pytest.importorskip("netCDF4")
    path = _write_wrfout(
        tmp_path / "wrfout_d02_1974-04-03_18-00-00.nc",
        ("1974-04-03_18:00:00", "1974-04-03_19:00:00"))
    with netCDF4.Dataset(path, "a") as dataset:
        speed = np.hypot(dataset["U10"][:], dataset["V10"][:])
        wspd10max = dataset.createVariable(
            "WSPD10MAX", "f4", ("Time", "south_north", "west_east"))
        wspd10max.setncattr("FieldType", np.int32(104))
        wspd10max.MemoryOrder = "XY "
        wspd10max.description = "WIND SPD MAX 10 M"
        wspd10max.units = "m s-1"
        wspd10max.stagger = ""
        wspd10max[:] = (speed + 2.5).astype(np.float32)

    rows, _summary = rustwx.list_products(
        RENDERER, path, store_root=tmp_path / "wind-store")
    detail = {slug: (status, detail) for slug, _kind, status, detail in rows}
    for slug in ("10m_wind_1h_max", "10m_wind_run_max"):
        status, text = detail[slug]
        assert status == "renderable", (slug, text)
        assert "WRF WSPD10MAX per-history-interval max at F001" in text, (
            slug, text)
        for wrong in ("snapshot", "hypot", "not the maximum", "lower bound",
                      "stores no"):
            assert wrong not in text, (slug, wrong, text)


@needs_renderer
def test_the_render_after_the_availability_listing_imports_nothing(
        wrfout_hourly, tmp_path, monkeypatch):
    """One import per render, through the door.

    ``woof render`` asks the catalog what the frames can draw and then
    draws them: two launches into one scratch store.  Each imported every
    frame, and the listing imported in full: nine long windows over four
    750 m frames cost 219 CPU-s through the door against 11 CPU-s in the
    renderer alone, and the listing was about 26 of the 34.7 CPU-s of
    every live frame.  The listing of named products imports them as their
    render does, and the render finds that run in the store.
    """

    transcripts: list[str] = []
    real = rustwx.subprocess.run

    def record(command, *args, **kwargs):
        result = real(command, *args, **kwargs)
        if "--store-root" in command:
            transcripts.append(result.stdout or "")
        return result

    monkeypatch.setattr(rustwx.subprocess, "run", record)
    out = tmp_path / "door"
    rc = cli.main(["render", str(wrfout_hourly), "--engine", "rust",
                   "--products", "qpf_1h,10m_wind_1h_max,2m_temperature",
                   "--out", str(out), "--run-stamp", "off"])
    assert rc == 0 and _delivered(out)
    assert len(transcripts) == 2, "one availability listing, one render"
    listing, draw = transcripts
    assert "PROCESS Opening WRF " in listing
    # The listing drops the unavailable wind window, changing the scoped
    # field set. Its saved import still supplies all surviving products.
    assert "IMPORTED from " in draw and "(no frame reopened)" in draw, draw
    assert "PROCESS Opening WRF " not in draw, draw
    assert "PROCESS Computing WRF " not in draw, draw


@needs_renderer
def test_list_products_rejects_identity_gated_rows(
        wrfout, wrfout_hourly, tmp_path, capsys):
    """Architectural guard: availability rows must justify themselves
    with FIELDS, never with the source model's identity.  A 'gated'
    status, a 'no fetch plan' reason, or any 'for model ...' wording is
    a regression -- a product that cannot render always names the
    fields it is missing."""
    allowed = {"renderable", "missing-fields", "blocked", "excluded"}
    for path in (wrfout, wrfout_hourly):
        rc = cli.main(["render", str(path), "--engine", "rust",
                       "--list-products", "--out", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        rows = [line for line in out.splitlines()
                if line.startswith("  ") and line.split()]
        assert rows, out
        for line in rows:
            status = line.split()[0]
            assert status in allowed, line
            low = line.lower()
            assert "gated" not in low, line
            assert "fetch plan" not in low, line
            assert "for model" not in low, line
        assert "gated=" not in out
        # The families the old model-identity gate hid (smoke, simulated
        # IR, categorical precip) now carry accurate field reasons.
        for slug in ("smoke_column", "simulated_ir_satellite",
                     "precipitation_type"):
            assert any(slug in line and "missing-fields" in line
                       and "not stored:" in line for line in rows), \
                (slug, out)


@needs_renderer
@pytest.mark.parametrize("heavy", [False, True], ids=["no-heavy", "heavy"])
def test_excluded_heavy_rows_each_name_their_own_missing_grid(
        heavy, wrfout, tmp_path):
    """Architectural guard: one reason per excluded heavy row.

    The catalog used to print a single literal for the whole ECAPE
    family, so a permanent exclusion (the three native-CAPE ratio pairs,
    which divide by the source model's own decoded CAPE plane and can
    never come off a wrfout) was indistinguishable from a per-hour input
    gap a re-import fixes.  Every excluded heavy row must name the grid
    it is missing and a way out, and no two may say the same thing.
    """

    rows, _summary = rustwx.list_products(
        RENDERER, wrfout, store_root=tmp_path / "heavy-store", heavy=heavy)
    excluded = [(slug, detail) for slug, kind, status, detail in rows
                if kind == "heavy" and status == "excluded"]
    assert excluded, [row for row in rows if row[1] == "heavy"]
    details = [detail for _slug, detail in excluded]
    assert len(set(details)) == len(details), (
        "excluded heavy rows share a reason string: "
        + repr(sorted(d for d in details if details.count(d) > 1)))
    for slug, detail in excluded:
        assert slug in detail, (slug, detail)
        assert "--heavy" in detail or "GRIB" in detail, (slug, detail)


@needs_renderer
@pytest.mark.parametrize("product", ["qpf_total", "qpf_1h"])
def test_windowed_products_render_on_whole_hour_stores(
        product, wrfout_hourly, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout_hourly), "--engine", "rust",
                   "--products", product, "--out", str(out)])
    assert rc == 0
    produced = _delivered(out)
    assert any(product in name for name in produced), produced


@needs_renderer
def test_split_history_window_matches_the_combined_native_render(tmp_path, capsys):
    """Hourly split output must draw the same native QPF as one multi-frame file."""
    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00")
    first = _write_wrfout(tmp_path / "wrfout_d02_first.nc", stamps[:1])
    last = _write_wrfout(tmp_path / "wrfout_d02_last.nc", stamps[1:], seed_offset=1)
    combined = _write_wrfout(tmp_path / "combined.nc", stamps)
    import netCDF4
    for path, source, updated in ((first, "initial_forcing", 0.), (last, "radiation_scheme", 2880.)):
        with netCDF4.Dataset(path, "a") as dataset:
            dataset.GPUWM_CARRIER_GLW_SOURCE = source
            dataset.GPUWM_CARRIER_SWDOWN_SOURCE = source
            dataset.GPUWM_CARRIER_GLW_LAST_UPDATE = updated
            dataset.GPUWM_CARRIER_SWDOWN_LAST_UPDATE = updated
            dataset.GPUWM_SURFACE_RADIATION_POLICY = "required"
    common = ["--engine", "rust", "--products", "qpf_1h", "--size", "400x300",
              "--run-stamp", "off"]
    isolated = tmp_path / "isolated"
    assert cli.main(["render", str(first), str(last), *common,
                     "--out", str(isolated)]) == 1
    assert not _delivered(isolated)
    assert cli.main(["render", str(last), str(first), "--series", "--engine", "rust",
                     "--list-products"]) == 0
    listed = capsys.readouterr().out
    assert any("qpf_1h" in line and "renderable" in line for line in listed.splitlines())
    outputs = []
    for label, paths, flags in (("split", [last, first], ["--series"]),
                                ("combined", [combined], [])):
        out = tmp_path / label
        rc = cli.main(["render", *map(str, paths), *common, *flags, "--out", str(out)])
        captured = capsys.readouterr()
        assert rc == 0, captured.out + captured.err
        outputs.append({path.relative_to(out).as_posix(): path.read_bytes()
                        for path in out.rglob("*.png")})
    assert outputs[0] and outputs[0] == outputs[1]


def test_series_grouping_keeps_actual_runs_grids_and_episodes_separate(tmp_path):
    import netCDF4
    from woof.render import group_history_series

    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00")
    first = _write_wrfout(tmp_path / "wrfout_d02_first.nc", stamps[:1])
    last = _write_wrfout(tmp_path / "wrfout_d02_last.nc", stamps[1:])
    other_domain = _write_wrfout(tmp_path / "wrfout_d03.nc", stamps[:1], grid_id=3)
    moved = _write_wrfout(tmp_path / "wrfout_d02_moved.nc", stamps[:1])
    with netCDF4.Dataset(moved, "a") as dataset:
        dataset.variables["XLAT"][0, 0, 0] += np.float32(0.25)
    other_start = _write_wrfout(tmp_path / "wrfout_d02_other_start.nc", stamps[:1])
    with netCDF4.Dataset(other_start, "a") as dataset:
        dataset.START_DATE = "1974-04-03_17:00:00"
    independent = []
    for folder in ("another-run", "episode-002"):
        directory = tmp_path / folder
        directory.mkdir()
        independent.append(_write_wrfout(directory / "wrfout_d02.nc", stamps[:1]))
    groups = group_history_series([last, other_domain, moved, other_start, *independent, first])
    assert [first, last] in groups
    assert {tuple(group) for group in groups if len(group) == 1} == {
        (path,) for path in (other_domain, moved, other_start, *independent)}
    duplicate = _write_wrfout(tmp_path / "wrfout_d02_duplicate.nc", stamps[:1])
    with pytest.raises(ValueError, match="overlapping valid times"):
        group_history_series([first, duplicate])


@pytest.mark.parametrize("attribute,value", [
    ("GPUWM_INITIAL_CONDITION_SOURCE", "different-source"),
])
def test_carrier_updates_do_not_join_different_source_or_scientific_histories(tmp_path, attribute, value):
    import netCDF4
    from woof.render import group_history_series
    first = _write_wrfout(tmp_path / "wrfout_d02_first.nc", ("1974-04-03_18:00:00",))
    last = _write_wrfout(tmp_path / "wrfout_d02_last.nc", ("1974-04-03_19:00:00",))
    for index, path in enumerate((first, last)):
        with netCDF4.Dataset(path, "a") as dataset:
            dataset.GPUWM_CARRIER_GLW_SOURCE = "initial_forcing" if index == 0 else "radiation_scheme"
            dataset.GPUWM_CARRIER_GLW_LAST_UPDATE = index * 2880.
            dataset.GPUWM_INITIAL_CONDITION_SOURCE = "era5"
            dataset.GPUWM_INITIAL_CONDITION_CYCLE = "1974-04-03_18:00:00"
            dataset.GPUWM_INITIAL_FORECAST_LEAD_HOURS = 0
            dataset.MP_PHYSICS = 8
            dataset.GPUWM_SURFACE_RADIATION_POLICY = "required"
    assert group_history_series([last, first]) == [[first, last]]
    with netCDF4.Dataset(last, "a") as dataset:
        dataset.setncattr(attribute, value)
    assert {tuple(group) for group in group_history_series([first, last])} == {(first,), (last,)}


def _write_moved_wrfout(path, stamps, *, i_parent_start, seed_offset=0):
    """A frame of the same nest as :func:`_write_wrfout`, moved east.

    One parent cell at ratio 3 is three nest cells, and the coordinates
    move with the place exactly as a relocated nest's do.
    """
    import netCDF4

    _write_wrfout(path, stamps, seed_offset=seed_offset)
    shift = 3 * (i_parent_start - 5)
    step = np.float32(3.0 / (_NX - 1))
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.I_PARENT_START = np.int32(i_parent_start)
        dataset.CEN_LON = np.float32(dataset.CEN_LON + shift * step)
        dataset.variables["XLONG"][:] = (
            dataset.variables["XLONG"][:] + np.float32(shift) * step)
    return path


def test_a_moving_nests_earlier_places_close_its_later_places_windows(tmp_path):
    """THE DEFECT: the series was split by place, so the first window after
    a move had no earlier frame and was never drawn.  Frames of one nest at
    an earlier place now join each later place's series as context."""

    import netCDF4
    from woof.render import group_history_series, history_series_groups

    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00",
              "1974-04-03_20:00:00", "1974-04-03_21:00:00")
    first = _write_wrfout(tmp_path / "wrfout_d02_1974-04-03_18:00:00", stamps[:1])
    second = _write_wrfout(tmp_path / "wrfout_d02_1974-04-03_19:00:00", stamps[1:2],
                           seed_offset=1)
    third = _write_moved_wrfout(tmp_path / "wrfout_d02_1974-04-03_20:00:00",
                                stamps[2:3], i_parent_start=6, seed_offset=2)
    fourth = _write_moved_wrfout(tmp_path / "wrfout_d02_1974-04-03_21:00:00",
                                 stamps[3:], i_parent_start=7, seed_offset=3)
    groups = history_series_groups([fourth, third, second, first])
    assert ([first, second], []) in groups
    assert ([first, second, third], [first, second]) in groups
    assert ([first, second, third, fourth], [first, second, third]) in groups
    assert len(groups) == 3

    # Another run of the same nest (another start) never joins, and
    # neither does another folder: only the place may differ.
    other = _write_moved_wrfout(tmp_path / "other_start", stamps[2:3],
                                i_parent_start=6)
    with netCDF4.Dataset(other, "a") as dataset:
        dataset.START_DATE = "1974-04-03_17:00:00"
    elsewhere = tmp_path / "another-run"
    elsewhere.mkdir()
    apart = _write_moved_wrfout(elsewhere / "wrfout_d02", stamps[2:3],
                                i_parent_start=6)
    assert [other] in group_history_series([first, second, other])
    assert [apart] in group_history_series([first, second, apart])


@needs_renderer
def test_the_first_hour_after_a_move_draws_its_1h_rain(tmp_path, capsys):
    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00")
    first = _write_wrfout(tmp_path / "wrfout_d02_1974-04-03_18:00:00", stamps[:1])
    moved = _write_moved_wrfout(tmp_path / "wrfout_d02_1974-04-03_19:00:00",
                                stamps[1:], i_parent_start=6, seed_offset=1)
    out = tmp_path / "png"
    rc = cli.main(["render", str(first), str(moved), "--engine", "rust", "--series",
                   "--products", "qpf_1h", "--size", "400x300", "--run-stamp", "off",
                   "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    drawn = sorted(path.name for path in out.rglob("*.png") if "qpf_1h" in path.parts)
    assert len(drawn) == 1 and "_f001" in drawn[0], (drawn, captured.out, captured.err)

    # A frame whose coordinates no whole-cell move lands on the later
    # frame's is refused by name, not differenced: the rain of other ground
    # is never subtracted.
    lying = tmp_path / "lying"
    lying.mkdir()
    early = _write_wrfout(lying / "wrfout_d02_1974-04-03_18:00:00", stamps[:1])
    late = _write_moved_wrfout(lying / "wrfout_d02_1974-04-03_19:00:00", stamps[1:],
                               i_parent_start=6, seed_offset=1)
    import netCDF4
    with netCDF4.Dataset(late, "a") as dataset:
        dataset.variables["XLONG"][:] = dataset.variables["XLONG"][:] + np.float32(0.1)
    rc = cli.main(["render", str(early), str(late), "--engine", "rust", "--series",
                   "--products", "qpf_1h", "--size", "400x300", "--run-stamp", "off",
                   "--out", str(tmp_path / "lying-png")])
    captured = capsys.readouterr()
    assert not any("qpf_1h" in path.parts for path in (tmp_path / "lying-png").rglob("*.png"))
    assert "different ground" in captured.out + captured.err, captured.out + captured.err


def _far_travelled_nest(root):
    """Four hourly frames of one nest at parent starts 5, 5, 8 and 11.

    One parent cell at ratio 3 is three nest cells, so the last place is
    18 cells east of the first on a 16-cell-wide nest: they share no
    ground, while each move on its own (9 cells) leaves 7 columns shared.
    """

    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00",
              "1974-04-03_20:00:00", "1974-04-03_21:00:00")
    first = _write_wrfout(root / "wrfout_d02_1974-04-03_18:00:00", stamps[:1])
    second = _write_wrfout(root / "wrfout_d02_1974-04-03_19:00:00", stamps[1:2],
                           seed_offset=1)
    third = _write_moved_wrfout(root / "wrfout_d02_1974-04-03_20:00:00",
                                stamps[2:3], i_parent_start=8, seed_offset=2)
    fourth = _write_moved_wrfout(root / "wrfout_d02_1974-04-03_21:00:00",
                                 stamps[3:], i_parent_start=11, seed_offset=3)
    return first, second, third, fourth


def test_a_place_that_shares_no_ground_never_joins_a_later_series(tmp_path):
    """THE DEFECT: every earlier frame at every earlier place joined each
    later place's series, so once the nest had travelled its own width the
    renderer was handed frames with no cell in common, refused the whole
    series, and each place re-imported the run so far.  Only places that
    share ground with the series' place join it."""

    from woof.render import history_series_groups

    first, second, third, fourth = _far_travelled_nest(tmp_path)
    groups = history_series_groups([fourth, third, second, first])
    assert ([first, second], []) in groups
    assert ([first, second, third], [first, second]) in groups
    assert ([third, fourth], [third]) in groups
    assert len(groups) == 3


@needs_renderer
def test_a_nest_that_left_its_first_footprint_draws_every_frame(tmp_path, capsys):
    first, second, third, fourth = _far_travelled_nest(tmp_path)
    out = tmp_path / "png"
    rc = cli.main(["render", str(first), str(second), str(third), str(fourth),
                   "--engine", "rust", "--series", "--products", "2m_temperature,qpf_1h",
                   "--size", "400x300", "--run-stamp", "off", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err

    def drawn(product):
        return sorted(re.search(r"_f(\d{3})", path.name).group(1)
                      for path in out.rglob("*.png") if product in path.parts)

    assert drawn("2m_temperature") == ["000", "001", "002", "003"], (
        captured.out, captured.err)
    # F003 is the hour after the last move; F002 the hour after the first.
    assert drawn("qpf_1h") == ["001", "002", "003"], (captured.out, captured.err)


@needs_renderer
def test_a_frame_with_no_ground_in_common_is_stored_missing_not_refused(tmp_path):
    """The renderer's own answer when such a frame reaches it anyway (an
    explicit context list): the frame holds nothing on the later place's
    ground, so it is stored missing there and the series is drawn.  Its
    refusal took every product of the later place with it, 2 m temperature
    included."""

    from woof.render import render_series_rust

    first, _second, _third, fourth = _far_travelled_nest(tmp_path)
    written, failures, _skipped = render_series_rust(
        [first, fourth], context_paths=[first], products="2m_temperature",
        timeidx=None, outdir=tmp_path / "png", size=(400, 300))
    assert not failures, failures
    assert [path.name for path in written if "_f003" in path.name], written
    assert not [path for path in written if "_f000" in path.name], written


def test_a_series_joined_across_moves_asks_for_each_wanted_frame(monkeypatch):
    """Joined across a nest's moves, a series' baselines are other places'
    frames and usually outnumber the frames it delivers: each wanted frame
    gets its own launch instead of every baseline being drawn and thrown
    away.  A series without moves keeps the rule it had."""

    from woof import render

    start = datetime.datetime(1974, 4, 3, 18)
    stamps = {Path(f"f{i}"): (None, (start + datetime.timedelta(hours=i),))
              for i in range(6)}
    monkeypatch.setattr(render, "_history_series_record",
                        lambda path: stamps[Path(path)])
    series = list(stamps)
    two = {start + datetime.timedelta(hours=i) for i in (4, 5)}
    assert render._wanted_slots(series, two, each=True) == ["4", "5"]
    assert render._wanted_slots(series, two) is None
    everything = {start + datetime.timedelta(hours=i) for i in range(6)}
    assert render._wanted_slots(series, everything, each=True) is None


@needs_renderer
def test_a_place_with_two_frames_draws_them_without_its_baselines(tmp_path, capsys,
                                                                   monkeypatch):
    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00", "1974-04-03_20:00:00")
    first = _write_wrfout(tmp_path / "wrfout_d02_1974-04-03_18:00:00", stamps[:1])
    second = _write_moved_wrfout(tmp_path / "wrfout_d02_1974-04-03_19:00:00",
                                 stamps[1:2], i_parent_start=8, seed_offset=1)
    third = _write_moved_wrfout(tmp_path / "wrfout_d02_1974-04-03_20:00:00",
                                stamps[2:], i_parent_start=8, seed_offset=2)
    launched = []
    real = rustwx.run_renderer_series

    def recording(renderer, paths, **kwargs):
        launched.append((len(list(paths)), kwargs["frames"]))
        return real(renderer, paths, **kwargs)

    monkeypatch.setattr(rustwx, "run_renderer_series", recording)
    out = tmp_path / "png"
    rc = cli.main(["render", str(first), str(second), str(third), "--engine", "rust",
                   "--series", "--products", "2m_temperature,qpf_1h", "--size", "400x300",
                   "--run-stamp", "off", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    # The first place's lone frame is one launch of its own; the second
    # place's series (the first place's frame its baseline) launches once
    # per frame it delivers, never drawing the baseline.
    assert sorted(launched) == [(1, "all"), (3, "1"), (3, "2")], launched

    def drawn(product):
        return sorted(re.search(r"_f(\d{3})", path.name).group(1)
                      for path in out.rglob("*.png") if product in path.parts)

    assert drawn("2m_temperature") == ["000", "001", "002"]
    assert drawn("qpf_1h") == ["001", "002"]


@needs_renderer
def test_series_context_provides_window_without_rewriting_early_picture(tmp_path, capsys):
    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00")
    first = _write_wrfout(tmp_path / "wrfout_d02_first.nc", stamps[:1])
    last = _write_wrfout(tmp_path / "wrfout_d02_last.nc", stamps[1:], seed_offset=1)
    out = tmp_path / "png"
    common = ["--engine", "rust", "--series", "--products", "2m_temperature,qpf_1h",
              "--size", "400x300", "--run-stamp", "off", "--out", str(out)]
    assert cli.main(["render", str(first), *common]) == 0
    early = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in out.rglob("*.png")}
    assert early
    rc = cli.main(["render", str(last), "--context-wrfout", str(first), *common])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in early} == early
    added = set(out.rglob("*.png")) - set(early)
    assert any("qpf_1h" in path.parts for path in added)
    assert any("2m_temperature" in path.parts for path in added)
    assert all("_f001" in path.name for path in added)


@needs_renderer
@pytest.mark.parametrize("missing_hour", [None, 12], ids=["complete-day", "missing-hour"])
def test_research_long_windows_require_complete_native_writer_history(tmp_path, missing_hour):
    """Synthetic production-writer fields qualify window availability, not weather skill."""
    start = datetime.datetime(1974, 4, 3, 18)
    stamps = [(start + datetime.timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
              for hour in range(25) if hour != missing_hour]
    history = _write_wrfout(tmp_path / "full-day.nc", stamps)
    products = "qpf_6h,2m_temp_0_24h_min,2m_temp_0_24h_max"
    written, failures, skipped = rustwx.run_renderer(
        RENDERER, history, store_root=tmp_path / "store", out_dir=tmp_path / "plots",
        products=products, frames=str(len(stamps) - 1), width=400, height=300)
    assert not failures, failures
    assert any("qpf_6h" in str(path) for path in written), (written, skipped)
    if missing_hour is None:
        assert len(written) == 3 and not skipped, (written, skipped)
        assert all(_png_size(path) == (400, 300) for path in written)
    else:
        assert len(written) == 1
        assert {slug for slug, _ in skipped} == {"2m_temp_0_24h_min", "2m_temp_0_24h_max"}
        assert all("missing" in reason and "12" in reason for _, reason in skipped), skipped


def test_list_products_matplotlib_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("woof.rustwx.find_renderer", lambda: None)
    path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    fields = _frame(seed=5)
    del fields["REFL_10CM"]
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0) \
            as writer:
        writer.write_frame(_STAMPS[0], fields)
    # `--engine matplotlib` by name: `auto` no longer resolves to this
    # engine, so its catalog is reachable only where a caller asked for
    # it (render law, audit F7).
    rc = cli.main(["render", str(path), "--list-products",
                   "--engine", "matplotlib", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "engine matplotlib" in out
    # The identity-gate guard holds on this engine too: reasons are
    # variable presence, never a model identity.
    assert "gated" not in out
    lines = out.splitlines()
    assert any("refl" in line and "missing-fields" in line
               and "REFL_10CM" in line for line in lines), out
    for product in ("t2", "wind10", "precip"):
        assert any(f" {product} " in f" {line} " and "renderable" in line
                   for line in lines), (product, out)


# ---------------------------------------------------------------------------
# --pair: compose two runs' rendered PNGs into comparison sheets
# ---------------------------------------------------------------------------

@needs_renderer
@pytest.mark.parametrize("sheet", [False, True])
def test_native_run_difference_reaches_the_cli_and_publishes_its_pictures(
        tmp_path, capsys, sheet):
    import os
    import shutil
    from woof.render_receipts import read_summary

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    name = "wrfout_d02_1974-04-03_18_00_00.nc"
    _write_wrfout(a_dir / name, _STAMPS[:1], model_label="Model A")
    _write_wrfout(b_dir / name, _STAMPS[:1], seed_offset=1, model_label="Model B")
    case = tmp_path / "difference"
    command = ["render", "--diff", str(a_dir), str(b_dir),
               "--products", "t2", "--diff-labels", "Forecast A", "Forecast B",
               "--out", str(case)]
    if sheet:
        command.append("--diff-sheet")
    assert cli.main(command) == 0
    transcript = capsys.readouterr()
    assert "difference d02-1km_2m_temperature" in transcript.out
    assert "cells=192" in transcript.out
    max_abs = re.search(r"max_abs=([0-9.]+)", transcript.out)
    assert max_abs and float(max_abs[1]) > 1.0, transcript.out
    assert "failed" not in transcript.err.lower(), transcript.err
    out = run_stamp.latest(case)
    assert out is not None
    pictures = sorted(out.rglob("*.png"))
    assert len(pictures) == (2 if sheet else 1), pictures
    difference = next(path for path in pictures if "sheet" not in str(path.relative_to(out)))
    assert _png_size(difference) == (954, 806)
    if sheet:
        comparison = next(path for path in pictures if "sheet" in str(path.relative_to(out)))
        # Three intact 954x806 panels, two 12-pixel gutters and 16-pixel
        # side margins. The 120-pixel band adds only the header and labels.
        assert _png_size(comparison) == (2918, 926)
    summary = read_summary(out)
    assert summary and summary["rendered_png_count"] == len(pictures)
    assert summary["failure_count"] == 0
    assert summary["skipped_count"] == 0
    gallery = os.environ.get("RUSTWX_RECOVERY_GALLERY")
    if gallery:
        folder = Path(gallery) / ("difference-sheet" if sheet else "difference")
        folder.mkdir(parents=True, exist_ok=True)
        for picture in pictures:
            shutil.copyfile(render_layout.fs_path(picture), folder / picture.name)
        (folder / "command.txt").write_text("woof " + " ".join(command) + "\n",
                                          encoding="utf-8")
        (folder / "stdout.log").write_text(transcript.out, encoding="utf-8")
        (folder / "stderr.log").write_text(transcript.err, encoding="utf-8")


def _png_rgb(path) -> tuple[int, int, list[tuple[int, int, int]]]:
    """(width, height, RGB pixels) of an 8-bit RGB or RGBA PNG, decoded with
    zlib alone, so the pixel checks add no image-library dependency."""

    import zlib

    data = Path(render_layout.fs_path(path)).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    offset, idat, header = 8, b"", None
    while offset < len(data):
        length, kind = struct.unpack(">I4s", data[offset:offset + 8])
        body = data[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        offset += 12 + length
    width, height, depth, color, _, _, interlace = header
    assert depth == 8 and color in (2, 6) and interlace == 0, header
    channels = 3 if color == 2 else 4
    stride = width * channels
    raw = zlib.decompress(idat)
    rows, previous = [], bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for i in range(stride):
            left = line[i - channels] if i >= channels else 0
            up = previous[i]
            corner = previous[i - channels] if i >= channels else 0
            if kind == 1:
                line[i] = (line[i] + left) & 255
            elif kind == 2:
                line[i] = (line[i] + up) & 255
            elif kind == 3:
                line[i] = (line[i] + (left + up) // 2) & 255
            elif kind == 4:
                guess = left + up - corner
                pa, pb, pc = abs(guess - left), abs(guess - up), abs(guess - corner)
                pick = left if pa <= pb and pa <= pc else (up if pb <= pc else corner)
                line[i] = (line[i] + pick) & 255
        rows.append(bytes(line))
        previous = line
    pixels = [tuple(row[x * channels:x * channels + 3])
              for row in rows for x in range(width)]
    return width, height, pixels


def _blue_share(path) -> float:
    """The share of the map's middle that is drawn in a blue difference band.

    The middle of the canvas (left of the colour bar, below the header) is
    all map, so a uniform negative difference that the bar draws fills it
    with blue, and one the bar's neutral bands swallow leaves it white.
    """

    width, height, pixels = _png_rgb(path)
    xs = range(int(width * 0.15), int(width * 0.75))
    ys = range(int(height * 0.25), int(height * 0.85))
    blue = sum(1 for y in ys for x in xs
               if pixels[y * width + x][2] - pixels[y * width + x][0] > 50)
    return blue / (len(xs) * len(ys))


@needs_renderer
def test_a_difference_inside_the_fixed_bars_neutral_bands_is_drawn_not_blank(
        tmp_path, capsys):
    """WHAT BREAKAGE THIS PREVENTS (gate law): a 2 m temperature difference
    of 0.9 °F on every cell drew as a blank white map, because the fixed
    +/-10 °F bar's neutral middle bands reach 2 °F; two runs that differ
    everywhere looked identical.  Now the bar is fitted to the difference,
    the map is coloured, the receipt names the rule, and identical runs
    still say so on the fixed bar."""

    name = "wrfout_d02_1974-04-03_18_00_00.nc"
    a_dir, b_dir, same_dir = tmp_path / "a", tmp_path / "b", tmp_path / "same"
    for folder in (a_dir, b_dir, same_dir):
        folder.mkdir()
    _write_wrfout(a_dir / name, _STAMPS[:1])
    _write_wrfout(b_dir / name, _STAMPS[:1], t2_offset=0.5)
    _write_wrfout(same_dir / name, _STAMPS[:1])

    def draw(other, case, *extra):
        command = ["render", "--diff", str(a_dir), str(other), "--products", "t2",
                   "--out", str(tmp_path / case), *extra]
        rc = cli.main(command)
        transcript = capsys.readouterr()
        out = run_stamp.latest(tmp_path / case)
        pictures = sorted(out.rglob("*.png")) if out is not None else []
        return rc, transcript, pictures

    rc, transcript, pictures = draw(b_dir, "warm")
    assert rc == 0, transcript.err
    line = next(row for row in transcript.out.splitlines()
                if "difference d02-1km_2m_temperature" in row)
    assert "(adaptive(table=10," in line, line
    assert "differing=192" in line and "cells=192" in line, line
    assert re.search(r"max_abs=0\.90\d", line), line
    assert len(pictures) == 1, pictures
    assert _blue_share(pictures[0]) > 0.5, "the difference was drawn as no difference"

    rc, transcript, pictures = draw(same_dir, "same")
    assert rc == 0, transcript.err
    line = next(row for row in transcript.out.splitlines()
                if "difference d02-1km_2m_temperature" in row)
    assert "(table)" in line and "differing=0" in line, line
    assert _blue_share(pictures[0]) < 0.01

    # Every bar fitted to its own difference, by name; an unreadable table
    # is refused before anything is drawn.
    rc, transcript, _ = draw(b_dir, "auto", "--diff-table", "auto")
    assert rc == 0, transcript.err
    assert "(p99=0.90" in transcript.out, transcript.out
    rc, transcript, pictures = draw(b_dir, "missing", "--diff-table",
                                    str(tmp_path / "no-such-table.json"))
    assert rc == 2 and "--diff-table" in transcript.err, transcript.err
    assert pictures == []


@needs_renderer
@pytest.mark.parametrize("selector", ["radar_colors", "wind", "overlays", "annotations"])
def test_native_difference_sheets_honor_the_existing_presentation_selectors(
        tmp_path, monkeypatch, capsys, selector):
    import hashlib
    import json
    import os

    monkeypatch.setenv(rustwx.RADAR_COLORS_ENV, "standard")
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    name = "wrfout_d02_1974-04-03_18_00_00.nc"
    _write_wrfout(a_dir / name, _STAMPS[:1])
    _write_wrfout(b_dir / name, _STAMPS[:1], seed_offset=1)
    if selector == "radar_colors":
        product, choices = "refl", [["--radar-colors", "standard"], ["--radar-colors", "classic"]]
    elif selector == "wind":
        product, choices = "wind10", [["--barbs"], ["--streamlines"]]
    else:
        product = "t2"
        config = tmp_path / (selector + ".json")
        if selector == "overlays":
            config.write_text(json.dumps({"points": [{"lat": 39.0, "lon": -96.5,
                "shape": "plus", "color": "#ff00ff", "radius_px": 12, "width": 4}]}),
                encoding="utf-8")
            choices = [[], ["--overlays", str(config)]]
        else:
            config.write_text('{"title_suffix": "CHECK", "subtitle_center": "CHECK"}',
                              encoding="utf-8")
            choices = [[], ["--annotate", str(config)]]
    rows = []
    for index, choice in enumerate(choices):
        case = tmp_path / ("pictures-" + str(index))
        command = ["render", "--diff", str(a_dir), str(b_dir), "--diff-sheet",
                   "--products", product, "--size", "640x480", "--out", str(case), *choice]
        assert cli.main(command) == 0
        transcript = capsys.readouterr()
        magnitude = re.search(r"max_abs=([0-9.]+)", transcript.out)
        assert magnitude, transcript.out
        out = run_stamp.latest(case)
        picture = next(path for path in out.rglob("*.png")
                       if "sheet" in str(path.relative_to(out)))
        row = {"command": command, "sha256": hashlib.sha256(picture.read_bytes()).hexdigest(),
               "max_abs": magnitude[1]}
        rows.append(row)
    assert rows[0]["sha256"] != rows[1]["sha256"], selector + " was discarded at the difference door"
    assert rows[0]["max_abs"] == rows[1]["max_abs"], selector + " changed the numerical difference"
    if os.environ.get("RUSTWX_RECOVERY_GALLERY"):
        folder = Path(os.environ["RUSTWX_RECOVERY_GALLERY"]) / "selector-receipts"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (selector + ".json")).write_text(json.dumps(rows, indent=2) + "\n",
                                                  encoding="utf-8")


def _tiny_png(path, *, width=32, height=24) -> None:
    PIL = pytest.importorskip(
        "PIL", reason="--pair needs Pillow (arrives with matplotlib; no "
                      "woof extra ships it)")
    from PIL import Image

    Image.new("RGB", (width, height), "#336699").save(path)


def _pair_dirs(tmp_path):
    left = tmp_path / "run-a"
    right = tmp_path / "run-b"
    left.mkdir()
    right.mkdir()
    # Rust-engine naming: domain token + slug after the _fNNN_ lead
    # marker is the pair key, so differing run prefixes still pair.
    _tiny_png(left / "rustwx_wrf_19740403_12z_f003_d02-3km_sbcape.png")
    _tiny_png(right / "rustwx_wrf_19740403_15z_f006_d02-3km_sbcape.png")
    # matplotlib naming pairs by identical stems.
    _tiny_png(left / "t2_d02-3km_1974-04-03_18-00-00.png")
    _tiny_png(right / "t2_d02-3km_1974-04-03_18-00-00.png")
    # Unmatched product must not produce a sheet.
    _tiny_png(left / "rustwx_wrf_19740403_12z_f003_d02-3km_mlcin.png")
    return left, right


def test_pair_composes_common_products(tmp_path, capsys):
    left, right = _pair_dirs(tmp_path)
    case = tmp_path / "pairs"
    rc = cli.main(["render", "--pair", str(left), str(right),
                   "--out", str(case)])
    assert rc == 0
    # A pair compose is an invocation of the render door like any other,
    # so it claims its own run folder under --out (woof.run_stamp) and
    # two composes into one directory cannot overwrite each other.
    out = run_stamp.latest(case)
    assert out is not None and run_stamp.is_run_folder(out)
    sheets = sorted(p.name for p in out.glob("*.png"))
    assert sheets == ["d02-3km_sbcape-pair.png",
                      "t2_d02-3km_1974-04-03_18-00-00-pair.png"]
    manifest = (out / "manifest.tsv").read_text(encoding="utf-8")
    assert manifest.startswith("product\tleft\tright\tpair\n")
    assert "sbcape" in manifest
    for sheet in out.glob("*.png"):
        assert sheet.stat().st_size > 500
        width, height = _png_size(sheet)
        renderer = rustwx.find_renderer()
        if renderer is not None and rustwx.probe_renderer(renderer)[0]:
            # Native composition preserves the two 32x24 fixtures. The
            # former compositor enlarged each to 900 pixels, so its
            # thousand-pixel floor would demand resampling here.
            assert (width, height) == (108, 120), sheet.name
            from PIL import Image

            with Image.open(sheet) as image:
                for x0 in (16, 60):
                    for y in range(80, 104):
                        for x in range(x0, x0 + 32):
                            assert image.getpixel((x, y))[:3] == (51, 102, 153)
        else:
            assert width > 1_000 and height > 100, sheet.name


def test_pair_keys_keep_the_domain_so_nests_do_not_cross_pair(tmp_path):
    """Two nests of one run in one directory must not pair with each
    other -- a 3 km panel beside a 333 m panel compares nothing."""

    from woof.pair_compose import product_name

    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_d02-3km_sbcape")
    ) == "d02-3km_sbcape"
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_d03-333m_sbcape")
    ) == "d03-333m_sbcape"
    # The pre-token naming still keys the same way it always did.
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_native_grid_sbcape")
    ) == "native_grid_sbcape"
    # The exact-time suffix rides along, as before.
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f000_d02-1km_sbcape_valid_"
             "19740403_183000z_lead_000h30m00s")
    ) == "d02-1km_sbcape_valid_19740403_183000z_lead_000h30m00s"
    # A matplotlib name has no lead marker; the stem is the key.
    assert product_name(
        Path("t2_d02-1km_1974-04-03_18-00-00")
    ) == "t2_d02-1km_1974-04-03_18-00-00"
    # The shipped brand spelling keys exactly like the engine's own...
    assert product_name(
        Path("arwen_wrf_19740403_12z_f003_d02-3km_sbcape")
    ) == "d02-3km_sbcape"
    assert product_name(
        Path("arwen_wrf_19740403_12z_f000_d02-1km_sbcape_valid_"
             "19740403_183000z_lead_000h30m00s")
    ) == "d02-1km_sbcape_valid_19740403_183000z_lead_000h30m00s"
    # ...so a directory rendered before the arwen_ output rebrand still
    # pairs against one rendered after it.
    assert product_name(
        Path("arwen_wrf_19740403_12z_f003_d02-3km_sbcape")
    ) == product_name(
        Path("rustwx_wrf_19740403_15z_f006_d02-3km_sbcape"))


def test_pair_matches_one_domain_per_side_across_domains(tmp_path, capsys):
    """DOWNSCALE.md's own compare -- parent dir vs child dir -- pairs.

    The two sides of that documented command are DIFFERENT domains by
    definition (a 12 km parent against its 3 km offline child), and the
    domain-bearing pairing key made them share no key at all: walked
    2026-08-17, the doc's exact command refused with "no matching
    product PNGs".  When each side's stripped key is unambiguous, the
    same product pairs across the domain tokens.
    """
    left = tmp_path / "parent"
    right = tmp_path / "child"
    left.mkdir()
    right.mkdir()
    _tiny_png(left / "arwen_wrf_19740403_12z_f002_d01-12km_sbcape.png")
    _tiny_png(right / "arwen_wrf_19740403_12z_f002_d02-3km_sbcape.png")
    rc = cli.main(["render", "--pair", str(left), str(right),
                   "--out", str(tmp_path / "pairs")])
    assert rc == 0
    sheets = list(run_stamp.latest(tmp_path / "pairs").glob("*-pair.png"))
    assert len(sheets) == 1
    capsys.readouterr()


def test_pair_never_cross_pairs_an_ambiguous_side(tmp_path, capsys):
    """Two nests of one run in one directory still never cross-pair."""
    from woof.pair_compose import compose_pairs

    left = tmp_path / "a"
    right = tmp_path / "b"
    left.mkdir()
    right.mkdir()
    # Left carries TWO nests of the same product: the stripped key is
    # ambiguous, so neither may pair with the right side's single nest.
    _tiny_png(left / "rustwx_wrf_19740403_12z_f000_d02-3km_sbcape.png")
    _tiny_png(left / "rustwx_wrf_19740403_12z_f000_d03-333m_sbcape.png")
    _tiny_png(right / "rustwx_wrf_19740403_12z_f000_d04-111m_sbcape.png")
    with pytest.raises(ValueError, match="no matching product"):
        compose_pairs(left, right, tmp_path / "pairs", title="t")


def test_pair_with_no_common_products_is_exit_2(tmp_path, capsys):
    left = tmp_path / "a"
    right = tmp_path / "b"
    left.mkdir()
    right.mkdir()
    _tiny_png(left / "rustwx_wrf_19740403_12z_f000_d02-1km_sbcape.png")
    _tiny_png(right / "rustwx_wrf_19740403_12z_f000_d02-1km_mucape.png")
    rc = cli.main(["render", "--pair", str(left), str(right),
                   "--out", str(tmp_path / "pairs")])
    assert rc == 2
    assert "no matching product PNGs" in capsys.readouterr().err


def test_pair_rejects_wrfout_arguments(tmp_path, capsys):
    rc = cli.main(["render", "some_wrfout.nc", "--pair",
                   str(tmp_path), str(tmp_path),
                   "--out", str(tmp_path / "pairs")])
    assert rc == 2
    assert "do not combine" in capsys.readouterr().err


def test_render_without_wrfouts_or_pair_is_exit_2(capsys):
    rc = cli.main(["render"])
    assert rc == 2
    assert "WRFOUT" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# doctor integration
# ---------------------------------------------------------------------------

def test_doctor_reports_the_rust_renderer():
    from woof.doctor import _renderer_tree_check, _rust_renderer_check
    from woof.provenance_gate import bridge_tree_match

    renderer = rustwx.find_renderer()
    check = _rust_renderer_check()
    assert check.name.startswith("renderer rw_wrfbatch")
    if renderer is None:
        # Not built: an info line with the build one-liner, never a gap
        # (matplotlib remains the documented fallback).
        assert check.status in ("info", "missing")
        assert check.remedy and "cargo build" in check.remedy
        return
    assert str(renderer) in check.detail
    # Doctor reports launch/ABI and source identity in separate rows.  The
    # collection-time renderer gate combines those two questions; its cached
    # answer cannot decide which doctor row must report a refusal.
    contract_ok, _ = rustwx.probe_renderer(renderer)
    if contract_ok:
        assert check.status == "verified"
        assert "basemap" in check.detail
        assert "--abi matches the render contract" in check.detail
    else:
        # A resolved binary with a broken launch/ABI is reported with its
        # rebuild remedy; file existence cannot satisfy this check.
        assert check.status == "missing"
        assert check.blocking is False
        assert check.remedy and "cargo build" in check.remedy

    match = bridge_tree_match(renderer, env_var=rustwx.RENDERER_ENV)
    tree_check = _renderer_tree_check()
    if match.matched:
        assert tree_check.status == "verified"
        assert match.verdict in tree_check.detail
    else:
        # Task #106: an ABI-compatible foreign renderer must still be named
        # and refused by the source-identity row that the render door reads.
        assert tree_check.status == "missing"
        assert str(renderer) in tree_check.detail
        assert "from another tree" in tree_check.detail
        assert tree_check.blocking is False
        assert tree_check.remedy and "cargo build" in tree_check.remedy


def test_doctor_env_override_naming_missing_file_is_hard(monkeypatch):
    from woof.doctor import _rust_renderer_check

    monkeypatch.setenv(rustwx.RENDERER_ENV, r"C:\nonexistent\rw.exe")
    check = _rust_renderer_check()
    assert check.status == "missing"
    assert "names a missing file" in check.detail


def test_theme_and_section_reach_the_renderer_invocation(monkeypatch,
                                                        tmp_path):
    """A theme is a front-door flag, and saying nothing leaves the
    invocation byte-identical: no ``--theme`` at all, so the engine
    draws its own look and the regression gate's hashes hold.  The
    section line, the isotherm set and the across-line frame travel the
    same way, verbatim, so the engine's own refusals name a mistake."""

    from woof import render as render_module

    seen = _renderer_spy(monkeypatch, tmp_path)

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="t2",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600))
    assert "--theme" not in seen[-1]
    assert "--section" not in seen[-1]
    assert "--isotherms" not in seen[-1]
    assert "--section-across" not in seen[-1]

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="t2",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        theme="dark")
    assert seen[-1][seen[-1].index("--theme") + 1] == "dark"

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"],
        products="xsec:agent:QCLOUD@cold/wa=1,2,5,10@5",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        theme=str(tmp_path / "brand.json"),
        section="38.32,-99.0,38.32,-98.4",
        isotherms="0,-5,-10,-15,-20@-10", section_across_km=60)
    argv = seen[-1]
    assert argv[argv.index("--theme") + 1] == str(tmp_path / "brand.json")
    assert argv[argv.index("--section") + 1] == "38.32,-99.0,38.32,-98.4"
    assert argv[argv.index("--isotherms") + 1] == "0,-5,-10,-15,-20@-10"
    assert argv[argv.index("--section-across") + 1] == "60.0"
    # The section product token is engine grammar and passes untouched.
    assert argv[argv.index("--products") + 1] == \
        "xsec:agent:QCLOUD@cold/wa=1,2,5,10@5"
    # A caller that never mentions the section size sends no flag: the
    # engine draws a cut landscape at the map's width by itself.
    assert "--section-size" not in argv

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"],
        products="xsec:QCLOUD@cold",
        timeidx=0, outdir=tmp_path / "png", size=(1800, 1464),
        section="38.32,-99.0,38.32,-98.4", section_size=(2400, 1200))
    argv = seen[-1]
    assert argv[argv.index("--section-size") + 1] == "2400x1200"
    assert argv[argv.index("--width") + 1] == "1800"


def test_theme_and_section_are_on_the_render_parser(monkeypatch, tmp_path):
    """Typed by a user, not only passed by a caller: ``woof render
    --theme dark`` and ``--section`` are the doors."""

    seen = _renderer_spy(monkeypatch, tmp_path)
    wrfout = tmp_path / "wrfout_d02_2026-08-19_00_00_00"
    wrfout.write_bytes(b"")

    parser = cli.build_parser()
    args = parser.parse_args(
        ["render", str(wrfout), "--engine", "rust", "--theme", "dark",
         "--section", "38.32,-99.0,38.32,-98.4",
         "--isotherms", "0,-10@-10", "--section-across", "60",
         "--out", str(tmp_path / "png")])
    assert args.theme == "dark"
    assert args.section == "38.32,-99.0,38.32,-98.4"
    assert args.isotherms == "0,-10@-10"
    assert args.section_across_km == 60.0

    args = parser.parse_args(
        ["render", str(wrfout), "--engine", "rust",
         "--out", str(tmp_path / "png")])
    assert args.theme is None
    assert args.section is None
    assert args.isotherms is None
    assert args.section_across_km is None
    assert args.section_size is None
    assert not seen

    # ``--section-size`` is on the same door, and a value that is not WxH
    # is refused by name rather than quietly taken as a map size.
    args = parser.parse_args(
        ["render", str(wrfout), "--engine", "rust",
         "--section", "38.32,-99.0,38.32,-98.4",
         "--section-size", "2400x1200", "--out", str(tmp_path / "png")])
    assert args.section_size == (2400, 1200)
    for bad in ("2400", "2400x", "12x9", "axb"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["render", str(wrfout), "--engine", "rust",
                 "--section-size", bad, "--out", str(tmp_path / "png")])


def test_the_section_top_reaches_the_renderer_and_stays_absent_by_default(
        monkeypatch, tmp_path):
    """The ceiling of a section's height axis is a front-door flag.

    The engine has taken ``--section-top-km`` from the start and no door
    forwarded it, so every published cut was fitted to the engine's
    14 km whatever was in it -- a one-kilometre feature drawn in the
    bottom fourteenth of the frame.  Saying nothing still sends no flag,
    so a render that never mentions the ceiling is byte-identical to
    every earlier release.
    """

    from woof import render as render_module

    seen = _renderer_spy(monkeypatch, tmp_path)

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="xsec:tk",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        section="38.32,-99.0,38.32,-98.4")
    assert "--section-top-km" not in seen[-1]

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="xsec:tk",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        section="38.32,-99.0,38.32,-98.4", section_top_km=3)
    argv = seen[-1]
    assert argv[argv.index("--section-top-km") + 1] == "3.0"

    # The series lane is the same door and forwards the same flag.
    render_module.render_series_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="xsec:tk",
        timeidx=None, outdir=tmp_path / "png-series", size=(800, 600),
        section="38.32,-99.0,38.32,-98.4", section_top_km=2.5)
    argv = seen[-1]
    assert argv[argv.index("--section-top-km") + 1] == "2.5"


def test_the_section_top_is_refused_outside_the_engine_range(monkeypatch,
                                                            tmp_path):
    """Out of 1-40 km the door answers in the engine's own sentence,
    before a render launches rather than after the frames are open."""

    from woof import render as render_module

    seen = _renderer_spy(monkeypatch, tmp_path)
    wrfout = tmp_path / "wrfout_d02_2026-08-19_00_00_00"
    wrfout.write_bytes(b"")

    parser = cli.build_parser()
    args = parser.parse_args(
        ["render", str(wrfout), "--engine", "rust",
         "--section", "38.32,-99.0,38.32,-98.4",
         "--section-top-km", "3", "--out", str(tmp_path / "png")])
    assert args.section_top_km == 3.0

    args = parser.parse_args(
        ["render", str(wrfout), "--engine", "rust",
         "--out", str(tmp_path / "png")])
    assert args.section_top_km is None

    for bad in ("0.5", "0", "41", "-3", "abc", "nan", "inf"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["render", str(wrfout), "--engine", "rust",
                 "--section-top-km", bad, "--out", str(tmp_path / "png")])

    # The same refusal on the in-process lane, in the same words, and no
    # render is launched with the value that was refused.
    with pytest.raises(ValueError) as excinfo:
        render_module.render_wrfouts_rust(
            [tmp_path / "wrfout_d02_x.nc"], products="xsec:tk",
            timeidx=0, outdir=tmp_path / "png", size=(800, 600),
            section="38.32,-99.0,38.32,-98.4", section_top_km=0.5)
    assert "--section-top-km '0.5' is not within 1-40 km" in str(excinfo.value)
    assert not any("--section-top-km" in argv for argv in seen)


def test_the_render_receipt_records_how_tall_the_cut_was(tmp_path):
    """Two cuts of one line differ only by their ceiling, so a receipt
    that does not carry it cannot tell the reader which is which."""

    from woof import render_receipts

    root = tmp_path / "png"
    root.mkdir()
    summary = render_receipts.publish_invocation(
        root=root, engine="rust", requested_spec="xsec:tk", written=[],
        failures=[], skipped=[], layout="nested", section_top_km=3.0)
    assert summary["section_tops_km"] == [3.0]

    summary = render_receipts.publish_invocation(
        root=root, engine="rust", requested_spec="xsec:tk", written=[],
        failures=[], skipped=[], layout="nested", section_top_km=14.0)
    assert summary["section_tops_km"] == [3.0, 14.0]

    # A render that drew no section records no ceiling and leaves the
    # list exactly as it was.
    summary = render_receipts.publish_invocation(
        root=root, engine="rust", requested_spec="2m_temperature",
        written=[], failures=[], skipped=[], layout="nested")
    assert summary["section_tops_km"] == [3.0, 14.0]
    assert summary["additional_section_tops_km"] == 0

    # Past eight distinct ceilings the list is capped, like every other
    # capped list in the summary, and the count of what was dropped is
    # what tells the reader the list is not the whole of it.
    for top in (2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        summary = render_receipts.publish_invocation(
            root=root, engine="rust", requested_spec="xsec:tk", written=[],
            failures=[], skipped=[], layout="nested", section_top_km=top)
    assert len(summary["section_tops_km"]) == 8
    assert summary["additional_section_tops_km"] == 2


def test_the_render_receipt_records_the_range_each_cut_was_drawn_on():
    """A cut's colour bar is fitted to its own frame at BOTH ends.

    The ceiling moved with the frame before this lane's work and the
    bottom moves with it now, so two cuts of one line an hour apart can
    be drawn on two different bars.  The receipt is the only place that
    can say which bar, so it carries the range and the rule that set it.
    """

    from woof import render_receipts

    line = ("SECTIONFILL xsec_tk lo=286.05 hi=310.2 absence=0 "
            "rule=own-minimum")
    assert rustwx.parse_section_fill(line) == {
        "family": "xsec_tk", "lo": 286.05, "hi": 310.2,
        "absence": False, "rule": "own-minimum"}
    # The event word is part of the handshake: a build predating it
    # answers the old grammar and is refused rather than drawing cuts
    # whose bars go unrecorded.
    assert "\tSECTIONFILL\t" in rustwx.RENDERER_ABI_MARKER
    # A line the engine spells differently is dropped, not raised on: a
    # receipt is metadata beside a picture that was drawn.
    assert rustwx.parse_section_fill("RENDERED xsec_tk /tmp/a.png") is None
    assert rustwx.parse_section_fill("SECTIONFILL xsec_tk lo=x hi=2") is None
    assert rustwx.parse_section_fill("SECTIONFILL  lo=1 hi=2") is None


def test_the_render_summary_carries_and_caps_the_drawn_fill_rows(tmp_path):
    """The same treatment every other list in the summary gets."""

    from woof import render_receipts

    root = tmp_path / "png"
    root.mkdir()
    kelvin = {"family": "xsec_tk", "lo": 286.05, "hi": 310.2,
              "absence": False, "rule": "own-minimum"}
    mixing = {"family": "xsec_qv", "lo": 0.0, "hi": 0.012,
              "absence": True, "rule": "zero-anchor"}
    summary = render_receipts.publish_invocation(
        root=root, engine="rust", requested_spec="xsec:tk,xsec:qv",
        written=[], failures=[], skipped=[], layout="nested",
        section_top_km=3.0, section_fills=[kelvin, mixing])
    assert summary["section_fills"] == [kelvin, mixing]
    assert summary["additional_section_fills"] == 0

    # The same cut drawn again on the same bar is one row, exactly as a
    # repeated ceiling is one entry.
    summary = render_receipts.publish_invocation(
        root=root, engine="rust", requested_spec="xsec:tk", written=[],
        failures=[], skipped=[], layout="nested", section_top_km=3.0,
        section_fills=[kelvin])
    assert summary["section_fills"] == [kelvin, mixing]

    # A render that drew no section leaves the list alone.
    summary = render_receipts.publish_invocation(
        root=root, engine="rust", requested_spec="2m_temperature",
        written=[], failures=[], skipped=[], layout="nested")
    assert summary["section_fills"] == [kelvin, mixing]

    for index in range(8):
        summary = render_receipts.publish_invocation(
            root=root, engine="rust", requested_spec="xsec:tk", written=[],
            failures=[], skipped=[], layout="nested",
            section_fills=[dict(kelvin, lo=float(index))])
    assert len(summary["section_fills"]) == 8
    assert summary["additional_section_fills"] == 2


def test_the_abi_marker_and_product_parser_carry_the_section_family():
    """``xsec:`` is engine vocabulary the Python side forwards, so it is
    part of the handshake (a build predating it answers the old grammar
    and is refused) and the product parser must not rewrite it."""

    assert "\txsec:\t" in rustwx.RENDERER_ABI_MARKER
    spec = parse_products_rust(
        "refl,xsec:cloud:QCLOUD+QRAIN~log/QVAPOR@cold/wa=1,2,5,10@5")
    assert spec == ("composite_reflectivity,"
                    "xsec:cloud:QCLOUD+QRAIN~log/QVAPOR@cold/wa=1,2,5,10@5")


@needs_renderer
def test_the_dark_theme_changes_the_pixels_and_no_theme_does_not(
        wrfout, tmp_path):
    """The one thing a theme must never do is move a render nobody
    themed.  Two untitled renders agree byte for byte; the dark built-in
    differs from them; a theme file with one unknown key is refused
    before anything is drawn, naming the key."""

    def render(out: Path, *extra: str) -> bytes:
        rc = cli.main(["render", str(wrfout), "--engine", "rust",
                       "--products", "t2", "--timeidx", "0",
                       "--size", "640x480", "--out", str(out), *extra])
        assert rc == 0, extra
        produced = [p for p in out.rglob("*.png")]
        assert len(produced) == 1, produced
        return Path(render_layout.fs_path(produced[0])).read_bytes()

    first = render(tmp_path / "a")
    second = render(tmp_path / "b")
    assert first == second
    dark = render(tmp_path / "c", "--theme", "dark")
    assert dark != first
    named = render(tmp_path / "d", "--theme", "default")
    assert named == first

    bad = tmp_path / "bad.json"
    bad.write_text('{"surface": {"canvas": "#000000", "paper": "#fff"}}',
                   encoding="utf-8")
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "t2", "--timeidx", "0",
                   "--size", "640x480", "--out", str(tmp_path / "e"),
                   "--theme", str(bad)])
    assert rc != 0
    assert not list((tmp_path / "e").rglob("*.png"))


def test_the_abi_marker_matches_the_rust_source_without_a_build():
    """The Python constant against ``main.rs`` AS TEXT, so it cannot skip.

    ``test_the_pinned_abi_marker_is_the_built_renderer_s_own_answer`` already
    compares the constant to a BUILT binary, and its docstring argues the
    no-build skip "cannot hide a mismatch, because there is nothing of this
    tree's to mismatch with".  That reasoning is what let a real mismatch
    ship: ``main.rs`` IS of this tree, it is merely not compiled.  Commit
    26469cd1e added ``mesh:`` and ``meshdiff:`` to the Rust ``ABI_MARKER`` and
    left ``RENDERER_ABI_MARKER`` behind, so on 2.6.5 ``woof render --engine
    rust`` refused its own freshly built renderer from ANY source checkout --
    no binary in existence could satisfy the door -- while every machine
    without a build skipped the one test that would have said so.

    This reads the Rust literal, so it is red on a bare checkout with no
    toolchain and no build.  The concrete breakage it prevents: the render
    door refusing every renderer, which under the render law leaves weather
    fields undrawn rather than drawn by something else.
    """

    source = (Path(__file__).resolve().parents[1] / "tools" / "rustwx"
              / "crates" / "rw-wrfbatch" / "src" / "main.rs")
    assert source.is_file(), f"{source} is missing"
    match = re.search(r'const ABI_MARKER:\s*&str\s*=\s*"(.*?)";',
                      source.read_text(encoding="utf-8"), re.S)
    assert match, "main.rs no longer declares `const ABI_MARKER: &str`"
    # Rust line continuations: a trailing backslash-newline eats the newline
    # and the next line's leading whitespace.
    literal = re.sub(r"\\\n\s*", "", match.group(1)).replace("\\t", "\t")
    assert literal == rustwx.RENDERER_ABI_MARKER, (
        "woof.rustwx.RENDERER_ABI_MARKER and the Rust ABI_MARKER in "
        f"{source.name} have drifted apart:\n"
        f"  rust  : {literal!r}\n"
        f"  python: {rustwx.RENDERER_ABI_MARKER!r}")


@needs_renderer
def test_native_difference_selects_inner_record_by_actual_valid_time(tmp_path, capsys):
    a = _write_wrfout(tmp_path / "wrfout_d02_1974-04-03_18_00_00.nc",
                      ("1974-04-03_18:00:00", "1974-04-03_19:00:00",
                       "1974-04-03_20:00:00"))
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    b = _write_wrfout(b_dir / "wrfout_d02_1974-04-03_19_00_00.nc",
                      ("1974-04-03_19:00:00", "1974-04-03_20:00:00"), seed_offset=4)
    out = tmp_path / "pictures"
    assert cli.main(["render", "--diff", str(a), str(b), "--timeidx", "2",
                     "--products", "t2", "--size", "400x300", "--out", str(out),
                     "--run-stamp", "off"]) == 0
    transcript = capsys.readouterr()
    assert "difference d02-1km_2m_temperature 1974-04-03 20:00Z" in transcript.out
    pictures = list(out.rglob("*.png"))
    assert len(pictures) == 1
    assert "f002" in pictures[0].name


@needs_renderer
def test_native_difference_folder_timeidx_selects_each_domain(tmp_path, capsys):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    for grid in (1, 2):
        for index, stamp in enumerate(("1974-04-03_18:00:00", "1974-04-03_19:00:00")):
            name = f"wrfout_d{grid:02d}_{stamp.replace(':', '_')}.nc"
            _write_wrfout(a_dir / name, (stamp,), grid_id=grid, seed_offset=index)
            _write_wrfout(b_dir / name, (stamp,), grid_id=grid, seed_offset=index + 2)
    out = tmp_path / "pictures"
    assert cli.main(["render", "--diff", str(a_dir), str(b_dir), "--timeidx", "1",
                     "--products", "t2", "--size", "400x300", "--out", str(out),
                     "--run-stamp", "off"]) == 0
    transcript = capsys.readouterr()
    assert "difference d01-1km_2m_temperature 1974-04-03 19:00Z" in transcript.out
    assert "difference d02-1km_2m_temperature 1974-04-03 19:00Z" in transcript.out
    assert len(list(out.rglob("*.png"))) == 2
    assert all("f001" in path.name for path in out.rglob("*.png"))


@needs_renderer
def test_native_difference_split_qpf_keeps_both_runs_window_context(tmp_path, capsys):
    import hashlib
    import json
    import os
    from woof.render_receipts import read_summary

    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00")
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    for index, stamp in enumerate(stamps):
        name = f"wrfout_d02_{stamp.replace(':', '_')}.nc"
        _write_wrfout(a_dir / name, (stamp,), seed_offset=index)
        _write_wrfout(b_dir / name, (stamp,), seed_offset=index + 2)
    a_combined = _write_wrfout(tmp_path / "a-combined.nc", stamps)
    b_combined = _write_wrfout(tmp_path / "b-combined.nc", stamps, seed_offset=2)
    rows, images = [], []
    for label, a, b in (("split", a_dir, b_dir),
                        ("combined", a_combined, b_combined)):
        out = tmp_path / (label + "-pictures")
        command = ["render", "--diff", str(a), str(b), "--timeidx", "1",
                   "--products", "qpf_1h", "--diff-labels", "A", "B",
                   "--size", "400x300", "--out", str(out), "--run-stamp", "off"]
        assert cli.main(command) == 0
        transcript = capsys.readouterr()
        assert "difference d02-1km_qpf_1h 1974-04-03 19:00Z" in transcript.out
        pictures = list(out.rglob("*.png"))
        assert len(pictures) == 1, transcript.out + transcript.err
        payload = pictures[0].read_bytes()
        images.append(payload)
        summary = read_summary(out)
        assert summary["failure_count"] == summary["skipped_count"] == 0
        rows.append({"command": command, "sha256": hashlib.sha256(payload).hexdigest(),
                     "stdout": transcript.out, "stderr": transcript.err})
    assert images[0] == images[1]
    if os.environ.get("RUSTWX_RECOVERY_GALLERY"):
        folder = Path(os.environ["RUSTWX_RECOVERY_GALLERY"]) / "difference-window"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "qpf_1h-difference.png").write_bytes(images[0])
        (folder / "receipt.json").write_text(json.dumps(rows, indent=2) + "\n",
                                            encoding="utf-8")


@needs_renderer
@pytest.mark.parametrize("first_only", [False, True])
def test_native_difference_registered_window_absence_is_a_skip_not_a_failure(
        tmp_path, capsys, first_only):
    from woof.render_receipts import read_summary

    stamps = ("1974-04-03_18:00:00", "1974-04-03_19:00:00")
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    for index, stamp in enumerate(stamps):
        name = f"wrfout_d02_{stamp.replace(':', '_')}.nc"
        _write_wrfout(a_dir / name, (stamp,), seed_offset=index)
        _write_wrfout(b_dir / name, (stamp,), seed_offset=index + 2)
    out = tmp_path / "pictures"
    command = ["render", "--diff", str(a_dir), str(b_dir), "--products", "qpf_1h",
               "--size", "400x300", "--out", str(out), "--run-stamp", "off"]
    if first_only:
        command += ["--timeidx", "0"]
    assert cli.main(command) == (1 if first_only else 0)
    transcript = capsys.readouterr()
    assert "F000" in transcript.err and "skipped qpf_1h" in transcript.err
    assert "failed" not in transcript.err.lower(), transcript.err
    summary = read_summary(out)
    assert summary["failure_count"] == 0
    assert summary["skipped_count"] >= 1
    assert summary["rendered_png_count"] == (0 if first_only else 1)
    assert summary["undrawn_family_count"] == (1 if first_only else 0)


@needs_renderer
@pytest.mark.parametrize("theme_name", ["woof-dark"])
def test_native_750m_nest_inherits_the_woof_theme_without_changing_generic_pixels(
        tmp_path, theme_name):
    """The site's extends file reaches Rust and keeps public WOOF labels.

    An inherited theme and its equivalent built-in draw identical bytes.
    The generic no-theme and default-theme controls remain identical.
    """
    import hashlib
    import json
    import os

    wrfout = _write_wrfout(
        tmp_path / "wrfout_d02_1974-04-03_18-00-00.nc", (_STAMPS[0],),
        grid_id=2, dx=750.0)
    inherited = tmp_path / "site-theme.json"
    inherited.write_text(json.dumps({"extends": theme_name, "name": "site"}),
                         encoding="utf-8")

    def render(name, *extra):
        out = tmp_path / name
        assert cli.main([
            "render", str(wrfout), "--engine", "rust", "--products", "t2",
            "--timeidx", "0", "--size", "640x480", "--out", str(out),
            "--source-label", "WOOF 2.8.6", *extra]) == 0
        pictures = list(out.rglob("*.png"))
        assert len(pictures) == 1, pictures
        assert "d02-750m" in pictures[0].as_posix(), pictures[0]
        return Path(render_layout.fs_path(pictures[0])).read_bytes()

    generic = render("generic")
    default = render("default", "--theme", "default")
    builtin = render("builtin", "--theme", theme_name)
    inherited_png = render("inherited", "--theme", str(inherited))
    assert generic == default
    assert builtin == inherited_png
    assert inherited_png != generic
    # The theme's labels are tested directly in Rust; these actual pixels
    # prove that the same inherited theme reaches the public render door.
    evidence = os.environ.get("WOOF_RENDERER_EXTRAS_EVIDENCE")
    if evidence:
        root = Path(evidence)
        root.mkdir(parents=True, exist_ok=True)
        for label, png in (("generic", generic), ("inherited", inherited_png)):
            (root / f"750m-{theme_name}-{label}.png").write_bytes(png)
        (root / f"750m-{theme_name}-receipt.json").write_text(json.dumps({
            "schema": "renderer-extras.woof-theme-proof.v1",
            "fixture": "WrfoutWriter single-frame d02 750m",
            "theme": {"extends": theme_name, "name": "site"},
            "generic_default_byte_identical": generic == default,
            "inherited_builtin_byte_identical": inherited_png == builtin,
            "generic_sha256": hashlib.sha256(generic).hexdigest(),
            "inherited_sha256": hashlib.sha256(inherited_png).hexdigest(),
        }, indent=2) + "\n", encoding="utf-8")



@needs_renderer
def test_native_pair_sheet_inherits_its_requested_theme_and_keeps_panel_pixels(tmp_path):
    from PIL import Image

    left, right = _pair_dirs(tmp_path)
    theme = tmp_path / "site-dark.json"
    theme.write_text('{"extends":"woof-dark"}', encoding="utf-8")

    def pair(name, *extra):
        out = tmp_path / name
        assert cli.main(["render", "--pair", str(left), str(right),
                         "--out", str(out), *extra]) == 0
        pictures = sorted(run_stamp.latest(out).glob("*.png"))
        assert len(pictures) == 2
        for picture in pictures:
            with Image.open(picture) as image:
                for x0 in (16, 60):
                    for y in range(80, 104):
                        for x in range(x0, x0 + 32):
                            assert image.getpixel((x, y))[:3] == (51, 102, 153)
        return [picture.read_bytes() for picture in pictures]

    generic = pair("pair-generic")
    assert pair("pair-default", "--theme", "default") == generic
    dark = pair("pair-dark", "--theme", "woof-dark")
    assert pair("pair-inherited", "--theme", str(theme)) == dark
    assert dark != generic


@needs_renderer
def test_native_compact_750m_store_accepts_inherited_woof_theme_and_explicit_auto(tmp_path):
    import hashlib
    import json
    import os
    import subprocess

    source = _write_wrfout(
        tmp_path / "wrfout_d02_1974-04-03_18-00-00.nc", (_STAMPS[0],), dx=750.0)
    process_request, process_result = tmp_path / "process-request.json", tmp_path / "process-result.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    process_request.write_text(json.dumps({
        "schema": "arwen.wrf-process-request.v2", "path": str(source),
        "source_sha256": digest, "case_id": "renderer-extras-theme", "domain": "d02",
        "valid_utc": "1974-04-03T18:00:00Z", "store_root": str(tmp_path / "native"),
        "lead_seconds": 0, "profile": "viewer-2d-v1", "products": ["2m_temperature"]}),
        encoding="utf-8")
    done = subprocess.run([
        str(RENDERER), "--process-request", str(process_request), "--process-result", str(process_result)],
        env=rustwx.renderer_env(), capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    processed = json.loads(process_result.read_text(encoding="utf-8"))
    theme = tmp_path / "site-light.json"
    theme.write_text('{"extends":"woof-light"}', encoding="utf-8")

    def render(name, **presentation):
        request, receipt = tmp_path / f"{name}-request.json", tmp_path / f"{name}-result.json"
        request.write_text(json.dumps({
            "schema": "arwen.native-store-render-request.v1", "process_result": str(process_result),
            "expected_frame_id": processed["frame"]["id"], "expected_source_sha256": digest,
            "out_dir": str(tmp_path / name), "products": ["2m_temperature"], "spacing_m": 750.0,
            "width": 640, "height": 480, **presentation}), encoding="utf-8")
        done = subprocess.run([
            str(RENDERER), "--render-store-request", str(request), "--render-store-result", str(receipt)],
            env=rustwx.renderer_env(), capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        result = json.loads(receipt.read_text(encoding="utf-8"))
        assert result["wrf_imported"] is False and result["volume_store_created"] is False
        assert len(result["panels"]) == 1
        picture = Path(result["panels"][0]["path"])
        return picture.read_bytes(), result

    generic, _ = render("compact-generic")
    default, _ = render("compact-default", theme="default")
    builtin, _ = render("compact-builtin", theme="woof-light")
    inherited, fixed = render("compact-inherited", theme=str(theme))
    auto, automatic = render("compact-auto", theme=str(theme), width=None, height=None)
    assert generic == default
    assert inherited == builtin and inherited != generic
    assert fixed["layout"] == "fixed" and (fixed["width"], fixed["height"]) == (640, 480)
    assert automatic["layout"] == "auto"
    assert (automatic["width"], automatic["height"]) != (640, 480)
    evidence = os.environ.get("WOOF_RENDERER_EXTRAS_EVIDENCE")
    if evidence:
        root = Path(evidence)
        root.mkdir(parents=True, exist_ok=True)
        for name, png in (("generic", generic), ("inherited", inherited), ("auto", auto)):
            (root / f"compact-750m-{name}.png").write_bytes(png)
        (root / "compact-750m-receipt.json").write_text(json.dumps({
            "schema": "renderer-extras.compact-theme-proof.v1", "fixture": "WrfoutWriter d02 750m",
            "generic_default_byte_identical": generic == default,
            "inherited_builtin_byte_identical": inherited == builtin,
            "wrf_imported_during_render": automatic["wrf_imported"],
            "fixed_dimensions": [fixed["width"], fixed["height"]],
            "auto_dimensions": [automatic["width"], automatic["height"]],
        }, indent=2) + "\n", encoding="utf-8")
