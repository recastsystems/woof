"""The rendered-output directory layout -- domain / product / valid day.

The defect: every PNG a run drew landed in ONE directory.  A three-nest
run of the rust catalog at hourly output puts five figures of files
there, of every product and every valid time, and the only way to find
one is to read filenames.  The field report that opened this was
"thousands of frames of all sorts of timestamps and plot types in one
directory".

The layout that replaces it is the 2026-08-06 project ruling (case folder ->
domain -> product subfolders) with the reporter's timestamp request
slotted into it as the leaf grouping:

    <--out>/<domain-token>/<product>/<valid-day>/<filename>.png

It is the DEFAULT, not a flag.  ``--layout flat`` is retained only so a
consumer written against the old flat directory has somewhere to stand
while it is updated; nothing in the product selects it.

These tests pin three things a script can rely on: the path is a pure
function of (domain, product, valid time) so it can be predicted without
globbing; the flat spelling is byte-for-byte the v2.4.1 spelling; and
every in-tree reader of these directories (``--pair``, the early-render
publisher) finds files in both layouts.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

import woof.cli as cli
from woof import render_layout, render_receipts, run_stamp
from woof.io.wrfout import WrfoutWriter

_NZ, _NY, _NX = 4, 12, 16
_STAMPS = ("1974-04-03_18:00:00", "1974-04-03_19:00:00")


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
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        # TOA outgoing longwave, the range a frame with cold cloud tops
        # over a warm surface actually spans (W m-2).
        "OLR": rng.uniform(90.0, 320.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }


class _RendererResult:
    returncode = 0
    stderr = ""

    def __init__(self, stdout: str):
        self.stdout = stdout


def _stub_flat_renderer(monkeypatch, tmp_path, names):
    import subprocess

    from woof import render as render_module

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for name in names:
            (out_dir / name).write_bytes(b"PNG")
            lines.append(f"RENDERED slug {out_dir / name}")
        return _RendererResult("\n".join(lines) + "\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(render_module, "renderer_refusal", lambda _r: None)
    monkeypatch.setattr("woof.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch")


def _early_runner(relative):
    def fake_runner(command):
        scratch = Path(command[command.index("--out") + 1])
        target = scratch / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PNG-bytes")

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return Completed()

    return fake_runner


def _write_wrfout(path: Path, *, dx: float = 1000.0) -> Path:
    # Gate the real artifact route; pure layout contracts require no SDK.
    pytest.importorskip(
        "wrf", reason="woof render requires the wrf package (wrf-rust)")
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=dx, dy=dx) as writer:
        for index, stamp in enumerate(_STAMPS):
            writer.write_frame(stamp, _frame(seed=11 + index))
    return path


@pytest.fixture(scope="module")
def wrfout(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("render-layout")
    return _write_wrfout(root / "wrfout_d02_1974-04-03_18-00-00.nc")


def _relative_pngs(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix()
                  for p in root.rglob("*.png"))


#: The render's own record, filed at the root of every render folder by
#: woof.render_receipts: the invocation receipts and their summary.  It
#: is not a picture and not a layout level, so a listing of the layout
#: leaves it out, and the listing checks it is present rather than
#: letting any other stray entry through.
_RENDER_RECORD = {".render-receipts", render_receipts.SUMMARY_FILENAME}


def _layout_entries(root: Path) -> list[str]:
    names = sorted(p.name for p in root.iterdir())
    assert _RENDER_RECORD <= set(names), names
    return [name for name in names if name not in _RENDER_RECORD]


def _run_dir(out: Path) -> Path:
    """The one run folder ``woof render`` claimed under ``--out``.

    2.5.0 puts a run-stamped level above this layout
    (:mod:`woof.run_stamp`) so two renders into one ``--out`` cannot
    overwrite each other.  The tree BELOW it is unchanged, and that
    tree is what this file is about, so every assertion here is made
    relative to the folder rather than to ``--out``.
    """

    folders = [child for child in out.iterdir()
               if child.is_dir() and run_stamp.is_run_folder(child)]
    assert len(folders) == 1, (
        f"expected exactly one run folder under {out}, found "
        f"{[f.name for f in folders]}")
    return folders[0]


# -- the law itself ---------------------------------------------------

def test_the_valid_day_is_read_from_every_stamp_spelling():
    """WRF Times, the filename-safe form, and ISO all name one day."""

    assert render_layout.valid_day("1974-04-03_18:00:00") == "1974-04-03"
    assert render_layout.valid_day("1974-04-03_18-00-00") == "1974-04-03"
    assert render_layout.valid_day("1974-04-03T18:00:00") == "1974-04-03"
    assert render_layout.valid_day("") is None
    assert render_layout.valid_day("not-a-time") is None


def test_nested_is_the_default():
    assert render_layout.DEFAULT_LAYOUT == render_layout.NESTED


def test_flat_is_the_v241_spelling_exactly():
    """The legacy escape hatch must not invent a third layout."""

    root = Path("out") / "case"
    assert render_layout.place(
        root, domain="d02-3km", product="refl", day="2026-05-20",
        filename="refl_d02-3km_x.png", layout=render_layout.FLAT,
    ) == root / "refl_d02-3km_x.png"


def test_an_unreadable_valid_time_gets_a_named_bucket_not_the_root():
    """A file whose day cannot be read still has ONE predictable home."""

    root = Path("out") / "case"
    assert render_layout.place(
        root, domain="native_grid", product="t2", day=None,
        filename="t2.png",
    ) == root / "native_grid" / "t2" / render_layout.UNDATED / "t2.png"


def test_the_path_is_a_pure_function_of_domain_product_and_day():
    """Predictable without globbing -- the whole point of documenting it."""

    root = Path("out") / "case"
    assert render_layout.place(
        root, domain="d04-100m", product="refl", day="2026-05-20",
        filename="refl_d04-100m_2026-05-20_18-00-00.png",
    ) == (root / "d04-100m" / "refl" / "2026-05-20"
          / "refl_d04-100m_2026-05-20_18-00-00.png")


# -- the matplotlib engine --------------------------------------------

def test_the_render_door_writes_the_nested_layout_by_default(wrfout,
                                                             tmp_path):
    """No flag: `woof render` splits by domain, product and valid day."""

    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout),
                   "--products", "refl,t2", "--out", str(out),
                   "--dpi", "72"])
    assert rc == 0
    run_dir = _run_dir(out)
    assert _relative_pngs(run_dir) == sorted(
        f"d02-1km/{product}/1974-04-03/"
        f"{product}_d02-1km_{stamp.replace(':', '-')}.png"
        for product in ("refl", "t2")
        for stamp in _STAMPS)
    # ...and nothing at all is left loose in the case root or the run
    # folder.
    assert [p.name for p in out.glob("*.png")] == []
    assert [p.name for p in run_dir.glob("*.png")] == []


def test_the_flat_layout_reproduces_the_published_release_paths(wrfout,
                                                                tmp_path):
    """`--layout flat` is the v2.4.1 directory, unchanged."""

    expected = sorted(f"refl_d02-1km_{stamp.replace(':', '-')}.png"
                      for stamp in _STAMPS)
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout),
                   "--products", "refl", "--out", str(out),
                   "--layout", "flat", "--dpi", "72"])
    assert rc == 0
    assert _relative_pngs(_run_dir(out)) == expected
    # And with the run stamp off as well it is the v2.4.1 tree exactly:
    # every picture directly under --out, no intervening directory of
    # any kind.  That pair of flags is the whole compatibility escape.
    bare = tmp_path / "bare"
    assert cli.main(["render", "--engine", "matplotlib", str(wrfout),
                     "--products", "refl", "--out", str(bare),
                     "--layout", "flat", "--run-stamp", "off",
                     "--dpi", "72"]) == 0
    assert _layout_entries(bare) == expected


def test_two_nests_and_five_products_no_longer_share_one_directory(
        tmp_path):
    """The reported defect, measured: files-per-directory, before/after.

    Two domains x five products x two frames = 20 PNGs.  Flat puts all
    20 in one directory; nested puts at most two (the two frames of one
    product of one nest, which is exactly the set a reader scrubs
    through).
    """

    inner = tmp_path / "in"
    inner.mkdir()
    _write_wrfout(inner / "wrfout_d02_1974-04-03_18-00-00.nc", dx=3000.0)
    _write_wrfout(inner / "wrfout_d03_1974-04-03_18-00-00.nc", dx=1000.0)
    frames = sorted(str(p) for p in inner.glob("wrfout_d*"))

    flat = tmp_path / "flat"
    assert cli.main(["render", "--engine", "matplotlib", *frames,
                     "--out", str(flat), "--layout", "flat",
                     "--dpi", "72"]) == 0
    nested = tmp_path / "nested"
    assert cli.main(["render", "--engine", "matplotlib", *frames,
                     "--out", str(nested), "--dpi", "72"]) == 0

    flat, nested = _run_dir(flat), _run_dir(nested)
    flat_pngs = list(flat.rglob("*.png"))
    nested_pngs = list(nested.rglob("*.png"))
    assert len(flat_pngs) == len(nested_pngs) == 20
    # Same pictures, same filenames -- only the directories moved.
    assert (sorted(p.name for p in flat_pngs)
            == sorted(p.name for p in nested_pngs))

    def busiest(paths):
        counts: dict[Path, int] = {}
        for path in paths:
            counts[path.parent] = counts.get(path.parent, 0) + 1
        return max(counts.values())

    assert busiest(flat_pngs) == 20
    assert busiest(nested_pngs) == 2
    assert _layout_entries(nested) == ["d02-3km", "d03-1km"]


# -- the rust engine ---------------------------------------------------

def test_the_engine_filename_yields_domain_product_and_valid_day():
    """`arwen_<model>_<date>_<cycle>z_f<lead>_<domain>_<product>.png`.

    The lead is added to the cycle, so the day a frame is filed under is
    the day it is VALID, not the day the run was initialised -- a 21z
    cycle at f+06 belongs to the next morning.
    """

    parsed = render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f001_d02-3km_composite_reflectivity.png")
    assert parsed == ("d02-3km", "composite_reflectivity", "1974-04-03")

    crossed = render_layout.parse_engine_output(
        "rustwx_wrf_19740403_21z_f006_d05-111m_total_qpf.png")
    assert crossed == ("d05-111m", "total_qpf", "1974-04-04")

    anonymous = render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f000_native_grid_2m_temperature.png")
    assert anonymous == ("native_grid", "2m_temperature", "1974-04-03")

    assert render_layout.parse_engine_output("not-an-engine-file.png") is None


def test_the_engine_writes_single_digit_cycle_hours_and_they_parse():
    """00Z-09Z cycles: the engine formats its u8 cycle hour UNPADDED.

    ``rusty-weather/src/store_render.rs`` line 488 writes
    ``rustwx_{}_{}_{}z_f{:03}_..`` with ``cycle_utc: u8`` in the plain
    ``{}`` slot, so a 06Z run is ``_6z_`` -- one digit, ten of the
    twenty-four cycle hours, GFS 00Z and 06Z among them.  A parser that
    demands two digits returns None for all of them, and every frame of
    those runs was left flat under a front door printing
    ``layout nested``.
    """

    # The release-blocker case, spelled as rw_wrfbatch spells it.
    assert render_layout.parse_engine_output(
        "rustwx_wrf_20260416_6z_f000_d02-3km_composite_reflectivity.png"
    ) == ("d02-3km", "composite_reflectivity", "2026-04-16")

    # 00Z is the shortest spelling the format string can produce.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_0z_f000_native_grid_2m_temperature.png"
    ) == ("native_grid", "2m_temperature", "1974-04-03")

    # The lead still advances the valid day across midnight.
    assert render_layout.parse_engine_output(
        "rustwx_wrf_19740403_9z_f018_d05-111m_total_qpf.png"
    ) == ("d05-111m", "total_qpf", "1974-04-04")

    # ...and the caller's domain hint splits the tail exactly as it
    # does for two-digit cycles.
    assert render_layout.parse_engine_output(
        "rustwx_wrf_20260416_6z_f001_odd_slug_srh_0_1km.png",
        domain="odd_slug",
    ) == ("odd_slug", "srh_0_1km", "2026-04-16")

    # Two-digit control: nothing about the afternoon cycles moved.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f001_d02-3km_composite_reflectivity.png"
    ) == ("d02-3km", "composite_reflectivity", "1974-04-03")

    # A cycle "hour" no clock has stays accurate: dated buckets are for
    # evidence, and 99z is evidence of nothing.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_99z_f000_d02-3km_composite_reflectivity.png"
    ) == ("d02-3km", "composite_reflectivity", render_layout.UNDATED)


def test_the_caller_supplied_domain_token_anchors_the_split():
    """The caller's token splits the tail; the filename names the folder.

    Domain slugs and product slugs both contain underscores, so the
    token the caller read from the wrfout is what tells the parser where
    one ends and the other begins.  What it does NOT do is rename the
    folder: a ``d02-3km`` file filed under ``d02/`` would leave a reader
    asking which of the two is lying.  In a real render the two strings
    are identical; they diverge only when one side could read less of
    the file than the other.
    """

    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity.png",
        domain="d02-3km",
    ) == ("d02-3km", "composite_reflectivity", "1974-04-03")

    # A caller that could not read DX still gets the engine's spelling.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity.png",
        domain="d02",
    ) == ("d02-3km", "composite_reflectivity", "1974-04-03")

    # An underscore-bearing product the grammar cannot split on its own
    # is split by the caller's token.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f000_odd_slug_srh_0_1km.png",
        domain="odd_slug",
    ) == ("odd_slug", "srh_0_1km", "1974-04-03")


def test_a_sub_hourly_frame_files_beside_its_siblings_not_in_its_own_folder():
    """The engine's exact-time suffix is a FRAME id, not a product name.

    ``f{NNN}`` carries whole hours only, so the vendored engine
    disambiguates sub-hourly frames with a suffix of its own --
    ``rusty-weather/src/render_all.rs`` builds
    ``valid_19740403_183000z_lead_000h30m00s`` and appends it to the
    filename.  A parser with no rule for it reads the whole thing as the
    product, and then every frame of one product gets a product folder of
    its own: the tile-streamed lane's 10-minute cadence turned twenty
    products into sixty folders, which is the flat directory the layout
    ruling exists to prevent, wearing a nested costume.

    On Windows it was worse than untidy.  ``<out>/<domain>/<product>_
    valid_..._lead_.../<day>/<same 100-character name>.png`` runs past
    MAX_PATH, ``os.replace`` fails, and the frame is left flat -- four
    of this lane's own panels landed that way in a real render.

    The suffix also carries the frame's EXACT valid stamp, which is
    strictly better evidence of the valid day than cycle + whole-hour
    lead: a 23z run's 30-minute frame is dated by what the engine wrote,
    not by an hour count that truncated it.
    """

    # 18Z + 30 min: same product folder as the whole-hour frames.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity"
        "_valid_19740403_183000z_lead_000h30m00s.png"
    ) == ("d02-3km", "composite_reflectivity", "1974-04-03")

    # ...and the whole-hour sibling lands in that same folder.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f001_d02-3km_composite_reflectivity.png"
    ) == ("d02-3km", "composite_reflectivity", "1974-04-03")

    # The suffix's own stamp dates the frame: a 23Z run's 90-minute
    # frame is valid the NEXT day, and f001 alone would not say so.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_23z_f001_d05-111m_total_qpf"
        "_valid_19740404_003000z_lead_001h30m00s.png"
    ) == ("d05-111m", "total_qpf", "1974-04-04")

    # A caller-supplied domain token splits the tail the same way, and
    # the suffix still comes off the product.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f000_odd_slug_srh_0_1km"
        "_valid_19740403_181500z_lead_000h15m00s.png",
        domain="odd_slug",
    ) == ("odd_slug", "srh_0_1km", "1974-04-03")

    # A product whose own name merely CONTAINS the word is untouched --
    # the rule is the engine's exact grammar, anchored at the end, not a
    # search for "valid".
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f000_d02-3km_valid_hours_since_analysis.png"
    ) == ("d02-3km", "valid_hours_since_analysis", "1974-04-03")

    # A suffix-shaped tail with an impossible stamp is not evidence of a
    # day, and must not become one; the product still loses the suffix,
    # because the engine wrote it and it is still not the product's name.
    assert render_layout.parse_engine_output(
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity"
        "_valid_19740231_183000z_lead_000h30m00s.png"
    ) == ("d02-3km", "composite_reflectivity", render_layout.UNDATED)


def test_the_rust_engine_output_is_relocated_into_the_layout(monkeypatch,
                                                             tmp_path):
    """The engine writes flat into --out; the product files it away.

    Stubbed at ``subprocess.run`` exactly as the other rust-engine unit
    tests are: what is under test is the placement, not the drawing.
    """

    from woof import render as render_module

    wrfout = _write_wrfout(tmp_path / "wrfout_d02_1974-04-03_18-00-00.nc",
                           dx=3000.0)
    out = tmp_path / "png"
    names = (
        "rustwx_wrf_19740403_18z_f000_d02-3km_composite_reflectivity.png",
        "rustwx_wrf_19740403_18z_f001_d02-3km_composite_reflectivity.png",
        "rustwx_wrf_19740403_18z_f000_d02-3km_2m_temperature.png",
    )

    _stub_flat_renderer(monkeypatch, tmp_path, names)

    written, failures, skipped = render_module.render_wrfouts_rust(
        [wrfout], products="all", timeidx=None, outdir=out,
        size=(800, 600), source_label="WOOF test")
    assert failures == [] and skipped == []
    # The delivered name carries the frame's identity only: the domain
    # and product tokens are the two folders directly above it, and a
    # frame repeating them ran real deliveries past the Windows ceiling.
    assert sorted(p.relative_to(out).as_posix() for p in written) == sorted((
        "d02-3km/composite_reflectivity/1974-04-03/"
        "arwen_wrf_19740403_18z_f000.png",
        "d02-3km/composite_reflectivity/1974-04-03/"
        "arwen_wrf_19740403_18z_f001.png",
        "d02-3km/2m_temperature/1974-04-03/"
        "arwen_wrf_19740403_18z_f000.png"))
    for path in written:
        assert path.is_file(), path
    assert [p.name for p in out.glob("*.png")] == []


def test_a_06z_case_is_filed_nested_not_silently_left_flat(monkeypatch,
                                                           tmp_path):
    """The release blocker at the placement seam, with real 06Z names.

    Same stub shape as the relocation test above; the names are what
    ``rw_wrfbatch`` actually writes for a 06Z-initialised run -- the
    cycle hour unpadded.  Before the parse accepted one digit, every one
    of these files stayed loose in the case root while the front door
    printed ``layout nested``.
    """

    from woof import render as render_module

    wrfout = _write_wrfout(tmp_path / "wrfout_d02_2026-04-16_06-00-00.nc",
                           dx=3000.0)
    out = tmp_path / "png"
    names = (
        "rustwx_wrf_20260416_6z_f000_d02-3km_composite_reflectivity.png",
        "rustwx_wrf_20260416_6z_f001_d02-3km_composite_reflectivity.png",
        "rustwx_wrf_20260416_6z_f000_d02-3km_2m_temperature.png",
    )

    _stub_flat_renderer(monkeypatch, tmp_path, names)

    written, failures, skipped = render_module.render_wrfouts_rust(
        [wrfout], products="all", timeidx=None, outdir=out,
        size=(800, 600), source_label="WOOF test")
    assert failures == [] and skipped == []
    assert sorted(p.relative_to(out).as_posix() for p in written) == sorted((
        "d02-3km/composite_reflectivity/2026-04-16/"
        "arwen_wrf_20260416_6z_f000.png",
        "d02-3km/composite_reflectivity/2026-04-16/"
        "arwen_wrf_20260416_6z_f001.png",
        "d02-3km/2m_temperature/2026-04-16/"
        "arwen_wrf_20260416_6z_f000.png"))
    # Nothing loose in the case root: the 06Z defect was ALL of these
    # staying exactly here.
    assert [p.name for p in out.glob("*.png")] == []


def test_an_engine_output_that_defeats_the_parse_lands_flat_but_loudly(
        monkeypatch, tmp_path, capsys):
    """Parse failure must never silently degrade the layout.

    A name the grammar cannot read is left where the engine put it --
    layout is not correctness, a flat picture beats a refusal -- but the
    degradation is ANNOUNCED with the file and the reason, because the
    silent form of this exact fallback is how ten of twenty-four cycle
    hours shipped a flat directory under a ``layout nested`` banner.
    """

    from woof import render as render_module

    wrfout = _write_wrfout(tmp_path / "wrfout_d02_2026-04-16_06-00-00.nc",
                           dx=3000.0)
    out = tmp_path / "png"
    names = (
        "rustwx_wrf_20260416_6z_f000_d02-3km_composite_reflectivity.png",
        # No date, no cycle, no lead: nothing the grammar can file by.
        "rustwx_scratch_note.png",
    )

    _stub_flat_renderer(monkeypatch, tmp_path, names)

    written, failures, skipped = render_module.render_wrfouts_rust(
        [wrfout], products="all", timeidx=None, outdir=out,
        size=(800, 600), source_label="WOOF test")
    assert failures == [] and skipped == []

    # The parseable frame is filed; the alien one stays flat (rebranded,
    # as every engine output is) and is still in the returned list.
    assert sorted(p.relative_to(out).as_posix() for p in written) == sorted((
        "d02-3km/composite_reflectivity/2026-04-16/"
        "arwen_wrf_20260416_6z_f000.png",
        "arwen_scratch_note.png"))
    assert (out / "arwen_scratch_note.png").is_file()

    # ...and the degradation says so, naming the file.
    err = capsys.readouterr().err
    assert "left flat" in err
    assert "arwen_scratch_note.png" in err
    # The frame that WAS filed is not warned about.
    assert "composite_reflectivity" not in err


# -- the consumers -----------------------------------------------------

def test_iter_rendered_reads_both_layouts_and_ignores_scratch(tmp_path):
    """Every in-tree reader goes through one walker, and it skips
    dot-directories -- the early render's scratch is a dot-sibling of
    the pictures, and a recursive reader would otherwise publish a
    half-written run's temporaries."""

    (tmp_path / "flat.png").write_bytes(b"PNG")
    nested = tmp_path / "d02-3km" / "refl" / "1974-04-03"
    nested.mkdir(parents=True)
    (nested / "deep.png").write_bytes(b"PNG")
    scratch = tmp_path / ".first-products-scratch" / "d02-3km"
    scratch.mkdir(parents=True)
    (scratch / "inflight.png").write_bytes(b"PNG")

    found = [p.name for p in render_layout.iter_rendered(tmp_path)]
    assert sorted(found) == ["deep.png", "flat.png"]


# -- path length is part of "it never loses a file" --------------------

_LONG_LEAF = ("arwen_wrf_19740403_18z_f000_d01-1km_"
              "composite_reflectivity.png")


def _too_long_for_windows(root: Path) -> Path:
    """A layout path past MAX_PATH, built out of real segment names."""

    padding = "a" * max(
        1, 268 - len(str(root / "d01-1km" / "composite_reflectivity"
                         / "1974-04-03" / _LONG_LEAF)))
    return (root / padding / "d01-1km" / "composite_reflectivity"
            / "1974-04-03" / _LONG_LEAF)


def test_a_path_past_the_windows_ceiling_is_still_spelled_and_written(
        tmp_path):
    """The layout is not allowed to lose the longest product names.

    Measured on the first-products path: at the default pytest temp
    root, ``composite_reflectivity`` came to 261 characters and its move
    into the tree failed with ERROR_PATH_NOT_FOUND, so the picture was
    left FLAT at the render root -- the 2026-08-06 ruling inverted for
    one product while ``2m_temperature`` beside it, 17 characters
    shorter, filed correctly.  A ceiling that files some frames and not
    others is the worst version of this: the directory looks right.
    """

    import os

    target = _too_long_for_windows(tmp_path)
    assert len(str(target)) > 260, len(str(target))

    spelled = render_layout.fs_path(target)
    Path(spelled).parent.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "engine-output.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\nfiled")
    os.replace(render_layout.fs_path(source), spelled)

    # Written, findable by the one walker, and readable -- all three, or
    # the file is filed somewhere nothing can reach.
    found = render_layout.iter_rendered(tmp_path)
    assert [p.name for p in found] == [_LONG_LEAF]
    assert found[0] == target, "the walker changed the caller's spelling"
    assert (Path(render_layout.fs_path(found[0])).read_bytes()
            == b"\x89PNG\r\n\x1a\nfiled")


def test_the_placement_seam_files_a_frame_the_ceiling_would_have_dropped(
        tmp_path, capsys):
    """The engine's own seam, at a depth that used to leave it flat.

    ``_place_engine_output`` degrades to "left flat" on any OSError, and
    ERROR_PATH_NOT_FOUND from a too-long target is an OSError like any
    other -- so before this, the deepest case roots silently inverted
    the layout ruling for their longest-named products.
    """

    from woof.render import _place_engine_output

    name = ("rustwx_wrf_19740403_18z_f000_d01-1km_"
            "composite_reflectivity.png")
    # The DELIVERED leaf, which is what the depth has to be measured
    # against: the shortening buys about fifty characters, and a case
    # root deeper than that still spends them.
    filed = "arwen_wrf_19740403_18z_f000.png"
    padding = "b" * max(1, 268 - len(str(
        tmp_path / "d01-1km" / "composite_reflectivity" / "1974-04-03"
        / filed)))
    outdir = tmp_path / padding
    outdir.mkdir(parents=True)
    (outdir / name).write_bytes(b"\x89PNG\r\n\x1a\ndeep")

    placed = _place_engine_output(outdir / name, outdir, "d01-1km",
                                  render_layout.NESTED)

    assert len(str(placed)) > 260, len(str(placed))
    assert placed.relative_to(outdir).as_posix() == (
        f"d01-1km/composite_reflectivity/1974-04-03/{filed}")
    assert "left flat" not in capsys.readouterr().err
    assert [p for p in render_layout.iter_rendered(outdir)] == [placed]


def test_a_path_within_the_ceiling_keeps_the_callers_own_spelling(tmp_path):
    """A spelling, never a different file, and never a louder one."""

    short = tmp_path / "d01-1km" / "composite_reflectivity" / _LONG_LEAF
    assert render_layout.fs_path(short) == str(short)
    assert render_layout.fs_path(Path("relative") / "x.png") == str(
        Path("relative") / "x.png")


def test_pair_compose_matches_products_across_the_nested_layout(tmp_path):
    """`woof render --pair` must keep working once the files moved."""

    pytest.importorskip("PIL", reason="--pair needs Pillow")
    from PIL import Image

    from woof.pair_compose import compose_pairs

    name = ("arwen_wrf_19740403_18z_f000_d02-3km_"
            "composite_reflectivity.png")
    left, right = tmp_path / "a", tmp_path / "b"
    for root in (left, right):
        target = root / "d02-3km" / "composite_reflectivity" / "1974-04-03"
        target.mkdir(parents=True)
        Image.new("RGB", (64, 48), "#123456").save(target / name)

    sheets = compose_pairs(left, right, tmp_path / "pairs", title="t")
    assert len(sheets) == 1
    assert sheets[0].name == "d02-3km_composite_reflectivity-pair.png"


def test_the_early_render_publishes_the_layout_it_drew(tmp_path):
    """first_products moves the scratch tree over, structure intact.

    The receipt names each picture by its path RELATIVE to the render
    directory, so the finalize stage's digest re-check finds them where
    the early render put them instead of looking for a flat name that
    no longer exists.
    """

    from woof import first_products

    render_dir = tmp_path / "png"
    render_dir.mkdir()
    frame = tmp_path / "wrfout_d01_1974-04-03_18_00_00"
    frame.write_bytes(b"not really a wrfout, only its digest is read")

    relative = "d01-1km/refl/1974-04-03/refl_d01-1km_1974-04-03_18-00-00.png"

    trigger = first_products.FirstProducts(
        {"render": render_dir, "render_products": "refl"},
        report=lambda _event: None,
        warn=lambda *args, **kwargs: None,
        runner=_early_runner(relative))
    trigger.frame_committed(domain=1, valid_time="1974-04-03T18:00:00",
                            path=frame)
    receipt = trigger.wait(timeout=60.0)

    assert receipt is not None
    assert [entry["name"] for entry in receipt["written"]] == [relative]
    assert (render_dir / relative).read_bytes() == b"PNG-bytes"
    # No scratch tree survives, and nothing was flattened on the way.
    assert not (render_dir / ".first-products-scratch").exists()

    remaining, already, note = first_products.published_frames(
        [frame], {"render": render_dir, "render_products": "refl"})
    assert remaining == [] and already == [frame]
    assert note is not None and "digests verified" in note


def test_the_documented_layout_string_names_every_segment():
    """The one sentence a script author reads; it must stay true."""

    described = render_layout.describe()
    for segment in ("<domain>", "<product>", "<valid-day>", "YYYY-MM-DD"):
        assert segment in described, described


# -- the front door ----------------------------------------------------

_DOC = Path(__file__).resolve().parents[1] / "docs" / "render-output-layout.md"


def test_the_document_a_script_author_reads_is_the_layout_the_code_writes():
    """The page's own recipe, executed, against `place()`.

    A layout is only predictable if the documented prediction is the
    real one, so the page's `product_dir` recipe is lifted out of it and
    run rather than read: a doc that drifts from the code recreates the
    globbing it exists to remove.
    """

    text = _DOC.read_text(encoding="utf-8")
    # The run-stamped level (woof.run_stamp) sits ABOVE this layout and
    # the page says so; the three segments below it are what this file
    # is the gate for, and they are unchanged.
    assert ("<--out>/<run folder>/<domain>/<product>/<valid-day>"
            "/<filename>.png") in text
    # And the one extra segment a nest that retires and re-arms gets,
    # spelled where a script author will read it.
    assert ("<--out>/<run folder>/<domain>/<episode>/<product>/"
            "<valid-day>/<filename>.png") in text
    assert "run-output-folders.md" in text
    # The escape hatch is named as a compatibility measure, never as the
    # fix -- "fixed means default".
    assert "--layout flat" in text
    assert "There is no flag to turn it on." in text

    recipe = text.split("```python", 1)[1].split("```", 1)[0]
    namespace: dict = {}
    exec(compile(recipe, str(_DOC), "exec"), namespace)      # noqa: S102
    predicted = namespace["product_dir"](
        "out/myarea/png", "d04-100m", "composite_reflectivity",
        datetime.datetime(1974, 4, 4, 0, 0))
    actual = render_layout.place(
        "out/myarea/png", domain="d04-100m",
        product="composite_reflectivity", day="1974-04-04",
        filename="x.png").parent
    assert Path(predicted) == actual
    # The same recipe, for a nest on its second life.
    predicted_episode = namespace["product_dir"](
        "out/myarea/png", "d05-500m", "composite_reflectivity",
        datetime.datetime(1974, 4, 4, 0, 0), episode=2)
    assert Path(predicted_episode) == render_layout.place(
        "out/myarea/png", domain="d05-500m",
        product="composite_reflectivity", day="1974-04-04", episode=2,
        filename="x.png").parent


def test_the_render_door_prints_where_it_is_about_to_write(wrfout,
                                                           tmp_path,
                                                           capsys):
    """Printed BEFORE the pictures, so a script can watch one path.

    The reachability leg for this feature: `--layout` is on the parser,
    it defaults to nested, and the line the door prints names the
    directory the files then actually appear in.
    """

    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout),
                   "--products", "t2", "--timeidx", "0",
                   "--out", str(out), "--dpi", "72"])
    assert rc == 0
    printed = capsys.readouterr().out
    run_dir = _run_dir(out)
    assert "render: layout nested" in printed
    # The line names THIS RUN's folder, not just --out: the run-stamped
    # level sits above the layout, and a script told only the case root
    # would watch a directory the pictures never appear in.
    assert render_layout.describe(str(run_dir)) in printed
    assert f"render: run folder {run_dir.name}" in printed
    drawn = sorted(run_dir.rglob("*.png"))
    assert [p.relative_to(run_dir).as_posix() for p in drawn] == [
        "d02-1km/t2/1974-04-03/"
        f"t2_d02-1km_{_STAMPS[0].replace(':', '-')}.png"]


# -- a delivered path has to fit the tools that OPEN it -----------------

#: A case root of the shape a real delivery has: the user's own folder,
#: a per-case directory under it, and the render directory the front
#: door hands the renderer (``woof go`` gives it ``<case>/png``).
#: Fifty-five characters, and nothing below the case root can shorten
#: any of them -- it is the fixed cost every delivered path starts with,
#: and a real one is often deeper than this.  The literal is assembled
#: from fragments so the machine-path scan reads no profile path here --
#: the name is fictional, but the scan cannot know that.
_TYPICAL_CASE_ROOT = (
    "C:\\Us" "ers\\forecaster\\Downloads\\spring-outbreak-study\\png"
)

#: The widest segments a real delivery is measured to carry, each taken
#: from one on disk: a run folder carrying launch instant AND init time;
#: a stored-variable product slug carrying the digest that disambiguates
#: it; and a sub-hourly frame carrying the engine's exact-time suffix,
#: which ``f{NNN}`` cannot express.  None is exotic -- it is one nest of
#: the generic-variable route at a sub-hourly cadence, both defaults.
_DEEP_RUN = "run-20260820-035106Z_i202608200000Z"
_DEEP_DOMAIN = "d01-12km"
_DEEP_PRODUCT = "var_geopotential_height_700hpa_38d0bbbc4b4b7e87"
_DEEP_DAY = "2026-08-20"
_DEEP_ENGINE_NAME = (
    f"rustwx_wrf_20260820_0z_f000_{_DEEP_DOMAIN}_{_DEEP_PRODUCT}"
    "_valid_20260820_003000z_lead_000h30m00s.png")

#: What a delivered path may cost, case root included.  Windows' classic
#: ceiling is 260; this sits twenty characters under it so a case folder
#: named a little longer than the representative one still opens in a
#: tool that carries no long-path manifest.
DELIVERED_PATH_BUDGET = 240


def _delivered_length(relative: Path) -> int:
    """A delivered path's real length: typical case root + the tree."""

    return len(_TYPICAL_CASE_ROOT) + 1 + len(relative.as_posix())


def test_a_delivered_path_fits_the_windows_tools_that_open_it(tmp_path,
                                                              capsys):
    """The deepest real layout, measured through the placement seam.

    :func:`woof.render_layout.fs_path` lets WOOF itself write and read
    past MAX_PATH, which is why the picture is filed rather than dropped
    -- but it does nothing for the tools the delivery is OPENED with.
    Explorer, the readers a user's own script imports, ``tar``, and any
    GUI without a long-path manifest all refuse the path, so a correctly
    filed picture becomes one the recipient cannot open.

    Measured on a delivery on disk: 310 characters, in which the
    filename repeated verbatim the two folder names directly above it.
    """

    from woof.render import _place_engine_output

    outdir = tmp_path / _DEEP_RUN
    outdir.mkdir(parents=True)
    (outdir / _DEEP_ENGINE_NAME).write_bytes(b"\x89PNG\r\n\x1a\nlong")

    placed = _place_engine_output(outdir / _DEEP_ENGINE_NAME, outdir,
                                  _DEEP_DOMAIN, render_layout.NESTED)
    assert "left flat" not in capsys.readouterr().err

    # The run folder is part of what is delivered, so it is measured.
    relative = Path(_DEEP_RUN) / placed.relative_to(outdir)
    assert relative.parts[1:4] == (_DEEP_DOMAIN, _DEEP_PRODUCT, _DEEP_DAY), (
        "the 2026-08-06 layout ruling must survive the shortening")

    length = _delivered_length(relative)
    assert length <= DELIVERED_PATH_BUDGET, (
        f"{length} characters: {_TYPICAL_CASE_ROOT}\\{relative}")


#: What an EPISODIC delivery may cost.  The episode segment is a folder,
#: so it costs its own name plus a separator on every path under it, and
#: that cost is stated here rather than discovered on a delivery: the
#: measured 310-character tree that produced :func:`delivered_name` was
#: found by a recipient, not by a test.  Still under Windows' 260 --
#: with less slack than an ordinary delivery has, which is the accurate
#: price of separating a nest's two lives.
EPISODIC_PATH_BUDGET = (DELIVERED_PATH_BUDGET
                        + len(render_layout.episode_segment(999)) + 1)


def test_an_episodic_delivery_still_fits_the_ceiling(tmp_path, capsys):
    """The deepest real layout AND a second life of the nest.

    The segment buys collision-free galleries with path length, and
    length is the currency this layout has already been billed in.  The
    same deepest-real-delivery measurement, one episode deeper.
    """

    from woof.render import _place_engine_output

    outdir = tmp_path / _DEEP_RUN
    outdir.mkdir(parents=True)
    (outdir / _DEEP_ENGINE_NAME).write_bytes(b"\x89PNG\r\n\x1a\nlong")

    placed = _place_engine_output(outdir / _DEEP_ENGINE_NAME, outdir,
                                  _DEEP_DOMAIN, render_layout.NESTED,
                                  episode=2)
    assert "left flat" not in capsys.readouterr().err

    relative = Path(_DEEP_RUN) / placed.relative_to(outdir)
    assert relative.parts[1:5] == (_DEEP_DOMAIN, "episode-002",
                                   _DEEP_PRODUCT, _DEEP_DAY)

    length = _delivered_length(relative)
    assert length <= EPISODIC_PATH_BUDGET, (
        f"{length} characters: {_TYPICAL_CASE_ROOT}\\{relative}")
    # And still inside the classic ceiling itself, which is the line
    # that decides whether a recipient's own tools can open the file.
    assert length < 260, length


def test_the_shortening_takes_only_what_the_folders_already_spell():
    """The two repeated tokens, and nothing else.

    What is dropped is exactly what the reader can see by looking up one
    and two folders.  What stays is the frame's own identity and the
    engine's exact-time suffix, which no folder carries -- take that off
    and two sub-hourly frames of one product on one day become one file.
    """

    assert render_layout.delivered_name(
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity.png",
        domain="d02-3km", product="composite_reflectivity",
    ) == "arwen_wrf_19740403_18z_f000.png"

    assert render_layout.delivered_name(
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity"
        "_valid_19740403_183000z_lead_000h30m00s.png",
        domain="d02-3km", product="composite_reflectivity",
    ) == "arwen_wrf_19740403_18z_f000_valid_19740403_183000z_lead_000h30m00s.png"

    # A one-digit cycle hour is what the engine writes for ten of the
    # twenty-four cycles, and it must not be the thing that opts a whole
    # run out of the shortening.
    assert render_layout.delivered_name(
        "arwen_wrf_20260416_6z_f001_d05-111m_total_qpf.png",
        domain="d05-111m", product="total_qpf",
    ) == "arwen_wrf_20260416_6z_f001.png"

    # Folders that do not spell what the name carries, and a name the
    # grammar cannot read at all: unchanged, never cut on a guess.
    assert render_layout.delivered_name(
        "arwen_wrf_19740403_18z_f000_d02-3km_total_qpf.png",
        domain="d05-111m", product="total_qpf",
    ) == "arwen_wrf_19740403_18z_f000_d02-3km_total_qpf.png"
    assert render_layout.delivered_name(
        "not-an-engine-file.png", domain="d02-3km", product="t2",
    ) == "not-an-engine-file.png"
    assert render_layout.delivered_name(
        "arwen_wrf_19740403_18z_f000_odd_tail.png",
        domain=None, product=None,
    ) == "arwen_wrf_19740403_18z_f000_odd_tail.png"


def test_a_delivered_name_still_names_exactly_one_frame():
    """Collision-freedom, proved by rebuilding what was taken away.

    :func:`render_layout.engine_name` reconstructs the engine's own
    filename from a delivered one and its two folders, so the shortening
    is a bijection within a folder: two frames that were different files
    before are different files after.  That is the property members and
    valid times need -- a member renders under its own model token and a
    sub-hourly frame under its own exact-time suffix, and both are on
    the side that survives.
    """

    folder = {"domain": "d02-3km", "product": "composite_reflectivity"}
    engine = [
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity.png",
        "arwen_wrf_19740403_18z_f001_d02-3km_composite_reflectivity.png",
        # Two members of one cycle, one valid time: the model token is
        # what tells them apart, and the shortening does not touch it.
        "arwen_gefs-m01_19740403_18z_f001_d02-3km_"
        "composite_reflectivity.png",
        "arwen_gefs-m02_19740403_18z_f001_d02-3km_"
        "composite_reflectivity.png",
        # Two sub-hourly frames inside one hour.
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity"
        "_valid_19740403_181000z_lead_000h10m00s.png",
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity"
        "_valid_19740403_182000z_lead_000h20m00s.png",
    ]
    delivered = [render_layout.delivered_name(name, **folder)
                 for name in engine]
    assert len(set(delivered)) == len(engine), delivered
    assert [render_layout.engine_name(name, **folder)
            for name in delivered] == engine

    # Idempotent both ways: a tree half-migrated by a re-render into an
    # existing run folder must not gain a second copy of either token.
    assert [render_layout.delivered_name(name, **folder)
            for name in delivered] == delivered
    assert [render_layout.engine_name(name, **folder)
            for name in engine] == engine


def test_pair_matching_survives_the_shortening(tmp_path):
    """`--pair` keys on the engine's spelling, rebuilt from the folders.

    Keyed off the delivered name alone, every frame in a directory keys
    to its run identity -- which two compared runs differ in by
    definition -- and ``--pair`` refuses every nested render with "no
    matching product PNGs".
    """

    from woof.pair_compose import product_name

    legacy = (tmp_path / "d02-3km" / "composite_reflectivity"
              / "1974-04-03"
              / "arwen_wrf_19740403_18z_f000_d02-3km_"
                "composite_reflectivity.png")
    short = (tmp_path / "d02-3km" / "composite_reflectivity"
             / "1974-04-03" / "arwen_wrf_19740403_22z_f002.png")
    assert product_name(short) == "d02-3km_composite_reflectivity"
    assert product_name(short) == product_name(legacy), (
        "a migrating directory must pair against a delivered one")

    # Two sub-hourly frames of one product keep separate keys, or the
    # sheet is composed from whichever frame the dict saw last.
    first, second = (
        (tmp_path / "d02-3km" / "composite_reflectivity" / "1974-04-03"
         / f"arwen_wrf_19740403_18z_f000_valid_19740403_18{minute}00z"
           "_lead_000h30m00s.png")
        for minute in ("10", "20"))
    assert product_name(first) != product_name(second)

    # A flat directory has no folders to read, and keys as it always did.
    flat = tmp_path / (
        "arwen_wrf_19740403_18z_f000_d02-3km_composite_reflectivity.png")
    assert product_name(flat) == "d02-3km_composite_reflectivity"


# -- one nest, two lives -----------------------------------------------
#
# A domain that declares ``retire``/``rearm`` puts a SECOND history run
# through one slot.  The history writer already separates those on disk
# (``d05/episode-002/``), because two episodes of one nest can publish
# the SAME valid time: the retiring episode's last frame and the
# re-armed episode's activation frame land on the same instant, and the
# writer's own duplicate guard names that as a frame-destroying event.
#
# The delivered PNG tree had no such segment.  Both of those frames
# carry one domain token, one product slug and one valid day, so they
# rendered to ONE delivered name and the later one replaced the earlier
# -- silently, because a render that overwrites reports success.  What
# is left is a single gallery holding frames from two different lives of
# the nest with no way to tell which is which.


def test_an_episode_is_spelled_one_way_for_the_whole_tree():
    """One fact, one spelling: the history tree's and the render tree's.

    The number is the lifecycle's own (``SpawnRunner.episodes`` through
    :func:`woof.core.nest_lifecycle.output_episode`), which reports 0
    for a domain that declares no lifecycle -- so 0 is not an episode
    here either, it is the absence of one.
    """

    assert render_layout.episode_segment(1) == "episode-001"
    assert render_layout.episode_segment(2) == "episode-002"
    # Three digits is a minimum width, not a ceiling.
    assert render_layout.episode_segment(1234) == "episode-1234"
    assert render_layout.episode_segment(0) is None
    assert render_layout.episode_segment(None) is None

    assert render_layout.episode_number("episode-002") == 2
    assert render_layout.episode_number("episode-1234") == 1234
    # `episode-000` is what a lifecycle-free domain would spell if
    # anything ever wrote it; it names no episode and must not be read
    # as one.
    assert render_layout.episode_number("episode-000") is None
    assert render_layout.episode_number("d05-500m") is None
    assert render_layout.episode_number("composite_reflectivity") is None
    assert render_layout.episode_number("1974-04-03") is None
    assert render_layout.episode_number(None) is None
    for number in (1, 2, 9, 10, 99, 100, 999, 1000):
        assert render_layout.episode_number(
            render_layout.episode_segment(number)) == number


def test_a_run_with_no_lifecycle_files_exactly_where_it_always_did():
    """The byte-identity promise: no episode, no segment, no change.

    Every existing run is this case.  ``retire``/``rearm`` are what put
    a second episode through a slot; a plain run, a one-shot spawn and a
    following nest all report no episode, and their paths must be the
    same strings as before this segment existed.
    """

    root = Path("out") / "case"
    legacy = (root / "d02-3km" / "composite_reflectivity" / "1974-04-03"
              / "arwen_wrf_19740403_18z_f000.png")
    for absent in (None, 0):
        assert render_layout.place(
            root, domain="d02-3km", product="composite_reflectivity",
            day="1974-04-03", episode=absent,
            filename="arwen_wrf_19740403_18z_f000.png") == legacy
    # And the call that does not mention episodes at all, which is every
    # caller written before this segment existed.
    assert render_layout.place(
        root, domain="d02-3km", product="composite_reflectivity",
        day="1974-04-03",
        filename="arwen_wrf_19740403_18z_f000.png") == legacy
    assert render_layout.product_dir(
        domain="d02-3km", product="composite_reflectivity",
        day="1974-04-03") == render_layout.product_dir(
            domain="d02-3km", product="composite_reflectivity",
            day="1974-04-03", episode=0)
    # The escape hatch has no folders at all, so it cannot grow one.
    assert render_layout.place(
        root, domain="d02-3km", product="refl", day="1974-04-03",
        episode=2, filename="x.png",
        layout=render_layout.FLAT) == root / "x.png"


def test_the_episode_extends_the_ruling_instead_of_inverting_it():
    """Domain, then episode, then product, then valid day.

    The 2026-08-06 ruling's order is preserved exactly -- domain before
    product before valid day -- with the episode slotted under the
    domain it belongs to, which is where the history tree already puts
    it.  Nothing collapses back towards flat.
    """

    root = Path("out") / "case"
    placed = render_layout.place(
        root, domain="d05-500m", product="composite_reflectivity",
        day="1974-04-03", episode=2,
        filename="arwen_wrf_19740403_18z_f001.png")
    assert placed == (root / "d05-500m" / "episode-002"
                      / "composite_reflectivity" / "1974-04-03"
                      / "arwen_wrf_19740403_18z_f001.png")
    ordering = [part for part in placed.parts
                if part in ("d05-500m", "episode-002",
                            "composite_reflectivity", "1974-04-03")]
    assert ordering == ["d05-500m", "episode-002",
                        "composite_reflectivity", "1974-04-03"]


def test_the_history_writer_and_the_delivered_tree_agree_on_the_segment():
    """The two trees spell one episode ONE way, from one definition.

    If they drift, a reader who found ``d05/episode-002`` in the history
    tree looks for the pictures of that episode under a name the render
    never wrote, and the collision this segment prevents comes back
    wearing a different spelling.
    """

    from datetime import datetime as _datetime

    from woof.io.wrfout import PerDomainWrfoutWriters

    class _Recorder:
        def __init__(self):
            self.paths = []
            self.global_attrs = {}

        def submit(self, path, valid_time, state, **kwargs):
            self.paths.append(path)

    import tempfile
    root = Path(tempfile.mkdtemp())
    writers = object.__new__(PerDomainWrfoutWriters)
    writers.output_dir = root
    writers.start_time = _datetime(1974, 4, 3, 18, 0, 0)
    writers._metadata_by_grid_id = {5: {}}
    writers._archived_paths = []
    writers._episode_by_grid_id = {5: 2}
    writers._writers = {5: _Recorder()}
    writers._published_paths = set()

    import types
    node = types.SimpleNamespace(
        cfg=types.SimpleNamespace(grid_id=5, run=types.SimpleNamespace(
            mean_diag=0, diurnal_diag=0, output_diagnostics=0, auxhist3_interval=0)),
        clock=types.SimpleNamespace(tick_den=1),
        state=types.SimpleNamespace())
    writers.submit(node, 0)
    written = writers._writers[5].paths[0]
    assert written.parent == (root / "d05"
                              / render_layout.episode_segment(2))


def test_two_episodes_of_one_domain_no_longer_share_a_delivered_name(
        monkeypatch, tmp_path):
    """The ledger's breakage, through the real placement seam.

    Two history files of ONE nest at ONE valid time -- the retiring
    episode's final frame and the re-armed episode's activation frame --
    render to the engine's identical filename.  Without the episode
    segment the second ``os.replace`` lands on the first and one picture
    is gone, with no failure, no warning and no way to notice from the
    tree.
    """

    import subprocess

    from woof import render as render_module

    first = tmp_path / "wrfout" / "d05" / "episode-001"
    second = tmp_path / "wrfout" / "d05" / "episode-002"
    for directory in (first, second):
        directory.mkdir(parents=True)
    stem = "wrfout_d05_1974-04-03_19-00-00.nc"
    inputs = [_write_wrfout(first / stem, dx=500.0),
              _write_wrfout(second / stem, dx=500.0)]
    out = tmp_path / "png"
    # ONE name, from both inputs: the engine's filename carries the
    # cycle, the lead, the domain and the product, and both episodes
    # agree on every one of them.
    name = ("rustwx_wrf_19740403_18z_f001_d05-500m_"
            "composite_reflectivity.png")

    drawn = []

    def fake_run(command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        # Different bytes per invocation, so a replaced picture is
        # detectable rather than merely suspected.
        drawn.append(len(drawn))
        (out_dir / name).write_bytes(f"PNG-{drawn[-1]}".encode())
        return _RendererResult(f"RENDERED slug {out_dir / name}\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(render_module, "renderer_refusal", lambda _r: None)
    monkeypatch.setattr("woof.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch")

    written, failures, skipped = render_module.render_wrfouts_rust(
        inputs, products="all", timeidx=None, outdir=out,
        size=(800, 600), source_label="WOOF test")
    assert failures == [] and skipped == []
    assert sorted(p.relative_to(out).as_posix() for p in written) == [
        "d05-500m/episode-001/composite_reflectivity/1974-04-03/"
        "arwen_wrf_19740403_18z_f001.png",
        "d05-500m/episode-002/composite_reflectivity/1974-04-03/"
        "arwen_wrf_19740403_18z_f001.png"]
    assert len(set(written)) == 2
    assert {path.read_bytes() for path in written} == {b"PNG-0", b"PNG-1"}
    assert len(list(out.rglob("*.png"))) == 2


def test_a_lifecycle_free_render_is_byte_identical_through_the_seam(
        monkeypatch, tmp_path):
    """The same seam, an input that declares no episode: no new segment.

    The paired half of the test above, and the one that has to hold for
    every run shipped so far.
    """

    from woof import render as render_module

    wrfout = _write_wrfout(tmp_path / "wrfout_d02_1974-04-03_18-00-00.nc",
                           dx=3000.0)
    out = tmp_path / "png"
    name = ("rustwx_wrf_19740403_18z_f000_d02-3km_"
            "composite_reflectivity.png")

    _stub_flat_renderer(monkeypatch, tmp_path, (name,))

    written, _failures, _skipped = render_module.render_wrfouts_rust(
        [wrfout], products="all", timeidx=None, outdir=out,
        size=(800, 600), source_label="WOOF test")
    assert [p.relative_to(out).as_posix() for p in written] == [
        "d02-3km/composite_reflectivity/1974-04-03/"
        "arwen_wrf_19740403_18z_f000.png"]


def test_the_matplotlib_fallback_files_episodes_the_same_way(tmp_path):
    """One layout for both engines, episode segment included."""

    from woof import render as render_module

    episode = tmp_path / "wrfout" / "d02" / "episode-003"
    episode.mkdir(parents=True)
    wrfout = _write_wrfout(episode / "wrfout_d02_1974-04-03_18-00-00.nc")
    out = tmp_path / "png"
    written, failures, skipped = render_module.render_wrfouts(
        [wrfout], products=("t2",), timeidx=0, outdir=out, dpi=72,
        source_label="WOOF test")
    assert failures == [] and skipped == []
    assert [p.relative_to(out).as_posix() for p in written] == [
        "d02-1km/episode-003/t2/1974-04-03/"
        f"t2_d02-1km_{_STAMPS[0].replace(':', '-')}.png"]


def test_pair_matching_reads_through_the_episode_segment(tmp_path):
    """`--pair` keys on domain and product, wherever the episode sits.

    The key is rebuilt from the folders above the frame.  With an
    episode segment in the way, an unchanged reader would key on
    ``episode-002_composite_reflectivity`` and pair nothing -- and two
    runs' second episodes are exactly what a lifecycle comparison wants
    to look at.
    """

    from woof.pair_compose import product_name

    episodic = (tmp_path / "d05-500m" / "episode-002"
                / "composite_reflectivity" / "1974-04-03"
                / "arwen_wrf_19740403_18z_f001.png")
    plain = (tmp_path / "d05-500m" / "composite_reflectivity"
             / "1974-04-03" / "arwen_wrf_19740403_22z_f002.png")
    assert product_name(episodic) == "d05-500m_composite_reflectivity"
    assert product_name(episodic) == product_name(plain)


def test_the_speedrun_capsule_names_products_not_episodes(tmp_path):
    """Every reader of the tree reads the SAME tree.

    The capsule records which products a run drew by naming the folder
    they are filed under.  One segment deeper, a reader counting from
    the left records ``episode-002`` as the product and loses the real
    name -- the layout's "one walker" property broken by arithmetic.
    """

    from woof.speedrun_cli import _rendered_products

    for relative in ("d02-3km/composite_reflectivity/1974-04-03/a.png",
                     "d05-500m/episode-001/2m_temperature/1974-04-03/b.png",
                     "d05-500m/episode-002/2m_temperature/1974-04-03/c.png"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PNG")
    products, files = _rendered_products(tmp_path)
    assert products == ["2m_temperature", "composite_reflectivity"]
    assert len(files) == 3


def test_the_early_render_publishes_an_episodic_layout(tmp_path):
    """first_products carries the whole relative path, episode included.

    Its receipt is the manifest a finalize stage re-checks digests
    against, so a segment the publisher flattened would be a picture the
    re-check looks for where it is not.
    """

    from woof import first_products

    render_dir = tmp_path / "png"
    render_dir.mkdir()
    frame = tmp_path / "wrfout_d05_1974-04-03_18_00_00"
    frame.write_bytes(b"not really a wrfout, only its digest is read")

    relative = ("d05-500m/episode-002/refl/1974-04-03/"
                "refl_d05-500m_1974-04-03_18-00-00.png")

    trigger = first_products.FirstProducts(
        {"render": render_dir, "render_products": "refl"},
        report=lambda _event: None,
        warn=lambda *args, **kwargs: None,
        runner=_early_runner(relative))
    trigger.frame_committed(domain=5, valid_time="1974-04-03T18:00:00",
                            path=frame)
    receipt = trigger.wait(timeout=60.0)

    assert receipt is not None
    assert [entry["name"] for entry in receipt["written"]] == [relative]
    assert (render_dir / relative).read_bytes() == b"PNG-bytes"


def test_the_door_prints_the_episode_segment_when_it_will_write_one(
        tmp_path, capsys):
    """The path a script watches has to be the path that appears.

    The door prints where it is about to write BEFORE it draws.  For an
    episodic input that sentence has one more segment in it, and a
    script told the shorter one watches a directory no picture arrives
    in.
    """

    episode = tmp_path / "wrfout" / "d02" / "episode-002"
    episode.mkdir(parents=True)
    wrfout = _write_wrfout(episode / "wrfout_d02_1974-04-03_18-00-00.nc")
    out = tmp_path / "png"
    rc = cli.main(["render", "--engine", "matplotlib", str(wrfout),
                   "--products", "t2", "--timeidx", "0",
                   "--out", str(out), "--dpi", "72"])
    assert rc == 0
    printed = capsys.readouterr().out
    run_dir = _run_dir(out)
    assert render_layout.describe(str(run_dir), episode=True) in printed
    drawn = sorted(run_dir.rglob("*.png"))
    assert [p.relative_to(run_dir).as_posix() for p in drawn] == [
        "d02-1km/episode-002/t2/1974-04-03/"
        f"t2_d02-1km_{_STAMPS[0].replace(':', '-')}.png"]


def test_the_described_sentence_is_unchanged_for_a_run_with_no_episodes():
    """The default sentence is the one shipped, character for character."""

    assert render_layout.describe("<--out>", sep="/") == (
        "<--out>/<domain>/<product>/<valid-day>/<file>.png "
        "(domain as d02-3km / d05-111m / native_grid, valid-day as "
        "YYYY-MM-DD)")
    described = render_layout.describe(episode=True)
    for segment in ("<domain>", "<episode>", "<product>", "<valid-day>"):
        assert segment in described, described
