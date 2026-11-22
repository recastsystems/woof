"""``render-georef.json`` lists every delivered picture, through ``woof.render``.

The breakage this pins: ``rw_wrfbatch`` keys its record by the flat name
it drew each picture under and drops an entry whose file has left the
folder, and ``woof.render`` moves and renames every launch's pictures
into ``<domain>/<product>/<valid-day>/`` right after that launch.  So on
the next launch every earlier batch fell out of the record: three d02
frames rendered through ``render_wrfouts_rust`` delivered 6 pictures and
listed 2.  The engine's own tests could not see it, because they launch
the binary into a folder where nothing ever moves.

The engine is stubbed at ``subprocess.run`` the way the other rust-engine
unit tests stub it, and the stub keeps the engine's record contract: it
writes each picture flat, then merges its batch into the folder's record
under the record's lock, keeping only entries whose file is still there.
Everything after the engine exits is the real render path.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "wrf", reason="woof render requires the wrf package (wrf-rust)")

from woof import render_georef
from woof.io.wrfout import WrfoutWriter

_NZ, _NY, _NX = 4, 12, 16
_PRODUCTS = ("composite_reflectivity", "2m_temperature")
_TOKENS = {"d01": "d01-12km", "d02": "d02-3km"}
_DX = {"d01": 12000.0, "d02": 3000.0}


def _wrfout(root: Path, domain: str, hour: int) -> Path:
    path = root / f"wrfout_{domain}_1974-04-03_{hour:02d}_00_00"
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    frame = {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "T2": np.full((_NY, _NX), 290.0, np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=_DX[domain],
                      dy=_DX[domain]) as writer:
        writer.write_frame(f"1974-04-03_{hour:02d}:00:00", frame)
    return path


class _Engine:
    """``rw_wrfbatch`` as far as the record goes.

    ``meet``, when set, holds every launch after it has written its
    pictures and merged its record until that many launches have, so
    launches that finish together are exercised deterministically.
    """

    def __init__(self) -> None:
        self.passes = 0
        self.meet: threading.Barrier | None = None
        self.lock = threading.Lock()

    def __call__(self, command, **kwargs):
        out_dir = Path(command[command.index("--out-dir") + 1])
        wrfouts = [part for part in map(str, command)
                   if re.search(r"wrfout_d0\d_", part)]
        domain, hour = re.search(r"wrfout_(d0\d)_1974-04-03_(\d\d)",
                                 wrfouts[-1]).groups()
        with self.lock:
            self.passes += 1
            stamp = self.passes
        out_dir.mkdir(parents=True, exist_ok=True)
        batch = {"schema": render_georef.GEOREF_SCHEMA,
                 "generated_utc": f"pass {stamp}", "panels": {},
                 "without_georeference": []}
        lines = []
        for product in _PRODUCTS:
            name = (f"rustwx_wrf_19740403_18z_f{int(hour) - 18:03d}_"
                    f"{_TOKENS[domain]}_{product}.png")
            (out_dir / name).write_bytes(b"PNG")
            lines.append(f"RENDERED {product} {out_dir / name}")
            batch["panels"][name] = {
                "schema": "rustwx.panel-georeference/v1", "pass": stamp}
        lock = render_georef._acquire(out_dir)
        try:
            held = render_georef.read(out_dir / render_georef.GEOREF_FILENAME)
            render_georef._replace(
                out_dir / render_georef.GEOREF_FILENAME,
                render_georef.merge(held, batch, root=out_dir))
        finally:
            render_georef._release(lock)
        if self.meet is not None:
            self.meet.wait(timeout=30)

        class Result:
            returncode = 0
            stderr = ""
            stdout = "\n".join(lines) + "\n"

        return Result()


def _record(out: Path) -> dict:
    return json.loads((out / render_georef.GEOREF_FILENAME)
                      .read_text(encoding="utf-8"))


def _delivered(out: Path) -> set[str]:
    return {path.relative_to(out).as_posix() for path in out.rglob("*.png")}


def test_every_launch_of_every_grid_stays_in_the_record_where_it_was_filed(
        monkeypatch, tmp_path):
    import subprocess

    from woof import render as render_module

    engine = _Engine()
    monkeypatch.setattr(subprocess, "run", engine)
    monkeypatch.setattr(render_module, "renderer_refusal", lambda _r: None)
    monkeypatch.setattr("woof.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch")
    inputs = tmp_path / "in"
    inputs.mkdir()
    out = tmp_path / "png"

    def launch(domain: str, hour: int) -> list[Path]:
        return launch_file(_wrfout(inputs, domain, hour))

    def launch_file(wrfout: Path) -> list[Path]:
        written, failures, skipped = render_module.render_wrfouts_rust(
            [wrfout], products="all", timeidx=None,
            outdir=out, size=(800, 600), source_label="WOOF-ArWen test")
        assert failures == [] and skipped == []
        return written

    # One launch per history file, one grid after the other, the way a
    # run draws each frame as it lands.
    for domain, hour in (("d02", 18), ("d02", 19), ("d01", 18), ("d01", 19)):
        written = launch(domain, hour)
        assert all(path.parent != out for path in written), written
        record = _record(out)
        assert set(record["panels"]) == _delivered(out)
    assert len(_delivered(out)) == 8

    # Two launches, one per grid, whose engines both finish before
    # either files its pictures.  The history files are written before
    # the threads start: netCDF4 over HDF5 is not thread safe, and two
    # threads writing files at once crashed the interpreter about one run
    # in five, before either launch reached the render path.
    engine.meet = threading.Barrier(2)
    errors: list[BaseException] = []
    racing = {domain: _wrfout(inputs, domain, 20) for domain in ("d02", "d01")}

    def racer(domain: str) -> None:
        try:
            launch_file(racing[domain])
        except BaseException as error:  # surfaced below
            errors.append(error)

    threads = [threading.Thread(target=racer, args=(domain,))
               for domain in ("d02", "d01")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    engine.meet = None
    record = _record(out)
    assert len(_delivered(out)) == 12
    assert set(record["panels"]) == _delivered(out)

    # The end-of-run pass draws every picture again: each entry is
    # replaced once, none is duplicated, none is lost.
    before = engine.passes
    written, failures, _ = render_module.render_wrfouts_rust(
        sorted(inputs.glob("wrfout_*")), products="all", timeidx=None,
        outdir=out, size=(800, 600), source_label="WOOF-ArWen test")
    assert failures == [] and len(written) == 12
    record = _record(out)
    assert set(record["panels"]) == _delivered(out)
    assert all(panel["pass"] > before for panel in record["panels"].values())
    assert record["without_georeference"] == []
    assert not (out / "render-georef.json.lock").exists()
    assert list(out.glob("render-georef.json.tmp-*")) == []
    # Keys name the delivered layout, each under its own grid's folder.
    assert {key.split("/")[0] for key in record["panels"]} == {
        "d01-12km", "d02-3km"}


def test_an_entry_of_another_launch_is_not_moved_and_a_gone_picture_goes(
        tmp_path):
    """The re-key touches only the pictures this launch moved."""

    out = tmp_path / "png"
    out.mkdir()
    (out / "mine.png").write_bytes(b"PNG")
    (out / "theirs.png").write_bytes(b"PNG")
    render_georef._replace(out / render_georef.GEOREF_FILENAME, {
        "schema": render_georef.GEOREF_SCHEMA, "generated_utc": "x",
        "panels": {"mine.png": {"n": 1}, "theirs.png": {"n": 2},
                   "gone.png": {"n": 3}},
        "without_georeference": []})

    def place(png: Path) -> Path:
        target = out / "d01-12km" / "t2" / "day" / "arwen_mine.png"
        target.parent.mkdir(parents=True)
        png.replace(target)
        return target

    filed = render_georef.file_pictures(out, [out / "mine.png"], place)
    assert filed == [out / "d01-12km/t2/day/arwen_mine.png"]
    assert _record(out)["panels"] == {
        "d01-12km/t2/day/arwen_mine.png": {"n": 1}, "theirs.png": {"n": 2}}
