"""woof.hex.prep_store: a hit is the builder's bytes, a miss builds exactly as before.

The breakage these guard: a store that published bytes other than the ones the
named engine writes would feed the dycore a static it was never proven on, and
a key that missed a dependency would hand one mesh's static to another.
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import sys
import threading

import pytest

from woof.hex import prep_store as ps

# A stand-in engine: writes deterministic bytes derived from its own argv
# (minus --out/--receipt/--clobber), so two runs of the same request agree and
# any request difference shows up in the output.
FAKE = r'''
import hashlib, json, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
keep = [x for i, x in enumerate(a) if x not in ("--out", "--receipt", "--clobber")
        and (i == 0 or a[i - 1] not in ("--out", "--receipt"))]
for i, x in enumerate(keep):
    if x in ("--spec", "--grid"):
        keep[i + 1] = hashlib.sha256(open(keep[i + 1], "rb").read()).hexdigest()
body = hashlib.sha256(json.dumps(keep).encode()).digest() * 4096
open(out, "wb").write(body)
if "--receipt" in a:
    open(a[a.index("--receipt") + 1], "w").write(json.dumps({"argv": keep}))
open(out + ".runs", "a").write("1")
'''


def _engine(tmp: Path, name: str) -> list[str]:
    script = tmp / f"{name}.py"
    script.write_text(FAKE, encoding="utf-8")
    exe = tmp / name
    exe.write_bytes(script.read_bytes())
    return [str(exe)]


def _runner(argv):
    import subprocess
    return subprocess.run([sys.executable, *argv], check=True)


def _geog(root: Path) -> Path:
    for _, (name, *_older) in ps.GEOG_DATASETS:
        d = root / ("soilgrids/" + name if name in ps.GEOG_CONTAINERS else name)
        d.mkdir(parents=True)
        (d / "index").write_text(f"type=categorical\n# {name}\n")
        (d / "00001-00010.00001-00010").write_bytes(b"\0" * 100)
    return root


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_DIGEST_CACHE", "off")
    store, box = tmp_path / "store", tmp_path / "outbox"
    monkeypatch.setenv(ps.ENV_STORE, str(store))
    monkeypatch.setenv(ps.ENV_OUTBOX, str(box))
    return tmp_path, store, box


def _mesh_argv(tmp, out, spec_text='{"background_km": 20.0}'):
    spec = tmp / "spec.json"
    spec.write_text(spec_text)
    return [*_engine(tmp, "rw_mpas_mesh"), "--spec", str(spec), "--out", str(out),
            "--receipt", str(out) + ".receipt.json"]


def test_no_store_configured_builds_exactly_as_before(tmp_path, monkeypatch):
    monkeypatch.delenv(ps.ENV_STORE, raising=False)
    monkeypatch.delenv(ps.ENV_OUTBOX, raising=False)
    out = tmp_path / "g.nc"
    result = ps.run_or_fetch(_mesh_argv(tmp_path, out), runner=_runner)
    assert result["source"] == "built" and out.is_file()
    assert not (tmp_path / "outbox").exists()


def test_build_stages_then_fetch_is_byte_identical(env):
    tmp, store, box = env
    first = tmp / "a" / "g.nc"
    first.parent.mkdir()
    built = ps.run_or_fetch(_mesh_argv(tmp, first), runner=_runner)
    assert built["source"] == "built"
    entry = json.loads((box / "mesh" / built["key"] / "entry.json").read_text())
    assert entry["files"]["grid.nc"]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()

    # publish = the outbox becomes the store
    os.replace(box, store)
    second = tmp / "b" / "g.nc"
    fetched = ps.run_or_fetch(_mesh_argv(tmp, second), runner=_runner)
    assert fetched["source"] == "store" and fetched["key"] == built["key"]
    assert second.read_bytes() == first.read_bytes()
    assert Path(str(second) + ".receipt.json").read_bytes() == Path(str(first) + ".receipt.json").read_bytes()
    assert not Path(str(second) + ".runs").exists(), "a hit must not run the engine"


def test_background_km_and_its_spec_share_a_key(env):
    tmp, _, _ = env
    reg = {}
    exe = _engine(tmp, "rw_mpas_mesh")
    by_flag = ps.mesh_request([*exe, "--background-km", "20", "--out", "x.nc"], reg)
    spec = tmp / "s.json"
    spec.write_text('{"background_km": 20.0, "name": null, "regions": []}')
    by_spec = ps.mesh_request([*exe, "--spec", str(spec), "--out", "y.nc"], reg)
    assert by_flag[0] == by_spec[0]


@pytest.mark.parametrize("extra", [["--cells", "1000"], ["--triangulation", "incremental"], ["--sweeps", "5"]])
def test_output_changing_flags_change_the_key(env, extra):
    tmp, _, _ = env
    base = _mesh_argv(tmp, tmp / "g.nc")
    assert ps.mesh_request(base, {})[0] != ps.mesh_request(base + extra, {})[0]


def test_default_valued_flags_do_not_change_the_key(env):
    tmp, _, _ = env
    base = _mesh_argv(tmp, tmp / "g.nc")
    assert ps.mesh_request(base, {})[0] == ps.mesh_request(base + ["--triangulation", "rebuild", "--sweeps", "300"], {})[0]


def test_unknown_flag_bypasses_the_store(env):
    tmp, _, box = env
    assert ps.mesh_request(_mesh_argv(tmp, tmp / "g.nc") + ["--dry-run"], {}) is None


def test_engine_identity_class_comes_only_from_attestation(env):
    tmp, _, _ = env
    exe = Path(_engine(tmp, "rw_mpas_mesh")[0])
    sha = hashlib.sha256(exe.read_bytes()).hexdigest()
    assert ps.engine_identity(exe, {})["class"] == f"sha256:{sha}"
    assert ps.engine_identity(exe, {sha: {"class": "rw-mpas@abc"}})["class"] == "rw-mpas@abc"


def test_static_key_depends_on_grid_geog_and_options(env):
    tmp, _, _ = env
    geog = _geog(tmp / "geog")
    grid = tmp / "grid.nc"
    grid.write_bytes(b"grid-1")
    exe = _engine(tmp, "rw_mpas_static")
    argv = [*exe, "--grid", str(grid), "--geog", str(geog), "--out", str(tmp / "s.nc")]
    k0 = ps.static_request(argv, {})[0]
    assert ps.static_request(argv + ["--terrain-only"], {})[0] != k0
    assert ps.static_request(argv + ["--nominal-dx-m", "20000"], {})[0] != k0
    (geog / "topo_gmted2010_30s" / "00011-00020.00001-00010").write_bytes(b"\0" * 7)
    k1 = ps.static_request(argv, {})[0]
    assert k1 != k0
    grid.write_bytes(b"grid-2")
    assert ps.static_request(argv, {})[0] != k1


def test_geog_layout_flat_or_container_is_not_keyed(tmp_path):
    nested = _geog(tmp_path / "a")
    flat = tmp_path / "b"
    for _, (name, *_older) in ps.GEOG_DATASETS:
        src = nested / ("soilgrids/" + name if name in ps.GEOG_CONTAINERS else name)
        dst = flat / name
        dst.mkdir(parents=True)
        for f in src.iterdir():
            (dst / f.name).write_bytes(f.read_bytes())
    assert ps.geog_fingerprint(nested)["sha256"] == ps.geog_fingerprint(flat)["sha256"]
    assert ps.geog_fingerprint(nested)["dirs"]["soilcomp"] == "soilgrids/soilcomp"


def test_incomplete_geog_builds_and_absent_geog_uses_the_declared_pin(env):
    tmp, store, _ = env
    geog = _geog(tmp / "geog")
    grid = tmp / "grid.nc"
    grid.write_bytes(b"g")
    exe = _engine(tmp, "rw_mpas_static")
    argv = lambda root: [*exe, "--grid", str(grid), "--geog", str(root), "--out", str(tmp / "s.nc")]  # noqa: E731
    full = ps.static_request(argv(geog), {})
    pin = ps.geog_fingerprint(geog)["sha256"]
    assert ps.static_request(argv(tmp / "nowhere"), {}, pinned=pin)[0] == full[0]
    assert ps.static_request(argv(tmp / "nowhere"), {}, pinned=None) is None  # built: the binary refuses by name
    import shutil
    shutil.rmtree(geog / "albedo_modis")
    assert ps.static_request(argv(geog), {}, pinned=pin) is None


def test_corrupt_store_bytes_are_refused_and_the_door_builds(env, capsys):
    tmp, store, box = env
    out = tmp / "a.nc"
    built = ps.run_or_fetch(_mesh_argv(tmp, out), runner=_runner)
    os.replace(box, store)
    (store / "mesh" / built["key"] / "grid.nc").write_bytes(b"not the grid")
    dest = tmp / "c" / "g.nc"
    calls = []
    got = ps.run_or_fetch(_mesh_argv(tmp, dest), runner=lambda a: calls.append(a) or _runner(a))
    assert got["source"] == "built" and calls
    assert dest.read_bytes() != b"not the grid"
    assert not list(dest.parent.glob(".*partial"))
    assert "BAD STORE ENTRY REFUSED" in capsys.readouterr().err


def test_store_missing_an_entry_file_builds(env):
    tmp, store, box = env
    out = tmp / "a.nc"
    built = ps.run_or_fetch(_mesh_argv(tmp, out), runner=_runner)
    os.replace(box, store)
    (store / "mesh" / built["key"] / "grid.nc").unlink()
    dest = tmp / "d" / "g.nc"
    assert ps.run_or_fetch(_mesh_argv(tmp, dest), runner=_runner)["source"] == "built"
    assert dest.exists() and not list(dest.parent.glob(".*partial"))


def test_existing_output_without_clobber_runs_the_engine(env):
    """The engine's own refusal stands; the store never overwrites a file it was not asked to."""
    tmp, store, box = env
    out = tmp / "g.nc"
    ps.run_or_fetch(_mesh_argv(tmp, out), runner=_runner)
    os.replace(box, store)
    out.write_bytes(b"user's own file")
    calls = []
    ps.run_or_fetch(_mesh_argv(tmp, out), runner=lambda a: calls.append(a))
    assert calls and out.read_bytes() == b"user's own file"


def test_failed_build_is_not_staged(env):
    tmp, _, box = env
    import subprocess
    argv = _mesh_argv(tmp, tmp / "g.nc")
    def half(a):
        Path(a[a.index("--out") + 1]).write_bytes(b"half")
        Path(a[a.index("--receipt") + 1]).write_text("{}")
        return subprocess.CompletedProcess(a, 3)

    result = ps.run_or_fetch(argv, runner=half)
    assert result["source"] == "built"
    assert not (box / "mesh").exists()


class _Ranged(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802 - http.server API
        path = Path(self.translate_path(self.path))
        data = path.read_bytes()
        first, last = self.headers["Range"].removeprefix("bytes=").split("-")
        part = data[int(first):int(last) + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)


def test_ticket_store_fetches_by_ranges_and_verifies(env, monkeypatch):
    tmp, store, box = env
    out = tmp / "a.nc"
    built = ps.run_or_fetch(_mesh_argv(tmp, out), runner=_runner)
    served = tmp / "served"
    os.replace(box, served)
    monkeypatch.setattr(ps, "_CHUNK", 1000)  # many ranges for a 128 kB file
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Ranged, directory=str(served)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        rel = f"mesh/{built['key']}"
        entry = json.loads((served / rel / "entry.json").read_text())
        ticket = tmp / "ticket.json"
        ticket.write_text(json.dumps({"schema": ps.TICKET_SCHEMA, "expires_epoch": 4e9, "entries": {
            rel: {"entry": entry, "urls": {n: f"{base}/{rel}/{n}" for n in entry["files"]}}}}))
        monkeypatch.setenv(ps.ENV_STORE, str(ticket))
        dest = tmp / "t" / "g.nc"
        got = ps.run_or_fetch(_mesh_argv(tmp, dest), runner=_runner)
        assert got["source"] == "store"
        assert dest.read_bytes() == out.read_bytes()
    finally:
        server.shutdown()


FAKE_ORDER = """#!{python}
import sys
a = sys.argv[1:]
if a[0] == "--make-map":
    open(a[2], "wb").write(b"map:" + open(a[1], "rb").read())
elif a[0] == "--apply":
    open(a[3], "wb").write(open(a[1], "rb").read() + b"|" + open(a[2], "rb").read())
elif a[0] == "--record-donor":
    open(a[2], "ab").write(b"|donor")
"""


@pytest.mark.skipif(os.name == "nt", reason="the stand-in hex_order is a script run directly")
def test_order_builds_stages_and_fetches_identically(env):
    tmp, store, box = env
    exe = tmp / "hex_order"
    exe.write_text(FAKE_ORDER.format(python=sys.executable))
    exe.chmod(0o755)
    grid, static = tmp / "g.nc", tmp / "s.nc"
    grid.write_bytes(b"grid")
    static.write_bytes(b"static")
    built = ps.order(grid, static, tmp / "a", exe)
    assert built["source"] == "built"
    os.replace(box, store)
    got = ps.order(grid, static, tmp / "b", exe)
    assert got["source"] == "store" and got["key"] == built["key"]
    for name in ps.ORDER_FILES:
        assert (tmp / "b" / name).read_bytes() == (tmp / "a" / name).read_bytes()
    static.write_bytes(b"other static")
    assert ps.order(grid, static, tmp / "c", exe)["source"] == "built"


def test_expired_ticket_is_a_miss(env, monkeypatch):
    tmp, _, _ = env
    ticket = tmp / "ticket.json"
    ticket.write_text(json.dumps({"schema": ps.TICKET_SCHEMA, "expires_epoch": 1, "entries": {}}))
    monkeypatch.setenv(ps.ENV_STORE, str(ticket))
    monkeypatch.delenv(ps.ENV_OUTBOX)
    result = ps.run_or_fetch(_mesh_argv(tmp, tmp / "g.nc"), runner=_runner)
    assert result["source"] == "built"


def test_the_engines_current_dataset_names_are_keyed(tmp_path):
    """woof 2.9.5's rw_mpas_static reads texture_top (in soilgrids), soiltemp_era5_1deg and the lake-free land use.
    A root laid out that way (the WOOF run-box image) keys; before, two slots read as missing and no static was
    ever stored (318 s rebuilt on every run of the 30 km region, 2026-10-23)."""
    geog = _geog(tmp_path / "geog")
    fp = ps.geog_fingerprint(geog)
    assert not any(row.get("missing") for row in fp["slots"].values())
    assert fp["dirs"]["soilcat"] == "soilgrids/texture_top"
    assert fp["dirs"]["soil_temperature"] == "soiltemp_era5_1deg"
    # an older layout (soiltype_top_30s, soiltemp_1deg) still keys, and differently
    old = tmp_path / "old"
    for slot, names in ps.GEOG_DATASETS:
        name = names[-1]
        d = old / ("soilgrids/" + name if name in ps.GEOG_CONTAINERS else name)
        d.mkdir(parents=True)
        (d / "index").write_text(f"type=categorical\n# {name}\n")
        (d / "00001-00010.00001-00010").write_bytes(b"\0" * 100)
    fp_old = ps.geog_fingerprint(old)
    assert not any(row.get("missing") for row in fp_old["slots"].values())
    assert fp_old["sha256"] != fp["sha256"]
    # a root holding both land-use directories keys both
    (geog / "modis_landuse_20class_30s_with_lakes").mkdir()
    (geog / "modis_landuse_20class_30s_with_lakes" / "index").write_text("x")
    assert ps.geog_fingerprint(geog)["sha256"] != fp["sha256"]


def test_the_native_static_flags_are_keyed(tmp_path):
    """The global and variable-resolution doors ask rw_mpas_static for --native-init --native-gwd-static; before,
    those flags made the request unparseable, so their statics were never stored."""
    geog = _geog(tmp_path / "geog")
    grid = tmp_path / "grid.nc"
    grid.write_bytes(b"g")
    exe = _engine(tmp_path, "rw_mpas_static")
    base = [*exe, "--grid", str(grid), "--geog", str(geog), "--out", str(tmp_path / "s.nc")]
    plain = ps.static_request(base, {})
    native = ps.static_request([*base, "--native-init", "--native-gwd-static", "--nominal-dx-m", "15000"], {})
    assert plain is not None and native is not None
    assert native[0] != plain[0]
    assert native[1]["options"]["native_init"] is True and native[1]["options"]["native_gwd_static"] is True
    assert "native_init" not in plain[1]["options"]  # keys built without the flags are unchanged
