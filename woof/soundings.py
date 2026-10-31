"""Public website-style soundings; Python discovers files and drives Rust."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ABI = "gpuwm-soundings.v1 plain-run png json html"


ENV = "WOOF_SOUNDINGS"
#: auto (the CPU rasteriser for a batch of up to 8 pictures per CPU thread, else GPU, else the CPU rasteriser),
#: gpu (every batch on the GPU, refuse without one) or cpu.
RENDERER_ENV = "GPUWM_SOUNDINGS_RENDERER"
#: Window settings a caller sets for every sounding a run draws, as a JSON object (e.g. {"brand": "WOOF Regional",
#: "init": "2026-10-22T00:00Z"}): how a host that launches `woof ensemble` labels the soundings the ensemble cuts.
#: A setting passed to render_run (or --setting) wins over the same key here.
SETTINGS_ENV = "WOOF_SOUNDINGS_SETTINGS"


def env_settings() -> dict:
    text = os.environ.get(SETTINGS_ENV, "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as e:
        raise ValueError(f"{SETTINGS_ENV} is not a JSON object: {e}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{SETTINGS_ENV} must be a JSON object of window settings")
    return {str(k): str(v) for k, v in value.items()}


def find_binary() -> Path | None:
    """The sounding renderer: the override, a checkout build, or the staged copy.

    ``None`` when nothing is staged; ``FileNotFoundError`` when the override
    names a file that is not there, so the gap is reported at its cause
    rather than as a missing build.  Whether the build is current is the
    door's own question, :func:`probe`.
    """
    from woof.bridges import (RUSTWX_CRATE_RELATIVE, accept_resolved,
                               default_bridge_dir, executable_name, packaged_bridge_dir)
    root = Path(__file__).resolve().parent.parent
    filename = executable_name("rw_soundings")
    override = os.environ.get(ENV)
    if override:
        candidate = Path(override)
        if not candidate.is_file():
            raise FileNotFoundError(f"{ENV}={override} names a file that is not there")
        return accept_resolved(candidate.resolve())
    candidates = [root / RUSTWX_CRATE_RELATIVE / "target" / kind / filename
                  for kind in ("release", "debug")]
    candidates += [root / "libexec" / "bridges" / filename,
                   packaged_bridge_dir() / filename, default_bridge_dir() / filename]
    for candidate in candidates:
        if candidate.is_file():
            return accept_resolved(candidate.resolve())
    # On demand since 2.9.3 (not in the platform wheel): staged from this release's bundle on first use.
    from woof.bridges import fetch_on_demand
    if fetch_on_demand("rw_soundings") and (default_bridge_dir() / filename).is_file():
        return accept_resolved((default_bridge_dir() / filename).resolve())
    return None


def probe(path) -> tuple[bool, str]:
    """Does this build answer the ``--abi`` line ``woof soundings`` refuses by?"""
    result = subprocess.run([str(path), "--abi"], capture_output=True, text=True, timeout=30)
    if result.returncode or ABI not in result.stdout:
        return False, f"--abi does not report {ABI!r}; rebuild rw-soundings"
    return True, f"--abi reports {ABI!r}"


def device(path) -> tuple[bool, str]:
    """Which device renders would use; the renderer creates a real one to say.

    ``(True, "gpu ...")`` or ``(True, "cpu rasteriser (<why no GPU, e.g. the
    Vulkan loader libvulkan1 is missing>)")``; ``(False, cause)`` only when no
    device at all can draw (``GPUWM_SOUNDINGS_RENDERER=gpu`` on a host without
    one) or the build predates the probe.
    """
    try:
        result = subprocess.run([str(path), "--probe-device"], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "--probe-device did not answer in 120 s"
    line = next((row for row in result.stdout.splitlines() if row.startswith("render device: ")), None)
    if result.returncode or line is None:
        cause = _cause(result.stderr) or "--probe-device is not supported; rebuild rw-soundings"
        return False, cause
    return True, line.removeprefix("render device: ")


def _cause(stderr: str) -> str:
    """The renderer's own ``soundings: <cause>`` line, else its last line."""
    rows = [row.strip() for row in (stderr or "").splitlines() if row.strip()]
    named = [row.removeprefix("soundings: ") for row in rows if row.startswith("soundings: ")]
    return named[-1] if named else rows[-1] if rows else ""


def binary() -> Path:
    found = find_binary()
    if found is None:
        # Same shape as woof.obs.frontdoor.FrontDoor.require: say why the
        # on-demand staging failed (a 404 for an unpublished bundle, no
        # network, a pin mismatch) and give the remedy true for THIS install
        # (a pip install has no tools/rustwx to cargo-build).
        from woof.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy, executable_name,
                                   on_demand_failure, rustwx_build_hint)
        failure = on_demand_failure("rw_soundings")
        raise FileNotFoundError(
            "the sounding renderer (rw_soundings) is not built or not found"
            + (f"; {failure}" if failure else "") + ".\n"
            + artifact_remedy(env_var=ENV, filename=executable_name("rw_soundings"),
                              subject="the sounding renderer",
                              crate_relative=RUSTWX_CRATE_RELATIVE,
                              one_liner=rustwx_build_hint(), artifact="rw_soundings"))
    ok, evidence = probe(found)
    if not ok:
        raise RuntimeError(f"{found} does not support this sounding request ({evidence})")
    return found


def point(text: str, stations=None) -> dict:
    """LAT,LON, NAME=LAT,LON, or an ID from a user-supplied station catalogue."""
    label, sep, coordinates = text.partition("=")
    if not sep:
        coordinates = text
        if stations and text.upper() in stations:
            coordinates = ",".join(map(str, stations[text.upper()]))
    try:
        lat, lon = map(float, coordinates.split(","))
    except ValueError:
        raise ValueError(f"point {text!r} needs LAT,LON, NAME=LAT,LON, or an ID in --stations JSON") from None
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"invalid latitude/longitude for {text!r}")
    return {"label": label if sep or coordinates != text else f"{lat:g},{lon:g}",
            "lat": lat, "lon": lon}


def inventory(run: Path, *, domain=None) -> list[dict]:
    """Plain single-run, members/<id>/..., rw-store v1 folders, and column stores (``*.wcol``)."""
    run = Path(run).resolve()
    if not run.exists():
        raise FileNotFoundError(run)
    files = [run] if run.is_file() else sorted(run.rglob("wrfout*"))
    files = [p for p in files if p.is_file() and (p.name.startswith("wrfout") and
             p.suffix not in {".json", ".partial", ".tmp", ".lock"} or p.suffix in (".rws", ".wcol"))]
    if not files and run.is_dir():
        files = sorted(run.rglob("*.rws"))
    if not files and run.is_dir():
        # The any-point column store (rw-store v2): a run that keeps no histories, such as a default ensemble,
        # still has every member's columns under columns/members/<member>/<domain>/.
        files = sorted(run.rglob("*.wcol"))
    rows = []
    for path in files:
        relative = path.relative_to(run if run.is_dir() else run.parent)
        parts = relative.parts
        if any(part.startswith(".") for part in parts[:-1]):
            continue
        match = re.search(r"(?:wrfout_)?(d\d{2}(?:-\d+(?:\.\d+)?km)?)", path.name)
        domains = [p for p in parts[:-1] if re.fullmatch(r"d\d{2}(?:-\d+(?:\.\d+)?km)?", p)]
        selected_domain = domains[-1] if domains else match[1] if match else "d01"
        if domain and selected_domain != domain:
            continue
        member = "det"
        if "members" in parts:
            pos = parts.index("members")
            if pos + 1 < len(parts) - 1:
                member = parts[pos + 1]
        else:
            labels = [p for p in parts[:-1] if re.fullmatch(r"(?:mem(?:ber)?|m)[_-]?\d+", p)]
            if labels:
                member = labels[-1]
        # Stored aggregate fields are not ensemble members.
        if any(p in {"mean", "spread", "min", "max", "render", "renders"} for p in parts[:-1]):
            continue
        rows.append({"path": str(path), "domain": selected_domain, "member": member})
    if not rows:
        raise ValueError(f"no wrfout, rw-store or column store sounding files found in {run}")
    return rows


def render_run(run, *, points, out=None, modes=("mean", "all"), domain=None,
               interval_minutes=60, settings=None, threads=4):
    exe = binary()
    resolved = [point(p) if isinstance(p, str) else p for p in points]
    labels = [re.sub(r"[^A-Za-z0-9_-]", "_", p["label"]) for p in resolved]
    if len(set(labels)) != len(labels):
        raise ValueError("point names must produce distinct output filenames")
    root = Path(run)
    output = Path(out) if out is not None else (root if root.is_dir() else root.parent) / "render"
    output.mkdir(parents=True, exist_ok=True)
    listing = member_listing(run)
    modes = tuple(modes)
    if listing and any(mode in ("mean", "all") for mode in modes):
        # The public picker can select every real member with the same native renderer.
        available = {row["member"] for row in inventory(Path(run), domain=domain)}
        modes = tuple(dict.fromkeys((*modes, *(row["member"] for row in listing if row["member"] in available))))
    request = {"sources": inventory(Path(run), domain=domain), "points": resolved,
               "out": str(output.resolve()), "modes": list(modes),
               "interval_minutes": interval_minutes, "settings": {**env_settings(), **(settings or {})},
               "threads": threads}
    # Request files are orchestration scratch, not retained machine-specific public metadata.
    with tempfile.TemporaryDirectory(prefix="gpuwm-soundings-") as temporary:
        path = Path(temporary) / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        # stderr is relayed line by line and its cause line kept, so a failure
        # names what broke instead of a bare exit status.
        process = subprocess.Popen([str(exe), str(path)], stderr=subprocess.PIPE, text=True)
        tail = []
        for row in process.stderr:
            sys.stderr.write(row)
            tail = (tail + [row])[-50:]
        returncode = process.wait()
    if returncode:
        cause = _cause("".join(tail))
        raise RuntimeError(f"sounding renderer exited {returncode}" + (f": {cause}" if cause else ""))
    if listing:
        lineage_viewer(output, listing)
    return output / "soundings-manifest.json"


COLUMNS_ABI = "columns-v1"


def write_columns(run, out, *, domain=None, threads=None, tile=16, every_minutes=0, exact=False, member=None,
                  as_domain=None, timeout=None):
    """The run's any-point column store (rw-store version 2, ``rw-store.columns.v2``): every column of every domain at each history time,
    one file per domain and time under ``out/<domain>/<YYYYMMDDTHHMMSSZ>.wcol``, chunked so one sounding anywhere is
    one small read. ``every_minutes`` keeps only valid times on that cadence (0 keeps every time). Returns the
    renderer's ``COLUMNS <path> bytes=<n> ms=<t>`` rows as dicts. ``exact`` stores every 3-D value's f32 instead of
    the quantized default. ``member`` and ``as_domain`` name the files' member and domain instead of the ones read
    off the path (an ensemble member's frame: ``out/members/<member>/<domain>/``). ``timeout`` (seconds) kills a
    writer that has not finished and raises, so a hung or stalled ``rw_soundings`` never holds its caller (the live
    store passes :func:`woof.live_columns.frame_timeout_seconds`)."""
    exe = binary()
    abi = subprocess.run([str(exe), "--abi"], capture_output=True, text=True, timeout=30).stdout
    if COLUMNS_ABI not in abi:
        raise RuntimeError(f"{exe} cannot write a column store (--abi: {abi.strip()!r}); rebuild rw-soundings")
    sources = inventory(Path(run), domain=domain)
    if every_minutes:
        def keep(row):
            m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})", Path(row["path"]).name)
            return m is None or (int(m[2]) * 60 + int(m[3])) % every_minutes == 0
        sources = [row for row in sources if keep(row)]
    if member is not None or as_domain is not None:
        sources = [{**row, **({"member": str(member)} if member is not None else {}),
                    **({"domain": str(as_domain)} if as_domain is not None else {})} for row in sources]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    request = {"sources": sources, "out": str(out.resolve()), "tile": tile, "exact": bool(exact),
               **({"threads": threads} if threads else {})}
    with tempfile.TemporaryDirectory(prefix="gpuwm-columns-") as temporary:
        path = Path(temporary) / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        try:
            result = subprocess.run([str(exe), "--columns", str(path)], capture_output=True, text=True,
                                    timeout=timeout)
        except subprocess.TimeoutExpired:  # subprocess.run has killed and reaped the writer
            raise RuntimeError(f"column store not written: rw_soundings --columns did not finish in {timeout:g} s; "
                               "it was stopped") from None
    if result.returncode:
        raise RuntimeError("column store not written: " + (_cause(result.stderr) or f"exit {result.returncode}"))
    rows = []
    for line in result.stdout.splitlines():
        if line.startswith("COLUMNS "):
            path_text, *fields = line[len("COLUMNS "):].rsplit(" ", 2)
            row = {"path": path_text}
            row.update({k: float(v) for k, v in (f.split("=", 1) for f in fields)})
            rows.append(row)
    return rows


def register_cli(sub):
    parser = sub.add_parser("soundings", help="Draw the website-style sounding window from a run folder")
    parser.add_argument("run", type=Path, help="single run, ensemble member folder, wrfout, or rw-store folder")
    parser.add_argument("--point", "--station", action="append",
                        help="LAT,LON, NAME=LAT,LON or an ID in --stations; repeat for multiple points")
    parser.add_argument("--stations", type=Path, help='JSON station catalogue: {"ID": [lat, lon]}')
    parser.add_argument("--out", type=Path, help="render root (default RUN/render)")
    parser.add_argument("--domain", help="domain identifier to select")
    parser.add_argument("--mode", action="append", help="mean, all (members + P10-P90), or member directory ID; repeat (default mean and all)")
    parser.add_argument("--interval-minutes", type=int, default=60, help="profile cadence from first available valid time (default hourly)")
    parser.add_argument("--threads", type=int, default=None, help="CPU workers (default 4 for pictures, every core for --columns)")
    parser.add_argument("--setting", action="append", default=[], metavar="KEY=VALUE", help="website window setting, e.g. theme=light, parcel=ml, brand=..., init=2026-10-22T00:00Z (the lead's start)")
    parser.add_argument("--columns", type=Path, metavar="DIR",
                        help="write the run's any-point column store into DIR instead of pictures (no --point needed)")
    parser.add_argument("--every-minutes", type=int, default=0, help="with --columns: only valid times on this cadence")
    parser.set_defaults(func=cli_main)


def cli_main(args):
    if args.columns is not None:
        rows = write_columns(args.run, args.columns, domain=args.domain, threads=args.threads,
                             every_minutes=args.every_minutes)
        total = sum(r.get("bytes", 0) for r in rows)
        print(f"Column store: {len(rows)} files, {total / 1e6:.1f} MB in {args.columns}")
        return 0
    if not args.point:
        raise ValueError("--point is required (or --columns DIR to write the column store)")
    stations = {} if args.stations is None else {k.upper(): v for k, v in json.loads(args.stations.read_text()).items()}
    if args.interval_minutes <= 0 or (args.threads is not None and args.threads <= 0):
        raise ValueError("interval-minutes and threads must be positive")
    settings = {}
    for value in args.setting:
        key, sep, val = value.partition("=")
        if not sep:
            raise ValueError("--setting takes KEY=VALUE")
        settings[key] = val
    manifest = render_run(args.run, points=[point(p, stations) for p in args.point],
                          out=args.out, modes=args.mode or ("mean", "all"), domain=args.domain,
                          interval_minutes=args.interval_minutes, settings=settings, threads=args.threads or 4)
    print(f"Soundings: {manifest}")
    return 0


def member_listing(run):
    """List durable member IDs and lineage without opening weather arrays."""
    from woof.ensemble.lineage import member_display
    root = Path(run).resolve()
    if root.is_file():
        root = root.parent
    manifest = next((parent / "ensemble-run.json" for parent in (root, *root.parents)
                     if (parent / "ensemble-run.json").is_file()), None)
    if manifest is None:
        return []
    document = json.loads(manifest.read_text(encoding="utf-8"))
    records = document.get("lineage", {}).get("members") or document.get("request", {}).get("provenance", [])
    return [{"member": f"member-{row['member']:04d}",
             "label": member_display(row, f"member-{row['member']:04d}"),
             "family": row.get("parent_family", ""), "model_key": row.get("model_key", ""),
             "provenance": row} for row in records]


def lineage_viewer(output, listing):
    """Attach the member picker and lineage legend to the public sounding window."""
    import html
    output = Path(output)
    manifest_path = output / "soundings-manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {row["member"]: row for row in listing}
    pages = {}
    for picture in document.get("pictures", []):
        png = Path(picture["png"])
        if not png.is_absolute():
            png = output / png
        profile = png.with_suffix(".json")
        data = json.loads(profile.read_text(encoding="utf-8"))
        mode = data.get("mode", "mean")
        row = by_id.get(mode)
        label = row["label"] if row else mode
        data["member_listing"] = listing
        if row:
            data["provenance"] = row["provenance"]
        for member in data.get("members", []):
            if member["member"] in by_id:
                member.update(by_id[member["member"]])
        profile.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        picture.update(mode=mode, label=label)
        pages.setdefault(png.parent, []).append((png, data, label))
    for folder, pictures in pages.items():
        options = "".join(f'<option value="{html.escape(mode, quote=True)}">{html.escape(by_id.get(mode, {}).get("label", mode))}</option>'
                          for mode in dict.fromkeys(data.get("mode", "mean") for _, data, _ in pictures))
        figures = "".join(f'<figure data-member="{html.escape(data.get("mode", "mean"), quote=True)}"><figcaption>{html.escape(data["valid"])} {html.escape(label)} '
                          f'<a href="{html.escape(png.with_suffix(".json").name, quote=True)}">profile JSON</a></figcaption>'
                          f'<img loading="lazy" src="{html.escape(png.name, quote=True)}" alt="{html.escape(label, quote=True)}"></figure>'
                          for png, data, label in pictures)
        legend = "".join(f'<li>{html.escape(row["label"])}</li>' for row in listing)
        page = ('<!doctype html><meta charset="utf-8"><title>Ensemble soundings</title>'
                '<style>body{background:#101018;color:#eee;font:16px sans-serif}img{max-width:100%}a{color:#8cf}</style>'
                '<h1>Ensemble soundings</h1><label>Member <select id="member-picker">' + options + '</select></label>'
                '<details open><summary>Member legend</summary><ul>' + legend + '</ul></details>' + figures +
                '<script>const picker=document.getElementById("member-picker");function selectMember(){'
                'for(const figure of document.querySelectorAll("figure[data-member]"))figure.hidden=figure.dataset.member!==picker.value;}'
                'picker.addEventListener("change",selectMember);selectMember();</script>')
        (folder / "index.html").write_text(page, encoding="utf-8")
    document["member_listing"] = listing
    manifest_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
