#!/usr/bin/env python3
"""Hurricane loops from our own Rust renderer.

Every weather-field panel is drawn by rw_wrfbatch (tc theme: fixed 10 m wind
ladder to 180 kt, isobars every 4 hPa from 880 hPa) on a storm-centred crop
(--bounds) with the best track and the model track overlaid (--overlays).
hex histories go through rw_mpas_convert first.  This script only orchestrates:
it builds the per-frame overlay JSON, draws the intensity strip (an analysis
chart, not a weather field), stacks the panels and encodes the loops.

  tcg.py frames JOB.json      render every frame of one job (parallel)
  tcg.py loop   JOB.json      compose strip + panels, write GIF and frames for MP4
"""
import concurrent.futures as cf
import datetime as dt
import glob
import json
import math
import os
import shutil
import subprocess
import sys

UTC = dt.timezone.utc


def T(s):
    s = s.replace("Z", "").replace("_", "T")
    if len(s) == 13:
        s += ":00"
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


# ----------------------------------------------------------------- truth tracks
def bdeck(path):
    rows = {}
    for line in open(path):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 10 or f[4] != "BEST":
            continue
        t = dt.datetime.strptime(f[2], "%Y%m%d%H").replace(tzinfo=UTC)
        lat = int(f[6][:-1]) / 10 * (1 if f[6][-1] == "N" else -1)
        lon = int(f[7][:-1]) / 10 * (-1 if f[7][-1] == "W" else 1)
        rows[t] = dict(t=t, lat=lat, lon=lon, vmax=float(f[8]), mslp=float(f[9]))
    return [rows[k] for k in sorted(rows)]


def interp(track, t, key):
    pts = [p for p in track if p.get(key) is not None]
    if not pts or t < pts[0]["t"] or t > pts[-1]["t"]:
        return None
    for a, b in zip(pts, pts[1:]):
        if a["t"] <= t <= b["t"]:
            w = (t - a["t"]).total_seconds() / max((b["t"] - a["t"]).total_seconds(), 1)
            return a[key] + w * (b[key] - a[key])
    return pts[-1][key]


def recon(path):
    out = []
    if not path or not os.path.exists(path):
        return out
    import csv
    for r in csv.DictReader(open(path)):
        try:
            out.append(dict(t=T(r["time"]), mslp=float(r["min_slp_hpa"]), lat=float(r["lat"]), lon=float(r["lon"])))
        except (ValueError, KeyError):
            pass
    return out


# ------------------------------------------------------------------ model fixes
def load_fixes(spec):
    kind, path = spec["kind"], spec["path"]
    out = []
    if kind == "arwen":  # fixes-d0N.json: valid, lat, lon, mslp_hpa, vmax_kt
        for f in json.load(open(path)):
            out.append(dict(t=T(f["valid"]), lat=f["lat"], lon=f["lon"], mslp=f["mslp_hpa"], vmax=f["vmax_kt"]))
    elif kind == "hex-fields":  # field-derived hex fixes
        for f in json.load(open(path)):
            out.append(dict(t=T(f["valid"]), lat=f["lat"], lon=f["lon"], mslp=f["mslp_hpa"], vmax=f["vmax10_kt"]))
    elif kind == "controller":  # hex moving-run controller.json
        c = json.load(open(path))
        t0 = T(spec["t0"])
        for fr in c["frames"]:
            fx = fr.get("fix") or {}
            if fx.get("ok"):
                out.append(dict(t=t0 + dt.timedelta(seconds=float(fr["model_seconds"])), lat=fx["lat"], lon=fx["lon"],
                                mslp=fx["mslp_min_hpa"], vmax=fx["vmax10_ms"] / 0.514444))
    out.sort(key=lambda p: p["t"])
    return out


def nearest(track, t, tol_min=40):
    best = min(track, key=lambda p: abs((p["t"] - t).total_seconds()), default=None)
    if best is None or abs((best["t"] - t).total_seconds()) > tol_min * 60:
        return None
    return best


# ------------------------------------------------------------------- overlays
def overlay_json(path, bt, model, t, centre_fix):
    lines, points = [], []
    bt_in = [p for p in bt if p["t"] <= t + dt.timedelta(hours=96)]
    if len(bt_in) > 1:
        lines.append(dict(points=[[p["lat"], p["lon"]] for p in bt_in], color="#111111", width=2))
        for p in bt_in:
            if p["t"].hour % 6 == 0 and p["t"].minute == 0:
                points.append(dict(lat=p["lat"], lon=p["lon"], color="#111111", radius_px=3, width_px=2, shape="circle"))
    past = [p for p in model if p["t"] <= t + dt.timedelta(minutes=5)]
    if len(past) > 1:
        lines.append(dict(points=[[p["lat"], p["lon"]] for p in past], color="#d0103a", width=3))
    blat, blon = interp(bt, t, "lat"), interp(bt, t, "lon")
    if blat is not None:
        points.append(dict(lat=blat, lon=blon, color="#111111", radius_px=9, width_px=3, shape="circle"))
    json.dump(dict(lines=lines, points=points), open(path, "w"))


# -------------------------------------------------------------------- renderer
RENDER_ENV = dict(os.environ, RUSTWX_BARB_DENSITY="0.55", RUSTWX_BARB_LENGTH_PX="18")


def run(cmd, log):
    with open(log, "a") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=RENDER_ENV)
    return r.returncode


def rendered(log):
    out = {}
    for line in open(log):
        if line.startswith("RENDERED "):
            _, prod, path = line.strip().split(" ", 2)
            out[prod] = path
    return out


def bounds_for(centre, half):
    la, lo = centre
    hx, hy = half
    return [lo - hx, lo + hx, la - hy, la + hy]


def clamp_inside(b, ext):
    """Shift box b=[W,E,S,N] so it stays inside ext=[W,E,S,N] (same size)."""
    W, E, S, N = b
    if ext:
        if W < ext[0]:
            E += ext[0] - W; W = ext[0]
        if E > ext[1]:
            W -= E - ext[1]; E = ext[1]
        if S < ext[2]:
            N += ext[2] - S; S = ext[2]
        if N > ext[3]:
            S -= N - ext[3]; N = ext[3]
    return [W, E, S, N]


def wrf_extent(path):
    from woof.tc_tracks import binary
    return json.loads(subprocess.run([binary("rw_tc_fields"), "extent", path],
                                    check=True, capture_output=True, text=True).stdout)


def frame_job(job, fr, view):
    """Render one frame of one view: returns {product: png}."""
    tools = job["tools"]
    t = T(fr["valid"])
    od = os.path.join(job["work"], view["name"], fr["tag"])
    if os.path.isdir(od) and os.path.exists(os.path.join(od, "done.json")):
        return json.load(open(os.path.join(od, "done.json")))
    shutil.rmtree(od, ignore_errors=True)
    os.makedirs(od)
    log = os.path.join(od, "render.log")
    model = job["_fixes"]
    fix = nearest(model, t)
    centre = None
    if view.get("centre") == "fix" and fix:
        centre = (fix["lat"], fix["lon"])
    elif view.get("centre") == "bt":
        la, lo = interp(job["_bt"], t, "lat"), interp(job["_bt"], t, "lon")
        if la is not None:
            centre = (la, lo)
    elif isinstance(view.get("centre"), list):
        centre = tuple(view["centre"])
    if centre is None and view.get("centre") not in (None, "native"):
        centre = (fix["lat"], fix["lon"]) if fix else None
    wrfout = fr.get("wrfout")
    if fr.get("history"):  # hex: convert the history onto a lat-lon window first
        sp = view.get("spacing", 0.03)
        # Convert a window 30 % wider than the drawn box: the renderer draws the
        # lat-lon window in a conic projection, and a window cut to the box
        # left empty wedges at its corners.
        hx, hy = view["half"][0] * 1.3, view["half"][1] * 1.3
        c = centre or tuple(view.get("fallback_centre"))
        snap = lambda v: round(v / sp) * sp
        S, N, Wl, El = snap(c[0] - hy), snap(c[0] + hy), snap(c[1] - hx), snap(c[1] + hx)
        cd = os.path.join(od, "conv")
        os.makedirs(cd)
        conv = [tools["convert"], "--history", fr["history"], "--mesh", fr["static"], "--out-dir", cd, "--window",
                f"latlon:{S:.4f},{N:.4f},{Wl:.4f},{El:.4f},{sp}", "--field-set", "full", "--simulation-start",
                job["forecast_start"], "--model-label", job.get("convert_label", "hex"), "--clobber"]
        if fr.get("init"):  # zgrid (heights: MSLP, simulated IR) and terrain come from the segment's init
            conv += ["--init", fr["init"]]
        rc = run(conv, log)
        cands = glob.glob(os.path.join(cd, "wrfout_d01_*"))
        if rc != 0 or not cands:
            return {"error": f"convert rc={rc}"}
        wrfout = cands[0]
    bounds = None
    if centre and view.get("half"):
        bounds = bounds_for(centre, view["half"])
        if view.get("clamp_to_grid") and wrfout:
            bounds = clamp_inside(bounds, wrf_extent(wrfout))
    ov = os.path.join(od, "overlays.json")
    overlay_json(ov, job["_bt"], model, t, fix)
    cmd = [tools["render"], "--store-root", os.path.join(od, "store"), "--out-dir", os.path.join(od, "png"),
           "--theme", job["theme"], "--products", ",".join(view["products"]),
           "--width", str(view.get("width", 1200)), "--height", str(view.get("height", 1000)),
           "--source-label", job["source_label"], "--overlays", ov]
    if job.get("forecast_start") and not fr.get("history"):
        cmd += ["--forecast-start", job["forecast_start"]]
    if bounds:
        cmd += ["--bounds", ",".join(f"{v:.4f}" for v in bounds)]
    if view.get("streamlines"):
        cmd += ["--streamlines"]
    cmd += [wrfout]
    rc = run(cmd, log)
    out = rendered(log)
    out["_rc"] = rc
    out["_fix"] = fix and dict(lat=fix["lat"], lon=fix["lon"], mslp=fix["mslp"], vmax=fix["vmax"])
    out["_bounds"] = bounds
    if view.get("goes_dir") and bounds:  # GOES-16 band 13 on the same box, drawn by the CF lane
        g = os.path.join(view["goes_dir"], "goes16_c13_" + t.strftime("%Y%m%d%H") + ".nc")
        if os.path.exists(g):
            bov = os.path.join(od, "overlays-bt.json")
            overlay_json(bov, job["_bt"], [], t, None)
            run([tools["render"], "--cf", g, "--variable", "bt_k", "--out-dir", os.path.join(od, "goes"),
                 "--cf-table", view["goes_table"], "--bounds", ",".join(f"{v:.4f}" for v in bounds),
                 "--overlays", bov, "--width", str(view.get("width", 1200)), "--height", str(view.get("height", 1000))], log)
            pngs = glob.glob(os.path.join(od, "goes", "**", "*.png"), recursive=True)
            if pngs:
                out["goes_ir"] = pngs[0]
    if view.get("mrms_dir") and wrfout and tools.get("compare"):  # MRMS beside the model, rw_compare
        before = sum(1 for _ in open(log))
        run([tools["compare"], "--store-root", os.path.join(od, "cstore"), "--out-dir", os.path.join(od, "cmp"),
             "--reference", "mrms", "--reference-dir", view["mrms_dir"], "--offline", "--products",
             "composite_reflectivity", "--difference", "off", "--theme", job["theme"], "--width", "1200",
             "--height", "1000", wrfout], log)
        for k, line in enumerate(open(log)):
            if k >= before and line.startswith("RENDERED\t"):
                out["mrms_sheet"] = line.rstrip("\n").split("\t")[-1]
        shutil.rmtree(os.path.join(od, "cstore"), ignore_errors=True)
    shutil.rmtree(os.path.join(od, "store"), ignore_errors=True)
    if fr.get("history"):
        shutil.rmtree(os.path.join(od, "conv"), ignore_errors=True)
    json.dump(out, open(os.path.join(od, "done.json"), "w"))
    return out


def prepare(job):
    job["_bt"] = bdeck(job["bdeck"])
    job["_fixes"] = load_fixes(job["fixes"])
    job["_recon"] = recon(job.get("recon"))
    return job


def cmd_frames(path):
    job = prepare(json.load(open(path)))
    work = []
    for view in job["views"]:
        for fr in job["frames"]:
            if view.get("domain") and fr.get("domain") != view["domain"]:
                continue
            if view.get("tags") and fr["tag"] not in view["tags"]:
                continue
            work.append((fr, view))
    results = {}
    with cf.ThreadPoolExecutor(job.get("parallel", 8)) as ex:
        futs = {ex.submit(frame_job, job, fr, view): (fr["tag"], view["name"]) for fr, view in work}
        for f in cf.as_completed(futs):
            tag, vn = futs[f]
            try:
                r = f.result()
            except Exception as e:  # report and continue: one bad frame must not stop the loop
                r = {"error": repr(e)}
            results.setdefault(vn, {})[tag] = r
            bad = [k for k in r if not k.startswith("_")] if "error" not in r else []
            print(f"{vn} {tag} {'ERR ' + r['error'] if 'error' in r else ','.join(sorted(bad))}", flush=True)
    json.dump(results, open(os.path.join(job["work"], "frames.json"), "w"), indent=1, default=str)


# ------------------------------------------------------------------ strip panel
def strip_png(job, t, path, width, t_range, models=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    bt, model, rc = job["_bt"], job["_fixes"], job["_recon"]
    if not models:
        models = [dict(fixes=model, name=job["model_name"], color="#d0103a")]
    t0, t1 = t_range
    pad = dt.timedelta(hours=1)
    dpi = 100
    fig, axes = plt.subplots(1, 2, figsize=(width / dpi, 2.5), dpi=dpi)
    fig.patch.set_facecolor("#f6f7f9")
    naive = lambda xs: [x["t"].replace(tzinfo=None) for x in xs]
    btw = [p for p in bt if t0 - dt.timedelta(hours=6) <= p["t"] <= t1 + dt.timedelta(hours=6)]
    rcw = [p for p in rc if t0 - pad <= p["t"] <= min(t, t1) + dt.timedelta(minutes=5)]
    for ax, key, title, unit in ((axes[0], "mslp", "Central pressure", "hPa"), (axes[1], "vmax", "Max 10 m wind", "kt")):
        ax.set_facecolor("#ffffff")
        ax.plot(naive(btw), [p[key] for p in btw], color="#111111", lw=1.6, ls="--", marker="o", ms=3, label="NHC best track")
        if key == "mslp" and rcw:
            ax.scatter(naive(rcw), [p["mslp"] for p in rcw], s=10, color="#8a8f98", zorder=3, label="recon (VDM)")
        vals = []
        for m in models:
            past = [p for p in m["fixes"] if t0 - pad <= p["t"] <= t + dt.timedelta(minutes=5)]
            if past:
                ax.plot(naive(past), [p[key] for p in past], color=m["color"], lw=2.2, label=m["name"])
                ax.scatter([past[-1]["t"].replace(tzinfo=None)], [past[-1][key]], s=28, color=m["color"], zorder=4)
                if abs((past[-1]["t"] - t).total_seconds()) < 3600:
                    vals.append(f"{m['name']} {past[-1][key]:.0f}")
        ax.axvline(t.replace(tzinfo=None), color="#3b82c4", lw=1.2)
        ax.set_xlim((t0 - pad).replace(tzinfo=None), (t1 + pad).replace(tzinfo=None))
        allv = [p[key] for p in btw] + [p[key] for m in models for p in m["fixes"] if t0 - pad <= p["t"] <= t1 + pad]
        if key == "mslp":
            allv += [p["mslp"] for p in rc if t0 - pad <= p["t"] <= t1 + pad]
        if allv:
            lo, hi = min(allv), max(allv)
            m = max(4.0, 0.08 * (hi - lo))
            ax.set_ylim(lo - m, hi + m)
        if key == "mslp":
            ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))
        ax.tick_params(labelsize=8, colors="#333333")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e3e6ea", lw=0.6)
        bv = interp(bt, t, key)
        txt = f"{title} ({unit})   " + "   ".join(vals)
        if bv is not None:
            txt += f"   best track {bv:.0f}"
        ax.set_title(txt, fontsize=9.5, loc="left", color="#111111")
    axes[0].legend(fontsize=7.5, loc="lower left", frameon=False, ncol=2 + len(models))
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def header_png(text, sub, width, path):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (width, 56), "#111418")
    d = ImageDraw.Draw(im)
    try:
        fb = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        fr = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        fb = fr = ImageFont.load_default()
    d.text((14, 6), text, fill="#ffffff", font=fb)
    d.text((14, 34), sub, fill="#b9c2cc", font=fr)
    im.save(path)


def stack(paths, out, cols=1, bg="#f6f7f9"):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    if cols == 1:
        w = max(i.width for i in ims)
        h = sum(i.height for i in ims)
        canvas = Image.new("RGB", (w, h), bg)
        y = 0
        for i in ims:
            canvas.paste(i, (0, y)); y += i.height
    else:
        w = sum(i.width for i in ims)
        h = max(i.height for i in ims)
        canvas = Image.new("RGB", (w, h), bg)
        x = 0
        for i in ims:
            canvas.paste(i, (x, 0)); x += i.width
    canvas.save(out)
    return out


def blank_panel(text, w, h, path):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (w, h), "#f6f7f9")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        f = ImageFont.load_default()
    d.text((40, h // 2 - 12), text, fill="#555555", font=f)
    im.save(path)
    return path


def gif(frames, out, width, ms=350, last_ms=1500):
    from PIL import Image
    seq = []
    for p in frames:
        im = Image.open(p).convert("RGB")
        if im.width != width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        seq.append(im.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    durs = [ms] * len(seq)
    durs[-1] = last_ms
    seq[0].save(out, save_all=True, append_images=seq[1:], duration=durs, loop=0, optimize=True, disposal=1)
    return os.path.getsize(out)


def fit(path, width):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    return im


def cmd_loop(spec_path):
    """Compose one loop: header + rows of Rust panels (+ intensity strip), GIF + MP4 frames."""
    from PIL import Image
    spec = json.load(open(spec_path))
    out = spec["out"]
    fdir = os.path.join(out, "frames-" + spec["name"])
    shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir)
    job = prepare(json.load(open(spec["job"]))) if spec.get("job") else None
    models = [dict(fixes=load_fixes(m["fixes"]), name=m["name"], color=m["color"]) for m in spec.get("strip_models", [])]
    pw = spec["panel_width"]
    t0, t1 = T(spec["t0"]), T(spec["t1"])
    paths = []
    for i, fr in enumerate(spec["frames"]):
        t = T(fr["valid"])
        rows = []
        for row in fr["panels"]:
            ims = []
            for p in row:
                span = 1
                if isinstance(p, dict):
                    span, p = p.get("span", 1), p.get("p")
                if isinstance(p, str) and os.path.exists(p):
                    ims.append(fit(p, pw * span))
                else:
                    msg = p[1:] if isinstance(p, str) and p.startswith("!") else "no data at this time"
                    ims.append(fit(blank_panel(msg, 1200 * span, 1000, os.path.join(fdir, "_blank.png")), pw * span))
            h = max(im.height for im in ims)
            row_im = Image.new("RGB", (sum(im.width for im in ims), h), "#f6f7f9")
            x = 0
            for im in ims:
                row_im.paste(im, (x, 0)); x += im.width
            rows.append(row_im)
        width = max(r.width for r in rows)
        parts = []
        hp = os.path.join(fdir, "_h.png")
        header_png(spec["title"], fr.get("sub", spec.get("sub", "")), width, hp)
        parts.append(Image.open(hp).convert("RGB"))
        parts += rows
        if job:
            sp = os.path.join(fdir, "_s.png")
            strip_png(job, t, sp, width, (t0, t1), models or None)
            parts.append(Image.open(sp).convert("RGB"))
        H = sum(p.height for p in parts)
        canvas = Image.new("RGB", (width, H + (H % 2)), "#f6f7f9")
        y = 0
        for p in parts:
            canvas.paste(p, (0, y)); y += p.height
        if canvas.width % 2:
            canvas = canvas.crop((0, 0, canvas.width - 1, canvas.height))
        fp = os.path.join(fdir, f"{i:03d}.png")
        canvas.save(fp)
        paths.append(fp)
    for f in ("_h.png", "_s.png", "_blank.png"):
        try:
            os.remove(os.path.join(fdir, f))
        except OSError:
            pass
    g = os.path.join(out, spec["name"] + ".gif")
    size = gif(paths, g, spec.get("gif_width", 1000), ms=spec.get("ms", 350))
    while size > 24e6 and spec.get("gif_width", 1000) > 500:
        spec["gif_width"] = int(spec.get("gif_width", 1000) * 0.85)
        size = gif(paths, g, spec["gif_width"], ms=spec.get("ms", 350))
    print(f"GIF {g} {size/1e6:.1f} MB {len(paths)} frames width {spec.get('gif_width', 1000)}", flush=True)
    ff = shutil.which("ffmpeg")
    if ff:
        m = os.path.join(out, spec["name"] + ".mp4")
        fps = 1000.0 / spec.get("ms", 350)
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i", os.path.join(fdir, "%03d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-preset", "medium", "-movflags", "+faststart", m], check=False)
        if os.path.exists(m):
            print(f"MP4 {m} {os.path.getsize(m)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "frames":
        cmd_frames(sys.argv[2])
    elif sys.argv[1] == "loop":
        cmd_loop(sys.argv[2])
