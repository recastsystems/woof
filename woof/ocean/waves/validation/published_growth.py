"""One bounded ST4 growth experiment; published curves remain separate referees.

No source coefficient is selected by this experiment. Numerical charts are not
weather maps. The 2010 paper supplies 8/48 h spectra, not a stationary PM gate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

G = 9.80665


def run(output, *, budget_seconds=570.):
    if not 30 <= budget_seconds <= 570:
        raise ValueError("growth wall budget must be between 30 and 570 seconds")
    import cupy as cp
    from .growth_smoke import HomogeneousBackend, fetch_backend
    from ..action import ActionSolver, PACKAGE
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {
        "schema": "gpuwm.waves.published-growth/v1", "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "physics_sha256": {p: hashlib.sha256((PACKAGE/p).read_bytes()).hexdigest()
                           for p in ("physics.cu", "kernels.cu", "source_packs.json")},
        "gpu": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
        "config": {"source_pack": "st4", "dt_seconds": 600, "nf": 36, "nd": 36,
                   "duration_hours": 240, "fetch_hours": 72, "depth_m": 1000,
                   "initial": "calm", "fetch_grid": [4, 128],
                   "crosswind_boundary": "periodic", "budget_seconds": budget_seconds},
        "references": {
            "st4": "10.1175/2010JPO4324.1, Figs 4/5: 10 m/s, 8/48 h; p1925: continued slow growth",
            "jonswap": "1973 DHZ A8(12), epsilon=1.6e-7 chi; nu=3.5 chi^(-1/3)",
            "kc1992": "10.1175/1520-0485(1992)022<1389:RWGDOA>2.0.CO;2; composite epsilon=5.2e-7 chi^.9, nu=13.7/(2pi) chi^(-.27)",
            "spm1984": "Vol I Eqs 3-28a,3-33..38: UA=.71 U10^1.23; H*=.0016 chiA^.5; T*=.2857 chiA^(1/3); t*=68.8 chiA^(2/3); full H*=.2433,T*=8.134",
            "pm": "10.1175/1520-0485(2003)033<1301:RTPALF>2.0.CO;2: epsilon_PM=.00364, reanalysis mean=.00402; nu_PM=.13",
        }, "results": [],
        "qualification": "Measurements and referee comparisons; DONE does not assert a science pass."
    }
    def save():
        temporary = output/"receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2)+"\n")
        temporary.replace(output/"receipt.json")
    save()
    try:
        for speed in (10., 20.):
            for mode in ("duration", "fetch"):
                backend = HomogeneousBackend() if mode == "duration" else fetch_backend()
                shape = backend.full_shape
                model = ActionSolver(None, None, {"grid_backend": backend, "source_pack": "st4"})
                model.set_forcing(SimpleNamespace(u10=cp.full(shape, speed, cp.float32),
                    v10=cp.zeros(shape, cp.float32), rho_air=cp.full(shape, 1.225, cp.float32), time_seconds=0.))
                item = {"wind_ms": speed, "mode": mode, "hourly": [], "spectra": []}
                stream = cp.cuda.Stream(non_blocking=True)
                stream.wait_event(cp.cuda.get_current_stream().record())
                with stream:
                    stream.begin_capture()
                    model.step(600.)
                    graph = stream.end_capture()
                cp.cuda.get_current_stream().wait_event(stream.record())
                hours = 240 if mode == "duration" else 72
                for hour in range(1, hours+1):
                    for _ in range(6):
                        graph.launch(stream=cp.cuda.get_current_stream())
                    summary = model.collect()
                    center = (0,) if mode == "duration" else (2, 64)
                    item["hourly"].append({"hour": hour, "hs_m": float(model.outputs.hs[center]),
                        "tp_s": float(model.outputs.tp[center]), "ustar": float(model.outputs.ustar[center]),
                        "z0_m": float(model.outputs.z0[center])})
                    if mode == "duration" and hour in (8, 48, 240):
                        terms = model.source_diagnostics(0)
                        item["spectra"].append({"hour": hour,
                            "frequency_hz": cp.asnumpy(model.frequency).tolist(),
                            "efth": cp.asnumpy(model.state[0]*(2*math.pi*model.frequency[:, None])).tolist(),
                            **{k: cp.asnumpy(v).tolist() for k,v in terms.items() if k != "keepalive"}})
                        del terms
                    if mode == "fetch" and hour in (48, 60, 72):
                        item.setdefault("profiles", []).append({"hour": hour,
                            "fetch_m": cp.asnumpy(backend.fetch_centers).tolist(),
                            "hs_m": cp.asnumpy(model.outputs.hs[2]).tolist(),
                            "tp_s": cp.asnumpy(model.outputs.tp[2]).tolist(),
                            "ustar": cp.asnumpy(model.outputs.ustar[2]).tolist()})
                    if time.perf_counter()-started > budget_seconds:
                        raise TimeoutError(f"bounded growth experiment exceeded {budget_seconds:g} seconds")
                if summary["steps"] != hours*6 or summary["time_seconds"] != hours*3600:
                    raise AssertionError("graph did not advance the exact physical clock")
                item["summary"] = summary
                receipt["results"].append(item)
                save()
                print(json.dumps({"finished": mode, "wind_ms": speed, "last": item["hourly"][-1],
                                  "wall_seconds": time.perf_counter()-started}), flush=True)
                del graph, model, backend
                cp.cuda.get_current_stream().synchronize()
                cp.get_default_memory_pool().free_all_blocks()
        receipt["status"] = "DONE"
    except BaseException as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter()-started
        save()
    charts(output, receipt)


def spm(speed, time_seconds):
    ua = .71*speed**1.23
    chi = (G*np.asarray(time_seconds)/ua/68.8)**1.5
    hs = np.minimum(.0016*np.sqrt(chi), .2433)*ua**2/G
    tp = np.minimum(.2857*np.cbrt(chi), 8.134)*ua/G
    return (G*hs/speed**2/4)**2, speed/(G*tp)


def charts(output, receipt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reference = json.loads(Path(__file__).with_name("published_spectra.json").read_text())
    for mode in ("fetch", "duration"):
        fig, axes = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
        for column, speed in enumerate((10.,20.)):
            a = next(a for a in receipt["results"] if a["wind_ms"] == speed and a["mode"] == mode)
            if mode == "fetch":
                p = a["profiles"][-1]
                x = np.asarray(p["fetch_m"])*G/speed**2
                h, tp = np.asarray(p["hs_m"]), np.asarray(p["tp_s"])
                for label, energy, nu in (("JONSWAP",1.6e-7*x,3.5*x**(-1/3)),
                        ("KC92 composite",5.2e-7*x**.9,13.7/(2*np.pi)*x**(-.27))):
                    # Empirical growth laws are not extrapolated past PM maturity.
                    young = nu >= .13
                    axes[0,column].loglog(x[young],energy[young],"--",label=label)
                    axes[1,column].loglog(x[young],nu[young],"--",label=label)
                for p in a["profiles"][:-1]:
                    axes[0,column].loglog(x,(G*np.asarray(p["hs_m"])/speed**2/4)**2,color=".75",lw=.7)
            else:
                times = np.asarray([h["hour"] for h in a["hourly"]])*3600
                x = times*G/speed
                h = np.asarray([h["hs_m"] for h in a["hourly"]])
                tp = np.asarray([h["tp_s"] for h in a["hourly"]])
                energy, nu = spm(speed,times)
                axes[0,column].loglog(x,energy,"--",label="SPM84 (published UA conversion)")
                axes[1,column].loglog(x,nu,"--",label="SPM84")
                if speed == 10:
                    ref = reference["st4_2010"]["figure5_TEST441"]
                    axes[0,column].plot([48*3600*G/speed],[(G*ref["hs_m"]/speed**2/4)**2],
                                       "o",label="2010 ST4, 48 h PDF estimate")
                    for hour, key in ((8,"figure4_TEST441"),(48,"figure5_TEST441")):
                        ref = reference["st4_2010"][key]
                        axes[1,column].plot([hour*3600*G/speed],[ref["peak_frequency_hz"]*speed/G],"ko")
                else:
                    refs = reference["st4_2019"]["rows"]
                    xt = np.asarray([v["hour"] for v in refs])*3600*G/speed
                    axes[0,column].plot(xt,[(G*v["hs_m"]/speed**2/4)**2 for v in refs],"o",
                                       label="2019 ST4D, PDF estimates (different settings)")
                    axes[1,column].plot(xt,[v["peak_frequency_hz"]*speed/G for v in refs],"ko")
            axes[0,column].loglog(x,(G*h/speed**2/4)**2,label="ST4")
            axes[1,column].loglog(x,speed/(G*tp),label="ST4")
            axes[0,column].axhline(.00364,color=".5",ls=":",label="PM64")
            axes[0,column].axhline(.00402,color=".3",ls="-.",label="PM reanalysis mean")
            axes[1,column].axhline(.13,color=".5",ls=":",label="PM64")
            axes[0,column].set_title(f"{speed:g} m/s; {mode}")
            axes[0,column].set_ylabel("g² m0 / U10⁴")
            axes[1,column].set_ylabel("fp U10 / g")
            for ax in axes[:,column]:
                ax.set_xlabel("g X / U10²" if mode=="fetch" else "g t / U10")
                ax.grid(True,which="both",alpha=.2)
                ax.legend(fontsize=8)
        fig.savefig(output/f"{mode}.png",dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--analyse", action="store_true", help="redraw saved results; no integration")
    args = parser.parse_args()
    if args.analyse:
        charts(args.output, json.loads((args.output/"receipt.json").read_text()))
    else:
        run(args.output)
