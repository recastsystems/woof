#!/usr/bin/env python3
"""Replay the real-input stream files emitted by instrument_native.py.

Run from the engine root with GPUWM_WRF_EXACT=1. The ordinary production
CUDA launchers are the implementation under test. Binary decoding and
comparison happen outside the forecast step loop.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os

import numpy as np


class Stream:
    def __init__(self, path):
        self.path = path
        self.file = path.open("rb")
        first = self.file.read(4)
        self.endian = "<" if int.from_bytes(first, "little") == 1 else ">"
        self.file.seek(0)

    def read(self, shape, kind="f4"):
        count = int(np.prod(shape))
        data = np.fromfile(self.file, dtype=self.endian + kind, count=count)
        if data.size != count:
            raise ValueError(f"truncated capture: {self.path}")
        return np.ascontiguousarray(data.reshape(shape).astype(kind))

    def finish(self):
        if self.file.read(1):
            raise ValueError(f"trailing capture data: {self.path}")
        self.file.close()


def compare(got, ref):
    got = np.ascontiguousarray(got, dtype=np.float32)
    ref = np.ascontiguousarray(ref, dtype=np.float32)
    if got.shape != ref.shape:
        raise ValueError(f"shape mismatch {got.shape} != {ref.shape}")
    unequal = got.view(np.uint32) != ref.view(np.uint32)
    return dict(words=got.size, differing_words=int(np.count_nonzero(unequal)),
                max_abs=float(np.max(np.abs(got - ref), initial=0)),
                reference_nonzero=int(np.count_nonzero(ref)))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def diffusion(path):
    import cupy as cp
    from woof.core.mynn_scalar_mix_gpu import scalar_pblmix_columns_cuda
    stream = Stream(path)
    h = stream.read((21,), "i4")
    version, ns, first = h[:3]
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte = h[3:]
    if version != 1 or ns - first + 1 != 8:
        raise ValueError("not the native eight-species tracer package")
    shape = (jme - jms + 1, kme - kms + 1, ime - ims + 1)
    dt = stream.read((1,))[0]
    q = stream.read((ns, *shape))
    dz, rho, kh = (stream.read(shape) for _ in range(3))
    before = stream.read((ns, *shape))
    stream.finish()
    output = path.with_name(path.name.replace(".in.bin", ".out.bin"))
    ref = np.fromfile(output, dtype=stream.endian + "f4").astype(np.float32).reshape(q.shape)
    sl = (slice(jts - jms, jte - jms + 1), slice(kts - kms, kte - kms + 1), slice(its - ims, ite - ims + 1))
    columns = lambda a: cp.asarray(np.ascontiguousarray(a[sl].transpose(0, 2, 1).reshape(-1, kte - kts + 1)))
    device = [columns(a) for a in (dz, rho, kh)]
    result = []
    for species in range(first - 1, ns):
        _, rate = scalar_pblmix_columns_cuda(columns(q[species]), *device, dt)
        expected = ref[species][sl].transpose(0, 2, 1).reshape(rate.shape)
        row = compare(cp.asnumpy(rate), expected)
        row.update(species=int(species - first + 2), module="diff4d", dt=float(dt))
        result.append(row)
    from woof.core.tracer_pblmix import tracer_pblmix_rates
    from woof.core.passive_tracer_contract import TRACER_NAMES
    cache = {}
    def scratch(shape, name):
        if name not in cache:
            cache[name] = cp.empty(shape, cp.float32)
        return cache[name]
    layout = lambda a: cp.asarray(np.ascontiguousarray(a[sl].transpose(1, 0, 2)))
    state = SimpleNamespace(scratch=scratch, **{name: layout(q[i]) for i, name
                            in enumerate(TRACER_NAMES, start=int(first)-1)})
    batch = tracer_pblmix_rates(state, *(layout(a) for a in (dz, rho, kh)), dt)
    for species, name in enumerate(TRACER_NAMES, start=int(first)-1):
        row = compare(cp.asnumpy(batch[name]), ref[species][sl].transpose(1, 0, 2))
        row.update(species=int(species-first+2), module="diff4d_shared_factor", dt=float(dt))
        result.append(row)
    # The capture also witnesses untouched cells and the inactive package member.
    untouched = np.ones(q.shape, bool)
    untouched[(slice(first - 1, ns), *sl)] = False
    guard = compare(before[untouched], ref[untouched])
    if guard["differing_words"]:
        raise ValueError("native diff4d changed outside its documented active bounds")
    return result, guard


def transport(path):
    import cupy as cp
    from woof.core.advection import launch_flux_div_scalar
    from woof.core.moist import launch_pd_fluxes, launch_pd_renorm_apply
    stream = Stream(path)
    h = stream.read((27,), "i4")
    version, mode = h[:2]
    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte = h[2:20]
    if version != 1 or mode not in (1, 5) or (its, jts, kts) != (ids, jds, kds):
        raise ValueError("expected one complete native MPI patch starting at the domain origin")
    rdx, rdy, dt = stream.read((3,))
    shape = (jme - jms + 1, kme - kms + 1, ime - ims + 1)
    q, old, before, ru, rv, ww, unused = (stream.read(shape) for _ in range(7))
    mu, mub, mu_old, msfux, msfuy, msfvx, msfvy, msftx, msfty = (stream.read((shape[0], shape[2])) for _ in range(9))
    c1, c2, fnm, fnp, rdnw, rdn = (stream.read((shape[1],)) for _ in range(6))
    stream.finish()
    output = path.with_name(path.name.replace(".in.bin", ".out.bin"))
    ref = np.fromfile(output, dtype=stream.endian + "f4").astype(np.float32).reshape(shape)
    nz, ny, nx = kde - kds, jde - jds, ide - ids
    def crop(a, stagger="mass"):
        sy, sx, sz = int(stagger == "v"), int(stagger == "u"), int(stagger == "w")
        if a.ndim == 2:
            b = a[jds - jms:jds - jms + ny + sy, ids - ims:ids - ims + nx + sx]
        else:
            b = a[jds - jms:jds - jms + ny + sy, kds - kms:kds - kms + nz + sz,
                  ids - ims:ids - ims + nx + sx].transpose(1, 0, 2)
        return np.ascontiguousarray(b)
    dev = lambda a, stagger="mass": cp.asarray(crop(a, stagger))
    coord = SimpleNamespace(**{name: cp.asarray(a[kds - kms:kds - kms + nz + 1])
                               for name, a in (("c1h", c1), ("c2h", c2), ("fnm", fnm),
                                               ("fnp", fnp), ("rdnw", rdnw), ("rdn", rdn))})
    coord.mub2d, coord.mup0 = dev(mub), dev(mu_old)
    flags = h[25]
    if not flags & (4 | 8) or flags & (1 | 2 | 16 | 32 | 64 | 128 | 4096):
        raise ValueError("this real-input replay expects specified/nested lateral boundaries")
    common = dict(open_x=True, open_y=True, has_msf=True)
    # Launchers' open flags mean nonperiodic axes, with spec selecting WRF's
    # specified/nested edge degradation rather than radiative open edges.
    ru, rv, ww = dev(ru, "u"), dev(rv, "v"), dev(ww, "w")
    tend = dev(before)
    msft = dev(msftx)
    dx, dy = float(np.float32(1) / rdx), float(np.float32(1) / rdy)
    if mode == 1:
        launch_flux_div_scalar(dev(q), ru, rv, ww, tend, coord, dx, dy,
                               msf=msft, spec=True, vorder=int(h[24]), **common)
    else:
        shapes = ((nz, ny, nx + 1),) * 2 + ((nz, ny + 1, nx),) * 2 + ((nz + 1, ny, nx),) * 2
        fluxes = [cp.empty(s, cp.float32) for s in shapes]
        launch_pd_fluxes(dev(q), dev(old), ru, rv, ww, dev(mu), coord, dx, dy, dt,
                         *fluxes, msft=msft, vorder=int(h[24]), **common)
        # Native receives base and perturbation mass as separate inputs;
        # production keeps their sum for this exact dry-mass conversion.
        launch_pd_renorm_apply(dev(old), dev(mub + mu_old), *fluxes, tend=tend,
                               coord=coord, dx=dx, dy=dy, dt=dt, msft=msft, **common)
    result = compare(cp.asnumpy(tend), crop(ref))
    result.update(module="advect_scalar_pd" if mode == 5 else "advect_scalar",
                  capture=path.name, shape=[int(nz), int(ny), int(nx)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        raise ValueError("bitwise proof requires GPUWM_WRF_EXACT=1")
    rows, guard = diffusion(args.captures / "tracer-diff4d.in.bin")
    for path in sorted(args.captures.glob("tracer-0*.in.bin")):
        rows.append(transport(path))
    result = dict(rows=rows, untouched_diffusion_guard=guard,
                  words=sum(row["words"] for row in rows),
                  differing_words=sum(row["differing_words"] for row in rows),
                  captures={p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                            for p in sorted(args.captures.glob("tracer-*.bin"))})
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "captures")}, indent=2))
    if result["differing_words"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
