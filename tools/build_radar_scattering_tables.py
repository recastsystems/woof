# SPDX-License-Identifier: Apache-2.0
"""Build finite-particle T-matrix tables with the MIT Rust solver rustmatrix.

Jung et al. (2008), doi:10.1175/2007MWR2083.1, sections 3b-c:
orientation averaging, additive covariance and two-stage dielectric mixing.
Ryzhkov et al. (2011), doi:10.1175/2010JAMC2363.1: integrate scattering
over particle sizes, retaining complex covariance, rather than averaging ZDR.
Wolfensberger and Berne (2018), doi:10.5194/amt-11-3883-2018: separate
microphysical closure, scattering and radar geometry.

This is a research spheroid closure, not a unique prediction of particle
shape by bulk microphysics. D is equal-volume diameter, density is bulk
particle density, and liquid fraction is a mass fraction. All tables carry
their coordinates, solver version, assumptions and content digest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

REVISION = "simradar-tmatrix-v1"

SPECIES = ("rain", "snow", "ice", "graupel", "hail")
BANDS_MM = {"S": 111.0, "C": 53.5, "X": 33.3}
# Closure rows: axis ratio c/a and orientation standard deviation in degrees.
CLOSURES = ((1., 10.), (.75, 20.), (.6, 20.), (.9, 40.), (.75, 60.))
DENSITIES = np.array([25., 50., 100., 200., 400., 500., 700., 900., 1000.])
LIQUID = np.array([0., .1, .3, .5, .8, 1.])
ELEVATIONS = np.array([0., 10., 30., 60., 90.])
COMPONENTS = ("zh_mm6", "zv_mm6", "cov_re_mm6", "cov_im_mm6", "kdp_deg_km", "ah_db_km", "av_db_km")


def table_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arrays[name]).tobytes())
    return h.hexdigest()


def build(path, *, band="S", jobs=6, diameters=192):
    """Build once using Rust/Rayon, with a bounded worker pool. No GPU lease."""
    if band not in BANDS_MM or not 1 <= jobs <= 256 or diameters < 64:
        raise ValueError("band S/C/X, 1..256 workers and >=64 sizes required")
    # Set before loading Rust. Its diameter loop releases the Python GIL.
    os.environ["RAYON_NUM_THREADS"] = str(jobs)
    from importlib.metadata import version
    from rustmatrix import _core, refractive
    start = time.perf_counter()
    wavelength = BANDS_MM[band]
    d = np.geomspace(.002, 80., diameters)
    values = np.empty((5, len(DENSITIES), len(LIQUID), len(ELEVATIONS), len(d), 7), np.float32)
    water = refractive.m_w_0C[wavelength]
    geoms = []
    for elev in ELEVATIONS:
        geoms.extend([(90.-elev, 90.+elev, 0., 180., 0., 0.),
                      (90.-elev, 90.-elev, 0., 0., 0., 0.)])
    # Gaussian canting on the sphere, including the surface measure. Fixed
    # 32-point Gauss-Legendre on the antipodally equivalent upper hemisphere.
    # beta is measured FROM VERTICAL: its spherical Jacobian is sin(beta),
    # not cos(beta). The old cosine measure over-weighted upright particles.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    tilt = (nodes+1)*np.pi/4
    alphas = np.arange(0., 360., 45.)
    from scipy.special import erf
    for sp, (aspect, sigma) in enumerate(CLOSURES):
        for wi, wet in enumerate(LIQUID):
            density, diameter = np.meshgrid(DENSITIES, d, indexing="ij")
            dm = np.minimum(diameter, 8.)
            rain_ratio = np.clip(1.0148-.020465*dm-.020048*dm**2+.003095*dm**3-.0001453*dm**4, .45, 1.)
            if sp == 0:
                ratios = rain_ratio
                indices = np.full(diameter.shape, water, dtype=np.complex128)
                sigma_wet = sigma
            else:
                # Ryzhkov et al. 2020, 10.3390/atmos11040362, section 2.2:
                # shape approaches rain continuously as liquid mass grows.
                ratios = (1.-wet)*aspect+wet*rain_ratio
                sigma_wet = (1.-wet)*sigma+wet*CLOSURES[0][1]
                vw = wet*density/1000.
                dry_density = np.minimum(density*(1.-wet)/np.maximum(1.-vw,1e-30),916.7)
                dry = refractive.mi(wavelength, dry_density/1000.)
                ew, ed = water**2, dry**2
                # Wolfensberger and Berne 2018, equations 33-37: recursive
                # Maxwell-Garnett, with both water/dry matrix assignments.
                inc_w = vw*(ew-ed)/(ew+2*ed)
                e1 = ed*(1+2*inc_w)/(1-inc_w)
                inc_d = (1-vw)*(ed-ew)/(ed+2*ew)
                e2 = ew*(1+2*inc_d)/(1-inc_d)
                tau = erf(2*(1-wet)/wet-1) if wet>.01 else 1.
                indices = np.sqrt(.5*((1+tau)*e1+(1-tau)*e2))
            cant_weights = weights*np.sin(tilt)*np.exp(-.5*(tilt/np.radians(sigma_wet))**2)
            cant_weights /= cant_weights.sum()
            sb, zb = _core.tabulate_scatter_table_orient_avg(
                np.ascontiguousarray(diameter.ravel()),
                np.ascontiguousarray((1./ratios).ravel()),
                np.ascontiguousarray(indices.real.ravel()),
                np.ascontiguousarray(indices.imag.ravel()), geoms, alphas,
                np.ascontiguousarray(np.degrees(np.abs(tilt))),
                np.ascontiguousarray(cant_weights),
                1., wavelength, -1, 1e-3, 2)
            scale = wavelength**4/(np.pi**5*.93)*2.*np.pi
            for ei in range(len(ELEVATIONS)):
                z = zb[:,2*ei]; sf = sb[:,2*ei+1]
                row = np.stack((scale*(z[:,0,0]-z[:,0,1]-z[:,1,0]+z[:,1,1]),
                                scale*(z[:,0,0]+z[:,0,1]+z[:,1,0]+z[:,1,1]),
                                scale*(-z[:,2,2]-z[:,3,3]),
                                scale*(z[:,2,3]-z[:,3,2]),
                                1e-3*180./np.pi*wavelength*(sf[:,1,1]-sf[:,0,0]).real,
                                4.343e-3*2*wavelength*sf[:,1,1].imag,
                                4.343e-3*2*wavelength*sf[:,0,0].imag), axis=-1)
                values[sp,:,wi,ei] = row.reshape(len(DENSITIES),len(d),7)
    if not np.isfinite(values).all():
        raise ValueError("T-matrix solver produced a nonfinite table")
    arrays = dict(values=values, diameter_mm=d.astype("f4"), density=DENSITIES.astype("f4"),
                  liquid=LIQUID.astype("f4"), elevation=ELEVATIONS.astype("f4"))
    verification = verify_solver(wavelength)
    meta = dict(verification=verification, solver_license="MIT", revision=REVISION, band=band, wavelength_mm=wavelength,
                solver="rustmatrix", solver_version=version("rustmatrix"), jobs=jobs,
                seconds=time.perf_counter()-start, species=SPECIES,
                components=COMPONENTS, sha256=table_digest(arrays),
                closure="equal-volume spheroids; 0 C dielectric; Gaussian spherical canting; wet shape/orientation transition; final mixture density; dual Maxwell-Garnett liquid mixing",
                dois=["10.1175/2007MWR2083.1", "10.1175/2010JAMC2363.1", "10.5194/amt-11-3883-2018"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+".part").open("wb") as stream:
        np.savez(stream, **arrays, metadata=json.dumps(meta))
    path.with_suffix(path.suffix+".part").replace(path)
    return meta


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        arrays = {key: archive[key] for key in ("values", "diameter_mm", "density", "liquid", "elevation")}
    if meta["revision"] != REVISION or meta["sha256"] != table_digest(arrays):
        raise ValueError("scattering table revision or digest mismatch")
    if meta.get("solver") != "rustmatrix" or not meta.get("verification", {}).get("passed"):
        raise ValueError("weather products require a verified T-matrix table; rebuild the cache")
    for name in ("diameter_mm", "density", "liquid", "elevation"):
        axis = arrays[name]
        if (axis.dtype != np.dtype("f4") or axis.ndim != 1 or len(axis) < 2
                or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0)):
            raise ValueError(f"invalid scattering coordinate {name}")
    d = arrays["diameter_mm"]
    if d[0] <= 0 or not np.allclose(np.diff(np.log(d)), np.log(d[-1]/d[0])/(len(d)-1), rtol=2e-5):
        raise ValueError("diameter grid must be positive and logarithmically uniform")
    expected = (len(SPECIES), len(arrays["density"]), len(arrays["liquid"]),
                len(arrays["elevation"]), len(d), len(COMPONENTS))
    if (arrays["values"].shape != expected or arrays["values"].dtype != np.dtype("f4")
            or not np.isfinite(arrays["values"]).all()):
        raise ValueError("invalid scattering values shape, dtype or finiteness")
    if (arrays["liquid"][0] != 0 or arrays["liquid"][-1] != 1
            or arrays["elevation"][0] != 0 or arrays["elevation"][-1] != 90
            or tuple(meta.get("species", ())) != SPECIES
            or tuple(meta.get("components", ())) != COMPONENTS):
        raise ValueError("scattering table axes or component conventions do not match")
    return arrays, meta


def verify_solver(wavelength):
    """Independent Rayleigh spheroid limit, plus public API/batch conventions.

    Electrostatic depolarization factors for an oblate equal-volume spheroid;
    no forecast values or fitted offsets enter this verification.
    """
    from rustmatrix import Scatterer, radar, refractive, _core
    from rustmatrix.tmatrix_aux import geom_horiz_back, geom_horiz_forw
    water = refractive.m_w_0C[wavelength]
    eps = water**2
    errors = []
    rows = []
    for ratio in (1., .8):
        d = .05
        if ratio == 1.:
            lv = 1/3
        else:
            e = np.sqrt(1/ratio**2-1)
            lv = (1+e*e)/e**3*(e-np.arctan(e))
        lh = (1-lv)/2
        fh = (eps-1)/(1+lh*(eps-1))/3
        fv = (eps-1)/(1+lv*(eps-1))/3
        expected = np.array([d**6*abs(fh)**2/.93, d**6*abs(fv)**2/.93])
        tm = Scatterer(radius=d/2, wavelength=wavelength, m=water,
                       axis_ratio=1/ratio, Kw_sqr=.93)
        tm.set_geometry(geom_horiz_back)
        actual = np.array([radar.refl(tm), radar.refl(tm, h_pol=False)])
        err = float(np.max(abs(actual/expected-1)))
        errors.append(err)
        sb, zb = _core.tabulate_scatter_table_orient_avg(
            np.array([d]), np.array([1/ratio]), np.array([water.real]),
            np.array([water.imag]), [geom_horiz_back, geom_horiz_forw],
            np.array([0.]), np.array([0.]), np.array([1.]),
            1., wavelength, -1, 1e-3, 2)
        z = zb[0,0]; scale=wavelength**4/(np.pi**5*.93)*2*np.pi
        batch=np.array([scale*(z[0,0]-z[0,1]-z[1,0]+z[1,1]),
                        scale*(z[0,0]+z[0,1]+z[1,0]+z[1,1])])
        batch_error=float(np.max(abs(batch/actual-1)))
        errors.append(batch_error)
        if ratio == 1.:
            # Electrostatic optical theorem for a small sphere, in mm:
            # sigma_ext = 8*pi^2*r^3/lambda * Im((epsilon-1)/(epsilon+2)).
            # 4.343e-3 converts mm^2 per m^3 to one-way dB/km.
            extinction = 8*np.pi**2*(d/2)**3/wavelength*((eps-1)/(eps+2)).imag
            expected_a = 4.343e-3*extinction
            batch_a = 4.343e-3*2*wavelength*sb[0,1,1,1].imag
            attenuation_error = float(abs(batch_a/expected_a-1))
            errors.append(attenuation_error)
            if batch_a <= 0:
                raise ValueError("forward-amplitude sign produces negative extinction")
        rows.append(dict(axis_ratio_vertical_horizontal=ratio,
            rayleigh_relative_error=err, batch_public_relative_error=batch_error,
            zdr_db=float(10*np.log10(actual[0]/actual[1]))))
    passed=max(errors)<.002
    if not passed: raise ValueError(f"T-matrix limiting/convention check failed: {rows}")
    return dict(passed=passed, limit=.002, checks=rows,
                scope="electrostatic spheroids and batch polarization conventions; not a calibrated radar")


def export_native(path, arrays, meta):
    """Fixed little-endian table, read and validated by the Rust sampler."""
    import struct
    payload = b"".join(np.asarray(arrays[k], dtype="<f4").tobytes()
                       for k in ("diameter_mm", "density", "liquid", "elevation", "values"))
    header = struct.pack("<8s5If", b"RWTMAT01", len(arrays["diameter_mm"]),
                         len(arrays["density"]), len(arrays["liquid"]),
                         len(arrays["elevation"]), len(COMPONENTS), meta["wavelength_mm"])
    target = Path(path).with_suffix(".bin")
    target.write_bytes(header + payload)
    meta["native_sha256"] = hashlib.sha256(header + payload).hexdigest()
    meta["native_bytes"] = len(header) + len(payload)
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", choices=tuple(BANDS_MM), required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.out, band=args.band, jobs=args.jobs)
    arrays, metadata = load(args.out)
    export_native(args.out, arrays, metadata)
    print(json.dumps(metadata, indent=2))
