"""Write the Pleim-Xiu LSM oracle's case files.

Every case is one call geometry of WRF v4.6.1's ``pxlsm`` driver
(phys/module_sf_pxlsm.F:27): a NX x NY tile of independent columns, run for
NSTEPS consecutive calls by run_pxlsm.F90 with the state carried between
calls exactly as the surface driver carries it.  The values are drawn from
a seeded generator inside physical ranges; nothing here is an expected
answer, every expected number is a word the Fortran wrote.

Coverage is chosen by what the scheme branches on, and each family is
placed so that it lands both on nudged interior columns and on the
unnudged edge ring (pxlsm zeroes NUDGE for I<=2, I>=IDE-1, J<=2, J>=JDE-1,
:600-602):

* land cover: bare soil, crop (the VEGFRAC-seasonal Zo/LAI branch), forest,
  urban with impervious and canopy fractions (the NLCD40 adjustment and the
  impervious heat capacity), wetlands (the WETFRA soil-moisture floor),
  partial water (the 0.10 < FWAT < 0.50 roughness arm), open water, cold
  water converted to ice (SST <= 270), analysed sea ice (XICE >= 0.5),
  snow-covered land (PXSNOW's fractional cover and the snow heat capacity);
* soil: every texture class SOILPROP's triangle reaches, a column with no
  soil fraction at all (the clay-loam default), wet soil above field
  capacity and dry soil below the wilting point (the Giard-Bazile C1 arm);
* forcing: day and night, unstable/stable/neutral RMOL (RMOL = 0 exactly is
  pxlsm's MOLX = 1000 arm), rain and no rain, dew (QSS < QV1);
* driver switches: every land-use table, pxlsm_smois_init 0 and 1,
  pxlsm_soil_nudge 0 and 1, pxlsm_modis_veg 0 and 1, and three time steps
  (one, two and three PX sub-steps).

Usage: python gen_cases.py OUTDIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import (ATM_3D, IN_2D, INOUT_2D, OUT_2D, OUT_SENTINEL,  # noqa: E402
                    SOIL_3D)

NX, NY = 12, 8
NSCAT = 16

#: Per-table category roles.  Indices are 1-based, as in
#: module_sf_pxlsm_data.F.  ``water`` is VEGELAND's KWAT.
DATASETS = {
    "USGS": dict(nlcat=24, water=16, crop=[2, 3, 4, 5, 6],
                 forest=[11, 12, 13, 14, 15], urban=[1], barren=[19, 23],
                 snow=[24], wet=[17, 18], grass=[7, 8, 9, 10, 20, 21, 22]),
    "USGS28": dict(nlcat=28, water=16, crop=[2, 3, 4, 5, 6],
                   forest=[11, 12, 13, 14, 15], urban=[1],
                   barren=[19, 23, 25, 26, 27], snow=[24], wet=[17, 18],
                   grass=[7, 8, 9, 10, 20, 21, 22]),
    "MODIS20": dict(nlcat=20, water=17, crop=[12, 14],
                    forest=[1, 2, 3, 4, 5], urban=[13], barren=[16, 20],
                    snow=[15], wet=[11], grass=[6, 7, 8, 9, 10, 18, 19]),
    "MODIS21": dict(nlcat=21, water=17, crop=[12, 14],
                    forest=[1, 2, 3, 4, 5], urban=[13], barren=[16, 20],
                    snow=[15], wet=[11], grass=[6, 7, 8, 9, 10, 18, 19]),
    "MODIS61": dict(nlcat=61, water=17, crop=[12, 14],
                    forest=[1, 2, 3, 4, 5], urban=[13, 51, 55, 61],
                    barren=[16, 20], snow=[15], wet=[11],
                    grass=[6, 7, 8, 9, 10, 18, 19]),
    "NLCD40": dict(nlcat=40, water=17, crop=[12, 14, 37, 38],
                   forest=[1, 2, 3, 4, 5, 28, 29, 30],
                   urban=[13, 23, 24, 25, 26], barren=[16, 27],
                   snow=[15, 22], wet=[11, 39, 40],
                   grass=[6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 36]),
    "NLCD50": dict(nlcat=50, water=1, crop=[19, 20, 43, 45],
                   forest=[9, 10, 11, 32, 33, 34, 35, 36],
                   urban=[3, 4, 5, 6, 44], barren=[7, 8, 47],
                   snow=[2, 46], wet=[21, 22, 23, 27, 28, 42],
                   grass=[12, 13, 14, 15, 16, 17, 18, 37, 38, 39, 40, 41]),
}

#: STATSGO categories (1-based) whose (coarse sand, fine sand, clay) mixes
#: reach each of SOILPROP's twelve texture classes.  14 is water, 15
#: bedrock, 16 other.
SOIL_MIXES = [
    {1: 1.0}, {2: 1.0}, {3: 1.0}, {4: 1.0}, {5: 1.0}, {6: 1.0},
    {7: 1.0}, {8: 1.0}, {9: 1.0}, {10: 1.0}, {11: 1.0}, {12: 1.0},
    {1: 0.5, 12: 0.5}, {4: 0.3, 9: 0.7}, {3: 0.6, 6: 0.4},
    {15: 1.0}, {16: 1.0}, {13: 1.0}, {5: 0.8, 11: 0.2}, {2: 0.4, 10: 0.6},
    {},          # no soil fraction at all: SOILPROP's clay-loam default
]

#: Column kinds, cycled over the tile so each lands on both nudged and
#: unnudged positions.
KINDS = ("crop", "forest", "bare", "urban", "grass", "wet", "partwater",
         "water", "coldwater", "seaice", "snowland", "drysoil", "wetsoil",
         "crop", "forest", "grass")

CASES = [
    # name, dataset, dt, smois_init, nudge, modis_veg, day, anal, curr0
    ("usgs-day-init", "USGS", 60.0, 1, 0, 0, True, 0, 0.0),
    ("usgs-day-nudge", "USGS", 60.0, 0, 1, 0, True, 10800, 1800.0),
    ("usgs-night-nudge-init", "USGS", 90.0, 1, 1, 0, False, 3600, 0.0),
    ("usgs-night", "USGS", 30.0, 0, 0, 0, False, 0, 0.0),
    ("usgs28-day", "USGS28", 60.0, 1, 0, 0, True, 0, 0.0),
    ("modis20-day-vegon", "MODIS20", 60.0, 1, 0, 1, True, 0, 0.0),
    ("modis21-night-nudge", "MODIS21", 40.0, 0, 1, 0, False, 21600, 7200.0),
    ("modis61-day-nudge-vegon", "MODIS61", 60.0, 1, 1, 1, True, 10800,
     10740.0),
    ("nlcd40-day-urban", "NLCD40", 60.0, 1, 0, 0, True, 0, 0.0),
    ("nlcd40-night-nudge", "NLCD40", 30.0, 0, 1, 0, False, 10800, 0.0),
    ("nlcd50-day", "NLCD50", 60.0, 1, 0, 0, True, 0, 0.0),
    ("nlcd50-day-nudge-vegon", "NLCD50", 90.0, 0, 1, 1, True, 3600, 600.0),
    # A land column covered only by a water category that is NOT VEGELAND's
    # KWAT (MODIS 21, inland lakes).  real.exe folds the lake fraction into
    # ISWATER before any LSM sees it (module_initialize_real.F:2857-2859),
    # so a WRF run never presents this; it is here to pin what pxlsm does
    # with it anyway -- every vegetated category's LAI0 is zero, so
    # RSTMIN = SUMLAI/SUMRSI = 0/0 (:997) and the column goes NaN -- which
    # is the defect woof declares and defines.  Only the columns listed in
    # DEFECT_COLUMNS take it.
    ("modis21-lake-defect", "MODIS21", 60.0, 1, 0, 0, True, 0, 0.0),
]

#: (j, i) columns the defect case covers with MODIS category 21 only.
DEFECT_COLUMNS = ((3, 4), (5, 8))

NSTEPS = 6


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def landuse_fractions(rng, ds, kind):
    """One column's LANDUSEF (sums to 1) for a column kind."""
    nlcat, water = ds["nlcat"], ds["water"]
    frac = np.zeros(nlcat, dtype=np.float64)

    def put(cat, value):
        frac[cat - 1] += value

    def mix_rest(primary_list, weight):
        put(int(rng.choice(primary_list)), weight)
        rest = 1.0 - weight
        if rest > 0:
            pool = ds["grass"] + ds["forest"] + ds["crop"]
            for cat in rng.choice(pool, size=2):
                put(int(cat), rest / 2.0)

    if kind in ("water", "coldwater"):
        put(water, 1.0 if rng.random() < 0.6 else 0.8)
        if frac.sum() < 1.0:
            put(int(rng.choice(ds["grass"])), 1.0 - frac.sum())
    elif kind == "seaice":
        put(water, 0.7)
        put(int(rng.choice(ds["snow"])), 0.3)
    elif kind == "partwater":
        w = 0.15 + 0.3 * rng.random()
        put(water, w)
        mix_rest(ds["crop"] + ds["forest"], 1.0 - w)
    elif kind == "crop":
        mix_rest(ds["crop"], 0.6 + 0.4 * rng.random())
    elif kind == "forest":
        mix_rest(ds["forest"], 0.7 + 0.3 * rng.random())
    elif kind == "bare":
        put(int(rng.choice(ds["barren"])), 1.0)
    elif kind == "urban":
        mix_rest(ds["urban"], 0.5 + 0.5 * rng.random())
    elif kind == "grass":
        mix_rest(ds["grass"], 0.8)
    elif kind == "wet":
        mix_rest(ds["wet"], 0.5 + 0.4 * rng.random())
    elif kind == "snowland":
        if rng.random() < 0.5:
            put(int(rng.choice(ds["snow"])), 1.0)
        else:
            mix_rest(ds["grass"] + ds["forest"], 1.0)
    elif kind in ("drysoil", "wetsoil"):
        mix_rest(ds["grass"] + ds["crop"], 0.9)
    else:
        raise ValueError(kind)
    frac /= frac.sum()
    return _f32(frac)


def soil_fractions(rng, index, kind):
    out = np.zeros(NSCAT, dtype=np.float64)
    if kind in ("water", "coldwater", "seaice"):
        out[13] = 1.0          # STATSGO 14, water
        return _f32(out)
    mix = SOIL_MIXES[index % len(SOIL_MIXES)]
    for cat, value in mix.items():
        out[cat - 1] = value
    return _f32(out)


def build_case(spec, seed):
    case_name, dsname, dt, smois_init, nudge, modis_veg, day, anal, curr0 = spec
    ds = DATASETS[dsname]
    nlcat = ds["nlcat"]
    rng = np.random.default_rng(seed)
    a2 = {field: np.zeros((NY, NX), np.float32)
          for field in IN_2D + INOUT_2D + OUT_2D}
    a3 = {field: np.zeros((NY, 2, NX), np.float32) for field in ATM_3D}
    soil = {field: np.zeros((NY, 2, NX), np.float32) for field in SOIL_3D}
    landusef = np.zeros((NY, nlcat, NX), np.float32)
    soilctop = np.zeros((NY, NSCAT, NX), np.float32)
    soilcbot = np.zeros((NY, NSCAT, NX), np.float32)
    isltyp = np.zeros((NY, NX), np.int32)

    for j in range(NY):
        for i in range(NX):
            index = j * NX + i
            kind = KINDS[(index * 5 + j) % len(KINDS)]
            landusef[j, :, i] = landuse_fractions(rng, ds, kind)
            soilcbot[j, :, i] = soil_fractions(rng, index, kind)
            soilctop[j, :, i] = soilcbot[j, :, i]
            isltyp[j, i] = 1 + (index % 16)
            wateris = kind in ("water", "coldwater", "seaice")

            # --- atmosphere, two levels ------------------------------------
            t1 = 270.0 + 35.0 * rng.random() if kind not in (
                "snowland", "seaice", "coldwater") else 250.0 + 20.0 * rng.random()
            psfc = 70000.0 + 33000.0 * rng.random()
            p1 = psfc - 300.0
            qsat = 0.622 * 611.2 * np.exp(17.67 * (t1 - 273.15) / (t1 - 29.65)) / p1
            rh = 0.2 + 0.85 * rng.random()
            qv1 = min(rh * qsat, 0.03)
            a3["t3d"][j, :, i] = [t1, t1 - 0.3 - 0.4 * rng.random()]
            a3["qv3d"][j, :, i] = [qv1, qv1 * (0.9 + 0.15 * rng.random())]
            a3["th3d"][j, :, i] = [t1 * (1.0e5 / p1) ** 0.2857,
                                   (t1 - 0.5) * (1.0e5 / (p1 - 600.0)) ** 0.2857]
            a3["rho"][j, :, i] = [p1 / (287.0 * t1), (p1 - 600.0) / (287.0 * (t1 - 0.5))]
            a3["dz8w"][j, :, i] = [18.0 + 50.0 * rng.random(),
                                   30.0 + 60.0 * rng.random()]
            a3["u3d"][j, :, i] = [10.0 * rng.standard_normal()] * 2
            a3["v3d"][j, :, i] = [10.0 * rng.standard_normal()] * 2

            a2["psfc"][j, i] = psfc
            if day:
                a2["gsw"][j, i] = 50.0 + 800.0 * rng.random()
            else:
                a2["gsw"][j, i] = 0.0 if rng.random() < 0.8 else 1.0e-6
            a2["glw"][j, i] = 180.0 + 270.0 * rng.random()
            rain = rng.random()
            a2["rainbl"][j, i] = (0.0 if rain < 0.55 else
                                  3.0 * rng.random() if rain < 0.9 else 15.0)
            a2["emiss"][j, i] = 0.88 + 0.11 * rng.random()
            a2["albbck"][j, i] = 0.1 + 0.2 * rng.random()
            vegfra = 100.0 * rng.random()
            shdmax = 0.0 if rng.random() < 0.12 else max(vegfra, 100.0 * rng.random())
            a2["shdmax"][j, i] = shdmax
            a2["shdmin"][j, i] = min(vegfra, shdmax) * rng.random()
            a2["vegfra"][j, i] = vegfra if rng.random() < 0.9 else 0.0
            a2["pblh"][j, i] = 40.0 + 2500.0 * rng.random()
            r = rng.random()
            if r < 0.1:
                rmol = 0.0
            elif r < 0.55:
                rmol = -0.2 * rng.random() ** 2
            else:
                rmol = 0.3 * rng.random() ** 2
            if r > 0.97:
                rmol = 0.5     # |1/RMOL| < Z/2: the stable PSIH > 1 arms
            a2["rmol"][j, i] = rmol
            a2["snowncv"][j, i] = 0.0 if rng.random() < 0.8 else rng.random()
            a2["ust"][j, i] = (0.003 if rng.random() < 0.08
                               else 0.02 + 0.8 * rng.random())
            a2["mavail"][j, i] = 0.05 + 0.95 * rng.random()
            if kind == "coldwater":
                a2["sst"][j, i] = 266.0 + 4.0 * rng.random()
            elif kind == "seaice":
                a2["sst"][j, i] = 271.0
            else:
                a2["sst"][j, i] = 272.0 + 30.0 * rng.random()
            a2["t2_ndg_old"][j, i] = t1 + 3.0 * rng.standard_normal()
            a2["t2_ndg_new"][j, i] = t1 + 3.0 * rng.standard_normal()
            a2["q2_ndg_old"][j, i] = max(qv1 * (0.7 + 0.6 * rng.random()), 1.0e-5)
            a2["q2_ndg_new"][j, i] = max(qv1 * (0.7 + 0.6 * rng.random()), 1.0e-5)
            snowy = kind in ("snowland", "seaice") or rng.random() < 0.05
            sn = (0.0 if not snowy else
                  0.3 + 150.0 * rng.random() ** 2)
            a2["sn_ndg_old"][j, i] = sn * (0.8 + 0.4 * rng.random())
            a2["sn_ndg_new"][j, i] = sn * (0.8 + 0.4 * rng.random())

            # --- carried / inout ------------------------------------------
            a2["ra"][j, i] = 50.0 * rng.random()
            a2["rs"][j, i] = 1234.5          # left alone where QFLUX skips it
            a2["lai"][j, i] = 6.0 * rng.random()
            if rng.random() < 0.05:
                a2["lai"][j, i] = 0.0
            a2["znt"][j, i] = 0.5 * rng.random()
            a2["qsfc"][j, i] = qv1
            a2["capg"][j, i] = 2.0e5 * rng.random()
            a2["canwat"][j, i] = (0.0 if rng.random() < 0.4
                                  else 0.6 * rng.random())
            a2["qfx"][j, i] = 1.0e-4 * rng.standard_normal()
            a2["hfx"][j, i] = 100.0 * rng.standard_normal()
            a2["lh"][j, i] = 100.0 * rng.standard_normal()
            a2["psih"][j, i] = rng.standard_normal()
            a2["vegf_px"][j, i] = rng.random()
            a2["snow"][j, i] = sn
            a2["snoalb"][j, i] = 0.6
            a2["snowh"][j, i] = sn * 0.0022
            a2["snowc"][j, i] = 1.0 if sn > 0 else 0.0
            a2["albedo"][j, i] = 0.1 + 0.3 * rng.random()
            a2["xland"][j, i] = 2.0 if wateris else 1.0
            a2["xice"][j, i] = (1.0 if kind == "seaice" else
                                0.6 if (kind == "snowland" and rng.random() < 0.15)
                                else 0.0)
            urbanish = kind == "urban" or rng.random() < 0.1
            a2["imperv"][j, i] = (100.0 * rng.random()) if urbanish else 0.0
            a2["canfra"][j, i] = (100.0 * rng.random()
                                  if (urbanish or kind == "forest") else 0.0)
            for field in OUT_2D:
                a2[field][j, i] = OUT_SENTINEL

            # --- soil -------------------------------------------------------
            if wateris:
                ts = a2["sst"][j, i]
                soil["tslb"][j, :, i] = [ts, ts]
                soil["smois"][j, :, i] = [1.0, 1.0]
            else:
                tg = t1 + 8.0 * rng.standard_normal()
                soil["tslb"][j, :, i] = [tg, 268.0 + 30.0 * rng.random()]
                if kind == "drysoil":
                    sm = [0.02 + 0.06 * rng.random(), 0.04 + 0.06 * rng.random()]
                elif kind == "wetsoil":
                    sm = [0.38 + 0.1 * rng.random(), 0.36 + 0.12 * rng.random()]
                else:
                    sm = [0.05 + 0.4 * rng.random(), 0.05 + 0.4 * rng.random()]
                soil["smois"][j, :, i] = sm

    if case_name.endswith("-defect"):
        for (jj, ii) in DEFECT_COLUMNS:
            landusef[jj, :, ii] = 0.0
            landusef[jj, 20, ii] = 1.0          # MODIS category 21
            soilcbot[jj, :, ii] = soil_fractions(rng, 3, "crop")
            a2["xland"][jj, ii] = 1.0
            a2["xice"][jj, ii] = 0.0
            soil["smois"][jj, :, ii] = [0.3, 0.3]
            soil["tslb"][jj, :, ii] = [290.0, 288.0]

    # A stable column with RMOL = 0 already sits in the mix; force one exact
    # dew column (QSS < QV1): very cold ground under moist warm air.
    j, i = 4, 6
    soil["tslb"][j, 0, i] = a3["t3d"][j, 0, i] - 15.0
    a3["qv3d"][j, 0, i] = 0.012

    header_i = np.zeros(16, np.int32)
    header_i[:10] = [NX, NY, nlcat, NSCAT, NSTEPS, smois_init, nudge,
                     modis_veg, anal, 1]
    header_f = np.zeros(8, np.float32)
    dzs = (0.01, 0.99)
    header_f[:7] = [dt, dt, curr0, dzs[0], dzs[1], 0.5 * dzs[0],
                    dzs[0] + 0.5 * dzs[1]]
    return dict(name=case_name, dataset=dsname, header_i=header_i,
                header_f=header_f, a2=a2, a3=a3, soil=soil,
                landusef=landusef, soilctop=soilctop, soilcbot=soilcbot,
                isltyp=isltyp)


def write_case(case, path: Path) -> None:
    with open(path, "wb") as f:
        f.write(case["header_i"].astype("<i4").tobytes())
        f.write(case["header_f"].astype("<f4").tobytes())
        for name in IN_2D + INOUT_2D + OUT_2D:
            f.write(case["a2"][name].astype("<f4").tobytes())
        f.write(case["isltyp"].astype("<i4").tobytes())
        for name in ATM_3D:
            f.write(case["a3"][name].astype("<f4").tobytes())
        for name in SOIL_3D:
            f.write(case["soil"][name].astype("<f4").tobytes())
        f.write(case["landusef"].astype("<f4").tobytes())
        f.write(case["soilctop"].astype("<f4").tobytes())
        f.write(case["soilcbot"].astype("<f4").tobytes())


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for seed, spec in enumerate(CASES):
        case = build_case(spec, 20261008 + seed)
        path = out / f"case-{case['name']}.bin"
        write_case(case, path)
        index.append({"name": case["name"], "dataset": case["dataset"],
                      "file": path.name,
                      "header_i": case["header_i"].tolist(),
                      "header_f": [float(x) for x in case["header_f"]]})
    (out / "cases-index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {len(index)} cases to {out}")


if __name__ == "__main__":
    main()
