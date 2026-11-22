"""Apply afgl_repairs.json to the parsed AFGL rows, re-check, write profiles.json."""
import json, math, sys
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from afgl_parse import ALT
parsed = json.load(open(sys.argv[1])); rep = json.load(open(sys.argv[2]))
out = {}
for name in ("mls", "mlw", "sas", "saw"):
    rows = {float(z): [r["p"], r["T"], *r["gas"]] for z, r in parsed[name].items()}
    for z, (vals, why) in rep[name].items():
        rows[float(z)] = vals
    assert sorted(rows) == ALT, (name, [z for z in ALT if z not in rows])
    prof = [rows[z] for z in ALT]
    for i in range(1, 50):
        assert prof[i][0] < prof[i-1][0], (name, ALT[i])
        # hydrostatic scale height check: H = R T / g within 25 %
        H = (ALT[i]-ALT[i-1])*1e3/math.log(prof[i-1][0]/prof[i][0])
        Tm = 0.5*(prof[i][1]+prof[i-1][1]); Hs = 287.05*Tm/9.80665
        assert abs(H/Hs-1) < (0.04 if ALT[i] <= 50 else 0.07 if ALT[i] <= 80 else 0.25), (name, ALT[i], H, Hs)
    for i in range(1, 49):
        assert abs(prof[i][1]-0.5*(prof[i-1][1]+prof[i+1][1])) < 13, (name, ALT[i])
        for k in (2, 3):
            mid = 0.5*(math.log(prof[i-1][k])+math.log(prof[i+1][k]))
            assert ALT[i] > 70 or abs(math.log(prof[i][k])-mid) < 0.9, (name, ALT[i], k, prof[i])
    out[name] = dict(alt_km=ALT, p_mb=[r[0] for r in prof], T_K=[r[1] for r in prof],
                     h2o_ppmv=[r[2] for r in prof], o3_ppmv=[r[3] for r in prof],
                     n2o_ppmv=[r[4] for r in prof], co_ppmv=[r[5] for r in prof], ch4_ppmv=[r[6] for r in prof])
    # column water (cm) and ozone (DU) for the provenance note
    pw = o3 = 0.
    for i in range(49):
        dp = (prof[i][0]-prof[i+1][0])*100
        q = 0.5*(prof[i][2]+prof[i+1][2])*1e-6*18.015/28.964
        pw += q*dp/9.80665/10  # kg/m2 -> cm
        o3 += 0.5*(prof[i][3]+prof[i+1][3])*1e-6*dp/9.80665/28.964e-3*6.022e23/2.687e20
    out[name]["column_water_cm"] = pw; out[name]["column_ozone_du"] = o3
    print(name, "PW %.3f cm  O3 %.0f DU  Ts %.1f  ps %.0f" % (pw, o3, prof[0][1], prof[0][0]))
json.dump(out, open(sys.argv[3], "w"), indent=1)
