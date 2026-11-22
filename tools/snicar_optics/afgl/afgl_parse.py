"""Parse AFGL-TR-86-0110 Table 1 (Anderson et al. 1986, US government report,
public domain) from the archive.org text layer of DTIC ADA175173.

Columns: alt km, p mb, T K, n cm-3, H2O O3 N2O CO CH4 ppmv. The text layer is
OCR, so each row is parsed structurally (mantissa widths fixed by the report:
4 significant digits for p and n, 3 for gases) and then checked physically:
ideal gas n = p/(kT) to 0.6 %, the fixed 50-level altitude grid, and
log-smoothness of every gas. Rows that fail are reported; repairs live in
afgl_repairs.json with a reason each, read against the page image.
"""
import re, json, sys, math
ALT = [float(z) for z in range(0, 26)] + [27.5+2.5*i for i in range(10)] + [55.+5*i for i in range(14)]
SPANS = {"tropical": (329, 392), "mls": (395, 462), "mlw": (467, 541), "sas": (541, 620), "saw": (624, 698), "us76": (698, 813)}
TR = str.maketrans({"O": "0", "o": "0", "S": "5", "s": "5", "!": "1", "l": "1", "I": "1", ",": ".", "J": "3", "B": "8", "Z": "2", "C": "E", "F": "E", "P": "E", "a": "E", "d": "E", "*": "+", "b": "6", "i": "1", ":": ".", "A": ""})
SEP = r"(?:E5|5E|1E|E|5|1)?"
P4 = r"(\d\.\d{3})" + SEP + r"([+-]\d\d)"
G3 = r"(\d\.\d{2})" + SEP + r"([+-]\d\d)"
ROW = re.compile(r"^(\d{1,3}\.\d\d)" + P4 + r"(\d{3}\.\d)" + P4 + G3*5)
K = 1.380649e-23

def clean(line):
    t = line.translate(TR)
    t = re.sub(r"[^0-9.E+\-]", "", t).replace("--", "-")
    return t.lstrip(".-+")

def parse(text, lo, hi):
    rows = {}
    bad = []
    for line in text[lo:hi]:
        t = clean(line)
        m = ROW.match(t)
        if not m:
            if re.match(r"^\d{1,3}\.\d\d\d", t):
                bad.append(line.strip())
            continue
        g = m.groups()
        alt = float(g[0])
        if alt not in ALT:
            bad.append("ALT " + line.strip()); continue
        p = float(g[1] + "E" + g[2]); T = float(g[3]); n = float(g[4] + "E" + g[5])
        gas = [float(g[6+2*i] + "E" + g[7+2*i]) for i in range(5)]
        rows[alt] = dict(p=p, T=T, n=n, gas=gas, raw=line.strip())
    return rows, bad

def check(rows):
    flags = []
    for z, r in rows.items():
        n = r["p"]*100/(K*r["T"])*1e-6
        if abs(n/r["n"]-1) > 0.006:
            flags.append((z, "idealgas %.4f" % (n/r["n"]), r["raw"]))
    zs = sorted(rows)
    for i in range(1, len(zs)-1):
        a, b, c = rows[zs[i-1]], rows[zs[i]], rows[zs[i+1]]
        if not (a["p"] > b["p"] > c["p"]):
            flags.append((zs[i], "p not monotone", b["raw"]))
        if abs(b["T"] - 0.5*(a["T"]+c["T"])) > 12:
            flags.append((zs[i], "T kink", b["raw"]))
        for k in range(5):
            mid = 0.5*(math.log(a["gas"][k]) + math.log(c["gas"][k]))
            if abs(math.log(b["gas"][k]) - mid) > 0.9:
                flags.append((zs[i], "gas%d kink" % k, b["raw"]))
    return flags

if __name__ == "__main__":
    text = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
    out = {}
    for name in ("mls", "mlw", "sas", "saw"):
        rows, bad = parse(text, *SPANS[name])
        missing = [z for z in ALT if z not in rows]
        flags = check(rows)
        print("==", name, "rows", len(rows), "missing", missing)
        for b in bad: print("  UNPARSED", b)
        for f in flags: print("  FLAG", f)
        out[name] = {str(z): {k: v for k, v in r.items() if k != "raw"} for z, r in rows.items()}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
