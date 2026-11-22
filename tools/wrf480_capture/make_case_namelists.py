import json, re, sys
from pathlib import Path
C = Path(sys.argv[1]); combos = json.loads(Path(sys.argv[2]).read_text())["combos"]
base = (C/"base.namelist.input").read_text()
def fmt(v):
    if isinstance(v, bool): return ".true." if v else ".false."
    if isinstance(v, float): return repr(v)
    return str(v)
def set_key(text, sec, key, val):
    pat = re.compile(r"(&%s\b.*?)(\n/)" % sec, re.S)
    m = pat.search(text); body = m.group(1)
    line = re.compile(r"^ %s\s*=.*$" % re.escape(key), re.M)
    new = " %-28s = %s," % (key, fmt(val))
    body = line.sub(new, body) if line.search(body) else body + "\n" + new
    return text[:m.start()] + body + m.group(2) + text[m.end():]
for cid in sys.argv[3:]:
    c = next(x for x in combos if x["id"] == cid)
    t = base
    # drop any physics/dynamics key the combo does not set, so the combo is the whole authority
    for sec in ("physics", "dynamics"):
        m = re.search(r"&%s\b(.*?)\n/" % sec, t, re.S)
        keep = [l for l in m.group(1).split("\n") if not l.strip() or l.split("=")[0].strip() in c["namelist"][sec]]
        t = t[:m.start(1)] + "\n".join(keep) + t[m.end(1):]
    for sec, kv in c["namelist"].items():
        for k, v in kv.items(): t = set_key(t, sec, k, v)
    t = set_key(t, "time_control", "run_seconds", 0); t = set_key(t, "time_control", "run_hours", 1)
    t = set_key(t, "time_control", "history_interval_s", 15)
    (C/cid).mkdir(exist_ok=True); (C/cid/"namelist.input.461").write_text(t)
    print(cid, "written")
