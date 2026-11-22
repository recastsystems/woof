#!/usr/bin/env python3
"""L5 sheets: the products door's own rendered maps (pin | 550b5b4) side by side, plus an ASOS analysis chart.

Weather-field panels are the products door's PNGs, composed unchanged (PIL paste only). The ASOS time series is an
analysis chart (matplotlib allowed for non-weather-field charts).
Usage: sheets.py PIN_PRODUCTS NEW_PRODUCTS SCORES.json OUTDIR
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

pin, new, scores, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
PRODUCTS = ("cloud_cover_levels", "2m_temperature_10m_winds", "2m_dewpoint_10m_winds", "simulated_ir_satellite")
LEADS = (6, 10, 13, 16)
font = ImageFont.load_default()


def panel(root, product, lead):
    p = os.path.join(root, "refined-core", product, "2024-06-21",
                     f"hex8.1-3.8km_refined-core_{product}_f{lead:03d}.png")
    return Image.open(p).convert("RGB")


for product in PRODUCTS:
    rows = []
    for lead in LEADS:
        a, b = panel(pin, product, lead), panel(new, product, lead)
        w, h = a.size
        row = Image.new("RGB", (2 * w + 10, h + 28), "white")
        row.paste(a, (0, 28))
        row.paste(b, (w + 10, 28))
        d = ImageDraw.Draw(row)
        d.text((8, 6), f"PIN MMM a4baf7f (YSU pblmix on)  f{lead:02d}  valid 2024-06-21 {lead:02d}Z", fill="black", font=font)
        d.text((w + 18, 6), f"MMM 550b5b4 = WRF v4.8.0 (top-down cap)  f{lead:02d}", fill="black", font=font)
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(os.path.join(out, f"sheet-{product}.png"), optimize=True)
    print("sheet", product, sheet.size)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = json.load(open(scores))
arms = list(s["arms"])
fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
leads = list(range(1, 19))
for arm, style in zip(arms, ("-o", "-s")):
    ser = s["arms"][arm]["series"]
    ax[0].plot(leads, [ser[str(l)].get("low_model_frac") for l in leads], style, label=f"{arm} model")
    ax[1].plot(leads, [ser[str(l)].get("low_csi") for l in leads], style, label=arm)
    ax[2].plot(leads, [ser[str(l)].get("t2_bias") for l in leads], style, label=arm)
ser = s["arms"][arms[0]]["series"]
ax[0].plot(leads, [ser[str(l)].get("low_obs_frac") for l in leads], "k-", lw=2.5, label="ASOS observed")
ax[0].set_ylabel("fraction of stations\nceiling <= 1500 ft")
ax[1].set_ylabel("low-ceiling CSI")
ax[2].set_ylabel("T2 bias (K)")
ax[2].axhline(0, color="grey", lw=0.8)
ax[2].set_xlabel("forecast hour (init 2024-06-21 00Z; 07Z = midnight PDT)")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
ax[0].set_title(f"SF Bay marine stratus, hex 3.75 km mesoscale_reference, YSU top-down on; {len(s['stations'])} ASOS")
fig.tight_layout()
fig.savefig(os.path.join(out, "asos-timeseries.png"), dpi=110)
print("chart done")
