"""Plot Rust receipt statistics, never weather-field maps."""
import argparse
from datetime import datetime
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records=[r for r in json.loads(args.receipt.read_text())["records"]
             if r["target"]=="SST" and r.get("mean") is not None]
    records.sort(key=lambda r:r["valid_time"])
    times=[datetime.fromisoformat(r["valid_time"].replace("Z","+00:00")) for r in records]
    mean=[r["mean"] for r in records]
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    ax.errorbar(times,mean,yerr=[[r["mean"]-r["min"] for r in records],
                               [r["max"]-r["mean"] for r in records]],
                color="#225b77",fmt="o-",capsize=5)
    ax.set(title="Daily SST analyses on coastal water cells",ylabel="SST (K)",xlabel="Source valid time (UTC)")
    ax.grid(axis="y",alpha=.2)
    ax.margins(x=.15)
    ax.text(.02,.97,"Mean and full cell range. Ingest check only.",transform=ax.transAxes,va="top",fontsize=9)
    for t,r in zip(times,records):
        ax.annotate(f'{r["valid_cells"]} valid cells',(t,r["mean"]),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
    fig.autofmt_xdate()
    fig.savefig(args.output,dpi=160)

if __name__=="__main__":main()
