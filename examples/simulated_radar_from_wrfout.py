"""Write radar volumes from saved atmospheric history without running a forecast."""
import argparse

from woof.simulated_radar import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", nargs="+")
    parser.add_argument("--site", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scan-strategy", default="vcp212", choices=("vcp212", "low_tilts"))
    args = parser.parse_args()
    result = simulate(args.history, site=args.site, outdir=args.outdir,
                      scan_strategy=args.scan_strategy, formats=("level2", "cfradial1", "odim"))
    for volume in result.volumes:
        print(volume.volume_time.isoformat(), volume.site.id, volume.scan_strategy)
        for format, path in volume.paths.items():
            print(format, path)


# In memory, with an already initialized and diagnosed georeferenced domain:
# result = simulate(domain, site="KTLX", outdir="products",
#                   valid_time="2026-01-01T00:00:00Z")
# A bare state also works: simulate(state, cfg=run_config, grid=grid, ...).

if __name__ == "__main__":
    main()
