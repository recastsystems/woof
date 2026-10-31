"""Run the real CLI with an untimed CUPTI/Python audit. Use the GPU pool."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import Audit


def main():
    from woof.hex.mpasphys.rrfs import sf_mynnsfc_v600 as sfc, resident_sfclay_rrfs as rs
    original = sfc.launch_slot

    def alternating(call):
        # The already-proven previous glue runs step 2; all other steps fuse.
        # Compare plain steps 2 and 3, excluding all first-use setup.
        rs.FUSED = call.itimestep != 2
        return original(call)

    sfc.launch_slot = alternating
    audit = Audit(os.environ['RRFS_AUDIT_CUPTI']).start()
    try:
        from woof.hex import cli
        a = cli.build_parser().parse_args(sys.argv[1:])
        cli._select_numerics(a)
        return int(a.handler(a))
    finally:
        audit.stop(os.environ['RRFS_AUDIT_OUT'])


if __name__ == '__main__':
    sys.exit(main())
