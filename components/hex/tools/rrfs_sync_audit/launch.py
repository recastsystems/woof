"""Run the real CLI with an untimed CUPTI/Python audit. Use the GPU pool."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import Audit


def main():
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
