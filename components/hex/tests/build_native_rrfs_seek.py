"""Build the complete fork seek routine with original native calendar arithmetic.

MPAS-Model-rrfs/src/framework/mpas_io_streams.F:142-335. The shared builder
stubs only timestamp storage; all timestamps in this proof come from real files.
"""
import json
import os
from pathlib import Path
import sys

import build_native_stream_seek as shared
from build_native_rrfs_calendar import build as fork_calendar


def build(destination):
    os.environ['MPAS_REF'] = os.environ['RRFS_MPAS_REF']
    def calendar_with_flags(path):
        binary = fork_calendar(path)
        receipt = Path(path) / 'clock-build.json'
        clock = json.loads(receipt.read_text())
        # The fork getter has conditional native/external ESMF interfaces.
        # Preserve the exact preprocessing used by the original fork build.
        clock['compiler_flags'] += ' -cpp -DSINGLE_PRECISION'
        receipt.write_text(json.dumps(clock, indent=2) + '\n')
        return binary
    shared.build_clock = calendar_with_flags
    binary = shared.build(destination)
    receipt = Path(destination) / 'seek-build.json'
    authority = json.loads(receipt.read_text())
    authority.update(schema='r11-native-fork-stream-seek-build/v1',
                     model_commit='1d6e370e7b43a332074809f170532a216c2212fb',
                     source_root=os.environ['RRFS_MPAS_REF'],
                     native_abs_wrapper_source='src/framework/mpas_timekeeping.F:2084-2106')
    receipt.write_text(json.dumps(authority, indent=2) + '\n')
    return binary


if __name__ == '__main__':
    print(build(Path(sys.argv[1])))
