"""Build source-checked MYNN packet images outside the GPU lease.

Usage: build_packets.py BASE_PACKET_DIR OUTPUT_DIR fast|strict
All other physics packet entries retain their original read-only paths.
"""
import json
import sys
from pathlib import Path

from woof.hex import numerics
from woof.hex.mpasphys.resources_build import build_mynnedmf_image

source, out, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
numerics.set_mode(mode, source='mynnedmf replay packet build')
row = build_mynnedmf_image(out / 'cache', log=print, arch='sm_120')
packet = json.loads((source / f'packet-{mode}.json').read_text())
packet['bl_mynnedmf'] = row
(out / f'packet-{mode}.json').write_text(json.dumps(packet, indent=2)+'\n')
print(json.dumps({'numerics': mode, **row}), flush=True)
