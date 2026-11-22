# SPDX-License-Identifier: Apache-2.0
"""ad-record: the frame-state cross-check for a recording whose mesh or timestep is not the six-step x1.2562/1200 s
sample's (configs/binder/hrrrv5.frame-state.sample.jsonl): every step's snapshot from recording B's own frame
observation, so stage 3 checks A's snapshots against B's (the two-recording standard the rest of the seal uses).
Breakage: the x1.40962 pair (pk120-1e, dt 720 s) stopped at 'observed frame snapshot differs from ...sample.jsonl
at step 1' because the sample encodes the x1.2562 clock (dt 1200 s, 12:20:00).
Usage: make_frame_states.py SEALER_TOOL_DIR OBSERVER_B/frames.jsonl.gz OUT.jsonl  (SEALER_TOOL_DIR: woof/hex/ad/record/tools)"""
import json, sys
sys.path.insert(0, sys.argv[1])
from seal_phys_stage3 import frame_states_from_observation, jsonl   # noqa: E402
states = frame_states_from_observation(list(jsonl(sys.argv[2])))
with open(sys.argv[3], "w") as out:
    for step in sorted(states):
        out.write(json.dumps({"step": step, "snapshot_hex": states[step].hex()}) + "\n")
print(f"frame states: {len(states)} steps from {sys.argv[2]}")
