RRFS driver glue proof tool. The production helpers live in driver_rrfs.py.

Run the real hrrrv5 forecast with `LAUNCHER=/absolute/path/ab_launch.py`,
`DRIVER_BASE=/absolute/path/base/tree`, `GLUE_AB_OUT=/absolute/path/receipt.jsonl`.
The base is 7876465. Steps default to 1,5,6,12. Use a 2.4 h simulation to reach
step 12 with the match deck's 720 s step; strict uses the strict table packet.
Use the project's GPU pool and per-lane recipe, with a 6000 MiB replay lease.
The measured replay fits a 4500 MiB CuPy allocator limit.

`DRIVER_AB=0` installs the three integration hooks without A/B, timing or
profiling. This permits a smoke while another owner carries the integration
changes in resident_rrfs.py and runtime_v842.py. For a combined tree with those
hooks already present, the previous arms must instead be loaded from the base
resident/runtime modules; this tool is for the isolated driver lane.

Each comparison must have `different=0`, empty differing and one-sided array
lists, and every plan must have `race_free=true`. Returned tendency buffers
and the mutating post-transport state are included. Reset is checked at the
step5 output boundary. The new state continues the live forecast.

Step12 timing uses nine restored-input old/new pairs, with synchronization
outside the host interval. CUDA-event times include launch gaps and shared
card contention. CUPTI runs afterward in separate restored-input arms and
its timing is never reported. The census includes memset calls. Its bundled
observer was supplied through the shared INBOX by the host-read lane.

`compare_all.py BASE_RUN TIP_RUN RECEIPT_JSON` checks the full file and variable
sets and all numeric words and text bytes in history, diag and restart files.
A missing file, field, shape or dtype fails the receipt, as does any changed word.
