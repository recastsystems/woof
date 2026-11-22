# Differentiable engine examples

These files travel in the wheel. `woof hex examples` prints the folder they are installed in, and
`woof hex examples --copy DIR` copies them to a folder of yours.

Each script is meant to run end to end from a plain install (`pip install "recast-woof[ad]"` in a Python 3.14 environment of its own, not on top of `[gpu]`) and
fetches its own inputs, once, each checked by SHA-256. `run-32gb.sh` ran whole that way on 0.6.0's published wheel
and on a pre-release build of 0.6.1 (one RTX 5090, empty home). `run-96gb.sh` has not been run whole: each of its
commands was read on a 96 GB card on builds of 0.6.0. `expected.json` holds the numbers to expect and the run each
was read from. All need an NVIDIA card of compute capability 12.0 (GeForce RTX 50 series, RTX PRO Blackwell); on another card
every command stops before any download and says why.

| script | card | host memory | what it runs | about |
|---|---|---|---|---|
| `run-32gb.sh [OUT]` | 32 GB (28,700 MiB free) | 47,104 MiB | the 12 km example: `adjoint-test` (PASS on the bump and the gradient's descent), `sensitivity`, then `own_measure.py` | 12 min |
| `run-96gb.sh [OUT]` | 96 GB (96,600 MiB free) | 258,057 MiB | the 120 km install check (all four commands) and the 3 km storm `sensitivity` | 25 to 30 min |
| `own_measure.py [OUT]` | 32 GB | 47,104 MiB | two gradients of your own measures from Python (`woof.hex.ad_session.SensitivitySession`) | 2 to 3 min |

Each script prints each command's exit status and ends with the highest of them (0 only when every command ended 0).

What the output means: `docs/differentiable/FIRST-HOUR.md`. What is not yet right: `docs/differentiable/LIMITS.md`.
