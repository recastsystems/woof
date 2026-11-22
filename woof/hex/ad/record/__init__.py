# SPDX-License-Identifier: Apache-2.0
"""woof.hex.ad.record: record a hex trajectory and seal it into a replayable plan package.

The differentiable engine replays the production forward from a recording of it: the forward runs unchanged under a
CUPTI driver-API recorder (twice, recordings A and B), the sealer explains every launch, copy and upload of A against
B by a rule, a source literal, an immutable input or a proved pair, and the plan package it writes is what the replay
runner, the tangent and the adjoint run. See README.md beside this file. Entry point: python -m woof.hex.ad.record.
"""
from .env import DATA, NATIVE, TOOLS  # noqa: F401
