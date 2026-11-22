#!/usr/bin/env python3
"""Add table hashes to the ordinary callback and host-state observer."""
# SPDX-License-Identifier: Apache-2.0
import gzip
import json
import sys
import observe_callbacks as callbacks
import observe_cupti_outer as base
from host_inventory_v2 import HostInventoryV2
from object_graph_inventory import discover


class CallbackObserverV2(callbacks.CallbackObserver):
    def __init__(self, root):
        super().__init__(root)
        self.host_inventory.close()
        self.host_inventory = HostInventoryV2(workers=4)
        self.graph_output = gzip.open(root / 'object-graph.jsonl.gz', 'wt',
                                      encoding='utf-8', compresslevel=1)

    def outer_boundary(self, frame, step, cp):
        super().outer_boundary(frame, step, cp)
        roots = {name: frame.f_locals[name] for name in ('session', 'dycore', 'coupling')
                 if name in frame.f_locals}
        row = discover(roots, cp, self.array_metadata)
        row.update({'boundary': 'outer_after_step', 'step': int(step)})
        self.graph_output.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
        self.graph_output.flush()

    def close(self):
        try:
            self.graph_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    base.Observer = CallbackObserverV2
    sys.exit(base.main())
