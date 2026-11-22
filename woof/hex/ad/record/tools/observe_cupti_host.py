#!/usr/bin/env python3
"""Add a partial host-state inventory to the ordinary outer-boundary observer."""
# SPDX-License-Identifier: Apache-2.0
import gzip
import json
import sys
from host_inventory import HostInventory
import observe_cupti_outer as base


class HostObserver(base.Observer):
    def __init__(self, root):
        super().__init__(root)
        self.host_inventory = HostInventory(workers=4)
        self.host_output = gzip.open(root / 'host-state.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)

    def outer_boundary(self, frame, step, cp):
        super().outer_boundary(frame, step, cp)
        row = self.host_inventory.sample(frame, step)
        self.host_output.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
        self.host_output.flush()

    def close(self):
        try:
            self.host_inventory.close()
            self.host_output.close()
        finally:
            super().close()


if __name__ == '__main__':
    base.Observer = HostObserver
    sys.exit(base.main())
