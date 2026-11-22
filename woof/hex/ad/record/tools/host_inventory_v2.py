#!/usr/bin/env python3
"""Add framework table observations without claiming a complete host census."""
# SPDX-License-Identifier: Apache-2.0
from host_inventory import HostInventory, _array


class HostInventoryV2(HostInventory):
    def sample(self, frame, step):
        result = super().sample(frame, step)
        native = frame.f_locals['session'].native
        jobs = []
        for name, value in sorted(native._tables.items()):
            if not isinstance(name, str) or not isinstance(value, tuple) or len(value) != 2:
                raise ValueError('unexpected framework table representation')
            label = 'framework.table.' + name
            array, lower = value
            jobs.append(self.executor.submit(_array, label, array, lower))
        result['items'].extend(job.result() for job in jobs)
        result['limitations'].append('Framework table hashes are observed, not replay inputs or ownership proofs.')
        return result
