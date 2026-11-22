#!/usr/bin/env python3
"""Observe additional slotted owners in an ordinary forward forecast.

Hashes and addresses are checks only. No recorded host output is replayed.
The held stochastic implementation and sensitive attributes remain excluded.
"""
# SPDX-License-Identifier: Apache-2.0
import sys
import observe_owner_binding as owner


def slotted_discover(roots, cp, metadata):
    return owner.deep.original_discover(roots, cp, metadata, max_nodes=100000, max_depth=24,
        include_scalar_hashes=True, include_slots=True, include_numpy_scalars=True, include_slices=True)


if __name__ == '__main__':
    owner.deep.deep_discover = slotted_discover
    owner.launches.sites = owner.sites_with_owners
    owner.callbacks.discover = owner.discover_with_owners
    owner.host.source_stack = owner.ruc.source_stack_with_radiation_counts
    owner.base.Observer = owner.launches.LaunchSiteObserver
    owner.base.boundary_trace = owner.ruc.boundary_ruc_and_existing_trace
    sys.exit(owner.base.main())
