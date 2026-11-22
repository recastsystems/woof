#!/usr/bin/env python3
"""Observe a wider, still incomplete, live ownership graph during forecast.

Each boundary is walked independently. Numeric scalar hashes and array
descriptors check owners only; they are not later values for native replay.
Held stochastic objects and sensitive-named attributes are never traversed.
"""
# SPDX-License-Identifier: Apache-2.0
import sys
import observe_callbacks_v2 as callbacks
import observe_cupti_outer as base
import observe_host_asarray as host
import observe_launch_sites as launches
import observe_ruc_pointers as ruc


original_discover = callbacks.discover


def deep_discover(roots, cp, metadata):
    return original_discover(roots, cp, metadata, max_nodes=100000, max_depth=24,
                             include_scalar_hashes=True)


if __name__ == '__main__':
    callbacks.discover = deep_discover
    host.source_stack = ruc.source_stack_with_radiation_counts
    base.Observer = launches.LaunchSiteObserver
    base.boundary_trace = ruc.boundary_ruc_and_existing_trace
    sys.exit(base.main())
