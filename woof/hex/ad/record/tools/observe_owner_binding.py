#!/usr/bin/env python3
"""Check same-run identities of mesh-scalar owners at calls and boundaries.

Object identifiers are association checks only. They are never replay inputs,
and neither they nor boundary samples establish a complete carried-state census.
"""
# SPDX-License-Identifier: Apache-2.0
import sys

import observe_callbacks_v2 as callbacks
import observe_cupti_outer as base
import observe_deep_census as deep
import observe_host_asarray as host
import observe_launch_sites as launches
import observe_ruc_pointers as ruc


original_sites = launches.sites
SOURCE_FILES = ('/hexcore/cuda_driver.py', '/hexcore/cuda_horizontal.py',
                '/hexcore/cuda_horizontal_v841.py')


def sites_with_owners(args, frame):
    result, excluded = original_sites(args, frame)
    index = 0
    while frame and index < len(result):
        path = frame.f_code.co_filename
        if 'stochastic' not in path.lower() and (('/hexcore/' in path and '/hexcore/ad/record/' not in path) or '/woof/' in path):
            row = result[index]
            if (row['path'], row['line'], row['function']) != (path, frame.f_lineno, frame.f_code.co_name):
                raise RuntimeError('source-frame observation changed during owner check')
            if path.endswith(SOURCE_FILES):
                owner = frame.f_locals.get('self')
                if owner is not None:
                    row['owner_object_id'] = id(owner)
            index += 1
        frame = frame.f_back
    if index != len(result):
        raise RuntimeError('source-frame observation ended during owner check')
    return result, excluded


def discover_with_owners(roots, cp, metadata):
    graph = deep.deep_discover(roots, cp, metadata)
    owners = {}
    dycore = roots.get('dycore')
    if dycore is not None:
        owners['dycore'] = id(dycore)
        try:
            attrs = object.__getattribute__(dycore, '__dict__')
        except AttributeError:
            attrs = None
        if type(attrs) is dict:
            horizontal = dict.get(attrs, 'horizontal')
            if horizontal is not None:
                owners['dycore.horizontal'] = id(horizontal)
    graph['owner_object_ids'] = owners
    graph['limitations'].append('Object identifiers are same-run checks, not native replay owners.')
    return graph


if __name__ == '__main__':
    launches.sites = sites_with_owners
    callbacks.discover = discover_with_owners
    host.source_stack = ruc.source_stack_with_radiation_counts
    base.Observer = launches.LaunchSiteObserver
    base.boundary_trace = ruc.boundary_ruc_and_existing_trace
    sys.exit(base.main())
