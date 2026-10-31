#!/usr/bin/env python3
"""Extend f0 publication with byte-authenticated native sparse file copies.

The original publisher still selects, hashes, seals and READY-lists all files.
GNU cp represents existing zero byte runs as filesystem holes; no NetCDF data
is decoded, normalized or rewritten. The selected native tool is recorded.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


spec = importlib.util.spec_from_file_location('rrfs_sparse_base_publication',
                                            Path(__file__).parent.parent / 'publish_cache.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class NativeCopies:
    def __init__(self, binary):
        self.binary = binary

    def __getattr__(self, name):
        return getattr(shutil, name)

    def copy2(self, source, destination):
        subprocess.run([str(self.binary), '--reflink=never', '--sparse=always', '--preserve=mode,timestamps',
                        '--', str(source), str(destination)], check=True)
        if base.sha(Path(source)) != base.sha(Path(destination)):
            raise ValueError('native sparse copy changed file bytes')
        return destination


def publish(source, destination, patterns, ready, description):
    binary = Path(shutil.which('cp')).resolve(strict=True)
    source = source.resolve(strict=True)
    recipe = source / 'SPARSE-COPY-RECIPE.json'
    value = dict(schema='mpas.rrfs-native-sparse-copy/v1', cp_sha256=base.sha(binary),
                 arguments=['--reflink=never', '--sparse=always', '--preserve=mode,timestamps'],
                 every_copied_file_sha256_checked=True, weather_arrays_transformed=False)
    if recipe.exists() and json.loads(recipe.read_text()) != value:
        raise ValueError('native sparse copy recipe changed')
    recipe.write_text(json.dumps(value, indent=2) + '\n')
    # This separately loaded publisher module receives its copy adapter;
    # neither f0's source nor the shared shutil module is changed.
    base.shutil = NativeCopies(binary)
    return base.publish(source, destination, [*patterns, recipe.name], ready, description)
