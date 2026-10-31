"""Pinned RRFS-MPAS RUC land, sea ice and surface diagnostic CUDA arithmetic.

Reference: 1d6e370e, NOAA-GSL/RUCLSM 8c9f1a8. Single precision and
one-based Fortran views; no numerical host fallback. The native MPAS driver
owns packing, persistent pools and surface coupling. Standalone ABI bindings
are available for the complete land/ice/diagnostics boundary.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics
from functools import lru_cache
from importlib.resources import files
import hashlib
import os
import json
from pathlib import Path
import numpy as np

SCHEME = "sf_ruc"
BACKEND = "hex-cuda"
FORK_PIN = "1d6e370e7b43a332074809f170532a216c2212fb"
RUC_PIN = "8c9f1a8bb2cfb5fb8d115440fced566361fe0f20"
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
WORK_WORDS = 16384
TABLE_WORDS = 955
# Words of Arena scratch a launch actually addresses per cell, read from CUDA_SOURCE (arena_words).
# THE BREAKAGE THIS PREVENTS: every forecast RUC launch reserved WORK_WORDS (16,384) float32 words, 64 KiB, per
# cell for every cell at once: 2.6 GB on x1.40962 (37 % of the hrrrv5 per-cell device cost), 10.7 GB on
# x1.163842, where one 32 GB RTX 5090 ran out of memory at this call (V5090, 2026-10-28), and 153 GB over
# conus3km.  The pinned source needs at most 623 words per cell for land (nsl 9, nlcat 21, nscat 16) and 190
# for sea ice (the saturation table is shared, ruc_tbq_init).  Rows past the bound are never addressed, so no
# output word changes.
ARENA_COLUMN_EXTENTS = dict(ids=1, ide=1, jds=1, jde=1, ims=1, ime=1, jms=1, jme=1,
                            its=1, ite=1, jts=1, jte=1, ncells=1)
ARENA_CELL_DIMS = frozenset(('ims:ime', 'its:ite', 'ncells', '1:ncells', 'jms:jme', 'jts:jte', '1'))

ARGUMENTS = {'land_ruc_land': [{'name': 'spp_lsm', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'pattern_spp_lsm', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': None, 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'field_sf', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': None, 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'dt', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ktau', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'nsl', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'restart', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'cycling', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'lakemodel', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'lakemask', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'graupelncv', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowncv', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rainncv', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'zs', 'type': 'float', 'dim': ['1:nsl'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rainbl', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snow', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowh', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'frzfrac', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'frpcpn', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'rhosnf', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'precipfr', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'z3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'p8w', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 't3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qv3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qc3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rho3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'glw', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'gsw', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'emiss', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chklowq', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chs', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'flqc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'flhc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'mavail', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'canwat', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'vegfra', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'alb', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'znt', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'z0', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snoalb', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'albbck', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'lai', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'mminlu', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': True}, {'name': 'landusef', 'type': 'float', 'dim': ['ims:ime', '1:nlcat', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'nlcat', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'mosaic_lu', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'mosaic_soil', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'soilctop', 'type': 'float', 'dim': ['ims:ime', '1:nscat', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'nscat', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qsfc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qsg', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qvg', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qcg', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'dew', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'soilt1', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tsnav', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tbot', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ivgtyp', 'type': 'int', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'isltyp', 'type': 'int', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xland', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'iswater', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'isice', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice_threshold', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cp', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rovcp', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'g0', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'lv', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'stbolt', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'soilmois', 'type': 'float', 'dim': ['ims:ime', '1:nsl', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sh2o', 'type': 'float', 'dim': ['ims:ime', '1:nsl', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'smavail', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'smmax', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tso', 'type': 'float', 'dim': ['ims:ime', '1:nsl', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'soilt', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'hfx', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qfx', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'lh', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfcrunoff', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'udrunoff', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'acrunoff', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfcexc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfcevp', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'grdflx', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowfallac', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'acsnow', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snom', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'smfr3d', 'type': 'float', 'dim': ['ims:ime', '1:nsl', 'jms:jme'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'keepfr3dflag', 'type': 'float', 'dim': ['ims:ime', '1:nsl', 'jms:jme'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'myjpbl', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'shdmin', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'shdmax', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rdlai2d', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'globalcells', 'type': 'int', 'dim': ['ims:ime'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ids', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ide', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jds', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jde', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kds', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kde', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ims', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ime', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jms', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jme', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kms', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kme', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'its', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ite', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jts', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jte', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kts', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kte', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}], 'ice_ruc_ice': [{'name': 'spp_lsm', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'pattern_spp_lsm', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': None, 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'field_sf', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': None, 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'dt', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ktau', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'nsl', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'restart', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'cycling', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'graupelncv', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowncv', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rainncv', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': True, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'zs', 'type': 'float', 'dim': ['1:nsl'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rainbl', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'frzfrac', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'frpcpn', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'rhosnf', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'precipfr', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'z3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'p8w', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 't3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qv3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qc3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rho3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'glw', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'gsw', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'emiss', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chklowq', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chs', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'flqc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'flhc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'alb', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'znt', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'z0', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snoalb', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'albbck', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'mminlu', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': True}, {'name': 'tbot', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ivgtyp', 'type': 'int', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'isltyp', 'type': 'int', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xland', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'myjpbl', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'iswater', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'isice', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice_threshold', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cp', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rovcp', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'g0', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'lv', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'stbolt', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tso', 'type': 'float', 'dim': ['ims:ime', '1:nsl', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'soilt', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'soilt1', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tsnav', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snow', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowh', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qsfc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qsg', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qvg', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qcg', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'hfx', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qfx', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'lh', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'dew', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfcrunoff', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'acrunoff', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfcevp', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'grdflx', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowfallac', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'out', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'acsnow', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snom', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'globalcells', 'type': 'int', 'dim': ['ims:ime'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ids', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ide', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jds', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jde', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kds', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kde', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ims', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ime', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jms', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jme', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kms', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kme', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'its', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ite', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jts', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jte', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kts', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kte', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}], 'diags_sfcdiags_ruclsm': [{'name': 'hfx', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qfx', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tsk', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qsfc', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cqs', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cqs2', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chs', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chs2', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 't2', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'th2', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'q2', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 't3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qv3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'dz', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rho3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'p3d', 'type': 'float', 'dim': ['ims:ime', 'kms:kme', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'psfc2d', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cp', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'r_d', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rovcp', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'globalcells', 'type': 'int', 'dim': ['ims:ime'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ids', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ide', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jds', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jde', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kds', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kde', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ims', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ime', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jms', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jme', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kms', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kme', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'its', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ite', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jts', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jte', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kts', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'kte', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}], 'init_rucinit': [{'name': 'restart', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'cycling', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'ncells', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'nsoillevels', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ivgtyp', 'type': 'int', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'isltyp', 'type': 'int', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tslb', 'type': 'float', 'dim': ['nsoillevels', 'ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'smois', 'type': 'float', 'dim': ['nsoillevels', 'ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'smfr3d', 'type': 'float', 'dim': ['nsoillevels', 'ncells'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sh2o', 'type': 'float', 'dim': ['nsoillevels', 'ncells'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'mavail', 'type': 'float', 'dim': ['ncells'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'znt', 'type': 'float', 'dim': ['ncells'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snow', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'snowh', 'type': 'float', 'dim': ['ncells'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}], 'drivers_seaice_prepare': [{'name': 'ncells', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice_threshold', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfc_albedo', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfc_emiss', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfc_albedo_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfc_emiss_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}], 'drivers_seaice_finish': [{'name': 'ncells', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice_threshold', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'config_frac_seaice', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'mynn', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'xice_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'flhc_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'flhc_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'flqc_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'flqc_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cqs_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cqs_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cpm_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cpm_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chs_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chs_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chs2_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'chs2_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cqs2_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'cqs2_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'hfx_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'hfx_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'lh_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'lh_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qfx_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qfx_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qgh_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qgh_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qsfc_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'qsfc_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tsk_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tsk_sea', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfc_albedo_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfc_emiss_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}], 'drivers_correct_tsk_over_seaice': [{'name': 'ims', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ime', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jms', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jme', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'its', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'ite', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jts', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'jte', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice_thresh', 'type': 'float', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'xice', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tsk', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tsk_sea', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'tsk_ice', 'type': 'float', 'dim': ['ims:ime', 'jms:jme'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}], 'drivers_driver_forcing': [{'name': 'ncells', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'micro_on', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'conv_on', 'type': 'int', 'dim': [], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': True, 'character': False}, {'name': 'sr', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rainncv', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'raincv', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'gsw', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sfc_albedo', 'type': 'float', 'dim': ['ncells'], 'intent': 'in', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'sr_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'rainbl_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, {'name': 'swdown_p', 'type': 'float', 'dim': ['ncells', '1'], 'intent': 'inout', 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}]}
TABLE_LAYOUT = {'lucats': {'offset': 0, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'bare': {'offset': 1, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'natural': {'offset': 2, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'crop': {'offset': 3, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'urban': {'offset': 4, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'lutype': {'offset': 5, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': True}, 'ifortbl': {'offset': 6, 'words': 50, 'type': 'int', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'snuptbl': {'offset': 56, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'rstbl': {'offset': 106, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'rgltbl': {'offset': 156, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'hstbl': {'offset': 206, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'laitbl': {'offset': 256, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'albtbl': {'offset': 306, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'z0tbl': {'offset': 356, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'lemitbl': {'offset': 406, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'pctbl': {'offset': 456, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'shdtbl': {'offset': 506, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'maxalb': {'offset': 556, 'words': 50, 'type': 'float', 'dim': ['1:nlus'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'topt_data': {'offset': 606, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'cmcmax_data': {'offset': 607, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'cfactr_data': {'offset': 608, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'rsmax_data': {'offset': 609, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'slcats': {'offset': 610, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'sltype': {'offset': 611, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': True}, 'bb': {'offset': 612, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'drysmc': {'offset': 642, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'hc': {'offset': 672, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'maxsmc': {'offset': 702, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'refsmc': {'offset': 732, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'satpsi': {'offset': 762, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'satdk': {'offset': 792, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'satdw': {'offset': 822, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'wltsmc': {'offset': 852, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'qtz': {'offset': 882, 'words': 30, 'type': 'float', 'dim': ['1:nsltype'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'slpcats': {'offset': 912, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'slope_data': {'offset': 913, 'words': 30, 'type': 'float', 'dim': ['1:nslope'], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'sbeta_data': {'offset': 943, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'fxexp_data': {'offset': 944, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'csoil_data': {'offset': 945, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'salp_data': {'offset': 946, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'refdk_data': {'offset': 947, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'refkdt_data': {'offset': 948, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'frzk_data': {'offset': 949, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'zbot_data': {'offset': 950, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'smlow_data': {'offset': 951, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'smhigh_data': {'offset': 952, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'czil_data': {'offset': 953, 'words': 1, 'type': 'float', 'dim': [], 'intent': None, 'optional': False, 'init': None, 'parameter': False, 'logical': False, 'character': False}, 'targetcell': {'offset': 954, 'words': 1, 'type': 'int', 'dim': [], 'intent': None, 'optional': False, 'init': '0', 'parameter': False, 'logical': False, 'character': False}}
CUDA_SOURCE = r'''

#ifndef RUCDEV
#define RUCDEV __device__
#endif
template<class T> RUCDEV T ruc_max(T a,T b) { return a>b?a:b; }
template<class T> RUCDEV T ruc_min(T a,T b) { return a<b?a:b; }
struct RRange { int lo,hi; };
template<class T> struct FArray {
 T *p; int lo1,lo2,lo3,s1,s2,s3,n1=1,n2=1,n3=1;
 RUCDEV T &operator()(int i) const { return p[(i-lo1)*s1]; }
 RUCDEV T &operator()(int i,int j) const { return p[(i-lo1)*s1+(j-lo2)*s2]; }
 RUCDEV T &operator()(int i,int j,int k) const { return p[(i-lo1)*s1+(j-lo2)*s2+(k-lo3)*s3]; }
 RUCDEV FArray operator()(RRange r) const {
  int lo=r.lo?r.lo:lo1,hi=r.hi?r.hi:lo1+n1-1;
  return {p+(lo-lo1)*s1,1,1,1,s1,0,0,hi-lo+1,1,1};
 }
 RUCDEV FArray operator()(int i,RRange r,int k) const {
  int lo=r.lo?r.lo:lo2,hi=r.hi?r.hi:lo2+n2-1;
  return {p+(i-lo1)*s1+(lo-lo2)*s2+(k-lo3)*s3,1,1,1,s2,0,0,hi-lo+1,1,1};
 }
};
template<class T,class U> RUCDEV void ruc_assign(FArray<T> a,U v) {
 for(int k=0;k<a.n3;++k)for(int j=0;j<a.n2;++j)for(int i=0;i<a.n1;++i)a.p[i*a.s1+j*a.s2+k*a.s3]=v;
}
template<class T> RUCDEV void ruc_assign(FArray<T> a,FArray<T> v) {
 for(int k=0;k<a.n3;++k)for(int j=0;j<a.n2;++j)for(int i=0;i<a.n1;++i)a.p[i*a.s1+j*a.s2+k*a.s3]=v.p[i*v.s1+j*v.s2+k*v.s3];
}
struct Arena {
 float *p; int used,stride; int *error; bool preparation_only=false;
 const int *entry_ilnb=nullptr;
 int *snowtemp_calls=nullptr;
 template<class T> RUCDEV FArray<T> alloc(int lo,int hi,int lo2=1,int hi2=1,int lo3=1,int hi3=1) {
  T *v=reinterpret_cast<T*>(p+used*stride);
  int n1=hi-lo+1,n2=hi2-lo2+1;
  used+=n1*n2*(hi3-lo3+1);
  return {v,lo,lo2,lo3,stride,n1*stride,n1*n2*stride,n1,n2,hi3-lo3+1};
 }
};
struct ArenaScope { Arena &a;int mark;RUCDEV ArenaScope(Arena &v,int m):a(v),mark(m){} RUCDEV ~ArenaScope(){a.used=mark;} };
// The saturation-pressure table tbq(1:5001) of module_ruc_land.F90:479-496 and module_ruc_ice.F90:364-380 depends on
// constants alone, so every cell of every land and sea-ice call rebuilt the same 5,001 words (10,002 exps) in its own
// arena.  THE BREAKAGE THIS PREVENTS: those words were 89 % of the RUC arena (5,001 of 5,624 per cell), 3.7 GB per
// call on x1.163842 (one RTX 5090 out of memory at this allocation, 2026-10-28), and with them a launch past about
// 373,000 cells overflowed the int32 arena offsets.  ruc_tbq_init runs the same statements in the same order once per
// loaded module (_module); land and sea ice read the table in place.  Explicitly rounded arithmetic only, so the
// words do not depend on the kernel they are evaluated in.
__device__ float ruc_tbq_table[5001];
extern "C" __global__ void ruc_tbq_init() {
 if(blockIdx.x!=0 || threadIdx.x!=0)return;
 float cq = __fsub_rn(173.15f, .05f);
 const float r61 = __fmul_rn(6.1153f, 0.62198f);
 for (int k = 1; k <= 5001; k += 1) {
  cq = __fadd_rn(cq, .05f);
  const float evs = gfk_exp_fma(__fdiv_rn(__fmul_rn(17.67f, __fsub_rn(cq, 273.15f)), __fsub_rn(cq, 29.65f)));
  const float eis = gfk_exp_fma(__fsub_rn(22.514f, __fdiv_rn(6.15e3f, cq)));
  if ((cq >= 273.15f)) { ruc_tbq_table[k - 1] = __fmul_rn(r61, evs); } else { ruc_tbq_table[k - 1] = __fmul_rn(r61, eis); }
 }
}

const int nlus = 50;
const int nsltype = 30;
const int nslope = 30;
const int irrig_opt = 0;
const int isncond_opt = 2;
__constant__ float sncovfac_values[] = {0.030f, 0.030f, 0.030f, 0.030f, 0.030f, 0.016f, 0.016f, 0.020f, 0.020f, 0.020f, 0.020f, 0.014f, 0.042f, 0.026f, 0.030f, 0.016f, 0.030f, 0.030f, 0.030f, 0.030f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f};
RUCDEV float sncovfac(int i) {return sncovfac_values[i-1];}
__constant__ float mfsno_values[] = {1.00f, 1.00f, 1.00f, 1.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 3.00f, 3.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 2.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f};
RUCDEV float mfsno(int i) {return mfsno_values[i-1];}
const int isncovr_opt = 2;
const float g = 9.80616f;
const float rhowater = 1000.0f;
const float piconst = 3.141592653589793f;
const float ep_1 = 0.6083624362945557f;
const float ep_2 = 0.6217504332755633f;
const float r_d = 287.0f;
const float r_v = 461.6f;
const float cp = 1004.5f;
const float xlv = 2.50e6f;
const float xls = 2.85e6f;
const float xlf = 3.50e5f;
const float t00 = 273.15f;
const float p1000mb = 100000.0f;
struct Tables {
 int &lucats;
 int &bare;
 int &natural;
 int &crop;
 int &urban;
 int &lutype;
 FArray<int> ifortbl;
 FArray<float> snuptbl;
 FArray<float> rstbl;
 FArray<float> rgltbl;
 FArray<float> hstbl;
 FArray<float> laitbl;
 FArray<float> albtbl;
 FArray<float> z0tbl;
 FArray<float> lemitbl;
 FArray<float> pctbl;
 FArray<float> shdtbl;
 FArray<float> maxalb;
 float &topt_data;
 float &cmcmax_data;
 float &cfactr_data;
 float &rsmax_data;
 int &slcats;
 int &sltype;
 FArray<float> bb;
 FArray<float> drysmc;
 FArray<float> hc;
 FArray<float> maxsmc;
 FArray<float> refsmc;
 FArray<float> satpsi;
 FArray<float> satdk;
 FArray<float> satdw;
 FArray<float> wltsmc;
 FArray<float> qtz;
 int &slpcats;
 FArray<float> slope_data;
 float &sbeta_data;
 float &fxexp_data;
 float &csoil_data;
 float &salp_data;
 float &refdk_data;
 float &refkdt_data;
 float &frzk_data;
 float &zbot_data;
 float &smlow_data;
 float &smhigh_data;
 float &czil_data;
 int &targetcell;
 RUCDEV Tables(float *p):lucats(reinterpret_cast<int*>(p)[0]),bare(reinterpret_cast<int*>(p)[1]),natural(reinterpret_cast<int*>(p)[2]),crop(reinterpret_cast<int*>(p)[3]),urban(reinterpret_cast<int*>(p)[4]),lutype(reinterpret_cast<int*>(p)[5]),ifortbl{reinterpret_cast<int*>(p+6),1,1,1,1,0,0},snuptbl{reinterpret_cast<float*>(p+56),1,1,1,1,0,0},rstbl{reinterpret_cast<float*>(p+106),1,1,1,1,0,0},rgltbl{reinterpret_cast<float*>(p+156),1,1,1,1,0,0},hstbl{reinterpret_cast<float*>(p+206),1,1,1,1,0,0},laitbl{reinterpret_cast<float*>(p+256),1,1,1,1,0,0},albtbl{reinterpret_cast<float*>(p+306),1,1,1,1,0,0},z0tbl{reinterpret_cast<float*>(p+356),1,1,1,1,0,0},lemitbl{reinterpret_cast<float*>(p+406),1,1,1,1,0,0},pctbl{reinterpret_cast<float*>(p+456),1,1,1,1,0,0},shdtbl{reinterpret_cast<float*>(p+506),1,1,1,1,0,0},maxalb{reinterpret_cast<float*>(p+556),1,1,1,1,0,0},topt_data(reinterpret_cast<float*>(p)[606]),cmcmax_data(reinterpret_cast<float*>(p)[607]),cfactr_data(reinterpret_cast<float*>(p)[608]),rsmax_data(reinterpret_cast<float*>(p)[609]),slcats(reinterpret_cast<int*>(p)[610]),sltype(reinterpret_cast<int*>(p)[611]),bb{reinterpret_cast<float*>(p+612),1,1,1,1,0,0},drysmc{reinterpret_cast<float*>(p+642),1,1,1,1,0,0},hc{reinterpret_cast<float*>(p+672),1,1,1,1,0,0},maxsmc{reinterpret_cast<float*>(p+702),1,1,1,1,0,0},refsmc{reinterpret_cast<float*>(p+732),1,1,1,1,0,0},satpsi{reinterpret_cast<float*>(p+762),1,1,1,1,0,0},satdk{reinterpret_cast<float*>(p+792),1,1,1,1,0,0},satdw{reinterpret_cast<float*>(p+822),1,1,1,1,0,0},wltsmc{reinterpret_cast<float*>(p+852),1,1,1,1,0,0},qtz{reinterpret_cast<float*>(p+882),1,1,1,1,0,0},slpcats(reinterpret_cast<int*>(p)[912]),slope_data{reinterpret_cast<float*>(p+913),1,1,1,1,0,0},sbeta_data(reinterpret_cast<float*>(p)[943]),fxexp_data(reinterpret_cast<float*>(p)[944]),csoil_data(reinterpret_cast<float*>(p)[945]),salp_data(reinterpret_cast<float*>(p)[946]),refdk_data(reinterpret_cast<float*>(p)[947]),refkdt_data(reinterpret_cast<float*>(p)[948]),frzk_data(reinterpret_cast<float*>(p)[949]),zbot_data(reinterpret_cast<float*>(p)[950]),smlow_data(reinterpret_cast<float*>(p)[951]),smhigh_data(reinterpret_cast<float*>(p)[952]),czil_data(reinterpret_cast<float*>(p)[953]),targetcell(reinterpret_cast<int*>(p)[954]) {}
};
RUCDEV void land_ruc_land(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> pattern_spp_lsm, FArray<float> field_sf, float  dt, int  ktau, int  nsl, int  restart, int  cycling, int  lakemodel, FArray<float> lakemask, FArray<float> graupelncv, FArray<float> snowncv, FArray<float> rainncv, FArray<float> zs, FArray<float> rainbl, FArray<float> snow, FArray<float> snowh, FArray<float> snowc, FArray<float> frzfrac, int  frpcpn, FArray<float> rhosnf, FArray<float> precipfr, FArray<float> z3d, FArray<float> p8w, FArray<float> t3d, FArray<float> qv3d, FArray<float> qc3d, FArray<float> rho3d, FArray<float> glw, FArray<float> gsw, FArray<float> emiss, FArray<float> chklowq, FArray<float> chs, FArray<float> flqc, FArray<float> flhc, FArray<float> mavail, FArray<float> canwat, FArray<float> vegfra, FArray<float> alb, FArray<float> znt, FArray<float> z0, FArray<float> snoalb, FArray<float> albbck, FArray<float> lai, int  mminlu, FArray<float> landusef, int  nlcat, int  mosaic_lu, int  mosaic_soil, FArray<float> soilctop, int  nscat, FArray<float> qsfc, FArray<float> qsg, FArray<float> qvg, FArray<float> qcg, FArray<float> dew, FArray<float> soilt1, FArray<float> tsnav, FArray<float> tbot, FArray<int> ivgtyp, FArray<int> isltyp, FArray<float> xland, int  iswater, int  isice, FArray<float> xice, float  xice_threshold, float  cp, float  rovcp, float  g0, float  lv, float  stbolt, FArray<float> soilmois, FArray<float> sh2o, FArray<float> smavail, FArray<float> smmax, FArray<float> tso, FArray<float> soilt, FArray<float> hfx, FArray<float> qfx, FArray<float> lh, FArray<float> sfcrunoff, FArray<float> udrunoff, FArray<float> acrunoff, FArray<float> sfcexc, FArray<float> sfcevp, FArray<float> grdflx, FArray<float> snowfallac, FArray<float> acsnow, FArray<float> snom, FArray<float> smfr3d, FArray<float> keepfr3dflag, int  myjpbl, FArray<float> shdmin, FArray<float> shdmax, int  rdlai2d, FArray<int> globalcells, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte, bool has_pattern_spp_lsm, bool has_field_sf, bool has_graupelncv, bool has_snowncv, bool has_rainncv);
RUCDEV void land_sfctmp(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, float  delt, int  ktau, float  conflx, int  i, int  j, int  nzs, int  nddzs, int  nroot, float  meltfactor, int & iland, int & isoil, float  xland, int  ivgtyp, int  isltyp, float & prcpms, float & newsnms, float & snwe, float & snhei, float & snowfrac, float & rhosn, float & rhonewsn, float & rhosnfall, float & snowrat, float & grauprat, float & icerat, float & curat, float  patm, float  tabs, float  qvatm, float  qcatm, float  rho, float  glw, float  gsw, float & emiss, float  qkms, float  tkms, float  pc, float & mavail, float & cst, float  vegfra, float & alb, float & znt, float & alb_snow, float  alb_snow_free, float  lai, int  myj, float  seaice, int  isice, float & qwrtz, float & rhocs, float & dqm, float & qmin, float & ref, float & wilt, float & psis, float & bclh, float & ksat, float & sat, float  cn, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  cp, float  rovcp, float  g0, float  lv, float  stbolt, float  cw, float  c1sn, float  c2sn, float  kqwrtz, float  kice, float  kwt, float & snweprint, float & snheiprint, float & rsm, FArray<float> soilm1d, FArray<float> ts1d, FArray<float> smfrkeep, FArray<float> keepfr, float & soilt, float & soilt1, float & tsnav, float & dew, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & snom, float & snowfallac, float & acsnow, float & edir1, float & ec1, float & ett1, float & eeta, float & qfx, float & hfx, float & s, float & sublim, float & evapl, float & prcpl, float & fltot, float & runoff1, float & runoff2, FArray<float> soilice, FArray<float> soiliqw, float & infiltr, float & smf, int  globalcellid);
RUCDEV float land_qsn(Tables &tb, Arena &arena, float  tn, FArray<float> t);
RUCDEV void land_soil(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, int  i, int  j, int  iland, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, int  nroot, float & prcpms, float & rainf, float  patm, float  qvatm, float  qcatm, float  glw, float  gsw, float  gswin, float  emiss, float & rnet, float  qkms, float  tkms, float  pc, float & cst, float & drip, float  infwater, float  rho, float  vegfrac, float  lai, int  myj, float  qwrtz, float  rhocs, float  dqm, float  qmin, float  ref, float  wilt, float  psis, float  bclh, float  ksat, float & sat, float  cn, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  xlv, float & cp, float & rovcp, float  g0_p, float  cw, float & stbolt, float & tabs, float  kqwrtz, float  kice, float  kwt, FArray<float> soilmois, FArray<float> tso, FArray<float> smfrkeep, FArray<float> keepfr, float & dew, float & soilt, float & qvg, float & qsg, float & qcg, float & edir1, float & ec1, float & ett1, float & eeta, float & qfx, float & hfx, float & s, float & evapl, float & prcpl, float & fltot, float & runoff1, float & runoff2, float & mavail, FArray<float> soilice, FArray<float> soiliqw, float & infiltrp, float & smf, int  globalcellid);
RUCDEV void land_snowsoil(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, int  i, int  j, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, int  nroot, float  meltfactor, float  rhonewsn, float  snhei_crit, int & iland, float  prcpms, float  rainf, float  newsnow, float & snhei, float & snwe, float & snowfrac, float & rhosn, float  patm, float  qvatm, float  qcatm, float  glw, float  gsw, float  gswin, float & emiss, float & rnet, int  ivgtyp, float  qkms, float  tkms, float  pc, float & cst, float & drip, float  infwater, float  rho, float  vegfrac, float & alb, float & znt, float  lai, int  myj, int  globalcellid, float  qwrtz, float  rhocs, float  dqm, float  qmin, float  ref, float  wilt, float  psis, float  bclh, float  ksat, float  sat, float  cn, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  xlv, float & cp, float & rovcp, float  g0_p, float  cw, float & stbolt, float & tabs, float  kqwrtz, float  kice, float  kwt, int & ilnb, float & snweprint, float & snheiprint, float & rsm, FArray<float> soilmois, FArray<float> tso, FArray<float> smfrkeep, FArray<float> keepfr, float & dew, float & soilt, float & soilt1, float & tsnav, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & snom, float & edir1, float & ec1, float & ett1, float & eeta, float & qfx, float & hfx, float & s, float & sublim, float & prcpl, float & fltot, float & runoff1, float & runoff2, float & mavail, FArray<float> soilice, FArray<float> soiliqw, float & infiltrp);
RUCDEV void land_soiltemp(Tables &tb, Arena &arena, int  i, int  j, int  iland, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, int  nroot, float  prcpms, float  rainf, float  patm, float  tabs, float  qvatm, float  qcatm, float  emiss, float  rnet, float  qkms, float  tkms, float  pc, float  rho, float  vegfrac, float  lai, FArray<float> thdif, FArray<float> cap, float & drycan, float & wetcan, float & transum, float  dew, float & mavail, float  soilres, float  alfa, int  globalid, float  dqm, float  qmin, float  bclh, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> tbq, float  xlv, float  cp, float  g0_p, float  cvw, float  stbolt, FArray<float> tso, float & soilt, float & qvg, float & qsg, float & qcg, float & x);
RUCDEV void land_snowtemp(Tables &tb, Arena &arena, int  i, int  j, int  iland, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, int  nroot, float & snwe, float  snwepr, float & snhei, float  newsnow, float & snowfrac, float & beta, float  deltsn, float  snth, float & rhosn, float  rhonewsn, float  meltfactor, float  prcpms, float  rainf, float  patm, float  tabs, float  qvatm, float  qcatm, float  glw, float  gsw, float & emiss, float & rnet, float  qkms, float  tkms, float  pc, float  rho, float  vegfrac, FArray<float> thdif, FArray<float> cap, float & drycan, float & wetcan, float & cst, FArray<float> tranf, float  transum, float & dew, float & mavail, int  globalcellid, float  dqm, float  qmin, float  psis, float  bclh, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> tbq, float  xlvm, float  cp, float  rovcp, float  g0_p, float  cvw, float  stbolt, float & snweprint, float & snheiprint, float & rsm, FArray<float> tso, float & soilt, float & soilt1, float & tsnav, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & s, int & ilnb, float & x);
RUCDEV void land_soilmoist(Tables &tb, Arena &arena, float  delt, int  nzs, int  nddzs, FArray<float> dtdzs, FArray<float> dtdzs2, float  riw, FArray<float> zsmain, FArray<float> zshalf, FArray<float> diffu, FArray<float> hydro, float  qsg, float  qvg, float  qcg, float  qcatm, float  qvatm, float  prcp, float  qkms, FArray<float> transp, float  drip, float  dew, float  smelt, FArray<float> soilice, float  vegfrac, float  snowfrac, float  soilres, int  globalid, float  dqm, float  qmin, float  ref, float  ksat, float  ras, float & infmax, FArray<float> soilmois, FArray<float> soiliqw, float & mavail, float & runoff, float & runoff2, float & infiltrp);
RUCDEV void land_soilprop(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, int  nzs, FArray<float> fwsat, FArray<float> lwsat, FArray<float> tav, FArray<float> keepfr, FArray<float> soilmois, FArray<float> soiliqw, FArray<float> soilice, FArray<float> soilmoism, FArray<float> soiliqwm, FArray<float> soilicem, int  globalid, float  qwrtz, float  rhocs, float  dqm, float  qmin, float  psis, float  bclh, float  ksat, float  riw, float  xlmelt, float  cp, float  g0_p, float  cvw, float & ci, float  kqwrtz, float  kice, float  kwt, FArray<float> thdif, FArray<float> diffu, FArray<float> hydro, FArray<float> cap);
RUCDEV void land_transf(Tables &tb, Arena &arena, int  i, int  j, int  nzs, int  nroot, FArray<float> soiliqw, float  tabs, float  lai, float  gswin, float  dqm, float  qmin, float  ref, float  wilt, FArray<float> zshalf, float  pc, int  iland, FArray<float> tranf, float & transum);
RUCDEV void land_vilka(Tables &tb, Arena &arena, float  tn, float  d1, float  d2, float  pp, float & qs, float & ts, FArray<float> tt, int  nstep, int  ii, int  j, int  iland, int  isoil);
RUCDEV void land_soilvegin(Tables &tb, Arena &arena, int  mosaic_lu, int  mosaic_soil, FArray<float> soilfrac, int  nscat, float  shdmin, float  shdmax, int  nlcat, int & ivgtyp, int & isltyp, int  iswater, int & iforest, FArray<float> lufrac, float  vegfrac, float & emiss, float & pc, float & znt, float & lai, int  rdlai2d, float & qwrtz, float & rhocs, float & bclh, float & dqm, float & ksat, float & psis, float & qmin, float & ref, float & wilt, int  i, int  j);
RUCDEV void ice_ruc_ice(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> pattern_spp_lsm, FArray<float> field_sf, float  dt, int  ktau, int  nsl, int  restart, int  cycling, FArray<float> graupelncv, FArray<float> snowncv, FArray<float> rainncv, FArray<float> zs, FArray<float> rainbl, FArray<float> frzfrac, int  frpcpn, FArray<float> rhosnf, FArray<float> precipfr, FArray<float> z3d, FArray<float> p8w, FArray<float> t3d, FArray<float> qv3d, FArray<float> qc3d, FArray<float> rho3d, FArray<float> glw, FArray<float> gsw, FArray<float> emiss, FArray<float> chklowq, FArray<float> chs, FArray<float> flqc, FArray<float> flhc, FArray<float> alb, FArray<float> znt, FArray<float> z0, FArray<float> snoalb, FArray<float> albbck, int  mminlu, FArray<float> tbot, FArray<int> ivgtyp, FArray<int> isltyp, FArray<float> xland, int  myjpbl, int  iswater, int  isice, FArray<float> xice, float  xice_threshold, float  cp, float  rovcp, float  g0, float  lv, float  stbolt, FArray<float> tso, FArray<float> soilt, FArray<float> soilt1, FArray<float> tsnav, FArray<float> snow, FArray<float> snowh, FArray<float> snowc, FArray<float> qsfc, FArray<float> qsg, FArray<float> qvg, FArray<float> qcg, FArray<float> hfx, FArray<float> qfx, FArray<float> lh, FArray<float> dew, FArray<float> sfcrunoff, FArray<float> acrunoff, FArray<float> sfcevp, FArray<float> grdflx, FArray<float> snowfallac, FArray<float> acsnow, FArray<float> snom, FArray<int> globalcells, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte, bool has_pattern_spp_lsm, bool has_field_sf, bool has_graupelncv, bool has_snowncv, bool has_rainncv);
RUCDEV void ice_icetmp(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, float  delt, int  ktau, float  conflx, int  i, int  j, int  nzs, int  nddzs, int & iland, int & isoil, float  xland, int  ivgtyp, int  isltyp, float & prcpms, float & newsnms, float & snowrat, float & grauprat, float & icerat, float & curat, float  patm, float  tabs, float  qvatm, float  qcatm, float  rho, float  glw, float  gsw, float & emiss, float  qkms, float  tkms, float & alb, float & znt, float  alb_snow, float  alb_snow_free, int  myj, float  seaice, int  isice, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  cp, float  rovcp, float  g0, float  lv, float  stbolt, float  cw, float  c1sn, float  c2sn, FArray<float> ts1d, float & soilt, float & soilt1, float & tsnav, float & rhosn, float & rhonewsn, float & rhosnfall, float & snwe, float & snhei, float & snowfrac, float & rsm, float & snweprint, float & snheiprint, float & dew, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & snom, float & snowfallac, float & acsnow, float & edir1, float & ec1, float & ett1, float & eeta, float & qfx, float & hfx, float & s, float & sublim, float & evapl, float & prcpl, float & fltot, float & runoff1, float & runoff2, FArray<float> soilice, FArray<float> soiliqw, float & infiltr, float & smf, int  globalcellid);
RUCDEV void ice_sice(Tables &tb, Arena &arena, int  i, int  j, int  iland, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, float & prcpms, float & rainf, float  patm, float  qvatm, float  qcatm, float  glw, float  gsw, float  emiss, float & rnet, float  qkms, float  tkms, float  rho, int  myj, int  globalcellid, FArray<float> tice, FArray<float> rhosice, FArray<float> capice, FArray<float> thdifice, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  xlv, float & cp, float & rovcp, float  cw, float & stbolt, float & tabs, FArray<float> tso, float & dew, float & soilt, float & qvg, float & qsg, float & qcg, float & eeta, float & qfx, float & hfx, float & s, float & evapl, float & prcpl, float & fltot);
RUCDEV void ice_snowseaice(Tables &tb, Arena &arena, int  i, int  j, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, float  rhonewsn, float  snhei_crit, int & iland, float  prcpms, float  rainf, float  newsnow, float & snhei, float & snwe, float & snowfrac, float & rhosn, float  patm, float  qvatm, float  qcatm, float  glw, float  gsw, float & emiss, float & rnet, float  qkms, float  tkms, float  rho, int  myj, int  globalcellid, float & alb, float & znt, FArray<float> tice, FArray<float> rhosice, FArray<float> capice, FArray<float> thdifice, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  xlv, float & cp, float & rovcp, float  cw, float & stbolt, float & tabs, int & ilnb, float & snweprint, float & snheiprint, float & rsm, FArray<float> tso, float & dew, float & soilt, float & soilt1, float & tsnav, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & snom, float & eeta, float & qfx, float & hfx, float & s, float & sublim, float & prcpl, float & fltot);
RUCDEV void ice_vilka(Tables &tb, Arena &arena, float  tn, float  d1, float  d2, float  pp, float & qs, float & ts, FArray<float> tt, int  nstep, int  ii, int  j, int  iland, int  isoil);
RUCDEV float ice_qsn(Tables &tb, Arena &arena, float  tn, FArray<float> t);
RUCDEV void diags_sfcdiags_ruclsm(Tables &tb, Arena &arena, FArray<float> hfx, FArray<float> qfx, FArray<float> tsk, FArray<float> qsfc, FArray<float> cqs, FArray<float> cqs2, FArray<float> chs, FArray<float> chs2, FArray<float> t2, FArray<float> th2, FArray<float> q2, FArray<float> t3d, FArray<float> qv3d, FArray<float> dz, FArray<float> rho3d, FArray<float> p3d, FArray<float> psfc2d, float  cp, float  r_d, float  rovcp, FArray<int> globalcells, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte);
RUCDEV float diags_rslf(Tables &tb, Arena &arena, float  p, float  t);
RUCDEV float diags_rsif(Tables &tb, Arena &arena, float  p, float  t);
RUCDEV void init_rucinit(Tables &tb, Arena &arena, int  restart, int  cycling, int  ncells, int  nsoillevels, FArray<int> ivgtyp, FArray<int> isltyp, FArray<float> tslb, FArray<float> smois, FArray<float> xice, FArray<float> smfr3d, FArray<float> sh2o, FArray<float> mavail, FArray<float> znt, FArray<float> snow, FArray<float> snowh);
RUCDEV void drivers_seaice_prepare(Tables &tb, Arena &arena, int  ncells, float  xice_threshold, FArray<float> xice, FArray<float> sfc_albedo, FArray<float> sfc_emiss, FArray<float> sfc_albedo_p, FArray<float> sfc_emiss_p);
RUCDEV void drivers_seaice_finish(Tables &tb, Arena &arena, int  ncells, float  xice_threshold, int  config_frac_seaice, int  mynn, FArray<float> xice_p, FArray<float> flhc_p, FArray<float> flhc_sea, FArray<float> flqc_p, FArray<float> flqc_sea, FArray<float> cqs_p, FArray<float> cqs_sea, FArray<float> cpm_p, FArray<float> cpm_sea, FArray<float> chs_p, FArray<float> chs_sea, FArray<float> chs2_p, FArray<float> chs2_sea, FArray<float> cqs2_p, FArray<float> cqs2_sea, FArray<float> hfx_p, FArray<float> hfx_sea, FArray<float> lh_p, FArray<float> lh_sea, FArray<float> qfx_p, FArray<float> qfx_sea, FArray<float> qgh_p, FArray<float> qgh_sea, FArray<float> qsfc_p, FArray<float> qsfc_sea, FArray<float> tsk_p, FArray<float> tsk_sea, FArray<float> sfc_albedo_p, FArray<float> sfc_emiss_p);
RUCDEV void drivers_correct_tsk_over_seaice(Tables &tb, Arena &arena, int  ims, int  ime, int  jms, int  jme, int  its, int  ite, int  jts, int  jte, float  xice_thresh, FArray<float> xice, FArray<float> tsk, FArray<float> tsk_sea, FArray<float> tsk_ice);
RUCDEV void drivers_driver_forcing(Tables &tb, Arena &arena, int  ncells, int  micro_on, int  conv_on, FArray<float> sr, FArray<float> rainncv, FArray<float> raincv, FArray<float> gsw, FArray<float> sfc_albedo, FArray<float> sr_p, FArray<float> rainbl_p, FArray<float> swdown_p);

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:104-1261
RUCDEV void land_ruc_land(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> pattern_spp_lsm, FArray<float> field_sf, float  dt, int  ktau, int  nsl, int  restart, int  cycling, int  lakemodel, FArray<float> lakemask, FArray<float> graupelncv, FArray<float> snowncv, FArray<float> rainncv, FArray<float> zs, FArray<float> rainbl, FArray<float> snow, FArray<float> snowh, FArray<float> snowc, FArray<float> frzfrac, int  frpcpn, FArray<float> rhosnf, FArray<float> precipfr, FArray<float> z3d, FArray<float> p8w, FArray<float> t3d, FArray<float> qv3d, FArray<float> qc3d, FArray<float> rho3d, FArray<float> glw, FArray<float> gsw, FArray<float> emiss, FArray<float> chklowq, FArray<float> chs, FArray<float> flqc, FArray<float> flhc, FArray<float> mavail, FArray<float> canwat, FArray<float> vegfra, FArray<float> alb, FArray<float> znt, FArray<float> z0, FArray<float> snoalb, FArray<float> albbck, FArray<float> lai, int  mminlu, FArray<float> landusef, int  nlcat, int  mosaic_lu, int  mosaic_soil, FArray<float> soilctop, int  nscat, FArray<float> qsfc, FArray<float> qsg, FArray<float> qvg, FArray<float> qcg, FArray<float> dew, FArray<float> soilt1, FArray<float> tsnav, FArray<float> tbot, FArray<int> ivgtyp, FArray<int> isltyp, FArray<float> xland, int  iswater, int  isice, FArray<float> xice, float  xice_threshold, float  cp, float  rovcp, float  g0, float  lv, float  stbolt, FArray<float> soilmois, FArray<float> sh2o, FArray<float> smavail, FArray<float> smmax, FArray<float> tso, FArray<float> soilt, FArray<float> hfx, FArray<float> qfx, FArray<float> lh, FArray<float> sfcrunoff, FArray<float> udrunoff, FArray<float> acrunoff, FArray<float> sfcexc, FArray<float> sfcevp, FArray<float> grdflx, FArray<float> snowfallac, FArray<float> acsnow, FArray<float> snom, FArray<float> smfr3d, FArray<float> keepfr3dflag, int  myjpbl, FArray<float> shdmin, FArray<float> shdmax, int  rdlai2d, FArray<int> globalcells, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte, bool has_pattern_spp_lsm, bool has_field_sf, bool has_graupelncv, bool has_snowncv, bool has_rainncv) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int nvegclas = (24 + 3);
    FArray<float> field_sf_loc = arena.alloc<float>(ims, ime, 1, nsl, jms, jme);
    FArray<float> pc = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> runoff1 = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> runoff2 = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> emissl = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> zntl = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> lmavail = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> smelt = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> snoh = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> snflx = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> edir = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> ec = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> ett = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> sublim = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> sflx = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> smf = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> evapl = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> prcpl = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> seaice = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> land_point = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> infiltr = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> budget = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> acbudget = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> waterbudget = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> acwaterbudget = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> smtotold = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> snowold = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> canwatold = arena.alloc<float>(its, ite, jts, jte);
    float rhocs;
    float rhonewsn;
    float rhosn;
    float rhosnfall;
    float bclh;
    float dqm;
    float ksat;
    float psis;
    float qmin;
    float qwrtz;
    float ref;
    float wilt;
    float canwatr;
    float snowfrac;
    float snhei;
    float snwe;
    float cn;
    float sat;
    float cw;
    float c1sn;
    float c2sn;
    float kqwrtz;
    float kice;
    float kwt;
    FArray<float> zsmain = arena.alloc<float>(1, nsl);
    FArray<float> zshalf = arena.alloc<float>(1, nsl);
    FArray<float> dtdzs2 = arena.alloc<float>(1, nsl);
    FArray<float> dtdzs = arena.alloc<float>(1, (2 * (nsl - 2)));
    FArray<float> tbq{ruc_tbq_table, 1, 1, 1, 1, 5001, 5001, 5001, 1, 1}; // the shared table (ruc_tbq_init)
    FArray<float> soilm1d = arena.alloc<float>(1, nsl);
    FArray<float> tso1d = arena.alloc<float>(1, nsl);
    FArray<float> soilice = arena.alloc<float>(1, nsl);
    FArray<float> soiliqw = arena.alloc<float>(1, nsl);
    FArray<float> smfrkeep = arena.alloc<float>(1, nsl);
    FArray<float> keepfr = arena.alloc<float>(1, nsl);
    FArray<float> lufrac = arena.alloc<float>(1, nlcat);
    FArray<float> soilfrac = arena.alloc<float>(1, nscat);
    float rsm;
    float snweprint;
    float snheiprint;
    float prcpms;
    float newsnms;
    float prcpncliq;
    float prcpncfr;
    float prcpculiq;
    float prcpcufr;
    float patm;
    float patmb;
    float tabs;
    float qvatm;
    float qcatm;
    float q2sat;
    float conflx;
    float rho;
    float qkms;
    float tkms;
    float snowrat;
    float grauprat;
    float graupamt;
    float icerat;
    float curat;
    float infiltrp;
    float cq;
    float r61;
    float r273;
    float arp;
    float brp;
    float x;
    float evs;
    float eis;
    float cropfr;
    float cropsm;
    float newsm;
    float factor;
    float meltfactor;
    float ac;
    float as;
    float wb;
    int nroot;
    int iland;
    int isoil;
    int iforest;
    int i;
    int j;
    int k;
    int nzs;
    int nzs1;
    int nddzs;
    int k1;
    int l;
    int k2;
    int kp;
    int km;
    int globalcellid;
    int message;
    int print_flag = false;
    FArray<float> rstoch = arena.alloc<float>(ims, ime, 1, nsl, jms, jme);
    FArray<float> rstoch_temp = arena.alloc<float>(1, nsl);
    FArray<float> field_sf_temp = arena.alloc<float>(1, nsl);
    FArray<float> emisso = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> vegfrao = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> albo = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> snoalbo = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> emisslo = arena.alloc<float>(its, ite, jts, jte);
    nzs = nsl; // L461
    nddzs = (2 * (nzs - 2)); // L462
    ruc_assign(rstoch, 0.0f); // L464
    ruc_assign(field_sf_loc, 0.0f); // L465
    if ((spp_lsm == 1)) { // L467
    for (j = jts; j <= jte; j += 1) { loop1_cycle: ; // L468
    for (i = its; i <= ite; i += 1) { loop2_cycle: ; // L469
    for (k = 1; k <= nsl; k += 1) { loop3_cycle: ; // L470
    rstoch(i, k, j) = pattern_spp_lsm(i, k, j); // L471
    field_sf_loc(i, k, j) = field_sf(i, k, j); // L472
    loop3_continue: ;
    } loop3_exit: ; // L473
    loop2_continue: ;
    } loop2_exit: ; // L474
    loop1_continue: ;
    } loop1_exit: ; // L475
    } // L476
    cq = __fsub_rn(173.15f, .05f); // L479
    r273 = __fdiv_rn(1.f, 273.15f); // L480
    r61 = __fmul_rn(6.1153f, 0.62198f); // L481
    arp = __fdiv_rn(__fmul_rn(77455.f, 41.9f), 461.525f); // L482
    brp = __fdiv_rn(__fmul_rn(64.f, 41.9f), 461.525f); // L483
    k = 5002; // L485-496: tbq(1:5001) is ruc_tbq_table (ruc_tbq_init); the DO loop leaves k past its bound
    for (j = jts; j <= jte; j += 1) { loop5_cycle: ; // L498
    for (i = its; i <= ite; i += 1) { loop6_cycle: ; // L499
    if (((lakemodel == 1.f) && (lakemask(i, j) == 1.f))) { return ; } // L501
    land_point(i, j) = 0.f; // L505
    if (((__fsub_rn(xland(i, j), 1.5f) < 0.f) && (xice(i, j) >= xice_threshold))) { // L506
    seaice(i, j) = 1.f; // L507
    } else { // L508
    seaice(i, j) = 0.f; // L509
    if ((soilmois(i, 1, j) < 0.7f)) { land_point(i, j) = 1.f; } // L510
    } // L511
    if (((__fsub_rn(xland(i, j), 1.5f) >= 0.f) || ((seaice(i, j) < 0.50f) && (land_point(i, j) < 0.5f)))) { // L513
    smavail(i, j) = 1.0f; // L515
    smmax(i, j) = 1.0f; // L516
    snow(i, j) = 0.0f; // L517
    snowh(i, j) = 0.0f; // L518
    snowc(i, j) = 0.0f; // L519
    lmavail(i, j) = 1.0f; // L520
    iland = iswater; // L522
    isoil = 14; // L523
    if (false) { // L524
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L525
    } // L527
    patmb = __fmul_rn(p8w(i, 1, j), 1.e-2f); // L529
    qvg(i, j) = __fdiv_rn(land_qsn(tb, arena, soilt(i, j), tbq), patmb); // L530
    qsfc(i, j) = __fdiv_rn(qvg(i, j), __fadd_rn(1.f, qvg(i, j))); // L531
    chklowq(i, j) = 1.f; // L532
    q2sat = __fdiv_rn(land_qsn(tb, arena, t3d(i, 1, j), tbq), patmb); // L533
    for (k = 1; k <= nzs; k += 1) { loop7_cycle: ; // L535
    soilmois(i, k, j) = 1.0f; // L536
    sh2o(i, k, j) = 1.0f; // L537
    smfr3d(i, k, j) = 0.0f; // L538
    smfr3d(i, k, j) = 0.0f; // L539
    tso(i, k, j) = soilt(i, j); // L540
    loop7_continue: ;
    } loop7_exit: ; // L541
    } else if (((__fsub_rn(xland(i, j), 1.5f) < 0.f) && (seaice(i, j) > 0.5f))) { // L544
    if (false) { // L546
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L548
    } // L549
    land_point(i, j) = 0.f; // L550
    iland = isice; // L551
    isoil = 16; // L552
    znt(i, j) = 0.011f; // L553
    snoalb(i, j) = 0.75f; // L554
    dqm = 1.f; // L555
    ref = 1.f; // L556
    qmin = 0.f; // L557
    wilt = 0.f; // L558
    emissl(i, j) = 0.98f; // L559
    patmb = __fmul_rn(p8w(i, 1, j), 1.e-2f); // L561
    qvg(i, j) = __fdiv_rn(land_qsn(tb, arena, soilt(i, j), tbq), patmb); // L562
    qsg(i, j) = qvg(i, j); // L563
    qsfc(i, j) = __fdiv_rn(qvg(i, j), __fadd_rn(1.f, qvg(i, j))); // L564
    for (k = 1; k <= nzs; k += 1) { loop8_cycle: ; // L566
    soilmois(i, k, j) = 1.f; // L567
    smfr3d(i, k, j) = 1.f; // L568
    sh2o(i, k, j) = 0.f; // L569
    keepfr3dflag(i, k, j) = 0.f; // L570
    tso(i, k, j) = ruc_min<float>(271.4f, tso(i, k, j)); // L571
    loop8_continue: ;
    } loop8_exit: ; // L572
    } else { // L574
    if (false) { // L575
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L576
    land_point(i, j) = 1.f; // L577
    } // L578
    } // L579
    loop6_continue: ;
    } loop6_exit: ; // L580
    loop5_continue: ;
    } loop5_exit: ; // L581
    if (arena.preparation_only) return; // Native tile return L501; prefix through L582
    if ((ktau == 1)) { // L587
    if (((!restart) && (!cycling))) { // L589
    for (j = jts; j <= jte; j += 1) { loop9_cycle: ; // L590
    for (i = its; i <= ite; i += 1) { loop10_cycle: ; // L591
    for (k = 1; k <= nsl; k += 1) { loop11_cycle: ; // L592
    keepfr3dflag(i, k, j) = 0.f; // L593
    loop11_continue: ;
    } loop11_exit: ; // L594
    if (((soilt1(i, j) < 170.f) || (soilt1(i, j) > 400.f))) { // L597
    if ((snowc(i, j) > 0.f)) { // L598
    soilt1(i, j) = __fmul_rn(0.5f, __fadd_rn(soilt(i, j), tso(i, 1, j))); // L599
    if (false) { // L600
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L601
    } // L603
    } else { // L604
    soilt1(i, j) = tso(i, 1, j); // L605
    } // L606
    } // L607
    tsnav(i, j) = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(soilt(i, j), tso(i, 1, j))), 273.15f); // L609
    patmb = __fmul_rn(p8w(i, kms, j), 1.e-2f); // L610
    qsg(i, j) = __fdiv_rn(land_qsn(tb, arena, soilt(i, j), tbq), patmb); // L611
    if (((qcg(i, j) < 0.f) || (qcg(i, j) > 0.1f))) { // L612
    qcg(i, j) = 0.f; // L613
    if (false) { // L614
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L615
    } // L617
    } // L618
    if (((qvg(i, j) <= 0.f) || (qvg(i, j) > 0.1f))) { // L620
    qvg(i, j) = qv3d(i, 1, j); // L621
    if (false) { // L622
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L623
    } // L625
    } // L626
    qsfc(i, j) = __fdiv_rn(qvg(i, j), __fadd_rn(1.f, qvg(i, j))); // L627
    smelt(i, j) = 0.f; // L628
    snom(i, j) = 0.f; // L629
    snowfallac(i, j) = 0.f; // L630
    precipfr(i, j) = 0.f; // L631
    rhosnf(i, j) = (-1.e3f); // L632
    snflx(i, j) = 0.f; // L633
    dew(i, j) = 0.f; // L634
    pc(i, j) = 0.f; // L635
    zntl(i, j) = 0.f; // L636
    runoff1(i, j) = 0.f; // L637
    runoff2(i, j) = 0.f; // L638
    sfcrunoff(i, j) = 0.f; // L639
    udrunoff(i, j) = 0.f; // L640
    acrunoff(i, j) = 0.f; // L641
    emissl(i, j) = 0.f; // L642
    budget(i, j) = 0.f; // L643
    acbudget(i, j) = 0.f; // L644
    waterbudget(i, j) = 0.f; // L645
    acwaterbudget(i, j) = 0.f; // L646
    smtotold(i, j) = 0.f; // L647
    canwatold(i, j) = 0.f; // L648
    chklowq(i, j) = 1.f; // L649
    infiltr(i, j) = 0.f; // L650
    snoh(i, j) = 0.f; // L651
    edir(i, j) = 0.f; // L652
    ec(i, j) = 0.f; // L653
    ett(i, j) = 0.f; // L654
    sublim(i, j) = 0.f; // L655
    sflx(i, j) = 0.f; // L656
    smf(i, j) = 0.f; // L657
    evapl(i, j) = 0.f; // L658
    prcpl(i, j) = 0.f; // L659
    loop10_continue: ;
    } loop10_exit: ; // L660
    loop9_continue: ;
    } loop9_exit: ; // L661
    for (k = 1; k <= nsl; k += 1) { loop12_cycle: ; // L663
    soilice(k) = 0.f; // L664
    soiliqw(k) = 0.f; // L665
    loop12_continue: ;
    } loop12_exit: ; // L666
    } // L667
    } // L668
    prcpms = 0.f; // L672
    newsnms = 0.f; // L673
    prcpncliq = 0.f; // L674
    prcpculiq = 0.f; // L675
    prcpncfr = 0.f; // L676
    prcpcufr = 0.f; // L677
    for (j = jts; j <= jte; j += 1) { loop13_cycle: ; // L680
    for (i = its; i <= ite; i += 1) { loop14_cycle: ; // L681
    if ((land_point(i, j) > 0.5f)) { // L683
    if ((globalcells(i) == tb.targetcell)) { // L686
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L688
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L689
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L690
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L691
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L692
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L693
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L694
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L695
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L696
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L697
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L698
    } // L699
    iland = ivgtyp(i, j); // L702
    isoil = isltyp(i, j); // L703
    tabs = t3d(i, kms, j); // L704
    qvatm = qv3d(i, kms, j); // L705
    qcatm = qc3d(i, kms, j); // L706
    patm = __fmul_rn(p8w(i, kms, j), 1.e-5f); // L707
    conflx = __fmul_rn(z3d(i, kms, j), 0.5f); // L711
    rho = rho3d(i, kms, j); // L712
    snowrat = 0.f; // L714
    grauprat = 0.f; // L715
    icerat = 0.f; // L716
    curat = 0.f; // L717
    if (frpcpn) { // L718
    prcpncliq = __fmul_rn(rainncv(i, j), __fsub_rn(1.f, frzfrac(i, j))); // L720
    prcpncfr = __fmul_rn(rainncv(i, j), frzfrac(i, j)); // L721
    if (((frzfrac(i, j) > 0.f) && (tabs < 273.f))) { // L725
    prcpculiq = ruc_max<float>(0.f, __fmul_rn(__fsub_rn(rainbl(i, j), rainncv(i, j)), __fsub_rn(1.f, frzfrac(i, j)))); // L726
    prcpcufr = ruc_max<float>(0.f, __fmul_rn(__fsub_rn(rainbl(i, j), rainncv(i, j)), frzfrac(i, j))); // L727
    } else { // L728
    if ((tabs < 273.f)) { // L729
    prcpcufr = ruc_max<float>(0.f, __fsub_rn(rainbl(i, j), rainncv(i, j))); // L730
    prcpculiq = 0.f; // L731
    } else { // L732
    prcpcufr = 0.f; // L733
    prcpculiq = ruc_max<float>(0.f, __fsub_rn(rainbl(i, j), rainncv(i, j))); // L734
    } // L735
    } // L736
    prcpms = __fmul_rn(__fdiv_rn(__fadd_rn(prcpncliq, prcpculiq), dt), 1.e-3f); // L738
    newsnms = __fmul_rn(__fdiv_rn(__fadd_rn(prcpncfr, prcpcufr), dt), 1.e-3f); // L739
    if (has_graupelncv) { // L741
    graupamt = graupelncv(i, j); // L742
    } else { // L743
    graupamt = 0.f; // L744
    } // L745
    if ((__fadd_rn(prcpncfr, prcpcufr) > 0.f)) { // L747
    snowrat = ruc_min<float>(1.f, ruc_max<float>(0.f, __fdiv_rn(snowncv(i, j), __fadd_rn(prcpncfr, prcpcufr)))); // L749
    grauprat = ruc_min<float>(1.f, ruc_max<float>(0.f, __fdiv_rn(graupamt, __fadd_rn(prcpncfr, prcpcufr)))); // L750
    icerat = ruc_min<float>(1.f, ruc_max<float>(0.f, __fdiv_rn(__fsub_rn(__fsub_rn(prcpncfr, snowncv(i, j)), graupamt), __fadd_rn(prcpncfr, prcpcufr)))); // L751
    curat = ruc_min<float>(1.f, ruc_max<float>(0.f, __fdiv_rn(prcpcufr, __fadd_rn(prcpncfr, prcpcufr)))); // L753
    } // L754
    } else { // L765
    if ((tabs <= 273.15f)) { // L766
    prcpms = 0.f; // L767
    newsnms = __fmul_rn(__fdiv_rn(rainbl(i, j), dt), 1.e-3f); // L768
    snowrat = 1.f; // L771
    } else { // L772
    prcpms = __fmul_rn(__fdiv_rn(rainbl(i, j), dt), 1.e-3f); // L773
    newsnms = 0.f; // L774
    } // L775
    } // L776
    precipfr(i, j) = __fmul_rn(__fmul_rn(newsnms, dt), 1.e3f); // L780
    qkms = __fdiv_rn(__fdiv_rn(flqc(i, j), rho), mavail(i, j)); // L783
    tkms = __fdiv_rn(__fdiv_rn(flhc(i, j), rho), __fmul_rn(cp, __fadd_rn(1.f, __fmul_rn(0.84f, qvatm)))); // L785
    snwe = __fmul_rn(snow(i, j), 1.e-3f); // L787
    snhei = snowh(i, j); // L788
    canwatr = __fmul_rn(canwat(i, j), 1.e-3f); // L789
    snowfrac = snowc(i, j); // L791
    rhosnfall = rhosnf(i, j); // L792
    snowold(i, j) = snwe; // L794
    zsmain(1) = 0.f; // L796
    zshalf(1) = 0.f; // L797
    for (k = 2; k <= nzs; k += 1) { loop15_cycle: ; // L798
    zsmain(k) = zs(k); // L799
    zshalf(k) = __fmul_rn(0.5f, __fadd_rn(zsmain((k - 1)), zsmain(k))); // L800
    loop15_continue: ;
    } loop15_exit: ; // L801
    for (k = 1; k <= nlcat; k += 1) { loop16_cycle: ; // L803
    lufrac(k) = landusef(i, k, j); // L804
    loop16_continue: ;
    } loop16_exit: ; // L805
    for (k = 1; k <= nscat; k += 1) { loop17_cycle: ; // L806
    soilfrac(k) = soilctop(i, k, j); // L807
    loop17_continue: ;
    } loop17_exit: ; // L808
    nzs1 = (nzs - 1); // L813
    if (false) { // L815
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L817
    } // L818
    for (k = 2; k <= nzs1; k += 1) { loop18_cycle: ; // L820
    k1 = ((2 * k) - 3); // L821
    k2 = (k1 + 1); // L822
    x = __fdiv_rn(__fdiv_rn(dt, 2.f), __fsub_rn(zshalf((k + 1)), zshalf(k))); // L823
    dtdzs(k1) = __fdiv_rn(x, __fsub_rn(zsmain(k), zsmain((k - 1)))); // L824
    dtdzs2((k - 1)) = x; // L825
    dtdzs(k2) = __fdiv_rn(x, __fsub_rn(zsmain((k + 1)), zsmain(k))); // L826
    if (false) { // L827
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L829
    } // L830
    loop18_continue: ;
    } loop18_exit: ; // L831
    cw = 4.183e6f; // L833
    kqwrtz = 7.7f; // L838
    kice = 2.2f; // L839
    kwt = 0.57f; // L840
    c1sn = 0.026f; // L845
    c2sn = 21.f; // L846
    nroot = 4; // L850
    rhonewsn = 200.f; // L852
    if (((snow(i, j) > 0.f) && (snowh(i, j) > 0.f))) { // L853
    rhosn = __fdiv_rn(snow(i, j), snowh(i, j)); // L854
    } else { // L855
    rhosn = 300.f; // L856
    } // L857
    if (false) { // L859
    if (((ktau == 1) && (globalcells(i) == tb.targetcell))) { ; /* Native diagnostic I/O omitted; no physics mutation. */ } // L860
    } // L862
    land_soilvegin(tb, arena, mosaic_lu, mosaic_soil, soilfrac, nscat, shdmin(i, j), shdmax(i, j), nlcat, iland, isoil, iswater, iforest, lufrac, vegfra(i, j), emissl(i, j), pc(i, j), znt(i, j), lai(i, j), rdlai2d, qwrtz, rhocs, bclh, dqm, ksat, psis, qmin, ref, wilt, i, j); // L864
    if (false) { // L868
    if (((ktau == 1) && (globalcells(i) == tb.targetcell))) { ; /* Native diagnostic I/O omitted; no physics mutation. */ } // L869
    if ((globalcells(i) == tb.targetcell)) { // L872
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L873
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L875
    } // L877
    } // L878
    cn = tb.cfactr_data; // L880
    sat = 5.e-4f; // L881
    if ((iforest > 2)) { // L884
    meltfactor = 2.0f; // L890
    for (k = 2; k <= nzs; k += 1) { loop19_cycle: ; // L892
    if ((zsmain(k) >= 0.4f)) { // L893
    nroot = k; // L894
    goto label_111; // L895
    } // L896
    loop19_continue: ;
    } loop19_exit: ; // L897
    } else { // L898
    meltfactor = 0.85f; // L905
    for (k = 2; k <= nzs; k += 1) { loop20_cycle: ; // L907
    if ((zsmain(k) >= 1.1f)) { // L908
    nroot = k; // L909
    goto label_111; // L910
    } // L911
    loop20_continue: ;
    } loop20_exit: ; // L912
    } // L913
    label_111: ;
    ; // L914
    if ((globalcells(i) == tb.targetcell)) { // L918
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L919
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L920
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L922
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L923
    } // L924
    for (k = 1; k <= nzs; k += 1) { loop21_cycle: ; // L929
    soilm1d(k) = ruc_min<float>(ruc_max<float>(0.f, __fsub_rn(soilmois(i, k, j), qmin)), dqm); // L931
    tso1d(k) = tso(i, k, j); // L932
    soiliqw(k) = ruc_min<float>(ruc_max<float>(0.f, __fsub_rn(sh2o(i, k, j), qmin)), soilm1d(k)); // L933
    soilice(k) = __fdiv_rn(__fsub_rn(soilm1d(k), soiliqw(k)), 0.9f); // L934
    loop21_continue: ;
    } loop21_exit: ; // L935
    for (k = 1; k <= nzs; k += 1) { loop22_cycle: ; // L937
    smfrkeep(k) = smfr3d(i, k, j); // L938
    keepfr(k) = keepfr3dflag(i, k, j); // L939
    loop22_continue: ;
    } loop22_exit: ; // L940
    lmavail(i, j) = ruc_max<float>(0.00001f, ruc_min<float>(1.f, __fdiv_rn(soilm1d(1), __fsub_rn(ref, qmin)))); // L942
    if ((globalcells(i) == tb.targetcell)) { // L945
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L946
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L947
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L949
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L950
    } // L951
    smtotold(i, j) = 0.f; // L953
    for (k = 1; k <= (nzs - 1); k += 1) { loop23_cycle: ; // L954
    smtotold(i, j) = __fadd_rn(smtotold(i, j), __fmul_rn(__fadd_rn(qmin, soilm1d(k)), __fsub_rn(zshalf((k + 1)), zshalf(k)))); // L955
    loop23_continue: ;
    } loop23_exit: ; // L957
    smtotold(i, j) = __fadd_rn(smtotold(i, j), __fmul_rn(__fadd_rn(qmin, soilm1d(nzs)), __fsub_rn(zsmain(nzs), zshalf(nzs)))); // L959
    canwatold(i, j) = canwatr; // L962
    if (false) { // L963
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L964
    } // L966
    ruc_assign(rstoch_temp, rstoch(i, RRange{0,0}, j)); // L967
    ruc_assign(field_sf_temp, field_sf_loc(i, RRange{0,0}, j)); // L968
    land_sfctmp(tb, arena, spp_lsm, rstoch_temp, field_sf_temp, dt, ktau, conflx, i, j, nzs, nddzs, nroot, meltfactor, iland, isoil, xland(i, j), ivgtyp(i, j), isltyp(i, j), prcpms, newsnms, snwe, snhei, snowfrac, rhosn, rhonewsn, rhosnfall, snowrat, grauprat, icerat, curat, patm, tabs, qvatm, qcatm, rho, glw(i, j), gsw(i, j), emissl(i, j), qkms, tkms, pc(i, j), lmavail(i, j), canwatr, vegfra(i, j), alb(i, j), znt(i, j), snoalb(i, j), albbck(i, j), lai(i, j), myjpbl, seaice(i, j), isice, qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat, sat, cn, zsmain, zshalf, dtdzs, dtdzs2, tbq, cp, rovcp, g0, lv, stbolt, cw, c1sn, c2sn, kqwrtz, kice, kwt, snweprint, snheiprint, rsm, soilm1d, tso1d, smfrkeep, keepfr, soilt(i, j), soilt1(i, j), tsnav(i, j), dew(i, j), qvg(i, j), qsg(i, j), qcg(i, j), smelt(i, j), snoh(i, j), snflx(i, j), snom(i, j), snowfallac(i, j), acsnow(i, j), edir(i, j), ec(i, j), ett(i, j), qfx(i, j), lh(i, j), hfx(i, j), sflx(i, j), sublim(i, j), evapl(i, j), prcpl(i, j), budget(i, j), runoff1(i, j), runoff2(i, j), soilice, soiliqw, infiltrp, smf(i, j), globalcells(i)); // L970
    ruc_assign(field_sf_loc(i, RRange{0,0}, j), field_sf_temp); // L1004
    if ((mosaic_lu == 1)) { // L1014
    if ((irrig_opt == 1)) { // L1015
    if (((lufrac(tb.crop) > 0) && (lai(i, j) > 1.1f))) { // L1016
    for (k = 1; k <= nroot; k += 1) { loop24_cycle: ; // L1019
    cropsm = __fsub_rn(__fmul_rn(1.1f, wilt), qmin); // L1020
    if ((soilm1d(k) < __fmul_rn(cropsm, lufrac(tb.crop)))) { // L1021
    if (false) { // L1022
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1024
    } // L1027
    soilm1d(k) = __fmul_rn(cropsm, lufrac(tb.crop)); // L1028
    if (false) { // L1029
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1031
    } // L1032
    } // L1033
    loop24_continue: ;
    } loop24_exit: ; // L1034
    } else if (((ivgtyp(i, j) == tb.natural) && (lai(i, j) > 0.7f))) { // L1036
    for (k = 1; k <= nroot; k += 1) { loop25_cycle: ; // L1038
    cropsm = __fsub_rn(__fmul_rn(1.2f, wilt), qmin); // L1039
    if ((soilm1d(k) < __fmul_rn(__fmul_rn(cropsm, lufrac(tb.natural)), 0.4f))) { // L1040
    if (false) { // L1041
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1043
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1046
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1047
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1048
    } // L1049
    soilm1d(k) = __fmul_rn(__fmul_rn(cropsm, lufrac(tb.natural)), 0.4f); // L1050
    if (false) { // L1051
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1053
    } // L1054
    } // L1055
    loop25_continue: ;
    } loop25_exit: ; // L1056
    } // L1057
    } else if ((irrig_opt == 2)) { // L1059
    factor = ruc_max<float>(0.f, ruc_min<float>(1.f, __fdiv_rn(__fsub_rn(vegfra(i, j), shdmin(i, j)), ruc_max<float>(1.f, __fsub_rn(shdmax(i, j), shdmin(i, j)))))); // L1061
    if ((((lufrac(tb.crop) > 0) || (lufrac(tb.natural) > 0.f)) && (factor > 0.75f))) { // L1063
    for (k = 1; k <= nroot; k += 1) { loop26_cycle: ; // L1066
    cropsm = __fsub_rn(__fmul_rn(1.1f, wilt), qmin); // L1067
    cropfr = ruc_min<float>(1.f, __fadd_rn(lufrac(tb.crop), __fmul_rn(0.4f, lufrac(tb.natural)))); // L1068
    newsm = __fadd_rn(__fmul_rn(cropsm, cropfr), __fmul_rn(__fsub_rn(1.f, cropfr), soilm1d(k))); // L1069
    if ((soilm1d(k) < newsm)) { // L1070
    if (false) { // L1071
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1072
    } // L1075
    soilm1d(k) = newsm; // L1076
    if (false) { // L1077
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1079
    } // L1080
    } // L1081
    loop26_continue: ;
    } loop26_exit: ; // L1082
    } // L1083
    } else { // L1084
    if (false) { // L1085
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1086
    } // L1087
    } // L1088
    } // L1090
    if ((spp_lsm == 1)) { // L1094
    for (k = 1; k <= nsl; k += 1) { loop27_cycle: ; // L1095
    field_sf(i, k, j) = field_sf_loc(i, k, j); // L1096
    loop27_continue: ;
    } loop27_exit: ; // L1097
    } // L1098
    smavail(i, j) = 0.f; // L1105
    smmax(i, j) = 0.f; // L1106
    for (k = 1; k <= (nzs - 1); k += 1) { loop28_cycle: ; // L1108
    smavail(i, j) = __fadd_rn(smavail(i, j), __fmul_rn(__fadd_rn(qmin, soilm1d(k)), __fsub_rn(zshalf((k + 1)), zshalf(k)))); // L1109
    smmax(i, j) = __fadd_rn(smmax(i, j), __fmul_rn(__fadd_rn(qmin, dqm), __fsub_rn(zshalf((k + 1)), zshalf(k)))); // L1111
    loop28_continue: ;
    } loop28_exit: ; // L1113
    smavail(i, j) = __fadd_rn(smavail(i, j), __fmul_rn(__fadd_rn(qmin, soilm1d(nzs)), __fsub_rn(zsmain(nzs), zshalf(nzs)))); // L1115
    smmax(i, j) = __fadd_rn(smmax(i, j), __fmul_rn(__fadd_rn(qmin, dqm), __fsub_rn(zsmain(nzs), zshalf(nzs)))); // L1117
    sfcrunoff(i, j) = __fadd_rn(sfcrunoff(i, j), __fmul_rn(__fmul_rn(runoff1(i, j), dt), 1000.0f)); // L1121
    udrunoff(i, j) = __fadd_rn(udrunoff(i, j), __fmul_rn(__fmul_rn(runoff2(i, j), dt), 1000.0f)); // L1122
    acrunoff(i, j) = __fadd_rn(acrunoff(i, j), __fmul_rn(__fmul_rn(runoff1(i, j), dt), 1000.0f)); // L1123
    smavail(i, j) = __fmul_rn(smavail(i, j), 1000.f); // L1124
    smmax(i, j) = __fmul_rn(smmax(i, j), 1000.f); // L1125
    smtotold(i, j) = __fmul_rn(smtotold(i, j), 1000.f); // L1126
    for (k = 1; k <= nzs; k += 1) { loop29_cycle: ; // L1128
    soilmois(i, k, j) = __fadd_rn(soilm1d(k), qmin); // L1130
    sh2o(i, k, j) = ruc_min<float>(__fadd_rn(soiliqw(k), qmin), soilmois(i, k, j)); // L1131
    tso(i, k, j) = tso1d(k); // L1132
    loop29_continue: ;
    } loop29_exit: ; // L1133
    for (k = 1; k <= nzs; k += 1) { loop30_cycle: ; // L1135
    smfr3d(i, k, j) = smfrkeep(k); // L1136
    keepfr3dflag(i, k, j) = keepfr(k); // L1137
    loop30_continue: ;
    } loop30_exit: ; // L1138
    z0(i, j) = znt(i, j); // L1140
    sfcexc(i, j) = tkms; // L1141
    patmb = __fmul_rn(p8w(i, 1, j), 1.e-2f); // L1142
    q2sat = __fdiv_rn(land_qsn(tb, arena, tabs, tbq), patmb); // L1143
    qsfc(i, j) = __fdiv_rn(qvg(i, j), __fadd_rn(1.f, qvg(i, j))); // L1144
    if (((qvatm >= __fmul_rn(q2sat, 0.95f)) && (qvatm < qvg(i, j)))) { // L1148
    chklowq(i, j) = 0.f; // L1149
    } else { // L1150
    chklowq(i, j) = 1.f; // L1151
    } // L1152
    if (false) { // L1154
    if ((chklowq(i, j) == 0.f)) { // L1155
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1156
    } // L1157
    } // L1158
    if ((snow(i, j) == 0.f)) { emissl(i, j) = tb.lemitbl(ivgtyp(i, j)); } // L1160
    emiss(i, j) = emissl(i, j); // L1161
    snow(i, j) = __fmul_rn(ruc_min<float>(3.f, snwe), 1000.f); // L1166
    snowh(i, j) = ruc_min<float>(7.5f, snhei); // L1167
    canwat(i, j) = __fmul_rn(canwatr, 1000.f); // L1168
    infiltr(i, j) = infiltrp; // L1170
    mavail(i, j) = lmavail(i, j); // L1172
    sfcevp(i, j) = __fadd_rn(sfcevp(i, j), __fmul_rn(qfx(i, j), dt)); // L1173
    grdflx(i, j) = __fmul_rn((-1.f), sflx(i, j)); // L1174
    if (((snowfrac > 0.f) && (xice(i, j) >= xice_threshold))) { // L1184
    snowfrac = __fmul_rn(snowfrac, xice(i, j)); // L1185
    } // L1186
    snowc(i, j) = snowfrac; // L1188
    rhosnf(i, j) = rhosnfall; // L1191
    sfcevp(i, j) = __fadd_rn(sfcevp(i, j), __fmul_rn(qfx(i, j), dt)); // L1194
    ac = 0.f; // L1203
    as = 0.f; // L1204
    ac = ruc_max<float>(0.f, __fsub_rn(canwat(i, j), canwatold(i, j))); // L1206
    as = ruc_max<float>(0.f, __fsub_rn(snwe, snowold(i, j))); // L1207
    wb = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fadd_rn(rainbl(i, j), __fmul_rn(__fmul_rn(smelt(i, j), dt), 1.e3f)), __fmul_rn(qfx(i, j), dt)), __fmul_rn(__fmul_rn(runoff1(i, j), dt), 1.e3f)), __fmul_rn(__fmul_rn(runoff2(i, j), dt), 1.e3f)), ac), as), __fsub_rn(smavail(i, j), smtotold(i, j))); // L1208
    waterbudget(i, j) = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fadd_rn(rainbl(i, j), __fmul_rn(__fmul_rn(smelt(i, j), dt), 1.e3f)), __fmul_rn(qfx(i, j), dt)), __fmul_rn(__fmul_rn(runoff1(i, j), dt), 1.e3f)), __fmul_rn(__fmul_rn(runoff2(i, j), dt), 1.e3f)), ac), as), __fsub_rn(smavail(i, j), smtotold(i, j))); // L1213
    acwaterbudget(i, j) = __fadd_rn(acwaterbudget(i, j), waterbudget(i, j)); // L1219
    if ((globalcells(i) == tb.targetcell)) { // L1222
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1223
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1224
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1225
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1226
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1227
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1228
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1233
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1234
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1235
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1236
    } // L1237
    if ((globalcells(i) == tb.targetcell)) { // L1241
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1242
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1244
    } // L1245
    if ((globalcells(i) == tb.targetcell)) { // L1246
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1247
    for (k = 1; k <= nzs; k += 1) { loop31_cycle: ; // L1248
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1249
    loop31_continue: ;
    } loop31_exit: ; // L1250
    } // L1251
    } // L1253
    label_2999: ;
    ; // L1255
    loop14_continue: ;
    } loop14_exit: ; // L1257
    loop13_continue: ;
    } loop13_exit: ; // L1258
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:1266-2038
RUCDEV void land_sfctmp(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, float  delt, int  ktau, float  conflx, int  i, int  j, int  nzs, int  nddzs, int  nroot, float  meltfactor, int & iland, int & isoil, float  xland, int  ivgtyp, int  isltyp, float & prcpms, float & newsnms, float & snwe, float & snhei, float & snowfrac, float & rhosn, float & rhonewsn, float & rhosnfall, float & snowrat, float & grauprat, float & icerat, float & curat, float  patm, float  tabs, float  qvatm, float  qcatm, float  rho, float  glw, float  gsw, float & emiss, float  qkms, float  tkms, float  pc, float & mavail, float & cst, float  vegfra, float & alb, float & znt, float & alb_snow, float  alb_snow_free, float  lai, int  myj, float  seaice, int  isice, float & qwrtz, float & rhocs, float & dqm, float & qmin, float & ref, float & wilt, float & psis, float & bclh, float & ksat, float & sat, float  cn, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  cp, float  rovcp, float  g0, float  lv, float  stbolt, float  cw, float  c1sn, float  c2sn, float  kqwrtz, float  kice, float  kwt, float & snweprint, float & snheiprint, float & rsm, FArray<float> soilm1d, FArray<float> ts1d, FArray<float> smfrkeep, FArray<float> keepfr, float & soilt, float & soilt1, float & tsnav, float & dew, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & snom, float & snowfallac, float & acsnow, float & edir1, float & ec1, float & ett1, float & eeta, float & qfx, float & hfx, float & s, float & sublim, float & evapl, float & prcpl, float & fltot, float & runoff1, float & runoff2, FArray<float> soilice, FArray<float> soiliqw, float & infiltr, float & smf, int  globalcellid) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int ilands;
    FArray<float> ts1ds = arena.alloc<float>(1, nzs);
    FArray<float> soilm1ds = arena.alloc<float>(1, nzs);
    FArray<float> smfrkeeps = arena.alloc<float>(1, nzs);
    FArray<float> soiliqws = arena.alloc<float>(1, nzs);
    FArray<float> soilices = arena.alloc<float>(1, nzs);
    FArray<float> keepfrs = arena.alloc<float>(1, nzs);
    float dews;
    float mavails;
    float edir1s;
    float ec1s;
    float csts;
    float ett1s;
    float eetas;
    float evapls;
    float infiltrs;
    float prcpls;
    float qvgs;
    float qsgs;
    float qcgs;
    float qfxs;
    float hfxs;
    float fltots;
    float runoff1s;
    float runoff2s;
    float ss;
    float soilts;
    int k;
    int ilnb = 0; // Native leaves this local undefined (module_ruc_land.F90:1467 / module_ruc_ice.F90 sfctmp/icetmp); snowtemp only sets it for snow at least snth deep and reads it at :4583 for thinner snow, so native's tsnav there comes from stale stack words. Hex fixes it at 0 (the one-layer formula) so the port is deterministic; native-undefined residue, see MATCH.md.
    float bsn;
    float xsn;
    float rainf;
    float snth;
    float newsn;
    float t3;
    float upflux;
    float xinet;
    float snhei_crit;
    float snhei_crit_newsn;
    float keep_snow_albedo;
    float snowfracnewsn;
    float newsnowratio;
    float dd1;
    float snowfrac2;
    float m;
    float rhonewgr;
    float rhonewice;
    float rnet;
    float gswnew;
    float gswin;
    float emissn;
    float zntsn;
    float emiss_snowfree;
    float vegfrac;
    float snow_mosaic;
    float snfr;
    float vgfr;
    float albsn;
    float drip;
    float dripsn;
    float dripliq;
    float interw;
    float intersn;
    float infwater;
    float intwratio;
    int ilsnow = 99;
    if ((globalcellid == tb.targetcell)) { // L1486
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1487
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1488
    } // L1490
    snhei_crit = __fdiv_rn(__fmul_rn(0.01601f, rhowater), rhosn); // L1502
    snhei_crit_newsn = __fdiv_rn(__fmul_rn(0.0005f, rhowater), rhosn); // L1503
    zntsn = tb.z0tbl(isice); // L1505
    snow_mosaic = 0.f; // L1507
    snfr = 1.f; // L1508
    newsn = 0.f; // L1509
    newsnowratio = 0.f; // L1510
    snowfracnewsn = 0.f; // L1511
    rhonewsn = 100.f; // L1512
    if ((snhei == 0.f)) { snowfrac = 0.f; } // L1513
    smelt = 0.f; // L1514
    rainf = 0.f; // L1515
    rsm = 0.f; // L1516
    dd1 = 0.f; // L1517
    infiltr = 0.f; // L1518
    vegfrac = __fmul_rn(0.01f, vegfra); // L1526
    drip = 0.f; // L1527
    dripsn = 0.f; // L1528
    dripliq = 0.f; // L1529
    smf = 0.f; // L1530
    interw = 0.f; // L1531
    intersn = 0.f; // L1532
    infwater = 0.f; // L1533
    gswnew = gsw; // L1535
    gswin = __fdiv_rn(gsw, __fsub_rn(1.f, alb)); // L1536
    albsn = alb_snow; // L1537
    emissn = 0.98f; // L1538
    emiss_snowfree = tb.lemitbl(ivgtyp); // L1539
    if ((globalcellid == tb.targetcell)) { // L1542
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1543
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1544
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1545
    } // L1547
    if ((snhei > __fdiv_rn(__fmul_rn(0.0081f, 1.e3f), rhosn))) { // L1549
    bsn = __fmul_rn(__fmul_rn(__fdiv_rn(delt, 3600.f), c1sn), gfk_exp_fma(__fsub_rn(__fmul_rn(0.08f, ruc_min<float>(0.f, tsnav)), __fmul_rn(__fmul_rn(c2sn, rhosn), 1.e-3f)))); // L1551
    if ((__fmul_rn(__fmul_rn(bsn, snwe), 100.f) < 1.e-4f)) { goto label_777; } // L1552
    xsn = __fdiv_rn(__fmul_rn(rhosn, __fsub_rn(gfk_exp_fma(__fmul_rn(__fmul_rn(bsn, snwe), 100.f)), 1.f)), __fmul_rn(__fmul_rn(bsn, snwe), 100.f)); // L1553
    rhosn = ruc_min<float>(ruc_max<float>(58.8f, xsn), 500.f); // L1554
    label_777: ;
    ; // L1555
    } // L1556
    if ((snowfrac < 0.75f)) { snow_mosaic = 1.f; } // L1559
    newsn = __fmul_rn(newsnms, delt); // L1561
    if ((newsn > 0.f)) { // L1563
    if ((globalcellid == tb.targetcell)) { // L1567
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1568
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1569
    } // L1570
    newsnowratio = ruc_min<float>(1.f, __fdiv_rn(newsn, __fadd_rn(snwe, newsn))); // L1572
    rhonewsn = ruc_min<float>(125.f, __fdiv_rn(1000.0f, ruc_max<float>(8.f, __fmul_rn(17.f, sfire_tanhf(__fmul_rn(__fsub_rn(276.65f, tabs), 0.15f)))))); // L1577
    rhonewgr = ruc_min<float>(500.f, __fdiv_rn(rhowater, ruc_max<float>(2.f, __fmul_rn(3.5f, sfire_tanhf(__fmul_rn(__fsub_rn(274.15f, tabs), 0.3333f)))))); // L1578
    rhonewice = rhonewsn; // L1579
    rhosnfall = ruc_min<float>(500.f, ruc_max<float>(58.8f, __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(rhonewsn, snowrat), __fmul_rn(rhonewgr, grauprat)), __fmul_rn(rhonewice, icerat)), __fmul_rn(rhonewgr, curat)))); // L1584
    rhonewsn = rhosnfall; // L1588
    xsn = __fdiv_rn(__fadd_rn(__fmul_rn(rhosn, snwe), __fmul_rn(rhonewsn, newsn)), __fadd_rn(snwe, newsn)); // L1593
    rhosn = ruc_min<float>(ruc_max<float>(58.8f, xsn), 500.f); // L1595
    } // L1597
    if ((prcpms != 0.f)) { // L1599
    rainf = 1.f; // L1607
    } // L1608
    drip = 0.f; // L1610
    intwratio = 0.f; // L1611
    if ((vegfrac > 0.01f)) { // L1612
    interw = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(0.25f, delt), prcpms), __fsub_rn(1.f, gfk_exp_fma(__fmul_rn((-0.5f), lai)))), vegfrac); // L1615
    intersn = __fmul_rn(__fmul_rn(__fmul_rn(0.25f, newsn), __fsub_rn(1.f, gfk_exp_fma(__fmul_rn((-0.5f), lai)))), vegfrac); // L1616
    infwater = __fsub_rn(prcpms, __fdiv_rn(interw, delt)); // L1617
    if ((__fadd_rn(interw, intersn) > 0.f)) { // L1618
    intwratio = __fdiv_rn(interw, __fadd_rn(interw, intersn)); // L1619
    } // L1620
    dd1 = __fadd_rn(__fadd_rn(cst, interw), intersn); // L1623
    cst = dd1; // L1624
    if ((cst > sat)) { // L1625
    cst = sat; // L1626
    drip = __fsub_rn(dd1, sat); // L1627
    } // L1628
    } else { // L1629
    cst = 0.f; // L1630
    drip = 0.f; // L1631
    interw = 0.f; // L1632
    intersn = 0.f; // L1633
    infwater = prcpms; // L1634
    } // L1635
    if ((newsn > 0.f)) { // L1637
    snwe = ruc_max<float>(0.f, __fsub_rn(__fadd_rn(snwe, newsn), intersn)); // L1639
    if ((drip > 0.f)) { // L1641
    if ((snow_mosaic == 1.f)) { // L1642
    dripliq = __fmul_rn(drip, intwratio); // L1643
    dripsn = __fsub_rn(drip, dripliq); // L1644
    snwe = __fadd_rn(snwe, dripsn); // L1645
    infwater = __fadd_rn(infwater, dripliq); // L1646
    dripliq = 0.f; // L1647
    dripsn = 0.f; // L1648
    } else { // L1649
    snwe = __fadd_rn(snwe, drip); // L1650
    } // L1651
    } // L1652
    snhei = __fdiv_rn(__fmul_rn(snwe, rhowater), rhosn); // L1653
    newsn = __fdiv_rn(__fmul_rn(newsn, rhowater), rhonewsn); // L1654
    } // L1655
    if ((snhei > 0.0f)) { // L1657
    iland = isice; // L1660
    if ((isncovr_opt == 1)) { // L1672
    snowfrac = ruc_min<float>(1.f, __fdiv_rn(snhei, __fmul_rn(2.f, snhei_crit))); // L1673
    } else if ((isncovr_opt == 2)) { // L1674
    snowfrac = ruc_min<float>(1.f, __fdiv_rn(snhei, __fmul_rn(2.f, snhei_crit))); // L1675
    snowfrac2 = sfire_tanhf(__fdiv_rn(snhei, __fmul_rn(__fmul_rn(2.5f, ruc_min<float>(0.2f, znt)), __fdiv_rn(rhosn, rhonewsn)))); // L1683
    snowfrac = __fmul_rn(0.5f, __fadd_rn(snowfrac, snowfrac2)); // L1686
    } else { // L1687
    m = 1.f; // L1690
    snowfrac = sfire_tanhf(__fdiv_rn(snhei, __fmul_rn(__fmul_rn(10.f, sncovfac(ivgtyp)), gfk_pow(__fdiv_rn(rhosn, rhonewsn), m)))); // L1694
    } // L1695
    if ((newsn > 0.f)) { // L1697
    snowfracnewsn = ruc_min<float>(1.f, __fdiv_rn(snowfallac, snhei_crit_newsn)); // L1698
    } // L1699
    if ((ivgtyp == tb.urban)) { snowfrac = ruc_min<float>(0.75f, snowfrac); } // L1702
    if ((snowfrac < 0.75f)) { snow_mosaic = 1.f; } // L1713
    keep_snow_albedo = 0.f; // L1715
    if (((snowfracnewsn > 0.99f) && (rhosnfall < 450.f))) { // L1716
    keep_snow_albedo = 1.f; // L1718
    snow_mosaic = 0.f; // L1719
    } // L1720
    if ((globalcellid == tb.targetcell)) { // L1723
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1724
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1725
    } // L1727
    if ((((newsn == 0.f) && (znt <= 0.2f)) && (ivgtyp != isice))) { // L1731
    if ((snhei <= __fmul_rn(2.f, znt))) { // L1732
    znt = __fadd_rn(__fmul_rn(0.55f, znt), __fmul_rn(0.45f, tb.z0tbl(iland))); // L1733
    } else if (((snhei > __fmul_rn(2.f, znt)) && (snhei <= __fmul_rn(4.f, znt)))) { // L1734
    znt = __fadd_rn(__fmul_rn(0.2f, znt), __fmul_rn(0.8f, tb.z0tbl(iland))); // L1735
    } else if ((snhei > __fmul_rn(4.f, znt))) { // L1736
    znt = tb.z0tbl(iland); // L1737
    } // L1738
    } // L1739
    if ((snow_mosaic == 1.f)) { // L1746
    albsn = alb_snow; // L1747
    if (((keep_snow_albedo > 0.9f) && (albsn < 0.4f))) { // L1748
    albsn = 0.7f; // L1753
    } // L1754
    emiss = emissn; // L1756
    } else { // L1757
    albsn = ruc_max<float>(__fmul_rn(keep_snow_albedo, alb_snow), ruc_min<float>(__fadd_rn(alb_snow_free, __fmul_rn(__fsub_rn(alb_snow, alb_snow_free), snowfrac)), alb_snow)); // L1758
    if ((((newsn > 0.f) && (keep_snow_albedo > 0.9f)) && (albsn < 0.4f))) { // L1761
    albsn = 0.7f; // L1766
    } // L1768
    emiss = ruc_max<float>(__fmul_rn(keep_snow_albedo, emissn), ruc_min<float>(__fadd_rn(emiss_snowfree, __fmul_rn(__fsub_rn(emissn, emiss_snowfree), snowfrac)), emissn)); // L1770
    } // L1773
    if ((globalcellid == tb.targetcell)) { // L1776
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1777
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1778
    } // L1779
    if (((albsn < 0.4f) || (keep_snow_albedo == 1))) { // L1786
    alb = albsn; // L1787
    } else { // L1788
    alb = ruc_min<float>(albsn, ruc_max<float>(__fsub_rn(albsn, __fmul_rn(__fdiv_rn(__fmul_rn(0.1f, __fsub_rn(soilt, 263.15f)), __fsub_rn(273.15f, 263.15f)), albsn)), __fsub_rn(albsn, 0.05f))); // L1790
    } // L1792
    if ((snow_mosaic == 1.f)) { // L1794
    gswnew = __fmul_rn(gswin, __fsub_rn(1.f, alb_snow_free)); // L1795
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, soilt), soilt), soilt); // L1797
    upflux = __fmul_rn(t3, soilt); // L1798
    xinet = __fmul_rn(emiss_snowfree, __fsub_rn(glw, upflux)); // L1799
    rnet = __fadd_rn(gswnew, xinet); // L1800
    if ((globalcellid == tb.targetcell)) { // L1802
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1803
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1804
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1805
    } // L1806
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L1807
    soilm1ds(k) = soilm1d(k); // L1808
    ts1ds(k) = ts1d(k); // L1809
    smfrkeeps(k) = smfrkeep(k); // L1810
    keepfrs(k) = keepfr(k); // L1811
    soilices(k) = soilice(k); // L1812
    soiliqws(k) = soiliqw(k); // L1813
    loop1_continue: ;
    } loop1_exit: ; // L1814
    soilts = soilt; // L1815
    qvgs = qvg; // L1816
    qsgs = qsg; // L1817
    qcgs = qcg; // L1818
    csts = cst; // L1819
    mavails = mavail; // L1820
    smelt = 0.f; // L1821
    runoff1s = 0.f; // L1822
    runoff2s = 0.f; // L1823
    ilands = ivgtyp; // L1825
    land_soil(tb, arena, spp_lsm, rstochcol, fieldcol_sf, i, j, ilands, isoil, delt, ktau, conflx, nzs, nddzs, nroot, prcpms, rainf, patm, qvatm, qcatm, glw, gswnew, gswin, emiss_snowfree, rnet, qkms, tkms, pc, csts, dripliq, infwater, rho, vegfrac, lai, myj, qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat, sat, cn, zsmain, zshalf, dtdzs, dtdzs2, tbq, lv, cp, rovcp, g0, cw, stbolt, tabs, kqwrtz, kice, kwt, soilm1ds, ts1ds, smfrkeeps, keepfrs, dews, soilts, qvgs, qsgs, qcgs, edir1s, ec1s, ett1s, eetas, qfxs, hfxs, ss, evapls, prcpls, fltots, runoff1s, runoff2s, mavails, soilices, soiliqws, infiltrs, smf, globalcellid); // L1826
    } // L1846
    gswnew = __fmul_rn(gswin, __fsub_rn(1.f, alb)); // L1850
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, soilt), soilt), soilt); // L1853
    upflux = __fmul_rn(t3, soilt); // L1854
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L1855
    rnet = __fadd_rn(gswnew, xinet); // L1856
    if ((globalcellid == tb.targetcell)) { // L1858
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1859
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1860
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1861
    } // L1863
    if ((snow_mosaic == 1.f)) { // L1865
    snfr = 1.f; // L1866
    } else { // L1867
    snfr = snowfrac; // L1868
    } // L1869
    land_snowsoil(tb, arena, spp_lsm, rstochcol, fieldcol_sf, i, j, isoil, delt, ktau, conflx, nzs, nddzs, nroot, meltfactor, rhonewsn, snhei_crit, iland, prcpms, rainf, newsn, snhei, snwe, snfr, rhosn, patm, qvatm, qcatm, glw, gswnew, gswin, emiss, rnet, ivgtyp, qkms, tkms, pc, cst, dripsn, infwater, rho, vegfrac, alb, znt, lai, myj, globalcellid, qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat, sat, cn, zsmain, zshalf, dtdzs, dtdzs2, tbq, lv, cp, rovcp, g0, cw, stbolt, tabs, kqwrtz, kice, kwt, ilnb, snweprint, snheiprint, rsm, soilm1d, ts1d, smfrkeep, keepfr, dew, soilt, soilt1, tsnav, qvg, qsg, qcg, smelt, snoh, snflx, snom, edir1, ec1, ett1, eeta, qfx, hfx, s, sublim, prcpl, fltot, runoff1, runoff2, mavail, soilice, soiliqw, infiltr); // L1870
    if ((snow_mosaic == 1.f)) { // L1893
    if ((globalcellid == tb.targetcell)) { // L1897
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1898
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1899
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1900
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1901
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1902
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1903
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1904
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1905
    } // L1906
    for (k = 1; k <= nzs; k += 1) { loop2_cycle: ; // L1907
    soilm1d(k) = __fadd_rn(__fmul_rn(soilm1ds(k), __fsub_rn(1.f, snowfrac)), __fmul_rn(soilm1d(k), snowfrac)); // L1908
    ts1d(k) = __fadd_rn(__fmul_rn(ts1ds(k), __fsub_rn(1.f, snowfrac)), __fmul_rn(ts1d(k), snowfrac)); // L1909
    smfrkeep(k) = __fadd_rn(__fmul_rn(smfrkeeps(k), __fsub_rn(1.f, snowfrac)), __fmul_rn(smfrkeep(k), snowfrac)); // L1910
    if ((snowfrac > 0.5f)) { // L1911
    keepfr(k) = keepfr(k); // L1912
    } else { // L1913
    keepfr(k) = keepfrs(k); // L1914
    } // L1915
    soilice(k) = __fadd_rn(__fmul_rn(soilices(k), __fsub_rn(1.f, snowfrac)), __fmul_rn(soilice(k), snowfrac)); // L1916
    soiliqw(k) = __fadd_rn(__fmul_rn(soiliqws(k), __fsub_rn(1.f, snowfrac)), __fmul_rn(soiliqw(k), snowfrac)); // L1917
    loop2_continue: ;
    } loop2_exit: ; // L1918
    dew = __fadd_rn(__fmul_rn(dews, __fsub_rn(1.f, snowfrac)), __fmul_rn(dew, snowfrac)); // L1919
    soilt = __fadd_rn(__fmul_rn(soilts, __fsub_rn(1.f, snowfrac)), __fmul_rn(soilt, snowfrac)); // L1920
    qvg = __fadd_rn(__fmul_rn(qvgs, __fsub_rn(1.f, snowfrac)), __fmul_rn(qvg, snowfrac)); // L1921
    qsg = __fadd_rn(__fmul_rn(qsgs, __fsub_rn(1.f, snowfrac)), __fmul_rn(qsg, snowfrac)); // L1922
    qcg = __fadd_rn(__fmul_rn(qcgs, __fsub_rn(1.f, snowfrac)), __fmul_rn(qcg, snowfrac)); // L1923
    edir1 = __fadd_rn(__fmul_rn(edir1s, __fsub_rn(1.f, snowfrac)), __fmul_rn(edir1, snowfrac)); // L1924
    ec1 = __fadd_rn(__fmul_rn(ec1s, __fsub_rn(1.f, snowfrac)), __fmul_rn(ec1, snowfrac)); // L1925
    cst = __fadd_rn(__fmul_rn(csts, __fsub_rn(1.f, snowfrac)), __fmul_rn(cst, snowfrac)); // L1926
    ett1 = __fadd_rn(__fmul_rn(ett1s, __fsub_rn(1.f, snowfrac)), __fmul_rn(ett1, snowfrac)); // L1927
    eeta = __fadd_rn(__fmul_rn(eetas, __fsub_rn(1.f, snowfrac)), __fmul_rn(eeta, snowfrac)); // L1928
    qfx = __fadd_rn(__fmul_rn(qfxs, __fsub_rn(1.f, snowfrac)), __fmul_rn(qfx, snowfrac)); // L1929
    hfx = __fadd_rn(__fmul_rn(hfxs, __fsub_rn(1.f, snowfrac)), __fmul_rn(hfx, snowfrac)); // L1930
    s = __fadd_rn(__fmul_rn(ss, __fsub_rn(1.f, snowfrac)), __fmul_rn(s, snowfrac)); // L1931
    evapl = __fmul_rn(evapls, __fsub_rn(1.f, snowfrac)); // L1932
    sublim = __fmul_rn(sublim, snowfrac); // L1933
    prcpl = __fadd_rn(__fmul_rn(prcpls, __fsub_rn(1.f, snowfrac)), __fmul_rn(prcpl, snowfrac)); // L1934
    fltot = __fadd_rn(__fmul_rn(fltots, __fsub_rn(1.f, snowfrac)), __fmul_rn(fltot, snowfrac)); // L1935
    alb = ruc_max<float>(__fmul_rn(keep_snow_albedo, alb), ruc_min<float>(__fadd_rn(alb_snow_free, __fmul_rn(__fsub_rn(alb, alb_snow_free), snowfrac)), alb)); // L1937
    emiss = ruc_max<float>(__fmul_rn(keep_snow_albedo, emissn), ruc_min<float>(__fadd_rn(emiss_snowfree, __fmul_rn(__fsub_rn(emissn, emiss_snowfree), snowfrac)), emissn)); // L1940
    runoff1 = __fadd_rn(__fmul_rn(runoff1s, __fsub_rn(1.f, snowfrac)), __fmul_rn(runoff1, snowfrac)); // L1944
    runoff2 = __fadd_rn(__fmul_rn(runoff2s, __fsub_rn(1.f, snowfrac)), __fmul_rn(runoff2, snowfrac)); // L1945
    mavail = __fadd_rn(__fmul_rn(mavails, __fsub_rn(1.f, snowfrac)), __fmul_rn(1.f, snowfrac)); // L1946
    infiltr = __fadd_rn(__fmul_rn(infiltrs, __fsub_rn(1.f, snowfrac)), __fmul_rn(infiltr, snowfrac)); // L1947
    if ((globalcellid == tb.targetcell)) { // L1950
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1951
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1952
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1953
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1954
    } // L1955
    } // L1956
    if ((snhei == 0.f)) { // L1958
    alb = alb_snow_free; // L1960
    iland = ivgtyp; // L1961
    } else { // L1962
    if ((isncovr_opt == 1)) { // L1964
    snowfrac = ruc_min<float>(1.f, __fdiv_rn(snhei, __fmul_rn(2.f, snhei_crit))); // L1965
    } else if ((isncovr_opt == 2)) { // L1966
    snowfrac = ruc_min<float>(1.f, __fdiv_rn(snhei, __fmul_rn(2.f, snhei_crit))); // L1967
    snowfrac2 = sfire_tanhf(__fdiv_rn(snhei, __fmul_rn(__fmul_rn(2.5f, ruc_min<float>(0.2f, znt)), __fdiv_rn(rhosn, rhonewsn)))); // L1975
    snowfrac = __fmul_rn(0.5f, __fadd_rn(snowfrac, snowfrac2)); // L1978
    } else { // L1979
    m = 1.f; // L1982
    snowfrac = sfire_tanhf(__fdiv_rn(snhei, __fmul_rn(__fmul_rn(10.f, sncovfac(ivgtyp)), gfk_pow(__fdiv_rn(rhosn, rhonewsn), m)))); // L1986
    } // L1987
    } // L1989
    if ((ivgtyp == tb.urban)) { snowfrac = ruc_min<float>(0.75f, snowfrac); } // L1991
    snowfallac = __fadd_rn(snowfallac, newsn); // L1995
    } else { // L1998
    snheiprint = 0.f; // L2000
    snweprint = 0.f; // L2001
    smelt = 0.f; // L2002
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, soilt), soilt), soilt); // L2005
    upflux = __fmul_rn(t3, soilt); // L2006
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L2007
    rnet = __fadd_rn(gswnew, xinet); // L2008
    if ((globalcellid == tb.targetcell)) { // L2010
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2011
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2012
    } // L2013
    land_soil(tb, arena, spp_lsm, rstochcol, fieldcol_sf, i, j, iland, isoil, delt, ktau, conflx, nzs, nddzs, nroot, prcpms, rainf, patm, qvatm, qcatm, glw, gswnew, gswin, emiss, rnet, qkms, tkms, pc, cst, drip, infwater, rho, vegfrac, lai, myj, qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat, sat, cn, zsmain, zshalf, dtdzs, dtdzs2, tbq, lv, cp, rovcp, g0, cw, stbolt, tabs, kqwrtz, kice, kwt, soilm1d, ts1d, smfrkeep, keepfr, dew, soilt, qvg, qsg, qcg, edir1, ec1, ett1, eeta, qfx, hfx, s, evapl, prcpl, fltot, runoff1, runoff2, mavail, soilice, soiliqw, infiltr, smf, globalcellid); // L2015
    } // L2035
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:2042-2062
RUCDEV float land_qsn(Tables &tb, Arena &arena, float  tn, FArray<float> t) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float qsn;
    float r;
    float r1;
    float r2;
    int i;
    r = __fadd_rn(__fdiv_rn(__fsub_rn(tn, 173.15f), .05f), 1.f); // L2050
    i = int(r); // L2051
    if ((i >= 1)) { goto label_10; } // L2052
    i = 1; // L2053
    r = 1.f; // L2054
    label_10: ;
    if ((i <= 5000)) { goto label_20; } // L2055
    i = 5000; // L2056
    r = 5001.f; // L2057
    label_20: ;
    r1 = t(i); // L2058
    r2 = __fsub_rn(r, i); // L2059
    qsn = __fadd_rn(__fmul_rn(__fsub_rn(t((i + 1)), r1), r2), r1); // L2060
    return qsn;
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:2066-2686
RUCDEV void land_soil(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, int  i, int  j, int  iland, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, int  nroot, float & prcpms, float & rainf, float  patm, float  qvatm, float  qcatm, float  glw, float  gsw, float  gswin, float  emiss, float & rnet, float  qkms, float  tkms, float  pc, float & cst, float & drip, float  infwater, float  rho, float  vegfrac, float  lai, int  myj, float  qwrtz, float  rhocs, float  dqm, float  qmin, float  ref, float  wilt, float  psis, float  bclh, float  ksat, float & sat, float  cn, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  xlv, float & cp, float & rovcp, float  g0_p, float  cw, float & stbolt, float & tabs, float  kqwrtz, float  kice, float  kwt, FArray<float> soilmois, FArray<float> tso, FArray<float> smfrkeep, FArray<float> keepfr, float & dew, float & soilt, float & qvg, float & qsg, float & qcg, float & edir1, float & ec1, float & ett1, float & eeta, float & qfx, float & hfx, float & s, float & evapl, float & prcpl, float & fltot, float & runoff1, float & runoff2, float & mavail, FArray<float> soilice, FArray<float> soiliqw, float & infiltrp, float & smf, int  globalcellid) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float transum;
    float t3;
    float upflux;
    float xinet;
    float g0;
    float lv;
    float xlmelt;
    float dzstop;
    float can;
    float epot;
    float fac;
    float ft;
    float fq;
    float hft;
    float q1;
    float ras;
    float rhoice;
    float sph;
    float trans;
    float zn;
    float ci;
    float cvw;
    float tln;
    float tavln;
    float pi;
    float dd1;
    float cmc2ms;
    float drycan;
    float wetcan;
    float infmax;
    float riw;
    float x;
    FArray<float> transp = arena.alloc<float>(1, nzs);
    FArray<float> cap = arena.alloc<float>(1, nzs);
    FArray<float> diffu = arena.alloc<float>(1, nzs);
    FArray<float> hydro = arena.alloc<float>(1, nzs);
    FArray<float> thdif = arena.alloc<float>(1, nzs);
    FArray<float> tranf = arena.alloc<float>(1, nzs);
    FArray<float> tav = arena.alloc<float>(1, nzs);
    FArray<float> soilmoism = arena.alloc<float>(1, nzs);
    FArray<float> soilicem = arena.alloc<float>(1, nzs);
    FArray<float> soiliqwm = arena.alloc<float>(1, nzs);
    FArray<float> detal = arena.alloc<float>(1, nzs);
    FArray<float> fwsat = arena.alloc<float>(1, nzs);
    FArray<float> lwsat = arena.alloc<float>(1, nzs);
    FArray<float> told = arena.alloc<float>(1, nzs);
    FArray<float> smold = arena.alloc<float>(1, nzs);
    float soiltold;
    float soilres;
    float alfa;
    float fex;
    float fex_fc;
    float fc;
    float psit;
    int nzs1;
    int nzs2;
    int k;
    rhoice = 900.f; // L2270
    ci = __fmul_rn(rhoice, 2100.f); // L2271
    xlmelt = 3.35e+5f; // L2272
    cvw = cw; // L2273
    prcpl = prcpms; // L2275
    smf = 0.f; // L2277
    soiltold = soilt; // L2278
    wetcan = 0.f; // L2280
    drycan = 1.f; // L2281
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L2284
    transp(k) = 0.f; // L2285
    soilmoism(k) = 0.f; // L2286
    soilice(k) = 0.f; // L2287
    soiliqw(k) = 0.f; // L2288
    soilicem(k) = 0.f; // L2289
    soiliqwm(k) = 0.f; // L2290
    lwsat(k) = 0.f; // L2291
    fwsat(k) = 0.f; // L2292
    tav(k) = 0.f; // L2293
    cap(k) = 0.f; // L2294
    thdif(k) = 0.f; // L2295
    diffu(k) = 0.f; // L2296
    hydro(k) = 0.f; // L2297
    tranf(k) = 0.f; // L2298
    detal(k) = 0.f; // L2299
    told(k) = 0.f; // L2300
    smold(k) = 0.f; // L2301
    loop1_continue: ;
    } loop1_exit: ; // L2302
    nzs1 = (nzs - 1); // L2304
    nzs2 = (nzs - 2); // L2305
    dzstop = __fdiv_rn(1.f, __fsub_rn(zsmain(2), zsmain(1))); // L2306
    ras = __fmul_rn(rho, 1.e-3f); // L2307
    riw = __fmul_rn(rhoice, 1.e-3f); // L2308
    for (k = 1; k <= nzs; k += 1) { loop2_cycle: ; // L2312
    tln = gfk_log(__fdiv_rn(tso(k), 273.15f)); // L2314
    if ((tln < 0.f)) { // L2315
    soiliqw(k) = __fsub_rn(__fmul_rn(__fadd_rn(dqm, qmin), gfk_pow(__fdiv_rn(__fdiv_rn(__fdiv_rn(__fmul_rn(xlmelt, __fsub_rn(tso(k), 273.15f)), tso(k)), 9.81f), psis), __fdiv_rn((-1.f), bclh))), qmin); // L2316
    soiliqw(k) = ruc_max<float>(0.f, soiliqw(k)); // L2319
    soiliqw(k) = ruc_min<float>(soiliqw(k), soilmois(k)); // L2320
    soilice(k) = __fdiv_rn(__fsub_rn(soilmois(k), soiliqw(k)), riw); // L2321
    if ((keepfr(k) == 1.f)) { // L2324
    soilice(k) = ruc_min<float>(soilice(k), smfrkeep(k)); // L2325
    soiliqw(k) = ruc_max<float>(0.f, __fsub_rn(soilmois(k), __fmul_rn(soilice(k), riw))); // L2326
    } // L2327
    } else { // L2329
    soilice(k) = 0.f; // L2330
    soiliqw(k) = soilmois(k); // L2331
    } // L2332
    loop2_continue: ;
    } loop2_exit: ; // L2333
    for (k = 1; k <= nzs1; k += 1) { loop3_cycle: ; // L2335
    tav(k) = __fmul_rn(0.5f, __fadd_rn(tso(k), tso((k + 1)))); // L2337
    soilmoism(k) = __fmul_rn(0.5f, __fadd_rn(soilmois(k), soilmois((k + 1)))); // L2338
    tavln = gfk_log(__fdiv_rn(tav(k), 273.15f)); // L2339
    if ((tavln < 0.f)) { // L2341
    soiliqwm(k) = __fsub_rn(__fmul_rn(__fadd_rn(dqm, qmin), gfk_pow(__fdiv_rn(__fdiv_rn(__fdiv_rn(__fmul_rn(xlmelt, __fsub_rn(tav(k), 273.15f)), tav(k)), 9.81f), psis), __fdiv_rn((-1.f), bclh))), qmin); // L2342
    fwsat(k) = __fsub_rn(dqm, soiliqwm(k)); // L2345
    lwsat(k) = __fadd_rn(soiliqwm(k), qmin); // L2346
    soiliqwm(k) = ruc_max<float>(0.f, soiliqwm(k)); // L2347
    soiliqwm(k) = ruc_min<float>(soiliqwm(k), soilmoism(k)); // L2348
    soilicem(k) = __fdiv_rn(__fsub_rn(soilmoism(k), soiliqwm(k)), riw); // L2349
    if ((keepfr(k) == 1.f)) { // L2351
    soilicem(k) = ruc_min<float>(soilicem(k), __fmul_rn(0.5f, __fadd_rn(smfrkeep(k), smfrkeep((k + 1))))); // L2352
    soiliqwm(k) = ruc_max<float>(0.f, __fsub_rn(soilmoism(k), __fmul_rn(soilicem(k), riw))); // L2354
    fwsat(k) = __fsub_rn(dqm, soiliqwm(k)); // L2355
    lwsat(k) = __fadd_rn(soiliqwm(k), qmin); // L2356
    } // L2357
    } else { // L2359
    soilicem(k) = 0.f; // L2360
    soiliqwm(k) = soilmoism(k); // L2361
    lwsat(k) = __fadd_rn(dqm, qmin); // L2362
    fwsat(k) = 0.f; // L2363
    } // L2364
    loop3_continue: ;
    } loop3_exit: ; // L2366
    for (k = 1; k <= nzs; k += 1) { loop4_cycle: ; // L2368
    if ((soilice(k) > 0.f)) { // L2369
    smfrkeep(k) = soilice(k); // L2370
    } else { // L2371
    smfrkeep(k) = __fdiv_rn(soilmois(k), riw); // L2372
    } // L2373
    loop4_continue: ;
    } loop4_exit: ; // L2374
    land_soilprop(tb, arena, spp_lsm, rstochcol, fieldcol_sf, nzs, fwsat, lwsat, tav, keepfr, soilmois, soiliqw, soilice, soilmoism, soiliqwm, soilicem, globalcellid, qwrtz, rhocs, dqm, qmin, psis, bclh, ksat, riw, xlmelt, cp, g0_p, cvw, ci, kqwrtz, kice, kwt, thdif, diffu, hydro, cap); // L2379
    fq = qkms; // L2397
    q1 = __fmul_rn(__fmul_rn((-qkms), ras), __fsub_rn(qvatm, qsg)); // L2399
    dew = 0.f; // L2401
    if ((qvatm >= qsg)) { // L2402
    dew = __fmul_rn(fq, __fsub_rn(qvatm, qsg)); // L2403
    } // L2404
    wetcan = ruc_min<float>(0.25f, gfk_pow(ruc_max<float>(0.f, __fdiv_rn(cst, sat)), cn)); // L2433
    drycan = __fsub_rn(1.f, wetcan); // L2435
    land_transf(tb, arena, i, j, nzs, nroot, soiliqw, tabs, lai, gswin, dqm, qmin, ref, wilt, zshalf, pc, iland, tranf, transum); // L2440
    for (k = 1; k <= nzs; k += 1) { loop5_cycle: ; // L2450
    told(k) = tso(k); // L2451
    smold(k) = soilmois(k); // L2452
    loop5_continue: ;
    } loop5_exit: ; // L2453
    fex = ruc_min<float>(1.f, __fdiv_rn(soilmois(1), dqm)); // L2459
    fex = ruc_max<float>(fex, 0.01f); // L2460
    psit = __fmul_rn(psis, gfk_pow(fex, (-bclh))); // L2461
    psit = ruc_max<float>((-1.e5f), psit); // L2462
    alfa = ruc_min<float>(1.f, gfk_exp_fma(__fdiv_rn(__fdiv_rn(__fmul_rn(g, psit), r_v), soilt))); // L2463
    alfa = 1.f; // L2465
    fc = ref; // L2467
    fex_fc = 1.f; // L2468
    if (((__fadd_rn(soilmois(1), qmin) > fc) || (__fsub_rn(qvatm, qvg) > 0.f))) { // L2469
    soilres = 1.f; // L2470
    } else { // L2471
    fex_fc = ruc_min<float>(1.f, __fdiv_rn(__fadd_rn(soilmois(1), qmin), fc)); // L2472
    fex_fc = ruc_max<float>(fex_fc, 0.01f); // L2473
    soilres = __fmul_rn(0.25f, __fmul_rn(__fsub_rn(1.f, glibc_cosf(__fmul_rn(piconst, fex_fc))), __fsub_rn(1.f, glibc_cosf(__fmul_rn(piconst, fex_fc))))); // L2474
    } // L2475
    if ((globalcellid == tb.targetcell)) { // L2477
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2478
    } // L2480
    land_soiltemp(tb, arena, i, j, iland, isoil, delt, ktau, conflx, nzs, nddzs, nroot, prcpms, rainf, patm, tabs, qvatm, qcatm, emiss, rnet, qkms, tkms, pc, rho, vegfrac, lai, thdif, cap, drycan, wetcan, transum, dew, mavail, soilres, alfa, globalcellid, dqm, qmin, bclh, zsmain, zshalf, dtdzs, tbq, xlv, cp, g0_p, cvw, stbolt, tso, soilt, qvg, qsg, qcg, x); // L2485
    ett1 = 0.f; // L2505
    dew = 0.f; // L2506
    if ((qvatm >= qsg)) { // L2508
    dew = __fmul_rn(qkms, __fsub_rn(qvatm, qsg)); // L2509
    ett1 = 0.f; // L2510
    for (k = 1; k <= nzs; k += 1) { loop6_cycle: ; // L2511
    transp(k) = 0.f; // L2512
    loop6_continue: ;
    } loop6_exit: ; // L2513
    } else { // L2514
    for (k = 1; k <= nroot; k += 1) { loop7_cycle: ; // L2516
    transp(k) = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(vegfrac, ras), qkms), __fsub_rn(qvatm, qsg)), tranf(k)), drycan), zshalf((nroot + 1))); // L2517
    if ((transp(k) > 0.f)) { transp(k) = 0.f; } // L2520
    ett1 = __fsub_rn(ett1, transp(k)); // L2521
    loop7_continue: ;
    } loop7_exit: ; // L2522
    for (k = (nroot + 1); k <= nzs; k += 1) { loop8_cycle: ; // L2523
    transp(k) = 0.f; // L2524
    loop8_continue: ;
    } loop8_exit: ; // L2525
    } // L2526
    for (k = 1; k <= nzs; k += 1) { loop9_cycle: ; // L2529
    tln = gfk_log(__fdiv_rn(tso(k), 273.15f)); // L2531
    if ((tln < 0.f)) { // L2532
    soiliqw(k) = __fsub_rn(__fmul_rn(__fadd_rn(dqm, qmin), gfk_pow(__fdiv_rn(__fdiv_rn(__fdiv_rn(__fmul_rn(xlmelt, __fsub_rn(tso(k), 273.15f)), tso(k)), 9.81f), psis), __fdiv_rn((-1.f), bclh))), qmin); // L2533
    soiliqw(k) = ruc_max<float>(0.f, soiliqw(k)); // L2536
    soiliqw(k) = ruc_min<float>(soiliqw(k), soilmois(k)); // L2537
    soilice(k) = __fdiv_rn(__fsub_rn(soilmois(k), soiliqw(k)), riw); // L2538
    if ((keepfr(k) == 1.f)) { // L2540
    soilice(k) = ruc_min<float>(soilice(k), smfrkeep(k)); // L2541
    soiliqw(k) = ruc_max<float>(0.f, __fsub_rn(soilmois(k), __fmul_rn(soilice(k), riw))); // L2542
    } // L2543
    } else { // L2544
    soilice(k) = 0.f; // L2545
    soiliqw(k) = soilmois(k); // L2546
    } // L2547
    loop9_continue: ;
    } loop9_exit: ; // L2548
    land_soilmoist(tb, arena, delt, nzs, nddzs, dtdzs, dtdzs2, riw, zsmain, zshalf, diffu, hydro, qsg, qvg, qcg, qcatm, qvatm, (-infwater), qkms, transp, drip, dew, 0.f, soilice, vegfrac, 0.f, soilres, globalcellid, dqm, qmin, ref, ksat, ras, infmax, soilmois, soiliqw, mavail, runoff1, runoff2, infiltrp); // L2553
    for (k = 1; k <= nzs; k += 1) { loop10_cycle: ; // L2575
    if ((soilice(k) > 0.f)) { // L2576
    if (((tso(k) > told(k)) && (soilmois(k) > smold(k)))) { // L2577
    keepfr(k) = 1.f; // L2578
    } else { // L2579
    keepfr(k) = 0.f; // L2580
    } // L2581
    } // L2582
    loop10_continue: ;
    } loop10_exit: ; // L2583
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, soiltold), soiltold), soiltold); // L2587
    upflux = __fmul_rn(__fmul_rn(t3, 0.5f), __fadd_rn(soiltold, soilt)); // L2588
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L2589
    hft = __fmul_rn(__fmul_rn(__fmul_rn((-tkms), cp), rho), __fsub_rn(tabs, soilt)); // L2590
    hfx = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-tkms), cp), rho), __fsub_rn(tabs, soilt)), gfk_pow(__fdiv_rn(__fmul_rn(p1000mb, 0.00001f), patm), rovcp)); // L2591
    q1 = __fmul_rn(__fmul_rn((-qkms), ras), __fsub_rn(qvatm, qsg)); // L2593
    cmc2ms = 0.f; // L2595
    if ((q1 <= 0.f)) { // L2596
    ec1 = 0.f; // L2598
    edir1 = 0.f; // L2599
    ett1 = 0.f; // L2600
    if (myj) { // L2601
    eeta = __fmul_rn(__fmul_rn(__fmul_rn((-qkms), ras), __fsub_rn(__fdiv_rn(qvatm, __fadd_rn(1.f, qvatm)), __fdiv_rn(qsg, __fadd_rn(1.f, qsg)))), 1.e3f); // L2603
    cst = __fsub_rn(cst, __fmul_rn(__fmul_rn(eeta, delt), vegfrac)); // L2604
    if (false) { // L2605
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2606
    } // L2607
    } else { // L2608
    eeta = __fmul_rn((-rho), dew); // L2610
    cst = __fadd_rn(cst, __fmul_rn(__fmul_rn(__fmul_rn(delt, dew), ras), vegfrac)); // L2611
    if (false) { // L2612
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2613
    } // L2614
    } // L2615
    qfx = __fmul_rn(xlv, eeta); // L2616
    eeta = __fmul_rn((-rho), dew); // L2617
    } else { // L2618
    edir1 = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-soilres), __fsub_rn(1.f, vegfrac)), qkms), ras), __fsub_rn(qvatm, qvg)); // L2620
    cmc2ms = __fmul_rn(__fdiv_rn(cst, delt), ras); // L2622
    ec1 = __fmul_rn(__fmul_rn(q1, wetcan), vegfrac); // L2623
    if (false) { // L2624
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2625
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2626
    } // L2627
    cst = ruc_max<float>(0.f, __fsub_rn(cst, __fmul_rn(ec1, delt))); // L2629
    if (myj) { // L2631
    eeta = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-soilres), qkms), ras), __fsub_rn(__fdiv_rn(qvatm, __fadd_rn(1.f, qvatm)), __fdiv_rn(qvg, __fadd_rn(1.f, qvg)))), 1.e3f); // L2633
    } else { // L2634
    if ((globalcellid == tb.targetcell)) { // L2636
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2637
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2639
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2640
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2641
    } // L2642
    eeta = __fmul_rn(__fadd_rn(__fadd_rn(edir1, ec1), ett1), 1.e3f); // L2644
    if (false) { // L2645
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2646
    } // L2647
    } // L2648
    qfx = __fmul_rn(xlv, eeta); // L2649
    eeta = __fmul_rn(__fadd_rn(__fadd_rn(edir1, ec1), ett1), 1.e3f); // L2650
    } // L2651
    if (false) { // L2652
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2653
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2654
    } // L2655
    evapl = eeta; // L2657
    s = __fmul_rn(__fmul_rn(__fmul_rn(thdif(1), cap(1)), dzstop), __fsub_rn(tso(1), tso(2))); // L2658
    fltot = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(rnet, hft), __fmul_rn(xlv, eeta)), s), x); // L2660
    if ((globalcellid == tb.targetcell)) { // L2662
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2663
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2664
    } // L2666
    if ((detal(1) != 0.f)) { // L2667
    smf = fltot; // L2670
    if ((globalcellid == tb.targetcell)) { // L2672
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2673
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2674
    } // L2675
    } // L2676
    label_222: ;
    ; // L2679
    label_1123: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2681
    label_1133: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2682
    label_123: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2683
    label_122: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2684
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:2689-3362
RUCDEV void land_snowsoil(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, int  i, int  j, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, int  nroot, float  meltfactor, float  rhonewsn, float  snhei_crit, int & iland, float  prcpms, float  rainf, float  newsnow, float & snhei, float & snwe, float & snowfrac, float & rhosn, float  patm, float  qvatm, float  qcatm, float  glw, float  gsw, float  gswin, float & emiss, float & rnet, int  ivgtyp, float  qkms, float  tkms, float  pc, float & cst, float & drip, float  infwater, float  rho, float  vegfrac, float & alb, float & znt, float  lai, int  myj, int  globalcellid, float  qwrtz, float  rhocs, float  dqm, float  qmin, float  ref, float  wilt, float  psis, float  bclh, float  ksat, float  sat, float  cn, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  xlv, float & cp, float & rovcp, float  g0_p, float  cw, float & stbolt, float & tabs, float  kqwrtz, float  kice, float  kwt, int & ilnb, float & snweprint, float & snheiprint, float & rsm, FArray<float> soilmois, FArray<float> tso, FArray<float> smfrkeep, FArray<float> keepfr, float & dew, float & soilt, float & soilt1, float & tsnav, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & snom, float & edir1, float & ec1, float & ett1, float & eeta, float & qfx, float & hfx, float & s, float & sublim, float & prcpl, float & fltot, float & runoff1, float & runoff2, float & mavail, FArray<float> soilice, FArray<float> soiliqw, float & infiltrp) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int nzs1;
    int nzs2;
    int k;
    float transum;
    float snth;
    float newsn;
    float t3;
    float upflux;
    float xinet;
    float beta;
    float snwepr;
    float epdt;
    float pp;
    float g0;
    float lv;
    float xlvm;
    float xlmelt;
    float dzstop;
    float can;
    float epot;
    float fac;
    float ft;
    float fq;
    float hft;
    float q1;
    float ras;
    float rhoice;
    float sph;
    float trans;
    float zn;
    float ci;
    float cvw;
    float tln;
    float tavln;
    float pi;
    float dd1;
    float cmc2ms;
    float drycan;
    float wetcan;
    float infmax;
    float riw;
    float deltsn;
    float h;
    float umveg;
    FArray<float> transp = arena.alloc<float>(1, nzs);
    FArray<float> cap = arena.alloc<float>(1, nzs);
    FArray<float> diffu = arena.alloc<float>(1, nzs);
    FArray<float> hydro = arena.alloc<float>(1, nzs);
    FArray<float> thdif = arena.alloc<float>(1, nzs);
    FArray<float> tranf = arena.alloc<float>(1, nzs);
    FArray<float> tav = arena.alloc<float>(1, nzs);
    FArray<float> soilmoism = arena.alloc<float>(1, nzs);
    FArray<float> soilicem = arena.alloc<float>(1, nzs);
    FArray<float> soiliqwm = arena.alloc<float>(1, nzs);
    FArray<float> detal = arena.alloc<float>(1, nzs);
    FArray<float> fwsat = arena.alloc<float>(1, nzs);
    FArray<float> lwsat = arena.alloc<float>(1, nzs);
    FArray<float> told = arena.alloc<float>(1, nzs);
    FArray<float> smold = arena.alloc<float>(1, nzs);
    float soiltold;
    float qgold;
    float x;
    cvw = cw; // L2934
    xlmelt = 3.35e+5f; // L2935
    xlvm = __fadd_rn(xlv, xlmelt); // L2937
    soiltold = soilt; // L2951
    qgold = qvg; // L2952
    x = 0.f; // L2954
    deltsn = __fdiv_rn(__fmul_rn(0.05f, 1.e3f), rhosn); // L2956
    snth = __fdiv_rn(__fmul_rn(0.01f, 1.e3f), rhosn); // L2957
    if ((snhei >= __fadd_rn(deltsn, snth))) { // L2962
    if ((__fsub_rn(__fsub_rn(snhei, deltsn), snth) < snth)) { deltsn = __fmul_rn(0.5f, __fsub_rn(snhei, snth)); } // L2963
    if (false) { // L2964
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2965
    } // L2966
    } // L2967
    rhoice = 900.f; // L2969
    ci = __fmul_rn(rhoice, 2100.f); // L2970
    ras = __fmul_rn(rho, 1.e-3f); // L2971
    riw = __fmul_rn(rhoice, 1.e-3f); // L2972
    rsm = 0.f; // L2973
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L2975
    transp(k) = 0.f; // L2976
    soilmoism(k) = 0.f; // L2977
    soiliqwm(k) = 0.f; // L2978
    soilice(k) = 0.f; // L2979
    soilicem(k) = 0.f; // L2980
    lwsat(k) = 0.f; // L2981
    fwsat(k) = 0.f; // L2982
    tav(k) = 0.f; // L2983
    cap(k) = 0.f; // L2984
    diffu(k) = 0.f; // L2985
    hydro(k) = 0.f; // L2986
    thdif(k) = 0.f; // L2987
    tranf(k) = 0.f; // L2988
    detal(k) = 0.f; // L2989
    told(k) = 0.f; // L2990
    smold(k) = 0.f; // L2991
    loop1_continue: ;
    } loop1_exit: ; // L2992
    snweprint = 0.f; // L2994
    snheiprint = 0.f; // L2995
    prcpl = prcpms; // L2996
    nzs1 = (nzs - 1); // L3003
    nzs2 = (nzs - 2); // L3004
    dzstop = __fdiv_rn(1.f, __fsub_rn(zsmain(2), zsmain(1))); // L3005
    for (k = 1; k <= nzs; k += 1) { loop2_cycle: ; // L3011
    tln = gfk_log(__fdiv_rn(tso(k), 273.15f)); // L3013
    if ((tln < 0.f)) { // L3014
    soiliqw(k) = __fsub_rn(__fmul_rn(__fadd_rn(dqm, qmin), gfk_pow(__fdiv_rn(__fdiv_rn(__fdiv_rn(__fmul_rn(xlmelt, __fsub_rn(tso(k), 273.15f)), tso(k)), 9.81f), psis), __fdiv_rn((-1.f), bclh))), qmin); // L3015
    soiliqw(k) = ruc_max<float>(0.f, soiliqw(k)); // L3018
    soiliqw(k) = ruc_min<float>(soiliqw(k), soilmois(k)); // L3019
    soilice(k) = __fdiv_rn(__fsub_rn(soilmois(k), soiliqw(k)), riw); // L3020
    if ((keepfr(k) == 1.f)) { // L3023
    soilice(k) = ruc_min<float>(soilice(k), smfrkeep(k)); // L3024
    soiliqw(k) = ruc_max<float>(0.f, __fsub_rn(soilmois(k), __fmul_rn(__fmul_rn(soilice(k), rhoice), 1.e-3f))); // L3025
    } // L3026
    } else { // L3028
    soilice(k) = 0.f; // L3029
    soiliqw(k) = soilmois(k); // L3030
    } // L3031
    loop2_continue: ;
    } loop2_exit: ; // L3033
    for (k = 1; k <= nzs1; k += 1) { loop3_cycle: ; // L3035
    tav(k) = __fmul_rn(0.5f, __fadd_rn(tso(k), tso((k + 1)))); // L3037
    soilmoism(k) = __fmul_rn(0.5f, __fadd_rn(soilmois(k), soilmois((k + 1)))); // L3038
    tavln = gfk_log(__fdiv_rn(tav(k), 273.15f)); // L3039
    if ((tavln < 0.f)) { // L3041
    soiliqwm(k) = __fsub_rn(__fmul_rn(__fadd_rn(dqm, qmin), gfk_pow(__fdiv_rn(__fdiv_rn(__fdiv_rn(__fmul_rn(xlmelt, __fsub_rn(tav(k), 273.15f)), tav(k)), 9.81f), psis), __fdiv_rn((-1.f), bclh))), qmin); // L3042
    fwsat(k) = __fsub_rn(dqm, soiliqwm(k)); // L3045
    lwsat(k) = __fadd_rn(soiliqwm(k), qmin); // L3046
    soiliqwm(k) = ruc_max<float>(0.f, soiliqwm(k)); // L3047
    soiliqwm(k) = ruc_min<float>(soiliqwm(k), soilmoism(k)); // L3048
    soilicem(k) = __fdiv_rn(__fsub_rn(soilmoism(k), soiliqwm(k)), riw); // L3049
    if ((keepfr(k) == 1.f)) { // L3051
    soilicem(k) = ruc_min<float>(soilicem(k), __fmul_rn(0.5f, __fadd_rn(smfrkeep(k), smfrkeep((k + 1))))); // L3052
    soiliqwm(k) = ruc_max<float>(0.f, __fsub_rn(soilmoism(k), __fmul_rn(soilicem(k), riw))); // L3054
    fwsat(k) = __fsub_rn(dqm, soiliqwm(k)); // L3055
    lwsat(k) = __fadd_rn(soiliqwm(k), qmin); // L3056
    } // L3057
    } else { // L3059
    soilicem(k) = 0.f; // L3060
    soiliqwm(k) = soilmoism(k); // L3061
    lwsat(k) = __fadd_rn(dqm, qmin); // L3062
    fwsat(k) = 0.f; // L3063
    } // L3065
    loop3_continue: ;
    } loop3_exit: ; // L3066
    for (k = 1; k <= nzs; k += 1) { loop4_cycle: ; // L3068
    if ((soilice(k) > 0.f)) { // L3069
    smfrkeep(k) = soilice(k); // L3070
    } else { // L3071
    smfrkeep(k) = __fdiv_rn(soilmois(k), riw); // L3072
    } // L3073
    loop4_continue: ;
    } loop4_exit: ; // L3074
    land_soilprop(tb, arena, spp_lsm, rstochcol, fieldcol_sf, nzs, fwsat, lwsat, tav, keepfr, soilmois, soiliqw, soilice, soilmoism, soiliqwm, soilicem, globalcellid, qwrtz, rhocs, dqm, qmin, psis, bclh, ksat, riw, xlmelt, cp, g0_p, cvw, ci, kqwrtz, kice, kwt, thdif, diffu, hydro, cap); // L3079
    smelt = 0.f; // L3096
    h = mavail; // L3097
    fq = qkms; // L3099
    dew = 0.f; // L3105
    umveg = __fsub_rn(1.f, vegfrac); // L3106
    epot = __fmul_rn((-fq), __fsub_rn(qvatm, qsg)); // L3107
    if ((globalcellid == tb.targetcell)) { // L3110
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3111
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3112
    } // L3113
    snwepr = snwe; // L3114
    beta = 1.f; // L3117
    epdt = __fmul_rn(__fmul_rn(epot, ras), delt); // L3118
    if (((epdt > 0.f) && (snwepr <= epdt))) { // L3119
    beta = __fdiv_rn(snwepr, ruc_max<float>(1.e-8f, epdt)); // L3120
    snwe = 0.f; // L3121
    } // L3122
    wetcan = ruc_min<float>(0.25f, gfk_pow(ruc_max<float>(0.f, __fdiv_rn(cst, sat)), cn)); // L3124
    drycan = __fsub_rn(1.f, wetcan); // L3126
    land_transf(tb, arena, i, j, nzs, nroot, soiliqw, tabs, lai, gswin, dqm, qmin, ref, wilt, zshalf, pc, iland, tranf, transum); // L3131
    for (k = 1; k <= nzs; k += 1) { loop5_cycle: ; // L3140
    told(k) = tso(k); // L3141
    smold(k) = soilmois(k); // L3142
    loop5_continue: ;
    } loop5_exit: ; // L3143
    if (false) { // L3149
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3150
    } // L3151
    land_snowtemp(tb, arena, i, j, iland, isoil, delt, ktau, conflx, nzs, nddzs, nroot, snwe, snwepr, snhei, newsnow, snowfrac, beta, deltsn, snth, rhosn, rhonewsn, meltfactor, prcpms, rainf, patm, tabs, qvatm, qcatm, glw, gsw, emiss, rnet, qkms, tkms, pc, rho, vegfrac, thdif, cap, drycan, wetcan, cst, tranf, transum, dew, mavail, globalcellid, dqm, qmin, psis, bclh, zsmain, zshalf, dtdzs, tbq, xlvm, cp, rovcp, g0_p, cvw, stbolt, snweprint, snheiprint, rsm, tso, soilt, soilt1, tsnav, qvg, qsg, qcg, smelt, snoh, snflx, s, ilnb, x); // L3152
    dew = 0.f; // L3176
    ett1 = 0.f; // L3177
    pp = __fmul_rn(patm, 1.e3f); // L3178
    epot = __fmul_rn((-fq), __fsub_rn(qvatm, qsg)); // L3179
    if ((epot > 0.f)) { // L3180
    for (k = 1; k <= nroot; k += 1) { loop6_cycle: ; // L3182
    transp(k) = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(vegfrac, ras), fq), __fsub_rn(qvatm, qsg)), tranf(k)), drycan), zshalf((nroot + 1))); // L3183
    ett1 = __fsub_rn(ett1, transp(k)); // L3185
    loop6_continue: ;
    } loop6_exit: ; // L3186
    for (k = (nroot + 1); k <= nzs; k += 1) { loop7_cycle: ; // L3187
    transp(k) = 0.f; // L3188
    loop7_continue: ;
    } loop7_exit: ; // L3189
    } else { // L3191
    dew = (-epot); // L3193
    for (k = 1; k <= nzs; k += 1) { loop8_cycle: ; // L3194
    transp(k) = 0.f; // L3195
    loop8_continue: ;
    } loop8_exit: ; // L3196
    ett1 = 0.f; // L3197
    } // L3198
    for (k = 1; k <= nzs; k += 1) { loop9_cycle: ; // L3201
    tln = gfk_log(__fdiv_rn(tso(k), 273.15f)); // L3202
    if ((tln < 0.f)) { // L3203
    soiliqw(k) = __fsub_rn(__fmul_rn(__fadd_rn(dqm, qmin), gfk_pow(__fdiv_rn(__fdiv_rn(__fdiv_rn(__fmul_rn(xlmelt, __fsub_rn(tso(k), 273.15f)), tso(k)), 9.81f), psis), __fdiv_rn((-1.f), bclh))), qmin); // L3204
    soiliqw(k) = ruc_max<float>(0.f, soiliqw(k)); // L3207
    soiliqw(k) = ruc_min<float>(soiliqw(k), soilmois(k)); // L3208
    soilice(k) = __fdiv_rn(__fsub_rn(soilmois(k), soiliqw(k)), riw); // L3209
    if ((keepfr(k) == 1.f)) { // L3211
    soilice(k) = ruc_min<float>(soilice(k), smfrkeep(k)); // L3212
    soiliqw(k) = ruc_max<float>(0.f, __fsub_rn(soilmois(k), __fmul_rn(soilice(k), riw))); // L3213
    } // L3214
    } else { // L3216
    soilice(k) = 0.f; // L3217
    soiliqw(k) = soilmois(k); // L3218
    } // L3219
    loop9_continue: ;
    } loop9_exit: ; // L3220
    land_soilmoist(tb, arena, delt, nzs, nddzs, dtdzs, dtdzs2, riw, zsmain, zshalf, diffu, hydro, qsg, qvg, qcg, qcatm, qvatm, (-infwater), qkms, transp, 0.f, 0.f, smelt, soilice, vegfrac, snowfrac, 1.f, globalcellid, dqm, qmin, ref, ksat, ras, infmax, soilmois, soiliqw, mavail, runoff1, runoff2, infiltrp); // L3226
    if ((snhei == 0.f)) { // L3243
    tsnav = __fsub_rn(soilt, 273.15f); // L3244
    } // L3245
    snom = __fadd_rn(snom, __fmul_rn(__fmul_rn(smelt, delt), 1.e3f)); // L3249
    for (k = 1; k <= nzs; k += 1) { loop10_cycle: ; // L3260
    if ((soilice(k) > 0.f)) { // L3261
    if (((tso(k) > told(k)) && (soilmois(k) > smold(k)))) { // L3262
    keepfr(k) = 1.f; // L3263
    } else { // L3264
    keepfr(k) = 0.f; // L3265
    } // L3266
    } // L3267
    loop10_continue: ;
    } loop10_exit: ; // L3268
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, soiltold), soiltold), soiltold); // L3271
    upflux = __fmul_rn(__fmul_rn(t3, 0.5f), __fadd_rn(soiltold, soilt)); // L3272
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L3273
    hfx = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-tkms), cp), rho), __fsub_rn(tabs, soilt)), gfk_pow(__fdiv_rn(__fmul_rn(p1000mb, 0.00001f), patm), rovcp)); // L3274
    if (false) { // L3276
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3277
    } // L3278
    hft = __fmul_rn(__fmul_rn(__fmul_rn((-tkms), cp), rho), __fsub_rn(tabs, soilt)); // L3279
    if (false) { // L3280
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3281
    } // L3282
    q1 = __fmul_rn(__fmul_rn((-fq), ras), __fsub_rn(qvatm, qsg)); // L3283
    cmc2ms = 0.f; // L3284
    if ((q1 < 0.f)) { // L3285
    edir1 = 0.f; // L3287
    ec1 = 0.f; // L3288
    ett1 = 0.f; // L3289
    if (myj) { // L3291
    eeta = __fmul_rn(__fmul_rn(__fmul_rn((-qkms), ras), __fsub_rn(__fdiv_rn(qvatm, __fadd_rn(1.f, qvatm)), __fdiv_rn(qsg, __fadd_rn(1.f, qsg)))), 1.e3f); // L3293
    cst = __fsub_rn(cst, __fmul_rn(__fmul_rn(eeta, delt), vegfrac)); // L3294
    if (false) { // L3295
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3296
    } // L3297
    } else { // L3298
    dew = __fmul_rn(qkms, __fsub_rn(qvatm, qsg)); // L3300
    eeta = __fmul_rn((-rho), dew); // L3301
    cst = __fadd_rn(cst, __fmul_rn(__fmul_rn(__fmul_rn(delt, dew), ras), vegfrac)); // L3302
    if (false) { // L3303
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3304
    } // L3305
    } // L3306
    qfx = __fmul_rn(xlvm, eeta); // L3307
    eeta = __fmul_rn((-rho), dew); // L3308
    } else { // L3309
    edir1 = __fmul_rn(__fmul_rn(q1, umveg), beta); // L3311
    cmc2ms = __fmul_rn(__fdiv_rn(cst, delt), ras); // L3312
    ec1 = __fmul_rn(__fmul_rn(q1, wetcan), vegfrac); // L3313
    cst = ruc_max<float>(0.f, __fsub_rn(cst, __fmul_rn(ec1, delt))); // L3315
    if ((globalcellid == tb.targetcell)) { // L3318
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3319
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3320
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3321
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3322
    } // L3323
    if (myj) { // L3325
    eeta = __fmul_rn((-__fmul_rn(__fmul_rn(__fmul_rn(qkms, ras), __fsub_rn(__fdiv_rn(qvatm, __fadd_rn(1.f, qvatm)), __fdiv_rn(qsg, __fadd_rn(1.f, qsg)))), 1.e3f)), beta); // L3327
    if (false) { // L3328
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3329
    } // L3330
    } else { // L3331
    eeta = __fmul_rn(__fadd_rn(__fadd_rn(edir1, ec1), ett1), 1.e3f); // L3334
    if (false) { // L3335
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3336
    } // L3337
    } // L3338
    qfx = __fmul_rn(xlvm, eeta); // L3339
    eeta = __fmul_rn(__fadd_rn(__fadd_rn(edir1, ec1), ett1), 1.e3f); // L3340
    } // L3341
    s = snflx; // L3342
    sublim = __fmul_rn(edir1, 1.e3f); // L3343
    fltot = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(rnet, hft), __fmul_rn(xlvm, eeta)), s), snoh), x); // L3345
    if ((globalcellid == tb.targetcell)) { // L3347
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3348
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3349
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3350
    } // L3352
    label_222: ;
    ; // L3354
    label_1123: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3356
    label_1133: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3357
    label_123: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3358
    label_122: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3359
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:3365-3673
RUCDEV void land_soiltemp(Tables &tb, Arena &arena, int  i, int  j, int  iland, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, int  nroot, float  prcpms, float  rainf, float  patm, float  tabs, float  qvatm, float  qcatm, float  emiss, float  rnet, float  qkms, float  tkms, float  pc, float  rho, float  vegfrac, float  lai, FArray<float> thdif, FArray<float> cap, float & drycan, float & wetcan, float & transum, float  dew, float & mavail, float  soilres, float  alfa, int  globalid, float  dqm, float  qmin, float  bclh, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> tbq, float  xlv, float  cp, float  g0_p, float  cvw, float  stbolt, FArray<float> tso, float & soilt, float & qvg, float & qsg, float & qcg, float & x) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float x1;
    float x2;
    float x4;
    float dzstop;
    float can;
    float ft;
    float sph;
    float tn;
    float trans;
    float umveg;
    float denom;
    float fex;
    float fkt;
    float d1;
    float d2;
    float d9;
    float d10;
    float did;
    float r211;
    float r21;
    float r22;
    float r6;
    float r7;
    float d11;
    float pi;
    float h;
    float fkq;
    float r210;
    float aa;
    float bb;
    float pp;
    float q1;
    float qs1;
    float ts1;
    float tq2;
    float tx2;
    float tdenom;
    float c;
    float cc;
    float aa1;
    float rhcs;
    float h1;
    float qgold;
    FArray<float> cotso = arena.alloc<float>(1, nzs);
    FArray<float> rhtso = arena.alloc<float>(1, nzs);
    int nzs1;
    int nzs2;
    int k;
    int k1;
    int kn;
    int kk;
    int iter;
    iter = 0; // L3521
    nzs1 = (nzs - 1); // L3523
    nzs2 = (nzs - 2); // L3524
    dzstop = __fdiv_rn(1.f, __fsub_rn(zsmain(2), zsmain(1))); // L3525
    qgold = qvg; // L3527
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L3529
    cotso(k) = 0.f; // L3530
    rhtso(k) = 0.f; // L3531
    loop1_continue: ;
    } loop1_exit: ; // L3532
    cotso(1) = 0.f; // L3541
    rhtso(1) = tso(nzs); // L3542
    for (k = 1; k <= nzs2; k += 1) { loop2_cycle: ; // L3543
    kn = (nzs - k); // L3544
    k1 = ((2 * kn) - 3); // L3545
    x1 = __fmul_rn(dtdzs(k1), thdif((kn - 1))); // L3546
    x2 = __fmul_rn(dtdzs((k1 + 1)), thdif(kn)); // L3547
    ft = __fsub_rn(__fadd_rn(tso(kn), __fmul_rn(x1, __fsub_rn(tso((kn - 1)), tso(kn)))), __fmul_rn(x2, __fsub_rn(tso(kn), tso((kn + 1))))); // L3548
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1), x2), __fmul_rn(x2, cotso(k))); // L3550
    cotso((k + 1)) = __fdiv_rn(x1, denom); // L3551
    rhtso((k + 1)) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso(k))), denom); // L3552
    loop2_continue: ;
    } loop2_exit: ; // L3553
    if ((globalid == tb.targetcell)) { // L3554
    for (k = 1; k <= nzs; k += 1) { loop3_cycle: ; // L3555
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3556
    loop3_continue: ;
    } loop3_exit: ; // L3557
    } // L3558
    rhcs = cap(1); // L3562
    h = mavail; // L3564
    trans = __fdiv_rn(__fmul_rn(transum, drycan), zshalf((nroot + 1))); // L3566
    can = __fadd_rn(wetcan, trans); // L3567
    umveg = __fmul_rn(__fsub_rn(1.f, vegfrac), soilres); // L3568
    label_2111: ;
    ; // L3569
    fkt = tkms; // L3570
    d1 = cotso(nzs1); // L3571
    d2 = rhtso(nzs1); // L3572
    tn = soilt; // L3573
    d9 = __fmul_rn(__fmul_rn(thdif(1), rhcs), dzstop); // L3574
    d10 = __fmul_rn(__fmul_rn(tkms, cp), rho); // L3575
    r211 = __fdiv_rn(__fmul_rn(.5f, conflx), delt); // L3576
    r21 = __fmul_rn(__fmul_rn(r211, cp), rho); // L3577
    r22 = __fdiv_rn(.5f, __fmul_rn(__fmul_rn(thdif(1), delt), __fmul_rn(dzstop, dzstop))); // L3578
    r6 = __fmul_rn(__fmul_rn(__fmul_rn(emiss, stbolt), .5f), __fmul_rn(__fmul_rn(tn, tn), __fmul_rn(tn, tn))); // L3579
    r7 = __fdiv_rn(r6, tn); // L3580
    d11 = __fadd_rn(rnet, r6); // L3581
    tdenom = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d9, __fadd_rn(__fsub_rn(1.f, d1), r22)), d10), r21), r7), __fmul_rn(__fmul_rn(rainf, cvw), prcpms)); // L3582
    fkq = __fmul_rn(qkms, rho); // L3584
    r210 = __fmul_rn(r211, rho); // L3585
    c = __fmul_rn(__fmul_rn(vegfrac, fkq), can); // L3586
    cc = __fdiv_rn(__fmul_rn(c, xlv), tdenom); // L3587
    aa = __fdiv_rn(__fmul_rn(xlv, __fadd_rn(__fmul_rn(fkq, umveg), r210)), tdenom); // L3588
    bb = __fdiv_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d10, tabs), __fmul_rn(r21, tn)), __fmul_rn(xlv, __fadd_rn(__fmul_rn(qvatm, __fadd_rn(__fmul_rn(fkq, umveg), c)), __fmul_rn(r210, qvg)))), d11), __fmul_rn(d9, __fadd_rn(d2, __fmul_rn(r22, tn)))), __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), ruc_max<float>(273.15f, tabs))), tdenom); // L3589
    aa1 = __fadd_rn(aa, cc); // L3594
    pp = __fmul_rn(patm, 1.e3f); // L3596
    aa1 = __fdiv_rn(aa1, pp); // L3597
    land_vilka(tb, arena, tn, aa1, bb, pp, qs1, ts1, tbq, ktau, i, j, iland, isoil); // L3598
    tq2 = qvatm; // L3599
    tx2 = __fmul_rn(tq2, __fsub_rn(1.f, h)); // L3600
    q1 = __fadd_rn(tx2, __fmul_rn(h, qs1)); // L3601
    if (false) { // L3603
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3604
    } // L3605
    if ((q1 < qs1)) { goto label_100; } // L3607
    label_90: ;
    qvg = qs1; // L3610
    qsg = qs1; // L3611
    tso(1) = ts1; // L3612
    qcg = ruc_max<float>(0.f, __fsub_rn(q1, qs1)); // L3613
    if (false) { // L3614
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3615
    } // L3616
    goto label_200; // L3617
    label_100: ;
    bb = __fsub_rn(bb, __fmul_rn(aa, tx2)); // L3618
    aa = __fdiv_rn(__fadd_rn(__fmul_rn(aa, h), cc), pp); // L3619
    land_vilka(tb, arena, tn, aa, bb, pp, qs1, ts1, tbq, ktau, i, j, iland, isoil); // L3620
    q1 = __fadd_rn(tx2, __fmul_rn(h, qs1)); // L3621
    if (false) { // L3622
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3623
    } // L3624
    if ((q1 >= qs1)) { goto label_90; } // L3625
    qsg = qs1; // L3627
    qvg = q1; // L3628
    tso(1) = ts1; // L3634
    qcg = 0.f; // L3635
    if (false) { // L3636
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3637
    } // L3638
    label_200: ;
    ; // L3639
    if (false) { // L3640
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3641
    } // L3642
    soilt = ts1; // L3645
    for (k = 2; k <= nzs; k += 1) { loop4_cycle: ; // L3648
    kk = ((nzs - k) + 1); // L3649
    tso(k) = __fadd_rn(rhtso(kk), __fmul_rn(cotso(kk), tso((k - 1)))); // L3650
    loop4_continue: ;
    } loop4_exit: ; // L3651
    x = __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fmul_rn(cp, rho), r211), __fdiv_rn(__fmul_rn(__fmul_rn(rhcs, zsmain(2)), 0.5f), delt)), __fsub_rn(soilt, tn)), __fmul_rn(__fmul_rn(__fmul_rn(xlv, rho), r211), __fsub_rn(qvg, qgold))); // L3653
    if ((globalid == tb.targetcell)) { // L3657
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3658
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3660
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3662
    } // L3663
    x = __fsub_rn(x, __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), __fsub_rn(ruc_max<float>(273.15f, tabs), soilt))); // L3664
    if (false) { // L3668
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3669
    } // L3670
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:3677-4595
RUCDEV void land_snowtemp(Tables &tb, Arena &arena, int  i, int  j, int  iland, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, int  nroot, float & snwe, float  snwepr, float & snhei, float  newsnow, float & snowfrac, float & beta, float  deltsn, float  snth, float & rhosn, float  rhonewsn, float  meltfactor, float  prcpms, float  rainf, float  patm, float  tabs, float  qvatm, float  qcatm, float  glw, float  gsw, float & emiss, float & rnet, float  qkms, float  tkms, float  pc, float  rho, float  vegfrac, FArray<float> thdif, FArray<float> cap, float & drycan, float & wetcan, float & cst, FArray<float> tranf, float  transum, float & dew, float & mavail, int  globalcellid, float  dqm, float  qmin, float  psis, float  bclh, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> tbq, float  xlvm, float  cp, float  rovcp, float  g0_p, float  cvw, float  stbolt, float & snweprint, float & snheiprint, float & rsm, FArray<float> tso, float & soilt, float & soilt1, float & tsnav, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & s, int & ilnb, float & x) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float rhonewcsn;
    int nzs1;
    int nzs2;
    int k;
    int k1;
    int kn;
    int kk;
    float x1;
    float x2;
    float x4;
    float dzstop;
    float can;
    float ft;
    float sph;
    float tn;
    float trans;
    float umveg;
    float denom;
    float cotsn;
    float rhtsn;
    float xsn1;
    float ddzsn1;
    float x1sn1;
    float ftsnow;
    float denomsn;
    float t3;
    float upflux;
    float xinet;
    float ras;
    float xlmelt;
    float rhocsn;
    float thdifsn;
    float epot;
    float xsn;
    float ddzsn;
    float x1sn;
    float d1sn;
    float d2sn;
    float d9sn;
    float r22sn;
    float fso;
    float fsn;
    float fkt;
    float d1;
    float d2;
    float d9;
    float d10;
    float did;
    float r211;
    float r21;
    float r22;
    float r6;
    float r7;
    float d11;
    float pi;
    float h;
    float fkq;
    float r210;
    float aa;
    float bb;
    float pp;
    float q1;
    float qs1;
    float ts1;
    float tq2;
    float tx2;
    float tdenom;
    float c;
    float cc;
    float aa1;
    float rhcs;
    float h1;
    float tsob;
    float snprim;
    float sh1;
    float sh2;
    float smeltg;
    float snohg;
    float snodif;
    float soh;
    float cmc2ms;
    float tnold;
    float qgold;
    float snohgnew;
    FArray<float> transp = arena.alloc<float>(1, nzs);
    FArray<float> cotso = arena.alloc<float>(1, nzs);
    FArray<float> rhtso = arena.alloc<float>(1, nzs);
    float edir1;
    float ec1;
    float ett1;
    float eeta;
    float qfx;
    float hfx;
    float rsmfrac;
    float soiltfrac;
    float hsn;
    float rr;
    float keff;
    float fact;
    int nmelt;
    int iter;
    if (arena.snowtemp_calls) ++*arena.snowtemp_calls;
    if (arena.entry_ilnb) ilnb = *arena.entry_ilnb;
    iter = 0; // L3871
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L3873
    transp(k) = 0.f; // L3874
    cotso(k) = 0.f; // L3875
    rhtso(k) = 0.f; // L3876
    loop1_continue: ;
    } loop1_exit: ; // L3877
    keff = 0.265f; // L3881
    if (false) { // L3883
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3884
    } // L3885
    xlmelt = 3.35e+5f; // L3886
    rhocsn = __fmul_rn(2090.f, rhosn); // L3887
    rhonewcsn = __fmul_rn(2090.f, rhonewsn); // L3889
    if ((isncond_opt == 1)) { // L3891
    thdifsn = __fdiv_rn(0.265f, rhocsn); // L3893
    } else { // L3894
    fact = 1.f; // L3897
    if (((rhosn < 156.f) || ((newsnow > 0.f) && (rhonewsn < 156.f)))) { // L3898
    keff = __fadd_rn(0.023f, __fmul_rn(__fmul_rn(0.234f, rhosn), 1.e-3f)); // L3899
    } else { // L3900
    keff = __fadd_rn(__fsub_rn(0.138f, __fmul_rn(__fmul_rn(1.01f, rhosn), 1.e-3f)), __fmul_rn(__fmul_rn(3.233f, __fmul_rn(rhosn, rhosn)), 1.e-6f)); // L3901
    } // L3902
    if ((((newsnow <= 0.f) && (snhei > 1.f)) && (rhosn > 250.f))) { // L3903
    thdifsn = 4.431718e-7f; // L3909
    } else { // L3910
    thdifsn = __fmul_rn(__fdiv_rn(keff, rhocsn), fact); // L3911
    } // L3912
    } // L3913
    ras = __fmul_rn(rho, 1.e-3f); // L3915
    soiltfrac = soilt; // L3917
    smelt = 0.f; // L3919
    soh = 0.f; // L3920
    smeltg = 0.f; // L3921
    snohg = 0.f; // L3922
    snodif = 0.f; // L3923
    rsm = 0.f; // L3924
    rsmfrac = 0.f; // L3925
    fsn = 1.f; // L3926
    fso = 0.f; // L3927
    nzs1 = (nzs - 1); // L3929
    nzs2 = (nzs - 2); // L3930
    qgold = qvg; // L3932
    dzstop = __fdiv_rn(1.f, __fsub_rn(zsmain(2), zsmain(1))); // L3933
    cotso(1) = 0.f; // L3944
    rhtso(1) = tso(nzs); // L3945
    for (k = 1; k <= nzs2; k += 1) { loop2_cycle: ; // L3946
    kn = (nzs - k); // L3947
    k1 = ((2 * kn) - 3); // L3948
    x1 = __fmul_rn(dtdzs(k1), thdif((kn - 1))); // L3949
    x2 = __fmul_rn(dtdzs((k1 + 1)), thdif(kn)); // L3950
    ft = __fsub_rn(__fadd_rn(tso(kn), __fmul_rn(x1, __fsub_rn(tso((kn - 1)), tso(kn)))), __fmul_rn(x2, __fsub_rn(tso(kn), tso((kn + 1))))); // L3951
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1), x2), __fmul_rn(x2, cotso(k))); // L3953
    cotso((k + 1)) = __fdiv_rn(x1, denom); // L3954
    rhtso((k + 1)) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso(k))), denom); // L3955
    loop2_continue: ;
    } loop2_exit: ; // L3956
    if ((snhei >= snth)) { // L3959
    if ((snhei <= __fadd_rn(deltsn, snth))) { // L3960
    if ((globalcellid == tb.targetcell)) { // L3963
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3964
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3965
    } // L3966
    ilnb = 1; // L3967
    snprim = ruc_max<float>(snth, snhei); // L3968
    tsob = tso(1); // L3969
    soilt1 = tso(1); // L3970
    xsn = __fdiv_rn(__fdiv_rn(delt, 2.f), __fadd_rn(zshalf(2), __fmul_rn(0.5f, snprim))); // L3971
    ddzsn = __fdiv_rn(xsn, snprim); // L3972
    x1sn = __fmul_rn(ddzsn, thdifsn); // L3973
    x2 = __fmul_rn(dtdzs(1), thdif(1)); // L3974
    ft = __fsub_rn(__fadd_rn(tso(1), __fmul_rn(x1sn, __fsub_rn(soilt, tso(1)))), __fmul_rn(x2, __fsub_rn(tso(1), tso(2)))); // L3975
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1sn), x2), __fmul_rn(x2, cotso(nzs1))); // L3977
    cotso(nzs) = __fdiv_rn(x1sn, denom); // L3978
    rhtso(nzs) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso(nzs1))), denom); // L3979
    cotsn = cotso(nzs); // L3980
    rhtsn = rhtso(nzs); // L3981
    tsnav = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(soilt, tso(1))), 273.15f); // L3983
    } else { // L3986
    if ((globalcellid == tb.targetcell)) { // L3989
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3990
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L3991
    } // L3992
    ilnb = 2; // L3993
    snprim = deltsn; // L3994
    tsob = soilt1; // L3995
    xsn = __fdiv_rn(__fdiv_rn(delt, 2.f), __fmul_rn(0.5f, deltsn)); // L3996
    xsn1 = __fdiv_rn(__fdiv_rn(delt, 2.f), __fadd_rn(zshalf(2), __fmul_rn(0.5f, __fsub_rn(snhei, deltsn)))); // L3997
    ddzsn = __fdiv_rn(xsn, deltsn); // L3998
    ddzsn1 = __fdiv_rn(xsn1, __fsub_rn(snhei, deltsn)); // L3999
    x1sn = __fmul_rn(ddzsn, thdifsn); // L4000
    x1sn1 = __fmul_rn(ddzsn1, thdifsn); // L4001
    x2 = __fmul_rn(dtdzs(1), thdif(1)); // L4002
    ft = __fsub_rn(__fadd_rn(tso(1), __fmul_rn(x1sn1, __fsub_rn(soilt1, tso(1)))), __fmul_rn(x2, __fsub_rn(tso(1), tso(2)))); // L4003
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1sn1), x2), __fmul_rn(x2, cotso(nzs1))); // L4005
    cotso(nzs) = __fdiv_rn(x1sn1, denom); // L4006
    rhtso(nzs) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso(nzs1))), denom); // L4007
    ftsnow = __fsub_rn(__fadd_rn(soilt1, __fmul_rn(x1sn, __fsub_rn(soilt, soilt1))), __fmul_rn(x1sn1, __fsub_rn(soilt1, tso(1)))); // L4008
    denomsn = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1sn), x1sn1), __fmul_rn(x1sn1, cotso(nzs))); // L4010
    cotsn = __fdiv_rn(x1sn, denomsn); // L4011
    rhtsn = __fdiv_rn(__fadd_rn(ftsnow, __fmul_rn(x1sn1, rhtso(nzs))), denomsn); // L4012
    tsnav = __fsub_rn(__fmul_rn(__fdiv_rn(0.5f, snhei), __fadd_rn(__fmul_rn(__fadd_rn(soilt, soilt1), deltsn), __fmul_rn(__fadd_rn(soilt1, tso(1)), __fsub_rn(snhei, deltsn)))), 273.15f); // L4014
    } // L4017
    } // L4018
    if (((snhei < snth) && (snhei > 0.f))) { // L4019
    snprim = __fadd_rn(snhei, zsmain(2)); // L4022
    fsn = __fdiv_rn(snhei, snprim); // L4023
    fso = __fsub_rn(1.f, fsn); // L4024
    soilt1 = tso(1); // L4025
    tsob = tso(2); // L4026
    xsn = __fdiv_rn(__fdiv_rn(delt, 2.f), __fadd_rn(__fsub_rn(zshalf(3), zsmain(2)), __fmul_rn(0.5f, snprim))); // L4027
    ddzsn = __fdiv_rn(xsn, snprim); // L4028
    x1sn = __fmul_rn(ddzsn, __fadd_rn(__fmul_rn(fsn, thdifsn), __fmul_rn(fso, thdif(1)))); // L4029
    x2 = __fmul_rn(dtdzs(2), thdif(2)); // L4030
    ft = __fsub_rn(__fadd_rn(tso(2), __fmul_rn(x1sn, __fsub_rn(soilt, tso(2)))), __fmul_rn(x2, __fsub_rn(tso(2), tso(3)))); // L4031
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1sn), x2), __fmul_rn(x2, cotso((nzs - 2)))); // L4033
    cotso(nzs1) = __fdiv_rn(x1sn, denom); // L4034
    rhtso(nzs1) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso((nzs - 2)))), denom); // L4035
    tsnav = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(soilt, tso(1))), 273.15f); // L4036
    cotso(nzs) = cotso(nzs1); // L4038
    rhtso(nzs) = rhtso(nzs1); // L4039
    cotsn = cotso(nzs); // L4040
    rhtsn = rhtso(nzs); // L4041
    } // L4043
    nmelt = 0; // L4048
    snoh = 0.f; // L4049
    ett1 = 0.f; // L4051
    epot = __fmul_rn((-qkms), __fsub_rn(qvatm, qgold)); // L4052
    rhcs = cap(1); // L4053
    h = mavail; // L4054
    trans = __fdiv_rn(__fmul_rn(transum, drycan), zshalf((nroot + 1))); // L4055
    can = __fadd_rn(wetcan, trans); // L4056
    umveg = __fsub_rn(1.f, vegfrac); // L4057
    fkt = tkms; // L4058
    d1 = cotso(nzs1); // L4059
    d2 = rhtso(nzs1); // L4060
    tn = soilt; // L4061
    d9 = __fmul_rn(__fmul_rn(thdif(1), rhcs), dzstop); // L4062
    d10 = __fmul_rn(__fmul_rn(tkms, cp), rho); // L4063
    r211 = __fdiv_rn(__fmul_rn(.5f, conflx), delt); // L4064
    r21 = __fmul_rn(__fmul_rn(r211, cp), rho); // L4065
    r22 = __fdiv_rn(.5f, __fmul_rn(__fmul_rn(thdif(1), delt), __fmul_rn(dzstop, dzstop))); // L4066
    r6 = __fmul_rn(__fmul_rn(__fmul_rn(emiss, stbolt), .5f), __fmul_rn(__fmul_rn(tn, tn), __fmul_rn(tn, tn))); // L4067
    r7 = __fdiv_rn(r6, tn); // L4068
    d11 = __fadd_rn(rnet, r6); // L4069
    if ((snhei >= snth)) { // L4071
    if ((snhei <= __fadd_rn(deltsn, snth))) { // L4072
    d1sn = cotso(nzs); // L4074
    d2sn = rhtso(nzs); // L4075
    if (false) { // L4076
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4077
    } // L4078
    } else { // L4079
    d1sn = cotsn; // L4081
    d2sn = rhtsn; // L4082
    if ((globalcellid == tb.targetcell)) { // L4084
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4085
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4086
    } // L4087
    } // L4088
    d9sn = __fdiv_rn(__fmul_rn(thdifsn, rhocsn), snprim); // L4089
    r22sn = __fdiv_rn(__fmul_rn(__fmul_rn(snprim, snprim), 0.5f), __fmul_rn(thdifsn, delt)); // L4090
    if (false) { // L4091
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4092
    } // L4093
    } // L4094
    if (((snhei < snth) && (snhei > 0.f))) { // L4096
    d1sn = d1; // L4098
    d2sn = d2; // L4099
    d9sn = __fdiv_rn(__fadd_rn(__fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn), __fmul_rn(__fmul_rn(fso, thdif(1)), rhcs)), snprim); // L4100
    r22sn = __fdiv_rn(__fmul_rn(__fmul_rn(snprim, snprim), 0.5f), __fmul_rn(__fadd_rn(__fmul_rn(fsn, thdifsn), __fmul_rn(fso, thdif(1))), delt)); // L4102
    if (false) { // L4104
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4105
    } // L4106
    } // L4107
    if ((snhei == 0.f)) { // L4108
    d9sn = d9; // L4110
    r22sn = r22; // L4111
    d1sn = d1; // L4112
    d2sn = d2; // L4113
    if (false) { // L4114
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4115
    } // L4116
    } // L4117
    label_2211: ;
    ; // L4119
    label_212: ;
    ; // L4122
    tdenom = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d9sn, __fadd_rn(__fsub_rn(1.f, d1sn), r22sn)), d10), r21), r7), __fmul_rn(__fmul_rn(rainf, cvw), prcpms)), __fdiv_rn(__fmul_rn(rhonewcsn, newsnow), delt)); // L4125
    fkq = __fmul_rn(qkms, rho); // L4129
    r210 = __fmul_rn(r211, rho); // L4130
    c = __fmul_rn(__fmul_rn(vegfrac, fkq), can); // L4131
    cc = __fdiv_rn(__fmul_rn(c, xlvm), tdenom); // L4132
    aa = __fdiv_rn(__fmul_rn(xlvm, __fadd_rn(__fmul_rn(__fmul_rn(beta, fkq), umveg), r210)), tdenom); // L4133
    bb = __fdiv_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d10, tabs), __fmul_rn(r21, tn)), __fmul_rn(xlvm, __fadd_rn(__fmul_rn(qvatm, __fadd_rn(__fmul_rn(__fmul_rn(beta, fkq), umveg), c)), __fmul_rn(r210, qgold)))), d11), __fmul_rn(d9sn, __fadd_rn(d2sn, __fmul_rn(r22sn, tn)))), __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), ruc_max<float>(273.15f, tabs))), __fmul_rn(__fdiv_rn(__fmul_rn(rhonewcsn, newsnow), delt), ruc_min<float>(273.15f, tabs))), tdenom); // L4134
    aa1 = __fadd_rn(aa, cc); // L4140
    pp = __fmul_rn(patm, 1.e3f); // L4141
    aa1 = __fdiv_rn(aa1, pp); // L4142
    bb = __fsub_rn(bb, __fdiv_rn(snoh, tdenom)); // L4143
    land_vilka(tb, arena, tn, aa1, bb, pp, qs1, ts1, tbq, ktau, i, j, iland, isoil); // L4145
    tq2 = qvatm; // L4146
    tx2 = __fmul_rn(tq2, __fsub_rn(1.f, h)); // L4147
    q1 = __fadd_rn(tx2, __fmul_rn(h, qs1)); // L4148
    if (false) { // L4149
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4150
    } // L4151
    if ((q1 < qs1)) { goto label_100; } // L4152
    label_90: ;
    qvg = qs1; // L4155
    qsg = qs1; // L4156
    qcg = ruc_max<float>(0.f, __fsub_rn(q1, qs1)); // L4157
    if (false) { // L4158
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4159
    } // L4160
    goto label_200; // L4161
    label_100: ;
    bb = __fsub_rn(bb, __fmul_rn(aa, tx2)); // L4162
    aa = __fdiv_rn(__fadd_rn(__fmul_rn(aa, h), cc), pp); // L4163
    land_vilka(tb, arena, tn, aa, bb, pp, qs1, ts1, tbq, ktau, i, j, iland, isoil); // L4164
    q1 = __fadd_rn(tx2, __fmul_rn(h, qs1)); // L4165
    if (false) { // L4166
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4167
    } // L4168
    if ((q1 > qs1)) { goto label_90; } // L4169
    qsg = qs1; // L4170
    qvg = q1; // L4171
    qcg = 0.f; // L4172
    if (false) { // L4173
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4174
    } // L4175
    label_200: ;
    ; // L4176
    soilt = ts1; // L4179
    if (((((nmelt == 1) && (snowfrac == 1.f)) && (snwe > 0.f)) && (soilt > 273.15f))) { // L4181
    soilt = ruc_min<float>(273.15f, soilt); // L4186
    } // L4187
    if (false) { // L4189
    if (((i == 266) && (j == 447))) { // L4190
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4191
    } // L4192
    } // L4193
    if ((snhei >= snth)) { // L4195
    if ((snhei > __fadd_rn(deltsn, snth))) { // L4196
    soilt1 = ruc_min<float>(273.15f, __fadd_rn(rhtsn, __fmul_rn(cotsn, soilt))); // L4198
    tso(1) = __fadd_rn(rhtso(nzs), __fmul_rn(cotso(nzs), soilt1)); // L4199
    tsob = soilt1; // L4200
    } else { // L4201
    tso(1) = __fadd_rn(rhtso(nzs), __fmul_rn(cotso(nzs), soilt)); // L4203
    soilt1 = tso(1); // L4204
    tsob = tso(1); // L4205
    } // L4206
    } else if (((snhei > 0.f) && (snhei < snth))) { // L4207
    tso(2) = __fadd_rn(rhtso(nzs1), __fmul_rn(cotso(nzs1), soilt)); // L4209
    tso(1) = __fadd_rn(tso(2), __fmul_rn(__fsub_rn(soilt, tso(2)), fso)); // L4210
    soilt1 = tso(1); // L4211
    tsob = tso(2); // L4212
    } else { // L4213
    tso(1) = soilt; // L4216
    soilt1 = soilt; // L4217
    tsob = tso(1); // L4218
    } // L4219
    if (((nmelt == 1) && (snowfrac == 1.f))) { // L4220
    soilt1 = ruc_min<float>(273.15f, soilt1); // L4222
    tso(1) = ruc_min<float>(273.15f, tso(1)); // L4223
    tsob = ruc_min<float>(273.15f, tsob); // L4224
    } // L4225
    if (((snhei > 0.f) && (snhei < snth))) { // L4229
    for (k = 3; k <= nzs; k += 1) { loop3_cycle: ; // L4231
    kk = ((nzs - k) + 1); // L4232
    tso(k) = __fadd_rn(rhtso(kk), __fmul_rn(cotso(kk), tso((k - 1)))); // L4233
    loop3_continue: ;
    } loop3_exit: ; // L4234
    } else { // L4236
    for (k = 2; k <= nzs; k += 1) { loop4_cycle: ; // L4237
    kk = ((nzs - k) + 1); // L4238
    tso(k) = __fadd_rn(rhtso(kk), __fmul_rn(cotso(kk), tso((k - 1)))); // L4239
    loop4_continue: ;
    } loop4_exit: ; // L4240
    } // L4241
    if (false) { // L4252
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4253
    } // L4254
    if ((nmelt == 1)) { goto label_220; } // L4256
    if ((((soilt > 273.15f) && (beta == 1.f)) && (snhei > 0.f))) { // L4261
    nmelt = 1; // L4262
    soiltfrac = __fadd_rn(__fmul_rn(snowfrac, 273.15f), __fmul_rn(__fsub_rn(1.f, snowfrac), soilt)); // L4263
    qsg = ruc_min<float>(qsg, __fdiv_rn(land_qsn(tb, arena, soiltfrac, tbq), pp)); // L4264
    qvg = __fadd_rn(__fmul_rn(snowfrac, qsg), __fmul_rn(__fsub_rn(1.f, snowfrac), qvg)); // L4265
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, tn), tn), tn); // L4266
    upflux = __fmul_rn(__fmul_rn(t3, 0.5f), __fadd_rn(tn, soiltfrac)); // L4267
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L4268
    epot = __fmul_rn((-qkms), __fsub_rn(qvatm, qsg)); // L4269
    q1 = __fmul_rn(epot, ras); // L4270
    if (((q1 <= 0.f) || (iter == 1))) { // L4272
    dew = (-epot); // L4274
    for (k = 1; k <= nzs; k += 1) { loop5_cycle: ; // L4275
    transp(k) = 0.f; // L4276
    loop5_continue: ;
    } loop5_exit: ; // L4277
    qfx = __fmul_rn(__fmul_rn((-xlvm), rho), dew); // L4279
    eeta = __fdiv_rn(qfx, xlvm); // L4280
    } else { // L4281
    for (k = 1; k <= nroot; k += 1) { loop6_cycle: ; // L4283
    transp(k) = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn((-vegfrac), q1), tranf(k)), drycan), zshalf((nroot + 1))); // L4284
    ett1 = __fsub_rn(ett1, transp(k)); // L4286
    loop6_continue: ;
    } loop6_exit: ; // L4287
    for (k = (nroot + 1); k <= nzs; k += 1) { loop7_cycle: ; // L4288
    transp(k) = 0.f; // L4289
    loop7_continue: ;
    } loop7_exit: ; // L4290
    edir1 = __fmul_rn(__fmul_rn(q1, umveg), beta); // L4292
    ec1 = __fmul_rn(__fmul_rn(q1, wetcan), vegfrac); // L4293
    cmc2ms = __fmul_rn(__fdiv_rn(cst, delt), ras); // L4294
    eeta = __fmul_rn(__fadd_rn(__fadd_rn(edir1, ec1), ett1), 1.e3f); // L4295
    qfx = __fmul_rn(xlvm, eeta); // L4297
    } // L4298
    hfx = __fmul_rn((-d10), __fsub_rn(tabs, soiltfrac)); // L4300
    if ((snhei >= snth)) { // L4302
    soh = __fdiv_rn(__fmul_rn(__fmul_rn(thdifsn, rhocsn), __fsub_rn(soiltfrac, tsob)), snprim); // L4303
    snflx = soh; // L4304
    } else { // L4305
    soh = __fdiv_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn), __fmul_rn(__fmul_rn(fso, thdif(1)), rhcs)), __fsub_rn(soiltfrac, tsob)), snprim); // L4306
    snflx = soh; // L4308
    } // L4309
    x = __fadd_rn(__fmul_rn(__fadd_rn(r21, __fmul_rn(d9sn, r22sn)), __fsub_rn(soiltfrac, tn)), __fmul_rn(__fmul_rn(xlvm, r210), __fsub_rn(qvg, qgold))); // L4312
    if ((globalcellid == tb.targetcell)) { // L4315
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4316
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4317
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4318
    } // L4320
    snoh = __fadd_rn(__fadd_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(rnet, qfx), hfx), soh), x), __fmul_rn(__fdiv_rn(__fmul_rn(rhonewcsn, newsnow), delt), __fsub_rn(ruc_min<float>(273.15f, tabs), soiltfrac))), __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), __fsub_rn(ruc_max<float>(273.15f, tabs), soiltfrac))); // L4323
    snoh = ruc_max<float>(0.f, snoh); // L4326
    smelt = __fmul_rn(__fdiv_rn(snoh, xlmelt), 1.e-3f); // L4328
    if ((globalcellid == tb.targetcell)) { // L4330
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4331
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4332
    } // L4333
    if (((epot > 0.f) && (snwepr <= __fmul_rn(__fmul_rn(epot, ras), delt)))) { // L4334
    beta = __fdiv_rn(snwepr, __fmul_rn(__fmul_rn(epot, ras), delt)); // L4336
    smelt = ruc_min<float>(smelt, __fsub_rn(__fdiv_rn(snwepr, delt), __fmul_rn(__fmul_rn(beta, epot), ras))); // L4337
    snwe = 0.f; // L4338
    if ((globalcellid == tb.targetcell)) { // L4340
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4341
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4342
    } // L4343
    goto label_88; // L4344
    } // L4345
    smelt = ruc_max<float>(0.f, smelt); // L4347
    if ((((rhosn < 350.f) || ((newsnow > 0.f) && (rhonewsn < 450.f))) && (soilt < 283.f))) { // L4352
    smelt = ruc_min<float>(smelt, __fmul_rn(__fmul_rn(__fmul_rn(__fdiv_rn(delt, 60.f), 5.6e-8f), meltfactor), ruc_max<float>(1.f, __fsub_rn(soilt, 273.15f)))); // L4354
    if ((globalcellid == tb.targetcell)) { // L4357
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4358
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4359
    } // L4360
    } // L4361
    rr = ruc_max<float>(0.f, __fsub_rn(__fdiv_rn(snwepr, delt), __fmul_rn(__fmul_rn(beta, epot), ras))); // L4364
    if ((smelt > rr)) { // L4365
    smelt = ruc_min<float>(smelt, rr); // L4366
    snwe = 0.f; // L4367
    if ((globalcellid == tb.targetcell)) { // L4369
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4370
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4371
    } // L4372
    } // L4373
    label_88: ;
    ; // L4375
    snohgnew = __fmul_rn(__fmul_rn(smelt, xlmelt), 1.e3f); // L4376
    snodif = ruc_max<float>(0.f, __fsub_rn(snoh, snohgnew)); // L4377
    snoh = snohgnew; // L4379
    if ((globalcellid == tb.targetcell)) { // L4381
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4382
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4383
    } // L4384
    if ((smelt > 0.f)) { // L4386
    rsmfrac = ruc_min<float>(0.18f, ruc_max<float>(0.08f, __fmul_rn(__fdiv_rn(snwepr, 0.10f), 0.13f))); // L4388
    if (((snhei > 0.01f) && (rhosn < 350.f))) { // L4389
    rsm = __fmul_rn(__fmul_rn(rsmfrac, smelt), delt); // L4390
    } else { // L4391
    rsm = 0.f; // L4393
    } // L4394
    if ((rsm > 0.f)) { // L4396
    smelt = ruc_max<float>(0.f, __fsub_rn(smelt, __fdiv_rn(rsm, delt))); // L4397
    if ((globalcellid == tb.targetcell)) { // L4399
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4400
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4401
    } // L4403
    } // L4404
    } // L4406
    if ((snwe > 0.f)) { // L4410
    snwe = ruc_max<float>(0.f, __fsub_rn(snwepr, __fmul_rn(__fadd_rn(smelt, __fmul_rn(__fmul_rn(beta, epot), ras)), delt))); // L4411
    } // L4414
    } else { // L4418
    if (((snhei != 0.f) && (beta == 1.f))) { // L4419
    epot = __fmul_rn((-qkms), __fsub_rn(qvatm, qsg)); // L4420
    snwe = ruc_max<float>(0.f, __fsub_rn(snwepr, __fmul_rn(__fmul_rn(__fmul_rn(beta, epot), ras), delt))); // L4421
    } else { // L4423
    snwe = 0.f; // L4425
    } // L4426
    } // L4428
    if ((nmelt == 1)) { goto label_212; } // L4430
    label_220: ;
    ; // L4431
    if (((smelt > 0.f) && (rsm > 0.f))) { // L4433
    if ((snwe <= rsm)) { // L4434
    if ((globalcellid == tb.targetcell)) { // L4436
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4437
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4438
    } // L4440
    } else { // L4441
    xsn = __fdiv_rn(__fadd_rn(__fmul_rn(rhosn, __fsub_rn(snwe, rsm)), __fmul_rn(1.e3f, rsm)), snwe); // L4446
    rhosn = ruc_min<float>(ruc_max<float>(58.8f, xsn), 500.f); // L4448
    rhocsn = __fmul_rn(2090.f, rhosn); // L4450
    if ((isncond_opt == 1)) { // L4451
    thdifsn = __fdiv_rn(0.265f, rhocsn); // L4453
    } else { // L4454
    fact = 1.f; // L4457
    if (((rhosn < 156.f) || ((newsnow > 0.f) && (rhonewsn < 156.f)))) { // L4458
    keff = __fadd_rn(0.023f, __fmul_rn(__fmul_rn(0.234f, rhosn), 1.e-3f)); // L4459
    } else { // L4460
    keff = __fadd_rn(__fsub_rn(0.138f, __fmul_rn(__fmul_rn(1.01f, rhosn), 1.e-3f)), __fmul_rn(__fmul_rn(3.233f, __fmul_rn(rhosn, rhosn)), 1.e-6f)); // L4461
    } // L4462
    if ((((newsnow <= 0.f) && (snhei > 1.f)) && (rhosn > 250.f))) { // L4463
    thdifsn = 4.431718e-7f; // L4469
    } else { // L4470
    thdifsn = __fmul_rn(__fdiv_rn(keff, rhocsn), fact); // L4471
    } // L4472
    } // L4473
    } // L4475
    } // L4476
    if ((snhei >= snth)) { // L4479
    s = __fdiv_rn(__fmul_rn(__fmul_rn(thdifsn, rhocsn), __fsub_rn(soilt, tsob)), snprim); // L4480
    snflx = s; // L4481
    s = __fmul_rn(d9, __fsub_rn(tso(1), tso(2))); // L4482
    } else if (((snhei < snth) && (snhei > 0.f))) { // L4483
    s = __fdiv_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn), __fmul_rn(__fmul_rn(fso, thdif(1)), rhcs)), __fsub_rn(soilt, tsob)), snprim); // L4484
    snflx = s; // L4486
    s = __fmul_rn(d9, __fsub_rn(tso(1), tso(2))); // L4487
    } else { // L4488
    s = __fmul_rn(d9sn, __fsub_rn(soilt, tsob)); // L4489
    snflx = s; // L4490
    s = __fmul_rn(d9, __fsub_rn(tso(1), tso(2))); // L4491
    } // L4492
    snhei = __fdiv_rn(__fmul_rn(snwe, 1.e3f), rhosn); // L4494
    if (((tso(1) > 273.15f) && (snhei > 0.f))) { // L4499
    if ((snhei > __fadd_rn(deltsn, snth))) { // L4500
    hsn = __fsub_rn(snhei, deltsn); // L4501
    if (false) { // L4502
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4503
    } // L4504
    } else { // L4505
    if (false) { // L4506
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4507
    } // L4508
    hsn = snhei; // L4509
    } // L4510
    soiltfrac = __fadd_rn(__fmul_rn(snowfrac, 273.15f), __fmul_rn(__fsub_rn(1.f, snowfrac), tso(1))); // L4512
    snohg = __fdiv_rn(__fmul_rn(__fsub_rn(tso(1), soiltfrac), __fadd_rn(__fmul_rn(cap(1), zshalf(2)), __fmul_rn(__fmul_rn(rhocsn, 0.5f), hsn))), delt); // L4514
    snohg = ruc_max<float>(0.f, snohg); // L4516
    snodif = 0.f; // L4517
    smeltg = __fmul_rn(__fdiv_rn(snohg, xlmelt), 1.e-3f); // L4518
    if ((((rhosn < 350.f) || ((newsnow > 0.f) && (rhonewsn < 450.f))) && (soilt < 283.f))) { // L4521
    smeltg = ruc_min<float>(smeltg, 5.8e-9f); // L4522
    } // L4523
    rr = __fdiv_rn(snwe, delt); // L4526
    smeltg = ruc_min<float>(smeltg, rr); // L4527
    snohgnew = __fmul_rn(__fmul_rn(smeltg, xlmelt), 1.e3f); // L4529
    snodif = ruc_max<float>(0.f, __fsub_rn(snohg, snohgnew)); // L4530
    if (false) { // L4531
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4532
    } // L4533
    snwe = ruc_max<float>(0.f, __fsub_rn(snwe, __fmul_rn(smeltg, delt))); // L4535
    snhei = __fdiv_rn(__fmul_rn(snwe, 1.e3f), rhosn); // L4536
    smelt = __fadd_rn(smelt, smeltg); // L4538
    if ((snhei > 0.f)) { tso(1) = soiltfrac; } // L4540
    if (false) { // L4541
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4542
    if ((snhei == 0.f)) { ; /* Native diagnostic I/O omitted; no physics mutation. */ } // L4543
    } // L4545
    } // L4547
    if (false) { // L4548
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4549
    } // L4550
    snweprint = snwe; // L4552
    snheiprint = __fdiv_rn(__fmul_rn(snweprint, 1.e3f), rhosn); // L4553
    if (false) { // L4555
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4556
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4557
    } // L4558
    x = __fadd_rn(__fmul_rn(__fadd_rn(r21, __fmul_rn(d9sn, r22sn)), __fsub_rn(soilt, tn)), __fmul_rn(__fmul_rn(xlvm, r210), __fsub_rn(qsg, qgold))); // L4560
    if ((globalcellid == tb.targetcell)) { // L4563
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4564
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4565
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4566
    } // L4568
    x = __fsub_rn(__fsub_rn(x, __fmul_rn(__fdiv_rn(__fmul_rn(rhonewcsn, newsnow), delt), __fsub_rn(ruc_min<float>(273.15f, tabs), soilt))), __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), __fsub_rn(ruc_max<float>(273.15f, tabs), soilt))); // L4570
    if ((globalcellid == tb.targetcell)) { // L4575
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4576
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4577
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4578
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4579
    } // L4580
    if ((snhei > 0.f)) { // L4582
    if ((ilnb > 1)) { // L4583
    tsnav = __fsub_rn(__fmul_rn(__fdiv_rn(0.5f, snhei), __fadd_rn(__fmul_rn(__fadd_rn(soilt, soilt1), deltsn), __fmul_rn(__fadd_rn(soilt1, tso(1)), __fsub_rn(snhei, deltsn)))), 273.15f); // L4584
    } else { // L4587
    tsnav = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(soilt, tso(1))), 273.15f); // L4588
    } // L4589
    } else { // L4590
    tsnav = __fsub_rn(soilt, 273.15f); // L4591
    } // L4592
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:4599-4968
RUCDEV void land_soilmoist(Tables &tb, Arena &arena, float  delt, int  nzs, int  nddzs, FArray<float> dtdzs, FArray<float> dtdzs2, float  riw, FArray<float> zsmain, FArray<float> zshalf, FArray<float> diffu, FArray<float> hydro, float  qsg, float  qvg, float  qcg, float  qcatm, float  qvatm, float  prcp, float  qkms, FArray<float> transp, float  drip, float  dew, float  smelt, FArray<float> soilice, float  vegfrac, float  snowfrac, float  soilres, int  globalid, float  dqm, float  qmin, float  ref, float  ksat, float  ras, float & infmax, FArray<float> soilmois, FArray<float> soiliqw, float & mavail, float & runoff, float & runoff2, float & infiltrp) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    FArray<float> cosmc = arena.alloc<float>(1, nzs);
    FArray<float> rhsmc = arena.alloc<float>(1, nzs);
    float dzs;
    float r1;
    float r2;
    float r3;
    float r4;
    float r5;
    float r6;
    float r7;
    float r8;
    float r9;
    float r10;
    float refkdt;
    float refdk;
    float delt1;
    float f1max;
    float f2max;
    float f1;
    float f2;
    float fd;
    float kdt;
    float val;
    float ddt;
    float px;
    float fk;
    float fkmax;
    float qq;
    float umveg;
    float infmax1;
    float trans;
    float totliq;
    float flx;
    float flxsat;
    float qtot;
    float did;
    float x1;
    float x2;
    float x4;
    float denom;
    float q2;
    float q4;
    float dice;
    float fcr;
    float acrt;
    float frzx;
    float sum;
    float cvfrz;
    int nzs1;
    int nzs2;
    int k;
    int kk;
    int k1;
    int kn;
    int ialp1;
    int jj;
    int jk;
    nzs1 = (nzs - 1); // L4701
    nzs2 = (nzs - 2); // L4702
    label_118: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4704
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L4706
    cosmc(k) = 0.f; // L4707
    rhsmc(k) = 0.f; // L4708
    loop1_continue: ;
    } loop1_exit: ; // L4709
    did = __fsub_rn(zsmain(nzs), zshalf(nzs)); // L4711
    x1 = __fsub_rn(zsmain(nzs), zsmain(nzs1)); // L4712
    denom = __fadd_rn(__fadd_rn(1.f, __fmul_rn(__fdiv_rn(__fdiv_rn(diffu(nzs1), x1), did), delt)), __fmul_rn(__fdiv_rn(hydro(nzs), __fmul_rn(2.f, did)), delt)); // L4722
    cosmc(1) = __fdiv_rn(__fmul_rn(delt, __fadd_rn(__fdiv_rn(__fdiv_rn(diffu(nzs1), did), x1), __fdiv_rn(__fdiv_rn(hydro(nzs1), 2.f), did))), denom); // L4723
    rhsmc(1) = __fdiv_rn(__fadd_rn(soilmois(nzs), __fdiv_rn(__fmul_rn(transp(nzs), delt), did)), denom); // L4725
    denom = __fadd_rn(1.f, __fmul_rn(__fdiv_rn(__fdiv_rn(diffu(nzs1), x1), did), delt)); // L4737
    cosmc(1) = __fdiv_rn(__fmul_rn(delt, __fadd_rn(__fdiv_rn(__fdiv_rn(diffu(nzs1), did), x1), __fdiv_rn(hydro(nzs1), did))), denom); // L4738
    rhsmc(1) = __fdiv_rn(__fadd_rn(__fsub_rn(soilmois(nzs), __fmul_rn(__fdiv_rn(__fmul_rn(hydro(nzs), delt), did), soilmois(nzs))), __fdiv_rn(__fmul_rn(transp(nzs), delt), did)), denom); // L4740
    cosmc(1) = 0.f; // L4742
    rhsmc(1) = soilmois(nzs); // L4743
    for (k = 1; k <= nzs2; k += 1) { loop2_cycle: ; // L4745
    kn = (nzs - k); // L4746
    k1 = ((2 * kn) - 3); // L4747
    x4 = __fmul_rn(__fmul_rn(2.f, dtdzs(k1)), diffu((kn - 1))); // L4748
    x2 = __fmul_rn(__fmul_rn(2.f, dtdzs((k1 + 1))), diffu(kn)); // L4749
    q4 = __fadd_rn(x4, __fmul_rn(hydro((kn - 1)), dtdzs2((kn - 1)))); // L4750
    q2 = __fsub_rn(x2, __fmul_rn(hydro((kn + 1)), dtdzs2((kn - 1)))); // L4751
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x2), x4), __fmul_rn(q2, cosmc(k))); // L4752
    cosmc((k + 1)) = __fdiv_rn(q4, denom); // L4753
    if (false) { // L4754
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4756
    } // L4758
    rhsmc((k + 1)) = __fdiv_rn(__fadd_rn(__fadd_rn(soilmois(kn), __fmul_rn(q2, rhsmc(k))), __fmul_rn(__fdiv_rn(transp(kn), __fsub_rn(zshalf((kn + 1)), zshalf(kn))), delt)), denom); // L4759
    if ((globalid == tb.targetcell)) { // L4763
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4764
    } // L4766
    loop2_continue: ;
    } loop2_exit: ; // L4767
    trans = transp(1); // L4770
    umveg = __fmul_rn(__fsub_rn(1.f, vegfrac), soilres); // L4771
    runoff = 0.f; // L4773
    runoff2 = 0.f; // L4774
    dzs = zsmain(2); // L4775
    r1 = cosmc(nzs1); // L4776
    r2 = rhsmc(nzs1); // L4777
    r3 = __fdiv_rn(diffu(1), dzs); // L4778
    r4 = __fadd_rn(r3, __fmul_rn(hydro(1), .5f)); // L4779
    r5 = __fsub_rn(r3, __fmul_rn(hydro(2), .5f)); // L4780
    r6 = __fmul_rn(qkms, ras); // L4781
    label_191: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4787
    totliq = __fsub_rn(__fsub_rn(__fsub_rn(prcp, __fdiv_rn(drip, delt)), __fmul_rn(__fmul_rn(umveg, dew), ras)), smelt); // L4791
    if ((globalid == tb.targetcell)) { // L4793
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4794
    } // L4796
    flx = totliq; // L4798
    infiltrp = totliq; // L4799
    cvfrz = 3.f; // L4810
    refkdt = 3.f; // L4813
    refdk = 3.4341e-6f; // L4814
    delt1 = __fdiv_rn(delt, 86400.f); // L4815
    f1max = __fmul_rn(dqm, zshalf(2)); // L4816
    f2max = __fmul_rn(dqm, __fsub_rn(zshalf(3), zshalf(2))); // L4817
    f1 = __fmul_rn(f1max, __fsub_rn(1.f, __fdiv_rn(soilmois(1), dqm))); // L4818
    dice = __fmul_rn(soilice(1), zshalf(2)); // L4819
    fd = f1; // L4820
    for (k = 2; k <= nzs1; k += 1) { loop3_cycle: ; // L4821
    dice = __fadd_rn(dice, __fmul_rn(__fsub_rn(zshalf((k + 1)), zshalf(k)), soilice(k))); // L4822
    fkmax = __fmul_rn(dqm, __fsub_rn(zshalf((k + 1)), zshalf(k))); // L4823
    fk = __fmul_rn(fkmax, __fsub_rn(1.f, __fdiv_rn(soilmois(k), dqm))); // L4824
    fd = __fadd_rn(fd, fk); // L4825
    loop3_continue: ;
    } loop3_exit: ; // L4826
    kdt = __fdiv_rn(__fmul_rn(refkdt, ksat), refdk); // L4827
    val = __fsub_rn(1.f, gfk_exp_fma(__fmul_rn((-kdt), delt1))); // L4828
    ddt = __fmul_rn(fd, val); // L4829
    px = __fmul_rn((-totliq), delt); // L4830
    if ((px < 0.0f)) { px = 0.0f; } // L4831
    if ((px > 0.0f)) { // L4832
    infmax1 = __fdiv_rn(__fmul_rn(px, __fdiv_rn(ddt, __fadd_rn(px, ddt))), delt); // L4833
    } else { // L4834
    infmax1 = 0.f; // L4835
    } // L4836
    if (false) { // L4837
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4838
    } // L4839
    frzx = __fmul_rn(__fmul_rn(0.15f, __fdiv_rn(__fadd_rn(dqm, qmin), ref)), __fdiv_rn(0.412f, 0.468f)); // L4846
    fcr = 1.f; // L4847
    if ((dice > 1.e-2f)) { // L4848
    acrt = __fdiv_rn(__fmul_rn(cvfrz, frzx), dice); // L4849
    sum = 1.f; // L4850
    ialp1 = __fsub_rn(cvfrz, 1); // L4851
    for (jk = 1; jk <= ialp1; jk += 1) { loop4_cycle: ; // L4852
    k = 1; // L4853
    for (jj = (jk + 1); jj <= ialp1; jj += 1) { loop5_cycle: ; // L4854
    k = (k * jj); // L4855
    loop5_continue: ;
    } loop5_exit: ; // L4856
    sum = __fadd_rn(sum, __fdiv_rn(gfk_pow(acrt, __fsub_rn(cvfrz, jk)), float(k))); // L4857
    loop4_continue: ;
    } loop4_exit: ; // L4858
    fcr = __fsub_rn(1.f, __fmul_rn(gfk_exp_fma((-acrt)), sum)); // L4859
    } // L4860
    if (false) { // L4861
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4862
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4863
    } // L4864
    infmax1 = __fmul_rn(infmax1, fcr); // L4865
    infmax = ruc_max<float>(infmax1, __fmul_rn(hydro(1), soilmois(1))); // L4868
    infmax = ruc_min<float>(infmax, (-totliq)); // L4869
    if ((globalid == tb.targetcell)) { // L4871
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4872
    } // L4874
    if (((-totliq) > infmax)) { // L4876
    runoff = __fsub_rn((-totliq), infmax); // L4877
    flx = (-infmax); // L4878
    if (false) { // L4879
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4881
    } // L4882
    } // L4883
    infiltrp = flx; // L4885
    r7 = __fdiv_rn(__fmul_rn(.5f, dzs), delt); // L4887
    r4 = __fadd_rn(r4, r7); // L4888
    flx = __fsub_rn(flx, __fmul_rn(soilmois(1), r7)); // L4889
    r8 = __fmul_rn(__fmul_rn(umveg, r6), __fsub_rn(1.f, snowfrac)); // L4893
    qtot = __fadd_rn(qvatm, qcatm); // L4894
    r9 = trans; // L4895
    r10 = __fsub_rn(qtot, qsg); // L4896
    if ((r10 <= 0.f)) { // L4899
    qq = __fdiv_rn(__fadd_rn(__fsub_rn(__fmul_rn(r5, r2), flx), r9), __fsub_rn(__fsub_rn(r4, __fmul_rn(r5, r1)), __fdiv_rn(__fmul_rn(r10, r8), __fsub_rn(ref, qmin)))); // L4900
    flxsat = __fadd_rn(__fadd_rn(__fmul_rn((-dqm), __fsub_rn(__fsub_rn(r4, __fmul_rn(r5, r1)), __fdiv_rn(__fmul_rn(r10, r8), __fsub_rn(ref, qmin)))), __fmul_rn(r5, r2)), r9); // L4901
    } else { // L4903
    qq = __fdiv_rn(__fadd_rn(__fadd_rn(__fsub_rn(__fmul_rn(r2, r5), flx), __fmul_rn(r8, __fsub_rn(__fsub_rn(qtot, qcg), qvg))), r9), __fsub_rn(r4, __fmul_rn(r1, r5))); // L4905
    flxsat = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn((-dqm), __fsub_rn(r4, __fmul_rn(r1, r5))), __fmul_rn(r2, r5)), __fmul_rn(r8, __fsub_rn(__fsub_rn(qtot, qvg), qcg))), r9); // L4906
    } // L4907
    if ((qq < 0.f)) { // L4909
    if ((globalid == tb.targetcell)) { ; /* Native diagnostic I/O omitted; no physics mutation. */ } // L4910
    soilmois(1) = 1.e-8f; // L4911
    } else if ((qq > dqm)) { // L4913
    soilmois(1) = dqm; // L4915
    if ((globalid == tb.targetcell)) { // L4917
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4918
    } // L4919
    runoff = __fadd_rn(runoff, __fsub_rn(flxsat, flx)); // L4921
    } else { // L4922
    soilmois(1) = ruc_min<float>(dqm, ruc_max<float>(1.e-8f, qq)); // L4923
    } // L4924
    if ((globalid == tb.targetcell)) { // L4927
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4928
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4929
    } // L4930
    for (k = 2; k <= nzs; k += 1) { loop6_cycle: ; // L4933
    kk = ((nzs - k) + 1); // L4934
    qq = __fadd_rn(__fmul_rn(cosmc(kk), soilmois((k - 1))), rhsmc(kk)); // L4935
    if ((qq < 0.f)) { // L4938
    soilmois(k) = 1.e-8f; // L4940
    } else if ((qq > dqm)) { // L4942
    soilmois(k) = dqm; // L4944
    if ((k == nzs)) { // L4945
    if ((globalid == tb.targetcell)) { // L4947
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4948
    } // L4949
    runoff2 = __fadd_rn(runoff2, __fdiv_rn(__fmul_rn(__fsub_rn(qq, dqm), __fsub_rn(zsmain(k), zshalf(k))), delt)); // L4950
    } else { // L4951
    runoff2 = __fadd_rn(runoff2, __fdiv_rn(__fmul_rn(__fsub_rn(qq, dqm), __fsub_rn(zshalf((k + 1)), zshalf(k))), delt)); // L4952
    } // L4953
    } else { // L4954
    soilmois(k) = ruc_min<float>(dqm, ruc_max<float>(1.e-8f, qq)); // L4955
    } // L4956
    loop6_continue: ;
    } loop6_exit: ; // L4957
    if ((globalid == tb.targetcell)) { // L4959
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L4960
    } // L4961
    mavail = ruc_max<float>(.00001f, ruc_min<float>(1.f, __fadd_rn(__fmul_rn(__fdiv_rn(soilmois(1), __fsub_rn(ref, qmin)), __fsub_rn(1.f, snowfrac)), __fmul_rn(1.f, snowfrac)))); // L4963
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:4972-5190
RUCDEV void land_soilprop(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, int  nzs, FArray<float> fwsat, FArray<float> lwsat, FArray<float> tav, FArray<float> keepfr, FArray<float> soilmois, FArray<float> soiliqw, FArray<float> soilice, FArray<float> soilmoism, FArray<float> soiliqwm, FArray<float> soilicem, int  globalid, float  qwrtz, float  rhocs, float  dqm, float  qmin, float  psis, float  bclh, float  ksat, float  riw, float  xlmelt, float  cp, float  g0_p, float  cvw, float & ci, float  kqwrtz, float  kice, float  kwt, FArray<float> thdif, FArray<float> diffu, FArray<float> hydro, FArray<float> cap) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    FArray<float> hk = arena.alloc<float>(1, nzs);
    FArray<float> detal = arena.alloc<float>(1, nzs);
    FArray<float> kasat = arena.alloc<float>(1, nzs);
    FArray<float> kjpl = arena.alloc<float>(1, nzs);
    float x;
    float x1;
    float x2;
    float x4;
    float ws;
    float wd;
    float fact;
    float fach;
    float facd;
    float psif;
    float tln;
    float tavln;
    float tn;
    float pf;
    float a;
    float am;
    float ame;
    float h;
    int nzs1;
    int k;
    float kzero;
    float gamd;
    float kdry;
    float kas;
    float x5;
    float sr;
    float ke;
    nzs1 = (nzs - 1); // L5058
    kzero = 2.f; // L5061
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L5064
    detal(k) = 0.f; // L5065
    kasat(k) = 0.f; // L5066
    kjpl(k) = 0.f; // L5067
    hk(k) = 0.f; // L5068
    loop1_continue: ;
    } loop1_exit: ; // L5069
    ws = __fadd_rn(dqm, qmin); // L5071
    x1 = __fdiv_rn(xlmelt, __fmul_rn(g0_p, psis)); // L5072
    x2 = __fmul_rn(__fdiv_rn(x1, bclh), ws); // L5073
    x4 = __fdiv_rn(__fadd_rn(bclh, 1.f), bclh); // L5074
    gamd = __fmul_rn(__fsub_rn(1.f, ws), 2700.f); // L5076
    kdry = __fdiv_rn(__fadd_rn(__fmul_rn(0.135f, gamd), 64.7f), __fsub_rn(2700.f, __fmul_rn(0.947f, gamd))); // L5077
    if ((qwrtz > 0.2f)) { // L5079
    kas = __fmul_rn(gfk_pow(kqwrtz, qwrtz), gfk_pow(kzero, __fsub_rn(1.f, qwrtz))); // L5080
    } else { // L5081
    kas = __fmul_rn(gfk_pow(kqwrtz, qwrtz), gfk_pow(3.f, __fsub_rn(1.f, qwrtz))); // L5082
    } // L5083
    for (k = 1; k <= nzs1; k += 1) { loop2_cycle: ; // L5085
    tn = __fsub_rn(tav(k), 273.15f); // L5086
    wd = __fsub_rn(ws, __fmul_rn(riw, soilicem(k))); // L5087
    psif = __fmul_rn(__fmul_rn(__fmul_rn(psis, 100.f), gfk_pow(__fdiv_rn(wd, __fadd_rn(soiliqwm(k), qmin)), bclh)), gfk_pow(__fdiv_rn(ws, wd), 3.f)); // L5088
    pf = ruc_log10f_rn(fabsf(psif)); // L5091
    fact = __fadd_rn(1.f, __fmul_rn(riw, soilicem(k))); // L5092
    if ((pf <= 5.2f)) { // L5094
    hk(k) = __fmul_rn(__fmul_rn(420.f, gfk_exp_fma((-__fadd_rn(pf, 2.7f)))), fact); // L5095
    } else { // L5096
    hk(k) = __fmul_rn(.1744f, fact); // L5097
    } // L5098
    if (((soilicem(k) != 0) && (tn < 0.f))) { // L5100
    detal(k) = __fmul_rn(__fdiv_rn(__fmul_rn(273.15f, x2), __fmul_rn(tav(k), tav(k))), gfk_pow(__fdiv_rn(tav(k), __fmul_rn(x1, tn)), x4)); // L5104
    if ((keepfr(k) == 1.f)) { // L5107
    detal(k) = 0.f; // L5108
    } // L5109
    } // L5111
    kasat(k) = __fmul_rn(__fmul_rn(gfk_pow(kas, __fsub_rn(1.f, ws)), gfk_pow(kice, fwsat(k))), gfk_pow(kwt, lwsat(k))); // L5114
    x5 = __fdiv_rn(__fadd_rn(soilmoism(k), qmin), ws); // L5117
    if ((soilicem(k) == 0.f)) { // L5118
    sr = ruc_max<float>(0.101f, x5); // L5119
    ke = __fadd_rn(ruc_log10f_rn(sr), 1.f); // L5120
    } else { // L5124
    ke = x5; // L5125
    } // L5126
    kjpl(k) = __fadd_rn(__fmul_rn(ke, __fsub_rn(kasat(k), kdry)), kdry); // L5128
    cap(k) = __fsub_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fsub_rn(1.f, ws), rhocs), __fmul_rn(__fadd_rn(soiliqwm(k), qmin), cvw)), __fmul_rn(soilicem(k), ci)), __fmul_rn(__fmul_rn(__fsub_rn(dqm, soilmoism(k)), cp), 1.2f)), __fmul_rn(__fmul_rn(detal(k), 1.e3f), xlmelt)); // L5131
    a = __fmul_rn(riw, soilicem(k)); // L5137
    if ((__fsub_rn(ws, a) < 0.12f)) { // L5139
    diffu(k) = 0.f; // L5140
    } else { // L5141
    h = ruc_max<float>(0.f, __fdiv_rn(__fsub_rn(__fadd_rn(soilmoism(k), qmin), a), ruc_max<float>(1.e-8f, __fsub_rn(ws, a)))); // L5142
    facd = 1.f; // L5143
    if ((a != 0.f)) { facd = __fsub_rn(1.f, __fdiv_rn(a, ruc_max<float>(1.e-8f, soilmoism(k)))); } // L5144
    ame = ruc_max<float>(1.e-8f, __fsub_rn(ws, a)); // L5145
    diffu(k) = __fmul_rn(__fmul_rn(__fmul_rn(__fdiv_rn(__fmul_rn(__fmul_rn((-bclh), ksat), psis), ame), gfk_pow(__fdiv_rn(ws, ame), 3.f)), gfk_pow(h, __fadd_rn(bclh, 2.f))), facd); // L5147
    } // L5151
    thdif(k) = __fdiv_rn(kjpl(k), cap(k)); // L5156
    loop2_continue: ;
    } loop2_exit: ; // L5158
    if (false) { // L5160
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5161
    } // L5162
    for (k = 1; k <= nzs; k += 1) { loop3_cycle: ; // L5169
    if ((__fsub_rn(ws, __fmul_rn(riw, soilice(k))) < 0.12f)) { // L5171
    hydro(k) = 0.f; // L5172
    } else { // L5173
    fach = 1.f; // L5174
    if ((soilice(k) != 0.f)) { fach = __fsub_rn(1.f, __fdiv_rn(__fmul_rn(riw, soilice(k)), ruc_max<float>(1.e-8f, soilmois(k)))); } // L5175
    am = ruc_max<float>(1.e-8f, __fsub_rn(ws, __fmul_rn(riw, soilice(k)))); // L5177
    hydro(k) = ruc_min<float>(ksat, __fmul_rn(__fmul_rn(__fdiv_rn(ksat, am), gfk_pow(__fdiv_rn(soiliqw(k), am), __fadd_rn(__fmul_rn(2.f, bclh), 2.f))), fach)); // L5179
    if ((hydro(k) < 1.e-10f)) { hydro(k) = 0.f; } // L5184
    } // L5185
    loop3_continue: ;
    } loop3_exit: ; // L5187
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:5194-5371
RUCDEV void land_transf(Tables &tb, Arena &arena, int  i, int  j, int  nzs, int  nroot, FArray<float> soiliqw, float  tabs, float  lai, float  gswin, float  dqm, float  qmin, float  ref, float  wilt, FArray<float> zshalf, float  pc, int  iland, FArray<float> tranf, float & transum) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float totliq;
    float did;
    int k;
    float gx;
    float sm1;
    float sm2;
    float sm3;
    float sm4;
    float ap0;
    float ap1;
    float ap2;
    float ap3;
    float ap4;
    float ftem;
    float pctot;
    float fsol;
    float f1;
    float cmin;
    float cmax;
    float totcnd;
    FArray<float> part = arena.alloc<float>(1, nzs);
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L5245
    part(k) = 0.f; // L5246
    tranf(k) = 0.f; // L5247
    loop1_continue: ;
    } loop1_exit: ; // L5248
    transum = 0.f; // L5250
    totliq = __fadd_rn(soiliqw(1), qmin); // L5251
    sm1 = totliq; // L5252
    sm2 = __fmul_rn(sm1, sm1); // L5253
    sm3 = __fmul_rn(sm2, sm1); // L5254
    sm4 = __fmul_rn(sm3, sm1); // L5255
    ap0 = 0.299f; // L5256
    ap1 = (-8.152f); // L5257
    ap2 = 61.653f; // L5258
    ap3 = (-115.876f); // L5259
    ap4 = 59.656f; // L5260
    gx = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(ap0, __fmul_rn(ap1, sm1)), __fmul_rn(ap2, sm2)), __fmul_rn(ap3, sm3)), __fmul_rn(ap4, sm4)); // L5261
    if ((totliq >= ref)) { gx = 1.f; } // L5262
    if ((totliq <= 0.f)) { gx = 0.f; } // L5263
    if ((gx > 1.f)) { gx = 1.f; } // L5264
    if ((gx < 0.f)) { gx = 0.f; } // L5265
    did = zshalf(2); // L5266
    part(1) = __fmul_rn(did, gx); // L5267
    if ((totliq > ref)) { // L5268
    tranf(1) = did; // L5269
    } else if ((totliq <= wilt)) { // L5270
    tranf(1) = 0.f; // L5271
    } else { // L5272
    tranf(1) = __fmul_rn(__fdiv_rn(__fsub_rn(totliq, wilt), __fsub_rn(ref, wilt)), did); // L5273
    } // L5274
    for (k = 2; k <= nroot; k += 1) { loop2_cycle: ; // L5278
    totliq = __fadd_rn(soiliqw(k), qmin); // L5279
    sm1 = totliq; // L5280
    sm2 = __fmul_rn(sm1, sm1); // L5281
    sm3 = __fmul_rn(sm2, sm1); // L5282
    sm4 = __fmul_rn(sm3, sm1); // L5283
    gx = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(ap0, __fmul_rn(ap1, sm1)), __fmul_rn(ap2, sm2)), __fmul_rn(ap3, sm3)), __fmul_rn(ap4, sm4)); // L5284
    if ((totliq >= ref)) { gx = 1.f; } // L5285
    if ((totliq <= 0.f)) { gx = 0.f; } // L5286
    if ((gx > 1.f)) { gx = 1.f; } // L5287
    if ((gx < 0.f)) { gx = 0.f; } // L5288
    did = __fsub_rn(zshalf((k + 1)), zshalf(k)); // L5289
    part(k) = __fmul_rn(did, gx); // L5290
    if ((totliq >= ref)) { // L5291
    tranf(k) = did; // L5292
    } else if ((totliq <= wilt)) { // L5293
    tranf(k) = 0.f; // L5294
    } else { // L5295
    tranf(k) = __fmul_rn(__fdiv_rn(__fsub_rn(totliq, wilt), __fsub_rn(ref, wilt)), did); // L5296
    } // L5298
    loop2_continue: ;
    } loop2_exit: ; // L5301
    if ((lai > 4.f)) { // L5304
    pctot = 0.8f; // L5305
    } else { // L5306
    pctot = pc; // L5307
    } // L5311
    if (false) { // L5312
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5313
    } // L5314
    if ((tabs <= 302.15f)) { // L5318
    ftem = __fdiv_rn(1.0f, __fadd_rn(1.0f, gfk_exp_fma(__fmul_rn((-0.41f), __fsub_rn(tabs, 282.05f))))); // L5319
    } else { // L5320
    ftem = __fdiv_rn(1.0f, __fadd_rn(1.0f, gfk_exp_fma(__fmul_rn(0.5f, __fsub_rn(tabs, 314.0f))))); // L5321
    } // L5322
    if (false) { // L5323
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5324
    } // L5325
    cmin = __fdiv_rn(1.f, tb.rsmax_data); // L5327
    cmax = __fdiv_rn(1.f, tb.rstbl(iland)); // L5328
    if ((lai > 1.f)) { // L5329
    cmax = __fdiv_rn(lai, tb.rstbl(iland)); // L5330
    } // L5331
    f1 = 0.f; // L5333
    if ((gswin < tb.rgltbl(iland))) { // L5343
    fsol = __fdiv_rn(1.f, __fadd_rn(1.f, gfk_exp_fma(__fmul_rn((-0.034f), __fsub_rn(gswin, 3.5f))))); // L5344
    } else { // L5345
    fsol = 1.f; // L5346
    } // L5347
    if (false) { // L5348
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5349
    } // L5350
    totcnd = __fdiv_rn(__fadd_rn(cmin, __fmul_rn(__fmul_rn(__fsub_rn(cmax, cmin), ftem), fsol)), cmax); // L5352
    if (false) { // L5354
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5355
    } // L5357
    transum = 0.f; // L5360
    for (k = 1; k <= nroot; k += 1) { loop3_cycle: ; // L5361
    tranf(k) = __fmul_rn(__fmul_rn(tranf(k), totcnd), pctot); // L5363
    transum = __fadd_rn(transum, tranf(k)); // L5364
    loop3_continue: ;
    } loop3_exit: ; // L5365
    if (false) { // L5366
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5367
    } // L5368
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:5375-5413
RUCDEV void land_vilka(Tables &tb, Arena &arena, float  tn, float  d1, float  d2, float  pp, float & qs, float & ts, FArray<float> tt, int  nstep, int  ii, int  j, int  iland, int  isoil) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float f1;
    float t1;
    float t2;
    float rn;
    int i;
    int i1;
    i = __fadd_rn(__fdiv_rn(__fsub_rn(tn, 1.7315e2f), .05f), 1); // L5389
    t1 = __fadd_rn(173.1f, __fmul_rn(float(i), .05f)); // L5390
    f1 = __fsub_rn(__fadd_rn(t1, __fmul_rn(d1, tt(i))), d2); // L5391
    i1 = __fsub_rn(i, __fdiv_rn(f1, __fadd_rn(.05f, __fmul_rn(d1, __fsub_rn(tt((i + 1)), tt(i)))))); // L5392
    i = i1; // L5393
    if (((i > 5000) || (i < 1))) { goto label_1; } // L5394
    label_10: ;
    i1 = i; // L5395
    t1 = __fadd_rn(173.1f, __fmul_rn(float(i), .05f)); // L5396
    f1 = __fsub_rn(__fadd_rn(t1, __fmul_rn(d1, tt(i))), d2); // L5397
    rn = __fdiv_rn(f1, __fadd_rn(.05f, __fmul_rn(d1, __fsub_rn(tt((i + 1)), tt(i))))); // L5398
    i = (i - int(rn)); // L5399
    if (((i > 5000) || (i < 1))) { goto label_1; } // L5400
    if ((i1 != i)) { goto label_10; } // L5401
    ts = __fsub_rn(t1, __fmul_rn(.05f, rn)); // L5402
    qs = __fdiv_rn(__fadd_rn(tt(i), __fmul_rn(__fsub_rn(tt(i), tt((i + 1))), rn)), pp); // L5403
    goto label_20; // L5404
    label_1: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5407
    atomicOr(arena.error, 1); return; // L5410
    label_20: ;
    ; // L5411
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:5416-5872
RUCDEV void land_soilvegin(Tables &tb, Arena &arena, int  mosaic_lu, int  mosaic_soil, FArray<float> soilfrac, int  nscat, float  shdmin, float  shdmax, int  nlcat, int & ivgtyp, int & isltyp, int  iswater, int & iforest, FArray<float> lufrac, float  vegfrac, float & emiss, float & pc, float & znt, float & lai, int  rdlai2d, float & qwrtz, float & rhocs, float & bclh, float & dqm, float & ksat, float & psis, float & qmin, float & ref, float & wilt, int  i, int  j) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int nsoilclas = 19;
    int nvegclas = (24 + 3);
    int ilsnow = 99;
    FArray<float> lqma = arena.alloc<float>(1, nsoilclas);
    FArray<float> lrhc = arena.alloc<float>(1, nsoilclas);
    FArray<float> lpsi = arena.alloc<float>(1, nsoilclas);
    FArray<float> lqmi = arena.alloc<float>(1, nsoilclas);
    FArray<float> lbcl = arena.alloc<float>(1, nsoilclas);
    FArray<float> lkas = arena.alloc<float>(1, nsoilclas);
    FArray<float> lwil = arena.alloc<float>(1, nsoilclas);
    FArray<float> lref = arena.alloc<float>(1, nsoilclas);
    FArray<float> datqtz = arena.alloc<float>(1, nsoilclas);
    FArray<float> lalb = arena.alloc<float>(1, nvegclas);
    FArray<float> lmoi = arena.alloc<float>(1, nvegclas);
    FArray<float> lemi = arena.alloc<float>(1, nvegclas);
    FArray<float> lrou = arena.alloc<float>(1, nvegclas);
    FArray<float> lthi = arena.alloc<float>(1, nvegclas);
    FArray<float> lsig = arena.alloc<float>(1, nvegclas);
    FArray<float> lpc = arena.alloc<float>(1, nvegclas);
    int kstart;
    int kfin;
    int lstart;
    int lfin;
    int k;
    float area;
    float factor;
    float znt1;
    float lb;
    FArray<float> znttoday = arena.alloc<float>(1, nlcat);
    FArray<float> laitoday = arena.alloc<float>(1, nlcat);
    FArray<float> deltalai = arena.alloc<float>(1, nlcat);
    iforest = tb.ifortbl(ivgtyp); // L5695
    if (false) { // L5697
    if (((i == 375) && (j == 254))) { // L5698
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5699
    } // L5701
    } // L5702
    ruc_assign(deltalai(RRange{0,0}), 0.f); // L5704
    if ((__fsub_rn(shdmax, shdmin) < 1)) { // L5710
    factor = 1.f; // L5711
    } else { // L5712
    factor = __fsub_rn(1.f, ruc_max<float>(0.f, ruc_min<float>(1.f, __fdiv_rn(__fsub_rn(vegfrac, shdmin), ruc_max<float>(1.f, __fsub_rn(shdmax, shdmin)))))); // L5713
    } // L5714
    for (k = 1; k <= nlcat; k += 1) { loop1_cycle: ; // L5717
    if ((tb.ifortbl(k) == 1)) { deltalai(k) = ruc_min<float>(0.2f, __fmul_rn(0.8f, tb.laitbl(k))); } // L5718
    if (((tb.ifortbl(k) == 2) || (tb.ifortbl(k) == 7))) { deltalai(k) = ruc_min<float>(0.5f, __fmul_rn(0.8f, tb.laitbl(k))); } // L5719
    if ((tb.ifortbl(k) == 3)) { deltalai(k) = ruc_min<float>(0.45f, __fmul_rn(0.8f, tb.laitbl(k))); } // L5720
    if ((tb.ifortbl(k) == 4)) { deltalai(k) = ruc_min<float>(0.75f, __fmul_rn(0.8f, tb.laitbl(k))); } // L5721
    if ((tb.ifortbl(k) == 5)) { deltalai(k) = ruc_min<float>(0.86f, __fmul_rn(0.8f, tb.laitbl(k))); } // L5722
    if ((k != iswater)) { // L5724
    laitoday(k) = __fsub_rn(tb.laitbl(k), __fmul_rn(deltalai(k), factor)); // L5726
    if ((tb.ifortbl(k) == 7)) { // L5728
    znttoday(k) = __fsub_rn(tb.z0tbl(k), __fmul_rn(0.125f, factor)); // L5730
    } else { // L5731
    znttoday(k) = tb.z0tbl(k); // L5732
    } // L5733
    } else { // L5734
    laitoday(k) = tb.laitbl(k); // L5735
    znttoday(k) = znt; // L5736
    } // L5737
    loop1_continue: ;
    } loop1_exit: ; // L5738
    if (false) { // L5740
    if (((i == 358) && (j == 260))) { // L5741
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5742
    } // L5744
    } // L5745
    emiss = 0.f; // L5747
    znt = 0.f; // L5748
    znt1 = 0.f; // L5749
    pc = 0.f; // L5750
    if ((!rdlai2d)) { lai = 0.f; } // L5751
    area = 0.f; // L5752
    lb = 5.f; // L5756
    if ((mosaic_lu == 1)) { // L5757
    for (k = 1; k <= nlcat; k += 1) { loop2_cycle: ; // L5758
    area = __fadd_rn(area, lufrac(k)); // L5759
    emiss = __fadd_rn(emiss, __fmul_rn(tb.lemitbl(k), lufrac(k))); // L5760
    znt = __fadd_rn(znt, __fdiv_rn(lufrac(k), __fmul_rn(gfk_log(__fdiv_rn(lb, znttoday(k))), gfk_log(__fdiv_rn(lb, znttoday(k)))))); // L5761
    znt1 = __fadd_rn(znt1, __fmul_rn(lufrac(k), znttoday(k))); // L5763
    if ((!rdlai2d)) { lai = __fadd_rn(lai, __fmul_rn(laitoday(k), lufrac(k))); } // L5764
    pc = __fadd_rn(pc, __fmul_rn(tb.pctbl(k), lufrac(k))); // L5765
    loop2_continue: ;
    } loop2_exit: ; // L5766
    if ((area > 1.f)) { area = 1.f; } // L5768
    if ((area <= 0.f)) { // L5769
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5770
    atomicOr(arena.error, 2); return; // L5771
    } // L5772
    if (false) { // L5774
    if (((i == 358) && (j == 260))) { // L5775
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5776
    } // L5777
    } // L5778
    emiss = __fdiv_rn(emiss, area); // L5780
    znt1 = __fdiv_rn(znt1, area); // L5781
    znt = __fdiv_rn(lb, gfk_exp_fma(__fsqrt_rn(__fdiv_rn(1.f, znt)))); // L5782
    if ((!rdlai2d)) { lai = __fdiv_rn(lai, area); } // L5783
    pc = __fdiv_rn(pc, area); // L5784
    if (false) { // L5786
    if (((i == 358) && (j == 260))) { // L5787
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L5788
    } // L5789
    } // L5790
    } else { // L5793
    emiss = tb.lemitbl(ivgtyp); // L5794
    znt = znttoday(ivgtyp); // L5795
    pc = tb.pctbl(ivgtyp); // L5796
    if ((!rdlai2d)) { lai = laitoday(ivgtyp); } // L5797
    } // L5798
    rhocs = 0.f; // L5801
    bclh = 0.f; // L5802
    dqm = 0.f; // L5803
    ksat = 0.f; // L5804
    psis = 0.f; // L5805
    qmin = 0.f; // L5806
    ref = 0.f; // L5807
    wilt = 0.f; // L5808
    qwrtz = 0.f; // L5809
    area = 0.f; // L5810
    if ((mosaic_soil == 1)) { // L5812
    for (k = 1; k <= nscat; k += 1) { loop3_cycle: ; // L5813
    if ((k != 14)) { // L5814
    area = __fadd_rn(area, soilfrac(k)); // L5816
    rhocs = __fadd_rn(rhocs, __fmul_rn(__fmul_rn(tb.hc(k), 1.e6f), soilfrac(k))); // L5817
    bclh = __fadd_rn(bclh, __fmul_rn(tb.bb(k), soilfrac(k))); // L5818
    dqm = __fadd_rn(dqm, __fmul_rn(__fsub_rn(tb.maxsmc(k), tb.drysmc(k)), soilfrac(k))); // L5819
    ksat = __fadd_rn(ksat, __fmul_rn(tb.satdk(k), soilfrac(k))); // L5821
    psis = __fsub_rn(psis, __fmul_rn(tb.satpsi(k), soilfrac(k))); // L5822
    qmin = __fadd_rn(qmin, __fmul_rn(tb.drysmc(k), soilfrac(k))); // L5823
    ref = __fadd_rn(ref, __fmul_rn(tb.refsmc(k), soilfrac(k))); // L5824
    wilt = __fadd_rn(wilt, __fmul_rn(tb.wltsmc(k), soilfrac(k))); // L5825
    qwrtz = __fadd_rn(qwrtz, __fmul_rn(tb.qtz(k), soilfrac(k))); // L5826
    } // L5827
    loop3_continue: ;
    } loop3_exit: ; // L5828
    if ((area > 1.f)) { area = 1.f; } // L5829
    if ((area <= 0.f)) { // L5830
    rhocs = __fmul_rn(tb.hc(isltyp), 1.e6f); // L5833
    bclh = tb.bb(isltyp); // L5834
    dqm = __fsub_rn(tb.maxsmc(isltyp), tb.drysmc(isltyp)); // L5835
    ksat = tb.satdk(isltyp); // L5837
    psis = (-tb.satpsi(isltyp)); // L5838
    qmin = tb.drysmc(isltyp); // L5839
    ref = tb.refsmc(isltyp); // L5840
    wilt = tb.wltsmc(isltyp); // L5841
    qwrtz = tb.qtz(isltyp); // L5842
    } else { // L5843
    rhocs = __fdiv_rn(rhocs, area); // L5844
    bclh = __fdiv_rn(bclh, area); // L5845
    dqm = __fdiv_rn(dqm, area); // L5846
    ksat = __fdiv_rn(ksat, area); // L5847
    psis = __fdiv_rn(psis, area); // L5848
    qmin = __fdiv_rn(qmin, area); // L5849
    ref = __fdiv_rn(ref, area); // L5850
    wilt = __fdiv_rn(wilt, area); // L5851
    qwrtz = __fdiv_rn(qwrtz, area); // L5852
    } // L5853
    } else { // L5856
    if ((isltyp != 14)) { // L5857
    rhocs = __fmul_rn(tb.hc(isltyp), 1.e6f); // L5858
    bclh = tb.bb(isltyp); // L5859
    dqm = __fsub_rn(tb.maxsmc(isltyp), tb.drysmc(isltyp)); // L5860
    ksat = tb.satdk(isltyp); // L5862
    psis = (-tb.satpsi(isltyp)); // L5863
    qmin = tb.drysmc(isltyp); // L5864
    ref = tb.refsmc(isltyp); // L5865
    wilt = tb.wltsmc(isltyp); // L5866
    qwrtz = tb.qtz(isltyp); // L5867
    } // L5868
    } // L5869
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_ice.F90:29-748
RUCDEV void ice_ruc_ice(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> pattern_spp_lsm, FArray<float> field_sf, float  dt, int  ktau, int  nsl, int  restart, int  cycling, FArray<float> graupelncv, FArray<float> snowncv, FArray<float> rainncv, FArray<float> zs, FArray<float> rainbl, FArray<float> frzfrac, int  frpcpn, FArray<float> rhosnf, FArray<float> precipfr, FArray<float> z3d, FArray<float> p8w, FArray<float> t3d, FArray<float> qv3d, FArray<float> qc3d, FArray<float> rho3d, FArray<float> glw, FArray<float> gsw, FArray<float> emiss, FArray<float> chklowq, FArray<float> chs, FArray<float> flqc, FArray<float> flhc, FArray<float> alb, FArray<float> znt, FArray<float> z0, FArray<float> snoalb, FArray<float> albbck, int  mminlu, FArray<float> tbot, FArray<int> ivgtyp, FArray<int> isltyp, FArray<float> xland, int  myjpbl, int  iswater, int  isice, FArray<float> xice, float  xice_threshold, float  cp, float  rovcp, float  g0, float  lv, float  stbolt, FArray<float> tso, FArray<float> soilt, FArray<float> soilt1, FArray<float> tsnav, FArray<float> snow, FArray<float> snowh, FArray<float> snowc, FArray<float> qsfc, FArray<float> qsg, FArray<float> qvg, FArray<float> qcg, FArray<float> hfx, FArray<float> qfx, FArray<float> lh, FArray<float> dew, FArray<float> sfcrunoff, FArray<float> acrunoff, FArray<float> sfcevp, FArray<float> grdflx, FArray<float> snowfallac, FArray<float> acsnow, FArray<float> snom, FArray<int> globalcells, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte, bool has_pattern_spp_lsm, bool has_field_sf, bool has_graupelncv, bool has_snowncv, bool has_rainncv) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int nvegclas = (24 + 3);
    FArray<float> field_sf_loc = arena.alloc<float>(ims, ime, 1, nsl, jms, jme);
    FArray<float> runoff1 = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> runoff2 = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> emissl = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> budget = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> zntl = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> smelt = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> snoh = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> snflx = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> edir = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> ec = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> ett = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> sublim = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> sflx = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> smf = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> evapl = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> prcpl = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> seaice = arena.alloc<float>(its, ite, jts, jte);
    FArray<float> infiltr = arena.alloc<float>(its, ite, jts, jte);
    float rhocs;
    float rhonewsn;
    float rhosn;
    float rhosnfall;
    float bclh;
    float dqm;
    float ksat;
    float psis;
    float qmin;
    float qwrtz;
    float ref;
    float wilt;
    float snowfrac;
    float snhei;
    float snwe;
    float cn;
    float sat;
    float cw;
    float c1sn;
    float c2sn;
    FArray<float> zsmain = arena.alloc<float>(1, nsl);
    FArray<float> zshalf = arena.alloc<float>(1, nsl);
    FArray<float> dtdzs2 = arena.alloc<float>(1, nsl);
    FArray<float> soilice = arena.alloc<float>(1, nsl);
    FArray<float> soiliqw = arena.alloc<float>(1, nsl);
    FArray<float> dtdzs = arena.alloc<float>(1, (2 * (nsl - 2)));
    FArray<float> tbq{ruc_tbq_table, 1, 1, 1, 1, 5001, 5001, 5001, 1, 1}; // the shared table (ruc_tbq_init)
    FArray<float> tso1d = arena.alloc<float>(1, nsl);
    float rsm;
    float snweprint;
    float snheiprint;
    float prcpms;
    float newsnms;
    float prcpncliq;
    float prcpncfr;
    float prcpculiq;
    float prcpcufr;
    float patm;
    float patmb;
    float tabs;
    float qvatm;
    float qcatm;
    float q2sat;
    float conflx;
    float rho;
    float qkms;
    float tkms;
    float snowrat;
    float grauprat;
    float graupamt;
    float icerat;
    float curat;
    float infiltrp;
    float cq;
    float r61;
    float r273;
    float arp;
    float brp;
    float x;
    float evs;
    float eis;
    int iland;
    int isoil;
    int iforest;
    int i;
    int j;
    int k;
    int nzs;
    int nzs1;
    int nddzs;
    int k1;
    int l;
    int k2;
    int kp;
    int km;
    int globalcellid;
    int message;
    int print_flag = false;
    FArray<float> rstoch = arena.alloc<float>(ims, ime, 1, nsl, jms, jme);
    FArray<float> rstoch_temp = arena.alloc<float>(1, nsl);
    FArray<float> field_sf_temp = arena.alloc<float>(1, nsl);
    FArray<float> emisso = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> vegfrao = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> albo = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> snoalbo = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> emisslo = arena.alloc<float>(its, ite, jts, jte);
    nzs = nsl; // L333
    nddzs = (2 * (nzs - 2)); // L334
    ruc_assign(rstoch, 0.0f); // L336
    ruc_assign(field_sf_loc, 0.0f); // L337
    if ((spp_lsm == 1)) { // L340
    for (j = jts; j <= jte; j += 1) { loop1_cycle: ; // L341
    for (i = its; i <= ite; i += 1) { loop2_cycle: ; // L342
    for (k = 1; k <= nsl; k += 1) { loop3_cycle: ; // L343
    rstoch(i, k, j) = pattern_spp_lsm(i, k, j); // L344
    field_sf_loc(i, k, j) = field_sf(i, k, j); // L345
    loop3_continue: ;
    } loop3_exit: ; // L346
    loop2_continue: ;
    } loop2_exit: ; // L347
    loop1_continue: ;
    } loop1_exit: ; // L348
    } // L349
    for (j = jts; j <= jte; j += 1) { loop4_cycle: ; // L351
    for (i = its; i <= ite; i += 1) { loop5_cycle: ; // L352
    if ((xice(i, j) >= xice_threshold)) { // L354
    seaice(i, j) = 1.f; // L355
    } else { // L356
    seaice(i, j) = 0.f; // L357
    } // L358
    loop5_continue: ;
    } loop5_exit: ; // L359
    loop4_continue: ;
    } loop4_exit: ; // L360
    cq = __fsub_rn(173.15f, .05f); // L364
    r273 = __fdiv_rn(1.f, 273.15f); // L365
    r61 = __fmul_rn(6.1153f, 0.62198f); // L366
    arp = __fdiv_rn(__fmul_rn(77455.f, 41.9f), 461.525f); // L367
    brp = __fdiv_rn(__fmul_rn(64.f, 41.9f), 461.525f); // L368
    k = 5002; // L370-380: tbq(1:5001) is ruc_tbq_table (ruc_tbq_init); the DO loop leaves k past its bound
    if ((ktau == 1)) { // L385
    if (((!restart) && (!cycling))) { // L387
    for (j = jts; j <= jte; j += 1) { loop7_cycle: ; // L389
    for (i = its; i <= ite; i += 1) { loop8_cycle: ; // L390
    if ((seaice(i, j) > 0.5f)) { // L392
    if (((soilt1(i, j) < 170.f) || (soilt1(i, j) > 400.f))) { // L396
    if ((snowc(i, j) > 0.f)) { // L397
    soilt1(i, j) = __fmul_rn(0.5f, __fadd_rn(soilt(i, j), tso(i, 1, j))); // L398
    if (false) { // L399
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L400
    } // L403
    } else { // L404
    soilt1(i, j) = tso(i, 1, j); // L405
    } // L406
    } // L407
    tsnav(i, j) = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(soilt(i, j), tso(i, 1, j))), 273.15f); // L409
    patmb = __fmul_rn(p8w(i, kms, j), 1.e-2f); // L410
    qsg(i, j) = __fdiv_rn(ice_qsn(tb, arena, soilt(i, j), tbq), patmb); // L411
    if (((qcg(i, j) < 0.f) || (qcg(i, j) > 0.1f))) { // L412
    qcg(i, j) = 0.f; // L413
    if (false) { // L414
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L415
    } // L417
    } // L418
    if (((qvg(i, j) <= 0.f) || (qvg(i, j) > 0.1f))) { // L419
    qvg(i, j) = qsg(i, j); // L420
    if (false) { // L421
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L422
    } // L424
    } // L425
    qsfc(i, j) = __fdiv_rn(qvg(i, j), __fadd_rn(1.f, qvg(i, j))); // L426
    smelt(i, j) = 0.f; // L427
    snom(i, j) = 0.f; // L428
    snowfallac(i, j) = 0.f; // L429
    acsnow(i, j) = 0.f; // L430
    precipfr(i, j) = 0.f; // L431
    rhosnf(i, j) = (-1.e3f); // L432
    snflx(i, j) = 0.f; // L433
    dew(i, j) = 0.f; // L434
    zntl(i, j) = 0.f; // L435
    runoff1(i, j) = 0.f; // L436
    runoff2(i, j) = 0.f; // L437
    sfcrunoff(i, j) = 0.f; // L438
    acrunoff(i, j) = 0.f; // L439
    emissl(i, j) = 0.f; // L440
    chklowq(i, j) = 1.f; // L445
    infiltr(i, j) = 0.f; // L446
    snoh(i, j) = 0.f; // L447
    edir(i, j) = 0.f; // L448
    ec(i, j) = 0.f; // L449
    ett(i, j) = 0.f; // L450
    sublim(i, j) = 0.f; // L451
    sflx(i, j) = 0.f; // L452
    smf(i, j) = 0.f; // L453
    evapl(i, j) = 0.f; // L454
    prcpl(i, j) = 0.f; // L455
    } // L456
    loop8_continue: ;
    } loop8_exit: ; // L457
    loop7_continue: ;
    } loop7_exit: ; // L458
    for (k = 1; k <= nsl; k += 1) { loop9_cycle: ; // L460
    soilice(k) = 1.f; // L461
    soiliqw(k) = 0.f; // L462
    loop9_continue: ;
    } loop9_exit: ; // L463
    } // L464
    } // L465
    prcpms = 0.f; // L469
    newsnms = 0.f; // L470
    prcpncliq = 0.f; // L471
    prcpculiq = 0.f; // L472
    prcpncfr = 0.f; // L473
    prcpcufr = 0.f; // L474
    for (j = jts; j <= jte; j += 1) { loop10_cycle: ; // L478
    for (i = its; i <= ite; i += 1) { loop11_cycle: ; // L479
    if ((seaice(i, j) > 0.5f)) { // L480
    if ((globalcells(i) == tb.targetcell)) { // L483
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L484
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L485
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L486
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L487
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L488
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L489
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L490
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L491
    } // L492
    iland = isice; // L494
    isoil = isltyp(i, j); // L495
    tabs = t3d(i, kms, j); // L496
    qvatm = qv3d(i, kms, j); // L497
    qcatm = qc3d(i, kms, j); // L498
    patm = __fmul_rn(p8w(i, kms, j), 1.e-5f); // L499
    conflx = __fmul_rn(z3d(i, kms, j), 0.5f); // L503
    rho = rho3d(i, kms, j); // L504
    snowrat = 0.f; // L506
    grauprat = 0.f; // L507
    icerat = 0.f; // L508
    curat = 0.f; // L509
    if (frpcpn) { // L511
    prcpncliq = __fmul_rn(rainncv(i, j), __fsub_rn(1.f, frzfrac(i, j))); // L513
    prcpncfr = __fmul_rn(rainncv(i, j), frzfrac(i, j)); // L514
    if (((frzfrac(i, j) > 0.f) && (tabs < 273.f))) { // L518
    prcpculiq = ruc_max<float>(0.f, __fmul_rn(__fsub_rn(rainbl(i, j), rainncv(i, j)), __fsub_rn(1.f, frzfrac(i, j)))); // L519
    prcpcufr = ruc_max<float>(0.f, __fmul_rn(__fsub_rn(rainbl(i, j), rainncv(i, j)), frzfrac(i, j))); // L520
    } else { // L521
    if ((tabs < 273.f)) { // L522
    prcpcufr = ruc_max<float>(0.f, __fsub_rn(rainbl(i, j), rainncv(i, j))); // L523
    prcpculiq = 0.f; // L524
    } else { // L525
    prcpcufr = 0.f; // L526
    prcpculiq = ruc_max<float>(0.f, __fsub_rn(rainbl(i, j), rainncv(i, j))); // L527
    } // L528
    } // L529
    prcpms = __fmul_rn(__fdiv_rn(__fadd_rn(prcpncliq, prcpculiq), dt), 1.e-3f); // L531
    newsnms = __fmul_rn(__fdiv_rn(__fadd_rn(prcpncfr, prcpcufr), dt), 1.e-3f); // L532
    if (has_graupelncv) { // L534
    graupamt = graupelncv(i, j); // L535
    } else { // L536
    graupamt = 0.f; // L537
    } // L538
    if ((__fadd_rn(prcpncfr, prcpcufr) > 0.f)) { // L540
    snowrat = ruc_min<float>(1.f, ruc_max<float>(0.f, __fdiv_rn(snowncv(i, j), __fadd_rn(prcpncfr, prcpcufr)))); // L542
    grauprat = ruc_min<float>(1.f, ruc_max<float>(0.f, __fdiv_rn(graupamt, __fadd_rn(prcpncfr, prcpcufr)))); // L543
    icerat = ruc_min<float>(1.f, ruc_max<float>(0.f, __fdiv_rn(__fsub_rn(__fsub_rn(prcpncfr, snowncv(i, j)), graupamt), __fadd_rn(prcpncfr, prcpcufr)))); // L544
    curat = ruc_min<float>(1.f, ruc_max<float>(0.f, __fdiv_rn(prcpcufr, __fadd_rn(prcpncfr, prcpcufr)))); // L546
    } // L547
    } else { // L558
    if ((tabs <= 273.15f)) { // L559
    prcpms = 0.f; // L560
    newsnms = __fmul_rn(__fdiv_rn(rainbl(i, j), dt), 1.e-3f); // L561
    snowrat = 1.f; // L564
    } else { // L565
    prcpms = __fmul_rn(__fdiv_rn(rainbl(i, j), dt), 1.e-3f); // L566
    newsnms = 0.f; // L567
    } // L568
    } // L569
    precipfr(i, j) = __fmul_rn(__fmul_rn(newsnms, dt), 1.e3f); // L573
    qkms = __fdiv_rn(flqc(i, j), rho); // L576
    tkms = __fdiv_rn(__fdiv_rn(flhc(i, j), rho), __fmul_rn(cp, __fadd_rn(1.f, __fmul_rn(0.84f, qvatm)))); // L577
    snwe = __fmul_rn(snow(i, j), 1.e-3f); // L579
    snhei = snowh(i, j); // L580
    snowfrac = snowc(i, j); // L582
    rhosnfall = rhosnf(i, j); // L583
    zsmain(1) = 0.f; // L585
    zshalf(1) = 0.f; // L586
    for (k = 2; k <= nzs; k += 1) { loop12_cycle: ; // L587
    zsmain(k) = zs(k); // L588
    zshalf(k) = __fmul_rn(0.5f, __fadd_rn(zsmain((k - 1)), zsmain(k))); // L589
    loop12_continue: ;
    } loop12_exit: ; // L590
    nzs1 = (nzs - 1); // L595
    if (false) { // L597
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L599
    } // L600
    for (k = 2; k <= nzs1; k += 1) { loop13_cycle: ; // L602
    k1 = ((2 * k) - 3); // L603
    k2 = (k1 + 1); // L604
    x = __fdiv_rn(__fdiv_rn(dt, 2.f), __fsub_rn(zshalf((k + 1)), zshalf(k))); // L605
    dtdzs(k1) = __fdiv_rn(x, __fsub_rn(zsmain(k), zsmain((k - 1)))); // L606
    dtdzs2((k - 1)) = x; // L607
    dtdzs(k2) = __fdiv_rn(x, __fsub_rn(zsmain((k + 1)), zsmain(k))); // L608
    loop13_continue: ;
    } loop13_exit: ; // L612
    cw = 4.183e6f; // L614
    c1sn = 0.026f; // L619
    c2sn = 21.f; // L620
    rhonewsn = 200.f; // L624
    if (((snow(i, j) > 0.f) && (snowh(i, j) > 0.f))) { // L625
    rhosn = __fdiv_rn(snow(i, j), snowh(i, j)); // L626
    } else { // L627
    rhosn = 300.f; // L628
    } // L629
    iland = isice; // L631
    isoil = 16; // L632
    znt(i, j) = 0.011f; // L633
    snoalb(i, j) = 0.75f; // L634
    dqm = 1.f; // L635
    ref = 1.f; // L636
    qmin = 0.f; // L637
    wilt = 0.f; // L638
    emissl(i, j) = 0.98f; // L639
    patmb = __fmul_rn(p8w(i, 1, j), 1.e-2f); // L641
    qvg(i, j) = __fdiv_rn(ice_qsn(tb, arena, soilt(i, j), tbq), patmb); // L642
    qsg(i, j) = qvg(i, j); // L643
    qsfc(i, j) = __fdiv_rn(qvg(i, j), __fadd_rn(1.f, qvg(i, j))); // L644
    for (k = 1; k <= nzs; k += 1) { loop14_cycle: ; // L646
    tso(i, k, j) = ruc_min<float>(271.4f, tso(i, k, j)); // L647
    tso1d(k) = tso(i, k, j); // L648
    loop14_continue: ;
    } loop14_exit: ; // L649
    if ((globalcells(i) == tb.targetcell)) { // L652
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L653
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L654
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L656
    } // L657
    ruc_assign(rstoch_temp, rstoch(i, RRange{0,0}, j)); // L659
    ruc_assign(field_sf_temp, field_sf_loc(i, RRange{0,0}, j)); // L660
    ice_icetmp(tb, arena, spp_lsm, rstoch_temp, field_sf_temp, dt, ktau, conflx, i, j, nzs, nddzs, iland, isoil, xland(i, j), ivgtyp(i, j), isltyp(i, j), prcpms, newsnms, snowrat, grauprat, icerat, curat, patm, tabs, qvatm, qcatm, rho, glw(i, j), gsw(i, j), emissl(i, j), qkms, tkms, alb(i, j), znt(i, j), snoalb(i, j), albbck(i, j), myjpbl, seaice(i, j), isice, zsmain, zshalf, dtdzs, dtdzs2, tbq, cp, rovcp, g0, lv, stbolt, cw, c1sn, c2sn, tso1d, soilt(i, j), soilt1(i, j), tsnav(i, j), rhosn, rhonewsn, rhosnfall, snwe, snhei, snowfrac, rsm, snweprint, snheiprint, dew(i, j), qvg(i, j), qsg(i, j), qcg(i, j), smelt(i, j), snoh(i, j), snflx(i, j), snom(i, j), snowfallac(i, j), acsnow(i, j), edir(i, j), ec(i, j), ett(i, j), qfx(i, j), lh(i, j), hfx(i, j), sflx(i, j), sublim(i, j), evapl(i, j), prcpl(i, j), budget(i, j), runoff1(i, j), runoff2(i, j), soilice, soiliqw, infiltrp, smf(i, j), globalcells(i)); // L664
    ruc_assign(field_sf_loc(i, RRange{0,0}, j), field_sf_temp); // L691
    if ((spp_lsm == 1)) { // L695
    for (k = 1; k <= nsl; k += 1) { loop15_cycle: ; // L696
    field_sf(i, k, j) = field_sf_loc(i, k, j); // L697
    loop15_continue: ;
    } loop15_exit: ; // L698
    } // L699
    for (k = 1; k <= nzs; k += 1) { loop16_cycle: ; // L703
    tso(i, k, j) = tso1d(k); // L704
    loop16_continue: ;
    } loop16_exit: ; // L705
    z0(i, j) = znt(i, j); // L707
    patmb = __fmul_rn(p8w(i, 1, j), 1.e-2f); // L708
    q2sat = __fdiv_rn(ice_qsn(tb, arena, tabs, tbq), patmb); // L709
    qsfc(i, j) = __fdiv_rn(qvg(i, j), __fadd_rn(1.f, qvg(i, j))); // L710
    if ((snow(i, j) == 0.f)) { emissl(i, j) = 0.98f; } // L712
    emiss(i, j) = emissl(i, j); // L713
    snow(i, j) = __fmul_rn(snwe, 1000.f); // L715
    snowh(i, j) = snhei; // L716
    if (false) { // L718
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L719
    } // L720
    grdflx(i, j) = __fmul_rn((-1.f), sflx(i, j)); // L721
    if (((snowfrac > 0.f) && (xice(i, j) >= xice_threshold))) { // L724
    snowfrac = __fmul_rn(snowfrac, xice(i, j)); // L725
    } // L726
    snowc(i, j) = snowfrac; // L728
    rhosnf(i, j) = rhosnfall; // L731
    sfcevp(i, j) = __fadd_rn(sfcevp(i, j), __fmul_rn(qfx(i, j), dt)); // L734
    if (false) { // L736
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L737
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L739
    } // L740
    } // L742
    loop11_continue: ;
    } loop11_exit: ; // L744
    loop10_continue: ;
    } loop10_exit: ; // L745
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_ice.F90:753-1321
RUCDEV void ice_icetmp(Tables &tb, Arena &arena, int  spp_lsm, FArray<float> rstochcol, FArray<float> fieldcol_sf, float  delt, int  ktau, float  conflx, int  i, int  j, int  nzs, int  nddzs, int & iland, int & isoil, float  xland, int  ivgtyp, int  isltyp, float & prcpms, float & newsnms, float & snowrat, float & grauprat, float & icerat, float & curat, float  patm, float  tabs, float  qvatm, float  qcatm, float  rho, float  glw, float  gsw, float & emiss, float  qkms, float  tkms, float & alb, float & znt, float  alb_snow, float  alb_snow_free, int  myj, float  seaice, int  isice, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  cp, float  rovcp, float  g0, float  lv, float  stbolt, float  cw, float  c1sn, float  c2sn, FArray<float> ts1d, float & soilt, float & soilt1, float & tsnav, float & rhosn, float & rhonewsn, float & rhosnfall, float & snwe, float & snhei, float & snowfrac, float & rsm, float & snweprint, float & snheiprint, float & dew, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & snom, float & snowfallac, float & acsnow, float & edir1, float & ec1, float & ett1, float & eeta, float & qfx, float & hfx, float & s, float & sublim, float & evapl, float & prcpl, float & fltot, float & runoff1, float & runoff2, FArray<float> soilice, FArray<float> soiliqw, float & infiltr, float & smf, int  globalcellid) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int ilands;
    FArray<float> tice = arena.alloc<float>(1, nzs);
    FArray<float> rhosice = arena.alloc<float>(1, nzs);
    FArray<float> capice = arena.alloc<float>(1, nzs);
    FArray<float> thdifice = arena.alloc<float>(1, nzs);
    FArray<float> ts1ds = arena.alloc<float>(1, nzs);
    float dews;
    float edir1s;
    float ec1s;
    float csts;
    float ett1s;
    float eetas;
    float evapls;
    float infiltrs;
    float prcpls;
    float qvgs;
    float qsgs;
    float qcgs;
    float qfxs;
    float hfxs;
    float fltots;
    float runoff1s;
    float runoff2s;
    float ss;
    float soilts;
    int k;
    int ilnb = 0; // Native leaves this local undefined (module_ruc_land.F90:1467 / module_ruc_ice.F90 sfctmp/icetmp); snowtemp only sets it for snow at least snth deep and reads it at :4583 for thinner snow, so native's tsnav there comes from stale stack words. Hex fixes it at 0 (the one-layer formula) so the port is deterministic; native-undefined residue, see MATCH.md.
    float bsn;
    float xsn;
    float rainf;
    float snth;
    float newsn;
    float t3;
    float upflux;
    float xinet;
    float snhei_crit;
    float snhei_crit_newsn;
    float keep_snow_albedo;
    float snowfracnewsn;
    float newsnowratio;
    float dd1;
    float snowfrac2;
    float m;
    float rhonewgr;
    float rhonewice;
    float rnet;
    float gswnew;
    float gswin;
    float emissn;
    float zntsn;
    float emiss_snowfree;
    float snow_mosaic;
    float snfr;
    float cice;
    float albice;
    float albsn;
    float drip;
    int ilsnow = 99;
    if ((globalcellid == tb.targetcell)) { // L946
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L947
    } // L948
    snhei_crit = __fdiv_rn(__fmul_rn(0.01601f, rhowater), rhosn); // L960
    snhei_crit_newsn = __fdiv_rn(__fmul_rn(0.0005f, rhowater), rhosn); // L961
    zntsn = 0.011f; // L963
    snow_mosaic = 0.f; // L965
    snfr = 1.f; // L966
    newsn = 0.f; // L967
    newsnowratio = 0.f; // L968
    snowfracnewsn = 0.f; // L969
    rhonewsn = 100.f; // L970
    if ((snhei == 0.f)) { snowfrac = 0.f; } // L971
    smelt = 0.f; // L972
    rainf = 0.f; // L973
    rsm = 0.f; // L974
    infiltr = 0.f; // L975
    drip = 0.f; // L976
    smf = 0.f; // L977
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L980
    tice(k) = 0.f; // L981
    rhosice(k) = 0.f; // L982
    cice = 0.f; // L983
    capice(k) = 0.f; // L984
    thdifice(k) = 0.f; // L985
    loop1_continue: ;
    } loop1_exit: ; // L986
    gswnew = gsw; // L988
    gswin = __fdiv_rn(gsw, __fsub_rn(1.f, alb)); // L989
    albice = alb_snow_free; // L990
    albsn = alb_snow; // L991
    emissn = 0.98f; // L992
    emiss_snowfree = 0.98f; // L993
    for (k = 1; k <= nzs; k += 1) { loop2_cycle: ; // L999
    tice(k) = __fsub_rn(ts1d(k), 273.15f); // L1000
    rhosice(k) = __fdiv_rn(917.6f, __fsub_rn(1, __fmul_rn(0.000165f, tice(k)))); // L1001
    cice = __fadd_rn(2115.85f, __fmul_rn(7.7948f, tice(k))); // L1002
    capice(k) = __fmul_rn(cice, rhosice(k)); // L1003
    thdifice(k) = __fdiv_rn(2.260872f, capice(k)); // L1004
    loop2_continue: ;
    } loop2_exit: ; // L1005
    albice = ruc_min<float>(alb_snow_free, ruc_max<float>(__fsub_rn(alb_snow_free, 0.05f), __fsub_rn(alb_snow_free, __fdiv_rn(__fmul_rn(0.1f, __fadd_rn(tice(1), 10.f)), 10.f)))); // L1010
    if ((globalcellid == tb.targetcell)) { // L1014
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1015
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1016
    } // L1018
    if ((snhei > __fdiv_rn(__fmul_rn(0.0081f, 1.e3f), rhosn))) { // L1020
    bsn = __fmul_rn(__fmul_rn(__fdiv_rn(delt, 3600.f), c1sn), gfk_exp_fma(__fsub_rn(__fmul_rn(0.08f, ruc_min<float>(0.f, tsnav)), __fmul_rn(__fmul_rn(c2sn, rhosn), 1.e-3f)))); // L1022
    if ((__fmul_rn(__fmul_rn(bsn, snwe), 100.f) > 1.e-4f)) { // L1023
    xsn = __fdiv_rn(__fmul_rn(rhosn, __fsub_rn(gfk_exp_fma(__fmul_rn(__fmul_rn(bsn, snwe), 100.f)), 1.f)), __fmul_rn(__fmul_rn(bsn, snwe), 100.f)); // L1024
    rhosn = ruc_min<float>(ruc_max<float>(58.8f, xsn), 500.f); // L1025
    } // L1026
    } // L1027
    if ((snowfrac < 0.75f)) { snow_mosaic = 1.f; } // L1030
    newsn = __fmul_rn(newsnms, delt); // L1032
    if ((newsn > 0.f)) { // L1034
    if (false) { // L1036
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1037
    } // L1038
    newsnowratio = ruc_min<float>(1.f, __fdiv_rn(newsn, __fadd_rn(snwe, newsn))); // L1040
    rhonewsn = ruc_min<float>(125.f, __fdiv_rn(1000.0f, ruc_max<float>(8.f, __fmul_rn(17.f, sfire_tanhf(__fmul_rn(__fsub_rn(276.65f, tabs), 0.15f)))))); // L1043
    rhonewgr = ruc_min<float>(500.f, __fdiv_rn(rhowater, ruc_max<float>(2.f, __fmul_rn(3.5f, sfire_tanhf(__fmul_rn(__fsub_rn(274.15f, tabs), 0.3333f)))))); // L1044
    rhonewice = rhonewsn; // L1045
    rhosnfall = ruc_min<float>(500.f, ruc_max<float>(58.8f, __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(rhonewsn, snowrat), __fmul_rn(rhonewgr, grauprat)), __fmul_rn(rhonewice, icerat)), __fmul_rn(rhonewgr, curat)))); // L1050
    rhonewsn = rhosnfall; // L1054
    xsn = __fdiv_rn(__fadd_rn(__fmul_rn(rhosn, snwe), __fmul_rn(rhonewsn, newsn)), __fadd_rn(snwe, newsn)); // L1059
    rhosn = ruc_min<float>(ruc_max<float>(58.8f, xsn), 500.f); // L1061
    } // L1063
    if ((prcpms > 0.f)) { rainf = 1.f; } // L1065
    drip = 0.f; // L1067
    if ((newsn > 0.f)) { // L1069
    snwe = ruc_max<float>(0.f, __fadd_rn(snwe, newsn)); // L1070
    } // L1071
    snhei = __fdiv_rn(__fmul_rn(snwe, rhowater), rhosn); // L1072
    newsn = __fdiv_rn(__fmul_rn(newsn, rhowater), rhonewsn); // L1073
    if ((snhei > 0.0f)) { // L1075
    iland = isice; // L1077
    snowfrac = ruc_min<float>(1.f, __fdiv_rn(snhei, __fmul_rn(2.f, snhei_crit))); // L1078
    if ((newsn > 0.f)) { // L1080
    snowfracnewsn = ruc_min<float>(1.f, __fdiv_rn(snowfallac, snhei_crit_newsn)); // L1081
    } // L1082
    if ((snowfrac < 0.75f)) { snow_mosaic = 1.f; } // L1084
    keep_snow_albedo = 0.f; // L1086
    if (((snowfracnewsn > 0.99f) && (rhosnfall < 450.f))) { // L1087
    keep_snow_albedo = 1.f; // L1089
    snow_mosaic = 0.f; // L1090
    } // L1091
    if ((globalcellid == tb.targetcell)) { // L1094
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1095
    } // L1097
    if ((snow_mosaic == 1.f)) { // L1100
    albsn = alb_snow; // L1101
    emiss = emissn; // L1102
    } else { // L1103
    albsn = ruc_max<float>(__fmul_rn(keep_snow_albedo, alb_snow), ruc_min<float>(__fadd_rn(albice, __fmul_rn(__fsub_rn(alb_snow, albice), snowfrac)), alb_snow)); // L1104
    emiss = ruc_max<float>(__fmul_rn(keep_snow_albedo, emissn), ruc_min<float>(__fadd_rn(emiss_snowfree, __fmul_rn(__fsub_rn(emissn, emiss_snowfree), snowfrac)), emissn)); // L1106
    } // L1109
    if ((globalcellid == tb.targetcell)) { // L1112
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1113
    } // L1114
    if (((albsn < alb_snow) || (keep_snow_albedo == 1.f))) { // L1118
    alb = albsn; // L1119
    } else { // L1120
    alb = ruc_min<float>(albsn, ruc_max<float>(__fsub_rn(albsn, __fdiv_rn(__fmul_rn(__fmul_rn(0.15f, albsn), __fsub_rn(soilt, 263.15f)), __fsub_rn(273.15f, 263.15f))), __fsub_rn(albsn, 0.1f))); // L1122
    } // L1124
    if ((snow_mosaic == 1.f)) { // L1126
    gswnew = __fmul_rn(gswin, __fsub_rn(1.f, albice)); // L1131
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, soilt), soilt), soilt); // L1133
    upflux = __fmul_rn(t3, soilt); // L1134
    xinet = __fmul_rn(emiss_snowfree, __fsub_rn(glw, upflux)); // L1135
    rnet = __fadd_rn(gswnew, xinet); // L1136
    if ((globalcellid == tb.targetcell)) { // L1138
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1139
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1140
    } // L1141
    for (k = 1; k <= nzs; k += 1) { loop3_cycle: ; // L1142
    ts1ds(k) = ts1d(k); // L1143
    loop3_continue: ;
    } loop3_exit: ; // L1144
    soilts = soilt; // L1145
    qvgs = qvg; // L1146
    qsgs = qsg; // L1147
    qcgs = qcg; // L1148
    smelt = 0.f; // L1149
    runoff1s = 0.f; // L1150
    runoff2s = 0.f; // L1151
    ice_sice(tb, arena, i, j, iland, isoil, delt, ktau, conflx, nzs, nddzs, prcpms, rainf, patm, qvatm, qcatm, glw, gswnew, 0.98f, rnet, qkms, tkms, rho, myj, globalcellid, tice, rhosice, capice, thdifice, zsmain, zshalf, dtdzs, dtdzs2, tbq, lv, cp, rovcp, cw, stbolt, tabs, ts1ds, dews, soilts, qvgs, qsgs, qcgs, eetas, qfxs, hfxs, ss, evapls, prcpls, fltots); // L1153
    edir1 = __fmul_rn(eeta, 1.e-3f); // L1167
    ec1 = 0.f; // L1168
    ett1 = 0.f; // L1169
    runoff1 = prcpms; // L1170
    runoff2 = 0.f; // L1171
    infiltr = 0.f; // L1172
    if ((globalcellid == tb.targetcell)) { // L1175
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1176
    } // L1177
    } // L1179
    gswnew = __fmul_rn(gswin, __fsub_rn(1.f, alb)); // L1183
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, soilt), soilt), soilt); // L1185
    upflux = __fmul_rn(t3, soilt); // L1186
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L1187
    rnet = __fadd_rn(gswnew, xinet); // L1188
    if ((globalcellid == tb.targetcell)) { // L1190
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1191
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1192
    } // L1194
    if ((snow_mosaic == 1.f)) { // L1196
    snfr = 1.f; // L1197
    } else { // L1198
    snfr = snowfrac; // L1199
    } // L1200
    ice_snowseaice(tb, arena, i, j, isoil, delt, ktau, conflx, nzs, nddzs, rhonewsn, snhei_crit, iland, prcpms, rainf, newsn, snhei, snwe, snfr, rhosn, patm, qvatm, qcatm, glw, gswnew, emiss, rnet, qkms, tkms, rho, myj, globalcellid, alb, znt, tice, rhosice, capice, thdifice, zsmain, zshalf, dtdzs, dtdzs2, tbq, lv, cp, rovcp, cw, stbolt, tabs, ilnb, snweprint, snheiprint, rsm, ts1d, dew, soilt, soilt1, tsnav, qvg, qsg, qcg, smelt, snoh, snflx, snom, eeta, qfx, hfx, s, sublim, prcpl, fltot); // L1202
    edir1 = __fmul_rn(eeta, 1.e-3f); // L1221
    ec1 = 0.f; // L1222
    ett1 = 0.f; // L1223
    runoff1 = smelt; // L1224
    runoff2 = 0.f; // L1225
    infiltr = 0.f; // L1226
    if ((snow_mosaic == 1.f)) { // L1228
    for (k = 1; k <= nzs; k += 1) { loop4_cycle: ; // L1231
    ts1d(k) = __fadd_rn(__fmul_rn(ts1ds(k), __fsub_rn(1.f, snowfrac)), __fmul_rn(ts1d(k), snowfrac)); // L1232
    loop4_continue: ;
    } loop4_exit: ; // L1233
    dew = __fadd_rn(__fmul_rn(dews, __fsub_rn(1.f, snowfrac)), __fmul_rn(dew, snowfrac)); // L1234
    soilt = __fadd_rn(__fmul_rn(soilts, __fsub_rn(1.f, snowfrac)), __fmul_rn(soilt, snowfrac)); // L1235
    qvg = __fadd_rn(__fmul_rn(qvgs, __fsub_rn(1.f, snowfrac)), __fmul_rn(qvg, snowfrac)); // L1236
    qsg = __fadd_rn(__fmul_rn(qsgs, __fsub_rn(1.f, snowfrac)), __fmul_rn(qsg, snowfrac)); // L1237
    qcg = __fadd_rn(__fmul_rn(qcgs, __fsub_rn(1.f, snowfrac)), __fmul_rn(qcg, snowfrac)); // L1238
    eeta = __fadd_rn(__fmul_rn(eetas, __fsub_rn(1.f, snowfrac)), __fmul_rn(eeta, snowfrac)); // L1239
    qfx = __fadd_rn(__fmul_rn(qfxs, __fsub_rn(1.f, snowfrac)), __fmul_rn(qfx, snowfrac)); // L1240
    hfx = __fadd_rn(__fmul_rn(hfxs, __fsub_rn(1.f, snowfrac)), __fmul_rn(hfx, snowfrac)); // L1241
    s = __fadd_rn(__fmul_rn(ss, __fsub_rn(1.f, snowfrac)), __fmul_rn(s, snowfrac)); // L1242
    sublim = eeta; // L1243
    prcpl = __fadd_rn(__fmul_rn(prcpls, __fsub_rn(1.f, snowfrac)), __fmul_rn(prcpl, snowfrac)); // L1244
    fltot = __fadd_rn(__fmul_rn(fltots, __fsub_rn(1.f, snowfrac)), __fmul_rn(fltot, snowfrac)); // L1245
    alb = ruc_max<float>(__fmul_rn(keep_snow_albedo, alb), ruc_min<float>(__fadd_rn(albice, __fmul_rn(__fsub_rn(alb, alb_snow_free), snowfrac)), alb)); // L1247
    emiss = ruc_max<float>(__fmul_rn(keep_snow_albedo, emissn), ruc_min<float>(__fadd_rn(emiss_snowfree, __fmul_rn(__fsub_rn(emissn, emiss_snowfree), snowfrac)), emissn)); // L1250
    runoff1 = __fadd_rn(__fmul_rn(runoff1s, __fsub_rn(1.f, snowfrac)), __fmul_rn(runoff1, snowfrac)); // L1254
    runoff2 = __fadd_rn(__fmul_rn(runoff2s, __fsub_rn(1.f, snowfrac)), __fmul_rn(runoff2, snowfrac)); // L1255
    if (false) { // L1256
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1257
    } // L1258
    } // L1260
    if ((snhei == 0.f)) { // L1262
    alb = alb_snow_free; // L1264
    iland = isice; // L1265
    } else { // L1266
    snowfrac = ruc_min<float>(1.f, __fdiv_rn(snhei, __fmul_rn(2.f, snhei_crit))); // L1268
    } // L1269
    snowfallac = __fadd_rn(snowfallac, newsn); // L1273
    acsnow = snowfallac; // L1275
    } else { // L1276
    snheiprint = 0.f; // L1278
    snweprint = 0.f; // L1279
    smelt = 0.f; // L1280
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, soilt), soilt), soilt); // L1281
    upflux = __fmul_rn(t3, soilt); // L1282
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L1283
    if ((alb != albice)) { gswnew = __fmul_rn(__fdiv_rn(gsw, __fsub_rn(1.f, alb)), __fsub_rn(1.f, albice)); } // L1287
    alb = albice; // L1288
    rnet = __fadd_rn(gswnew, xinet); // L1289
    if ((globalcellid == tb.targetcell)) { // L1292
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1293
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1294
    } // L1295
    ice_sice(tb, arena, i, j, iland, isoil, delt, ktau, conflx, nzs, nddzs, prcpms, rainf, patm, qvatm, qcatm, glw, gswnew, emiss, rnet, qkms, tkms, rho, myj, globalcellid, tice, rhosice, capice, thdifice, zsmain, zshalf, dtdzs, dtdzs2, tbq, lv, cp, rovcp, cw, stbolt, tabs, ts1d, dew, soilt, qvg, qsg, qcg, eeta, qfx, hfx, s, evapl, prcpl, fltot); // L1297
    edir1 = __fmul_rn(eeta, 1.e-3f); // L1311
    ec1 = 0.f; // L1312
    ett1 = 0.f; // L1313
    runoff1 = prcpms; // L1314
    runoff2 = 0.f; // L1315
    infiltr = 0.f; // L1316
    } // L1318
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_ice.F90:1324-1588
RUCDEV void ice_sice(Tables &tb, Arena &arena, int  i, int  j, int  iland, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, float & prcpms, float & rainf, float  patm, float  qvatm, float  qcatm, float  glw, float  gsw, float  emiss, float & rnet, float  qkms, float  tkms, float  rho, int  myj, int  globalcellid, FArray<float> tice, FArray<float> rhosice, FArray<float> capice, FArray<float> thdifice, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  xlv, float & cp, float & rovcp, float  cw, float & stbolt, float & tabs, FArray<float> tso, float & dew, float & soilt, float & qvg, float & qsg, float & qcg, float & eeta, float & qfx, float & hfx, float & s, float & evapl, float & prcpl, float & fltot) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float x;
    float x1;
    float x2;
    float x4;
    float tn;
    float denom;
    float t3;
    float upflux;
    float xinet;
    float g0;
    float lv;
    float xlmelt;
    float dzstop;
    float epot;
    float ft;
    float fq;
    float hft;
    float ras;
    float cvw;
    float fkt;
    float d1;
    float d2;
    float d9;
    float d10;
    float did;
    float r211;
    float r21;
    float r22;
    float r6;
    float r7;
    float d11;
    float pi;
    float h;
    float fkq;
    float r210;
    float aa;
    float bb;
    float pp;
    float q1;
    float qs1;
    float ts1;
    float tq2;
    float tx2;
    float tdenom;
    float qgold;
    float snoh;
    float aa1;
    float rhcs;
    float icemelt;
    FArray<float> cotso = arena.alloc<float>(1, nzs);
    FArray<float> rhtso = arena.alloc<float>(1, nzs);
    int nzs1;
    int nzs2;
    int k;
    int k1;
    int kn;
    int kk;
    xlmelt = 3.35e+5f; // L1429
    cvw = cw; // L1430
    prcpl = prcpms; // L1432
    nzs1 = (nzs - 1); // L1434
    nzs2 = (nzs - 2); // L1435
    dzstop = __fdiv_rn(1.f, __fsub_rn(zsmain(2), zsmain(1))); // L1436
    ras = __fmul_rn(rho, 1.e-3f); // L1437
    for (k = 1; k <= nzs; k += 1) { loop1_cycle: ; // L1439
    cotso(k) = 0.f; // L1440
    rhtso(k) = 0.f; // L1441
    loop1_continue: ;
    } loop1_exit: ; // L1442
    cotso(1) = 0.f; // L1444
    rhtso(1) = tso(nzs); // L1445
    for (k = 1; k <= nzs2; k += 1) { loop2_cycle: ; // L1447
    kn = (nzs - k); // L1448
    k1 = ((2 * kn) - 3); // L1449
    x1 = __fmul_rn(dtdzs(k1), thdifice((kn - 1))); // L1450
    x2 = __fmul_rn(dtdzs((k1 + 1)), thdifice(kn)); // L1451
    ft = __fsub_rn(__fadd_rn(tso(kn), __fmul_rn(x1, __fsub_rn(tso((kn - 1)), tso(kn)))), __fmul_rn(x2, __fsub_rn(tso(kn), tso((kn + 1))))); // L1452
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1), x2), __fmul_rn(x2, cotso(k))); // L1454
    cotso((k + 1)) = __fdiv_rn(x1, denom); // L1455
    rhtso((k + 1)) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso(k))), denom); // L1456
    loop2_continue: ;
    } loop2_exit: ; // L1457
    rhcs = capice(1); // L1461
    h = 1.f; // L1462
    fkt = tkms; // L1463
    d1 = cotso(nzs1); // L1464
    d2 = rhtso(nzs1); // L1465
    tn = soilt; // L1466
    d9 = __fmul_rn(__fmul_rn(thdifice(1), rhcs), dzstop); // L1467
    d10 = __fmul_rn(__fmul_rn(tkms, cp), rho); // L1468
    r211 = __fdiv_rn(__fmul_rn(.5f, conflx), delt); // L1469
    r21 = __fmul_rn(__fmul_rn(r211, cp), rho); // L1470
    r22 = __fdiv_rn(.5f, __fmul_rn(__fmul_rn(thdifice(1), delt), __fmul_rn(dzstop, dzstop))); // L1471
    r6 = __fmul_rn(__fmul_rn(__fmul_rn(emiss, stbolt), .5f), __fmul_rn(__fmul_rn(tn, tn), __fmul_rn(tn, tn))); // L1472
    r7 = __fdiv_rn(r6, tn); // L1473
    d11 = __fadd_rn(rnet, r6); // L1474
    tdenom = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d9, __fadd_rn(__fsub_rn(1.f, d1), r22)), d10), r21), r7), __fmul_rn(__fmul_rn(rainf, cvw), prcpms)); // L1475
    fkq = __fmul_rn(qkms, rho); // L1477
    r210 = __fmul_rn(r211, rho); // L1478
    aa = __fdiv_rn(__fmul_rn(xls, __fadd_rn(fkq, r210)), tdenom); // L1479
    bb = __fdiv_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d10, tabs), __fmul_rn(r21, tn)), __fmul_rn(xls, __fadd_rn(__fmul_rn(qvatm, fkq), __fmul_rn(r210, qvg)))), d11), __fmul_rn(d9, __fadd_rn(d2, __fmul_rn(r22, tn)))), __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), ruc_max<float>(273.15f, tabs))), tdenom); // L1480
    aa1 = aa; // L1484
    pp = __fmul_rn(patm, 1.e3f); // L1485
    aa1 = __fdiv_rn(aa1, pp); // L1486
    if ((globalcellid == tb.targetcell)) { // L1488
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1489
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1490
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1492
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1493
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1495
    } // L1497
    qgold = qsg; // L1498
    ice_vilka(tb, arena, tn, aa1, bb, pp, qs1, ts1, tbq, ktau, i, j, iland, isoil); // L1499
    qvg = qs1; // L1501
    qsg = qs1; // L1502
    tso(1) = ruc_min<float>(271.4f, ts1); // L1503
    qcg = 0.f; // L1504
    soilt = tso(1); // L1507
    for (k = 2; k <= nzs; k += 1) { loop3_cycle: ; // L1509
    kk = ((nzs - k) + 1); // L1510
    tso(k) = ruc_min<float>(271.4f, __fadd_rn(rhtso(kk), __fmul_rn(cotso(kk), tso((k - 1))))); // L1511
    loop3_continue: ;
    } loop3_exit: ; // L1512
    dew = 0.f; // L1514
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, tn), tn), tn); // L1517
    upflux = __fmul_rn(__fmul_rn(t3, 0.5f), __fadd_rn(tn, soilt)); // L1518
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L1519
    hft = __fmul_rn(__fmul_rn(__fmul_rn((-tkms), cp), rho), __fsub_rn(tabs, soilt)); // L1520
    hfx = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-tkms), cp), rho), __fsub_rn(tabs, soilt)), gfk_pow(__fdiv_rn(__fmul_rn(p1000mb, 0.00001f), patm), rovcp)); // L1521
    q1 = __fmul_rn(__fmul_rn((-qkms), ras), __fsub_rn(qvatm, qsg)); // L1523
    if ((q1 <= 0.f)) { // L1524
    if (myj) { // L1526
    eeta = __fmul_rn(__fmul_rn(__fmul_rn((-qkms), ras), __fsub_rn(__fdiv_rn(qvatm, __fadd_rn(1.f, qvatm)), __fdiv_rn(qsg, __fadd_rn(1.f, qsg)))), 1.e3f); // L1528
    if (false) { // L1529
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1530
    } // L1531
    } else { // L1532
    dew = __fmul_rn(qkms, __fsub_rn(qvatm, qsg)); // L1534
    eeta = __fmul_rn((-rho), dew); // L1535
    if (false) { // L1536
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1537
    } // L1538
    } // L1539
    qfx = __fmul_rn(xls, eeta); // L1540
    eeta = __fmul_rn((-rho), dew); // L1541
    } else { // L1542
    if (myj) { // L1544
    eeta = __fmul_rn(__fmul_rn(__fmul_rn((-qkms), ras), __fsub_rn(__fdiv_rn(qvatm, __fadd_rn(1.f, qvatm)), __fdiv_rn(qvg, __fadd_rn(1.f, qvg)))), 1.e3f); // L1546
    if (false) { // L1547
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1548
    } // L1549
    } else { // L1550
    eeta = __fmul_rn(q1, 1.e3f); // L1553
    if (false) { // L1554
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1555
    } // L1556
    } // L1557
    qfx = __fmul_rn(xls, eeta); // L1558
    eeta = __fmul_rn(q1, 1.e3f); // L1559
    } // L1560
    evapl = eeta; // L1562
    s = __fmul_rn(__fmul_rn(__fmul_rn(thdifice(1), capice(1)), dzstop), __fsub_rn(tso(1), tso(2))); // L1564
    snoh = 0.f; // L1566
    x = __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fmul_rn(cp, rho), r211), __fdiv_rn(__fmul_rn(__fmul_rn(rhcs, zsmain(2)), 0.5f), delt)), __fsub_rn(soilt, tn)), __fmul_rn(__fmul_rn(__fmul_rn(xls, rho), r211), __fsub_rn(qsg, qgold))); // L1568
    x = __fsub_rn(x, __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), __fsub_rn(ruc_max<float>(273.15f, tabs), soilt))); // L1570
    icemelt = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(rnet, __fmul_rn(xls, eeta)), hft), s), x); // L1575
    if (false) { // L1576
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1577
    } // L1578
    fltot = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(rnet, __fmul_rn(xls, eeta)), hft), s), x), icemelt); // L1580
    if ((globalcellid == tb.targetcell)) { // L1582
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1583
    } // L1585
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_ice.F90:1591-2317
RUCDEV void ice_snowseaice(Tables &tb, Arena &arena, int  i, int  j, int  isoil, float  delt, int  ktau, float  conflx, int  nzs, int  nddzs, float  rhonewsn, float  snhei_crit, int & iland, float  prcpms, float  rainf, float  newsnow, float & snhei, float & snwe, float & snowfrac, float & rhosn, float  patm, float  qvatm, float  qcatm, float  glw, float  gsw, float & emiss, float & rnet, float  qkms, float  tkms, float  rho, int  myj, int  globalcellid, float & alb, float & znt, FArray<float> tice, FArray<float> rhosice, FArray<float> capice, FArray<float> thdifice, FArray<float> zsmain, FArray<float> zshalf, FArray<float> dtdzs, FArray<float> dtdzs2, FArray<float> tbq, float  xlv, float & cp, float & rovcp, float  cw, float & stbolt, float & tabs, int & ilnb, float & snweprint, float & snheiprint, float & rsm, FArray<float> tso, float & dew, float & soilt, float & soilt1, float & tsnav, float & qvg, float & qsg, float & qcg, float & smelt, float & snoh, float & snflx, float & snom, float & eeta, float & qfx, float & hfx, float & s, float & sublim, float & prcpl, float & fltot) {
#ifdef RUC_ICE_ENTRY_CONTEXT
    // Authentic optional PRE storage for the native INOUT argument.
    // RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_ice.F90:1591-1698
    if (arena.snowtemp_calls) ++*arena.snowtemp_calls;
    if (arena.entry_ilnb) ilnb = *arena.entry_ilnb;
#endif
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float rhonewcsn;
    int nzs1;
    int nzs2;
    int k;
    int k1;
    int kn;
    int kk;
    float x;
    float x1;
    float x2;
    float dzstop;
    float ft;
    float tn;
    float denom;
    float snth;
    float newsn;
    float t3;
    float upflux;
    float xinet;
    float beta;
    float snwepr;
    float epdt;
    float pp;
    float g0;
    float lv;
    float xlvm;
    float xlmelt;
    float epot;
    float fq;
    float hft;
    float q1;
    float ras;
    float rhoice;
    float ci;
    float cvw;
    float riw;
    float deltsn;
    float h;
    float rhocsn;
    float thdifsn;
    float xsn;
    float ddzsn;
    float x1sn;
    float d1sn;
    float d2sn;
    float d9sn;
    float r22sn;
    float cotsn;
    float rhtsn;
    float xsn1;
    float ddzsn1;
    float x1sn1;
    float ftsnow;
    float denomsn;
    float fso;
    float fsn;
    float fkt;
    float d1;
    float d2;
    float d9;
    float d10;
    float did;
    float r211;
    float r21;
    float r22;
    float r6;
    float r7;
    float d11;
    float fkq;
    float r210;
    float aa;
    float bb;
    float qs1;
    float ts1;
    float tq2;
    float tx2;
    float tdenom;
    float aa1;
    float rhcs;
    float h1;
    float tsob;
    float snprim;
    float snodif;
    float soh;
    float tnold;
    float qgold;
    float snohgnew;
    FArray<float> cotso = arena.alloc<float>(1, nzs);
    FArray<float> rhtso = arena.alloc<float>(1, nzs);
    float rsmfrac;
    float soiltfrac;
    float hsn;
    float icemelt;
    float rr;
    int nmelt;
    xlmelt = 3.35e+5f; // L1732
    xlvm = __fadd_rn(xlv, xlmelt); // L1734
    deltsn = __fdiv_rn(__fmul_rn(0.05f, 1.e3f), rhosn); // L1744
    snth = __fdiv_rn(__fmul_rn(0.01f, 1.e3f), rhosn); // L1745
    if ((snhei >= __fadd_rn(deltsn, snth))) { // L1749
    if ((__fsub_rn(__fsub_rn(snhei, deltsn), snth) < snth)) { deltsn = __fmul_rn(0.5f, __fsub_rn(snhei, snth)); } // L1750
    if (false) { // L1751
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1752
    } // L1754
    } // L1755
    rhoice = 900.f; // L1757
    ci = __fmul_rn(rhoice, 2100.f); // L1758
    ras = __fmul_rn(rho, 1.e-3f); // L1759
    riw = __fmul_rn(rhoice, 1.e-3f); // L1760
    xlmelt = 3.35e+5f; // L1762
    rhocsn = __fmul_rn(2090.f, rhosn); // L1763
    rhonewcsn = __fmul_rn(2090.f, rhonewsn); // L1764
    thdifsn = __fdiv_rn(0.265f, rhocsn); // L1765
    ras = __fmul_rn(rho, 1.e-3f); // L1766
    soiltfrac = soilt; // L1768
    smelt = 0.f; // L1770
    soh = 0.f; // L1771
    snodif = 0.f; // L1772
    snoh = 0.f; // L1773
    snohgnew = 0.f; // L1774
    rsm = 0.f; // L1775
    rsmfrac = 0.f; // L1776
    fsn = 1.f; // L1777
    fso = 0.f; // L1778
    cvw = cw; // L1779
    nzs1 = (nzs - 1); // L1781
    nzs2 = (nzs - 2); // L1782
    qgold = qsg; // L1784
    tnold = soilt; // L1785
    dzstop = __fdiv_rn(1.f, __fsub_rn(zsmain(2), zsmain(1))); // L1786
    snweprint = 0.f; // L1788
    snheiprint = 0.f; // L1789
    prcpl = prcpms; // L1790
    h = 1.f; // L1796
    smelt = 0.f; // L1797
    fq = qkms; // L1799
    snhei = __fdiv_rn(__fmul_rn(snwe, 1.e3f), rhosn); // L1800
    snwepr = snwe; // L1801
    beta = 1.f; // L1804
    epot = __fmul_rn((-fq), __fsub_rn(qvatm, qsg)); // L1805
    epdt = __fmul_rn(__fmul_rn(epot, ras), delt); // L1806
    if (((epdt > 0.f) && (snwepr <= epdt))) { // L1807
    beta = __fdiv_rn(snwepr, ruc_max<float>(1.e-8f, epdt)); // L1808
    snwe = 0.f; // L1809
    } // L1810
    cotso(1) = 0.f; // L1816
    rhtso(1) = tso(nzs); // L1817
    for (k = 1; k <= nzs2; k += 1) { loop1_cycle: ; // L1818
    kn = (nzs - k); // L1819
    k1 = ((2 * kn) - 3); // L1820
    x1 = __fmul_rn(dtdzs(k1), thdifice((kn - 1))); // L1821
    x2 = __fmul_rn(dtdzs((k1 + 1)), thdifice(kn)); // L1822
    ft = __fsub_rn(__fadd_rn(tso(kn), __fmul_rn(x1, __fsub_rn(tso((kn - 1)), tso(kn)))), __fmul_rn(x2, __fsub_rn(tso(kn), tso((kn + 1))))); // L1823
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1), x2), __fmul_rn(x2, cotso(k))); // L1825
    cotso((k + 1)) = __fdiv_rn(x1, denom); // L1826
    rhtso((k + 1)) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso(k))), denom); // L1827
    loop1_continue: ;
    } loop1_exit: ; // L1828
    if ((snhei >= snth)) { // L1831
    if ((snhei <= __fadd_rn(deltsn, snth))) { // L1832
    ilnb = 1; // L1834
    snprim = ruc_max<float>(snth, snhei); // L1835
    soilt1 = tso(1); // L1836
    tsob = tso(1); // L1837
    xsn = __fdiv_rn(__fdiv_rn(delt, 2.f), __fadd_rn(zshalf(2), __fmul_rn(0.5f, snprim))); // L1838
    ddzsn = __fdiv_rn(xsn, snprim); // L1839
    x1sn = __fmul_rn(ddzsn, thdifsn); // L1840
    x2 = __fmul_rn(dtdzs(1), thdifice(1)); // L1841
    ft = __fsub_rn(__fadd_rn(tso(1), __fmul_rn(x1sn, __fsub_rn(soilt, tso(1)))), __fmul_rn(x2, __fsub_rn(tso(1), tso(2)))); // L1842
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1sn), x2), __fmul_rn(x2, cotso(nzs1))); // L1844
    cotso(nzs) = __fdiv_rn(x1sn, denom); // L1845
    rhtso(nzs) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso(nzs1))), denom); // L1846
    cotsn = cotso(nzs); // L1847
    rhtsn = rhtso(nzs); // L1848
    tsnav = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(soilt, tso(1))), 273.15f); // L1850
    } else { // L1853
    ilnb = 2; // L1855
    snprim = deltsn; // L1856
    tsob = soilt1; // L1857
    xsn = __fdiv_rn(__fdiv_rn(delt, 2.f), __fmul_rn(0.5f, snhei)); // L1858
    xsn1 = __fdiv_rn(__fdiv_rn(delt, 2.f), __fadd_rn(zshalf(2), __fmul_rn(0.5f, __fsub_rn(snhei, deltsn)))); // L1859
    ddzsn = __fdiv_rn(xsn, deltsn); // L1860
    ddzsn1 = __fdiv_rn(xsn1, __fsub_rn(snhei, deltsn)); // L1861
    x1sn = __fmul_rn(ddzsn, thdifsn); // L1862
    x1sn1 = __fmul_rn(ddzsn1, thdifsn); // L1863
    x2 = __fmul_rn(dtdzs(1), thdifice(1)); // L1864
    ft = __fsub_rn(__fadd_rn(tso(1), __fmul_rn(x1sn1, __fsub_rn(soilt1, tso(1)))), __fmul_rn(x2, __fsub_rn(tso(1), tso(2)))); // L1865
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1sn1), x2), __fmul_rn(x2, cotso(nzs1))); // L1867
    cotso(nzs) = __fdiv_rn(x1sn1, denom); // L1868
    rhtso(nzs) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso(nzs1))), denom); // L1869
    ftsnow = __fsub_rn(__fadd_rn(soilt1, __fmul_rn(x1sn, __fsub_rn(soilt, soilt1))), __fmul_rn(x1sn1, __fsub_rn(soilt1, tso(1)))); // L1870
    denomsn = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1sn), x1sn1), __fmul_rn(x1sn1, cotso(nzs))); // L1872
    cotsn = __fdiv_rn(x1sn, denomsn); // L1873
    rhtsn = __fdiv_rn(__fadd_rn(ftsnow, __fmul_rn(x1sn1, rhtso(nzs))), denomsn); // L1874
    tsnav = __fsub_rn(__fmul_rn(__fdiv_rn(0.5f, snhei), __fadd_rn(__fmul_rn(__fadd_rn(soilt, soilt1), deltsn), __fmul_rn(__fadd_rn(soilt1, tso(1)), __fsub_rn(snhei, deltsn)))), 273.15f); // L1876
    } // L1879
    } // L1880
    if (((snhei < snth) && (snhei > 0.f))) { // L1882
    snprim = __fadd_rn(snhei, zsmain(2)); // L1885
    fsn = __fdiv_rn(snhei, snprim); // L1886
    fso = __fsub_rn(1.f, fsn); // L1887
    soilt1 = tso(1); // L1888
    tsob = tso(2); // L1889
    xsn = __fdiv_rn(__fdiv_rn(delt, 2.f), __fadd_rn(__fsub_rn(zshalf(3), zsmain(2)), __fmul_rn(0.5f, snprim))); // L1890
    ddzsn = __fdiv_rn(xsn, snprim); // L1891
    x1sn = __fmul_rn(ddzsn, __fadd_rn(__fmul_rn(fsn, thdifsn), __fmul_rn(fso, thdifice(1)))); // L1892
    x2 = __fmul_rn(dtdzs(2), thdifice(2)); // L1893
    ft = __fsub_rn(__fadd_rn(tso(2), __fmul_rn(x1sn, __fsub_rn(soilt, tso(2)))), __fmul_rn(x2, __fsub_rn(tso(2), tso(3)))); // L1894
    denom = __fsub_rn(__fadd_rn(__fadd_rn(1.f, x1sn), x2), __fmul_rn(x2, cotso((nzs - 2)))); // L1896
    cotso(nzs1) = __fdiv_rn(x1sn, denom); // L1897
    rhtso(nzs1) = __fdiv_rn(__fadd_rn(ft, __fmul_rn(x2, rhtso((nzs - 2)))), denom); // L1898
    tsnav = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(soilt, tso(1))), 273.15f); // L1899
    cotso(nzs) = cotso(nzs1); // L1901
    rhtso(nzs) = rhtso(nzs1); // L1902
    cotsn = cotso(nzs); // L1903
    rhtsn = rhtso(nzs); // L1904
    } // L1905
    nmelt = 0; // L1910
    snoh = 0.f; // L1911
    epot = __fmul_rn((-qkms), __fsub_rn(qvatm, qsg)); // L1913
    rhcs = capice(1); // L1914
    h = 1.f; // L1915
    fkt = tkms; // L1916
    d1 = cotso(nzs1); // L1917
    d2 = rhtso(nzs1); // L1918
    tn = soilt; // L1919
    d9 = __fmul_rn(__fmul_rn(thdifice(1), rhcs), dzstop); // L1920
    d10 = __fmul_rn(__fmul_rn(tkms, cp), rho); // L1921
    r211 = __fdiv_rn(__fmul_rn(.5f, conflx), delt); // L1922
    r21 = __fmul_rn(__fmul_rn(r211, cp), rho); // L1923
    r22 = __fdiv_rn(.5f, __fmul_rn(__fmul_rn(thdifice(1), delt), __fmul_rn(dzstop, dzstop))); // L1924
    r6 = __fmul_rn(__fmul_rn(__fmul_rn(emiss, stbolt), .5f), __fmul_rn(__fmul_rn(tn, tn), __fmul_rn(tn, tn))); // L1925
    r7 = __fdiv_rn(r6, tn); // L1926
    d11 = __fadd_rn(rnet, r6); // L1927
    if ((snhei >= snth)) { // L1929
    if ((snhei <= __fadd_rn(deltsn, snth))) { // L1930
    d1sn = cotso(nzs); // L1932
    d2sn = rhtso(nzs); // L1933
    } else { // L1934
    d1sn = cotsn; // L1936
    d2sn = rhtsn; // L1937
    } // L1938
    d9sn = __fdiv_rn(__fmul_rn(thdifsn, rhocsn), snprim); // L1939
    r22sn = __fdiv_rn(__fmul_rn(__fmul_rn(snprim, snprim), 0.5f), __fmul_rn(thdifsn, delt)); // L1940
    } // L1941
    if (((snhei < snth) && (snhei > 0.f))) { // L1943
    d1sn = d1; // L1945
    d2sn = d2; // L1946
    d9sn = __fdiv_rn(__fadd_rn(__fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn), __fmul_rn(__fmul_rn(fso, thdifice(1)), rhcs)), snprim); // L1947
    r22sn = __fdiv_rn(__fmul_rn(__fmul_rn(snprim, snprim), 0.5f), __fmul_rn(__fadd_rn(__fmul_rn(fsn, thdifsn), __fmul_rn(fso, thdifice(1))), delt)); // L1949
    } // L1951
    if ((snhei == 0.f)) { // L1953
    d9sn = d9; // L1955
    r22sn = r22; // L1956
    d1sn = d1; // L1957
    d2sn = d2; // L1958
    } // L1959
    tdenom = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d9sn, __fadd_rn(__fsub_rn(1.f, d1sn), r22sn)), d10), r21), r7), __fmul_rn(__fmul_rn(rainf, cvw), prcpms)), __fdiv_rn(__fmul_rn(rhonewcsn, newsnow), delt)); // L1963
    fkq = __fmul_rn(qkms, rho); // L1967
    r210 = __fmul_rn(r211, rho); // L1968
    aa = __fdiv_rn(__fmul_rn(xlvm, __fadd_rn(__fmul_rn(beta, fkq), r210)), tdenom); // L1969
    bb = __fdiv_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(d10, tabs), __fmul_rn(r21, tn)), __fmul_rn(xlvm, __fadd_rn(__fmul_rn(qvatm, __fmul_rn(beta, fkq)), __fmul_rn(r210, qvg)))), d11), __fmul_rn(d9sn, __fadd_rn(d2sn, __fmul_rn(r22sn, tn)))), __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), ruc_max<float>(273.15f, tabs))), __fmul_rn(__fdiv_rn(__fmul_rn(rhonewcsn, newsnow), delt), ruc_min<float>(273.15f, tabs))), tdenom); // L1970
    aa1 = aa; // L1976
    pp = __fmul_rn(patm, 1.e3f); // L1977
    aa1 = __fdiv_rn(aa1, pp); // L1978
    label_212: ;
    ; // L1980
    bb = __fsub_rn(bb, __fdiv_rn(snoh, tdenom)); // L1981
    if ((globalcellid == tb.targetcell)) { // L1983
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1984
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1985
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L1987
    } // L1988
    ice_vilka(tb, arena, tn, aa1, bb, pp, qs1, ts1, tbq, ktau, i, j, iland, isoil); // L1990
    qvg = qs1; // L1992
    qsg = qs1; // L1993
    qcg = 0.f; // L1994
    soilt = ts1; // L1997
    if ((globalcellid == tb.targetcell)) { // L2000
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2001
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2002
    } // L2003
    if ((snhei >= snth)) { // L2005
    if ((snhei > __fadd_rn(deltsn, snth))) { // L2006
    soilt1 = ruc_min<float>(273.15f, __fadd_rn(rhtsn, __fmul_rn(cotsn, soilt))); // L2008
    tso(1) = ruc_min<float>(271.4f, __fadd_rn(rhtso(nzs), __fmul_rn(cotso(nzs), soilt1))); // L2009
    tsob = soilt1; // L2010
    } else { // L2011
    tso(1) = ruc_min<float>(271.4f, __fadd_rn(rhtso(nzs), __fmul_rn(cotso(nzs), soilt))); // L2013
    soilt1 = tso(1); // L2014
    tsob = tso(1); // L2015
    } // L2016
    } else if (((snhei > 0.f) && (snhei < snth))) { // L2017
    tso(2) = ruc_min<float>(271.4f, __fadd_rn(rhtso(nzs1), __fmul_rn(cotso(nzs1), soilt))); // L2019
    tso(1) = ruc_min<float>(271.4f, __fadd_rn(tso(2), __fmul_rn(__fsub_rn(soilt, tso(2)), fso))); // L2020
    soilt1 = tso(1); // L2021
    tsob = tso(2); // L2022
    } else { // L2023
    tso(1) = ruc_min<float>(271.4f, soilt); // L2025
    soilt1 = ruc_min<float>(271.4f, soilt); // L2026
    tsob = tso(1); // L2027
    } // L2028
    if (((snhei > 0.f) && (snhei < snth))) { // L2030
    for (k = 3; k <= nzs; k += 1) { loop2_cycle: ; // L2032
    kk = ((nzs - k) + 1); // L2033
    tso(k) = ruc_min<float>(271.4f, __fadd_rn(rhtso(kk), __fmul_rn(cotso(kk), tso((k - 1))))); // L2034
    loop2_continue: ;
    } loop2_exit: ; // L2035
    } else { // L2036
    for (k = 2; k <= nzs; k += 1) { loop3_cycle: ; // L2037
    kk = ((nzs - k) + 1); // L2038
    tso(k) = ruc_min<float>(271.4f, __fadd_rn(rhtso(kk), __fmul_rn(cotso(kk), tso((k - 1))))); // L2039
    loop3_continue: ;
    } loop3_exit: ; // L2040
    } // L2041
    if ((nmelt == 1)) { goto label_220; } // L2043
    if ((((soilt > 273.15f) && (__fsub_rn(snwepr, __fmul_rn(__fmul_rn(__fmul_rn(beta, epot), ras), delt)) > 0.f)) && (snhei > 0.f))) { // L2046
    nmelt = 1; // L2048
    soiltfrac = __fadd_rn(__fmul_rn(snowfrac, 273.15f), __fmul_rn(__fsub_rn(1.f, snowfrac), ruc_min<float>(271.4f, soilt))); // L2049
    qsg = __fdiv_rn(ice_qsn(tb, arena, soiltfrac, tbq), pp); // L2051
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, tnold), tnold), tnold); // L2052
    upflux = __fmul_rn(__fmul_rn(t3, 0.5f), __fadd_rn(tnold, soiltfrac)); // L2053
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L2054
    epot = __fmul_rn((-qkms), __fsub_rn(qvatm, qsg)); // L2055
    q1 = __fmul_rn(epot, ras); // L2056
    if ((q1 <= 0.f)) { // L2058
    dew = (-epot); // L2060
    qfx = __fmul_rn(__fmul_rn(xlvm, rho), dew); // L2061
    eeta = __fdiv_rn(qfx, xlvm); // L2062
    } else { // L2063
    eeta = __fmul_rn(__fmul_rn(q1, beta), 1.e3f); // L2065
    qfx = __fmul_rn((-xlvm), eeta); // L2067
    } // L2068
    hfx = __fmul_rn(d10, __fsub_rn(tabs, soiltfrac)); // L2070
    if ((snhei >= snth)) { // L2072
    soh = __fdiv_rn(__fmul_rn(__fmul_rn(thdifsn, rhocsn), __fsub_rn(soiltfrac, tsob)), snprim); // L2073
    snflx = soh; // L2074
    } else { // L2075
    soh = __fdiv_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn), __fmul_rn(__fmul_rn(fso, thdifice(1)), rhcs)), __fsub_rn(soiltfrac, tsob)), snprim); // L2076
    snflx = soh; // L2078
    } // L2079
    x = __fadd_rn(__fmul_rn(__fadd_rn(r21, __fmul_rn(d9sn, r22sn)), __fsub_rn(soiltfrac, tnold)), __fmul_rn(__fmul_rn(xlvm, r210), __fsub_rn(qsg, qgold))); // L2080
    snoh = __fadd_rn(__fsub_rn(__fsub_rn(__fadd_rn(__fadd_rn(__fadd_rn(rnet, qfx), hfx), __fmul_rn(__fdiv_rn(__fmul_rn(rhonewcsn, newsnow), delt), __fsub_rn(ruc_min<float>(273.15f, tabs), soiltfrac))), soh), x), __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), __fsub_rn(ruc_max<float>(273.15f, tabs), soiltfrac))); // L2083
    if ((globalcellid == tb.targetcell)) { // L2089
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2090
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2092
    } // L2094
    snoh = ruc_max<float>(0.f, snoh); // L2095
    smelt = __fmul_rn(__fdiv_rn(snoh, xlmelt), 1.e-3f); // L2097
    smelt = ruc_min<float>(smelt, __fsub_rn(__fdiv_rn(snwepr, delt), __fmul_rn(__fmul_rn(beta, epot), ras))); // L2098
    smelt = ruc_max<float>(0.f, smelt); // L2099
    if (false) { // L2101
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2102
    } // L2103
    smelt = ruc_min<float>(smelt, __fmul_rn(5.6e-8f, ruc_max<float>(1.f, __fsub_rn(soilt, 273.15f)))); // L2105
    if (false) { // L2106
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2107
    } // L2108
    rr = __fsub_rn(__fdiv_rn(snwepr, delt), __fmul_rn(__fmul_rn(beta, epot), ras)); // L2111
    smelt = ruc_min<float>(smelt, rr); // L2112
    if (false) { // L2113
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2114
    } // L2115
    snohgnew = __fmul_rn(__fmul_rn(smelt, xlmelt), 1.e3f); // L2116
    snodif = ruc_max<float>(0.f, __fsub_rn(snoh, snohgnew)); // L2117
    snoh = snohgnew; // L2119
    if ((globalcellid == tb.targetcell)) { // L2122
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2123
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2125
    } // L2126
    rsmfrac = ruc_min<float>(0.18f, ruc_max<float>(0.08f, __fmul_rn(__fdiv_rn(snwepr, 0.10f), 0.13f))); // L2129
    if ((snhei > 0.01f)) { // L2130
    rsm = __fmul_rn(__fmul_rn(rsmfrac, smelt), delt); // L2131
    } else { // L2132
    rsm = 0.f; // L2134
    } // L2135
    smelt = ruc_max<float>(0.f, __fsub_rn(smelt, __fdiv_rn(rsm, delt))); // L2137
    if ((globalcellid == tb.targetcell)) { // L2140
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2141
    } // L2143
    snwe = ruc_max<float>(0.f, __fsub_rn(snwepr, __fmul_rn(__fadd_rn(smelt, __fmul_rn(__fmul_rn(beta, epot), ras)), delt))); // L2147
    soilt = soiltfrac; // L2150
    } else { // L2153
    if ((snhei != 0.f)) { // L2154
    epot = __fmul_rn((-qkms), __fsub_rn(qvatm, qsg)); // L2155
    snwe = ruc_max<float>(0.f, __fsub_rn(snwepr, __fmul_rn(__fmul_rn(__fmul_rn(beta, epot), ras), delt))); // L2156
    } // L2158
    } // L2159
    label_220: ;
    ; // L2164
    if (((smelt > 0.f) && (rsm > 0.f))) { // L2166
    if ((snwe <= rsm)) { // L2167
    if ((globalcellid == tb.targetcell)) { // L2169
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2170
    } // L2172
    } else { // L2173
    xsn = __fdiv_rn(__fadd_rn(__fmul_rn(rhosn, __fsub_rn(snwe, rsm)), __fmul_rn(1.e3f, rsm)), snwe); // L2179
    rhosn = ruc_min<float>(ruc_max<float>(58.8f, xsn), 500.f); // L2180
    rhocsn = __fmul_rn(2090.f, rhosn); // L2182
    thdifsn = __fdiv_rn(0.265f, rhocsn); // L2183
    } // L2184
    } // L2185
    snweprint = snwe; // L2187
    snheiprint = __fdiv_rn(__fmul_rn(snweprint, 1.e3f), rhosn); // L2192
    if ((globalcellid == tb.targetcell)) { // L2195
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2196
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2197
    } // L2198
    if ((snhei > 0.f)) { // L2200
    if ((ilnb > 1)) { // L2201
    tsnav = __fsub_rn(__fmul_rn(__fdiv_rn(0.5f, snhei), __fadd_rn(__fmul_rn(__fadd_rn(soilt, soilt1), deltsn), __fmul_rn(__fadd_rn(soilt1, tso(1)), __fsub_rn(snhei, deltsn)))), 273.15f); // L2202
    } else { // L2205
    tsnav = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(soilt, tso(1))), 273.15f); // L2206
    } // L2207
    } // L2208
    dew = 0.f; // L2210
    pp = __fmul_rn(patm, 1.e3f); // L2211
    qsg = __fdiv_rn(ice_qsn(tb, arena, soilt, tbq), pp); // L2212
    epot = __fmul_rn((-fq), __fsub_rn(qvatm, qsg)); // L2213
    if ((epot < 0.f)) { // L2214
    dew = (-epot); // L2216
    } // L2217
    snom = __fadd_rn(snom, __fmul_rn(__fmul_rn(smelt, delt), 1.e3f)); // L2219
    t3 = __fmul_rn(__fmul_rn(__fmul_rn(stbolt, tnold), tnold), tnold); // L2223
    upflux = __fmul_rn(__fmul_rn(t3, 0.5f), __fadd_rn(soilt, tnold)); // L2224
    xinet = __fmul_rn(emiss, __fsub_rn(glw, upflux)); // L2225
    hft = __fmul_rn(__fmul_rn(__fmul_rn((-tkms), cp), rho), __fsub_rn(tabs, soilt)); // L2226
    hfx = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-tkms), cp), rho), __fsub_rn(tabs, soilt)), gfk_pow(__fdiv_rn(__fmul_rn(p1000mb, 0.00001f), patm), rovcp)); // L2227
    q1 = __fmul_rn(__fmul_rn((-fq), ras), __fsub_rn(qvatm, qsg)); // L2229
    if ((q1 < 0.f)) { // L2230
    if (myj) { // L2232
    eeta = __fmul_rn(__fmul_rn(__fmul_rn((-qkms), ras), __fsub_rn(__fdiv_rn(qvatm, __fadd_rn(1.f, qvatm)), __fdiv_rn(qsg, __fadd_rn(1.f, qsg)))), 1.e3f); // L2234
    } else { // L2235
    dew = __fmul_rn(qkms, __fsub_rn(qvatm, qsg)); // L2237
    eeta = __fmul_rn((-rho), dew); // L2238
    } // L2239
    qfx = __fmul_rn(xlvm, eeta); // L2240
    eeta = __fmul_rn((-rho), dew); // L2241
    sublim = eeta; // L2242
    } else { // L2243
    if (myj) { // L2245
    eeta = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-qkms), ras), beta), __fsub_rn(__fdiv_rn(qvatm, __fadd_rn(1.f, qvatm)), __fdiv_rn(qvg, __fadd_rn(1.f, qvg)))), 1.e3f); // L2247
    } else { // L2248
    eeta = __fmul_rn(__fmul_rn(q1, beta), 1.e3f); // L2251
    } // L2252
    qfx = __fmul_rn(xlvm, eeta); // L2253
    eeta = __fmul_rn(__fmul_rn(q1, beta), 1.e3f); // L2254
    sublim = eeta; // L2255
    } // L2256
    icemelt = 0.f; // L2258
    if ((snhei >= snth)) { // L2259
    s = __fdiv_rn(__fmul_rn(__fmul_rn(thdifsn, rhocsn), __fsub_rn(soilt, tsob)), snprim); // L2260
    snflx = s; // L2261
    } else if (((snhei < snth) && (snhei > 0.f))) { // L2262
    s = __fdiv_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fmul_rn(fsn, thdifsn), rhocsn), __fmul_rn(__fmul_rn(fso, thdifice(1)), rhcs)), __fsub_rn(soilt, tsob)), snprim); // L2263
    snflx = s; // L2265
    if (false) { // L2266
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2267
    } // L2268
    } else { // L2269
    snflx = __fmul_rn(d9sn, __fsub_rn(soilt, tsob)); // L2270
    if (false) { // L2271
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2272
    } // L2273
    } // L2274
    snhei = __fdiv_rn(__fmul_rn(snwe, 1.e3f), rhosn); // L2276
    if (false) { // L2278
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2279
    } // L2280
    x = __fadd_rn(__fmul_rn(__fadd_rn(r21, __fmul_rn(d9sn, r22sn)), __fsub_rn(soilt, tnold)), __fmul_rn(__fmul_rn(xlvm, r210), __fsub_rn(qsg, qgold))); // L2282
    if ((globalcellid == tb.targetcell)) { // L2285
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2286
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2287
    } // L2289
    x = __fsub_rn(__fsub_rn(x, __fmul_rn(__fdiv_rn(__fmul_rn(rhonewcsn, newsnow), delt), __fsub_rn(ruc_min<float>(273.15f, tabs), soilt))), __fmul_rn(__fmul_rn(__fmul_rn(rainf, cvw), prcpms), __fsub_rn(ruc_max<float>(273.15f, tabs), soilt))); // L2290
    icemelt = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(rnet, hft), __fmul_rn(xlvm, eeta)), s), snoh), x); // L2295
    if (false) { // L2296
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2297
    } // L2298
    fltot = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(rnet, hft), __fmul_rn(xlvm, eeta)), s), snoh), x), icemelt); // L2300
    if ((globalcellid == tb.targetcell)) { // L2302
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2303
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2305
    } // L2307
    if ((snhei == 0.f)) { // L2309
    tsnav = __fsub_rn(soilt, 273.15f); // L2310
    emiss = 0.98f; // L2311
    znt = 0.011f; // L2312
    alb = 0.55f; // L2313
    } // L2314
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_ice.F90:2320-2358
RUCDEV void ice_vilka(Tables &tb, Arena &arena, float  tn, float  d1, float  d2, float  pp, float & qs, float & ts, FArray<float> tt, int  nstep, int  ii, int  j, int  iland, int  isoil) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float f1;
    float t1;
    float t2;
    float rn;
    int i;
    int i1;
    i = __fadd_rn(__fdiv_rn(__fsub_rn(tn, 1.7315e2f), .05f), 1); // L2334
    t1 = __fadd_rn(173.1f, __fmul_rn(float(i), .05f)); // L2335
    f1 = __fsub_rn(__fadd_rn(t1, __fmul_rn(d1, tt(i))), d2); // L2336
    i1 = __fsub_rn(i, __fdiv_rn(f1, __fadd_rn(.05f, __fmul_rn(d1, __fsub_rn(tt((i + 1)), tt(i)))))); // L2337
    i = i1; // L2338
    if (((i > 5000) || (i < 1))) { goto label_1; } // L2339
    label_10: ;
    i1 = i; // L2340
    t1 = __fadd_rn(173.1f, __fmul_rn(float(i), .05f)); // L2341
    f1 = __fsub_rn(__fadd_rn(t1, __fmul_rn(d1, tt(i))), d2); // L2342
    rn = __fdiv_rn(f1, __fadd_rn(.05f, __fmul_rn(d1, __fsub_rn(tt((i + 1)), tt(i))))); // L2343
    i = (i - int(rn)); // L2344
    if (((i > 5000) || (i < 1))) { goto label_1; } // L2345
    if ((i1 != i)) { goto label_10; } // L2346
    ts = __fsub_rn(t1, __fmul_rn(.05f, rn)); // L2347
    qs = __fdiv_rn(__fadd_rn(tt(i), __fmul_rn(__fsub_rn(tt(i), tt((i + 1))), rn)), pp); // L2348
    goto label_20; // L2349
    label_1: ;
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L2352
    atomicOr(arena.error, 1); return; // L2355
    label_20: ;
    ; // L2356
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_ice.F90:2361-2381
RUCDEV float ice_qsn(Tables &tb, Arena &arena, float  tn, FArray<float> t) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float qsn;
    float r;
    float r1;
    float r2;
    int i;
    r = __fadd_rn(__fdiv_rn(__fsub_rn(tn, 173.15f), .05f), 1.f); // L2369
    i = int(r); // L2370
    if ((i >= 1)) { goto label_10; } // L2371
    i = 1; // L2372
    r = 1.f; // L2373
    label_10: ;
    if ((i <= 5000)) { goto label_20; } // L2374
    i = 5000; // L2375
    r = 5001.f; // L2376
    label_20: ;
    r1 = t(i); // L2377
    r2 = __fsub_rn(r, i); // L2378
    qsn = __fadd_rn(__fmul_rn(__fsub_rn(t((i + 1)), r1), r2), r1); // L2379
    return qsn;
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_sf_sfcdiags_ruclsm.F:9-187
RUCDEV void diags_sfcdiags_ruclsm(Tables &tb, Arena &arena, FArray<float> hfx, FArray<float> qfx, FArray<float> tsk, FArray<float> qsfc, FArray<float> cqs, FArray<float> cqs2, FArray<float> chs, FArray<float> chs2, FArray<float> t2, FArray<float> th2, FArray<float> q2, FArray<float> t3d, FArray<float> qv3d, FArray<float> dz, FArray<float> rho3d, FArray<float> p3d, FArray<float> psfc2d, float  cp, float  r_d, float  rovcp, FArray<int> globalcells, int  ids, int  ide, int  jds, int  jde, int  kds, int  kde, int  ims, int  ime, int  jms, int  jme, int  kms, int  kme, int  its, int  ite, int  jts, int  jte, int  kts, int  kte) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int i;
    int j;
    float rho;
    float x2m;
    float qlev1;
    float tempc;
    float qsat;
    float p2m;
    float qsfcprox;
    float qsfcmr;
    float psfc;
    float dt;
    float dq;
    float fh;
    float fac;
    float dz1;
    FArray<float> t2_alt = arena.alloc<float>(ims, ime, jms, jme);
    FArray<float> q2_alt = arena.alloc<float>(ims, ime, jms, jme);
    int flux;
    flux = true; // L54
    for (j = jts; j <= jte; j += 1) { loop1_cycle: ; // L57
    for (i = its; i <= ite; i += 1) { loop2_cycle: ; // L58
    rho = rho3d(i, 1, j); // L59
    psfc = psfc2d(i, j); // L60
    if (flux) { // L63
    if ((chs2(i, j) < 1.e-5f)) { // L64
    th2(i, j) = __fmul_rn(t3d(i, 1, j), gfk_pow(__fdiv_rn(1.e5f, psfc), rovcp)); // L66
    } else { // L67
    th2(i, j) = __fsub_rn(__fmul_rn(tsk(i, j), gfk_pow(__fdiv_rn(1.e5f, psfc), rovcp)), __fdiv_rn(hfx(i, j), __fmul_rn(__fmul_rn(rho, cp), chs2(i, j)))); // L68
    } // L69
    t2(i, j) = __fmul_rn(th2(i, j), gfk_pow(__fmul_rn(1.e-5f, psfc), rovcp)); // L70
    } else { // L71
    t2(i, j) = __fsub_rn(tsk(i, j), __fmul_rn(__fdiv_rn(chs(i, j), chs2(i, j)), __fsub_rn(tsk(i, j), t3d(i, 1, j)))); // L72
    } // L73
    th2(i, j) = __fmul_rn(t2(i, j), gfk_pow(__fdiv_rn(1.e5f, psfc), rovcp)); // L74
    qlev1 = qv3d(i, 1, j); // L77
    tempc = __fsub_rn(t3d(i, 1, j), 273.15f); // L79
    if ((tempc <= 0.0f)) { // L80
    qsat = diags_rsif(tb, arena, p3d(i, 1, j), t3d(i, 1, j)); // L82
    } else { // L83
    qsat = diags_rslf(tb, arena, p3d(i, 1, j), t3d(i, 1, j)); // L84
    } // L85
    qlev1 = ruc_min<float>(qsat, qlev1); // L87
    qsfcprox = __fadd_rn(qlev1, __fdiv_rn(qfx(i, j), __fmul_rn(rho, cqs(i, j)))); // L91
    qsfcmr = __fdiv_rn(qsfc(i, j), __fsub_rn(1.f, qsfc(i, j))); // L92
    if ((globalcells(i) == tb.targetcell)) { // L94
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L95
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L96
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L97
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L98
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L99
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L100
    } // L101
    if (flux) { // L103
    if ((cqs2(i, j) < 1.e-5f)) { // L104
    q2(i, j) = qlev1; // L106
    } else { // L107
    x2m = __fsub_rn(qsfcprox, __fdiv_rn(qfx(i, j), __fmul_rn(rho, cqs2(i, j)))); // L108
    q2(i, j) = x2m; // L109
    } // L110
    } else { // L111
    q2(i, j) = __fsub_rn(qsfcmr, __fmul_rn(__fdiv_rn(cqs(i, j), cqs2(i, j)), __fsub_rn(qsfcmr, qlev1))); // L113
    } // L114
    if ((globalcells(i) == tb.targetcell)) { // L116
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L117
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L118
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L119
    } // L120
    dt = __fsub_rn(t3d(i, 1, j), tsk(i, j)); // L123
    dq = __fsub_rn(qlev1, qsfcmr); // L124
    dz1 = __fmul_rn(0.5f, dz(i, 1, j)); // L125
    if ((dt > 0.f)) { // L126
    fh = ruc_min<float>(ruc_max<float>(__fsub_rn(1.f, __fdiv_rn(dt, 10.f)), 0.01f), 1.0f); // L128
    fac = __fdiv_rn(gfk_log(__fdiv_rn(__fadd_rn(2.f, .05f), __fadd_rn(0.05f, fh))), gfk_log(__fdiv_rn(__fadd_rn(dz1, .05f), __fadd_rn(0.05f, fh)))); // L130
    t2_alt(i, j) = __fadd_rn(tsk(i, j), __fmul_rn(fac, __fsub_rn(t3d(i, 1, j), tsk(i, j)))); // L132
    } else { // L133
    t2_alt(i, j) = t2(i, j); // L135
    } // L136
    if ((dq > 0.f)) { // L137
    fh = ruc_min<float>(ruc_max<float>(__fsub_rn(1.f, __fdiv_rn(dq, 0.003f)), 0.01f), 1.0f); // L138
    fac = __fdiv_rn(gfk_log(__fdiv_rn(__fadd_rn(2.f, .05f), __fadd_rn(0.05f, fh))), gfk_log(__fdiv_rn(__fadd_rn(dz1, .05f), __fadd_rn(0.05f, fh)))); // L140
    q2_alt(i, j) = __fadd_rn(qsfcmr, __fmul_rn(fac, __fsub_rn(qlev1, qsfcmr))); // L142
    } else { // L143
    q2_alt(i, j) = q2(i, j); // L145
    } // L146
    if ((globalcells(i) == tb.targetcell)) { // L147
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L148
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L149
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L150
    } // L151
    t2(i, j) = t2_alt(i, j); // L154
    q2(i, j) = q2_alt(i, j); // L155
    x2m = ruc_max<float>(ruc_min<float>(tsk(i, j), t3d(i, 1, j)), t2(i, j)); // L158
    t2(i, j) = ruc_min<float>(ruc_max<float>(tsk(i, j), t3d(i, 1, j)), x2m); // L159
    th2(i, j) = __fmul_rn(t2(i, j), gfk_pow(__fdiv_rn(1.e5f, psfc), rovcp)); // L160
    x2m = ruc_max<float>(ruc_min<float>(qsfcmr, qlev1), q2(i, j)); // L163
    q2(i, j) = ruc_min<float>(ruc_max<float>(qsfcmr, qlev1), x2m); // L164
    tempc = __fsub_rn(t2(i, j), 273.15f); // L166
    if ((tempc <= 0.0f)) { // L167
    qsat = diags_rsif(tb, arena, psfc, t2(i, j)); // L169
    } else { // L170
    qsat = diags_rslf(tb, arena, psfc, t2(i, j)); // L172
    } // L173
    q2(i, j) = ruc_min<float>(qsat, q2(i, j)); // L174
    if ((globalcells(i) == tb.targetcell)) { // L175
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L176
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L177
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L179
    } // L181
    loop2_continue: ;
    } loop2_exit: ; // L184
    loop1_continue: ;
    } loop1_exit: ; // L185
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_sf_sfcdiags_ruclsm.F:190-210
RUCDEV float diags_rslf(Tables &tb, Arena &arena, float  p, float  t) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float esl;
    float x;
    float c0 = .611583699e03f;
    float c1 = .444606896e02f;
    float c2 = .143177157e01f;
    float c3 = .264224321e-1f;
    float c4 = .299291081e-3f;
    float c5 = .203154182e-5f;
    float c6 = .702620698e-8f;
    float c7 = .379534310e-11f;
    float c8 = (-.321582393e-13f);
    float rslf;
    x = ruc_max<float>((-80.f), __fsub_rn(t, 273.16f)); // L205
    esl = __fadd_rn(c0, __fmul_rn(x, __fadd_rn(c1, __fmul_rn(x, __fadd_rn(c2, __fmul_rn(x, __fadd_rn(c3, __fmul_rn(x, __fadd_rn(c4, __fmul_rn(x, __fadd_rn(c5, __fmul_rn(x, __fadd_rn(c6, __fmul_rn(x, __fadd_rn(c7, __fmul_rn(x, c8)))))))))))))))); // L207
    rslf = __fdiv_rn(__fmul_rn(.622f, esl), __fsub_rn(p, esl)); // L208
    return rslf;
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_sf_sfcdiags_ruclsm.F:224-243
RUCDEV float diags_rsif(Tables &tb, Arena &arena, float  p, float  t) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    float esi;
    float x;
    float c0 = .609868993e03f;
    float c1 = .499320233e02f;
    float c2 = .184672631e01f;
    float c3 = .402737184e-1f;
    float c4 = .565392987e-3f;
    float c5 = .521693933e-5f;
    float c6 = .307839583e-7f;
    float c7 = .105785160e-9f;
    float c8 = .161444444e-12f;
    float rsif;
    x = ruc_max<float>((-80.f), __fsub_rn(t, 273.16f)); // L239
    esi = __fadd_rn(c0, __fmul_rn(x, __fadd_rn(c1, __fmul_rn(x, __fadd_rn(c2, __fmul_rn(x, __fadd_rn(c3, __fmul_rn(x, __fadd_rn(c4, __fmul_rn(x, __fadd_rn(c5, __fmul_rn(x, __fadd_rn(c6, __fmul_rn(x, __fadd_rn(c7, __fmul_rn(x, c8)))))))))))))))); // L240
    rsif = __fdiv_rn(__fmul_rn(.622f, esi), __fsub_rn(p, esi)); // L241
    return rsif;
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_lsm_rucinit.F:72-245
RUCDEV void init_rucinit(Tables &tb, Arena &arena, int  restart, int  cycling, int  ncells, int  nsoillevels, FArray<int> ivgtyp, FArray<int> isltyp, FArray<float> tslb, FArray<float> smois, FArray<float> xice, FArray<float> smfr3d, FArray<float> sh2o, FArray<float> mavail, FArray<float> znt, FArray<float> snow, FArray<float> snowh) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int icell;
    int l;
    int errflag;
    float dqm;
    float ref;
    float psis;
    float qmin;
    float bclh;
    float tln;
    FArray<float> soiliqw = arena.alloc<float>(1, nsoillevels);
    int fndsnowh = true;
    float xlmelt = 3.35e5f;
    float riw = 900.e-3f;
    if (((!restart) && (!cycling))) { // L176
    errflag = 0; // L178
    for (icell = 1; icell <= ncells; icell += 1) { loop1_cycle: ; // L179
    if ((isltyp(icell) < 1)) { // L180
    errflag = 1; // L181
    ; /* Native diagnostic I/O omitted; no physics mutation. */ // L182
    ; /* Native diagnostic I/O; no numerical state. */ // L184
    } // L185
    loop1_continue: ;
    } loop1_exit: ; // L186
    if ((errflag == 1)) { atomicOr(arena.error, 1); return; } // L187
    for (icell = 1; icell <= ncells; icell += 1) { loop2_cycle: ; // L190
    znt(icell) = tb.z0tbl(ivgtyp(icell)); // L191
    dqm = __fsub_rn(tb.maxsmc(isltyp(icell)), tb.drysmc(isltyp(icell))); // L195
    ref = tb.refsmc(isltyp(icell)); // L197
    psis = (-tb.satpsi(isltyp(icell))); // L198
    qmin = tb.drysmc(isltyp(icell)); // L199
    bclh = tb.bb(isltyp(icell)); // L200
    if ((xice(icell) > 0.f)) { // L202
    for (l = 1; l <= nsoillevels; l += 1) { loop3_cycle: ; // L203
    smfr3d(l, icell) = 1.f; // L204
    sh2o(l, icell) = 0.f; // L205
    mavail(icell) = 1.f; // L206
    loop3_continue: ;
    } loop3_exit: ; // L207
    } else { // L208
    if ((isltyp(icell) != 14)) { // L209
    mavail(icell) = ruc_max<float>(0.00001f, ruc_min<float>(1.f, __fdiv_rn(__fsub_rn(smois(1, icell), qmin), __fsub_rn(ref, qmin)))); // L211
    for (l = 1; l <= nsoillevels; l += 1) { loop4_cycle: ; // L212
    tln = gfk_log(__fdiv_rn(tslb(l, icell), 273.15f)); // L213
    if ((tln < 0.f)) { // L214
    soiliqw(l) = __fmul_rn(__fadd_rn(dqm, qmin), gfk_pow(__fdiv_rn(__fdiv_rn(__fdiv_rn(__fmul_rn(xlmelt, __fsub_rn(tslb(l, icell), 273.15f)), tslb(l, icell)), 9.81f), psis), __fdiv_rn((-1.f), bclh))); // L215
    soiliqw(l) = ruc_max<float>(0.f, soiliqw(l)); // L217
    soiliqw(l) = ruc_min<float>(soiliqw(l), smois(l, icell)); // L218
    sh2o(l, icell) = soiliqw(l); // L219
    smfr3d(l, icell) = __fdiv_rn(__fsub_rn(smois(l, icell), soiliqw(l)), riw); // L220
    } else { // L222
    smfr3d(l, icell) = 0.f; // L223
    sh2o(l, icell) = smois(l, icell); // L224
    } // L225
    loop4_continue: ;
    } loop4_exit: ; // L226
    } else { // L227
    for (l = 1; l <= nsoillevels; l += 1) { loop5_cycle: ; // L228
    smfr3d(l, icell) = 0.f; // L229
    sh2o(l, icell) = 1.f; // L230
    mavail(icell) = 1.f; // L231
    loop5_continue: ;
    } loop5_exit: ; // L232
    } // L233
    } // L234
    if ((!fndsnowh)) { // L237
    ; /* Native diagnostic I/O; no numerical state. */ // L239
    snowh(icell) = __fmul_rn(snow(icell), 0.005f); // L240
    } // L241
    loop2_continue: ;
    } loop2_exit: ; // L242
    } // L243
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:436-442
RUCDEV void drivers_seaice_prepare(Tables &tb, Arena &arena, int  ncells, float  xice_threshold, FArray<float> xice, FArray<float> sfc_albedo, FArray<float> sfc_emiss, FArray<float> sfc_albedo_p, FArray<float> sfc_emiss_p) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int i;
    int j = 1;
    for (i = 1; i <= ncells; i += 1) { loop1_cycle: ; // L10
    if (((xice(i) >= xice_threshold) && (xice(i) < 1.f))) { // L436
    sfc_albedo_p(i, j) = __fdiv_rn(__fsub_rn(sfc_albedo(i), __fmul_rn(0.08f, __fsub_rn(1.f, xice(i)))), xice(i)); // L437
    sfc_emiss_p(i, j) = __fdiv_rn(__fsub_rn(sfc_emiss(i), __fmul_rn(0.98f, __fsub_rn(1.f, xice(i)))), xice(i)); // L438
    } else { // L439
    sfc_emiss_p(i, j) = sfc_emiss(i); // L440
    sfc_albedo_p(i, j) = sfc_albedo(i); // L441
    } // L442
    loop1_continue: ;
    } loop1_exit: ; // L443
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:571-601
RUCDEV void drivers_seaice_finish(Tables &tb, Arena &arena, int  ncells, float  xice_threshold, int  config_frac_seaice, int  mynn, FArray<float> xice_p, FArray<float> flhc_p, FArray<float> flhc_sea, FArray<float> flqc_p, FArray<float> flqc_sea, FArray<float> cqs_p, FArray<float> cqs_sea, FArray<float> cpm_p, FArray<float> cpm_sea, FArray<float> chs_p, FArray<float> chs_sea, FArray<float> chs2_p, FArray<float> chs2_sea, FArray<float> cqs2_p, FArray<float> cqs2_sea, FArray<float> hfx_p, FArray<float> hfx_sea, FArray<float> lh_p, FArray<float> lh_sea, FArray<float> qfx_p, FArray<float> qfx_sea, FArray<float> qgh_p, FArray<float> qgh_sea, FArray<float> qsfc_p, FArray<float> qsfc_sea, FArray<float> tsk_p, FArray<float> tsk_sea, FArray<float> sfc_albedo_p, FArray<float> sfc_emiss_p) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int i;
    int j = 1;
    for (i = 1; i <= ncells; i += 1) { loop1_cycle: ; // L453
    if (config_frac_seaice) { // L454
    if (((xice_p(i, j) >= xice_threshold) && (xice_p(i, j) < 1.f))) { // L575
    if (true) { // L576
    flhc_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), flhc_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), flhc_sea(i, j))); // L577
    flqc_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), flqc_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), flqc_sea(i, j))); // L578
    if (mynn) { // L579
    cqs_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), cqs_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), cqs_sea(i, j))); // L580
    } else { // L581
    cqs_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), chs_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), chs_sea(i, j))); // L582
    } // L583
    } // L584
    cpm_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), cpm_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), cpm_sea(i, j))); // L586
    chs_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), chs_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), chs_sea(i, j))); // L587
    chs2_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), chs2_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), chs2_sea(i, j))); // L588
    cqs2_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), cqs2_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), cqs2_sea(i, j))); // L589
    hfx_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), hfx_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), hfx_sea(i, j))); // L590
    lh_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), lh_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), lh_sea(i, j))); // L591
    qfx_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), qfx_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), qfx_sea(i, j))); // L592
    qgh_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), qgh_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), qgh_sea(i, j))); // L593
    qsfc_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), qsfc_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), qsfc_sea(i, j))); // L594
    tsk_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), tsk_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), tsk_sea(i, j))); // L595
    sfc_albedo_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), sfc_albedo_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), 0.08f)); // L596
    sfc_emiss_p(i, j) = __fadd_rn(__fmul_rn(xice_p(i, j), sfc_emiss_p(i, j)), __fmul_rn(__fsub_rn(1.f, xice_p(i, j)), 0.98f)); // L597
    } // L598
    } // L599
    loop1_continue: ;
    } loop1_exit: ; // L600
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_lsm_shared.F:22-57
RUCDEV void drivers_correct_tsk_over_seaice(Tables &tb, Arena &arena, int  ims, int  ime, int  jms, int  jme, int  its, int  ite, int  jts, int  jte, float  xice_thresh, FArray<float> xice, FArray<float> tsk, FArray<float> tsk_sea, FArray<float> tsk_ice) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int i;
    int j;
    for (j = jts; j <= jte; j += 1) { loop1_cycle: ; // L40
    for (i = its; i <= ite; i += 1) { loop2_cycle: ; // L41
    tsk_sea(i, j) = tsk(i, j); // L42
    tsk_ice(i, j) = tsk(i, j); // L43
    if (((xice(i, j) >= xice_thresh) && (xice(i, j) <= 1.f))) { // L45
    tsk_sea(i, j) = ruc_max<float>(tsk_sea(i, j), 271.4f); // L47
    if (((xice(i, j) < 0.2f) && (tsk_ice(i, j) < 253.15f))) { tsk_ice(i, j) = 253.15f; } // L51
    if (((xice(i, j) < 0.1f) && (tsk_ice(i, j) < 263.15f))) { tsk_ice(i, j) = 263.15f; } // L52
    } // L53
    loop2_continue: ;
    } loop2_exit: ; // L54
    loop1_continue: ;
    } loop1_exit: ; // L55
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:606-640
RUCDEV void drivers_driver_forcing(Tables &tb, Arena &arena, int  ncells, int  micro_on, int  conv_on, FArray<float> sr, FArray<float> rainncv, FArray<float> raincv, FArray<float> gsw, FArray<float> sfc_albedo, FArray<float> sr_p, FArray<float> rainbl_p, FArray<float> swdown_p) {
    const int arena_mark = arena.used;
    ArenaScope arena_scope(arena, arena_mark);
    int i;
    int j = 1;
    for (i = 1; i <= ncells; i += 1) { loop1_cycle: ; // L65
    sr_p(i, j) = 0.f; // L610
    rainbl_p(i, j) = 0.f; // L611
    if (micro_on) { // L612
    sr_p(i, j) = sr(i); // L621
    rainbl_p(i, j) = __fadd_rn(rainbl_p(i, j), rainncv(i)); // L622
    } // L623
    if (conv_on) { // L624
    rainbl_p(i, j) = __fadd_rn(rainbl_p(i, j), raincv(i)); // L631
    } // L632
    swdown_p(i, j) = __fdiv_rn(gsw(i), __fsub_rn(1.f, sfc_albedo(i))); // L638
    loop1_continue: ;
    } loop1_exit: ; // L639
}
// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:498-582
extern "C" __global__ void ruc_land_tile_preflight(unsigned long long *ptrs,int *stop,int nc) {
 if(blockIdx.x!=0 || threadIdx.x!=0)return;
 stop[0]=nc; stop[1]=0;
 if(((int*)ptrs[8])[0]!=1)return;
 for(int c=0;c<nc;++c)if(((float*)ptrs[9])[c]==1.f){stop[0]=c;stop[1]=1;return;}
}
extern "C" __global__ void oracle_land_ruc_land(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
 Tables tb(tables); Arena arena{work+c,0,nc,error};
 int *tile_stop=(int*)ptrs[119]; if(c>=tile_stop[0])return; arena.preparation_only=tile_stop[1]!=0;
 if(ptrs[120]) arena.entry_ilnb=((int*)ptrs[120])+c;
 if(ptrs[121]) arena.snowtemp_calls=((int*)ptrs[121])+c;
 int &spp_lsm = ((int*)ptrs[0])[c];
 float &dt = ((float*)ptrs[3])[c];
 int &ktau = ((int*)ptrs[4])[c];
 int &nsl = ((int*)ptrs[5])[c];
 int &restart = ((int*)ptrs[6])[c];
 int &cycling = ((int*)ptrs[7])[c];
 int &lakemodel = ((int*)ptrs[8])[c];
 int &frpcpn = ((int*)ptrs[19])[c];
 int &mminlu = ((int*)ptrs[44])[c];
 int &nlcat = ((int*)ptrs[46])[c];
 int &mosaic_lu = ((int*)ptrs[47])[c];
 int &mosaic_soil = ((int*)ptrs[48])[c];
 int &nscat = ((int*)ptrs[50])[c];
 int &iswater = ((int*)ptrs[62])[c];
 int &isice = ((int*)ptrs[63])[c];
 float &xice_threshold = ((float*)ptrs[65])[c];
 float &cp = ((float*)ptrs[66])[c];
 float &rovcp = ((float*)ptrs[67])[c];
 float &g0 = ((float*)ptrs[68])[c];
 float &lv = ((float*)ptrs[69])[c];
 float &stbolt = ((float*)ptrs[70])[c];
 int &myjpbl = ((int*)ptrs[91])[c];
 int &rdlai2d = ((int*)ptrs[94])[c];
 int &ids = ((int*)ptrs[96])[c];
 int &ide = ((int*)ptrs[97])[c];
 int &jds = ((int*)ptrs[98])[c];
 int &jde = ((int*)ptrs[99])[c];
 int &kds = ((int*)ptrs[100])[c];
 int &kde = ((int*)ptrs[101])[c];
 int &ims = ((int*)ptrs[102])[c];
 int &ime = ((int*)ptrs[103])[c];
 int &jms = ((int*)ptrs[104])[c];
 int &jme = ((int*)ptrs[105])[c];
 int &kms = ((int*)ptrs[106])[c];
 int &kme = ((int*)ptrs[107])[c];
 int &its = ((int*)ptrs[108])[c];
 int &ite = ((int*)ptrs[109])[c];
 int &jts = ((int*)ptrs[110])[c];
 int &jte = ((int*)ptrs[111])[c];
 int &kts = ((int*)ptrs[112])[c];
 int &kte = ((int*)ptrs[113])[c];
 FArray<float> pattern_spp_lsm{((float*)ptrs[1])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> field_sf{((float*)ptrs[2])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> lakemask{((float*)ptrs[9])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> graupelncv{((float*)ptrs[10])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snowncv{((float*)ptrs[11])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> rainncv{((float*)ptrs[12])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> zs{((float*)ptrs[13])+c,1,1,1,nc,nc*(nsl-1+1),nc*(nsl-1+1)*1,(nsl-1+1),1,1};
 FArray<float> rainbl{((float*)ptrs[14])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snow{((float*)ptrs[15])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snowh{((float*)ptrs[16])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snowc{((float*)ptrs[17])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> frzfrac{((float*)ptrs[18])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> rhosnf{((float*)ptrs[20])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> precipfr{((float*)ptrs[21])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> z3d{((float*)ptrs[22])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> p8w{((float*)ptrs[23])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> t3d{((float*)ptrs[24])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> qv3d{((float*)ptrs[25])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> qc3d{((float*)ptrs[26])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> rho3d{((float*)ptrs[27])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> glw{((float*)ptrs[28])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> gsw{((float*)ptrs[29])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> emiss{((float*)ptrs[30])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> chklowq{((float*)ptrs[31])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> chs{((float*)ptrs[32])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> flqc{((float*)ptrs[33])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> flhc{((float*)ptrs[34])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> mavail{((float*)ptrs[35])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> canwat{((float*)ptrs[36])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> vegfra{((float*)ptrs[37])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> alb{((float*)ptrs[38])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> znt{((float*)ptrs[39])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> z0{((float*)ptrs[40])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snoalb{((float*)ptrs[41])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> albbck{((float*)ptrs[42])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> lai{((float*)ptrs[43])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> landusef{((float*)ptrs[45])+c,ims,1,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(nlcat-1+1),(ime-ims+1),(nlcat-1+1),(jme-jms+1)};
 FArray<float> soilctop{((float*)ptrs[49])+c,ims,1,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(nscat-1+1),(ime-ims+1),(nscat-1+1),(jme-jms+1)};
 FArray<float> qsfc{((float*)ptrs[51])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qsg{((float*)ptrs[52])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qvg{((float*)ptrs[53])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qcg{((float*)ptrs[54])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> dew{((float*)ptrs[55])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> soilt1{((float*)ptrs[56])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tsnav{((float*)ptrs[57])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tbot{((float*)ptrs[58])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<int> ivgtyp{((int*)ptrs[59])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<int> isltyp{((int*)ptrs[60])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> xland{((float*)ptrs[61])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> xice{((float*)ptrs[64])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> soilmois{((float*)ptrs[71])+c,ims,1,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(nsl-1+1),(ime-ims+1),(nsl-1+1),(jme-jms+1)};
 FArray<float> sh2o{((float*)ptrs[72])+c,ims,1,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(nsl-1+1),(ime-ims+1),(nsl-1+1),(jme-jms+1)};
 FArray<float> smavail{((float*)ptrs[73])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> smmax{((float*)ptrs[74])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tso{((float*)ptrs[75])+c,ims,1,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(nsl-1+1),(ime-ims+1),(nsl-1+1),(jme-jms+1)};
 FArray<float> soilt{((float*)ptrs[76])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> hfx{((float*)ptrs[77])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qfx{((float*)ptrs[78])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> lh{((float*)ptrs[79])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> sfcrunoff{((float*)ptrs[80])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> udrunoff{((float*)ptrs[81])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> acrunoff{((float*)ptrs[82])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> sfcexc{((float*)ptrs[83])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> sfcevp{((float*)ptrs[84])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> grdflx{((float*)ptrs[85])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snowfallac{((float*)ptrs[86])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> acsnow{((float*)ptrs[87])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snom{((float*)ptrs[88])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> smfr3d{((float*)ptrs[89])+c,ims,1,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(nsl-1+1),(ime-ims+1),(nsl-1+1),(jme-jms+1)};
 FArray<float> keepfr3dflag{((float*)ptrs[90])+c,ims,1,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(nsl-1+1),(ime-ims+1),(nsl-1+1),(jme-jms+1)};
 FArray<float> shdmin{((float*)ptrs[92])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> shdmax{((float*)ptrs[93])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<int> globalcells{((int*)ptrs[95])+c,ims,1,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*1,(ime-ims+1),1,1};
 land_ruc_land(tb,arena,spp_lsm,pattern_spp_lsm,field_sf,dt,ktau,nsl,restart,cycling,lakemodel,lakemask,graupelncv,snowncv,rainncv,zs,rainbl,snow,snowh,snowc,frzfrac,frpcpn,rhosnf,precipfr,z3d,p8w,t3d,qv3d,qc3d,rho3d,glw,gsw,emiss,chklowq,chs,flqc,flhc,mavail,canwat,vegfra,alb,znt,z0,snoalb,albbck,lai,mminlu,landusef,nlcat,mosaic_lu,mosaic_soil,soilctop,nscat,qsfc,qsg,qvg,qcg,dew,soilt1,tsnav,tbot,ivgtyp,isltyp,xland,iswater,isice,xice,xice_threshold,cp,rovcp,g0,lv,stbolt,soilmois,sh2o,smavail,smmax,tso,soilt,hfx,qfx,lh,sfcrunoff,udrunoff,acrunoff,sfcexc,sfcevp,grdflx,snowfallac,acsnow,snom,smfr3d,keepfr3dflag,myjpbl,shdmin,shdmax,rdlai2d,globalcells,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,(((int*)ptrs[114])[c]!=0),(((int*)ptrs[115])[c]!=0),(((int*)ptrs[116])[c]!=0),(((int*)ptrs[117])[c]!=0),(((int*)ptrs[118])[c]!=0));
}
extern "C" __global__ void oracle_ice_ruc_ice(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
 Tables tb(tables); Arena arena{work+c,0,nc,error};
 if(ptrs[96]) arena.entry_ilnb=((int*)ptrs[96])+c;
 if(ptrs[97]) arena.snowtemp_calls=((int*)ptrs[97])+c;
 int &spp_lsm = ((int*)ptrs[0])[c];
 float &dt = ((float*)ptrs[3])[c];
 int &ktau = ((int*)ptrs[4])[c];
 int &nsl = ((int*)ptrs[5])[c];
 int &restart = ((int*)ptrs[6])[c];
 int &cycling = ((int*)ptrs[7])[c];
 int &frpcpn = ((int*)ptrs[14])[c];
 int &mminlu = ((int*)ptrs[35])[c];
 int &myjpbl = ((int*)ptrs[40])[c];
 int &iswater = ((int*)ptrs[41])[c];
 int &isice = ((int*)ptrs[42])[c];
 float &xice_threshold = ((float*)ptrs[44])[c];
 float &cp = ((float*)ptrs[45])[c];
 float &rovcp = ((float*)ptrs[46])[c];
 float &g0 = ((float*)ptrs[47])[c];
 float &lv = ((float*)ptrs[48])[c];
 float &stbolt = ((float*)ptrs[49])[c];
 int &ids = ((int*)ptrs[73])[c];
 int &ide = ((int*)ptrs[74])[c];
 int &jds = ((int*)ptrs[75])[c];
 int &jde = ((int*)ptrs[76])[c];
 int &kds = ((int*)ptrs[77])[c];
 int &kde = ((int*)ptrs[78])[c];
 int &ims = ((int*)ptrs[79])[c];
 int &ime = ((int*)ptrs[80])[c];
 int &jms = ((int*)ptrs[81])[c];
 int &jme = ((int*)ptrs[82])[c];
 int &kms = ((int*)ptrs[83])[c];
 int &kme = ((int*)ptrs[84])[c];
 int &its = ((int*)ptrs[85])[c];
 int &ite = ((int*)ptrs[86])[c];
 int &jts = ((int*)ptrs[87])[c];
 int &jte = ((int*)ptrs[88])[c];
 int &kts = ((int*)ptrs[89])[c];
 int &kte = ((int*)ptrs[90])[c];
 FArray<float> pattern_spp_lsm{((float*)ptrs[1])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> field_sf{((float*)ptrs[2])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> graupelncv{((float*)ptrs[8])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snowncv{((float*)ptrs[9])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> rainncv{((float*)ptrs[10])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> zs{((float*)ptrs[11])+c,1,1,1,nc,nc*(nsl-1+1),nc*(nsl-1+1)*1,(nsl-1+1),1,1};
 FArray<float> rainbl{((float*)ptrs[12])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> frzfrac{((float*)ptrs[13])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> rhosnf{((float*)ptrs[15])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> precipfr{((float*)ptrs[16])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> z3d{((float*)ptrs[17])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> p8w{((float*)ptrs[18])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> t3d{((float*)ptrs[19])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> qv3d{((float*)ptrs[20])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> qc3d{((float*)ptrs[21])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> rho3d{((float*)ptrs[22])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> glw{((float*)ptrs[23])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> gsw{((float*)ptrs[24])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> emiss{((float*)ptrs[25])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> chklowq{((float*)ptrs[26])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> chs{((float*)ptrs[27])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> flqc{((float*)ptrs[28])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> flhc{((float*)ptrs[29])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> alb{((float*)ptrs[30])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> znt{((float*)ptrs[31])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> z0{((float*)ptrs[32])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snoalb{((float*)ptrs[33])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> albbck{((float*)ptrs[34])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tbot{((float*)ptrs[36])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<int> ivgtyp{((int*)ptrs[37])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<int> isltyp{((int*)ptrs[38])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> xland{((float*)ptrs[39])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> xice{((float*)ptrs[43])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tso{((float*)ptrs[50])+c,ims,1,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(nsl-1+1),(ime-ims+1),(nsl-1+1),(jme-jms+1)};
 FArray<float> soilt{((float*)ptrs[51])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> soilt1{((float*)ptrs[52])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tsnav{((float*)ptrs[53])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snow{((float*)ptrs[54])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snowh{((float*)ptrs[55])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snowc{((float*)ptrs[56])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qsfc{((float*)ptrs[57])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qsg{((float*)ptrs[58])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qvg{((float*)ptrs[59])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qcg{((float*)ptrs[60])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> hfx{((float*)ptrs[61])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qfx{((float*)ptrs[62])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> lh{((float*)ptrs[63])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> dew{((float*)ptrs[64])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> sfcrunoff{((float*)ptrs[65])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> acrunoff{((float*)ptrs[66])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> sfcevp{((float*)ptrs[67])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> grdflx{((float*)ptrs[68])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snowfallac{((float*)ptrs[69])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> acsnow{((float*)ptrs[70])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> snom{((float*)ptrs[71])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<int> globalcells{((int*)ptrs[72])+c,ims,1,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*1,(ime-ims+1),1,1};
 ice_ruc_ice(tb,arena,spp_lsm,pattern_spp_lsm,field_sf,dt,ktau,nsl,restart,cycling,graupelncv,snowncv,rainncv,zs,rainbl,frzfrac,frpcpn,rhosnf,precipfr,z3d,p8w,t3d,qv3d,qc3d,rho3d,glw,gsw,emiss,chklowq,chs,flqc,flhc,alb,znt,z0,snoalb,albbck,mminlu,tbot,ivgtyp,isltyp,xland,myjpbl,iswater,isice,xice,xice_threshold,cp,rovcp,g0,lv,stbolt,tso,soilt,soilt1,tsnav,snow,snowh,snowc,qsfc,qsg,qvg,qcg,hfx,qfx,lh,dew,sfcrunoff,acrunoff,sfcevp,grdflx,snowfallac,acsnow,snom,globalcells,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,(((int*)ptrs[91])[c]!=0),(((int*)ptrs[92])[c]!=0),(((int*)ptrs[93])[c]!=0),(((int*)ptrs[94])[c]!=0),(((int*)ptrs[95])[c]!=0));
}
extern "C" __global__ void oracle_diags_sfcdiags_ruclsm(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
 Tables tb(tables); Arena arena{work+c,0,nc,error};
 float &cp = ((float*)ptrs[17])[c];
 float &r_d = ((float*)ptrs[18])[c];
 float &rovcp = ((float*)ptrs[19])[c];
 int &ids = ((int*)ptrs[21])[c];
 int &ide = ((int*)ptrs[22])[c];
 int &jds = ((int*)ptrs[23])[c];
 int &jde = ((int*)ptrs[24])[c];
 int &kds = ((int*)ptrs[25])[c];
 int &kde = ((int*)ptrs[26])[c];
 int &ims = ((int*)ptrs[27])[c];
 int &ime = ((int*)ptrs[28])[c];
 int &jms = ((int*)ptrs[29])[c];
 int &jme = ((int*)ptrs[30])[c];
 int &kms = ((int*)ptrs[31])[c];
 int &kme = ((int*)ptrs[32])[c];
 int &its = ((int*)ptrs[33])[c];
 int &ite = ((int*)ptrs[34])[c];
 int &jts = ((int*)ptrs[35])[c];
 int &jte = ((int*)ptrs[36])[c];
 int &kts = ((int*)ptrs[37])[c];
 int &kte = ((int*)ptrs[38])[c];
 FArray<float> hfx{((float*)ptrs[0])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qfx{((float*)ptrs[1])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tsk{((float*)ptrs[2])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> qsfc{((float*)ptrs[3])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> cqs{((float*)ptrs[4])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> cqs2{((float*)ptrs[5])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> chs{((float*)ptrs[6])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> chs2{((float*)ptrs[7])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> t2{((float*)ptrs[8])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> th2{((float*)ptrs[9])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> q2{((float*)ptrs[10])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> t3d{((float*)ptrs[11])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> qv3d{((float*)ptrs[12])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> dz{((float*)ptrs[13])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> rho3d{((float*)ptrs[14])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> p3d{((float*)ptrs[15])+c,ims,kms,jms,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(kme-kms+1),(ime-ims+1),(kme-kms+1),(jme-jms+1)};
 FArray<float> psfc2d{((float*)ptrs[16])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<int> globalcells{((int*)ptrs[20])+c,ims,1,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*1,(ime-ims+1),1,1};
 diags_sfcdiags_ruclsm(tb,arena,hfx,qfx,tsk,qsfc,cqs,cqs2,chs,chs2,t2,th2,q2,t3d,qv3d,dz,rho3d,p3d,psfc2d,cp,r_d,rovcp,globalcells,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
}
extern "C" __global__ void oracle_init_rucinit(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
 Tables tb(tables); Arena arena{work+c,0,nc,error};
 int &restart = ((int*)ptrs[0])[c];
 int &cycling = ((int*)ptrs[1])[c];
 int &ncells = ((int*)ptrs[2])[c];
 int &nsoillevels = ((int*)ptrs[3])[c];
 FArray<int> ivgtyp{((int*)ptrs[4])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<int> isltyp{((int*)ptrs[5])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> tslb{((float*)ptrs[6])+c,1,1,1,nc,nc*(nsoillevels-1+1),nc*(nsoillevels-1+1)*(ncells-1+1),(nsoillevels-1+1),(ncells-1+1),1};
 FArray<float> smois{((float*)ptrs[7])+c,1,1,1,nc,nc*(nsoillevels-1+1),nc*(nsoillevels-1+1)*(ncells-1+1),(nsoillevels-1+1),(ncells-1+1),1};
 FArray<float> xice{((float*)ptrs[8])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> smfr3d{((float*)ptrs[9])+c,1,1,1,nc,nc*(nsoillevels-1+1),nc*(nsoillevels-1+1)*(ncells-1+1),(nsoillevels-1+1),(ncells-1+1),1};
 FArray<float> sh2o{((float*)ptrs[10])+c,1,1,1,nc,nc*(nsoillevels-1+1),nc*(nsoillevels-1+1)*(ncells-1+1),(nsoillevels-1+1),(ncells-1+1),1};
 FArray<float> mavail{((float*)ptrs[11])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> znt{((float*)ptrs[12])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> snow{((float*)ptrs[13])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> snowh{((float*)ptrs[14])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 init_rucinit(tb,arena,restart,cycling,ncells,nsoillevels,ivgtyp,isltyp,tslb,smois,xice,smfr3d,sh2o,mavail,znt,snow,snowh);
}
extern "C" __global__ void oracle_drivers_seaice_prepare(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
 Tables tb(tables); Arena arena{work+c,0,nc,error};
 int &ncells = ((int*)ptrs[0])[c];
 float &xice_threshold = ((float*)ptrs[1])[c];
 FArray<float> xice{((float*)ptrs[2])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> sfc_albedo{((float*)ptrs[3])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> sfc_emiss{((float*)ptrs[4])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> sfc_albedo_p{((float*)ptrs[5])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> sfc_emiss_p{((float*)ptrs[6])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 drivers_seaice_prepare(tb,arena,ncells,xice_threshold,xice,sfc_albedo,sfc_emiss,sfc_albedo_p,sfc_emiss_p);
}
extern "C" __global__ void oracle_drivers_seaice_finish(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
 Tables tb(tables); Arena arena{work+c,0,nc,error};
 int &ncells = ((int*)ptrs[0])[c];
 float &xice_threshold = ((float*)ptrs[1])[c];
 int &config_frac_seaice = ((int*)ptrs[2])[c];
 int &mynn = ((int*)ptrs[3])[c];
 FArray<float> xice_p{((float*)ptrs[4])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> flhc_p{((float*)ptrs[5])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> flhc_sea{((float*)ptrs[6])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> flqc_p{((float*)ptrs[7])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> flqc_sea{((float*)ptrs[8])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> cqs_p{((float*)ptrs[9])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> cqs_sea{((float*)ptrs[10])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> cpm_p{((float*)ptrs[11])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> cpm_sea{((float*)ptrs[12])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> chs_p{((float*)ptrs[13])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> chs_sea{((float*)ptrs[14])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> chs2_p{((float*)ptrs[15])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> chs2_sea{((float*)ptrs[16])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> cqs2_p{((float*)ptrs[17])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> cqs2_sea{((float*)ptrs[18])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> hfx_p{((float*)ptrs[19])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> hfx_sea{((float*)ptrs[20])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> lh_p{((float*)ptrs[21])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> lh_sea{((float*)ptrs[22])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> qfx_p{((float*)ptrs[23])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> qfx_sea{((float*)ptrs[24])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> qgh_p{((float*)ptrs[25])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> qgh_sea{((float*)ptrs[26])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> qsfc_p{((float*)ptrs[27])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> qsfc_sea{((float*)ptrs[28])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> tsk_p{((float*)ptrs[29])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> tsk_sea{((float*)ptrs[30])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> sfc_albedo_p{((float*)ptrs[31])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> sfc_emiss_p{((float*)ptrs[32])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 drivers_seaice_finish(tb,arena,ncells,xice_threshold,config_frac_seaice,mynn,xice_p,flhc_p,flhc_sea,flqc_p,flqc_sea,cqs_p,cqs_sea,cpm_p,cpm_sea,chs_p,chs_sea,chs2_p,chs2_sea,cqs2_p,cqs2_sea,hfx_p,hfx_sea,lh_p,lh_sea,qfx_p,qfx_sea,qgh_p,qgh_sea,qsfc_p,qsfc_sea,tsk_p,tsk_sea,sfc_albedo_p,sfc_emiss_p);
}
extern "C" __global__ void oracle_drivers_correct_tsk_over_seaice(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
 Tables tb(tables); Arena arena{work+c,0,nc,error};
 int &ims = ((int*)ptrs[0])[c];
 int &ime = ((int*)ptrs[1])[c];
 int &jms = ((int*)ptrs[2])[c];
 int &jme = ((int*)ptrs[3])[c];
 int &its = ((int*)ptrs[4])[c];
 int &ite = ((int*)ptrs[5])[c];
 int &jts = ((int*)ptrs[6])[c];
 int &jte = ((int*)ptrs[7])[c];
 float &xice_thresh = ((float*)ptrs[8])[c];
 FArray<float> xice{((float*)ptrs[9])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tsk{((float*)ptrs[10])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tsk_sea{((float*)ptrs[11])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 FArray<float> tsk_ice{((float*)ptrs[12])+c,ims,jms,1,nc,nc*(ime-ims+1),nc*(ime-ims+1)*(jme-jms+1),(ime-ims+1),(jme-jms+1),1};
 drivers_correct_tsk_over_seaice(tb,arena,ims,ime,jms,jme,its,ite,jts,jte,xice_thresh,xice,tsk,tsk_sea,tsk_ice);
}
extern "C" __global__ void oracle_drivers_driver_forcing(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;
 Tables tb(tables); Arena arena{work+c,0,nc,error};
 int &ncells = ((int*)ptrs[0])[c];
 int &micro_on = ((int*)ptrs[1])[c];
 int &conv_on = ((int*)ptrs[2])[c];
 FArray<float> sr{((float*)ptrs[3])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> rainncv{((float*)ptrs[4])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> raincv{((float*)ptrs[5])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> gsw{((float*)ptrs[6])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> sfc_albedo{((float*)ptrs[7])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*1,(ncells-1+1),1,1};
 FArray<float> sr_p{((float*)ptrs[8])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> rainbl_p{((float*)ptrs[9])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 FArray<float> swdown_p{((float*)ptrs[10])+c,1,1,1,nc,nc*(ncells-1+1),nc*(ncells-1+1)*(1-1+1),(ncells-1+1),(1-1+1),1};
 drivers_driver_forcing(tb,arena,ncells,micro_on,conv_on,sr,rainncv,raincv,gsw,sfc_albedo,sr_p,rainbl_p,swdown_p);
}
'''

def shared_math_source():
    """Use the engine's shared GNU math; no elementary functions rederived here."""
    root = files('woof.core.kernels')
    urban = root.joinpath('urban_bem.cuh').read_text(encoding='utf-8')
    # Existing glibc 2.43 CORE-MATH routine, including its original notice.
    # The older ruc.cu log10 predates this native library; full RUC replay
    # traced its first differing soil-conductivity word to native F90:5120.
    log10 = urban[urban.index('// log10f: CORE-MATH'):urban.index('__device__ void bem_bem')]
    return ('typedef float real;\n' + '\n'.join(root.joinpath(name).read_text(encoding='utf-8') for name in
        ('glibc_flt32.cuh','flt32_expf_fma.cuh','glibc_trig_flt32.cuh','sfire_libm.cuh')) + '\n' + log10 + r'''
#define ruc_log10f_rn ubm_log10f
extern "C" __global__ void ruc_math_probe(float *x,float *y,float *out,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 out[i]=__fmul_rn(x[i],y[i]);
 out[n+i]=gfk_pow(x[i],y[i]);
 out[2*n+i]=gfk_log(x[i]);
 out[3*n+i]=gfk_exp_fma(x[i]);
 out[4*n+i]=sfire_tanhf(x[i]);
 out[5*n+i]=glibc_cosf(x[i]);
 out[6*n+i]=ruc_log10f_rn(x[i]);
}
''')


def source_digest():
    return hashlib.sha256((shared_math_source() + CUDA_SOURCE).encode()).hexdigest()


@lru_cache(maxsize=None)
def _module(device, ice_entry_context=False):
    import cupy as cp
    from cupy.cuda import compiler
    with cp.cuda.Device(device):
        image = os.environ.get('MPAS_RUC_ICE_DEVICE_IMAGE' if ice_entry_context else 'MPAS_RUC_DEVICE_IMAGE')
        if image:
            path = Path(image)
            metadata = json.loads(path.with_suffix('.json').read_text())
            binary = path.read_bytes()
            if metadata['source_sha256'] != source_digest() or metadata['image_sha256'] != hashlib.sha256(binary).hexdigest():
                raise ValueError('RUC device image differs from current source or recorded image hash')
            if bool(metadata.get('ice_entry_context',False)) != ice_entry_context:
                raise ValueError('RUC device image native entry-context mode differs')
            if metadata['arch'] != cp.cuda.Device(device).compute_capability:
                raise ValueError('RUC device image architecture differs from selected CUDA device')
        else:
            binary, _ = compiler.compile_using_nvrtc(shared_math_source() + CUDA_SOURCE,
                options=_hex_numerics.options(CUDA_OPTIONS, keep_ftz=True)+(('-DRUC_ICE_ENTRY_CONTEXT',) if ice_entry_context else ()), arch=cp.cuda.Device(device).compute_capability, cache_in_memory=True)
        module = cp.cuda.function.Module()
        module.load(binary)
        # The shared saturation table every land and sea-ice launch of this module reads (ruc_tbq_init), filled
        # once per loaded module and complete before any stream of this device launches an entry.
        module.get_function('ruc_tbq_init')((1,), (1,), ())
        cp.cuda.runtime.deviceSynchronize()
        return module


@lru_cache(maxsize=None)
def _arena_plan():
    """Each CUDA function's own Arena allocations and its arena callees, or None if the source cannot be bounded.

    The bound is exact for the translation's storage discipline, which this checks rather than assumes:
    every arena.alloc is a 4-byte-word declaration in its function's prologue (before the first statement),
    ArenaScope hands a function's words back when it returns, nothing else writes Arena.used, there is no
    recursion, and every extent a callee allocates with is passed to it under the same name.  Any other
    shape returns None and launches keep the full WORK_WORDS rows.
    """
    import re
    body_re = re.compile(r'^(?:RUCDEV\s+[\w<>&]+|extern "C" __global__ void)\s+(\w+)\(([^)]*)\)\s*\{', re.M)
    alloc_re = re.compile(r'arena\.alloc<(\w+)>\(([^;]*)\);')
    if re.search(r'\.used\s*[-+*]?=(?!=)', CUDA_SOURCE.replace('a.used=mark', '')):
        return None
    funcs = {}
    for m in body_re.finditer(CUDA_SOURCE):
        i, depth = m.end(), 1
        while depth:
            depth += {'{': 1, '}': -1}.get(CUDA_SOURCE[i], 0)
            i += 1
        params = [q.strip().split()[-1].lstrip('&*') for q in m.group(2).split(',') if q.strip()]
        funcs[m.group(1)] = (params, CUDA_SOURCE[m.end():i - 1])
    own, calls, params_of = {}, {}, {}
    for name, (params, body) in funcs.items():
        lines = body.split('\n')
        first = next((k for k, line in enumerate(lines) if re.search(r'// L\d+', line)), len(lines))
        allocs, local = [], []
        for k, line in enumerate(lines):
            # Prologue integer constants (int nvegclas = (24 + 3);) size some declarations.
            d = re.match(r'    int (\w+) = ([\d\s+\-*()]+);', line)
            if d and k < first:
                local.append((d.group(1), d.group(2)))
            for a in alloc_re.finditer(line):
                if a.group(1) not in ('float', 'int') or k > first or not line.startswith('    FArray'):
                    return None
                exprs = tuple(x.strip() for x in a.group(2).split(','))
                if len(exprs) % 2 or any(not re.fullmatch(r'[\w\s+\-*()]+', x) for x in exprs):
                    return None
                allocs.append((exprs, tuple(local)))
        own[name] = tuple(allocs)
        params_of[name] = params[2:] if params[:2] == ['tb', 'arena'] else params
        sites = []
        for c in re.finditer(r'\b(\w+)\(tb,\s*arena,', body):
            j, depth = c.end(), 1
            while depth:
                depth += {'(': 1, ')': -1}.get(body[j], 0)
                j += 1
            args, cur, depth = [], '', 0
            for ch in body[c.end():j - 1]:
                depth += {'(': 1, ')': -1}.get(ch, 0)
                if ch == ',' and depth == 0:
                    args.append(cur.strip()); cur = ''
                else:
                    cur += ch
            args.append(cur.strip())
            sites.append((c.group(1), args))
        calls[name] = sites
    def symbols(fn, stack=()):
        if fn in stack or fn not in own:
            raise ValueError(fn)
        out = {t for exprs, local in own[fn] for x in exprs for t in re.findall(r'[A-Za-z_]\w*', x)}
        out -= {name for exprs, local in own[fn] for name, _ in local}
        for callee, _ in calls[fn]:
            out |= symbols(callee, stack + (fn,))
        return out
    try:
        for fn in own:
            for callee, args in calls[fn]:
                needed = symbols(callee)
                if len(args) != len(params_of[callee]):
                    return None
                if any(q in needed and a != q for q, a in zip(params_of[callee], args)):
                    return None
    except ValueError:
        return None
    return own, {fn: tuple(sorted({c for c, _ in sites})) for fn, sites in calls.items()}


def arena_words(entry, extents):
    """Most Arena words one cell of this entry can address (int), or None when it cannot be bounded.

    ``extents`` maps the soil/category extents (nsl, nlcat, nscat, nsoillevels) to their values; per-cell
    tile extents are 1, as the provider packs one column per thread (COLUMN_EXTENTS).
    """
    plan = _arena_plan()
    kernel = 'oracle_' + entry
    if plan is None or kernel not in plan[0]:
        return None
    own, calls = plan
    env = dict(ARENA_COLUMN_EXTENTS)
    env.update({k: int(v) for k, v in extents.items()})
    nsl = env.get('nsl', env.get('nsoillevels'))
    if nsl is not None:
        for k, v in (('nsl', nsl), ('nzs', nsl), ('nsoillevels', nsl), ('nddzs', 2 * (nsl - 2))):
            env.setdefault(k, v)
    memo = {}
    def need(fn):
        if fn not in memo:
            words = 0
            for exprs, local in own[fn]:
                scope = dict(env)
                for name, value in local:
                    scope[name] = eval(value, {'__builtins__': {}}, {})
                vals = [eval(x, {'__builtins__': {}}, scope) for x in exprs]
                size = 1
                for lo, hi in zip(vals[0::2], vals[1::2]):
                    size *= max(0, hi - lo + 1)
                words += size
            memo[fn] = words + max((need(c) for c in calls[fn]), default=0)
        return memo[fn]
    try:
        return need(kernel)
    except NameError:
        return None


def arena_rows(entry, arguments, scalars=None):
    """(rows a workspace must hold, bound or None): extents from the argument shapes (no device read) and,
    when given, the provider's host scalars, taking the larger of the two for each extent."""
    extents = {}
    for field in ARGUMENTS[entry]:
        real = [d for d in field['dim'] if d not in ARENA_CELL_DIMS]
        if len(real) != 1:
            continue
        lo, _, hi = real[0].rpartition(':')
        a = arguments.get(field['name'])
        if lo not in ('', '1') or not hi.isidentifier() or a is None or getattr(a, 'ndim', 0) != 2:
            continue
        extents[hi] = max(extents.get(hi, 0), int(a.shape[0]))
    for k in ('nsl', 'nlcat', 'nscat', 'nsoillevels'):
        if scalars is not None and k in scalars:
            extents[k] = max(extents.get(k, 0), int(scalars[k]))
    bound = arena_words(entry, extents)
    return (WORK_WORDS if bound is None else max(1, bound)), bound


def _launch_boundary(entry, arguments, tables, *, workspace=None, errors=None, stream=None, present=None, entry_storage=None, trace_entry_calls=False, scalars=None):
    """Launch a complete native RUC entry; cells are fastest in every argument.

    Arrays have shape (Fortran dim3, dim2, dim1, cell), omitting singleton
    dimensions. Scalar arguments are arrays (cell,). Tables are the 955 exact
    words defined by TABLE_LAYOUT. No file decoding or numerical packing occurs
    here. Callers retain returned storage until the supplied stream completes.
    """
    import cupy as cp
    layout = ARGUMENTS[entry]
    n = arguments[layout[0]['name']].shape[-1]
    pointers = []
    for field in layout:
        a = arguments[field['name']]
        dtype = np.dtype('float32' if field['type'] == 'float' else 'int32')
        if not isinstance(a, cp.ndarray) or a.dtype != dtype or not a.flags.c_contiguous or a.shape[-1] != n:
            raise ValueError(f"{field['name']} must be a contiguous CUDA {dtype} array with {n} cells")
        pointers.append(a.data.ptr)
    presence = []
    for field in layout:
        if field['optional']:
            flag = cp.full(n, np.int32((present or {}).get(field['name'], True)), dtype=cp.int32)
            pointers.append(flag.data.ptr)
            presence.append(flag)
    if not isinstance(tables, cp.ndarray) or tables.dtype != cp.float32 or tables.size != TABLE_WORDS:
        raise ValueError('RUC parameter tables must be 955 resident float32 words')
    rows, bound = arena_rows(entry, arguments, scalars)
    if workspace is not None and (not isinstance(workspace,cp.ndarray) or workspace.dtype!=cp.float32 or workspace.ndim!=2
                                  or workspace.shape[1]!=n or workspace.shape[0]<rows or not workspace.flags.c_contiguous):
        raise ValueError(f'RUC workspace must have shape (>= {rows} arena words, {n} cells), float32 and contiguous storage')
    # Arena offsets are int32 (Arena::alloc used*stride, stride = cells).  THE BREAKAGE THIS PREVENTS: past
    # 2**31 words the product wraps and cells address another cell's scratch or memory outside the workspace.
    if n * (WORK_WORDS if bound is None else bound) >= 2**31:
        raise ValueError(f'RUC launch of {n} cells x {WORK_WORDS if bound is None else bound} arena words overflows '
                         'the kernel int32 arena offsets; run this tile on more ranks')
    if errors is not None and (not isinstance(errors,cp.ndarray) or errors.dtype!=cp.int32 or errors.shape!=(1,)):
        raise ValueError('RUC fatal flag must be one resident int32 word')
    # Rows are the entry arena bound (WORK_WORDS only when the source or extents cannot bound it).  THE BREAKAGE
    # THIS PREVENTS: framework_init_rrfs.initialize_ruc launches init_rucinit (9 words per cell) with no workspace, and
    # took WORK_WORDS rows: 10 GB at initialization on x1.163842 (one RTX 5090, 2026-10-28), 19.6 GB on a conus3km rank.
    if workspace is None: workspace = cp.empty((rows,n), dtype=cp.float32)
    if errors is None: errors = cp.zeros(1,dtype=cp.int32)
    # Native solve loops over its:ite / jts:jte and rucinit over 1:ncells
    # write no cell fields for an empty local tile. Avoid a zero CUDA grid
    # and the preflight's read of a nonexistent per-cell scalar.
    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:495-581
    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_lsm_rucinit.F:179-243
    if n == 0:
        return {'workspace':workspace,'pointers':cp.asarray(pointers,dtype=cp.uint64),
                'presence':presence,'tile_stop':None,'errors':errors,
                'arguments':arguments,'tables':tables}
    tile_stop = None
    if entry == 'land_ruc_land':
        # Native L501 exits the whole tile; L505-582 mutates preceding cells.
        tile_stop = cp.empty(2, dtype=cp.int32)
        pointers.append(tile_stop.data.ptr)
    native_entry = None
    entry_calls = None
    if entry in ('land_ruc_land','ice_ruc_ice'):
        # Native tap PRE bytes for land OUT / sea-ice INOUT dummy storage. This explicit
        # context is replay input, never an expected output or inferred value.
        # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:3677-3698
        entry_key = 'snowtemp.ilnb' if entry == 'land_ruc_land' else 'snowseaice.ilnb'
        native_entry = (entry_storage or {}).get(entry_key)
        if native_entry is not None and (not isinstance(native_entry,cp.ndarray) or native_entry.dtype!=cp.int32 or native_entry.shape!=(n,) or not native_entry.flags.c_contiguous):
            raise ValueError('Native snowtemp entry storage must be a contiguous resident int32 word per cell')
        pointers.append(native_entry.data.ptr if native_entry is not None else 0)
        entry_calls = cp.zeros(n,dtype=cp.int32) if trace_entry_calls else None
        pointers.append(entry_calls.data.ptr if entry_calls is not None else 0)
    ptrs = cp.asarray(pointers,dtype=cp.uint64)
    module = _module(cp.cuda.Device().id, entry == 'ice_ruc_ice' and (native_entry is not None or trace_entry_calls))
    if tile_stop is not None:
        module.get_function('ruc_land_tile_preflight')((1,), (1,),
            (ptrs,tile_stop,np.int32(n)), stream=stream)
    module.get_function('oracle_'+entry)(
        ((n+63)//64,), (64,), (ptrs,tables,workspace,errors,np.int32(n)), stream=stream)
    return {'workspace':workspace,'pointers':ptrs,'presence':presence,'tile_stop':tile_stop,'errors':errors,
            'arguments':arguments,'tables':tables,'entry_storage':entry_storage,'native_entry':native_entry,'entry_calls':entry_calls}

def launch_boundary(entry, arguments, tables, *, workspace=None, errors=None, stream=None, present=None, entry_storage=None, trace_entry_calls=False, scalars=None):
    """Allocate and launch on one stream, preserving producer dependencies."""
    import cupy as cp
    with stream or cp.cuda.get_current_stream():
        return _launch_boundary(entry,arguments,tables,workspace=workspace,errors=errors,stream=stream,present=present,entry_storage=entry_storage,trace_entry_calls=trace_entry_calls,scalars=scalars)


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:717-847
# Native pool names are restart identities; no persistent array is hidden in
# a launcher cache. The provider owns table words, boundary work and retention.
PERSISTENT_FIELDS = {
    'sfc_input': ('sh2o','smois','tslb','snoalb','sfc_albbck','snow','snowc','snowh',
                  'skintemp','tmn','vegfra','xice','xland'),
    'diag_physics': ('acsnom','acsnow','canwat','chklowq','grdflx','hfx','lai','lh',
                     'qfx','qsfc','sfc_albedo','sfc_emiss','sfcrunoff','smstav','smstot',
                     'udrunoff','z0','znt','rhosnf','precipfr','qsg','qvg','dew',
                     'soilt1','tsnav','acrunoff','snowfallac','smfr3d','keepfr3d',
                     'graupelncv','snowncv','rainncv','qcg','flqc','flhc','sfcexc',
                     'sfcevp','mavail'),
}


class RucBackend:
    """Complete entries bound to the RRFS provider's resident native work pools.

    ``provider.prepare_ruc(entry, call)`` performs native driver preparation in
    CUDA/Rust and returns an argument mapping with the exact ARGUMENTS layout.
    ``provider.finish_ruc`` updates persistent pools and retains returned CUDA
    storage until its stream finishes. File/table decoding belongs to Rust or
    the native table loader; this class contains only dispatch and allocation.
    """
    def __init__(self, provider):
        self.provider = provider

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:988-1036
    def land(self, call):
        return self._entry('land_ruc_land', call)

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:774-814
    def ice(self, call):
        return self._entry('ice_ruc_ice', call)

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_sfc_diagnostics.F:125-145
    def diagnostics(self, call):
        return self._entry('diags_sfcdiags_ruclsm', call)

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_lsm_rucinit.F:145-244
    def initialize(self, call):
        return self._entry('init_rucinit', call)

    def _entry(self, entry, call):
        arguments = self.provider.prepare_ruc(entry, call)
        storage = launch_boundary(entry, arguments, self.provider.ruc_tables,
                                  stream=self.provider.stream,
                                  workspace=getattr(self.provider,'workspace',None),
                                  scalars=getattr(self.provider,'scalars',None),
                                  present={'pattern_spp_lsm':entry!='ice_ruc_ice',
                                           'field_sf':entry!='ice_ruc_ice'})
        return self.provider.finish_ruc(entry, call, arguments, storage)


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:277-298
# The RRFS registry table must list sf_ruc. Registering into the v8.4.2 registry
# correctly fails; that tree has no native sf_ruc selector.
def register(registry, provider):
    backend = RucBackend(provider)
    registry.register('land_surface', SCHEME, BACKEND, backend.land)
    registry.register('seaice', SCHEME, BACKEND, backend.ice)
    return backend

# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:470-640,994-1030
SFC_BINDINGS = {
    'zs':'zs','snow':'snow','snowh':'snowh','snowc':'snowc',
    'vegfra':'vegfra','snoalb':'snoalb','albbck':'sfc_albbck','tbot':'tmn',
    'ivgtyp':'ivgtyp','isltyp':'isltyp','xland':'xland','xice':'xice',
    'soilmois':'smois','smois':'smois','sh2o':'sh2o','tslb':'tslb','tso':'tslb',
    'soilt':'skintemp','tsk':'skintemp','shdmin':'shdmin','shdmax':'shdmax',
    'landusef':'landusef','soilctop':'soilf',
}
DIAG_BINDINGS = {
    'alb':'sfc_albedo','emiss':'sfc_emiss','smavail':'smstav','smmax':'smstot',
    'snom':'acsnom','keepfr3dflag':'keepfr3d','t2':'t2m','th2':'th2m',
    'psfc2d':'psfc',
}
ATM_BINDINGS = {'z3d':'dz_p','dz':'dz_p','p8w':'pres2_hyd_p','p3d':'pres2_hyd_p',
                't3d':'t_p','qv3d':'qv_p','qc3d':'qc_p','rho3d':'rho_p'}
# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:607-669
ICE_COPYBACK = frozenset(('tso','z0','snow','snowc','snowh','soilt','snom','acsnow',
    'sfcrunoff','znt','grdflx','chs','qsfc','hfx','qfx','lh','alb','emiss',
    'flhc','flqc','qcg','qsg','qvg','dew','soilt1','tsnav','snowfallac'))
ICE_FINISH_FIELDS = ('flhc','flqc','cqs','cpm','chs','chs2','cqs2','hfx',
                     'lh','qfx','qgh','qsfc','tsk')
# Threads evaluate per-cell arithmetic with the native tile return preserved
# by preflight. The packed cell axis replaces i/j; vertical extents are intact.
# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:469-499,681-686
COLUMN_EXTENTS = frozenset(('ids','ide','jds','jde','ims','ime','jms','jme',
                           'its','ite','jts','jte','ncells'))


class ResidentRucProvider:
    """Native pool adapter with complete RUC driver conversions on CUDA.

    Pools contain resident sfc_input, diag_physics, mesh and physics mappings.
    Physics includes the original MPAS_to_physics atmospheric and open-water
    work arrays. Scalars contain resolved native constants and configuration.
    Rust/native table decoding and framework scheduling precede this adapter.
    """
    def __init__(self, pools, scalars, tables, *, stream=None):
        import cupy as cp
        self.pools, self.scalars, self.ruc_tables = pools, scalars, tables
        self.stream = stream or cp.cuda.get_current_stream()
        self.retained = []
        self.work = {}
        self.workspace = None

    def _scalar(self, name, n, value=None, dtype='float32'):
        import cupy as cp
        return cp.full(n, self.scalars[name] if value is None else value, dtype=dtype)

    def _boundary(self, entry, arguments):
        storage = launch_boundary(entry, arguments, self.ruc_tables, stream=self.stream,
                                  workspace=self.workspace, scalars=self.scalars)
        self.retained.append((arguments, storage))
        return storage

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:470-640,752-854
    # ruclsminit initializes exactly nCells; native storage can include an
    # extra cell that must stay untouched. Driver solves only its native tile.
    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_lsm_rucinit.F:179-243
    def _cells(self, entry):
        return (slice(0,self.scalars['ncells']) if entry=='init_rucinit' else
                slice(self.scalars['its']-1,self.scalars['ite']))

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:470-640
    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_sfc_diagnostics.F:130-143
    def prepare_ruc(self, entry, call):
        import cupy as cp
        with self.stream:
            sfc,diag,physics = (self.pools[p] for p in ('sfc_input','diag_physics','physics'))
            cells=self._cells(entry)
            # Numeric initialization queries soil storage, not skintemp.
            # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_lsm_rucinit.F:111-136,176-243
            cell_field='tslb' if entry=='init_rucinit' else 'skintemp'
            n=sfc[cell_field][...,cells].shape[-1]
            if entry=='diags_sfcdiags_ruclsm' and not self.scalars['mynn']:
                cp.copyto(diag['cqs'],diag['chs'])
            result={}
            for field in ARGUMENTS[entry]:
                name=field['name']
                if name in ATM_BINDINGS:
                    array=physics[ATM_BINDINGS[name]][...,cells]
                elif entry=='diags_sfcdiags_ruclsm' and name=='psfc2d':
                    # The diagnostic driver reads diag.surface_pressure, not
                    # the surface-layer psfc work value in diag_physics.
                    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_sfc_diagnostics.F:69
                    array=self.pools['diag']['surface_pressure'][...,cells]
                elif name=='globalcells':
                    array=self.pools['mesh']['indexToCellID'][...,cells]
                elif name in ('pattern_spp_lsm','field_sf'):
                    array=cp.zeros((self.scalars['kme']-self.scalars['kms']+1,n),cp.float32)
                elif name=='lakemask':
                    array=cp.zeros(n,cp.float32)
                elif name in ('rainbl','frzfrac'):
                    array=cp.empty(n,cp.float32)
                elif name in SFC_BINDINGS:
                    array=sfc[SFC_BINDINGS[name]][...,cells]
                elif field['dim']:
                    # The native land driver rebinds its qsg pointer to qcg.
                    # Its output copy later reacquires the actual qsg pool.
                    key='qcg' if entry=='land_ruc_land' and name=='qsg' else DIAG_BINDINGS.get(name,name)
                    array=diag[key][...,cells]
                else:
                    if name in ('spp_lsm','mosaic_lu','mosaic_soil'):
                        key='spp_lsm_loc' if name=='spp_lsm' else name
                        flag=diag[key]
                        if flag.ndim:flag=flag[...,cells]
                        result[name]=cp.broadcast_to(flag,(n,)).astype(cp.int32,copy=True)
                        continue
                    value=(1 if name in COLUMN_EXTENTS else call.dt if name=='dt'
                           else call.itimestep if name=='ktau' else self.scalars[name])
                    result[name]=self._scalar(name,n,value,dtype='float32' if field['type']=='float' else 'int32')
                    continue
                # Native drivers copy into independent work arrays. Scheme
                # writes must not alias their incoming MPAS pools or each other.
                result[name]=cp.array(array,copy=True,order='C')
            # Calls on this stream complete in order, so scratch is reusable.  Rows are the entry's own
            # arena bound (arena_rows); returned storage retains previous buffers if the tile size changes.
            rows,_=arena_rows(entry,result,self.scalars)
            if self.workspace is None or self.workspace.shape[1]!=n or self.workspace.shape[0]<rows:
                self.workspace=cp.empty((rows,n),cp.float32)
            if entry in ('land_ruc_land','ice_ruc_ice'):
                # Driver flags are hardcoded in the pinned MPAS fork.
                result['myjpbl'].fill(0)
                result['frpcpn'].fill(1)
                if entry=='land_ruc_land':result['rdlai2d'].fill(1)
                # Native zs work profile is taken from the first tile cell.
                cp.copyto(result['zs'],cp.broadcast_to(sfc['zs'][:,cells][:,:1],result['zs'].shape))
                forcing={'gsw':diag['gsw'][...,cells],'ncells':self._scalar('ncells',n,1,dtype='int32')}
                for name,flag in (('sr','micro_on'),('rainncv','micro_on'),('raincv','conv_on')):
                    forcing[name]=diag[name][...,cells] if self.scalars[flag] else cp.zeros(n,cp.float32)
                forcing['sfc_albedo']=diag['sfc_albedo'][...,cells]
                forcing.update(sr_p=result['frzfrac'],rainbl_p=result['rainbl'],swdown_p=cp.empty(n,cp.float32))
                for flag in ('micro_on','conv_on'):
                    forcing[flag]=self._scalar(flag,n,dtype='int32')
                self._boundary('drivers_driver_forcing',forcing)
                if entry=='ice_ruc_ice':self._prepare_ice(result,n)
            if entry=='init_rucinit':
                # Initialization writes these scalar MPAS pool flags even on
                # restart/cycling runs; native table reading occurs beforehand.
                for name,value in (('spp_lsm_loc',0),('mosaic_lu',1),('mosaic_soil',1)):
                    diag[name].fill(value)
            self.work[entry]=result
            return result

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:373-458
    def _prepare_ice(self, result, n):
        import cupy as cp
        sfc,diag,physics=(self.pools[p] for p in ('sfc_input','diag_physics','physics'))
        cells=self._cells('ice_ruc_ice')
        background=cp.full(n,self.scalars['seaice_albedo_default'],cp.float32)
        # This selection is the original driver branch, with no arithmetic.
        cp.copyto(background,sfc['sfc_albbck'][...,cells],where=result['xice']<result['xice_threshold'])
        result['albbck']=background
        prep={k:result[k] for k in ('xice_threshold','xice')}
        prep['ncells']=self._scalar('ncells',n,1,dtype='int32')
        prep.update(sfc_albedo=diag['sfc_albedo'][...,cells],sfc_emiss=diag['sfc_emiss'][...,cells],
                    sfc_albedo_p=result['alb'],sfc_emiss_p=result['emiss'])
        self._boundary('drivers_seaice_prepare',prep)
        correction={k:result[k] for k in ('ims','ime','jms','jme','its','ite','jts','jte','xice')}
        correction.update(xice_thresh=result['xice_threshold'],tsk=result['soilt'],
                          tsk_sea=physics['tsk_sea'][...,cells],tsk_ice=cp.empty(n,cp.float32))
        self._boundary('drivers_correct_tsk_over_seaice',correction)
        cp.copyto(result['soilt'],correction['tsk_ice'])
        # Surface-layer work values are available before the RUC ice call.
        for name in ICE_FINISH_FIELDS:
            if name not in ('tsk',) and name not in result:result[name]=cp.array(diag[name][...,cells],copy=True)

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:571-601
    def _finish_ice(self, arguments, n):
        import cupy as cp
        cells=self._cells('ice_ruc_ice')
        blend={'ncells':self._scalar('ncells',n,1,dtype='int32'),'xice_p':arguments['xice'],'xice_threshold':arguments['xice_threshold'],
               'sfc_albedo_p':arguments['alb'],'sfc_emiss_p':arguments['emiss']}
        for flag in ('config_frac_seaice','mynn'):blend[flag]=self._scalar(flag,n,dtype='int32')
        for name in ICE_FINISH_FIELDS:
            blend[name+'_p']=arguments['soilt' if name=='tsk' else name]
            blend[name+'_sea']=self.pools['physics'][name+'_sea'][...,cells]
        self._boundary('drivers_seaice_finish',blend)
        for name in ('cqs','cpm','chs2','cqs2','qgh'):
            cp.copyto(self.pools['diag_physics'][name][...,cells],arguments[name])

    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:767-847
    # RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:603-668
    def finish_ruc(self, entry, call, arguments, storage):
        import cupy as cp
        with self.stream:
            cells=self._cells(entry)
            if entry=='ice_ruc_ice':self._finish_ice(arguments,arguments['dt'].size)
            for field in ARGUMENTS[entry]:
                name=field['name']
                if not field['dim']:continue
                if entry=='ice_ruc_ice':
                    if name not in ICE_COPYBACK:continue
                elif field['intent']=='in':continue
                if name in ('pattern_spp_lsm','field_sf'):continue
                if name in SFC_BINDINGS:
                    target=self.pools['sfc_input'][SFC_BINDINGS[name]][...,cells]
                else:
                    target=self.pools['diag_physics'][DIAG_BINDINGS.get(name,name)][...,cells]
                cp.copyto(target,arguments[name])
            if entry=='ice_ruc_ice':
                # Native copies these outputs explicitly, including its flqc
                # assignment from flhc. Native-defect policy requires parity.
                diag=self.pools['diag_physics']
                cp.copyto(diag['flqc'][...,cells],arguments['flhc'])
                cp.copyto(self.pools['sfc_input']['sfc_albbck'][...,cells],arguments['albbck'])
            if entry=='land_ruc_land' and self.scalars['micro_on']:
                cp.copyto(self.pools['diag_physics']['sr'][...,cells],arguments['frzfrac'])
            self.retained.append((arguments,storage))
        return storage

    def synchronize(self):
        import cupy as cp
        # Every flag is set only at a source-native fatal-error call site. On the current stream the words are read
        # off the step path (deferred_checks_v842: a forecast still stops on them, a replay at once); THE COST THIS
        # REMOVES: a host sync and a device-to-host copy per RUC call (land, sea ice, surface diagnostics: eight per
        # hrrrv5 step). Another stream keeps the synchronous read.
        if self.stream.ptr == cp.cuda.get_current_stream().ptr:
            from ..deferred_checks_v842 import defer
            for _,storage in self.retained:
                defer(storage['errors'][:1], 'RUC reached a pinned native fatal-error branch')
            self.retained.clear()
            self.work.clear()
            return
        self.stream.synchronize()
        native_fatal=any(int(storage['errors'].get()[0]) for _,storage in self.retained)
        self.retained.clear()
        self.work.clear()
        if native_fatal:raise RuntimeError('RUC reached a pinned native fatal-error branch')


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:902-1036
# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:688-817
_NATIVE_CONSTANTS = {'cp':'cp', 'r_d':'R_d', 'rovcp':'rcp', 'g0':'gravity',
                     'lv':'xlv', 'stbolt':'stbolt'}
_NATIVE_VARIABLES = {'nsl':'num_soils', 'nlcat':'num_landcat', 'nscat':'num_soilcat'}


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_lsm_rucinit.F:148-174
# Character tokens are the port's representation, not numerical table values.
def _native_ruc_tables(native, cp, landuse):
    words = cp.empty(TABLE_WORDS, dtype=cp.float32)
    tokens = {'lutype':4 if landuse == 'USGS' else 5, 'sltype':6}
    for name, spec in TABLE_LAYOUT.items():
        if spec['character']:
            value = cp.asarray([tokens[name]], dtype=cp.int32)
        else:
            record = native.get_table('module_ruc_land.' + name)
            if record is None:
                raise RuntimeError('Native initialized RUC table is absent: ' + name)
            value = cp.asarray(record[0]).reshape(-1, order='F')
        if value.size != spec['words']:
            raise ValueError('Native RUC table extent differs: ' + name)
        begin = spec['offset']
        cp.copyto(words.view(cp.uint32)[begin:begin + spec['words']], value.view(cp.uint32))
    return words


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:388-640,902-1036
# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:253-458,688-817
# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_sfc_diagnostics.F:32-145
# Full ABI allocations are retained: the generic solve-cell views lose tails.
def _run_scalar(provider, key, read):
    """Framework constants belong to the provider's run, including fallback bindings.

    These words are initialized before physics starts. Never cache a field or a
    prognostic value here. Keeping the native owner guards a provider rebound to
    a new framework without relying on object ids that Python can reuse.
    """
    cached = getattr(provider, '_rrfs_ruc_constants', None)
    if cached is None or cached[0] is not provider.native:
        cached = (provider.native, {})
        provider._rrfs_ruc_constants = cached
    values = cached[1]
    if key not in values:
        values[key] = read()
    return values[key]


def _native_scalar(provider, name):
    native = provider.native
    def read():
        if name in _NATIVE_CONSTANTS:
            return native.get_constant(_NATIVE_CONSTANTS[name])
        if name in ('isice', 'iswater'):
            # Land-use categories are immutable initialized words, read once per run through the framework
            # (on the resident facade that is the device authority), as before the scalar cache. Not the facade's
            # _host storage: on trees whose physics-session hydration adopts the read payload on the device
            # (rrfs/c3-8card*), the host sfc_input words are not the initialized values, and RUC took its pinned
            # native fatal-error branch at step 2 (mv3km hrrrv5 1 card, rrfs/c3-8card-speed 826f1d6).
            return native.get_field('sfc_input', name, level=1).item()
        record = native.get_variable(_NATIVE_VARIABLES.get(name, name))
        if record is None:
            raise RuntimeError('Native RUC scalar is absent: ' + name)
        return record[0].item()
    return _run_scalar(provider, ('scalar', name), read)


def _native_interval(provider):
    def read():
        interval = provider.native.get_variable('dt_pbl')
        return None if interval is None else interval[0].item()
    return _run_scalar(provider, ('interval', 'dt_pbl'), read)


def _bind_native_ruc(provider, call, entry):
    native, cp = provider.native, provider.cp
    options = dict(provider.session.layout.configs)
    options.update(call.options)
    scalars = {}
    landuse = None
    if entry in ('land_ruc_land','ice_ruc_ice'):
        landuse = native.get_text_field('sfc_input', 'mminlu')
        scalars.update(myjpbl=0, frpcpn=1, rdlai2d=1,
                       micro_on=options['config_microp_scheme'].strip() != 'off',
                       conv_on=options['config_convection_scheme'].strip() != 'off',
                       restart=options['config_do_restart'], cycling=options['config_do_DAcycling'],
                       mminlu={'USGS':1, 'MODIS':2, 'MODIFIED_IGBP_MODIS_NOAH':3}[landuse])
        if entry == 'land_ruc_land':
            scalars['lakemodel'] = options['config_lakemodel']
        else:
            scalars.update(config_frac_seaice=options['config_frac_seaice'],
                           mynn=options['config_sfclayer_scheme'].strip() == 'sf_mynnsfclay')
            # Parameter in MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_vars.F:824, not a runtime export.
            scalars['seaice_albedo_default'] = np.float32('0.65')
    else:
        scalars['mynn'] = options['config_sfclayer_scheme'].strip() == 'sf_mynnsfclay'
    for field in ARGUMENTS[entry]:
        name = field['name']
        if field['dim'] or name in scalars or name in ('spp_lsm','mosaic_lu','mosaic_soil',
                'dt','ktau','frpcpn','myjpbl','rdlai2d'):
            continue
        scalars[name] = _native_scalar(provider, name)
    scalars.update(its=provider.session.arguments.its, ite=provider.session.arguments.ite)
    # dt_pbl is the native manager interval, not the dynamics time step.
    interval = _native_interval(provider) if entry != 'diags_sfcdiags_ruclsm' else None
    required = {pool:set() for pool in ('sfc_input','diag_physics','mesh','diag','physics')}
    required['sfc_input'].add('skintemp')
    for field in ARGUMENTS[entry]:
        name = field['name']
        if not field['dim']:
            if name in ('spp_lsm','mosaic_lu','mosaic_soil'):
                required['diag_physics'].add('spp_lsm_loc' if name == 'spp_lsm' else name)
        elif name in ATM_BINDINGS:
            required['physics'].add(ATM_BINDINGS[name])
        elif name == 'globalcells':
            required['mesh'].add('indexToCellID')
        elif entry == 'diags_sfcdiags_ruclsm' and name == 'psfc2d':
            required['diag'].add('surface_pressure')
        elif name in SFC_BINDINGS:
            required['sfc_input'].add(SFC_BINDINGS[name])
        elif name not in ('pattern_spp_lsm','field_sf','lakemask','rainbl','frzfrac'):
            required['diag_physics'].add('qcg' if entry == 'land_ruc_land' and name == 'qsg'
                                         else DIAG_BINDINGS.get(name,name))
    # Copyback can target a different native pool than its input pointer
    # (land qsg is the pinned example). Preserve both full allocations.
    for field in ARGUMENTS[entry]:
        name = field['name']
        if not field['dim'] or field['intent'] == 'in' or name in ('pattern_spp_lsm','field_sf'):
            continue
        if entry == 'ice_ruc_ice' and name not in ICE_COPYBACK:
            continue
        pool = 'sfc_input' if name in SFC_BINDINGS else 'diag_physics'
        required[pool].add(SFC_BINDINGS[name] if pool == 'sfc_input' else DIAG_BINDINGS.get(name,name))
    if entry in ('land_ruc_land','ice_ruc_ice'):
        required['sfc_input'].add('zs')
        required['diag_physics'].update(('gsw','sfc_albedo'))
        if scalars['micro_on']: required['diag_physics'].update(('sr','rainncv'))
        if scalars['conv_on']: required['diag_physics'].add('raincv')
    if entry == 'diags_sfcdiags_ruclsm':
        required['diag_physics'].update(('cqs','chs'))
    if entry == 'ice_ruc_ice':
        required['diag_physics'].update(name for name in ICE_FINISH_FIELDS if name != 'tsk')
        required['physics'].update(name + '_sea' for name in ICE_FINISH_FIELDS)
    pools = {pool:{} for pool in required}
    prepared_shapes, lower = {}, {}
    for pool, names in required.items():
        for name in sorted(names):
            if pool == 'physics':
                host, bounds = native.get_variable(name)
                array = cp.asarray(host, order='F')
                prepared_shapes[name], lower[name] = array.shape, bounds
                array = cp.moveaxis(array,0,-1)
                array = array.reshape(tuple(d for d in array.shape[:-1] if d != 1) + (array.shape[-1],))
            else:
                array = cp.asarray(native.get_field(pool,name,level=1))
            pools[pool][name] = cp.ascontiguousarray(array) if array.ndim else array
    # Surface diagnostics has no table references; its launch still accepts
    # the common pointer, which is not dereferenced by that entry.
    tables = cp.empty(TABLE_WORDS,cp.float32) if landuse is None else _native_ruc_tables(native,cp,landuse)
    resident = ResidentRucProvider(pools, scalars, tables)
    return dict(resident=resident, native=native, entry=entry, lower=lower,
                prepared_shapes=prepared_shapes, dt=interval)


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:277-325
# The owner publishes the table descriptor; this module supplies only its slot.
def bind_native_pools(provider, call):
    entry = 'ice_ruc_ice' if call.slot == 'seaice' else 'land_ruc_land'
    from .resident_ruc_rrfs import resident_slot
    slot = resident_slot(provider, call, entry)
    if slot is not None:
        # Resident facade route: constants, tables and work arrays once, fused gather/scatter (resident_ruc_rrfs).
        return {'ruc_native':{'resident_slot':slot}}
    return {'ruc_native':_bind_native_ruc(provider, call, entry)}


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F:767-847
# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F:603-668
# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_sfc_diagnostics.F:125-145
# entry_storage is only supplied by authenticated native-capture replay, never
# synthesized by the forecast callback. Every unchanged allocation tail stays.
def execute_native_context(context, call, *, entry_storage=None):
    import cupy as cp
    from dataclasses import replace
    resident, entry = context['resident'], context['entry']
    scientific_call = replace(call, dt=context['dt']) if context['dt'] is not None else call
    arguments = resident.prepare_ruc(entry,scientific_call)
    storage = launch_boundary(entry,arguments,resident.ruc_tables,stream=resident.stream,
                              workspace=resident.workspace,scalars=resident.scalars,
                              present={'pattern_spp_lsm':entry != 'ice_ruc_ice',
                                       'field_sf':entry != 'ice_ruc_ice'},
                              entry_storage=entry_storage)
    resident.finish_ruc(entry,scientific_call,arguments,storage)
    resident.synchronize()
    native = context['native']
    for pool in ('sfc_input','diag_physics'):
        for name,value in resident.pools[pool].items():
            native.put_field(pool,name,value,level=1,nlevels=native._fields[pool,name][2])
    for name,value in resident.pools['physics'].items():
        value = cp.moveaxis(value,-1,0).reshape(context['prepared_shapes'][name])
        native.put_variable(name,value,lower=context['lower'][name])
    return storage


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:277-325
# Native table/flag initialization remains the owner's initialization hook;
# RucBackend.initialize is the separately replayed complete CUDA numerical block.
def run_slot(call):
    context = call.pools['ruc_native']
    if 'resident_slot' in context:
        return context['resident_slot'].run(call)
    return execute_native_context(context,call)


# RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_sfc_diagnostics.F:32-145
# This complete callback replaces the native post-LSM diagnostics hook only.
def run_native_diagnostics(provider,call):
    from .resident_ruc_rrfs import resident_slot
    slot = resident_slot(provider, call, 'diags_sfcdiags_ruclsm')
    if slot is not None:
        return slot.run(call)
    return execute_native_context(_bind_native_ruc(provider,call,'diags_sfcdiags_ruclsm'),call)
