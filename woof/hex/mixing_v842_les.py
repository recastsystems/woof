"""MPAS-A v8.4.2 LES dissipation at the native module boundary.

Pristine source authority: src/core_atmosphere/dynamics/
mpas_atm_dissipation_models.F at MPAS commit f34984b2c89353cf31c235b08d3b4acd96b37bb3.
The entire five-routine LES path is included, including the non-LES filter
arms that share those routines. Python only validates and launches CUDA.

Arrays use native dimension order, Fortran contiguous float32/int32 storage,
and one-based connectivity (including the native n+1 sentinel). Bounds and
scalar indices are one-based inclusive. Inputs and outputs stay on device.
The caller owns halo exchanges and MPAS stage ordering; these routines never
advance prognostic fields except the native negative-TKE clamp. An existing
value in an intent(out) tendency is preserved wherever MPAS reads it first.

Each native stencil pass is a separate CUDA launch, with one thread per
column, edge or vertex. Stream ordering preserves native stencil barriers;
vertical loops and edge reductions retain their original order. No forecast
identity claim follows from a standalone replay.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Any
import ast
import operator
import numpy as np

LES_INVALID_OPT = -1
LES_MODEL_NONE = 0
LES_MODEL_3D_SMAGORINSKY = 1
LES_MODEL_PROGNOSTIC_15_ORDER = 2
LES_SURFACE_NONE = 0
LES_SURFACE_SPECIFIED = 1
LES_SURFACE_VARYING = 2


def les_model_from_string(value: str) -> int:
    """mpas_atm_dissipation_models.F:57-78 (Fortran TRIM removes trailing spaces)."""
    return {"none": 0, "3d_smagorinsky": 1,
            "prognostic_1.5_order": 2}.get(value.rstrip(" "), -1)


def les_surface_from_string(value: str) -> int:
    """mpas_atm_dissipation_models.F:95-116."""
    return {"none": 0, "specified": 1, "varying": 2}.get(value.rstrip(" "), -1)


ROUTINES = {'les_models': {'args': ['les_model_opt',
                         'les_surface_opt',
                         'dynamics_substep',
                         'eddy_visc_horz',
                         'eddy_visc_vert',
                         'u',
                         'v',
                         'ucell',
                         'vcell',
                         'w',
                         'c_s',
                         'bv_freq2',
                         'zgrid',
                         'config_len_disp',
                         'deformation_coef_c2',
                         'deformation_coef_s2',
                         'deformation_coef_cs',
                         'deformation_coef_c',
                         'deformation_coef_s',
                         'prandtl_3d_inv',
                         'invdt',
                         'h_mom_eddy_visc4',
                         'config_visc4_2dsmag',
                         'h_theta_eddy_visc4',
                         'scalars',
                         'tend_scalars',
                         'index_tke',
                         'rho_zz',
                         'meshscalingdel2',
                         'cellstart',
                         'cellend',
                         'nedgesoncell',
                         'edgesoncell',
                         'cellsonedge',
                         'ncells',
                         'nedges',
                         'nvertlevels',
                         'maxedges',
                         'num_scalars'],
                'fields': {'les_model_opt': {'type': 'int',
                                             'dims': [],
                                             'logical': False,
                                             'intent': 'in'},
                           'les_surface_opt': {'type': 'int',
                                               'dims': [],
                                               'logical': False,
                                               'intent': 'in'},
                           'dynamics_substep': {'type': 'int',
                                                'dims': [],
                                                'logical': False,
                                                'intent': 'in'},
                           'eddy_visc_horz': {'type': 'float',
                                              'dims': ['nvertlevels', 'ncells+1'],
                                              'logical': False,
                                              'intent': 'out'},
                           'eddy_visc_vert': {'type': 'float',
                                              'dims': ['nvertlevels', 'ncells+1'],
                                              'logical': False,
                                              'intent': 'out'},
                           'u': {'type': 'float',
                                 'dims': ['nvertlevels', 'nedges+1'],
                                 'logical': False,
                                 'intent': 'in'},
                           'v': {'type': 'float',
                                 'dims': ['nvertlevels', 'nedges+1'],
                                 'logical': False,
                                 'intent': 'in'},
                           'ucell': {'type': 'float',
                                     'dims': ['nvertlevels', 'ncells+1'],
                                     'logical': False,
                                     'intent': 'in'},
                           'vcell': {'type': 'float',
                                     'dims': ['nvertlevels', 'ncells+1'],
                                     'logical': False,
                                     'intent': 'in'},
                           'w': {'type': 'float',
                                 'dims': ['nvertlevels+1', 'ncells+1'],
                                 'logical': False,
                                 'intent': 'in'},
                           'c_s': {'type': 'float',
                                   'dims': [],
                                   'logical': False,
                                   'intent': 'in'},
                           'bv_freq2': {'type': 'float',
                                        'dims': ['nvertlevels', 'ncells+1'],
                                        'logical': False,
                                        'intent': 'in'},
                           'zgrid': {'type': 'float',
                                     'dims': ['nvertlevels+1', 'ncells+1'],
                                     'logical': False,
                                     'intent': 'in'},
                           'config_len_disp': {'type': 'float',
                                               'dims': [],
                                               'logical': False,
                                               'intent': 'in'},
                           'deformation_coef_c2': {'type': 'float',
                                                   'dims': ['maxedges', 'ncells+1'],
                                                   'logical': False,
                                                   'intent': 'in'},
                           'deformation_coef_s2': {'type': 'float',
                                                   'dims': ['maxedges', 'ncells+1'],
                                                   'logical': False,
                                                   'intent': 'in'},
                           'deformation_coef_cs': {'type': 'float',
                                                   'dims': ['maxedges', 'ncells+1'],
                                                   'logical': False,
                                                   'intent': 'in'},
                           'deformation_coef_c': {'type': 'float',
                                                  'dims': ['maxedges', 'ncells+1'],
                                                  'logical': False,
                                                  'intent': 'in'},
                           'deformation_coef_s': {'type': 'float',
                                                  'dims': ['maxedges', 'ncells+1'],
                                                  'logical': False,
                                                  'intent': 'in'},
                           'prandtl_3d_inv': {'type': 'float',
                                              'dims': ['nvertlevels', 'ncells+1'],
                                              'logical': False,
                                              'intent': 'out'},
                           'invdt': {'type': 'float',
                                     'dims': [],
                                     'logical': False,
                                     'intent': 'in'},
                           'h_mom_eddy_visc4': {'type': 'float',
                                                'dims': [],
                                                'logical': False,
                                                'intent': 'out'},
                           'config_visc4_2dsmag': {'type': 'float',
                                                   'dims': [],
                                                   'logical': False,
                                                   'intent': 'in'},
                           'h_theta_eddy_visc4': {'type': 'float',
                                                  'dims': [],
                                                  'logical': False,
                                                  'intent': 'out'},
                           'scalars': {'type': 'float',
                                       'dims': ['num_scalars', 'nvertlevels', 'ncells+1'],
                                       'logical': False,
                                       'intent': 'inout'},
                           'tend_scalars': {'type': 'float',
                                            'dims': ['num_scalars', 'nvertlevels', 'ncells+1'],
                                            'logical': False,
                                            'intent': 'inout'},
                           'index_tke': {'type': 'int',
                                         'dims': [],
                                         'logical': False,
                                         'intent': 'in'},
                           'rho_zz': {'type': 'float',
                                      'dims': ['nvertlevels', 'ncells+1'],
                                      'logical': False,
                                      'intent': 'in'},
                           'meshscalingdel2': {'type': 'float',
                                               'dims': ['nedges+1'],
                                               'logical': False,
                                               'intent': 'in'},
                           'cellstart': {'type': 'int',
                                         'dims': [],
                                         'logical': False,
                                         'intent': 'in'},
                           'cellend': {'type': 'int',
                                       'dims': [],
                                       'logical': False,
                                       'intent': 'in'},
                           'nedgesoncell': {'type': 'int',
                                            'dims': ['ncells+1'],
                                            'logical': False,
                                            'intent': 'in'},
                           'edgesoncell': {'type': 'int',
                                           'dims': ['maxedges', 'ncells+1'],
                                           'logical': False,
                                           'intent': 'in'},
                           'cellsonedge': {'type': 'int',
                                           'dims': ['2', 'nedges+1'],
                                           'logical': False,
                                           'intent': 'in'},
                           'ncells': {'type': 'int',
                                      'dims': [],
                                      'logical': False,
                                      'intent': 'in'},
                           'nedges': {'type': 'int',
                                      'dims': [],
                                      'logical': False,
                                      'intent': 'in'},
                           'nvertlevels': {'type': 'int',
                                           'dims': [],
                                           'logical': False,
                                           'intent': 'in'},
                           'maxedges': {'type': 'int',
                                        'dims': [],
                                        'logical': False,
                                        'intent': 'in'},
                           'num_scalars': {'type': 'int',
                                           'dims': [],
                                           'logical': False,
                                           'intent': 'in'}},
                'scratch': {'d_11': ['nvertlevels'],
                            'd_22': ['nvertlevels'],
                            'd_33': ['nvertlevels'],
                            'd_12': ['nvertlevels'],
                            'd_13': ['nvertlevels'],
                            'd_23': ['nvertlevels'],
                            'dudx': ['nvertlevels'],
                            'dudy': ['nvertlevels'],
                            'dvdx': ['nvertlevels'],
                            'dvdy': ['nvertlevels'],
                            'dwdx': ['nvertlevels+1'],
                            'dwdy': ['nvertlevels+1'],
                            'dudz': ['nvertlevels'],
                            'dvdz': ['nvertlevels'],
                            'dwdz': ['nvertlevels']},
                'lines': [208, 460]},
 'calculate_n2': {'args': ['bn2',
                           'theta_m',
                           'exner',
                           'pressure_b',
                           'pp',
                           'zgrid',
                           'scalars',
                           'index_qv',
                           'index_qc',
                           'qtot',
                           'cellstart',
                           'cellend',
                           'ncells',
                           'nvertlevels',
                           'maxedges',
                           'num_scalars'],
                  'fields': {'bn2': {'type': 'float',
                                     'dims': ['nvertlevels', 'ncells+1'],
                                     'logical': False,
                                     'intent': 'out'},
                             'theta_m': {'type': 'float',
                                         'dims': ['nvertlevels', 'ncells+1'],
                                         'logical': False,
                                         'intent': 'in'},
                             'exner': {'type': 'float',
                                       'dims': ['nvertlevels', 'ncells+1'],
                                       'logical': False,
                                       'intent': 'in'},
                             'pressure_b': {'type': 'float',
                                            'dims': ['nvertlevels', 'ncells+1'],
                                            'logical': False,
                                            'intent': 'in'},
                             'pp': {'type': 'float',
                                    'dims': ['nvertlevels', 'ncells+1'],
                                    'logical': False,
                                    'intent': 'in'},
                             'zgrid': {'type': 'float',
                                       'dims': ['nvertlevels+1', 'ncells+1'],
                                       'logical': False,
                                       'intent': 'in'},
                             'scalars': {'type': 'float',
                                         'dims': ['num_scalars', 'nvertlevels', 'ncells+1'],
                                         'logical': False,
                                         'intent': 'in'},
                             'index_qv': {'type': 'int',
                                          'dims': [],
                                          'logical': False,
                                          'intent': 'in'},
                             'index_qc': {'type': 'int',
                                          'dims': [],
                                          'logical': False,
                                          'intent': 'in'},
                             'qtot': {'type': 'float',
                                      'dims': ['nvertlevels', 'ncells+1'],
                                      'logical': False,
                                      'intent': 'in'},
                             'cellstart': {'type': 'int',
                                           'dims': [],
                                           'logical': False,
                                           'intent': 'in'},
                             'cellend': {'type': 'int',
                                         'dims': [],
                                         'logical': False,
                                         'intent': 'in'},
                             'ncells': {'type': 'int',
                                        'dims': [],
                                        'logical': False,
                                        'intent': 'in'},
                             'nvertlevels': {'type': 'int',
                                             'dims': [],
                                             'logical': False,
                                             'intent': 'in'},
                             'maxedges': {'type': 'int',
                                          'dims': [],
                                          'logical': False,
                                          'intent': 'in'},
                             'num_scalars': {'type': 'int',
                                             'dims': [],
                                             'logical': False,
                                             'intent': 'in'}},
                  'scratch': {'theta': ['nvertlevels'],
                              'qvsw': ['nvertlevels'],
                              'temp': ['nvertlevels'],
                              'coefa': ['nvertlevels']},
                  'lines': [464, 575]},
 'u_dissipation_3d': {'args': ['edgestart',
                               'edgeend',
                               'edgesolvestart',
                               'edgesolveend',
                               'vertexstart',
                               'vertexend',
                               'cellstart',
                               'cellend',
                               'ncells',
                               'nedges',
                               'nvertices',
                               'vertexdegree',
                               'cellsonedge',
                               'verticesonedge',
                               'edgesoncell',
                               'edgesonvertex',
                               'nedgesoncell',
                               'edgesoncell_sign',
                               'edgesonvertex_sign',
                               'invareacell',
                               'invareatriangle',
                               'invdvedge',
                               'invdcedge',
                               'angleedge',
                               'dcedge',
                               'dvedge',
                               'meshscalingdel2',
                               'meshscalingdel4',
                               'config_mix_full',
                               'h_mom_eddy_visc4',
                               'v_mom_eddy_visc2',
                               'config_del4u_div_factor',
                               'zgrid',
                               'eddy_visc_horz',
                               'eddy_visc_vert',
                               'zz',
                               'rdzu',
                               'rdzw',
                               'fzm',
                               'fzp',
                               'les_model_opt',
                               'les_surface_opt',
                               'config_surface_drag_coefficient',
                               'delsq_u',
                               'delsq_vorticity',
                               'delsq_divergence',
                               'u',
                               'v',
                               'divergence',
                               'vorticity',
                               'rho_edge',
                               'rho_zz',
                               'u_init',
                               'v_init',
                               'ustm',
                               'tend_u_euler',
                               'nvertlevels',
                               'maxedges',
                               'num_scalars'],
                      'fields': {'edgestart': {'type': 'int',
                                               'dims': [],
                                               'logical': False,
                                               'intent': 'in'},
                                 'edgeend': {'type': 'int',
                                             'dims': [],
                                             'logical': False,
                                             'intent': 'in'},
                                 'edgesolvestart': {'type': 'int',
                                                    'dims': [],
                                                    'logical': False,
                                                    'intent': 'in'},
                                 'edgesolveend': {'type': 'int',
                                                  'dims': [],
                                                  'logical': False,
                                                  'intent': 'in'},
                                 'vertexstart': {'type': 'int',
                                                 'dims': [],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'vertexend': {'type': 'int',
                                               'dims': [],
                                               'logical': False,
                                               'intent': 'in'},
                                 'cellstart': {'type': 'int',
                                               'dims': [],
                                               'logical': False,
                                               'intent': 'in'},
                                 'cellend': {'type': 'int',
                                             'dims': [],
                                             'logical': False,
                                             'intent': 'in'},
                                 'ncells': {'type': 'int',
                                            'dims': [],
                                            'logical': False,
                                            'intent': 'in'},
                                 'nedges': {'type': 'int',
                                            'dims': [],
                                            'logical': False,
                                            'intent': 'in'},
                                 'nvertices': {'type': 'int',
                                               'dims': [],
                                               'logical': False,
                                               'intent': 'in'},
                                 'vertexdegree': {'type': 'int',
                                                  'dims': [],
                                                  'logical': False,
                                                  'intent': 'in'},
                                 'cellsonedge': {'type': 'int',
                                                 'dims': ['2', 'nedges+1'],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'verticesonedge': {'type': 'int',
                                                    'dims': ['2', 'nedges+1'],
                                                    'logical': False,
                                                    'intent': 'in'},
                                 'edgesoncell': {'type': 'int',
                                                 'dims': ['maxedges', 'ncells+1'],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'edgesonvertex': {'type': 'int',
                                                   'dims': ['vertexdegree', 'nvertices+1'],
                                                   'logical': False,
                                                   'intent': 'in'},
                                 'nedgesoncell': {'type': 'int',
                                                  'dims': ['ncells+1'],
                                                  'logical': False,
                                                  'intent': 'in'},
                                 'edgesoncell_sign': {'type': 'float',
                                                      'dims': ['maxedges', 'ncells+1'],
                                                      'logical': False,
                                                      'intent': 'in'},
                                 'edgesonvertex_sign': {'type': 'float',
                                                        'dims': ['vertexdegree',
                                                                 'nvertices+1'],
                                                        'logical': False,
                                                        'intent': 'in'},
                                 'invareacell': {'type': 'float',
                                                 'dims': ['ncells+1'],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'invareatriangle': {'type': 'float',
                                                     'dims': ['nvertices+1'],
                                                     'logical': False,
                                                     'intent': 'in'},
                                 'invdvedge': {'type': 'float',
                                               'dims': ['nedges+1'],
                                               'logical': False,
                                               'intent': 'in'},
                                 'invdcedge': {'type': 'float',
                                               'dims': ['nedges+1'],
                                               'logical': False,
                                               'intent': 'in'},
                                 'angleedge': {'type': 'float',
                                               'dims': ['nedges+1'],
                                               'logical': False,
                                               'intent': 'in'},
                                 'dcedge': {'type': 'float',
                                            'dims': ['nedges+1'],
                                            'logical': False,
                                            'intent': 'in'},
                                 'dvedge': {'type': 'float',
                                            'dims': ['nedges+1'],
                                            'logical': False,
                                            'intent': 'in'},
                                 'meshscalingdel2': {'type': 'float',
                                                     'dims': ['nedges+1'],
                                                     'logical': False,
                                                     'intent': 'in'},
                                 'meshscalingdel4': {'type': 'float',
                                                     'dims': ['nedges+1'],
                                                     'logical': False,
                                                     'intent': 'in'},
                                 'config_mix_full': {'type': 'int',
                                                     'dims': [],
                                                     'logical': True,
                                                     'intent': 'in'},
                                 'h_mom_eddy_visc4': {'type': 'float',
                                                      'dims': [],
                                                      'logical': False,
                                                      'intent': 'in'},
                                 'v_mom_eddy_visc2': {'type': 'float',
                                                      'dims': [],
                                                      'logical': False,
                                                      'intent': 'in'},
                                 'config_del4u_div_factor': {'type': 'float',
                                                             'dims': [],
                                                             'logical': False,
                                                             'intent': 'in'},
                                 'zgrid': {'type': 'float',
                                           'dims': ['nvertlevels+1', 'ncells+1'],
                                           'logical': False,
                                           'intent': 'in'},
                                 'eddy_visc_horz': {'type': 'float',
                                                    'dims': ['nvertlevels', 'ncells+1'],
                                                    'logical': False,
                                                    'intent': 'in'},
                                 'eddy_visc_vert': {'type': 'float',
                                                    'dims': ['nvertlevels', 'ncells+1'],
                                                    'logical': False,
                                                    'intent': 'in'},
                                 'zz': {'type': 'float',
                                        'dims': ['nvertlevels', 'ncells+1'],
                                        'logical': False,
                                        'intent': 'in'},
                                 'rdzu': {'type': 'float',
                                          'dims': ['nvertlevels'],
                                          'logical': False,
                                          'intent': 'in'},
                                 'rdzw': {'type': 'float',
                                          'dims': ['nvertlevels'],
                                          'logical': False,
                                          'intent': 'in'},
                                 'fzm': {'type': 'float',
                                         'dims': ['nvertlevels'],
                                         'logical': False,
                                         'intent': 'in'},
                                 'fzp': {'type': 'float',
                                         'dims': ['nvertlevels'],
                                         'logical': False,
                                         'intent': 'in'},
                                 'les_model_opt': {'type': 'int',
                                                   'dims': [],
                                                   'logical': False,
                                                   'intent': 'in'},
                                 'les_surface_opt': {'type': 'int',
                                                     'dims': [],
                                                     'logical': False,
                                                     'intent': 'in'},
                                 'config_surface_drag_coefficient': {'type': 'float',
                                                                     'dims': [],
                                                                     'logical': False,
                                                                     'intent': 'in'},
                                 'delsq_u': {'type': 'float',
                                             'dims': ['nvertlevels', 'nedges+1'],
                                             'logical': False,
                                             'intent': 'inout'},
                                 'delsq_vorticity': {'type': 'float',
                                                     'dims': ['nvertlevels', 'nvertices+1'],
                                                     'logical': False,
                                                     'intent': 'inout'},
                                 'delsq_divergence': {'type': 'float',
                                                      'dims': ['nvertlevels', 'ncells+1'],
                                                      'logical': False,
                                                      'intent': 'inout'},
                                 'u': {'type': 'float',
                                       'dims': ['nvertlevels', 'nedges+1'],
                                       'logical': False,
                                       'intent': 'in'},
                                 'v': {'type': 'float',
                                       'dims': ['nvertlevels', 'nedges+1'],
                                       'logical': False,
                                       'intent': 'in'},
                                 'divergence': {'type': 'float',
                                                'dims': ['nvertlevels', 'ncells+1'],
                                                'logical': False,
                                                'intent': 'in'},
                                 'vorticity': {'type': 'float',
                                               'dims': ['nvertlevels', 'nvertices+1'],
                                               'logical': False,
                                               'intent': 'in'},
                                 'rho_edge': {'type': 'float',
                                              'dims': ['nvertlevels', 'nedges+1'],
                                              'logical': False,
                                              'intent': 'in'},
                                 'rho_zz': {'type': 'float',
                                            'dims': ['nvertlevels', 'ncells+1'],
                                            'logical': False,
                                            'intent': 'in'},
                                 'u_init': {'type': 'float',
                                            'dims': ['nvertlevels'],
                                            'logical': False,
                                            'intent': 'in'},
                                 'v_init': {'type': 'float',
                                            'dims': ['nvertlevels'],
                                            'logical': False,
                                            'intent': 'in'},
                                 'ustm': {'type': 'float',
                                          'dims': [':'],
                                          'logical': False,
                                          'intent': 'in'},
                                 'tend_u_euler': {'type': 'float',
                                                  'dims': ['nvertlevels', 'nedges+1'],
                                                  'logical': False,
                                                  'intent': 'out'},
                                 'nvertlevels': {'type': 'int',
                                                 'dims': [],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'maxedges': {'type': 'int',
                                              'dims': [],
                                              'logical': False,
                                              'intent': 'in'},
                                 'num_scalars': {'type': 'int',
                                                 'dims': [],
                                                 'logical': False,
                                                 'intent': 'in'}},
                      'scratch': {'u_mix': ['nvertlevels'], 'turb_vflux': ['nvertlevels+1']},
                      'lines': [579, 947]},
 'w_dissipation_3d': {'args': ['cellstart',
                               'cellend',
                               'cellsolvestart',
                               'cellsolveend',
                               'ncells',
                               'nedges',
                               'nedgesoncell',
                               'edgesoncell',
                               'cellsonedge',
                               'edgesoncell_sign',
                               'invareacell',
                               'invdcedge',
                               'dvedge',
                               'meshscalingdel2',
                               'meshscalingdel4',
                               'rdzw',
                               'rdzu',
                               'v_mom_eddy_visc2',
                               'h_mom_eddy_visc4',
                               'delsq_w',
                               'w',
                               'rho_edge',
                               'rho_zz',
                               'divergence',
                               'zz',
                               'eddy_visc_horz',
                               'eddy_visc_vert',
                               'les_model_opt',
                               'les_surface_opt',
                               'tend_w_euler',
                               'nvertlevels',
                               'maxedges',
                               'num_scalars'],
                      'fields': {'cellstart': {'type': 'int',
                                               'dims': [],
                                               'logical': False,
                                               'intent': 'in'},
                                 'cellend': {'type': 'int',
                                             'dims': [],
                                             'logical': False,
                                             'intent': 'in'},
                                 'cellsolvestart': {'type': 'int',
                                                    'dims': [],
                                                    'logical': False,
                                                    'intent': 'in'},
                                 'cellsolveend': {'type': 'int',
                                                  'dims': [],
                                                  'logical': False,
                                                  'intent': 'in'},
                                 'ncells': {'type': 'int',
                                            'dims': [],
                                            'logical': False,
                                            'intent': 'in'},
                                 'nedges': {'type': 'int',
                                            'dims': [],
                                            'logical': False,
                                            'intent': 'in'},
                                 'nedgesoncell': {'type': 'int',
                                                  'dims': ['ncells+1'],
                                                  'logical': False,
                                                  'intent': 'in'},
                                 'edgesoncell': {'type': 'int',
                                                 'dims': ['maxedges', 'ncells+1'],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'cellsonedge': {'type': 'int',
                                                 'dims': ['2', 'nedges+1'],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'edgesoncell_sign': {'type': 'float',
                                                      'dims': ['maxedges', 'ncells+1'],
                                                      'logical': False,
                                                      'intent': 'in'},
                                 'invareacell': {'type': 'float',
                                                 'dims': ['ncells+1'],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'invdcedge': {'type': 'float',
                                               'dims': ['nedges+1'],
                                               'logical': False,
                                               'intent': 'in'},
                                 'dvedge': {'type': 'float',
                                            'dims': ['nedges+1'],
                                            'logical': False,
                                            'intent': 'in'},
                                 'meshscalingdel2': {'type': 'float',
                                                     'dims': ['nedges+1'],
                                                     'logical': False,
                                                     'intent': 'in'},
                                 'meshscalingdel4': {'type': 'float',
                                                     'dims': ['nedges+1'],
                                                     'logical': False,
                                                     'intent': 'in'},
                                 'rdzw': {'type': 'float',
                                          'dims': ['nvertlevels'],
                                          'logical': False,
                                          'intent': 'in'},
                                 'rdzu': {'type': 'float',
                                          'dims': ['nvertlevels'],
                                          'logical': False,
                                          'intent': 'in'},
                                 'v_mom_eddy_visc2': {'type': 'float',
                                                      'dims': [],
                                                      'logical': False,
                                                      'intent': 'in'},
                                 'h_mom_eddy_visc4': {'type': 'float',
                                                      'dims': [],
                                                      'logical': False,
                                                      'intent': 'in'},
                                 'delsq_w': {'type': 'float',
                                             'dims': ['nvertlevels', 'ncells+1'],
                                             'logical': False,
                                             'intent': 'inout'},
                                 'w': {'type': 'float',
                                       'dims': ['nvertlevels+1', 'ncells+1'],
                                       'logical': False,
                                       'intent': 'in'},
                                 'rho_edge': {'type': 'float',
                                              'dims': ['nvertlevels', 'nedges+1'],
                                              'logical': False,
                                              'intent': 'in'},
                                 'rho_zz': {'type': 'float',
                                            'dims': ['nvertlevels', 'ncells+1'],
                                            'logical': False,
                                            'intent': 'in'},
                                 'divergence': {'type': 'float',
                                                'dims': ['nvertlevels', 'ncells+1'],
                                                'logical': False,
                                                'intent': 'in'},
                                 'zz': {'type': 'float',
                                        'dims': ['nvertlevels', 'ncells+1'],
                                        'logical': False,
                                        'intent': 'in'},
                                 'eddy_visc_horz': {'type': 'float',
                                                    'dims': ['nvertlevels', 'ncells+1'],
                                                    'logical': False,
                                                    'intent': 'in'},
                                 'eddy_visc_vert': {'type': 'float',
                                                    'dims': ['nvertlevels', 'ncells+1'],
                                                    'logical': False,
                                                    'intent': 'in'},
                                 'les_model_opt': {'type': 'int',
                                                   'dims': [],
                                                   'logical': False,
                                                   'intent': 'in'},
                                 'les_surface_opt': {'type': 'int',
                                                     'dims': [],
                                                     'logical': False,
                                                     'intent': 'in'},
                                 'tend_w_euler': {'type': 'float',
                                                  'dims': ['nvertlevels+1', 'ncells+1'],
                                                  'logical': False,
                                                  'intent': 'inout'},
                                 'nvertlevels': {'type': 'int',
                                                 'dims': [],
                                                 'logical': False,
                                                 'intent': 'in'},
                                 'maxedges': {'type': 'int',
                                              'dims': [],
                                              'logical': False,
                                              'intent': 'in'},
                                 'num_scalars': {'type': 'int',
                                                 'dims': [],
                                                 'logical': False,
                                                 'intent': 'in'}},
                      'scratch': {'turb_vflux': ['nvertlevels+1']},
                      'lines': [951, 1153]},
 'scalar_dissipation_3d_les': {'args': ['cellstart',
                                        'cellend',
                                        'cellsolvestart',
                                        'cellsolveend',
                                        'ncells',
                                        'nedges',
                                        'nedgesoncell',
                                        'edgesoncell',
                                        'cellsonedge',
                                        'edgesoncell_sign',
                                        'invareacell',
                                        'invdcedge',
                                        'dvedge',
                                        'meshscalingdel2',
                                        'meshscalingdel4',
                                        'config_mix_full',
                                        't_init',
                                        'zgrid',
                                        'rdzw',
                                        'rdzu',
                                        'fzm',
                                        'fzp',
                                        'v_theta_eddy_visc2',
                                        'h_theta_eddy_visc4',
                                        'prandtl_inv',
                                        'prandtl_3d_inv',
                                        'delsq_theta',
                                        'theta_m',
                                        'rho_edge',
                                        'rho_zz',
                                        'zz',
                                        'eddy_visc_horz',
                                        'eddy_visc_vert',
                                        'bv_freq2',
                                        'config_len_disp',
                                        'scalars',
                                        'tend_scalars',
                                        'index_tke',
                                        'index_qv',
                                        'num_scalars_dummy',
                                        'mix_scalars',
                                        'les_model_opt',
                                        'les_surface_opt',
                                        'dt',
                                        'config_surface_heat_flux',
                                        'config_surface_moisture_flux',
                                        'ureconstructzonal',
                                        'ureconstructmeridional',
                                        'hfx',
                                        'qfx',
                                        'tend_theta_euler',
                                        'dynamics_substep',
                                        'nvertlevels',
                                        'maxedges',
                                        'num_scalars'],
                               'fields': {'cellstart': {'type': 'int',
                                                        'dims': [],
                                                        'logical': False,
                                                        'intent': 'in'},
                                          'cellend': {'type': 'int',
                                                      'dims': [],
                                                      'logical': False,
                                                      'intent': 'in'},
                                          'cellsolvestart': {'type': 'int',
                                                             'dims': [],
                                                             'logical': False,
                                                             'intent': 'in'},
                                          'cellsolveend': {'type': 'int',
                                                           'dims': [],
                                                           'logical': False,
                                                           'intent': 'in'},
                                          'ncells': {'type': 'int',
                                                     'dims': [],
                                                     'logical': False,
                                                     'intent': 'in'},
                                          'nedges': {'type': 'int',
                                                     'dims': [],
                                                     'logical': False,
                                                     'intent': 'in'},
                                          'nedgesoncell': {'type': 'int',
                                                           'dims': ['ncells+1'],
                                                           'logical': False,
                                                           'intent': 'in'},
                                          'edgesoncell': {'type': 'int',
                                                          'dims': ['maxedges', 'ncells+1'],
                                                          'logical': False,
                                                          'intent': 'in'},
                                          'cellsonedge': {'type': 'int',
                                                          'dims': ['2', 'nedges+1'],
                                                          'logical': False,
                                                          'intent': 'in'},
                                          'edgesoncell_sign': {'type': 'float',
                                                               'dims': ['maxedges',
                                                                        'ncells+1'],
                                                               'logical': False,
                                                               'intent': 'in'},
                                          'invareacell': {'type': 'float',
                                                          'dims': ['ncells+1'],
                                                          'logical': False,
                                                          'intent': 'in'},
                                          'invdcedge': {'type': 'float',
                                                        'dims': ['nedges+1'],
                                                        'logical': False,
                                                        'intent': 'in'},
                                          'dvedge': {'type': 'float',
                                                     'dims': ['nedges+1'],
                                                     'logical': False,
                                                     'intent': 'in'},
                                          'meshscalingdel2': {'type': 'float',
                                                              'dims': ['nedges+1'],
                                                              'logical': False,
                                                              'intent': 'in'},
                                          'meshscalingdel4': {'type': 'float',
                                                              'dims': ['nedges+1'],
                                                              'logical': False,
                                                              'intent': 'in'},
                                          'config_mix_full': {'type': 'int',
                                                              'dims': [],
                                                              'logical': True,
                                                              'intent': 'in'},
                                          't_init': {'type': 'float',
                                                     'dims': ['nvertlevels', 'ncells+1'],
                                                     'logical': False,
                                                     'intent': 'in'},
                                          'zgrid': {'type': 'float',
                                                    'dims': ['nvertlevels+1', 'ncells+1'],
                                                    'logical': False,
                                                    'intent': 'in'},
                                          'rdzw': {'type': 'float',
                                                   'dims': ['nvertlevels'],
                                                   'logical': False,
                                                   'intent': 'in'},
                                          'rdzu': {'type': 'float',
                                                   'dims': ['nvertlevels'],
                                                   'logical': False,
                                                   'intent': 'in'},
                                          'fzm': {'type': 'float',
                                                  'dims': ['nvertlevels'],
                                                  'logical': False,
                                                  'intent': 'in'},
                                          'fzp': {'type': 'float',
                                                  'dims': ['nvertlevels'],
                                                  'logical': False,
                                                  'intent': 'in'},
                                          'v_theta_eddy_visc2': {'type': 'float',
                                                                 'dims': [],
                                                                 'logical': False,
                                                                 'intent': 'in'},
                                          'h_theta_eddy_visc4': {'type': 'float',
                                                                 'dims': [],
                                                                 'logical': False,
                                                                 'intent': 'in'},
                                          'prandtl_inv': {'type': 'float',
                                                          'dims': [],
                                                          'logical': False,
                                                          'intent': 'in'},
                                          'prandtl_3d_inv': {'type': 'float',
                                                             'dims': ['nvertlevels',
                                                                      'ncells+1'],
                                                             'logical': False,
                                                             'intent': 'in'},
                                          'delsq_theta': {'type': 'float',
                                                          'dims': ['nvertlevels', 'ncells+1'],
                                                          'logical': False,
                                                          'intent': 'inout'},
                                          'theta_m': {'type': 'float',
                                                      'dims': ['nvertlevels', 'ncells+1'],
                                                      'logical': False,
                                                      'intent': 'in'},
                                          'rho_edge': {'type': 'float',
                                                       'dims': ['nvertlevels', 'nedges+1'],
                                                       'logical': False,
                                                       'intent': 'in'},
                                          'rho_zz': {'type': 'float',
                                                     'dims': ['nvertlevels', 'ncells+1'],
                                                     'logical': False,
                                                     'intent': 'in'},
                                          'zz': {'type': 'float',
                                                 'dims': ['nvertlevels', 'ncells+1'],
                                                 'logical': False,
                                                 'intent': 'in'},
                                          'eddy_visc_horz': {'type': 'float',
                                                             'dims': ['nvertlevels',
                                                                      'ncells+1'],
                                                             'logical': False,
                                                             'intent': 'in'},
                                          'eddy_visc_vert': {'type': 'float',
                                                             'dims': ['nvertlevels',
                                                                      'ncells+1'],
                                                             'logical': False,
                                                             'intent': 'in'},
                                          'bv_freq2': {'type': 'float',
                                                       'dims': ['nvertlevels', 'ncells+1'],
                                                       'logical': False,
                                                       'intent': 'in'},
                                          'config_len_disp': {'type': 'float',
                                                              'dims': [],
                                                              'logical': False,
                                                              'intent': 'in'},
                                          'scalars': {'type': 'float',
                                                      'dims': ['num_scalars',
                                                               'nvertlevels',
                                                               'ncells+1'],
                                                      'logical': False,
                                                      'intent': 'in'},
                                          'tend_scalars': {'type': 'float',
                                                           'dims': ['num_scalars',
                                                                    'nvertlevels',
                                                                    'ncells+1'],
                                                           'logical': False,
                                                           'intent': 'inout'},
                                          'index_tke': {'type': 'int',
                                                        'dims': [],
                                                        'logical': False,
                                                        'intent': 'in'},
                                          'index_qv': {'type': 'int',
                                                       'dims': [],
                                                       'logical': False,
                                                       'intent': 'in'},
                                          'num_scalars_dummy': {'type': 'int',
                                                                'dims': [],
                                                                'logical': False,
                                                                'intent': 'in'},
                                          'mix_scalars': {'type': 'int',
                                                          'dims': [],
                                                          'logical': True,
                                                          'intent': 'in'},
                                          'les_model_opt': {'type': 'int',
                                                            'dims': [],
                                                            'logical': False,
                                                            'intent': 'in'},
                                          'les_surface_opt': {'type': 'int',
                                                              'dims': [],
                                                              'logical': False,
                                                              'intent': 'in'},
                                          'dt': {'type': 'float',
                                                 'dims': [],
                                                 'logical': False,
                                                 'intent': 'in'},
                                          'config_surface_heat_flux': {'type': 'float',
                                                                       'dims': [],
                                                                       'logical': False,
                                                                       'intent': 'in'},
                                          'config_surface_moisture_flux': {'type': 'float',
                                                                           'dims': [],
                                                                           'logical': False,
                                                                           'intent': 'in'},
                                          'ureconstructzonal': {'type': 'float',
                                                                'dims': ['nvertlevels',
                                                                         'ncells+1'],
                                                                'logical': False,
                                                                'intent': 'in'},
                                          'ureconstructmeridional': {'type': 'float',
                                                                     'dims': ['nvertlevels',
                                                                              'ncells+1'],
                                                                     'logical': False,
                                                                     'intent': 'in'},
                                          'hfx': {'type': 'float',
                                                  'dims': [':'],
                                                  'logical': False,
                                                  'intent': 'in'},
                                          'qfx': {'type': 'float',
                                                  'dims': [':'],
                                                  'logical': False,
                                                  'intent': 'in'},
                                          'tend_theta_euler': {'type': 'float',
                                                               'dims': ['nvertlevels',
                                                                        'ncells+1'],
                                                               'logical': False,
                                                               'intent': 'inout'},
                                          'dynamics_substep': {'type': 'int',
                                                               'dims': [],
                                                               'logical': False,
                                                               'intent': 'in'},
                                          'nvertlevels': {'type': 'int',
                                                          'dims': [],
                                                          'logical': False,
                                                          'intent': 'in'},
                                          'maxedges': {'type': 'int',
                                                       'dims': [],
                                                       'logical': False,
                                                       'intent': 'in'},
                                          'num_scalars': {'type': 'int',
                                                          'dims': [],
                                                          'logical': False,
                                                          'intent': 'in'}},
                               'scratch': {'turb_vflux': ['nvertlevels+1'],
                                           'prandtl_1d_inverse': ['nvertlevels+1'],
                                           'turb_vflux_scalars': ['num_scalars',
                                                                  'nvertlevels+1'],
                                           'rho_k_at_w': ['nvertlevels'],
                                           'zz_at_w': ['nvertlevels']},
                               'lines': [1157, 1606]}}

PHASES = {'les_models': [{'kernel': 'les_models_setup', 'bounds': ['1', '1']},
                {'kernel': 'les_models_phase_0', 'bounds': ['cellstart', 'cellend']}],
 'calculate_n2': [{'kernel': 'calculate_n2_phase_0', 'bounds': ['cellstart', 'cellend']}],
 'u_dissipation_3d': [{'kernel': 'u_dissipation_3d_phase_0',
                       'bounds': ['edgestart', 'edgeend']},
                      {'kernel': 'u_dissipation_3d_phase_1',
                       'bounds': ['vertexstart', 'vertexend']},
                      {'kernel': 'u_dissipation_3d_phase_2',
                       'bounds': ['cellstart', 'cellend']},
                      {'kernel': 'u_dissipation_3d_phase_3',
                       'bounds': ['edgesolvestart', 'edgesolveend']},
                      {'kernel': 'u_dissipation_3d_phase_4',
                       'bounds': ['edgesolvestart', 'edgesolveend']},
                      {'kernel': 'u_dissipation_3d_phase_5',
                       'bounds': ['edgesolvestart', 'edgesolveend']},
                      {'kernel': 'u_dissipation_3d_phase_6',
                       'bounds': ['edgesolvestart', 'edgesolveend']}],
 'w_dissipation_3d': [{'kernel': 'w_dissipation_3d_phase_0',
                       'bounds': ['cellstart', 'cellend']},
                      {'kernel': 'w_dissipation_3d_phase_1',
                       'bounds': ['cellsolvestart', 'cellsolveend']},
                      {'kernel': 'w_dissipation_3d_phase_2',
                       'bounds': ['cellsolvestart', 'cellsolveend']},
                      {'kernel': 'w_dissipation_3d_phase_3',
                       'bounds': ['cellsolvestart', 'cellsolveend']}],
 'scalar_dissipation_3d_les': [{'kernel': 'scalar_dissipation_3d_les_phase_0',
                                'bounds': ['cellstart', 'cellend']},
                               {'kernel': 'scalar_dissipation_3d_les_phase_1',
                                'bounds': ['cellsolvestart', 'cellsolveend']},
                               {'repeat': 'iscalar',
                                'bounds': ['1', 'num_scalars'],
                                'body': [{'kernel': 'scalar_dissipation_3d_les_phase_2',
                                          'bounds': ['cellstart', 'cellend']},
                                         {'kernel': 'scalar_dissipation_3d_les_phase_3',
                                          'bounds': ['cellsolvestart', 'cellsolveend']}]},
                               {'kernel': 'scalar_dissipation_3d_les_phase_4',
                                'bounds': ['cellsolvestart', 'cellsolveend']},
                               {'kernel': 'scalar_dissipation_3d_les_phase_5',
                                'bounds': ['cellsolvestart', 'cellsolveend']},
                               {'kernel': 'scalar_dissipation_3d_les_phase_6',
                                'bounds': ['cellsolvestart', 'cellsolveend']}]}

CUDA_SOURCE = r"""
// MPAS v8.4.2 mpas_atm_dissipation_models.F:208-460
extern "C" __global__ void les_models_phase_0(int les_model_opt, int les_surface_opt, int dynamics_substep, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_u, float *p_v, float *p_ucell, float *p_vcell, float *p_w, float *p_c_s, float *p_bv_freq2, float *p_zgrid, float *p_config_len_disp, float *p_deformation_coef_c2, float *p_deformation_coef_s2, float *p_deformation_coef_cs, float *p_deformation_coef_c, float *p_deformation_coef_s, float *p_prandtl_3d_inv, float *p_invdt, float *p_h_mom_eddy_visc4, float *p_config_visc4_2dsmag, float *p_h_theta_eddy_visc4, float *p_scalars, float *p_tend_scalars, int index_tke, float *p_rho_zz, float *p_meshscalingdel2, int cellstart, int cellend, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, int ncells, int nedges, int nvertlevels, int maxedges, int num_scalars, float *p_d_11, float *p_d_22, float *p_d_33, float *p_d_12, float *p_d_13, float *p_d_23, float *p_dudx, float *p_dudy, float *p_dvdx, float *p_dvdy, float *p_dwdx, float *p_dwdy, float *p_dudz, float *p_dvdz, float *p_dwdz, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define ucell(x0,x1) p_ucell[(x0-1)+(nvertlevels)*((x1-1))]
#define vcell(x0,x1) p_vcell[(x0-1)+(nvertlevels)*((x1-1))]
#define w(x0,x1) p_w[(x0-1)+(nvertlevels+1)*((x1-1))]
#define c_s (*p_c_s)
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define deformation_coef_c2(x0,x1) p_deformation_coef_c2[(x0-1)+(maxedges)*((x1-1))]
#define deformation_coef_s2(x0,x1) p_deformation_coef_s2[(x0-1)+(maxedges)*((x1-1))]
#define deformation_coef_cs(x0,x1) p_deformation_coef_cs[(x0-1)+(maxedges)*((x1-1))]
#define deformation_coef_c(x0,x1) p_deformation_coef_c[(x0-1)+(maxedges)*((x1-1))]
#define deformation_coef_s(x0,x1) p_deformation_coef_s[(x0-1)+(maxedges)*((x1-1))]
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define invdt (*p_invdt)
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define config_visc4_2dsmag (*p_config_visc4_2dsmag)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define d_11(x0) p_d_11[tid*((nvertlevels))+(x0-1)]
#define d_22(x0) p_d_22[tid*((nvertlevels))+(x0-1)]
#define d_33(x0) p_d_33[tid*((nvertlevels))+(x0-1)]
#define d_12(x0) p_d_12[tid*((nvertlevels))+(x0-1)]
#define d_13(x0) p_d_13[tid*((nvertlevels))+(x0-1)]
#define d_23(x0) p_d_23[tid*((nvertlevels))+(x0-1)]
#define dudx(x0) p_dudx[tid*((nvertlevels))+(x0-1)]
#define dudy(x0) p_dudy[tid*((nvertlevels))+(x0-1)]
#define dvdx(x0) p_dvdx[tid*((nvertlevels))+(x0-1)]
#define dvdy(x0) p_dvdy[tid*((nvertlevels))+(x0-1)]
#define dwdx(x0) p_dwdx[tid*((nvertlevels+1))+(x0-1)]
#define dwdy(x0) p_dwdy[tid*((nvertlevels+1))+(x0-1)]
#define dudz(x0) p_dudz[tid*((nvertlevels))+(x0-1)]
#define dvdz(x0) p_dvdz[tid*((nvertlevels))+(x0-1)]
#define dwdz(x0) p_dwdz[tid*((nvertlevels))+(x0-1)]
int icell;
int iedge;
int k;
int ie;
int cell1;
int cell2;
float rdz;
float def2;
float pr_inv;
float wk;
float shear_production;
float buoyancy;
float dissipation;
float delta_z;
float delta_s;
float bv;
float tke_length;
float diss_length;
float l_horizontal;
float l_vertical;
float c_dissipation;
float prandtl_horizontal_inv;
float eddy_visc_h;
float eddy_visc_v;
int test_tke;
icell=tid+cellstart;
if(icell>cellend || !(1)) return;
pr_inv = 0x1.0000000000000p+0f;
for(k=1;k<=nvertlevels;++k) {
dudx(k) = 0x0.0p+0f;
dudy(k) = 0x0.0p+0f;
dvdx(k) = 0x0.0p+0f;
dvdy(k) = 0x0.0p+0f;
dudz(k) = 0x0.0p+0f;
dvdz(k) = 0x0.0p+0f;
dwdz(k) = 0x0.0p+0f;
}
for(k=1;k<=(nvertlevels+1);++k) {
dwdx(k) = 0x0.0p+0f;
dwdy(k) = 0x0.0p+0f;
}
for(iedge=1;iedge<=nedgesoncell(icell);++iedge) {
ie = edgesoncell(iedge,icell);
cell1 = cellsonedge(1,ie);
cell2 = cellsonedge(2,ie);
for(k=1;k<=nvertlevels;++k) {
dudx(k) = __fsub_rn(__fadd_rn(dudx(k),__fmul_rn(deformation_coef_c2(iedge,icell),u(k,ie))),__fmul_rn(deformation_coef_cs(iedge,icell),v(k,ie)));
dudy(k) = __fsub_rn(__fadd_rn(dudy(k),__fmul_rn(deformation_coef_cs(iedge,icell),u(k,ie))),__fmul_rn(deformation_coef_s2(iedge,icell),v(k,ie)));
dvdx(k) = __fadd_rn(__fadd_rn(dvdx(k),__fmul_rn(deformation_coef_cs(iedge,icell),u(k,ie))),__fmul_rn(deformation_coef_c2(iedge,icell),v(k,ie)));
dvdy(k) = __fadd_rn(__fadd_rn(dvdy(k),__fmul_rn(deformation_coef_s2(iedge,icell),u(k,ie))),__fmul_rn(deformation_coef_cs(iedge,icell),v(k,ie)));
}
for(k=1;k<=(nvertlevels+1);++k) {
wk = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(w(k,cell1),w(k,cell2)));
dwdx(k) = __fadd_rn(dwdx(k),__fmul_rn(deformation_coef_c(iedge,icell),wk));
dwdy(k) = __fadd_rn(dwdy(k),__fmul_rn(deformation_coef_s(iedge,icell),wk));
}
}
for(k=1;k<=nvertlevels;++k) {
rdz = __fdiv_rn(0x1.0000000000000p+0f,__fsub_rn(zgrid((k+1),icell),zgrid(k,icell)));
dwdz(k) = __fmul_rn(__fsub_rn(w((k+1),icell),w(k,icell)),rdz);
}
for(k=2;k<=(nvertlevels-1);++k) {
rdz = __fdiv_rn(0x1.0000000000000p+0f,__fsub_rn(__fsub_rn(__fadd_rn(zgrid((k+2),icell),zgrid((k+1),icell)),zgrid(k,icell)),zgrid((k-1),icell)));
dudz(k) = __fmul_rn(__fsub_rn(ucell((k+1),icell),ucell((k-1),icell)),rdz);
dvdz(k) = __fmul_rn(__fsub_rn(vcell((k+1),icell),vcell((k-1),icell)),rdz);
}
k = 1;
rdz = __fdiv_rn(0x1.0000000000000p+0f,__fsub_rn(zgrid((k+1),icell),zgrid(k,icell)));
dudz(k) = __fmul_rn(__fsub_rn(ucell((k+1),icell),ucell(k,icell)),rdz);
dvdz(k) = __fmul_rn(__fsub_rn(vcell((k+1),icell),vcell(k,icell)),rdz);
k = (nvertlevels-1);
rdz = __fdiv_rn(0x1.0000000000000p+0f,__fsub_rn(zgrid((k+1),icell),zgrid(k,icell)));
dudz((k+1)) = __fmul_rn(__fsub_rn(ucell((k+1),icell),ucell(k,icell)),rdz);
dvdz((k+1)) = __fmul_rn(__fsub_rn(vcell((k+1),icell),vcell(k,icell)),rdz);
for(k=1;k<=nvertlevels;++k) {
d_11(k) = __fmul_rn(0x1.0000000000000p+1f,dudx(k));
d_22(k) = __fmul_rn(0x1.0000000000000p+1f,dvdy(k));
d_33(k) = __fmul_rn(0x1.0000000000000p+1f,dwdz(k));
d_12(k) = __fadd_rn(dudy(k),dvdx(k));
d_13(k) = __fadd_rn(dwdx(k),dudz(k));
d_23(k) = __fadd_rn(dwdy(k),dvdz(k));
}
if((les_model_opt==1)) {
for(k=1;k<=nvertlevels;++k) {
def2 = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(__fadd_rn(__fmul_rn(d_11(k),d_11(k)),__fmul_rn(d_22(k),d_22(k))),__fmul_rn(d_33(k),d_33(k)))),__fmul_rn(d_12(k),d_12(k))),__fmul_rn(d_13(k),d_13(k))),__fmul_rn(d_23(k),d_23(k)));
eddy_visc_horz(k,icell) = __fmul_rn(__fmul_rn(__fmul_rn(c_s,config_len_disp),__fmul_rn(c_s,config_len_disp)),__fsqrt_rn(fmaxf(0x0.0p+0f,__fsub_rn(def2,__fmul_rn(pr_inv,bv_freq2(k,icell))))));
eddy_visc_horz(k,icell) = fminf(eddy_visc_horz(k,icell),__fmul_rn(__fmul_rn(0x1.47ae140000000p-7f,__fmul_rn(config_len_disp,config_len_disp)),invdt));
delta_z = __fsub_rn(zgrid((k+1),icell),zgrid(k,icell));
eddy_visc_vert(k,icell) = __fmul_rn(__fmul_rn(__fmul_rn(c_s,delta_z),__fmul_rn(c_s,delta_z)),__fsqrt_rn(fmaxf(0x0.0p+0f,__fsub_rn(def2,__fmul_rn(pr_inv,bv_freq2(k,icell))))));
}
} else if((les_model_opt==2)) {
for(k=1;k<=nvertlevels;++k) {
scalars(index_tke,k,icell) = fmaxf(0x0.0p+0f,scalars(index_tke,k,icell));
}
for(k=1;k<=nvertlevels;++k) {
delta_z = __fsub_rn(zgrid((k+1),icell),zgrid(k,icell));
delta_s = gfk_pow(__fmul_rn(__fmul_rn(config_len_disp,config_len_disp),delta_z),0x1.5555560000000p-2f);
bv = fmaxf(__fsqrt_rn(fabsf(bv_freq2(k,icell))),0x1.0c6f7a0000000p-20f);
tke_length = delta_s;
if((bv_freq2(k,icell)>0x1.0c6f7a0000000p-20f)) {
tke_length = __fdiv_rn(__fmul_rn(0x1.851eb80000000p-1f,__fsqrt_rn(scalars(index_tke,k,icell))),bv);
}
tke_length = fminf(tke_length,delta_z);
diss_length = fminf(delta_s,fmaxf(tke_length,__fmul_rn(0x1.47ae140000000p-7f,delta_s)));
if((bv_freq2(k,icell)<=0)) {
diss_length = delta_s;
}
l_horizontal = config_len_disp;
l_vertical = fminf(delta_z,tke_length);
if((bv_freq2(k,icell)<=0)) {
diss_length = delta_z;
}
eddy_visc_h = __fmul_rn(__fmul_rn(0x1.0000000000000p-2f,l_horizontal),__fsqrt_rn(scalars(index_tke,k,icell)));
eddy_visc_h = fminf(eddy_visc_h,__fmul_rn(__fmul_rn(0x1.47ae140000000p-7f,__fmul_rn(config_len_disp,config_len_disp)),invdt));
#ifndef MPAS_LES_NO_VISCOSITY_FLOOR
eddy_visc_h = fmaxf(0x1.47ae140000000p-7f,eddy_visc_h);
#endif
eddy_visc_v = __fmul_rn(__fmul_rn(0x1.0000000000000p-2f,l_vertical),__fsqrt_rn(scalars(index_tke,k,icell)));
eddy_visc_v = fminf(eddy_visc_v,__fmul_rn(__fmul_rn(0x1.47ae140000000p-7f,__fmul_rn(delta_z,delta_z)),invdt));
#ifndef MPAS_LES_NO_VISCOSITY_FLOOR
eddy_visc_v = fmaxf(0x1.47ae140000000p-7f,eddy_visc_v);
#endif
eddy_visc_horz(k,icell) = eddy_visc_h;
eddy_visc_vert(k,icell) = eddy_visc_v;
shear_production = __fadd_rn(__fmul_rn(eddy_visc_h,__fadd_rn(__fadd_rn(__fmul_rn(d_11(k),d_11(k)),__fmul_rn(d_22(k),d_22(k))),__fmul_rn(d_12(k),d_12(k)))),__fmul_rn(eddy_visc_v,__fadd_rn(__fadd_rn(__fmul_rn(d_33(k),d_33(k)),__fmul_rn(d_13(k),d_13(k))),__fmul_rn(d_23(k),d_23(k)))));
buoyancy = __fmul_rn((-eddy_visc_v),bv_freq2(k,icell));
c_dissipation = __fadd_rn(0x1.e666660000000p-2f,__fdiv_rn(__fmul_rn(fmaxf(0x0.0p+0f,0x1.d1eb860000000p-2f),diss_length),delta_s));
dissipation = __fdiv_rn(__fmul_rn((-c_dissipation),gfk_pow(scalars(index_tke,k,icell),0x1.8000000000000p+0f)),diss_length);
prandtl_horizontal_inv = 0x1.8000000000000p+1f;
prandtl_3d_inv(k,icell) = __fadd_rn(0x1.0000000000000p+0f,__fdiv_rn(__fmul_rn(0x1.0000000000000p+1f,l_vertical),delta_z));
if((dynamics_substep==1)) {
tend_scalars(index_tke,k,icell) = __fmul_rn(rho_zz(k,icell),__fadd_rn(__fadd_rn(shear_production,buoyancy),dissipation));
}
}
} else if(1) {
}
}
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef u
#undef v
#undef ucell
#undef vcell
#undef w
#undef c_s
#undef bv_freq2
#undef zgrid
#undef config_len_disp
#undef deformation_coef_c2
#undef deformation_coef_s2
#undef deformation_coef_cs
#undef deformation_coef_c
#undef deformation_coef_s
#undef prandtl_3d_inv
#undef invdt
#undef h_mom_eddy_visc4
#undef config_visc4_2dsmag
#undef h_theta_eddy_visc4
#undef scalars
#undef tend_scalars
#undef rho_zz
#undef meshscalingdel2
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef d_11
#undef d_22
#undef d_33
#undef d_12
#undef d_13
#undef d_23
#undef dudx
#undef dudy
#undef dvdx
#undef dvdy
#undef dwdx
#undef dwdy
#undef dudz
#undef dvdz
#undef dwdz

extern "C" __global__ void les_models_setup(int les_model_opt, int les_surface_opt, int dynamics_substep, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_u, float *p_v, float *p_ucell, float *p_vcell, float *p_w, float *p_c_s, float *p_bv_freq2, float *p_zgrid, float *p_config_len_disp, float *p_deformation_coef_c2, float *p_deformation_coef_s2, float *p_deformation_coef_cs, float *p_deformation_coef_c, float *p_deformation_coef_s, float *p_prandtl_3d_inv, float *p_invdt, float *p_h_mom_eddy_visc4, float *p_config_visc4_2dsmag, float *p_h_theta_eddy_visc4, float *p_scalars, float *p_tend_scalars, int index_tke, float *p_rho_zz, float *p_meshscalingdel2, int cellstart, int cellend, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, int ncells, int nedges, int nvertlevels, int maxedges, int num_scalars, float *p_d_11, float *p_d_22, float *p_d_33, float *p_d_12, float *p_d_13, float *p_d_23, float *p_dudx, float *p_dudy, float *p_dvdx, float *p_dvdy, float *p_dwdx, float *p_dwdy, float *p_dudz, float *p_dvdz, float *p_dwdz, int iscalar_arg) {
if(blockIdx.x || threadIdx.x)return;
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define ucell(x0,x1) p_ucell[(x0-1)+(nvertlevels)*((x1-1))]
#define vcell(x0,x1) p_vcell[(x0-1)+(nvertlevels)*((x1-1))]
#define w(x0,x1) p_w[(x0-1)+(nvertlevels+1)*((x1-1))]
#define c_s (*p_c_s)
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define deformation_coef_c2(x0,x1) p_deformation_coef_c2[(x0-1)+(maxedges)*((x1-1))]
#define deformation_coef_s2(x0,x1) p_deformation_coef_s2[(x0-1)+(maxedges)*((x1-1))]
#define deformation_coef_cs(x0,x1) p_deformation_coef_cs[(x0-1)+(maxedges)*((x1-1))]
#define deformation_coef_c(x0,x1) p_deformation_coef_c[(x0-1)+(maxedges)*((x1-1))]
#define deformation_coef_s(x0,x1) p_deformation_coef_s[(x0-1)+(maxedges)*((x1-1))]
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define invdt (*p_invdt)
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define config_visc4_2dsmag (*p_config_visc4_2dsmag)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define d_11(x0) p_d_11[(x0-1)]
#define d_22(x0) p_d_22[(x0-1)]
#define d_33(x0) p_d_33[(x0-1)]
#define d_12(x0) p_d_12[(x0-1)]
#define d_13(x0) p_d_13[(x0-1)]
#define d_23(x0) p_d_23[(x0-1)]
#define dudx(x0) p_dudx[(x0-1)]
#define dudy(x0) p_dudy[(x0-1)]
#define dvdx(x0) p_dvdx[(x0-1)]
#define dvdy(x0) p_dvdy[(x0-1)]
#define dwdx(x0) p_dwdx[(x0-1)]
#define dwdy(x0) p_dwdy[(x0-1)]
#define dudz(x0) p_dudz[(x0-1)]
#define dvdz(x0) p_dvdz[(x0-1)]
#define dwdz(x0) p_dwdz[(x0-1)]
int icell;
int iedge;
int k;
int ie;
int cell1;
int cell2;
float rdz;
float def2;
float pr_inv;
float wk;
float shear_production;
float buoyancy;
float dissipation;
float delta_z;
float delta_s;
float bv;
float tke_length;
float diss_length;
float l_horizontal;
float l_vertical;
float c_dissipation;
float prandtl_horizontal_inv;
float eddy_visc_h;
float eddy_visc_v;
int test_tke;
h_mom_eddy_visc4 = __fmul_rn(config_visc4_2dsmag,__fmul_rn(__fmul_rn(config_len_disp,config_len_disp),config_len_disp));
h_theta_eddy_visc4 = h_mom_eddy_visc4;
}
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef u
#undef v
#undef ucell
#undef vcell
#undef w
#undef c_s
#undef bv_freq2
#undef zgrid
#undef config_len_disp
#undef deformation_coef_c2
#undef deformation_coef_s2
#undef deformation_coef_cs
#undef deformation_coef_c
#undef deformation_coef_s
#undef prandtl_3d_inv
#undef invdt
#undef h_mom_eddy_visc4
#undef config_visc4_2dsmag
#undef h_theta_eddy_visc4
#undef scalars
#undef tend_scalars
#undef rho_zz
#undef meshscalingdel2
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef d_11
#undef d_22
#undef d_33
#undef d_12
#undef d_13
#undef d_23
#undef dudx
#undef dudy
#undef dvdx
#undef dvdy
#undef dwdx
#undef dwdy
#undef dudz
#undef dvdz
#undef dwdz

// MPAS v8.4.2 mpas_atm_dissipation_models.F:464-575
extern "C" __global__ void calculate_n2_phase_0(float *p_bn2, float *p_theta_m, float *p_exner, float *p_pressure_b, float *p_pp, float *p_zgrid, float *p_scalars, int index_qv, int index_qc, float *p_qtot, int cellstart, int cellend, int ncells, int nvertlevels, int maxedges, int num_scalars, float *p_theta, float *p_qvsw, float *p_temp, float *p_coefa, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define bn2(x0,x1) p_bn2[(x0-1)+(nvertlevels)*((x1-1))]
#define theta_m(x0,x1) p_theta_m[(x0-1)+(nvertlevels)*((x1-1))]
#define exner(x0,x1) p_exner[(x0-1)+(nvertlevels)*((x1-1))]
#define pressure_b(x0,x1) p_pressure_b[(x0-1)+(nvertlevels)*((x1-1))]
#define pp(x0,x1) p_pp[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define qtot(x0,x1) p_qtot[(x0-1)+(nvertlevels)*((x1-1))]
#define theta(x0) p_theta[tid*((nvertlevels))+(x0-1)]
#define qvsw(x0) p_qvsw[tid*((nvertlevels))+(x0-1)]
#define temp(x0) p_temp[tid*((nvertlevels))+(x0-1)]
#define coefa(x0) p_coefa[tid*((nvertlevels))+(x0-1)]
int icell;
int k;
float dz;
float rdz;
float esw;
float p;
int dry_bv_frequency;
icell=tid+cellstart;
if(icell>cellend || !(1)) return;
for(k=1;k<=nvertlevels;++k) {
theta(k) = __fdiv_rn(theta_m(k,icell),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.9bbda40000000p+0f,scalars(index_qv,k,icell))));
temp(k) = __fmul_rn(exner(k,icell),theta(k));
p = __fadd_rn(pressure_b(k,icell),pp(k,icell));
esw = __fmul_rn(0x1.3199980000000p+9f,gfk_exp(__fdiv_rn(__fmul_rn(0x1.1ab8520000000p+4f,__fsub_rn(temp(k),0x1.1126660000000p+8f)),__fsub_rn(temp(k),0x1.da66660000000p+4f))));
if((p<esw)) {
esw = __fmul_rn(p,0x1.fae1480000000p-1f);
}
qvsw(k) = __fdiv_rn(__fmul_rn(0x1.3e56120000000p-1f,esw),__fsub_rn(p,esw));
coefa(k) = __fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fdiv_rn(__fdiv_rn(__fmul_rn(0x1.312d000000000p+21f,qvsw(k)),0x1.1f00000000000p+8f),temp(k))),__fadd_rn(0x1.0000000000000p+0f,__fdiv_rn(__fdiv_rn(__fdiv_rn(__fdiv_rn(__fmul_rn(0x1.6bcc420000000p+42f,qvsw(k)),0x1.f640000000000p+9f),0x1.cd999a0000000p+8f),temp(k)),temp(k))));
}
for(k=2;k<=(nvertlevels-1);++k) {
dz = __fsub_rn(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid((k+2),icell),zgrid((k+1),icell))),__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid(k,icell),zgrid((k-1),icell))));
rdz = __fdiv_rn(0x1.0000000000000p+0f,dz);
dry_bv_frequency = 1;
if((index_qc>0)) {
if((scalars(index_qc,k,icell)>=0x1.4f8b580000000p-17f)) {
dry_bv_frequency = 0;
}
}
if(dry_bv_frequency) {
bn2(k,icell) = __fmul_rn(0x1.39cc100000000p+3f,__fsub_rn(__fadd_rn(__fmul_rn(__fdiv_rn(__fsub_rn(theta((k+1)),theta((k-1))),theta(k)),rdz),__fmul_rn(__fmul_rn(0x1.9bbda40000000p+0f,__fsub_rn(scalars(index_qv,(k+1),icell),scalars(index_qv,(k-1),icell))),rdz)),__fmul_rn(__fsub_rn(qtot((k+1),icell),qtot((k-1),icell)),rdz)));
} else if(1) {
bn2(k,icell) = __fmul_rn(0x1.39cc100000000p+3f,__fsub_rn(__fmul_rn(coefa(k),__fadd_rn(__fmul_rn(__fdiv_rn(__fsub_rn(theta((k+1)),theta((k-1))),theta(k)),rdz),__fmul_rn(__fmul_rn(__fdiv_rn(0x1.37199c0000000p+11f,temp(k)),__fsub_rn(qvsw((k+1)),qvsw((k-1)))),rdz))),__fmul_rn(__fsub_rn(qtot((k+1),icell),qtot((k-1),icell)),rdz)));
}
}
bn2(1,icell) = bn2(2,icell);
bn2(nvertlevels,icell) = bn2((nvertlevels-1),icell);
}
#undef bn2
#undef theta_m
#undef exner
#undef pressure_b
#undef pp
#undef zgrid
#undef scalars
#undef qtot
#undef theta
#undef qvsw
#undef temp
#undef coefa

// MPAS v8.4.2 mpas_atm_dissipation_models.F:579-947
extern "C" __global__ void u_dissipation_3d_phase_0(int edgestart, int edgeend, int edgesolvestart, int edgesolveend, int vertexstart, int vertexend, int cellstart, int cellend, int ncells, int nedges, int nvertices, int vertexdegree, int *p_cellsonedge, int *p_verticesonedge, int *p_edgesoncell, int *p_edgesonvertex, int *p_nedgesoncell, float *p_edgesoncell_sign, float *p_edgesonvertex_sign, float *p_invareacell, float *p_invareatriangle, float *p_invdvedge, float *p_invdcedge, float *p_angleedge, float *p_dcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_h_mom_eddy_visc4, float *p_v_mom_eddy_visc2, float *p_config_del4u_div_factor, float *p_zgrid, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_zz, float *p_rdzu, float *p_rdzw, float *p_fzm, float *p_fzp, int les_model_opt, int les_surface_opt, float *p_config_surface_drag_coefficient, float *p_delsq_u, float *p_delsq_vorticity, float *p_delsq_divergence, float *p_u, float *p_v, float *p_divergence, float *p_vorticity, float *p_rho_edge, float *p_rho_zz, float *p_u_init, float *p_v_init, float *p_ustm, float *p_tend_u_euler, int nvertlevels, int maxedges, int num_scalars, float *p_u_mix, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define verticesonedge(x0,x1) p_verticesonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex(x0,x1) p_edgesonvertex[(x0-1)+(vertexdegree)*((x1-1))]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex_sign(x0,x1) p_edgesonvertex_sign[(x0-1)+(vertexdegree)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invareatriangle(x0) p_invareatriangle[(x0-1)]
#define invdvedge(x0) p_invdvedge[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define angleedge(x0) p_angleedge[(x0-1)]
#define dcedge(x0) p_dcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define config_del4u_div_factor (*p_config_del4u_div_factor)
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define rdzu(x0) p_rdzu[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define config_surface_drag_coefficient (*p_config_surface_drag_coefficient)
#define delsq_u(x0,x1) p_delsq_u[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_vorticity(x0,x1) p_delsq_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_divergence(x0,x1) p_delsq_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define vorticity(x0,x1) p_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define u_init(x0) p_u_init[(x0-1)]
#define v_init(x0) p_v_init[(x0-1)]
#define ustm(x0) p_ustm[(x0-1)]
#define tend_u_euler(x0,x1) p_tend_u_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define u_mix(x0) p_u_mix[tid*((nvertlevels))+(x0-1)]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int iedge;
int cell1;
int cell2;
int vertex1;
int vertex2;
int ivertex;
int icell;
int i;
int k;
float r_dc;
float r_dv;
float u_diffusion;
float u_diffusion_les;
float kdiffu;
float r;
float edge_sign;
float u_mix_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float rho_k_cell1;
float rho_k_cell2;
float rho_k_at_w;
float zz_cell1;
float zz_cell2;
float zz_at_w;
float ust_edge;
float velocity_magnitude;
float tau_12_factor;
iedge=tid+edgestart;
if(iedge>edgeend || !(1)) return;
tau_12_factor = 0x0.0p+0f;
if((les_model_opt!=0)) {
tau_12_factor = 0x1.0000000000000p+0f;
}
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
vertex1 = verticesonedge(1,iedge);
vertex2 = verticesonedge(2,iedge);
r_dc = invdcedge(iedge);
r_dv = fminf(invdvedge(iedge),__fmul_rn(4,invdcedge(iedge)));
for(k=1;k<=nvertlevels;++k) {
delsq_u(k,iedge) = 0x0.0p+0f;
}
for(k=1;k<=nvertlevels;++k) {
u_diffusion = __fsub_rn(__fmul_rn(__fsub_rn(divergence(k,cell2),divergence(k,cell1)),r_dc),__fmul_rn(__fsub_rn(vorticity(k,vertex2),vorticity(k,vertex1)),r_dv));
u_diffusion_les = __fadd_rn(u_diffusion,__fmul_rn(__fmul_rn(tau_12_factor,__fsub_rn(divergence(k,cell2),divergence(k,cell1))),r_dc));
delsq_u(k,iedge) = __fadd_rn(delsq_u(k,iedge),u_diffusion);
kdiffu = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(eddy_visc_horz(k,cell1),eddy_visc_horz(k,cell2)));
tend_u_euler(k,iedge) = __fadd_rn(tend_u_euler(k,iedge),__fmul_rn(__fmul_rn(__fmul_rn(rho_edge(k,iedge),kdiffu),u_diffusion_les),meshscalingdel2(iedge)));
}
}
#undef cellsonedge
#undef verticesonedge
#undef edgesoncell
#undef edgesonvertex
#undef nedgesoncell
#undef edgesoncell_sign
#undef edgesonvertex_sign
#undef invareacell
#undef invareatriangle
#undef invdvedge
#undef invdcedge
#undef angleedge
#undef dcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef h_mom_eddy_visc4
#undef v_mom_eddy_visc2
#undef config_del4u_div_factor
#undef zgrid
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef zz
#undef rdzu
#undef rdzw
#undef fzm
#undef fzp
#undef config_surface_drag_coefficient
#undef delsq_u
#undef delsq_vorticity
#undef delsq_divergence
#undef u
#undef v
#undef divergence
#undef vorticity
#undef rho_edge
#undef rho_zz
#undef u_init
#undef v_init
#undef ustm
#undef tend_u_euler
#undef u_mix
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:579-947
extern "C" __global__ void u_dissipation_3d_phase_1(int edgestart, int edgeend, int edgesolvestart, int edgesolveend, int vertexstart, int vertexend, int cellstart, int cellend, int ncells, int nedges, int nvertices, int vertexdegree, int *p_cellsonedge, int *p_verticesonedge, int *p_edgesoncell, int *p_edgesonvertex, int *p_nedgesoncell, float *p_edgesoncell_sign, float *p_edgesonvertex_sign, float *p_invareacell, float *p_invareatriangle, float *p_invdvedge, float *p_invdcedge, float *p_angleedge, float *p_dcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_h_mom_eddy_visc4, float *p_v_mom_eddy_visc2, float *p_config_del4u_div_factor, float *p_zgrid, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_zz, float *p_rdzu, float *p_rdzw, float *p_fzm, float *p_fzp, int les_model_opt, int les_surface_opt, float *p_config_surface_drag_coefficient, float *p_delsq_u, float *p_delsq_vorticity, float *p_delsq_divergence, float *p_u, float *p_v, float *p_divergence, float *p_vorticity, float *p_rho_edge, float *p_rho_zz, float *p_u_init, float *p_v_init, float *p_ustm, float *p_tend_u_euler, int nvertlevels, int maxedges, int num_scalars, float *p_u_mix, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define verticesonedge(x0,x1) p_verticesonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex(x0,x1) p_edgesonvertex[(x0-1)+(vertexdegree)*((x1-1))]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex_sign(x0,x1) p_edgesonvertex_sign[(x0-1)+(vertexdegree)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invareatriangle(x0) p_invareatriangle[(x0-1)]
#define invdvedge(x0) p_invdvedge[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define angleedge(x0) p_angleedge[(x0-1)]
#define dcedge(x0) p_dcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define config_del4u_div_factor (*p_config_del4u_div_factor)
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define rdzu(x0) p_rdzu[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define config_surface_drag_coefficient (*p_config_surface_drag_coefficient)
#define delsq_u(x0,x1) p_delsq_u[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_vorticity(x0,x1) p_delsq_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_divergence(x0,x1) p_delsq_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define vorticity(x0,x1) p_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define u_init(x0) p_u_init[(x0-1)]
#define v_init(x0) p_v_init[(x0-1)]
#define ustm(x0) p_ustm[(x0-1)]
#define tend_u_euler(x0,x1) p_tend_u_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define u_mix(x0) p_u_mix[tid*((nvertlevels))+(x0-1)]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int iedge;
int cell1;
int cell2;
int vertex1;
int vertex2;
int ivertex;
int icell;
int i;
int k;
float r_dc;
float r_dv;
float u_diffusion;
float u_diffusion_les;
float kdiffu;
float r;
float edge_sign;
float u_mix_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float rho_k_cell1;
float rho_k_cell2;
float rho_k_at_w;
float zz_cell1;
float zz_cell2;
float zz_at_w;
float ust_edge;
float velocity_magnitude;
float tau_12_factor;
ivertex=tid+vertexstart;
if(ivertex>vertexend || !((1) && ((h_mom_eddy_visc4>0x0.0p+0f)))) return;
tau_12_factor = 0x0.0p+0f;
if((les_model_opt!=0)) {
tau_12_factor = 0x1.0000000000000p+0f;
}
for(k=1;k<=nvertlevels;++k) {
delsq_vorticity(k,ivertex) = 0x0.0p+0f;
}
for(i=1;i<=vertexdegree;++i) {
iedge = edgesonvertex(i,ivertex);
edge_sign = __fmul_rn(__fmul_rn(invareatriangle(ivertex),dcedge(iedge)),edgesonvertex_sign(i,ivertex));
for(k=1;k<=nvertlevels;++k) {
delsq_vorticity(k,ivertex) = __fadd_rn(delsq_vorticity(k,ivertex),__fmul_rn(edge_sign,delsq_u(k,iedge)));
}
}
}
#undef cellsonedge
#undef verticesonedge
#undef edgesoncell
#undef edgesonvertex
#undef nedgesoncell
#undef edgesoncell_sign
#undef edgesonvertex_sign
#undef invareacell
#undef invareatriangle
#undef invdvedge
#undef invdcedge
#undef angleedge
#undef dcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef h_mom_eddy_visc4
#undef v_mom_eddy_visc2
#undef config_del4u_div_factor
#undef zgrid
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef zz
#undef rdzu
#undef rdzw
#undef fzm
#undef fzp
#undef config_surface_drag_coefficient
#undef delsq_u
#undef delsq_vorticity
#undef delsq_divergence
#undef u
#undef v
#undef divergence
#undef vorticity
#undef rho_edge
#undef rho_zz
#undef u_init
#undef v_init
#undef ustm
#undef tend_u_euler
#undef u_mix
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:579-947
extern "C" __global__ void u_dissipation_3d_phase_2(int edgestart, int edgeend, int edgesolvestart, int edgesolveend, int vertexstart, int vertexend, int cellstart, int cellend, int ncells, int nedges, int nvertices, int vertexdegree, int *p_cellsonedge, int *p_verticesonedge, int *p_edgesoncell, int *p_edgesonvertex, int *p_nedgesoncell, float *p_edgesoncell_sign, float *p_edgesonvertex_sign, float *p_invareacell, float *p_invareatriangle, float *p_invdvedge, float *p_invdcedge, float *p_angleedge, float *p_dcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_h_mom_eddy_visc4, float *p_v_mom_eddy_visc2, float *p_config_del4u_div_factor, float *p_zgrid, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_zz, float *p_rdzu, float *p_rdzw, float *p_fzm, float *p_fzp, int les_model_opt, int les_surface_opt, float *p_config_surface_drag_coefficient, float *p_delsq_u, float *p_delsq_vorticity, float *p_delsq_divergence, float *p_u, float *p_v, float *p_divergence, float *p_vorticity, float *p_rho_edge, float *p_rho_zz, float *p_u_init, float *p_v_init, float *p_ustm, float *p_tend_u_euler, int nvertlevels, int maxedges, int num_scalars, float *p_u_mix, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define verticesonedge(x0,x1) p_verticesonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex(x0,x1) p_edgesonvertex[(x0-1)+(vertexdegree)*((x1-1))]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex_sign(x0,x1) p_edgesonvertex_sign[(x0-1)+(vertexdegree)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invareatriangle(x0) p_invareatriangle[(x0-1)]
#define invdvedge(x0) p_invdvedge[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define angleedge(x0) p_angleedge[(x0-1)]
#define dcedge(x0) p_dcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define config_del4u_div_factor (*p_config_del4u_div_factor)
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define rdzu(x0) p_rdzu[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define config_surface_drag_coefficient (*p_config_surface_drag_coefficient)
#define delsq_u(x0,x1) p_delsq_u[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_vorticity(x0,x1) p_delsq_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_divergence(x0,x1) p_delsq_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define vorticity(x0,x1) p_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define u_init(x0) p_u_init[(x0-1)]
#define v_init(x0) p_v_init[(x0-1)]
#define ustm(x0) p_ustm[(x0-1)]
#define tend_u_euler(x0,x1) p_tend_u_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define u_mix(x0) p_u_mix[tid*((nvertlevels))+(x0-1)]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int iedge;
int cell1;
int cell2;
int vertex1;
int vertex2;
int ivertex;
int icell;
int i;
int k;
float r_dc;
float r_dv;
float u_diffusion;
float u_diffusion_les;
float kdiffu;
float r;
float edge_sign;
float u_mix_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float rho_k_cell1;
float rho_k_cell2;
float rho_k_at_w;
float zz_cell1;
float zz_cell2;
float zz_at_w;
float ust_edge;
float velocity_magnitude;
float tau_12_factor;
icell=tid+cellstart;
if(icell>cellend || !((1) && ((h_mom_eddy_visc4>0x0.0p+0f)))) return;
tau_12_factor = 0x0.0p+0f;
if((les_model_opt!=0)) {
tau_12_factor = 0x1.0000000000000p+0f;
}
for(k=1;k<=nvertlevels;++k) {
delsq_divergence(k,icell) = 0x0.0p+0f;
}
r = invareacell(icell);
for(i=1;i<=nedgesoncell(icell);++i) {
iedge = edgesoncell(i,icell);
edge_sign = __fmul_rn(__fmul_rn(r,dvedge(iedge)),edgesoncell_sign(i,icell));
for(k=1;k<=nvertlevels;++k) {
delsq_divergence(k,icell) = __fadd_rn(delsq_divergence(k,icell),__fmul_rn(edge_sign,delsq_u(k,iedge)));
}
}
}
#undef cellsonedge
#undef verticesonedge
#undef edgesoncell
#undef edgesonvertex
#undef nedgesoncell
#undef edgesoncell_sign
#undef edgesonvertex_sign
#undef invareacell
#undef invareatriangle
#undef invdvedge
#undef invdcedge
#undef angleedge
#undef dcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef h_mom_eddy_visc4
#undef v_mom_eddy_visc2
#undef config_del4u_div_factor
#undef zgrid
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef zz
#undef rdzu
#undef rdzw
#undef fzm
#undef fzp
#undef config_surface_drag_coefficient
#undef delsq_u
#undef delsq_vorticity
#undef delsq_divergence
#undef u
#undef v
#undef divergence
#undef vorticity
#undef rho_edge
#undef rho_zz
#undef u_init
#undef v_init
#undef ustm
#undef tend_u_euler
#undef u_mix
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:579-947
extern "C" __global__ void u_dissipation_3d_phase_3(int edgestart, int edgeend, int edgesolvestart, int edgesolveend, int vertexstart, int vertexend, int cellstart, int cellend, int ncells, int nedges, int nvertices, int vertexdegree, int *p_cellsonedge, int *p_verticesonedge, int *p_edgesoncell, int *p_edgesonvertex, int *p_nedgesoncell, float *p_edgesoncell_sign, float *p_edgesonvertex_sign, float *p_invareacell, float *p_invareatriangle, float *p_invdvedge, float *p_invdcedge, float *p_angleedge, float *p_dcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_h_mom_eddy_visc4, float *p_v_mom_eddy_visc2, float *p_config_del4u_div_factor, float *p_zgrid, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_zz, float *p_rdzu, float *p_rdzw, float *p_fzm, float *p_fzp, int les_model_opt, int les_surface_opt, float *p_config_surface_drag_coefficient, float *p_delsq_u, float *p_delsq_vorticity, float *p_delsq_divergence, float *p_u, float *p_v, float *p_divergence, float *p_vorticity, float *p_rho_edge, float *p_rho_zz, float *p_u_init, float *p_v_init, float *p_ustm, float *p_tend_u_euler, int nvertlevels, int maxedges, int num_scalars, float *p_u_mix, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define verticesonedge(x0,x1) p_verticesonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex(x0,x1) p_edgesonvertex[(x0-1)+(vertexdegree)*((x1-1))]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex_sign(x0,x1) p_edgesonvertex_sign[(x0-1)+(vertexdegree)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invareatriangle(x0) p_invareatriangle[(x0-1)]
#define invdvedge(x0) p_invdvedge[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define angleedge(x0) p_angleedge[(x0-1)]
#define dcedge(x0) p_dcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define config_del4u_div_factor (*p_config_del4u_div_factor)
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define rdzu(x0) p_rdzu[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define config_surface_drag_coefficient (*p_config_surface_drag_coefficient)
#define delsq_u(x0,x1) p_delsq_u[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_vorticity(x0,x1) p_delsq_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_divergence(x0,x1) p_delsq_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define vorticity(x0,x1) p_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define u_init(x0) p_u_init[(x0-1)]
#define v_init(x0) p_v_init[(x0-1)]
#define ustm(x0) p_ustm[(x0-1)]
#define tend_u_euler(x0,x1) p_tend_u_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define u_mix(x0) p_u_mix[tid*((nvertlevels))+(x0-1)]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int iedge;
int cell1;
int cell2;
int vertex1;
int vertex2;
int ivertex;
int icell;
int i;
int k;
float r_dc;
float r_dv;
float u_diffusion;
float u_diffusion_les;
float kdiffu;
float r;
float edge_sign;
float u_mix_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float rho_k_cell1;
float rho_k_cell2;
float rho_k_at_w;
float zz_cell1;
float zz_cell2;
float zz_at_w;
float ust_edge;
float velocity_magnitude;
float tau_12_factor;
iedge=tid+edgesolvestart;
if(iedge>edgesolveend || !((1) && ((h_mom_eddy_visc4>0x0.0p+0f)))) return;
tau_12_factor = 0x0.0p+0f;
if((les_model_opt!=0)) {
tau_12_factor = 0x1.0000000000000p+0f;
}
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
vertex1 = verticesonedge(1,iedge);
vertex2 = verticesonedge(2,iedge);
u_mix_scale = __fmul_rn(meshscalingdel4(iedge),h_mom_eddy_visc4);
r_dc = __fmul_rn(__fmul_rn(u_mix_scale,config_del4u_div_factor),invdcedge(iedge));
r_dv = __fmul_rn(u_mix_scale,fminf(invdvedge(iedge),__fmul_rn(4,invdcedge(iedge))));
for(k=1;k<=nvertlevels;++k) {
u_diffusion = __fmul_rn(rho_edge(k,iedge),__fsub_rn(__fmul_rn(__fsub_rn(delsq_divergence(k,cell2),delsq_divergence(k,cell1)),r_dc),__fmul_rn(__fsub_rn(delsq_vorticity(k,vertex2),delsq_vorticity(k,vertex1)),r_dv)));
tend_u_euler(k,iedge) = __fsub_rn(tend_u_euler(k,iedge),u_diffusion);
}
}
#undef cellsonedge
#undef verticesonedge
#undef edgesoncell
#undef edgesonvertex
#undef nedgesoncell
#undef edgesoncell_sign
#undef edgesonvertex_sign
#undef invareacell
#undef invareatriangle
#undef invdvedge
#undef invdcedge
#undef angleedge
#undef dcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef h_mom_eddy_visc4
#undef v_mom_eddy_visc2
#undef config_del4u_div_factor
#undef zgrid
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef zz
#undef rdzu
#undef rdzw
#undef fzm
#undef fzp
#undef config_surface_drag_coefficient
#undef delsq_u
#undef delsq_vorticity
#undef delsq_divergence
#undef u
#undef v
#undef divergence
#undef vorticity
#undef rho_edge
#undef rho_zz
#undef u_init
#undef v_init
#undef ustm
#undef tend_u_euler
#undef u_mix
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:579-947
extern "C" __global__ void u_dissipation_3d_phase_4(int edgestart, int edgeend, int edgesolvestart, int edgesolveend, int vertexstart, int vertexend, int cellstart, int cellend, int ncells, int nedges, int nvertices, int vertexdegree, int *p_cellsonedge, int *p_verticesonedge, int *p_edgesoncell, int *p_edgesonvertex, int *p_nedgesoncell, float *p_edgesoncell_sign, float *p_edgesonvertex_sign, float *p_invareacell, float *p_invareatriangle, float *p_invdvedge, float *p_invdcedge, float *p_angleedge, float *p_dcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_h_mom_eddy_visc4, float *p_v_mom_eddy_visc2, float *p_config_del4u_div_factor, float *p_zgrid, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_zz, float *p_rdzu, float *p_rdzw, float *p_fzm, float *p_fzp, int les_model_opt, int les_surface_opt, float *p_config_surface_drag_coefficient, float *p_delsq_u, float *p_delsq_vorticity, float *p_delsq_divergence, float *p_u, float *p_v, float *p_divergence, float *p_vorticity, float *p_rho_edge, float *p_rho_zz, float *p_u_init, float *p_v_init, float *p_ustm, float *p_tend_u_euler, int nvertlevels, int maxedges, int num_scalars, float *p_u_mix, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define verticesonedge(x0,x1) p_verticesonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex(x0,x1) p_edgesonvertex[(x0-1)+(vertexdegree)*((x1-1))]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex_sign(x0,x1) p_edgesonvertex_sign[(x0-1)+(vertexdegree)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invareatriangle(x0) p_invareatriangle[(x0-1)]
#define invdvedge(x0) p_invdvedge[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define angleedge(x0) p_angleedge[(x0-1)]
#define dcedge(x0) p_dcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define config_del4u_div_factor (*p_config_del4u_div_factor)
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define rdzu(x0) p_rdzu[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define config_surface_drag_coefficient (*p_config_surface_drag_coefficient)
#define delsq_u(x0,x1) p_delsq_u[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_vorticity(x0,x1) p_delsq_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_divergence(x0,x1) p_delsq_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define vorticity(x0,x1) p_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define u_init(x0) p_u_init[(x0-1)]
#define v_init(x0) p_v_init[(x0-1)]
#define ustm(x0) p_ustm[(x0-1)]
#define tend_u_euler(x0,x1) p_tend_u_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define u_mix(x0) p_u_mix[tid*((nvertlevels))+(x0-1)]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int iedge;
int cell1;
int cell2;
int vertex1;
int vertex2;
int ivertex;
int icell;
int i;
int k;
float r_dc;
float r_dv;
float u_diffusion;
float u_diffusion_les;
float kdiffu;
float r;
float edge_sign;
float u_mix_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float rho_k_cell1;
float rho_k_cell2;
float rho_k_at_w;
float zz_cell1;
float zz_cell2;
float zz_at_w;
float ust_edge;
float velocity_magnitude;
float tau_12_factor;
iedge=tid+edgesolvestart;
if(iedge>edgesolveend || !(((1) && ((v_mom_eddy_visc2>0x0.0p+0f))) && (config_mix_full))) return;
tau_12_factor = 0x0.0p+0f;
if((les_model_opt!=0)) {
tau_12_factor = 0x1.0000000000000p+0f;
}
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
for(k=2;k<=(nvertlevels-1);++k) {
z1 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid((k-1),cell1),zgrid((k-1),cell2)));
z2 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid(k,cell1),zgrid(k,cell2)));
z3 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid((k+1),cell1),zgrid((k+1),cell2)));
z4 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid((k+2),cell1),zgrid((k+2),cell2)));
zm = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z1,z2));
z0 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z2,z3));
zp = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z3,z4));
tend_u_euler(k,iedge) = __fadd_rn(tend_u_euler(k,iedge),__fdiv_rn(__fmul_rn(__fmul_rn(rho_edge(k,iedge),v_mom_eddy_visc2),__fsub_rn(__fdiv_rn(__fsub_rn(u((k+1),iedge),u(k,iedge)),__fsub_rn(zp,z0)),__fdiv_rn(__fsub_rn(u(k,iedge),u((k-1),iedge)),__fsub_rn(z0,zm)))),__fmul_rn(0x1.0000000000000p-1f,__fsub_rn(zp,zm))));
}
}
#undef cellsonedge
#undef verticesonedge
#undef edgesoncell
#undef edgesonvertex
#undef nedgesoncell
#undef edgesoncell_sign
#undef edgesonvertex_sign
#undef invareacell
#undef invareatriangle
#undef invdvedge
#undef invdcedge
#undef angleedge
#undef dcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef h_mom_eddy_visc4
#undef v_mom_eddy_visc2
#undef config_del4u_div_factor
#undef zgrid
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef zz
#undef rdzu
#undef rdzw
#undef fzm
#undef fzp
#undef config_surface_drag_coefficient
#undef delsq_u
#undef delsq_vorticity
#undef delsq_divergence
#undef u
#undef v
#undef divergence
#undef vorticity
#undef rho_edge
#undef rho_zz
#undef u_init
#undef v_init
#undef ustm
#undef tend_u_euler
#undef u_mix
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:579-947
extern "C" __global__ void u_dissipation_3d_phase_5(int edgestart, int edgeend, int edgesolvestart, int edgesolveend, int vertexstart, int vertexend, int cellstart, int cellend, int ncells, int nedges, int nvertices, int vertexdegree, int *p_cellsonedge, int *p_verticesonedge, int *p_edgesoncell, int *p_edgesonvertex, int *p_nedgesoncell, float *p_edgesoncell_sign, float *p_edgesonvertex_sign, float *p_invareacell, float *p_invareatriangle, float *p_invdvedge, float *p_invdcedge, float *p_angleedge, float *p_dcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_h_mom_eddy_visc4, float *p_v_mom_eddy_visc2, float *p_config_del4u_div_factor, float *p_zgrid, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_zz, float *p_rdzu, float *p_rdzw, float *p_fzm, float *p_fzp, int les_model_opt, int les_surface_opt, float *p_config_surface_drag_coefficient, float *p_delsq_u, float *p_delsq_vorticity, float *p_delsq_divergence, float *p_u, float *p_v, float *p_divergence, float *p_vorticity, float *p_rho_edge, float *p_rho_zz, float *p_u_init, float *p_v_init, float *p_ustm, float *p_tend_u_euler, int nvertlevels, int maxedges, int num_scalars, float *p_u_mix, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define verticesonedge(x0,x1) p_verticesonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex(x0,x1) p_edgesonvertex[(x0-1)+(vertexdegree)*((x1-1))]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex_sign(x0,x1) p_edgesonvertex_sign[(x0-1)+(vertexdegree)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invareatriangle(x0) p_invareatriangle[(x0-1)]
#define invdvedge(x0) p_invdvedge[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define angleedge(x0) p_angleedge[(x0-1)]
#define dcedge(x0) p_dcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define config_del4u_div_factor (*p_config_del4u_div_factor)
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define rdzu(x0) p_rdzu[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define config_surface_drag_coefficient (*p_config_surface_drag_coefficient)
#define delsq_u(x0,x1) p_delsq_u[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_vorticity(x0,x1) p_delsq_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_divergence(x0,x1) p_delsq_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define vorticity(x0,x1) p_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define u_init(x0) p_u_init[(x0-1)]
#define v_init(x0) p_v_init[(x0-1)]
#define ustm(x0) p_ustm[(x0-1)]
#define tend_u_euler(x0,x1) p_tend_u_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define u_mix(x0) p_u_mix[tid*((nvertlevels))+(x0-1)]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int iedge;
int cell1;
int cell2;
int vertex1;
int vertex2;
int ivertex;
int icell;
int i;
int k;
float r_dc;
float r_dv;
float u_diffusion;
float u_diffusion_les;
float kdiffu;
float r;
float edge_sign;
float u_mix_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float rho_k_cell1;
float rho_k_cell2;
float rho_k_at_w;
float zz_cell1;
float zz_cell2;
float zz_at_w;
float ust_edge;
float velocity_magnitude;
float tau_12_factor;
iedge=tid+edgesolvestart;
if(iedge>edgesolveend || !(((1) && ((v_mom_eddy_visc2>0x0.0p+0f))) && !(config_mix_full) && (1))) return;
tau_12_factor = 0x0.0p+0f;
if((les_model_opt!=0)) {
tau_12_factor = 0x1.0000000000000p+0f;
}
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
for(k=1;k<=nvertlevels;++k) {
u_mix(k) = __fadd_rn(__fsub_rn(u(k,iedge),__fmul_rn(u_init(k),glibc_cosf(angleedge(iedge)))),__fmul_rn(v_init(k),glibc_sinf(angleedge(iedge))));
}
for(k=2;k<=(nvertlevels-1);++k) {
z1 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid((k-1),cell1),zgrid((k-1),cell2)));
z2 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid(k,cell1),zgrid(k,cell2)));
z3 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid((k+1),cell1),zgrid((k+1),cell2)));
z4 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zgrid((k+2),cell1),zgrid((k+2),cell2)));
zm = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z1,z2));
z0 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z2,z3));
zp = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z3,z4));
tend_u_euler(k,iedge) = __fadd_rn(tend_u_euler(k,iedge),__fdiv_rn(__fmul_rn(__fmul_rn(rho_edge(k,iedge),v_mom_eddy_visc2),__fsub_rn(__fdiv_rn(__fsub_rn(u_mix((k+1)),u_mix(k)),__fsub_rn(zp,z0)),__fdiv_rn(__fsub_rn(u_mix(k),u_mix((k-1))),__fsub_rn(z0,zm)))),__fmul_rn(0x1.0000000000000p-1f,__fsub_rn(zp,zm))));
}
}
#undef cellsonedge
#undef verticesonedge
#undef edgesoncell
#undef edgesonvertex
#undef nedgesoncell
#undef edgesoncell_sign
#undef edgesonvertex_sign
#undef invareacell
#undef invareatriangle
#undef invdvedge
#undef invdcedge
#undef angleedge
#undef dcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef h_mom_eddy_visc4
#undef v_mom_eddy_visc2
#undef config_del4u_div_factor
#undef zgrid
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef zz
#undef rdzu
#undef rdzw
#undef fzm
#undef fzp
#undef config_surface_drag_coefficient
#undef delsq_u
#undef delsq_vorticity
#undef delsq_divergence
#undef u
#undef v
#undef divergence
#undef vorticity
#undef rho_edge
#undef rho_zz
#undef u_init
#undef v_init
#undef ustm
#undef tend_u_euler
#undef u_mix
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:579-947
extern "C" __global__ void u_dissipation_3d_phase_6(int edgestart, int edgeend, int edgesolvestart, int edgesolveend, int vertexstart, int vertexend, int cellstart, int cellend, int ncells, int nedges, int nvertices, int vertexdegree, int *p_cellsonedge, int *p_verticesonedge, int *p_edgesoncell, int *p_edgesonvertex, int *p_nedgesoncell, float *p_edgesoncell_sign, float *p_edgesonvertex_sign, float *p_invareacell, float *p_invareatriangle, float *p_invdvedge, float *p_invdcedge, float *p_angleedge, float *p_dcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_h_mom_eddy_visc4, float *p_v_mom_eddy_visc2, float *p_config_del4u_div_factor, float *p_zgrid, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_zz, float *p_rdzu, float *p_rdzw, float *p_fzm, float *p_fzp, int les_model_opt, int les_surface_opt, float *p_config_surface_drag_coefficient, float *p_delsq_u, float *p_delsq_vorticity, float *p_delsq_divergence, float *p_u, float *p_v, float *p_divergence, float *p_vorticity, float *p_rho_edge, float *p_rho_zz, float *p_u_init, float *p_v_init, float *p_ustm, float *p_tend_u_euler, int nvertlevels, int maxedges, int num_scalars, float *p_u_mix, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define verticesonedge(x0,x1) p_verticesonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex(x0,x1) p_edgesonvertex[(x0-1)+(vertexdegree)*((x1-1))]
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define edgesonvertex_sign(x0,x1) p_edgesonvertex_sign[(x0-1)+(vertexdegree)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invareatriangle(x0) p_invareatriangle[(x0-1)]
#define invdvedge(x0) p_invdvedge[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define angleedge(x0) p_angleedge[(x0-1)]
#define dcedge(x0) p_dcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define config_del4u_div_factor (*p_config_del4u_div_factor)
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define rdzu(x0) p_rdzu[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define config_surface_drag_coefficient (*p_config_surface_drag_coefficient)
#define delsq_u(x0,x1) p_delsq_u[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_vorticity(x0,x1) p_delsq_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_divergence(x0,x1) p_delsq_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define u(x0,x1) p_u[(x0-1)+(nvertlevels)*((x1-1))]
#define v(x0,x1) p_v[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define vorticity(x0,x1) p_vorticity[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define u_init(x0) p_u_init[(x0-1)]
#define v_init(x0) p_v_init[(x0-1)]
#define ustm(x0) p_ustm[(x0-1)]
#define tend_u_euler(x0,x1) p_tend_u_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define u_mix(x0) p_u_mix[tid*((nvertlevels))+(x0-1)]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int iedge;
int cell1;
int cell2;
int vertex1;
int vertex2;
int ivertex;
int icell;
int i;
int k;
float r_dc;
float r_dv;
float u_diffusion;
float u_diffusion_les;
float kdiffu;
float r;
float edge_sign;
float u_mix_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float rho_k_cell1;
float rho_k_cell2;
float rho_k_at_w;
float zz_cell1;
float zz_cell2;
float zz_at_w;
float ust_edge;
float velocity_magnitude;
float tau_12_factor;
iedge=tid+edgesolvestart;
if(iedge>edgesolveend || !((1) && ((les_model_opt!=0)))) return;
tau_12_factor = 0x0.0p+0f;
if((les_model_opt!=0)) {
tau_12_factor = 0x1.0000000000000p+0f;
}
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
turb_vflux((nvertlevels+1)) = 0x0.0p+0f;
turb_vflux(1) = 0x0.0p+0f;
for(k=2;k<=nvertlevels;++k) {
rho_k_cell1 = __fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(fzm(k),rho_zz(k,cell1)),zz(k,cell1)),eddy_visc_vert(k,cell1)),__fmul_rn(__fmul_rn(__fmul_rn(fzp(k),rho_zz((k-1),cell1)),zz((k-1),cell1)),eddy_visc_vert((k-1),cell1)));
rho_k_cell2 = __fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(fzm(k),rho_zz(k,cell2)),zz(k,cell2)),eddy_visc_vert(k,cell2)),__fmul_rn(__fmul_rn(__fmul_rn(fzp(k),rho_zz((k-1),cell2)),zz((k-1),cell2)),eddy_visc_vert((k-1),cell2)));
rho_k_at_w = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(rho_k_cell1,rho_k_cell2));
zz_cell1 = __fadd_rn(__fmul_rn(fzm(k),zz(k,cell1)),__fmul_rn(fzp(k),zz((k-1),cell1)));
zz_cell2 = __fadd_rn(__fmul_rn(fzm(k),zz(k,cell2)),__fmul_rn(fzp(k),zz((k-1),cell2)));
zz_at_w = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(zz_cell1,zz_cell2));
turb_vflux(k) = __fmul_rn(__fmul_rn(__fmul_rn((-rho_k_at_w),zz_at_w),rdzu(k)),__fsub_rn(u(k,iedge),u((k-1),iedge)));
}
if((les_surface_opt==1)) {
velocity_magnitude = __fsqrt_rn(__fadd_rn(__fmul_rn(u(1,iedge),u(1,iedge)),__fmul_rn(v(1,iedge),v(1,iedge))));
turb_vflux(1) = __fmul_rn(__fmul_rn(__fmul_rn((-rho_edge(1,iedge)),config_surface_drag_coefficient),u(1,iedge)),velocity_magnitude);
turb_vflux((nvertlevels+1)) = turb_vflux(nvertlevels);
} else if((les_surface_opt==2)) {
ust_edge = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(ustm(cell1),ustm(cell2)));
velocity_magnitude = fmaxf(__fsqrt_rn(__fadd_rn(__fmul_rn(u(1,iedge),u(1,iedge)),__fmul_rn(v(1,iedge),v(1,iedge)))),0x1.99999a0000000p-4f);
turb_vflux(1) = __fmul_rn(__fmul_rn(__fmul_rn((-rho_edge(1,iedge)),ust_edge),ust_edge),__fdiv_rn(u(1,iedge),velocity_magnitude));
turb_vflux((nvertlevels+1)) = turb_vflux(nvertlevels);
} else if(1) {
turb_vflux(1) = turb_vflux(2);
turb_vflux((nvertlevels+1)) = turb_vflux(nvertlevels);
}
for(k=1;k<=nvertlevels;++k) {
tend_u_euler(k,iedge) = __fsub_rn(tend_u_euler(k,iedge),__fmul_rn(rdzw(k),__fsub_rn(turb_vflux((k+1)),turb_vflux(k))));
}
}
#undef cellsonedge
#undef verticesonedge
#undef edgesoncell
#undef edgesonvertex
#undef nedgesoncell
#undef edgesoncell_sign
#undef edgesonvertex_sign
#undef invareacell
#undef invareatriangle
#undef invdvedge
#undef invdcedge
#undef angleedge
#undef dcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef h_mom_eddy_visc4
#undef v_mom_eddy_visc2
#undef config_del4u_div_factor
#undef zgrid
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef zz
#undef rdzu
#undef rdzw
#undef fzm
#undef fzp
#undef config_surface_drag_coefficient
#undef delsq_u
#undef delsq_vorticity
#undef delsq_divergence
#undef u
#undef v
#undef divergence
#undef vorticity
#undef rho_edge
#undef rho_zz
#undef u_init
#undef v_init
#undef ustm
#undef tend_u_euler
#undef u_mix
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:951-1153
extern "C" __global__ void w_dissipation_3d_phase_0(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, float *p_rdzw, float *p_rdzu, float *p_v_mom_eddy_visc2, float *p_h_mom_eddy_visc4, float *p_delsq_w, float *p_w, float *p_rho_edge, float *p_rho_zz, float *p_divergence, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, int les_model_opt, int les_surface_opt, float *p_tend_w_euler, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define delsq_w(x0,x1) p_delsq_w[(x0-1)+(nvertlevels)*((x1-1))]
#define w(x0,x1) p_w[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define tend_w_euler(x0,x1) p_tend_w_euler[(x0-1)+(nvertlevels+1)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
float r_areacell;
float edge_sign;
float w_turb_flux;
icell=tid+cellstart;
if(icell>cellend || !(1)) return;
for(k=1;k<=nvertlevels;++k) {
delsq_w(k,icell) = 0x0.0p+0f;
}
for(k=1;k<=(nvertlevels+1);++k) {
tend_w_euler(k,icell) = 0x0.0p+0f;
}
r_areacell = invareacell(icell);
for(i=1;i<=nedgesoncell(icell);++i) {
iedge = edgesoncell(i,icell);
edge_sign = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(0x1.0000000000000p-1f,r_areacell),edgesoncell_sign(i,icell)),dvedge(iedge)),invdcedge(iedge));
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
for(k=2;k<=nvertlevels;++k) {
w_turb_flux = __fmul_rn(__fmul_rn(edge_sign,__fadd_rn(rho_edge(k,iedge),rho_edge((k-1),iedge))),__fsub_rn(w(k,cell2),w(k,cell1)));
delsq_w(k,icell) = __fadd_rn(delsq_w(k,icell),w_turb_flux);
w_turb_flux = __fmul_rn(__fmul_rn(__fmul_rn(w_turb_flux,meshscalingdel2(iedge)),0x1.0000000000000p-2f),__fadd_rn(__fadd_rn(__fadd_rn(eddy_visc_horz(k,cell1),eddy_visc_horz(k,cell2)),eddy_visc_horz((k-1),cell1)),eddy_visc_horz((k-1),cell2)));
tend_w_euler(k,icell) = __fadd_rn(tend_w_euler(k,icell),w_turb_flux);
}
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef rdzw
#undef rdzu
#undef v_mom_eddy_visc2
#undef h_mom_eddy_visc4
#undef delsq_w
#undef w
#undef rho_edge
#undef rho_zz
#undef divergence
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef tend_w_euler
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:951-1153
extern "C" __global__ void w_dissipation_3d_phase_1(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, float *p_rdzw, float *p_rdzu, float *p_v_mom_eddy_visc2, float *p_h_mom_eddy_visc4, float *p_delsq_w, float *p_w, float *p_rho_edge, float *p_rho_zz, float *p_divergence, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, int les_model_opt, int les_surface_opt, float *p_tend_w_euler, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define delsq_w(x0,x1) p_delsq_w[(x0-1)+(nvertlevels)*((x1-1))]
#define w(x0,x1) p_w[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define tend_w_euler(x0,x1) p_tend_w_euler[(x0-1)+(nvertlevels+1)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
float r_areacell;
float edge_sign;
float w_turb_flux;
icell=tid+cellsolvestart;
if(icell>cellsolveend || !((1) && ((h_mom_eddy_visc4>0x0.0p+0f)))) return;
r_areacell = __fmul_rn(h_mom_eddy_visc4,invareacell(icell));
for(i=1;i<=nedgesoncell(icell);++i) {
iedge = edgesoncell(i,icell);
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
edge_sign = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(meshscalingdel4(iedge),r_areacell),dvedge(iedge)),edgesoncell_sign(i,icell)),invdcedge(iedge));
for(k=2;k<=nvertlevels;++k) {
tend_w_euler(k,icell) = __fsub_rn(tend_w_euler(k,icell),__fmul_rn(edge_sign,__fsub_rn(delsq_w(k,cell2),delsq_w(k,cell1))));
}
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef rdzw
#undef rdzu
#undef v_mom_eddy_visc2
#undef h_mom_eddy_visc4
#undef delsq_w
#undef w
#undef rho_edge
#undef rho_zz
#undef divergence
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef tend_w_euler
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:951-1153
extern "C" __global__ void w_dissipation_3d_phase_2(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, float *p_rdzw, float *p_rdzu, float *p_v_mom_eddy_visc2, float *p_h_mom_eddy_visc4, float *p_delsq_w, float *p_w, float *p_rho_edge, float *p_rho_zz, float *p_divergence, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, int les_model_opt, int les_surface_opt, float *p_tend_w_euler, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define delsq_w(x0,x1) p_delsq_w[(x0-1)+(nvertlevels)*((x1-1))]
#define w(x0,x1) p_w[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define tend_w_euler(x0,x1) p_tend_w_euler[(x0-1)+(nvertlevels+1)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
float r_areacell;
float edge_sign;
float w_turb_flux;
icell=tid+cellsolvestart;
if(icell>cellsolveend || !((1) && ((v_mom_eddy_visc2>0x0.0p+0f)))) return;
for(k=2;k<=nvertlevels;++k) {
tend_w_euler(k,icell) = __fadd_rn(tend_w_euler(k,icell),__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(v_mom_eddy_visc2,0x1.0000000000000p-1f),__fadd_rn(rho_zz(k,icell),rho_zz((k-1),icell))),__fsub_rn(__fmul_rn(__fsub_rn(w((k+1),icell),w(k,icell)),rdzw(k)),__fmul_rn(__fsub_rn(w(k,icell),w((k-1),icell)),rdzw((k-1))))),rdzu(k)));
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef rdzw
#undef rdzu
#undef v_mom_eddy_visc2
#undef h_mom_eddy_visc4
#undef delsq_w
#undef w
#undef rho_edge
#undef rho_zz
#undef divergence
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef tend_w_euler
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:951-1153
extern "C" __global__ void w_dissipation_3d_phase_3(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, float *p_rdzw, float *p_rdzu, float *p_v_mom_eddy_visc2, float *p_h_mom_eddy_visc4, float *p_delsq_w, float *p_w, float *p_rho_edge, float *p_rho_zz, float *p_divergence, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, int les_model_opt, int les_surface_opt, float *p_tend_w_euler, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define v_mom_eddy_visc2 (*p_v_mom_eddy_visc2)
#define h_mom_eddy_visc4 (*p_h_mom_eddy_visc4)
#define delsq_w(x0,x1) p_delsq_w[(x0-1)+(nvertlevels)*((x1-1))]
#define w(x0,x1) p_w[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define divergence(x0,x1) p_divergence[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define tend_w_euler(x0,x1) p_tend_w_euler[(x0-1)+(nvertlevels+1)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
float r_areacell;
float edge_sign;
float w_turb_flux;
icell=tid+cellsolvestart;
if(icell>cellsolveend || !((1) && ((les_model_opt!=0)))) return;
for(k=1;k<=nvertlevels;++k) {
turb_vflux(k) = __fmul_rn(__fmul_rn(__fmul_rn((-rho_zz(k,icell)),eddy_visc_vert(k,icell)),zz(k,icell)),__fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(0x1.0000000000000p+1f,zz(k,icell)),rdzw(k)),__fsub_rn(w((k+1),icell),w(k,icell))),divergence(k,icell)));
}
turb_vflux((nvertlevels+1)) = 0x0.0p+0f;
for(k=2;k<=nvertlevels;++k) {
tend_w_euler(k,icell) = __fsub_rn(tend_w_euler(k,icell),__fmul_rn(rdzu(k),__fsub_rn(turb_vflux(k),turb_vflux((k-1)))));
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef rdzw
#undef rdzu
#undef v_mom_eddy_visc2
#undef h_mom_eddy_visc4
#undef delsq_w
#undef w
#undef rho_edge
#undef rho_zz
#undef divergence
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef tend_w_euler
#undef turb_vflux

// MPAS v8.4.2 mpas_atm_dissipation_models.F:1157-1606
extern "C" __global__ void scalar_dissipation_3d_les_phase_0(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_t_init, float *p_zgrid, float *p_rdzw, float *p_rdzu, float *p_fzm, float *p_fzp, float *p_v_theta_eddy_visc2, float *p_h_theta_eddy_visc4, float *p_prandtl_inv, float *p_prandtl_3d_inv, float *p_delsq_theta, float *p_theta_m, float *p_rho_edge, float *p_rho_zz, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_bv_freq2, float *p_config_len_disp, float *p_scalars, float *p_tend_scalars, int index_tke, int index_qv, int num_scalars_dummy, int mix_scalars, int les_model_opt, int les_surface_opt, float *p_dt, float *p_config_surface_heat_flux, float *p_config_surface_moisture_flux, float *p_ureconstructzonal, float *p_ureconstructmeridional, float *p_hfx, float *p_qfx, float *p_tend_theta_euler, int dynamics_substep, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, float *p_prandtl_1d_inverse, float *p_turb_vflux_scalars, float *p_rho_k_at_w, float *p_zz_at_w, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define t_init(x0,x1) p_t_init[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define v_theta_eddy_visc2 (*p_v_theta_eddy_visc2)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define prandtl_inv (*p_prandtl_inv)
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_theta(x0,x1) p_delsq_theta[(x0-1)+(nvertlevels)*((x1-1))]
#define theta_m(x0,x1) p_theta_m[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define dt (*p_dt)
#define config_surface_heat_flux (*p_config_surface_heat_flux)
#define config_surface_moisture_flux (*p_config_surface_moisture_flux)
#define ureconstructzonal(x0,x1) p_ureconstructzonal[(x0-1)+(nvertlevels)*((x1-1))]
#define ureconstructmeridional(x0,x1) p_ureconstructmeridional[(x0-1)+(nvertlevels)*((x1-1))]
#define hfx(x0) p_hfx[(x0-1)]
#define qfx(x0) p_qfx[(x0-1)]
#define tend_theta_euler(x0,x1) p_tend_theta_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
#define prandtl_1d_inverse(x0) p_prandtl_1d_inverse[tid*((nvertlevels+1))+(x0-1)]
#define turb_vflux_scalars(x0,x1) p_turb_vflux_scalars[tid*((num_scalars)*(nvertlevels+1))+(x0-1)+(num_scalars)*((x1-1))]
#define rho_k_at_w(x0) p_rho_k_at_w[tid*((nvertlevels))+(x0-1)]
#define zz_at_w(x0) p_zz_at_w[tid*((nvertlevels))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
int iscalar;
float r_areacell;
float edge_sign;
float theta_turb_flux;
float pr_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float moisture_flux;
float heat_flux;
float theta_m_flux;
float qv_cell;
float theta_m_cell;
float theta_cell;
icell=tid+cellstart;
if(icell>cellend || !(1)) return;
iscalar=iscalar_arg;
for(k=1;k<=nvertlevels;++k) {
delsq_theta(k,icell) = 0x0.0p+0f;
tend_theta_euler(k,icell) = 0x0.0p+0f;
}
r_areacell = invareacell(icell);
for(i=1;i<=nedgesoncell(icell);++i) {
iedge = edgesoncell(i,icell);
edge_sign = __fmul_rn(__fmul_rn(__fmul_rn(r_areacell,edgesoncell_sign(i,icell)),dvedge(iedge)),invdcedge(iedge));
pr_scale = __fmul_rn(prandtl_inv,meshscalingdel2(iedge));
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
for(k=1;k<=nvertlevels;++k) {
theta_turb_flux = __fmul_rn(__fmul_rn(edge_sign,__fsub_rn(theta_m(k,cell2),theta_m(k,cell1))),rho_edge(k,iedge));
delsq_theta(k,icell) = __fadd_rn(delsq_theta(k,icell),theta_turb_flux);
theta_turb_flux = __fmul_rn(__fmul_rn(__fmul_rn(theta_turb_flux,0x1.0000000000000p-1f),__fadd_rn(eddy_visc_horz(k,cell1),eddy_visc_horz(k,cell2))),pr_scale);
tend_theta_euler(k,icell) = __fadd_rn(tend_theta_euler(k,icell),theta_turb_flux);
}
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef t_init
#undef zgrid
#undef rdzw
#undef rdzu
#undef fzm
#undef fzp
#undef v_theta_eddy_visc2
#undef h_theta_eddy_visc4
#undef prandtl_inv
#undef prandtl_3d_inv
#undef delsq_theta
#undef theta_m
#undef rho_edge
#undef rho_zz
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef bv_freq2
#undef config_len_disp
#undef scalars
#undef tend_scalars
#undef dt
#undef config_surface_heat_flux
#undef config_surface_moisture_flux
#undef ureconstructzonal
#undef ureconstructmeridional
#undef hfx
#undef qfx
#undef tend_theta_euler
#undef turb_vflux
#undef prandtl_1d_inverse
#undef turb_vflux_scalars
#undef rho_k_at_w
#undef zz_at_w

// MPAS v8.4.2 mpas_atm_dissipation_models.F:1157-1606
extern "C" __global__ void scalar_dissipation_3d_les_phase_1(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_t_init, float *p_zgrid, float *p_rdzw, float *p_rdzu, float *p_fzm, float *p_fzp, float *p_v_theta_eddy_visc2, float *p_h_theta_eddy_visc4, float *p_prandtl_inv, float *p_prandtl_3d_inv, float *p_delsq_theta, float *p_theta_m, float *p_rho_edge, float *p_rho_zz, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_bv_freq2, float *p_config_len_disp, float *p_scalars, float *p_tend_scalars, int index_tke, int index_qv, int num_scalars_dummy, int mix_scalars, int les_model_opt, int les_surface_opt, float *p_dt, float *p_config_surface_heat_flux, float *p_config_surface_moisture_flux, float *p_ureconstructzonal, float *p_ureconstructmeridional, float *p_hfx, float *p_qfx, float *p_tend_theta_euler, int dynamics_substep, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, float *p_prandtl_1d_inverse, float *p_turb_vflux_scalars, float *p_rho_k_at_w, float *p_zz_at_w, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define t_init(x0,x1) p_t_init[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define v_theta_eddy_visc2 (*p_v_theta_eddy_visc2)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define prandtl_inv (*p_prandtl_inv)
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_theta(x0,x1) p_delsq_theta[(x0-1)+(nvertlevels)*((x1-1))]
#define theta_m(x0,x1) p_theta_m[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define dt (*p_dt)
#define config_surface_heat_flux (*p_config_surface_heat_flux)
#define config_surface_moisture_flux (*p_config_surface_moisture_flux)
#define ureconstructzonal(x0,x1) p_ureconstructzonal[(x0-1)+(nvertlevels)*((x1-1))]
#define ureconstructmeridional(x0,x1) p_ureconstructmeridional[(x0-1)+(nvertlevels)*((x1-1))]
#define hfx(x0) p_hfx[(x0-1)]
#define qfx(x0) p_qfx[(x0-1)]
#define tend_theta_euler(x0,x1) p_tend_theta_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
#define prandtl_1d_inverse(x0) p_prandtl_1d_inverse[tid*((nvertlevels+1))+(x0-1)]
#define turb_vflux_scalars(x0,x1) p_turb_vflux_scalars[tid*((num_scalars)*(nvertlevels+1))+(x0-1)+(num_scalars)*((x1-1))]
#define rho_k_at_w(x0) p_rho_k_at_w[tid*((nvertlevels))+(x0-1)]
#define zz_at_w(x0) p_zz_at_w[tid*((nvertlevels))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
int iscalar;
float r_areacell;
float edge_sign;
float theta_turb_flux;
float pr_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float moisture_flux;
float heat_flux;
float theta_m_flux;
float qv_cell;
float theta_m_cell;
float theta_cell;
icell=tid+cellsolvestart;
if(icell>cellsolveend || !((1) && ((h_theta_eddy_visc4>0x0.0p+0f)))) return;
iscalar=iscalar_arg;
r_areacell = __fmul_rn(__fmul_rn(h_theta_eddy_visc4,prandtl_inv),invareacell(icell));
for(i=1;i<=nedgesoncell(icell);++i) {
iedge = edgesoncell(i,icell);
edge_sign = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(meshscalingdel4(iedge),r_areacell),dvedge(iedge)),edgesoncell_sign(i,icell)),invdcedge(iedge));
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
for(k=1;k<=nvertlevels;++k) {
tend_theta_euler(k,icell) = __fsub_rn(tend_theta_euler(k,icell),__fmul_rn(edge_sign,__fsub_rn(delsq_theta(k,cell2),delsq_theta(k,cell1))));
}
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef t_init
#undef zgrid
#undef rdzw
#undef rdzu
#undef fzm
#undef fzp
#undef v_theta_eddy_visc2
#undef h_theta_eddy_visc4
#undef prandtl_inv
#undef prandtl_3d_inv
#undef delsq_theta
#undef theta_m
#undef rho_edge
#undef rho_zz
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef bv_freq2
#undef config_len_disp
#undef scalars
#undef tend_scalars
#undef dt
#undef config_surface_heat_flux
#undef config_surface_moisture_flux
#undef ureconstructzonal
#undef ureconstructmeridional
#undef hfx
#undef qfx
#undef tend_theta_euler
#undef turb_vflux
#undef prandtl_1d_inverse
#undef turb_vflux_scalars
#undef rho_k_at_w
#undef zz_at_w

// MPAS v8.4.2 mpas_atm_dissipation_models.F:1157-1606
extern "C" __global__ void scalar_dissipation_3d_les_phase_2(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_t_init, float *p_zgrid, float *p_rdzw, float *p_rdzu, float *p_fzm, float *p_fzp, float *p_v_theta_eddy_visc2, float *p_h_theta_eddy_visc4, float *p_prandtl_inv, float *p_prandtl_3d_inv, float *p_delsq_theta, float *p_theta_m, float *p_rho_edge, float *p_rho_zz, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_bv_freq2, float *p_config_len_disp, float *p_scalars, float *p_tend_scalars, int index_tke, int index_qv, int num_scalars_dummy, int mix_scalars, int les_model_opt, int les_surface_opt, float *p_dt, float *p_config_surface_heat_flux, float *p_config_surface_moisture_flux, float *p_ureconstructzonal, float *p_ureconstructmeridional, float *p_hfx, float *p_qfx, float *p_tend_theta_euler, int dynamics_substep, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, float *p_prandtl_1d_inverse, float *p_turb_vflux_scalars, float *p_rho_k_at_w, float *p_zz_at_w, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define t_init(x0,x1) p_t_init[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define v_theta_eddy_visc2 (*p_v_theta_eddy_visc2)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define prandtl_inv (*p_prandtl_inv)
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_theta(x0,x1) p_delsq_theta[(x0-1)+(nvertlevels)*((x1-1))]
#define theta_m(x0,x1) p_theta_m[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define dt (*p_dt)
#define config_surface_heat_flux (*p_config_surface_heat_flux)
#define config_surface_moisture_flux (*p_config_surface_moisture_flux)
#define ureconstructzonal(x0,x1) p_ureconstructzonal[(x0-1)+(nvertlevels)*((x1-1))]
#define ureconstructmeridional(x0,x1) p_ureconstructmeridional[(x0-1)+(nvertlevels)*((x1-1))]
#define hfx(x0) p_hfx[(x0-1)]
#define qfx(x0) p_qfx[(x0-1)]
#define tend_theta_euler(x0,x1) p_tend_theta_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
#define prandtl_1d_inverse(x0) p_prandtl_1d_inverse[tid*((nvertlevels+1))+(x0-1)]
#define turb_vflux_scalars(x0,x1) p_turb_vflux_scalars[tid*((num_scalars)*(nvertlevels+1))+(x0-1)+(num_scalars)*((x1-1))]
#define rho_k_at_w(x0) p_rho_k_at_w[tid*((nvertlevels))+(x0-1)]
#define zz_at_w(x0) p_zz_at_w[tid*((nvertlevels))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
int iscalar;
float r_areacell;
float edge_sign;
float theta_turb_flux;
float pr_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float moisture_flux;
float heat_flux;
float theta_m_flux;
float qv_cell;
float theta_m_cell;
float theta_cell;
icell=tid+cellstart;
if(icell>cellend || !((1) && ((mix_scalars && (dynamics_substep==1))))) return;
iscalar=iscalar_arg;
for(k=1;k<=nvertlevels;++k) {
delsq_theta(k,icell) = 0x0.0p+0f;
}
r_areacell = invareacell(icell);
for(i=1;i<=nedgesoncell(icell);++i) {
iedge = edgesoncell(i,icell);
edge_sign = __fmul_rn(__fmul_rn(__fmul_rn(r_areacell,edgesoncell_sign(i,icell)),dvedge(iedge)),invdcedge(iedge));
pr_scale = __fmul_rn(prandtl_inv,meshscalingdel2(iedge));
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
for(k=1;k<=nvertlevels;++k) {
theta_turb_flux = __fmul_rn(__fmul_rn(edge_sign,__fsub_rn(scalars(iscalar,k,cell2),scalars(iscalar,k,cell1))),rho_edge(k,iedge));
delsq_theta(k,icell) = __fadd_rn(delsq_theta(k,icell),theta_turb_flux);
theta_turb_flux = __fmul_rn(__fmul_rn(__fmul_rn(theta_turb_flux,0x1.0000000000000p-1f),__fadd_rn(eddy_visc_horz(k,cell1),eddy_visc_horz(k,cell2))),pr_scale);
tend_scalars(iscalar,k,icell) = __fadd_rn(tend_scalars(iscalar,k,icell),theta_turb_flux);
}
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef t_init
#undef zgrid
#undef rdzw
#undef rdzu
#undef fzm
#undef fzp
#undef v_theta_eddy_visc2
#undef h_theta_eddy_visc4
#undef prandtl_inv
#undef prandtl_3d_inv
#undef delsq_theta
#undef theta_m
#undef rho_edge
#undef rho_zz
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef bv_freq2
#undef config_len_disp
#undef scalars
#undef tend_scalars
#undef dt
#undef config_surface_heat_flux
#undef config_surface_moisture_flux
#undef ureconstructzonal
#undef ureconstructmeridional
#undef hfx
#undef qfx
#undef tend_theta_euler
#undef turb_vflux
#undef prandtl_1d_inverse
#undef turb_vflux_scalars
#undef rho_k_at_w
#undef zz_at_w

// MPAS v8.4.2 mpas_atm_dissipation_models.F:1157-1606
extern "C" __global__ void scalar_dissipation_3d_les_phase_3(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_t_init, float *p_zgrid, float *p_rdzw, float *p_rdzu, float *p_fzm, float *p_fzp, float *p_v_theta_eddy_visc2, float *p_h_theta_eddy_visc4, float *p_prandtl_inv, float *p_prandtl_3d_inv, float *p_delsq_theta, float *p_theta_m, float *p_rho_edge, float *p_rho_zz, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_bv_freq2, float *p_config_len_disp, float *p_scalars, float *p_tend_scalars, int index_tke, int index_qv, int num_scalars_dummy, int mix_scalars, int les_model_opt, int les_surface_opt, float *p_dt, float *p_config_surface_heat_flux, float *p_config_surface_moisture_flux, float *p_ureconstructzonal, float *p_ureconstructmeridional, float *p_hfx, float *p_qfx, float *p_tend_theta_euler, int dynamics_substep, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, float *p_prandtl_1d_inverse, float *p_turb_vflux_scalars, float *p_rho_k_at_w, float *p_zz_at_w, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define t_init(x0,x1) p_t_init[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define v_theta_eddy_visc2 (*p_v_theta_eddy_visc2)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define prandtl_inv (*p_prandtl_inv)
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_theta(x0,x1) p_delsq_theta[(x0-1)+(nvertlevels)*((x1-1))]
#define theta_m(x0,x1) p_theta_m[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define dt (*p_dt)
#define config_surface_heat_flux (*p_config_surface_heat_flux)
#define config_surface_moisture_flux (*p_config_surface_moisture_flux)
#define ureconstructzonal(x0,x1) p_ureconstructzonal[(x0-1)+(nvertlevels)*((x1-1))]
#define ureconstructmeridional(x0,x1) p_ureconstructmeridional[(x0-1)+(nvertlevels)*((x1-1))]
#define hfx(x0) p_hfx[(x0-1)]
#define qfx(x0) p_qfx[(x0-1)]
#define tend_theta_euler(x0,x1) p_tend_theta_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
#define prandtl_1d_inverse(x0) p_prandtl_1d_inverse[tid*((nvertlevels+1))+(x0-1)]
#define turb_vflux_scalars(x0,x1) p_turb_vflux_scalars[tid*((num_scalars)*(nvertlevels+1))+(x0-1)+(num_scalars)*((x1-1))]
#define rho_k_at_w(x0) p_rho_k_at_w[tid*((nvertlevels))+(x0-1)]
#define zz_at_w(x0) p_zz_at_w[tid*((nvertlevels))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
int iscalar;
float r_areacell;
float edge_sign;
float theta_turb_flux;
float pr_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float moisture_flux;
float heat_flux;
float theta_m_flux;
float qv_cell;
float theta_m_cell;
float theta_cell;
icell=tid+cellsolvestart;
if(icell>cellsolveend || !(((1) && ((mix_scalars && (dynamics_substep==1)))) && ((h_theta_eddy_visc4>0x0.0p+0f)))) return;
iscalar=iscalar_arg;
r_areacell = __fmul_rn(__fmul_rn(h_theta_eddy_visc4,prandtl_inv),invareacell(icell));
for(i=1;i<=nedgesoncell(icell);++i) {
iedge = edgesoncell(i,icell);
edge_sign = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(meshscalingdel4(iedge),r_areacell),dvedge(iedge)),edgesoncell_sign(i,icell)),invdcedge(iedge));
cell1 = cellsonedge(1,iedge);
cell2 = cellsonedge(2,iedge);
for(k=1;k<=nvertlevels;++k) {
tend_scalars(iscalar,k,icell) = __fsub_rn(tend_scalars(iscalar,k,icell),__fmul_rn(edge_sign,__fsub_rn(delsq_theta(k,cell2),delsq_theta(k,cell1))));
}
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef t_init
#undef zgrid
#undef rdzw
#undef rdzu
#undef fzm
#undef fzp
#undef v_theta_eddy_visc2
#undef h_theta_eddy_visc4
#undef prandtl_inv
#undef prandtl_3d_inv
#undef delsq_theta
#undef theta_m
#undef rho_edge
#undef rho_zz
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef bv_freq2
#undef config_len_disp
#undef scalars
#undef tend_scalars
#undef dt
#undef config_surface_heat_flux
#undef config_surface_moisture_flux
#undef ureconstructzonal
#undef ureconstructmeridional
#undef hfx
#undef qfx
#undef tend_theta_euler
#undef turb_vflux
#undef prandtl_1d_inverse
#undef turb_vflux_scalars
#undef rho_k_at_w
#undef zz_at_w

// MPAS v8.4.2 mpas_atm_dissipation_models.F:1157-1606
extern "C" __global__ void scalar_dissipation_3d_les_phase_4(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_t_init, float *p_zgrid, float *p_rdzw, float *p_rdzu, float *p_fzm, float *p_fzp, float *p_v_theta_eddy_visc2, float *p_h_theta_eddy_visc4, float *p_prandtl_inv, float *p_prandtl_3d_inv, float *p_delsq_theta, float *p_theta_m, float *p_rho_edge, float *p_rho_zz, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_bv_freq2, float *p_config_len_disp, float *p_scalars, float *p_tend_scalars, int index_tke, int index_qv, int num_scalars_dummy, int mix_scalars, int les_model_opt, int les_surface_opt, float *p_dt, float *p_config_surface_heat_flux, float *p_config_surface_moisture_flux, float *p_ureconstructzonal, float *p_ureconstructmeridional, float *p_hfx, float *p_qfx, float *p_tend_theta_euler, int dynamics_substep, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, float *p_prandtl_1d_inverse, float *p_turb_vflux_scalars, float *p_rho_k_at_w, float *p_zz_at_w, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define t_init(x0,x1) p_t_init[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define v_theta_eddy_visc2 (*p_v_theta_eddy_visc2)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define prandtl_inv (*p_prandtl_inv)
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_theta(x0,x1) p_delsq_theta[(x0-1)+(nvertlevels)*((x1-1))]
#define theta_m(x0,x1) p_theta_m[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define dt (*p_dt)
#define config_surface_heat_flux (*p_config_surface_heat_flux)
#define config_surface_moisture_flux (*p_config_surface_moisture_flux)
#define ureconstructzonal(x0,x1) p_ureconstructzonal[(x0-1)+(nvertlevels)*((x1-1))]
#define ureconstructmeridional(x0,x1) p_ureconstructmeridional[(x0-1)+(nvertlevels)*((x1-1))]
#define hfx(x0) p_hfx[(x0-1)]
#define qfx(x0) p_qfx[(x0-1)]
#define tend_theta_euler(x0,x1) p_tend_theta_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
#define prandtl_1d_inverse(x0) p_prandtl_1d_inverse[tid*((nvertlevels+1))+(x0-1)]
#define turb_vflux_scalars(x0,x1) p_turb_vflux_scalars[tid*((num_scalars)*(nvertlevels+1))+(x0-1)+(num_scalars)*((x1-1))]
#define rho_k_at_w(x0) p_rho_k_at_w[tid*((nvertlevels))+(x0-1)]
#define zz_at_w(x0) p_zz_at_w[tid*((nvertlevels))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
int iscalar;
float r_areacell;
float edge_sign;
float theta_turb_flux;
float pr_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float moisture_flux;
float heat_flux;
float theta_m_flux;
float qv_cell;
float theta_m_cell;
float theta_cell;
icell=tid+cellsolvestart;
if(icell>cellsolveend || !(((1) && ((v_theta_eddy_visc2>0x0.0p+0f))) && (config_mix_full))) return;
iscalar=iscalar_arg;
for(k=2;k<=(nvertlevels-1);++k) {
z1 = zgrid((k-1),icell);
z2 = zgrid(k,icell);
z3 = zgrid((k+1),icell);
z4 = zgrid((k+2),icell);
zm = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z1,z2));
z0 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z2,z3));
zp = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z3,z4));
tend_theta_euler(k,icell) = __fadd_rn(tend_theta_euler(k,icell),__fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(v_theta_eddy_visc2,prandtl_inv),rho_zz(k,icell)),__fsub_rn(__fdiv_rn(__fsub_rn(theta_m((k+1),icell),theta_m(k,icell)),__fsub_rn(zp,z0)),__fdiv_rn(__fsub_rn(theta_m(k,icell),theta_m((k-1),icell)),__fsub_rn(z0,zm)))),__fmul_rn(0x1.0000000000000p-1f,__fsub_rn(zp,zm))));
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef t_init
#undef zgrid
#undef rdzw
#undef rdzu
#undef fzm
#undef fzp
#undef v_theta_eddy_visc2
#undef h_theta_eddy_visc4
#undef prandtl_inv
#undef prandtl_3d_inv
#undef delsq_theta
#undef theta_m
#undef rho_edge
#undef rho_zz
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef bv_freq2
#undef config_len_disp
#undef scalars
#undef tend_scalars
#undef dt
#undef config_surface_heat_flux
#undef config_surface_moisture_flux
#undef ureconstructzonal
#undef ureconstructmeridional
#undef hfx
#undef qfx
#undef tend_theta_euler
#undef turb_vflux
#undef prandtl_1d_inverse
#undef turb_vflux_scalars
#undef rho_k_at_w
#undef zz_at_w

// MPAS v8.4.2 mpas_atm_dissipation_models.F:1157-1606
extern "C" __global__ void scalar_dissipation_3d_les_phase_5(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_t_init, float *p_zgrid, float *p_rdzw, float *p_rdzu, float *p_fzm, float *p_fzp, float *p_v_theta_eddy_visc2, float *p_h_theta_eddy_visc4, float *p_prandtl_inv, float *p_prandtl_3d_inv, float *p_delsq_theta, float *p_theta_m, float *p_rho_edge, float *p_rho_zz, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_bv_freq2, float *p_config_len_disp, float *p_scalars, float *p_tend_scalars, int index_tke, int index_qv, int num_scalars_dummy, int mix_scalars, int les_model_opt, int les_surface_opt, float *p_dt, float *p_config_surface_heat_flux, float *p_config_surface_moisture_flux, float *p_ureconstructzonal, float *p_ureconstructmeridional, float *p_hfx, float *p_qfx, float *p_tend_theta_euler, int dynamics_substep, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, float *p_prandtl_1d_inverse, float *p_turb_vflux_scalars, float *p_rho_k_at_w, float *p_zz_at_w, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define t_init(x0,x1) p_t_init[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define v_theta_eddy_visc2 (*p_v_theta_eddy_visc2)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define prandtl_inv (*p_prandtl_inv)
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_theta(x0,x1) p_delsq_theta[(x0-1)+(nvertlevels)*((x1-1))]
#define theta_m(x0,x1) p_theta_m[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define dt (*p_dt)
#define config_surface_heat_flux (*p_config_surface_heat_flux)
#define config_surface_moisture_flux (*p_config_surface_moisture_flux)
#define ureconstructzonal(x0,x1) p_ureconstructzonal[(x0-1)+(nvertlevels)*((x1-1))]
#define ureconstructmeridional(x0,x1) p_ureconstructmeridional[(x0-1)+(nvertlevels)*((x1-1))]
#define hfx(x0) p_hfx[(x0-1)]
#define qfx(x0) p_qfx[(x0-1)]
#define tend_theta_euler(x0,x1) p_tend_theta_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
#define prandtl_1d_inverse(x0) p_prandtl_1d_inverse[tid*((nvertlevels+1))+(x0-1)]
#define turb_vflux_scalars(x0,x1) p_turb_vflux_scalars[tid*((num_scalars)*(nvertlevels+1))+(x0-1)+(num_scalars)*((x1-1))]
#define rho_k_at_w(x0) p_rho_k_at_w[tid*((nvertlevels))+(x0-1)]
#define zz_at_w(x0) p_zz_at_w[tid*((nvertlevels))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
int iscalar;
float r_areacell;
float edge_sign;
float theta_turb_flux;
float pr_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float moisture_flux;
float heat_flux;
float theta_m_flux;
float qv_cell;
float theta_m_cell;
float theta_cell;
icell=tid+cellsolvestart;
if(icell>cellsolveend || !(((1) && ((v_theta_eddy_visc2>0x0.0p+0f))) && !(config_mix_full) && (1))) return;
iscalar=iscalar_arg;
for(k=2;k<=(nvertlevels-1);++k) {
z1 = zgrid((k-1),icell);
z2 = zgrid(k,icell);
z3 = zgrid((k+1),icell);
z4 = zgrid((k+2),icell);
zm = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z1,z2));
z0 = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z2,z3));
zp = __fmul_rn(0x1.0000000000000p-1f,__fadd_rn(z3,z4));
tend_theta_euler(k,icell) = __fadd_rn(tend_theta_euler(k,icell),__fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(v_theta_eddy_visc2,prandtl_inv),rho_zz(k,icell)),__fsub_rn(__fdiv_rn(__fsub_rn(__fsub_rn(theta_m((k+1),icell),t_init((k+1),icell)),__fsub_rn(theta_m(k,icell),t_init(k,icell))),__fsub_rn(zp,z0)),__fdiv_rn(__fsub_rn(__fsub_rn(theta_m(k,icell),t_init(k,icell)),__fsub_rn(theta_m((k-1),icell),t_init((k-1),icell))),__fsub_rn(z0,zm)))),__fmul_rn(0x1.0000000000000p-1f,__fsub_rn(zp,zm))));
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef t_init
#undef zgrid
#undef rdzw
#undef rdzu
#undef fzm
#undef fzp
#undef v_theta_eddy_visc2
#undef h_theta_eddy_visc4
#undef prandtl_inv
#undef prandtl_3d_inv
#undef delsq_theta
#undef theta_m
#undef rho_edge
#undef rho_zz
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef bv_freq2
#undef config_len_disp
#undef scalars
#undef tend_scalars
#undef dt
#undef config_surface_heat_flux
#undef config_surface_moisture_flux
#undef ureconstructzonal
#undef ureconstructmeridional
#undef hfx
#undef qfx
#undef tend_theta_euler
#undef turb_vflux
#undef prandtl_1d_inverse
#undef turb_vflux_scalars
#undef rho_k_at_w
#undef zz_at_w

// MPAS v8.4.2 mpas_atm_dissipation_models.F:1157-1606
extern "C" __global__ void scalar_dissipation_3d_les_phase_6(int cellstart, int cellend, int cellsolvestart, int cellsolveend, int ncells, int nedges, int *p_nedgesoncell, int *p_edgesoncell, int *p_cellsonedge, float *p_edgesoncell_sign, float *p_invareacell, float *p_invdcedge, float *p_dvedge, float *p_meshscalingdel2, float *p_meshscalingdel4, int config_mix_full, float *p_t_init, float *p_zgrid, float *p_rdzw, float *p_rdzu, float *p_fzm, float *p_fzp, float *p_v_theta_eddy_visc2, float *p_h_theta_eddy_visc4, float *p_prandtl_inv, float *p_prandtl_3d_inv, float *p_delsq_theta, float *p_theta_m, float *p_rho_edge, float *p_rho_zz, float *p_zz, float *p_eddy_visc_horz, float *p_eddy_visc_vert, float *p_bv_freq2, float *p_config_len_disp, float *p_scalars, float *p_tend_scalars, int index_tke, int index_qv, int num_scalars_dummy, int mix_scalars, int les_model_opt, int les_surface_opt, float *p_dt, float *p_config_surface_heat_flux, float *p_config_surface_moisture_flux, float *p_ureconstructzonal, float *p_ureconstructmeridional, float *p_hfx, float *p_qfx, float *p_tend_theta_euler, int dynamics_substep, int nvertlevels, int maxedges, int num_scalars, float *p_turb_vflux, float *p_prandtl_1d_inverse, float *p_turb_vflux_scalars, float *p_rho_k_at_w, float *p_zz_at_w, int iscalar_arg) {
const int tid=blockIdx.x*blockDim.x+threadIdx.x;
#define nedgesoncell(x0) p_nedgesoncell[(x0-1)]
#define edgesoncell(x0,x1) p_edgesoncell[(x0-1)+(maxedges)*((x1-1))]
#define cellsonedge(x0,x1) p_cellsonedge[(x0-1)+(2)*((x1-1))]
#define edgesoncell_sign(x0,x1) p_edgesoncell_sign[(x0-1)+(maxedges)*((x1-1))]
#define invareacell(x0) p_invareacell[(x0-1)]
#define invdcedge(x0) p_invdcedge[(x0-1)]
#define dvedge(x0) p_dvedge[(x0-1)]
#define meshscalingdel2(x0) p_meshscalingdel2[(x0-1)]
#define meshscalingdel4(x0) p_meshscalingdel4[(x0-1)]
#define t_init(x0,x1) p_t_init[(x0-1)+(nvertlevels)*((x1-1))]
#define zgrid(x0,x1) p_zgrid[(x0-1)+(nvertlevels+1)*((x1-1))]
#define rdzw(x0) p_rdzw[(x0-1)]
#define rdzu(x0) p_rdzu[(x0-1)]
#define fzm(x0) p_fzm[(x0-1)]
#define fzp(x0) p_fzp[(x0-1)]
#define v_theta_eddy_visc2 (*p_v_theta_eddy_visc2)
#define h_theta_eddy_visc4 (*p_h_theta_eddy_visc4)
#define prandtl_inv (*p_prandtl_inv)
#define prandtl_3d_inv(x0,x1) p_prandtl_3d_inv[(x0-1)+(nvertlevels)*((x1-1))]
#define delsq_theta(x0,x1) p_delsq_theta[(x0-1)+(nvertlevels)*((x1-1))]
#define theta_m(x0,x1) p_theta_m[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_edge(x0,x1) p_rho_edge[(x0-1)+(nvertlevels)*((x1-1))]
#define rho_zz(x0,x1) p_rho_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define zz(x0,x1) p_zz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_horz(x0,x1) p_eddy_visc_horz[(x0-1)+(nvertlevels)*((x1-1))]
#define eddy_visc_vert(x0,x1) p_eddy_visc_vert[(x0-1)+(nvertlevels)*((x1-1))]
#define bv_freq2(x0,x1) p_bv_freq2[(x0-1)+(nvertlevels)*((x1-1))]
#define config_len_disp (*p_config_len_disp)
#define scalars(x0,x1,x2) p_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define tend_scalars(x0,x1,x2) p_tend_scalars[(x0-1)+(num_scalars)*((x1-1)+(nvertlevels)*((x2-1)))]
#define dt (*p_dt)
#define config_surface_heat_flux (*p_config_surface_heat_flux)
#define config_surface_moisture_flux (*p_config_surface_moisture_flux)
#define ureconstructzonal(x0,x1) p_ureconstructzonal[(x0-1)+(nvertlevels)*((x1-1))]
#define ureconstructmeridional(x0,x1) p_ureconstructmeridional[(x0-1)+(nvertlevels)*((x1-1))]
#define hfx(x0) p_hfx[(x0-1)]
#define qfx(x0) p_qfx[(x0-1)]
#define tend_theta_euler(x0,x1) p_tend_theta_euler[(x0-1)+(nvertlevels)*((x1-1))]
#define turb_vflux(x0) p_turb_vflux[tid*((nvertlevels+1))+(x0-1)]
#define prandtl_1d_inverse(x0) p_prandtl_1d_inverse[tid*((nvertlevels+1))+(x0-1)]
#define turb_vflux_scalars(x0,x1) p_turb_vflux_scalars[tid*((num_scalars)*(nvertlevels+1))+(x0-1)+(num_scalars)*((x1-1))]
#define rho_k_at_w(x0) p_rho_k_at_w[tid*((nvertlevels))+(x0-1)]
#define zz_at_w(x0) p_zz_at_w[tid*((nvertlevels))+(x0-1)]
int cell1;
int cell2;
int iedge;
int icell;
int i;
int k;
int iscalar;
float r_areacell;
float edge_sign;
float theta_turb_flux;
float pr_scale;
float z1;
float z2;
float z3;
float z4;
float zm;
float z0;
float zp;
float moisture_flux;
float heat_flux;
float theta_m_flux;
float qv_cell;
float theta_m_cell;
float theta_cell;
icell=tid+cellsolvestart;
if(icell>cellsolveend || !((1) && ((les_model_opt!=0)))) return;
iscalar=iscalar_arg;
turb_vflux((nvertlevels+1)) = 0x0.0p+0f;
turb_vflux(1) = 0x0.0p+0f;
if((les_model_opt==1)) {
for(k=2;k<=nvertlevels;++k) {
prandtl_1d_inverse(k) = prandtl_inv;
}
} else if(1) {
for(k=2;k<=nvertlevels;++k) {
prandtl_1d_inverse(k) = __fadd_rn(__fmul_rn(fzm(k),prandtl_3d_inv(k,icell)),__fmul_rn(fzp(k),prandtl_3d_inv((k-1),icell)));
}
}
for(k=2;k<=nvertlevels;++k) {
rho_k_at_w(k) = __fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(fzm(k),rho_zz(k,icell)),zz(k,icell)),zz(k,icell)),eddy_visc_vert(k,icell)),__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(fzp(k),rho_zz((k-1),icell)),zz((k-1),icell)),zz((k-1),icell)),eddy_visc_vert((k-1),icell)));
zz_at_w(k) = __fadd_rn(__fmul_rn(fzm(k),zz(k,icell)),__fmul_rn(fzp(k),zz((k-1),icell)));
turb_vflux(k) = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-prandtl_1d_inverse(k)),rho_k_at_w(k)),zz_at_w(k)),rdzu(k)),__fsub_rn(theta_m(k,icell),theta_m((k-1),icell)));
}
if(((les_surface_opt==1) || (les_surface_opt==2))) {
if((les_surface_opt==1)) {
moisture_flux = config_surface_moisture_flux;
heat_flux = config_surface_heat_flux;
} else if((les_surface_opt==2)) {
heat_flux = __fdiv_rn(__fdiv_rn(hfx(icell),rho_zz(1,icell)),0x1.f640000000000p+9f);
moisture_flux = __fdiv_rn(qfx(icell),rho_zz(1,icell));
}
qv_cell = scalars(index_qv,1,icell);
theta_m_cell = theta_m(1,icell);
theta_cell = __fdiv_rn(theta_m_cell,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.9bbda40000000p+0f,qv_cell)));
theta_m_flux = __fadd_rn(__fmul_rn(heat_flux,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.9bbda40000000p+0f,qv_cell))),__fmul_rn(__fmul_rn(0x1.9bbda40000000p+0f,theta_cell),moisture_flux));
turb_vflux(1) = __fmul_rn(theta_m_flux,rho_zz(1,icell));
moisture_flux = __fmul_rn(moisture_flux,rho_zz(1,icell));
turb_vflux((nvertlevels+1)) = turb_vflux(nvertlevels);
} else if(1) {
turb_vflux(1) = turb_vflux(2);
turb_vflux((nvertlevels+1)) = turb_vflux(nvertlevels);
}
for(k=1;k<=nvertlevels;++k) {
tend_theta_euler(k,icell) = __fsub_rn(tend_theta_euler(k,icell),__fmul_rn(rdzw(k),__fsub_rn(turb_vflux((k+1)),turb_vflux(k))));
}
if(mix_scalars) {
for(iscalar=1;iscalar<=num_scalars;++iscalar) {
turb_vflux_scalars(iscalar,(nvertlevels+1)) = 0x0.0p+0f;
turb_vflux_scalars(iscalar,1) = 0x0.0p+0f;
}
for(k=2;k<=nvertlevels;++k) {
rho_k_at_w(k) = __fadd_rn(__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(fzm(k),rho_zz(k,icell)),zz(k,icell)),zz(k,icell)),eddy_visc_vert(k,icell)),__fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(fzp(k),rho_zz((k-1),icell)),zz((k-1),icell)),zz((k-1),icell)),eddy_visc_vert((k-1),icell)));
zz_at_w(k) = __fadd_rn(__fmul_rn(fzm(k),zz(k,icell)),__fmul_rn(fzp(k),zz((k-1),icell)));
}
for(k=2;k<=nvertlevels;++k) {
for(iscalar=1;iscalar<=num_scalars;++iscalar) {
turb_vflux_scalars(iscalar,k) = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn((-prandtl_1d_inverse(k)),rho_k_at_w(k)),zz_at_w(k)),rdzu(k)),__fsub_rn(scalars(iscalar,k,icell),scalars(iscalar,(k-1),icell)));
}
}
if(((les_surface_opt==1) || (les_surface_opt==2))) {
turb_vflux_scalars(index_qv,1) = moisture_flux;
}
for(k=1;k<=nvertlevels;++k) {
for(iscalar=1;iscalar<=num_scalars;++iscalar) {
tend_scalars(iscalar,k,icell) = __fsub_rn(tend_scalars(iscalar,k,icell),__fmul_rn(rdzw(k),__fsub_rn(turb_vflux_scalars(iscalar,(k+1)),turb_vflux_scalars(iscalar,k))));
}
}
}
}
#undef nedgesoncell
#undef edgesoncell
#undef cellsonedge
#undef edgesoncell_sign
#undef invareacell
#undef invdcedge
#undef dvedge
#undef meshscalingdel2
#undef meshscalingdel4
#undef t_init
#undef zgrid
#undef rdzw
#undef rdzu
#undef fzm
#undef fzp
#undef v_theta_eddy_visc2
#undef h_theta_eddy_visc4
#undef prandtl_inv
#undef prandtl_3d_inv
#undef delsq_theta
#undef theta_m
#undef rho_edge
#undef rho_zz
#undef zz
#undef eddy_visc_horz
#undef eddy_visc_vert
#undef bv_freq2
#undef config_len_disp
#undef scalars
#undef tend_scalars
#undef dt
#undef config_surface_heat_flux
#undef config_surface_moisture_flux
#undef ureconstructzonal
#undef ureconstructmeridional
#undef hfx
#undef qfx
#undef tend_theta_euler
#undef turb_vflux
#undef prandtl_1d_inverse
#undef turb_vflux_scalars
#undef rho_k_at_w
#undef zz_at_w
"""


def _extent(expression: str, arguments: Mapping[str, Any]) -> int:
    """Evaluate the fixed native dimension grammar without eval()."""
    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return int(arguments[node.id])
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            op = operator.add if isinstance(node.op, ast.Add) else operator.mul
            return op(visit(node.left), visit(node.right))
        raise ValueError(f"invalid native dimension {expression!r}")
    return visit(ast.parse(expression, mode="eval").body)


def _kernel_names():
    def walk(plan):
        for phase in plan:
            if "kernel" in phase:
                yield phase["kernel"]
            else:
                yield from walk(phase["body"])
    for plan in PHASES.values():
        yield from walk(plan)


CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                "--prec-div=true", "--prec-sqrt=true")


def _cuda_source(prognostic_viscosity_floor=True):
    import woof
    headers = Path(woof.__file__).parent / "core" / "kernels"
    math_source = (headers / "glibc_flt32.cuh").read_text()
    math_source += "\n" + (headers / "glibc_trig_flt32.cuh").read_text()
    policy = '' if prognostic_viscosity_floor else '#define MPAS_LES_NO_VISCOSITY_FLOOR\n'
    return policy + math_source + "\n" + CUDA_SOURCE


@lru_cache(maxsize=None)
def _module(device_id: int, prognostic_viscosity_floor=True):
    import cupy as cp
    from cupy.cuda import compiler
    # CuPy RawModule appends forced FTZ after caller options, erasing native
    # subnormal tendencies. Direct NVRTC preserves the explicit strict flags.
    with cp.cuda.Device(device_id):
        binary, _ = compiler.compile_using_nvrtc(
            _cuda_source(prognostic_viscosity_floor), options=_hex_numerics.options(CUDA_OPTIONS),
            arch=cp.cuda.Device(device_id).compute_capability)
        module = cp.cuda.function.Module()
        module.load(binary)
        return module


def _surface_extent(name, values):
    """Native assumed-shape reads: mpas_atm_dissipation_models.F:919,1533-1534.

    Native physics-off calls pass length-one unused flux/friction arrays.
    Only the varying surface reads these arguments; no sentinel is read by
    the cell flux loop. Edge friction uses the native cell connectivity.
    """
    if int(values['les_model_opt']) not in (1, 2) or int(values['les_surface_opt']) != 2:
        return 0
    return int(values['ncells'] if name == 'ustm' else values['cellend'])


class LesDissipationV842:
    """Resident native-interface launcher with reusable private scratch."""

    def __init__(self, *, prognostic_viscosity_floor=True):
        self._scratch = {}
        self._constants = {}
        self.prognostic_viscosity_floor = bool(prognostic_viscosity_floor)

    def call(self, routine: str, /, **arguments):
        """Launch one complete routine; mutate the caller's device outputs.

        MPAS citations for each routine are in ROUTINES and CUDA_SOURCE.
        Keys accept the original Fortran capitalization. Supply nVertLevels,
        maxEdges and num_scalars explicitly even when MPAS imports them from
        mpas_atm_dimensions. The inert clock argument is omitted: its only
        use in pristine MPAS is in commented-out flux_les_sas code.
        """
        import cupy as cp
        spec = ROUTINES[routine]
        values = {name.lower(): value for name, value in arguments.items()}
        if len(values) != len(arguments):
            raise ValueError("duplicate argument names after Fortran case folding")
        missing = set(spec["args"]) - values.keys()
        extra = values.keys() - set(spec["args"])
        if missing or extra:
            raise ValueError(f"{routine}: missing {sorted(missing)}, extra {sorted(extra)}")
        for name in ("nvertlevels", "maxedges", "num_scalars"):
            if int(values[name]) < (3 if name == "nvertlevels" else 1):
                raise ValueError(f"{name} is below the native stencil minimum")
        packed = []
        for name in spec["args"]:
            field = spec["fields"][name]
            value = values[name]
            is_array = bool(field["dims"])
            is_output_scalar = not is_array and field["type"] == "float"
            dtype = np.dtype(np.float32 if field["type"] == "float" else np.int32)
            if not is_array and field["type"] == "float" and field["intent"] == "in" and not isinstance(value, cp.ndarray):
                scalar = np.float32(value)
                key = (cp.cuda.runtime.getDevice(), cp.cuda.get_current_stream().ptr,
                       name, scalar.tobytes())
                if key not in self._constants:
                    self._constants[key] = cp.asarray(np.array([scalar], dtype=np.float32))
                value = self._constants[key]
            if is_array or is_output_scalar:
                if not isinstance(value, cp.ndarray) or value.dtype != dtype:
                    raise TypeError(f"{name} requires a device {dtype} array")
                if not value.flags.f_contiguous:
                    raise ValueError(f"{name} must use native Fortran contiguous storage")
                expected = tuple(_extent(d, values) for d in field["dims"] if d != ":")
                if ":" in field["dims"]:
                    if value.ndim != 1 or value.size < _surface_extent(name, values):
                        raise ValueError(f"{name} is shorter than its native varying-surface read extent")
                elif is_array and value.shape != expected:
                    raise ValueError(f"{name} shape {value.shape} != {expected}")
                elif is_output_scalar and value.size != 1:
                    raise ValueError(f"{name} must contain one output scalar")
                packed.append(value)
            else:
                packed.append(dtype.type(value))
        dimensions = tuple(int(values[n]) for n in ("nvertlevels", "maxedges", "num_scalars"))
        device = cp.cuda.runtime.getDevice()
        stream = cp.cuda.get_current_stream().ptr
        workitems = max(int(values.get(name, 0)) for name in ("ncells", "nedges", "nvertices")) + 1
        for name, dims in spec["scratch"].items():
            shape = tuple(_extent(d, values) for d in dims) + (workitems,)
            key = (device, stream, routine, dimensions, workitems, name)
            if key not in self._scratch:
                self._scratch[key] = cp.empty(shape, dtype=cp.float32, order="F")
            packed.append(self._scratch[key])
        def launch(plan, scalar=0):
            for phase in plan:
                start, end = (_extent(bound, values) for bound in phase["bounds"])
                if "repeat" in phase:
                    for index in range(start, end + 1):
                        launch(phase["body"], index)
                elif end >= start:
                    blocks = (end - start + 128) // 128
                    _module(device, self.prognostic_viscosity_floor).get_function(phase["kernel"])(
                        (blocks,), (128,), tuple(packed) + (np.int32(scalar),))
        launch(PHASES[routine])
        return arguments

    def les_models(self, **arguments):
        """mpas_atm_dissipation_models.F:208-460."""
        return self.call("les_models", **arguments)

    def calculate_n2(self, **arguments):
        """mpas_atm_dissipation_models.F:464-575."""
        return self.call("calculate_n2", **arguments)

    def u_dissipation_3d(self, **arguments):
        """mpas_atm_dissipation_models.F:579-947."""
        return self.call("u_dissipation_3d", **arguments)

    def w_dissipation_3d(self, **arguments):
        """mpas_atm_dissipation_models.F:951-1153."""
        return self.call("w_dissipation_3d", **arguments)

    def scalar_dissipation_3d_les(self, **arguments):
        """mpas_atm_dissipation_models.F:1157-1606."""
        return self.call("scalar_dissipation_3d_les", **arguments)


    def apply_rk_mixing(self, rk_step: int, /, **fields):
        """LES driver slice: mpas_atm_time_integration.F:6331-6390,6581-6606,6724-6741,6881-6905.

        All mixing tendencies are computed at RK stage 1 and retained at later
        stages. The enclosing driver must add the retained tendencies in native
        order, and own any halo exchanges before/after this slice. This launcher
        expects the union of native routine argument names; bn2 supplies the
        closure's bv_freq2. The five routines execute on the caller's stream.
        """
        if rk_step != 1:
            return fields
        values = {name.lower(): value for name, value in fields.items()}
        if int(values["les_model_opt"]) not in (1, 2):
            raise ValueError("the LES driver slice requires an active native LES model")
        start, end = int(values["edgestart"]), int(values["edgeend"])
        values["tend_u_euler"][:, start-1:end].fill(0)
        values["bv_freq2"] = values["bn2"]
        for routine in ("calculate_n2", "les_models", "u_dissipation_3d",
                        "w_dissipation_3d", "scalar_dissipation_3d_les"):
            arguments = {name: values[name] for name in ROUTINES[routine]["args"]}
            self.call(routine, **arguments)
        return fields


def validate_les_configuration(config, *, les_dissipation=None) -> tuple[int, int]:
    """Validate Registry LES options and require an installed execution route.

    Selectors: mpas_atm_dissipation_models.F:57-116. The route guard prevents
    cuda_driver.py's existing 2-D mixing path from silently ignoring LES,
    TKE production and surface fluxes. Pass the resident launcher only from
    a driver that actually calls these routines at the native RK boundary.
    """
    from .errors import ConfigurationRefusal
    def refuse(knob, reason, admitted):
        raise ConfigurationRefusal(knob, getattr(config, knob), reason, admitted)
    model = (les_model_from_string(config.config_les_model)
             if isinstance(config.config_les_model, str) else -1)
    surface = (les_surface_from_string(config.config_les_surface)
               if isinstance(config.config_les_surface, str) else -1)
    if model < 0:
        refuse("config_les_model", "unknown native LES model selector",
               "none, 3d_smagorinsky, or prognostic_1.5_order")
    if surface < 0:
        refuse("config_les_surface", "unknown native LES surface selector",
               "none, specified, or varying")
    if not isinstance(config.config_mix_scalars, (bool, np.bool_)):
        refuse("config_mix_scalars", "the Registry option is logical", "True or False")
    knobs = ("config_surface_heat_flux", "config_surface_moisture_flux",
             "config_surface_drag_coefficient")
    for knob in knobs:
        if not np.isfinite(float(getattr(config, knob))):
            refuse(knob, "the LES flux or drag value must be finite", "a finite real value")
    # Native surface/flux branches are nested under an active LES model
    # (mpas_atm_dissipation_models.F:888-946,1472-1606). With model=none,
    # these options have no effect and need no execution route. Scalar
    # filtering remains active independently (same file:1334-1467).
    active = (["config_les_model"] if model else [])
    active += (["config_mix_scalars"] if config.config_mix_scalars else [])
    if les_dissipation is not None and not isinstance(les_dissipation, LesDissipationV842):
        raise TypeError("les_dissipation must be a resident LesDissipationV842 launcher")
    if active and les_dissipation is None:
        refuse(active[0],
               "the forecast driver has not connected LesDissipationV842 at the RK mixing boundary; "
               "its 2-D path would omit LES/TKE tendencies, surface fluxes or scalar filtering",
               "the forecast front door (WOOF or --physics-backend mpasphys), which connects it; "
               "or config_les_model='none' and config_mix_scalars=false")
    return model, surface


def les_boundary_arguments() -> dict[str, dict[str, Any]]:
    """Native argument union for the RK driver slice (mpas_atm_time_integration.F:6371-6390)."""
    fields = {}
    for routine in ROUTINES.values():
        for name, field in routine["fields"].items():
            # Closure scalar outputs remain device buffers through all applications.
            if name not in fields or field["intent"] != "in":
                fields[name] = field
    return fields
