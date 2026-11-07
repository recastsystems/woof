"""Whole MPAS-A v8.4.2 CAM radiation core/support transcription.

MPAS module_ra_cam.F and module_ra_cam_support.F are authoritative; each
numeric routine and statement cites its source. The host-glibc witness passes
all defined arguments in 30 synthetic calls. Device initialization, driver
conversion, real CLI selection and whole forecasts remain unqualified.
No WRF radiation kernel is reused and WOOF defaults remain unchanged.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target, image_target as _image_target
from woof.hex import numerics as _hex_numerics

OPTIONAL_ARGUMENTS = ('swupt', 'swuptc', 'swdnt', 'swdntc', 'lwupt', 'lwuptc', 'lwdnt', 'lwdntc', 'swupb', 'swupbc', 'swdnb', 'swdnbc', 'lwupb', 'lwupbc', 'lwdnb', 'lwdnbc')

STATE_FIELDS = (('ixcldliq', 'int', 0), ('ixcldice', 'int', 0), ('idxsul', 'int', 0), ('idxsslt', 'int', 0), ('idxdustfirst', 'int', 0), ('idxcarbonfirst', 'int', 0), ('idxocpho', 'int', 0), ('idxbcpho', 'int', 0), ('idxocphi', 'int', 0), ('idxbcphi', 'int', 0), ('idxbg', 'int', 0), ('idxvolc', 'int', 0), ('mxaerl', 'int', 0), ('sulscl_rf', 'double', 0), ('carscl_rf', 'double', 0), ('ssltscl_rf', 'double', 0), ('dustscl_rf', 'double', 0), ('bgscl_rf', 'double', 0), ('volcscl_rf', 'double', 0), ('tauback', 'double', 0), ('sulscl', 'double', 0), ('carscl', 'double', 0), ('ssltscl', 'double', 0), ('dustscl', 'double', 0), ('volcscl', 'double', 0), ('abs_cff_mss_aer', 'double', 1), ('c16', 'double', 0), ('c17', 'double', 0), ('c26', 'double', 0), ('c27', 'double', 0), ('c28', 'double', 0), ('c29', 'double', 0), ('c30', 'double', 0), ('c31', 'double', 0), ('fwcoef', 'double', 0), ('fwc1', 'double', 0), ('fwc2', 'double', 0), ('fc1', 'double', 0), ('amco2', 'double', 0), ('amd', 'double', 0), ('p0', 'double', 0), ('ah2onw', 'double', 5), ('eh2onw', 'double', 5), ('ah2ow', 'double', 5), ('cn_ah2ow', 'double', 5), ('cn_eh2ow', 'double', 5), ('ln_ah2ow', 'double', 5), ('ln_eh2ow', 'double', 5), ('coefh', 'double', 2), ('coefj', 'double', 2), ('coefk', 'double', 2), ('estblh2o', 'double', 1), ('fat', 'double', 2), ('fet', 'double', 2), ('gravit', 'double', 0), ('rga', 'double', 0), ('gravmks', 'double', 0), ('cpair', 'double', 0), ('epsilo', 'double', 0), ('epsqs', 'double', 0), ('sslp', 'double', 0), ('stebol', 'double', 0), ('rgsslp', 'double', 0), ('dpfo3', 'double', 0), ('dpfco2', 'double', 0), ('dayspy', 'double', 0), ('pie', 'double', 0), ('mwdry', 'double', 0), ('scon', 'double', 0), ('co2mmr', 'double', 0), ('mwco2', 'double', 0), ('mwh2o', 'double', 0), ('mwch4', 'double', 0), ('mwn2o', 'double', 0), ('mwf11', 'double', 0), ('mwf12', 'double', 0), ('cappa', 'double', 0), ('rair', 'double', 0), ('tmelt', 'double', 0), ('r_universal', 'double', 0), ('latvap', 'double', 0), ('latice', 'double', 0), ('zvir', 'double', 0), ('estbl', 'double', 1), ('tmin', 'double', 0), ('tmax', 'double', 0), ('pcf', 'double', 1), ('ksul', 'double', 2), ('wsul', 'double', 2), ('gsul', 'double', 2), ('ksslt', 'double', 2), ('wsslt', 'double', 2), ('gsslt', 'double', 2), ('kcphil', 'double', 2), ('wcphil', 'double', 2), ('gcphil', 'double', 2), ('kbg', 'double', 1), ('wbg', 'double', 1), ('gbg', 'double', 1), ('kcphob', 'double', 1), ('wcphob', 'double', 1), ('gcphob', 'double', 1), ('kcb', 'double', 1), ('wcb', 'double', 1), ('gcb', 'double', 1), ('kvolc', 'double', 1), ('wvolc', 'double', 1), ('gvolc', 'double', 1), ('kdst', 'double', 2), ('wdst', 'double', 2), ('gdst', 'double', 2), ('cplos', 'double', 0), ('cplol', 'double', 0), ('co2vmr', 'double', 0), ('n2ovmr', 'double', 0), ('ch4vmr', 'double', 0), ('f11vmr', 'double', 0), ('f12vmr', 'double', 0), ('yrdata', 'int', 1), ('co2', 'double', 1), ('ntoplw', 'int', 0), ('masterproc', 'bool', 0), ('ozncyc', 'bool', 0), ('indirect', 'bool', 0), ('radforce', 'bool', 0), ('trace_gas', 'bool', 0), ('strat_volcanic', 'bool', 0), ('retab', 'double', 1), ('abarl', 'double', 1), ('bbarl', 'double', 1), ('cbarl', 'double', 1), ('dbarl', 'double', 1), ('ebarl', 'double', 1), ('fbarl', 'double', 1), ('abari', 'double', 1), ('bbari', 'double', 1), ('cbari', 'double', 1), ('dbari', 'double', 1), ('ebari', 'double', 1), ('fbari', 'double', 1), ('delta', 'double', 0), ('o2mmr', 'double', 0), ('frcsol', 'double', 1), ('wavmin', 'double', 1), ('wavmax', 'double', 1), ('raytau', 'double', 1), ('abh2o', 'double', 1), ('abo3', 'double', 1), ('abco2', 'double', 1), ('abo2', 'double', 1), ('ph2o', 'double', 1), ('pco2', 'double', 1), ('po2', 'double', 1), ('nirwgt', 'double', 1), ('amo', 'double', 0))
STATE_LAYOUT = {'abs_cff_mss_aer': ((1,), (7,), (1,)), 'ah2onw': ((1, 1, 1, 1, 1), (10, 10, 25, 21, 7), (1, 10, 100, 2500, 52500)), 'eh2onw': ((1, 1, 1, 1, 1), (10, 10, 25, 21, 7), (1, 10, 100, 2500, 52500)), 'ah2ow': ((1, 1, 1, 1, 1), (10, 10, 25, 21, 7), (1, 10, 100, 2500, 52500)), 'cn_ah2ow': ((1, 1, 1, 1, 1), (10, 10, 25, 21, 7), (1, 10, 100, 2500, 52500)), 'cn_eh2ow': ((1, 1, 1, 1, 1), (10, 10, 25, 21, 7), (1, 10, 100, 2500, 52500)), 'ln_ah2ow': ((1, 1, 1, 1, 1), (10, 10, 25, 21, 7), (1, 10, 100, 2500, 52500)), 'ln_eh2ow': ((1, 1, 1, 1, 1), (10, 10, 25, 21, 7), (1, 10, 100, 2500, 52500)), 'coefh': ((1, 1), (2, 4), (1, 2)), 'coefj': ((1, 1), (3, 2), (1, 3)), 'coefk': ((1, 1), (3, 2), (1, 3)), 'estblh2o': ((0,), (193,), (1,)), 'fat': ((1, 1), (6, 2), (1, 6)), 'fet': ((1, 1), (6, 2), (1, 6)), 'estbl': ((1,), (250,), (1,)), 'pcf': ((1,), (6,), (1,)), 'ksul': ((1, 1), (1000, 19), (1, 1000)), 'wsul': ((1, 1), (1000, 19), (1, 1000)), 'gsul': ((1, 1), (1000, 19), (1, 1000)), 'ksslt': ((1, 1), (1000, 19), (1, 1000)), 'wsslt': ((1, 1), (1000, 19), (1, 1000)), 'gsslt': ((1, 1), (1000, 19), (1, 1000)), 'kcphil': ((1, 1), (1000, 19), (1, 1000)), 'wcphil': ((1, 1), (1000, 19), (1, 1000)), 'gcphil': ((1, 1), (1000, 19), (1, 1000)), 'kbg': ((1,), (19,), (1,)), 'wbg': ((1,), (19,), (1,)), 'gbg': ((1,), (19,), (1,)), 'kcphob': ((1,), (19,), (1,)), 'wcphob': ((1,), (19,), (1,)), 'gcphob': ((1,), (19,), (1,)), 'kcb': ((1,), (19,), (1,)), 'wcb': ((1,), (19,), (1,)), 'gcb': ((1,), (19,), (1,)), 'kvolc': ((1,), (19,), (1,)), 'wvolc': ((1,), (19,), (1,)), 'gvolc': ((1,), (19,), (1,)), 'kdst': ((1, 1), (4, 19), (1, 4)), 'wdst': ((1, 1), (4, 19), (1, 4)), 'gdst': ((1, 1), (4, 19), (1, 4)), 'yrdata': ((1,), (233,), (1,)), 'co2': ((1,), (233,), (1,)), 'retab': ((1,), (95,), (1,)), 'abarl': ((1,), (4,), (1,)), 'bbarl': ((1,), (4,), (1,)), 'cbarl': ((1,), (4,), (1,)), 'dbarl': ((1,), (4,), (1,)), 'ebarl': ((1,), (4,), (1,)), 'fbarl': ((1,), (4,), (1,)), 'abari': ((1,), (4,), (1,)), 'bbari': ((1,), (4,), (1,)), 'cbari': ((1,), (4,), (1,)), 'dbari': ((1,), (4,), (1,)), 'ebari': ((1,), (4,), (1,)), 'fbari': ((1,), (4,), (1,)), 'frcsol': ((1,), (19,), (1,)), 'wavmin': ((1,), (19,), (1,)), 'wavmax': ((1,), (19,), (1,)), 'raytau': ((1,), (19,), (1,)), 'abh2o': ((1,), (19,), (1,)), 'abo3': ((1,), (19,), (1,)), 'abco2': ((1,), (19,), (1,)), 'abo2': ((1,), (19,), (1,)), 'ph2o': ((1,), (19,), (1,)), 'pco2': ((1,), (19,), (1,)), 'po2': ((1,), (19,), (1,)), 'nirwgt': ((1,), (19,), (1,))}
STATE_SCALARS = {'ixcldliq': (0, 'int'), 'ixcldice': (1, 'int'), 'idxsul': (2, 'int'), 'idxsslt': (3, 'int'), 'idxdustfirst': (4, 'int'), 'idxcarbonfirst': (5, 'int'), 'idxocpho': (6, 'int'), 'idxbcpho': (7, 'int'), 'idxocphi': (8, 'int'), 'idxbcphi': (9, 'int'), 'idxbg': (10, 'int'), 'idxvolc': (11, 'int'), 'mxaerl': (12, 'int'), 'sulscl_rf': (13, 'double'), 'carscl_rf': (14, 'double'), 'ssltscl_rf': (15, 'double'), 'dustscl_rf': (16, 'double'), 'bgscl_rf': (17, 'double'), 'volcscl_rf': (18, 'double'), 'tauback': (19, 'double'), 'sulscl': (20, 'double'), 'carscl': (21, 'double'), 'ssltscl': (22, 'double'), 'dustscl': (23, 'double'), 'volcscl': (24, 'double'), 'c16': (43, 'double'), 'c17': (44, 'double'), 'c26': (45, 'double'), 'c27': (46, 'double'), 'c28': (47, 'double'), 'c29': (48, 'double'), 'c30': (49, 'double'), 'c31': (50, 'double'), 'fwcoef': (51, 'double'), 'fwc1': (52, 'double'), 'fwc2': (53, 'double'), 'fc1': (54, 'double'), 'amco2': (55, 'double'), 'amd': (56, 'double'), 'p0': (57, 'double'), 'gravit': (292, 'double'), 'rga': (293, 'double'), 'gravmks': (294, 'double'), 'cpair': (295, 'double'), 'epsilo': (296, 'double'), 'epsqs': (297, 'double'), 'sslp': (298, 'double'), 'stebol': (299, 'double'), 'rgsslp': (300, 'double'), 'dpfo3': (301, 'double'), 'dpfco2': (302, 'double'), 'dayspy': (303, 'double'), 'pie': (304, 'double'), 'mwdry': (305, 'double'), 'scon': (306, 'double'), 'co2mmr': (307, 'double'), 'mwco2': (308, 'double'), 'mwh2o': (309, 'double'), 'mwch4': (310, 'double'), 'mwn2o': (311, 'double'), 'mwf11': (312, 'double'), 'mwf12': (313, 'double'), 'cappa': (314, 'double'), 'rair': (315, 'double'), 'tmelt': (316, 'double'), 'r_universal': (317, 'double'), 'latvap': (318, 'double'), 'latice': (319, 'double'), 'zvir': (320, 'double'), 'tmin': (339, 'double'), 'tmax': (340, 'double'), 'cplos': (791, 'double'), 'cplol': (792, 'double'), 'co2vmr': (793, 'double'), 'n2ovmr': (794, 'double'), 'ch4vmr': (795, 'double'), 'f11vmr': (796, 'double'), 'f12vmr': (797, 'double'), 'ntoplw': (834, 'int'), 'masterproc': (835, 'bool'), 'ozncyc': (836, 'bool'), 'indirect': (837, 'bool'), 'radforce': (838, 'bool'), 'trace_gas': (839, 'bool'), 'strat_volcanic': (840, 'bool'), 'delta': (1075, 'double'), 'o2mmr': (1076, 'double'), 'amo': (1293, 'double')}
ARGUMENT_FIELDS = (('rthratenlw', 'float', 3), ('rthratensw', 'float', 3), ('dolw', 'bool', 0), ('dosw', 'bool', 0), ('swupt', 'float', 2), ('swuptc', 'float', 2), ('swdnt', 'float', 2), ('swdntc', 'float', 2), ('lwupt', 'float', 2), ('lwuptc', 'float', 2), ('lwdnt', 'float', 2), ('lwdntc', 'float', 2), ('swupb', 'float', 2), ('swupbc', 'float', 2), ('swdnb', 'float', 2), ('swdnbc', 'float', 2), ('lwupb', 'float', 2), ('lwupbc', 'float', 2), ('lwdnb', 'float', 2), ('lwdnbc', 'float', 2), ('swcf', 'float', 2), ('lwcf', 'float', 2), ('olr', 'float', 2), ('cemiss', 'float', 3), ('taucldc', 'float', 3), ('taucldi', 'float', 3), ('coszr', 'float', 2), ('gsw', 'float', 2), ('glw', 'float', 2), ('xlat', 'float', 2), ('xlong', 'float', 2), ('albedo', 'float', 2), ('t_phy', 'float', 3), ('tsk', 'float', 2), ('emiss', 'float', 2), ('qv3d', 'float', 3), ('qc3d', 'float', 3), ('qr3d', 'float', 3), ('qi3d', 'float', 3), ('qs3d', 'float', 3), ('qg3d', 'float', 3), ('f_qc', 'bool', 0), ('f_qr', 'bool', 0), ('f_qi', 'bool', 0), ('f_qs', 'bool', 0), ('f_ice_phy', 'float', 3), ('f_rain_phy', 'float', 3), ('p_phy', 'float', 3), ('p8w', 'float', 3), ('z', 'float', 3), ('pi_phy', 'float', 3), ('rho_phy', 'float', 3), ('dz8w', 'float', 3), ('cldfra', 'float', 3), ('xland', 'float', 2), ('xice', 'float', 2), ('snow', 'float', 2), ('ozmixm', 'float', 4), ('pin0', 'float', 1), ('levsiz', 'int', 0), ('num_months', 'int', 0), ('m_psp', 'float', 2), ('m_psn', 'float', 2), ('aerosolcp', 'float', 4), ('aerosolcn', 'float', 4), ('m_hybi0', 'float', 1), ('cam_abs_dim1', 'int', 0), ('cam_abs_dim2', 'int', 0), ('paerlev', 'int', 0), ('naer_c', 'int', 0), ('gmt', 'float', 0), ('julday', 'int', 0), ('julian', 'float', 0), ('yr', 'int', 0), ('dt', 'float', 0), ('xtime', 'float', 0), ('declin', 'float', 0), ('solcon', 'float', 0), ('radt', 'float', 0), ('degrad', 'float', 0), ('n_cldadv', 'int', 0), ('abstot_3d', 'float', 4), ('absnxt_3d', 'float', 4), ('emstot_3d', 'float', 3), ('doabsems', 'bool', 0), ('ids', 'int', 0), ('ide', 'int', 0), ('jds', 'int', 0), ('jde', 'int', 0), ('kds', 'int', 0), ('kde', 'int', 0), ('ims', 'int', 0), ('ime', 'int', 0), ('jms', 'int', 0), ('jme', 'int', 0), ('kms', 'int', 0), ('kme', 'int', 0), ('its', 'int', 0), ('ite', 'int', 0), ('jts', 'int', 0), ('jte', 'int', 0), ('kts', 'int', 0), ('kte', 'int', 0))
STAGE_FIELDS = {'camradinit': (('cp', 'float', 0), ('r_d', 'float', 0), ('r_v', 'float', 0), ('gravity', 'float', 0), ('ep_2', 'float', 0), ('stbolt', 'float', 0)), 'radini': (('gravx', 'float', 0), ('cpairx', 'float', 0), ('epsilox', 'float', 0), ('stebolx', 'float', 0), ('pstdx', 'double', 0)), 'radaeini': (('pstdx', 'double', 0), ('mwdryx', 'double', 0), ('mwco2x', 'double', 0)), 'aerosol_init': (('naerlevels', 'int', 0), ('ncells', 'int', 0), ('ncellssolve', 'int', 0), ('nvertlevels', 'int', 0), ('m_hybi', 'float', 2), ('m_psp', 'float', 1), ('pressure_b', 'float', 2), ('aerosolcn', 'float', 3), ('aerosolcp', 'float', 3), ('idxsul_l', 'int', 0), ('idxsslt_l', 'int', 0), ('idxdustfirst_l', 'int', 0), ('idxocpho_l', 'int', 0), ('idxcarbonfirst_l', 'int', 0), ('idxbcpho_l', 'int', 0), ('idxocphi_l', 'int', 0), ('idxbcphi_l', 'int', 0), ('idxbg_l', 'int', 0), ('idxvolc_l', 'int', 0), ('max_mxaerl', 'int', 0)), 'aer_optics_initialize': (('rh_opac', 'double', 1), ('ksul_opac', 'double', 2), ('wsul_opac', 'double', 2), ('gsul_opac', 'double', 2), ('kssam_opac', 'double', 2), ('wssam_opac', 'double', 2), ('gssam_opac', 'double', 2), ('ksscm_opac', 'double', 2), ('wsscm_opac', 'double', 2), ('gsscm_opac', 'double', 2), ('kcphil_opac', 'double', 2), ('wcphil_opac', 'double', 2), ('gcphil_opac', 'double', 2)), 'oznini': (('ncells', 'int', 0), ('num_months', 'int', 0), ('levsiz', 'int', 0), ('latcell', 'float', 1), ('loncell', 'float', 1), ('pin', 'float', 1), ('ozmixm', 'float', 3), ('lat_ozone', 'float', 1), ('ozmixin', 'float', 4), ('degrad', 'float', 0)), 'radiation_lw_from_mpas': (('absnxt_p', 'float', 4), ('abstot_p', 'float', 4), ('aerosolcn_p', 'float', 4), ('aerosolcp_p', 'float', 4), ('aerosols', 'float', 3), ('cam_abs_dim1', 'int', 0), ('cam_abs_dim2', 'int', 0), ('cemiss_p', 'float', 3), ('cldfrac', 'float', 2), ('cldfrac_p', 'float', 3), ('coszr_p', 'float', 2), ('degrad', 'float', 0), ('emstot_p', 'float', 3), ('f_ice', 'float', 3), ('f_rain', 'float', 3), ('glw_p', 'float', 2), ('gsw_p', 'float', 2), ('ite', 'int', 0), ('its', 'int', 0), ('jte', 'int', 0), ('jts', 'int', 0), ('kte', 'int', 0), ('kts', 'int', 0), ('latcell', 'float', 1), ('loncell', 'float', 1), ('lwcf_p', 'float', 2), ('lwdnb_p', 'float', 2), ('lwdnbc_p', 'float', 2), ('lwdnt_p', 'float', 2), ('lwdntc_p', 'float', 2), ('lwupb_p', 'float', 2), ('lwupbc_p', 'float', 2), ('lwupt_p', 'float', 2), ('lwuptc_p', 'float', 2), ('m_hybi', 'float', 2), ('m_hybi_p', 'float', 1), ('m_ps', 'float', 1), ('m_psn_p', 'float', 2), ('m_psp_p', 'float', 2), ('num_aerlevels', 'int', 0), ('num_aerosols', 'int', 0), ('num_months', 'int', 0), ('num_oznlevels', 'int', 0), ('olrtoa_p', 'float', 2), ('ozmixm', 'float', 3), ('ozmixm_p', 'float', 4), ('pin', 'float', 1), ('pin_p', 'float', 1), ('rthratenlw_p', 'float', 3), ('rthratensw_p', 'float', 3), ('sfc_albedo', 'float', 1), ('sfc_albedo_p', 'float', 2), ('sfc_emiss', 'float', 1), ('sfc_emiss_p', 'float', 2), ('skintemp', 'float', 1), ('snow', 'float', 1), ('snow_p', 'float', 2), ('swcf_p', 'float', 2), ('swdnb_p', 'float', 2), ('swdnbc_p', 'float', 2), ('swdnt_p', 'float', 2), ('swdntc_p', 'float', 2), ('swupb_p', 'float', 2), ('swupbc_p', 'float', 2), ('swupt_p', 'float', 2), ('swuptc_p', 'float', 2), ('taucldc_p', 'float', 3), ('taucldi_p', 'float', 3), ('tsk_p', 'float', 2), ('xice', 'float', 1), ('xice_p', 'float', 2), ('xland', 'float', 1), ('xland_p', 'float', 2), ('xlat_p', 'float', 2), ('xlon_p', 'float', 2), ('xtime_s', 'float', 0)), 'radiation_lw_to_mpas': (('glw', 'float', 1), ('glw_p', 'float', 2), ('ite', 'int', 0), ('its', 'int', 0), ('jte', 'int', 0), ('jts', 'int', 0), ('kte', 'int', 0), ('kts', 'int', 0), ('lwcf', 'float', 1), ('lwcf_p', 'float', 2), ('lwdnb', 'float', 1), ('lwdnb_p', 'float', 2), ('lwdnbc', 'float', 1), ('lwdnbc_p', 'float', 2), ('lwdnt', 'float', 1), ('lwdnt_p', 'float', 2), ('lwdntc', 'float', 1), ('lwdntc_p', 'float', 2), ('lwupb', 'float', 1), ('lwupb_p', 'float', 2), ('lwupbc', 'float', 1), ('lwupbc_p', 'float', 2), ('lwupt', 'float', 1), ('lwupt_p', 'float', 2), ('lwuptc', 'float', 1), ('lwuptc_p', 'float', 2), ('olrtoa', 'float', 1), ('olrtoa_p', 'float', 2), ('rthratenlw', 'float', 2), ('rthratenlw_p', 'float', 3)), 'radiation_camlw_to_mpas': (('absnxt', 'float', 3), ('absnxt_p', 'float', 4), ('abstot', 'float', 3), ('abstot_p', 'float', 4), ('cam_abs_dim1', 'int', 0), ('cam_abs_dim2', 'int', 0), ('emstot', 'float', 2), ('emstot_p', 'float', 3), ('ite', 'int', 0), ('its', 'int', 0), ('jte', 'int', 0), ('jts', 'int', 0), ('kte', 'int', 0), ('kts', 'int', 0)), 'radiation_sw_from_mpas': (('absnxt_p', 'float', 4), ('abstot_p', 'float', 4), ('aer_opt', 'int', 0), ('aerosolcn_p', 'float', 4), ('aerosolcp_p', 'float', 4), ('aerosols', 'float', 3), ('cam_abs_dim1', 'int', 0), ('cam_abs_dim2', 'int', 0), ('cemiss_p', 'float', 3), ('cldfrac', 'float', 2), ('cldfrac_p', 'float', 3), ('coszr_p', 'float', 2), ('degrad', 'float', 0), ('emstot_p', 'float', 3), ('f_ice', 'float', 3), ('f_rain', 'float', 3), ('glw_p', 'float', 2), ('gsw_p', 'float', 2), ('ite', 'int', 0), ('its', 'int', 0), ('jte', 'int', 0), ('jts', 'int', 0), ('kte', 'int', 0), ('kts', 'int', 0), ('latcell', 'float', 1), ('loncell', 'float', 1), ('lwcf_p', 'float', 2), ('lwdnb_p', 'float', 2), ('lwdnbc_p', 'float', 2), ('lwdnt_p', 'float', 2), ('lwdntc_p', 'float', 2), ('lwupb_p', 'float', 2), ('lwupbc_p', 'float', 2), ('lwupt_p', 'float', 2), ('lwuptc_p', 'float', 2), ('m_hybi', 'float', 2), ('m_hybi_p', 'float', 1), ('m_ps', 'float', 1), ('m_psn_p', 'float', 2), ('m_psp_p', 'float', 2), ('num_aerlevels', 'int', 0), ('num_aerosols', 'int', 0), ('num_months', 'int', 0), ('num_oznlevels', 'int', 0), ('olrtoa_p', 'float', 2), ('ozmixm', 'float', 3), ('ozmixm_p', 'float', 4), ('pin', 'float', 1), ('pin_p', 'float', 1), ('rthratenlw_p', 'float', 3), ('rthratensw_p', 'float', 3), ('sfc_albedo', 'float', 1), ('sfc_albedo_p', 'float', 2), ('sfc_emiss', 'float', 1), ('sfc_emiss_p', 'float', 2), ('skintemp', 'float', 1), ('snow', 'float', 1), ('snow_p', 'float', 2), ('swcf_p', 'float', 2), ('swdnb_p', 'float', 2), ('swdnbc_p', 'float', 2), ('swdnt_p', 'float', 2), ('swdntc_p', 'float', 2), ('swupb_p', 'float', 2), ('swupbc_p', 'float', 2), ('swupt_p', 'float', 2), ('swuptc_p', 'float', 2), ('taucldc_p', 'float', 3), ('taucldi_p', 'float', 3), ('tsk_p', 'float', 2), ('xice', 'float', 1), ('xice_p', 'float', 2), ('xland', 'float', 1), ('xland_p', 'float', 2), ('xlat_p', 'float', 2), ('xlon_p', 'float', 2), ('xtime_s', 'float', 0)), 'radiation_sw_to_mpas': (('coszr', 'float', 1), ('coszr_p', 'float', 2), ('gsw', 'float', 1), ('gsw_p', 'float', 2), ('ite', 'int', 0), ('its', 'int', 0), ('jte', 'int', 0), ('jts', 'int', 0), ('kte', 'int', 0), ('kts', 'int', 0), ('rthratensw', 'float', 2), ('rthratensw_p', 'float', 3), ('swcf', 'float', 1), ('swcf_p', 'float', 2), ('swdnb', 'float', 1), ('swdnb_p', 'float', 2), ('swdnbc', 'float', 1), ('swdnbc_p', 'float', 2), ('swdnt', 'float', 1), ('swdnt_p', 'float', 2), ('swdntc', 'float', 1), ('swdntc_p', 'float', 2), ('swupb', 'float', 1), ('swupb_p', 'float', 2), ('swupbc', 'float', 1), ('swupbc_p', 'float', 2), ('swupt', 'float', 1), ('swupt_p', 'float', 2), ('swuptc', 'float', 1), ('swuptc_p', 'float', 2)), 'radconst': (('declin', 'float', 0), ('degrad', 'float', 0), ('dpd', 'float', 0), ('julian', 'float', 0), ('solcon', 'float', 0), ('solcon_0', 'float', 0)), 'driver_clock': (('xtime_s', 'float', 0), ('dt_radt', 'float', 0), ('xtime_m', 'float', 0), ('radt', 'float', 0)), 'initialize_defaults': ()}
STAGE_SCALARS = {'camradinit': {'cp': (0, 'float'), 'r_d': (1, 'float'), 'r_v': (2, 'float'), 'gravity': (3, 'float'), 'ep_2': (4, 'float'), 'stbolt': (5, 'float')}, 'radini': {'gravx': (0, 'float'), 'cpairx': (1, 'float'), 'epsilox': (2, 'float'), 'stebolx': (3, 'float'), 'pstdx': (4, 'double')}, 'radaeini': {'pstdx': (0, 'double'), 'mwdryx': (1, 'double'), 'mwco2x': (2, 'double')}, 'aerosol_init': {'naerlevels': (0, 'int'), 'ncells': (1, 'int'), 'ncellssolve': (2, 'int'), 'nvertlevels': (3, 'int'), 'idxsul_l': (94, 'int'), 'idxsslt_l': (95, 'int'), 'idxdustfirst_l': (96, 'int'), 'idxocpho_l': (97, 'int'), 'idxcarbonfirst_l': (98, 'int'), 'idxbcpho_l': (99, 'int'), 'idxocphi_l': (100, 'int'), 'idxbcphi_l': (101, 'int'), 'idxbg_l': (102, 'int'), 'idxvolc_l': (103, 'int'), 'max_mxaerl': (104, 'int')}, 'aer_optics_initialize': {}, 'oznini': {'ncells': (0, 'int'), 'num_months': (1, 'int'), 'levsiz': (2, 'int'), 'degrad': (111, 'float')}, 'radiation_lw_from_mpas': {'cam_abs_dim1': (90, 'int'), 'cam_abs_dim2': (91, 'int'), 'degrad': (164, 'float'), 'ite': (255, 'int'), 'its': (256, 'int'), 'jte': (257, 'int'), 'jts': (258, 'int'), 'kte': (259, 'int'), 'kts': (260, 'int'), 'num_aerlevels': (549, 'int'), 'num_aerosols': (550, 'int'), 'num_months': (551, 'int'), 'num_oznlevels': (552, 'int'), 'xtime_s': (1129, 'float')}, 'radiation_lw_to_mpas': {'ite': (36, 'int'), 'its': (37, 'int'), 'jte': (38, 'int'), 'jts': (39, 'int'), 'kte': (40, 'int'), 'kts': (41, 'int')}, 'radiation_camlw_to_mpas': {'cam_abs_dim1': (72, 'int'), 'cam_abs_dim2': (73, 'int'), 'ite': (110, 'int'), 'its': (111, 'int'), 'jte': (112, 'int'), 'jts': (113, 'int'), 'kte': (114, 'int'), 'kts': (115, 'int')}, 'radiation_sw_from_mpas': {'aer_opt': (36, 'int'), 'cam_abs_dim1': (91, 'int'), 'cam_abs_dim2': (92, 'int'), 'degrad': (165, 'float'), 'ite': (256, 'int'), 'its': (257, 'int'), 'jte': (258, 'int'), 'jts': (259, 'int'), 'kte': (260, 'int'), 'kts': (261, 'int'), 'num_aerlevels': (550, 'int'), 'num_aerosols': (551, 'int'), 'num_months': (552, 'int'), 'num_oznlevels': (553, 'int'), 'xtime_s': (1130, 'float')}, 'radiation_sw_to_mpas': {'ite': (72, 'int'), 'its': (73, 'int'), 'jte': (74, 'int'), 'jts': (75, 'int'), 'kte': (76, 'int'), 'kts': (77, 'int')}, 'radconst': {'declin': (0, 'float'), 'degrad': (1, 'float'), 'dpd': (2, 'float'), 'julian': (3, 'float'), 'solcon': (4, 'float'), 'solcon_0': (5, 'float')}, 'driver_clock': {'xtime_s': (0, 'float'), 'dt_radt': (1, 'float'), 'xtime_m': (2, 'float'), 'radt': (3, 'float')}, 'initialize_defaults': {}}
INIT_STAGES = ('camradinit', 'radini', 'radaeini', 'aerosol_init', 'aer_optics_initialize', 'oznini', 'initialize_defaults')
KERNEL_NAMES = ('cam_restart_message', 'cam_boundary', 'cam_prepare_cache', 'cam_camradinit', 'cam_radini', 'cam_radaeini', 'cam_aerosol_init', 'cam_aer_optics_initialize', 'cam_oznini', 'cam_radiation_lw_from_mpas', 'cam_radiation_lw_to_mpas', 'cam_radiation_camlw_to_mpas', 'cam_radiation_sw_from_mpas', 'cam_radiation_sw_to_mpas', 'cam_radconst', 'cam_driver_clock', 'cam_initialize_defaults')
DOABSEMS_OFFSET = 1121

CUDA_SOURCE = r"""

#include "glibc_flt32.cuh"
#include "glibc_trig_flt32.cuh"
#include "tempo/tempo_common.cuh"
__device__ __noinline__ float cam_exp(float a) {return gfk_exp(a);}
__device__ __noinline__ double cam_exp(double a) {return glibc_exp(a);}
__device__ __noinline__ float cam_log(float a) {return gfk_log(a);}
__device__ __noinline__ double cam_log(double a) {return glibc_log(a);}
__device__ __noinline__ double cam_log10(double a) {return tempo_device_log10(a);}
__device__ __noinline__ float cam_log10(float a) {return __double2float_rn(tempo_device_log10(double(a)));}
__device__ __noinline__ float cam_sqrt(float a) {return __fsqrt_rn(a);}
__device__ __noinline__ double cam_sqrt(double a) {return __dsqrt_rn(a);}
__device__ __noinline__ float cam_sin(float a) {return glibc_sinf(a);}
__device__ __noinline__ float cam_asin(float a) {return glibc_asinf(a);}
__device__ __noinline__ float cam_atan(float a) {return glibc_atanf(a);}
__device__ __noinline__ float cam_cos(float a) {return glibc_cosf(a);}
// Shared CORE-MATH cosine is explicitly distinguished from host glibc.
__device__ __noinline__ double cam_cos(double a) {return uw_cos(a);}
__device__ __noinline__ float cam_pow(float a,float b) {return gfk_pow(a,b);}
__device__ __noinline__ double cam_pow(double a,double b) {return glibc_pow(a,b);}
__device__ __noinline__ double cam_pow(double a,float b) {return glibc_pow(a,double(b));}
__device__ __noinline__ double cam_pow(float a,double b) {return glibc_pow(double(a),b);}
__device__ __noinline__ float cam_mod(float a,float b) {return fmodf(a,b);}
__device__ __noinline__ double cam_mod(double a,double b) {return fmod(a,b);}
// Fortran descriptor and arithmetic ABI for the whole CAM transcription.
// This host witness uses host glibc. CUDA transcendental bindings must come
// from the shared glibc 2.43 device header; they are deliberately separate.
#pragma once
#ifndef __CUDACC__
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#define CAM_HD
#define CAM_ROUTINE
inline const char* cam_context="entry";
#define CAM_TRACE(x) (cam_context=(x))
#define CAM_ASSERT(x) do {if(!(x)) {fprintf(stderr,"CAM_ABI_ASSERT %s: %s\n",cam_context,#x);abort();}} while(0)
#else
#define CAM_HD __host__ __device__
#define CAM_ROUTINE __device__ __noinline__
#define CAM_ASSERT(x) ((void)0)
#define CAM_TRACE(x) ((void)0)
#endif

template<class A,class B> struct CamSame {static constexpr bool value=false;};
template<class A> struct CamSame<A,A> {static constexpr bool value=true;};


// MPAS module_ra_cam.F:7308: GNU unary sign preserves NaN sign/payload.
template<class T> CAM_HD T cam_neg(T a) {
#if defined(__CUDA_ARCH__)
  if constexpr (CamSame<T,double>::value) {
    unsigned long long bits=__double_as_longlong(a);
    asm("xor.b64 %0,%1,0x8000000000000000;" : "=l"(bits) : "l"(bits));
    return __longlong_as_double(bits);
  }
  if constexpr (CamSame<T,float>::value) {
    unsigned int bits=__float_as_uint(a);
    asm("xor.b32 %0,%1,0x80000000;" : "=r"(bits) : "r"(bits));
    return __uint_as_float(bits);
  }
#endif
  return -a;
}

template<class A,class B> CAM_HD constexpr auto cam_add(A a,B b)->decltype(a+b) {
#if defined(__CUDA_ARCH__)
  using R=decltype(a+b);
  if constexpr (CamSame<R,double>::value) return __dadd_rn(a,b);
  if constexpr (CamSame<R,float>::value) return __fadd_rn(a,b);
#endif
  return a+b;
}
template<class A,class B> CAM_HD constexpr auto cam_sub(A a,B b)->decltype(a-b) {
#if defined(__CUDA_ARCH__)
  using R=decltype(a-b);
  if constexpr (CamSame<R,double>::value) return __dsub_rn(a,b);
  if constexpr (CamSame<R,float>::value) return __fsub_rn(a,b);
#endif
  return a-b;
}
template<class A,class B> CAM_HD constexpr auto cam_mul(A a,B b)->decltype(a*b) {
#if defined(__CUDA_ARCH__)
  using R=decltype(a*b);
  if constexpr (CamSame<R,double>::value) return __dmul_rn(a,b);
  if constexpr (CamSame<R,float>::value) return __fmul_rn(a,b);
#endif
  return a*b;
}
template<class A,class B> CAM_HD constexpr auto cam_div(A a,B b)->decltype(a/b) {
#if defined(__CUDA_ARCH__)
  using R=decltype(a/b);
  if constexpr (CamSame<R,double>::value) return __ddiv_rn(a,b);
  if constexpr (CamSame<R,float>::value) return __fdiv_rn(a,b);
#endif
  return a/b;
}
template<class T> CAM_HD constexpr T cam_abs(T a) {return a<0?-a:a;}
template<class A,class B> CAM_HD constexpr auto cam_min(A a,B b)->decltype(a+b) {return a<b?a:b;}
template<class A,class B,class... X> CAM_HD constexpr auto cam_min(A a,B b,X... x) {return cam_min(cam_min(a,b),x...);}
template<class A,class B> CAM_HD constexpr auto cam_max(A a,B b)->decltype(a+b) {return a>b?a:b;}
template<class A,class B,class... X> CAM_HD constexpr auto cam_max(A a,B b,X... x) {return cam_max(cam_max(a,b),x...);}
template<class A,class B> CAM_HD constexpr auto cam_min0(A a,B b) {return cam_min(a,b);}
template<class T> CAM_HD int cam_nint(T a) {return int(a>=0?cam_add(a,T(.5)):cam_sub(a,T(.5)));}
CAM_HD constexpr float cam_float(int a) {return float(a);}
CAM_HD constexpr int cam_mod(int a,int b) {return a%b;}
template<class T> CAM_HD T cam_aint(T a,int kind=0) {return T((long long)a);}
// GNU integer powers use multiplication trees, independent of libm pow.
template<class T> CAM_HD T cam_pow(T a,int n) {
  const bool reciprocal=n<0;
  unsigned exponent=reciprocal ? 0u-unsigned(n) : unsigned(n);
  T result=T(1);
  bool first=true;
  while(exponent) {
    if(exponent&1) {result=first?a:cam_mul(result,a);first=false;}
    exponent>>=1;
    if(exponent) a=cam_mul(a,a);
  }
  return reciprocal ? cam_div(T(1),result) : result;
}
#ifndef __CUDACC__
inline float cam_exp(float a) {return ::expf(a);}
inline double cam_exp(double a) {return ::exp(a);}
inline float cam_log(float a) {return ::logf(a);}
inline double cam_log(double a) {return ::log(a);}
inline float cam_log10(float a) {return ::log10f(a);}
inline double cam_log10(double a) {return ::log10(a);}
inline float cam_sqrt(float a) {return ::sqrtf(a);}
inline double cam_sqrt(double a) {return ::sqrt(a);}
inline float cam_cos(float a) {return ::cosf(a);}
inline double cam_cos(double a) {return ::cos(a);}
inline float cam_sin(float a) {return ::sinf(a);}
inline double cam_sin(double a) {return ::sin(a);}
inline float cam_asin(float a) {return ::asinf(a);}
inline double cam_asin(double a) {return ::asin(a);}
inline float cam_atan(float a) {return ::atanf(a);}
inline double cam_atan(double a) {return ::atan(a);}
inline float cam_mod(float a,float b) {return ::fmodf(a,b);}
inline double cam_mod(double a,double b) {return ::fmod(a,b);}
inline float cam_pow(float a,float b) {return ::powf(a,b);}
inline double cam_pow(double a,double b) {return ::pow(a,b);}
inline double cam_pow(double a,float b) {return ::pow(a,double(b));}
inline double cam_pow(float a,double b) {return ::pow(double(a),b);}
#endif

struct CamSection {int low,high;};
CAM_HD inline CamSection cam_section(int a,int b) {return {a,b};}
template<class T> struct CamArray {
  T* data=nullptr;
  int rank=0,count=0,low[5]{},extent[5]{},stride[5]{};
  CAM_HD CamArray()=default;
  CAM_HD static CamArray from(T* p) {CamArray r;r.data=p;return r;}
  template<class... B> CAM_HD static CamArray contiguous(T* p,B... b) {
    int bounds[]={int(b)...};
    CamArray r;r.data=p;r.rank=sizeof...(b)/2;r.count=1;
    CAM_ASSERT(r.rank<=5);
    for(int d=0;d<r.rank;++d) {
      r.low[d]=bounds[2*d];r.extent[d]=bounds[2*d+1]-bounds[2*d]+1;
      r.stride[d]=r.count;r.count*=r.extent[d];
    }
    return r;
  }
  template<class... I> CAM_HD T& operator()(I... index) const {
    int a[]={int(index)...};int offset=0;
    CAM_ASSERT(sizeof...(index)==rank);
    for(int d=0;d<rank;++d) {
      CAM_ASSERT(a[d]>=low[d] && a[d]<low[d]+extent[d]);
      offset+=(a[d]-low[d])*stride[d];
    }
    return data[offset];
  }
  CAM_HD T& linear(int index) const {
    CAM_ASSERT(index>=0 && index<count);
    int offset=0;
    for(int d=0;d<rank;++d) {offset+=(index%extent[d])*stride[d];index/=extent[d];}
    return data[offset];
  }
  template<class... B> CAM_HD CamArray rebound(B... b) const {
    if(rank==0) return contiguous(data,b...);
    int bounds[]={int(b)...};CamArray r=*this;
    int cumulative=1;bool packed=true;
    for(int d=0;d<rank;++d) {packed=packed && stride[d]==cumulative;cumulative*=extent[d];}
    if(packed) {
      auto result=contiguous(data,b...);
      // Explicit-shape dummy extents do not bound sequence association.
      // MPAS radctl declares pdel(pcols,pverp), while camrad supplies pver;
      // the extra last plane is never referenced by the native routine.
      return result;
    }
    CAM_ASSERT(sizeof...(b)/2==rank);
    r.count=1;
    for(int d=0;d<rank;++d) {
      CAM_ASSERT(bounds[2*d+1]-bounds[2*d]+1<=extent[d]);
      r.low[d]=bounds[2*d];r.extent[d]=bounds[2*d+1]-bounds[2*d]+1;
      r.count*=r.extent[d];
    }
    return r;
  }
  CAM_HD void take(CamArray& r,int d,int a) const {
    CAM_ASSERT(a>=low[d] && a<low[d]+extent[d]);
    r.data+=(a-low[d])*stride[d];
  }
  CAM_HD void take(CamArray& r,int d,CamSection a) const {
    CAM_ASSERT(a.low>=low[d] && a.high<low[d]+extent[d]);
    r.data+=(a.low-low[d])*stride[d];
    int j=r.rank++;r.low[j]=1;r.extent[j]=a.high-a.low+1;
    r.stride[j]=stride[d];r.count*=r.extent[j];
  }
  template<class... S> CAM_HD CamArray section(S... s) const {
    CAM_ASSERT(sizeof...(s)==rank);
    CamArray r;r.data=data;r.count=1;int d=0;
    (take(r,d++,s),...);
    return r;
  }
  CAM_HD void init(const T* p,int n) const {
    CAM_ASSERT(n==count);
    for(int i=0;i<n;++i) linear(i)=p[i];
  }
};
struct CamArena {
  char* data;size_t used=0,capacity,peak=0;
  CAM_HD CamArena(char* p,size_t n):data(p),capacity(n) {}
  template<class T,class... B> CAM_HD CamArray<T> array(B... b) {
    used=(used+alignof(T)-1)&~size_t(alignof(T)-1);
    auto r=CamArray<T>::contiguous(reinterpret_cast<T*>(data+used),b...);
    used+=r.count*sizeof(T);peak=used>peak?used:peak;CAM_ASSERT(used<=capacity);return r;
  }
};
struct CamFrame {
  CamArena& arena;size_t mark;
  CAM_HD CamFrame(CamArena& a):arena(a),mark(a.used) {}
  CAM_HD ~CamFrame() {arena.used=mark;}
};
template<class T> CAM_HD int cam_size(CamArray<T> a) {return a.count;}
template<class T> CAM_HD bool cam_present(CamArray<T> a) {return a.data!=nullptr;}
CAM_HD inline int cam_count(CamArray<bool> a) {
  int n=0;for(int i=0;i<a.count;++i) n+=a.linear(i);return n;
}
template<class T> CAM_HD T cam_sum(CamArray<T> a) {
  T n=0;for(int i=0;i<a.count;++i) n=cam_add(n,a.linear(i));return n;
}
template<class T> CAM_HD T cam_product(CamArray<T> a) {
  T n=1;for(int i=0;i<a.count;++i) n=cam_mul(n,a.linear(i));return n;
}
template<class T> CAM_HD T cam_maxval(CamArray<T> a,int dim,int index) {
  CAM_ASSERT(a.rank==2 && dim==2);
  T m=a(index+1,1);
  for(int k=2;k<=a.extent[1];++k) m=cam_max(m,a(index+1,k));
  return m;
}
template<class T> CAM_HD T cam_maxval(CamArray<T> a) {
  CAM_ASSERT(a.count>0);
  T m=a.linear(0);
  for(int i=1;i<a.count;++i) m=cam_max(m,a.linear(i));
  return m;
}
// Whole MPAS CAM module transcription. No acceptance claim.
struct CamColumn {
CamArena &arena; int error=0;
const unsigned long long* oracle_locals=nullptr;
template<class T> __device__ CamArray<T> oracle_array(int offset) {
 const auto* p=oracle_locals+offset;CamArray<T> a;
 a.data=reinterpret_cast<T*>(p[0]);a.rank=int(p[1]);a.count=int(p[2]);
 for(int d=0;d<5;++d) {a.low[d]=int(p[3+d]);a.extent[d]=int(p[8+d]);a.stride[d]=int(p[13+d]);}
 return a;
}

// Partition metadata: preserve the native chunk first-column cache test.
int native_chunk_its=0;
static constexpr int r8kind=8;
static constexpr int r8=r8kind;
static constexpr double inf=1.e20f;
static constexpr int bigint=2147483647;
int ixcldliq;
int ixcldice;
static constexpr int nbands=2;
static constexpr int naer_all=12;
static constexpr int naer=10;
static constexpr int bnd_nbr_lw=7;
static constexpr int ndstsz=4;
int idxsul;
int idxsslt;
int idxdustfirst;
int idxcarbonfirst;
int idxocpho;
int idxbcpho;
int idxocphi;
int idxbcphi;
int idxbg;
int idxvolc;
int mxaerl;
static constexpr int numdust=4;
static constexpr int numcarbon=4;
double sulscl_rf;
double carscl_rf;
double ssltscl_rf;
double dustscl_rf;
double bgscl_rf;
double volcscl_rf;
double tauback;
double sulscl;
double carscl;
double ssltscl;
double dustscl;
double volcscl;
static constexpr int idx_lw_0500_0650=3;
static constexpr int idx_lw_0650_0800=4;
static constexpr int idx_lw_0800_1000=5;
static constexpr int idx_lw_1000_1200=6;
static constexpr int idx_lw_1200_2000=7;
CamArray<double> abs_cff_mss_aer;
static constexpr double min_tp_h2o=160.0f;
static constexpr double max_tp_h2o=349.999999f;
static constexpr double dtp_h2o=21.111111111111f;
static constexpr double min_te_h2o=(-120.0f);
static constexpr double max_te_h2o=79.999999f;
static constexpr double dte_h2o=10.0f;
static constexpr double min_rh_h2o=0.0f;
static constexpr double max_rh_h2o=1.19999999f;
static constexpr double drh_h2o=0.2f;
static constexpr double min_lu_h2o=(-8.0f);
static constexpr double min_u_h2o=1.0e-8f;
static constexpr double max_lu_h2o=3.9999999f;
static constexpr double dlu_h2o=0.5f;
static constexpr double min_lp_h2o=(-3.0f);
static constexpr double min_p_h2o=1.0e-3f;
static constexpr double max_lp_h2o=(-0.0000001f);
static constexpr double dlp_h2o=0.3333333333333f;
static constexpr int n_u=25;
static constexpr int n_p=10;
static constexpr int n_tp=10;
static constexpr int n_te=21;
static constexpr int n_rh=7;
double c16;
double c17;
double c26;
double c27;
double c28;
double c29;
double c30;
double c31;
double fwcoef;
double fwc1;
double fwc2;
double fc1;
double amco2;
double amd;
double p0;
CamArray<double> ah2onw;
CamArray<double> eh2onw;
CamArray<double> ah2ow;
CamArray<double> cn_ah2ow;
CamArray<double> cn_eh2ow;
CamArray<double> ln_ah2ow;
CamArray<double> ln_eh2ow;
CamArray<double> coefh;
CamArray<double> coefj;
CamArray<double> coefk;
static constexpr int ntemp=192;
CamArray<double> estblh2o;
static constexpr int o_fa=6;
static constexpr int o_fe=6;
CamArray<double> fat;
CamArray<double> fet;
double gravit;
double rga;
double gravmks;
double cpair;
double epsilo;
double epsqs;
double sslp;
double stebol;
double rgsslp;
double dpfo3;
double dpfco2;
double dayspy;
double pie;
double mwdry;
double scon;
double co2mmr;
double mwco2;
double mwh2o;
double mwch4;
double mwn2o;
double mwf11;
double mwf12;
double cappa;
double rair;
double tmelt;
double r_universal;
double latvap;
double latice;
double zvir;
static constexpr int plenest=250;
CamArray<double> estbl;
double tmin;
double tmax;
CamArray<double> pcf;
static constexpr int idxvis=8;
static constexpr int nrh=1000;
static constexpr int nspint=19;
CamArray<double> ksul;
CamArray<double> wsul;
CamArray<double> gsul;
CamArray<double> ksslt;
CamArray<double> wsslt;
CamArray<double> gsslt;
CamArray<double> kcphil;
CamArray<double> wcphil;
CamArray<double> gcphil;
CamArray<double> kbg;
CamArray<double> wbg;
CamArray<double> gbg;
CamArray<double> kcphob;
CamArray<double> wcphob;
CamArray<double> gcphob;
CamArray<double> kcb;
CamArray<double> wcb;
CamArray<double> gcb;
CamArray<double> kvolc;
CamArray<double> wvolc;
CamArray<double> gvolc;
CamArray<double> kdst;
CamArray<double> wdst;
CamArray<double> gdst;
double cplos;
double cplol;
double co2vmr;
double n2ovmr;
double ch4vmr;
double f11vmr;
double f12vmr;
static constexpr int cyr=233;
CamArray<int> yrdata;
CamArray<double> co2;
int ntoplw;
bool masterproc;
bool ozncyc;
bool indirect;
bool radforce;
bool trace_gas;
bool strat_volcanic;
CamArray<double> retab;
CamArray<double> abarl;
CamArray<double> bbarl;
CamArray<double> cbarl;
CamArray<double> dbarl;
CamArray<double> ebarl;
CamArray<double> fbarl;
CamArray<double> abari;
CamArray<double> bbari;
CamArray<double> cbari;
CamArray<double> dbari;
CamArray<double> ebari;
CamArray<double> fbari;
double delta;
double o2mmr;
CamArray<double> frcsol;
CamArray<double> wavmin;
CamArray<double> wavmax;
CamArray<double> raytau;
CamArray<double> abh2o;
CamArray<double> abo3;
CamArray<double> abco2;
CamArray<double> abo2;
CamArray<double> ph2o;
CamArray<double> pco2;
CamArray<double> po2;
CamArray<double> nirwgt;
static constexpr double v_raytau_35=0.155208f;
static constexpr double v_raytau_64=0.0392f;
static constexpr double v_abo3_35=2.4058030e+01f;
static constexpr double v_abo3_64=2.210e+01f;
double amo;
CAM_HD CamColumn(CamArena &a):arena(a) {}
// MPAS module_ra_cam_support.F:385-440
CAM_ROUTINE void sortarray(const int& n,const CamArray<double>& _ain,const CamArray<int>& _indxa) {
CamFrame _frame(arena);
CAM_TRACE("sortarray.ain");
auto ain=_ain.rebound(1,n);
CAM_TRACE("sortarray.indxa");
auto indxa=_indxa.rebound(1,n);
int i;
int j;
int ni;
int itmp;
double atmp;
// MPAS module_ra_cam_support.F:411
ni=1;
// MPAS module_ra_cam_support.F:412
while(true) {
// MPAS module_ra_cam_support.F:413
ni=cam_add(cam_mul(3,ni),1);
// MPAS module_ra_cam_support.F:414
if((ni<=n)) {continue;}
// MPAS module_ra_cam_support.F:415
break;
// MPAS module_ra_cam_support.F:416
}
// MPAS module_ra_cam_support.F:418
while(true) {
// MPAS module_ra_cam_support.F:419
ni=cam_div(ni,3);
// MPAS module_ra_cam_support.F:420
for(i=cam_add(ni,1);i<=n;i+=1) {
// MPAS module_ra_cam_support.F:421
atmp=ain(i);
// MPAS module_ra_cam_support.F:422
itmp=indxa(i);
// MPAS module_ra_cam_support.F:423
j=i;
// MPAS module_ra_cam_support.F:424
while(true) {
// MPAS module_ra_cam_support.F:425
if((ain(cam_sub(j,ni))<=atmp)) {break;}
// MPAS module_ra_cam_support.F:426
ain(j)=ain(cam_sub(j,ni));
// MPAS module_ra_cam_support.F:427
indxa(j)=indxa(cam_sub(j,ni));
// MPAS module_ra_cam_support.F:428
j=cam_sub(j,ni);
// MPAS module_ra_cam_support.F:429
if((j>ni)) {continue;}
// MPAS module_ra_cam_support.F:430
break;
// MPAS module_ra_cam_support.F:431
}
// MPAS module_ra_cam_support.F:432
ain(j)=atmp;
// MPAS module_ra_cam_support.F:433
indxa(j)=itmp;
// MPAS module_ra_cam_support.F:434
}
// MPAS module_ra_cam_support.F:435
if((ni>1)) {continue;}
// MPAS module_ra_cam_support.F:436
break;
// MPAS module_ra_cam_support.F:437
}
// MPAS module_ra_cam_support.F:438
return;
}
// MPAS module_ra_cam_support.F:441-725
CAM_ROUTINE void trcab(const int& lchnk,const int& ncol,const int& pcols,const int& pverp,const int& k1,const int& k2,const CamArray<double>& _ucfc11,const CamArray<double>& _ucfc12,const CamArray<double>& _un2o0,const CamArray<double>& _un2o1,const CamArray<double>& _uch4,const CamArray<double>& _uco211,const CamArray<double>& _uco212,const CamArray<double>& _uco213,const CamArray<double>& _uco221,const CamArray<double>& _uco222,const CamArray<double>& _uco223,const CamArray<double>& _bn2o0,const CamArray<double>& _bn2o1,const CamArray<double>& _bch4,const CamArray<double>& _to3co2,const CamArray<double>& _pnm,const CamArray<double>& _dw,const CamArray<double>& _pnew,const CamArray<double>& _s2c,const CamArray<double>& _uptype,const CamArray<double>& _dplh2o,const CamArray<double>& _abplnk1,const CamArray<double>& _tco2,const CamArray<double>& _th2o,const CamArray<double>& _to3,const CamArray<double>& _abstrc,const CamArray<double>& _aer_trn_ttl) {
CamFrame _frame(arena);
CAM_TRACE("trcab.ucfc11");
auto ucfc11=_ucfc11.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.ucfc12");
auto ucfc12=_ucfc12.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.un2o0");
auto un2o0=_un2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.un2o1");
auto un2o1=_un2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.uch4");
auto uch4=_uch4.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.uco211");
auto uco211=_uco211.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.uco212");
auto uco212=_uco212.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.uco213");
auto uco213=_uco213.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.uco221");
auto uco221=_uco221.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.uco222");
auto uco222=_uco222.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.uco223");
auto uco223=_uco223.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.bn2o0");
auto bn2o0=_bn2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.bn2o1");
auto bn2o1=_bn2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.bch4");
auto bch4=_bch4.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.to3co2");
auto to3co2=_to3co2.rebound(1,pcols);
CAM_TRACE("trcab.pnm");
auto pnm=_pnm.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.dw");
auto dw=_dw.rebound(1,pcols);
CAM_TRACE("trcab.pnew");
auto pnew=_pnew.rebound(1,pcols);
CAM_TRACE("trcab.s2c");
auto s2c=_s2c.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.uptype");
auto uptype=_uptype.rebound(1,pcols,1,pverp);
CAM_TRACE("trcab.dplh2o");
auto dplh2o=_dplh2o.rebound(1,pcols);
CAM_TRACE("trcab.abplnk1");
auto abplnk1=_abplnk1.rebound(1,14,1,pcols,1,pverp);
CAM_TRACE("trcab.tco2");
auto tco2=_tco2.rebound(1,pcols);
CAM_TRACE("trcab.th2o");
auto th2o=_th2o.rebound(1,pcols);
CAM_TRACE("trcab.to3");
auto to3=_to3.rebound(1,pcols);
CAM_TRACE("trcab.abstrc");
auto abstrc=_abstrc.rebound(1,pcols);
CAM_TRACE("trcab.aer_trn_ttl");
auto aer_trn_ttl=_aer_trn_ttl.rebound(1,pcols,1,pverp,1,pverp,1,bnd_nbr_lw);
int i;
int l;
auto sqti=arena.array<double>(1,pcols);
double du1;
double du2;
double acfc1;
double acfc2;
double acfc3;
double acfc4;
double acfc5;
double acfc6;
double acfc7;
double acfc8;
double du01;
double dbeta01;
double dbeta11;
double an2o1;
double du02;
double dbeta02;
double an2o2;
double du03;
double dbeta03;
double an2o3;
double duch4;
double dbetac;
double ach4;
double du11;
double du12;
double du13;
double dbetc1;
double dbetc2;
double aco21;
double du21;
double du22;
double du23;
double aco22;
auto tt=arena.array<double>(1,pcols);
double psi1;
double phi1;
double p1;
double w1;
auto ds2c=arena.array<double>(1,pcols);
auto duptyp=arena.array<double>(1,pcols);
auto tw=arena.array<double>(1,pcols,1,6);
auto g1=arena.array<double>(1,6);
{ const double _init[]={0.0468556f,0.0397454f,0.0407664f,0.0304380f,0.0540398f,0.0321962f}; g1.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:441-725: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[72]) g1=oracle_array<double>(72);
auto g2=arena.array<double>(1,6);
{ const double _init[]={14.4832f,4.30242f,5.23523f,3.25342f,0.698935f,16.5599f}; g2.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:441-725: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[90]) g2=oracle_array<double>(90);
auto g3=arena.array<double>(1,6);
{ const double _init[]={26.1898f,18.4476f,15.3633f,12.1927f,9.14992f,8.07092f}; g3.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:441-725: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[108]) g3=oracle_array<double>(108);
auto g4=arena.array<double>(1,6);
{ const double _init[]={0.0261782f,0.0369516f,0.0307266f,0.0243854f,0.0182932f,0.0161418f}; g4.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:441-725: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[126]) g4=oracle_array<double>(126);
auto ab=arena.array<double>(1,6);
{ const double _init[]={3.0857e-2f,2.3524e-2f,1.7310e-2f,2.6661e-2f,2.8074e-2f,2.2915e-2f}; ab.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:441-725: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[0]) ab=oracle_array<double>(0);
auto bb=arena.array<double>(1,6);
{ const double _init[]={cam_neg(1.3512e-4f),cam_neg(6.8320e-5f),cam_neg(3.2609e-5f),cam_neg(1.0228e-5f),cam_neg(9.5743e-5f),cam_neg(1.0304e-4f)}; bb.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:441-725: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[36]) bb=oracle_array<double>(36);
auto abp=arena.array<double>(1,6);
{ const double _init[]={2.9129e-2f,2.4101e-2f,1.9821e-2f,2.6904e-2f,2.9458e-2f,1.9892e-2f}; abp.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:441-725: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[18]) abp=oracle_array<double>(18);
auto bbp=arena.array<double>(1,6);
{ const double _init[]={cam_neg(1.3139e-4f),cam_neg(5.5688e-5f),cam_neg(4.6380e-5f),cam_neg(8.0362e-5f),cam_neg(1.0115e-4f),cam_neg(8.8061e-5f)}; bbp.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:441-725: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[54]) bbp=oracle_array<double>(54);
double tcfc3;
double tcfc4;
double tcfc6;
double tcfc7;
double tcfc8;
double tlw;
double tch4;
double u;
double b;
auto func=[&](double u,double b) { return cam_div(u,cam_sqrt(cam_add(4.0f,cam_mul(u,(cam_add(1.0f,cam_div(1.0f,b))))))); };
// MPAS module_ra_cam_support.F:604
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:605
sqti(i)=cam_sqrt(to3co2(i));
// MPAS module_ra_cam_support.F:609
tt(i)=cam_abs(cam_sub(to3co2(i),250.0f));
// MPAS module_ra_cam_support.F:610
ds2c(i)=cam_abs(cam_sub(s2c(i,k1),s2c(i,k2)));
// MPAS module_ra_cam_support.F:611
duptyp(i)=cam_abs(cam_sub(uptype(i,k1),uptype(i,k2)));
// MPAS module_ra_cam_support.F:612
}
// MPAS module_ra_cam_support.F:614
for(l=1;l<=6;l+=1) {
// MPAS module_ra_cam_support.F:615
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:616
psi1=cam_exp(cam_add(cam_mul(abp(l),tt(i)),cam_mul(cam_mul(bbp(l),tt(i)),tt(i))));
// MPAS module_ra_cam_support.F:617
phi1=cam_exp(cam_add(cam_mul(ab(l),tt(i)),cam_mul(cam_mul(bb(l),tt(i)),tt(i))));
// MPAS module_ra_cam_support.F:618
p1=cam_div(cam_mul(pnew(i),(cam_div(psi1,phi1))),sslp);
// MPAS module_ra_cam_support.F:619
w1=cam_mul(dw(i),phi1);
// MPAS module_ra_cam_support.F:620
tw(i,l)=cam_exp(cam_sub(cam_sub(cam_neg(cam_mul(cam_mul(g1(l),p1),(cam_sub(cam_sqrt(cam_add(1.0f,cam_mul(g2(l),(cam_div(w1,p1))))),1.0f)))),cam_mul(g3(l),ds2c(i))),cam_mul(g4(l),duptyp(i))));
// MPAS module_ra_cam_support.F:622
}
// MPAS module_ra_cam_support.F:623
}
// MPAS module_ra_cam_support.F:625
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:626
tw(i,1)=cam_mul(tw(i,1),(cam_add(cam_mul(0.7f,aer_trn_ttl(i,k1,k2,idx_lw_0650_0800)),cam_mul(0.3f,aer_trn_ttl(i,k1,k2,idx_lw_0800_1000)))));
// MPAS module_ra_cam_support.F:628
tw(i,2)=cam_mul(tw(i,2),aer_trn_ttl(i,k1,k2,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:629
tw(i,3)=cam_mul(tw(i,3),aer_trn_ttl(i,k1,k2,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:630
tw(i,4)=cam_mul(tw(i,4),aer_trn_ttl(i,k1,k2,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:631
tw(i,5)=cam_mul(tw(i,5),aer_trn_ttl(i,k1,k2,idx_lw_1000_1200));
// MPAS module_ra_cam_support.F:632
tw(i,6)=cam_mul(tw(i,6),aer_trn_ttl(i,k1,k2,idx_lw_1000_1200));
// MPAS module_ra_cam_support.F:633
}
// MPAS module_ra_cam_support.F:634
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:635
du1=cam_abs(cam_sub(ucfc11(i,k1),ucfc11(i,k2)));
// MPAS module_ra_cam_support.F:636
du2=cam_abs(cam_sub(ucfc12(i,k1),ucfc12(i,k2)));
// MPAS module_ra_cam_support.F:640
tcfc3=cam_exp(cam_neg(cam_mul(175.005f,du1)));
// MPAS module_ra_cam_support.F:641
tcfc4=cam_exp(cam_neg(cam_mul(1202.18f,du1)));
// MPAS module_ra_cam_support.F:642
tcfc6=cam_exp(cam_neg(cam_mul(5786.73f,du2)));
// MPAS module_ra_cam_support.F:643
tcfc7=cam_exp(cam_neg(cam_mul(2873.51f,du2)));
// MPAS module_ra_cam_support.F:644
tcfc8=cam_exp(cam_neg(cam_mul(2085.59f,du2)));
// MPAS module_ra_cam_support.F:648
acfc1=cam_mul(cam_mul(cam_mul(50.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(54.09f,du1)))))),tw(i,1)),abplnk1(7,i,k2));
// MPAS module_ra_cam_support.F:649
acfc2=cam_mul(cam_mul(cam_mul(60.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(5130.03f,du1)))))),tw(i,2)),abplnk1(8,i,k2));
// MPAS module_ra_cam_support.F:650
acfc3=cam_mul(cam_mul(cam_mul(cam_mul(60.0f,(cam_sub(1.0f,tcfc3))),tw(i,4)),tcfc6),abplnk1(9,i,k2));
// MPAS module_ra_cam_support.F:651
acfc4=cam_mul(cam_mul(cam_mul(100.0f,(cam_sub(1.0f,tcfc4))),tw(i,5)),abplnk1(10,i,k2));
// MPAS module_ra_cam_support.F:655
acfc5=cam_mul(cam_mul(cam_mul(45.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(1272.35f,du2)))))),tw(i,3)),abplnk1(11,i,k2));
// MPAS module_ra_cam_support.F:656
acfc6=cam_mul(cam_mul(cam_mul(50.0f,(cam_sub(1.0f,tcfc6))),tw(i,4)),abplnk1(12,i,k2));
// MPAS module_ra_cam_support.F:657
acfc7=cam_mul(cam_mul(cam_mul(cam_mul(80.0f,(cam_sub(1.0f,tcfc7))),tw(i,5)),tcfc4),abplnk1(13,i,k2));
// MPAS module_ra_cam_support.F:658
acfc8=cam_mul(cam_mul(cam_mul(70.0f,(cam_sub(1.0f,tcfc8))),tw(i,6)),abplnk1(14,i,k2));
// MPAS module_ra_cam_support.F:662
tlw=cam_exp(cam_neg(cam_mul(1.0f,cam_sqrt(dplh2o(i)))));
// MPAS module_ra_cam_support.F:663
tlw=cam_mul(tlw,aer_trn_ttl(i,k1,k2,idx_lw_1200_2000));
// MPAS module_ra_cam_support.F:664
duch4=cam_abs(cam_sub(uch4(i,k1),uch4(i,k2)));
// MPAS module_ra_cam_support.F:665
dbetac=cam_div(cam_abs(cam_sub(bch4(i,k1),bch4(i,k2))),duch4);
// MPAS module_ra_cam_support.F:666
ach4=cam_mul(cam_mul(cam_mul(cam_mul(6.00444f,sqti(i)),cam_log(cam_add(1.0f,func(duch4,dbetac)))),tlw),abplnk1(3,i,k2));
// MPAS module_ra_cam_support.F:667
tch4=cam_div(1.0f,(cam_add(1.0f,cam_mul(0.02f,func(duch4,dbetac)))));
// MPAS module_ra_cam_support.F:671
du01=cam_abs(cam_sub(un2o0(i,k1),un2o0(i,k2)));
// MPAS module_ra_cam_support.F:672
du11=cam_abs(cam_sub(un2o1(i,k1),un2o1(i,k2)));
// MPAS module_ra_cam_support.F:673
dbeta01=cam_div(cam_abs(cam_sub(bn2o0(i,k1),bn2o0(i,k2))),du01);
// MPAS module_ra_cam_support.F:674
dbeta11=cam_div(cam_abs(cam_sub(bn2o1(i,k1),bn2o1(i,k2))),du11);
// MPAS module_ra_cam_support.F:678
an2o1=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.35558f,sqti(i)),cam_log(cam_add(cam_add(1.0f,func(du01,dbeta01)),func(du11,dbeta11)))),tlw),tch4),abplnk1(4,i,k2));
// MPAS module_ra_cam_support.F:680
du02=cam_mul(0.100090f,du01);
// MPAS module_ra_cam_support.F:681
du12=cam_mul(0.0992746f,du11);
// MPAS module_ra_cam_support.F:682
dbeta02=cam_mul(0.964282f,dbeta01);
// MPAS module_ra_cam_support.F:686
an2o2=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.65581f,sqti(i)),cam_log(cam_add(cam_add(1.0f,func(du02,dbeta02)),func(du12,dbeta02)))),th2o(i)),tco2(i)),abplnk1(5,i,k2));
// MPAS module_ra_cam_support.F:688
du03=cam_mul(0.0333767f,du01);
// MPAS module_ra_cam_support.F:689
dbeta03=cam_mul(0.982143f,dbeta01);
// MPAS module_ra_cam_support.F:693
an2o3=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.54034f,sqti(i)),cam_log(cam_add(1.0f,func(du03,dbeta03)))),tw(i,6)),tcfc8),abplnk1(6,i,k2));
// MPAS module_ra_cam_support.F:698
du11=cam_abs(cam_sub(uco211(i,k1),uco211(i,k2)));
// MPAS module_ra_cam_support.F:699
du12=cam_abs(cam_sub(uco212(i,k1),uco212(i,k2)));
// MPAS module_ra_cam_support.F:700
du13=cam_abs(cam_sub(uco213(i,k1),uco213(i,k2)));
// MPAS module_ra_cam_support.F:701
dbetc1=cam_div(cam_mul(2.97558f,cam_abs(cam_add(pnm(i,k1),pnm(i,k2)))),(cam_mul(cam_mul(2.0f,sslp),sqti(i))));
// MPAS module_ra_cam_support.F:702
dbetc2=cam_mul(2.0f,dbetc1);
// MPAS module_ra_cam_support.F:703
aco21=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(3.7571f,sqti(i)),cam_log(cam_add(cam_add(cam_add(1.0f,func(du11,dbetc1)),func(du12,dbetc2)),func(du13,dbetc2)))),to3(i)),tw(i,5)),tcfc4),tcfc7),abplnk1(2,i,k2));
// MPAS module_ra_cam_support.F:709
du21=cam_abs(cam_sub(uco221(i,k1),uco221(i,k2)));
// MPAS module_ra_cam_support.F:710
du22=cam_abs(cam_sub(uco222(i,k1),uco222(i,k2)));
// MPAS module_ra_cam_support.F:711
du23=cam_abs(cam_sub(uco223(i,k1),uco223(i,k2)));
// MPAS module_ra_cam_support.F:712
aco22=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(3.8443f,sqti(i)),cam_log(cam_add(cam_add(cam_add(1.0f,func(du21,dbetc1)),func(du22,dbetc1)),func(du23,dbetc2)))),tw(i,4)),tcfc3),tcfc6),abplnk1(1,i,k2));
// MPAS module_ra_cam_support.F:718
abstrc(i)=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(acfc1,acfc2),acfc3),acfc4),acfc5),acfc6),acfc7),acfc8),an2o1),an2o2),an2o3),ach4),aco21),aco22);
// MPAS module_ra_cam_support.F:721
}
// MPAS module_ra_cam_support.F:723
return;
}
// MPAS module_ra_cam_support.F:729-1018
CAM_ROUTINE void trcabn(const int& lchnk,const int& ncol,const int& pcols,const int& pverp,const int& k2,const int& kn,const CamArray<double>& _ucfc11,const CamArray<double>& _ucfc12,const CamArray<double>& _un2o0,const CamArray<double>& _un2o1,const CamArray<double>& _uch4,const CamArray<double>& _uco211,const CamArray<double>& _uco212,const CamArray<double>& _uco213,const CamArray<double>& _uco221,const CamArray<double>& _uco222,const CamArray<double>& _uco223,const CamArray<double>& _tbar,const CamArray<double>& _bplnk,const CamArray<double>& _winpl,const CamArray<double>& _pinpl,const CamArray<double>& _tco2,const CamArray<double>& _th2o,const CamArray<double>& _to3,const CamArray<double>& _uptype,const CamArray<double>& _dw,const CamArray<double>& _s2c,const CamArray<double>& _up2,const CamArray<double>& _pnew,const CamArray<double>& _abstrc,const CamArray<double>& _uinpl,const CamArray<double>& _aer_trn_ngh) {
CamFrame _frame(arena);
CAM_TRACE("trcabn.ucfc11");
auto ucfc11=_ucfc11.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.ucfc12");
auto ucfc12=_ucfc12.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.un2o0");
auto un2o0=_un2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.un2o1");
auto un2o1=_un2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.uch4");
auto uch4=_uch4.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.uco211");
auto uco211=_uco211.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.uco212");
auto uco212=_uco212.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.uco213");
auto uco213=_uco213.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.uco221");
auto uco221=_uco221.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.uco222");
auto uco222=_uco222.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.uco223");
auto uco223=_uco223.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.tbar");
auto tbar=_tbar.rebound(1,pcols,1,4);
CAM_TRACE("trcabn.bplnk");
auto bplnk=_bplnk.rebound(1,14,1,pcols,1,4);
CAM_TRACE("trcabn.winpl");
auto winpl=_winpl.rebound(1,pcols,1,4);
CAM_TRACE("trcabn.pinpl");
auto pinpl=_pinpl.rebound(1,pcols,1,4);
CAM_TRACE("trcabn.tco2");
auto tco2=_tco2.rebound(1,pcols);
CAM_TRACE("trcabn.th2o");
auto th2o=_th2o.rebound(1,pcols);
CAM_TRACE("trcabn.to3");
auto to3=_to3.rebound(1,pcols);
CAM_TRACE("trcabn.uptype");
auto uptype=_uptype.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.dw");
auto dw=_dw.rebound(1,pcols);
CAM_TRACE("trcabn.s2c");
auto s2c=_s2c.rebound(1,pcols,1,pverp);
CAM_TRACE("trcabn.up2");
auto up2=_up2.rebound(1,pcols);
CAM_TRACE("trcabn.pnew");
auto pnew=_pnew.rebound(1,pcols);
CAM_TRACE("trcabn.abstrc");
auto abstrc=_abstrc.rebound(1,pcols);
CAM_TRACE("trcabn.uinpl");
auto uinpl=_uinpl.rebound(1,pcols,1,4);
CAM_TRACE("trcabn.aer_trn_ngh");
auto aer_trn_ngh=_aer_trn_ngh.rebound(1,pcols,1,bnd_nbr_lw);
int i;
int l;
auto sqti=arena.array<double>(1,pcols);
auto rsqti=arena.array<double>(1,pcols);
double du1;
double du2;
double acfc1;
double acfc2;
double acfc3;
double acfc4;
double acfc5;
double acfc6;
double acfc7;
double acfc8;
double du01;
double dbeta01;
double dbeta11;
double an2o1;
double du02;
double dbeta02;
double an2o2;
double du03;
double dbeta03;
double an2o3;
double duch4;
double dbetac;
double ach4;
double du11;
double du12;
double du13;
double dbetc1;
double dbetc2;
double aco21;
double du21;
double du22;
double du23;
double aco22;
auto tt=arena.array<double>(1,pcols);
double psi1;
double phi1;
double p1;
double w1;
auto ds2c=arena.array<double>(1,pcols);
auto duptyp=arena.array<double>(1,pcols);
auto tw=arena.array<double>(1,pcols,1,6);
auto g1=arena.array<double>(1,6);
{ const double _init[]={0.0468556f,0.0397454f,0.0407664f,0.0304380f,0.0540398f,0.0321962f}; g1.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:729-1018: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[216]) g1=oracle_array<double>(216);
auto g2=arena.array<double>(1,6);
{ const double _init[]={14.4832f,4.30242f,5.23523f,3.25342f,0.698935f,16.5599f}; g2.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:729-1018: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[234]) g2=oracle_array<double>(234);
auto g3=arena.array<double>(1,6);
{ const double _init[]={26.1898f,18.4476f,15.3633f,12.1927f,9.14992f,8.07092f}; g3.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:729-1018: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[252]) g3=oracle_array<double>(252);
auto g4=arena.array<double>(1,6);
{ const double _init[]={0.0261782f,0.0369516f,0.0307266f,0.0243854f,0.0182932f,0.0161418f}; g4.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:729-1018: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[270]) g4=oracle_array<double>(270);
auto ab=arena.array<double>(1,6);
{ const double _init[]={3.0857e-2f,2.3524e-2f,1.7310e-2f,2.6661e-2f,2.8074e-2f,2.2915e-2f}; ab.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:729-1018: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[144]) ab=oracle_array<double>(144);
auto bb=arena.array<double>(1,6);
{ const double _init[]={cam_neg(1.3512e-4f),cam_neg(6.8320e-5f),cam_neg(3.2609e-5f),cam_neg(1.0228e-5f),cam_neg(9.5743e-5f),cam_neg(1.0304e-4f)}; bb.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:729-1018: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[180]) bb=oracle_array<double>(180);
auto abp=arena.array<double>(1,6);
{ const double _init[]={2.9129e-2f,2.4101e-2f,1.9821e-2f,2.6904e-2f,2.9458e-2f,1.9892e-2f}; abp.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:729-1018: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[162]) abp=oracle_array<double>(162);
auto bbp=arena.array<double>(1,6);
{ const double _init[]={cam_neg(1.3139e-4f),cam_neg(5.5688e-5f),cam_neg(4.6380e-5f),cam_neg(8.0362e-5f),cam_neg(1.0115e-4f),cam_neg(8.8061e-5f)}; bbp.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:729-1018: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[198]) bbp=oracle_array<double>(198);
double tcfc3;
double tcfc4;
double tcfc6;
double tcfc7;
double tcfc8;
double tlw;
double tch4;
double u;
double b;
auto func=[&](double u,double b) { return cam_div(u,cam_sqrt(cam_add(4.0f,cam_mul(u,(cam_add(1.0f,cam_div(1.0f,b))))))); };
// MPAS module_ra_cam_support.F:894
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:895
sqti(i)=cam_sqrt(tbar(i,kn));
// MPAS module_ra_cam_support.F:896
rsqti(i)=cam_div(1.f,sqti(i));
// MPAS module_ra_cam_support.F:900
tt(i)=cam_abs(cam_sub(tbar(i,kn),250.0f));
// MPAS module_ra_cam_support.F:901
ds2c(i)=cam_mul(cam_abs(cam_sub(s2c(i,cam_add(k2,1)),s2c(i,k2))),uinpl(i,kn));
// MPAS module_ra_cam_support.F:902
duptyp(i)=cam_mul(cam_abs(cam_sub(uptype(i,cam_add(k2,1)),uptype(i,k2))),uinpl(i,kn));
// MPAS module_ra_cam_support.F:903
}
// MPAS module_ra_cam_support.F:905
for(l=1;l<=6;l+=1) {
// MPAS module_ra_cam_support.F:906
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:907
psi1=cam_exp(cam_add(cam_mul(abp(l),tt(i)),cam_mul(cam_mul(bbp(l),tt(i)),tt(i))));
// MPAS module_ra_cam_support.F:908
phi1=cam_exp(cam_add(cam_mul(ab(l),tt(i)),cam_mul(cam_mul(bb(l),tt(i)),tt(i))));
// MPAS module_ra_cam_support.F:909
p1=cam_div(cam_mul(pnew(i),(cam_div(psi1,phi1))),sslp);
// MPAS module_ra_cam_support.F:910
w1=cam_mul(cam_mul(dw(i),winpl(i,kn)),phi1);
// MPAS module_ra_cam_support.F:911
tw(i,l)=cam_exp(cam_sub(cam_sub(cam_neg(cam_mul(cam_mul(g1(l),p1),(cam_sub(cam_sqrt(cam_add(1.0f,cam_mul(g2(l),(cam_div(w1,p1))))),1.0f)))),cam_mul(g3(l),ds2c(i))),cam_mul(g4(l),duptyp(i))));
// MPAS module_ra_cam_support.F:913
}
// MPAS module_ra_cam_support.F:914
}
// MPAS module_ra_cam_support.F:916
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:917
tw(i,1)=cam_mul(tw(i,1),(cam_add(cam_mul(0.7f,aer_trn_ngh(i,idx_lw_0650_0800)),cam_mul(0.3f,aer_trn_ngh(i,idx_lw_0800_1000)))));
// MPAS module_ra_cam_support.F:919
tw(i,2)=cam_mul(tw(i,2),aer_trn_ngh(i,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:920
tw(i,3)=cam_mul(tw(i,3),aer_trn_ngh(i,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:921
tw(i,4)=cam_mul(tw(i,4),aer_trn_ngh(i,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:922
tw(i,5)=cam_mul(tw(i,5),aer_trn_ngh(i,idx_lw_1000_1200));
// MPAS module_ra_cam_support.F:923
tw(i,6)=cam_mul(tw(i,6),aer_trn_ngh(i,idx_lw_1000_1200));
// MPAS module_ra_cam_support.F:924
}
// MPAS module_ra_cam_support.F:926
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:928
du1=cam_mul(cam_abs(cam_sub(ucfc11(i,cam_add(k2,1)),ucfc11(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:929
du2=cam_mul(cam_abs(cam_sub(ucfc12(i,cam_add(k2,1)),ucfc12(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:933
tcfc3=cam_exp(cam_neg(cam_mul(175.005f,du1)));
// MPAS module_ra_cam_support.F:934
tcfc4=cam_exp(cam_neg(cam_mul(1202.18f,du1)));
// MPAS module_ra_cam_support.F:935
tcfc6=cam_exp(cam_neg(cam_mul(5786.73f,du2)));
// MPAS module_ra_cam_support.F:936
tcfc7=cam_exp(cam_neg(cam_mul(2873.51f,du2)));
// MPAS module_ra_cam_support.F:937
tcfc8=cam_exp(cam_neg(cam_mul(2085.59f,du2)));
// MPAS module_ra_cam_support.F:941
acfc1=cam_mul(cam_mul(cam_mul(50.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(54.09f,du1)))))),tw(i,1)),bplnk(7,i,kn));
// MPAS module_ra_cam_support.F:942
acfc2=cam_mul(cam_mul(cam_mul(60.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(5130.03f,du1)))))),tw(i,2)),bplnk(8,i,kn));
// MPAS module_ra_cam_support.F:943
acfc3=cam_mul(cam_mul(cam_mul(cam_mul(60.0f,(cam_sub(1.0f,tcfc3))),tw(i,4)),tcfc6),bplnk(9,i,kn));
// MPAS module_ra_cam_support.F:944
acfc4=cam_mul(cam_mul(cam_mul(100.0f,(cam_sub(1.0f,tcfc4))),tw(i,5)),bplnk(10,i,kn));
// MPAS module_ra_cam_support.F:948
acfc5=cam_mul(cam_mul(cam_mul(45.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(1272.35f,du2)))))),tw(i,3)),bplnk(11,i,kn));
// MPAS module_ra_cam_support.F:949
acfc6=cam_mul(cam_mul(cam_mul(50.0f,(cam_sub(1.0f,tcfc6))),tw(i,4)),bplnk(12,i,kn));
// MPAS module_ra_cam_support.F:950
acfc7=cam_mul(cam_mul(cam_mul(cam_mul(80.0f,(cam_sub(1.0f,tcfc7))),tw(i,5)),tcfc4),bplnk(13,i,kn));
// MPAS module_ra_cam_support.F:951
acfc8=cam_mul(cam_mul(cam_mul(70.0f,(cam_sub(1.0f,tcfc8))),tw(i,6)),bplnk(14,i,kn));
// MPAS module_ra_cam_support.F:955
tlw=cam_exp(cam_neg(cam_mul(1.0f,cam_sqrt(up2(i)))));
// MPAS module_ra_cam_support.F:956
tlw=cam_mul(tlw,aer_trn_ngh(i,idx_lw_1200_2000));
// MPAS module_ra_cam_support.F:957
duch4=cam_mul(cam_abs(cam_sub(uch4(i,cam_add(k2,1)),uch4(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:958
dbetac=cam_div(cam_mul(cam_mul(2.94449f,pinpl(i,kn)),rsqti(i)),sslp);
// MPAS module_ra_cam_support.F:959
ach4=cam_mul(cam_mul(cam_mul(cam_mul(6.00444f,sqti(i)),cam_log(cam_add(1.0f,func(duch4,dbetac)))),tlw),bplnk(3,i,kn));
// MPAS module_ra_cam_support.F:960
tch4=cam_div(1.0f,(cam_add(1.0f,cam_mul(0.02f,func(duch4,dbetac)))));
// MPAS module_ra_cam_support.F:964
du01=cam_mul(cam_abs(cam_sub(un2o0(i,cam_add(k2,1)),un2o0(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:965
du11=cam_mul(cam_abs(cam_sub(un2o1(i,cam_add(k2,1)),un2o1(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:966
dbeta01=cam_div(cam_mul(cam_mul(19.399f,pinpl(i,kn)),rsqti(i)),sslp);
// MPAS module_ra_cam_support.F:967
dbeta11=dbeta01;
// MPAS module_ra_cam_support.F:971
an2o1=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.35558f,sqti(i)),cam_log(cam_add(cam_add(1.0f,func(du01,dbeta01)),func(du11,dbeta11)))),tlw),tch4),bplnk(4,i,kn));
// MPAS module_ra_cam_support.F:973
du02=cam_mul(0.100090f,du01);
// MPAS module_ra_cam_support.F:974
du12=cam_mul(0.0992746f,du11);
// MPAS module_ra_cam_support.F:975
dbeta02=cam_mul(0.964282f,dbeta01);
// MPAS module_ra_cam_support.F:979
an2o2=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.65581f,sqti(i)),cam_log(cam_add(cam_add(1.0f,func(du02,dbeta02)),func(du12,dbeta02)))),tco2(i)),th2o(i)),bplnk(5,i,kn));
// MPAS module_ra_cam_support.F:981
du03=cam_mul(0.0333767f,du01);
// MPAS module_ra_cam_support.F:982
dbeta03=cam_mul(0.982143f,dbeta01);
// MPAS module_ra_cam_support.F:986
an2o3=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.54034f,sqti(i)),cam_log(cam_add(1.0f,func(du03,dbeta03)))),tw(i,6)),tcfc8),bplnk(6,i,kn));
// MPAS module_ra_cam_support.F:991
du11=cam_mul(cam_abs(cam_sub(uco211(i,cam_add(k2,1)),uco211(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:992
du12=cam_mul(cam_abs(cam_sub(uco212(i,cam_add(k2,1)),uco212(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:993
du13=cam_mul(cam_abs(cam_sub(uco213(i,cam_add(k2,1)),uco213(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:994
dbetc1=cam_div(cam_mul(cam_mul(2.97558f,pinpl(i,kn)),rsqti(i)),sslp);
// MPAS module_ra_cam_support.F:995
dbetc2=cam_mul(2.0f,dbetc1);
// MPAS module_ra_cam_support.F:996
aco21=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(3.7571f,sqti(i)),cam_log(cam_add(cam_add(cam_add(1.0f,func(du11,dbetc1)),func(du12,dbetc2)),func(du13,dbetc2)))),to3(i)),tw(i,5)),tcfc4),tcfc7),bplnk(2,i,kn));
// MPAS module_ra_cam_support.F:1002
du21=cam_mul(cam_abs(cam_sub(uco221(i,cam_add(k2,1)),uco221(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:1003
du22=cam_mul(cam_abs(cam_sub(uco222(i,cam_add(k2,1)),uco222(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:1004
du23=cam_mul(cam_abs(cam_sub(uco223(i,cam_add(k2,1)),uco223(i,k2))),winpl(i,kn));
// MPAS module_ra_cam_support.F:1005
aco22=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(3.8443f,sqti(i)),cam_log(cam_add(cam_add(cam_add(1.0f,func(du21,dbetc1)),func(du22,dbetc1)),func(du23,dbetc2)))),tw(i,4)),tcfc3),tcfc6),bplnk(1,i,kn));
// MPAS module_ra_cam_support.F:1011
abstrc(i)=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(acfc1,acfc2),acfc3),acfc4),acfc5),acfc6),acfc7),acfc8),an2o1),an2o2),an2o3),ach4),aco21),aco22);
// MPAS module_ra_cam_support.F:1014
}
// MPAS module_ra_cam_support.F:1016
return;
}
// MPAS module_ra_cam_support.F:1022-1293
CAM_ROUTINE void trcems(const int& lchnk,const int& ncol,const int& pcols,const int& pverp,const int& k,const CamArray<double>& _co2t,const CamArray<double>& _pnm,const CamArray<double>& _ucfc11,const CamArray<double>& _ucfc12,const CamArray<double>& _un2o0,const CamArray<double>& _un2o1,const CamArray<double>& _bn2o0,const CamArray<double>& _bn2o1,const CamArray<double>& _uch4,const CamArray<double>& _bch4,const CamArray<double>& _uco211,const CamArray<double>& _uco212,const CamArray<double>& _uco213,const CamArray<double>& _uco221,const CamArray<double>& _uco222,const CamArray<double>& _uco223,const CamArray<double>& _uptype,const CamArray<double>& _w,const CamArray<double>& _s2c,const CamArray<double>& _up2,const CamArray<double>& _emplnk,const CamArray<double>& _th2o,const CamArray<double>& _tco2,const CamArray<double>& _to3,const CamArray<double>& _emstrc,const CamArray<double>& _aer_trn_ttl) {
CamFrame _frame(arena);
CAM_TRACE("trcems.co2t");
auto co2t=_co2t.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.pnm");
auto pnm=_pnm.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.ucfc11");
auto ucfc11=_ucfc11.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.ucfc12");
auto ucfc12=_ucfc12.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.un2o0");
auto un2o0=_un2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.un2o1");
auto un2o1=_un2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.bn2o0");
auto bn2o0=_bn2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.bn2o1");
auto bn2o1=_bn2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.uch4");
auto uch4=_uch4.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.bch4");
auto bch4=_bch4.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.uco211");
auto uco211=_uco211.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.uco212");
auto uco212=_uco212.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.uco213");
auto uco213=_uco213.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.uco221");
auto uco221=_uco221.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.uco222");
auto uco222=_uco222.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.uco223");
auto uco223=_uco223.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.uptype");
auto uptype=_uptype.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.w");
auto w=_w.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.s2c");
auto s2c=_s2c.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.up2");
auto up2=_up2.rebound(1,pcols);
CAM_TRACE("trcems.emplnk");
auto emplnk=_emplnk.rebound(1,14,1,pcols);
CAM_TRACE("trcems.th2o");
auto th2o=_th2o.rebound(1,pcols);
CAM_TRACE("trcems.tco2");
auto tco2=_tco2.rebound(1,pcols);
CAM_TRACE("trcems.to3");
auto to3=_to3.rebound(1,pcols);
CAM_TRACE("trcems.emstrc");
auto emstrc=_emstrc.rebound(1,pcols,1,pverp);
CAM_TRACE("trcems.aer_trn_ttl");
auto aer_trn_ttl=_aer_trn_ttl.rebound(1,pcols,1,pverp,1,pverp,1,bnd_nbr_lw);
int i;
int l;
auto sqti=arena.array<double>(1,pcols);
double ecfc1;
double ecfc2;
double ecfc3;
double ecfc4;
double ecfc5;
double ecfc6;
double ecfc7;
double ecfc8;
double u01;
double u11;
double beta01;
double beta11;
double en2o1;
double u02;
double u12;
double beta02;
double en2o2;
double u03;
double beta03;
double en2o3;
double betac;
double ech4;
double betac1;
double betac2;
double eco21;
double eco22;
auto tt=arena.array<double>(1,pcols);
double psi1;
double phi1;
double p1;
double w1;
auto tw=arena.array<double>(1,pcols,1,6);
auto g1=arena.array<double>(1,6);
{ const double _init[]={0.0468556f,0.0397454f,0.0407664f,0.0304380f,0.0540398f,0.0321962f}; g1.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1022-1293: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[360]) g1=oracle_array<double>(360);
auto g2=arena.array<double>(1,6);
{ const double _init[]={14.4832f,4.30242f,5.23523f,3.25342f,0.698935f,16.5599f}; g2.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1022-1293: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[378]) g2=oracle_array<double>(378);
auto g3=arena.array<double>(1,6);
{ const double _init[]={26.1898f,18.4476f,15.3633f,12.1927f,9.14992f,8.07092f}; g3.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1022-1293: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[396]) g3=oracle_array<double>(396);
auto g4=arena.array<double>(1,6);
{ const double _init[]={0.0261782f,0.0369516f,0.0307266f,0.0243854f,0.0182932f,0.0161418f}; g4.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1022-1293: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[414]) g4=oracle_array<double>(414);
auto ab=arena.array<double>(1,6);
{ const double _init[]={3.0857e-2f,2.3524e-2f,1.7310e-2f,2.6661e-2f,2.8074e-2f,2.2915e-2f}; ab.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1022-1293: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[288]) ab=oracle_array<double>(288);
auto bb=arena.array<double>(1,6);
{ const double _init[]={cam_neg(1.3512e-4f),cam_neg(6.8320e-5f),cam_neg(3.2609e-5f),cam_neg(1.0228e-5f),cam_neg(9.5743e-5f),cam_neg(1.0304e-4f)}; bb.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1022-1293: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[324]) bb=oracle_array<double>(324);
auto abp=arena.array<double>(1,6);
{ const double _init[]={2.9129e-2f,2.4101e-2f,1.9821e-2f,2.6904e-2f,2.9458e-2f,1.9892e-2f}; abp.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1022-1293: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[306]) abp=oracle_array<double>(306);
auto bbp=arena.array<double>(1,6);
{ const double _init[]={cam_neg(1.3139e-4f),cam_neg(5.5688e-5f),cam_neg(4.6380e-5f),cam_neg(8.0362e-5f),cam_neg(1.0115e-4f),cam_neg(8.8061e-5f)}; bbp.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1022-1293: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[342]) bbp=oracle_array<double>(342);
double tcfc3;
double tcfc4;
double tcfc6;
double tcfc7;
double tcfc8;
double tlw;
double tch4;
double u;
double b;
auto func=[&](double u,double b) { return cam_div(u,cam_sqrt(cam_add(4.0f,cam_mul(u,(cam_add(1.0f,cam_div(1.0f,b))))))); };
// MPAS module_ra_cam_support.F:1175
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1176
sqti(i)=cam_sqrt(co2t(i,k));
// MPAS module_ra_cam_support.F:1180
tt(i)=cam_abs(cam_sub(co2t(i,k),250.0f));
// MPAS module_ra_cam_support.F:1181
}
// MPAS module_ra_cam_support.F:1183
for(l=1;l<=6;l+=1) {
// MPAS module_ra_cam_support.F:1184
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1185
psi1=cam_exp(cam_add(cam_mul(abp(l),tt(i)),cam_mul(cam_mul(bbp(l),tt(i)),tt(i))));
// MPAS module_ra_cam_support.F:1186
phi1=cam_exp(cam_add(cam_mul(ab(l),tt(i)),cam_mul(cam_mul(bb(l),tt(i)),tt(i))));
// MPAS module_ra_cam_support.F:1187
p1=cam_div(cam_mul(pnm(i,k),(cam_div(psi1,phi1))),sslp);
// MPAS module_ra_cam_support.F:1188
w1=cam_mul(w(i,k),phi1);
// MPAS module_ra_cam_support.F:1189
tw(i,l)=cam_exp(cam_sub(cam_sub(cam_neg(cam_mul(cam_mul(g1(l),p1),(cam_sub(cam_sqrt(cam_add(1.0f,cam_mul(g2(l),(cam_div(w1,p1))))),1.0f)))),cam_mul(g3(l),s2c(i,k))),cam_mul(g4(l),uptype(i,k))));
// MPAS module_ra_cam_support.F:1191
}
// MPAS module_ra_cam_support.F:1192
}
// MPAS module_ra_cam_support.F:1197
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1198
tw(i,1)=cam_mul(tw(i,1),(cam_add(cam_mul(0.7f,aer_trn_ttl(i,k,1,idx_lw_0650_0800)),cam_mul(0.3f,aer_trn_ttl(i,k,1,idx_lw_0800_1000)))));
// MPAS module_ra_cam_support.F:1200
tw(i,2)=cam_mul(tw(i,2),aer_trn_ttl(i,k,1,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:1201
tw(i,3)=cam_mul(tw(i,3),aer_trn_ttl(i,k,1,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:1202
tw(i,4)=cam_mul(tw(i,4),aer_trn_ttl(i,k,1,idx_lw_0800_1000));
// MPAS module_ra_cam_support.F:1203
tw(i,5)=cam_mul(tw(i,5),aer_trn_ttl(i,k,1,idx_lw_1000_1200));
// MPAS module_ra_cam_support.F:1204
tw(i,6)=cam_mul(tw(i,6),aer_trn_ttl(i,k,1,idx_lw_1000_1200));
// MPAS module_ra_cam_support.F:1205
}
// MPAS module_ra_cam_support.F:1207
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1211
tcfc3=cam_exp(cam_neg(cam_mul(175.005f,ucfc11(i,k))));
// MPAS module_ra_cam_support.F:1212
tcfc4=cam_exp(cam_neg(cam_mul(1202.18f,ucfc11(i,k))));
// MPAS module_ra_cam_support.F:1213
tcfc6=cam_exp(cam_neg(cam_mul(5786.73f,ucfc12(i,k))));
// MPAS module_ra_cam_support.F:1214
tcfc7=cam_exp(cam_neg(cam_mul(2873.51f,ucfc12(i,k))));
// MPAS module_ra_cam_support.F:1215
tcfc8=cam_exp(cam_neg(cam_mul(2085.59f,ucfc12(i,k))));
// MPAS module_ra_cam_support.F:1219
ecfc1=cam_mul(cam_mul(cam_mul(50.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(54.09f,ucfc11(i,k))))))),tw(i,1)),emplnk(7,i));
// MPAS module_ra_cam_support.F:1220
ecfc2=cam_mul(cam_mul(cam_mul(60.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(5130.03f,ucfc11(i,k))))))),tw(i,2)),emplnk(8,i));
// MPAS module_ra_cam_support.F:1221
ecfc3=cam_mul(cam_mul(cam_mul(cam_mul(60.0f,(cam_sub(1.0f,tcfc3))),tw(i,4)),tcfc6),emplnk(9,i));
// MPAS module_ra_cam_support.F:1222
ecfc4=cam_mul(cam_mul(cam_mul(100.0f,(cam_sub(1.0f,tcfc4))),tw(i,5)),emplnk(10,i));
// MPAS module_ra_cam_support.F:1226
ecfc5=cam_mul(cam_mul(cam_mul(45.0f,(cam_sub(1.0f,cam_exp(cam_neg(cam_mul(1272.35f,ucfc12(i,k))))))),tw(i,3)),emplnk(11,i));
// MPAS module_ra_cam_support.F:1227
ecfc6=cam_mul(cam_mul(cam_mul(50.0f,(cam_sub(1.0f,tcfc6))),tw(i,4)),emplnk(12,i));
// MPAS module_ra_cam_support.F:1228
ecfc7=cam_mul(cam_mul(cam_mul(cam_mul(80.0f,(cam_sub(1.0f,tcfc7))),tw(i,5)),tcfc4),emplnk(13,i));
// MPAS module_ra_cam_support.F:1229
ecfc8=cam_mul(cam_mul(cam_mul(70.0f,(cam_sub(1.0f,tcfc8))),tw(i,6)),emplnk(14,i));
// MPAS module_ra_cam_support.F:1233
tlw=cam_exp(cam_neg(cam_mul(1.0f,cam_sqrt(up2(i)))));
// MPAS module_ra_cam_support.F:1238
tlw=cam_mul(tlw,aer_trn_ttl(i,k,1,idx_lw_1200_2000));
// MPAS module_ra_cam_support.F:1239
betac=cam_div(bch4(i,k),uch4(i,k));
// MPAS module_ra_cam_support.F:1240
ech4=cam_mul(cam_mul(cam_mul(cam_mul(6.00444f,sqti(i)),cam_log(cam_add(1.0f,func(uch4(i,k),betac)))),tlw),emplnk(3,i));
// MPAS module_ra_cam_support.F:1241
tch4=cam_div(1.0f,(cam_add(1.0f,cam_mul(0.02f,func(uch4(i,k),betac)))));
// MPAS module_ra_cam_support.F:1245
u01=un2o0(i,k);
// MPAS module_ra_cam_support.F:1246
u11=un2o1(i,k);
// MPAS module_ra_cam_support.F:1247
beta01=cam_div(bn2o0(i,k),un2o0(i,k));
// MPAS module_ra_cam_support.F:1248
beta11=cam_div(bn2o1(i,k),un2o1(i,k));
// MPAS module_ra_cam_support.F:1252
en2o1=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.35558f,sqti(i)),cam_log(cam_add(cam_add(1.0f,func(u01,beta01)),func(u11,beta11)))),tlw),tch4),emplnk(4,i));
// MPAS module_ra_cam_support.F:1254
u02=cam_mul(0.100090f,u01);
// MPAS module_ra_cam_support.F:1255
u12=cam_mul(0.0992746f,u11);
// MPAS module_ra_cam_support.F:1256
beta02=cam_mul(0.964282f,beta01);
// MPAS module_ra_cam_support.F:1260
en2o2=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.65581f,sqti(i)),cam_log(cam_add(cam_add(1.0f,func(u02,beta02)),func(u12,beta02)))),tco2(i)),th2o(i)),emplnk(5,i));
// MPAS module_ra_cam_support.F:1262
u03=cam_mul(0.0333767f,u01);
// MPAS module_ra_cam_support.F:1263
beta03=cam_mul(0.982143f,beta01);
// MPAS module_ra_cam_support.F:1267
en2o3=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(2.54034f,sqti(i)),cam_log(cam_add(1.0f,func(u03,beta03)))),tw(i,6)),tcfc8),emplnk(6,i));
// MPAS module_ra_cam_support.F:1272
betac1=cam_div(cam_mul(2.97558f,pnm(i,k)),(cam_mul(sslp,sqti(i))));
// MPAS module_ra_cam_support.F:1273
betac2=cam_mul(2.0f,betac1);
// MPAS module_ra_cam_support.F:1274
eco21=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(3.7571f,sqti(i)),cam_log(cam_add(cam_add(cam_add(1.0f,func(uco211(i,k),betac1)),func(uco212(i,k),betac2)),func(uco213(i,k),betac2)))),to3(i)),tw(i,5)),tcfc4),tcfc7),emplnk(2,i));
// MPAS module_ra_cam_support.F:1280
eco22=cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(3.8443f,sqti(i)),cam_log(cam_add(cam_add(cam_add(1.0f,func(uco221(i,k),betac1)),func(uco222(i,k),betac1)),func(uco223(i,k),betac2)))),tw(i,4)),tcfc3),tcfc6),emplnk(1,i));
// MPAS module_ra_cam_support.F:1286
emstrc(i,k)=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(ecfc1,ecfc2),ecfc3),ecfc4),ecfc5),ecfc6),ecfc7),ecfc8),en2o1),en2o2),en2o3),ech4),eco21),eco22);
// MPAS module_ra_cam_support.F:1289
}
// MPAS module_ra_cam_support.F:1291
return;
}
// MPAS module_ra_cam_support.F:1295-1429
CAM_ROUTINE void trcmix(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const CamArray<double>& _pmid,const CamArray<double>& _clat,const CamArray<double>& _n2o,const CamArray<double>& _ch4,const CamArray<double>& _cfc11,const CamArray<double>& _cfc12) {
CamFrame _frame(arena);
CAM_TRACE("trcmix.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("trcmix.clat");
auto clat=_clat.rebound(1,pcols);
CAM_TRACE("trcmix.n2o");
auto n2o=_n2o.rebound(1,pcols,1,pver);
CAM_TRACE("trcmix.ch4");
auto ch4=_ch4.rebound(1,pcols,1,pver);
CAM_TRACE("trcmix.cfc11");
auto cfc11=_cfc11.rebound(1,pcols,1,pver);
CAM_TRACE("trcmix.cfc12");
auto cfc12=_cfc12.rebound(1,pcols,1,pver);
double rmwn2o;
double rmwch4;
double rmwf11;
double rmwf12;
int i;
int k;
auto coslat=arena.array<double>(1,pcols);
double dlat;
double ptrop;
double pratio;
double xn2o;
double xch4;
double xcfc11;
double xcfc12;
double ch40;
double n2o0;
double cfc110;
double cfc120;
// MPAS module_ra_cam_support.F:1368
rmwn2o=cam_div(mwn2o,mwdry);
// MPAS module_ra_cam_support.F:1369
rmwch4=cam_div(mwch4,mwdry);
// MPAS module_ra_cam_support.F:1370
rmwf11=cam_div(mwf11,mwdry);
// MPAS module_ra_cam_support.F:1371
rmwf12=cam_div(mwf12,mwdry);
// MPAS module_ra_cam_support.F:1376
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1377
coslat(i)=cam_cos(clat(i));
// MPAS module_ra_cam_support.F:1378
}
// MPAS module_ra_cam_support.F:1382
ch40=cam_mul(rmwch4,ch4vmr);
// MPAS module_ra_cam_support.F:1383
n2o0=cam_mul(rmwn2o,n2ovmr);
// MPAS module_ra_cam_support.F:1384
cfc110=cam_mul(rmwf11,f11vmr);
// MPAS module_ra_cam_support.F:1385
cfc120=cam_mul(rmwf12,f12vmr);
// MPAS module_ra_cam_support.F:1387
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1388
coslat(i)=cam_cos(clat(i));
// MPAS module_ra_cam_support.F:1389
}
// MPAS module_ra_cam_support.F:1391
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:1392
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1395
dlat=cam_abs(cam_mul(57.2958f,clat(i)));
// MPAS module_ra_cam_support.F:1396
if((dlat<=45.0f)) {
// MPAS module_ra_cam_support.F:1397
xn2o=cam_add(0.3478f,cam_mul(0.00116f,dlat));
// MPAS module_ra_cam_support.F:1398
xch4=0.2353f;
// MPAS module_ra_cam_support.F:1399
xcfc11=cam_add(0.7273f,cam_mul(0.00606f,dlat));
// MPAS module_ra_cam_support.F:1400
xcfc12=cam_add(0.4000f,cam_mul(0.00222f,dlat));
// MPAS module_ra_cam_support.F:1401
} else {
// MPAS module_ra_cam_support.F:1402
xn2o=cam_add(0.4000f,cam_mul(0.013333f,(cam_sub(dlat,45))));
// MPAS module_ra_cam_support.F:1403
xch4=cam_add(0.2353f,cam_mul(0.0225489f,(cam_sub(dlat,45))));
// MPAS module_ra_cam_support.F:1404
xcfc11=cam_add(1.00f,cam_mul(0.013333f,(cam_sub(dlat,45))));
// MPAS module_ra_cam_support.F:1405
xcfc12=cam_add(0.50f,cam_mul(0.024444f,(cam_sub(dlat,45))));
// MPAS module_ra_cam_support.F:1406
}
// MPAS module_ra_cam_support.F:1409
ptrop=cam_sub(250.0e2f,cam_mul(150.0e2f,cam_pow(coslat(i),2.0f)));
// MPAS module_ra_cam_support.F:1412
if((pmid(i,k)>=ptrop)) {
// MPAS module_ra_cam_support.F:1413
ch4(i,k)=ch40;
// MPAS module_ra_cam_support.F:1414
n2o(i,k)=n2o0;
// MPAS module_ra_cam_support.F:1415
cfc11(i,k)=cfc110;
// MPAS module_ra_cam_support.F:1416
cfc12(i,k)=cfc120;
// MPAS module_ra_cam_support.F:1417
} else {
// MPAS module_ra_cam_support.F:1418
pratio=cam_div(pmid(i,k),ptrop);
// MPAS module_ra_cam_support.F:1419
ch4(i,k)=cam_mul(ch40,cam_pow((pratio),xch4));
// MPAS module_ra_cam_support.F:1420
n2o(i,k)=cam_mul(n2o0,cam_pow((pratio),xn2o));
// MPAS module_ra_cam_support.F:1421
cfc11(i,k)=cam_mul(cfc110,cam_pow((pratio),xcfc11));
// MPAS module_ra_cam_support.F:1422
cfc12(i,k)=cam_mul(cfc120,cam_pow((pratio),xcfc12));
// MPAS module_ra_cam_support.F:1423
}
// MPAS module_ra_cam_support.F:1424
}
// MPAS module_ra_cam_support.F:1425
}
// MPAS module_ra_cam_support.F:1427
return;
}
// MPAS module_ra_cam_support.F:1431-1521
CAM_ROUTINE void trcplk(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _tint,const CamArray<double>& _tlayr,const CamArray<double>& _tplnke,const CamArray<double>& _emplnk,const CamArray<double>& _abplnk1,const CamArray<double>& _abplnk2) {
CamFrame _frame(arena);
CAM_TRACE("trcplk.tint");
auto tint=_tint.rebound(1,pcols,1,pverp);
CAM_TRACE("trcplk.tlayr");
auto tlayr=_tlayr.rebound(1,pcols,1,pverp);
CAM_TRACE("trcplk.tplnke");
auto tplnke=_tplnke.rebound(1,pcols);
CAM_TRACE("trcplk.emplnk");
auto emplnk=_emplnk.rebound(1,14,1,pcols);
CAM_TRACE("trcplk.abplnk1");
auto abplnk1=_abplnk1.rebound(1,14,1,pcols,1,pverp);
CAM_TRACE("trcplk.abplnk2");
auto abplnk2=_abplnk2.rebound(1,14,1,pcols,1,pverp);
int wvl;
int i;
int k;
auto f1=arena.array<double>(1,14);
{ const double _init[]={5.85713e8f,7.94950e8f,1.47009e9f,1.40031e9f,1.34853e8f,1.05158e9f,3.35370e8f,3.99601e8f,5.35994e8f,8.42955e8f,4.63682e8f,5.18944e8f,8.83202e8f,1.03279e9f}; f1.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1431-1521: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[432]) f1=oracle_array<double>(432);
auto f2=arena.array<double>(1,14);
{ const double _init[]={2.02493e11f,3.04286e11f,6.90698e11f,6.47333e11f,2.85744e10f,4.41862e11f,9.62780e10f,1.21618e11f,1.79905e11f,3.29029e11f,1.48294e11f,1.72315e11f,3.50140e11f,4.31364e11f}; f2.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1431-1521: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[450]) f2=oracle_array<double>(450);
auto f3=arena.array<double>(1,14);
{ const double _init[]={1383.0f,1531.0f,1879.0f,1849.0f,848.0f,1681.0f,1148.0f,1217.0f,1343.0f,1561.0f,1279.0f,1328.0f,1586.0f,1671.0f}; f3.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:1431-1521: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[468]) f3=oracle_array<double>(468);
// MPAS module_ra_cam_support.F:1495
for(wvl=1;wvl<=14;wvl+=1) {
// MPAS module_ra_cam_support.F:1496
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1497
emplnk(wvl,i)=cam_div(f1(wvl),(cam_mul(cam_pow(tplnke(i),4.0f),(cam_sub(cam_exp(cam_div(f3(wvl),tplnke(i))),1.0f)))));
// MPAS module_ra_cam_support.F:1498
}
// MPAS module_ra_cam_support.F:1499
}
// MPAS module_ra_cam_support.F:1503
for(wvl=1;wvl<=14;wvl+=1) {
// MPAS module_ra_cam_support.F:1504
for(k=ntoplw;k<=pverp;k+=1) {
// MPAS module_ra_cam_support.F:1505
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1509
abplnk1(wvl,i,k)=cam_div((cam_mul(f2(wvl),cam_exp(cam_div(f3(wvl),tint(i,k))))),(cam_mul(cam_pow(tint(i,k),5.0f),cam_pow((cam_sub(cam_exp(cam_div(f3(wvl),tint(i,k))),1.0f)),2.0f))));
// MPAS module_ra_cam_support.F:1514
abplnk2(wvl,i,k)=cam_div((cam_mul(f2(wvl),cam_exp(cam_div(f3(wvl),tlayr(i,k))))),(cam_mul(cam_pow(tlayr(i,k),5.0f),cam_pow((cam_sub(cam_exp(cam_div(f3(wvl),tlayr(i,k))),1.0f)),2.0f))));
// MPAS module_ra_cam_support.F:1516
}
// MPAS module_ra_cam_support.F:1517
}
// MPAS module_ra_cam_support.F:1518
}
// MPAS module_ra_cam_support.F:1520
return;
}
// MPAS module_ra_cam_support.F:1523-1673
CAM_ROUTINE void trcpth(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _tnm,const CamArray<double>& _pnm,const CamArray<double>& _cfc11,const CamArray<double>& _cfc12,const CamArray<double>& _n2o,const CamArray<double>& _ch4,const CamArray<double>& _qnm,const CamArray<double>& _ucfc11,const CamArray<double>& _ucfc12,const CamArray<double>& _un2o0,const CamArray<double>& _un2o1,const CamArray<double>& _uch4,const CamArray<double>& _uco211,const CamArray<double>& _uco212,const CamArray<double>& _uco213,const CamArray<double>& _uco221,const CamArray<double>& _uco222,const CamArray<double>& _uco223,const CamArray<double>& _bn2o0,const CamArray<double>& _bn2o1,const CamArray<double>& _bch4,const CamArray<double>& _uptype) {
CamFrame _frame(arena);
CAM_TRACE("trcpth.tnm");
auto tnm=_tnm.rebound(1,pcols,1,pver);
CAM_TRACE("trcpth.pnm");
auto pnm=_pnm.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.cfc11");
auto cfc11=_cfc11.rebound(1,pcols,1,pver);
CAM_TRACE("trcpth.cfc12");
auto cfc12=_cfc12.rebound(1,pcols,1,pver);
CAM_TRACE("trcpth.n2o");
auto n2o=_n2o.rebound(1,pcols,1,pver);
CAM_TRACE("trcpth.ch4");
auto ch4=_ch4.rebound(1,pcols,1,pver);
CAM_TRACE("trcpth.qnm");
auto qnm=_qnm.rebound(1,pcols,1,pver);
CAM_TRACE("trcpth.ucfc11");
auto ucfc11=_ucfc11.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.ucfc12");
auto ucfc12=_ucfc12.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.un2o0");
auto un2o0=_un2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.un2o1");
auto un2o1=_un2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.uch4");
auto uch4=_uch4.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.uco211");
auto uco211=_uco211.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.uco212");
auto uco212=_uco212.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.uco213");
auto uco213=_uco213.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.uco221");
auto uco221=_uco221.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.uco222");
auto uco222=_uco222.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.uco223");
auto uco223=_uco223.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.bn2o0");
auto bn2o0=_bn2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.bn2o1");
auto bn2o1=_bn2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.bch4");
auto bch4=_bch4.rebound(1,pcols,1,pverp);
CAM_TRACE("trcpth.uptype");
auto uptype=_uptype.rebound(1,pcols,1,pverp);
int i;
int k;
auto co2fac=arena.array<double>(1,pcols,1,1);
auto alpha1=arena.array<double>(1,pcols);
auto alpha2=arena.array<double>(1,pcols);
auto rt=arena.array<double>(1,pcols);
auto rsqrt=arena.array<double>(1,pcols);
auto pbar=arena.array<double>(1,pcols);
auto dpnm=arena.array<double>(1,pcols);
// MPAS module_ra_cam_support.F:1523-1673: native local SAVE/DATA storage.
double _oracle_default_diff=1.66f;
double& diff=(oracle_locals && oracle_locals[486]) ? *reinterpret_cast<double*>(oracle_locals[486]) : _oracle_default_diff;
// MPAS module_ra_cam_support.F:1609
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1610
ucfc11(i,ntoplw)=cam_mul(cam_mul(cam_mul(1.8f,cfc11(i,ntoplw)),pnm(i,ntoplw)),rga);
// MPAS module_ra_cam_support.F:1611
ucfc12(i,ntoplw)=cam_mul(cam_mul(cam_mul(1.8f,cfc12(i,ntoplw)),pnm(i,ntoplw)),rga);
// MPAS module_ra_cam_support.F:1612
un2o0(i,ntoplw)=cam_div(cam_mul(cam_mul(cam_mul(cam_mul(diff,1.02346e5f),n2o(i,ntoplw)),pnm(i,ntoplw)),rga),cam_sqrt(tnm(i,ntoplw)));
// MPAS module_ra_cam_support.F:1613
un2o1(i,ntoplw)=cam_mul(cam_mul(cam_mul(diff,2.01909f),un2o0(i,ntoplw)),cam_exp(cam_neg(cam_div(847.36f,tnm(i,ntoplw)))));
// MPAS module_ra_cam_support.F:1614
uch4(i,ntoplw)=cam_div(cam_mul(cam_mul(cam_mul(cam_mul(diff,8.60957e4f),ch4(i,ntoplw)),pnm(i,ntoplw)),rga),cam_sqrt(tnm(i,ntoplw)));
// MPAS module_ra_cam_support.F:1615
co2fac(i,1)=cam_mul(cam_mul(cam_mul(diff,co2mmr),pnm(i,ntoplw)),rga);
// MPAS module_ra_cam_support.F:1616
alpha1(i)=cam_div(cam_pow((cam_sub(1.0f,cam_exp(cam_neg(cam_div(1540.0f,tnm(i,ntoplw)))))),3.0f),cam_sqrt(tnm(i,ntoplw)));
// MPAS module_ra_cam_support.F:1617
alpha2(i)=cam_div(cam_pow((cam_sub(1.0f,cam_exp(cam_neg(cam_div(1360.0f,tnm(i,ntoplw)))))),3.0f),cam_sqrt(tnm(i,ntoplw)));
// MPAS module_ra_cam_support.F:1618
uco211(i,ntoplw)=cam_mul(cam_mul(cam_mul(3.42217e3f,co2fac(i,1)),alpha1(i)),cam_exp(cam_neg(cam_div(1849.7f,tnm(i,ntoplw)))));
// MPAS module_ra_cam_support.F:1619
uco212(i,ntoplw)=cam_mul(cam_mul(cam_mul(6.02454e3f,co2fac(i,1)),alpha1(i)),cam_exp(cam_neg(cam_div(2782.1f,tnm(i,ntoplw)))));
// MPAS module_ra_cam_support.F:1620
uco213(i,ntoplw)=cam_mul(cam_mul(cam_mul(5.53143e3f,co2fac(i,1)),alpha1(i)),cam_exp(cam_neg(cam_div(3723.2f,tnm(i,ntoplw)))));
// MPAS module_ra_cam_support.F:1621
uco221(i,ntoplw)=cam_mul(cam_mul(cam_mul(3.88984e3f,co2fac(i,1)),alpha2(i)),cam_exp(cam_neg(cam_div(1997.6f,tnm(i,ntoplw)))));
// MPAS module_ra_cam_support.F:1622
uco222(i,ntoplw)=cam_mul(cam_mul(cam_mul(3.67108e3f,co2fac(i,1)),alpha2(i)),cam_exp(cam_neg(cam_div(3843.8f,tnm(i,ntoplw)))));
// MPAS module_ra_cam_support.F:1623
uco223(i,ntoplw)=cam_mul(cam_mul(cam_mul(6.50642e3f,co2fac(i,1)),alpha2(i)),cam_exp(cam_neg(cam_div(2989.7f,tnm(i,ntoplw)))));
// MPAS module_ra_cam_support.F:1624
bn2o0(i,ntoplw)=cam_div(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(diff,19.399f),cam_pow(pnm(i,ntoplw),2.0f)),n2o(i,ntoplw)),1.02346e5f),rga),(cam_mul(sslp,tnm(i,ntoplw))));
// MPAS module_ra_cam_support.F:1626
bn2o1(i,ntoplw)=cam_mul(cam_mul(bn2o0(i,ntoplw),cam_exp(cam_neg(cam_div(847.36f,tnm(i,ntoplw))))),2.06646e5f);
// MPAS module_ra_cam_support.F:1627
bch4(i,ntoplw)=cam_div(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(diff,2.94449f),ch4(i,ntoplw)),cam_pow(pnm(i,ntoplw),2.0f)),rga),8.60957e4f),(cam_mul(sslp,tnm(i,ntoplw))));
// MPAS module_ra_cam_support.F:1629
uptype(i,ntoplw)=cam_div(cam_mul(cam_mul(cam_mul(cam_mul(diff,qnm(i,ntoplw)),cam_pow(pnm(i,ntoplw),2.0f)),cam_exp(cam_mul(1800.0f,(cam_sub(cam_div(1.0f,tnm(i,ntoplw)),cam_div(1.0f,296.0f)))))),rga),sslp);
// MPAS module_ra_cam_support.F:1631
}
// MPAS module_ra_cam_support.F:1635
for(k=ntoplw;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:1636
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1637
rt(i)=cam_div(1.f,tnm(i,k));
// MPAS module_ra_cam_support.F:1638
rsqrt(i)=cam_sqrt(rt(i));
// MPAS module_ra_cam_support.F:1639
pbar(i)=cam_div(cam_mul(0.5f,(cam_add(pnm(i,cam_add(k,1)),pnm(i,k)))),sslp);
// MPAS module_ra_cam_support.F:1640
dpnm(i)=cam_mul((cam_sub(pnm(i,cam_add(k,1)),pnm(i,k))),rga);
// MPAS module_ra_cam_support.F:1641
alpha1(i)=cam_mul(cam_mul(diff,rsqrt(i)),cam_pow((cam_sub(1.0f,cam_exp(cam_neg(cam_div(1540.0f,tnm(i,k)))))),3.0f));
// MPAS module_ra_cam_support.F:1642
alpha2(i)=cam_mul(cam_mul(diff,rsqrt(i)),cam_pow((cam_sub(1.0f,cam_exp(cam_neg(cam_div(1360.0f,tnm(i,k)))))),3.0f));
// MPAS module_ra_cam_support.F:1643
ucfc11(i,cam_add(k,1))=cam_add(ucfc11(i,k),cam_mul(cam_mul(1.8f,cfc11(i,k)),dpnm(i)));
// MPAS module_ra_cam_support.F:1644
ucfc12(i,cam_add(k,1))=cam_add(ucfc12(i,k),cam_mul(cam_mul(1.8f,cfc12(i,k)),dpnm(i)));
// MPAS module_ra_cam_support.F:1645
un2o0(i,cam_add(k,1))=cam_add(un2o0(i,k),cam_mul(cam_mul(cam_mul(cam_mul(diff,1.02346e5f),n2o(i,k)),rsqrt(i)),dpnm(i)));
// MPAS module_ra_cam_support.F:1646
un2o1(i,cam_add(k,1))=cam_add(un2o1(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(diff,2.06646e5f),n2o(i,k)),rsqrt(i)),cam_exp(cam_neg(cam_div(847.36f,tnm(i,k))))),dpnm(i)));
// MPAS module_ra_cam_support.F:1648
uch4(i,cam_add(k,1))=cam_add(uch4(i,k),cam_mul(cam_mul(cam_mul(cam_mul(diff,8.60957e4f),ch4(i,k)),rsqrt(i)),dpnm(i)));
// MPAS module_ra_cam_support.F:1649
uco211(i,cam_add(k,1))=cam_add(uco211(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(1.15f,3.42217e3f),alpha1(i)),co2mmr),cam_exp(cam_neg(cam_div(1849.7f,tnm(i,k))))),dpnm(i)));
// MPAS module_ra_cam_support.F:1651
uco212(i,cam_add(k,1))=cam_add(uco212(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(1.15f,6.02454e3f),alpha1(i)),co2mmr),cam_exp(cam_neg(cam_div(2782.1f,tnm(i,k))))),dpnm(i)));
// MPAS module_ra_cam_support.F:1653
uco213(i,cam_add(k,1))=cam_add(uco213(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(1.15f,5.53143e3f),alpha1(i)),co2mmr),cam_exp(cam_neg(cam_div(3723.2f,tnm(i,k))))),dpnm(i)));
// MPAS module_ra_cam_support.F:1655
uco221(i,cam_add(k,1))=cam_add(uco221(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(1.15f,3.88984e3f),alpha2(i)),co2mmr),cam_exp(cam_neg(cam_div(1997.6f,tnm(i,k))))),dpnm(i)));
// MPAS module_ra_cam_support.F:1657
uco222(i,cam_add(k,1))=cam_add(uco222(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(1.15f,3.67108e3f),alpha2(i)),co2mmr),cam_exp(cam_neg(cam_div(3843.8f,tnm(i,k))))),dpnm(i)));
// MPAS module_ra_cam_support.F:1659
uco223(i,cam_add(k,1))=cam_add(uco223(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(1.15f,6.50642e3f),alpha2(i)),co2mmr),cam_exp(cam_neg(cam_div(2989.7f,tnm(i,k))))),dpnm(i)));
// MPAS module_ra_cam_support.F:1661
bn2o0(i,cam_add(k,1))=cam_add(bn2o0(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(diff,19.399f),pbar(i)),rt(i)),1.02346e5f),n2o(i,k)),dpnm(i)));
// MPAS module_ra_cam_support.F:1663
bn2o1(i,cam_add(k,1))=cam_add(bn2o1(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(diff,19.399f),pbar(i)),rt(i)),2.06646e5f),cam_exp(cam_neg(cam_div(847.36f,tnm(i,k))))),n2o(i,k)),dpnm(i)));
// MPAS module_ra_cam_support.F:1665
bch4(i,cam_add(k,1))=cam_add(bch4(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(diff,2.94449f),rt(i)),pbar(i)),8.60957e4f),ch4(i,k)),dpnm(i)));
// MPAS module_ra_cam_support.F:1667
uptype(i,cam_add(k,1))=cam_add(uptype(i,k),cam_mul(cam_mul(cam_mul(cam_mul(diff,qnm(i,k)),cam_exp(cam_mul(1800.0f,(cam_sub(cam_div(1.0f,tnm(i,k)),cam_div(1.0f,296.0f)))))),pbar(i)),dpnm(i)));
// MPAS module_ra_cam_support.F:1669
}
// MPAS module_ra_cam_support.F:1670
}
// MPAS module_ra_cam_support.F:1672
return;
}
// MPAS module_ra_cam_support.F:1677-1740
CAM_ROUTINE void aqsat(const CamArray<double>& _t,const CamArray<double>& _p,const CamArray<double>& _es,const CamArray<double>& _qs,const int& ii,const int& ilen,const int& kk,const int& kstart,const int& kend) {
CamFrame _frame(arena);
CAM_TRACE("aqsat.t");
auto t=_t.rebound(1,ii,1,kk);
CAM_TRACE("aqsat.p");
auto p=_p.rebound(1,ii,1,kk);
CAM_TRACE("aqsat.es");
auto es=_es.rebound(1,ii,1,kk);
CAM_TRACE("aqsat.qs");
auto qs=_qs.rebound(1,ii,1,kk);
double omeps;
int i;
int k;
// MPAS module_ra_cam_support.F:1718
omeps=cam_sub(1.0f,epsqs);
// MPAS module_ra_cam_support.F:1719
for(k=kstart;k<=kend;k+=1) {
// MPAS module_ra_cam_support.F:1720
for(i=1;i<=ilen;i+=1) {
// MPAS module_ra_cam_support.F:1721
es(i,k)=estblf(t(i,k));
// MPAS module_ra_cam_support.F:1725
qs(i,k)=cam_div(cam_mul(epsqs,es(i,k)),(cam_sub(p(i,k),cam_mul(omeps,es(i,k)))));
// MPAS module_ra_cam_support.F:1730
qs(i,k)=cam_min(1.0,qs(i,k));
// MPAS module_ra_cam_support.F:1732
if((qs(i,k)<0.0f)) {
// MPAS module_ra_cam_support.F:1733
qs(i,k)=1.0f;
// MPAS module_ra_cam_support.F:1734
es(i,k)=p(i,k);
// MPAS module_ra_cam_support.F:1735
}
// MPAS module_ra_cam_support.F:1736
}
// MPAS module_ra_cam_support.F:1737
}
// MPAS module_ra_cam_support.F:1739
return;
}
// MPAS module_ra_cam_support.F:1742-1789
CAM_ROUTINE void cldefr(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _landfrac,const CamArray<double>& _t,const CamArray<double>& _rel,const CamArray<double>& _rei,const CamArray<double>& _ps,const CamArray<double>& _pmid,const CamArray<double>& _landm,const CamArray<double>& _icefrac,const CamArray<double>& _snowh) {
CamFrame _frame(arena);
CAM_TRACE("cldefr.landfrac");
auto landfrac=_landfrac.rebound(1,pcols);
CAM_TRACE("cldefr.t");
auto t=_t.rebound(1,pcols,1,pver);
CAM_TRACE("cldefr.rel");
auto rel=_rel.rebound(1,pcols,1,pver);
CAM_TRACE("cldefr.rei");
auto rei=_rei.rebound(1,pcols,1,pver);
CAM_TRACE("cldefr.ps");
auto ps=_ps.rebound(1,pcols);
CAM_TRACE("cldefr.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("cldefr.landm");
auto landm=_landm.rebound(1,pcols);
CAM_TRACE("cldefr.icefrac");
auto icefrac=_icefrac.rebound(1,pcols);
CAM_TRACE("cldefr.snowh");
auto snowh=_snowh.rebound(1,pcols);
// MPAS module_ra_cam_support.F:1781
reltab(ncol,pcols,pver,t,landfrac,landm,icefrac,rel,snowh); if(error) return;
// MPAS module_ra_cam_support.F:1784
reitab(ncol,pcols,pver,t,rei); if(error) return;
// MPAS module_ra_cam_support.F:1788
return;
}
// MPAS module_ra_cam_support.F:1792-1871
CAM_ROUTINE void background(const int& lchnk,const int& ncol,const CamArray<double>& _pint,const int& pcols,const int& pverr,const int& pverrp,const CamArray<double>& _mmr) {
CamFrame _frame(arena);
CAM_TRACE("background.pint");
auto pint=_pint.rebound(1,pcols,1,pverrp);
CAM_TRACE("background.mmr");
auto mmr=_mmr.rebound(1,pcols,1,pverr);
int i;
int k;
double mass2mmr;
double mass;
// MPAS module_ra_cam_support.F:1846
if((mxaerl==0)) {error=1846; return;}
// MPAS module_ra_cam_support.F:1850
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1851
mass2mmr=cam_div(gravmks,(cam_sub(pint(i,pverrp),pint(i,cam_sub(pverrp,mxaerl)))));
// MPAS module_ra_cam_support.F:1852
for(k=1;k<=pverr;k+=1) {
// MPAS module_ra_cam_support.F:1857
if((k>=cam_sub(pverrp,mxaerl))) {
// MPAS module_ra_cam_support.F:1861
mass=cam_div(tauback,(cam_mul(1.e3f,kbg(idxvis))));
// MPAS module_ra_cam_support.F:1862
mmr(i,k)=cam_mul(mass2mmr,mass);
// MPAS module_ra_cam_support.F:1863
} else {
// MPAS module_ra_cam_support.F:1864
mmr(i,k)=0.;
// MPAS module_ra_cam_support.F:1865
}
// MPAS module_ra_cam_support.F:1867
}
// MPAS module_ra_cam_support.F:1868
}
// MPAS module_ra_cam_support.F:1870
return;
}
// MPAS module_ra_cam_support.F:1873-1888
CAM_ROUTINE void scale_aerosols(const CamArray<double>& _aerosolt,const int& pcols,const int& pver,const int& ncol,const int& lchnk,const CamArray<double>& _scale) {
CamFrame _frame(arena);
CAM_TRACE("scale_aerosols.aerosolt");
auto aerosolt=_aerosolt.rebound(1,pcols,1,pver,1,naer_all);
CAM_TRACE("scale_aerosols.scale");
auto scale=_scale.rebound(1,naer_all);
int m;
// MPAS module_ra_cam_support.F:1883
for(m=1;m<=naer_all;m+=1) {
// MPAS module_ra_cam_support.F:1884
{ auto _lhs=aerosolt.section(cam_section(1,ncol),cam_section(1,pver),m); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_mul(scale(m),aerosolt.section(cam_section(1,ncol),cam_section(1,pver),m).linear(_slice)); }
// MPAS module_ra_cam_support.F:1885
}
// MPAS module_ra_cam_support.F:1887
return;
}
// MPAS module_ra_cam_support.F:1890-1912
CAM_ROUTINE void get_int_scales(const CamArray<double>& _scales) {
CamFrame _frame(arena);
CAM_TRACE("get_int_scales.scales");
auto scales=_scales.rebound(1,naer_all);
int i;
// MPAS module_ra_cam_support.F:1895
{ auto _lhs=scales; for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=1.f; }
// MPAS module_ra_cam_support.F:1897
scales(idxbg)=1.;
// MPAS module_ra_cam_support.F:1898
scales(idxsul)=sulscl;
// MPAS module_ra_cam_support.F:1899
scales(idxsslt)=ssltscl;
// MPAS module_ra_cam_support.F:1901
for(i=idxcarbonfirst;i<=cam_sub(cam_add(idxcarbonfirst,numcarbon),1);i+=1) {
// MPAS module_ra_cam_support.F:1902
scales(i)=carscl;
// MPAS module_ra_cam_support.F:1903
}
// MPAS module_ra_cam_support.F:1905
for(i=idxdustfirst;i<=cam_sub(cam_add(idxdustfirst,numdust),1);i+=1) {
// MPAS module_ra_cam_support.F:1906
scales(i)=dustscl;
// MPAS module_ra_cam_support.F:1907
}
// MPAS module_ra_cam_support.F:1909
scales(idxvolc)=volcscl;
// MPAS module_ra_cam_support.F:1911
return;
}
// MPAS module_ra_cam_support.F:1914-2098
CAM_ROUTINE void vert_interpolate(const CamArray<double>& _match_ps,const CamArray<double>& _aerosolc,const CamArray<double>& _m_hybi,const int& paerlev,const int& naer_c,const CamArray<double>& _pint,const int& n,const CamArray<double>& _aerosol_mmr,const int& pcols,const int& pver,const int& pverp,const int& ncol,const int& c) {
CamFrame _frame(arena);
CAM_TRACE("vert_interpolate.match_ps");
auto match_ps=_match_ps.rebound(1,pcols);
CAM_TRACE("vert_interpolate.aerosolc");
auto aerosolc=_aerosolc.rebound(1,pcols,1,paerlev,1,naer_c);
CAM_TRACE("vert_interpolate.m_hybi");
auto m_hybi=_m_hybi.rebound(1,paerlev);
CAM_TRACE("vert_interpolate.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("vert_interpolate.aerosol_mmr");
auto aerosol_mmr=_aerosol_mmr.rebound(1,pcols,1,pver,1,naer);
int m;
auto kupper=arena.array<int>(1,pcols);
int i;
int k;
int kk;
int kkstart;
int kount;
int isv;
int ksv;
int msv;
bool bad;
bool lev_interp_comp;
auto aerosol=arena.array<double>(1,pcols,1,pverp,1,naer);
double dpl;
double dpu;
double v_coord;
double m_to_mmr;
double aer_diff;
// MPAS module_ra_cam_support.F:1961
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1962
kupper(i)=1;
// MPAS module_ra_cam_support.F:1963
}
// MPAS module_ra_cam_support.F:1968
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1969
for(m=1;m<=naer;m+=1) {
// MPAS module_ra_cam_support.F:1970
aerosol(i,1,m)=aerosolc(i,1,m);
// MPAS module_ra_cam_support.F:1971
}
// MPAS module_ra_cam_support.F:1972
}
// MPAS module_ra_cam_support.F:1976
for(k=2;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:1981
kkstart=paerlev;
// MPAS module_ra_cam_support.F:1982
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1983
kkstart=cam_min0(kkstart,kupper(i));
// MPAS module_ra_cam_support.F:1984
}
// MPAS module_ra_cam_support.F:1985
kount=0;
// MPAS module_ra_cam_support.F:1992
lev_interp_comp=false;
// MPAS module_ra_cam_support.F:1993
for(kk=kkstart;kk<=cam_sub(paerlev,1);kk+=1) {
// MPAS module_ra_cam_support.F:1994
if((!lev_interp_comp)) {
// MPAS module_ra_cam_support.F:1995
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:1996
v_coord=pint(i,k);
// MPAS module_ra_cam_support.F:1997
if(((cam_mul(m_hybi(kk),match_ps(i))<v_coord)&&(v_coord<=cam_mul(m_hybi(cam_add(kk,1)),match_ps(i))))) {
// MPAS module_ra_cam_support.F:1998
kupper(i)=kk;
// MPAS module_ra_cam_support.F:1999
kount=cam_add(kount,1);
// MPAS module_ra_cam_support.F:2000
}
// MPAS module_ra_cam_support.F:2001
}
// MPAS module_ra_cam_support.F:2008
if((kount==ncol)) {
// MPAS module_ra_cam_support.F:2009
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2010
for(m=1;m<=naer;m+=1) {
// MPAS module_ra_cam_support.F:2011
dpu=cam_sub(pint(i,k),cam_mul(m_hybi(kupper(i)),match_ps(i)));
// MPAS module_ra_cam_support.F:2012
dpl=cam_sub(cam_mul(m_hybi(cam_add(kupper(i),1)),match_ps(i)),pint(i,k));
// MPAS module_ra_cam_support.F:2013
aerosol(i,k,m)=cam_div((cam_add(cam_mul(aerosolc(i,kupper(i),m),dpl),cam_mul(aerosolc(i,cam_add(kupper(i),1),m),dpu))),(cam_add(dpl,dpu)));
// MPAS module_ra_cam_support.F:2016
}
// MPAS module_ra_cam_support.F:2017
}
// MPAS module_ra_cam_support.F:2018
lev_interp_comp=true;
// MPAS module_ra_cam_support.F:2019
}
// MPAS module_ra_cam_support.F:2020
}
// MPAS module_ra_cam_support.F:2021
}
// MPAS module_ra_cam_support.F:2029
if((!lev_interp_comp)) {
// MPAS module_ra_cam_support.F:2030
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2031
for(m=1;m<=naer;m+=1) {
// MPAS module_ra_cam_support.F:2032
if((pint(i,k)<cam_mul(m_hybi(1),match_ps(i)))) {
// MPAS module_ra_cam_support.F:2033
aerosol(i,k,m)=aerosolc(i,1,m);
// MPAS module_ra_cam_support.F:2034
} else if((pint(i,k)>cam_mul(m_hybi(paerlev),match_ps(i)))) {
// MPAS module_ra_cam_support.F:2035
aerosol(i,k,m)=0.0f;
// MPAS module_ra_cam_support.F:2036
} else {
// MPAS module_ra_cam_support.F:2037
dpu=cam_sub(pint(i,k),cam_mul(m_hybi(kupper(i)),match_ps(i)));
// MPAS module_ra_cam_support.F:2038
dpl=cam_sub(cam_mul(m_hybi(cam_add(kupper(i),1)),match_ps(i)),pint(i,k));
// MPAS module_ra_cam_support.F:2039
aerosol(i,k,m)=cam_div((cam_add(cam_mul(aerosolc(i,kupper(i),m),dpl),cam_mul(aerosolc(i,cam_add(kupper(i),1),m),dpu))),(cam_add(dpl,dpu)));
// MPAS module_ra_cam_support.F:2042
}
// MPAS module_ra_cam_support.F:2043
}
// MPAS module_ra_cam_support.F:2044
}
// MPAS module_ra_cam_support.F:2046
if((kount>ncol)) {
// MPAS module_ra_cam_support.F:2048
}
// MPAS module_ra_cam_support.F:2049
}
// MPAS module_ra_cam_support.F:2050
}
// MPAS module_ra_cam_support.F:2056
{ auto _lhs=aerosol.section(cam_section(1,ncol),pverp,cam_section(1,naer)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam_support.F:2061
for(m=1;m<=naer;m+=1) {
// MPAS module_ra_cam_support.F:2062
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:2063
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2064
if((aerosol(i,k,m)<1.e-40)) {aerosol(i,k,m)=0.f;}
// MPAS module_ra_cam_support.F:2065
}
// MPAS module_ra_cam_support.F:2066
}
// MPAS module_ra_cam_support.F:2067
}
// MPAS module_ra_cam_support.F:2074
for(m=1;m<=naer;m+=1) {
// MPAS module_ra_cam_support.F:2075
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:2076
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2077
aer_diff=cam_sub(aerosol(i,k,m),aerosol(i,cam_add(k,1),m));
// MPAS module_ra_cam_support.F:2078
if((cam_abs(aer_diff)<cam_mul(1e-15f,aerosol(i,1,m)))) {
// MPAS module_ra_cam_support.F:2079
aer_diff=0.f;
// MPAS module_ra_cam_support.F:2080
}
// MPAS module_ra_cam_support.F:2081
m_to_mmr=cam_div(gravmks,(cam_sub(pint(i,cam_add(k,1)),pint(i,k))));
// MPAS module_ra_cam_support.F:2082
aerosol_mmr(i,k,m)=cam_mul(aer_diff,m_to_mmr);
// MPAS module_ra_cam_support.F:2083
if((aerosol_mmr(i,k,m)<0)) {
// MPAS module_ra_cam_support.F:2084
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2085
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2086
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2087
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2089
}
// MPAS module_ra_cam_support.F:2090
}
// MPAS module_ra_cam_support.F:2091
}
// MPAS module_ra_cam_support.F:2092
}
// MPAS module_ra_cam_support.F:2097
return;
}
// MPAS module_ra_cam_support.F:2102-2155
CAM_ROUTINE void cldems(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _clwp,const CamArray<double>& _fice,const CamArray<double>& _rei,const CamArray<double>& _emis) {
CamFrame _frame(arena);
const double kabsl=0.090361f;
CAM_TRACE("cldems.clwp");
auto clwp=_clwp.rebound(1,pcols,1,pver);
CAM_TRACE("cldems.fice");
auto fice=_fice.rebound(1,pcols,1,pver);
CAM_TRACE("cldems.rei");
auto rei=_rei.rebound(1,pcols,1,pver);
CAM_TRACE("cldems.emis");
auto emis=_emis.rebound(1,pcols,1,pver);
int i;
int k;
double kabs;
double kabsi;
// MPAS module_ra_cam_support.F:2146
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:2147
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2148
kabsi=cam_add(0.005f,cam_div(1.f,rei(i,k)));
// MPAS module_ra_cam_support.F:2149
kabs=cam_add(cam_mul(kabsl,(cam_sub(1.f,fice(i,k)))),cam_mul(kabsi,fice(i,k)));
// MPAS module_ra_cam_support.F:2150
emis(i,k)=cam_sub(1.f,cam_exp(cam_neg(cam_mul(cam_mul(1.66f,kabs),clwp(i,k)))));
// MPAS module_ra_cam_support.F:2151
}
// MPAS module_ra_cam_support.F:2152
}
// MPAS module_ra_cam_support.F:2154
return;
}
// MPAS module_ra_cam_support.F:2158-2233
CAM_ROUTINE void cldovrlap(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _pint,const CamArray<double>& _cld,const CamArray<int>& _nmxrgn,const CamArray<double>& _pmxrgn) {
CamFrame _frame(arena);
CAM_TRACE("cldovrlap.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("cldovrlap.cld");
auto cld=_cld.rebound(1,pcols,1,pver);
CAM_TRACE("cldovrlap.nmxrgn");
auto nmxrgn=_nmxrgn.rebound(1,pcols);
CAM_TRACE("cldovrlap.pmxrgn");
auto pmxrgn=_pmxrgn.rebound(1,pcols,1,pverp);
int i;
int k;
int n;
auto pnm=arena.array<double>(1,pcols,1,pverp);
bool cld_found;
auto cld_layer=arena.array<bool>(1,pver);
// MPAS module_ra_cam_support.F:2210
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2211
cld_found=false;
// MPAS module_ra_cam_support.F:2212
{ auto _lhs=cld_layer.section(cam_section(1,pver)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=(cld.section(i,cam_section(1,pver)).linear(_slice)>0.0); }
// MPAS module_ra_cam_support.F:2213
{ auto _lhs=pmxrgn.section(i,cam_section(1,pverp)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.0f; }
// MPAS module_ra_cam_support.F:2214
{ auto _lhs=pnm.section(i,cam_section(1,pverp)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_mul(pint.section(i,cam_section(1,pverp)).linear(_slice),10.f); }
// MPAS module_ra_cam_support.F:2215
n=1;
// MPAS module_ra_cam_support.F:2216
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:2217
if((cld_layer(k)&&(!cld_found))) {
// MPAS module_ra_cam_support.F:2218
cld_found=true;
// MPAS module_ra_cam_support.F:2219
} else if(((!cld_layer(k))&&cld_found)) {
// MPAS module_ra_cam_support.F:2220
cld_found=false;
// MPAS module_ra_cam_support.F:2221
if((cam_count(cld_layer.section(cam_section(k,pver)))==0)) {
// MPAS module_ra_cam_support.F:2222
break;
// MPAS module_ra_cam_support.F:2223
}
// MPAS module_ra_cam_support.F:2224
pmxrgn(i,n)=pnm(i,k);
// MPAS module_ra_cam_support.F:2225
n=cam_add(n,1);
// MPAS module_ra_cam_support.F:2226
}
// MPAS module_ra_cam_support.F:2227
}
// MPAS module_ra_cam_support.F:2228
pmxrgn(i,n)=pnm(i,pverp);
// MPAS module_ra_cam_support.F:2229
nmxrgn(i)=n;
// MPAS module_ra_cam_support.F:2230
}
// MPAS module_ra_cam_support.F:2232
return;
}
// MPAS module_ra_cam_support.F:2236-2302
CAM_ROUTINE void cldclw(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _zi,const CamArray<double>& _clwp,const CamArray<double>& _tpw,const CamArray<double>& _hl) {
CamFrame _frame(arena);
CAM_TRACE("cldclw.zi");
auto zi=_zi.rebound(1,pcols,1,pverp);
CAM_TRACE("cldclw.clwp");
auto clwp=_clwp.rebound(1,pcols,1,pver);
CAM_TRACE("cldclw.tpw");
auto tpw=_tpw.rebound(1,pcols);
CAM_TRACE("cldclw.hl");
auto hl=_hl.rebound(1,pcols);
auto rhl=arena.array<double>(1,pcols);
int i;
int k;
double clwc0;
auto emziohl=arena.array<double>(1,pcols,1,pverp);
// MPAS module_ra_cam_support.F:2279
clwc0=0.21f;
// MPAS module_ra_cam_support.F:2283
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2284
hl(i)=cam_mul(700.0f,cam_log(cam_max(cam_add(tpw(i),1.0),1.0)));
// MPAS module_ra_cam_support.F:2285
rhl(i)=cam_div(1.0f,hl(i));
// MPAS module_ra_cam_support.F:2286
}
// MPAS module_ra_cam_support.F:2290
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam_support.F:2291
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2292
emziohl(i,k)=cam_exp(cam_neg(cam_mul(zi(i,k),rhl(i))));
// MPAS module_ra_cam_support.F:2293
}
// MPAS module_ra_cam_support.F:2294
}
// MPAS module_ra_cam_support.F:2295
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:2296
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2297
clwp(i,k)=cam_mul(cam_mul(clwc0,hl(i)),(cam_sub(emziohl(i,cam_add(k,1)),emziohl(i,k))));
// MPAS module_ra_cam_support.F:2298
}
// MPAS module_ra_cam_support.F:2299
}
// MPAS module_ra_cam_support.F:2301
return;
}
// MPAS module_ra_cam_support.F:2306-2364
CAM_ROUTINE void reltab(const int& ncol,const int& pcols,const int& pver,const CamArray<double>& _t,const CamArray<double>& _landfrac,const CamArray<double>& _landm,const CamArray<double>& _icefrac,const CamArray<double>& _rel,const CamArray<double>& _snowh) {
CamFrame _frame(arena);
CAM_TRACE("reltab.t");
auto t=_t.rebound(1,pcols,1,pver);
CAM_TRACE("reltab.landfrac");
auto landfrac=_landfrac.rebound(1,pcols);
CAM_TRACE("reltab.landm");
auto landm=_landm.rebound(1,pcols);
CAM_TRACE("reltab.icefrac");
auto icefrac=_icefrac.rebound(1,pcols);
CAM_TRACE("reltab.rel");
auto rel=_rel.rebound(1,pcols,1,pver);
CAM_TRACE("reltab.snowh");
auto snowh=_snowh.rebound(1,pcols);
int i;
int k;
double rliqland;
double rliqocean;
double rliqice;
// MPAS module_ra_cam_support.F:2346
rliqocean=14.0;
// MPAS module_ra_cam_support.F:2347
rliqice=14.0;
// MPAS module_ra_cam_support.F:2348
rliqland=8.0;
// MPAS module_ra_cam_support.F:2349
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:2350
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2354
rel(i,k)=cam_add(rliqland,cam_mul((cam_sub(rliqocean,rliqland)),cam_min(1.0,cam_max(0.0,cam_mul((cam_sub(tmelt,t(i,k))),0.05f)))));
// MPAS module_ra_cam_support.F:2356
rel(i,k)=cam_add(rel(i,k),cam_mul((cam_sub(rliqocean,rel(i,k))),cam_min(1.0,cam_max(0.0,cam_mul(snowh(i),10.f)))));
// MPAS module_ra_cam_support.F:2358
rel(i,k)=cam_add(rel(i,k),cam_mul((cam_sub(rliqocean,rel(i,k))),cam_min(1.0,cam_max(0.0,cam_sub(1.0f,landm(i))))));
// MPAS module_ra_cam_support.F:2360
rel(i,k)=cam_add(rel(i,k),cam_mul((cam_sub(rliqice,rel(i,k))),cam_min(1.0,cam_max(0.0,icefrac(i)))));
// MPAS module_ra_cam_support.F:2362
}
// MPAS module_ra_cam_support.F:2363
}
}
// MPAS module_ra_cam_support.F:2366-2389
CAM_ROUTINE void reitab(const int& ncol,const int& pcols,const int& pver,const CamArray<double>& _t,const CamArray<double>& _re) {
CamFrame _frame(arena);
CAM_TRACE("reitab.t");
auto t=_t.rebound(1,pcols,1,pver);
CAM_TRACE("reitab.re");
auto re=_re.rebound(1,pcols,1,pver);
double corr;
int i;
int k;
int index;
// MPAS module_ra_cam_support.F:2377
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam_support.F:2378
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam_support.F:2379
index=((int)(cam_sub(t(i,k),179.f)));
// MPAS module_ra_cam_support.F:2380
index=cam_min(cam_max(index,1),94);
// MPAS module_ra_cam_support.F:2381
corr=cam_sub(t(i,k),((int)(t(i,k))));
// MPAS module_ra_cam_support.F:2382
re(i,k)=cam_add(cam_mul(retab(index),(cam_sub(1.f,corr))),cam_mul(retab(cam_add(index,1)),corr));
// MPAS module_ra_cam_support.F:2385
}
// MPAS module_ra_cam_support.F:2386
}
// MPAS module_ra_cam_support.F:2388
return;
}
// MPAS module_ra_cam_support.F:2391-2435
CAM_ROUTINE double exp_interpol(const CamArray<double>& _x,const CamArray<double>& _f,const double& y) {
CamFrame _frame(arena);
CAM_TRACE("exp_interpol.x");
auto x=_x;
CAM_TRACE("exp_interpol.f");
auto f=_f;
double g;
int k;
int n;
double a;
// MPAS module_ra_cam_support.F:2415
n=cam_size(x);
// MPAS module_ra_cam_support.F:2420
if((y<=x(1))) {
// MPAS module_ra_cam_support.F:2421
k=1;
// MPAS module_ra_cam_support.F:2422
} else if((y>=x(n))) {
// MPAS module_ra_cam_support.F:2423
k=cam_sub(n,1);
// MPAS module_ra_cam_support.F:2424
} else {
// MPAS module_ra_cam_support.F:2425
k=1;
// MPAS module_ra_cam_support.F:2426
while(((y>x(cam_add(k,1)))&&(k<n))) {
// MPAS module_ra_cam_support.F:2427
k=cam_add(k,1);
// MPAS module_ra_cam_support.F:2428
}
// MPAS module_ra_cam_support.F:2429
}
// MPAS module_ra_cam_support.F:2432
a=cam_div((cam_log(cam_div(f(cam_add(k,1)),f(k)))),(cam_sub(x(cam_add(k,1)),x(k))));
// MPAS module_ra_cam_support.F:2433
g=cam_mul(f(k),cam_exp(cam_mul(a,(cam_sub(y,x(k))))));
return g;
}
// MPAS module_ra_cam_support.F:2437-2481
CAM_ROUTINE double lin_interpol(const CamArray<double>& _x,const CamArray<double>& _f,const double& y) {
CamFrame _frame(arena);
CAM_TRACE("lin_interpol.x");
auto x=_x;
CAM_TRACE("lin_interpol.f");
auto f=_f;
double g;
int k;
int n;
double a;
// MPAS module_ra_cam_support.F:2461
n=cam_size(x);
// MPAS module_ra_cam_support.F:2466
if((y<=x(1))) {
// MPAS module_ra_cam_support.F:2467
k=1;
// MPAS module_ra_cam_support.F:2468
} else if((y>=x(n))) {
// MPAS module_ra_cam_support.F:2469
k=cam_sub(n,1);
// MPAS module_ra_cam_support.F:2470
} else {
// MPAS module_ra_cam_support.F:2471
k=1;
// MPAS module_ra_cam_support.F:2472
while(((y>x(cam_add(k,1)))&&(k<n))) {
// MPAS module_ra_cam_support.F:2473
k=cam_add(k,1);
// MPAS module_ra_cam_support.F:2474
}
// MPAS module_ra_cam_support.F:2475
}
// MPAS module_ra_cam_support.F:2478
a=cam_div((cam_sub(f(cam_add(k,1)),f(k))),(cam_sub(x(cam_add(k,1)),x(k))));
// MPAS module_ra_cam_support.F:2479
g=cam_add(f(k),cam_mul(a,(cam_sub(y,x(k)))));
return g;
}
// MPAS module_ra_cam_support.F:2483-2525
CAM_ROUTINE float lin_interpol2(const CamArray<float>& _x,const CamArray<float>& _f,const float& y) {
CamFrame _frame(arena);
CAM_TRACE("lin_interpol2.x");
auto x=_x;
CAM_TRACE("lin_interpol2.f");
auto f=_f;
float g;
int k;
int n;
float a;
// MPAS module_ra_cam_support.F:2505
n=cam_size(x);
// MPAS module_ra_cam_support.F:2510
if((y<=x(1))) {
// MPAS module_ra_cam_support.F:2511
k=1;
// MPAS module_ra_cam_support.F:2512
} else if((y>=x(n))) {
// MPAS module_ra_cam_support.F:2513
k=cam_sub(n,1);
// MPAS module_ra_cam_support.F:2514
} else {
// MPAS module_ra_cam_support.F:2515
k=1;
// MPAS module_ra_cam_support.F:2516
while(((y>x(cam_add(k,1)))&&(k<n))) {
// MPAS module_ra_cam_support.F:2517
k=cam_add(k,1);
// MPAS module_ra_cam_support.F:2518
}
// MPAS module_ra_cam_support.F:2519
}
// MPAS module_ra_cam_support.F:2522
a=cam_div((cam_sub(f(cam_add(k,1)),f(k))),(cam_sub(x(cam_add(k,1)),x(k))));
// MPAS module_ra_cam_support.F:2523
g=cam_add(f(k),cam_mul(a,(cam_sub(y,x(k)))));
return g;
}
// MPAS module_ra_cam_support.F:2528-2611
CAM_ROUTINE void getfactors(const bool& cycflag,const int& np1,const double& cdayminus,const double& cdayplus,const double& cday,double& fact1,double& fact2) {
CamFrame _frame(arena);
const double daysperyear=365.f;
double deltat;
// MPAS module_ra_cam_support.F:2575
if(cycflag) {
// MPAS module_ra_cam_support.F:2576
if((((cday<1.f))||((cday>(cam_add(daysperyear,1.f)))))) {
// MPAS module_ra_cam_support.F:2577
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2579
}
// MPAS module_ra_cam_support.F:2580
} else {
// MPAS module_ra_cam_support.F:2581
if((cday<1.f)) {
// MPAS module_ra_cam_support.F:2582
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2584
}
// MPAS module_ra_cam_support.F:2585
}
// MPAS module_ra_cam_support.F:2590
if((cycflag&&(np1==1))) {
// MPAS module_ra_cam_support.F:2591
deltat=cam_sub(cam_add(cdayplus,daysperyear),cdayminus);
// MPAS module_ra_cam_support.F:2592
if((cday>cdayplus)) {
// MPAS module_ra_cam_support.F:2593
fact1=cam_div((cam_sub(cam_add(cdayplus,daysperyear),cday)),deltat);
// MPAS module_ra_cam_support.F:2594
fact2=cam_div((cam_sub(cday,cdayminus)),deltat);
// MPAS module_ra_cam_support.F:2595
} else {
// MPAS module_ra_cam_support.F:2596
fact1=cam_div((cam_sub(cdayplus,cday)),deltat);
// MPAS module_ra_cam_support.F:2597
fact2=cam_div((cam_sub(cam_add(cday,daysperyear),cdayminus)),deltat);
// MPAS module_ra_cam_support.F:2598
}
// MPAS module_ra_cam_support.F:2599
} else {
// MPAS module_ra_cam_support.F:2600
deltat=cam_sub(cdayplus,cdayminus);
// MPAS module_ra_cam_support.F:2601
fact1=cam_div((cam_sub(cdayplus,cday)),deltat);
// MPAS module_ra_cam_support.F:2602
fact2=cam_div((cam_sub(cday,cdayminus)),deltat);
// MPAS module_ra_cam_support.F:2603
}
// MPAS module_ra_cam_support.F:2605
if((!validfactors(fact1,fact2))) {
// MPAS module_ra_cam_support.F:2606
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2608
}
// MPAS module_ra_cam_support.F:2610
return;
}
// MPAS module_ra_cam_support.F:2613-2632
CAM_ROUTINE bool validfactors(const double& fact1,const double& fact2) {
CamFrame _frame(arena);
bool validfactors;
// MPAS module_ra_cam_support.F:2623
validfactors=true;
// MPAS module_ra_cam_support.F:2624
if((((((cam_abs(cam_sub(cam_add(fact1,fact2),1.f))>1.e-6f)||(fact1>1.000001f))||(fact1<cam_neg(1.e-6f)))||(fact2>1.000001f))||(fact2<cam_neg(1.e-6f)))) {
// MPAS module_ra_cam_support.F:2628
validfactors=false;
// MPAS module_ra_cam_support.F:2629
}
// MPAS module_ra_cam_support.F:2631
return validfactors;
return validfactors;
}
// MPAS module_ra_cam_support.F:2634-2667
CAM_ROUTINE void get_rf_scales(const CamArray<double>& _scales) {
CamFrame _frame(arena);
CAM_TRACE("get_rf_scales.scales");
auto scales=_scales.rebound(1,naer_all);
int i;
// MPAS module_ra_cam_support.F:2640
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2641
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2642
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2643
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2644
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2645
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2646
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2647
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2648
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2649
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2650
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2651
error=2651; return;
// MPAS module_ra_cam_support.F:2653
scales(idxbg)=bgscl_rf;
// MPAS module_ra_cam_support.F:2654
scales(idxsul)=sulscl_rf;
// MPAS module_ra_cam_support.F:2655
scales(idxsslt)=ssltscl_rf;
// MPAS module_ra_cam_support.F:2657
for(i=idxcarbonfirst;i<=cam_sub(cam_add(idxcarbonfirst,numcarbon),1);i+=1) {
// MPAS module_ra_cam_support.F:2658
scales(i)=carscl_rf;
// MPAS module_ra_cam_support.F:2659
}
// MPAS module_ra_cam_support.F:2661
for(i=idxdustfirst;i<=cam_sub(cam_add(idxdustfirst,numdust),1);i+=1) {
// MPAS module_ra_cam_support.F:2662
scales(i)=dustscl_rf;
// MPAS module_ra_cam_support.F:2663
}
// MPAS module_ra_cam_support.F:2665
scales(idxvolc)=volcscl_rf;
}
// MPAS module_ra_cam_support.F:2669-2706
CAM_ROUTINE double psi(const double& tpx,const int& iband) {
CamFrame _frame(arena);
double psi;
auto psi_r0=arena.array<double>(1,nbands);
{ const double _init[]={5.65308452e-01f,cam_neg(7.30087891e+01f)}; psi_r0.init(_init,sizeof(_init)/sizeof(_init[0])); }
auto psi_r1=arena.array<double>(1,nbands);
{ const double _init[]={4.07519005e-03f,1.22199547e+00f}; psi_r1.init(_init,sizeof(_init)/sizeof(_init[0])); }
auto psi_r2=arena.array<double>(1,nbands);
{ const double _init[]={cam_neg(1.04347237e-05f),cam_neg(7.12256227e-03f)}; psi_r2.init(_init,sizeof(_init)/sizeof(_init[0])); }
auto psi_r3=arena.array<double>(1,nbands);
{ const double _init[]={1.23765354e-08f,1.47852825e-05f}; psi_r3.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:2705
psi=cam_add(cam_mul((cam_add(cam_mul((cam_add((cam_mul(psi_r3(iband),tpx)),psi_r2(iband))),tpx),psi_r1(iband))),tpx),psi_r0(iband));
return psi;
}
// MPAS module_ra_cam_support.F:2708-2746
CAM_ROUTINE double phi(const double& tpx,const int& iband) {
CamFrame _frame(arena);
double phi;
auto phi_r0=arena.array<double>(1,nbands);
{ const double _init[]={9.60917711e-01f,cam_neg(2.21031342e+01f)}; phi_r0.init(_init,sizeof(_init)/sizeof(_init[0])); }
auto phi_r1=arena.array<double>(1,nbands);
{ const double _init[]={4.86076751e-04f,4.24062610e-01f}; phi_r1.init(_init,sizeof(_init)/sizeof(_init[0])); }
auto phi_r2=arena.array<double>(1,nbands);
{ const double _init[]={cam_neg(1.84806265e-06f),cam_neg(2.95543415e-03f)}; phi_r2.init(_init,sizeof(_init)/sizeof(_init[0])); }
auto phi_r3=arena.array<double>(1,nbands);
{ const double _init[]={2.11239959e-09f,7.52470896e-06f}; phi_r3.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam_support.F:2744
phi=cam_add(cam_mul((cam_add(cam_mul((cam_add((cam_mul(phi_r3(iband),tpx)),phi_r2(iband))),tpx),phi_r1(iband))),tpx),phi_r0(iband));
return phi;
}
// MPAS module_ra_cam_support.F:2748-2776
CAM_ROUTINE double fh2oself(const double& temp) {
CamFrame _frame(arena);
double fh2oself;
// MPAS module_ra_cam_support.F:2775
fh2oself=cam_pow(2.0727484f,(cam_div((cam_sub(296.0f,temp)),36.0f)));
return fh2oself;
}
// MPAS module_ra_cam_support.F:2780-2829
CAM_ROUTINE void esinti(const double& epslon,const double& latvap,const double& latice,const double& rh2o,const double& cpair,const double& tmelt) {
CamFrame _frame(arena);
double tmn;
double tmx;
double trice;
bool ip;
// MPAS module_ra_cam_support.F:2818
tmn=173.16f;
// MPAS module_ra_cam_support.F:2819
tmx=375.16f;
// MPAS module_ra_cam_support.F:2820
trice=20.00f;
// MPAS module_ra_cam_support.F:2821
ip=true;
// MPAS module_ra_cam_support.F:2825
gestbl(tmn,tmx,trice,ip,epslon,latvap,latice,rh2o,cpair,tmelt); if(error) return;
// MPAS module_ra_cam_support.F:2828
return;
}
// MPAS module_ra_cam_support.F:2831-2962
CAM_ROUTINE void gestbl(const double& tmn,const double& tmx,const double& trice,bool& ip,const double& epsil,const double& latvap,const double& latice,const double& rh2o,const double& cpair,const double& tmeltx) {
CamFrame _frame(arena);
double t;
double rgasv;
double cp;
double hlatf;
double ttrice;
double hlatv;
int n;
int lentbl;
int itype;
bool icephs;
// MPAS module_ra_cam_support.F:2883
tmin=tmn;
// MPAS module_ra_cam_support.F:2884
tmax=tmx;
// MPAS module_ra_cam_support.F:2885
ttrice=trice;
// MPAS module_ra_cam_support.F:2886
icephs=ip;
// MPAS module_ra_cam_support.F:2890
epsqs=epsil;
// MPAS module_ra_cam_support.F:2891
hlatv=latvap;
// MPAS module_ra_cam_support.F:2892
hlatf=latice;
// MPAS module_ra_cam_support.F:2893
rgasv=rh2o;
// MPAS module_ra_cam_support.F:2894
cp=cpair;
// MPAS module_ra_cam_support.F:2895
tmelt=tmeltx;
// MPAS module_ra_cam_support.F:2897
lentbl=((int)(cam_add(cam_sub(tmax,tmin),2.000001f)));
// MPAS module_ra_cam_support.F:2898
if((lentbl>plenest)) {
// MPAS module_ra_cam_support.F:2899
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:2901
}
// MPAS module_ra_cam_support.F:2907
if(icephs) {
// MPAS module_ra_cam_support.F:2908
if((ttrice!=0.0f)) {
// MPAS module_ra_cam_support.F:2909
itype=cam_neg(ttrice);
// MPAS module_ra_cam_support.F:2910
} else {
// MPAS module_ra_cam_support.F:2911
itype=1;
// MPAS module_ra_cam_support.F:2912
}
// MPAS module_ra_cam_support.F:2913
} else {
// MPAS module_ra_cam_support.F:2914
itype=0;
// MPAS module_ra_cam_support.F:2915
}
// MPAS module_ra_cam_support.F:2917
t=cam_sub(tmin,1.0f);
// MPAS module_ra_cam_support.F:2918
for(n=1;n<=lentbl;n+=1) {
// MPAS module_ra_cam_support.F:2919
t=cam_add(t,1.0f);
// MPAS module_ra_cam_support.F:2920
gffgch(t,estbl(n),itype); if(error) return;
// MPAS module_ra_cam_support.F:2921
}
// MPAS module_ra_cam_support.F:2923
for(n=cam_add(lentbl,1);n<=plenest;n+=1) {
// MPAS module_ra_cam_support.F:2924
estbl(n)=cam_neg(99999.0f);
// MPAS module_ra_cam_support.F:2925
}
// MPAS module_ra_cam_support.F:2934
pcf(1)=5.04469588506e-01f;
// MPAS module_ra_cam_support.F:2935
pcf(2)=cam_neg(5.47288442819e+00f);
// MPAS module_ra_cam_support.F:2936
pcf(3)=cam_neg(3.67471858735e-01f);
// MPAS module_ra_cam_support.F:2937
pcf(4)=cam_neg(8.95963532403e-03f);
// MPAS module_ra_cam_support.F:2938
pcf(5)=cam_neg(7.78053686625e-05f);
// MPAS module_ra_cam_support.F:2955
return;
// MPAS module_ra_cam_support.F:2957
// MPAS diagnostic only
}
// MPAS module_ra_cam_support.F:2964-3089
CAM_ROUTINE void gffgch(const double& t,double& es,int& itype) {
CamFrame _frame(arena);
double e1;
double e2;
double eswtr;
double f;
double f1;
double f2;
double f3;
double f4;
double f5;
double ps;
double t0;
double term1;
double term2;
double term3;
double tr;
double ts;
double weight;
int itypo;
// MPAS module_ra_cam_support.F:3032
if((itype<0)) {
// MPAS module_ra_cam_support.F:3033
tr=cam_abs(cam_float(itype));
// MPAS module_ra_cam_support.F:3034
itypo=itype;
// MPAS module_ra_cam_support.F:3035
itype=1;
// MPAS module_ra_cam_support.F:3036
} else {
// MPAS module_ra_cam_support.F:3037
tr=0.0f;
// MPAS module_ra_cam_support.F:3038
itypo=itype;
// MPAS module_ra_cam_support.F:3039
}
// MPAS module_ra_cam_support.F:3040
if((tr>40.0f)) {
// MPAS module_ra_cam_support.F:3041
// MPAS diagnostic only
// MPAS module_ra_cam_support.F:3043
}
// MPAS module_ra_cam_support.F:3045
if(((t<(cam_sub(tmelt,tr)))&&(itype==1))) {goto label_10;}
// MPAS module_ra_cam_support.F:3049
ps=1013.246f;
// MPAS module_ra_cam_support.F:3050
ts=373.16f;
// MPAS module_ra_cam_support.F:3051
e1=cam_mul(11.344f,(cam_sub(1.0f,cam_div(t,ts))));
// MPAS module_ra_cam_support.F:3052
e2=cam_neg(cam_mul(3.49149f,(cam_sub(cam_div(ts,t),1.0f))));
// MPAS module_ra_cam_support.F:3053
f1=cam_neg(cam_mul(7.90298f,(cam_sub(cam_div(ts,t),1.0f))));
// MPAS module_ra_cam_support.F:3054
f2=cam_mul(5.02808f,cam_log10(cam_div(ts,t)));
// MPAS module_ra_cam_support.F:3055
f3=cam_neg(cam_div(cam_mul(1.3816f,(cam_sub(cam_pow(10.0f,e1),1.0f))),10000000.0f));
// MPAS module_ra_cam_support.F:3056
f4=cam_div(cam_mul(8.1328f,(cam_sub(cam_pow(10.0f,e2),1.0f))),1000.0f);
// MPAS module_ra_cam_support.F:3057
f5=cam_log10(ps);
// MPAS module_ra_cam_support.F:3058
f=cam_add(cam_add(cam_add(cam_add(f1,f2),f3),f4),f5);
// MPAS module_ra_cam_support.F:3059
es=cam_mul((cam_pow(10.0f,f)),100.0f);
// MPAS module_ra_cam_support.F:3060
eswtr=es;
// MPAS module_ra_cam_support.F:3062
if(((t>=tmelt)||(itype==0))) {goto label_20;}
// MPAS module_ra_cam_support.F:3066
label_10: ; 
// MPAS module_ra_cam_support.F:3067
t0=tmelt;
// MPAS module_ra_cam_support.F:3068
term1=cam_div(2.01889049f,(cam_div(t0,t)));
// MPAS module_ra_cam_support.F:3069
term2=cam_mul(3.56654f,cam_log(cam_div(t0,t)));
// MPAS module_ra_cam_support.F:3070
term3=cam_mul(20.947031f,(cam_div(t0,t)));
// MPAS module_ra_cam_support.F:3071
es=cam_mul(575.185606e10f,cam_exp(cam_neg((cam_add(cam_add(term1,term2),term3)))));
// MPAS module_ra_cam_support.F:3073
if((t<(cam_sub(tmelt,tr)))) {goto label_20;}
// MPAS module_ra_cam_support.F:3077
weight=cam_min(cam_div((cam_sub(tmelt,t)),tr),1.0);
// MPAS module_ra_cam_support.F:3078
es=cam_add(cam_mul(weight,es),cam_mul((cam_sub(1.0f,weight)),eswtr));
// MPAS module_ra_cam_support.F:3080
label_20: ; 
// MPAS module_ra_cam_support.F:3081
itype=itypo;
// MPAS module_ra_cam_support.F:3082
return;
// MPAS module_ra_cam_support.F:3084
// MPAS diagnostic only
}
// MPAS module_ra_cam_support.F:3091-3107
CAM_ROUTINE double estblf(const double& td) {
CamFrame _frame(arena);
double estblf;
double e;
double ai;
int i;
// MPAS module_ra_cam_support.F:3101
e=cam_max(cam_min(td,tmax),tmin);
// MPAS module_ra_cam_support.F:3102
i=cam_add(((int)(cam_sub(e,tmin))),1);
// MPAS module_ra_cam_support.F:3103
ai=cam_aint(cam_sub(e,tmin));
// MPAS module_ra_cam_support.F:3104
estblf=cam_sub(cam_mul((cam_add(cam_sub(cam_add(tmin,ai),e),1.f)),estbl(i)),cam_mul((cam_sub(cam_add(tmin,ai),e)),estbl(cam_add(i,1))));
return estblf;
}
// MPAS module_ra_cam_support.F:3110-3202
CAM_ROUTINE int findvalue(const int& ix,const int& n,const CamArray<double>& _ain,const CamArray<int>& _indxa) {
CamFrame _frame(arena);
CAM_TRACE("findvalue.ain");
auto ain=_ain.rebound(1,n);
CAM_TRACE("findvalue.indxa");
auto indxa=_indxa.rebound(1,n);
int findvalue;
int i;
int j;
int il;
int im;
int ir;
int ia;
int itmp;
// MPAS module_ra_cam_support.F:3146
il=1;
// MPAS module_ra_cam_support.F:3147
ir=n;
// MPAS module_ra_cam_support.F:3148
while(true) {
// MPAS module_ra_cam_support.F:3149
if((cam_sub(ir,il)<=1)) {
// MPAS module_ra_cam_support.F:3150
if((cam_sub(ir,il)==1)) {
// MPAS module_ra_cam_support.F:3151
if((ain(indxa(ir))<ain(indxa(il)))) {
// MPAS module_ra_cam_support.F:3152
itmp=indxa(il);
// MPAS module_ra_cam_support.F:3153
indxa(il)=indxa(ir);
// MPAS module_ra_cam_support.F:3154
indxa(ir)=itmp;
// MPAS module_ra_cam_support.F:3155
}
// MPAS module_ra_cam_support.F:3156
}
// MPAS module_ra_cam_support.F:3157
findvalue=indxa(ix);
// MPAS module_ra_cam_support.F:3158
return findvalue;
// MPAS module_ra_cam_support.F:3159
} else {
// MPAS module_ra_cam_support.F:3160
im=cam_div((cam_add(il,ir)),2);
// MPAS module_ra_cam_support.F:3161
itmp=indxa(im);
// MPAS module_ra_cam_support.F:3162
indxa(im)=indxa(cam_add(il,1));
// MPAS module_ra_cam_support.F:3163
indxa(cam_add(il,1))=itmp;
// MPAS module_ra_cam_support.F:3164
if((ain(indxa(cam_add(il,1)))>ain(indxa(ir)))) {
// MPAS module_ra_cam_support.F:3165
itmp=indxa(cam_add(il,1));
// MPAS module_ra_cam_support.F:3166
indxa(cam_add(il,1))=indxa(ir);
// MPAS module_ra_cam_support.F:3167
indxa(ir)=itmp;
// MPAS module_ra_cam_support.F:3168
}
// MPAS module_ra_cam_support.F:3169
if((ain(indxa(il))>ain(indxa(ir)))) {
// MPAS module_ra_cam_support.F:3170
itmp=indxa(il);
// MPAS module_ra_cam_support.F:3171
indxa(il)=indxa(ir);
// MPAS module_ra_cam_support.F:3172
indxa(ir)=itmp;
// MPAS module_ra_cam_support.F:3173
}
// MPAS module_ra_cam_support.F:3174
if((ain(indxa(cam_add(il,1)))>ain(indxa(il)))) {
// MPAS module_ra_cam_support.F:3175
itmp=indxa(cam_add(il,1));
// MPAS module_ra_cam_support.F:3176
indxa(cam_add(il,1))=indxa(il);
// MPAS module_ra_cam_support.F:3177
indxa(il)=itmp;
// MPAS module_ra_cam_support.F:3178
}
// MPAS module_ra_cam_support.F:3179
i=cam_add(il,1);
// MPAS module_ra_cam_support.F:3180
j=ir;
// MPAS module_ra_cam_support.F:3181
ia=indxa(il);
// MPAS module_ra_cam_support.F:3182
while(true) {
// MPAS module_ra_cam_support.F:3183
while(true) {
// MPAS module_ra_cam_support.F:3184
i=cam_add(i,1);
// MPAS module_ra_cam_support.F:3185
if((ain(indxa(i))>=ain(ia))) {break;}
// MPAS module_ra_cam_support.F:3186
}
// MPAS module_ra_cam_support.F:3187
while(true) {
// MPAS module_ra_cam_support.F:3188
j=cam_sub(j,1);
// MPAS module_ra_cam_support.F:3189
if((ain(indxa(j))<=ain(ia))) {break;}
// MPAS module_ra_cam_support.F:3190
}
// MPAS module_ra_cam_support.F:3191
if((j<i)) {break;}
// MPAS module_ra_cam_support.F:3192
itmp=indxa(i);
// MPAS module_ra_cam_support.F:3193
indxa(i)=indxa(j);
// MPAS module_ra_cam_support.F:3194
indxa(j)=itmp;
// MPAS module_ra_cam_support.F:3195
}
// MPAS module_ra_cam_support.F:3196
indxa(il)=indxa(j);
// MPAS module_ra_cam_support.F:3197
indxa(j)=ia;
// MPAS module_ra_cam_support.F:3198
if((j>=ix)) {ir=cam_sub(j,1);}
// MPAS module_ra_cam_support.F:3199
if((j<=ix)) {il=i;}
// MPAS module_ra_cam_support.F:3200
}
// MPAS module_ra_cam_support.F:3201
}
return findvalue;
}
// MPAS module_ra_cam.F:197-787
CAM_ROUTINE void camrad(const CamArray<float>& _rthratenlw,const CamArray<float>& _rthratensw,const bool& dolw,const bool& dosw,const CamArray<float>& _swupt,const CamArray<float>& _swuptc,const CamArray<float>& _swdnt,const CamArray<float>& _swdntc,const CamArray<float>& _lwupt,const CamArray<float>& _lwuptc,const CamArray<float>& _lwdnt,const CamArray<float>& _lwdntc,const CamArray<float>& _swupb,const CamArray<float>& _swupbc,const CamArray<float>& _swdnb,const CamArray<float>& _swdnbc,const CamArray<float>& _lwupb,const CamArray<float>& _lwupbc,const CamArray<float>& _lwdnb,const CamArray<float>& _lwdnbc,const CamArray<float>& _swcf,const CamArray<float>& _lwcf,const CamArray<float>& _olr,const CamArray<float>& _cemiss,const CamArray<float>& _taucldc,const CamArray<float>& _taucldi,const CamArray<float>& _coszr,const CamArray<float>& _gsw,const CamArray<float>& _glw,const CamArray<float>& _xlat,const CamArray<float>& _xlong,const CamArray<float>& _albedo,const CamArray<float>& _t_phy,const CamArray<float>& _tsk,const CamArray<float>& _emiss,const CamArray<float>& _qv3d,const CamArray<float>& _qc3d,const CamArray<float>& _qr3d,const CamArray<float>& _qi3d,const CamArray<float>& _qs3d,const CamArray<float>& _qg3d,const bool& f_qc,const bool& f_qr,const bool& f_qi,const bool& f_qs,const CamArray<float>& _f_ice_phy,const CamArray<float>& _f_rain_phy,const CamArray<float>& _p_phy,const CamArray<float>& _p8w,const CamArray<float>& _z,const CamArray<float>& _pi_phy,const CamArray<float>& _rho_phy,const CamArray<float>& _dz8w,const CamArray<float>& _cldfra,const CamArray<float>& _xland,const CamArray<float>& _xice,const CamArray<float>& _snow,const CamArray<float>& _ozmixm,const CamArray<float>& _pin0,const int& levsiz,const int& num_months,const CamArray<float>& _m_psp,const CamArray<float>& _m_psn,const CamArray<float>& _aerosolcp,const CamArray<float>& _aerosolcn,const CamArray<float>& _m_hybi0,const int& cam_abs_dim1,const int& cam_abs_dim2,const int& paerlev,const int& naer_c,const float& gmt,const int& julday,const float& julian,const int& yr,const float& dt,const float& xtime,const float& declin,const float& solcon,const float& radt,const float& degrad,const int& n_cldadv,const CamArray<float>& _abstot_3d,const CamArray<float>& _absnxt_3d,const CamArray<float>& _emstot_3d,bool& doabsems,const int& ids,const int& ide,const int& jds,const int& jde,const int& kds,const int& kde,const int& ims,const int& ime,const int& jms,const int& jme,const int& kms,const int& kme,const int& its,const int& ite,const int& jts,const int& jte,const int& kts,const int& kte) {
CamFrame _frame(arena);
const int cam_cache_its=native_chunk_its?native_chunk_its:its;
CAM_TRACE("camrad.rthratenlw");
auto rthratenlw=_rthratenlw.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.rthratensw");
auto rthratensw=_rthratensw.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.swupt");
auto swupt=_swupt.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.swuptc");
auto swuptc=_swuptc.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.swdnt");
auto swdnt=_swdnt.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.swdntc");
auto swdntc=_swdntc.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwupt");
auto lwupt=_lwupt.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwuptc");
auto lwuptc=_lwuptc.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwdnt");
auto lwdnt=_lwdnt.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwdntc");
auto lwdntc=_lwdntc.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.swupb");
auto swupb=_swupb.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.swupbc");
auto swupbc=_swupbc.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.swdnb");
auto swdnb=_swdnb.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.swdnbc");
auto swdnbc=_swdnbc.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwupb");
auto lwupb=_lwupb.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwupbc");
auto lwupbc=_lwupbc.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwdnb");
auto lwdnb=_lwdnb.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwdnbc");
auto lwdnbc=_lwdnbc.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.swcf");
auto swcf=_swcf.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.lwcf");
auto lwcf=_lwcf.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.olr");
auto olr=_olr.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.cemiss");
auto cemiss=_cemiss.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.taucldc");
auto taucldc=_taucldc.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.taucldi");
auto taucldi=_taucldi.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.coszr");
auto coszr=_coszr.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.gsw");
auto gsw=_gsw.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.glw");
auto glw=_glw.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.xlat");
auto xlat=_xlat.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.xlong");
auto xlong=_xlong.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.albedo");
auto albedo=_albedo.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.t_phy");
auto t_phy=_t_phy.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.tsk");
auto tsk=_tsk.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.emiss");
auto emiss=_emiss.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.qv3d");
auto qv3d=_qv3d.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.qc3d");
auto qc3d=_qc3d.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.qr3d");
auto qr3d=_qr3d.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.qi3d");
auto qi3d=_qi3d.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.qs3d");
auto qs3d=_qs3d.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.qg3d");
auto qg3d=_qg3d.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.f_ice_phy");
auto f_ice_phy=_f_ice_phy.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.f_rain_phy");
auto f_rain_phy=_f_rain_phy.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.p_phy");
auto p_phy=_p_phy.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.p8w");
auto p8w=_p8w.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.z");
auto z=_z.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.pi_phy");
auto pi_phy=_pi_phy.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.rho_phy");
auto rho_phy=_rho_phy.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.dz8w");
auto dz8w=_dz8w.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.cldfra");
auto cldfra=_cldfra.rebound(ims,ime,kms,kme,jms,jme);
CAM_TRACE("camrad.xland");
auto xland=_xland.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.xice");
auto xice=_xice.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.snow");
auto snow=_snow.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.ozmixm");
auto ozmixm=_ozmixm.rebound(ims,ime,1,levsiz,jms,jme,1,num_months);
CAM_TRACE("camrad.pin0");
auto pin0=_pin0.rebound(1,levsiz);
CAM_TRACE("camrad.m_psp");
auto m_psp=_m_psp.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.m_psn");
auto m_psn=_m_psn.rebound(ims,ime,jms,jme);
CAM_TRACE("camrad.aerosolcp");
auto aerosolcp=_aerosolcp.rebound(ims,ime,1,paerlev,jms,jme,1,naer_c);
CAM_TRACE("camrad.aerosolcn");
auto aerosolcn=_aerosolcn.rebound(ims,ime,1,paerlev,jms,jme,1,naer_c);
CAM_TRACE("camrad.m_hybi0");
auto m_hybi0=_m_hybi0.rebound(1,paerlev);
CAM_TRACE("camrad.abstot_3d");
auto abstot_3d=_abstot_3d.rebound(ims,ime,kms,kme,1,cam_abs_dim2,jms,jme);
CAM_TRACE("camrad.absnxt_3d");
auto absnxt_3d=_absnxt_3d.rebound(ims,ime,kms,kme,1,cam_abs_dim1,jms,jme);
CAM_TRACE("camrad.emstot_3d");
auto emstot_3d=_emstot_3d.rebound(ims,ime,kms,kme,jms,jme);
int lchnk;
int ncol;
int pcols;
int pver;
int pverp;
int pverr;
int pverrp;
int pcnst;
int pnats;
int ppcnst;
int i;
int j;
int k;
int ii;
int kk;
int kk1;
int m;
int n;
int begchunk;
int endchunk;
int nyrm;
int nyrp;
double doymodel;
double doydatam;
double doydatap;
double deltat;
double fact1;
double fact2;
float xt24;
float tloctm;
float hrang;
float xxlat;
float oldxt24;
auto coszrs=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto landfrac=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto landm=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto snowh=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto icefrac=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto lwups=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto asdir=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto asdif=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto aldir=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto aldif=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto ps=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto cld=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto pmid=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto lnpmid=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto pdel=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto zm=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto t=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto pint=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto lnpint=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto q=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1),1,n_cldadv);
auto cicewp=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto cliqwp=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto tauxcl=arena.array<double>(1,cam_add(cam_sub(ite,its),1),0,cam_add(cam_sub(kte,kts),1));
// MPAS module_ra_cam.F:364-365,678-679: observable automatic PRE storage.
if(oracle_locals && oracle_locals[648]) {
 auto entry=oracle_array<double>(648);
 const int column=its-native_chunk_its+entry.low[0];
 for(int k=tauxcl.low[1];k<tauxcl.low[1]+tauxcl.extent[1];++k)
  tauxcl(tauxcl.low[0],k)=entry(column,k);
}

auto tauxci=arena.array<double>(1,cam_add(cam_sub(ite,its),1),0,cam_add(cam_sub(kte,kts),1));
// MPAS module_ra_cam.F:364-365,678-679: observable automatic PRE storage.
if(oracle_locals && oracle_locals[630]) {
 auto entry=oracle_array<double>(630);
 const int column=its-native_chunk_its+entry.low[0];
 for(int k=tauxci.low[1];k<tauxci.low[1]+tauxci.extent[1];++k)
  tauxci(tauxci.low[0],k)=entry(column,k);
}

auto emis=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto rel=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto rei=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto pmxrgn=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto nmxrgn=arena.array<int>(1,cam_add(cam_sub(ite,its),1));
auto fsns=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto fsnt=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto flns=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto flnt=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto fsup=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto fsupc=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto fsdn=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto fsdnc=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto flup=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto flupc=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto fldn=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto fldnc=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),2));
auto swcftoa=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto lwcftoa=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto olrtoa=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto sols=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto soll=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto solsd=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto solld=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto qrs=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto fsds=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto qrl=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,cam_add(cam_sub(kte,kts),1));
auto flwds=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto ozmixmj=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,levsiz,1,num_months);
auto ozmix=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,levsiz);
auto pin=arena.array<double>(1,levsiz);
auto m_psjp=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto m_psjn=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto aerosoljp=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,paerlev,1,naer_c);
auto aerosoljn=arena.array<double>(1,cam_add(cam_sub(ite,its),1),1,paerlev,1,naer_c);
auto m_hybi=arena.array<double>(1,paerlev);
auto clat=arena.array<double>(1,cam_add(cam_sub(ite,its),1));
auto abstot=arena.array<double>(its,ite,kts,cam_add(kte,1),kts,cam_add(kte,1));
auto absnxt=arena.array<double>(its,ite,kts,kte,1,4);
auto emstot=arena.array<double>(its,ite,kts,cam_add(kte,1));
// MPAS module_ra_cam.F:411
lchnk=1;
// MPAS module_ra_cam.F:412
begchunk=ims;
// MPAS module_ra_cam.F:413
endchunk=ime;
// MPAS module_ra_cam.F:414
ncol=cam_add(cam_sub(ite,its),1);
// MPAS module_ra_cam.F:415
pcols=cam_add(cam_sub(ite,its),1);
// MPAS module_ra_cam.F:416
pver=cam_add(cam_sub(kte,kts),1);
// MPAS module_ra_cam.F:417
pverp=cam_add(pver,1);
// MPAS module_ra_cam.F:418
pverr=cam_add(cam_sub(kte,kts),1);
// MPAS module_ra_cam.F:419
pverrp=cam_add(pverr,1);
// MPAS module_ra_cam.F:421
ppcnst=n_cldadv;
// MPAS module_ra_cam.F:423
pnats=0;
// MPAS module_ra_cam.F:424
pcnst=cam_sub(ppcnst,pnats);
// MPAS module_ra_cam.F:431
if((naer_c!=naer_all)) {
// MPAS module_ra_cam.F:432
// MPAS diagnostic only
// MPAS module_ra_cam.F:433
error=433; return;
// MPAS module_ra_cam.F:434
}
// MPAS module_ra_cam.F:450
nyrm=cam_add(cam_sub(yr,yrdata(1)),1);
// MPAS module_ra_cam.F:451
nyrp=cam_add(nyrm,1);
// MPAS module_ra_cam.F:452
doymodel=cam_add(cam_mul(yr,365.f),julian);
// MPAS module_ra_cam.F:453
doydatam=cam_add(cam_mul(yrdata(nyrm),365.f),1.f);
// MPAS module_ra_cam.F:454
doydatap=cam_add(cam_mul(yrdata(nyrp),365.f),1.f);
// MPAS module_ra_cam.F:455
deltat=cam_sub(doydatap,doydatam);
// MPAS module_ra_cam.F:456
fact1=cam_div((cam_sub(doydatap,doymodel)),deltat);
// MPAS module_ra_cam.F:457
fact2=cam_div((cam_sub(doymodel,doydatam)),deltat);
// MPAS module_ra_cam.F:458
co2vmr=cam_mul((cam_add(cam_mul(co2(nyrm),fact1),cam_mul(co2(nyrp),fact2))),1.e-06f);
// MPAS module_ra_cam.F:460
co2mmr=cam_div(cam_mul(co2vmr,mwco2),mwdry);
// MPAS module_ra_cam.F:466
for(k=1;k<=levsiz;k+=1) {
// MPAS module_ra_cam.F:467
pin(k)=pin0(k);
// MPAS module_ra_cam.F:468
}
// MPAS module_ra_cam.F:470
for(k=1;k<=paerlev;k+=1) {
// MPAS module_ra_cam.F:471
m_hybi(k)=m_hybi0(k);
// MPAS module_ra_cam.F:472
}
// MPAS module_ra_cam.F:476
if((((abstot_3d(cam_cache_its,kts,kts,jts)==0.0f)&&(!doabsems))&&dolw)) {
// MPAS module_ra_cam.F:477
// MPAS diagnostic only
// MPAS module_ra_cam.F:478
// MPAS logging only
// MPAS module_ra_cam.F:479
doabsems=true;
// MPAS module_ra_cam.F:480
}
// MPAS module_ra_cam.F:488
for(j=jts;j<=jte;j+=1) {
// MPAS module_ra_cam.F:496
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:497
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:502
xt24=cam_mod(cam_add(xtime,cam_mul(radt,0.5f)),1440.f);
// MPAS module_ra_cam.F:503
tloctm=cam_add(cam_add(gmt,cam_div(xt24,60.f)),cam_div(xlong(i,j),15.f));
// MPAS module_ra_cam.F:504
hrang=cam_mul(cam_mul(15.f,(cam_sub(tloctm,12.f))),degrad);
// MPAS module_ra_cam.F:505
xxlat=cam_mul(xlat(i,j),degrad);
// MPAS module_ra_cam.F:506
clat(ii)=xxlat;
// MPAS module_ra_cam.F:507
coszrs(ii)=cam_add(cam_mul(cam_sin(xxlat),cam_sin(declin)),cam_mul(cam_mul(cam_cos(xxlat),cam_cos(declin)),cam_cos(hrang)));
// MPAS module_ra_cam.F:508
}
// MPAS module_ra_cam.F:512
for(k=kts;k<=kte;k+=1) {
// MPAS module_ra_cam.F:513
kk=cam_add(cam_sub(kte,k),kts);
// MPAS module_ra_cam.F:514
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:515
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:517
q(ii,kk,1)=cam_max(1.e-10f,cam_div(qv3d(i,k,j),(cam_add(1.f,qv3d(i,k,j)))));
// MPAS module_ra_cam.F:518
if(((f_qi&&f_qc)&&f_qs)) {
// MPAS module_ra_cam.F:519
q(ii,kk,ixcldliq)=cam_max(0.f,cam_div(qc3d(i,k,j),(cam_add(1.f,qv3d(i,k,j)))));
// MPAS module_ra_cam.F:520
q(ii,kk,ixcldice)=cam_max(0.f,cam_div((cam_add(qi3d(i,k,j),qs3d(i,k,j))),(cam_add(1.f,qv3d(i,k,j)))));
// MPAS module_ra_cam.F:521
} else if((f_qc&&f_qr)) {
// MPAS module_ra_cam.F:523
q(ii,kk,ixcldliq)=0.f;
// MPAS module_ra_cam.F:524
q(ii,kk,ixcldice)=0.f;
// MPAS module_ra_cam.F:525
if((t_phy(i,k,j)>273.15f)) {q(ii,kk,ixcldliq)=cam_max(0.f,cam_div(qc3d(i,k,j),(cam_add(1.f,qv3d(i,k,j)))));}
// MPAS module_ra_cam.F:526
if((t_phy(i,k,j)<=273.15f)) {q(ii,kk,ixcldice)=cam_max(0.f,cam_div(qc3d(i,k,j),(cam_add(1.f,qv3d(i,k,j)))));}
// MPAS module_ra_cam.F:527
} else if((f_qc&&f_qs)) {
// MPAS module_ra_cam.F:529
q(ii,kk,ixcldice)=cam_max(0.f,cam_mul(cam_div(qc3d(i,k,j),(cam_add(1.f,qv3d(i,k,j)))),f_ice_phy(i,k,j)));
// MPAS module_ra_cam.F:530
q(ii,kk,ixcldliq)=cam_max(0.f,cam_mul(cam_mul(cam_div(qc3d(i,k,j),(cam_add(1.f,qv3d(i,k,j)))),(cam_sub(1.f,f_ice_phy(i,k,j)))),(cam_sub(1.f,f_rain_phy(i,k,j)))));
// MPAS module_ra_cam.F:531
} else {
// MPAS module_ra_cam.F:532
q(ii,kk,ixcldliq)=0.f;
// MPAS module_ra_cam.F:533
q(ii,kk,ixcldice)=0.f;
// MPAS module_ra_cam.F:534
}
// MPAS module_ra_cam.F:535
cld(ii,kk)=cldfra(i,k,j);
// MPAS module_ra_cam.F:536
}
// MPAS module_ra_cam.F:537
}
// MPAS module_ra_cam.F:539
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:540
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:541
landfrac(ii)=cam_sub(2.f,xland(i,j));
// MPAS module_ra_cam.F:542
landm(ii)=landfrac(ii);
// MPAS module_ra_cam.F:543
snowh(ii)=cam_mul(0.001f,snow(i,j));
// MPAS module_ra_cam.F:544
icefrac(ii)=xice(i,j);
// MPAS module_ra_cam.F:545
}
// MPAS module_ra_cam.F:549
for(m=1;m<=num_months;m+=1) {
// MPAS module_ra_cam.F:550
for(k=1;k<=levsiz;k+=1) {
// MPAS module_ra_cam.F:551
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:552
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:553
ozmixmj(ii,k,m)=ozmixm(i,k,j,m);
// MPAS module_ra_cam.F:554
}
// MPAS module_ra_cam.F:555
}
// MPAS module_ra_cam.F:556
}
// MPAS module_ra_cam.F:568
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:569
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:570
m_psjp(ii)=m_psp(i,j);
// MPAS module_ra_cam.F:571
m_psjn(ii)=m_psn(i,j);
// MPAS module_ra_cam.F:572
}
// MPAS module_ra_cam.F:574
for(n=1;n<=naer_c;n+=1) {
// MPAS module_ra_cam.F:575
for(k=1;k<=paerlev;k+=1) {
// MPAS module_ra_cam.F:576
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:577
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:578
aerosoljp(ii,k,n)=aerosolcp(i,k,j,n);
// MPAS module_ra_cam.F:579
aerosoljn(ii,k,n)=aerosolcn(i,k,j,n);
// MPAS module_ra_cam.F:580
}
// MPAS module_ra_cam.F:581
}
// MPAS module_ra_cam.F:582
}
// MPAS module_ra_cam.F:587
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:588
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:589
lwups(ii)=cam_mul(cam_mul(stebol,emiss(i,j)),cam_pow(tsk(i,j),4));
// MPAS module_ra_cam.F:590
}
// MPAS module_ra_cam.F:592
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS module_ra_cam.F:593
kk=cam_add(cam_add(cam_sub(kte,k),kts),1);
// MPAS module_ra_cam.F:594
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:595
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:596
pint(ii,kk)=p8w(i,k,j);
// MPAS module_ra_cam.F:597
if((k==kts)) {ps(ii)=pint(ii,kk);}
// MPAS module_ra_cam.F:598
lnpint(ii,kk)=cam_log(pint(ii,kk));
// MPAS module_ra_cam.F:599
}
// MPAS module_ra_cam.F:600
}
// MPAS module_ra_cam.F:602
if(((!doabsems)&&dolw)) {
// MPAS module_ra_cam.F:604
for(kk=1;kk<=cam_abs_dim2;kk+=1) {
// MPAS module_ra_cam.F:605
for(kk1=kts;kk1<=cam_add(kte,1);kk1+=1) {
// MPAS module_ra_cam.F:606
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:607
abstot(i,kk1,kk)=abstot_3d(i,kk1,kk,j);
// MPAS module_ra_cam.F:608
}
// MPAS module_ra_cam.F:609
}
// MPAS module_ra_cam.F:610
}
// MPAS module_ra_cam.F:612
for(kk=1;kk<=cam_abs_dim1;kk+=1) {
// MPAS module_ra_cam.F:613
for(kk1=kts;kk1<=kte;kk1+=1) {
// MPAS module_ra_cam.F:614
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:615
absnxt(i,kk1,kk)=absnxt_3d(i,kk1,kk,j);
// MPAS module_ra_cam.F:616
}
// MPAS module_ra_cam.F:617
}
// MPAS module_ra_cam.F:618
}
// MPAS module_ra_cam.F:619
for(kk=kts;kk<=cam_add(kte,1);kk+=1) {
// MPAS module_ra_cam.F:620
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:621
emstot(i,kk)=emstot_3d(i,kk,j);
// MPAS module_ra_cam.F:622
}
// MPAS module_ra_cam.F:623
}
// MPAS module_ra_cam.F:624
}
// MPAS module_ra_cam.F:626
for(k=kts;k<=kte;k+=1) {
// MPAS module_ra_cam.F:627
kk=cam_add(cam_sub(kte,k),kts);
// MPAS module_ra_cam.F:628
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:629
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:630
pmid(ii,kk)=p_phy(i,k,j);
// MPAS module_ra_cam.F:631
lnpmid(ii,kk)=cam_log(pmid(ii,kk));
// MPAS module_ra_cam.F:632
lnpint(ii,kk)=cam_log(pint(ii,kk));
// MPAS module_ra_cam.F:633
pdel(ii,kk)=cam_sub(pint(ii,cam_add(kk,1)),pint(ii,kk));
// MPAS module_ra_cam.F:634
t(ii,kk)=t_phy(i,k,j);
// MPAS module_ra_cam.F:635
zm(ii,kk)=z(i,k,j);
// MPAS module_ra_cam.F:636
}
// MPAS module_ra_cam.F:637
}
// MPAS module_ra_cam.F:642
param_cldoptics_calc(ncol,pcols,pver,pverp,pverr,pverrp,ppcnst,q,cld,landfrac,landm,icefrac,pdel,t,ps,pmid,pint,cicewp,cliqwp,emis,rel,rei,pmxrgn,nmxrgn,snowh); if(error) return;
// MPAS module_ra_cam.F:645
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:646
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:649
asdir(ii)=albedo(i,j);
// MPAS module_ra_cam.F:650
asdif(ii)=albedo(i,j);
// MPAS module_ra_cam.F:651
aldir(ii)=albedo(i,j);
// MPAS module_ra_cam.F:652
aldif(ii)=albedo(i,j);
// MPAS module_ra_cam.F:653
}
// MPAS module_ra_cam.F:658
radctl(j,lchnk,ncol,pcols,pver,pverp,pverr,pverrp,ppcnst,pcnst,lwups,emis,pmid,pint,lnpmid,lnpint,pdel,t,q,cld,cicewp,cliqwp,tauxcl,tauxci,coszrs,clat,asdir,asdif,aldir,aldif,solcon,gmt,julday,julian,dt,xtime,pin,ozmixmj,ozmix,levsiz,num_months,m_psjp,m_psjn,aerosoljp,aerosoljn,m_hybi,paerlev,naer_c,pmxrgn,nmxrgn,dolw,dosw,doabsems,abstot,absnxt,emstot,fsup,fsupc,fsdn,fsdnc,flup,flupc,fldn,fldnc,swcftoa,lwcftoa,olrtoa,fsns,fsnt,flns,flnt,qrs,qrl,flwds,rel,rei,sols,soll,solsd,solld,landfrac,zm,fsds); if(error) return;
// MPAS module_ra_cam.F:671
for(k=kts;k<=kte;k+=1) {
// MPAS module_ra_cam.F:672
kk=cam_add(cam_sub(kte,k),kts);
// MPAS module_ra_cam.F:673
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:674
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:675
if(dolw) {rthratenlw(i,k,j)=cam_div(cam_mul(1.e4f,qrl(ii,kk)),(cam_mul(cpair,pi_phy(i,k,j))));}
// MPAS module_ra_cam.F:676
if(dosw) {rthratensw(i,k,j)=cam_div(cam_mul(1.e4f,qrs(ii,kk)),(cam_mul(cpair,pi_phy(i,k,j))));}
// MPAS module_ra_cam.F:677
cemiss(i,k,j)=emis(ii,kk);
// MPAS module_ra_cam.F:678
taucldc(i,k,j)=tauxcl(ii,kk);
// MPAS module_ra_cam.F:679
taucldi(i,k,j)=tauxci(ii,kk);
// MPAS module_ra_cam.F:680
}
// MPAS module_ra_cam.F:681
}
// MPAS module_ra_cam.F:683
if((doabsems&&dolw)) {
// MPAS module_ra_cam.F:685
for(kk=1;kk<=cam_abs_dim2;kk+=1) {
// MPAS module_ra_cam.F:686
for(kk1=kts;kk1<=cam_add(kte,1);kk1+=1) {
// MPAS module_ra_cam.F:687
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:688
abstot_3d(i,kk1,kk,j)=abstot(i,kk1,kk);
// MPAS module_ra_cam.F:689
}
// MPAS module_ra_cam.F:690
}
// MPAS module_ra_cam.F:691
}
// MPAS module_ra_cam.F:693
for(kk=1;kk<=cam_abs_dim1;kk+=1) {
// MPAS module_ra_cam.F:694
for(kk1=kts;kk1<=kte;kk1+=1) {
// MPAS module_ra_cam.F:695
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:696
absnxt_3d(i,kk1,kk,j)=absnxt(i,kk1,kk);
// MPAS module_ra_cam.F:697
}
// MPAS module_ra_cam.F:698
}
// MPAS module_ra_cam.F:699
}
// MPAS module_ra_cam.F:700
for(kk=kts;kk<=cam_add(kte,1);kk+=1) {
// MPAS module_ra_cam.F:701
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:702
emstot_3d(i,kk,j)=emstot(i,kk);
// MPAS module_ra_cam.F:703
}
// MPAS module_ra_cam.F:704
}
// MPAS module_ra_cam.F:705
}
// MPAS module_ra_cam.F:707
if(cam_present(swupt)) {
// MPAS module_ra_cam.F:708
if(dosw) {
// MPAS module_ra_cam.F:710
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS module_ra_cam.F:711
kk=cam_add(cam_sub(cam_add(kte,1),k),kts);
// MPAS module_ra_cam.F:712
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:713
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:718
if((k==cam_add(kte,1))) {
// MPAS module_ra_cam.F:719
swupt(i,j)=fsup(ii,kk);
// MPAS module_ra_cam.F:720
swuptc(i,j)=fsupc(ii,kk);
// MPAS module_ra_cam.F:721
swdnt(i,j)=fsdn(ii,kk);
// MPAS module_ra_cam.F:722
swdntc(i,j)=fsdnc(ii,kk);
// MPAS module_ra_cam.F:723
}
// MPAS module_ra_cam.F:724
if((k==kts)) {
// MPAS module_ra_cam.F:725
swupb(i,j)=fsup(ii,kk);
// MPAS module_ra_cam.F:726
swupbc(i,j)=fsupc(ii,kk);
// MPAS module_ra_cam.F:727
swdnb(i,j)=fsdn(ii,kk);
// MPAS module_ra_cam.F:728
swdnbc(i,j)=fsdnc(ii,kk);
// MPAS module_ra_cam.F:729
}
// MPAS module_ra_cam.F:734
}
// MPAS module_ra_cam.F:735
}
// MPAS module_ra_cam.F:736
}
// MPAS module_ra_cam.F:737
if(dolw) {
// MPAS module_ra_cam.F:739
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS module_ra_cam.F:740
kk=cam_add(cam_sub(cam_add(kte,1),k),kts);
// MPAS module_ra_cam.F:741
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:742
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:747
if((k==cam_add(kte,1))) {
// MPAS module_ra_cam.F:748
lwupt(i,j)=flup(ii,kk);
// MPAS module_ra_cam.F:749
lwuptc(i,j)=flupc(ii,kk);
// MPAS module_ra_cam.F:750
lwdnt(i,j)=fldn(ii,kk);
// MPAS module_ra_cam.F:751
lwdntc(i,j)=fldnc(ii,kk);
// MPAS module_ra_cam.F:752
}
// MPAS module_ra_cam.F:753
if((k==kts)) {
// MPAS module_ra_cam.F:754
lwupb(i,j)=flup(ii,kk);
// MPAS module_ra_cam.F:755
lwupbc(i,j)=flupc(ii,kk);
// MPAS module_ra_cam.F:756
lwdnb(i,j)=fldn(ii,kk);
// MPAS module_ra_cam.F:757
lwdnbc(i,j)=fldnc(ii,kk);
// MPAS module_ra_cam.F:758
}
// MPAS module_ra_cam.F:763
}
// MPAS module_ra_cam.F:764
}
// MPAS module_ra_cam.F:765
}
// MPAS module_ra_cam.F:766
}
// MPAS module_ra_cam.F:768
for(i=its;i<=ite;i+=1) {
// MPAS module_ra_cam.F:769
ii=cam_add(cam_sub(i,its),1);
// MPAS module_ra_cam.F:771
if(dolw) {
// MPAS module_ra_cam.F:772
glw(i,j)=flwds(ii);
// MPAS module_ra_cam.F:773
lwcf(i,j)=lwcftoa(ii);
// MPAS module_ra_cam.F:774
olr(i,j)=olrtoa(ii);
// MPAS module_ra_cam.F:775
}
// MPAS module_ra_cam.F:776
if(dosw) {
// MPAS module_ra_cam.F:777
gsw(i,j)=fsns(ii);
// MPAS module_ra_cam.F:778
swcf(i,j)=swcftoa(ii);
// MPAS module_ra_cam.F:779
coszr(i,j)=coszrs(ii);
// MPAS module_ra_cam.F:780
}
// MPAS module_ra_cam.F:781
}
// MPAS module_ra_cam.F:783
}
}
// MPAS module_ra_cam.F:917-985
CAM_ROUTINE void oznint(const int& julday,const float& julian,const float& dt,const float& gmt,const float& xtime,const CamArray<double>& _ozmixmj,const CamArray<double>& _ozmix,const int& levsiz,const int& num_months,const int& pcols) {
CamFrame _frame(arena);
CAM_TRACE("oznint.ozmixmj");
auto ozmixmj=_ozmixmj.rebound(1,pcols,1,levsiz,1,num_months);
CAM_TRACE("oznint.ozmix");
auto ozmix=_ozmix.rebound(1,pcols,1,levsiz);
double intjulian;
int np1;
int np;
int nm;
int m;
int k;
int i;
int ijul;
auto date_oz=arena.array<int>(1,12);
{ const int _init[]={16,45,75,105,136,166,197,228,258,289,319,350}; date_oz.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam.F:917-985: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[504]) date_oz=oracle_array<int>(504);
double cdayozp;
double cdayozm;
double fact1;
double fact2;
bool finddate;
// MPAS module_ra_cam.F:945
intjulian=cam_add(julian,1.0);
// MPAS module_ra_cam.F:947
ijul=((int)(intjulian));
// MPAS module_ra_cam.F:950
intjulian=cam_sub(intjulian,cam_float(ijul));
// MPAS module_ra_cam.F:951
ijul=cam_mod(ijul,365);
// MPAS module_ra_cam.F:952
if((ijul==0)) {ijul=365;}
// MPAS module_ra_cam.F:953
intjulian=cam_add(intjulian,ijul);
// MPAS module_ra_cam.F:954
np1=1;
// MPAS module_ra_cam.F:955
finddate=false;
// MPAS module_ra_cam.F:957
for(m=1;m<=12;m+=1) {
// MPAS module_ra_cam.F:958
if(((date_oz(m)>intjulian)&&(!finddate))) {
// MPAS module_ra_cam.F:959
np1=m;
// MPAS module_ra_cam.F:960
finddate=true;
// MPAS module_ra_cam.F:961
}
// MPAS module_ra_cam.F:962
}
// MPAS module_ra_cam.F:963
cdayozp=date_oz(np1);
// MPAS module_ra_cam.F:964
if((np1>1)) {
// MPAS module_ra_cam.F:965
cdayozm=date_oz(cam_sub(np1,1));
// MPAS module_ra_cam.F:966
np=np1;
// MPAS module_ra_cam.F:967
nm=cam_sub(np,1);
// MPAS module_ra_cam.F:968
} else {
// MPAS module_ra_cam.F:969
cdayozm=date_oz(12);
// MPAS module_ra_cam.F:970
np=np1;
// MPAS module_ra_cam.F:971
nm=12;
// MPAS module_ra_cam.F:972
}
// MPAS module_ra_cam.F:973
getfactors(ozncyc,np1,cdayozm,cdayozp,intjulian,fact1,fact2); if(error) return;
// MPAS module_ra_cam.F:979
for(k=1;k<=levsiz;k+=1) {
// MPAS module_ra_cam.F:980
for(i=1;i<=pcols;i+=1) {
// MPAS module_ra_cam.F:981
ozmix(i,k)=cam_add(cam_mul(ozmixmj(i,k,nm),fact1),cam_mul(ozmixmj(i,k,np),fact2));
// MPAS module_ra_cam.F:982
}
// MPAS module_ra_cam.F:983
}
}
// MPAS module_ra_cam.F:988-1169
CAM_ROUTINE void get_aerosol(const int& c,const int& julday,const float& julian,const float& dt,const float& gmt,const float& xtime,const CamArray<double>& _m_psp,const CamArray<double>& _m_psn,const CamArray<double>& _aerosoljp,const CamArray<double>& _aerosoljn,const CamArray<double>& _m_hybi,const int& paerlev,const int& naer_c,const CamArray<double>& _pint,const int& pcols,const int& pver,const int& pverp,const int& pverr,const int& pverrp,const CamArray<double>& _aerosolt,const CamArray<double>& _scale) {
CamFrame _frame(arena);
CAM_TRACE("get_aerosol.m_psp");
auto m_psp=_m_psp.rebound(1,pcols);
CAM_TRACE("get_aerosol.m_psn");
auto m_psn=_m_psn.rebound(1,pcols);
CAM_TRACE("get_aerosol.aerosoljp");
auto aerosoljp=_aerosoljp.rebound(1,pcols,1,paerlev,1,naer_c);
CAM_TRACE("get_aerosol.aerosoljn");
auto aerosoljn=_aerosoljn.rebound(1,pcols,1,paerlev,1,naer_c);
CAM_TRACE("get_aerosol.m_hybi");
auto m_hybi=_m_hybi.rebound(1,paerlev);
CAM_TRACE("get_aerosol.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("get_aerosol.aerosolt");
auto aerosolt=_aerosolt.rebound(1,pcols,1,pver,1,naer_all);
CAM_TRACE("get_aerosol.scale");
auto scale=_scale.rebound(1,naer_all);
double caldayloc;
double fact1;
double fact2;
// MPAS module_ra_cam.F:988-1169: native local SAVE/DATA storage.
int _oracle_default_nm=1;
int& nm=(oracle_locals && oracle_locals[594]) ? *reinterpret_cast<int*>(oracle_locals[594]) : _oracle_default_nm;
// MPAS module_ra_cam.F:988-1169: native local SAVE/DATA storage.
int _oracle_default_np=2;
int& np=(oracle_locals && oracle_locals[612]) ? *reinterpret_cast<int*>(oracle_locals[612]) : _oracle_default_np;
// MPAS module_ra_cam.F:988-1169: native local SAVE/DATA storage.
int _oracle_default_mo_nxt=bigint;
int& mo_nxt=(oracle_locals && oracle_locals[576]) ? *reinterpret_cast<int*>(oracle_locals[576]) : _oracle_default_mo_nxt;
int mo_prv;
// MPAS module_ra_cam.F:988-1169: native local SAVE/DATA storage.
double _oracle_default_cdaym=inf;
double& cdaym=(oracle_locals && oracle_locals[522]) ? *reinterpret_cast<double*>(oracle_locals[522]) : _oracle_default_cdaym;
// MPAS module_ra_cam.F:988-1169: native local SAVE/DATA storage.
double _oracle_default_cdayp=inf;
double& cdayp=(oracle_locals && oracle_locals[540]) ? *reinterpret_cast<double*>(oracle_locals[540]) : _oracle_default_cdayp;
auto mid=arena.array<double>(1,12);
{ const double _init[]={16.5f,46.0f,75.5f,106.0f,136.5f,167.0f,197.5f,228.5f,259.0f,289.5f,320.0f,350.5f}; mid.init(_init,sizeof(_init)/sizeof(_init[0])); }
// MPAS module_ra_cam.F:988-1169: native local SAVE/DATA storage.
if(oracle_locals && oracle_locals[558]) mid=oracle_array<double>(558);
int i;
int k;
int j;
int m;
auto lats=arena.array<int>(1,pcols);
auto lons=arena.array<int>(1,pcols);
int ncol;
int ijul;
double intjulian;
auto speciesmin=arena.array<double>(1,naer);
auto aerosolm=arena.array<double>(1,pcols,1,pver,1,naer);
auto aerosolp=arena.array<double>(1,pcols,1,pver,1,naer);
// MPAS module_ra_cam.F:1068
intjulian=cam_add(julian,1.0);
// MPAS module_ra_cam.F:1070
ijul=((int)(intjulian));
// MPAS module_ra_cam.F:1073
intjulian=cam_sub(intjulian,cam_float(ijul));
// MPAS module_ra_cam.F:1074
ijul=cam_mod(ijul,365);
// MPAS module_ra_cam.F:1075
if((ijul==0)) {ijul=365;}
// MPAS module_ra_cam.F:1076
caldayloc=cam_add(intjulian,ijul);
// MPAS module_ra_cam.F:1078
if((caldayloc<mid(1))) {
// MPAS module_ra_cam.F:1079
mo_prv=12;
// MPAS module_ra_cam.F:1080
mo_nxt=1;
// MPAS module_ra_cam.F:1081
} else if((caldayloc>=mid(12))) {
// MPAS module_ra_cam.F:1082
mo_prv=12;
// MPAS module_ra_cam.F:1083
mo_nxt=1;
// MPAS module_ra_cam.F:1084
} else {
// MPAS module_ra_cam.F:1085
for(i=2;i<=12;i+=1) {
// MPAS module_ra_cam.F:1086
if((caldayloc<mid(i))) {
// MPAS module_ra_cam.F:1087
mo_prv=cam_sub(i,1);
// MPAS module_ra_cam.F:1088
mo_nxt=i;
// MPAS module_ra_cam.F:1089
break;
// MPAS module_ra_cam.F:1090
}
// MPAS module_ra_cam.F:1091
}
// MPAS module_ra_cam.F:1092
}
// MPAS module_ra_cam.F:1096
cdaym=mid(mo_prv);
// MPAS module_ra_cam.F:1097
cdayp=mid(mo_nxt);
// MPAS module_ra_cam.F:1102
getfactors(true,mo_nxt,cdaym,cdayp,caldayloc,fact1,fact2); if(error) return;
// MPAS module_ra_cam.F:1110
ncol=pcols;
// MPAS module_ra_cam.F:1115
vert_interpolate(m_psp,aerosoljp,m_hybi,paerlev,naer_c,pint,nm,aerosolm,pcols,pver,pverp,ncol,c); if(error) return;
// MPAS module_ra_cam.F:1116
vert_interpolate(m_psn,aerosoljn,m_hybi,paerlev,naer_c,pint,np,aerosolp,pcols,pver,pverp,ncol,c); if(error) return;
// MPAS module_ra_cam.F:1121
for(m=1;m<=naer;m+=1) {
// MPAS module_ra_cam.F:1122
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:1123
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1124
aerosolt(i,k,m)=cam_add(cam_mul(aerosolm(i,k,m),fact1),cam_mul(aerosolp(i,k,m),fact2));
// MPAS module_ra_cam.F:1125
}
// MPAS module_ra_cam.F:1126
}
// MPAS module_ra_cam.F:1127
}
// MPAS module_ra_cam.F:1135
background(c,ncol,pint,pcols,pverr,pverrp,aerosolt.section(cam_section(1,pcols),cam_section(1,pver),idxbg)); if(error) return;
// MPAS module_ra_cam.F:1143
{ auto _lhs=aerosolt.section(cam_section(1,pcols),cam_section(1,pver),idxvolc); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.; }
// MPAS module_ra_cam.F:1150
{ auto _lhs=speciesmin.section(cam_section(1,naer)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam.F:1152
for(m=1;m<=naer;m+=1) {
// MPAS module_ra_cam.F:1153
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:1154
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1155
if((aerosolt(i,k,m)<speciesmin(m))) {
// MPAS module_ra_cam.F:1156
// MPAS diagnostic only
// MPAS module_ra_cam.F:1157
// MPAS diagnostic only
// MPAS module_ra_cam.F:1159
}
// MPAS module_ra_cam.F:1160
}
// MPAS module_ra_cam.F:1161
}
// MPAS module_ra_cam.F:1162
}
// MPAS module_ra_cam.F:1166
scale_aerosols(aerosolt,pcols,pver,ncol,c,scale); if(error) return;
// MPAS module_ra_cam.F:1168
return;
}
// MPAS module_ra_cam.F:1172-1310
CAM_ROUTINE void aerosol_indirect(const int& ncol,const int& lchnk,const int& pcols,const int& pver,const int& ppcnst,const CamArray<double>& _landfrac,const CamArray<double>& _pmid,const CamArray<double>& _t,const CamArray<double>& _qm1,const CamArray<double>& _cld,const CamArray<double>& _zm,const CamArray<double>& _rel) {
CamFrame _frame(arena);
const double cmarn=50.0f;
const double cland=100.0f;
const double hmarn=1000.0f;
const double hland=2000.0f;
CAM_TRACE("aerosol_indirect.landfrac");
auto landfrac=_landfrac.rebound(1,pcols);
CAM_TRACE("aerosol_indirect.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("aerosol_indirect.t");
auto t=_t.rebound(1,pcols,1,pver);
CAM_TRACE("aerosol_indirect.qm1");
auto qm1=_qm1.rebound(1,pcols,1,pver,1,ppcnst);
CAM_TRACE("aerosol_indirect.cld");
auto cld=_cld.rebound(1,pcols,1,pver);
CAM_TRACE("aerosol_indirect.zm");
auto zm=_zm.rebound(1,pcols,1,pver);
CAM_TRACE("aerosol_indirect.rel");
auto rel=_rel.rebound(1,pcols,1,pver);
auto locrhoair=arena.array<double>(1,pcols,1,pver);
auto lwcwat=arena.array<double>(1,pcols,1,pver);
auto sulfmix=arena.array<double>(1,pcols,1,pver);
auto so4mass=arena.array<double>(1,pcols,1,pver);
auto aso4=arena.array<double>(1,pcols,1,pver);
auto ntot=arena.array<double>(1,pcols,1,pver);
auto relmod=arena.array<double>(1,pcols,1,pver);
auto wrel=arena.array<double>(1,pcols,1,pver);
auto wlwc=arena.array<double>(1,pcols,1,pver);
auto cldfrq=arena.array<double>(1,pcols,1,pver);
double locpi;
double rdryair;
double rhowat;
double acoef;
double rekappa;
double recoef;
double reexp;
double ntotb;
double bgaer;
int i;
int k;
auto land=[&](int i) { return (cam_nint(landfrac(i))>0.5); };
// MPAS module_ra_cam.F:1234
if(indirect) {
// MPAS module_ra_cam.F:1239
{ auto _lhs=sulfmix.section(cam_section(1,ncol),cam_section(1,pver)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.; }
// MPAS module_ra_cam.F:1241
locpi=3.141592654f;
// MPAS module_ra_cam.F:1242
rdryair=287.04f;
// MPAS module_ra_cam.F:1243
rhowat=1000.0f;
// MPAS module_ra_cam.F:1244
acoef=1.2930e14f;
// MPAS module_ra_cam.F:1245
recoef=cam_div(3.0f,(cam_mul(cam_mul(4.0f,locpi),rhowat)));
// MPAS module_ra_cam.F:1246
reexp=cam_div(1.0f,3.0f);
// MPAS module_ra_cam.F:1249
for(k=pver;k>=1;k+=cam_neg(1)) {
// MPAS module_ra_cam.F:1250
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1251
locrhoair(i,k)=cam_div(pmid(i,k),(cam_mul(rdryair,t(i,k))));
// MPAS module_ra_cam.F:1252
lwcwat(i,k)=cam_mul((cam_div(qm1(i,k,ixcldliq),cam_max(0.01,cld(i,k)))),locrhoair(i,k));
// MPAS module_ra_cam.F:1255
so4mass(i,k)=cam_mul(cam_mul(sulfmix(i,k),locrhoair(i,k)),0.001f);
// MPAS module_ra_cam.F:1256
aso4(i,k)=cam_mul(so4mass(i,k),acoef);
// MPAS module_ra_cam.F:1258
if((aso4(i,k)<=280.0f)) {
// MPAS module_ra_cam.F:1259
aso4(i,k)=cam_max(36.0,aso4(i,k));
// MPAS module_ra_cam.F:1260
ntot(i,k)=cam_add(cam_add(cam_neg(cam_mul(1.15e-3f,cam_pow(aso4(i,k),2))),cam_mul(0.963f,aso4(i,k))),5.30f);
// MPAS module_ra_cam.F:1261
rekappa=0.80f;
// MPAS module_ra_cam.F:1262
} else {
// MPAS module_ra_cam.F:1263
aso4(i,k)=cam_min(1500.0,aso4(i,k));
// MPAS module_ra_cam.F:1264
ntot(i,k)=cam_sub(cam_add(cam_neg(cam_mul(2.10e-4f,cam_pow(aso4(i,k),2))),cam_mul(0.568f,aso4(i,k))),27.9f);
// MPAS module_ra_cam.F:1265
rekappa=0.67f;
// MPAS module_ra_cam.F:1266
}
// MPAS module_ra_cam.F:1267
if(land(i)) {
// MPAS module_ra_cam.F:1268
bgaer=cam_mul(cland,cam_exp(cam_neg((cam_div(zm(i,k),hland)))));
// MPAS module_ra_cam.F:1269
ntot(i,k)=cam_max(bgaer,ntot(i,k));
// MPAS module_ra_cam.F:1270
} else {
// MPAS module_ra_cam.F:1271
bgaer=cam_mul(cmarn,cam_exp(cam_neg((cam_div(zm(i,k),hmarn)))));
// MPAS module_ra_cam.F:1272
ntot(i,k)=cam_max(bgaer,ntot(i,k));
// MPAS module_ra_cam.F:1273
}
// MPAS module_ra_cam.F:1275
if((k==pver)) {
// MPAS module_ra_cam.F:1276
ntotb=ntot(i,k);
// MPAS module_ra_cam.F:1277
} else {
// MPAS module_ra_cam.F:1278
ntotb=ntot(i,cam_add(k,1));
// MPAS module_ra_cam.F:1279
}
// MPAS module_ra_cam.F:1281
relmod(i,k)=cam_mul((cam_pow((cam_div((cam_mul(recoef,lwcwat(i,k))),(cam_mul(rekappa,ntotb)))),reexp)),10000.0f);
// MPAS module_ra_cam.F:1282
relmod(i,k)=cam_max(4.0,relmod(i,k));
// MPAS module_ra_cam.F:1283
relmod(i,k)=cam_min(20.0,relmod(i,k));
// MPAS module_ra_cam.F:1284
if((cld(i,k)>=0.01f)) {
// MPAS module_ra_cam.F:1285
cldfrq(i,k)=1.0f;
// MPAS module_ra_cam.F:1286
} else {
// MPAS module_ra_cam.F:1287
cldfrq(i,k)=0.0f;
// MPAS module_ra_cam.F:1288
}
// MPAS module_ra_cam.F:1289
wrel(i,k)=cam_mul(relmod(i,k),cldfrq(i,k));
// MPAS module_ra_cam.F:1290
wlwc(i,k)=cam_mul(lwcwat(i,k),cldfrq(i,k));
// MPAS module_ra_cam.F:1291
}
// MPAS module_ra_cam.F:1292
}
// MPAS module_ra_cam.F:1299
} else {
// MPAS module_ra_cam.F:1300
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:1301
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1302
relmod(i,k)=rel(i,k);
// MPAS module_ra_cam.F:1303
}
// MPAS module_ra_cam.F:1304
}
// MPAS module_ra_cam.F:1305
}
// MPAS module_ra_cam.F:1309
return;
}
// MPAS module_ra_cam.F:1313-1391
CAM_ROUTINE void aer_trn(const CamArray<double>& _aer_mpp,const CamArray<double>& _aer_trn_ttl,const int& pcols,const int& plev,const int& plevp) {
CamFrame _frame(arena);
CAM_TRACE("aer_trn.aer_mpp");
auto aer_mpp=_aer_mpp.rebound(1,pcols,1,plevp);
CAM_TRACE("aer_trn.aer_trn_ttl");
auto aer_trn_ttl=_aer_trn_ttl.rebound(1,pcols,1,plevp,1,plevp,1,bnd_nbr_lw);
int bnd_idx;
int i;
int k1;
int k2;
double aer_pth_dlt;
double odap_aer_ttl;
// MPAS module_ra_cam.F:1352
if(strat_volcanic) {
// MPAS module_ra_cam.F:1353
for(bnd_idx=1;bnd_idx<=bnd_nbr_lw;bnd_idx+=1) {
// MPAS module_ra_cam.F:1354
for(i=1;i<=pcols;i+=1) {
// MPAS module_ra_cam.F:1355
aer_trn_ttl(i,1,1,bnd_idx)=1.0f;
// MPAS module_ra_cam.F:1356
}
// MPAS module_ra_cam.F:1357
for(k1=2;k1<=plevp;k1+=1) {
// MPAS module_ra_cam.F:1358
for(i=1;i<=pcols;i+=1) {
// MPAS module_ra_cam.F:1359
aer_trn_ttl(i,k1,k1,bnd_idx)=1.0f;
// MPAS module_ra_cam.F:1361
aer_pth_dlt=cam_abs(cam_sub(aer_mpp(i,k1),aer_mpp(i,1)));
// MPAS module_ra_cam.F:1362
odap_aer_ttl=cam_mul(abs_cff_mss_aer(bnd_idx),aer_pth_dlt);
// MPAS module_ra_cam.F:1364
aer_trn_ttl(i,1,k1,bnd_idx)=cam_exp(cam_neg(cam_mul(1.66f,odap_aer_ttl)));
// MPAS module_ra_cam.F:1365
}
// MPAS module_ra_cam.F:1366
}
// MPAS module_ra_cam.F:1368
for(k1=2;k1<=plev;k1+=1) {
// MPAS module_ra_cam.F:1369
for(k2=cam_add(k1,1);k2<=plevp;k2+=1) {
// MPAS module_ra_cam.F:1370
for(i=1;i<=pcols;i+=1) {
// MPAS module_ra_cam.F:1371
aer_trn_ttl(i,k1,k2,bnd_idx)=cam_div(aer_trn_ttl(i,1,k2,bnd_idx),aer_trn_ttl(i,1,k1,bnd_idx));
// MPAS module_ra_cam.F:1374
}
// MPAS module_ra_cam.F:1375
}
// MPAS module_ra_cam.F:1376
}
// MPAS module_ra_cam.F:1378
for(k1=2;k1<=plevp;k1+=1) {
// MPAS module_ra_cam.F:1379
for(k2=1;k2<=cam_sub(k1,1);k2+=1) {
// MPAS module_ra_cam.F:1380
for(i=1;i<=pcols;i+=1) {
// MPAS module_ra_cam.F:1381
aer_trn_ttl(i,k1,k2,bnd_idx)=aer_trn_ttl(i,k2,k1,bnd_idx);
// MPAS module_ra_cam.F:1382
}
// MPAS module_ra_cam.F:1383
}
// MPAS module_ra_cam.F:1384
}
// MPAS module_ra_cam.F:1385
}
// MPAS module_ra_cam.F:1386
} else {
// MPAS module_ra_cam.F:1387
{ auto _lhs=aer_trn_ttl; for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=1.0f; }
// MPAS module_ra_cam.F:1388
}
// MPAS module_ra_cam.F:1390
return;
}
// MPAS module_ra_cam.F:1393-1424
CAM_ROUTINE void aer_pth(const CamArray<double>& _aer_mass,const CamArray<double>& _aer_mpp,const int& ncol,const int& pcols,const int& plev,const int& plevp) {
CamFrame _frame(arena);
CAM_TRACE("aer_pth.aer_mass");
auto aer_mass=_aer_mass.rebound(1,pcols,1,plev);
CAM_TRACE("aer_pth.aer_mpp");
auto aer_mpp=_aer_mpp.rebound(1,pcols,1,plevp);
int i;
int k;
// MPAS module_ra_cam.F:1418
{ auto _lhs=aer_mpp.section(cam_section(1,ncol),1); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.; }
// MPAS module_ra_cam.F:1419
for(k=2;k<=plevp;k+=1) {
// MPAS module_ra_cam.F:1420
{ auto _lhs=aer_mpp.section(cam_section(1,ncol),k); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_add(aer_mpp.section(cam_section(1,ncol),cam_sub(k,1)).linear(_slice),aer_mass.section(cam_section(1,ncol),cam_sub(k,1)).linear(_slice)); }
// MPAS module_ra_cam.F:1421
}
// MPAS module_ra_cam.F:1423
return;
}
// MPAS module_ra_cam.F:1426-1909
CAM_ROUTINE void radctl(const int& j,const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const int& pverr,const int& pverrp,const int& ppcnst,const int& pcnst,const CamArray<double>& _lwups,const CamArray<double>& _emis,const CamArray<double>& _pmid,const CamArray<double>& _pint,const CamArray<double>& _pmln,const CamArray<double>& _piln,const CamArray<double>& _pdel,const CamArray<double>& _t,const CamArray<double>& _qm1,const CamArray<double>& _cld,const CamArray<double>& _cicewp,const CamArray<double>& _cliqwp,const CamArray<double>& _tauxcl,const CamArray<double>& _tauxci,const CamArray<double>& _coszrs,const CamArray<double>& _clat,const CamArray<double>& _asdir,const CamArray<double>& _asdif,const CamArray<double>& _aldir,const CamArray<double>& _aldif,const float& solcon,const float& gmt,const int& julday,const float& julian,const float& dt,const float& xtime,const CamArray<double>& _pin,const CamArray<double>& _ozmixmj,const CamArray<double>& _ozmix,const int& levsiz,const int& num_months,const CamArray<double>& _m_psp,const CamArray<double>& _m_psn,const CamArray<double>& _aerosoljp,const CamArray<double>& _aerosoljn,const CamArray<double>& _m_hybi,const int& paerlev,const int& naer_c,const CamArray<double>& _pmxrgn,const CamArray<int>& _nmxrgn,const bool& dolw,const bool& dosw,const bool& doabsems,const CamArray<double>& _abstot,const CamArray<double>& _absnxt,const CamArray<double>& _emstot,const CamArray<double>& _fsup,const CamArray<double>& _fsupc,const CamArray<double>& _fsdn,const CamArray<double>& _fsdnc,const CamArray<double>& _flup,const CamArray<double>& _flupc,const CamArray<double>& _fldn,const CamArray<double>& _fldnc,const CamArray<double>& _swcf,const CamArray<double>& _lwcf,const CamArray<double>& _flut,const CamArray<double>& _fsns,const CamArray<double>& _fsnt,const CamArray<double>& _flns,const CamArray<double>& _flnt,const CamArray<double>& _qrs,const CamArray<double>& _qrl,const CamArray<double>& _flwds,const CamArray<double>& _rel,const CamArray<double>& _rei,const CamArray<double>& _sols,const CamArray<double>& _soll,const CamArray<double>& _solsd,const CamArray<double>& _solld,const CamArray<double>& _landfrac,const CamArray<double>& _zm,const CamArray<double>& _fsds) {
CamFrame _frame(arena);
const int nspint=19;
const int naer_groups=7;
CAM_TRACE("radctl.lwups");
auto lwups=_lwups.rebound(1,pcols);
CAM_TRACE("radctl.emis");
auto emis=_emis.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.pmln");
auto pmln=_pmln.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.piln");
auto piln=_piln.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.pdel");
auto pdel=_pdel.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.t");
auto t=_t.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.qm1");
auto qm1=_qm1.rebound(1,pcols,1,pver,1,ppcnst);
CAM_TRACE("radctl.cld");
auto cld=_cld.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.cicewp");
auto cicewp=_cicewp.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.cliqwp");
auto cliqwp=_cliqwp.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.tauxcl");
auto tauxcl=_tauxcl.rebound(1,pcols,0,pver);
CAM_TRACE("radctl.tauxci");
auto tauxci=_tauxci.rebound(1,pcols,0,pver);
CAM_TRACE("radctl.coszrs");
auto coszrs=_coszrs.rebound(1,pcols);
CAM_TRACE("radctl.clat");
auto clat=_clat.rebound(1,pcols);
CAM_TRACE("radctl.asdir");
auto asdir=_asdir.rebound(1,pcols);
CAM_TRACE("radctl.asdif");
auto asdif=_asdif.rebound(1,pcols);
CAM_TRACE("radctl.aldir");
auto aldir=_aldir.rebound(1,pcols);
CAM_TRACE("radctl.aldif");
auto aldif=_aldif.rebound(1,pcols);
CAM_TRACE("radctl.pin");
auto pin=_pin.rebound(1,levsiz);
CAM_TRACE("radctl.ozmixmj");
auto ozmixmj=_ozmixmj.rebound(1,pcols,1,levsiz,1,num_months);
CAM_TRACE("radctl.ozmix");
auto ozmix=_ozmix.rebound(1,pcols,1,levsiz);
CAM_TRACE("radctl.m_psp");
auto m_psp=_m_psp.rebound(1,pcols);
CAM_TRACE("radctl.m_psn");
auto m_psn=_m_psn.rebound(1,pcols);
CAM_TRACE("radctl.aerosoljp");
auto aerosoljp=_aerosoljp.rebound(1,pcols,1,paerlev,1,naer_c);
CAM_TRACE("radctl.aerosoljn");
auto aerosoljn=_aerosoljn.rebound(1,pcols,1,paerlev,1,naer_c);
CAM_TRACE("radctl.m_hybi");
auto m_hybi=_m_hybi.rebound(1,paerlev);
CAM_TRACE("radctl.pmxrgn");
auto pmxrgn=_pmxrgn.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.nmxrgn");
auto nmxrgn=_nmxrgn.rebound(1,pcols);
CAM_TRACE("radctl.abstot");
auto abstot=_abstot.rebound(1,pcols,1,pverp,1,pverp);
CAM_TRACE("radctl.absnxt");
auto absnxt=_absnxt.rebound(1,pcols,1,pver,1,4);
CAM_TRACE("radctl.emstot");
auto emstot=_emstot.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.fsup");
auto fsup=_fsup.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.fsupc");
auto fsupc=_fsupc.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.fsdn");
auto fsdn=_fsdn.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.fsdnc");
auto fsdnc=_fsdnc.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.flup");
auto flup=_flup.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.flupc");
auto flupc=_flupc.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.fldn");
auto fldn=_fldn.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.fldnc");
auto fldnc=_fldnc.rebound(1,pcols,1,pverp);
CAM_TRACE("radctl.swcf");
auto swcf=_swcf.rebound(1,pcols);
CAM_TRACE("radctl.lwcf");
auto lwcf=_lwcf.rebound(1,pcols);
CAM_TRACE("radctl.flut");
auto flut=_flut.rebound(1,pcols);
CAM_TRACE("radctl.fsns");
auto fsns=_fsns.rebound(1,pcols);
CAM_TRACE("radctl.fsnt");
auto fsnt=_fsnt.rebound(1,pcols);
CAM_TRACE("radctl.flns");
auto flns=_flns.rebound(1,pcols);
CAM_TRACE("radctl.flnt");
auto flnt=_flnt.rebound(1,pcols);
CAM_TRACE("radctl.qrs");
auto qrs=_qrs.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.qrl");
auto qrl=_qrl.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.flwds");
auto flwds=_flwds.rebound(1,pcols);
CAM_TRACE("radctl.rel");
auto rel=_rel.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.rei");
auto rei=_rei.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.sols");
auto sols=_sols.rebound(1,pcols);
CAM_TRACE("radctl.soll");
auto soll=_soll.rebound(1,pcols);
CAM_TRACE("radctl.solsd");
auto solsd=_solsd.rebound(1,pcols);
CAM_TRACE("radctl.solld");
auto solld=_solld.rebound(1,pcols);
CAM_TRACE("radctl.landfrac");
auto landfrac=_landfrac.rebound(1,pcols);
CAM_TRACE("radctl.zm");
auto zm=_zm.rebound(1,pcols,1,pver);
CAM_TRACE("radctl.fsds");
auto fsds=_fsds.rebound(1,pcols);
auto pmxrgnrf=arena.array<double>(1,pcols,1,pverp);
auto nmxrgnrf=arena.array<int>(1,pcols);
int i;
int k;
int in2o;
int ich4;
int if11;
int if12;
auto solin=arena.array<double>(1,pcols);
auto fsntoa=arena.array<double>(1,pcols);
auto fsntoac=arena.array<double>(1,pcols);
auto fsnirt=arena.array<double>(1,pcols);
auto fsnrtc=arena.array<double>(1,pcols);
auto fsnirtsq=arena.array<double>(1,pcols);
auto fsntc=arena.array<double>(1,pcols);
auto fsnsc=arena.array<double>(1,pcols);
auto fsdsc=arena.array<double>(1,pcols);
auto flutc=arena.array<double>(1,pcols);
auto flntc=arena.array<double>(1,pcols);
auto flnsc=arena.array<double>(1,pcols);
auto ftem=arena.array<double>(1,pcols,1,pver);
auto pbr=arena.array<double>(1,pcols,1,pverr);
auto pnm=arena.array<double>(1,pcols,1,pverrp);
auto o3vmr=arena.array<double>(1,pcols,1,pverr);
auto o3mmr=arena.array<double>(1,pcols,1,pverr);
double eccf;
auto n2o=arena.array<double>(1,pcols,1,pver);
auto ch4=arena.array<double>(1,pcols,1,pver);
auto cfc11=arena.array<double>(1,pcols,1,pver);
auto cfc12=arena.array<double>(1,pcols,1,pver);
auto rh=arena.array<double>(1,pcols,1,pverr);
auto lwupcgs=arena.array<double>(1,pcols);
auto esat=arena.array<double>(1,pcols,1,pverr);
auto qsat=arena.array<double>(1,pcols,1,pverr);
auto frc_day=arena.array<double>(1,pcols);
auto aertau=arena.array<double>(1,pcols,1,nspint,1,naer_groups);
auto aerssa=arena.array<double>(1,pcols,1,nspint,1,naer_groups);
auto aerasm=arena.array<double>(1,pcols,1,nspint,1,naer_groups);
auto aerfwd=arena.array<double>(1,pcols,1,nspint,1,naer_groups);
auto aerosol=arena.array<double>(1,pcols,1,pver,1,naer_all);
auto scales=arena.array<double>(1,naer_all);
// MPAS module_ra_cam.F:1628
oznint(julday,julian,dt,gmt,xtime,ozmixmj,ozmix,levsiz,num_months,pcols); if(error) return;
// MPAS module_ra_cam.F:1630
radozn(lchnk,ncol,pcols,pver,pmid,pin,levsiz,ozmix,o3vmr); if(error) return;
// MPAS module_ra_cam.F:1639
radinp(lchnk,ncol,pcols,pver,pverp,pmid,pint,o3vmr,pbr,pnm,eccf,o3mmr); if(error) return;
// MPAS module_ra_cam.F:1646
if(dosw) {
// MPAS module_ra_cam.F:1651
aqsat(t,pmid,esat,qsat,pcols,ncol,pver,1,pver); if(error) return;
// MPAS module_ra_cam.F:1658
{ auto _lhs=rh.section(cam_section(1,ncol),cam_section(1,pver)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(cam_mul(cam_div(qm1.section(cam_section(1,ncol),cam_section(1,pver),1).linear(_slice),qsat.section(cam_section(1,ncol),cam_section(1,pver)).linear(_slice)),(cam_add(cam_mul((cam_sub(1.0f,epsilo)),qsat.section(cam_section(1,ncol),cam_section(1,pver)).linear(_slice)),epsilo))),(cam_add(cam_mul((cam_sub(1.0f,epsilo)),qm1.section(cam_section(1,ncol),cam_section(1,pver),1).linear(_slice)),epsilo))); }
// MPAS module_ra_cam.F:1662
if(radforce) {
// MPAS module_ra_cam.F:1664
{ auto _lhs=pmxrgnrf; for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=pmxrgn.linear(_slice); }
// MPAS module_ra_cam.F:1665
{ auto _lhs=nmxrgnrf; for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=nmxrgn.linear(_slice); }
// MPAS module_ra_cam.F:1667
get_rf_scales(scales); if(error) return;
// MPAS module_ra_cam.F:1669
get_aerosol(lchnk,julday,julian,dt,gmt,xtime,m_psp,m_psn,aerosoljp,aerosoljn,m_hybi,paerlev,naer,pint,pcols,pver,pverp,pverr,pverrp,aerosol,scales); if(error) return;
// MPAS module_ra_cam.F:1677
aerosol_indirect(ncol,lchnk,pcols,pver,ppcnst,landfrac,pmid,t,qm1,cld,zm,rel); if(error) return;
// MPAS module_ra_cam.F:1680
radcswmx(j,lchnk,ncol,pcols,pver,pverp,pnm,pbr,qm1,rh,o3mmr,aerosol,cld,cicewp,cliqwp,rel,rei,tauxcl,tauxci,eccf,coszrs,scon,solin,solcon,asdir,asdif,aldir,aldif,nmxrgnrf,pmxrgnrf,qrs,fsnt,fsntc,fsntoa,fsntoac,fsnirt,fsnrtc,fsnirtsq,fsns,fsnsc,fsdsc,fsds,sols,soll,solsd,solld,frc_day,fsup,fsupc,fsdn,fsdnc,aertau,aerssa,aerasm,aerfwd); if(error) return;
// MPAS module_ra_cam.F:1698
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1699
solin(i)=cam_mul(solin(i),1.e-3f);
// MPAS module_ra_cam.F:1700
fsnt(i)=cam_mul(fsnt(i),1.e-3f);
// MPAS module_ra_cam.F:1701
fsns(i)=cam_mul(fsns(i),1.e-3f);
// MPAS module_ra_cam.F:1702
fsntc(i)=cam_mul(fsntc(i),1.e-3f);
// MPAS module_ra_cam.F:1703
fsnsc(i)=cam_mul(fsnsc(i),1.e-3f);
// MPAS module_ra_cam.F:1704
}
// MPAS module_ra_cam.F:1705
{ auto _lhs=ftem.section(cam_section(1,ncol),cam_section(1,pver)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(qrs.section(cam_section(1,ncol),cam_section(1,pver)).linear(_slice),cpair); }
// MPAS module_ra_cam.F:1715
}
// MPAS module_ra_cam.F:1717
get_int_scales(scales); if(error) return;
// MPAS module_ra_cam.F:1719
get_aerosol(lchnk,julday,julian,dt,gmt,xtime,m_psp,m_psn,aerosoljp,aerosoljn,m_hybi,paerlev,naer,pint,pcols,pver,pverp,pverr,pverrp,aerosol,scales); if(error) return;
// MPAS module_ra_cam.F:1725
aerosol_indirect(ncol,lchnk,pcols,pver,ppcnst,landfrac,pmid,t,qm1,cld,zm,rel); if(error) return;
// MPAS module_ra_cam.F:1728
radcswmx(j,lchnk,ncol,pcols,pver,pverp,pnm,pbr,qm1,rh,o3mmr,aerosol,cld,cicewp,cliqwp,rel,rei,tauxcl,tauxci,eccf,coszrs,scon,solin,solcon,asdir,asdif,aldir,aldif,nmxrgn,pmxrgn,qrs,fsnt,fsntc,fsntoa,fsntoac,fsnirt,fsnrtc,fsnirtsq,fsns,fsnsc,fsdsc,fsds,sols,soll,solsd,solld,frc_day,fsup,fsupc,fsdn,fsdnc,aertau,aerssa,aerasm,aerfwd); if(error) return;
// MPAS module_ra_cam.F:1746
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1747
solin(i)=cam_mul(solin(i),1.e-3f);
// MPAS module_ra_cam.F:1748
fsds(i)=cam_mul(fsds(i),1.e-3f);
// MPAS module_ra_cam.F:1749
fsnirt(i)=cam_mul(fsnirt(i),1.e-3f);
// MPAS module_ra_cam.F:1750
fsnrtc(i)=cam_mul(fsnrtc(i),1.e-3f);
// MPAS module_ra_cam.F:1751
fsnirtsq(i)=cam_mul(fsnirtsq(i),1.e-3f);
// MPAS module_ra_cam.F:1752
fsnt(i)=cam_mul(fsnt(i),1.e-3f);
// MPAS module_ra_cam.F:1753
fsns(i)=cam_mul(fsns(i),1.e-3f);
// MPAS module_ra_cam.F:1754
fsntc(i)=cam_mul(fsntc(i),1.e-3f);
// MPAS module_ra_cam.F:1755
fsnsc(i)=cam_mul(fsnsc(i),1.e-3f);
// MPAS module_ra_cam.F:1756
fsdsc(i)=cam_mul(fsdsc(i),1.e-3f);
// MPAS module_ra_cam.F:1757
fsntoa(i)=cam_mul(fsntoa(i),1.e-3f);
// MPAS module_ra_cam.F:1758
fsntoac(i)=cam_mul(fsntoac(i),1.e-3f);
// MPAS module_ra_cam.F:1759
swcf(i)=cam_sub(fsntoa(i),fsntoac(i));
// MPAS module_ra_cam.F:1760
}
// MPAS module_ra_cam.F:1761
{ auto _lhs=ftem.section(cam_section(1,ncol),cam_section(1,pver)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(qrs.section(cam_section(1,ncol),cam_section(1,pver)).linear(_slice),cpair); }
// MPAS module_ra_cam.F:1764
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:1765
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1766
fsup(i,k)=cam_mul(fsup(i,k),1.e-3f);
// MPAS module_ra_cam.F:1767
fsupc(i,k)=cam_mul(fsupc(i,k),1.e-3f);
// MPAS module_ra_cam.F:1768
fsdn(i,k)=cam_mul(fsdn(i,k),1.e-3f);
// MPAS module_ra_cam.F:1769
fsdnc(i,k)=cam_mul(fsdnc(i,k),1.e-3f);
// MPAS module_ra_cam.F:1770
}
// MPAS module_ra_cam.F:1771
}
// MPAS module_ra_cam.F:1808
}
// MPAS module_ra_cam.F:1812
if(dolw) {
// MPAS module_ra_cam.F:1814
get_int_scales(scales); if(error) return;
// MPAS module_ra_cam.F:1816
get_aerosol(lchnk,julday,julian,dt,gmt,xtime,m_psp,m_psn,aerosoljp,aerosoljn,m_hybi,paerlev,naer,pint,pcols,pver,pverp,pverr,pverrp,aerosol,scales); if(error) return;
// MPAS module_ra_cam.F:1822
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1824
lwupcgs(i)=lwups(i);
// MPAS module_ra_cam.F:1825
}
// MPAS module_ra_cam.F:1834
if(trace_gas) {
// MPAS module_ra_cam.F:1840
radclwmx(lchnk,ncol,pcols,pver,pverp,lwupcgs,t,CamArray<double>::from(&qm1(1,1,1)),o3vmr,pbr,pnm,pmln,piln,CamArray<double>::from(&qm1(1,1,in2o)),CamArray<double>::from(&qm1(1,1,ich4)),CamArray<double>::from(&qm1(1,1,if11)),CamArray<double>::from(&qm1(1,1,if12)),cld,emis,pmxrgn,nmxrgn,qrl,doabsems,abstot,absnxt,emstot,flns,flnt,flnsc,flntc,flwds,flut,flutc,flup,flupc,fldn,fldnc,aerosol.section(cam_section(1,pcols),cam_section(1,pver),idxvolc)); if(error) return;
// MPAS module_ra_cam.F:1852
} else {
// MPAS module_ra_cam.F:1853
trcmix(lchnk,ncol,pcols,pver,pmid,clat,n2o,ch4,cfc11,cfc12); if(error) return;
// MPAS module_ra_cam.F:1858
radclwmx(lchnk,ncol,pcols,pver,pverp,lwupcgs,t,CamArray<double>::from(&qm1(1,1,1)),o3vmr,pbr,pnm,pmln,piln,n2o,ch4,cfc11,cfc12,cld,emis,pmxrgn,nmxrgn,qrl,doabsems,abstot,absnxt,emstot,flns,flnt,flnsc,flntc,flwds,flut,flutc,flup,flupc,fldn,fldnc,aerosol.section(cam_section(1,pcols),cam_section(1,pver),idxvolc)); if(error) return;
// MPAS module_ra_cam.F:1869
}
// MPAS module_ra_cam.F:1873
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1874
flnt(i)=cam_mul(flnt(i),1.e-3f);
// MPAS module_ra_cam.F:1875
flut(i)=cam_mul(flut(i),1.e-3f);
// MPAS module_ra_cam.F:1876
flutc(i)=cam_mul(flutc(i),1.e-3f);
// MPAS module_ra_cam.F:1877
flns(i)=cam_mul(flns(i),1.e-3f);
// MPAS module_ra_cam.F:1878
flntc(i)=cam_mul(flntc(i),1.e-3f);
// MPAS module_ra_cam.F:1879
flnsc(i)=cam_mul(flnsc(i),1.e-3f);
// MPAS module_ra_cam.F:1880
flwds(i)=cam_mul(flwds(i),1.e-3f);
// MPAS module_ra_cam.F:1881
lwcf(i)=cam_sub(flutc(i),flut(i));
// MPAS module_ra_cam.F:1882
}
// MPAS module_ra_cam.F:1885
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:1886
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1887
flup(i,k)=cam_mul(flup(i,k),1.e-3f);
// MPAS module_ra_cam.F:1888
flupc(i,k)=cam_mul(flupc(i,k),1.e-3f);
// MPAS module_ra_cam.F:1889
fldn(i,k)=cam_mul(fldn(i,k),1.e-3f);
// MPAS module_ra_cam.F:1890
fldnc(i,k)=cam_mul(fldnc(i,k),1.e-3f);
// MPAS module_ra_cam.F:1891
}
// MPAS module_ra_cam.F:1892
}
// MPAS module_ra_cam.F:1906
}
// MPAS module_ra_cam.F:1908
return;
}
// MPAS module_ra_cam.F:1910-2030
CAM_ROUTINE void param_cldoptics_calc(const int& ncol,const int& pcols,const int& pver,const int& pverp,const int& pverr,const int& pverrp,const int& ppcnst,const CamArray<double>& _q,const CamArray<double>& _cldn,const CamArray<double>& _landfrac,const CamArray<double>& _landm,const CamArray<double>& _icefrac,const CamArray<double>& _pdel,const CamArray<double>& _t,const CamArray<double>& _ps,const CamArray<double>& _pmid,const CamArray<double>& _pint,const CamArray<double>& _cicewp,const CamArray<double>& _cliqwp,const CamArray<double>& _emis,const CamArray<double>& _rel,const CamArray<double>& _rei,const CamArray<double>& _pmxrgn,const CamArray<int>& _nmxrgn,const CamArray<double>& _snowh) {
CamFrame _frame(arena);
CAM_TRACE("param_cldoptics_calc.q");
auto q=_q.rebound(1,pcols,1,pver,1,ppcnst);
CAM_TRACE("param_cldoptics_calc.cldn");
auto cldn=_cldn.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.landfrac");
auto landfrac=_landfrac.rebound(1,pcols);
CAM_TRACE("param_cldoptics_calc.landm");
auto landm=_landm.rebound(1,pcols);
CAM_TRACE("param_cldoptics_calc.icefrac");
auto icefrac=_icefrac.rebound(1,pcols);
CAM_TRACE("param_cldoptics_calc.pdel");
auto pdel=_pdel.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.t");
auto t=_t.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.ps");
auto ps=_ps.rebound(1,pcols);
CAM_TRACE("param_cldoptics_calc.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("param_cldoptics_calc.cicewp");
auto cicewp=_cicewp.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.cliqwp");
auto cliqwp=_cliqwp.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.emis");
auto emis=_emis.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.rel");
auto rel=_rel.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.rei");
auto rei=_rei.rebound(1,pcols,1,pver);
CAM_TRACE("param_cldoptics_calc.pmxrgn");
auto pmxrgn=_pmxrgn.rebound(1,pcols,1,cam_add(pver,1));
CAM_TRACE("param_cldoptics_calc.nmxrgn");
auto nmxrgn=_nmxrgn.rebound(1,pcols);
CAM_TRACE("param_cldoptics_calc.snowh");
auto snowh=_snowh.rebound(1,pcols);
auto cwp=arena.array<double>(1,pcols,1,pver);
auto effcld=arena.array<double>(1,pcols,1,pver);
auto gicewp=arena.array<double>(1,pcols,1,pver);
auto gliqwp=arena.array<double>(1,pcols,1,pver);
auto gwp=arena.array<double>(1,pcols,1,pver);
auto hl=arena.array<double>(1,pcols);
auto tgicewp=arena.array<double>(1,pcols);
auto tgliqwp=arena.array<double>(1,pcols);
auto tgwp=arena.array<double>(1,pcols);
auto tpw=arena.array<double>(1,pcols);
auto clwpold=arena.array<double>(1,pcols,1,pver);
auto ficemr=arena.array<double>(1,pcols,1,pver);
double rgrav;
int i;
int k;
int lchnk;
// MPAS module_ra_cam.F:1972
{ auto _lhs=tgicewp.section(cam_section(1,ncol)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam.F:1973
{ auto _lhs=tgliqwp.section(cam_section(1,ncol)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam.F:1974
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:1975
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1976
gicewp(i,k)=cam_mul(cam_div(cam_mul(q(i,k,ixcldice),pdel(i,k)),gravmks),1000.0f);
// MPAS module_ra_cam.F:1977
gliqwp(i,k)=cam_mul(cam_div(cam_mul(q(i,k,ixcldliq),pdel(i,k)),gravmks),1000.0f);
// MPAS module_ra_cam.F:1979
cicewp(i,k)=cam_div(gicewp(i,k),cam_max(0.01,cldn(i,k)));
// MPAS module_ra_cam.F:1980
cliqwp(i,k)=cam_div(gliqwp(i,k),cam_max(0.01,cldn(i,k)));
// MPAS module_ra_cam.F:1982
ficemr(i,k)=cam_div(q(i,k,ixcldice),cam_max(1.e-10,(cam_add(q(i,k,ixcldice),q(i,k,ixcldliq)))));
// MPAS module_ra_cam.F:1985
tgicewp(i)=cam_add(tgicewp(i),gicewp(i,k));
// MPAS module_ra_cam.F:1986
tgliqwp(i)=cam_add(tgliqwp(i),gliqwp(i,k));
// MPAS module_ra_cam.F:1987
}
// MPAS module_ra_cam.F:1988
}
// MPAS module_ra_cam.F:1989
{ auto _lhs=tgwp.section(cam_section(1,ncol)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_add(tgicewp.section(cam_section(1,ncol)).linear(_slice),tgliqwp.section(cam_section(1,ncol)).linear(_slice)); }
// MPAS module_ra_cam.F:1990
{ auto _lhs=gwp.section(cam_section(1,ncol),cam_section(1,pver)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_add(gicewp.section(cam_section(1,ncol),cam_section(1,pver)).linear(_slice),gliqwp.section(cam_section(1,ncol),cam_section(1,pver)).linear(_slice)); }
// MPAS module_ra_cam.F:1991
{ auto _lhs=cwp.section(cam_section(1,ncol),cam_section(1,pver)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_add(cicewp.section(cam_section(1,ncol),cam_section(1,pver)).linear(_slice),cliqwp.section(cam_section(1,ncol),cam_section(1,pver)).linear(_slice)); }
// MPAS module_ra_cam.F:1994
{ auto _lhs=tpw.section(cam_section(1,ncol)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.0f; }
// MPAS module_ra_cam.F:1995
rgrav=cam_div(1.0f,gravmks);
// MPAS module_ra_cam.F:1996
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:1997
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:1998
tpw(i)=cam_add(tpw(i),cam_mul(cam_mul(pdel(i,k),q(i,k,1)),rgrav));
// MPAS module_ra_cam.F:1999
}
// MPAS module_ra_cam.F:2000
}
// MPAS module_ra_cam.F:2006
cldefr(lchnk,ncol,pcols,pver,pverp,landfrac,t,rel,rei,ps,pmid,landm,icefrac,snowh); if(error) return;
// MPAS module_ra_cam.F:2009
cldems(lchnk,ncol,pcols,pver,pverp,cwp,ficemr,rei,emis); if(error) return;
// MPAS module_ra_cam.F:2012
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:2013
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2014
effcld(i,k)=cam_mul(cldn(i,k),emis(i,k));
// MPAS module_ra_cam.F:2015
}
// MPAS module_ra_cam.F:2016
}
// MPAS module_ra_cam.F:2019
cldovrlap(lchnk,ncol,pcols,pver,pverp,pint,cldn,nmxrgn,pmxrgn); if(error) return;
}
// MPAS module_ra_cam.F:2032-3438
CAM_ROUTINE void radabs(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _pbr,const CamArray<double>& _pnm,const CamArray<double>& _co2em,const CamArray<double>& _co2eml,const CamArray<double>& _tplnka,const CamArray<double>& _s2c,const CamArray<double>& _tcg,const CamArray<double>& _w,const CamArray<double>& _h2otr,const CamArray<double>& _plco2,const CamArray<double>& _plh2o,const CamArray<double>& _co2t,const CamArray<double>& _tint,const CamArray<double>& _tlayr,const CamArray<double>& _plol,const CamArray<double>& _plos,const CamArray<double>& _pmln,const CamArray<double>& _piln,const CamArray<double>& _ucfc11,const CamArray<double>& _ucfc12,const CamArray<double>& _un2o0,const CamArray<double>& _un2o1,const CamArray<double>& _uch4,const CamArray<double>& _uco211,const CamArray<double>& _uco212,const CamArray<double>& _uco213,const CamArray<double>& _uco221,const CamArray<double>& _uco222,const CamArray<double>& _uco223,const CamArray<double>& _uptype,const CamArray<double>& _bn2o0,const CamArray<double>& _bn2o1,const CamArray<double>& _bch4,const CamArray<double>& _abplnk1,const CamArray<double>& _abplnk2,const CamArray<double>& _abstot,const CamArray<double>& _absnxt,const CamArray<double>& _plh2ob,const CamArray<double>& _wb,const CamArray<double>& _aer_mpp,const CamArray<double>& _aer_trn_ttl) {
CamFrame _frame(arena);
CAM_TRACE("radabs.pbr");
auto pbr=_pbr.rebound(1,pcols,1,pver);
CAM_TRACE("radabs.pnm");
auto pnm=_pnm.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.co2em");
auto co2em=_co2em.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.co2eml");
auto co2eml=_co2eml.rebound(1,pcols,1,pver);
CAM_TRACE("radabs.tplnka");
auto tplnka=_tplnka.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.s2c");
auto s2c=_s2c.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.tcg");
auto tcg=_tcg.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.w");
auto w=_w.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.h2otr");
auto h2otr=_h2otr.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.plco2");
auto plco2=_plco2.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.plh2o");
auto plh2o=_plh2o.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.co2t");
auto co2t=_co2t.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.tint");
auto tint=_tint.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.tlayr");
auto tlayr=_tlayr.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.plol");
auto plol=_plol.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.plos");
auto plos=_plos.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.pmln");
auto pmln=_pmln.rebound(1,pcols,1,pver);
CAM_TRACE("radabs.piln");
auto piln=_piln.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.ucfc11");
auto ucfc11=_ucfc11.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.ucfc12");
auto ucfc12=_ucfc12.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.un2o0");
auto un2o0=_un2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.un2o1");
auto un2o1=_un2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.uch4");
auto uch4=_uch4.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.uco211");
auto uco211=_uco211.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.uco212");
auto uco212=_uco212.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.uco213");
auto uco213=_uco213.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.uco221");
auto uco221=_uco221.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.uco222");
auto uco222=_uco222.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.uco223");
auto uco223=_uco223.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.uptype");
auto uptype=_uptype.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.bn2o0");
auto bn2o0=_bn2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.bn2o1");
auto bn2o1=_bn2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.bch4");
auto bch4=_bch4.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.abplnk1");
auto abplnk1=_abplnk1.rebound(1,14,1,pcols,1,pverp);
CAM_TRACE("radabs.abplnk2");
auto abplnk2=_abplnk2.rebound(1,14,1,pcols,1,pverp);
CAM_TRACE("radabs.abstot");
auto abstot=_abstot.rebound(1,pcols,1,pverp,1,pverp);
CAM_TRACE("radabs.absnxt");
auto absnxt=_absnxt.rebound(1,pcols,1,pver,1,4);
CAM_TRACE("radabs.plh2ob");
auto plh2ob=_plh2ob.rebound(1,nbands,1,pcols,1,pverp);
CAM_TRACE("radabs.wb");
auto wb=_wb.rebound(1,nbands,1,pcols,1,pverp);
CAM_TRACE("radabs.aer_mpp");
auto aer_mpp=_aer_mpp.rebound(1,pcols,1,pverp);
CAM_TRACE("radabs.aer_trn_ttl");
auto aer_trn_ttl=_aer_trn_ttl.rebound(1,pcols,1,pverp,1,pverp,1,bnd_nbr_lw);
int i;
int k;
int k1;
int k2;
int kn;
int wvl;
auto abstrc=arena.array<double>(1,pcols);
auto bplnk=arena.array<double>(1,14,1,pcols,1,4);
auto pnew=arena.array<double>(1,pcols);
auto pnewb=arena.array<double>(1,nbands);
auto u=arena.array<double>(1,pcols);
auto ub=arena.array<double>(1,nbands);
auto tbar=arena.array<double>(1,pcols,1,4);
auto emm=arena.array<double>(1,pcols,1,4);
auto o3emm=arena.array<double>(1,pcols,1,4);
double o3bndi;
auto temh2o=arena.array<double>(1,pcols,1,4);
double k21;
double k22;
auto uc1=arena.array<double>(1,pcols);
auto to3h2o=arena.array<double>(1,pcols);
double pi;
auto sqti=arena.array<double>(1,pcols);
double et;
double et2;
double et4;
double omet;
double f1co2;
auto f2co2=arena.array<double>(1,pcols);
auto f3co2=arena.array<double>(1,pcols);
auto t1co2=arena.array<double>(1,pcols);
double sqwp;
auto f1sqwp=arena.array<double>(1,pcols);
double oneme;
double alphat;
double wco2;
double posqt;
auto u7=arena.array<double>(1,pcols);
double u8;
double u9;
double u13;
auto rbeta7=arena.array<double>(1,pcols);
double rbeta8;
double rbeta9;
double rbeta13;
double tpatha;
auto abso=arena.array<double>(1,pcols,1,4);
auto dtx=arena.array<double>(1,pcols);
auto dty=arena.array<double>(1,pcols);
auto term7=arena.array<double>(1,pcols,1,2);
auto term8=arena.array<double>(1,pcols,1,2);
double tr1;
auto tr10=arena.array<double>(1,pcols);
double tr2;
double tr5;
double tr6;
auto tr9=arena.array<double>(1,pcols);
auto sqrtu=arena.array<double>(1,pcols);
auto fwk=arena.array<double>(1,pcols);
auto fwku=arena.array<double>(1,pcols);
auto to3co2=arena.array<double>(1,pcols);
auto dpnm=arena.array<double>(1,pcols);
auto pnmsq=arena.array<double>(1,pcols,1,pverp);
auto dw=arena.array<double>(1,pcols);
auto uinpl=arena.array<double>(1,pcols,1,4);
auto winpl=arena.array<double>(1,pcols,1,4);
auto zinpl=arena.array<double>(1,pcols,1,4);
auto pinpl=arena.array<double>(1,pcols,1,4);
auto dplh2o=arena.array<double>(1,pcols);
double r293;
double r250;
double r3205;
double r300;
double rsslp;
double r2sslp;
double ds2c;
double dplos;
double dplol;
double tlocal;
double beta;
double rphat;
double tcrfac;
double tmp1;
double u1;
double realnu;
double tmp2;
double u2;
double rsqti;
double tpath;
double tmp3;
double rdpnmsq;
double rdpnm;
double p1;
double p2;
double dtym10;
double dplco2;
double te;
double denom;
auto th2o=arena.array<double>(1,pcols);
auto tco2=arena.array<double>(1,pcols);
auto to3=arena.array<double>(1,pcols);
auto trab2=arena.array<double>(1,pcols);
double absbnd;
auto dbvtit=arena.array<double>(1,pcols,1,pverp);
auto dbvtly=arena.array<double>(1,pcols,1,pver);
double fa;
double a_star;
double l_star;
double c_star;
double te1;
double te2;
double te3;
double te4;
double te5;
double log_u;
double log_uc;
double log_p;
double t_p;
double t_e;
int iu;
int iu1;
int iuc;
int iuc1;
int ip;
int ip1;
int itp;
int itp1;
int ite;
int ite1;
int irh;
int irh1;
double dvar;
double uvar;
double uscl;
double wu;
double wu1;
double wuc;
double wuc1;
double wp;
double wp1;
double wtp;
double wtp1;
double wte;
double wte1;
double wrh;
double wrh1;
double w_0_0_;
double w_0_1_;
double w_1_0_;
double w_1_1_;
double w_0_00;
double w_0_01;
double w_0_10;
double w_0_11;
double w_1_00;
double w_1_01;
double w_1_10;
double w_1_11;
double w00_00;
double w00_01;
double w00_10;
double w00_11;
double w01_00;
double w01_01;
double w01_10;
double w01_11;
double w10_00;
double w10_01;
double w10_10;
double w10_11;
double w11_00;
double w11_01;
double w11_10;
double w11_11;
int ib;
double pch2o;
double fch2o;
double uch2o;
double fdif;
double sslp_mks;
double esx;
double qsx;
double pnew_mks;
double q_path;
double rh_path;
double omeps;
int iest;
int bnd_idx;
double aer_pth_dlt;
auto aer_pth_ngh=arena.array<double>(1,pcols);
double odap_aer_ttl;
auto aer_trn_ngh=arena.array<double>(1,pcols,1,bnd_nbr_lw);
double t;
auto dbvt=[&](double t) { return cam_div((cam_add(cam_neg(2.8911366682e-4f),cam_mul((cam_add(2.3771251896e-6f,cam_mul(1.1305188929e-10f,t))),t))),(cam_add(1.0f,cam_mul((cam_add(cam_neg(6.1364820707e-3f),cam_mul(1.5550319767e-5f,t))),t)))); };
// MPAS module_ra_cam.F:2415
for(k2=1;k2<=cam_sub(ntoplw,1);k2+=1) {
// MPAS module_ra_cam.F:2416
for(k1=1;k1<=cam_sub(ntoplw,1);k1+=1) {
// MPAS module_ra_cam.F:2417
{ auto _lhs=abstot.section(cam_section(1,pcols),k1,k2); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=inf; }
// MPAS module_ra_cam.F:2418
}
// MPAS module_ra_cam.F:2419
}
// MPAS module_ra_cam.F:2420
for(k2=1;k2<=4;k2+=1) {
// MPAS module_ra_cam.F:2421
for(k1=1;k1<=cam_sub(ntoplw,1);k1+=1) {
// MPAS module_ra_cam.F:2422
{ auto _lhs=absnxt.section(cam_section(1,pcols),k1,k2); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=inf; }
// MPAS module_ra_cam.F:2423
}
// MPAS module_ra_cam.F:2424
}
// MPAS module_ra_cam.F:2426
for(k=ntoplw;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:2427
{ auto _lhs=abstot.section(cam_section(1,pcols),k,k); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=inf; }
// MPAS module_ra_cam.F:2428
}
// MPAS module_ra_cam.F:2430
for(k=ntoplw;k<=pver;k+=1) {
// MPAS module_ra_cam.F:2431
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2432
dbvtly(i,k)=dbvt(tlayr(i,cam_add(k,1)));
// MPAS module_ra_cam.F:2433
dbvtit(i,k)=dbvt(tint(i,k));
// MPAS module_ra_cam.F:2434
}
// MPAS module_ra_cam.F:2435
}
// MPAS module_ra_cam.F:2436
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2437
dbvtit(i,pverp)=dbvt(tint(i,pverp));
// MPAS module_ra_cam.F:2438
}
// MPAS module_ra_cam.F:2440
r293=cam_div(1.f,293.f);
// MPAS module_ra_cam.F:2441
r250=cam_div(1.f,250.f);
// MPAS module_ra_cam.F:2442
r3205=cam_div(1.f,.3205f);
// MPAS module_ra_cam.F:2443
r300=cam_div(1.f,300.f);
// MPAS module_ra_cam.F:2444
rsslp=cam_div(1.f,sslp);
// MPAS module_ra_cam.F:2445
r2sslp=cam_div(1.f,(cam_mul(2.f,sslp)));
// MPAS module_ra_cam.F:2449
fdif=1.66f;
// MPAS module_ra_cam.F:2450
sslp_mks=cam_div(sslp,10.0f);
// MPAS module_ra_cam.F:2451
omeps=cam_sub(1.0f,epsilo);
// MPAS module_ra_cam.F:2468
for(k=ntoplw;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:2469
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2470
pnmsq(i,k)=cam_pow(pnm(i,k),2);
// MPAS module_ra_cam.F:2471
dtx(i)=cam_sub(tplnka(i,k),250.f);
// MPAS module_ra_cam.F:2472
}
// MPAS module_ra_cam.F:2473
}
// MPAS module_ra_cam.F:2477
for(k1=pverp;k1>=ntoplw;k1+=cam_neg(1)) {
// MPAS module_ra_cam.F:2478
for(k2=pverp;k2>=ntoplw;k2+=cam_neg(1)) {
// MPAS module_ra_cam.F:2479
if((k1==k2)) {continue;}
// MPAS module_ra_cam.F:2480
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2481
dplh2o(i)=cam_sub(plh2o(i,k1),plh2o(i,k2));
// MPAS module_ra_cam.F:2482
u(i)=cam_abs(dplh2o(i));
// MPAS module_ra_cam.F:2483
sqrtu(i)=cam_sqrt(u(i));
// MPAS module_ra_cam.F:2484
ds2c=cam_abs(cam_sub(s2c(i,k1),s2c(i,k2)));
// MPAS module_ra_cam.F:2485
dw(i)=cam_abs(cam_sub(w(i,k1),w(i,k2)));
// MPAS module_ra_cam.F:2486
uc1(i)=cam_div(cam_mul((cam_add(ds2c,cam_mul(1.7e-3f,u(i)))),(cam_add(1.f,cam_mul(2.f,ds2c)))),(cam_add(1.f,cam_mul(15.f,ds2c))));
// MPAS module_ra_cam.F:2487
pch2o=ds2c;
// MPAS module_ra_cam.F:2488
pnew(i)=cam_div(u(i),dw(i));
// MPAS module_ra_cam.F:2489
pnew_mks=cam_mul(pnew(i),sslp_mks);
// MPAS module_ra_cam.F:2493
tpatha=cam_div(cam_abs(cam_sub(tcg(i,k1),tcg(i,k2))),dw(i));
// MPAS module_ra_cam.F:2494
t_p=cam_min(cam_max(tpatha,min_tp_h2o),max_tp_h2o);
// MPAS module_ra_cam.F:2495
iest=cam_sub(floor(t_p),min_tp_h2o);
// MPAS module_ra_cam.F:2496
esx=cam_add(estblh2o(iest),cam_mul((cam_sub(estblh2o(cam_add(iest,1)),estblh2o(iest))),(cam_sub(cam_sub(t_p,min_tp_h2o),iest))));
// MPAS module_ra_cam.F:2498
qsx=cam_div(cam_mul(epsilo,esx),(cam_sub(pnew_mks,cam_mul(omeps,esx))));
// MPAS module_ra_cam.F:2502
q_path=cam_div(cam_div(dw(i),cam_abs(cam_sub(pnm(i,k1),pnm(i,k2)))),rga);
// MPAS module_ra_cam.F:2513
ub(1)=cam_div(cam_abs(cam_sub(plh2ob(1,i,k1),plh2ob(1,i,k2))),psi(t_p,1));
// MPAS module_ra_cam.F:2514
ub(2)=cam_div(cam_abs(cam_sub(plh2ob(2,i,k1),plh2ob(2,i,k2))),psi(t_p,2));
// MPAS module_ra_cam.F:2516
pnewb(1)=cam_mul(cam_div(ub(1),cam_abs(cam_sub(wb(1,i,k1),wb(1,i,k2)))),phi(t_p,1));
// MPAS module_ra_cam.F:2517
pnewb(2)=cam_mul(cam_div(ub(2),cam_abs(cam_sub(wb(2,i,k1),wb(2,i,k2)))),phi(t_p,2));
// MPAS module_ra_cam.F:2519
dtx(i)=cam_sub(tplnka(i,k2),250.f);
// MPAS module_ra_cam.F:2520
dty(i)=cam_sub(tpatha,250.f);
// MPAS module_ra_cam.F:2522
fwk(i)=cam_add(fwcoef,cam_div(fwc1,(cam_add(1.f,cam_mul(fwc2,u(i))))));
// MPAS module_ra_cam.F:2523
fwku(i)=cam_mul(fwk(i),u(i));
// MPAS module_ra_cam.F:2546
te1=tplnka(i,k2);
// MPAS module_ra_cam.F:2547
te2=cam_mul(te1,te1);
// MPAS module_ra_cam.F:2548
te3=cam_mul(te2,te1);
// MPAS module_ra_cam.F:2549
te4=cam_mul(te3,te1);
// MPAS module_ra_cam.F:2550
te5=cam_mul(te4,te1);
// MPAS module_ra_cam.F:2555
dvar=cam_div((cam_sub(t_p,min_tp_h2o)),dtp_h2o);
// MPAS module_ra_cam.F:2556
itp=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_tp,1));
// MPAS module_ra_cam.F:2557
itp1=cam_add(itp,1);
// MPAS module_ra_cam.F:2558
wtp=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:2559
wtp1=cam_sub(1.0f,wtp);
// MPAS module_ra_cam.F:2561
t_e=cam_min(cam_max(cam_sub(tplnka(i,k2),t_p),min_te_h2o),max_te_h2o);
// MPAS module_ra_cam.F:2562
dvar=cam_div((cam_sub(t_e,min_te_h2o)),dte_h2o);
// MPAS module_ra_cam.F:2563
ite=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_te,1));
// MPAS module_ra_cam.F:2564
ite1=cam_add(ite,1);
// MPAS module_ra_cam.F:2565
wte=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:2566
wte1=cam_sub(1.0f,wte);
// MPAS module_ra_cam.F:2568
rh_path=cam_min(cam_max(cam_div(q_path,qsx),min_rh_h2o),max_rh_h2o);
// MPAS module_ra_cam.F:2569
dvar=cam_div((cam_sub(rh_path,min_rh_h2o)),drh_h2o);
// MPAS module_ra_cam.F:2570
irh=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_rh,1));
// MPAS module_ra_cam.F:2571
irh1=cam_add(irh,1);
// MPAS module_ra_cam.F:2572
wrh=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:2573
wrh1=cam_sub(1.0f,wrh);
// MPAS module_ra_cam.F:2575
w_0_0_=cam_mul(wtp,wte);
// MPAS module_ra_cam.F:2576
w_0_1_=cam_mul(wtp,wte1);
// MPAS module_ra_cam.F:2577
w_1_0_=cam_mul(wtp1,wte);
// MPAS module_ra_cam.F:2578
w_1_1_=cam_mul(wtp1,wte1);
// MPAS module_ra_cam.F:2580
w_0_00=cam_mul(w_0_0_,wrh);
// MPAS module_ra_cam.F:2581
w_0_01=cam_mul(w_0_0_,wrh1);
// MPAS module_ra_cam.F:2582
w_0_10=cam_mul(w_0_1_,wrh);
// MPAS module_ra_cam.F:2583
w_0_11=cam_mul(w_0_1_,wrh1);
// MPAS module_ra_cam.F:2584
w_1_00=cam_mul(w_1_0_,wrh);
// MPAS module_ra_cam.F:2585
w_1_01=cam_mul(w_1_0_,wrh1);
// MPAS module_ra_cam.F:2586
w_1_10=cam_mul(w_1_1_,wrh);
// MPAS module_ra_cam.F:2587
w_1_11=cam_mul(w_1_1_,wrh1);
// MPAS module_ra_cam.F:2640
fch2o=fh2oself(t_p);
// MPAS module_ra_cam.F:2641
uch2o=cam_div((cam_mul(pch2o,epsilo)),(cam_mul(q_path,fch2o)));
// MPAS module_ra_cam.F:2646
ib=1;
// MPAS module_ra_cam.F:2648
uvar=cam_mul(ub(ib),fdif);
// MPAS module_ra_cam.F:2649
log_u=cam_min(cam_log10(cam_max(uvar,min_u_h2o)),max_lu_h2o);
// MPAS module_ra_cam.F:2650
dvar=cam_div((cam_sub(log_u,min_lu_h2o)),dlu_h2o);
// MPAS module_ra_cam.F:2651
iu=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_u,1));
// MPAS module_ra_cam.F:2652
iu1=cam_add(iu,1);
// MPAS module_ra_cam.F:2653
wu=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:2654
wu1=cam_sub(1.0f,wu);
// MPAS module_ra_cam.F:2656
log_p=cam_min(cam_log10(cam_max(pnewb(ib),min_p_h2o)),max_lp_h2o);
// MPAS module_ra_cam.F:2657
dvar=cam_div((cam_sub(log_p,min_lp_h2o)),dlp_h2o);
// MPAS module_ra_cam.F:2658
ip=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_p,1));
// MPAS module_ra_cam.F:2659
ip1=cam_add(ip,1);
// MPAS module_ra_cam.F:2660
wp=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:2661
wp1=cam_sub(1.0f,wp);
// MPAS module_ra_cam.F:2663
w00_00=cam_mul(wp,w_0_00);
// MPAS module_ra_cam.F:2664
w00_01=cam_mul(wp,w_0_01);
// MPAS module_ra_cam.F:2665
w00_10=cam_mul(wp,w_0_10);
// MPAS module_ra_cam.F:2666
w00_11=cam_mul(wp,w_0_11);
// MPAS module_ra_cam.F:2667
w01_00=cam_mul(wp,w_1_00);
// MPAS module_ra_cam.F:2668
w01_01=cam_mul(wp,w_1_01);
// MPAS module_ra_cam.F:2669
w01_10=cam_mul(wp,w_1_10);
// MPAS module_ra_cam.F:2670
w01_11=cam_mul(wp,w_1_11);
// MPAS module_ra_cam.F:2671
w10_00=cam_mul(wp1,w_0_00);
// MPAS module_ra_cam.F:2672
w10_01=cam_mul(wp1,w_0_01);
// MPAS module_ra_cam.F:2673
w10_10=cam_mul(wp1,w_0_10);
// MPAS module_ra_cam.F:2674
w10_11=cam_mul(wp1,w_0_11);
// MPAS module_ra_cam.F:2675
w11_00=cam_mul(wp1,w_1_00);
// MPAS module_ra_cam.F:2676
w11_01=cam_mul(wp1,w_1_01);
// MPAS module_ra_cam.F:2677
w11_10=cam_mul(wp1,w_1_10);
// MPAS module_ra_cam.F:2678
w11_11=cam_mul(wp1,w_1_11);
// MPAS module_ra_cam.F:2682
fa=cam_add(cam_add(cam_add(cam_add(cam_add(fat(1,ib),cam_mul(fat(2,ib),te1)),cam_mul(fat(3,ib),te2)),cam_mul(fat(4,ib),te3)),cam_mul(fat(5,ib),te4)),cam_mul(fat(6,ib),te5));
// MPAS module_ra_cam.F:2689
a_star=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_mul(cam_mul(ah2onw(ip,itp,iu,ite,irh),w11_11),wu1),cam_mul(cam_mul(ah2onw(ip,itp,iu,ite,irh1),w11_10),wu1)),cam_mul(cam_mul(ah2onw(ip,itp,iu,ite1,irh),w11_01),wu1)),cam_mul(cam_mul(ah2onw(ip,itp,iu,ite1,irh1),w11_00),wu1)),cam_mul(cam_mul(ah2onw(ip,itp,iu1,ite,irh),w11_11),wu)),cam_mul(cam_mul(ah2onw(ip,itp,iu1,ite,irh1),w11_10),wu)),cam_mul(cam_mul(ah2onw(ip,itp,iu1,ite1,irh),w11_01),wu)),cam_mul(cam_mul(ah2onw(ip,itp,iu1,ite1,irh1),w11_00),wu)),cam_mul(cam_mul(ah2onw(ip,itp1,iu,ite,irh),w10_11),wu1)),cam_mul(cam_mul(ah2onw(ip,itp1,iu,ite,irh1),w10_10),wu1)),cam_mul(cam_mul(ah2onw(ip,itp1,iu,ite1,irh),w10_01),wu1)),cam_mul(cam_mul(ah2onw(ip,itp1,iu,ite1,irh1),w10_00),wu1)),cam_mul(cam_mul(ah2onw(ip,itp1,iu1,ite,irh),w10_11),wu)),cam_mul(cam_mul(ah2onw(ip,itp1,iu1,ite,irh1),w10_10),wu)),cam_mul(cam_mul(ah2onw(ip,itp1,iu1,ite1,irh),w10_01),wu)),cam_mul(cam_mul(ah2onw(ip,itp1,iu1,ite1,irh1),w10_00),wu)),cam_mul(cam_mul(ah2onw(ip1,itp,iu,ite,irh),w01_11),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp,iu,ite,irh1),w01_10),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp,iu,ite1,irh),w01_01),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp,iu,ite1,irh1),w01_00),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp,iu1,ite,irh),w01_11),wu)),cam_mul(cam_mul(ah2onw(ip1,itp,iu1,ite,irh1),w01_10),wu)),cam_mul(cam_mul(ah2onw(ip1,itp,iu1,ite1,irh),w01_01),wu)),cam_mul(cam_mul(ah2onw(ip1,itp,iu1,ite1,irh1),w01_00),wu)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu,ite,irh),w00_11),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu,ite,irh1),w00_10),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu,ite1,irh),w00_01),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu,ite1,irh1),w00_00),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu1,ite,irh),w00_11),wu)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu1,ite,irh1),w00_10),wu)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu1,ite1,irh),w00_01),wu)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu1,ite1,irh1),w00_00),wu));
// MPAS module_ra_cam.F:2722
abso(i,ib)=cam_min(cam_max(cam_mul(fa,(cam_sub(1.0f,cam_mul((cam_sub(1.0f,a_star)),aer_trn_ttl(i,k1,k2,ib))))),0.0),1.0);
// MPAS module_ra_cam.F:2728
if((uvar<min_u_h2o)) {
// MPAS module_ra_cam.F:2729
uscl=cam_div(uvar,min_u_h2o);
// MPAS module_ra_cam.F:2730
abso(i,ib)=cam_mul(abso(i,ib),uscl);
// MPAS module_ra_cam.F:2731
}
// MPAS module_ra_cam.F:2736
ib=2;
// MPAS module_ra_cam.F:2738
uvar=cam_mul(ub(ib),fdif);
// MPAS module_ra_cam.F:2739
log_u=cam_min(cam_log10(cam_max(uvar,min_u_h2o)),max_lu_h2o);
// MPAS module_ra_cam.F:2740
dvar=cam_div((cam_sub(log_u,min_lu_h2o)),dlu_h2o);
// MPAS module_ra_cam.F:2741
iu=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_u,1));
// MPAS module_ra_cam.F:2742
iu1=cam_add(iu,1);
// MPAS module_ra_cam.F:2743
wu=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:2744
wu1=cam_sub(1.0f,wu);
// MPAS module_ra_cam.F:2746
log_p=cam_min(cam_log10(cam_max(pnewb(ib),min_p_h2o)),max_lp_h2o);
// MPAS module_ra_cam.F:2747
dvar=cam_div((cam_sub(log_p,min_lp_h2o)),dlp_h2o);
// MPAS module_ra_cam.F:2748
ip=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_p,1));
// MPAS module_ra_cam.F:2749
ip1=cam_add(ip,1);
// MPAS module_ra_cam.F:2750
wp=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:2751
wp1=cam_sub(1.0f,wp);
// MPAS module_ra_cam.F:2753
w00_00=cam_mul(wp,w_0_00);
// MPAS module_ra_cam.F:2754
w00_01=cam_mul(wp,w_0_01);
// MPAS module_ra_cam.F:2755
w00_10=cam_mul(wp,w_0_10);
// MPAS module_ra_cam.F:2756
w00_11=cam_mul(wp,w_0_11);
// MPAS module_ra_cam.F:2757
w01_00=cam_mul(wp,w_1_00);
// MPAS module_ra_cam.F:2758
w01_01=cam_mul(wp,w_1_01);
// MPAS module_ra_cam.F:2759
w01_10=cam_mul(wp,w_1_10);
// MPAS module_ra_cam.F:2760
w01_11=cam_mul(wp,w_1_11);
// MPAS module_ra_cam.F:2761
w10_00=cam_mul(wp1,w_0_00);
// MPAS module_ra_cam.F:2762
w10_01=cam_mul(wp1,w_0_01);
// MPAS module_ra_cam.F:2763
w10_10=cam_mul(wp1,w_0_10);
// MPAS module_ra_cam.F:2764
w10_11=cam_mul(wp1,w_0_11);
// MPAS module_ra_cam.F:2765
w11_00=cam_mul(wp1,w_1_00);
// MPAS module_ra_cam.F:2766
w11_01=cam_mul(wp1,w_1_01);
// MPAS module_ra_cam.F:2767
w11_10=cam_mul(wp1,w_1_10);
// MPAS module_ra_cam.F:2768
w11_11=cam_mul(wp1,w_1_11);
// MPAS module_ra_cam.F:2770
log_uc=cam_min(cam_log10(cam_max(cam_mul(uch2o,fdif),min_u_h2o)),max_lu_h2o);
// MPAS module_ra_cam.F:2771
dvar=cam_div((cam_sub(log_uc,min_lu_h2o)),dlu_h2o);
// MPAS module_ra_cam.F:2772
iuc=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_u,1));
// MPAS module_ra_cam.F:2773
iuc1=cam_add(iuc,1);
// MPAS module_ra_cam.F:2774
wuc=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:2775
wuc1=cam_sub(1.0f,wuc);
// MPAS module_ra_cam.F:2779
fa=cam_add(cam_add(cam_add(cam_add(cam_add(fat(1,ib),cam_mul(fat(2,ib),te1)),cam_mul(fat(3,ib),te2)),cam_mul(fat(4,ib),te3)),cam_mul(fat(5,ib),te4)),cam_mul(fat(6,ib),te5));
// MPAS module_ra_cam.F:2786
l_star=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_mul(cam_mul(ln_ah2ow(ip,itp,iu,ite,irh),w11_11),wu1),cam_mul(cam_mul(ln_ah2ow(ip,itp,iu,ite,irh1),w11_10),wu1)),cam_mul(cam_mul(ln_ah2ow(ip,itp,iu,ite1,irh),w11_01),wu1)),cam_mul(cam_mul(ln_ah2ow(ip,itp,iu,ite1,irh1),w11_00),wu1)),cam_mul(cam_mul(ln_ah2ow(ip,itp,iu1,ite,irh),w11_11),wu)),cam_mul(cam_mul(ln_ah2ow(ip,itp,iu1,ite,irh1),w11_10),wu)),cam_mul(cam_mul(ln_ah2ow(ip,itp,iu1,ite1,irh),w11_01),wu)),cam_mul(cam_mul(ln_ah2ow(ip,itp,iu1,ite1,irh1),w11_00),wu)),cam_mul(cam_mul(ln_ah2ow(ip,itp1,iu,ite,irh),w10_11),wu1)),cam_mul(cam_mul(ln_ah2ow(ip,itp1,iu,ite,irh1),w10_10),wu1)),cam_mul(cam_mul(ln_ah2ow(ip,itp1,iu,ite1,irh),w10_01),wu1)),cam_mul(cam_mul(ln_ah2ow(ip,itp1,iu,ite1,irh1),w10_00),wu1)),cam_mul(cam_mul(ln_ah2ow(ip,itp1,iu1,ite,irh),w10_11),wu)),cam_mul(cam_mul(ln_ah2ow(ip,itp1,iu1,ite,irh1),w10_10),wu)),cam_mul(cam_mul(ln_ah2ow(ip,itp1,iu1,ite1,irh),w10_01),wu)),cam_mul(cam_mul(ln_ah2ow(ip,itp1,iu1,ite1,irh1),w10_00),wu)),cam_mul(cam_mul(ln_ah2ow(ip1,itp,iu,ite,irh),w01_11),wu1)),cam_mul(cam_mul(ln_ah2ow(ip1,itp,iu,ite,irh1),w01_10),wu1)),cam_mul(cam_mul(ln_ah2ow(ip1,itp,iu,ite1,irh),w01_01),wu1)),cam_mul(cam_mul(ln_ah2ow(ip1,itp,iu,ite1,irh1),w01_00),wu1)),cam_mul(cam_mul(ln_ah2ow(ip1,itp,iu1,ite,irh),w01_11),wu)),cam_mul(cam_mul(ln_ah2ow(ip1,itp,iu1,ite,irh1),w01_10),wu)),cam_mul(cam_mul(ln_ah2ow(ip1,itp,iu1,ite1,irh),w01_01),wu)),cam_mul(cam_mul(ln_ah2ow(ip1,itp,iu1,ite1,irh1),w01_00),wu)),cam_mul(cam_mul(ln_ah2ow(ip1,itp1,iu,ite,irh),w00_11),wu1)),cam_mul(cam_mul(ln_ah2ow(ip1,itp1,iu,ite,irh1),w00_10),wu1)),cam_mul(cam_mul(ln_ah2ow(ip1,itp1,iu,ite1,irh),w00_01),wu1)),cam_mul(cam_mul(ln_ah2ow(ip1,itp1,iu,ite1,irh1),w00_00),wu1)),cam_mul(cam_mul(ln_ah2ow(ip1,itp1,iu1,ite,irh),w00_11),wu)),cam_mul(cam_mul(ln_ah2ow(ip1,itp1,iu1,ite,irh1),w00_10),wu)),cam_mul(cam_mul(ln_ah2ow(ip1,itp1,iu1,ite1,irh),w00_01),wu)),cam_mul(cam_mul(ln_ah2ow(ip1,itp1,iu1,ite1,irh1),w00_00),wu));
// MPAS module_ra_cam.F:2820
c_star=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_mul(cam_mul(cn_ah2ow(ip,itp,iuc,ite,irh),w11_11),wuc1),cam_mul(cam_mul(cn_ah2ow(ip,itp,iuc,ite,irh1),w11_10),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip,itp,iuc,ite1,irh),w11_01),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip,itp,iuc,ite1,irh1),w11_00),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip,itp,iuc1,ite,irh),w11_11),wuc)),cam_mul(cam_mul(cn_ah2ow(ip,itp,iuc1,ite,irh1),w11_10),wuc)),cam_mul(cam_mul(cn_ah2ow(ip,itp,iuc1,ite1,irh),w11_01),wuc)),cam_mul(cam_mul(cn_ah2ow(ip,itp,iuc1,ite1,irh1),w11_00),wuc)),cam_mul(cam_mul(cn_ah2ow(ip,itp1,iuc,ite,irh),w10_11),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip,itp1,iuc,ite,irh1),w10_10),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip,itp1,iuc,ite1,irh),w10_01),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip,itp1,iuc,ite1,irh1),w10_00),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip,itp1,iuc1,ite,irh),w10_11),wuc)),cam_mul(cam_mul(cn_ah2ow(ip,itp1,iuc1,ite,irh1),w10_10),wuc)),cam_mul(cam_mul(cn_ah2ow(ip,itp1,iuc1,ite1,irh),w10_01),wuc)),cam_mul(cam_mul(cn_ah2ow(ip,itp1,iuc1,ite1,irh1),w10_00),wuc)),cam_mul(cam_mul(cn_ah2ow(ip1,itp,iuc,ite,irh),w01_11),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip1,itp,iuc,ite,irh1),w01_10),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip1,itp,iuc,ite1,irh),w01_01),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip1,itp,iuc,ite1,irh1),w01_00),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip1,itp,iuc1,ite,irh),w01_11),wuc)),cam_mul(cam_mul(cn_ah2ow(ip1,itp,iuc1,ite,irh1),w01_10),wuc)),cam_mul(cam_mul(cn_ah2ow(ip1,itp,iuc1,ite1,irh),w01_01),wuc)),cam_mul(cam_mul(cn_ah2ow(ip1,itp,iuc1,ite1,irh1),w01_00),wuc)),cam_mul(cam_mul(cn_ah2ow(ip1,itp1,iuc,ite,irh),w00_11),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip1,itp1,iuc,ite,irh1),w00_10),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip1,itp1,iuc,ite1,irh),w00_01),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip1,itp1,iuc,ite1,irh1),w00_00),wuc1)),cam_mul(cam_mul(cn_ah2ow(ip1,itp1,iuc1,ite,irh),w00_11),wuc)),cam_mul(cam_mul(cn_ah2ow(ip1,itp1,iuc1,ite,irh1),w00_10),wuc)),cam_mul(cam_mul(cn_ah2ow(ip1,itp1,iuc1,ite1,irh),w00_01),wuc)),cam_mul(cam_mul(cn_ah2ow(ip1,itp1,iuc1,ite1,irh1),w00_00),wuc));
// MPAS module_ra_cam.F:2853
abso(i,ib)=cam_min(cam_max(cam_mul(fa,(cam_sub(1.0f,cam_mul(cam_mul(l_star,c_star),aer_trn_ttl(i,k1,k2,ib))))),0.0),1.0);
// MPAS module_ra_cam.F:2859
if((uvar<min_u_h2o)) {
// MPAS module_ra_cam.F:2860
uscl=cam_div(uvar,min_u_h2o);
// MPAS module_ra_cam.F:2861
abso(i,ib)=cam_mul(abso(i,ib),uscl);
// MPAS module_ra_cam.F:2862
}
// MPAS module_ra_cam.F:2864
}
// MPAS module_ra_cam.F:2868
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2869
term7(i,1)=cam_add(coefj(1,1),cam_mul(cam_mul(coefj(2,1),dty(i)),(cam_add(1.f,cam_mul(c16,dty(i))))));
// MPAS module_ra_cam.F:2870
term8(i,1)=cam_add(coefk(1,1),cam_mul(cam_mul(coefk(2,1),dty(i)),(cam_add(1.f,cam_mul(c17,dty(i))))));
// MPAS module_ra_cam.F:2871
term7(i,2)=cam_add(coefj(1,2),cam_mul(cam_mul(coefj(2,2),dty(i)),(cam_add(1.f,cam_mul(c26,dty(i))))));
// MPAS module_ra_cam.F:2872
term8(i,2)=cam_add(coefk(1,2),cam_mul(cam_mul(coefk(2,2),dty(i)),(cam_add(1.f,cam_mul(c27,dty(i))))));
// MPAS module_ra_cam.F:2873
}
// MPAS module_ra_cam.F:2877
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2878
k21=cam_add(term7(i,1),cam_div(term8(i,1),(cam_add(1.f,cam_mul((cam_add(c30,cam_mul(cam_mul(c31,(cam_sub(dty(i),10.f))),(cam_sub(dty(i),10.f))))),sqrtu(i))))));
// MPAS module_ra_cam.F:2880
k22=cam_add(term7(i,2),cam_div(term8(i,2),(cam_add(1.f,cam_mul((cam_add(c28,cam_mul(c29,(cam_sub(dty(i),10.f))))),sqrtu(i))))));
// MPAS module_ra_cam.F:2882
tr1=cam_exp(cam_neg((cam_mul(k21,(cam_add(sqrtu(i),cam_mul(fc1,fwku(i))))))));
// MPAS module_ra_cam.F:2883
tr2=cam_exp(cam_neg((cam_mul(k22,(cam_add(sqrtu(i),cam_mul(fc1,fwku(i))))))));
// MPAS module_ra_cam.F:2884
tr1=cam_mul(tr1,aer_trn_ttl(i,k1,k2,idx_lw_0650_0800));
// MPAS module_ra_cam.F:2886
tr2=cam_mul(tr2,aer_trn_ttl(i,k1,k2,idx_lw_0500_0650));
// MPAS module_ra_cam.F:2888
tr5=cam_exp(cam_neg((cam_mul((cam_add(coefh(1,3),cam_mul(coefh(2,3),dtx(i)))),uc1(i)))));
// MPAS module_ra_cam.F:2889
tr6=cam_exp(cam_neg((cam_mul((cam_add(coefh(1,4),cam_mul(coefh(2,4),dtx(i)))),uc1(i)))));
// MPAS module_ra_cam.F:2890
tr9(i)=cam_mul(tr1,tr5);
// MPAS module_ra_cam.F:2891
tr10(i)=cam_mul(tr2,tr6);
// MPAS module_ra_cam.F:2892
th2o(i)=tr10(i);
// MPAS module_ra_cam.F:2893
trab2(i)=cam_add(cam_mul(0.65f,tr9(i)),cam_mul(0.35f,tr10(i)));
// MPAS module_ra_cam.F:2894
}
// MPAS module_ra_cam.F:2895
if((k2<k1)) {
// MPAS module_ra_cam.F:2896
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2897
to3h2o(i)=cam_div(h2otr(i,k1),h2otr(i,k2));
// MPAS module_ra_cam.F:2898
}
// MPAS module_ra_cam.F:2899
} else {
// MPAS module_ra_cam.F:2900
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2901
to3h2o(i)=cam_div(h2otr(i,k2),h2otr(i,k1));
// MPAS module_ra_cam.F:2902
}
// MPAS module_ra_cam.F:2903
}
// MPAS module_ra_cam.F:2907
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2908
dpnm(i)=cam_sub(pnm(i,k1),pnm(i,k2));
// MPAS module_ra_cam.F:2909
to3co2(i)=cam_div((cam_sub(cam_mul(pnm(i,k1),co2t(i,k1)),cam_mul(pnm(i,k2),co2t(i,k2)))),dpnm(i));
// MPAS module_ra_cam.F:2910
te=cam_pow((cam_mul(to3co2(i),r293)),.7f);
// MPAS module_ra_cam.F:2911
dplos=cam_sub(plos(i,k1),plos(i,k2));
// MPAS module_ra_cam.F:2912
dplol=cam_sub(plol(i,k1),plol(i,k2));
// MPAS module_ra_cam.F:2913
u1=cam_div(cam_mul(18.29f,cam_abs(dplos)),te);
// MPAS module_ra_cam.F:2914
u2=cam_div(cam_mul(.5649f,cam_abs(dplos)),te);
// MPAS module_ra_cam.F:2915
rphat=cam_div(dplol,dplos);
// MPAS module_ra_cam.F:2916
tlocal=tint(i,k2);
// MPAS module_ra_cam.F:2917
tcrfac=cam_mul(cam_sqrt(cam_mul(tlocal,r250)),te);
// MPAS module_ra_cam.F:2918
beta=cam_mul(r3205,(cam_add(rphat,cam_mul(dpfo3,tcrfac))));
// MPAS module_ra_cam.F:2919
realnu=cam_div(te,beta);
// MPAS module_ra_cam.F:2920
tmp1=cam_div(u1,cam_sqrt(cam_add(4.f,cam_mul(u1,(cam_add(1.f,realnu))))));
// MPAS module_ra_cam.F:2921
tmp2=cam_div(u2,cam_sqrt(cam_add(4.f,cam_mul(u2,(cam_add(1.f,realnu))))));
// MPAS module_ra_cam.F:2922
o3bndi=cam_mul(cam_mul(74.f,te),cam_log(cam_add(cam_add(1.f,tmp1),tmp2)));
// MPAS module_ra_cam.F:2923
abso(i,3)=cam_mul(cam_mul(o3bndi,to3h2o(i)),dbvtit(i,k2));
// MPAS module_ra_cam.F:2924
to3(i)=cam_div(1.0f,(cam_add(cam_add(1.f,cam_mul(0.1f,tmp1)),cam_mul(0.1f,tmp2))));
// MPAS module_ra_cam.F:2925
}
// MPAS module_ra_cam.F:2929
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2930
sqwp=cam_sqrt(cam_abs(cam_sub(plco2(i,k1),plco2(i,k2))));
// MPAS module_ra_cam.F:2931
et=cam_exp(cam_neg(cam_div(480.f,to3co2(i))));
// MPAS module_ra_cam.F:2932
sqti(i)=cam_sqrt(to3co2(i));
// MPAS module_ra_cam.F:2933
rsqti=cam_div(1.f,sqti(i));
// MPAS module_ra_cam.F:2934
et2=cam_mul(et,et);
// MPAS module_ra_cam.F:2935
et4=cam_mul(et2,et2);
// MPAS module_ra_cam.F:2936
omet=cam_sub(1.f,cam_mul(1.5f,et2));
// MPAS module_ra_cam.F:2937
f1co2=cam_mul(cam_mul(cam_mul(899.70f,omet),(cam_add(cam_add(1.f,cam_mul(1.94774f,et)),cam_mul(4.73486f,et2)))),rsqti);
// MPAS module_ra_cam.F:2938
f1sqwp(i)=cam_mul(f1co2,sqwp);
// MPAS module_ra_cam.F:2939
t1co2(i)=cam_div(1.f,(cam_add(1.f,(cam_mul(cam_mul(cam_mul(245.18f,omet),sqwp),rsqti)))));
// MPAS module_ra_cam.F:2940
oneme=cam_sub(1.f,et2);
// MPAS module_ra_cam.F:2941
alphat=cam_mul(cam_pow(oneme,3),rsqti);
// MPAS module_ra_cam.F:2942
pi=cam_abs(dpnm(i));
// MPAS module_ra_cam.F:2943
wco2=cam_mul(cam_mul(cam_mul(2.5221f,co2vmr),pi),rga);
// MPAS module_ra_cam.F:2944
u7(i)=cam_mul(cam_mul(cam_mul(4.9411e4f,alphat),et2),wco2);
// MPAS module_ra_cam.F:2945
u8=cam_mul(cam_mul(cam_mul(3.9744e4f,alphat),et4),wco2);
// MPAS module_ra_cam.F:2946
u9=cam_mul(cam_mul(cam_mul(cam_mul(1.0447e5f,alphat),et4),et2),wco2);
// MPAS module_ra_cam.F:2947
u13=cam_mul(cam_mul(cam_mul(2.8388e3f,alphat),et4),wco2);
// MPAS module_ra_cam.F:2948
tpath=to3co2(i);
// MPAS module_ra_cam.F:2949
tlocal=tint(i,k2);
// MPAS module_ra_cam.F:2950
tcrfac=cam_sqrt(cam_mul(cam_mul(cam_mul(tlocal,r250),tpath),r300));
// MPAS module_ra_cam.F:2951
posqt=cam_mul((cam_add(cam_mul((cam_add(pnm(i,k2),pnm(i,k1))),r2sslp),cam_mul(dpfco2,tcrfac))),rsqti);
// MPAS module_ra_cam.F:2952
rbeta7(i)=cam_div(1.f,(cam_mul(5.3228f,posqt)));
// MPAS module_ra_cam.F:2953
rbeta8=cam_div(1.f,(cam_mul(10.6576f,posqt)));
// MPAS module_ra_cam.F:2954
rbeta9=rbeta7(i);
// MPAS module_ra_cam.F:2955
rbeta13=rbeta9;
// MPAS module_ra_cam.F:2956
f2co2(i)=cam_add(cam_add((cam_div(u7(i),cam_sqrt(cam_add(4.f,cam_mul(u7(i),(cam_add(1.f,rbeta7(i)))))))),(cam_div(u8,cam_sqrt(cam_add(4.f,cam_mul(u8,(cam_add(1.f,rbeta8)))))))),(cam_div(u9,cam_sqrt(cam_add(4.f,cam_mul(u9,(cam_add(1.f,rbeta9))))))));
// MPAS module_ra_cam.F:2959
f3co2(i)=cam_div(u13,cam_sqrt(cam_add(4.f,cam_mul(u13,(cam_add(1.f,rbeta13))))));
// MPAS module_ra_cam.F:2960
}
// MPAS module_ra_cam.F:2961
if((k2>=k1)) {
// MPAS module_ra_cam.F:2962
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2963
sqti(i)=cam_sqrt(tlayr(i,k2));
// MPAS module_ra_cam.F:2964
}
// MPAS module_ra_cam.F:2965
}
// MPAS module_ra_cam.F:2967
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2968
tmp1=cam_log(cam_add(1.f,f1sqwp(i)));
// MPAS module_ra_cam.F:2969
tmp2=cam_log(cam_add(1.f,f2co2(i)));
// MPAS module_ra_cam.F:2970
tmp3=cam_log(cam_add(1.f,f3co2(i)));
// MPAS module_ra_cam.F:2971
absbnd=cam_mul((cam_add(cam_add(tmp1,cam_mul(cam_mul(2.f,t1co2(i)),tmp2)),cam_mul(2.f,tmp3))),sqti(i));
// MPAS module_ra_cam.F:2972
abso(i,4)=cam_mul(cam_mul(trab2(i),co2em(i,k2)),absbnd);
// MPAS module_ra_cam.F:2973
tco2(i)=cam_div(1.f,(cam_add(1.0f,cam_mul(10.0f,(cam_div(u7(i),cam_sqrt(cam_add(4.f,cam_mul(u7(i),(cam_add(1.f,rbeta7(i))))))))))));
// MPAS module_ra_cam.F:2974
}
// MPAS module_ra_cam.F:2978
trcab(lchnk,ncol,pcols,pverp,k1,k2,ucfc11,ucfc12,un2o0,un2o1,uch4,uco211,uco212,uco213,uco221,uco222,uco223,bn2o0,bn2o1,bch4,to3co2,pnm,dw,pnew,s2c,uptype,u,abplnk1,tco2,th2o,to3,abstrc,aer_trn_ttl); if(error) return;
// MPAS module_ra_cam.F:2989
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:2990
abstot(i,k1,k2)=cam_add(cam_add(cam_add(cam_add(abso(i,1),abso(i,2)),abso(i,3)),abso(i,4)),abstrc(i));
// MPAS module_ra_cam.F:2992
}
// MPAS module_ra_cam.F:2993
}
// MPAS module_ra_cam.F:2994
}
// MPAS module_ra_cam.F:3013
for(k2=pver;k2>=ntoplw;k2+=cam_neg(1)) {
// MPAS module_ra_cam.F:3014
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3015
tbar(i,1)=cam_mul(0.5f,(cam_add(tint(i,cam_add(k2,1)),tlayr(i,cam_add(k2,1)))));
// MPAS module_ra_cam.F:3016
emm(i,1)=cam_mul(0.5f,(cam_add(co2em(i,cam_add(k2,1)),co2eml(i,k2))));
// MPAS module_ra_cam.F:3017
tbar(i,2)=cam_mul(0.5f,(cam_add(tlayr(i,cam_add(k2,1)),tint(i,k2))));
// MPAS module_ra_cam.F:3018
emm(i,2)=cam_mul(0.5f,(cam_add(co2em(i,k2),co2eml(i,k2))));
// MPAS module_ra_cam.F:3019
tbar(i,3)=cam_mul(0.5f,(cam_add(tbar(i,2),tbar(i,1))));
// MPAS module_ra_cam.F:3020
emm(i,3)=emm(i,1);
// MPAS module_ra_cam.F:3021
tbar(i,4)=tbar(i,3);
// MPAS module_ra_cam.F:3022
emm(i,4)=emm(i,2);
// MPAS module_ra_cam.F:3023
o3emm(i,1)=cam_mul(0.5f,(cam_add(dbvtit(i,cam_add(k2,1)),dbvtly(i,k2))));
// MPAS module_ra_cam.F:3024
o3emm(i,2)=cam_mul(0.5f,(cam_add(dbvtit(i,k2),dbvtly(i,k2))));
// MPAS module_ra_cam.F:3025
o3emm(i,3)=o3emm(i,1);
// MPAS module_ra_cam.F:3026
o3emm(i,4)=o3emm(i,2);
// MPAS module_ra_cam.F:3027
temh2o(i,1)=tbar(i,1);
// MPAS module_ra_cam.F:3028
temh2o(i,2)=tbar(i,2);
// MPAS module_ra_cam.F:3029
temh2o(i,3)=tbar(i,1);
// MPAS module_ra_cam.F:3030
temh2o(i,4)=tbar(i,2);
// MPAS module_ra_cam.F:3031
dpnm(i)=cam_sub(pnm(i,cam_add(k2,1)),pnm(i,k2));
// MPAS module_ra_cam.F:3032
}
// MPAS module_ra_cam.F:3036
for(wvl=1;wvl<=14;wvl+=1) {
// MPAS module_ra_cam.F:3037
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3038
bplnk(wvl,i,1)=cam_mul(0.5f,(cam_add(abplnk1(wvl,i,cam_add(k2,1)),abplnk2(wvl,i,k2))));
// MPAS module_ra_cam.F:3039
bplnk(wvl,i,2)=cam_mul(0.5f,(cam_add(abplnk1(wvl,i,k2),abplnk2(wvl,i,k2))));
// MPAS module_ra_cam.F:3040
bplnk(wvl,i,3)=bplnk(wvl,i,1);
// MPAS module_ra_cam.F:3041
bplnk(wvl,i,4)=bplnk(wvl,i,2);
// MPAS module_ra_cam.F:3042
}
// MPAS module_ra_cam.F:3043
}
// MPAS module_ra_cam.F:3045
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3046
rdpnmsq=cam_div(1.f,(cam_sub(pnmsq(i,cam_add(k2,1)),pnmsq(i,k2))));
// MPAS module_ra_cam.F:3047
rdpnm=cam_div(1.f,dpnm(i));
// MPAS module_ra_cam.F:3048
p1=cam_mul(.5f,(cam_add(pbr(i,k2),pnm(i,cam_add(k2,1)))));
// MPAS module_ra_cam.F:3049
p2=cam_mul(.5f,(cam_add(pbr(i,k2),pnm(i,k2))));
// MPAS module_ra_cam.F:3050
uinpl(i,1)=cam_mul((cam_sub(pnmsq(i,cam_add(k2,1)),cam_pow(p1,2))),rdpnmsq);
// MPAS module_ra_cam.F:3051
uinpl(i,2)=cam_neg(cam_mul((cam_sub(pnmsq(i,k2),cam_pow(p2,2))),rdpnmsq));
// MPAS module_ra_cam.F:3052
uinpl(i,3)=cam_neg(cam_mul((cam_sub(pnmsq(i,k2),cam_pow(p1,2))),rdpnmsq));
// MPAS module_ra_cam.F:3053
uinpl(i,4)=cam_mul((cam_sub(pnmsq(i,cam_add(k2,1)),cam_pow(p2,2))),rdpnmsq);
// MPAS module_ra_cam.F:3054
winpl(i,1)=cam_mul((cam_mul(.5f,(cam_sub(pnm(i,cam_add(k2,1)),pbr(i,k2))))),rdpnm);
// MPAS module_ra_cam.F:3055
winpl(i,2)=cam_mul((cam_mul(.5f,(cam_add(cam_neg(pnm(i,k2)),pbr(i,k2))))),rdpnm);
// MPAS module_ra_cam.F:3056
winpl(i,3)=cam_mul((cam_sub(cam_mul(.5f,(cam_add(pnm(i,cam_add(k2,1)),pbr(i,k2)))),pnm(i,k2))),rdpnm);
// MPAS module_ra_cam.F:3057
winpl(i,4)=cam_mul((cam_add(cam_mul(.5f,(cam_sub(cam_neg(pnm(i,k2)),pbr(i,k2)))),pnm(i,cam_add(k2,1)))),rdpnm);
// MPAS module_ra_cam.F:3058
tmp1=cam_div(1.f,(cam_sub(piln(i,cam_add(k2,1)),piln(i,k2))));
// MPAS module_ra_cam.F:3059
tmp2=cam_sub(piln(i,cam_add(k2,1)),pmln(i,k2));
// MPAS module_ra_cam.F:3060
tmp3=cam_sub(piln(i,k2),pmln(i,k2));
// MPAS module_ra_cam.F:3061
zinpl(i,1)=cam_mul((cam_mul(.5f,tmp2)),tmp1);
// MPAS module_ra_cam.F:3062
zinpl(i,2)=cam_mul((cam_neg(cam_mul(.5f,tmp3))),tmp1);
// MPAS module_ra_cam.F:3063
zinpl(i,3)=cam_mul((cam_sub(cam_mul(.5f,tmp2),tmp3)),tmp1);
// MPAS module_ra_cam.F:3064
zinpl(i,4)=cam_mul((cam_sub(tmp2,cam_mul(.5f,tmp3))),tmp1);
// MPAS module_ra_cam.F:3065
pinpl(i,1)=cam_mul(0.5f,(cam_add(p1,pnm(i,cam_add(k2,1)))));
// MPAS module_ra_cam.F:3066
pinpl(i,2)=cam_mul(0.5f,(cam_add(p2,pnm(i,k2))));
// MPAS module_ra_cam.F:3067
pinpl(i,3)=cam_mul(0.5f,(cam_add(p1,pnm(i,k2))));
// MPAS module_ra_cam.F:3068
pinpl(i,4)=cam_mul(0.5f,(cam_add(p2,pnm(i,cam_add(k2,1)))));
// MPAS module_ra_cam.F:3069
if(strat_volcanic) {
// MPAS module_ra_cam.F:3070
aer_pth_ngh(i)=cam_abs(cam_sub(aer_mpp(i,k2),aer_mpp(i,cam_add(k2,1))));
// MPAS module_ra_cam.F:3071
}
// MPAS module_ra_cam.F:3072
}
// MPAS module_ra_cam.F:3073
for(kn=1;kn<=4;kn+=1) {
// MPAS module_ra_cam.F:3074
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3075
u(i)=cam_mul(uinpl(i,kn),cam_abs(cam_sub(plh2o(i,k2),plh2o(i,cam_add(k2,1)))));
// MPAS module_ra_cam.F:3076
sqrtu(i)=cam_sqrt(u(i));
// MPAS module_ra_cam.F:3077
dw(i)=cam_abs(cam_sub(w(i,k2),w(i,cam_add(k2,1))));
// MPAS module_ra_cam.F:3078
pnew(i)=cam_div(u(i),(cam_mul(winpl(i,kn),dw(i))));
// MPAS module_ra_cam.F:3079
pnew_mks=cam_mul(pnew(i),sslp_mks);
// MPAS module_ra_cam.F:3080
t_p=cam_min(cam_max(tbar(i,kn),min_tp_h2o),max_tp_h2o);
// MPAS module_ra_cam.F:3081
iest=cam_sub(floor(t_p),min_tp_h2o);
// MPAS module_ra_cam.F:3082
esx=cam_add(estblh2o(iest),cam_mul((cam_sub(estblh2o(cam_add(iest,1)),estblh2o(iest))),(cam_sub(cam_sub(t_p,min_tp_h2o),iest))));
// MPAS module_ra_cam.F:3084
qsx=cam_div(cam_mul(epsilo,esx),(cam_sub(pnew_mks,cam_mul(omeps,esx))));
// MPAS module_ra_cam.F:3085
q_path=cam_div(cam_div(dw(i),cam_abs(dpnm(i))),rga);
// MPAS module_ra_cam.F:3087
ds2c=cam_abs(cam_sub(s2c(i,k2),s2c(i,cam_add(k2,1))));
// MPAS module_ra_cam.F:3088
uc1(i)=cam_mul(uinpl(i,kn),ds2c);
// MPAS module_ra_cam.F:3089
pch2o=uc1(i);
// MPAS module_ra_cam.F:3090
uc1(i)=cam_div(cam_mul((cam_add(uc1(i),cam_mul(1.7e-3f,u(i)))),(cam_add(1.f,cam_mul(2.f,uc1(i))))),(cam_add(1.f,cam_mul(15.f,uc1(i)))));
// MPAS module_ra_cam.F:3091
dtx(i)=cam_sub(temh2o(i,kn),250.f);
// MPAS module_ra_cam.F:3092
dty(i)=cam_sub(tbar(i,kn),250.f);
// MPAS module_ra_cam.F:3094
fwk(i)=cam_add(fwcoef,cam_div(fwc1,(cam_add(1.f,cam_mul(fwc2,u(i))))));
// MPAS module_ra_cam.F:3095
fwku(i)=cam_mul(fwk(i),u(i));
// MPAS module_ra_cam.F:3097
if(strat_volcanic) {
// MPAS module_ra_cam.F:3098
aer_pth_dlt=cam_mul(uinpl(i,kn),aer_pth_ngh(i));
// MPAS module_ra_cam.F:3100
for(bnd_idx=1;bnd_idx<=bnd_nbr_lw;bnd_idx+=1) {
// MPAS module_ra_cam.F:3101
odap_aer_ttl=cam_mul(abs_cff_mss_aer(bnd_idx),aer_pth_dlt);
// MPAS module_ra_cam.F:3102
aer_trn_ngh(i,bnd_idx)=cam_exp(cam_neg(cam_mul(fdif,odap_aer_ttl)));
// MPAS module_ra_cam.F:3103
}
// MPAS module_ra_cam.F:3104
} else {
// MPAS module_ra_cam.F:3105
{ auto _lhs=aer_trn_ngh.section(i,cam_section(1,bnd_nbr_lw)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=1.0f; }
// MPAS module_ra_cam.F:3106
}
// MPAS module_ra_cam.F:3130
te1=temh2o(i,kn);
// MPAS module_ra_cam.F:3131
te2=cam_mul(te1,te1);
// MPAS module_ra_cam.F:3132
te3=cam_mul(te2,te1);
// MPAS module_ra_cam.F:3133
te4=cam_mul(te3,te1);
// MPAS module_ra_cam.F:3134
te5=cam_mul(te4,te1);
// MPAS module_ra_cam.F:3142
uvar=cam_mul(u(i),fdif);
// MPAS module_ra_cam.F:3143
log_u=cam_min(cam_log10(cam_max(uvar,min_u_h2o)),max_lu_h2o);
// MPAS module_ra_cam.F:3144
dvar=cam_div((cam_sub(log_u,min_lu_h2o)),dlu_h2o);
// MPAS module_ra_cam.F:3145
iu=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_u,1));
// MPAS module_ra_cam.F:3146
iu1=cam_add(iu,1);
// MPAS module_ra_cam.F:3147
wu=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:3148
wu1=cam_sub(1.0f,wu);
// MPAS module_ra_cam.F:3150
log_p=cam_min(cam_log10(cam_max(pnew(i),min_p_h2o)),max_lp_h2o);
// MPAS module_ra_cam.F:3151
dvar=cam_div((cam_sub(log_p,min_lp_h2o)),dlp_h2o);
// MPAS module_ra_cam.F:3152
ip=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_p,1));
// MPAS module_ra_cam.F:3153
ip1=cam_add(ip,1);
// MPAS module_ra_cam.F:3154
wp=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:3155
wp1=cam_sub(1.0f,wp);
// MPAS module_ra_cam.F:3157
dvar=cam_div((cam_sub(t_p,min_tp_h2o)),dtp_h2o);
// MPAS module_ra_cam.F:3158
itp=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_tp,1));
// MPAS module_ra_cam.F:3159
itp1=cam_add(itp,1);
// MPAS module_ra_cam.F:3160
wtp=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:3161
wtp1=cam_sub(1.0f,wtp);
// MPAS module_ra_cam.F:3163
t_e=cam_min(cam_max(cam_sub(temh2o(i,kn),t_p),min_te_h2o),max_te_h2o);
// MPAS module_ra_cam.F:3164
dvar=cam_div((cam_sub(t_e,min_te_h2o)),dte_h2o);
// MPAS module_ra_cam.F:3165
ite=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_te,1));
// MPAS module_ra_cam.F:3166
ite1=cam_add(ite,1);
// MPAS module_ra_cam.F:3167
wte=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:3168
wte1=cam_sub(1.0f,wte);
// MPAS module_ra_cam.F:3170
rh_path=cam_min(cam_max(cam_div(q_path,qsx),min_rh_h2o),max_rh_h2o);
// MPAS module_ra_cam.F:3171
dvar=cam_div((cam_sub(rh_path,min_rh_h2o)),drh_h2o);
// MPAS module_ra_cam.F:3172
irh=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_rh,1));
// MPAS module_ra_cam.F:3173
irh1=cam_add(irh,1);
// MPAS module_ra_cam.F:3174
wrh=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:3175
wrh1=cam_sub(1.0f,wrh);
// MPAS module_ra_cam.F:3177
w_0_0_=cam_mul(wtp,wte);
// MPAS module_ra_cam.F:3178
w_0_1_=cam_mul(wtp,wte1);
// MPAS module_ra_cam.F:3179
w_1_0_=cam_mul(wtp1,wte);
// MPAS module_ra_cam.F:3180
w_1_1_=cam_mul(wtp1,wte1);
// MPAS module_ra_cam.F:3182
w_0_00=cam_mul(w_0_0_,wrh);
// MPAS module_ra_cam.F:3183
w_0_01=cam_mul(w_0_0_,wrh1);
// MPAS module_ra_cam.F:3184
w_0_10=cam_mul(w_0_1_,wrh);
// MPAS module_ra_cam.F:3185
w_0_11=cam_mul(w_0_1_,wrh1);
// MPAS module_ra_cam.F:3186
w_1_00=cam_mul(w_1_0_,wrh);
// MPAS module_ra_cam.F:3187
w_1_01=cam_mul(w_1_0_,wrh1);
// MPAS module_ra_cam.F:3188
w_1_10=cam_mul(w_1_1_,wrh);
// MPAS module_ra_cam.F:3189
w_1_11=cam_mul(w_1_1_,wrh1);
// MPAS module_ra_cam.F:3191
w00_00=cam_mul(wp,w_0_00);
// MPAS module_ra_cam.F:3192
w00_01=cam_mul(wp,w_0_01);
// MPAS module_ra_cam.F:3193
w00_10=cam_mul(wp,w_0_10);
// MPAS module_ra_cam.F:3194
w00_11=cam_mul(wp,w_0_11);
// MPAS module_ra_cam.F:3195
w01_00=cam_mul(wp,w_1_00);
// MPAS module_ra_cam.F:3196
w01_01=cam_mul(wp,w_1_01);
// MPAS module_ra_cam.F:3197
w01_10=cam_mul(wp,w_1_10);
// MPAS module_ra_cam.F:3198
w01_11=cam_mul(wp,w_1_11);
// MPAS module_ra_cam.F:3199
w10_00=cam_mul(wp1,w_0_00);
// MPAS module_ra_cam.F:3200
w10_01=cam_mul(wp1,w_0_01);
// MPAS module_ra_cam.F:3201
w10_10=cam_mul(wp1,w_0_10);
// MPAS module_ra_cam.F:3202
w10_11=cam_mul(wp1,w_0_11);
// MPAS module_ra_cam.F:3203
w11_00=cam_mul(wp1,w_1_00);
// MPAS module_ra_cam.F:3204
w11_01=cam_mul(wp1,w_1_01);
// MPAS module_ra_cam.F:3205
w11_10=cam_mul(wp1,w_1_10);
// MPAS module_ra_cam.F:3206
w11_11=cam_mul(wp1,w_1_11);
// MPAS module_ra_cam.F:3211
ib=1;
// MPAS module_ra_cam.F:3213
fa=cam_add(cam_add(cam_add(cam_add(cam_add(fat(1,ib),cam_mul(fat(2,ib),te1)),cam_mul(fat(3,ib),te2)),cam_mul(fat(4,ib),te3)),cam_mul(fat(5,ib),te4)),cam_mul(fat(6,ib),te5));
// MPAS module_ra_cam.F:3220
a_star=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_mul(cam_mul(ah2onw(ip,itp,iu,ite,irh),w11_11),wu1),cam_mul(cam_mul(ah2onw(ip,itp,iu,ite,irh1),w11_10),wu1)),cam_mul(cam_mul(ah2onw(ip,itp,iu,ite1,irh),w11_01),wu1)),cam_mul(cam_mul(ah2onw(ip,itp,iu,ite1,irh1),w11_00),wu1)),cam_mul(cam_mul(ah2onw(ip,itp,iu1,ite,irh),w11_11),wu)),cam_mul(cam_mul(ah2onw(ip,itp,iu1,ite,irh1),w11_10),wu)),cam_mul(cam_mul(ah2onw(ip,itp,iu1,ite1,irh),w11_01),wu)),cam_mul(cam_mul(ah2onw(ip,itp,iu1,ite1,irh1),w11_00),wu)),cam_mul(cam_mul(ah2onw(ip,itp1,iu,ite,irh),w10_11),wu1)),cam_mul(cam_mul(ah2onw(ip,itp1,iu,ite,irh1),w10_10),wu1)),cam_mul(cam_mul(ah2onw(ip,itp1,iu,ite1,irh),w10_01),wu1)),cam_mul(cam_mul(ah2onw(ip,itp1,iu,ite1,irh1),w10_00),wu1)),cam_mul(cam_mul(ah2onw(ip,itp1,iu1,ite,irh),w10_11),wu)),cam_mul(cam_mul(ah2onw(ip,itp1,iu1,ite,irh1),w10_10),wu)),cam_mul(cam_mul(ah2onw(ip,itp1,iu1,ite1,irh),w10_01),wu)),cam_mul(cam_mul(ah2onw(ip,itp1,iu1,ite1,irh1),w10_00),wu)),cam_mul(cam_mul(ah2onw(ip1,itp,iu,ite,irh),w01_11),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp,iu,ite,irh1),w01_10),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp,iu,ite1,irh),w01_01),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp,iu,ite1,irh1),w01_00),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp,iu1,ite,irh),w01_11),wu)),cam_mul(cam_mul(ah2onw(ip1,itp,iu1,ite,irh1),w01_10),wu)),cam_mul(cam_mul(ah2onw(ip1,itp,iu1,ite1,irh),w01_01),wu)),cam_mul(cam_mul(ah2onw(ip1,itp,iu1,ite1,irh1),w01_00),wu)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu,ite,irh),w00_11),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu,ite,irh1),w00_10),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu,ite1,irh),w00_01),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu,ite1,irh1),w00_00),wu1)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu1,ite,irh),w00_11),wu)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu1,ite,irh1),w00_10),wu)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu1,ite1,irh),w00_01),wu)),cam_mul(cam_mul(ah2onw(ip1,itp1,iu1,ite1,irh1),w00_00),wu));
// MPAS module_ra_cam.F:3254
abso(i,ib)=cam_min(cam_max(cam_mul(fa,(cam_sub(1.0f,cam_mul((cam_sub(1.0f,a_star)),aer_trn_ngh(i,ib))))),0.0),1.0);
// MPAS module_ra_cam.F:3261
if((uvar<min_u_h2o)) {
// MPAS module_ra_cam.F:3262
uscl=cam_div(uvar,min_u_h2o);
// MPAS module_ra_cam.F:3263
abso(i,ib)=cam_mul(abso(i,ib),uscl);
// MPAS module_ra_cam.F:3264
}
// MPAS module_ra_cam.F:3269
ib=2;
// MPAS module_ra_cam.F:3271
fa=cam_add(cam_add(cam_add(cam_add(cam_add(fat(1,ib),cam_mul(fat(2,ib),te1)),cam_mul(fat(3,ib),te2)),cam_mul(fat(4,ib),te3)),cam_mul(fat(5,ib),te4)),cam_mul(fat(6,ib),te5));
// MPAS module_ra_cam.F:3278
a_star=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_mul(cam_mul(ah2ow(ip,itp,iu,ite,irh),w11_11),wu1),cam_mul(cam_mul(ah2ow(ip,itp,iu,ite,irh1),w11_10),wu1)),cam_mul(cam_mul(ah2ow(ip,itp,iu,ite1,irh),w11_01),wu1)),cam_mul(cam_mul(ah2ow(ip,itp,iu,ite1,irh1),w11_00),wu1)),cam_mul(cam_mul(ah2ow(ip,itp,iu1,ite,irh),w11_11),wu)),cam_mul(cam_mul(ah2ow(ip,itp,iu1,ite,irh1),w11_10),wu)),cam_mul(cam_mul(ah2ow(ip,itp,iu1,ite1,irh),w11_01),wu)),cam_mul(cam_mul(ah2ow(ip,itp,iu1,ite1,irh1),w11_00),wu)),cam_mul(cam_mul(ah2ow(ip,itp1,iu,ite,irh),w10_11),wu1)),cam_mul(cam_mul(ah2ow(ip,itp1,iu,ite,irh1),w10_10),wu1)),cam_mul(cam_mul(ah2ow(ip,itp1,iu,ite1,irh),w10_01),wu1)),cam_mul(cam_mul(ah2ow(ip,itp1,iu,ite1,irh1),w10_00),wu1)),cam_mul(cam_mul(ah2ow(ip,itp1,iu1,ite,irh),w10_11),wu)),cam_mul(cam_mul(ah2ow(ip,itp1,iu1,ite,irh1),w10_10),wu)),cam_mul(cam_mul(ah2ow(ip,itp1,iu1,ite1,irh),w10_01),wu)),cam_mul(cam_mul(ah2ow(ip,itp1,iu1,ite1,irh1),w10_00),wu)),cam_mul(cam_mul(ah2ow(ip1,itp,iu,ite,irh),w01_11),wu1)),cam_mul(cam_mul(ah2ow(ip1,itp,iu,ite,irh1),w01_10),wu1)),cam_mul(cam_mul(ah2ow(ip1,itp,iu,ite1,irh),w01_01),wu1)),cam_mul(cam_mul(ah2ow(ip1,itp,iu,ite1,irh1),w01_00),wu1)),cam_mul(cam_mul(ah2ow(ip1,itp,iu1,ite,irh),w01_11),wu)),cam_mul(cam_mul(ah2ow(ip1,itp,iu1,ite,irh1),w01_10),wu)),cam_mul(cam_mul(ah2ow(ip1,itp,iu1,ite1,irh),w01_01),wu)),cam_mul(cam_mul(ah2ow(ip1,itp,iu1,ite1,irh1),w01_00),wu)),cam_mul(cam_mul(ah2ow(ip1,itp1,iu,ite,irh),w00_11),wu1)),cam_mul(cam_mul(ah2ow(ip1,itp1,iu,ite,irh1),w00_10),wu1)),cam_mul(cam_mul(ah2ow(ip1,itp1,iu,ite1,irh),w00_01),wu1)),cam_mul(cam_mul(ah2ow(ip1,itp1,iu,ite1,irh1),w00_00),wu1)),cam_mul(cam_mul(ah2ow(ip1,itp1,iu1,ite,irh),w00_11),wu)),cam_mul(cam_mul(ah2ow(ip1,itp1,iu1,ite,irh1),w00_10),wu)),cam_mul(cam_mul(ah2ow(ip1,itp1,iu1,ite1,irh),w00_01),wu)),cam_mul(cam_mul(ah2ow(ip1,itp1,iu1,ite1,irh1),w00_00),wu));
// MPAS module_ra_cam.F:3312
abso(i,ib)=cam_min(cam_max(cam_mul(fa,(cam_sub(1.0f,cam_mul((cam_sub(1.0f,a_star)),aer_trn_ngh(i,ib))))),0.0),1.0);
// MPAS module_ra_cam.F:3319
if((uvar<min_u_h2o)) {
// MPAS module_ra_cam.F:3320
uscl=cam_div(uvar,min_u_h2o);
// MPAS module_ra_cam.F:3321
abso(i,ib)=cam_mul(abso(i,ib),uscl);
// MPAS module_ra_cam.F:3322
}
// MPAS module_ra_cam.F:3324
}
// MPAS module_ra_cam.F:3328
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3329
term7(i,1)=cam_add(coefj(1,1),cam_mul(cam_mul(coefj(2,1),dty(i)),(cam_add(1.f,cam_mul(c16,dty(i))))));
// MPAS module_ra_cam.F:3330
term8(i,1)=cam_add(coefk(1,1),cam_mul(cam_mul(coefk(2,1),dty(i)),(cam_add(1.f,cam_mul(c17,dty(i))))));
// MPAS module_ra_cam.F:3331
term7(i,2)=cam_add(coefj(1,2),cam_mul(cam_mul(coefj(2,2),dty(i)),(cam_add(1.f,cam_mul(c26,dty(i))))));
// MPAS module_ra_cam.F:3332
term8(i,2)=cam_add(coefk(1,2),cam_mul(cam_mul(coefk(2,2),dty(i)),(cam_add(1.f,cam_mul(c27,dty(i))))));
// MPAS module_ra_cam.F:3333
}
// MPAS module_ra_cam.F:3337
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3338
dtym10=cam_sub(dty(i),10.f);
// MPAS module_ra_cam.F:3339
denom=cam_add(1.f,cam_mul((cam_add(c30,cam_mul(cam_mul(c31,dtym10),dtym10))),sqrtu(i)));
// MPAS module_ra_cam.F:3340
k21=cam_add(term7(i,1),cam_div(term8(i,1),denom));
// MPAS module_ra_cam.F:3341
denom=cam_add(1.f,cam_mul((cam_add(c28,cam_mul(c29,dtym10))),sqrtu(i)));
// MPAS module_ra_cam.F:3342
k22=cam_add(term7(i,2),cam_div(term8(i,2),denom));
// MPAS module_ra_cam.F:3343
tr1=cam_exp(cam_neg((cam_mul(k21,(cam_add(sqrtu(i),cam_mul(fc1,fwku(i))))))));
// MPAS module_ra_cam.F:3344
tr2=cam_exp(cam_neg((cam_mul(k22,(cam_add(sqrtu(i),cam_mul(fc1,fwku(i))))))));
// MPAS module_ra_cam.F:3345
tr1=cam_mul(tr1,aer_trn_ngh(i,idx_lw_0650_0800));
// MPAS module_ra_cam.F:3347
tr2=cam_mul(tr2,aer_trn_ngh(i,idx_lw_0500_0650));
// MPAS module_ra_cam.F:3349
tr5=cam_exp(cam_neg((cam_mul((cam_add(coefh(1,3),cam_mul(coefh(2,3),dtx(i)))),uc1(i)))));
// MPAS module_ra_cam.F:3350
tr6=cam_exp(cam_neg((cam_mul((cam_add(coefh(1,4),cam_mul(coefh(2,4),dtx(i)))),uc1(i)))));
// MPAS module_ra_cam.F:3351
tr9(i)=cam_mul(tr1,tr5);
// MPAS module_ra_cam.F:3352
tr10(i)=cam_mul(tr2,tr6);
// MPAS module_ra_cam.F:3353
trab2(i)=cam_add(cam_mul(0.65f,tr9(i)),cam_mul(0.35f,tr10(i)));
// MPAS module_ra_cam.F:3354
th2o(i)=tr10(i);
// MPAS module_ra_cam.F:3355
}
// MPAS module_ra_cam.F:3359
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3360
te=cam_pow((cam_mul(tbar(i,kn),r293)),.7f);
// MPAS module_ra_cam.F:3361
dplos=cam_abs(cam_sub(plos(i,cam_add(k2,1)),plos(i,k2)));
// MPAS module_ra_cam.F:3362
u1=cam_div(cam_mul(cam_mul(zinpl(i,kn),18.29f),dplos),te);
// MPAS module_ra_cam.F:3363
u2=cam_div(cam_mul(cam_mul(zinpl(i,kn),.5649f),dplos),te);
// MPAS module_ra_cam.F:3364
tlocal=tbar(i,kn);
// MPAS module_ra_cam.F:3365
tcrfac=cam_mul(cam_sqrt(cam_mul(tlocal,r250)),te);
// MPAS module_ra_cam.F:3366
beta=cam_mul(r3205,(cam_add(cam_mul(pinpl(i,kn),rsslp),cam_mul(dpfo3,tcrfac))));
// MPAS module_ra_cam.F:3367
realnu=cam_div(te,beta);
// MPAS module_ra_cam.F:3368
tmp1=cam_div(u1,cam_sqrt(cam_add(4.f,cam_mul(u1,(cam_add(1.f,realnu))))));
// MPAS module_ra_cam.F:3369
tmp2=cam_div(u2,cam_sqrt(cam_add(4.f,cam_mul(u2,(cam_add(1.f,realnu))))));
// MPAS module_ra_cam.F:3370
o3bndi=cam_mul(cam_mul(74.f,te),cam_log(cam_add(cam_add(1.f,tmp1),tmp2)));
// MPAS module_ra_cam.F:3371
abso(i,3)=cam_mul(cam_mul(o3bndi,o3emm(i,kn)),(cam_div(h2otr(i,cam_add(k2,1)),h2otr(i,k2))));
// MPAS module_ra_cam.F:3372
to3(i)=cam_div(1.0f,(cam_add(cam_add(1.f,cam_mul(0.1f,tmp1)),cam_mul(0.1f,tmp2))));
// MPAS module_ra_cam.F:3373
}
// MPAS module_ra_cam.F:3377
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3378
dplco2=cam_sub(plco2(i,cam_add(k2,1)),plco2(i,k2));
// MPAS module_ra_cam.F:3379
sqwp=cam_sqrt(cam_mul(uinpl(i,kn),dplco2));
// MPAS module_ra_cam.F:3380
et=cam_exp(cam_neg(cam_div(480.f,tbar(i,kn))));
// MPAS module_ra_cam.F:3381
sqti(i)=cam_sqrt(tbar(i,kn));
// MPAS module_ra_cam.F:3382
rsqti=cam_div(1.f,sqti(i));
// MPAS module_ra_cam.F:3383
et2=cam_mul(et,et);
// MPAS module_ra_cam.F:3384
et4=cam_mul(et2,et2);
// MPAS module_ra_cam.F:3385
omet=(cam_sub(1.f,cam_mul(1.5f,et2)));
// MPAS module_ra_cam.F:3386
f1co2=cam_mul(cam_mul(cam_mul(899.70f,omet),(cam_add(cam_add(1.f,cam_mul(1.94774f,et)),cam_mul(4.73486f,et2)))),rsqti);
// MPAS module_ra_cam.F:3387
f1sqwp(i)=cam_mul(f1co2,sqwp);
// MPAS module_ra_cam.F:3388
t1co2(i)=cam_div(1.f,(cam_add(1.f,(cam_mul(cam_mul(cam_mul(245.18f,omet),sqwp),rsqti)))));
// MPAS module_ra_cam.F:3389
oneme=cam_sub(1.f,et2);
// MPAS module_ra_cam.F:3390
alphat=cam_mul(cam_pow(oneme,3),rsqti);
// MPAS module_ra_cam.F:3391
pi=cam_mul(cam_abs(dpnm(i)),winpl(i,kn));
// MPAS module_ra_cam.F:3392
wco2=cam_mul(cam_mul(cam_mul(2.5221f,co2vmr),pi),rga);
// MPAS module_ra_cam.F:3393
u7(i)=cam_mul(cam_mul(cam_mul(4.9411e4f,alphat),et2),wco2);
// MPAS module_ra_cam.F:3394
u8=cam_mul(cam_mul(cam_mul(3.9744e4f,alphat),et4),wco2);
// MPAS module_ra_cam.F:3395
u9=cam_mul(cam_mul(cam_mul(cam_mul(1.0447e5f,alphat),et4),et2),wco2);
// MPAS module_ra_cam.F:3396
u13=cam_mul(cam_mul(cam_mul(2.8388e3f,alphat),et4),wco2);
// MPAS module_ra_cam.F:3397
tpath=tbar(i,kn);
// MPAS module_ra_cam.F:3398
tlocal=tbar(i,kn);
// MPAS module_ra_cam.F:3399
tcrfac=cam_sqrt(cam_mul((cam_mul(tlocal,r250)),(cam_mul(tpath,r300))));
// MPAS module_ra_cam.F:3400
posqt=cam_mul((cam_add(cam_mul(pinpl(i,kn),rsslp),cam_mul(dpfco2,tcrfac))),rsqti);
// MPAS module_ra_cam.F:3401
rbeta7(i)=cam_div(1.f,(cam_mul(5.3228f,posqt)));
// MPAS module_ra_cam.F:3402
rbeta8=cam_div(1.f,(cam_mul(10.6576f,posqt)));
// MPAS module_ra_cam.F:3403
rbeta9=rbeta7(i);
// MPAS module_ra_cam.F:3404
rbeta13=rbeta9;
// MPAS module_ra_cam.F:3405
f2co2(i)=cam_add(cam_add(cam_div(u7(i),cam_sqrt(cam_add(4.f,cam_mul(u7(i),(cam_add(1.f,rbeta7(i))))))),cam_div(u8,cam_sqrt(cam_add(4.f,cam_mul(u8,(cam_add(1.f,rbeta8))))))),cam_div(u9,cam_sqrt(cam_add(4.f,cam_mul(u9,(cam_add(1.f,rbeta9)))))));
// MPAS module_ra_cam.F:3408
f3co2(i)=cam_div(u13,cam_sqrt(cam_add(4.f,cam_mul(u13,(cam_add(1.f,rbeta13))))));
// MPAS module_ra_cam.F:3409
tmp1=cam_log(cam_add(1.f,f1sqwp(i)));
// MPAS module_ra_cam.F:3410
tmp2=cam_log(cam_add(1.f,f2co2(i)));
// MPAS module_ra_cam.F:3411
tmp3=cam_log(cam_add(1.f,f3co2(i)));
// MPAS module_ra_cam.F:3412
absbnd=cam_mul((cam_add(cam_add(tmp1,cam_mul(cam_mul(2.f,t1co2(i)),tmp2)),cam_mul(2.f,tmp3))),sqti(i));
// MPAS module_ra_cam.F:3413
abso(i,4)=cam_mul(cam_mul(trab2(i),emm(i,kn)),absbnd);
// MPAS module_ra_cam.F:3414
tco2(i)=cam_div(1.0f,(cam_add(1.0f,cam_div(cam_mul(10.0f,u7(i)),cam_sqrt(cam_add(4.f,cam_mul(u7(i),(cam_add(1.f,rbeta7(i))))))))));
// MPAS module_ra_cam.F:3415
}
// MPAS module_ra_cam.F:3419
trcabn(lchnk,ncol,pcols,pverp,k2,kn,ucfc11,ucfc12,un2o0,un2o1,uch4,uco211,uco212,uco213,uco221,uco222,uco223,tbar,bplnk,winpl,pinpl,tco2,th2o,to3,uptype,dw,s2c,u,pnew,abstrc,uinpl,aer_trn_ngh); if(error) return;
// MPAS module_ra_cam.F:3430
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3431
absnxt(i,k2,kn)=cam_add(cam_add(cam_add(cam_add(abso(i,1),abso(i,2)),abso(i,3)),abso(i,4)),abstrc(i));
// MPAS module_ra_cam.F:3433
}
// MPAS module_ra_cam.F:3434
}
// MPAS module_ra_cam.F:3435
}
// MPAS module_ra_cam.F:3437
return;
}
// MPAS module_ra_cam.F:3442-4375
CAM_ROUTINE void radems(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _s2c,const CamArray<double>& _tcg,const CamArray<double>& _w,const CamArray<double>& _tplnke,const CamArray<double>& _plh2o,const CamArray<double>& _pnm,const CamArray<double>& _plco2,const CamArray<double>& _tint,const CamArray<double>& _tint4,const CamArray<double>& _tlayr,const CamArray<double>& _tlayr4,const CamArray<double>& _plol,const CamArray<double>& _plos,const CamArray<double>& _ucfc11,const CamArray<double>& _ucfc12,const CamArray<double>& _un2o0,const CamArray<double>& _un2o1,const CamArray<double>& _uch4,const CamArray<double>& _uco211,const CamArray<double>& _uco212,const CamArray<double>& _uco213,const CamArray<double>& _uco221,const CamArray<double>& _uco222,const CamArray<double>& _uco223,const CamArray<double>& _uptype,const CamArray<double>& _bn2o0,const CamArray<double>& _bn2o1,const CamArray<double>& _bch4,const CamArray<double>& _co2em,const CamArray<double>& _co2eml,const CamArray<double>& _co2t,const CamArray<double>& _h2otr,const CamArray<double>& _abplnk1,const CamArray<double>& _abplnk2,const CamArray<double>& _emstot,const CamArray<double>& _plh2ob,const CamArray<double>& _wb,const CamArray<double>& _aer_trn_ttl) {
CamFrame _frame(arena);
CAM_TRACE("radems.s2c");
auto s2c=_s2c.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.tcg");
auto tcg=_tcg.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.w");
auto w=_w.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.tplnke");
auto tplnke=_tplnke.rebound(1,pcols);
CAM_TRACE("radems.plh2o");
auto plh2o=_plh2o.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.pnm");
auto pnm=_pnm.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.plco2");
auto plco2=_plco2.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.tint");
auto tint=_tint.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.tint4");
auto tint4=_tint4.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.tlayr");
auto tlayr=_tlayr.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.tlayr4");
auto tlayr4=_tlayr4.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.plol");
auto plol=_plol.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.plos");
auto plos=_plos.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.ucfc11");
auto ucfc11=_ucfc11.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.ucfc12");
auto ucfc12=_ucfc12.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.un2o0");
auto un2o0=_un2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.un2o1");
auto un2o1=_un2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.uch4");
auto uch4=_uch4.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.uco211");
auto uco211=_uco211.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.uco212");
auto uco212=_uco212.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.uco213");
auto uco213=_uco213.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.uco221");
auto uco221=_uco221.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.uco222");
auto uco222=_uco222.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.uco223");
auto uco223=_uco223.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.uptype");
auto uptype=_uptype.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.bn2o0");
auto bn2o0=_bn2o0.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.bn2o1");
auto bn2o1=_bn2o1.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.bch4");
auto bch4=_bch4.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.co2em");
auto co2em=_co2em.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.co2eml");
auto co2eml=_co2eml.rebound(1,pcols,1,pver);
CAM_TRACE("radems.co2t");
auto co2t=_co2t.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.h2otr");
auto h2otr=_h2otr.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.abplnk1");
auto abplnk1=_abplnk1.rebound(1,14,1,pcols,1,pverp);
CAM_TRACE("radems.abplnk2");
auto abplnk2=_abplnk2.rebound(1,14,1,pcols,1,pverp);
CAM_TRACE("radems.emstot");
auto emstot=_emstot.rebound(1,pcols,1,pverp);
CAM_TRACE("radems.plh2ob");
auto plh2ob=_plh2ob.rebound(1,nbands,1,pcols,1,pverp);
CAM_TRACE("radems.wb");
auto wb=_wb.rebound(1,nbands,1,pcols,1,pverp);
CAM_TRACE("radems.aer_trn_ttl");
auto aer_trn_ttl=_aer_trn_ttl.rebound(1,pcols,1,pverp,1,pverp,1,bnd_nbr_lw);
int i;
int k;
int k1;
auto h2oems=arena.array<double>(1,pcols,1,pverp);
double tpathe;
auto dtx=arena.array<double>(1,pcols);
auto dty=arena.array<double>(1,pcols);
auto emis=arena.array<double>(1,pcols,1,2);
auto term7=arena.array<double>(1,pcols,1,2);
auto term8=arena.array<double>(1,pcols,1,2);
auto tr1=arena.array<double>(1,pcols);
auto tr2=arena.array<double>(1,pcols);
auto tr3=arena.array<double>(1,pcols);
auto tr4=arena.array<double>(1,pcols);
auto tr7=arena.array<double>(1,pcols);
auto tr8=arena.array<double>(1,pcols);
auto k21=arena.array<double>(1,pcols);
auto k22=arena.array<double>(1,pcols);
auto u=arena.array<double>(1,pcols);
auto ub=arena.array<double>(1,nbands);
double pnew;
auto pnewb=arena.array<double>(1,nbands);
auto uc1=arena.array<double>(1,pcols);
double fwk;
auto troco2=arena.array<double>(1,pcols,1,pverp);
auto emplnk=arena.array<double>(1,14,1,pcols);
auto emstrc=arena.array<double>(1,pcols,1,pverp);
auto co2ems=arena.array<double>(1,pcols,1,pverp);
auto co2plk=arena.array<double>(1,pcols);
auto sum=arena.array<double>(1,pcols);
double t1i;
double sqti;
double pi;
double et;
double et2;
double et4;
double omet;
double ex;
double f1co2;
double f2co2;
double f3co2;
double t1co2;
double sqwp;
double f1sqwp;
double oneme;
double alphat;
double wco2;
double posqt;
double rbeta7;
double rbeta8;
double rbeta9;
double rbeta13;
double tpath;
double tmp1;
double tmp2;
double tmp3;
double tlayr5;
double rsqti;
double exm1sq;
double u7;
double u8;
double u9;
double u13;
double r250;
double r300;
double rsslp;
auto o3ems=arena.array<double>(1,pcols,1,pverp);
auto dbvtt=arena.array<double>(1,pcols);
double t;
double ux;
double vx;
double te;
double u1;
double u2;
double phat;
double tlocal;
double tcrfac;
double beta;
double realnu;
double o3bndi;
double absbnd;
auto tco2=arena.array<double>(1,pcols);
auto th2o=arena.array<double>(1,pcols);
auto to3=arena.array<double>(1,pcols);
double fe;
double e_star;
double l_star;
double c_star;
double te1;
double te2;
double te3;
double te4;
double te5;
double log_u;
double log_uc;
double log_p;
double t_p;
double t_e;
int iu;
int iu1;
int iuc;
int iuc1;
int ip;
int ip1;
int itp;
int itp1;
int ite;
int ite1;
int irh;
int irh1;
double dvar;
double uvar;
double uscl;
double wu;
double wu1;
double wuc;
double wuc1;
double wp;
double wp1;
double wtp;
double wtp1;
double wte;
double wte1;
double wrh;
double wrh1;
double w_0_0_;
double w_0_1_;
double w_1_0_;
double w_1_1_;
double w_0_00;
double w_0_01;
double w_0_10;
double w_0_11;
double w_1_00;
double w_1_01;
double w_1_10;
double w_1_11;
double w00_00;
double w00_01;
double w00_10;
double w00_11;
double w01_00;
double w01_01;
double w01_10;
double w01_11;
double w10_00;
double w10_01;
double w10_10;
double w10_11;
double w11_00;
double w11_01;
double w11_10;
double w11_11;
int ib;
double pch2o;
double fch2o;
double uch2o;
double fdif;
double sslp_mks;
double esx;
double qsx;
double pnew_mks;
double q_path;
double rh_path;
double omeps;
int iest;
auto dbvt=[&](double t) { return cam_div((cam_add(cam_neg(2.8911366682e-4f),cam_mul((cam_add(2.3771251896e-6f,cam_mul(1.1305188929e-10f,t))),t))),(cam_add(1.0f,cam_mul((cam_add(cam_neg(6.1364820707e-3f),cam_mul(1.5550319767e-5f,t))),t)))); };
auto fo3=[&](double ux,double vx) { return cam_div(ux,cam_sqrt(cam_add(4.f,cam_mul(ux,(cam_add(1.f,vx)))))); };
// MPAS module_ra_cam.F:3807
r250=cam_div(1.f,250.f);
// MPAS module_ra_cam.F:3808
r300=cam_div(1.f,300.f);
// MPAS module_ra_cam.F:3809
rsslp=cam_div(1.f,sslp);
// MPAS module_ra_cam.F:3813
fdif=1.66f;
// MPAS module_ra_cam.F:3814
sslp_mks=cam_div(sslp,10.0f);
// MPAS module_ra_cam.F:3815
omeps=cam_sub(1.0f,epsilo);
// MPAS module_ra_cam.F:3819
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3820
ex=cam_exp(cam_div(960.f,tplnke(i)));
// MPAS module_ra_cam.F:3821
co2plk(i)=cam_div(5.e8f,(cam_mul((cam_pow(tplnke(i),4)),(cam_sub(ex,1.f)))));
// MPAS module_ra_cam.F:3822
co2t(i,ntoplw)=tplnke(i);
// MPAS module_ra_cam.F:3823
sum(i)=cam_mul(co2t(i,ntoplw),pnm(i,ntoplw));
// MPAS module_ra_cam.F:3824
}
// MPAS module_ra_cam.F:3825
k=ntoplw;
// MPAS module_ra_cam.F:3826
for(k1=pverp;k1>=cam_add(ntoplw,1);k1+=cam_neg(1)) {
// MPAS module_ra_cam.F:3827
k=cam_add(k,1);
// MPAS module_ra_cam.F:3828
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3829
sum(i)=cam_add(sum(i),cam_mul(tlayr(i,k),(cam_sub(pnm(i,k),pnm(i,cam_sub(k,1))))));
// MPAS module_ra_cam.F:3830
ex=cam_exp(cam_div(960.f,tlayr(i,k1)));
// MPAS module_ra_cam.F:3831
tlayr5=cam_mul(tlayr(i,k1),tlayr4(i,k1));
// MPAS module_ra_cam.F:3832
co2eml(i,cam_sub(k1,1))=cam_div(cam_mul(1.2e11f,ex),(cam_mul(tlayr5,cam_pow((cam_sub(ex,1.f)),2))));
// MPAS module_ra_cam.F:3833
co2t(i,k)=cam_div(sum(i),pnm(i,k));
// MPAS module_ra_cam.F:3834
}
// MPAS module_ra_cam.F:3835
}
// MPAS module_ra_cam.F:3839
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3840
dbvtt(i)=dbvt(tplnke(i));
// MPAS module_ra_cam.F:3841
}
// MPAS module_ra_cam.F:3845
trcplk(lchnk,ncol,pcols,pver,pverp,tint,tlayr,tplnke,emplnk,abplnk1,abplnk2); if(error) return;
// MPAS module_ra_cam.F:3851
for(k1=ntoplw;k1<=pverp;k1+=1) {
// MPAS module_ra_cam.F:3866
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:3867
u(i)=plh2o(i,k1);
// MPAS module_ra_cam.F:3868
pnew=cam_div(u(i),w(i,k1));
// MPAS module_ra_cam.F:3869
pnew_mks=cam_mul(pnew,sslp_mks);
// MPAS module_ra_cam.F:3873
uc1(i)=cam_div(cam_mul((cam_add(s2c(i,k1),cam_mul(1.7e-3f,plh2o(i,k1)))),(cam_add(1.f,cam_mul(2.f,s2c(i,k1))))),(cam_add(1.f,cam_mul(15.f,s2c(i,k1)))));
// MPAS module_ra_cam.F:3875
pch2o=s2c(i,k1);
// MPAS module_ra_cam.F:3879
tpathe=cam_div(tcg(i,k1),w(i,k1));
// MPAS module_ra_cam.F:3880
t_p=cam_min(cam_max(tpathe,min_tp_h2o),max_tp_h2o);
// MPAS module_ra_cam.F:3881
iest=cam_sub(floor(t_p),min_tp_h2o);
// MPAS module_ra_cam.F:3882
esx=cam_add(estblh2o(iest),cam_mul((cam_sub(estblh2o(cam_add(iest,1)),estblh2o(iest))),(cam_sub(cam_sub(t_p,min_tp_h2o),iest))));
// MPAS module_ra_cam.F:3884
qsx=cam_div(cam_mul(epsilo,esx),(cam_sub(pnew_mks,cam_mul(omeps,esx))));
// MPAS module_ra_cam.F:3888
q_path=cam_div(cam_div(w(i,k1),pnm(i,k1)),rga);
// MPAS module_ra_cam.F:3899
ub(1)=cam_div(plh2ob(1,i,k1),psi(t_p,1));
// MPAS module_ra_cam.F:3900
ub(2)=cam_div(plh2ob(2,i,k1),psi(t_p,2));
// MPAS module_ra_cam.F:3902
pnewb(1)=cam_mul(cam_div(ub(1),wb(1,i,k1)),phi(t_p,1));
// MPAS module_ra_cam.F:3903
pnewb(2)=cam_mul(cam_div(ub(2),wb(2,i,k1)),phi(t_p,2));
// MPAS module_ra_cam.F:3907
dtx(i)=cam_sub(tplnke(i),250.f);
// MPAS module_ra_cam.F:3908
dty(i)=cam_sub(tpathe,250.f);
// MPAS module_ra_cam.F:3932
te1=tplnke(i);
// MPAS module_ra_cam.F:3933
te2=cam_mul(te1,te1);
// MPAS module_ra_cam.F:3934
te3=cam_mul(te2,te1);
// MPAS module_ra_cam.F:3935
te4=cam_mul(te3,te1);
// MPAS module_ra_cam.F:3936
te5=cam_mul(te4,te1);
// MPAS module_ra_cam.F:3940
dvar=cam_div((cam_sub(t_p,min_tp_h2o)),dtp_h2o);
// MPAS module_ra_cam.F:3941
itp=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_tp,1));
// MPAS module_ra_cam.F:3942
itp1=cam_add(itp,1);
// MPAS module_ra_cam.F:3943
wtp=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:3944
wtp1=cam_sub(1.0f,wtp);
// MPAS module_ra_cam.F:3946
t_e=cam_min(cam_max(cam_sub(tplnke(i),t_p),min_te_h2o),max_te_h2o);
// MPAS module_ra_cam.F:3947
dvar=cam_div((cam_sub(t_e,min_te_h2o)),dte_h2o);
// MPAS module_ra_cam.F:3948
ite=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_te,1));
// MPAS module_ra_cam.F:3949
ite1=cam_add(ite,1);
// MPAS module_ra_cam.F:3950
wte=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:3951
wte1=cam_sub(1.0f,wte);
// MPAS module_ra_cam.F:3953
rh_path=cam_min(cam_max(cam_div(q_path,qsx),min_rh_h2o),max_rh_h2o);
// MPAS module_ra_cam.F:3954
dvar=cam_div((cam_sub(rh_path,min_rh_h2o)),drh_h2o);
// MPAS module_ra_cam.F:3955
irh=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_rh,1));
// MPAS module_ra_cam.F:3956
irh1=cam_add(irh,1);
// MPAS module_ra_cam.F:3957
wrh=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:3958
wrh1=cam_sub(1.0f,wrh);
// MPAS module_ra_cam.F:3960
w_0_0_=cam_mul(wtp,wte);
// MPAS module_ra_cam.F:3961
w_0_1_=cam_mul(wtp,wte1);
// MPAS module_ra_cam.F:3962
w_1_0_=cam_mul(wtp1,wte);
// MPAS module_ra_cam.F:3963
w_1_1_=cam_mul(wtp1,wte1);
// MPAS module_ra_cam.F:3965
w_0_00=cam_mul(w_0_0_,wrh);
// MPAS module_ra_cam.F:3966
w_0_01=cam_mul(w_0_0_,wrh1);
// MPAS module_ra_cam.F:3967
w_0_10=cam_mul(w_0_1_,wrh);
// MPAS module_ra_cam.F:3968
w_0_11=cam_mul(w_0_1_,wrh1);
// MPAS module_ra_cam.F:3969
w_1_00=cam_mul(w_1_0_,wrh);
// MPAS module_ra_cam.F:3970
w_1_01=cam_mul(w_1_0_,wrh1);
// MPAS module_ra_cam.F:3971
w_1_10=cam_mul(w_1_1_,wrh);
// MPAS module_ra_cam.F:3972
w_1_11=cam_mul(w_1_1_,wrh1);
// MPAS module_ra_cam.F:4024
fch2o=fh2oself(t_p);
// MPAS module_ra_cam.F:4025
uch2o=cam_div((cam_mul(pch2o,epsilo)),(cam_mul(q_path,fch2o)));
// MPAS module_ra_cam.F:4030
ib=1;
// MPAS module_ra_cam.F:4032
uvar=cam_mul(ub(ib),fdif);
// MPAS module_ra_cam.F:4033
log_u=cam_min(cam_log10(cam_max(uvar,min_u_h2o)),max_lu_h2o);
// MPAS module_ra_cam.F:4034
dvar=cam_div((cam_sub(log_u,min_lu_h2o)),dlu_h2o);
// MPAS module_ra_cam.F:4035
iu=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_u,1));
// MPAS module_ra_cam.F:4036
iu1=cam_add(iu,1);
// MPAS module_ra_cam.F:4037
wu=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:4038
wu1=cam_sub(1.0f,wu);
// MPAS module_ra_cam.F:4040
log_p=cam_min(cam_log10(cam_max(pnewb(ib),min_p_h2o)),max_lp_h2o);
// MPAS module_ra_cam.F:4041
dvar=cam_div((cam_sub(log_p,min_lp_h2o)),dlp_h2o);
// MPAS module_ra_cam.F:4042
ip=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_p,1));
// MPAS module_ra_cam.F:4043
ip1=cam_add(ip,1);
// MPAS module_ra_cam.F:4044
wp=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:4045
wp1=cam_sub(1.0f,wp);
// MPAS module_ra_cam.F:4047
w00_00=cam_mul(wp,w_0_00);
// MPAS module_ra_cam.F:4048
w00_01=cam_mul(wp,w_0_01);
// MPAS module_ra_cam.F:4049
w00_10=cam_mul(wp,w_0_10);
// MPAS module_ra_cam.F:4050
w00_11=cam_mul(wp,w_0_11);
// MPAS module_ra_cam.F:4051
w01_00=cam_mul(wp,w_1_00);
// MPAS module_ra_cam.F:4052
w01_01=cam_mul(wp,w_1_01);
// MPAS module_ra_cam.F:4053
w01_10=cam_mul(wp,w_1_10);
// MPAS module_ra_cam.F:4054
w01_11=cam_mul(wp,w_1_11);
// MPAS module_ra_cam.F:4055
w10_00=cam_mul(wp1,w_0_00);
// MPAS module_ra_cam.F:4056
w10_01=cam_mul(wp1,w_0_01);
// MPAS module_ra_cam.F:4057
w10_10=cam_mul(wp1,w_0_10);
// MPAS module_ra_cam.F:4058
w10_11=cam_mul(wp1,w_0_11);
// MPAS module_ra_cam.F:4059
w11_00=cam_mul(wp1,w_1_00);
// MPAS module_ra_cam.F:4060
w11_01=cam_mul(wp1,w_1_01);
// MPAS module_ra_cam.F:4061
w11_10=cam_mul(wp1,w_1_10);
// MPAS module_ra_cam.F:4062
w11_11=cam_mul(wp1,w_1_11);
// MPAS module_ra_cam.F:4067
fe=cam_add(cam_add(cam_add(cam_add(cam_add(fet(1,ib),cam_mul(fet(2,ib),te1)),cam_mul(fet(3,ib),te2)),cam_mul(fet(4,ib),te3)),cam_mul(fet(5,ib),te4)),cam_mul(fet(6,ib),te5));
// MPAS module_ra_cam.F:4074
e_star=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_mul(cam_mul(eh2onw(ip,itp,iu,ite,irh),w11_11),wu1),cam_mul(cam_mul(eh2onw(ip,itp,iu,ite,irh1),w11_10),wu1)),cam_mul(cam_mul(eh2onw(ip,itp,iu,ite1,irh),w11_01),wu1)),cam_mul(cam_mul(eh2onw(ip,itp,iu,ite1,irh1),w11_00),wu1)),cam_mul(cam_mul(eh2onw(ip,itp,iu1,ite,irh),w11_11),wu)),cam_mul(cam_mul(eh2onw(ip,itp,iu1,ite,irh1),w11_10),wu)),cam_mul(cam_mul(eh2onw(ip,itp,iu1,ite1,irh),w11_01),wu)),cam_mul(cam_mul(eh2onw(ip,itp,iu1,ite1,irh1),w11_00),wu)),cam_mul(cam_mul(eh2onw(ip,itp1,iu,ite,irh),w10_11),wu1)),cam_mul(cam_mul(eh2onw(ip,itp1,iu,ite,irh1),w10_10),wu1)),cam_mul(cam_mul(eh2onw(ip,itp1,iu,ite1,irh),w10_01),wu1)),cam_mul(cam_mul(eh2onw(ip,itp1,iu,ite1,irh1),w10_00),wu1)),cam_mul(cam_mul(eh2onw(ip,itp1,iu1,ite,irh),w10_11),wu)),cam_mul(cam_mul(eh2onw(ip,itp1,iu1,ite,irh1),w10_10),wu)),cam_mul(cam_mul(eh2onw(ip,itp1,iu1,ite1,irh),w10_01),wu)),cam_mul(cam_mul(eh2onw(ip,itp1,iu1,ite1,irh1),w10_00),wu)),cam_mul(cam_mul(eh2onw(ip1,itp,iu,ite,irh),w01_11),wu1)),cam_mul(cam_mul(eh2onw(ip1,itp,iu,ite,irh1),w01_10),wu1)),cam_mul(cam_mul(eh2onw(ip1,itp,iu,ite1,irh),w01_01),wu1)),cam_mul(cam_mul(eh2onw(ip1,itp,iu,ite1,irh1),w01_00),wu1)),cam_mul(cam_mul(eh2onw(ip1,itp,iu1,ite,irh),w01_11),wu)),cam_mul(cam_mul(eh2onw(ip1,itp,iu1,ite,irh1),w01_10),wu)),cam_mul(cam_mul(eh2onw(ip1,itp,iu1,ite1,irh),w01_01),wu)),cam_mul(cam_mul(eh2onw(ip1,itp,iu1,ite1,irh1),w01_00),wu)),cam_mul(cam_mul(eh2onw(ip1,itp1,iu,ite,irh),w00_11),wu1)),cam_mul(cam_mul(eh2onw(ip1,itp1,iu,ite,irh1),w00_10),wu1)),cam_mul(cam_mul(eh2onw(ip1,itp1,iu,ite1,irh),w00_01),wu1)),cam_mul(cam_mul(eh2onw(ip1,itp1,iu,ite1,irh1),w00_00),wu1)),cam_mul(cam_mul(eh2onw(ip1,itp1,iu1,ite,irh),w00_11),wu)),cam_mul(cam_mul(eh2onw(ip1,itp1,iu1,ite,irh1),w00_10),wu)),cam_mul(cam_mul(eh2onw(ip1,itp1,iu1,ite1,irh),w00_01),wu)),cam_mul(cam_mul(eh2onw(ip1,itp1,iu1,ite1,irh1),w00_00),wu));
// MPAS module_ra_cam.F:4107
emis(i,ib)=cam_min(cam_max(cam_mul(fe,(cam_sub(1.0f,cam_mul((cam_sub(1.0f,e_star)),aer_trn_ttl(i,k1,1,ib))))),0.0),1.0);
// MPAS module_ra_cam.F:4113
if((uvar<min_u_h2o)) {
// MPAS module_ra_cam.F:4114
uscl=cam_div(uvar,min_u_h2o);
// MPAS module_ra_cam.F:4115
emis(i,ib)=cam_mul(emis(i,ib),uscl);
// MPAS module_ra_cam.F:4116
}
// MPAS module_ra_cam.F:4123
ib=2;
// MPAS module_ra_cam.F:4125
uvar=cam_mul(ub(ib),fdif);
// MPAS module_ra_cam.F:4126
log_u=cam_min(cam_log10(cam_max(uvar,min_u_h2o)),max_lu_h2o);
// MPAS module_ra_cam.F:4127
dvar=cam_div((cam_sub(log_u,min_lu_h2o)),dlu_h2o);
// MPAS module_ra_cam.F:4128
iu=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_u,1));
// MPAS module_ra_cam.F:4129
iu1=cam_add(iu,1);
// MPAS module_ra_cam.F:4130
wu=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:4131
wu1=cam_sub(1.0f,wu);
// MPAS module_ra_cam.F:4133
log_p=cam_min(cam_log10(cam_max(pnewb(ib),min_p_h2o)),max_lp_h2o);
// MPAS module_ra_cam.F:4134
dvar=cam_div((cam_sub(log_p,min_lp_h2o)),dlp_h2o);
// MPAS module_ra_cam.F:4135
ip=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_p,1));
// MPAS module_ra_cam.F:4136
ip1=cam_add(ip,1);
// MPAS module_ra_cam.F:4137
wp=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:4138
wp1=cam_sub(1.0f,wp);
// MPAS module_ra_cam.F:4140
w00_00=cam_mul(wp,w_0_00);
// MPAS module_ra_cam.F:4141
w00_01=cam_mul(wp,w_0_01);
// MPAS module_ra_cam.F:4142
w00_10=cam_mul(wp,w_0_10);
// MPAS module_ra_cam.F:4143
w00_11=cam_mul(wp,w_0_11);
// MPAS module_ra_cam.F:4144
w01_00=cam_mul(wp,w_1_00);
// MPAS module_ra_cam.F:4145
w01_01=cam_mul(wp,w_1_01);
// MPAS module_ra_cam.F:4146
w01_10=cam_mul(wp,w_1_10);
// MPAS module_ra_cam.F:4147
w01_11=cam_mul(wp,w_1_11);
// MPAS module_ra_cam.F:4148
w10_00=cam_mul(wp1,w_0_00);
// MPAS module_ra_cam.F:4149
w10_01=cam_mul(wp1,w_0_01);
// MPAS module_ra_cam.F:4150
w10_10=cam_mul(wp1,w_0_10);
// MPAS module_ra_cam.F:4151
w10_11=cam_mul(wp1,w_0_11);
// MPAS module_ra_cam.F:4152
w11_00=cam_mul(wp1,w_1_00);
// MPAS module_ra_cam.F:4153
w11_01=cam_mul(wp1,w_1_01);
// MPAS module_ra_cam.F:4154
w11_10=cam_mul(wp1,w_1_10);
// MPAS module_ra_cam.F:4155
w11_11=cam_mul(wp1,w_1_11);
// MPAS module_ra_cam.F:4157
log_uc=cam_min(cam_log10(cam_max(cam_mul(uch2o,fdif),min_u_h2o)),max_lu_h2o);
// MPAS module_ra_cam.F:4158
dvar=cam_div((cam_sub(log_uc,min_lu_h2o)),dlu_h2o);
// MPAS module_ra_cam.F:4159
iuc=cam_min(cam_max(cam_add(((int)(cam_aint(dvar,r8))),1),1),cam_sub(n_u,1));
// MPAS module_ra_cam.F:4160
iuc1=cam_add(iuc,1);
// MPAS module_ra_cam.F:4161
wuc=cam_sub(dvar,floor(dvar));
// MPAS module_ra_cam.F:4162
wuc1=cam_sub(1.0f,wuc);
// MPAS module_ra_cam.F:4166
fe=cam_add(cam_add(cam_add(cam_add(cam_add(fet(1,ib),cam_mul(fet(2,ib),te1)),cam_mul(fet(3,ib),te2)),cam_mul(fet(4,ib),te3)),cam_mul(fet(5,ib),te4)),cam_mul(fet(6,ib),te5));
// MPAS module_ra_cam.F:4173
l_star=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_mul(cam_mul(ln_eh2ow(ip,itp,iu,ite,irh),w11_11),wu1),cam_mul(cam_mul(ln_eh2ow(ip,itp,iu,ite,irh1),w11_10),wu1)),cam_mul(cam_mul(ln_eh2ow(ip,itp,iu,ite1,irh),w11_01),wu1)),cam_mul(cam_mul(ln_eh2ow(ip,itp,iu,ite1,irh1),w11_00),wu1)),cam_mul(cam_mul(ln_eh2ow(ip,itp,iu1,ite,irh),w11_11),wu)),cam_mul(cam_mul(ln_eh2ow(ip,itp,iu1,ite,irh1),w11_10),wu)),cam_mul(cam_mul(ln_eh2ow(ip,itp,iu1,ite1,irh),w11_01),wu)),cam_mul(cam_mul(ln_eh2ow(ip,itp,iu1,ite1,irh1),w11_00),wu)),cam_mul(cam_mul(ln_eh2ow(ip,itp1,iu,ite,irh),w10_11),wu1)),cam_mul(cam_mul(ln_eh2ow(ip,itp1,iu,ite,irh1),w10_10),wu1)),cam_mul(cam_mul(ln_eh2ow(ip,itp1,iu,ite1,irh),w10_01),wu1)),cam_mul(cam_mul(ln_eh2ow(ip,itp1,iu,ite1,irh1),w10_00),wu1)),cam_mul(cam_mul(ln_eh2ow(ip,itp1,iu1,ite,irh),w10_11),wu)),cam_mul(cam_mul(ln_eh2ow(ip,itp1,iu1,ite,irh1),w10_10),wu)),cam_mul(cam_mul(ln_eh2ow(ip,itp1,iu1,ite1,irh),w10_01),wu)),cam_mul(cam_mul(ln_eh2ow(ip,itp1,iu1,ite1,irh1),w10_00),wu)),cam_mul(cam_mul(ln_eh2ow(ip1,itp,iu,ite,irh),w01_11),wu1)),cam_mul(cam_mul(ln_eh2ow(ip1,itp,iu,ite,irh1),w01_10),wu1)),cam_mul(cam_mul(ln_eh2ow(ip1,itp,iu,ite1,irh),w01_01),wu1)),cam_mul(cam_mul(ln_eh2ow(ip1,itp,iu,ite1,irh1),w01_00),wu1)),cam_mul(cam_mul(ln_eh2ow(ip1,itp,iu1,ite,irh),w01_11),wu)),cam_mul(cam_mul(ln_eh2ow(ip1,itp,iu1,ite,irh1),w01_10),wu)),cam_mul(cam_mul(ln_eh2ow(ip1,itp,iu1,ite1,irh),w01_01),wu)),cam_mul(cam_mul(ln_eh2ow(ip1,itp,iu1,ite1,irh1),w01_00),wu)),cam_mul(cam_mul(ln_eh2ow(ip1,itp1,iu,ite,irh),w00_11),wu1)),cam_mul(cam_mul(ln_eh2ow(ip1,itp1,iu,ite,irh1),w00_10),wu1)),cam_mul(cam_mul(ln_eh2ow(ip1,itp1,iu,ite1,irh),w00_01),wu1)),cam_mul(cam_mul(ln_eh2ow(ip1,itp1,iu,ite1,irh1),w00_00),wu1)),cam_mul(cam_mul(ln_eh2ow(ip1,itp1,iu1,ite,irh),w00_11),wu)),cam_mul(cam_mul(ln_eh2ow(ip1,itp1,iu1,ite,irh1),w00_10),wu)),cam_mul(cam_mul(ln_eh2ow(ip1,itp1,iu1,ite1,irh),w00_01),wu)),cam_mul(cam_mul(ln_eh2ow(ip1,itp1,iu1,ite1,irh1),w00_00),wu));
// MPAS module_ra_cam.F:4207
c_star=cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_add(cam_mul(cam_mul(cn_eh2ow(ip,itp,iuc,ite,irh),w11_11),wuc1),cam_mul(cam_mul(cn_eh2ow(ip,itp,iuc,ite,irh1),w11_10),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip,itp,iuc,ite1,irh),w11_01),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip,itp,iuc,ite1,irh1),w11_00),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip,itp,iuc1,ite,irh),w11_11),wuc)),cam_mul(cam_mul(cn_eh2ow(ip,itp,iuc1,ite,irh1),w11_10),wuc)),cam_mul(cam_mul(cn_eh2ow(ip,itp,iuc1,ite1,irh),w11_01),wuc)),cam_mul(cam_mul(cn_eh2ow(ip,itp,iuc1,ite1,irh1),w11_00),wuc)),cam_mul(cam_mul(cn_eh2ow(ip,itp1,iuc,ite,irh),w10_11),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip,itp1,iuc,ite,irh1),w10_10),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip,itp1,iuc,ite1,irh),w10_01),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip,itp1,iuc,ite1,irh1),w10_00),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip,itp1,iuc1,ite,irh),w10_11),wuc)),cam_mul(cam_mul(cn_eh2ow(ip,itp1,iuc1,ite,irh1),w10_10),wuc)),cam_mul(cam_mul(cn_eh2ow(ip,itp1,iuc1,ite1,irh),w10_01),wuc)),cam_mul(cam_mul(cn_eh2ow(ip,itp1,iuc1,ite1,irh1),w10_00),wuc)),cam_mul(cam_mul(cn_eh2ow(ip1,itp,iuc,ite,irh),w01_11),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip1,itp,iuc,ite,irh1),w01_10),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip1,itp,iuc,ite1,irh),w01_01),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip1,itp,iuc,ite1,irh1),w01_00),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip1,itp,iuc1,ite,irh),w01_11),wuc)),cam_mul(cam_mul(cn_eh2ow(ip1,itp,iuc1,ite,irh1),w01_10),wuc)),cam_mul(cam_mul(cn_eh2ow(ip1,itp,iuc1,ite1,irh),w01_01),wuc)),cam_mul(cam_mul(cn_eh2ow(ip1,itp,iuc1,ite1,irh1),w01_00),wuc)),cam_mul(cam_mul(cn_eh2ow(ip1,itp1,iuc,ite,irh),w00_11),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip1,itp1,iuc,ite,irh1),w00_10),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip1,itp1,iuc,ite1,irh),w00_01),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip1,itp1,iuc,ite1,irh1),w00_00),wuc1)),cam_mul(cam_mul(cn_eh2ow(ip1,itp1,iuc1,ite,irh),w00_11),wuc)),cam_mul(cam_mul(cn_eh2ow(ip1,itp1,iuc1,ite,irh1),w00_10),wuc)),cam_mul(cam_mul(cn_eh2ow(ip1,itp1,iuc1,ite1,irh),w00_01),wuc)),cam_mul(cam_mul(cn_eh2ow(ip1,itp1,iuc1,ite1,irh1),w00_00),wuc));
// MPAS module_ra_cam.F:4240
emis(i,ib)=cam_min(cam_max(cam_mul(fe,(cam_sub(1.0f,cam_mul(cam_mul(l_star,c_star),aer_trn_ttl(i,k1,1,ib))))),0.0),1.0);
// MPAS module_ra_cam.F:4246
if((uvar<min_u_h2o)) {
// MPAS module_ra_cam.F:4247
uscl=cam_div(uvar,min_u_h2o);
// MPAS module_ra_cam.F:4248
emis(i,ib)=cam_mul(emis(i,ib),uscl);
// MPAS module_ra_cam.F:4249
}
// MPAS module_ra_cam.F:4255
h2oems(i,k1)=cam_add(emis(i,1),emis(i,2));
// MPAS module_ra_cam.F:4257
}
// MPAS module_ra_cam.F:4262
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4263
term7(i,1)=cam_add(coefj(1,1),cam_mul(cam_mul(coefj(2,1),dty(i)),(cam_add(1.f,cam_mul(c16,dty(i))))));
// MPAS module_ra_cam.F:4264
term8(i,1)=cam_add(coefk(1,1),cam_mul(cam_mul(coefk(2,1),dty(i)),(cam_add(1.f,cam_mul(c17,dty(i))))));
// MPAS module_ra_cam.F:4265
term7(i,2)=cam_add(coefj(1,2),cam_mul(cam_mul(coefj(2,2),dty(i)),(cam_add(1.f,cam_mul(c26,dty(i))))));
// MPAS module_ra_cam.F:4266
term8(i,2)=cam_add(coefk(1,2),cam_mul(cam_mul(coefk(2,2),dty(i)),(cam_add(1.f,cam_mul(c27,dty(i))))));
// MPAS module_ra_cam.F:4267
}
// MPAS module_ra_cam.F:4268
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4272
k21(i)=cam_add(term7(i,1),cam_div(term8(i,1),(cam_add(1.f,cam_mul((cam_add(c30,cam_mul(cam_mul(c31,(cam_sub(dty(i),10.f))),(cam_sub(dty(i),10.f))))),cam_sqrt(u(i)))))));
// MPAS module_ra_cam.F:4274
k22(i)=cam_add(term7(i,2),cam_div(term8(i,2),(cam_add(1.f,cam_mul((cam_add(c28,cam_mul(c29,(cam_sub(dty(i),10.f))))),cam_sqrt(u(i)))))));
// MPAS module_ra_cam.F:4276
fwk=cam_add(fwcoef,cam_div(fwc1,(cam_add(1.f,cam_mul(fwc2,u(i))))));
// MPAS module_ra_cam.F:4277
tr1(i)=cam_exp(cam_neg((cam_mul(k21(i),(cam_add(cam_sqrt(u(i)),cam_mul(cam_mul(fc1,fwk),u(i))))))));
// MPAS module_ra_cam.F:4278
tr2(i)=cam_exp(cam_neg((cam_mul(k22(i),(cam_add(cam_sqrt(u(i)),cam_mul(cam_mul(fc1,fwk),u(i))))))));
// MPAS module_ra_cam.F:4279
tr1(i)=cam_mul(tr1(i),aer_trn_ttl(i,k1,1,idx_lw_0650_0800));
// MPAS module_ra_cam.F:4281
tr2(i)=cam_mul(tr2(i),aer_trn_ttl(i,k1,1,idx_lw_0500_0650));
// MPAS module_ra_cam.F:4283
tr3(i)=cam_exp(cam_neg((cam_mul((cam_add(coefh(1,1),cam_mul(coefh(2,1),dtx(i)))),uc1(i)))));
// MPAS module_ra_cam.F:4284
tr4(i)=cam_exp(cam_neg((cam_mul((cam_add(coefh(1,2),cam_mul(coefh(2,2),dtx(i)))),uc1(i)))));
// MPAS module_ra_cam.F:4285
tr7(i)=cam_mul(tr1(i),tr3(i));
// MPAS module_ra_cam.F:4286
tr8(i)=cam_mul(tr2(i),tr4(i));
// MPAS module_ra_cam.F:4287
troco2(i,k1)=cam_add(cam_mul(0.65f,tr7(i)),cam_mul(0.35f,tr8(i)));
// MPAS module_ra_cam.F:4288
th2o(i)=tr8(i);
// MPAS module_ra_cam.F:4289
}
// MPAS module_ra_cam.F:4293
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4294
t1i=cam_exp(cam_neg(cam_div(480.f,co2t(i,k1))));
// MPAS module_ra_cam.F:4295
sqti=cam_sqrt(co2t(i,k1));
// MPAS module_ra_cam.F:4296
rsqti=cam_div(1.f,sqti);
// MPAS module_ra_cam.F:4297
et=t1i;
// MPAS module_ra_cam.F:4298
et2=cam_mul(et,et);
// MPAS module_ra_cam.F:4299
et4=cam_mul(et2,et2);
// MPAS module_ra_cam.F:4300
omet=cam_sub(1.f,cam_mul(1.5f,et2));
// MPAS module_ra_cam.F:4301
f1co2=cam_mul(cam_mul(cam_mul(899.70f,omet),(cam_add(cam_add(1.f,cam_mul(1.94774f,et)),cam_mul(4.73486f,et2)))),rsqti);
// MPAS module_ra_cam.F:4302
sqwp=cam_sqrt(plco2(i,k1));
// MPAS module_ra_cam.F:4303
f1sqwp=cam_mul(f1co2,sqwp);
// MPAS module_ra_cam.F:4304
t1co2=cam_div(1.f,(cam_add(1.f,cam_mul(cam_mul(cam_mul(245.18f,omet),sqwp),rsqti))));
// MPAS module_ra_cam.F:4305
oneme=cam_sub(1.f,et2);
// MPAS module_ra_cam.F:4306
alphat=cam_mul(cam_pow(oneme,3),rsqti);
// MPAS module_ra_cam.F:4307
wco2=cam_mul(cam_mul(cam_mul(2.5221f,co2vmr),pnm(i,k1)),rga);
// MPAS module_ra_cam.F:4308
u7=cam_mul(cam_mul(cam_mul(4.9411e4f,alphat),et2),wco2);
// MPAS module_ra_cam.F:4309
u8=cam_mul(cam_mul(cam_mul(3.9744e4f,alphat),et4),wco2);
// MPAS module_ra_cam.F:4310
u9=cam_mul(cam_mul(cam_mul(cam_mul(1.0447e5f,alphat),et4),et2),wco2);
// MPAS module_ra_cam.F:4311
u13=cam_mul(cam_mul(cam_mul(2.8388e3f,alphat),et4),wco2);
// MPAS module_ra_cam.F:4313
tpath=co2t(i,k1);
// MPAS module_ra_cam.F:4314
tlocal=tplnke(i);
// MPAS module_ra_cam.F:4315
tcrfac=cam_sqrt(cam_mul((cam_mul(tlocal,r250)),(cam_mul(tpath,r300))));
// MPAS module_ra_cam.F:4316
pi=cam_add(cam_mul(pnm(i,k1),rsslp),cam_mul(cam_mul(2.f,dpfco2),tcrfac));
// MPAS module_ra_cam.F:4317
posqt=cam_div(pi,(cam_mul(2.f,sqti)));
// MPAS module_ra_cam.F:4318
rbeta7=cam_div(1.f,(cam_mul(5.3288f,posqt)));
// MPAS module_ra_cam.F:4319
rbeta8=cam_div(1.f,(cam_mul(10.6576f,posqt)));
// MPAS module_ra_cam.F:4320
rbeta9=rbeta7;
// MPAS module_ra_cam.F:4321
rbeta13=rbeta9;
// MPAS module_ra_cam.F:4322
f2co2=cam_add(cam_add((cam_div(u7,cam_sqrt(cam_add(4.f,cam_mul(u7,(cam_add(1.f,rbeta7))))))),(cam_div(u8,cam_sqrt(cam_add(4.f,cam_mul(u8,(cam_add(1.f,rbeta8)))))))),(cam_div(u9,cam_sqrt(cam_add(4.f,cam_mul(u9,(cam_add(1.f,rbeta9))))))));
// MPAS module_ra_cam.F:4325
f3co2=cam_div(u13,cam_sqrt(cam_add(4.f,cam_mul(u13,(cam_add(1.f,rbeta13))))));
// MPAS module_ra_cam.F:4326
tmp1=cam_log(cam_add(1.f,f1sqwp));
// MPAS module_ra_cam.F:4327
tmp2=cam_log(cam_add(1.f,f2co2));
// MPAS module_ra_cam.F:4328
tmp3=cam_log(cam_add(1.f,f3co2));
// MPAS module_ra_cam.F:4329
absbnd=cam_mul((cam_add(cam_add(tmp1,cam_mul(cam_mul(2.f,t1co2),tmp2)),cam_mul(2.f,tmp3))),sqti);
// MPAS module_ra_cam.F:4330
tco2(i)=cam_div(1.0f,(cam_add(1.0f,cam_mul(10.0f,(cam_div(u7,cam_sqrt(cam_add(4.f,cam_mul(u7,(cam_add(1.f,rbeta7)))))))))));
// MPAS module_ra_cam.F:4331
co2ems(i,k1)=cam_mul(cam_mul(troco2(i,k1),absbnd),co2plk(i));
// MPAS module_ra_cam.F:4332
ex=cam_exp(cam_div(960.f,tint(i,k1)));
// MPAS module_ra_cam.F:4333
exm1sq=cam_pow((cam_sub(ex,1.f)),2);
// MPAS module_ra_cam.F:4334
co2em(i,k1)=cam_div(cam_mul(1.2e11f,ex),(cam_mul(cam_mul(tint(i,k1),tint4(i,k1)),exm1sq)));
// MPAS module_ra_cam.F:4335
}
// MPAS module_ra_cam.F:4339
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4340
h2otr(i,k1)=cam_exp(cam_neg(cam_mul(12.f,s2c(i,k1))));
// MPAS module_ra_cam.F:4341
h2otr(i,k1)=cam_mul(h2otr(i,k1),aer_trn_ttl(i,k1,1,idx_lw_1000_1200));
// MPAS module_ra_cam.F:4342
te=cam_pow((cam_div(co2t(i,k1),293.f)),.7f);
// MPAS module_ra_cam.F:4343
u1=cam_div(cam_mul(18.29f,plos(i,k1)),te);
// MPAS module_ra_cam.F:4344
u2=cam_div(cam_mul(.5649f,plos(i,k1)),te);
// MPAS module_ra_cam.F:4345
phat=cam_div(plos(i,k1),plol(i,k1));
// MPAS module_ra_cam.F:4346
tlocal=tplnke(i);
// MPAS module_ra_cam.F:4347
tcrfac=cam_mul(cam_sqrt(cam_mul(tlocal,r250)),te);
// MPAS module_ra_cam.F:4348
beta=cam_mul((cam_div(1.f,.3205f)),(cam_add((cam_div(1.f,phat)),(cam_mul(dpfo3,tcrfac)))));
// MPAS module_ra_cam.F:4349
realnu=cam_mul((cam_div(1.f,beta)),te);
// MPAS module_ra_cam.F:4350
o3bndi=cam_mul(cam_mul(cam_mul(74.f,te),(cam_div(tplnke(i),375.f))),cam_log(cam_add(cam_add(1.f,fo3(u1,realnu)),fo3(u2,realnu))));
// MPAS module_ra_cam.F:4351
o3ems(i,k1)=cam_mul(cam_mul(dbvtt(i),h2otr(i,k1)),o3bndi);
// MPAS module_ra_cam.F:4352
to3(i)=cam_div(1.0f,(cam_add(cam_add(1.f,cam_mul(0.1f,fo3(u1,realnu))),cam_mul(0.1f,fo3(u2,realnu)))));
// MPAS module_ra_cam.F:4353
}
// MPAS module_ra_cam.F:4357
trcems(lchnk,ncol,pcols,pverp,k1,co2t,pnm,ucfc11,ucfc12,un2o0,un2o1,bn2o0,bn2o1,uch4,bch4,uco211,uco212,uco213,uco221,uco222,uco223,uptype,w,s2c,u,emplnk,th2o,tco2,to3,emstrc,aer_trn_ttl); if(error) return;
// MPAS module_ra_cam.F:4368
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4369
emstot(i,k1)=cam_add(cam_add(cam_add(h2oems(i,k1),co2ems(i,k1)),o3ems(i,k1)),emstrc(i,k1));
// MPAS module_ra_cam.F:4371
}
// MPAS module_ra_cam.F:4372
}
// MPAS module_ra_cam.F:4374
return;
}
// MPAS module_ra_cam.F:4377-4562
CAM_ROUTINE void radtpl(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _tnm,const CamArray<double>& _lwupcgs,const CamArray<double>& _qnm,const CamArray<double>& _pnm,const CamArray<double>& _plco2,const CamArray<double>& _plh2o,const CamArray<double>& _tplnka,const CamArray<double>& _s2c,const CamArray<double>& _tcg,const CamArray<double>& _w,const CamArray<double>& _tplnke,const CamArray<double>& _tint,const CamArray<double>& _tint4,const CamArray<double>& _tlayr,const CamArray<double>& _tlayr4,const CamArray<double>& _pmln,const CamArray<double>& _piln,const CamArray<double>& _plh2ob,const CamArray<double>& _wb) {
CamFrame _frame(arena);
CAM_TRACE("radtpl.tnm");
auto tnm=_tnm.rebound(1,pcols,1,pver);
CAM_TRACE("radtpl.lwupcgs");
auto lwupcgs=_lwupcgs.rebound(1,pcols);
CAM_TRACE("radtpl.qnm");
auto qnm=_qnm.rebound(1,pcols,1,pver);
CAM_TRACE("radtpl.pnm");
auto pnm=_pnm.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.plco2");
auto plco2=_plco2.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.plh2o");
auto plh2o=_plh2o.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.tplnka");
auto tplnka=_tplnka.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.s2c");
auto s2c=_s2c.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.tcg");
auto tcg=_tcg.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.w");
auto w=_w.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.tplnke");
auto tplnke=_tplnke.rebound(1,pcols);
CAM_TRACE("radtpl.tint");
auto tint=_tint.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.tint4");
auto tint4=_tint4.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.tlayr");
auto tlayr=_tlayr.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.tlayr4");
auto tlayr4=_tlayr4.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.pmln");
auto pmln=_pmln.rebound(1,pcols,1,pver);
CAM_TRACE("radtpl.piln");
auto piln=_piln.rebound(1,pcols,1,pverp);
CAM_TRACE("radtpl.plh2ob");
auto plh2ob=_plh2ob.rebound(1,nbands,1,pcols,1,pverp);
CAM_TRACE("radtpl.wb");
auto wb=_wb.rebound(1,nbands,1,pcols,1,pverp);
int i;
int k;
int kp1;
double repsil;
double dy;
double dpnm;
double dpnmsq;
double dw;
double dplh2o;
double cpwpl;
// MPAS module_ra_cam.F:4447
repsil=cam_div(1.f,epsilo);
// MPAS module_ra_cam.F:4451
cpwpl=cam_div(cam_mul(cam_div(amco2,amd),0.5f),(cam_mul(gravit,p0)));
// MPAS module_ra_cam.F:4452
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4453
plh2o(i,ntoplw)=cam_mul(cam_mul(cam_mul(rgsslp,qnm(i,ntoplw)),pnm(i,ntoplw)),pnm(i,ntoplw));
// MPAS module_ra_cam.F:4454
plco2(i,ntoplw)=cam_mul(cam_mul(cam_mul(co2vmr,cpwpl),pnm(i,ntoplw)),pnm(i,ntoplw));
// MPAS module_ra_cam.F:4455
}
// MPAS module_ra_cam.F:4456
for(k=ntoplw;k<=pver;k+=1) {
// MPAS module_ra_cam.F:4457
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4458
plh2o(i,cam_add(k,1))=cam_add(plh2o(i,k),cam_mul(cam_mul(rgsslp,(cam_sub(cam_pow(pnm(i,cam_add(k,1)),2),cam_pow(pnm(i,k),2)))),qnm(i,k)));
// MPAS module_ra_cam.F:4460
plco2(i,cam_add(k,1))=cam_mul(cam_mul(co2vmr,cpwpl),cam_pow(pnm(i,cam_add(k,1)),2));
// MPAS module_ra_cam.F:4461
}
// MPAS module_ra_cam.F:4462
}
// MPAS module_ra_cam.F:4470
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4471
tint4(i,pverp)=cam_div(lwupcgs(i),stebol);
// MPAS module_ra_cam.F:4472
tint(i,pverp)=cam_sqrt(cam_sqrt(tint4(i,pverp)));
// MPAS module_ra_cam.F:4473
tplnka(i,ntoplw)=tnm(i,ntoplw);
// MPAS module_ra_cam.F:4474
tint(i,ntoplw)=tplnka(i,ntoplw);
// MPAS module_ra_cam.F:4475
tlayr4(i,ntoplw)=cam_pow(tplnka(i,ntoplw),4);
// MPAS module_ra_cam.F:4476
tint4(i,ntoplw)=tlayr4(i,ntoplw);
// MPAS module_ra_cam.F:4477
}
// MPAS module_ra_cam.F:4482
for(k=cam_add(ntoplw,1);k<=pver;k+=1) {
// MPAS module_ra_cam.F:4483
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4484
dy=cam_div((cam_sub(piln(i,k),pmln(i,k))),(cam_sub(pmln(i,cam_sub(k,1)),pmln(i,k))));
// MPAS module_ra_cam.F:4485
tint(i,k)=cam_sub(tnm(i,k),cam_mul(dy,(cam_sub(tnm(i,k),tnm(i,cam_sub(k,1))))));
// MPAS module_ra_cam.F:4486
tint4(i,k)=cam_pow(tint(i,k),4);
// MPAS module_ra_cam.F:4487
}
// MPAS module_ra_cam.F:4488
}
// MPAS module_ra_cam.F:4495
for(k=cam_add(ntoplw,1);k<=pverp;k+=1) {
// MPAS module_ra_cam.F:4496
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4497
tlayr(i,k)=tnm(i,cam_sub(k,1));
// MPAS module_ra_cam.F:4498
tlayr4(i,k)=cam_pow(tlayr(i,k),4);
// MPAS module_ra_cam.F:4499
tplnka(i,k)=cam_mul(.5f,(cam_add(tint(i,k),tint(i,cam_sub(k,1)))));
// MPAS module_ra_cam.F:4500
}
// MPAS module_ra_cam.F:4501
}
// MPAS module_ra_cam.F:4506
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4507
tplnke(i)=tplnka(i,ntoplw);
// MPAS module_ra_cam.F:4508
tlayr(i,ntoplw)=tint(i,ntoplw);
// MPAS module_ra_cam.F:4509
}
// MPAS module_ra_cam.F:4513
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4517
tcg(i,ntoplw)=cam_mul(cam_mul(cam_mul(rga,qnm(i,ntoplw)),pnm(i,ntoplw)),tnm(i,ntoplw));
// MPAS module_ra_cam.F:4518
w(i,ntoplw)=cam_div(cam_mul(sslp,(cam_mul(plh2o(i,ntoplw),2.f))),pnm(i,ntoplw));
// MPAS module_ra_cam.F:4522
wb(1,i,ntoplw)=cam_mul(w(i,ntoplw),phi(tnm(i,ntoplw),1));
// MPAS module_ra_cam.F:4523
wb(2,i,ntoplw)=cam_mul(w(i,ntoplw),phi(tnm(i,ntoplw),2));
// MPAS module_ra_cam.F:4527
plh2ob(1,i,ntoplw)=cam_mul(plh2o(i,ntoplw),psi(tnm(i,ntoplw),1));
// MPAS module_ra_cam.F:4528
plh2ob(2,i,ntoplw)=cam_mul(plh2o(i,ntoplw),psi(tnm(i,ntoplw),2));
// MPAS module_ra_cam.F:4530
s2c(i,ntoplw)=cam_mul(cam_mul(cam_mul(plh2o(i,ntoplw),fh2oself(tnm(i,ntoplw))),qnm(i,ntoplw)),repsil);
// MPAS module_ra_cam.F:4531
}
// MPAS module_ra_cam.F:4533
for(k=ntoplw;k<=pver;k+=1) {
// MPAS module_ra_cam.F:4534
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4535
dpnm=cam_sub(pnm(i,cam_add(k,1)),pnm(i,k));
// MPAS module_ra_cam.F:4536
dpnmsq=cam_sub(cam_pow(pnm(i,cam_add(k,1)),2),cam_pow(pnm(i,k),2));
// MPAS module_ra_cam.F:4537
dw=cam_mul(cam_mul(rga,qnm(i,k)),dpnm);
// MPAS module_ra_cam.F:4538
kp1=cam_add(k,1);
// MPAS module_ra_cam.F:4539
w(i,kp1)=cam_add(w(i,k),dw);
// MPAS module_ra_cam.F:4543
wb(1,i,kp1)=cam_add(wb(1,i,k),cam_mul(dw,phi(tnm(i,k),1)));
// MPAS module_ra_cam.F:4544
wb(2,i,kp1)=cam_add(wb(2,i,k),cam_mul(dw,phi(tnm(i,k),2)));
// MPAS module_ra_cam.F:4548
dplh2o=cam_sub(plh2o(i,kp1),plh2o(i,k));
// MPAS module_ra_cam.F:4550
plh2ob(1,i,kp1)=cam_add(plh2ob(1,i,k),cam_mul(dplh2o,psi(tnm(i,k),1)));
// MPAS module_ra_cam.F:4551
plh2ob(2,i,kp1)=cam_add(plh2ob(2,i,k),cam_mul(dplh2o,psi(tnm(i,k),2)));
// MPAS module_ra_cam.F:4555
tcg(i,kp1)=cam_add(tcg(i,k),cam_mul(dw,tnm(i,k)));
// MPAS module_ra_cam.F:4556
s2c(i,kp1)=cam_add(s2c(i,k),cam_mul(cam_mul(cam_mul(cam_mul(cam_mul(rgsslp,dpnmsq),qnm(i,k)),fh2oself(tnm(i,k))),qnm(i,k)),repsil));
// MPAS module_ra_cam.F:4558
}
// MPAS module_ra_cam.F:4559
}
// MPAS module_ra_cam.F:4561
return;
}
// MPAS module_ra_cam.F:4565-5512
CAM_ROUTINE void radclwmx(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _lwupcgs,const CamArray<double>& _tnm,const CamArray<double>& _qnm,const CamArray<double>& _o3vmr,const CamArray<double>& _pmid,const CamArray<double>& _pint,const CamArray<double>& _pmln,const CamArray<double>& _piln,const CamArray<double>& _n2o,const CamArray<double>& _ch4,const CamArray<double>& _cfc11,const CamArray<double>& _cfc12,const CamArray<double>& _cld,const CamArray<double>& _emis,const CamArray<double>& _pmxrgn,const CamArray<int>& _nmxrgn,const CamArray<double>& _qrl,const bool& doabsems,const CamArray<double>& _abstot,const CamArray<double>& _absnxt,const CamArray<double>& _emstot,const CamArray<double>& _flns,const CamArray<double>& _flnt,const CamArray<double>& _flnsc,const CamArray<double>& _flntc,const CamArray<double>& _flwds,const CamArray<double>& _flut,const CamArray<double>& _flutc,const CamArray<double>& _flup,const CamArray<double>& _flupc,const CamArray<double>& _fldn,const CamArray<double>& _fldnc,const CamArray<double>& _aer_mass) {
CamFrame _frame(arena);
const double cldmin=1.0e-80;
CAM_TRACE("radclwmx.lwupcgs");
auto lwupcgs=_lwupcgs.rebound(1,pcols);
CAM_TRACE("radclwmx.tnm");
auto tnm=_tnm.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.qnm");
auto qnm=_qnm.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.o3vmr");
auto o3vmr=_o3vmr.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("radclwmx.pmln");
auto pmln=_pmln.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.piln");
auto piln=_piln.rebound(1,pcols,1,pverp);
CAM_TRACE("radclwmx.n2o");
auto n2o=_n2o.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.ch4");
auto ch4=_ch4.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.cfc11");
auto cfc11=_cfc11.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.cfc12");
auto cfc12=_cfc12.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.cld");
auto cld=_cld.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.emis");
auto emis=_emis.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.pmxrgn");
auto pmxrgn=_pmxrgn.rebound(1,pcols,1,pverp);
CAM_TRACE("radclwmx.nmxrgn");
auto nmxrgn=_nmxrgn.rebound(1,pcols);
CAM_TRACE("radclwmx.qrl");
auto qrl=_qrl.rebound(1,pcols,1,pver);
CAM_TRACE("radclwmx.abstot");
auto abstot=_abstot.rebound(1,pcols,1,pverp,1,pverp);
CAM_TRACE("radclwmx.absnxt");
auto absnxt=_absnxt.rebound(1,pcols,1,pver,1,4);
CAM_TRACE("radclwmx.emstot");
auto emstot=_emstot.rebound(1,pcols,1,pverp);
CAM_TRACE("radclwmx.flns");
auto flns=_flns.rebound(1,pcols);
CAM_TRACE("radclwmx.flnt");
auto flnt=_flnt.rebound(1,pcols);
CAM_TRACE("radclwmx.flnsc");
auto flnsc=_flnsc.rebound(1,pcols);
CAM_TRACE("radclwmx.flntc");
auto flntc=_flntc.rebound(1,pcols);
CAM_TRACE("radclwmx.flwds");
auto flwds=_flwds.rebound(1,pcols);
CAM_TRACE("radclwmx.flut");
auto flut=_flut.rebound(1,pcols);
CAM_TRACE("radclwmx.flutc");
auto flutc=_flutc.rebound(1,pcols);
CAM_TRACE("radclwmx.flup");
auto flup=_flup.rebound(1,pcols,1,pverp);
CAM_TRACE("radclwmx.flupc");
auto flupc=_flupc.rebound(1,pcols,1,pverp);
CAM_TRACE("radclwmx.fldn");
auto fldn=_fldn.rebound(1,pcols,1,pverp);
CAM_TRACE("radclwmx.fldnc");
auto fldnc=_fldnc.rebound(1,pcols,1,pverp);
CAM_TRACE("radclwmx.aer_mass");
auto aer_mass=_aer_mass.rebound(1,pcols,1,pver);
int pverp2;
int pverp3;
int pverp4;
int i;
int ilon;
int ii;
int iimx;
int k;
int k1;
int k2;
int k3;
int km;
int km1;
int km3;
int km4;
int irgn;
int l;
int l1;
int n;
auto plco2=arena.array<double>(1,pcols,1,pverp);
auto plh2o=arena.array<double>(1,pcols,1,pverp);
auto tmp=arena.array<double>(1,pcols);
auto tmp2=arena.array<double>(1,pcols);
auto absbt=arena.array<double>(1,pcols);
auto plol=arena.array<double>(1,pcols,1,pverp);
auto plos=arena.array<double>(1,pcols,1,pverp);
auto aer_mpp=arena.array<double>(1,pcols,1,pverp);
auto co2em=arena.array<double>(1,pcols,1,pverp);
auto co2eml=arena.array<double>(1,pcols,1,pver);
auto delt=arena.array<double>(1,pcols);
auto delt1=arena.array<double>(1,pcols);
auto bk1=arena.array<double>(1,pcols);
auto bk2=arena.array<double>(1,pcols);
auto cldp=arena.array<double>(1,pcols,1,pverp);
auto ful=arena.array<double>(1,pcols,1,pverp);
auto fsul=arena.array<double>(1,pcols,1,pverp);
auto fdl=arena.array<double>(1,pcols,1,pverp);
auto fsdl=arena.array<double>(1,pcols,1,pverp);
auto fclb4=arena.array<double>(1,pcols,cam_neg(1),pver);
auto fclt4=arena.array<double>(1,pcols,0,pver);
auto s=arena.array<double>(1,pcols,1,pverp,1,pverp);
auto tplnka=arena.array<double>(1,pcols,1,pverp);
auto s2c=arena.array<double>(1,pcols,1,pverp);
auto tcg=arena.array<double>(1,pcols,1,pverp);
auto w=arena.array<double>(1,pcols,1,pverp);
auto tplnke=arena.array<double>(1,pcols);
auto h2otr=arena.array<double>(1,pcols,1,pverp);
auto co2t=arena.array<double>(1,pcols,1,pverp);
auto tint=arena.array<double>(1,pcols,1,pverp);
auto tint4=arena.array<double>(1,pcols,1,pverp);
auto tlayr=arena.array<double>(1,pcols,1,pverp);
auto tlayr4=arena.array<double>(1,pcols,1,pverp);
auto plh2ob=arena.array<double>(1,nbands,1,pcols,1,pverp);
auto wb=arena.array<double>(1,nbands,1,pcols,1,pverp);
double cld0;
double cld1;
auto emx=arena.array<double>(0,pverp);
double emx0;
double trans;
auto asort=arena.array<double>(1,pver);
double atmp;
auto maxcld=arena.array<double>(1,pcols);
auto indx=arena.array<int>(1,pcols);
auto indxmx=arena.array<int>(1,pcols,1,pverp);
auto nrgn=arena.array<int>(1,pcols);
int npts;
auto ncolmx=arena.array<int>(1,pverp);
auto kx1=arena.array<int>(1,pcols,1,pverp);
auto kx2=arena.array<int>(1,pcols,0,pverp);
auto kxs=arena.array<int>(0,pverp,1,pcols,1,pverp);
auto nxs=arena.array<int>(1,pcols,1,pverp);
int nxsk;
auto ksort=arena.array<int>(0,pverp);
int ktmp;
auto aer_trn_ttl=arena.array<double>(1,pcols,1,pverp,1,pverp,1,bnd_nbr_lw);
auto ucfc11=arena.array<double>(1,pcols,1,pverp);
auto ucfc12=arena.array<double>(1,pcols,1,pverp);
auto un2o0=arena.array<double>(1,pcols,1,pverp);
auto un2o1=arena.array<double>(1,pcols,1,pverp);
auto uch4=arena.array<double>(1,pcols,1,pverp);
auto uco211=arena.array<double>(1,pcols,1,pverp);
auto uco212=arena.array<double>(1,pcols,1,pverp);
auto uco213=arena.array<double>(1,pcols,1,pverp);
auto uco221=arena.array<double>(1,pcols,1,pverp);
auto uco222=arena.array<double>(1,pcols,1,pverp);
auto uco223=arena.array<double>(1,pcols,1,pverp);
auto bn2o0=arena.array<double>(1,pcols,1,pverp);
auto bn2o1=arena.array<double>(1,pcols,1,pverp);
auto bch4=arena.array<double>(1,pcols,1,pverp);
auto uptype=arena.array<double>(1,pcols,1,pverp);
auto abplnk1=arena.array<double>(1,14,1,pcols,1,pverp);
auto abplnk2=arena.array<double>(1,14,1,pcols,1,pverp);
// MPAS module_ra_cam.F:4786
pverp2=cam_add(pver,2);
// MPAS module_ra_cam.F:4787
pverp3=cam_add(pver,3);
// MPAS module_ra_cam.F:4788
pverp4=cam_add(pver,4);
// MPAS module_ra_cam.F:4798
aer_pth(aer_mass,aer_mpp,ncol,pcols,pver,pverp); if(error) return;
// MPAS module_ra_cam.F:4804
radtpl(lchnk,ncol,pcols,pver,pverp,tnm,lwupcgs,qnm,pint,plco2,plh2o,tplnka,s2c,tcg,w,tplnke,tint,tint4,tlayr,tlayr4,pmln,piln,plh2ob,wb); if(error) return;
// MPAS module_ra_cam.F:4809
if(doabsems) {
// MPAS module_ra_cam.F:4813
radoz2(lchnk,ncol,pcols,pver,pverp,o3vmr,pint,plol,plos,ntoplw); if(error) return;
// MPAS module_ra_cam.F:4817
trcpth(lchnk,ncol,pcols,pver,pverp,tnm,pint,cfc11,cfc12,n2o,ch4,qnm,ucfc11,ucfc12,un2o0,un2o1,uch4,uco211,uco212,uco213,uco221,uco222,uco223,bn2o0,bn2o1,bch4,uptype); if(error) return;
// MPAS module_ra_cam.F:4825
aer_trn(aer_mpp,aer_trn_ttl,pcols,pver,pverp); if(error) return;
// MPAS module_ra_cam.F:4831
radems(lchnk,ncol,pcols,pver,pverp,s2c,tcg,w,tplnke,plh2o,pint,plco2,tint,tint4,tlayr,tlayr4,plol,plos,ucfc11,ucfc12,un2o0,un2o1,uch4,uco211,uco212,uco213,uco221,uco222,uco223,uptype,bn2o0,bn2o1,bch4,co2em,co2eml,co2t,h2otr,abplnk1,abplnk2,emstot,plh2ob,wb,aer_trn_ttl); if(error) return;
// MPAS module_ra_cam.F:4844
radabs(lchnk,ncol,pcols,pver,pverp,pmid,pint,co2em,co2eml,tplnka,s2c,tcg,w,h2otr,plco2,plh2o,co2t,tint,tlayr,plol,plos,pmln,piln,ucfc11,ucfc12,un2o0,un2o1,uch4,uco211,uco212,uco213,uco221,uco222,uco223,uptype,bn2o0,bn2o1,bch4,abplnk1,abplnk2,abstot,absnxt,plh2ob,wb,aer_mpp,aer_trn_ttl); if(error) return;
// MPAS module_ra_cam.F:4854
}
// MPAS module_ra_cam.F:4865
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4866
delt(i)=cam_sub(tint4(i,pver),tlayr4(i,pverp));
// MPAS module_ra_cam.F:4867
delt1(i)=cam_sub(tlayr4(i,pverp),tint4(i,pverp));
// MPAS module_ra_cam.F:4868
s(i,pverp,pverp)=cam_mul(stebol,(cam_add(cam_mul(delt1(i),absnxt(i,pver,1)),cam_mul(delt(i),absnxt(i,pver,4)))));
// MPAS module_ra_cam.F:4869
s(i,pver,pverp)=cam_mul(stebol,(cam_add(cam_mul(delt(i),absnxt(i,pver,2)),cam_mul(delt1(i),absnxt(i,pver,3)))));
// MPAS module_ra_cam.F:4870
}
// MPAS module_ra_cam.F:4871
for(k=ntoplw;k<=cam_sub(pver,1);k+=1) {
// MPAS module_ra_cam.F:4872
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4873
bk2(i)=cam_mul((cam_add(abstot(i,k,pver),abstot(i,k,pverp))),0.5f);
// MPAS module_ra_cam.F:4874
bk1(i)=bk2(i);
// MPAS module_ra_cam.F:4875
s(i,k,pverp)=cam_mul(stebol,(cam_add(cam_mul(bk2(i),delt(i)),cam_mul(bk1(i),delt1(i)))));
// MPAS module_ra_cam.F:4876
}
// MPAS module_ra_cam.F:4877
}
// MPAS module_ra_cam.F:4881
for(km=pver;km>=cam_add(ntoplw,1);km+=cam_neg(1)) {
// MPAS module_ra_cam.F:4882
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4883
delt(i)=cam_sub(tint4(i,cam_sub(km,1)),tlayr4(i,km));
// MPAS module_ra_cam.F:4884
delt1(i)=cam_sub(tlayr4(i,km),tint4(i,km));
// MPAS module_ra_cam.F:4885
}
// MPAS module_ra_cam.F:4886
for(k=pverp;k>=ntoplw;k+=cam_neg(1)) {
// MPAS module_ra_cam.F:4887
if((k==km)) {
// MPAS module_ra_cam.F:4888
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4889
bk2(i)=absnxt(i,cam_sub(km,1),4);
// MPAS module_ra_cam.F:4890
bk1(i)=absnxt(i,cam_sub(km,1),1);
// MPAS module_ra_cam.F:4891
}
// MPAS module_ra_cam.F:4892
} else if((k==cam_sub(km,1))) {
// MPAS module_ra_cam.F:4893
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4894
bk2(i)=absnxt(i,cam_sub(km,1),2);
// MPAS module_ra_cam.F:4895
bk1(i)=absnxt(i,cam_sub(km,1),3);
// MPAS module_ra_cam.F:4896
}
// MPAS module_ra_cam.F:4897
} else {
// MPAS module_ra_cam.F:4898
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4899
bk2(i)=cam_mul((cam_add(abstot(i,k,cam_sub(km,1)),abstot(i,k,km))),0.5f);
// MPAS module_ra_cam.F:4900
bk1(i)=bk2(i);
// MPAS module_ra_cam.F:4901
}
// MPAS module_ra_cam.F:4902
}
// MPAS module_ra_cam.F:4903
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4904
s(i,k,km)=cam_add(s(i,k,cam_add(km,1)),cam_mul(stebol,(cam_add(cam_mul(bk2(i),delt(i)),cam_mul(bk1(i),delt1(i))))));
// MPAS module_ra_cam.F:4905
}
// MPAS module_ra_cam.F:4906
}
// MPAS module_ra_cam.F:4907
}
// MPAS module_ra_cam.F:4911
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4912
fsul(i,pverp)=lwupcgs(i);
// MPAS module_ra_cam.F:4913
}
// MPAS module_ra_cam.F:4918
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4919
tmp(i)=cam_sub(fsul(i,pverp),cam_mul(stebol,tint4(i,pverp)));
// MPAS module_ra_cam.F:4920
fsul(i,ntoplw)=cam_add(cam_sub(fsul(i,pverp),cam_mul(abstot(i,ntoplw,pverp),tmp(i))),s(i,ntoplw,cam_add(ntoplw,1)));
// MPAS module_ra_cam.F:4921
fsdl(i,ntoplw)=cam_mul(cam_mul(stebol,(cam_pow(tplnke(i),4))),emstot(i,ntoplw));
// MPAS module_ra_cam.F:4922
}
// MPAS module_ra_cam.F:4926
for(k=cam_add(ntoplw,1);k<=pver;k+=1) {
// MPAS module_ra_cam.F:4927
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4928
fsul(i,k)=cam_add(cam_sub(fsul(i,pverp),cam_mul(abstot(i,k,pverp),tmp(i))),s(i,k,cam_add(k,1)));
// MPAS module_ra_cam.F:4929
fsdl(i,k)=cam_sub(cam_mul(cam_mul(stebol,(cam_pow(tplnke(i),4))),emstot(i,k)),(cam_sub(s(i,k,cam_add(ntoplw,1)),s(i,k,cam_add(k,1)))));
// MPAS module_ra_cam.F:4930
}
// MPAS module_ra_cam.F:4931
}
// MPAS module_ra_cam.F:4936
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4937
absbt(i)=cam_mul(cam_mul(stebol,(cam_pow(tplnke(i),4))),emstot(i,pverp));
// MPAS module_ra_cam.F:4938
fsdl(i,pverp)=cam_sub(absbt(i),s(i,pverp,cam_add(ntoplw,1)));
// MPAS module_ra_cam.F:4939
}
// MPAS module_ra_cam.F:4963
{ auto _lhs=cldp.section(cam_section(1,ncol),cam_section(ntoplw,pver)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cld.section(cam_section(1,ncol),cam_section(ntoplw,pver)).linear(_slice); }
// MPAS module_ra_cam.F:4964
{ auto _lhs=cldp.section(cam_section(1,ncol),pverp); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.0f; }
// MPAS module_ra_cam.F:4971
{ auto _lhs=maxcld.section(cam_section(1,ncol)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_maxval(cldp.section(cam_section(1,ncol),cam_section(ntoplw,pver)),2,_slice); }
// MPAS module_ra_cam.F:4973
npts=0;
// MPAS module_ra_cam.F:4974
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4975
if((maxcld(i)<cldmin)) {
// MPAS module_ra_cam.F:4976
npts=cam_add(npts,1);
// MPAS module_ra_cam.F:4977
indx(npts)=i;
// MPAS module_ra_cam.F:4978
}
// MPAS module_ra_cam.F:4979
}
// MPAS module_ra_cam.F:4981
for(ii=1;ii<=npts;ii+=1) {
// MPAS module_ra_cam.F:4982
i=indx(ii);
// MPAS module_ra_cam.F:4983
for(k=ntoplw;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:4984
fdl(i,k)=fsdl(i,k);
// MPAS module_ra_cam.F:4985
ful(i,k)=fsul(i,k);
// MPAS module_ra_cam.F:4986
}
// MPAS module_ra_cam.F:4987
}
// MPAS module_ra_cam.F:4991
npts=0;
// MPAS module_ra_cam.F:4992
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:4993
if((maxcld(i)>=cldmin)) {
// MPAS module_ra_cam.F:4994
npts=cam_add(npts,1);
// MPAS module_ra_cam.F:4995
indx(npts)=i;
// MPAS module_ra_cam.F:4996
}
// MPAS module_ra_cam.F:4997
}
// MPAS module_ra_cam.F:5002
for(ii=1;ii<=npts;ii+=1) {
// MPAS module_ra_cam.F:5003
i=indx(ii);
// MPAS module_ra_cam.F:5004
fdl(i,ntoplw)=fsdl(i,ntoplw);
// MPAS module_ra_cam.F:5005
fdl(i,pverp)=0.0f;
// MPAS module_ra_cam.F:5006
ful(i,ntoplw)=0.0f;
// MPAS module_ra_cam.F:5007
ful(i,pverp)=fsul(i,pverp);
// MPAS module_ra_cam.F:5008
for(k=cam_add(ntoplw,1);k<=pver;k+=1) {
// MPAS module_ra_cam.F:5009
fdl(i,k)=0.0f;
// MPAS module_ra_cam.F:5010
ful(i,k)=0.0f;
// MPAS module_ra_cam.F:5011
}
// MPAS module_ra_cam.F:5015
for(k=ntoplw;k<=pver;k+=1) {
// MPAS module_ra_cam.F:5016
fclt4(i,cam_sub(k,1))=cam_mul(stebol,tint4(i,k));
// MPAS module_ra_cam.F:5017
fclb4(i,cam_sub(k,1))=cam_mul(stebol,tint4(i,cam_add(k,1)));
// MPAS module_ra_cam.F:5018
}
// MPAS module_ra_cam.F:5019
fclb4(i,cam_sub(ntoplw,2))=cam_mul(stebol,tint4(i,ntoplw));
// MPAS module_ra_cam.F:5020
fclt4(i,pver)=cam_mul(stebol,tint4(i,pverp));
// MPAS module_ra_cam.F:5024
for(irgn=0;irgn<=nmxrgn(i);irgn+=1) {
// MPAS module_ra_cam.F:5025
kx2(i,irgn)=cam_sub(ntoplw,1);
// MPAS module_ra_cam.F:5026
}
// MPAS module_ra_cam.F:5027
nrgn(i)=0;
// MPAS module_ra_cam.F:5028
}
// MPAS module_ra_cam.F:5033
for(ii=1;ii<=npts;ii+=1) {
// MPAS module_ra_cam.F:5034
ilon=indx(ii);
// MPAS module_ra_cam.F:5039
for(irgn=1;irgn<=nmxrgn(ilon);irgn+=1) {
// MPAS module_ra_cam.F:5043
n=0;
// MPAS module_ra_cam.F:5044
if((kx2(ilon,cam_sub(irgn,1))<pver)) {
// MPAS module_ra_cam.F:5045
nrgn(ilon)=irgn;
// MPAS module_ra_cam.F:5046
k1=cam_add(kx2(ilon,cam_sub(irgn,1)),1);
// MPAS module_ra_cam.F:5047
kx1(ilon,irgn)=k1;
// MPAS module_ra_cam.F:5048
kx2(ilon,irgn)=0;
// MPAS module_ra_cam.F:5049
for(k2=pver;k2>=k1;k2+=cam_neg(1)) {
// MPAS module_ra_cam.F:5050
if((pmid(ilon,k2)<=pmxrgn(ilon,irgn))) {
// MPAS module_ra_cam.F:5051
kx2(ilon,irgn)=k2;
// MPAS module_ra_cam.F:5052
break;
// MPAS module_ra_cam.F:5053
}
// MPAS module_ra_cam.F:5054
}
// MPAS module_ra_cam.F:5058
for(k=k1;k<=k2;k+=1) {
// MPAS module_ra_cam.F:5059
if((cldp(ilon,k)>=cldmin)) {
// MPAS module_ra_cam.F:5060
n=cam_add(n,1);
// MPAS module_ra_cam.F:5061
indxmx(n,irgn)=ilon;
// MPAS module_ra_cam.F:5062
break;
// MPAS module_ra_cam.F:5063
}
// MPAS module_ra_cam.F:5064
}
// MPAS module_ra_cam.F:5065
}
// MPAS module_ra_cam.F:5066
ncolmx(irgn)=n;
// MPAS module_ra_cam.F:5074
for(iimx=1;iimx<=ncolmx(irgn);iimx+=1) {
// MPAS module_ra_cam.F:5075
i=indxmx(iimx,irgn);
// MPAS module_ra_cam.F:5079
n=0;
// MPAS module_ra_cam.F:5080
for(k=kx1(i,irgn);k<=kx2(i,irgn);k+=1) {
// MPAS module_ra_cam.F:5081
if((cldp(i,k)>=cldmin)) {
// MPAS module_ra_cam.F:5082
n=cam_add(n,1);
// MPAS module_ra_cam.F:5083
ksort(n)=k;
// MPAS module_ra_cam.F:5088
asort(n)=cam_sub(1.0f,cldp(i,k));
// MPAS module_ra_cam.F:5089
}
// MPAS module_ra_cam.F:5090
}
// MPAS module_ra_cam.F:5091
nxs(i,irgn)=n;
// MPAS module_ra_cam.F:5097
if((nxs(i,irgn)==2)) {
// MPAS module_ra_cam.F:5098
if((asort(2)<asort(1))) {
// MPAS module_ra_cam.F:5099
ktmp=ksort(1);
// MPAS module_ra_cam.F:5100
ksort(1)=ksort(2);
// MPAS module_ra_cam.F:5101
ksort(2)=ktmp;
// MPAS module_ra_cam.F:5103
atmp=asort(1);
// MPAS module_ra_cam.F:5104
asort(1)=asort(2);
// MPAS module_ra_cam.F:5105
asort(2)=atmp;
// MPAS module_ra_cam.F:5106
}
// MPAS module_ra_cam.F:5107
} else if((nxs(i,irgn)>=3)) {
// MPAS module_ra_cam.F:5108
sortarray(nxs(i,irgn),asort,ksort.section(cam_section(1,pverp))); if(error) return;
// MPAS module_ra_cam.F:5109
}
// MPAS module_ra_cam.F:5111
for(l=1;l<=nxs(i,irgn);l+=1) {
// MPAS module_ra_cam.F:5112
kxs(l,i,irgn)=ksort(l);
// MPAS module_ra_cam.F:5113
}
// MPAS module_ra_cam.F:5117
}
// MPAS module_ra_cam.F:5121
}
// MPAS module_ra_cam.F:5127
for(irgn=1;irgn<=nmxrgn(ilon);irgn+=1) {
// MPAS module_ra_cam.F:5131
iimx=1;
// MPAS module_ra_cam.F:5132
if(((ilon<indxmx(iimx,irgn))&&(irgn<=nrgn(ilon)))) {
// MPAS module_ra_cam.F:5140
k1=kx1(ilon,irgn);
// MPAS module_ra_cam.F:5141
for(km1=cam_sub(ntoplw,2);km1<=cam_sub(k1,2);km1+=1) {
// MPAS module_ra_cam.F:5142
km4=cam_add(km1,3);
// MPAS module_ra_cam.F:5143
k2=k1;
// MPAS module_ra_cam.F:5144
k3=cam_add(k2,1);
// MPAS module_ra_cam.F:5145
tmp(ilon)=cam_mul(s(ilon,k2,cam_min(k3,pverp)),cam_min(1,cam_sub(pverp2,k3)));
// MPAS module_ra_cam.F:5146
emx0=cam_div((cam_sub(fdl(ilon,k1),fsdl(ilon,k1))),(cam_sub((cam_add(cam_sub(fclb4(ilon,km1),s(ilon,k2,km4)),tmp(ilon))),fsdl(ilon,k1))));
// MPAS module_ra_cam.F:5148
if(((emx0>=0.0f)&&(emx0<=1.0f))) {break;}
// MPAS module_ra_cam.F:5149
}
// MPAS module_ra_cam.F:5150
km1=cam_min(km1,cam_sub(k1,2));
// MPAS module_ra_cam.F:5151
for(k2=cam_add(kx1(ilon,irgn),1);k2<=cam_add(kx2(ilon,irgn),1);k2+=1) {
// MPAS module_ra_cam.F:5152
k3=cam_add(k2,1);
// MPAS module_ra_cam.F:5153
tmp(ilon)=cam_mul(s(ilon,k2,cam_min(k3,pverp)),cam_min(1,cam_sub(pverp2,k3)));
// MPAS module_ra_cam.F:5154
fdl(ilon,k2)=cam_add(cam_mul((cam_sub(1.0f,emx0)),fsdl(ilon,k2)),cam_mul(emx0,(cam_add(cam_sub(fclb4(ilon,km1),s(ilon,k2,km4)),tmp(ilon)))));
// MPAS module_ra_cam.F:5156
}
// MPAS module_ra_cam.F:5157
} else if(((ilon==indxmx(iimx,irgn))&&(iimx<=ncolmx(irgn)))) {
// MPAS module_ra_cam.F:5158
iimx=cam_add(iimx,1);
// MPAS module_ra_cam.F:5159
}
// MPAS module_ra_cam.F:5163
for(iimx=1;iimx<=ncolmx(irgn);iimx+=1) {
// MPAS module_ra_cam.F:5164
i=indxmx(iimx,irgn);
// MPAS module_ra_cam.F:5173
k1=kx1(i,irgn);
// MPAS module_ra_cam.F:5174
for(km1=cam_sub(ntoplw,2);km1<=cam_sub(k1,2);km1+=1) {
// MPAS module_ra_cam.F:5175
km4=cam_add(km1,3);
// MPAS module_ra_cam.F:5176
k2=k1;
// MPAS module_ra_cam.F:5177
k3=cam_add(k2,1);
// MPAS module_ra_cam.F:5178
tmp(i)=cam_mul(s(i,k2,cam_min(k3,pverp)),cam_min(1,cam_sub(pverp2,k3)));
// MPAS module_ra_cam.F:5179
tmp2(i)=cam_mul(s(i,k2,cam_min(km4,pverp)),cam_min(1,cam_sub(pverp2,km4)));
// MPAS module_ra_cam.F:5180
emx0=cam_div((cam_sub(fdl(i,k1),fsdl(i,k1))),(cam_sub((cam_add(cam_sub(fclb4(i,km1),tmp2(i)),tmp(i))),fsdl(i,k1))));
// MPAS module_ra_cam.F:5181
if(((emx0>=0.0f)&&(emx0<=1.0f))) {break;}
// MPAS module_ra_cam.F:5182
}
// MPAS module_ra_cam.F:5183
km1=cam_min(km1,cam_sub(k1,2));
// MPAS module_ra_cam.F:5184
ksort(0)=cam_add(km1,1);
// MPAS module_ra_cam.F:5188
nxsk=0;
// MPAS module_ra_cam.F:5189
for(k=kx1(i,irgn);k<=kx2(i,irgn);k+=1) {
// MPAS module_ra_cam.F:5195
if((nxsk<nxs(i,irgn))) {
// MPAS module_ra_cam.F:5196
nxsk=0;
// MPAS module_ra_cam.F:5197
for(l=1;l<=nxs(i,irgn);l+=1) {
// MPAS module_ra_cam.F:5198
k1=kxs(l,i,irgn);
// MPAS module_ra_cam.F:5199
if((k>=k1)) {
// MPAS module_ra_cam.F:5200
nxsk=cam_add(nxsk,1);
// MPAS module_ra_cam.F:5201
ksort(nxsk)=k1;
// MPAS module_ra_cam.F:5202
}
// MPAS module_ra_cam.F:5203
}
// MPAS module_ra_cam.F:5204
}
// MPAS module_ra_cam.F:5208
ksort(cam_add(nxsk,1))=pverp;
// MPAS module_ra_cam.F:5212
for(l=1;l<=nxsk;l+=1) {
// MPAS module_ra_cam.F:5213
emx(l)=emis(i,ksort(l));
// MPAS module_ra_cam.F:5214
}
// MPAS module_ra_cam.F:5218
emx(0)=emx0;
// MPAS module_ra_cam.F:5222
cld0=1.0f;
// MPAS module_ra_cam.F:5226
k2=cam_add(k,1);
// MPAS module_ra_cam.F:5227
k3=cam_add(k2,1);
// MPAS module_ra_cam.F:5228
tmp(i)=cam_mul(s(i,k2,cam_min(k3,pverp)),cam_min(1,cam_sub(pverp2,k3)));
// MPAS module_ra_cam.F:5232
for(l=1;l<=cam_add(nxsk,1);l+=1) {
// MPAS module_ra_cam.F:5236
cld1=cam_mul(cldp(i,ksort(l)),cam_min(1,cam_sub(cam_add(nxsk,1),l)));
// MPAS module_ra_cam.F:5237
if((cld0!=cld1)) {
// MPAS module_ra_cam.F:5238
fdl(i,k2)=cam_add(fdl(i,k2),cam_mul((cam_sub(cld0,cld1)),fsdl(i,k2)));
// MPAS module_ra_cam.F:5239
for(l1=0;l1<=cam_sub(l,1);l1+=1) {
// MPAS module_ra_cam.F:5240
km1=cam_sub(ksort(l1),1);
// MPAS module_ra_cam.F:5241
km4=cam_add(km1,3);
// MPAS module_ra_cam.F:5242
tmp2(i)=cam_mul(s(i,k2,cam_min(km4,pverp)),cam_min(1,cam_sub(pverp2,km4)));
// MPAS module_ra_cam.F:5243
fdl(i,k2)=cam_add(fdl(i,k2),cam_mul(cam_mul((cam_sub(cld0,cld1)),emx(l1)),(cam_sub(cam_add(cam_sub(fclb4(i,km1),tmp2(i)),tmp(i)),fsdl(i,k2)))));
// MPAS module_ra_cam.F:5245
}
// MPAS module_ra_cam.F:5246
}
// MPAS module_ra_cam.F:5247
cld0=cld1;
// MPAS module_ra_cam.F:5251
if((l<=nxsk)) {
// MPAS module_ra_cam.F:5252
k1=ksort(l);
// MPAS module_ra_cam.F:5253
trans=cam_sub(1.0f,emis(i,k1));
// MPAS module_ra_cam.F:5258
for(l1=0;l1<=nxsk;l1+=1) {
// MPAS module_ra_cam.F:5259
if((ksort(l1)<k1)) {
// MPAS module_ra_cam.F:5260
emx(l1)=cam_mul(emx(l1),trans);
// MPAS module_ra_cam.F:5261
}
// MPAS module_ra_cam.F:5262
}
// MPAS module_ra_cam.F:5263
}
// MPAS module_ra_cam.F:5267
}
// MPAS module_ra_cam.F:5271
}
// MPAS module_ra_cam.F:5275
}
// MPAS module_ra_cam.F:5279
}
// MPAS module_ra_cam.F:5286
for(irgn=nmxrgn(ilon);irgn>=1;irgn+=cam_neg(1)) {
// MPAS module_ra_cam.F:5290
iimx=1;
// MPAS module_ra_cam.F:5291
if(((ilon<indxmx(iimx,irgn))&&(irgn<=nrgn(ilon)))) {
// MPAS module_ra_cam.F:5301
k1=cam_add(kx2(ilon,irgn),1);
// MPAS module_ra_cam.F:5302
if((k1<pverp)) {
// MPAS module_ra_cam.F:5303
for(km1=cam_sub(pver,1);km1>=kx2(ilon,irgn);km1+=cam_neg(1)) {
// MPAS module_ra_cam.F:5304
km3=cam_add(km1,2);
// MPAS module_ra_cam.F:5305
k2=k1;
// MPAS module_ra_cam.F:5306
k3=cam_add(k2,1);
// MPAS module_ra_cam.F:5307
tmp(ilon)=cam_mul(s(ilon,k2,cam_min(km3,pverp)),cam_min(1,cam_sub(pverp2,km3)));
// MPAS module_ra_cam.F:5308
emx0=cam_div((cam_sub(ful(ilon,k1),fsul(ilon,k1))),(cam_sub((cam_sub(cam_add(fclt4(ilon,km1),s(ilon,k2,k3)),tmp(ilon))),fsul(ilon,k1))));
// MPAS module_ra_cam.F:5310
if(((emx0>=0.0f)&&(emx0<=1.0f))) {break;}
// MPAS module_ra_cam.F:5311
}
// MPAS module_ra_cam.F:5312
km1=cam_max(km1,kx2(ilon,irgn));
// MPAS module_ra_cam.F:5313
} else {
// MPAS module_ra_cam.F:5314
km1=cam_sub(k1,1);
// MPAS module_ra_cam.F:5315
km3=cam_add(km1,2);
// MPAS module_ra_cam.F:5316
emx0=1.0f;
// MPAS module_ra_cam.F:5317
}
// MPAS module_ra_cam.F:5319
for(k2=kx1(ilon,irgn);k2<=kx2(ilon,irgn);k2+=1) {
// MPAS module_ra_cam.F:5320
k3=cam_add(k2,1);
// MPAS module_ra_cam.F:5324
tmp(ilon)=cam_mul(s(ilon,k2,cam_min(km3,pverp)),cam_min(1,cam_sub(pverp2,km3)));
// MPAS module_ra_cam.F:5325
ful(ilon,k2)=cam_add(cam_mul((cam_sub(1.0f,emx0)),fsul(ilon,k2)),cam_mul(emx0,(cam_sub(cam_add(fclt4(ilon,km1),s(ilon,k2,k3)),tmp(ilon)))));
// MPAS module_ra_cam.F:5327
}
// MPAS module_ra_cam.F:5328
} else if(((ilon==indxmx(iimx,irgn))&&(iimx<=ncolmx(irgn)))) {
// MPAS module_ra_cam.F:5329
iimx=cam_add(iimx,1);
// MPAS module_ra_cam.F:5330
}
// MPAS module_ra_cam.F:5334
for(iimx=1;iimx<=ncolmx(irgn);iimx+=1) {
// MPAS module_ra_cam.F:5335
i=indxmx(iimx,irgn);
// MPAS module_ra_cam.F:5346
k1=cam_add(kx2(i,irgn),1);
// MPAS module_ra_cam.F:5347
if((k1<pverp)) {
// MPAS module_ra_cam.F:5348
for(km1=cam_sub(pver,1);km1>=kx2(i,irgn);km1+=cam_neg(1)) {
// MPAS module_ra_cam.F:5349
km3=cam_add(km1,2);
// MPAS module_ra_cam.F:5350
k2=k1;
// MPAS module_ra_cam.F:5351
k3=cam_add(k2,1);
// MPAS module_ra_cam.F:5352
tmp(i)=cam_mul(s(i,k2,cam_min(km3,pverp)),cam_min(1,cam_sub(pverp2,km3)));
// MPAS module_ra_cam.F:5353
emx0=cam_div((cam_sub(ful(i,k1),fsul(i,k1))),(cam_sub((cam_sub(cam_add(fclt4(i,km1),s(i,k2,k3)),tmp(i))),fsul(i,k1))));
// MPAS module_ra_cam.F:5354
if(((emx0>=0.0f)&&(emx0<=1.0f))) {break;}
// MPAS module_ra_cam.F:5355
}
// MPAS module_ra_cam.F:5356
km1=cam_max(km1,kx2(i,irgn));
// MPAS module_ra_cam.F:5357
} else {
// MPAS module_ra_cam.F:5358
emx0=1.0f;
// MPAS module_ra_cam.F:5359
km1=cam_sub(k1,1);
// MPAS module_ra_cam.F:5360
}
// MPAS module_ra_cam.F:5361
ksort(0)=cam_add(km1,1);
// MPAS module_ra_cam.F:5366
nxsk=0;
// MPAS module_ra_cam.F:5367
for(k=kx2(i,irgn);k>=kx1(i,irgn);k+=cam_neg(1)) {
// MPAS module_ra_cam.F:5373
if((nxsk<nxs(i,irgn))) {
// MPAS module_ra_cam.F:5374
nxsk=0;
// MPAS module_ra_cam.F:5375
for(l=1;l<=nxs(i,irgn);l+=1) {
// MPAS module_ra_cam.F:5376
k1=kxs(l,i,irgn);
// MPAS module_ra_cam.F:5377
if((k<=k1)) {
// MPAS module_ra_cam.F:5378
nxsk=cam_add(nxsk,1);
// MPAS module_ra_cam.F:5379
ksort(nxsk)=k1;
// MPAS module_ra_cam.F:5380
}
// MPAS module_ra_cam.F:5381
}
// MPAS module_ra_cam.F:5382
}
// MPAS module_ra_cam.F:5386
ksort(cam_add(nxsk,1))=pverp;
// MPAS module_ra_cam.F:5390
for(l=1;l<=nxsk;l+=1) {
// MPAS module_ra_cam.F:5391
emx(l)=emis(i,ksort(l));
// MPAS module_ra_cam.F:5392
}
// MPAS module_ra_cam.F:5396
emx(0)=emx0;
// MPAS module_ra_cam.F:5400
cld0=1.0f;
// MPAS module_ra_cam.F:5404
k2=k;
// MPAS module_ra_cam.F:5405
k3=cam_add(k2,1);
// MPAS module_ra_cam.F:5409
for(l=1;l<=cam_add(nxsk,1);l+=1) {
// MPAS module_ra_cam.F:5413
cld1=cam_mul(cldp(i,ksort(l)),cam_min(1,cam_sub(cam_add(nxsk,1),l)));
// MPAS module_ra_cam.F:5414
if((cld0!=cld1)) {
// MPAS module_ra_cam.F:5415
ful(i,k2)=cam_add(ful(i,k2),cam_mul((cam_sub(cld0,cld1)),fsul(i,k2)));
// MPAS module_ra_cam.F:5416
for(l1=0;l1<=cam_sub(l,1);l1+=1) {
// MPAS module_ra_cam.F:5417
km1=cam_sub(ksort(l1),1);
// MPAS module_ra_cam.F:5418
km3=cam_add(km1,2);
// MPAS module_ra_cam.F:5422
tmp(i)=cam_mul(s(i,k2,cam_min(km3,pverp)),cam_min(1,cam_sub(pverp2,km3)));
// MPAS module_ra_cam.F:5423
ful(i,k2)=cam_add(ful(i,k2),cam_mul(cam_mul((cam_sub(cld0,cld1)),emx(l1)),(cam_sub(cam_sub(cam_add(fclt4(i,km1),s(i,k2,k3)),tmp(i)),fsul(i,k2)))));
// MPAS module_ra_cam.F:5425
}
// MPAS module_ra_cam.F:5426
}
// MPAS module_ra_cam.F:5427
cld0=cld1;
// MPAS module_ra_cam.F:5431
if((l<=nxsk)) {
// MPAS module_ra_cam.F:5432
k1=ksort(l);
// MPAS module_ra_cam.F:5433
trans=cam_sub(1.0f,emis(i,k1));
// MPAS module_ra_cam.F:5438
for(l1=0;l1<=nxsk;l1+=1) {
// MPAS module_ra_cam.F:5439
if((ksort(l1)>k1)) {
// MPAS module_ra_cam.F:5440
emx(l1)=cam_mul(emx(l1),trans);
// MPAS module_ra_cam.F:5441
}
// MPAS module_ra_cam.F:5442
}
// MPAS module_ra_cam.F:5443
}
// MPAS module_ra_cam.F:5447
}
// MPAS module_ra_cam.F:5451
}
// MPAS module_ra_cam.F:5455
}
// MPAS module_ra_cam.F:5459
}
// MPAS module_ra_cam.F:5463
}
// MPAS module_ra_cam.F:5470
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:5471
flwds(i)=fdl(i,pverp);
// MPAS module_ra_cam.F:5472
flns(i)=cam_sub(ful(i,pverp),fdl(i,pverp));
// MPAS module_ra_cam.F:5473
flnsc(i)=cam_sub(fsul(i,pverp),fsdl(i,pverp));
// MPAS module_ra_cam.F:5474
flnt(i)=cam_sub(ful(i,ntoplw),fdl(i,ntoplw));
// MPAS module_ra_cam.F:5475
flntc(i)=cam_sub(fsul(i,ntoplw),fsdl(i,ntoplw));
// MPAS module_ra_cam.F:5476
flut(i)=ful(i,ntoplw);
// MPAS module_ra_cam.F:5477
flutc(i)=fsul(i,ntoplw);
// MPAS module_ra_cam.F:5478
}
// MPAS module_ra_cam.F:5482
for(k=ntoplw;k<=pver;k+=1) {
// MPAS module_ra_cam.F:5483
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:5484
qrl(i,k)=cam_div(cam_mul(cam_mul((cam_add(cam_sub(cam_sub(ful(i,k),fdl(i,k)),ful(i,cam_add(k,1))),fdl(i,cam_add(k,1)))),1.e-4f),gravit),((cam_sub(pint(i,k),pint(i,cam_add(k,1))))));
// MPAS module_ra_cam.F:5486
}
// MPAS module_ra_cam.F:5487
}
// MPAS module_ra_cam.F:5489
if((ntoplw>1)) {
// MPAS module_ra_cam.F:5490
{ auto _lhs=qrl.section(cam_section(1,ncol),cam_section(1,cam_sub(ntoplw,1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam.F:5491
}
// MPAS module_ra_cam.F:5495
for(k=ntoplw;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:5496
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:5497
flup(i,k)=ful(i,k);
// MPAS module_ra_cam.F:5498
flupc(i,k)=fsul(i,k);
// MPAS module_ra_cam.F:5499
fldn(i,k)=fdl(i,k);
// MPAS module_ra_cam.F:5500
fldnc(i,k)=fsdl(i,k);
// MPAS module_ra_cam.F:5501
}
// MPAS module_ra_cam.F:5502
}
// MPAS module_ra_cam.F:5504
if((ntoplw>1)) {
// MPAS module_ra_cam.F:5505
{ auto _lhs=flup.section(cam_section(1,ncol),cam_section(1,cam_sub(ntoplw,1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam.F:5506
{ auto _lhs=flupc.section(cam_section(1,ncol),cam_section(1,cam_sub(ntoplw,1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam.F:5507
{ auto _lhs=fldn.section(cam_section(1,ncol),cam_section(1,cam_sub(ntoplw,1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam.F:5508
{ auto _lhs=fldnc.section(cam_section(1,ncol),cam_section(1,cam_sub(ntoplw,1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.f; }
// MPAS module_ra_cam.F:5509
}
// MPAS module_ra_cam.F:5511
return;
}
// MPAS module_ra_cam.F:5514-7331
CAM_ROUTINE void radcswmx(const int& jj,const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _pint,const CamArray<double>& _pmid,const CamArray<double>& _h2ommr,const CamArray<double>& _rh,const CamArray<double>& _o3mmr,const CamArray<double>& _aermmr,const CamArray<double>& _cld,const CamArray<double>& _cicewp,const CamArray<double>& _cliqwp,const CamArray<double>& _rel,const CamArray<double>& _rei,const CamArray<double>& _tauxcl,const CamArray<double>& _tauxci,const double& eccf,const CamArray<double>& _coszrs,const double& scon,const CamArray<double>& _solin,const float& solcon,const CamArray<double>& _asdir,const CamArray<double>& _asdif,const CamArray<double>& _aldir,const CamArray<double>& _aldif,const CamArray<int>& _nmxrgn,const CamArray<double>& _pmxrgn,const CamArray<double>& _qrs,const CamArray<double>& _fsnt,const CamArray<double>& _fsntc,const CamArray<double>& _fsntoa,const CamArray<double>& _fsntoac,const CamArray<double>& _fsnirtoa,const CamArray<double>& _fsnrtoac,const CamArray<double>& _fsnrtoaq,const CamArray<double>& _fsns,const CamArray<double>& _fsnsc,const CamArray<double>& _fsdsc,const CamArray<double>& _fsds,const CamArray<double>& _sols,const CamArray<double>& _soll,const CamArray<double>& _solsd,const CamArray<double>& _solld,const CamArray<double>& _frc_day,const CamArray<double>& _fsup,const CamArray<double>& _fsupc,const CamArray<double>& _fsdn,const CamArray<double>& _fsdnc,const CamArray<double>& _aertau,const CamArray<double>& _aerssa,const CamArray<double>& _aerasm,const CamArray<double>& _aerfwd) {
CamFrame _frame(arena);
const int nspint=19;
const int naer_groups=7;
const double cldmin=1.0e-80;
const double areamin=0.01;
const double cldeps=0.0;
const int nconfgmax=15;
CAM_TRACE("radcswmx.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("radcswmx.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.h2ommr");
auto h2ommr=_h2ommr.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.rh");
auto rh=_rh.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.o3mmr");
auto o3mmr=_o3mmr.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.aermmr");
auto aermmr=_aermmr.rebound(1,pcols,1,pver,1,naer_all);
CAM_TRACE("radcswmx.cld");
auto cld=_cld.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.cicewp");
auto cicewp=_cicewp.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.cliqwp");
auto cliqwp=_cliqwp.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.rel");
auto rel=_rel.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.rei");
auto rei=_rei.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.tauxcl");
auto tauxcl=_tauxcl.rebound(1,pcols,0,pver);
CAM_TRACE("radcswmx.tauxci");
auto tauxci=_tauxci.rebound(1,pcols,0,pver);
CAM_TRACE("radcswmx.coszrs");
auto coszrs=_coszrs.rebound(1,pcols);
CAM_TRACE("radcswmx.solin");
auto solin=_solin.rebound(1,pcols);
CAM_TRACE("radcswmx.asdir");
auto asdir=_asdir.rebound(1,pcols);
CAM_TRACE("radcswmx.asdif");
auto asdif=_asdif.rebound(1,pcols);
CAM_TRACE("radcswmx.aldir");
auto aldir=_aldir.rebound(1,pcols);
CAM_TRACE("radcswmx.aldif");
auto aldif=_aldif.rebound(1,pcols);
CAM_TRACE("radcswmx.nmxrgn");
auto nmxrgn=_nmxrgn.rebound(1,pcols);
CAM_TRACE("radcswmx.pmxrgn");
auto pmxrgn=_pmxrgn.rebound(1,pcols,1,pverp);
CAM_TRACE("radcswmx.qrs");
auto qrs=_qrs.rebound(1,pcols,1,pver);
CAM_TRACE("radcswmx.fsnt");
auto fsnt=_fsnt.rebound(1,pcols);
CAM_TRACE("radcswmx.fsntc");
auto fsntc=_fsntc.rebound(1,pcols);
CAM_TRACE("radcswmx.fsntoa");
auto fsntoa=_fsntoa.rebound(1,pcols);
CAM_TRACE("radcswmx.fsntoac");
auto fsntoac=_fsntoac.rebound(1,pcols);
CAM_TRACE("radcswmx.fsnirtoa");
auto fsnirtoa=_fsnirtoa.rebound(1,pcols);
CAM_TRACE("radcswmx.fsnrtoac");
auto fsnrtoac=_fsnrtoac.rebound(1,pcols);
CAM_TRACE("radcswmx.fsnrtoaq");
auto fsnrtoaq=_fsnrtoaq.rebound(1,pcols);
CAM_TRACE("radcswmx.fsns");
auto fsns=_fsns.rebound(1,pcols);
CAM_TRACE("radcswmx.fsnsc");
auto fsnsc=_fsnsc.rebound(1,pcols);
CAM_TRACE("radcswmx.fsdsc");
auto fsdsc=_fsdsc.rebound(1,pcols);
CAM_TRACE("radcswmx.fsds");
auto fsds=_fsds.rebound(1,pcols);
CAM_TRACE("radcswmx.sols");
auto sols=_sols.rebound(1,pcols);
CAM_TRACE("radcswmx.soll");
auto soll=_soll.rebound(1,pcols);
CAM_TRACE("radcswmx.solsd");
auto solsd=_solsd.rebound(1,pcols);
CAM_TRACE("radcswmx.solld");
auto solld=_solld.rebound(1,pcols);
CAM_TRACE("radcswmx.frc_day");
auto frc_day=_frc_day.rebound(1,pcols);
CAM_TRACE("radcswmx.fsup");
auto fsup=_fsup.rebound(1,pcols,1,pverp);
CAM_TRACE("radcswmx.fsupc");
auto fsupc=_fsupc.rebound(1,pcols,1,pverp);
CAM_TRACE("radcswmx.fsdn");
auto fsdn=_fsdn.rebound(1,pcols,1,pverp);
CAM_TRACE("radcswmx.fsdnc");
auto fsdnc=_fsdnc.rebound(1,pcols,1,pverp);
CAM_TRACE("radcswmx.aertau");
auto aertau=_aertau.rebound(1,pcols,1,nspint,1,naer_groups);
CAM_TRACE("radcswmx.aerssa");
auto aerssa=_aerssa.rebound(1,pcols,1,nspint,1,naer_groups);
CAM_TRACE("radcswmx.aerasm");
auto aerasm=_aerasm.rebound(1,pcols,1,nspint,1,naer_groups);
CAM_TRACE("radcswmx.aerfwd");
auto aerfwd=_aerfwd.rebound(1,pcols,1,nspint,1,naer_groups);
auto asort=arena.array<double>(1,pverp);
double atmp;
double cld0;
double totwgt;
auto wgtv=arena.array<double>(1,nconfgmax);
auto wstr=arena.array<double>(1,pverp,1,pverp);
double xexpt;
double xrdnd;
double xrupd;
double xrups;
double xtdnt;
double xwgt;
double yexpt;
double yrdnd;
double yrupd;
double ytdnd;
double ytupd;
double zexpt;
double zrdnd;
double zrupd;
double zrups;
double ztdnt;
bool new_term;
bool region_found;
auto ccon=arena.array<int>(0,pverp,1,nconfgmax);
auto cstr=arena.array<int>(0,pverp,1,pverp);
auto icond=arena.array<int>(0,pverp,1,nconfgmax);
auto iconu=arena.array<int>(0,pverp,1,nconfgmax);
int iconfig;
int irgn;
int is0;
int is1;
int isn;
auto istr=arena.array<int>(1,cam_add(pverp,1));
auto istrtd=arena.array<int>(0,pverp,0,cam_add(nconfgmax,1));
auto istrtu=arena.array<int>(0,pverp,0,cam_add(nconfgmax,1));
int j;
int k1;
int k2;
auto ksort=arena.array<int>(1,pverp);
int ktmp;
auto kx1=arena.array<int>(0,pverp);
auto kx2=arena.array<int>(0,pverp);
int l;
int l0;
int mrgn;
int mstr;
int n0;
int n1;
int nconfig;
int nconfigm;
int npasses;
int nrgn;
auto nstr=arena.array<int>(1,pverp);
int nuniq;
auto nuniqd=arena.array<int>(0,pverp);
auto nuniqu=arena.array<int>(0,pverp);
int nxs;
auto ptr0=arena.array<int>(1,nconfgmax);
auto ptr1=arena.array<int>(1,nconfgmax);
auto ptrc=arena.array<int>(1,nconfgmax);
int ns;
int i;
int k;
int km1;
int kp1;
int n;
int ndayc;
auto idayc=arena.array<int>(1,pcols);
int indxsl;
int ksz;
int krh;
int kaer;
double wrh;
double rhtrunc;
auto albdir=arena.array<double>(1,pcols,1,nspint);
auto albdif=arena.array<double>(1,pcols,1,nspint);
double wgtint;
double solflx;
double sfltot;
auto totfld=arena.array<double>(0,pver);
auto fswup=arena.array<double>(0,pverp);
auto fswdn=arena.array<double>(0,pverp);
auto fswupc=arena.array<double>(0,pverp);
auto fswdnc=arena.array<double>(0,pverp);
auto wcl=arena.array<double>(1,pcols,0,pver);
auto gcl=arena.array<double>(1,pcols,0,pver);
auto fcl=arena.array<double>(1,pcols,0,pver);
auto wci=arena.array<double>(1,pcols,0,pver);
auto gci=arena.array<double>(1,pcols,0,pver);
auto fci=arena.array<double>(1,pcols,0,pver);
auto usul=arena.array<double>(1,pcols,1,pver);
auto ubg=arena.array<double>(1,pcols,1,pver);
auto usslt=arena.array<double>(1,pcols,1,pver);
auto ucphil=arena.array<double>(1,pcols,1,pver);
auto ucphob=arena.array<double>(1,pcols,1,pver);
auto ucb=arena.array<double>(1,pcols,1,pver);
auto uvolc=arena.array<double>(1,pcols,1,pver);
auto udst=arena.array<double>(1,ndstsz,1,pcols,1,pver);
double tau_sul;
double tau_bg;
double tau_sslt;
double tau_cphil;
double tau_cphob;
double tau_cb;
double tau_volc;
auto tau_dst=arena.array<double>(1,ndstsz);
double tau_dst_tot;
double tau_tot;
double tau_w_sul;
double tau_w_bg;
double tau_w_sslt;
double tau_w_cphil;
double tau_w_cphob;
double tau_w_cb;
double tau_w_volc;
auto tau_w_dst=arena.array<double>(1,ndstsz);
double tau_w_dst_tot;
double tau_w_tot;
double tau_w_g_sul;
double tau_w_g_bg;
double tau_w_g_sslt;
double tau_w_g_cphil;
double tau_w_g_cphob;
double tau_w_g_cb;
double tau_w_g_volc;
auto tau_w_g_dst=arena.array<double>(1,ndstsz);
double tau_w_g_dst_tot;
double tau_w_g_tot;
double f_sul;
double f_bg;
double f_sslt;
double f_cphil;
double f_cphob;
double f_cb;
double f_volc;
auto f_dst=arena.array<double>(1,ndstsz);
double f_dst_tot;
double f_tot;
double tau_w_f_sul;
double tau_w_f_bg;
double tau_w_f_sslt;
double tau_w_f_cphil;
double tau_w_f_cphob;
double tau_w_f_cb;
double tau_w_f_volc;
auto tau_w_f_dst=arena.array<double>(1,ndstsz);
double tau_w_f_dst_tot;
double tau_w_f_tot;
double w_dst_tot;
double w_tot;
double g_dst_tot;
double g_tot;
double ksuli;
double ksslti;
double kcphili;
double wsuli;
double wsslti;
double wcphili;
double gsuli;
double gsslti;
double gcphili;
auto tauxar=arena.array<double>(1,pcols,0,pver);
auto wa=arena.array<double>(1,pcols,0,pver);
auto ga=arena.array<double>(1,pcols,0,pver);
auto fa=arena.array<double>(1,pcols,0,pver);
auto pflx=arena.array<double>(1,pcols,0,pverp);
auto zenfac=arena.array<double>(1,pcols);
double sqrco2;
double tmp1;
double tmp2;
double pdel;
double path;
double ptop;
double ptho2;
double ptho3;
double pthco2;
double pthh2o;
double h2ostr;
auto wavmid=arena.array<double>(1,nspint);
double trayoslp;
double tmp1l;
double tmp2l;
double tmp3l;
double tmp1i;
double tmp2i;
double tmp3i;
double rdenom;
double rdirexp;
double tdnmexp;
auto psf=arena.array<double>(1,nspint);
auto uh2o=arena.array<double>(1,pcols,0,pver);
auto uo3=arena.array<double>(1,pcols,0,pver);
auto uco2=arena.array<double>(1,pcols,0,pver);
auto uo2=arena.array<double>(1,pcols,0,pver);
auto uaer=arena.array<double>(1,pcols,0,pver);
auto uth2o=arena.array<double>(1,pcols);
auto uto3=arena.array<double>(1,pcols);
auto utco2=arena.array<double>(1,pcols);
auto uto2=arena.array<double>(1,pcols);
auto rdir=arena.array<double>(1,nspint,1,pcols,0,pver);
auto rdif=arena.array<double>(1,nspint,1,pcols,0,pver);
auto tdir=arena.array<double>(1,nspint,1,pcols,0,pver);
auto tdif=arena.array<double>(1,nspint,1,pcols,0,pver);
auto explay=arena.array<double>(1,nspint,1,pcols,0,pver);
auto rdirc=arena.array<double>(1,nspint,1,pcols,0,pver);
auto rdifc=arena.array<double>(1,nspint,1,pcols,0,pver);
auto tdirc=arena.array<double>(1,nspint,1,pcols,0,pver);
auto tdifc=arena.array<double>(1,nspint,1,pcols,0,pver);
auto explayc=arena.array<double>(1,nspint,1,pcols,0,pver);
double flxdiv;
auto exptdn=arena.array<double>(0,pverp,1,nconfgmax);
auto rdndif=arena.array<double>(0,pverp,1,nconfgmax);
auto rupdif=arena.array<double>(0,pverp,1,nconfgmax);
auto rupdir=arena.array<double>(0,pverp,1,nconfgmax);
auto tdntot=arena.array<double>(0,pverp,1,nconfgmax);
auto exptdnc=arena.array<double>(0,pverp);
auto rdndifc=arena.array<double>(0,pverp);
auto rupdifc=arena.array<double>(0,pverp);
auto rupdirc=arena.array<double>(0,pverp);
auto tdntotc=arena.array<double>(0,pverp);
auto fluxup=arena.array<double>(0,pverp);
auto fluxdn=arena.array<double>(0,pverp);
double wexptdn;
double abarli;
double bbarli;
double cbarli;
double dbarli;
double ebarli;
double fbarli;
double abarii;
double bbarii;
double cbarii;
double dbarii;
double ebarii;
double fbarii;
// MPAS module_ra_cam.F:6096
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:6100
fsds(i)=0.0;
// MPAS module_ra_cam.F:6102
fsnirtoa(i)=0.0;
// MPAS module_ra_cam.F:6103
fsnrtoac(i)=0.0;
// MPAS module_ra_cam.F:6104
fsnrtoaq(i)=0.0;
// MPAS module_ra_cam.F:6106
fsns(i)=0.0;
// MPAS module_ra_cam.F:6107
fsnsc(i)=0.0;
// MPAS module_ra_cam.F:6108
fsdsc(i)=0.0;
// MPAS module_ra_cam.F:6110
fsnt(i)=0.0;
// MPAS module_ra_cam.F:6111
fsntc(i)=0.0;
// MPAS module_ra_cam.F:6112
fsntoa(i)=0.0;
// MPAS module_ra_cam.F:6113
fsntoac(i)=0.0;
// MPAS module_ra_cam.F:6115
solin(i)=0.0;
// MPAS module_ra_cam.F:6117
sols(i)=0.0;
// MPAS module_ra_cam.F:6118
soll(i)=0.0;
// MPAS module_ra_cam.F:6119
solsd(i)=0.0;
// MPAS module_ra_cam.F:6120
solld(i)=0.0;
// MPAS module_ra_cam.F:6124
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:6125
fsup(i,k)=0.0;
// MPAS module_ra_cam.F:6126
fsupc(i,k)=0.0;
// MPAS module_ra_cam.F:6127
fsdn(i,k)=0.0;
// MPAS module_ra_cam.F:6128
fsdnc(i,k)=0.0;
// MPAS module_ra_cam.F:6129
tauxcl(i,cam_sub(k,1))=0.0;
// MPAS module_ra_cam.F:6130
tauxci(i,cam_sub(k,1))=0.0;
// MPAS module_ra_cam.F:6131
}
// MPAS module_ra_cam.F:6133
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:6134
qrs(i,k)=0.0;
// MPAS module_ra_cam.F:6135
}
// MPAS module_ra_cam.F:6139
for(kaer=1;kaer<=naer_groups;kaer+=1) {
// MPAS module_ra_cam.F:6140
for(ns=1;ns<=nspint;ns+=1) {
// MPAS module_ra_cam.F:6141
frc_day(i)=0.0;
// MPAS module_ra_cam.F:6142
aertau(i,ns,kaer)=0.0;
// MPAS module_ra_cam.F:6143
aerssa(i,ns,kaer)=0.0;
// MPAS module_ra_cam.F:6144
aerasm(i,ns,kaer)=0.0;
// MPAS module_ra_cam.F:6145
aerfwd(i,ns,kaer)=0.0;
// MPAS module_ra_cam.F:6146
}
// MPAS module_ra_cam.F:6147
}
// MPAS module_ra_cam.F:6149
}
// MPAS module_ra_cam.F:6155
ndayc=0;
// MPAS module_ra_cam.F:6156
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:6157
if((coszrs(i)>0.0)) {
// MPAS module_ra_cam.F:6158
ndayc=cam_add(ndayc,1);
// MPAS module_ra_cam.F:6159
idayc(ndayc)=i;
// MPAS module_ra_cam.F:6160
}
// MPAS module_ra_cam.F:6161
}
// MPAS module_ra_cam.F:6165
if((ndayc==0)) {return;}
// MPAS module_ra_cam.F:6169
tmp1=cam_div(0.5,(cam_mul(gravit,sslp)));
// MPAS module_ra_cam.F:6170
tmp2=cam_div(delta,gravit);
// MPAS module_ra_cam.F:6171
sqrco2=cam_sqrt(co2mmr);
// MPAS module_ra_cam.F:6173
for(n=1;n<=ndayc;n+=1) {
// MPAS module_ra_cam.F:6174
i=idayc(n);
// MPAS module_ra_cam.F:6180
solin(i)=cam_mul(cam_mul(solcon,coszrs(i)),1000.f);
// MPAS module_ra_cam.F:6181
pflx(i,0)=0.;
// MPAS module_ra_cam.F:6182
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:6183
pflx(i,k)=pint(i,k);
// MPAS module_ra_cam.F:6184
}
// MPAS module_ra_cam.F:6188
ptop=pflx(i,1);
// MPAS module_ra_cam.F:6189
ptho2=cam_div(cam_mul(o2mmr,ptop),gravit);
// MPAS module_ra_cam.F:6190
ptho3=cam_div(cam_mul(o3mmr(i,1),ptop),gravit);
// MPAS module_ra_cam.F:6191
pthco2=cam_mul(sqrco2,(cam_div(ptop,gravit)));
// MPAS module_ra_cam.F:6192
h2ostr=cam_sqrt(cam_div(1.,h2ommr(i,1)));
// MPAS module_ra_cam.F:6193
zenfac(i)=cam_sqrt(coszrs(i));
// MPAS module_ra_cam.F:6194
pthh2o=cam_add(cam_mul(cam_pow(ptop,2),tmp1),cam_mul((cam_mul(ptop,rga)),(cam_mul(cam_mul(h2ostr,zenfac(i)),delta))));
// MPAS module_ra_cam.F:6196
uh2o(i,0)=cam_mul(h2ommr(i,1),pthh2o);
// MPAS module_ra_cam.F:6197
uco2(i,0)=cam_mul(zenfac(i),pthco2);
// MPAS module_ra_cam.F:6198
uo2(i,0)=cam_mul(zenfac(i),ptho2);
// MPAS module_ra_cam.F:6199
uo3(i,0)=ptho3;
// MPAS module_ra_cam.F:6200
uaer(i,0)=0.0;
// MPAS module_ra_cam.F:6201
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:6202
pdel=cam_sub(pflx(i,cam_add(k,1)),pflx(i,k));
// MPAS module_ra_cam.F:6203
path=cam_div(pdel,gravit);
// MPAS module_ra_cam.F:6204
ptho2=cam_mul(o2mmr,path);
// MPAS module_ra_cam.F:6205
ptho3=cam_mul(o3mmr(i,k),path);
// MPAS module_ra_cam.F:6206
pthco2=cam_mul(sqrco2,path);
// MPAS module_ra_cam.F:6207
h2ostr=cam_sqrt(cam_div(1.0,h2ommr(i,k)));
// MPAS module_ra_cam.F:6208
pthh2o=cam_add(cam_mul((cam_sub(cam_pow(pflx(i,cam_add(k,1)),2),cam_pow(pflx(i,k),2))),tmp1),cam_mul(cam_mul(cam_mul(pdel,h2ostr),zenfac(i)),tmp2));
// MPAS module_ra_cam.F:6209
uh2o(i,k)=cam_mul(h2ommr(i,k),pthh2o);
// MPAS module_ra_cam.F:6210
uco2(i,k)=cam_mul(zenfac(i),pthco2);
// MPAS module_ra_cam.F:6211
uo2(i,k)=cam_mul(zenfac(i),ptho2);
// MPAS module_ra_cam.F:6212
uo3(i,k)=ptho3;
// MPAS module_ra_cam.F:6213
usul(i,k)=cam_mul(aermmr(i,k,idxsul),path);
// MPAS module_ra_cam.F:6214
ubg(i,k)=cam_mul(aermmr(i,k,idxbg),path);
// MPAS module_ra_cam.F:6215
usslt(i,k)=cam_mul(aermmr(i,k,idxsslt),path);
// MPAS module_ra_cam.F:6216
if((usslt(i,k)<0.0f)) {
// MPAS module_ra_cam.F:6217
usslt(i,k)=0.0f;
// MPAS module_ra_cam.F:6218
}
// MPAS module_ra_cam.F:6219
ucphil(i,k)=cam_mul(aermmr(i,k,idxocphi),path);
// MPAS module_ra_cam.F:6220
ucphob(i,k)=cam_mul(aermmr(i,k,idxocpho),path);
// MPAS module_ra_cam.F:6221
ucb(i,k)=cam_mul((cam_add(aermmr(i,k,idxbcpho),aermmr(i,k,idxbcphi))),path);
// MPAS module_ra_cam.F:6222
uvolc(i,k)=aermmr(i,k,idxvolc);
// MPAS module_ra_cam.F:6223
for(ksz=1;ksz<=ndstsz;ksz+=1) {
// MPAS module_ra_cam.F:6224
udst(ksz,i,k)=cam_mul(aermmr(i,k,cam_add(cam_sub(idxdustfirst,1),ksz)),path);
// MPAS module_ra_cam.F:6225
}
// MPAS module_ra_cam.F:6226
}
// MPAS module_ra_cam.F:6230
uth2o(i)=0.0;
// MPAS module_ra_cam.F:6231
uto3(i)=0.0;
// MPAS module_ra_cam.F:6232
utco2(i)=0.0;
// MPAS module_ra_cam.F:6233
uto2(i)=0.0;
// MPAS module_ra_cam.F:6235
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:6236
uth2o(i)=cam_add(uth2o(i),uh2o(i,k));
// MPAS module_ra_cam.F:6237
uto3(i)=cam_add(uto3(i),uo3(i,k));
// MPAS module_ra_cam.F:6238
utco2(i)=cam_add(utco2(i),uco2(i,k));
// MPAS module_ra_cam.F:6239
uto2(i)=cam_add(uto2(i),uo2(i,k));
// MPAS module_ra_cam.F:6240
}
// MPAS module_ra_cam.F:6246
tauxcl(i,0)=0.;
// MPAS module_ra_cam.F:6247
wcl(i,0)=0.999999;
// MPAS module_ra_cam.F:6248
gcl(i,0)=0.85;
// MPAS module_ra_cam.F:6249
fcl(i,0)=0.725;
// MPAS module_ra_cam.F:6250
tauxci(i,0)=0.;
// MPAS module_ra_cam.F:6251
wci(i,0)=0.999999;
// MPAS module_ra_cam.F:6252
gci(i,0)=0.85;
// MPAS module_ra_cam.F:6253
fci(i,0)=0.725;
// MPAS module_ra_cam.F:6257
tauxar(i,0)=0.;
// MPAS module_ra_cam.F:6258
wa(i,0)=0.925;
// MPAS module_ra_cam.F:6259
ga(i,0)=0.850;
// MPAS module_ra_cam.F:6260
fa(i,0)=0.7225;
// MPAS module_ra_cam.F:6264
}
// MPAS module_ra_cam.F:6268
for(ns=1;ns<=nspint;ns+=1) {
// MPAS module_ra_cam.F:6281
if((wavmax(ns)<=0.7)) {
// MPAS module_ra_cam.F:6282
indxsl=1;
// MPAS module_ra_cam.F:6283
} else if((wavmin(ns)==0.700)) {
// MPAS module_ra_cam.F:6284
indxsl=2;
// MPAS module_ra_cam.F:6285
} else if((wavmin(ns)==0.701)) {
// MPAS module_ra_cam.F:6286
indxsl=3;
// MPAS module_ra_cam.F:6287
} else if(((wavmin(ns)==0.702)||(wavmin(ns)>2.38))) {
// MPAS module_ra_cam.F:6288
indxsl=4;
// MPAS module_ra_cam.F:6289
}
// MPAS module_ra_cam.F:6294
abarli=abarl(indxsl);
// MPAS module_ra_cam.F:6295
bbarli=bbarl(indxsl);
// MPAS module_ra_cam.F:6296
cbarli=cbarl(indxsl);
// MPAS module_ra_cam.F:6297
dbarli=dbarl(indxsl);
// MPAS module_ra_cam.F:6298
ebarli=ebarl(indxsl);
// MPAS module_ra_cam.F:6299
fbarli=fbarl(indxsl);
// MPAS module_ra_cam.F:6301
abarii=abari(indxsl);
// MPAS module_ra_cam.F:6302
bbarii=bbari(indxsl);
// MPAS module_ra_cam.F:6303
cbarii=cbari(indxsl);
// MPAS module_ra_cam.F:6304
dbarii=dbari(indxsl);
// MPAS module_ra_cam.F:6305
ebarii=ebari(indxsl);
// MPAS module_ra_cam.F:6306
fbarii=fbari(indxsl);
// MPAS module_ra_cam.F:6311
psf(ns)=1.0;
// MPAS module_ra_cam.F:6312
if((ph2o(ns)!=0.)) {psf(ns)=cam_mul(psf(ns),ph2o(ns));}
// MPAS module_ra_cam.F:6313
if((pco2(ns)!=0.)) {psf(ns)=cam_mul(psf(ns),pco2(ns));}
// MPAS module_ra_cam.F:6314
if((po2(ns)!=0.)) {psf(ns)=cam_mul(psf(ns),po2(ns));}
// MPAS module_ra_cam.F:6316
for(n=1;n<=ndayc;n+=1) {
// MPAS module_ra_cam.F:6317
i=idayc(n);
// MPAS module_ra_cam.F:6319
frc_day(i)=1.0;
// MPAS module_ra_cam.F:6320
for(kaer=1;kaer<=naer_groups;kaer+=1) {
// MPAS module_ra_cam.F:6321
aertau(i,ns,kaer)=0.0f;
// MPAS module_ra_cam.F:6322
aerssa(i,ns,kaer)=0.0f;
// MPAS module_ra_cam.F:6323
aerasm(i,ns,kaer)=0.0f;
// MPAS module_ra_cam.F:6324
aerfwd(i,ns,kaer)=0.0f;
// MPAS module_ra_cam.F:6325
}
// MPAS module_ra_cam.F:6327
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:6331
tmp1l=cam_add(abarli,cam_div(bbarli,rel(i,k)));
// MPAS module_ra_cam.F:6332
tmp2l=cam_sub(cam_sub(1.,cbarli),cam_mul(dbarli,rel(i,k)));
// MPAS module_ra_cam.F:6333
tmp3l=cam_mul(fbarli,rel(i,k));
// MPAS module_ra_cam.F:6337
tmp1i=cam_add(abarii,cam_div(bbarii,rei(i,k)));
// MPAS module_ra_cam.F:6338
tmp2i=cam_sub(cam_sub(1.,cbarii),cam_mul(dbarii,rei(i,k)));
// MPAS module_ra_cam.F:6339
tmp3i=cam_mul(fbarii,rei(i,k));
// MPAS module_ra_cam.F:6341
if(((cld(i,k)>=cldmin)&&(cld(i,k)>=cldeps))) {
// MPAS module_ra_cam.F:6342
tauxcl(i,k)=cam_mul(cliqwp(i,k),tmp1l);
// MPAS module_ra_cam.F:6343
tauxci(i,k)=cam_mul(cicewp(i,k),tmp1i);
// MPAS module_ra_cam.F:6344
} else {
// MPAS module_ra_cam.F:6345
tauxcl(i,k)=0.0f;
// MPAS module_ra_cam.F:6346
tauxci(i,k)=0.0f;
// MPAS module_ra_cam.F:6347
}
// MPAS module_ra_cam.F:6353
wcl(i,k)=cam_min(tmp2l,.999999);
// MPAS module_ra_cam.F:6354
gcl(i,k)=cam_add(ebarli,tmp3l);
// MPAS module_ra_cam.F:6355
fcl(i,k)=cam_mul(gcl(i,k),gcl(i,k));
// MPAS module_ra_cam.F:6357
wci(i,k)=cam_min(tmp2i,.999999);
// MPAS module_ra_cam.F:6358
gci(i,k)=cam_add(ebarii,tmp3i);
// MPAS module_ra_cam.F:6359
fci(i,k)=cam_mul(gci(i,k),gci(i,k));
// MPAS module_ra_cam.F:6364
rhtrunc=rh(i,k);
// MPAS module_ra_cam.F:6365
rhtrunc=cam_min(rh(i,k),1.);
// MPAS module_ra_cam.F:6367
krh=cam_min(cam_add(floor(cam_mul(rhtrunc,nrh)),1),cam_sub(nrh,1));
// MPAS module_ra_cam.F:6368
wrh=cam_sub(cam_mul(rhtrunc,nrh),krh);
// MPAS module_ra_cam.F:6371
ksuli=cam_sub(cam_mul(ksul(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(ksul(krh,ns),wrh));
// MPAS module_ra_cam.F:6372
ksslti=cam_sub(cam_mul(ksslt(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(ksslt(krh,ns),wrh));
// MPAS module_ra_cam.F:6373
kcphili=cam_sub(cam_mul(kcphil(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(kcphil(krh,ns),wrh));
// MPAS module_ra_cam.F:6374
wsuli=cam_sub(cam_mul(wsul(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(wsul(krh,ns),wrh));
// MPAS module_ra_cam.F:6375
wsslti=cam_sub(cam_mul(wsslt(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(wsslt(krh,ns),wrh));
// MPAS module_ra_cam.F:6376
wcphili=cam_sub(cam_mul(wcphil(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(wcphil(krh,ns),wrh));
// MPAS module_ra_cam.F:6377
gsuli=cam_sub(cam_mul(gsul(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(gsul(krh,ns),wrh));
// MPAS module_ra_cam.F:6378
gsslti=cam_sub(cam_mul(gsslt(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(gsslt(krh,ns),wrh));
// MPAS module_ra_cam.F:6379
gcphili=cam_sub(cam_mul(gcphil(cam_add(krh,1),ns),(cam_add(wrh,1))),cam_mul(gcphil(krh,ns),wrh));
// MPAS module_ra_cam.F:6381
tau_sul=cam_mul(cam_mul(1.e4f,ksuli),usul(i,k));
// MPAS module_ra_cam.F:6382
tau_sslt=cam_mul(cam_mul(1.e4f,ksslti),usslt(i,k));
// MPAS module_ra_cam.F:6383
tau_cphil=cam_mul(cam_mul(1.e4f,kcphili),ucphil(i,k));
// MPAS module_ra_cam.F:6384
tau_cphob=cam_mul(cam_mul(1.e4f,kcphob(ns)),ucphob(i,k));
// MPAS module_ra_cam.F:6385
tau_cb=cam_mul(cam_mul(1.e4f,kcb(ns)),ucb(i,k));
// MPAS module_ra_cam.F:6386
tau_volc=cam_mul(cam_mul(1.e3f,kvolc(ns)),uvolc(i,k));
// MPAS module_ra_cam.F:6387
{ auto _lhs=tau_dst.section(cam_section(1,ndstsz)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_mul(cam_mul(1.e4f,kdst.section(cam_section(1,ndstsz),ns).linear(_slice)),udst.section(cam_section(1,ndstsz),i,k).linear(_slice)); }
// MPAS module_ra_cam.F:6388
tau_bg=cam_mul(cam_mul(1.e4f,kbg(ns)),ubg(i,k));
// MPAS module_ra_cam.F:6390
tau_w_sul=cam_mul(tau_sul,wsuli);
// MPAS module_ra_cam.F:6391
tau_w_sslt=cam_mul(tau_sslt,wsslti);
// MPAS module_ra_cam.F:6392
tau_w_cphil=cam_mul(tau_cphil,wcphili);
// MPAS module_ra_cam.F:6393
tau_w_cphob=cam_mul(tau_cphob,wcphob(ns));
// MPAS module_ra_cam.F:6394
tau_w_cb=cam_mul(tau_cb,wcb(ns));
// MPAS module_ra_cam.F:6395
tau_w_volc=cam_mul(tau_volc,wvolc(ns));
// MPAS module_ra_cam.F:6396
{ auto _lhs=tau_w_dst.section(cam_section(1,ndstsz)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_mul(tau_dst.section(cam_section(1,ndstsz)).linear(_slice),wdst.section(cam_section(1,ndstsz),ns).linear(_slice)); }
// MPAS module_ra_cam.F:6397
tau_w_bg=cam_mul(tau_bg,wbg(ns));
// MPAS module_ra_cam.F:6399
tau_w_g_sul=cam_mul(tau_w_sul,gsuli);
// MPAS module_ra_cam.F:6400
tau_w_g_sslt=cam_mul(tau_w_sslt,gsslti);
// MPAS module_ra_cam.F:6401
tau_w_g_cphil=cam_mul(tau_w_cphil,gcphili);
// MPAS module_ra_cam.F:6402
tau_w_g_cphob=cam_mul(tau_w_cphob,gcphob(ns));
// MPAS module_ra_cam.F:6403
tau_w_g_cb=cam_mul(tau_w_cb,gcb(ns));
// MPAS module_ra_cam.F:6404
tau_w_g_volc=cam_mul(tau_w_volc,gvolc(ns));
// MPAS module_ra_cam.F:6405
{ auto _lhs=tau_w_g_dst.section(cam_section(1,ndstsz)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_mul(tau_w_dst.section(cam_section(1,ndstsz)).linear(_slice),gdst.section(cam_section(1,ndstsz),ns).linear(_slice)); }
// MPAS module_ra_cam.F:6406
tau_w_g_bg=cam_mul(tau_w_bg,gbg(ns));
// MPAS module_ra_cam.F:6408
f_sul=cam_mul(gsuli,gsuli);
// MPAS module_ra_cam.F:6409
f_sslt=cam_mul(gsslti,gsslti);
// MPAS module_ra_cam.F:6410
f_cphil=cam_mul(gcphili,gcphili);
// MPAS module_ra_cam.F:6411
f_cphob=cam_mul(gcphob(ns),gcphob(ns));
// MPAS module_ra_cam.F:6412
f_cb=cam_mul(gcb(ns),gcb(ns));
// MPAS module_ra_cam.F:6413
f_volc=cam_mul(gvolc(ns),gvolc(ns));
// MPAS module_ra_cam.F:6414
{ auto _lhs=f_dst.section(cam_section(1,ndstsz)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_mul(gdst.section(cam_section(1,ndstsz),ns).linear(_slice),gdst.section(cam_section(1,ndstsz),ns).linear(_slice)); }
// MPAS module_ra_cam.F:6415
f_bg=cam_mul(gbg(ns),gbg(ns));
// MPAS module_ra_cam.F:6417
tau_w_f_sul=cam_mul(tau_w_sul,f_sul);
// MPAS module_ra_cam.F:6418
tau_w_f_bg=cam_mul(tau_w_bg,f_bg);
// MPAS module_ra_cam.F:6419
tau_w_f_sslt=cam_mul(tau_w_sslt,f_sslt);
// MPAS module_ra_cam.F:6420
tau_w_f_cphil=cam_mul(tau_w_cphil,f_cphil);
// MPAS module_ra_cam.F:6421
tau_w_f_cphob=cam_mul(tau_w_cphob,f_cphob);
// MPAS module_ra_cam.F:6422
tau_w_f_cb=cam_mul(tau_w_cb,f_cb);
// MPAS module_ra_cam.F:6423
tau_w_f_volc=cam_mul(tau_w_volc,f_volc);
// MPAS module_ra_cam.F:6424
{ auto _lhs=tau_w_f_dst.section(cam_section(1,ndstsz)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_mul(tau_w_dst.section(cam_section(1,ndstsz)).linear(_slice),f_dst.section(cam_section(1,ndstsz)).linear(_slice)); }
// MPAS module_ra_cam.F:6430
tau_dst_tot=cam_sum(tau_dst);
// MPAS module_ra_cam.F:6431
tau_w_dst_tot=cam_sum(tau_w_dst);
// MPAS module_ra_cam.F:6432
tau_w_g_dst_tot=cam_sum(tau_w_g_dst);
// MPAS module_ra_cam.F:6433
tau_w_f_dst_tot=cam_sum(tau_w_f_dst);
// MPAS module_ra_cam.F:6435
if((tau_dst_tot>0.0f)) {
// MPAS module_ra_cam.F:6436
w_dst_tot=cam_div(tau_w_dst_tot,tau_dst_tot);
// MPAS module_ra_cam.F:6437
} else {
// MPAS module_ra_cam.F:6438
w_dst_tot=0.0f;
// MPAS module_ra_cam.F:6439
}
// MPAS module_ra_cam.F:6441
if((tau_w_dst_tot>0.0f)) {
// MPAS module_ra_cam.F:6442
g_dst_tot=cam_div(tau_w_g_dst_tot,tau_w_dst_tot);
// MPAS module_ra_cam.F:6443
f_dst_tot=cam_div(tau_w_f_dst_tot,tau_w_dst_tot);
// MPAS module_ra_cam.F:6444
} else {
// MPAS module_ra_cam.F:6445
g_dst_tot=0.0f;
// MPAS module_ra_cam.F:6446
f_dst_tot=0.0f;
// MPAS module_ra_cam.F:6447
}
// MPAS module_ra_cam.F:6451
tau_tot=cam_add(cam_add(cam_add(cam_add(cam_add(tau_sul,tau_sslt),tau_cphil),tau_cphob),tau_cb),tau_dst_tot);
// MPAS module_ra_cam.F:6453
tau_tot=cam_add(cam_add(tau_tot,tau_bg),tau_volc);
// MPAS module_ra_cam.F:6455
tau_w_tot=cam_add(cam_add(cam_add(cam_add(cam_add(tau_w_sul,tau_w_sslt),tau_w_cphil),tau_w_cphob),tau_w_cb),tau_w_dst_tot);
// MPAS module_ra_cam.F:6457
tau_w_tot=cam_add(cam_add(tau_w_tot,tau_w_bg),tau_w_volc);
// MPAS module_ra_cam.F:6459
tau_w_g_tot=cam_add(cam_add(cam_add(cam_add(cam_add(tau_w_g_sul,tau_w_g_sslt),tau_w_g_cphil),tau_w_g_cphob),tau_w_g_cb),tau_w_g_dst_tot);
// MPAS module_ra_cam.F:6461
tau_w_g_tot=cam_add(cam_add(tau_w_g_tot,tau_w_g_bg),tau_w_g_volc);
// MPAS module_ra_cam.F:6463
tau_w_f_tot=cam_add(cam_add(cam_add(cam_add(cam_add(tau_w_f_sul,tau_w_f_sslt),tau_w_f_cphil),tau_w_f_cphob),tau_w_f_cb),tau_w_f_dst_tot);
// MPAS module_ra_cam.F:6465
tau_w_f_tot=cam_add(cam_add(tau_w_f_tot,tau_w_f_bg),tau_w_f_volc);
// MPAS module_ra_cam.F:6467
if((tau_tot>0.0f)) {
// MPAS module_ra_cam.F:6468
w_tot=cam_div(tau_w_tot,tau_tot);
// MPAS module_ra_cam.F:6469
} else {
// MPAS module_ra_cam.F:6470
w_tot=0.0f;
// MPAS module_ra_cam.F:6471
}
// MPAS module_ra_cam.F:6473
if((tau_w_tot>0.0f)) {
// MPAS module_ra_cam.F:6474
g_tot=cam_div(tau_w_g_tot,tau_w_tot);
// MPAS module_ra_cam.F:6475
f_tot=cam_div(tau_w_f_tot,tau_w_tot);
// MPAS module_ra_cam.F:6476
} else {
// MPAS module_ra_cam.F:6477
g_tot=0.0f;
// MPAS module_ra_cam.F:6478
f_tot=0.0f;
// MPAS module_ra_cam.F:6479
}
// MPAS module_ra_cam.F:6481
tauxar(i,k)=tau_tot;
// MPAS module_ra_cam.F:6482
wa(i,k)=cam_min(w_tot,0.999999);
// MPAS module_ra_cam.F:6483
if((g_tot>1.)) {/* MPAS diagnostic only */}
// MPAS module_ra_cam.F:6484
if((g_tot<cam_neg(1.))) {/* MPAS diagnostic only */}
// MPAS module_ra_cam.F:6487
ga(i,k)=g_tot;
// MPAS module_ra_cam.F:6488
if((f_tot>1.)) {/* MPAS diagnostic only */}
// MPAS module_ra_cam.F:6489
if((f_tot<0.)) {/* MPAS diagnostic only */}
// MPAS module_ra_cam.F:6492
fa(i,k)=f_tot;
// MPAS module_ra_cam.F:6494
aertau(i,ns,1)=cam_add(aertau(i,ns,1),tau_sul);
// MPAS module_ra_cam.F:6495
aertau(i,ns,2)=cam_add(aertau(i,ns,2),tau_sslt);
// MPAS module_ra_cam.F:6496
aertau(i,ns,3)=cam_add(cam_add(cam_add(aertau(i,ns,3),tau_cphil),tau_cphob),tau_cb);
// MPAS module_ra_cam.F:6497
aertau(i,ns,4)=cam_add(aertau(i,ns,4),tau_dst_tot);
// MPAS module_ra_cam.F:6498
aertau(i,ns,5)=cam_add(aertau(i,ns,5),tau_bg);
// MPAS module_ra_cam.F:6499
aertau(i,ns,6)=cam_add(aertau(i,ns,6),tau_volc);
// MPAS module_ra_cam.F:6500
aertau(i,ns,7)=cam_add(aertau(i,ns,7),tau_tot);
// MPAS module_ra_cam.F:6502
aerssa(i,ns,1)=cam_add(aerssa(i,ns,1),tau_w_sul);
// MPAS module_ra_cam.F:6503
aerssa(i,ns,2)=cam_add(aerssa(i,ns,2),tau_w_sslt);
// MPAS module_ra_cam.F:6504
aerssa(i,ns,3)=cam_add(cam_add(cam_add(aerssa(i,ns,3),tau_w_cphil),tau_w_cphob),tau_w_cb);
// MPAS module_ra_cam.F:6505
aerssa(i,ns,4)=cam_add(aerssa(i,ns,4),tau_w_dst_tot);
// MPAS module_ra_cam.F:6506
aerssa(i,ns,5)=cam_add(aerssa(i,ns,5),tau_w_bg);
// MPAS module_ra_cam.F:6507
aerssa(i,ns,6)=cam_add(aerssa(i,ns,6),tau_w_volc);
// MPAS module_ra_cam.F:6508
aerssa(i,ns,7)=cam_add(aerssa(i,ns,7),tau_w_tot);
// MPAS module_ra_cam.F:6510
aerasm(i,ns,1)=cam_add(aerasm(i,ns,1),tau_w_g_sul);
// MPAS module_ra_cam.F:6511
aerasm(i,ns,2)=cam_add(aerasm(i,ns,2),tau_w_g_sslt);
// MPAS module_ra_cam.F:6512
aerasm(i,ns,3)=cam_add(cam_add(cam_add(aerasm(i,ns,3),tau_w_g_cphil),tau_w_g_cphob),tau_w_g_cb);
// MPAS module_ra_cam.F:6513
aerasm(i,ns,4)=cam_add(aerasm(i,ns,4),tau_w_g_dst_tot);
// MPAS module_ra_cam.F:6514
aerasm(i,ns,5)=cam_add(aerasm(i,ns,5),tau_w_g_bg);
// MPAS module_ra_cam.F:6515
aerasm(i,ns,6)=cam_add(aerasm(i,ns,6),tau_w_g_volc);
// MPAS module_ra_cam.F:6516
aerasm(i,ns,7)=cam_add(aerasm(i,ns,7),tau_w_g_tot);
// MPAS module_ra_cam.F:6518
aerfwd(i,ns,1)=cam_add(aerfwd(i,ns,1),tau_w_f_sul);
// MPAS module_ra_cam.F:6519
aerfwd(i,ns,2)=cam_add(aerfwd(i,ns,2),tau_w_f_sslt);
// MPAS module_ra_cam.F:6520
aerfwd(i,ns,3)=cam_add(cam_add(cam_add(aerfwd(i,ns,3),tau_w_f_cphil),tau_w_f_cphob),tau_w_f_cb);
// MPAS module_ra_cam.F:6521
aerfwd(i,ns,4)=cam_add(aerfwd(i,ns,4),tau_w_f_dst_tot);
// MPAS module_ra_cam.F:6522
aerfwd(i,ns,5)=cam_add(aerfwd(i,ns,5),tau_w_f_bg);
// MPAS module_ra_cam.F:6523
aerfwd(i,ns,6)=cam_add(aerfwd(i,ns,6),tau_w_f_volc);
// MPAS module_ra_cam.F:6524
aerfwd(i,ns,7)=cam_add(aerfwd(i,ns,7),tau_w_f_tot);
// MPAS module_ra_cam.F:6529
}
// MPAS module_ra_cam.F:6532
for(kaer=1;kaer<=naer_groups;kaer+=1) {
// MPAS module_ra_cam.F:6534
if((aerssa(i,ns,kaer)>0.0f)) {
// MPAS module_ra_cam.F:6535
aerasm(i,ns,kaer)=cam_div(aerasm(i,ns,kaer),aerssa(i,ns,kaer));
// MPAS module_ra_cam.F:6536
aerfwd(i,ns,kaer)=cam_div(aerfwd(i,ns,kaer),aerssa(i,ns,kaer));
// MPAS module_ra_cam.F:6537
} else {
// MPAS module_ra_cam.F:6538
aerasm(i,ns,kaer)=0.0;
// MPAS module_ra_cam.F:6539
aerfwd(i,ns,kaer)=0.0;
// MPAS module_ra_cam.F:6540
}
// MPAS module_ra_cam.F:6542
if((aertau(i,ns,kaer)>0.0f)) {
// MPAS module_ra_cam.F:6543
aerssa(i,ns,kaer)=cam_div(aerssa(i,ns,kaer),aertau(i,ns,kaer));
// MPAS module_ra_cam.F:6544
} else {
// MPAS module_ra_cam.F:6545
aerssa(i,ns,kaer)=0.0;
// MPAS module_ra_cam.F:6546
}
// MPAS module_ra_cam.F:6548
}
// MPAS module_ra_cam.F:6554
}
// MPAS module_ra_cam.F:6559
wavmid(ns)=cam_mul(0.5,(cam_add(wavmin(ns),wavmax(ns))));
// MPAS module_ra_cam.F:6563
if((wavmid(ns)<0.7)) {
// MPAS module_ra_cam.F:6564
for(n=1;n<=ndayc;n+=1) {
// MPAS module_ra_cam.F:6565
i=idayc(n);
// MPAS module_ra_cam.F:6566
albdir(i,ns)=asdir(i);
// MPAS module_ra_cam.F:6567
albdif(i,ns)=asdif(i);
// MPAS module_ra_cam.F:6568
}
// MPAS module_ra_cam.F:6572
} else {
// MPAS module_ra_cam.F:6573
for(n=1;n<=ndayc;n+=1) {
// MPAS module_ra_cam.F:6574
i=idayc(n);
// MPAS module_ra_cam.F:6575
albdir(i,ns)=aldir(i);
// MPAS module_ra_cam.F:6576
albdif(i,ns)=aldif(i);
// MPAS module_ra_cam.F:6577
}
// MPAS module_ra_cam.F:6578
}
// MPAS module_ra_cam.F:6579
trayoslp=cam_div(raytau(ns),sslp);
// MPAS module_ra_cam.F:6585
raddedmx(pver,pverp,pcols,coszrs,ndayc,idayc,abh2o(ns),abo3(ns),abco2(ns),abo2(ns),uh2o,uo3,uco2,uo2,trayoslp,pflx,ns,tauxcl,wcl,gcl,fcl,tauxci,wci,gci,fci,tauxar,wa,ga,fa,rdir,rdif,tdir,tdif,explay,rdirc,rdifc,tdirc,tdifc,explayc); if(error) return;
// MPAS module_ra_cam.F:6597
}
// MPAS module_ra_cam.F:6624
for(n=1;n<=ndayc;n+=1) {
// MPAS module_ra_cam.F:6625
i=idayc(n);
// MPAS module_ra_cam.F:6673
npasses=0;
// MPAS module_ra_cam.F:6674
while(true) {
// MPAS module_ra_cam.F:6675
for(irgn=0;irgn<=nmxrgn(i);irgn+=1) {
// MPAS module_ra_cam.F:6676
kx2(irgn)=0;
// MPAS module_ra_cam.F:6677
}
// MPAS module_ra_cam.F:6678
mrgn=0;
// MPAS module_ra_cam.F:6682
for(irgn=1;irgn<=nmxrgn(i);irgn+=1) {
// MPAS module_ra_cam.F:6686
region_found=false;
// MPAS module_ra_cam.F:6687
if((kx2(cam_sub(irgn,1))<pver)) {
// MPAS module_ra_cam.F:6688
k1=cam_add(kx2(cam_sub(irgn,1)),1);
// MPAS module_ra_cam.F:6689
kx1(irgn)=k1;
// MPAS module_ra_cam.F:6690
kx2(irgn)=cam_sub(k1,1);
// MPAS module_ra_cam.F:6691
for(k2=pver;k2>=k1;k2+=cam_neg(1)) {
// MPAS module_ra_cam.F:6692
if((pmid(i,k2)<=pmxrgn(i,irgn))) {
// MPAS module_ra_cam.F:6693
kx2(irgn)=k2;
// MPAS module_ra_cam.F:6694
mrgn=cam_add(mrgn,1);
// MPAS module_ra_cam.F:6695
region_found=true;
// MPAS module_ra_cam.F:6696
break;
// MPAS module_ra_cam.F:6697
}
// MPAS module_ra_cam.F:6698
}
// MPAS module_ra_cam.F:6699
} else {
// MPAS module_ra_cam.F:6700
break;
// MPAS module_ra_cam.F:6701
}
// MPAS module_ra_cam.F:6703
if(region_found) {
// MPAS module_ra_cam.F:6707
nxs=0;
// MPAS module_ra_cam.F:6708
if((cldeps>0)) {
// MPAS module_ra_cam.F:6709
for(k=k1;k<=k2;k+=1) {
// MPAS module_ra_cam.F:6710
if(((cld(i,k)>=cldmin)&&(cld(i,k)>=cldeps))) {
// MPAS module_ra_cam.F:6711
nxs=cam_add(nxs,1);
// MPAS module_ra_cam.F:6712
ksort(nxs)=k;
// MPAS module_ra_cam.F:6717
asort(nxs)=cam_sub(1.0,(cam_mul(floor(cam_div(cld(i,k),cldeps)),cldeps)));
// MPAS module_ra_cam.F:6718
}
// MPAS module_ra_cam.F:6719
}
// MPAS module_ra_cam.F:6720
} else {
// MPAS module_ra_cam.F:6721
for(k=k1;k<=k2;k+=1) {
// MPAS module_ra_cam.F:6722
if((cld(i,k)>=cldmin)) {
// MPAS module_ra_cam.F:6723
nxs=cam_add(nxs,1);
// MPAS module_ra_cam.F:6724
ksort(nxs)=k;
// MPAS module_ra_cam.F:6729
asort(nxs)=cam_sub(1.0,cld(i,k));
// MPAS module_ra_cam.F:6730
}
// MPAS module_ra_cam.F:6731
}
// MPAS module_ra_cam.F:6732
}
// MPAS module_ra_cam.F:6738
if((nxs==2)) {
// MPAS module_ra_cam.F:6739
if((asort(2)<asort(1))) {
// MPAS module_ra_cam.F:6740
ktmp=ksort(1);
// MPAS module_ra_cam.F:6741
ksort(1)=ksort(2);
// MPAS module_ra_cam.F:6742
ksort(2)=ktmp;
// MPAS module_ra_cam.F:6744
atmp=asort(1);
// MPAS module_ra_cam.F:6745
asort(1)=asort(2);
// MPAS module_ra_cam.F:6746
asort(2)=atmp;
// MPAS module_ra_cam.F:6747
}
// MPAS module_ra_cam.F:6748
} else if((nxs>=3)) {
// MPAS module_ra_cam.F:6749
sortarray(nxs,asort,ksort); if(error) return;
// MPAS module_ra_cam.F:6750
}
// MPAS module_ra_cam.F:6754
{ auto _lhs=cstr.section(cam_section(k1,k2),cam_section(1,cam_add(nxs,1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0; }
// MPAS module_ra_cam.F:6755
mstr=1;
// MPAS module_ra_cam.F:6756
cld0=0.0;
// MPAS module_ra_cam.F:6757
for(l=1;l<=nxs;l+=1) {
// MPAS module_ra_cam.F:6758
if((asort(l)!=cld0)) {
// MPAS module_ra_cam.F:6759
wstr(mstr,mrgn)=cam_sub(asort(l),cld0);
// MPAS module_ra_cam.F:6760
cld0=asort(l);
// MPAS module_ra_cam.F:6761
mstr=cam_add(mstr,1);
// MPAS module_ra_cam.F:6762
}
// MPAS module_ra_cam.F:6763
{ auto _lhs=cstr.section(ksort(l),cam_section(mstr,cam_add(nxs,1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=1; }
// MPAS module_ra_cam.F:6764
}
// MPAS module_ra_cam.F:6765
nstr(mrgn)=mstr;
// MPAS module_ra_cam.F:6766
wstr(mstr,mrgn)=cam_sub(1.0,cld0);
// MPAS module_ra_cam.F:6770
}
// MPAS module_ra_cam.F:6774
}
// MPAS module_ra_cam.F:6775
nrgn=mrgn;
// MPAS module_ra_cam.F:6779
{ auto _lhs=cstr.section(0,cam_section(1,nstr(1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0; }
// MPAS module_ra_cam.F:6798
nconfigm=cam_product(nstr.section(cam_section(1,nrgn)));
// MPAS module_ra_cam.F:6802
{ auto _lhs=istr.section(cam_section(1,nrgn)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=1; }
// MPAS module_ra_cam.F:6803
nconfig=0;
// MPAS module_ra_cam.F:6804
totwgt=0.0;
// MPAS module_ra_cam.F:6805
new_term=true;
// MPAS module_ra_cam.F:6798-6847: exact upper bound for rejected permutations.
{
bool _weights_valid=(nrgn>0 && nrgn<=nstr.count && nrgn<=wstr.extent[1]);
double _largest_weight=1.0;
if(_weights_valid) {
  for(int _region=1;_region<=nrgn;++_region) {
    int _streams=nstr(_region);
    if(_streams<1 || _streams>wstr.extent[0]) {_weights_valid=false;break;}
    double _largest=0.0;
    for(int _stream=1;_stream<=_streams;++_stream) {
      double _weight=wstr(_stream,_region);
      if(!(_weight>=0.0 && _weight<=1.0)) {_weights_valid=false;break;}
      _largest=cam_max(_largest,_weight);
    }
    if(!_weights_valid) break;
    _largest_weight=cam_mul(_largest_weight,_largest);
  }
}
if(!_weights_valid || !(_largest_weight<areamin)) {
// MPAS module_ra_cam.F:6806
for(iconfig=1;iconfig<=nconfigm;iconfig+=1) {
// MPAS module_ra_cam.F:6807
xwgt=1.0;
// MPAS module_ra_cam.F:6808
for(mrgn=1;mrgn<=nrgn;mrgn+=1) {
// MPAS module_ra_cam.F:6809
xwgt=cam_mul(xwgt,wstr(istr(mrgn),mrgn));
// MPAS module_ra_cam.F:6810
}
// MPAS module_ra_cam.F:6811
if((xwgt>=areamin)) {
// MPAS module_ra_cam.F:6812
nconfig=cam_add(nconfig,1);
// MPAS module_ra_cam.F:6813
if((nconfig<=nconfgmax)) {
// MPAS module_ra_cam.F:6814
j=nconfig;
// MPAS module_ra_cam.F:6815
ptrc(nconfig)=nconfig;
// MPAS module_ra_cam.F:6816
} else {
// MPAS module_ra_cam.F:6817
nconfig=nconfgmax;
// MPAS module_ra_cam.F:6818
if(new_term) {
// MPAS module_ra_cam.F:6819
j=findvalue(1,nconfig,wgtv,ptrc);
// MPAS module_ra_cam.F:6820
}
// MPAS module_ra_cam.F:6821
if((wgtv(j)<xwgt)) {
// MPAS module_ra_cam.F:6822
totwgt=cam_sub(totwgt,wgtv(j));
// MPAS module_ra_cam.F:6823
new_term=true;
// MPAS module_ra_cam.F:6824
} else {
// MPAS module_ra_cam.F:6825
new_term=false;
// MPAS module_ra_cam.F:6826
}
// MPAS module_ra_cam.F:6827
}
// MPAS module_ra_cam.F:6828
if(new_term) {
// MPAS module_ra_cam.F:6829
wgtv(j)=xwgt;
// MPAS module_ra_cam.F:6830
totwgt=cam_add(totwgt,xwgt);
// MPAS module_ra_cam.F:6831
for(mrgn=1;mrgn<=nrgn;mrgn+=1) {
// MPAS module_ra_cam.F:6832
{ auto _lhs=ccon.section(cam_section(kx1(mrgn),kx2(mrgn)),j); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cstr.section(cam_section(kx1(mrgn),kx2(mrgn)),istr(mrgn)).linear(_slice); }
// MPAS module_ra_cam.F:6833
}
// MPAS module_ra_cam.F:6834
}
// MPAS module_ra_cam.F:6835
}
// MPAS module_ra_cam.F:6837
mrgn=nrgn;
// MPAS module_ra_cam.F:6838
istr(mrgn)=cam_add(istr(mrgn),1);
// MPAS module_ra_cam.F:6839
while(((istr(mrgn)>nstr(mrgn))&&(mrgn>1))) {
// MPAS module_ra_cam.F:6840
istr(mrgn)=1;
// MPAS module_ra_cam.F:6841
mrgn=cam_sub(mrgn,1);
// MPAS module_ra_cam.F:6842
istr(mrgn)=cam_add(istr(mrgn),1);
// MPAS module_ra_cam.F:6843
}
// MPAS module_ra_cam.F:6847
}
} else if(nconfigm>0) {
  // Preserve the native permutation counter's final automatic-array words.
  int _remaining=nconfigm;
  for(int _region=nrgn;_region>=2;--_region) {
    istr(_region)=cam_add(1,cam_mod(_remaining,nstr(_region)));
    _remaining=cam_div(_remaining,nstr(_region));
  }
  istr(1)=cam_add(1,_remaining);
}
}
// MPAS module_ra_cam.F:6852
if((totwgt>0.f)) {
// MPAS module_ra_cam.F:6853
break;
// MPAS module_ra_cam.F:6854
} else {
// MPAS module_ra_cam.F:6855
npasses=cam_add(npasses,1);
// MPAS module_ra_cam.F:6856
if((npasses>=2)) {
// MPAS module_ra_cam.F:6857
// MPAS diagnostic only
// MPAS module_ra_cam.F:6859
}
// MPAS module_ra_cam.F:6860
nmxrgn(i)=1;
// MPAS module_ra_cam.F:6861
pmxrgn(i,1)=1.0e30f;
// MPAS module_ra_cam.F:6862
}
// MPAS module_ra_cam.F:6866
}
// MPAS module_ra_cam.F:6871
{ auto _lhs=ccon.section(0,cam_section(1,nconfgmax)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0; }
// MPAS module_ra_cam.F:6872
{ auto _lhs=ccon.section(pverp,cam_section(1,nconfgmax)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0; }
// MPAS module_ra_cam.F:6876
nuniqd(0)=1;
// MPAS module_ra_cam.F:6877
nuniqu(pverp)=1;
// MPAS module_ra_cam.F:6879
istrtd(0,1)=1;
// MPAS module_ra_cam.F:6880
istrtu(pverp,1)=1;
// MPAS module_ra_cam.F:6882
for(j=1;j<=nconfig;j+=1) {
// MPAS module_ra_cam.F:6883
icond(0,j)=j;
// MPAS module_ra_cam.F:6884
iconu(pverp,j)=j;
// MPAS module_ra_cam.F:6885
}
// MPAS module_ra_cam.F:6887
istrtd(0,2)=cam_add(nconfig,1);
// MPAS module_ra_cam.F:6888
istrtu(pverp,2)=cam_add(nconfig,1);
// MPAS module_ra_cam.F:6890
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:6891
km1=cam_sub(k,1);
// MPAS module_ra_cam.F:6892
nuniq=0;
// MPAS module_ra_cam.F:6893
istrtd(k,1)=1;
// MPAS module_ra_cam.F:6894
for(l0=1;l0<=nuniqd(km1);l0+=1) {
// MPAS module_ra_cam.F:6895
is0=istrtd(km1,l0);
// MPAS module_ra_cam.F:6896
is1=cam_sub(istrtd(km1,cam_add(l0,1)),1);
// MPAS module_ra_cam.F:6897
n0=0;
// MPAS module_ra_cam.F:6898
n1=0;
// MPAS module_ra_cam.F:6899
for(isn=is0;isn<=is1;isn+=1) {
// MPAS module_ra_cam.F:6900
j=icond(km1,isn);
// MPAS module_ra_cam.F:6901
if((ccon(k,j)==0)) {
// MPAS module_ra_cam.F:6902
n0=cam_add(n0,1);
// MPAS module_ra_cam.F:6903
ptr0(n0)=j;
// MPAS module_ra_cam.F:6904
}
// MPAS module_ra_cam.F:6905
if((ccon(k,j)==1)) {
// MPAS module_ra_cam.F:6906
n1=cam_add(n1,1);
// MPAS module_ra_cam.F:6907
ptr1(n1)=j;
// MPAS module_ra_cam.F:6908
}
// MPAS module_ra_cam.F:6909
}
// MPAS module_ra_cam.F:6910
if((n0>0)) {
// MPAS module_ra_cam.F:6911
nuniq=cam_add(nuniq,1);
// MPAS module_ra_cam.F:6912
istrtd(k,cam_add(nuniq,1))=cam_add(istrtd(k,nuniq),n0);
// MPAS module_ra_cam.F:6913
{ auto _lhs=icond.section(k,cam_section(istrtd(k,nuniq),cam_sub(istrtd(k,cam_add(nuniq,1)),1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=ptr0.section(cam_section(1,n0)).linear(_slice); }
// MPAS module_ra_cam.F:6914
}
// MPAS module_ra_cam.F:6915
if((n1>0)) {
// MPAS module_ra_cam.F:6916
nuniq=cam_add(nuniq,1);
// MPAS module_ra_cam.F:6917
istrtd(k,cam_add(nuniq,1))=cam_add(istrtd(k,nuniq),n1);
// MPAS module_ra_cam.F:6918
{ auto _lhs=icond.section(k,cam_section(istrtd(k,nuniq),cam_sub(istrtd(k,cam_add(nuniq,1)),1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=ptr1.section(cam_section(1,n1)).linear(_slice); }
// MPAS module_ra_cam.F:6919
}
// MPAS module_ra_cam.F:6920
}
// MPAS module_ra_cam.F:6921
nuniqd(k)=nuniq;
// MPAS module_ra_cam.F:6922
}
// MPAS module_ra_cam.F:6924
for(k=pver;k>=0;k+=cam_neg(1)) {
// MPAS module_ra_cam.F:6925
kp1=cam_add(k,1);
// MPAS module_ra_cam.F:6926
nuniq=0;
// MPAS module_ra_cam.F:6927
istrtu(k,1)=1;
// MPAS module_ra_cam.F:6928
for(l0=1;l0<=nuniqu(kp1);l0+=1) {
// MPAS module_ra_cam.F:6929
is0=istrtu(kp1,l0);
// MPAS module_ra_cam.F:6930
is1=cam_sub(istrtu(kp1,cam_add(l0,1)),1);
// MPAS module_ra_cam.F:6931
n0=0;
// MPAS module_ra_cam.F:6932
n1=0;
// MPAS module_ra_cam.F:6933
for(isn=is0;isn<=is1;isn+=1) {
// MPAS module_ra_cam.F:6934
j=iconu(kp1,isn);
// MPAS module_ra_cam.F:6935
if((ccon(k,j)==0)) {
// MPAS module_ra_cam.F:6936
n0=cam_add(n0,1);
// MPAS module_ra_cam.F:6937
ptr0(n0)=j;
// MPAS module_ra_cam.F:6938
}
// MPAS module_ra_cam.F:6939
if((ccon(k,j)==1)) {
// MPAS module_ra_cam.F:6940
n1=cam_add(n1,1);
// MPAS module_ra_cam.F:6941
ptr1(n1)=j;
// MPAS module_ra_cam.F:6942
}
// MPAS module_ra_cam.F:6943
}
// MPAS module_ra_cam.F:6944
if((n0>0)) {
// MPAS module_ra_cam.F:6945
nuniq=cam_add(nuniq,1);
// MPAS module_ra_cam.F:6946
istrtu(k,cam_add(nuniq,1))=cam_add(istrtu(k,nuniq),n0);
// MPAS module_ra_cam.F:6947
{ auto _lhs=iconu.section(k,cam_section(istrtu(k,nuniq),cam_sub(istrtu(k,cam_add(nuniq,1)),1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=ptr0.section(cam_section(1,n0)).linear(_slice); }
// MPAS module_ra_cam.F:6948
}
// MPAS module_ra_cam.F:6949
if((n1>0)) {
// MPAS module_ra_cam.F:6950
nuniq=cam_add(nuniq,1);
// MPAS module_ra_cam.F:6951
istrtu(k,cam_add(nuniq,1))=cam_add(istrtu(k,nuniq),n1);
// MPAS module_ra_cam.F:6952
{ auto _lhs=iconu.section(k,cam_section(istrtu(k,nuniq),cam_sub(istrtu(k,cam_add(nuniq,1)),1))); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=ptr1.section(cam_section(1,n1)).linear(_slice); }
// MPAS module_ra_cam.F:6953
}
// MPAS module_ra_cam.F:6954
}
// MPAS module_ra_cam.F:6955
nuniqu(k)=nuniq;
// MPAS module_ra_cam.F:6956
}
// MPAS module_ra_cam.F:6969
for(k=0;k<=pver;k+=1) {
// MPAS module_ra_cam.F:6970
totfld(k)=0.0;
// MPAS module_ra_cam.F:6971
fswup(k)=0.0;
// MPAS module_ra_cam.F:6972
fswdn(k)=0.0;
// MPAS module_ra_cam.F:6973
fswupc(k)=0.0;
// MPAS module_ra_cam.F:6974
fswdnc(k)=0.0;
// MPAS module_ra_cam.F:6975
}
// MPAS module_ra_cam.F:6977
sfltot=0.0;
// MPAS module_ra_cam.F:6978
fswup(pverp)=0.0;
// MPAS module_ra_cam.F:6979
fswdn(pverp)=0.0;
// MPAS module_ra_cam.F:6980
fswupc(pverp)=0.0;
// MPAS module_ra_cam.F:6981
fswdnc(pverp)=0.0;
// MPAS module_ra_cam.F:6985
for(ns=1;ns<=nspint;ns+=1) {
// MPAS module_ra_cam.F:6986
wgtint=nirwgt(ns);
// MPAS module_ra_cam.F:6995
{ auto _lhs=rdndif.section(0,cam_section(1,nconfig)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0.0; }
// MPAS module_ra_cam.F:6996
{ auto _lhs=exptdn.section(0,cam_section(1,nconfig)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=1.0; }
// MPAS module_ra_cam.F:6997
{ auto _lhs=tdntot.section(0,cam_section(1,nconfig)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=1.0; }
// MPAS module_ra_cam.F:7006
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:7007
km1=cam_sub(k,1);
// MPAS module_ra_cam.F:7008
for(l0=1;l0<=nuniqd(km1);l0+=1) {
// MPAS module_ra_cam.F:7009
is0=istrtd(km1,l0);
// MPAS module_ra_cam.F:7010
is1=cam_sub(istrtd(km1,cam_add(l0,1)),1);
// MPAS module_ra_cam.F:7012
j=icond(km1,is0);
// MPAS module_ra_cam.F:7014
xexpt=exptdn(km1,j);
// MPAS module_ra_cam.F:7015
xrdnd=rdndif(km1,j);
// MPAS module_ra_cam.F:7016
tdnmexp=cam_sub(tdntot(km1,j),xexpt);
// MPAS module_ra_cam.F:7018
if((ccon(km1,j)==1)) {
// MPAS module_ra_cam.F:7022
ytdnd=tdif(ns,i,km1);
// MPAS module_ra_cam.F:7023
yrdnd=rdif(ns,i,km1);
// MPAS module_ra_cam.F:7025
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(yrdnd,xrdnd))));
// MPAS module_ra_cam.F:7026
rdirexp=cam_mul(rdir(ns,i,km1),xexpt);
// MPAS module_ra_cam.F:7028
zexpt=cam_mul(xexpt,explay(ns,i,km1));
// MPAS module_ra_cam.F:7029
zrdnd=cam_add(yrdnd,cam_mul(cam_mul(xrdnd,(cam_pow(ytdnd,2))),rdenom));
// MPAS module_ra_cam.F:7030
ztdnt=cam_add(cam_mul(xexpt,tdir(ns,i,km1)),cam_mul(cam_mul(ytdnd,(cam_add(tdnmexp,cam_mul(xrdnd,rdirexp)))),rdenom));
// MPAS module_ra_cam.F:7031
} else {
// MPAS module_ra_cam.F:7035
ytdnd=tdifc(ns,i,km1);
// MPAS module_ra_cam.F:7036
yrdnd=rdifc(ns,i,km1);
// MPAS module_ra_cam.F:7038
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(yrdnd,xrdnd))));
// MPAS module_ra_cam.F:7039
rdirexp=cam_mul(rdirc(ns,i,km1),xexpt);
// MPAS module_ra_cam.F:7041
zexpt=cam_mul(xexpt,explayc(ns,i,km1));
// MPAS module_ra_cam.F:7042
zrdnd=cam_add(yrdnd,cam_mul(cam_mul(xrdnd,(cam_pow(ytdnd,2))),rdenom));
// MPAS module_ra_cam.F:7043
ztdnt=cam_add(cam_mul(xexpt,tdirc(ns,i,km1)),cam_mul(cam_mul(ytdnd,(cam_add(tdnmexp,cam_mul(xrdnd,rdirexp)))),rdenom));
// MPAS module_ra_cam.F:7045
}
// MPAS module_ra_cam.F:7052
for(isn=is0;isn<=is1;isn+=1) {
// MPAS module_ra_cam.F:7053
j=icond(km1,isn);
// MPAS module_ra_cam.F:7054
exptdn(k,j)=zexpt;
// MPAS module_ra_cam.F:7055
rdndif(k,j)=zrdnd;
// MPAS module_ra_cam.F:7056
tdntot(k,j)=ztdnt;
// MPAS module_ra_cam.F:7057
}
// MPAS module_ra_cam.F:7061
}
// MPAS module_ra_cam.F:7065
}
// MPAS module_ra_cam.F:7075
{ auto _lhs=rupdir.section(pverp,cam_section(1,nconfig)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=albdir(i,ns); }
// MPAS module_ra_cam.F:7076
{ auto _lhs=rupdif.section(pverp,cam_section(1,nconfig)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=albdif(i,ns); }
// MPAS module_ra_cam.F:7078
for(k=pver;k>=0;k+=cam_neg(1)) {
// MPAS module_ra_cam.F:7079
for(l0=1;l0<=nuniqu(k);l0+=1) {
// MPAS module_ra_cam.F:7080
is0=istrtu(k,l0);
// MPAS module_ra_cam.F:7081
is1=cam_sub(istrtu(k,cam_add(l0,1)),1);
// MPAS module_ra_cam.F:7083
j=iconu(k,is0);
// MPAS module_ra_cam.F:7085
xrupd=rupdif(cam_add(k,1),j);
// MPAS module_ra_cam.F:7086
xrups=rupdir(cam_add(k,1),j);
// MPAS module_ra_cam.F:7088
if((ccon(k,j)==1)) {
// MPAS module_ra_cam.F:7092
yexpt=explay(ns,i,k);
// MPAS module_ra_cam.F:7093
yrupd=rdif(ns,i,k);
// MPAS module_ra_cam.F:7094
ytupd=tdif(ns,i,k);
// MPAS module_ra_cam.F:7096
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(yrupd,xrupd))));
// MPAS module_ra_cam.F:7097
tdnmexp=(cam_sub(tdir(ns,i,k),yexpt));
// MPAS module_ra_cam.F:7098
rdirexp=cam_mul(xrups,yexpt);
// MPAS module_ra_cam.F:7100
zrupd=cam_add(yrupd,cam_mul(cam_mul(xrupd,(cam_pow(ytupd,2))),rdenom));
// MPAS module_ra_cam.F:7101
zrups=cam_add(rdir(ns,i,k),cam_mul(cam_mul(ytupd,(cam_add(rdirexp,cam_mul(xrupd,tdnmexp)))),rdenom));
// MPAS module_ra_cam.F:7102
} else {
// MPAS module_ra_cam.F:7106
yexpt=explayc(ns,i,k);
// MPAS module_ra_cam.F:7107
yrupd=rdifc(ns,i,k);
// MPAS module_ra_cam.F:7108
ytupd=tdifc(ns,i,k);
// MPAS module_ra_cam.F:7110
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(yrupd,xrupd))));
// MPAS module_ra_cam.F:7111
tdnmexp=(cam_sub(tdirc(ns,i,k),yexpt));
// MPAS module_ra_cam.F:7112
rdirexp=cam_mul(xrups,yexpt);
// MPAS module_ra_cam.F:7114
zrupd=cam_add(yrupd,cam_mul(cam_mul(xrupd,(cam_pow(ytupd,2))),rdenom));
// MPAS module_ra_cam.F:7115
zrups=cam_add(rdirc(ns,i,k),cam_mul(cam_mul(ytupd,(cam_add(rdirexp,cam_mul(xrupd,tdnmexp)))),rdenom));
// MPAS module_ra_cam.F:7116
}
// MPAS module_ra_cam.F:7123
for(isn=is0;isn<=is1;isn+=1) {
// MPAS module_ra_cam.F:7124
j=iconu(k,isn);
// MPAS module_ra_cam.F:7125
rupdif(k,j)=zrupd;
// MPAS module_ra_cam.F:7126
rupdir(k,j)=zrups;
// MPAS module_ra_cam.F:7127
}
// MPAS module_ra_cam.F:7131
}
// MPAS module_ra_cam.F:7135
}
// MPAS module_ra_cam.F:7149
for(k=0;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:7153
fluxup(k)=0.0;
// MPAS module_ra_cam.F:7154
fluxdn(k)=0.0;
// MPAS module_ra_cam.F:7156
for(iconfig=1;iconfig<=nconfig;iconfig+=1) {
// MPAS module_ra_cam.F:7157
xwgt=wgtv(iconfig);
// MPAS module_ra_cam.F:7158
xexpt=exptdn(k,iconfig);
// MPAS module_ra_cam.F:7159
xtdnt=tdntot(k,iconfig);
// MPAS module_ra_cam.F:7160
xrdnd=rdndif(k,iconfig);
// MPAS module_ra_cam.F:7161
xrupd=rupdif(k,iconfig);
// MPAS module_ra_cam.F:7162
xrups=rupdir(k,iconfig);
// MPAS module_ra_cam.F:7166
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(xrdnd,xrupd))));
// MPAS module_ra_cam.F:7168
fluxup(k)=cam_add(fluxup(k),cam_mul(xwgt,(cam_mul((cam_add(cam_mul(xexpt,xrups),cam_mul((cam_sub(xtdnt,xexpt)),xrupd))),rdenom))));
// MPAS module_ra_cam.F:7170
fluxdn(k)=cam_add(fluxdn(k),cam_mul(xwgt,(cam_add(xexpt,cam_mul((cam_add(cam_sub(xtdnt,xexpt),cam_mul(cam_mul(xexpt,xrups),xrdnd))),rdenom)))));
// MPAS module_ra_cam.F:7175
}
// MPAS module_ra_cam.F:7180
fluxup(k)=cam_div(fluxup(k),totwgt);
// MPAS module_ra_cam.F:7181
fluxdn(k)=cam_div(fluxdn(k),totwgt);
// MPAS module_ra_cam.F:7185
}
// MPAS module_ra_cam.F:7189
wexptdn=0.0;
// MPAS module_ra_cam.F:7191
for(iconfig=1;iconfig<=nconfig;iconfig+=1) {
// MPAS module_ra_cam.F:7192
wexptdn=cam_add(wexptdn,cam_mul(wgtv(iconfig),exptdn(pverp,iconfig)));
// MPAS module_ra_cam.F:7193
}
// MPAS module_ra_cam.F:7195
wexptdn=cam_div(wexptdn,totwgt);
// MPAS module_ra_cam.F:7199
solflx=cam_mul(cam_mul(solin(i),frcsol(ns)),psf(ns));
// MPAS module_ra_cam.F:7200
fsnt(i)=cam_add(fsnt(i),cam_mul(solflx,(cam_sub(fluxdn(1),fluxup(1)))));
// MPAS module_ra_cam.F:7201
fsntoa(i)=cam_add(fsntoa(i),cam_mul(solflx,(cam_sub(fluxdn(0),fluxup(0)))));
// MPAS module_ra_cam.F:7202
fsns(i)=cam_add(fsns(i),cam_mul(solflx,(cam_sub(fluxdn(pverp),fluxup(pverp)))));
// MPAS module_ra_cam.F:7203
sfltot=cam_add(sfltot,solflx);
// MPAS module_ra_cam.F:7204
fswup(0)=cam_add(fswup(0),cam_mul(solflx,fluxup(0)));
// MPAS module_ra_cam.F:7205
fswdn(0)=cam_add(fswdn(0),cam_mul(solflx,fluxdn(0)));
// MPAS module_ra_cam.F:7209
if((wavmid(ns)<0.7)) {
// MPAS module_ra_cam.F:7210
sols(i)=cam_add(sols(i),cam_mul(cam_mul(wexptdn,solflx),0.001));
// MPAS module_ra_cam.F:7211
solsd(i)=cam_add(solsd(i),cam_mul(cam_mul((cam_sub(fluxdn(pverp),wexptdn)),solflx),0.001));
// MPAS module_ra_cam.F:7212
} else {
// MPAS module_ra_cam.F:7213
soll(i)=cam_add(soll(i),cam_mul(cam_mul(wexptdn,solflx),0.001));
// MPAS module_ra_cam.F:7214
solld(i)=cam_add(solld(i),cam_mul(cam_mul((cam_sub(fluxdn(pverp),wexptdn)),solflx),0.001));
// MPAS module_ra_cam.F:7215
fsnrtoaq(i)=cam_add(fsnrtoaq(i),cam_mul(solflx,(cam_sub(fluxdn(0),fluxup(0)))));
// MPAS module_ra_cam.F:7216
}
// MPAS module_ra_cam.F:7217
fsnirtoa(i)=cam_add(fsnirtoa(i),cam_mul(cam_mul(wgtint,solflx),(cam_sub(fluxdn(0),fluxup(0)))));
// MPAS module_ra_cam.F:7219
for(k=0;k<=pver;k+=1) {
// MPAS module_ra_cam.F:7224
kp1=cam_add(k,1);
// MPAS module_ra_cam.F:7225
flxdiv=cam_add((cam_sub(fluxdn(k),fluxdn(kp1))),(cam_sub(fluxup(kp1),fluxup(k))));
// MPAS module_ra_cam.F:7226
totfld(k)=cam_add(totfld(k),cam_mul(solflx,flxdiv));
// MPAS module_ra_cam.F:7227
fswdn(kp1)=cam_add(fswdn(kp1),cam_mul(solflx,fluxdn(kp1)));
// MPAS module_ra_cam.F:7228
fswup(kp1)=cam_add(fswup(kp1),cam_mul(solflx,fluxup(kp1)));
// MPAS module_ra_cam.F:7229
}
// MPAS module_ra_cam.F:7233
exptdnc(0)=1.0;
// MPAS module_ra_cam.F:7234
rdndifc(0)=0.0;
// MPAS module_ra_cam.F:7235
tdntotc(0)=1.0;
// MPAS module_ra_cam.F:7236
rupdirc(pverp)=albdir(i,ns);
// MPAS module_ra_cam.F:7237
rupdifc(pverp)=albdif(i,ns);
// MPAS module_ra_cam.F:7239
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:7240
km1=cam_sub(k,1);
// MPAS module_ra_cam.F:7241
xexpt=exptdnc(km1);
// MPAS module_ra_cam.F:7242
xrdnd=rdndifc(km1);
// MPAS module_ra_cam.F:7243
yrdnd=rdifc(ns,i,km1);
// MPAS module_ra_cam.F:7244
ytdnd=tdifc(ns,i,km1);
// MPAS module_ra_cam.F:7246
exptdnc(k)=cam_mul(xexpt,explayc(ns,i,km1));
// MPAS module_ra_cam.F:7248
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(yrdnd,xrdnd))));
// MPAS module_ra_cam.F:7249
rdirexp=cam_mul(rdirc(ns,i,km1),xexpt);
// MPAS module_ra_cam.F:7250
tdnmexp=cam_sub(tdntotc(km1),xexpt);
// MPAS module_ra_cam.F:7252
tdntotc(k)=cam_add(cam_mul(xexpt,tdirc(ns,i,km1)),cam_mul(cam_mul(ytdnd,(cam_add(tdnmexp,cam_mul(xrdnd,rdirexp)))),rdenom));
// MPAS module_ra_cam.F:7254
rdndifc(k)=cam_add(yrdnd,cam_mul(cam_mul(xrdnd,(cam_pow(ytdnd,2))),rdenom));
// MPAS module_ra_cam.F:7255
}
// MPAS module_ra_cam.F:7257
for(k=pver;k>=0;k+=cam_neg(1)) {
// MPAS module_ra_cam.F:7258
xrupd=rupdifc(cam_add(k,1));
// MPAS module_ra_cam.F:7259
yexpt=explayc(ns,i,k);
// MPAS module_ra_cam.F:7260
yrupd=rdifc(ns,i,k);
// MPAS module_ra_cam.F:7261
ytupd=tdifc(ns,i,k);
// MPAS module_ra_cam.F:7263
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(yrupd,xrupd))));
// MPAS module_ra_cam.F:7265
rupdirc(k)=cam_add(rdirc(ns,i,k),cam_mul(cam_mul(ytupd,(cam_add(cam_mul(rupdirc(cam_add(k,1)),yexpt),cam_mul(xrupd,(cam_sub(tdirc(ns,i,k),yexpt)))))),rdenom));
// MPAS module_ra_cam.F:7267
rupdifc(k)=cam_add(yrupd,cam_mul(cam_mul(xrupd,cam_pow(ytupd,2)),rdenom));
// MPAS module_ra_cam.F:7268
}
// MPAS module_ra_cam.F:7270
for(k=0;k<=1;k+=1) {
// MPAS module_ra_cam.F:7271
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(rdndifc(k),rupdifc(k)))));
// MPAS module_ra_cam.F:7272
fluxup(k)=cam_mul((cam_add(cam_mul(exptdnc(k),rupdirc(k)),cam_mul((cam_sub(tdntotc(k),exptdnc(k))),rupdifc(k)))),rdenom);
// MPAS module_ra_cam.F:7274
fluxdn(k)=cam_add(exptdnc(k),cam_mul((cam_add(cam_sub(tdntotc(k),exptdnc(k)),cam_mul(cam_mul(exptdnc(k),rupdirc(k)),rdndifc(k)))),rdenom));
// MPAS module_ra_cam.F:7277
fswupc(k)=cam_add(fswupc(k),cam_mul(solflx,fluxup(k)));
// MPAS module_ra_cam.F:7278
fswdnc(k)=cam_add(fswdnc(k),cam_mul(solflx,fluxdn(k)));
// MPAS module_ra_cam.F:7279
}
// MPAS module_ra_cam.F:7281
for(k=2;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:7282
rdenom=cam_div(1.,(cam_sub(1.,cam_mul(rdndifc(k),rupdifc(k)))));
// MPAS module_ra_cam.F:7283
fluxup(k)=cam_mul((cam_add(cam_mul(exptdnc(k),rupdirc(k)),cam_mul((cam_sub(tdntotc(k),exptdnc(k))),rupdifc(k)))),rdenom);
// MPAS module_ra_cam.F:7285
fluxdn(k)=cam_add(exptdnc(k),cam_mul((cam_add(cam_sub(tdntotc(k),exptdnc(k)),cam_mul(cam_mul(exptdnc(k),rupdirc(k)),rdndifc(k)))),rdenom));
// MPAS module_ra_cam.F:7287
fswupc(k)=cam_add(fswupc(k),cam_mul(solflx,fluxup(k)));
// MPAS module_ra_cam.F:7288
fswdnc(k)=cam_add(fswdnc(k),cam_mul(solflx,fluxdn(k)));
// MPAS module_ra_cam.F:7289
}
// MPAS module_ra_cam.F:7291
fsntc(i)=cam_add(fsntc(i),cam_mul(solflx,(cam_sub(fluxdn(1),fluxup(1)))));
// MPAS module_ra_cam.F:7292
fsntoac(i)=cam_add(fsntoac(i),cam_mul(solflx,(cam_sub(fluxdn(0),fluxup(0)))));
// MPAS module_ra_cam.F:7293
fsnsc(i)=cam_add(fsnsc(i),cam_mul(solflx,(cam_sub(fluxdn(pverp),fluxup(pverp)))));
// MPAS module_ra_cam.F:7294
fsdsc(i)=cam_add(fsdsc(i),cam_mul(solflx,(fluxdn(pverp))));
// MPAS module_ra_cam.F:7295
fsnrtoac(i)=cam_add(fsnrtoac(i),cam_mul(cam_mul(wgtint,solflx),(cam_sub(fluxdn(0),fluxup(0)))));
// MPAS module_ra_cam.F:7303
}
// MPAS module_ra_cam.F:7307
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:7308
qrs(i,k)=cam_neg(cam_div(cam_mul(cam_mul(1.e-4f,gravit),totfld(k)),(cam_sub(pint(i,k),pint(i,cam_add(k,1))))));
// MPAS module_ra_cam.F:7309
}
// MPAS module_ra_cam.F:7313
for(k=1;k<=pverp;k+=1) {
// MPAS module_ra_cam.F:7314
fsup(i,k)=fswup(k);
// MPAS module_ra_cam.F:7315
fsupc(i,k)=fswupc(k);
// MPAS module_ra_cam.F:7316
fsdn(i,k)=fswdn(k);
// MPAS module_ra_cam.F:7317
fsdnc(i,k)=fswdnc(k);
// MPAS module_ra_cam.F:7318
}
// MPAS module_ra_cam.F:7322
fsds(i)=fswdn(pverp);
// MPAS module_ra_cam.F:7326
}
// MPAS module_ra_cam.F:7330
return;
}
// MPAS module_ra_cam.F:7333-7607
CAM_ROUTINE void raddedmx(const int& pver,const int& pverp,const int& pcols,const CamArray<double>& _coszrs,const int& ndayc,const CamArray<int>& _idayc,const double& abh2o,const double& abo3,const double& abco2,const double& abo2,const CamArray<double>& _uh2o,const CamArray<double>& _uo3,const CamArray<double>& _uco2,const CamArray<double>& _uo2,const double& trayoslp,const CamArray<double>& _pflx,const int& ns,const CamArray<double>& _tauxcl,const CamArray<double>& _wcl,const CamArray<double>& _gcl,const CamArray<double>& _fcl,const CamArray<double>& _tauxci,const CamArray<double>& _wci,const CamArray<double>& _gci,const CamArray<double>& _fci,const CamArray<double>& _tauxar,const CamArray<double>& _wa,const CamArray<double>& _ga,const CamArray<double>& _fa,const CamArray<double>& _rdir,const CamArray<double>& _rdif,const CamArray<double>& _tdir,const CamArray<double>& _tdif,const CamArray<double>& _explay,const CamArray<double>& _rdirc,const CamArray<double>& _rdifc,const CamArray<double>& _tdirc,const CamArray<double>& _tdifc,const CamArray<double>& _explayc) {
CamFrame _frame(arena);
const int nspint=19;
const double trmin=1.e-3f;
const double wray=0.999999f;
const double gray=0.0f;
const double fray=0.1f;
CAM_TRACE("raddedmx.coszrs");
auto coszrs=_coszrs.rebound(1,pcols);
CAM_TRACE("raddedmx.idayc");
auto idayc=_idayc.rebound(1,pcols);
CAM_TRACE("raddedmx.uh2o");
auto uh2o=_uh2o.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.uo3");
auto uo3=_uo3.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.uco2");
auto uco2=_uco2.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.uo2");
auto uo2=_uo2.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.pflx");
auto pflx=_pflx.rebound(1,pcols,0,pverp);
CAM_TRACE("raddedmx.tauxcl");
auto tauxcl=_tauxcl.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.wcl");
auto wcl=_wcl.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.gcl");
auto gcl=_gcl.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.fcl");
auto fcl=_fcl.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.tauxci");
auto tauxci=_tauxci.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.wci");
auto wci=_wci.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.gci");
auto gci=_gci.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.fci");
auto fci=_fci.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.tauxar");
auto tauxar=_tauxar.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.wa");
auto wa=_wa.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.ga");
auto ga=_ga.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.fa");
auto fa=_fa.rebound(1,pcols,0,pver);
CAM_TRACE("raddedmx.rdir");
auto rdir=_rdir.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.rdif");
auto rdif=_rdif.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.tdir");
auto tdir=_tdir.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.tdif");
auto tdif=_tdif.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.explay");
auto explay=_explay.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.rdirc");
auto rdirc=_rdirc.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.rdifc");
auto rdifc=_rdifc.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.tdirc");
auto tdirc=_tdirc.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.tdifc");
auto tdifc=_tdifc.rebound(1,nspint,1,pcols,0,pver);
CAM_TRACE("raddedmx.explayc");
auto explayc=_explayc.rebound(1,nspint,1,pcols,0,pver);
int i;
int k;
int nn;
auto taugab=arena.array<double>(1,pcols);
auto tauray=arena.array<double>(1,pcols);
double taucsc;
double tautot;
double wtot;
double gtot;
double ftot;
double wtau;
double wt;
double ts;
double ws;
double gs;
double lm;
double ne;
double w;
double uu;
double g;
double e;
double f;
double t;
double et;
double alp;
double gam;
double ue;
double arg;
double extins;
double amg;
double apg;
auto alpha=[&](double w,double uu,double g,double e) { return cam_mul(cam_mul(cam_mul(.75,w),uu),(cam_div((cam_add(1.,cam_mul(g,(cam_sub(1.,w))))),(cam_sub(1.,cam_mul(cam_mul(cam_mul(e,e),uu),uu)))))); };
auto gamma=[&](double w,double uu,double g,double e) { return cam_mul(cam_mul(.50,w),(cam_div((cam_add(cam_mul(cam_mul(cam_mul(cam_mul(3.,g),(cam_sub(1.,w))),uu),uu),1.)),(cam_sub(1.,cam_mul(cam_mul(cam_mul(e,e),uu),uu)))))); };
auto el=[&](double w,double g) { return cam_sqrt(cam_mul(cam_mul(3.,(cam_sub(1.,w))),(cam_sub(1.,cam_mul(w,g))))); };
auto taus=[&](double w,double f,double t) { return cam_mul((cam_sub(1.,cam_mul(w,f))),t); };
auto omgs=[&](double w,double f) { return cam_div(cam_mul((cam_sub(1.,f)),w),(cam_sub(1.,cam_mul(w,f)))); };
auto asys=[&](double g,double f) { return cam_div((cam_sub(g,f)),(cam_sub(1.,f))); };
auto u=[&](double w,double g,double e) { return cam_div(cam_mul(1.5,(cam_sub(1.,cam_mul(w,g)))),e); };
auto n=[&](double uu,double et) { return cam_sub((cam_div(cam_mul((cam_add(uu,1.)),(cam_add(uu,1.))),et)),(cam_mul(cam_mul((cam_sub(uu,1.)),(cam_sub(uu,1.))),et))); };
// MPAS module_ra_cam.F:7502
for(k=0;k<=pver;k+=1) {
// MPAS module_ra_cam.F:7503
for(nn=1;nn<=ndayc;nn+=1) {
// MPAS module_ra_cam.F:7504
i=idayc(nn);
// MPAS module_ra_cam.F:7505
tauray(i)=cam_mul(trayoslp,(cam_sub(pflx(i,cam_add(k,1)),pflx(i,k))));
// MPAS module_ra_cam.F:7506
taugab(i)=cam_add(cam_add(cam_add(cam_mul(abh2o,uh2o(i,k)),cam_mul(abo3,uo3(i,k))),cam_mul(abco2,uco2(i,k))),cam_mul(abo2,uo2(i,k)));
// MPAS module_ra_cam.F:7507
tautot=cam_add(cam_add(cam_add(cam_add(tauxcl(i,k),tauxci(i,k)),tauray(i)),taugab(i)),tauxar(i,k));
// MPAS module_ra_cam.F:7508
taucsc=cam_add(cam_add(cam_mul(tauxcl(i,k),wcl(i,k)),cam_mul(tauxci(i,k),wci(i,k))),cam_mul(tauxar(i,k),wa(i,k)));
// MPAS module_ra_cam.F:7509
wtau=cam_mul(wray,tauray(i));
// MPAS module_ra_cam.F:7510
wt=cam_add(wtau,taucsc);
// MPAS module_ra_cam.F:7511
wtot=cam_div(wt,tautot);
// MPAS module_ra_cam.F:7512
gtot=cam_div((cam_add(cam_add(cam_add(cam_mul(wtau,gray),cam_mul(cam_mul(gcl(i,k),wcl(i,k)),tauxcl(i,k))),cam_mul(cam_mul(gci(i,k),wci(i,k)),tauxci(i,k))),cam_mul(cam_mul(ga(i,k),wa(i,k)),tauxar(i,k)))),wt);
// MPAS module_ra_cam.F:7514
ftot=cam_div((cam_add(cam_add(cam_add(cam_mul(wtau,fray),cam_mul(cam_mul(fcl(i,k),wcl(i,k)),tauxcl(i,k))),cam_mul(cam_mul(fci(i,k),wci(i,k)),tauxci(i,k))),cam_mul(cam_mul(fa(i,k),wa(i,k)),tauxar(i,k)))),wt);
// MPAS module_ra_cam.F:7516
ts=taus(wtot,ftot,tautot);
// MPAS module_ra_cam.F:7517
ws=omgs(wtot,ftot);
// MPAS module_ra_cam.F:7518
gs=asys(gtot,ftot);
// MPAS module_ra_cam.F:7519
lm=el(ws,gs);
// MPAS module_ra_cam.F:7520
alp=alpha(ws,coszrs(i),gs,lm);
// MPAS module_ra_cam.F:7521
gam=gamma(ws,coszrs(i),gs,lm);
// MPAS module_ra_cam.F:7522
ue=u(ws,gs,lm);
// MPAS module_ra_cam.F:7526
arg=cam_min(cam_mul(lm,ts),25.);
// MPAS module_ra_cam.F:7527
extins=cam_exp(cam_neg(arg));
// MPAS module_ra_cam.F:7528
ne=n(ue,extins);
// MPAS module_ra_cam.F:7529
rdif(ns,i,k)=cam_div(cam_mul(cam_mul((cam_add(ue,1.)),(cam_sub(ue,1.))),(cam_sub(cam_div(1.,extins),extins))),ne);
// MPAS module_ra_cam.F:7530
tdif(ns,i,k)=cam_div(cam_mul(4.,ue),ne);
// MPAS module_ra_cam.F:7534
arg=cam_min(cam_div(ts,coszrs(i)),25.);
// MPAS module_ra_cam.F:7535
explay(ns,i,k)=cam_exp(cam_neg(arg));
// MPAS module_ra_cam.F:7536
apg=cam_add(alp,gam);
// MPAS module_ra_cam.F:7537
amg=cam_sub(alp,gam);
// MPAS module_ra_cam.F:7538
rdir(ns,i,k)=cam_add(cam_mul(amg,(cam_sub(cam_mul(tdif(ns,i,k),explay(ns,i,k)),1.))),cam_mul(apg,rdif(ns,i,k)));
// MPAS module_ra_cam.F:7539
tdir(ns,i,k)=cam_add(cam_mul(apg,tdif(ns,i,k)),cam_mul((cam_sub(cam_mul(amg,rdif(ns,i,k)),(cam_sub(apg,1.)))),explay(ns,i,k)));
// MPAS module_ra_cam.F:7544
rdir(ns,i,k)=cam_max(rdir(ns,i,k),0.0);
// MPAS module_ra_cam.F:7545
tdir(ns,i,k)=cam_max(tdir(ns,i,k),0.0);
// MPAS module_ra_cam.F:7546
rdif(ns,i,k)=cam_max(rdif(ns,i,k),0.0);
// MPAS module_ra_cam.F:7547
tdif(ns,i,k)=cam_max(tdif(ns,i,k),0.0);
// MPAS module_ra_cam.F:7551
if(((tauxcl(i,k)==0.0)&&(tauxci(i,k)==0.0))) {
// MPAS module_ra_cam.F:7553
rdirc(ns,i,k)=rdir(ns,i,k);
// MPAS module_ra_cam.F:7554
tdirc(ns,i,k)=tdir(ns,i,k);
// MPAS module_ra_cam.F:7555
rdifc(ns,i,k)=rdif(ns,i,k);
// MPAS module_ra_cam.F:7556
tdifc(ns,i,k)=tdif(ns,i,k);
// MPAS module_ra_cam.F:7557
explayc(ns,i,k)=explay(ns,i,k);
// MPAS module_ra_cam.F:7558
} else {
// MPAS module_ra_cam.F:7559
tautot=cam_add(cam_add(tauray(i),taugab(i)),tauxar(i,k));
// MPAS module_ra_cam.F:7560
taucsc=cam_mul(tauxar(i,k),wa(i,k));
// MPAS module_ra_cam.F:7564
wt=cam_add(wtau,taucsc);
// MPAS module_ra_cam.F:7565
wtot=cam_div(wt,tautot);
// MPAS module_ra_cam.F:7566
gtot=cam_div((cam_add(cam_mul(wtau,gray),cam_mul(cam_mul(ga(i,k),wa(i,k)),tauxar(i,k)))),wt);
// MPAS module_ra_cam.F:7567
ftot=cam_div((cam_add(cam_mul(wtau,fray),cam_mul(cam_mul(fa(i,k),wa(i,k)),tauxar(i,k)))),wt);
// MPAS module_ra_cam.F:7568
ts=taus(wtot,ftot,tautot);
// MPAS module_ra_cam.F:7569
ws=omgs(wtot,ftot);
// MPAS module_ra_cam.F:7570
gs=asys(gtot,ftot);
// MPAS module_ra_cam.F:7571
lm=el(ws,gs);
// MPAS module_ra_cam.F:7572
alp=alpha(ws,coszrs(i),gs,lm);
// MPAS module_ra_cam.F:7573
gam=gamma(ws,coszrs(i),gs,lm);
// MPAS module_ra_cam.F:7574
ue=u(ws,gs,lm);
// MPAS module_ra_cam.F:7578
arg=cam_min(cam_mul(lm,ts),25.);
// MPAS module_ra_cam.F:7579
extins=cam_exp(cam_neg(arg));
// MPAS module_ra_cam.F:7580
ne=n(ue,extins);
// MPAS module_ra_cam.F:7581
rdifc(ns,i,k)=cam_div(cam_mul(cam_mul((cam_add(ue,1.)),(cam_sub(ue,1.))),(cam_sub(cam_div(1.,extins),extins))),ne);
// MPAS module_ra_cam.F:7582
tdifc(ns,i,k)=cam_div(cam_mul(4.,ue),ne);
// MPAS module_ra_cam.F:7586
arg=cam_min(cam_div(ts,coszrs(i)),25.);
// MPAS module_ra_cam.F:7587
explayc(ns,i,k)=cam_exp(cam_neg(arg));
// MPAS module_ra_cam.F:7588
apg=cam_add(alp,gam);
// MPAS module_ra_cam.F:7589
amg=cam_sub(alp,gam);
// MPAS module_ra_cam.F:7590
rdirc(ns,i,k)=cam_add(cam_mul(amg,(cam_sub(cam_mul(tdifc(ns,i,k),explayc(ns,i,k)),1.))),cam_mul(apg,rdifc(ns,i,k)));
// MPAS module_ra_cam.F:7592
tdirc(ns,i,k)=cam_add(cam_mul(apg,tdifc(ns,i,k)),cam_mul((cam_sub(cam_mul(amg,rdifc(ns,i,k)),(cam_sub(apg,1.)))),explayc(ns,i,k)));
// MPAS module_ra_cam.F:7598
rdirc(ns,i,k)=cam_max(rdirc(ns,i,k),0.0);
// MPAS module_ra_cam.F:7599
tdirc(ns,i,k)=cam_max(tdirc(ns,i,k),0.0);
// MPAS module_ra_cam.F:7600
rdifc(ns,i,k)=cam_max(rdifc(ns,i,k),0.0);
// MPAS module_ra_cam.F:7601
tdifc(ns,i,k)=cam_max(tdifc(ns,i,k),0.0);
// MPAS module_ra_cam.F:7602
}
// MPAS module_ra_cam.F:7603
}
// MPAS module_ra_cam.F:7604
}
// MPAS module_ra_cam.F:7606
return;
}
// MPAS module_ra_cam.F:7609-7693
CAM_ROUTINE void radinp(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _pmid,const CamArray<double>& _pint,const CamArray<double>& _o3vmr,const CamArray<double>& _pmidrd,const CamArray<double>& _pintrd,double& eccf,const CamArray<double>& _o3mmr) {
CamFrame _frame(arena);
CAM_TRACE("radinp.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("radinp.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("radinp.o3vmr");
auto o3vmr=_o3vmr.rebound(1,pcols,1,pver);
CAM_TRACE("radinp.pmidrd");
auto pmidrd=_pmidrd.rebound(1,pcols,1,pver);
CAM_TRACE("radinp.pintrd");
auto pintrd=_pintrd.rebound(1,pcols,1,pverp);
CAM_TRACE("radinp.o3mmr");
auto o3mmr=_o3mmr.rebound(1,pcols,1,pver);
int i;
int k;
double calday;
double vmmr;
double delta;
// MPAS module_ra_cam.F:7666
eccf=1.f;
// MPAS module_ra_cam.F:7673
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:7674
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7675
pmidrd(i,k)=cam_mul(pmid(i,k),10.0f);
// MPAS module_ra_cam.F:7676
pintrd(i,k)=cam_mul(pint(i,k),10.0f);
// MPAS module_ra_cam.F:7677
}
// MPAS module_ra_cam.F:7678
}
// MPAS module_ra_cam.F:7679
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7680
pintrd(i,pverp)=cam_mul(pint(i,pverp),10.0f);
// MPAS module_ra_cam.F:7681
}
// MPAS module_ra_cam.F:7685
vmmr=cam_div(amo,amd);
// MPAS module_ra_cam.F:7686
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:7687
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7688
o3mmr(i,k)=cam_mul(vmmr,o3vmr(i,k));
// MPAS module_ra_cam.F:7689
}
// MPAS module_ra_cam.F:7690
}
// MPAS module_ra_cam.F:7692
return;
}
// MPAS module_ra_cam.F:7694-7754
CAM_ROUTINE void radoz2(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const int& pverp,const CamArray<double>& _o3vmr,const CamArray<double>& _pint,const CamArray<double>& _plol,const CamArray<double>& _plos,const int& ntoplw) {
CamFrame _frame(arena);
CAM_TRACE("radoz2.o3vmr");
auto o3vmr=_o3vmr.rebound(1,pcols,1,pver);
CAM_TRACE("radoz2.pint");
auto pint=_pint.rebound(1,pcols,1,pverp);
CAM_TRACE("radoz2.plol");
auto plol=_plol.rebound(1,pcols,1,pverp);
CAM_TRACE("radoz2.plos");
auto plos=_plos.rebound(1,pcols,1,pverp);
int i;
int k;
// MPAS module_ra_cam.F:7741
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7742
plos(i,ntoplw)=cam_mul(cam_mul(cam_mul(0.1f,cplos),o3vmr(i,ntoplw)),pint(i,ntoplw));
// MPAS module_ra_cam.F:7743
plol(i,ntoplw)=cam_mul(cam_mul(cam_mul(cam_mul(0.01f,cplol),o3vmr(i,ntoplw)),pint(i,ntoplw)),pint(i,ntoplw));
// MPAS module_ra_cam.F:7744
}
// MPAS module_ra_cam.F:7745
for(k=cam_add(ntoplw,1);k<=pverp;k+=1) {
// MPAS module_ra_cam.F:7746
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7747
plos(i,k)=cam_add(plos(i,cam_sub(k,1)),cam_mul(cam_mul(cam_mul(0.1f,cplos),o3vmr(i,cam_sub(k,1))),(cam_sub(pint(i,k),pint(i,cam_sub(k,1))))));
// MPAS module_ra_cam.F:7748
plol(i,k)=cam_add(plol(i,cam_sub(k,1)),cam_mul(cam_mul(cam_mul(0.01f,cplol),o3vmr(i,cam_sub(k,1))),(cam_sub(cam_mul(pint(i,k),pint(i,k)),cam_mul(pint(i,cam_sub(k,1)),pint(i,cam_sub(k,1)))))));
// MPAS module_ra_cam.F:7750
}
// MPAS module_ra_cam.F:7751
}
// MPAS module_ra_cam.F:7753
return;
}
// MPAS module_ra_cam.F:7757-7871
CAM_ROUTINE void radozn(const int& lchnk,const int& ncol,const int& pcols,const int& pver,const CamArray<double>& _pmid,const CamArray<double>& _pin,const int& levsiz,const CamArray<double>& _ozmix,const CamArray<double>& _o3vmr) {
CamFrame _frame(arena);
CAM_TRACE("radozn.pmid");
auto pmid=_pmid.rebound(1,pcols,1,pver);
CAM_TRACE("radozn.pin");
auto pin=_pin.rebound(1,levsiz);
CAM_TRACE("radozn.ozmix");
auto ozmix=_ozmix.rebound(1,pcols,1,levsiz);
CAM_TRACE("radozn.o3vmr");
auto o3vmr=_o3vmr.rebound(1,pcols,1,pver);
int i;
int k;
int kk;
int kkstart;
auto kupper=arena.array<int>(1,pcols);
int kount;
auto lats=arena.array<int>(1,pcols);
auto lons=arena.array<int>(1,pcols);
double dpu;
double dpl;
// MPAS module_ra_cam.F:7808
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7809
kupper(i)=1;
// MPAS module_ra_cam.F:7810
}
// MPAS module_ra_cam.F:7812
for(k=1;k<=pver;k+=1) {
// MPAS module_ra_cam.F:7817
kkstart=levsiz;
// MPAS module_ra_cam.F:7818
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7819
kkstart=cam_min0(kkstart,kupper(i));
// MPAS module_ra_cam.F:7820
}
// MPAS module_ra_cam.F:7821
kount=0;
// MPAS module_ra_cam.F:7825
for(kk=kkstart;kk<=cam_sub(levsiz,1);kk+=1) {
// MPAS module_ra_cam.F:7826
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7827
if(((pin(kk)<pmid(i,k))&&(pmid(i,k)<=pin(cam_add(kk,1))))) {
// MPAS module_ra_cam.F:7828
kupper(i)=kk;
// MPAS module_ra_cam.F:7829
kount=cam_add(kount,1);
// MPAS module_ra_cam.F:7830
}
// MPAS module_ra_cam.F:7831
}
// MPAS module_ra_cam.F:7836
if((kount==ncol)) {
// MPAS module_ra_cam.F:7837
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7838
dpu=cam_sub(pmid(i,k),pin(kupper(i)));
// MPAS module_ra_cam.F:7839
dpl=cam_sub(pin(cam_add(kupper(i),1)),pmid(i,k));
// MPAS module_ra_cam.F:7840
o3vmr(i,k)=cam_div((cam_add(cam_mul(ozmix(i,kupper(i)),dpl),cam_mul(ozmix(i,cam_add(kupper(i),1)),dpu))),(cam_add(dpl,dpu)));
// MPAS module_ra_cam.F:7842
}
// MPAS module_ra_cam.F:7843
goto label_35;
// MPAS module_ra_cam.F:7844
}
// MPAS module_ra_cam.F:7845
}
// MPAS module_ra_cam.F:7851
for(i=1;i<=ncol;i+=1) {
// MPAS module_ra_cam.F:7852
if((pmid(i,k)<pin(1))) {
// MPAS module_ra_cam.F:7853
o3vmr(i,k)=cam_div(cam_mul(ozmix(i,1),pmid(i,k)),pin(1));
// MPAS module_ra_cam.F:7854
} else if((pmid(i,k)>pin(levsiz))) {
// MPAS module_ra_cam.F:7855
o3vmr(i,k)=ozmix(i,levsiz);
// MPAS module_ra_cam.F:7856
} else {
// MPAS module_ra_cam.F:7857
dpu=cam_sub(pmid(i,k),pin(kupper(i)));
// MPAS module_ra_cam.F:7858
dpl=cam_sub(pin(cam_add(kupper(i),1)),pmid(i,k));
// MPAS module_ra_cam.F:7859
o3vmr(i,k)=cam_div((cam_add(cam_mul(ozmix(i,kupper(i)),dpl),cam_mul(ozmix(i,cam_add(kupper(i),1)),dpu))),(cam_add(dpl,dpu)));
// MPAS module_ra_cam.F:7861
}
// MPAS module_ra_cam.F:7862
}
// MPAS module_ra_cam.F:7864
if((kount>ncol)) {
// MPAS module_ra_cam.F:7866
}
// MPAS module_ra_cam.F:7867
label_35: ; 
// MPAS module_ra_cam.F:7868
}
// MPAS module_ra_cam.F:7870
return;
}
// MPAS mpas_atmphys_camrad_init.F:70-153
CAM_ROUTINE void camradinit(const float& cp,const float& r_d,const float& r_v,const float& gravity,const float& ep_2,const float& stbolt) {
CamFrame _frame(arena);
double pstd;
double rh2o;
double cpair;
// MPAS mpas_atmphys_camrad_init.F:110
ozncyc=true;
// MPAS mpas_atmphys_camrad_init.F:111
indirect=true;
// MPAS mpas_atmphys_camrad_init.F:112
ixcldliq=2;
// MPAS mpas_atmphys_camrad_init.F:113
ixcldice=3;
// MPAS mpas_atmphys_camrad_init.F:115
pstd=101325.0f;
// MPAS mpas_atmphys_camrad_init.F:118
mwdry=28.966f;
// MPAS mpas_atmphys_camrad_init.F:119
mwco2=44.f;
// MPAS mpas_atmphys_camrad_init.F:120
mwh2o=18.016f;
// MPAS mpas_atmphys_camrad_init.F:121
mwch4=16.f;
// MPAS mpas_atmphys_camrad_init.F:122
mwn2o=44.f;
// MPAS mpas_atmphys_camrad_init.F:123
mwf11=136.f;
// MPAS mpas_atmphys_camrad_init.F:124
mwf12=120.f;
// MPAS mpas_atmphys_camrad_init.F:125
cappa=cam_div(r_d,cp);
// MPAS mpas_atmphys_camrad_init.F:126
rair=r_d;
// MPAS mpas_atmphys_camrad_init.F:127
tmelt=273.16f;
// MPAS mpas_atmphys_camrad_init.F:128
r_universal=cam_mul(6.02214e26f,stbolt);
// MPAS mpas_atmphys_camrad_init.F:129
latvap=2.501e6f;
// MPAS mpas_atmphys_camrad_init.F:130
latice=3.336e5f;
// MPAS mpas_atmphys_camrad_init.F:131
zvir=cam_sub(cam_div(r_v,r_d),1.f);
// MPAS mpas_atmphys_camrad_init.F:132
rh2o=r_v;
// MPAS mpas_atmphys_camrad_init.F:133
cpair=cp;
// MPAS mpas_atmphys_camrad_init.F:135
epsqs=ep_2;
// MPAS mpas_atmphys_camrad_init.F:138
{
float temp0=gravity;
float temp1=cp;
float temp2=ep_2;
float temp3=stbolt;
double temp4=cam_mul(pstd,10.0f);
radini(temp0,temp1,temp2,temp3,temp4);if(error) return;}
// MPAS mpas_atmphys_camrad_init.F:142
esinti(epsqs,latvap,latice,rh2o,cpair,tmelt); if(error) return;
}
// MPAS mpas_atmphys_camrad_init.F:156-249
CAM_ROUTINE void radini(const float& gravx,const float& cpairx,const float& epsilox,const float& stebolx,const double& pstdx) {
CamFrame _frame(arena);
double v0;
double p0;
double amd;
double goz;
// MPAS mpas_atmphys_camrad_init.F:206
gravit=cam_mul(100.f,gravx);
// MPAS mpas_atmphys_camrad_init.F:207
rga=cam_div(1.f,gravit);
// MPAS mpas_atmphys_camrad_init.F:208
gravmks=gravx;
// MPAS mpas_atmphys_camrad_init.F:209
cpair=cam_mul(1.e4f,cpairx);
// MPAS mpas_atmphys_camrad_init.F:210
epsilo=epsilox;
// MPAS mpas_atmphys_camrad_init.F:211
sslp=1.013250e6f;
// MPAS mpas_atmphys_camrad_init.F:212
stebol=cam_mul(1.e3f,stebolx);
// MPAS mpas_atmphys_camrad_init.F:213
rgsslp=cam_div(0.5f,(cam_mul(gravit,sslp)));
// MPAS mpas_atmphys_camrad_init.F:214
dpfo3=2.5e-3f;
// MPAS mpas_atmphys_camrad_init.F:215
dpfco2=5.0e-3f;
// MPAS mpas_atmphys_camrad_init.F:216
dayspy=365.f;
// MPAS mpas_atmphys_camrad_init.F:217
pie=cam_mul(4.f,cam_atan(1.f));
// MPAS mpas_atmphys_camrad_init.F:221
v0=22.4136f;
// MPAS mpas_atmphys_camrad_init.F:222
p0=cam_mul(0.1f,sslp);
// MPAS mpas_atmphys_camrad_init.F:223
amd=28.9644f;
// MPAS mpas_atmphys_camrad_init.F:224
goz=gravx;
// MPAS mpas_atmphys_camrad_init.F:229
cplos=cam_mul(cam_div(v0,(cam_mul(amd,goz))),100.0f);
// MPAS mpas_atmphys_camrad_init.F:230
cplol=cam_mul(cam_mul(cam_div(v0,(cam_mul(cam_mul(amd,goz),p0))),0.5f),100.0f);
// MPAS mpas_atmphys_camrad_init.F:241
ntoplw=1;
// MPAS mpas_atmphys_camrad_init.F:247
{
double temp0=pstdx;
double temp1=mwdry;
double temp2=mwco2;
radaeini(temp0,temp1,temp2);if(error) return;}
}
// MPAS mpas_atmphys_camrad_init.F:252-390
CAM_ROUTINE void radaeini(const double& pstdx,const double& mwdryx,const double& mwco2x) {
CamFrame _frame(arena);
int t;
int tmin;
int tmax;
int itype;
double tdbl;
// MPAS mpas_atmphys_camrad_init.F:309
p0=pstdx;
// MPAS mpas_atmphys_camrad_init.F:310
amd=mwdryx;
// MPAS mpas_atmphys_camrad_init.F:311
amco2=mwco2x;
// MPAS mpas_atmphys_camrad_init.F:314
c16=cam_div(coefj(3,1),coefj(2,1));
// MPAS mpas_atmphys_camrad_init.F:315
c17=cam_div(coefk(3,1),coefk(2,1));
// MPAS mpas_atmphys_camrad_init.F:316
c26=cam_div(coefj(3,2),coefj(2,2));
// MPAS mpas_atmphys_camrad_init.F:317
c27=cam_div(coefk(3,2),coefk(2,2));
// MPAS mpas_atmphys_camrad_init.F:318
c28=.5f;
// MPAS mpas_atmphys_camrad_init.F:319
c29=.002053f;
// MPAS mpas_atmphys_camrad_init.F:320
c30=.1f;
// MPAS mpas_atmphys_camrad_init.F:321
c31=3.0e-5f;
// MPAS mpas_atmphys_camrad_init.F:328
fwcoef=.1f;
// MPAS mpas_atmphys_camrad_init.F:329
fwc1=.30f;
// MPAS mpas_atmphys_camrad_init.F:330
fwc2=4.5f;
// MPAS mpas_atmphys_camrad_init.F:331
fc1=2.6f;
// MPAS mpas_atmphys_camrad_init.F:381
tmin=cam_nint(min_tp_h2o);
// MPAS mpas_atmphys_camrad_init.F:382
tmax=cam_add(cam_nint(max_tp_h2o),1);
// MPAS mpas_atmphys_camrad_init.F:383
itype=0;
// MPAS mpas_atmphys_camrad_init.F:384
for(t=tmin;t<=tmax;t+=1) {
// MPAS mpas_atmphys_camrad_init.F:386
tdbl=t;
// MPAS mpas_atmphys_camrad_init.F:387
gffgch(tdbl,estblh2o(cam_sub(t,tmin)),itype); if(error) return;
// MPAS mpas_atmphys_camrad_init.F:388
}
}
// MPAS mpas_atmphys_camrad_init.F:393-554
CAM_ROUTINE void aerosol_init(const int& naerlevels,const int& ncells,const int& ncellssolve,const int& nvertlevels,CamArray<float> _m_hybi,CamArray<float> _m_psp,CamArray<float> _pressure_b,CamArray<float> _aerosolcn,CamArray<float> _aerosolcp,const int& idxsul_l,const int& idxsslt_l,const int& idxdustfirst_l,const int& idxocpho_l,const int& idxcarbonfirst_l,const int& idxbcpho_l,const int& idxocphi_l,const int& idxbcphi_l,const int& idxbg_l,const int& idxvolc_l,int& max_mxaerl) {
CamFrame _frame(arena);
auto m_hybi=_m_hybi;
auto m_psp=_m_psp;
auto pressure_b=_pressure_b;
auto aerosolcn=_aerosolcn;
auto aerosolcp=_aerosolcp;
auto mxaerl_local=arena.array<int>(1,ncellssolve);
int icell;
int k;
float psurf;
auto hybi=arena.array<float>(1,29);
{const float init[]={0,0.0065700002014637f,0.0138600002974272f,0.023089999333024f,0.0346900001168251f,0.0491999983787537f,0.0672300010919571f,0.0894500017166138f,0.116539999842644f,0.149159997701645f,0.187830001115799f,0.232859998941422f,0.284209996461868f,0.341369986534119f,0.403340011835098f,0.468600004911423f,0.535290002822876f,0.601350009441376f,0.66482001543045f,0.724009990692139f,0.777729988098145f,0.825269997119904f,0.866419970989227f,0.901350021362305f,0.930540025234222f,0.954590022563934f,0.974179983139038f,0.990000009536743f,1};hybi.init(init,sizeof(init)/sizeof(init[0]));}
// MPAS mpas_atmphys_camrad_init.F:456
for(k=1;k<=naerlevels;k+=1) {
// MPAS mpas_atmphys_camrad_init.F:457
for(icell=1;icell<=ncells;icell+=1) {
// MPAS mpas_atmphys_camrad_init.F:458
m_hybi(k,icell)=hybi(k);
// MPAS mpas_atmphys_camrad_init.F:459
}
// MPAS mpas_atmphys_camrad_init.F:460
}
// MPAS mpas_atmphys_camrad_init.F:462
psurf=1.e05f;
// MPAS mpas_atmphys_camrad_init.F:463
for(icell=1;icell<=ncells;icell+=1) {
// MPAS mpas_atmphys_camrad_init.F:464
m_psp(icell)=psurf;
// MPAS mpas_atmphys_camrad_init.F:467
}
// MPAS mpas_atmphys_camrad_init.F:472
{ auto _lhs=mxaerl_local.section(cam_section(1,ncellssolve)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=0; }
// MPAS mpas_atmphys_camrad_init.F:473
for(k=1;k<=nvertlevels;k+=1) {
// MPAS mpas_atmphys_camrad_init.F:474
for(icell=1;icell<=ncellssolve;icell+=1) {
// MPAS mpas_atmphys_camrad_init.F:475
if((pressure_b(k,icell)>=9.e04f)) {mxaerl_local(icell)=cam_add(mxaerl_local(icell),1);}
// MPAS mpas_atmphys_camrad_init.F:476
}
// MPAS mpas_atmphys_camrad_init.F:477
}
// MPAS mpas_atmphys_camrad_init.F:479
max_mxaerl=cam_maxval(mxaerl_local.section(cam_section(1,ncellssolve)));
// MPAS mpas_atmphys_camrad_init.F:486
ozncyc=true;
// MPAS mpas_atmphys_camrad_init.F:487
indirect=true;
// MPAS mpas_atmphys_camrad_init.F:488
ixcldliq=2;
// MPAS mpas_atmphys_camrad_init.F:489
ixcldice=3;
// MPAS mpas_atmphys_camrad_init.F:503
idxsul=idxsul_l;
// MPAS mpas_atmphys_camrad_init.F:504
idxsslt=idxsslt_l;
// MPAS mpas_atmphys_camrad_init.F:505
idxdustfirst=idxdustfirst_l;
// MPAS mpas_atmphys_camrad_init.F:506
idxocpho=idxocpho_l;
// MPAS mpas_atmphys_camrad_init.F:507
idxcarbonfirst=idxcarbonfirst_l;
// MPAS mpas_atmphys_camrad_init.F:508
idxbcpho=idxbcpho_l;
// MPAS mpas_atmphys_camrad_init.F:509
idxocphi=idxocphi_l;
// MPAS mpas_atmphys_camrad_init.F:510
idxbcphi=idxbcphi_l;
// MPAS mpas_atmphys_camrad_init.F:511
idxbg=idxbg_l;
// MPAS mpas_atmphys_camrad_init.F:512
idxvolc=idxvolc_l;
// MPAS mpas_atmphys_camrad_init.F:525
for(icell=1;icell<=ncells;icell+=1) {
// MPAS mpas_atmphys_camrad_init.F:526
for(k=1;k<=naerlevels;k+=1) {
// MPAS mpas_atmphys_camrad_init.F:529
aerosolcp(idxsul,k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:530
aerosolcn(idxsul,k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:531
aerosolcp(idxsslt,k,icell)=cam_mul(1.e-22f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:532
aerosolcn(idxsslt,k,icell)=cam_mul(1.e-22f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:533
aerosolcp(idxdustfirst,k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:534
aerosolcn(idxdustfirst,k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:535
aerosolcp(cam_add(idxdustfirst,1),k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:536
aerosolcn(cam_add(idxdustfirst,1),k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:537
aerosolcp(cam_add(idxdustfirst,2),k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:538
aerosolcn(cam_add(idxdustfirst,2),k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:539
aerosolcp(cam_add(idxdustfirst,3),k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:540
aerosolcn(cam_add(idxdustfirst,3),k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:541
aerosolcp(idxocpho,k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:542
aerosolcn(idxocpho,k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:543
aerosolcp(idxbcpho,k,icell)=cam_mul(1.e-9f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:544
aerosolcn(idxbcpho,k,icell)=cam_mul(1.e-9f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:545
aerosolcp(idxocphi,k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:546
aerosolcn(idxocphi,k,icell)=cam_mul(1.e-7f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:547
aerosolcp(idxbcphi,k,icell)=cam_mul(1.e-8f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:548
aerosolcn(idxbcphi,k,icell)=cam_mul(1.e-8f,(cam_sub(1.f,hybi(k))));
// MPAS mpas_atmphys_camrad_init.F:549
}
// MPAS mpas_atmphys_camrad_init.F:550
}
}
// MPAS mpas_atmphys_camrad_init.F:557-736
CAM_ROUTINE void aer_optics_initialize(CamArray<double> _rh_opac,CamArray<double> _ksul_opac,CamArray<double> _wsul_opac,CamArray<double> _gsul_opac,CamArray<double> _kssam_opac,CamArray<double> _wssam_opac,CamArray<double> _gssam_opac,CamArray<double> _ksscm_opac,CamArray<double> _wsscm_opac,CamArray<double> _gsscm_opac,CamArray<double> _kcphil_opac,CamArray<double> _wcphil_opac,CamArray<double> _gcphil_opac) {
CamFrame _frame(arena);
const double wgt_sscm=cam_div(6.0f,7.0f);
const int irh=8;
auto rh_opac=_rh_opac.rebound(1,irh);
auto ksul_opac=_ksul_opac.rebound(1,irh,1,nspint);
auto wsul_opac=_wsul_opac.rebound(1,irh,1,nspint);
auto gsul_opac=_gsul_opac.rebound(1,irh,1,nspint);
auto kssam_opac=_kssam_opac.rebound(1,irh,1,nspint);
auto wssam_opac=_wssam_opac.rebound(1,irh,1,nspint);
auto gssam_opac=_gssam_opac.rebound(1,irh,1,nspint);
auto ksscm_opac=_ksscm_opac.rebound(1,irh,1,nspint);
auto wsscm_opac=_wsscm_opac.rebound(1,irh,1,nspint);
auto gsscm_opac=_gsscm_opac.rebound(1,irh,1,nspint);
auto kcphil_opac=_kcphil_opac.rebound(1,irh,1,nspint);
auto wcphil_opac=_wcphil_opac.rebound(1,irh,1,nspint);
auto gcphil_opac=_gcphil_opac.rebound(1,irh,1,nspint);
int nbnd;
int krh;
int kbnd;
double rh;
auto ksslt_opac=arena.array<double>(1,irh,1,nspint);
auto wsslt_opac=arena.array<double>(1,irh,1,nspint);
auto gsslt_opac=arena.array<double>(1,irh,1,nspint);
int i;
auto _opac_ratio=arena.array<double>(1,irh);
// MPAS mpas_atmphys_camrad_init.F:689
{ auto _lhs=ksslt_opac.section(cam_section(1,irh),cam_section(1,nspint)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_add(cam_mul((cam_sub(1.0f,wgt_sscm)),kssam_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),cam_mul(wgt_sscm,ksscm_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice))); }
// MPAS mpas_atmphys_camrad_init.F:691
{ auto _lhs=wsslt_opac.section(cam_section(1,irh),cam_section(1,nspint)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div((cam_add(cam_mul(cam_mul((cam_sub(1.0f,wgt_sscm)),kssam_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),wssam_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),cam_mul(cam_mul(wgt_sscm,ksscm_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),wsscm_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)))),ksslt_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)); }
// MPAS mpas_atmphys_camrad_init.F:695
{ auto _lhs=gsslt_opac.section(cam_section(1,irh),cam_section(1,nspint)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div((cam_add(cam_mul(cam_mul(cam_mul((cam_sub(1.0f,wgt_sscm)),kssam_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),wssam_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),gssam_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),cam_mul(cam_mul(cam_mul(wgt_sscm,ksscm_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),wsscm_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)),gsscm_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)))),(cam_mul(ksslt_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice),wsslt_opac.section(cam_section(1,irh),cam_section(1,nspint)).linear(_slice)))); }
// MPAS mpas_atmphys_camrad_init.F:699
for(i=1;i<=nspint;i+=1) {
// MPAS mpas_atmphys_camrad_init.F:700
kcphob(i)=kcphil_opac(1,i);
// MPAS mpas_atmphys_camrad_init.F:701
wcphob(i)=wcphil_opac(1,i);
// MPAS mpas_atmphys_camrad_init.F:702
gcphob(i)=gcphil_opac(1,i);
// MPAS mpas_atmphys_camrad_init.F:703
}
// MPAS mpas_atmphys_camrad_init.F:708
nbnd=nspint;
// MPAS mpas_atmphys_camrad_init.F:710
for(krh=1;krh<=nrh;krh+=1) {
// MPAS mpas_atmphys_camrad_init.F:711
rh=cam_mul(cam_div(1.0,nrh),(cam_sub(krh,1)));
// MPAS mpas_atmphys_camrad_init.F:712
for(kbnd=1;kbnd<=nbnd;kbnd+=1) {
// MPAS mpas_atmphys_camrad_init.F:713
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(ksul_opac.section(cam_section(1,irh),kbnd).linear(_slice),ksul_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:713
ksul(krh,kbnd)=cam_mul(exp_interpol(rh_opac,_opac_ratio,rh),ksul_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:715
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(wsul_opac.section(cam_section(1,irh),kbnd).linear(_slice),wsul_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:715
wsul(krh,kbnd)=cam_mul(lin_interpol(rh_opac,_opac_ratio,rh),wsul_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:717
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(gsul_opac.section(cam_section(1,irh),kbnd).linear(_slice),gsul_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:717
gsul(krh,kbnd)=cam_mul(lin_interpol(rh_opac,_opac_ratio,rh),gsul_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:719
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(ksslt_opac.section(cam_section(1,irh),kbnd).linear(_slice),ksslt_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:719
ksslt(krh,kbnd)=cam_mul(exp_interpol(rh_opac,_opac_ratio,rh),ksslt_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:721
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(wsslt_opac.section(cam_section(1,irh),kbnd).linear(_slice),wsslt_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:721
wsslt(krh,kbnd)=cam_mul(lin_interpol(rh_opac,_opac_ratio,rh),wsslt_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:723
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(gsslt_opac.section(cam_section(1,irh),kbnd).linear(_slice),gsslt_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:723
gsslt(krh,kbnd)=cam_mul(lin_interpol(rh_opac,_opac_ratio,rh),gsslt_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:725
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(kcphil_opac.section(cam_section(1,irh),kbnd).linear(_slice),kcphil_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:725
kcphil(krh,kbnd)=cam_mul(exp_interpol(rh_opac,_opac_ratio,rh),kcphil_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:727
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(wcphil_opac.section(cam_section(1,irh),kbnd).linear(_slice),wcphil_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:727
wcphil(krh,kbnd)=cam_mul(lin_interpol(rh_opac,_opac_ratio,rh),wcphil_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:729
{ auto _lhs=_opac_ratio.section(cam_section(1,irh)); for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)=cam_div(gcphil_opac.section(cam_section(1,irh),kbnd).linear(_slice),gcphil_opac(1,kbnd)); }
// MPAS mpas_atmphys_camrad_init.F:729
gcphil(krh,kbnd)=cam_mul(lin_interpol(rh_opac,_opac_ratio,rh),gcphil_opac(1,kbnd));
// MPAS mpas_atmphys_camrad_init.F:731
}
// MPAS mpas_atmphys_camrad_init.F:732
}
}
// MPAS mpas_atmphys_camrad_init.F:739-874
CAM_ROUTINE void oznini(const int& ncells,const int& num_months,const int& levsiz,CamArray<float> _latcell,CamArray<float> _loncell,CamArray<float> _pin,CamArray<float> _ozmixm,CamArray<float> _lat_ozone,CamArray<float> _ozmixin,const float& degrad) {
CamFrame _frame(arena);
const int open_ok=0;
const int latsiz=64;
const int lonsiz=1;
auto latcell=_latcell;
auto loncell=_loncell;
auto pin=_pin;
auto ozmixm=_ozmixm;
auto lat_ozone=_lat_ozone.rebound(1,latsiz);
auto ozmixin=_ozmixin;
int i;
int i1;
int i2;
int k;
int j;
int m;
int icell;
float lat;
float lon;
float dlat;
float dlatcell;
// MPAS mpas_atmphys_camrad_init.F:790
for(k=1;k<=levsiz;k+=1) {
// MPAS mpas_atmphys_camrad_init.F:792
pin(k)=cam_mul(pin(k),100.f);
// MPAS mpas_atmphys_camrad_init.F:794
}
// MPAS mpas_atmphys_camrad_init.F:836
for(icell=1;icell<=ncells;icell+=1) {
// MPAS mpas_atmphys_camrad_init.F:837
lat=cam_div(latcell(icell),degrad);
// MPAS mpas_atmphys_camrad_init.F:838
lon=cam_div(loncell(icell),degrad);
// MPAS mpas_atmphys_camrad_init.F:839
if((lat>lat_ozone(latsiz))) {
// MPAS mpas_atmphys_camrad_init.F:840
i1=latsiz;
// MPAS mpas_atmphys_camrad_init.F:841
i2=latsiz;
// MPAS mpas_atmphys_camrad_init.F:842
} else if((lat<lat_ozone(1))) {
// MPAS mpas_atmphys_camrad_init.F:843
i1=1;
// MPAS mpas_atmphys_camrad_init.F:844
i2=1;
// MPAS mpas_atmphys_camrad_init.F:845
} else {
// MPAS mpas_atmphys_camrad_init.F:846
for(i=1;i<=latsiz;i+=1) {
// MPAS mpas_atmphys_camrad_init.F:847
if(((lat>=lat_ozone(i))&&(lat<lat_ozone(cam_add(i,1))))) {break;}
// MPAS mpas_atmphys_camrad_init.F:848
}
// MPAS mpas_atmphys_camrad_init.F:849
i1=i;
// MPAS mpas_atmphys_camrad_init.F:850
i2=cam_add(i,1);
// MPAS mpas_atmphys_camrad_init.F:851
}
// MPAS mpas_atmphys_camrad_init.F:853
for(m=1;m<=num_months;m+=1) {
// MPAS mpas_atmphys_camrad_init.F:854
for(k=1;k<=levsiz;k+=1) {
// MPAS mpas_atmphys_camrad_init.F:855
for(j=1;j<=lonsiz;j+=1) {
// MPAS mpas_atmphys_camrad_init.F:856
dlat=cam_sub(lat_ozone(i2),lat_ozone(i1));
// MPAS mpas_atmphys_camrad_init.F:857
dlatcell=cam_sub(lat,lat_ozone(i1));
// MPAS mpas_atmphys_camrad_init.F:858
if((dlat==0.f)) {
// MPAS mpas_atmphys_camrad_init.F:859
ozmixm(m,k,icell)=ozmixin(j,k,i1,m);
// MPAS mpas_atmphys_camrad_init.F:860
} else {
// MPAS mpas_atmphys_camrad_init.F:861
ozmixm(m,k,icell)=cam_add(ozmixin(j,k,i1,m),cam_div(cam_mul((cam_sub(ozmixin(j,k,i2,m),ozmixin(j,k,i1,m))),dlatcell),dlat));
// MPAS mpas_atmphys_camrad_init.F:863
}
// MPAS mpas_atmphys_camrad_init.F:864
}
// MPAS mpas_atmphys_camrad_init.F:865
}
// MPAS mpas_atmphys_camrad_init.F:866
}
// MPAS mpas_atmphys_camrad_init.F:871
}
}
// MPAS module_ra_cam_support.F:22-1265 module DATA/initializers
CAM_ROUTINE void initialize_defaults() {
ixcldliq={};
ixcldice={};
idxsul={};
idxsslt={};
idxdustfirst={};
idxcarbonfirst={};
idxocpho={};
idxbcpho={};
idxocphi={};
idxbcphi={};
idxbg={};
idxvolc={};
mxaerl={};
sulscl_rf=0.;
carscl_rf=0.;
ssltscl_rf=0.;
dustscl_rf=0.;
bgscl_rf=0.;
volcscl_rf=0.;
tauback=0.;
sulscl=1.;
carscl=1.;
ssltscl=1.;
dustscl=1.;
volcscl=1.;
{const double init[]={70.257384f,285.282943f,1.0273851e+02f,6.3073303e+01f,1.2039569e+02f,3.6343643e+02f,2.7138528e+02f};abs_cff_mss_aer.init(init,sizeof(init)/sizeof(init[0]));}
c16={};
c17={};
c26={};
c27={};
c28={};
c29={};
c30={};
c31={};
fwcoef={};
fwc1={};
fwc2={};
fc1={};
amco2={};
amd={};
p0={};
for(int i=0;i<ah2onw.count;++i) ah2onw.linear(i)={};
for(int i=0;i<eh2onw.count;++i) eh2onw.linear(i)={};
for(int i=0;i<ah2ow.count;++i) ah2ow.linear(i)={};
for(int i=0;i<cn_ah2ow.count;++i) cn_ah2ow.linear(i)={};
for(int i=0;i<cn_eh2ow.count;++i) cn_eh2ow.linear(i)={};
for(int i=0;i<ln_ah2ow.count;++i) ln_ah2ow.linear(i)={};
for(int i=0;i<ln_eh2ow.count;++i) ln_eh2ow.linear(i)={};
{const double init[]={5.46557e+01f,cam_neg(7.30387e-02f),1.09311e+02f,cam_neg(1.46077e-01f),5.11479e+01f,cam_neg(6.82615e-02f),1.02296e+02f,cam_neg(1.36523e-01f)};coefh.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={2.82096e-02f,2.47836e-04f,1.16904e-06f,9.27379e-02f,8.04454e-04f,6.88844e-06f};coefj.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={2.48852e-01f,2.09667e-03f,2.60377e-06f,1.03594e+00f,6.58620e-03f,4.04456e-06f};coefk.init(init,sizeof(init)/sizeof(init[0]));}
for(int i=0;i<estblh2o.count;++i) estblh2o.linear(i)={};
{const double init[]={cam_neg(1.06665373e-01f),2.90617375e-02f,cam_neg(2.70642049e-04f),1.07595511e-06f,cam_neg(1.97419681e-09f),1.37763374e-12f,1.10666537e+00f,cam_neg(2.90617375e-02f),2.70642049e-04f,cam_neg(1.07595511e-06f),1.97419681e-09f,cam_neg(1.37763374e-12f)};fat.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={3.46148163e-01f,1.51240299e-02f,cam_neg(1.21846479e-04f),4.04970123e-07f,cam_neg(6.15368936e-10f),3.52415071e-13f,6.53851837e-01f,cam_neg(1.51240299e-02f),1.21846479e-04f,cam_neg(4.04970123e-07f),6.15368936e-10f,cam_neg(3.52415071e-13f)};fet.init(init,sizeof(init)/sizeof(init[0]));}
gravit={};
rga={};
gravmks={};
cpair={};
epsilo={};
epsqs={};
sslp={};
stebol={};
rgsslp={};
dpfo3={};
dpfco2={};
dayspy={};
pie={};
mwdry={};
scon={};
co2mmr={};
mwco2={};
mwh2o={};
mwch4={};
mwn2o={};
mwf11={};
mwf12={};
cappa={};
rair={};
tmelt={};
r_universal={};
latvap={};
latice={};
zvir={};
for(int i=0;i<estbl.count;++i) estbl.linear(i)={};
tmin={};
tmax={};
for(int i=0;i<pcf.count;++i) pcf.linear(i)={};
for(int i=0;i<ksul.count;++i) ksul.linear(i)={};
for(int i=0;i<wsul.count;++i) wsul.linear(i)={};
for(int i=0;i<gsul.count;++i) gsul.linear(i)={};
for(int i=0;i<ksslt.count;++i) ksslt.linear(i)={};
for(int i=0;i<wsslt.count;++i) wsslt.linear(i)={};
for(int i=0;i<gsslt.count;++i) gsslt.linear(i)={};
for(int i=0;i<kcphil.count;++i) kcphil.linear(i)={};
for(int i=0;i<wcphil.count;++i) wcphil.linear(i)={};
for(int i=0;i<gcphil.count;++i) gcphil.linear(i)={};
for(int i=0;i<kbg.count;++i) kbg.linear(i)={};
for(int i=0;i<wbg.count;++i) wbg.linear(i)={};
for(int i=0;i<gbg.count;++i) gbg.linear(i)={};
for(int i=0;i<kcphob.count;++i) kcphob.linear(i)={};
for(int i=0;i<wcphob.count;++i) wcphob.linear(i)={};
for(int i=0;i<gcphob.count;++i) gcphob.linear(i)={};
for(int i=0;i<kcb.count;++i) kcb.linear(i)={};
for(int i=0;i<wcb.count;++i) wcb.linear(i)={};
for(int i=0;i<gcb.count;++i) gcb.linear(i)={};
for(int i=0;i<kvolc.count;++i) kvolc.linear(i)={};
for(int i=0;i<wvolc.count;++i) wvolc.linear(i)={};
for(int i=0;i<gvolc.count;++i) gvolc.linear(i)={};
for(int i=0;i<kdst.count;++i) kdst.linear(i)={};
for(int i=0;i<wdst.count;++i) wdst.linear(i)={};
for(int i=0;i<gdst.count;++i) gdst.linear(i)={};
cplos={};
cplol={};
co2vmr=3.550e-4f;
n2ovmr=0.311e-6f;
ch4vmr=1.714e-6f;
f11vmr=0.280e-9f;
f12vmr=0.503e-9f;
{const int init[]={1869,1870,1871,1872,1873,1874,1875,1876,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887,1888,1889,1890,1891,1892,1893,1894,1895,1896,1897,1898,1899,1900,1901,1902,1903,1904,1905,1906,1907,1908,1909,1910,1911,1912,1913,1914,1915,1916,1917,1918,1919,1920,1921,1922,1923,1924,1925,1926,1927,1928,1929,1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943,1944,1945,1946,1947,1948,1949,1950,1951,1952,1953,1954,1955,1956,1957,1958,1959,1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044,2045,2046,2047,2048,2049,2050,2051,2052,2053,2054,2055,2056,2057,2058,2059,2060,2061,2062,2063,2064,2065,2066,2067,2068,2069,2070,2071,2072,2073,2074,2075,2076,2077,2078,2079,2080,2081,2082,2083,2084,2085,2086,2087,2088,2089,2090,2091,2092,2093,2094,2095,2096,2097,2098,2099,2100,2101};yrdata.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={289.263f,289.263f,289.416f,289.577f,289.745f,289.919f,290.102f,290.293f,290.491f,290.696f,290.909f,291.129f,291.355f,291.587f,291.824f,292.066f,292.313f,292.563f,292.815f,293.071f,293.328f,293.586f,293.843f,294.098f,294.35f,294.598f,294.842f,295.082f,295.32f,295.558f,295.797f,296.038f,296.284f,296.535f,296.794f,297.062f,297.338f,297.62f,297.91f,298.204f,298.504f,298.806f,299.111f,299.419f,299.729f,300.04f,300.352f,300.666f,300.98f,301.294f,301.608f,301.923f,302.237f,302.551f,302.863f,303.172f,303.478f,303.779f,304.075f,304.366f,304.651f,304.93f,305.206f,305.478f,305.746f,306.013f,306.28f,306.546f,306.815f,307.087f,307.365f,307.65f,307.943f,308.246f,308.56f,308.887f,309.228f,309.584f,309.956f,310.344f,310.749f,311.172f,311.614f,312.077f,312.561f,313.068f,313.599f,314.154f,314.737f,315.347f,315.984f,316.646f,317.328f,318.026f,318.742f,319.489f,320.282f,321.133f,322.045f,323.021f,324.06f,325.155f,326.299f,327.484f,328.698f,329.933f,331.194f,332.499f,333.854f,335.254f,336.69f,338.15f,339.628f,341.125f,342.65f,344.206f,345.797f,347.397f,348.98f,350.551f,352.1f,354.3637f,355.7772f,357.1601f,358.5306f,359.9046f,361.4157f,363.0445f,364.7761f,366.6064f,368.5322f,370.534f,372.5798f,374.6564f,376.7656f,378.9087f,381.0864f,383.2994f,385.548f,387.8326f,390.1536f,392.523f,394.9625f,397.4806f,400.075f,402.7444f,405.4875f,408.3035f,411.1918f,414.1518f,417.1831f,420.2806f,423.4355f,426.6442f,429.9076f,433.2261f,436.6002f,440.0303f,443.5168f,447.06f,450.6603f,454.3059f,457.9756f,461.6612f,465.3649f,469.0886f,472.8335f,476.6008f,480.3916f,484.2069f,488.0473f,491.9184f,495.8295f,499.7849f,503.7843f,507.8278f,511.9155f,516.0476f,520.2243f,524.4459f,528.7127f,533.0213f,537.3655f,541.7429f,546.1544f,550.6005f,555.0819f,559.5991f,564.1525f,568.7429f,573.3701f,578.0399f,582.7611f,587.5379f,592.3701f,597.2572f,602.1997f,607.1975f,612.2507f,617.3596f,622.524f,627.7528f,633.0616f,638.457f,643.9384f,649.505f,655.1568f,660.8936f,666.7153f,672.6219f,678.6133f,684.6945f,690.8745f,697.1569f,703.5416f,710.0284f,716.6172f,723.308f,730.1008f,736.9958f,743.993f,751.0975f,758.3183f,765.6594f,773.1207f,780.702f,788.4033f,796.2249f,804.1667f,812.2289f,820.4118f,828.6444f,828.6444f};co2.init(init,sizeof(init)/sizeof(init[0]));}
ntoplw={};
masterproc=true;
ozncyc={};
indirect={};
radforce=false;
trace_gas=false;
strat_volcanic=false;
{const double init[]={5.92779f,6.26422f,6.61973f,6.99539f,7.39234f,7.81177f,8.25496f,8.72323f,9.21800f,9.74075f,10.2930f,10.8765f,11.4929f,12.1440f,12.8317f,13.5581f,14.2319f,15.0351f,15.8799f,16.7674f,17.6986f,18.6744f,19.6955f,20.7623f,21.8757f,23.0364f,24.2452f,25.5034f,26.8125f,27.7895f,28.6450f,29.4167f,30.1088f,30.7306f,31.2943f,31.8151f,32.3077f,32.7870f,33.2657f,33.7540f,34.2601f,34.7892f,35.3442f,35.9255f,36.5316f,37.1602f,37.8078f,38.4720f,39.1508f,39.8442f,40.5552f,41.2912f,42.0635f,42.8876f,43.7863f,44.7853f,45.9170f,47.2165f,48.7221f,50.4710f,52.4980f,54.8315f,57.4898f,60.4785f,63.7898f,65.5604f,71.2885f,75.4113f,79.7368f,84.2351f,88.8833f,93.6658f,98.5739f,103.603f,108.752f,114.025f,119.424f,124.954f,130.630f,136.457f,142.446f,148.608f,154.956f,161.503f,168.262f,175.248f,182.473f,189.952f,197.699f,205.728f,214.055f,222.694f,231.661f,240.971f,250.639f};retab.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={2.817e-02f,2.682e-02f,2.264e-02f,1.281e-02f};abarl.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={1.305f,1.346f,1.454f,1.641f};bbarl.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={cam_neg(5.62e-08f),cam_neg(6.94e-06f),4.64e-04f,0.201f};cbarl.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={1.63e-07f,2.35e-05f,1.24e-03f,7.56e-03f};dbarl.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={0.829f,0.794f,0.754f,0.826f};ebarl.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={2.482e-03f,4.226e-03f,6.560e-03f,4.353e-03f};fbarl.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={3.448e-03f,3.448e-03f,3.448e-03f,3.448e-03f};abari.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={2.431f,2.431f,2.431f,2.431f};bbari.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={1.00e-05f,1.10e-04f,1.861e-02f,.46658f};cbari.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={0.0f,1.405e-05f,8.328e-04f,2.05e-05f};dbari.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={0.7661f,0.7730f,0.794f,0.9595f};ebari.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={5.851e-04f,5.665e-04f,7.267e-04f,1.076e-04f};fbari.init(init,sizeof(init)/sizeof(init[0]));}
delta=0.0014257179260883f;
o2mmr=.23143f;
{const double init[]={.001488f,.001389f,.001290f,.001686f,.002877f,.003869f,.026336f,.360739f,.065392f,.526861f,.526861f,.526861f,.526861f,.526861f,.526861f,.526861f,.006239f,.001834f,.001834f};frcsol.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={.200f,.245f,.265f,.275f,.285f,.295f,.305f,.350f,.640f,.700f,.701f,.701f,.701f,.701f,.702f,.702f,2.630f,4.160f,4.160f};wavmin.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={.245f,.265f,.275f,.285f,.295f,.305f,.350f,.640f,.700f,5.000f,5.000f,5.000f,5.000f,5.000f,5.000f,5.000f,2.860f,4.550f,4.550f};wavmax.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={4.020f,2.180f,1.700f,1.450f,1.250f,1.085f,0.730f,v_raytau_35,v_raytau_64,0.02899756f,0.01356763f,0.00537341f,0.00228515f,0.00105028f,0.00046631f,0.00025734f,.0001f,.0001f,.0001f};raytau.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,0.00256608f,0.06310504f,0.42287445f,2.45397941f,11.20070807f,47.66091389f,240.19010243f,.000f,.000f,.000f};abh2o.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={5.370e+04f,13.080e+04f,9.292e+04f,4.530e+04f,1.616e+04f,4.441e+03f,1.775e+02f,v_abo3_35,v_abo3_64,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f};abo3.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.094f,.196f,1.963f};abco2.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,1.11e-05f,6.69e-05f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f};abo2.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.505f,.210f,.120f,.070f,.048f,.029f,.018f,.000f,.000f,.000f};ph2o.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,1.000f,.640f,.360f};pco2.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,1.000f,1.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f,.000f};po2.init(init,sizeof(init)/sizeof(init[0]));}
{const double init[]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.320518f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};nirwgt.init(init,sizeof(init)/sizeof(init[0]));}
amo=48.0000f;
}
};
// MPAS mpas_atmphys_driver_radiation_lw.F:297-610
CAM_ROUTINE void radiation_lw_from_mpas(CamArray<float> absnxt_p,CamArray<float> abstot_p,CamArray<float> aerosolcn_p,CamArray<float> aerosolcp_p,CamArray<float> aerosols,int cam_abs_dim1,int cam_abs_dim2,CamArray<float> cemiss_p,CamArray<float> cldfrac,CamArray<float> cldfrac_p,CamArray<float> coszr_p,float degrad,CamArray<float> emstot_p,CamArray<float> f_ice,CamArray<float> f_rain,CamArray<float> glw_p,CamArray<float> gsw_p,int ite,int its,int jte,int jts,int kte,int kts,CamArray<float> latcell,CamArray<float> loncell,CamArray<float> lwcf_p,CamArray<float> lwdnb_p,CamArray<float> lwdnbc_p,CamArray<float> lwdnt_p,CamArray<float> lwdntc_p,CamArray<float> lwupb_p,CamArray<float> lwupbc_p,CamArray<float> lwupt_p,CamArray<float> lwuptc_p,CamArray<float> m_hybi,CamArray<float> m_hybi_p,CamArray<float> m_ps,CamArray<float> m_psn_p,CamArray<float> m_psp_p,int num_aerlevels,int num_aerosols,int num_months,int num_oznlevels,CamArray<float> olrtoa_p,CamArray<float> ozmixm,CamArray<float> ozmixm_p,CamArray<float> pin,CamArray<float> pin_p,CamArray<float> rthratenlw_p,CamArray<float> rthratensw_p,CamArray<float> sfc_albedo,CamArray<float> sfc_albedo_p,CamArray<float> sfc_emiss,CamArray<float> sfc_emiss_p,CamArray<float> skintemp,CamArray<float> snow,CamArray<float> snow_p,CamArray<float> swcf_p,CamArray<float> swdnb_p,CamArray<float> swdnbc_p,CamArray<float> swdnt_p,CamArray<float> swdntc_p,CamArray<float> swupb_p,CamArray<float> swupbc_p,CamArray<float> swupt_p,CamArray<float> swuptc_p,CamArray<float> taucldc_p,CamArray<float> taucldi_p,CamArray<float> tsk_p,CamArray<float> xice,CamArray<float> xice_p,CamArray<float> xland,CamArray<float> xland_p,CamArray<float> xlat_p,CamArray<float> xlon_p,float xtime_s) {
int i;
int j;
int k;
int n;
// MPAS mpas_atmphys_driver_radiation_lw.F:364
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:365
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:366
sfc_emiss_p(i,j)=sfc_emiss(i);
// MPAS mpas_atmphys_driver_radiation_lw.F:367
tsk_p(i,j)=skintemp(i);
// MPAS mpas_atmphys_driver_radiation_lw.F:368
snow_p(i,j)=snow(i);
// MPAS mpas_atmphys_driver_radiation_lw.F:369
xice_p(i,j)=xice(i);
// MPAS mpas_atmphys_driver_radiation_lw.F:370
xland_p(i,j)=xland(i);
// MPAS mpas_atmphys_driver_radiation_lw.F:371
}
// MPAS mpas_atmphys_driver_radiation_lw.F:372
}
// MPAS mpas_atmphys_driver_radiation_lw.F:373
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:374
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:375
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:376
cldfrac_p(i,k,j)=cldfrac(k,i);
// MPAS mpas_atmphys_driver_radiation_lw.F:377
}
// MPAS mpas_atmphys_driver_radiation_lw.F:378
}
// MPAS mpas_atmphys_driver_radiation_lw.F:379
}
// MPAS mpas_atmphys_driver_radiation_lw.F:382
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:383
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:384
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:385
f_ice(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:386
f_rain(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:387
}
// MPAS mpas_atmphys_driver_radiation_lw.F:388
}
// MPAS mpas_atmphys_driver_radiation_lw.F:389
}
// MPAS mpas_atmphys_driver_radiation_lw.F:391
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:392
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:393
glw_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:394
lwcf_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:395
lwdnb_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:396
lwdnbc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:397
lwdnt_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:398
lwdntc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:399
lwupb_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:400
lwupbc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:401
lwupt_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:402
lwuptc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:403
olrtoa_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:404
}
// MPAS mpas_atmphys_driver_radiation_lw.F:406
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:407
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:408
rthratenlw_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:409
}
// MPAS mpas_atmphys_driver_radiation_lw.F:410
}
// MPAS mpas_atmphys_driver_radiation_lw.F:411
}
// MPAS mpas_atmphys_driver_radiation_lw.F:517
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:518
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:519
xlat_p(i,j)=cam_div(latcell(i),degrad);
// MPAS mpas_atmphys_driver_radiation_lw.F:520
xlon_p(i,j)=cam_div(loncell(i),degrad);
// MPAS mpas_atmphys_driver_radiation_lw.F:521
sfc_albedo_p(i,j)=sfc_albedo(i);
// MPAS mpas_atmphys_driver_radiation_lw.F:523
coszr_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:524
gsw_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:525
swcf_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:526
swdnb_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:527
swdnbc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:528
swdnt_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:529
swdntc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:530
swupb_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:531
swupbc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:532
swupt_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:533
swuptc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:534
}
// MPAS mpas_atmphys_driver_radiation_lw.F:535
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:536
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:537
rthratensw_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:538
cemiss_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:539
taucldc_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:540
taucldi_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:541
}
// MPAS mpas_atmphys_driver_radiation_lw.F:542
}
// MPAS mpas_atmphys_driver_radiation_lw.F:543
}
// MPAS mpas_atmphys_driver_radiation_lw.F:549
if((xtime_s<1.e-12f)) {
// MPAS mpas_atmphys_driver_radiation_lw.F:550
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:551
for(n=1;n<=cam_abs_dim1;n+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:552
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:553
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:554
absnxt_p(i,k,n,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:555
}
// MPAS mpas_atmphys_driver_radiation_lw.F:556
}
// MPAS mpas_atmphys_driver_radiation_lw.F:557
}
// MPAS mpas_atmphys_driver_radiation_lw.F:558
for(n=1;n<=cam_abs_dim2;n+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:559
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:560
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:561
abstot_p(i,k,n,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:562
}
// MPAS mpas_atmphys_driver_radiation_lw.F:563
}
// MPAS mpas_atmphys_driver_radiation_lw.F:564
}
// MPAS mpas_atmphys_driver_radiation_lw.F:565
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:566
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:567
emstot_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_lw.F:568
}
// MPAS mpas_atmphys_driver_radiation_lw.F:569
}
// MPAS mpas_atmphys_driver_radiation_lw.F:570
}
// MPAS mpas_atmphys_driver_radiation_lw.F:571
}
// MPAS mpas_atmphys_driver_radiation_lw.F:574
for(k=1;k<=num_oznlevels;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:575
pin_p(k)=pin(k);
// MPAS mpas_atmphys_driver_radiation_lw.F:576
}
// MPAS mpas_atmphys_driver_radiation_lw.F:577
for(n=1;n<=num_months;n+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:578
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:579
for(k=1;k<=num_oznlevels;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:580
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:581
ozmixm_p(i,k,j,n)=ozmixm(n,k,i);
// MPAS mpas_atmphys_driver_radiation_lw.F:582
}
// MPAS mpas_atmphys_driver_radiation_lw.F:583
}
// MPAS mpas_atmphys_driver_radiation_lw.F:584
}
// MPAS mpas_atmphys_driver_radiation_lw.F:585
}
// MPAS mpas_atmphys_driver_radiation_lw.F:587
for(k=1;k<=num_aerlevels;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:588
m_hybi_p(k)=m_hybi(k,1);
// MPAS mpas_atmphys_driver_radiation_lw.F:589
}
// MPAS mpas_atmphys_driver_radiation_lw.F:590
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:591
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:592
m_psp_p(i,j)=m_ps(i);
// MPAS mpas_atmphys_driver_radiation_lw.F:593
m_psn_p(i,j)=m_ps(i);
// MPAS mpas_atmphys_driver_radiation_lw.F:594
}
// MPAS mpas_atmphys_driver_radiation_lw.F:595
}
// MPAS mpas_atmphys_driver_radiation_lw.F:596
for(n=1;n<=num_aerosols;n+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:597
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:598
for(k=1;k<=num_aerlevels;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:599
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:600
aerosolcp_p(i,k,j,n)=aerosols(n,k,i);
// MPAS mpas_atmphys_driver_radiation_lw.F:601
aerosolcn_p(i,k,j,n)=aerosols(n,k,i);
// MPAS mpas_atmphys_driver_radiation_lw.F:602
}
// MPAS mpas_atmphys_driver_radiation_lw.F:603
}
// MPAS mpas_atmphys_driver_radiation_lw.F:604
}
// MPAS mpas_atmphys_driver_radiation_lw.F:605
}
}
// MPAS mpas_atmphys_driver_radiation_lw.F:613-718
CAM_ROUTINE void radiation_lw_to_mpas(CamArray<float> glw,CamArray<float> glw_p,int ite,int its,int jte,int jts,int kte,int kts,CamArray<float> lwcf,CamArray<float> lwcf_p,CamArray<float> lwdnb,CamArray<float> lwdnb_p,CamArray<float> lwdnbc,CamArray<float> lwdnbc_p,CamArray<float> lwdnt,CamArray<float> lwdnt_p,CamArray<float> lwdntc,CamArray<float> lwdntc_p,CamArray<float> lwupb,CamArray<float> lwupb_p,CamArray<float> lwupbc,CamArray<float> lwupbc_p,CamArray<float> lwupt,CamArray<float> lwupt_p,CamArray<float> lwuptc,CamArray<float> lwuptc_p,CamArray<float> olrtoa,CamArray<float> olrtoa_p,CamArray<float> rthratenlw,CamArray<float> rthratenlw_p) {
int i;
int j;
int k;
// MPAS mpas_atmphys_driver_radiation_lw.F:660
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:661
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:662
glw(i)=glw_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:663
lwcf(i)=lwcf_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:664
lwdnb(i)=lwdnb_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:665
lwdnbc(i)=lwdnbc_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:666
lwdnt(i)=lwdnt_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:667
lwdntc(i)=lwdntc_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:668
lwupb(i)=lwupb_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:669
lwupbc(i)=lwupbc_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:670
lwupt(i)=lwupt_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:671
lwuptc(i)=lwuptc_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:672
olrtoa(i)=olrtoa_p(i,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:673
}
// MPAS mpas_atmphys_driver_radiation_lw.F:675
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:676
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:677
rthratenlw(k,i)=rthratenlw_p(i,k,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:678
}
// MPAS mpas_atmphys_driver_radiation_lw.F:679
}
// MPAS mpas_atmphys_driver_radiation_lw.F:680
}
}
// MPAS mpas_atmphys_driver_radiation_lw.F:721-765
CAM_ROUTINE void radiation_camlw_to_mpas(CamArray<float> absnxt,CamArray<float> absnxt_p,CamArray<float> abstot,CamArray<float> abstot_p,int cam_abs_dim1,int cam_abs_dim2,CamArray<float> emstot,CamArray<float> emstot_p,int ite,int its,int jte,int jts,int kte,int kts) {
int i;
int j;
int k;
int n;
// MPAS mpas_atmphys_driver_radiation_lw.F:743
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:744
for(n=1;n<=cam_abs_dim1;n+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:745
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:746
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:747
absnxt(k,n,i)=absnxt_p(i,k,n,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:748
}
// MPAS mpas_atmphys_driver_radiation_lw.F:749
}
// MPAS mpas_atmphys_driver_radiation_lw.F:750
}
// MPAS mpas_atmphys_driver_radiation_lw.F:751
for(n=1;n<=cam_abs_dim2;n+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:752
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:753
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:754
abstot(k,n,i)=abstot_p(i,k,n,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:755
}
// MPAS mpas_atmphys_driver_radiation_lw.F:756
}
// MPAS mpas_atmphys_driver_radiation_lw.F:757
}
// MPAS mpas_atmphys_driver_radiation_lw.F:758
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:759
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_lw.F:760
emstot(k,i)=emstot_p(i,k,j);
// MPAS mpas_atmphys_driver_radiation_lw.F:761
}
// MPAS mpas_atmphys_driver_radiation_lw.F:762
}
// MPAS mpas_atmphys_driver_radiation_lw.F:763
}
}
// MPAS mpas_atmphys_driver_radiation_sw.F:355-711
CAM_ROUTINE void radiation_sw_from_mpas(CamArray<float> absnxt_p,CamArray<float> abstot_p,int& aer_opt,CamArray<float> aerosolcn_p,CamArray<float> aerosolcp_p,CamArray<float> aerosols,int cam_abs_dim1,int cam_abs_dim2,CamArray<float> cemiss_p,CamArray<float> cldfrac,CamArray<float> cldfrac_p,CamArray<float> coszr_p,float degrad,CamArray<float> emstot_p,CamArray<float> f_ice,CamArray<float> f_rain,CamArray<float> glw_p,CamArray<float> gsw_p,int ite,int its,int jte,int jts,int kte,int kts,CamArray<float> latcell,CamArray<float> loncell,CamArray<float> lwcf_p,CamArray<float> lwdnb_p,CamArray<float> lwdnbc_p,CamArray<float> lwdnt_p,CamArray<float> lwdntc_p,CamArray<float> lwupb_p,CamArray<float> lwupbc_p,CamArray<float> lwupt_p,CamArray<float> lwuptc_p,CamArray<float> m_hybi,CamArray<float> m_hybi_p,CamArray<float> m_ps,CamArray<float> m_psn_p,CamArray<float> m_psp_p,int num_aerlevels,int num_aerosols,int num_months,int num_oznlevels,CamArray<float> olrtoa_p,CamArray<float> ozmixm,CamArray<float> ozmixm_p,CamArray<float> pin,CamArray<float> pin_p,CamArray<float> rthratenlw_p,CamArray<float> rthratensw_p,CamArray<float> sfc_albedo,CamArray<float> sfc_albedo_p,CamArray<float> sfc_emiss,CamArray<float> sfc_emiss_p,CamArray<float> skintemp,CamArray<float> snow,CamArray<float> snow_p,CamArray<float> swcf_p,CamArray<float> swdnb_p,CamArray<float> swdnbc_p,CamArray<float> swdnt_p,CamArray<float> swdntc_p,CamArray<float> swupb_p,CamArray<float> swupbc_p,CamArray<float> swupt_p,CamArray<float> swuptc_p,CamArray<float> taucldc_p,CamArray<float> taucldi_p,CamArray<float> tsk_p,CamArray<float> xice,CamArray<float> xice_p,CamArray<float> xland,CamArray<float> xland_p,CamArray<float> xlat_p,CamArray<float> xlon_p,float xtime_s) {
int i;
int j;
int k;
int n;
// MPAS mpas_atmphys_driver_radiation_sw.F:419
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:420
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:421
xlat_p(i,j)=cam_div(latcell(i),degrad);
// MPAS mpas_atmphys_driver_radiation_sw.F:422
xlon_p(i,j)=cam_div(loncell(i),degrad);
// MPAS mpas_atmphys_driver_radiation_sw.F:424
sfc_albedo_p(i,j)=sfc_albedo(i);
// MPAS mpas_atmphys_driver_radiation_sw.F:425
snow_p(i,j)=snow(i);
// MPAS mpas_atmphys_driver_radiation_sw.F:426
tsk_p(i,j)=skintemp(i);
// MPAS mpas_atmphys_driver_radiation_sw.F:427
xice_p(i,j)=xice(i);
// MPAS mpas_atmphys_driver_radiation_sw.F:428
xland_p(i,j)=xland(i);
// MPAS mpas_atmphys_driver_radiation_sw.F:429
}
// MPAS mpas_atmphys_driver_radiation_sw.F:430
}
// MPAS mpas_atmphys_driver_radiation_sw.F:431
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:432
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:433
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:434
cldfrac_p(i,k,j)=cldfrac(k,i);
// MPAS mpas_atmphys_driver_radiation_sw.F:435
}
// MPAS mpas_atmphys_driver_radiation_sw.F:436
}
// MPAS mpas_atmphys_driver_radiation_sw.F:437
}
// MPAS mpas_atmphys_driver_radiation_sw.F:440
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:441
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:442
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:443
f_ice(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:444
f_rain(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:445
}
// MPAS mpas_atmphys_driver_radiation_sw.F:446
}
// MPAS mpas_atmphys_driver_radiation_sw.F:447
}
// MPAS mpas_atmphys_driver_radiation_sw.F:449
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:450
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:451
coszr_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:452
gsw_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:453
swcf_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:454
swdnb_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:455
swdnbc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:456
swdnt_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:457
swdntc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:458
swupb_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:459
swupbc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:460
swupt_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:461
swuptc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:462
}
// MPAS mpas_atmphys_driver_radiation_sw.F:464
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:465
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:466
rthratensw_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:467
}
// MPAS mpas_atmphys_driver_radiation_sw.F:468
}
// MPAS mpas_atmphys_driver_radiation_sw.F:469
}
// MPAS mpas_atmphys_driver_radiation_sw.F:471
aer_opt=0;
// MPAS mpas_atmphys_driver_radiation_sw.F:625
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:626
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:627
sfc_emiss_p(i,j)=sfc_emiss(i);
// MPAS mpas_atmphys_driver_radiation_sw.F:629
olrtoa_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:630
glw_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:631
lwcf_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:632
lwdnb_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:633
lwdnbc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:634
lwdnt_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:635
lwdntc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:636
lwupb_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:637
lwupbc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:638
lwupt_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:639
lwuptc_p(i,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:640
}
// MPAS mpas_atmphys_driver_radiation_sw.F:641
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:642
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:643
rthratenlw_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:644
cemiss_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:645
taucldc_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:646
taucldi_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:647
}
// MPAS mpas_atmphys_driver_radiation_sw.F:648
}
// MPAS mpas_atmphys_driver_radiation_sw.F:649
}
// MPAS mpas_atmphys_driver_radiation_sw.F:651
if((xtime_s<1.e-12f)) {
// MPAS mpas_atmphys_driver_radiation_sw.F:652
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:653
for(n=1;n<=cam_abs_dim1;n+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:654
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:655
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:656
absnxt_p(i,k,n,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:657
}
// MPAS mpas_atmphys_driver_radiation_sw.F:658
}
// MPAS mpas_atmphys_driver_radiation_sw.F:659
}
// MPAS mpas_atmphys_driver_radiation_sw.F:660
for(n=1;n<=cam_abs_dim2;n+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:661
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:662
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:663
abstot_p(i,k,n,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:664
}
// MPAS mpas_atmphys_driver_radiation_sw.F:665
}
// MPAS mpas_atmphys_driver_radiation_sw.F:666
}
// MPAS mpas_atmphys_driver_radiation_sw.F:667
for(k=kts;k<=cam_add(kte,1);k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:668
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:669
emstot_p(i,k,j)=0.0f;
// MPAS mpas_atmphys_driver_radiation_sw.F:670
}
// MPAS mpas_atmphys_driver_radiation_sw.F:671
}
// MPAS mpas_atmphys_driver_radiation_sw.F:672
}
// MPAS mpas_atmphys_driver_radiation_sw.F:673
}
// MPAS mpas_atmphys_driver_radiation_sw.F:675
for(k=1;k<=num_oznlevels;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:676
pin_p(k)=pin(k);
// MPAS mpas_atmphys_driver_radiation_sw.F:677
}
// MPAS mpas_atmphys_driver_radiation_sw.F:678
for(n=1;n<=num_months;n+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:679
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:680
for(k=1;k<=num_oznlevels;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:681
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:682
ozmixm_p(i,k,j,n)=ozmixm(n,k,i);
// MPAS mpas_atmphys_driver_radiation_sw.F:683
}
// MPAS mpas_atmphys_driver_radiation_sw.F:684
}
// MPAS mpas_atmphys_driver_radiation_sw.F:685
}
// MPAS mpas_atmphys_driver_radiation_sw.F:686
}
// MPAS mpas_atmphys_driver_radiation_sw.F:688
for(k=1;k<=num_aerlevels;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:689
m_hybi_p(k)=m_hybi(k,1);
// MPAS mpas_atmphys_driver_radiation_sw.F:690
}
// MPAS mpas_atmphys_driver_radiation_sw.F:691
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:692
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:693
m_psp_p(i,j)=m_ps(i);
// MPAS mpas_atmphys_driver_radiation_sw.F:694
m_psn_p(i,j)=m_ps(i);
// MPAS mpas_atmphys_driver_radiation_sw.F:695
}
// MPAS mpas_atmphys_driver_radiation_sw.F:696
}
// MPAS mpas_atmphys_driver_radiation_sw.F:697
for(n=1;n<=num_aerosols;n+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:698
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:699
for(k=1;k<=num_aerlevels;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:700
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:701
aerosolcp_p(i,k,j,n)=aerosols(n,k,i);
// MPAS mpas_atmphys_driver_radiation_sw.F:702
aerosolcn_p(i,k,j,n)=aerosols(n,k,i);
// MPAS mpas_atmphys_driver_radiation_sw.F:703
}
// MPAS mpas_atmphys_driver_radiation_sw.F:704
}
// MPAS mpas_atmphys_driver_radiation_sw.F:705
}
// MPAS mpas_atmphys_driver_radiation_sw.F:706
}
}
// MPAS mpas_atmphys_driver_radiation_sw.F:714-794
CAM_ROUTINE void radiation_sw_to_mpas(CamArray<float> coszr,CamArray<float> coszr_p,CamArray<float> gsw,CamArray<float> gsw_p,int ite,int its,int jte,int jts,int kte,int kts,CamArray<float> rthratensw,CamArray<float> rthratensw_p,CamArray<float> swcf,CamArray<float> swcf_p,CamArray<float> swdnb,CamArray<float> swdnb_p,CamArray<float> swdnbc,CamArray<float> swdnbc_p,CamArray<float> swdnt,CamArray<float> swdnt_p,CamArray<float> swdntc,CamArray<float> swdntc_p,CamArray<float> swupb,CamArray<float> swupb_p,CamArray<float> swupbc,CamArray<float> swupbc_p,CamArray<float> swupt,CamArray<float> swupt_p,CamArray<float> swuptc,CamArray<float> swuptc_p) {
int i;
int j;
int k;
// MPAS mpas_atmphys_driver_radiation_sw.F:757
for(j=jts;j<=jte;j+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:759
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:760
coszr(i)=coszr_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:761
gsw(i)=gsw_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:762
swcf(i)=swcf_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:763
swdnb(i)=swdnb_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:764
swdnbc(i)=swdnbc_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:765
swdnt(i)=swdnt_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:766
swdntc(i)=swdntc_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:767
swupb(i)=swupb_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:768
swupbc(i)=swupbc_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:769
swupt(i)=swupt_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:770
swuptc(i)=swuptc_p(i,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:771
}
// MPAS mpas_atmphys_driver_radiation_sw.F:783
for(k=kts;k<=kte;k+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:784
for(i=its;i<=ite;i+=1) {
// MPAS mpas_atmphys_driver_radiation_sw.F:785
rthratensw(k,i)=rthratensw_p(i,k,j);
// MPAS mpas_atmphys_driver_radiation_sw.F:786
}
// MPAS mpas_atmphys_driver_radiation_sw.F:787
}
// MPAS mpas_atmphys_driver_radiation_sw.F:789
}
}
// MPAS mpas_atmphys_driver_radiation_sw.F:1009-1050
CAM_ROUTINE void radconst(float& declin,float degrad,float dpd,float julian,float& solcon,float solcon_0) {
float arg;
float decdeg;
float djul;
float eccfac;
float obecl;
float rjul;
float sinob;
float sxlong;
// MPAS mpas_atmphys_driver_radiation_sw.F:1025
declin=0.f;
// MPAS mpas_atmphys_driver_radiation_sw.F:1026
solcon=0.f;
// MPAS mpas_atmphys_driver_radiation_sw.F:1030
obecl=cam_mul(23.5f,degrad);
// MPAS mpas_atmphys_driver_radiation_sw.F:1031
sinob=cam_sin(obecl);
// MPAS mpas_atmphys_driver_radiation_sw.F:1035
if((julian>=80.f)) {sxlong=cam_mul(dpd,(cam_sub(julian,80.f)));}
// MPAS mpas_atmphys_driver_radiation_sw.F:1036
if((julian<80.f)) {sxlong=cam_mul(dpd,(cam_add(julian,285.f)));}
// MPAS mpas_atmphys_driver_radiation_sw.F:1037
sxlong=cam_mul(sxlong,degrad);
// MPAS mpas_atmphys_driver_radiation_sw.F:1038
arg=cam_mul(sinob,cam_sin(sxlong));
// MPAS mpas_atmphys_driver_radiation_sw.F:1039
declin=cam_asin(arg);
// MPAS mpas_atmphys_driver_radiation_sw.F:1040
decdeg=cam_div(declin,degrad);
// MPAS mpas_atmphys_driver_radiation_sw.F:1044
djul=cam_div(cam_mul(julian,360.f),365.f);
// MPAS mpas_atmphys_driver_radiation_sw.F:1045
rjul=cam_mul(djul,degrad);
// MPAS mpas_atmphys_driver_radiation_sw.F:1046
eccfac=cam_add(cam_add(cam_add(cam_add(1.000110f,cam_mul(0.034221f,cam_cos(rjul))),cam_mul(0.001280f,cam_sin(rjul))),cam_mul(0.000719f,cam_cos(cam_mul(2,rjul)))),cam_mul(0.000077f,cam_sin(cam_mul(2,rjul))));
// MPAS mpas_atmphys_driver_radiation_sw.F:1048
solcon=cam_mul(solcon_0,eccfac);
}
// MPAS mpas_atmphys_driver_radiation_lw.F:888-902; mpas_atmphys_driver_radiation_sw.F:874-899
CAM_ROUTINE void driver_clock(float xtime_s,float dt_radt,float& xtime_m,float& radt) {
xtime_m=cam_div(xtime_s,60.f);radt=cam_div(dt_radt,60.f);
}

template<class T> __device__ CamArray<T> cam_bind(const unsigned long long* p) {
 CamArray<T> a;a.data=reinterpret_cast<T*>(p[0]);a.rank=int(p[1]);a.count=int(p[2]);
 for(int d=0;d<5;++d) {a.low[d]=int(p[3+d]);a.extent[d]=int(p[8+d]);a.stride[d]=int(p[13+d]);}
 return a;
}

extern "C" __global__ void cam_boundary(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error,int first_column,const unsigned long long* locals) {
const int column=int(blockIdx.x*blockDim.x+threadIdx.x);
const int active=min(int(args[1135])-int(args[1134])+1-first_column,int(gridDim.x*blockDim.x));
if(column>=active) return;
const unsigned long long slice=(bytes/active)&~7ULL;
CamArena arena(workspace+column*slice,slice);CamColumn cam(arena);cam.oracle_locals=locals;
cam.ixcldliq=int(state[0]);
cam.ixcldice=int(state[1]);
cam.idxsul=int(state[2]);
cam.idxsslt=int(state[3]);
cam.idxdustfirst=int(state[4]);
cam.idxcarbonfirst=int(state[5]);
cam.idxocpho=int(state[6]);
cam.idxbcpho=int(state[7]);
cam.idxocphi=int(state[8]);
cam.idxbcphi=int(state[9]);
cam.idxbg=int(state[10]);
cam.idxvolc=int(state[11]);
cam.mxaerl=int(state[12]);
cam.sulscl_rf=__longlong_as_double(state[13]);
cam.carscl_rf=__longlong_as_double(state[14]);
cam.ssltscl_rf=__longlong_as_double(state[15]);
cam.dustscl_rf=__longlong_as_double(state[16]);
cam.bgscl_rf=__longlong_as_double(state[17]);
cam.volcscl_rf=__longlong_as_double(state[18]);
cam.tauback=__longlong_as_double(state[19]);
cam.sulscl=__longlong_as_double(state[20]);
cam.carscl=__longlong_as_double(state[21]);
cam.ssltscl=__longlong_as_double(state[22]);
cam.dustscl=__longlong_as_double(state[23]);
cam.volcscl=__longlong_as_double(state[24]);
cam.abs_cff_mss_aer=cam_bind<double>(state+25);
cam.c16=__longlong_as_double(state[43]);
cam.c17=__longlong_as_double(state[44]);
cam.c26=__longlong_as_double(state[45]);
cam.c27=__longlong_as_double(state[46]);
cam.c28=__longlong_as_double(state[47]);
cam.c29=__longlong_as_double(state[48]);
cam.c30=__longlong_as_double(state[49]);
cam.c31=__longlong_as_double(state[50]);
cam.fwcoef=__longlong_as_double(state[51]);
cam.fwc1=__longlong_as_double(state[52]);
cam.fwc2=__longlong_as_double(state[53]);
cam.fc1=__longlong_as_double(state[54]);
cam.amco2=__longlong_as_double(state[55]);
cam.amd=__longlong_as_double(state[56]);
cam.p0=__longlong_as_double(state[57]);
cam.ah2onw=cam_bind<double>(state+58);
cam.eh2onw=cam_bind<double>(state+76);
cam.ah2ow=cam_bind<double>(state+94);
cam.cn_ah2ow=cam_bind<double>(state+112);
cam.cn_eh2ow=cam_bind<double>(state+130);
cam.ln_ah2ow=cam_bind<double>(state+148);
cam.ln_eh2ow=cam_bind<double>(state+166);
cam.coefh=cam_bind<double>(state+184);
cam.coefj=cam_bind<double>(state+202);
cam.coefk=cam_bind<double>(state+220);
cam.estblh2o=cam_bind<double>(state+238);
cam.fat=cam_bind<double>(state+256);
cam.fet=cam_bind<double>(state+274);
cam.gravit=__longlong_as_double(state[292]);
cam.rga=__longlong_as_double(state[293]);
cam.gravmks=__longlong_as_double(state[294]);
cam.cpair=__longlong_as_double(state[295]);
cam.epsilo=__longlong_as_double(state[296]);
cam.epsqs=__longlong_as_double(state[297]);
cam.sslp=__longlong_as_double(state[298]);
cam.stebol=__longlong_as_double(state[299]);
cam.rgsslp=__longlong_as_double(state[300]);
cam.dpfo3=__longlong_as_double(state[301]);
cam.dpfco2=__longlong_as_double(state[302]);
cam.dayspy=__longlong_as_double(state[303]);
cam.pie=__longlong_as_double(state[304]);
cam.mwdry=__longlong_as_double(state[305]);
cam.scon=__longlong_as_double(state[306]);
cam.co2mmr=__longlong_as_double(state[307]);
cam.mwco2=__longlong_as_double(state[308]);
cam.mwh2o=__longlong_as_double(state[309]);
cam.mwch4=__longlong_as_double(state[310]);
cam.mwn2o=__longlong_as_double(state[311]);
cam.mwf11=__longlong_as_double(state[312]);
cam.mwf12=__longlong_as_double(state[313]);
cam.cappa=__longlong_as_double(state[314]);
cam.rair=__longlong_as_double(state[315]);
cam.tmelt=__longlong_as_double(state[316]);
cam.r_universal=__longlong_as_double(state[317]);
cam.latvap=__longlong_as_double(state[318]);
cam.latice=__longlong_as_double(state[319]);
cam.zvir=__longlong_as_double(state[320]);
cam.estbl=cam_bind<double>(state+321);
cam.tmin=__longlong_as_double(state[339]);
cam.tmax=__longlong_as_double(state[340]);
cam.pcf=cam_bind<double>(state+341);
cam.ksul=cam_bind<double>(state+359);
cam.wsul=cam_bind<double>(state+377);
cam.gsul=cam_bind<double>(state+395);
cam.ksslt=cam_bind<double>(state+413);
cam.wsslt=cam_bind<double>(state+431);
cam.gsslt=cam_bind<double>(state+449);
cam.kcphil=cam_bind<double>(state+467);
cam.wcphil=cam_bind<double>(state+485);
cam.gcphil=cam_bind<double>(state+503);
cam.kbg=cam_bind<double>(state+521);
cam.wbg=cam_bind<double>(state+539);
cam.gbg=cam_bind<double>(state+557);
cam.kcphob=cam_bind<double>(state+575);
cam.wcphob=cam_bind<double>(state+593);
cam.gcphob=cam_bind<double>(state+611);
cam.kcb=cam_bind<double>(state+629);
cam.wcb=cam_bind<double>(state+647);
cam.gcb=cam_bind<double>(state+665);
cam.kvolc=cam_bind<double>(state+683);
cam.wvolc=cam_bind<double>(state+701);
cam.gvolc=cam_bind<double>(state+719);
cam.kdst=cam_bind<double>(state+737);
cam.wdst=cam_bind<double>(state+755);
cam.gdst=cam_bind<double>(state+773);
cam.cplos=__longlong_as_double(state[791]);
cam.cplol=__longlong_as_double(state[792]);
cam.co2vmr=__longlong_as_double(state[793]);
cam.n2ovmr=__longlong_as_double(state[794]);
cam.ch4vmr=__longlong_as_double(state[795]);
cam.f11vmr=__longlong_as_double(state[796]);
cam.f12vmr=__longlong_as_double(state[797]);
cam.yrdata=cam_bind<int>(state+798);
cam.co2=cam_bind<double>(state+816);
cam.ntoplw=int(state[834]);
cam.masterproc=(state[835]!=0);
cam.ozncyc=(state[836]!=0);
cam.indirect=(state[837]!=0);
cam.radforce=(state[838]!=0);
cam.trace_gas=(state[839]!=0);
cam.strat_volcanic=(state[840]!=0);
cam.retab=cam_bind<double>(state+841);
cam.abarl=cam_bind<double>(state+859);
cam.bbarl=cam_bind<double>(state+877);
cam.cbarl=cam_bind<double>(state+895);
cam.dbarl=cam_bind<double>(state+913);
cam.ebarl=cam_bind<double>(state+931);
cam.fbarl=cam_bind<double>(state+949);
cam.abari=cam_bind<double>(state+967);
cam.bbari=cam_bind<double>(state+985);
cam.cbari=cam_bind<double>(state+1003);
cam.dbari=cam_bind<double>(state+1021);
cam.ebari=cam_bind<double>(state+1039);
cam.fbari=cam_bind<double>(state+1057);
cam.delta=__longlong_as_double(state[1075]);
cam.o2mmr=__longlong_as_double(state[1076]);
cam.frcsol=cam_bind<double>(state+1077);
cam.wavmin=cam_bind<double>(state+1095);
cam.wavmax=cam_bind<double>(state+1113);
cam.raytau=cam_bind<double>(state+1131);
cam.abh2o=cam_bind<double>(state+1149);
cam.abo3=cam_bind<double>(state+1167);
cam.abco2=cam_bind<double>(state+1185);
cam.abo2=cam_bind<double>(state+1203);
cam.ph2o=cam_bind<double>(state+1221);
cam.pco2=cam_bind<double>(state+1239);
cam.po2=cam_bind<double>(state+1257);
cam.nirwgt=cam_bind<double>(state+1275);
cam.amo=__longlong_as_double(state[1293]);
auto rthratenlw=cam_bind<float>(args+0);
auto rthratensw=cam_bind<float>(args+18);
auto dolw=(args[36]!=0);
auto dosw=(args[37]!=0);
auto swupt=cam_bind<float>(args+38);
auto swuptc=cam_bind<float>(args+56);
auto swdnt=cam_bind<float>(args+74);
auto swdntc=cam_bind<float>(args+92);
auto lwupt=cam_bind<float>(args+110);
auto lwuptc=cam_bind<float>(args+128);
auto lwdnt=cam_bind<float>(args+146);
auto lwdntc=cam_bind<float>(args+164);
auto swupb=cam_bind<float>(args+182);
auto swupbc=cam_bind<float>(args+200);
auto swdnb=cam_bind<float>(args+218);
auto swdnbc=cam_bind<float>(args+236);
auto lwupb=cam_bind<float>(args+254);
auto lwupbc=cam_bind<float>(args+272);
auto lwdnb=cam_bind<float>(args+290);
auto lwdnbc=cam_bind<float>(args+308);
auto swcf=cam_bind<float>(args+326);
auto lwcf=cam_bind<float>(args+344);
auto olr=cam_bind<float>(args+362);
auto cemiss=cam_bind<float>(args+380);
auto taucldc=cam_bind<float>(args+398);
auto taucldi=cam_bind<float>(args+416);
auto coszr=cam_bind<float>(args+434);
auto gsw=cam_bind<float>(args+452);
auto glw=cam_bind<float>(args+470);
auto xlat=cam_bind<float>(args+488);
auto xlong=cam_bind<float>(args+506);
auto albedo=cam_bind<float>(args+524);
auto t_phy=cam_bind<float>(args+542);
auto tsk=cam_bind<float>(args+560);
auto emiss=cam_bind<float>(args+578);
auto qv3d=cam_bind<float>(args+596);
auto qc3d=cam_bind<float>(args+614);
auto qr3d=cam_bind<float>(args+632);
auto qi3d=cam_bind<float>(args+650);
auto qs3d=cam_bind<float>(args+668);
auto qg3d=cam_bind<float>(args+686);
auto f_qc=(args[704]!=0);
auto f_qr=(args[705]!=0);
auto f_qi=(args[706]!=0);
auto f_qs=(args[707]!=0);
auto f_ice_phy=cam_bind<float>(args+708);
auto f_rain_phy=cam_bind<float>(args+726);
auto p_phy=cam_bind<float>(args+744);
auto p8w=cam_bind<float>(args+762);
auto z=cam_bind<float>(args+780);
auto pi_phy=cam_bind<float>(args+798);
auto rho_phy=cam_bind<float>(args+816);
auto dz8w=cam_bind<float>(args+834);
auto cldfra=cam_bind<float>(args+852);
auto xland=cam_bind<float>(args+870);
auto xice=cam_bind<float>(args+888);
auto snow=cam_bind<float>(args+906);
auto ozmixm=cam_bind<float>(args+924);
auto pin0=cam_bind<float>(args+942);
auto levsiz=int(args[960]);
auto num_months=int(args[961]);
auto m_psp=cam_bind<float>(args+962);
auto m_psn=cam_bind<float>(args+980);
auto aerosolcp=cam_bind<float>(args+998);
auto aerosolcn=cam_bind<float>(args+1016);
auto m_hybi0=cam_bind<float>(args+1034);
auto cam_abs_dim1=int(args[1052]);
auto cam_abs_dim2=int(args[1053]);
auto paerlev=int(args[1054]);
auto naer_c=int(args[1055]);
auto gmt=__int_as_float(int(args[1056]));
auto julday=int(args[1057]);
auto julian=__int_as_float(int(args[1058]));
auto yr=int(args[1059]);
auto dt=__int_as_float(int(args[1060]));
auto xtime=__int_as_float(int(args[1061]));
auto declin=__int_as_float(int(args[1062]));
auto solcon=__int_as_float(int(args[1063]));
auto radt=__int_as_float(int(args[1064]));
auto degrad=__int_as_float(int(args[1065]));
auto n_cldadv=int(args[1066]);
auto abstot_3d=cam_bind<float>(args+1067);
auto absnxt_3d=cam_bind<float>(args+1085);
auto emstot_3d=cam_bind<float>(args+1103);
auto doabsems=(args[1121]!=0);
auto ids=int(args[1122]);
auto ide=int(args[1123]);
auto jds=int(args[1124]);
auto jde=int(args[1125]);
auto kds=int(args[1126]);
auto kde=int(args[1127]);
auto ims=int(args[1128]);
auto ime=int(args[1129]);
auto jms=int(args[1130]);
auto jme=int(args[1131]);
auto kms=int(args[1132]);
auto kme=int(args[1133]);
auto its=int(args[1134]);
auto ite=int(args[1135]);
auto jts=int(args[1136]);
auto jte=int(args[1137]);
auto kts=int(args[1138]);
auto kte=int(args[1139]);
// MPAS module_ra_cam.F:476: the original chunk first-column cache controls every column.
cam.native_chunk_its=its;its+=first_column+column;ite=its;
cam.camrad(rthratenlw,rthratensw,dolw,dosw,swupt,swuptc,swdnt,swdntc,lwupt,lwuptc,lwdnt,lwdntc,swupb,swupbc,swdnb,swdnbc,lwupb,lwupbc,lwdnb,lwdnbc,swcf,lwcf,olr,cemiss,taucldc,taucldi,coszr,gsw,glw,xlat,xlong,albedo,t_phy,tsk,emiss,qv3d,qc3d,qr3d,qi3d,qs3d,qg3d,f_qc,f_qr,f_qi,f_qs,f_ice_phy,f_rain_phy,p_phy,p8w,z,pi_phy,rho_phy,dz8w,cldfra,xland,xice,snow,ozmixm,pin0,levsiz,num_months,m_psp,m_psn,aerosolcp,aerosolcn,m_hybi0,cam_abs_dim1,cam_abs_dim2,paerlev,naer_c,gmt,julday,julian,yr,dt,xtime,declin,solcon,radt,degrad,n_cldadv,abstot_3d,absnxt_3d,emstot_3d,doabsems,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte);
// MPAS module_ra_cam.F:458-460: persist the module CO2 update.
if(first_column==0 && column==0) {
state[793]=__double_as_longlong(cam.co2vmr);
state[307]=__double_as_longlong(cam.co2mmr);
}if(cam.error) atomicCAS(error,0,cam.error);}
// MPAS module_ra_cam.F:476-478; src/core_atmosphere/physics/mpas_atmphys_utilities.F:18.
// GNU list-directed internal CHARACTER write adds one leading blank and pads.
extern "C" __global__ void cam_restart_message(unsigned long long* args,unsigned char* message) {
if(blockIdx.x || threadIdx.x) return;
auto abstot_3d=cam_bind<float>(args+1067);
auto its=int(args[1134]),kts=int(args[1138]),jts=int(args[1136]);
if(abstot_3d(its,kts,kts,jts)==0.f && args[1121]==0 && args[36]!=0) {
 const char text[]="    camrad lw: CAUTION: re-calculating abstot,absnxt, on restart";
 for(int n=0;n<512;++n) message[n]=n<int(sizeof(text)-1)?text[n]:' ';
}
}
extern "C" __global__ void cam_prepare_cache(unsigned long long* args) {
if(blockIdx.x || threadIdx.x) return;
auto abstot_3d=cam_bind<float>(args+1067);
auto its=int(args[1134]);
auto kts=int(args[1138]);
auto jts=int(args[1136]);
auto dolw=(args[36]!=0);
auto doabsems=(args[1121]!=0);
// MPAS module_ra_cam.F:476-478. Resolve before any column updates its cache.
if(abstot_3d(its,kts,kts,jts)==0.f && !doabsems && dolw) doabsems=true;
args[1121]=doabsems;}
extern "C" __global__ void cam_camradinit(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
cam.ixcldliq=int(state[0]);
cam.ixcldice=int(state[1]);
cam.idxsul=int(state[2]);
cam.idxsslt=int(state[3]);
cam.idxdustfirst=int(state[4]);
cam.idxcarbonfirst=int(state[5]);
cam.idxocpho=int(state[6]);
cam.idxbcpho=int(state[7]);
cam.idxocphi=int(state[8]);
cam.idxbcphi=int(state[9]);
cam.idxbg=int(state[10]);
cam.idxvolc=int(state[11]);
cam.mxaerl=int(state[12]);
cam.sulscl_rf=__longlong_as_double(state[13]);
cam.carscl_rf=__longlong_as_double(state[14]);
cam.ssltscl_rf=__longlong_as_double(state[15]);
cam.dustscl_rf=__longlong_as_double(state[16]);
cam.bgscl_rf=__longlong_as_double(state[17]);
cam.volcscl_rf=__longlong_as_double(state[18]);
cam.tauback=__longlong_as_double(state[19]);
cam.sulscl=__longlong_as_double(state[20]);
cam.carscl=__longlong_as_double(state[21]);
cam.ssltscl=__longlong_as_double(state[22]);
cam.dustscl=__longlong_as_double(state[23]);
cam.volcscl=__longlong_as_double(state[24]);
cam.abs_cff_mss_aer=cam_bind<double>(state+25);
cam.c16=__longlong_as_double(state[43]);
cam.c17=__longlong_as_double(state[44]);
cam.c26=__longlong_as_double(state[45]);
cam.c27=__longlong_as_double(state[46]);
cam.c28=__longlong_as_double(state[47]);
cam.c29=__longlong_as_double(state[48]);
cam.c30=__longlong_as_double(state[49]);
cam.c31=__longlong_as_double(state[50]);
cam.fwcoef=__longlong_as_double(state[51]);
cam.fwc1=__longlong_as_double(state[52]);
cam.fwc2=__longlong_as_double(state[53]);
cam.fc1=__longlong_as_double(state[54]);
cam.amco2=__longlong_as_double(state[55]);
cam.amd=__longlong_as_double(state[56]);
cam.p0=__longlong_as_double(state[57]);
cam.ah2onw=cam_bind<double>(state+58);
cam.eh2onw=cam_bind<double>(state+76);
cam.ah2ow=cam_bind<double>(state+94);
cam.cn_ah2ow=cam_bind<double>(state+112);
cam.cn_eh2ow=cam_bind<double>(state+130);
cam.ln_ah2ow=cam_bind<double>(state+148);
cam.ln_eh2ow=cam_bind<double>(state+166);
cam.coefh=cam_bind<double>(state+184);
cam.coefj=cam_bind<double>(state+202);
cam.coefk=cam_bind<double>(state+220);
cam.estblh2o=cam_bind<double>(state+238);
cam.fat=cam_bind<double>(state+256);
cam.fet=cam_bind<double>(state+274);
cam.gravit=__longlong_as_double(state[292]);
cam.rga=__longlong_as_double(state[293]);
cam.gravmks=__longlong_as_double(state[294]);
cam.cpair=__longlong_as_double(state[295]);
cam.epsilo=__longlong_as_double(state[296]);
cam.epsqs=__longlong_as_double(state[297]);
cam.sslp=__longlong_as_double(state[298]);
cam.stebol=__longlong_as_double(state[299]);
cam.rgsslp=__longlong_as_double(state[300]);
cam.dpfo3=__longlong_as_double(state[301]);
cam.dpfco2=__longlong_as_double(state[302]);
cam.dayspy=__longlong_as_double(state[303]);
cam.pie=__longlong_as_double(state[304]);
cam.mwdry=__longlong_as_double(state[305]);
cam.scon=__longlong_as_double(state[306]);
cam.co2mmr=__longlong_as_double(state[307]);
cam.mwco2=__longlong_as_double(state[308]);
cam.mwh2o=__longlong_as_double(state[309]);
cam.mwch4=__longlong_as_double(state[310]);
cam.mwn2o=__longlong_as_double(state[311]);
cam.mwf11=__longlong_as_double(state[312]);
cam.mwf12=__longlong_as_double(state[313]);
cam.cappa=__longlong_as_double(state[314]);
cam.rair=__longlong_as_double(state[315]);
cam.tmelt=__longlong_as_double(state[316]);
cam.r_universal=__longlong_as_double(state[317]);
cam.latvap=__longlong_as_double(state[318]);
cam.latice=__longlong_as_double(state[319]);
cam.zvir=__longlong_as_double(state[320]);
cam.estbl=cam_bind<double>(state+321);
cam.tmin=__longlong_as_double(state[339]);
cam.tmax=__longlong_as_double(state[340]);
cam.pcf=cam_bind<double>(state+341);
cam.ksul=cam_bind<double>(state+359);
cam.wsul=cam_bind<double>(state+377);
cam.gsul=cam_bind<double>(state+395);
cam.ksslt=cam_bind<double>(state+413);
cam.wsslt=cam_bind<double>(state+431);
cam.gsslt=cam_bind<double>(state+449);
cam.kcphil=cam_bind<double>(state+467);
cam.wcphil=cam_bind<double>(state+485);
cam.gcphil=cam_bind<double>(state+503);
cam.kbg=cam_bind<double>(state+521);
cam.wbg=cam_bind<double>(state+539);
cam.gbg=cam_bind<double>(state+557);
cam.kcphob=cam_bind<double>(state+575);
cam.wcphob=cam_bind<double>(state+593);
cam.gcphob=cam_bind<double>(state+611);
cam.kcb=cam_bind<double>(state+629);
cam.wcb=cam_bind<double>(state+647);
cam.gcb=cam_bind<double>(state+665);
cam.kvolc=cam_bind<double>(state+683);
cam.wvolc=cam_bind<double>(state+701);
cam.gvolc=cam_bind<double>(state+719);
cam.kdst=cam_bind<double>(state+737);
cam.wdst=cam_bind<double>(state+755);
cam.gdst=cam_bind<double>(state+773);
cam.cplos=__longlong_as_double(state[791]);
cam.cplol=__longlong_as_double(state[792]);
cam.co2vmr=__longlong_as_double(state[793]);
cam.n2ovmr=__longlong_as_double(state[794]);
cam.ch4vmr=__longlong_as_double(state[795]);
cam.f11vmr=__longlong_as_double(state[796]);
cam.f12vmr=__longlong_as_double(state[797]);
cam.yrdata=cam_bind<int>(state+798);
cam.co2=cam_bind<double>(state+816);
cam.ntoplw=int(state[834]);
cam.masterproc=(state[835]!=0);
cam.ozncyc=(state[836]!=0);
cam.indirect=(state[837]!=0);
cam.radforce=(state[838]!=0);
cam.trace_gas=(state[839]!=0);
cam.strat_volcanic=(state[840]!=0);
cam.retab=cam_bind<double>(state+841);
cam.abarl=cam_bind<double>(state+859);
cam.bbarl=cam_bind<double>(state+877);
cam.cbarl=cam_bind<double>(state+895);
cam.dbarl=cam_bind<double>(state+913);
cam.ebarl=cam_bind<double>(state+931);
cam.fbarl=cam_bind<double>(state+949);
cam.abari=cam_bind<double>(state+967);
cam.bbari=cam_bind<double>(state+985);
cam.cbari=cam_bind<double>(state+1003);
cam.dbari=cam_bind<double>(state+1021);
cam.ebari=cam_bind<double>(state+1039);
cam.fbari=cam_bind<double>(state+1057);
cam.delta=__longlong_as_double(state[1075]);
cam.o2mmr=__longlong_as_double(state[1076]);
cam.frcsol=cam_bind<double>(state+1077);
cam.wavmin=cam_bind<double>(state+1095);
cam.wavmax=cam_bind<double>(state+1113);
cam.raytau=cam_bind<double>(state+1131);
cam.abh2o=cam_bind<double>(state+1149);
cam.abo3=cam_bind<double>(state+1167);
cam.abco2=cam_bind<double>(state+1185);
cam.abo2=cam_bind<double>(state+1203);
cam.ph2o=cam_bind<double>(state+1221);
cam.pco2=cam_bind<double>(state+1239);
cam.po2=cam_bind<double>(state+1257);
cam.nirwgt=cam_bind<double>(state+1275);
cam.amo=__longlong_as_double(state[1293]);
auto cp=__int_as_float(int(args[0]));
auto r_d=__int_as_float(int(args[1]));
auto r_v=__int_as_float(int(args[2]));
auto gravity=__int_as_float(int(args[3]));
auto ep_2=__int_as_float(int(args[4]));
auto stbolt=__int_as_float(int(args[5]));
cam.camradinit(cp,r_d,r_v,gravity,ep_2,stbolt);
state[0]=cam.ixcldliq;
state[1]=cam.ixcldice;
state[2]=cam.idxsul;
state[3]=cam.idxsslt;
state[4]=cam.idxdustfirst;
state[5]=cam.idxcarbonfirst;
state[6]=cam.idxocpho;
state[7]=cam.idxbcpho;
state[8]=cam.idxocphi;
state[9]=cam.idxbcphi;
state[10]=cam.idxbg;
state[11]=cam.idxvolc;
state[12]=cam.mxaerl;
state[13]=__double_as_longlong(cam.sulscl_rf);
state[14]=__double_as_longlong(cam.carscl_rf);
state[15]=__double_as_longlong(cam.ssltscl_rf);
state[16]=__double_as_longlong(cam.dustscl_rf);
state[17]=__double_as_longlong(cam.bgscl_rf);
state[18]=__double_as_longlong(cam.volcscl_rf);
state[19]=__double_as_longlong(cam.tauback);
state[20]=__double_as_longlong(cam.sulscl);
state[21]=__double_as_longlong(cam.carscl);
state[22]=__double_as_longlong(cam.ssltscl);
state[23]=__double_as_longlong(cam.dustscl);
state[24]=__double_as_longlong(cam.volcscl);
state[43]=__double_as_longlong(cam.c16);
state[44]=__double_as_longlong(cam.c17);
state[45]=__double_as_longlong(cam.c26);
state[46]=__double_as_longlong(cam.c27);
state[47]=__double_as_longlong(cam.c28);
state[48]=__double_as_longlong(cam.c29);
state[49]=__double_as_longlong(cam.c30);
state[50]=__double_as_longlong(cam.c31);
state[51]=__double_as_longlong(cam.fwcoef);
state[52]=__double_as_longlong(cam.fwc1);
state[53]=__double_as_longlong(cam.fwc2);
state[54]=__double_as_longlong(cam.fc1);
state[55]=__double_as_longlong(cam.amco2);
state[56]=__double_as_longlong(cam.amd);
state[57]=__double_as_longlong(cam.p0);
state[292]=__double_as_longlong(cam.gravit);
state[293]=__double_as_longlong(cam.rga);
state[294]=__double_as_longlong(cam.gravmks);
state[295]=__double_as_longlong(cam.cpair);
state[296]=__double_as_longlong(cam.epsilo);
state[297]=__double_as_longlong(cam.epsqs);
state[298]=__double_as_longlong(cam.sslp);
state[299]=__double_as_longlong(cam.stebol);
state[300]=__double_as_longlong(cam.rgsslp);
state[301]=__double_as_longlong(cam.dpfo3);
state[302]=__double_as_longlong(cam.dpfco2);
state[303]=__double_as_longlong(cam.dayspy);
state[304]=__double_as_longlong(cam.pie);
state[305]=__double_as_longlong(cam.mwdry);
state[306]=__double_as_longlong(cam.scon);
state[307]=__double_as_longlong(cam.co2mmr);
state[308]=__double_as_longlong(cam.mwco2);
state[309]=__double_as_longlong(cam.mwh2o);
state[310]=__double_as_longlong(cam.mwch4);
state[311]=__double_as_longlong(cam.mwn2o);
state[312]=__double_as_longlong(cam.mwf11);
state[313]=__double_as_longlong(cam.mwf12);
state[314]=__double_as_longlong(cam.cappa);
state[315]=__double_as_longlong(cam.rair);
state[316]=__double_as_longlong(cam.tmelt);
state[317]=__double_as_longlong(cam.r_universal);
state[318]=__double_as_longlong(cam.latvap);
state[319]=__double_as_longlong(cam.latice);
state[320]=__double_as_longlong(cam.zvir);
state[339]=__double_as_longlong(cam.tmin);
state[340]=__double_as_longlong(cam.tmax);
state[791]=__double_as_longlong(cam.cplos);
state[792]=__double_as_longlong(cam.cplol);
state[793]=__double_as_longlong(cam.co2vmr);
state[794]=__double_as_longlong(cam.n2ovmr);
state[795]=__double_as_longlong(cam.ch4vmr);
state[796]=__double_as_longlong(cam.f11vmr);
state[797]=__double_as_longlong(cam.f12vmr);
state[834]=cam.ntoplw;
state[835]=cam.masterproc;
state[836]=cam.ozncyc;
state[837]=cam.indirect;
state[838]=cam.radforce;
state[839]=cam.trace_gas;
state[840]=cam.strat_volcanic;
state[1075]=__double_as_longlong(cam.delta);
state[1076]=__double_as_longlong(cam.o2mmr);
state[1293]=__double_as_longlong(cam.amo);
args[0]=__float_as_uint(cp);
args[1]=__float_as_uint(r_d);
args[2]=__float_as_uint(r_v);
args[3]=__float_as_uint(gravity);
args[4]=__float_as_uint(ep_2);
args[5]=__float_as_uint(stbolt);
*error=cam.error;}
extern "C" __global__ void cam_radini(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
cam.ixcldliq=int(state[0]);
cam.ixcldice=int(state[1]);
cam.idxsul=int(state[2]);
cam.idxsslt=int(state[3]);
cam.idxdustfirst=int(state[4]);
cam.idxcarbonfirst=int(state[5]);
cam.idxocpho=int(state[6]);
cam.idxbcpho=int(state[7]);
cam.idxocphi=int(state[8]);
cam.idxbcphi=int(state[9]);
cam.idxbg=int(state[10]);
cam.idxvolc=int(state[11]);
cam.mxaerl=int(state[12]);
cam.sulscl_rf=__longlong_as_double(state[13]);
cam.carscl_rf=__longlong_as_double(state[14]);
cam.ssltscl_rf=__longlong_as_double(state[15]);
cam.dustscl_rf=__longlong_as_double(state[16]);
cam.bgscl_rf=__longlong_as_double(state[17]);
cam.volcscl_rf=__longlong_as_double(state[18]);
cam.tauback=__longlong_as_double(state[19]);
cam.sulscl=__longlong_as_double(state[20]);
cam.carscl=__longlong_as_double(state[21]);
cam.ssltscl=__longlong_as_double(state[22]);
cam.dustscl=__longlong_as_double(state[23]);
cam.volcscl=__longlong_as_double(state[24]);
cam.abs_cff_mss_aer=cam_bind<double>(state+25);
cam.c16=__longlong_as_double(state[43]);
cam.c17=__longlong_as_double(state[44]);
cam.c26=__longlong_as_double(state[45]);
cam.c27=__longlong_as_double(state[46]);
cam.c28=__longlong_as_double(state[47]);
cam.c29=__longlong_as_double(state[48]);
cam.c30=__longlong_as_double(state[49]);
cam.c31=__longlong_as_double(state[50]);
cam.fwcoef=__longlong_as_double(state[51]);
cam.fwc1=__longlong_as_double(state[52]);
cam.fwc2=__longlong_as_double(state[53]);
cam.fc1=__longlong_as_double(state[54]);
cam.amco2=__longlong_as_double(state[55]);
cam.amd=__longlong_as_double(state[56]);
cam.p0=__longlong_as_double(state[57]);
cam.ah2onw=cam_bind<double>(state+58);
cam.eh2onw=cam_bind<double>(state+76);
cam.ah2ow=cam_bind<double>(state+94);
cam.cn_ah2ow=cam_bind<double>(state+112);
cam.cn_eh2ow=cam_bind<double>(state+130);
cam.ln_ah2ow=cam_bind<double>(state+148);
cam.ln_eh2ow=cam_bind<double>(state+166);
cam.coefh=cam_bind<double>(state+184);
cam.coefj=cam_bind<double>(state+202);
cam.coefk=cam_bind<double>(state+220);
cam.estblh2o=cam_bind<double>(state+238);
cam.fat=cam_bind<double>(state+256);
cam.fet=cam_bind<double>(state+274);
cam.gravit=__longlong_as_double(state[292]);
cam.rga=__longlong_as_double(state[293]);
cam.gravmks=__longlong_as_double(state[294]);
cam.cpair=__longlong_as_double(state[295]);
cam.epsilo=__longlong_as_double(state[296]);
cam.epsqs=__longlong_as_double(state[297]);
cam.sslp=__longlong_as_double(state[298]);
cam.stebol=__longlong_as_double(state[299]);
cam.rgsslp=__longlong_as_double(state[300]);
cam.dpfo3=__longlong_as_double(state[301]);
cam.dpfco2=__longlong_as_double(state[302]);
cam.dayspy=__longlong_as_double(state[303]);
cam.pie=__longlong_as_double(state[304]);
cam.mwdry=__longlong_as_double(state[305]);
cam.scon=__longlong_as_double(state[306]);
cam.co2mmr=__longlong_as_double(state[307]);
cam.mwco2=__longlong_as_double(state[308]);
cam.mwh2o=__longlong_as_double(state[309]);
cam.mwch4=__longlong_as_double(state[310]);
cam.mwn2o=__longlong_as_double(state[311]);
cam.mwf11=__longlong_as_double(state[312]);
cam.mwf12=__longlong_as_double(state[313]);
cam.cappa=__longlong_as_double(state[314]);
cam.rair=__longlong_as_double(state[315]);
cam.tmelt=__longlong_as_double(state[316]);
cam.r_universal=__longlong_as_double(state[317]);
cam.latvap=__longlong_as_double(state[318]);
cam.latice=__longlong_as_double(state[319]);
cam.zvir=__longlong_as_double(state[320]);
cam.estbl=cam_bind<double>(state+321);
cam.tmin=__longlong_as_double(state[339]);
cam.tmax=__longlong_as_double(state[340]);
cam.pcf=cam_bind<double>(state+341);
cam.ksul=cam_bind<double>(state+359);
cam.wsul=cam_bind<double>(state+377);
cam.gsul=cam_bind<double>(state+395);
cam.ksslt=cam_bind<double>(state+413);
cam.wsslt=cam_bind<double>(state+431);
cam.gsslt=cam_bind<double>(state+449);
cam.kcphil=cam_bind<double>(state+467);
cam.wcphil=cam_bind<double>(state+485);
cam.gcphil=cam_bind<double>(state+503);
cam.kbg=cam_bind<double>(state+521);
cam.wbg=cam_bind<double>(state+539);
cam.gbg=cam_bind<double>(state+557);
cam.kcphob=cam_bind<double>(state+575);
cam.wcphob=cam_bind<double>(state+593);
cam.gcphob=cam_bind<double>(state+611);
cam.kcb=cam_bind<double>(state+629);
cam.wcb=cam_bind<double>(state+647);
cam.gcb=cam_bind<double>(state+665);
cam.kvolc=cam_bind<double>(state+683);
cam.wvolc=cam_bind<double>(state+701);
cam.gvolc=cam_bind<double>(state+719);
cam.kdst=cam_bind<double>(state+737);
cam.wdst=cam_bind<double>(state+755);
cam.gdst=cam_bind<double>(state+773);
cam.cplos=__longlong_as_double(state[791]);
cam.cplol=__longlong_as_double(state[792]);
cam.co2vmr=__longlong_as_double(state[793]);
cam.n2ovmr=__longlong_as_double(state[794]);
cam.ch4vmr=__longlong_as_double(state[795]);
cam.f11vmr=__longlong_as_double(state[796]);
cam.f12vmr=__longlong_as_double(state[797]);
cam.yrdata=cam_bind<int>(state+798);
cam.co2=cam_bind<double>(state+816);
cam.ntoplw=int(state[834]);
cam.masterproc=(state[835]!=0);
cam.ozncyc=(state[836]!=0);
cam.indirect=(state[837]!=0);
cam.radforce=(state[838]!=0);
cam.trace_gas=(state[839]!=0);
cam.strat_volcanic=(state[840]!=0);
cam.retab=cam_bind<double>(state+841);
cam.abarl=cam_bind<double>(state+859);
cam.bbarl=cam_bind<double>(state+877);
cam.cbarl=cam_bind<double>(state+895);
cam.dbarl=cam_bind<double>(state+913);
cam.ebarl=cam_bind<double>(state+931);
cam.fbarl=cam_bind<double>(state+949);
cam.abari=cam_bind<double>(state+967);
cam.bbari=cam_bind<double>(state+985);
cam.cbari=cam_bind<double>(state+1003);
cam.dbari=cam_bind<double>(state+1021);
cam.ebari=cam_bind<double>(state+1039);
cam.fbari=cam_bind<double>(state+1057);
cam.delta=__longlong_as_double(state[1075]);
cam.o2mmr=__longlong_as_double(state[1076]);
cam.frcsol=cam_bind<double>(state+1077);
cam.wavmin=cam_bind<double>(state+1095);
cam.wavmax=cam_bind<double>(state+1113);
cam.raytau=cam_bind<double>(state+1131);
cam.abh2o=cam_bind<double>(state+1149);
cam.abo3=cam_bind<double>(state+1167);
cam.abco2=cam_bind<double>(state+1185);
cam.abo2=cam_bind<double>(state+1203);
cam.ph2o=cam_bind<double>(state+1221);
cam.pco2=cam_bind<double>(state+1239);
cam.po2=cam_bind<double>(state+1257);
cam.nirwgt=cam_bind<double>(state+1275);
cam.amo=__longlong_as_double(state[1293]);
auto gravx=__int_as_float(int(args[0]));
auto cpairx=__int_as_float(int(args[1]));
auto epsilox=__int_as_float(int(args[2]));
auto stebolx=__int_as_float(int(args[3]));
auto pstdx=__longlong_as_double(args[4]);
cam.radini(gravx,cpairx,epsilox,stebolx,pstdx);
state[0]=cam.ixcldliq;
state[1]=cam.ixcldice;
state[2]=cam.idxsul;
state[3]=cam.idxsslt;
state[4]=cam.idxdustfirst;
state[5]=cam.idxcarbonfirst;
state[6]=cam.idxocpho;
state[7]=cam.idxbcpho;
state[8]=cam.idxocphi;
state[9]=cam.idxbcphi;
state[10]=cam.idxbg;
state[11]=cam.idxvolc;
state[12]=cam.mxaerl;
state[13]=__double_as_longlong(cam.sulscl_rf);
state[14]=__double_as_longlong(cam.carscl_rf);
state[15]=__double_as_longlong(cam.ssltscl_rf);
state[16]=__double_as_longlong(cam.dustscl_rf);
state[17]=__double_as_longlong(cam.bgscl_rf);
state[18]=__double_as_longlong(cam.volcscl_rf);
state[19]=__double_as_longlong(cam.tauback);
state[20]=__double_as_longlong(cam.sulscl);
state[21]=__double_as_longlong(cam.carscl);
state[22]=__double_as_longlong(cam.ssltscl);
state[23]=__double_as_longlong(cam.dustscl);
state[24]=__double_as_longlong(cam.volcscl);
state[43]=__double_as_longlong(cam.c16);
state[44]=__double_as_longlong(cam.c17);
state[45]=__double_as_longlong(cam.c26);
state[46]=__double_as_longlong(cam.c27);
state[47]=__double_as_longlong(cam.c28);
state[48]=__double_as_longlong(cam.c29);
state[49]=__double_as_longlong(cam.c30);
state[50]=__double_as_longlong(cam.c31);
state[51]=__double_as_longlong(cam.fwcoef);
state[52]=__double_as_longlong(cam.fwc1);
state[53]=__double_as_longlong(cam.fwc2);
state[54]=__double_as_longlong(cam.fc1);
state[55]=__double_as_longlong(cam.amco2);
state[56]=__double_as_longlong(cam.amd);
state[57]=__double_as_longlong(cam.p0);
state[292]=__double_as_longlong(cam.gravit);
state[293]=__double_as_longlong(cam.rga);
state[294]=__double_as_longlong(cam.gravmks);
state[295]=__double_as_longlong(cam.cpair);
state[296]=__double_as_longlong(cam.epsilo);
state[297]=__double_as_longlong(cam.epsqs);
state[298]=__double_as_longlong(cam.sslp);
state[299]=__double_as_longlong(cam.stebol);
state[300]=__double_as_longlong(cam.rgsslp);
state[301]=__double_as_longlong(cam.dpfo3);
state[302]=__double_as_longlong(cam.dpfco2);
state[303]=__double_as_longlong(cam.dayspy);
state[304]=__double_as_longlong(cam.pie);
state[305]=__double_as_longlong(cam.mwdry);
state[306]=__double_as_longlong(cam.scon);
state[307]=__double_as_longlong(cam.co2mmr);
state[308]=__double_as_longlong(cam.mwco2);
state[309]=__double_as_longlong(cam.mwh2o);
state[310]=__double_as_longlong(cam.mwch4);
state[311]=__double_as_longlong(cam.mwn2o);
state[312]=__double_as_longlong(cam.mwf11);
state[313]=__double_as_longlong(cam.mwf12);
state[314]=__double_as_longlong(cam.cappa);
state[315]=__double_as_longlong(cam.rair);
state[316]=__double_as_longlong(cam.tmelt);
state[317]=__double_as_longlong(cam.r_universal);
state[318]=__double_as_longlong(cam.latvap);
state[319]=__double_as_longlong(cam.latice);
state[320]=__double_as_longlong(cam.zvir);
state[339]=__double_as_longlong(cam.tmin);
state[340]=__double_as_longlong(cam.tmax);
state[791]=__double_as_longlong(cam.cplos);
state[792]=__double_as_longlong(cam.cplol);
state[793]=__double_as_longlong(cam.co2vmr);
state[794]=__double_as_longlong(cam.n2ovmr);
state[795]=__double_as_longlong(cam.ch4vmr);
state[796]=__double_as_longlong(cam.f11vmr);
state[797]=__double_as_longlong(cam.f12vmr);
state[834]=cam.ntoplw;
state[835]=cam.masterproc;
state[836]=cam.ozncyc;
state[837]=cam.indirect;
state[838]=cam.radforce;
state[839]=cam.trace_gas;
state[840]=cam.strat_volcanic;
state[1075]=__double_as_longlong(cam.delta);
state[1076]=__double_as_longlong(cam.o2mmr);
state[1293]=__double_as_longlong(cam.amo);
args[0]=__float_as_uint(gravx);
args[1]=__float_as_uint(cpairx);
args[2]=__float_as_uint(epsilox);
args[3]=__float_as_uint(stebolx);
args[4]=__double_as_longlong(pstdx);
*error=cam.error;}
extern "C" __global__ void cam_radaeini(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
cam.ixcldliq=int(state[0]);
cam.ixcldice=int(state[1]);
cam.idxsul=int(state[2]);
cam.idxsslt=int(state[3]);
cam.idxdustfirst=int(state[4]);
cam.idxcarbonfirst=int(state[5]);
cam.idxocpho=int(state[6]);
cam.idxbcpho=int(state[7]);
cam.idxocphi=int(state[8]);
cam.idxbcphi=int(state[9]);
cam.idxbg=int(state[10]);
cam.idxvolc=int(state[11]);
cam.mxaerl=int(state[12]);
cam.sulscl_rf=__longlong_as_double(state[13]);
cam.carscl_rf=__longlong_as_double(state[14]);
cam.ssltscl_rf=__longlong_as_double(state[15]);
cam.dustscl_rf=__longlong_as_double(state[16]);
cam.bgscl_rf=__longlong_as_double(state[17]);
cam.volcscl_rf=__longlong_as_double(state[18]);
cam.tauback=__longlong_as_double(state[19]);
cam.sulscl=__longlong_as_double(state[20]);
cam.carscl=__longlong_as_double(state[21]);
cam.ssltscl=__longlong_as_double(state[22]);
cam.dustscl=__longlong_as_double(state[23]);
cam.volcscl=__longlong_as_double(state[24]);
cam.abs_cff_mss_aer=cam_bind<double>(state+25);
cam.c16=__longlong_as_double(state[43]);
cam.c17=__longlong_as_double(state[44]);
cam.c26=__longlong_as_double(state[45]);
cam.c27=__longlong_as_double(state[46]);
cam.c28=__longlong_as_double(state[47]);
cam.c29=__longlong_as_double(state[48]);
cam.c30=__longlong_as_double(state[49]);
cam.c31=__longlong_as_double(state[50]);
cam.fwcoef=__longlong_as_double(state[51]);
cam.fwc1=__longlong_as_double(state[52]);
cam.fwc2=__longlong_as_double(state[53]);
cam.fc1=__longlong_as_double(state[54]);
cam.amco2=__longlong_as_double(state[55]);
cam.amd=__longlong_as_double(state[56]);
cam.p0=__longlong_as_double(state[57]);
cam.ah2onw=cam_bind<double>(state+58);
cam.eh2onw=cam_bind<double>(state+76);
cam.ah2ow=cam_bind<double>(state+94);
cam.cn_ah2ow=cam_bind<double>(state+112);
cam.cn_eh2ow=cam_bind<double>(state+130);
cam.ln_ah2ow=cam_bind<double>(state+148);
cam.ln_eh2ow=cam_bind<double>(state+166);
cam.coefh=cam_bind<double>(state+184);
cam.coefj=cam_bind<double>(state+202);
cam.coefk=cam_bind<double>(state+220);
cam.estblh2o=cam_bind<double>(state+238);
cam.fat=cam_bind<double>(state+256);
cam.fet=cam_bind<double>(state+274);
cam.gravit=__longlong_as_double(state[292]);
cam.rga=__longlong_as_double(state[293]);
cam.gravmks=__longlong_as_double(state[294]);
cam.cpair=__longlong_as_double(state[295]);
cam.epsilo=__longlong_as_double(state[296]);
cam.epsqs=__longlong_as_double(state[297]);
cam.sslp=__longlong_as_double(state[298]);
cam.stebol=__longlong_as_double(state[299]);
cam.rgsslp=__longlong_as_double(state[300]);
cam.dpfo3=__longlong_as_double(state[301]);
cam.dpfco2=__longlong_as_double(state[302]);
cam.dayspy=__longlong_as_double(state[303]);
cam.pie=__longlong_as_double(state[304]);
cam.mwdry=__longlong_as_double(state[305]);
cam.scon=__longlong_as_double(state[306]);
cam.co2mmr=__longlong_as_double(state[307]);
cam.mwco2=__longlong_as_double(state[308]);
cam.mwh2o=__longlong_as_double(state[309]);
cam.mwch4=__longlong_as_double(state[310]);
cam.mwn2o=__longlong_as_double(state[311]);
cam.mwf11=__longlong_as_double(state[312]);
cam.mwf12=__longlong_as_double(state[313]);
cam.cappa=__longlong_as_double(state[314]);
cam.rair=__longlong_as_double(state[315]);
cam.tmelt=__longlong_as_double(state[316]);
cam.r_universal=__longlong_as_double(state[317]);
cam.latvap=__longlong_as_double(state[318]);
cam.latice=__longlong_as_double(state[319]);
cam.zvir=__longlong_as_double(state[320]);
cam.estbl=cam_bind<double>(state+321);
cam.tmin=__longlong_as_double(state[339]);
cam.tmax=__longlong_as_double(state[340]);
cam.pcf=cam_bind<double>(state+341);
cam.ksul=cam_bind<double>(state+359);
cam.wsul=cam_bind<double>(state+377);
cam.gsul=cam_bind<double>(state+395);
cam.ksslt=cam_bind<double>(state+413);
cam.wsslt=cam_bind<double>(state+431);
cam.gsslt=cam_bind<double>(state+449);
cam.kcphil=cam_bind<double>(state+467);
cam.wcphil=cam_bind<double>(state+485);
cam.gcphil=cam_bind<double>(state+503);
cam.kbg=cam_bind<double>(state+521);
cam.wbg=cam_bind<double>(state+539);
cam.gbg=cam_bind<double>(state+557);
cam.kcphob=cam_bind<double>(state+575);
cam.wcphob=cam_bind<double>(state+593);
cam.gcphob=cam_bind<double>(state+611);
cam.kcb=cam_bind<double>(state+629);
cam.wcb=cam_bind<double>(state+647);
cam.gcb=cam_bind<double>(state+665);
cam.kvolc=cam_bind<double>(state+683);
cam.wvolc=cam_bind<double>(state+701);
cam.gvolc=cam_bind<double>(state+719);
cam.kdst=cam_bind<double>(state+737);
cam.wdst=cam_bind<double>(state+755);
cam.gdst=cam_bind<double>(state+773);
cam.cplos=__longlong_as_double(state[791]);
cam.cplol=__longlong_as_double(state[792]);
cam.co2vmr=__longlong_as_double(state[793]);
cam.n2ovmr=__longlong_as_double(state[794]);
cam.ch4vmr=__longlong_as_double(state[795]);
cam.f11vmr=__longlong_as_double(state[796]);
cam.f12vmr=__longlong_as_double(state[797]);
cam.yrdata=cam_bind<int>(state+798);
cam.co2=cam_bind<double>(state+816);
cam.ntoplw=int(state[834]);
cam.masterproc=(state[835]!=0);
cam.ozncyc=(state[836]!=0);
cam.indirect=(state[837]!=0);
cam.radforce=(state[838]!=0);
cam.trace_gas=(state[839]!=0);
cam.strat_volcanic=(state[840]!=0);
cam.retab=cam_bind<double>(state+841);
cam.abarl=cam_bind<double>(state+859);
cam.bbarl=cam_bind<double>(state+877);
cam.cbarl=cam_bind<double>(state+895);
cam.dbarl=cam_bind<double>(state+913);
cam.ebarl=cam_bind<double>(state+931);
cam.fbarl=cam_bind<double>(state+949);
cam.abari=cam_bind<double>(state+967);
cam.bbari=cam_bind<double>(state+985);
cam.cbari=cam_bind<double>(state+1003);
cam.dbari=cam_bind<double>(state+1021);
cam.ebari=cam_bind<double>(state+1039);
cam.fbari=cam_bind<double>(state+1057);
cam.delta=__longlong_as_double(state[1075]);
cam.o2mmr=__longlong_as_double(state[1076]);
cam.frcsol=cam_bind<double>(state+1077);
cam.wavmin=cam_bind<double>(state+1095);
cam.wavmax=cam_bind<double>(state+1113);
cam.raytau=cam_bind<double>(state+1131);
cam.abh2o=cam_bind<double>(state+1149);
cam.abo3=cam_bind<double>(state+1167);
cam.abco2=cam_bind<double>(state+1185);
cam.abo2=cam_bind<double>(state+1203);
cam.ph2o=cam_bind<double>(state+1221);
cam.pco2=cam_bind<double>(state+1239);
cam.po2=cam_bind<double>(state+1257);
cam.nirwgt=cam_bind<double>(state+1275);
cam.amo=__longlong_as_double(state[1293]);
auto pstdx=__longlong_as_double(args[0]);
auto mwdryx=__longlong_as_double(args[1]);
auto mwco2x=__longlong_as_double(args[2]);
cam.radaeini(pstdx,mwdryx,mwco2x);
state[0]=cam.ixcldliq;
state[1]=cam.ixcldice;
state[2]=cam.idxsul;
state[3]=cam.idxsslt;
state[4]=cam.idxdustfirst;
state[5]=cam.idxcarbonfirst;
state[6]=cam.idxocpho;
state[7]=cam.idxbcpho;
state[8]=cam.idxocphi;
state[9]=cam.idxbcphi;
state[10]=cam.idxbg;
state[11]=cam.idxvolc;
state[12]=cam.mxaerl;
state[13]=__double_as_longlong(cam.sulscl_rf);
state[14]=__double_as_longlong(cam.carscl_rf);
state[15]=__double_as_longlong(cam.ssltscl_rf);
state[16]=__double_as_longlong(cam.dustscl_rf);
state[17]=__double_as_longlong(cam.bgscl_rf);
state[18]=__double_as_longlong(cam.volcscl_rf);
state[19]=__double_as_longlong(cam.tauback);
state[20]=__double_as_longlong(cam.sulscl);
state[21]=__double_as_longlong(cam.carscl);
state[22]=__double_as_longlong(cam.ssltscl);
state[23]=__double_as_longlong(cam.dustscl);
state[24]=__double_as_longlong(cam.volcscl);
state[43]=__double_as_longlong(cam.c16);
state[44]=__double_as_longlong(cam.c17);
state[45]=__double_as_longlong(cam.c26);
state[46]=__double_as_longlong(cam.c27);
state[47]=__double_as_longlong(cam.c28);
state[48]=__double_as_longlong(cam.c29);
state[49]=__double_as_longlong(cam.c30);
state[50]=__double_as_longlong(cam.c31);
state[51]=__double_as_longlong(cam.fwcoef);
state[52]=__double_as_longlong(cam.fwc1);
state[53]=__double_as_longlong(cam.fwc2);
state[54]=__double_as_longlong(cam.fc1);
state[55]=__double_as_longlong(cam.amco2);
state[56]=__double_as_longlong(cam.amd);
state[57]=__double_as_longlong(cam.p0);
state[292]=__double_as_longlong(cam.gravit);
state[293]=__double_as_longlong(cam.rga);
state[294]=__double_as_longlong(cam.gravmks);
state[295]=__double_as_longlong(cam.cpair);
state[296]=__double_as_longlong(cam.epsilo);
state[297]=__double_as_longlong(cam.epsqs);
state[298]=__double_as_longlong(cam.sslp);
state[299]=__double_as_longlong(cam.stebol);
state[300]=__double_as_longlong(cam.rgsslp);
state[301]=__double_as_longlong(cam.dpfo3);
state[302]=__double_as_longlong(cam.dpfco2);
state[303]=__double_as_longlong(cam.dayspy);
state[304]=__double_as_longlong(cam.pie);
state[305]=__double_as_longlong(cam.mwdry);
state[306]=__double_as_longlong(cam.scon);
state[307]=__double_as_longlong(cam.co2mmr);
state[308]=__double_as_longlong(cam.mwco2);
state[309]=__double_as_longlong(cam.mwh2o);
state[310]=__double_as_longlong(cam.mwch4);
state[311]=__double_as_longlong(cam.mwn2o);
state[312]=__double_as_longlong(cam.mwf11);
state[313]=__double_as_longlong(cam.mwf12);
state[314]=__double_as_longlong(cam.cappa);
state[315]=__double_as_longlong(cam.rair);
state[316]=__double_as_longlong(cam.tmelt);
state[317]=__double_as_longlong(cam.r_universal);
state[318]=__double_as_longlong(cam.latvap);
state[319]=__double_as_longlong(cam.latice);
state[320]=__double_as_longlong(cam.zvir);
state[339]=__double_as_longlong(cam.tmin);
state[340]=__double_as_longlong(cam.tmax);
state[791]=__double_as_longlong(cam.cplos);
state[792]=__double_as_longlong(cam.cplol);
state[793]=__double_as_longlong(cam.co2vmr);
state[794]=__double_as_longlong(cam.n2ovmr);
state[795]=__double_as_longlong(cam.ch4vmr);
state[796]=__double_as_longlong(cam.f11vmr);
state[797]=__double_as_longlong(cam.f12vmr);
state[834]=cam.ntoplw;
state[835]=cam.masterproc;
state[836]=cam.ozncyc;
state[837]=cam.indirect;
state[838]=cam.radforce;
state[839]=cam.trace_gas;
state[840]=cam.strat_volcanic;
state[1075]=__double_as_longlong(cam.delta);
state[1076]=__double_as_longlong(cam.o2mmr);
state[1293]=__double_as_longlong(cam.amo);
args[0]=__double_as_longlong(pstdx);
args[1]=__double_as_longlong(mwdryx);
args[2]=__double_as_longlong(mwco2x);
*error=cam.error;}
extern "C" __global__ void cam_aerosol_init(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
cam.ixcldliq=int(state[0]);
cam.ixcldice=int(state[1]);
cam.idxsul=int(state[2]);
cam.idxsslt=int(state[3]);
cam.idxdustfirst=int(state[4]);
cam.idxcarbonfirst=int(state[5]);
cam.idxocpho=int(state[6]);
cam.idxbcpho=int(state[7]);
cam.idxocphi=int(state[8]);
cam.idxbcphi=int(state[9]);
cam.idxbg=int(state[10]);
cam.idxvolc=int(state[11]);
cam.mxaerl=int(state[12]);
cam.sulscl_rf=__longlong_as_double(state[13]);
cam.carscl_rf=__longlong_as_double(state[14]);
cam.ssltscl_rf=__longlong_as_double(state[15]);
cam.dustscl_rf=__longlong_as_double(state[16]);
cam.bgscl_rf=__longlong_as_double(state[17]);
cam.volcscl_rf=__longlong_as_double(state[18]);
cam.tauback=__longlong_as_double(state[19]);
cam.sulscl=__longlong_as_double(state[20]);
cam.carscl=__longlong_as_double(state[21]);
cam.ssltscl=__longlong_as_double(state[22]);
cam.dustscl=__longlong_as_double(state[23]);
cam.volcscl=__longlong_as_double(state[24]);
cam.abs_cff_mss_aer=cam_bind<double>(state+25);
cam.c16=__longlong_as_double(state[43]);
cam.c17=__longlong_as_double(state[44]);
cam.c26=__longlong_as_double(state[45]);
cam.c27=__longlong_as_double(state[46]);
cam.c28=__longlong_as_double(state[47]);
cam.c29=__longlong_as_double(state[48]);
cam.c30=__longlong_as_double(state[49]);
cam.c31=__longlong_as_double(state[50]);
cam.fwcoef=__longlong_as_double(state[51]);
cam.fwc1=__longlong_as_double(state[52]);
cam.fwc2=__longlong_as_double(state[53]);
cam.fc1=__longlong_as_double(state[54]);
cam.amco2=__longlong_as_double(state[55]);
cam.amd=__longlong_as_double(state[56]);
cam.p0=__longlong_as_double(state[57]);
cam.ah2onw=cam_bind<double>(state+58);
cam.eh2onw=cam_bind<double>(state+76);
cam.ah2ow=cam_bind<double>(state+94);
cam.cn_ah2ow=cam_bind<double>(state+112);
cam.cn_eh2ow=cam_bind<double>(state+130);
cam.ln_ah2ow=cam_bind<double>(state+148);
cam.ln_eh2ow=cam_bind<double>(state+166);
cam.coefh=cam_bind<double>(state+184);
cam.coefj=cam_bind<double>(state+202);
cam.coefk=cam_bind<double>(state+220);
cam.estblh2o=cam_bind<double>(state+238);
cam.fat=cam_bind<double>(state+256);
cam.fet=cam_bind<double>(state+274);
cam.gravit=__longlong_as_double(state[292]);
cam.rga=__longlong_as_double(state[293]);
cam.gravmks=__longlong_as_double(state[294]);
cam.cpair=__longlong_as_double(state[295]);
cam.epsilo=__longlong_as_double(state[296]);
cam.epsqs=__longlong_as_double(state[297]);
cam.sslp=__longlong_as_double(state[298]);
cam.stebol=__longlong_as_double(state[299]);
cam.rgsslp=__longlong_as_double(state[300]);
cam.dpfo3=__longlong_as_double(state[301]);
cam.dpfco2=__longlong_as_double(state[302]);
cam.dayspy=__longlong_as_double(state[303]);
cam.pie=__longlong_as_double(state[304]);
cam.mwdry=__longlong_as_double(state[305]);
cam.scon=__longlong_as_double(state[306]);
cam.co2mmr=__longlong_as_double(state[307]);
cam.mwco2=__longlong_as_double(state[308]);
cam.mwh2o=__longlong_as_double(state[309]);
cam.mwch4=__longlong_as_double(state[310]);
cam.mwn2o=__longlong_as_double(state[311]);
cam.mwf11=__longlong_as_double(state[312]);
cam.mwf12=__longlong_as_double(state[313]);
cam.cappa=__longlong_as_double(state[314]);
cam.rair=__longlong_as_double(state[315]);
cam.tmelt=__longlong_as_double(state[316]);
cam.r_universal=__longlong_as_double(state[317]);
cam.latvap=__longlong_as_double(state[318]);
cam.latice=__longlong_as_double(state[319]);
cam.zvir=__longlong_as_double(state[320]);
cam.estbl=cam_bind<double>(state+321);
cam.tmin=__longlong_as_double(state[339]);
cam.tmax=__longlong_as_double(state[340]);
cam.pcf=cam_bind<double>(state+341);
cam.ksul=cam_bind<double>(state+359);
cam.wsul=cam_bind<double>(state+377);
cam.gsul=cam_bind<double>(state+395);
cam.ksslt=cam_bind<double>(state+413);
cam.wsslt=cam_bind<double>(state+431);
cam.gsslt=cam_bind<double>(state+449);
cam.kcphil=cam_bind<double>(state+467);
cam.wcphil=cam_bind<double>(state+485);
cam.gcphil=cam_bind<double>(state+503);
cam.kbg=cam_bind<double>(state+521);
cam.wbg=cam_bind<double>(state+539);
cam.gbg=cam_bind<double>(state+557);
cam.kcphob=cam_bind<double>(state+575);
cam.wcphob=cam_bind<double>(state+593);
cam.gcphob=cam_bind<double>(state+611);
cam.kcb=cam_bind<double>(state+629);
cam.wcb=cam_bind<double>(state+647);
cam.gcb=cam_bind<double>(state+665);
cam.kvolc=cam_bind<double>(state+683);
cam.wvolc=cam_bind<double>(state+701);
cam.gvolc=cam_bind<double>(state+719);
cam.kdst=cam_bind<double>(state+737);
cam.wdst=cam_bind<double>(state+755);
cam.gdst=cam_bind<double>(state+773);
cam.cplos=__longlong_as_double(state[791]);
cam.cplol=__longlong_as_double(state[792]);
cam.co2vmr=__longlong_as_double(state[793]);
cam.n2ovmr=__longlong_as_double(state[794]);
cam.ch4vmr=__longlong_as_double(state[795]);
cam.f11vmr=__longlong_as_double(state[796]);
cam.f12vmr=__longlong_as_double(state[797]);
cam.yrdata=cam_bind<int>(state+798);
cam.co2=cam_bind<double>(state+816);
cam.ntoplw=int(state[834]);
cam.masterproc=(state[835]!=0);
cam.ozncyc=(state[836]!=0);
cam.indirect=(state[837]!=0);
cam.radforce=(state[838]!=0);
cam.trace_gas=(state[839]!=0);
cam.strat_volcanic=(state[840]!=0);
cam.retab=cam_bind<double>(state+841);
cam.abarl=cam_bind<double>(state+859);
cam.bbarl=cam_bind<double>(state+877);
cam.cbarl=cam_bind<double>(state+895);
cam.dbarl=cam_bind<double>(state+913);
cam.ebarl=cam_bind<double>(state+931);
cam.fbarl=cam_bind<double>(state+949);
cam.abari=cam_bind<double>(state+967);
cam.bbari=cam_bind<double>(state+985);
cam.cbari=cam_bind<double>(state+1003);
cam.dbari=cam_bind<double>(state+1021);
cam.ebari=cam_bind<double>(state+1039);
cam.fbari=cam_bind<double>(state+1057);
cam.delta=__longlong_as_double(state[1075]);
cam.o2mmr=__longlong_as_double(state[1076]);
cam.frcsol=cam_bind<double>(state+1077);
cam.wavmin=cam_bind<double>(state+1095);
cam.wavmax=cam_bind<double>(state+1113);
cam.raytau=cam_bind<double>(state+1131);
cam.abh2o=cam_bind<double>(state+1149);
cam.abo3=cam_bind<double>(state+1167);
cam.abco2=cam_bind<double>(state+1185);
cam.abo2=cam_bind<double>(state+1203);
cam.ph2o=cam_bind<double>(state+1221);
cam.pco2=cam_bind<double>(state+1239);
cam.po2=cam_bind<double>(state+1257);
cam.nirwgt=cam_bind<double>(state+1275);
cam.amo=__longlong_as_double(state[1293]);
auto naerlevels=int(args[0]);
auto ncells=int(args[1]);
auto ncellssolve=int(args[2]);
auto nvertlevels=int(args[3]);
auto m_hybi=cam_bind<float>(args+4);
auto m_psp=cam_bind<float>(args+22);
auto pressure_b=cam_bind<float>(args+40);
auto aerosolcn=cam_bind<float>(args+58);
auto aerosolcp=cam_bind<float>(args+76);
auto idxsul_l=int(args[94]);
auto idxsslt_l=int(args[95]);
auto idxdustfirst_l=int(args[96]);
auto idxocpho_l=int(args[97]);
auto idxcarbonfirst_l=int(args[98]);
auto idxbcpho_l=int(args[99]);
auto idxocphi_l=int(args[100]);
auto idxbcphi_l=int(args[101]);
auto idxbg_l=int(args[102]);
auto idxvolc_l=int(args[103]);
auto max_mxaerl=int(args[104]);
cam.aerosol_init(naerlevels,ncells,ncellssolve,nvertlevels,m_hybi,m_psp,pressure_b,aerosolcn,aerosolcp,idxsul_l,idxsslt_l,idxdustfirst_l,idxocpho_l,idxcarbonfirst_l,idxbcpho_l,idxocphi_l,idxbcphi_l,idxbg_l,idxvolc_l,max_mxaerl);
state[0]=cam.ixcldliq;
state[1]=cam.ixcldice;
state[2]=cam.idxsul;
state[3]=cam.idxsslt;
state[4]=cam.idxdustfirst;
state[5]=cam.idxcarbonfirst;
state[6]=cam.idxocpho;
state[7]=cam.idxbcpho;
state[8]=cam.idxocphi;
state[9]=cam.idxbcphi;
state[10]=cam.idxbg;
state[11]=cam.idxvolc;
state[12]=cam.mxaerl;
state[13]=__double_as_longlong(cam.sulscl_rf);
state[14]=__double_as_longlong(cam.carscl_rf);
state[15]=__double_as_longlong(cam.ssltscl_rf);
state[16]=__double_as_longlong(cam.dustscl_rf);
state[17]=__double_as_longlong(cam.bgscl_rf);
state[18]=__double_as_longlong(cam.volcscl_rf);
state[19]=__double_as_longlong(cam.tauback);
state[20]=__double_as_longlong(cam.sulscl);
state[21]=__double_as_longlong(cam.carscl);
state[22]=__double_as_longlong(cam.ssltscl);
state[23]=__double_as_longlong(cam.dustscl);
state[24]=__double_as_longlong(cam.volcscl);
state[43]=__double_as_longlong(cam.c16);
state[44]=__double_as_longlong(cam.c17);
state[45]=__double_as_longlong(cam.c26);
state[46]=__double_as_longlong(cam.c27);
state[47]=__double_as_longlong(cam.c28);
state[48]=__double_as_longlong(cam.c29);
state[49]=__double_as_longlong(cam.c30);
state[50]=__double_as_longlong(cam.c31);
state[51]=__double_as_longlong(cam.fwcoef);
state[52]=__double_as_longlong(cam.fwc1);
state[53]=__double_as_longlong(cam.fwc2);
state[54]=__double_as_longlong(cam.fc1);
state[55]=__double_as_longlong(cam.amco2);
state[56]=__double_as_longlong(cam.amd);
state[57]=__double_as_longlong(cam.p0);
state[292]=__double_as_longlong(cam.gravit);
state[293]=__double_as_longlong(cam.rga);
state[294]=__double_as_longlong(cam.gravmks);
state[295]=__double_as_longlong(cam.cpair);
state[296]=__double_as_longlong(cam.epsilo);
state[297]=__double_as_longlong(cam.epsqs);
state[298]=__double_as_longlong(cam.sslp);
state[299]=__double_as_longlong(cam.stebol);
state[300]=__double_as_longlong(cam.rgsslp);
state[301]=__double_as_longlong(cam.dpfo3);
state[302]=__double_as_longlong(cam.dpfco2);
state[303]=__double_as_longlong(cam.dayspy);
state[304]=__double_as_longlong(cam.pie);
state[305]=__double_as_longlong(cam.mwdry);
state[306]=__double_as_longlong(cam.scon);
state[307]=__double_as_longlong(cam.co2mmr);
state[308]=__double_as_longlong(cam.mwco2);
state[309]=__double_as_longlong(cam.mwh2o);
state[310]=__double_as_longlong(cam.mwch4);
state[311]=__double_as_longlong(cam.mwn2o);
state[312]=__double_as_longlong(cam.mwf11);
state[313]=__double_as_longlong(cam.mwf12);
state[314]=__double_as_longlong(cam.cappa);
state[315]=__double_as_longlong(cam.rair);
state[316]=__double_as_longlong(cam.tmelt);
state[317]=__double_as_longlong(cam.r_universal);
state[318]=__double_as_longlong(cam.latvap);
state[319]=__double_as_longlong(cam.latice);
state[320]=__double_as_longlong(cam.zvir);
state[339]=__double_as_longlong(cam.tmin);
state[340]=__double_as_longlong(cam.tmax);
state[791]=__double_as_longlong(cam.cplos);
state[792]=__double_as_longlong(cam.cplol);
state[793]=__double_as_longlong(cam.co2vmr);
state[794]=__double_as_longlong(cam.n2ovmr);
state[795]=__double_as_longlong(cam.ch4vmr);
state[796]=__double_as_longlong(cam.f11vmr);
state[797]=__double_as_longlong(cam.f12vmr);
state[834]=cam.ntoplw;
state[835]=cam.masterproc;
state[836]=cam.ozncyc;
state[837]=cam.indirect;
state[838]=cam.radforce;
state[839]=cam.trace_gas;
state[840]=cam.strat_volcanic;
state[1075]=__double_as_longlong(cam.delta);
state[1076]=__double_as_longlong(cam.o2mmr);
state[1293]=__double_as_longlong(cam.amo);
args[0]=naerlevels;
args[1]=ncells;
args[2]=ncellssolve;
args[3]=nvertlevels;
args[94]=idxsul_l;
args[95]=idxsslt_l;
args[96]=idxdustfirst_l;
args[97]=idxocpho_l;
args[98]=idxcarbonfirst_l;
args[99]=idxbcpho_l;
args[100]=idxocphi_l;
args[101]=idxbcphi_l;
args[102]=idxbg_l;
args[103]=idxvolc_l;
args[104]=max_mxaerl;
*error=cam.error;}
extern "C" __global__ void cam_aer_optics_initialize(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
cam.ixcldliq=int(state[0]);
cam.ixcldice=int(state[1]);
cam.idxsul=int(state[2]);
cam.idxsslt=int(state[3]);
cam.idxdustfirst=int(state[4]);
cam.idxcarbonfirst=int(state[5]);
cam.idxocpho=int(state[6]);
cam.idxbcpho=int(state[7]);
cam.idxocphi=int(state[8]);
cam.idxbcphi=int(state[9]);
cam.idxbg=int(state[10]);
cam.idxvolc=int(state[11]);
cam.mxaerl=int(state[12]);
cam.sulscl_rf=__longlong_as_double(state[13]);
cam.carscl_rf=__longlong_as_double(state[14]);
cam.ssltscl_rf=__longlong_as_double(state[15]);
cam.dustscl_rf=__longlong_as_double(state[16]);
cam.bgscl_rf=__longlong_as_double(state[17]);
cam.volcscl_rf=__longlong_as_double(state[18]);
cam.tauback=__longlong_as_double(state[19]);
cam.sulscl=__longlong_as_double(state[20]);
cam.carscl=__longlong_as_double(state[21]);
cam.ssltscl=__longlong_as_double(state[22]);
cam.dustscl=__longlong_as_double(state[23]);
cam.volcscl=__longlong_as_double(state[24]);
cam.abs_cff_mss_aer=cam_bind<double>(state+25);
cam.c16=__longlong_as_double(state[43]);
cam.c17=__longlong_as_double(state[44]);
cam.c26=__longlong_as_double(state[45]);
cam.c27=__longlong_as_double(state[46]);
cam.c28=__longlong_as_double(state[47]);
cam.c29=__longlong_as_double(state[48]);
cam.c30=__longlong_as_double(state[49]);
cam.c31=__longlong_as_double(state[50]);
cam.fwcoef=__longlong_as_double(state[51]);
cam.fwc1=__longlong_as_double(state[52]);
cam.fwc2=__longlong_as_double(state[53]);
cam.fc1=__longlong_as_double(state[54]);
cam.amco2=__longlong_as_double(state[55]);
cam.amd=__longlong_as_double(state[56]);
cam.p0=__longlong_as_double(state[57]);
cam.ah2onw=cam_bind<double>(state+58);
cam.eh2onw=cam_bind<double>(state+76);
cam.ah2ow=cam_bind<double>(state+94);
cam.cn_ah2ow=cam_bind<double>(state+112);
cam.cn_eh2ow=cam_bind<double>(state+130);
cam.ln_ah2ow=cam_bind<double>(state+148);
cam.ln_eh2ow=cam_bind<double>(state+166);
cam.coefh=cam_bind<double>(state+184);
cam.coefj=cam_bind<double>(state+202);
cam.coefk=cam_bind<double>(state+220);
cam.estblh2o=cam_bind<double>(state+238);
cam.fat=cam_bind<double>(state+256);
cam.fet=cam_bind<double>(state+274);
cam.gravit=__longlong_as_double(state[292]);
cam.rga=__longlong_as_double(state[293]);
cam.gravmks=__longlong_as_double(state[294]);
cam.cpair=__longlong_as_double(state[295]);
cam.epsilo=__longlong_as_double(state[296]);
cam.epsqs=__longlong_as_double(state[297]);
cam.sslp=__longlong_as_double(state[298]);
cam.stebol=__longlong_as_double(state[299]);
cam.rgsslp=__longlong_as_double(state[300]);
cam.dpfo3=__longlong_as_double(state[301]);
cam.dpfco2=__longlong_as_double(state[302]);
cam.dayspy=__longlong_as_double(state[303]);
cam.pie=__longlong_as_double(state[304]);
cam.mwdry=__longlong_as_double(state[305]);
cam.scon=__longlong_as_double(state[306]);
cam.co2mmr=__longlong_as_double(state[307]);
cam.mwco2=__longlong_as_double(state[308]);
cam.mwh2o=__longlong_as_double(state[309]);
cam.mwch4=__longlong_as_double(state[310]);
cam.mwn2o=__longlong_as_double(state[311]);
cam.mwf11=__longlong_as_double(state[312]);
cam.mwf12=__longlong_as_double(state[313]);
cam.cappa=__longlong_as_double(state[314]);
cam.rair=__longlong_as_double(state[315]);
cam.tmelt=__longlong_as_double(state[316]);
cam.r_universal=__longlong_as_double(state[317]);
cam.latvap=__longlong_as_double(state[318]);
cam.latice=__longlong_as_double(state[319]);
cam.zvir=__longlong_as_double(state[320]);
cam.estbl=cam_bind<double>(state+321);
cam.tmin=__longlong_as_double(state[339]);
cam.tmax=__longlong_as_double(state[340]);
cam.pcf=cam_bind<double>(state+341);
cam.ksul=cam_bind<double>(state+359);
cam.wsul=cam_bind<double>(state+377);
cam.gsul=cam_bind<double>(state+395);
cam.ksslt=cam_bind<double>(state+413);
cam.wsslt=cam_bind<double>(state+431);
cam.gsslt=cam_bind<double>(state+449);
cam.kcphil=cam_bind<double>(state+467);
cam.wcphil=cam_bind<double>(state+485);
cam.gcphil=cam_bind<double>(state+503);
cam.kbg=cam_bind<double>(state+521);
cam.wbg=cam_bind<double>(state+539);
cam.gbg=cam_bind<double>(state+557);
cam.kcphob=cam_bind<double>(state+575);
cam.wcphob=cam_bind<double>(state+593);
cam.gcphob=cam_bind<double>(state+611);
cam.kcb=cam_bind<double>(state+629);
cam.wcb=cam_bind<double>(state+647);
cam.gcb=cam_bind<double>(state+665);
cam.kvolc=cam_bind<double>(state+683);
cam.wvolc=cam_bind<double>(state+701);
cam.gvolc=cam_bind<double>(state+719);
cam.kdst=cam_bind<double>(state+737);
cam.wdst=cam_bind<double>(state+755);
cam.gdst=cam_bind<double>(state+773);
cam.cplos=__longlong_as_double(state[791]);
cam.cplol=__longlong_as_double(state[792]);
cam.co2vmr=__longlong_as_double(state[793]);
cam.n2ovmr=__longlong_as_double(state[794]);
cam.ch4vmr=__longlong_as_double(state[795]);
cam.f11vmr=__longlong_as_double(state[796]);
cam.f12vmr=__longlong_as_double(state[797]);
cam.yrdata=cam_bind<int>(state+798);
cam.co2=cam_bind<double>(state+816);
cam.ntoplw=int(state[834]);
cam.masterproc=(state[835]!=0);
cam.ozncyc=(state[836]!=0);
cam.indirect=(state[837]!=0);
cam.radforce=(state[838]!=0);
cam.trace_gas=(state[839]!=0);
cam.strat_volcanic=(state[840]!=0);
cam.retab=cam_bind<double>(state+841);
cam.abarl=cam_bind<double>(state+859);
cam.bbarl=cam_bind<double>(state+877);
cam.cbarl=cam_bind<double>(state+895);
cam.dbarl=cam_bind<double>(state+913);
cam.ebarl=cam_bind<double>(state+931);
cam.fbarl=cam_bind<double>(state+949);
cam.abari=cam_bind<double>(state+967);
cam.bbari=cam_bind<double>(state+985);
cam.cbari=cam_bind<double>(state+1003);
cam.dbari=cam_bind<double>(state+1021);
cam.ebari=cam_bind<double>(state+1039);
cam.fbari=cam_bind<double>(state+1057);
cam.delta=__longlong_as_double(state[1075]);
cam.o2mmr=__longlong_as_double(state[1076]);
cam.frcsol=cam_bind<double>(state+1077);
cam.wavmin=cam_bind<double>(state+1095);
cam.wavmax=cam_bind<double>(state+1113);
cam.raytau=cam_bind<double>(state+1131);
cam.abh2o=cam_bind<double>(state+1149);
cam.abo3=cam_bind<double>(state+1167);
cam.abco2=cam_bind<double>(state+1185);
cam.abo2=cam_bind<double>(state+1203);
cam.ph2o=cam_bind<double>(state+1221);
cam.pco2=cam_bind<double>(state+1239);
cam.po2=cam_bind<double>(state+1257);
cam.nirwgt=cam_bind<double>(state+1275);
cam.amo=__longlong_as_double(state[1293]);
auto rh_opac=cam_bind<double>(args+0);
auto ksul_opac=cam_bind<double>(args+18);
auto wsul_opac=cam_bind<double>(args+36);
auto gsul_opac=cam_bind<double>(args+54);
auto kssam_opac=cam_bind<double>(args+72);
auto wssam_opac=cam_bind<double>(args+90);
auto gssam_opac=cam_bind<double>(args+108);
auto ksscm_opac=cam_bind<double>(args+126);
auto wsscm_opac=cam_bind<double>(args+144);
auto gsscm_opac=cam_bind<double>(args+162);
auto kcphil_opac=cam_bind<double>(args+180);
auto wcphil_opac=cam_bind<double>(args+198);
auto gcphil_opac=cam_bind<double>(args+216);
cam.aer_optics_initialize(rh_opac,ksul_opac,wsul_opac,gsul_opac,kssam_opac,wssam_opac,gssam_opac,ksscm_opac,wsscm_opac,gsscm_opac,kcphil_opac,wcphil_opac,gcphil_opac);
state[0]=cam.ixcldliq;
state[1]=cam.ixcldice;
state[2]=cam.idxsul;
state[3]=cam.idxsslt;
state[4]=cam.idxdustfirst;
state[5]=cam.idxcarbonfirst;
state[6]=cam.idxocpho;
state[7]=cam.idxbcpho;
state[8]=cam.idxocphi;
state[9]=cam.idxbcphi;
state[10]=cam.idxbg;
state[11]=cam.idxvolc;
state[12]=cam.mxaerl;
state[13]=__double_as_longlong(cam.sulscl_rf);
state[14]=__double_as_longlong(cam.carscl_rf);
state[15]=__double_as_longlong(cam.ssltscl_rf);
state[16]=__double_as_longlong(cam.dustscl_rf);
state[17]=__double_as_longlong(cam.bgscl_rf);
state[18]=__double_as_longlong(cam.volcscl_rf);
state[19]=__double_as_longlong(cam.tauback);
state[20]=__double_as_longlong(cam.sulscl);
state[21]=__double_as_longlong(cam.carscl);
state[22]=__double_as_longlong(cam.ssltscl);
state[23]=__double_as_longlong(cam.dustscl);
state[24]=__double_as_longlong(cam.volcscl);
state[43]=__double_as_longlong(cam.c16);
state[44]=__double_as_longlong(cam.c17);
state[45]=__double_as_longlong(cam.c26);
state[46]=__double_as_longlong(cam.c27);
state[47]=__double_as_longlong(cam.c28);
state[48]=__double_as_longlong(cam.c29);
state[49]=__double_as_longlong(cam.c30);
state[50]=__double_as_longlong(cam.c31);
state[51]=__double_as_longlong(cam.fwcoef);
state[52]=__double_as_longlong(cam.fwc1);
state[53]=__double_as_longlong(cam.fwc2);
state[54]=__double_as_longlong(cam.fc1);
state[55]=__double_as_longlong(cam.amco2);
state[56]=__double_as_longlong(cam.amd);
state[57]=__double_as_longlong(cam.p0);
state[292]=__double_as_longlong(cam.gravit);
state[293]=__double_as_longlong(cam.rga);
state[294]=__double_as_longlong(cam.gravmks);
state[295]=__double_as_longlong(cam.cpair);
state[296]=__double_as_longlong(cam.epsilo);
state[297]=__double_as_longlong(cam.epsqs);
state[298]=__double_as_longlong(cam.sslp);
state[299]=__double_as_longlong(cam.stebol);
state[300]=__double_as_longlong(cam.rgsslp);
state[301]=__double_as_longlong(cam.dpfo3);
state[302]=__double_as_longlong(cam.dpfco2);
state[303]=__double_as_longlong(cam.dayspy);
state[304]=__double_as_longlong(cam.pie);
state[305]=__double_as_longlong(cam.mwdry);
state[306]=__double_as_longlong(cam.scon);
state[307]=__double_as_longlong(cam.co2mmr);
state[308]=__double_as_longlong(cam.mwco2);
state[309]=__double_as_longlong(cam.mwh2o);
state[310]=__double_as_longlong(cam.mwch4);
state[311]=__double_as_longlong(cam.mwn2o);
state[312]=__double_as_longlong(cam.mwf11);
state[313]=__double_as_longlong(cam.mwf12);
state[314]=__double_as_longlong(cam.cappa);
state[315]=__double_as_longlong(cam.rair);
state[316]=__double_as_longlong(cam.tmelt);
state[317]=__double_as_longlong(cam.r_universal);
state[318]=__double_as_longlong(cam.latvap);
state[319]=__double_as_longlong(cam.latice);
state[320]=__double_as_longlong(cam.zvir);
state[339]=__double_as_longlong(cam.tmin);
state[340]=__double_as_longlong(cam.tmax);
state[791]=__double_as_longlong(cam.cplos);
state[792]=__double_as_longlong(cam.cplol);
state[793]=__double_as_longlong(cam.co2vmr);
state[794]=__double_as_longlong(cam.n2ovmr);
state[795]=__double_as_longlong(cam.ch4vmr);
state[796]=__double_as_longlong(cam.f11vmr);
state[797]=__double_as_longlong(cam.f12vmr);
state[834]=cam.ntoplw;
state[835]=cam.masterproc;
state[836]=cam.ozncyc;
state[837]=cam.indirect;
state[838]=cam.radforce;
state[839]=cam.trace_gas;
state[840]=cam.strat_volcanic;
state[1075]=__double_as_longlong(cam.delta);
state[1076]=__double_as_longlong(cam.o2mmr);
state[1293]=__double_as_longlong(cam.amo);
*error=cam.error;}
extern "C" __global__ void cam_oznini(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
cam.ixcldliq=int(state[0]);
cam.ixcldice=int(state[1]);
cam.idxsul=int(state[2]);
cam.idxsslt=int(state[3]);
cam.idxdustfirst=int(state[4]);
cam.idxcarbonfirst=int(state[5]);
cam.idxocpho=int(state[6]);
cam.idxbcpho=int(state[7]);
cam.idxocphi=int(state[8]);
cam.idxbcphi=int(state[9]);
cam.idxbg=int(state[10]);
cam.idxvolc=int(state[11]);
cam.mxaerl=int(state[12]);
cam.sulscl_rf=__longlong_as_double(state[13]);
cam.carscl_rf=__longlong_as_double(state[14]);
cam.ssltscl_rf=__longlong_as_double(state[15]);
cam.dustscl_rf=__longlong_as_double(state[16]);
cam.bgscl_rf=__longlong_as_double(state[17]);
cam.volcscl_rf=__longlong_as_double(state[18]);
cam.tauback=__longlong_as_double(state[19]);
cam.sulscl=__longlong_as_double(state[20]);
cam.carscl=__longlong_as_double(state[21]);
cam.ssltscl=__longlong_as_double(state[22]);
cam.dustscl=__longlong_as_double(state[23]);
cam.volcscl=__longlong_as_double(state[24]);
cam.abs_cff_mss_aer=cam_bind<double>(state+25);
cam.c16=__longlong_as_double(state[43]);
cam.c17=__longlong_as_double(state[44]);
cam.c26=__longlong_as_double(state[45]);
cam.c27=__longlong_as_double(state[46]);
cam.c28=__longlong_as_double(state[47]);
cam.c29=__longlong_as_double(state[48]);
cam.c30=__longlong_as_double(state[49]);
cam.c31=__longlong_as_double(state[50]);
cam.fwcoef=__longlong_as_double(state[51]);
cam.fwc1=__longlong_as_double(state[52]);
cam.fwc2=__longlong_as_double(state[53]);
cam.fc1=__longlong_as_double(state[54]);
cam.amco2=__longlong_as_double(state[55]);
cam.amd=__longlong_as_double(state[56]);
cam.p0=__longlong_as_double(state[57]);
cam.ah2onw=cam_bind<double>(state+58);
cam.eh2onw=cam_bind<double>(state+76);
cam.ah2ow=cam_bind<double>(state+94);
cam.cn_ah2ow=cam_bind<double>(state+112);
cam.cn_eh2ow=cam_bind<double>(state+130);
cam.ln_ah2ow=cam_bind<double>(state+148);
cam.ln_eh2ow=cam_bind<double>(state+166);
cam.coefh=cam_bind<double>(state+184);
cam.coefj=cam_bind<double>(state+202);
cam.coefk=cam_bind<double>(state+220);
cam.estblh2o=cam_bind<double>(state+238);
cam.fat=cam_bind<double>(state+256);
cam.fet=cam_bind<double>(state+274);
cam.gravit=__longlong_as_double(state[292]);
cam.rga=__longlong_as_double(state[293]);
cam.gravmks=__longlong_as_double(state[294]);
cam.cpair=__longlong_as_double(state[295]);
cam.epsilo=__longlong_as_double(state[296]);
cam.epsqs=__longlong_as_double(state[297]);
cam.sslp=__longlong_as_double(state[298]);
cam.stebol=__longlong_as_double(state[299]);
cam.rgsslp=__longlong_as_double(state[300]);
cam.dpfo3=__longlong_as_double(state[301]);
cam.dpfco2=__longlong_as_double(state[302]);
cam.dayspy=__longlong_as_double(state[303]);
cam.pie=__longlong_as_double(state[304]);
cam.mwdry=__longlong_as_double(state[305]);
cam.scon=__longlong_as_double(state[306]);
cam.co2mmr=__longlong_as_double(state[307]);
cam.mwco2=__longlong_as_double(state[308]);
cam.mwh2o=__longlong_as_double(state[309]);
cam.mwch4=__longlong_as_double(state[310]);
cam.mwn2o=__longlong_as_double(state[311]);
cam.mwf11=__longlong_as_double(state[312]);
cam.mwf12=__longlong_as_double(state[313]);
cam.cappa=__longlong_as_double(state[314]);
cam.rair=__longlong_as_double(state[315]);
cam.tmelt=__longlong_as_double(state[316]);
cam.r_universal=__longlong_as_double(state[317]);
cam.latvap=__longlong_as_double(state[318]);
cam.latice=__longlong_as_double(state[319]);
cam.zvir=__longlong_as_double(state[320]);
cam.estbl=cam_bind<double>(state+321);
cam.tmin=__longlong_as_double(state[339]);
cam.tmax=__longlong_as_double(state[340]);
cam.pcf=cam_bind<double>(state+341);
cam.ksul=cam_bind<double>(state+359);
cam.wsul=cam_bind<double>(state+377);
cam.gsul=cam_bind<double>(state+395);
cam.ksslt=cam_bind<double>(state+413);
cam.wsslt=cam_bind<double>(state+431);
cam.gsslt=cam_bind<double>(state+449);
cam.kcphil=cam_bind<double>(state+467);
cam.wcphil=cam_bind<double>(state+485);
cam.gcphil=cam_bind<double>(state+503);
cam.kbg=cam_bind<double>(state+521);
cam.wbg=cam_bind<double>(state+539);
cam.gbg=cam_bind<double>(state+557);
cam.kcphob=cam_bind<double>(state+575);
cam.wcphob=cam_bind<double>(state+593);
cam.gcphob=cam_bind<double>(state+611);
cam.kcb=cam_bind<double>(state+629);
cam.wcb=cam_bind<double>(state+647);
cam.gcb=cam_bind<double>(state+665);
cam.kvolc=cam_bind<double>(state+683);
cam.wvolc=cam_bind<double>(state+701);
cam.gvolc=cam_bind<double>(state+719);
cam.kdst=cam_bind<double>(state+737);
cam.wdst=cam_bind<double>(state+755);
cam.gdst=cam_bind<double>(state+773);
cam.cplos=__longlong_as_double(state[791]);
cam.cplol=__longlong_as_double(state[792]);
cam.co2vmr=__longlong_as_double(state[793]);
cam.n2ovmr=__longlong_as_double(state[794]);
cam.ch4vmr=__longlong_as_double(state[795]);
cam.f11vmr=__longlong_as_double(state[796]);
cam.f12vmr=__longlong_as_double(state[797]);
cam.yrdata=cam_bind<int>(state+798);
cam.co2=cam_bind<double>(state+816);
cam.ntoplw=int(state[834]);
cam.masterproc=(state[835]!=0);
cam.ozncyc=(state[836]!=0);
cam.indirect=(state[837]!=0);
cam.radforce=(state[838]!=0);
cam.trace_gas=(state[839]!=0);
cam.strat_volcanic=(state[840]!=0);
cam.retab=cam_bind<double>(state+841);
cam.abarl=cam_bind<double>(state+859);
cam.bbarl=cam_bind<double>(state+877);
cam.cbarl=cam_bind<double>(state+895);
cam.dbarl=cam_bind<double>(state+913);
cam.ebarl=cam_bind<double>(state+931);
cam.fbarl=cam_bind<double>(state+949);
cam.abari=cam_bind<double>(state+967);
cam.bbari=cam_bind<double>(state+985);
cam.cbari=cam_bind<double>(state+1003);
cam.dbari=cam_bind<double>(state+1021);
cam.ebari=cam_bind<double>(state+1039);
cam.fbari=cam_bind<double>(state+1057);
cam.delta=__longlong_as_double(state[1075]);
cam.o2mmr=__longlong_as_double(state[1076]);
cam.frcsol=cam_bind<double>(state+1077);
cam.wavmin=cam_bind<double>(state+1095);
cam.wavmax=cam_bind<double>(state+1113);
cam.raytau=cam_bind<double>(state+1131);
cam.abh2o=cam_bind<double>(state+1149);
cam.abo3=cam_bind<double>(state+1167);
cam.abco2=cam_bind<double>(state+1185);
cam.abo2=cam_bind<double>(state+1203);
cam.ph2o=cam_bind<double>(state+1221);
cam.pco2=cam_bind<double>(state+1239);
cam.po2=cam_bind<double>(state+1257);
cam.nirwgt=cam_bind<double>(state+1275);
cam.amo=__longlong_as_double(state[1293]);
auto ncells=int(args[0]);
auto num_months=int(args[1]);
auto levsiz=int(args[2]);
auto latcell=cam_bind<float>(args+3);
auto loncell=cam_bind<float>(args+21);
auto pin=cam_bind<float>(args+39);
auto ozmixm=cam_bind<float>(args+57);
auto lat_ozone=cam_bind<float>(args+75);
auto ozmixin=cam_bind<float>(args+93);
auto degrad=__int_as_float(int(args[111]));
cam.oznini(ncells,num_months,levsiz,latcell,loncell,pin,ozmixm,lat_ozone,ozmixin,degrad);
state[0]=cam.ixcldliq;
state[1]=cam.ixcldice;
state[2]=cam.idxsul;
state[3]=cam.idxsslt;
state[4]=cam.idxdustfirst;
state[5]=cam.idxcarbonfirst;
state[6]=cam.idxocpho;
state[7]=cam.idxbcpho;
state[8]=cam.idxocphi;
state[9]=cam.idxbcphi;
state[10]=cam.idxbg;
state[11]=cam.idxvolc;
state[12]=cam.mxaerl;
state[13]=__double_as_longlong(cam.sulscl_rf);
state[14]=__double_as_longlong(cam.carscl_rf);
state[15]=__double_as_longlong(cam.ssltscl_rf);
state[16]=__double_as_longlong(cam.dustscl_rf);
state[17]=__double_as_longlong(cam.bgscl_rf);
state[18]=__double_as_longlong(cam.volcscl_rf);
state[19]=__double_as_longlong(cam.tauback);
state[20]=__double_as_longlong(cam.sulscl);
state[21]=__double_as_longlong(cam.carscl);
state[22]=__double_as_longlong(cam.ssltscl);
state[23]=__double_as_longlong(cam.dustscl);
state[24]=__double_as_longlong(cam.volcscl);
state[43]=__double_as_longlong(cam.c16);
state[44]=__double_as_longlong(cam.c17);
state[45]=__double_as_longlong(cam.c26);
state[46]=__double_as_longlong(cam.c27);
state[47]=__double_as_longlong(cam.c28);
state[48]=__double_as_longlong(cam.c29);
state[49]=__double_as_longlong(cam.c30);
state[50]=__double_as_longlong(cam.c31);
state[51]=__double_as_longlong(cam.fwcoef);
state[52]=__double_as_longlong(cam.fwc1);
state[53]=__double_as_longlong(cam.fwc2);
state[54]=__double_as_longlong(cam.fc1);
state[55]=__double_as_longlong(cam.amco2);
state[56]=__double_as_longlong(cam.amd);
state[57]=__double_as_longlong(cam.p0);
state[292]=__double_as_longlong(cam.gravit);
state[293]=__double_as_longlong(cam.rga);
state[294]=__double_as_longlong(cam.gravmks);
state[295]=__double_as_longlong(cam.cpair);
state[296]=__double_as_longlong(cam.epsilo);
state[297]=__double_as_longlong(cam.epsqs);
state[298]=__double_as_longlong(cam.sslp);
state[299]=__double_as_longlong(cam.stebol);
state[300]=__double_as_longlong(cam.rgsslp);
state[301]=__double_as_longlong(cam.dpfo3);
state[302]=__double_as_longlong(cam.dpfco2);
state[303]=__double_as_longlong(cam.dayspy);
state[304]=__double_as_longlong(cam.pie);
state[305]=__double_as_longlong(cam.mwdry);
state[306]=__double_as_longlong(cam.scon);
state[307]=__double_as_longlong(cam.co2mmr);
state[308]=__double_as_longlong(cam.mwco2);
state[309]=__double_as_longlong(cam.mwh2o);
state[310]=__double_as_longlong(cam.mwch4);
state[311]=__double_as_longlong(cam.mwn2o);
state[312]=__double_as_longlong(cam.mwf11);
state[313]=__double_as_longlong(cam.mwf12);
state[314]=__double_as_longlong(cam.cappa);
state[315]=__double_as_longlong(cam.rair);
state[316]=__double_as_longlong(cam.tmelt);
state[317]=__double_as_longlong(cam.r_universal);
state[318]=__double_as_longlong(cam.latvap);
state[319]=__double_as_longlong(cam.latice);
state[320]=__double_as_longlong(cam.zvir);
state[339]=__double_as_longlong(cam.tmin);
state[340]=__double_as_longlong(cam.tmax);
state[791]=__double_as_longlong(cam.cplos);
state[792]=__double_as_longlong(cam.cplol);
state[793]=__double_as_longlong(cam.co2vmr);
state[794]=__double_as_longlong(cam.n2ovmr);
state[795]=__double_as_longlong(cam.ch4vmr);
state[796]=__double_as_longlong(cam.f11vmr);
state[797]=__double_as_longlong(cam.f12vmr);
state[834]=cam.ntoplw;
state[835]=cam.masterproc;
state[836]=cam.ozncyc;
state[837]=cam.indirect;
state[838]=cam.radforce;
state[839]=cam.trace_gas;
state[840]=cam.strat_volcanic;
state[1075]=__double_as_longlong(cam.delta);
state[1076]=__double_as_longlong(cam.o2mmr);
state[1293]=__double_as_longlong(cam.amo);
args[0]=ncells;
args[1]=num_months;
args[2]=levsiz;
args[111]=__float_as_uint(degrad);
*error=cam.error;}
extern "C" __global__ void cam_radiation_lw_from_mpas(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
auto absnxt_p=cam_bind<float>(args+0);
auto abstot_p=cam_bind<float>(args+18);
auto aerosolcn_p=cam_bind<float>(args+36);
auto aerosolcp_p=cam_bind<float>(args+54);
auto aerosols=cam_bind<float>(args+72);
auto cam_abs_dim1=int(args[90]);
auto cam_abs_dim2=int(args[91]);
auto cemiss_p=cam_bind<float>(args+92);
auto cldfrac=cam_bind<float>(args+110);
auto cldfrac_p=cam_bind<float>(args+128);
auto coszr_p=cam_bind<float>(args+146);
auto degrad=__int_as_float(int(args[164]));
auto emstot_p=cam_bind<float>(args+165);
auto f_ice=cam_bind<float>(args+183);
auto f_rain=cam_bind<float>(args+201);
auto glw_p=cam_bind<float>(args+219);
auto gsw_p=cam_bind<float>(args+237);
auto ite=int(args[255]);
auto its=int(args[256]);
auto jte=int(args[257]);
auto jts=int(args[258]);
auto kte=int(args[259]);
auto kts=int(args[260]);
auto latcell=cam_bind<float>(args+261);
auto loncell=cam_bind<float>(args+279);
auto lwcf_p=cam_bind<float>(args+297);
auto lwdnb_p=cam_bind<float>(args+315);
auto lwdnbc_p=cam_bind<float>(args+333);
auto lwdnt_p=cam_bind<float>(args+351);
auto lwdntc_p=cam_bind<float>(args+369);
auto lwupb_p=cam_bind<float>(args+387);
auto lwupbc_p=cam_bind<float>(args+405);
auto lwupt_p=cam_bind<float>(args+423);
auto lwuptc_p=cam_bind<float>(args+441);
auto m_hybi=cam_bind<float>(args+459);
auto m_hybi_p=cam_bind<float>(args+477);
auto m_ps=cam_bind<float>(args+495);
auto m_psn_p=cam_bind<float>(args+513);
auto m_psp_p=cam_bind<float>(args+531);
auto num_aerlevels=int(args[549]);
auto num_aerosols=int(args[550]);
auto num_months=int(args[551]);
auto num_oznlevels=int(args[552]);
auto olrtoa_p=cam_bind<float>(args+553);
auto ozmixm=cam_bind<float>(args+571);
auto ozmixm_p=cam_bind<float>(args+589);
auto pin=cam_bind<float>(args+607);
auto pin_p=cam_bind<float>(args+625);
auto rthratenlw_p=cam_bind<float>(args+643);
auto rthratensw_p=cam_bind<float>(args+661);
auto sfc_albedo=cam_bind<float>(args+679);
auto sfc_albedo_p=cam_bind<float>(args+697);
auto sfc_emiss=cam_bind<float>(args+715);
auto sfc_emiss_p=cam_bind<float>(args+733);
auto skintemp=cam_bind<float>(args+751);
auto snow=cam_bind<float>(args+769);
auto snow_p=cam_bind<float>(args+787);
auto swcf_p=cam_bind<float>(args+805);
auto swdnb_p=cam_bind<float>(args+823);
auto swdnbc_p=cam_bind<float>(args+841);
auto swdnt_p=cam_bind<float>(args+859);
auto swdntc_p=cam_bind<float>(args+877);
auto swupb_p=cam_bind<float>(args+895);
auto swupbc_p=cam_bind<float>(args+913);
auto swupt_p=cam_bind<float>(args+931);
auto swuptc_p=cam_bind<float>(args+949);
auto taucldc_p=cam_bind<float>(args+967);
auto taucldi_p=cam_bind<float>(args+985);
auto tsk_p=cam_bind<float>(args+1003);
auto xice=cam_bind<float>(args+1021);
auto xice_p=cam_bind<float>(args+1039);
auto xland=cam_bind<float>(args+1057);
auto xland_p=cam_bind<float>(args+1075);
auto xlat_p=cam_bind<float>(args+1093);
auto xlon_p=cam_bind<float>(args+1111);
auto xtime_s=__int_as_float(int(args[1129]));
radiation_lw_from_mpas(absnxt_p,abstot_p,aerosolcn_p,aerosolcp_p,aerosols,cam_abs_dim1,cam_abs_dim2,cemiss_p,cldfrac,cldfrac_p,coszr_p,degrad,emstot_p,f_ice,f_rain,glw_p,gsw_p,ite,its,jte,jts,kte,kts,latcell,loncell,lwcf_p,lwdnb_p,lwdnbc_p,lwdnt_p,lwdntc_p,lwupb_p,lwupbc_p,lwupt_p,lwuptc_p,m_hybi,m_hybi_p,m_ps,m_psn_p,m_psp_p,num_aerlevels,num_aerosols,num_months,num_oznlevels,olrtoa_p,ozmixm,ozmixm_p,pin,pin_p,rthratenlw_p,rthratensw_p,sfc_albedo,sfc_albedo_p,sfc_emiss,sfc_emiss_p,skintemp,snow,snow_p,swcf_p,swdnb_p,swdnbc_p,swdnt_p,swdntc_p,swupb_p,swupbc_p,swupt_p,swuptc_p,taucldc_p,taucldi_p,tsk_p,xice,xice_p,xland,xland_p,xlat_p,xlon_p,xtime_s);
args[90]=cam_abs_dim1;
args[91]=cam_abs_dim2;
args[164]=__float_as_uint(degrad);
args[255]=ite;
args[256]=its;
args[257]=jte;
args[258]=jts;
args[259]=kte;
args[260]=kts;
args[549]=num_aerlevels;
args[550]=num_aerosols;
args[551]=num_months;
args[552]=num_oznlevels;
args[1129]=__float_as_uint(xtime_s);
*error=cam.error;}
extern "C" __global__ void cam_radiation_lw_to_mpas(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
auto glw=cam_bind<float>(args+0);
auto glw_p=cam_bind<float>(args+18);
auto ite=int(args[36]);
auto its=int(args[37]);
auto jte=int(args[38]);
auto jts=int(args[39]);
auto kte=int(args[40]);
auto kts=int(args[41]);
auto lwcf=cam_bind<float>(args+42);
auto lwcf_p=cam_bind<float>(args+60);
auto lwdnb=cam_bind<float>(args+78);
auto lwdnb_p=cam_bind<float>(args+96);
auto lwdnbc=cam_bind<float>(args+114);
auto lwdnbc_p=cam_bind<float>(args+132);
auto lwdnt=cam_bind<float>(args+150);
auto lwdnt_p=cam_bind<float>(args+168);
auto lwdntc=cam_bind<float>(args+186);
auto lwdntc_p=cam_bind<float>(args+204);
auto lwupb=cam_bind<float>(args+222);
auto lwupb_p=cam_bind<float>(args+240);
auto lwupbc=cam_bind<float>(args+258);
auto lwupbc_p=cam_bind<float>(args+276);
auto lwupt=cam_bind<float>(args+294);
auto lwupt_p=cam_bind<float>(args+312);
auto lwuptc=cam_bind<float>(args+330);
auto lwuptc_p=cam_bind<float>(args+348);
auto olrtoa=cam_bind<float>(args+366);
auto olrtoa_p=cam_bind<float>(args+384);
auto rthratenlw=cam_bind<float>(args+402);
auto rthratenlw_p=cam_bind<float>(args+420);
radiation_lw_to_mpas(glw,glw_p,ite,its,jte,jts,kte,kts,lwcf,lwcf_p,lwdnb,lwdnb_p,lwdnbc,lwdnbc_p,lwdnt,lwdnt_p,lwdntc,lwdntc_p,lwupb,lwupb_p,lwupbc,lwupbc_p,lwupt,lwupt_p,lwuptc,lwuptc_p,olrtoa,olrtoa_p,rthratenlw,rthratenlw_p);
args[36]=ite;
args[37]=its;
args[38]=jte;
args[39]=jts;
args[40]=kte;
args[41]=kts;
*error=cam.error;}
extern "C" __global__ void cam_radiation_camlw_to_mpas(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
auto absnxt=cam_bind<float>(args+0);
auto absnxt_p=cam_bind<float>(args+18);
auto abstot=cam_bind<float>(args+36);
auto abstot_p=cam_bind<float>(args+54);
auto cam_abs_dim1=int(args[72]);
auto cam_abs_dim2=int(args[73]);
auto emstot=cam_bind<float>(args+74);
auto emstot_p=cam_bind<float>(args+92);
auto ite=int(args[110]);
auto its=int(args[111]);
auto jte=int(args[112]);
auto jts=int(args[113]);
auto kte=int(args[114]);
auto kts=int(args[115]);
radiation_camlw_to_mpas(absnxt,absnxt_p,abstot,abstot_p,cam_abs_dim1,cam_abs_dim2,emstot,emstot_p,ite,its,jte,jts,kte,kts);
args[72]=cam_abs_dim1;
args[73]=cam_abs_dim2;
args[110]=ite;
args[111]=its;
args[112]=jte;
args[113]=jts;
args[114]=kte;
args[115]=kts;
*error=cam.error;}
extern "C" __global__ void cam_radiation_sw_from_mpas(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
auto absnxt_p=cam_bind<float>(args+0);
auto abstot_p=cam_bind<float>(args+18);
auto aer_opt=int(args[36]);
auto aerosolcn_p=cam_bind<float>(args+37);
auto aerosolcp_p=cam_bind<float>(args+55);
auto aerosols=cam_bind<float>(args+73);
auto cam_abs_dim1=int(args[91]);
auto cam_abs_dim2=int(args[92]);
auto cemiss_p=cam_bind<float>(args+93);
auto cldfrac=cam_bind<float>(args+111);
auto cldfrac_p=cam_bind<float>(args+129);
auto coszr_p=cam_bind<float>(args+147);
auto degrad=__int_as_float(int(args[165]));
auto emstot_p=cam_bind<float>(args+166);
auto f_ice=cam_bind<float>(args+184);
auto f_rain=cam_bind<float>(args+202);
auto glw_p=cam_bind<float>(args+220);
auto gsw_p=cam_bind<float>(args+238);
auto ite=int(args[256]);
auto its=int(args[257]);
auto jte=int(args[258]);
auto jts=int(args[259]);
auto kte=int(args[260]);
auto kts=int(args[261]);
auto latcell=cam_bind<float>(args+262);
auto loncell=cam_bind<float>(args+280);
auto lwcf_p=cam_bind<float>(args+298);
auto lwdnb_p=cam_bind<float>(args+316);
auto lwdnbc_p=cam_bind<float>(args+334);
auto lwdnt_p=cam_bind<float>(args+352);
auto lwdntc_p=cam_bind<float>(args+370);
auto lwupb_p=cam_bind<float>(args+388);
auto lwupbc_p=cam_bind<float>(args+406);
auto lwupt_p=cam_bind<float>(args+424);
auto lwuptc_p=cam_bind<float>(args+442);
auto m_hybi=cam_bind<float>(args+460);
auto m_hybi_p=cam_bind<float>(args+478);
auto m_ps=cam_bind<float>(args+496);
auto m_psn_p=cam_bind<float>(args+514);
auto m_psp_p=cam_bind<float>(args+532);
auto num_aerlevels=int(args[550]);
auto num_aerosols=int(args[551]);
auto num_months=int(args[552]);
auto num_oznlevels=int(args[553]);
auto olrtoa_p=cam_bind<float>(args+554);
auto ozmixm=cam_bind<float>(args+572);
auto ozmixm_p=cam_bind<float>(args+590);
auto pin=cam_bind<float>(args+608);
auto pin_p=cam_bind<float>(args+626);
auto rthratenlw_p=cam_bind<float>(args+644);
auto rthratensw_p=cam_bind<float>(args+662);
auto sfc_albedo=cam_bind<float>(args+680);
auto sfc_albedo_p=cam_bind<float>(args+698);
auto sfc_emiss=cam_bind<float>(args+716);
auto sfc_emiss_p=cam_bind<float>(args+734);
auto skintemp=cam_bind<float>(args+752);
auto snow=cam_bind<float>(args+770);
auto snow_p=cam_bind<float>(args+788);
auto swcf_p=cam_bind<float>(args+806);
auto swdnb_p=cam_bind<float>(args+824);
auto swdnbc_p=cam_bind<float>(args+842);
auto swdnt_p=cam_bind<float>(args+860);
auto swdntc_p=cam_bind<float>(args+878);
auto swupb_p=cam_bind<float>(args+896);
auto swupbc_p=cam_bind<float>(args+914);
auto swupt_p=cam_bind<float>(args+932);
auto swuptc_p=cam_bind<float>(args+950);
auto taucldc_p=cam_bind<float>(args+968);
auto taucldi_p=cam_bind<float>(args+986);
auto tsk_p=cam_bind<float>(args+1004);
auto xice=cam_bind<float>(args+1022);
auto xice_p=cam_bind<float>(args+1040);
auto xland=cam_bind<float>(args+1058);
auto xland_p=cam_bind<float>(args+1076);
auto xlat_p=cam_bind<float>(args+1094);
auto xlon_p=cam_bind<float>(args+1112);
auto xtime_s=__int_as_float(int(args[1130]));
radiation_sw_from_mpas(absnxt_p,abstot_p,aer_opt,aerosolcn_p,aerosolcp_p,aerosols,cam_abs_dim1,cam_abs_dim2,cemiss_p,cldfrac,cldfrac_p,coszr_p,degrad,emstot_p,f_ice,f_rain,glw_p,gsw_p,ite,its,jte,jts,kte,kts,latcell,loncell,lwcf_p,lwdnb_p,lwdnbc_p,lwdnt_p,lwdntc_p,lwupb_p,lwupbc_p,lwupt_p,lwuptc_p,m_hybi,m_hybi_p,m_ps,m_psn_p,m_psp_p,num_aerlevels,num_aerosols,num_months,num_oznlevels,olrtoa_p,ozmixm,ozmixm_p,pin,pin_p,rthratenlw_p,rthratensw_p,sfc_albedo,sfc_albedo_p,sfc_emiss,sfc_emiss_p,skintemp,snow,snow_p,swcf_p,swdnb_p,swdnbc_p,swdnt_p,swdntc_p,swupb_p,swupbc_p,swupt_p,swuptc_p,taucldc_p,taucldi_p,tsk_p,xice,xice_p,xland,xland_p,xlat_p,xlon_p,xtime_s);
args[36]=aer_opt;
args[91]=cam_abs_dim1;
args[92]=cam_abs_dim2;
args[165]=__float_as_uint(degrad);
args[256]=ite;
args[257]=its;
args[258]=jte;
args[259]=jts;
args[260]=kte;
args[261]=kts;
args[550]=num_aerlevels;
args[551]=num_aerosols;
args[552]=num_months;
args[553]=num_oznlevels;
args[1130]=__float_as_uint(xtime_s);
*error=cam.error;}
extern "C" __global__ void cam_radiation_sw_to_mpas(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
auto coszr=cam_bind<float>(args+0);
auto coszr_p=cam_bind<float>(args+18);
auto gsw=cam_bind<float>(args+36);
auto gsw_p=cam_bind<float>(args+54);
auto ite=int(args[72]);
auto its=int(args[73]);
auto jte=int(args[74]);
auto jts=int(args[75]);
auto kte=int(args[76]);
auto kts=int(args[77]);
auto rthratensw=cam_bind<float>(args+78);
auto rthratensw_p=cam_bind<float>(args+96);
auto swcf=cam_bind<float>(args+114);
auto swcf_p=cam_bind<float>(args+132);
auto swdnb=cam_bind<float>(args+150);
auto swdnb_p=cam_bind<float>(args+168);
auto swdnbc=cam_bind<float>(args+186);
auto swdnbc_p=cam_bind<float>(args+204);
auto swdnt=cam_bind<float>(args+222);
auto swdnt_p=cam_bind<float>(args+240);
auto swdntc=cam_bind<float>(args+258);
auto swdntc_p=cam_bind<float>(args+276);
auto swupb=cam_bind<float>(args+294);
auto swupb_p=cam_bind<float>(args+312);
auto swupbc=cam_bind<float>(args+330);
auto swupbc_p=cam_bind<float>(args+348);
auto swupt=cam_bind<float>(args+366);
auto swupt_p=cam_bind<float>(args+384);
auto swuptc=cam_bind<float>(args+402);
auto swuptc_p=cam_bind<float>(args+420);
radiation_sw_to_mpas(coszr,coszr_p,gsw,gsw_p,ite,its,jte,jts,kte,kts,rthratensw,rthratensw_p,swcf,swcf_p,swdnb,swdnb_p,swdnbc,swdnbc_p,swdnt,swdnt_p,swdntc,swdntc_p,swupb,swupb_p,swupbc,swupbc_p,swupt,swupt_p,swuptc,swuptc_p);
args[72]=ite;
args[73]=its;
args[74]=jte;
args[75]=jts;
args[76]=kte;
args[77]=kts;
*error=cam.error;}
extern "C" __global__ void cam_radconst(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
auto declin=__int_as_float(int(args[0]));
auto degrad=__int_as_float(int(args[1]));
auto dpd=__int_as_float(int(args[2]));
auto julian=__int_as_float(int(args[3]));
auto solcon=__int_as_float(int(args[4]));
auto solcon_0=__int_as_float(int(args[5]));
radconst(declin,degrad,dpd,julian,solcon,solcon_0);
args[0]=__float_as_uint(declin);
args[1]=__float_as_uint(degrad);
args[2]=__float_as_uint(dpd);
args[3]=__float_as_uint(julian);
args[4]=__float_as_uint(solcon);
args[5]=__float_as_uint(solcon_0);
*error=cam.error;}
extern "C" __global__ void cam_driver_clock(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
auto xtime_s=__int_as_float(int(args[0]));
auto dt_radt=__int_as_float(int(args[1]));
auto xtime_m=__int_as_float(int(args[2]));
auto radt=__int_as_float(int(args[3]));
driver_clock(xtime_s,dt_radt,xtime_m,radt);
args[0]=__float_as_uint(xtime_s);
args[1]=__float_as_uint(dt_radt);
args[2]=__float_as_uint(xtime_m);
args[3]=__float_as_uint(radt);
*error=cam.error;}
extern "C" __global__ void cam_initialize_defaults(unsigned long long* args,unsigned long long* state,char* workspace,unsigned long long bytes,int* error) {
if(blockIdx.x || threadIdx.x) return;CamArena arena(workspace,bytes);CamColumn cam(arena);
cam.ixcldliq=int(state[0]);
cam.ixcldice=int(state[1]);
cam.idxsul=int(state[2]);
cam.idxsslt=int(state[3]);
cam.idxdustfirst=int(state[4]);
cam.idxcarbonfirst=int(state[5]);
cam.idxocpho=int(state[6]);
cam.idxbcpho=int(state[7]);
cam.idxocphi=int(state[8]);
cam.idxbcphi=int(state[9]);
cam.idxbg=int(state[10]);
cam.idxvolc=int(state[11]);
cam.mxaerl=int(state[12]);
cam.sulscl_rf=__longlong_as_double(state[13]);
cam.carscl_rf=__longlong_as_double(state[14]);
cam.ssltscl_rf=__longlong_as_double(state[15]);
cam.dustscl_rf=__longlong_as_double(state[16]);
cam.bgscl_rf=__longlong_as_double(state[17]);
cam.volcscl_rf=__longlong_as_double(state[18]);
cam.tauback=__longlong_as_double(state[19]);
cam.sulscl=__longlong_as_double(state[20]);
cam.carscl=__longlong_as_double(state[21]);
cam.ssltscl=__longlong_as_double(state[22]);
cam.dustscl=__longlong_as_double(state[23]);
cam.volcscl=__longlong_as_double(state[24]);
cam.abs_cff_mss_aer=cam_bind<double>(state+25);
cam.c16=__longlong_as_double(state[43]);
cam.c17=__longlong_as_double(state[44]);
cam.c26=__longlong_as_double(state[45]);
cam.c27=__longlong_as_double(state[46]);
cam.c28=__longlong_as_double(state[47]);
cam.c29=__longlong_as_double(state[48]);
cam.c30=__longlong_as_double(state[49]);
cam.c31=__longlong_as_double(state[50]);
cam.fwcoef=__longlong_as_double(state[51]);
cam.fwc1=__longlong_as_double(state[52]);
cam.fwc2=__longlong_as_double(state[53]);
cam.fc1=__longlong_as_double(state[54]);
cam.amco2=__longlong_as_double(state[55]);
cam.amd=__longlong_as_double(state[56]);
cam.p0=__longlong_as_double(state[57]);
cam.ah2onw=cam_bind<double>(state+58);
cam.eh2onw=cam_bind<double>(state+76);
cam.ah2ow=cam_bind<double>(state+94);
cam.cn_ah2ow=cam_bind<double>(state+112);
cam.cn_eh2ow=cam_bind<double>(state+130);
cam.ln_ah2ow=cam_bind<double>(state+148);
cam.ln_eh2ow=cam_bind<double>(state+166);
cam.coefh=cam_bind<double>(state+184);
cam.coefj=cam_bind<double>(state+202);
cam.coefk=cam_bind<double>(state+220);
cam.estblh2o=cam_bind<double>(state+238);
cam.fat=cam_bind<double>(state+256);
cam.fet=cam_bind<double>(state+274);
cam.gravit=__longlong_as_double(state[292]);
cam.rga=__longlong_as_double(state[293]);
cam.gravmks=__longlong_as_double(state[294]);
cam.cpair=__longlong_as_double(state[295]);
cam.epsilo=__longlong_as_double(state[296]);
cam.epsqs=__longlong_as_double(state[297]);
cam.sslp=__longlong_as_double(state[298]);
cam.stebol=__longlong_as_double(state[299]);
cam.rgsslp=__longlong_as_double(state[300]);
cam.dpfo3=__longlong_as_double(state[301]);
cam.dpfco2=__longlong_as_double(state[302]);
cam.dayspy=__longlong_as_double(state[303]);
cam.pie=__longlong_as_double(state[304]);
cam.mwdry=__longlong_as_double(state[305]);
cam.scon=__longlong_as_double(state[306]);
cam.co2mmr=__longlong_as_double(state[307]);
cam.mwco2=__longlong_as_double(state[308]);
cam.mwh2o=__longlong_as_double(state[309]);
cam.mwch4=__longlong_as_double(state[310]);
cam.mwn2o=__longlong_as_double(state[311]);
cam.mwf11=__longlong_as_double(state[312]);
cam.mwf12=__longlong_as_double(state[313]);
cam.cappa=__longlong_as_double(state[314]);
cam.rair=__longlong_as_double(state[315]);
cam.tmelt=__longlong_as_double(state[316]);
cam.r_universal=__longlong_as_double(state[317]);
cam.latvap=__longlong_as_double(state[318]);
cam.latice=__longlong_as_double(state[319]);
cam.zvir=__longlong_as_double(state[320]);
cam.estbl=cam_bind<double>(state+321);
cam.tmin=__longlong_as_double(state[339]);
cam.tmax=__longlong_as_double(state[340]);
cam.pcf=cam_bind<double>(state+341);
cam.ksul=cam_bind<double>(state+359);
cam.wsul=cam_bind<double>(state+377);
cam.gsul=cam_bind<double>(state+395);
cam.ksslt=cam_bind<double>(state+413);
cam.wsslt=cam_bind<double>(state+431);
cam.gsslt=cam_bind<double>(state+449);
cam.kcphil=cam_bind<double>(state+467);
cam.wcphil=cam_bind<double>(state+485);
cam.gcphil=cam_bind<double>(state+503);
cam.kbg=cam_bind<double>(state+521);
cam.wbg=cam_bind<double>(state+539);
cam.gbg=cam_bind<double>(state+557);
cam.kcphob=cam_bind<double>(state+575);
cam.wcphob=cam_bind<double>(state+593);
cam.gcphob=cam_bind<double>(state+611);
cam.kcb=cam_bind<double>(state+629);
cam.wcb=cam_bind<double>(state+647);
cam.gcb=cam_bind<double>(state+665);
cam.kvolc=cam_bind<double>(state+683);
cam.wvolc=cam_bind<double>(state+701);
cam.gvolc=cam_bind<double>(state+719);
cam.kdst=cam_bind<double>(state+737);
cam.wdst=cam_bind<double>(state+755);
cam.gdst=cam_bind<double>(state+773);
cam.cplos=__longlong_as_double(state[791]);
cam.cplol=__longlong_as_double(state[792]);
cam.co2vmr=__longlong_as_double(state[793]);
cam.n2ovmr=__longlong_as_double(state[794]);
cam.ch4vmr=__longlong_as_double(state[795]);
cam.f11vmr=__longlong_as_double(state[796]);
cam.f12vmr=__longlong_as_double(state[797]);
cam.yrdata=cam_bind<int>(state+798);
cam.co2=cam_bind<double>(state+816);
cam.ntoplw=int(state[834]);
cam.masterproc=(state[835]!=0);
cam.ozncyc=(state[836]!=0);
cam.indirect=(state[837]!=0);
cam.radforce=(state[838]!=0);
cam.trace_gas=(state[839]!=0);
cam.strat_volcanic=(state[840]!=0);
cam.retab=cam_bind<double>(state+841);
cam.abarl=cam_bind<double>(state+859);
cam.bbarl=cam_bind<double>(state+877);
cam.cbarl=cam_bind<double>(state+895);
cam.dbarl=cam_bind<double>(state+913);
cam.ebarl=cam_bind<double>(state+931);
cam.fbarl=cam_bind<double>(state+949);
cam.abari=cam_bind<double>(state+967);
cam.bbari=cam_bind<double>(state+985);
cam.cbari=cam_bind<double>(state+1003);
cam.dbari=cam_bind<double>(state+1021);
cam.ebari=cam_bind<double>(state+1039);
cam.fbari=cam_bind<double>(state+1057);
cam.delta=__longlong_as_double(state[1075]);
cam.o2mmr=__longlong_as_double(state[1076]);
cam.frcsol=cam_bind<double>(state+1077);
cam.wavmin=cam_bind<double>(state+1095);
cam.wavmax=cam_bind<double>(state+1113);
cam.raytau=cam_bind<double>(state+1131);
cam.abh2o=cam_bind<double>(state+1149);
cam.abo3=cam_bind<double>(state+1167);
cam.abco2=cam_bind<double>(state+1185);
cam.abo2=cam_bind<double>(state+1203);
cam.ph2o=cam_bind<double>(state+1221);
cam.pco2=cam_bind<double>(state+1239);
cam.po2=cam_bind<double>(state+1257);
cam.nirwgt=cam_bind<double>(state+1275);
cam.amo=__longlong_as_double(state[1293]);
cam.initialize_defaults();
state[0]=cam.ixcldliq;
state[1]=cam.ixcldice;
state[2]=cam.idxsul;
state[3]=cam.idxsslt;
state[4]=cam.idxdustfirst;
state[5]=cam.idxcarbonfirst;
state[6]=cam.idxocpho;
state[7]=cam.idxbcpho;
state[8]=cam.idxocphi;
state[9]=cam.idxbcphi;
state[10]=cam.idxbg;
state[11]=cam.idxvolc;
state[12]=cam.mxaerl;
state[13]=__double_as_longlong(cam.sulscl_rf);
state[14]=__double_as_longlong(cam.carscl_rf);
state[15]=__double_as_longlong(cam.ssltscl_rf);
state[16]=__double_as_longlong(cam.dustscl_rf);
state[17]=__double_as_longlong(cam.bgscl_rf);
state[18]=__double_as_longlong(cam.volcscl_rf);
state[19]=__double_as_longlong(cam.tauback);
state[20]=__double_as_longlong(cam.sulscl);
state[21]=__double_as_longlong(cam.carscl);
state[22]=__double_as_longlong(cam.ssltscl);
state[23]=__double_as_longlong(cam.dustscl);
state[24]=__double_as_longlong(cam.volcscl);
state[43]=__double_as_longlong(cam.c16);
state[44]=__double_as_longlong(cam.c17);
state[45]=__double_as_longlong(cam.c26);
state[46]=__double_as_longlong(cam.c27);
state[47]=__double_as_longlong(cam.c28);
state[48]=__double_as_longlong(cam.c29);
state[49]=__double_as_longlong(cam.c30);
state[50]=__double_as_longlong(cam.c31);
state[51]=__double_as_longlong(cam.fwcoef);
state[52]=__double_as_longlong(cam.fwc1);
state[53]=__double_as_longlong(cam.fwc2);
state[54]=__double_as_longlong(cam.fc1);
state[55]=__double_as_longlong(cam.amco2);
state[56]=__double_as_longlong(cam.amd);
state[57]=__double_as_longlong(cam.p0);
state[292]=__double_as_longlong(cam.gravit);
state[293]=__double_as_longlong(cam.rga);
state[294]=__double_as_longlong(cam.gravmks);
state[295]=__double_as_longlong(cam.cpair);
state[296]=__double_as_longlong(cam.epsilo);
state[297]=__double_as_longlong(cam.epsqs);
state[298]=__double_as_longlong(cam.sslp);
state[299]=__double_as_longlong(cam.stebol);
state[300]=__double_as_longlong(cam.rgsslp);
state[301]=__double_as_longlong(cam.dpfo3);
state[302]=__double_as_longlong(cam.dpfco2);
state[303]=__double_as_longlong(cam.dayspy);
state[304]=__double_as_longlong(cam.pie);
state[305]=__double_as_longlong(cam.mwdry);
state[306]=__double_as_longlong(cam.scon);
state[307]=__double_as_longlong(cam.co2mmr);
state[308]=__double_as_longlong(cam.mwco2);
state[309]=__double_as_longlong(cam.mwh2o);
state[310]=__double_as_longlong(cam.mwch4);
state[311]=__double_as_longlong(cam.mwn2o);
state[312]=__double_as_longlong(cam.mwf11);
state[313]=__double_as_longlong(cam.mwf12);
state[314]=__double_as_longlong(cam.cappa);
state[315]=__double_as_longlong(cam.rair);
state[316]=__double_as_longlong(cam.tmelt);
state[317]=__double_as_longlong(cam.r_universal);
state[318]=__double_as_longlong(cam.latvap);
state[319]=__double_as_longlong(cam.latice);
state[320]=__double_as_longlong(cam.zvir);
state[339]=__double_as_longlong(cam.tmin);
state[340]=__double_as_longlong(cam.tmax);
state[791]=__double_as_longlong(cam.cplos);
state[792]=__double_as_longlong(cam.cplol);
state[793]=__double_as_longlong(cam.co2vmr);
state[794]=__double_as_longlong(cam.n2ovmr);
state[795]=__double_as_longlong(cam.ch4vmr);
state[796]=__double_as_longlong(cam.f11vmr);
state[797]=__double_as_longlong(cam.f12vmr);
state[834]=cam.ntoplw;
state[835]=cam.masterproc;
state[836]=cam.ozncyc;
state[837]=cam.indirect;
state[838]=cam.radforce;
state[839]=cam.trace_gas;
state[840]=cam.strat_volcanic;
state[1075]=__double_as_longlong(cam.delta);
state[1076]=__double_as_longlong(cam.o2mmr);
state[1293]=__double_as_longlong(cam.amo);
*error=cam.error;}



"""


from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

COMPILE_OPTIONS = ('--std=c++17', '--fmad=false', '--ftz=false',
                   '--prec-div=true', '--prec-sqrt=true')
ACCEPTANCE = 'unqualified: standalone host witness only'

class RustTables:
    """Owned raw buffers decoded exclusively by rw-mpas's table bridge."""
    def __init__(self, library, directory, *, ozone_levels, months):
        import ctypes as ct
        self.ct=ct
        class View(ct.Structure):
            _fields_=[('name',ct.c_char*32),('kind',ct.c_uint32),('rank',ct.c_uint32),
                      ('lower',ct.c_int32*5),('extent',ct.c_uint64*5),
                      ('stride',ct.c_uint64*5),('count',ct.c_uint64),('data',ct.c_void_p)]
        self.View=View;self.library=ct.CDLL(str(library));lib=self.library
        lib.rw_mpas_cam_tables_abi.restype=ct.c_uint32
        if lib.rw_mpas_cam_tables_abi()!=1:raise RuntimeError('rw-mpas CAM table ABI differs')
        lib.rw_mpas_cam_tables_open.argtypes=[ct.c_char_p,ct.c_size_t,ct.c_size_t,ct.c_void_p,ct.c_size_t]
        lib.rw_mpas_cam_tables_open.restype=ct.c_void_p
        lib.rw_mpas_cam_tables_count.argtypes=[ct.c_void_p];lib.rw_mpas_cam_tables_count.restype=ct.c_size_t
        lib.rw_mpas_cam_tables_view.argtypes=[ct.c_void_p,ct.c_size_t,ct.POINTER(View)]
        lib.rw_mpas_cam_tables_view.restype=ct.c_int32
        lib.rw_mpas_cam_tables_free.argtypes=[ct.c_void_p]
        error=ct.create_string_buffer(2048)
        self.handle=lib.rw_mpas_cam_tables_open(str(directory).encode(),ozone_levels,months,error,len(error))
        if not self.handle:raise RuntimeError(error.value.decode())

    def host_views(self):
        """Borrow typed Rust buffers; Python performs no file or number decode."""
        import numpy as np
        if not self.handle:raise RuntimeError('CAM Rust table owner is closed')
        result={};ct=self.ct
        for index in range(self.library.rw_mpas_cam_tables_count(self.handle)):
            view=self.View()
            if self.library.rw_mpas_cam_tables_view(self.handle,index,ct.byref(view)):
                raise RuntimeError('rw-mpas returned an invalid table descriptor')
            typ={4:ct.c_float,8:ct.c_double}[view.kind]
            pointer=ct.cast(view.data,ct.POINTER(typ))
            array=np.ctypeslib.as_array(pointer,shape=(view.count,))
            result[view.name.decode()]=(array,tuple(view.lower[:view.rank]),
                                      tuple(view.extent[:view.rank]),tuple(view.stride[:view.rank]))
        return result

    def upload(self,device):
        import cupy as cp
        with cp.cuda.Device(device):
            arrays={n:DeviceArray(cp.asarray(a),lo,ex,st) for n,(a,lo,ex,st) in self.host_views().items()}
            cp.cuda.get_current_stream().synchronize()
        return arrays

    def close(self):
        if self.handle:self.library.rw_mpas_cam_tables_free(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*exception):self.close()

@dataclass(frozen=True)
class DeviceScalar:
    """A scalar output word retained on device between driver stages."""
    storage: object
    offset: int
    kind: str

@dataclass(frozen=True)
class DeviceArray:
    """Device storage with explicit Fortran bounds/strides, in elements.

    Rust/framework providers own preparation and decoded tables. This wrapper
    carries their descriptor; it never decodes or interpolates physical data.
    """
    storage: object
    lower: tuple[int, ...]
    extent: tuple[int, ...]
    stride: tuple[int, ...]

@dataclass(frozen=True)
class BoundaryResult:
    metadata: object
    error: object
    workspace: object
    doabsems_offset: int
    global_metadata: object
    fields: object

@dataclass(frozen=True)
class DeviceState:
    """Persistent device constants and tables, including scalar initialization."""
    metadata: object
    values: object
    device: int

def make_device_state(values: Mapping[str, object], device: int) -> DeviceState:
    return DeviceState(_pack(STATE_FIELDS,values,device),values,device)

def allocate_state(device: int) -> DeviceState:
    """Allocate MPAS module dimensions; defaults are evaluated on the device."""
    import cupy as cp
    with cp.cuda.Device(device):
        values={}
        for name,kind,rank in STATE_FIELDS:
            if rank:
                lower,extent,stride=STATE_LAYOUT[name]
                count=1
                for n in extent:count*=n
                a=cp.empty(count,dtype={'float':cp.float32,'double':cp.float64,
                                       'int':cp.int32,'bool':cp.bool_}[kind])
                values[name]=DeviceArray(a,lower,extent,stride)
            else:values[name]=False if kind=='bool' else 0
        return make_device_state(values,device)

def initialize(state: DeviceState, tables, *, constants, aerosols, ozone, reduce_max):
    """Complete mpas_atmphys_camrad_init.F numeric initialization.

    Geometry/pool arguments arrive from Rust/framework; reduce_max performs
    native mpas_dmpar_max_int. Raw pressure stays in hPa until oznini's own
    float multiplication. Numeric interpolation executes in CUDA.
    """
    import cupy as cp
    results=[]
    def stage(name,arguments):
        result=launch_stage(name,arguments,state=state)
        code=int(result.error.get()[0])
        if code:raise RuntimeError(f'MPAS CAM initialization fatal source line {code}')
        results.append(result);return result
    with cp.cuda.Device(state.device):
        stage('initialize_defaults',{})
        for name in ('ah2onw','eh2onw','ah2ow','cn_ah2ow','cn_eh2ow','ln_ah2ow','ln_eh2ow',
                     'kcb','wcb','gcb','kdst','wdst','gdst','kbg','wbg','gbg','kvolc','wvolc','gvolc'):
            source=tables[name];destination=state.values[name]
            if (source.lower,source.extent,source.stride)!=(destination.lower,destination.extent,destination.stride):
                raise ValueError(f'{name} decoded table dimensions differ from MPAS')
            destination.storage[:]=source.storage
        stage('camradinit',constants)
        stage('aer_optics_initialize',{name:tables[name] for name,_,_ in STAGE_FIELDS['aer_optics_initialize']})
        result=stage('aerosol_init',aerosols)
        maximum=stage_scalar(result,'aerosol_init','max_mxaerl')
        reduced=reduce_max(int(maximum.storage[maximum.offset].get()))
        state.metadata[STATE_SCALARS['mxaerl'][0]]=reduced
        ozone=dict(ozone)
        ozone['pin'].storage[:]=tables['pin'].storage
        ozone.update(lat_ozone=tables['lat_ozone'],ozmixin=tables['ozmixin'])
        stage('oznini',ozone)
    return tuple(results)

def _state_metadata(state,device):
    if isinstance(state,DeviceState):
        if state.device!=device: raise ValueError('CAM state belongs to a different device')
        return state.metadata
    return _pack(STATE_FIELDS,state,device)

def _pack(fields, values, device):
    import cupy as cp
    import numpy as np
    if set(values) != {name for name, _, _ in fields}:
        raise ValueError('CAM ABI requires every named field in its source interface')
    packed = []
    scalar_links=[]
    for name, kind, rank in fields:
        value = values[name]
        if rank:
            if value is None and fields is ARGUMENT_FIELDS and name in OPTIONAL_ARGUMENTS:
                # MPAS module_ra_cam.F:707-726: absent optional flux descriptors.
                packed.extend((0,rank,0)+ (1,)*5 + (0,)*10)
                continue
            if not isinstance(value, DeviceArray):
                raise TypeError(f'{name} requires an explicit device Fortran descriptor')
            a = value.storage
            dtype = {'float': np.float32, 'double': np.float64, 'int': np.int32, 'bool': np.bool_}[kind]
            if not isinstance(a, cp.ndarray) or a.dtype != dtype or a.device.id != device:
                raise TypeError(f'{name} has the wrong device or MPAS kind')
            if not (a.flags.c_contiguous or a.flags.f_contiguous):
                raise ValueError(f'{name} storage must be contiguous: descriptor strides own addressing')
            if not (len(value.lower) == len(value.extent) == len(value.stride) == rank):
                raise ValueError(f'{name} descriptor rank differs from MPAS')
            if any(n <= 0 for n in value.extent) or any(s < 0 for s in value.stride):
                raise ValueError(f'{name} requires positive extents and nonnegative strides')
            high = sum((n-1)*s for n,s in zip(value.extent,value.stride))
            if (high+1)*a.itemsize > a.nbytes:
                raise ValueError(f'{name} descriptor exceeds its device allocation')
            packed.extend((a.data.ptr, rank, int(np.prod(value.extent))))
            for sequence in (value.lower,value.extent,value.stride):
                packed.extend(int(n) & ((1 << 64)-1) for n in sequence)
                packed.extend([0]*(5-rank))
        elif isinstance(value,DeviceScalar):
            if (value.kind!=kind or not isinstance(value.storage,cp.ndarray) or
                    value.storage.dtype!=cp.uint64 or value.storage.device.id!=device or
                    not 0<=value.offset<value.storage.size):
                raise TypeError(f'{name} scalar link differs from the native kind/device')
            scalar_links.append((len(packed),value));packed.append(0)
        elif kind == 'double':
            packed.append(int(np.float64(value).view(np.uint64)))
        elif kind == 'float':
            packed.append(int(np.float32(value).view(np.uint32)))
        else:
            packed.append(int(value) & ((1 << 64)-1))
    metadata=cp.asarray(packed,dtype=cp.uint64)
    for offset,value in scalar_links:
        metadata[offset:offset+1]=value.storage[value.offset:value.offset+1]
    return metadata

@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from woof.core import kernels
    directory = Path(kernels.__file__).parent
    with cp.cuda.Device(device):
        import os
        artifact=os.environ.get('HEX_CAM_CUDA_ARTIFACT')
        if artifact:
            import hashlib
            import json
            path=Path(artifact);record=json.loads(path.with_suffix('.json').read_text())
            required=_hex_numerics.options(COMPILE_OPTIONS)+('-I'+str(directory),'--gpu-architecture=' + _image_target(device, xp=cp))
            if (record['source_sha256']!=hashlib.sha256(CUDA_SOURCE.encode()).hexdigest() or
                    record['artifact_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest() or
                    tuple(record['options'])!=required or any(
                        hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest
                        for name,digest in record['shared_headers'].items())):
                raise RuntimeError('CAM CUDA artifact differs from source, strict flags or shared math')
            module=cp.cuda.function.Module()
            module.load(path.read_bytes())
            return module
        # RawModule's cache compiler appends --ftz=true, overriding the
        # requested native subnormal semantics (p5's strict witness).
        # Invoke NVRTC directly and load its binary without that override.
        from cupy.cuda import compiler
        binary,_=compiler.compile_using_nvrtc(
            CUDA_SOURCE,options=_hex_numerics.options(COMPILE_OPTIONS)+('-I'+str(directory),),
            arch=_compilation_target(device, xp=cp),cache_in_memory=True)
        module=cp.cuda.function.Module()
        module.load(binary)
        return module

def workspace_bytes(arguments: Mapping[str, object]) -> int:
    """Storage upper bound for the complete nested CAM call stack."""
    nc = min(1024, int(arguments['ite']) - int(arguments['its']) + 1)
    nv = int(arguments['kte']) - int(arguments['kts']) + 1
    no, nm = int(arguments['levsiz']), int(arguments['num_months'])
    na, ns = int(arguments['paerlev']), int(arguments['naer_c'])
    return nc*(4096 + 256*(nv+1)**2 + 8192*(nv+1) + 512*no*nm + 128*na*ns)

def launch_boundary(arguments: Mapping[str, object], state: Mapping[str, object], *, workspace=None, local_state=None, utility_state=None):
    """Complete module_ra_cam.F:179-823 native boundary, in place.

    All single-precision fields, flags and persistent absorption caches are
    supplied by the driver. State carries initialized tables/constants.
    This entry is column test tooling until driver/forecast qualification.
    """
    import cupy as cp
    import numpy as np
    first = arguments['rthratenlw']
    if not isinstance(first,DeviceArray) or not isinstance(first.storage,cp.ndarray):
        raise TypeError('rthratenlw must supply a device descriptor')
    device=first.storage.device.id
    with cp.cuda.Device(device):
        meta=_pack(ARGUMENT_FIELDS,arguments,device)
        global_meta=_state_metadata(state,device)
        needed=workspace_bytes(arguments)
        if workspace is None:
            workspace=cp.zeros(needed,dtype=cp.uint8)
        if (not isinstance(workspace,cp.ndarray) or workspace.dtype!=cp.uint8 or
                workspace.device.id!=device or not workspace.flags.c_contiguous or workspace.nbytes<needed):
            raise ValueError('CAM workspace does not meet the complete call-stack bound')
        error=cp.zeros(1,dtype=cp.int32)
        columns=int(arguments['ite'])-int(arguments['its'])+1
        module=_module(device)
        if utility_state and 'mpas_err_message' in utility_state:
            message=utility_state['mpas_err_message']
            if (not isinstance(message,cp.ndarray) or message.dtype!=cp.uint8
                    or message.nbytes!=512 or message.device.id!=device
                    or not message.flags.c_contiguous):
                raise ValueError('Native CHARACTER message requires its complete 512-byte device buffer')
            module.get_function('cam_restart_message')((1,),(1,),(meta,message))
        module.get_function('cam_prepare_cache')((1,),(1,),(meta,))
        local_meta=cp.zeros(18*len(ORACLE_LOCAL_FIELDS),dtype=cp.uint64)
        if local_state:
            for index,(name,kind,rank) in enumerate(ORACLE_LOCAL_FIELDS):
                if name in local_state:
                    descriptor=_pack(((name,kind,rank or 1),),{name:local_state[name]},device)
                    local_meta[index*18:(index+1)*18]=descriptor
        kernel=module.get_function('cam_boundary')
        for start in range(0,columns,1024):
            kernel(((min(1024,columns-start)+31)//32,),(32,),
                   (meta,global_meta,workspace,
                    # MPAS module_ra_cam.F:364-365,678-679: preserve arena addresses.
                    np.uint64(((workspace.nbytes//min(1024,columns))&~7)*min(1024,columns-start)),
                    error,np.int32(start),local_meta))
    return BoundaryResult(meta,error,workspace,DOABSEMS_OFFSET,global_meta,(arguments,state,local_meta,local_state,utility_state))

def launch_stage(name: str, arguments: Mapping[str, object], *, state=None, device=None, workspace=None):
    """MPAS numeric initialization/driver slice, with every output in place.

    Initialization stages require a persistent DeviceState. The framework must
    reduce aerosol_init's max_mxaerl across ranks and update the mxaerl state
    word before radiation. Table decoding and pool retrieval belong to Rust.
    """
    import cupy as cp
    import numpy as np
    fields=STAGE_FIELDS[name]
    if device is None:
        device=next((v.storage.device.id for v in arguments.values() if isinstance(v,DeviceArray)),
                    state.device if isinstance(state,DeviceState) else None)
    if device is None: raise ValueError('scalar stages require an explicit device')
    if name in INIT_STAGES and not isinstance(state,DeviceState):
        raise TypeError('initialization requires persistent DeviceState')
    with cp.cuda.Device(device):
        meta=_pack(fields,arguments,device)
        global_meta=_state_metadata(state,device) if state is not None else cp.empty(0,dtype=cp.uint64)
        # Initialization uses fixed OPAC and saturation tables; array scratch
        # is bounded by those dimensions plus one integer per solved column.
        needed=1048576 + 4*int(arguments.get('ncellssolve',0))
        if workspace is None: workspace=cp.empty(needed,dtype=cp.uint8)
        if (not isinstance(workspace,cp.ndarray) or workspace.dtype!=cp.uint8 or
                workspace.device.id!=device or not workspace.flags.c_contiguous or workspace.nbytes<needed):
            raise ValueError('CAM stage workspace is too small or belongs to another device')
        error=cp.zeros(1,dtype=cp.int32)
        _module(device).get_function('cam_'+name)(
            (1,),(1,),(meta,global_meta,workspace,np.uint64(workspace.nbytes),error))
    return BoundaryResult(meta,error,workspace,-1,global_meta,(arguments,state))

def stage_scalar(result: BoundaryResult, stage: str, field: str) -> DeviceScalar:
    offset,kind=STAGE_SCALARS[stage][field]
    return DeviceScalar(result.metadata,offset,kind)

@dataclass(frozen=True)
class DriverFrame:
    """Rust/framework pool descriptors at the MPAS driver boundary."""
    state: DeviceState
    from_mpas: Mapping[str, object]
    camrad: Mapping[str, object]
    to_mpas: Mapping[str, object]
    solar: Mapping[str, object]
    clock: Mapping[str, object]
    utility_state: Mapping[str, object] | None = None

def run_driver(call, frame: DriverFrame):
    """MPAS LW:804-960 and SW:834-1006 driver slices, with shared prep external.

    The manager supplies config_camrad_abs_update through its cam_abs flag.
    The pinned LW/SW drivers do not call radiation_camlw_to_MPAS; the
    separately exposed conversion is not inserted into this driver slice.
    """
    longwave=call.slot=='radiation_lw'
    expected='cam_lw' if longwave else 'cam_sw'
    if call.slot not in ('radiation_lw','radiation_sw') or call.scheme!=expected:
        raise ValueError('CAM driver called for a different MPAS radiation slot')
    suffix='lw' if longwave else 'sw'
    if not isinstance(frame,DriverFrame):
        raise TypeError('CAM provider must supply native pool/driver descriptors')
    device=frame.state.device
    stages=[]
    stages.append(launch_stage('radiation_'+suffix+'_from_mpas',frame.from_mpas,device=device))
    solar=launch_stage('radconst',frame.solar,device=device);stages.append(solar)
    clock=launch_stage('driver_clock',frame.clock,device=device);stages.append(clock)
    arguments=dict(frame.camrad)
    arguments.update(dolw=longwave,dosw=not longwave,doabsems=bool(call.flags.cam_abs),n_cldadv=3,
                     declin=stage_scalar(solar,'radconst','declin'),
                     solcon=stage_scalar(solar,'radconst','solcon'),
                     xtime=stage_scalar(clock,'driver_clock','xtime_m'),
                     radt=stage_scalar(clock,'driver_clock','radt'))
    for result in stages:
        error=int(result.error.get()[0])
        if error:raise RuntimeError(f'MPAS CAM driver fatal source line {error}')
    boundary=launch_boundary(arguments,frame.state,utility_state=frame.utility_state)
    error=int(boundary.error.get()[0])
    if error:raise RuntimeError(f'MPAS CAM fatal source line {error}')
    copied=launch_stage('radiation_'+suffix+'_to_mpas',frame.to_mpas,device=device)
    # Retain the asynchronous output conversion and every input owner.
    return boundary,tuple(stages),copied,frame

def register_backends(registry, provider):
    """Register the two MPAS selectors on f4's SlotRegistry callback contract.

    provider(call) must provide DriverFrame, prepared by Rust/native framework
    hooks. Registration is explicit; this function changes no suite default.
    """
    def radiation(call):
        return run_driver(call,provider(call))
    registry.register('radiation_lw','cam_lw','hex-cuda',radiation)
    registry.register('radiation_sw','cam_sw','hex-cuda',radiation)

NATIVE_SOURCE_PINS = {'module_ra_cam_support.F': 'be9d78c6821b919eda62b35cc7c0182136688f40058b89b716591e3eaadc3c8e', 'module_ra_cam.F': 'd26f11487621228c6ee5a08c9fa1b7a2e7277e4044287d782ebd172bd31a327a'}
NATIVE_STATE_SYMBOLS = {'ixcldliq': ('module_ra_cam_support', 'module_ra_cam_support.F', 18), 'ixcldice': ('module_ra_cam_support', 'module_ra_cam_support.F', 19), 'idxsul': ('module_ra_cam_support', 'module_ra_cam_support.F', 28), 'idxsslt': ('module_ra_cam_support', 'module_ra_cam_support.F', 29), 'idxdustfirst': ('module_ra_cam_support', 'module_ra_cam_support.F', 30), 'idxcarbonfirst': ('module_ra_cam_support', 'module_ra_cam_support.F', 31), 'idxocpho': ('module_ra_cam_support', 'module_ra_cam_support.F', 32), 'idxbcpho': ('module_ra_cam_support', 'module_ra_cam_support.F', 33), 'idxocphi': ('module_ra_cam_support', 'module_ra_cam_support.F', 34), 'idxbcphi': ('module_ra_cam_support', 'module_ra_cam_support.F', 35), 'idxbg': ('module_ra_cam_support', 'module_ra_cam_support.F', 36), 'idxvolc': ('module_ra_cam_support', 'module_ra_cam_support.F', 37), 'mxaerl': ('module_ra_cam_support', 'module_ra_cam_support.F', 39), 'sulscl_rf': ('module_ra_cam_support', 'module_ra_cam_support.F', 51), 'carscl_rf': ('module_ra_cam_support', 'module_ra_cam_support.F', 52), 'ssltscl_rf': ('module_ra_cam_support', 'module_ra_cam_support.F', 53), 'dustscl_rf': ('module_ra_cam_support', 'module_ra_cam_support.F', 54), 'bgscl_rf': ('module_ra_cam_support', 'module_ra_cam_support.F', 55), 'volcscl_rf': ('module_ra_cam_support', 'module_ra_cam_support.F', 56), 'tauback': ('module_ra_cam_support', 'module_ra_cam_support.F', 59), 'sulscl': ('module_ra_cam_support', 'module_ra_cam_support.F', 63), 'carscl': ('module_ra_cam_support', 'module_ra_cam_support.F', 64), 'ssltscl': ('module_ra_cam_support', 'module_ra_cam_support.F', 65), 'dustscl': ('module_ra_cam_support', 'module_ra_cam_support.F', 66), 'volcscl': ('module_ra_cam_support', 'module_ra_cam_support.F', 67), 'abs_cff_mss_aer': ('module_ra_cam_support', 'module_ra_cam_support.F', 84), 'c16': ('module_ra_cam_support', 'module_ra_cam_support.F', 112), 'c17': ('module_ra_cam_support', 'module_ra_cam_support.F', 112), 'c26': ('module_ra_cam_support', 'module_ra_cam_support.F', 112), 'c27': ('module_ra_cam_support', 'module_ra_cam_support.F', 112), 'c28': ('module_ra_cam_support', 'module_ra_cam_support.F', 112), 'c29': ('module_ra_cam_support', 'module_ra_cam_support.F', 112), 'c30': ('module_ra_cam_support', 'module_ra_cam_support.F', 112), 'c31': ('module_ra_cam_support', 'module_ra_cam_support.F', 112), 'fwcoef': ('module_ra_cam_support', 'module_ra_cam_support.F', 113), 'fwc1': ('module_ra_cam_support', 'module_ra_cam_support.F', 114), 'fwc2': ('module_ra_cam_support', 'module_ra_cam_support.F', 114), 'fc1': ('module_ra_cam_support', 'module_ra_cam_support.F', 115), 'amco2': ('module_ra_cam_support', 'module_ra_cam_support.F', 116), 'amd': ('module_ra_cam_support', 'module_ra_cam_support.F', 117), 'p0': ('module_ra_cam_support', 'module_ra_cam_support.F', 118), 'ah2onw': ('module_ra_cam_support', 'module_ra_cam_support.F', 121), 'eh2onw': ('module_ra_cam_support', 'module_ra_cam_support.F', 122), 'ah2ow': ('module_ra_cam_support', 'module_ra_cam_support.F', 123), 'cn_ah2ow': ('module_ra_cam_support', 'module_ra_cam_support.F', 124), 'cn_eh2ow': ('module_ra_cam_support', 'module_ra_cam_support.F', 125), 'ln_ah2ow': ('module_ra_cam_support', 'module_ra_cam_support.F', 126), 'ln_eh2ow': ('module_ra_cam_support', 'module_ra_cam_support.F', 127), 'coefh': ('module_ra_cam_support', 'module_ra_cam_support.F', 135), 'coefj': ('module_ra_cam_support', 'module_ra_cam_support.F', 141), 'coefk': ('module_ra_cam_support', 'module_ra_cam_support.F', 145), 'estblh2o': ('module_ra_cam_support', 'module_ra_cam_support.F', 150), 'fat': ('module_ra_cam_support', 'module_ra_cam_support.F', 165), 'fet': ('module_ra_cam_support', 'module_ra_cam_support.F', 174), 'gravit': ('module_ra_cam_support', 'module_ra_cam_support.F', 182), 'rga': ('module_ra_cam_support', 'module_ra_cam_support.F', 183), 'gravmks': ('module_ra_cam_support', 'module_ra_cam_support.F', 184), 'cpair': ('module_ra_cam_support', 'module_ra_cam_support.F', 185), 'epsilo': ('module_ra_cam_support', 'module_ra_cam_support.F', 186), 'epsqs': ('module_ra_cam_support', 'module_ra_cam_support.F', 187), 'sslp': ('module_ra_cam_support', 'module_ra_cam_support.F', 188), 'stebol': ('module_ra_cam_support', 'module_ra_cam_support.F', 189), 'rgsslp': ('module_ra_cam_support', 'module_ra_cam_support.F', 190), 'dpfo3': ('module_ra_cam_support', 'module_ra_cam_support.F', 191), 'dpfco2': ('module_ra_cam_support', 'module_ra_cam_support.F', 192), 'dayspy': ('module_ra_cam_support', 'module_ra_cam_support.F', 193), 'pie': ('module_ra_cam_support', 'module_ra_cam_support.F', 194), 'mwdry': ('module_ra_cam_support', 'module_ra_cam_support.F', 195), 'scon': ('module_ra_cam_support', 'module_ra_cam_support.F', 196), 'co2mmr': ('module_ra_cam_support', 'module_ra_cam_support.F', 197), 'mwco2': ('module_ra_cam_support', 'module_ra_cam_support.F', 198), 'mwh2o': ('module_ra_cam_support', 'module_ra_cam_support.F', 199), 'mwch4': ('module_ra_cam_support', 'module_ra_cam_support.F', 200), 'mwn2o': ('module_ra_cam_support', 'module_ra_cam_support.F', 201), 'mwf11': ('module_ra_cam_support', 'module_ra_cam_support.F', 202), 'mwf12': ('module_ra_cam_support', 'module_ra_cam_support.F', 203), 'cappa': ('module_ra_cam_support', 'module_ra_cam_support.F', 204), 'rair': ('module_ra_cam_support', 'module_ra_cam_support.F', 205), 'tmelt': ('module_ra_cam_support', 'module_ra_cam_support.F', 206), 'r_universal': ('module_ra_cam_support', 'module_ra_cam_support.F', 207), 'latvap': ('module_ra_cam_support', 'module_ra_cam_support.F', 208), 'latice': ('module_ra_cam_support', 'module_ra_cam_support.F', 209), 'zvir': ('module_ra_cam_support', 'module_ra_cam_support.F', 210), 'estbl': ('module_ra_cam_support', 'module_ra_cam_support.F', 218), 'tmin': ('module_ra_cam_support', 'module_ra_cam_support.F', 219), 'tmax': ('module_ra_cam_support', 'module_ra_cam_support.F', 220), 'pcf': ('module_ra_cam_support', 'module_ra_cam_support.F', 221), 'ksul': ('module_ra_cam_support', 'module_ra_cam_support.F', 234), 'wsul': ('module_ra_cam_support', 'module_ra_cam_support.F', 235), 'gsul': ('module_ra_cam_support', 'module_ra_cam_support.F', 236), 'ksslt': ('module_ra_cam_support', 'module_ra_cam_support.F', 237), 'wsslt': ('module_ra_cam_support', 'module_ra_cam_support.F', 238), 'gsslt': ('module_ra_cam_support', 'module_ra_cam_support.F', 239), 'kcphil': ('module_ra_cam_support', 'module_ra_cam_support.F', 240), 'wcphil': ('module_ra_cam_support', 'module_ra_cam_support.F', 241), 'gcphil': ('module_ra_cam_support', 'module_ra_cam_support.F', 242), 'kbg': ('module_ra_cam_support', 'module_ra_cam_support.F', 244), 'wbg': ('module_ra_cam_support', 'module_ra_cam_support.F', 245), 'gbg': ('module_ra_cam_support', 'module_ra_cam_support.F', 246), 'kcphob': ('module_ra_cam_support', 'module_ra_cam_support.F', 247), 'wcphob': ('module_ra_cam_support', 'module_ra_cam_support.F', 248), 'gcphob': ('module_ra_cam_support', 'module_ra_cam_support.F', 249), 'kcb': ('module_ra_cam_support', 'module_ra_cam_support.F', 250), 'wcb': ('module_ra_cam_support', 'module_ra_cam_support.F', 251), 'gcb': ('module_ra_cam_support', 'module_ra_cam_support.F', 252), 'kvolc': ('module_ra_cam_support', 'module_ra_cam_support.F', 253), 'wvolc': ('module_ra_cam_support', 'module_ra_cam_support.F', 254), 'gvolc': ('module_ra_cam_support', 'module_ra_cam_support.F', 255), 'kdst': ('module_ra_cam_support', 'module_ra_cam_support.F', 257), 'wdst': ('module_ra_cam_support', 'module_ra_cam_support.F', 258), 'gdst': ('module_ra_cam_support', 'module_ra_cam_support.F', 259), 'cplos': ('module_ra_cam_support', 'module_ra_cam_support.F', 262), 'cplol': ('module_ra_cam_support', 'module_ra_cam_support.F', 263), 'co2vmr': ('module_ra_cam_support', 'module_ra_cam_support.F', 266), 'n2ovmr': ('module_ra_cam_support', 'module_ra_cam_support.F', 267), 'ch4vmr': ('module_ra_cam_support', 'module_ra_cam_support.F', 268), 'f11vmr': ('module_ra_cam_support', 'module_ra_cam_support.F', 269), 'f12vmr': ('module_ra_cam_support', 'module_ra_cam_support.F', 270), 'yrdata': ('module_ra_cam_support', 'module_ra_cam_support.F', 274), 'co2': ('module_ra_cam_support', 'module_ra_cam_support.F', 311), 'ntoplw': ('module_ra_cam_support', 'module_ra_cam_support.F', 345), 'masterproc': ('module_ra_cam_support', 'module_ra_cam_support.F', 347), 'ozncyc': ('module_ra_cam_support', 'module_ra_cam_support.F', 348), 'indirect': ('module_ra_cam_support', 'module_ra_cam_support.F', 351), 'radforce': ('module_ra_cam_support', 'module_ra_cam_support.F', 353), 'trace_gas': ('module_ra_cam_support', 'module_ra_cam_support.F', 354), 'strat_volcanic': ('module_ra_cam_support', 'module_ra_cam_support.F', 355), 'retab': ('module_ra_cam_support', 'module_ra_cam_support.F', 357), 'abarl': ('module_ra_cam', 'module_ra_cam.F', 10), 'bbarl': ('module_ra_cam', 'module_ra_cam.F', 11), 'cbarl': ('module_ra_cam', 'module_ra_cam.F', 12), 'dbarl': ('module_ra_cam', 'module_ra_cam.F', 13), 'ebarl': ('module_ra_cam', 'module_ra_cam.F', 14), 'fbarl': ('module_ra_cam', 'module_ra_cam.F', 15), 'abari': ('module_ra_cam', 'module_ra_cam.F', 41), 'bbari': ('module_ra_cam', 'module_ra_cam.F', 42), 'cbari': ('module_ra_cam', 'module_ra_cam.F', 43), 'dbari': ('module_ra_cam', 'module_ra_cam.F', 44), 'ebari': ('module_ra_cam', 'module_ra_cam.F', 45), 'fbari': ('module_ra_cam', 'module_ra_cam.F', 46), 'delta': ('module_ra_cam', 'module_ra_cam.F', 67), 'o2mmr': ('module_ra_cam', 'module_ra_cam.F', 68), 'frcsol': ('module_ra_cam', 'module_ra_cam.F', 83), 'wavmin': ('module_ra_cam', 'module_ra_cam.F', 84), 'wavmax': ('module_ra_cam', 'module_ra_cam.F', 85), 'raytau': ('module_ra_cam', 'module_ra_cam.F', 86), 'abh2o': ('module_ra_cam', 'module_ra_cam.F', 87), 'abo3': ('module_ra_cam', 'module_ra_cam.F', 88), 'abco2': ('module_ra_cam', 'module_ra_cam.F', 89), 'abo2': ('module_ra_cam', 'module_ra_cam.F', 90), 'ph2o': ('module_ra_cam', 'module_ra_cam.F', 91), 'pco2': ('module_ra_cam', 'module_ra_cam.F', 92), 'po2': ('module_ra_cam', 'module_ra_cam.F', 93), 'nirwgt': ('module_ra_cam', 'module_ra_cam.F', 94), 'amo': ('module_ra_cam', 'module_ra_cam.F', 191)}
NATIVE_POOL_BINDINGS = {'latcell': ('mesh', 'latcell'), 'loncell': ('mesh', 'loncell'), 'aerosols': ('state', 'aerosols'), 'skintemp': ('sfc_input', 'skintemp'), 'snow': ('sfc_input', 'snow'), 'xice': ('sfc_input', 'xice'), 'xland': ('sfc_input', 'xland'), 'pin': ('atm_input', 'pin'), 'ozmixm': ('atm_input', 'ozmixm'), 'sfc_albedo': ('diag_physics', 'sfc_albedo'), 'sfc_emiss': ('diag_physics', 'sfc_emiss'), 'cldfrac': ('diag_physics', 'cldfrac'), 'o3clim': ('diag_physics', 'o3clim'), 'o3vmr': ('diag_physics', 'o3vmr'), 'm_hybi': ('diag_physics', 'm_hybi'), 'm_ps': ('diag_physics', 'm_ps'), 're_cloud': ('diag_physics', 're_cloud'), 're_ice': ('diag_physics', 're_ice'), 're_snow': ('diag_physics', 're_snow'), 'glw': ('diag_physics', 'glw'), 'lwcf': ('diag_physics', 'lwcf'), 'lwdnb': ('diag_physics', 'lwdnb'), 'lwdnbc': ('diag_physics', 'lwdnbc'), 'lwdnt': ('diag_physics', 'lwdnt'), 'lwdntc': ('diag_physics', 'lwdntc'), 'lwupb': ('diag_physics', 'lwupb'), 'lwupbc': ('diag_physics', 'lwupbc'), 'lwupt': ('diag_physics', 'lwupt'), 'lwuptc': ('diag_physics', 'lwuptc'), 'olrtoa': ('diag_physics', 'olrtoa'), 'rthratenlw': ('tend_physics', 'rthratenlw'), 'rre_cloud': ('diag_physics', 'rre_cloud'), 'rre_ice': ('diag_physics', 'rre_ice'), 'rre_snow': ('diag_physics', 'rre_snow'), 'absnxt': ('diag_physics', 'absnxt'), 'abstot': ('diag_physics', 'abstot'), 'emstot': ('diag_physics', 'emstot'), 'zgrid': ('mesh', 'zgrid'), 'taod5502d': ('diag_physics', 'taod5502d'), 'taod5503d': ('diag_physics', 'taod5503d'), 'coszr': ('diag_physics', 'coszr'), 'gsw': ('diag_physics', 'gsw'), 'swcf': ('diag_physics', 'swcf'), 'swdnb': ('diag_physics', 'swdnb'), 'swdnbc': ('diag_physics', 'swdnbc'), 'swdnt': ('diag_physics', 'swdnt'), 'swdntc': ('diag_physics', 'swdntc'), 'swupb': ('diag_physics', 'swupb'), 'swupbc': ('diag_physics', 'swupbc'), 'swupt': ('diag_physics', 'swupt'), 'swuptc': ('diag_physics', 'swuptc'), 'swddir': ('diag_physics', 'swddir'), 'swddni': ('diag_physics', 'swddni'), 'swddif': ('diag_physics', 'swddif'), 'rthratensw': ('tend_physics', 'rthratensw')}
NATIVE_ARGUMENT_BINDINGS = {'dolw': '.true.', 'dosw': '.false.', 'p_phy': 'pres_hyd_p', 'p8w': 'pres2_hyd_p', 'pi_phy': 'pi_p', 't_phy': 't_p', 'z': 'zmid_p', 'dz8w': 'dz_p', 'rthratenlw': 'rthratenlw_p', 'rthratensw': 'rthratensw_p', 'swupt': 'swupt_p', 'swuptc': 'swuptc_p', 'swdnt': 'swdnt_p', 'swdntc': 'swdntc_p', 'lwupt': 'lwupt_p', 'lwuptc': 'lwuptc_p', 'lwdnt': 'lwdnt_p', 'lwdntc': 'lwdntc_p', 'swupb': 'swupb_p', 'swupbc': 'swupbc_p', 'swdnb': 'swdnb_p', 'swdnbc': 'swdnbc_p', 'lwupb': 'lwupb_p', 'lwupbc': 'lwupbc_p', 'lwdnb': 'lwdnb_p', 'lwdnbc': 'lwdnbc_p', 'swcf': 'swcf_p', 'lwcf': 'lwcf_p', 'gsw': 'gsw_p', 'glw': 'glw_p', 'olr': 'olrtoa_p', 'cemiss': 'cemiss_p', 'taucldc': 'taucldc_p', 'taucldi': 'taucldi_p', 'coszr': 'coszr_p', 'albedo': 'sfc_albedo_p', 'emiss': 'sfc_emiss_p', 'tsk': 'tsk_p', 'xlat': 'xlat_p', 'xlong': 'xlon_p', 'rho_phy': 'rho_p', 'qv3d': 'qv_p', 'qc3d': 'qc_p', 'qr3d': 'qr_p', 'qi3d': 'qi_p', 'qs3d': 'qs_p', 'qg3d': 'qg_p', 'f_qc': 'f_qc', 'f_qr': 'f_qr', 'f_qi': 'f_qi', 'f_qs': 'f_qs', 'f_ice_phy': 'f_ice', 'f_rain_phy': 'f_rain', 'cldfra': 'cldfrac_p', 'xland': 'xland_p', 'xice': 'xice_p', 'num_months': 'num_months', 'levsiz': 'num_oznlevels', 'pin0': 'pin_p', 'ozmixm': 'ozmixm_p', 'paerlev': 'num_aerlevels', 'naer_c': 'num_aerosols', 'm_psp': 'm_psp_p', 'm_psn': 'm_psn_p', 'aerosolcp': 'aerosolcp_p', 'aerosolcn': 'aerosolcn_p', 'm_hybi0': 'm_hybi_p', 'snow': 'snow_p', 'cam_abs_dim1': 'cam_abs_dim1', 'cam_abs_dim2': 'cam_abs_dim2', 'gmt': 'gmt', 'yr': 'year', 'julday': 'julday', 'julian': 'curr_julday', 'dt': 'dt_dyn', 'xtime': 'xtime_m', 'declin': 'declin', 'solcon': 'solcon', 'radt': 'radt', 'degrad': 'degrad', 'n_cldadv': '3', 'abstot_3d': 'abstot_p', 'absnxt_3d': 'absnxt_p', 'emstot_3d': 'emstot_p', 'doabsems': 'doabsems', 'ids': 'ids', 'ide': 'ide', 'jds': 'jds', 'jde': 'jde', 'kds': 'kds', 'kde': 'kde', 'ims': 'ims', 'ime': 'ime', 'jms': 'jms', 'jme': 'jme', 'kms': 'kms', 'kme': 'kme', 'its': 'its', 'ite': 'ite', 'jts': 'jts', 'jte': 'jte', 'kts': 'kts', 'kte': 'kte'}
NATIVE_STATE_MUTABLE = ('ixcldliq', 'ixcldice', 'idxsul', 'idxsslt', 'idxdustfirst', 'idxcarbonfirst', 'idxocpho', 'idxbcpho', 'idxocphi', 'idxbcphi', 'idxbg', 'idxvolc', 'mxaerl', 'sulscl_rf', 'carscl_rf', 'ssltscl_rf', 'dustscl_rf', 'bgscl_rf', 'volcscl_rf', 'tauback', 'sulscl', 'carscl', 'ssltscl', 'dustscl', 'volcscl', 'abs_cff_mss_aer', 'c16', 'c17', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'fwcoef', 'fwc1', 'fwc2', 'fc1', 'amco2', 'amd', 'p0', 'ah2onw', 'eh2onw', 'ah2ow', 'cn_ah2ow', 'cn_eh2ow', 'ln_ah2ow', 'ln_eh2ow', 'coefh', 'coefj', 'coefk', 'estblh2o', 'gravit', 'rga', 'gravmks', 'cpair', 'epsilo', 'epsqs', 'sslp', 'stebol', 'rgsslp', 'dpfo3', 'dpfco2', 'dayspy', 'pie', 'mwdry', 'scon', 'co2mmr', 'mwco2', 'mwh2o', 'mwch4', 'mwn2o', 'mwf11', 'mwf12', 'cappa', 'rair', 'tmelt', 'r_universal', 'latvap', 'latice', 'zvir', 'estbl', 'tmin', 'tmax', 'pcf', 'ksul', 'wsul', 'gsul', 'ksslt', 'wsslt', 'gsslt', 'kcphil', 'wcphil', 'gcphil', 'kbg', 'wbg', 'gbg', 'kcphob', 'wcphob', 'gcphob', 'kcb', 'wcb', 'gcb', 'kvolc', 'wvolc', 'gvolc', 'kdst', 'wdst', 'gdst', 'cplos', 'cplol', 'co2vmr', 'n2ovmr', 'ch4vmr', 'f11vmr', 'f12vmr', 'yrdata', 'co2', 'ntoplw', 'masterproc', 'ozncyc', 'indirect', 'radforce', 'trace_gas', 'strat_volcanic', 'retab', 'abarl', 'bbarl', 'cbarl', 'dbarl', 'ebarl', 'fbarl', 'abari', 'bbari', 'cbari', 'dbari', 'ebari', 'fbari', 'delta', 'o2mmr', 'frcsol', 'wavmin', 'wavmax', 'raytau', 'abh2o', 'abo3', 'abco2', 'abo2', 'ph2o', 'pco2', 'po2', 'nirwgt', 'amo')


# MPAS module_ra_cam_support.F:121-127,234-242 declares GNU descriptors,
# rather than inline array storage, for these initialized CAM tables.
NATIVE_STATE_ALLOCATABLE = frozenset((
    'ah2onw', 'eh2onw', 'ah2ow', 'cn_ah2ow', 'cn_eh2ow', 'ln_ah2ow',
    'ln_eh2ow', 'ksul', 'wsul', 'gsul', 'ksslt', 'wsslt', 'gsslt',
    'kcphil', 'wcphil', 'gcphil'))


def _native_allocatable_words(descriptor, scalar, layout):
    """Borrow initialized GNU table data; module_ra_cam_support.F:121-127,234-242.

    GNU 15's descriptor contains a data pointer, offset, dtype, span and
    dimensions. Its symbol address is not the address of the table words.
    Validate the original bounds and contiguous strides before borrowing.
    """
    import ctypes as ct
    import math
    class Dtype(ct.Structure):
        _fields_ = [('elem_len', ct.c_size_t), ('version', ct.c_int),
                    ('rank', ct.c_byte), ('type', ct.c_byte),
                    ('attribute', ct.c_short)]
    class Dimension(ct.Structure):
        _fields_ = [('stride', ct.c_ssize_t), ('lower', ct.c_ssize_t),
                    ('upper', ct.c_ssize_t)]
    lower, extent, stride = layout
    class Descriptor(ct.Structure):
        _fields_ = [('base', ct.c_void_p), ('offset', ct.c_size_t),
                    ('dtype', Dtype), ('span', ct.c_ssize_t),
                    ('dim', Dimension * len(lower))]
    value = Descriptor.from_address(descriptor)
    dimensions = tuple((d.lower, d.upper-d.lower+1, d.stride) for d in value.dim)
    if (not value.base or value.dtype.rank != len(lower)
            or value.dtype.elem_len != ct.sizeof(scalar)
            or value.span != ct.sizeof(scalar)
            or dimensions != tuple(zip(lower, extent, stride))):
        raise ValueError('Initialized native CAM allocatable descriptor differs from its pinned table layout')
    return (scalar * math.prod(extent)).from_address(value.base)


class NativeCAMWords:
    """Raw strict GNU module storage, module_ra_cam_support.F:22-383.

    The native stand-in initializes constants/tables once. These copies retain
    its exact kind and every mutable module word across mixed CPU/CUDA slots.
    This GNU binding is restricted to f3's authenticated v8.4.2 library; it
    evaluates no physics and is not an independent forecast acceptance claim.
    """
    def __init__(self, native):
        """Pristine globals; module_ra_cam.F:1-176; module_ra_cam_support.F:22-383."""
        import ctypes as ct
        import hashlib
        import math
        import numpy as np
        self.native = native
        sources = native.build['sources']
        for filename, digest in NATIVE_SOURCE_PINS.items():
            key = 'src/core_atmosphere/physics/physics_wrf/' + filename
            if sources.get(key) != digest:
                raise ValueError('CAM native module source differs: ' + filename)
        if native.build['source_pin'] != 'f34984b2c89353cf31c235b08d3b4acd96b37bb3':
            raise ValueError('CAM native initialized tables require pinned MPAS v8.4.2')
        self.views = {}
        for name, kind, rank in STATE_FIELDS:
            module, _, _ = NATIVE_STATE_SYMBOLS[name]
            scalar = {'float':ct.c_float, 'double':ct.c_double,
                      'int':ct.c_int32, 'bool':ct.c_int32}[kind]
            count = math.prod(STATE_LAYOUT[name][1]) if rank else 1
            symbol = '__' + module + '_MOD_' + name
            if name in NATIVE_STATE_ALLOCATABLE:
                descriptor = ct.c_void_p.in_dll(native.lib, symbol)
                native_words = _native_allocatable_words(ct.addressof(descriptor), scalar,
                                                         STATE_LAYOUT[name])
            else:
                native_words = (scalar * count).in_dll(native.lib, symbol)
            self.views[name] = np.ctypeslib.as_array(native_words)

    def read(self):
        """Copy every initialized module word; module_ra_cam_support.F:22-383."""
        return {name: words.copy() for name, words in self.views.items()}

    def write(self, values):
        """Return every mutable module word; module_ra_cam.F:424-443,709-710."""
        if set(values) != set(self.views):
            raise ValueError('CAM native module transfer requires its whole state')
        for name, words in self.views.items():
            value = values[name]
            if value.shape != words.shape or value.dtype != words.dtype:
                raise ValueError('CAM native module transfer kind/extent differs: ' + name)
        for name, words in self.views.items():
            if name not in NATIVE_STATE_MUTABLE and values[name].tobytes() != words.tobytes():
                raise ValueError('CAM transfer attempted to change a native PARAMETER: ' + name)
        for name in NATIVE_STATE_MUTABLE:
            self.views[name][:] = values[name]


@dataclass(frozen=True)
class NativeCAMFrame:
    """Owned ABI copy transaction; mpas_atmphys_driver_radiation_lw.F:804-960."""
    frame: DriverFrame
    owner: object


class _NativeCAMTransfer:
    """Retain raw Fortran bounds and shared aliases; mpas_atmphys_vars.F:179-217,770-803."""
    def __init__(self, provider, call):
        """Prepared arrays follow native allocation; mpas_atmphys_driver.F:185-244."""
        self.provider, self.native, self.cp, self.call = provider, provider.native, provider.cp, call
        self.copies = {}
        self.words = NativeCAMWords(self.native)
        # MPAS src/core_atmosphere/physics/mpas_atmphys_utilities.F:18; mpas_kind_types.F:32 (StrKIND=512).
        import ctypes as ct
        import numpy as np
        words=(ct.c_ubyte*512).in_dll(self.native.lib,
            '__mpas_atmphys_utilities_MOD_mpas_err_message')
        self.message_words=np.ctypeslib.as_array(words)
        self.utility_state={'mpas_err_message':self.cp.asarray(self.message_words)}

    def variable(self, name):
        """Prepared i,k,j descriptors, including rank4; mpas_atmphys_vars.F:179-217,770-803."""
        key = ('prepared', name)
        if key in self.copies:
            return self.copies[key][1]
        original = self.native.get_variable(name)
        if original is None:
            raise RuntimeError('CAM native driver preparation omitted allocated variable ' + name)
        host, lower = original
        if host.ndim == 0:
            return host[()]
        return self._copy(key, host, lower)

    def field(self, pool, name):
        """Original pool order includes halos; mpas_atmphys_driver_radiation_lw.F:343-362,646-658."""
        name = next(field for p, field in self.native._fields
                    if p == pool and field.casefold() == name.casefold())
        level = self.provider.level if pool == 'state' else 1
        key = (pool, name, level)
        if key in self.copies:
            return self.copies[key][1]
        host = self.native.get_field(pool, name, level=level)
        return self._copy(key, host, (1,) * host.ndim)

    def _copy(self, key, host, lower):
        """No numerical conversion of pool words; mpas_atmphys_vars.F:179-217,770-803."""
        stride, count = [], 1
        for size in host.shape:
            stride.append(count)
            count *= size
        value = DeviceArray(self.cp.asarray(host.ravel(order='F')), tuple(lower),
                            tuple(host.shape), tuple(stride))
        self.copies[key] = (host, value, tuple(lower))
        return value

    def argument(self, name):
        """Native pool retrieval and scalar bounds; mpas_atmphys_driver_radiation_lw.F:343-362,905-951."""
        if name == 'its':
            return 1
        if name == 'ite':
            return self.provider.session.layout.dimensions['nCellsSolve']
        if name == 'xtime_s':
            return self.call.time_seconds
        if name in NATIVE_POOL_BINDINGS:
            return self.field(*NATIVE_POOL_BINDINGS[name])
        if name in ('curr_julday', 'gmt', 'julday', 'year'):
            value, lower = self.native.get_table('mpas_atmphys_manager.' + name)
            return value[()]
        if name in ('degrad', 'dpd', 'solcon_0'):
            return self.native.get_constant(name)
        return self.variable(name)

    def state(self):
        """Share native initialized tables with the test seam; mpas_atmphys_camrad_init.F:39-76,119-293."""
        values = {}
        for name, kind, rank in STATE_FIELDS:
            host = self.words.views[name]
            if rank:
                lower, extent, stride = STATE_LAYOUT[name]
                storage = self.cp.asarray(host)
                if kind == 'bool':
                    storage = storage.astype(self.cp.bool_)
                values[name] = DeviceArray(storage, lower, extent, stride)
            else:
                values[name] = host[0]
        return make_device_state(values, int(self.cp.cuda.Device().id))

    def commit(self, result):
        """Copy native driver outputs and shared state; mpas_atmphys_driver_radiation_lw.F:960; mpas_atmphys_driver_radiation_sw.F:1006."""
        import numpy as np
        boundary, stages, copied, frame = result
        error = int(copied.error.get()[0])
        if error:
            raise RuntimeError(f'MPAS CAM output conversion fatal source line {error}')
        for key, (host, value, lower) in self.copies.items():
            host[...] = value.storage.get().reshape(host.shape, order='F')
            if key[0] == 'prepared':
                self.native.put_variable(key[1], host, lower=lower)
            else:
                self.native.put_field(key[0], key[1], host, level=key[2],
                                      nlevels=self.native._fields[key[0], key[1]][2])
        metadata = frame.state.metadata.get()
        values = {}
        for name, kind, rank in STATE_FIELDS:
            if rank:
                value = frame.state.values[name].storage.get()
                values[name] = value.astype(self.words.views[name].dtype, copy=False)
            else:
                offset, _ = STATE_SCALARS[name]
                word = metadata[offset:offset+1]
                if kind == 'double':
                    value = word.view(np.float64)
                elif kind == 'float':
                    value = word.astype(np.uint32).view(np.float32)
                else:
                    value = word.astype(np.uint32).view(np.int32)
                values[name] = value
        self.words.write(values)
        for stage, scalar in ((stages[1], 'declin'), (stages[1], 'solcon')):
            value = stage_scalar(stage, 'radconst', scalar)
            word = value.storage[value.offset:value.offset+1].get().astype(np.uint32).view(np.float32)[0]
            self.native.put_variable(scalar, np.asarray(word))
        # MPAS module_ra_cam.F:477 writes the framework's complete CHARACTER word set.
        self.message_words[:]=self.utility_state['mpas_err_message'].get()
        self.native.put_variable('doabsems', np.asarray(int(boundary.metadata[boundary.doabsems_offset].get()), dtype=np.int32))


def bind_native_pools(provider, call):
    """Whole native CAM driver interface; mpas_atmphys_driver_radiation_lw.F:804-960; mpas_atmphys_driver_radiation_sw.F:834-1006."""
    owner = _NativeCAMTransfer(provider, call)
    suffix = 'lw' if call.slot == 'radiation_lw' else 'sw'
    def stage(name):
        return {field: owner.argument(field) for field, _, _ in STAGE_FIELDS[name]}
    dynamic = {'dolw', 'dosw', 'doabsems', 'n_cldadv', 'declin', 'solcon', 'xtime', 'radt'}
    arguments = {field: owner.argument(NATIVE_ARGUMENT_BINDINGS[field])
                 for field, _, _ in ARGUMENT_FIELDS if field not in dynamic}
    # run_driver replaces these words with exact native manager/device outputs.
    arguments.update(dolw=False, dosw=False, doabsems=False, n_cldadv=3,
                     declin=0., solcon=0., xtime=0., radt=0.)
    solar = {field: owner.argument('curr_julday' if field == 'julian' else field)
             for field, _, _ in STAGE_FIELDS['radconst']}
    clock = {'xtime_s':call.time_seconds, 'dt_radt':owner.variable('dt_radt' + suffix),
             'xtime_m':0., 'radt':0.}
    frame = DriverFrame(owner.state(), stage('radiation_' + suffix + '_from_mpas'),
                        arguments, stage('radiation_' + suffix + '_to_mpas'), solar, clock, owner.utility_state)
    return NativeCAMFrame(frame, owner)


def run_slot(call):
    """Commit complete CAM driver execution; mpas_atmphys_driver_radiation_lw.F:804-960; mpas_atmphys_driver_radiation_sw.F:834-1006."""
    binding = call.pools
    if not isinstance(binding, NativeCAMFrame):
        raise TypeError('CAM native-pool callback requires its bound whole driver frame')
    result = run_driver(call, binding.frame)
    binding.owner.commit(result)
    return result

ORACLE_LOCAL_FIELDS = (('local.module_ra_cam_support.trcab.ab', 'double', 1), ('local.module_ra_cam_support.trcab.abp', 'double', 1), ('local.module_ra_cam_support.trcab.bb', 'double', 1), ('local.module_ra_cam_support.trcab.bbp', 'double', 1), ('local.module_ra_cam_support.trcab.g1', 'double', 1), ('local.module_ra_cam_support.trcab.g2', 'double', 1), ('local.module_ra_cam_support.trcab.g3', 'double', 1), ('local.module_ra_cam_support.trcab.g4', 'double', 1), ('local.module_ra_cam_support.trcabn.ab', 'double', 1), ('local.module_ra_cam_support.trcabn.abp', 'double', 1), ('local.module_ra_cam_support.trcabn.bb', 'double', 1), ('local.module_ra_cam_support.trcabn.bbp', 'double', 1), ('local.module_ra_cam_support.trcabn.g1', 'double', 1), ('local.module_ra_cam_support.trcabn.g2', 'double', 1), ('local.module_ra_cam_support.trcabn.g3', 'double', 1), ('local.module_ra_cam_support.trcabn.g4', 'double', 1), ('local.module_ra_cam_support.trcems.ab', 'double', 1), ('local.module_ra_cam_support.trcems.abp', 'double', 1), ('local.module_ra_cam_support.trcems.bb', 'double', 1), ('local.module_ra_cam_support.trcems.bbp', 'double', 1), ('local.module_ra_cam_support.trcems.g1', 'double', 1), ('local.module_ra_cam_support.trcems.g2', 'double', 1), ('local.module_ra_cam_support.trcems.g3', 'double', 1), ('local.module_ra_cam_support.trcems.g4', 'double', 1), ('local.module_ra_cam_support.trcplk.f1', 'double', 1), ('local.module_ra_cam_support.trcplk.f2', 'double', 1), ('local.module_ra_cam_support.trcplk.f3', 'double', 1), ('local.module_ra_cam_support.trcpth.diff', 'double', 0), ('local.module_ra_cam.oznint.date_oz', 'int', 1), ('local.module_ra_cam.get_aerosol.cdaym', 'double', 0), ('local.module_ra_cam.get_aerosol.cdayp', 'double', 0), ('local.module_ra_cam.get_aerosol.mid', 'double', 1), ('local.module_ra_cam.get_aerosol.mo_nxt', 'int', 0), ('local.module_ra_cam.get_aerosol.nm', 'int', 0), ('local.module_ra_cam.get_aerosol.np', 'int', 0), ('allocation.module_ra_cam.camrad.tauxci', 'double', 2), ('allocation.module_ra_cam.camrad.tauxcl', 'double', 2))
