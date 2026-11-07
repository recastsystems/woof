"""CUDA-free WRF 4.7.1 Deng state and launch inventory."""
ARGUMENTS = ('dt', 'ktau', 'dx', 'xtime', 'gmt', 'xlv', 'xls', 'xlv0', 'xlv1', 'xls0', 'xls1', 'cp', 'r', 'g', 'svp1', 'svp2', 'svp3', 'svpt0', 'adapt_step_flag', 'dsigma', 'xlong', 'ht', 'pblh', 'u', 'v', 'w', 'th', 't', 'qv', 'qc', 'qr', 'dz8w', 'pcps', 'rho', 'z_at_w', 'pi', 'tke', 'kth', 'bbls', 'ten_radl', 'ten_rads', 'rainsh', 'rainshv', 'rainshvb', 'pblmax', 'capesave', 'xtime1', 'radsave', 'clddpthb', 'cldtopb', 'mavail', 'pblhavg', 'ainckfsa', 'ltopb', 'kdcldtop', 'kdcldbas', 'w0avg', 'tkeavg', 'cldareaa', 'cldareab', 'cldliqa', 'cldliqb', 'cldfra_sh', 'ca_rad', 'cw_rad', 'wub', 'rushten', 'rvshten', 'rthshten', 'rqvshten', 'rqcshten', 'rqrshten', 'rdcashten', 'rqcdcshten')
SURFACE = ('rainsh', 'rainshv', 'rainshvb', 'pblmax', 'capesave', 'xtime1', 'radsave', 'clddpthb', 'cldtopb', 'pblhavg', 'ltopb', 'kdcldtop', 'kdcldbas')
MASS = ('w0avg', 'tkeavg', 'cldareaa', 'cldareab', 'cldliqa', 'cldliqb', 'cldfra_sh', 'ca_rad', 'cw_rad', 'wub', 'rushten', 'rvshten', 'rthshten', 'rqvshten', 'rqcshten', 'rqrshten', 'rdcashten', 'rqcdcshten')
INTEGER = ("ltopb", "kdcldtop", "kdcldbas")
def arena_slots(nz): return 512*nz+8192
def field_shapes(cfg):
    m=(cfg.nz,cfg.ny,cfg.nx);s=(cfg.ny,cfg.nx)
    return {**{n:s for n in SURFACE},**{n:m for n in MASS},"ainckfsa":(100,*s),"deng_xlong":s}
