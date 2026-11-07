"""CPU-only KF-CuP field order and retained-state declaration.

WRF v4.6.1 module_cu_kfcup.F and Registry.EM_COMMON:1355-1384.
LNTERMS is native-undefined outside shallow columns; the port carries zero.
CLDFRATEND_CUP is cold-initialized and never written by native KF_cup_CPS.
"""
KFCUP_LEVEL_INPUTS = ('u','v','th','t','dz8w','p','pi','rho','qv','z','z_at_w','w0avg')
KFCUP_SURFACE_INPUTS = ('ht','hfx','qfx','xland','pblh')
KFCUP_HELD_SCALARS = ('nca','raincv','cutop','cubot','shall','slopesfc','slopeez',
    'sigmasfc','sigmaez','taucloud','tactive','tstar','lnint','activefrac',
    'wcloudbase','wact_cup','wulcl_cup','tcloud_cup')
KFCUP_HELD_INTS = ('cupflag','cu_act_flag')
KFCUP_LEVEL_OUTPUTS = ('rthcuten','rqvcuten','rqccuten','rqrcuten','rqicuten','rqscuten',
    'cldfra_cup','wup_cup','qc_ic_cup','qndrop_ic_cup','qc_iu_cup','fcvt_qc_to_pr_cup',
    'fcvt_qc_to_qi_cup','fcvt_qi_to_pr_cup','mfup_cup','mfup_ent_cup','mfdn_cup',
    'mfdn_ent_cup','updfra_cup')
KFCUP_WS_SLOTS = 148
from woof.physics_vertical_contract import KFCUP_VERTICAL_LEVEL_BOUNDS
KFCUP_PARAMETER_DEFAULTS = {'numbins':1,'thbinsize':1.0,'rbinsize':1.0,
    'mindeepfreq':1.0,'minshallowfreq':1.0,'shallowcu_forced_ra':False}
KFCUP_STATE_ROWS = tuple(
    {'name':name, 'shape':'mass', 'dtype':'float32', 'initial':0.0}
    for name in (*KFCUP_LEVEL_OUTPUTS[6:], 'cldfratend_cup', 'lnterms')) + tuple(
    {'name':name, 'shape':'surface', 'dtype':'float32',
     'initial':2.0 if name=='shall' else 0.0}
    for name in KFCUP_HELD_SCALARS[2:]) + tuple(
    {'name':name, 'shape':'surface', 'dtype':'int32', 'initial':0}
    for name in KFCUP_HELD_INTS)
