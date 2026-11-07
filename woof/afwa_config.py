"""Pinned WRF 4.7.1 AFWA controls and native cross-domain normalization."""
DEFAULTS={f'afwa_{k}_opt':0 for k in ('diag','therm','ptype','vil','radar','vis','icing','buoy','severe')}
PTYPE_DEFAULTS={'afwa_ptype_ccn_tmp':264.15,'afwa_ptype_tot_melt':50.}
UNPORTED=('cloud','turb')
def drop_inert(values):
    if values.get("afwa_bad_data_check",0)==0:values.pop("afwa_bad_data_check",None)
    # Preserve fingerprints made before this package existed, including
    # forecasts that already enabled other AFWA packages.
    for key in ("afwa_icing_opt","afwa_buoy_opt","afwa_severe_opt"):
        if values.get(key,0)==0:values.pop(key,None)
    if not any(values.get(k,0) for k in DEFAULTS):
        for k in (*DEFAULTS,*PTYPE_DEFAULTS,"afwa_bad_data_check"):values.pop(k,None)
def validate(cfg):
    if type(cfg.afwa_bad_data_check) is not int or cfg.afwa_bad_data_check<0:
        raise ValueError("afwa_bad_data_check must be a nonnegative integer")
    for k in DEFAULTS:
        v=getattr(cfg,k)
        if type(v) is not int or v not in (0,1):raise ValueError(k+' must be 0 or 1')
    if any(getattr(cfg,k) for k in DEFAULTS if k!='afwa_diag_opt') and not cfg.afwa_diag_opt:
        raise ValueError('AFWA subpackages require afwa_diag_opt=1, as native WRF enables automatically')
    if cfg.afwa_diag_opt and cfg.output_interval_s<=0:raise ValueError('AFWA requires a nonzero history interval, as native WRF')
    if cfg.afwa_diag_opt and not cfg.moist:
        raise ValueError('AFWA MSLP reads qvapr unconditionally; native WRF leaves it undefined without a water-vapor Registry carrier')
    if cfg.afwa_vis_opt and cfg.chem_sets:
        raise ValueError("AFWA visibility requires the native five-bin dust concentration coupling; its chemistry-to-AFWA carrier is not yet wired")
    # The native radar/VIL routines read these arrays without F_Q* guards.
    # Registry.EM_COMMON:3020-3053 and registry.sbm:318-319 declare them.
    graupel=(2,6,7,8,9,10,16,18,24,26,27,28,29,30,32,38,40,56,97,106)
    if (cfg.afwa_radar_opt or cfg.afwa_vil_opt or cfg.afwa_vis_opt) and cfg.mp_physics not in graupel:
        raise ValueError('AFWA radar/VIL/visibility require native QR, QS and QG carriers; the selected WRF microphysics leaves their AFWA input arrays undefined')
def import_controls(section,max_dom,error):
    values={}
    for k,default in DEFAULTS.items():
        raw=section.registry_col(k,max_dom,default)
        if any(type(x) is not int or x not in (0,1) for x in raw):raise error('afwa',k,raw,'requires 0 or 1')
        values[k]=int(any(raw))  # module_check_a_mundo: any domain -> all domains
    for stem in UNPORTED:
        k='afwa_'+stem+'_opt';raw=section.registry_col(k,max_dom,0)
        if any(x!=0 for x in raw):raise error('afwa',k,raw,'native AFWA '+stem+' reads undefined '+('ceil_min' if stem=='cloud' else 'terrain_dx/terrain_dy')+' on valid real inputs; a trapped native undefined-value witness prevents assigning this option a substitute result')
    if any(values.values()):values['afwa_diag_opt']=1
    v=section.scalar('afwa_bad_data_check',0)
    if type(v) is not int or v<0:raise error('afwa','afwa_bad_data_check',[v],'requires a nonnegative integer')
    if v:values['afwa_bad_data_check']=v
    for k,default in PTYPE_DEFAULTS.items():values[k]=section.scalar(k,default)
    section.finish();return values

def registry_rows(registry):
    from woof.core.afwa_diagnostics import PACKAGES,HISTORY,ICING_RAW
    packages=registry.setdefault('diagnostic_packages',{})
    for selector,fields in PACKAGES.items():
        packages[selector]={'0':{'label':'Off','fields':[]},'1':{'label':'WRF 4.7.1 '+selector,'fields':list(fields),'source_pin':'f52c197ed39d12e087d02c50f412d90d418f6186','consumer':'woof/core/afwa_diagnostics.py'}}
        registry['parameters'][selector]={'type':'integer','default':0,'enum':[0,1],'per_domain':False,'consuming_read':'woof/core/afwa_diagnostics.py'}
    for k,default in PTYPE_DEFAULTS.items():
        registry['parameters'][k]={'type':'real','default':default,'per_domain':False,'consuming_read':'woof/core/afwa_diagnostics.py'}
    registry['parameters']['afwa_bad_data_check']={'type':'integer','default':0,'minimum':0,'per_domain':False,'source_pin':'f52c197ed39d12e087d02c50f412d90d418f6186','consuming_read':'woof/core/afwa_diagnostics.py'}
    state={k:{'description':d,'units':u,'history':True,'restart':True} for k,(_,u,d) in HISTORY.items()}
    state['afwa_precip']={'description':'AFWA Diagnostic: Precipitation bucket','units':'mm','history':False,'restart':True}
    for k in ICING_RAW:state[k]={'description':'SUPERCOOLED WATER MIXING RATIO','units':'kg kg-1','history':False,'restart':True}
    from woof.core.afwa_diagnostics import SEVERE_STATE
    for k in SEVERE_STATE:
        state[k]={'description':'Native AFWA severe interval state','units':'s' if k.endswith('dur') else '%','history':False,'restart':True}
    registry.setdefault('diagnostic_state',{})['afwa']=state
    packages['afwa_diag_opt']['1']['fields'].append('afwa_precip')
    packages['afwa_icing_opt']['1']['fields'].extend(ICING_RAW)
