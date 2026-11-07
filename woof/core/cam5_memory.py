"""CUDA-free CAM column allocation contract; pools persist per domain."""
CAMMGMP_BLOCK=64
CAMMGMP_R8_SLOTS_PER_LEVEL=200
CAMMGMP_WORKSPACE_BUDGET_BYTES=1024*1024*1024
CAMZM_BLOCK=64
CAMZM_R8_SLOTS_PER_LEVEL=96
CAMZM_WORKSPACE_BUDGET_BYTES=768*1024*1024
CAMMGMP_MASS_INPUTS = ("p_hyd", "t_phy", "pi_phy", "z", "alt",
                       "icwmrsh", "icwmrdp", "shfrc")
CAMMGMP_FULL_INPUTS = ("p8w", "tke", "kvh", "cmfmc", "cmfmc2")
CAMMGMP_SURFACE_INPUTS = ("ht", "xland", "snowh")
CAMMGMP_INOUT_3D = ("th", "qv", "qc", "qi", "qs", "qr", "nc", "ni", "ns",
                    "nr", "qndrop", "cldfra_old_mp", "cldfra_mp",
                    "cldfra_mp_all", "lradius", "iradius", "cldfrai",
                    "cldfral", "cldfra_conv", "wsedl3d", "rh_old_mp",
                    "lcd_old_mp")
CAMMGMP_INOUT_2D = ("rainnc", "rainncv", "snownc", "snowncv", "sr")
CAMZM_MASS_INPUTS = ("th", "t", "p", "pi", "qv", "qc", "qi", "z", "dz8w",
                     "u", "v", "cldfra")
CAMZM_FULL_INPUTS = ("p8w", "z_at_w", "tke_pbl")
CAMZM_SURFACE_INPUTS = ("tsk", "ust", "mavail", "pblh", "xland", "ht", "psfc",
                        "hfx", "qfx", "tpert")
CAMZM_MASS_OUTPUTS = (
    "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqcncuten", "rqincuten",
    "rucuten", "rvcuten", "zmmu", "zmmd", "zmdt", "zmdq", "dlf", "evaptzm",
    "fzsntzm", "evsntzm", "evapqzm", "zmflxprc", "zmflxsnw", "zmntprpd",
    "zmntsnpd", "zmeiheat", "zmmtu", "zmmtv", "zmupgu", "zmupgd", "zmvpgu",
    "zmvpgd", "zmicuu", "zmicud", "zmicvu", "zmicvd", "zmdice", "zmdliq",
    "evapcdp3d", "icwmrdp3d", "rprddp3d", "dp3d", "du3d", "ed3d", "eu3d",
    "md3d", "mu3d")
CAMZM_FULL_OUTPUTS = ("cmfmc", "cmfmcdzm")
CAMZM_SURFACE_OUTPUTS = ("raincv", "pratec", "cape", "rliq", "pconvb",
                         "pconvt", "cubot", "cutop", "preccdzm", "precz",
                         "dsubcld2d")
CAMZM_INT_OUTPUTS = ("ideep2d", "jt2d", "maxg2d", "lengath2d")
CAMZM_FORECAST_OUTPUTS = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                          "rqcncuten", "rqincuten", "rucuten", "rvcuten",
                          "raincv", "pratec", "cape", "cubot", "cutop")

CAMMGMP_INPUTS=CAMMGMP_MASS_INPUTS+CAMMGMP_FULL_INPUTS+CAMMGMP_SURFACE_INPUTS
CAMZM_INPUTS=CAMZM_MASS_INPUTS+CAMZM_FULL_INPUTS+CAMZM_SURFACE_INPUTS
CAMZM_OUTPUTS=CAMZM_MASS_OUTPUTS+CAMZM_FULL_OUTPUTS+CAMZM_SURFACE_OUTPUTS

def chunk_columns(nz,ncols,slots,budget,block=64, *, pool_first=False):
    raw=max(1,budget//(8*slots*(nz+1)))
    if not pool_first:raw=min(raw,ncols)
    if raw>=block:raw=raw//block*block
    return min(raw,ncols)

def workspace_bytes(cfg):
    """Exact pool/owned buffer payload, excluding DomainState/physics fields.

    CuPy allocator bins and short-lived coupling expressions are charged by
    preflight's existing pool/transient envelope, as for other physics slots.
    """
    from woof.core.physics_inventory import (UWSHCU_MASS_INPUTS,UWSHCU_FULL_INPUTS,
        UWSHCU_SURFACE_INPUTS,UWSHCU_MASS_OUTPUTS,UWSHCU_FULL_OUTPUTS,
        UWSHCU_SURFACE_OUTPUTS,uwshcu_chunk_columns,uwshcu_workspace_slots)
    nz,n=cfg.nz,cfg.nx*cfg.ny;m=nz*n*4;w=(nz+1)*n*4;s=n*4
    total=0
    if cfg.bl_pbl_physics == 9 and (cfg.mp_physics == 11 or cfg.cu_physics == 7
                                   or getattr(cfg, 'shcu_physics', 0) == 2):
        from woof.core.physics_inventory import (UWPBL_MASS_OUTPUTS,
            UWPBL_DIAGNOSTICS_FULL, UWPBL_SURFACE_OUTPUTS)
        # Its two column pools/status are charged by preflight's UW term.
        total += (len(UWPBL_MASS_OUTPUTS) * m + len(UWPBL_DIAGNOSTICS_FULL) * w
                  + (len(UWPBL_SURFACE_OUTPUTS) + 1) * s)
    if cfg.mp_physics==11:
        total+=8*CAMMGMP_R8_SLOTS_PER_LEVEL*(nz+1)*chunk_columns(nz,n,
            CAMMGMP_R8_SLOTS_PER_LEVEL,CAMMGMP_WORKSPACE_BUDGET_BYTES,pool_first=True)+4
    if cfg.cu_physics==7:
        want=set(CAMZM_OUTPUTS)
        total+=sum(m for x in CAMZM_MASS_OUTPUTS if x in want)
        total+=sum(w for x in CAMZM_FULL_OUTPUTS if x in want)
        total+=sum(s for x in CAMZM_SURFACE_OUTPUTS if x in want)+len(CAMZM_INT_OUTPUTS)*s
        total+=len(CAMZM_MASS_INPUTS)*m+len(CAMZM_FULL_INPUTS)*w+(len(CAMZM_SURFACE_INPUTS)+1)*s
        # Fallback zero cloud/temperature perturbation, only when read.
        if cfg.mp_physics!=11:total+=m
        if cfg.bl_pbl_physics!=9:total+=s
        total+=8*CAMZM_R8_SLOTS_PER_LEVEL*(nz+1)*chunk_columns(nz,n,
            CAMZM_R8_SLOTS_PER_LEVEL,CAMZM_WORKSPACE_BUDGET_BYTES)+8
        total+=8*(len(CAMZM_INPUTS)+len(CAMZM_OUTPUTS)+len(CAMZM_INT_OUTPUTS))
    if getattr(cfg,'shcu_physics',0)==2:
        total+=(len(UWSHCU_MASS_INPUTS)+len(UWSHCU_MASS_OUTPUTS)+1)*m
        total+=(len(UWSHCU_FULL_INPUTS)+len(UWSHCU_FULL_OUTPUTS))*w
        total+=(len(UWSHCU_SURFACE_INPUTS)+len(UWSHCU_SURFACE_OUTPUTS))*s
        a,b=uwshcu_workspace_slots(nz)
        total+=uwshcu_chunk_columns(nz,n)*(8*a+4*max(b,1))+4
    if getattr(cfg, 'shcu_physics', 0) in (3, 4):
        # Owned surfaces plus raw outputs and coupled mass tendencies.
        total += 4*s + (8 if cfg.shcu_physics == 3 else 12)*m
        if cfg.shcu_physics == 3:
            total += 2*s + 343270
        else:
            # Full-grid upper bound on the bounded 22-plane tiled arena.
            total += 22*nz*((n+127)//128)*128*4 + s + 4
    if getattr(cfg, 'shcu_physics', 0) == 5:
        from woof.core.deng_contract import arena_slots, MASS, SURFACE
        total += 4*arena_slots(nz)*n + (len(MASS)+8)*m + (len(SURFACE)+101)*s + 441696
    return total
