"""CAM5 carrier inventory, shared by runtime allocation and memory accounting."""
from __future__ import annotations

MP_DIAGNOSTICS = ("qndrop", "cldfra_old_mp", "cldfra_mp", "cldfra_mp_all",
    "lradius", "iradius", "cldfrai", "cldfral", "cldfra_conv", "wsedl3d",
    "rh_old_mp", "lcd_old_mp")
DEEP_EXPORTS = ("dlf", "cmfmc", "rliq", "cutop", "cubot")
LIQUID_NUMBER = {9:"nc",10:"nc",11:"nc",16:"nc",18:"qndrop",28:"nc",29:"nc",900:"nc"}

def selected(cfg):
    return cfg.mp_physics == 11 or cfg.cu_physics == 7 or getattr(cfg,"shcu_physics",0)==2

def field_shapes(cfg):
    extra = {}
    if getattr(cfg, 'shcu_physics', 0) == 5:
        from woof.core.deng_contract import field_shapes as deng_shapes
        extra = deng_shapes(cfg)
    if not selected(cfg):
        if getattr(cfg, 'shcu_physics', 0) in (3, 4):
            s = (cfg.ny, cfg.nx)
            return {name:s for name in ('rainsh', 'hbot', 'htop')}
        return extra
    from woof.core.physics_inventory import (UWSHCU_MASS_OUTPUTS,
        UWSHCU_FULL_OUTPUTS,UWSHCU_SURFACE_OUTPUTS)
    m=(cfg.nz,cfg.ny,cfg.nx);w=(cfg.nz+1,cfg.ny,cfg.nx);s=(cfg.ny,cfg.nx)
    fields={n:m for n in UWSHCU_MASS_OUTPUTS+("icwmrdp","cam_t_phy","cam_p_hyd")}
    fields.update({n:w for n in UWSHCU_FULL_OUTPUTS+("cam_p8w","cam_kvh","tke_pbl")})
    fields.update({n:s for n in UWSHCU_SURFACE_OUTPUTS+("cush","rainsh")})
    # Native shallow convection adds to retained deep exports. They reset
    # only on a due CU call (positive CUDT holds them between calls).
    if cfg.cu_physics==7:
        for n in ('rucuten','rvcuten','rqcncuten','rqincuten'):
            fields['cam_held_'+n]=m
        fields['cape']=s
    if cfg.cu_physics == 7:
        from woof.core.cam5_memory import (CAMZM_MASS_OUTPUTS, CAMZM_FULL_OUTPUTS,
            CAMZM_SURFACE_OUTPUTS, CAMZM_INT_OUTPUTS, CAMZM_FORECAST_OUTPUTS)
        from woof.io.cam5_schema import CAM_OUTPUT_ROWS
        for names, shape in ((CAMZM_MASS_OUTPUTS,m),(CAMZM_FULL_OUTPUTS,w),
                             (CAMZM_SURFACE_OUTPUTS,s),(CAMZM_INT_OUTPUTS,s)):
            for name in names:
                if name in CAMZM_FORECAST_OUTPUTS or name in DEEP_EXPORTS:continue
                if name == 'icwmrdp3d':continue  # canonical alias icwmrdp
                stagger = CAM_OUTPUT_ROWS.get(name, ('','',''))[2]
                fields[name] = w if stagger == 'Z' else shape
    if getattr(cfg, 'shcu_physics', 0) in (3, 4):
        fields.update({name:s for name in ('rainsh', 'hbot', 'htop')})
    fields.update(extra)
    return fields
