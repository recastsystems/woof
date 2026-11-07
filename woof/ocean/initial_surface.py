"""Publish a dated PWP analysis consistently to atmospheric surface carriers."""


def synchronize(fields):
    """Use the initialized top ocean layer over supported ocean cells.

    Called once after a cold dated ocean initialization, before the first
    surface flux. Land, lakes and unsupported columns keep their own owner.
    This is surface consistency, not a balanced atmospheric DA increment.
    Existing ice skin/concentration retain ownership of the ice fraction.
    """
    import cupy as cp
    wet=fields['_pwp_land']>1.5
    temperature=fields['om_tmp'][0]
    cp.copyto(fields['sst'],temperature,where=wet)
    if 'ice_volume' in fields:
        fraction=cp.clip(fields['xice'],0,1)
        skin=fraction*fields['ice_skin']+(1-fraction)*temperature
        cp.copyto(fields['tsk'],skin,where=wet)
    else:
        cp.copyto(fields['tsk'],temperature,where=wet)
