"""Additive fork numerics configuration; defaults of existing classes do not move.

Source: RRFSx/MPAS-Model src/core_atmosphere/Registry.xml (&nhyd_model, &damping,
&limited_area) at 1d6e370e. v8.4.1 names below denote the shared container layout;
the suite-table variant and driver receipt identify the executed numerics.
"""
from dataclasses import dataclass, fields
import numpy as np
from .config_v841 import V841DryDycoreConfig
from .mpasphys.config_v842 import NativeMpasDycoreConfigV842


@dataclass(frozen=True, slots=True)
class NoaaV841DycoreConfig(NativeMpasDycoreConfigV842):
    """Upstream v8.4.1 dynamics with the NOAA transport and boundary delta.

    RRFSx 44c0849 time_integration.F:4839-4950,8030-8105. The June
    fork configuration below remains independent of this table selection.
    """
    dycore_variant: str = 'mpas-v8.4.1-noaa'
    source_commit: str = ''
    physics_pin_set: str = ''
    config_scalar_eddy_mix: bool = False
    config_h_scalar_filter4: float = 0.0
    config_lbc_w: str = 'zero'
    config_relax_zone_divdamp_coef: float = 6.0
    config_number_cam_damping_levels: int = 4
    config_number_rayleigh_damp_u_levels: int = 6
    config_rayleigh_damp_u_timescale_days: float = 5.0
    # Named differences are supplied by the route's dycore defaults.
    prognostic_les_viscosity_floor: float = 0.0

    def validate(self, *, les_dissipation=None):
        NativeMpasDycoreConfigV842.validate(self, les_dissipation=les_dissipation)
        if self.config_lbc_w.strip() not in ('zero', 'nearest'):
            raise ValueError('unknown native config_lbc_w option')
        if self.prognostic_les_viscosity_floor not in (0.0, 0.01):
            raise ValueError('prognostic LES supports the native absent or 0.01 viscosity floor')
        if not isinstance(self.config_scalar_eddy_mix, (bool, np.bool_)):
            raise TypeError('config_scalar_eddy_mix is a native logical')
        if not np.isfinite(self.config_h_scalar_filter4):
            raise ValueError('config_h_scalar_filter4 must be finite')


@dataclass(frozen=True,slots=True)
class RRFSDryDycoreConfig(V841DryDycoreConfig):
    dycore_variant: str = 'mpas-v8.3.1-noaa'
    # MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:73-283 at 1d6e370e.
    # Standalone fork dynamics defaults. The table-resolved native configuration
    # and explicit namelist fields override these when constructing a forecast.
    config_dt: float = 720.0
    config_time_integration_order: int = 2
    config_number_of_sub_steps: int = 2
    config_dynamics_split_steps: int = 3
    config_split_dynamics_transport: bool = True
    config_positive_definite: bool = False
    config_apvm_upwinding: float = 0.5
    config_horiz_mixing: str = '2d_smagorinsky'
    config_visc4_2dsmag: float = 0.05
    config_smagorinsky_coef: float = 0.125
    config_xnutr: float = 0.2
    config_epssm: float = 0.1
    config_scalar_eddy_mix: bool = False
    config_h_scalar_filter4: float = 0.0
    config_lbc_w: str = 'zero'
    config_mix_full: bool = True
    config_number_cam_damping_levels: int = 4
    config_number_rayleigh_damp_u_levels: int = 6
    config_rayleigh_damp_u_timescale_days: float = 5.0

    # MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:148-321 at 1d6e370e.
    def validate(self):
        """Fork Registry.xml at 1d6e370e; scalar epssm is active on this type."""
        values={f.name:getattr(self,f.name) for f in fields(V841DryDycoreConfig)}
        values['config_epssm']=0.0
        # The selected native provider validates suite metadata. The inherited
        # dry validator sees only dynamics, as NativeMpasDycoreConfigV842 does.
        values['config_physics_suite']='none'
        values['config_dynamics_split_steps']=3
        values['config_split_dynamics_transport']=True
        # Parent guards describe missing v8.4.2 branches, now implemented here.
        for key in ('config_h_theta_eddy_visc2','config_h_mom_eddy_visc2','config_h_theta_eddy_visc4','config_h_mom_eddy_visc4','config_v_theta_eddy_visc2','config_v_mom_eddy_visc2'):
            values[key]=0.0
        values['config_vertical_mixing']=False
        values['config_horiz_mixing']='off'
        values['config_len_disp']=0.0
        values['config_visc4_2dsmag']=0.0
        values['config_smagorinsky_coef']=0.0
        values['config_mpas_cam_coef']=0.0
        V841DryDycoreConfig(**values).validate()
        from .mixing_rrfs import RRFSMixingConfig
        RRFSMixingConfig(**{f.name:getattr(self,f.name) for f in fields(RRFSMixingConfig)}).validate()
        if isinstance(self.config_epssm,(bool,np.bool_)) or not isinstance(self.config_epssm,(int,float,np.integer,np.floating)):
            raise TypeError('config_epssm is a native real')
        if not isinstance(self.config_scalar_eddy_mix,(bool,np.bool_)):
            raise TypeError('config_scalar_eddy_mix is a native logical')
        if isinstance(self.config_h_scalar_filter4,(bool,np.bool_)) or not isinstance(self.config_h_scalar_filter4,(int,float,np.integer,np.floating)):
            raise TypeError('config_h_scalar_filter4 is a native real')
        if isinstance(self.config_dynamics_split_steps,(bool,np.bool_)) or not isinstance(self.config_dynamics_split_steps,(int,np.integer)) or self.config_dynamics_split_steps < 1:
            raise ValueError('config_dynamics_split_steps must be a positive native integer')
        if not isinstance(self.config_split_dynamics_transport,(bool,np.bool_)):
            raise TypeError('config_split_dynamics_transport is a native logical')
        if self.config_lbc_w.strip() not in ('zero','nearest'):
            raise ValueError('unknown native config_lbc_w option')
