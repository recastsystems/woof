"""Native physics configuration at the reused CUDA dynamics boundary.

MPAS source: mpas_atm_core.F:1036-1040; mpas_atm_time_integration.F:2140-2176.
Physics selection belongs to the native session; dynamics retain their checks.
"""
from dataclasses import dataclass, fields

from ..config_v841 import V841DryDycoreConfig


@dataclass(frozen=True, slots=True)
class NativeMpasDycoreConfigV842(V841DryDycoreConfig):
    """Preserve native suite metadata; mpas_atm_core.F:1036-1040."""

    # Read only by the LES dissipation frame (cuda_les_v842): u, w and scalar
    # dissipation pass it to mpas_atm_dissipation_models.F (Registry default
    # true). Carried here so an LES namelist's value reaches the kernels.
    config_mix_full: bool = True
    # The &damping level counts and timescale MPAS-A v8.4.2 reads beside config_mpas_cam_coef and
    # config_rayleigh_damp_u (Registry.xml:306-325, mpas_atm_time_integration.F:6400-6402 and :6608-6630), with
    # the Registry defaults.  The dycore driver applies them (cuda_driver.py, the CAM viscosity floor and the
    # horizontal Rayleigh tendency); without them on this configuration a namelist or --dynamics value was refused
    # (fork jacobbieker/woof PR 509).
    config_number_cam_damping_levels: int = 4
    config_number_rayleigh_damp_u_levels: int = 6
    config_rayleigh_damp_u_timescale_days: float = 5.0

    def validate(self, *, les_dissipation=None) -> None:
        """Validate dynamics separately; mpas_atm_time_integration.F:2140-2176."""
        values = {field.name: getattr(self, field.name)
                  for field in fields(V841DryDycoreConfig)}
        # Physics has already been resolved/checked by mpasphys control and is
        # supplied through the staged callbacks, independently of ArWen's
        # held-column adapter. Only the temporary validation view is dry-suite.
        values['config_physics_suite'] = 'none'
        V841DryDycoreConfig(**values).validate(les_dissipation=les_dissipation)
        import math
        from ..errors import ConfigurationRefusal
        for knob in ("config_number_cam_damping_levels", "config_number_rayleigh_damp_u_levels"):
            value = getattr(self, knob)
            if type(value) is not int or value < 1:
                raise ConfigurationRefusal(knob, value, "the native damping loop counts down from the top level",
                                           "a positive integer")
        days = self.config_rayleigh_damp_u_timescale_days
        if type(days) not in (int, float) or not math.isfinite(days) or days <= 0:
            raise ConfigurationRefusal("config_rayleigh_damp_u_timescale_days", days,
                                       "the native Rayleigh coefficient divides by the timescale", ">0")
