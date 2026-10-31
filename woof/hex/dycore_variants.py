"""Table-resolved dycore numerics. Suite names and source pins remain table data.

RRFS variant: RRFSx/MPAS-Model src/core_atmosphere/dynamics/mpas_atm_time_integration.F
at 1d6e370e. An absent dycore entry preserves the exact incoming configuration.
"""
from dataclasses import fields
from typing import Mapping, Any
from .config_rrfs import RRFSDryDycoreConfig, NoaaV841DycoreConfig
from .config_v841 import V841DryDycoreConfig

_CONFIGURATIONS = {'mpas-v8.3.1-noaa': RRFSDryDycoreConfig,
                   'mpas-v8.4.1-noaa': NoaaV841DycoreConfig}

# MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:1-170 at 1d6e370e.
def resolve_dycore_variant(config: V841DryDycoreConfig, suite_row: Mapping[str, Any] | None,
    *, overrides: Mapping[str, Any] | None = None):
    """Resolve one already-selected table row; no suite name or pin dispatch."""
    entry=None if suite_row is None else suite_row.get('dycore')
    if entry is None:return config
    if not isinstance(entry,Mapping):raise TypeError('suite dycore must be a mapping')
    variant=entry.get('variant')
    try:kind=_CONFIGURATIONS[variant]
    except (KeyError,TypeError):raise ValueError(f'dycore variant {variant!r} has no numerical implementation') from None
    if not isinstance(config,V841DryDycoreConfig):raise TypeError('table dycore selection requires an MPAS configuration')
    base={f.name:getattr(config,f.name) for f in fields(V841DryDycoreConfig)}
    # Variant-only defaults, then the table's matched deck settings, then
    # explicit run inputs. The caller passes imported namelist/CLI overrides.
    base.update(config_epssm=kind().config_epssm)
    if kind is NoaaV841DycoreConfig:
        base.update(source_commit=str(suite_row.get('source_commit', suite_row.get('model_commit', ''))),
                    physics_pin_set=str(suite_row.get('pin_set', '')))
    defaults=entry.get('defaults',{})
    allowed={f.name for f in fields(kind)}-{'source_release','dycore_variant'}
    for source in (defaults,overrides or {}):
        if not isinstance(source,Mapping):raise TypeError('dycore defaults and overrides must be mappings')
        unknown=set(source)-allowed
        if unknown:raise ValueError(f'dycore row contains unknown settings: {sorted(unknown)}')
        base.update(source)
    resolved=kind(**base)
    resolved.validate()
    return resolved

# MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:1-170 at 1d6e370e.
def dycore_driver_type(config):
    """Resolve implementation from the configuration produced by the table."""
    if isinstance(config,RRFSDryDycoreConfig):
        from .cuda_driver_rrfs import CudaRRFSDycoreDriver
        return CudaRRFSDycoreDriver
    from .cuda_driver import CudaDryDycoreDriver
    return CudaDryDycoreDriver

# MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:1-170 at 1d6e370e.
def dycore_kernel_cache_type(config):
    """Select the compiler together with its numerical implementation."""
    if isinstance(config,(RRFSDryDycoreConfig, NoaaV841DycoreConfig)):
        from .cuda_compile_rrfs import RRFSKernelCache
        return RRFSKernelCache
    from .cuda_backend import KernelCache
    return KernelCache

# MPAS-Model-rrfs/src/core_atmosphere/Registry.xml:1-170 at 1d6e370e.
def dycore_regional_constructor(config):
    """Regional construction follows the same table-produced configuration."""
    if isinstance(config,RRFSDryDycoreConfig):
        from .cuda_regional_forecast_rrfs import assemble_regional_driver_rrfs
        return assemble_regional_driver_rrfs
    from .cuda_regional_forecast_v841 import open_regional_forecast_v841
    return open_regional_forecast_v841
