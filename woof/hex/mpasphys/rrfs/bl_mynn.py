"""RRFS MYNN: shared MMM a4baf7f numerics with fork configuration binding.

MPAS-Model-rrfs 1d6e370e driver_pbl.F:1442-1443 reads two configuration
integers where v8.4.2 uses zero-valued module parameters. The selected MYNN
preparation, numerical call, and writeback otherwise match v8.4.2.
"""
from .. import bl_mynn_v842 as shared


def bind_native_pools(provider, call):
    packet = shared.bind_native_pools(provider, call)
    arguments = packet["mynn_v842"]["arguments"]
    for name in ("icloud_bl", "spp_pbl"):
        arguments[name] = int(provider["configs"]["config_" + name])
    return packet


def run_slot(call):
    # The fork also defines config_mynn_sfcflux_land/water for its surface
    # scheme. The MMM PBL routine takes only its own seventeen options.
    from dataclasses import replace
    keys = {'config_mynn_' + field for field in shared.MynnOptions.__dataclass_fields__}
    return shared.run_slot(replace(call, options={key: value for key, value in call.options.items()
                                                 if key in keys}))
