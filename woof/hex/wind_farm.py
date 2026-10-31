"""Opt-in Fitch turbine forcing on the model's spherical Voronoi cells."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class WindFarmConfig:
    wind_farms: str = "off"
    windfarm_tke_factor: float = .25


def source_table(choice, *, year=0):
    if choice in (None, "", "off"):
        return None
    if str(choice) == "uswtdb":
        from woof.wind_inputs import inventory_table
        return str(inventory_table(year))
    return str(Path(choice).expanduser().resolve())


def require_mynn(pbl, *, interval=None):
    if pbl not in (5, "bl_mynn"):
        raise ValueError(f"Fitch turbines require MYNN; PBL {pbl!r} has no coupled turbine TKE source")
    if interval not in (None, "none", "", 0, 0.0):
        raise ValueError("Fitch turbines require an every-step PBL: a held momentum tendency would repeat drag without its matching TKE source")


class WindFarm:
    def __init__(self, choice, *, dt, latitude, longitude, area, interfaces, factor=.25):
        from woof.core.wind_farm import attach_voronoi_wind_farm
        table = source_table(choice)
        self.farm = attach_voronoi_wind_farm(
            SimpleNamespace(wind_farm_table=table, dt=dt, windfarm_tke_factor=factor),
            latitude=latitude, longitude=longitude, area=area, z_interface=interfaces)

    def history_fields(self):
        if self.farm is None:
            return {}
        import cupy as cp
        from woof.io.wrf_output_schema import REGISTRY_VAR_META
        out = {}
        for name, value in self.farm.history_fields().items():
            if name in ("POWER", "WIND_TKE_SOURCE_MAX"):
                value = value.reshape(-1)
                dims = ("nCells",)
            elif name == "WIND_TKE_SOURCE":
                value = value.reshape(self.farm.shape[0], -1)
                dims = ("nVertLevels", "nCells")
            else:
                dims = ("wind_turbine" if name.startswith("TURBINE_") else "wind_farm",)
            out[name] = (cp.asnumpy(value), dims, REGISTRY_VAR_META[name][1])
        return out

    def metadata(self):
        if self.farm is None:
            return json.dumps({"enabled": False, "reason": "empty turbine table"})
        farm = self.farm
        return json.dumps({"schema": "gpuwm-hex.wind-farms/v1", "scheme": "Fitch",
                           "turbine_count": len(farm.ids), "farm_count": len(farm.farm_ids),
                           "table_sha256": farm.table_sha256, "options": farm.options,
                           **getattr(self, '_metadata_reference', {})}, sort_keys=True)

    def write_metadata(self, directory):
        """Keep inventory-sized IDs outside HDF5's 65535-byte compact attributes."""
        if self.farm is None:
            return
        farm = self.farm
        document = {"schema": "gpuwm-hex.wind-farm-index/v1",
                    "turbine_ids": farm.ids, "farm_ids": farm.farm_ids,
                    "table_sha256": farm.table_sha256, "options": farm.options,
                    "input_provenance": farm.data.get('provenance', {})}
        payload = (json.dumps(document, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')
        path = Path(directory) / 'wind-farms.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
        self._metadata_reference = {'index_file': path.name,
                                    'index_sha256': hashlib.sha256(payload).hexdigest()}


def attach_native(arguments, session, mesh, output, *, dt, resident, coupling, halo=None):
    choice = getattr(arguments, "wind_farms", None)
    if choice in (None, '', 'off'):
        return None
    require_mynn(session.config.selectors.get("pbl"),
                 interval=session.config.options.get("config_pbl_interval"))
    if not resident or halo is not None:
        raise ValueError("Fitch turbines require resident single-rank cells; partition ownership has not been assigned and would duplicate turbine forcing")
    if session.layout.configs.get("config_apply_lbcs", False):
        raise ValueError("Fitch turbine mapping requires the complete Voronoi globe; a culled boundary needs explicit outside-domain ownership to avoid placing distant turbines on edge cells")
    p = session.device_pools
    nc = session.layout.dimensions['nCellsSolve']
    table = source_table(choice, year=int(str(session.layout.configs['config_start_time'])[:4]))
    farm = WindFarm(table, dt=dt, latitude=p.field('mesh', 'latCell')[:nc],
                    longitude=p.field('mesh', 'lonCell')[:nc], area=p.field('mesh', 'areaCell')[:nc],
                    interfaces=p.field('mesh', 'zgrid')[:, :nc])
    p.wind_farm = farm.farm
    if farm.farm is not None:
        from .cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        columns = prepare_mpas_to_phys_cuda_v841(coupling.dycore.atmosphere,
            scalar_names=coupling.names, geometry=coupling.geometry, kernel_cache=coupling.physics_cache)
        farm.farm.sample({'z_interface':columns.z_p, 'u':columns.u_p,
                          'v':columns.v_p, 'rho':columns.rho_p}, {})
    previous = getattr(output, 'extra_history', None)
    farm.write_metadata(output.output)
    output.extra_history = lambda: {**(previous() if previous else {}), **farm.history_fields()}
    output.extra_history_attrs = {**(getattr(output, 'extra_history_attrs', None) or {}),
                                  'WIND_FARMS': farm.metadata()}
    return farm


def apply_native(provider, fields):
    farm = getattr(provider, 'wind_farm', None)
    if farm is None:
        return
    nz = farm.shape[0]
    atmosphere = {name: fields[key][:, :nz].T for name, key in
                  (('u', 'u'), ('v', 'v'), ('rho', 'rho'))}
    atmosphere['z_interface'] = provider.prepared('z_p')[:nz+1]
    farm.apply_mynn(atmosphere,
        {'qke': fields['qke'][:, :nz].T, 'qke_adv': fields['qke_adv'][:, :nz].T},
        {'du': fields['rublten'][:, :nz].T, 'dv': fields['rvblten'][:, :nz].T},
        step_seconds=fields['delt'])


def attach_arwen(backend, host, driver, *, output_root=None):
    choice = host.get('wind_farms')
    if choice in (None, '', 'off'):
        return
    seam = backend._seam
    require_mynn(seam._cfg.bl_pbl_physics, interval=seam._cfg.bldt)
    if host.get('regional') is not None:
        raise ValueError("Fitch turbine mapping requires complete Voronoi ownership; a culled boundary could assign outside-domain turbines to edge cells")
    mesh = host['prepared'].mesh
    arrays = getattr(mesh, 'arrays', mesh)
    from woof.hex.cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
    columns = prepare_mpas_to_phys_cuda_v841(driver.atmosphere,
        scalar_names=host['scalar_names'], geometry=backend._prep_geometry,
        kernel_cache=backend._kernel_cache)
    table = source_table(choice, year=int(host['start_time_text'][:4]))
    wind = WindFarm(table, dt=seam._cfg.dt, latitude=arrays['latCell'],
                    longitude=arrays['lonCell'], area=arrays['areaCell'], interfaces=columns.z_p)
    if wind.farm is not None:
        wind.farm.sample({'z_interface':columns.z_p, 'u':columns.u_p,
                          'v':columns.v_p, 'rho':columns.rho_p}, {})
    seam._driver.wind_farm = wind.farm
    if output_root is not None:
        wind.write_metadata(output_root)
    backend.wind_farms = wind
