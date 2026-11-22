"""Aerosol mass transport on the MPAS-A v8.4.2 suite route: the WOOF-ArWen route's chemistry, same operator.

``forecast --chemistry-config FILE`` appends one dry-air mass mixing ratio per configured species to the
atmosphere.  On the WOOF-ArWen route (``woof.hex.chem_transport``) those rows ride the dycore's monotonic scalar
transport and, after each step, the conservative implicit column operator (PBL mixing with the active
scheme's diffusivity, downward settling, dry deposition at the ground, surface emissions).  This module
gives the suite route the same thing with the same code:

* the species rows are appended after the Registry's scalars in the dycore's ``state.scalars`` (the MPAS
  physics never sees them: every exchange with the physics pools takes the Registry rows only, and the
  physics scalar sources carry zero for the appended rows);
* after each completed step, ``woof.hex.chem_transport.ChemistryTransport.step`` runs on the appended rows
  with the step-end dry density and interface heights and MYNN's ``kzh`` (the suite's eddy diffusivity for
  heat, MPAS's ``exch_h``);
* initial composition, emissions and their coverage checks are ``woof.hex.chem_emissions.attach`` (the
  provider contract: dated frames bound to the grid's SHA-256, missing coverage refused);
* every history frame carries the mixing ratios and the per-species column mass, surface concentration,
  AOD proxy, dry deposition and settled mass, plus the ``CHEMISTRY`` attribute; ``chemistry-receipt.json``
  records the mass closure; a checkpoint carries the ledgers and a restart refuses another configuration.

THE BREAKAGE THIS FIXES: ``--chemistry-config`` with an MPAS suite was accepted and dropped without a word
(no tracer, no attribute, no receipt, output bitwise the plain run's; HEX-PHYSICS-PARITY run group 6 #24).
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _Row:
    """The species-row surface ChemistryTransport and chem_emissions read: Registry scalars, then species."""

    def __init__(self, registry, species):
        self.registry = tuple(registry)
        self.species = self.registry + tuple(species)

    def names(self):
        return self.species

    def index_of(self, name):
        return self.species.index(name)


class SuiteChemistry:
    """Owner of the appended species rows on the suite route."""

    def __init__(self, path, *, registry_names, mesh, nlev, grid_path, start_text):
        from ..chem_transport import RESERVED_FIELDS, load_config
        path, raw, species = load_config(path)
        names = tuple(s.name for s in species)
        overlap = set(names) & (set(registry_names) | RESERVED_FIELDS)
        if overlap:
            raise ValueError(f'chemistry species overlap the MPAS suite scalars or fields: {sorted(overlap)}')
        self.path, self.raw, self.species = path, raw, species
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.registry_names = tuple(registry_names)
        self.names = names
        self.row = _Row(registry_names, names)
        self.n_registry = len(self.registry_names)
        self.nlev, self.ncells = int(nlev), int(mesh.n_cells)
        area = np.asarray(mesh.arrays['areaCell'], dtype=np.float64)[:self.ncells]
        if float(area.sum()) < 4 * math.pi * 1.001:
            # A unit-sphere mesh (sphere_radius = 1) stores areas in radians squared.
            area = area * float(mesh.attrs.get('sphere_radius_m', 6371229.0)) ** 2
        self.area = area
        self.grid_path = Path(grid_path)
        self.start_text = start_text
        self.transport = None
        self.restart = None

    # ------------------------------------------------------------------ before the dycore exists
    def extend_state(self, state):
        """Append zero rows for the species to the host scalar block (the provider binds the start)."""
        if state.scalars.shape[0] == self.n_registry + len(self.names):
            return state      # a checkpoint's state already holds them
        if state.scalars.shape[0] != self.n_registry:
            raise ValueError('the start state scalar block differs from the MPAS Registry declaration')
        extra = np.zeros((len(self.names),) + tuple(state.scalars.shape[1:]), dtype=state.scalars.dtype)
        state.scalars = np.ascontiguousarray(np.concatenate([state.scalars, extra], axis=0))
        return state

    # ------------------------------------------------------------------ once the dycore exists
    def bind(self, *, dycore, cache, coupling):
        from ..chem_transport import ChemistryTransport
        config = (self.path, self.raw, self.species, self.row, self.digest)
        self.transport = ChemistryTransport(config, cache=cache, area=self.area,
                                            nlev=self.nlev, ncells=self.ncells)
        self.coupling, self.dycore = coupling, dycore
        self.transport.driver = SimpleNamespace(atmosphere=None)
        self._refresh_driver()
        if self.restart is not None:
            self._restore(self.restart)
        self._attach_inputs()

    def _refresh_driver(self):
        self.transport.driver.atmosphere = self.dycore.atmosphere

    def _attach_inputs(self):
        """The WOOF-ArWen route's attach (chem_transport.attach), on this route's grid authority."""
        from .. import chem_emissions
        from datetime import datetime, timezone
        runtime = self.transport
        if not runtime.raw.get('inputs'):
            raise ValueError('chemistry forecast requires dated initial composition and emission coverage in its '
                             'inputs block')
        source_start = chem_emissions.instant(runtime.raw['inputs']['valid_time'])
        weather_start = int(datetime.strptime(self.start_text, '%Y-%m-%d_%H:%M:%S')
                            .replace(tzinfo=timezone.utc).timestamp())
        grid_hash = chem_emissions.digest(self.grid_path)
        host = dict(prepared=SimpleNamespace(input_bytes={'grid': {'sha256': grid_hash}}), regional=None)
        if self.restart is None:
            if source_start != weather_start:
                raise ValueError('chemistry inputs and weather initialization valid times differ')
        if runtime.raw['inputs']['mesh_sha256'] != grid_hash:
            raise ValueError('chemistry inputs and weather grid SHA-256 differ')
        # attach binds the emission/boundary update to model time since the provider's start; a restart's
        # dycore clock counts from the case start too (the carried time_seconds), so the update is the same.
        runtime.provider_receipt = chem_emissions.attach(SimpleNamespace(chemistry=runtime), host)
        if (runtime.restart_provider_receipt is not None
                and runtime.restart_provider_receipt != runtime.provider_receipt):
            raise ValueError('chemistry restart source-file identity differs')
        import cupy as cp
        scalars = self.dycore.atmosphere.state.scalars
        if runtime.steps == 0:
            for s in runtime.species:
                if s.name in runtime.initial:
                    values = cp.asarray(runtime.initial[s.name], dtype=cp.float32)
                    target = scalars[self.row.index_of(s.name)]
                    if values.shape != target.shape or not bool(cp.all(cp.isfinite(values) & (values >= 0))):
                        raise ValueError(f'chemistry IC for {s.name} has invalid shape or mass')
                    target[...] = values
        if runtime.initial_mass_kg is None:
            products = self.products()
            runtime.initial_mass_kg = [float(cp.sum(products[s.name + '_column_mass'] * runtime.area))
                                       for s in runtime.species]

    # ------------------------------------------------------------------ registry views
    def registry_view(self, atmosphere):
        """The atmosphere with the Registry's scalar rows only (what the MPAS physics preparation reads)."""
        return replace(atmosphere, state=replace(atmosphere.state,
                                                 scalars=atmosphere.state.scalars[:self.n_registry]))

    def pad_sources(self, cp, scalars):
        """Physics scalar sources with zero rows for the species (they have no MPAS physics source)."""
        extra = cp.zeros((len(self.names),) + tuple(scalars.shape[1:]), dtype=scalars.dtype)
        return cp.ascontiguousarray(cp.concatenate([scalars, extra], axis=0))

    def _columns(self, atmosphere):
        from ..cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        coupling = self.coupling
        return prepare_mpas_to_phys_cuda_v841(self.registry_view(atmosphere), scalar_names=coupling.names,
                                              geometry=coupling.geometry, kernel_cache=coupling.physics_cache,
                                              registry_scalar_names=coupling.names)

    # ------------------------------------------------------------------ per step
    def step(self, *, session, dt):
        """After the completed step: mixing, settling, deposition and emissions on the species rows."""
        self._refresh_driver()
        atmosphere = self.dycore.atmosphere
        columns = self._columns(atmosphere)
        if ('diag_physics', 'kzh') not in getattr(session.native, '_fields', {}) and not self._has_kzh(session):
            raise ValueError('the active PBL publishes no kzh (exch_h) for chemistry mixing')
        kh = session.field('diag_physics', 'kzh')[:self.nlev, :self.ncells]
        self.transport.step(atmosphere, rho=columns.rho_dry, z=columns.z_p, kh=kh, dt=float(dt))
        self.transport.commit()

    @staticmethod
    def _has_kzh(session):
        try:
            session.field('diag_physics', 'kzh')
            return True
        except (KeyError, RuntimeError):
            return False

    def products(self):
        self._refresh_driver()
        atmosphere = self.dycore.atmosphere
        columns = self._columns(atmosphere)
        return self.transport.products(atmosphere, columns.rho_dry, columns.z_p)

    # ------------------------------------------------------------------ history
    def history_fields(self):
        """Extra history variables for one frame: name -> (array, dims, units), on the host."""
        from ..chem_transport import history_units
        import cupy as cp
        scalars = self.dycore.atmosphere.state.scalars
        out = {}
        for s in self.species:
            value = cp.asnumpy(scalars[self.row.index_of(s.name)][:, :self.ncells])
            out[s.name] = (np.ascontiguousarray(value), ('nVertLevels', 'nCells'), 'kg kg-1')
        for name, value in self.products().items():
            out[name] = (np.ascontiguousarray(cp.asnumpy(value)[:self.ncells]), ('nCells',),
                         history_units(name) or '')
        return out

    def history_attribute(self):
        from ..chem_transport import history_metadata
        return history_metadata(self.path, self.digest)

    # ------------------------------------------------------------------ checkpoint
    def restart_state(self):
        t = self.transport
        return dict(config_sha256=self.digest, names=list(self.names), dry=t.dry.get(),
                    settled=t.settled.get(), emitted=t.emitted.get(), steps=t.steps,
                    initial_mass_kg=t.initial_mass_kg, provider=t.provider_receipt)

    def _restore(self, chem):
        import cupy as cp
        t = self.transport
        if chem is None or chem.get('config_sha256') != self.digest or list(chem.get('names', [])) != list(self.names):
            raise ValueError('chemistry restart config identity differs')
        for name in ('dry', 'settled', 'emitted'):
            array = cp.asarray(chem[name])
            target = getattr(t, name)
            if (array.shape != target.shape or array.dtype != target.dtype
                    or not bool(cp.all(cp.isfinite(array) & (array >= 0)))):
                raise ValueError(f'invalid chemistry restart {name}')
            target[...] = array
        steps = chem['steps']
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('invalid chemistry restart step count')
        t.steps = steps
        t.initial_mass_kg = list(chem['initial_mass_kg'])
        t.restart_provider_receipt = chem.get('provider')

    # ------------------------------------------------------------------ receipt
    def write_ledger(self, output):
        import cupy as cp
        t = self.transport
        totals = {key: cp.sum(getattr(t, name) * t.area, axis=1).get().tolist()
                  for key, name in (('dry_deposition_kg', 'dry'), ('settled_kg', 'settled'),
                                    ('emitted_kg', 'emitted'))}
        fields = self.products()
        totals['final_mass_kg'] = [float(cp.sum(fields[s.name + '_column_mass'] * t.area)) for s in t.species]
        totals['initial_mass_kg'] = t.initial_mass_kg
        totals['mass_closure_kg'] = [final + dry + settled - initial - emitted
                                     for final, dry, settled, initial, emitted in zip(
                                         totals['final_mass_kg'], totals['dry_deposition_kg'],
                                         totals['settled_kg'], t.initial_mass_kg, totals['emitted_kg'])]
        totals['surface_max_ug_m3'] = [float(cp.max(fields[s.name + '_surface_concentration']))
                                       for s in t.species]
        receipt = {**t.receipt(), **totals,
                   'route': 'mpas-suite (species rows after the Registry scalars; MYNN kzh mixing)'}
        Path(output, 'chemistry-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt
