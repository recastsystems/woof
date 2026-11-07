"""Public checkpoint flags over the existing sealed MPAS physics carrier."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .moving_v842 import DYCORE_POOLS, MovingRoute, TIME_FORMAT, load_carrier

#: Physics manager words a run sets once from its own start (manager.F physics_run_init): the start hour.
CLOCK_START_WORDS = ('mpas_atmphys_manager.gmt',)


def persistent_names(config):
    """The (pool, name) words the active schemes declare as their persistent state (module PERSISTENT_FIELDS).

    A scheme module names the pool words it reads back on its next call (RUC: ``{pool: names}``; MYNN-EDMF:
    ``{argument: name}``, pool None = any physics pool), so the set comes from the suite table and the scheme
    modules, never from a list here.  The Registry restart stream does not hold all of them: the RRFS fork's
    misses RUC's frozen-soil flag ``keepfr3d`` (Registry.xml:616-918 at 1d6e370e; keepfr3dflag is inout in
    module_ruc_land.F90).
    """
    from importlib import import_module
    from ..physics_suites import load_table
    slots = load_table().get(getattr(config, 'table_key', 'mpasphys_v842'), {}).get('slots', {})
    names = set()
    for slot, scheme in (getattr(config, 'selectors', None) or {}).items():
        if not config.active(slot):
            continue
        entry = (slots.get(slot, {}).get('cuda_entries') or {}).get(scheme) or {}
        if not entry.get('module'):
            continue
        declared = getattr(import_module('.' + entry['module'], __package__), 'PERSISTENT_FIELDS', None) or {}
        for key, value in declared.items():
            if isinstance(value, str):
                names.add((None, value))
            else:
                names.update((key, name) for name in value)
    return frozenset(names)



def rank_continuation(session):
    """One session's continuation words (CheckpointRoute.continuation_state): its schemes' boolean SAVE flags,
    its output diagnostics' running accumulators (its own cells) and the physics start hour."""
    native = getattr(session, 'native', None)
    scheme_state = getattr(native, 'scheme_state', None) or {}
    flags = {scheme: {key: value for key, value in state.items() if isinstance(value, bool)}
             for scheme, state in scheme_state.items() if isinstance(state, dict)}
    diagnostics = getattr(getattr(session, 'output_streams', None), 'diagnostics', None)
    accumulators = getattr(diagnostics, 'accumulator_state', None)
    clock = {}
    for name in CLOCK_START_WORDS:
        try:
            clock[name] = np.array(native.get_table(name)[0], copy=True)
        except (AttributeError, KeyError, RuntimeError):
            pass
    window = getattr(session, 'iau_window', None)
    return dict(scheme_flags={k: v for k, v in flags.items() if v},
                output_accumulators=accumulators() if callable(accumulators) else None,
                clock_start=clock,
                iau_window=None if window is None else window.checkpoint_state())

def _seconds(hours):
    values = [Fraction(value.strip()) * 3600 for value in str(hours).split(',')]
    result = tuple(sorted({Fraction(round(value)) for value in values}))
    if (not result or any(value <= 0 for value in result)
            or any(abs(value-round(value)) > Fraction(1, 10**9) for value in values)):
        raise ValueError('--checkpoint-hours needs positive integral-second boundaries')
    return result


def _identity(inputs):
    mesh, grid, _, _, reference, _, names = inputs
    arrays = [(f'mesh/{name}', value) for name, value in sorted(mesh.arrays.items())
              if isinstance(value, np.ndarray)]
    arrays += [(f'vertical/{name}', getattr(grid.vertical_grid, name))
               for name in ('zgrid', 'dzw')]
    arrays += [(f'reference/{name}', getattr(reference, name))
               for name in ('rho_base', 'rho_theta_base', 'pressure_base', 'exner_base')]

    def digest(item):
        name, value = item
        value = np.ascontiguousarray(value)
        return name, dict(shape=list(value.shape), dtype=value.dtype.str,
                          sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())

    # Bind large static arrays before CUDA, without a serial hash queue.
    with ThreadPoolExecutor(max_workers=min(8, len(arrays))) as pool:
        result = dict(pool.map(digest, arrays))
    return dict(arrays=result, scalar_names=list(names))


@dataclass
class CheckpointRoute(MovingRoute):
    checkpoint_seconds: tuple = ()
    cli_restart: bool = False
    da_analysis: bool = False
    checkpoint_dt: str | None = None
    checkpoint_identity: dict | None = None
    checkpoint_physics: dict | None = None
    restart_metadata: dict | None = None
    checkpoints: list = field(default_factory=list)
    checkpoint_preparation: dict | None = None
    frame_from_cli: bool = False
    restart_dt_change: dict | None = None
    restart_corrections_change: dict | None = None
    #: A correct-mode same-mesh resume continues the straight run (bind_inputs): see restore_continuation.
    resume_continues: bool = False
    continuation_receipt: dict | None = None
    #: A checkpoint/restart/DA route on the run's own mesh, with no WOOF_HEX_MOVING_* switch: not a move, so a
    #: limited-area run keeps its --lbc-dir series (forecast_v842, runtime_v842; fix/hex-lam-da-v842).
    fixed_mesh: bool = False
    #: The run's CyclingExecution (features_v842), when it writes JEDI backgrounds: a checkpoint sealed at a
    #: background's instant names that background, the uncoupled words an analysis of it is compared with.
    cycling: object = None
    #: A moving segment resumed on its own member (WOOF_HEX_MOVING_CHECKPOINT_IN, no operator): crash recovery,
    #: a stay on the carrier member, a controller --resume.  It continues the run that sealed the carrier, as a
    #: --restart-from does; its --hours stays the segment's length (resolve_schedule).
    moving_resume: bool = False
    #: A correct-mode native MPAS restart (config_do_restart) from a restart file this engine wrote: it resumes the
    #: continuation sealed beside that file (``CONTINUATION_SUFFIX``), as a --restart-from does; its duration stays
    #: the restart's own (native semantics) and its pools still read the restart stream.
    native_restart: bool = False
    #: The restart stream's output alarm (seconds, filename template), when this correct-mode run writes native
    #: restart files: each one gets its continuation sealed beside it (``restart_instants``).
    restart_output: tuple | None = None


    def mark_restart(self, layout):
        super().mark_restart(layout)
        if self.da_analysis:
            layout.configs['config_do_DAcycling'] = True

    def resume_namelist(self, text):
        """A native restart's namelist already starts at the restart's valid time (MPAS restart semantics)."""
        from .forecast_v842 import namelist_option
        if self.native_restart:
            named = namelist_option(text, 'nhyd_model', 'config_start_time')
            if named is None or str(named).strip() != self.carrier['valid_time']:
                raise ValueError(f"the restart's continuation is valid at {self.carrier['valid_time']}; this "
                                 f"restart starts {named}")
            return text
        return super().resume_namelist(text)

    def resolve_schedule(self, duration, dt, streams):
        """CLI --hours is the original end, not another duration after resume."""
        dt, duration = Fraction(str(dt)), Fraction(str(duration))
        if dt > 0 and abs(duration/dt-round(duration/dt)) <= Fraction(1, 10**9):
            duration = round(duration/dt) * dt
        if not self.frame_from_cli:
            from .driver_v842 import interval_seconds
            history = next((stream for stream in streams if stream.name in ('output', 'history')
                            and 'output' in stream.directions), None)
            if history is None or history.output_interval == 'none':
                raise ValueError('native checkpoints need a recurring history stream')
            self.frame_seconds = float(interval_seconds(history.output_interval))
        offset = Fraction(str(self.offset_seconds))
        if dt <= 0 or offset % dt:
            raise ValueError('native checkpoint must lie on an outer-step boundary')
        # A resume may change the timestep (e.g. to run past an instability). The carrier holds
        # model state at one instant: prognostic and diagnostic fields, physics rates per second
        # and the physics alarms' phase in model seconds, none of it a per-step quantity, so it is
        # read the same under any step that lands on the checkpoint's instant. Native MPAS-A
        # restarts under a changed config_dt the same way. Only the instant must be a whole number
        # of new steps from the start, which the offset check above enforces.
        if self.restart_metadata is not None and Fraction(self.restart_metadata['dt']) != dt:
            self.restart_dt_change = dict(checkpoint_dt=str(Fraction(self.restart_metadata['dt'])),
                                          resumed_dt=str(dt))
            print(f"woof hex: resuming checkpoint written at dt {Fraction(self.restart_metadata['dt'])} s "
                  f"under dt {dt} s", flush=True)
        self.checkpoint_dt = str(dt)
        if self.cli_restart:
            duration -= offset
        if duration <= 0:
            raise ValueError('--restart-from must precede the original forecast end')
        end = offset + duration
        for boundary in self.checkpoint_seconds:
            if boundary > end or boundary % dt or boundary % Fraction(str(self.frame_seconds)):
                raise ValueError('--checkpoint-hours must name outer-step history frames before the forecast end')
        self.restart_seals = {}
        if self.restart_output is not None:
            # MPAS rings the restart stream's alarm every interval from this run's start (a restart's start
            # included).  Each instant on the outer step seals its continuation beside the restart file.
            interval, template = self.restart_output
            instant = offset + interval
            while instant <= end:
                if not instant % dt:
                    self.restart_seals[instant] = template
                instant += interval
        return duration

    def bind_inputs(self, inputs, config):
        began = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat()
        from .. import numerics
        physics = dict(config=config.receipt(), numerics=numerics.mode())
        physics = json.loads(json.dumps(physics))
        if self.restart_metadata is not None and physics != self.restart_metadata.get('physics'):
            # The applied correct-mode findings may differ (a release that adds a correction, e.g. RP7):
            # a correction changes scheme arithmetic from the resume on, never the carried state's
            # fields or layout. Suite, schemes, options, backends, mode and numerics must still match.
            def _scheme_identity(value):
                value = json.loads(json.dumps(value))
                config = value.get('config') if isinstance(value, dict) else None
                if isinstance(config, dict):
                    for key in ('corrections', 'correct_skip', 'correct_mode'):
                        config.pop(key, None)
                return value
            carried = self.restart_metadata.get('physics')
            if _scheme_identity(physics) != _scheme_identity(carried):
                raise ValueError('native checkpoint physics configuration or numerics differ')
            before = (carried or {}).get('config', {}).get('corrections', [])
            after = physics['config'].get('corrections', [])
            self.restart_corrections_change = dict(checkpoint=list(before), resumed=list(after))
            print(f"woof hex: resuming checkpoint written with corrections {before} under {after}", flush=True)
        identity = _identity(inputs)
        if self.restart_metadata is not None and identity != self.restart_metadata['identity']:
            carried = self.restart_metadata['identity']
            ours, theirs = identity['arrays'], carried.get('arrays', {})
            names = sorted(name for name in set(ours) & set(theirs) if ours[name] != theirs[name])
            names += sorted(f'{name} (only here)' for name in set(ours) - set(theirs))
            names += sorted(f'{name} (only in the checkpoint)' for name in set(theirs) - set(ours))
            if identity.get('scalar_names') != carried.get('scalar_names'):
                names.append('scalar_names')
            raise ValueError('native checkpoint mesh, vertical grid, base state or species differ: '
                             + ', '.join(names[:40]))
        self.checkpoint_identity = identity
        self.checkpoint_physics = physics
        self.persistent = persistent_names(config)
        # Correct mode (the door default) continues the straight run from a same-mesh checkpoint, as it carries the
        # physics alarms' phase (MovingRoute.restore_alarm_phase).  Exact mode keeps native MPAS's restart (its
        # itimestep starts at 1 again, mpas_atm_core.F:756 at 1d6e370e).  A DA analysis in correct mode continues the
        # physics clock too: the analysis replaces atmospheric words, not the physics' memory, and native's restart
        # re-initialisation (MYNN-SFC's restart branch, RUC ktau = 1, RRTMG's hour angle) is the restart defect the
        # continuation fixes, not part of an analysis.  An IAU window still begins anew (config_do_DAcycling).
        # THE BREAKAGE THIS FIXES (fix/hex-lam-da-v842): an analysis equal to its background did not continue the
        # forecast it was made from, so no cycle could be told apart from its insert's own disturbance.
        self.resume_continues = (self.carrier is not None
                                 and (self.cli_restart or self.moving_resume or self.native_restart)
                                 and getattr(config, 'mode', 'exact') == 'correct')
        self.checkpoint_preparation = dict(seconds=time.perf_counter()-began,
            started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
            parallelism='up to 8 host hash workers; before CUDA')

    def physics_clock_offset(self, dt):
        """The straight run's physics steps and seconds before this segment, (0, 0) unless a continuation.

        A continuation's physics runs on the straight run's clock: its itimestep and its time since the start
        (xtime_s) run on from the checkpoint.  THE BREAKAGES THIS FIXES: a resume called the physics with
        itimestep 1 again, as native's restart does, so MYNN-SFC took its restart branch (the stored z/L, no
        Li et al. 2010 / zolrib solve: module_sf_mynnsfc flag_restart .and. itimestep > 1) and RUC its ktau = 1
        block: 1.71 M of 3.67 M history words differed from the straight run one hour after a checkpoint (hrrrv5
        x1.2562, rel/hex-0.4.10).  And RRTMG-SW's hour angle, gmt + (fmod(xtime + radt/2, 1440) + equation of
        time)/60 (module_ra_rrtmg_sw.F:10208-10219 at 1d6e370e), rounded differently from a later start hour and a
        shorter xtime: coszr differed by an ulp in 2,213 of 2,563 cells at the second radiation call after a 3 h
        checkpoint, and 1.59 M history words by 6 h.
        """
        if not self.resume_continues:
            return 0, 0.0
        elapsed = self.sealed_physics_seconds()
        steps = elapsed / Fraction(str(dt))
        if steps.denominator != 1:
            raise ValueError('native checkpoint must lie on an outer-step boundary')
        return int(steps), float(elapsed)

    def sealed_physics_seconds(self):
        """The physics clock of the run that sealed the carrier, at the seal (``manager_elapsed``).

        A continuation runs on that clock.  It is the model time since the run's start for a checkpoint of a fresh
        run or of a continuation; a remapped moving segment starts its own physics clock at the move
        (MovingRoute.physics_clock_offset), so its recovery seal's clock is the time since the move, not
        ``offset_seconds``."""
        physics = self.carrier.get('physics') or {}
        return Fraction(str(physics.get('manager_elapsed', self.offset_seconds)))

    def restore_alarm_phase(self, session, config):
        """The carried alarms' phase, on the straight run's clock when the resume continues it."""
        phase = super().restore_alarm_phase(session, config)
        if phase is None or not self.resume_continues:
            return phase
        manager = session.driver.manager
        for alarm in manager.alarms.values():
            alarm.next_ring += self.sealed_physics_seconds()
        return {name: str(alarm.next_ring) for name, alarm in manager.alarms.items()}

    def restore_continuation(self, *, session, diagnostics, dt, accumulators=None):
        """After the physics init: the scheme SAVE flags, accumulators and start hour the straight run holds.

        ``accumulators``: this session's own words when a same-host rank restores (DistributedMoving projects the
        carried global words onto the rank's cells); None takes the carrier's, as on one card.
        """
        if not self.resume_continues:
            return None
        carried = (self.carrier.get('physics') or {}).get('continuation')
        steps, seconds = self.physics_clock_offset(dt)
        receipt = dict(physics_step_offset=steps, physics_seconds_offset=seconds, scheme_flags={},
                       output_accumulators=[], clock_start={})
        if carried is None:
            receipt['note'] = 'checkpoint sealed before continuation state was carried: scheme flags and accumulators start again'
        else:
            for name, value in (carried.get('clock_start') or {}).items():
                table = session.native.get_table(name)[0]
                table[...] = value
                receipt['clock_start'][name] = value.tolist()
            scheme_state = getattr(session.native, 'scheme_state', None) or {}
            for scheme, flags in carried['scheme_flags'].items():
                state = scheme_state.get(scheme)
                if isinstance(state, dict):
                    state.update(flags)
                    receipt['scheme_flags'][scheme] = dict(flags)
            if accumulators is None:
                accumulators = carried.get('output_accumulators')
            if accumulators and diagnostics is not None:
                diagnostics.resume_accumulators(accumulators)
                receipt['output_accumulators'] = sorted(accumulators)
        self.continuation_receipt = receipt
        return receipt

    def checkpoint_due(self, step, dt):
        seconds = Fraction(str(self.offset_seconds)) + step * Fraction(str(dt))
        return step > 0 and (seconds in self.checkpoint_seconds or seconds in getattr(self, 'restart_seals', {}))

    def checkpoint_target(self, step, dt, out):
        """The checkpoint file a due ``step`` seals: checkpoint.fHHH.pkl on the hour, else checkpoint.tSSSSSSSS.pkl;
        at a native restart instant, the restart file's continuation (``<restart file>`` + CONTINUATION_SUFFIX)."""
        exact = Fraction(str(self.offset_seconds)) + step * Fraction(str(dt))
        template = getattr(self, 'restart_seals', {}).get(exact)
        if template is not None:
            from .runtime_v842 import expand_input_filename
            valid = (datetime.strptime(self.origin_time, TIME_FORMAT) + timedelta(seconds=float(exact)))
            return Path(out) / (expand_input_filename(template, valid.strftime(TIME_FORMAT)) + CONTINUATION_SUFFIX)
        seconds = int(exact)
        label = f'f{seconds // 3600:03d}' if seconds % 3600 == 0 else f't{seconds:08d}'
        return Path(out) / f'checkpoint.{label}.pkl'

    def background_at(self, step):
        """The JEDI background this run wrote at ``step`` (cycling_v842.recouple_analysis), if any."""
        return next((dict(path=str(Path(row['path']).resolve()), sha256=row['sha256'])
                     for row in getattr(self.cycling, 'backgrounds', ()) or ()
                     if row and row.get('step') == step), None)

    def checkpoint_metadata(self, background=None):
        """The receipt's ``checkpoint`` block, which --restart-from checks (resolve_route, bind_inputs)."""
        if self.checkpoint_identity is None or self.checkpoint_dt is None:
            raise ValueError('native checkpoint identity/schedule must bind before export')
        metadata = dict(schema='mpas-native-checkpoint/v1', dt=self.checkpoint_dt,
                        identity=self.checkpoint_identity, physics=self.checkpoint_physics)
        if background is not None:
            # The JEDI background this run wrote at the sealed instant (cycling_v842.recouple_analysis).
            metadata['da_background'] = background
        return metadata

    def after_step(self, step, **values):
        stop = super().after_step(step, **values)
        if self.checkpoint_due(step, values['dt']):
            target = self.checkpoint_target(step, values['dt'], values['out'])
            # The step's publication ran before its output event; the event's resets (refl10cm_1km_max, the hail
            # maxima: mpas_atm_core.F:877-890,968-990) changed the device words since.  Seal what the next step
            # reads.  THE BREAKAGE THIS FIXES: a resume carried the last frame's 1 km reflectivity maximum into the
            # next frame (MPAS-A route x1.2562, 5 diag words at the first frame after a checkpoint).
            publish = getattr(values['session'], 'publish', None)
            if callable(publish):
                publish()
            export = {key: value for key, value in values.items() if key not in ('steps', 'columns', 'out')}
            self._background = self.background_at(step)
            try:
                self.checkpoints.append(self.export(target, step=step, **export))
            finally:
                self._background = None
        return stop

    def export(self, path, **values):
        metadata = self.checkpoint_metadata(getattr(self, '_background', None))
        began = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat()
        receipt = super().export(path, **values)
        if values.get('handoff') is not None:
            # A moving carrier this resident process continues in memory (moving_resident_v842): no file to seal.
            return receipt
        receipt['checkpoint'] = metadata
        receipt['checkpoint_export'] = dict(started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.perf_counter()-began,
            parallelism='host carrier serialization and SHA256; native state already published')
        # The payload is the existing carrier, written once. Metadata stays in
        # its small sealed receipt; never decode/rewrite a weather payload.
        Path(path).with_suffix(Path(path).suffix+'.json').write_text(
            json.dumps(receipt, indent=2, default=str)+'\n')
        return receipt

    def receipt(self):
        result = {**super().receipt(), 'native_checkpoints': self.checkpoints,
                  'checkpoint_preparation': self.checkpoint_preparation}
        if self.restart_dt_change is not None:
            result['restart_dt_change'] = self.restart_dt_change
        if self.restart_corrections_change is not None:
            result['restart_corrections_change'] = self.restart_corrections_change
        if self.continuation_receipt is not None:
            result['checkpoint_continuation'] = self.continuation_receipt
        return result


#: The file beside a native restart file that holds the exact continuation a correct-mode restart resumes.
CONTINUATION_SUFFIX = '.hex-continuation.pkl'


def restart_output(streams_path):
    """(interval seconds, filename template) of a deck's native restart output stream, or None."""
    if not streams_path:
        return None
    import xml.etree.ElementTree as ET
    from .driver_v842 import interval_seconds
    try:
        tree = ET.parse(streams_path).getroot()
    except (OSError, ET.ParseError):
        return None
    element = next((e for e in tree if e.get('name') == 'restart'), None)
    if element is None or 'output' not in element.get('type', '').split(';'):
        return None
    interval = element.get('output_interval', 'none')
    if interval in ('none', 'initial_only', '') or not element.get('filename_template'):
        return None
    try:
        seconds = interval_seconds(interval)
    except ValueError:
        return None
    return None if not seconds else (Fraction(seconds), element.get('filename_template'))


#: The valid-time stamp MPAS stream filename templates write ($Y-$M-$D_$h.$m.$s).
import re as _re
_FRAME_STAMP = _re.compile(r'\.(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})\.nc$')


def set_aside_frames_after(out, valid_time):
    """Rename stream files in ``out`` stamped after ``valid_time`` to ``<name>.pre-restart``.

    A resume from a checkpoint regenerates every frame after the checkpoint's valid time (MPAS resets the
    output alarms on a restart, so the frame AT that time is the earlier leg's and is kept).  THE BREAKAGE THIS
    FIXES (IFS-9KM-DEFAULT-2026-11-04): a leg cut after writing frames past its last checkpoint left
    ``history.<valid>.nc`` files that the never_modify writer refused on the resume, so every pool-cut 96 -> 3 km
    run had to have them moved aside by hand.  Kept, not deleted: a ``.pre-restart`` copy from an earlier resume
    is replaced.  Returns the moved names."""
    cut = datetime.strptime(valid_time, TIME_FORMAT)
    moved = []
    for path in sorted(Path(out).glob('*.nc')):
        match = _FRAME_STAMP.search(path.name)
        if match is None or datetime.strptime(match.group(1), '%Y-%m-%d_%H.%M.%S') <= cut:
            continue
        path.replace(path.with_name(path.name + '.pre-restart'))
        moved.append(path.name)
    if moved:
        import sys
        print(f'woof hex: resuming at {valid_time}: {len(moved)} frame(s) after the checkpoint set aside as '
              f'*.pre-restart ({moved[0]} .. {moved[-1]})', file=sys.stderr, flush=True)
    return moved


def native_restart_continuation(arguments, config, initial_stream, initial_file):
    """The continuation beside the native restart file a correct-mode run starts from, or None.

    None in exact mode (native restart semantics, bitwise to native's own restart), on a moving region, with
    --restart-from, or when no continuation lies beside the file (a restart another model wrote: one stderr line).
    """
    from ..fullstep_lts_admission import moving_region_requested
    if (getattr(config, 'mode', 'exact') != 'correct' or initial_stream != 'restart' or moving_region_requested()
            or getattr(arguments, 'restart_from', None) is not None):
        return None
    candidate = Path(str(initial_file) + CONTINUATION_SUFFIX)
    if candidate.is_file():
        return candidate
    import sys
    print(f'woof hex: restart {initial_file} has no {CONTINUATION_SUFFIX} beside it (not written by a correct-mode '
          'run): native MPAS restart semantics', file=sys.stderr, flush=True)
    return None


def resolve_route(arguments, text, config=None):
    """Resolve moving environment plus CLI restart before native pool decoding.

    ``config``: the run's physics configuration; its schemes' persistent words ride every carrier this run seals
    (``persistent_names``, MovingRoute.carried_rows).  ``initial_state``: (stream, file) the run starts from.

    Correct mode (the door default) makes a native MPAS restart an exact continuation of the run that wrote it,
    which native MPAS is not: a run whose deck writes restart files seals the continuation beside each one, and a
    config_do_restart run from such a file resumes it.  THE BREAKAGE THIS FIXES: native restarts itimestep at 1
    (mpas_atm_core.F:782), so MYNN-SFC re-ran its first-step block (zeroed hfx/qfx/mol, reset pblh), RUC its
    ktau = 1 block, and the restart stream lacks words the schemes read back (RUC keepfr3d); hex restarted from
    its own +24 h restart differed from the uninterrupted run in 22.4 M of 36.4 M diag words one hour later
    (x1.163842 hrrrv5, restart-check).  Exact mode keeps native's restart, word for word."""
    from ..fullstep_lts_admission import moving_region_requested
    from .forecast_v842 import set_namelist_options
    moving = MovingRoute.from_environ(arguments) if moving_region_requested() else None
    restart = getattr(arguments, 'restart_from', None)
    hours = getattr(arguments, 'checkpoint_hours', None)
    analysis = bool(getattr(arguments, 'da_analysis', None))
    if analysis and not (restart or moving is not None and moving.carrier is not None):
        raise ValueError('native --da-analysis requires --restart-from or a moving carrier')
    if moving is not None and config is not None:
        moving.persistent = persistent_names(config)
    # A moving segment resumed on its own member continues the run that sealed its carrier (CheckpointRoute
    # moving_resume).  THE BREAKAGE THIS FIXES: it resumed as a native MPAS restart (itimestep 1, scheme flags,
    # accumulators and start hour begun again), so same-mesh stop/resume on the MPAS physics moving route was not
    # bit for bit (x-build-moving-mpasphys).
    moving_resume = (moving is not None and moving.carrier is not None and not moving.operator
                     and restart is None)
    correct = getattr(config, 'mode', 'exact') == 'correct'
    writes_restarts = restart_output(getattr(arguments, 'streams', None)) if correct and moving is None else None
    if (restart is None and hours is None and not analysis and not moving_resume
            and writes_restarts is None):
        if moving is not None:
            text = moving.resume_namelist(text)
        return moving, text
    route = CheckpointRoute(**({item.name: getattr(moving, item.name)
                               for item in fields(MovingRoute)} if moving is not None else {}))
    route.fixed_mesh = moving is None
    route.moving_resume = moving_resume
    route.restart_output = writes_restarts
    # forecast_v842 turned a correct-mode native restart with a continuation into --restart-from that continuation
    # (native_restart_continuation): the run's own init supplies the static inputs, as for any checkpoint resume.
    route.native_restart = getattr(arguments, '_native_restart_continuation', None) is not None
    if hours is not None:
        route.checkpoint_seconds = _seconds(hours)
    minutes = getattr(arguments, 'history_every_minutes', None)
    if minutes is not None:
        route.frame_seconds = float(minutes) * 60
    route.frame_from_cli = minutes is not None or moving is not None
    if route.frame_seconds <= 0:
        raise ValueError('native checkpoint history interval must be positive')
    if restart is not None:
        path = Path(restart).expanduser().resolve()
        if route.carrier_path and Path(route.carrier_path).resolve() != path:
            raise ValueError('CLI and moving environment name different native restart carriers')
        route.carrier_path, route.carrier = str(path), load_carrier(path)
        route.origin_time = route.carrier['origin_time']
        route.offset_seconds = float(route.carrier['time_seconds'])
        valid = datetime.strptime(route.origin_time, TIME_FORMAT) + timedelta(seconds=route.offset_seconds)
        if (route.offset_seconds <= 0 or route.offset_seconds % 1 or
                valid.strftime(TIME_FORMAT) != route.carrier['valid_time']):
            raise ValueError('native checkpoint elapsed time and valid clock disagree')
        route.restart_metadata = json.loads(path.with_suffix(path.suffix+'.json').read_text()).get('checkpoint')
        if not route.restart_metadata or route.restart_metadata.get('schema') != 'mpas-native-checkpoint/v1':
            raise ValueError('--restart-from needs a checkpoint written by the native checkpoint flags')
        route.cli_restart = not route.native_restart
        if route.operator:
            raise ValueError('native --restart-from uses the same mesh; moving remaps use the moving carrier input')
    route.da_analysis = analysis
    text = route.resume_namelist(text)
    # Native init pools supply static fields. The carrier supplies restart
    # fields after decoding, avoiding init/restart Registry shape collisions.
    if route.carrier is not None:
        text = set_namelist_options(text, {'restart': {'config_do_restart': False}})
    return route, text
