"""Generate the external ABI observer for RRFS 44c0849 mynnsfc_driver (capture tooling, not evidence).

Adapted from the A3 MYNN-EDMF observer builder; build the observer with `make` in its folder (rustc).
"""
from dataclasses import asdict
import hashlib, json, shutil, sys
from pathlib import Path

# Usage: sfc_observer.py MPAS_44C0849_TREE LANE   (the observer lands in LANE/sfc-observer/mynnsfc_driver)
reference, lane = Path(sys.argv[1]), Path(sys.argv[2])
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'native_physics_reference/oracle'))
from fortran_source import routines, call, bind
from external_abi import emit

relative = 'src/core_atmosphere/physics/physics_noaa/MYNN-SFC/MPAS/module_sf_mynnsfc_driver.F90'
root = lane/'observer-source'
target = root/relative
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(reference/relative, target)
routine = next(r for r in routines(target, root) if r.name == 'mynnsfc_driver')
driver_path = 'src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F'
driver = next(r for r in routines(reference/driver_path, reference)
              if any(call(s) and call(s)[0] == routine.name for s in r.body))
statements = [s for s in driver.body if call(s) and call(s)[0] == routine.name]
print('call sites', [s.start+1 for s in statements])
statement = statements[0]
actuals = bind(routine, call(statement)[1])
site = dict(scheme='sf_mynnsfclay', callee=dict(name=routine.name, module=routine.module,
    path=relative, line=routine.start+1, end_line=routine.end),
    caller=dict(name=driver.name, path=driver_path, line=statement.start+1),
    arguments=[dict(asdict(routine.variables[n]), actual=actuals.get(n), present=n in actuals) for n in routine.args])
manifest = dict(reference='rrfs-44c0849', sites=[site],
    source_set_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    scope='selected source ABI only; unchanged pristine native binary required')
out = lane/'sfc-observer'
emit(root, out, manifest, ['sf_mynnsfclay'])
(lane/'sfc-observer-generation.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('ok', out)
