"""DFI history snapshots and initialized input through the Rust NC writer.

History copies are CUDA graph nodes. Publication and the sole host download
occur after a complete pass, never between its weather solves.
"""
from pathlib import Path
from datetime import timedelta
from contextlib import contextmanager


@contextmanager
def _classic_template(template, directory):
    """Keep classic templates intact; transcribe NetCDF-4 through Rust IO.

    ClassicPatch preserves the template layout, but cannot read HDF5. WRF
    inputs may use either container. Stored-word reads avoid CF masking or
    scaling while the Rust writer carries every variable into CDF-5, whose
    types also accommodate the source's integer attributes.
    """
    template = Path(template)
    with template.open('rb') as source:
        if source.read(3) == b'CDF':
            yield template
            return
    from tempfile import TemporaryDirectory
    from woof.netcdf_bridge import Dataset
    from woof.io.nc_writer_bridge import ClassicSchema
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.dfi-template-',dir=directory) as temporary:
        converted = Path(temporary)/'wrfinput'
        with Dataset(template) as source:
            schema = ClassicSchema('cdf5')
            dimensions = {name:schema.def_dim(name,len(dim),unlimited=dim.isunlimited())
                          for name,dim in source.dimensions.items()}
            for name in source.ncattrs():
                schema.put_global_attr(name,source.getncattr(name))
            variables = {}
            for name,var in source.variables.items():
                index = schema.def_var(name,'S1' if var.is_character else var.dtype,
                                       [dimensions[d] for d in var.dimensions])
                for attr in var.ncattrs():
                    schema.put_var_attr(index,attr,var.getncattr(attr))
                variables[name] = index
            values = source.stored_words(source.variables)
            with schema.create(converted) as writer:
                for name,index in variables.items():
                    var = source.variables[name]
                    if var.dimensions and source.dimensions[var.dimensions[0]].isunlimited():
                        for record in range(var.shape[0]):
                            writer.write_record(record,index,values[name][record])
                    else:
                        writer.write_var(index,values[name])
        yield converted


class DfiOutput:
    def __init__(self, model, start_time, directory):
        self.model, self.start_time = model,start_time
        self.directory = Path(directory)
        self.frames = []
        self.seen = set()
        self.clocks = {n.cfg.grid_id:n.clock for n in model.walk_parent_first()}

    def capture(self, node, elapsed, phase):
        cfg = node.cfg.run
        if not cfg.dfi_write_dfi_history:
            return
        clock = self.clocks[node.cfg.grid_id]
        offset = (elapsed-phase.start)*clock.tick_den
        if offset != int(offset) or not clock.spec.history_offset_due(int(offset)):
            return
        key = (node.cfg.grid_id,bool(phase.backward),elapsed)
        if key in self.seen:
            return
        self.seen.add(key)
        import cupy as cp
        from woof.io.wrfout import _device_state_frame
        state = node.state
        driver = state.physics
        if driver is None:
            state.physics = state._dfi_physics_initialization.driver
        try:
            fields = _device_state_frame(state,include_diagnostic_pressure=True)
            frozen = {name:cp.array(value,copy=True,order='C')
                      for name,value in fields.items()}
        finally:
            state.physics = driver
        self.frames.append((node,elapsed,bool(phase.backward),frozen))

    def _write(self, node, elapsed, fields, path, stage, start_seconds=0):
        import cupy as cp
        from woof.runtime import _metadata_frame, _global_wrf_attrs
        from woof.io.wrfout import WrfoutWriter
        from woof.config import soil_layer_count
        case = self.model._prepared_by_grid_id[node.cfg.grid_id]
        cfg = node.cfg.run
        frame = {name:cp.asnumpy(value) for name,value in fields.items()}
        frame.update(_metadata_frame(node.grid,case.static_fields))
        frame.update(getattr(case,'wrf_file_grid',None) or {})
        attrs = _global_wrf_attrs(node.grid,self.start_time+timedelta(seconds=start_seconds),
            simulation_start_time=self.start_time,domain=node.cfg,
            coord=case.initial_result.coord,static_fields=case.static_fields)
        attrs.update(DFI_OPT=cfg.dfi_opt,DFI_STAGE=stage)
        path.parent.mkdir(parents=True,exist_ok=True)
        valid = self.start_time+timedelta(seconds=elapsed)
        with WrfoutWriter(path,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,dx=cfg.dx,dy=cfg.dy,
                title='WRF digital-filter initialization',global_attrs=attrs,
                field_schema=frame,soil_layers=soil_layer_count(cfg)) as writer:
            writer.write_frame(valid.strftime('%Y-%m-%d_%H:%M:%S'),frame)

    def flush(self):
        from woof.io.wrfout import wrfout_filename
        earliest = min((row[1] for row in self.frames),default=0)
        for node,elapsed,backward,fields in self.frames:
            phase = 'backward' if backward else 'forward'
            valid = self.start_time+timedelta(seconds=elapsed)
            path = self.directory/'dfi'/phase/f'd{node.cfg.grid_id:02d}'/wrfout_filename(valid,node.cfg.grid_id)
            self._write(node,elapsed,fields,path,phase,start_seconds=min(0,earliest))
        self.frames.clear()

    def initialized(self, elapsed):
        from woof.io.wrfout import _device_state_frame
        for node in self.model.walk_parent_first():
            if node.cfg.run.dfi_write_filtered_input:
                fields = _device_state_frame(node.state,include_diagnostic_pressure=True)
                case = self.model._prepared_by_grid_id[node.cfg.grid_id]
                template = getattr(case,'dfi_input_template',None)
                target = self.directory/f'wrfinput_initialized_d{node.cfg.grid_id:02d}'
                if template is None:
                    self._initialized_prepared(node,elapsed,fields,target,case)
                    continue
                import cupy as cp
                import numpy as np
                from woof.io.nc_writer_bridge import ClassicPatch
                from woof.io.wrfout import wrf_axis_array
                with _classic_template(template,self.directory) as classic, ClassicPatch(classic,target) as patch:
                    for name,variable in patch.variables.items():
                        if name == 'Times':
                            value = np.frombuffer((self.start_time+timedelta(seconds=elapsed)).
                                strftime('%Y-%m-%d_%H:%M:%S').encode('ascii'),dtype='S1')
                        elif name == 'XTIME':
                            value = np.asarray([elapsed/60.],dtype=variable.dtype)
                        elif name in fields:
                            value = wrf_axis_array(name,cp.asnumpy(fields[name]),
                                nx=node.cfg.run.nx,ny=node.cfg.run.ny,nz=node.cfg.run.nz)
                        else:
                            continue
                        patch.put(name,value,record=0 if variable.is_record else None)

    def _initialized_prepared(self, node, elapsed, fields, target, case):
        """Use the stock input exporter for a prepared-cache initialization.

        History and wrfinput have different mandatory inventories. Reuse
        the existing input contract and fixed geography, then replace its
        live fields with the filtered state. Never label a history file
        as an initialized input.
        """
        import cupy as cp
        import numpy as np
        from dataclasses import asdict
        from woof.config import soil_layer_count
        from woof.io.wrfout import wrf_axis_array
        from woof.wrf_direct import (_load_contract, _physics_contract_bundle,
            _load_static_geometry_receipt, _wrfinput_fields, _land_category_contract,
            _dimensions, _global_updates, _write_wrfinput)
        cfg = node.cfg.run
        configuration = asdict(cfg)
        valid = self.start_time+timedelta(seconds=elapsed)
        bundle = getattr(case, 'dfi_prepared_bundle', None)
        if bundle is None:
            geometry, snapshot = node.state._dfi_input_snapshot
            values = dict(snapshot)
        else:
            geometry, _ = _load_static_geometry_receipt(
                bundle.geometry_receipt_path,bundle.static_path)
            with np.load(bundle.static_path,allow_pickle=False) as static:
                values = _wrfinput_fields(bundle.cache_reader,static,geometry,valid,
                    p_top=float(node.state.p_top),mp_physics=cfg.mp_physics,
                    sf_surface_physics=cfg.sf_surface_physics,
                    num_soil_layers=soil_layer_count(cfg),run_config=configuration)
        for name in values.keys() & fields.keys():
            values[name] = wrf_axis_array(name,cp.asnumpy(fields[name]),
                nx=cfg.nx,ny=cfg.ny,nz=cfg.nz)
        contract = _physics_contract_bundle(_load_contract(),cfg.mp_physics,
                                            configuration)['wrfinput']
        contract,land_attrs = _land_category_contract(contract,values)
        attrs = _global_updates(valid_time=valid,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,
            dx=cfg.dx,dy=cfg.dy,dt=cfg.dt,geometry=geometry,
            run_config=configuration)
        attrs.update(land_attrs)
        attrs.update(START_DATE=self.start_time.strftime('%Y-%m-%d_%H:%M:%S'),
                     SIMULATION_START_DATE=self.start_time.strftime('%Y-%m-%d_%H:%M:%S'))
        attrs.update(GRID_ID=node.cfg.grid_id,PARENT_ID=node.cfg.parent_id,
            I_PARENT_START=node.cfg.i_parent_start,J_PARENT_START=node.cfg.j_parent_start,
            PARENT_GRID_RATIO=node.cfg.parent_grid_ratio)
        target.parent.mkdir(parents=True,exist_ok=True)
        _write_wrfinput(target,contract,_dimensions(contract,nx=cfg.nx,ny=cfg.ny,
            nz=cfg.nz,num_soil_layers=soil_layer_count(cfg)),attrs,values,
            valid.strftime('%Y-%m-%d_%H:%M:%S'),engine='rust')
