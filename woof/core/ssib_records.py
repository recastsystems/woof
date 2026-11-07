"""One device gather and scatter for SSiB's field-major column records."""
from __future__ import annotations
import numpy as np
from woof.core import ssib

class SurfaceRecords:
    """Bind persistent device fields once; atmosphere arrays arrive at launch."""
    def __init__(self, fields, contract, *, pending_rainbl=None, resident=False,
                 direct_forcing=False):
        import cupy as cp
        from woof.core.kernels import get_kernel
        self.resident = resident
        self.direct_forcing = direct_forcing
        if direct_forcing and not resident:
            raise ValueError('direct SSiB forcing needs resident fields')
        self.shape=fields['tsk'].shape
        self.ny,self.nx=map(int,self.shape)
        self.ncol=self.ny*self.nx
        self.ncol_arg = np.int32(self.ncol)
        self.nx_arg = np.int32(self.nx)
        self.grid = ((self.ncol + 63) // 64,)
        self.block = (64,)
        self.fixes_arg = np.int32(ssib.fixes_for_run())
        self.cloud_flags = (np.int32(0), np.int32(1))
        def field(key):
            if ':' in key:
                key,index=key.split(':')
                return fields[key][int(index)]
            return fields[key]
        state=tuple(field(contract['state_bindings'][n]) for n in ssib.STATE)
        dynamic={'dz8w1','u1','v1','qv1','t1','p1','julian'}
        forcing_names = {'rainncv': 'surface_rainncv', 'raincv': 'surface_raincv',
                         't2_sfclay': 't2'}
        forcing=tuple(None if n in dynamic else fields[forcing_names.get(n, n)]
                      for n in ssib.FORCING)
        self.fields=(*state,*forcing,fields['xland'],fields['ivgtyp'])
        descriptors=[]
        for a in self.fields:
            if a is None:
                descriptors.append((0,0,0,0));continue
            if a.shape!=self.shape or a.dtype not in (cp.float32,cp.int32):
                raise ValueError('SSiB record fields need the surface grid and float32/int32 storage')
            descriptors.append((a.data.ptr,*a.strides,int(a.dtype==cp.int32)))
        self.descriptors=cp.asarray(np.asarray(descriptors,dtype=np.uint64).view(np.uint32))
        self.commit_descriptors = self.descriptors
        self.status = self.descriptors
        self.finish = pending_rainbl is not None
        self.finish_arg = np.int32(self.finish)
        if self.finish:
            from woof.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS
            arrays = (fields['xice_save'], fields['xice'],
                      *(fields[n] for n in SURFACE_PRECIPITATION_FIELDS),
                      fields['rainbl'], pending_rainbl)
            self.commit_descriptors = cp.asarray(np.asarray([
                (a.data.ptr, *a.strides, 0) for a in arrays],
                dtype=np.uint64).view(np.uint32))
            self.status = fields['_ssib_status']
        self.pack_kernel=get_kernel('ssib_records',
            'ssib_forcing_pack' if resident else 'ssib_records_pack')
        self.unpack_kernel=get_kernel('ssib_records',
            'ssib_fields_finish' if resident else 'ssib_records_unpack')

    def pack(self, owner, atmosphere, cloud, *, nz, julian, dt):
        values=tuple(atmosphere[n] for n in ('dz','u','v','qv','temperature','pressure'))
        if any(not a.flags.c_contiguous for a in values) or (cloud is not None and not cloud.flags.c_contiguous):
            raise ValueError('SSiB atmospheric record inputs need contiguous column planes')
        if self.direct_forcing:
            # Kernel arguments bind the current atmosphere, even when its
            # publisher rotates buffers. The surface descriptors stay resident.
            self.inputs = (values, values[0] if cloud is None else cloud,
                           cloud is not None, np.float32(julian), np.float32(dt))
            if not hasattr(self, 'nz_arg') or self.nz_arg != nz:
                self.nz_arg = np.int32(nz)
            return
        self.pack_kernel(((self.ncol+255)//256,(24 if self.resident else 135)+nz),(256,),
            (self.descriptors,owner.forcing if self.resident else owner.state,owner.forcing,owner.static,*values,
             owner.forcing if cloud is None else cloud,np.int32(cloud is not None),
             np.int32(self.ncol),np.int32(self.nx),np.int32(nz),np.float32(julian),np.float32(dt)))

    def unpack(self, owner):
        if self.direct_forcing:
            # The column kernel commits its own precipitation and ice state.
            return
        if self.resident:
            self.unpack_kernel(((self.ncol+255)//256,), (256,),
                (np.int32(self.ncol), np.int32(self.nx),
                 self.commit_descriptors, self.status, np.int32(self.finish)))
            return
        self.unpack_kernel(((self.ncol+255)//256,111),(256,),
            (self.descriptors,owner.state,np.int32(self.ncol),np.int32(self.nx),
             self.commit_descriptors, self.status, np.int32(self.finish)))
