"""Device record transport preserves strided floats and native integers."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.core import ssib
from woof.physics_registry import column_surface_contract

@pytest.mark.gpu
def test_strided_fields_and_rebound_atmosphere_round_trip():
    import cupy as cp
    from woof.core.ssib_records import SurfaceRecords
    contract=column_surface_contract(8)
    shape=(3,5); ncol=15; nz=2
    fields={}
    for k,name in enumerate(ssib.STATE):
        key=contract['state_bindings'][name]
        base,index=(key.split(':')+[None])[:2] if ':' in key else (key,None)
        if base not in fields:
            full=(3,*shape) if index is not None else shape
            # Real interleaved ensemble banks and nest views can be strided.
            bank=cp.zeros((*full,2), dtype=cp.int32 if base=='isnow' else cp.float32)
            fields[base]=bank[...,0]
        a=fields[base] if index is None else fields[base][int(index)]
        a[...] = cp.arange(ncol,dtype=cp.float32).reshape(shape)+k
    for k,name in enumerate(ssib.FORCING):
        if name not in {'dz8w1','u1','v1','qv1','t1','p1','julian'}:
            key={'rainncv':'surface_rainncv','raincv':'surface_raincv',
                 't2_sfclay':'t2'}.get(name,name)
            fields.setdefault(key,
                              cp.full(shape, k+500, cp.float32))
    fields['xland']=cp.ones(shape,cp.float32)
    fields['ivgtyp']=cp.full(shape,7,cp.int32)
    rec=SurfaceRecords(fields,contract)
    owner=SimpleNamespace(state=cp.empty((111,ncol),cp.float32),
        forcing=cp.empty((24+nz,ncol),cp.float32),static=cp.zeros((6,ncol),cp.float32))
    def target(name):
        key=contract['state_bindings'][name]
        if ':' not in key:return fields[key]
        key,index=key.split(':');return fields[key][int(index)]
    expected=cp.asnumpy(cp.stack([target(n).reshape(-1) for n in ssib.STATE]))
    names=('dz','u','v','qv','temperature','pressure')
    for base in (1000,2000):
        atmosphere={n:cp.full((nz,*shape),base+k,cp.float32) for k,n in enumerate(names)}
        cloud=cp.full((nz,*shape),0.25,cp.float32)
        rec.pack(owner,atmosphere,cloud,nz=nz,julian=145.5,dt=12)
        np.testing.assert_array_equal(owner.state.get(),expected)
        for k,n in enumerate(names):
            np.testing.assert_array_equal(owner.forcing[9+k].get(),base+k)
        np.testing.assert_array_equal(owner.forcing[22].get(),145.5)
        for name in ('rainncv','raincv'):
            k=ssib.FORCING.index(name)
            np.testing.assert_array_equal(owner.forcing[k].get(),k+500)
        np.testing.assert_array_equal(owner.forcing[24:].get(),0.25)
        np.testing.assert_array_equal(owner.static[1].get(),7)
    owner.state+=1
    rec.unpack(owner)
    for k,n in enumerate(ssib.STATE):
        np.testing.assert_array_equal(target(n).get().reshape(-1),expected[k]+1)

    from woof.core.surface_forcing import SURFACE_PRECIPITATION_FIELDS
    for name in SURFACE_PRECIPITATION_FIELDS + ('rainbl',):
        fields[name] = cp.full(shape, 17., cp.float32)
    fields['_ssib_status'] = cp.asarray([15], dtype=cp.int32)
    fields['xice'].fill(.4)
    fields['xice_save'].fill(0.)
    pending = cp.full(shape, 23., cp.float32)
    rec = SurfaceRecords(fields, contract, pending_rainbl=pending)
    rec.unpack(owner)
    for name in SURFACE_PRECIPITATION_FIELDS + ('rainbl',):
        np.testing.assert_array_equal(fields[name].get(), 0.)
    np.testing.assert_array_equal(pending.get(), 0.)
    np.testing.assert_array_equal(fields['xice_save'].get(), fields['xice'].get())
    assert fields['_ssib_status'].get()[0] == 7


@pytest.mark.gpu
def test_resident_fields_match_records_bitwise_on_all_oracle_columns():
    """The same arithmetic reaches strided fields, including integer snow state."""
    import json
    from pathlib import Path
    import cupy as cp
    from woof.core.ssib_records import SurfaceRecords
    fixture = Path(__file__).parent/'data/oracles/ssib/ssib-wrf461-columns.npz'
    with np.load(fixture) as z:
        raw = bytes(z['columns'])
    ncol, nstep, nz = map(int, np.frombuffer(raw, '<i4', 3))
    words = np.frombuffer(raw, '<f4', offset=12)
    nc, ns, nf = len(ssib.STATIC), len(ssib.STATE), len(ssib.FORCING)+nz
    static = cp.asarray(np.ascontiguousarray(words[:ncol*nc].reshape(ncol,nc).T))
    at = ncol*nc
    initial = words[at:at+ncol*ns].reshape(ncol,ns).T
    at += ncol*ns
    forcing = words[at:].reshape(nstep,ncol,nf)
    contract = column_surface_contract(8)
    fields = {}
    shape = (1,ncol)
    def target(name):
        key = contract['state_bindings'][name]
        if ':' not in key:return fields[key]
        key,index = key.split(':');return fields[key][int(index)]
    for k,name in enumerate(ssib.STATE):
        key = contract['state_bindings'][name]
        base = key.split(':')[0]
        if base not in fields:
            full = (3,*shape) if ':' in key else shape
            fields[base] = cp.zeros((*full,2), cp.int32 if base=='isnow' else cp.float32)[...,0]
        target(name)[...] = cp.asarray(initial[k]).reshape(shape)
    for name in ssib.FORCING:
        if name not in {'dz8w1','u1','v1','qv1','t1','p1','julian'}:
            fields.setdefault({'rainncv':'surface_rainncv','raincv':'surface_raincv',
                               't2_sfclay':'t2'}.get(name,name), cp.zeros(shape,cp.float32))
    fields['xland'] = cp.ones(shape,cp.float32)
    fields['ivgtyp'] = cp.ones(shape,cp.int32)
    rec = SurfaceRecords(fields,contract,resident=True)
    packed = cp.asarray(np.ascontiguousarray(initial))
    owner = SimpleNamespace(static=static,function=ssib.kernel(resident=True))
    status = cp.zeros(1,cp.int32)
    for step in range(4):
        owner.forcing = cp.asarray(np.ascontiguousarray(forcing[step].T))
        ssib.step(static,packed,owner.forcing,nz=nz,itimestep=step+1)
        ssib.step_fields(owner,rec,nz=nz,itimestep=step+1,status=status)
        actual = cp.stack([target(name).astype(cp.float32).reshape(-1) for name in ssib.STATE])
        np.testing.assert_array_equal(actual.get().view('u4'),packed.get().view('u4'))
