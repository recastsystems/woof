"""Exercise mixed host/device grid transport and the actual compressed writer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

from woof.io.nc_writer_bridge import ClassicSchema
from woof.netcdf_bridge import Dataset
from woof.ocean.automatic import _grid_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # Noncontiguous float64 host and device inputs reproduce the production
    # geometry boundary, rather than a mock of CuPy's conversion rules.
    base=np.arange(48,dtype=np.float64).reshape(6,8)[:,::2]
    fields={'latitude':base+20,'longitude':base-90,
            'sinalpha':base*.001,'cosalpha':base*.002+1,'map_factor':base*.003+1}
    receipts=[]
    for mode in ('host','device','mixed'):
        values={key:(cp.asarray(value) if mode=='device' or
                     (mode=='mixed' and key in ('latitude','longitude','map_factor')) else value)
                for key,value in fields.items()}
        grid=SimpleNamespace(rotation_m=lambda:(values['sinalpha'],values['cosalpha']))
        path=args.output/f'grid-{mode}.nc'
        _grid_file(path,grid,values['latitude'],values['longitude'],
                   SimpleNamespace(msft=values['map_factor']))
        with Dataset(path) as dataset:
            for key,expected in fields.items():
                actual=dataset.variables[key][:]
                assert np.array_equal(actual,expected.astype(np.float32)),(mode,key)
        receipts.append({'mode':mode,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'bytes':path.stat().st_size})
    assert len({row['sha256'] for row in receipts})==1,receipts
    schema=ClassicSchema();y=schema.def_dim('y',6);x=schema.def_dim('x',4)
    variable=schema.def_var('field','f4',(y,x));path=args.output/'compressed.nc'
    expected=base.astype(np.float32)
    with schema.create_netcdf4(path) as writer:writer.write_var(variable,expected)
    with Dataset(path) as dataset:
        # Normal CF reads promote numeric values to float64. Compare the
        # stored float32 words so the check measures the writer's actual bytes.
        actual=dataset.stored_words(['field'])['field']
        assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
    receipt={'status':'PASS','grid_modes':receipts,'compressed_words':expected.size,
             'compressed_differing_words':0,'scope':'native byte transport and compressed writer'}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
