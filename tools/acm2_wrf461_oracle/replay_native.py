"""Replay sparse real native ACMPBL captures and grade every returned word."""
import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from woof.core.acm2pbl import acm2_pbl_step
from woof.certify.kernel_manifest import kernel_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('case', type=Path)
parser.add_argument('--receipt', type=Path, required=True)
args = parser.parse_args()
case = args.case
from woof.numerics import is_strict
if not is_strict():
    parser.error('native word replay requires GPUWM_NUMERICS=strict')
three = ('u','v','p','dz8w','theta','t','qv','qc','qi','rho')
two = ('ust','hfx','qfx','tsk','psfc','wspd','psim','gz1oz0','mut','pblh','regime','rmol')
tendencies = ('du','dv','dtheta','dqv','dqc','dqi')
extra = ('exch_h','exch_m',*tendencies)
output3 = (*tendencies,'exch_h','exch_m')

class Reader:
    def __init__(self, path):
        self.path = path
        self.data = path.read_bytes()
        self.pos = 0
        self.head = self.take('i4', 10)
        assert int(self.head[0]) == 1094929714
        self.ncol,self.nz,self.xtime = map(int,self.head[1:4])
        self.constants = dict(zip(('dtpbl','ep1','g','rovcp','rd','cpd'), self.take('f4',6)))
    def take(self, dtype, count):
        # WRF's FCBASEOPTS records -fconvert=big-endian for stream I/O.
        data = np.frombuffer(self.data, dtype='>'+dtype, count=count, offset=self.pos)
        self.pos += 4*count
        return data.astype('<'+dtype)
    def field(self):
        return self.take('f4',self.ncol*self.nz).reshape(self.ncol,self.nz).T.copy()
    def done(self):
        assert self.pos == len(self.data), (self.path,self.pos,len(self.data))

receipts = []
for path in sorted(case.glob('acm2-*-in.bin')):
    before = Reader(path)
    after = Reader(path.with_name(path.name.replace('-in.bin','-out.bin')))
    assert np.array_equal(before.head,after.head)
    assert before.constants == after.constants
    locations = before.take('i4',2*before.ncol).reshape(-1,2)
    inputs = {name:before.field() for name in three}
    inputs.update({name:before.take('f4',before.ncol) for name in two})
    inputs.update({name:before.field() for name in extra})
    before.done()
    desired = {name:after.field() for name in output3}
    desired.update({name:after.take('i4' if name=='kpbl' else 'f4',after.ncol)
                    for name in ('pblh','regime','rmol','kpbl','ust')})
    after.done()
    device = {name:cp.asarray(value) for name,value in inputs.items()}
    arguments = {name:device[name] for name in ('u','v','theta','t','qv','qc','qi',
                  'rho','dz8w','ust','hfx','qfx','wspd','pblh','regime','rmol','exch_h','exch_m')}
    arguments.update({name:float(before.constants[name]) for name in
                      ('dtpbl','ep1','g','rd','cpd')})
    out = acm2_pbl_step(**arguments,xtime=before.xtime)
    # ACMPBL adds its six increments to the caller's tendency arrays.
    # Keep that last native addition on the GPU, including any prior value.
    got = {name:cp.asnumpy(out[name]+device[name]) for name in tendencies}
    got.update({name:cp.asnumpy(device[name]) for name in
                ('pblh','regime','rmol','exch_h','exch_m','ust')})
    got['kpbl'] = cp.asnumpy(out['kpbl'])
    fields = {}
    for name,want in desired.items():
        mismatch = got[name].view(np.uint32) != want.view(np.uint32)
        fields[name] = {'words':want.size,'different':int(mismatch.sum())}
        if mismatch.any():
            points = np.argwhere(mismatch)[:8]
            fields[name]['examples'] = [
                {'index':point.tolist(), 'gpu_word':hex(int(got[name].view(np.uint32)[tuple(point)])),
                 'native_word':hex(int(want.view(np.uint32)[tuple(point)]))}
                for point in points]
    row = {'input':str(path),'input_bytes':len(before.data),
           'input_sha256':hashlib.sha256(before.data).hexdigest(),
           'output':str(after.path),'output_bytes':len(after.data),
           'output_sha256':hashlib.sha256(after.data).hexdigest(),
           'xtime':before.xtime,'nz':before.nz,'ncol':before.ncol,
           'points':locations.tolist(), 'constants':{k:float(v) for k,v in before.constants.items()},
           'nonzero_initial_tendency_words':sum(int(np.count_nonzero(inputs[k])) for k in tendencies),
           'fields':fields}
    receipts.append(row)
    print(json.dumps(row),flush=True)
assert receipts,'No actual native capture files were produced'
totals = {'words':sum(field['words'] for row in receipts for field in row['fields'].values()),
          'different':sum(field['different'] for row in receipts for field in row['fields'].values()),
          'columns':sum(row['ncol'] for row in receipts), 'captures':len(receipts),
          'calls':sorted(set(row['xtime'] for row in receipts)), 'strict':True,
          'rows':receipts, 'kernel_manifest':kernel_manifest()}
args.receipt.write_text(json.dumps(totals,indent=2)+'\n')
print(json.dumps({k:v for k,v in totals.items() if k not in ('rows','kernel_manifest')}))
assert totals['different'] == 0,totals['different']
