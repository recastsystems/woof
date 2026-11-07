"""Arithmetic samples from two complete GFDL cloud columns."""
import os
from pathlib import Path
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('GPUWM_NO_LOCAL_GPU')=='1',reason='CPU-only check')
@pytest.mark.parametrize('n',[24,60])
def test_column_arithmetic(n):
 import cupy as cp
 from woof.core.kernels import get_kernel
 with np.load(Path(__file__).parent/f'fixtures/gfdleta/column/math-{n}.npz') as z:
  kinds=z['kinds'];words=z['words'];out=cp.empty(len(kinds),cp.float32)
  get_kernel('gfdleta_math_probe','gd_math_probe')(((len(kinds)+127)//128,),(128,),(cp.asarray(kinds),cp.asarray(words.view(np.float32)),out,np.int32(len(kinds))))
  got=cp.asnumpy(out).view(np.uint32);bad=np.flatnonzero(got!=words[:,2])
  assert not len(bad),[(int(k),int(kinds[k]),f'{words[k,0]:08X}',f'{words[k,1]:08X}',f'{got[k]:08X}',f'{words[k,2]:08X}') for k in bad[:30]]
