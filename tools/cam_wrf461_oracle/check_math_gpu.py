"""Numerical probe of WOOF's own CAM base-ten reduction against scalar calls."""
from pathlib import Path
import argparse,json,math
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof.core.kernels import get_kernel
 # Public deterministic mathematical inputs, including exact decimal powers.
 values=np.exp(np.linspace(-30,30,20001));values=np.concatenate((values,np.array([10.**n for n in range(-10,11)])))
 reference=np.array([math.log10(float(x)) for x in values]);actual=cp.zeros((len(values),5),cp.float64)
 get_kernel('cam3_emissivity','cam_log10_variants')((math.ceil(len(values)/128),),(128,),(np.int32(len(values)),cp.asarray(values),actual))
 cp.cuda.Device().synchronize();actual=cp.asnumpy(actual);report=[]
 for i in range(5):
  difference=np.abs(actual[:,i].view(np.int64)-reference.view(np.int64))
  report.append({'variant':i,'changed':int(np.count_nonzero(difference)),'max_ulp':int(difference.max()),'max_abs':float(np.abs(actual[:,i]-reference).max())})
 a.out.write_text(json.dumps({'inputs':len(values),'results':report},indent=2)+'\n');print(report)
if __name__=='__main__':main()
