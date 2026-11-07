"""Grade native table state and time interpolation against WRF captures."""
from pathlib import Path
import argparse,ctypes,json
import numpy as np
from validate import load,compare

def main():
 p=argparse.ArgumentParser();p.add_argument('table_root',type=Path);p.add_argument('library',type=Path);p.add_argument('oracle',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 lib=ctypes.CDLL(str(a.library));lib.cam_tables_open.argtypes=[ctypes.c_char_p];lib.cam_tables_open.restype=ctypes.c_void_p
 lib.cam_tables_state.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)];lib.cam_tables_state.restype=ctypes.POINTER(ctypes.c_ubyte)
 lib.cam_tables_gases.argtypes=[ctypes.c_void_p,ctypes.c_int32,ctypes.c_float,ctypes.c_void_p];lib.cam_tables_gases.restype=ctypes.c_int32
 handle=lib.cam_tables_open(str(a.table_root).encode())
 if not handle:
  error=ctypes.create_string_buffer(4096);lib.cam_tables_error(error,4096);raise RuntimeError(error.value.decode())
 size=ctypes.c_size_t();pointer=lib.cam_tables_state(handle,ctypes.byref(size));state=np.empty(size.value,np.uint8);physical=np.array([9.81,1004.5,0.622,5.67e-8,287.,461.6,0.622,2.501e6,3.336e5,461.6,1004.5,273.16],np.float64)
 lib.cam_tables_copy_state.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int32,ctypes.c_void_p,ctypes.c_size_t];status=lib.cam_tables_copy_state(handle,ctypes.c_void_p(physical.ctypes.data),3,ctypes.c_void_p(state.ctypes.data),size.value)
 if status:raise RuntimeError("native state initialization failed")
 reference=np.frombuffer((a.oracle/'state.bin').read_bytes(),np.uint8)
 layout=json.loads((Path(__file__).resolve().parents[2]/'woof/data/cam3/state-layout.json').read_text());mismatches=[]
 for field in layout:
  lo=field['offset'];hi=lo+field['bytes'];actual=state[lo:hi];expected=reference[lo:hi]
  if not np.array_equal(actual,expected):mismatches.append({'field':field['name'],'changed_bytes':int(np.count_nonzero(actual!=expected))})
 data=load(a.oracle);expected={};actual={}
 for column in range(1,65):
  year=int(data[('camrad',column,'input:yr')][0]);julian=np.float32(data[('camrad',column,'input:julian')][0]);value=np.zeros(5,np.float64)
  status=lib.cam_tables_gases(handle,year,julian,ctypes.c_void_p(value.ctypes.data))
  if status:raise RuntimeError('native CAM gas status '+str(status))
  key=('camrad',column,'input:gases');expected[key]=data[key];actual[key]=value
 gas=compare(expected,actual);result={'pass':not mismatches and gas['pass'],'state_bytes':size.value,'state_mismatches':mismatches,'gas':gas}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':result['pass'],'state_mismatches':mismatches,'gas_pass':gas['pass'],'gas_words':gas['words'],'gas_failures':gas['mismatches'][:2]}));raise SystemExit(not result['pass'])
if __name__=='__main__':main()
