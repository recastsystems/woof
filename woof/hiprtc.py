"""Direct HIPRTC compilation, including off-card gfx target validation.

CuPy's ROCm RawModule path appends an unconditional denormal-flush flag.
This boundary owns the effective numeric flags and returns a real AMD ELF.
"""
import ctypes as C
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
from functools import lru_cache
from woof.hip_source import translate,options_for_amd,REVISION

@lru_cache(maxsize=1)
def library():
 root=Path(os.environ.get('ROCM_PATH',os.environ.get('ROCM_HOME','/opt/rocm')))
 name=os.environ.get('HIPRTC_LIBRARY') or str(root/'lib/libhiprtc.so')
 lib=C.CDLL(name)
 for fn,args in {
  'hiprtcCreateProgram':[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p],
  'hiprtcCompileProgram':[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)],
  'hiprtcGetProgramLogSize':[C.c_void_p,C.POINTER(C.c_size_t)],
  'hiprtcGetProgramLog':[C.c_void_p,C.c_void_p],
  'hiprtcGetCodeSize':[C.c_void_p,C.POINTER(C.c_size_t)],
  'hiprtcGetCode':[C.c_void_p,C.c_void_p],
  'hiprtcDestroyProgram':[C.POINTER(C.c_void_p)],
  'hiprtcAddNameExpression':[C.c_void_p,C.c_char_p],
  'hiprtcGetLoweredName':[C.c_void_p,C.c_char_p,C.POINTER(C.c_char_p)],
  'hiprtcVersion':[C.POINTER(C.c_int),C.POINTER(C.c_int)]}.items():
  getattr(lib,fn).argtypes=args;getattr(lib,fn).restype=C.c_int
 return lib,str(Path(name).resolve())

def identity():
 lib,path=library();major=C.c_int();minor=C.c_int()
 if lib.hiprtcVersion(C.byref(major),C.byref(minor)):raise RuntimeError('hiprtcVersion failed')
 return dict(version=[major.value,minor.value],library_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

def compile(source,options,target,filename='kernel.hip.cpp',name_expressions=None):
 lib,_=library();hip=translate(source)
 # HIPRTC 7 does not discover clang's resource headers when a CuPy template
 # pulls in the host C++ standard library (cstddef -> stddef.h). Explicitly
 # supply the installed compiler's headers; their contents are also hashed.
 from woof.hip_backend import compiler_include_directories,toolchain_identity
 root=Path(toolchain_identity()[0]).resolve().parent.parent
 resources=tuple('-I'+str(p) for p in compiler_include_directories(root)
                 if (p/'stddef.h').is_file())
 opts=(*options_for_amd(options),*resources,'--gpu-architecture='+target)
 # Bind external package headers as well as HIPRTC's builtin headers.
 from woof.hip_backend import header_identity,compiler_dependencies,toolchain_identity
 headers=[compiler_dependencies(toolchain_identity()[0]),*header_identity(opts)]
 ident=identity()
 key=hashlib.sha256(json.dumps([hip,opts,target,filename,name_expressions,ident,headers,REVISION],sort_keys=True).encode()).hexdigest()
 cache=Path(os.environ.get('CUPY_CACHE_DIR',str(Path.home()/'.cache/cupy')))/'gpuwm-hiprtc'
 cache.mkdir(parents=True,exist_ok=True);image=cache/(key+'.hsaco');meta=cache/(key+'.json')
 if image.exists() and meta.exists():
  row=json.loads(meta.read_text());blob=image.read_bytes()
  if hashlib.sha256(blob).hexdigest()==row['image_sha256']:return blob,row['mapping']
 program=C.c_void_p()
 def check(code):
  if code:
   size=C.c_size_t();lib.hiprtcGetProgramLogSize(program,C.byref(size));log=C.create_string_buffer(size.value or 1);lib.hiprtcGetProgramLog(program,log)
   raise RuntimeError('HIPRTC error '+str(code)+'\n'+log.value.decode(errors='replace'))
 check(lib.hiprtcCreateProgram(C.byref(program),hip.encode(),filename.encode(),0,None,None))
 try:
  for name in name_expressions or ():check(lib.hiprtcAddNameExpression(program,name.encode()))
  argv=(C.c_char_p*len(opts))(*(o.encode() for o in opts))
  check(lib.hiprtcCompileProgram(program,len(opts),argv))
  size=C.c_size_t();check(lib.hiprtcGetCodeSize(program,C.byref(size)))
  buffer=C.create_string_buffer(size.value);check(lib.hiprtcGetCode(program,buffer));blob=buffer.raw
  if blob[:4]!=b'\x7fELF':raise RuntimeError('HIPRTC did not emit an AMD ELF')
  mapping={}
  for name in name_expressions or ():
   lowered=C.c_char_p();check(lib.hiprtcGetLoweredName(program,name.encode(),C.byref(lowered)));mapping[name]=lowered.value.decode()
 finally:lib.hiprtcDestroyProgram(C.byref(program))
 row=dict(key=key,arch=target,compiler=ident,headers=headers,options=opts,mapping=mapping,image_sha256=hashlib.sha256(blob).hexdigest())
 with tempfile.NamedTemporaryFile(dir=cache,delete=False) as f:f.write(blob);temp=f.name
 os.replace(temp,image);meta.write_text(json.dumps(row,indent=2))
 return blob,mapping
