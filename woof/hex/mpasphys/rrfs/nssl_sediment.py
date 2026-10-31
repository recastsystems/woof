"""Carry native fallout fluxes without materializing the column temporary.

Native F:4151-4183 computes all old-state fluxes, then updates bottom to top.
The next level remains old at each update, so its flux can be carried in a
scalar. Field operations, operand order and the top zero boundary stay exact.
"""
from woof.hex.mpasphys.rrfs.nssl_phases import _block

def stream_fallout(source):
 declaration='NsslArray<float, 1> qtmp1 = nssl_array<float, 1>(arena, {1}, {((nz) + (1))});'
 assert source.count(declaration)==1
 source=source.replace(declaration,'')
 begin=source.index('qtmp1(((nz) + (1))) = 0.0f;')
 return source[:begin]+'''// Native F:4151-4183; retain each flux expression and bottom-to-top updates.
// At update k, level k+1 is still old. Carry its flux into the next update.
if(nz <= 2 || a(ix,jgs,1,ia) != 0.0f || a(ix,jgs,2,ia) != 0.0f) {
 kz=1;
 xfall(ix,jy,ia)=nssl_fadd(xfall(ix,jy,ia),nssl_fmul(nssl_fmul(a(ix,jgs,kz,ia),vt(kz,ix)),dtfrac));
}
float flux = id==1 ? nssl_fmul(nssl_fmul(a(ix,jgs,1,ia),vt(1,ix)),db1(ix,1)) : nssl_fmul(a(ix,jgs,1,ia),vt(1,ix));
for(kz=1;kz<=nz;++kz) {
 float next_flux=0.0f;
 if(kz<nz) next_flux=id==1 ? nssl_fmul(nssl_fmul(a(ix,jgs,kz+1,ia),vt(kz+1,ix)),db1(ix,kz+1)) : nssl_fmul(a(ix,jgs,kz+1,ia),vt(kz+1,ix));
 a(ix,jgs,kz,ia)=nssl_fadd(a(ix,jgs,kz,ia),nssl_fmul(nssl_fmul(dtp,dtz1(kz,ix,id)),nssl_fsub(next_flux,flux)));
 flux=next_flux;
}
return;
}'''

def lower(module):
    begin = module.index("__device__ __forceinline__ void f_fallout1d(")
    _, end = _block(module, begin)
    return module[:begin] + stream_fallout(module[begin:end+1]) + module[end+1:]
