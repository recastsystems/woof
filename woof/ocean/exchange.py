"""Persistent device maps between an atmosphere grid and its wave blocks.

One wave cell represents a clipped integer block of atmosphere mass cells.
Restriction averages water-cell area after rotating each fine-grid vector.
Prolongation is piecewise constant and never writes atmosphere land or ice.
This is a finite-volume block map, not a reshaping of geographic coordinates.
"""
from __future__ import annotations

_CUDA = r"""
extern "C" __global__ void restrict_forcing(
 const float* u,const float* v,const float* p,const float* us,const float* t,
 const float* q,const float* ca,const float* sa,const float* weight,
 const float* land,float* east,float* north,float* pressure,float* ustar,
 float* tx,float* ty,float* rho,int nx,int ny,int wx,int wy,int factor,int rotate) {
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=wx*wy)return;
 int x0=(k%wx)*factor,y0=(k/wx)*factor;
 float sw=0.f,sx=0.f,sy=0.f,sp=0.f,su=0.f,sr=0.f,stx=0.f,sty=0.f;
 for(int y=y0;y<min(y0+factor,ny);++y)for(int x=x0;x<min(x0+factor,nx);++x){
  int i=y*nx+x;if(land[i]<1.5f)continue;
  float w=weight[i];float a=fmaxf(us[i],0.f);
  float ex=rotate?ca[i]*u[i]-sa[i]*v[i]:u[i];
  float no=rotate?sa[i]*u[i]+ca[i]*v[i]:v[i];
  float r=p[i]/(287.05f*t[i]*(1.f+0.61f*fmaxf(q[i],0.f)));
  float stress=r*a*a/fmaxf(hypotf(ex,no),1.e-12f);
  sw+=w;sx+=w*ex;sy+=w*no;sp+=w*p[i];su+=w*a;sr+=w*r;
  stx+=w*stress*ex;sty+=w*stress*no;
 }
 float inv=sw>0.f?1.f/sw:0.f;
 east[k]=sx*inv;north[k]=sy*inv;pressure[k]=sw>0.f?sp*inv:101325.f;
 ustar[k]=su*inv;rho[k]=sw>0.f?sr*inv:1.225f;tx[k]=stx*inv;ty[k]=sty*inv;
}
extern "C" __global__ void prolong(const float* coarse,const float* wet,
 const float* land,const float* ice,float* fine,int nx,int ny,int wx,int factor,int mask){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nx*ny)return;
 int k=(i/nx/factor)*wx+(i%nx/factor);
 bool active=wet[k]>.5f&&land[i]>=1.5f&&ice[i]<=.01f;
 fine[i]=(!mask||active)?coarse[k]:0.f;
}
extern "C" __global__ void apply_roughness(const float* wave,const float* land,
 const float* ice,float* target,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(land[i]>=1.5f&&ice[i]<=.01f&&wave[i]>0.f)target[i]=wave[i];
}
"""


class BlockExchange:
    def __init__(self, source_shape, wave_grid, *, factor, cosalpha, sinalpha,
                 area_weight, xland):
        import cupy as cp
        import numpy as np
        from .interface import validate_plane
        from woof.certify.kernel_manifest import record_module
        self.shape = tuple(source_shape)
        self.wave_grid = wave_grid
        if type(factor) is not int or factor < 1:
            raise ValueError("wave coarsening must be a positive integer")
        self.factor = factor
        ny, nx = self.shape
        if wave_grid.shape != ((ny+factor-1)//factor, (nx+factor-1)//factor):
            raise ValueError("wave grid does not match the configured atmosphere blocks")
        self.geometry = (cosalpha, sinalpha, area_weight, xland)
        for name, value in zip(("cosalpha", "sinalpha", "area_weight", "xland"), self.geometry):
            validate_plane(value, self.shape, name)
        self.dimensions = tuple(np.int32(n) for n in
                                (nx, ny, wave_grid.nx, wave_grid.ny, factor))
        self.fine_dimensions = tuple(np.int32(n) for n in
                                     (nx, ny, wave_grid.nx, factor))
        self.zero = cp.zeros(self.shape, cp.float32)
        self.roughness = cp.zeros(self.shape, cp.float32)
        self.history = {}
        module = cp.RawModule(code=_CUDA, options=())
        record_module("woof.ocean.exchange:blocks", source=_CUDA, options=(), module=module)
        self._restrict = module.get_function("restrict_forcing")
        self._prolong = module.get_function("prolong")
        self._apply = module.get_function("apply_roughness")

    def forcing(self, inputs, outputs, *, grid_relative):
        import numpy as np
        from .interface import validate_plane
        for name, value in zip(("u10", "v10", "psfc", "ustar", "temperature", "qv"), inputs):
            validate_plane(value, self.shape, name)
        n = self.wave_grid.nx*self.wave_grid.ny
        self._restrict(((n+255)//256,), (256,),
                       (*inputs, *self.geometry, *outputs, *self.dimensions,
                        np.int32(grid_relative)))

    def prolong(self, array, wet, *, target=None, ice=None, mask=True):
        import cupy as cp
        import numpy as np
        if target is None:
            target = cp.empty(self.shape, cp.float32)
        n = self.shape[0]*self.shape[1]
        self._prolong(((n+255)//256,), (256,),
                      (array, wet, self.geometry[3], self.zero if ice is None else ice,
                       target, *self.fine_dimensions, np.int32(mask)))
        return target

    def apply(self, target, *, ice=None):
        import numpy as np
        n = self.shape[0]*self.shape[1]
        self._apply(((n+255)//256,), (256,),
                    (self.roughness, self.geometry[3], self.zero if ice is None else ice,
                     target, np.int32(n)))

    def history_field(self, key, array, wet):
        import cupy as cp
        if key not in self.history:
            self.history[key] = cp.empty(self.shape, cp.float32)
        return self.prolong(array, wet, target=self.history[key], mask=key not in ("elevation", "wet"))
