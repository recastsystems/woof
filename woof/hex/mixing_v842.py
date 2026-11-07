"""MPAS-A v8.4.2 complete non-LES saved dissipation boundary.

Pristine authority: mpas_atm_dissipation_models.F:579-1606 and
mpas_atm_time_integration.F:6319-6410,6585-6598,6880-6906, commit f34984b.
The existing horizontal device stencils are reused; fixed coefficients,
physical vertical diffusion and first-substep scalar accumulation are added.
This module does not select a physics suite or change a forecast default.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING
import numpy as np
from .mixing_v841 import V841MixingConfig
from .errors import ConfigurationRefusal
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .native_powf import CUDA_NATIVE_POWF
from .mixing import _finite_scalar

if TYPE_CHECKING:
    from .cuda_horizontal import CudaDryMixingTendencies


@dataclass(frozen=True, slots=True)
class V842MixingConfig(V841MixingConfig):
    """Non-LES options; mpas_atm_time_integration.F:6241-6256,6330-6369."""

    config_h_theta_eddy_visc2: float = 0.0
    config_h_mom_eddy_visc4: float = 0.0
    config_h_theta_eddy_visc4: float = 0.0
    config_v_mom_eddy_visc2: float = 0.0
    config_v_theta_eddy_visc2: float = 0.0
    config_mix_full: bool = True
    config_mix_scalars: bool = False

    def validate(self):
        """Registry.xml:111-291; mpas_atm_time_integration.F:6326-6369."""
        if self.config_horiz_mixing not in ("2d_smagorinsky", "2d_fixed"):
            raise ConfigurationRefusal("config_horiz_mixing",self.config_horiz_mixing,
                "this boundary implements the two native non-LES mixing branches",
                "2d_smagorinsky or 2d_fixed")
        configured_length = _finite_scalar("config_len_disp", self.config_len_disp)
        # Native core lines168-201 use nominalMinDc for every nonpositive
        # configured length. Retire the inherited negative-length refusal
        # only at this new boundary; the established dispatcher is unchanged.
        V841MixingConfig.validate(replace(self,config_horiz_mixing="2d_smagorinsky",
            config_len_disp=max(0.0, configured_length)))
        for name in ("config_h_theta_eddy_visc2", "config_h_mom_eddy_visc4",
                     "config_h_theta_eddy_visc4", "config_v_mom_eddy_visc2",
                     "config_v_theta_eddy_visc2"):
            value=getattr(self,name)
            if not np.isfinite(value) or value < 0 or not np.isfinite(np.float32(value)):
                raise ConfigurationRefusal(name,value,"native viscosity must be finite and nonnegative in RKIND",">=0")
        for name in ("config_mix_full", "config_mix_scalars"):
            if not isinstance(getattr(self,name),(bool,np.bool_)):
                raise ConfigurationRefusal(name,getattr(self,name),"Registry option is logical","True or False")


def resolve_config_len_disp(nominal_min_dc, configured_length):
    """mpas_atm_core.F:168-201: resolve config_len_disp from nominalMinDc."""
    configured = np.float32(configured_length)
    length=float(np.float32(nominal_min_dc) if configured <= 0 else configured)
    if not np.isfinite(length) or length <= 0 or not np.isfinite(np.float32(length)):
        raise ValueError("mixing requires a finite positive RKIND length")
    return length


def compute_nonles_mixing_tendencies_v842(
    horizontal,
    normal_velocity: Any,
    tangential_velocity: Any,
    vertical_velocity: Any,
    theta_m: Any,
    rho_edge: Any,
    divergence: Any,
    vorticity: Any,
    *,
    dt: float,
    config: Any | None = None,
    cam_damping_levels: int = 4,
    n_cells_solve: int | None = None,
    initial_u_tendency: Any | None = None,
    n_edges_solve: int | None = None,
    rho_zz: Any | None = None,
    zgrid: Any | None = None,
    rdzw: Any | None = None,
    rdzu: Any | None = None,
    u_init: Any | None = None,
    v_init: Any | None = None,
    t_init: Any | None = None,
    cos_angle: Any | None = None,
    sin_angle: Any | None = None,
    scalars: Any | None = None,
    tend_scalars: Any | None = None,
    dynamics_substep: int = 1,
    rk_step: int = 1,
) -> CudaDryMixingTendencies:
    """Complete non-LES RK1 dissipation boundary on resident device arrays.

    MPAS v8.4.2 mpas_atm_time_integration.F:6319-6410,6585-6598,6880-6906;
    mpas_atm_dissipation_models.F:579-1606. Tendencies are saved by the caller
    for subsequent RK stages. Scalar tendencies accumulate only in dynamics
    substep one. Reference sine/cosine arrays must come from the shared GNU
    host maths route, matching the native angleEdge intrinsics.
    """
    from .cuda_horizontal import CudaDryMixingTendencies
    # Reused horizontal stencils must obey the same gradual-underflow contract.
    # Copy only the launcher; retain resident arrays and leave the driver's
    # original cache untouched so its default dispatch is unchanged.
    from copy import copy
    from .cuda_upper_damping_v842 import direct_cache
    horizontal = copy(horizontal)
    horizontal.kernel_cache = direct_cache()
    horizontal._kernels = {}
    horizontal._v841_kernels = {}

    cfg = V842MixingConfig() if config is None else config
    if not isinstance(cfg, V842MixingConfig):
        raise TypeError("config must be V842MixingConfig")
    cfg.validate()
    if rk_step != 1:
        raise ValueError("MPAS computes saved dissipation only on RK stage one")
    if type(dynamics_substep) is not int or dynamics_substep < 1:
        raise ValueError("dynamics_substep must be a positive integer")
    timestep = float(dt)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("dt must be finite and positive")
    deformation = getattr(horizontal, "_deformation_v841", None)
    if cfg.config_horiz_mixing == "2d_smagorinsky" and deformation is None:
        raise RuntimeError(
            "v8.4.1 deformation weights are not attached; call "
            "attach_deformation_weights_v841 first"
        )

    normal = horizontal._edge_field("normal_velocity", normal_velocity)
    tangential = horizontal._edge_field(
        "tangential_velocity", tangential_velocity
    )
    vertical = horizontal._interface_field("vertical_velocity", vertical_velocity)
    theta = horizontal._cell_field("theta_m", theta_m)
    edge_rho = horizontal._edge_field("rho_edge", rho_edge)
    div = horizontal._cell_field("divergence", divergence)
    vort = horizontal._field(
        "vorticity", vorticity, (horizontal.nlev, horizontal.nvertices)
    )
    cp = horizontal.cp

    length = resolve_config_len_disp(horizontal.mesh.nominal_min_dc, cfg.config_len_disp)
    length32 = np.float32(length)
    if cfg.config_horiz_mixing == "2d_smagorinsky":
        cs = np.float32(cfg.config_smagorinsky_coef)
        strain_scale = np.float32((cs * length32) * (cs * length32))
        inv_dt = np.float32(np.float32(1.0) / np.float32(timestep))
        ceiling = np.float32((np.float32(0.01) * (length32 * length32)) * inv_dt)
        h4_mom = h4_theta = np.float32(np.float32(cfg.config_visc4_2dsmag) *
            ((length32 * length32) * length32))
    else:
        h4_mom = np.float32(cfg.config_h_mom_eddy_visc4)
        h4_theta = np.float32(cfg.config_h_theta_eddy_visc4)

    cell_solve = horizontal.ncells if n_cells_solve is None else n_cells_solve
    edge_solve = horizontal.nedges if n_edges_solve is None else n_edges_solve
    for label, count, total in (("cells",cell_solve,horizontal.ncells),
                                ("edges",edge_solve,horizontal.nedges)):
        if type(count) is not int or not 0 <= count <= total:
            raise ValueError(f"solve {label} must lie inside the field")
    # Check every optional data carrier before launching kernels or mutating
    # the caller's scalar tendency pool.
    nl, nc, ne = horizontal.nlev, horizontal.ncells, horizontal.nedges
    if cfg.config_v_mom_eddy_visc2 > 0 or cfg.config_v_theta_eddy_visc2 > 0:
        _resident(horizontal, "rho_zz", rho_zz, (nl, nc))
        _resident(horizontal, "zgrid", zgrid, (nl + 1, nc))
    if cfg.config_v_mom_eddy_visc2 > 0:
        _resident(horizontal, "rdzw", rdzw, (nl,))
        _resident(horizontal, "rdzu", rdzu, (nl,))
        if not cfg.config_mix_full:
            for name, value, shape in (("u_init", u_init, (nl,)),
                ("v_init", v_init, (nl,)), ("cos_angle", cos_angle, (ne,)),
                ("sin_angle", sin_angle, (ne,))):
                _resident(horizontal, name, value, shape)
    if cfg.config_v_theta_eddy_visc2 > 0 and not cfg.config_mix_full:
        _resident(horizontal, "t_init", t_init, (nl, nc))
    if cfg.config_mix_scalars and dynamics_substep == 1:
        from .cuda_backend import require_resident_array
        q = require_resident_array("scalars", scalars, dtype=np.float32)
        if q.ndim != 3 or q.shape[1:] != (nl, nc) or not q.flags.c_contiguous:
            raise ValueError("scalars must be contiguous species/level/cell arrays")
        _resident(horizontal, "tend_scalars", tend_scalars, q.shape)
    if cfg.config_mpas_cam_coef > 0 and (
            type(cam_damping_levels) is not int or not 1 <= cam_damping_levels <= nl):
        raise ValueError("CAM level count must lie inside the field")
    del2 = cp.empty((horizontal.nedges,), dtype=cp.float32)
    del4 = cp.empty_like(del2)
    _launch842(horizontal,
        "nonles_scaling842", horizontal.nedges,
        (
            horizontal.mesh.cells_on_edge,
            horizontal.mesh.mesh_density,
            np.int32(horizontal.nedges),
            np.int32(bool(cfg.config_h_ScaleWithMesh)),
            del2,
            del4,
        ),
    )

    kdiff = cp.empty((horizontal.nlev, horizontal.ncells), dtype=cp.float32)
    if cfg.config_horiz_mixing == "2d_smagorinsky":
        horizontal._launch_v841(
            "smagorinsky_v841_f32",
            horizontal.nlev * horizontal.ncells,
            (
                normal,
                tangential,
                horizontal.mesh.edges_on_cell,
                horizontal.mesh.n_edges_on_cell,
                deformation["coef_c2"],
                deformation["coef_s2"],
                deformation["coef_cs"],
                strain_scale,
                ceiling,
                np.int32(horizontal.nlev),
                np.int32(horizontal.ncells),
                np.int32(horizontal.nedges),
                np.int32(horizontal.max_edges),
                kdiff,
            ),
        )
    else:
        kdiff.fill(np.float32(cfg.config_h_theta_eddy_visc2))

    if cfg.config_mpas_cam_coef > 0:
        from .cuda_upper_damping_v842 import apply_cam_floor_cuda
        apply_cam_floor_cuda(kdiff, length=length, coefficient=cfg.config_mpas_cam_coef,
            levels=cam_damping_levels, solved=horizontal.ncells, cache=horizontal.kernel_cache)
    delsq_u = cp.empty_like(normal)
    # Native adds the filter into caller-owned Euler storage, including a
    # zero seed (mpas_atm_dissipation_models.F:715-722). Assigning the product
    # directly instead can retain a negative zero that native's addition loses.
    tend_u = cp.zeros_like(normal) if initial_u_tendency is None else cp.array(
        horizontal._edge_field("initial_u_tendency", initial_u_tendency), copy=True
    )
    horizontal._launch_v841(
        "momentum_filter_seed_v841_f32",
        horizontal.nedges,
        (
            edge_rho,
            div,
            vort,
            kdiff,
            del2,
            horizontal.mesh.cells_on_edge,
            horizontal.mesh.vertices_on_edge,
            horizontal.mesh.dc_edge,
            horizontal.mesh.dv_edge,
            np.int32(horizontal.nlev),
            np.int32(horizontal.ncells),
            np.int32(horizontal.nedges),
            np.int32(horizontal.nvertices),
            delsq_u,
            tend_u,
        ),
    )
    delsq_div = cp.zeros((horizontal.nlev, horizontal.ncells), dtype=cp.float32)
    delsq_vort = cp.zeros((horizontal.nlev, horizontal.nvertices), dtype=cp.float32)
    halo_u = tend_u[:, edge_solve:].copy()
    if h4_mom > np.float32(0.0):
        horizontal._launch_v841(
            "laplacian_divergence_v841_f32",
            horizontal.ncells,
            (
                delsq_u,
                horizontal.mesh.edges_on_cell,
                horizontal.mesh.n_edges_on_cell,
                horizontal.mesh.cells_on_edge,
                horizontal.mesh.dv_edge,
                horizontal.mesh.area_cell,
                np.int32(horizontal.nlev),
                np.int32(horizontal.ncells),
                np.int32(horizontal.nedges),
                np.int32(horizontal.max_edges),
                delsq_div,
            ),
        )
        horizontal._launch_v841(
            "laplacian_vorticity_v841_f32",
            horizontal.nvertices,
            (
                delsq_u,
                horizontal.mesh.edges_on_vertex,
                horizontal.mesh.vertices_on_edge,
                horizontal.mesh.dc_edge,
                horizontal.mesh.area_triangle,
                np.int32(horizontal.nlev),
                np.int32(horizontal.nedges),
                np.int32(horizontal.nvertices),
                np.int32(horizontal.vertex_degree),
                delsq_vort,
            ),
        )
        horizontal._launch_v841(
            "momentum_filter_lap4_v841_f32",
            horizontal.nedges,
            (
                edge_rho,
                delsq_div,
                delsq_vort,
                del4,
                horizontal.mesh.cells_on_edge,
                horizontal.mesh.vertices_on_edge,
                horizontal.mesh.dc_edge,
                horizontal.mesh.dv_edge,
                h4_mom,
                np.float32(cfg.config_del4u_div_factor),
                np.int32(horizontal.nlev),
                np.int32(horizontal.ncells),
                np.int32(horizontal.nedges),
                np.int32(horizontal.nvertices),
                tend_u,
            ),
        )

    tend_u[:, edge_solve:] = halo_u

    delsq_theta = cp.empty_like(theta)
    tend_theta = cp.empty_like(theta)
    horizontal._launch_v841(
        "theta_filter_lap2_v841_f32",
        horizontal.ncells,
        (
            theta,
            edge_rho,
            kdiff,
            del2,
            horizontal.mesh.edges_on_cell,
            horizontal.mesh.n_edges_on_cell,
            horizontal.mesh.cells_on_edge,
            horizontal.mesh.dc_edge,
            horizontal.mesh.dv_edge,
            horizontal.mesh.area_cell,
            np.int32(horizontal.nlev),
            np.int32(horizontal.ncells),
            np.int32(horizontal.nedges),
            np.int32(horizontal.max_edges),
            delsq_theta,
            tend_theta,
        ),
    )
    halo_theta = tend_theta[:, cell_solve:].copy()
    if h4_theta > np.float32(0.0):
        horizontal._launch_v841(
            "theta_filter_lap4_v841_f32",
            horizontal.ncells,
            (
                delsq_theta,
                del4,
                horizontal.mesh.edges_on_cell,
                horizontal.mesh.n_edges_on_cell,
                horizontal.mesh.cells_on_edge,
                horizontal.mesh.dc_edge,
                horizontal.mesh.dv_edge,
                horizontal.mesh.area_cell,
                h4_theta,
                np.int32(horizontal.nlev),
                np.int32(horizontal.ncells),
                np.int32(horizontal.max_edges),
                tend_theta,
            ),
        )

    tend_theta[:, cell_solve:] = halo_theta

    delsq_w = cp.empty((horizontal.nlev, horizontal.ncells), dtype=cp.float32)
    tend_w = cp.zeros((horizontal.nlev + 1, horizontal.ncells), dtype=cp.float32)
    horizontal._launch_v841(
        "w_filter_lap2_v841_f32",
        horizontal.ncells,
        (
            vertical,
            edge_rho,
            kdiff,
            del2,
            horizontal.mesh.edges_on_cell,
            horizontal.mesh.n_edges_on_cell,
            horizontal.mesh.cells_on_edge,
            horizontal.mesh.dc_edge,
            horizontal.mesh.dv_edge,
            horizontal._w_mixing_geometry_v841(),
            np.int32(horizontal.nlev),
            np.int32(horizontal.ncells),
            np.int32(horizontal.nedges),
            np.int32(horizontal.max_edges),
            delsq_w,
            tend_w,
        ),
    )
    halo_w = tend_w[:, cell_solve:].copy()
    if h4_mom > np.float32(0.0):
        horizontal._launch_v841(
            "w_filter_lap4_v841_f32",
            horizontal.ncells,
            (
                delsq_w,
                del4,
                horizontal.mesh.edges_on_cell,
                horizontal.mesh.n_edges_on_cell,
                horizontal.mesh.cells_on_edge,
                horizontal.mesh.dc_edge,
                horizontal.mesh.dv_edge,
                horizontal.mesh.area_cell,
                h4_mom,
                np.int32(horizontal.nlev),
                np.int32(horizontal.ncells),
                np.int32(horizontal.max_edges),
                tend_w,
            ),
        )

    tend_w[:, cell_solve:] = halo_w

    apply_nonles_vertical_mixing_v842(horizontal, tend_u, tend_w, tend_theta,
        normal, vertical, theta, edge_rho, config=cfg, rho_zz=rho_zz,
        zgrid=zgrid, rdzw=rdzw, rdzu=rdzu, u_init=u_init, v_init=v_init,
        t_init=t_init, cos_angle=cos_angle, sin_angle=sin_angle,
        n_cells_solve=cell_solve, n_edges_solve=edge_solve)
    if cfg.config_mix_scalars and dynamics_substep == 1:
        scalar_lap = accumulate_scalar_mixing_v842(horizontal, scalars, tend_scalars,
            edge_rho, kdiff, del2, del4, h4_theta,
            n_cells_solve=cell_solve)
        if scalar_lap.shape[0]:
            delsq_theta = scalar_lap[-1].copy()

    return CudaDryMixingTendencies(
        kdiff=kdiff,
        h_mom_eddy_visc4=h4_mom,
        h_theta_eddy_visc4=h4_theta,
        tend_u_euler=tend_u,
        tend_w_euler=tend_w,
        tend_theta_euler=tend_theta,
        delsq_u=delsq_u,
        delsq_divergence=delsq_div,
        delsq_vorticity=delsq_vort,
        delsq_w=delsq_w,
        delsq_theta=delsq_theta,
    )


# Physical-coordinate diffusion, MPAS v8.4.2
# mpas_atm_dissipation_models.F:818-889,1102-1119,1414-1471.
_VERTICAL_SOURCE = CUDA_FTZ_HELPERS + r"""
__device__ __forceinline__ float midpoint842(float a,float b) {
    return mpas_mul(0.5f,mpas_add(a,b));
}
__device__ __forceinline__ float perturb_u842(int k,int e,int ne,int full,
    const float* u,const float* ui,const float* vi,const float* ca,const float* sa) {
    float value=u[k*ne+e];
    return full ? value : mpas_add(mpas_sub(value,mpas_mul(ui[k],ca[e])),mpas_mul(vi[k],sa[e]));
}
extern "C" __global__ void nonles_vertical_u842(
    int nl,int nc,int ne,int solved,int full,float viscosity,
    const int* coe,const float* z,const float* u,const float* rho,
    const float* ui,const float* vi,const float* ca,const float* sa,float* tendency) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nl*ne)return;
    int k=i/ne,e=i%ne;if(e>=solved || k<1 || k>=nl-1)return;
    int c1=coe[2*e],c2=coe[2*e+1];
    float z1=midpoint842(z[(k-1)*nc+c1],z[(k-1)*nc+c2]);
    float z2=midpoint842(z[k*nc+c1],z[k*nc+c2]);
    float z3=midpoint842(z[(k+1)*nc+c1],z[(k+1)*nc+c2]);
    float z4=midpoint842(z[(k+2)*nc+c1],z[(k+2)*nc+c2]);
    float zm=midpoint842(z1,z2),z0=midpoint842(z2,z3),zp=midpoint842(z3,z4);
    float um=perturb_u842(k-1,e,ne,full,u,ui,vi,ca,sa);
    float u0=perturb_u842(k,e,ne,full,u,ui,vi,ca,sa);
    float up=perturb_u842(k+1,e,ne,full,u,ui,vi,ca,sa);
    float curvature=mpas_sub(mpas_div(mpas_sub(up,u0),mpas_sub(zp,z0)),
                            mpas_div(mpas_sub(u0,um),mpas_sub(z0,zm)));
    float increment=mpas_div(mpas_mul(mpas_mul(rho[i],viscosity),curvature),
                            mpas_mul(0.5f,mpas_sub(zp,zm)));
    tendency[i]=mpas_add(tendency[i],increment);
}
extern "C" __global__ void nonles_vertical_w842(
    int nl,int nc,int solved,float viscosity,const float* rho,const float* w,
    const float* rdzw,const float* rdzu,float* tendency) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nl*nc)return;
    int k=i/nc,c=i%nc;if(c>=solved || k<1)return;
    float curvature=mpas_sub(mpas_mul(mpas_sub(w[i+nc],w[i]),rdzw[k]),
                            mpas_mul(mpas_sub(w[i],w[i-nc]),rdzw[k-1]));
    float increment=mpas_mul(mpas_mul(mpas_mul(mpas_mul(viscosity,0.5f),
                      mpas_add(rho[i],rho[i-nc])),curvature),rdzu[k]);
    tendency[i]=mpas_add(tendency[i],increment);
}
extern "C" __global__ void nonles_vertical_theta842(
    int nl,int nc,int solved,int full,float viscosity,const float* z,
    const float* theta,const float* initial,const float* rho,float* tendency) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nl*nc)return;
    int k=i/nc,c=i%nc;if(c>=solved || k<1 || k>=nl-1)return;
    float zm=midpoint842(z[i-nc],z[i]),z0=midpoint842(z[i],z[i+nc]);
    float zp=midpoint842(z[i+nc],z[i+2*nc]);
    float tm=theta[i-nc],t0=theta[i],tp=theta[i+nc];
    if(!full) {tm=mpas_sub(tm,initial[i-nc]);t0=mpas_sub(t0,initial[i]);tp=mpas_sub(tp,initial[i+nc]);}
    float curvature=mpas_sub(mpas_div(mpas_sub(tp,t0),mpas_sub(zp,z0)),
                            mpas_div(mpas_sub(t0,tm),mpas_sub(z0,zm)));
    float increment=mpas_div(mpas_mul(mpas_mul(viscosity,rho[i]),curvature),
                            mpas_mul(0.5f,mpas_sub(zp,zm)));
    tendency[i]=mpas_add(tendency[i],increment);
}
"""

# Scalar accumulation must add each edge flux to the existing tendency; adding
# a pre-summed theta-filter tendency loses native rounding. MPAS v8.4.2
# mpas_atm_dissipation_models.F:1334-1409.
_SCALAR_SOURCE = CUDA_FTZ_HELPERS + r"""
extern "C" __global__ void nonles_scalar_lap2_842(
    int nl,int nc,int ne,int maxe,int ns,
    const int* eoc,const int* count,const int* coe,
    const float* area,const float* dc,const float* dv,
    const float* del2,const float* rho,const float* eddy,const float* q,
    float* lap,float* tendency) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ns*nl*nc)return;
    int c=i%nc,k=(i/nc)%nl,s=i/(nl*nc);
    float result=0.0f,tend=tendency[i];
    for(int j=0;j<count[c];++j) {
        int e=eoc[c*maxe+j],c1=coe[2*e],c2=coe[2*e+1];
        float sign=(c==c1 ? 1.0f : -1.0f);
        float edge_sign=mpas_mul(mpas_mul(mpas_mul(mpas_div(1.0f,area[c]),sign),dv[e]),mpas_div(1.0f,dc[e]));
        float flux=mpas_mul(mpas_mul(edge_sign,mpas_sub(q[(s*nl+k)*nc+c2],q[(s*nl+k)*nc+c1])),rho[k*ne+e]);
        result=mpas_add(result,flux);
        flux=mpas_mul(mpas_mul(mpas_mul(flux,0.5f),mpas_add(eddy[k*nc+c1],eddy[k*nc+c2])),del2[e]);
        tend=mpas_add(tend,flux);
    }
    lap[i]=result;tendency[i]=tend;
}
extern "C" __global__ void nonles_scalar_lap4_842(
    int nl,int nc,int maxe,int ns,int solved,float h4,
    const int* eoc,const int* count,const int* coe,
    const float* area,const float* dc,const float* dv,const float* del4,
    const float* lap,float* tendency) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ns*nl*nc)return;
    int c=i%nc,k=(i/nc)%nl,s=i/(nl*nc);if(c>=solved)return;
    float tend=tendency[i],r_area=mpas_mul(h4,mpas_div(1.0f,area[c]));
    for(int j=0;j<count[c];++j) {
        int e=eoc[c*maxe+j],c1=coe[2*e],c2=coe[2*e+1];
        float sign=(c==c1 ? 1.0f : -1.0f);
        float edge_sign=mpas_mul(mpas_mul(mpas_mul(mpas_mul(del4[e],r_area),dv[e]),sign),mpas_div(1.0f,dc[e]));
        tend=mpas_sub(tend,mpas_mul(edge_sign,mpas_sub(lap[(s*nl+k)*nc+c2],lap[(s*nl+k)*nc+c1])));
    }
    tendency[i]=tend;
}
"""


# MPAS mpas_atm_core.F:1134-1160; use the shared GNU powf authority.
_SCALING_SOURCE = r"""
// Keep divisors in constant memory so every target retains division.
#ifndef HEX_DIVISOR_40000000_DEFINED
#define HEX_DIVISOR_40000000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_40000000 = 2.0f;
#else
static const float hex_divisor_40000000 = 2.0f;
#endif
#endif

extern "C" __global__ void nonles_scaling842(const int* coe,
    const float* density,int ne,int enabled,float* del2,float* del4) {
    int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=ne)return;
    del2[e]=1.0f;del4[e]=1.0f;if(!enabled)return;
    float mean=mpas_div(mpas_add(density[coe[2*e]],density[coe[2*e+1]]),hex_divisor_40000000);
    del2[e]=mpas_div(1.0f,mpas_native_powf_positive(mean,0.25f));
    del4[e]=mpas_div(1.0f,mpas_native_powf_positive(mean,0.75f));
}
"""
_CUDA_SOURCE = (_VERTICAL_SOURCE + _SCALAR_SOURCE[len(CUDA_FTZ_HELPERS):]
                + CUDA_NATIVE_POWF + _SCALING_SOURCE)

def _resident(horizontal, name, value, shape):
    """Array contract for mpas_atm_dissipation_models.F:579-1606."""
    from .cuda_backend import require_resident_array
    field=require_resident_array(name,value,dtype=np.float32)
    if field.shape != shape or not field.flags.c_contiguous:
        raise ValueError(f"{name} requires a contiguous level-major array of shape {shape}")
    return field


def _launch842(horizontal, name, size, arguments):
    """CUDA dispatch for mpas_atm_dissipation_models.F:579-1606."""
    if size:
        from .cuda_upper_damping_v842 import direct_cache
        kernel=direct_cache().raw_kernel(name,_CUDA_SOURCE,module_key="hexcore.mixing_v842")
        kernel(((size+127)//128,),(128,),arguments)


def apply_nonles_vertical_mixing_v842(horizontal,tend_u,tend_w,tend_theta,
    u,w,theta,rho_edge,*,config,rho_zz,zgrid,rdzw,rdzu,u_init,v_init,t_init,
    cos_angle,sin_angle,n_cells_solve,n_edges_solve):
    """mpas_atm_dissipation_models.F:818-889,1102-1119,1414-1471."""
    nl,nc,ne=horizontal.nlev,horizontal.ncells,horizontal.nedges
    vm=np.float32(config.config_v_mom_eddy_visc2)
    vt=np.float32(config.config_v_theta_eddy_visc2)
    if vm == 0 and vt == 0:
        return
    rho=_resident(horizontal,"rho_zz",rho_zz,(nl,nc))
    z=_resident(horizontal,"zgrid",zgrid,(nl+1,nc))
    full=np.int32(bool(config.config_mix_full))
    # Full-state kernels do not dereference these pointer placeholders.
    ui=vi=ca=sa=initial=theta
    if not full:
        if vm > 0:
            ui=_resident(horizontal,"u_init",u_init,(nl,))
            vi=_resident(horizontal,"v_init",v_init,(nl,))
            ca=_resident(horizontal,"cos_angle",cos_angle,(ne,))
            sa=_resident(horizontal,"sin_angle",sin_angle,(ne,))
        if vt > 0:
            initial=_resident(horizontal,"t_init",t_init,(nl,nc))
    if vm > 0:
        zw=_resident(horizontal,"rdzw",rdzw,(nl,))
        zu=_resident(horizontal,"rdzu",rdzu,(nl,))
        _launch842(horizontal,"nonles_vertical_u842",nl*ne,
            (np.int32(nl),np.int32(nc),np.int32(ne),np.int32(n_edges_solve),full,vm,
             horizontal.mesh.cells_on_edge,z,u,rho_edge,ui,vi,ca,sa,tend_u))
        _launch842(horizontal,"nonles_vertical_w842",nl*nc,
            (np.int32(nl),np.int32(nc),np.int32(n_cells_solve),vm,rho,w,zw,zu,tend_w))
    if vt > 0:
        _launch842(horizontal,"nonles_vertical_theta842",nl*nc,
            (np.int32(nl),np.int32(nc),np.int32(n_cells_solve),full,vt,z,theta,initial,rho,tend_theta))


def accumulate_scalar_mixing_v842(horizontal,scalars,tend_scalars,rho_edge,kdiff,
    del2,del4,h4,*,n_cells_solve):
    """mpas_atm_dissipation_models.F:1334-1409; accumulate native edge order."""
    from .cuda_backend import require_resident_array
    q=require_resident_array("scalars",scalars,dtype=np.float32)
    nl,nc,ne=horizontal.nlev,horizontal.ncells,horizontal.nedges
    if q.ndim != 3 or q.shape[1:] != (nl,nc) or not q.flags.c_contiguous:
        raise ValueError("scalars must be contiguous species/level/cell arrays")
    tendency=_resident(horizontal,"tend_scalars",tend_scalars,q.shape)
    lap=horizontal.cp.empty_like(q)
    mesh=horizontal.mesh
    common=(np.int32(nl),np.int32(nc))
    topology=(mesh.edges_on_cell,mesh.n_edges_on_cell,mesh.cells_on_edge,
              mesh.area_cell,mesh.dc_edge,mesh.dv_edge)
    _launch842(horizontal,"nonles_scalar_lap2_842",q.size,
        (*common,np.int32(ne),np.int32(horizontal.max_edges),np.int32(q.shape[0]),
         *topology,del2,rho_edge,kdiff,q,lap,tendency))
    if h4 > np.float32(0):
        _launch842(horizontal,"nonles_scalar_lap4_842",q.size,
            (*common,np.int32(horizontal.max_edges),np.int32(q.shape[0]),
             np.int32(n_cells_solve),np.float32(h4),*topology,del4,lap,tendency))
    return lap
