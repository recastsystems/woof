"""Native-mesh, source-free geographic transport experiment across a pole.

This diagnoses the geographic operator alone, not forecast skill. All
geographic numerics, including dispersion, stay enabled. Run inside a GPU
pool lease. Field diagnostics execute on CUDA; matplotlib draws analysis
profiles and time series, not weather maps.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    import cupy as cp
    import numpy as np
    from woof.hex.mesh import Mesh
    from woof.ocean.waves.backends.voronoi import VoronoiBackend
    began=time.perf_counter()
    mesh=Mesh.from_netcdf(args.grid)
    geometry={key:cp.asarray(mesh.arrays[key]) for key in
        ("latCell","lonCell","latEdge","lonEdge","areaCell","cellsOnEdge",
         "edgesOnCell","nEdgesOnCell","dvEdge","dcEdge","angleEdge")}
    for key in ("cellsOnEdge","edgesOnCell"):
        geometry[key]=cp.ascontiguousarray(geometry[key]+1,dtype=cp.int32)
    nc=int(mesh.dimensions["nCells"])
    bathy=SimpleNamespace(depth=cp.full(nc,1000.,cp.float32),wet=cp.ones(nc,cp.float32))
    backend=VoronoiBackend(geometry,bathy)
    nf=nd=36
    f=cp.float32(.035)*cp.power(cp.float32(1.1),cp.arange(nf,dtype=cp.float32))
    group=cp.broadcast_to((cp.float32(9.80665)/(cp.float32(4*math.pi)*f))[None],(nc,nf)).copy()
    selected=5
    speed=9.80665/(4*math.pi*(.035*1.1**selected))
    radius=float(mesh.attrs.get("sphere_radius",6371220.))
    latitude,longitude=geometry["latCell"],geometry["lonCell"]
    point=cp.stack((cp.cos(latitude)*cp.cos(longitude),cp.cos(latitude)*cp.sin(longitude),cp.sin(latitude)))
    east=cp.stack((-cp.sin(longitude),cp.cos(longitude),cp.zeros(nc,cp.float64)))
    north=cp.stack((-cp.sin(latitude)*cp.cos(longitude),-cp.sin(latitude)*cp.sin(longitude),cp.cos(latitude)))
    center_lat=math.radians(84.)
    center=cp.asarray([math.cos(center_lat),0.,math.sin(center_lat)])
    tangent=cp.asarray([-math.sin(center_lat),0.,math.cos(center_lat)])
    dot=cp.sum(center[:,None]*point,axis=0)
    vector=tangent[:,None]-(cp.sum(tangent[:,None]*point,axis=0)/cp.maximum(1+dot,1.e-12))[None]*(center[:,None]+point)
    heading=cp.arctan2(cp.sum(vector*north,axis=0),cp.sum(vector*east,axis=0))
    distance=cp.arccos(cp.clip(dot,-1,1))
    spatial=cp.where(distance<math.radians(8.4),cp.exp(-.5*(distance/math.radians(1.4))**2),0.)
    theta=cp.arange(nd,dtype=cp.float32)*(2*math.pi/nd)
    directional=cp.exp(80*(cp.cos(theta[None]-heading[:,None])-1))
    state=cp.zeros((nc,nf,nd),cp.float32)
    state[:,selected]=spatial[:,None]*directional
    stream=cp.cuda.get_current_stream()
    backend.advect(state,group,0.,stream,None)
    area=backend.area.astype(cp.float64)
    # Unwrapped angle along the initial x-z great circle: 84 -> 90 -> 96 deg.
    # This is an analysis coordinate, not a map projection.
    along=cp.rad2deg(cp.arctan2(point[2],point[0]))
    edges=cp.linspace(75.,105.,121,dtype=cp.float64)
    bins=cp.clip(cp.searchsorted(edges,along)-1,0,119)
    rows=[]; initial_energy=None
    def sample(seconds):
        nonlocal initial_energy
        weights=cp.sum(state,dtype=cp.float64,axis=(1,2))*area
        total=float(cp.sum(weights))
        if initial_energy is None:initial_energy=total
        centroid=cp.sum(point*weights[None],axis=1)/cp.sum(weights)
        center_words=cp.asnumpy(centroid).tolist()
        actual=math.degrees(math.atan2(center_words[2],center_words[0]))
        q=state[:,selected].astype(cp.float64)
        u=cp.sum(q*cp.cos(theta)[None],axis=1)
        v=cp.sum(q*cp.sin(theta)[None],axis=1)
        exit_cells=along>92.
        east_sum=float(cp.sum(u[exit_cells]*area[exit_cells]))
        north_sum=float(cp.sum(v[exit_cells]*area[exit_cells]))
        to_bearing=math.degrees(math.atan2(east_sum,north_sum))%360
        profile=cp.bincount(bins,weights=weights,minlength=120)/total
        row=dict(seconds=seconds,energy=total,relative_energy=total/initial_energy,
                 centroid_angle_deg=actual,expected_angle_deg=84+math.degrees(speed*seconds/radius),
                 exit_toward_deg=to_bearing,exit_fraction=float(cp.sum(weights[exit_cells]))/total,
                 minimum=float(cp.min(state)),nonfinite=int(cp.count_nonzero(~cp.isfinite(state))),
                 profile=cp.asnumpy(profile).tolist())
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!="profile"}),flush=True)
    sample(0.)
    end=math.radians(12.)*radius/speed
    steps=math.ceil(end/300.)
    dt=end/steps
    snapshots={round(steps*x/4) for x in range(1,5)}
    for step in range(1,steps+1):
        state=backend.advect(state,group,dt,stream,None)
        if step in snapshots:sample(step*dt)
    final=rows[-1]
    energy_error=max(abs(r["relative_energy"]-1) for r in rows)
    position_error=abs(final["centroid_angle_deg"]-final["expected_angle_deg"])
    heading_error=abs((final["exit_toward_deg"]-180+180)%360-180)
    passed=(all(r["nonfinite"]==0 and r["minimum"]>=0 for r in rows)
            and energy_error<2.e-5 and position_error<2.5 and heading_error<20 and final["exit_fraction"]>.65)
    receipt=dict(schema="gpuwm-hex.waves.pole-crossing/v1",status="PASS" if passed else "FAIL",
        grid=str(args.grid),cells=nc,nf=nf,nd=nd,frequency_index=selected,
        frequency_hz=.035*1.1**selected,group_speed_m_s=speed,steps=steps,dt_seconds=dt,
        sources="not part of this geographic operator experiment",depth_m=1000,
        initial_center_deg=[84.,0.],expected_final_center_deg=[84.,180.],
        energy_relative_error=energy_error,centroid_error_deg=position_error,exit_heading_error_deg=heading_error,
        rows=rows,backend=backend.receipt(),wall_seconds=time.perf_counter()-began,
        scope="native 40962-cell spherical geographic propagation, including polar cells and default dispersion; not forecast skill")
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout="constrained")
    x=np.linspace(75.125,104.875,120)
    for row in rows:axes[0].plot(x,row["profile"],label=f'{row["seconds"]/3600:.1f} h')
    axes[0].axvline(90,color="black",ls="--",lw=1,label="North Pole")
    axes[0].set(xlabel="Unwrapped angle along initial great circle (degrees)",ylabel="Fraction of packet energy per 0.25 degree bin",
                title="Swell packet transported across the North Pole on the native 120 km mesh")
    axes[0].legend(ncols=3)
    hours=[r["seconds"]/3600 for r in rows]
    axes[1].plot(hours,[(r["relative_energy"]-1)*1.e6 for r in rows],"o-",color="#245b88")
    axes[1].axhline(0,color="black",lw=.7)
    axes[1].set(xlabel="Elapsed hours",ylabel="Total energy change (parts per million)")
    axes[1].text(.02,.96,f'Final centroid error: {position_error:.3f} deg\nExit heading: {final["exit_toward_deg"]:.2f} deg toward (expected 180)\nEnergy beyond 92 deg: {100*final["exit_fraction"]:.1f}%',
                 transform=axes[1].transAxes,va="top",bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
    fig.savefig(args.output/"pole-crossing.png",dpi=160)
    plt.close(fig)
    if not passed:raise RuntimeError("pole-crossing criteria failed; see receipt and analysis figure")


if __name__=="__main__":main()
