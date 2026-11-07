# Averaged wind fluxes

Set `do_avgflx_em = 1` in a domain's `[[domain]]` table, or pass
`do_avgflx_em=1` when constructing its Python `RunConfig`. The default is 0.
The shared configuration is used by `woof go` and `woof run`.
For a two-domain WRF namelist, write:

```fortran
&dynamics
 do_avgflx_em = 1, 1,
/
```

`woof import-namelist`, the WRF input directory door, and
`woof-wrf-init --namelist-support-report` read the same setting.
A missing domain entry retains WRF's default 0. To enable only the child,
write `do_avgflx_em = 0, 1`.

Each completed acoustic step updates three running means on the GPU using
WRF 4.6.1's single-precision expression `(count * mean + flux)/(count + 1)`.
These are mass-coupled fluxes from the last Runge-Kutta stage's acoustic
average, not averages of the history U, V and W fields. The count is the
number of completed steps, not elapsed seconds. Variable time steps do not
change the weights.

| Variable | Grid | Units | WRF description |
| --- | --- | --- | --- |
| `AVGFLX_RUM` | U faces | Pa m s-1 | hist-time-averaged mu-coupled u |
| `AVGFLX_RVM` | V faces | Pa m s-1 | hist-time-averaged mu-coupled v |
| `AVGFLX_WWM` | Vertical interfaces | Pa s-1 | hist-time-averaged mu-coupled eta-dot |

The initial history frame contains zeros. After a history frame, its means
and count remain available until the next completed step. That step starts
a new interval with count 1. Each nest owns its means, count and history
schedule. Feedback does not copy these diagnostics into the parent. A
moving nest clears its entire interval, matching WRF's startup after a move.
The parent keeps its interval. WRF also restarts the parent when
`input_from_hires` is enabled; WOOF already refuses that unsupported terrain
input setting.
Quicklook frames do not reset the full-history interval.

WOOF checkpoints carry the means, count and pending history reset. This
preserves continuation both inside an interval and at its end. Stock WRF
4.6.1 has a different restart behavior: `start_em.F` unconditionally sets
`avgflx_count=0`, even when restarting. WOOF deliberately preserves it.
Importing native `wrfrst` averaged-flux state is refused because that door
does not restore the native history alarm state. Use WOOF checkpoints for
continuation with averaged fluxes.

The real two-domain continuation check passes with default numerics.
Strict numerics currently encounters an existing two-way-nest restart
failure: feedback changes the parent's base state, so rebuilding the
initial setup does not match the checkpoint's setup fingerprint. This
failure also occurs with averaged fluxes disabled on the earlier engine.

Both WRF NetCDF writers and full state snapshots carry these variables.
A reduced history selection must explicitly retain them with `history_vars`
or `history_include`; the minimal and severe presets do not include them.
The GRIB2 product catalog and thin snapshots have no averaged-flux products.
The count and pending reset are checkpoint state, not history variables.
Memory admission includes three staggered arrays and two integer planes for
the count and reset state. A dry run also needs the acoustic flux arrays.
At 0 no averaged-flux arrays or update kernels are used.

`do_avgflx_cugd=1` is refused. In WRF 4.6.1 those convective means read
CFU1/CFD1/DFU1/EFU1/DFD1/EFD1 from Grell-Devenyi (`cu_physics=93`), which
WOOF does not implement. Grell-3 (`cu_physics=5`) does not produce those
fields in that WRF version.
