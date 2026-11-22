# Run an existing WRF initial state

WOOF-ArWen's native preparation remains the recommended route. When you already have
a compatible CPU WRF `real.exe` run, you can keep its initialization workflow:

```sh
woof run --wrfinput /path/to/wrf-run --outdir out/forecast
```

The input directory must contain `namelist.input`, `wrfinput_d01` (and a file for
each additional domain), and `wrfbdy_d01`. WOOF-ArWen reads the producing namelist,
checks the files against its geometry and physics, verifies the first boundary
state against the initial state, and uses the shared forecast engine, radiation
workspace, clocks, health checks, restart validation and output. Input bytes
are hashed internally; no hashes need to be copied into a command.

To shorten the forecast within the supplied boundary coverage:

```sh
woof run --wrfinput /path/to/wrf-run --outdir out/short --run-seconds 600
```

`CONFIG` and `--wrfinput` are alternative inputs. Existing `woof run CONFIG`
commands keep their behavior. The WRF route starts a fresh worker on the selected
GPU under the existing UUID lock; `--gpu-uuid` selects a card on multiple-GPU
hosts. Automatic supervisor recovery options currently apply to CONFIG runs.
Explicit `--restart` uses the shared runtime's checkpoint identity checks.

Initial-file `T` is dry perturbation potential temperature under both WRF theta
flags. Both dry (`use_theta_m=0`) and standard moist (`use_theta_m=1`) boundary
files are supported. Moist boundary temperature is converted at each forcing
time, together with its tendency, using the file's interpolated mass and water
vapor. The producing namelist must agree with the files; editing a namelist does
not convert existing data.

File-defined vertical levels and land-use identity are retained. Relocation
requires geography covering future nest positions; initial WRF footprints alone
do not supply that coverage. This adapter does not yet provide a statics corridor
or sealed forcing extension. Its positive execution witness is a 199×199×79
single-domain WRF case with WSM6, Noah, YSU and RTE+RRTMGP: the dry pair completes
ten minutes and 11 history frames; the standard moist pair completes thirty
minutes, 150 steps and 31 frames. Shared linear and nonlinear forcing also pass
an actual resident versus streamed dycore comparison. Other layouts and physics retain their capability checks;
this witness does not certify all combinations.
