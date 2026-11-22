# The WOOF models

WOOF is two weather models that run on NVIDIA GPUs. Each has a product name,
a package of its own, a command, and a version number of its own. This page
says which is which, and which installs can share a Python environment.

| Model | What it is | Status |
| --- | --- | --- |
| WOOF-ArWen | The WRF engine: WRF-ARW on GPUs for regions you draw, with nests and storm-following runs. Fast numerics by default; an exact mode is checked against native WRF 4.8.0 on the cases its verification page names. | Production |
| WOOF-HEX-A (hexagonal mesh; A for Atmosphere, as in MPAS-A) | The MPAS engine: MPAS-Atmosphere v8.4.2 and NOAA's hrrrv5 (RRFS v2) configuration on GPUs, on hexagonal meshes: whole globes, zoom globes, limited-area and storm-following regions, with a tangent-linear and adjoint for sensitivity work. | Preview |

WRF and MPAS are developed at NCAR; WOOF is not affiliated with or endorsed
by NCAR or UCAR.

## Packages, commands and versions

| Model | In `recast-woof` 1.1.7 | Its own package | Command | Python import |
| --- | --- | --- | --- | --- |
| WOOF-ArWen | yes, gpuwm 3.0.4 folded in | [`gpuwm`](https://pypi.org/project/gpuwm/) | `woof` in `recast-woof`; `gpuwm` in its own package | `woof`; `gpuwm` |
| WOOF-HEX-A | yes, gpuwm-hex 0.6.2 folded in | [`gpuwm-hex`](https://pypi.org/project/gpuwm-hex/) | `woof hex` in `recast-woof`; `gpuwm-hex` in its own package | `woof.hex`; `hexcore` |

The names are what people read. The package, command and import names do not
change with them, so scripts, installs and saved configurations keep working.
The engines keep their own version numbers: WOOF 1.1.7 is the version of
`recast-woof` and `recast-woof-data`, and it lists the engine versions it
carries.

On the WOOF site's API the two models keep their values:

| API `model` value | Model |
| --- | --- |
| `regional`, `cyclone` (and `kind: arwen` in run lists) | WOOF-ArWen |
| `hex` | WOOF-HEX-A |

## Which installs can share an environment

- **`recast-woof`** carries WOOF-ArWen and WOOF-HEX-A together.
- **`gpuwm` and `gpuwm-hex`** are the same two engines under their own names.
  gpuwm-hex 0.6.2 needs `gpuwm>=3.0.0,<3.1`.
- **Not `recast-woof` beside `gpuwm`.** Both install the same top-level
  packages (`tools`, `tilestream`, `configs`, `docs`, `mpas_cycle_bridge`)
  and the same `rw-wps` command, so the second install overwrites the
  first's files. Use one or the other.

The WOOF-HEX-A derivative commands (`woof hex tangent`, `adjoint-test`,
`sensitivity`, `param-grad`) need `recast-woof[ad]` in a Python 3.14
environment of its own; the [README](README.md) gives the commands.
