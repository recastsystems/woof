# Execution backend interface

CUDA is the default. `WOOF_BACKEND=amd` or `--backend amd` selects the
experimental AMD ROCm adapter before model imports. Selection freezes at
first use. `select`, `get_backend`, `array_module` and registered factories
share one process-wide contract across the engines. A backend supplies
`name`, `xp`, `is_host`, `synchronize`, `compile` and `to_host`; optional
memory, occupancy and compilation-target hooks describe its real device.

Both adapters compile the canonical kernel sources. `hip-nvidia` is an
internal translation qualification mode. See `docs/amd-backend.md`.
