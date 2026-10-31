# Device-memory census for a hex run (CuPy pool, by call site)

Used for the conus3km memory lane (RRFS-NATIVE-FREE-2026-10-28/CONUS3KM/memory.md).

- Run any hex forecast with `PYTHONPATH=<this dir>` and `HEX_MEMCENSUS=<out>.{pid}.json`. `sitecustomize.py` installs a
  CuPy MemoryHook in the main thread and in every thread started later (CuPy hooks are thread-local, so rank threads
  must be covered). No device syncs are added. Output: the live peak and its sites, each module's own high-water live
  set, per-site maxima, the live set at exit and a coarse timeline.
- `fit.py`, `estimate.py`: fixed + per-cell fit between two meshes, per phase, with the elastic sites (RRTMG batches,
  the MYNN-EDMF arena) left out. `estimate.py` prints the need at given local-column counts; its NONPOOL constant is
  the CUDA context and kernel images measured on an unshared RTX 5090.
- `phases.py`, `phasediff.py`: what is live when each phase peaks, and what it holds beyond the resident set.
- `cmpdirs.py`: every word of every NetCDF file two runs wrote (bit patterns); the file sets must match.
