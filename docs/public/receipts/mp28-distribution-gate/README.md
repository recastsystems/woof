# Receipts: the mp28 distribution gate

Everything the gate declared in
`docs/public/validation/mp28-distribution-gate.md` read or produced,
committed beside the document. The declaration was approved at design
commit `0d69a648` before any build started; the evaluator
(`tools/mp28_matched/distribution_gate.py`) transcribes its constants.

- `distribution-gate.json`, THE gate receipt: all four row sets, both
  arms' order statistics, every screen, and `verdict.outcome`.
- `node1/`, `node2/`, each weather node's `provenance.txt`,
  `SHA256SUMS-node.txt` (every raw artifact hashed on the node before
  transfer), `chain.log` (stage log with coexistence snapshots), and the
  scripts exactly as executed there: the four banked recipes (adapted in
  the work-root path and, on this OS generation, the `NETCDF` root, an
  environment path; see ADAPTATIONS below) plus the orchestration
  wrapper and CPU provenance collector.
- `local-provenance.txt`, the GPU half: device, driver, toolchain,
  WOOF-ArWen source commit, table-root hashes, the canonical
  derived-constant asset check, post-transfer input hashes (the
  transfer manifest), and the WOOF-ArWen output hashes.
- `arwen-phase.log`: the four WOOF-ArWen runs' stdout with before/after GPU
  coexistence snapshots.
- `sw-arwen-*-run.json`, each WOOF-ArWen run's own receipt.

ADAPTATIONS to the banked scripts, in full: (1) the work root, `mp28sw`
→ `mp28dist` (each under its node's own home directory), the adaptation
the declaration names; (2) `NETCDF=/usr` → `NETCDF=<work root>/nc`,
where `nc/` is a directory of symlinks (`include` → `/usr/include`,
`lib` → `/usr/lib/x86_64-linux-gnu`, `bin` → `/usr/bin`): Ubuntu 26.04
puts the
netCDF libraries in the multiarch directory, WRF's configure probes
`$NETCDF/lib`, and without the view it silently omits `-lnetcdff` and
the link fails; (3) after `./configure`, one `sed` on `configure.wrf`
rewrites the C compiler line to
`SCC = gcc -std=gnu89 -fpermissive -Wno-implicit-function-declaration
-Wno-implicit-int -Wno-incompatible-pointer-types`: gcc 15 makes
implicit declarations and C89 empty-parameter declarations hard errors,
which stops WRF's legacy grib1/grib-share C (`pack_spatial.c`'s
`unsigned long grib_local_ibm();`) from compiling at all, the archive
step then aborts and every executable link fails on MEL_grib1 symbols.
The dialect flags affect only how that pre-C99 IO code is *accepted*;
`SFC`, both `FCOPTIM` variants, the configure option (32, GNU serial)
and the source are byte-unchanged from the banked recipes, and the
Fortran that computes every gated number is compiled exactly as banked.
All three adaptations are environment plumbing with no numerical
content, and the scripts as executed are committed here, with one
release-hygiene delta: the machine-local work-root prefix in the
committed copies is relativized (the release bar is zero developer
paths), so `node2/script-hashes.txt` (which hashes the bytes both
nodes actually executed, verified identical across them) will not
match a rehash of these copies. The relativizing commit records the
exact string mapping; nothing else differs.

Precedent: the em_les oracle lane had already built WRF v4.6.1 on these
same nodes (`tools/wrf_em_les_oracle/README.md` on its branch) with the
same two node-specific fixes, a netCDF shim prefix of symlinks over
the Ubuntu multiarch layout, and a GCC-15 legacy-C dialect flag (theirs
`-std=gnu17` appended to the `-w -O3 -c` flag string; this lane's
`-std=gnu89 -fpermissive -Wno-implicit-*` on `SCC`, a superset spelling
of the same fix class, adopted independently before that recipe was
pointed out). Their build is dmpar/em_les; this gate's is the banked
serial option 32: the Fortran side is untouched by either spelling.
