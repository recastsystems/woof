# Input and output recovery

WRF input soil moisture uses volume fractions. Some upstream WPS workflows
send layer water amounts to `real.exe` before converting them. The resulting
`SMOIS` may still be labelled as volume fraction even though its values are
wrong. WOOF-ArWen recovers this cold-initialization case from the original source
layers, converts their amounts using their own thicknesses, and then repeats
the vertical interpolation. Dividing the final four WRF layers by their
thicknesses would give a different, incorrect answer.

Keep the first matching `met_em.dNN.*.nc` and its producing `Vtable` beside
`wrfinput_dNN`. An ordinary `woof run --wrfinput DIR` then discovers the
source inputs automatically when land `SMOIS` exceeds a volume fraction of
one. If the original WPS files are elsewhere, supply their directory:

```sh
woof run --wrfinput WRF_RUN --soil-source ORIGINAL_WPS --outdir FORECAST
```

The producing table may declare more soil layers than that cycle actually
carried, which is the ordinary case for a table that covers every layer its
source can publish. Each stacked depth is paired with the declared layer whose
bounds hold it, and converts on that layer's thickness. A depth that no
declared layer holds, a pairing that would run out of order, and a file that
stacks more layers than the table declares are refused with the stacked depths
in centimetres and the declared bounds in metres both printed, beside the
met_em and the table the run read and the rule that chose each. A source plane
carrying WRF's one extra staggered row and column keeps its leading mass
block, admitted only when the leading block of both source coordinates is this
domain to the bit; any other extent is refused with both extents named.

The original files remain unchanged. Recovery checks the initialization time,
domain coordinates, source depths, and a forward reconstruction of the bad
values before using the corrected ones. Already physical input values remain
unchanged. Existing liquid water is preserved only when it is consistent with
the recovered total. A separately supplied liquid source needs its own quantity
and layer authority; total water cannot identify the frozen/liquid partition.
The import receipt records source hashes and the conversion, and these inputs
participate in preparation and restart identity. A stepped WRF state cannot
be replaced by this cold-initialization recovery.

Custom source tables can provide `wrf-soil-authority.dNN.json` (or a common
`wrf-soil-authority.json`) instead of a WPS `Vtable`. Its schema is
`gpuwm-wrf-soil-authority-v1`; it declares `source_variable`,
`source_depth_variable`, `source_quantity`, `source_units`,
`source_layer_depths_m` and `source_layer_bounds_m`, ordered by ascending
source depth. Supported quantities are `layer_water_mass`,
`equivalent_water_depth` and `volume_fraction`. These declarations must come
from the producing source. Without the original layer evidence an already
interpolated file is underdetermined; use native model preparation or recover
the original WPS inputs.

A prepared forecast retry keeps the earlier forecast at its original address.
The next attempt receives a separate generation containing its forecast and
pictures. The same selected path reaches execution, early rendering, final
rendering and the returned summary. Existing receipts therefore continue to
name the original bytes, including a replayed valid time.

WRF-input and metgrid preparation reuse compares both source digests and the
resolved authority documents. Matching documents are read without rewriting
their bytes or modification times. Changed settings, missing authorities or
changed sources select a separate preparation generation. A later preparation
failure cannot rewrite the retained authorities. The returned input object
names the actual preparation directory.

Output inventories hash one open file descriptor. Its identity, size and
timestamps must remain stable, and the final literal path must still name
that file. A concurrent replacement or mutation fails finalization instead
of publishing a digest and size from different files.

A skipped render cannot publish a previous image as a new result. Successful
publications update the latest pointer atomically under a shared filesystem
lock. An older completion cannot replace a newer published launch.

Metgrid restart identity distinguishes scientific preparation from resource
observations. The versioned scientific authority includes the entire verified
import receipt except `memory_admission`. Source digests, domain metadata,
vertical-coordinate choices, cache content, configuration and runtime identity
remain bound. The raw import receipt is still digest-verified; changing it
after binding is an error. Direct reuse retains the original preparation's
memory observation instead of rewriting the receipt with a later reading.

The scientific authority has an explicit new identity. Older checkpoints
bound to the raw import-receipt digest are not silently migrated or exempted
from comparison. They require their matching implementation. This metadata
change does not add WRF warm-state import or change any physics calculation.
