WOOF-ArWen keeps a surviving nest's grid ID when another nest is removed. WPS
continues to use compact Fortran array slots. For example, WOOF-ArWen domains
`[1, 3, 4]` with parents `[0, 1, 3]` occupy WPS slots `[1, 2, 3]` and use
WPS `parent_id = 1, 1, 2`. A standard Fortran comment binds that slot order:

```fortran
! GPUWM_DOMAIN_IDS_V1 = 1,3,4
```

`woof.wps_domain_ids` owns this metadata. An absent comment means the
historical IDs `1..max_dom`; ordinary contiguous output remains unchanged.
The declared IDs must be unique positive integers, begin with root d01,
and match `max_dom`. Duplicate, malformed, unsupported-version and
mis-sized declarations fail before use. Experiment order must contain one
root and place each parent before its children.

The WPS renderer writes compact parents. Native preparation binds both the
declared ID order and parent slots to the experiment before its existing
exact binary64 geometry and explicit vertical-coordinate checks. Geography
selection uses a domain's declared slot. Candidate editing retains each
survivor's `geog_data_res`; a missing Fortran entry remains `default`, and
a new nest inherits its parent's effective selection. Tile rebasing and
remote staging preserve the mapping while relocating declared paths.

Native hierarchy and artifact checks require the exact stable ID inventory,
with every published prepared cache accounted for. Prepared scientific
arrays, static hashes, physics admission and runtime coupling are unchanged.
Stock WRF namelist import retains its separate contiguous-ID contract; this
comment does not reinterpret a stock namelist's grid IDs.

The companion Remove action creates a new configuration and WPS companion.
It preserves the root, requires explicit inclusion of descendants, and
refuses surviving tracking, relocation or output references to a removed
domain. Removing the owner of a relocation policy removes that policy.
Original files remain unchanged. The UI previews the exact subtree and
requires the backend's `remove_nest` capability before enabling the action.

Regression coverage includes native middle-nest removal and a later edit,
exact surviving outlines, distinct geography choices, missing/reordered
metadata, a wrong parent hidden by numerically identical sibling geometry,
canonical artifact publication/reuse, remote and tile rebasing, and the
existing native geometry/physics regressions. These are configuration and
preparation-contract checks; they do not claim a new GPU forecast run.
