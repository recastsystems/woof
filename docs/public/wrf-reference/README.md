# WRF reference manifests

A matched-run comparison is a claim about two binaries on one case. The
[certification capsule](../DETERMINISM.md) witnesses the WOOF side. These
manifests are the other side: which WRF executable produced the reference
stream, from which build recipe, under which namelists, and with which output
bytes.

Each manifest is a `gpuwm.wrf-reference-manifest/v1` document validating
against [`manifest.schema.json`](manifest.schema.json), named by the SHA-256 of
the WOOF configuration it is the counterpart of. Identity, not a case name:
two configurations that differ by one byte get two manifests, and a manifest
cannot drift onto a configuration it was not measured against.

## What certify requires

`woof certify` refuses to return a verdict unless all four hash groups are
present:

| Field | What it pins |
|---|---|
| `wrf_exe_sha256` | the executable bytes that produced the reference stream |
| `build_recipe_sha256` | the committed recipe that binary was built from |
| `namelist_sha256` | every namelist the reference run consumed |
| `reference_wrfout_sha256` | every reference frame the comparison scored |

The refusal is unconditional: it does not consult a flag, and there is no way
to ask certify to proceed without it. A verdict that did not know what it was
compared against is not a verdict.

## What is here, and what is not

The reference wrfout stream and the meteorological inputs are not
redistributable by this repository, so only their hashes appear. The build
recipe and the namelists are text, and they are committed here in full
alongside their digests. Two things differ from the files the reference run
used: each `namelist.wps` gives `geog_data_path`, a folder on the machine
that ran WRF, as `/path/to/WPS_GEOG`, and each recipe names its build machine
generically. `namelist_sha256` and `build_recipe_sha256` hash the published
bytes.

Those two are therefore checkable from a clone, and are checked: for every
manifest here, `tests/test_acceptance_band.py` recomputes the digest of each
artifact the manifest names and ships beside it
(`gpuwm.certify.wrf_reference.mismatched_reference_artifacts`). `certify`
itself asks only whether each of the four values is a well-formed digest, not
whether it hashes anything -- a value that pins nothing satisfies
`wrf_reference_hashes_present` -- which is what that repository test exists to
catch, and it was catching something: this manifest's `build_recipe_sha256`
did not hash the committed recipe until the value was recomputed from it.

A manifest may carry an `unmeasured` block naming, per hash group, why it is
absent. Such a manifest parses, validates, and is **refused** by certify. The
block exists so the refusal is legible: a reader who runs certify and sees
`wrf_reference_hashes_present` refuse can read here what has to be measured and
where.
