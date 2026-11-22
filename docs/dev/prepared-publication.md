# Publishing a prepared WOOF-ArWen release

The [publish workflow](../../.github/workflows/publish.yml) promotes a complete,
qualified set of files. It does not rebuild release distributions. Keep the
prepared packet and its verification records until publication and any retries
are complete.

## What a failed run means

A failed workflow may already have uploaded some or all files to PyPI. Inspect
the upload logs and the public file inventory before deciding how to recover.
A Git tag and a PyPI distribution filename are separate objects: a test failure
before upload does not consume a filename, while a later verification failure
does not undo an upload.

PyPI permanently reserves uploaded filenames, including after deletion. An
existing filename may be reused by this workflow only in the sense that its
already-published, matching bytes are skipped. Changed content needs a different
release version; deleting a distribution does not make its filename available.
See [PyPI's filename policy](https://pypi.org/help/#file-name-reuse).

The public JSON index can lag behind an accepted upload. A missing version or
partially populated response is retried within a bounded deadline. A conflicting
hash, unexpected file, or yanked file stops reconciliation. If a deadline expires,
retain the original packet and retry after inspecting the index. Do not rebuild,
retag, or delete files as an automatic response to a red workflow.

## Prepare the source and files

Finish documentation, packaging, and workflow changes before sealing the public
source commit. The eventual release tag must point at that exact commit. Authored
native artifacts contain the full commit identifier; build them from the public
checkout, rather than an internal checkout whose commit differs even if most
files are identical.

Committed [bridge pins](../../woof/data/bridges/bridge-pins.json) remain empty.
Generate the working release pins from the qualified Windows and Linux bundles
using [build_bridge_bundle.py](../../tools/build_bridge_bundle.py), and run
[verify_release_build_tree.py](../../tools/verify_release_build_tree.py) and
[verify_release_artifacts.py](../../tools/verify_release_artifacts.py). Only the
declared working pin file may differ during distribution assembly.

For a release version `X.Y.Z`, prepare all six PyPI distributions:

- `gpuwm-X.Y.Z-py3-none-any.whl`
- `gpuwm-X.Y.Z-py3-none-manylinux_2_28_x86_64.whl`
- `gpuwm-X.Y.Z-py3-none-win_amd64.whl`
- `gpuwm-X.Y.Z.tar.gz`
- `woof_data-X.Y.Z-py3-none-any.whl`
- `woof_data-X.Y.Z.tar.gz`

The platform wheels, native bundles, desktop TUI files, installed Python
inventories, integration kit, and qualification records must agree on this cut.
Run strict Twine validation on all six final files with a metadata-2.4-capable
toolchain. The publication workflow uses Twine 7 or newer. Validate the README
inside the distributions as well as the source file.

## Stage the complete release

Create `PUBLICATION-ASSETS.json` with schema `arwen.publication-assets.v1`.
It binds `engine_source_revision` to the public tag commit and records the
separate `desktop_source_revision`. Its `github.repository` and
`github.target_version` identify the destination. Every `github.assets` and
`pypi.artifacts` row has a `filename`, byte count in `bytes`, and lowercase
SHA-256 in `sha256`.

The GitHub asset list includes both native bundle ZIPs and
`bridge-bundle-manifest.json`, plus the selected desktop packages, integration
kit, source archive, data assets, documentation, and qualification records.
Include the six PyPI files in the GitHub release as transport inputs even when
they are listed only under `pypi.artifacts`. The workflow requires the exact
union of the two lists, the manifest itself, and declared auxiliary files.
It rejects both missing and undeclared assets. Use explicit filenames for
uploading; a wildcard over a working directory can include an old candidate.

The manifest cannot hash itself. `github.also_attach` may name
`PUBLICATION-ASSETS.json` and `DOWNLOAD-SHA256SUMS.txt`; the checksum file, when
present, must agree with the declared files. Calculate the final manifest hash
after all contents and filenames are fixed. Keep that hash with the packet.

Stage the complete files on the release before invoking publication. Both
supported trigger motions use the same byte checks:

1. Dispatch `publish.yml` with the exact tag selected as the workflow ref,
   `release_tag` set to that tag, and `publication_manifest_sha256` set to the
   prepared manifest hash. A matching draft can be promoted by the workflow.
2. Publish the prepared GitHub release with a single hash marker in its body:

   ```html
   <!-- arwen-publication-sha256: INSERT_THE_64_CHARACTER_SHA256_HERE -->
   ```

If both an explicit hash and a body marker are supplied, they must match. The
optional stable-version and release-immutability inputs retain their existing
meaning; the workflow does not silently change the release's prerelease flag.
Pull-request runs perform validation and cannot publish.

The PyPI Trusted Publisher coordinates remain the repository's `publish.yml`
workflow and `pypi` environment for both projects. Confirm the project settings
when maintaining those bindings; possessing a prepared packet does not verify
the private PyPI configuration. See the
[Trusted Publisher setup documentation](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Promotion and recovery

Before its first public write, the workflow verifies the captured manifest,
every declared asset, the tag commit, package metadata, native bundle pins,
fresh Windows and Linux installations, and both PyPI file inventories. It
rechecks source and asset identity at the publication boundaries.

The publication order is:

1. Make the verified GitHub native and data assets public.
2. Upload missing matching-version `recast-woof-data` files and verify both index
   entries.
3. Upload missing Windows and Linux engine wheels and verify their index entries.
4. Upload the universal engine wheel and source distribution, then verify the
   complete four-file engine inventory.
5. Record the release ID, tag commit, manifest hash, GitHub assets, and exact
   PyPI file sets in the final workflow receipt.

GitHub and PyPI do not support a shared atomic transaction. There can be a short
interval in which the GitHub release is public but pip publication is incomplete.
Check the successful final receipt before announcing that installation is ready.
The asset-first order ensures the engine's native and data download URLs are
available when the engine reaches PyPI.

For a retry, use the same tag, source commit, manifest hash, and prepared files.
The workflow revalidates them, skips already-published files only when their
names, sizes, and hashes match, and uploads the remaining files. It does not
replace assets, recreate tags, increment versions, or regenerate distributions.
A 404 from the public index is not proof that a filename was never uploaded and
subsequently deleted; retained release records remain necessary.
