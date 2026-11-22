# Offline Zarr reader

Registry dependencies resolve from `../rustwx/vendor/crates-io`, shared with
the other offline workspaces. Existing locally corrected rustwx copies are
retained; `tools/rustwx/VENDOR.md` records those corrections.
`vendor-manifest.json` records each registry checksum and declared licence.
The source reader owns acquisition orchestration, while zarrs owns Zarr and
codec decoding, netcrust owns NetCDF reads, and WOOF-ArWen's existing NetCDF writer
and mapped engine supply writes and meteorological derivations.

Build from this directory with `cargo build --release --locked --offline`.
Linux selects vendored OpenSSL for zarrs_http's native TLS transport, avoiding
a runtime dependency on the build host's libssl ABI. The release build still
checks every ELF dependency and symbol version against its glibc 2.28 policy.
The new reader must pass that check before a Linux release is sealed.

Add missing crates using the scratch-vendoring instructions in
`tools/rustwx/VENDOR.md`; never regenerate or overwrite the shared registry.
Then run
`python tools/update_zarr_license_notice.py` from the repository root.
The generated notices travel in both distributions and the bridge asset tree.
They include nested native-library licence texts and distinguish the complete
build closure from the packages actually linked into a target binary.
