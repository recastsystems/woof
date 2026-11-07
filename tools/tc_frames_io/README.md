# Canonical TC frame writer

Internal Rust C ABI, version 3, for reading and publishing device-computed field updates in an existing `gpuwm-mapped-frameset-v1` source. Python orchestration is `woof.tc.frames.read` and `publish`. Build with `cargo build --release --locked`; output `libtc_frames_io.so`. Linux file I/O uses parallel field reads and bounded positioned writes. Numerics stay in CUDA. Reader buffers must be writable, disjoint and remain valid until return; failed reads expose no usable fields.

The caller seals the transform and background manifest hashes before computing any update. One exact valid time/member is patched, field geometry stays unchanged, and all source field digests are checked, including unchanged times. Both payload and header data references are updated. The new output directory becomes consumable only when `frames.json` is published last. Failed outputs have no published manifest; do not retry into the same directory. Input arrays must remain valid and immutable through the C call.

This is storage plumbing, not a completed relocation initializer. Unit checks cover header/payload digest agreement, later-frame preservation, corrupted untouched input and changed background identity. They do not establish forecast OFF equivalence.

ABI 3 adds an exact-lattice storage overlay: canonical before/after bits select replacement cells on the native decoder lattice. Unchanged cells retain the native source bits, including its rounding. Numerical transforms remain in CUDA.
