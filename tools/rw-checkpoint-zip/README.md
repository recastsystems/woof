# Parallel lossless checkpoint compression

Build `cargo build --release --locked` here. Set `WOOF_RESTART_ZIP_TOOL` to
the executable and `WOOF_RESTART_ZIP_THREADS` to the allowed CPU count.
The restart writer invokes it before atomic publication. Unset the tool
variable to retain the original uncompressed writer.

Each NPY entry is independently DEFLATE-compressed at level 1 using Rayon;
assembly copies compressed entries in original archive order. The tool does
not interpret or change array values, headers, names, order, or floating-point
bits. It refuses existing output files. The writer retains its file and
directory fsync and root-last tree publication. Compression stage times and
sizes are emitted as JSON alongside forecast logs.

This source path is opt-in and awaits full-case compression cost and restart
qualification. Compressed archive bytes differ; decoded checkpoint arrays
must remain bitwise identical. Peak host memory includes the compressed
archive plus one codec workspace per worker. Retention still belongs to the
engine; this tool never deletes a previously published checkpoint.

Dependencies are permissive: zip 8.6.0 (MIT), Rayon (MIT or Apache-2.0),
anyhow (MIT or Apache-2.0), serde_json (MIT or Apache-2.0), and zlib-rs
(zlib). Their upstream notices accompany the dependency sources.
