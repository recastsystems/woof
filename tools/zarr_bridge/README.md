# Zarr bridge

The vendored OpenSSL Makefile uses its configured runtime directory for engine and module search paths. Its local installation directory remains inside the Cargo build output. This packaging patch avoids embedding a build user’s directory in the static binary; no cryptographic implementation changes.
