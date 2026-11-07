Raw `.bin` output and prepared `.cases` inputs were disposed after their
hashes and diagnostic results were recorded. The generator, manifests,
case index and historical SHA256SUMS remain. Regenerate with
`tools/camzm_wrf461_oracle/build.sh` and point `CAMZM_ORACLE_DIR` at the
result. Missing data skips the dependent diagnostic and grants no pass.

The `-stock` stream is the unmodified native reference. The unsuffixed
stream is a historical altered-source attribution diagnostic. Both use
generated soundings, not captured inputs from a real forecast.
