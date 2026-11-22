# Forecasting from your own mapping

A caller-authored mapping can prepare and run through the same forecast engine
as a packaged source. `woof sim` reports `source=mapped` when the preparation's
mapping, composition and manifest do not identify a packaged profile. Forecast
skill and stability remain properties to measure for your data and settings;
absence of a packaged profile or a previous acceptance run is not a refusal.

Use the data and provenance role names declared in your composition. For example,
a composition declaring `terrain` and `terrain_provenance` can be prepared with:

```sh
woof prep --source mapped --source-format grib2 \
  --mapping mapping.json --composition composition.json \
  --input-list inputs.txt --supplement terrain=terrain.grib2 \
  --provenance terrain_provenance=terrain-provenance.json \
  --author-input-manifest input-manifest.json \
  --experiment-config experiment.toml --wps-namelist namelist.wps \
  --geog-root WPS_GEOG --preprocess-backend cpu --output-root prepared
woof sim prepared --experiment-config experiment.toml \
  --wps-namelist namelist.wps --outdir forecast
```

The experiment retains your physics, domain tree and other runtime settings.
A named physics profile is optional. The stage selects the tree runner for a
nested preparation; `--runner single` selects d01 alone. `--print-command` prints
the underlying runner command so scripts can use its additional execution
controls, including the single-domain runner's `--tiles` JSON option.

Keep the prepared bundle together. It contains the copied mapping, composition,
input manifest and provenance, plus receipts binding their hashes to every
prepared cache. These identities are rechecked before execution and after the
run. Changed or missing arrays, mismatched configuration, or altered evidence
still fail. Large raw inputs may be moved after preparation; their recorded
identities remain bound into the bundle. An explicit packaged-source claim must
match that profile's actual authority bytes.

`--stock-wrf-export off` skips the optional companion files for running unchanged
WRF. Both the full tree and d01-only forecast can still use the prepared caches.
An optional export refused because WRF cannot represent the chosen settings is
reported in the preparation receipt; it does not disable WOOF-ArWen's implemented
forecast settings.
