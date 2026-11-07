# Retained lane registry documents

`gpuwm/physics_registry_history.json` keeps five batch007 lane registry
documents (83b3749be2) so preparation receipts written under those lanes still
resolve. Their lane commits were exported with `git am` and are not in this
repository's history, so `tools/registry_physics_history.py --check` reads each
document by its recorded `registry_blob_sha1`. These files are those documents,
byte for byte: each file's git blob id is its file name, which makes the blob
present in every clone. Never edit them.
