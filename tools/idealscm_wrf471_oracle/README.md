# WRF 4.7.1 ideal SCM surface replay

The native observer writes the six forcing scalars, all native arguments
and all outputs at entry and exit of IDEALSCMSFCLAY. It changes no native
arithmetic. Install it in a private strict WRF tree with:

```
python instrument.py /private/WRF/phys/module_sf_idealscmsfclay.F
```

Build the whole native program and run a real case with surface layer 89,
PBL 0 and land surface 0. Set `MISC_SF89_CAPTURE_DIR` to an existing folder.
The observer captures calls 1, 2, 30 and 100 from every tile.

Replay the retained streams through the production GPU launcher:

```
GPUWM_WRF_EXACT=1 python replay.py CAPTURE_FOLDER replay.json
python read_frames.py > frames-fast.json
GPUWM_WRF_EXACT=1 python read_frames.py > frames-strict.json
```

Every output word is checked, including outputs the native routine leaves
untouched. No undefined-word exclusion is applied. A zero result alone does
not qualify the option: the whole forecast and interleaved stage timing
must also pass. The lane report names the retained real-input capture and
its manifest.
