"""Locate the built Rust bridge executables outside a source checkout.

A platform wheel SHIPS the compiled Rust: the fail-closed GRIB decoders
and the CPU preprocessing library are built from the vendored
``tools/grib1_bridge`` workspace (``cargo build --release --locked
--offline``) and staged into :func:`packaged_bridge_dir` before the
wheel is built, so ``pip install recast-woof`` lands a complete install on
every platform a bundle exists for.  This module is the single
resolution mechanism shared by ingest (:func:`woof.ingest.grib
.build_rust_bridge`), ``woof doctor``, and documentation:

1. an explicit per-executable environment variable
   (:data:`BRIDGE_ENV`) naming the built file;
2. a source checkout's own ``tools/grib1_bridge/target/{release,debug}``
   build tree (the developer path -- ingest may also *build* there);
3. ``<root>/libexec/bridges`` beside the package (the sealed runtime
   archive layout);
4. :func:`packaged_bridge_dir` -- ``woof/libexec/bridges`` INSIDE the
   installed package, which is what a platform wheel carries.  It sits
   below the checkout rungs so a developer's own build still wins, and
   above ``~/.woof/bridges`` because bytes shipped with this version
   are version-matched by construction while a fetched bundle is only
   as fresh as the last ``woof fetch-bridges``;
5. the user-level default directory :func:`default_bridge_dir`
   (``~/.woof/bridges/<release>-<bundle digest>`` for a pinned
   install, the flat ``~/.woof/bridges`` otherwise), which ``woof
   fetch-bridges`` (:mod:`woof.bridge_assets`) stages the release's
   prebuilt bundle into, and where a wheel user otherwise copies their
   own build once;
6. for a pinned install only, the flat legacy ``~/.woof/bridges``
   (:func:`legacy_bridge_candidates`), read when its bytes are this
   release's pin and never written.

The ``py3-none-any`` fallback wheel -- the one pip resolves on a
platform with no published bundle -- carries no rung 4, and every
consumer of a missing artifact must refuse BY NAME with
:func:`artifact_remedy` rather than degrade into a Python
reimplementation of the decoder.

Nothing here runs cargo.  Resolution has one side effect and one
only: an artifact staged in rung 5 that is not the one this release
pinned is re-fetched before it is handed to a door
(:func:`require_release_pin`), because that rung is the one a ``pip
install -U woof`` leaves untouched -- new Python, last release's
binaries.  Readers that must see the estate as it is, ``woof doctor``
first among them, resolve inside :func:`inspection_only`, where the
guarantee is the original one.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re
import shutil
import sys
import threading

#: Bridge executable -> the environment variable that names a prebuilt
#: copy.  The same variables drive the sealed-runtime decoder binding in
#: :mod:`woof.source_cli`, so one mechanism serves both install modes.
BRIDGE_ENV = {
    "grib1_bridge": "WOOF_GRIB1_BRIDGE",
    "met_intermediate": "WOOF_MET_INTERMEDIATE",
    "gfs_grib2_bridge": "WOOF_GFS_GRIB2_BRIDGE",
    "hrrr_grib2_bridge": "WOOF_HRRR_DECODER",
    "grib2_inventory": "WOOF_GRIB2_INVENTORY",
    "grib2_dump": "WOOF_GRIB2_DUMP",
    "gdt101_remap": "WOOF_GDT101_REMAP",
}
# `gpuwm_mapped_engine` is deliberately NOT in this map, for the same
# reason `rw_netcdf` is not: this map's consumers resolve through
# :func:`crate_dir`, which is the grib1_bridge crate, and the mapped
# engine builds in `tools/rw_wps`.  An entry here would make
# :func:`find_bridge` miss a checkout build and answer from a staged
# copy instead -- a stale-binary answer wearing a fresh-checkout face.
# Its ladder lives in :mod:`woof.mapped_engine_bridge`; its contract
# marker is in :data:`BRIDGE_ABI_MARKERS` below, which is keyed by
# artifact and not by this map.

#: The bridge crate's path inside a checkout.
CRATE_RELATIVE = "tools/grib1_bridge"

#: Bridge executable -> a byte marker of the CONTRACT it speaks, which
#: must appear in the built binary.
#:
#: A bridge is not "whatever executable has the right basename".  The
#: wheel ships no Rust, so the binaries on a machine were built from
#: some checkout at some time, and an upgrade of the Python half does
#: not touch them.  1.1.0 changed the GFS series file from two columns
#: to three; a 1.0.1 bridge still launched, still printed its usage
#: diagnostic, and `woof doctor` therefore reported it `ok` -- and then
#: every preparation died with `series line 1 must be HOUR<TAB>GRIB2`,
#: a message that blames the series file woof had just written
#: correctly.  A test host G validation run found the cause only by diffing
#: two git tags.
#:
#: Each marker is a literal the CURRENT contract compiles into the
#: binary and the previous one did not, so a stale build fails the
#: handshake statically -- no execution, no new bridge CLI surface, and
#: it works on the already-built binaries a user has on disk today.
#: This is the same mechanism `woof.native_wrf_distribution` applies
#: before sealing a distribution, applied one step earlier: at the
#: doctor report, which is where a user looks before they burn a run.
#:
#: Adding a marker is part of changing a bridge contract.  Choose the
#: literal that spells the contract out (the series grammar, the usage
#: line naming the argument vector), never a version number, which a
#: rebuild bumps whether or not anything changed.
# Optional contracts are probed by their caller. They must not make a
# library with a working base contract unusable for its existing entry points.
BRIDGE_OPTIONAL_ABI_MARKERS = {
    "static_fields": {"extra_continuous": b"gpuwm_static_extra_continuous_v1"},
}

BRIDGE_ABI_MARKERS = {
    "rw_icechunk": (b"rw_icechunk --request REQUEST.json schema=icechunk-output.request/v1 "
                    b"modes=run,append,finalize progress=jsonl native-arrays=v1"),
    "rw_ice_verify": b"gpuwm-ice-observation-binding.v1",
    "rw_fire_indices": b"GPUWM_FIRE_INDICES_V1",
    "rw_ocean": b"gpuwm.ocean.prepared/v1",
    "rw_bathy": b"gpuwm-rw-bathy-v1",
    "rw_ocean_verify": b"gpuwm-ocean-sst-verify-v1\tdaily-mean\tndbc-WTMP\tgreat-circle\tpaired-change",
    "rw_tc_fields": b"gpuwm.tc-fields.v1 inspect extract tempest table-units masked-quarter-degree",
    "nextgen_hydrofabric": b"arwen_hydrofabric_read_into",
    "arwen_nextgen_forcing": b"arwen_forcing_read_packet",
    "rw_nextgen_verify": b"nextgen-gauge-table-v1",
    "rw_soundings": b"gpuwm-soundings.v1 plain-run png json html",
    "rw_globe_pack": b"gpuwm-rw-globe-pack-v1",
    "rw_ensbatch": (
        b"gpuwm-ensemble-diagnostic-reduce.v1\tf32-words\tf64-two-pass\t"
        b"member-order\tsha256\tmember-weights-v1"),
    "rw_probgrid": b"gpuwm-rw-probgrid-products-v1\tsevere\tprobability\tlow",
    "rw_verify": b"gpuwm.verify-visuals.request.v1",
    "rw_scorecard": b"gpuwm.obs-scorecard.manifest.v1",
    "rw_compare": b"gpuwm.reference-input.v1",
    # The nowcast frames converter's ``--abi`` line, which its main()
    # prints from the one constant it compiles in.  Spelled to match
    # woof.rustwx_lanes.NOWCAST_FRAMES_ABI_MARKER; a test binds the two
    # and both to the Rust source.
    "rw_nowcast_frames": (
        b"gpuwm-rw-nowcast-frames-v1\tfrom-netcdf\t--netcdf\t--out\t"
        b"--source-id\t--lattice\t--variable\t--lat\t--lon\t--issue\t"
        b"--causal\t--receipt\t--source-kind\t--package\t--revision"),
    "rw_simradar": (b"rw_simradar --request REQUEST.json "
                   b"schema=simulated-radar.request/v1 manifest=simulated-radar.manifest/v1 "
                   b"volume_paths=v1 scene_shapes=v1"),
    "rw_netcdf": b"dtype\t<f8\t|S1\twater_layer_conversion\tsource_soil_recovery",
    "gdt101_remap": b"arwen.gdt101-regional-remap.v1",
    # The reader's newest behaviour, not its record schema: two fixes
    # changed what it accepts without changing the record it prints.  A
    # parenthesised unit (ARCO ERA5 publishes its level axis as
    # "Hectopascal(hPa)") refused every request before the first chunk,
    # and a 30 s whole-request timeout cut every 80 MB chunk off on a slow
    # link.  A build older than both still carried the schema literal and
    # passed; that literal ("http transfer failed after ") arrived with the
    # later of the two fixes.  The marker now is the deep-soil temperature
    # producer's receipt schema: fetch-geog builds the default
    # soiltemp_era5_1deg with `rw_zarr soiltemp-climatology`, which an
    # older build answers with its usage line.
    # Older cached readers lack `coverage`, so live hydro requests ERA5 beyond
    # the published boundary instead of carrying its spin-up to live NWM soil.
    "rw_zarr": b"gpuwm.zarr-coverage.v1",
    "grib1_bridge": b"usage: grib1_bridge INPUT.grb OUTPUT_DIR",
    "gfs_grib2_bridge": (
        b" must be HOUR<TAB>GRIB2[<TAB>FORECAST_PROCESS_ID]"),
    "hrrr_grib2_bridge": (
        b"usage: hrrr_grib2_bridge WRFNAT_F00 WRFNAT_F01 SOIL_F00 "
        b"SOIL_F01 OUTPUT_DIR EXPECTED_CYCLE I_START I_END J_START "
        b"J_END"),
    # The inventory contract grew the ensemble-identity columns
    # (typeOfEnsembleForecast, encoded ensemble size, derived-forecast
    # statistic code) beside the perturbation number it always carried,
    # and then the pv column (Section 4's coordinate octets -- the
    # hybrid A/B coefficient channel).  The marker is the header tail,
    # which only a binary speaking the grown contract contains: a stale
    # build would inventory a hybrid model-level file without the
    # coefficients that price its pressure ladder, and the decode gate
    # would refuse it at run time with a rebuild remedy -- this catches
    # it statically instead.
    "grib2_inventory": (
        b"minimum\tmaximum\t"
        b"ensemble_type\tensemble_size\tderived_forecast\tpv"),
    "grib2_dump": (
        b"parameter\tcenter\tsubcenter\tmaster_table_version\t"
        b"local_table_version\tlevel_type"),
    # A library, so the literal is an exported symbol name rather than a
    # usage line: `bw_dealias_rift_v1` is the refinement entry point, and
    # it is the contract that matters here because the default engine
    # runs refinement.  A build predating it exports `bw_dealias` alone,
    # loads cleanly, answers the legacy ABI probe with 1, and then fails
    # inside the first refined solve -- which is exactly the class of
    # stale build this table exists to catch statically.
    "region_global_dealias": b"bw_dealias_rift_v1",
    # The parallel CPU preprocessing library.  A library, so the literal
    # is an exported symbol name, and it names the newest entry the land
    # surface depends on under BOTH backends: the native HRRR route's soil
    # stencil, built in the same change set as the masked surface chain
    # (soil moisture and temperature, snow, skin temperature, sea ice)
    # every other source maps its land through, then the water entries
    # (the lake skin search, the water-temperature blends, the water-body
    # labelling, the water repairs, the per-body assembly and the source
    # owner of each body), and newest of all the CPU backend's bounded
    # surface-nearest search.  A build predating them loads cleanly,
    # answers the ABI probe with 1 and maps every other field, and then
    # cannot map the land surface or assemble a water temperature at all.
    # The host Noah cold start also uses this library. A build without
    # its soil-liquid-water entry cannot prepare a frozen soil column.
    # Spelled to match woof.noah_init_bridge.NOAH_SH2O_ENTRY.
    "gpuwm_preprocess_cpu": b"gpuwm_noah_initialize_sh2o_f64",
    # The NetCDF writer cdylib behind the DEFAULT wrfout engine AND the
    # DEFAULT wrfinput/wrfbdy export.  A library, so the literal is an
    # exported symbol name, and it names the newest capability a default
    # path depends on: it was `gpuwm_ncwrite_write_record` while the
    # record dimension was that capability (a build predating it exports
    # every other entry point, loads cleanly, answers the version probe
    # -- and then cannot write a `Times` variable), and it is now the
    # read-back sweep, because the wrfinput export verifies every float
    # variable for finiteness before it publishes.  Spelled to match
    # woof.io.nc_writer_bridge.ABI_MARKER; a test binds the two.
    "netcdf_writer": b"gpuwm_ncwrite_scan_nonfinite",
    # The mapped decode engine behind `woof prep --source mapped`.  The
    # marker is a contract line rather than a usage line, and it carries
    # TWO contracts because this binary has two a stale build can break.
    # Its OUTPUT contract is the frameset schema: a binary built before a
    # frameset change still launches, still refuses politely, and would
    # then write a directory `woof.mapped_engine_bridge.read_frameset`
    # no longer reads -- the 1.1.0 GFS series-file failure class, one
    # layer further in.  Its DECODE contract is the set of Section-5 data
    # representations its vendored reader has a reader for: a binary
    # built before the IEEE-packed (template 5.4) reader landed passes a
    # frameset-only handshake, then meets a conformant 5.4 message and
    # refuses it as "Section 5 simple packing too short", telling the
    # user to re-fetch bytes that were correct all along.  Both move the
    # marker now.  Spelled to match
    # woof.mapped_engine_bridge.ABI_MARKER and
    # mapped_engine::ABI_CONTRACT; tests bind all three.
    "gpuwm_mapped_engine": (
        b"gpuwm-mapped-engine-abi frameset=gpuwm-mapped-frameset-v1 "
        b"height-interfaces=1 window-fields=omega bound-clocks=window-start "
        b"grib2-drt=0,2,3,4,40,41,42,50,51,61,200"),
    # The static-field builder cdylib (tools/rustwx/crates/static-fields),
    # the default engine for the WPS-geogrid-equivalent statics from the
    # static-rust-port lanes on.  A library, so the literal is an
    # exported symbol name: a build that loads and answers the version
    # probe but predates the field build cannot produce a single static
    # field.  Spelled to match woof.static.rust_bridge.ABI_MARKER; a
    # test binds the two.
    "static_fields": b"gpuwm_static_sampling_portable_v1",
    # The observation remap cdylib (tools/rustwx/crates/obs-regrid),
    # behind the DEFAULT plan build of the observation battery.  A
    # library, so the literal is an exported symbol name: a build that
    # answers the version probe but predates the plan builder cannot
    # produce a single remap, and the battery would fall back to scipy
    # -- whose tie-breaking is traversal order -- while reporting the
    # Rust engine as present.
    "obs_regrid": b"gpuwm_obsregrid_build_plan",
    "obs_score": b"gpuwm_obsscore_masked_fss",
    "met_intermediate": b"usage: met_intermediate --vtable VTABLE --date YYYY-MM-DD_HH:MM:SS",
    "rw_mpas_geometry": b"rw_mpas_geometry --protocol hex-geometry-v1",
    "rw_mpas_hostprep": b"rw_mpas_hostprep --protocol hex-hostprep-v1",
    # The isobaric-height reader cdylib (tools/rustwx/crates/rw-isobaric),
    # behind every Python consumer of a height on a pressure surface (the
    # vortex tracker on a host state, the GNSS-RO operator, the
    # verification maps, the flagship products).  A library, so the
    # literal is an exported symbol name: a build that answers the version
    # probe but predates the height reader cannot read one surface.
    # Spelled to match woof.isobaric_bridge.ABI_MARKER; a test binds the
    # two.
    "rw_isobaric": b"gpuwm_isobaric_heights",
    # The TEMPO tempo_init port's ctypes seam (woof.core.tempo_rust): the
    # blob-layout contract symbol, which changes when the blob does.
    "rw_tempo": b"gpuwm_tempo_init_v1",
    # The MPAS point renderer: its --abi line (output schema and stdout
    # tokens).  Spelled to match woof.mpas_mesh.POINTS_ABI_MARKER.
    "rw_hex_points": (
        b"gpuwm-rw-hex-points-v1\tPOINT\tSOUNDING\tMETEOGRAM\tFINISHED"),
    # The radar superob cdylib (tools/rustwx/crates/rw-superob), behind
    # the DEFAULT superob stage of every radar observation build (gate
    # placement, QC, the alias masks, the multi-radar merge and the
    # region-global dealias of each sweep).  A library, so the literal is
    # an exported symbol name.  Spelled to match
    # woof.obs.superob_bridge.ABI_MARKER; a test binds the two.
    "rw_superob": b"gpuwm_superob_call",
    # The MPAS mesh generator behind `woof mesh`.  The marker is its
    # ARGUMENT VECTOR, spelled out, because that is the literal which
    # changes exactly when the request contract changes: a binary built
    # before `--cells` meant a cell count still launches, still prints a
    # usage line, and would then size a mesh from a flag it reads
    # differently -- the 1.1.0 GFS series-file failure with a mesh on the
    # other end, and more expensive, because a mesh is minutes of
    # relaxation before anything looks wrong.  `--card` is in the vector
    # for the same reason and a sharper one: a build predating it accepts
    # `--vram-gib 16` alone and answers 79,717 cells from ONE card's baked
    # fixed term, which is the wrong number on any other part -- 133,144
    # on the 70 SM card, sized against a measured fixed term instead of a
    # borrowed one.  Spelled to match woof.mpas_mesh.MESH_ABI_MARKER; a
    # test binds the two.  `--triangulation` is in the vector for the
    # sharpest version of the same reason yet: a build predating it takes
    # `--triangulation incremental` as an unknown flag, generates on the
    # REBUILD arm anyway, and hands back a mesh that took the two hours the
    # caller was trying not to spend -- or, on the other side of it, a
    # caller who believes the flag took effect registers a digest against an
    # arm that never ran.
    "rw_mpas_mesh": (
        b"rw_mpas_mesh --out GRID.nc [--spec SPEC.json | --background-km KM "
        b"| --from-centres GRID.nc] [--cells N | --card KEY [--vram-gib X]] "
        b"[--fit-spacing yes|no] [--sweeps N] [--tolerance X] [--omega X] "
        b"[--receipt JSON] [--triangulation rebuild|incremental] [--clobber] "
        b"[--dry-run] [--list-cards]"),
    # The MPAS static builder, the other half of what `woof mesh`
    # delivers.  The literal is the argument vector for the same reason
    # the generator's is: a build predating the `--compare` grading route
    # or the `--nominal-dx-m` declaration still launches and still
    # answers `--version`, and would then write a static whose
    # nominalMinDc the mesh registry rejects -- after the geography pass,
    # which is the whole cost of the run.  Spelled to match
    # woof.rustwx_static.STATIC_ABI_MARKER; a test binds the two.
    "rw_mpas_static": b"rw_mpas_static --grid GRID.nc --out STATIC.nc",
    # The initial-condition builder, the mesh generator's sibling out of
    # the same crate.  Its own argument vector, truncated at the head
    # rather than carried whole: the crate's literal embeds a `\n` in the
    # middle of the vector, and a marker that has to reproduce an
    # embedded newline exactly is a marker that breaks on a reflow that
    # changed nothing.  The head names the four arguments whose meaning
    # a stale build would get wrong.
    "rw_mpas_init": (
        b"rw_mpas_init --met MET --static STATIC.nc --capsule CAPSULE.nc"),
    # The lateral-boundary producer, the fifth binary out of the same
    # crate.  The crate's literal is four usage lines with embedded
    # newlines between them, and a marker that has to reproduce one
    # exactly breaks on a reflow that changed nothing, so this is the
    # whole FIRST line and no more.  It runs through the physics-mode
    # switch (2.9.0, correct by default, so spelled default first):
    # unknown flags are read without complaint, so a build from before
    # the switch takes --physics-mode correct, writes native case-9
    # boundary winds (IN4) from any first-guess time (IN5), and a caller
    # relying on the default gets exact while 2.9.0 says correct.
    # Spelled to match woof.mpas_mesh.LBC_ABI_MARKER; a test binds them,
    # and binds both to the crate's literal.
    # The RRFS-MPAS initializer behind `woof hex init --physics-suite
    # hrrrv5` (the fork's init_atmosphere cases 7 and 9, its static and its
    # vertical grid, in Rust).  Its --abi line, the literal that changes
    # when its job contract does: a build from before the hex-init-job/v1
    # stages would take the door's JSON and write an init under another
    # contract.  Spelled to match the crate's rrfs_init::job::ABI_MARKER.
    "rw_mpas_init_v831": (
        b"rw_mpas_init_v831 stdin: hex-init-job/v1 JSON; stage=case7|case9|"
        b"static|terrain|vertical|artifact|prepare|gwd|surface"),
    # The GSL orographic statistics generator (ugwp_oro_data.nc for the
    # RRFS suites' bl_ugwp_gwdo drag), the head of its usage line.
    "rw_mpas_gsl_oro": (
        b"rw_mpas_gsl_oro --grid MESH.nc --geog WPS_GEOG --out ORO.nc"),
    "rw_mpas_lbc": (
        b"rw_mpas_lbc --grid INIT.nc --out-dir DIR "
        b"--start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS "
        b"--interval YYYY-MM-DD_HH:MM:SS=MET_PATH [--interval ...] "
        b"--nfglevels N --fg-interval-seconds N --extrap-airtemp MODE "
        b"--use-spechumd yes|no --theta-adv-order N --coef-3rd-order X "
        b"--oned-underflow preserve|reproduce-ifx-ftz "
        b"[--physics-mode correct|exact]"),
    # The history converter.  Its marker is the OUTPUT SCHEMA name and
    # the progress tokens, like the mapped engine's, because that is the
    # literal which changes exactly when the tape contract changes, and the
    # flags a caller passes that an older build refuses (--remap with its
    # REMAP/BUDGET lines, --jobs): without them a pre-2.9.0 converter
    # matched and the woof hex product stage died on "unknown flag --remap".
    "rw_mpas_convert": (
        b"gpuwm-rw-mpas-convert-v1\tCONVERTED\tWINDOW\tWEIGHTS\tFINISHED"
        b"\tREMAP\tBUDGET\t--remap nearest|product\t--jobs N"),
    # The ML dataset exporter behind `woof ml-export`.  The marker is the
    # request schema, the modes and the progress grammar, the literal that
    # changes when the contract does: a build predating it would read a
    # request it does not speak.  Spelled to match
    # woof.ml_export.ABI_MARKER and rw_mlexport::ABI; a test binds them.
    "rw_scoreboard": (b"rw_scoreboard --request JSON schema=woof.scoreboard.request.v1 "
                      b"method=scoreboard-v1 actions=extract,observe,score,plan,merge,interpolate,accumulate,"
                      b"interpolate_to_reports columns=WFSBCOL1"),
    "rw_mlexport": (
        b"rw_mlexport --request REQUEST.json schema=ml-export.request/v1 "
        b"modes=run,append,finalize progress=jsonl earth2=per-channel-v1"),
    # The clean-room GRIB2 exporter (2.8.7) behind `export-grib2`, `go
    # --grib2`, run_options.grib2 and `render --grib2-out`.  The marker
    # names the request schema, the WOOF catalog, the post device choice
    # and the restored options (renderer definitions, extrema windows,
    # composite level): a pre-2.8.7 build (the held-out UPP-derived
    # exporter, catalog upp-post-products) or a 2.8.7 build without the
    # options does not carry it and is refused.  Spelled to match
    # woof.grib2_export.ABI_MARKER and rw_grib2export::ABI; tests bind them.
    "rw_grib2export": (
        b"rw_grib2export --request REQUEST.json schema=grib2-export.request/v1 "
        b"modes=run,append,finalize progress=jsonl default_definitions=woof "
        b"grid_geometry=wrf-native-locations/v1 surface_catalog=woof-post/v1 "
        b"post_device=auto,gpu,cpu definitions=woof,renderer extrema_window=seconds "
        b"composite_level=200,10 gust=auto,similarity,tke"),
    # The projected MET intermediate cdylib (tools/rw_wps/crates/
    # projected-intermediate) behind `woof hex intermediate` on HRRR and
    # every other projected source.  A library, so the literal is an
    # exported symbol name: the decoder-crop entry is the newest of the
    # four woof hex resolves, so a build that predates it is refused here
    # rather than by woof hex after the fetch.
    "rw_projected_intermediate": b"gpuwm_projected_window_v1",
}

#: True when the shell a remedy will be pasted into is Windows
#: PowerShell rather than a POSIX shell.  Windows PowerShell 5.1 -- the
#: in-box shell the project's own PowerShell install route reaches --
#: has no ``&&`` and no ``. "$HOME/..."``, so a remedy written for one
#: shell is a parse error in the other.
WINDOWS_SHELL = os.name == "nt"


def _shell_path(*parts: str) -> str:
    """A relative path spelled for the shell the remedy targets."""

    joined = "/".join(part.strip("/") for part in parts if part)
    return joined.replace("/", "\\") if WINDOWS_SHELL else joined


def _parent_hops(*parts: str) -> str:
    """The ``..`` walk that undoes ``_shell_path(*parts)``, exactly.

    One ``..`` per component the ``cd`` descended, counted rather than
    hardcoded: ``tools/rustwx`` is two, ``woof/tools/rustwx`` is three,
    and a remedy that guesses walks the reader somewhere else.
    """

    depth = sum(len([c for c in part.strip("/").split("/") if c])
                for part in parts if part)
    return _shell_path(*[".."] * depth)


#: The renderer/fetch-backbone crate's path inside a checkout.  Declared
#: here so all three artifacts share one shell-correctness rule.
RUSTWX_CRATE_RELATIVE = "tools/rustwx"

# These workspaces use locked registry dependencies without a vendor config.
# Source remedies must populate the cache before promising an offline build.
_CARGO_REGISTRY_WORKSPACES = frozenset({
    "tools/nextgen_forcing", "tools/nextgen_verify",
})


def cargo_build_one_liner(crate_relative: str = CRATE_RELATIVE) -> str:
    """``cd <crate>``, the offline build, and the ``cd`` back -- one line.

    The separator is the shell's, not a habit: ``&&`` is a parse error in
    Windows PowerShell 5.1, so there it is ``;``.

    It returns to the directory it started in.  ``woof doctor`` prints
    one remedy block per gap and says they run in the order printed, so
    a block that leaves the shell two levels down inside the crate is a
    block that breaks the next one's relative paths -- and on a fresh
    machine there are always several.  The tidier POSIX spelling is a
    subshell, ``(cd X && cargo build ...)``, which the README uses; it
    has no Windows PowerShell 5.1 equivalent (there is no subshell that
    contains a location change there), so adopting it would give the two
    shells different SHAPES rather than one shape with two separators.
    The explicit ``cd`` back is accurate and identical in both, and is
    already the form the README documents for PowerShell.
    """

    separator = ";" if WINDOWS_SHELL else " &&"
    fetch = (f"cargo fetch --locked{separator} "
             if crate_relative in _CARGO_REGISTRY_WORKSPACES else "")
    return (f"cd {_shell_path(crate_relative)}{separator} "
            f"{fetch}cargo build --release --locked --offline{separator} "
            f"cd {_parent_hops(crate_relative)}")


def run_if_first_succeeds(first: str, then: str) -> str:
    """``first``, then ``then`` only when ``first`` succeeded -- one line.

    For a remedy whose second command must not run after the first one
    failed, such as ``woof check X`` before ``woof run X``.  A POSIX
    shell spells that ``&&``.  Windows PowerShell 5.1 has no ``&&`` (the
    pipeline chain operators arrived in PowerShell 7) and rejects a line
    carrying it with a parser error, and the bare ``;`` the cargo build
    line uses would start ``then`` even after ``first`` refused -- a run
    the check had just said does not fit.  After a native command ``$?``
    is false when it exited non-zero, so ``first; if ($?) { then }`` is
    the same meaning in that shell.
    """

    if WINDOWS_SHELL:
        return f"{first}; if ($?) {{ {then} }}"
    return f"{first} && {then}"


def shell_line(*words) -> str:
    """``words`` as one command, quoted for the shell the remedy targets.

    A remedy that interpolates a path bare breaks on the first folder
    with a space in it: ``woof check C:\\my runs\\case.toml`` reaches
    the program as two arguments in either shell.  Each word is quoted
    only where that shell needs it, by the same rule
    :func:`woof.prep_output.shell_command` prints forecast lines with,
    chosen from :data:`WINDOWS_SHELL`.
    """

    from woof.prep_output import shell_command

    return shell_command(words,
                         shell="powershell" if WINDOWS_SHELL else "posix")


def lazy_build_hints(module: str, **hints: str):
    """A module ``__getattr__`` that spells each build hint when it is read.

    ``hints`` maps an attribute name to the crate it builds, and reading
    that attribute returns :func:`cargo_build_one_liner` of the crate
    under the shell rule in force at that moment.

    THE BREAKAGE: every module that prints a cargo build line kept it as
    a constant computed at import, so the line was spelled by whatever
    :data:`WINDOWS_SHELL` said when the module was first imported and a
    test forcing the other shell could not move it.  The PowerShell
    spelling of the renderer, fetch, observation and mesh remedies was
    then never checked on a POSIX runner, nor the POSIX one on Windows:
    a forced-shell test compared the host's frozen line against the
    other shell's rules, which is how ``&&`` reached a PowerShell check
    on the ubuntu publish runner.  Read through this, the attribute
    keeps its name for every caller and cannot go stale.
    """

    def __getattr__(name: str) -> str:
        crate = hints.get(name)
        if crate is None:
            raise AttributeError(
                f"module {module!r} has no attribute {name!r}")
        return cargo_build_one_liner(crate)

    return __getattr__


def rustwx_build_hint() -> str:
    """The one-liner that builds the renderer workspace, spelled now.

    ``tools/rustwx`` builds the renderer, the fetch backbone, the
    observation front doors, the static builder and the mesh tools, so
    every one of their remedies prints this line and reads the shell
    rule when it prints, not when its module was imported.
    """

    return cargo_build_one_liner(RUSTWX_CRATE_RELATIVE)


#: ``CARGO_BUILD_HINT`` is the one-liner that builds every bridge, run
#: from a source checkout's own root, spelled when it is read
#: (:func:`lazy_build_hints`).  ``--offline`` works because the crate
#: vendors its dependencies (``tools/rustwx/vendor/crates-io``
#: plus that workspace's ``.cargo/config.toml``): no network, no
#: registry.
__getattr__ = lazy_build_hints(__name__, CARGO_BUILD_HINT=CRATE_RELATIVE)

#: Where a pip user gets the sources the wheel does not carry.  Same URL
#: and same clone directory as README's install section, so the two
#: cannot drift into telling different stories.
REPOSITORY_URL = "https://github.com/recastsystems/woof"
CLONE_DIR = "recast-woof"


def cargo_executable() -> str | None:
    """The cargo to run here, or None when this machine has none.

    THE CONCRETE BREAKAGE, measured on a rented Linux node 2026-09-17.
    rustup was installed and working; ``cargo --version`` answered 1.93.1
    in a login shell.  In a NON-LOGIN shell -- ``ssh host 'pytest ...'``,
    a cron entry, a systemd unit, a desktop-launched process -- rustup's
    profile edit has not run, ``cargo`` is not on PATH, and every bridge
    build refused with "no Rust toolchain is on PATH ... install Rust
    first", telling the owner to install what was already installed two
    directories away.  Ten tests errored in their fixture on exactly
    that, and nothing about the machine was wrong.

    ``cargo_activation_command`` above records the same fact from the
    other side ("rustup edits the login profile, which does nothing for
    the shell already running"), so this is that knowledge applied at the
    place that runs the command instead of only in the remedy text.

    The ladder, in order, and it runs nothing:

    1. ``CARGO``, the toolchain's own environment variable, so an
       explicit choice always wins -- the same override
       ``tools/battery/run_cargo_gates.py --cargo`` offers its lane;
    2. PATH, which is what a developer shell has;
    3. rustup's own home, ``$CARGO_HOME/bin`` or ``~/.cargo/bin``, which
       is where every rustup install puts the shim.

    Resolving to the SHIM rather than to a pinned toolchain is
    deliberate: the shim is what reads a ``rust-toolchain.toml``, so a
    crate in this tree can declare the toolchain it needs and be obeyed
    by every route without a version being hard-coded here.  No such
    file exists today and none is added by this function: measured, the
    bridge crates build on 1.93.1, and pinning a newer toolchain would
    refuse a toolchain that works.
    """

    explicit = os.environ.get("CARGO")
    if explicit:
        found = shutil.which(explicit) or (
            explicit if os.path.isfile(explicit) else None)
        if found:
            return found
    found = shutil.which("cargo")
    if found:
        return found
    name = "cargo.exe" if os.name == "nt" else "cargo"
    home = os.environ.get("CARGO_HOME")
    roots = [Path(home)] if home else []
    roots.append(Path.home() / ".cargo")
    for root in roots:
        candidate = root / "bin" / name
        if candidate.is_file():
            return str(candidate)
    return None


def cargo_is_installed() -> bool:
    """Is a Rust toolchain reachable here?  Read-only, runs nothing.

    Reachable, not "on PATH": see :func:`cargo_executable` for the
    measured reason those are not the same question.
    """

    return cargo_executable() is not None


def rust_toolchain_install_command() -> str:
    """The command that installs Rust here.  A command, and nothing else.

    No label, no parenthetical.  ``install Rust: winget ... (or
    https://rustup.rs)`` reads fine to someone who already knows what a
    shell is and is a syntax error to everyone the remedy exists for;
    anything that is not the command belongs on its own ``#`` line.
    """

    if WINDOWS_SHELL:
        return "winget install --id Rustlang.Rustup -e --source winget"
    return ("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y")


def cargo_activation_command() -> str:
    """Put a freshly installed cargo on PATH in the CURRENT shell.

    rustup edits the login profile, which does nothing for the shell
    already running -- so a bootstrap that installs Rust and then calls
    ``cargo`` fails on the very machine the bootstrap is for, with
    ``cargo: command not found`` two lines after installing cargo.
    """

    if WINDOWS_SHELL:
        return '$env:Path = "$env:USERPROFILE\\.cargo\\bin;$env:Path"'
    return '. "$HOME/.cargo/env"'


def build_from_clone_hint(
        crate_relative: str = CRATE_RELATIVE) -> tuple[str, ...]:
    """The whole bootstrap, for an install with no Rust sources in it.

    The pip wheel ships no crates, so on a pip-only machine NOTHING can
    decode GRIB until these are built -- and the short
    :data:`CARGO_BUILD_HINT` names ``tools/grib1_bridge``, a directory
    that does not exist there.  A remedy pointing at a missing directory
    is worse than no remedy: it reads as a broken install rather than a
    missing step, and the v1.0.0 field report says exactly that.

    Every line below is either a command that runs as printed, in order,
    from any working directory -- in the shell this platform actually
    has -- or a ``#`` comment, which is inert if pasted with the rest.
    Nothing is a mixture of the two.  The measured cost of the whole
    chain is roughly two minutes on a warm machine, dominated by the
    compile.

    Two properties the *sequence* needs, which no single line shows.
    ``git clone`` fails when the directory is already there, and a
    pip-only machine gaps every bridge at once, so doctor prints this
    same clone six or seven times in one report: the note above it says
    to skip the line rather than leaving the reader to read an error as
    a failure.  And the last line walks back out of the crate, because
    the next block starts where this one ends -- ``cd`` with no return
    is how a paste that "runs in the order printed" stops doing so
    after the first block.
    """

    lines: list[str] = []
    if not cargo_is_installed():
        lines += [
            "  # Rust is not on PATH.  These two lines install it and make",
            "  # cargo usable in THIS shell (rustup only edits the profile,",
            "  # which the already-running shell never re-reads).",
            f"  {rust_toolchain_install_command()}",
            f"  {cargo_activation_command()}",
        ]
    lines += [
        f"  # skip the clone if {CLONE_DIR}/ already exists -- doctor prints",
        "  # one block per gap and they all start from the same clone, so",
        "  # this line repeats; a second clone into it just errors.",
        f"  git clone {REPOSITORY_URL} {CLONE_DIR}",
        f"  cd {_shell_path(CLONE_DIR, crate_relative)}",
        *(["  cargo fetch --locked"]
          if crate_relative in _CARGO_REGISTRY_WORKSPACES else []),
        "  cargo build --release --locked --offline",
        "  # back out to where you started, so the next block's relative",
        "  # paths still mean what they say.",
        f"  cd {_parent_hops(CLONE_DIR, crate_relative)}",
    ]
    return tuple(lines)


def prebuilt_bundle_offer(artifact: str | None = None
                          ) -> tuple[str, ...] | None:
    """The ``woof fetch-bridges`` lead-in, or None when it would lie.

    Returned only when this install can actually do it: a platform key
    for this OS/architecture, and a bundle pinned for that platform in
    the pins document *this wheel carries*.  Both are properties of
    artifacts on disk, so the offer is never a promise about a release
    that has not happened -- a tree whose pins name no platform gets the
    build-from-source remedy it has always got, unchanged.

    ``artifact``, when given, adds the third question, and it is the one
    whose absence cost a whole wave.  ``woof.obs.frontdoor`` refused
    with this block at its head -- "the MRMS front door (rw_mrms) is not
    built or not found.  woof fetch-bridges ..." -- because *a* bundle
    existed for the platform.  It never asked whether ``rw_mrms`` was
    IN that bundle, and it was not; running the offered command printed
    "all artifacts already staged and pin-valid" and the refusal
    repeated verbatim.  A remedy that cannot supply what the refusal is
    about is worse than no remedy: it reads as a broken machine rather
    than a missing feature, and it costs the reader the download before
    it tells them nothing changed.  Named here, once, so no caller can
    reintroduce it: an artifact the pinned bundle does not carry gets
    the build-from-source route, which is true.

    Every line is a command or a ``#`` comment, because doctor prints
    these verbatim and claims exactly that of them.
    """

    try:
        from woof import bridge_assets

        pins = bridge_assets.load_pins()
        platform = bridge_assets.host_platform()
        bundle = pins.bundle_for(platform)
    except Exception:  # a broken/absent pins document is not an offer
        return None
    if bundle is None:
        return None
    if artifact is not None and not any(
            pin.artifact == artifact for pin in bundle.binaries):
        return None
    mib = bundle.bytes / (1024 * 1024)
    return (
        "  woof fetch-bridges",
        f"  # one {mib:.0f} MiB download: the prebuilt {bundle.platform} "
        "bundle,",
        "  # every artifact verified against the SHA-256 pins packaged "
        "with",
        f"  # this release before it is staged into {default_bridge_dir()}.",
        "  # --from DIR stages the same bundle from a local directory, "
        "offline.",
    )


def _offer_for(artifact: str | None) -> tuple[str, ...] | None:
    """:func:`prebuilt_bundle_offer`, called the way it has always been.

    The artifact-aware parameter is an ADDITION, so a caller with
    nothing to say about a specific artifact must still reach the
    function through its original zero-argument shape.  ``woof doctor``
    substitutes a stand-in for this function in its own tests, and a
    stand-in written against the old signature is not a stale test --
    it is every out-of-tree caller that ever wrapped it.  Passing
    ``None`` positionally would break all of them for no gain.
    """

    if artifact is None:
        return prebuilt_bundle_offer()
    return prebuilt_bundle_offer(artifact)


def _as_comments(block: str) -> str:
    """Comment out a build block so a paste does not also run it.

    When the prebuilt bundle is offered first, the source build is the
    alternative rather than the next step: a reader who selects the whole
    report must not clone 2.5 GB and compile for two minutes to obtain
    files the line above already staged.  The block's own ``#`` notes
    stay as they are; only its commands are demoted, indented one level
    so it is visible which lines are the ones to uncomment.
    """

    lines: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(f"  {stripped}" if stripped.startswith("#")
                     else f"  #   {stripped}")
    return "\n".join(lines)


def _bundle_first(build_block: str, artifact: str | None = None) -> str:
    """``woof fetch-bridges`` first, the source build commented after it."""

    offer = _offer_for(artifact)
    if offer is None:
        return build_block
    return "\n".join(offer + (
        "  # Or build the artifacts from source instead -- the route that",
        "  # works on every platform, including ones with no published",
        "  # bundle.  Uncomment these to take it:",
        _as_comments(build_block),
    ))


def sources_present(crate_relative: str = CRATE_RELATIVE) -> bool:
    """Does this install carry the Rust sources for ``crate_relative``?

    False on a pip install, where every ``cd tools/...`` instruction
    names a directory that does not exist.  It answers about the crate
    the caller is actually going to name: this used to answer for
    ``tools/grib1_bridge`` whatever it was asked, so a tree carrying one
    crate and not the other got a ``cd`` into the missing one.

    The crate's own ``Cargo.toml`` is the evidence, not the folder: a wheel
    ships package data under some crate folders (tools/hydro_world/data and
    tools/hydro_world_params/data, read by the default-on hydro door), so on a
    pip install those folders exist with no sources in them.  Breakage it
    prevents (2.9.4 R11 from the packet's Linux wheel, 2026-10-22): the doctor
    took the data folder for a checkout, withheld the on-demand row of
    rw_hydro_world and rw_hydro_params and told a pip user to run
    ``cd site-packages/tools/hydro_world && cargo build``.
    """

    return (_package_parent() / Path(crate_relative) / "Cargo.toml").is_file()


def install_aware_build_hint(one_liner: str,
                             crate_relative: str = CRATE_RELATIVE,
                             artifact: str | None = None) -> str:
    """A cargo one-liner, or the whole bootstrap when there is no crate.

    One call site, two true answers.  Handing a pip user
    ``cd tools/rustwx && cargo build`` sends them to a directory that
    does not exist, which reads as a broken install rather than a
    missing step.

    Neither answer carries a leading newline.  The bootstrap used to,
    which meant every caller that wrote its own headline --
    ``"# rebuild it:\\n" + hint`` -- printed a blank line and then a
    command at column 0, out from under the label it belonged to.  A
    caller that wants the hint on its own line adds the newline it is
    asking for.
    """

    if sources_present(crate_relative):
        return one_liner
    return _bundle_first("\n".join(build_from_clone_hint(crate_relative)),
                         artifact)


def install_aware_one_line_hint(one_liner: str,
                                crate_relative: str = CRATE_RELATIVE,
                                artifact: str | None = None) -> str:
    """The same two truths, for a caller that may emit only ONE line.

    ``install_aware_build_hint``'s pip answer is the whole bootstrap --
    Rust install, PATH activation, clone, build -- and a caller bound to
    a single physical line cannot print it.  Inlining the checkout
    one-liner anyway is the failure this exists to avoid: it names a
    directory a wheel install does not have.

    So the accurate one-line composition is a pointer.  ``woof doctor``
    already assembles the full bootstrap for this exact machine, and
    naming it is true on every install, where ``cd tools/rustwx`` is
    true on only some.
    """

    if sources_present(crate_relative):
        return one_liner
    if _offer_for(artifact) is not None:
        return ("run `woof fetch-bridges`, which stages this platform's "
                "prebuilt artifacts under the SHA-256 pins packaged with "
                "this release (`woof doctor` prints the source-build "
                "route as well)")
    return ("run `woof doctor`, which prints the build steps for this "
            "install (a wheel carries no Rust sources, so this one needs "
            "a clone)")


#: The two native executable formats this project runs, by magic bytes.
#: An ELF header is four bytes; a PE is ``MZ`` plus a signature at the
#: offset stored at 0x3c, so 0x40 bytes is always enough to classify.
_EXECUTABLE_HEADER_BYTES = 0x40


def native_executable_format(path: Path) -> tuple[str | None, str]:
    """``('elf'|'pe'|None, evidence)`` from the file's own header.

    A cheap read, and the ONLY safe way to ask "can this be executed"
    about a file of unknown provenance -- because on Windows, asking the
    operating system can hang forever.

    ``subprocess.run(..., timeout=...)`` bounds the *wait*, never
    ``CreateProcess`` itself.  A file with a corrupt or absent PE header
    can make the image loader raise a modal error dialog inside
    ``CreateProcess``, and in a session with no interactive desktop
    there is nothing to dismiss it: the call never returns, the timeout
    never starts, and the process is unkillable-by-timeout.  A release
    battery froze twice at exactly that call, on a probe of a file whose
    contents were sixteen bytes of ASCII.

    So every probe reads the header first and refuses non-executables
    itself.  The answer for a real bridge is unchanged -- a genuine ELF
    or PE still goes on to be launched -- and the answer for the file
    that used to hang is now a finding, in microseconds.
    """

    try:
        with Path(path).open("rb") as stream:
            head = stream.read(_EXECUTABLE_HEADER_BYTES)
    except OSError as error:
        return None, f"cannot be read: {error}"
    if head[:4] == b"\x7fELF":
        return "elf", "ELF executable header"
    if head[:2] == b"MZ" and len(head) >= 0x40:
        offset = int.from_bytes(head[0x3c:0x40], "little")
        # The signature lives past this window; its OFFSET is what a
        # truncated or fabricated MZ stub gets wrong, and a plausible
        # one is all that is needed to make launching safe to attempt.
        if 0 < offset < (1 << 24):
            return "pe", "PE executable header"
        return None, ("has an MZ stub whose PE signature offset is out of "
                      "range -- truncated or not an executable")
    return None, (f"is not a native executable ({len(head)} byte(s) read, "
                  "no ELF or PE header)")


def launchable(path: Path) -> tuple[bool, str]:
    """Is ``path`` safe and sensible to hand to the operating system?

    Format first, then the POSIX execute bit.  Both are properties of
    the file, so neither can hang, and together they cover every way a
    probe used to discover the answer the expensive way.
    """

    path = Path(path)
    if not path.is_file():
        return False, "does not exist"
    binary_format, evidence = native_executable_format(path)
    if binary_format is None:
        return False, f"exists but {evidence}"
    expected = "pe" if os.name == "nt" else "elf"
    if binary_format != expected:
        return False, (f"exists but is a {binary_format.upper()} binary on a "
                       f"host that runs {expected.upper()} -- built for "
                       "another platform")
    if os.name != "nt" and not os.access(path, os.X_OK):
        return False, "exists but is not marked executable"
    return True, evidence


class quiet_loader_errors:  # noqa: N801 - a context manager, used as a verb
    """Make the Windows image loader FAIL instead of prompting.

    ``SetErrorMode`` is per-process and inherited by children, so
    setting it around a probe covers the loader dialog that would
    otherwise appear inside ``CreateProcess``.  This is the backstop
    behind :func:`launchable`, not a replacement for it: a header check
    cannot know about a missing DLL, and a missing-DLL dialog hangs
    exactly the same way.

    A no-op everywhere but Windows.
    """

    _FAIL_FAST = 0x0001 | 0x0002 | 0x8000  # CRITICALERRORS|GPFAULT|OPENFILE

    def __enter__(self):
        self._previous = None
        if os.name != "nt":
            return self
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            self._previous = kernel32.SetErrorMode(self._FAIL_FAST)
            # SetErrorMode replaces rather than merges, so restore the
            # union: another library may have asked for its own bits.
            kernel32.SetErrorMode(self._previous | self._FAIL_FAST)
        except (AttributeError, OSError):  # pragma: no cover - not Windows
            self._previous = None
        return self

    def __exit__(self, *_exception):
        if self._previous is None:
            return False
        try:
            import ctypes

            ctypes.windll.kernel32.SetErrorMode(self._previous)
        except (AttributeError, OSError):  # pragma: no cover
            pass
        return False


def legacy_bridge_dir() -> Path:
    """The shared, unversioned ``~/.woof/bridges``.

    Every release before 2.8.6 staged its bundle flat into this one
    directory, and a source checkout (whose pins declare no release) or
    a platform with no published bundle still stages there.  It is
    also the PARENT of every versioned staging directory.

    It is shared by every woof install under this home, which is the
    defect the versioned layout closes: on 2026-10-05 a PyPI 2.8.0 venv
    on test host B found the flat ``rw_netcdf`` stamped by a local build
    (``fc5b34e26``), auto-fetched 2.8.0's bundle and overwrote all 31
    files, breaking the other install that relied on them.  A pinned
    install now only READS a file here, and only when its bytes are the
    ones that install's own release pinned; it never writes here.
    """

    return Path.home() / ".woof" / "bridges"


#: Characters allowed in a versioned staging directory name.  A release
#: name is a git tag, which may carry ``/``; anything outside this set
#: becomes ``_`` so the name is always one path component.
_TAG_SAFE = re.compile(r"[^A-Za-z0-9._+-]")


def staged_version_tag() -> str | None:
    """This install's versioned staging directory name, or None.

    ``<release>-<first 12 hex of this platform's bundle SHA-256>``: the
    release names it for a reader, the bundle digest makes it exact, so
    two installs that both call themselves the same release but carry
    different pins (a local re-cut, a private mirror) still never share
    a directory.  None when this install carries no pins for this
    platform -- a source checkout, or a platform with no bundle -- and
    then there is nothing to fetch and nothing to version: such an
    install keeps staging into :func:`legacy_bridge_dir` exactly as
    before.
    """

    try:
        from woof import bridge_assets

        pins = bridge_assets.load_pins()
        bundle = pins.bundle_for(bridge_assets.host_platform())
    except Exception:                                # noqa: BLE001
        return None
    if bundle is None or not pins.release:
        return None
    release = _TAG_SAFE.sub("_", str(pins.release)).strip(".") or "release"
    return f"{release}-{bundle.sha256[:12]}"


def default_bridge_dir() -> Path:
    """THIS install's staging directory for prebuilt bridges.

    ``~/.woof/bridges/<release>-<bundle digest>`` for an install that
    carries release pins for this platform: ``woof fetch-bridges`` and
    the automatic refresh write there and nowhere else, so two engine
    versions on one machine (two venvs, a user and a lane, an upgrade
    side by side) each own their own directory and can never replace
    each other's files.  An install with no pins keeps the flat
    :func:`legacy_bridge_dir`.
    """

    tag = staged_version_tag()
    root = legacy_bridge_dir()
    return root if tag is None else root / tag


def staging_location_note() -> str:
    """Where this engine stages and reads bridges, in one sentence.

    For ``woof doctor``: the resolved directory, not the
    ``~/.woof/bridges`` spelling, because on a machine with two engine
    versions the question a reader is asking is WHICH directory this one
    uses.
    """

    own = default_bridge_dir()
    if staged_version_tag() is None:
        return (f"this engine stages bridges in {own} (the flat layout: "
                "this install carries no release pins to version it by)")
    return (f"this engine stages bridges in {own}, its own versioned "
            f"directory; the shared flat {legacy_bridge_dir()} is read only "
            "for a file whose bytes are this release's pin, and never "
            "written")


def legacy_bridge_candidates(filename: str) -> tuple[Path, ...]:
    """The flat-layout rung that follows :func:`default_bridge_dir`.

    Empty when this install's staging directory IS the flat one (no
    pins).  Otherwise the flat copy of ``filename``, which every ladder
    lists right after the versioned one so an estate staged by an older
    release keeps working without a download -- but only while its bytes
    are this release's pin: :func:`require_release_pin` judges it, and a
    mismatch is fetched into the versioned directory instead, leaving
    the flat file exactly as it was.
    """

    # Decided by the pins, not only by comparing directories: an install
    # with no pins never versioned anything, so there is no older layout
    # for it to fall back to and its staging directory IS the flat one.
    if staged_version_tag() is None:
        return ()
    root = legacy_bridge_dir()
    try:
        if default_bridge_dir().resolve() == root.resolve():
            return ()
    except (OSError, ValueError):
        return ()
    return (root / filename,)


#: Directory INSIDE the package that a platform wheel stages its
#: prebuilt Rust artifacts into.  Named once here because six separate
#: resolution ladders consume it (this module, :mod:`woof.rustwx`,
#: :mod:`woof.rustwx_fetch`, :mod:`woof.obs.nexrad`,
#: :mod:`woof.obs.frontdoor` and :mod:`woof.obs.dealias_region`) and a
#: seventh copy of the string is how a rung goes missing on one door.
PACKAGED_BRIDGE_SUBDIR = ("libexec", "bridges")


def ensure_executable(path: Path) -> Path:
    """Give a wheel-shipped artifact back its executable bit.

    ``pip`` honours a staged member's recorded mode when the high half
    of its zip ``external_attr`` is a whole ``st_mode`` with the
    regular-file type bit included: its ``zip_item_is_executable``
    tests ``S_ISREG`` before it reads the execute bits.  ``setup.py``
    writes that mode today, so a wheel built from this tree installs its
    bridge binaries executable and this function finds nothing to
    repair.  Measured on the rebuilt manylinux wheel: all 28 staged
    artifacts read back ``0o100755``, and a ``pip install`` into a clean
    venv leaves every one of them 775 and passing
    ``os.access(..., os.X_OK)``.

    It is kept for the wheels that do not carry that mode.  A wheel
    published before that fix stamped ``0o755`` with no file-type bits,
    so pip's predicate rejected it and wrote the binaries 0644, and the
    same is true of a bundle laid down by any other route that stamps no
    mode.  That failure is measured, not theorised: a clean-venv install
    of such a wheel put all eleven artifacts in place and then died with
    ``PermissionError: [Errno 13] Permission denied`` on the first
    ``subprocess.run``.  This function is the repair for bytes that are
    already published, not a substitute for the stamping: reverting
    ``setup.py``'s constants on the strength of it existing would put
    that defect back for every door that resolves a bridge by another
    route.

    The repair is done at resolution rather than asked of the user,
    because "fixed" means a bare ``pip install recast-woof`` works: a flag or a
    documented ``chmod +x`` would be a workaround, and this defect is
    invisible until a door is already being opened.

    Only files inside :func:`packaged_bridge_dir` are touched -- a
    checkout build, a ``~/.woof/bridges`` copy or an environment
    override belongs to the user, and silently re-permissioning those
    would be changing something this package does not own.  A no-op on
    Windows, where execution does not consult a mode bit.
    """

    if os.name == "nt":
        return path
    try:
        packaged = packaged_bridge_dir()
        if not path.is_relative_to(packaged):
            return path
        mode = path.stat().st_mode
        if mode & 0o111:
            return path
        # Mirror the read bits: a file the user can read, they may run.
        path.chmod(mode | ((mode & 0o444) >> 2))
    except OSError as error:
        raise PermissionError(
            f"{path} is not executable and its mode could not be repaired "
            f"({error}).  A bridge installed from a wheel published before "
            f"its staged modes carried the regular-file type bit, "
            f"or laid down by a route that stamps no mode at all, "
            f"arrives without the execute bit, so it needs one of:\n"
            f"  chmod +x {path}\n"
            f"  # or point woof at a copy you control via its environment "
            f"variable") from None
    return path


class StaleBridgeError(RuntimeError):
    """A staged artifact is not the binary this release published.

    Distinct from ``FileNotFoundError`` on purpose.  The artifact IS
    there and it WILL launch, so a door that falls back on "not found"
    must not treat this as the same event: the remedy is to replace the
    bytes, not to build or fetch a missing file, and a silent
    degradation is exactly the outcome this refusal exists to prevent.
    """


#: True while a caller is READING the estate rather than opening a door.
#: ``woof doctor`` resolves every artifact to report on it and must see
#: what a door would see without refusing on it or fetching anything --
#: this module's contract is that resolution has no side effects, and
#: the automatic refresh below is the one exception, carved out here so
#: the reporting paths keep the original guarantee.
_INSPECTION_ONLY = False

#: Set once a process has tried the automatic refresh, with the failure
#: text if it did not work.  One attempt per run: a box with no network
#: must not re-dial for every artifact a door resolves.
_REFRESH_ATTEMPTED = False
_REFRESH_FAILURE: str | None = None

#: Artifacts already reported under the ``allow`` workaround, so a run
#: that resolves one twelve times says so once.
_STALE_ALLOWED: set[str] = set()


@contextlib.contextmanager
def inspection_only():
    """Resolve without acting on a stale staged artifact.

    For readers: ``woof doctor``, the receipt writers, anything whose
    job is to say what the estate IS.  Inside this scope a resolution
    hands back whatever the ladder found, exactly as it did before the
    pin check existed, so a report can name a mismatch instead of dying
    on it -- and so that reading the estate never triggers a download.
    """

    global _INSPECTION_ONLY
    previous = _INSPECTION_ONLY
    _INSPECTION_ONLY = True
    try:
        yield
    finally:
        _INSPECTION_ONLY = previous


def _artifact_env_var(artifact: str) -> str | None:
    """The environment variable that overrides ``artifact``, if any."""

    if artifact in BRIDGE_ENV:
        return BRIDGE_ENV[artifact]
    try:
        from woof.bridge_assets import BUNDLED_ARTIFACTS
    except Exception:                                # noqa: BLE001
        return None
    for entry in BUNDLED_ARTIFACTS:
        if entry.name == artifact:
            return entry.env_var
    return None


def _stale_refusal(status, *, refresh_note: str | None = None) -> str:
    """Why this staged file may not be used, and the way out of it.

    Names the file, what it is, what this release pinned instead, and
    the single command that replaces it.  The breakage is stated in
    terms of what the run would DO with it rather than as a hash
    mismatch, because the hash is not the thing that hurts.
    """

    from woof.bridge_assets import STALE_POLICY_ENV

    env_var = _artifact_env_var(status.pin.artifact)
    lines = [
        f"the staged {status.pin.artifact} is not the binary "
        f"{status.release} published: {status.describe()}.",
        "This woof's Python would drive another release's binary: "
        "every behaviour that moved between the two -- an argument the "
        "door now passes, a default that changed, a capability a "
        "predicate tests for -- reverts or diverges with nothing in the "
        "run saying so, which is how ten gates come back skipped "
        "instead of failed.",
    ]
    if refresh_note:
        lines.append(refresh_note)
    lines.append("remedy:")
    lines.append("  woof fetch-bridges")
    lines.append(f"  # stages {status.release}'s bundle into its own "
                 f"{default_bridge_dir()}, verifying every artifact's")
    lines.append("  # size and SHA-256 against the pins packaged in this "
                 "install")
    if env_var:
        lines.append(f"  # or name a copy you control: {env_var}=<path> "
                     "(an override is never judged)")
    lines.append(f"  # or {STALE_POLICY_ENV}=allow to run the staged file "
                 "regardless -- a WORKAROUND, and the divergence above "
                 "stays")
    return "\n".join(lines)


def _refresh_staged_estate(status) -> str | None:
    """Re-fetch this release's bundle once; None on success.

    The offline arm is the return value: whatever went wrong, phrased
    as the sentence the refusal carries, so a box with no route to the
    release assets is told that is what happened rather than being left
    to read a traceback.
    """

    global _REFRESH_ATTEMPTED, _REFRESH_FAILURE

    if _REFRESH_ATTEMPTED:
        return _REFRESH_FAILURE
    _REFRESH_ATTEMPTED = True
    from woof import bridge_assets
    from woof.explain import warn

    warn(f"the staged {status.pin.artifact} is not {status.release}'s "
         f"binary ({status.provenance()}); fetching {status.release}'s "
         f"bridge bundle into {default_bridge_dir()} before this run "
         f"continues (the file at {status.path} is left as it is)")
    try:
        bridge_assets.refresh_staged_bundle(
            progress=lambda message: print(message, file=sys.stderr))
    except bridge_assets.BridgeAssetError as error:
        _REFRESH_FAILURE = (
            f"the automatic refresh could not complete ({error}), so the "
            "staged file is still the one described above.")
        return _REFRESH_FAILURE
    except OSError as error:
        _REFRESH_FAILURE = (
            f"the automatic refresh could not complete ({type(error).__name__}"
            f": {error}), so the staged file is still the one described "
            "above.")
        return _REFRESH_FAILURE
    _REFRESH_FAILURE = None
    return None


def require_release_pin(path: Path) -> Path:
    """``path`` is bytes this release published, or it is not used.

    Asked of the staged rungs only: :func:`default_bridge_dir` and the
    shared :func:`legacy_bridge_dir` beneath it.  Those are the
    directories ``woof fetch-bridges`` writes (today, or under an older
    release) and the only ones a wheel upgrade leaves behind -- ``pip install -U recast-woof`` replaces the
    Python half and never looks at it -- so it is where new Python ends
    up driving an older release's binaries.  Everything above it on the
    ladder is exempt by construction and stays exempt: an environment
    override is an explicit declaration, a checkout's ``target/release``
    is a build the developer just made (and can never match a release
    pin), and both ``libexec`` rungs arrived with this version.

    ``woof doctor`` already reported this class as a BROKEN line, and
    no door consulted it, which is the whole defect: the report said the
    estate was wrong while the resolution ladder kept handing the same
    bytes to the routes.  The two now read one judgement
    (:func:`woof.bridge_assets.staged_pin_status`) and differ only in
    what they do with it -- doctor reports, a door acts -- so they
    cannot disagree again.

    Default is to fix it: fetch this release's bundle into this
    release's own versioned directory, verified by size and SHA-256
    exactly as the command does, and carry on with the new bytes --
    returning that path, which is not ``path`` when the stale file sat
    in the shared flat layout.  The stale file itself is never
    replaced: it may be exactly what another install on this machine
    runs.  Offline, that becomes the refusal.
    """

    if _INSPECTION_ONLY:
        return path
    if not _staged_rung(path):
        return path
    from woof import bridge_assets

    status = bridge_assets.staged_pin_status(path)
    if status is None or status.matches:
        return path
    policy = bridge_assets.stale_policy()
    if policy == "allow":
        if status.pin.artifact not in _STALE_ALLOWED:
            _STALE_ALLOWED.add(status.pin.artifact)
            from woof.explain import warn

            warn(f"running the staged {status.pin.artifact} anyway "
                 f"({bridge_assets.STALE_POLICY_ENV}=allow): "
                 f"{status.describe()}")
        return path
    if policy == "refresh":
        # The refresh writes THIS install's directory and nothing else.
        # A stale file found anywhere else under the shared root -- the
        # flat legacy layout, or (never listed by a ladder, but judged
        # the same if handed here) another release's directory -- is
        # left byte-for-byte as it was, and the door gets this release's
        # copy of the same filename instead.
        own = default_bridge_dir() / path.name
        failure = _refresh_staged_estate(status)
        if failure is None:
            after = bridge_assets.staged_pin_status(own)
            if after is not None and after.matches:
                return own
            raise StaleBridgeError(_stale_refusal(
                after or status, refresh_note=(
                    "the automatic refresh ran and this artifact still "
                    "does not match its pin.")))
        raise StaleBridgeError(_stale_refusal(status, refresh_note=failure))
    raise StaleBridgeError(_stale_refusal(status))


def _staged_rung(path: Path) -> bool:
    """Is ``path`` in a directory a fetch stages into (own or shared)?

    This install's own :func:`default_bridge_dir`, or anywhere under the
    shared :func:`legacy_bridge_dir` root.  Never raises.
    """

    try:
        resolved = path.resolve()
        return any(resolved.is_relative_to(directory.resolve())
                   for directory in (default_bridge_dir(),
                                     legacy_bridge_dir()))
    except (OSError, ValueError):
        return False


def accept_resolved(path: Path, *, executable: bool = True) -> Path:
    """The last step of every resolution ladder: this file, or a refusal.

    One place, because fourteen ladders in this package end here and a
    fifteenth copy of the accept step is how one door keeps resolving
    what the other fourteen refuse.  ``executable=False`` is for the
    libraries, which are loaded rather than launched and whose modes pip
    does not break.

    Two judgements, one for each rung that can hand over bytes this
    release did not make: the staged bundle must be this release's pin,
    and a checkout's own build must be current with the checkout.
    """

    if executable:
        path = ensure_executable(path)
    return require_current_checkout_build(require_release_pin(path))


# ---------------------------------------------------------------------------
# Rung 2: a checkout's own build, against the tree it serves
# ---------------------------------------------------------------------------

#: Suffixes of the files a cargo build compiles or resolves against.  A
#: workspace holds far more than this (vendored crates, map assets, the
#: target directory); a vendored crate only changes when ``Cargo.lock``
#: does, and the lock IS scanned, so the walk stays at a few hundred
#: files instead of tens of thousands.
_BUILD_INPUT_SUFFIXES = (".rs", ".toml", ".lock", ".c", ".h", ".cc",
                         ".cpp", ".cu", ".cuh")

#: Directories a build never reads its sources from.
_BUILD_INPUT_SKIP = frozenset({"target", ".git", "vendor", "assets",
                               "patches", "tests", "benches", "examples",
                               "fixtures"})

#: Built name -> crate directory, per workspace.  Manifests are static
#: configuration and parsing them all is the only part of this that is
#: not a handful of `stat` calls, so it is read once per process.  The
#: file times themselves are never cached: a resolution must answer
#: about the tree as it is at that moment, not as it was at import.
_CRATE_INDEX: dict[Path, dict[str, Path]] = {}


class StaleCheckoutBuildError(RuntimeError):
    """A checkout's own build is older than the sources that build it."""


def checkout_workspace_of(path: Path) -> Path | None:
    """The checkout cargo workspace whose own build ``path`` is, or None.

    The question is about the FILE, not about the rung that named it.
    ``<workspace>/target/{release,debug}/<file>`` inside this
    installation is a binary somebody built here from sources that are
    still here, and that stays true when an environment override is
    what named it: an override pointing into this checkout's own target
    IS this checkout's build, and a stale one there fails inside the
    same run as the rung that would have found it anyway.  An override
    naming a built copy anywhere else -- the remedies that say "point
    GPUWM_*_BIN at a built copy" mean exactly that -- answers None on
    the workspace test and is handed over untouched, as does everything
    that arrived built: a ``libexec`` directory, the wheel's own copy, a
    fetched bundle.  A wheel install has no workspace at all and answers
    None on the first test, which is why this cannot narrow one.
    """

    try:
        resolved = Path(path).resolve()
        parents = resolved.parents
        if (len(parents) < 3 or parents[0].name not in ("release", "debug")
                or parents[1].name != "target"):
            return None
        workspace = parents[2]
        if not (workspace / "Cargo.toml").is_file():
            return None
        if not workspace.is_relative_to(_package_parent().resolve()):
            return None
    except (OSError, ValueError):                    # pragma: no cover - rare
        return None
    return workspace


def _manifest(path: Path) -> dict:
    import tomllib

    try:
        with open(path, "rb") as stream:
            return tomllib.load(stream)
    except (OSError, ValueError):
        return {}


def _crate_index(workspace: Path) -> dict[str, Path]:
    """Built name -> the crate directory that declares it.

    Read from the manifests rather than guessed from the filename:
    ``rw_netcdf`` is declared by ``crates/rw-netcdf``, ``rw_mpas_mesh``
    by ``crates/rw-mpas``, and a library's file name resembles neither.
    """

    cached = _CRATE_INDEX.get(workspace)
    if cached is not None:
        return cached
    index: dict[str, Path] = {}
    roots = [workspace] + sorted(
        item for item in (workspace / "crates").glob("*") if item.is_dir())
    for crate in roots:
        manifest = _manifest(crate / "Cargo.toml")
        package = manifest.get("package", {}).get("name")
        if isinstance(package, str):
            index.setdefault(package.replace("-", "_"), crate)
        for section in ("bin", "lib"):
            entries = manifest.get(section, [])
            if isinstance(entries, dict):
                entries = [entries]
            for entry in entries:
                name = entry.get("name") if isinstance(entry, dict) else None
                if isinstance(name, str):
                    index[name] = crate
    _CRATE_INDEX[workspace] = index
    return index


def _path_dependency_roots(crate: Path, workspace: Path) -> list[Path]:
    """``crate`` and every crate it depends on by path, transitively.

    A binary is not only its own crate: ``rw_netcdf`` compiles the
    vendored ``netcrust`` reader in with it, and a change there is a
    change to the binary.  Path dependencies are followed, including the
    ``workspace = true`` spelling, which resolves in the workspace root's
    own ``[workspace.dependencies]``.

    Dev dependencies are NOT followed.  They build the test binaries and
    never the shipped one: rw-netcdf's own suite writes its classic
    fixtures with ``netcdf-writer``, and a change there would otherwise
    condemn a reader that does not contain a byte of it.
    """

    shared = _manifest(workspace / "Cargo.toml").get(
        "workspace", {}).get("dependencies", {})
    roots: list[Path] = []
    pending = [crate]
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        try:
            key = current.resolve()
        except OSError:                              # pragma: no cover - rare
            continue
        if key in seen or not (current / "Cargo.toml").is_file():
            continue
        seen.add(key)
        roots.append(current)
        manifest = _manifest(current / "Cargo.toml")
        sections = [manifest.get(name, {}) for name in (
            "dependencies", "build-dependencies")]
        for platform in manifest.get("target", {}).values():
            sections += [platform.get(name, {}) for name in (
                "dependencies", "build-dependencies")]
        for section in sections:
            if not isinstance(section, dict):
                continue
            for name, spec in section.items():
                if not isinstance(spec, dict):
                    continue
                if spec.get("workspace") is True:
                    spec = shared.get(name, {})
                relative = spec.get("path") if isinstance(spec, dict) else None
                if isinstance(relative, str):
                    # Resolved, so a nested path dependency is named by
                    # where it is rather than by the walk that found it.
                    pending.append(Path(os.path.normpath(current / relative)))
    return roots


def _newest_build_input(roots: tuple[Path, ...]) -> tuple[Path | None, float]:
    """The most recently written build input under ``roots``, and when."""

    newest: Path | None = None
    newest_at = 0.0
    pending = [Path(root) for root in roots]
    while pending:
        directory = pending.pop()
        if directory.is_file():
            # A named file rather than a tree: the workspace manifest and
            # its lock, which price every crate under them.
            try:
                written = directory.stat().st_mtime
            except OSError:                          # pragma: no cover - rare
                continue
            if written > newest_at:
                newest, newest_at = directory, written
            continue
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _BUILD_INPUT_SKIP:
                        pending.append(Path(entry.path))
                    continue
                if not entry.name.endswith(_BUILD_INPUT_SUFFIXES):
                    continue
                written = entry.stat().st_mtime
            except OSError:                          # pragma: no cover - rare
                continue
            if written > newest_at:
                newest, newest_at = Path(entry.path), written
    return newest, newest_at


def _dep_info_inputs(binary: Path) -> tuple[Path, ...] | None:
    """The checkout files cargo recorded as ``binary``'s inputs, or None.

    Cargo writes ``<artifact>.d`` beside every artifact it links, naming
    each source file that artifact was compiled from.  A crate can hold
    a source only one of its artifacts reads (a module one binary
    includes by ``#[path]``); when that file moves, cargo relinks that
    binary alone, so judged against the whole crate every other artifact
    stayed condemned however often the remedy's build was run.  Files
    outside this checkout (the registry, the toolchain) and anything
    under a ``target`` directory are not sources this tree moves.  None
    when there is no record, when it names no file here, or when a file
    it names is gone: the crate walk then decides, as it always did.
    """

    record = binary.with_name(binary.stem + ".d")
    try:
        text = record.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    _targets, separator, body = text.partition(": ")
    if not separator:
        return None
    # One rule; a long one may be continued with a backslash-newline, and
    # a space inside a path is written as a backslash-space.
    body = body.split("\n\n", 1)[0].replace("\\\n", " ")
    root = _package_parent().resolve()
    inputs: list[Path] = []
    for token in re.split(r"(?<!\\)\s+", body.strip()):
        if not token:
            continue
        try:
            path = Path(token.replace("\\ ", " ")).resolve()
            relative = path.relative_to(root)
        except (OSError, ValueError):
            continue
        if "target" in relative.parts:
            continue
        if ".git" in relative.parts:
            # The build scripts' rerun-if-changed on HEAD, packed-refs and
            # the branch ref (tools/grib1_bridge/build.rs and the rustwx
            # crates': they keep the revision stamp fresh in an incremental
            # target).  Cargo writes those into this record, and they move
            # on EVERY commit: a Python-only commit made the CPU library
            # "older than its sources" and `woof go` refused until 15
            # workspaces were rebuilt (V60, 2026-10-21).  Git metadata is
            # not a source the bytes are compiled from.
            continue
        if not path.is_file():
            return None
        inputs.append(path)
    return tuple(inputs) or None


#: How long the content comparison may take before the file-time verdict
#: stands (a git that hangs must not hang a run).
_CONTENT_CHECK_TIMEOUT_S = 30.0


def _inputs_match_stamped_revision(binary: Path, roots: tuple[Path, ...]) -> bool:
    """True when every input under ``roots`` has the content it had at the
    revision stamped into ``binary``.

    The build scripts stamp ``GPUWM_BRIDGE_SOURCE_REV`` only from a tree
    whose workspace has no tracked modifications, so the stamp names the
    exact source content the bytes were compiled from; ``git diff --quiet
    <stamp> -- <inputs>`` compares that content (git's content hashes)
    with the files as they are now.  File times alone condemned a build
    whose sources were byte-identical: a checkout, a revert or a rebase
    rewrites a file's time without changing a byte of it.  False whenever
    it cannot answer (no single stamp, no git, the revision not in this
    repository, a timeout): the file-time verdict then stands.
    """

    import subprocess

    try:
        from woof import bridge_assets

        revisions = set(bridge_assets.embedded_source_revisions(binary.read_bytes()))
    except Exception:                                # noqa: BLE001
        return False
    if len(revisions) != 1:
        return False
    revision = revisions.pop()
    root = _package_parent().resolve()
    paths = []
    for item in roots:
        try:
            relative = Path(item).resolve().relative_to(root)
        except (OSError, ValueError):
            return False
        paths.append(relative.as_posix())
    if not paths:
        return False
    try:
        # git diff does not see untracked files, so an input git does not
        # track leaves the file-time verdict standing.
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard",
             "--", *paths],
            capture_output=True, timeout=_CONTENT_CHECK_TIMEOUT_S, check=False)
        if untracked.returncode != 0 or untracked.stdout.strip():
            return False
        answer = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", revision, "--", *paths],
            capture_output=True, timeout=_CONTENT_CHECK_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return answer.returncode == 0


class CheckoutBuildStatus:
    """One checkout-built artifact, measured against its own sources."""

    __slots__ = ("artifact", "binary", "workspace", "built_at",
                 "newest_source", "newest_at", "roots", "_content_current")

    def __init__(self, artifact: str, binary: Path, workspace: Path,
                 built_at: float, newest_source: Path | None,
                 newest_at: float, roots: tuple[Path, ...] = ()):
        self.artifact = artifact
        self.binary = binary
        self.workspace = workspace
        self.built_at = built_at
        self.newest_source = newest_source
        self.newest_at = newest_at
        self.roots = tuple(roots)
        self._content_current: bool | None = None

    @property
    def current(self) -> bool:
        """Is every input older than the output, or, where a file time
        says otherwise, unchanged in content since the stamped revision?"""

        if self.newest_source is None or self.newest_at <= self.built_at:
            return True
        if self._content_current is None:
            self._content_current = _inputs_match_stamped_revision(
                self.binary, self.roots)
        return self._content_current

    def built_from(self) -> str:
        """The revision stamped into the bytes, in words.

        Every gpuwm-authored bridge embeds ``GPUWM_BRIDGE_SOURCE_REV``
        (``tools/rustwx/crates/*/build.rs``), and a build from a tree
        with modifications in it deliberately stamps ``unknown`` rather
        than a commit it is not.  Read for the refusal only: the
        staleness itself is decided by the files, so a binary with no
        stamp at all is still measured.
        """

        try:
            from woof import bridge_assets

            revisions = bridge_assets.embedded_source_revisions(
                self.binary.read_bytes())
        except Exception:                            # noqa: BLE001
            revisions = ()
        if len(revisions) == 1:
            return f"from source revision {revisions[0]}"
        if revisions:
            return "from more than one source revision " + ", ".join(revisions)
        return "from a tree with local modifications, which stamps no revision"

    def describe(self) -> str:
        """One line for a report: what is old, and by how much."""

        relative = self.newest_source
        try:
            relative = self.newest_source.resolve().relative_to(
                _package_parent().resolve())
        except (AttributeError, OSError, ValueError):  # pragma: no cover
            pass
        return (f"{self.binary} was built {_when(self.built_at)} "
                f"({self.built_from()}) and {relative} was written "
                f"{_when(self.newest_at)}, so this build is "
                f"{_elapsed(self.newest_at - self.built_at)} behind the "
                "sources it is built from")

    def remedy(self) -> str:
        """The command that ends it, in this platform's shell."""

        try:
            crate = self.workspace.relative_to(_package_parent()).as_posix()
        except ValueError:                           # pragma: no cover - rare
            crate = str(self.workspace)
        return cargo_build_one_liner(crate)

    def refusal(self) -> str:
        return (
            f"the Rust bridge `{self.artifact}` in this checkout is older "
            "than the sources it is built from.\n"
            f"  what: {self.describe()}.\n"
            "  why: a pull moves this repository's Rust half and leaves the "
            "binary where it was, so the Python half of the new release "
            "drives the previous one's reader and fails inside a run "
            "instead of before it.\n"
            f"  remedy: {self.remedy()}")


def _when(stamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(stamp, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%SZ")


def _elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def checkout_build_status(path: Path) -> CheckoutBuildStatus | None:
    """How ``path`` stands against the checkout that builds it, or None.

    None for everything that is not rung 2, which is every wheel install
    and every staged or fetched artifact: those have no sources here to
    be compared with, and this function is the whole of what a report
    and the resolver both read, so the two cannot disagree.
    """

    workspace = checkout_workspace_of(path)
    if workspace is None:
        return None
    binary = Path(path).resolve()
    try:
        built_at = binary.stat().st_mtime
    except OSError:                                  # pragma: no cover - rare
        return None
    index = _crate_index(workspace)
    # A shared library arrives as `libgpuwm_preprocess_cpu.so`, and the
    # crate declaring that name is `tools/grib1_bridge`: it is the
    # `[lib] name`, beside the `grib1_bridge` package name.  So the
    # platform prefix is stripped and the index asked a second time.
    #
    # The name that ANSWERED is kept, because the refusal below prints it
    # and every other surface spells the artifact without the platform's
    # `lib`: the CPU library's staleness refusal read "the Rust bridge
    # `libgpuwm_preprocess_cpu`", which is not a name a reader can look
    # up or a remedy can be matched against.  Stripping only when the
    # stripped spelling is the one the workspace DECLARES keeps a binary
    # whose name genuinely starts with `lib` intact.
    artifact = binary.stem
    crate = index.get(artifact)
    if crate is None:
        stripped = artifact.removeprefix("lib")
        crate = index.get(stripped)
        if crate is not None:
            artifact = stripped
    # The workspace manifest and its lock price every crate under them,
    # and a dependency bump moves the lock and nothing else.
    roots = tuple(workspace / name for name in ("Cargo.toml", "Cargo.lock"))
    recorded = _dep_info_inputs(binary)
    if recorded is not None:
        # Cargo's own record of the files this artifact was compiled
        # from, beside every crate manifest it names.
        dependencies = _path_dependency_roots(crate, workspace) if crate else ()
        roots += tuple(root / "Cargo.toml" for root in dependencies)
        roots += recorded
    else:
        roots += tuple(_path_dependency_roots(crate, workspace)) if crate else ()
        if len(roots) == 2:
            # No manifest claims this name, so nothing narrows the
            # question: measure the whole workspace rather than answering
            # "current" about a binary whose sources were not located.
            roots = (workspace,)
    newest_source, newest_at = _newest_build_input(roots)
    return CheckoutBuildStatus(artifact, binary, workspace, built_at,
                               newest_source, newest_at, roots)


def require_current_checkout_build(path: Path) -> Path:
    """``path`` is this checkout's current build of it, or it is not used.

    The gate the reported defect wanted: a user who pulled 2.7.4 onto
    2.7.5 kept a ``tools/rustwx/target/release/rw_netcdf`` built from the
    older tree, and the first thing that told them was a decode failure
    in the middle of a run.  Nothing on the ladder had asked whether the
    build was current, because the contract marker only answers when
    somebody remembers to change it and a pull that changes a decoder
    usually does not change its marker.

    Asked of ONE rung, :func:`checkout_workspace_of`, and read-only:
    file times, exactly the question cargo answers before it rebuilds.
    Nothing here runs cargo, so a report may call it.
    """

    if _INSPECTION_ONLY:
        return path
    status = checkout_build_status(path)
    if status is None or status.current:
        return path
    raise StaleCheckoutBuildError(status.refusal())



def packaged_bridge_dir() -> Path:
    """The wheel-bundled bridge directory inside this installation.

    ``<site-packages>/woof/libexec/bridges`` for an installed wheel and
    ``<checkout>/woof/libexec/bridges`` in a source tree, where
    ``tools/stage_wheel_bridges.py`` puts the built artifacts before a
    platform wheel is built.  Distinct from ``<root>/libexec/bridges``,
    which is BESIDE the package (the sealed-runtime archive layout) and
    therefore cannot be package data.
    """

    return Path(__file__).resolve().parent.joinpath(*PACKAGED_BRIDGE_SUBDIR)


def _package_parent() -> Path:
    """The directory containing the ``woof`` package.

    A source checkout's repository root, or ``site-packages`` for an
    installed wheel (where the crate does not exist).
    """

    return Path(__file__).resolve().parent.parent


def crate_dir() -> Path:
    """The vendored Rust workspace of a source checkout (may not exist)."""

    return _package_parent() / "tools" / "grib1_bridge"


def executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def artifact_candidates(env_var: str, filename: str, *,
                        crate_relative: str | None = None) -> tuple[Path, ...]:
    """Deterministic candidate paths for one built artifact, best first.

    The resolution order for the selected crate (``tools/grib1_bridge``
    by default, for bridge executables and the CPU preprocessing library):
    environment override, checkout release, checkout debug, ``libexec``
    beside the package, the wheel-bundled :func:`packaged_bridge_dir`,
    user-level default directory.  The environment override comes first;
    a missing file it names is the caller's error to raise (never
    silently skipped).
    """

    candidates: list[Path] = []
    override = os.environ.get(env_var)
    if override:
        candidates.append(Path(override))
    root = _package_parent()
    crate = crate_dir() if crate_relative is None else root / crate_relative
    candidates.extend((
        crate / "target" / "release" / filename,
        crate / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def find_artifact(env_var: str, filename: str, *,
                  crate_relative: str | None = None,
                  executable: bool = True) -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error:
    explicit configuration must fail loudly, not fall through to a
    different executable or library.
    """

    override = os.environ.get(env_var)
    candidates = (artifact_candidates(env_var, filename) if crate_relative is None
                  else artifact_candidates(env_var, filename,
                                           crate_relative=crate_relative))
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            return (accept_resolved(resolved) if executable else
                    accept_resolved(resolved, executable=False))
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{env_var} names a missing file: {candidate}")
    return None


def bridge_candidates(name: str) -> tuple[Path, ...]:
    """Deterministic candidate paths for bridge ``name``, best first."""

    if name not in BRIDGE_ENV:
        raise ValueError(f"unknown bridge executable {name!r}; known: "
                         f"{sorted(BRIDGE_ENV)}")
    return artifact_candidates(BRIDGE_ENV[name], executable_name(name))


def find_bridge(name: str) -> Path | None:
    """First existing candidate for bridge ``name``, or None.

    See :func:`find_artifact` for the fail-loud override contract.
    """

    if name not in BRIDGE_ENV:
        raise ValueError(f"unknown bridge executable {name!r}; known: "
                         f"{sorted(BRIDGE_ENV)}")
    return find_artifact(BRIDGE_ENV[name], executable_name(name))


#: Public data source -> the bridge executable its preparation route
#: launches.  These are product names (the data a user asks for), never
#: case names.
SOURCE_DECODERS = {
    "hrrr": "hrrr_grib2_bridge",
    "gfs": "gfs_grib2_bridge",
    "era5": "grib1_bridge",
}

#: Decoder executable -> the optional MODES it implements beyond its base
#: contract.  A mode is a command a newer build has and an older one does
#: not, so a decoder that speaks this release's base contract
#: (:data:`BRIDGE_ABI_MARKERS`) can still be too old for the mode a caller
#: needs.  Each row is the capability, declared: ``marker`` is the literal
#: the mode compiles into the binary (checked statically, exactly like the
#: base marker), ``command`` is what the decoder calls the mode,
#: ``required_by`` is the option that needs it and ``label`` is how a
#: refusal names it.  Keyed by executable, never by source, so a decoder
#: that gains a mode is one row here and no branch in
#: :func:`resolve_source_decoder`.
DECODER_MODE_CONTRACTS = {
    "hrrr_grib2_bridge": {
        # Append-only lead admission: the decoder takes each lead as it
        # posts instead of a complete window.
        "as_posted": {
            "marker": (b"--series-workers-posted WORKERS SERIES_TSV "
                       b"OUTPUT_DIR SIGNAL_DIR ADMIT_DIR"),
            "command": "--series-workers-posted",
            "required_by": "--as-posted",
            "label": "posted-mode",
        },
    },
}


class DecoderContractError(RuntimeError):
    """A decoder is installed but does not speak this release's contract.

    Distinct from ``FileNotFoundError`` because the two have different
    remedies -- install one versus replace this one -- and because a
    caller that catches "missing" to offer a build must not silently
    swallow "wrong".
    """


class BridgeBuildError(RuntimeError):
    """A cargo build that could not produce a bridge, named by CLASS.

    A third state beside "installed and wrong" and "not installed at
    all": the sources are here, the build was attempted, and the build
    itself failed -- for a reason that is almost never about the code.
    Its own class because the remedy differs from both neighbours and
    because a caller relaying it must be able to say "this was a build,
    not your data".

    ``failure_class`` carries the short slug
    :func:`classify_cargo_failure` assigned, so a caller can branch on
    the CLASS without re-parsing English.
    """

    def __init__(self, message: str, *, failure_class: str = "build-failed"):
        super().__init__(message)
        self.failure_class = failure_class


#: Cargo/linker failure CLASSES, as ``(slug, needles, sentence)`` rows.
#:
#: A TABLE, so a newly-observed failure mode is a row and not another
#: branch.  Each ``needles`` entry is matched case-insensitively against
#: cargo's combined output; the first row that matches names the class.
#: Order matters only where one output could match two rows, and the
#: specific rows are therefore first.
#:
#: The first row is the one that cost the reproduction: on Windows a
#: cdylib that any live process has mapped cannot be replaced, so cargo
#: fails at the *link* step with a filesystem error, several screens
#: below a wall of unrelated compiler warnings.  It is not a code
#: failure and re-running it changes nothing until the holder exits.
CARGO_FAILURE_CLASSES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("artifact-held-open",
     ("being used by another process", "os error 32", "access is denied",
      "os error 5", "text file busy", "os error 26", "permission denied"),
     "a build artifact could not be replaced because another process on "
     "this machine has it open -- a running woof, another worktree's "
     "build, a debugger or an antivirus scan.  Nothing about the source "
     "is wrong and re-running changes nothing until the holder exits"),
    ("build-lock-held",
     ("blocking waiting for file lock",),
     "another cargo is already building in this target directory and "
     "this one could not take the lock"),
    ("lockfile-out-of-date",
     ("the lock file needs to be updated", "--locked"),
     "the checked-in Cargo.lock does not match the manifests, and this "
     "build runs --locked so it will not silently update it"),
    ("vendor-incomplete",
     ("no matching package", "failed to select a version",
      "unable to get packages from source", "not in the vendored sources"),
     "the vendored dependency set does not satisfy this lockfile, and "
     "this build runs --offline so it cannot fetch the difference"),
)

#: What a caller sees when no row matched.  Still names the class -- a
#: build -- because the defect being guarded is a build error read as
#: something else entirely.
_UNCLASSIFIED_CARGO_FAILURE = (
    "build-failed",
    "the cargo build did not complete; its own error lines are below")

#: How many of cargo's error lines are relayed as evidence.  The point
#: of classifying is that the reader does not have to read a compiler's
#: warning wall to find the cause, so this is a tail and not a dump.
_CARGO_EVIDENCE_LINES = 6


def classify_cargo_failure(output: str) -> tuple[str, str]:
    """``(slug, sentence)`` for cargo's own output.  Never raises."""

    lowered = (output or "").lower()
    for slug, needles, sentence in CARGO_FAILURE_CLASSES:
        if any(needle in lowered for needle in needles):
            return slug, sentence
    return _UNCLASSIFIED_CARGO_FAILURE


def _cargo_evidence(output: str) -> str:
    """Cargo's error lines, without the warning wall around them.

    ``warning:`` blocks are the bulk of a normal build's output and they
    are why the reproduction's real cause -- ``failed to remove file`` --
    arrived twelve lines down.  Prefer the ``error``/``Caused by`` lines;
    fall back to the tail only when there are none.
    """

    lines = [line.rstrip() for line in (output or "").splitlines()
             if line.strip()]
    keep: list[str] = []
    in_error = False
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith(("error", "caused by")):
            in_error = True
        elif stripped.startswith(("warning:", "warning[", "note:", "help:",
                                  "= note:", "= help:", "compiling ",
                                  "finished ", "checking ")):
            in_error = False
            continue
        if in_error:
            keep.append(line)
    tail = keep or lines
    return "\n".join(f"    {line}" for line in tail[-_CARGO_EVIDENCE_LINES:])


def cargo_build_refusal(artifact: str, crate_relative: str, *,
                        returncode: int, output: str,
                        one_liner: str | None = None) -> str:
    """The sentence a failed ``cargo build`` for ``artifact`` deserves.

    Three parts, in the order a stuck reader needs them: WHAT failed (a
    build, named), WHY (the class), and the REMEDY this install can
    actually take -- the staged bundle where one exists, the build
    one-liner where the sources do.
    """

    slug, sentence = classify_cargo_failure(output)
    remedy = install_aware_one_line_hint(
        one_liner or cargo_build_one_liner(crate_relative),
        crate_relative, artifact)
    if slug == "artifact-held-open":
        remedy = ("close whatever holds the file (the path is in the "
                  "evidence below), then re-run this command -- or use a "
                  "copy that is already built: " + remedy)
    elif slug == "build-lock-held":
        remedy = ("wait for the other build to finish and re-run this "
                  "command -- or use a copy that is already built: "
                  + remedy)
    return (
        f"the Rust bridge `{artifact}` is not built here and building it "
        f"in {crate_relative} FAILED (cargo exited {returncode}).\n"
        f"  why: {sentence}.\n"
        f"  remedy: {remedy}\n"
        f"  cargo said:\n{_cargo_evidence(output)}")


def cargo_missing_refusal(artifact: str, crate_relative: str) -> str:
    """``cargo`` itself could not be started.  A refusal, not an OSError.

    The build path used to let ``FileNotFoundError: [WinError 2]`` out
    of ``subprocess.run`` untouched, so an install with sources and no
    Rust toolchain ended in a traceback naming ``cargo`` with no hint
    that a toolchain is what is missing.
    """

    return (
        f"the Rust bridge `{artifact}` is not built here, and `cargo` "
        "could not be started to build it -- no Rust toolchain is on "
        "PATH.\n"
        "  why: this install carries the crate sources but nothing that "
        "can compile them, so no route to the decoder exists until one "
        "of the two is supplied.\n"
        "  remedy: " + install_aware_one_line_hint(
            cargo_build_one_liner(crate_relative), crate_relative, artifact)
        + "\n  # install Rust first if that route is the one taken:\n"
        f"  {rust_toolchain_install_command()}\n"
        f"  {cargo_activation_command()}")


def resolve_source_decoder(source: str, *, mode: str | None = None) -> Path:
    """THE decoder ``source``'s preparation will launch, or a refusal.

    One function, called by the preparation wrapper AND by ``woof
    doctor``, because the alternative was two resolvers with different
    answers: the wrapper resolved a *source-tree cargo workspace*
    (``<root>/tools/grib1_bridge/target/release/...``) that exists only
    in a checkout, while doctor resolved the shared ladder in this
    module.  On a wheel install doctor therefore reported the bridge
    ``woof setup`` had staged in ``~/.woof/bridges`` -- correctly --
    and the preparation then went looking under ``site-packages/tools``
    and refused.  "No gaps" followed by a missing file is the exact
    failure a pre-flight exists to prevent, and it can only be prevented
    by asking the same question through the same code.

    Resolution is :func:`bridge_candidates`' order: the environment
    override (a missing file it names is a hard error, never a silent
    fall-through), then a checkout's own build, then ``libexec/bridges``
    beside the package, then the user-level ``~/.woof/bridges`` that
    ``woof fetch-bridges`` stages into.  A checkout's build coming
    before the staged copy is deliberate: a developer's rebuild must win
    over whatever they downloaded last month.

    Existence is not the whole question, so it is not the whole answer.
    A binary built before this release's decoder contract is present,
    executable, and wrong, and until 1.8.9 this function returned it:
    ``woof doctor`` asked :func:`bridge_abi_matches` afterwards and the
    production preparation did not, so a stale bridge passed the door
    that runs it while the report that checks it said green.  A 12-byte
    text file planted at each override path was returned by all three
    sources.  The gate lives INSIDE the resolver now -- one question,
    one answer, no second call for a caller to forget.

    ``mode`` additionally requires a capability the decoder declares in
    :data:`DECODER_MODE_CONTRACTS` (``"as_posted"`` is append-only lead
    admission). An older decoder remains valid for its complete window
    mode but cannot be returned for a command it does not implement, and
    a decoder that declares no such mode is refused by name.

    Nothing here runs cargo, so it is safe to call from a report.
    """

    if source not in SOURCE_DECODERS:
        raise ValueError(
            f"no decoder is declared for source {source!r}; known: "
            f"{sorted(SOURCE_DECODERS)}")
    name = SOURCE_DECODERS[source]
    declared = (None if mode is None
                else DECODER_MODE_CONTRACTS.get(name, {}).get(mode))
    if mode is not None and declared is None:
        raise ValueError(f"no decoder mode contract is declared for {source!r} {mode!r}")
    found = find_bridge(name)
    if found is not None:
        ok, evidence = bridge_abi_matches(name, found)
        if ok and declared is not None:
            try:
                ok = declared["marker"] in Path(found).read_bytes()
            except OSError as error:
                ok, evidence = False, (
                    f"cannot read its {declared['label']} contract: {error}")
            else:
                if not ok:
                    evidence = (
                        f"does not implement {declared['command']}, which "
                        f"{declared['required_by']} requires; "
                        "rebuild it from a matching checkout")
        if ok:
            return found
        raise DecoderContractError(
            f"the {source} route's decoder at {found} {evidence}.\n"
            + install_aware_build_hint(cargo_build_one_liner(CRATE_RELATIVE)))
    raise FileNotFoundError(
        f"the {source} route's decoder ({executable_name(name)}) is not "
        "installed here.  Searched, in order: "
        + ", ".join(str(candidate) for candidate in bridge_candidates(name))
        + "\n" + bridge_remedy(name))


def bridge_abi_matches(name: str, path: Path) -> tuple[bool, str]:
    """Does the built bridge at ``path`` speak the contract woof uses?

    Read-only and static: it searches the binary for the marker rather
    than adding a ``--abi`` subcommand the already-built binaries on
    every user's disk would not have.  That is the whole point -- the
    skew this catches is precisely a binary that predates the change,
    and a handshake only new builds can answer would report the stale
    ones as broken rather than as stale, or not at all.

    A bridge with no declared marker answers ``True``: the absence of a
    marker means nobody has yet named this bridge's contract, which is
    not evidence of skew.  Fail closed on the check that IS declared,
    never on the one that is not.
    """

    marker = BRIDGE_ABI_MARKERS.get(name)
    if marker is None:
        return True, "no declared contract marker for this bridge"
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        return False, f"cannot read it to check its contract: {error}"
    if marker in payload:
        return True, "speaks this release's contract"
    return False, (
        "was built from a checkout that predates this release's "
        f"{name} contract, so it will refuse inputs woof writes "
        "correctly; rebuild it from a matching checkout")


def fetched_on_demand(artifact: str | None) -> bool:
    """Is ``artifact`` one the platform wheel leaves to the bundle?

    True for :func:`woof.bridge_assets.on_demand_artifacts`: pinned in the
    release bundle and not carried in the wheel, so on a fresh pip install
    its absence is the expected state and its first use stages it.
    """

    if artifact is None:
        return False
    try:
        from woof import bridge_assets
        return any(entry.name == artifact
                   for entry in bridge_assets.on_demand_artifacts())
    except Exception:                                # noqa: BLE001
        return False


def on_demand_fetch_offered(artifact: str | None,
                            crate_relative: str = RUSTWX_CRATE_RELATIVE) -> bool:
    """Can this install stage the absent on-demand ``artifact`` by download?

    True only where it can: the artifact is declared on demand, this install
    carries no Rust sources for its crate (a checkout builds it instead), and
    the pins this install carries name a bundle for this platform that holds
    it.  Anywhere else a fetch would refuse, so nothing offers one.
    """

    if not fetched_on_demand(artifact):
        return False
    try:
        return (not sources_present(crate_relative)
                and _offer_for(artifact) is not None)
    except Exception:                                # noqa: BLE001
        return False


#: On-demand artifacts this process already tried to stage, with the failure
#: text when it did not work.  One attempt per artifact per run, so a box
#: with no route to the release assets does not re-dial for every resolve.
_ON_DEMAND_ATTEMPTS: dict[str, str | None] = {}
# Held across the whole fetch: concurrent callers (a scoring pool's threads)
# used to all pass the "already tried" check before the first finished, so
# each warned and dialled the bundle again (rw_mrms warned 12 times in hex
# scoring).  Waiters see the first caller's result instead.
_ON_DEMAND_LOCK = threading.Lock()


def on_demand_failure(artifact: str) -> str | None:
    """Why this process's automatic staging of ``artifact`` failed, if it did."""

    return _ON_DEMAND_ATTEMPTS.get(artifact)


def fetch_on_demand(artifact: str) -> bool:
    """Stage the on-demand ``artifact`` now; True when it was staged.

    Called by a door that is about to run and found nothing on the ladder.
    It downloads the one pinned ZIP member by HTTP range from this release's
    bundle (:func:`woof.bridge_partial.fetch_partial`), verifies its size,
    SHA-256 and contract marker against the pins packaged with this install,
    and installs it in :func:`default_bridge_dir`, the rung the ladder
    already reads.  Never inside :func:`inspection_only` (doctor and the
    receipt writers report the estate; they never download), never for an
    artifact the wheel carries, and never where the fetch could not deliver
    it (:func:`on_demand_fetch_offered`).  A failure is remembered and
    returned by :func:`on_demand_failure` so the door's refusal can say
    what happened.
    """

    if _INSPECTION_ONLY:
        return False
    with _ON_DEMAND_LOCK:
        if artifact in _ON_DEMAND_ATTEMPTS:
            # Tried already in this process: True when that attempt staged
            # it (a waiter re-reads the ladder), False when it failed and
            # on_demand_failure() holds the reason.
            return _ON_DEMAND_ATTEMPTS[artifact] is None
        if not on_demand_fetch_offered(artifact):
            return False
        return _fetch_on_demand_locked(artifact)


def _fetch_on_demand_locked(artifact: str) -> bool:
    from woof import bridge_assets
    from woof.bridge_partial import fetch_partial
    from woof.explain import warn

    destination = default_bridge_dir()
    warn(f"{artifact} is fetched on demand: the platform wheel does not carry "
         f"it, so its pinned copy is being staged from this release's bundle "
         f"into {destination} before this run continues")
    try:
        pins = bridge_assets.load_pins()
        bundle = pins.bundle_for(bridge_assets.host_platform())
        fetch_partial(pins, bundle, destination, [artifact],
                      progress=lambda message: print(message, file=sys.stderr))
    except (bridge_assets.BridgeAssetError, OSError) as error:
        _ON_DEMAND_ATTEMPTS[artifact] = (
            f"staging it from this release's bundle failed ({error})")
        return False
    _ON_DEMAND_ATTEMPTS[artifact] = None
    return True


def artifact_remedy(*, env_var: str, filename: str, subject: str,
                    crate_relative: str = CRATE_RELATIVE,
                    one_liner: str = "", artifact: str | None = None) -> str:
    """The remedy for one missing built artifact, true for THIS install.

    Two installs, two different true answers.  In a source checkout the
    crate is right there and the fix is one ``cargo build`` -- with the
    real destination path, not a ``<clone>`` the reader has to expand.
    On a pip install there is no crate at all, so the remedy starts from
    the clone; printing the checkout's one-liner there names a directory
    that does not exist and reads as a broken install.

    Every continuation line is a command or a ``#`` comment, so the
    whole block survives being pasted in one go.  The guidance that used
    to trail as bare prose (``then EITHER set ...``) is commented for
    exactly that reason: a reader who selects the block should not have
    to prune it first.

    One implementation for the bridges, the renderer and the fetch
    backbone.  Three copies is how two of them kept ``<clone>`` and the
    "exact copy-pasteable" claim after the third stopped saying it.

    On a pip install there is now a third true answer, when the release
    published a bundle this platform can run: one download, verified
    against the packaged pins.  It leads, and the clone-and-build route
    follows it commented out -- still printed, because it is the only
    route on a platform with no bundle, and commented because a reader
    who pastes the report must not also compile what the line above
    already staged.

    ``artifact`` is the bundle name of the thing being remedied, and
    passing it is how a caller says "check that the bundle carries THIS
    one" rather than "check that a bundle exists".  See
    :func:`prebuilt_bundle_offer` for the wave that cost.  Callers that
    omit it get the old behaviour, which is correct for the artifacts
    that have always been in the bundle and is what the checkout branch
    above does anyway.
    """

    crate = _package_parent() / Path(crate_relative)
    built = crate / "target" / "release" / filename
    if crate.is_dir():
        # The build line must name the crate this remedy is ABOUT.  The
        # fallback used to be CARGO_BUILD_HINT, which is frozen to
        # CRATE_RELATIVE (tools/grib1_bridge): every caller that passed
        # a different `crate_relative` -- rw_netcdf lives in
        # tools/rustwx -- printed `cd tools/grib1_bridge` one line above
        # a "that produces .../rustwx/target/release/rw_netcdf.exe"
        # promise the command cannot keep.  Building grib1_bridge
        # produces no rw_netcdf, so a reader who pasted the block landed
        # exactly where they started, on the refusal that sent them.
        # This is the same defect
        # test_the_bridge_remedy_is_a_real_bootstrap_on_a_pip_install
        # already records being fixed once for the six bridges; it
        # survived here because this fallback ignores its own argument.
        return (
            f"# build {subject} once, from this checkout's root:\n"
            f"  {one_liner or cargo_build_one_liner(crate_relative)}\n"
            f"  # that produces {built},\n"
            f"  # which woof then finds on its own (or set {env_var} "
            f"to it).")
    steps = "\n".join(build_from_clone_hint(crate_relative))
    clone_built = _shell_path(CLONE_DIR, crate_relative,
                              "target/release") + \
        ("\\" if WINDOWS_SHELL else "/") + filename
    source_route = (
        f"{steps}\n"
        f"  # building it is not the same as wiring it, so finish the job:\n"
        + install_into_default_bridge_dir(clone_built) + "\n"
        f"  # OR, instead of copying, point woof at the build in place:\n"
        f"  #   {env_var}={clone_built}\n"
        f"  #   (relative to the directory you ran git clone in)")
    offer = _offer_for(artifact)
    if offer is None:
        return (
            "# this install carries no Rust sources -- the wheel ships "
            "none --\n"
            f"# so {subject} must be built from a clone.  About two "
            "minutes, once:\n"
            f"{source_route}")
    if fetched_on_demand(artifact):
        # The expected state of a fresh pip install, so it is said as one:
        # nothing is broken, and the one-file download is the whole remedy.
        return (
            f"# {subject} is fetched on demand: the platform wheel does not\n"
            "# carry it, and this one command stages it from this release's\n"
            "# prebuilt bundle, verified against the packaged pins:\n"
            f"  woof fetch-bridges --partial --artifact {artifact}\n"
            "  # (plain `woof fetch-bridges` stages the whole bundle instead)\n"
            "  # Or build it from source instead -- the route that works on\n"
            "  # every platform, including ones with no published bundle.\n"
            "  # Uncomment these to take it:\n"
            + _as_comments(source_route))
    return (
        "# this install carries no Rust sources -- the wheel ships none --\n"
        f"# so there are two routes to {subject}: the prebuilt bundle\n"
        "# this release published, or a clone and a build.  The download\n"
        "# is one command:\n"
        + "\n".join(offer) + "\n"
        "  # Or build it from source instead -- the route that works on\n"
        "  # every platform, including ones with no published bundle.\n"
        "  # Uncomment these to take it:\n"
        + _as_comments(source_route))


def install_into_default_bridge_dir(built: str) -> str:
    """Copy one built artifact where woof looks by default: COMMANDS.

    The wiring step used to be offered as two ``#`` alternatives -- copy
    here, or export that -- and a reader who did exactly what the
    contract promises ("every line is a command to run as printed, in
    the order printed, or a ``#`` comment") ran the whole report and
    still ended at six MISSING bridges, because the only step that
    finishes the job was commented out on both branches.  A choice
    between two alternatives is real, but one of them can be the
    default: this is the copy, as commands, with the environment
    variable demoted to the comment beneath it.

    The destination is :func:`default_bridge_dir` spelled out in full
    rather than through ``$HOME``/``$env:USERPROFILE``, so the path the
    reader pastes is the path :func:`artifact_candidates` searches --
    even on a machine whose ``HOME`` disagrees with its passwd entry,
    which is exactly the environment a scratch-HOME validation run has.
    """

    destination = str(default_bridge_dir())
    if WINDOWS_SHELL:
        return (f'  New-Item -ItemType Directory -Force "{destination}"\n'
                f'  Copy-Item "{built}" "{destination}"')
    return (f'  mkdir -p "{destination}"\n'
            f'  cp "{built}" "{destination}"')


def bridge_remedy(name: str) -> str:
    """The remedy for a missing GRIB bridge, true for THIS install.

    Deliberately does NOT pass ``artifact``, unlike
    :func:`cpu_bridge_remedy` and :meth:`woof.obs.frontdoor.FrontDoor
    .remedy`.  The five GRIB decoders have been in every bundle this
    project has published, so the membership question the parameter asks
    has one answer for them and adding it would change no output -- while
    it WOULD change the arity of the ``prebuilt_bundle_offer`` call, and
    ``woof doctor``'s tests substitute a zero-argument stand-in for that
    function.  Breaking a stand-in to produce identical text is a bad
    trade.  The doctor lane widening that stand-in is the handoff that
    unblocks it; until then this is the historical call, unchanged.
    """

    return artifact_remedy(
        env_var=BRIDGE_ENV[name], filename=executable_name(name),
        subject="the GRIB bridges")


#: The bundled name and environment variable of the parallel CPU
#: preprocessing library.  Declared here beside every other artifact's
#: because :mod:`woof.ingest.cpu_backend` is a numpy-importing module
#: and its resolver's REMEDY has to be composable from this one, which
#: imports nothing but the standard library.
CPU_BRIDGE_ARTIFACT = "gpuwm_preprocess_cpu"
CPU_BRIDGE_ENV = "WOOF_CPU_PREPROCESS_BRIDGE"


def cpu_bridge_remedy(filename: str) -> str:
    """The remedy for a missing CPU preprocessing library.

    ``ingest/cpu_backend.py``'s resolver was the one refusal in the
    estate whose message never said how to fix it: it listed the paths
    it searched and stopped.  Every path it lists is a rung of the same
    ladder ``woof fetch-bridges`` stages into, and the library IS in
    the bundle -- so the answer existed the whole time and the message
    just never carried it.  Composed from the shared builder so it says
    what this install can actually do, exactly like the GRIB bridges'.
    """

    return artifact_remedy(
        env_var=CPU_BRIDGE_ENV, filename=filename,
        subject="the parallel CPU preprocessing library",
        artifact=CPU_BRIDGE_ARTIFACT)


#: A bridge's out-of-range refusal, as it reaches Python: the field, the
#: decoded value, and the range it left.  Matched only to explain the
#: refusal -- never to suppress it.
_BOUND_REFUSAL = re.compile(
    r"(?P<field>\S+) value (?P<value>\S+) outside \[(?P<range>[^\]]*)\]")


def decode_failure_message(subject: str, stderr: str) -> str:
    """A decoder's refusal, with the reason it gives and what to do next.

    The bridges fail closed and say why on stderr; Python's job is to
    carry that verbatim and add the sentence a user cannot be expected
    to supply -- what the number means, and whether re-running can help.

    An out-of-range refusal earns a specific remedy, because the obvious
    reading of it is wrong.  A field value a hair outside a physical
    bound is GRIB2 packing, not bad data, and the bridge now clamps
    those against the record's own quantization step; so a refusal that
    survives that says the value is further out than the encoding can
    explain.  That points at the bytes, which a re-fetch can fix, rather
    than at the bound, which no one should widen.

    Every remedy line is a command or a ``#`` comment, so the block
    survives being pasted whole.
    """

    detail = stderr.strip()
    message = f"{subject} failed: {detail}" if detail else f"{subject} failed"
    match = _BOUND_REFUSAL.search(detail)
    if match is None:
        return message
    field = match.group("field")
    value = match.group("value")
    bounds = match.group("range")
    return message + (
        "\n  remedy:"
        f"\n  # {field} decoded {value}, outside its physical range "
        f"[{bounds}]."
        "\n  # A value that merely touches a bound is expected -- GRIB2"
        "\n  # packing rounds onto a fixed grid, so a cell encoded AT a"
        "\n  # limit can decode a step past it -- and the bridge already"
        "\n  # clamps those, within that record's own packing step."
        "\n  # This one is further out than the encoding can account for,"
        "\n  # so the suspect is the source record, not the range."
        "\n  # Re-fetch the cycle and re-run; a truncated or corrupted"
        "\n  # download is the usual cause and is not detectable any"
        "\n  # earlier than here.  If it survives a clean re-fetch, the"
        "\n  # published record is bad and the refusal is correct.")


__all__ = [
    "decode_failure_message",
    "SOURCE_DECODERS", "DECODER_MODE_CONTRACTS", "resolve_source_decoder",
    "DecoderContractError",
    "launchable", "native_executable_format", "quiet_loader_errors",
    "BRIDGE_ABI_MARKERS", "bridge_abi_matches",
    "CheckoutBuildStatus", "StaleCheckoutBuildError", "checkout_build_status",
    "checkout_workspace_of", "require_current_checkout_build",
    "BRIDGE_ENV", "CARGO_BUILD_HINT", "CLONE_DIR", "CRATE_RELATIVE",  # noqa: F822 -- module __getattr__
    "CPU_BRIDGE_ARTIFACT", "CPU_BRIDGE_ENV", "cpu_bridge_remedy",
    "REPOSITORY_URL", "RUSTWX_CRATE_RELATIVE", "WINDOWS_SHELL",
    "artifact_candidates", "cargo_build_one_liner",
    "artifact_remedy", "bridge_candidates", "bridge_remedy",
    "BridgeBuildError", "CARGO_FAILURE_CLASSES", "cargo_build_refusal",
    "cargo_missing_refusal", "classify_cargo_failure",
    "build_from_clone_hint", "cargo_activation_command",
    "cargo_executable", "cargo_is_installed", "crate_dir",
    "default_bridge_dir", "legacy_bridge_dir", "legacy_bridge_candidates",
    "staged_version_tag", "staging_location_note",
    "executable_name", "find_artifact", "find_bridge",
    "StaleBridgeError", "accept_resolved", "inspection_only",
    "require_release_pin",
    "install_aware_build_hint", "install_into_default_bridge_dir",
    "lazy_build_hints", "run_if_first_succeeds", "rustwx_build_hint",
    "rust_toolchain_install_command", "shell_line",
    "install_aware_one_line_hint",
    "prebuilt_bundle_offer",
    "sources_present",
]
