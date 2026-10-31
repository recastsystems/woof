"""Immutable source-family authorities shipped with the RW-WPS wheel.

A PACKAGED PROFILE is a source whose three declarative authorities -- the
``rw-wps.mapping.v1`` mapping, the composition contract, and the terrain
provenance document -- ship inside the wheel and are pinned here by
SHA-256.  Everything else about such a source is table data too: its row in
:mod:`woof.source_adapters` names the profile, and the front door reads
the profile rather than a per-source function.

Adding a model to this file is three JSON documents and one row of
:data:`_PACKAGED_PROFILES`.  That is the whole point: a source whose
mapping is shipped is not a code path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_AUTHORITY_ROOT = Path(__file__).with_name("authorities")

#: The three roles every packaged profile declares, in the order the front
#: door passes them.
PROFILE_ROLES = ("mapping", "composition", "provenance")


def _profile(
    stem: str,
    *,
    source_format: str,
    mapping: str,
    composition: str,
    provenance: str,
    data_role: str | None = None,
    provenance_role: str | None = None,
    composition_state: str = "composed",
    contributing_mappings: Mapping[str, Mapping[str, str]] | None = None,
    contributing_provenances: Mapping[str, Mapping[str, str]] | None = None,
    input_normalizer: str | None = None,
    normalization: str | None = None,
) -> Mapping[str, object]:
    """One packaged profile: file names, byte pins, and its two roles.

    ``data_role``/``provenance_role`` repeat what the composition document
    declares in ``supplements.terrain_height`` (or, for a cross-source
    profile, on the binding that provides terrain); they are stated here
    as well because the front door has to spell them on the command line
    BEFORE anything opens the composition, and a role the caller guesses is
    a role that silently binds the wrong file.  They are checked against
    the composition at decode time by
    :func:`woof.mapped_composition.decode_composed_source`, which refuses
    on any difference -- so a wrong row here fails loudly, not quietly.

    ``composition_state`` is ``"composed"`` for a runnable profile.  An
    atmosphere-only source -- complete 3-D state, zero published land
    surface -- ships ``"pending_cross_source"`` instead: its composition
    document is an explicit PENDING declaration
    (:data:`woof.mapped_composition.PENDING_COMPOSITION_SCHEMA`) that
    refuses to load by naming the state the source does not publish, it
    supplies no terrain supplement, and therefore it has no roles.  A
    pending profile decodes and inspects through the mapped route; it
    does not initialize anything until the cross-source composition
    supplying the missing state lands.

    ``contributing_mappings`` is the CROSS-SOURCE slot: a composed profile
    whose composition declares ``field_sources`` bindings ships each
    contributing source's own mapping document as an additional pinned
    authority, keyed by the binding's ``mapping_role`` --
    ``{role: {"file": name, "sha256": digest}}``.  The front door passes
    each one as ``--contributing-mapping role=path`` so the prepared
    runner can hand ``contributing_mappings`` to the composed decode; the
    composition independently pins the same digest, so a wrong row here
    fails loudly at decode, not quietly.  Only a composed profile may
    declare the slot: a pending profile has no runnable composition to
    bind anything to.  Shipping the row is what lets the front door pass
    the donor table to the prepared runner without the caller supplying
    (or being able to substitute) one.
    """

    if composition_state not in {"composed", "pending_cross_source"}:
        raise ValueError(
            f"unknown composition_state {composition_state!r} for {stem}"
        )
    if (input_normalizer is None) != (normalization is None):
        raise ValueError(
            f"profile {stem} must declare an input normalizer NAME and the "
            "SHA-256 of the normalization document that defines it, or "
            "neither"
        )
    if input_normalizer is not None and composition_state != "composed":
        raise ValueError("an input normalizer requires a composed profile")
    roles_declared = data_role is not None and provenance_role is not None
    if composition_state == "composed" and not roles_declared:
        raise ValueError(
            f"composed profile {stem} must declare data_role and "
            "provenance_role; only a pending_cross_source profile has none"
        )
    if composition_state == "pending_cross_source" and (
        data_role is not None or provenance_role is not None
    ):
        raise ValueError(
            f"pending profile {stem} has no terrain supplement and "
            "therefore no roles to declare"
        )
    contributing: dict[str, Mapping[str, str]] = {}
    for role, pin in (contributing_mappings or {}).items():
        if composition_state != "composed":
            raise ValueError(
                f"profile {stem} declares a contributing mapping but is "
                "not composed; only a runnable composition binds donors"
            )
        if set(pin) != {"file", "sha256"}:
            raise ValueError(
                f"profile {stem} contributing mapping {role!r} must pin "
                "exactly a file name and its sha256"
            )
        contributing[str(role)] = MappingProxyType({
            "file": str(pin["file"]), "sha256": str(pin["sha256"]),
        })
    provenance_pins: dict[str, Mapping[str, str]] = {}
    for role, pin in (contributing_provenances or {}).items():
        if composition_state != "composed" or set(pin) != {"file", "sha256"}:
            raise ValueError(
                f"profile {stem} contributing provenance {role!r} needs a "
                "composed profile and exactly a file name and its sha256")
        provenance_pins[str(role)] = MappingProxyType({
            "file": str(pin["file"]), "sha256": str(pin["sha256"]),
        })
    return MappingProxyType({
        "source_format": source_format,
        "files": MappingProxyType({
            "mapping": f"{stem}.mapping.json",
            "composition": f"{stem}.composition.json",
            "provenance": f"{stem}.provenance.json",
            **({"normalization": f"{stem}.normalization.json"}
               if input_normalizer else {}),
        }),
        "sha256": MappingProxyType({
            "mapping": mapping,
            "composition": composition,
            "provenance": provenance,
            **({"normalization": normalization} if input_normalizer else {}),
        }),
        "data_role": data_role,
        "provenance_role": provenance_role,
        "composition_state": composition_state,
        "contributing_mappings": MappingProxyType(contributing),
        **({"contributing_provenances": MappingProxyType(provenance_pins)}
           if provenance_pins else {}),
        # Absent for a profile whose source publishes bytes the mapped engine
        # already reads: their public declarations remain unchanged.
        **({"input_normalizer": input_normalizer} if input_normalizer else {}),
    })


_PACKAGED_PROFILES = MappingProxyType({
    "gefs-reforecast-grib2-v1": _profile(
        "rw-wps-gefs-reforecast-grib2", source_format="grib2",
        mapping="576de4015f181e72b63b9afb7e6c50e08772cbb9ff009efa74c38c255a252e87", composition="593368d096f71277706214d0cf1ab6898ac0bd428a71bb64be9049c982225d92", provenance="20bd3d3641cbade7948114b08cae950153173f28e464567c6b560cbfbcc9c45c",
        data_role="gefs_member_in_band_surface", provenance_role="gefs_member_in_band_surface_provenance",
        contributing_mappings={'land_mapping': {'file': 'rw-wps-gefs-retrospective-land.mapping.json', 'sha256': '55af957db69d20353833dc4f47a8b270cb164e8c7abe5c8bbcd4ce96c4449df2'}, 'seaice_mapping': {'file': 'rw-wps-ncep-r1-seaice-netcdf.mapping.json', 'sha256': '754251fa04dedae5dff4ff65bc040546871a82951424431c5de240f2850caa9b'}},
        contributing_provenances={'land_provenance': {'file': 'rw-wps-gefs-retrospective-land.provenance.json', 'sha256': 'e9660dffeb7f622f1d10162302f5694d9c40e44f8566bab12f426d12f1e8b4c3'}, 'seaice_provenance': {'file': 'rw-wps-ncep-r1-seaice-netcdf.provenance.json', 'sha256': '50313934795556e2bccbd3ed1f9e81d7c780fc8cb52abbc6ce4219b37f045266'}},
    ),

    "ecmwf-ens-open-grib2-v1": _profile(
        "rw-wps-ecmwf-ens-open-grib2",
        source_format="grib2",
        mapping="e60114f2949e80df748d2d9fd1955aec3c1c2471db19b5ecf665712383806740",
        composition="a58313084790807306ddca80fa1009d5187c2485aff3596a0ba1330d8940f9b5",
        provenance="e6127a3c9da1ba3834a529c135a177d3652d80bbb2c5c7ca68891e8c87d0bec3",
        data_role="ecmwf_open_data_in_band_surface",
        provenance_role="ecmwf_open_data_in_band_surface_provenance",
        # Sea ice: the ENS open data publishes thickness only, never a
        # concentration, so every member start had no sea ice anywhere.
        # The composition binds sea_ice_fraction to the same cycle's GFS
        # analysis ICEC (the route's declared gfs f000 donor), nearest
        # 0.25-degree cell, as ecmwf-open-data-oper-grib2-v1 does.
        contributing_mappings={
            "seaice_mapping": {
                "file": "rw-wps-earth2-gdas-donor.mapping.json",
                "sha256": "e24542b3a9747c37614451d29eb4047b5f17f6068e3e63c299a2cc5367d8d8a1",
            },
        },
        contributing_provenances={
            "seaice_provenance": {
                "file": "rw-wps-gfs-analysis-seaice.provenance.json",
                "sha256": "2ced4ae3b2d6f8974fd9a259b97bf18483f972b41e66ff1b112b297468b0af64",
            },
        },
    ),

    # Native global ICON carries its coordinates in separate GDT-101
    # CLAT/CLON records. Normalize before the existing mapped authority is
    # authored, never disguise its unstructured array as an embedded grid.
    # A173 moved the mapping and normalization digests: DWD posts every
    # field hourly to f078 (listed 2026-10-01 00Z), so both now declare the
    # publisher's 1 h spacing and take its whole multiples; 3 h stays the
    # default (the registry row and the route's default_cadence).
    "icon-global-grib2-v1": _profile(
        "rw-wps-icon-global-grib2",
        source_format="grib2",
        mapping="7bf357241b1be4682df42371a2c722db1437b8df9d004d1e7851e00a61e7853f",
        composition="a75dc9deabf72d750eb5d3f333274ed85d2d190e69a09f5aaf08f5c44891604b",
        provenance="1770e4b1c4092d53db0ee5b18aa371465215c04325f4de2610f0d4f740eb0b09",
        data_role="icon_global_invariant_surface",
        provenance_role="icon_global_invariant_surface_provenance",
        input_normalizer="icon-gdt101-pressure-v1",
        normalization="d1fc9b43d7c4815cbb1f09e0a456a68097310718643a66b9b887001cc0106e07",
    ),
    # DWD's 2.2 km ICON-D2 on its native R19B07 mesh (542,040 cells), the
    # same GDT-101 normalization with its own mesh, ladder and cadence rows.
    # DWD also publishes a regular-lat-lon ICON-D2, but that product masks
    # the 17 % of its bounding box outside the model domain in every field,
    # and the mapped route validates each field over its whole extent, so
    # the native mesh is the product the route can read.  Selectors were
    # authored from real 2026-09-27 12Z bytes (grib_ls inventory).
    "icon-d2-grib2-v1": _profile(
        "rw-wps-icon-d2-grib2",
        source_format="grib2",
        mapping="26900d41048587a750f71a104d2a9d36c05c556f5871e8025a745bb0b1ff6f47",
        composition="a4b05c684fee8db595ea2b16f4cb2ba8af807415af4e494b57f948061268e8a6",
        provenance="0241a993bd971863aed47e34fba254448f3c301f1f362a7ef7228701a4d58559",
        data_role="icon_d2_invariant_surface",
        provenance_role="icon_d2_invariant_surface_provenance",
        input_normalizer="icon-d2-gdt101-model-level-v1",
        normalization="7e4d50967e93c0df585cc2f9a18f331b2d2b90217fd7891c9d6ef6a5234be150",
    ),
    "20crv3-member-grib2-v1": _profile(
        "rw-wps-20crv3-member-grib2",
        source_format="grib2",
        mapping="75089fd4973e5e9f246f2dc63f9862d7d625e28a02e6b7e56133c1753101e648",
        composition="aa4f3fac03c09e8461c5e6c5e04a6bed48b5ad477babc4c75e8dd10fd92fe7b2",
        provenance="d1248e1b091f59841757a98a024cbe2868cebc25308f4eb4f9608e2c1755f3b1",
        data_role="twentycrv3_in_band_surface",
        provenance_role="twentycrv3_in_band_surface_provenance",
    ),
    "20crv3-netcdf-v1": _profile(
        "rw-wps-20crv3-netcdf",
        source_format="netcdf",
        mapping="6d7e33c944e36a1d23f621cd8ba08bb3ac46f7b5ce5ea94584ca5490c6de74ec",
        composition="2c243fe4c4dba1c8f47178f2be583f3a20148d54d77101ee3421d1824d10b1c5",
        provenance="28f6dfbb13b866ec336d0532fd731ffb2d49b4990c2f8a7204a70058339a99af",
        data_role="twentycrv3_netcdf_recovered_invariant",
        provenance_role="twentycrv3_netcdf_recovered_invariant_provenance",
    ),
    # A woof hex run as an ArWen parent: the export door's frames on a
    # regular lat-lon window (region-latlon: or global-mesh), decoded on
    # the 55 MPAS model levels.  Terrain is in band (HGT in every frame).
    "hex-history-netcdf-v1": _profile(
        "rw-wps-hex-history-netcdf",
        source_format="netcdf",
        mapping="e992c19fd9957ca050e8038e65033b1b886857a3ce80e47448515e67e081279a",
        composition="bbd2c5b52f13ee8f1ebfac8a138fd109ae883877d5d6fee3a5684172bec8d875",
        provenance="cd38d89e487695716cea9298be4ecb1f2ec0a93b96955ce1b0ac34a1ba195890",
        data_role="hex_in_band_surface",
        provenance_role="hex_in_band_surface_provenance",
    ),
    # HRRR's public pressure-level product (wrfprs), decoded through the
    # generic mapped route: the Lambert CONUS grid, grid-relative wind
    # rotation and nine-node RUC soil are all TABLE DATA in these three
    # documents.  Selectors were authored from real 2026-08-15 00Z bytes
    # through the converged grib-core inventory.  Cloud ice also lists
    # CICE (0/6/0), the identity HRRR published it under before HRRRv3
    # (July 2018), after CIMIXR (0/1/82), read from real 2017-01-19 00Z
    # bytes: with CIMIXR alone every earlier cycle was refused at
    # preparation for lacking cloud_ice_mixing_ratio.
    "hrrr-prs-grib2-v1": _profile(
        "rw-wps-hrrr-prs-grib2",
        source_format="grib2",
        mapping="8e86e27af452391159881059fbaae3ef6451e86538d5caec75f8ca639e12c6eb",
        composition="7e1739721286ed0c2bf0c44ef480bc80cf3a6c0f5fb032e46ae5c2a423544eb4",
        provenance="c7f2ea4142fe09c687f2b1ac0f4d4c462cb55398bfe5149fe9008d9d03ff2cc0",
        data_role="hrrr_prs_in_band_surface",
        provenance_role="hrrr_prs_in_band_surface_provenance",
        contributing_provenances={
            "vegetation_surface_provenance": {
                "file": "rw-wps-hrrr-surface-vegetation-grib2.provenance.json",
                "sha256": "af36e1c31cc24813294a9af015b894529f5b9f3beb6103b3ae593b677763f566",
            },
        },
        contributing_mappings={
            "vegetation_surface_mapping": {
                "file": "rw-wps-hrrr-surface-vegetation-grib2.mapping.json",
                "sha256": "adefc1d166ab4932616d91e206542e527b9c38e9a7e4e72ba8e04a9404049f8a",
            },
        },
    ),
    # RAP's awp130bgrb product (13 km CONUS Lambert, 50 native hybrid
    # levels): the complete state, in-band terrain and the nine-node RUC
    # soil column in one file per valid time.
    "rap-native-grib2-v1": _profile(
        "rw-wps-rap-native-grib2", source_format="grib2",
        mapping="dac1731f40e9a2334dc48d690419880f0ea01434a6a8f15a955cb1de8b197b27",
        composition="5e1191db5f310fef5e6f27af7a8268662785b0e10a1bee844d825931be0e7cc5",
        provenance="89ba6be652b95b99da759d1ae88d09235335f45b41117f767f48fcb228ac3c1c",
        data_role="rap_native_in_band_surface",
        provenance_role="rap_native_in_band_surface_provenance",
    ),
    # HRRR's wrfnat hybrid atmosphere at each valid time; terrain and the
    # nine-node soil column come from the same cycle's wrfprs analysis.
    "hrrr-native-grib2-v1": _profile(
        "rw-wps-hrrr-native-grib2", source_format="grib2",
        mapping="b8227dfb5a30165661f8268bd2b8b45d7040d9f71dd49103054df14f82320a0d",
        composition="3197e0372f333489cdfb017246cdf4809647229ffde97c8ec928639ee52daf87",
        provenance="23f05eb17c5d3373c118df6d3ea84a97530f7e9c386bfae9ae37f34fffd44495",
        data_role="soil_surface_data",
        provenance_role="soil_surface_provenance",
        contributing_provenances={
            "vegetation_surface_provenance": {
                "file": "rw-wps-hrrr-surface-vegetation-grib2.provenance.json",
                "sha256": "af36e1c31cc24813294a9af015b894529f5b9f3beb6103b3ae593b677763f566",
            },
        },
        contributing_mappings={
            "soil_surface_mapping": {
                "file": "rw-wps-hrrr-prs-grib2.mapping.json",
                "sha256": "8e86e27af452391159881059fbaae3ef6451e86538d5caec75f8ca639e12c6eb",
            },
            "vegetation_surface_mapping": {
                "file": "rw-wps-hrrr-surface-vegetation-grib2.mapping.json",
                "sha256": "adefc1d166ab4932616d91e206542e527b9c38e9a7e4e72ba8e04a9404049f8a",
            },
        },
    ),
    # RAP's awip32 product (AWIPS grid 221, 32 km Lambert, all of North
    # America): the one public RAP product that carries the complete
    # state in a single file per valid time -- 39 pressure levels
    # (byte-identical ladder to HRRR wrfprs), surface/2 m/10 m fields,
    # in-band terrain, LAND/ICEC, and the nine-node RUC soil column.
    # Selectors were authored from real 2026-08-16 00Z bytes through the
    # converged grib-core inventory.  Pressure-level humidity is RH, so
    # the mapping reuses the GFS profile's declared derivation; no new
    # engine capability was needed -- HRRR's Lambert grid family, wind
    # rotation and node soil carry every RAP-specific fact as table data.
    "rap-awip32-grib2-v1": _profile(
        "rw-wps-rap-awip32-grib2",
        source_format="grib2",
        mapping="5bc58e43b2cb997c2946aa28b2211ffca0a778c5543642c45dda919a94158de0",
        composition="bae76db6052e933906713d877b9029079357a45426ab52387acee0d9a11f385f",
        provenance="9d861f738b9ba72661130b9b174d969c8e134fbf11e191183fccc092a43a6abb",
        data_role="rap_awip32_in_band_surface",
        provenance_role="rap_awip32_in_band_surface_provenance",
    ),
    # ECCC's GDPS (GEM global) 15 km regular lat-lon GRIB2 product on MSC
    # Datamart: 33 pressure levels, surface/2m/10m state, a single
    # 0-10 cm ISBA soil layer, and the once-per-cycle analysis
    # invariants (orography, land mask, ice analysis) declared through
    # the generic cycle-invariant/broadcast grammar.  Selectors were
    # authored from real 2026-08-16 00Z bytes through the converged
    # grib-core inventory.
    "gem-gdps-grib2-v1": _profile(
        "rw-wps-gem-gdps-grib2",
        source_format="grib2",
        mapping="093f2286700f539baa33b425e0d2a2f30a8622bbff3918f588c189b2c1c3f9fa",
        composition="13d7e4ca06f8012cfd252c03d70d9fae69d2e18b9b51975445d70a506e1d90ed",
        provenance="823e07b38677e3c0c83da984637a4fda83d1eb09be2401cb4e0a9e433820de22",
        data_role="gdps_analysis_invariant_surface",
        provenance_role="gdps_analysis_invariant_surface_provenance",
    ),
    # ECMWF's AIFS single deterministic forecast (open data, 0.25-degree
    # GDT-0 GRIB2): the reduced AI-model field set as TABLE DATA.  The
    # north-to-south row order, the geopotential-to-metres terrain scale
    # and the two-layer ordinal (type 151) soil column are rows in these
    # documents.  The land mask and surface geopotential ride the step-0
    # object alone (the f012 object carries neither), so the mapping
    # declares both composition_bound and the composition binds them from
    # the SAME cycle's step-0 object through a fourth pinned authority,
    # the step-0 donor mapping (this table under its own name, the land
    # mask without a time binding), on the source_cycle_analysis_broadcast
    # clock: every lead of every window reads the cycle's own statics,
    # and the route table fetches that object beside a window starting
    # later.  Selectors were authored from real 2026-08-17 00Z bytes
    # through the converged grib-core inventory.  Earlier AIFS cycles
    # publish pressure-level geopotential and no geopotential height, so
    # the height field lists those records second, scaled to metres.
    "aifs-single-grib2-v1": _profile(
        "rw-wps-aifs-single-grib2",
        source_format="grib2",
        mapping="4f30cce175e902799077f0d69905e25b6d6fca38ccdccf6fd067c07b6d624e42",
        composition="3e5d36143454b46c3fb8bf59d7ff7eba401d80606aa7fb65e99ef97e813a3426",
        provenance="26e16197183f4d2b6dffc9c732b9d18933ae26118b40d61617663a1355557f2c",
        data_role="aifs_single_in_band_surface",
        provenance_role="aifs_single_in_band_surface_provenance",
        contributing_mappings={
            "aifs_single_step0_donor_mapping": {
                "file": "rw-wps-aifs-single-step0-donor.mapping.json",
                "sha256": (
                    "99adc3a8894c2d19ba37d6da6d53418a95dcf32c2b1094b86"
                    "e8ccd1feecb7b02"
                ),
            },
            # Sea ice: AIFS publishes no sea-ice field at any lead, so
            # every start had no sea ice anywhere.  The composition binds
            # sea_ice_fraction to the same cycle's GFS analysis ICEC (the
            # route's declared gfs f000 donor), nearest 0.25-degree cell.
            "seaice_mapping": {
                "file": "rw-wps-earth2-gdas-donor.mapping.json",
                "sha256": "e24542b3a9747c37614451d29eb4047b5f17f6068e3e63c299a2cc5367d8d8a1",
            },
        },
        contributing_provenances={
            "seaice_provenance": {
                "file": "rw-wps-gfs-analysis-seaice.provenance.json",
                "sha256": "2ced4ae3b2d6f8974fd9a259b97bf18483f972b41e66ff1b112b297468b0af64",
            },
        },
    ),
    # ECMWF's open-data IFS oper product at 0.25 degrees: a plain global
    # GDT-0 latitude/longitude GRIB2 feed with 14 pressure levels,
    # in-band surface geopotential for terrain, and the four IFS soil
    # layers addressed by ordinal on fixed-surface type 151 (declared as
    # the composition's indexed selector_depth_binding).  Selectors were
    # authored from real 2026-08-16 00Z bytes through the converged
    # grib-core inventory.
    "ecmwf-open-data-oper-grib2-v1": _profile(
        "rw-wps-ecmwf-open-data-oper-grib2",
        source_format="grib2",
        mapping="e3637e86c9998a1e49197206a13b7e5338651ae70f8efd5ebd7c8a4dfcffb6fb",
        composition="cae4f4c02cb2a730bd8430891743220c4a7f9c036175b01584ee876151c89bd2",
        provenance="c574a07fb303620eab432eea89dad3e765ccd510d965f2b0734c4e278d435b8b",
        data_role="ecmwf_open_data_in_band_surface",
        provenance_role="ecmwf_open_data_in_band_surface_provenance",
        # Sea ice: ECMWF open data publishes thickness only, never a
        # concentration (0.25- and 0.1-degree indexes checked 2026-10-28),
        # so every IFS start had no sea ice anywhere.  The composition
        # binds sea_ice_fraction to the same cycle's GFS analysis ICEC (the
        # route's declared gfs f000 donor), nearest 0.25-degree cell.
        contributing_mappings={
            "seaice_mapping": {
                "file": "rw-wps-earth2-gdas-donor.mapping.json",
                "sha256": "e24542b3a9747c37614451d29eb4047b5f17f6068e3e63c299a2cc5367d8d8a1",
            },
        },
        contributing_provenances={
            "seaice_provenance": {
                "file": "rw-wps-gfs-analysis-seaice.provenance.json",
                "sha256": "2ced4ae3b2d6f8974fd9a259b97bf18483f972b41e66ff1b112b297468b0af64",
            },
        },
    ),
    # NCEP's GDAS analysis-cycle pgrb2 at 0.25 degrees: record-for-record
    # the GFS pgrb2 catalogue (696/696 at f000, MEASURED), arriving at a
    # different acquisition path with analysis-cycle semantics -- hourly
    # f000..f009 only, ~+7 h latency.  The four Noah soil layers are
    # bound by their scaled type-106 depth pairs (the integer level key
    # collides between the first two layers), soil moisture is an NCEP
    # local-table row selected by octets, and the accuracy facts ride the
    # provenance document: even f000 is stamped a forecast in the bytes,
    # and the one analysis-stamped product has no land surface at all.
    # Selectors were authored from real 2026-08-17 06Z bytes through the
    # converged grib-core inventory.
    "gdas-pgrb2-0p25-grib2-v1": _profile(
        "rw-wps-gdas-pgrb2-0p25-grib2",
        source_format="grib2",
        mapping="8df20187d7d953dce19875f0aaad215fbaa926134ee50610396864290a07b88e",
        composition="7d57188638a53ecf8771ce4779c7923015cee519b63318f6dcd0982798abf75d",
        provenance="13105c86f74d247d8795c075458ccc87bea1de5f663d53b805a1765aaa07c5a6",
        data_role="gdas_pgrb2_in_band_surface",
        provenance_role="gdas_pgrb2_in_band_surface_provenance",
    ),
    # NCEP's AIGFS (GraphCast-based 0.25-degree global AI forecast): the
    # barest product in the catalog, and the first ATMOSPHERE-ONLY
    # profile -- six 3-D fields on 13 pressure levels plus 2 m/10 m/MSLP
    # state, and NO land surface of any kind.  Its composition role is
    # therefore the explicit PENDING declaration (loading it refuses by
    # naming the missing state), and the profile decodes/inspects without
    # initializing until the cross-source land-surface donor lands.  The
    # acquisition identity is part of the profile: operational bytes are
    # NOMADS-only and carry subCentre 0, while an S3 bucket serves a
    # DIFFERENT experimental run under identical filenames with
    # subCentre 2 -- every selector pins subcenter=0 so the imposter
    # refuses by name.  Selectors were authored from real 2026-08-17 00Z
    # bytes through the converged grib-core inventory.
    "aigfs-nomads-grib2-v1": _profile(
        "rw-wps-aigfs-nomads-grib2",
        source_format="grib2",
        mapping="0bb2fd3721a3feebf08eab3340331c6833e61afbbf6ed18313ae1187d5d3fcb9",
        composition="39953c876827616ee0142a26729a2d9813d062ca9350500e0d232a901cd00b41",
        provenance="7199ecd0f94c06d8c1d98829ebf66f24a78905ccc6cf52643aeba6e2f2979fad",
        composition_state="pending_cross_source",
    ),
    # NVIDIA Earth-2 model output (FourCastNet 3, SFNO, Pangu, Aurora,
    # FengWu, GraphCast, AIFS through earth2studio) as Earth2Studio's
    # NetCDF4Backend writes it: one variable per field and level, (time,
    # lead_time, lat, lon), no units in the file (the mapping's
    # producer_attributes carries Earth2Studio's lexicon).  The land
    # surface these models do not publish is borrowed from the ERA5
    # analysis at the forecast's initialization (ARCO, keyless) through the
    # cross-source composition, as the AIGFS hybrid borrows it from GDAS.
    # Authored from woof/authorities/earth2-models.v1.json by
    # tools/author_earth2_profile.py; proven on real FCN3 bytes 2026-10-19.
    "earth2-stormcast-hrrr-netcdf-v1": _profile(
        "rw-wps-earth2-stormcast-hrrr-netcdf",
        source_format="netcdf",
        mapping="bd9435782a2f7e0315419a89f2b584ef9d663ee1bd4833fa02d5a0215a3f997c",
        composition="41c3cefb12cda90c50f19777a5e35238ae93eca19e5f32f1686d0214061b0dd3",
        provenance="c49ac5b2da4e02631d44d61bb210f63e992f8c243388937f57f2695e314598f1",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-earth2-hrrr-native-donor.mapping.json",
                "sha256": "d97b028807c5ac090a95674c10e508dd0598b30b82336f971c57b781d2f12634",
            },
        },
    ),
    "earth2-stormcast-conus-hrrr-netcdf-v1": _profile(
        "rw-wps-earth2-stormcast-conus-hrrr-netcdf",
        source_format="netcdf",
        mapping="e54a5a39bf69a38fdefdbc14afb3c3db45811d053e97f4ff75179cebd845a89d",
        composition="41c3cefb12cda90c50f19777a5e35238ae93eca19e5f32f1686d0214061b0dd3",
        provenance="20e1eeb5e9ec1bac254fb77e3ba670b8e961ba6d58ba6845c9e4bfe6af3d1847",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-earth2-hrrr-native-donor.mapping.json",
                "sha256": "d97b028807c5ac090a95674c10e508dd0598b30b82336f971c57b781d2f12634",
            },
        },
    ),
    "earth2-pl13-q-netcdf-v1": _profile(
        "rw-wps-earth2-pl13-q-netcdf",
        source_format="netcdf",
        mapping="814639a651cb44d861489c46c583921ccccccfac51a06cd1a3c390b8ef4441b2",
        composition="8b88da8dd45e6d2067babdd3706df8b3c11321fcc5d7ea02a479360b3564871b",
        provenance="ab7aa121392c15667befc84351431431c49604055705e711e70d5f3e89b1a7cb",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-era5-arco-netcdf-donor.mapping.json",
                "sha256": "fc6d7a02be6610e29b8f4da8c723c3ce32a55407a4fd70543ab81bf2c7034682",
            },
        },
    ),
    "earth2-sparse-r-netcdf-v1": _profile(
        "rw-wps-earth2-sparse-r-netcdf",
        source_format="netcdf",
        mapping="e7207f7f3819cb6cf6b0e312f85fdf17fbc7db628762fda0ac04aec3e077cfaa",
        composition="1b028d4da5d95189344c49b215b9fb303000767d7464a9f23af6cf1e5eb11cd6",
        provenance="0204d362fe49f9d0d2be9b8358f8245798654771a2a13f1cabc9ea57fd163b08",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-era5-arco-netcdf-donor.mapping.json",
                "sha256": "fc6d7a02be6610e29b8f4da8c723c3ce32a55407a4fd70543ab81bf2c7034682",
            },
        },
    ),
    "earth2-pl13-q-gfs-netcdf-v1": _profile(
        "rw-wps-earth2-pl13-q-gfs-netcdf",
        source_format="netcdf",
        mapping="7ce88cdbf032252329e166ef2a7c465de93a9dbdfab076d80fafd32bbc59e9f2",
        composition="3f5ce2527fb36a2d85db1d0f1d46a55c81e6dcca01a0ec3aff7fa2e533313b11",
        provenance="80777b61866d834251b9ca7985f45ea828b2c2b2100167b189048f2202173ddf",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-earth2-gdas-donor.mapping.json",
                "sha256": "e24542b3a9747c37614451d29eb4047b5f17f6068e3e63c299a2cc5367d8d8a1",
            },
        },
    ),
    "earth2-pl13-q-ifs-netcdf-v1": _profile(
        "rw-wps-earth2-pl13-q-ifs-netcdf",
        source_format="netcdf",
        mapping="3880baab923f8ca07bbb059795a086142fd76f26a5258e9e96f2b241061ef8d6",
        composition="4ceec22d014ece8c8dc663e1f977ccc7ca193223ae225371ed033e71a84fac8b",
        provenance="72e860df39577f20a196ee02b48ecd0ca5c802522af4f1ab3bc7b4aee8bd03cc",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-earth2-ifs-donor.mapping.json",
                "sha256": "ac39bd4c9a59a5bc2e7fded1863d70c8bcb4e163367c6b187f62ca8bbc5cbb04",
            },
        },
    ),
    "earth2-sparse-r-gfs-netcdf-v1": _profile(
        "rw-wps-earth2-sparse-r-gfs-netcdf",
        source_format="netcdf",
        mapping="36bbacf994db8edbea01b91d1c4532a9891837e88acf55ffdd48cd262bf5f56a",
        composition="f8ca19754a1e5cec8f5878975df6cccba3ed1a1e175ff8c7a8560cf27e45c933",
        provenance="2e29f28ea798f1e67e334ba0e29f43875bb3b2097391ab463485573f0a21ade1",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-earth2-gdas-donor.mapping.json",
                "sha256": "e24542b3a9747c37614451d29eb4047b5f17f6068e3e63c299a2cc5367d8d8a1",
            },
        },
    ),
    "earth2-sparse-r-ifs-netcdf-v1": _profile(
        "rw-wps-earth2-sparse-r-ifs-netcdf",
        source_format="netcdf",
        mapping="baf137c563208c250ee9f7aa13d9fdc1519830e668dd47afa4a65c17a8fddbfc",
        composition="9fd1eae2df0c6ec3da2c4dfe51228816ab0450ca50f5e52b00bfcd22b4fcfa92",
        provenance="cbb4079023cbb7e35a8eb17c0b9814ec76e132d37c0e62b7c85afae8ce3d53d6",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-earth2-ifs-donor.mapping.json",
                "sha256": "ac39bd4c9a59a5bc2e7fded1863d70c8bcb4e163367c6b187f62ca8bbc5cbb04",
            },
        },
    ),
    "earth2-pl13-rh-netcdf-v1": _profile(
        "rw-wps-earth2-pl13-rh-netcdf",
        source_format="netcdf",
        mapping="abceb83c784e6749fd650af10463b2b23bee32f86b1b5829d71fd60fae9c773d",
        composition="8b88da8dd45e6d2067babdd3706df8b3c11321fcc5d7ea02a479360b3564871b",
        provenance="e7e4ead6b5e269ebbba8682461f37815bffe9b4804607af1526b242006fe65d9",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-era5-arco-netcdf-donor.mapping.json",
                "sha256": "fc6d7a02be6610e29b8f4da8c723c3ce32a55407a4fd70543ab81bf2c7034682",
            },
        },
    ),
    "earth2-pl13-rh-gfs-netcdf-v1": _profile(
        "rw-wps-earth2-pl13-rh-gfs-netcdf",
        source_format="netcdf",
        mapping="3113c80d149eec08dd3747a2955bd281c2746fbe394da41d584356253c458dad",
        composition="3f5ce2527fb36a2d85db1d0f1d46a55c81e6dcca01a0ec3aff7fa2e533313b11",
        provenance="3df59fd76ffcac01aa98d431fa77cd591b951012b9e9856f28af9bdc5db92e7e",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-earth2-gdas-donor.mapping.json",
                "sha256": "e24542b3a9747c37614451d29eb4047b5f17f6068e3e63c299a2cc5367d8d8a1",
            },
        },
    ),
    "earth2-pl13-rh-ifs-netcdf-v1": _profile(
        "rw-wps-earth2-pl13-rh-ifs-netcdf",
        source_format="netcdf",
        mapping="129661c93e1ba1017ca3030db89551b8d7144b4dd4d1fbd5a4fd6eacb55c5685",
        composition="4ceec22d014ece8c8dc663e1f977ccc7ca193223ae225371ed033e71a84fac8b",
        provenance="1dd4542a617245af8640da504f5a9a31d2d10bc72fbd9834aeaa71923e36c67e",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-earth2-ifs-donor.mapping.json",
                "sha256": "ac39bd4c9a59a5bc2e7fded1863d70c8bcb4e163367c6b187f62ca8bbc5cbb04",
            },
        },
    ),
    # The AIGFS HYBRID: the same operational NOMADS atmosphere (every
    # selector still pins subcenter=0 against the S3/EAGLE imposter) made
    # RUNNABLE by borrowing the six canonicals AIGFS does not publish --
    # terrain, land mask, skin temperature, 2 m humidity, and the
    # four-layer soil column -- from the SAME CYCLE's GDAS 0.25-degree
    # analysis through the cross-source composition (field_sources,
    # source_cycle_analysis_broadcast clock).  Surface pressure, which
    # AIGFS does not publish either, is derived at every lead from its own
    # mean-sea-level pressure at the analysis terrain height, so the
    # column mass follows the forecast instead of holding the analysis
    # value.  The donor
    # decodes through its own mapping, shipped here as a fourth pinned
    # authority: the checked-in GFS pressure-level table with one
    # table-data change (2 m specific humidity DIRECTLY selected, because
    # a borrowed field must be directly selected in the donor).  Proven on
    # real 2026-08-17 00Z NOMADS + GDAS bytes.
    "aigfs-gdas-hybrid-grib2-v1": _profile(
        "rw-wps-aigfs-gdas-hybrid-grib2",
        source_format="grib2",
        mapping="64cab09660beffc3b4cd35a8fc7a711e35c3034bfe930362c72a3ef68581a320",
        composition="e11db42390a6b1f08b2edcd7d25e4574dc5be74021748e65c3c7b1115c34d07c",
        provenance="72299bca17576bce3a0bedfe846806dbde31d126781d4d3ebaafb9dbf78318ed",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-gdas-pgrb2-donor.mapping.json",
                "sha256": (
                    "35e0d4e2895b38a2702952fabc97fb9f53594b41dbb2d511"
                    "e39a13458f72f8cf"
                ),
            },
        },
    ),
    # One MEMBER of NCEP's GEFS v12 through the generic mapped route:
    # the 0.5-degree pgrb2a + pgrb2b pair of a single member.  The two
    # products' isobaric level sets are exactly disjoint (measured zero
    # overlap on every variable), so the mapping's 31-level ladder is
    # only satisfiable by both files of the SAME member; every selector
    # pins PDT 1, which is what refuses the geavg/gespr statistic files
    # (PDT 2) and the PDT-11 accumulation twins at the byte level.  The
    # four Noah soil layers split across the pair (0-0.1 m in pgrb2a,
    # the rest in pgrb2b) under the measured 66-percent ocean bitmap;
    # terrain is in band but migrates products (analysis: pgrb2a;
    # forecast steps: pgrb2b -- measured), so the terrain supplement is
    # the same a+b pair per valid time, proven invariant across the
    # window.  Which members exist and how their
    # bytes verify is the sibling members grammar below -- this profile
    # answers only how a verified member's FIELDS decode.  Selectors
    # were authored from real 2026-08-17 00Z bytes through the converged
    # grib-core inventory.
    "gefs-v11-flat-grib2-v1": _profile(
        "rw-wps-gefs-v11-flat-grib2",
        source_format="grib2",
        mapping="dd6a686fbcc739ec30369b377abb4148a1fdc742145d928a2ca4f9c6f740b6c8",
        composition="20b454f08f5a6a1995e9c3051cfd75c0f7a87948038794a7df5df9c96b554658",
        provenance="8659bf35897a1c35a01fce996f5ce5c5682701c2198944485bce3947baf03dd0",
        data_role="gefs_member_in_band_surface",
        provenance_role="gefs_member_in_band_surface_provenance",
    ),
    "gefs-v11-grib2-v1": _profile(
        "rw-wps-gefs-v11-grib2",
        source_format="grib2",
        mapping="ff6560082eaec715934a35fecc8d84be6226d62f012a70f03ffe2fa61a6ac599",
        composition="65fa7756915f178d96ac67455f381caba3d466c2b33efe86eaeedf54d542a4fc",
        provenance="815a4bee91429d90d29a1a172f9e7e2e1479e7538cb3fa5e5c2fde0eb1b69739",
        data_role="gefs_member_in_band_surface",
        provenance_role="gefs_member_in_band_surface_provenance",
    ),
    "gefs-ensemble-grib2-v1": _profile(
        "rw-wps-gefs-ensemble-grib2",
        source_format="grib2",
        mapping="2445d2a0b985c24096a729e5f7b443613d5da579828627fba239356c6efbf668",
        composition="f7db1f0399456334c86b4aad0a1591a3e1faf38fba190d7a439ce96fecebf529",
        provenance="5f7644c1cb68f06a347602d60b63443170c33973efa98e530c254a463f3fe154",
        data_role="gefs_member_in_band_surface",
        provenance_role="gefs_member_in_band_surface_provenance",
    ),
    # DWD open data's ICON-EU regular-lat-lon product set through the
    # generic mapped route: field-per-file bz2 GRIB2 objects, the 20-level
    # pressure ladder, once-per-cycle invariant FR_LAND/HSURF, and the
    # TERRA soil column (temperature at nine depth nodes, water as
    # column-integrated mass over the eight layers whose midpoints are
    # those interior nodes).  Selectors were authored from real
    # 2026-08-17 00Z bytes through the converged grib-core inventory.
    "icon-eu-regular-grib2-v1": _profile(
        "rw-wps-icon-eu-regular-grib2",
        source_format="grib2",
        mapping="ae8075c069206c4f3917830aa769a2c4ef8f39a03ae31b3d7ee8c31c63f23cea",
        composition="6220800aa224b2ef8ae40d899760d8ed9100d0ce69a44fa06a80e039bdc74b2b",
        provenance="79ed94eee82ef0dc90b0e5b4ea79ce437f9006aa80d873bc297542002421fa57",
        data_role="icon_eu_invariant_surface",
        provenance_role="icon_eu_invariant_surface_provenance",
    ),
    # NCEP's operational AI global ensemble member state (Project EAGLE,
    # 0.25-degree GDT-0 GRIB2, 13 pressure levels), completed by a
    # same-cycle physical analysis through the CROSS-SOURCE composition:
    # the member product publishes no soil, no land mask, no orography,
    # no skin temperature and no 2 m humidity, so the mapping declares
    # those six gaps composition_bound and the composition binds them to
    # the packaged donor mapping under source_cycle_analysis_broadcast.
    # Surface pressure is derived at every lead from the member's own
    # mean-sea-level pressure at the analysis terrain height: the NOMADS
    # sfc product carries no surface pressure, and the record the AWS
    # mirror appends sits on the AI model's own orography, not on the
    # terrain the composition pairs it with.
    # Every selector pins PDT 1 -- the individual-member template -- so
    # deterministic bytes and ensemble statistics refuse at the mapped
    # decode itself.  Member identity is verified separately by the
    # packaged rw-wps.members.v1 grammar (`woof-member-prep`).
    # Selectors were authored from real 2026-08-17 00Z bytes through the
    # extended grib-core inventory.
    "aigefs-member-hybrid-grib2-v1": _profile(
        "rw-wps-aigefs-member-hybrid-grib2",
        source_format="grib2",
        mapping="d34ff39693af3a6cef6d4f3d83a72c7bff1d710470518d986724eae368bde623",
        composition="9965904c92f08cd71323e2565d7a663863c8ca061f3e991b813b52ad2a6a5d10",
        provenance="4050afea573ffa3b590deb6ef0f0e2a86f2d6feca25423b38cd31f4e025b6bb2",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-gdas-pgrb2-donor.mapping.json",
                "sha256": (
                    "35e0d4e2895b38a2702952fabc97fb9f53594b41dbb2d511"
                    "e39a13458f72f8cf"
                ),
            },
        },
    ),
    # RRFS -- HRRR's operational successor, flowing today on the
    # noaa-rrfs-ops-pds bucket and NOMADS rrfs/v1.0.  The 3 km CONUS grid
    # is bit-for-bit HRRR's Lambert (measured from real bytes: every
    # geolocating octet identical), so the entire HRRR wrfprs machinery
    # carries it as table data.  What is RRFS's own is also table data:
    # the 45-level pressure ladder (2 hPa top, 70 hPa where HRRR has 75,
    # no 1013.2 hPa entry) and the split of the state across a
    # prslev/2dfld file PAIR -- prslev is pure upper air, so the terrain
    # supplement and every surface/soil selector resolve in the 2dfld
    # files, which the caller passes alongside.  Selectors were authored
    # from real prototype-cycle bytes (2026-08-12 00Z) and cross-checked
    # against the live operational cycle (2026-08-17 00Z) through the
    # converged grib-core inventory: 19 of 19 fields, exactly one record
    # each per valid time, pair disjoint.
    "rrfs-prslev-2dfld-grib2-v1": _profile(
        "rw-wps-rrfs-prslev-2dfld-grib2",
        source_format="grib2",
        mapping="6b89a9d1eed5ad8868505e5560301599736837340ca2fb8d87dd4fdf872d68cb",
        composition="a632eae5203e92eeb4f6af5ee1fcfe2f24c1e03c7a5124f73a669fbb4e1ca6b1",
        provenance="e364f22ec917bfd991ec3bb756aa49b697d03c7cd425d40a35501dd769f279de",
        data_role="rrfs_prslev_2dfld_in_band_surface",
        provenance_role="rrfs_prslev_2dfld_in_band_surface_provenance",
    ),
    # One RRFS ensemble MEMBER from the public rrfsens subset, completed by
    # its DECLARED COMPANION: the deterministic prslev+2dfld pair of the
    # same cycle, decoded through the deterministic RRFS mapping above as
    # a pinned contributing authority.  The member subset stops at 250 hPa
    # and has no soil, skin temperature or land mask.  Every member field
    # pins product definition template 1, so the member's own decode never
    # mixes runs; the composition's companion row stacks the companion's
    # 14 levels above 250 hPa onto the member's columns and lends the land
    # state through the ordinary field_sources grammar.  The composition
    # receipt names which levels and fields came from which run.  Without
    # the companion's files there is no soil state and the model top sits
    # above the data, and the front door refuses with the row's words.
    "rrfs-ens-member-companion-grib2-v1": _profile(
        "rw-wps-rrfs-ens-member-companion-grib2",
        source_format="grib2",
        mapping="d7386655a10547a8e95a4a44e1edae7d9258e668258a5a19b70657be58ce6644",
        composition="242d529a511e21bbb2a62195a49320b1547fb4f606cd03e58d0af5d4a776b7b6",
        provenance="e1c3997fb2b0e24cfcc671347d3cd6aea68af7f39523e5bc86040a87102649e9",
        data_role="rrfs_ens_member_in_band_surface",
        provenance_role="rrfs_ens_member_in_band_surface_provenance",
        contributing_provenances={
            "rrfs_ens_companion_provenance": {
                "file": "rw-wps-rrfs-prslev-2dfld-grib2.provenance.json",
                "sha256": "e364f22ec917bfd991ec3bb756aa49b697d03c7cd425d40a35501dd769f279de",
            },
        },
        contributing_mappings={
            "rrfs_ens_companion_mapping": {
                "file": "rw-wps-rrfs-prslev-2dfld-grib2.mapping.json",
                "sha256": "6b89a9d1eed5ad8868505e5560301599736837340ca2fb8d87dd4fdf872d68cb",
            },
        },
    ),
    # ERA5 on its NATIVE 137 hybrid sigma-pressure model levels -- the
    # first shipped profile whose vertical coordinate is not a pressure
    # ladder.  The A/B interface coefficients ride IN BAND as the GRIB2
    # Section-4 coordinate-values (pv) octets, so nothing about this
    # source needs a per-model channel: the decode reads the 276 values
    # the producer wrote, materializes p = A + B*ps, and integrates
    # geopotential height hydrostatically from the borrowed terrain.
    # The model-level product publishes the prognostic atmosphere and
    # NOTHING else -- no soil, no land mask, no orography, no skin
    # temperature, no 2 m or 10 m diagnostics, and surface pressure only
    # as lnsp (which affine-only unit transforms cannot exponentiate) --
    # so the same-hour ERA5 pressure-level/single-level analysis is the
    # cross-source donor for all ten of those fields, decoded through
    # its own pinned mapping.  It also lends the water state the
    # pressure-level route reads from the same file -- sea surface
    # temperature, sea ice and the lake model's water and ice -- so the
    # two routes assemble the same water temperature from one donor
    # instead of this one giving every lake its skin temperature, and
    # the snow water equivalent, which this route otherwise started at
    # zero everywhere.
    # Proven on real 2026-05-30 CDS bytes: both
    # engines byte-identical on air_pressure and geopotential_height
    # (max ULP 0) and a preparation to rc 0.
    "era5-model-level-l137-grib2-v1": _profile(
        "rw-wps-era5-model-level-l137-grib2",
        source_format="grib2",
        mapping="65cfca7fc9fc6be102c796bd481126fe9e22491df8342fbaa1ff54b087fb09c6",
        composition="eab1fcd5c10a746ebdaf269e910e08ea77a40d0db9789539423cbda477201e62",
        provenance="3c6477f94c1ab3428c5f3f1d6fb29c57ca9620dfecf41d600ca165df8609da7c",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-era5-plev-surface-donor.mapping.json",
                "sha256": (
                    "a16c6a9b615d5999030051aa3fb2170de7993467580c4f44"
                    "840eebd0f1ec5e69"
                ),
            },
        },
    ),
})


#: Packaged ``rw-wps.members.v1`` documents: the ensemble member-
#: addressing grammars.  A members document is table data of the same
#: standing as a profile -- an ensemble source's member set, filename
#: patterns, byte-verification contract and statistic namespace are
#: rows in one JSON, pinned here by SHA-256, and adding an ensemble is
#: one document plus one row.  It is a separate table from
#: :data:`_PACKAGED_PROFILES` because the two answer different
#: questions (how to decode fields vs. which trajectory a file is), and
#: an ensemble can have a members grammar before its field mapping
#: exists -- which is exactly the state GEFS and AIGEFS ship in.
_PACKAGED_MEMBER_GRAMMARS = MappingProxyType({
    "rrfs-ops-subset-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-rrfs-ops-subset-grib2.members.json",
        "sha256": "2924d472498e298b01cf3a34fe876a950803b1007be8a22e7b639c13a82ad217",
    }),
    "ecmwf-ens-open-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-ecmwf-ens-open-grib2.members.json",
        "sha256": "f5e3103d949b1851171bdad56e19203c2716170fa77c5003728324cda7104167",
    }),
    # NCEP GEFS v12, the 0.5-degree atmos pgrb2a/b (and 0.25-degree
    # pgrb2s) member files: 31 forecasts whose encoded ensemble size
    # says 30 (the octet excludes the control), whose control is flagged
    # low-resolution control (type 1), and whose mean/spread files
    # (geavg/gespr) share the member directories and filename pattern.
    # Every declared value was measured from real 2026-08-17 00Z bytes
    # through the extended grib-core inventory.
    "gefs-v11-flat-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-gefs-v11-flat-grib2.members.json",
        "sha256": "c6eb2d709eb1be7641c41cedfeb43c2441918abd980ecde1931ff3882fc0cd5f",
    }),
    "gefs-reforecast-grib2-members-v1": MappingProxyType({'file': 'rw-wps-gefs-reforecast-grib2.members.json', 'sha256': 'fe51794a16335247b4b177fe6901ff5b8f66b94ebbde1e716407351a2ebaca10'}),
    "gefs-reforecast-weekly-grib2-members-v1": MappingProxyType({'file': 'rw-wps-gefs-reforecast-weekly-grib2.members.json', 'sha256': 'cf8668dff0ca704ac0382a24990212b91c236160a550a392d7c1d430203b3bd1'}),
    "gefs-v11-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-gefs-v11-grib2.members.json",
        "sha256": "cd18697dffe60a37ec7cd968460eb51595286c63e6add20f465076be334aa593",
    }),
    "gefs-ensemble-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-gefs-ensemble-grib2.members.json",
        "sha256": (
            "7342f58bb0c01c5c8a6b051a7d7619245a2f48743e3760cbfcff64de28f6ca7c"
        ),
    }),
    # NCEP AIGEFS (Project EAGLE), the operational AI global ensemble:
    # 31 members whose member identity is a PATH component (every leaf
    # filename is byte-identical across members), whose encoded ensemble
    # size says 31 (control included -- the opposite octet convention
    # from GEFS), and whose control carries NO control flag
    # (typeOfEnsembleForecast like every perturbed member: 6 as NOMADS
    # serves it, 3 in the AWS mirror's rewritten copy;
    # perturbationNumber == 0 is the only discriminator), and whose
    # mirror serves part of its archive under another writer's octets
    # (declared per class as rewrites).  Every declared value was
    # measured from real bytes through the extended grib-core inventory.
    "aigefs-ensemble-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-aigefs-ensemble-grib2.members.json",
        "sha256": (
            "7d8237d0ad3d5010380527beaeea9ea8f68c3b4154a37bb4394dc0e71d21795c"
        ),
    }),
})


def packaged_member_grammar_ids() -> tuple[str, ...]:
    """Every packaged members document this distribution ships, sorted."""

    return tuple(sorted(_PACKAGED_MEMBER_GRAMMARS))


def packaged_member_grammar_sha256(grammar_id: str) -> str:
    """One members document's immutable SHA-256, without touching disk."""

    return str(_member_grammar_row(grammar_id)["sha256"])


def _member_grammar_row(grammar_id: str) -> Mapping[str, object]:
    try:
        return _PACKAGED_MEMBER_GRAMMARS[grammar_id]
    except KeyError:
        raise KeyError(
            f"unknown packaged member grammar {grammar_id!r}; this "
            f"distribution ships {sorted(_PACKAGED_MEMBER_GRAMMARS)}"
        ) from None


def packaged_member_grammar(grammar_id: str) -> Path:
    """Resolve and byte-verify one packaged members document."""

    row = _member_grammar_row(grammar_id)
    path = (_AUTHORITY_ROOT / str(row["file"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged member grammar {grammar_id} is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != row["sha256"]:
        raise RuntimeError(
            f"packaged member grammar {grammar_id} hash differs: "
            f"expected {row['sha256']}, got {observed}")
    return path


#: The physical field contract of a NATIVE preparation implementation: the
#: units, target-grid dimensions, vector basis and source operation of every
#: array its physical store carries, its vertical coordinate, the evidence
#: it must be given, and the sentences it refuses with.  A mapped source
#: needs no row here, because its contract is read from its own packaged
#: mapping.  These two were Python tables in one module per source; they
#: are documents now, pinned by SHA-256 like every other authority and read
#: by one module (:mod:`woof.ensemble.physical_fields`), so an
#: implementation with the same capabilities is one document plus one row.
_PACKAGED_PHYSICAL_CONTRACTS = MappingProxyType({
    "gfs-pgrb2-0p25-physical-fields-v1": MappingProxyType({
        "file": "rw-wps-gfs-pgrb2-0p25.physical-fields.json",
        "sha256": "93732b11f63ad5446569f867b97c71d77fe4a3302dca912fd52e383b4a0431a3",
    }),
    "hrrr-f00-f12-physical-fields-v1": MappingProxyType({
        "file": "rw-wps-hrrr-f00-f12.physical-fields.json",
        "sha256": "80358e88924cbf4920a4a7f85cd7ce7b852be8749e77218c4a9c26c3eb4be8e9",
    }),
})


def packaged_physical_contract_ids() -> tuple[str, ...]:
    """Every packaged physical field contract this distribution ships, sorted."""

    return tuple(sorted(_PACKAGED_PHYSICAL_CONTRACTS))


def _physical_contract_row(contract_id: str) -> Mapping[str, object]:
    try:
        return _PACKAGED_PHYSICAL_CONTRACTS[contract_id]
    except KeyError:
        raise KeyError(
            f"unknown packaged physical field contract {contract_id!r}; this "
            f"distribution ships {sorted(_PACKAGED_PHYSICAL_CONTRACTS)}"
        ) from None


def packaged_physical_contract_sha256(contract_id: str) -> str:
    """One physical field contract's immutable SHA-256, without touching disk."""

    return str(_physical_contract_row(contract_id)["sha256"])


def packaged_physical_contract(contract_id: str) -> Path:
    """Resolve and byte-verify one packaged physical field contract."""

    row = _physical_contract_row(contract_id)
    path = (_AUTHORITY_ROOT / str(row["file"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged physical field contract {contract_id} is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != row["sha256"]:
        raise RuntimeError(
            f"packaged physical field contract {contract_id} hash differs: "
            f"expected {row['sha256']}, got {observed}")
    return path


def packaged_normalizer_ids() -> tuple[str, ...]:
    """Every input normalizer this distribution ships, sorted."""

    return tuple(sorted(
        str(row["input_normalizer"]) for row in _PACKAGED_PROFILES.values()
        if row.get("input_normalizer")))


def _normalizer_profile(name: str) -> str:
    for profile_id, row in _PACKAGED_PROFILES.items():
        if row.get("input_normalizer") == name:
            return profile_id
    raise KeyError(
        f"unknown packaged input normalizer {name!r}; this distribution "
        f"ships {list(packaged_normalizer_ids())}"
    )


def packaged_normalization(name: str) -> Path:
    """Resolve and byte-verify one packaged normalization document.

    The argument is the NORMALIZER NAME a profile declares, not a path: a
    name this distribution does not ship never becomes a file read.
    """

    profile_id = _normalizer_profile(name)
    profile = _PACKAGED_PROFILES[profile_id]
    file_name = profile["files"]["normalization"]      # type: ignore[index]
    expected = profile["sha256"]["normalization"]      # type: ignore[index]
    path = (_AUTHORITY_ROOT / str(file_name)).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged {profile_id} normalization authority is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(
            f"packaged {profile_id} normalization authority hash differs: "
            f"expected {expected}, got {observed}")
    return path


def packaged_profile(profile_id: str) -> Mapping[str, object]:
    """The declaration for one packaged profile, or a useful refusal."""

    try:
        return _PACKAGED_PROFILES[profile_id]
    except KeyError:
        raise KeyError(
            f"unknown packaged source profile {profile_id!r}; this "
            f"distribution ships {sorted(_PACKAGED_PROFILES)}"
        ) from None


def packaged_profile_ids() -> tuple[str, ...]:
    """Every packaged profile this distribution ships, sorted."""

    return tuple(sorted(_PACKAGED_PROFILES))


def packaged_authorities(profile_id: str) -> Mapping[str, Path]:
    """Resolve and byte-verify one packaged profile's three authorities."""

    profile = packaged_profile(profile_id)
    names: Mapping[str, str] = profile["files"]        # type: ignore[assignment]
    expected: Mapping[str, str] = profile["sha256"]    # type: ignore[assignment]
    resolved: dict[str, Path] = {}
    for role in PROFILE_ROLES:
        path = (_AUTHORITY_ROOT / names[role]).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"packaged {profile_id} {role} authority is missing: {path}"
            )
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected[role]:
            raise RuntimeError(
                f"packaged {profile_id} {role} authority hash differs: "
                f"expected {expected[role]}, got {observed}"
            )
        resolved[role] = path
    return MappingProxyType(resolved)


#: Parsing is cached by the verified bytes, never by mutable filesystem times.
_COMPOSITIONS: dict[str, tuple[str, Mapping[str, object]]] = {}


def packaged_composition(profile_id: str) -> Mapping[str, object]:
    """Verify every authority and parse the exact verified composition bytes."""
    profile = packaged_profile(profile_id)
    names = profile["files"]
    expected = profile["sha256"]
    composition = None
    for role in PROFILE_ROLES:
        path = (_AUTHORITY_ROOT / names[role]).resolve()
        data = path.read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if observed != expected[role]:
            raise RuntimeError(f"packaged {profile_id} {role} authority hash differs: "
                               f"expected {expected[role]}, got {observed}")
        if role == "composition":
            composition = data
    digest = hashlib.sha256(composition).hexdigest()
    cached = _COMPOSITIONS.get(profile_id)
    if cached is None or cached[0] != digest:
        _COMPOSITIONS[profile_id] = (digest, MappingProxyType(json.loads(composition)))
    return _COMPOSITIONS[profile_id][1]


def packaged_contributing_mappings(profile_id: str) -> Mapping[str, Path]:
    """Resolve and byte-verify one profile's contributing mapping documents.

    Empty for a profile whose composition declares no ``field_sources``
    bindings.  Each returned path is the packaged donor mapping the front
    door must pass as ``--contributing-mapping role=path``; the bytes are
    verified against the profile pin here, and the composition's own
    pinned digest re-verifies them at decode.
    """

    profile = packaged_profile(profile_id)
    declared: Mapping[str, Mapping[str, str]] = (
        profile["contributing_mappings"])    # type: ignore[assignment]
    resolved: dict[str, Path] = {}
    for role, pin in declared.items():
        path = (_AUTHORITY_ROOT / pin["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"packaged {profile_id} contributing mapping {role!r} is "
                f"missing: {path}"
            )
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != pin["sha256"]:
            raise RuntimeError(
                f"packaged {profile_id} contributing mapping {role!r} hash "
                f"differs: expected {pin['sha256']}, got {observed}"
            )
        resolved[role] = path
    return MappingProxyType(resolved)


def packaged_provenance_files(profile_id: str) -> Mapping[str, Path]:
    """The pinned provenance document of every composition role.

    A contributor can carry its own evidence without aliasing another
    authority file. Undeclared contributors retain the primary evidence.
    """
    profile = packaged_profile(profile_id)
    primary = packaged_authorities(profile_id)["provenance"]
    composition = packaged_composition(profile_id)
    roles = {str(profile["provenance_role"]), *(
        str(binding["provenance_role"])
        for binding in composition.get("field_sources", {}).values())}
    declared = profile.get("contributing_provenances", {})
    if set(declared) - roles:
        raise ValueError("packaged contributing provenance names a role the composition does not use")
    result = {}
    for role in sorted(roles):
        pin = declared.get(role)
        if pin is None:
            result[role] = primary
            continue
        path = (_AUTHORITY_ROOT / pin["file"]).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
            raise RuntimeError(
                f"packaged {profile_id} provenance {role!r} hash differs; "
                "other evidence must not be attributed to the contributor")
        result[role] = path
    return MappingProxyType(result)


def packaged_authority_sha256(profile_id: str) -> Mapping[str, str]:
    """One profile's immutable SHA-256 contract, without touching disk."""

    return packaged_profile(profile_id)["sha256"]   # type: ignore[return-value]


def packaged_contributing_sha256(profile_id: str) -> Mapping[str, str]:
    """The contributing-mapping pin table, without touching the filesystem."""

    profile = packaged_profile(profile_id)
    declared: Mapping[str, Mapping[str, str]] = (
        profile["contributing_mappings"])       # type: ignore[assignment]
    return MappingProxyType({
        role: str(row["sha256"]) for role, row in declared.items()
    })


def packaged_contributing_provenance_sha256(profile_id: str) -> Mapping[str, str]:
    """The contributor provenance pin table, without touching the filesystem.

    Keyed by composition provenance role: exactly the roles whose own
    document :func:`woof.initial_source.decode_initial_analysis` retains as
    ``<role>.json`` evidence beside the primary provenance.  Empty for a
    profile whose contributors all carry the primary provenance document.
    """

    profile = packaged_profile(profile_id)
    primary = profile["files"]["provenance"]            # type: ignore[index]
    declared: Mapping[str, Mapping[str, str]] = profile.get(
        "contributing_provenances", {})                 # type: ignore[assignment]
    return MappingProxyType({
        role: str(row["sha256"]) for role, row in declared.items()
        if row["file"] != primary
    })


def twentycrv3_authorities() -> Mapping[str, Path]:
    """Resolve and byte-verify the exact packaged 20CRv3 GRIB2 authorities."""

    return packaged_authorities("20crv3-member-grib2-v1")


def twentycrv3_authority_sha256() -> Mapping[str, str]:
    """Return the immutable SHA-256 contract without touching the filesystem."""

    return packaged_authority_sha256("20crv3-member-grib2-v1")


#: The GFS WPS Vtable `woof adapt` documents as its worked example.
#:
#: It lived in `configs/`, which is not a package, so the wheel did not
#: carry it and a pip user following the documented adapt flow was told
#: to pass a file their install did not have.  It ships beside the
#: 20CRv3 authorities now, under the same recursive package-data glob
#: and the same byte contract, because it is the same kind of thing: an
#: immutable input a front door reads, not a config anyone edits.
_GFS_VTABLE_NAME = "Vtable.GFS.rw-wps"
#: Re-pinned 2026-07-30 to the committed bytes: 9e391880... -> ec8e615b...
#:
#: The old value was this file's CRLF form.  It is not a JSON file, so
#: the per-path ``woof/authorities/*.json text eol=lf`` rule did not
#: cover it, and on a Windows clone (git-for-Windows defaults
#: core.autocrlf=true) it materialized with CRLF -- which is the
#: checkout the constant was taken from, and the wheel that was built
#: from it.  The same file on Linux, and in the object database, is LF,
#: so this gate could only ever have been true on one platform: the
#: byte contract it exists to enforce was itself platform-dependent.
#:
#: The repository now declares ``* -text``, so every checkout gets the
#: committed bytes and this is the one hash for all of them.  Nothing
#: was widened: the file is unchanged, the check is unchanged, and the
#: constant now names what the file actually is.
_GFS_VTABLE_SHA256 = (
    "ec8e615ba724b3ddf114c4c199a81083b3a17b4e1705055ec016f1769144090e")


def packaged_gfs_vtable() -> Path:
    """Resolve and byte-verify the packaged GFS WPS Vtable."""

    path = (_AUTHORITY_ROOT / _GFS_VTABLE_NAME).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged GFS Vtable is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != _GFS_VTABLE_SHA256:
        raise RuntimeError(
            f"packaged GFS Vtable hash differs: expected "
            f"{_GFS_VTABLE_SHA256}, got {observed}")
    return path


def packaged_gfs_vtable_sha256() -> str:
    """The immutable SHA-256 contract, without touching the filesystem."""

    return _GFS_VTABLE_SHA256


#: The mapping target key that widens ``boundary_interval_seconds`` from
#: the one spacing a series must have to the finest one it may have.
BOUNDARY_MULTIPLES_KEY = "accept_boundary_interval_multiples"


def boundary_interval_refusal(target: Mapping[str, object], seconds: int,
                              *, subject: str = "mapped cadence") -> str | None:
    """Why a mapping target refuses a uniform boundary interval, or None.

    ``boundary_interval_seconds`` names the publisher's own spacing.  A
    target that also declares :data:`BOUNDARY_MULTIPLES_KEY` takes a
    uniform series at any whole multiple of that spacing: every valid
    time in such a series is one the publisher wrote, and no mapped field
    depends on the spacing, because the mapping grammar has no
    time-window statistic to bind one to.  Without the key the declared
    spacing is the only one the target takes, which is what every
    mapping that does not declare it keeps.

    ONE function for the decode's frame check, the direct export's
    contract check and the doors that write a cadence, so a cadence one
    of them accepts cannot be refused by another.
    """

    declared = target.get("boundary_interval_seconds")
    if target.get(BOUNDARY_MULTIPLES_KEY) is True:
        declared = int(declared)
        if seconds > 0 and seconds % declared == 0:
            return None
        return (f"{subject} {seconds} seconds is not a whole multiple of "
                f"the {declared} seconds the target contract declares")
    if declared is not None and int(declared) == seconds:
        return None
    return f"{subject} {seconds} seconds differs from target contract {declared!r}"


def boundary_interval_takes(target: Mapping[str, object]) -> str:
    """The spacings a mapping target takes, as a refusal sentence says them.

    Read by every door that refuses a cadence on a preparation's behalf,
    so each says what :func:`boundary_interval_refusal` decides.
    """

    spacing_h = int(target["boundary_interval_seconds"]) / 3600
    if target.get(BOUNDARY_MULTIPLES_KEY) is True:
        return f"any whole multiple of {spacing_h:g} h"
    return f"{spacing_h:g} h and no other spacing"


#: Parsed mapping targets, keyed by profile and the verified bytes' digest.
_TARGETS: dict[str, tuple[str, Mapping[str, object]]] = {}


def packaged_mapping_target(profile_id: str) -> Mapping[str, object]:
    """One packaged profile's verified mapping ``target`` contract."""

    path = packaged_authorities(profile_id)["mapping"]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    cached = _TARGETS.get(profile_id)
    if cached is None or cached[0] != digest:
        target = json.loads(data).get("target")
        if not isinstance(target, dict):
            raise RuntimeError(
                f"packaged {profile_id} mapping declares no target contract")
        _TARGETS[profile_id] = (digest, MappingProxyType(target))
    return _TARGETS[profile_id][1]


#: Mapping ``target`` keys that decide which requests a mapping admits and
#: nothing it decodes, each with its reason.  A preparation copies the
#: mapping it ran into its evidence and binds that document's digest
#: everywhere (manifest, composition receipt, cache identity), so when a
#: release adds or changes one of these keys every earlier preparation's
#: digest stops matching the packaged pin although its frames are the
#: same bytes (A166: A159 added the first key below to ten mappings).
#:
#: - :data:`BOUNDARY_MULTIPLES_KEY`: which uniform boundary spacings a
#:   series may have.  Frames are decoded one valid time at a time by the
#:   same ``fields`` table, and the mapping grammar has no time-window
#:   statistic, so a frame does not depend on it; the spacing a
#:   preparation was made at is still held to the packaged target by
#:   :func:`bound_mapping_refusal`.
#: - ``boundary_interval_seconds``: the spacing those multiples count
#:   from (A173 moved icon-global's from 3 h to the 1 h DWD posts).  For
#:   the same reason it decides which series are admitted and no frame:
#:   a frame is one valid time, and a preparation's own spacing is held to
#:   the packaged target, so a 3 h icon-global preparation made before
#:   A173 is still icon-global's.
#:
#: A key joins this set only with the reason it cannot change a frame.
ADMISSION_ONLY_TARGET_KEYS = frozenset({BOUNDARY_MULTIPLES_KEY,
                                        "boundary_interval_seconds"})

#: Mapping ``grid`` keys that decide which files are admitted and nothing
#: decoded, on the same rule:
#:
#: - ``published_first_points``: other encodings of the declared first
#:   grid point that a publisher writes for the same grid (HRRRv1's
#:   millidegree corner, 2.8.9).  A record matching one is admitted as the
#:   declared grid; the frame header and every array come from the
#:   declared ``parameters``, so a preparation made before the key existed
#:   holds the same frames.
ADMISSION_ONLY_GRID_KEYS = frozenset({"published_first_points"})

#: Mapping ``fields.<name>`` keys of the same kind:
#:
#: - ``repeated_record``: which of two messages a publisher wrote for one
#:   record the field reads (HRRRv1's repeated HGT, 2.8.9).  Without it a
#:   repeat is refused, so every frame a preparation decoded before the
#:   key existed had no repeat, and the key reads it the same way.
ADMISSION_ONLY_FIELD_KEYS = frozenset({"repeated_record"})


def mapping_decode_identity(document: Mapping[str, object]) -> str:
    """SHA-256 of a mapping document without its admission-only keys.

    Canonical JSON of the parsed document (sorted keys, no whitespace), so
    two documents with this digest equal parse to the same fields,
    derivations, grid and target, which is everything the engine decodes
    from; only :data:`ADMISSION_ONLY_TARGET_KEYS`,
    :data:`ADMISSION_ONLY_GRID_KEYS` and :data:`ADMISSION_ONLY_FIELD_KEYS`
    are left out.
    """

    stripped = dict(document)
    target = stripped.get("target")
    if isinstance(target, Mapping):
        stripped["target"] = {key: value for key, value in target.items()
                              if key not in ADMISSION_ONLY_TARGET_KEYS}
    grid = stripped.get("grid")
    if isinstance(grid, Mapping):
        stripped["grid"] = {key: value for key, value in grid.items()
                            if key not in ADMISSION_ONLY_GRID_KEYS}
    fields = stripped.get("fields")
    if isinstance(fields, Mapping):
        stripped["fields"] = {
            name: ({key: value for key, value in field.items()
                    if key not in ADMISSION_ONLY_FIELD_KEYS}
                   if isinstance(field, Mapping) else field)
            for name, field in fields.items()}
    canonical = json.dumps(stripped, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: Packaged mappings' decode identities, keyed by profile and pinned digest.
_DECODE_IDENTITIES: dict[str, tuple[str, str]] = {}


def _packaged_mapping_decode_identity(profile_id: str) -> str:
    pinned = packaged_authority_sha256(profile_id)["mapping"]
    cached = _DECODE_IDENTITIES.get(profile_id)
    if cached is None or cached[0] != pinned:
        path = packaged_authorities(profile_id)["mapping"]
        _DECODE_IDENTITIES[profile_id] = (
            pinned, mapping_decode_identity(json.loads(path.read_bytes())))
    return _DECODE_IDENTITIES[profile_id][1]


def bound_mapping_refusal(profile_id: str, bound: bytes, *,
                          spacings_seconds=()) -> str | None:
    """Why a preparation's bound mapping is not this profile's, or None.

    ``bound`` is the mapping document a preparation copied into its
    evidence.  The packaged bytes resolve.  A document that differs from
    the packaged one only in :data:`ADMISSION_ONLY_TARGET_KEYS` resolves
    too, when every boundary spacing the preparation was made at
    (``spacings_seconds``) is one the packaged target takes: its frames
    are the packaged mapping's frames.  Anything else is refused, because
    a frame the packaged mapping would not decode is not this profile's
    preparation.
    """

    pinned = packaged_authority_sha256(profile_id)["mapping"]
    digest = hashlib.sha256(bound).hexdigest()
    if digest == pinned:
        # The packaged bytes: the decode held the series to this very
        # target when it ran.
        return None
    try:
        document = json.loads(bound)
    except (UnicodeError, ValueError):
        return f"its mapping ({digest}) is not a JSON document"
    if (not isinstance(document, dict)
            or mapping_decode_identity(document)
            != _packaged_mapping_decode_identity(profile_id)):
        return (f"its mapping ({digest}) decodes differently from the "
                f"packaged one ({pinned})")
    target = packaged_mapping_target(profile_id)
    for seconds in sorted(set(spacings_seconds)):
        reason = boundary_interval_refusal(
            target, int(seconds), subject="its boundary spacing")
        if reason is not None:
            return reason
    return None


__all__ = [
    "ADMISSION_ONLY_FIELD_KEYS", "ADMISSION_ONLY_GRID_KEYS",
    "ADMISSION_ONLY_TARGET_KEYS",
    "bound_mapping_refusal",
    "mapping_decode_identity",
    "BOUNDARY_MULTIPLES_KEY", "boundary_interval_refusal",
    "boundary_interval_takes", "packaged_mapping_target",
    "PROFILE_ROLES", "packaged_authorities", "packaged_authority_sha256",
    "packaged_normalization", "packaged_normalizer_ids",
    "packaged_composition",
    "packaged_contributing_mappings", "packaged_contributing_sha256",
    "packaged_gfs_vtable", "packaged_gfs_vtable_sha256",
    "packaged_member_grammar", "packaged_member_grammar_ids",
    "packaged_member_grammar_sha256",
    "packaged_physical_contract", "packaged_physical_contract_ids",
    "packaged_physical_contract_sha256", "packaged_profile",
    "packaged_profile_ids", "twentycrv3_authorities",
    "twentycrv3_authority_sha256",
]
