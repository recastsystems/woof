# Hydro-world data licences

Every candidate global source for worldwide `woof hydro`, checked against its licence text at the source on
2026-10-18. Each licence or terms page was fetched without logging in or registering, and the bytes are kept
with a SHA256 under `$HYDRO_WORLD_DATA/licences/<source>/` on the verifying node. That folder's
`SOURCE.md` lists every file with its URL, fetch time and hash, and `SHA256SUMS` covers every file.

The bar (the licence ruling, 2026-10-14): WOOF stays Apache-2.0, and no default path goes into data whose
licence limits sale or use. A source passes only if its licence, read at the source, allows commercial use and
redistribution of derived products. Anything NonCommercial, NoDerivatives, available only after registration, or
unclear is excluded, and the excluded sentence is quoted.

## Verdicts

| Source | Licence as read | Verdict | Free access without an account |
|---|---|---|---|
| Copernicus DEM GLO-30 / GLO-90 | Copernicus WorldDEM-30 / -90 licence | ALLOWED with attribution | yes (AWS open data bucket `copernicus-dem-30m`, HTTP 200) |
| Products derived from Copernicus DEM GLO-30 Public / GLO-90 (flow direction, accumulation, streams, catchments, the shipped hydrofabric) | same licence, Art. 4(d) and Art. 6(b), 6(c), 6(e) | ALLOWED with the "produced using" notice, the no-liability sentence, and the same obligations passed to whoever we let redistribute; only the -F (Free & Open) instances, never GLO-30-R | yes (same buckets) |
| HydroLAKES v1.0 | CC BY 4.0 | ALLOWED with attribution | yes (`data.hydrosheds.org`, HTTP 200) |
| GDW (Global Dam Watch) v1.0 | CC BY 4.0 | ALLOWED with attribution | yes (figshare 10.6084/m9.figshare.25988293) |
| GRanD v1.3 | no licence on its own page; merged into GDW v1.0 | ALLOWED through GDW only | through GDW |
| SoilGrids 2.0 | CC BY 4.0 | ALLOWED with attribution | yes (`files.isric.org`, HTTP 200) |
| ESA WorldCover 2021 | CC BY 4.0 | ALLOWED with attribution | yes (AWS `esa-worldcover`, HTTP 200) |
| USGS NWIS | US public domain | ALLOWED (credit asked for) | yes (`waterservices.usgs.gov`, HTTP 200) |
| German PEGELONLINE (WSV) | Datenlizenz Deutschland Zero 2.0 | ALLOWED, no attribution required | yes (REST API v2, HTTP 200) |
| ECCC real-time hydrometric API (`api.weather.gc.ca`, collection `hydrometric-realtime`) | ECCC Data Servers End-use Licence v2.1 (page header 2.1.1, August 2026) | ALLOWED with attribution ("Data Source: Environment and Climate Change Canada"); provisional data, last 30 days | yes (OGC API items, HTTP 200) |
| Swiss BAFU hydrological data (hydrodaten.admin.ch, LINDAS, geo.admin OGD layers) | BAFU Liefer- und Nutzungsbedingungen 2020 + opendata.swiss "Open use" | ALLOWED, source recommended; open routes are LINDAS and the geo.admin current-state files; historical series come by a BAFU order (an order is not an account, but it is not automatic) | yes for current values (LINDAS SPARQL HTTP 200, geo.admin JSON HTTP 200); historical: order form |
| French Hub'eau hydrometrie API (PHyC data of the Service Central Vigicrues) | Licence Ouverte / Etalab 2.0 | ALLOWED with attribution (source and date of last update) | yes (`hubeau.eaufrance.fr/api/v2/hydrometrie`, HTTP 206 paged JSON) |
| French Vigicrues site content | Licence Ouverte / Etalab 2.0 ("Sauf mention contraire") | ALLOWED with attribution; gauge data is read through Hub'eau (same PHyC source, documented API) rather than the Vigicrues pages | yes |
| Dutch Rijkswaterstaat Waterinfo / WaterWebservices (DDAPI20) | CC0 statement | ALLOWED, no attribution required (courtesy credit printed) | yes (`ddapi20-waterwebservices.rijkswaterstaat.nl`, HTTP 200 POST) |
| ERA5 single levels through Google ARCO-ERA5 | CC BY 4.0 (CDS catalogue); ARCO code Apache-2.0 | ALLOWED with C3S attribution | yes (public ARCO object store `gcp-public-data-arco-era5`; distinct from ERA5-Land) |
| NOAA NWM v3.1.6 parameter files (`parm/domain`: GWBUCKPARM, soilproperties, wrfinput and the rest) | US government work: NWS "in the public domain, unless specifically annotated otherwise"; NODD "can be used as desired" | ALLOWED (public domain), NOAA credit, no endorsement, modified data not presented as NOAA's; parameter priors, not calibrated basins | yes (`nco.ncep.noaa.gov`, HTTP 200) |
| BfG Undine gauge pages (published gauge catchment areas, e.g. Rhine at Kaub, Koeln, Rees) | no reuse licence: Undine/BfG Impressum, "dienen ausschließlich zur individuellen Information des Nutzers"; private and own use under § 53 UrhG | VERIFICATION REFERENCE ONLY: never an input, never bundled or shipped; the published catchment areas are cited as facts with the page URL in receipts; page bytes are never copied into the tree or a release | yes (plain HTTPS pages, HTTP 200) |
| NOAA NextGen hydrofabric v2.0.1 LowerColorado subset (`LowerColorado_NGEN_v201.gpkg` in the NOAA-OWP t-route repository, test case LowerColorado_TX_v4) | Apache-2.0 (t-route LICENSE) plus the NOAA TERMS.md disclaimer; no licence field in the gpkg | REFERENCE AND RECIPE PROVENANCE: no fabric bytes bundled; recipes.v1.json records channel/lake constants and fitted side-slope coefficients from this subset. Upstream release terms remain unverified; this row does not clear those derivations for distribution | yes (public GitHub repository) |
| Environment Canada HYDAT | Open Government Licence - Canada | ALLOWED with attribution | yes (`collaboration.cmc.ec.gc.ca`, HTTP 200) |
| Australian BOM Water Data Online | per data owner: 43 owners CC BY, 1 owner other | ALLOWED per owner with that owner's attribution; Snowy Hydro (w00205) EXCLUDED | yes |
| Natural Earth | public domain | ALLOWED | yes (`naciscdn.org`, HTTP 200) |
| OpenStreetMap waterways | ODbL 1.0 | ALLOWED, SHARE-ALIKE: a derived database we publish must also be ODbL | yes |
| HydroSHEDS v1 (core DEM/DIR/ACC, HydroRIVERS, HydroBASINS) | WWF HydroSHEDS v1 License Agreement (custom) | RESEARCH-ONLY (coordinator ruling 2026-10-18T11:50Z): cross-check only, never a default input, never bundled, never fetched by default; four clauses below | yes (`data.hydrosheds.org`, HTTP 200) |
| HydroSHEDS v2 | not released | NOT AVAILABLE: "Release of the data under a free license will begin in 2026." Re-check on release | no data yet |
| ERA5-Land | CC BY 4.0 | ALLOWED (coordinator ruling 2026-10-18T09:05Z): fetched through the engine's existing CDS path and credential, no new account | through the engine's configured CDS credential (anonymous retrieve returns 401) |
| GLDAS (NASA) | CC0 by NASA ESDIS default for NASA-led data | Licence ALLOWED; EXCLUDED pending approval because downloads need an Earthdata Login | no (GES DISC file returns 401 and redirects to `urs.earthdata.nasa.gov`) |
| GloFAS data (CEMS) | CEMS-FLOODS datasets licence | Licence ALLOWED with attribution; EXCLUDED pending approval because downloads need an EWDS account | no (EWDS retrieve API returns 401) |
| LISFLOOD code | EUPL-1.2 (copyleft) | EXCLUDED from any code we ship (copyleft; Apache-2.0 is not on EUPL's compatible list) | n/a |
| MERIT Hydro / MERIT DEM | dual CC BY-NC 4.0 or ODbL 1.0 | EXCLUDED: download needs registration on a Google Form; the ODbL branch would also make derived data share-alike | no (password sent after registration) |
| ESA CCI Land Cover | ESA CCI LC licence plus VITO licence | EXCLUDED: educational/scientific use only; commercial promotion needs prior approval | n/a |
| GRDC | GRDC data policy | EXCLUDED: no commercial use, no redistribution | n/a |
| UK NRFA | UKCEH NRFA click-through licence | EXCLUDED: no sublicensing, no redistribution, 30-day cache limit, confidentiality clause | n/a |

## What this means for a worldwide default

- **Global hydrography (river network, basins, flow directions):** the one fully open global DEM is Copernicus
  GLO-90 / GLO-30. To ship hydrography that is open by default, we derive flow direction, accumulation, the
  river network and basins from Copernicus DEM ourselves (Rust and CUDA, table-driven). The licence lets us
  redistribute that derived hydrography (Art. 4(b) and 4(d)) with the notices in the Copernicus-derived section
  below. HydroSHEDS v1 is RESEARCH-ONLY by the 2026-10-18T11:50Z ruling: a cross-check, never a default input,
  never bundled. MERIT Hydro is out (registration, and NonCommercial or share-alike). HydroSHEDS v2 has no
  released data yet.
- **Lakes and reservoirs:** HydroLAKES (CC BY 4.0) and GDW v1.0 (CC BY 4.0, which includes GRanD v1.3).
- **Land cover and soils:** ESA WorldCover 2021 and SoilGrids 2.0 (both CC BY 4.0). ESA CCI Land Cover is out.
- **Gauges for verification:** USGS NWIS, PEGELONLINE, HYDAT and BOM Water Data Online (per-owner attribution,
  owner table at `docs/public/HYDRO-WORLD-BOM-WDO-OWNERS.tsv`), plus ECCC real-time, Swiss BAFU, French
  Hub'eau and Dutch Rijkswaterstaat. GRDC and NRFA are out. For the Rhine box: PEGELONLINE (German federal
  waterways), BAFU (Basel-Rheinhalle and the Swiss tributaries), Hub'eau (French Rhine, Ill, Moselle) and
  Rijkswaterstaat (Lobith and the Dutch branches) are all ALLOWED. One access note: BAFU and Hub'eau publish
  recent raw values openly (Hub'eau keeps one month of real-time values), but older 10-minute or hourly series
  (for example January 2023) come from a BAFU order, and in France Hub'eau serves daily and monthly values
  (daily mean, daily instantaneous max and min) over the whole record; see the per-source sections.
- **Forcing and land reanalysis:** ERA5 single levels through ARCO-ERA5 (no account) and ERA5-Land (through the
  engine's existing CDS credential, 09:05Z ruling) are ALLOWED. GLDAS and GloFAS still need an account
  (Earthdata, EWDS) and stay EXCLUDED pending approval.
- **Parameters:** NOAA NWM v3.1.6 parameter files are US government work in the public domain and may seed
  parameter priors. This covers the NOAA parameter data only, never WRF-Hydro code (UCAR) and never the NC-licensed
  NextGen v2.2 fabric.
- **Verification references (never inputs):** BfG Undine gauge pages (published catchment areas, no reuse
  licence, cited as facts only) and the NOAA NextGen v2.0.1 LowerColorado subset from t-route (Apache-2.0 as
  distributed, terms-held upstream). BfG page bytes never enter a build or shipped file. LowerColorado is also recorded as recipe provenance; see the qualification below.
- **HydroSHEDS v1:** the brief listed it as "custom permissive ... with attribution". The actual licence is
  stricter (next section), and the 11:50Z ruling makes it RESEARCH-ONLY.

## Per-source evidence (verbatim quotes)

### Copernicus DEM GLO-30 / GLO-90 (ALLOWED with attribution)
Saved: `copernicus-dem/licences-pdf.pdf` (sha256 bb4a01dc...), from
https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf
(linked from the CDSE COP-DEM collection page). The GLO-90-F licence in the same PDF grants the same rights.

> "The Licensor grants to the User the following non-exclusive rights of use regarding the Copernicus WorldDEM-30:
> (a) reproduction; (b) distribution; (c) communication to the General Public; (d) adaptation, modification and
> combination with other data and information." ... "The rights granted under this Licence are worldwide and
> without limitation in time." ... "The use rights granted under this licence are free of charge to the User."

No commercial restriction. Article 6 obligations: print the notice, add the no-liability sentence when
distributing, and bind any Subsequent Users to the same obligations.

Attribution (modified data): `produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and
Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved` (use
WorldDEM-90 for GLO-90), plus `The organisations in charge of the Copernicus programme by law or by delegation do
not incur any liability for any use of the Copernicus WorldDEM-30`.

### HydroLAKES v1.0 (ALLOWED with attribution)
Saved: `hydrolakes/techdoc.pdf` (sha256 3d1e30ab...), `hydrolakes/page.html`.

> "The HydroLAKES database (version 1.a) is licensed under a Creative Commons Attribution 4.0 International
> License." ... "Notwithstanding this free license, we ask users to refrain from redistributing the data in whole
> in its original format on other websites without the explicit written permission from the authors."

Do not mirror the raw files (it is a request, and we do not need to). Attribution: `Contains HydroLAKES v1.0
data (Messager, M.L., Lehner, B., Grill, G., Nedeva, I., Schmitt, O. (2016), Nature Communications 13603,
doi:10.1038/ncomms13603), CC BY 4.0, www.hydrosheds.org`.

### GDW v1.0, including GRanD v1.3 (ALLOWED with attribution)
Saved: `gdw/GDW_TechDoc_v1_0.pdf` (sha256 b299aacb...), `gdw/figshare-api.json` (licence field "CC BY 4.0"),
`gdw/scidata-paper.html`, `gdw/page.html`, `grand/page.html`.

> "The Global Dam Watch (GDW) database is licensed under a Creative Commons Attribution 4.0 International
> License." (TechDoc 4.1). GDW page: "the GOODD and GRanD datasets are fully integrated into the GDW database and
> thus will be discontinued." GRanD page: "GRanD will no longer be updated outside of GDW v1."

The GRanD page itself carries no licence sentence, and the SEDAC GRanD record timed out from the build host, so GRanD is
used only through GDW. Same request as HydroLAKES not to mirror the raw files. Attribution: `Contains Global Dam
Watch database v1.0 (Lehner et al. 2024, Scientific Data, doi:10.1038/s41597-024-03752-9; data
doi:10.6084/m9.figshare.25988293), CC BY 4.0`.

### SoilGrids 2.0 (ALLOWED with attribution)
Saved: `soilgrids/main.html` (sha256 01906ff1...), `soilgrids/faq.html`.

> "The SoilGrids maps are publicly available on the SoilGrids website under the CC-BY 4.0 License."

Attribution: `Contains SoilGrids 2.0 data, ISRIC World Soil Information, CC BY 4.0 (Poggio et al. 2021, SOIL 7,
217-240)`.

### ESA WorldCover 2021 (ALLOWED with attribution)
Saved: `worldcover/access.html` (sha256 71dccb4c...), `worldcover/aws.html`.

> "The ESA WorldCover product is provided free of charge, without restriction of use. For the full license
> information see the Creative Commons Attribution 4.0 International License."

Attribution (the text the page gives): `© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data
(2021) processed by ESA WorldCover consortium`; citation Zanaga et al. 2022, doi:10.5281/zenodo.7254221.

### USGS NWIS (ALLOWED)
Saved: `usgs-nwis/policy.html` (sha256 f24e8478...), `usgs-nwis/datalicensing.html`.

> "USGS-authored or produced data and information are considered to be in the U.S. Public Domain." ... "Credit
> can be provided by including a citation such as the following: Credit: U.S. Geological Survey"

Attribution: `Credit: U.S. Geological Survey (NWIS)`.

### German PEGELONLINE, WSV (ALLOWED, no attribution required)
Saved: `pegelonline/lizenz.html` (sha256 5a7f2204...), `pegelonline/home.html`.

> "Dies geschieht auf der Grundlage der Lizenz DL-DE->Zero-2.0 ... Die bereitgestellten Daten und Metadaten
> dürfen für die kommerzielle und nicht kommerzielle Nutzung insbesondere vervielfältigt, ausgedruckt,
> präsentiert, verändert, bearbeitet sowie an Dritte übermittelt werden"

(In English: under DL-DE Zero 2.0, the data and metadata may be copied, printed, presented, changed, edited
and passed to third parties, for commercial and non-commercial use.) We print a courtesy line anyway:
`Pegeldaten: Wasserstraßen- und Schifffahrtsverwaltung des Bundes (WSV), via PEGELONLINE, dl-de/zero-2-0`.

### Environment Canada HYDAT (ALLOWED with attribution)
Saved: `hydat/ogl.html` (sha256 d7648d6d...), `hydat/opencanada-water-level-data.json` (open.canada.ca record
9128b287-..., license_id `ca-ogl-lgo`, includes "a MS Access database file containing archived daily water
level"), `hydat/opencanada-ckan.json` (record 29df4910-..., `ca-ogl-lgo`, resource
`collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/` where the HYDAT files are published),
`hydat/eccc-open-data-licence.html`. The canada.ca HYDAT landing page timed out from the build host on three tries; the
licence is taken from the Open Government records above.

> "The Information Provider grants you a worldwide, royalty-free, perpetual, non-exclusive licence to use the
> Information, including for commercial purposes, subject to the terms below."

Attribution: `Contains information licensed under the Open Government Licence – Canada.` (Water Survey of
Canada, HYDAT).

### Australian BOM Water Data Online (ALLOWED per data owner)
Saved: `bom/wdo-copyright.html` (sha256 49ca3c2e...; the page links creativecommons.org/licenses/by/4.0/),
`bom/wdo-owners.tsv` (parsed by this lane), `bom/copyright.html`.

> "Use of data from Water Data Online is subject to the terms and conditions of the supplying organisations" ...
> "The data owners listed below have made their data available under the Creative Commons licence." ... "Other
> data owners: The data owners listed below have unspecified or other licence conditions (not Creative Commons).
> Please contact them using the details provided to obtain their licence conditions before using their material."

43 owners are CC BY, each with its own attribution string, and one (Snowy Hydro Limited, w00205) is "other",
so it is EXCLUDED. The general BOM copyright page sets the default for anything not CC: "You must not supply it
to any other person or use it for any commercial purpose." So any series whose owner is missing from the CC list
is excluded. The owner table is `docs/public/HYDRO-WORLD-BOM-WDO-OWNERS.tsv` (owner code, state, name,
attribution, licence) and should be loaded as data, not code.

### Natural Earth (ALLOWED)
Saved: `natural-earth/terms.html` (sha256 c3472c86...), `natural-earth/gh-license.txt`.

> "All versions of Natural Earth raster + vector map data found on this website are in the public domain. You
> may use the maps in any manner ..."

Optional credit: `Made with Natural Earth.`

### OpenStreetMap waterways (ALLOWED, SHARE-ALIKE)
Saved: `osm/copyright.html` (sha256 2de6a026...), `osm/odbl.html` (sha256 b8d8aebb...).

> "You are free to copy, distribute, transmit and adapt our data, as long as you credit OpenStreetMap and its
> contributors. If you alter or build upon our data, you may distribute the result only under the same license."
> ODbL 4.4: "Any Derivative Database that You Publicly Use must be only under the terms of: i. This License ..."

Produced Works (images, maps) need attribution only (ODbL 4.5 and "Produced Work"). If we publish a river
network database built from OSM, that database must be ODbL. Attribution: `Data © OpenStreetMap and its
contributors, available under the Open Database License (https://www.openstreetmap.org/copyright)`.

### HydroSHEDS v1: core layers, HydroRIVERS, HydroBASINS (RESEARCH-ONLY, ruling 2026-10-18T11:50Z)
Saved: `hydrosheds-v1/techdoc-v1_4.pdf` (sha256 66d20ae9...; Appendix A holds the licence),
`hydrosheds-v1/hydrorivers-techdoc.pdf`, `hydrosheds-v1/hydrorivers.html`, `hydrosheds-v1/hydrobasins.html`.

Commercial use is granted:
> "All core data of HydroSHEDS version 1 ... are free for non-commercial and commercial use." HydroRIVERS and
> HydroBASINS: "distributed under the same license agreement as the HydroSHEDS core products".

But the licence is not open redistribution:
The four clauses the ruling rests on (Appendix A of the technical documentation v1.4, quoted in full):
> 2.1.2: "A worldwide, nonexclusive, non-transferable, paid up right and license to (i) market and distribute the
> Licensed Materials incorporated into any Derivative Works to End Users, subject to the terms and conditions of an
> end user license agreement with terms that are at least as protective of the Licensed Materials as are the terms
> of this License ... In no event shall Licensee license or distribute the Licensed Materials as a stand-alone
> product."
> 1.2: "'End Users' means those parties to whom Licensee or an authorized distributor has licensed the Licensed
> Materials integrated with the Licensee Programs pursuant to a written license agreement for such party's use and
> not for redistribution."
> 10.7: "Licensee agrees to maintain complete, clear and accurate records for at least two (2) years backwards at
> any point in time of its activities under this License, including, without limitation, its sales of the Licensed
> Materials integrated with any Derivative Works and the identity and address of each End User"
> 7.1: "This License and the rights granted hereunder will terminate (i) automatically upon any breach by
> Licensee of the terms of this License; or (ii) in the sole discretion of WWF."

In practice: an Apache-2.0 package that ships HydroSHEDS-derived layers would need an EULA flow and two years of
end-user records, and WWF can withdraw it at will. That is a limited licence under the 2026-10-14 ruling.

**Ruling (coordinator, 2026-10-18T11:50Z, applying the licence policy): RESEARCH-ONLY.** HydroSHEDS v1
(HydroRIVERS, HydroBASINS, the HydroSHEDS DEM) is a research and cross-check source only, exactly like the
NextGen v2.2 fabric: never bundled, never fetched by default, never a default input or the default path. A fabric
built from it (the Rhine proof fabric) is a research artifact: the proof stands as a proof of the pipeline, not of
a shippable fabric. The shipped worldwide default derives its own hydrography from Copernicus DEM (section below).
Any cross-check use shows the Exhibit B notice. HydroSHEDS v2 is re-checked when its data exists. Required notice
(Exhibit B): `This product [name] incorporates data from the HydroSHEDS version 1 database which is © World
Wildlife Fund, Inc. (2006-2022) and has been used herein under license. WWF has not evaluated the data as altered
and incorporated within [name], and therefore gives no warranty regarding its accuracy, completeness, currency or
suitability for any particular purpose. Portions of the HydroSHEDS v1 database incorporate data which are the
intellectual property rights of © USGS (2006-2008), NASA (2000-2005), ESRI (1992-1998), CIAT (2004-2006),
UNEP-WCMC (1993), WWF (2004), Commonwealth of Australia (2007), and Her Royal Majesty and the British Crown and
are used under license. The HydroSHEDS v1 database and more information are available at
https://www.hydrosheds.org.`

### HydroSHEDS v2 (NOT AVAILABLE)
Saved: `hydrosheds-v2/v2.html`.
> "The resulting HydroSHEDS v2 database will provide river networks and catchment boundaries at full global
> coverage. Release of the data under a free license will begin in 2026."

Check again when the data is released.

### ERA5-Land (ALLOWED, coordinator ruling 2026-10-18T09:05Z)
Saved: `era5-land/licence.json` (CDS catalogue: `"license": "CC-BY-4.0"`, sha256 78e1afce...), `era5-land/cds.html`,
`era5-land/ccby.txt` (CC BY 4.0 legal code). Access check: a POST to
`cds.climate.copernicus.eu/api/retrieve/v1/processes/reanalysis-era5-land/execution` without a token returns HTTP 401.
Ruling 09:05Z: the engine already fetches ERA5 through its own init-source path with the CDS credential configured
in the environment; ERA5-Land is a CDS dataset under the same credential, so it is ALLOWED and AVAILABLE through
that path. No new registration or account. Attribution: `Contains modified Copernicus Climate Change Service
information (ERA5-Land, Munoz Sabater 2019, doi:10.24381/cds.e2161bac), CC BY 4.0`.

### GLDAS (licence ALLOWED; EXCLUDED pending approval: account needed)
Saved: `gldas/policy.html` (sha256 0814d026...). The GES DISC dataset page is drawn by script and has no
licence text.
> "Unless the content is marked with a use restriction or license, data provided from a NASA-led mission are
> licensed as Creative Commons Zero (CC0)."

Access check: a GLDAS_NOAH025_3H.2.1 file returns HTTP 401 and redirects to `urs.earthdata.nasa.gov` (Earthdata
Login).

### GloFAS data, CEMS (licence ALLOWED; EXCLUDED pending approval: account needed)
Saved: `glofas/licence.pdf` (sha256 428a6e10..., the "CEMS-FLOODS datasets licence" linked from the EWDS
catalogue record for `cems-glofas-historical`), `glofas/ewds.json`.
> "users are granted free access to the data of CEMS EFAS & GloFAS for the following purposes ... (a)
> reproduction; (b) distribution; (c) communication to the public; (d) adaptation, modification and combination
> with other data and information"

Attribution: `Contains modified Copernicus Emergency Management Service information [Year]`. Access check: the
EWDS retrieve API without a token returns HTTP 401.

### LISFLOOD code (EXCLUDED from shipped code)
Saved: `lisflood/code-license.txt` (sha256 275ae014..., ec-jrc/lisflood-code master at 05a2f58b, 2026-09-07).
> "Copyleft clause: If the Licensee distributes or communicates copies of the Original Works or Derivative Works,
> this Distribution or Communication will be done under the terms of this Licence"
> "'Distribution' or 'Communication': any act of ... making available, online or offline, copies of the Work or
> providing access to its essential functionalities at the disposal of any other natural or legal person."

Apache-2.0 is not on the EUPL list of compatible licences (GPL, AGPL, OSL, EPL, CeCILL, MPL, LGPL, CC BY-SA 3.0,
EUPL, LiLiQ). No LISFLOOD code may be ported into WOOF, and it may not be offered as a service. GloFAS output
data falls under the CEMS licence above, not under EUPL.

### MERIT Hydro and MERIT DEM (EXCLUDED)
Saved: `merit-hydro/page.html`, `merit-dem/page2.html` (global-hydrodynamics.github.io).
> "MERIT Hydro is licensed under either of the following licenses: Creative Commons Attribution-NonCommercial 4.0
> International (CC BY-NC 4.0) / Open Data Commons Open Database License (ODbL 1.0)" ... "ODbL 1.0 license:
> Commercial use is permitted; however, derived data based on MERIT Hydro must be made publicly available under
> the same ODbL license." ... "The password for downloading the data will be sent to you by email after
> registration."

MERIT DEM has the same dual licence and the same registration. Excluded because download needs registration.
Even after registration, the ODbL branch would make every derived hydrofabric and flood map ODbL.

### ESA CCI Land Cover (EXCLUDED)
Saved: `esa-cci-lc/esa-cci-licence.pdf` (sha256 8423d13d..., the "ESA CCI licence" on the C3S
`satellite-land-cover` record), `esa-cci-lc/vito-licence.pdf`, `esa-cci-lc/c3s.json`, `esa-cci-lc/policy.html`.
> "You may use one or several CCI-LC products land cover map for educational and/or scientific purposes, without
> any fee" ... "If you wish to use one or several CCI-LC products in advertising or in any commercial promotion,
> you shall acknowledge the ESA CCI Land Cover project and you must submit the layout to the project for approval
> beforehand"

The C3S record attaches three licences (ESA CCI, VITO, CC BY) and does not say which covers which years. The
VITO terms grant use "for its own internal purposes". Excluded as limited and unclear. WorldCover replaces it.

### GRDC (EXCLUDED)
Saved: `grdc/datapolicy.html` (sha256 a7c55edf...), `grdc/policy-guidelines.pdf`, `grdc/portal.html`.
> "No commercial use of the data" ... "No redistribution of the data by the user". Portal: "Commercial use of the
> original data is not allowed. The redistribution of the downloaded data either in part or total to
> unauthorized persons, third parties or to the general public (incl. distribution via Internet) is not
> allowed."

### UK NRFA (EXCLUDED)
Saved: `nrfa/api-licence.txt.html` (sha256 77c844bf..., the licence-url returned by the NRFA API itself),
`nrfa/costs-terms.html`.
> "Your licence does not permit You to sub-license the API or the Data for any purpose." ... "3.1 You may not make
> the Data available for download from any internet site without prior agreement." ... "3.5 You are not permitted
> to cache or store any Data accessed via the API except for a limited volume ... (no longer than 30 calendar
> days)" ... "4.1 You undertake to hold all data and all other information relating to the API and Data
> confidential"

Commercial use is allowed, but the no-sublicensing, caching and confidentiality terms rule out a verification
archive and redistributed products.

## Follow-up rows, verified at the source (2026-10-18, 11:49Z to 12:10Z)

Every file named here is under `$HYDRO_WORLD_DATA/licences/<source>/` on the verifying node with its URL, fetch
time and SHA256 in that folder's `SOURCE.md` and `SHA256SUMS`. Plain-text extractions (`*.txt`) are derived, not source.
These rows replace the provisional "merged-provider" rows that reached `development branch` without saved bytes in
this folder; their content is kept and now carries evidence.

### Products derived from Copernicus DEM (ALLOWED with notices)
The shipped worldwide hydrography (flow direction, flow accumulation, stream extraction, catchments, the HYDRO-WORLD
fabric built on them) derives from Copernicus DEM, so it is an adapted and modified work of the DEM. Saved:
`copernicus-dem/licences-pdf.pdf` (sha256 bb4a01dcd7f61ace...), sections "Licence for COP-DEM-GLO-90-F Global 90m
Full, Free & Open" and "Licence for Copernicus DEM instance COP-DEM-GLO-30-F Global 30m Full, Free & Open";
`copernicus-dem/bucketreadme.html` (the AWS bucket readme).

The operative grant covers derived works and their redistribution:
> Art. 4: "The Licensor grants to the User the following non-exclusive rights of use regarding the Copernicus
> WorldDEM-90: (a) reproduction; (b) distribution; (c) communication to the General Public; (d) adaptation,
> modification and combination with other data and information."
> Art. 3: "The rights granted under this Licence are worldwide and without limitation in time."
> Art. 5: "The use rights granted under this licence are free of charge to the User."

The conditions that travel with a derived product:
> Art. 6(b): "Where the Copernicus WorldDEM-90 data have been adapted or modified, the User shall provide the
> following notice: "produced using Copernicus WorldDEM-90 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
> 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved"."
> Art. 6(c): "The following sentence or its translation in any language shall be added by such Users in a licence or
> any legal warning or notice covering their distribution or communication to the General Public of the Copernicus
> WorldDEM-90: "The organisations in charge of the Copernicus programme by law or by delegation do not incur any
> liability for any use of the Copernicus WorldDEM-90"."
> Art. 6(d): "User shall make sure not to convey the impression to the General Public that the user's activities are
> officially endorsed by the Provider, the Licensor or any other legal entities in charge of the Copernicus
> programme"
> Art. 6(e): "Where the user grants to any Subsequent User the rights to distribute or communicate to the General
> Public the Copernicus WorldDEM-90, modified or not, he shall ensure that its Subsequent Users are bound by the
> above obligations."

The GLO-30-F licence has the same articles with "WorldDEM-30" in place of "WorldDEM-90". Nothing limits sale or commercial use;
the obligations are notice, disclaimer and pass-through, like CC BY. Verdict: ALLOWED. What a shipped fabric must
carry (in its `fabric_metadata` provenance and in the package NOTICE):

`Hydrography produced using Copernicus WorldDEM-90 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The organisations in charge
of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus
WorldDEM-90. Anyone who redistributes this data must keep this notice.`

(Use WorldDEM-30 for GLO-30 fabrics.) Two limits: (1) only the Free & Open instances served by the AWS buckets
`copernicus-dem-90m` (GLO-90) and `copernicus-dem-30m` ("GLO-30 Public") qualify; the same PDF carries a different,
restricted end-user licence for COP-DEM-GLO-30-R and EEA-10-R, which are never used. (2) GLO-30 Public omits a small
list of tiles ("a small subset of tiles covering specific countries are not yet released to the public", bucket
readme); a GLO-30 domain over one of them falls back to GLO-90, never to a restricted instance. "WorldDEM" is an
Airbus trade mark (licence footnote 1), so product names do not use it.

### ECCC real-time hydrometric API (ALLOWED with attribution)
Distinct from the HYDAT archive row. Saved: `eccc-realtime/licence.html` (https://eccc-msc.github.io/open-data/licence/readme_en/,
sha256 b77b61cda7d2b995..., byte-identical to the copy the gauges lane cited as `hydat/eccc-open-data-licence.html`),
`eccc-realtime/hydrometric-readme.html` (the MSC page for real-time hydrometric data, whose "Licence" paragraph points
to that licence), `eccc-realtime/collection.json` (OGC API collection `hydrometric-realtime`, title "Hydrometric Data
- Real-time"), `eccc-realtime/items-probe.json` (one item, HTTP 200, no key).
> "The Information Provider grants you a worldwide, royalty-free, perpetual, non-exclusive licence to use the
> Information, including for commercial purposes, subject to the terms below."
> "Attribute data to the originator, including sources outside Environment and Climate Change Canada. For example;
> "Data Source: Environment and Climate Change Canada""
> "If the Information Provider or the Third Party Information Contributor does not provide a specific attribution
> statement, or if you are using Information from several information providers and multiple attributions are not
> practical for your product or application, you must use the following attribution statement: Contains information
> licenced under the Data Server End-use Licence of Environment and Climate Change Canada."

Version: the page header says "Version 2.1.1 - August 2026" and the versioning paragraph says "This is version 2.1";
both are recorded. The MSC page describes the data as "Real-time water level and flow (discharge) data collected at
over 2100 hydrometric stations across Canada (last 30 days)"; it is provisional. Attribution: `Data Source:
Environment and Climate Change Canada (real-time hydrometric data, provisional), licensed under the ECCC Data Server
End-use Licence (https://eccc-msc.github.io/open-data/licence/readme_en/)`.

### Swiss BAFU hydrological data (ALLOWED, source recommended)
Saved: `bafu/liefer-nutzung-2020-pdf.pdf` ("Liefer- und Nutzungsbedingungen für hydrologische Daten des Bundesamts für
Umwelt BAFU", Stand 13.7.2020, sha256 79e60354b15ab4b8..., linked from the BAFU "Datenservice Hydrologie" page
`bafu/datenservice-de.html`), `bafu/fragen-de.html` (hydrodaten.admin.ch FAQ), `bafu/bedingungen-pdf.pdf` (2019 terms
for the account-based raw-data server), `bafu/pkg-*.html` (opendata.swiss records), `bafu/ogd-terms.html`,
`bafu/geoadmin-terms.html`, probes `bafu/probe-lindas-ask.html` and `bafu/probe-zustand-en.html` (both HTTP 200, no
account).
> 2020 terms, 2.: "Die gelieferten Daten stehen zur freien Nutzung zur Verfügung. Der Kunde darf die Daten zu
> kommerziellen und nicht kommerziellen Zwecken verwenden. Die Angabe der Quelle wird empfohlen."
> (The delivered data are free to use. The customer may use the data for commercial and non-commercial purposes.
> Citing the source is recommended.)
> 2020 terms, 1.: the data "sind gemäss den Nutzungsbedingungen der Datensätze von Bundesbehörden auf opendata.swiss
> ausgestaltet worden" (follow the opendata.swiss terms for federal datasets).
> hydrodaten FAQ: "Die Daten können frei genutzt werden. Die Angabe der Quelle wird empfohlen." Suggested credit:
> "Daten Oberflächengewässer: Abteilung Hydrologie, Bundesamt für Umwelt BAFU (Bezugsdatum)."
> opendata.swiss records for the BAFU station layers (`basismessnetz-oberflachengewasser`,
> `allgemeine-lage-der-fliessgewasser-und-seen`, `vergleich-von-abfluss-und-pegeldaten-mit-den-gefahrenstufen`):
> rights `terms_open`, which the terms page defines as "Open use ... You may use this dataset for commercial
> purposes. You are recommended to provide the source".

Verdict: ALLOWED. Access, as read: current raw values are open through LINDAS (the federal linked-data SPARQL
service, "werden alle 10 Minuten aktualisiert") and the geo.admin JSON files of the OGD layers; the older
account-based raw-data server (2019 terms, "Zugriff ... mit Hilfe eines privaten Leistungsbezügern-Accounts") is
not needed. Historical series (needed for a past event such as January 2023) are ordered from the BAFU Datenservice
Hydrologie ("Die Datenlieferungen sind seit dem 1. Januar 2020 in der Regel kostenlos"); the delivered data fall under
the same free-use terms, but an order is a manual step, so the gauge door cannot fetch them automatically. Note: the
OGD zip `bafu/zustand-zip-probe.zip` still carries a 2013 generic federal website disclaimer (`license.txt`, "Für die
Reproduktion jeglicher Elemente ist die schriftliche Zustimmung der Urheberrechtsträger im Voraus einzuholen"); the
dataset-specific 2020 terms and the current opendata.swiss record are later and specific, and they govern. Basel-Stadt
republishes the Rhine at Basel series on data.bs.ch (opendata.swiss `rhein-wasserstand-pegel-und-abfluss`, publisher
BAFU, `terms_open`). Attribution: `Daten Oberflächengewässer: Abteilung Hydrologie, Bundesamt für Umwelt BAFU
(retrieved <date>)`.

### French Hub'eau hydrometrie (ALLOWED with attribution)
Saved: `hubeau/cgu.html` (https://hubeau.eaufrance.fr/page/conditions-generales, sha256 32451c23520476e2...),
`hubeau/api-hydrometrie.html`, `hubeau/probe.json` (stations, HTTP 206 paged JSON, count 6507, no key),
`vigicrues/lo2-md.html` and `vigicrues/lo2-pdf.pdf` (the Licence Ouverte 2.0 text).
> CGU: "La réutilisation des Jeux de données est régie par la licence ouverte Etalab ... Les Jeux de données sont
> donc librement et gratuitement utilisables et réutilisables, y compris dans un but commercial. L'utilisateur de ces
> données doit néanmoins veiller à citer l'auteur des Jeux de données."
> Licence Ouverte 2.0: "Le « Concédant » concède au « Réutilisateur » un droit non exclusif et gratuit de libre
> « Réutilisation » de l'« Information » objet de la présente licence, à des fins commerciales ou non, dans le monde
> entier et pour une durée illimitée" ... the reuser must "mentionner la paternité de l'« Information » : sa source
> (a minima le nom du « Concédant ») et la date de la dernière mise à jour de l'« Information » réutilisée."

The API page: "Les données de l'API "Hydrométrie" sont issues de la plate-forme HYDRO Centrale (PHyC), opérée par le
Service Central Vigicrues (SCV, ex-SCHAPI)"; real-time values: "maintient un historique d'un mois"; daily and monthly
values ("débits moyens journaliers et débits moyens mensuels", daily instantaneous max and min) "sur tout
l'historique (depuis 1900 pour certaines stations)". The Licence Ouverte states it is compatible with CC BY. Verdict:
ALLOWED. Attribution: `Données hydrométriques : Service Central Vigicrues / PHyC, via Hub'Eau (hubeau.eaufrance.fr),
Licence Ouverte 2.0, mise à jour <date>`.

### French Vigicrues (ALLOWED with attribution; read gauges through Hub'eau)
Saved: `vigicrues/home.html` (https://www.vigicrues.gouv.fr/, sha256 028dbc14d171b0a9...), `vigicrues/datagouv.html`,
`vigicrues/ds-schapi.html` (data.gouv.fr records of the SCHAPI/Vigicrues reach layer, licence `fr-lo`).
> Site footer: "Sauf mention contraire, tous les contenus de ce site sont sous licence etalab-2.0" (linked to the
> Licence Ouverte text).

Verdict: ALLOWED, same terms as Hub'eau. Vigicrues has no documented public data API in the saved pages, and its
own banner says flows on the site "sont arrondies à 0,1 m³/seconde" and points to HydroPortail for reference data;
Hub'eau serves the same PHyC data through a documented API, so the gauge door reads French gauges through Hub'eau.
(HydroPortail's own terms were not read here and it is not a source in this table.) Attribution: as Hub'eau,
naming Vigicrues.

### Dutch Rijkswaterstaat Waterinfo / WaterWebservices (ALLOWED, CC0)
Saved: `rws/rwsdata.html` (https://rijkswaterstaatdata.nl/waterdata/, sha256 bb8da258c5c89e4e...), `rws/opendata.html`
(rijkswaterstaat.nl open-data page), `rws/waterinfo.html` (the Waterinfo app shell, script only),
`rws/probe-catalogus-compartimenten.json` (POST to the DDAPI20 catalogue, HTTP 200, no key).
> "Op de inhoud van de WaterWebservices is de Creative Commons zero verklaring (CC0) van toepassing. Dit houdt in
> dat hergebruik van de inhoud van deze service is toegestaan, tenzij bij een bepaald onderdeel staat aangegeven dat
> een auteursrechtelijke uitzondering van toepassing is."
> (The CC0 declaration applies to the content of the WaterWebservices: reuse is allowed unless a part is marked with
> a copyright exception.)
> rijkswaterstaat.nl: "Iedereen mag de open data van Rijkswaterstaat vrij inzien ... Hergebruik mag meestal zonder
> bronvermelding."

The same page documents the API (`ddapi20-waterwebservices.rijkswaterstaat.nl`, OphalenWaarnemingen, up to 160,000
observations per request, "ca. 3 jaar aan 10-minuut gegevens") and a fair-use clause ("Bij onredelijk gebruik ... kan
Rijkswaterstaat genoodzaakt zijn uw toegang (tijdelijk) te beperken"). Verdict: ALLOWED, no attribution required.
Courtesy credit printed: `Waterdata: Rijkswaterstaat (WaterWebservices / Waterinfo), CC0`.

### ERA5 single levels through Google ARCO-ERA5 (ALLOWED with attribution)
The source is ERA5 single levels, not ERA5-Land. Saved: `era5-arco/cds-era5-sl.json` (CDS catalogue record
`reanalysis-era5-single-levels`: `"license": "CC-BY-4.0"`, link `rel: license` to https://spdx.org/licenses/CC-BY-4.0),
`era5-arco/readme.html` (the ARCO-ERA5 README.md, sha256 790e4b77ee742d55..., byte-identical to the copy pinned by the
init source table), `era5-arco/license.txt` (ARCO-ERA5 code licence, Apache-2.0, identical to the init lane's copy),
`era5-arco/cc-by-4.0.txt` (sha256 9ba9550ad48438d0..., identical). The fresh catalogue record differs from the init
lane's pinned copy (sha256 8565fb917c1d7647...) only in its `cads:sanity_check` timestamp and `links`; the licence
field is the same.
> ARCO README: "Can I use the data for {research,commercial} purposes? Yes, you can use our ERA5 data according to
> the terms of the Copernicus license." ... "Commercial users, please be sure to provide acknowledgement to the
> Copernicus Climate Change Service according to the Copernicus Licence terms."

The README's link to the old "licence-to-use-copernicus-products.pdf" now returns 404 on the CDS; the CDS catalogue
record (CC BY 4.0) is the operative licence. Data access uses the public store `gcp-public-data-arco-era5` without a
CDS account. Attribution: `Contains modified Copernicus Climate Change Service information (ERA5, Hersbach et al.
2020, doi:10.24381/cds.adbb2d47), CC BY 4.0, read through Google ARCO-ERA5 (Carver and Merose 2023)`.

### NOAA NWM v3.1.6 parameter files (ALLOWED, public domain)
Saved: `nwm-v3/nco-parm-domain.html` (directory listing of
https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/nwm.v3.1.6/parm/domain/: GWBUCKPARM, soilproperties, wrfinput,
RouteLink, Fulldom, spatialweights, reservoir indexes and the rest), `nwm-v3/nws-disclaimer.html`
(https://www.weather.gov/disclaimer), `nwm-v3/aws-registry.html` (NOAA NWM on the AWS Open Data registry, the NODD
statement), `nwm-v3/about-nwm.html`, `nwm-v3/noaa-disclaimer.html`, `nwm-v3/nodd.html`.
> NWS: "The information on National Weather Service Web servers and Web sites is in the public domain, unless
> specifically annotated otherwise, and may be used freely by the public. ... This information shall not be modified
> in content and then presented as official government material."
> NODD (AWS registry, License): "NOAA data disseminated through NODD are open to the public and can be used as
> desired. ... NOAA requests attribution for the use or dissemination of unaltered NOAA data. However, it is not
> permissible to state or imply endorsement by or affiliation with NOAA. If you modify NOAA data, you may not state
> or imply that it is original, unaltered NOAA data."

Annotation check: the NetCDF headers of the three files the params lane holds (GWBUCKPARM_CONUS_FullRouting.nc,
soilproperties_CONUS_FullRouting.nc, wrfinput_CONUS.nc, under `$HYDRO_WORLD_DATA/nwm-v3-parameters/`)
carry no licence, copyright or terms attribute; their `history` records NWM v2.1/v3.0 processing (NLCD 2016 land
cover, GEOGRID) and no third-party terms, so the "unless specifically annotated otherwise" exception does not apply.
Verdict: ALLOWED as US government data in the public domain, used as parameter priors (not calibrated basin
parameters). Conditions: credit NOAA, no endorsement, modified values never presented as NOAA's. Attribution:
`Parameter priors derived from NOAA National Water Model v3.1.6 parameter files (NOAA/NWS/OWP, public domain);
modified, not NOAA data`. Scope: this row covers NOAA's parameter data only. It grants nothing for WRF-Hydro code
(UCAR terms), the NC-licensed NextGen v2.2 fabric, or the private v201 reference fabric; and the RouteLink 3.1.6
subset that the nwm-v4-config lane holds as "terms-held" (its processing notes cite the Lynker Reference Fabric 3.0)
stays held until that citation is resolved: this lane did not open RouteLink.

## Verification references and recipe provenance (2026-10-18)

BfG pages are used only to check published drainage areas. LowerColorado also informed the channel and lake recipe, as recorded below. Licence
bytes sit in the licence store (`bfg-undine/`, `noaa-troute-v201/`) and again beside the data in a `licence/`
folder, with a `SOURCE.md` next to each data folder.

### BfG Undine gauge pages (VERIFICATION REFERENCE ONLY; no reuse licence)
Data: `$HYDRO_WORLD_DATA/gauges/bfg-undine/` (three gauge pages, Kaub, Koeln and Rees, fetched by
hydro-world-dem-hydrography for the published "Einzugsgebietsgroesse"). Saved terms: `bfg-undine/undine-impressum.html`
(https://undine.bafg.de/impressum.html, sha256 6dcf06a8f6d8796f...), `bfg-undine/bafg-impressum.html`
(https://www.bafg.de/DE/Service/Impressum/impressum_node.html, sha256 d9f7fd2b6aa6a787...). The Undine site has no
separate data licence; its impressum ("Haftungsausschluss und Nutzungsbedingungen") is the operative text.
> Undine: "Die Inhalte dieser Internetpräsenz dienen ausschließlich zur individuellen Information des Nutzers."
> Undine, Urheberrecht: "Das Urheberrecht und die Nutzungsrechte der Inhalte der Webseiten liegen - sofern nicht anders
> angegeben - bei den Autoren der Undine-Seiten und bei der Bundesanstalt für Gewässerkunde. ... Durch das Herunterladen
> oder Kopieren von Daten, Inhalten, Bildern, Fotos oder anderen Dateien werden keinerlei Rechte bezüglich der Inhalte
> übertragen."
> BfG: "Sie dürfen von Ihnen nur zum privaten und sonstigen eigenen Gebrauch im Rahmen des § 53 Urheberrechtsgesetz
> (UrhG) verwendet werden. ... Eine Vervielfältigung oder Verwendung dieser Seiten oder von Teilen davon in anderen
> elektronischen oder gedruckten Publikationen ist nur mit unserer Einwilligung gestattet."
> (The content is for the user's individual information only; downloading or copying transfers no rights; use is
> limited to private and own use under § 53 UrhG; reproduction in other publications needs BfG's consent.)

Verdict: no reuse licence, so VERIFICATION REFERENCE ONLY. The pages are kept on the build host as own-use evidence and are
never bundled, shipped, republished or read by a build. What we use is three published numbers (catchment area at
Kaub 103,729 km2, Koeln 144,232 km2, Rees 159,300 km2), cited as facts with the page URL in verification receipts;
no page text, table or figure is reproduced. Citation line: `Published catchment area: Bundesanstalt für
Gewässerkunde (BfG), Informationsplattform Undine, <page URL> (retrieved 2026-10-18)`. A gauge series from these
pages would need BfG's consent; the open route for the same German federal gauges is PEGELONLINE (row above, DL-DE
Zero 2.0).

### NOAA NextGen hydrofabric v2.0.1 LowerColorado subset (REFERENCE AND RECIPE PROVENANCE)
Data: `$HYDRO_WORLD_DATA/noaa-troute-lowercolorado-v201/LowerColorado_NGEN_v201.gpkg` (sha256
04f0f385220465eb..., 29,609,984 bytes), copied from the NOAA-OWP t-route repository at commit
12a8eae0cdfed437143c590659fa7077605a5e70, path `test/LowerColorado_TX_v4/domain/`. Used by hydro-world-fabric and
hydro-world-dem-hydrography as the NHDPlus-derived reference network for the CONUS topology check. Saved terms:
`noaa-troute-v201/LICENSE` (sha256 cfc7749b96f63bd3..., 11,358 bytes) and `noaa-troute-v201/TERMS.md` (sha256
fe44727d2099f4ba..., 1,317 bytes), both from that commit.
> LICENSE: "Apache License Version 2.0, January 2004" (two header lines)
> TERMS.md: "This repository is a scientific product and is not official communication of the National Oceanic and
> Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided
> on an 'as is' basis and the user assumes responsibility for its use."
> TERMS.md: "When using or citing the work, you should not imply endorsement by the author or the affirmer."

The gpkg carries no licence, copyright or terms field of its own (its `gpkg_metadata` rows are empty QGIS/GDAL
templates). The saved t-route distribution carries Apache-2.0, but the upstream NextGen hydrofabric v2.0.1
release terms were not read here. The fabric bytes remain terms-held and are not bundled.

Provenance correction: tools/hydro_world/data/recipes.v1.json explicitly derives channel conventions, fitted
side-slope coefficients (6,971 reaches), and level-pool constants (19 complete lake rows) from this subset.
The builder consumes those recipe constants. The earlier statement that nothing derived from this file
enters a build was incorrect. This licence receipt does not resolve distribution of those derivations;
the fabric/licence owners must resolve the upstream terms or replace them with independently sourced
parameters before public qualification. No verdict here covers the NC-licensed NextGen v2.2 fabric. Citation line: `Reference network: NOAA NextGen hydrofabric v2.0.1 subset
LowerColorado_NGEN_v201.gpkg, from NOAA-OWP t-route (Apache-2.0) at commit 12a8eae; verification reference and recipe provenance (see recipes.v1.json); no
NOAA endorsement implied`.

## Gaps and follow-ups

- HydroSHEDS v1: settled, RESEARCH-ONLY (11:50Z ruling). The init-source row is marked by hydro-lead (ruling item 5).
- HydroSHEDS v2: check again when data is released ("Release of the data under a free license will begin in 2026").
- GLDAS and GloFAS still need an account (Earthdata, EWDS) and stay EXCLUDED pending approval.
- BAFU historical series need a manual BAFU order (free, same open terms). The gauge door can use LINDAS for current
  values; a past-event verification over Switzerland needs the ordered file placed under the gauges data root.
- NWM RouteLink 3.1.6: held by nwm-v4-config until its Lynker Reference Fabric 3.0 citation is resolved.
- Gauge-table rows for BAFU, Hub'eau and Rijkswaterstaat (`woof/hydro/usgs_table.json`) are the gauges lane's work;
  this table gives each one its verdict, evidence path and attribution string.
