# Research workspaces

WOOF-ArWen's research catalog organizes 12 weather families into 23 submodes, 38 research questions and 114 recommended experiment configurations. Each final question has three distinct starting configurations; every family and submode also names at least three starting recommendations. The machine-readable authority is `woof/data/tui/research-workspaces.json`.

These are **development recommendations**. Every configuration starts with `qualification_status = "unqualified"`. This status records execution qualification, not accuracy against observations. A recommendation becomes qualified only through execution evidence tied to its catalog identity, source revision, actual configuration, inputs and hardware. Existing release receipts do not qualify this catalog or establish forecast skill. The historical `validation_status = "unvalidated"` field and value remain readable and are exported as compatibility aliases; receipts add `execution_qualification` and retain `science_validation` as a legacy alias. The terminal display and new configuration headings use execution qualification. A qualified configuration still needs separate observation-based validation.

## Reading a recommendation

Start with the physical question, then choose the method. The catalog's resolution, duration, cadence and minimum root span are proposed experiment settings. They are design hypotheses, not literature-derived accuracy thresholds. Native sizing must show the resulting geometry and resource use; a fit that sacrifices the required domain or changes the question must be made explicit.

- **Regional:** an economical reference retaining larger-scale context.
- **Nested:** a live child or hierarchy receives the evolving parent forcing; compare its actual terrain, fields and event timing with the parent.
- **Archived downscale:** replay a selected interval from an existing parent archive. Parent production, archive storage and child preparation remain part of the cost.
- **Moving nest:** retain a fixed-size child around a supported pressure/height, UH, reflectivity or native model-attribute signal within a prepared corridor.
- **Controlled scenario:** compare an explicit supported initial-state change with a fresh unchanged control using identical forcing and physics.

The five hardware classes (8, 12, 16, 24 and 32 GiB) are resource choices separate from these scientific questions. A selected class is not measured free memory or a fit guarantee. Native resource checks must use the actual device, free-memory headroom and host-memory budget. Restricted-budget tests must be labelled as such. This catalog contains no physical-card qualification results.

## Methods and their limits

The packaged diagnostic capabilities in `woof/data/tui/research-diagnostics.json` describe a reviewed subset of the WOOF-ArWen history-import path, so creating a research TOML does not require a built plotting executable. Actual plotting still uses the native renderer and checks saved fields and complete time windows. Each setup names additional analysis when the available charts do not directly calculate its requested quantity. Plain-temperature lapse-rate browse fields are explicitly distinct from the virtual-temperature products.

The hail questions separate growth-supporting storm conditions from the subsequent melting environment. Their diagnostics are proxies: this catalog has no hail-diameter or hail-trajectory product. [NOAA NSSL's hail overview](https://www.nssl.noaa.gov/education/svrwx101/hail/forecasting/) motivates that separation. Rotation comparisons follow the established use of buoyancy and shear in storm-structure experiments; kilometer-scale UH remains a rotating-updraft diagnostic, not a tornado prediction. [Weisman and Klemp (1982)](https://journals.ametsoc.org/view/journals/mwre/110/6/1520-0493_1982_110_0504_tdonsc_2_0_co_2.xml), [Bryan, Wyngaard and Fritsch (2003)](https://journals.ametsoc.org/view/journals/mwre/131/10/1520-0493_2003_131_2394_rrftso_2.0.co_2.xml).

Derecho research retains a whole-system corridor as a separate option from detailed study of one bowing segment. A bow echo or resolved 10 m wind maximum alone does not establish a derecho. [Corfidi et al. (2016)](https://training.weather.gov/wdtd/courses/woc/severe/storm-structures-hazards/winds/derechos/story_content/external_files/Corfidi2016.pdf). Tropical work similarly separates track, pressure/wind evolution, rainbands and landfall, reflecting distinct sampling questions in hurricane research. [NOAA AOML modeling](https://www.aoml.noaa.gov/hurricane-modeling-prediction/), [Hurricane Field Program](https://www.aoml.noaa.gov/wp-content/uploads/2023/04/1_2023HFP_Introduction.pdf).

Archived downscaling requires a continuous parent series and companion physics evidence: a woof restart or the supported stock-WRF namelist route. Plan parent history at 15 minutes or denser; fast evolving comparisons in this catalog prefer five-minute parent history when available. The boundary-cadence ceiling is an explicit choice. Dense output cannot recover changes that were never archived. Repeating refinement stages cannot reconstruct missing source information, and finite-cadence forcing may shift convective initiation relative to a live child. Exact child-grid surface fields can supply finer land-use, lake and coastline identity; terrain remains inherited from the parent. [Native downscale contract and cadence evidence](DOWNSCALE.md).

Tile streaming can reduce resident GPU demand by moving a domain through host memory. It still needs host capacity and incurs transfer/runtime cost. Compact children reduce covered area; moving children require a prepared corridor; sequential archived children reduce concurrent grids while adding stages, storage and inherited-boundary limits. None of these options makes an arbitrary region, finest grid or stage count free. [Native tile-streaming contract](TILES.md).


| Signal | Catalog starting value | Meaning and constraint |
| --- | --- | --- |
| Pressure/height | 850 hPa; 20 m height depth | Height depression on a pressure surface. Sea-level pressure uses `level_hpa = 0` and a separately chosen absolute hPa ceiling. |
| Updraft helicity | 50 m²/s²; 25 dBZ fallback | Requires the reflectivity fallback before rotation develops; a fix can switch between cells. |
| Reflectivity | 35 dBZ | Follows a selected echo region, not a gust front, hailstone, rain maximum or derecho identity. |
| Native model attribute | Explicit case-reviewed threshold; no catalog default | `theta`: total potential temperature in K; `qv`, `qc`, `qr`: vapor, cloud-water and rain-water mixing ratios in kg/kg of dry air; `w`: vertical velocity in m/s, averaged from adjacent faces to mass levels. |

Attribute following selects `max` or `min`, then `column_max`, `column_min`, `column_mean` or `model_level`. The column mean is an unweighted mean over model levels. A model-level selection requires an explicit zero-based mass-level index valid on the source domain; it is not a fixed pressure or height surface. Moisture attributes require a moist source. Columns with non-finite selected values are excluded. Potential temperature is distinct from air temperature, and water-vapor mixing ratio is distinct from relative humidity. These native values establish no additional hazard diagnosis. [Native attribute registry and reduction implementation](../../woof/core/attribute_tracking.py).

All proposed moving configurations use 300 s evaluation/history intent. Native clock and signal availability must accept the actual cadence. Review search margin, shifts, overlap, cooldown and track-fix evidence. Dormant nests still reserve resources.


## Catalog

The tables show scientific intent, not a hardware-sized grid. `12 → 4 → 1.33 km` describes the proposed hierarchy. For archived configurations it describes the intended parent-to-child spacing; the actual archive determines the eligible parent. Root-span minima are planning constraints and must be enlarged for the selected weather case. Output cadence is history cadence, not the dynamical time step.

### Everyday forecast

Build a regional reference, then ask what finer geography or denser sampling changes.

Family starts: `regional-evolution.reference`, Regional reference; `local-contrast.transect`, Regional contrast map; `regional-evolution.front`, Frontal passage detail.

**Regional learning.** Regional evolution and local contrasts share a common source-control approach.
Submode starts: `regional-evolution.reference`, `local-contrast.transect`, `regional-evolution.front`.

#### Regional weather evolution · `regional-evolution`

Relate the source's pressure pattern, fronts and precipitation to the local forecast window.
Physical context: [Prevailing Winds: Flight Environment](https://www.weather.gov/source/zhu/ZHU_Training_Page/winds/Wx_Terms/Flight_Environment.htm).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Regional reference**<br>`regional-evolution.reference` | Regional; 12 km; 12 h; history 15 min | How does the regional pressure and rainfall pattern evolve? Compare source and model at matched valid times. |
| **Frontal passage detail**<br>`regional-evolution.front` | Live nested; 12 → 4 km; 6 h; history 5 min | When does the frontal wind and temperature change cross the focus area? Compare passage timing inside and outside the child. |
| **Full diurnal context**<br>`regional-evolution.diurnal` | Regional; 12 km; 24 h; history 30 min | How much of the local temperature change follows the daily cycle? Compare morning, afternoon and overnight phases from the same source. |

Coverage: proposed root minimum 600 km in both horizontal directions. Cover the region plus the approaching pressure and frontal pattern.
Diagnostics: `mslp_10m_winds`, `2m_temperature`, `total_qpf`, `500mb_height_winds`.
Inputs: A supported source and explicit UTC cycle; retain the source fields for comparison.
Limit: One short regional run samples one realization; it does not establish forecast skill.

#### Local weather contrasts · `local-contrast`

Measure differences between adjacent resolved terrain, land or coastal areas.
Physical context: [Prevailing Winds: Flight Environment](https://www.weather.gov/source/zhu/ZHU_Training_Page/winds/Wx_Terms/Flight_Environment.htm).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Regional contrast map**<br>`local-contrast.transect` | Regional; 12 km; 24 h; history 30 min | Which contrasts already exist at the parent scale? Use the same area definitions and valid times in later refinements. |
| **Resolved geography child**<br>`local-contrast.terrain` | Live nested; 12 → 4 km; 12 h; history 15 min | Does a child with actual finer geography alter the local contrast? Compare native terrain/landmask and model fields, not spacing labels alone. |
| **Archived local replay**<br>`local-contrast.archive` | Archived child; 4 → 1.33 km; 6 h; history 5 min | How does a compact child respond during the strongest contrast? Replay a selected parent interval and compare area means after spin-up. |

Coverage: proposed root minimum 300 km in both horizontal directions. Include both comparison areas and the flow that connects them.
Diagnostics: `2m_temperature`, `2m_relative_humidity`, `10m_wind_speed_and_direction`, `cloud_cover_levels`.
Inputs: Document the land/terrain source and choose comparison areas larger than a few model cells.
Limit: Grid-cell weather does not represent a particular street, shelter or building.

Additional analysis required:

- The requested cloud panel shows low, middle and high layer cloud cover. It does not calculate total cloud fraction.

### Supercells & severe storms

Separate hail growth, hail survival, rotating storms, initiation and long-lived convective wind systems.

Family starts: `hail-growth.environment`, Growth-layer context; `rotation.structure`, Fixed fine structure; `derecho.corridor`, Whole-system corridor.

**Hail.** Separate conditions for hail growth aloft from melting and surface survival.
Submode starts: `hail-growth.environment`, `hail-survival.airmasses`, `hail-growth.structure`.

#### Hail-growth environment · `hail-growth`

Relate buoyancy, shear and storm persistence to conditions that can support hail growth aloft.
Physical context: [Severe Weather 101: Hail Forecasting](https://www.nssl.noaa.gov/education/svrwx101/hail/forecasting/); [Resolution Requirements for the Simulation of Deep Moist Convection (2003)](https://journals.ametsoc.org/view/journals/mwre/131/10/1520-0493_2003_131_2394_rrftso_2.0.co_2.xml).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Growth-layer context**<br>`hail-growth.environment` | Live nested; 12 → 4 km; 6 h; history 15 min | Which storm environments combine stronger buoyancy, steeper lapse rates and sustained shear? Compare pre-storm environments with the later echo and UH paths. |
| **Persistent updraft proxy**<br>`hail-growth.structure` | Live nested; 12 → 4 → 1.33 km; 3 h; history 5 min | Does a finer live child sustain a different rotating echo structure? Compare storm duration and co-location of echo/UH at matched times. |
| **Short hail-case replay**<br>`hail-growth.archive` | Archived child; 4 → 1.33 km; 3 h; history 5 min | How sensitive is the mature-storm structure to offline child forcing? Compare a dense-cadence archived replay with the live-child reference. |

Coverage: proposed root minimum 400 km in both horizontal directions. Cover inflow, expected initiation and the complete storm path; keep the child wider than the storm and its inflow.
Diagnostics: `mucape`, `var:wrf_lapse_rate_700_500`, `bulk_shear_0_6km`, `composite_reflectivity`, `uh_2to5km`.
Inputs: Choose a convectively active case; retain full thermodynamic and hydrometeor history for any separate growth-layer analysis.
Limit: These are environmental and storm-structure proxies; this catalog does not calculate hail diameter, trajectories or verified hail at the ground.

Additional analysis required:

- The stored 700-500 hPa lapse-rate plot uses plain temperature. It is not the virtual-temperature lapse rate; the virtual-temperature quantity requires separate analysis.

#### Hail melting & survival · `hail-survival`

Separate hail-favorable storm structure aloft from the thermodynamic path toward the surface.
Physical context: [Severe Weather 101: Hail Forecasting](https://www.nssl.noaa.gov/education/svrwx101/hail/forecasting/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Warm versus cool inflow**<br>`hail-survival.airmasses` | Live nested; 12 → 4 km; 6 h; history 15 min | How do hail-case thermodynamic profiles differ across the nearby air-mass boundary? Compare consistent profile locations and low-level versus composite echoes. |
| **Low-level echo evolution**<br>`hail-survival.transition` | Live nested; 12 → 4 → 1.33 km; 3 h; history 5 min | Does the low-level echo evolve differently as the storm crosses the boundary? Compare low-level/composite reflectivity changes with profile evolution. |
| **Boundary-crossing replay**<br>`hail-survival.replay` | Archived child; 4 → 1.33 km; 3 h; history 5 min | Is the inferred melting-environment contrast retained in a compact archived replay? Check thermodynamic profiles against the parent before interpreting any echo difference. |

Coverage: proposed root minimum 400 km in both horizontal directions. Include warm and cool sides of the storm path plus representative inflow.
Diagnostics: `composite_reflectivity`, `1km_reflectivity`, `mucape`, `2m_dewpoint_10m_winds`, `var:wrf_lapse_rate_700_500`.
Inputs: Save vertical temperature and moisture fields for an external melting-layer profile; the severe preset alone has no hail-survival product.
Limit: A bright echo or favorable sounding does not prove surface hail; microphysics and observed hail reports are needed for validation.

**Supercell rotation.** Study persistent rotating updrafts and their storm-relative environment.
Submode starts: `rotation.environment`, `rotation.structure`, `rotation.follow`.

Additional analysis required:

- The stored 700-500 hPa lapse-rate plot uses plain temperature. It is not the virtual-temperature lapse rate; the virtual-temperature quantity requires separate analysis.

#### Supercell rotation · `rotation`

Study storm-relative environment and persistent modeled rotating updrafts.
Physical context: [The Dependence of Numerically Simulated Convective Storms on Vertical Wind Shear and Buoyancy (1982)](https://journals.ametsoc.org/view/journals/mwre/110/6/1520-0493_1982_110_0504_tdonsc_2_0_co_2.xml); [Resolution Requirements for the Simulation of Deep Moist Convection (2003)](https://journals.ametsoc.org/view/journals/mwre/131/10/1520-0493_2003_131_2394_rrftso_2.0.co_2.xml).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Shear and buoyancy context**<br>`rotation.environment` | Live nested; 12 → 4 km; 6 h; history 15 min | Which pre-storm environments accompany persistent modeled rotation? Compare environment before convection rather than sampling only storm-modified air. |
| **Fixed fine structure**<br>`rotation.structure` | Live nested; 12 → 4 → 1.33 km; 3 h; history 5 min | How do UH duration and echo organization change in a finer live child? Compare fixed-domain UH paths and echo morphology with the coarser parent. |
| **Follow the rotating updraft**<br>`rotation.follow` | Moving child; 12 → 4 → 1.33 km; 6 h; history 5 min | Can a UH follower keep the rotating updraft and inflow within the child? Compare track fixes, fallback periods and UH paths against a fixed child. |

Coverage: proposed root minimum 600 km in both horizontal directions. Cover storm inflow and the full plausible path, including alternative cells within the search corridor.
Diagnostics: `uh_2to5km`, `uh_2to5km_run_max`, `srh_0_1km`, `bulk_shear_0_6km`, `composite_reflectivity_uh`.
Inputs: Use a source case that supports convection and save fields required for UH and shear.
Limit: UH is a grid-scale rotating-updraft diagnostic, not a tornado, tornado intensity or damage prediction.

**Initiation & organized systems.** Follow initiation into organized convection, including a separate derecho research path.
Submode starts: `convective-initiation.boundary`, `derecho.corridor`, `convective-initiation.fine`.

#### Convective initiation · `convective-initiation`

Test where and when the resolved atmosphere first develops organized echoes.
Physical context: [Severe Weather 101: Thunderstorm Basics](https://www.nssl.noaa.gov/education/svrwx101/thunderstorms/); [The Dependence of Numerically Simulated Convective Storms on Vertical Wind Shear and Buoyancy (1982)](https://journals.ametsoc.org/view/journals/mwre/110/6/1520-0493_1982_110_0504_tdonsc_2_0_co_2.xml).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Boundary and inhibition**<br>`convective-initiation.boundary` | Live nested; 12 → 4 km; 6 h; history 5 min | Where do moisture convergence proxies and weakening inhibition precede echoes? Compare the boundary evolution and first echo times across the area. |
| **Live initiation child**<br>`convective-initiation.fine` | Live nested; 12 → 4 → 1.33 km; 3 h; history 5 min | Does the initiation timing or location shift when a finer child starts before initiation? Compare first sustained echoes using identical thresholds and valid times. |
| **Dense-boundary replay**<br>`convective-initiation.archive` | Archived child; 4 → 1.33 km; 3 h; history 5 min | How much does archived forcing change initiation compared with a live child? Hold source and domain fixed and document parent boundary cadence explicitly. |

Coverage: proposed root minimum 500 km in both horizontal directions. Include the boundary, both neighboring air masses and downstream storm room.
Diagnostics: `mlcape`, `mlcin`, `2m_dewpoint_10m_winds`, `composite_reflectivity`, `bulk_shear_0_6km`.
Inputs: Start before the observed initiation window and preserve an unchanged-source control.
Limit: Subgrid triggers and source errors can dominate initiation; a warm bubble is a separate explicit scenario, never an automatic correction.

#### Derecho & bowing systems · `derecho`

Study the persistence and propagation of an organized convective wind system over its full corridor.
Physical context: [A Proposed Revision to the Definition of Derecho (2016)](https://training.weather.gov/wdtd/courses/woc/severe/storm-structures-hazards/winds/derechos/story_content/external_files/Corfidi2016.pdf); [Severe Weather 101: Thunderstorm Basics](https://www.nssl.noaa.gov/education/svrwx101/thunderstorms/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Whole-system corridor**<br>`derecho.corridor` | Live nested; 12 → 4 km; 12 h; history 5 min | How does the line organize and maintain its wind swath across the regional corridor? Compare propagation, echo organization and the continuity of resolved 10 m wind-snapshot maxima over the entire run. |
| **Bowing-segment detail**<br>`derecho.segment` | Live nested; 12 → 4 → 1.33 km; 3 h; history 5 min | What changes near one bow apex while the live parent retains the larger cold pool? Compare the segment against the full-system reference; retain environmental inflow. |
| **Follow an echo segment**<br>`derecho.follow` | Moving child; 12 → 4 → 1.33 km; 6 h; history 5 min | Can an echo-following child retain the selected bow segment during propagation? Inspect track switches and child clipping against a fixed regional reference. |

Coverage: proposed root minimum 800 km in both horizontal directions. Plan roughly 800–1500 km along the event path when the case requires it, with inflow and lateral clearance; a smaller domain must narrow the research question.
Diagnostics: `composite_reflectivity`, `10m_wind_run_max`, `mslp_10m_winds`, `2m_dewpoint`, `700mb_rh_height_winds`, `bulk_shear_0_6km`.
Inputs: Choose an observed organized-wind case and retain the storm report/analysis reference separately.
Limit: A modeled bow or gust swath does not establish a derecho; a compact child cannot represent the entire system lifetime.

Additional analysis required:

- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.
- The current WOOF-ArWen history renderer does not produce DCAPE. Moisture and temperature context cannot establish downdraft energy; that parcel calculation requires separate analysis.

### Hurricanes & tropical weather

Study an existing cyclone's track, pressure/wind evolution, rainbands and landfall environment.

Family starts: `tropical-track.steering`, Broad steering reference; `tropical-intensity.evolution`, Pressure-wind baseline; `tropical-rainbands.envelope`, Rainband envelope.

**Vortex evolution.** Track and intensity are related but require different coverage and sampling.
Submode starts: `tropical-track.steering`, `tropical-intensity.evolution`, `tropical-track.fixed`.

#### Track & steering · `tropical-track`

Compare a source cyclone's motion with the evolving surrounding flow.
Physical context: [Hurricane Modeling and Prediction Program](https://www.aoml.noaa.gov/hurricane-modeling-prediction/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Broad steering reference**<br>`tropical-track.steering` | Regional; 12 km; 24 h; history 15 min | How does the cyclone move within the source's evolving steering pattern? Compare vortex centre and surrounding flow at matched valid times. |
| **Fixed coastal corridor**<br>`tropical-track.fixed` | Live nested; 12 → 4 km; 24 h; history 15 min | Does a fixed child retain the relevant track corridor while refining regional flow? Compare centre evolution and boundary proximity with the broad reference. |
| **Pressure-centre follower**<br>`tropical-track.follow` | Moving child; 12 → 4 km; 24 h; history 5 min | Can the native pressure-height follower retain the circulation through the forecast? Compare track fixes and parent/child centres, including weak or competing lows. |

Coverage: proposed root minimum 1200 km in both horizontal directions. Cover the full forecast track envelope and upstream steering features; use a basin/regional parent as the case demands.
Diagnostics: `mslp_10m_winds`, `850mb_height_winds`, `500mb_height_winds`, `200mb_height_winds`.
Inputs: The source or supplied scenario must already contain the cyclone; retain an independent track reference for retrospective study.
Limit: Tracker centres are diagnostic choices; a model track is not official guidance and a lost fix is not storm disappearance.

#### Pressure & wind evolution · `tropical-intensity`

Study how modeled central pressure, surface wind and convective structure change together.
Physical context: [Hurricane Modeling and Prediction Program](https://www.aoml.noaa.gov/hurricane-modeling-prediction/); [2023 Hurricane Field Program: Introduction](https://www.aoml.noaa.gov/wp-content/uploads/2023/04/1_2023HFP_Introduction.pdf).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Pressure-wind baseline**<br>`tropical-intensity.evolution` | Live nested; 12 → 4 km; 24 h; history 15 min | Do central pressure and peak resolved winds evolve consistently through the period? Compare both metrics and their locations rather than interpreting either alone. |
| **Inner-core structure**<br>`tropical-intensity.core` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | How does finer live nesting change the modeled wind/echo structure near the core? Compare radial/area summaries after an explicit adjustment period. |
| **Moving core experiment**<br>`tropical-intensity.follow` | Moving child; 12 → 4 → 1.33 km; 12 h; history 5 min | Does following the core preserve useful coverage while sampling pressure and wind evolution? Compare fixed and moving configurations with the same source and physics. |

Coverage: proposed root minimum 800 km in both horizontal directions. Keep the inner circulation, outer rainbands and expected movement comfortably inside the parent corridor.
Diagnostics: `mslp_10m_winds`, `10m_wind_speed_and_direction`, `10m_wind_run_max`, `composite_reflectivity`, `bulk_shear_0_6km`.
Inputs: Use a developed source vortex and document SST, land-surface and physics choices; retain pressure/wind reference data.
Limit: Minimum pressure and grid-cell winds depend on resolution and source vortex quality; no balanced insertion or coupled ocean response is supplied.

**Rainbands & landfall.** Study precipitation asymmetry and the storm's encounter with land.
Submode starts: `tropical-rainbands.envelope`, `tropical-landfall.approach`, `tropical-rainbands.training`.

#### Rainband asymmetry · `tropical-rainbands`

Relate modeled rainband organization and accumulation to the cyclone environment.
Physical context: [2023 Hurricane Field Program: Introduction](https://www.aoml.noaa.gov/wp-content/uploads/2023/04/1_2023HFP_Introduction.pdf); [Severe Weather 101: Flood Forecasting](https://www.nssl.noaa.gov/education/svrwx101/floods/forecasting/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Rainband envelope**<br>`tropical-rainbands.envelope` | Live nested; 12 → 4 km; 12 h; history 15 min | Which quadrants maintain repeated rainband passage and larger rainfall totals? Compare rolling rain windows and cloud/echo evolution relative to the vortex. |
| **Fixed outer-band child**<br>`tropical-rainbands.training` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | How does one outer rainband repeatedly affect the same region? Keep the child fixed over the accumulation target and compare passage timing. |
| **Rainband interval replay**<br>`tropical-rainbands.archive` | Archived child; 4 → 1.33 km; 6 h; history 5 min | How does a dense-cadence replay redistribute rainfall in a selected band interval? Compare area totals and displacement with the live-parent reference. |

Coverage: proposed root minimum 800 km in both horizontal directions. Cover the rainband arc and its moisture supply, including land if accumulation there is the question.
Diagnostics: `composite_reflectivity`, `qpf_1h`, `qpf_6h`, `precipitable_water`, `700mb_rh_height_winds`, `bulk_shear_0_6km`.
Inputs: Choose a period with rainbands present and preserve accumulated-precipitation fields across the requested windows.
Limit: Atmospheric rainfall does not provide streamflow or inundation; a vortex follower may leave an outer rainband behind.

#### Landfall wind & rain · `tropical-landfall`

Separate storm-scale motion from local atmospheric changes as the cyclone encounters land.
Physical context: [Hurricane Modeling and Prediction Program](https://www.aoml.noaa.gov/hurricane-modeling-prediction/); [Severe Weather 101: Flood Forecasting](https://www.nssl.noaa.gov/education/svrwx101/floods/forecasting/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Approach-to-inland context**<br>`tropical-landfall.approach` | Live nested; 12 → 4 km; 24 h; history 15 min | How do regional wind and rainfall patterns change across coast crossing? Compare the storm-relative approach and inland periods. |
| **Fixed landfall detail**<br>`tropical-landfall.coast` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | How does resolved coastline and terrain affect the local wind/rain pattern? Compare child geography and local weather with the parent at matched times. |
| **Coastal archived replay**<br>`tropical-landfall.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | Does the replay preserve coast-crossing timing while altering local extremes? Use an exact child-grid surface file and compare area/track displacement. |

Coverage: proposed root minimum 1000 km in both horizontal directions. Cover offshore approach, the coast and inland path, including terrain that can modify rainfall.
Diagnostics: `mslp_10m_winds`, `10m_wind_run_max`, `10m_wind_speed_and_direction`, `qpf_1h`, `total_qpf`, `2m_temperature`.
Inputs: Use valid land/sea masks and surface state; start before coast crossing and include the post-landfall interval.
Limit: This atmosphere-only configuration does not calculate storm surge, tides, waves, inundation or building-level winds.

Additional analysis required:

- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.

### Scenario & hurricane lab

Compare documented source scenarios and supported initial-state changes against an unchanged control.

Family starts: `scenario-convection.gentle`, 1 K thermal perturbation; `scenario-tropical-input.source`, Supplied-source baseline; `scenario-convection.stronger`, 2 K amplitude comparison.

**Controlled comparisons.** Supported warm bubbles and externally supplied tropical scenarios have separate launch requirements.
Submode starts: `scenario-convection.gentle`, `scenario-tropical-input.source`, `scenario-convection.stronger`.

#### Warm-bubble sensitivity · `scenario-convection`

Measure the response to a clearly specified initial theta perturbation relative to an unchanged prepared control.
Physical context: [The Dependence of Numerically Simulated Convective Storms on Vertical Wind Shear and Buoyancy (1982)](https://journals.ametsoc.org/view/journals/mwre/110/6/1520-0493_1982_110_0504_tdonsc_2_0_co_2.xml).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **1 K thermal perturbation**<br>`scenario-convection.gentle` | Controlled scenario; 12 → 4 km; 3 h; history 5 min | How does a 1 K warm bubble change initiation relative to the unchanged control? Subtract/control-compare first-echo timing and thermodynamic fields. |
| **2 K amplitude comparison**<br>`scenario-convection.stronger` | Controlled scenario; 12 → 4 km; 3 h; history 5 min | Does doubling only the theta amplitude change the response beyond the 1 K case? Hold location, dimensions, vapor treatment and forcing equal to the 1 K case. |
| **Preserve-RH comparison**<br>`scenario-convection.humidity` | Controlled scenario; 12 → 4 km; 3 h; history 5 min | How much of the response changes when the 1 K bubble preserves relative humidity? Compare against 1 K with unchanged vapor; record the accompanying vapor adjustment. |

Coverage: proposed root minimum 300 km in both horizontal directions. Keep the bubble, its surrounding environment and anticipated response well inside the prepared root and child.
Diagnostics: `composite_reflectivity`, `mlcape`, `mlcin`, `uh_2to5km`, `2m_dewpoint_10m_winds`.
Inputs: Use a supported prepared tree with at least two domains; preserve a fresh t=0 control with identical forcing and physics.
Limit: A warm bubble is an imposed local theta change, not balanced storm insertion. It applies at initialization, not as a restart-state edit; route acceptance must be checked before launch.

#### Supplied tropical scenario · `scenario-tropical-input`

Explore an explicitly supplied tropical source scenario while preserving its unchanged source/control history.
Physical context: [Hurricane Modeling and Prediction Program](https://www.aoml.noaa.gov/hurricane-modeling-prediction/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Supplied-source baseline**<br>`scenario-tropical-input.source` | Regional; 12 km; 24 h; history 15 min | What weather follows from the supplied hypothetical source as provided? Compare with the scenario author's unchanged control at matched valid times. |
| **Scenario landfall detail**<br>`scenario-tropical-input.coast` | Live nested; 12 → 4 km; 12 h; history 5 min | How does the supplied scenario's atmospheric landfall structure vary locally? Retain source and physics; compare regional and nested scenario runs. |
| **Follow supplied vortex**<br>`scenario-tropical-input.follow` | Moving child; 12 → 4 km; 24 h; history 5 min | Can a pressure follower retain the supplied scenario's actual vortex? Compare the moving run's centre and coverage with the fixed scenario baseline. |

Coverage: proposed root minimum 1200 km in both horizontal directions. Cover the supplied vortex and track envelope with enough surrounding flow to interpret scenario differences.
Diagnostics: `mslp_10m_winds`, `10m_wind_speed_and_direction`, `total_qpf`, `composite_reflectivity`.
Inputs: Supply a documented scenario TOML plus actual compatible forcing/prepared state that contains the intended cyclone; retain the original source/control.
Limit: No hurricane insertion, relocation or balanced-state operator is available through this catalog. Different source states are not an isolated one-variable causal experiment.

### Snow, ice & blizzards

Study snow bands, lake influence, precipitation transitions and concurrent winter wind.

Family starts: `snow-band.shield`, Storm and snow shield; `lake-snow.fetch`, Lake-fetch environment; `ice-transition.thermal`, Thermal-layer context.

**Snow organization.** Contrast synoptic snow bands with lake-generated bands.
Submode starts: `snow-band.shield`, `lake-snow.fetch`, `snow-band.band`.

#### Synoptic snow bands · `snow-band`

Relate organized winter precipitation to the cyclone, thermal field and moisture.
Physical context: [What Causes a Wintry Mix of Precipitation?](https://www.weather.gov/arx/why_wintrymix).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Storm and snow shield**<br>`snow-band.shield` | Live nested; 12 → 4 km; 24 h; history 15 min | Where does the snow shield persist as the cyclone evolves? Compare phase and rolling QPF with the pressure/thermal pattern. |
| **Fixed snow-band child**<br>`snow-band.band` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Does finer nesting alter the placement or persistence of a narrow band? Compare band position and area totals with the wider reference. |
| **Snow-band replay**<br>`snow-band.archive` | Archived child; 4 → 1.33 km; 6 h; history 5 min | How much does offline forcing alter a selected band interval? Compare dense-boundary replay and live nesting using the same phase diagnostics. |

Coverage: proposed root minimum 900 km in both horizontal directions. Include the approaching cyclone, the cold-side precipitation shield and the possible band displacement.
Diagnostics: `composite_reflectivity`, `2m_temperature`, `850mb_temperature_height_winds`, `qpf_1h`, `700mb_rh_height_winds`.
Inputs: Choose a winter event; save thermodynamic profiles and precipitation fields across the full accumulation window.
Limit: QPF and snow-category fields are not snow depth; snow density, settling and accumulation require their own evidence.

Additional analysis required:

- Precipitation-type categories are not supplied by the current WOOF-ArWen history renderer. Temperature and moisture plots provide environmental context; rain/snow/freezing-rain/ice-pellet classification requires separate analysis or observations.

#### Lake-effect bands · `lake-snow`

Study band placement under flow across an unfrozen or partly frozen lake.
Physical context: [What Is Lake Effect Snow?](https://www.weather.gov/safety/winter-lake-effect-snow).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Lake-fetch environment**<br>`lake-snow.fetch` | Live nested; 12 → 4 km; 12 h; history 15 min | Which wind and thermal conditions coincide with sustained downwind snowfall? Compare the lake crossing and downwind precipitation through wind shifts. |
| **Shoreline-band detail**<br>`lake-snow.band` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | How do band location and inland extent change with resolved shoreline detail? Compare actual lake masks and band displacement at matched times. |
| **Lake-state controlled replay**<br>`lake-snow.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | Does a compact replay preserve the parent band orientation and timing? Use an exact child-grid surface source and document any lake-state interpolation. |

Coverage: proposed root minimum 300 km in both horizontal directions. Cover upstream air, the whole relevant fetch, shoreline and downwind band corridor.
Diagnostics: `composite_reflectivity`, `10m_wind_speed_and_direction`, `850mb_temperature_height_winds`, `qpf_1h`, `2m_temperature`.
Inputs: Inspect the actual lake mask, lake/surface temperature and ice state in source/preparation; preserve low-level wind profiles.
Limit: A refined grid with parent-interpolated lake geometry or surface state does not add the missing lake information.

**Mixed phase & wind.** Thermal transitions and the overlap of snow with wind need separate evidence.
Submode starts: `ice-transition.thermal`, `blizzard.overlap`, `ice-transition.edge`.

Additional analysis required:

- Precipitation-type categories are not supplied by the current WOOF-ArWen history renderer. Temperature and moisture plots provide environmental context; rain/snow/freezing-rain/ice-pellet classification requires separate analysis or observations.

#### Freezing rain & sleet · `ice-transition`

Study the atmospheric transition between liquid and frozen precipitation.
Physical context: [What Causes a Wintry Mix of Precipitation?](https://www.weather.gov/arx/why_wintrymix).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Thermal-layer context**<br>`ice-transition.thermal` | Live nested; 12 → 4 km; 24 h; history 15 min | How does the transition move as warm air advances above surface cold air? Compare phase maps with profiles at stable geographic locations. |
| **Transition-edge detail**<br>`ice-transition.edge` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Does finer terrain and horizontal spacing change the transition timing? Compare the same transect and inspect vertical layer representation. |
| **Mixed-phase replay**<br>`ice-transition.archive` | Archived child; 4 → 1.33 km; 6 h; history 5 min | Are shallow cold-layer changes retained in an archived child? Compare parent/child thermodynamic profiles before interpreting phase differences. |

Coverage: proposed root minimum 600 km in both horizontal directions. Cover cold-air supply, the warm-air intrusion and both sides of the transition boundary.
Diagnostics: `2m_temperature`, `850mb_temperature_height_winds`, `2m_relative_humidity`, `700mb_temperature_height_winds`.
Inputs: Save vertically resolved temperature/moisture, not only 2 m temperature; place a profile transect across the transition.
Limit: Categorical precipitation type does not calculate ice accretion on roads, trees or wires; shallow layers can remain unresolved.

Additional analysis required:

- Precipitation-type categories are not supplied by the current WOOF-ArWen history renderer. Temperature and moisture plots provide environmental context; rain/snow/freezing-rain/ice-pellet classification requires separate analysis or observations.
- The current WOOF-ArWen history renderer does not produce 2 m wet-bulb temperature. Air temperature and humidity are context only; wet-bulb temperature requires a separate calculation.

#### Snow and strong wind · `blizzard`

Study the overlap and timing of snowfall with modeled strong surface winds.
Physical context: [What Causes a Wintry Mix of Precipitation?](https://www.weather.gov/arx/why_wintrymix); [Prevailing Winds: Flight Environment](https://www.weather.gov/source/zhu/ZHU_Training_Page/winds/Wx_Terms/Flight_Environment.htm).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Regional snow-wind overlap**<br>`blizzard.overlap` | Live nested; 12 → 4 km; 24 h; history 15 min | When do snowfall and strong resolved winds coincide over the region? Compare simultaneous fields and duration rather than separate run maxima. |
| **Peak-passage sampling**<br>`blizzard.passage` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | How sharply does the snow-wind overlap change through the peak passage? Compare onset, duration and timing within a fixed child. |
| **Winter-wind local replay**<br>`blizzard.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | Does local refinement alter wind contrasts during the selected snow interval? Compare terrain and resolved sustained-wind snapshots with the parent. |

Coverage: proposed root minimum 900 km in both horizontal directions. Cover the cyclone pressure gradient, snow shield and the focus area's approach/departure.
Diagnostics: `2m_temperature`, `850mb_temperature_height_winds`, `qpf_1h`, `10m_wind_speed_and_direction`, `10m_wind_run_max`, `mslp_10m_winds`.
Inputs: Choose a case spanning snow and wind; retain snow-state fields if interpreting existing snow cover.
Limit: The overlap of modeled snow and wind does not establish blizzard visibility or drifting; no blowing-snow transport model is enabled.

Additional analysis required:

- Precipitation-type categories are not supplied by the current WOOF-ArWen history renderer. Temperature and moisture plots provide environmental context; rain/snow/freezing-rain/ice-pellet classification requires separate analysis or observations.
- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.
- Air temperature and 10 m wind are shown separately. Wind chill is not calculated by this history-rendering path and requires separate analysis.

### Heavy rain & flooding

Study where atmospheric rainfall accumulates and which transport or convective process sustains it.

Family starts: `rain-training.anchor`, Training-region reference; `atmospheric-river.transport`, Moisture-corridor context; `monsoon-rain.cycle`, Monsoon diurnal cycle.

**Convective rainfall.** Training and monsoon convection need storm lifecycle and accumulation views.
Submode starts: `rain-training.anchor`, `monsoon-rain.cycle`, `rain-training.redevelopment`.

#### Training thunderstorms · `rain-training`

Study repeated convective rainfall over a fixed region.
Physical context: [Severe Weather 101: Flood Forecasting](https://www.nssl.noaa.gov/education/svrwx101/floods/forecasting/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Training-region reference**<br>`rain-training.anchor` | Live nested; 12 → 4 km; 12 h; history 5 min | Where does repeated passage produce larger multi-hour rainfall? Compare echo tracks and fixed-area rolling totals. |
| **Fixed redevelopment child**<br>`rain-training.redevelopment` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Does finer live nesting change upstream redevelopment or cell spacing? Compare initiation sites and rainfall accumulation over the same target area. |
| **Dense-cadence rain replay**<br>`rain-training.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | How much does archived forcing shift a training rainfall maximum? Compare area totals and spatial displacement separately against live nesting. |

Coverage: proposed root minimum 500 km in both horizontal directions. Include the fixed rainfall target, upstream redevelopment region and moisture inflow.
Diagnostics: `composite_reflectivity`, `qpf_1h`, `qpf_6h`, `total_qpf`, `precipitable_water`, `2m_temperature_10m_winds`, `2m_dewpoint_10m_winds`.
Inputs: Use an interval containing repeated cells and preserve cumulative precipitation plus dense echo output.
Limit: Rainfall location and amount do not determine runoff or flood depth; a moving child may abandon the accumulation target.

Additional analysis required:

- The plots show temperature, dewpoint and winds separately. The current WOOF-ArWen history renderer does not calculate 2 m equivalent potential temperature; that quantity requires separate analysis.

#### Monsoon convection · `monsoon-rain`

Study the daily development and movement of convection in a moist monsoon regime.
Physical context: [Severe Weather 101: Thunderstorm Basics](https://www.nssl.noaa.gov/education/svrwx101/thunderstorms/); [Severe Weather 101: Flood Forecasting](https://www.nssl.noaa.gov/education/svrwx101/floods/forecasting/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Monsoon diurnal cycle**<br>`monsoon-rain.cycle` | Live nested; 12 → 4 km; 24 h; history 15 min | How does moisture and convection evolve through a full heating cycle? Compare morning moisture with afternoon initiation and evening rain. |
| **Terrain initiation detail**<br>`monsoon-rain.terrain` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Where does the finer child initiate convection relative to resolved terrain? Compare initiation sites and outflow-driven secondary echoes. |
| **Afternoon archived child**<br>`monsoon-rain.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | How much does replaying the afternoon alter rain placement? Compare displaced peaks and fixed-area totals rather than peak amount alone. |

Coverage: proposed root minimum 700 km in both horizontal directions. Include moisture inflow, heated terrain and downstream outflow/precipitation areas.
Diagnostics: `precipitable_water`, `2m_dewpoint`, `sbcape`, `composite_reflectivity`, `qpf_1h`, `total_qpf`.
Inputs: Choose a monsoon case and a UTC window spanning morning through evening; retain moisture-source context.
Limit: One realization does not determine the probability of local rain; terrain and convective initiation remain resolution-sensitive.

**Moisture transport.** Cover the atmospheric-river supply corridor and its landfall terrain.
Submode starts: `atmospheric-river.transport`, `atmospheric-river.barrier`, `atmospheric-river.replay`.

#### Atmospheric-river landfall · `atmospheric-river`

Connect the upstream moisture corridor to precipitation and phase over coastal terrain.
Physical context: [What Are Atmospheric Rivers?](https://www.noaa.gov/stories/what-are-atmospheric-rivers); [Severe Weather 101: Flood Forecasting](https://www.nssl.noaa.gov/education/svrwx101/floods/forecasting/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Moisture-corridor context**<br>`atmospheric-river.transport` | Regional; 12 km; 36 h; history 15 min | How does the offshore moisture corridor approach and shift along the coast? Compare moisture, low-level flow and regional rainfall onset. |
| **Coastal-barrier precipitation**<br>`atmospheric-river.barrier` | Live nested; 12 → 4 km; 24 h; history 15 min | Where does landfall flow coincide with enhanced terrain precipitation? Compare upwind, crest and lee area totals through the event. |
| **Mountain-interval replay**<br>`atmospheric-river.replay` | Archived child; 4 → 1.33 km; 12 h; history 5 min | How does local terrain refinement redistribute rain within a selected peak interval? Use exact child surface geography and compare phase and area totals. |

Coverage: proposed root minimum 1200 km in both horizontal directions. Use an offshore-to-inland region broad enough for the moisture corridor, landfall shift and downstream mountains.
Diagnostics: `precipitable_water`, `850mb_height_winds`, `700mb_rh_height_winds`, `total_qpf`, `qpf_6h`, `2m_temperature`, `850mb_temperature_height_winds`.
Inputs: Retain the offshore moisture corridor and true terrain; save full moisture and wind profiles for external integrated-vapor-transport calculation.
Limit: Precipitable water is not integrated vapor transport. QPF does not include a routed river or inundation forecast.

Additional analysis required:

- Precipitation-type categories are not supplied by the current WOOF-ArWen history renderer. Temperature and moisture plots provide environmental context; rain/snow/freezing-rain/ice-pellet classification requires separate analysis or observations.

### Fire weather

Compare atmospheric drying, wind, overnight recovery and weakly precipitating convection.

Family starts: `fire-drywind.overlap`, Regional dry-wind timing; `fire-recovery.night`, Sunset-to-sunrise reference; `fire-dry-convection.environment`, Dry-layer environment.

**Dryness & wind.** Compare the daytime dry/windy overlap with humidity recovery overnight.
Submode starts: `fire-drywind.overlap`, `fire-recovery.night`, `fire-drywind.terrain`.

#### Dry and windy overlap · `fire-drywind`

Measure when dry near-surface air and strong flow coincide.
Physical context: [Fire Weather Threat Categories](https://www.weather.gov/source/abq/snippets/forecasts-fireweather-dss-threat.htm); [Prevailing Winds: Flight Environment](https://www.weather.gov/source/zhu/ZHU_Training_Page/winds/Wx_Terms/Flight_Environment.htm).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Regional dry-wind timing**<br>`fire-drywind.overlap` | Live nested; 12 → 4 km; 24 h; history 15 min | When and where do the lowest RH and strongest winds overlap? Use simultaneous fields, not unrelated minimum/maximum times. |
| **Terrain-channel detail**<br>`fire-drywind.terrain` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Does resolved terrain concentrate wind within the dry-air period? Compare actual terrain, wind direction and RH patterns. |
| **Dry-wind interval replay**<br>`fire-drywind.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | How does local replay change dry-wind duration at the focus area? Compare time spent in analyst-selected RH/wind bins with the parent. |

Coverage: proposed root minimum 600 km in both horizontal directions. Include the pressure-gradient flow, terrain channels and the dry-air source.
Diagnostics: `2m_relative_humidity_10m_winds`, `10m_wind_run_max`, `2m_temperature`, `2m_relative_humidity`, `total_qpf`.
Inputs: Document surface/terrain data and source soil state; use independent fuel observations for any later fire-impact interpretation.
Limit: Atmospheric dryness and wind do not simulate fuel moisture, ignition, wildfire spread or smoke.

Additional analysis required:

- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.
- Air temperature and relative humidity are context only. Vapour-pressure deficit is not calculated by this history-rendering path and requires separate analysis.

#### Overnight humidity recovery · `fire-recovery`

Study the timing and spatial variation of nighttime moisture and wind recovery.
Physical context: [Fire Weather Threat Categories](https://www.weather.gov/source/abq/snippets/forecasts-fireweather-dss-threat.htm); [Dew and Frost Development](https://www.weather.gov/source/zhu/ZHU_Training_Page/fog_stuff/Dew_Frost/Dew_Frost.htm).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Sunset-to-sunrise reference**<br>`fire-recovery.night` | Regional; 12 km; 24 h; history 15 min | How much does RH recover through the night as air cools and moisture changes? Separate temperature-driven RH changes from dewpoint changes. |
| **Ridge-valley recovery**<br>`fire-recovery.slope` | Live nested; 12 → 4 km; 12 h; history 5 min | Do ridge and valley areas recover differently under persistent flow? Compare same-elevation or documented terrain groups at matched times. |
| **Nocturnal terrain replay**<br>`fire-recovery.replay` | Archived child; 4 → 1.33 km; 12 h; history 5 min | Does finer local geography change recovery timing and morning drying? Compare fixed-area RH/dewpoint sequences and the surface-state provenance. |

Coverage: proposed root minimum 300 km in both horizontal directions. Include ridge, slope and valley comparison areas plus upstream nocturnal flow.
Diagnostics: `2m_relative_humidity_10m_winds`, `2m_dewpoint`, `2m_temperature`, `2m_relative_humidity`, `10m_wind_1h_max`.
Inputs: Start before sunset and include sunrise; preserve surface thermal/moisture state.
Limit: 2 m humidity recovery is not measured fuel-moisture recovery; unresolved shelter and slope exposure can matter.

**Dry convection.** Explore convective precipitation efficiency and outflow in dry air.
Submode starts: `fire-dry-convection.environment`, `fire-dry-convection.outflow`, `fire-dry-convection.replay`.

Additional analysis required:

- Air temperature and relative humidity are context only. Vapour-pressure deficit is not calculated by this history-rendering path and requires separate analysis.

#### Dry convection & outflow · `fire-dry-convection`

Study convection whose atmospheric rain production and surface wetting differ.
Physical context: [Fire Weather Topics: Dry Thunderstorms](https://www.weather.gov/abq/clifeature2010drythunderstorms); [Severe Weather 101: Thunderstorm Basics](https://www.nssl.noaa.gov/education/svrwx101/thunderstorms/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Dry-layer environment**<br>`fire-dry-convection.environment` | Live nested; 12 → 4 km; 12 h; history 15 min | Where do dry low levels and convective instability overlap? Compare the environment before storm modification and later surface rainfall. |
| **Outflow-pulse detail**<br>`fire-dry-convection.outflow` | Live nested; 12 → 4 → 1.33 km; 3 h; history 5 min | Do resolved surface-wind increases occur where convective rain reaches the surface weakly? Compare simultaneous echo, rainfall and resolved-wind timing. |
| **Short dry-storm replay**<br>`fire-dry-convection.replay` | Archived child; 4 → 1.33 km; 3 h; history 5 min | How much does a short replay alter surface rain and outflow placement? Compare wetting-pattern displacement and resolved-wind increases against live nesting. |

Coverage: proposed root minimum 400 km in both horizontal directions. Cover convective source terrain, the dry low-level environment and potential outflow travel.
Diagnostics: `total_qpf`, `2m_dewpoint`, `700mb_rh_height_winds`, `var:wrf_lapse_rate_0_3km`, `2m_temperature`, `10m_wind_run_max`.
Inputs: Save reflectivity and precipitation fields in addition to fire diagnostics; include the pre-convective dry layer.
Limit: No lightning or ignition model is enabled; low QPF alone does not prove dry lightning or a fuel response.

Additional analysis required:

- The current WOOF-ArWen history renderer does not produce DCAPE. Moisture and temperature context cannot establish downdraft energy; that parcel calculation requires separate analysis.
- The stored 0-3 km lapse-rate plot uses plain temperature. It is not the virtual-temperature lapse rate; the virtual-temperature quantity requires separate analysis.
- Air temperature and dewpoint are shown separately. Dewpoint depression is not calculated by this history-rendering path and requires separate analysis.
- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.

### Wind & damaging gusts

Distinguish pressure-gradient wind, convective outflow and nighttime low-level flow.

Family starts: `synoptic-gust.gradient`, Broad pressure-gradient run; `lowlevel-jet.night`, Evening-to-morning corridor; `downburst.environment`, Downdraft context.

**Pressure flow & low-level jets.** Separate synoptic pressure-gradient flow from diurnally varying flow aloft.
Submode starts: `synoptic-gust.gradient`, `lowlevel-jet.night`, `synoptic-gust.front`.

#### Pressure-gradient wind · `synoptic-gust`

Study widespread surface wind under an evolving synoptic pressure gradient.
Physical context: [Prevailing Winds: Flight Environment](https://www.weather.gov/source/zhu/ZHU_Training_Page/winds/Wx_Terms/Flight_Environment.htm).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Broad pressure-gradient run**<br>`synoptic-gust.gradient` | Regional; 12 km; 24 h; history 15 min | How do regional surface winds evolve as the pressure pattern changes? Compare pressure gradients and resolved flow with maxima of saved 10 m wind snapshots. |
| **Frontal-gust child**<br>`synoptic-gust.front` | Live nested; 12 → 4 km; 12 h; history 5 min | How do wind direction and resolved-wind timing change during frontal passage? Compare passage time and the duration of stronger resolved winds across the fixed child. |
| **Local wind exposure replay**<br>`synoptic-gust.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | Does local refinement alter terrain-related wind contrasts during peak flow? Compare terrain and area wind distributions, not a single maximum cell. |

Coverage: proposed root minimum 1000 km in both horizontal directions. Include the controlling high/low pressure features and enough upstream fetch for surface adjustment.
Diagnostics: `mslp_10m_winds`, `10m_wind_speed_and_direction`, `10m_wind_run_max`, `850mb_height_winds`.
Inputs: Cover the upstream pressure pattern and document land-surface roughness and terrain.
Limit: Grid-scale sustained wind and parameterized gusts differ; neither resolves flow around individual structures.

Additional analysis required:

- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.

#### Nocturnal low-level jet · `lowlevel-jet`

Study the nighttime evolution of lower-tropospheric flow and its surface connection.
Physical context: [Temperatures: Low Level Jet](https://www.weather.gov/source/zhu/ZHU_Training_Page/Miscellaneous/lowleveljet/lowleveljet.html); [Prevailing Winds: Flight Environment](https://www.weather.gov/source/zhu/ZHU_Training_Page/winds/Wx_Terms/Flight_Environment.htm).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Evening-to-morning corridor**<br>`lowlevel-jet.night` | Regional; 12 km; 24 h; history 15 min | When does lower-level flow strengthen relative to surface wind? Compare vertical profiles through evening, overnight and morning. |
| **Morning surface coupling**<br>`lowlevel-jet.coupling` | Live nested; 12 → 4 km; 12 h; history 5 min | How does surface wind change as the nocturnal structure evolves into morning? Compare profile wind maxima and resolved 10 m winds at matched locations. |
| **Focused jet-profile replay**<br>`lowlevel-jet.replay` | Archived child; 4 → 1.33 km; 12 h; history 5 min | Does a local replay retain the parent jet timing and surface decoupling? Check vertical-profile agreement before interpreting local wind differences. |

Coverage: proposed root minimum 1000 km in both horizontal directions. Cover the jet corridor, moisture inflow and representative surface regions.
Diagnostics: `850mb_height_winds`, `bulk_shear_0_1km`, `10m_wind_speed_and_direction`, `2m_temperature_10m_winds`, `2m_dewpoint_10m_winds`.
Inputs: Save full vertical winds and thermodynamics; an 850 hPa map alone may miss the jet maximum.
Limit: Layer shear and a pressure-level map do not locate a shallow jet precisely; vertical spacing and surface physics require review.

**Downbursts & outflow.** Study a convective wind pulse on the storm's time and space scales.
Submode starts: `downburst.environment`, `downburst.pulse`, `downburst.replay`.

Additional analysis required:

- The comparison uses resolved 10 m wind snapshots. Surface gusts are not calculated by this history-rendering path and require separate analysis or observations.

#### Downburst & gust-front pulse · `downburst`

Study short-lived convective outflow and its thermodynamic context.
Physical context: [Severe Weather 101: Thunderstorm Basics](https://www.nssl.noaa.gov/education/svrwx101/thunderstorms/); [Resolution Requirements for the Simulation of Deep Moist Convection (2003)](https://journals.ametsoc.org/view/journals/mwre/131/10/1520-0493_2003_131_2394_rrftso_2.0.co_2.xml).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Downdraft context**<br>`downburst.environment` | Live nested; 12 → 4 km; 6 h; history 5 min | Which storm environments precede larger modeled outflow pulses? Compare the pre-storm temperature/moisture environment with later cold/wind/pressure changes; assess downdraft parcel energy separately. |
| **Short pulse detail**<br>`downburst.pulse` | Live nested; 12 → 4 → 1.33 km; 3 h; history 5 min | How do surface cooling, pressure and resolved-wind signals evolve around one storm? Compare simultaneous fields and pulse timing rather than separate run maxima. |
| **Outflow archived replay**<br>`downburst.replay` | Archived child; 4 → 1.33 km; 3 h; history 5 min | How sensitive is one outflow footprint to offline boundary forcing? Compare placement and duration with a live fine-child reference. |

Coverage: proposed root minimum 300 km in both horizontal directions. Cover the cell, its dry/wet surrounding environment and the expected outflow footprint.
Diagnostics: `2m_dewpoint`, `700mb_rh_height_winds`, `composite_reflectivity`, `10m_wind_run_max`, `10m_wind_1h_max`, `2m_temperature_10m_winds`, `mslp_10m_winds`.
Inputs: Choose a convective case and retain dense surface and echo output; allow space for outflow beyond the parent echo.
Limit: DCAPE is an environmental diagnostic, not a realized downdraft speed; these grids may not resolve individual microbursts.

Additional analysis required:

- The current WOOF-ArWen history renderer does not produce DCAPE. Moisture and temperature context cannot establish downdraft energy; that parcel calculation requires separate analysis.
- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.

### Heat, cold & frost

Study humid heat, cold-air advection and local nocturnal cooling separately.

Family starts: `heat-humidity.daynight`, Day and night reference; `cold-advection.airmass`, Cold-air mass evolution; `radiation-frost.night`, Clear-night reference.

**Heat & cold air masses.** Compare humid heat and advected cold air with their larger-scale controls.
Submode starts: `heat-humidity.daynight`, `cold-advection.airmass`, `heat-humidity.humidity`.

#### Humid heat & warm nights · `heat-humidity`

Compare daytime heat, humidity and incomplete overnight cooling across a full daily cycle.
Physical context: [Heat Index](https://www.weather.gov/ctp/heat).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Day and night reference**<br>`heat-humidity.daynight` | Regional; 12 km; 24 h; history 30 min | Where do daytime warmth and elevated overnight minima occur together? Compare temperature, dewpoint and separate daytime/nighttime extrema. |
| **Humid-heat local contrast**<br>`heat-humidity.humidity` | Live nested; 12 → 4 km; 24 h; history 15 min | Does resolved geography alter the relative contributions of heat and humidity? Compare temperature/dewpoint as well as index maps at identical times. |
| **Two-day persistence**<br>`heat-humidity.persistence` | Regional; 12 km; 48 h; history 30 min | Does a second night cool differently under persistent warm conditions? Compare consecutive complete 24-hour windows with the same domain. |

Coverage: proposed root minimum 800 km in both horizontal directions. Cover the warm air mass and coastal/terrain contrasts relevant to the comparison.
Diagnostics: `2m_temperature`, `2m_dewpoint`, `2m_relative_humidity`, `2m_temp_0_24h_max`, `2m_temp_0_24h_min`.
Inputs: Use at least a complete 24-hour window for daily extrema; preserve source soil moisture, surface state and cloud fields.
Limit: Heat index and wet-bulb temperature describe different quantities; neither is a complete personal exposure or health-impact model.

Additional analysis required:

- Air temperature and relative humidity are shown separately. Heat index is not calculated by this history-rendering path and requires separate analysis.
- The current WOOF-ArWen history renderer does not produce 2 m wet-bulb temperature. Air temperature and humidity are context only; wet-bulb temperature requires a separate calculation.

#### Cold-air outbreak · `cold-advection`

Study the arrival and persistence of a cold air mass with its wind field.
Physical context: [Prevailing Winds: Flight Environment](https://www.weather.gov/source/zhu/ZHU_Training_Page/winds/Wx_Terms/Flight_Environment.htm); [What Causes a Wintry Mix of Precipitation?](https://www.weather.gov/arx/why_wintrymix).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Cold-air mass evolution**<br>`cold-advection.airmass` | Regional; 12 km; 36 h; history 15 min | How does the regional cold air advance and persist after frontal passage? Compare lower-tropospheric cooling with 2 m temperature. |
| **Frontal arrival detail**<br>`cold-advection.arrival` | Live nested; 12 → 4 km; 12 h; history 5 min | How rapidly do temperature and wind change at the local front passage? Compare onset timing and wind/temperature co-evolution across the child. |
| **Post-front first night**<br>`cold-advection.night` | Live nested; 12 → 4 km; 24 h; history 15 min | How much additional cooling occurs after the first post-front sunset? Separate advection-period cooling from the following nighttime change. |

Coverage: proposed root minimum 1200 km in both horizontal directions. Cover the cold-air source, front and downstream focus region.
Diagnostics: `2m_temperature`, `850mb_temperature_height_winds`, `10m_wind_speed_and_direction`, `cloud_cover_levels`.
Inputs: Include upstream cold air and a UTC window beginning before the cold-front arrival.
Limit: Wind chill has a defined applicability and is not object temperature; shelter and local surface effects remain unresolved.

**Nighttime cooling.** Resolve the timing and spatial contrasts of a radiative cooling period.
Submode starts: `radiation-frost.night`, `radiation-frost.valley`, `radiation-frost.dawn`.

Additional analysis required:

- Air temperature and 10 m wind are shown separately. Wind chill is not calculated by this history-rendering path and requires separate analysis.
- The requested cloud panel shows low, middle and high layer cloud cover. It does not calculate total cloud fraction.

#### Radiative frost environment · `radiation-frost`

Study clear-night cooling and its relation to moisture, cloud and local terrain.
Physical context: [Dew and Frost Development](https://www.weather.gov/source/zhu/ZHU_Training_Page/fog_stuff/Dew_Frost/Dew_Frost.htm); [Mountain/Valley Fog](https://www.weather.gov/safety/fog-mountain-valley).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Clear-night reference**<br>`radiation-frost.night` | Regional; 12 km; 24 h; history 15 min | Where does the source produce strongest nighttime air-temperature cooling? Compare cooling with cloud, dewpoint and surface wind through the night. |
| **Terrain cooling contrasts**<br>`radiation-frost.valley` | Live nested; 12 → 4 km; 24 h; history 5 min | Do resolved valley and ridge areas cool at different rates? Compare fixed terrain groups and record the actual model elevation. |
| **Dawn-transition detail**<br>`radiation-frost.dawn` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | When does local cooling stop and morning warming begin? Start before the minimum and compare temperature tendency across terrain. |

Coverage: proposed root minimum 250 km in both horizontal directions. Include representative open, ridge and valley areas plus upstream low-level flow.
Diagnostics by configuration:

- `radiation-frost.night`: `2m_temperature`, `2m_dewpoint`, `2m_relative_humidity`, `10m_wind_speed_and_direction`, `cloud_cover_levels`, `2m_temp_0_24h_min`.
- `radiation-frost.valley`: `2m_temperature`, `2m_dewpoint`, `2m_relative_humidity`, `10m_wind_speed_and_direction`, `cloud_cover_levels`, `2m_temp_0_24h_min`.
- `radiation-frost.dawn`: `2m_temperature`, `2m_dewpoint`, `2m_relative_humidity`, `10m_wind_speed_and_direction`, `cloud_cover_levels`.
Inputs: Span sunset through sunrise; use actual terrain and document soil/land-surface state.
Limit: 2 m temperature is not crop, leaf or surface temperature; frost deposition, irrigation and shelter are not resolved by this mode.

Additional analysis required:

- The requested cloud panel shows low, middle and high layer cloud cover. It does not calculate total cloud fraction.
- This six-hour dawn study does not supply a 24-hour minimum. Inspect its saved 2 m temperatures over the stated window; use a full-day study for a daily minimum.

### Clouds, fog & aviation weather

Investigate fog regimes and low-level wind structure with explicit history-field requirements.

Family starts: `radiation-fog.overnight`, Overnight fog environment; `advection-fog.supply`, Moist-air supply context; `aviation-shear.profiles`, Nighttime profile context.

**Fog regimes.** Separate in-place radiative cooling from transported marine or land fog.
Submode starts: `radiation-fog.overnight`, `advection-fog.supply`, `radiation-fog.dawn`.

#### Radiation fog lifecycle · `radiation-fog`

Study low-level saturation, cloud growth and dissipation during a nocturnal cooling cycle.
Physical context: [Radiation Fog](https://www.weather.gov/safety/fog-radiation).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Overnight fog environment**<br>`radiation-fog.overnight` | Live nested; 12 → 4 km; 12 h; history 15 min | When does the surface approach saturation as cooling proceeds? Compare moisture, temperature and low cloud through the same night. |
| **Dawn dissipation child**<br>`radiation-fog.dawn` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Does the finer child change the time or spatial pattern of low-cloud retreat? Compare near-surface saturation and low-cloud diagnostics together; assess visibility separately. |
| **Complete fog-cycle reference**<br>`radiation-fog.cycle` | Regional; 12 km; 24 h; history 15 min | How do formation and breakup relate to the full daily cloud/wind cycle? Compare nighttime formation and morning dissipation without changing source. |

Coverage: proposed root minimum 250 km in both horizontal directions. Include the cooling basin and nearby higher or windier comparison areas.
Diagnostics: `low_cloud_cover`, `2m_relative_humidity_10m_winds`, `2m_temperature`, `2m_dewpoint`, `2m_relative_humidity`, `10m_wind_speed_and_direction`.
Inputs: Start before sunset and include sunrise; retain cloud and near-surface temperature/moisture fields.
Limit: Near-surface saturation alone does not prove fog or visibility; shallow fog depends on vertical resolution and surface physics.

Additional analysis required:

- Low cloud cover and near-surface humidity describe the fog environment, not visibility distance. Visibility and verified fog boundaries require separate analysis or observations.
- Air temperature and dewpoint are shown separately. Dewpoint depression is not calculated by this history-rendering path and requires separate analysis.

#### Advection fog transport · `advection-fog`

Study moist low-level air passing over a colder surface and its inland or alongshore movement.
Physical context: [Advection Fog](https://www.weather.gov/safety/fog-advection).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Moist-air supply context**<br>`advection-fog.supply` | Regional; 12 km; 24 h; history 15 min | How does the source bring moist air across the colder surface? Compare flow and dewpoint/temperature contrast before low cloud arrives. |
| **Fixed fog-arrival child**<br>`advection-fog.arrival` | Live nested; 12 → 4 km; 12 h; history 5 min | When does the modeled low-cloud boundary reach the focus region? Compare actual diagnostic arrival with saturation proxies. |
| **Shoreline penetration detail**<br>`advection-fog.coast` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Does resolved shoreline/terrain alter the inland edge of low cloud? Compare the prepared surface mask and penetration distance. |

Coverage: proposed root minimum 500 km in both horizontal directions. Cover the cold surface, upstream moist-air supply and downstream fog corridor.
Diagnostics: `low_cloud_cover`, `2m_relative_humidity_10m_winds`, `2m_dewpoint`, `2m_temperature`, `10m_wind_speed_and_direction`, `mslp_10m_winds`.
Inputs: Inspect surface temperature and land/sea mask; include upstream moisture and enough time for arrival.
Limit: Coarse or interpolated surface temperature can dominate the answer; this is not a certified ceiling/visibility forecast.

**Low-level shear.** Explore wind changes with height and time near the lower boundary layer.
Submode starts: `aviation-shear.profiles`, `aviation-shear.transition`, `aviation-shear.terrain`.

Additional analysis required:

- Low cloud cover and near-surface humidity describe the fog environment, not visibility distance. Visibility and verified fog boundaries require separate analysis or observations.

#### Low-level wind shear · `aviation-shear`

Study time-varying wind differences through the lowest atmosphere.
Physical context: [Prevailing Winds: Flight Environment](https://www.weather.gov/source/zhu/ZHU_Training_Page/winds/Wx_Terms/Flight_Environment.htm); [Temperatures: Low Level Jet](https://www.weather.gov/source/zhu/ZHU_Training_Page/Miscellaneous/lowleveljet/lowleveljet.html).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Nighttime profile context**<br>`aviation-shear.profiles` | Regional; 12 km; 24 h; history 15 min | How do surface winds differ from winds above the nocturnal layer? Compare full profiles with the 0–1 km diagnostic at fixed locations. |
| **Morning shear transition**<br>`aviation-shear.transition` | Live nested; 12 → 4 km; 12 h; history 5 min | How rapidly does the lower-level shear pattern evolve after sunrise? Compare wind-profile changes with surface wind and cloud evolution. |
| **Terrain-modified shear**<br>`aviation-shear.terrain` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Does resolved terrain alter local low-level wind contrasts? Compare model elevations and profiles across ridge/valley or coastal sites. |

Coverage: proposed root minimum 600 km in both horizontal directions. Cover the low-level flow corridor and terrain/cloud transitions relevant to the profiles.
Diagnostics: `bulk_shear_0_1km`, `10m_wind_speed_and_direction`, `10m_wind_run_max`, `2m_temperature`, `cloud_cover_levels`.
Inputs: Save full vertical winds and thermodynamic fields for profile inspection; retain the full research route/region.
Limit: Bulk layer shear is not runway-specific wind shear, turbulence intensity, icing or a flight-clearance product.

Additional analysis required:

- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.

### Mountains & local terrain

Connect resolved terrain to precipitation, downslope flow and valley cold pools.

Family starts: `orographic-rain.barrier`, Barrier-scale reference; `downslope-flow.crossbarrier`, Cross-barrier environment; `valley-coldpool.formation`, One-night basin context.

**Barrier weather.** Compare upwind, crest and lee precipitation or wind.
Submode starts: `orographic-rain.barrier`, `downslope-flow.crossbarrier`, `orographic-rain.ridge`.

#### Orographic precipitation · `orographic-rain`

Compare precipitation upwind, over a barrier and in its lee under changing flow.
Physical context: [What Are Atmospheric Rivers?](https://www.noaa.gov/stories/what-are-atmospheric-rivers); [What Causes a Wintry Mix of Precipitation?](https://www.weather.gov/arx/why_wintrymix).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Barrier-scale reference**<br>`orographic-rain.barrier` | Live nested; 12 → 4 km; 24 h; history 15 min | How do windward and lee totals differ as the flow changes? Compare fixed slope/crest/lee areas and precipitation phase. |
| **Ridge precipitation detail**<br>`orographic-rain.ridge` | Live nested; 12 → 4 → 1.33 km; 12 h; history 5 min | Does finer terrain shift the placement of precipitation relative to the ridge? Compare actual prepared terrain and area totals against the parent. |
| **Exact-terrain replay**<br>`orographic-rain.replay` | Archived child; 4 → 1.33 km; 12 h; history 5 min | How does a replay with exact child-grid terrain redistribute the peak interval? Hold parent archive and physics evidence fixed and document surface provenance. |

Coverage: proposed root minimum 600 km in both horizontal directions. Cover upstream slope, crest, lee and the moisture-bearing flow approaching the barrier.
Diagnostics: `terrain_height`, `700mb_rh_height_winds`, `850mb_height_winds`, `total_qpf`, `2m_temperature`, `850mb_temperature_height_winds`.
Inputs: Use real terrain at the intended scale and save thermodynamic profiles; include upstream moisture.
Limit: Finer numerical spacing does not supply finer terrain or guarantee a better precipitation distribution; phase and snow accumulation are separate.

Additional analysis required:

- Precipitation-type categories are not supplied by the current WOOF-ArWen history renderer. Temperature and moisture plots provide environmental context; rain/snow/freezing-rain/ice-pellet classification requires separate analysis or observations.

#### Downslope wind & warming · `downslope-flow`

Study cross-barrier flow and the resulting lee-side wind and thermal contrasts.
Physical context: [Unexpected Warming Induced by Foehn Winds in the Lee of the Smoky Mountains](https://www.weather.gov/mrx/downslope).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Cross-barrier environment**<br>`downslope-flow.crossbarrier` | Live nested; 12 → 4 km; 12 h; history 15 min | Which upstream flow and stability changes accompany lee warming and drying? Compare profiles and simultaneous wind/temperature changes across the ridge. |
| **Fixed lee-side child**<br>`downslope-flow.lee` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | How does finer terrain change the local extent of strong wind and warming? Compare ridge representation and fixed-area flow with the parent. |
| **Downslope episode replay**<br>`downslope-flow.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | Does the archived child reproduce the episode timing and broad lee response? Compare upstream profile forcing before assessing differences in resolved surface winds. |

Coverage: proposed root minimum 400 km in both horizontal directions. Include a broad upstream column, windward slope, crest and enough lee distance for adjustment.
Diagnostics: `terrain_height`, `10m_wind_speed_and_direction`, `10m_wind_run_max`, `2m_temperature`, `2m_relative_humidity`, `700mb_height_winds`.
Inputs: Save vertical wind and stability profiles upstream; inspect the actual resolved ridge shape.
Limit: A surface gust map does not prove resolved mountain-wave structure; wave/rotor analysis requires vertical output and adequate resolution.

**Valley cold pools.** Study low-level cooling and persistence within resolved terrain.
Submode starts: `valley-coldpool.formation`, `valley-coldpool.erosion`, `valley-coldpool.persistence`.

Additional analysis required:

- The wind maximum uses stored 10 m wind snapshots. It is not a gust diagnostic; gust magnitude or a gust swath requires separate analysis or observations.

#### Valley cold-pool persistence · `valley-coldpool`

Study the formation, retention and erosion of a resolved valley cold pool.
Physical context: [Mountain/Valley Fog](https://www.weather.gov/safety/fog-mountain-valley); [Dew and Frost Development](https://www.weather.gov/source/zhu/ZHU_Training_Page/fog_stuff/Dew_Frost/Dew_Frost.htm).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **One-night basin context**<br>`valley-coldpool.formation` | Live nested; 12 → 4 km; 24 h; history 15 min | Where does valley air cool relative to surrounding terrain overnight? Compare valley/ridge profiles and fixed-area surface temperatures. |
| **Morning erosion detail**<br>`valley-coldpool.erosion` | Live nested; 12 → 4 → 1.33 km; 12 h; history 5 min | When and where does the cold pool erode as mixing or external flow develops? Compare profile inversion strength with temperature and wind changes. |
| **Two-night persistence**<br>`valley-coldpool.persistence` | Live nested; 12 → 4 km; 48 h; history 30 min | Does the cold pool survive the daytime period and strengthen on a second night? Compare complete daily cycles and document source/surface evolution. |

Coverage: proposed root minimum 250 km in both horizontal directions. Include the full basin, neighboring ridges and pathways for cold-air drainage and ventilation.
Diagnostics: `terrain_height`, `2m_temperature`, `2m_relative_humidity`, `10m_wind_speed_and_direction`, `var:wrf_lapse_rate_0_3km`.
Inputs: Span an evening, night and mixing period; save full profiles and inspect valley depth in model terrain.
Limit: A shallow valley inversion can be poorly represented even on a fine horizontal grid; surface and vertical resolution require separate review.

Additional analysis required:

- The stored 0-3 km lapse-rate plot uses plain temperature. It is not the virtual-temperature lapse rate; the virtual-temperature quantity requires separate analysis.

### Coasts & sea breezes

Study shoreline circulations, coastal convergence and marine low cloud.

Family starts: `sea-breeze.day`, Full heating-cycle reference; `coastal-convergence.boundary`, Coastal-boundary context; `marine-stratus.supply`, Marine-layer daily context.

**Breezes & convergence.** Study the daily shoreline circulation and its interaction with other boundaries.
Submode starts: `sea-breeze.day`, `coastal-convergence.boundary`, `sea-breeze.front`.

#### Sea-breeze lifecycle · `sea-breeze`

Study the daily development, inland movement and evening change of a shoreline circulation.
Physical context: [NWS Charleston Science: Sea Breeze](https://www.weather.gov/chs/science).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Full heating-cycle reference**<br>`sea-breeze.day` | Live nested; 12 → 4 km; 24 h; history 15 min | When does the coastal wind/temperature contrast develop and weaken? Compare offshore, shoreline and inland conditions through the same day. |
| **Breeze-front detail**<br>`sea-breeze.front` | Live nested; 12 → 4 → 1.33 km; 12 h; history 5 min | How does finer shoreline geometry change front position and inland penetration? Compare prepared land/sea masks and consistent front-position criteria. |
| **Evening land-breeze transition**<br>`sea-breeze.return` | Live nested; 12 → 4 km; 12 h; history 5 min | How does the shoreline flow change as land cooling begins? Start before sunset and compare the evening transition with the daytime reference. |

Coverage: proposed root minimum 300 km in both horizontal directions. Include offshore water, heated land and sufficient inland distance for frontal movement.
Diagnostics: `2m_temperature_10m_winds`, `2m_dewpoint_10m_winds`, `mslp_10m_winds`, `10m_wind_speed_and_direction`, `terrain_height`.
Inputs: Retain realistic land/sea mask and surface temperatures; span morning heating through evening.
Limit: Selecting a coast does not create a breeze; source flow, coastline geometry and surface temperatures control the experiment.

#### Coastal convergence & storms · `coastal-convergence`

Study convection near a breeze front or interacting coastal boundary.
Physical context: [Weather in Action: Lake Shadow/Breeze](https://www.weather.gov/bgm/WeatherInActionLakeShadowBreeze); [Severe Weather 101: Thunderstorm Basics](https://www.nssl.noaa.gov/education/svrwx101/thunderstorms/).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Coastal-boundary context**<br>`coastal-convergence.boundary` | Live nested; 12 → 4 km; 12 h; history 5 min | Where do shoreline and background flow changes precede convection? Compare boundary timing and first echoes through the heating period. |
| **Boundary-interaction child**<br>`coastal-convergence.collision` | Live nested; 12 → 4 → 1.33 km; 6 h; history 5 min | Does a finer live child alter where interacting boundaries initiate echoes? Compare boundary position and storm displacement with the regional reference. |
| **Coastal initiation replay**<br>`coastal-convergence.replay` | Archived child; 4 → 1.33 km; 6 h; history 5 min | How much does archived forcing change the coastal initiation and rainfall pattern? Use an exact child coastline and compare against a live-child control. |

Coverage: proposed root minimum 400 km in both horizontal directions. Include coast, inland boundary interaction area and downstream storm/outflow room.
Diagnostics: `composite_reflectivity`, `2m_dewpoint_10m_winds`, `qpf_1h`, `precipitable_water`, `10m_wind_speed_and_direction`.
Inputs: Start before boundary formation; include both interacting flows and the moist convective environment.
Limit: A convergent surface pattern does not ensure convective initiation; tiny boundary errors can shift storm placement.

**Marine low cloud.** Study marine cloud arrival, inland penetration and daytime retreat.
Submode starts: `marine-stratus.supply`, `marine-stratus.inland`, `marine-stratus.replay`.

#### Marine stratus penetration · `marine-stratus`

Study marine low cloud arrival, inland extent and daytime retreat.
Physical context: [Advection Fog](https://www.weather.gov/safety/fog-advection); [NWS Charleston Science: Sea Breeze](https://www.weather.gov/chs/science).

| Configuration | Method and sampling intent | Research comparison |
| --- | --- | --- |
| **Marine-layer daily context**<br>`marine-stratus.supply` | Live nested; 12 → 4 km; 24 h; history 15 min | How do offshore cloud supply and coastal flow relate to inland low cloud? Compare arrival and retreat across a complete daily cycle. |
| **Inland-edge detail**<br>`marine-stratus.inland` | Live nested; 12 → 4 → 1.33 km; 12 h; history 5 min | Does resolved coastal terrain alter cloud penetration through gaps or valleys? Compare actual terrain and low-cloud edge timing at fixed locations. |
| **Marine-layer replay**<br>`marine-stratus.replay` | Archived child; 4 → 1.33 km; 12 h; history 5 min | Does a local replay retain the parent cloud arrival while changing inland extent? Use exact child surface geography and compare thermodynamic profiles. |

Coverage: proposed root minimum 400 km in both horizontal directions. Include offshore cloud supply, coastal gaps and the inland cloud edge.
Diagnostics: `low_cloud_cover`, `2m_relative_humidity_10m_winds`, `2m_temperature_10m_winds`, `850mb_height_winds`.
Inputs: Preserve marine surface temperature and full low-level cloud/thermodynamic output; inspect coastal terrain.
Limit: Low cloud cover alone is not cloud-base height or flight category; inversion depth and surface state may be unresolved.

## Outputs and qualification

All listed diagnostic identifiers exist in the native plot catalog. Required history inputs and time windows remain mandatory. A preset requests products; it does not enable physics, create missing variables, infer a missing 24-hour window or guarantee a useful scientific result. Some listed diagnostics supplement the family plot preset and should be retained in the reviewed plot selection. [Native plot data](../../woof/data/tui/plot-presets.json), [output-variable contract](OUTPUT-VARIABLES.md).

The separate execution-qualification matrix must identify every configuration by ID and record creation, preserved settings, review, preparation, forecast, diagnostics, restart/resume and the method-specific checks that apply. A failed or unrun recommendation remains visible and unqualified. For paired experiments, retain both the control and treatment receipts. Record actual device/driver/OS/free-memory conditions, distinguishing native hardware runs from restricted-budget tests and estimates.

Sources were checked on 2026-09-06. External primary sources motivate the physical questions; native source files establish the admitted features. The selected geometry, cadence and comparison plans are this catalog's research proposals.

Additional analysis required:

- Low cloud cover and near-surface humidity describe the fog environment, not visibility distance. Visibility and verified fog boundaries require separate analysis or observations.
