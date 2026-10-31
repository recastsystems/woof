# Map assets for the renderer

These three directories are the geography `rw_wrfbatch` draws on every
weather picture: coastlines, land, ocean and lakes, national borders,
state and province lines, and US counties.

| Directory | Source | Licence |
|---|---|---|
| `natural_earth_10m/` | [Natural Earth](https://www.naturalearthdata.com/) 1:10m physical and cultural layers | public domain |
| `natural_earth_110m/` | [Natural Earth](https://www.naturalearthdata.com/) 1:110m layers, including `ne_110m_admin_0_countries` 5.1.1 | public domain |
| `us_counties_5m/` | [US Census Bureau cartographic boundaries](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html), `cb_2023_us_county_5m` | work of the US Government, not subject to copyright (17 U.S.C. 105) |

Each directory keeps its own README naming the exact upstream archives.

## Why they are here

A pip install has to draw its maps with no extra step. The platform
`gpuwm` wheel carries the renderer but not these files, and before 2.8.0
they reached an install only through `woof fetch-bridges`, which neither
install text ran: every picture from a wheel install was drawn with no
coastlines, borders or state lines, and nothing said so. `gpuwm-data` is
a hard dependency that every install already pulls, so the files ship
here and `gpuwm.rustwx` hands this directory to the renderer.

## One copy of record

The copy of record is `tools/rustwx/assets/basemap/` in the source
repository, which the renderer's own build and the bridge bundle read.
These directories are byte-for-byte copies of its three layer
directories. `tests/test_render_basemap_delivery.py` fails when the two
disagree by a byte or when the renderer's tree gains a layer directory
this one lacks, and its message gives the copy command that brings them
back together.
