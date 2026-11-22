# Geography repairs

`gmted2010_e180_glo30.npz`: the GMTED EAST OF 180 patch read by `gpuwm.geog_assets.repair_gmted_east_of_180`.
Arrays `x`, `y` (1-based global indices of the WPS_GEOG `topo_gmted2010_30s` 43200 x 21600 grid, column 1 at
0.0042 E, row 1 at 89.996 S) and `h` (int16 metres): each MODIS 20-class land pixel the GMTED tiles store as 0 m
between 180 and 190 E, with the mean Copernicus GLO-30 height of its 30-arc-second cell. Built by
`tools/build_gmted_e180_patch.py`.

Source: Copernicus DEM GLO-30, produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence
and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA; all rights reserved. Free to use
and redistribute with this attribution.
