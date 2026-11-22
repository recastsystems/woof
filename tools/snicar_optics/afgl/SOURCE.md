# AFGL reference atmospheres (input to the incident solar spectra)

Source: G. P. Anderson, S. A. Clough, F. X. Kneizys, J. H. Chetwynd and
E. P. Shettle, "AFGL Atmospheric Constituent Profiles (0-120 km)",
AFGL-TR-86-0110, Environmental Research Papers No. 954, Air Force
Geophysics Laboratory, 15 May 1986, Tables 1b-1e. A US government work in
the public domain.

Copy read: DTIC ADA175173 as hosted by the Internet Archive,
https://archive.org/download/DTIC_ADA175173/DTIC_ADA175173.pdf,
SHA256 4da4f1c5c8b2c6c7c7de78faf921e120860a1fca122deaa9d4497010baa80c1b.
Text layer from `pdftotext -layout` (poppler 26.01.0), SHA256
6828cf11118b1a24a54a830e12000ec175f54d2711562b0b4f976a51cee3bf63.

Rebuild:

    pdftotext -layout DTIC_ADA175173.pdf a.txt
    python afgl_parse.py a.txt parsed.json
    python afgl_build.py parsed.json afgl_repairs.json profiles.json

`afgl_parse.py` reads each row with the report's fixed column widths and
flags rows that fail the ideal-gas, monotone-pressure, temperature-kink or
gas-smoothness checks. `afgl_repairs.json` holds every hand repair with its
reason, read from the report text. `afgl_build.py` rechecks every level
(hydrostatic scale height, smoothness) and prints the column water and
ozone. They match the report's published values for each model (MLS 2.93 cm
/ 334 DU, MLW 0.86 cm / 378 DU, SAS 2.08 cm / 347 DU, SAW 0.42 cm / 375 DU).
No library or program that redistributes these tables was used.
