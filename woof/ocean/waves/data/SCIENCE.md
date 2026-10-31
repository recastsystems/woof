# Wave input and verification equations

Original Apache-2.0 numerical code. No implementation from another ocean model
was consulted. Public station files are data, including their identifying header.

The reconstruction table cites Hasselmann et al. (1973), JONSWAP, Deutsche
Hydrographische Zeitschrift supplement A8(12), equation 2.4.5. The institutional
full text is https://pure.mpg.de/rest/items/item_3262854_4/component/file_3282032/content.
No DOI was verified for this original report. The values gamma=3.3,
sigma_a=0.07 and sigma_b=0.09 specify the JONSWAP shape. Each partition's discrete
integral is normalized to Hs squared divided by 16 by the core's GPU kernel.

Cosine half-angle spreading follows Mitsuyasu et al. (1975), Journal of Physical
Oceanography 5, 750-760, DOI 10.1175/1520-0485(1975)005<0750:OOTDSO>2.0.CO;2.
The peak spreading parameters 10, 25 and 75 are reviewed by Goda (2008),
Proceedings of the Japan Academy B 84, 374-385, DOI 10.2183/pjab/84.374, equations
5-6. Their categories are wind sea, short-decay swell and long-decay swell.
The 16-second partition switch is an engineering proxy, not a published law
relating period to travel distance. Constant s across each reconstructed
partition follows the shared v1 contract. It does not implement the published
frequency-dependent exponents 5 and -2.5. Bulk partitions do not specify a unique
frequency width or within-partition directional shape; the table is a prior.

NDBC coefficient interpretation follows the provider's measurement documentation:
https://www.ndbc.noaa.gov/faq/measdes.shtml. Archived r1 and r2 are divided by
100; alpha1 and alpha2 are nautical FROM angles. The Fourier moments are
c1 = r1 exp(i alpha1) and c2 = r2 exp(2 i alpha2).
Maximum entropy reconstruction follows Lygre and Krogstad (1986), Journal of
Physical Oceanography 16, 2052-2060, DOI
10.1175/1520-0485(1986)016<2052:MEEOTD>2.0.CO;2:
p1=(c1-c2 conjugate(c1))/(1-|c1|^2), p2=c2-p1 c1,
D(theta) proportional to
[1-Re(p1 conjugate(c1)+p2 conjugate(c2))] /
|1-p1 exp(-i theta)-p2 exp(-2i theta)|^2.
The discrete directional integral is one. Nonpositive innovation variance or
invalid Fourier moments produce missing values, not invented peaks.
Directional spread is sqrt(2(1-r1)) radians.

Spectral moments are m_n = integral f^n E(f,theta) df dtheta.
Hs=4 sqrt(m0), Tm01=m0/m1, Tm02=sqrt(m0/m2). Peak period uses the highest
discrete frequency-density bin; source operational peak periods can use a
parabolic frequency fit. Peak direction is the circular mean at that bin.
The source ASCII frequencies are rounded, so integration uses the available
frequency centres with adjacent-centre widths and geometric edge extrapolation.
No unresolved spectral tail is invented.

Bias is mean(model-observation), RMSE is root mean squared error, and scatter
index is the bias-removed error standard deviation divided by mean observed Hs.
The raw RMSE/observed-mean ratio is also retained. Correlation is Pearson's
coefficient and is undefined for constant series. Direction errors wrap into
[-180,180); direction bias is their circular mean. Tm02 versus APD compares the
corresponding moments, but the instrument and model frequency ranges differ.
Tm01 versus APD and mean direction versus MWD are labelled diagnostic comparisons.
NDBC MWD is the mean direction at the dominant period, matching the peak-direction
definition rather than an integrated mean. Pairing is per valid hour, without
reusing an observation, and uses common finite samples across metric providers.

Per-frequency scores cover the full prepared time window. Model frequency-bin
densities are integrated over overlapping observed bins without extrapolation.
The first directional moments are energy-weighted vectors; direction and spread
are calculated from the resulting vectors. Every provider uses the same finite
observed bins and times. Directional spectrum pictures use a shared square-root
colour spacing, with values still labelled in m2/Hz/rad, to show weak lobes while
retaining the full peak range.

GFS-Wave input conventions were checked against station spectra and NOAA
Technical Note 329 (2016), section 2.6 and the field-output tables, using only
published data descriptions. Current GRIB directions are nautical FROM.
Operational station ASCII angles are nautical TOWARD radians, converted with
(180 + angle*180/pi) modulo 360. The older Cartesian description produced a
mirrored time series. Comparison with the independent GRIB bearing identified
the correct orientation; this is an archive check, not a model-skill adjustment.
The
older EMC product webpage describes an older TOWARD convention and cannot
override the archive check. Partition PTP is peak period despite legacy
WVPER/SWPER field labels. Partition direction is a partition mean, used as the
centre of the unimodal reconstruction. Public map arrows point TOWARD after
conversion from the FROM field. Variable values are never rotated into the
atmospheric grid frame; atmospheric source vectors must explicitly declare
their frame. GFS winds are earth east/north.

ETOPO depth is positive down from EGM2008, with dry cells zero. Interpolation
uses valid source ocean cells; missing partitions over wet source cells have
zero energy. Period/direction interpolation is weighted by partition Hs squared.
Bounded coast fill never treats land missing values as zero sea height.
The pixel-centred global DEM's seam uses the nearest edge within half a pixel.
Area is spherical lat-lon cell area or dx*dy/map_factor^2 for conformal mass
grids; arbitrary point grids supply cell areas and retain their original order.
