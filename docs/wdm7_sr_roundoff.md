# WDM7 surface frozen-fraction validation

A stock `mp_physics=26`, `use_theta_m=0` forecast could stop at step 19
with `microphysics SR is outside its validated range [0, 1]`. The WRF
expression can round above one in an all-frozen column. The validator had
no expression bound for WDM7. This change preserves the computed SR word
and gives it the bound implied by its arithmetic.

The public-domain WRF 4.6.1 source is pinned at
`d66e442fccc04111067e29274c9f9eaccc3cef28`.
`phys/module_mp_wdm7.F:1029-1066` and `kernels/wdm7.cu:763-802` form:

```
total = (((rain + snow) + graupel) + hail) + ice
snow_pair = snow + ice
scale(x) = ((x * delz / denr) * dtcld) * 1000
SR = ((sum(scale(snow_pair)) + sum(scale(graupel)))
      + sum(scale(hail))) / (sum(scale(total)) + 1e-12)
```

All fallout components are nonnegative. In exact arithmetic the numerator
is at most the denominator. The two sides associate their additions
differently. Hail adds one total-sum addition and one final-numerator
addition to WDM6's expression. Copying the WDM6 scaling count alone would
miss both changes.

Let binary32 unit roundoff be `u=2^-24`, let
`gamma(n)=n*u/(1-n*u)`, and let
`L=max(floor(float32(dt/120+0.5)),1)` be the native minor-loop count.
The numerator has one snow-pair addition, two final additions and one
quotient rounding. Each side has four scaling roundings per term and
`L-1` accumulating additions; the first addition to zero is exact.
The denominator's five-component total takes four additions. Dropping
its monotone positive epsilon only increases this upper bound:

```
B = (1+u)^4 * ((1+u)/(1-u))^4 * (1+gamma(L-1))
    / ((1-gamma(4)) * (1-gamma(L-1)))
```

FMA contraction can remove scaling roundings, so the bound also covers
the default fast arithmetic. Near-unity SR has normal-sized numerator
and denominator because the denominator contains the positive `1e-12`.

For `S=2^24` and `a=L-1`, the exact rational form is:

```
N = (S+1)^8 * (S-4)
D = S^3 * (S-1)^4 * (S-8) * (S-2*a)
B = N/D
max_ulps = floor((N-D)*S/(2*D))
```

The validator uses integer arithmetic to select the largest binary32
word no greater than this analytic bound. It refuses loop counts whose
bound is undefined or reaches two, where ULP(1) spacing changes. For the
60-second case `L=1`, the allowance is exactly 8 ULPs, upper word
`0x3f800008` or `1.00000095367431640625`. WDM6's corresponding analytic
bound admits 7 ULPs. This is an exact evaluation of a conservative
arithmetic bound, not a claim that every endpoint is attainable.

WSM7 already has the identical five-component/three-frozen-accumulator
expression, so WDM7 shares `_wsm7_sr_roundoff_limit`. Both hail-expression
members refresh their envelope when an adaptive step changes `L`.
No microphysics kernel or output value changes. The validator still
rejects negative values and the first binary32 word above its bound.

Regression checks use independent `Fraction` arithmetic for the operation
counts, accept the exact endpoint unchanged, reject its next word, and
exercise step changes in both directions. The real forecast, strict
native replay, retained capture hashes and map observations are recorded
in the lane report and its `skeptic2/` evidence directory.
