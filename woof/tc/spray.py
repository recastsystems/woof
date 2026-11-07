"""Spray parameter rows. Units are SI, including droplet radii.

Bao et al. (2000), doi:10.1175/1520-0493(2000)128<2190:NSOASI>2.0.CO;2,
equations 12--17. The wind cap applies to the spray multipliers, not to
the interfacial fluxes or model winds. Partition=1 is the complete
evaporation cooling limit; partition=0 is the limiting large-drop case.
"""
PARAMETERS = {
    "bao2000": dict(wind_cap=30., latent_factor=3.e-6, latent_power=4.4,
                    sensible_factor=2.7e-5, sensible_power=2.4),
    "barr2023": dict(c1=1.35, c2=.1116, c3=.719, c4=2.17, c5=.852,
                     source_strength=2.2, dissipation_factor=100.,
                     kolmogorov_constant=1.5, minimum_radius=2.e-6,
                     maximum_radius=.002, radius_bins=48,
                     max_iterations=24, flux_tolerance=.05),
}
