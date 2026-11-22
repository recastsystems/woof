"""WOOF-ArWen's surface and scalar contract for the resident MPAS LES closure."""

from .species_row import SpeciesDeclaration, ENERGY, species_row_for_scheme as species_row_for_selector


def active(config):
    return getattr(config, "config_les_model", "none") != "none"


def model_row(config):
    row = species_row_for_selector(config.config_microp_scheme)
    if getattr(config, "config_les_model", "none") == "prognostic_1.5_order":
        return row.with_extras((SpeciesDeclaration(
            "tke", ENERGY, True, "tke",
            "specific subgrid turbulent kinetic energy (m2 s-2)"),))
    return row


def validate(config):
    # Host admission describes the launcher the stack will construct. The
    # device driver's own validation still requires that actual launcher.
    from .mixing_v842_les import LesDissipationV842
    config.validate(**({"les_dissipation": LesDissipationV842()}
                       if active(config) else {}))
