"""Bind the selected WRF Held–Suarez forcing to resident RK momentum."""


def attach(backend, driver):
    binding = getattr(backend, "_held_suarez_momentum", None)
    if binding is None:
        return
    atmosphere = driver.atmosphere
    regional = getattr(driver, "regional_v841", None)
    binding.bind(atmosphere.mesh.cells_on_edge,
                 atmosphere.reference.pressure_base,
                 solved_edges=None if regional is None else int(regional.n_edges_solve))
    driver.held_suarez_momentum = binding
