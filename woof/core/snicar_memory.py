"""Additional device allocations selected by SNICAR spectra and LAP.

Shapes use byte-sized elements, like the surrounding Noah-MP inventory.
Default G173, 480-bin, LAP-off allocations remain in the existing inventory.
No CUDA import or table transfer is needed to price a configuration.
"""


def option_cache_shapes(cfg):
    if cfg.sf_surface_physics != 4 or cfg.opt_alb != 3:
        return {}
    from woof.core.snow_papers import TABLE_WORDS, PACKET_STRIDE, LAP_TABLE_WORDS, LAP_SPECIES
    spectrum = int(cfg.snicar_solarspec_opt)
    five = int(cfg.snicar_bandnumber_opt) == 1
    # The digest-checked solar_spectra.bin has twelve cosine nodes.
    nodes = 12 if spectrum else 1
    rows = {}
    if five:
        rows["noahmp_lsm/snicar_band_optics"] = (2, TABLE_WORDS, 4)
        rows["noahmp_lsm/snicar_band_packets"] = (2, nodes, PACKET_STRIDE, 4)
    elif spectrum:
        rows["noahmp_lsm/snicar_spectrum_packet"] = (nodes, PACKET_STRIDE, 4)
    if cfg.snicar_lap_opt:
        rows["noahmp_lsm/snicar_lap_optics"] = (LAP_TABLE_WORDS, 4)
        rows["noahmp_lsm/snicar_lap_scavenging"] = (len(LAP_SPECIES), 4)
        if five:
            rows["noahmp_lsm/snicar_lap_band_optics"] = (2, LAP_TABLE_WORDS, 4)
    return rows


def option_transient_shapes(cfg):
    if cfg.sf_surface_physics != 4 or cfg.opt_alb != 3:
        return {}
    rows = {}
    columns = int(cfg.nx) * int(cfg.ny)
    if cfg.snicar_bandnumber_opt == 1:
        # Second beam outputs (6 + 28), and operands for merging both beams.
        rows["noahmp_lsm/snicar_beam_operands"] = (columns, 48, 4)
        # Conservative first-use bound for _five_band and _lap_five_band:
        # f64 table copies, extinction/scattering/moment arrays, reduction
        # operands, and the temporary cast packets. These builders run once
        # per device/options tuple. The bound is not a measured high water.
        rows["noahmp_lsm/snicar_band_construction"] = (32 * 1024 * 1024,)
    elif cfg.snicar_solarspec_opt:
        # Input cosine nodes and spectral weights coexist with the cached
        # concatenated packet on its first construction.
        rows["noahmp_lsm/snicar_spectrum_construction"] = (12, 961, 4)
    if cfg.snicar_lap_opt:
        # New state plus contiguous input state and deposition packets.
        rows["noahmp_lsm/snicar_lap_operands"] = (columns, 47 + 47 + 27, 4)
    return rows
