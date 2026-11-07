"""Native WRF-v4 tile parameter and conditional carrier inventory."""
PARAMETERS=("klat_fac","tdsmc_fac","td_dc","td_dcoef","td_radi","td_spac","td_ddrain","td_adepth","td_d","td_depth","drain_layer_opt")
FIELDS=("td_fraction","qtdrain","zwtxy")
def shapes(option,shape):
    return dict.fromkeys(FIELDS,shape) if int(option)>0 else {}
