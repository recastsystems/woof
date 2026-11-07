"""Observe complete native RCON driver calls without changing their arithmetic."""
from pathlib import Path
import json
import sys

source, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
bounds = "ids ide jds jde kds kde ims ime jms jme kms kme its ite jts jte kts kte itimestep".split()
volume = "qv qc qr qi qs qg ni nr th pii p w dz refl_10cm re_cloud re_ice re_snow".split()
optional_volume = "nc nwfa nifa nbca qb ng".split()
surface = "rainnc rainncv cloudnc sr".split()
optional_surface = "snownc snowncv graupelnc graupelncv nwfa2d nifa2d nbca2d".split()
optional = optional_volume + optional_surface + ["aer_init_opt", "wif_input_opt", "diagflag", "ke_diag", "do_radar_ref"]
written_volume = "qv qc qr qi qs qg ni nr th refl_10cm re_cloud re_ice re_snow".split()
written_surface = "snownc snowncv graupelnc graupelncv".split()
v_slice = "its:MIN(ite,ide-1),kts:kte,jts:MIN(jte,jde-1)"
s_slice = "its:MIN(ite,ide-1),jts:MIN(jte,jde-1)"

def write_array(name, dims, is_optional=False):
    value = f"      WRITE(rcd_unit) {name}({dims})\n"
    return f"      IF (PRESENT({name})) THEN\n" + value + "      END IF\n" if is_optional else value

def stream(suffix):
    text = f"""      IF (rcd_capture) THEN
      WRITE(rcd_file,'(A,A,I6.6,A,I6.6,A,I4.4,A)') &
         TRIM(rcd_dir),'/rcon-driver-i',its,'-j',jts,'-s',itimestep,'{suffix}'
      OPEN(NEWUNIT=rcd_unit,FILE=TRIM(rcd_file),ACCESS='STREAM', &
         FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
"""
    if suffix == ".input":
        text += "      WRITE(rcd_unit) " + ",".join(bounds[:12]) + "\n"
        text += "      WRITE(rcd_unit) " + ",".join(bounds[12:]) + "\n"
        text += "      WRITE(rcd_unit) dt_in,has_reqc,has_reqi,has_reqs\n"
        for name in optional:
            text += f"      WRITE(rcd_unit) MERGE(1,0,PRESENT({name}))\n"
        for name in optional[-5:]:
            text += f"      IF (PRESENT({name})) WRITE(rcd_unit) {name}\n"
    for name in volume if suffix == ".input" else written_volume:
        text += write_array(name, v_slice)
    for name in optional_volume:
        text += write_array(name, v_slice, True)
    for name in surface:
        text += write_array(name, s_slice)
    for name in optional_surface if suffix == ".input" else written_surface:
        text += write_array(name, s_slice, True)
    return text + "      CLOSE(rcd_unit)\n      END IF\n"

text = source.read_text()
start = text.index("      SUBROUTINE mp_rcon_driver(")
position = text.index("      i_start = its", start)
declarations = """      INTEGER :: rcd_unit
      LOGICAL :: rcd_capture
      CHARACTER(LEN=512) :: rcd_dir,rcd_file
      CALL GET_ENVIRONMENT_VARIABLE('MISC_RCON_DRIVER_CAPTURE_DIR',rcd_dir)
      rcd_capture=LEN_TRIM(rcd_dir)>0 .AND. &
        (itimestep==1.OR.itimestep==2.OR.itimestep==30.OR.itimestep==100)
"""
text = text[:position] + declarations + stream(".input") + text[position:]
end = text.index("      END SUBROUTINE mp_rcon_driver", start)
text = text[:end] + stream(".output") + text[end:]
(out / source.name).write_text(text)
(out / "driver-schema.json").write_text(json.dumps(dict(
    bounds_i4=bounds, scalar_dt_f4=True, flags_i4=["has_reqc", "has_reqi", "has_reqs"],
    present_i4=optional, optional_scalars=optional[-5:],
    volume=volume, optional_volume=optional_volume, surface=surface,
    optional_surface=optional_surface, written_volume=written_volume,
    written_optional_surface=written_surface,
    array_order="Fortran i,k,j over active tile; surface i,j; optional values only when present",
), indent=2) + "\n")
