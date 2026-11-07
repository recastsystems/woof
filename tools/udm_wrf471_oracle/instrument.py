"""Capture UDM's real two-dimensional calls and native initialized state."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "udm_declarations", ROOT / "tools/thompson_generation_oracle/translate.py")
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)
source, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=True)
module = {}
for n, line in t.statements(source):
    if line == "contains":
        break
    decl = t.declaration(line)
    if decl:
        module.update((v.name, v) for v in decl)
names = sorted(module)
(out / "module-schema.json").write_text(json.dumps({n: vars(module[n]) for n in names}, indent=2) + "\n")
scalars = "delt g cpd cpv rd rv t0c ep1 ep2 qmin xls xlv0 xlf0 den0 denr cliq cice psat".split()
bounds = "ids ide jds jde kds kde ims ime jms jme kms kme its ite jts jte kts kte lat udm_call".split()
arrays = {
    "t1": "its:ite,kts:kte", "q1": "its:ite,kts:kte",
    "qci1": "its:ite,kts:kte,1:2", "qrs1": "its:ite,kts:kte,1:4",
    "ncr1": "its:ite,kts:kte,1:3", "den1": "its:ite,kts:kte",
    "p1": "its:ite,kts:kte", "delz1": "its:ite,kts:kte",
    "slimsk": "its:ite",
}
surfaces = "rain rainncv sr snow snowncv graupel graupelncv hail hailncv".split()
outputs = ["t1", "q1", "qci1", "qrs1", "ncr1"]
def stream(suffix, data):
    s = f"""   IF (udm_capture) THEN
   WRITE(udm_file,'(A,A,I4.4,A,I6.6,A,I6.6,A)') &
     TRIM(udm_dir),'/udm-s',udm_call,'-i',its,'-j',lat,'{suffix}'
   OPEN(NEWUNIT=udm_unit,FILE=TRIM(udm_file),ACCESS='STREAM', &
     FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
"""
    if suffix == ".input":
        s += "   WRITE(udm_unit) " + ",".join(bounds) + "\n"
        s += "   WRITE(udm_unit) " + ",".join(scalars) + "\n"
    for name in data:
        s += f"   WRITE(udm_unit) {name}({arrays[name]})\n"
    for name in surfaces:
        s += f"   WRITE(udm_unit) {name}(its:ite)\n"
    return s + "   CLOSE(udm_unit)\n   END IF\n"
text = source.read_text()
marker = "   idim = ite-its+1"
assert text.count(marker) == 1
pre = """   INTEGER :: udm_unit
   INTEGER, SAVE :: udm_call=0
   LOGICAL :: udm_capture
   LOGICAL, SAVE :: udm_dumped=.FALSE.
   CHARACTER(LEN=512) :: udm_dir,udm_file
   CALL GET_ENVIRONMENT_VARIABLE('MISC_UDM_CAPTURE_DIR',udm_dir)
   IF (LEN_TRIM(udm_dir)>0.AND..NOT.udm_dumped) THEN
   WRITE(udm_file,'(A,A,I6.6,A,I6.6,A)') &
      TRIM(udm_dir),'/udm-i',its,'-j',lat,'-module.bin'
   OPEN(NEWUNIT=udm_unit,FILE=TRIM(udm_file),ACCESS='STREAM', &
     FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='LITTLE_ENDIAN')
"""
pre += "".join(f"   WRITE(udm_unit) {n}\n" for n in names)
pre += """   CLOSE(udm_unit)
   udm_dumped=.TRUE.
   END IF
   IF (lat==jts) udm_call=udm_call+1
   udm_capture=LEN_TRIM(udm_dir)>0.AND.lat==jts &
     .AND.(udm_call==1.OR.udm_call==2.OR.udm_call==30.OR.udm_call==100)
"""
text = text.replace(marker, pre + stream(".input", list(arrays)) + marker)
marker = "end subroutine udm2d"
assert text.count(marker) == 1
text = text.replace(marker, stream(".output", outputs) + marker)
(out / "column-schema.json").write_text(json.dumps(dict(
    header_i4=bounds, header_f4=scalars, input_arrays=arrays,
    output_arrays={n: arrays[n] for n in outputs}, surfaces=surfaces), indent=2) + "\n")
(out / source.name).write_text(text)
