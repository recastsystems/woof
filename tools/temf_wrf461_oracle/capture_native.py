"""Instrument owned copies of the pinned native modules, without changing physics."""
from pathlib import Path
import json
import sys

root, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
carried3 = 'te_temf kh_temf km_temf shf_temf qf_temf uw_temf vw_temf wupd_temf mf_temf thup_temf qtup_temf qlup_temf cf3d_temf'.split()
carried2 = 'hd_temf lcl_temf hct_temf cfm_temf'.split()
rates = 'rublten rvblten rthblten rqvblten rqcblten'.split()
def v3(name):
    return f'{name}(its:cap_hi,kts:kte,jts)'
def v2(name):
    return f'{name}(its:cap_hi,jts)'
pbl_inputs = [(n, v3(n)) for n in 'u3d v3d th3d t3d qv3d qc3d p3d pi3d rho z'.split()]
pbl_inputs += [('p3di', 'p3di(its:cap_hi,kts:kte+1,jts)')]
pbl_inputs += [(n, v2(n)) for n in 'znt ht hfx qfx tsk fCor ust hpbl kpbl2d u10 v10 t2'.split()]
pbl_inputs += [(n, v3(n)) for n in carried3]
pbl_inputs += [(n, v2(n)) for n in carried2]
pbl_outputs = [(n, v3(n)) for n in rates + carried3]
pbl_outputs += [(n, v2(n)) for n in 'ust hpbl kpbl2d u10 v10 t2'.split() + carried2]
sfc_inputs = [(n, f'{n}(its:cap_hi,kts,jts)') for n in 'u3d v3d th3d qv3d p3d pi3d rho z'.split()]
sfc_inputs += [(n, v2(n)) for n in 'ht psfc znt mavail tsk fCor hd_temf hfx wm_temf'.split()]
sfc_inputs += [('te_temf', 'te_temf(its:cap_hi,kts,jts)')]
sfc_outputs = [(n, v2(n)) for n in 'hfx qfx lh ust wm_temf qsfc flhc flqc exch_temf chs chs2 cqs2 u10 v10 th2 t2 q2 qgh cpm'.split()]
sfc_outputs += [('te_temf', 'te_temf(its:cap_hi,kts,jts)')]
manifest = {}
for unit, routine, anchor, inputs, outputs in [
    ('module_bl_temf', 'temfpbl', '   integer :: j\n', pbl_inputs, pbl_outputs),
    ('module_sf_temfsfclay', 'temfsfclay', '      INTEGER ::  J\n', sfc_inputs, sfc_outputs),
]:
    text = (root/'phys'/f'{unit}.F').read_text()
    decl = ('\n      integer :: cap_unit, cap_hi\n'
            '      logical, save :: cap_done = .false.\n'
            '      logical :: cap_take\n')
    before = ('\n      cap_take = .not. cap_done .and. its == ids .and. jts == jds\n'
              '      if (cap_take) then\n'
              '        cap_hi = min(ite, its+7)\n'
              f"        open(newunit=cap_unit,file='{routine}-capture.bin', &\n"
              "          access='stream', form='unformatted', status='replace', convert='little_endian')\n"
              '        write(cap_unit) cap_hi-its+1, kte-kts+1\n')
    if routine == 'temfpbl':
        before += '        write(cap_unit) dt\n'
    before += ''.join(f'        write(cap_unit) {expr}\n' for _, expr in inputs)
    before += '      end if\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor+decl+before, 1)
    end = f'   END SUBROUTINE {routine}' if routine == 'temfsfclay' else '   end subroutine temfpbl'
    # The native PBL spelling differs by source generation only in indentation.
    import re
    match = re.search(rf'^\s*end subroutine {routine}\s*$', text, re.I|re.M)
    assert match
    after = '\n      if (cap_take) then\n'
    after += ''.join(f'        write(cap_unit) {expr}\n' for _, expr in outputs)
    after += '        close(cap_unit)\n        cap_done = .true.\n      end if\n'
    text = text[:match.start()] + after + text[match.start():]
    (out/f'{unit}.F').write_text(text)
    manifest[routine] = {'inputs': inputs, 'outputs': outputs,
                         'capture_columns': 8, 'native_physics_edits': 0}
(out/'capture-layout.json').write_text(json.dumps(manifest, indent=2)+'\n')
