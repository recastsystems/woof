"""Capture WRF 4.6.1 SMS-3DTKE arguments without changing its equations.

Run against a private copy of dyn_em/module_diffusion_em.F. One tile per
MPI rank is required: call counters then identify actual forecast steps.
"""
from pathlib import Path
import re
import sys

ROUTINES = ('nonlocal_flux', 'free_atmos_length', 'meso_length_scale',
            'tke_km', 'tke_rhs', 'vertical_diffusion_implicit')
FLAGS_I = 'km_opt diff_opt bl_pbl_physics isfflx use_theta_m mix_isotropic sfs_opt m_opt'.split()
FLAGS_R = 'c_k c_s tke_heat_flux tke_drag_coefficient mix_upper_bound'.split()
FLAGS_L = 'mix_full_fields periodic_x periodic_y open_xs open_xe open_ys open_ye specified nested'.split()


def statements(source):
    out = []
    part = ''
    for line in source.splitlines():
        line = line.split('!')[0].strip().lower()
        if not line: continue
        part += ' ' + line.lstrip('&').rstrip('&')
        if not line.endswith('&'):
            out.append(part.strip())
            part = ''
    return out


def patch(src):
    for routine in ROUTINES:
        m = re.search(r'(?ims)^\s*subroutine\s+' + routine + r'\s*\(.*?^\s*end subroutine\s+' + routine + r'\b', src)
        assert m, routine
        body = m.group()
        marker = r'(?im)^\s*call tke_shear\(' if routine == 'tke_rhs' else r'(?im)^\s*ktf\s*='
        first = re.search(marker, body).start()
        decl = statements(body[:first])
        signature = decl[0]
        args = set(signature[signature.index('(')+1:signature.rindex(')')].replace(' ', '').split(','))
        fields = []
        for stmt in decl[1:]:
            if '::' not in stmt: continue
            attrs, names = stmt.split('::', 1)
            kind = attrs.split(',')[0].strip()
            dim = re.search(r'dimension\s*\(([^)]+)\)', attrs)
            dims = [] if dim is None else dim[1].replace(' ', '').split(',')
            for name in names.replace(' ', '').split(','):
                if name not in args or name == 'config_flags': continue
                assert kind in ('integer', 'real', 'logical'), (routine, stmt)
                fields.append((name, kind, dims))
        assert set(n for n, _, _ in fields) == args - {'config_flags'}, (routine, args-set(n for n,_,_ in fields))

        def dump(phase):
            lines = [f"IF (cap_want(km5_step)) THEN", "BLOCK", "INTEGER :: km5_species", "CHARACTER(len=16) :: km5_name", f"CALL cap_open('{routine}','{phase}',km5_step,its,jts)"]
            for name in ('p_qv', 'p_qns', 'p_qnr', 'p_qng', 'p_qnh', 'p_qt', 'p_qvolg', 'param_first_scalar'):
                lines.append(f"CALL cap_is('{name}',{name})")
            for name in FLAGS_I: lines.append(f"CALL cap_is('{name}',config_flags%{name})")
            for name in FLAGS_R: lines.append(f"CALL cap_rs('{name}',config_flags%{name})")
            for name in FLAGS_L: lines.append(f"CALL cap_is('{name}',MERGE(1,0,config_flags%{name}))")
            for name, kind, dims in fields:
                if not dims:
                    fun = 'rs' if kind == 'real' else 'is'
                    val = name if kind != 'logical' else f'MERGE(1,0,{name})'
                    lines.append(f"CALL cap_{fun}('{name}',{val})")
                elif len(dims) == 1:
                    assert kind == 'real'
                    lo,hi = dims[0].split(':')
                    lines += [f"CALL cap_hdr('{name}',3,1,1,{lo},{hi},1,1)", f"WRITE(cap_u) {name}"]
                elif len(dims) == 2:
                    fun = 'r2' if kind == 'real' else 'i2'
                    assert dims == ['ims:ime','jms:jme']
                    lines.append(f"CALL cap_{fun}('{name}',{name},ims,ime,jms,jme,ims,ime,jms,jme)")
                elif len(dims) in (3,4):
                    assert dims[:3] == ['ims:ime','kms:kme','jms:jme']
                    if len(dims) == 4:
                        lines += [f'DO km5_species=1,{dims[3]}', f"WRITE(km5_name,'(a,i2.2)') '{name}_',km5_species", f"CALL cap_r3(km5_name,{name}(:,:,:,km5_species),ims,ime,kms,kme,jms,jme,ims,ime,kms,kme,jms,jme)", 'ENDDO']
                    else: lines.append(f"CALL cap_r3('{name}',{name},ims,ime,kms,kme,jms,jme,ims,ime,kms,kme,jms,jme)")
                else: raise AssertionError((name,dims))
            lines += ['CALL cap_close()', 'END BLOCK', 'ENDIF']
            return '\n'.join(lines)+'\n'
        end = re.search(r'(?im)^\s*end subroutine', body).start()
        body = body[:end] + '\n' + dump('out') + body[end:]
        body = body[:first] + '\nINTEGER, SAVE :: km5_step=0\nkm5_step=km5_step+1\n' + dump('in') + body[first:]
        src = src[:m.start()] + body + src[m.end():]
    src = src.replace('    MODULE module_diffusion_em', '    MODULE module_diffusion_em\n    USE module_cap_io')
    helper = (Path(__file__).parents[1]/'myj_wrf461_capture/module_cap_io.f90').read_text()
    return helper + '\n' + src

if __name__ == '__main__':
    Path(sys.argv[2]).write_text(patch(Path(sys.argv[1]).read_text()))
