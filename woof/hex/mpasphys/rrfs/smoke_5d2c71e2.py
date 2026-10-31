"""Pinned RRFS-SD subsystem at the shipping default pin: SMOKE 5d2c71e2 in RRFSx MPAS-Model 44c0849.

The device graph is the 9652bbf graph (smoke_9652bbf: its translation, column lowering, phases, launcher and
pool binding) with the pinned source delta applied as exact edits, each required to match once:

* SMOKE 9652bbf..5d2c71e2 (5 files): MPAS/mpas_smoke_wrapper.F90 (FENGSHA rdrag_in/ssm_in inputs; the TEMPO
  aerosol emission: nwfa2d, nifa2d, config_mp_aero_emission, the GOCART sea-salt call and mp_aero_emission),
  dust_fengsha_mod.F90 (rdrag, ssm and the fine-dust emission diagnostic), ssalt_mod.F90 (absent sea-salt
  species are not written), the new module_mp_aero_emissions.F90; MPAS/Makefile is build order only.
* MPAS-Model 1d6e370e..44c0849 src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F: rdrag/ssm_in replace
  uthres_sg_in/albedo_drag/feff/sep_in, e_bb_in enters as max(e_bb_in,0), and nwfa2d/nifa2d pass through the
  TEMPO aerosol-aware branch; registry.chemistry.xml adds config_mp_aero_emission (default false).

``// Lnnn`` comments inside unchanged code cite the 9652bbf wrapper / 1d6e370e driver lines they were translated
from; every line this delta adds cites its 5d2c71e2 / 44c0849 line with the pin named.
"""
from __future__ import annotations

import copy
import importlib.util
import re

from . import smoke_9652bbf as _base

FORK_PIN = '44c084944e17f7098668bef6d1e353d12e3c2d1a'
SMOKE_PIN = '5d2c71e2cf4083b6beeedbfa4d2091674b1556ea'
_W = 'SMOKE 5d2c71e2 MPAS/mpas_smoke_wrapper.F90'
_D = 'RRFSx 44c0849 mpas_atmphys_driver_smoke.F'


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'pinned SMOKE 5d2c71e2 delta anchor changed ({text.count(old)}): {old[:100]}')
    return text.replace(old, new, 1)


def _function(text, name):
    """(start, end) of a routine's definition (the forward declaration ends with ');')."""
    key = 'SDDEV void ' + name + '('
    start = text.index(key)
    start = text.index(key, start + 1)
    return start, text.index('\n}\n', start) + 3


def _in_function(text, name, edits):
    start, end = _function(text, name)
    body = text[start:end]
    for old, new in edits:
        body = _once(body, old, new)
    return text[:start] + body + text[end:]


# ----------------------------------------------------------------------------------------------- routine table
def _symbol(**fields):
    row = dict(type='float', dim=[], intent=None, optional=False, parameter=False, init=None, save=False,
               decl='', str_len='100')
    row.update(fields)
    return row


def _replace_args(row, old, new):
    args = row['args']
    at = args.index(old[0])
    if args[at:at + len(old)] != list(old):
        raise ValueError('pinned SMOKE argument run changed: ' + ','.join(old))
    row['args'] = args[:at] + list(new) + args[at + len(old):]
    for name in old:
        if name not in new:
            del row['symbols'][name]


def _sorted_args(row, remove, add):
    if row['args'] != sorted(row['args']):
        raise ValueError('pinned SMOKE driver slice arguments are no longer sorted')
    for name in remove:
        row['args'].remove(name)
        del row['symbols'][name]
    for name, symbol in add.items():
        row['symbols'][name] = symbol
    row['args'] = sorted(row['args'] + list(add))


def routines(base):
    """ROUTINES at 5d2c71e2 / 44c0849 (dummy order and declarations of the pinned sources)."""
    rows = copy.deepcopy(base)
    wrapper = rows['mpas_smoke_driver']
    optional_in = 'real(rkind),intent(in), dimension(ims:ime, jms:jme),optional  :: sandfrac_in, clayfrac_in, uthres_in, rdrag_in, ssm_in'
    _replace_args(wrapper, ('uthres_sg_in', 'albedo_drag_in', 'feff_in', 'sep_in'), ('rdrag_in', 'ssm_in'))
    for name in ('rdrag_in', 'ssm_in'):
        wrapper['symbols'][name] = _symbol(dim=['ims:ime', 'jms:jme'], intent='in', optional=True, decl=optional_in)
    _replace_args(wrapper, ('ktop_deep',), ('ktop_deep', 'nwfa2d', 'nifa2d', 'config_mp_aero_emission'))
    for name in ('nwfa2d', 'nifa2d'):
        wrapper['symbols'][name] = _symbol(
            dim=['ims:ime', 'jms:jme'], intent='inout', optional=True,
            decl='real(rkind),intent(inout), dimension(ims:ime, jms:jme),optional :: nwfa2d, nifa2d')
    wrapper['symbols']['config_mp_aero_emission'] = _symbol(
        type='bool', intent='in', decl='logical,intent(in)                :: config_mp_aero_emission')
    wrapper.update(first_line=34, last_line=686)

    dust = rows['gocart_dust_fengsha_driver']
    _replace_args(dust, ('uthr', 'uthr_sg', 'albedo_drag', 'feff', 'sep'), ('uthr', 'rdrag', 'ssm'))
    dust['symbols']['rdrag'] = _symbol(dim=['ims:ime', 'jms:jme'], intent='in',
                                       decl='real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: rdrag')
    dust['symbols']['ssm'] = _symbol(dim=['ims:ime', 'jms:jme'], intent='in',
                                     decl='real(rkind), dimension( ims:ime , jms:jme ), intent(in) :: ssm')
    dust.update(first_line=24, last_line=217)

    pool1 = lambda name: _symbol(dim=[':'], decl=f'real(rkind) :: {name}(:)')
    pool2 = lambda name: _symbol(dim=[':', ':'], decl=f'real(rkind) :: {name}(:,:)')
    tempo = dict(config_microp_scheme=_symbol(type='FStr', decl='character(100) :: config_microp_scheme'),
                 config_tempo_aerosolaware=_symbol(type='bool', decl='logical :: config_tempo_aerosolaware'),
                 nifa2d=pool1('nifa2d'), nifa2d_p=pool2('nifa2d_p'),
                 nwfa2d=pool1('nwfa2d'), nwfa2d_p=pool2('nwfa2d_p'))
    _sorted_args(rows['smoke_from_mpas_fields'],
                 ('albedo_drag', 'albedo_drag_p', 'feff', 'feff_p', 'sep_in', 'sep_in_p',
                  'uthres_sg_in', 'uthres_sg_in_p'),
                 dict(tempo, rdrag=pool1('rdrag'), rdrag_p=pool2('rdrag_p'),
                      ssm_in=pool1('ssm_in'), ssm_in_p=pool2('ssm_in_p')))
    _sorted_args(rows['smoke_to_mpas_fields'], (), copy.deepcopy(tempo))

    bounds = [name + suffix for name in 'ijk' for suffix in ('ds', 'de', 'ms', 'me', 'ts', 'te')]
    order = ['ids', 'ide', 'jds', 'jde', 'kds', 'kde', 'ims', 'ime', 'jms', 'jme', 'kms', 'kme',
             'its', 'ite', 'jts', 'jte', 'kts', 'kte']
    assert sorted(order) == sorted(bounds)
    aero = dict(module='module_mp_aero_emissions', source='module_mp_aero_emissions.F90', first_line=20,
                last_line=78, args=['em_dust', 'em_fire_oc', 'em_antho_oc', 'em_seas', 'dt', 'land', 'nwfa2d',
                                    'nifa2d', 'rri', 'dz8w', *order], symbols={})
    two = 'real(rkind),intent(in), dimension(ims:ime, jms:jme):: em_dust,em_fire_oc,em_antho_oc,em_seas,land'
    for name in ('em_dust', 'em_fire_oc', 'em_antho_oc', 'em_seas', 'land'):
        aero['symbols'][name] = _symbol(dim=['ims:ime', 'jms:jme'], intent='in', decl=two)
    aero['symbols']['dt'] = _symbol(intent='in', decl='real(rkind),intent(in) :: dt')
    for name in ('nwfa2d', 'nifa2d'):
        aero['symbols'][name] = _symbol(dim=['ims:ime', 'jms:jme'], intent='inout',
                                        decl='real(rkind),intent(inout), dimension(ims:ime, jms:jme):: nwfa2d,nifa2d')
    for name in ('rri', 'dz8w'):
        aero['symbols'][name] = _symbol(dim=['ims:ime', 'kms:kme', 'jms:jme'], intent='in',
                                        decl='real(rkind),intent(in), dimension(ims:ime, kms:kme, jms:jme):: rri,dz8w')
    for name in order:
        aero['symbols'][name] = _symbol(type='int', intent='in', decl='integer,intent(in):: ' + name)
    rows['mp_aero_emission'] = aero
    return rows


# ------------------------------------------------------------------------------------- generated declarations
def signature(name, row):
    """translate.signature for routines with no mutated scalar dummies."""
    args = ['State &state', 'Arena &arena']
    for arg in row['args']:
        symbol = row['symbols'][arg]
        args.append(('FArray<' + symbol['type'] + '>' if symbol['dim'] else symbol['type']) + ' ' + arg)
    args += ['bool has_' + arg for arg in row['args'] if row['symbols'][arg]['optional']]
    return 'SDDEV void ' + name + '(' + ','.join(args) + ')'


def entry(name, row):
    """assemble.py's exported launch entry for one routine (no scalar write-back)."""
    lines = []
    for index, arg in enumerate(row['args']):
        symbol = row['symbols'][arg]
        if symbol['dim']:
            lines.append(f'auto {arg} = sd_arg_{symbol["type"]}(args[{index}],{len(symbol["dim"])});')
        elif symbol['type'] == 'FStr':
            lines.append(f'FStr {arg}((char*)args[{index}].data,100);')
        elif symbol['type'] == 'bool':
            lines.append(f'bool {arg} = args[{index}].ivalue;')
        else:
            lines.append(f'{symbol["type"]} {arg} = args[{index}].{"fvalue" if symbol["type"] == "float" else "ivalue"};')
    actual = list(row['args']) + [f'bool(args[{index}].present)' for index, arg in enumerate(row['args'])
                                  if row['symbols'][arg]['optional']]
    return '\n'.join(['extern "C" SDKERNEL void sd_run_' + name + '(State *state,float *scratch,int capacity,'
                      'Argument *args) {SDGUARD;Arena arena{scratch,0,capacity};', *lines,
                      name + '(*state,arena,' + ','.join(actual) + ');', '}'])


def _redeclare(text, name, old, new):
    """Swap one routine's forward declaration, definition signature and launch entry."""
    old_signature, new_signature = signature(name, old), signature(name, new)
    text = _once(text, old_signature + ';', new_signature + ';')
    text = _once(text, old_signature + ' {', new_signature + ' {')
    return _once(text, entry(name, old) + '\n', entry(name, new) + '\n')


# ------------------------------------------------------------------------------------------ the device graph
_MP_AERO_EMISSION = f'''// SMOKE 5d2c71e2 module_mp_aero_emissions.F90:20-78 (new at this pin). fact_wfa/fact_ifa are constant
// expressions of named constants: GNU Fortran 15.2 folds them at compile time (strict and -O0 alike, 0x57007A45 and
// 0x5006B7F3, a development machine 2026-10-29); the words are kept, not re-derived through device exp/log.
SIGNATURE {{
ArenaScope scope(arena);
em_dust.bind(ims,ime,jms,jme);
em_fire_oc.bind(ims,ime,jms,jme);
em_antho_oc.bind(ims,ime,jms,jme);
em_seas.bind(ims,ime,jms,jme);
land.bind(ims,ime,jms,jme);
nwfa2d.bind(ims,ime,jms,jme);
nifa2d.bind(ims,ime,jms,jme);
rri.bind(ims,ime,kms,kme,jms,jme);
dz8w.bind(ims,ime,kms,kme,jms,jme);
auto daero_emis_wfa = arena.alloc<float>(ims,ime,jms,jme);
auto daero_emis_ifa = arena.alloc<float>(ims,ime,jms,jme);
float density_dust = 2.6e+3f;
float density_oc = 1.4e+3f;
float density_seasalt = 2.2e+3f;
float fact_wfa;
float fact_ifa;
int i;
int j;
sd_assign(nwfa2d,0.f,arena); // L48
sd_assign(nifa2d,0.f,arena); // L49
fact_wfa = __uint_as_float(0x57007A45U); // L51
fact_ifa = __uint_as_float(0x5006B7F3U); // L52
for (i=its;i<=ite;i+=1) {{ // L54
for (j=jts;j<=jte;j+=1) {{ // L55
daero_emis_wfa(i,j) = sd_add(sd_div(sd_add(em_fire_oc(i,j),em_antho_oc(i,j)),density_oc),sd_div(em_seas(i,j),density_seasalt)); // L56
daero_emis_ifa(i,j) = sd_div(em_dust(i,j),density_dust); // L57
daero_emis_wfa(i,j) = sd_div(sd_mul(sd_mul(daero_emis_wfa(i,j),fact_wfa),rri(i,1,j)),dz8w(i,1,j)); // L59
daero_emis_ifa(i,j) = sd_div(sd_mul(sd_mul(daero_emis_ifa(i,j),fact_ifa),rri(i,1,j)),dz8w(i,1,j)); // L60
nwfa2d(i,j) = daero_emis_wfa(i,j); // L62
nifa2d(i,j) = daero_emis_ifa(i,j); // L63
if ((land(i,j) == 1.f)) {{ // L66
nwfa2d(i,j) = sd_mul(nwfa2d(i,j),sd_sub(1.f,sd_div(sd_mul(0.10f,dt),86400.f))); // L67
nifa2d(i,j) = sd_mul(nifa2d(i,j),sd_sub(1.f,sd_div(sd_mul(0.10f,dt),86400.f))); // L68
}} else {{ // L69
nwfa2d(i,j) = sd_mul(nwfa2d(i,j),sd_sub(1.f,sd_div(sd_mul(0.05f,dt),86400.f))); // L70
nifa2d(i,j) = sd_mul(nifa2d(i,j),sd_sub(1.f,sd_div(sd_mul(0.05f,dt),86400.f))); // L71
}} // L72
nwfa2d(i,j) = sd_min(9999.E6f,nwfa2d(i,j)); // L73
nifa2d(i,j) = sd_min(9999.E6f,nifa2d(i,j)); // L74
}} // L75
}} // L76
}}
'''


def cuda_graph(text, old, new):
    """CUDA_SOURCE at 5d2c71e2 / 44c0849 from the 9652bbf graph."""
    for name in ('mpas_smoke_driver', 'gocart_dust_fengsha_driver', 'smoke_from_mpas_fields',
                 'smoke_to_mpas_fields'):
        text = _redeclare(text, name, old[name], new[name])
    aero = signature('mp_aero_emission', new['mp_aero_emission'])
    prep = signature('mpas_smoke_prep', old['mpas_smoke_prep']) + ';'
    text = _once(text, prep, aero + ';\n' + prep)

    # The wrapper (5d2c71e2 mpas_smoke_wrapper.F90:34-686).
    text = _in_function(text, 'mpas_smoke_driver', [
        ('uthres_sg_in.bind(ims,ime,jms,jme);\nalbedo_drag_in.bind(ims,ime,jms,jme);\nfeff_in.bind(ims,ime,jms,jme);\n'
         'sep_in.bind(ims,ime,jms,jme);\n',
         f'rdrag_in.bind(ims,ime,jms,jme); // {_W}:164\nssm_in.bind(ims,ime,jms,jme); // {_W}:164\n'),
        ('ktop_deep.bind(ims,ime,jms,jme);\n',
         f'ktop_deep.bind(ims,ime,jms,jme);\nnwfa2d.bind(ims,ime,jms,jme); // {_W}:128\n'
         f'nifa2d.bind(ims,ime,jms,jme); // {_W}:128\n'),
        ('auto dz8w_flip = arena.alloc<float>(ims,ime,kms,kme,jms,jme);\n',
         'auto dz8w_flip = arena.alloc<float>(ims,ime,kms,kme,jms,jme);\n'
         f'int num_e_ss_out = 2; // {_W}:261\n'
         f'int index_e_ss_out_ssalt_fine = 1; // {_W}:262\n'
         f'int index_e_ss_out_ssalt_coarse = 2; // {_W}:262\n'
         f'auto e_ss_out = arena.alloc<float>(ims,ime,kms,kme,jms,jme,1,num_e_ss_out); // {_W}:263\n'
         f'auto em_dust = arena.alloc<float>(ims,ime,jms,jme); // {_W}:264\n'
         f'auto em_fire_oc = arena.alloc<float>(ims,ime,jms,jme); // {_W}:264\n'
         f'auto em_antho_oc = arena.alloc<float>(ims,ime,jms,jme); // {_W}:264\n'
         f'auto em_seas = arena.alloc<float>(ims,ime,jms,jme); // {_W}:264\n'),
        ('sd_assign(uspdavg2d,0.f,arena); // L281\n',
         f'sd_assign(e_ss_out,0.f,arena); // {_W}:288\nsd_assign(em_dust,0.f,arena); // {_W}:289\n'
         f'sd_assign(em_seas,0.f,arena); // {_W}:290\nsd_assign(em_fire_oc,0.f,arena); // {_W}:291\n'
         f'sd_assign(em_antho_oc,0.f,arena); // {_W}:292\nsd_assign(uspdavg2d,0.f,arena); // L281\n'),
        ('if (do_mpas_dust) { // L511\n',
         f'if (config_mp_aero_emission) {{ // {_W}:523\n'
         'gocart_seasalt_driver(state,arena,dt,rri,t_phy,u_phy,v_phy,num_chem,chem,rho_phy,dz8w,u10,v10,ust,p8w,'
         'tskin,xland,xlat,xlong,area,g,e_ss_out,num_e_ss_out,index_e_ss_out_ssalt_fine,index_e_ss_out_ssalt_coarse,'
         'state.mpas_smoke_config.pi,state.mpas_smoke_config.num_emis_seas,state.mpas_smoke_config.seas_opt,'
         f'ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // {_W}:525\n'
         f'}} // {_W}:537\nif (do_mpas_dust) {{ // L511\n'),
        ('uthres_in,uthres_sg_in,albedo_drag_in,feff_in,sep_in,e_dust_out,',
         'uthres_in,rdrag_in,ssm_in,e_dust_out,'),
        ('loop15_continue: ; } loop15_exit: ; // L636\n}\n',
         'loop15_continue: ; } loop15_exit: ; // L636\n'
         f'if (((config_mp_aero_emission && has_nwfa2d) && has_nifa2d)) {{ // {_W}:665\n'
         f'for (j=jts;j<=jte;j+=1) {{ // {_W}:667\n'
         f'for (i=its;i<=ite;i+=1) {{ // {_W}:668\n'
         f'em_dust(i,j) = sd_mul(e_dust_out(i,kts,j,index_e_dust_out_dust_fine),0.05f); // {_W}:669\n'
         f'em_seas(i,j) = e_ss_out(i,kts,j,index_e_ss_out_ssalt_fine); // {_W}:670\n'
         f'em_fire_oc(i,j) = e_bb_out(i,kts,j,index_e_bb_out_smoke_fine); // {_W}:671\n'
         f'em_antho_oc(i,j) = e_ant_out(i,kts,j,index_e_ant_out_unspc_fine); // {_W}:672\n'
         f'em_antho_oc(i,j) = sd_min(sd_mul(sd_mul(em_antho_oc(i,j),0.2f),0.05f),0.002f); // {_W}:673\n'
         f'em_fire_oc(i,j) = sd_min(sd_mul(em_fire_oc(i,j),0.01f),0.05f); // {_W}:674\n'
         f'em_seas(i,j) = sd_mul(em_seas(i,j),0.05f); // {_W}:675\n'
         f'}} // {_W}:676\n}} // {_W}:677\n'
         'mp_aero_emission(state,arena,em_dust,em_fire_oc,em_antho_oc,em_seas,dt,xland,nwfa2d,nifa2d,rri,dz8w,'
         f'ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte); // {_W}:679\n'
         f'}} // {_W}:684\n}}\n'),
    ])
    start, end = _function(text, 'mpas_smoke_driver')
    text = text[:end] + _MP_AERO_EMISSION.replace('SIGNATURE', aero) + text[end:]

    # FENGSHA (5d2c71e2 dust_fengsha_mod.F90:24-217).
    fengsha = 'SMOKE 5d2c71e2 dust_fengsha_mod.F90'
    text = _in_function(text, 'gocart_dust_fengsha_driver', [
        ('uthr_sg.bind(ims,ime,jms,jme);\nalbedo_drag.bind(ims,ime,jms,jme);\nfeff.bind(ims,ime,jms,jme);\n'
         'sep.bind(ims,ime,jms,jme);\n',
         f'rdrag.bind(ims,ime,jms,jme); // {fengsha}:57\nssm.bind(ims,ime,jms,jme); // {fengsha}:49\n'),
        ('erodtot = sep(i,j); // L137\n', f'erodtot = ssm(i,j); // {fengsha}:136\n'),
        ('if ((feff(i,j) > 1.e-10f)) { // L189\n', f'if ((rdrag(i,j) > 1.e-10f)) {{ // {fengsha}:188\n'),
        ('r = float(feff(i,j)); // L190\n', f'r = float(rdrag(i,j)); // {fengsha}:189\n'),
        ('e_dust_out(i,kts,j,index_e_dust_out_dust_fine) = reason_nodust; // L211\n',
         f'e_dust_out(i,kts,j,index_e_dust_out_dust_fine) = bems(1); // {fengsha}:210\n'),
    ])

    # GOCART sea salt (5d2c71e2 ssalt_mod.F90:135-140): absent species (index -1) are not written.
    salt = 'SMOKE 5d2c71e2 ssalt_mod.F90'
    text = _in_function(text, 'gocart_seasalt_driver', [
        ('chem(i,kts,j,state.mpas_smoke_init.p_ssalt_fine) = ',
         'if (((state.mpas_smoke_init.p_ssalt_fine != -1) && (state.mpas_smoke_init.p_ssalt_coarse != -1))) { '
         f'// {salt}:135\nchem(i,kts,j,state.mpas_smoke_init.p_ssalt_fine) = '),
        ('converi)); // L137\n', f'converi)); // L137\n}} // {salt}:140\n'),
    ])

    # The fork driver's boundaries (44c0849 mpas_atmphys_driver_smoke.F).
    tempo_in = (f'if (((config_microp_scheme == FStr("mp_tempo")) && config_tempo_aerosolaware)) {{ // {_D}:712\n'
                f'nifa2d_p(i,j) = nifa2d(i); // {_D}:713\nnwfa2d_p(i,j) = nwfa2d(i); // {_D}:714\n}} // {_D}:715\n')
    text = _in_function(text, 'smoke_from_mpas_fields', [
        ('e_bb_in_p(i,k,j,n) = e_bb_in(n,k,i); // L531\n',
         f'e_bb_in_p(i,k,j,n) = sd_max(e_bb_in(n,k,i),0.f); // {_D}:547\n'),
        ('mavail_p(i,j) = mavail(i); // L695\n', 'mavail_p(i,j) = mavail(i); // L695\n' + tempo_in),
        ('uthres_sg_in_p(i,j) = uthres_sg_in(i); // L707\nalbedo_drag_p(i,j) = albedo_drag(i); // L708\n'
         'feff_p(i,j) = feff(i); // L709\nsep_in_p(i,j) = sep_in(i); // L710\n',
         f'rdrag_p(i,j) = rdrag(i); // {_D}:727\nssm_in_p(i,j) = ssm_in(i); // {_D}:728\n'),
    ])
    text = _in_function(text, 'smoke_to_mpas_fields', [
        ('} // L991\nloop25_continue:',
         '} // L991\n'
         f'if (((config_microp_scheme == FStr("mp_tempo")) && config_tempo_aerosolaware)) {{ // {_D}:1020\n'
         f'nifa2d(i) = nifa2d_p(i,j); // {_D}:1021\nnwfa2d(i) = nwfa2d_p(i,j); // {_D}:1022\n}} // {_D}:1023\n'
         'loop25_continue:'),
    ])
    return text


def driver_bindings(base):
    """DRIVER_BINDINGS at 44c0849 (mpas_atmphys_driver_smoke.F:35-1335, sha256 ff42cdcd...)."""
    rows = copy.deepcopy(base)
    rows['source_sha256'] = 'ff42cdcd2de6c6c7ba2f91d34c818c3746204614e9a83041280dcb66bc06fa1d'
    pools = rows['pool_bindings']
    config = lambda name: {'kind': 'config', 'pool': 'configs', 'field': name, 'local': name, 'time_level': None}
    array = lambda name: {'kind': 'array', 'pool': 'diag_physics', 'field': name, 'local': name, 'time_level': None}
    removed = {'uthres_sg_in', 'albedo_drag', 'feff', 'sep_in'}
    source = pools['smoke_from_mpas']
    if sum(row['field'] in removed for row in source) != 4:
        raise ValueError('pinned smoke_from_MPAS dust getters changed')
    at = next(index for index, row in enumerate(source) if row['field'] in removed)
    source[:] = [row for row in source if row['field'] not in removed]
    source[at:at] = [array('rdrag'), array('ssm_in')]
    tempo = [config('config_microp_scheme'), config('config_tempo_aerosolaware'), array('nifa2d'), array('nwfa2d')]
    source.extend(copy.deepcopy(tempo))
    pools['smoke_to_mpas'].extend(copy.deepcopy(tempo))
    pools['allocate_smoke'].extend([config('config_microp_scheme'), config('config_tempo_aerosolaware')])
    pools['driver_smoke'].append(config('config_mp_aero_emission'))
    wrapper = rows['wrapper_bindings']
    for name in ('uthres_sg_in', 'albedo_drag_in', 'feff_in', 'sep_in'):
        del wrapper[name]
    wrapper.update(rdrag_in='rdrag_p', ssm_in='ssm_in_p', nwfa2d='nwfa2d_p', nifa2d='nifa2d_p',
                   config_mp_aero_emission='config_mp_aero_emission')
    allocations = rows['allocations']
    for name in ('uthres_sg_in_p', 'albedo_drag_p', 'sep_in_p', 'feff_p'):
        del allocations[name]
    allocations.update(rdrag_p='ims:ime,jms:jme', ssm_in_p='ims:ime,jms:jme',
                       nwfa2d_p='ims:ime,jms:jme', nifa2d_p='ims:ime,jms:jme')
    return rows


def chemistry_schemas(base):
    """registry.chemistry.xml at 44c0849 (sha256 269240d8...): the delta's options and dust fields."""
    schemas = copy.deepcopy(base)
    atmosphere = schemas['registry']['atmosphere']
    atmosphere['sha256'] = '269240d893f86b2c68722f727657d8bfe7aad33f508c0af8e09bf9f861306f9f'

    def walk(node):
        yield node
        for child in node['children']:
            yield from walk(child)
    nodes = list(walk(atmosphere['tree']))
    # 44c0849 registry.chemistry.xml:195-198 closes the chemistry record (the one holding config_rwc_scheme).
    record = next(n for n in nodes if n['tag'] == 'nml_record'
                  and any(c['attributes'].get('name') == 'config_rwc_scheme' for c in n['children']))
    record['children'].append({'tag': 'nml_option', 'children': [], 'attributes': {
        'name': 'config_mp_aero_emission', 'type': 'logical', 'default_value': 'false', 'in_defaults': 'false',
        'units': '-', 'description': 'Flag that controls aerosol emissions for TEMPO microphysics',
        'possible_values': '.true. or .false.'}})
    renames = {'albedo_drag': 'rdrag', 'albedo_drag_m_in': 'rdrag_m_in', 'sep_in': 'ssm_in'}
    dropped = {'uthres_sg_in', 'feff_m_in'}
    for node in nodes:
        node['children'] = [c for c in node['children'] if c['attributes'].get('name') not in dropped]
        for child in node['children']:
            name = child['attributes'].get('name')
            if name in renames:
                child['attributes']['name'] = renames[name]
    return schemas


def source_files(base):
    """Pinned source provenance at 44c0849 / 5d2c71e2."""
    pins = {'ssalt_mod.F90': ('aaeb529fa9209873c6387d963123ce985c742b18032ff4f80a2a28cb62f933c0', 324),
            'dust_fengsha_mod.F90': ('91e98b0da09513215eb933b9ae06d00f90a6e1e3faaa374239ce693957f619cd', 657),
            'mpas_smoke_wrapper.F90': ('061d25c470776bcd986f1f8cb1c7f8b8dc9407bf1aa4c023225a44a0215a3366', 886)}
    rows = []
    for row in copy.deepcopy(base):
        name = row['path'].rsplit('/', 1)[-1]
        if name in pins:
            row['sha256'], row['lines'] = pins[name]
        rows.append(row)
    rows.append({'path': 'src/core_atmosphere/physics/physics_noaa/SMOKE/module_mp_aero_emissions.F90',
                 'sha256': 'cdee9220b69355fbca9a9ee012d43ff44e8498790cca2717e85741bdedfb00f1', 'lines': 81})
    rows.append({'path': 'src/core_atmosphere/physics/mpas_atmphys_driver_smoke.F',
                 'sha256': 'ff42cdcd2de6c6c7ba2f91d34c818c3746204614e9a83041280dcb66bc06fa1d', 'lines': 1335})
    return rows


def _load():
    """A separate instance of the 9652bbf graph module whose pinned tables become the 5d2c71e2 ones.

    Its functions read the tables from their own module globals at call time, so the launcher, column lowering,
    pool binding and storage transport are the 9652bbf code with this pin's tables.
    """
    spec = importlib.util.spec_from_file_location(__name__ + '_graph', _base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module.ROUTINES
    new = routines(old)
    module.CUDA_SOURCE = cuda_graph(module.CUDA_SOURCE, old, new)
    module.ROUTINES = new
    module.DRIVER_BINDINGS = driver_bindings(module.DRIVER_BINDINGS)
    module.CHEMISTRY_SCHEMAS = chemistry_schemas(module.CHEMISTRY_SCHEMAS)
    module.SOURCE_FILES = source_files(module.SOURCE_FILES)
    module.FORK_PIN, module.SMOKE_PIN = FORK_PIN, SMOKE_PIN
    # Full-grid automatic arrays: the 9652bbf 28, the wrapper's e_ss_out and em_* (4), mp_aero_emission's 2.
    module.COLUMN_AUTOMATIC_ARRAYS = 28 + 5 + 2
    module.DUST_INPUT_FIELDS = ('sandfrac_in', 'clayfrac_in', 'uthres_in', 'rdrag', 'ssm_in')
    return module


graph = _load()


def __getattr__(name):
    return getattr(graph, name)
