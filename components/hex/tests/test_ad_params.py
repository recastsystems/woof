# SPDX-License-Identifier: Apache-2.0
"""woof.hex.ad.params / carrier / the forward override: the constants table, the runner files and the source rewrites.

Breakage each test prevents is named in its docstring. CPU only.
"""
from __future__ import annotations

import json
import struct

import pytest

from woof.hex.ad import carrier, params
from woof.hex.mpasphys import ad_params_override as override


def test_defaults_are_the_source_literals_bit_for_bit():
    """A default that is not the literal's float32 would make the carrier forward differ from the shipped one."""
    for c in params.CONSTANTS:
        assert params.float32_from_bits(c.bits) == params.float32_literal(c.literal), c.name
    assert params.BY_NAME["coare30_charnock"].bits == "0x3c343958"
    assert params.BY_NAME["mynnsfc_zil_kczil"].bits == "0x3ecccccd"
    assert params.BY_NAME["tofd_beljaars_a1"].bits == "0x398b8a44"
    assert params.BY_NAME["mesoscale_gwd_xlinv"].bits == "0x37144f83"
    assert params.BY_NAME["flow_blocking_cd"].bits == "0x3949539c"
    assert params.BY_NAME["mynn_qke_dissipation_b1"].bits == "0x41c00000"


def test_table_rows_name_units_and_scheme():
    rows = params.table()
    assert [r["name"] for r in rows] == [c.name for c in params.CONSTANTS]
    assert all(r["units"] and r["scheme"] and r["upstream"] for r in rows)


def test_carrier_document_and_runner_arguments(tmp_path):
    """The runner reads family, slot and p0_bits; a moved value needs --params-unchecked or the checked forward refuses."""
    doc = params.carrier_document(["flow_blocking_cd", "coare30_charnock"])
    assert [(r["name"], r["family"], r["slot"]) for r in doc["params"]] == [("coare30_charnock", "mynnsfc", 0),
                                                                               ("flow_blocking_cd", "ugwp", 2)]
    assert not params.moved(doc)
    f = tmp_path / "p.json"
    params.write_carrier_document(f, ["coare30_charnock", "flow_blocking_cd"])
    args = params.runner_arguments(f, ["a.json", "b.json"])
    assert args[args.index("--param-tl") + 1] == "1e-12,1e-12"
    assert "--params-unchecked" not in args
    params.write_carrier_document(f, values={"coare30_charnock": 0.012})
    assert "--params-unchecked" in params.runner_arguments(f, ["a.json"], tangent_scale=None)
    with pytest.raises(ValueError):
        params.carrier_document(["not_a_constant"])


def test_mean_objective_words_follow_the_allocation_map():
    """Allocation ids belong to a recording; the objective must come from that recording's map, never a fixed id."""
    amap = {"fields": {"theta_m": {"canonical": {"alloc": 368, "offset": 0, "shape": [59, 2562]}},
                       "u": {"canonical": {"alloc": 377, "offset": 5437440, "shape": [59, 7680]}}}}
    o = params.mean_objective(amap, "theta_m", range(3, 5), [6, 73])
    assert o["terms"] == [[368, 2 * 2562 + 6, 0.25], [368, 2 * 2562 + 73, 0.25],
                          [368, 3 * 2562 + 6, 0.25], [368, 3 * 2562 + 73, 0.25]]
    o = params.mean_objective({"fields": {"t": {"allocation": 9, "offset": 8, "shape": [2, 3]}}}, "t", [2], None)
    assert [t[1] for t in o["terms"]] == [2 + 3, 2 + 4, 2 + 5]
    with pytest.raises(KeyError):
        params.mean_objective(amap, "qv", [1], None)


def test_gradients_from_receipt():
    r = {"param_gradients": [{"name": "coare30_charnock", "p0": 0.011, "dJdp": 2.0, "dJdlnp": 0.022}],
         "param_tl": {"all": [{"name": "J1", "tl_dJ": 1.0}]},
         "param_dot": {"tl_dJ": 1.0, "adjoint_sum_c_dJdlnp": 1.0, "rel": 0.0, "pass": True},
         "param_gradients_more": [{"name": "J2", "file": "j2.json", "param_gradients": [], "param_dot": None}]}
    g = params.gradients_from_receipt(r)
    assert g[0]["objective"] == "J1" and g[0]["rows"][0]["dJ_dlnp"] == 0.022 and g[1]["objective"] == "J2"
    assert params.param_dot(r)[0]["pass"] is True


P4_SYNTH = ("czc = __fadd_rn(0x1.6872b00000000p-7f,__fmul_rn(0x1.cac0840000000p-8f, x));\n"
            "m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,sf_mynn_sfcflux_land,lsm);\n"
            "m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,0,lsm,lsm_ruc,true);\n"
            + carrier.P4_MARK + "\n"
            "czc = __fadd_rn(0x1.6872b00000000p-7f,__fmul_rn(0x1.cac0840000000p-8f, x));\n"
            "adx_m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,sf_mynn_sfcflux_land,lsm);\n"
            "adx_m_land_zilitinkevich_1995(zntstoch,zt,zq,restar,ust,0x1.99999a0000000p-2f,0,lsm,lsm_ruc,true);\n"
            "__device__ __forceinline__ void p4_sweep(const int2 *ab, const double2 *w, double *adj, int cnt, long long stride, int t) {\n"
            "        if (p.x <= -2) continue;\n"
            "p4_sweep(ab, w, adj, cnt, stride, t);\n"
            + carrier.P4_ENTRIES + "\n")


def test_carry_p4_rewrites_every_site_once():
    """The forward reads p4_parf (literal by default), the derivative p4_parA, the sweep collects leaves 8 + k."""
    out = carrier.carry_p4(P4_SYNTH, "mynnsfc")
    verb, deriv = out.split(carrier.P4_MARK)
    assert verb.count("p4_parf(0, 0x1.6872b00000000p-7f)") == 1
    assert verb.count("p4_parf(1, 0x1.99999a0000000p-2f)") == 2
    assert deriv.count("p4_parA(0, 0x1.6872b00000000p-7f)") == 1 and deriv.count("p4_parA(1, 0x1.99999a0000000p-2f)") == 2
    assert "p4_par_bar[(long long)(-10 - p.x) * P4_PAR_COLS + c] += a;" in deriv
    only = carrier.carry_p4(P4_SYNTH, "mynnsfc", ["coare30_charnock"])
    assert "p4_parf(1," not in only


def test_carry_p4_refuses_a_changed_source():
    """A source that no longer carries a pattern must stop generation, not ship a carrier without that constant."""
    with pytest.raises(carrier.CarrierError):
        carrier.carry_p4(P4_SYNTH.replace("0x1.6872b00000000p-7f", "0x1.0p-6f"), "mynnsfc")


def test_override_is_a_no_op_when_unset(monkeypatch):
    """Gradients off: with the variable unset no source string may change."""
    monkeypatch.delenv(override.ENV, raising=False)
    ns = {"SRC": "xlinv=coefm*8.84e-6f; taufb[1]=0.5f*1.92e-4f*rhobar a1=0.00026615161f*(var_temp*var_temp)"}
    before = dict(ns)
    override.patch_module(ns, "ugwp")
    assert ns == before


def test_override_replaces_the_carried_occurrence(monkeypatch, tmp_path):
    f = tmp_path / "p.json"
    params.write_carrier_document(f, values={"mesoscale_gwd_xlinv": 1.25e-5, "mynn_qke_dissipation_b1": 21.5})
    monkeypatch.setenv(override.ENV, str(f))
    ns = {"SRC": "xlinv=coefm*8.84e-6f; other", "lower": "xlinv=coefm*8.84e-6f;"}
    override.patch_module(ns, "ugwp")
    v = struct.unpack("<f", struct.pack("<f", 1.25e-5))[0]
    assert ns["SRC"] == f"xlinv=coefm*{float.hex(v)}f; other" and ns["lower"] == "xlinv=coefm*8.84e-6f;"
    edmf = {"K": "b1l = r6_mul(r6_mul(b1, 0.25f), r6_add(el(((k) + (1))), el(k)));"}
    override.patch_module(edmf, "mynnedmf")
    assert edmf["K"] == f"b1l = r6_mul(r6_mul({float.hex(21.5)}f, 0.25f), r6_add(el(((k) + (1))), el(k)));"
    with pytest.raises(RuntimeError):
        override.patch_module({"SRC": "no pattern here"}, "ugwp")
    with pytest.raises(RuntimeError):
        override.refuse_family("mynnedmf", "another MYNN-EDMF version")
    override.refuse_family("mynnsfc", "MYNN-SFC v6.1.0")   # nothing of that family asked: no refusal


def test_prepare_writes_the_runner_inputs(tmp_path):
    """The door's one call: parameter file + objectives by field name + the runner arguments that read them."""
    amap = {"fields": {"theta_m": {"canonical": {"alloc": 368, "offset": 0, "shape": [59, 4]}}}}
    lat, lon = [20.0, 21.0, -40.0, 60.0], [-26.0, -27.0, 100.0, 0.0]
    args = params.prepare(tmp_path, amap, ["theta_m:1-8", "theta_m:3-12:20.4,-26.2,1500"],
                          names=["coare30_charnock"], lat_deg=lat, lon_deg=lon)
    o0 = json.loads((tmp_path / "objective-0.json").read_text())
    o1 = json.loads((tmp_path / "objective-1.json").read_text())
    assert len(o0["terms"]) == 8 * 4 and o1["cells"] == [0, 1] and len(o1["terms"]) == 10 * 2
    assert args[:2] == ["--params", str(tmp_path / "params.json")]
    with pytest.raises(ValueError):
        params.prepare(tmp_path, amap, ["theta_m:1-8:0,0,10"])
    assert params.parse_objective_spec("qv:2-3") == {"field": "qv", "levels": [2, 3], "region": None}


def test_column_gradients_reads_the_runner_dump(tmp_path):
    (tmp_path / "pcol-o1-flow_blocking_cd.f64").write_bytes(struct.pack("<4d", 1.0, -2.0, 0.0, 9.0))
    assert params.column_gradients(tmp_path, 3) == {(1, "flow_blocking_cd"): [1.0, -2.0, 0.0]}
