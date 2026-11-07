"""Write woof/core/kernels/cammgmp_words.cuh from the oracle's words.txt.

words.txt is what tools/cammgmp_wrf461_oracle/dump_words.c read out of the
linked WRF objects after CAM_INIT and CAMMGMP_INIT (module_cam_physconst.F,
module_cam_wv_saturation.F gestbl, module_cam_mp_cldwat2m_micro.F
ini_micro, module_cam_mp_microp_aero.F ini_microp_aero,
module_cam_mp_ndrop.F activate_init).  Each word becomes one #define as a
hexadecimal float, so no decimal conversion stands between WRF's value and
the port; tests/test_cammgmp_wrf461_parity.py re-reads the committed
words fixture and fails if the header drifts from it.

The modal-aerosol tables CAM_INIT published (module_data_cam_mam_aero.F
values through modal_aero_initialize_phys) come from the oracle's
helpers.bin, which run_cammgmp.F90 wrote from the public *_mp arrays.

usage: python gen_words.py words.txt helpers.bin OUT.cuh
"""

from __future__ import annotations

import re
import struct
import sys

#: (header macro, words.txt key) -- only the words the port reads.
WORDS = [
    ("MG_G", "cldwat2m_micro.g"), ("MG_R", "cldwat2m_micro.r"),
    ("MG_RV", "cldwat2m_micro.rv"), ("MG_CPP", "cldwat2m_micro.cpp"),
    ("MG_RHOW", "cldwat2m_micro.rhow"), ("MG_XXLV", "cldwat2m_micro.xxlv"),
    ("MG_XLF", "cldwat2m_micro.xlf"), ("MG_XXLS", "cldwat2m_micro.xxls"),
    ("MG_RHOSN", "cldwat2m_micro.rhosn"), ("MG_RHOI", "cldwat2m_micro.rhoi"),
    ("MG_AC", "cldwat2m_micro.ac"), ("MG_BC", "cldwat2m_micro.bc"),
    ("MG_AS", "cldwat2m_micro.as"), ("MG_BS", "cldwat2m_micro.bs"),
    ("MG_AI", "cldwat2m_micro.ai"), ("MG_BI", "cldwat2m_micro.bi"),
    ("MG_AR", "cldwat2m_micro.ar"), ("MG_BR", "cldwat2m_micro.br"),
    ("MG_CI", "cldwat2m_micro.ci"), ("MG_DI", "cldwat2m_micro.di"),
    ("MG_CS", "cldwat2m_micro.cs"), ("MG_DS", "cldwat2m_micro.ds"),
    ("MG_F1S", "cldwat2m_micro.f1s"), ("MG_F2S", "cldwat2m_micro.f2s"),
    ("MG_EII", "cldwat2m_micro.eii"), ("MG_ECR", "cldwat2m_micro.ecr"),
    ("MG_F1R", "cldwat2m_micro.f1r"), ("MG_F2R", "cldwat2m_micro.f2r"),
    ("MG_DCS", "cldwat2m_micro.dcs"), ("MG_QSMALL", "cldwat2m_micro.qsmall"),
    ("MG_BIMM", "cldwat2m_micro.bimm"), ("MG_AIMM", "cldwat2m_micro.aimm"),
    ("MG_RHOSU", "cldwat2m_micro.rhosu"), ("MG_MI0", "cldwat2m_micro.mi0"),
    ("MG_QCVAR", "cldwat2m_micro.qcvar"), ("MG_PI", "cldwat2m_micro.pi"),
] + [(f"MG_CONS{i}", f"cldwat2m_micro.cons{i}") for i in range(1, 29) if i != 26] + [
    ("MG_LAMMINI", "cldwat2m_micro.lammini"),
    ("MG_LAMMAXI", "cldwat2m_micro.lammaxi"),
    ("MG_LAMMINR", "cldwat2m_micro.lamminr"),
    ("MG_LAMMAXR", "cldwat2m_micro.lammaxr"),
    ("MG_LAMMINS", "cldwat2m_micro.lammins"),
    ("MG_LAMMAXS", "cldwat2m_micro.lammaxs"),
    ("ND_T0", "ndrop.t0"), ("ND_ATEN", "ndrop.aten"),
    ("ND_ALOG2", "ndrop.alog2"), ("ND_ALOG3", "ndrop.alog3"),
    ("ND_ALOGATEN", "ndrop.alogaten"), ("ND_TWOTHIRD", "ndrop.twothird"),
    ("ND_SIXTH", "ndrop.sixth"), ("ND_SQ2", "ndrop.sq2"),
    ("ND_SQPI", "ndrop.sqpi"), ("ND_PI", "ndrop.pi"),
] + [(f"ND_{name.upper()}{m}", f"ndrop.{name}({m})")
     for name in ("npv", "alogsig", "exp45logsig", "argfactor", "f1", "f2")
     for m in (1, 2, 3)] + [
    ("MA_RN_DST1", "microp_aero.rn_dst1"), ("MA_RN_DST2", "microp_aero.rn_dst2"),
    ("MA_RN_DST3", "microp_aero.rn_dst3"), ("MA_RN_DST4", "microp_aero.rn_dst4"),
    ("MA_QSMALL", "microp_aero.qsmall"), ("MA_R", "microp_aero.r"),
    ("WV_TMIN", "wv_saturation.tmin"), ("WV_EPSQS", "wv_saturation.epsqs"),
    ("WV_RGASV", "wv_saturation.rgasv"), ("WV_HLATV", "wv_saturation.hlatv"),
    ("WV_CP", "wv_saturation.cp"), ("WV_TMELT", "wv_saturation.tmelt"),
    ("PC_GRAVIT", "physconst.gravit"), ("PC_RAIR", "physconst.rair"),
    ("PC_TMELT", "physconst.tmelt"), ("PC_EPSILO", "physconst.epsilo"),
    ("PC_RH2O", "physconst.rh2o"),
]


def read_words(path):
    out = {}
    for line in open(path, encoding="ascii"):
        name, word = line.split()
        out[name] = struct.unpack("<d", int(word, 16).to_bytes(8, "little"))[0]
    return out


MODAL = [("MD_VOLTONUMB", "voltonumb"), ("MD_VOLTONUMBLO", "voltonumblo"),
         ("MD_VOLTONUMBHI", "voltonumbhi"), ("MD_SIGMAG", "sigmag")]


def modal_words(helpers_path):
    """{macro: (value, key)} for the modal tables in helpers.bin."""
    from pathlib import Path
    data = Path(helpers_path).read_bytes()
    nt, ne, ng = struct.unpack_from("<3i", data, 0)
    pos = 12 + 8 * (4 * nt + 2 * ne + 2 * ng)
    (nmode,) = struct.unpack_from("<i", data, pos)
    pos += 4
    tables = {}
    for name in ("sigmag", "dgnum", "voltonumb", "voltonumblo", "voltonumbhi",
                 "alnsg"):
        tables[name] = struct.unpack_from(f"<{nmode}d", data, pos)
        pos += 8 * nmode
    tables["specdens"] = struct.unpack_from("<3d", data, pos)
    pos += 24
    tables["spechygro"] = struct.unpack_from("<3d", data, pos)
    out = {}
    for macro, name in MODAL:
        for m in range(nmode):
            out[f"{macro}{m + 1}"] = (tables[name][m], f"modal_aero_data.{name}_amode_mp({m + 1})")
    out["MD_SPECDENS"] = (tables["specdens"][0], "modal_aero_data.specdens_amode_mp(1)")
    out["MD_SPECHYGRO"] = (tables["spechygro"][0], "modal_aero_data.spechygro_mp(1)")
    if len(set(tables["specdens"])) != 1 or len(set(tables["spechygro"])) != 1:
        raise SystemExit("specdens/spechygro are no longer one value per type")
    return out


def header_text(words, modal=None) -> str:
    lines = [
        "// cammgmp_words.cuh -- GENERATED by tools/cammgmp_wrf461_oracle/"
        "gen_words.py; do not edit.",
        "// The run-time words of the WRF v4.6.1 CAMMGMP modules, read out of",
        "// the linked WRF objects after CAM_INIT and CAMMGMP_INIT by the",
        "// column oracle (dump_words.c).  Hexadecimal floats: no decimal",
        "// conversion stands between WRF's value and the port.",
        "#ifndef CAMMGMP_WORDS_CUH",
        "#define CAMMGMP_WORDS_CUH",
    ]
    for macro, key in WORDS:
        lines.append(f"#define {macro} {words[key].hex()}  // {key}")
    for macro, (value, key) in (modal or {}).items():
        lines.append(f"#define {macro} {value.hex()}  // {key}")
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def parse_header(text):
    """macro -> float, for the drift test."""
    pat = re.compile(r"^#define (\w+) (\S+)  // (\S+)$")
    out = {}
    for line in text.splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = (float.fromhex(m.group(2)), m.group(3))
    return out


def main():
    words = read_words(sys.argv[1])
    modal = modal_words(sys.argv[2])
    open(sys.argv[3], "w", encoding="ascii", newline="\n").write(
        header_text(words, modal))


if __name__ == "__main__":
    main()
