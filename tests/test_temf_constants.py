"""The TEMF pair's constant table: one source, re-derived, pinned.

woof/core/temf_constants.py holds every non-exact constant of WRF v4.6.1's
TEMF modules as (name, Fortran text, float32 word).  These checks keep the
three copies of it one: the recorded word is the fold of the text (the
oracle build checks the same words against gfortran itself), the kernel's
TEMFC block is the generator's output byte for byte, and the table hash is
pinned.
"""
from __future__ import annotations

from pathlib import Path

from woof.core import temf_constants as tc

KERNEL = (Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"
          / "temf.cu")


def test_every_recorded_word_is_the_fold_of_its_fortran_text():
    derived = tc.derived_words()
    wrong = {name: (f"{word:08X}", f"{derived[name]:08X}")
             for name, _, word in tc.TEMF_CONSTANTS if derived[name] != word}
    assert wrong == {}


def test_the_kernel_carries_the_generated_table_verbatim():
    text = KERNEL.read_text(encoding="utf-8")
    assert tc.cuda_table() in text
    assert tc.splice_cuda(text) == text


def test_the_table_hash_is_pinned():
    assert tc.table_sha256() == tc.TEMF_CONSTANTS_SHA256


def test_packaged_table_manifest_and_noftz_loader_are_bound():
    from woof.table_assets import packaged_constant_table_manifest
    from woof.core.kernels import module_options, module_source
    manifest = packaged_constant_table_manifest()["temf"]
    assert manifest["sha256"] == tc.TEMF_CONSTANTS_SHA256
    assert manifest["words"] == len(tc.TEMF_CONSTANTS)
    from woof.wrf_exact import ENABLED
    assert module_options("temf") == (
        ("-std=c++17", "--ftz=false", "--fmad=false") if ENABLED else
        ("-std=c++17", "--ftz=false", "--fmad=true"))
    assert tc.cuda_table() in module_source("temf")


def test_names_are_unique_and_the_order_is_the_index():
    names = [name for name, _, _ in tc.TEMF_CONSTANTS]
    assert len(names) == len(set(names))
    block = tc.cuda_table()
    for index, name in enumerate(names):
        assert f"#define TC_{name} __uint_as_float(TEMFC[{index}])" in block


def test_a_literal_rounds_once_from_its_decimal():
    # The decimal is read exactly and rounded once to binary32.
    assert int(tc._literal("2.45e-6").view("uint32")) == 0x36246AAE
    assert int(tc._literal("0.1").view("uint32")) == 0x3DCCCCCD
    assert int(tc.fold("7.*R_D/2.", {"R_D": tc.F32(287.0)}).view(
        "uint32")) == 0x447B2000
