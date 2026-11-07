"""Verify MPAS source citations and per-routine coverage against pinned v8.4.2.

Line numbers rot while continuing to look plausible. The campaign audit
resolves new MPAS citations inside the supplied pristine v8.4.2 source tree,
checks its immutable SHA manifest. --routine-coverage additionally requires
a citation on each routine; the default G0 mode is whole-file authentication.
Existing v8.2.3 citations retain their committed source-window pins. Unknown
release tags, missing files and invalid ranges refuse G0.

Comma continuations (``file.F:10-20,30-40``) and inherited ranges
(``file.F:10-20`` followed by ``:30-40``) are separate windows.

The historical v8.2.3 frozen-window audit is retained under --legacy-freeze.
The campaign audit validates the immutable v8.4.2 source manifest and checks
each cited file's bytes against it; it does not silently regenerate source pins.

Usage::

    python tools/check_mpas_citations.py
    python tools/check_mpas_citations.py --inventory
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "hexcore"
VENDOR_ROOT = ROOT / "vendor" / "MPAS_source_v8.2.3_group" / "MPAS-Model-8.2.3"
FREEZE_PATH = ROOT / "SOURCE_FREEZE.json"
ARCHIVE_PATH = ROOT / "vendor" / "MPAS_source_v8.2.3_group" / "MPAS-Model-v8.2.3.tar.gz"

EXPECTED_RELEASE = "v8.2.3"
EXPECTED_TAG_COMMIT = "ac3866c1e5b05f6d4f5bd41aeab7d3882bace514"
EXPECTED_ARCHIVE_SHA256 = (
    "bb3b02c30abffe9ff0318165b25724e6855fb69076fd89243f06a24e11912ee1"
)


@dataclass(frozen=True, slots=True)
class CitationPin:
    source_path: str
    first_line: int
    last_line: int
    window_sha256: str
    occurrences: int = 1


@dataclass(frozen=True, slots=True)
class CitationOccurrence:
    citation: str
    python_path: str
    python_line: int


# Bare aliases actually published by the port.  They are declarations, not
# guesses: any new ambiguous spelling is rejected until it receives a row.
ALIASES: Mapping[str, str] = {
    "Registry.xml": "src/core_atmosphere/Registry.xml",
    "core.F": "src/core_atmosphere/mpas_atm_core.F",
    "helpers.F": "src/operators/mpas_tracer_advection_helpers.F",
    "mono.F": "src/operators/mpas_tracer_advection_mono.F",
    "reconstruction.F": "src/operators/mpas_vector_reconstruction.F",
    "std.F": "src/operators/mpas_tracer_advection_std.F",
    "time_integration.F": ("src/core_atmosphere/dynamics/mpas_atm_time_integration.F"),
}


_RANGE = r"\d+(?:\s*-\s*\d+)?"
_FULL = re.compile(
    rf"(?P<file>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+"
    rf"\.(?:F90|f90|F|xml|c|h)):(?P<range>{_RANGE})"
    rf"(?P<continuations>(?:\s*,\s*{_RANGE})*)"
)
_CONTINUATION = re.compile(rf",\s*(?P<range>{_RANGE})")
_INHERITED = re.compile(rf"(?:``|\band\s+):(?P<range>{_RANGE})")


# citation -> immutable source-window pin.  Generated candidates are useful
# during an audit, but these values are deliberately committed and never
# constructed from the current source at check time.
PINS: Mapping[str, CitationPin] = {
    "MPAS-Model-v8.2.3/src/core_atmosphere/Registry.xml:1520-1532": CitationPin(
        "src/core_atmosphere/Registry.xml",
        1520,
        1532,
        "a3bc2c59ce940eec48f0df82b950c86973ec9cfb45f99a1b85daca3aea471a0e",
    ),
    "MPAS-Model-v8.2.3/src/core_atmosphere/Registry.xml:851-974": CitationPin(
        "src/core_atmosphere/Registry.xml",
        851,
        974,
        "fcdfae8fee7a5ed920ed9a24ac4caa103111ba787a5ee12cfaecf87408c15866",
    ),
    "core.F:1419-1436": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1419,
        1436,
        "0fc3fbb0bb1449353dbe68d9775c143364340c372bdfedef0fb4b79d273115bf",
    ),
    "helpers.F:315-722": CitationPin(
        "src/operators/mpas_tracer_advection_helpers.F",
        315,
        722,
        "91b049294a0800565b7cd9041c163e6ac16b7b3a262b13831101cd21a3927b57",
    ),
    "helpers.F:44-51": CitationPin(
        "src/operators/mpas_tracer_advection_helpers.F",
        44,
        51,
        "7505761ad2c2725193cbc475d173ae21e0f914a5eab0b3dc2a14be8ab760b5a4",
    ),
    "helpers.F:87-302": CitationPin(
        "src/operators/mpas_tracer_advection_helpers.F",
        87,
        302,
        "f2c3386d74024bedc68c4593adf2e419175d6179d3a9ec4574a4f36a859ac5d4",
    ),
    "mono.F:465-508": CitationPin(
        "src/operators/mpas_tracer_advection_mono.F",
        465,
        508,
        "46f233493352e43e2c19a3e9efdb235029175ff81a61b40e9ea8673abcd58507",
    ),
    "mpas_atm_core.F:1079-1134": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1079,
        1134,
        "2ea4fcafbb79fadd4bc1164fd0eca9cefcaebcb7cec476501aab8361194b55cc",
    ),
    "mpas_atm_core.F:1137-1203": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1137,
        1203,
        "fc12416c5ebf7dea0a7b816d3aeaf9af29f940c1aeeeb981ac82033b5be04dba",
    ),
    "mpas_atm_core.F:1172-1184": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1172,
        1184,
        "4435ffc2d608c39696f7ac9191d9d7af5ca65cbc1f7eadb6af36abf161226d5e",
    ),
    "mpas_atm_core.F:1186-1202": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1186,
        1202,
        "4d7fbbee55cc15d464030c432f8250741ef51d0630dd8c74c1b8bda4c4e444b3",
    ),
    "mpas_atm_core.F:1204-1213": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1204,
        1213,
        "1f849f8ccf8d3ef864d844d0d241d8af33c97a4872e73f916087b3ccbab5c1ee",
    ),
    "mpas_atm_core.F:1230-1268": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1230,
        1268,
        "315bb14f4c99b65b87b8a5ac596ddae50ef1604ee53c0e34a82d7296470c8ce3",
    ),
    "mpas_atm_core.F:1419-1437": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1419,
        1437,
        "25a253f498eb22f358bf8509b61b94ebf5f2655411987a865dcf6df83820ec23",
    ),
    "mpas_atm_core.F:177-207": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        177,
        207,
        "feb112899997e28f442de0124cdce9f8b1ebf5621b60f94cc3a922ef6d829e96",
        2,
    ),
    "mpas_atm_core.F:887-936": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        887,
        936,
        "c30b3490b1da3c2540cc8e9e250cb024189fbdf801594a30f7e0fd6c62178d80",
    ),
    "mpas_atm_time_integration.F:2089-2112": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        2089,
        2112,
        "d45e2b0540a275793948d08a7f305c5d761a09d8c59f1915a2580e788d94aa85",
    ),
    "mpas_atm_time_integration.F:1818-2523": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        1818,
        2523,
        "8ded742f931452d644cf7c88a088e892c7a1bb2d432e49820338408fe3561dd3",
    ),
    "mpas_atm_time_integration.F:2409-2417": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        2409,
        2417,
        "ff2e504f2d0fc50b70460ea77deb480233f6791bbf8e18a5a67e6acbe191cc71",
    ),
    "mpas_atm_time_integration.F:2532-2601": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        2532,
        2601,
        "99cd21cb7d88bdb7480d87c034ad9df66b4fca48cc234ad6ee477a7e568a528c",
    ),
    "mpas_atm_time_integration.F:2715-2904": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        2715,
        2904,
        "13bc30bd6923f47a9f9fbab44f34f6d55409ed0da1732479c68cdc3e0fc062bf",
    ),
    "mpas_atm_time_integration.F:2793-2903": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        2793,
        2903,
        "711d79700a7111602f7deba0c23bcf7320417ffcf35eb48b9f9636343992d3a3",
    ),
    "mpas_atm_time_integration.F:3108-3113": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        3108,
        3113,
        "eae39780d5c067ebef4a24f6dead426086af5d49f8e525902bbdd273c9410dde",
    ),
    "mpas_atm_time_integration.F:3049-4211": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        3049,
        4211,
        "4cc3657ada4cd69bb8751766a6d0f3eb76e63ffada3f6fa787f682897118e879",
    ),
    "mpas_atm_time_integration.F:3566-3571": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        3566,
        3571,
        "eae39780d5c067ebef4a24f6dead426086af5d49f8e525902bbdd273c9410dde",
    ),
    "mpas_atm_time_integration.F:4640-4701": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        4640,
        4701,
        "e8acd43d64a943c95038569304a9d9d22709f435b50852ef58d63d1333445332",
    ),
    "mpas_atm_time_integration.F:4822-4923": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        4822,
        4923,
        "c61aa7a77aa2789b6031206019ca2fab238bf33bb96c23fdac7774ca05e5f7db",
    ),
    "mpas_atm_time_integration.F:5071-5133": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5071,
        5133,
        "93e831878dfc0955a94e0f06d1ed91148bcdcbf78d388da254f9a3c49d16c96b",
    ),
    "mpas_atm_time_integration.F:5250-5304": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5250,
        5304,
        "130c81b7ffe219d315eecb94ddae8caa8f75942b54aca6d64aceb86dd15543d0",
    ),
    "mpas_atm_time_integration.F:5887-5930": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5887,
        5930,
        "279609c9e832748d11b9220efb8b4b36ef7c3295b17d0f480ed6464b65079b56",
    ),
    "mpas_atm_time_integration.F:638-1105": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        638,
        1105,
        "ad0e1ba9cb3e9ebb9fae3741726bfb5234dfab829135cd058265947a9ebd4af7",
    ),
    "mpas_atm_time_integration.F:638-1224": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        638,
        1224,
        "6acad252dc5bfbae8995081c2cd99b009fdb41be95b4f6bb608898f96340903d",
        2,
    ),
    "mpas_atm_time_integration.F:638-686": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        638,
        686,
        "cb5e32b3f6936efe39c53c1bf67a68d005b989213efb9063fa221b767f013720",
    ),
    "mpas_atmphys_driver.F:180-370": CitationPin(
        "src/core_atmosphere/physics/mpas_atmphys_driver.F",
        180,
        370,
        "490a1a3be3ef6b8f05f3ad3c7c8aec3fbae11a45a4a61f171b89c28ea8962d10",
    ),
    "mpas_atmphys_interface.F:222-546": CitationPin(
        "src/core_atmosphere/physics/mpas_atmphys_interface.F",
        222,
        546,
        "6172b660144314a58a0247d454e5aa24961468e62077178b43f20483a20b871a",
    ),
    "mpas_atmphys_todynamics.F:436-437": CitationPin(
        "src/core_atmosphere/physics/mpas_atmphys_todynamics.F",
        436,
        437,
        "ef7b0f25b40ae8a846927df4ee55ee63741317257e8ec85559662cd3df15a7e3",
    ),
    "mpas_atmphys_todynamics.F:65-456": CitationPin(
        "src/core_atmosphere/physics/mpas_atmphys_todynamics.F",
        65,
        456,
        "1a3ffe5d6e4b11171558c052fcc3beac1f1bc3e15cea9f9e7ea37e6495146e3a",
    ),
    "mpas_geometry_utils.F:239-296": CitationPin(
        "src/operators/mpas_geometry_utils.F",
        239,
        296,
        "85f261fa5c9a4c33a9fcbcc0acc90eaa23eb3c381630975483034281be5d2f3d",
    ),
    "mpas_init_atm_cases.F:2974-3331": CitationPin(
        "src/core_init_atmosphere/mpas_init_atm_cases.F",
        2974,
        3331,
        "060adce6bfe12f3306daa3a7c268c95701b73d164d34f610819e3330abd01d09",
    ),
    "mpas_rbf_interpolation.F:1079-1145": CitationPin(
        "src/operators/mpas_rbf_interpolation.F",
        1079,
        1145,
        "da750a8dbb0e27d337cfdce64635d0afc22d4723162fd71ca58ff0cbf1cd6542",
        2,
    ),
    "mpas_rbf_interpolation.F:1527-1559": CitationPin(
        "src/operators/mpas_rbf_interpolation.F",
        1527,
        1559,
        "c807bf87640a97f0dfd9a121b43d85b7ca01899cf6a30f178fea54201b974781",
        2,
    ),
    "mpas_rbf_interpolation.F:1670-1846": CitationPin(
        "src/operators/mpas_rbf_interpolation.F",
        1670,
        1846,
        "576babac9973e68e640011eb7268ec1094b9ddd873b57241f261de35a6d26155",
        2,
    ),
    "mpas_tracer_advection_helpers.F:64-73": CitationPin(
        "src/operators/mpas_tracer_advection_helpers.F",
        64,
        73,
        "b092d8e9f7246281bdbb860088c757f1c62f1f51074a01d02df2fae10c02a534",
    ),
    "mpas_vector_operations.F:113-121": CitationPin(
        "src/operators/mpas_vector_operations.F",
        113,
        121,
        "d7a2d8a753c5b973ce69514763f65886e128d92012102618829e002109d760b6",
    ),
    "mpas_vector_operations.F:136-209": CitationPin(
        "src/operators/mpas_vector_operations.F",
        136,
        209,
        "cab47f4c9f0c275340c875acc2b161f9c119a8c09b3b9a406a022197dde793eb",
    ),
    "mpas_vector_operations.F:223-289": CitationPin(
        "src/operators/mpas_vector_operations.F",
        223,
        289,
        "f7f54b463e22ebca81cecf25c0b8709d8edaf87613f8e0d86a31552c3621a237",
    ),
    "mpas_vector_operations.F:304-363": CitationPin(
        "src/operators/mpas_vector_operations.F",
        304,
        363,
        "3f861dfed902c1b1046738efe6a44872e2399df4436776eb0feeb2f15681df8e",
    ),
    "mpas_vector_operations.F:381-438": CitationPin(
        "src/operators/mpas_vector_operations.F",
        381,
        438,
        "7a05f5d572797fdf2f0945e3a48c423c73591e32f697f5ddad579888f864a3b7",
    ),
    "mpas_vector_operations.F:454-503": CitationPin(
        "src/operators/mpas_vector_operations.F",
        454,
        503,
        "174e2fa2dce2afb1da45344ef5b951bd521c8054aff063e6d96cb51b2b8154ff",
    ),
    "mpas_vector_operations.F:518-568": CitationPin(
        "src/operators/mpas_vector_operations.F",
        518,
        568,
        "fc1ce00aeffd12906fac47f4b0d86ca3a2a80c5a045cadb32294dd62244c7dd2",
    ),
    "mpas_vector_operations.F:583-631": CitationPin(
        "src/operators/mpas_vector_operations.F",
        583,
        631,
        "9f6663594e8e1a1aeaf643824c1437b5569619e2d35752c11bdd41cfd3d4418b",
    ),
    "mpas_vector_operations.F:652-771": CitationPin(
        "src/operators/mpas_vector_operations.F",
        652,
        771,
        "4f2344f1abbe8f992354a21c17ff25f04824bdd06b47642ac452a3e4d4001206",
    ),
    "mpas_vector_operations.F:788-842": CitationPin(
        "src/operators/mpas_vector_operations.F",
        788,
        842,
        "6450ef6b109032d872829ece53e0c2215accc2ea53d75220604ce3d6369637fd",
    ),
    "mpas_vector_operations.F:79-83": CitationPin(
        "src/operators/mpas_vector_operations.F",
        79,
        83,
        "a9dea23103728f137437f2c03d8491021ad6b21398edeabace8827990136591b",
    ),
    "mpas_vector_operations.F:861-889": CitationPin(
        "src/operators/mpas_vector_operations.F",
        861,
        889,
        "7e013f0c06f6b34808944897c75dfba820f6210ca9d7b4eab4be4d4cf564812e",
    ),
    "mpas_vector_operations.F:95-101": CitationPin(
        "src/operators/mpas_vector_operations.F",
        95,
        101,
        "d4078d39f43a57411a011626ef7efb65459de29d84d22b47a4f8d8a2803989e1",
    ),
    "mpas_vector_reconstruction.F:195-294": CitationPin(
        "src/operators/mpas_vector_reconstruction.F",
        195,
        294,
        "20954dc4865863bd7bf7e47d098949226032bc38f9afe8027091810e49f97652",
    ),
    "mpas_vector_reconstruction.F:309-408": CitationPin(
        "src/operators/mpas_vector_reconstruction.F",
        309,
        408,
        "60c961476806219f249b00bbbc7f3c57a2a1c103c1e7881ae6e0bdea19b450fe",
    ),
    "mpas_vector_reconstruction.F:51-181": CitationPin(
        "src/operators/mpas_vector_reconstruction.F",
        51,
        181,
        "627cce6b9cc4064ac9a90164f518f310e45a89be47faa8a4cdab76d5d5a9272d",
    ),
    "reconstruction.F:256-265": CitationPin(
        "src/operators/mpas_vector_reconstruction.F",
        256,
        265,
        "d04e9100b01cbd542c5ad1529c675a61098f7e46b8be4169e070119621d29463",
    ),
    "reconstruction.F:370-379": CitationPin(
        "src/operators/mpas_vector_reconstruction.F",
        370,
        379,
        "ee9aecea1deca3ed75219dce480dca2adbc327ecd814a0cc3acbe15938920281",
    ),
    "src/core_atmosphere/Registry.xml:851-974": CitationPin(
        "src/core_atmosphere/Registry.xml",
        851,
        974,
        "fcdfae8fee7a5ed920ed9a24ac4caa103111ba787a5ee12cfaecf87408c15866",
    ),
    "src/core_atmosphere/diagnostics/mpas_isobaric_diagnostics.F:632-640": CitationPin(
        "src/core_atmosphere/diagnostics/mpas_isobaric_diagnostics.F",
        632,
        640,
        "112e639ecc16fc419cf5fc9ea7584b87d4ebe1d3d162dab2904b505b1bb70f68",
    ),
    "src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4636-4638": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        4636,
        4638,
        "f97ec5c6c9503100b9e06b6f0d42ea236d69a83693a1e0e423c59735ffd13195",
    ),
    "src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5491-5799": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5491,
        5799,
        "a06efd0e71a734ab5853ae08c7518ba886195172bcc068cbc20de62e05cd15d7",
    ),
    "src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5561-5569": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5561,
        5569,
        "06d34f60e47cd3daaa4affa933804a8ab77fb550cfb19445247f24423e38b6e6",
    ),
    "src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5582-5593": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5582,
        5593,
        "ce1112201c420e29172f155b312cc63dda88e2af6bd802d8398210cf125a8261",
    ),
    "src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5582-5597": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5582,
        5597,
        "83a6a58b36b3df2b083ecf3bd47e6808a1e007b3ef7aa93bb548fa33df31af52",
    ),
    "src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5602-5617": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5602,
        5617,
        "c8f2fb77a6931c47af70b8b480da1cb434cd710499de980ec1d4ce6a2e1d9eb8",
    ),
    "src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5787-5794": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        5787,
        5794,
        "a81bae902d081ba1fa9898d0de1e92b12b756efeeb12d2cea70d148b26b3f8fb",
    ),
    "src/core_atmosphere/dynamics/mpas_atm_time_integration.F:638-1105": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        638,
        1105,
        "ad0e1ba9cb3e9ebb9fae3741726bfb5234dfab829135cd058265947a9ebd4af7",
    ),
    "src/core_atmosphere/mpas_atm_core.F:1137-1224": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1137,
        1224,
        "00620350ab37c2af7a341fa265069254bb7cf1b67168f4704609440dc34da332",
    ),
    "src/core_atmosphere/mpas_atm_core.F:1173-1185": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1173,
        1185,
        "d7ae8090f33d726704c12ab541c01a8429dae95e75d5e0644908bebf80ad0755",
    ),
    "src/core_atmosphere/mpas_atm_core.F:1187-1203": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1187,
        1203,
        "f389fe48a043ac2fc5798794ee326a8d1ca45f74cb14aa5dd1dd4863a13718d6",
    ),
    "src/core_atmosphere/mpas_atm_core.F:1205-1222": CitationPin(
        "src/core_atmosphere/mpas_atm_core.F",
        1205,
        1222,
        "1ed6c2a1c4d2249d6161e6367fcd4492170af136b40f124d74dbe1f2024301e7",
    ),
    "src/core_init_atmosphere/mpas_atmphys_functions.F:155-180": CitationPin(
        "src/core_init_atmosphere/mpas_atmphys_functions.F",
        155,
        180,
        "c5a127881c7f428b4352ff0e43e955241e54e3c69a2e556d6a14bfa603c6bed7",
    ),
    "src/core_init_atmosphere/mpas_init_atm_cases.F:2974-3331": CitationPin(
        "src/core_init_atmosphere/mpas_init_atm_cases.F",
        2974,
        3331,
        "060adce6bfe12f3306daa3a7c268c95701b73d164d34f610819e3330abd01d09",
    ),
    "src/core_init_atmosphere/mpas_init_atm_read_met.F:50-407": CitationPin(
        "src/core_init_atmosphere/mpas_init_atm_read_met.F",
        50,
        407,
        "c55d951765da89b2672d687111d630dd4ecb36e9420a2caf2789812840ff9e50",
    ),
    "src/framework/xml_stream_parser.c:1239-1325": CitationPin(
        "src/framework/xml_stream_parser.c",
        1239,
        1325,
        "6e3a8c7e0f4a1c7fad126782b49f1b22d23dba08814926cc8274930f755abfac",
    ),
    "std.F:273-309": CitationPin(
        "src/operators/mpas_tracer_advection_std.F",
        273,
        309,
        "780d86061f655150fdb298ee4c067718b27d207513bb71a5de767afd21f88555",
    ),
    "time_integration.F:3192-3203": CitationPin(
        "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
        3192,
        3203,
        "fe4eb60f9bcbd783a09ee61fe3adcbcdc2927d5d162cf25446667b2ac420cbc1",
    ),
}


def _parse_range(value: str) -> tuple[int, int]:
    first_text, separator, last_text = value.replace(" ", "").partition("-")
    first = int(first_text)
    return first, int(last_text) if separator else first


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def citation_occurrences(
    source_root: Path = SOURCE_ROOT,
) -> tuple[CitationOccurrence, ...]:
    """Return every explicit and continued vendored-source citation."""

    found: list[CitationOccurrence] = []
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(source_root.parent.parent).as_posix()
        full_matches = list(_FULL.finditer(text))
        for match in full_matches:
            cited_path = match.group("file")
            first, last = _parse_range(match.group("range"))
            found.append(
                CitationOccurrence(
                    f"{cited_path}:{first}-{last}"
                    if first != last
                    else f"{cited_path}:{first}",
                    relative,
                    _line_number(text, match.start()),
                )
            )
            for continuation in _CONTINUATION.finditer(match.group("continuations")):
                extra_first, extra_last = _parse_range(continuation.group("range"))
                suffix = (
                    f"{extra_first}-{extra_last}"
                    if extra_first != extra_last
                    else str(extra_first)
                )
                found.append(
                    CitationOccurrence(
                        f"{cited_path}:{suffix}",
                        relative,
                        _line_number(text, match.start()),
                    )
                )

        for inherited in _INHERITED.finditer(text):
            preceding = [
                match for match in full_matches if match.end() < inherited.start()
            ]
            if not preceding:
                continue
            parent = preceding[-1]
            between = text[parent.end() : inherited.start()]
            if len(between) > 500 or "\n\n" in between:
                continue
            first, last = _parse_range(inherited.group("range"))
            suffix = f"{first}-{last}" if first != last else str(first)
            found.append(
                CitationOccurrence(
                    f"{parent.group('file')}:{suffix}",
                    relative,
                    _line_number(text, inherited.start()),
                )
            )
    return tuple(
        sorted(
            found, key=lambda item: (item.citation, item.python_path, item.python_line)
        )
    )


def citations(
    source_root: Path = SOURCE_ROOT,
) -> dict[str, tuple[CitationOccurrence, ...]]:
    grouped: dict[str, list[CitationOccurrence]] = {}
    for occurrence in citation_occurrences(source_root):
        grouped.setdefault(occurrence.citation, []).append(occurrence)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _split_citation(citation: str) -> tuple[str, int, int]:
    cited_path, separator, range_text = citation.rpartition(":")
    if not separator or not cited_path:
        raise ValueError(f"invalid citation {citation!r}")
    first, last = _parse_range(range_text)
    return cited_path, first, last


def resolve_cited_path(cited_path: str, vendor_root: Path = VENDOR_ROOT) -> str:
    """Resolve a published spelling to one declared vendored source file."""

    if cited_path.startswith("MPAS-Model-v8.2.3/"):
        cited_path = cited_path.removeprefix("MPAS-Model-v8.2.3/")
    if cited_path.startswith("src/"):
        candidate = vendor_root / cited_path
        if not candidate.is_file():
            raise FileNotFoundError(cited_path)
        return cited_path
    if cited_path in ALIASES:
        candidate_path = ALIASES[cited_path]
        if not (vendor_root / candidate_path).is_file():
            raise FileNotFoundError(candidate_path)
        return candidate_path

    matches = sorted(
        path.relative_to(vendor_root).as_posix()
        for path in (vendor_root / "src").rglob(Path(cited_path).name)
        if path.is_file()
    )
    suffix_matches = [
        path
        for path in matches
        if path == cited_path or path.endswith("/" + cited_path)
    ]
    if len(suffix_matches) != 1:
        raise ValueError(
            f"{cited_path!r} resolves to {suffix_matches or matches}; add an ALIASES row"
        )
    return suffix_matches[0]


def source_window(
    source_path: str,
    first_line: int,
    last_line: int,
    vendor_root: Path = VENDOR_ROOT,
) -> str:
    path = vendor_root / source_path
    if not path.is_file():
        raise FileNotFoundError(source_path)
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not 1 <= first_line <= last_line <= len(lines):
        raise IndexError(
            f"{source_path} has {len(lines)} lines, not {first_line}-{last_line}"
        )
    return "\n".join(lines[first_line - 1 : last_line])


def window_sha256(window: str) -> str:
    return hashlib.sha256(window.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_freeze() -> list[str]:
    failures: list[str] = []
    try:
        freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        authority = freeze["authority"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return [f"cannot read SOURCE_FREEZE.json authority: {error}"]
    expected = {
        "release": EXPECTED_RELEASE,
        "tag_commit": EXPECTED_TAG_COMMIT,
        "sha256": EXPECTED_ARCHIVE_SHA256,
    }
    observed = {name: authority.get(name) for name in expected}
    if observed != expected:
        failures.append(f"SOURCE_FREEZE authority mismatch: {observed} != {expected}")
    if not ARCHIVE_PATH.is_file():
        failures.append(f"vendored v8.2.3 archive is missing: {ARCHIVE_PATH}")
    elif _sha256_file(ARCHIVE_PATH) != EXPECTED_ARCHIVE_SHA256:
        failures.append("vendored v8.2.3 archive SHA-256 mismatch")
    return failures


def check_pins(
    found: Mapping[str, Sequence[CitationOccurrence]],
    *,
    pins: Mapping[str, CitationPin] = PINS,
    vendor_root: Path = VENDOR_ROOT,
) -> list[str]:
    failures: list[str] = []
    for citation, occurrences in sorted(found.items()):
        pin = pins.get(citation)
        if pin is None:
            where = ", ".join(
                f"{item.python_path}:{item.python_line}" for item in occurrences
            )
            failures.append(f"{citation} has no citation pin (published at {where})")
            continue
        cited_path, first, last = _split_citation(citation)
        try:
            resolved = resolve_cited_path(cited_path, vendor_root)
        except (FileNotFoundError, ValueError) as error:
            failures.append(f"{citation} cannot resolve in vendored v8.2.3: {error}")
            continue
        if resolved != pin.source_path:
            failures.append(
                f"{citation} resolves to {resolved}, but its pin names {pin.source_path}"
            )
        if (first, last) != (pin.first_line, pin.last_line):
            failures.append(
                f"{citation} range disagrees with pin {pin.first_line}-{pin.last_line}"
            )
        if len(occurrences) != pin.occurrences:
            failures.append(
                f"{citation} occurs {len(occurrences)} times, expected "
                f"{pin.occurrences}"
            )
        try:
            window = source_window(
                pin.source_path,
                pin.first_line,
                pin.last_line,
                vendor_root,
            )
        except (FileNotFoundError, IndexError, UnicodeError) as error:
            failures.append(f"{citation} pin is missing or out of range: {error}")
            continue
        actual_digest = window_sha256(window)
        if actual_digest != pin.window_sha256:
            failures.append(
                f"{citation} source-window pin mismatch: "
                f"{actual_digest} != {pin.window_sha256}"
            )

    for citation in sorted(set(pins) - set(found)):
        failures.append(
            f"{citation} has a stale citation pin but no published citation"
        )
    return failures


def check(
    *,
    source_root: Path = SOURCE_ROOT,
    vendor_root: Path = VENDOR_ROOT,
    pins: Mapping[str, CitationPin] = PINS,
    verify_freeze: bool = True,
) -> list[str]:
    failures = _check_freeze() if verify_freeze else []
    failures.extend(
        check_pins(citations(source_root), pins=pins, vendor_root=vendor_root)
    )
    return failures


def candidate_inventory(
    *,
    source_root: Path = SOURCE_ROOT,
    vendor_root: Path = VENDOR_ROOT,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for citation, occurrences in citations(source_root).items():
        cited_path, first, last = _split_citation(citation)
        resolved = resolve_cited_path(cited_path, vendor_root)
        window = source_window(resolved, first, last, vendor_root)
        rows.append(
            {
                "citation": citation,
                "source_path": resolved,
                "first_line": first,
                "last_line": last,
                "window_sha256": window_sha256(window),
                "occurrences": len(occurrences),
                "published_at": [
                    f"{item.python_path}:{item.python_line}" for item in occurrences
                ],
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="print the resolved candidate inventory as JSON",
    )
    parser.add_argument("--routine-coverage", action="store_true",
                        help="also require a direct MPAS citation on every physics/dynamics/diagnostics routine")
    parser.add_argument("--mpas-source", type=Path,
                        default=Path(os.environ.get("MPAS_REF", str(ROOT / "vendor" / "MPAS-Model-v8.4.2"))))
    parser.add_argument("--legacy-freeze", action="store_true",
                        help="run the archived v8.2.3 frozen-window audit")
    arguments = parser.parse_args(argv)
    if arguments.inventory:
        if arguments.legacy_freeze:
            rows = candidate_inventory()
        else:
            rows = []
            for citation, occurrences in citations().items():
                cited, first, last = _split_citation(citation)
                if not is_mpas_citation(cited):
                    continue
                row = {"citation": citation, "occurrences": len(occurrences)}
                rrfs_errors = check_rrfs_window(citation, occurrences)
                if rrfs_errors is not None:
                    row.update(release="RRFS " + RRFS_COMMIT, source_path=RRFS_SOURCE_FILE,
                               source_sha256=RRFS_SOURCE_SHA256,
                               authority="committed historical RRFS whole-file pin")
                    if rrfs_errors:
                        row["error"] = "; ".join(rrfs_errors)
                    rows.append(row)
                    continue
                if citation in PINS:
                    pin = PINS[citation]
                    row.update(release="v8.2.3", source_path=pin.source_path,
                               window_sha256=pin.window_sha256,
                               authority="committed historical source-window pin")
                    errors = check_legacy_window(citation)
                    if errors:
                        row["error"] = "; ".join(errors)
                    rows.append(row)
                    continue
                try:
                    declared = declared_fork_window(cited, first, last, occurrences[0].python_path)
                    if declared is not None:
                        row.update(declared)
                        rows.append(row)
                        continue
                    if fork_citation_window(cited,first,last):
                        root=Path(os.environ["RRFS_REF"]) if os.environ.get("RRFS_REF") else RRFS_SOURCE_ROOT
                        resolved=resolve_cited_path(cited.removeprefix("MPAS-Model-rrfs/"),root)
                        row.update(release="rrfs-1d6e370e",source_path=resolved,
                            window_sha256=window_sha256(source_window(resolved,first,last,root)))
                        rows.append(row)
                        continue
                    if cited.startswith("MPAS-Model-") and not cited.startswith("MPAS-Model-v8.4.2/"):
                        raise ValueError("citation names an older MPAS release")
                    resolved = resolve_cited_path(cited.removeprefix("MPAS-Model-v8.4.2/"), arguments.mpas_source)
                    row.update(release="v8.4.2", source_path=resolved, window_sha256=window_sha256(
                        source_window(resolved, first, last, arguments.mpas_source)))
                except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as error:
                    row["error"] = str(error)
                rows.append(row)
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    failures = check() if arguments.legacy_freeze else check_campaign_citations(SOURCE_ROOT, arguments.mpas_source)
    if arguments.routine_coverage:
        failures.extend(check_routine_coverage(SOURCE_ROOT, arguments.mpas_source))
    for failure in failures:
        print(f"FAIL: {failure}")
    found = citations()
    print(
        f"{sum(len(items) for items in found.values())} citation occurrences, "
        f"{len(found)} unique ranges, {len(failures)} failures"
    )
    return 1 if failures else 0


SOURCE_MANIFEST_SHA256 = "5ec8c4c614f29a230fb1a38ced564a0cccb0d8029e8e6c056c170175b3251c8e"

# This existing adapter explicitly cites the workflow's RRFS commit, rather
# than MPAS v8.4.2. Authenticate that historical source; never reinterpret its
# out-of-range v8.4.2 line numbers as a v8.4.2 citation.
# MPAS_RRFS_REF names the pinned RRFS checkout; a pool host that exports MPAS_LANES keeps its
# references under ref/rrfs there; a public checkout looks in vendor/ like the v8.4.2 source.
RRFS_SOURCE_ROOT = Path(os.environ.get("MPAS_RRFS_REF") or (
    str(Path(os.environ["MPAS_LANES"]) / "ref/rrfs/MPAS-Model-rrfs") if os.environ.get("MPAS_LANES")
    else str(ROOT / "vendor" / "MPAS-Model-rrfs")))
RRFS_COMMIT = "1d6e370e7b43a332074809f170532a216c2212fb"
RRFS_CITATION = "mpas_atmphys_driver_convection.F:1152-1154"
RRFS_SOURCE_FILE = "src/core_atmosphere/physics/mpas_atmphys_driver_convection.F"
RRFS_SOURCE_SHA256 = "46b356413aa850ed8b906962d81f0d68ce1050528fc24265097cfe401c002cd5"


def check_rrfs_window(citation: str, occurrences: Sequence[CitationOccurrence]) -> list[str] | None:
    if citation != RRFS_CITATION or not all(
            Path(item.python_path).name == "cuda_ntiedtke_column_v841.py" for item in occurrences):
        return None
    try:
        path = RRFS_SOURCE_ROOT / RRFS_SOURCE_FILE
        if _sha256_file(path) != RRFS_SOURCE_SHA256:
            return [f"historical RRFS {RRFS_COMMIT} source SHA-256 mismatch"]
        source_window(RRFS_SOURCE_FILE, 1152, 1154, RRFS_SOURCE_ROOT)
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as error:
        return [f"historical RRFS {RRFS_COMMIT} source unavailable: {error}"]
    return []


def is_mpas_citation(cited: str) -> bool:
    basename = Path(cited).name
    # WRF's driver is cited by the retained ArWen seam. MPAS has its own
    # mpas_atmphys_driver_microphysics.F and does not contain this WRF file.
    # WRF's Noah-MP land surface (module_sf_noahmplsm.F) is cited by the ArWen
    # Noah-MP shdmax binding (cuda_arwen_physics_v841, hex 0.4.0); MPAS v8.4.2
    # carries the refactored physics_noahmp/ and does not contain this WRF file.
    # An explicitly MPAS-qualified spelling still fails source resolution.
    if cited in ("module_microphysics_driver.F", "module_sf_noahmplsm.F", "module_radiation_driver.F"):
        return False
    return (basename.startswith(("mpas_", "module_")) or basename == "Registry.xml" or
            cited in ALIASES or "MPAS-Model" in cited)



@lru_cache(maxsize=2)
def _fork_manifest(newer):
    return json.loads((ROOT / "src/hexcore/data/mpasphys-native" /
        ("mpas-source-rrfs-44c0849.json" if newer else "mpas-source-rrfs.json")).read_text())


@lru_cache(maxsize=128)
def _authenticated_fork_lines(target, expected):
    path = Path(target)
    if _sha256_file(path) != expected:
        raise ValueError("fork citation source SHA-256 differs from its committed manifest: " + target)
    return path.read_text().splitlines(keepends=True)


def declared_fork_window(cited: str, first: int, last: int, python_path: str):
    """Authenticate a fork citation with the committed source/submodule manifest.

    The generated newer MYNN and smoke modules declare their newer parent pin;
    the retained fork modules use the original parent. Bare references inside
    either fork belong to that fork, rather than the main MPAS source tree.
    """
    filename = Path(python_path).name
    fork = "/mpasphys/rrfs/" in python_path.replace("\\", "/")
    if not fork and not (filename == "forecast_v842.py" and cited == "Registry.xml") and not (
            filename == "bl_mynn_v842.py" and cited == "module_bl_mynnedmf.F90"):
        return None
    newer = (filename in {"bl_mynnedmf_c6d08709.py", "smoke_5d2c71e2.py", "smoke_resident.py"} or
             (filename == "stream_xml_rrfs.py" and cited == "mpas_output_timelevels.F"))
    manifest = _fork_manifest(newer)
    relative = cited.removeprefix("MPAS-Model-rrfs/")
    rows = manifest["files"]
    if relative in rows:
        resolved = relative
    else:
        matches = [name for name in rows if name.endswith("/" + relative)]
        if len(matches) != 1:
            # The source manifest covers scheme inputs. Other fork files keep
            # their immutable parent Git-object authentication.
            if fork_citation_window("MPAS-Model-rrfs/" + relative, first, last):
                root = Path(os.environ.get("RRFS_REF", RRFS_SOURCE_ROOT))
                resolved = resolve_cited_path(relative, root)
                return {"release": "rrfs-" + RRFS_COMMIT, "source_path": resolved,
                        "window_sha256": window_sha256(source_window(resolved, first, last, root)),
                        "authority": "immutable parent Git object"}
            raise ValueError(f"fork citation {cited!r} resolves to {matches}")
        resolved = matches[0]
    if "stochastic_physics" in resolved:
        raise ValueError("withheld source cannot be read by the citation checker")
    row = rows[resolved]
    cache = Path(os.environ.get("RRFS_CITATION_CACHE", ROOT / ".citation-cache"))
    target = cache / row["sha256"] / Path(resolved).name
    if not target.is_file():
        raise ValueError(f"pinned fork citation source unavailable: {target}; stage the declared source manifest")
    lines = _authenticated_fork_lines(str(target), row["sha256"])
    if first < 1 or last < first or last > len(lines):
        raise ValueError(f"{resolved} has {len(lines)} lines, not {first}-{last}")
    return {"release": "rrfs-" + manifest["pins"]["rrfs-model"], "source_path": resolved,
            "window_sha256": window_sha256("".join(lines[first-1:last])),
            "authority": "committed fork and submodule source manifest"}


def fork_citation_window(cited: str, first: int, last: int) -> bool:
    """Authenticate explicit fork authority; shared citations retain v8.4.2."""
    prefix='MPAS-Model-rrfs/'
    if not cited.startswith(prefix):
        return False
    import subprocess
    root=Path(os.environ["RRFS_REF"]) if os.environ.get("RRFS_REF") else RRFS_SOURCE_ROOT
    relative=cited.removeprefix(prefix)
    resolved=resolve_cited_path(relative,root)
    canonical=(root/resolved).resolve().relative_to(root.resolve()).as_posix()
    # Main-tree driver references are compared to immutable objects. A changed
    # checkout, dirty source or mutable table cannot authenticate itself.
    pin='1d6e370e7b43a332074809f170532a216c2212fb'
    source=(root/resolved).resolve()
    repository=Path(subprocess.check_output(
        ['git','-C',str(source.parent),'rev-parse','--show-toplevel'],text=True).strip()).resolve()
    if repository!=root.resolve():
        # The parent model's immutable gitlink pins the scheme repository.
        submodule=repository.relative_to(root.resolve()).as_posix()
        link=subprocess.check_output(['git','-C',str(root),'ls-tree',pin,submodule],text=True).strip()
        metadata,_,name=link.partition('\t')
        mode,kind,child_pin=metadata.split()
        if mode!='160000' or kind!='commit' or name!=submodule:
            raise ValueError('fork citation repository is not a pinned submodule: '+submodule)
        head=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
        if head!=child_pin:
            raise ValueError('fork citation submodule differs from parent gitlink: '+submodule)
        authority=subprocess.run(['git','-C',str(repository),'show',
            child_pin+':'+source.relative_to(repository).as_posix()],capture_output=True,check=True).stdout
    else:
        authority=subprocess.run(['git','-C',str(root),'show',pin+':'+canonical],
                                  capture_output=True,check=True).stdout
    if (root/resolved).read_bytes()!=authority:
        raise ValueError('fork citation source differs from the pinned Git object: '+resolved)
    source_window(resolved,first,last,root)
    return True


def check_campaign_citations(source_root: Path, mpas_source: Path) -> list[str]:
    manifest = mpas_source.parent / "MPAS-Model-v8.4.2.sha256"
    if not manifest.is_file() or _sha256_file(manifest) != SOURCE_MANIFEST_SHA256:
        return ["pinned v8.4.2 source manifest unavailable or changed; cannot authenticate cited MPAS bytes"]
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        sha, name = line.split(maxsplit=1)
        expected[name.removeprefix("./")] = sha
    failures = []
    verified = set()
    for citation, occurrences in citations(source_root).items():
        cited, first, last = _split_citation(citation)
        if not is_mpas_citation(cited):
            continue  # glibc and other non-MPAS source references
        label = f"{citation} ({occurrences[0].python_path}:{occurrences[0].python_line})"
        # This shared elementary-function note explicitly names WRF's Thompson
        # source, whose line numbering differs from MPAS's vendored version.
        if citation == "module_mp_thompson.F:5950" and all(
                Path(item.python_path).name == "rkind_libm.py" for item in occurrences):
            continue
        rrfs_errors = check_rrfs_window(citation, occurrences)
        if rrfs_errors is not None:
            failures.extend(f"{label}: {error}" for error in rrfs_errors)
            continue
        try:
            declared = [declared_fork_window(cited, first, last, item.python_path) for item in occurrences]
            if all(record is not None for record in declared):
                continue
        except (OSError, ValueError, IndexError) as error:
            failures.append(f"{label}: {error}")
            continue
        if citation in PINS:
            failures.extend(f"{label}: {error}" for error in check_legacy_window(citation))
            continue
        if cited.startswith("MPAS-Model-rrfs/"):
            try:
                fork_citation_window(cited,first,last)
            except (OSError,ValueError,IndexError,subprocess.CalledProcessError) as error:
                failures.append(f"{label}: {error}")
            continue
        if cited.startswith("MPAS-Model-") and not cited.startswith("MPAS-Model-v8.4.2/"):
            failures.append(f"{label}: citation names an older MPAS release")
            continue
        cited = cited.removeprefix("MPAS-Model-v8.4.2/")
        try:
            resolved = resolve_cited_path(cited, mpas_source)
            source_window(resolved, first, last, mpas_source)
            if resolved not in verified:
                # MPAS intentionally shares framework files through symlinks;
                # its source manifest pins their canonical file only.
                canonical = (mpas_source / resolved).resolve().relative_to(mpas_source.resolve()).as_posix()
                if expected.get(canonical) != _sha256_file(mpas_source / resolved):
                    raise ValueError(f"source SHA-256 differs from pinned manifest: {resolved}")
                verified.add(resolved)
        except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as error:
            failures.append(f"{label}: {error}")
    return failures


def check_legacy_window(citation: str) -> list[str]:
    """Retain previously audited windows without relabelling them as v8.4.2.

    PINS is the committed historical authority, not a generated inventory.
    When the archived source is installed, additionally authenticate its
    window. New historical references require an explicit reviewed pin.
    """
    pin = PINS[citation]
    if not VENDOR_ROOT.is_dir():
        return []
    try:
        window = source_window(pin.source_path, pin.first_line, pin.last_line, VENDOR_ROOT)
        if window_sha256(window) != pin.window_sha256:
            return ["v8.2.3 source-window pin mismatch"]
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as error:
        return [f"v8.2.3 pinned window unavailable: {error}"]
    return []


def check_routine_coverage(source_root: Path, mpas_source: Path) -> list[str]:
    """A module-level citation cannot silently cover newly added routines.

    Physics includes helpers and launchers, as requested by the campaign.
    Dynamics/diagnostics includes Python functions and CUDA device/global
    routines embedded in source strings. Legacy citations remain frozen by
    check(); new MPAS citations resolve inside the supplied read-only tree.
    """
    failures = []
    modules = []
    for path in sorted(source_root.rglob("*.py")):
        if "mpasphys" in path.parts or any(word in path.stem for word in ("dynamics", "diagnostics")):
            modules.append(path)
    for path in modules:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        parsed = ast.parse(text, filename=str(path))
        scopes = []
        for node in ast.walk(parsed):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Citation belongs to this routine's docstring or its adjacent
                # comment, never to a nested routine or another function body.
                leading = node.lineno - 2
                comments = []
                while leading >= 0 and lines[leading].lstrip().startswith("#"):
                    comments.insert(0, lines[leading]); leading -= 1
                scopes.append((node.name, node.lineno, "\n".join(comments) + "\n" + (ast.get_docstring(node) or "")))
        cuda = re.compile(r"(?:__global__|__device__)\s+(?:__forceinline__\s+)?(?:[\w*]+\s+)+(?P<name>\w+)\s*\(")
        for match in cuda.finditer(text):
            lineno = _line_number(text, match.start())
            leading = lineno - 2
            comments = []
            while leading >= 0 and (lines[leading].lstrip().startswith(("//", "/*", "*"))):
                comments.insert(0, lines[leading]); leading -= 1
            scopes.append((match.group("name"), lineno, "\n".join(comments)))
        for name, line, citation_text in scopes:
            citations_here = list(_FULL.finditer(citation_text))
            label = f"{path.relative_to(source_root)}:{line} {name}"
            if not citations_here:
                failures.append(f"{label}: MPAS file:line citation missing on routine")
                continue
            for match in citations_here:
                cited = match.group("file")
                first, last = _parse_range(match.group("range"))
                # Existing immutable pins are still checked against their
                # original release. New citations bind v8.4.2 directly.
                old_key = f"{cited}:{first}-{last}" if first != last else f"{cited}:{first}"
                rrfs_errors = check_rrfs_window(old_key, [CitationOccurrence(
                    old_key, str(path), line)])
                if rrfs_errors is not None:
                    failures.extend(f"{label}: {error}" for error in rrfs_errors)
                    continue
                if old_key in PINS:
                    failures.extend(f"{label}: {error}" for error in check_legacy_window(old_key))
                    continue
                if not mpas_source.is_dir():
                    failures.append(f"{label}: pinned MPAS source unavailable: {mpas_source}")
                    continue
                cited = cited.removeprefix("MPAS-Model-v8.4.2/")
                try:
                    if declared_fork_window(cited, first, last, str(path)) is not None:
                        continue
                    if not fork_citation_window(cited,first,last):
                        resolved = resolve_cited_path(cited, mpas_source)
                        source_window(resolved, first, last, mpas_source)
                except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as error:
                    failures.append(f"{label}: invalid MPAS citation: {error}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
