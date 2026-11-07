"""The native reference correction must accept public and preprocessed source."""
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).with_name("wrf_bugfix.py")
FRAGMENT = """      raindn(mgs,lh) = Min( Max( rimc3, rimdn(mgs,lh) ), rimc4 )
      qiacr = dab1lh(mgs,li,lr)*xdia(mgs,lh,3)*xdia(mgs,li,3)
      IF ( ipconc .ge. 3 ) THEN
        IF ( size_large ) THEN
          large_rate = 1
        ELSE
          small_rate = 1
        ENDIF
        IF ( active_collection ) THEN
          rate = 1
        ENDIF
      ENDIF
      ELSE ! single-moment rain
        rate = ((0.25/gf4)*pi)*eri(mgs)*cx(mgs,li)*qx(mgs,lr)
      ENDIF
      fac = ( qrfrz(mgs) )*dtp/qx(mgs,lr)
"""


def run(tmp_path, source):
    src, dst = tmp_path / "native.F", tmp_path / "corrected.F"
    src.write_text(source)
    result = subprocess.run([sys.executable, str(SCRIPT), str(src), str(dst)],
                            capture_output=True, text=True, timeout=10)
    return result, dst


@pytest.mark.parametrize("comments", [True, False])
def test_single_moment_else_moves_only_the_outer_endif(tmp_path, comments):
    source = FRAGMENT if comments else FRAGMENT.replace(" ! single-moment rain", "")
    result, dst = run(tmp_path, source)
    assert result.returncode == 0, result.stderr
    expected = source.replace("rimc3, rimdn(mgs,lh)", "rimc3, raindn(mgs,lh)")
    expected = expected.replace("dab1lh(mgs,li,lr)*xdia(mgs,lh,3)",
                                "dab1lh(mgs,li,lr)*xdia(mgs,lr,3)")
    expected = expected.replace("      ENDIF\n      ELSE", "      ELSE")
    expected = expected.replace("      fac = ( qrfrz(mgs) )*dtp/qx(mgs,lr)",
                                "      ENDIF\n      fac = qx(mgs,lr)/(( qrfrz(mgs) )*dtp)")
    assert dst.read_text() == expected


def test_unexpected_control_flow_refuses_before_writing_reference(tmp_path):
    result, dst = run(tmp_path, FRAGMENT.replace(
        "      ELSE ! single-moment rain", "      unexpected = 1\n      ELSE ! single-moment rain"))
    assert result.returncode != 0
    assert "unexpected statements" in result.stderr
    assert not dst.exists()
