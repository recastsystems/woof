"""Installs the TEMPO column capture when WOOF_TEMPO_CAPTURE is set (see ../capture_forecast_columns.py)."""
import os as _os
import sys as _sys

if _os.environ.get("WOOF_TEMPO_CAPTURE"):
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import capture_forecast_columns as _capture
    _capture.install_lazy()
