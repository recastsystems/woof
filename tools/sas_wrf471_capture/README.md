# Real WRF CU_SAS capture

Use WRF's public-domain v4.7.1 source. `capture.py` checks the SAS source
hash before writing an instrumented copy and a digest receipt. It adds
stream writes at CU_SAS entry and exit on the first call per MPI rank.
It changes no physics arithmetic. Copy only that instrumented file into
an isolated strict native build, then rebuild the whole executable.

Set `WOOF_SAS_CAPTURE=/absolute/own/run/sas` for the real forecast.
The capture samples up to 256 actual columns per rank. It records the
four tendencies, precipitation increment and rate, and cloud bounds.
Capture byte order follows the native Fortran `-fconvert` setting.

    python tools/sas_wrf471_capture/capture.py SOURCE/phys/module_cu_sas.F OWN/instrumentation
    python tools/sas_wrf471_capture/canonicalize.py OWN/run OWN/canonical --byte-order big
    GPUWM_NUMERICS=strict python -m woof.verify.sas_oracle --real-capture OWN/canonical --plant --json OWN/replay.json

Canonicalization only reverses four-byte words and joins rank records.
It validates every record's length and counts. It performs no floating-point
arithmetic. The real input and native output captures are permanent paper
evidence. Keep them compressed beside the instrumentation receipt, namelist,
native build/run logs, canonicalization receipt and replay JSON. A receipt or
hash without the captured bytes cannot support an independently rerunnable
replay claim.

Before cleaning any run directory, preserve both canonical files and the raw
rank streams in the evidence destination. For example:

    mkdir -p EVIDENCE/capture EVIDENCE/raw-capture
    gzip -n -c OWN/canonical/cases.bin > EVIDENCE/capture/cases.bin.gz
    gzip -n -c OWN/canonical/stage3.bin > EVIDENCE/capture/stage3.bin.gz
    cp OWN/canonical/canonicalization.json EVIDENCE/capture/
    for file in OWN/run/sas-*-cases.bin OWN/run/sas-*-stage3.bin; do
        gzip -n -c "$file" > "EVIDENCE/raw-capture/$(basename "$file").gz"
    done
    GPUWM_NUMERICS=strict GPUWM_WRF_EXACT=1 python -m woof.verify.sas_oracle --real-capture EVIDENCE/capture --plant --json EVIDENCE/replay.json

The oracle reads `.bin.gz` directly. Record compressed and uncompressed
SHA-256 values, verify decompression against the canonicalization receipt,
and run the replay from the retained evidence directory. Only then remove
duplicate scratch captures and full forecast history. Never delete the
retained real capture during run-output cleanup. A synthetic fixture pass
is separate evidence and cannot replace this replay.
