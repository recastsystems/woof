# Security policy

Only the latest released version of WOOF-ArWen (currently the 1.0.x line) receives
security fixes; no longer-term security-fix window is promised yet. Report
suspected path traversal, unsafe archive handling, manifest bypass,
hash/identity confusion, malformed GRIB/NetCDF crashes, resource-exhaustion
problems, or unintended overwrite behavior privately, through the WOOF
repository's private vulnerability report (https://github.com/recastsystems/woof/security).
Do not include sensitive source data in a public issue.

Never send credentials, rented-node access details, or private
meteorological payloads as part of a report. A minimal synthetic file,
exact version/commit, platform, command, observed result, and expected safe
failure are preferred.

`woof report` assembles the version, commit, platform, command context and
observed failure for you, with usernames, home paths, hostnames, addresses
and credential-shaped strings replaced by class placeholders, and prints a
manifest of what it collected before you send it -- see
[reporting a problem](docs/public/REPORTING-A-PROBLEM.md). Read the bundle
before attaching it; it is a plain zip of UTF-8 text, and redaction by class
is not a substitute for your own look at a payload you know is sensitive.
