# Reporting a problem

When something goes wrong, run one command in the run directory and
send the file it writes.

```bash
cd <the directory the run was writing to>
woof report
```

That writes `gpuwm-report-<timestamp>.zip` in the current directory and
prints a manifest of exactly what went into it. Attach the zip to your
issue or e-mail. Nothing else is needed from you.

To see what would be collected before anything is written:

```bash
woof report --dry-run
```

Other forms:

```bash
woof report /path/to/run       # a run directory that is not this one
woof report -o /other/volume   # write the zip somewhere else
woof report --exit-code 1      # record the status the command returned
woof report --log ~/run.log    # add output you redirected elsewhere
```

Run with no arguments, `woof report` reads the current directory. If
the current directory holds no receipt but `out/run` below it does, that
one is read instead and the manifest says so.

## What is in it

Everything that explains a failure, and nothing that is expensive:

- **The run's receipts** -- `report.json`, `evidence/run-receipt.json`,
  `progress.json`, `failure-capsule.json`, the certification capsule.
- **The failure itself** -- exception type, message and traceback, taken
  from the receipt that recorded it; failing that, from the last
  traceback in a collected log; failing that, from the last line the
  product printed in its own voice. Most WOOF-ArWen failures are one-line
  refusals with no traceback, on purpose, so the manifest reports that
  line and says which of the three it came from. Plus the exit status
  if you passed `--exit-code`.
- **The warnings the run printed and continued past.** WOOF-ArWen warns
  rather than blocking wherever it can, so what a run went *through* is
  often what explains where it ended up.
- **The logs** -- the supervisor's `worker-NN.stderr.log` and
  `worker-NN.stdout.log`, and any other text log in the run directory.
  A large file is kept as its first 8 KiB and last 128 KiB with the
  elided byte count marked, so both the invocation and the failure
  survive.
- **Your resolved config**, and any namelist or command script beside it.
- **What this install is** -- WOOF-ArWen version, and either the commit of
  your checkout or the wheel's `RECORD` identity, through the same
  provenance resolver a run receipt uses.
- **The stack** -- Python, NumPy, CuPy, netCDF4, matplotlib and wrf-rust
  versions; GPU model, memory, driver and CUDA runtime; OS family and
  CPU count.
- **Free space** on the run, working, temporary and home volumes. If one
  of them is nearly full the manifest says so, because a full disk is
  the most common cause of a failure whose message is empty.
- **A name-and-size inventory** of the whole run directory, so a missing
  or truncated output is visible without shipping it.

## What is NOT in it

- **No model output.** `wrfout` frames, restart files and NPZ caches are
  listed by name and size only.
- **No input data.** GRIB, `met_em` and static tiles appear only as the
  digests their receipts already record.
- **No identity.** See below.
- **Nothing this product did not write.** See next.

## It only reads what WOOF-ArWen writes

Two rules, and neither of them is redaction.

**It refuses to run anywhere that is not a run directory.** If the
target holds nothing WOOF-ArWen wrote -- no `report.json`, `progress.json`,
`run-progress.json`, `failure-capsule.json`, certification capsule,
`evidence/*.json`, `worker-*.log`, `gate.txt`, `wrfout*`, `gpuwmrst_*`,
namelist or prepared cache -- it stops with one sentence and reads
nothing. Pointing it at your home directory collects nothing, because
your home directory is not a run.

**Content is read only from files WOOF-ArWen writes.** Collection is by
allowlist, not by sweep. Anything else in the run directory is
inventoried by name and size and never opened.

On top of that, a deny-set refuses paths **before anything opens,
reads, hashes or lists them** -- not scrubbed afterwards, never looked
at:

- any dot-file or dot-directory (`.aws`, `.ssh`, `.docker`, `.azure`,
  `.kube`, `.gcloud`, `.gnupg`, `.config`, `.env`, `.netrc`,
  `.git-credentials`, `.npmrc`, `.pypirc`, ...);
- directories named `secrets`, `credentials`, `private`, `keys`;
- files named `credentials`, `token`, `secret`, `password`,
  `kubeconfig`, `id_rsa`, `id_ed25519`, `authorized_keys`, ...;
- anything shaped like a key: `*.pem`, `*.key`, `*.p12`, `*.pfx`,
  `*.keytab`, `*.jks`, `*_rsa`, `*credential*`, `*secret*`, `*token*`,
  `*password*`, ...;
- anything that **resolves** outside the run directory. Paths are
  resolved first and tested second, so a symlink named
  `worker-02.stderr.log` pointing at your private key is refused on
  what it is, not admitted on what it is called.

The manifest reports refusals **counted by class, never by name** --
because a file name can itself be the secret.

Redaction is the second line of defence here, not the first. A secret
that was read and then scrubbed is still a secret that was read.

A bundle also cannot explain a run that died before writing its first
receipt: there is nothing on disk to collect. It says so plainly rather
than looking complete. And an unsupervised run
(`woof run --no-supervise`) writes no log file at all -- if that is
what you ran, redirect the output to a file and pass it with `--log`.

## Anonymity

Removed from every path, log line, receipt value and file name in the
bundle, and counted by class in the manifest:

| Removed | Replaced by |
| --- | --- |
| Home-directory prefixes of absolute paths | `<home>` |
| Account names -- yours, and any other found in a path | `<user>` |
| Machine names, FQDNs, private-domain names | `<host>` |
| IPv4 and IPv6 addresses | `<ip>`, `<ipv6>` |
| MAC addresses | `<mac>` |
| E-mail addresses | `<email>` |
| API keys, tokens, passwords, private-key blocks | `<redacted-credential>` |
| Environment variables outside the allowlist | dropped: names kept when clean, values never |

The allowlist is every `GPUWM_*`, `CUPY_*`, `CUDA_*`, `NVIDIA_*`,
`NCCL_*` and `RW_WPS_*` variable plus a short list of thread-count and
locale names. Everything else keeps its name and loses its value.

Within the allowlist, a variable whose **name** says it holds a secret
-- anything containing `secret`, `token`, `password`, `key`, `cred`,
`auth`, `signature` or `session` -- loses its value too, before any
redaction runs. That rule exists because the allowlist is by prefix, so
it admits names nobody enumerated, and a bare value has no `key=` around
it for the credential detector to recognise. Values that are kept are
still redacted for paths, hosts and addresses.

**Your domain is kept.** Latitudes, longitudes, projections, dates, grid
shapes, physics choices and file names are scientific content, not
identity: they say what you asked the model to compute, and they are
kept verbatim. The manifest says so too, so you never have to guess
whether your study area was in the file.

SHA-256 digests survive redaction. They are how this project names
inputs and outputs, and a bundle without them cannot be matched to a
receipt.

Anonymity is by class, not by review. Your own account, home directory
and machine name come from what your machine reports about itself; a
second person's name is removed once it appears as a path segment
anywhere in what was collected. A name that appears only as a bare word
in prose, in nothing shaped like a path, cannot be told from any other
word.

**Which is why the bundle is a plain zip of UTF-8 text.** Open it. Read
`MANIFEST.txt` first -- it is the same manifest the command printed --
then read anything else you want. Nothing in it is opaque.

## If it cannot write

`woof report` builds the archive in memory and writes it once, so a
full volume costs a relocation rather than the bundle. It tries, in
order: the path given to `--output`, the current directory, the system
temporary directory, your home directory. When a location refuses, one
warning line says which, and where the bundle went instead.

Only when all of them refuse does it fail, with one sentence naming
everything it tried and asking for a volume that has room:

```bash
woof report --output /path/on/another/volume
```

## Related

- `woof doctor` answers the other question -- "is my install and my
  data estate complete" -- and prints the command that closes each gap.
  Run it first if the problem is that a command will not start.
- `woof version` answers "which copy of WOOF-ArWen is actually executing",
  which is not always the one you installed. Worth running before you
  report a problem, because an editable install, a second environment
  or a shell sitting inside a source checkout all make the version
  number lie. [NOCTURNAL-DEWPOINTS.md](NOCTURNAL-DEWPOINTS.md#1-find-out-which-version-is-really-executing)
  is the full checklist.
- Wrong overnight 2 m dewpoints and a night-time skin-temperature
  crash have a known cause and a known fix:
  [NOCTURNAL-DEWPOINTS.md](NOCTURNAL-DEWPOINTS.md).
- Security-sensitive findings go to the address in
  [SECURITY.md](../../SECURITY.md), not to a public issue.
