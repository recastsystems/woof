# Overnight dewpoints far below the airmass

## The symptom

Your night-time 2 m dewpoints collapse, and only the night is wrong.

An October run over Florida that should hold dewpoints in the low 70s
F right through the night instead prints 50s F by a few hours after
sunset. Around sunrise it recovers, and the whole afternoon looks
right. The next night it does the same thing. Plotted as a time
series it is a sawtooth: correct by day, far too dry by night, every
night of the forecast.

Skin temperature has the same shape. It falls through the night to
values nothing in the observations supports, then comes back with the
sun. 2 m temperature usually follows it down.

If that is what you are looking at, this page is about your run.

## The cause

The run has shortwave radiation switched on and longwave radiation
switched off. That pairing is a **daytime-only test configuration**.
Its name does not establish either a comparison with WRF or validation against
observations. With no longwave scheme running, the downward longwave
the land surface integrates is not computed at all. It is a fixed
300 W/m2, the buffer's allocation value, held constant for the whole
run. That number answers to nothing: not to the clouds, not to the
moisture, not to the hour. Over a warm, moist night the real downward
longwave is well above it, often by 100 W/m2 or more, so the surface
energy budget runs a deficit all night. By day the shortwave heats
the ground and hides the shortfall. After sunset the deficit is the
whole budget: skin temperature craters, the saturation humidity right
at the surface collapses with it, and the 2 m dewpoint the model
reports ends up far below the airmass that is actually sitting over
the site. Nothing is broken in the microphysics or the boundary
layer. The run is answering the question it was configured to answer,
and that question was a daytime one.

Earlier editions of this page described the missing longwave as zero.
It is not zero. It is a constant 300 W/m2 that nothing computed, which
is why the symptom is a nightly sawtooth rather than an immediate
crash. Current versions refuse to consume a downward longwave with no
producer unless the config declares it (see the acknowledgement
section below), and the mechanism over open water differs: the marine
surface path derives its endpoint from the prescribed skin temperature
and pressure alone, so a single surface call over water cannot see the
longwave at all. The land and shoreline columns are where the deficit
does its damage.

## Am I affected

Two things have to be true for this to be your problem: the version of
WOOF actually executing does not refuse this pairing over a night
window, and your config uses the pairing. Check them in that order.
The version question comes first because it is the one that gets
answered wrongly -- the number you installed is not always the number
that runs, and a config check you interpret against the wrong version
tells you nothing.

### 1. Find out which version is really executing

The number you *installed* and the number that *runs* are different
questions. Several ordinary situations make them disagree, and none
of them announce themselves.

**1a. Ask the product, if your version can answer.**

```bash
woof version
```

A normal install answers like this:

```
gpuwm 1.8.7 (installed wheel at /path/to/site-packages/gpuwm)
  Installed as a wheel; upgrade with: '/usr/bin/python3' -m pip install --upgrade recast-woof
  PyPI latest is 1.8.7 -- this install is current.
```

That is the good case: a wheel, in site-packages, and pip can replace
it. Only the third line needs the network: it reads `this install is
behind it` when a newer release exists, and it is dropped altogether
when the index cannot be reached, or when you pass `--offline`.

The other answer that matters is a tree pip did not install:

```
woof (source tree at /path/to/gpuwm, git <commit> on <branch>; no installed distribution provides it)
  Nothing pip knows about provides this code, so `pip install --upgrade` would install a SECOND copy beside it rather than replace it.
  Update the tree at /path/to/gpuwm the way you obtained it.
```

`<commit>` and `<branch>` stand in for your own checkout's, which the
real command prints. There is no version number ahead of the
parenthesis, and that absence is the message: no installed
distribution claims this code, so pip has nothing here to upgrade.

**An editable install reads the same way, and that is the trap.**
`pip install -e` leaves the running code in your source tree, where it
shadows any wheel pip later puts in site-packages, so
`pip install --upgrade recast-woof` reports success and changes nothing
about the code that runs. Measured on 1.8.7: an editable install
answers `woof version` with the source-tree shape above rather than
naming itself editable. Do not read the absence of the word
"editable" as evidence you do not have one -- step 1b is what settles
it.

If your shell cannot find `woof`, use the same command through the
interpreter you run the model with:

```bash
python -m woof.cli version
```

If *that* fails with an argument error, the `version` command does not
exist in what you are running: it was added in **1.8.1**, so you are
on something older. Keep going with the steps below, which work on
every version.

**1b. Ask pip -- through the same interpreter, never a bare `pip`.**

```bash
python -m pip show woof
```

```
Name: woof
Version: 1.8.7
Location: /path/to/.local/lib/python3.13/site-packages
Editable project location: /path/to/gpuwm
```

`Location` tells you which environment holds it. An **`Editable
project location` line means the version above it is a label on
metadata, not a description of the code** -- the real code is in the
directory that line names, and it is whatever that directory currently
contains.

A `Location` under `.local` or `AppData\Roaming\Python` is a `--user`
install. A user install is visible from every interpreter of that
Python version, including one inside a virtual environment that was
created without `--no-site-packages` isolation, so it can quietly
serve a venv you believed was clean.

**1c. Ask Python where the code came from, from your run directory.**

```bash
cd /path/to/your/run/directory
python -c "import woof, sys; print(sys.executable); print(woof.__file__); print(woof.__version__)"
```

```
/usr/bin/python3
/usr/lib/python3.13/site-packages/woof/__init__.py
1.8.7
```

Read `woof.__file__`, not `woof.__version__`. **`__version__` is
read out of installed package metadata, so it reports the number pip
last wrote, which is not necessarily the number of the code doing the
reporting.** `__file__` is the actual file that was imported. It
should sit under the `Location` that step 1b printed. If it does not,
that mismatch is your answer, and step 1d is the most likely reason.

**1d. Check that you are not standing inside a source directory.**

This one is measured, not hypothetical -- it is the trap that has
produced the most wrong conclusions on this project.

`python -c ...` and `python -m ...` both put the **current directory**
first on the import path. If the current directory contains a folder
named `woof`, that folder wins over anything you installed, silently:

```bash
$ cd /path/to/gpuwm-checkout       # a source tree: it has a gpuwm/ subfolder
$ python -c "import woof; print(woof.__file__)"
/path/to/gpuwm-checkout/woof/__init__.py
```

The installed copy was never consulted. Add `PYTHONSAFEPATH=1`, which
removes the current directory from the import path, and the same
command reaches the install instead:

```bash
$ PYTHONSAFEPATH=1 python -c "import woof; print(woof.__file__)"
/usr/lib/python3.13/site-packages/woof/__init__.py
```

The rule that follows from this: **run your diagnostics from your run
directory, not from a checkout**, and if you are not sure, put
`PYTHONSAFEPATH=1` in front. A run launched by the `woof` command
itself is not exposed to this, because a console script puts its own
directory on the path rather than yours -- but every `python -c` and
`python -m` you type is.

**1e. Check you are not talking to a second Python.**

```bash
which woof     # Windows: where woof
which pip       # Windows: where pip
```

More than one result means more than one environment, and the first
one wins. No result at all means the console script is not on your
PATH, and whatever you have been running is something else -- use
`python -m woof.cli` for everything until that is resolved. Compare
what these print against the `sys.executable` from step 1c: they
should belong to the same install.

**1f. Check the renderer separately -- it has its own version.**

The plotting engine and the GRIB decoders are compiled binaries that
live outside the Python package, in `~/.woof/bridges`. **Upgrading the
Python package does not rewrite that directory**, so a version stamped
on an image is the renderer's statement, not the engine's. A door that
resolves one of those binaries now checks it against the pins this
install carries and re-fetches the release bundle before the run
continues; with no route to the release assets it refuses instead,
naming the file and the revision it was built from. `woof doctor` is
how you see the whole estate at once, before anything opens a door.

```bash
woof doctor
```

The first line it prints is the same version headline as step 1a, so
the report says which copy of WOOF produced it. Then look for these:

```
  ok      renderer rw_wrfbatch (rust render engine): /path/to/.woof/bridges/rw_wrfbatch -- probe --help exited 0 with its usage line; basemaps ...
  ok      bridge grib1_bridge: /path/to/.woof/bridges/grib1_bridge -- speaks this release's contract
```

"speaks this release's contract" is the answer you want; it is an
actual ABI check, not just "the file exists". Anything reported as
`missing` or as not speaking this release's contract is stale, and
`woof fetch-bridges` replaces it. The line named `staged bridge
estate` is the stricter one: it compares every staged artifact's exact
bytes against this release's pins, and it is the same comparison a door
makes before it runs one.

### 2. Find out whether your config is an asymmetric pairing

**2a. Read the lines that decide it.**

```bash
grep -n "ra_lw_physics\|ra_sw_physics\|acknowledgements" myconfig.toml
```

On Windows PowerShell:

```powershell
Select-String -Path myconfig.toml -Pattern 'ra_lw_physics|ra_sw_physics|acknowledgements'
```

```
myconfig.toml:66:ra_lw_physics = 0
myconfig.toml:67:ra_sw_physics = 1
```

**`ra_lw_physics = 0` with `ra_sw_physics` greater than 0 is the
affected pairing.** Both greater than zero is fine. Both zero is fine
(no radiation at all is a different, deliberate choice, and it does
not produce this symptom).

If an `acknowledgements` line appears carrying
`asymmetric-radiation-nocturnal-window-v1`, then this config has
declared the pairing on purpose and every version that would otherwise
refuse it will stay silent. See the last section.

**2b. Let the product decide.**

```bash
woof check myconfig.toml
```

On an affected config and a version that carries the guard, that
prints:

```
woof check: experiment config myconfig.toml: this run's window includes local night (first at 2024-10-09T23:15Z at 27.5, -82.5) while domain(s) 1 run shortwave radiation with longwave OFF (ra_sw_physics 1 = Dudhia, ra_lw_physics 0; a suite matching no shipped profile).  Choose a nocturnally valid profile (both radiation streams on -- e.g. morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1, the wizard's default), or declare the daytime-only experiment by adding acknowledgements = ["asymmetric-radiation-nocturnal-window-v1"] to [experiment].  With ra_lw_physics 0 this configuration also FABRICATES its downward longwave, which is a second and separate claim, so the declaration it needs is both tokens together: acknowledgements = ["asymmetric-radiation-nocturnal-window-v1", "constant-downward-longwave-v1"].  Two claims, two tokens
  (run woof check myconfig.toml --explain for the reason)
```

The tail is your own invocation with `--explain` appended, re-runnable
exactly as printed; it adds the paragraph explaining why. A config it
accepts still prints the ordinary memory-sizing report, so there is no
silence to wait for: read the refusal, not the volume of output.

**Do not screen a directory of configs on the exit code.** `woof
check` exits **2** for every refusal it has -- an unfetched forcing
file, a schema error, a domain that will not fit, and this guard
alike -- and **0** when nothing refused the config. So exit 2 does not
mean "nocturnally affected": a config that declares the pairing
correctly, and whose only problem is that its input data has not been
downloaded yet, exits 2 as well. What identifies this guard is its
message. Screen on that instead:

```bash
for f in *.toml; do
  woof check "$f" 2>&1 >/dev/null | grep -q "local night" \
    && echo "AFFECTED  $f"
done
```

**A clean `woof check` is only good news once step 1 is settled.** It
means one of three things: the config is fine, or the config declares
the acknowledgement, or the version actually executing is older than
1.7.1 and has no opinion.

## The fix

**The guard shipped in 1.7.1.** From that release on, every front door
refuses an undeclared shortwave-on/longwave-off pairing whose window
includes local night at the reference point, and names both remedies.
1.7.0 and everything before it -- including 1.6.2 and 1.6.3 -- has no
guard, no warning, and no mention of the problem: an affected run
there produces the sawtooth in complete silence.

Upgrade:

```bash
python -m pip install --upgrade recast-woof
woof version           # confirm the number MOVED
woof fetch-bridges     # the compiled engines upgrade separately
```

Run `woof version` again afterwards and check the number actually
changed. If step 1 found an editable install, the upgrade will report
success and change nothing -- update that source tree instead, the way
you obtained it.

To correct an affected config, turn longwave on to match the
shortwave. Either edit the two lines:

```toml
ra_lw_physics = 4
ra_sw_physics = 4
```

or pick a profile whose two radiation streams are both on;
`morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1` is the wizard's default
and is one of them. The full table of shipped profiles, with a column
saying which are valid overnight, is in
[PHYSICS.md](PHYSICS.md#nocturnal-validity).

There is nothing to salvage in output already produced this way. The
night-time surface fields are wrong, so anything downstream of them --
2 m dewpoint, 2 m temperature, CAPE and CIN computed off that surface
parcel, and any overnight convective signal -- is wrong with them.
Re-run the case.

## If you want to run it anyway

Shortwave-on with longwave-off remains selectable as a daytime-only
experiment. To run it across a window that includes night, declare it in
the config:

```toml
[experiment]
acknowledgements = ["asymmetric-radiation-nocturnal-window-v1",
                    "constant-downward-longwave-v1"]
```

**Two tokens, because there are two claims.** The first says the window
has night in it. The second says the downward longwave this run
integrates is a declared constant rather than a computed flux, which is
true of a `ra_lw_physics = 0` suite on any window, day or night --
1.8.8 added that guard, and a config that declares only the first is
refused by the second. The refusal names both, so one edit clears both.

The declaration goes in the config file rather than on a command-line
flag because the refusal happens when the config is loaded, before any
runner reads its flags.

Two things to be clear about once that line is there.

**It silences the guard everywhere, for that file.** Not just at the
door you were using. `woof check`, `woof run`, `woof go`,
`run-plan` and both prepared runners will all load it without
comment, then and later, including for whoever inherits the file from
you. Anyone reading a green `woof check` on that config is reading a
statement that you declared this, not a statement that the run is
sound.

**If your config came out of `woof domain`, check which release wrote
it.** Through 1.8.7 the wizard wrote that line into the file *for you*,
silently, whenever you picked an asymmetric profile for a night window
-- so an older emitted config can carry a declaration you never made.
Look for it before you conclude the file is clean.

The wizard no longer does that. It now refuses the combination outright:

```
$ woof domain --point 36.7,-88.6 --cycle 2021-12-11T00 --hours 6 \
    --source gfs --physics-profile wsm6-ysu-mm5-noah-no-radiation-v1 --out case.toml
woof domain: profile wsm6-ysu-mm5-noah-no-radiation-v1 runs shortwave with
longwave OFF and this window includes local night (first at 2021-12-11T00:00Z
at 36.7, -88.6), so the config this would emit is one every front door refuses
at load.  Choose a nocturnally valid profile with both radiation streams on --
--physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1 -- or, if you
mean the daytime-only suite and accept the night, declare it yourself
with --ack asymmetric-radiation-nocturnal-window-v1
```

No file is written. Adding `--ack asymmetric-radiation-nocturnal-window-v1`
is what writes the line, and the wizard says so on screen as it does:

```
warning: case.toml is NOT NOCTURNALLY VALID: profile
wsm6-ysu-mm5-noah-no-radiation-v1 runs shortwave with longwave OFF and this
window includes local night (first at 2021-12-11T00:00Z). The emitted
[experiment] declares acknowledgements = ["asymmetric-radiation-nocturnal-window-v1",
"constant-downward-longwave-v1"] -- the first because you passed --ack, the
second because this suite fabricates its downward longwave and the file would
not load without it -- so no later command will stop this run. Re-emit with
a full lw+sw profile (e.g. morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1) for a
forecast.
```

The emitted file repeats it as a comment block above `[experiment]`, so
the declaration outlives the terminal it was printed in.

**The overnight output is still wrong.** The acknowledgement changes
what the software does, not what the atmosphere does. Skin
temperature, 2 m temperature and 2 m dewpoint after sunset remain
unusable, and so does every quantity derived from them. Treat the
night hours as a gap in the run and use the daylight hours the
configuration was built for.

## Related

- [PHYSICS.md](PHYSICS.md#nocturnal-validity) -- every shipped physics
  profile, its two radiation streams, and whether it is valid
  overnight.
- [Reporting a problem](REPORTING-A-PROBLEM.md) -- `woof report`
  bundles the resolved config, the warnings the run printed, and the
  identity of the install that produced it, which is every input this
  page asks you to gather.
