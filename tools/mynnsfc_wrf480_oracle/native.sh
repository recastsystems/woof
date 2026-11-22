#!/usr/bin/env bash
# Run the native replay inside the pinned referee container (glibc 2.39 libm),
# with the port folder mounted at the same path.  REPLAY_EXE overrides the
# executable built by build.sh.
P=${WRF480_PORT:-$HOME/wrf480-port}
exec podman run --rm -v "$P:$P" localhost/wrfref2404:1 \
  "${REPLAY_EXE:?set REPLAY_EXE to the replay_mynnsfc.exe this oracle builds}" "$1" "$2"
