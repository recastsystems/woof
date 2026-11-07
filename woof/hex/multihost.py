"""hex ranks on any number of hosts: the host list, the rendezvous, the launcher and a joined host's side.

``woof hex forecast ... --hosts nodeA:0-7,nodeB:0-7`` runs one rank process per listed card.  The command runs on
the first host, which holds rank 0 and writes every output file.  Every other host runs the same command, started
here (``ssh`` for a host name, a local process for a ``local<N>`` virtual host) with ``WOOF_HEX_MULTIHOST_JOIN``
set, so each host prepares its own ranks from its own copy of the inputs (the same paths on every host: a shared
filesystem or staged copies) on its own cores, in parallel with rank 0.  The hosts meet:

- once at preparation: rank 0 sends each host the partitions, the rank-to-host assignment and the NCCL id;
- every halo round, through NCCL only (``partition_nccl_transport_v841``; no host-to-host header);
- for collectives and output (``rank_processes.ProcessComm``): ranks on rank 0's host over socketpairs and shared
  memory as before, ranks on other hosts over TCP (``rank_processes.Link`` carries large buffers in band there);
- once at the end: each host sends its ranks' receipts to rank 0.

Virtual hosts (``local0:0-3,local1:4-7``, or the same cards twice) run every rank on this machine but give each
host its own ``NCCL_HOSTID``, so NCCL treats them as separate machines and moves every halo between them through its
network transport (``NCCL_P2P_DISABLE``/``NCCL_SHM_DISABLE`` push the halos inside a virtual host there too): the
cross-host code path, proven without a rental.

Nothing here does numerics.  Halos move bytes and gathers keep rank order, so the words written do not depend on how
many hosts the ranks span.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

__all__ = ["HostSpec", "HostPlan", "parse_hosts", "assign_hosts", "begin", "current", "Rendezvous", "connect",
           "ENV_JOIN", "ENV_HOSTS"]

ENV_JOIN = "WOOF_HEX_MULTIHOST_JOIN"
ENV_HOSTS = "WOOF_HEX_HOSTS"
_HELLO = struct.Struct("!I")
#: Environment a remote (ssh) host's command needs to run the same installation the same way.
_FORWARD_PREFIXES = ("GPUWM_", "RW_MPAS_", "MPAS_", "NCCL_", "CUPY_", "PYTHON", "OMP_", "OPENBLAS_", "UCX_",
                     "FI_", "HEX_", "RCCL_", "HIP_", "HSA_", "ROCM_", "ROCR_")
_FORWARD_NAMES = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "VIRTUAL_ENV", "TMPDIR", "XDG_CACHE_HOME", "HOME", "HIPCC", "HIP_PATH", "ROCM_PATH")
_NOT_FORWARDED = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                  "GPU_MUTEX_CARDS", "NCCL_HOSTID", ENV_JOIN)


class MultiHostError(RuntimeError):
    """A host could not join, failed, or disagreed with rank 0; the run stops instead of waiting or mixing."""


# -- the host list -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class HostSpec:
    name: str
    cards: tuple[int, ...]
    launch: str   # "self" (rank 0's host), "local" (a virtual host on this machine) or "ssh"


def _cards(text: str) -> tuple[int, ...]:
    cards: list[int] = []
    for item in text.split("+"):
        item = item.strip()
        if not item:
            raise ValueError("empty card entry")
        if "-" in item:
            low, high = (int(value) for value in item.split("-", 1))
            if high < low:
                raise ValueError(f"card range {item} runs backwards")
            cards.extend(range(low, high + 1))
        else:
            cards.append(int(item))
    if any(card < 0 for card in cards):
        raise ValueError("card ordinals are non-negative")
    return tuple(cards)


def parse_hosts(text: str) -> tuple[HostSpec, ...]:
    """``name:cards[,name:cards...]``; cards are ``a-b`` ranges or ordinals joined by ``+`` (``n1:0-3+6+7``).

    The first host is where the command runs (rank 0).  A name starting with ``local`` is a virtual host on this
    machine; any other name after the first is reached with ``ssh``.
    """
    entries = [entry.strip() for entry in str(text).replace(";", ",").split(",") if entry.strip()]
    if not entries:
        raise ValueError("--hosts needs at least one host:cards entry")
    hosts, names = [], set()
    for index, entry in enumerate(entries):
        name, sep, cards = entry.rpartition(":")
        if not sep or not name:
            raise ValueError(f"--hosts entry {entry!r} needs host:cards (for example hostA:0-7)")
        if name in names:
            raise ValueError(f"--hosts names {name!r} twice; list each host once with all its cards")
        names.add(name)
        try:
            parsed = _cards(cards)
        except ValueError as error:
            raise ValueError(f"--hosts entry {entry!r}: {error}") from None
        launch = "self" if index == 0 else ("local" if name.startswith("local") else "ssh")
        hosts.append(HostSpec(name, parsed, launch))
    return tuple(hosts)


@dataclass
class HostPlan:
    """Which host and which of its cards each global rank runs on."""

    hosts: tuple[HostSpec, ...]
    host_of_rank: tuple[int, ...] = ()
    card_of_rank: tuple[int, ...] = ()
    assignment: dict[str, Any] = field(default_factory=dict)

    @property
    def ranks(self) -> int:
        return sum(len(host.cards) for host in self.hosts)

    @property
    def virtual(self) -> bool:
        return any(host.launch == "local" or (host.launch == "self" and host.name.startswith("local"))
                   for host in self.hosts)

    def local_ranks(self, host: int) -> list[int]:
        return [rank for rank, owner in enumerate(self.host_of_rank) if owner == int(host)]

    def receipt(self) -> dict[str, Any]:
        return dict(hosts=[dict(name=h.name, cards=list(h.cards), launch=h.launch) for h in self.hosts],
                    ranks=self.ranks, host_of_rank=list(self.host_of_rank), card_of_rank=list(self.card_of_rank),
                    virtual_hosts=self.virtual, assignment=self.assignment)


def assign_hosts(tables: Sequence[Any], hosts: Sequence[HostSpec]) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Which ranks share a host: least halo traffic between hosts, rank 0 on the first host.

    Ranks keep their partitions (rank r runs partition r; nothing about the words changes); only where each rank runs
    is chosen.  The weight of a pair is its two directional cell and edge halo index counts (the entity transfers
    ``partition_topology_v841.place_ranks`` weighs).  Hosts are grown greedily from a seed (rank 0 for the first,
    then the unassigned rank least connected to the hosts already grown), taking the rank most connected to the
    host so far, then pairwise swaps between hosts that cut cross-host weight are made until none helps.
    """
    n = len(tables)
    sizes = [len(host.cards) for host in hosts]
    if sum(sizes) != n:
        raise ValueError(f"the host list has {sum(sizes)} cards for {n} ranks")
    weight = [[0] * n for _ in range(n)]
    for rank, table in enumerate(tables):
        for peer in table.peers:
            other = int(peer.peer)
            amount = len(peer.cell_send) + len(peer.edge_send)
            weight[rank][other] += amount
            weight[other][rank] += amount
    owner = [-1] * n
    for host, size in enumerate(sizes):
        if size == 0:
            continue
        unassigned = [rank for rank in range(n) if owner[rank] < 0]
        if host == 0:
            seed = 0
        else:
            assigned = [rank for rank in range(n) if owner[rank] >= 0]
            seed = min(unassigned, key=lambda rank: (sum(weight[rank][a] for a in assigned), rank))
        members = [seed]
        owner[seed] = host
        while len(members) < size:
            candidates = [rank for rank in range(n) if owner[rank] < 0]
            pick = max(candidates, key=lambda rank: (sum(weight[rank][m] for m in members), -rank))
            owner[pick] = host
            members.append(pick)

    def cut() -> int:
        return sum(weight[a][b] for a in range(n) for b in range(a + 1, n) if owner[a] != owner[b])

    before = cut()
    swaps = 0
    improved = True
    passes = 0
    while improved and passes < 64:
        improved = False
        passes += 1
        best = (0, None, None)
        to_host = [[0] * len(sizes) for _ in range(n)]
        for a in range(n):
            for b in range(n):
                if a != b:
                    to_host[a][owner[b]] += weight[a][b]
        for a in range(1, n):
            for b in range(a + 1, n):
                p, q = owner[a], owner[b]
                if p == q or b == 0:
                    continue
                gain = (to_host[a][q] - to_host[a][p]) + (to_host[b][p] - to_host[b][q]) - 2 * weight[a][b]
                if gain > best[0]:
                    best = (gain, a, b)
        if best[1] is not None:
            a, b = best[1], best[2]
            owner[a], owner[b] = owner[b], owner[a]
            swaps += 1
            improved = True
    after = cut()
    total = sum(weight[a][b] for a in range(n) for b in range(a + 1, n))
    return tuple(owner), dict(objective="cross-host cell+edge halo entity transfers",
                              total_transfers=total, cross_host_before_swaps=before, cross_host=after, swaps=swaps)


def plan_ranks(hosts: Sequence[HostSpec], tables: Sequence[Any]) -> HostPlan:
    """The plan for these tables: ranks to hosts by :func:`assign_hosts`, then each host's cards in rank order."""
    hosts = tuple(hosts)
    if len(hosts) == 1:
        owner = tuple(0 for _ in tables)
        record = dict(objective="one host", cross_host=0)
    else:
        owner, record = assign_hosts(tables, hosts)
    cards = [0] * len(tables)
    for host_index, host in enumerate(hosts):
        mine = [rank for rank, value in enumerate(owner) if value == host_index]
        for rank, card in zip(mine, host.cards):
            cards[rank] = card
    return HostPlan(hosts, tuple(owner), tuple(cards), record)


# -- framing and the rendezvous ----------------------------------------------------------------------------------------

def _send_json(sock: socket.socket, value: Any) -> None:
    data = json.dumps(value).encode()
    sock.sendall(_HELLO.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    out = bytearray()
    while len(out) < size:
        chunk = sock.recv(size - len(out))
        if not chunk:
            raise ConnectionError("peer closed during the hello")
        out.extend(chunk)
    return bytes(out)


def _recv_json(sock: socket.socket) -> Any:
    (size,) = _HELLO.unpack(_recv_exact(sock, _HELLO.size))
    if size > 1 << 20:
        raise ConnectionError("oversized hello")
    return json.loads(_recv_exact(sock, size))


def _tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, option, 16 << 20)
        except OSError:
            pass


class Rendezvous:
    """Rank 0's listener: every other host's links arrive here, each named by (kind, id) and the run's token."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = 0, token: str | None = None):
        self.token = token or secrets.token_hex(16)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.listen(1024)
        self.port = int(self.sock.getsockname()[1])
        self._arrived: dict[tuple[str, str], socket.socket] = {}
        self._condition = threading.Condition()
        self._closed = False
        self.rejected = 0
        self._thread = threading.Thread(target=self._accept, name="hex-multihost-rendezvous", daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn, _address = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._hello, args=(conn,), daemon=True).start()

    def _hello(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(30.0)
            hello = _recv_json(conn)
            if not isinstance(hello, dict) or hello.get("token") != self.token:
                raise ConnectionError("wrong token")
            key = (str(hello["kind"]), str(hello["id"]))
            _tune(conn)
            conn.settimeout(None)
        except (OSError, ValueError, KeyError, ConnectionError):
            self.rejected += 1
            try:
                conn.close()
            except OSError:
                pass
            return
        with self._condition:
            if key in self._arrived:
                conn.close()
                return
            self._arrived[key] = conn
            self._condition.notify_all()

    def wait(self, kind: str, ident: Any, timeout: float, *, failed=None) -> socket.socket:
        """The link named (kind, ident), waiting up to ``timeout`` s; ``failed()`` names a host that has stopped."""
        key = (str(kind), str(ident))
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while key not in self._arrived:
                reason = failed() if failed is not None else None
                if reason:
                    raise MultiHostError(f"{kind} link {ident} never arrived: {reason}")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MultiHostError(f"{kind} link {ident} did not arrive in {timeout:.0f} s (its host failed "
                                         "before joining, or cannot reach this host's address and port)")
                self._condition.wait(min(left, 2.0))
            return self._arrived.pop(key)

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass
        with self._condition:
            for conn in self._arrived.values():
                try:
                    conn.close()
                except OSError:
                    pass
            self._arrived.clear()


def connect(address: str, port: int, token: str, kind: str, ident: Any, *, timeout: float = 600.0) -> socket.socket:
    """A link to rank 0's rendezvous, retried until ``timeout`` (rank 0 may still be preparing)."""
    deadline = time.monotonic() + float(timeout)
    last = None
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((address, int(port)), timeout=30.0)
            _tune(sock)
            _send_json(sock, dict(token=token, kind=kind, id=str(ident)))
            sock.settimeout(None)
            return sock
        except OSError as error:
            last = error
            time.sleep(0.5)
    raise MultiHostError(f"could not reach rank 0 at {address}:{port} for the {kind} link {ident} in "
                         f"{timeout:.0f} s ({last})")


def _route_address(target: str) -> str:
    """This host's address on the route to ``target`` (no packet is sent)."""
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((socket.gethostbyname(target), 22))
            return probe.getsockname()[0]
        finally:
            probe.close()
    except OSError:
        return socket.gethostbyname(socket.gethostname())


# -- sessions ----------------------------------------------------------------------------------------------------------

_SESSION = None


def current():
    """The run's multi-host session in this process: a :class:`LaunchSession` on rank 0's host, a
    :class:`JoinedHost` on another host, or None for a run on one host without ``--hosts``."""
    return _SESSION


def _host_text(arguments) -> str | None:
    return getattr(arguments, "hosts", None) or os.environ.get(ENV_HOSTS) or None


def begin(arguments):
    """Start the multi-host run if ``--hosts`` (or WOOF_HEX_HOSTS) names one; returns the session or None.

    On rank 0's host this launches every other host's command at once, so their preparation runs beside this
    host's.  On a joined host it connects the lifeline that stops this host when rank 0 stops.  Either way the
    run's ``samehost_devices`` becomes one entry per rank, so the door takes its multi-card route.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    text = _host_text(arguments)
    if text is None:
        return None
    hosts = parse_hosts(text)
    joined = os.environ.get(ENV_JOIN)
    arguments.samehost_devices = ",".join(str(card) for host in hosts for card in host.cards)
    executor = (getattr(arguments, "samehost_ranks", None) or os.environ.get("WOOF_HEX_SAMEHOST_RANKS") or "auto")
    if str(executor).lower() == "threads":
        raise ValueError("--hosts runs one process per card (rank threads cannot span hosts); drop "
                         "--samehost-ranks threads")
    if joined:
        _SESSION = JoinedHost(json.loads(joined), hosts)
    else:
        _SESSION = LaunchSession(hosts, arguments)
        _SESSION.start()
    return _SESSION


class LaunchSession:
    """Rank 0's side: the rendezvous, every other host's process, and the host links."""

    joined = False
    host_index = 0

    def __init__(self, hosts: tuple[HostSpec, ...], arguments):
        self.hosts = hosts
        self.plan: HostPlan | None = None
        self.out = Path(getattr(arguments, "out")).resolve()
        self.timeout = float(os.environ.get("WOOF_HEX_MULTIHOST_TIMEOUT", "3600"))
        self.rendezvous = Rendezvous(port=int(os.environ.get("WOOF_HEX_MULTIHOST_PORT", "0") or 0))
        self.address = self._address()
        self.processes: dict[int, subprocess.Popen] = {}
        self.failures: dict[int, str] = {}
        self._links: dict[int, Any] = {}
        self._lifelines: dict[int, socket.socket] = {}
        self._finished = False
        self.started_utc = None
        self.logs: dict[int, str] = {}
        if self.virtual:
            # NCCL keys "same host" on its host hash: a virtual host gets its own, so every halo between virtual
            # hosts goes through NCCL's network transport, as between machines.
            os.environ["NCCL_HOSTID"] = self._hostid(0)

    @property
    def virtual(self) -> bool:
        return HostPlan(self.hosts).virtual

    def _hostid(self, host: int) -> str:
        return f"{socket.gethostname()}:{self.hosts[host].name}"

    def _address(self) -> str:
        named = os.environ.get("WOOF_HEX_MULTIHOST_ADDRESS")
        if named:
            return named
        remote = [host for host in self.hosts[1:] if host.launch == "ssh"]
        if not remote:
            return "127.0.0.1"
        return _route_address(remote[0].name)

    def _joined_argv(self, host: int) -> list[str]:
        argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
        argv[0] = sys.executable
        host_out = self.out / ".multihost" / f"host{host}"
        # Local virtual hosts and hosts sharing a filesystem must not claim
        # rank 0's scratch owner or write its exclusive measurement receipt.
        paths = {"--out": host_out, "--scratch": host_out / "scratch",
                 "--report": host_out / "measurement.json"}
        found_out = False
        for index, token in enumerate(argv):
            option = token.split("=", 1)[0]
            if option not in paths:
                continue
            if "=" in token:
                argv[index] = f"{option}={paths[option]}"
            elif index + 1 < len(argv):
                argv[index + 1] = str(paths[option])
            else:
                raise MultiHostError(f"{option} needs a path on the command line")
            found_out = found_out or option == "--out"
        if found_out:
            return argv
        raise MultiHostError("--hosts needs --out on the command line: every host writes its own preparation "
                             "under it")

    def start(self) -> None:
        self.started_utc = time.time()
        (self.out / ".multihost").mkdir(parents=True, exist_ok=True)
        for host in range(1, len(self.hosts)):
            spec = self.hosts[host]
            join = dict(address=self.address, port=self.rendezvous.port, token=self.rendezvous.token, host=host,
                        timeout=self.timeout)
            argv = self._joined_argv(host)
            log_path = self.out / ".multihost" / f"host{host}.log"
            self.logs[host] = str(log_path)
            log = open(log_path, "wb")
            if spec.launch == "local":
                environment = dict(os.environ)
                environment[ENV_JOIN] = json.dumps(join)
                if self.virtual:
                    environment["NCCL_HOSTID"] = self._hostid(host)
                process = subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=os.getcwd(), preexec_fn=_die_with_parent)
            else:
                forwarded = {name: value for name, value in os.environ.items()
                             if (name in _FORWARD_NAMES or name.startswith(_FORWARD_PREFIXES))
                             and name not in _NOT_FORWARDED}
                forwarded[ENV_JOIN] = json.dumps(join)
                # The same working directory (made if absent): relative paths in the command mean the same there.
                remote = "mkdir -p {0} && cd {0} && exec env {1} {2}".format(
                    shlex.quote(os.getcwd()),
                    " ".join(f"{name}={shlex.quote(value)}" for name, value in sorted(forwarded.items())),
                    " ".join(shlex.quote(token) for token in argv))
                ssh = os.environ.get("WOOF_HEX_MULTIHOST_SSH", "ssh -o BatchMode=yes -o ServerAliveInterval=15")
                process = subprocess.Popen([*shlex.split(ssh), spec.name, remote], stdout=log,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           preexec_fn=_die_with_parent)
            log.close()
            self.processes[host] = process
            threading.Thread(target=self._watch, args=(host, process), name=f"hex-multihost-watch-{host}",
                             daemon=True).start()
        for host in range(1, len(self.hosts)):
            threading.Thread(target=self._lifeline, args=(host,), daemon=True).start()

    def _lifeline(self, host: int) -> None:
        try:
            self._lifelines[host] = self.rendezvous.wait("lifeline", host, self.timeout,
                                                         failed=lambda: self.failures.get(host))
        except MultiHostError:
            pass

    def _watch(self, host: int, process: subprocess.Popen) -> None:
        code = process.wait()
        if code != 0 and not self._finished:
            tail = ""
            try:
                tail = Path(self.logs[host]).read_bytes()[-4000:].decode(errors="replace")
            except OSError:
                pass
            self.failures[host] = (f"host {host} ({self.hosts[host].name}) exited with code {code}; "
                                   f"its log {self.logs[host]} ends:\n{tail}")
            for link in self._links.values():
                try:
                    link.shutdown()
                except Exception:
                    pass

    def failure(self) -> str | None:
        return next(iter(self.failures.values()), None)

    def host_link(self, host: int):
        """The control link to host ``host``'s leader (TCP, in-band buffers)."""
        from .rank_processes import Link
        link = self._links.get(host)
        if link is None:
            sock = self.rendezvous.wait("host", host, self.timeout, failed=lambda: self.failures.get(host))
            link = Link(sock, timeout=self.timeout, name=f"host-0-{host}")
            self._links[host] = link
        return link

    def rank_link(self, kind: str, rank: int, timeout: float):
        from .rank_processes import Link
        sock = self.rendezvous.wait(kind, rank, timeout, failed=self.failure)
        return Link(sock, timeout=timeout, name=f"{kind}-0-{rank}")

    def finish(self) -> None:
        """Every host reported; wait for their processes, then close the links."""
        self._finished = True
        for host, process in self.processes.items():
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
        self.close()

    def kill(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                try:
                    process.kill()   # the ssh client or the local host process, by its own PID
                except OSError:
                    pass

    def close(self) -> None:
        for link in self._links.values():
            link.close()
        for sock in self._lifelines.values():
            try:
                sock.close()
            except OSError:
                pass
        self.rendezvous.close()

    def receipt(self) -> dict[str, Any]:
        return dict(role="rank 0 host", address=self.address, port=self.rendezvous.port,
                    hosts=[dict(name=h.name, cards=list(h.cards), launch=h.launch) for h in self.hosts],
                    plan=None if self.plan is None else self.plan.receipt(), logs=self.logs,
                    exit_codes={str(h): p.poll() for h, p in self.processes.items()},
                    nccl_hostid=os.environ.get("NCCL_HOSTID"))


class JoinedHost:
    """A host other than rank 0's: its links to rank 0 and the lifeline that stops it if rank 0 stops."""

    joined = True

    def __init__(self, join: dict, hosts: tuple[HostSpec, ...]):
        self.hosts = hosts
        self.address = str(join["address"])
        self.port = int(join["port"])
        self.token = str(join["token"])
        self.host_index = int(join["host"])
        self.timeout = float(join.get("timeout", 3600))
        self.plan: HostPlan | None = None
        self._finished = False
        self._link = None
        self._lifeline = connect(self.address, self.port, self.token, "lifeline", self.host_index,
                                 timeout=self.timeout)
        threading.Thread(target=self._watch_lifeline, name="hex-multihost-lifeline", daemon=True).start()

    def _watch_lifeline(self) -> None:
        try:
            while True:
                data = self._lifeline.recv(1)
                if not data:
                    break
        except OSError:
            pass
        if not self._finished:
            print(f"woof hex: host {self.host_index}: rank 0's host stopped; stopping this host's ranks",
                  file=sys.stderr, flush=True)
            # This host's rank processes die with this process (rank_processes.set_parent_death_signal).
            os._exit(3)

    def host_link(self):
        from .rank_processes import Link
        if self._link is None:
            sock = connect(self.address, self.port, self.token, "host", self.host_index, timeout=self.timeout)
            self._link = Link(sock, timeout=self.timeout, name=f"host-{self.host_index}-0")
        return self._link

    def connect_spec(self) -> dict:
        return dict(address=self.address, port=self.port, token=self.token, timeout=self.timeout)

    def finish(self) -> None:
        self._finished = True
        if self._link is not None:
            self._link.close()
        try:
            self._lifeline.close()
        except OSError:
            pass

    def receipt(self) -> dict[str, Any]:
        return dict(role=f"joined host {self.host_index}", rank0=f"{self.address}:{self.port}",
                    nccl_hostid=os.environ.get("NCCL_HOSTID"))


def rank_links(spec: dict, rank: int, timeout: float):
    """A rank process on a joined host: its step and output links to rank 0, over TCP."""
    from .rank_processes import Link
    step = Link(connect(spec["address"], spec["port"], spec["token"], "step", rank, timeout=timeout),
                timeout=timeout, name=f"step-{rank}-0")
    output = Link(connect(spec["address"], spec["port"], spec["token"], "output", rank, timeout=timeout),
                  timeout=timeout, name=f"output-{rank}-0")
    return step, output


def _die_with_parent() -> None:
    """A host process (or its ssh client) dies with rank 0's process (Linux PR_SET_PDEATHSIG)."""
    try:
        import ctypes
        ctypes.CDLL(None).prctl(1, int(signal.SIGKILL), 0, 0, 0)
    except Exception:
        pass


def is_loopback(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False
