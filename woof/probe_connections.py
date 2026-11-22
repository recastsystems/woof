"""Kept-alive connections for publication probes.

A publication check asks a host about every object of a lead: a HEAD, or a
one-byte GET (:data:`woof.fetch_endpoints.CONFIRM_RANGE`) where a host
refuses HEAD.  :func:`urllib.request.urlopen` opens a new TCP and TLS
connection for each of them and sends ``Connection: close``.

Named breakage (WOOF canary gate 2026-11-10, row icon-offgrid-3-1-0.5,
run_e515f5f4331317e5): ICON global's start needs are about 234 bz2 objects
on opendata.dwd.de.  From a US box a fresh TLS connection to that host
costs about 2.6 s and a request on an open one about 0.17 s (measured with
curl, 2026-11-10), so the worker's readiness probe took 52-58 s
and ``woof go`` asked the same objects again before its first byte moved:
the largest part of the run's 214 s preparation, with the CPU idle.

:func:`probe_urlopen` answers the same question with the same HTTP status
semantics as ``urlopen`` (2xx returned, 4xx/5xx raised as
:class:`urllib.error.HTTPError` with the answer body, redirects followed by
``urlopen`` itself), on HTTP/1.1 connections kept open per host and reused
by the next probe.  Only bodiless probes take this path: a HEAD, or a GET
of :data:`PROBE_RANGE`.  Everything else, and any request a configured
proxy would carry, goes to ``urlopen`` exactly as before.
"""

from __future__ import annotations

import http.client
import io
import ssl
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, getproxies, proxy_bypass, urlopen

#: The only GET a probe sends (``fetch_endpoints.CONFIRM_RANGE``).
PROBE_RANGE = "bytes=0-0"
#: Idle connections kept per host.  Bounded so idle sockets stay well under
#: the box's outbound cap (fetch-width-rows.json: stalls near 150 sockets).
IDLE_PER_HOST = 16
#: An idle connection older than this is closed rather than reused: nginx's
#: default keepalive_timeout is 75 s, and a reuse after the host closed it
#: costs a failed send before the retry on a new connection.
IDLE_SECONDS = 50.0
#: Largest answer body read so its connection can be reused.  A host that
#: ignores the one-byte range and starts sending the whole object is never
#: read: that connection is closed instead.
REUSE_BODY_BYTES = 64 * 1024

_STALE = (http.client.RemoteDisconnected, http.client.BadStatusLine,
          ConnectionResetError, ConnectionAbortedError, BrokenPipeError)

_lock = threading.Lock()
_idle: dict[tuple[str, str, int], list[tuple[float, http.client.HTTPConnection]]] = {}
_context: ssl.SSLContext | None = None


def _ssl_context() -> ssl.SSLContext:
    # What urlopen uses when no context is given (http.client's default).
    global _context
    if _context is None:
        _context = ssl._create_default_https_context()  # noqa: SLF001
    return _context


def _header(request: Request, name: str) -> str | None:
    for key, value in request.header_items():
        if key.lower() == name.lower():
            return value
    return None


def kept_alive(request: Request) -> bool:
    """Whether ``request`` is a bodiless probe this module serves."""

    if request.data is not None:
        return False
    parts = urlsplit(request.full_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    method = request.get_method()
    if method == "GET":
        if _header(request, "Range") != PROBE_RANGE:
            return False
    elif method != "HEAD":
        return False
    if parts.scheme in getproxies() and not proxy_bypass(parts.hostname):
        return False
    return True


def _key(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return parts.scheme, parts.hostname or "", port


def _take(key, timeout) -> tuple[http.client.HTTPConnection, bool]:
    now = time.monotonic()
    with _lock:
        stack = _idle.get(key, [])
        while stack:
            since, connection = stack.pop()
            if now - since <= IDLE_SECONDS:
                break
            connection.close()
        else:
            connection = None
    if connection is not None:
        connection.timeout = timeout
        if connection.sock is not None:
            connection.sock.settimeout(timeout)
        return connection, True
    scheme, host, port = key
    if scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=timeout,
                                           context=_ssl_context()), False
    return http.client.HTTPConnection(host, port, timeout=timeout), False


def _give_back(key, connection: http.client.HTTPConnection) -> None:
    with _lock:
        stack = _idle.setdefault(key, [])
        if len(stack) < IDLE_PER_HOST:
            stack.append((time.monotonic(), connection))
            return
    connection.close()


def _finish(key, connection, response) -> bytes:
    """Read a small answer body and keep the connection, or close it."""

    length = response.length
    if response.will_close or response.chunked or length is None or length > REUSE_BODY_BYTES:
        response.close()
        connection.close()
        return b""
    body = response.read()
    _give_back(key, connection)
    return body


def close_idle() -> None:
    """Close every idle probe connection (tests, and a process winding down)."""

    with _lock:
        stacks = list(_idle.values())
        _idle.clear()
    for stack in stacks:
        for _, connection in stack:
            connection.close()


class ProbeResponse:
    """The parts of a urllib response a probe reads, on a kept-alive connection."""

    def __init__(self, url, response, body):
        self.url = url
        self.status = self.code = response.status
        self.reason = response.reason
        self.headers = self.msg = response.headers
        self._body = io.BytesIO(body)

    def getcode(self):
        return self.status

    def info(self):
        return self.headers

    def geturl(self):
        return self.url

    def read(self, amount=-1):
        return self._body.read(amount)

    def close(self):
        self._body.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def probe_urlopen(request: Request | str, timeout: float | None = None):
    """``urlopen(request, timeout)`` for a probe, on a kept-alive connection."""

    if isinstance(request, str):
        request = Request(request)
    if not kept_alive(request):
        return urlopen(request) if timeout is None else urlopen(request, timeout=timeout)
    parts = urlsplit(request.full_url)
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    headers = {key: value for key, value in request.header_items()
               if key.lower() not in ("host", "connection")}
    key = _key(request.full_url)
    method = request.get_method()
    for attempt in (1, 2):
        connection, reused = _take(key, timeout)
        try:
            connection.request(method, target, headers=headers)
            response = connection.getresponse()
        except _STALE:
            connection.close()
            if reused and attempt == 1:
                continue          # the host closed an idle connection: ask on a new one
            raise
        except OSError as error:
            connection.close()
            if isinstance(error, TimeoutError):
                raise
            raise URLError(error) from error
        except BaseException:
            connection.close()
            raise
        break
    body = _finish(key, connection, response)
    status = int(response.status)
    if 300 <= status < 400:
        # urllib's redirect handling, unchanged: urlopen follows it.
        return urlopen(request) if timeout is None else urlopen(request, timeout=timeout)
    if not 200 <= status < 300:
        raise HTTPError(request.full_url, status, response.reason, response.headers,
                        io.BytesIO(body))
    return ProbeResponse(request.full_url, response, body)


__all__ = ["IDLE_PER_HOST", "PROBE_RANGE", "ProbeResponse", "close_idle",
           "kept_alive", "probe_urlopen"]
