"""Live god-mode server: streams game events to a browser (Server-Sent Events) as the game runs.

    python -m arena play --live        # opens http://127.0.0.1:8765
    python -m arena serve              # same page plus a setup screen (see lobby.py)
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .viewer import render_html


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return          # a browser tab closed mid-request: not worth a traceback
        super().handle_error(request, client_address)


class LiveServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, setup: bool = False):
        self.host, self.port = host, port
        self.events: list[str] = []          # serialized events; SSE id == self.base + index
        self.base = 0
        self.cond = threading.Condition()
        self.api = None                      # object with handle(method, path, body) -> (status, dict); see lobby.py
        self._httpd: ThreadingHTTPServer | None = None
        self._page = render_html(None, live=True, setup=setup)

    # the Game calls this for every event
    def push(self, event: dict) -> None:
        with self.cond:
            self.events.append(json.dumps(event, separators=(",", ":")))
            self.cond.notify_all()

    def reset(self) -> None:
        """Forget the finished game's events (ids keep counting up so connected browsers stay in sync)."""
        with self.cond:
            self.base += len(self.events)
            self.events.clear()
            self.cond.notify_all()

    def _allowed(self, host_header: str | None, origin: str | None) -> bool:
        """POSTs can spend money, so only accept them from our own page (blocks other sites and DNS rebinding)."""
        hosts = {f"{self.host}:{self.port}", f"localhost:{self.port}", f"127.0.0.1:{self.port}"}
        if host_header not in hosts:
            return False
        return origin is None or origin in {f"http://{h}" for h in hosts}

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the terminal clean
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    body = server._page.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/events":
                    self._stream()
                elif path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                elif path.startswith("/api/") and server.api:
                    self._api("GET", path, None)
                else:
                    self.send_error(404)

            def do_POST(self):
                path = self.path.split("?")[0]
                if not (path.startswith("/api/") and server.api):
                    return self.send_error(404)
                if not server._allowed(self.headers.get("Host"), self.headers.get("Origin")):
                    return self._json(403, {"error": "forbidden origin"})
                if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                    return self._json(415, {"error": "expected application/json"})
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    if not 0 <= n <= 65536:
                        raise ValueError
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    return self._json(400, {"error": "bad JSON body"})
                self._api("POST", path, body)

            def _api(self, method, path, body):
                if not server._allowed(self.headers.get("Host"), None):
                    return self._json(403, {"error": "forbidden host"})
                status, payload = server.api.handle(method, path, body)
                self._json(status, payload)

            def _json(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    nxt = int(self.headers.get("Last-Event-ID", "-1")) + 1
                except ValueError:
                    nxt = 0
                try:
                    while True:
                        with server.cond:
                            nxt = max(nxt, server.base)         # skip events from before a reset
                            while nxt >= server.base + len(server.events):
                                if not server.cond.wait(timeout=15):
                                    break
                                nxt = max(nxt, server.base)
                            batch = server.events[nxt - server.base:]
                        if not batch:
                            self.wfile.write(b": ping\n\n")
                        for data in batch:
                            self.wfile.write(f"id: {nxt}\ndata: {data}\n\n".encode())
                            nxt += 1
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self._httpd = _Server((self.host, self.port), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
