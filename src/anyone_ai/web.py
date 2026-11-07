"""Local web UI: ``anyone serve``. Standard library only; binds to loopback by default.

Conversations live in server memory only and are never written to disk or logs.
"""
from __future__ import annotations

import json
import secrets
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from .engine import Conversation, Persona
from .providers import Provider, ProviderError

MAX_BODY = 16_000
MAX_SESSIONS = 200
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/app.css": ("app.css", "text/css; charset=utf-8")}


class App:
    def __init__(self, root: Path, provider: Provider | None = None, provider_override: str | None = None):
        self.persona = Persona(root, provider=provider, provider_override=provider_override)
        self.lock = threading.Lock()  # serializes store + model access (single local user)
        self.sessions: OrderedDict[str, Conversation] = OrderedDict()

    def session(self, sid: str | None) -> tuple[str, Conversation]:
        if sid and sid in self.sessions:
            self.sessions.move_to_end(sid)
            return sid, self.sessions[sid]
        sid = secrets.token_urlsafe(16)
        self.sessions[sid] = Conversation(self.persona)
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)
        return sid, self.sessions[sid]

    def info(self) -> dict:
        m = self.persona.manifest
        with self.lock:
            sources = [s for s in self.persona.store.list_sources() if not s.excluded]
        return {"name": m.subject.name, "subject_kind": m.subject.kind, "bio": m.subject.bio, "disclosure": m.disclosure,
                "blocked_topics": m.boundaries.blocked_topics, "source_count": len(sources),
                "provider": self.persona.provider.name}


def make_handler(app: App, allowed_hosts: set[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "anyone"
        sys_version = ""

        def log_message(self, *args):  # never log requests: URLs/bodies may contain questions
            return

        def _headers(self, status: int, ctype: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()

        def _json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self._headers(status, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _host_ok(self) -> bool:
            # Defends against DNS rebinding: only answer to the hostnames we bound to.
            return (self.headers.get("Host") or "").lower() in allowed_hosts

        def do_GET(self):
            if not self._host_ok():
                return self._json({"error": "bad_host"}, 421)
            path = self.path.split("?", 1)[0]
            if path in STATIC:
                fname, ctype = STATIC[path]
                body = resources.files("anyone_ai").joinpath("web", fname).read_bytes()
                self._headers(200, ctype, len(body))
                return self.wfile.write(body)
            if path == "/api/info":
                return self._json(app.info())
            self._json({"error": "not_found"}, 404)

        def do_POST(self):
            if not self._host_ok():
                return self._json({"error": "bad_host"}, 421)
            if self.path != "/api/ask":
                return self._json({"error": "not_found"}, 404)
            # Requiring JSON forces a CORS preflight for cross-site requests, which we never approve.
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json({"error": "content_type_must_be_json"}, 415)
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json({"error": "bad_length"}, 400)
            if n <= 0 or n > MAX_BODY:
                return self._json({"error": "bad_length"}, 413 if n > MAX_BODY else 400)
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return self._json({"error": "invalid_json"}, 400)
            if not isinstance(body, dict):
                return self._json({"error": "invalid_json"}, 400)
            question, sid = body.get("question"), body.get("session")
            if not isinstance(question, str) or (sid is not None and not isinstance(sid, str)):
                return self._json({"error": "invalid_request"}, 400)
            with app.lock:
                sid, convo = app.session(sid)
                try:
                    answer = convo.send(question)
                except ProviderError as exc:
                    return self._json({"error": "model_unavailable", "message": str(exc), "session": sid}, 503)
                except ValueError as exc:
                    return self._json({"error": "invalid_question", "message": str(exc), "session": sid}, 400)
            self._json({**answer.to_dict(), "session": sid})

    return Handler


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765, provider_override: str | None = None) -> None:
    app = App(root, provider_override=provider_override)
    httpd = make_server(app, host, port)
    print(f"Anyone: {app.persona.manifest.subject.name} at http://{host}:{httpd.server_port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.persona.close()


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), None)  # handler set below once the real port is known
    p = httpd.server_port
    hosts = {f"{h}:{p}" for h in ("127.0.0.1", "localhost", host)} | ({"[::1]:%d" % p} if host in ("::1", "::") else set())
    httpd.RequestHandlerClass = make_handler(app, hosts)
    return httpd
