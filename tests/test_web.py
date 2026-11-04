import json
import threading
import urllib.error
import urllib.request

import pytest

from citeself.providers import ExtractiveProvider
from citeself.web import App, make_server


@pytest.fixture
def server(persona_dir):
    app = App(persona_dir, provider=ExtractiveProvider())
    httpd = make_server(app, "127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}", httpd.server_port
    httpd.shutdown()
    httpd.server_close()
    app.persona.close()


def call(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def post(base, body, ctype="application/json"):
    return call(base + "/api/ask", json.dumps(body).encode(), {"Content-Type": ctype}, "POST")


def test_static_and_info_with_security_headers(server):
    base, _ = server
    status, headers, body = call(base + "/")
    assert status == 200 and b"/app.js" in body and b"<script>" not in body
    assert "default-src 'none'" in headers["Content-Security-Policy"] and headers["X-Content-Type-Options"] == "nosniff"
    for path in ("/app.js", "/app.css"):
        assert call(base + path)[0] == 200
    info = json.loads(call(base + "/api/info")[2])
    assert info["name"] == "Avery Moreno" and info["source_count"] == 4 and info["provider"] == "extractive"


def test_ask_multi_turn_session(server):
    base, _ = server
    status, _, body = post(base, {"question": "release quality rollback"})
    first = json.loads(body)
    assert status == 200 and first["segments"][0]["verified"] and first["session"]
    status, _, body = post(base, {"question": "why?", "session": first["session"]})
    assert status == 200 and json.loads(body)["session"] == first["session"]


def test_rejects_bad_requests(server):
    base, port = server
    assert post(base, {"question": "x"}, ctype="text/plain")[0] == 415
    assert post(base, {"question": 5})[0] == 400
    assert post(base, {"question": ""})[0] == 400
    assert post(base, {"question": "x" * 15000})[0] == 400  # over question limit
    assert post(base, {"question": "x" * 17000})[0] == 413  # over body limit
    assert call(base + "/api/ask", b"{not json", {"Content-Type": "application/json"}, "POST")[0] == 400
    assert call(base + "/../persona.yaml")[0] == 404
    assert call(base + "/api/info", headers={"Host": f"evil.example:{port}"})[0] == 421
