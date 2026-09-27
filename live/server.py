"""A tiny local web server for the dashboard (Python standard library only).

    GET  /                     the dashboard page
    GET  /static/<file>        its stylesheet and script
    GET  /api/state?since=N    status, stats, and every game after game N
    POST /api/pause | /api/continue | /api/reset
    POST /api/speed            body: {"games_per_second": 10}   (0 = as fast as possible)
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {  # only these files are ever served
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def make_server(session, port=8765, host="127.0.0.1"):
    """Build (but don't start) a server bound to this computer only."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path in STATIC_FILES:
                name, content_type = STATIC_FILES[url.path]
                self._send(200, (STATIC_DIR / name).read_bytes(), content_type)
            elif url.path == "/api/state":
                since = parse_qs(url.query).get("since", ["0"])[0]
                self._send_json(session.snapshot(since=max(0, int(since)) if since.isdigit() else 0))
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            actions = {"/api/pause": session.pause, "/api/continue": session.resume,
                       "/api/reset": session.reset}
            path = urlparse(self.path).path
            if path in actions:
                actions[path]()
            elif path == "/api/speed":
                try:
                    session.set_speed(float(self._read_json()["games_per_second"]))
                except (KeyError, ValueError, TypeError):
                    return self._send_json({"error": "expected {\"games_per_second\": number}"}, 400)
            else:
                return self._send_json({"error": "not found"}, status=404)
            self._send_json(session.snapshot(since=0) | {"results": []})

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def _send_json(self, data, status=200):
            self._send(status, json.dumps(data).encode(), "application/json")

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the terminal quiet (it polls 4x a second)
            pass

    return ThreadingHTTPServer((host, port), Handler)
