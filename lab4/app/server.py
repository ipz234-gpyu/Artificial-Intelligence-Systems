import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import generator

PORT = int(os.environ.get("PORT", "8000"))
INDEX = Path(__file__).with_name("index.html")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, generator.check_health())
        else:
            self._send(404, {"error": "Не знайдено"})

    def do_POST(self):
        if self.path != "/api/generate":
            return self._send(404, {"error": "Не знайдено"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 10_000:
                raise ValueError
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"ok": False, "error": "Некоректний запит до сервера."})
        try:
            result = generator.generate(payload.get("prompt", ""), bool(payload.get("variation")))
        except generator.GenerationError as e:
            return self._send(200, {"ok": False, "error": str(e)})
        except Exception as e:  # непередбачена помилка не повинна ламати сервер
            return self._send(500, {"ok": False, "error": f"Внутрішня помилка: {e}"})
        self._send(200, result)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    print(f"Модель: {generator.MODEL}  |  Ollama: {generator.BASE_URL}")
    print(f"Відкрийте http://localhost:{PORT}  (зупинка: Ctrl+C)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
