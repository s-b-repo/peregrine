#!/usr/bin/env python3
"""
Ollama proxy that converts Qwen3 reasoning field to content field.
This is needed because Qwen3 models put all output in the 'reasoning' field
when accessed via Ollama's /v1/chat/completions endpoint, but the AI SDK
OpenAI-compatible provider reads from the 'content' field.

Listens on 127.0.0.1:11435, forwards to Ollama on 127.0.0.1:11434/v1,
and remaps the response.
"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

OLLAMA_URL = "http://127.0.0.1:11434/v1"
LISTEN_PORT = 11435
API_KEY = "ollama"

class OllamaProxyHandler(BaseHTTPRequestHandler):
    def _proxy_request(self, method="POST"):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b''

        # Forward to Ollama
        url = f"{OLLAMA_URL}{self.path}"
        req = Request(url, data=body, method=method)
        req.add_header('Content-Type', 'application/json')

        try:
            resp = urlopen(req, timeout=300)
            resp_body = resp.read()
            
            # If this is a chat completion, remap reasoning -> content
            if self.path == '/chat/completions' and body:
                try:
                    data = json.loads(resp_body)
                    for choice in data.get('choices', []):
                        msg = choice.get('message', {})
                        if msg.get('reasoning') and not msg.get('content'):
                            # Move reasoning to content
                            msg['content'] = msg['reasoning']
                            del msg['reasoning']
                    resp_body = json.dumps(data).encode()
                except (json.JSONDecodeError, KeyError):
                    pass

            self.send_response(resp.status)
            for key, val in resp.headers.items():
                if key.lower() not in ('transfer-encoding', 'connection'):
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(resp_body)
        except HTTPError as e:
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(e.read())
        except URLError as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Backend error: {e}"}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        self._proxy_request("POST")

    def do_GET(self):
        self._proxy_request("GET")

    def log_message(self, format, *args):
        pass  # Suppress logs

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', LISTEN_PORT), OllamaProxyHandler)
    print(f"Ollama proxy listening on :{LISTEN_PORT} -> {OLLAMA_URL}")
    server.serve_forever()
