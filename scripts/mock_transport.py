"""Socket-free HTTP-handler transport for environments that prohibit loopback binding."""
import io
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from urllib.parse import urlparse
from unittest.mock import patch
import urllib.request
from mock_server import Handler


class Connection:
    def __init__(self, raw):
        self.input = io.BytesIO(raw); self.output = io.BytesIO()

    def makefile(self, mode, *args):
        return self.input

    def sendall(self, data):
        self.output.write(data)


class Transport:
    """Pass real serialized requests through BaseHTTPRequestHandler without sockets."""
    def __init__(self, mode="clean"):
        self.server = SimpleNamespace(mode=mode, calls=0)

    def urlopen(self, request, timeout=None):
        url = urlparse(request.full_url)
        if url.hostname != "127.0.0.1" or url.scheme != "http":
            raise ValueError("the synthetic transport accepts loopback HTTP URLs only")
        body = request.data or b""
        raw = (f"POST {url.path} HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        connection = Connection(raw)
        Handler(connection, ("127.0.0.1", 0), self.server)
        header, payload = connection.output.getvalue().split(b"\r\n\r\n", 1)
        if not header.startswith(b"HTTP/1.0 200"):
            raise ValueError("mock handler rejected the request")
        return io.BytesIO(payload)


if __name__ == "__main__":
    script = sys.argv[1]; sys.argv = sys.argv[1:]
    transport = Transport(os.environ.get("COOKBOOK_MOCK_MODE", "clean"))
    with patch.object(urllib.request, "urlopen", transport.urlopen):
        runpy.run_path(script, run_name="__main__")
