"""Loopback-only mock chat logprobs and typed decisions; never loads a model."""
import argparse
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from synthetic_model import predict


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/systemone":
            probs = predict(payload["state"])
            result = {"answers": {q: {"noul": probs[q]} for q in payload["questions"]}}
        elif self.path == "/v1/chat/completions":
            user = payload["messages"][-1]["content"]
            self.server.calls += 1
            high = self.server.mode != "jitter" or self.server.calls % 2 == 0
            p = 0.99 if high else 0.85
            if "single letter only" in user:
                candidates = re.findall(r"^([A-Z])\. (.*)$", user, re.M)
                task = user.split("\n\n", 1)[0].lower()
                target = "support-analyst" if "support tickets" in task else "data-analyst"
                if "unrelated noise" in task:
                    target = "None of these"
                winner = next((key for key, text in candidates if text.startswith(target)), candidates[-1][0])
                options = [key for key, _ in candidates]
            else:
                options = ["Yes", "No"]
                winner = "No" if "already cited" in user.lower() else "Yes"
            rest = (1 - p) / max(1, len(options) - 1)
            raw = [{"token": opt, "logprob": math.log(p if opt == winner else rest)} for opt in options]
            logprobs = None if self.server.mode == "none-logprobs" else {"content": [{"top_logprobs": raw}]}
            result = {"choices": [{"message": {"role": "assistant", "content": winner}, "logprobs": logprobs}]}
        else:
            self.send_error(404); return
        body = json.dumps(result).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def make_server(port=8000, mode="clean"):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.mode = mode; server.calls = 0
    return server


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", choices=("clean", "jitter", "none-logprobs"), default="clean")
    a = p.parse_args()
    with make_server(a.port, a.mode) as server:
        print(f"mock listening on http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
