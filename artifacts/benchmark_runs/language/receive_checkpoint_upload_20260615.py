from __future__ import annotations

import argparse
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class UploadHandler(BaseHTTPRequestHandler):
    output_dir: Path
    token: str

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/upload":
            self.send_error(404)
            return
        if self.headers.get("X-Upload-Token", "") != self.token:
            self.send_error(403)
            return
        name = self.headers.get("X-File-Name", "checkpoint_from_3080.pt")
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_error(411)
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.output_dir / safe_name
        with tempfile.NamedTemporaryFile(dir=self.output_dir, delete=False) as handle:
            tmp_path = Path(handle.name)
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        if remaining:
            tmp_path.unlink(missing_ok=True)
            self.send_error(400, "incomplete upload")
            return
        tmp_path.replace(final_path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(final_path).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    UploadHandler.output_dir = args.output_dir
    UploadHandler.token = args.token
    server = ThreadingHTTPServer((args.host, args.port), UploadHandler)
    print(f"UPLOAD_SERVER http://{args.host}:{args.port}/upload -> {args.output_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
