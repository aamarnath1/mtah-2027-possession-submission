"""Serve the match viewer on this machine only (127.0.0.1). Nothing leaves your computer.

Routes:
    /                    the viewer page
    /static/<file>       viewer page assets
    /data/<file>         prepared viewer data (meta.json, positions.f32, ...)
    /video               the tactical MP4, with HTTP Range support so seeking works
    /api/labels          JSON list of label CSVs found in the label folders
    /labels/<name>       one label CSV (frame,label)

Usage:
    python tools/viewer/serve_viewer.py
    python tools/viewer/serve_viewer.py --video data/raw/tactical/BAR-ATH_tactical.mp4 --port 8765
"""

import argparse
import json
import mimetypes
import re
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
CHUNK = 1 << 20


def _inside(base, name):
    """Resolve name inside base, or None if it would escape base."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    path = (base / name).resolve()
    return path if path.parent == base.resolve() and path.is_file() else None


class ViewerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, config, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # keep the console quiet apart from errors
        if args and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._file(STATIC / "index.html", head)
        if path.startswith("/static/"):
            return self._file(_inside(STATIC, path[len("/static/"):]), head)
        if path.startswith("/data/"):
            return self._file(_inside(self.config["data_dir"], path[len("/data/"):]), head)
        if path == "/video":
            video = self.config["video"]
            return self._file(video if video and video.is_file() else None, head, ranged=True)
        if path == "/api/labels":
            return self._json(self._label_files(), head)
        if path.startswith("/labels/"):
            return self._file(self._label_files().get(path[len("/labels/"):]), head)
        self.send_error(HTTPStatus.NOT_FOUND)

    def _label_files(self):
        files = {}
        for folder in self.config["label_dirs"]:
            if folder.is_dir():
                for csv_path in sorted(folder.glob("*.csv")):
                    key = csv_path.name if csv_path.name not in files else \
                        f"{folder.name}__{csv_path.name}"
                    files[key] = csv_path
        return files

    def _json(self, files, head):
        body = json.dumps(sorted(files)).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _file(self, path, head, ranged=False):
        if path is None or not Path(path).is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        header = self.headers.get("Range")
        if ranged and header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not match or (not match.group(1) and not match.group(2)):
                return self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:  # suffix range: last N bytes
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".f32", ".f64"):
            content_type = "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if ranged:
            self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        try:
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser cancels range requests all the time while seeking


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(REPO / "data" / "viewer"),
                        help="prepared viewer data (default: data/viewer)")
    parser.add_argument("--video", default=str(REPO / "data" / "raw" / "tactical" /
                                               "BAR-ATH_tactical.mp4"),
                        help="tactical MP4 (optional; the pitch view works without it)")
    parser.add_argument("--labels", action="append", default=None,
                        help="folder of label CSVs; repeatable "
                             "(default: submissions/ and data/predictions/)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not (data_dir / "meta.json").is_file():
        raise SystemExit(f"No meta.json in {data_dir}. Run tools/viewer/prepare_viewer_from_current.py first.")
    video = Path(args.video).resolve() if args.video else None
    label_dirs = [Path(d).resolve() for d in (args.labels or
                  [REPO / "submissions", REPO / "data" / "predictions"])]
    config = {"data_dir": data_dir, "video": video, "label_dirs": label_dirs}

    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(ViewerHandler, config=config))
    print(f"Viewer: http://127.0.0.1:{args.port}/")
    print(f"  video:  {video if video and video.is_file() else 'not found (pitch view only)'}")
    print(f"  labels: {', '.join(str(d) for d in label_dirs)}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
