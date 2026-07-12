from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "src" else SCRIPT_PATH.parent
DEFAULT_OUTPUT_DIR = ROOT / "runtime" / "captures" / "rpi"
DEFAULT_REQUEST_FILE = ROOT / "runtime" / "capture_request.txt"


class CameraState:
    def __init__(self, output_dir: Path, jpeg_quality: int) -> None:
        self.output_dir = output_dir
        self.jpeg_quality = jpeg_quality
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()

    def update_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()
        self._frame_ready.set()

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def jpeg_frame(self) -> bytes | None:
        frame = self.latest_frame()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        return encoded.tobytes() if ok else None

    def save_frame(self, raw_name: str = "") -> Path:
        frame = self.latest_frame()
        if frame is None:
            raise RuntimeError("No camera frame is available yet.")
        output_path = self.output_dir / normalize_image_name(raw_name)
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Failed to save image: {output_path}")
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and capture a Raspberry Pi camera.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--window-name", type=str, default="NUEDC Camera Preview")
    parser.add_argument("--fullscreen", action="store_true", help="Show HDMI preview fullscreen.")
    parser.add_argument(
        "--preview-mode",
        choices=("hdmi", "web", "both", "none"),
        default="hdmi",
        help="Preview destination. Default: hdmi.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Web preview bind address.")
    parser.add_argument("--port", type=int, default=8080, help="Web preview port.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Web preview JPEG quality, 1-100.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-file", type=Path, default=DEFAULT_REQUEST_FILE)
    return parser.parse_args()


def normalize_image_name(raw_name: str) -> str:
    name = raw_name.strip()
    if not name:
        name = datetime.now().strftime("rpi_%Y%m%d_%H%M%S_%f")
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        name += ".jpg"
    return Path(name).name


def consume_capture_request(request_file: Path) -> str | None:
    if not request_file.exists():
        return None
    try:
        raw_name = request_file.read_text(encoding="utf-8").strip()
        request_file.unlink()
    except OSError as exc:
        print(f"WARNING: failed to consume request file: {exc}")
        return None
    return normalize_image_name(raw_name)


def _page_html() -> bytes:
    return b"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>NUEDC Camera</title><style>
body{margin:0;background:#171a1f;color:#eef1f5;font:16px system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:20px}.bar{display:flex;align-items:center;gap:12px;margin-bottom:16px}h1{font-size:20px;margin:0;font-weight:600}button{background:#0b7a75;color:#fff;border:0;border-radius:5px;padding:9px 16px;font:inherit;cursor:pointer}button:disabled{opacity:.6;cursor:wait}#status{color:#b8c2cc;font-size:14px}img{display:block;width:100%;max-height:calc(100vh - 120px);object-fit:contain;background:#050607}
</style></head><body><main><div class=\"bar\"><h1>NUEDC Camera</h1><button id=\"capture\">Capture</button><span id=\"status\">Ready</span></div><img src=\"/stream\" alt=\"Camera preview\"></main>
<script>const b=document.getElementById('capture'),s=document.getElementById('status');b.onclick=async()=>{b.disabled=true;s.textContent='Capturing...';try{const r=await fetch('/capture',{method:'POST'}),j=await r.json();if(!r.ok)throw Error(j.error);s.textContent='Saved: '+j.file}catch(e){s.textContent='Capture failed: '+e.message}finally{b.disabled=false}};</script></body></html>"""


def make_handler(state: CameraState) -> type[BaseHTTPRequestHandler]:
    class PreviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            print(f"WEB {self.client_address[0]} - {format % args}")

        def do_GET(self) -> None:
            if self.path == "/":
                page = _page_html()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if self.path == "/stream":
                self._stream_frames()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/capture":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                output_path = state.save_frame()
                payload = json.dumps({"file": output_path.name}).encode("utf-8")
                self.send_response(HTTPStatus.OK)
            except RuntimeError as exc:
                payload = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _stream_frames(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    if not state._frame_ready.wait(timeout=3):
                        continue
                    encoded = state.jpeg_frame()
                    if encoded is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return

    return PreviewHandler


def start_web_server(state: CameraState, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, name="camera-web", daemon=True)
    thread.start()
    print(f"Web preview: http://<raspberry-pi-ip>:{port}")
    return server


def main() -> int:
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.request_file.parent.mkdir(parents=True, exist_ok=True)
    state = CameraState(args.output_dir, args.jpeg_quality)
    use_hdmi = args.preview_mode in ("hdmi", "both")
    use_web = args.preview_mode in ("web", "both")
    server: ThreadingHTTPServer | None = None

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {args.camera_index}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        if use_web:
            server = start_web_server(state, args.host, args.port)
        if use_hdmi:
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
            if args.fullscreen:
                cv2.setWindowProperty(args.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print(f"Preview mode: {args.preview_mode}")
        print(f"Capture output dir: {args.output_dir}")
        print(f"Remote request file: {args.request_file}")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: camera opened, but no frame was captured.")
                return 2
            state.update_frame(frame)

            requested_name = consume_capture_request(args.request_file)
            if requested_name is not None:
                output_path = state.save_frame(requested_name)
                print(f"saved: {output_path}")

            if use_hdmi:
                cv2.imshow(args.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    output_path = state.save_frame()
                    print(f"saved: {output_path}")
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("Camera service stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if server is not None:
            server.shutdown()
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
