from __future__ import annotations

import json
import math
from datetime import datetime
from http import HTTPStatus

import camera_preview as preview


BASE_MAKE_HANDLER = preview.make_handler


def distance_capture_name(raw_distance_cm: object) -> str:
    """Encode a measured distance in millimeters for reliable batch parsing."""
    try:
        distance_cm = float(str(raw_distance_cm).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("distance_cm must be a number in centimeters.") from exc
    if not math.isfinite(distance_cm) or distance_cm <= 0:
        raise ValueError("distance_cm must be a positive finite number.")

    distance_mm = round(distance_cm * 10.0)
    if abs(distance_cm * 10.0 - distance_mm) > 1e-6:
        raise ValueError("distance_cm must use at most one decimal place.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"d{distance_mm:04d}mm_{timestamp}.jpg"


def _page_html() -> bytes:
    return b"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NUEDC Distance Capture</title><style>
body{margin:0;background:#171a1f;color:#eef1f5;font:16px system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:20px}.bar{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap}h1{font-size:20px;margin:0;font-weight:600}input{width:116px;background:#242a31;color:#eef1f5;border:1px solid #53606d;border-radius:4px;padding:8px;font:inherit}button{background:#0b7a75;color:#fff;border:0;border-radius:5px;padding:9px 16px;font:inherit;cursor:pointer}button:disabled{opacity:.6;cursor:wait}#status{color:#b8c2cc;font-size:14px}img{display:block;width:100%;max-height:calc(100vh - 120px);object-fit:contain;background:#050607}
</style></head><body><main><div class="bar"><h1>NUEDC Distance Capture</h1><label>Distance <input id="distance" type="number" min="0.1" step="0.1" placeholder="cm" required></label><button id="capture">Capture</button><span id="status">Enter measured distance in cm</span></div><img src="/stream" alt="Camera preview"></main>
<script>const b=document.getElementById('capture'),d=document.getElementById('distance'),s=document.getElementById('status');b.onclick=async()=>{const v=d.value.trim();if(!v){s.textContent='Enter distance first';d.focus();return}b.disabled=true;s.textContent='Capturing...';try{const r=await fetch('/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({distance_cm:v})}),j=await r.json();if(!r.ok)throw Error(j.error);s.textContent='Saved: '+j.file}catch(e){s.textContent='Capture failed: '+e.message}finally{b.disabled=false}};</script></body></html>"""


def make_handler(state: preview.CameraState):
    base_handler = BASE_MAKE_HANDLER(state)

    class DistanceCaptureHandler(base_handler):
        def do_POST(self) -> None:
            if self.path != "/capture":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1024:
                    raise ValueError("A small JSON request body is required.")
                request_data = json.loads(self.rfile.read(content_length).decode("utf-8"))
                output_path = state.save_frame(distance_capture_name(request_data.get("distance_cm")))
                payload = json.dumps({"file": output_path.name}).encode("utf-8")
                status = HTTPStatus.OK
            except (ValueError, json.JSONDecodeError) as exc:
                payload = json.dumps({"error": str(exc)}).encode("utf-8")
                status = HTTPStatus.BAD_REQUEST
            except RuntimeError as exc:
                payload = json.dumps({"error": str(exc)}).encode("utf-8")
                status = HTTPStatus.SERVICE_UNAVAILABLE
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return DistanceCaptureHandler


preview._page_html = _page_html
preview.make_handler = make_handler


if __name__ == "__main__":
    raise SystemExit(preview.main())
