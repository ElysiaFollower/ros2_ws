#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import html
import json
import queue
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "bool" | "int" | "float" | "str"
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    help: str = ""


def _coerce_value(kind: str, raw: str) -> Any:
    raw = (raw or "").strip()
    if kind == "bool":
        return raw.lower() in ("1", "true", "t", "yes", "y", "on")
    if kind == "int":
        return int(float(raw))
    if kind == "float":
        return float(raw)
    return raw


def _render_input(spec: ParamSpec, current: Any) -> str:
    name = html.escape(spec.name)
    help_txt = html.escape(spec.help or "")
    if spec.kind == "bool":
        checked = "checked" if bool(current) else ""
        return f"""
        <div class="row">
          <label title="{help_txt}">{name}</label>
          <input type="checkbox" name="{name}" {checked}/>
        </div>
        """

    input_type = "text" if spec.kind == "str" else "number"
    attrs = []
    if spec.min is not None:
        attrs.append(f'min="{spec.min}"')
    if spec.max is not None:
        attrs.append(f'max="{spec.max}"')
    if spec.step is not None:
        attrs.append(f'step="{spec.step}"')
    attrs_s = " ".join(attrs)
    value = html.escape(str(current))
    return f"""
    <div class="row">
      <label title="{help_txt}">{name}</label>
      <input type="{input_type}" name="{name}" value="{value}" {attrs_s}/>
    </div>
    """


class ParamWebPanel:
    def __init__(
        self,
        node: Node,
        *,
        title: str,
        host: str,
        port: int,
        specs: Dict[str, ParamSpec],
        queue_max: int = 100,
        open_browser: bool = False,
    ):
        self._node = node
        self._title = title
        self._host = host
        self._port = int(port)
        self._specs = specs
        self._updates: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=queue_max)

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        self._last_error: Optional[str] = None
        self._last_apply_ts: float = 0.0
        self._open_browser = bool(open_browser)

    @property
    def updates_queue(self) -> "queue.Queue[Dict[str, Any]]":
        return self._updates

    def start(self) -> Tuple[str, int]:
        if self._thread is not None:
            return self._host, self._port

        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send(self, code: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/api/state"):
                    state = panel._get_state()
                    self._send(200, json.dumps(state, ensure_ascii=False).encode("utf-8"), "application/json")
                    return

                if self.path not in ("/", "/index.html"):
                    self._send(404, b"not found", "text/plain")
                    return

                html_body = panel._render_page()
                self._send(200, html_body.encode("utf-8"))

            def do_POST(self) -> None:  # noqa: N802
                if self.path.startswith("/api/set"):
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("payload must be object")
                        panel._enqueue_update(payload)
                        self._send(200, b'{"ok": true}', "application/json")
                    except Exception as e:
                        self._send(400, json.dumps({"ok": False, "error": str(e)}).encode("utf-8"), "application/json")
                    return

                if self.path.startswith("/set"):
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    raw = self.rfile.read(length) if length > 0 else b""
                    form = urllib.parse.parse_qs(raw.decode("utf-8"), keep_blank_values=True)

                    update: Dict[str, Any] = {}
                    for name, spec in panel._specs.items():
                        if spec.kind == "bool":
                            update[name] = name in form
                        else:
                            if name not in form:
                                continue
                            update[name] = form[name][-1]

                    panel._enqueue_update(update)
                    self.send_response(303)
                    self.send_header("Location", "/")
                    self.end_headers()
                    return

                self._send(404, b"not found", "text/plain")

        self._httpd = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://{self._host}:{self._port}/"
        self._node.get_logger().info(f"Param panel: {url}")
        if self._open_browser:
            try:
                webbrowser.open(url, new=1, autoraise=True)
            except Exception as e:
                self._node.get_logger().warning(f"Failed to open browser: {e}")

        return self._host, self._port

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
        self._httpd = None
        self._thread = None

    def _enqueue_update(self, payload: Dict[str, Any]) -> None:
        filtered: Dict[str, Any] = {}
        for k, v in payload.items():
            if k in self._specs:
                filtered[k] = v
        if not filtered:
            return
        try:
            self._updates.put_nowait(filtered)
        except queue.Full:
            self._node.get_logger().warning("Param panel update queue full; dropping update")

    def _get_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "title": self._title,
            "time": time.time(),
            "last_error": self._last_error,
            "last_apply_ts": self._last_apply_ts,
            "params": {},
        }
        for name in self._specs.keys():
            out["params"][name] = self._node.get_parameter(name).value
        return out

    def _render_page(self) -> str:
        state = self._get_state()
        rows = []
        for name, spec in self._specs.items():
            current = state["params"].get(name, "")
            rows.append(_render_input(spec, current))
        rows_html = "\n".join(rows)

        err = html.escape(state.get("last_error") or "")
        err_html = f'<div class="err">{err}</div>' if err else ""

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(self._title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
    .card {{ max-width: 720px; padding: 16px; border: 1px solid #ddd; border-radius: 12px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: center; margin: 8px 0; }}
    label {{ font-size: 14px; color: #222; }}
    input[type=number], input[type=text] {{ width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 8px; }}
    .actions {{ display: flex; gap: 12px; margin-top: 12px; }}
    button {{ padding: 10px 14px; border: 0; border-radius: 10px; background: #1f6feb; color: white; cursor: pointer; }}
    button.secondary {{ background: #6e7781; }}
    .hint {{ font-size: 12px; color: #666; margin-top: 8px; }}
    .err {{ color: #b42318; background: #fee4e2; padding: 10px; border-radius: 10px; margin-bottom: 12px; }}
    code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="card">
    <h2>{html.escape(self._title)}</h2>
    {err_html}
    <form method="POST" action="/set">
      {rows_html}
      <div class="actions">
        <button type="submit">Apply</button>
        <button type="button" class="secondary" onclick="location.reload()">Refresh</button>
      </div>
    </form>
    <div class="hint">
      Changes are applied asynchronously by the node. You can also use <code>ros2 param set</code>.
    </div>
  </div>
</body>
</html>
"""

    def drain_updates(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        while True:
            try:
                item = self._updates.get_nowait()
            except queue.Empty:
                break
            merged.update(item)
        return merged

    def apply_updates(self, updates: Dict[str, Any]) -> bool:
        if not updates:
            return True

        params = []
        errors = []
        for name, raw in updates.items():
            spec = self._specs.get(name)
            if spec is None:
                continue
            try:
                val = _coerce_value(spec.kind, str(raw))
                params.append(Parameter(name, value=val))
            except Exception as e:
                errors.append(f"{name}: {e}")

        if errors:
            self._last_error = "; ".join(errors)
            return False

        results = self._node.set_parameters(params)
        bad = [r.reason for r in results if not r.successful]
        if bad:
            self._last_error = "; ".join([b for b in bad if b] or ["parameter rejected"])
            return False

        self._last_error = None
        self._last_apply_ts = time.time()
        return True

