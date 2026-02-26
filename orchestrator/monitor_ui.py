"""Live oversight UI for the orchestrator (runs alongside the agent loop).

The monitor is independent of debug mode and provides:
  - live iteration timeline
  - image batches requested / viewed
  - request rationale and rendered judgments
  - pause/resume controls
  - supervisor interventions (guidance / follow-up questions)
"""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def _json_deepcopy(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _first_user_text(messages: list[dict]) -> str | None:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        return text
    return None


def _message_image_data_urls(messages: list[dict]) -> list[str]:
    urls: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = image_url.get("url")
            if isinstance(url, str) and url.startswith("data:image/png;base64,"):
                urls.append(url)
    return urls


def _extract_images_from_result(result: Any) -> list[dict[str, Any]]:
    """Extract image-bearing instance records from MCP results (instances/items/root/list)."""

    out: list[dict[str, Any]] = []

    def add(root: dict[str, Any] | None, inst: dict[str, Any], source: str) -> None:
        png_b64 = inst.get("png_base64")
        if not isinstance(png_b64, str) or not png_b64:
            return
        out.append(
            {
                "png_base64": png_b64,
                "series_id": (root or {}).get("series_id") or inst.get("series_id"),
                "index": inst.get("index"),
                "instance_number": inst.get("instance_number"),
                "file_name": inst.get("file_name"),
                "path": inst.get("path"),
                "rows": inst.get("rows"),
                "columns": inst.get("columns"),
                "render": inst.get("render"),
                "resolved_indices": (root or {}).get("resolved_indices"),
                "requested_ranges": (root or {}).get("requested_ranges"),
                "requested_range": (root or {}).get("requested_range"),
                "step": (root or {}).get("step"),
                "window": (root or {}).get("window"),
                "crop": (root or {}).get("crop"),
                "source": source,
            }
        )

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "png_base64" in value:
                add(None, value, "root")
            for key in ("instances", "items"):
                nested = value.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        if isinstance(item, dict):
                            add(value, item, key)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    if "png_base64" in item:
                        add(None, item, "list")
                    else:
                        walk(item)

    walk(result)
    return out


def _looks_like_placeholder_image(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def _strip_png_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k == "png_base64" and isinstance(v, str) and v:
                out[k] = "<image>"
            elif k in {"instances", "items"} and isinstance(v, list):
                # Keep list shape but avoid giant payloads.
                out[k] = [_strip_png_fields(item) for item in v[:5]]
                if len(v) > 5:
                    out[f"{k}_truncated_count"] = len(v) - 5
            else:
                out[k] = _strip_png_fields(v)
        return out
    if isinstance(value, list):
        return [_strip_png_fields(v) for v in value[:5]] + (
            [{"_truncated": len(value) - 5}] if len(value) > 5 else []
        )
    return value


def _result_summary(result: Any) -> str:
    if result is None:
        return "none"
    if isinstance(result, dict):
        for key in ("instances", "items"):
            items = result.get(key)
            if isinstance(items, list) and items:
                return f"{len(items)} {key}"
        return f"dict keys={list(result.keys())[:8]}"
    if isinstance(result, list):
        return f"{len(result)} entries"
    return str(result)[:120]


def _minimal_config_dict(config: Any) -> dict[str, Any]:
    return {
        "dicom_root": getattr(config, "dicom_root", None),
        "model": getattr(config, "model", None),
        "base_url": getattr(config, "base_url", None),
        "max_iterations": getattr(config, "max_iterations", None),
        "default_media_resolution": getattr(config, "default_media_resolution", None),
        "default_thinking_level": getattr(config, "default_thinking_level", None),
        "prompt_path": getattr(config, "prompt_path", None),
        "clinical_question": getattr(config, "clinical_question", None),
        "debug": getattr(config, "debug", None),
        "monitor_ui": getattr(config, "monitor_ui", None),
        "monitor_host": getattr(config, "monitor_host", None),
        "monitor_port": getattr(config, "monitor_port", None),
    }


class NullOrchestratorMonitor:
    """No-op monitor used when live oversight UI is disabled."""

    url: str | None = None

    def start(self, config: Any) -> None:
        return

    def close(self) -> None:
        return

    def set_phase(self, phase: str, iteration: int | None = None) -> None:
        return

    def wait_if_paused(self, phase: str, iteration: int | None = None) -> None:
        return

    def consume_interventions(self, iteration: int | None = None) -> list[dict[str, Any]]:
        return []

    def request_stop(self) -> None:
        return

    def is_stop_requested(self) -> bool:
        return False

    def record_available_images(self, available_images: list[dict]) -> None:
        return

    def record_iteration_state(self, iteration: int, state_dict: dict[str, Any]) -> None:
        return

    def record_messages(self, iteration: int, messages: list[dict]) -> None:
        return

    def record_applied_interventions(self, iteration: int, interventions: list[dict[str, Any]]) -> None:
        return

    def record_parsed_response(self, iteration: int, response_dict: dict[str, Any]) -> None:
        return

    def record_parse_error(self, iteration: int, error_text: str) -> None:
        return

    def record_convergence(self, iteration: int, *, converged: bool, reasons: list[str], next_request_null: bool) -> None:
        return

    def record_mcp_execution(
        self,
        iteration: int,
        *,
        rationale: str | None,
        suggested_thinking_level: str | None,
        suggested_media_resolution: str | None,
        mcp_calls: list[dict],
        mcp_results: list[dict],
    ) -> None:
        return

    def record_final_report(self, final_report: dict[str, Any]) -> None:
        return

    def record_annotated_images(
        self,
        iteration: int,
        annotated_b64: list[str],
        response_dict: dict[str, Any],
    ) -> None:
        return

    def record_status(self, status: str, message: str | None = None) -> None:
        return


@dataclass
class _Intervention:
    id: int
    kind: str
    text: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "created_at": self.created_at,
        }


class LiveOrchestratorMonitor:
    """Threaded local HTTP server + in-memory state for live oversight."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending_interventions: list[_Intervention] = []
        self._next_intervention_id = 1
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._session_dir: Path | None = None
        self._assets_dir: Path | None = None
        self.url: str | None = None

        self._state: dict[str, Any] = {
            "session": {
                "started_at": None,
                "status": "idle",
                "phase": "idle",
                "current_iteration": None,
                "paused": False,
                "stop_requested": False,
                "url": None,
                "run_id": None,
                "session_dir": None,
            },
            "config": {},
            "available_images": [],
            "iterations": {},
            "interventions": {
                "pending": [],
                "history": [],
            },
            "final_report": None,
            "errors": [],
        }

    # Public lifecycle -------------------------------------------------

    def start(self, config: Any) -> None:
        with self._lock:
            if self._server is not None:
                return

            session_dir = self._resolve_session_dir(config)
            assets_dir = session_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            self._session_dir = session_dir
            self._assets_dir = assets_dir

            self._state["session"].update(
                {
                    "started_at": _utc_now(),
                    "status": "running",
                    "phase": "initializing",
                    "run_id": session_dir.name,
                    "session_dir": str(session_dir.resolve()),
                }
            )
            self._state["config"] = _minimal_config_dict(config)

        host = getattr(config, "monitor_host", "127.0.0.1")
        port = int(getattr(config, "monitor_port", 8765))

        monitor = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "OrchestratorMonitor/1.0"

            def log_message(self, fmt: str, *args) -> None:  # silence default logging
                return

            def _json_response(self, status: int, payload: Any) -> None:
                data = json.dumps(payload, indent=2, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _bytes_response(
                self,
                status: int,
                body: bytes,
                content_type: str,
                *,
                cache_control: str = "no-store",
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", cache_control)
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    n = 0
                raw = self.rfile.read(n) if n > 0 else b"{}"
                if not raw:
                    return {}
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/":
                    self._bytes_response(HTTPStatus.OK, monitor._render_ui_html().encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == "/api/state":
                    self._json_response(HTTPStatus.OK, monitor.snapshot_state())
                    return
                if path.startswith("/assets/"):
                    file_path = monitor._asset_path_for_request(path)
                    if file_path is None or not file_path.exists() or not file_path.is_file():
                        self._json_response(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
                        return
                    content_type, _ = mimetypes.guess_type(str(file_path))
                    self._bytes_response(
                        HTTPStatus.OK,
                        file_path.read_bytes(),
                        content_type or "application/octet-stream",
                        cache_control="public, max-age=31536000, immutable",
                    )
                    return
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                payload = self._read_json()

                if path == "/api/interventions":
                    kind = str(payload.get("kind", "directive")).strip().lower() or "directive"
                    text = str(payload.get("text", "")).strip()
                    if kind not in {"directive", "question"}:
                        self._json_response(HTTPStatus.BAD_REQUEST, {"error": "kind must be 'directive' or 'question'"})
                        return
                    if not text:
                        self._json_response(HTTPStatus.BAD_REQUEST, {"error": "text is required"})
                        return
                    item = monitor.submit_intervention(kind=kind, text=text)
                    self._json_response(HTTPStatus.OK, {"ok": True, "intervention": item})
                    return

                if path == "/api/control":
                    action = str(payload.get("action", "")).strip().lower()
                    if action == "pause":
                        monitor.set_paused(True)
                        self._json_response(HTTPStatus.OK, {"ok": True, "paused": True})
                        return
                    if action == "resume":
                        monitor.set_paused(False)
                        self._json_response(HTTPStatus.OK, {"ok": True, "paused": False})
                        return
                    if action == "stop":
                        monitor.request_stop()
                        self._json_response(HTTPStatus.OK, {"ok": True, "stop_requested": True})
                        return
                    self._json_response(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "action must be 'pause', 'resume', or 'stop'"},
                    )
                    return

                self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})

        server = ThreadingHTTPServer((host, port), _Handler)
        server.daemon_threads = True
        self._server = server
        actual_host, actual_port = server.server_address[:2]
        self.url = f"http://{actual_host}:{actual_port}"
        with self._lock:
            self._state["session"]["url"] = self.url

        thread = threading.Thread(target=server.serve_forever, name="orchestrator-monitor-ui", daemon=True)
        thread.start()
        self._thread = thread

        print(f"[monitor-ui] Live oversight UI: {self.url}")
        print(f"[monitor-ui] Assets/session dir: {self._session_dir}")

    def close(self) -> None:
        server = self._server
        thread = self._thread
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    # Runner integration hooks ----------------------------------------

    def set_phase(self, phase: str, iteration: int | None = None) -> None:
        with self._lock:
            self._state["session"]["phase"] = phase
            self._state["session"]["current_iteration"] = iteration
            self._state["session"]["updated_at"] = _utc_now()

    def wait_if_paused(self, phase: str, iteration: int | None = None) -> None:
        self.set_phase(phase, iteration)
        while True:
            with self._lock:
                paused = bool(self._state["session"].get("paused"))
                stop_requested = bool(self._state["session"].get("stop_requested"))
            if not paused or stop_requested:
                return
            time.sleep(0.2)

    def record_status(self, status: str, message: str | None = None) -> None:
        with self._lock:
            self._state["session"]["status"] = status
            self._state["session"]["updated_at"] = _utc_now()
            if message:
                self._state["session"]["status_message"] = message

    def record_available_images(self, available_images: list[dict]) -> None:
        with self._lock:
            self._state["available_images"] = _json_deepcopy(available_images)
            self._state["session"]["updated_at"] = _utc_now()

    def record_iteration_state(self, iteration: int, state_dict: dict[str, Any]) -> None:
        it = self._iter_entry(iteration)
        with self._lock:
            it["iteration_state"] = _json_deepcopy(state_dict)
            it["updated_at"] = _utc_now()

    def record_messages(self, iteration: int, messages: list[dict]) -> None:
        it = self._iter_entry(iteration)
        viewed_images = self._save_viewed_images(iteration, messages)
        with self._lock:
            it["messages"] = {
                "image_count": len(viewed_images),
                "user_text_preview": _first_user_text(messages),
                "viewed_images": viewed_images,
            }
            it["updated_at"] = _utc_now()

    def record_applied_interventions(self, iteration: int, interventions: list[dict[str, Any]]) -> None:
        if not interventions:
            return
        it = self._iter_entry(iteration)
        with self._lock:
            it.setdefault("applied_interventions", [])
            it["applied_interventions"].extend(_json_deepcopy(interventions))
            it["updated_at"] = _utc_now()

    def record_parsed_response(self, iteration: int, response_dict: dict[str, Any]) -> None:
        it = self._iter_entry(iteration)
        with self._lock:
            it["agent_response"] = _json_deepcopy(response_dict)
            it["parse_error"] = None
            it["updated_at"] = _utc_now()

    def record_parse_error(self, iteration: int, error_text: str) -> None:
        it = self._iter_entry(iteration)
        with self._lock:
            it["parse_error"] = error_text
            it["updated_at"] = _utc_now()

    def record_convergence(
        self,
        iteration: int,
        *,
        converged: bool,
        reasons: list[str],
        next_request_null: bool,
    ) -> None:
        it = self._iter_entry(iteration)
        with self._lock:
            it["convergence"] = {
                "converged": converged,
                "reasons": _json_deepcopy(reasons),
                "next_request_null": next_request_null,
            }
            it["updated_at"] = _utc_now()

    def record_mcp_execution(
        self,
        iteration: int,
        *,
        rationale: str | None,
        suggested_thinking_level: str | None,
        suggested_media_resolution: str | None,
        mcp_calls: list[dict],
        mcp_results: list[dict],
    ) -> None:
        it = self._iter_entry(iteration)
        extracted_images, call_summaries = self._save_extracted_images_and_summaries(iteration, mcp_results)

        with self._lock:
            it["request_batch"] = {
                "rationale": rationale,
                "suggested_thinking_level": suggested_thinking_level,
                "suggested_media_resolution": suggested_media_resolution,
                "mcp_calls": _json_deepcopy(mcp_calls),
                "call_summaries": call_summaries,
                "images": extracted_images,
                "extracted_image_count": len(extracted_images),
            }
            it["updated_at"] = _utc_now()

    def record_annotated_images(
        self,
        iteration: int,
        annotated_b64: list[str],
        response_dict: dict[str, Any],
    ) -> None:
        """Save annotated PNGs (with bounding boxes drawn) and attach metadata to iteration state."""
        it = self._iter_entry(iteration)
        saved: list[dict[str, Any]] = []

        if self._assets_dir is not None and annotated_b64:
            dst_dir = self._assets_dir / f"iter_{iteration:03d}" / "annotated"
            dst_dir.mkdir(parents=True, exist_ok=True)

            for idx, b64 in enumerate(annotated_b64):
                if not b64:
                    continue
                file_name = f"annotated_{idx:03d}.png"
                dst = dst_dir / file_name
                try:
                    dst.write_bytes(base64.b64decode(b64))
                except Exception:
                    continue
                saved.append({
                    "slot": idx,
                    "src": f"/assets/iter_{iteration:03d}/annotated/{file_name}",
                })

        # Extract bounding box metadata from the response for frontend rendering
        boxes_meta: list[dict[str, Any]] = []
        for f in response_dict.get("findings", []):
            for bb in f.get("bounding_boxes", []):
                if not isinstance(bb, dict):
                    continue
                boxes_meta.append({
                    "finding_id": f.get("finding_id", ""),
                    "region": f.get("region", ""),
                    "confidence": f.get("confidence", ""),
                    "label": bb.get("label", ""),
                    "box_2d": bb.get("box_2d", []),
                    "slice_ref": bb.get("slice_ref", ""),
                })

        with self._lock:
            it["annotated_images"] = saved
            it["bounding_boxes"] = boxes_meta
            it["updated_at"] = _utc_now()

    def record_final_report(self, final_report: dict[str, Any]) -> None:
        with self._lock:
            self._state["final_report"] = _json_deepcopy(final_report)
            self._state["session"]["updated_at"] = _utc_now()

    # Interventions -----------------------------------------------------

    def submit_intervention(self, *, kind: str, text: str) -> dict[str, Any]:
        with self._lock:
            item = _Intervention(
                id=self._next_intervention_id,
                kind=kind,
                text=text,
                created_at=_utc_now(),
            )
            self._next_intervention_id += 1
            self._pending_interventions.append(item)
            self._state["interventions"]["pending"].append(item.to_dict())
            self._state["session"]["updated_at"] = _utc_now()
            return item.to_dict()

    def consume_interventions(self, iteration: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if not self._pending_interventions:
                return []
            consumed = [i.to_dict() for i in self._pending_interventions]
            self._pending_interventions.clear()
            pending_ids = {c["id"] for c in consumed}
            pending_list = self._state["interventions"]["pending"]
            remaining = [p for p in pending_list if p.get("id") not in pending_ids]
            self._state["interventions"]["pending"] = remaining
            for c in consumed:
                c["applied_at"] = _utc_now()
                c["applied_iteration"] = iteration
                self._state["interventions"]["history"].append(c)
            self._state["session"]["updated_at"] = _utc_now()
            return consumed

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._state["session"]["paused"] = bool(paused)
            self._state["session"]["updated_at"] = _utc_now()

    def request_stop(self) -> None:
        with self._lock:
            self._state["session"]["stop_requested"] = True
            self._state["session"]["updated_at"] = _utc_now()

    def is_stop_requested(self) -> bool:
        with self._lock:
            return bool(self._state["session"].get("stop_requested"))

    # State access ------------------------------------------------------

    def snapshot_state(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    # Internal helpers --------------------------------------------------

    def _resolve_session_dir(self, config: Any) -> Path:
        if getattr(config, "monitor_dir", None):
            return Path(config.monitor_dir).expanduser().resolve()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (Path("monitor_runs") / ts).resolve()

    def _iter_entry(self, iteration: int) -> dict[str, Any]:
        key = f"{iteration}"
        with self._lock:
            entry = self._state["iterations"].get(key)
            if entry is None:
                entry = {
                    "iteration": iteration,
                    "created_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "iteration_state": None,
                    "messages": None,
                    "agent_response": None,
                    "parse_error": None,
                    "convergence": None,
                    "request_batch": None,
                    "applied_interventions": [],
                    "annotated_images": [],
                    "bounding_boxes": [],
                }
                self._state["iterations"][key] = entry
            return entry

    def _asset_path_for_request(self, request_path: str) -> Path | None:
        if self._assets_dir is None:
            return None
        rel = request_path.removeprefix("/assets/").lstrip("/")
        candidate = (self._assets_dir / rel).resolve()
        try:
            candidate.relative_to(self._assets_dir.resolve())
        except Exception:
            return None
        return candidate

    def _save_viewed_images(self, iteration: int, messages: list[dict]) -> list[dict[str, Any]]:
        if self._assets_dir is None:
            return []

        urls = _message_image_data_urls(messages)
        out: list[dict[str, Any]] = []
        if not urls:
            return out

        dst_dir = self._assets_dir / f"iter_{iteration:03d}" / "viewed"
        dst_dir.mkdir(parents=True, exist_ok=True)

        for idx, url in enumerate(urls):
            b64 = url.split(",", 1)[1]
            file_name = f"viewed_{idx:03d}.png"
            path = dst_dir / file_name
            try:
                path.write_bytes(base64.b64decode(b64))
            except Exception:
                continue
            out.append(
                {
                    "slot": idx,
                    "src": f"/assets/iter_{iteration:03d}/viewed/{file_name}",
                }
            )
        return out

    def _save_extracted_images_and_summaries(
        self,
        iteration: int,
        mcp_results: list[dict],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self._assets_dir is None:
            return [], []

        iter_dir = self._assets_dir / f"iter_{iteration:03d}" / "extracted"
        iter_dir.mkdir(parents=True, exist_ok=True)

        images_out: list[dict[str, Any]] = []
        call_summaries: list[dict[str, Any]] = []
        global_img_idx = 0

        for call_idx, row in enumerate(mcp_results or []):
            if not isinstance(row, dict):
                continue

            tool_name = str(row.get("tool_name", "?"))
            arguments = row.get("arguments") or {}
            error = row.get("error")
            result = row.get("result")
            raw_images = [x for x in (row.get("images") or []) if isinstance(x, str) and x]

            extracted = _extract_images_from_result(result)
            if not extracted and raw_images:
                extracted = [{"png_base64": b64, "source": "images"} for b64 in raw_images]
            for local_idx, img in enumerate(extracted):
                png_b64 = img.get("png_base64")
                if _looks_like_placeholder_image(png_b64):
                    if local_idx < len(raw_images):
                        png_b64 = raw_images[local_idx]
                    else:
                        png_b64 = None
                if not isinstance(png_b64, str) or not png_b64:
                    continue

                file_parts = [
                    f"c{call_idx:02d}",
                    _slug(tool_name),
                    f"g{global_img_idx:03d}",
                    f"l{local_idx:02d}",
                ]
                if img.get("series_id"):
                    file_parts.append(_slug(str(img["series_id"])))
                if img.get("index") is not None:
                    file_parts.append(f"idx{img['index']}")
                file_name = "_".join(file_parts) + ".png"
                dst = iter_dir / file_name
                try:
                    dst.write_bytes(base64.b64decode(png_b64))
                except Exception:
                    continue

                images_out.append(
                    {
                        "call_index": call_idx,
                        "tool_name": tool_name,
                        "arguments": _json_deepcopy(arguments),
                        "src": f"/assets/iter_{iteration:03d}/extracted/{file_name}",
                        "series_id": img.get("series_id"),
                        "index": img.get("index"),
                        "instance_number": img.get("instance_number"),
                        "file_name": img.get("file_name"),
                        "path": img.get("path"),
                        "rows": img.get("rows"),
                        "columns": img.get("columns"),
                        "render": _json_deepcopy(img.get("render")),
                        "resolved_indices": _json_deepcopy(img.get("resolved_indices")),
                        "requested_ranges": _json_deepcopy(img.get("requested_ranges")),
                        "requested_range": _json_deepcopy(img.get("requested_range")),
                        "step": img.get("step"),
                        "window": _json_deepcopy(img.get("window")),
                        "crop": _json_deepcopy(img.get("crop")),
                    }
                )
                global_img_idx += 1

            call_summaries.append(
                {
                    "call_index": call_idx,
                    "tool_name": tool_name,
                    "arguments": _json_deepcopy(arguments),
                    "error": error,
                    "summary": _result_summary(result),
                    "extracted_image_count": max(len(extracted), len(raw_images)),
                    "result_preview": _strip_png_fields(result),
                }
            )

        return images_out, call_summaries

    def _render_ui_html(self) -> str:
        # Single-page app (polling /api/state)
        html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Orchestrator Live Oversight</title>
  <style>
    :root {
      --bg: #f5efe5;
      --paper: #fffaf0;
      --ink: #1f1a14;
      --muted: #6b6255;
      --line: #d9cfbd;
      --accent: #0c6a6f;
      --warn: #b35a00;
      --ok: #1b7a3a;
      --error: #b3261e;
      --radius: 14px;
      --shadow: 0 12px 36px rgba(31, 26, 20, 0.08);
      --mono: "IBM Plex Mono", Menlo, Consolas, monospace;
      --sans: "Avenir Next", "Segoe UI Variable", "Trebuchet MS", sans-serif;
      --serif: "Iowan Old Style", Georgia, serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      line-height: 1.4;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 0%, rgba(12,106,111,.10), transparent 42%),
        radial-gradient(circle at 96% 8%, rgba(179,90,0,.09), transparent 35%),
        linear-gradient(180deg, #f7f2e9 0%, var(--bg) 100%);
    }
    .app {
      display: grid;
      grid-template-columns: clamp(300px, 24vw, 380px) minmax(0, 1fr);
      gap: 22px;
      min-height: 100vh;
      padding: 20px;
      width: min(1880px, 100%);
      margin: 0 auto;
    }
    .sidebar, .panel, .hero, .batch {
      background: rgba(255,250,240,.95);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .sidebar {
      position: sticky;
      top: 20px;
      align-self: start;
      overflow: auto;
      max-height: calc(100vh - 40px);
    }
    .side-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(12,106,111,.08), rgba(179,90,0,.05));
    }
    .side-head h1 {
      margin: 0 0 6px;
      font-family: var(--serif);
      font-size: 1.05rem;
    }
    .muted { color: var(--muted); }
    .mono { font-family: var(--mono); }
    .side-body { padding: 16px; display: grid; gap: 16px; }
    .controls { display: grid; gap: 8px; }
    .button-row { display: flex; gap: 8px; flex-wrap: wrap; }
    button {
      border: 1px solid var(--line);
      background: #fffdf7;
      color: var(--ink);
      border-radius: 10px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.primary { border-color: rgba(12,106,111,.35); color: var(--accent); }
    button.warn { border-color: rgba(179,90,0,.3); color: var(--warn); }
    textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fffdf8;
      color: var(--ink);
      font: inherit;
      padding: 9px 10px;
    }
    textarea { min-height: 90px; resize: vertical; }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      border: 1px solid var(--line);
      background: #fffdf7;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: .74rem;
      white-space: nowrap;
    }
    .chip.ok { color: var(--ok); border-color: rgba(27,122,58,.28); }
    .chip.warn { color: var(--warn); border-color: rgba(179,90,0,.28); }
    .chip.error { color: var(--error); border-color: rgba(179,38,30,.28); }
    .chip-row { display: flex; gap: 6px; flex-wrap: wrap; }
    .main { display: grid; gap: 18px; min-width: 0; }
    .hero { padding: 16px 18px; }
    .hero h2 { margin: 0 0 6px; font-family: var(--serif); font-size: 1.35rem; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fffdf7;
      padding: 10px 12px;
    }
    .metric .k { color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; }
    .metric .v { font-weight: 700; margin-top: 2px; }
    .batch {
      overflow: hidden;
    }
    .batch-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(12,106,111,.06), rgba(179,90,0,.03));
      display: grid;
      gap: 6px;
    }
    .batch-head h3 {
      margin: 0;
      font-size: 1rem;
      font-family: var(--serif);
    }
    .batch-body { padding: 14px 16px; display: grid; gap: 14px; }
    .block {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fffdf8;
      padding: 12px;
    }
    .block h4 {
      margin: 0 0 8px;
      font-size: .82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .05em;
    }
    .rationale {
      line-height: 1.45;
      margin: 0;
      border-left: 4px solid var(--accent);
      padding-left: 10px;
      max-width: 110ch;
    }
    .calls { display: grid; gap: 8px; }
    details.call {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      overflow: hidden;
    }
    details.call > summary {
      cursor: pointer;
      list-style: none;
      padding: 8px 10px;
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      border-bottom: 1px solid rgba(217,207,189,.55);
    }
    details.call > summary::-webkit-details-marker { display:none; }
    .call-name { color: var(--accent); font-weight: 700; }
    .call-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      padding: 12px;
    }
    pre {
      margin: 0;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: .78rem;
      line-height: 1.5;
      background: #fbf7ee;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }
    .thumb-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 12px;
    }
    .thumb {
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      background: #fffdf7;
      position: relative;
    }
    .thumb img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      display: block;
      background:
        linear-gradient(45deg, #ece4d4 25%, #f5eee1 25%, #f5eee1 50%, #ece4d4 50%, #ece4d4 75%, #f5eee1 75%, #f5eee1 100%);
      background-size: 18px 18px;
    }
    .thumb .thumb-media { position: relative; }
    .thumb .img-stack { position: relative; }
    .thumb .img-stack img { transition: opacity .15s; }
    .thumb .img-stack img.hidden { display: none; }
    .thumb .img-stack a.hidden { display: none; }
    .thumb .source-overlay {
      position: absolute;
      left: 6px;
      right: 6px;
      bottom: 6px;
      z-index: 2;
      background: rgba(31, 26, 20, 0.88);
      color: #fff8ec;
      border: 1px solid rgba(255,255,255,.15);
      border-radius: 8px;
      padding: 6px 7px;
      font-size: .69rem;
      line-height: 1.35;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity .15s ease, transform .15s ease;
      pointer-events: none;
      max-height: 62%;
      overflow: auto;
      box-shadow: 0 8px 24px rgba(0,0,0,.2);
    }
    .thumb:hover .source-overlay,
    .thumb:focus-within .source-overlay {
      opacity: 1;
      transform: translateY(0);
    }
    .thumb .source-overlay .row { margin-top: 2px; }
    .thumb .source-overlay .row:first-child { margin-top: 0; }
    .thumb .source-overlay .k { color: rgba(255,248,236,.75); }
    .thumb .cap { padding: 10px; display: grid; gap: 5px; font-size: .8rem; }
    .thumb .sub { color: var(--muted); font-size: .76rem; }
    .thumb .cap-chips { display:flex; flex-wrap:wrap; gap:4px; margin-top: 2px; }
    .thumb .mini-chip {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 2px 6px;
      font-size: .72rem;
      color: var(--muted);
    }
    .toggle-row {
      display: flex;
      gap: 6px;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }
    .toggle-btn {
      border: 1px solid var(--line);
      background: #fffdf7;
      border-radius: 8px;
      padding: 5px 10px;
      font: inherit;
      font-size: .78rem;
      cursor: pointer;
      transition: border-color .15s, background .15s;
    }
    .toggle-btn.active {
      border-color: var(--accent);
      background: rgba(12,106,111,.08);
      color: var(--accent);
      font-weight: 600;
    }
    .bb-badge {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      font-size: .72rem;
      border-radius: 6px;
      padding: 2px 6px;
      border: 1px solid;
    }
    .bb-badge.high { color: #b3261e; border-color: rgba(179,38,30,.3); background: rgba(179,38,30,.06); }
    .bb-badge.moderate { color: #b35a00; border-color: rgba(179,90,0,.3); background: rgba(179,90,0,.06); }
    .bb-badge.low { color: #0c6a6f; border-color: rgba(12,106,111,.3); background: rgba(12,106,111,.06); }
    .thumb .title { font-weight: 700; }
    .judgments { display: grid; gap: 8px; }
    .judgment-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
    }
    .interventions-list { display: grid; gap: 6px; max-height: 180px; overflow: auto; }
    .intervention {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fffdf8;
      padding: 8px;
      font-size: .8rem;
    }
    .small { font-size: .78rem; }
    .err { color: var(--error); }
    @media (max-width: 1400px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { position: static; max-height: none; }
      .call-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .metrics { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .call-grid { grid-template-columns: 1fr; }
      .thumb-grid { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="side-head">
        <h1>Live Oversight UI</h1>
        <div id="sessionMeta" class="muted small">Loading…</div>
      </div>
      <div class="side-body">
        <div class="controls panel">
          <div class="chip-row" id="statusChips"></div>
          <div class="button-row">
            <button class="warn" id="pauseBtn" type="button">Pause</button>
            <button class="primary" id="resumeBtn" type="button">Resume</button>
          </div>
          <div class="button-row">
            <button class="warn" id="stopBtn" type="button">Stop &amp; Finalize</button>
            <button class="primary" id="exportFinalBtn" type="button">Export Final Report</button>
          </div>
          <div id="controlMsg" class="muted small"></div>
        </div>

        <div class="panel">
          <div class="small muted" style="margin-bottom:6px">Intervene before the next LLM iteration. Guidance is injected as either a prompt directive or follow-up user query.</div>
          <label class="small muted" for="intKind">Type</label>
          <select id="intKind">
            <option value="directive">Guidance Directive</option>
            <option value="question">Follow-up Query</option>
          </select>
          <label class="small muted" for="intText" style="margin-top:8px; display:block;">Text</label>
          <textarea id="intText" placeholder="Example: Focus on Achilles myotendinous junction and explain which series best supports the conclusion."></textarea>
          <div class="button-row" style="margin-top:8px">
            <button class="primary" id="sendInterventionBtn" type="button">Queue Intervention</button>
          </div>
          <div id="intMsg" class="muted small" style="margin-top:6px"></div>
        </div>

        <div class="panel">
          <h4 style="margin:0 0 8px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-size:.8rem">Pending Interventions</h4>
          <div id="pendingInterventions" class="interventions-list"></div>
        </div>
        <div class="panel">
          <h4 style="margin:0 0 8px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-size:.8rem">Applied Interventions</h4>
          <div id="appliedInterventions" class="interventions-list"></div>
        </div>
      </div>
    </aside>

    <main class="main">
      <section class="hero">
        <h2>Agent Image Review Timeline</h2>
        <div id="heroText" class="muted">Waiting for orchestrator state…</div>
        <div class="metrics">
          <div class="metric"><div class="k">Iterations</div><div class="v" id="metricIterations">0</div></div>
          <div class="metric"><div class="k">Batches</div><div class="v" id="metricBatches">0</div></div>
          <div class="metric"><div class="k">Recovered Images</div><div class="v" id="metricImages">0</div></div>
          <div class="metric"><div class="k">Current Phase</div><div class="v" id="metricPhase">idle</div></div>
        </div>
      </section>

      <section class="panel" style="padding:12px 14px">
        <h4 style="margin:0 0 8px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-size:.8rem">Available Series</h4>
        <div id="availableSeries"></div>
      </section>

      <div id="timeline"></div>

      <section class="panel" style="padding:12px 14px">
        <h4 style="margin:0 0 8px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-size:.8rem">Final Report</h4>
        <div id="finalReport" class="muted">Not available yet.</div>
      </section>
    </main>
  </div>

  <script>
    let latestState = null;
    let pollTimer = null;
    let lastRenderSignature = null;

    function esc(v) {{
      return String(v ?? "").replace(/[&<>"']/g, (ch) => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}})[ch]);
    }}

    function j(v) {{
      return JSON.stringify(v, null, 2);
    }}

    function stableCloneForRender(v) {{
      if (Array.isArray(v)) {{
        return v.map(stableCloneForRender);
      }}
      if (!v || typeof v !== "object") {{
        return v;
      }}
      const out = {{}};
      for (const [k, value] of Object.entries(v)) {{
        if (k === "updated_at" || k === "created_at" || k === "started_at") continue;
        out[k] = stableCloneForRender(value);
      }}
      return out;
    }}

    function stateRenderSignature(state) {{
      try {{
        return JSON.stringify(stableCloneForRender(state));
      }} catch (err) {{
        console.warn("signature failed", err);
        return null;
      }}
    }}

    function chip(text, cls="") {{
      return `<span class="chip ${cls}">${{esc(text)}}</span>`;
    }}

    async function postJson(url, payload) {{
      const res = await fetch(url, {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(payload),
      }});
      const data = await res.json().catch(() => ({{}}));
      if (!res.ok) throw new Error(data.error || `HTTP ${{res.status}}`);
      return data;
    }}

    function renderInterventions(state) {{
      const pending = state?.interventions?.pending || [];
      const history = state?.interventions?.history || [];
      const pendingEl = document.getElementById("pendingInterventions");
      const appliedEl = document.getElementById("appliedInterventions");

      pendingEl.innerHTML = pending.length
        ? pending.map(x => `<div class="intervention"><div class="chip-row">${{chip(x.kind)}}</div><div style="margin-top:6px">${{esc(x.text)}}</div><div class="small muted" style="margin-top:6px">${{esc(x.created_at || "")}}</div></div>`).join("")
        : `<div class="muted small">None</div>`;

      const histTail = history.slice(-20).reverse();
      appliedEl.innerHTML = histTail.length
        ? histTail.map(x => `<div class="intervention"><div class="chip-row">${{chip(x.kind)}} ${{chip(`iter=${{x.applied_iteration}}`, "ok")}}</div><div style="margin-top:6px">${{esc(x.text)}}</div><div class="small muted" style="margin-top:6px">${{esc(x.applied_at || x.created_at || "")}}</div></div>`).join("")
        : `<div class="muted small">None</div>`;
    }}

    function renderAvailableSeries(state) {{
      const items = state?.available_images || [];
      const el = document.getElementById("availableSeries");
      if (!items.length) {{
        el.innerHTML = `<div class="muted small">No series loaded yet.</div>`;
        return;
      }}
      const rows = items.map(s => `
        <tr>
          <td class="mono">${{esc(s.series_id || "")}}</td>
          <td>${{esc(s.modality || "")}}</td>
          <td>${{esc(s.description || "")}}</td>
          <td>${{esc(s.slice_range || "")}}</td>
        </tr>`).join("");
      el.innerHTML = `
        <div style="overflow:auto; border:1px solid var(--line); border-radius:10px; background:#fffdf7">
          <table style="width:100%; border-collapse:collapse; font-size:.82rem">
            <thead>
              <tr style="background:#f6f0e3; color:var(--muted)">
                <th style="padding:8px 10px; text-align:left">Series</th>
                <th style="padding:8px 10px; text-align:left">Modality</th>
                <th style="padding:8px 10px; text-align:left">Description</th>
                <th style="padding:8px 10px; text-align:left">Range</th>
              </tr>
            </thead>
            <tbody>${{rows}}</tbody>
          </table>
        </div>`;
    }}

    function renderJudgments(agentResponse) {{
      if (!agentResponse) return `<div class="muted small">No parsed response yet.</div>`;
      const findings = Array.isArray(agentResponse.findings) ? agentResponse.findings : [];
      const negatives = Array.isArray(agentResponse.negative_findings) ? agentResponse.negative_findings : [];
      const uncertainties = Array.isArray(agentResponse.uncertainty_flags) ? agentResponse.uncertainty_flags : [];
      const coverage = agentResponse.coverage_summary || {{}};

      const findingHtml = findings.length ? findings.map(f => {{
        const boxes = Array.isArray(f.bounding_boxes) ? f.bounding_boxes : [];
        const boxHtml = boxes.length ? `
          <div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:4px">
            ${{boxes.map(bb => `<span class="bb-badge ${{f.confidence || ""}}">${{esc(bb.label || f.finding_id || "box")}} [${{(bb.box_2d||[]).join(",")}}]</span>`).join("")}}
          </div>` : "";
        return `
          <div class="judgment-item">
            <div class="chip-row">
              ${{chip(`confidence=${{f.confidence || "?"}}`)}}
              ${{chip(`evidence=${{f.evidence_type || "?"}}`)}}
              ${{chip(`region=${{f.region || "?"}}`)}}
              ${{boxes.length ? chip(`${{boxes.length}} box(es)`, "ok") : ""}}
            </div>
            <div style="margin-top:6px">${{esc(f.observation || "")}}</div>
            ${{Array.isArray(f.slices_examined) && f.slices_examined.length ? `<div class="small muted" style="margin-top:6px">Slices: ${{esc(f.slices_examined.join(", "))}}</div>` : ""}}
            ${{boxHtml}}
          </div>`;
      }}).join("") : `<div class="muted small">No positive findings recorded.</div>`;

      const negativeHtml = negatives.length ? negatives.map(n => `
        <div class="judgment-item">
          <div class="chip-row">
            ${{chip(n.region || "region")}}
            ${{chip(n.looked_for || "looked_for")}}
          </div>
          <div style="margin-top:6px">${{esc(n.result || "")}}</div>
          ${{Array.isArray(n.slices_examined) && n.slices_examined.length ? `<div class="small muted" style="margin-top:6px">Slices: ${{esc(n.slices_examined.join(", "))}}</div>` : ""}}
        </div>`).join("") : `<div class="muted small">No negative findings recorded.</div>`;

      const uncertaintyHtml = uncertainties.length
        ? `<ul style="margin:0; padding-left:18px">${{uncertainties.map(u => `<li>${{esc(u)}}</li>`).join("")}}</ul>`
        : `<div class="muted small">None</div>`;

      const coverageHtml = `
        <div class="judgment-item">
          <div><strong>Regions Examined:</strong> ${{esc((coverage.regions_examined || []).join(", ") || "None")}}</div>
          <div style="margin-top:4px"><strong>Regions Remaining:</strong> ${{esc((coverage.regions_remaining || []).join(", ") || "None")}}</div>
          <div style="margin-top:4px"><strong>Window Levels:</strong> ${{esc((coverage.window_levels_applied || []).join(", ") || "None")}}</div>
        </div>`;

      return `
        <div class="judgments">
          <div class="block"><h4>Findings (${{findings.length}})</h4>${{findingHtml}}</div>
          <div class="block"><h4>Negative Findings (${{negatives.length}})</h4>${{negativeHtml}}</div>
          <div class="block"><h4>Uncertainty Flags (${{uncertainties.length}})</h4>${{uncertaintyHtml}}</div>
          <div class="block"><h4>Coverage</h4>${{coverageHtml}}</div>
          <details class="call">
            <summary><span class="call-name">Raw Parsed Response JSON</span></summary>
            <div style="padding:10px"><pre>${{esc(j(agentResponse))}}</pre></div>
          </details>
        </div>`;
    }}

    function renderCalls(callSummaries) {{
      if (!callSummaries || !callSummaries.length) {{
        return `<div class="muted small">No MCP calls executed.</div>`;
      }}
      return `<div class="calls">${{callSummaries.map(c => `
        <details class="call">
          <summary>
            <span class="call-name">${{esc(c.tool_name || "?")}}</span>
            ${{c.error ? chip("error", "error") : chip("ok", "ok")}}
            ${{chip(`images=${{c.extracted_image_count || 0}}`)}}
            <span class="muted small">${{esc(c.summary || "")}}</span>
          </summary>
          <div class="call-grid">
            <div>
              <div class="small muted" style="margin-bottom:4px">Arguments</div>
              <pre>${{esc(j(c.arguments || {{}}))}}</pre>
            </div>
            <div>
              <div class="small muted" style="margin-bottom:4px">Result Preview</div>
              <pre>${{esc(j(c.result_preview))}}</pre>
            </div>
          </div>
          ${{c.error ? `<div style="padding:0 10px 10px" class="err small">${{esc(c.error)}}</div>` : ""}}
        </details>`).join("")}}</div>`;
    }}

    function imageLabelParts(img, idx) {{
      const parts = [];
      if (img?.series_id) parts.push(String(img.series_id));
      if (img?.index !== null && img?.index !== undefined) {{
        parts.push(`idx ${{img.index}}`);
      }} else if (img?.instance_number !== null && img?.instance_number !== undefined) {{
        parts.push(`inst ${{img.instance_number}}`);
      }} else if (img?.slot !== null && img?.slot !== undefined) {{
        parts.push(`slot ${{img.slot}}`);
      }} else {{
        parts.push(`image ${{idx + 1}}`);
      }}
      return parts;
    }}

    function imageSubtitle(img) {{
      const parts = [];
      if (img?.tool_name) parts.push(String(img.tool_name));
      if (img?.call_index !== null && img?.call_index !== undefined) parts.push(`call ${{img.call_index}}`);
      if (img?.rows && img?.columns) parts.push(`${{img.columns}}x${{img.rows}}`);
      if (img?.source_meta_match === false) parts.push("viewed (metadata unmatched)");
      else if (img?.source_kind === "viewed") parts.push("viewed by agent");
      return parts.join(" • ");
    }}

    function imageCaptionChips(img) {{
      const chips = [];
      if (img?.slot !== null && img?.slot !== undefined) chips.push(`view ${{img.slot}}`);
      if (img?.instance_number !== null && img?.instance_number !== undefined) chips.push(`inst ${{img.instance_number}}`);
      if (img?.path || img?.file_name) chips.push("source");
      if (img?.window) chips.push("windowed");
      if (img?.crop) chips.push("cropped");
      if (img?.render?.output_size) chips.push("rendered");
      return chips;
    }}

    function imageHoverRows(img, idx) {{
      const rows = [];
      rows.push(`Series: ${{img?.series_id || "unknown"}}`);
      rows.push(`Label: ${{imageLabelParts(img, idx).join(" • ")}}`);
      if (img?.tool_name) rows.push(`Tool: ${{img.tool_name}}`);
      if (img?.call_index !== null && img?.call_index !== undefined) rows.push(`Call Index: ${{img.call_index}}`);
      if (img?.slot !== null && img?.slot !== undefined) rows.push(`Viewed Slot: ${{img.slot}}`);
      if (img?.index !== null && img?.index !== undefined) rows.push(`Slice Index: ${{img.index}}`);
      if (img?.instance_number !== null && img?.instance_number !== undefined) rows.push(`Instance Number: ${{img.instance_number}}`);
      if (img?.file_name) rows.push(`File: ${{img.file_name}}`);
      if (img?.path) rows.push(`Path: ${{img.path}}`);
      if (img?.rows && img?.columns) rows.push(`Native Size: ${{img.columns}}x${{img.rows}}`);
      if (img?.render?.output_size) rows.push(`Rendered Size: ${{img.render.output_size.width}}x${{img.render.output_size.height}}`);
      if (img?.requested_range) rows.push(`Requested Range: ${{JSON.stringify(img.requested_range)}}`);
      if (img?.requested_ranges) rows.push(`Requested Ranges: ${{JSON.stringify(img.requested_ranges)}}`);
      if (img?.resolved_indices) rows.push(`Resolved Indices: ${{JSON.stringify(img.resolved_indices)}}`);
      if (img?.window) rows.push(`Window: ${{JSON.stringify(img.window)}}`);
      if (img?.crop) rows.push(`Crop: ${{JSON.stringify(img.crop)}}`);
      if (img?.arguments) rows.push(`Args: ${{JSON.stringify(img.arguments)}}`);
      return rows;
    }}

    function renderImageGrid(images, annotatedImages, gridId) {{
      if (!images || !images.length) {{
        return `<div class="muted small">No images in this batch.</div>`;
      }}
      const hasAnnotated = annotatedImages && annotatedImages.length > 0;
      const toggleHtml = hasAnnotated ? `
        <div class="toggle-row">
          <button class="toggle-btn active" data-grid="${{gridId}}" data-mode="original" onclick="toggleImageMode(this)">Original</button>
          <button class="toggle-btn" data-grid="${{gridId}}" data-mode="annotated" onclick="toggleImageMode(this)">Annotated</button>
          <span class="muted small">${{annotatedImages.length}} annotated image(s) with bounding boxes</span>
        </div>` : "";

      // Build a map from slot index → annotated src
      const annotatedMap = {{}};
      if (hasAnnotated) {{
        for (const a of annotatedImages) {{
          annotatedMap[a.slot] = a.src;
        }}
      }}

      const thumbsHtml = images.map((img, idx) => {{
        const label = imageLabelParts(img, idx).join(" • ");
        const sub = imageSubtitle(img);
        const hoverRows = imageHoverRows(img, idx);
        const hoverText = hoverRows.join("\\n");
        const meta = [];
        if (img.path) meta.push(`path: ${{img.path}}`);
        if (img.columns && img.rows) meta.push(`size: ${{img.columns}}x${{img.rows}}`);
        if (img.render && img.render.output_size) meta.push(`rendered: ${{img.render.output_size.width}}x${{img.render.output_size.height}}`);
        const annSrc = annotatedMap[idx];
        const imgHtml = annSrc
          ? `<div class="img-stack" data-grid="${{gridId}}">
              <a href="${{esc(img.src)}}" target="_blank" rel="noopener noreferrer" title="${{esc(hoverText)}}"><img loading="lazy" src="${{esc(img.src)}}" alt="${{esc(label)}}" class="img-original" /></a>
              <a href="${{esc(annSrc)}}" target="_blank" rel="noopener noreferrer" title="${{esc(hoverText)}}"><img loading="lazy" src="${{esc(annSrc)}}" alt="${{esc(label)}} (annotated)" class="img-annotated hidden" /></a>
            </div>`
          : `<a href="${{esc(img.src)}}" target="_blank" rel="noopener noreferrer" title="${{esc(hoverText)}}"><img loading="lazy" src="${{esc(img.src)}}" alt="${{esc(label)}}" title="${{esc(hoverText)}}" /></a>`;
        const overlayHtml = `
          <div class="source-overlay" aria-hidden="true">
            ${{hoverRows.map((row) => {{
              const [k, ...rest] = String(row).split(": ");
              const v = rest.length ? rest.join(": ") : "";
              return `<div class="row"><span class="k">${{esc(k)}}${{v ? ":" : ""}}</span> ${{esc(v)}}</div>`;
            }}).join("")}}
          </div>`;
        const chipHtml = imageCaptionChips(img)
          .map(c => `<span class="mini-chip">${{esc(c)}}</span>`)
          .join("");
        return `
          <figure class="thumb">
            <div class="thumb-media">
              ${{imgHtml}}
              ${{overlayHtml}}
            </div>
            <figcaption class="cap">
              <div class="title">${{esc(label)}}</div>
              ${{sub ? `<div class="sub">${{esc(sub)}}</div>` : ""}}
              ${{chipHtml ? `<div class="cap-chips">${{chipHtml}}</div>` : ""}}
              ${{annSrc ? '<div class="muted" style="color:var(--accent)">has annotations</div>' : ""}}
              ${{meta.map(m => `<div class="muted">${{esc(m)}}</div>`).join("")}}
            </figcaption>
          </figure>`;
      }}).join("");

      return `${{toggleHtml}}<div class="thumb-grid">${{thumbsHtml}}</div>`;
    }}

    function toggleImageMode(btn) {{
      const gridId = btn.dataset.grid;
      const mode = btn.dataset.mode;
      // Update active button state
      const row = btn.parentElement;
      row.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      // Toggle images in this grid's thumbs
      document.querySelectorAll(`.img-stack[data-grid="${{gridId}}"]`).forEach(stack => {{
        const orig = stack.querySelector(".img-original");
        const ann = stack.querySelector(".img-annotated");
        if (!orig || !ann) return;
        if (mode === "annotated") {{
          orig.parentElement.classList.add("hidden");
          orig.classList.add("hidden");
          ann.parentElement.classList.remove("hidden");
          ann.classList.remove("hidden");
        }} else {{
          orig.parentElement.classList.remove("hidden");
          orig.classList.remove("hidden");
          ann.parentElement.classList.add("hidden");
          ann.classList.add("hidden");
        }}
      }});
    }}

    function buildBatches(iterationsObj) {{
      const entries = Object.values(iterationsObj || {{}}).sort((a,b) => (a.iteration ?? 0) - (b.iteration ?? 0));
      const byIteration = new Map(entries.map(e => [e.iteration, e]));
      const batches = [];
      for (const e of entries) {{
        if (!e.request_batch) continue;
        const review = byIteration.get((e.iteration ?? 0) + 1) || null;
        const reviewImageCount = review?.messages?.image_count ?? null;
        const reviewViewedImages = Array.isArray(review?.messages?.viewed_images) ? review.messages.viewed_images : [];
        const extracted = e.request_batch.images || [];
        const displayImages = reviewViewedImages.length
          ? reviewViewedImages.map((v, idx) => {{
              const meta = extracted[idx] || {{}};
              return {{
                ...meta,
                ...v,
                source_kind: "viewed",
                source_meta_match: !!extracted[idx],
              }};
            }})
          : extracted.map((img) => ({{ ...img, source_kind: "extracted", source_meta_match: true }}));
        // Annotated images from the review iteration (boxes drawn after agent response)
        const reviewAnnotated = Array.isArray(review?.annotated_images) ? review.annotated_images : [];
        const reviewBoxes = Array.isArray(review?.bounding_boxes) ? review.bounding_boxes : [];

        batches.push({{
          requestIteration: e.iteration,
          reviewIteration: review?.iteration ?? null,
          rationale: e.request_batch.rationale || "",
          suggestedThinkingLevel: e.request_batch.suggested_thinking_level || null,
          suggestedMediaResolution: e.request_batch.suggested_media_resolution || null,
          calls: e.request_batch.call_summaries || [],
          images: extracted,
          viewedImages: reviewViewedImages,
          displayImages,
          extractedCount: e.request_batch.extracted_image_count || extracted.length || 0,
          reviewImageCount,
          reviewAgentResponse: review?.agent_response || null,
          reviewParseError: review?.parse_error || null,
          deliveryMismatch: reviewImageCount !== null && reviewImageCount !== (e.request_batch.extracted_image_count || extracted.length || 0),
          appliedInterventions: e.applied_interventions || [],
          annotatedImages: reviewAnnotated,
          boundingBoxes: reviewBoxes,
        }});
      }}
      return batches;
    }}

    function renderTimeline(state) {{
      const timeline = document.getElementById("timeline");
      const iterations = state?.iterations || {{}};
      const entries = Object.values(iterations);
      const batches = buildBatches(iterations);

      document.getElementById("metricIterations").textContent = String(entries.length);
      document.getElementById("metricBatches").textContent = String(batches.length);
      document.getElementById("metricImages").textContent = String(batches.reduce((n, b) => n + (b.images?.length || 0), 0));
      document.getElementById("metricPhase").textContent = String(state?.session?.phase || "idle");

      if (!batches.length) {{
        timeline.innerHTML = `<section class="panel" style="padding:12px 14px"><div class="muted">No request/image batches yet. The agent may still be in initialization or the first LLM call.</div></section>`;
        return;
      }}

      timeline.innerHTML = batches.map((batch, batchIdx) => {{
        const headChips = [
          chip(`extracted=${{batch.extractedCount}}`),
          batch.reviewImageCount !== null ? chip(`viewed=${{batch.reviewImageCount}}`) : "",
          batch.suggestedThinkingLevel ? chip(`thinking=${{batch.suggestedThinkingLevel}}`) : "",
          batch.suggestedMediaResolution ? chip(`media=${{batch.suggestedMediaResolution}}`) : "",
          batch.deliveryMismatch ? chip("extracted vs viewed mismatch", "warn") : "",
          batch.reviewParseError ? chip("review parse error", "error") : "",
        ].filter(Boolean).join("");

        const interventionNote = batch.appliedInterventions?.length
          ? `<div class="block"><h4>Applied Supervisor Interventions (request iteration)</h4><div class="chip-row">${{batch.appliedInterventions.map(x => chip(`${{x.kind}} #${{x.id}}`, "ok")).join("")}}</div><div style="margin-top:8px" class="small">${{batch.appliedInterventions.map(x => `<div style="margin-bottom:6px">${{esc(x.text)}}</div>`).join("")}}</div></div>`
          : "";

        return `
          <section class="batch">
            <div class="batch-head">
              <h3>Request Iteration ${{batch.requestIteration}} → Review Iteration ${{batch.reviewIteration ?? "pending"}}</h3>
              <div class="chip-row">${{headChips}}</div>
            </div>
            <div class="batch-body">
              <div class="block">
                <h4>Why These Images Were Requested</h4>
                <p class="rationale">${{esc(batch.rationale || "No rationale captured.")}}</p>
              </div>
              ${{interventionNote}}
              <div class="block">
                <h4>MCP Calls / Results</h4>
                ${{renderCalls(batch.calls)}}
              </div>
              <div class="block">
                <h4>Images Viewed By Agent</h4>
                <div class="small muted" style="margin-bottom:8px">Uses actual images from the next LLM message payload when available, otherwise falls back to extracted images from the request batch.</div>
                ${{renderImageGrid(batch.displayImages, batch.annotatedImages, `grid-${{batchIdx}}`)}}
              </div>
              ${{batch.boundingBoxes.length ? `
              <div class="block">
                <h4>Bounding Boxes (${{batch.boundingBoxes.length}})</h4>
                <div style="display:flex; flex-wrap:wrap; gap:6px">
                  ${{batch.boundingBoxes.map(bb => `
                    <div class="bb-badge ${{bb.confidence || ""}}">
                      <strong>${{esc(bb.label || bb.finding_id || "?")}}</strong>
                      ${{esc(bb.region || "")}}
                      <span class="mono" style="font-size:.68rem">[y1=${{(bb.box_2d||[])[0]}}, x1=${{(bb.box_2d||[])[1]}}, y2=${{(bb.box_2d||[])[2]}}, x2=${{(bb.box_2d||[])[3]}}]</span>
                      ${{bb.slice_ref ? `<span class="muted" style="font-size:.68rem">${{esc(bb.slice_ref)}}</span>` : ""}}
                    </div>`).join("")}}
                </div>
              </div>` : ""}}
              <div class="block">
                <h4>Rendered Judgments on This Batch (from review iteration)</h4>
                ${{batch.reviewParseError ? `<div class="err small">Parse error: ${{esc(batch.reviewParseError)}}</div>` : ""}}
                ${{renderJudgments(batch.reviewAgentResponse)}}
              </div>
            </div>
          </section>`;
      }}).join("");
    }}

    function renderFinalReport(state) {{
      const el = document.getElementById("finalReport");
      if (!state?.final_report) {{
        el.innerHTML = `<div class="muted">Not available yet.</div>`;
        return;
      }}
      el.innerHTML = `<pre>${{esc(j(state.final_report))}}</pre>`;
    }}

    function renderHeader(state) {{
      const s = state?.session || {{}};
      const cfg = state?.config || {{}};
      document.getElementById("sessionMeta").innerHTML = `
        <div>Status: <strong>${{esc(s.status || "unknown")}}</strong></div>
        <div>Phase: <code>${{esc(s.phase || "idle")}}</code></div>
        <div>Iteration: <code>${{esc(s.current_iteration ?? "—")}}</code></div>
        <div>Paused: <code>${{esc(String(!!s.paused))}}</code></div>
        <div>Stop Requested: <code>${{esc(String(!!s.stop_requested))}}</code></div>`;

      const chips = [
        chip(`status=${{s.status || "unknown"}}`, s.status === "running" ? "ok" : ""),
        chip(`phase=${{s.phase || "idle"}}`),
        chip(`paused=${{!!s.paused}}`, s.paused ? "warn" : "ok"),
        chip(`stop=${{!!s.stop_requested}}`, s.stop_requested ? "warn" : "ok"),
        cfg.model ? chip(`model=${{cfg.model}}`) : "",
      ].filter(Boolean).join("");
      document.getElementById("statusChips").innerHTML = chips;

      const hero = cfg.clinical_question
        ? `${{esc(cfg.clinical_question)}}`
        : "Clinical question not populated yet.";
      document.getElementById("heroText").innerHTML = `
        <div>${{hero}}</div>
        <div class="small muted" style="margin-top:6px">Monitor URL: <code>${{esc(s.url || location.origin)}}</code></div>`;
    }}

    async function refreshState() {{
      try {{
        const res = await fetch("/api/state", {{cache: "no-store"}});
        const state = await res.json();
        latestState = state;
        const signature = stateRenderSignature(state);
        if (signature && signature === lastRenderSignature) {{
          return;
        }}
        lastRenderSignature = signature;
        renderHeader(state);
        renderInterventions(state);
        renderAvailableSeries(state);
        renderTimeline(state);
        renderFinalReport(state);
      }} catch (err) {{
        console.error(err);
        document.getElementById("heroText").innerHTML = `<span class="err">Failed to load state: ${{esc(err.message || err)}}</span>`;
      }}
    }}

    async function queueIntervention() {{
      const kind = document.getElementById("intKind").value;
      const text = document.getElementById("intText").value.trim();
      const msgEl = document.getElementById("intMsg");
      if (!text) {{
        msgEl.textContent = "Enter text before queuing an intervention.";
        return;
      }}
      msgEl.textContent = "Submitting…";
      try {{
        await postJson("/api/interventions", {{kind, text}});
        document.getElementById("intText").value = "";
        msgEl.textContent = "Queued for next iteration.";
        await refreshState();
      }} catch (err) {{
        msgEl.textContent = `Error: ${{err.message || err}}`;
      }}
    }}

    async function setPause(paused) {{
      const msgEl = document.getElementById("controlMsg");
      msgEl.textContent = paused ? "Pausing…" : "Resuming…";
      try {{
        await postJson("/api/control", {{action: paused ? "pause" : "resume"}});
        msgEl.textContent = paused ? "Paused (agent stops at next safe checkpoint)." : "Resumed.";
        await refreshState();
      }} catch (err) {{
        msgEl.textContent = `Error: ${{err.message || err}}`;
      }}
    }}

    function downloadJson(filename, obj) {{
      const blob = new Blob([JSON.stringify(obj, null, 2)], {{type: "application/json"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}

    function exportFinalReport() {{
      const msgEl = document.getElementById("controlMsg");
      const report = latestState?.final_report;
      if (!report) {{
        msgEl.textContent = "No final report available yet to export.";
        return;
      }}
      const runId = latestState?.session?.run_id || "monitor";
      const ts = new Date().toISOString().replace(/[:.]/g, "-");
      downloadJson(`${{runId}}_final_report_${{ts}}.json`, report);
      msgEl.textContent = "Final report JSON downloaded.";
    }}

    async function requestStopAndFinalize() {{
      const msgEl = document.getElementById("controlMsg");
      msgEl.textContent = "Requesting stop…";
      try {{
        await postJson("/api/control", {{action: "stop"}});
        msgEl.textContent = "Stop requested. Orchestrator will exit after the current safe checkpoint and return the latest final report.";
        await refreshState();
      }} catch (err) {{
        msgEl.textContent = `Error: ${{err.message || err}}`;
      }}
    }}

    function installHandlers() {{
      document.getElementById("sendInterventionBtn").addEventListener("click", queueIntervention);
      document.getElementById("pauseBtn").addEventListener("click", () => setPause(true));
      document.getElementById("resumeBtn").addEventListener("click", () => setPause(false));
      document.getElementById("stopBtn").addEventListener("click", requestStopAndFinalize);
      document.getElementById("exportFinalBtn").addEventListener("click", exportFinalReport);
    }}

    installHandlers();
    refreshState();
    pollTimer = setInterval(refreshState, 1500);
  </script>
</body>
</html>"""
        # The JS template was authored with doubled braces to avoid accidental
        # Python string formatting interpolation; normalize to real JS braces.
        return html.replace("{{", "{").replace("}}", "}")


__all__ = ["LiveOrchestratorMonitor", "NullOrchestratorMonitor"]
