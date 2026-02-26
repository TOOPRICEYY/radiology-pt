"""Bridge between agent MCP call specs and DicomRepository methods."""

from __future__ import annotations

import ast
import json
import re
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dicom_mcp.dicom_ops import DicomRepository


# ── Tool descriptions for prompt injection ───────────────────────────────────

_TOOL_DESCRIPTIONS = """\
Available MCP tools. In your mcp_calls array, each entry MUST be a JSON object with "tool_name" and "arguments" keys.

Example mcp_calls format:
  "mcp_calls": [
    {"tool_name": "list_dicom_series", "arguments": {}},
    {"tool_name": "extract_dicom_range", "arguments": {"series_id": "SER00001", "start": 0, "end": 10, "step": 2, "window": "soft_tissue"}}
  ]

Tools:

1. list_dicom_series(refresh: bool = false)
   List all available DICOM series with metadata (series_id, modality, description, num_instances).

2. get_dicom_series_details(series_id: str)
   Get detailed metadata for a specific series including per-instance info.

3. extract_dicom_range(series_id: str, start: int, end: int, step: int = 1, crop: dict = null, annotations: list = null, include_png_base64: bool = true, normalize_mode: str = "percentile", window: str|dict = null)
   Extract a range of slices [start, end] inclusive. Returns metadata + base64 PNG per slice.
   - crop: {"x": int, "y": int, "width": int, "height": int}
   - window: preset name ("bone", "soft_tissue", "lung", "brain", "mediastinum", "liver", "stroke", "subdural") or {"center": float, "width": float}
   - annotations: [{"type": "bbox"|"circle"|"ellipse", ...coords, "color": str}]
   - step: extract every Nth slice (default 1)

4. extract_dicom_ranges(series_id: str, ranges: list[dict], crop: dict = null, annotations: list = null, include_png_base64: bool = true, normalize_mode: str = "percentile", window: str|dict = null)
   Extract multiple non-contiguous ranges in one call. Each range: {"start": int, "end": int, "step": int}.

5. annotate_dicom_slice(series_id: str, index: int, shapes: list[dict], crop: dict = null, include_png_base64: bool = true, normalize_mode: str = "percentile", window: str|dict = null)
   Extract a single slice with shape overlays. shapes: [{"type": "bbox"|"circle"|"ellipse", ...coords, "color": str}]

6. get_pixel_stats(series_id: str, index: int, roi: dict = null)
   Get HU statistics (mean, std, min, max, median) for a slice or ROI within a slice.
   - roi: {"x": int, "y": int, "width": int, "height": int}
"""


def build_tool_descriptions() -> str:
    """Return tool descriptions for injection into the prompt."""
    return _TOOL_DESCRIPTIONS


# ── MCP call dispatch ────────────────────────────────────────────────────────

def _normalize_window(window) -> str | dict | None:
    """Normalize window parameter to what DicomRepository expects."""
    if window is None:
        return None
    if isinstance(window, str):
        return window
    if isinstance(window, dict):
        return window
    return None


def _parse_call_spec(raw) -> dict:
    """Normalize an MCP call spec into {"tool_name": str, "arguments": dict}.

    Accepts:
      - Already-structured dict: {"tool_name": "...", "arguments": {...}}
      - Function-call string:    'get_dicom_series_details(series_id="SER00014")'
      - Plain tool name string:  'list_dicom_series'  or  'list_dicom_series()'
    """
    if isinstance(raw, dict):
        return {"tool_name": raw.get("tool_name", ""), "arguments": raw.get("arguments", {})}

    if not isinstance(raw, str):
        return {"tool_name": "", "arguments": {}, "_raw": str(raw)}

    raw = raw.strip()

    # Match: tool_name(  ...args...  )
    m = re.match(r"^(\w+)\s*\((.*)\)\s*$", raw, re.DOTALL)
    if not m:
        # Bare name like "list_dicom_series"
        return {"tool_name": raw, "arguments": {}}

    tool_name = m.group(1)
    args_str = m.group(2).strip()

    if not args_str:
        return {"tool_name": tool_name, "arguments": {}}

    # Parse keyword arguments.  The model tends to produce Python-style kwargs:
    #   series_id="SER00014", start=0, end=10
    # Strategy: wrap in dict() and eval safely via ast.literal_eval on a dict literal.
    try:
        # Try JSON first: {"series_id": "SER00014", "start": 0}
        arguments = json.loads("{" + args_str + "}")
        return {"tool_name": tool_name, "arguments": arguments}
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        # Try Python dict literal: dict(series_id="SER00014", start=0) → build dict literal
        # Convert  key=value, key=value  →  {"key": value, "key": value}
        dict_str = "dict(" + args_str + ")"
        # ast.literal_eval can't handle dict(), so parse manually
        arguments = {}
        # Split on commas that aren't inside brackets/braces/parens
        parts = _split_kwargs(args_str)
        for part in parts:
            part = part.strip()
            if "=" not in part:
                continue
            key, _, val = part.partition("=")
            key = key.strip()
            val = val.strip()
            try:
                arguments[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # Last resort: keep as string
                arguments[key] = val
        return {"tool_name": tool_name, "arguments": arguments}
    except Exception:
        return {"tool_name": tool_name, "arguments": {}, "_raw": raw}


def _split_kwargs(s: str) -> list[str]:
    """Split a kwargs string on top-level commas, respecting brackets/parens/braces."""
    parts = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in ("(", "[", "{"):
            depth += 1
        elif ch in (")", "]", "}"):
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def execute_mcp_calls(repo: "DicomRepository", mcp_calls: list) -> list[dict]:
    """Execute a list of MCP call specs and return results.

    Each call spec can be:
      - A dict: {"tool_name": "...", "arguments": {...}}
      - A string: 'tool_name(arg1="val", arg2=123)'

    Returns a list of result dicts, each with:
      - tool_name, arguments (echo back)
      - result: the method return value
      - error: str if the call failed
      - images: list of base64 PNGs extracted from the result
    """
    results = []
    for raw_spec in mcp_calls:
        call_spec = _parse_call_spec(raw_spec)
        tool_name = call_spec.get("tool_name", "")
        arguments = call_spec.get("arguments", {})
        result_entry = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": None,
            "error": None,
            "images": [],
        }

        try:
            data = _dispatch(repo, tool_name, arguments)
            result_entry["result"] = _strip_images(data)
            result_entry["images"] = _collect_images(data)
        except Exception as e:
            result_entry["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        results.append(result_entry)

    return results


def _dispatch(repo: "DicomRepository", tool_name: str, args: dict):
    """Route a tool call to the appropriate DicomRepository method."""
    if tool_name == "list_dicom_series":
        return repo.list_series(refresh=args.get("refresh", False))

    if tool_name == "get_dicom_series_details":
        return repo.get_series_details(args["series_id"])

    if tool_name == "extract_dicom_range":
        return repo.extract_range(
            series_id=args["series_id"],
            start=int(args["start"]),
            end=int(args["end"]),
            step=int(args.get("step", 1)),
            crop=args.get("crop"),
            annotations=args.get("annotations"),
            include_png_base64=args.get("include_png_base64", True),
            normalize_mode=args.get("normalize_mode", "percentile"),
            window=_normalize_window(args.get("window")),
        )

    if tool_name == "extract_dicom_ranges":
        return repo.extract_ranges(
            series_id=args["series_id"],
            ranges=args["ranges"],
            crop=args.get("crop"),
            annotations=args.get("annotations"),
            include_png_base64=args.get("include_png_base64", True),
            normalize_mode=args.get("normalize_mode", "percentile"),
            window=_normalize_window(args.get("window")),
        )

    if tool_name == "annotate_dicom_slice":
        # annotate_dicom_slice maps to extract_range with a single index + annotations
        shapes = args.get("shapes", [])
        return repo.extract_range(
            series_id=args["series_id"],
            start=int(args["index"]),
            end=int(args["index"]),
            crop=args.get("crop"),
            annotations=shapes,
            include_png_base64=args.get("include_png_base64", True),
            normalize_mode=args.get("normalize_mode", "percentile"),
            window=_normalize_window(args.get("window")),
        )

    if tool_name == "get_pixel_stats":
        return repo.get_pixel_stats(
            series_id=args["series_id"],
            index=int(args["index"]),
            roi=args.get("roi"),
        )

    raise ValueError(f"Unknown tool: {tool_name}")


def _collect_images(data) -> list[str]:
    """Extract base64 PNG strings from MCP call results."""
    images: list[str] = []

    def walk(value):
        if isinstance(value, dict):
            png_b64 = value.get("png_base64")
            if isinstance(png_b64, str) and png_b64:
                images.append(png_b64)
            # Support both legacy "items" and current "instances" result shapes.
            for key in ("items", "instances"):
                nested = value.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return images


def _strip_images(data):
    """Return a copy of data with png_base64 fields replaced by presence flags."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if k == "png_base64" and v:
                out[k] = "<sent as image>"
            elif k in {"items", "instances"} and isinstance(v, list):
                out[k] = [_strip_images(item) for item in v]
            else:
                out[k] = v
        return out
    if isinstance(data, list):
        return [_strip_images(item) for item in data]
    return data
