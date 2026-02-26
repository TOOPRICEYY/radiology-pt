from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dicom_mcp.dicom_ops import DicomRepository

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - only hit when SDK is missing
    if __name__ == "__main__":
        raise SystemExit(
            "The MCP Python SDK is not installed. Install with: pip install 'mcp[cli]'"
        ) from exc
    raise


JsonDict = dict[str, Any]

DEFAULT_DICOM_ROOT = (Path(__file__).resolve().parents[1] / "dicom").resolve()


@dataclass
class AppState:
    root_dir: Path
    repository: DicomRepository


def _initial_root() -> Path:
    env_root = os.environ.get("DICOM_ROOT_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DEFAULT_DICOM_ROOT


_STATE = AppState(
    root_dir=_initial_root(),
    repository=DicomRepository(_initial_root()),
)

mcp = FastMCP("dicom-mcp")


def _repo() -> DicomRepository:
    return _STATE.repository


def _set_root(root_dir: str) -> JsonDict:
    path = Path(root_dir).expanduser().resolve()
    _STATE.root_dir = path
    _STATE.repository.set_root_dir(path)
    _STATE.repository.ensure_root_exists()
    return {"root_dir": str(_STATE.root_dir)}


def _json_text(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False)


def _parse_ranges_spec(ranges_spec: str) -> list[JsonDict]:
    """Parse comma-separated inclusive ranges: '0-4,10-12'."""

    if not ranges_spec.strip():
        raise ValueError("ranges_spec cannot be empty")
    parsed: list[JsonDict] = []
    for part in ranges_spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" not in token:
            value = int(token)
            parsed.append({"start": value, "end": value})
            continue
        left, right = token.split("-", 1)
        parsed.append({"start": int(left.strip()), "end": int(right.strip())})
    if not parsed:
        raise ValueError("No valid ranges found in ranges_spec")
    return parsed


@mcp.resource("dicom://config")
def dicom_config() -> str:
    """Server configuration including the current DICOM root directory."""

    repo = _repo()
    exists = repo.root_dir.exists()
    return _json_text(
        {
            "server": "dicom-mcp",
            "root_dir": str(repo.root_dir),
            "root_exists": exists,
            "usage": {
                "resources": [
                    "dicom://config",
                    "dicom://series",
                    "dicom://series/{series_id}",
                    "dicom://series/{series_id}/range/{start}/{end}",
                    "dicom://series/{series_id}/ranges/{ranges_spec}",
                    "dicom://series/{series_id}/range/{start}/{end}/crop/{x}/{y}/{width}/{height}",
                ],
                "tools": [
                    "set_dicom_root",
                    "list_dicom_series",
                    "get_dicom_series_details",
                    "extract_dicom_range",
                    "extract_dicom_ranges",
                    "annotate_dicom_slice",
                    "get_pixel_stats",
                ],
            },
        }
    )


@mcp.resource("dicom://series")
def list_series_resource() -> str:
    """List available DICOM series discovered under the configured root."""

    return _json_text(_repo().list_series())


@mcp.resource("dicom://series/{series_id}")
def series_details_resource(series_id: str) -> str:
    """Return per-series details and instance metadata."""

    return _json_text(_repo().get_series_details(series_id))


@mcp.resource("dicom://series/{series_id}/range/{start}/{end}")
def range_metadata_resource(series_id: str, start: int, end: int) -> str:
    """Metadata-only preview for a contiguous DICOM range (inclusive indices)."""

    return _json_text(
        _repo().preview_series_range_metadata(series_id=series_id, start=start, end=end)
    )


@mcp.resource("dicom://series/{series_id}/ranges/{ranges_spec}")
def multi_range_metadata_resource(series_id: str, ranges_spec: str) -> str:
    """Metadata-only preview for multiple ranges (e.g. ranges/0-4,10-12)."""

    return _json_text(
        _repo().extract_ranges(
            series_id=series_id,
            ranges=_parse_ranges_spec(ranges_spec),
            include_png_base64=False,
            include_dicom_bytes_base64=False,
            include_pixels=False,
        )
    )


@mcp.resource("dicom://series/{series_id}/range/{start}/{end}/crop/{x}/{y}/{width}/{height}")
def cropped_range_metadata_resource(
    series_id: str,
    start: int,
    end: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    """Metadata preview for a cropped extraction request (no image payloads)."""

    return _json_text(
        _repo().extract_range(
            series_id=series_id,
            start=start,
            end=end,
            crop={"x": x, "y": y, "width": width, "height": height},
            include_png_base64=False,
            include_dicom_bytes_base64=False,
            include_pixels=False,
        )
    )


@mcp.tool()
def set_dicom_root(root_dir: str) -> JsonDict:
    """Set the DICOM root directory used by resources and tools."""

    return _set_root(root_dir)


@mcp.tool()
def get_dicom_root() -> JsonDict:
    """Get the current DICOM root directory."""

    repo = _repo()
    return {
        "root_dir": str(repo.root_dir),
        "exists": repo.root_dir.exists(),
        "is_dir": repo.root_dir.is_dir(),
    }


@mcp.tool()
def refresh_dicom_index() -> JsonDict:
    """Clear cached series metadata and rebuild the index."""

    repo = _repo()
    repo.refresh()
    series = repo.list_series()
    return {"root_dir": str(repo.root_dir), "series_count": len(series)}


@mcp.tool()
def list_dicom_series(refresh: bool = False) -> list[JsonDict]:
    """List available DICOM series under the configured root directory."""

    return _repo().list_series(refresh=refresh)


@mcp.tool()
def get_dicom_series_details(series_id: str) -> JsonDict:
    """Get detailed metadata for one DICOM series including per-instance entries."""

    return _repo().get_series_details(series_id=series_id)


@mcp.tool()
def extract_dicom_range(
    series_id: str,
    start: int,
    end: int,
    crop: JsonDict | None = None,
    annotations: list[JsonDict] | None = None,
    include_png_base64: bool = True,
    include_dicom_bytes_base64: bool = False,
    include_pixels: bool = False,
    normalize_mode: str = "percentile",
    window: str | dict | None = None,
    step: int = 1,
) -> JsonDict:
    """Extract a contiguous inclusive index range from a series with optional crop/annotations.

    Indexing is zero-based and inclusive (`start` and `end` are positions inside the sorted
    series order).

    `window` applies radiological windowing (HU-based). Pass a preset name string
    ("bone", "soft_tissue", "lung", "brain", "mediastinum", "liver", "stroke", "subdural")
    or a dict {"center": float, "width": float}. When set, pixel values are converted to
    Hounsfield Units using RescaleSlope/Intercept then mapped through the window.

    `step` controls stride for survey extraction. step=1 returns every slice (default),
    step=10 returns every 10th slice within the range.

    `crop` schema:
      {"x": 10, "y": 20, "width": 200, "height": 150}

    `annotations` schema (applied only to a specific DICOM via `target_index` or
    `target_instance_number`):
      bbox:    {"type":"bbox","target_index":5,"x":20,"y":30,"width":60,"height":80}
      circle:  {"type":"circle","target_index":5,"cx":120,"cy":140,"radius":30}
      ellipse: {"type":"ellipse","target_index":5,"cx":120,"cy":140,"rx":40,"ry":20}

    Annotation coordinates are in source-image coordinates by default. Set
    `coordinate_space` to `"cropped"` to use post-crop coordinates.
    """

    return _repo().extract_range(
        series_id=series_id,
        start=start,
        end=end,
        crop=crop,
        annotations=annotations,
        include_png_base64=include_png_base64,
        include_dicom_bytes_base64=include_dicom_bytes_base64,
        include_pixels=include_pixels,
        normalize_mode=normalize_mode,
        window=window,
        step=step,
    )


@mcp.tool()
def extract_dicom_ranges(
    series_id: str,
    ranges: list[JsonDict],
    crop: JsonDict | None = None,
    annotations: list[JsonDict] | None = None,
    include_png_base64: bool = True,
    include_dicom_bytes_base64: bool = False,
    include_pixels: bool = False,
    normalize_mode: str = "percentile",
    window: str | dict | None = None,
    step: int = 1,
) -> JsonDict:
    """Extract multiple inclusive ranges from a series with optional crop and annotations.

    `ranges` schema:
      [{"start": 0, "end": 4}, {"start": 10, "end": 12}]

    `window` applies radiological windowing (HU-based). Pass a preset name string
    ("bone", "soft_tissue", "lung", "brain", "mediastinum", "liver", "stroke", "subdural")
    or a dict {"center": float, "width": float}.

    `step` controls stride for survey extraction. step=1 returns every slice (default),
    step=10 returns every 10th slice within each range.

    Overlapping indices are de-duplicated in the order encountered.
    `annotations` uses the same schema as `extract_dicom_range`.
    """

    return _repo().extract_ranges(
        series_id=series_id,
        ranges=ranges,
        crop=crop,
        annotations=annotations,
        include_png_base64=include_png_base64,
        include_dicom_bytes_base64=include_dicom_bytes_base64,
        include_pixels=include_pixels,
        normalize_mode=normalize_mode,
        window=window,
        step=step,
    )


@mcp.tool()
def annotate_dicom_slice(
    series_id: str,
    index: int,
    shapes: list[JsonDict],
    crop: JsonDict | None = None,
    include_png_base64: bool = True,
    include_dicom_bytes_base64: bool = False,
    include_pixels: bool = False,
    normalize_mode: str = "percentile",
    window: str | dict | None = None,
) -> JsonDict:
    """Extract a single DICOM slice with bbox/circle/ellipse overlays.

    `index` is the zero-based series index. `shapes` uses the same annotation schema as
    `extract_dicom_range`, but `target_index` is optional and will default to `index`.

    `window` applies radiological windowing (HU-based). Pass a preset name string
    ("bone", "soft_tissue", "lung", "brain", "mediastinum", "liver", "stroke", "subdural")
    or a dict {"center": float, "width": float}.
    """

    normalized_shapes: list[JsonDict] = []
    for shape in shapes:
        item = dict(shape)
        item.setdefault("target_index", index)
        normalized_shapes.append(item)

    return _repo().extract_range(
        series_id=series_id,
        start=index,
        end=index,
        crop=crop,
        annotations=normalized_shapes,
        include_png_base64=include_png_base64,
        include_dicom_bytes_base64=include_dicom_bytes_base64,
        include_pixels=include_pixels,
        normalize_mode=normalize_mode,
        window=window,
    )


@mcp.tool()
def get_pixel_stats(
    series_id: str,
    index: int,
    roi: JsonDict | None = None,
) -> JsonDict:
    """Compute HU pixel statistics for a DICOM slice, optionally within an ROI.

    Returns min/max/mean/median/std and percentiles (p5, p25, p50, p75, p95) in
    Hounsfield Units. Use this to characterize tissue density:
    - Air: ~-1000 HU
    - Lung parenchyma: ~-700 to -600 HU
    - Fat: ~-100 to -50 HU
    - Water/fluid: ~0 HU
    - Soft tissue: ~40-80 HU
    - Calcification: >100 HU
    - Bone: ~400-1000 HU

    `roi` schema (optional): {"x": int, "y": int, "width": int, "height": int}
    """

    return _repo().get_pixel_stats(series_id=series_id, index=index, roi=roi)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DICOM MCP server")
    parser.add_argument(
        "--root",
        dest="root_dir",
        default=os.environ.get("DICOM_ROOT_DIR", str(DEFAULT_DICOM_ROOT)),
        help="Root directory containing DICOM files/series",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _set_root(args.root_dir)
    mcp.run()


if __name__ == "__main__":
    main()
