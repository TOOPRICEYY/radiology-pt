from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image as PILImage
from PIL import ImageDraw


JsonDict = dict[str, Any]


@dataclass(slots=True)
class CropSpec:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_dict(cls, value: JsonDict | None) -> CropSpec | None:
        if value is None:
            return None
        required = ("x", "y", "width", "height")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"Crop spec missing keys: {', '.join(missing)}")
        crop = cls(
            x=int(value["x"]),
            y=int(value["y"]),
            width=int(value["width"]),
            height=int(value["height"]),
        )
        if crop.width <= 0 or crop.height <= 0:
            raise ValueError("Crop width/height must be > 0")
        return crop


class DicomRepository:
    """Scans a root directory and exposes DICOM indexing/extraction helpers."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._series_cache: list[JsonDict] | None = None
        self._series_entries_cache: dict[str, list[JsonDict]] = {}

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def set_root_dir(self, root_dir: str | Path) -> None:
        new_root = Path(root_dir).expanduser().resolve()
        self._root_dir = new_root
        self.refresh()

    def refresh(self) -> None:
        self._series_cache = None
        self._series_entries_cache.clear()

    def ensure_root_exists(self) -> None:
        if not self._root_dir.exists():
            raise FileNotFoundError(f"DICOM root does not exist: {self._root_dir}")
        if not self._root_dir.is_dir():
            raise NotADirectoryError(f"DICOM root is not a directory: {self._root_dir}")

    def list_series(self, refresh: bool = False) -> list[JsonDict]:
        if refresh:
            self.refresh()
        if self._series_cache is None:
            self._series_cache = self._scan_series()
        return [json.loads(json.dumps(item)) for item in self._series_cache]

    def get_series_details(self, series_id: str) -> JsonDict:
        summary = next((s for s in self.list_series() if s["series_id"] == series_id), None)
        if summary is None:
            raise KeyError(f"Unknown series_id: {series_id}")
        entries = self._get_series_entries(series_id)
        details = json.loads(json.dumps(summary))
        details["instances"] = [
            {
                "index": entry["index"],
                "file_name": entry["file_name"],
                "instance_number": entry.get("instance_number"),
                "sop_instance_uid": entry.get("sop_instance_uid"),
                "rows": entry.get("rows"),
                "columns": entry.get("columns"),
                "path": entry["relative_path"],
            }
            for entry in entries
        ]
        return details

    def extract_range(
        self,
        series_id: str,
        start: int,
        end: int,
        *,
        crop: JsonDict | None = None,
        annotations: list[JsonDict] | None = None,
        include_png_base64: bool = True,
        include_dicom_bytes_base64: bool = False,
        include_pixels: bool = False,
        normalize_mode: str = "percentile",
    ) -> JsonDict:
        return self.extract_ranges(
            series_id=series_id,
            ranges=[{"start": start, "end": end}],
            crop=crop,
            annotations=annotations,
            include_png_base64=include_png_base64,
            include_dicom_bytes_base64=include_dicom_bytes_base64,
            include_pixels=include_pixels,
            normalize_mode=normalize_mode,
        )

    def extract_ranges(
        self,
        *,
        series_id: str,
        ranges: list[JsonDict],
        crop: JsonDict | None = None,
        annotations: list[JsonDict] | None = None,
        include_png_base64: bool = True,
        include_dicom_bytes_base64: bool = False,
        include_pixels: bool = False,
        normalize_mode: str = "percentile",
    ) -> JsonDict:
        entries = self._get_series_entries(series_id)
        if not entries:
            raise ValueError(f"No DICOM files found in series '{series_id}'")

        crop_spec = CropSpec.from_dict(crop)
        selected_indices = _expand_ranges(ranges=ranges, total=len(entries))
        annotations = annotations or []

        extracted: list[JsonDict] = []
        for idx in selected_indices:
            entry = entries[idx]
            ds = pydicom.dcmread(entry["absolute_path"])
            item = self._build_extracted_item(
                ds=ds,
                entry=entry,
                crop_spec=crop_spec,
                annotations=annotations,
                include_png_base64=include_png_base64,
                include_dicom_bytes_base64=include_dicom_bytes_base64,
                include_pixels=include_pixels,
                normalize_mode=normalize_mode,
            )
            extracted.append(item)

        return {
            "series_id": series_id,
            "root_dir": str(self._root_dir),
            "total_instances_in_series": len(entries),
            "requested_ranges": [
                {"start": int(r["start"]), "end": int(r["end"])} for r in ranges
            ],
            "resolved_indices": selected_indices,
            "crop": _crop_to_dict(crop_spec),
            "annotation_count": len(annotations),
            "normalize_mode": normalize_mode,
            "extracted_count": len(extracted),
            "instances": extracted,
        }

    def preview_series_range_metadata(
        self,
        *,
        series_id: str,
        start: int,
        end: int,
    ) -> JsonDict:
        entries = self._get_series_entries(series_id)
        selected_indices = _expand_ranges(
            ranges=[{"start": start, "end": end}], total=len(entries)
        )
        return {
            "series_id": series_id,
            "root_dir": str(self._root_dir),
            "requested_range": {"start": start, "end": end},
            "resolved_indices": selected_indices,
            "instances": [
                {
                    "index": entries[i]["index"],
                    "file_name": entries[i]["file_name"],
                    "instance_number": entries[i].get("instance_number"),
                    "sop_instance_uid": entries[i].get("sop_instance_uid"),
                    "rows": entries[i].get("rows"),
                    "columns": entries[i].get("columns"),
                    "path": entries[i]["relative_path"],
                }
                for i in selected_indices
            ],
        }

    def _scan_series(self) -> list[JsonDict]:
        self.ensure_root_exists()

        grouped: dict[str, list[Path]] = {}
        for path in self._root_dir.rglob("*.dcm"):
            if not path.is_file():
                continue
            rel_parent = path.parent.relative_to(self._root_dir).as_posix()
            grouped.setdefault(rel_parent, []).append(path)

        series_summaries: list[JsonDict] = []
        for series_id in sorted(grouped):
            entries = self._build_series_entries(series_id, grouped[series_id])
            self._series_entries_cache[series_id] = entries
            if not entries:
                continue
            series_summaries.append(self._build_series_summary(series_id, entries))
        return series_summaries

    def _get_series_entries(self, series_id: str) -> list[JsonDict]:
        if self._series_cache is None:
            self.list_series()
        entries = self._series_entries_cache.get(series_id)
        if entries is None:
            raise KeyError(f"Unknown series_id: {series_id}")
        return entries

    def _build_series_entries(self, series_id: str, paths: list[Path]) -> list[JsonDict]:
        raw_entries: list[JsonDict] = []
        for path in sorted(paths):
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            raw_entries.append(
                {
                    "absolute_path": str(path),
                    "relative_path": path.relative_to(self._root_dir).as_posix(),
                    "file_name": path.name,
                    "instance_number": _safe_int(getattr(ds, "InstanceNumber", None)),
                    "sop_instance_uid": _safe_str(getattr(ds, "SOPInstanceUID", None)),
                    "rows": _safe_int(getattr(ds, "Rows", None)),
                    "columns": _safe_int(getattr(ds, "Columns", None)),
                    "_first_meta": {
                        "patient_id": _safe_str(getattr(ds, "PatientID", None)),
                        "patient_name": _safe_str(getattr(ds, "PatientName", None)),
                        "study_instance_uid": _safe_str(
                            getattr(ds, "StudyInstanceUID", None)
                        ),
                        "series_instance_uid": _safe_str(
                            getattr(ds, "SeriesInstanceUID", None)
                        ),
                        "series_description": _safe_str(
                            getattr(ds, "SeriesDescription", None)
                        ),
                        "modality": _safe_str(getattr(ds, "Modality", None)),
                        "study_date": _safe_str(getattr(ds, "StudyDate", None)),
                        "study_time": _safe_str(getattr(ds, "StudyTime", None)),
                    },
                }
            )

        raw_entries.sort(
            key=lambda item: (
                item["instance_number"] is None,
                item["instance_number"] if item["instance_number"] is not None else 0,
                item["file_name"],
            )
        )
        for idx, item in enumerate(raw_entries):
            item["index"] = idx
            item.pop("_first_meta", None)

        if raw_entries:
            # Re-read first sorted file for canonical series metadata after ordering.
            first_path = Path(raw_entries[0]["absolute_path"])
            first_ds = pydicom.dcmread(first_path, stop_before_pixels=True)
            first_meta = {
                "patient_id": _safe_str(getattr(first_ds, "PatientID", None)),
                "patient_name": _safe_str(getattr(first_ds, "PatientName", None)),
                "study_instance_uid": _safe_str(getattr(first_ds, "StudyInstanceUID", None)),
                "series_instance_uid": _safe_str(
                    getattr(first_ds, "SeriesInstanceUID", None)
                ),
                "series_description": _safe_str(
                    getattr(first_ds, "SeriesDescription", None)
                ),
                "modality": _safe_str(getattr(first_ds, "Modality", None)),
                "study_date": _safe_str(getattr(first_ds, "StudyDate", None)),
                "study_time": _safe_str(getattr(first_ds, "StudyTime", None)),
            }
            for item in raw_entries:
                item["_series_meta"] = first_meta

        return raw_entries

    def _build_series_summary(self, series_id: str, entries: list[JsonDict]) -> JsonDict:
        first = entries[0]
        series_meta = first.get("_series_meta", {})
        instance_numbers = [
            e["instance_number"] for e in entries if e.get("instance_number") is not None
        ]
        summary = {
            "series_id": series_id,
            "directory": str(self._root_dir / series_id),
            "instance_count": len(entries),
            "instance_number_min": min(instance_numbers) if instance_numbers else None,
            "instance_number_max": max(instance_numbers) if instance_numbers else None,
            "rows": first.get("rows"),
            "columns": first.get("columns"),
            "modality": series_meta.get("modality"),
            "series_description": series_meta.get("series_description"),
            "series_instance_uid": series_meta.get("series_instance_uid"),
            "study_instance_uid": series_meta.get("study_instance_uid"),
            "patient_id": series_meta.get("patient_id"),
            "patient_name": series_meta.get("patient_name"),
            "study_date": series_meta.get("study_date"),
            "study_time": series_meta.get("study_time"),
            "first_file": first.get("relative_path"),
            "last_file": entries[-1].get("relative_path"),
        }
        return _json_ready(summary)

    def _build_extracted_item(
        self,
        *,
        ds: pydicom.dataset.FileDataset,
        entry: JsonDict,
        crop_spec: CropSpec | None,
        annotations: list[JsonDict],
        include_png_base64: bool,
        include_dicom_bytes_base64: bool,
        include_pixels: bool,
        normalize_mode: str,
    ) -> JsonDict:
        result: JsonDict = {
            "index": entry["index"],
            "file_name": entry["file_name"],
            "path": entry["relative_path"],
            "instance_number": _safe_int(getattr(ds, "InstanceNumber", None)),
            "sop_instance_uid": _safe_str(getattr(ds, "SOPInstanceUID", None)),
            "rows": _safe_int(getattr(ds, "Rows", None)),
            "columns": _safe_int(getattr(ds, "Columns", None)),
            "photometric_interpretation": _safe_str(
                getattr(ds, "PhotometricInterpretation", None)
            ),
        }

        applied_annotations = _matching_annotations(annotations, result)

        render_meta = None
        if include_png_base64 or include_pixels or applied_annotations or crop_spec is not None:
            pixel_array = ds.pixel_array
            png_bytes, processed_pixels, render_meta = render_dicom_pixels(
                pixel_array=pixel_array,
                photometric_interpretation=result["photometric_interpretation"],
                crop_spec=crop_spec,
                annotations=applied_annotations,
                normalize_mode=normalize_mode,
            )
            result["render"] = render_meta
            if include_png_base64:
                result["png_base64"] = base64.b64encode(png_bytes).decode("ascii")
            if include_pixels:
                result["pixels"] = processed_pixels.tolist()

        if include_dicom_bytes_base64:
            with open(entry["absolute_path"], "rb") as fh:
                result["dicom_bytes_base64"] = base64.b64encode(fh.read()).decode("ascii")

        return _json_ready(result)


def render_dicom_pixels(
    *,
    pixel_array: np.ndarray,
    photometric_interpretation: str | None,
    crop_spec: CropSpec | None = None,
    annotations: list[JsonDict] | None = None,
    normalize_mode: str = "percentile",
) -> tuple[bytes, np.ndarray, JsonDict]:
    """Normalize -> optional crop -> annotate -> PNG encode."""

    annotations = annotations or []
    arr = np.asarray(pixel_array)
    if arr.ndim == 3:
        # Common DICOM variations: (frames, rows, cols) or RGB.
        # We default to the first frame for multi-frame grayscale.
        if arr.shape[0] in (1, 3) and arr.shape[-1] != 3:
            arr = arr[0]
        elif arr.shape[-1] == 3:
            # Already RGB-like; convert to uint8 if needed.
            arr = _normalize_rgb(arr)
        else:
            arr = arr[0]
    if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[-1] == 3):
        raise ValueError(f"Unsupported pixel array shape: {arr.shape}")

    if arr.ndim == 2:
        arr_u8 = _normalize_grayscale(arr, normalize_mode=normalize_mode)
        if (photometric_interpretation or "").upper() == "MONOCHROME1":
            arr_u8 = 255 - arr_u8
        image = PILImage.fromarray(arr_u8, mode="L").convert("RGB")
    else:
        image = PILImage.fromarray(arr.astype(np.uint8), mode="RGB")

    original_size = {"width": image.width, "height": image.height}
    crop_box = None
    if crop_spec is not None:
        crop_box = _clamp_crop(crop_spec, image.width, image.height)
        image = image.crop((crop_box["x"], crop_box["y"], crop_box["x2"], crop_box["y2"]))

    if annotations:
        _draw_annotations(
            image=image,
            annotations=annotations,
            crop_box=crop_box,
        )

    processed_pixels = np.asarray(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    render_meta: JsonDict = {
        "original_size": original_size,
        "output_size": {"width": image.width, "height": image.height},
        "crop": crop_box,
        "applied_annotations": len(annotations),
    }
    return png_bytes, processed_pixels, render_meta


def _expand_ranges(*, ranges: list[JsonDict], total: int) -> list[int]:
    if total <= 0:
        raise ValueError("Cannot expand ranges for empty series")
    if not ranges:
        raise ValueError("At least one range is required")

    indices: list[int] = []
    seen: set[int] = set()
    for idx, r in enumerate(ranges):
        if "start" not in r or "end" not in r:
            raise ValueError(f"Range #{idx} must include 'start' and 'end'")
        start = int(r["start"])
        end = int(r["end"])
        if start < 0 or end < 0:
            raise ValueError("Range indices must be >= 0")
        if start > end:
            raise ValueError(f"Range #{idx} start must be <= end")
        if end >= total:
            raise ValueError(
                f"Range #{idx} end={end} is out of bounds for total={total}"
            )
        for value in range(start, end + 1):
            if value not in seen:
                seen.add(value)
                indices.append(value)
    return indices


def _normalize_grayscale(arr: np.ndarray, *, normalize_mode: str = "percentile") -> np.ndarray:
    data = arr.astype(np.float32)
    if normalize_mode == "minmax":
        lo = float(np.min(data))
        hi = float(np.max(data))
    elif normalize_mode == "none":
        # Best-effort cast.
        lo = 0.0
        hi = 255.0 if data.dtype.kind in ("u", "i") else float(np.max(data))
    else:
        lo = float(np.percentile(data, 1.0))
        hi = float(np.percentile(data, 99.0))
        if hi <= lo:
            lo = float(np.min(data))
            hi = float(np.max(data))

    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    data = arr.astype(np.float32)
    if data.max() <= 255 and data.min() >= 0:
        return data.astype(np.uint8)
    lo = float(np.percentile(data, 1.0))
    hi = float(np.percentile(data, 99.0))
    if hi <= lo:
        lo = float(np.min(data))
        hi = float(np.max(data))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.uint8)
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _clamp_crop(crop: CropSpec, width: int, height: int) -> JsonDict:
    x = max(0, min(int(crop.x), width - 1))
    y = max(0, min(int(crop.y), height - 1))
    x2 = max(x + 1, min(x + int(crop.width), width))
    y2 = max(y + 1, min(y + int(crop.height), height))
    return {"x": x, "y": y, "x2": x2, "y2": y2, "width": x2 - x, "height": y2 - y}


def _draw_annotations(
    *,
    image: PILImage.Image,
    annotations: list[JsonDict],
    crop_box: JsonDict | None,
) -> None:
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        shape_type = str(annotation.get("type", "")).lower().strip()
        color = _parse_color(annotation.get("color", [255, 0, 0]))
        line_width = max(1, int(annotation.get("line_width", 2)))
        coords = _annotation_coords(annotation, crop_box=crop_box)

        if shape_type in {"bbox", "bounding_box", "rectangle", "rect"}:
            draw.rectangle(
                [coords["x1"], coords["y1"], coords["x2"], coords["y2"]],
                outline=color,
                width=line_width,
            )
        elif shape_type == "circle":
            draw.ellipse(
                [coords["x1"], coords["y1"], coords["x2"], coords["y2"]],
                outline=color,
                width=line_width,
            )
        elif shape_type == "ellipse":
            draw.ellipse(
                [coords["x1"], coords["y1"], coords["x2"], coords["y2"]],
                outline=color,
                width=line_width,
            )
        else:
            raise ValueError(
                "Unsupported annotation type. Use bbox/rectangle, circle, or ellipse."
            )


def _annotation_coords(annotation: JsonDict, *, crop_box: JsonDict | None) -> JsonDict:
    coord_space = str(annotation.get("coordinate_space", "source")).lower()
    x_offset = 0
    y_offset = 0
    if crop_box is not None and coord_space == "source":
        x_offset = crop_box["x"]
        y_offset = crop_box["y"]

    shape_type = str(annotation.get("type", "")).lower().strip()
    if shape_type in {"bbox", "bounding_box", "rectangle", "rect"}:
        if all(k in annotation for k in ("x1", "y1", "x2", "y2")):
            x1 = float(annotation["x1"]) - x_offset
            y1 = float(annotation["y1"]) - y_offset
            x2 = float(annotation["x2"]) - x_offset
            y2 = float(annotation["y2"]) - y_offset
        elif all(k in annotation for k in ("x", "y", "width", "height")):
            x1 = float(annotation["x"]) - x_offset
            y1 = float(annotation["y"]) - y_offset
            x2 = x1 + float(annotation["width"])
            y2 = y1 + float(annotation["height"])
        else:
            raise ValueError(
                "BBox annotation must include (x,y,width,height) or (x1,y1,x2,y2)"
            )
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    if shape_type == "circle":
        if not all(k in annotation for k in ("cx", "cy", "radius")):
            raise ValueError("Circle annotation must include cx, cy, radius")
        cx = float(annotation["cx"]) - x_offset
        cy = float(annotation["cy"]) - y_offset
        radius = float(annotation["radius"])
        return {
            "x1": cx - radius,
            "y1": cy - radius,
            "x2": cx + radius,
            "y2": cy + radius,
        }

    if shape_type == "ellipse":
        if not all(k in annotation for k in ("cx", "cy", "rx", "ry")):
            raise ValueError("Ellipse annotation must include cx, cy, rx, ry")
        cx = float(annotation["cx"]) - x_offset
        cy = float(annotation["cy"]) - y_offset
        rx = float(annotation["rx"])
        ry = float(annotation["ry"])
        return {"x1": cx - rx, "y1": cy - ry, "x2": cx + rx, "y2": cy + ry}

    return {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}


def _matching_annotations(annotations: list[JsonDict], instance_meta: JsonDict) -> list[JsonDict]:
    matches: list[JsonDict] = []
    for ann in annotations:
        target_index = ann.get("target_index")
        target_instance_number = ann.get("target_instance_number")

        if target_index is not None and int(target_index) != int(instance_meta["index"]):
            continue
        if (
            target_instance_number is not None
            and instance_meta.get("instance_number") is not None
            and int(target_instance_number) != int(instance_meta["instance_number"])
        ):
            continue
        if (
            target_instance_number is not None
            and instance_meta.get("instance_number") is None
        ):
            continue
        matches.append(ann)
    return matches


def _parse_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        simple = value.strip().lower()
        named = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        if simple in named:
            return named[simple]
        if simple.startswith("#") and len(simple) == 7:
            return tuple(int(simple[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return tuple(max(0, min(255, int(v))) for v in value[:3])  # type: ignore[return-value]
    return (255, 0, 0)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _crop_to_dict(crop: CropSpec | None) -> JsonDict | None:
    if crop is None:
        return None
    return {"x": crop.x, "y": crop.y, "width": crop.width, "height": crop.height}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
