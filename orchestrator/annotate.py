"""Render bounding-box annotations from findings onto extracted images."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from orchestrator.models import AgentResponse, Finding

# Colors per confidence level — high contrast on medical images
_COLORS = {
    "high": (255, 50, 50),       # red
    "moderate": (255, 200, 0),   # yellow
    "low": (0, 180, 255),        # cyan
}
_DEFAULT_COLOR = (0, 255, 0)     # green fallback

# Colors for evidence type badge
_EVIDENCE_COLORS = {
    "primary": (255, 255, 255),
    "confirmatory": (100, 255, 100),
    "contradictory": (255, 100, 100),
}


def annotate_images(
    images_b64: list[str],
    image_metadata: list[dict],
    response: "AgentResponse",
) -> list[str]:
    """Draw bounding boxes from findings onto the extracted images.

    Args:
        images_b64: base64-encoded PNGs (flat list across all MCP calls).
        image_metadata: per-MCP-call metadata dicts (from current_images_metadata).
            Each has tool_name, arguments with series_id/start/end/index, etc.
        response: the parsed agent response with findings that may have bounding_boxes.

    Returns:
        List of base64-encoded PNGs with annotations drawn. Same length as images_b64.
        Images without matching boxes are returned unchanged.
    """
    # Build a map from slice reference → list of (BoundingBox, Finding) pairs
    box_map: dict[str, list[tuple]] = {}
    for finding in response.findings:
        for bb in finding.bounding_boxes:
            ref = bb.slice_ref
            if ref:
                box_map.setdefault(ref, []).append((bb, finding))

    if not box_map:
        return images_b64

    # Build a map from image index → slice reference
    # Walk through metadata to assign refs to images in order
    image_refs = _build_image_refs(image_metadata)

    annotated = []
    for img_idx, b64 in enumerate(images_b64):
        ref = image_refs.get(img_idx, "")
        boxes_for_image = box_map.get(ref, [])

        if not boxes_for_image:
            # Also try matching without the preprocessing suffix
            # e.g., ref "SER00014:15:soft_tissue" matches "SER00014:15"
            base_ref = ":".join(ref.split(":")[:2]) if ":" in ref else ref
            boxes_for_image = box_map.get(base_ref, [])

        if not boxes_for_image:
            # Try matching any box whose ref shares the same series:slice prefix
            for box_ref, box_list in box_map.items():
                box_base = ":".join(box_ref.split(":")[:2]) if ":" in box_ref else box_ref
                if base_ref and box_base == base_ref:
                    boxes_for_image = box_list
                    break

        if not boxes_for_image:
            annotated.append(b64)
            continue

        annotated.append(_draw_boxes(b64, boxes_for_image))

    return annotated


def _build_image_refs(image_metadata: list[dict]) -> dict[int, str]:
    """Map flat image index → slice reference string.

    Walks through MCP call metadata and assigns refs like "SER00014:15:soft_tissue"
    to each image in order.
    """
    refs: dict[int, str] = {}
    global_idx = 0

    for meta in image_metadata:
        args = meta.get("arguments", {})
        series_id = args.get("series_id", "")
        n_images = meta.get("num_images", 0)
        window = args.get("window", "")
        if isinstance(window, dict):
            window = f"w{window.get('center', '')}-{window.get('width', '')}"
        window_str = str(window) if window else ""

        tool = meta.get("tool_name", "")
        start = args.get("start")
        end = args.get("end")
        step = int(args.get("step", 1))
        index = args.get("index")

        if start is not None and end is not None:
            # extract_dicom_range: each image is a slice in [start, end] stepping by step
            slice_indices = list(range(int(start), int(end) + 1, step))
            for i, slice_idx in enumerate(slice_indices):
                if global_idx + i >= global_idx + n_images:
                    break
                parts = [series_id, str(slice_idx)]
                if window_str:
                    parts.append(window_str)
                refs[global_idx + i] = ":".join(parts)
        elif index is not None:
            # annotate_dicom_slice or single index
            parts = [series_id, str(index)]
            if window_str:
                parts.append(window_str)
            refs[global_idx] = ":".join(parts)
        else:
            # Unknown tool shape — assign generic refs
            for i in range(n_images):
                refs[global_idx + i] = f"{series_id}:{global_idx + i}"

        global_idx += n_images

    return refs


def _draw_boxes(b64: str, boxes: list[tuple]) -> str:
    """Draw bounding boxes on a base64-encoded PNG. Returns annotated base64 PNG."""
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try to get a small font for labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(10, height // 30))
    except (OSError, AttributeError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=max(10, height // 30))
        except (OSError, AttributeError):
            font = ImageFont.load_default()

    for bb, finding in boxes:
        x1, y1, x2, y2 = bb.to_pixel_coords(width, height)
        color = _COLORS.get(finding.confidence, _DEFAULT_COLOR)

        # Draw rectangle (3px border for visibility)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Build label
        label_parts = []
        if bb.label:
            label_parts.append(bb.label)
        else:
            label_parts.append(finding.finding_id)
        label_parts.append(finding.confidence[0].upper())  # H/M/L
        label = " ".join(label_parts)

        # Draw label background + text
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Position label above the box, or inside if near top edge
        label_y = y1 - text_h - 4
        if label_y < 0:
            label_y = y1 + 2

        draw.rectangle(
            [x1, label_y, x1 + text_w + 6, label_y + text_h + 4],
            fill=color,
        )
        # Text in black or white depending on brightness
        brightness = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        draw.text((x1 + 3, label_y + 2), label, fill=text_color, font=font)

    # Encode back to base64 PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
