"""Build injectable context blocks for each iteration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dicom_mcp.dicom_ops import DicomRepository
    from orchestrator.models import IterationState, LedgerEntry, NegativeLedgerEntry


def build_available_images(repo: "DicomRepository") -> list[dict]:
    """Query the repository for all available series."""
    series_list = repo.list_series()
    result = []
    for s in series_list:
        result.append({
            "series_id": s.get("series_id", ""),
            "modality": s.get("modality", ""),
            "body_region": s.get("body_part_examined", ""),
            "slice_range": f"0-{s.get('num_instances', 1) - 1}",
            "description": s.get("series_description", s.get("description", "")),
        })
    return result


def build_observation_ledger(
    ledger: list["LedgerEntry"],
    negative_ledger: list["NegativeLedgerEntry"],
) -> list[dict]:
    """Serialize the cumulative observation ledger."""
    entries = [e.to_dict() for e in ledger]
    neg_entries = [e.to_dict() for e in negative_ledger]
    return {"findings": entries, "negative_findings": neg_entries}


def build_current_images(images: list[dict]) -> list[dict]:
    """Format the MCP call results from the previous iteration.

    Each entry should have metadata (series_id, slices, window, etc.)
    and a rationale. The actual base64 image data is sent separately
    as vision content blocks, not embedded here.
    """
    # Strip base64 data from the metadata representation
    sanitized = []
    for img in images:
        entry = {k: v for k, v in img.items() if k != "png_base64"}
        sanitized.append(entry)
    return sanitized


def build_iteration_state(state: "IterationState") -> dict:
    return state.to_dict()


def render_prompt(
    template: str,
    available_images: list[dict],
    ledger: dict,
    current_images: list[dict],
    iteration_state: dict,
    mcp_tool_descriptions: str,
    user_prompt: str,
) -> str:
    """Fill all placeholders in the prompt template."""
    rendered = template

    # {MCP_TOOL_USE} — no double braces
    rendered = rendered.replace("{MCP_TOOL_USE}", mcp_tool_descriptions)

    # {USER_PROMPT} — no double braces
    rendered = rendered.replace("{USER_PROMPT}", user_prompt)

    # {{PLACEHOLDER}} — double-brace context blocks
    rendered = rendered.replace(
        "{{AVAILABLE_IMAGES}}", json.dumps(available_images, indent=2)
    )
    rendered = rendered.replace(
        "{{OBSERVATION_LEDGER}}", json.dumps(ledger, indent=2)
    )
    rendered = rendered.replace(
        "{{CURRENT_IMAGES}}", json.dumps(current_images, indent=2)
    )
    rendered = rendered.replace(
        "{{ITERATION_STATE}}", json.dumps(iteration_state, indent=2)
    )

    return rendered
