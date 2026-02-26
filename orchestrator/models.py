"""Data models for the orchestrator's iteration loop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


# ── Agent response components ────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """A bounding box in Gemini's normalized 0-1000 coordinate space.

    box_2d is [y1, x1, y2, x2] where each value is 0-1000.
    slice_ref ties the box to a specific image (e.g. "SER00014:15:soft_tissue").
    """
    box_2d: list[int]  # [y1, x1, y2, x2] in 0-1000 space
    slice_ref: str = ""
    label: str = ""

    def to_dict(self) -> dict:
        d: dict = {"box_2d": self.box_2d}
        if self.slice_ref:
            d["slice_ref"] = self.slice_ref
        if self.label:
            d["label"] = self.label
        return d

    def to_pixel_coords(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert 0-1000 normalized coords to absolute pixel coords.

        Returns (x1, y1, x2, y2) in pixel space.
        """
        y1 = int(self.box_2d[0] / 1000 * height)
        x1 = int(self.box_2d[1] / 1000 * width)
        y2 = int(self.box_2d[2] / 1000 * height)
        x2 = int(self.box_2d[3] / 1000 * width)
        return x1, y1, x2, y2


@dataclass
class Finding:
    finding_id: str
    region: str
    observation: str
    confidence: str  # high | moderate | low
    evidence_type: str  # primary | confirmatory | contradictory
    slices_examined: list[str] = field(default_factory=list)
    differential: list[str] = field(default_factory=list)
    bounding_boxes: list[BoundingBox] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "finding_id": self.finding_id,
            "region": self.region,
            "observation": self.observation,
            "confidence": self.confidence,
            "evidence_type": self.evidence_type,
            "slices_examined": self.slices_examined,
            "differential": self.differential,
        }
        if self.bounding_boxes:
            d["bounding_boxes"] = [bb.to_dict() for bb in self.bounding_boxes]
        return d


@dataclass
class NegativeFinding:
    region: str
    looked_for: str
    result: str  # "absent"
    slices_examined: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "looked_for": self.looked_for,
            "result": self.result,
            "slices_examined": self.slices_examined,
        }


@dataclass
class CoverageSummary:
    regions_examined: list[str] = field(default_factory=list)
    regions_remaining: list[str] = field(default_factory=list)
    window_levels_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "regions_examined": self.regions_examined,
            "regions_remaining": self.regions_remaining,
            "window_levels_applied": self.window_levels_applied,
        }


@dataclass
class NextRequest:
    rationale: str
    mcp_calls: list[dict] = field(default_factory=list)
    suggested_thinking_level: str = "low"
    suggested_media_resolution: str = "low"

    def to_dict(self) -> dict:
        return {
            "rationale": self.rationale,
            "mcp_calls": self.mcp_calls,
            "suggested_thinking_level": self.suggested_thinking_level,
            "suggested_media_resolution": self.suggested_media_resolution,
        }


@dataclass
class AgentResponse:
    findings: list[Finding] = field(default_factory=list)
    negative_findings: list[NegativeFinding] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    coverage_summary: CoverageSummary = field(default_factory=CoverageSummary)
    next_request: NextRequest | None = None
    # For final report mode
    report: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {}
        if self.report is not None:
            d["report"] = self.report
            return d
        d["findings"] = [f.to_dict() for f in self.findings]
        d["negative_findings"] = [nf.to_dict() for nf in self.negative_findings]
        d["uncertainty_flags"] = self.uncertainty_flags
        d["coverage_summary"] = self.coverage_summary.to_dict()
        d["next_request"] = self.next_request.to_dict() if self.next_request else None
        return d


# ── Orchestrator bookkeeping ─────────────────────────────────────────────────

@dataclass
class LedgerEntry:
    """A Finding annotated with iteration metadata."""
    finding: Finding
    iteration: int
    image_set_hash: str = ""

    def to_dict(self) -> dict:
        d = self.finding.to_dict()
        d["iteration"] = self.iteration
        d["image_set_hash"] = self.image_set_hash
        return d


@dataclass
class NegativeLedgerEntry:
    """A NegativeFinding annotated with iteration number."""
    negative_finding: NegativeFinding
    iteration: int

    def to_dict(self) -> dict:
        d = self.negative_finding.to_dict()
        d["iteration"] = self.iteration
        return d


@dataclass
class IterationState:
    iteration: int = 0
    max_iterations: int = 10
    thinking_level: str = "low"
    media_resolution: str = "low"
    directives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "thinking_level": self.thinking_level,
            "media_resolution": self.media_resolution,
            "directives": self.directives,
        }


# ── Parsing ──────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """Extract JSON from raw LLM output that may contain markdown fences."""
    # Try to find ```json ... ``` block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Otherwise try raw as-is (maybe it's already pure JSON)
    return raw.strip()


def parse_agent_response(raw: str) -> AgentResponse:
    """Parse raw LLM JSON output into an AgentResponse."""
    cleaned = _extract_json(raw)
    data = json.loads(cleaned)

    # Final report mode
    if "report" in data:
        return AgentResponse(report=data["report"])

    findings = [
        Finding(
            finding_id=f.get("finding_id", ""),
            region=f.get("region", ""),
            observation=f.get("observation", ""),
            confidence=f.get("confidence", "low"),
            evidence_type=f.get("evidence_type", "primary"),
            slices_examined=f.get("slices_examined", []),
            differential=f.get("differential", []),
            bounding_boxes=[
                BoundingBox(
                    box_2d=bb.get("box_2d", [0, 0, 0, 0]),
                    slice_ref=bb.get("slice_ref", ""),
                    label=bb.get("label", ""),
                )
                for bb in f.get("bounding_boxes", [])
                if isinstance(bb, dict) and "box_2d" in bb
            ],
        )
        for f in data.get("findings", [])
    ]

    negative_findings = [
        NegativeFinding(
            region=nf.get("region", ""),
            looked_for=nf.get("looked_for", ""),
            result=nf.get("result", "absent"),
            slices_examined=nf.get("slices_examined", []),
        )
        for nf in data.get("negative_findings", [])
    ]

    cs_data = data.get("coverage_summary", {})
    coverage_summary = CoverageSummary(
        regions_examined=cs_data.get("regions_examined", []),
        regions_remaining=cs_data.get("regions_remaining", []),
        window_levels_applied=cs_data.get("window_levels_applied", []),
    )

    nr_data = data.get("next_request")
    next_request = None
    if nr_data is not None:
        next_request = NextRequest(
            rationale=nr_data.get("rationale", ""),
            mcp_calls=nr_data.get("mcp_calls", []),
            suggested_thinking_level=nr_data.get("suggested_thinking_level", "low"),
            suggested_media_resolution=nr_data.get("suggested_media_resolution", "low"),
        )

    return AgentResponse(
        findings=findings,
        negative_findings=negative_findings,
        uncertainty_flags=data.get("uncertainty_flags", []),
        coverage_summary=coverage_summary,
        next_request=next_request,
    )
