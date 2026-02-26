"""Runtime configuration for the orchestrator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OrchestratorConfig:
    dicom_root: str = "./dicom"
    model: str = "gemini-3.1-pro-preview"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    max_iterations: int = 10
    default_media_resolution: str = "low"
    default_thinking_level: str = "low"
    prompt_path: str = "prompt 1.0.0.txt"
    convergence_min_confirmatory: int = 1
    convergence_min_negative_findings: int = 3
    clinical_question: str = ""
    output_path: str | None = None
    debug: bool = False
    debug_dir: str | None = None
    live: bool = True
    expand_llm: bool = False
    monitor_ui: bool = False
    monitor_host: str = "127.0.0.1"
    monitor_port: int = 8765
    monitor_dir: str | None = None

    @classmethod
    def from_args(cls, namespace) -> OrchestratorConfig:
        return cls(
            dicom_root=namespace.dicom_root,
            model=namespace.model,
            base_url=namespace.base_url,
            api_key=namespace.api_key or os.environ.get("GEMINI_API_KEY", ""),
            max_iterations=namespace.max_iterations,
            default_media_resolution=namespace.media_resolution,
            default_thinking_level=namespace.thinking_level,
            prompt_path=namespace.prompt,
            clinical_question=namespace.question,
            output_path=namespace.output,
            debug=namespace.debug,
            debug_dir=namespace.debug_dir,
            live=namespace.live,
            expand_llm=namespace.expand_llm,
            monitor_ui=getattr(namespace, "monitor_ui", False),
            monitor_host=getattr(namespace, "monitor_host", "127.0.0.1"),
            monitor_port=getattr(namespace, "monitor_port", 8765),
            monitor_dir=getattr(namespace, "monitor_dir", None),
        )

    def resolve_prompt_path(self) -> Path:
        """Resolve prompt path relative to project root if not absolute."""
        p = Path(self.prompt_path)
        if p.is_absolute():
            return p
        # Try relative to CWD first, then relative to this package's parent
        if p.exists():
            return p.resolve()
        pkg_root = Path(__file__).resolve().parent.parent
        candidate = pkg_root / self.prompt_path
        if candidate.exists():
            return candidate
        return p  # fall through, will error on read
