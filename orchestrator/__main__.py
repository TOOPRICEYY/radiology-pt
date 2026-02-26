"""CLI entry point: python -m orchestrator"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

from orchestrator.config import OrchestratorConfig
from orchestrator.runner import run


def _load_repo_root_env() -> None:
    """Load .env from repository root for CLI runs, without overriding real env vars."""

    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parent.parent
    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def main():
    _load_repo_root_env()
    parser = argparse.ArgumentParser(
        description="Radiology diagnostic orchestrator — agentic loop over DICOM series"
    )
    parser.add_argument(
        "--dicom-root", default="./dicom",
        help="Path to the DICOM root directory (default: ./dicom)",
    )
    parser.add_argument(
        "--model", default="gemini-3.1-pro-preview",
        help="LLM model name (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--base-url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("GEMINI_API_KEY", ""),
        help="API key (default: $GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum orchestrator iterations (default: 10)",
    )
    parser.add_argument(
        "--prompt", default="prompt 1.0.0.txt",
        help="Path to the prompt template file",
    )
    parser.add_argument(
        "--question", required=True,
        help="Clinical question / indication for the study",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write JSON output (default: stdout)",
    )
    parser.add_argument(
        "--media-resolution", default="low", choices=["low", "medium", "high"],
        help="Default media resolution (default: low)",
    )
    parser.add_argument(
        "--thinking-level", default="low", choices=["low", "medium", "high"],
        help="Default thinking level (default: low)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode: dump full context, messages, raw LLM output, "
             "parsed responses, convergence state, and ledger each iteration",
    )
    parser.add_argument(
        "--debug-dir", default=None,
        help="Directory for debug artifacts (default: ./debug_runs/<timestamp>)",
    )
    parser.add_argument(
        "--live", action="store_true", default=True,
        help="Stream LLM output in real-time with collapsible view (default: on)",
    )
    parser.add_argument(
        "--no-live", dest="live", action="store_false",
        help="Disable real-time LLM output streaming",
    )
    parser.add_argument(
        "--expand-llm", action="store_true", default=False,
        help="Keep full LLM output expanded after each iteration (default: collapse)",
    )
    parser.add_argument(
        "--monitor-ui", action="store_true", default=False,
        help="Start a live local oversight UI (independent of --debug)",
    )
    parser.add_argument(
        "--monitor-host", default="127.0.0.1",
        help="Host/interface for the monitor UI server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--monitor-port", type=int, default=8765,
        help="Port for the monitor UI server (default: 8765, use 0 for auto)",
    )
    parser.add_argument(
        "--monitor-dir", default=None,
        help="Directory for live monitor assets (default: ./monitor_runs/<timestamp>)",
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: No API key. Set GEMINI_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    config = OrchestratorConfig.from_args(args)
    result = run(config)

    output_json = json.dumps(result, indent=2, default=str)

    if config.output_path:
        with open(config.output_path, "w") as f:
            f.write(output_json)
        print(f"\n[orchestrator] Output written to {config.output_path}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
