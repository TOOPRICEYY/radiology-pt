"""Debug utilities — dump full orchestrator state to terminal and disk each iteration."""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.config import OrchestratorConfig


# ── ANSI colors ──────────────────────────────────────────────────────────────

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"

_USE_COLOR = sys.stdout.isatty()


def _c(color: str, text: str) -> str:
    if _USE_COLOR:
        return f"{color}{text}{_RESET}"
    return text


# ── Debug session manager ────────────────────────────────────────────────────

class DebugSession:
    """Manages debug artifact output for one orchestrator run."""

    def __init__(self, config: OrchestratorConfig):
        self.enabled = config.debug
        if not self.enabled:
            return

        if config.debug_dir:
            self.dir = Path(config.debug_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dir = Path("debug_runs") / ts

        self.dir.mkdir(parents=True, exist_ok=True)
        self._print_header("DEBUG MODE ACTIVE")
        self._log(f"Artifacts → {self.dir.resolve()}")

        # Save config
        self._write_json("config.json", {
            "model": config.model,
            "base_url": config.base_url,
            "max_iterations": config.max_iterations,
            "dicom_root": config.dicom_root,
            "clinical_question": config.clinical_question,
            "default_media_resolution": config.default_media_resolution,
            "default_thinking_level": config.default_thinking_level,
            "prompt_path": config.prompt_path,
            "convergence_min_confirmatory": config.convergence_min_confirmatory,
            "convergence_min_negative_findings": config.convergence_min_negative_findings,
        })

    # ── Per-iteration dumps ──────────────────────────────────────────────

    def dump_available_images(self, available_images: list[dict]) -> None:
        if not self.enabled:
            return
        self._write_json("available_images.json", available_images)
        self._print_section("AVAILABLE IMAGES")
        for s in available_images:
            self._log(f"  {s['series_id']}: {s.get('modality','')} | "
                      f"{s.get('description','')} | slices {s.get('slice_range','')}")

    def dump_iteration_start(self, iteration: int) -> None:
        if not self.enabled:
            return
        self._print_header(f"ITERATION {iteration}")

    def dump_rendered_prompt(self, iteration: int, rendered: str) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        (idir / "system_prompt.txt").write_text(rendered)
        self._print_section(f"SYSTEM PROMPT ({len(rendered)} chars)")
        # Show first and last portions
        lines = rendered.splitlines()
        preview_lines = 30
        if len(lines) <= preview_lines * 2:
            self._log_block(rendered)
        else:
            self._log_block("\n".join(lines[:preview_lines]))
            self._log(_c(_DIM, f"  ... ({len(lines) - preview_lines * 2} lines omitted) ..."))
            self._log_block("\n".join(lines[-preview_lines:]))
        self._log(_c(_DIM, f"  Full prompt saved → {idir / 'system_prompt.txt'}"))

    def dump_messages(self, iteration: int, messages: list[dict]) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        # Save full messages with images truncated
        sanitized = _sanitize_messages(messages)
        self._write_json_to(idir / "messages.json", sanitized)

        self._print_section("MESSAGES TO LLM")
        for idx, msg in enumerate(sanitized):
            role = msg["role"]
            content = msg["content"]
            self._log(f"  [{idx}] role={_c(_CYAN, role)}")
            if isinstance(content, str):
                preview = content[:300]
                if len(content) > 300:
                    preview += f"... ({len(content)} chars total)"
                self._log(f"      {preview}")
            elif isinstance(content, list):
                for part in content:
                    ptype = part.get("type", "?")
                    if ptype == "text":
                        text = part["text"]
                        preview = text[:200]
                        if len(text) > 200:
                            preview += "..."
                        self._log(f"      [text] {preview}")
                    elif ptype == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # Extract size info
                            b64_part = url.split(",", 1)[1] if "," in url else url
                            size_kb = len(b64_part) * 3 / 4 / 1024
                            self._log(f"      [image] base64 PNG ~{size_kb:.0f} KB")
                        else:
                            self._log(f"      [image] {url[:100]}")

        # Save images as actual PNGs for visual inspection
        img_count = 0
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/png;base64,"):
                        b64 = url.split(",", 1)[1]
                        img_path = idir / f"image_{img_count}.png"
                        img_path.write_bytes(base64.b64decode(b64))
                        img_count += 1
        if img_count:
            self._log(_c(_DIM, f"  Saved {img_count} image(s) as PNG → {idir}"))

    def dump_raw_llm_output(self, iteration: int, raw: str) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        (idir / "raw_llm_output.txt").write_text(raw)
        self._print_section(f"RAW LLM OUTPUT ({len(raw)} chars)")
        lines = raw.splitlines()
        if len(lines) <= 60:
            self._log_block(raw)
        else:
            self._log_block("\n".join(lines[:30]))
            self._log(_c(_DIM, f"  ... ({len(lines) - 60} lines omitted) ..."))
            self._log_block("\n".join(lines[-30:]))
        self._log(_c(_DIM, f"  Full output saved → {idir / 'raw_llm_output.txt'}"))

    def dump_parsed_response(self, iteration: int, response_dict: dict) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        self._write_json_to(idir / "parsed_response.json", response_dict)
        self._print_section("PARSED AGENT RESPONSE")
        formatted = json.dumps(response_dict, indent=2, default=str)
        lines = formatted.splitlines()
        if len(lines) <= 80:
            self._log_block(formatted)
        else:
            self._log_block("\n".join(lines[:40]))
            self._log(_c(_DIM, f"  ... ({len(lines) - 80} lines omitted) ..."))
            self._log_block("\n".join(lines[-40:]))

    def dump_parse_error(self, iteration: int, error: Exception, raw: str) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        (idir / "parse_error.txt").write_text(f"{type(error).__name__}: {error}\n\n{raw}")
        self._print_section("PARSE ERROR")
        self._log(_c(_RED, f"  {type(error).__name__}: {error}"))

    def dump_convergence(
        self, iteration: int, converged: bool, reasons: list[str],
        next_request_null: bool,
    ) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        state = {
            "converged": converged,
            "next_request_null": next_request_null,
            "unmet_reasons": reasons,
        }
        self._write_json_to(idir / "convergence.json", state)
        self._print_section("CONVERGENCE CHECK")
        status = _c(_GREEN, "CONVERGED") if converged else _c(_YELLOW, "NOT CONVERGED")
        self._log(f"  next_request=null: {next_request_null}")
        self._log(f"  Status: {status}")
        if reasons:
            self._log("  Unmet criteria:")
            for r in reasons:
                self._log(f"    - {_c(_YELLOW, r)}")

    def dump_mcp_calls(self, iteration: int, calls: list[dict], results: list[dict]) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        self._write_json_to(idir / "mcp_calls.json", calls)

        # Save results (metadata without base64) + extracted images as PNGs
        img_dir = idir / "images"
        img_dir.mkdir(exist_ok=True)
        sanitized_results = []
        image_manifest = []  # tracks every saved image
        global_img_idx = 0

        for call_idx, r in enumerate(results):
            sr = {k: v for k, v in r.items() if k != "images"}
            images = r.get("images", [])
            sr["num_images"] = len(images)
            sr["image_files"] = []

            tool = r.get("tool_name", "?")
            args = r.get("arguments", {})

            for local_idx, b64 in enumerate(images):
                # Build a descriptive filename from the call
                fname = _image_filename(tool, args, global_img_idx, local_idx)
                img_path = img_dir / fname
                try:
                    img_path.write_bytes(base64.b64decode(b64))
                except Exception:
                    img_path.write_text(f"<decode error, {len(b64)} base64 chars>")

                sr["image_files"].append(fname)
                image_manifest.append({
                    "file": fname,
                    "call_index": call_idx,
                    "tool_name": tool,
                    "arguments": args,
                    "image_index_in_call": local_idx,
                })
                global_img_idx += 1

            sanitized_results.append(sr)

        self._write_json_to(idir / "mcp_results.json", sanitized_results)
        if image_manifest:
            self._write_json_to(img_dir / "manifest.json", image_manifest)

        # Terminal output
        self._print_section(f"MCP CALLS ({len(calls)} calls)")
        for idx, call in enumerate(calls):
            tool = call.get("tool_name", "?")
            args = call.get("arguments", {})
            self._log(f"  [{idx}] {_c(_MAGENTA, tool)}")
            for k, v in args.items():
                self._log(f"       {k}: {v}")
        self._print_section(f"MCP RESULTS")
        for idx, res in enumerate(sanitized_results):
            tool = res.get("tool_name", "?")
            err = res.get("error")
            n_img = res.get("num_images", 0)
            files = res.get("image_files", [])
            if err:
                self._log(f"  [{idx}] {tool}: {_c(_RED, f'ERROR: {err}')}")
            else:
                self._log(f"  [{idx}] {tool}: {_c(_GREEN, f'{n_img} images')}")
                for f in files:
                    self._log(f"       → {_c(_DIM, str(img_dir / f))}")

    def dump_annotated_images(
        self, iteration: int, annotated_b64: list[str], response,
    ) -> None:
        """Save bounding-box-annotated images to the debug directory."""
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        ann_dir = idir / "annotated"
        ann_dir.mkdir(exist_ok=True)

        for idx, b64 in enumerate(annotated_b64):
            img_path = ann_dir / f"annotated_{idx:03d}.png"
            try:
                img_path.write_bytes(base64.b64decode(b64))
            except Exception:
                img_path.write_text(f"<decode error, {len(b64)} base64 chars>")

        # Save a summary of which boxes were drawn
        box_summary = []
        for f in response.findings:
            for bb in f.bounding_boxes:
                box_summary.append({
                    "finding_id": f.finding_id,
                    "confidence": f.confidence,
                    "slice_ref": bb.slice_ref,
                    "box_2d": bb.box_2d,
                    "label": bb.label,
                })
        self._write_json_to(ann_dir / "boxes.json", box_summary)

        n_boxes = sum(len(f.bounding_boxes) for f in response.findings)
        self._print_section(f"ANNOTATED IMAGES ({n_boxes} boxes on {len(annotated_b64)} images)")
        for bs in box_summary:
            self._log(
                f"  {_c(_MAGENTA, bs['finding_id'])} [{bs['confidence']}] "
                f"@ {bs['slice_ref']} box={bs['box_2d']}"
            )
        self._log(_c(_DIM, f"  Saved → {ann_dir}"))

    def dump_ledger(
        self, iteration: int,
        findings_ledger: list[dict],
        negative_ledger: list[dict],
    ) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        ledger = {"findings": findings_ledger, "negative_findings": negative_ledger}
        self._write_json_to(idir / "ledger_snapshot.json", ledger)
        self._print_section(
            f"LEDGER SNAPSHOT ({len(findings_ledger)} findings, "
            f"{len(negative_ledger)} negative)"
        )
        for f in findings_ledger:
            fid = f.get("finding_id", "?")
            conf = f.get("confidence", "?")
            etype = f.get("evidence_type", "?")
            it = f.get("iteration", "?")
            color = _GREEN if etype == "confirmatory" else (_RED if etype == "contradictory" else _CYAN)
            self._log(f"  [{it}] {_c(color, etype):>20s}  {fid}  conf={conf}  {f.get('observation', '')[:60]}")
        if negative_ledger:
            self._log(f"  + {len(negative_ledger)} negative findings")

    def dump_iteration_state(self, iteration: int, state_dict: dict) -> None:
        if not self.enabled:
            return
        idir = self._iter_dir(iteration)
        self._write_json_to(idir / "iteration_state.json", state_dict)
        self._print_section("ITERATION STATE")
        self._log(f"  iteration:        {state_dict.get('iteration')}")
        self._log(f"  max_iterations:   {state_dict.get('max_iterations')}")
        self._log(f"  thinking_level:   {state_dict.get('thinking_level')}")
        self._log(f"  media_resolution: {state_dict.get('media_resolution')}")
        directives = state_dict.get("directives", [])
        if directives:
            self._log(f"  directives:")
            for d in directives:
                self._log(f"    - {_c(_YELLOW, d)}")

    def dump_final_report(self, raw: str, parsed: dict) -> None:
        if not self.enabled:
            return
        fdir = self.dir / "final_report"
        fdir.mkdir(exist_ok=True)
        (fdir / "raw_llm_output.txt").write_text(raw)
        self._write_json_to(fdir / "parsed_report.json", parsed)
        self._print_header("FINAL REPORT")
        formatted = json.dumps(parsed, indent=2, default=str)
        self._log_block(formatted[:3000])
        if len(formatted) > 3000:
            self._log(_c(_DIM, f"  ... ({len(formatted)} chars total, see {fdir / 'parsed_report.json'})"))

    def dump_summary(self, total_iterations: int, n_findings: int, n_negative: int) -> None:
        if not self.enabled:
            return
        self._print_header("DEBUG SESSION SUMMARY")
        self._log(f"  Total iterations: {total_iterations}")
        self._log(f"  Findings:         {n_findings}")
        self._log(f"  Negative:         {n_negative}")
        self._log(f"  Artifacts dir:    {self.dir.resolve()}")
        self._log("")

    # ── Internals ────────────────────────────────────────────────────────

    def _iter_dir(self, iteration: int) -> Path:
        if iteration < 0:
            d = self.dir / "final_report"
        else:
            d = self.dir / f"iter_{iteration:03d}"
        d.mkdir(exist_ok=True)
        return d

    def _write_json(self, name: str, data: Any) -> None:
        path = self.dir / name
        path.write_text(json.dumps(data, indent=2, default=str))

    def _write_json_to(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, default=str))

    def _print_header(self, text: str) -> None:
        bar = "=" * 72
        print(f"\n{_c(_BOLD, bar)}")
        print(f"{_c(_BOLD, f'  [DEBUG] {text}')}")
        print(f"{_c(_BOLD, bar)}")

    def _print_section(self, text: str) -> None:
        print(f"\n{_c(_BOLD + _BLUE, f'  ── {text} ──')}")

    def _log(self, text: str) -> None:
        print(text)

    def _log_block(self, text: str) -> None:
        for line in text.splitlines():
            print(f"  {_c(_DIM, '│')} {line}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _image_filename(tool: str, args: dict, global_idx: int, local_idx: int) -> str:
    """Build a descriptive PNG filename from the MCP call that produced it.

    Examples:
        extract_dicom_range_SER00014_s0-10_soft_tissue_002.png
        get_pixel_stats_SER00001_idx5_003.png
    """
    parts = [tool.replace("dicom_", "")]

    series = args.get("series_id", "")
    if series:
        parts.append(series)

    # Slice info
    if "start" in args and "end" in args:
        parts.append(f"s{args['start']}-{args['end']}")
    elif "index" in args:
        parts.append(f"idx{args['index']}")

    # Window
    window = args.get("window")
    if isinstance(window, str):
        parts.append(window)
    elif isinstance(window, dict):
        parts.append(f"w{window.get('center', '')}-{window.get('width', '')}")

    # Step
    step = args.get("step")
    if step and int(step) > 1:
        parts.append(f"step{step}")

    parts.append(f"{global_idx:03d}")

    name = "_".join(str(p) for p in parts)
    # Sanitize for filesystem
    name = "".join(c if c.isalnum() or c in "_-." else "_" for c in name)
    return f"{name}.png"


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Replace base64 image data with size placeholders for JSON logging."""
    sanitized = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            sanitized.append(msg)
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        b64_part = url.split(",", 1)[1] if "," in url else ""
                        size_kb = len(b64_part) * 3 / 4 / 1024
                        new_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"<base64 PNG ~{size_kb:.0f} KB>"},
                        })
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            sanitized.append({"role": msg["role"], "content": new_parts})
        else:
            sanitized.append(msg)
    return sanitized
