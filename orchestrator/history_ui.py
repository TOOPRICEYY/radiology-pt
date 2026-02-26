"""Build a local HTML UI for inspecting orchestrator image-view history.

Usage:
    python -m orchestrator.history_ui --run debug_runs/<timestamp>

The UI focuses on:
  - what image batches were requested (and why)
  - what images were extracted/viewed by the agent next iteration
  - what judgments the agent rendered after that review
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ITER_DIR_RE = re.compile(r"^iter_(\d{3})$")


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text()
    except Exception:
        return None


def _latest_debug_run(debug_runs_root: Path) -> Path:
    candidates = [
        p for p in debug_runs_root.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No debug runs found under {debug_runs_root}")
    return sorted(candidates)[-1]


def _iter_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = ITER_DIR_RE.match(child.name)
        if not m:
            continue
        out.append((int(m.group(1)), child))
    return sorted(out, key=lambda t: t[0])


def _count_message_images(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                count += 1
    return count


def _first_user_text(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
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


def _json_html(value: Any) -> str:
    return html.escape(json.dumps(value, indent=2, default=str))


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def _relative_path(from_file: Path, to_file: Path) -> str:
    return os.path.relpath(to_file, start=from_file.parent)


def _decode_image_to_asset(
    png_b64: str,
    dst_dir: Path,
    *,
    iter_idx: int,
    call_idx: int,
    local_idx: int,
    series_id: str | None,
    instance_index: int | None,
) -> Path | None:
    if not png_b64 or not isinstance(png_b64, str):
        return None
    if png_b64.startswith("<"):  # placeholder from sanitized debug artifacts
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    parts = [f"it{iter_idx:03d}", f"c{call_idx:02d}", f"i{local_idx:02d}"]
    if series_id:
        parts.append(_slug(series_id))
    if instance_index is not None:
        parts.append(f"idx{instance_index}")
    dst = dst_dir / ("_".join(parts) + ".png")
    try:
        dst.write_bytes(base64.b64decode(png_b64))
    except Exception:
        return None
    return dst


def _strip_png_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k == "png_base64" and isinstance(v, str) and v:
                out[k] = "<image>"
            else:
                out[k] = _strip_png_fields(v)
        return out
    if isinstance(value, list):
        return [_strip_png_fields(v) for v in value]
    return value


def _extract_instance_records_from_result(result: Any) -> list[dict[str, Any]]:
    """Extract image-bearing instances from a call result (supports current and legacy shapes)."""

    extracted: list[dict[str, Any]] = []

    def add_instance(root: dict[str, Any] | None, inst: dict[str, Any], source_key: str) -> None:
        extracted.append(
            {
                "png_base64": inst.get("png_base64"),
                "series_id": (root or {}).get("series_id") or inst.get("series_id"),
                "instance_index": inst.get("index"),
                "instance_number": inst.get("instance_number"),
                "file_name": inst.get("file_name"),
                "path": inst.get("path"),
                "rows": inst.get("rows"),
                "columns": inst.get("columns"),
                "render": inst.get("render"),
                "requested_ranges": (root or {}).get("requested_ranges"),
                "resolved_indices": (root or {}).get("resolved_indices"),
                "step": (root or {}).get("step"),
                "window": (root or {}).get("window"),
                "crop": (root or {}).get("crop"),
                "source_key": source_key,
            }
        )

    if isinstance(result, dict):
        if isinstance(result.get("png_base64"), str):
            add_instance(None, result, "root")
        for key in ("instances", "items"):
            instances = result.get(key)
            if isinstance(instances, list):
                for inst in instances:
                    if isinstance(inst, dict) and isinstance(inst.get("png_base64"), str):
                        add_instance(result, inst, key)
    elif isinstance(result, list):
        for inst in result:
            if isinstance(inst, dict) and isinstance(inst.get("png_base64"), str):
                add_instance(None, inst, "list")

    return extracted


def _build_call_image_records(
    *,
    run_dir: Path,
    output_html: Path,
    iter_idx: int,
    call_idx: int,
    call_result: dict[str, Any],
    assets_root: Path,
    manifest_entries: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build normalized image records for one call, using saved image files or fallback decode."""

    result = call_result.get("result")
    extracted_instances = _extract_instance_records_from_result(result)

    manifest_entries = manifest_entries or []
    manifest_entries = [
        m for m in manifest_entries
        if isinstance(m, dict) and int(m.get("call_index", -1)) == call_idx
    ]
    manifest_entries.sort(key=lambda m: int(m.get("image_index_in_call", 0)))

    records: list[dict[str, Any]] = []

    if manifest_entries:
        for local_idx, m in enumerate(manifest_entries):
            file_name = m.get("file")
            if not isinstance(file_name, str):
                continue
            img_path = run_dir / f"iter_{iter_idx:03d}" / "images" / file_name
            if not img_path.exists():
                continue
            meta = extracted_instances[local_idx] if local_idx < len(extracted_instances) else {}
            records.append(
                {
                    "call_index": call_idx,
                    "tool_name": call_result.get("tool_name"),
                    "arguments": call_result.get("arguments") or {},
                    "src": _relative_path(output_html, img_path),
                    "series_id": meta.get("series_id"),
                    "instance_index": meta.get("instance_index"),
                    "instance_number": meta.get("instance_number"),
                    "rows": meta.get("rows"),
                    "columns": meta.get("columns"),
                    "path": meta.get("path"),
                    "render": meta.get("render"),
                    "resolved_indices": meta.get("resolved_indices"),
                    "window": meta.get("window"),
                    "crop": meta.get("crop"),
                    "step": meta.get("step"),
                }
            )
        return records

    # Fallback: recover PNGs from base64 still present in mcp_results.json (old runs).
    for local_idx, meta in enumerate(extracted_instances):
        png_b64 = meta.get("png_base64")
        dst_dir = assets_root / f"iter_{iter_idx:03d}"
        img_path = _decode_image_to_asset(
            png_b64,
            dst_dir,
            iter_idx=iter_idx,
            call_idx=call_idx,
            local_idx=local_idx,
            series_id=meta.get("series_id"),
            instance_index=meta.get("instance_index"),
        )
        if img_path is None:
            continue
        records.append(
            {
                "call_index": call_idx,
                "tool_name": call_result.get("tool_name"),
                "arguments": call_result.get("arguments") or {},
                "src": _relative_path(output_html, img_path),
                "series_id": meta.get("series_id"),
                "instance_index": meta.get("instance_index"),
                "instance_number": meta.get("instance_number"),
                "rows": meta.get("rows"),
                "columns": meta.get("columns"),
                "path": meta.get("path"),
                "render": meta.get("render"),
                "resolved_indices": meta.get("resolved_indices"),
                "window": meta.get("window"),
                "crop": meta.get("crop"),
                "step": meta.get("step"),
            }
        )

    return records


def _collect_iteration_images(
    *,
    run_dir: Path,
    output_html: Path,
    iter_idx: int,
    mcp_results: Any,
    assets_root: Path,
) -> list[dict[str, Any]]:
    if not isinstance(mcp_results, list):
        return []

    manifest_path = run_dir / f"iter_{iter_idx:03d}" / "images" / "manifest.json"
    manifest_entries = _load_json(manifest_path) if manifest_path.exists() else None
    if not isinstance(manifest_entries, list):
        manifest_entries = None

    all_images: list[dict[str, Any]] = []
    for call_idx, call_result in enumerate(mcp_results):
        if not isinstance(call_result, dict):
            continue
        all_images.extend(
            _build_call_image_records(
                run_dir=run_dir,
                output_html=output_html,
                iter_idx=iter_idx,
                call_idx=call_idx,
                call_result=call_result,
                assets_root=assets_root,
                manifest_entries=manifest_entries,
            )
        )
    return all_images


def _call_summaries(mcp_results: Any, recovered_images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(mcp_results, list):
        return []
    recovered_by_call: dict[int, int] = {}
    for img in recovered_images:
        call_idx = int(img.get("call_index", -1))
        recovered_by_call[call_idx] = recovered_by_call.get(call_idx, 0) + 1

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(mcp_results):
        if not isinstance(row, dict):
            continue
        result = row.get("result")
        result_kind = type(result).__name__
        result_keys: list[str] | None = list(result.keys()) if isinstance(result, dict) else None
        out.append(
            {
                "call_index": idx,
                "tool_name": row.get("tool_name"),
                "arguments": row.get("arguments") or {},
                "error": row.get("error"),
                "bridge_num_images": row.get("num_images"),
                "bridge_image_files": row.get("image_files") or [],
                "recovered_num_images": recovered_by_call.get(idx, 0),
                "result_kind": result_kind,
                "result_keys": result_keys,
                "result_preview": _strip_png_fields(result),
            }
        )
    return out


def _build_run_model(run_dir: Path, output_html: Path, assets_root: Path) -> dict[str, Any]:
    available_images = _load_json(run_dir / "available_images.json") or []
    config = _load_json(run_dir / "config.json") or {}
    final_report = _load_json(run_dir / "final_report" / "parsed_report.json")

    iter_infos: list[dict[str, Any]] = []
    for iter_idx, idir in _iter_dirs(run_dir):
        parsed = _load_json(idir / "parsed_response.json")
        mcp_calls = _load_json(idir / "mcp_calls.json")
        mcp_results = _load_json(idir / "mcp_results.json")
        messages = _load_json(idir / "messages.json")
        convergence = _load_json(idir / "convergence.json")
        ledger_snapshot = _load_json(idir / "ledger_snapshot.json")

        recovered_images = _collect_iteration_images(
            run_dir=run_dir,
            output_html=output_html,
            iter_idx=iter_idx,
            mcp_results=mcp_results,
            assets_root=assets_root,
        )

        iter_infos.append(
            {
                "iteration": iter_idx,
                "parsed_response": parsed,
                "mcp_calls": mcp_calls or [],
                "mcp_results": mcp_results or [],
                "call_summaries": _call_summaries(mcp_results, recovered_images),
                "recovered_images": recovered_images,
                "messages_image_count": _count_message_images(messages),
                "messages_first_user_text": _first_user_text(messages),
                "convergence": convergence,
                "ledger_snapshot": ledger_snapshot,
            }
        )

    initial_judgment = None
    if iter_infos:
        initial_judgment = {
            "iteration": 0,
            "parsed_response": iter_infos[0].get("parsed_response"),
            "messages_image_count": iter_infos[0].get("messages_image_count"),
            "messages_first_user_text": iter_infos[0].get("messages_first_user_text"),
        }

    batches: list[dict[str, Any]] = []
    for idx, info in enumerate(iter_infos):
        parsed = info.get("parsed_response") or {}
        next_request = parsed.get("next_request") if isinstance(parsed, dict) else None
        has_call_artifacts = bool(info.get("mcp_calls")) or bool(info.get("mcp_results"))
        if not next_request and not has_call_artifacts:
            continue

        review_info = iter_infos[idx + 1] if idx + 1 < len(iter_infos) else None
        recovered_images = info.get("recovered_images") or []
        batches.append(
            {
                "request_iteration": info["iteration"],
                "review_iteration": review_info["iteration"] if review_info else None,
                "rationale": (next_request or {}).get("rationale") if isinstance(next_request, dict) else None,
                "suggested_thinking_level": (next_request or {}).get("suggested_thinking_level") if isinstance(next_request, dict) else None,
                "suggested_media_resolution": (next_request or {}).get("suggested_media_resolution") if isinstance(next_request, dict) else None,
                "mcp_calls": info.get("mcp_calls") or ((next_request or {}).get("mcp_calls") if isinstance(next_request, dict) else []),
                "call_summaries": info.get("call_summaries") or [],
                "images": recovered_images,
                "recovered_image_count": len(recovered_images),
                "review_messages_image_count": review_info.get("messages_image_count") if review_info else None,
                "review_messages_first_user_text": review_info.get("messages_first_user_text") if review_info else None,
                "review_parsed_response": review_info.get("parsed_response") if review_info else None,
                "review_convergence": review_info.get("convergence") if review_info else None,
                "delivery_warning": (
                    review_info is not None
                    and (review_info.get("messages_image_count") or 0) != len(recovered_images)
                ),
            }
        )

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "available_images": available_images,
        "initial_judgment": initial_judgment,
        "batches": batches,
        "final_report": final_report,
        "iteration_count": len(iter_infos),
    }


def _chip(text: str, klass: str = "") -> str:
    classes = "chip" + (f" {klass}" if klass else "")
    return f"<span class='{classes}'>{html.escape(text)}</span>"


def _render_kv_table(d: dict[str, Any]) -> str:
    if not d:
        return "<div class='muted'>No arguments</div>"
    rows = []
    for k, v in d.items():
        rendered = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        rows.append(
            f"<tr><th>{html.escape(str(k))}</th><td><code>{html.escape(rendered)}</code></td></tr>"
        )
    return f"<table class='kv-table'>{''.join(rows)}</table>"


def _render_image_card(img: dict[str, Any]) -> str:
    title_parts = []
    if img.get("series_id"):
        title_parts.append(str(img["series_id"]))
    if img.get("instance_index") is not None:
        title_parts.append(f"idx {img['instance_index']}")
    elif img.get("instance_number") is not None:
        title_parts.append(f"instance {img['instance_number']}")
    title = " • ".join(title_parts) or "Image"

    meta_lines = []
    if img.get("path"):
        meta_lines.append(f"path: <code>{html.escape(str(img['path']))}</code>")
    if img.get("rows") and img.get("columns"):
        meta_lines.append(f"size: {img['columns']} x {img['rows']}")
    render = img.get("render") or {}
    if isinstance(render, dict):
        out_size = render.get("output_size")
        if isinstance(out_size, dict):
            meta_lines.append(f"rendered: {out_size.get('width')} x {out_size.get('height')}")
        if render.get("window"):
            meta_lines.append(f"window: <code>{html.escape(json.dumps(render['window']))}</code>")
        if render.get("crop"):
            meta_lines.append(f"crop: <code>{html.escape(json.dumps(render['crop']))}</code>")
    if img.get("step"):
        meta_lines.append(f"step: {img['step']}")

    meta_html = "".join(f"<div class='meta-line'>{line}</div>" for line in meta_lines)
    return (
        "<figure class='thumb-card'>"
        f"<a href='{html.escape(str(img['src']))}' target='_blank' rel='noopener noreferrer'>"
        f"<img loading='lazy' src='{html.escape(str(img['src']))}' alt='{html.escape(title)}' />"
        "</a>"
        f"<figcaption><div class='thumb-title'>{html.escape(title)}</div>{meta_html}</figcaption>"
        "</figure>"
    )


def _render_findings(parsed: dict[str, Any] | None) -> str:
    if not isinstance(parsed, dict):
        return "<div class='muted'>No parsed response available.</div>"

    findings = parsed.get("findings") or []
    negatives = parsed.get("negative_findings") or []
    uncertainties = parsed.get("uncertainty_flags") or []
    coverage = parsed.get("coverage_summary") or {}

    sections: list[str] = []

    # Findings
    if findings:
        items = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            chips = " ".join(
                [
                    _chip(f"confidence: {f.get('confidence', '?')}", "conf"),
                    _chip(f"evidence: {f.get('evidence_type', '?')}", "etype"),
                    _chip(f"region: {f.get('region', '?')}", "region"),
                ]
            )
            slices = f.get("slices_examined") or []
            diff = f.get("differential") or []
            items.append(
                "<li class='judgment-item'>"
                f"<div class='judgment-head'>{chips}</div>"
                f"<div class='judgment-body'>{html.escape(str(f.get('observation', '')))}</div>"
                + (f"<div class='judgment-sub'><strong>Slices:</strong> {html.escape(', '.join(map(str, slices)))}</div>" if slices else "")
                + (f"<div class='judgment-sub'><strong>Differential:</strong> {html.escape(', '.join(map(str, diff)))}</div>" if diff else "")
                + "</li>"
            )
        sections.append(f"<section><h4>Findings ({len(findings)})</h4><ul class='judgment-list'>{''.join(items)}</ul></section>")
    else:
        sections.append("<section><h4>Findings (0)</h4><div class='muted'>No positive findings recorded.</div></section>")

    # Negative findings
    if negatives:
        items = []
        for nf in negatives:
            if not isinstance(nf, dict):
                continue
            slices = nf.get("slices_examined") or []
            items.append(
                "<li class='judgment-item'>"
                f"<div class='judgment-head'>{_chip(str(nf.get('region', '?')), 'region')} {_chip(str(nf.get('looked_for', '?')), 'looked')}</div>"
                f"<div class='judgment-body'>{html.escape(str(nf.get('result', '')))}</div>"
                + (f"<div class='judgment-sub'><strong>Slices:</strong> {html.escape(', '.join(map(str, slices)))}</div>" if slices else "")
                + "</li>"
            )
        sections.append(f"<section><h4>Negative Findings ({len(negatives)})</h4><ul class='judgment-list'>{''.join(items)}</ul></section>")
    else:
        sections.append("<section><h4>Negative Findings (0)</h4><div class='muted'>None recorded.</div></section>")

    # Uncertainty
    if uncertainties:
        items = "".join(f"<li>{html.escape(str(u))}</li>" for u in uncertainties)
        sections.append(f"<section><h4>Uncertainty Flags ({len(uncertainties)})</h4><ul class='plain-list'>{items}</ul></section>")
    else:
        sections.append("<section><h4>Uncertainty Flags (0)</h4><div class='muted'>None.</div></section>")

    # Coverage
    if isinstance(coverage, dict):
        sections.append(
            "<section><h4>Coverage Summary</h4>"
            + "<div class='coverage-grid'>"
            + f"<div><div class='label'>Regions Examined</div><div class='value'>{html.escape(', '.join(map(str, coverage.get('regions_examined', []) or [])) or 'None')}</div></div>"
            + f"<div><div class='label'>Regions Remaining</div><div class='value'>{html.escape(', '.join(map(str, coverage.get('regions_remaining', []) or [])) or 'None')}</div></div>"
            + f"<div><div class='label'>Window Levels</div><div class='value'>{html.escape(', '.join(map(str, coverage.get('window_levels_applied', []) or [])) or 'None')}</div></div>"
            + "</div></section>"
        )

    return "".join(sections)


def _render_call_summaries(call_summaries: list[dict[str, Any]]) -> str:
    if not call_summaries:
        return "<div class='muted'>No MCP calls executed this iteration.</div>"

    blocks: list[str] = []
    for c in call_summaries:
        err = c.get("error")
        status = _chip("error", "error") if err else _chip("ok", "ok")
        counts = " ".join(
            [
                _chip(f"bridge_images={c.get('bridge_num_images', 0)}"),
                _chip(f"recovered_images={c.get('recovered_num_images', 0)}"),
            ]
        )
        blocks.append(
            "<details class='call-card'>"
            f"<summary><span class='call-name'>{html.escape(str(c.get('tool_name', '?')))}</span> {status} {counts}</summary>"
            + (f"<div class='error-text'>{html.escape(str(err))}</div>" if err else "")
            + "<div class='call-grid'>"
            + "<div><h5>Arguments</h5>"
            + _render_kv_table(c.get("arguments") or {})
            + "</div>"
            + "<div><h5>Result Shape</h5>"
            + (
                f"<div class='muted'>kind={html.escape(str(c.get('result_kind')))} "
                f"keys={html.escape(', '.join(c.get('result_keys') or []))}</div>"
            )
            + "</div>"
            + "</div>"
            + "<details class='raw-json'><summary>Result Preview</summary>"
            + f"<pre>{_json_html(c.get('result_preview'))}</pre></details>"
            + "</details>"
        )
    return "".join(blocks)


def _render_batch(batch: dict[str, Any]) -> str:
    req_iter = batch.get("request_iteration")
    rev_iter = batch.get("review_iteration")
    rationale = batch.get("rationale") or "No rationale captured."
    images = batch.get("images") or []
    review_count = batch.get("review_messages_image_count")
    recovered_count = batch.get("recovered_image_count", 0)

    title = (
        f"Request Iteration {req_iter} → Review Iteration {rev_iter}"
        if rev_iter is not None
        else f"Request Iteration {req_iter} (not reviewed before run ended)"
    )

    badges = [
        _chip(f"recovered_images={recovered_count}", "count"),
    ]
    if review_count is not None:
        badges.append(_chip(f"llm_message_images={review_count}", "count"))
    if batch.get("suggested_thinking_level"):
        badges.append(_chip(f"thinking={batch['suggested_thinking_level']}", "hint"))
    if batch.get("suggested_media_resolution"):
        badges.append(_chip(f"media={batch['suggested_media_resolution']}", "hint"))
    if batch.get("delivery_warning"):
        badges.append(_chip("delivery mismatch (extracted vs sent)", "warn"))

    img_grid = (
        f"<div class='thumb-grid'>{''.join(_render_image_card(img) for img in images)}</div>"
        if images else "<div class='muted'>No recoverable images in this batch.</div>"
    )

    review_text = batch.get("review_messages_first_user_text")
    review_text_html = (
        f"<details class='raw-json'><summary>Review Iteration Prompt (user text)</summary><pre>{html.escape(str(review_text))}</pre></details>"
        if review_text else ""
    )

    return (
        "<section class='batch-card'>"
        f"<div class='batch-header'><h3>{html.escape(title)}</h3><div class='chip-row'>{''.join(badges)}</div></div>"
        "<div class='panel'>"
        "<h4>Why This Batch Was Requested</h4>"
        f"<p class='rationale'>{html.escape(str(rationale))}</p>"
        "</div>"
        "<div class='panel'>"
        "<h4>MCP Calls and Results</h4>"
        f"{_render_call_summaries(batch.get('call_summaries') or [])}"
        "</div>"
        "<div class='panel'>"
        "<h4>Images Viewed by the Agent (Batch)</h4>"
        f"{img_grid}"
        "</div>"
        "<div class='panel'>"
        f"<h4>Rendered Judgments (Review Iteration {rev_iter if rev_iter is not None else 'N/A'})</h4>"
        f"{review_text_html}"
        f"{_render_findings(batch.get('review_parsed_response'))}"
        "<details class='raw-json'><summary>Raw Parsed Review Response</summary>"
        f"<pre>{_json_html(batch.get('review_parsed_response'))}</pre></details>"
        "</div>"
        "</section>"
    )


def _render_initial_judgment(initial_judgment: dict[str, Any] | None) -> str:
    if not initial_judgment:
        return ""
    parsed = initial_judgment.get("parsed_response")
    if not parsed:
        return ""
    image_count_chip = _chip(
        f"llm_message_images={initial_judgment.get('messages_image_count', 0)}",
        "count",
    )
    return (
        "<section class='batch-card intro-card'>"
        "<div class='batch-header'><h3>Initial Agent State (Iteration 0, before any images)</h3></div>"
        "<div class='panel'>"
        f"<div class='chip-row'>{image_count_chip}</div>"
        f"{_render_findings(parsed)}"
        "<details class='raw-json'><summary>Raw Parsed Iteration 0 Response</summary>"
        f"<pre>{_json_html(parsed)}</pre></details>"
        "</div>"
        "</section>"
    )


def _render_available_series(available_images: Any) -> str:
    if not isinstance(available_images, list) or not available_images:
        return "<div class='muted'>No available_images.json found.</div>"
    rows = []
    for s in available_images:
        if not isinstance(s, dict):
            continue
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(s.get('series_id', '')))}</code></td>"
            f"<td>{html.escape(str(s.get('modality', '')))}</td>"
            f"<td>{html.escape(str(s.get('description', '')))}</td>"
            f"<td>{html.escape(str(s.get('slice_range', '')))}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table class='series-table'>"
        "<thead><tr><th>Series</th><th>Modality</th><th>Description</th><th>Slice Range</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_final_report(final_report: Any) -> str:
    if final_report is None:
        return ""
    report = final_report.get("report") if isinstance(final_report, dict) else None
    summary = ""
    if isinstance(report, dict):
        impression = report.get("impression") or []
        negatives = report.get("negative_findings") or []
        summary = (
            "<div class='coverage-grid'>"
            f"<div><div class='label'>Impression Items</div><div class='value'>{len(impression) if isinstance(impression, list) else 0}</div></div>"
            f"<div><div class='label'>Negative Findings</div><div class='value'>{len(negatives) if isinstance(negatives, list) else 0}</div></div>"
            "</div>"
        )
    return (
        "<section class='batch-card'>"
        "<div class='batch-header'><h3>Final Report</h3></div>"
        f"<div class='panel'>{summary}<details class='raw-json' open><summary>Parsed Final Report JSON</summary><pre>{_json_html(final_report)}</pre></details></div>"
        "</section>"
    )


def _render_html_page(model: dict[str, Any]) -> str:
    config = model.get("config") or {}
    clinical_q = config.get("clinical_question", "")
    total_batches = len(model.get("batches") or [])
    total_images = sum(len(b.get("images") or []) for b in (model.get("batches") or []))
    mismatch_batches = sum(1 for b in (model.get("batches") or []) if b.get("delivery_warning"))

    batch_nav = []
    for b in model.get("batches") or []:
        req = b.get("request_iteration")
        rev = b.get("review_iteration")
        nimg = len(b.get("images") or [])
        txt = f"iter {req} -> {rev if rev is not None else 'end'} ({nimg} images)"
        cls = "warn" if b.get("delivery_warning") else ""
        batch_nav.append(f"<a class='toc-link {cls}' href='#batch-{req}'>{html.escape(txt)}</a>")

    batch_sections = []
    for b in model.get("batches") or []:
        req = b.get("request_iteration")
        batch_sections.append(f"<a id='batch-{req}'></a>" + _render_batch(b))

    styles = """
    :root {
      --bg: #f4efe7;
      --paper: #fffaf1;
      --ink: #1f1b16;
      --muted: #6d6458;
      --line: #d9cfbe;
      --accent: #0f6c6d;
      --accent-2: #a73f2d;
      --warn: #b85b00;
      --shadow: 0 14px 40px rgba(31, 27, 22, 0.08);
      --radius: 16px;
      --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
      --sans: "Avenir Next", "Segoe UI Variable", "Trebuchet MS", sans-serif;
      --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 0%, rgba(15,108,109,0.10), transparent 45%),
        radial-gradient(circle at 95% 10%, rgba(167,63,45,0.10), transparent 40%),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
    }
    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: 100vh;
      gap: 20px;
      padding: 20px;
    }
    .sidebar {
      position: sticky;
      top: 20px;
      align-self: start;
      background: rgba(255,250,241,0.92);
      backdrop-filter: blur(6px);
      border: 1px solid var(--line);
      border-radius: calc(var(--radius) + 4px);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .side-head {
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(15,108,109,0.08), rgba(167,63,45,0.06));
    }
    .side-head h1 {
      margin: 0 0 8px;
      font-size: 1.08rem;
      font-family: var(--serif);
      line-height: 1.2;
    }
    .side-meta { color: var(--muted); font-size: 0.86rem; line-height: 1.45; }
    .toc {
      padding: 14px;
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 220px);
      overflow: auto;
    }
    .toc-link {
      text-decoration: none;
      color: var(--ink);
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 0.88rem;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }
    .toc-link:hover { transform: translateY(-1px); border-color: var(--accent); background: #fffdf7; }
    .toc-link.warn { border-left: 4px solid var(--warn); }
    .main { display: grid; gap: 18px; }
    .hero, .batch-card {
      background: rgba(255,250,241,0.95);
      border: 1px solid var(--line);
      border-radius: calc(var(--radius) + 2px);
      box-shadow: var(--shadow);
      overflow: hidden;
      animation: fadeIn 260ms ease;
    }
    .hero { padding: 18px; }
    .hero h2 {
      margin: 0 0 8px;
      font-family: var(--serif);
      font-size: 1.45rem;
      line-height: 1.15;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0,1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: linear-gradient(180deg, #fffdf7, #fbf5ea);
    }
    .metric .k { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: .04em; }
    .metric .v { margin-top: 3px; font-size: 1.05rem; font-weight: 700; }
    .batch-header {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(15,108,109,0.07), rgba(167,63,45,0.04));
    }
    .batch-header h3 {
      margin: 0 0 8px;
      font-size: 1.02rem;
      font-family: var(--serif);
    }
    .chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fffdf7;
      font-size: 0.75rem;
      color: var(--ink);
      white-space: nowrap;
    }
    .chip.ok { border-color: rgba(15,108,109,0.4); color: var(--accent); }
    .chip.error, .chip.warn { border-color: rgba(184,91,0,0.35); color: var(--warn); background: #fff8ee; }
    .chip.conf { border-color: rgba(15,108,109,0.25); }
    .chip.etype { border-color: rgba(167,63,45,0.25); }
    .panel {
      padding: 14px 16px;
      border-top: 1px solid rgba(217,207,190,0.55);
    }
    .panel:first-of-type { border-top: 0; }
    .panel h4 {
      margin: 0 0 10px;
      font-size: 0.92rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .panel h5 { margin: 6px 0 8px; font-size: 0.86rem; color: var(--muted); }
    .rationale {
      margin: 0;
      line-height: 1.5;
      font-size: 0.96rem;
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      border-left: 4px solid var(--accent);
    }
    .thumb-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
    }
    .thumb-card {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: #fffdf7;
    }
    .thumb-card a { display: block; background: #e9e2d6; }
    .thumb-card img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background:
        linear-gradient(45deg, #ebe2d1 25%, #f4ecdf 25%, #f4ecdf 50%, #ebe2d1 50%, #ebe2d1 75%, #f4ecdf 75%, #f4ecdf 100%);
      background-size: 18px 18px;
    }
    .thumb-card figcaption { padding: 9px 10px; }
    .thumb-title { font-weight: 700; font-size: 0.84rem; margin-bottom: 4px; }
    .meta-line { font-size: 0.75rem; color: var(--muted); line-height: 1.35; }
    .meta-line code, .kv-table code { font-family: var(--mono); font-size: 0.74rem; }
    .call-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fffdf8;
      margin-bottom: 10px;
      overflow: hidden;
    }
    .call-card > summary {
      cursor: pointer;
      list-style: none;
      padding: 10px 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      border-bottom: 1px solid rgba(217,207,190,0.55);
    }
    .call-card > summary::-webkit-details-marker { display: none; }
    .call-name { font-weight: 700; color: var(--accent); }
    .call-grid {
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 12px;
      padding: 0 12px 12px;
    }
    .kv-table { width: 100%; border-collapse: collapse; }
    .kv-table th, .kv-table td {
      vertical-align: top;
      text-align: left;
      border-top: 1px solid rgba(217,207,190,0.4);
      padding: 6px 4px;
      font-size: 0.8rem;
    }
    .kv-table th { color: var(--muted); width: 32%; }
    .error-text {
      color: var(--warn);
      background: #fff7ec;
      border: 1px solid rgba(184,91,0,0.2);
      border-radius: 10px;
      margin: 10px 12px 0;
      padding: 8px 10px;
      font-size: 0.83rem;
    }
    .judgment-list, .plain-list {
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 8px;
    }
    .judgment-list { list-style: none; padding-left: 0; }
    .judgment-item {
      border: 1px solid var(--line);
      background: #fffdf8;
      border-radius: 10px;
      padding: 10px;
    }
    .judgment-head { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
    .judgment-body { font-size: 0.88rem; line-height: 1.4; }
    .judgment-sub { margin-top: 6px; font-size: 0.78rem; color: var(--muted); line-height: 1.35; }
    .coverage-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0,1fr));
      gap: 10px;
    }
    .coverage-grid > div {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fffdf7;
      padding: 10px;
    }
    .label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
    .value { font-size: 0.84rem; line-height: 1.35; }
    .raw-json {
      margin-top: 10px;
      border: 1px dashed var(--line);
      border-radius: 10px;
      background: rgba(255,255,255,0.5);
      overflow: hidden;
    }
    .raw-json > summary {
      cursor: pointer;
      padding: 8px 10px;
      font-size: 0.8rem;
      color: var(--muted);
    }
    .raw-json pre {
      margin: 0;
      padding: 12px;
      border-top: 1px solid rgba(217,207,190,0.5);
      overflow: auto;
      font-family: var(--mono);
      font-size: 0.75rem;
      line-height: 1.45;
      background: #fbf7ee;
    }
    .series-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
      background: #fffdf7;
      border-radius: 12px;
      overflow: hidden;
    }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 12px; }
    .series-table th, .series-table td { border-top: 1px solid rgba(217,207,190,0.4); padding: 8px 10px; text-align: left; white-space: nowrap; }
    .series-table thead th { background: #f6f0e3; color: var(--muted); border-top: 0; position: sticky; top: 0; }
    .muted { color: var(--muted); font-size: 0.84rem; }
    code { font-family: var(--mono); }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(2px); } to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) {
      * { animation: none !important; transition: none !important; }
    }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .metrics { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .coverage-grid { grid-template-columns: 1fr; }
      .call-grid { grid-template-columns: 1fr; }
    }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Orchestrator Image History UI — {html.escape(str(model.get("run_name", "")))}</title>
  <style>{styles}</style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="side-head">
        <h1>Agent Image History</h1>
        <div class="side-meta">
          <div><strong>Run:</strong> <code>{html.escape(str(model.get("run_name", "")))}</code></div>
          <div><strong>Iterations:</strong> {html.escape(str(model.get("iteration_count", 0)))}</div>
          <div><strong>Generated:</strong> {html.escape(str(model.get("generated_at", "")))}</div>
        </div>
      </div>
      <div class="toc">
        <a class="toc-link" href="#overview">Overview</a>
        <a class="toc-link" href="#initial">Initial Judgment</a>
        <a class="toc-link" href="#series">Available Series</a>
        {''.join(batch_nav) or '<div class="muted">No request batches found.</div>'}
        <a class="toc-link" href="#final-report">Final Report</a>
      </div>
    </aside>
    <main class="main">
      <section class="hero" id="overview">
        <h2>Orchestrator Review Timeline</h2>
        <p>{html.escape(str(clinical_q or "No clinical question found in config.json"))}</p>
        <div class="metrics">
          <div class="metric"><div class="k">Batches</div><div class="v">{total_batches}</div></div>
          <div class="metric"><div class="k">Recovered Images</div><div class="v">{total_images}</div></div>
          <div class="metric"><div class="k">Delivery Mismatches</div><div class="v">{mismatch_batches}</div></div>
          <div class="metric"><div class="k">Model</div><div class="v">{html.escape(str(config.get("model", "?")))}</div></div>
        </div>
        <details class="raw-json" style="margin-top:12px">
          <summary>Run Config JSON</summary>
          <pre>{_json_html(config)}</pre>
        </details>
      </section>

      <a id="initial"></a>
      {_render_initial_judgment(model.get("initial_judgment"))}

      <section class="batch-card" id="series">
        <div class="batch-header"><h3>Available Series at Run Start</h3></div>
        <div class="panel">
          {_render_available_series(model.get("available_images"))}
        </div>
      </section>

      {''.join(batch_sections)}

      <a id="final-report"></a>
      {_render_final_report(model.get("final_report"))}
    </main>
  </div>
</body>
</html>"""


def build_history_ui(run_dir: Path, output_html: Path, assets_root: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if assets_root is None:
        assets_root = run_dir / "_ui_assets"

    model = _build_run_model(run_dir, output_html, assets_root)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_render_html_page(model))
    return output_html


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML UI for orchestrator debug run image history."
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Path to debug run directory (default: latest under ./debug_runs)",
    )
    parser.add_argument(
        "--debug-runs-root",
        default="debug_runs",
        help="Root directory containing debug runs (default: ./debug_runs)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path (default: <run>/agent_image_history_ui.html)",
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Directory for recovered image assets when manifest images are unavailable (default: <run>/_ui_assets)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.run:
        run_dir = Path(args.run).expanduser().resolve()
    else:
        run_dir = _latest_debug_run(Path(args.debug_runs_root).expanduser().resolve())

    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Debug run directory not found: {run_dir}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (run_dir / "agent_image_history_ui.html")
    )
    assets_dir = (
        Path(args.assets_dir).expanduser().resolve()
        if args.assets_dir
        else (run_dir / "_ui_assets")
    )

    built = build_history_ui(run_dir, out_path, assets_dir)
    print(f"[history-ui] Wrote {built}")


if __name__ == "__main__":
    main()
