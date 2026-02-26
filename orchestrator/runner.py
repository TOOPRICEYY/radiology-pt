"""Main orchestrator loop: iterate, call LLM, parse, execute MCP, check convergence."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from typing import Any

from openai import OpenAI

from dicom_mcp.dicom_ops import DicomRepository
from orchestrator.config import OrchestratorConfig
from orchestrator.context import (
    build_available_images,
    build_current_images,
    build_iteration_state,
    build_observation_ledger,
    render_prompt,
)
from orchestrator.annotate import annotate_images
from orchestrator.debug import DebugSession
from orchestrator.mcp_bridge import build_tool_descriptions, execute_mcp_calls
from orchestrator.monitor_ui import LiveOrchestratorMonitor, NullOrchestratorMonitor
from orchestrator.terminal import StreamingDisplay
from orchestrator.models import (
    AgentResponse,
    IterationState,
    LedgerEntry,
    NegativeLedgerEntry,
    parse_agent_response,
)


def _image_set_hash(images: list[str]) -> str:
    """Hash of the base64 image set for concordance tracking."""
    h = hashlib.sha256()
    for img in sorted(images):
        h.update(img[:64].encode())  # hash prefix for speed
    return h.hexdigest()[:12]


def _check_convergence(
    response: AgentResponse,
    ledger: list[LedgerEntry],
    negative_ledger: list[NegativeLedgerEntry],
    config: OrchestratorConfig,
) -> tuple[bool, list[str]]:
    """Check whether the orchestrator's convergence criteria are met.

    Returns (converged: bool, reasons: list of unmet criteria).
    """
    reasons = []

    # 1. Agent must have set next_request to null
    if response.next_request is not None:
        reasons.append("Agent has not set next_request to null.")
        return False, reasons

    # 2. Every moderate/high finding needs >= 1 confirmatory
    finding_ids_needing_confirm = set()
    for f in response.findings:
        if f.confidence in ("moderate", "high"):
            finding_ids_needing_confirm.add(f.finding_id)

    # Also check the full ledger for moderate/high findings
    for entry in ledger:
        if entry.finding.confidence in ("moderate", "high"):
            finding_ids_needing_confirm.add(entry.finding.finding_id)

    # Count confirmatory evidence per finding_id across all ledger entries
    confirmatory_counts: dict[str, int] = {}
    for entry in ledger:
        if entry.finding.evidence_type == "confirmatory":
            fid = entry.finding.finding_id
            confirmatory_counts[fid] = confirmatory_counts.get(fid, 0) + 1
    # Also count from current response
    for f in response.findings:
        if f.evidence_type == "confirmatory":
            confirmatory_counts[f.finding_id] = confirmatory_counts.get(f.finding_id, 0) + 1

    for fid in finding_ids_needing_confirm:
        if confirmatory_counts.get(fid, 0) < config.convergence_min_confirmatory:
            reasons.append(
                f"Finding '{fid}' (moderate/high confidence) lacks confirmatory evidence."
            )

    # 3. regions_remaining should be empty
    if response.coverage_summary.regions_remaining:
        regions = ", ".join(response.coverage_summary.regions_remaining)
        reasons.append(f"Regions still remaining: {regions}")

    # 4. At least N negative findings total
    total_negatives = len(negative_ledger) + len(response.negative_findings)
    if total_negatives < config.convergence_min_negative_findings:
        reasons.append(
            f"Only {total_negatives} negative findings total "
            f"(need {config.convergence_min_negative_findings})."
        )

    return len(reasons) == 0, reasons


def _build_messages(
    rendered_prompt: str,
    iteration: int,
    current_images_b64: list[str],
    clinical_question: str,
    is_final_report: bool = False,
    supervisor_followups: list[str] | None = None,
) -> list[dict]:
    """Build the OpenAI-format messages list for one LLM call."""
    messages: list[dict] = [
        {"role": "system", "content": rendered_prompt},
    ]

    if is_final_report:
        final_text = (
            "All convergence criteria are met. Produce the final radiology report "
            "using the REPORT FORMAT specified in the prompt. "
            "Include all findings from the observation ledger."
        )
        if supervisor_followups:
            followup_block = "\n\n".join(
                f"Supervisor follow-up (must address): {q}" for q in supervisor_followups
            )
            final_text = f"{final_text}\n\n{followup_block}"
        messages.append({"role": "user", "content": final_text})
        return messages

    # User message: text + optional images
    content_parts: list[dict] = []

    if iteration == 0:
        content_parts.append({
            "type": "text",
            "text": (
                f"Iteration {iteration}. Clinical question: {clinical_question}\n\n"
                "No images from previous iterations. Begin your survey phase — "
                "examine the available series and request initial image extractions."
            ),
        })
    else:
        content_parts.append({
            "type": "text",
            "text": f"Iteration {iteration}. Analyze the provided images and continue your assessment.",
        })

    if supervisor_followups:
        content_parts.append(
            {
                "type": "text",
                "text": "\n\n".join(
                    f"Supervisor follow-up query (must address): {q}"
                    for q in supervisor_followups
                ),
            }
        )

    # Add base64 images as vision content blocks
    for b64 in current_images_b64:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    messages.append({"role": "user", "content": content_parts})
    return messages


def _call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict],
    display: "StreamingDisplay | None" = None,
) -> str:
    """Call the LLM and return the raw content string.

    If a StreamingDisplay is provided, streams tokens through it in real-time.
    """
    from orchestrator.terminal import StreamingDisplay

    if display is not None:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                display.feed(delta.content)
        return display.finish()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
        )
        return response.choices[0].message.content or ""


def _split_interventions(
    interventions: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Partition monitor interventions into directives and follow-up user queries."""
    directives: list[str] = []
    followups: list[str] = []
    for item in interventions or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        kind = str(item.get("kind", "directive")).strip().lower()
        if kind == "question":
            followups.append(text)
        else:
            directives.append(text)
    return directives, followups


def run(config: OrchestratorConfig) -> dict:
    """Execute the full orchestrator loop.

    Returns a dict with the final report, ledger history, and metadata.
    """
    monitor: LiveOrchestratorMonitor | NullOrchestratorMonitor
    if getattr(config, "monitor_ui", False):
        try:
            monitor = LiveOrchestratorMonitor()
            monitor.start(config)
        except Exception as e:
            print(f"[monitor-ui] Failed to start live UI: {type(e).__name__}: {e}")
            monitor = NullOrchestratorMonitor()
    else:
        monitor = NullOrchestratorMonitor()

    monitor.record_status("running")

    try:
        # ── Init ─────────────────────────────────────────────────────────────
        monitor.set_phase("initializing")
        dbg = DebugSession(config)

        repo = DicomRepository(config.dicom_root)
        available_images = build_available_images(repo)
        print(f"[orchestrator] Found {len(available_images)} series")
        for s in available_images:
            print(f"  - {s['series_id']}: {s['modality']} | {s['description']} | slices {s['slice_range']}")
        dbg.dump_available_images(available_images)
        monitor.record_available_images(available_images)

        prompt_template = config.resolve_prompt_path().read_text()
        tool_descriptions = build_tool_descriptions()

        client = OpenAI(base_url=config.base_url, api_key=config.api_key)

        ledger: list[LedgerEntry] = []
        negative_ledger: list[NegativeLedgerEntry] = []
        iteration_state = IterationState(
            iteration=0,
            max_iterations=config.max_iterations,
            thinking_level=config.default_thinking_level,
            media_resolution=config.default_media_resolution,
        )

        current_images_metadata: list[dict] = []  # metadata for prompt context
        current_images_b64: list[str] = []  # base64 PNGs for vision
        all_responses: list[dict] = []  # history of all agent responses

        # ── Main loop ────────────────────────────────────────────────────────
        for i in range(config.max_iterations):
            monitor.wait_if_paused("iteration_start", i)
            iteration_state.iteration = i
            print(f"\n[orchestrator] === Iteration {i} ===")
            dbg.dump_iteration_start(i)

            interventions = monitor.consume_interventions(i)
            supervisor_followups: list[str] = []
            if interventions:
                directives, supervisor_followups = _split_interventions(interventions)
                if directives:
                    iteration_state.directives = [*iteration_state.directives, *directives]
                monitor.record_applied_interventions(i, interventions)

            # Build context
            obs_ledger = build_observation_ledger(ledger, negative_ledger)
            cur_images_ctx = build_current_images(current_images_metadata)
            iter_state_ctx = build_iteration_state(iteration_state)
            dbg.dump_iteration_state(i, iter_state_ctx)
            monitor.record_iteration_state(i, iter_state_ctx)

            rendered = render_prompt(
                template=prompt_template,
                available_images=available_images,
                ledger=obs_ledger,
                current_images=cur_images_ctx,
                iteration_state=iter_state_ctx,
                mcp_tool_descriptions=tool_descriptions,
                user_prompt=config.clinical_question,
            )
            dbg.dump_rendered_prompt(i, rendered)

            # Build messages and call LLM
            messages = _build_messages(
                rendered,
                i,
                current_images_b64,
                config.clinical_question,
                supervisor_followups=supervisor_followups,
            )
            dbg.dump_messages(i, messages)
            monitor.record_messages(i, messages)

            monitor.wait_if_paused("before_llm", i)
            print(f"[orchestrator] Calling LLM ({config.model})...")
            display = StreamingDisplay(
                label=f"Iteration {i} — {config.model}",
                live=config.live,
                expand=config.expand_llm,
            ) if config.live else None
            raw_output = _call_llm(client, config.model, messages, display=display)
            if not config.live:
                print(f"[orchestrator] Got response ({len(raw_output)} chars)")
            dbg.dump_raw_llm_output(i, raw_output)

            # Parse response
            try:
                agent_response = parse_agent_response(raw_output)
            except Exception as e:
                print(f"[orchestrator] Failed to parse agent response: {e}")
                print(f"[orchestrator] Raw output:\n{raw_output[:500]}")
                dbg.dump_parse_error(i, e, raw_output)
                monitor.record_parse_error(i, f"{type(e).__name__}: {e}")
                # Inject directive to produce valid JSON next iteration
                iteration_state.directives = [
                    "Your previous response was not valid JSON. "
                    "You MUST respond with ONLY the JSON structure specified in the prompt."
                ]
                current_images_metadata = []
                current_images_b64 = []
                continue

            all_responses.append(agent_response.to_dict())
            dbg.dump_parsed_response(i, agent_response.to_dict())
            monitor.record_parsed_response(i, agent_response.to_dict())

            # Render bounding-box annotations onto current images
            has_boxes = any(f.bounding_boxes for f in agent_response.findings)
            if has_boxes and current_images_b64:
                annotated_b64 = annotate_images(
                    current_images_b64, current_images_metadata, agent_response
                )
                dbg.dump_annotated_images(i, annotated_b64, agent_response)
                monitor.record_annotated_images(i, annotated_b64, agent_response.to_dict())
                n_boxes = sum(len(f.bounding_boxes) for f in agent_response.findings)
                print(f"[orchestrator] Drew {n_boxes} bounding box(es) on {len(current_images_b64)} image(s)")

            # Record findings in ledger
            img_hash = _image_set_hash(current_images_b64) if current_images_b64 else ""
            for finding in agent_response.findings:
                ledger.append(LedgerEntry(
                    finding=finding,
                    iteration=i,
                    image_set_hash=img_hash,
                ))
            for nf in agent_response.negative_findings:
                negative_ledger.append(NegativeLedgerEntry(
                    negative_finding=nf,
                    iteration=i,
                ))

            print(f"[orchestrator] Findings: {len(agent_response.findings)}, "
                  f"Negative: {len(agent_response.negative_findings)}")

            # Convergence check
            converged, unmet = _check_convergence(
                agent_response, ledger, negative_ledger, config
            )
            dbg.dump_convergence(
                i, converged, unmet,
                next_request_null=agent_response.next_request is None,
            )
            monitor.record_convergence(
                i,
                converged=converged,
                reasons=unmet,
                next_request_null=agent_response.next_request is None,
            )
            dbg.dump_ledger(
                i,
                [e.to_dict() for e in ledger],
                [e.to_dict() for e in negative_ledger],
            )

            if agent_response.next_request is None:
                if converged:
                    print("[orchestrator] Convergence criteria met!")
                    break
                else:
                    # Agent said done but criteria not met → force continue
                    print("[orchestrator] Agent set next_request=null but convergence not met:")
                    for reason in unmet:
                        print(f"  - {reason}")
                    iteration_state.directives = [
                        "The orchestrator has determined that convergence criteria are NOT met. "
                        "You must continue exploration. Unmet criteria:",
                        *unmet,
                        "Set next_request with appropriate MCP calls to address these gaps.",
                    ]
                    current_images_metadata = []
                    current_images_b64 = []
                    continue

            # Execute MCP calls from agent's next_request
            if agent_response.next_request is not None:
                mcp_calls = agent_response.next_request.mcp_calls
                mcp_results: list[dict] = []
                if mcp_calls:
                    monitor.wait_if_paused("before_mcp", i)
                    print(f"[orchestrator] Executing {len(mcp_calls)} MCP call(s)...")
                    mcp_results = execute_mcp_calls(repo, mcp_calls)
                    dbg.dump_mcp_calls(i, mcp_calls, mcp_results)
                else:
                    dbg.dump_mcp_calls(i, mcp_calls, mcp_results)

                monitor.record_mcp_execution(
                    i,
                    rationale=agent_response.next_request.rationale,
                    suggested_thinking_level=agent_response.next_request.suggested_thinking_level,
                    suggested_media_resolution=agent_response.next_request.suggested_media_resolution,
                    mcp_calls=mcp_calls,
                    mcp_results=mcp_results,
                )

                # Collect images and metadata for next iteration
                current_images_b64 = []
                current_images_metadata = []
                for res in mcp_results:
                    current_images_b64.extend(res.get("images", []))
                    current_images_metadata.append({
                        "tool_name": res["tool_name"],
                        "arguments": res["arguments"],
                        "error": res["error"],
                        "num_images": len(res.get("images", [])),
                        "result_summary": _summarize_result(res.get("result")),
                    })
                    if res["error"]:
                        print(f"  [error] {res['tool_name']}: {res['error']}")
                    else:
                        print(f"  [ok] {res['tool_name']}: {len(res.get('images', []))} images")
                if not mcp_calls:
                    current_images_metadata = []
                    current_images_b64 = []

                # Update iteration state from agent suggestions
                if agent_response.next_request.suggested_thinking_level:
                    iteration_state.thinking_level = agent_response.next_request.suggested_thinking_level
                if agent_response.next_request.suggested_media_resolution:
                    iteration_state.media_resolution = agent_response.next_request.suggested_media_resolution

                monitor.wait_if_paused("after_mcp", i)
            else:
                current_images_metadata = []
                current_images_b64 = []

            # Clear directives for next iteration
            iteration_state.directives = []

        # ── Final report ─────────────────────────────────────────────────────
        last_final_report_iteration = -1

        def _generate_final_report(
            supervisor_followups: list[str] | None = None,
            *,
            phase: str,
            display_label: str,
        ) -> dict[str, Any]:
            nonlocal last_final_report_iteration
            final_iteration_local = iteration_state.iteration + 1
            obs_ledger = build_observation_ledger(ledger, negative_ledger)
            iter_state_ctx = build_iteration_state(IterationState(
                iteration=final_iteration_local,
                max_iterations=iteration_state.max_iterations,
                thinking_level="high",
                media_resolution="low",
                directives=["Produce the final radiology report using the REPORT FORMAT."],
            ))

            rendered = render_prompt(
                template=prompt_template,
                available_images=available_images,
                ledger=obs_ledger,
                current_images=[],
                iteration_state=iter_state_ctx,
                mcp_tool_descriptions=tool_descriptions,
                user_prompt=config.clinical_question,
            )

            messages = _build_messages(
                rendered,
                -1,
                [],
                config.clinical_question,
                is_final_report=True,
                supervisor_followups=supervisor_followups or [],
            )
            dbg.dump_messages(-1, messages)
            monitor.record_messages(-1, messages)
            monitor.wait_if_paused(phase, final_iteration_local)

            report_display = StreamingDisplay(
                label=f"{display_label} — {config.model}",
                live=config.live,
                expand=config.expand_llm,
            ) if config.live else None
            raw_report = _call_llm(client, config.model, messages, display=report_display)
            dbg.dump_raw_llm_output(-1, raw_report)

            try:
                report_response = parse_agent_response(raw_report)
                generated = report_response.to_dict()
            except Exception:
                generated = {"raw_report": raw_report}

            dbg.dump_final_report(raw_report, generated)
            monitor.record_final_report(generated)
            last_final_report_iteration = iteration_state.iteration
            return generated

        def _run_followup_reconvergence_cycle(
            seed_interventions: list[dict[str, Any]],
        ) -> bool:
            """Resume the normal agent/MCP loop after a post-final follow-up request."""
            nonlocal current_images_b64, current_images_metadata

            start_iteration = iteration_state.iteration + 1
            extra_budget = max(1, config.max_iterations)
            end_iteration = start_iteration + extra_budget
            iteration_state.max_iterations = max(iteration_state.max_iterations, end_iteration)

            # Force fresh exploration rather than only revising the final report.
            current_images_metadata = []
            current_images_b64 = []
            iteration_state.directives = []

            resume_directive = (
                "Supervisor follow-up requires additional investigation. Resume the normal "
                "iterative search workflow, request MCP image extractions as needed, and only "
                "set next_request to null after convergence criteria are satisfied again."
            )
            seed_directives, seed_followups = _split_interventions(seed_interventions)
            first_iteration_followups = list(seed_followups)
            first_iteration_seed_directives = [resume_directive, *seed_directives]

            print(
                "[orchestrator] Re-entering normal loop for follow-up investigation "
                f"(iterations {start_iteration}-{end_iteration - 1})..."
            )

            for i in range(start_iteration, end_iteration):
                monitor.wait_if_paused("iteration_start", i)
                if monitor.is_stop_requested():
                    return False

                iteration_state.iteration = i
                print(f"\n[orchestrator] === Follow-up Iteration {i} ===")
                dbg.dump_iteration_start(i)

                supervisor_followups: list[str] = []
                if first_iteration_seed_directives:
                    iteration_state.directives = [*iteration_state.directives, *first_iteration_seed_directives]
                    first_iteration_seed_directives = []
                if first_iteration_followups:
                    supervisor_followups.extend(first_iteration_followups)
                    first_iteration_followups = []

                interventions = monitor.consume_interventions(i)
                if interventions:
                    directives, more_followups = _split_interventions(interventions)
                    if directives:
                        iteration_state.directives = [*iteration_state.directives, *directives]
                    if more_followups:
                        supervisor_followups.extend(more_followups)
                    monitor.record_applied_interventions(i, interventions)

                # Build context
                obs_ledger = build_observation_ledger(ledger, negative_ledger)
                cur_images_ctx = build_current_images(current_images_metadata)
                iter_state_ctx = build_iteration_state(iteration_state)
                dbg.dump_iteration_state(i, iter_state_ctx)
                monitor.record_iteration_state(i, iter_state_ctx)

                rendered = render_prompt(
                    template=prompt_template,
                    available_images=available_images,
                    ledger=obs_ledger,
                    current_images=cur_images_ctx,
                    iteration_state=iter_state_ctx,
                    mcp_tool_descriptions=tool_descriptions,
                    user_prompt=config.clinical_question,
                )
                dbg.dump_rendered_prompt(i, rendered)

                messages = _build_messages(
                    rendered,
                    i,
                    current_images_b64,
                    config.clinical_question,
                    supervisor_followups=supervisor_followups,
                )
                dbg.dump_messages(i, messages)
                monitor.record_messages(i, messages)

                monitor.wait_if_paused("before_llm", i)
                if monitor.is_stop_requested():
                    return False

                print(f"[orchestrator] Calling LLM ({config.model})...")
                display = StreamingDisplay(
                    label=f"Follow-up Iteration {i} — {config.model}",
                    live=config.live,
                    expand=config.expand_llm,
                ) if config.live else None
                raw_output = _call_llm(client, config.model, messages, display=display)
                if not config.live:
                    print(f"[orchestrator] Got response ({len(raw_output)} chars)")
                dbg.dump_raw_llm_output(i, raw_output)

                try:
                    agent_response = parse_agent_response(raw_output)
                except Exception as e:
                    print(f"[orchestrator] Failed to parse agent response: {e}")
                    print(f"[orchestrator] Raw output:\n{raw_output[:500]}")
                    dbg.dump_parse_error(i, e, raw_output)
                    monitor.record_parse_error(i, f"{type(e).__name__}: {e}")
                    iteration_state.directives = [
                        "Your previous response was not valid JSON. "
                        "You MUST respond with ONLY the JSON structure specified in the prompt."
                    ]
                    current_images_metadata = []
                    current_images_b64 = []
                    continue

                all_responses.append(agent_response.to_dict())
                dbg.dump_parsed_response(i, agent_response.to_dict())
                monitor.record_parsed_response(i, agent_response.to_dict())

                img_hash = _image_set_hash(current_images_b64) if current_images_b64 else ""
                for finding in agent_response.findings:
                    ledger.append(LedgerEntry(
                        finding=finding,
                        iteration=i,
                        image_set_hash=img_hash,
                    ))
                for nf in agent_response.negative_findings:
                    negative_ledger.append(NegativeLedgerEntry(
                        negative_finding=nf,
                        iteration=i,
                    ))

                print(
                    f"[orchestrator] Findings: {len(agent_response.findings)}, "
                    f"Negative: {len(agent_response.negative_findings)}"
                )

                converged, unmet = _check_convergence(
                    agent_response, ledger, negative_ledger, config
                )
                dbg.dump_convergence(
                    i, converged, unmet,
                    next_request_null=agent_response.next_request is None,
                )
                monitor.record_convergence(
                    i,
                    converged=converged,
                    reasons=unmet,
                    next_request_null=agent_response.next_request is None,
                )
                dbg.dump_ledger(
                    i,
                    [e.to_dict() for e in ledger],
                    [e.to_dict() for e in negative_ledger],
                )

                if agent_response.next_request is None:
                    if converged:
                        print("[orchestrator] Follow-up reconvergence criteria met!")
                        iteration_state.directives = []
                        return True
                    print("[orchestrator] Agent set next_request=null but convergence not met:")
                    for reason in unmet:
                        print(f"  - {reason}")
                    iteration_state.directives = [
                        "The orchestrator has determined that convergence criteria are NOT met. "
                        "You must continue exploration. Unmet criteria:",
                        *unmet,
                        "Set next_request with appropriate MCP calls to address these gaps.",
                    ]
                    current_images_metadata = []
                    current_images_b64 = []
                    continue

                if agent_response.next_request is not None:
                    mcp_calls = agent_response.next_request.mcp_calls
                    mcp_results: list[dict] = []
                    if mcp_calls:
                        monitor.wait_if_paused("before_mcp", i)
                        if monitor.is_stop_requested():
                            return False
                        print(f"[orchestrator] Executing {len(mcp_calls)} MCP call(s)...")
                        mcp_results = execute_mcp_calls(repo, mcp_calls)
                        dbg.dump_mcp_calls(i, mcp_calls, mcp_results)
                    else:
                        dbg.dump_mcp_calls(i, mcp_calls, mcp_results)

                    monitor.record_mcp_execution(
                        i,
                        rationale=agent_response.next_request.rationale,
                        suggested_thinking_level=agent_response.next_request.suggested_thinking_level,
                        suggested_media_resolution=agent_response.next_request.suggested_media_resolution,
                        mcp_calls=mcp_calls,
                        mcp_results=mcp_results,
                    )

                    current_images_b64 = []
                    current_images_metadata = []
                    for res in mcp_results:
                        current_images_b64.extend(res.get("images", []))
                        current_images_metadata.append({
                            "tool_name": res["tool_name"],
                            "arguments": res["arguments"],
                            "error": res["error"],
                            "num_images": len(res.get("images", [])),
                            "result_summary": _summarize_result(res.get("result")),
                        })
                        if res["error"]:
                            print(f"  [error] {res['tool_name']}: {res['error']}")
                        else:
                            print(f"  [ok] {res['tool_name']}: {len(res.get('images', []))} images")
                    if not mcp_calls:
                        current_images_metadata = []
                        current_images_b64 = []

                    if agent_response.next_request.suggested_thinking_level:
                        iteration_state.thinking_level = agent_response.next_request.suggested_thinking_level
                    if agent_response.next_request.suggested_media_resolution:
                        iteration_state.media_resolution = agent_response.next_request.suggested_media_resolution

                    monitor.wait_if_paused("after_mcp", i)
                    if monitor.is_stop_requested():
                        return False
                else:
                    current_images_metadata = []
                    current_images_b64 = []

                iteration_state.directives = []

            print(
                "[orchestrator] Follow-up reconvergence loop exhausted its additional "
                f"budget ({extra_budget} iterations) before meeting convergence criteria."
            )
            return False

        print("\n[orchestrator] === Generating final report ===")
        initial_final_iteration = iteration_state.iteration + 1
        monitor.wait_if_paused("before_final_report", initial_final_iteration)
        final_interventions = monitor.consume_interventions(initial_final_iteration)
        final_supervisor_followups: list[str] = []
        if final_interventions:
            directives, final_supervisor_followups = _split_interventions(final_interventions)
            if directives:
                final_supervisor_followups = [
                    *final_supervisor_followups,
                    *(f"Supervisor directive: {d}" for d in directives),
                ]
            monitor.record_applied_interventions(initial_final_iteration, final_interventions)

        final_report = _generate_final_report(
            final_supervisor_followups,
            phase="final_report_llm",
            display_label="Final Report",
        )

        if getattr(config, "monitor_ui", False):
            print(
                "[orchestrator] Final report ready. Live monitor remains active for follow-up "
                "queries/guidance. Press Ctrl+C to exit and print the latest JSON output."
            )
            monitor.record_status(
                "awaiting_followup",
                "Final report ready. Queue follow-up queries or guidance in the monitor UI.",
            )
            monitor.set_phase("awaiting_followup", iteration_state.iteration + 1)

            while True:
                try:
                    followup_resume_iteration = iteration_state.iteration + 1
                    monitor.wait_if_paused("awaiting_followup", followup_resume_iteration)
                    if monitor.is_stop_requested():
                        print(
                            "[orchestrator] Stop requested from live monitor UI. "
                            "Returning the latest final report."
                        )
                        break
                    followup_interventions = monitor.consume_interventions(followup_resume_iteration)
                    if not followup_interventions:
                        time.sleep(0.25)
                        continue

                    meaningful_interventions = [
                        item for item in followup_interventions
                        if isinstance(item, dict) and str(item.get("text", "")).strip()
                    ]
                    monitor.record_applied_interventions(
                        followup_resume_iteration, followup_interventions
                    )
                    if not meaningful_interventions:
                        continue

                    print(
                        f"[orchestrator] Applying {len(followup_interventions)} follow-up "
                        "intervention(s): re-entering the normal loop to continue search "
                        "and reconverge..."
                    )
                    monitor.record_status(
                        "running",
                        f"Re-entering iterative agent loop for {len(followup_interventions)} "
                        "follow-up intervention(s).",
                    )
                    reconverged = _run_followup_reconvergence_cycle(meaningful_interventions)

                    if monitor.is_stop_requested():
                        print(
                            "[orchestrator] Stop requested from live monitor UI during "
                            "follow-up exploration. Finalizing latest state and exiting..."
                        )
                        break

                    if reconverged:
                        final_report = _generate_final_report(
                            [],
                            phase="final_report_followup_llm",
                            display_label="Final Report Follow-up",
                        )
                        print(
                            "[orchestrator] Updated final report ready after reconvergence. "
                            "Waiting for additional follow-up interventions."
                        )
                        monitor.record_status(
                            "awaiting_followup",
                            "Updated final report ready after reconvergence. Queue another "
                            "follow-up or click Stop & Finalize.",
                        )
                    else:
                        print(
                            "[orchestrator] Follow-up exploration ended without reconvergence "
                            "(additional iteration budget exhausted)."
                        )
                        monitor.record_status(
                            "awaiting_followup",
                            "Follow-up exploration ran, but convergence criteria are still not "
                            "met. Queue more guidance or click Stop & Finalize.",
                        )

                    monitor.set_phase("awaiting_followup", iteration_state.iteration + 1)
                except KeyboardInterrupt:
                    print(
                        "\n[orchestrator] Exiting live follow-up mode and returning the latest "
                        "final report."
                    )
                    break

        if last_final_report_iteration != iteration_state.iteration:
            print(
                "[orchestrator] Finalizing report from the latest ledger state before exit..."
            )
            monitor.record_status(
                "running",
                "Finalizing report from the latest ledger state before exit.",
            )
            final_report = _generate_final_report(
                [],
                phase="final_report_finalize_exit",
                display_label="Final Report Finalize",
            )

        # ── Assemble output ──────────────────────────────────────────────────
        total_iters = iteration_state.iteration + 1
        output = {
            "config": {
                "model": config.model,
                "max_iterations": config.max_iterations,
                "clinical_question": config.clinical_question,
                "dicom_root": config.dicom_root,
            },
            "available_images": available_images,
            "iterations": all_responses,
            "ledger": {
                "findings": [e.to_dict() for e in ledger],
                "negative_findings": [e.to_dict() for e in negative_ledger],
            },
            "final_report": final_report,
            "total_iterations": total_iters,
        }

        print(f"\n[orchestrator] Done. {len(ledger)} findings, "
              f"{len(negative_ledger)} negative findings across "
              f"{total_iters} iterations.")

        dbg.dump_summary(total_iters, len(ledger), len(negative_ledger))
        monitor.record_status("completed")

        return output
    except Exception as e:
        monitor.record_status("failed", f"{type(e).__name__}: {e}")
        raise
    finally:
        monitor.close()


def _summarize_result(result: Any) -> str | None:
    """Create a short summary of MCP result for metadata."""
    if result is None:
        return None
    if isinstance(result, dict):
        for key in ("items", "instances"):
            items = result.get(key, [])
            if items:
                label = "items" if key == "items" else "instances"
                return f"{len(items)} {label} extracted"
        return str(list(result.keys()))
    if isinstance(result, list):
        return f"{len(result)} entries"
    return str(result)[:100]
