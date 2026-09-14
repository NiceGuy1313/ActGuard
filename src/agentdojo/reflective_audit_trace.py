#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any


def content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(block.get("content", "")) for block in content if isinstance(block, dict))
    return str(content)


def shorten(text: str, width: int = 1200) -> str:
    text = text.strip()
    if len(text) <= width:
        return text
    return f"{text[:width]}... [truncated {len(text) - width} chars]"


def indent(text: str, prefix: str = "    ") -> str:
    if not text:
        return prefix + "<empty>"
    return textwrap.indent(text, prefix)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def prediction_summary(prediction: dict[str, Any]) -> str:
    return (
        f"stage={prediction.get('stage')} "
        f"current_tools={compact_json(prediction.get('current_tool_calls', []))} "
        f"previous_allowed={prediction.get('previous_allowed_next_tools')} "
        f"allowed_next_functions={prediction.get('allowed_next_functions')} "
        f"missing={prediction.get('prediction_missing')}"
    )


def action_key(action: dict[str, Any]) -> tuple[str | None, str]:
    return action.get("tool"), compact_json(action.get("args", {}))


def build_indexes(data: dict[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    predictions_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in data.get("reflective_audit_predictions") or []:
        index = prediction.get("message_index")
        if isinstance(index, int):
            predictions_by_index[index].append(prediction)

    _, events_by_index = build_attempts_and_event_index(predictions_by_index, data.get("reflective_audit_events") or [])
    return predictions_by_index, events_by_index


def _same_allowed(left: Any, right: Any) -> bool:
    return (left or []) == (right or [])


def _prediction_calls(prediction: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not prediction:
        return []
    return prediction.get("current_tool_calls") or []


def _attempt_action_keys(attempt: dict[str, Any]) -> set[tuple[str | None, str]]:
    keys = set()
    for prediction_key in ("guard_input_prediction", "llm_output_prediction"):
        for call in _prediction_calls(attempt.get(prediction_key)):
            keys.add(action_key(call))
    return keys


def _attempt_previous_allowed(attempt: dict[str, Any]) -> Any:
    guard_prediction = attempt.get("guard_input_prediction")
    if guard_prediction:
        return guard_prediction.get("previous_allowed_next_tools")
    llm_prediction = attempt.get("llm_output_prediction")
    if llm_prediction:
        return llm_prediction.get("previous_allowed_next_tools")
    return None


def _group_prediction_attempts(
    predictions_by_index: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    attempts_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ordered_attempts: list[dict[str, Any]] = []

    for message_index in sorted(predictions_by_index):
        current_attempt: dict[str, Any] | None = None
        for prediction in predictions_by_index[message_index]:
            stage = prediction.get("stage")
            if stage == "llm_output" or current_attempt is None:
                current_attempt = {
                    "message_index": message_index,
                    "attempt_index": len(attempts_by_index[message_index]),
                    "llm_output_prediction": None,
                    "guard_input_prediction": None,
                    "audit_events": [],
                }
                attempts_by_index[message_index].append(current_attempt)
                ordered_attempts.append(current_attempt)
            if stage == "llm_output":
                current_attempt["llm_output_prediction"] = prediction
            elif stage == "guard_input":
                current_attempt["guard_input_prediction"] = prediction
            else:
                current_attempt.setdefault("other_predictions", []).append(prediction)

    return attempts_by_index, ordered_attempts


def build_attempts_and_event_index(
    predictions_by_index: dict[int, list[dict[str, Any]]], events: list[dict[str, Any]]
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    attempts_by_index, ordered_attempts = _group_prediction_attempts(predictions_by_index)
    unmatched_events: list[dict[str, Any]] = []

    for event in events:
        event_key = action_key(event.get("current_action", {}))
        event_previous_allowed = event.get("previous_allowed_next_tools")
        matched_attempt = None

        for attempt in ordered_attempts:
            if attempt.get("_matched_audit_event"):
                continue
            if event_key not in _attempt_action_keys(attempt):
                continue
            if not _same_allowed(_attempt_previous_allowed(attempt), event_previous_allowed):
                continue
            matched_attempt = attempt
            break

        if matched_attempt is None:
            for attempt in ordered_attempts:
                if attempt.get("_matched_audit_event"):
                    continue
                if event_key in _attempt_action_keys(attempt):
                    matched_attempt = attempt
                    break

        if matched_attempt is None:
            unmatched_events.append(event)
            continue

        matched_attempt["audit_events"].append(event)
        matched_attempt["_matched_audit_event"] = True

    events_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for message_index, attempts in attempts_by_index.items():
        for attempt in attempts:
            for event in attempt.get("audit_events", []):
                events_by_index[message_index].append(event)
    events_by_index[-1].extend(unmatched_events)
    return attempts_by_index, events_by_index


def compact_prediction(
    prediction: dict[str, Any] | None,
    *,
    include_content: bool = False,
    include_tool_calls: bool = True,
) -> dict[str, Any] | None:
    if prediction is None:
        return None
    compacted = {
        "stage": prediction.get("stage"),
        "previous_allowed_next_tools": prediction.get("previous_allowed_next_tools"),
        "allowed_next_functions": prediction.get("allowed_next_functions"),
        "raw_next_function_thought": prediction.get("raw_next_function_thought"),
        "raw_allowed_next_tools": prediction.get("raw_allowed_next_tools"),
        "raw_allowed_next_tools_text": prediction.get("raw_allowed_next_tools_text"),
        "allowed_next_tools_parse_error": prediction.get("allowed_next_tools_parse_error"),
        "has_external_tool_result_context": prediction.get("has_external_tool_result_context"),
        "tool_permissions": prediction.get("tool_permissions"),
        "permission_policy": prediction.get("permission_policy"),
        "allowed_next_tools_requiring_audit": prediction.get("allowed_next_tools_requiring_audit"),
        "prediction_missing": prediction.get("prediction_missing"),
    }
    if include_content:
        compacted["assistant_content"] = prediction.get("assistant_content")
    compacted["raw_assistant_content"] = prediction.get("raw_assistant_content")
    if include_tool_calls:
        compacted["current_tool_calls"] = prediction.get("current_tool_calls") or []
    return compacted


def compact_verifier(verifier: dict[str, Any]) -> dict[str, Any]:
    raw = verifier.get("raw") if isinstance(verifier.get("raw"), dict) else {}
    raw_output = verifier.get("raw_output") or raw.get("raw_output") or raw.get("raw_text")
    return {
        "is_ipi": verifier.get("is_ipi"),
        "ipi_chunk_ids": verifier.get("ipi_chunk_ids") or [],
        "reason": verifier.get("reason"),
        "repair_action": verifier.get("repair_action"),
        "parse_error": verifier.get("parse_error"),
        "raw_output": raw_output,
    }


def verifier_request(verifier: dict[str, Any]) -> Any:
    return verifier.get("request")


def compact_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "source_type": chunk.get("source_type"),
        "metadata": chunk.get("metadata"),
        "text_preview": chunk.get("text_preview"),
    }


def compact_retrieval_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": candidate.get("chunk_id"),
        "retrieval_score": candidate.get("retrieval_score"),
        "source_type": candidate.get("source_type"),
        "metadata": candidate.get("metadata"),
    }


def compact_audit_event(event: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    diagnostics = event.get("localization_diagnostics") or {}
    chunk_by_id: dict[str, dict[str, Any]] = {}
    for chunk in event.get("localized_chunks") or []:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            chunk_by_id[chunk_id] = compact_chunk(chunk)
    retrieval_candidates = diagnostics.get("retrieval_candidates") or []
    for rank, candidate in enumerate(retrieval_candidates, 1):
        chunk_id = candidate.get("chunk_id")
        if chunk_id and chunk_id not in chunk_by_id:
            chunk_by_id[chunk_id] = compact_chunk(candidate)
        if chunk_id and chunk_id in chunk_by_id:
            chunk_by_id[chunk_id]["embedding_rank"] = rank
            chunk_by_id[chunk_id]["embedding_score"] = candidate.get("retrieval_score")

    attribution_by_chunk = event.get("contrastive_attribution") or {}
    for chunk_id, attribution in attribution_by_chunk.items():
        if chunk_id not in chunk_by_id:
            chunk_by_id[chunk_id] = {"chunk_id": chunk_id}
        chunk_by_id[chunk_id]["logprob_delta"] = attribution.get("delta")
        chunk_by_id[chunk_id]["margin_full"] = attribution.get("margin_full")
        chunk_by_id[chunk_id]["margin_without_chunk"] = attribution.get("margin_without_chunk")
        chunk_by_id[chunk_id]["reference_action"] = attribution.get("reference_action")
        chunk_by_id[chunk_id]["actual_score"] = attribution.get("actual_score")
        chunk_by_id[chunk_id]["reference_score"] = attribution.get("reference_score")
        chunk_by_id[chunk_id]["logprob_target"] = attribution.get("target")
        chunk_by_id[chunk_id]["logprob_scoring_unit"] = attribution.get("scoring_unit")

    selected_ids = set(event.get("localized_chunk_ids") or [])
    seed_ids = set(event.get("seed_localized_chunk_ids") or diagnostics.get("seed_selected_chunk_ids") or [])
    verifier_ids = set((event.get("verifier") or {}).get("ipi_chunk_ids") or [])
    for chunk_id, chunk in chunk_by_id.items():
        chunk["seed_selected"] = chunk_id in seed_ids
        chunk["selected_for_verifier"] = chunk_id in selected_ids
        chunk["verifier_marked_ipi"] = chunk_id in verifier_ids

    verifier_ipi_chunks = [
        chunk_by_id[chunk_id]
        for chunk_id in sorted(verifier_ids)
        if chunk_id in chunk_by_id
    ]

    def chunk_sort_key(chunk: dict[str, Any]) -> tuple[int, str]:
        rank = chunk.get("embedding_rank")
        return (rank if isinstance(rank, int) else 10**9, str(chunk.get("chunk_id")))

    verifier_chunk_results = event.get("verifier_chunk_results") or []
    if not verifier_chunk_results:
        verifier_chunk_results = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "is_seed": chunk.get("seed_selected"),
                "is_neighbor": not chunk.get("seed_selected"),
                "sent_to_verifier": chunk.get("selected_for_verifier"),
                "is_ipi": chunk.get("verifier_marked_ipi"),
                "text_preview": chunk.get("text_preview"),
                "source_type": chunk.get("source_type"),
                "metadata": chunk.get("metadata"),
            }
            for chunk in sorted(chunk_by_id.values(), key=chunk_sort_key)
            if chunk.get("selected_for_verifier")
        ]

    selected_chunk_ids = event.get("localized_chunk_ids") or []
    sent_chunk_ids = {chunk.get("chunk_id") for chunk in verifier_chunk_results if chunk.get("sent_to_verifier")}
    sent_chunks = [
        chunk for chunk in sorted(chunk_by_id.values(), key=chunk_sort_key) if chunk.get("chunk_id") in sent_chunk_ids
    ]
    if not sent_chunks:
        sent_chunks = [
            chunk for chunk in sorted(chunk_by_id.values(), key=chunk_sort_key) if chunk.get("selected_for_verifier")
        ]
    logprob_chunks = [
        chunk
        for chunk in sorted(chunk_by_id.values(), key=chunk_sort_key)
        if "logprob_delta" in chunk
    ]

    compacted = {
        "current_action": event.get("current_action"),
        "current_action_permission": event.get("current_action_permission"),
        "previous_allowed_next_tools": event.get("previous_allowed_next_tools"),
        "trigger_reason": event.get("trigger_reason"),
        "audit_context_scope": event.get("audit_context_scope") or diagnostics.get("audit_context_scope"),
        "audit_context_tool_result_indices": event.get("audit_context_tool_result_indices")
        or diagnostics.get("audit_context_tool_result_indices"),
        "audit_context_candidate_chunk_count": event.get("audit_context_candidate_chunk_count")
        or diagnostics.get("audit_context_candidate_chunk_count"),
        "repair": {
            "status": event.get("repair_status"),
            "masked_chunk_ids": event.get("masked_chunk_ids") or [],
            "repair_count_total_before": event.get("repair_count_total_before"),
            "action_repair_count_before": event.get("action_repair_count_before"),
            "action_repair_count": event.get("action_repair_count"),
            "max_repairs_per_action": event.get("max_repairs_per_action"),
        },
        "localization": {
            "method": event.get("localization_method"),
            "seed_chunk_ids": event.get("seed_localized_chunk_ids") or diagnostics.get("seed_selected_chunk_ids") or [],
            "selected_chunk_ids": selected_chunk_ids,
            "chunk_ids_sent_to_verifier": diagnostics.get("chunk_ids_sent_to_verifier") or [],
            "neighbor_expansion_window": diagnostics.get("neighbor_expansion_window"),
            "selected_chunks": sent_chunks,
        },
        "logprob": {
            "scoring_unit": diagnostics.get("logprob_scoring_unit"),
            "effective": diagnostics.get("logprob_effective"),
            "attribution_stopped": diagnostics.get("attribution_stopped"),
            "attribution_stop_reason": diagnostics.get("attribution_stop_reason"),
            "reference_strategy": diagnostics.get("reference_strategy"),
            "parameter_targets": diagnostics.get("parameter_targets"),
            "delta_min": diagnostics.get("delta_min"),
            "delta_max": diagnostics.get("delta_max"),
            "delta_mean": diagnostics.get("delta_mean"),
            "delta_threshold": diagnostics.get("delta_threshold_used_for_effective"),
            "chunks": logprob_chunks,
        },
        "verifier": compact_verifier(event.get("verifier") or {}),
        "verifier_ipi_chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "text_preview": chunk.get("text_preview"),
                "source_type": chunk.get("source_type"),
                "metadata": chunk.get("metadata"),
            }
            for chunk in verifier_ipi_chunks
        ],
        "verifier_override": event.get("verifier_override"),
    }
    if full:
        compacted["permission_policy"] = event.get("permission_policy")
        compacted["previous_allowed_next_tools_requires_audit"] = event.get(
            "previous_allowed_next_tools_requires_audit"
        )
        compacted["prediction_requires_audit"] = event.get("prediction_requires_audit")
        compacted["localization"]["audit_context_source"] = diagnostics.get("audit_context_source")
        compacted["localization"]["embedding_candidate_chunk_ids"] = [
            candidate.get("chunk_id") for candidate in retrieval_candidates if candidate.get("chunk_id")
        ]
        compacted["localization"]["chunks"] = sorted(chunk_by_id.values(), key=chunk_sort_key)
        compacted["localization"]["reranked_chunk_ids"] = diagnostics.get("reranked_chunk_ids") or []
        compacted["localization"]["dangerous_chunk_ids"] = diagnostics.get("dangerous_chunk_ids") or []
        compacted["localization"]["selected_chunk_ids_after_delta_threshold"] = (
            diagnostics.get("selected_chunk_ids_after_delta_threshold") or []
        )
        compacted["localization"]["selection_fallback_to_top_reranked"] = diagnostics.get(
            "selection_fallback_to_top_reranked"
        )
        compacted["localization"]["retrieval_metrics_against_verifier_chunks"] = event.get(
            "retrieval_metrics_against_verifier_chunks"
        )
        compacted["verifier_request"] = event.get("verifier_request") or verifier_request(event.get("verifier") or {})
        compacted["verifier_raw_output"] = event.get("verifier_raw_output") or compact_verifier(
            event.get("verifier") or {}
        ).get("raw_output")
        compacted["verifier_chunk_results"] = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "is_seed": chunk.get("is_seed"),
                "is_neighbor": chunk.get("is_neighbor"),
                "sent_to_verifier": chunk.get("sent_to_verifier"),
                "is_ipi": chunk.get("is_ipi"),
                "text_preview": chunk.get("text_preview"),
                "source_type": chunk.get("source_type"),
                "metadata": chunk.get("metadata"),
            }
            for chunk in verifier_chunk_results
        ]
    return compacted


def _normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": call.get("tool") or call.get("function"),
        "args": call.get("args") or {},
    }


def _tool_call_keys(calls: list[dict[str, Any]]) -> set[tuple[str | None, str]]:
    return {action_key(_normalize_tool_call(call)) for call in calls}


def _attempt_current_tool_calls(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    guard_calls = _prediction_calls(attempt.get("guard_input_prediction"))
    if guard_calls:
        return guard_calls
    return _prediction_calls(attempt.get("llm_output_prediction"))


def _attempt_outcome(attempt: dict[str, Any], is_final_message: bool) -> str:
    audit_events = attempt.get("audit_events") or []
    if audit_events:
        statuses = [event.get("repair_status") for event in audit_events]
        if "blocked_after_retry_limit" in statuses:
            return "blocked"
        if "masked_and_retry" in statuses:
            return "masked_and_regenerated"
        if all(status == "not_needed" for status in statuses):
            return "audit_safe_executed"
        return "audited"
    if is_final_message:
        return "kept_for_execution" if _attempt_current_tool_calls(attempt) else "final_answer"
    return "replaced_by_later_attempt"


def compact_attempt(
    attempt: dict[str, Any],
    final_tool_calls: list[dict[str, Any]],
    final_assistant_content: str,
    full: bool = False,
) -> dict[str, Any]:
    llm_prediction = attempt.get("llm_output_prediction")
    guard_prediction = attempt.get("guard_input_prediction")
    current_tool_calls = _attempt_current_tool_calls(attempt)
    is_final_message = _tool_call_keys(current_tool_calls) == _tool_call_keys(final_tool_calls)
    assistant_content = None
    raw_assistant_content = None
    if guard_prediction:
        assistant_content = guard_prediction.get("assistant_content")
    if assistant_content is None and llm_prediction:
        assistant_content = llm_prediction.get("assistant_content")
    if llm_prediction:
        raw_assistant_content = llm_prediction.get("raw_assistant_content")
    if raw_assistant_content is None and guard_prediction:
        raw_assistant_content = guard_prediction.get("raw_assistant_content")
    if is_final_message and assistant_content is None:
        assistant_content = final_assistant_content

    compacted = {
        "attempt_index": attempt.get("attempt_index"),
        "is_final_assistant_message": is_final_message,
        "outcome": _attempt_outcome(attempt, is_final_message),
        "assistant_content": assistant_content,
        "current_tool_calls": current_tool_calls,
        "llm_output": compact_prediction(llm_prediction, include_tool_calls=False),
        "guard_input": compact_prediction(guard_prediction, include_tool_calls=False),
        "audit_events": [compact_audit_event(event, full=full) for event in attempt.get("audit_events") or []],
    }
    compacted["raw_assistant_content"] = raw_assistant_content
    if attempt.get("other_predictions"):
        compacted["other_predictions"] = [
            compact_prediction(prediction, include_tool_calls=False)
            for prediction in attempt.get("other_predictions") or []
        ]
    return compacted


def final_attempt_summary(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    final_attempt = next((attempt for attempt in attempts if attempt.get("is_final_assistant_message")), None)
    if final_attempt is None and attempts:
        final_attempt = attempts[-1]
    if final_attempt is None:
        return None

    guard_input = final_attempt.get("guard_input") or {}
    llm_output = final_attempt.get("llm_output") or {}
    prediction = guard_input or llm_output
    raw_prediction = llm_output or guard_input
    return {
        "allowed_next_functions": prediction.get("allowed_next_functions"),
        "raw_next_function_thought": raw_prediction.get("raw_next_function_thought"),
        "raw_allowed_next_tools": raw_prediction.get("raw_allowed_next_tools"),
        "raw_allowed_next_tools_text": raw_prediction.get("raw_allowed_next_tools_text"),
        "allowed_next_tools_parse_error": raw_prediction.get("allowed_next_tools_parse_error"),
        "previous_allowed_next_tools": prediction.get("previous_allowed_next_tools"),
        "prediction_missing": prediction.get("prediction_missing"),
    }


def build_trace_json(
    data: dict[str, Any],
    source_path: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    predictions_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in data.get("reflective_audit_predictions") or []:
        index = prediction.get("message_index")
        if isinstance(index, int):
            predictions_by_index[index].append(prediction)
    attempts_by_index, events_by_index = build_attempts_and_event_index(
        predictions_by_index, data.get("reflective_audit_events") or []
    )
    steps = []

    for index, message in enumerate(data.get("messages") or []):
        role = message.get("role")
        step = {
            "index": index,
            "role": role,
        }

        if role == "assistant":
            step["assistant_content"] = content_text(message)
            step["tool_calls"] = message.get("tool_calls") or []
            attempts = [
                compact_attempt(
                    attempt,
                    message.get("tool_calls") or [],
                    content_text(message),
                    full=full,
                )
                for attempt in attempts_by_index.get(index, [])
            ]
            allowed_next = final_attempt_summary(attempts)
            step["allowed_next_functions"] = (
                allowed_next.get("allowed_next_functions") if allowed_next is not None else None
            )
            step["raw_next_function_thought"] = (
                allowed_next.get("raw_next_function_thought") if allowed_next is not None else None
            )
            step["raw_allowed_next_tools"] = (
                allowed_next.get("raw_allowed_next_tools") if allowed_next is not None else None
            )
            step["raw_allowed_next_tools_text"] = (
                allowed_next.get("raw_allowed_next_tools_text") if allowed_next is not None else None
            )
            step["allowed_next_tools_parse_error"] = (
                allowed_next.get("allowed_next_tools_parse_error") if allowed_next is not None else None
            )
            step["previous_allowed_next_tools"] = (
                allowed_next.get("previous_allowed_next_tools") if allowed_next is not None else None
            )
            step["prediction_missing"] = (
                allowed_next.get("prediction_missing") if allowed_next is not None else None
            )
            step["attempts"] = attempts
        elif role == "tool":
            step["tool_call"] = message.get("tool_call")
            step["error"] = message.get("error")
            step["tool_result"] = content_text(message)
        else:
            step["content"] = content_text(message)

        if role != "assistant":
            predictions = [
                compact_prediction(prediction)
                for prediction in predictions_by_index.get(index, [])
            ]
            audit_events = [compact_audit_event(event, full=full) for event in events_by_index.get(index, [])]
            if predictions:
                step["predictions"] = predictions
            if audit_events:
                step["audit_events"] = audit_events

        if full:
            step["raw_message"] = message
            step["raw_reflective_audit_predictions"] = predictions_by_index.get(index, [])
            step["raw_reflective_audit_events"] = events_by_index.get(index, [])

        steps.append(step)

    unmatched_events = events_by_index.get(-1, [])
    return {
        "source_path": str(source_path) if source_path is not None else None,
        "metadata": {
            "suite_name": data.get("suite_name"),
            "pipeline_name": data.get("pipeline_name"),
            "user_task_id": data.get("user_task_id"),
            "attack_type": data.get("attack_type"),
            "injection_task_id": data.get("injection_task_id"),
            "utility": data.get("utility"),
            "security_or_asr": data.get("security"),
            "reflective_audit_config": data.get("reflective_audit_config"),
            "reflective_audit_prompts": data.get("reflective_audit_prompts"),
        },
        "steps": steps,
        "unmatched_audit_events": [compact_audit_event(event, full=full) for event in unmatched_events],
    }


def print_audit_event(event: dict[str, Any], max_text: int) -> None:
    print("  [AUDIT]")
    print(f"    action: {compact_json(event.get('current_action'))}")
    print(f"    previous_allowed_next_tools: {event.get('previous_allowed_next_tools')}")
    print(f"    trigger_reason: {event.get('trigger_reason')}")
    print(f"    repair_count_total_before: {event.get('repair_count_total_before')}")
    print(f"    action_repair_count_before: {event.get('action_repair_count_before')}")
    print(f"    localization_method: {event.get('localization_method')}")

    diagnostics = event.get("localization_diagnostics") or {}
    candidates = diagnostics.get("retrieval_candidates") or []
    if candidates:
        print("    retrieval candidates before logprob rerank:")
        for rank, candidate in enumerate(candidates, 1):
            print(
                "      "
                f"#{rank} {candidate.get('chunk_id')} "
                f"score={candidate.get('retrieval_score')} "
                f"source={candidate.get('source_type')} "
                f"metadata={compact_json(candidate.get('metadata'))}"
            )
            print(indent(shorten(str(candidate.get("text_preview", "")), max_text), "        "))

    print(f"    reranked_chunk_ids: {diagnostics.get('reranked_chunk_ids')}")
    print(f"    selected localized_chunk_ids: {event.get('localized_chunk_ids')}")
    print(
        "    logprob diagnostics: "
        f"unit={diagnostics.get('logprob_scoring_unit')} "
        f"effective={diagnostics.get('logprob_effective')} "
        f"stopped={diagnostics.get('attribution_stopped')} "
        f"stop_reason={diagnostics.get('attribution_stop_reason')} "
        f"reference_strategy={diagnostics.get('reference_strategy')} "
        f"delta_min={diagnostics.get('delta_min')} "
        f"delta_max={diagnostics.get('delta_max')} "
        f"delta_mean={diagnostics.get('delta_mean')} "
        f"threshold={diagnostics.get('delta_threshold_used_for_effective')}"
    )

    attribution = event.get("contrastive_attribution") or {}
    if attribution:
        print("    contrastive logprob attribution:")
        for chunk_id, item in sorted(attribution.items()):
            print(
                "      "
                f"{chunk_id}: delta={item.get('delta')} "
                f"margin_full={item.get('margin_full')} "
                f"margin_without={item.get('margin_without_chunk')} "
                f"actual_score={item.get('actual_score')} "
                f"reference_action={item.get('reference_action')} "
                f"reference_score={item.get('reference_score')} "
                f"unit={item.get('scoring_unit')} "
                f"target={item.get('target')}"
            )

    localized_chunks = event.get("localized_chunks") or []
    if localized_chunks:
        print("    selected chunk text:")
        for chunk in localized_chunks:
            print(
                "      "
                f"{chunk.get('chunk_id')} source={chunk.get('source_type')} "
                f"metadata={compact_json(chunk.get('metadata'))}"
            )
            print(indent(shorten(str(chunk.get("text_preview", "")), max_text), "        "))

    metrics = event.get("retrieval_metrics_against_verifier_chunks")
    if metrics:
        print(f"    retrieval pseudo-metrics: {compact_json(metrics)}")

    verifier = event.get("verifier") or {}
    print("    verifier:")
    print(indent(pretty_json(verifier), "      "))
    request = event.get("verifier_request")
    if request:
        print("    verifier request:")
        print(indent(shorten(pretty_json(request), max_text), "      "))
    raw_output = event.get("verifier_raw_output") or verifier.get("raw_output")
    if raw_output:
        print("    verifier raw output:")
        print(indent(shorten(str(raw_output), max_text), "      "))
    chunk_results = event.get("verifier_chunk_results") or []
    if chunk_results:
        print("    verifier chunk results:")
        for chunk in chunk_results:
            relation = "seed" if chunk.get("is_seed") else "neighbor"
            print(
                "      "
                f"{chunk.get('chunk_id')} relation={relation} "
                f"is_ipi={chunk.get('is_ipi')} sent={chunk.get('sent_to_verifier')}"
            )
            print(indent(shorten(str(chunk.get("text_preview", "")), max_text), "        "))
    verifier_ids = set(verifier.get("ipi_chunk_ids") or [])
    if verifier_ids:
        localized_by_id = {
            chunk.get("chunk_id"): chunk
            for chunk in event.get("localized_chunks") or []
            if chunk.get("chunk_id")
        }
        print("    verifier marked chunk text:")
        for chunk_id in sorted(verifier_ids):
            chunk = localized_by_id.get(chunk_id)
            if chunk is None:
                print(f"      {chunk_id}: <not found in localized_chunks>")
                continue
            print(f"      {chunk_id}:")
            print(indent(shorten(str(chunk.get("text_preview", "")), max_text), "        "))
    print(f"    repair_status: {event.get('repair_status')}")
    if event.get("masked_chunk_ids") is not None:
        print(f"    masked_chunk_ids: {event.get('masked_chunk_ids')}")
    if event.get("repair_note"):
        print(f"    repair_note: {event.get('repair_note')}")


def print_trace(path: Path, max_text: int) -> None:
    data = json.loads(path.read_text())
    predictions_by_index, events_by_index = build_indexes(data)

    print(f"FILE: {path}")
    print(
        "META: "
        f"suite={data.get('suite_name')} "
        f"pipeline={data.get('pipeline_name')} "
        f"user_task={data.get('user_task_id')} "
        f"attack={data.get('attack_type')} "
        f"injection_task={data.get('injection_task_id')} "
        f"utility={data.get('utility')} "
        f"security_or_asr={data.get('security')}"
    )
    print("=" * 100)

    for index, message in enumerate(data.get("messages") or []):
        role = message.get("role")
        print(f"\n[{index}] {role.upper()}")
        print("-" * 100)

        for prediction in predictions_by_index.get(index, []):
            print(f"  [PREDICTION] {prediction_summary(prediction)}")
            if prediction.get("raw_next_function_thought") is not None:
                print("    raw_next_function_thought:")
                print(indent(shorten(str(prediction.get("raw_next_function_thought")), max_text), "      "))
            if prediction.get("raw_allowed_next_tools_text") is not None:
                print(f"    raw_allowed_next_tools: {prediction.get('raw_allowed_next_tools')}")
                if prediction.get("allowed_next_tools_parse_error"):
                    print(f"    allowed_next_tools_parse_error: {prediction.get('allowed_next_tools_parse_error')}")
            if prediction.get("raw_assistant_content") is not None:
                print("    raw_assistant_content:")
                print(indent(shorten(str(prediction.get("raw_assistant_content")), max_text), "      "))

        if role == "assistant":
            print("  content:")
            print(indent(shorten(content_text(message), max_text)))
            calls = message.get("tool_calls")
            if calls:
                print("  tool_calls:")
                print(indent(pretty_json(calls)))
        elif role == "tool":
            call = message.get("tool_call")
            if call:
                print(f"  tool_call: {compact_json(call)}")
            if message.get("error"):
                print(f"  error: {message.get('error')}")
            print("  tool_result:")
            print(indent(shorten(content_text(message), max_text)))
        else:
            print(indent(shorten(content_text(message), max_text)))

        for event in events_by_index.get(index, []):
            print_audit_event(event, max_text)

    if events_by_index.get(-1):
        print("\n[UNMATCHED AUDIT EVENTS]")
        print("-" * 100)
        for event in events_by_index[-1]:
            print_audit_event(event, max_text)


def write_trace_json(input_path: Path, output_path: Path, full: bool = False) -> None:
    data = json.loads(input_path.read_text())
    trace = build_trace_json(data, input_path, full=full)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or export a chronological reflective_audit trace from one run JSON.")
    parser.add_argument("path", type=Path, help="Path to one benchmark JSON file.")
    parser.add_argument("--max-text", type=int, default=1200, help="Maximum text chars printed per block.")
    parser.add_argument("--output", type=Path, help="Write the chronological trace to this JSON file.")
    parser.add_argument("--no-print", action="store_true", help="Only write --output JSON and skip terminal printing.")
    parser.add_argument("--full-json", action="store_true", help="Keep raw messages/events in --output JSON.")
    args = parser.parse_args()
    if args.output is not None:
        write_trace_json(args.path, args.output, full=args.full_json)
        print(f"Wrote chronological trace JSON to {args.output}")
    if not args.no_print:
        print_trace(args.path, args.max_text)


if __name__ == "__main__":
    main()
