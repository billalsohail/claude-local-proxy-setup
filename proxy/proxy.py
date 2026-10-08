"""
Ollama routing proxy for Claude Code.
Accepts Anthropic API format, orchestrates eligible requests through local Ollama
(using ECC agent personas), and falls back to the real Anthropic API when needed.
"""

import asyncio
import hashlib
import json
import os
import random
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from ecc_router import (
    detect_agent,
    is_project_request,
    classify_request_tools,
    classify_tool_intent,
    all_history_tools_are_local,
    load_agent_system_prompt,
    last_message_has_fresh_text,
    classify_mid_conversation_intent,
    LOCAL_TOOLS,
)
from orchestrator import orchestrate_single, orchestrate_project

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


cfg = load_config()

OLLAMA_BASE = cfg["ollama_base_url"]
ANTHROPIC_BASE = cfg["anthropic_base_url"]
PREFERRED_MODEL = cfg["preferred_model"]
CHAR_THRESHOLD = cfg["char_threshold"]
ROUTE_HAIKU = cfg.get("route_haiku_to_ollama", True)
ORCHESTRATOR_MODE = cfg.get("orchestrator_mode", False)
ECC_PATH = cfg.get("ecc_path", "")
ORCHESTRATOR_CEILING = cfg.get("orchestrator_ceiling", 15000)
REFINEMENT_THRESHOLD = cfg.get("refinement_threshold", 500)
MAX_SUBTASKS = cfg.get("max_subtasks", 5)
OLLAMA_THINK = cfg.get("ollama_think", False)
DECOMP_MAX_TOKENS = cfg.get("decomp_max_tokens", 768)
MIN_VIABLE_DRAFT_CHARS = cfg.get("min_viable_draft_chars", 20)
INTENT_ROUTING_MODE = cfg.get("intent_routing_mode", "off")
INTENT_ROUTING_LIVE_PCT = cfg.get("intent_routing_live_pct", 0)
SHADOW_VALIDATION_SAMPLE_RATE = cfg.get("shadow_validation_sample_rate", 0.3)
MID_CONV_ROUTING_MODE = cfg.get("mid_conv_routing_mode", "off")
MID_CONV_LIVE_PCT = cfg.get("mid_conv_live_pct", 0)

LOG_PATH = Path.home() / ".claude" / "ollama-proxy" / "savings.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_stats = {"anthropic": 0, "ollama": 0, "ollama-tools": 0, "orchestrate-single": 0, "orchestrate-project": 0}
_orchestration_escalation = {"total": 0, "escalated": 0}

app = FastAPI()

# ---------------------------------------------------------------------------
# Helpers (kept for non-orchestrator fallback path)
# ---------------------------------------------------------------------------


def _message_char_count(messages: list) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("text", ""))
    return total


def _has_tool_content(messages: list) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result"):
                    return True
    return False


def _build_forward_headers(request: Request) -> dict:
    headers = {
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        "content-type": "application/json",
    }
    for hdr in ("x-api-key", "authorization"):
        if val := request.headers.get(hdr):
            headers[hdr] = val
    if "x-api-key" not in headers and "authorization" not in headers:
        if key := os.environ.get("ANTHROPIC_API_KEY"):
            headers["x-api-key"] = key
    if beta := request.headers.get("anthropic-beta"):
        headers["anthropic-beta"] = beta
    return headers


# ---------------------------------------------------------------------------
# Ollama tool-call translation helpers
# ---------------------------------------------------------------------------


def _anthropic_tools_to_ollama(tools: list) -> list:
    """Convert Anthropic tool definitions to Ollama function format."""
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def _translate_messages_for_ollama(body: dict) -> list:
    """
    Convert Anthropic message history (including tool_use / tool_result blocks)
    to Ollama chat format:
      - assistant tool_use blocks  → assistant.tool_calls
      - user tool_result blocks    → role:tool messages
    """
    msgs: list = []

    if system := body.get("system"):
        if isinstance(system, str):
            msgs.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
            msgs.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            msg_dict: dict = {
                "role": "assistant",
                "content": " ".join(text_parts) if text_parts else "",
            }
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            msgs.append(msg_dict)

        elif role == "user":
            tool_results: list[dict] = []
            text_parts = []
            for block in content:
                btype = block.get("type")
                if btype == "tool_result":
                    rc = block.get("content", "")
                    if isinstance(rc, list):
                        rc = "\n".join(
                            b.get("text", "") for b in rc
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    elif not isinstance(rc, str):
                        rc = str(rc)
                    tool_results.append({"role": "tool", "content": rc})
                elif btype == "text":
                    text_parts.append(block.get("text", ""))

            msgs.extend(tool_results)
            if text_parts:
                msgs.append({"role": "user", "content": " ".join(text_parts)})

    return msgs


async def _call_ollama_with_tools(body: dict) -> tuple[dict, str]:
    """Send a tool-enabled request to Ollama; returns (response, agent_name)."""
    messages = body.get("messages", [])
    agent_name = detect_agent(messages)

    ollama_messages = _translate_messages_for_ollama(body)

    if ECC_PATH:
        ecc_prompt = load_agent_system_prompt(agent_name, ECC_PATH)
        if ollama_messages and ollama_messages[0]["role"] == "system":
            ollama_messages[0]["content"] = (
                ecc_prompt + "\n\n---\n\n" + ollama_messages[0]["content"]
            )
        else:
            ollama_messages.insert(0, {"role": "system", "content": ecc_prompt})

    # Defense-in-depth: only ever hand Ollama the sanctioned local-safe tools,
    # even though the routing gate (classify_request_tools == "local") should
    # already guarantee the declared tool set is a subset of LOCAL_TOOLS.
    local_only_tools = [
        t for t in body.get("tools", [])
        if isinstance(t, dict) and t.get("name") in LOCAL_TOOLS
    ]
    payload = {
        "model": PREFERRED_MODEL,
        "messages": ollama_messages,
        "tools": _anthropic_tools_to_ollama(local_only_tools),
        "stream": False,
        "options": {"num_predict": body.get("max_tokens", 4096)},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json(), agent_name


async def _emit_ollama_tool_response(ollama_resp: dict, original_model: str):
    """
    Convert an Ollama chat response (possibly containing tool_calls) into
    Anthropic-format SSE events.
    """
    msg_id = f"msg_ollama_{uuid.uuid4().hex[:16]}"
    message = ollama_resp.get("message", {})
    tool_calls = message.get("tool_calls") or []
    text_content = message.get("content") or ""

    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':original_model,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"

    block_index = 0

    if text_content:
        yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':block_index,'content_block':{'type':'text','text':''}})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':block_index,'delta':{'type':'text_delta','text':text_content}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':block_index})}\n\n"
        block_index += 1

    for tc in tool_calls:
        func = tc.get("function", {})
        tool_name = func.get("name", "")
        arguments = func.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        tool_id = f"toolu_{uuid.uuid4().hex[:16]}"

        yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':block_index,'content_block':{'type':'tool_use','id':tool_id,'name':tool_name,'input':{}}})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':block_index,'delta':{'type':'input_json_delta','partial_json':json.dumps(arguments)}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':block_index})}\n\n"
        block_index += 1

    stop_reason = "tool_use" if tool_calls else "end_turn"
    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _last_user_char_count(messages: list) -> int:
    """Char count of only the last user message — used for orchestrator ceiling."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            return sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
    return 0


def _in_canary(body: dict, pct: float) -> bool:
    """
    Stable per-conversation sampling for the intent-routing canary rollout.
    Hashes the first message in the history (which persists unchanged across
    turns within a session) so a conversation samples the same way on every
    turn, rather than flip-flopping request to request.
    """
    if pct <= 0:
        return False
    if pct >= 100:
        return True
    messages = body.get("messages", [])
    key = ""
    if messages:
        content = messages[0].get("content", "")
        if isinstance(content, str):
            key = content
        elif isinstance(content, list):
            key = json.dumps(content, sort_keys=True)[:500]
    if not key:
        key = body.get("model", "")
    digest = hashlib.sha256(key.encode("utf-8", "ignore")).hexdigest()
    return (int(digest, 16) % 100) < pct


def route_request(body: dict, chars: int) -> tuple[str, str, dict | None]:
    """
    Returns (destination, reason, shadow_info).

    shadow_info is None except when a has-remote-tools request was scored by
    the intent-routing heuristic — it then carries the heuristic's prediction
    (would_route/needs_tool/matched) for shadow validation logging, regardless
    of whether the request actually got rerouted.

    Priority order:
    1. Tool-tier routing (takes precedence over everything else):
       a. All defined tools are LOCAL_TOOLS + history is clean → ollama-tools
       b. Any remote tool defined:
          - tool_use/tool_result in history (mid-tool-call sequence) →
            mid_conv_routing_mode != "off" scores the turn: if every prior
            tool call was read-only (all_history_tools_are_local) and the
            latest genuine human instruction shows no action-verb intent
            (classify_tool_intent) and it's under ORCHESTRATOR_CEILING, it
            reroutes live to orchestrate-single/-project ("safe-only"/"live"
            modes) — but only when the conversation is canary-sampled into
            mid_conv_live_pct (_in_canary, same stable per-conversation
            hashing as the intent-routing canary below). Everything else
            with tool history stays on Anthropic but is shadow-scored the
            same way. Fail-safe default: Anthropic.
          - non-streaming → Anthropic
          - intent_routing_mode != "off" → classify_tool_intent scores the
            request; only reroutes in "live" mode, canary-sampled, and only
            when no tool intent is detected. Fail-safe default: Anthropic.
          - otherwise → Anthropic
    2. No tools defined:
       a. tool_use/tool_result in history → Anthropic (mid-tool-call sequence)
       b. non-streaming → Anthropic
       c. orchestrator_mode → orchestrate-single / orchestrate-project
       d. legacy: haiku / short-context → Ollama, else Anthropic
    """
    messages = body.get("messages", [])
    tool_tier = classify_request_tools(body)

    if tool_tier == "local":
        if all_history_tools_are_local(messages):
            return "ollama-tools", "local-tools-only", None
        return "anthropic", "remote-tool-in-history", None

    if tool_tier == "remote":
        if _has_tool_content(messages):
            is_stream = body.get("stream", False)
            if is_stream and MID_CONV_ROUTING_MODE != "off" and ORCHESTRATOR_MODE and ECC_PATH:
                is_safe = all_history_tools_are_local(messages)
                needs_tool, matched = classify_tool_intent(body)
                would_route = (
                    "anthropic" if needs_tool
                    else ("orchestrate-project" if is_project_request(messages) else "orchestrate-single")
                )
                fresh_turn = last_message_has_fresh_text(messages)
                needs_tool_candidate, matched_candidate = classify_mid_conversation_intent(
                    messages, fresh_turn
                )
                would_route_candidate = (
                    "anthropic" if needs_tool_candidate
                    else ("orchestrate-project" if is_project_request(messages) else "orchestrate-single")
                )
                shadow_info = {
                    "would_route": would_route,
                    "needs_tool": needs_tool,
                    "matched": matched,
                    "stage": "mid-conversation",
                    "safe_slice": is_safe,
                    "fresh_turn": fresh_turn,
                    "would_route_candidate": would_route_candidate,
                    "needs_tool_candidate": needs_tool_candidate,
                    "matched_candidate": matched_candidate,
                }

                is_safe_candidate = is_safe and not needs_tool_candidate
                under_ceiling = chars <= ORCHESTRATOR_CEILING
                in_canary = _in_canary(body, MID_CONV_LIVE_PCT)

                if (
                    MID_CONV_ROUTING_MODE in ("safe-only", "live")
                    and is_safe_candidate
                    and under_ceiling
                    and in_canary
                ):
                    if would_route == "orchestrate-project":
                        return "orchestrate-project", "project-decomp:safe-continuation", None
                    return "orchestrate-single", f"safe-continuation:{detect_agent(messages)}", None

                # Not in the safe slice, mode is shadow-only, blocked by the
                # ceiling, or not sampled into the live canary: stay on
                # Anthropic (fail-safe), keep the prediction for shadow
                # validation.
                if is_safe_candidate and not under_ceiling:
                    reason = "tool-content-in-history:ceiling-exceeded"
                elif is_safe_candidate and under_ceiling and not in_canary:
                    reason = "tool-content-in-history:not-in-canary"
                else:
                    reason = "tool-content-in-history"
                return "anthropic", reason, shadow_info

            return "anthropic", "tool-content-in-history", None
        if not body.get("stream", False):
            return "anthropic", "non-streaming", None

        if INTENT_ROUTING_MODE != "off" and ORCHESTRATOR_MODE and ECC_PATH:
            needs_tool, matched = classify_tool_intent(body)
            would_route = (
                "anthropic" if needs_tool
                else ("orchestrate-project" if is_project_request(messages) else "orchestrate-single")
            )
            shadow_info = {"would_route": would_route, "needs_tool": needs_tool, "matched": matched}

            if (
                INTENT_ROUTING_MODE == "live"
                and not needs_tool
                and _in_canary(body, INTENT_ROUTING_LIVE_PCT)
            ):
                if would_route == "orchestrate-project":
                    return "orchestrate-project", "project-decomp:intent-fallthrough", None
                return "orchestrate-single", f"orchestrated-{detect_agent(messages)}:intent-fallthrough", None

            # Shadow mode, or live but tool-needed/not canary-sampled: stay on
            # Anthropic (fail-safe), but keep the prediction for validation.
            return "anthropic", "has-remote-tools", shadow_info

        return "anthropic", "has-remote-tools", None

    # No tools defined from here on.
    if _has_tool_content(messages):
        return "anthropic", "tool-content-in-history", None

    # Non-streaming: Claude Code's internal classifier/safety calls always use
    # stream=false and need real Claude responses — not Ollama drafts.
    if not body.get("stream", False):
        return "anthropic", "non-streaming", None

    if ORCHESTRATOR_MODE and ECC_PATH:
        if is_project_request(messages):
            return "orchestrate-project", "project-decomp", None
        return "orchestrate-single", f"orchestrated-{detect_agent(messages)}", None

    # Legacy routing (orchestrator disabled)
    model = body.get("model", "")
    if ROUTE_HAIKU and "haiku" in model.lower():
        return "ollama", "haiku+no-tools", None
    if chars < CHAR_THRESHOLD:
        return "ollama", f"short-context<{CHAR_THRESHOLD}", None
    return "anthropic", "long-context", None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_decision(route: str, model: str, chars: int, reason: str,
                 agent_used: str = "", subtasks: int = 0, escalated: bool | None = None,
                 shadow: dict | None = None):
    _stats[route] = _stats.get(route, 0) + 1
    if route in ("orchestrate-single", "orchestrate-project") and escalated is not None:
        _orchestration_escalation["total"] += 1
        if escalated:
            _orchestration_escalation["escalated"] += 1
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "model": model,
        "chars": chars,
        "reason": reason,
    }
    if agent_used:
        entry["agent_used"] = agent_used
    if subtasks:
        entry["subtasks"] = subtasks
    if escalated is not None:
        entry["escalated"] = escalated
    if shadow is not None:
        entry["shadow"] = shadow
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_shadow_validation(shadow_info: dict, actual_stop_reason: str | None):
    """
    Records ground truth for the intent-routing heuristic: what it predicted
    vs. what the real Anthropic call's stop_reason turned out to be. Tagged
    with its own route value so it doesn't skew the main routing stats in
    /status or stats.sh.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "route": "shadow-validation",
        "would_route": shadow_info.get("would_route"),
        "needs_tool": shadow_info.get("needs_tool"),
        "matched": shadow_info.get("matched"),
        "actual_stop_reason": actual_stop_reason,
        "false_negative": (not shadow_info.get("needs_tool")) and actual_stop_reason == "tool_use",
    }
    if "stage" in shadow_info:
        entry["stage"] = shadow_info["stage"]
    if "safe_slice" in shadow_info:
        entry["safe_slice"] = shadow_info["safe_slice"]
    if "fresh_turn" in shadow_info:
        entry["fresh_turn"] = shadow_info["fresh_turn"]
    if "needs_tool_candidate" in shadow_info:
        entry["needs_tool_candidate"] = shadow_info["needs_tool_candidate"]
        entry["matched_candidate"] = shadow_info.get("matched_candidate")
        entry["false_negative_candidate"] = (
            not shadow_info.get("needs_tool_candidate")
        ) and actual_stop_reason == "tool_use"
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Legacy Ollama helpers (used when orchestrator is disabled)
# ---------------------------------------------------------------------------


def _anthropic_to_ollama_messages(body: dict) -> list:
    msgs = []
    if system := body.get("system"):
        if isinstance(system, str):
            msgs.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
            msgs.append({"role": "system", "content": text})
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            msgs.append({"role": role, "content": text})
    return msgs


def _ollama_to_anthropic_response(ollama_resp: dict, original_model: str) -> dict:
    text = ollama_resp.get("message", {}).get("content", "")
    return {
        "id": f"msg_ollama_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": original_model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": len(text.split())},
    }


async def _stream_from_ollama_legacy(body: dict, original_model: str):
    msg_id = f"msg_ollama_{uuid.uuid4().hex[:16]}"
    ollama_messages = _anthropic_to_ollama_messages(body)
    payload = {
        "model": PREFERRED_MODEL,
        "messages": ollama_messages,
        "stream": True,
        "options": {"num_predict": body.get("max_tokens", 4096)},
    }
    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','content':[],'model':original_model,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
    yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta_text = chunk.get("message", {}).get("content", "")
                if delta_text:
                    yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':delta_text}})}\n\n"
                if chunk.get("done"):
                    break
    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"


async def _stream_from_anthropic(body: dict, headers: dict):
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", f"{ANTHROPIC_BASE}/v1/messages",
            json=body, headers=headers,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk


_STOP_REASON_RE = re.compile(rb'"stop_reason"\s*:\s*"([a-zA-Z_]+)"')


async def _stream_from_anthropic_with_shadow(body: dict, headers: dict, shadow_info: dict):
    """
    Same as _stream_from_anthropic, but also scrapes the real stop_reason out
    of the SSE stream to validate the intent-routing heuristic's prediction
    against ground truth. Never alters what the client receives.
    """
    actual_stop_reason = None
    tail = b""
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", f"{ANTHROPIC_BASE}/v1/messages",
            json=body, headers=headers,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
                if match := _STOP_REASON_RE.search(tail + chunk):
                    actual_stop_reason = match.group(1).decode()
                tail = chunk[-64:]
    log_shadow_validation(shadow_info, actual_stop_reason)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    original_model = body.get("model", "unknown")
    msgs = body.get("messages", [])
    chars = _message_char_count(msgs)
    destination, reason, shadow_info = route_request(body, chars)
    is_stream = body.get("stream", False)

    # -----------------------------------------------------------------------
    # Orchestrator paths
    # -----------------------------------------------------------------------

    if destination == "orchestrate-single":
        forward_headers = _build_forward_headers(request)
        try:
            stream, agent_used, escalated = await orchestrate_single(
                body=body,
                forward_headers=forward_headers,
                ollama_base=OLLAMA_BASE,
                preferred_model=PREFERRED_MODEL,
                ecc_path=ECC_PATH,
                anthropic_base=ANTHROPIC_BASE,
                refinement_threshold=REFINEMENT_THRESHOLD,
                ollama_think=OLLAMA_THINK,
                min_viable_draft_chars=MIN_VIABLE_DRAFT_CHARS,
            )
            log_decision("orchestrate-single", original_model, chars, reason,
                         agent_used=agent_used, escalated=escalated)
            return StreamingResponse(stream, media_type="text/event-stream")
        except Exception as e:
            log_decision("anthropic", original_model, chars, f"orchestrate-error:{e}")
            destination = "anthropic"
            reason = f"orchestrate-fallback:{e}"

    elif destination == "orchestrate-project":
        forward_headers = _build_forward_headers(request)
        try:
            stream, agent_used, escalated, subtask_count = await orchestrate_project(
                body=body,
                forward_headers=forward_headers,
                ollama_base=OLLAMA_BASE,
                preferred_model=PREFERRED_MODEL,
                ecc_path=ECC_PATH,
                anthropic_base=ANTHROPIC_BASE,
                max_subtasks=MAX_SUBTASKS,
                refinement_threshold=REFINEMENT_THRESHOLD,
                ollama_think=OLLAMA_THINK,
                decomp_max_tokens=DECOMP_MAX_TOKENS,
                min_viable_draft_chars=MIN_VIABLE_DRAFT_CHARS,
            )
            log_decision("orchestrate-project", original_model, chars, reason,
                         agent_used=agent_used, subtasks=subtask_count, escalated=escalated)
            return StreamingResponse(stream, media_type="text/event-stream")
        except Exception as e:
            log_decision("anthropic", original_model, chars, f"project-error:{e}")
            destination = "anthropic"
            reason = f"project-fallback:{e}"

    # -----------------------------------------------------------------------
    # Local-tool Ollama path (qwen3.5 with native tool calling)
    # -----------------------------------------------------------------------

    elif destination == "ollama-tools":
        try:
            ollama_resp, agent_used = await _call_ollama_with_tools(body)
            log_decision("ollama-tools", original_model, chars, reason, agent_used=agent_used)
            return StreamingResponse(
                _emit_ollama_tool_response(ollama_resp, original_model),
                media_type="text/event-stream",
            )
        except Exception as e:
            log_decision("anthropic", original_model, chars, f"ollama-tools-error:{e}")
            destination = "anthropic"
            reason = f"ollama-tools-fallback:{e}"

    # -----------------------------------------------------------------------
    # Legacy Ollama path (orchestrator disabled)
    # -----------------------------------------------------------------------

    elif destination == "ollama":
        log_decision("ollama", original_model, chars, reason)
        if is_stream:
            return StreamingResponse(
                _stream_from_ollama_legacy(body, original_model),
                media_type="text/event-stream",
            )
        ollama_messages = _anthropic_to_ollama_messages(body)
        payload = {
            "model": PREFERRED_MODEL,
            "messages": ollama_messages,
            "stream": False,
            "options": {"num_predict": body.get("max_tokens", 4096)},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
                resp.raise_for_status()
                return _ollama_to_anthropic_response(resp.json(), original_model)
            except Exception as e:
                log_decision("anthropic", original_model, chars, f"ollama-error:{e}")
                # fall through to Anthropic

    # -----------------------------------------------------------------------
    # Anthropic passthrough
    # -----------------------------------------------------------------------

    log_decision("anthropic", original_model, chars, reason, shadow=shadow_info)
    forward_headers = _build_forward_headers(request)

    if is_stream:
        if shadow_info is not None and random.random() < SHADOW_VALIDATION_SAMPLE_RATE:
            return StreamingResponse(
                _stream_from_anthropic_with_shadow(body, forward_headers, shadow_info),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            _stream_from_anthropic(body, forward_headers),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{ANTHROPIC_BASE}/v1/messages",
            json=body,
            headers=forward_headers,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "claude-sonnet-4-6", "object": "model"},
            {"id": "claude-opus-4-8", "object": "model"},
            {"id": "claude-haiku-4-5-20251001", "object": "model"},
        ],
    }


@app.get("/status")
async def status():
    total = sum(_stats.values())
    pct_local = round(
        100 * (
            _stats.get("ollama", 0) + _stats.get("ollama-tools", 0)
            + _stats.get("orchestrate-single", 0) + _stats.get("orchestrate-project", 0)
        ) / total, 1
    ) if total else 0
    orch_total = _orchestration_escalation["total"]
    fully_local_orchestrated_pct = round(
        100 * (orch_total - _orchestration_escalation["escalated"]) / orch_total, 1
    ) if orch_total else None
    return {
        "total_requests": total,
        "routes": dict(_stats),
        "local_pct": pct_local,
        "fully_local_orchestrated_pct": fully_local_orchestrated_pct,
        "config": {
            "orchestrator_mode": ORCHESTRATOR_MODE,
            "ecc_path": ECC_PATH,
            "preferred_model": PREFERRED_MODEL,
            "orchestrator_ceiling": ORCHESTRATOR_CEILING,
            "refinement_threshold": REFINEMENT_THRESHOLD,
            "char_threshold": CHAR_THRESHOLD,
            "intent_routing_mode": INTENT_ROUTING_MODE,
            "intent_routing_live_pct": INTENT_ROUTING_LIVE_PCT,
            "mid_conv_routing_mode": MID_CONV_ROUTING_MODE,
            "mid_conv_live_pct": MID_CONV_LIVE_PCT,
        },
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

_LOCAL_ROUTES = ("ollama", "ollama-tools", "orchestrate-single", "orchestrate-project")


def _load_savings_records() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _local_dt(ts: str):
    try:
        return datetime.fromisoformat(ts).astimezone()
    except (ValueError, TypeError):
        return None


def _timeseries_today() -> dict:
    """Bucket today's (local time) savings.jsonl entries by hour for the dashboard."""
    today = datetime.now().astimezone().date()
    hourly = {h: {r: 0 for r in _LOCAL_ROUTES + ("anthropic",)} for h in range(24)}
    reason_counts: Counter = Counter()
    agent_counts: Counter = Counter()
    ollama_chars = 0
    safe_slice_total = 0
    safe_slice_escalated = 0
    shadow_by_stage: dict[str, dict[str, int]] = {}

    for r in _load_savings_records():
        dt = _local_dt(r.get("ts", ""))
        if dt is None or dt.date() != today:
            continue
        route = r.get("route")

        if route == "shadow-validation":
            stage = r.get("stage", "first-turn")
            bucket = shadow_by_stage.setdefault(stage, {
                "reroutable": 0, "false_negative": 0,
                "reroutable_candidate": 0, "false_negative_candidate": 0,
            })
            if r.get("needs_tool") is False:
                bucket["reroutable"] += 1
                if r.get("false_negative"):
                    bucket["false_negative"] += 1
            if "needs_tool_candidate" in r and r.get("needs_tool_candidate") is False:
                bucket["reroutable_candidate"] += 1
                if r.get("false_negative_candidate"):
                    bucket["false_negative_candidate"] += 1
            continue

        if route in hourly[dt.hour]:
            hourly[dt.hour][route] += 1

        if reason := r.get("reason"):
            reason_counts[reason] += 1
        if agent := r.get("agent_used"):
            agent_counts[agent] += 1
        if route in _LOCAL_ROUTES:
            ollama_chars += r.get("chars", 0)

        reason = str(r.get("reason", ""))
        if route in ("orchestrate-single", "orchestrate-project") and (
            reason.startswith("safe-continuation") or reason.startswith("project-decomp:safe-continuation")
        ):
            safe_slice_total += 1
            if r.get("escalated"):
                safe_slice_escalated += 1

    total_today = sum(sum(h.values()) for h in hourly.values())
    local_today = sum(sum(h[r] for r in _LOCAL_ROUTES) for h in hourly.values())
    pct_local_today = round(100 * local_today / total_today, 1) if total_today else 0

    est_tokens_saved = ollama_chars / 4
    est_cost_saved = est_tokens_saved / 1_000_000 * 3.0

    safe_slice_escalation_rate = (
        round(100 * safe_slice_escalated / safe_slice_total, 1) if safe_slice_total else None
    )

    shadow_summary = {}
    for stage, b in shadow_by_stage.items():
        fnr = round(100 * b["false_negative"] / b["reroutable"], 1) if b["reroutable"] else None
        fnr_c = (
            round(100 * b["false_negative_candidate"] / b["reroutable_candidate"], 1)
            if b["reroutable_candidate"] else None
        )
        shadow_summary[stage] = {
            **b,
            "false_negative_rate": fnr,
            "false_negative_rate_candidate": fnr_c,
        }

    return {
        "date": today.isoformat(),
        "hours": [{"hour": h, **hourly[h]} for h in range(24)],
        "total_requests_today": total_today,
        "local_pct_today": pct_local_today,
        "est_tokens_saved_today": int(est_tokens_saved),
        "est_cost_saved_today": round(est_cost_saved, 4),
        "reasons_today": reason_counts.most_common(10),
        "agents_today": agent_counts.most_common(10),
        "safe_slice_today": {
            "total": safe_slice_total,
            "escalated": safe_slice_escalated,
            "escalation_rate": safe_slice_escalation_rate,
        },
        "shadow_validation_today": shadow_summary,
    }


@app.get("/api/timeseries")
async def api_timeseries():
    return _timeseries_today()


@app.get("/dashboard")
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return Response(content=html_path.read_text(), media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    port = cfg.get("port", 8082)
    print(f"Ollama proxy starting on http://localhost:{port}")
    print(f"  Orchestrator mode : {ORCHESTRATOR_MODE}")
    if ORCHESTRATOR_MODE:
        print(f"  ECC path          : {ECC_PATH}")
        print(f"  Orchestrator ceil : {ORCHESTRATOR_CEILING} chars")
        print(f"  Refinement thresh : {REFINEMENT_THRESHOLD} chars")
    else:
        print(f"  Char threshold    : {CHAR_THRESHOLD}")
        print(f"  Route haiku→Ollama: {ROUTE_HAIKU}")
    print(f"  Preferred model   : {PREFERRED_MODEL}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
