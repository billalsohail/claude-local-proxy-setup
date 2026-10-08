"""
Two-mode orchestration for the Ollama proxy.

Mode A (orchestrate_single):
  Ollama drafts with an ECC agent persona.
  Short draft (<= refinement_threshold chars) → return directly (zero Claude tokens).
  Long draft → Claude refines and streams the improved answer.

Mode B (orchestrate_project):
  Claude decomposes the request into subtasks (JSON).
  Parallel Ollama calls per subtask with matched ECC agent personas.
  Claude synthesizes all results and streams the final answer.
  Falls back to Mode A if decomposition fails.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import AsyncGenerator, NamedTuple

import httpx

from ecc_router import detect_agent, load_agent_system_prompt, _extract_last_user_text


class OllamaDraft(NamedTuple):
    content: str
    done_reason: str

# Internal fallback/refine calls (decomposition parse failure, synthesis too
# short, orchestrate_single's refine step) are small mechanical tasks — JSON
# extraction, text merge/cleanup — that don't need the session's main model.
FALLBACK_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DECOMP_SYSTEM = (
    "You are a task decomposition specialist. Break down project requests into subtasks "
    "and return valid JSON only — no markdown fences, no explanation."
)

_DECOMP_USER_TEMPLATE = """\
Break the following request into 3-5 concrete, independent subtasks.
For each subtask pick the single most appropriate agent from:
code-reviewer, code-explorer, architect, code-architect, code-simplifier,
build-error-resolver, security-reviewer, performance-optimizer,
database-reviewer, fastapi-reviewer, doc-updater, tdd-guide

Return ONLY a JSON array:
[{{"task": "...", "agent": "agent-name"}}, ...]

PROJECT REQUEST:
{user_message}
"""

_REFINE_USER_TEMPLATE = """\
A local model produced the draft below. Review it for accuracy and completeness, \
correct any errors, and return the improved final answer only \
(no commentary about the draft).

ORIGINAL REQUEST:
{user_message}

LOCAL DRAFT:
{draft}
"""

_SYNTHESIS_USER_TEMPLATE = """\
Specialist agents completed the following subtasks. Synthesise their outputs into one \
coherent, well-structured final answer. Resolve conflicts, fill gaps, add missing details.

ORIGINAL REQUEST:
{user_message}

AGENT OUTPUTS:
{outputs}
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _last_user_text(body: dict) -> str:
    """The operative human instruction driving this request — see
    ecc_router._extract_last_user_text for why this isn't just "the last
    user message" (that's often a bare tool_result mid-tool-loop)."""
    return _extract_last_user_text(body.get("messages", []))


def _render_block(block: dict) -> str:
    """
    Render a single content block as plain text for Ollama, which gets no
    native tool schema here (orchestrate_single/project are pure prose
    drafting). tool_use/tool_result carry the actual information a
    mid-conversation draft needs (e.g. file contents from a prior Read) —
    dropping them silently would leave the draft uninformed about exactly
    the context a read-only-tool-history turn is continuing from.
    """
    btype = block.get("type")
    if btype == "text":
        return block.get("text", "")
    if btype == "tool_use":
        return f"[Called {block.get('name', '')}({json.dumps(block.get('input', {}))})]"
    if btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        elif not isinstance(content, str):
            content = str(content)
        return f"[Tool result]\n{content}"
    return ""


def _anthropic_to_ollama_msgs(body: dict, max_messages: int = 6) -> list[dict]:
    """
    Convert Anthropic messages to Ollama format, keeping only the last N messages.
    Ollama has a limited context window (default ~4k tokens for most models), so
    sending the full 100k+ history would either be truncated silently or cause errors.
    The last few exchanges contain all the relevant context for drafting.
    """
    msgs = []
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            text = "\n".join(
                rendered for b in content
                if isinstance(b, dict) and (rendered := _render_block(b))
            )
            if text:
                msgs.append({"role": role, "content": text})
    return msgs[-max_messages:]


async def _ollama_chat(
    ollama_base: str,
    model: str,
    messages: list[dict],
    system_prompt: str,
    max_tokens: int,
    think: bool = False,
    timeout: int = 120,
) -> OllamaDraft:
    all_msgs = (
        [{"role": "system", "content": system_prompt}] + messages
        if system_prompt else messages
    )
    # Qwen3 family (and other thinking models) spend tokens on internal reasoning before
    # producing output; the client's max_tokens is a Claude budget, not Ollama's.
    # Ensure enough headroom for thinking + actual answer.
    num_predict = max(max_tokens, 2000)
    payload = {
        "model": model,
        "messages": all_msgs,
        "stream": False,
        "think": think,
        "options": {"num_predict": num_predict},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{ollama_base}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        done_reason = data.get("done_reason", "")
        return OllamaDraft(content=content, done_reason=done_reason)


def _build_claude_body(body: dict, user_content: str) -> dict:
    """Minimal Claude request body for a refinement/synthesis call. Always
    uses FALLBACK_MODEL — these are internal cleanup calls, not the main task."""
    return {
        "model": FALLBACK_MODEL,
        "max_tokens": body.get("max_tokens", 4096),
        "messages": [{"role": "user", "content": user_content}],
        "stream": True,
    }


async def _stream_anthropic(
    body: dict, headers: dict, anthropic_base: str
) -> AsyncGenerator[bytes, None]:
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", f"{anthropic_base}/v1/messages",
            json=body, headers=headers,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk


async def _synthetic_sse(text: str, original_model: str) -> AsyncGenerator[bytes, None]:
    """Emit an Anthropic-compatible SSE stream for a plain-text Ollama-only response."""
    msg_id = f"msg_ollama_{uuid.uuid4().hex[:16]}"

    def evt(name: str, data: dict) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

    yield evt("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": original_model,
            "stop_reason": None, "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield evt("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    yield evt("ping", {"type": "ping"})

    chunk_size = 120
    for i in range(0, len(text), chunk_size):
        yield evt("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text[i:i + chunk_size]},
        })

    yield evt("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield evt("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": len(text.split())},
    })
    yield evt("message_stop", {"type": "message_stop"})


def _parse_subtasks(raw: str, max_subtasks: int) -> list[dict] | None:
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        tasks = json.loads(match.group())
        if not isinstance(tasks, list) or len(tasks) < 2:
            return None
        valid = [t for t in tasks if isinstance(t, dict) and "task" in t and "agent" in t]
        return valid[:max_subtasks] if valid else None
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def orchestrate_single(
    body: dict,
    forward_headers: dict,
    ollama_base: str,
    preferred_model: str,
    ecc_path: str,
    anthropic_base: str,
    refinement_threshold: int,
    ollama_think: bool = False,
    min_viable_draft_chars: int = 20,
) -> tuple[AsyncGenerator[bytes, None], str, bool]:
    """
    Returns (stream, agent_name, escalated).
    stream is an async bytes generator suitable for StreamingResponse.
    escalated is True if Claude tokens were spent (refine step used).
    """
    messages = body.get("messages", [])
    agent_name = detect_agent(messages)
    system_prompt = load_agent_system_prompt(agent_name, ecc_path)
    ollama_msgs = _anthropic_to_ollama_msgs(body)
    max_tokens = body.get("max_tokens", 4096)

    draft = await _ollama_chat(
        ollama_base=ollama_base,
        model=preferred_model,
        messages=ollama_msgs,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        think=ollama_think,
    )

    # Trust the draft only if it completed normally, isn't degenerate, and
    # isn't so long it warrants a second look — otherwise escalate to Claude.
    escalate = (
        draft.done_reason != "stop"
        or len(draft.content.strip()) < min_viable_draft_chars
        or len(draft.content) > refinement_threshold
    )
    if not escalate:
        return _synthetic_sse(draft.content, body.get("model", "unknown")), agent_name, False

    # Draft untrustworthy or too long → Claude refines
    user_text = _last_user_text(body)
    refine_content = _REFINE_USER_TEMPLATE.format(user_message=user_text, draft=draft.content)
    claude_body = _build_claude_body(body, refine_content)
    stream = _stream_anthropic(claude_body, dict(forward_headers), anthropic_base)
    return stream, agent_name, True


async def orchestrate_project(
    body: dict,
    forward_headers: dict,
    ollama_base: str,
    preferred_model: str,
    ecc_path: str,
    anthropic_base: str,
    max_subtasks: int,
    refinement_threshold: int,
    ollama_think: bool = False,
    decomp_max_tokens: int = 768,
    min_viable_draft_chars: int = 20,
) -> tuple[AsyncGenerator[bytes, None], str, bool, int]:
    """
    Returns (stream, reason_tag, escalated, subtask_count).
    escalated is True if Claude tokens were spent anywhere in this request
    (decomposition or synthesis fallback). Falls back to orchestrate_single
    (subtask_count=0) if decomposition fails entirely, both locally and via Claude.
    """
    user_text = _last_user_text(body)
    max_tokens = body.get("max_tokens", 4096)

    # Step 1a: try decomposition locally first (free)
    decomp_user = _DECOMP_USER_TEMPLATE.format(user_message=user_text)
    try:
        decomp_draft = await _ollama_chat(
            ollama_base=ollama_base,
            model=preferred_model,
            messages=[{"role": "user", "content": decomp_user}],
            system_prompt=_DECOMP_SYSTEM,
            max_tokens=decomp_max_tokens,
            think=ollama_think,
        )
        decomp_raw = decomp_draft.content
    except Exception:
        decomp_raw = ""

    subtasks = _parse_subtasks(decomp_raw, max_subtasks) if decomp_raw else None
    decomp_escalated = False

    # Step 1b: local decomposition failed to parse → fall back to Claude
    if not subtasks:
        decomp_escalated = True
        decomp_body = {
            "model": FALLBACK_MODEL,
            "max_tokens": 512,
            "system": _DECOMP_SYSTEM,
            "messages": [{"role": "user", "content": decomp_user}],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{anthropic_base}/v1/messages",
                    json=decomp_body,
                    headers=dict(forward_headers),
                )
                resp.raise_for_status()
                decomp_raw = resp.json().get("content", [{}])[0].get("text", "")
        except Exception:
            decomp_raw = ""
        subtasks = _parse_subtasks(decomp_raw, max_subtasks) if decomp_raw else None

    # Both local and Claude decomposition failed → fall back to Mode A
    if not subtasks:
        stream, agent, escalated = await orchestrate_single(
            body, forward_headers, ollama_base, preferred_model,
            ecc_path, anthropic_base, refinement_threshold,
            ollama_think=ollama_think, min_viable_draft_chars=min_viable_draft_chars,
        )
        return stream, agent, escalated, 0

    # Step 2: Parallel Ollama subtask execution (always local)
    async def run_subtask(subtask: dict) -> str:
        agent_name = subtask.get("agent", "code-explorer")
        task_text = subtask.get("task", "")
        sp = load_agent_system_prompt(agent_name, ecc_path)
        try:
            result = await _ollama_chat(
                ollama_base=ollama_base,
                model=preferred_model,
                messages=[{"role": "user", "content": task_text}],
                system_prompt=sp,
                max_tokens=min(max_tokens, 1024),
                think=ollama_think,
            )
            return f"[{agent_name}]\nTask: {task_text}\n\n{result.content}"
        except Exception as exc:
            return f"[{agent_name}]\nTask: {task_text}\n\n(subtask failed: {exc})"

    results = await asyncio.gather(*[run_subtask(t) for t in subtasks])
    aggregated = "\n\n---\n\n".join(results)

    # Step 3a: try synthesis locally first (free)
    synthesis_content = _SYNTHESIS_USER_TEMPLATE.format(
        user_message=user_text,
        outputs=aggregated,
    )
    try:
        local_synthesis = await _ollama_chat(
            ollama_base=ollama_base,
            model=preferred_model,
            messages=[{"role": "user", "content": synthesis_content}],
            system_prompt="",
            max_tokens=max_tokens,
            think=ollama_think,
        )
    except Exception:
        local_synthesis = OllamaDraft(content="", done_reason="error")

    # Local synthesis completed normally and isn't degenerate → return directly,
    # zero Claude tokens. (Completeness, not length, is what matters here — a
    # short-but-finished synthesis is still trustworthy.)
    synth_escalate = (
        local_synthesis.done_reason != "stop"
        or len(local_synthesis.content.strip()) < min_viable_draft_chars
    )
    if not synth_escalate:
        stream = _synthetic_sse(local_synthesis.content, body.get("model", "unknown"))
        return stream, "project-decomp", decomp_escalated, len(subtasks)

    # Step 3b: local synthesis incomplete/degenerate → Claude synthesizes
    claude_body = _build_claude_body(body, synthesis_content)
    stream = _stream_anthropic(claude_body, dict(forward_headers), anthropic_base)
    return stream, "project-decomp", True, len(subtasks)
