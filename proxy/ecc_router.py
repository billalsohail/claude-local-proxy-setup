"""
ECC agent routing: detects task type from messages and loads ECC agent system prompts.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool tier classification
#
# LOCAL_TOOLS: read-only tools with no side effects — safe for qwen3.5.
# Any tool not in this set is considered "remote" and must go to Anthropic.
# ---------------------------------------------------------------------------

LOCAL_TOOLS = frozenset({
    "Read",        # reads files, no writes
    "ToolSearch",  # looks up deferred tool schemas — read-only
    "WebSearch",   # web search — read-only
    "WebFetch",    # fetches a URL — read-only
})


def classify_request_tools(body: dict) -> str:
    """
    Classify the tools defined in the request body.
    Returns 'local', 'remote', or 'none'.
    - 'local'  : all defined tools are in LOCAL_TOOLS → safe for qwen3.5
    - 'remote' : at least one tool is not in LOCAL_TOOLS → must use Anthropic
    - 'none'   : no tools defined
    """
    tools = body.get("tools", [])
    if not tools:
        return "none"
    names = {t.get("name") for t in tools if isinstance(t, dict)}
    return "local" if names.issubset(LOCAL_TOOLS) else "remote"


def all_history_tools_are_local(messages: list) -> bool:
    """Return True if every tool_use block in the message history is a local tool."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("name") not in LOCAL_TOOLS:
                        return False
    return True

# ---------------------------------------------------------------------------
# Keyword → ECC agent mapping (order matters: first match wins)
#
# Priority logic:
#   1. High-signal verbs (build/implement/create) fire before technology nouns
#      (database, api) — the BUILD intent overrides the topic.
#   2. Diagnostic verbs (review, refactor, fix) come next.
#   3. Technology-specific reviewers come last: they only fire when the
#      primary intent is about that technology, not incidentally mentioning it.
# ---------------------------------------------------------------------------

_TASK_AGENT_MAP: list[tuple[list[str], str]] = [
    # Tier 1: always-primary intents regardless of topic
    (["security", "vulnerab", "injection", "xss", "csrf", "owasp", "pentest"], "security-reviewer"),
    (["error", " bug ", "crash", "broken", "failing", "exception", "traceback"], "build-error-resolver"),
    # Tier 2: action verbs — what the user wants to DO (beat technology nouns)
    (["build me", "implement a", "create a", "develop a", "make me a", "write me a",
      "i want to build", "i want to create", "i need a"], "code-architect"),
    (["plan ", "roadmap", "breakdown into", "phase", "milestone", "sprint plan"], "planner"),
    (["architect", "system design", "scalab", "design pattern", "microservice"], "architect"),
    (["refactor", "simplif", "clean up", "cleanup", "rewrite", "restructure"], "code-simplifier"),
    (["review", "audit", "code quality", "best practice", "code smell", "lint"], "code-reviewer"),
    (["test", "coverage", "spec", "tdd", "unit test", "integration test"], "tdd-guide"),
    (["document", "readme", "docstring", "changelog"], "doc-updater"),
    # Tier 3: technology-specific (fire when no action verb matched above)
    (["database", " sql ", "schema", "migrat", "quer", "postgres", "mysql", " orm "], "database-reviewer"),
    (["performance", "latency", " slow", "optimi", "bottleneck", "profil", "speed"], "performance-optimizer"),
    (["api", "endpoint", " rest ", "fastapi", "openapi", "swagger"], "fastapi-reviewer"),
    # Tier 4: catch-all
    (["explain", "how does", "understand", "explore", "trace", "walk me through", "what is"], "code-explorer"),
]

DEFAULT_AGENT = "code-explorer"

# Action verbs that indicate Claude needs to USE tools (read/edit files, run commands).
# When tools are defined in the request AND the message contains these, the request
# must go to Anthropic — Ollama cannot execute tool calls.
# Err toward over-inclusion: a false positive here only costs an Anthropic call
# (safe direction); a false negative risks silently returning prose where a real
# tool call was expected (unsafe direction — orchestrator escalation can never
# recover this, since escalation calls never forward tool definitions).
_ACTION_VERBS = [
    "fix ", "fix\n", "fix the", "edit ", "update ", "change ", "modify ",
    "delete ", "remove ", "rename ", "move ", "refactor ", "rewrite ",
    "add ", "append ", "insert ", "implement ", "create the", "create this",
    "write the", "write this", "write a ", "write me", "generate ", "run ", "execute ", "install ",
    "build ", "compile ", "deploy ", "commit ", "push ", "pull ",
    "debug ", "resolve ", "patch ", "revert ", "merge ", "rebase ", "checkout ",
    "stage ", "grep ", "search for", "format ", "lint ", "apply ", "diagnose ",
    "investigate the", "npm ", "pip ", "yarn ", "git ",
]


def _action_verb_scan_window(full_text: str) -> str:
    """
    Claude Code prepends environment context (file listings, README, etc.) before
    the user's actual question, and the instruction itself can also be followed
    by a long pasted context (e.g. "Fix this: <stack trace>"). Scanning the whole
    blob for action verbs picks up unrelated boilerplate (git-workflow docs,
    project instructions, etc.) buried in the middle — check only the leading and
    trailing slice, where the real human ask actually lives, so a verb at either
    end isn't missed but middle-of-context noise doesn't false-trigger.
    """
    if len(full_text) > 400:
        return full_text[:200] + " " + full_text[-400:]
    return full_text


def classify_tool_intent(body: dict) -> tuple[bool, list[str]]:
    """
    Returns (needs_tool_execution, matched_keywords).

    needs_tool_execution is True if the request has tools defined AND the
    message looks like it needs actual tool execution (file edits, bash
    commands, etc.). Conversational questions that happen to have tools
    available return False. matched_keywords is returned for observability
    (logging) even when empty.
    """
    if not body.get("tools"):
        return False, []
    full_text = _extract_last_user_text(body.get("messages", [])).lower()
    text = _action_verb_scan_window(full_text)
    matched = [kw for kw in _ACTION_VERBS if kw in text]
    return bool(matched), matched


def needs_tool_execution(body: dict) -> bool:
    """Back-compat wrapper around classify_tool_intent — see that function."""
    needs, _ = classify_tool_intent(body)
    return needs


# Keywords that signal a project-level decomposition request
_PROJECT_KEYWORDS = [
    "build me", "create a", "implement a", "i want to build", "i want to create",
    "design a", "develop a", "make a", "project", "application", "system",
    "full stack", "end-to-end", "from scratch", "new feature", "new service",
]


def _extract_last_user_text(messages: list) -> str:
    """
    Return the most recent human-authored instruction: the latest user
    message that actually contains free text, skipping over turns that are
    purely tool_result blocks. Mid-conversation (tool-loop) turns often have
    a literally-last user message that's just a tool_result with no text —
    walking back to the last real instruction is what keeps intent
    classification and refine/synthesis prompts meaningful in that case.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if content.strip():
                return content
            continue
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if text.strip():
                return text
    return ""


def last_message_has_fresh_text(messages: list) -> bool:
    """
    True if the very last message (no walk-back, unlike _extract_last_user_text)
    is a user turn carrying real free-text content — i.e. a fresh human ask.
    False when the last message is a bare tool_result (or empty): Claude is
    mid-agentic-loop, deciding what's next, not responding to new input.
    """
    if not messages:
        return False
    msg = messages[-1]
    if msg.get("role") != "user":
        return False
    content = msg.get("content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
            for b in content
        )
    return False


# Small, precision-oriented set of conversational/no-action-needed phrases.
# UNVALIDATED — a first-pass guess, not derived from labeled data. Its only job
# right now is to be shadow-scored (see classify_mid_conversation_intent) so we
# get real accuracy numbers before ever trusting it for live routing.
_CONVERSATIONAL_SIGNALS = [
    "what do you think", "any thoughts", "your thoughts",
    "does that make sense", "does this make sense",
    "why did you", "why does", "why is",
    "what's the difference", "what is the difference",
    "can you explain", "can you clarify",
    "what would you", "how would you",
]


def classify_mid_conversation_intent(messages: list, fresh_turn: bool) -> tuple[bool, list[str]]:
    """
    Mid-conversation intent signal. Two structurally different populations:

    - Tool-loop continuation (fresh_turn=False): always needs_tool=True. The
      last real human text is stale relative to the current step of an
      in-progress agentic loop, so it says nothing useful here. Shadow data:
      94.7% of these actually end in tool_use (n=189).
    - Fresh turn (fresh_turn=True): defaults to needs_tool=False. Shadow data
      on real traffic found 13/13 fresh mid-conversation turns ended in
      end_turn — none needed a tool. A fresh mid-conversation message is
      structurally the same shape as a fresh first-turn ask, so we reuse
      _ACTION_VERBS (already tuned for exactly that shape of text, per its
      own docstring) as an override: an explicit action verb flips the
      prediction back to needs_tool=True, fail-safe toward Anthropic.

      First live iteration of this override scanned the *entire* extracted
      text and cratered coverage to 0% (10/10 real fresh-turn samples matched
      an action verb yet all ended in end_turn) — action verbs buried in
      environment context/docs pasted mid-message (e.g. git-workflow
      boilerplate) were false-triggering it. Now uses the same leading+
      trailing scan window as classify_tool_intent, which exists for exactly
      this reason. Still not itself shadow-validated on a real action-verb-
      near-the-actual-ask case — keep watching the dashboard's candidate
      columns after this ships.
    """
    if not fresh_turn:
        return True, ["continuation"]
    full_text = _extract_last_user_text(messages).lower()
    text = _action_verb_scan_window(full_text)
    action_matched = [kw for kw in _ACTION_VERBS if kw in text]
    if action_matched:
        return True, action_matched
    conversational_matched = [kw for kw in _CONVERSATIONAL_SIGNALS if kw in text]
    return False, conversational_matched or ["fresh-no-action-verb"]


def detect_agent(messages: list) -> str:
    """Keyword-match the last user message to the best ECC agent name."""
    text = _extract_last_user_text(messages).lower()
    if not text:
        return DEFAULT_AGENT
    for keywords, agent in _TASK_AGENT_MAP:
        if any(kw in text for kw in keywords):
            return agent
    return DEFAULT_AGENT


def is_project_request(messages: list) -> bool:
    """
    Heuristic: is this a multi-step project description that warrants decomposition?

    Two tiers:
    - Explicit build/create verbs always trigger decomposition (short or long).
    - Broader project nouns require length (>250 chars) to confirm complexity.
    - Very long multi-paragraph messages always trigger regardless of keywords.
    """
    text = _extract_last_user_text(messages)
    if not text:
        return False
    lower = text.lower()

    # Tier 1: explicit "build me / create a / implement a" always decomposes
    _EXPLICIT = ["build me", "create a", "implement a", "i want to build", "i want to create",
                 "develop a", "make a", "write a", "i need a"]
    if any(kw in lower for kw in _EXPLICIT):
        return True

    # Tier 2: general project nouns need some length to confirm non-trivial scope
    _GENERAL = ["project", "application", "system", "full stack", "end-to-end",
                "from scratch", "new service", "design a", "new feature"]
    if any(kw in lower for kw in _GENERAL) and len(text) > 250:
        return True

    # Tier 3: long multi-paragraph messages imply complexity
    if text.count("\n") >= 2 and len(text) > 400:
        return True

    return False


def load_agent_system_prompt(agent_name: str, ecc_path: str | Path) -> str:
    """
    Read an ECC agent .md file, strip YAML frontmatter, return the body as a system prompt.
    Falls back to a minimal generic prompt if the agent file is not found.
    """
    path = Path(ecc_path) / "agents" / f"{agent_name}.md"
    if not path.exists():
        return f"You are a helpful software engineering assistant specialising in {agent_name.replace('-', ' ')}."

    raw = path.read_text(encoding="utf-8")

    # Strip YAML frontmatter (content between first and second ---)
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            raw = raw[end + 3:].lstrip("\n")

    # Truncate very large agent prompts to keep Ollama context manageable (~4k chars)
    if len(raw) > 4000:
        raw = raw[:4000] + "\n[...truncated for context efficiency]"

    return raw.strip()
