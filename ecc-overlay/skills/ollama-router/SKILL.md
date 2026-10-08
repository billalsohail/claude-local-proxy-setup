---
name: ollama-router
description: >
  Multi-model orchestration for reducing Anthropic API costs. Routes simple text-generation
  subtasks to local Ollama models (qwen3.5/mistral) via a transparent proxy, while keeping
  architecture decisions, reasoning, and all tool-use on Claude/Anthropic.
  Activate when starting a complex project build to enable automatic cost-saving routing.
origin: ECC
---

# Ollama Router — Multi-Model Orchestration

Routes work between local Ollama models and Anthropic based on task complexity. The proxy
(`proxy.py`, inside the `claude-local-proxy-setup/proxy/` clone) runs in the background and intercepts Claude Code API calls.

## How It Works

```
User prompt
    │
    ▼
Claude (Anthropic) — orchestrator for all reasoning + tool use
    │
    ├── delegates text-generation subtasks ──► local-drafter agent
    │   (model: haiku, tools: [])                    │
    │   proxy intercepts → Ollama (qwen3.5) ◄────────┘
    │   returns raw text
    │
    └── uses Write/Edit/Bash for file ops → Anthropic (tool use required)
```

**Proxy routing rules (auto-applied, no action needed):**

| Condition | Destination |
|---|---|
| Request has `tools` defined | Anthropic |
| Message history has `tool_use`/`tool_result` | Anthropic |
| Model is `claude-haiku-*` + no tools | Ollama |
| Context < 3000 chars + no tools | Ollama |
| Everything else | Anthropic |

## When to Delegate to `local-drafter`

Invoke the `local-drafter` agent for these subtasks:

**Good candidates for local:**
- Draft a README or CONTRIBUTING guide
- Generate boilerplate file content (requirements.txt, .gitignore, Dockerfile, pyproject.toml)
- Write docstrings or inline comments for a set of functions
- Produce a directory scaffold outline
- Answer simple factual questions mid-task ("what's the FastAPI lifespan pattern?")
- Generate example data, fixtures, or test stubs from a schema

**Keep on Anthropic:**
- Architecture and design decisions
- Any task that reads existing code or files
- Security-sensitive code (auth, crypto, secret handling)
- Debugging and root-cause analysis
- Code review and correctness checks
- Anything requiring multi-step reasoning

## Delegation Pattern

When breaking down a complex task, frame subtasks for `local-drafter` as self-contained
text-generation prompts:

```
# Good delegation prompt for local-drafter:
"Generate a complete pyproject.toml for a FastAPI project named 'myapp' using Python 3.12,
with dependencies: fastapi, uvicorn, sqlalchemy, pydantic-settings, pytest, httpx."

# Not suitable for local-drafter (needs file reading):
"Update the existing pyproject.toml to add these dependencies."
```

## Workflow Example: New Project

```
User: "Create a FastAPI project with JWT auth, PostgreSQL, and full tests"

1. Claude (Anthropic) — plans architecture, identifies components
2. Claude invokes local-drafter: "Generate pyproject.toml for FastAPI+SQLAlchemy+JWT" → Ollama
3. Claude invokes local-drafter: "Draft README for this FastAPI project" → Ollama
4. Claude invokes local-drafter: "Generate .gitignore for Python/FastAPI" → Ollama
5. Claude uses Write tool to create files from drafter output → Anthropic
6. Claude writes auth logic (security-sensitive) → Anthropic
7. Claude uses Bash to run tests → Anthropic
```

Steps 2–4 cost zero Anthropic tokens.

## Checking Savings

```bash
bash /path/to/claude-local-proxy-setup/proxy/stats.sh   # routing breakdown + cost estimate
curl http://localhost:8082/status       # live request counters
tail -f ~/.claude/ollama-proxy/savings.jsonl   # per-request log
```

## Proxy Management

```bash
# Status
launchctl list | grep ollama-proxy

# Restart
launchctl kickstart -k gui/$(id -u)/com.ollama-proxy.local

# View logs
tail -f ~/.claude/ollama-proxy/proxy.log
tail -f ~/.claude/ollama-proxy/proxy-error.log

# Temporarily bypass proxy (current shell only)
unset ANTHROPIC_BASE_URL
```

## Configuration

Edit `config.json` in the `claude-local-proxy-setup/proxy/` clone:

```json
{
  "preferred_model": "qwen3.5:latest",   // local model for text generation
  "fallback_model": "mistral:latest",    // used if qwen3.5 fails
  "char_threshold": 3000,                // chars below this → Ollama
  "route_haiku_to_ollama": true          // haiku+no-tools → Ollama
}
```

Restart the proxy after changes: `launchctl kickstart -k gui/$(id -u)/com.ollama-proxy.local`
