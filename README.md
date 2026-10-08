# claude-local-proxy-setup

Reproduces a Claude Code token-saving setup on a new macOS machine: an RTK
command-rewriting hook, plus a local proxy that routes cheap/tool-free
requests to a local Ollama model instead of the Anthropic API.

## What gets installed

1. **RTK** (`brew install rtk`) — a `PreToolUse`/`Bash` hook
   (`~/.claude/hooks/rtk-rewrite.sh`) that transparently rewrites routine
   shell commands Claude Code runs into token-cheaper equivalents.
2. **ollama-proxy** — a FastAPI reverse proxy that sits in front of the
   Anthropic API. Routing rules, in priority order:

   | Condition | Destination |
   |---|---|
   | Request has `tools` defined | Anthropic |
   | Message history has `tool_use`/`tool_result` | Anthropic |
   | Model is `claude-haiku-*` + no tools | Ollama |
   | Message content < 3000 chars + no tools | Ollama |
   | Everything else | Anthropic |

   It runs as a background `launchd` service (`com.ollama-proxy.local`) on
   `http://localhost:8082` by default, and forwards your real
   `x-api-key`/`authorization` headers through for passthrough requests — no
   separate API key setup needed.
3. **(Optional) ECC orchestrator mode** — multi-model task decomposition
   using agent prompts from the public
   [everything-claude-code](https://github.com/affaan-m/everything-claude-code)
   repo, pulled in as a git submodule plus two local overlay files
   (`local-drafter` agent + `ollama-router` skill) not present upstream.

## Prerequisites

- macOS with [Homebrew](https://brew.sh)
- [Claude Code](https://claude.com/claude-code) installed
- [Ollama](https://ollama.com) models need several GB of disk space each if
  you opt into `--with-models`

## Install

```bash
git clone <this-repo> && cd claude-local-proxy-setup
./install.sh                       # RTK hook + proxy service, Anthropic-only passthrough
./install.sh --with-models         # also pull qwen3.5 + mistral via Ollama
./install.sh --with-ecc            # also set up the ECC orchestrator submodule
./install.sh --with-models --with-ecc --yes   # everything, no prompts
```

The installer:
- Installs `rtk`, `jq`, `ollama` via Homebrew (skips ones already present)
- Resolves a Python 3.10+ interpreter (installs `python@3.11` if none found)
- Generates `proxy/config.json` from `proxy/config.example.json` (git-ignored,
  so your local tuning never gets committed)
- Renders and loads the `launchd` plist, pointed at the proxy files inside
  this cloned repo (no copying elsewhere)
- Copies the RTK hook into `~/.claude/hooks/` and merges **only** the
  `hooks.PreToolUse` Bash-matcher entry into `~/.claude/settings.json` —
  every other key in that file is left untouched

## `ANTHROPIC_BASE_URL` is never set globally

Unlike a naive install, this script never writes `env.ANTHROPIC_BASE_URL`
into `~/.claude/settings.json`. If the proxy is down or misbehaving, every
Claude Code session elsewhere on the machine should keep working. Opt in
per use instead:

```bash
# one-off, current shell only
ANTHROPIC_BASE_URL=http://localhost:8082 claude

# always-on for one project: add to that project's .claude/settings.json
{
  "env": { "ANTHROPIC_BASE_URL": "http://localhost:8082" }
}
```

## Day-to-day

```bash
curl http://localhost:8082/status             # live routing counters
bash proxy/stats.sh                           # routing breakdown + cost estimate
tail -f ~/.claude/ollama-proxy/proxy.log      # service logs
tail -f ~/.claude/ollama-proxy/savings.jsonl  # per-request routing log

# restart after editing proxy/config.json
launchctl kickstart -k gui/$(id -u)/com.ollama-proxy.local
```

## Uninstall

```bash
./uninstall.sh
```

Stops and removes the `launchd` service and the RTK hook files. It
intentionally does **not** auto-edit `~/.claude/settings.json` — it prints
the one hook entry to remove by hand if you want it fully gone, since that
file may contain other, unrelated configuration.
