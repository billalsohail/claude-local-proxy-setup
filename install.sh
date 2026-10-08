#!/usr/bin/env bash
# Installs RTK (Claude Code token-saving CLI hook) + ollama-proxy (local
# Ollama routing for cheap/tool-free Claude Code requests) on this machine.
#
# Usage: ./install.sh [--with-models] [--with-ecc] [--yes]
#   --with-models   pull the Ollama models the proxy routes to (multi-GB)
#   --with-ecc      add the everything-claude-code submodule + enable
#                   orchestrator mode in the proxy config
#   --yes           skip confirmation prompts
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_DIR="$REPO_DIR/proxy"
PLIST_LABEL="com.ollama-proxy.local"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
SETTINGS_PATH="$HOME/.claude/settings.json"

WITH_MODELS=0
WITH_ECC=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --with-models) WITH_MODELS=1 ;;
    --with-ecc) WITH_ECC=1 ;;
    --yes) ASSUME_YES=1 ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

confirm() {
  local prompt="$1"
  if [ "$ASSUME_YES" -eq 1 ]; then
    return 0
  fi
  read -r -p "$prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

echo "==> Preflight checks"
if [[ "$(uname)" != "Darwin" ]]; then
  echo "This installer only supports macOS." >&2
  exit 1
fi
if ! command -v brew &>/dev/null; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi

echo "==> Installing Homebrew dependencies (rtk, jq, ollama)"
brew list rtk &>/dev/null || brew install rtk
brew list jq &>/dev/null || brew install jq
brew list ollama &>/dev/null || brew install ollama

echo "==> Resolving a Python 3.10+ interpreter"
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  fi
done
if [ -z "$PYTHON" ]; then
  echo "    No Python 3.10+ found, installing python@3.11 via Homebrew..."
  brew install python@3.11
  PYTHON="$(brew --prefix python@3.11)/bin/python3.11"
fi
echo "    Using $PYTHON"

echo "==> Setting up proxy config"
if [ ! -f "$PROXY_DIR/config.json" ]; then
  cp "$PROXY_DIR/config.example.json" "$PROXY_DIR/config.json"
  echo "    Created $PROXY_DIR/config.json from config.example.json"
else
  echo "    $PROXY_DIR/config.json already exists, leaving it as-is"
fi

echo "==> Installing proxy Python dependencies"
"$PYTHON" -m pip install -q -r "$PROXY_DIR/requirements.txt"

echo "==> Rendering launchd plist"
sed -e "s|__PYTHON__|$PYTHON|g" \
    -e "s|__PROXY_DIR__|$PROXY_DIR|g" \
    -e "s|__HOME__|$HOME|g" \
    "$PROXY_DIR/com.ollama-proxy.local.plist.template" > "$PLIST_PATH"
mkdir -p "$HOME/.claude/ollama-proxy"

echo "==> Loading the proxy service"
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

if [ "$WITH_MODELS" -eq 1 ]; then
  echo "==> Starting Ollama and pulling models (qwen3.5, mistral)"
  if confirm "This will download multiple GB of model weights. Continue?"; then
    brew services start ollama
    sleep 2
    ollama pull qwen3.5:latest
    ollama pull mistral:latest
  else
    echo "    Skipped model pull. Run manually later: ollama pull qwen3.5:latest && ollama pull mistral:latest"
  fi
else
  echo "==> Skipping Ollama model pull (pass --with-models to fetch qwen3.5 + mistral)"
fi

echo "==> Installing RTK hook"
mkdir -p "$HOME/.claude/hooks"
cp "$REPO_DIR/rtk/hooks/rtk-rewrite.sh" "$HOME/.claude/hooks/rtk-rewrite.sh"
chmod +x "$HOME/.claude/hooks/rtk-rewrite.sh"
cp "$REPO_DIR/rtk/RTK.md" "$HOME/.claude/RTK.md"

echo "==> Merging the RTK hook into ~/.claude/settings.json"
"$PYTHON" "$REPO_DIR/claude-settings/merge_hooks.py" "$SETTINGS_PATH" "$HOME/.claude/hooks/rtk-rewrite.sh"

if [ "$WITH_ECC" -eq 1 ]; then
  echo "==> Setting up everything-claude-code submodule"
  if [ ! -d "$REPO_DIR/ecc/.git" ] && [ ! -f "$REPO_DIR/.gitmodules" ]; then
    (cd "$REPO_DIR" && git submodule add https://github.com/affaan-m/everything-claude-code.git ecc)
  else
    (cd "$REPO_DIR" && git submodule update --init --recursive)
  fi
  mkdir -p "$REPO_DIR/ecc/agents" "$REPO_DIR/ecc/skills/ollama-router"
  cp "$REPO_DIR/ecc-overlay/agents/local-drafter.md" "$REPO_DIR/ecc/agents/local-drafter.md"
  cp "$REPO_DIR/ecc-overlay/skills/ollama-router/SKILL.md" "$REPO_DIR/ecc/skills/ollama-router/SKILL.md"

  "$PYTHON" - "$PROXY_DIR/config.json" "$REPO_DIR/ecc" <<'PYEOF'
import json, sys
config_path, ecc_path = sys.argv[1], sys.argv[2]
data = json.loads(open(config_path).read())
data["ecc_path"] = ecc_path
data["orchestrator_mode"] = True
open(config_path, "w").write(json.dumps(data, indent=2) + "\n")
print(f"    Set ecc_path={ecc_path!r}, orchestrator_mode=true in {config_path}")
PYEOF
  echo "    Restart the proxy for the config change to take effect: launchctl kickstart -k gui/$(id -u)/${PLIST_LABEL}"
else
  echo "==> Skipping ECC orchestrator setup (pass --with-ecc to enable)"
fi

PORT="$("$PYTHON" -c "import json; print(json.load(open('$PROXY_DIR/config.json'))['port'])")"

cat <<SUMMARY

==================================================================
 Install complete
==================================================================
 Proxy service : $PLIST_LABEL (launchctl list | grep ollama-proxy)
 Status check  : curl http://localhost:${PORT}/status
 Routing stats : bash "$PROXY_DIR/stats.sh"
 Logs          : tail -f ~/.claude/ollama-proxy/proxy.log
 Config        : $PROXY_DIR/config.json (edit, then:
                 launchctl kickstart -k gui/\$(id -u)/${PLIST_LABEL})

 ANTHROPIC_BASE_URL was NOT set globally. To route a specific Claude
 Code session or project through the proxy, use one of:

   # one-off, current shell only
   ANTHROPIC_BASE_URL=http://localhost:${PORT} claude

   # always-on for one project: add to that project's .claude/settings.json
   {
     "env": { "ANTHROPIC_BASE_URL": "http://localhost:${PORT}" }
   }
==================================================================
SUMMARY
