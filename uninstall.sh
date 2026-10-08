#!/usr/bin/env bash
# Reverses the parts of install.sh that touch system/global state:
# stops + removes the launchd service, removes the RTK hook files.
# Does NOT edit ~/.claude/settings.json automatically — prints the
# one hook entry to remove by hand, since settings.json may have
# other unrelated content you don't want touched automatically.
set -euo pipefail

PLIST_LABEL="com.ollama-proxy.local"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"

echo "==> Stopping and removing the proxy service"
launchctl unload "$PLIST_PATH" 2>/dev/null || true
rm -f "$PLIST_PATH"

echo "==> Removing RTK hook files"
rm -f "$HOME/.claude/hooks/rtk-rewrite.sh"
rm -f "$HOME/.claude/RTK.md"

cat <<SUMMARY

Done. Not removed automatically (edit by hand if you want them gone too):

  - The "hooks.PreToolUse" entry in ~/.claude/settings.json pointing at
    ~/.claude/hooks/rtk-rewrite.sh
  - Homebrew packages: rtk, jq, ollama
  - Any Ollama models you pulled (ollama rm qwen3.5:latest mistral:latest)
  - ~/.claude/ollama-proxy/ logs
  - This cloned repo itself
SUMMARY
