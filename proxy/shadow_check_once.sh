#!/usr/bin/env bash
# One-shot check of the intent-routing shadow-mode rollout. Appends stats.sh
# output to a log file, then removes its own crontab entry so it never fires
# again. Scheduled once via `crontab` (not Claude's session-scoped cron) so it
# survives Claude sessions ending. Marker comment below is how it finds and
# deletes itself.
PROXY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HOME/.claude/ollama-proxy/shadow_check.log"

{
  echo "===== Shadow-routing check: $(date) ====="
  bash "$PROXY_DIR/stats.sh"
  echo
} >> "$LOG" 2>&1

crontab -l 2>/dev/null | grep -v "ollama-proxy-shadow-check-once" | crontab -
