#!/usr/bin/env python3
"""
Idempotently merge the RTK PreToolUse/Bash hook into ~/.claude/settings.json.

Only ever touches hooks.PreToolUse: it appends one matcher/hook entry if an
entry pointing at the same command isn't already present, and never touches
any other key (in particular, never writes env.ANTHROPIC_BASE_URL).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HOOK_MATCHER = "Bash"


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: merge_hooks.py <settings.json path> <hook command path>", file=sys.stderr)
        sys.exit(1)

    settings_path = Path(sys.argv[1])
    hook_command = sys.argv[2]

    data = json.loads(settings_path.read_text()) if settings_path.exists() else {}

    hooks = data.setdefault("hooks", {})
    pre_tool_use = hooks.setdefault("PreToolUse", [])

    for entry in pre_tool_use:
        if entry.get("matcher") != HOOK_MATCHER:
            continue
        for hook in entry.get("hooks", []):
            if hook.get("command") == hook_command:
                print(f"  Hook already present in {settings_path}, nothing to do.")
                return

    pre_tool_use.append(
        {
            "matcher": HOOK_MATCHER,
            "hooks": [{"type": "command", "command": hook_command}],
        }
    )

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  Added RTK PreToolUse/Bash hook to {settings_path}")


if __name__ == "__main__":
    main()
