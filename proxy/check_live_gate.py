#!/usr/bin/env python3
"""
Follow-up check for the mid-conversation safe-continuation live gate
(fixed in commit 8934f76, deployed 2026-09-21 ~08:15 UTC — see
project_orchestrator.md memory). Run via a launchd one-time job (see
com.ollama-proxy.local.live-gate-check.plist) so it doesn't depend on
any particular Claude Code session still being alive.

Before this fix, `safe-continuation` had fired zero times ever — the
live gate checked the old, broken `needs_tool` heuristic instead of
the new `needs_tool_candidate`. This check reports, restricted to
entries logged since the fix went live:

  - how many times safe-continuation actually fired (live coverage)
  - of those, how many were escalated back to Anthropic by
    orchestrate_single's safety net (draft quality escalation, not a
    missed-tool-call detector — see plan/memory caveat)
  - candidate accuracy/coverage/false-negative rate on the full
    mid-conversation shadow-scored population in the same window, for
    comparison against the pre-fix numbers (91.6% / 3.0% / 0.0% on 202
    pre-fix samples)

Writes a dated report to ~/.claude/ollama-proxy/ and fires a macOS
notification with the headline numbers.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path.home() / ".claude" / "ollama-proxy" / "savings.jsonl"
REPORT_DIR = Path.home() / ".claude" / "ollama-proxy"

# Live-gate fix (commit 8934f76) was deployed and the daemon restarted
# at this UTC timestamp — only count traffic since then.
DEPLOY_CUTOFF = "2026-09-21T08:15:00+00:00"


def load_entries_since_deploy() -> list[dict]:
    entries = []
    if not LOG_PATH.exists():
        return entries
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts")
            if not ts or ts < DEPLOY_CUTOFF:
                continue
            entries.append(rec)
    return entries


def score_candidate(shadow_entries: list[dict]) -> dict:
    n = len(shadow_entries)
    if n == 0:
        return {"n": 0}
    would_reroute = [e for e in shadow_entries if not e.get("needs_tool_candidate")]
    correct = sum(
        1
        for e in shadow_entries
        if e.get("needs_tool_candidate") == (e.get("actual_stop_reason") == "tool_use")
    )
    fn = sum(1 for e in would_reroute if e.get("false_negative_candidate"))
    return {
        "n": n,
        "coverage": len(would_reroute) / n * 100,
        "accuracy": correct / n * 100,
        "reroutable": len(would_reroute),
        "false_negative": fn,
        "false_negative_rate": (fn / len(would_reroute) * 100) if would_reroute else None,
    }


def main() -> None:
    entries = load_entries_since_deploy()
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    lines = [
        f"Mid-conversation live-gate follow-up check — {date_str}",
        f"(live gate fixed in commit 8934f76, deployed {DEPLOY_CUTOFF})",
        f"Window: all savings.jsonl entries with ts >= {DEPLOY_CUTOFF}",
        "",
    ]

    if not entries:
        lines.append(
            "No entries logged since the fix was deployed. Either the proxy hasn't "
            "handled any requests, or savings.jsonl is missing/rotated."
        )
    else:
        live_fires = [
            e for e in entries
            if e.get("route") in ("orchestrate-single", "orchestrate-project")
            and isinstance(e.get("reason"), str)
            and "safe-continuation" in e["reason"]
        ]
        escalated = [e for e in live_fires if e.get("escalated")]

        shadow_entries = [
            e for e in entries
            if e.get("route") == "shadow-validation"
            and e.get("stage") == "mid-conversation"
            and "needs_tool_candidate" in e
        ]
        mid_conv_total = [
            e for e in entries
            if e.get("route") == "anthropic"
            and isinstance(e.get("reason"), str)
            and e["reason"].startswith("tool-content-in-history")
        ]
        mid_conv_population = len(mid_conv_total) + len(live_fires) + len(shadow_entries)

        cand = score_candidate(shadow_entries)

        lines += [
            f"Total log entries since deploy: {len(entries)}",
            "",
            "-- Live gate (the actual fix) --",
            f"  safe-continuation live reroutes: {len(live_fires)}",
            f"  of those, escalated back to Anthropic (draft-quality safety net): "
            f"{len(escalated)}"
            + (f" ({len(escalated)/len(live_fires)*100:.0f}%)" if live_fires else ""),
            f"  mid-conversation population seen (anthropic + live-fires + shadow): "
            f"{mid_conv_population}",
            "",
            "-- Candidate signal accuracy (shadow-scored population, this window) --",
        ]

        if cand["n"] == 0:
            lines.append("  No shadow-validation samples in this window yet.")
        else:
            fn_str = "n/a" if cand["false_negative_rate"] is None else f"{cand['false_negative_rate']:.1f}%"
            lines += [
                f"  n = {cand['n']}",
                f"  accuracy: {cand['accuracy']:.1f}%",
                f"  coverage (would reroute): {cand['coverage']:.1f}%",
                f"  false-negative rate: {fn_str}",
                "",
                "  For comparison, pre-fix (2026-09-20 check, n=202): "
                "accuracy 91.6%, coverage 3.0%, false-negative rate 0.0%.",
            ]

        lines.append("")
        if not live_fires:
            lines.append(
                "safe-continuation has not fired at all in this window — either too "
                "little mid-conversation traffic yet, or the safe_slice intersection "
                "(read-only tool history) remains too rare. Consider whether it's "
                "time to revisit loosening safe_slice (see memory: the ~1.5% "
                "safe_slice rate, not classifier accuracy, is the real bottleneck)."
            )
        elif escalated and len(escalated) / len(live_fires) > 0.5:
            lines.append(
                "More than half of live safe-continuation reroutes are escalating "
                "back to Anthropic — Ollama's local drafts aren't landing cleanly "
                "for this slice. Worth a closer look before expanding scope."
            )
        else:
            lines.append(
                "Gate is firing and mostly not escalating — looks healthy. Review "
                "with Claude Code if considering loosening safe_slice further."
            )

    report_text = "\n".join(lines)
    report_path = REPORT_DIR / f"live_gate_report_{date_str}.txt"
    report_path.write_text(report_text + "\n")

    if entries:
        fires = sum(
            1 for e in entries
            if e.get("route") in ("orchestrate-single", "orchestrate-project")
            and isinstance(e.get("reason"), str)
            and "safe-continuation" in e["reason"]
        )
        headline = f"safe-continuation fired {fires}x since deploy — see {report_path.name}"
    else:
        headline = f"no traffic since deploy yet — see {report_path.name}"

    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{headline}" with title "Ollama Proxy — live gate follow-up"',
            ],
            check=False,
        )
    except FileNotFoundError:
        pass

    print(report_text)


if __name__ == "__main__":
    main()
