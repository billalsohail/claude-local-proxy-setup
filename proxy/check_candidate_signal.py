#!/usr/bin/env python3
"""
One-shot follow-up check for the mid-conversation candidate signal
(classify_mid_conversation_intent, shadow-only since commit d5fdedd,
2026-09-18). Run via a launchd one-time job (see
com.ollama-proxy.local.candidate-check.plist) so it doesn't depend on any
particular Claude Code session still being alive.

Reads ~/.claude/ollama-proxy/savings.jsonl, restricts to mid-conversation
shadow-validation entries that carry the candidate fields, and reports:
  - coverage: how often the candidate would actually reroute
    (needs_tool_candidate == False)
  - accuracy: how often its prediction matched actual_stop_reason
  - false-negative rate among its "would reroute" predictions
  - the same three numbers for the old classify_tool_intent heuristic,
    restricted to the identical entries, for a fair side-by-side
  - the trivial "always assume tool needed" baseline accuracy

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


def load_candidate_entries() -> list[dict]:
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
            if rec.get("route") != "shadow-validation":
                continue
            if rec.get("stage") != "mid-conversation":
                continue
            if "needs_tool_candidate" not in rec:
                continue
            entries.append(rec)
    return entries


def score(entries: list[dict], needs_tool_key: str, fn_key: str) -> dict:
    n = len(entries)
    if n == 0:
        return {"n": 0}
    would_reroute = [e for e in entries if not e[needs_tool_key]]
    correct = sum(
        1
        for e in entries
        if e[needs_tool_key] == (e.get("actual_stop_reason") == "tool_use")
    )
    fn = sum(1 for e in would_reroute if e.get(fn_key))
    return {
        "n": n,
        "coverage": len(would_reroute) / n * 100,
        "accuracy": correct / n * 100,
        "reroutable": len(would_reroute),
        "false_negative": fn,
        "false_negative_rate": (fn / len(would_reroute) * 100) if would_reroute else None,
    }


def main() -> None:
    entries = load_candidate_entries()
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    lines = [
        f"Mid-conversation candidate signal check — {date_str}",
        f"(candidate signal shadow-only since commit d5fdedd, deployed 2026-09-18)",
        "",
    ]

    if not entries:
        lines.append(
            "No mid-conversation shadow-validation entries with candidate fields "
            "found yet. Either the proxy hasn't seen mid-conversation traffic since "
            "deployment, or savings.jsonl is missing/rotated."
        )
    else:
        baseline_tool_use = sum(1 for e in entries if e.get("actual_stop_reason") == "tool_use")
        baseline_rate = baseline_tool_use / len(entries) * 100

        old = score(entries, "needs_tool", "false_negative")
        new = score(entries, "needs_tool_candidate", "false_negative_candidate")

        def pct_or_na(v):
            return "n/a" if v is None else f"{v:.1f}%"

        old_fn_str = pct_or_na(old["false_negative_rate"])
        new_fn_str = pct_or_na(new["false_negative_rate"])

        lines += [
            f"Sample size: {len(entries)} mid-conversation shadow entries since deployment",
            f"Base rate (actual tool_use): {baseline_rate:.1f}%  "
            f"(trivial 'always assume tool needed' baseline accuracy)",
            "",
            "                         old heuristic      candidate signal",
            f"  accuracy               {old['accuracy']:.1f}%              {new['accuracy']:.1f}%",
            f"  coverage (would reroute) {old['coverage']:.1f}%              {new['coverage']:.1f}%",
            f"  reroutable count        {old['reroutable']}                 {new['reroutable']}",
            f"  false-negative rate     {old_fn_str}              {new_fn_str}",
            "",
        ]

        if new["n"] < 100:
            lines.append(
                f"Only {new['n']} samples — likely still too few to promote. Let it "
                "collect more real traffic before deciding."
            )
        elif new["coverage"] < 5:
            lines.append(
                "Coverage is very low — the candidate rarely predicts 'no tool "
                "needed' even if accurate when it does. Promoting it wouldn't move "
                "the local-usage needle much."
            )
        elif new["false_negative_rate"] is not None and new["false_negative_rate"] > 2:
            lines.append(
                "False-negative rate is above the 2% flag threshold used on the "
                "dashboard — not yet safe to promote to live routing."
            )
        else:
            lines.append(
                "Candidate looks both accurate and has meaningful coverage — worth "
                "considering for promotion to mid_conv_routing_mode. Review with "
                "Claude Code before flipping any config."
            )

    report_text = "\n".join(lines)
    report_path = REPORT_DIR / f"candidate_signal_report_{date_str}.txt"
    report_path.write_text(report_text + "\n")

    if entries:
        headline = (
            f"candidate: {new['accuracy']:.0f}% acc / {new['coverage']:.0f}% coverage "
            f"({new['n']} samples) — see {report_path.name}"
        )
    else:
        headline = f"no candidate data yet — see {report_path.name}"

    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{headline}" with title "Ollama Proxy — candidate signal check"',
            ],
            check=False,
        )
    except FileNotFoundError:
        pass

    print(report_text)


if __name__ == "__main__":
    main()
