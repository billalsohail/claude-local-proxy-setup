#!/usr/bin/env bash
# Print routing stats from savings.jsonl

LOG="$HOME/.claude/ollama-proxy/savings.jsonl"

if [[ ! -f "$LOG" ]]; then
  echo "No log found at $LOG"
  exit 0
fi

python3 - "$LOG" <<'PYEOF'
import sys, json
from collections import Counter

log_path = sys.argv[1]
records = []
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

if not records:
    print("No records yet.")
    sys.exit(0)

# shadow-validation entries are outcome records tied to an earlier routing
# decision, not a routing decision themselves (no "reason" field) — keep them
# out of the routing-stats denominator or every percentage below is wrong.
shadow_records = [r for r in records if r.get("route") == "shadow-validation"]
records = [r for r in records if r.get("route") != "shadow-validation"]

total = len(records)
by_route = Counter(r["route"] for r in records)
by_reason = Counter(r["reason"] for r in records if r.get("reason"))
by_agent = Counter(r["agent_used"] for r in records if r.get("agent_used"))

orchestrated = by_route.get("orchestrate-single", 0) + by_route.get("orchestrate-project", 0)
local_total = by_route.get("ollama", 0) + orchestrated
anthropic_direct = by_route.get("anthropic", 0)

ollama_chars = sum(r.get("chars", 0) for r in records if r["route"] in ("ollama", "orchestrate-single", "orchestrate-project"))
anthropic_chars = sum(r.get("chars", 0) for r in records if r["route"] == "anthropic")

# Token savings estimate:
# Orchestrated: Ollama does full draft (saved), Claude does short refinement (~30% of full cost)
# Direct Ollama: 100% saved
est_tokens_saved = ollama_chars / 4
est_cost_saved = est_tokens_saved / 1_000_000 * 3.0  # sonnet-level pricing as baseline ($3/M input)

subtask_records = [r for r in records if r.get("subtasks", 0) > 0]
avg_subtasks = (
    sum(r["subtasks"] for r in subtask_records) / len(subtask_records)
    if subtask_records else 0
)

escalation_records = [
    r for r in records
    if r["route"] in ("orchestrate-single", "orchestrate-project") and "escalated" in r
]
escalated_count = sum(1 for r in escalation_records if r["escalated"])
escalation_rate = (
    100 * escalated_count / len(escalation_records) if escalation_records else None
)

# Shadow-mode intent-routing validation: of the requests the heuristic said
# didn't need a tool call, how often was the real call's stop_reason
# "tool_use" anyway? That's the false-negative rate the go/no-go gate for
# "live" mode depends on (target: <1-2%).
reroutable = [r for r in shadow_records if r.get("needs_tool") is False]
false_negatives = [r for r in reroutable if r.get("false_negative")]
false_negative_rate = 100 * len(false_negatives) / len(reroutable) if reroutable else None
reroutable_fraction = 100 * len(reroutable) / len(shadow_records) if shadow_records else None

W = 52
print(f"\n{'='*W}")
print(f"  Ollama Proxy — Routing Stats")
print(f"{'='*W}")
print(f"  Total requests      : {total}")
print(f"  → Anthropic direct  : {anthropic_direct}  ({100*anthropic_direct//total if total else 0}%)")
print(f"  → Orchestrate-single: {by_route['orchestrate-single']}  ({100*by_route['orchestrate-single']//total if total else 0}%)")
print(f"  → Orchestrate-project:{by_route['orchestrate-project']}  ({100*by_route['orchestrate-project']//total if total else 0}%)")
print(f"  → Ollama direct     : {by_route['ollama']}  ({100*by_route['ollama']//total if total else 0}%)")
print(f"  Local handling total: {local_total}  ({100*local_total//total if total else 0}%)")

if subtask_records:
    print(f"\n  Project decompositions : {len(subtask_records)}")
    print(f"  Avg subtasks/project   : {avg_subtasks:.1f}")

if escalation_rate is not None:
    print(f"\n  Orchestrator escalation rate: {escalation_rate:.1f}%  ({escalated_count}/{len(escalation_records)})")

if by_agent:
    print(f"\n  ECC agents used (top 10):")
    for agent, count in by_agent.most_common(10):
        bar = "█" * min(count, 20)
        print(f"    {agent:<30} {count:>4}  {bar}")

print(f"\n  Top routing reasons:")
for reason, count in by_reason.most_common(8):
    print(f"    {reason:<38} {count}")

print(f"\n  Chars routed locally   : {ollama_chars:,}")
print(f"  Chars sent to Anthropic: {anthropic_chars:,}")
print(f"\n  Est. tokens saved  : ~{int(est_tokens_saved):,}")
print(f"  Est. cost saved    : ~${est_cost_saved:.4f}  (sonnet baseline)")

if shadow_records:
    print(f"\n  {'-'*W}")
    print(f"  Intent-routing shadow validation")
    print(f"  {'-'*W}")
    print(f"  Shadow-scored requests   : {len(shadow_records)}")
    print(f"  Reroutable fraction      : {reroutable_fraction:.1f}%  ({len(reroutable)}/{len(shadow_records)})")
    if false_negative_rate is not None:
        flag = "  <-- above 1-2% go/no-go target" if false_negative_rate > 2 else ""
        print(f"  False-negative rate      : {false_negative_rate:.1f}%  ({len(false_negatives)}/{len(reroutable)}){flag}")
    else:
        print(f"  False-negative rate      : n/a (no reroutable samples yet)")

print(f"{'='*W}\n")
PYEOF
