"""Turn run logs (runs/*.jsonl) into an instruction-tuning dataset (goal-sets).

Each NPC think was logged as a `prompt` event immediately followed by a `response`
event carrying the model's raw JSON (`raw`). We pair them, keep only the responses
that are a valid GOAL-SET (the decision), and emit chat triples:

    {"system": ..., "user": ..., "assistant": <canonical goal-set JSON>}

`recall` / `look` (thinking steps) and malformed replies are NOT training
targets here — we distill the DECISIONS. A goal-set containing `wait` IS a
decision (choosing to hold is behavior the student must learn too). The assistant text is
json.dumps(goals_to_obj(validated goals)) — exactly the object the game would run
(extras stripped, importance clamped), so the model learns the behavior the sim
accepts.

With --scores (a scores.jsonl from train/run_episodes.py), only decisions from
HIGH-RETURN trajectories are kept: NPCs are bucketed by drive profile and the
top --top quantile of each bucket survives — returns are normalized per profile,
so good survivalists AND good explorers both make the cut, not just whichever
profile scores higher numerically.

Usage:
    python train/extract_dataset.py                # all runs/*.jsonl -> train/data/sft.jsonl
    python train/extract_dataset.py --runs runs --out train/data/sft.jsonl --report
    python train/extract_dataset.py --runs runs/episodes --scores runs/episodes/scores.jsonl --top 0.5
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

# import the game's own response filter so targets match what the sim accepts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.brain import filter_response, goals_to_obj  # noqa: E402


def iter_events(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_keep_set(scores_path: str, top: float, max_keep: int | None = None) -> set:
    """The (run file, npc) pairs whose trajectories train: per drive-profile
    bucket, the NPCs in the top `top` quantile by return. `max_keep` caps each
    bucket (per bucket, never globally, so no profile's winners evict
    another's) — it bounds training-set size as a cross-round elite pool grows."""
    buckets = collections.defaultdict(list)
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                buckets[row["profile"]].append(row)
    keep = set()
    for prof, rows in sorted(buckets.items()):
        rows.sort(key=lambda r: r["return"], reverse=True)
        n_keep = max(1, math.ceil(len(rows) * top))
        if max_keep is not None:
            n_keep = min(n_keep, max_keep)
        kept = rows[:n_keep]
        keep.update((r["file"], r["npc"]) for r in kept)
        print(f"profile {prof}: kept {n_keep}/{len(rows)} "
              f"(return >= {kept[-1]['return']:g})")
    return keep


def extract(paths, keep=None):
    """Yield (system, user, assistant_json, meta) for each prompt->goal-set pair.

    Pairing mirrors the logger: within one actor's stream, a `prompt` is
    immediately followed by its `response`. We hold the last prompt per actor and
    consume it on the next response. Only goal-set responses become examples;
    recall/look/wait/bad replies are counted and skipped. With `keep` (a set of
    (run file basename, actor)), decisions from any other trajectory are skipped.
    """
    pending = {}  # actor -> prompt payload
    stats = collections.Counter()
    for path, ev in iter_events(paths):
        etype = ev.get("type")
        actor = ev.get("actor")
        if keep is not None and etype in ("prompt", "response") \
                and (os.path.basename(path), actor) not in keep:
            stats["filtered_low_return"] += 1
            continue
        if etype == "prompt":
            pending[actor] = ev.get("payload", {})
            stats["prompts"] += 1
        elif etype == "response":
            payload = ev.get("payload", {})
            prompt = pending.pop(actor, None)
            if "raw" not in payload:
                # error / streamed-error responses: not a decision pair
                if "error" in payload:
                    stats["response_errors"] += 1
                continue
            stats["responses_with_raw"] += 1
            if prompt is None:
                stats["orphan_responses"] += 1
                continue
            verdict = filter_response(payload["raw"])
            kind = verdict["kind"]
            if kind != "goals":
                stats[f"skipped_{kind}"] += 1     # recall / look / wait / bad — not a goal-set
                continue
            system = prompt.get("system", "")
            user = prompt.get("user", "")
            if not system or not user:
                stats["missing_prompt_text"] += 1
                continue
            obj = goals_to_obj(verdict["goals"])
            assistant = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            stats["goal_sets"] += 1
            actions = [a["action"] for g in obj["goals"] for a in g["actions"]]
            yield system, user, assistant, {"actions": actions, "n_goals": len(obj["goals"])}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="dir of *.jsonl run logs")
    ap.add_argument("--out", default="train/data/sft.jsonl")
    ap.add_argument("--report", action="store_true", help="print stats only, write nothing")
    ap.add_argument("--scores", help="scores.jsonl from run_episodes: keep only high-return trajectories")
    ap.add_argument("--top", type=float, default=0.5, help="per-profile quantile of NPCs to keep (with --scores)")
    ap.add_argument("--max-keep", type=int, default=None, help="cap NPCs kept per profile bucket (with --scores)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = args.runs if os.path.isabs(args.runs) else os.path.join(root, args.runs)
    paths = sorted(p for p in glob.glob(os.path.join(runs_dir, "*.jsonl"))
                   if os.path.basename(p) != "scores.jsonl")   # the score index, not a run log
    if not paths:
        print(f"no run logs found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    keep = None
    if args.scores:
        scores = args.scores if os.path.isabs(args.scores) else os.path.join(root, args.scores)
        keep = load_keep_set(scores, args.top, args.max_keep)

    seen = set()
    rows = []
    action_counts = collections.Counter()
    goal_counts = collections.Counter()
    dup = 0
    gen = extract(paths, keep=keep)
    stats = collections.Counter()
    try:
        while True:
            system, user, assistant, meta = next(gen)
            key = (system, user, assistant)
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            rows.append({"system": system, "user": user, "assistant": assistant})
            action_counts.update(meta["actions"])
            goal_counts[meta["n_goals"]] += 1
    except StopIteration as stop:
        stats = stop.value or collections.Counter()

    print(f"run files scanned  : {len(paths)}")
    print(f"prompt events      : {stats['prompts']}")
    print(f"responses w/ raw   : {stats['responses_with_raw']}")
    print(f"orphan responses   : {stats['orphan_responses']}")
    print(f"skipped (thinking) : recall={stats['skipped_recall']} look={stats['skipped_look']}")
    print(f"skipped (bad)      : {stats['skipped_bad']} (failed the response filter)")
    if keep is not None:
        print(f"filtered (return)  : {stats['filtered_low_return']} events from low-return trajectories")
    print(f"goal-set pairs     : {stats['goal_sets']}")
    print(f"exact duplicates   : {dup} (dropped)")
    print(f"UNIQUE examples    : {len(rows)}")
    print("goals per reply    :", dict(sorted(goal_counts.items())))
    print("by action          :", dict(action_counts.most_common()))

    if args.report:
        return
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} examples -> {out}")


if __name__ == "__main__":
    main()
