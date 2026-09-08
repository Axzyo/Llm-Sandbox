"""Expert-iteration orchestrator: rollout -> score -> filter -> LoRA train ->
swap policy -> repeat.

Round 1 rolls out with gemma4 (Ollama) — that rollout's scores ARE the baseline
(spec task G). Every later round rolls out with the adapter trained the round
before (TransformersProvider), so each round's scores.jsonl is the eval of the
previous round's training. Each round persists under train/rounds/round_N/:
episodes + scores.jsonl (rollout), sft.jsonl (filtered data), adapter/
(checkpoint). Aggregates append to train/rounds/summary.jsonl — watch the
returns climb, or stop when they don't.

Usage:
    python train/expert_iter.py --rounds 3 --episodes 8 --npcs 3 --budget 240
"""
import argparse
import collections
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, os.path.join(ROOT, "train", script), *args]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def serve_ollama(adapter: str, merged: str, base: str, model_name: str) -> str:
    """Turn a trained LoRA adapter into a fast Ollama model: merge to fp16, then
    import + quantize. Ollama (llama.cpp) serves the student ~30x faster than
    bitsandbytes 4-bit inference, which is what makes many rounds affordable.
    Returns the Ollama model name for the next round's rollout."""
    run("merge_lora.py", "--adapter", adapter, "--out", merged, "--base", base)
    modelfile = os.path.join(os.path.dirname(merged), "Modelfile")
    with open(modelfile, "w", encoding="utf-8") as f:
        f.write(f"FROM {os.path.abspath(merged)}\n")
    cmd = ["ollama", "create", model_name, "--quantize", "q4_K_M", "-f", modelfile]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return model_name


def round_aggregates(scores_path: str) -> dict:
    with open(scores_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    per_profile = collections.defaultdict(list)
    for r in rows:
        per_profile[r["profile"]].append(r["return"])
    return {
        "npcs": len(rows),
        "mean_return": round(sum(r["return"] for r in rows) / len(rows), 3),
        "mean_survival_s": round(sum(r["survival_s"] for r in rows) / len(rows), 1),
        "mean_novelty": round(sum(r["novelty"] for r in rows) / len(rows), 2),
        "deaths": sum(1 for r in rows if r["cause"] != "alive"),
        "per_profile_return": {p: round(sum(v) / len(v), 3) for p, v in sorted(per_profile.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--npcs", type=int, default=3)
    ap.add_argument("--budget", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=float, default=0.5, help="per-profile quantile kept for training")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct", help="student base model")
    ap.add_argument("--start-round", type=int, default=1,
                    help="round number to begin at (resume a partial run without redoing earlier rounds)")
    ap.add_argument("--init-model", default=None,
                    help="Ollama model to roll out the FIRST round with (e.g. student-r1); "
                         "default gemma4 baseline when starting fresh")
    args = ap.parse_args()

    rounds_dir = os.path.join(ROOT, "train", "rounds")
    os.makedirs(rounds_dir, exist_ok=True)
    summary_path = os.path.join(rounds_dir, "summary.jsonl")

    prev_model = args.init_model            # Ollama model of the last round's student
    for r in range(args.start_round, args.start_round + args.rounds):
        round_dir = os.path.join(rounds_dir, f"round_{r}")
        ep_dir = os.path.join(round_dir, "episodes")
        scores = os.path.join(ep_dir, "scores.jsonl")
        sft = os.path.join(round_dir, "sft.jsonl")
        adapter = os.path.join(round_dir, "adapter")
        merged = os.path.join(round_dir, "merged")

        # 1. rollout with the current policy (round 1 = gemma4: the baseline).
        #    Every policy is served through Ollama, so rollouts stay fast.
        rollout = ["run_episodes.py", "--episodes", str(args.episodes), "--npcs", str(args.npcs),
                   "--budget", str(args.budget), "--seed", str(args.seed + r),
                   "--out-dir", ep_dir, "--tag", f"round{r}"]
        if prev_model is not None:
            rollout += ["--model", prev_model]     # ollama provider (default) serves the student
        run(*rollout)

        agg = round_aggregates(scores)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"round": r, "policy": prev_model or "gemma4 (baseline)", **agg}) + "\n")
        print(f"round {r} rollout: {agg}")

        # 2. filter to high-return trajectories, per drive profile
        run("extract_dataset.py", "--runs", ep_dir, "--scores", scores,
            "--out", sft, "--top", str(args.top))

        # 3. imitate the winners, then 4. serve the student back through Ollama
        run("train_lora.py", "--data", sft, "--out", adapter, "--base", args.base)
        prev_model = serve_ollama(adapter, merged, args.base, f"student-r{r}")

    print(f"\ndone: {args.rounds} rounds. Final policy served as Ollama model '{prev_model}'.\n"
          f"  python train/run_episodes.py --model {prev_model} --episodes {args.episodes} --npcs {args.npcs}\n"
          f"round-over-round numbers: {summary_path}")


if __name__ == "__main__":
    main()
