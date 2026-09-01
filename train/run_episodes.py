"""Headless episode runner: rollouts for reward-based expert iteration.

Runs N episodes of the real sim (sim/engine.py — the same Engine main.py drives,
thinking synchronously so a run is reproducible given a seed and provider). Each
episode spawns K NPCs with randomized drive profiles on a fresh resource layout;
every step each living NPC accrues the drive-weighted, world-computed reward
(sim/reward.py). Decisions are journaled exactly like a live run, so
train/extract_dataset.py can pair them; per-NPC returns land in a score index.

Outputs, under --out-dir:
    ep_<tag>_<n>.jsonl   one journal per episode (the usual run-log format)
    scores.jsonl         one row per NPC: drives, return, survival, novelty, cause

Usage:
    python train/run_episodes.py --episodes 5 --npcs 3 --budget 240 --seed 0
    python train/run_episodes.py --summarize runs/episodes/scores.jsonl
"""
import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.brain import Brain                                              # noqa: E402
from sim.config import load_config                                       # noqa: E402
from sim.engine import Engine, death_cause                               # noqa: E402
from sim.entities import Entity                                          # noqa: E402
from sim.journal import Journal                                          # noqa: E402
from sim.provider import OllamaProvider                                  # noqa: E402
from sim.reward import DISCOUNT_PER_S, note_novelty, reward, validate_drives  # noqa: E402
from sim.terrain import _free_floor_tiles, build_test_map, place_resources  # noqa: E402

# Drive-profile spread the ONE policy must generalize over. Weights are shares
# of one whole (they must sum to 1 — see sim/reward.py), so the sampler picks
# survival's share and curiosity gets the rest. A small grid keeps profiles
# bucketable for per-profile reward filtering. Even at survival=0 (the crazed
# scholar who values only discovery) death is not free: reward accrues only
# while alive, so staying alive remains instrumentally valuable — the agent
# must live to keep learning.
SURVIVAL_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sample_drives(rng: random.Random) -> dict:
    survival = rng.choice(SURVIVAL_WEIGHTS)
    return {"survival": survival, "curiosity": round(1.0 - survival, 2)}


def profile_key(drives: dict) -> str:
    """Canonical bucket label for a drive profile (weights rounded to 0.25)."""
    return ",".join(f"{d}={round(float(w) * 4) / 4:g}" for d, w in sorted(drives.items()))


def make_provider(args, cfg):
    # rollout temperature: --temperature overrides config. Selection-based training
    # mines behavioral variance for lucky successes, so rollouts often want to run
    # hotter than the near-deterministic game default.
    temp = cfg["temperature"] if args.temperature is None else args.temperature
    if args.provider == "transformers":
        from sim.provider import TransformersProvider
        return TransformersProvider(args.model or cfg["model"], adapter=args.adapter,
                                    temperature=temp, num_predict=cfg["num_predict"])
    if args.provider == "anthropic":
        from sim.provider import AnthropicProvider
        return AnthropicProvider(args.model or "claude-sonnet-5", num_predict=cfg["num_predict"])
    return OllamaProvider(cfg["ollama_url"], args.model or cfg["model"],
                          temp, cfg["num_predict"],
                          keep_alive=cfg.get("keep_alive", "30m"))


def run_episode(ep_id: str, path: str, provider, cfg: dict, rng: random.Random,
                n_npcs: int, budget: float, dt: float) -> list:
    """One headless episode. Returns per-NPC score rows."""
    world, _ = build_test_map()
    tiles = _free_floor_tiles(world, taken=set())
    rng.shuffle(tiles)

    journal = Journal(path, ep_id)
    npcs, brains = [], {}
    for i in range(1, n_npcs + 1):
        x, y = tiles.pop()
        npc = Entity(f"npc_{i}", f"npc_{i}", "npc", x, y)
        npc.properties["interact_range"] = int(cfg["interact_range"])
        npc.drives = sample_drives(rng)
        validate_drives(npc.drives)      # the fairness contract: weights are shares of 1
        npcs.append(npc)
        world.entities[npc.id] = npc
        brains[npc.id] = Brain(npc.id, provider, journal,
                               memory_k=cfg["memory_k"], memory_halflife_s=cfg["memory_halflife_s"])
    place_resources(world, rng)

    journal.log("system", "spawn", model=getattr(provider, "model", "?"),
                entities={e.id: list(e.pos) for e in world.entities.values()},
                drives={n.id: n.drives for n in npcs})

    engine = Engine(world, npcs, brains, journal)   # dispatch_think=None -> think inline
    drives = {n.id: dict(n.drives) for n in npcs}
    returns = collections.defaultdict(float)
    novelty = collections.defaultdict(int)
    seen: dict = {}                                  # per-NPC familiar-type watermark
    result = {}                                      # npc_id -> (survival_s, cause)

    while engine.npcs and engine.sim_t < budget:
        died = engine.step(dt)
        discount = DISCOUNT_PER_S ** engine.sim_t
        for npc in engine.npcs:
            novelty[npc.id] += note_novelty(npc, brains[npc.id], seen)
            returns[npc.id] += discount * reward(npc, world) * dt
        for npc in died:
            result[npc.id] = (engine.sim_t, death_cause(npc))
    for npc in engine.npcs:                          # survivors: episode ran out, not them
        result[npc.id] = (engine.sim_t, "alive")

    rows = []
    for nid in drives:
        survival_s, cause = result[nid]
        row = {"episode": ep_id, "file": os.path.basename(path), "npc": nid,
               "drives": drives[nid], "profile": profile_key(drives[nid]),
               "return": round(returns[nid], 3), "survival_s": round(survival_s, 1),
               "novelty": novelty[nid], "cause": cause}
        journal.log(nid, "episode_result", **{k: v for k, v in row.items() if k not in ("episode", "file")})
        rows.append(row)
    journal.log("system", "shutdown", sim_t=round(engine.sim_t, 1))
    journal.close()
    return rows


def summarize(rows: list) -> None:
    """Per-drive-profile aggregates: the yardstick every training round must beat."""
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[r["profile"]].append(r)
    print(f"\n{'profile':<32} {'n':>3} {'return':>8} {'surv_s':>7} {'novelty':>7} {'died':>5}")
    for prof in sorted(buckets):
        rs = buckets[prof]
        n = len(rs)
        print(f"{prof:<32} {n:>3} "
              f"{sum(r['return'] for r in rs) / n:>8.2f} "
              f"{sum(r['survival_s'] for r in rs) / n:>7.1f} "
              f"{sum(r['novelty'] for r in rs) / n:>7.2f} "
              f"{sum(1 for r in rs if r['cause'] != 'alive'):>5}")
    total = len(rows)
    print(f"{'ALL':<32} {total:>3} {sum(r['return'] for r in rows) / total:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--npcs", type=int, default=3)
    ap.add_argument("--budget", type=float, default=240.0, help="sim seconds per episode")
    ap.add_argument("--dt", type=float, default=0.2, help="sim seconds per step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/episodes")
    ap.add_argument("--provider", choices=["ollama", "transformers", "anthropic"], default="ollama")
    ap.add_argument("--model", default=None, help="override model (ollama name, HF path, or claude-* id)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (transformers provider)")
    ap.add_argument("--temperature", type=float, default=None, help="rollout sampling temperature (default: config.json)")
    ap.add_argument("--tag", default=None, help="filename tag; default = timestamp")
    ap.add_argument("--summarize", metavar="SCORES", help="re-print aggregates from a scores.jsonl and exit")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.summarize:
        with open(args.summarize, encoding="utf-8") as f:
            summarize([json.loads(line) for line in f if line.strip()])
        return

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config()
    provider = make_provider(args, cfg)
    if hasattr(provider, "warm"):
        provider.warm()
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    rng = random.Random(args.seed)

    scores_path = os.path.join(out_dir, "scores.jsonl")
    all_rows = []
    for n in range(1, args.episodes + 1):
        ep_id = f"ep_{tag}_{n:03d}"
        path = os.path.join(out_dir, f"{ep_id}.jsonl")
        t0 = time.monotonic()
        rows = run_episode(ep_id, path, provider, cfg, rng, args.npcs, args.budget, args.dt)
        all_rows.extend(rows)
        # append each episode's scores as it finishes: a rollout killed partway
        # (slow local model, resource contention) keeps every completed episode
        with open(scores_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{ep_id}: {len(rows)} NPCs, wall {time.monotonic() - t0:.0f}s -> "
              + ", ".join(f"{r['npc']} R={r['return']:g} ({r['cause']})" for r in rows), flush=True)

    print(f"\nappended {len(all_rows)} score rows -> {scores_path}")
    summarize(all_rows)


if __name__ == "__main__":
    main()
